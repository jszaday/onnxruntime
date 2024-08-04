#include "contrib_ops/cuda/bert/cudnn_fmha/cudnn_flash_attention.h"
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <iostream>

#if CUDART_VERSION < 12000 || CUDNN_MAJOR < 9
namespace onnxruntime::cudnn_sdpa {

bool is_supported(const cudaDeviceProp& /*dprops*/,
                  int /*num_heads_q*/,
                  int /*num_heads_kv*/,
                  int /*head_size_qk*/,
                  int /*head_size_v*/,
                  int /*sequence_length_q*/,
                  int /*sequence_length_kv*/,
                  bool /*is_causal*/) {

  return false;
}

void run(
    void* /*q*/,
    void* /*k*/,
    void* /*v*/,
    void* /*output*/,
    int /*batch_size*/,
    int /*num_heads_q*/,
    int /*num_heads_kv*/,
    int /*head_size_qk*/,
    int /*head_size_v*/,
    int /*sequence_length_q*/,
    int /*sequence_length_kv*/,
    float /*scale*/,
    bool /*is_causal*/,
    bool /*is_bf16*/,
    AttentionQkvFormat /*qkv_format*/,
    cudnnHandle_t /*handle*/,
    Stream* /*stream*/,
    AllocatorPtr /*allocator*/) {
  ORT_THROW("OnnxRuntime was not compiled with cuDNN Flash Attention.");
}

}  // namespace onnxruntime::cudnn_sdpa

#else  // CUDART_VERSION >= 12000 && CUDNN_MAJOR >= 9

#include <cudnn_frontend.h>
#include "core/providers/cuda/shared_inc/cudnn_fe_call.h"
#include "core/providers/cuda/cuda_stream_handle.h"

namespace onnxruntime::cudnn_sdpa {

namespace fe = cudnn_frontend;

bool is_supported(const cudaDeviceProp& dprops,
                  int num_heads_q,
                  int num_heads_kv,
                  int head_size_qk,
                  int head_size_v,
                  int sequence_length_q,
                  int sequence_length_kv,
                  bool is_causal) {
  bool is_sm8x = dprops.major == 8 && dprops.minor >= 0;
  bool is_sm90 = dprops.major == 9 && dprops.minor == 0;
  // See https://github.com/NVIDIA/cudnn-frontend/blob/1.0/release/docs/operations/Attention.md
  return (is_sm8x || is_sm90) &&
         (head_size_qk % 8 == 0) && (head_size_qk <= 256) &&
         (head_size_v % 8 == 0) && (head_size_v <= 256) &&
         (num_heads_q % num_heads_kv == 0) &&
         // Bottom right causal mask is only supported with s_q multiple of 64 and s_kv multiple of 64
         (!is_causal || (sequence_length_q % 64 == 0 && sequence_length_kv % 64 == 0));
}

// A helper function to set stride for q, k, v or output tensor.
inline void set_stride(std::vector<int64_t>& stride,
                       int64_t batch_size, int64_t num_heads, int64_t sequence_length, int64_t head_size,
                       bool is_bsnh) {
  stride = {num_heads * sequence_length * head_size,
            is_bsnh ? head_size : (head_size * sequence_length),
            is_bsnh ? (num_heads * head_size) : head_size,
            1};
}


// It is used as a key for hash table to store cached graphs.
// It contains all parameters used in builing graph. Do not include data pointers that only needed in graph execution.
struct GraphBuildParams {
  int batch_size;
  int num_heads_q;
  int num_heads_kv;
  int head_size_qk;
  int head_size_v;
  int sequence_length_q;
  int sequence_length_kv;
  float scale;
  bool is_causal;
  bool is_bf16;  // True if bfloat16, otherwise float16
  AttentionQkvFormat qkv_format;
  cudnnHandle_t handle;

  bool operator == (const GraphBuildParams &rhs) const {
    return batch_size == rhs.batch_size &&
           num_heads_q == rhs.num_heads_q &&
           num_heads_kv == rhs.num_heads_kv &&
           head_size_qk == rhs.head_size_qk &&
           head_size_v == rhs.head_size_v &&
           sequence_length_q == rhs.sequence_length_q &&
           sequence_length_kv == rhs.sequence_length_kv &&
            scale == rhs.scale &&
           is_causal == rhs.is_causal &&
           is_bf16 == rhs.is_bf16 &&
           qkv_format == rhs.qkv_format &&
           handle == rhs.handle;
  }
};


#define Q_UID 1
#define K_UID 2
#define V_UID 3
#define O_UID 4
//#define SCALE_UID 5
#define BIAS_UID 6
#define SEQ_LEN_Q_UID 7
#define SEQ_LEN_KV_UID 8

std::shared_ptr<fe::graph::Graph> build_graph(GraphBuildParams& params) {
  int batch_size = params.batch_size;
  int num_heads_q = params.num_heads_q;
  int num_heads_kv = params.num_heads_kv;
  int head_size_qk= params.head_size_qk;
  int head_size_v= params.head_size_v;
  int sequence_length_q= params.sequence_length_q;
  int sequence_length_kv= params.sequence_length_kv;
  float scale= params.scale;
  bool is_causal= params.is_causal;
  bool is_bf16= params.is_bf16;
  AttentionQkvFormat qkv_format = params.qkv_format;
  cudnnHandle_t handle = params.handle;

  assert(qkv_format == contrib::AttentionQkvFormat::Q_K_V_BSNH ||
         qkv_format == contrib::AttentionQkvFormat::Q_K_V_BSNH_BNSH_BNSH ||
         qkv_format == contrib::AttentionQkvFormat::Q_K_V_BSNH);

  auto mha_graph = std::make_shared<fe::graph::Graph>();
  mha_graph->set_io_data_type(is_bf16 ? fe::DataType_t::BFLOAT16 : fe::DataType_t::HALF)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);

  // cuDNN FrontEnd API requires setting dimensions (B, N, S, H) for Q, K, V and O tensors.
  // The stride is calculated for batch, head, sequence and hidden dim in actual data format which might not be BNSH.
  bool is_q_bsnh = (qkv_format == contrib::AttentionQkvFormat::Q_K_V_BSNH ||
                    qkv_format == contrib::AttentionQkvFormat::Q_K_V_BSNH_BNSH_BNSH);
  bool is_kv_bsnh = qkv_format == contrib::AttentionQkvFormat::Q_K_V_BSNH;

  std::vector<int64_t> stride;
  set_stride(stride, batch_size, num_heads_q, sequence_length_q, head_size_qk, is_q_bsnh);

  auto Q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("Q")
                                .set_uid(Q_UID)
                                .set_dim({batch_size, num_heads_q, sequence_length_q, head_size_qk})
                                .set_stride(stride));

  set_stride(stride, batch_size, num_heads_kv, sequence_length_kv, head_size_qk, is_kv_bsnh);
  auto K = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("K")
                                .set_uid(K_UID)
                                .set_dim({batch_size, num_heads_kv, sequence_length_kv, head_size_qk})
                                .set_stride(stride));

  set_stride(stride, batch_size, num_heads_kv, sequence_length_kv, head_size_v, is_kv_bsnh);
  auto V = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("V")
                                .set_uid(V_UID)
                                .set_dim({batch_size, num_heads_kv, sequence_length_kv, head_size_v})
                                .set_stride(stride));

  // auto attn_scale = mha_graph->tensor(fe::graph::Tensor_attributes()
  //                                        .set_name("attn_scale")
  //                                        .set_uid(SCALE_UID)
  //                                        .set_dim({1, 1, 1, 1})
  //                                        .set_stride({1, 1, 1, 1})
  //                                        .set_is_pass_by_value(true)
  //                                        .set_data_type(fe::DataType_t::FLOAT));

  // auto sdpa_options = fe::graph::Scaled_dot_product_flash_attention_attributes()
  auto attributes = fe::graph::SDPA_attributes()
                        .set_name("SDPA")
                        .set_is_inference(true)
                        .set_causal_mask(is_causal)
                        .set_causal_mask_bottom_right(is_causal)
                        .set_attn_scale(scale);
                      //.set_sliding_window_length(sliding_window_value);

  // auto bias = mha_graph.tensor(fe::graph::Tensor_attributes()
  //                                  .set_name("bias")
  //                                  .set_uid(BIAS_UID)
  //                                  .set_dim({b, 1, s_q, s_kv})
  //                                  .set_stride({s_q * s_kv, s_q * s_kv, s_kv, 1}));
  // attributes.set_bias(bias);

  // if (padding_mask) {
  //       auto seq_q  = graph->tensor(fe::graph::Tensor_attributes()
  //                                      .set_name("seq_q")
  //                                      .set_uid(SEQ_LEN_Q_UID)
  //                                      .set_dim({b, 1, 1, 1})
  //                                      .set_stride({1, 1, 1, 1})
  //                                      .set_data_type(fe::DataType_t::INT32));
  //       auto seq_kv = graph->tensor(fe::graph::Tensor_attributes()
  //                                       .set_name("seq_kv")
  //                                       .set_uid(SEQ_LEN_KV_UID)
  //                                       .set_dim({b, 1, 1, 1})
  //                                       .set_stride({1, 1, 1, 1})
  //                                       .set_data_type(fe::DataType_t::INT32));
  //       attributes.set_padding_mask(padding_mask).set_seq_len_q(seq_q).set_seq_len_kv(seq_kv);
  //   }

  auto [O, Stats] = mha_graph->sdpa(Q, K, V, attributes);

  constexpr bool is_output_bsnh = true;
  set_stride(stride, batch_size, num_heads_q, sequence_length_q, head_size_v, is_output_bsnh);

  O->set_output(true)
    .set_dim({batch_size, num_heads_q, sequence_length_q, head_size_v})
    .set_stride(stride)
    .set_uid(O_UID);

  std::cout << "cudnn graph:" << *mha_graph;
  /*
  graph:{"context":{
    "compute_data_type":"FLOAT","intermediate_data_type":"FLOAT","io_data_type":"HALF","name":""},
    "nodes":[
    {
      "alibi_mask":false,
      "attn_scale_value":"3E800000",
      "causal_mask":false,
      "causal_mask_bottom_right":false,
      "dropout_probability":null,
      "inputs":{"K":"K","Q":"Q","V":"V"},
      "is_inference":true,
      "name":"SDPA",
      "outputs":{"O":"SDPA::O"},
      "padding_mask":false,
      "sliding_window_length":null,
      "tag":"SDPA_FWD"
      }
    ],
    "tensors":{
      "Q":{"data_type":null,
          "dim":[1,2,2,16],
          "is_pass_by_value":false,"is_virtual":false,"name":"Q","pass_by_value":null,"reordering_type":"NONE",
          "stride":[64,16,32,1],
          "uid":1,"uid_assigned":true},
      "K":{"data_type":null,
          "dim":[1,2,3,16],
          "is_pass_by_value":false,"is_virtual":false,"name":"K","pass_by_value":null,"reordering_type":"NONE",
          "stride":[96,16,32,1],
          "uid":2,"uid_assigned":true},
      "V":{"data_type":null,
          "dim":[1,2,3,16],
          "is_pass_by_value":false,"is_virtual":false,"name":"V","pass_by_value":null,"reordering_type":"NONE",
          "stride":[96,16,32,1],
          "uid":3,"uid_assigned":true},
      "SDPA::O":{
        "data_type":null,
        "dim":[1,2,2,16],
        "is_pass_by_value":false,"is_virtual":false,"name":"SDPA::O","pass_by_value":null,"reordering_type":"NONE",
        "stride":[64,16,32,1],
        "uid":4,"uid_assigned":true}
    }}
  */
  // TODO:  CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failed cudnn_status: CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED
  CUDNN_FE_CALL_THROW(mha_graph->build(handle, {fe::HeurMode_t::A}));

  return mha_graph;
}


// Compute hash based on content in memory byte by byte. This can be moved to a common header file if needed.
template <typename T>
struct BytesHash {
  // Verify that Params is good to hash byte by byte.
  static_assert(std::is_standard_layout_v<T>, "Params is not POD");

  size_t operator()(const T& params) const {
    auto ptr = reinterpret_cast<const uint8_t*>(&params);
    // Fowler–Noll–Vo hash function
    uint32_t value = 0x811C9DC5;
    constexpr size_t bytes = sizeof(T);
    for (size_t i = 0; i < bytes; ++i) {
      value ^= ptr[i];
      value *= 0x01000193;
    }
    return (size_t)value;
  }
};

template <typename KeyType, typename T>
struct MHAGraphCache {
  std::unordered_map<KeyType, T, BytesHash<KeyType> > cache_;

  // no mutexes here as caches are now thread local
  T* find(const KeyType& key) {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      return nullptr;
    }
    return &(it->second);
  }

  void update(const KeyType& key, T& results) {
    cache_.erase(key);
    cache_.emplace(key, std::move(results));
  }
};

// Use thread local caches because cuDNN execution plans are not guaranteed to be thread safe.
thread_local MHAGraphCache<GraphBuildParams, std::shared_ptr<fe::graph::Graph>> mha_graph_cache;

void run(
    void* q,
    void* k,
    void* v,
    void* output,
    int batch_size,
    int num_heads_q,
    int num_heads_kv,
    int head_size_qk,
    int head_size_v,
    int sequence_length_q,
    int sequence_length_kv,
    float scale,
    bool is_causal,
    bool is_bf16,  // True if bfloat16, otherwise float16
    AttentionQkvFormat qkv_format,
    cudnnHandle_t handle,
    Stream* stream,
    AllocatorPtr allocator) {

  GraphBuildParams params;
  params.batch_size = batch_size;
  params.num_heads_q = num_heads_q;
  params.num_heads_kv = num_heads_kv;
  params.head_size_qk = head_size_qk;
  params.head_size_v = head_size_v;
  params.sequence_length_q = sequence_length_q;
  params.sequence_length_kv = sequence_length_kv;
  params.scale = scale;
  params.is_causal = is_causal;
  params.is_bf16 = is_bf16;
  params.qkv_format = qkv_format;
  params.handle = handle;

  std::shared_ptr<fe::graph::Graph>* graph_ptr = mha_graph_cache.find(params);
  std::shared_ptr<fe::graph::Graph> mha_graph;
  if (graph_ptr) {
    mha_graph = *graph_ptr;
  } else {
    mha_graph = build_graph(params);
    mha_graph_cache.update(params, mha_graph);
  }

  std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
      {Q_UID, q},
      {K_UID, k},
      {V_UID, v},
      {O_UID, output},
      //{bias, bTensor.devPtr},
      //{SCALE_UID, &scale}
      };

  // Allocate workspace.
  auto bytes = mha_graph->get_workspace_size();

  IAllocatorUniquePtr<void> buffer = IAllocator::MakeUniquePtr<void>(
    allocator, bytes, false, stream, WaitCudaNotificationOnDevice);

  CUDNN_FE_CALL_THROW(mha_graph->execute(handle, variant_pack, buffer.get()));
}

}  // namespace onnxruntime::cudnn_sdpa
#endif
