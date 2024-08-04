#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cpu/bert/attention_common.h"

using onnxruntime::contrib::AttentionQkvFormat;
using onnxruntime::Stream;

namespace onnxruntime::cudnn_sdpa {
bool is_supported(const cudaDeviceProp& dprops,
                  int num_heads_q,
                  int num_heads_kv,
                  int head_size_qk,
                  int head_size_v,
                  int sequence_length_q,
                  int sequence_length_kv,
                  bool is_causal);

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
    bool is_bf16,                   // True if bfloat16, otherwise float16
    AttentionQkvFormat qkv_format,  // Q_K_V_BNSH, Q_K_V_BSNH, Q_K_V_BSNH_BNSH_BNSH are supported
    cudnnHandle_t handle,
    Stream* stream,
    AllocatorPtr allocator);

}  // namespace onnxruntime::cudnn_sdpa
