#include <cuda_fp16.h>
#include "emb_kernel.cuh"
namespace kernel {
// FP32 embedding with float4 vectorization
__global__ void emb_kernel_cu_fp32(int32_t vocab_size, int32_t token_num, int32_t weight_dim,
                                   const int32_t* input_ptr, const float* weight_ptr,
                                   float* output_ptr) {
  int32_t token_idx = blockIdx.x;
  if (token_idx >= token_num) {
    return;
  }
  int32_t token = input_ptr[token_idx];
  if (token >= vocab_size) {
    return;
  }

  float* output_ptr_start = output_ptr + token_idx * weight_dim;
  const float* weight_ptr_start = weight_ptr + token * weight_dim;

  // float4 vectorized copy
  int32_t vec_dim = weight_dim / 4;
  const float4* src = reinterpret_cast<const float4*>(weight_ptr_start);
  float4* dst = reinterpret_cast<float4*>(output_ptr_start);
  for (int32_t i = threadIdx.x; i < vec_dim; i += blockDim.x) {
    dst[i] = src[i];
  }
  // Remainder
  for (int32_t i = vec_dim * 4 + threadIdx.x; i < weight_dim; i += blockDim.x) {
    output_ptr_start[i] = weight_ptr_start[i];
  }
}

// FP16 embedding: weights stored as half, output half
__global__ void emb_kernel_cu_fp16_impl(int32_t vocab_size, int32_t token_num, int32_t weight_dim,
                                        const int32_t* input_ptr, const half* weight_ptr,
                                        half* output_ptr) {
  int32_t token_idx = blockIdx.x;
  if (token_idx >= token_num) {
    return;
  }
  int32_t token = input_ptr[token_idx];
  if (token >= vocab_size) {
    return;
  }

  half* output_start = output_ptr + token_idx * weight_dim;
  const half* weight_start = weight_ptr + token * weight_dim;

  // float4 vectorized copy: 8 halfs per float4 (16 bytes)
  int32_t vec_dim = weight_dim / 8;
  const float4* src = reinterpret_cast<const float4*>(weight_start);
  float4* dst = reinterpret_cast<float4*>(output_start);
  for (int32_t i = threadIdx.x; i < vec_dim; i += blockDim.x) {
    dst[i] = src[i];
  }
  for (int32_t i = vec_dim * 8 + threadIdx.x; i < weight_dim; i += blockDim.x) {
    output_start[i] = weight_start[i];
  }
}

void emb_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                   const tensor::Tensor& output, int32_t vocab_size, void* stream) {
  tensor::Tensor input_cu;
  if (input.device_type() != base::DeviceType::kDeviceCUDA) {
    input_cu = input.clone();
    input_cu.to_cuda();
  }
  const int32_t input_num = static_cast<int32_t>(input.size());
  const int32_t weight_dim = weight.get_dim(1);
  CHECK(weight.device_type() == output.device_type());
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  constexpr int32_t max_seq_len = 512;
  constexpr int32_t thread_num = 128;
  int32_t* in_ptr = input_cu.ptr<int32_t>();
  float* wei_ptr = const_cast<float*>(weight.ptr<float>());
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
  if (stream_) {
    emb_kernel_cu_fp32<<<max_seq_len, thread_num, 0, stream_>>>(vocab_size, input_num, weight_dim,
                                                                in_ptr, wei_ptr, out_ptr);
  } else {
    emb_kernel_cu_fp32<<<max_seq_len, thread_num>>>(vocab_size, input_num, weight_dim, in_ptr,
                                                    wei_ptr, out_ptr);
  }
}

void emb_kernel_cu_fp16(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tensor::Tensor& output, int32_t vocab_size, void* stream) {
  tensor::Tensor input_cu;
  if (input.device_type() != base::DeviceType::kDeviceCUDA) {
    input_cu = input.clone();
    input_cu.to_cuda();
  }
  const int32_t input_num = static_cast<int32_t>(input.size());
  const int32_t weight_dim = weight.get_dim(1);

  constexpr int32_t max_seq_len = 512;
  constexpr int32_t thread_num = 128;
  int32_t* in_ptr = input_cu.ptr<int32_t>();
  const half* wei_ptr = reinterpret_cast<const half*>(weight.ptr<void>());
  half* out_ptr = reinterpret_cast<half*>(const_cast<void*>(output.ptr<void>()));
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
  if (stream_) {
    emb_kernel_cu_fp16_impl<<<max_seq_len, thread_num, 0, stream_>>>(vocab_size, input_num,
                                                                      weight_dim, in_ptr, wei_ptr,
                                                                      out_ptr);
  } else {
    emb_kernel_cu_fp16_impl<<<max_seq_len, thread_num>>>(vocab_size, input_num, weight_dim, in_ptr,
                                                         wei_ptr, out_ptr);
  }
}
}  // namespace kernel
