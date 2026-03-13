#include <cuda_fp16.h>
#include "add_kernel.cuh"

namespace kernel {
// Original FP32 kernel - now with float4 vectorization for better bandwidth
__global__ void add_kernel_cu_fp32(int32_t size, const float* in1, const float* in2, float* out) {
  int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  // float4 vectorized path: process 4 elements per thread
  int32_t vec_tid = tid;
  int32_t vec_size = size / 4;
  if (vec_tid < vec_size) {
    float4 a = reinterpret_cast<const float4*>(in1)[vec_tid];
    float4 b = reinterpret_cast<const float4*>(in2)[vec_tid];
    reinterpret_cast<float4*>(out)[vec_tid] =
        make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    return;
  }
  // Handle remainder
  int32_t idx = vec_size * 4 + (tid - vec_size);
  if (idx < size) {
    out[idx] = in1[idx] + in2[idx];
  }
}

// FP16 kernel with half2 vectorized add
__global__ void add_kernel_cu_fp16_impl(int32_t size, const half* in1, const half* in2, half* out) {
  int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  // half2 vectorized: process 2 half elements per thread
  int32_t vec_size = size / 2;
  if (tid < vec_size) {
    half2 a = reinterpret_cast<const half2*>(in1)[tid];
    half2 b = reinterpret_cast<const half2*>(in2)[tid];
    reinterpret_cast<half2*>(out)[tid] = __hadd2(a, b);
    return;
  }
  // Handle remainder
  int32_t idx = vec_size * 2 + (tid - vec_size);
  if (idx < size) {
    out[idx] = __hadd(in1[idx], in2[idx]);
  }
}

void add_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                   const tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK_EQ(input2.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);
  int32_t size = static_cast<int32_t>(input1.size());
  CHECK_EQ(size, input2.size());
  CHECK_EQ(size, output.size());
  int32_t thread_num = 512;
  int32_t block_num = (size + thread_num - 1) / thread_num;
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
  if (stream_) {
    add_kernel_cu_fp32<<<block_num, thread_num, 0, stream_>>>(
        size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
  } else {
    add_kernel_cu_fp32<<<block_num, thread_num>>>(size, input1.ptr<float>(), input2.ptr<float>(),
                                                  const_cast<float*>(output.ptr<float>()));
  }
}

void add_kernel_cu_fp16(const tensor::Tensor& input1, const tensor::Tensor& input2,
                        const tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK_EQ(input2.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);
  int32_t size = static_cast<int32_t>(input1.size());
  int32_t thread_num = 512;
  // half2 vectorized: each thread handles 2 elements
  int32_t vec_size = (size + 1) / 2;
  int32_t block_num = (vec_size + thread_num - 1) / thread_num;

  const half* in1_ptr = reinterpret_cast<const half*>(input1.ptr<void>());
  const half* in2_ptr = reinterpret_cast<const half*>(input2.ptr<void>());
  half* out_ptr = reinterpret_cast<half*>(const_cast<void*>(output.ptr<void>()));

  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
  if (stream_) {
    add_kernel_cu_fp16_impl<<<block_num, thread_num, 0, stream_>>>(size, in1_ptr, in2_ptr,
                                                                    out_ptr);
  } else {
    add_kernel_cu_fp16_impl<<<block_num, thread_num>>>(size, in1_ptr, in2_ptr, out_ptr);
  }
}
}  // namespace kernel
