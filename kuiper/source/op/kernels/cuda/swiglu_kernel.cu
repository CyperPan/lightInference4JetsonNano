#include <tensor/tensor.h>
#include <cuda_fp16.h>
#include "swiglu_kernel.cuh"
namespace kernel {
// FP32 SwiGLU - optimized: removed unnecessary shared memory, use registers directly
__global__ void swiglu_kernel_cu_fp32(int size, const float* in1, const float* in2, float* out) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size) {
    return;
  }
  float x = in1[idx];
  float gate = in2[idx];
  float sigmoid = 1.0f / (1.0f + __expf(-x));
  out[idx] = (x * sigmoid) * gate;
}

// FP16 SwiGLU with half2 vectorization
__global__ void swiglu_kernel_cu_fp16_impl(int size, const half* in1, const half* in2, half* out) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int vec_size = size / 2;
  if (idx < vec_size) {
    half2 x = reinterpret_cast<const half2*>(in1)[idx];
    half2 gate = reinterpret_cast<const half2*>(in2)[idx];
    // Compute SwiGLU in float for precision, output in half
    float x0 = __half2float(x.x);
    float x1 = __half2float(x.y);
    float g0 = __half2float(gate.x);
    float g1 = __half2float(gate.y);
    float s0 = x0 * (1.0f / (1.0f + __expf(-x0))) * g0;
    float s1 = x1 * (1.0f / (1.0f + __expf(-x1))) * g1;
    reinterpret_cast<half2*>(out)[idx] = __halves2half2(__float2half(s0), __float2half(s1));
    return;
  }
  int scalar_idx = vec_size * 2 + (idx - vec_size);
  if (scalar_idx < size) {
    float x = __half2float(in1[scalar_idx]);
    float gate = __half2float(in2[scalar_idx]);
    float sigmoid = 1.0f / (1.0f + __expf(-x));
    out[scalar_idx] = __float2half((x * sigmoid) * gate);
  }
}

void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                      const tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK(input1.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK_EQ(input2.is_empty(), false);
  CHECK(input2.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK_EQ(output.is_empty(), false);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  int size = static_cast<int32_t>(input1.size());
  int threads = 128;
  int blocks = (size + threads - 1) / threads;
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
  if (stream_) {
    swiglu_kernel_cu_fp32<<<blocks, threads, 0, stream_>>>(
        size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
  } else {
    swiglu_kernel_cu_fp32<<<blocks, threads>>>(
        size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
  }
}

void swiglu_kernel_cu_fp16(const tensor::Tensor& input1, const tensor::Tensor& input2,
                           const tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK_EQ(input2.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);

  int size = static_cast<int32_t>(input1.size());
  int threads = 128;
  int vec_size = (size + 1) / 2;
  int blocks = (vec_size + threads - 1) / threads;

  const half* in1_ptr = reinterpret_cast<const half*>(input1.ptr<void>());
  const half* in2_ptr = reinterpret_cast<const half*>(input2.ptr<void>());
  half* out_ptr = reinterpret_cast<half*>(const_cast<void*>(output.ptr<void>()));

  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
  if (stream_) {
    swiglu_kernel_cu_fp16_impl<<<blocks, threads, 0, stream_>>>(size, in1_ptr, in2_ptr, out_ptr);
  } else {
    swiglu_kernel_cu_fp16_impl<<<blocks, threads>>>(size, in1_ptr, in2_ptr, out_ptr);
  }
}
}  // namespace kernel
