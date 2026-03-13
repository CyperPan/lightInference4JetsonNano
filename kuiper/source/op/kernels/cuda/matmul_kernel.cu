#include <tensor/tensor.h>
#include <cub/block/block_reduce.cuh>
#include <cuda_fp16.h>
#include "../kernels_interface.h"
#include "matmul_kernel.cuh"
namespace kernel {
template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32(const float* input, const float* weight, float* output, int M,
                                      int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }

  constexpr int pack_size = 4;
  const int pack_num = M / pack_size;
  const int pack_off = pack_size * pack_num;

#pragma unroll
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    int row_offset = p * M;
    float4* input_float4_ptr = (float4*)input;
    float4* weight_float4_ptr = (float4*)(weight + row_offset);

#pragma unroll
    for (int i = tid; i < pack_num; i += blockDim.x) {
      float4 input_float4 = *(input_float4_ptr + i);
      float4 weight_float4 = *(weight_float4_ptr + i);
      float part_sum = input_float4.x * weight_float4.x + input_float4.y * weight_float4.y +
                       input_float4.z * weight_float4.z + input_float4.w * weight_float4.w;
      sdata[tid] += part_sum;
    }

    for (int i = pack_off + tid; i < M; i += blockDim.x) {
      sdata[tid] += input[i] * weight[row_offset + i];
    }

    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

// FP16 MatMul kernel: half input/weight, float accumulation, half output
// Optimized for Jetson Orin Nano memory bandwidth (half2 vectorized loads)
template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp16_impl(const half* input, const half* weight, half* output,
                                           int M, int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }

  // half2 packing: process 2 half elements at a time (same bandwidth as float)
  // half4 (via float2 reinterpret): process 4 half elements at a time
  constexpr int pack_size = 8;  // 8 halfs = 4 half2 = 16 bytes per load
  const int pack_num = M / pack_size;
  const int pack_off = pack_size * pack_num;

#pragma unroll
  for (int p = start_row; p < end_row; ++p) {
    float local_sum = 0.0f;
    int row_offset = p * M;

    // Use float4 to load 8 half values at once (16 bytes)
    const float4* input_f4_ptr = reinterpret_cast<const float4*>(input);
    const float4* weight_f4_ptr = reinterpret_cast<const float4*>(weight + row_offset);

#pragma unroll
    for (int i = tid; i < pack_num; i += blockDim.x) {
      float4 in_f4 = input_f4_ptr[i];
      float4 wt_f4 = weight_f4_ptr[i];

      // Reinterpret as half2 pairs for fused multiply-add
      const half2* in_h2 = reinterpret_cast<const half2*>(&in_f4);
      const half2* wt_h2 = reinterpret_cast<const half2*>(&wt_f4);

      // 4 half2 pairs = 8 half elements, accumulate in float
      for (int j = 0; j < 4; ++j) {
        half2 prod = __hmul2(in_h2[j], wt_h2[j]);
        local_sum += __half2float(prod.x) + __half2float(prod.y);
      }
    }

    // Handle remainder elements
    for (int i = pack_off + tid; i < M; i += blockDim.x) {
      local_sum += __half2float(input[i]) * __half2float(weight[row_offset + i]);
    }

    sdata[tid] = local_sum;
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = __float2half(part_sum);
    }
    __syncthreads();
  }
}

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32int8(const float* input, const int8_t* weight,
                                          const float* scales, const int32_t group_size,
                                          float* output, int M, int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
      const int weight_idx = p * M + i;
      const int group_idx = weight_idx / group_size;
      sdata[tid] += input[i] * scales[group_idx] * static_cast<float>(weight[weight_idx]);
    }
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, const float scale, const CudaConfig* config) {
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col
  int packet_size = 4;
  // CHECK_EQ(M % packet_size, 0);

  CHECK_EQ(M, input.get_dim(0));
  if (config && config->stream) {
    matmul_kernel_cu_fp32<128, 1><<<K, 128, 0, config->stream>>>(
        input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M, K);
  } else {
    matmul_kernel_cu_fp32<128, 1><<<K, 128>>>(input.ptr<float>(), weight.ptr<float>(),
                                              const_cast<float*>(output.ptr<float>()), M, K);
  }
}

// FP16 MatMul dispatch: uses hand-written FP16 kernel with half2 vectorization
void matmul_kernel_cu_fp16(const tensor::Tensor& input, const tensor::Tensor& weight,
                           const tensor::Tensor& output, const float scale,
                           const CudaConfig* config) {
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);
  const int32_t M = weight.get_dim(1);
  CHECK_EQ(M, input.get_dim(0));

  const half* input_ptr = reinterpret_cast<const half*>(input.ptr<void>());
  const half* weight_ptr = reinterpret_cast<const half*>(weight.ptr<void>());
  half* output_ptr = reinterpret_cast<half*>(const_cast<void*>(output.ptr<void>()));

  cudaStream_t stream = (config && config->stream) ? config->stream : nullptr;
  matmul_kernel_cu_fp16_impl<128, 1><<<K, 128, 0, stream>>>(input_ptr, weight_ptr, output_ptr, M,
                                                             K);
}

// cuBLAS FP16 MatMul with Tensor Core acceleration (SM 8.7 Orin Nano)
void matmul_kernel_cublas_fp16(const tensor::Tensor& input, const tensor::Tensor& weight,
                               const tensor::Tensor& output, const float scale,
                               const CudaConfig* config) {
  CHECK(config != nullptr && config->cublas_handle != nullptr);
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(weight.is_empty() == false && weight.dims_size() == 2);

  const int32_t K = weight.get_dim(0);  // output dim (rows of weight)
  const int32_t M = weight.get_dim(1);  // input dim (cols of weight)
  CHECK_EQ(M, input.get_dim(0));

  const half* input_ptr = reinterpret_cast<const half*>(input.ptr<void>());
  const half* weight_ptr = reinterpret_cast<const half*>(weight.ptr<void>());
  half* output_ptr = reinterpret_cast<half*>(const_cast<void*>(output.ptr<void>()));

  // weight is [K, M] row-major. cuBLAS uses column-major.
  // output = weight * input is a GEMV: [K, M] * [M, 1] = [K, 1]
  // In cuBLAS column-major: treat weight as [M, K] transposed, so use CUBLAS_OP_T
  const half alpha = __float2half(1.0f);
  const half beta = __float2half(0.0f);

  cublasGemmEx(config->cublas_handle,
               CUBLAS_OP_T,    // transpose weight (row-major -> col-major)
               CUBLAS_OP_N,    // input as-is
               K, 1, M,        // output_rows, output_cols, inner_dim
               &alpha,
               weight_ptr, CUDA_R_16F, M,   // weight [M x K] in col-major (= [K x M] row-major)
               input_ptr, CUDA_R_16F, M,    // input [M x 1]
               &beta,
               output_ptr, CUDA_R_16F, K,   // output [K x 1]
               CUBLAS_COMPUTE_16F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, const CudaConfig* config) {
  CHECK(config != nullptr);
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col
  int packet_size = 4;
  CHECK_EQ(M % packet_size, 0);
  CHECK_EQ(M, input.get_dim(0));
  if (config->stream) {
    matmul_kernel_cu_fp32int8<128, 1><<<K, 128, 0, config->stream>>>(
        input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
        const_cast<float*>(output.ptr<float>()), M, K);
  } else {
    matmul_kernel_cu_fp32int8<128, 1><<<K, 128>>>(input.ptr<float>(), weight.ptr<int8_t>(),
                                                  scale.ptr<float>(), group_size,
                                                  const_cast<float*>(output.ptr<float>()), M, K);
  }
}
}  // namespace kernel
