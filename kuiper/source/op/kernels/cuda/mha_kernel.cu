#include <base/cuda_config.h>
#include <tensor/tensor.h>
#include <cfloat>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include "mha_kernel.cuh"
#include <base/tick.h>
namespace kernel {
constexpr static int thread_num = 256;

__device__ void softmax_gpu(float* __restrict__ x, int size) {
  int tid = threadIdx.x;
  int step = blockDim.x;

  float max_val = tid < size ? x[tid] : -FLT_MAX;
  for (int i = tid + step; i < size; i += step) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  using BlockReduce = cub::BlockReduce<float, thread_num>;
  __shared__ BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
  if (threadIdx.x == 0) {
    shared_val = max_val;
  }
  __syncthreads();
  max_val = shared_val;

  float sum = 0.0f;
  for (int i = tid; i < size; i += step) {
    x[i] = __expf(x[i] - max_val);
    sum += x[i];
  }
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;

  for (int i = tid; i < size; i += step) {
    x[i] /= sum;
  }
}


__global__ void multi_head_attention_kernel(int32_t pos, int32_t seq_len, float* query,
                                            float* score_ptr, float* output, float* key_cache,
                                            float* value_cache, int32_t kv_dim, int32_t kv_mul,
                                            int32_t head_num, int32_t head_size,
                                            int32_t layer_offset) {
  int head = blockIdx.x;
  if (head >= head_num) {
    return;
  }

  extern __shared__ float s_query_head[];
  float scale = 1.f / sqrtf(float(head_size));
  float* query_head = query + head * head_size;

  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query_head[i] = query_head[i];
  }
  __syncthreads();

  float* score_head = score_ptr + head * seq_len;
  int head_offset = (head / kv_mul) * head_size;
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    float* key_head = key_cache + layer_offset + t * kv_dim + head_offset;

    float score = 0.0f;
    for (int i = 0; i < head_size; i += 4) {
      float4 key_val = *reinterpret_cast<float4*>(key_head + i);
      float4 query_val = *reinterpret_cast<float4*>(s_query_head + i);

      score += key_val.x * query_val.x + key_val.y * query_val.y + key_val.z * query_val.z +
               key_val.w * query_val.w;
    }

    score *= scale;
    score_head[t] = score;
  }
  __syncthreads();

  softmax_gpu(score_head, pos + 1);
  __syncthreads();

  float* output_head = output + head * head_size;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float value = 0.0f;
    for (int t = 0; t <= pos; t++) {
      float* value_head = value_cache + layer_offset + t * kv_dim + head_offset;
      float score = score_head[t];
      value += score * value_head[i];
    }
    output_head[i] = value;
  }
}

// FP16 MHA: query/key/value in half, scores/accumulation in float for precision
__global__ void multi_head_attention_kernel_fp16(int32_t pos, int32_t seq_len, const half* query,
                                                 float* score_ptr, half* output,
                                                 const half* key_cache, const half* value_cache,
                                                 int32_t kv_dim, int32_t kv_mul,
                                                 int32_t head_num, int32_t head_size,
                                                 int32_t layer_offset) {
  int head = blockIdx.x;
  if (head >= head_num) {
    return;
  }

  extern __shared__ float s_query_head_f[];  // query in float for precision
  float scale = 1.f / sqrtf(float(head_size));
  const half* query_head = query + head * head_size;

  // Load query to shared memory, convert to float
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query_head_f[i] = __half2float(query_head[i]);
  }
  __syncthreads();

  float* score_head = score_ptr + head * seq_len;
  int head_offset = (head / kv_mul) * head_size;

  // Q-K dot product
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    const half* key_head = key_cache + layer_offset + t * kv_dim + head_offset;

    float score = 0.0f;
    // half2 vectorized dot product
    for (int i = 0; i < head_size; i += 2) {
      half2 key_h2 = *reinterpret_cast<const half2*>(key_head + i);
      float k0 = __half2float(key_h2.x);
      float k1 = __half2float(key_h2.y);
      score += s_query_head_f[i] * k0 + s_query_head_f[i + 1] * k1;
    }

    score *= scale;
    score_head[t] = score;
  }
  __syncthreads();

  softmax_gpu(score_head, pos + 1);
  __syncthreads();

  // Value aggregation
  half* output_head = output + head * head_size;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float value = 0.0f;
    for (int t = 0; t <= pos; t++) {
      const half* value_head = value_cache + layer_offset + t * kv_dim + head_offset;
      float score = score_head[t];
      value += score * __half2float(value_head[i]);
    }
    output_head[i] = __float2half(value);
  }
}

void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                   int32_t kv_dim, int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
                   const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                   const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                   base::DeviceType device_type, CudaConfig* config) {
  UNUSED(device_type);
  int32_t layer_offset = layer_index * seq_len * kv_dim;
  float* query = const_cast<float*>(query_tensor.ptr<float>());
  float* score = const_cast<float*>(score_tensor.ptr<float>());
  float* output = const_cast<float*>(mha_out.ptr<float>());

  float* key_cache = const_cast<float*>(key_cache_tensor.ptr<float>());
  float* value_cache = const_cast<float*>(value_cache_tensor.ptr<float>());

  cudaStream_t stream = config->stream;
  multi_head_attention_kernel<<<head_num, thread_num, head_size * sizeof(float), stream>>>(
      pos, seq_len, query, score, output, key_cache, value_cache, kv_dim, kv_mul, head_num,
      head_size, layer_offset);
}

void mha_kernel_cu_fp16(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                        int32_t kv_dim, int32_t kv_mul, int32_t head_size,
                        const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor,
                        const tensor::Tensor& score_tensor, const tensor::Tensor& key_cache_tensor,
                        const tensor::Tensor& value_cache_tensor, base::DeviceType device_type,
                        CudaConfig* config) {
  UNUSED(device_type);
  int32_t layer_offset = layer_index * seq_len * kv_dim;

  const half* query = reinterpret_cast<const half*>(query_tensor.ptr<void>());
  float* score = const_cast<float*>(score_tensor.ptr<float>());  // scores stay float
  half* output = reinterpret_cast<half*>(const_cast<void*>(mha_out.ptr<void>()));
  const half* key_cache = reinterpret_cast<const half*>(key_cache_tensor.ptr<void>());
  const half* value_cache = reinterpret_cast<const half*>(value_cache_tensor.ptr<void>());

  cudaStream_t stream = config->stream;
  // Shared memory for query in float
  multi_head_attention_kernel_fp16<<<head_num, thread_num, head_size * sizeof(float), stream>>>(
      pos, seq_len, query, score, output, key_cache, value_cache, kv_dim, kv_mul, head_num,
      head_size, layer_offset);
}

}  // namespace kernel
