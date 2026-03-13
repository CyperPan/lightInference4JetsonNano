#ifndef BLAS_HELPER_H
#define BLAS_HELPER_H
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
namespace kernel {
struct CudaConfig {
  cudaStream_t stream = nullptr;
  cublasHandle_t cublas_handle = nullptr;
  bool use_fp16 = false;
  ~CudaConfig() {
    if (stream) {
      cudaStreamDestroy(stream);
    }
    if (cublas_handle) {
      cublasDestroy(cublas_handle);
    }
  }
};
}  // namespace kernel
#endif  // BLAS_HELPER_H
