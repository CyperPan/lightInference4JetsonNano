# LightInference — Lightweight LLM Inference Engine for Edge Devices

A high-performance, from-scratch LLM inference framework written in **C++17 / CUDA**, optimized for **NVIDIA Jetson** edge devices. Supports LLaMA 2/3, Qwen 2.5/3 model families with INT8 quantization and CUDA acceleration.

> Tested on **NVIDIA Jetson Orin Nano (8GB)** with aarch64 architecture.

---

## Highlights

- **Pure C++/CUDA implementation** — all Transformer operators hand-written, no dependency on PyTorch/TensorRT
- **Custom CUDA kernels** — MatMul, Multi-Head Attention, RoPE, RMSNorm, SwiGLU, Embedding, Softmax
- **INT8 group quantization** — reduce memory footprint for edge deployment while preserving accuracy
- **KV-Cache** — avoid redundant computation during autoregressive decoding
- **CUDA memory pool** — pre-allocated GPU memory management to eliminate runtime `cudaMalloc` overhead
- **Memory-mapped model loading** — efficient `mmap`-based weight loading for large models
- **Device-agnostic design** — CPU (Armadillo + OpenBLAS) and CUDA backends with unified operator interface
- **Multi-model support** — LLaMA 2, LLaMA 3/3.2, Qwen 2.5, Qwen 3
- **HTTP serving** — FastAPI-based inference server with streaming (SSE) support
- **Docker support** — containerized build environment

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    HTTP API (FastAPI)                    │
├─────────────────────────────────────────────────────────┤
│                   Model Layer                           │
│         LLaMA2Model / LLaMA3Model / Qwen2/3Model       │
├─────────────────────────────────────────────────────────┤
│                  Operator Layer                         │
│  Embedding │ RMSNorm │ RoPE │ MHA │ SwiGLU │ MatMul    │
├──────────────────────┬──────────────────────────────────┤
│   CPU Kernels        │       CUDA Kernels               │
│  (Armadillo/BLAS)    │  (Float4 vectorization, CUB,     │
│                      │   shared memory, block reduce)   │
├──────────────────────┴──────────────────────────────────┤
│                 Memory Management                       │
│   DeviceAllocator │ CUDA MemPool │ mmap │ Buffer/Tensor │
└─────────────────────────────────────────────────────────┘
```

### Inference Pipeline

```
Input Text
  → Tokenizer (SentencePiece / BPE-Tiktoken)
  → Embedding Lookup
  → N × Transformer Block:
      → RMSNorm → Q/K/V Projection → RoPE → Multi-Head Attention (+ KV-Cache) → Residual
      → RMSNorm → FFN (SwiGLU: W1, W2, W3) → Residual
  → Final RMSNorm → Linear → Sampling
  → Output Token
```

---

## Project Structure

```
├── kuiper/
│   ├── include/                # Header files
│   │   ├── base/               # Buffer, Allocator, DeviceType
│   │   ├── model/              # Model definitions (LLaMA, Qwen)
│   │   ├── op/                 # Layer interfaces, Encoder
│   │   ├── sampler/            # Sampling strategies
│   │   └── tensor/             # Tensor data structure
│   └── source/                 # Implementations
│       ├── base/               # CPU/CUDA allocators, memory pool
│       ├── model/              # Model forward pass logic
│       ├── op/
│       │   └── kernels/
│       │       ├── cpu/        # CPU kernels (Armadillo + OpenBLAS)
│       │       └── cuda/       # CUDA kernels (hand-written)
│       ├── sampler/
│       └── tensor/
├── demo/                       # Inference entry points
├── server/                     # FastAPI HTTP server
├── test/                       # Unit tests (GTest)
├── tools/                      # Model export scripts (HuggingFace → binary)
├── dockerfile                  # Containerized build
└── CMakeLists.txt
```

---

## Key Technical Details

### CUDA Kernel Optimizations

| Kernel | Key Techniques |
|--------|---------------|
| **MatMul** | Float4 vectorized loads, CUB block reduction, per-row parallelism |
| **MatMul (INT8)** | On-the-fly group dequantization, fused scale multiplication |
| **Multi-Head Attention** | Query preload to shared memory, causal masking, GQA support |
| **RMSNorm** | Float4 vectorization, block reduction, batched variant |
| **RoPE** | Per-head-pair parallelism, separate LLaMA3 / Qwen2 implementations |
| **Softmax** | Numerically stable (max subtraction), parallel reduction |

### Memory Management

- **CUDA Memory Pool**: pre-allocates GPU buffers and tracks availability via hash maps, eliminating per-inference `cudaMalloc`/`cudaFree` overhead
- **mmap Model Loading**: maps model files directly into virtual address space, enabling lazy page-fault-driven loading without copying entire weights into RAM
- **Zero-Copy Weights**: model weight buffers wrap `mmap` pointers directly — no redundant allocation

### INT8 Quantization

- Group-based quantization with configurable `group_size`
- Per-group scale factors stored alongside INT8 weights
- CUDA kernel performs fused dequantization during MatMul: `output += input × scale × weight_int8`

### Grouped Query Attention (GQA)

- Supports models where `num_q_heads > num_kv_heads`
- Multiple query heads share the same KV head via `kv_mul = head_num / kv_head_num`
- Reduces KV-Cache memory by up to 8× compared to standard MHA

---

## Supported Models

| Model | Sizes | Tokenizer | Notes |
|-------|-------|-----------|-------|
| LLaMA 2 | 7B–70B | SentencePiece | Original architecture |
| LLaMA 3 / 3.2 | 1B, 8B, 70B | BPE (Tiktoken) | GQA, updated RoPE |
| Qwen 2.5 | 0.5B–72B | BPE (Tiktoken) | GQA, different RoPE freq |
| Qwen 3 | Various | BPE (Tiktoken) | Chain-of-thought mode |

---

## Build & Run

### Prerequisites

- NVIDIA GPU with CUDA support (tested on Jetson Orin Nano, CUDA 12.x)
- CMake ≥ 3.16
- C++17 compiler with CUDA support

### Dependencies

| Library | Purpose |
|---------|---------|
| [Google glog](https://github.com/google/glog) | Logging |
| [Google Test](https://github.com/google/googletest) | Unit testing |
| [SentencePiece](https://github.com/google/sentencepiece) | LLaMA 2 tokenizer |
| [Armadillo](https://arma.sourceforge.net/) + OpenBLAS | CPU matrix operations |
| CUDA Toolkit | GPU acceleration |

### Compile

```bash
mkdir build && cd build

# Auto-download dependencies via CPM
cmake -DUSE_CPM=ON -DQWEN2_SUPPORT=ON ..
make -j$(nproc)

# Other model options:
#   -DLLAMA3_SUPPORT=ON    for LLaMA 3/3.2
#   -DQWEN3_SUPPORT=ON     for Qwen 3
```

### Export Model Weights

```bash
# Example: Qwen2.5-0.5B
python3 tools/export_qwen2.py Qwen2.5-0.5B.bin --hf=Qwen/Qwen2.5-0.5B

# Example: LLaMA 3.2-1B
python3 tools/export.py Llama-3.2-1B.bin --hf=meta-llama/Llama-3.2-1B
```

### Run Inference

```bash
# Qwen 2.5
./build/demo/qwen_infer Qwen2.5-0.5B.bin Qwen/Qwen2.5-0.5B/tokenizer.json

# LLaMA 3.2
./build/demo/llama_infer Llama-3.2-1B.bin meta-llama/Llama-3.2-1B/tokenizer.json
```

### HTTP Server

```bash
pip install -r requirements.txt
export MODEL_NAME=Qwen/Qwen2.5-0.5B
python -m server.app

# API endpoints:
#   GET  /health              — health check
#   POST /generate            — text generation
#   POST /generate_stream     — streaming generation (SSE)
```

---

## Testing

```bash
cd build
ctest --output-on-failure
```

Unit tests cover tensor operations, CUDA kernel correctness (validated against CPU reference), buffer management, and model-level inference.

---

## Third-Party Dependencies

- [Google glog](https://github.com/google/glog) — Logging framework
- [Google Test](https://github.com/google/googletest) — Testing framework
- [SentencePiece](https://github.com/google/sentencepiece) — Tokenizer (LLaMA 2)
- [Armadillo](https://arma.sourceforge.net/) + OpenBLAS — CPU linear algebra
- [abseil-cpp](https://github.com/abseil/abseil-cpp) — Utilities (LLaMA 3 / Qwen)
- [re2](https://github.com/google/re2) — Regex (BPE tokenizer)
- [nlohmann/json](https://github.com/nlohmann/json) — JSON parsing

## License

This project is for educational and research purposes.
