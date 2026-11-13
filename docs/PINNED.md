# Pinned Upstream Dependency State (`docs/PINNED.md`)

Date: 2026-11-12 (UTC)

## 1. Pinned `llama.cpp` Source Repository
- **Path:** `third_party/llama.cpp`
- **Upstream Commit:** `3ba1ce0e28d991800e1585b7d3533f77213939d6` (cloned `--depth 1` from `https://github.com/ggml-org/llama.cpp.git`)
- **Compiler Toolchain:** MSVC `19.44.35222.0` (`Visual Studio 2022 Community`), CMake, Ninja, Windows SDK `10.0.26100.0`, C++20 Release (`/O2`).
- **GPU SDK Status on Machine:** No standalone CUDA Toolkit (`nvcc` / `cuda_runtime.h` absent) or `VulkanSDK` installed locally (`Test-Path` negative). CPU `ggml` / `llama.cpp` builds natively; GPU compilation requires installing CUDA Toolkit or Vulkan SDK.

## 2. Installed Python Runtime Package (Phase 0/1)
- **Package:** `llama-cpp-python` version `0.3.21` (`llama_supports_gpu_offload() == False`)
- **PyTorch Runtime:** `2.6.0+cu124` (active CUDA driver/runtime on `NVIDIA GeForce RTX 3060` 12 GB, used for Phase 0 hardware probe).
- **Local Test GGUF Model:** `E:\AI_Research\sophia\models\qwen2.5-1.5b-instruct-q4_k_m.gguf` (1,117,320,736 bytes, GGUF v3, 339 tensors, `general.architecture = "qwen2"`).
