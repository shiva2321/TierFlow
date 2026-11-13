# TierFlow Integration Architecture (`docs/INTEGRATION.md`)

Based on inspecting `third_party/llama.cpp` (`3ba1ce0e28d991800e1585b7d3533f77213939d6`):

## Chosen Integration Strategy: Option (c) + Option (a) Extension
Per `GEMINI_IMPLEMENTATION_SPEC.md` Section 3 (Phase 3), because upstream `llama.cpp` already implements **Option (a) (`slot_map` / `cpu_map` indirection via `ggml_get_rows` before `ggml_mul_mat_id`)** in `llama_moe_cache`, TierFlow **extends `llama_moe_cache`** rather than duplicating it:

1. **SSD $\to$ RAM Backed Slot Bank (`tf_io_win` / `tf_cache` / `.tfpack`):**
   - Replace/augment host `w->data` backing for unpinned/cold experts with a fixed host-RAM slot bank managed by `tf_governor` and `tf_cache` (`W-TinyLFU + SLRU + Pinned`), populated on misses and via $L \to L+1$ prefetch from `.tfpack` over `tf_io` (`FILE_FLAG_NO_BUFFERING` / `O_DIRECT`).
2. **RAM $\to$ VRAM Tier (`llama_moe_cache` + `tf_compute_place`):**
   - Keep `build_moe_cache_slots` / `slot_map` / `cpu_map` on GPU backends when CUDA/Vulkan SDK is built, driving `cpu_share` from `tf_compute_place`'s measured batch-size crossover (`B <= 2` CPU in-place, `B >= 4` PCIe+GPU on RTX 3060 / Ryzen 5 7600).
3. **Dense & CPU-Only Build Compatibility:**
   - Current development machine lacks local `nvcc` (CUDA Toolkit) and `VulkanSDK` headers; CPU AVX2/AVX-512 C++20 builds compile natively with MSVC `cl.exe` + Ninja.

## Implemented Upstream Patch (`third_party/llama.cpp`, `37 insertions(+), 6 deletions(-)`)
- `include/llama.h`, `src/llama.cpp`: added `tf_is_streamed_tensor` / `tf_is_streamed_ud` callbacks to `llama_model_params` so `llama_model_init_from_user` routes streamed weights through a 0-byte dummy host buffer without OS `mmap`.
- `src/llama-model-loader.{h,cpp}`, `src/llama-model.cpp`: bypassed backend weight allocation for streamed tensors and registered `tf_eval_cb` via `llama_context_params::cb_eval` so `tf_llama_stream` swaps live `tf_cache` slots into tensor views on `ask=true` and releases refcounts / polls async completions on `ask=false`.
- **Verified Live:** Phases 3–8 on dense `qwen2.5-1.5b-instruct-q4_k_m.gguf` (CPU AVX-512). **UNVERIFIED:** live GPU (`CUDA`/`Vulkan`) execution and live MoE GGUF execution (no local CUDA/Vulkan SDK or MoE GGUF weights available on any drive).


