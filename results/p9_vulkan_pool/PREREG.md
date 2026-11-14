# Pre-Registration: Phase 9 — Vulkan Hybrid Tiering + 3-Size-Class Arena Pool (`p9_vulkan_pool`)

- **Model:** `E:\AI_Research\sophia\models\qwen2.5-1.5b-instruct-q4_k_m.gguf` (`1.11 GB` packed `.tfpack`)
- **Prompts:** `results/prompts_20.jsonl` (20 frozen prompts, `n_ctx=2048`, `n_batch=512`, `n_threads=6`, `n_predict=64`, greedy `temp=0.0`)
- **GPU Device:** `NVIDIA GeForce RTX 3060` (`ggml-vulkan`, 512 MiB desktop safety reserve unless constrained by `--vram-mb`)
- **Conditions:**
  1. `stock_mmap_cpu`: `--mode stock_mmap --n-gpu-layers 0`
  2. `stock_mmap_vulkan`: `--mode stock_mmap --n-gpu-layers 99`
  3. `stock_mmap_hybrid9`: `--mode stock_mmap --n-gpu-layers 9` (matched 9-layer GPU offload reference for `tierflow_hybrid_512_512`)
  4. `tierflow_vulkan_full`: `--mode tierflow --vram-mb 2048 --n-gpu-layers 99`
  5. `tierflow_hybrid_512_512`: `--mode tierflow --vram-mb 512 --ram-budget-mb 512 --n-gpu-layers 99 --allow-slow-stream 1` (`cparams.op_offload = false` when CPU-streamed dense weights are present)
- **Acceptance Checks (reported verbatim, zero cherry-picking):**
  - `tierflow_vulkan_full` exact token parity vs `stock_mmap_vulkan` (`20/20` target).
  - `tierflow_hybrid_512_512` executes split VRAM (`<= 512 MiB` governor) + 3-size-class CPU arena streaming + SSD streaming without hangs/crashes, reporting parity vs `stock_mmap_hybrid9`, `stock_mmap_vulkan`, and `stock_mmap_cpu` plus OS `peak_system_used_delta_bytes`.
  - Compare 3-size-class arena allocation (`slot_bank_bytes`) vs v1 uniform-stride allocation for the same `n_cache_slots` (`n_cache_slots * max_stride_bytes`).

