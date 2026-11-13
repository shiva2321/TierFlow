# Phase 10 Pre-Registration: Sliced MoE Streaming & Causal EAM Prefetch

**Date:** 2026-11-13
**Protocol:** Strict post-60s NVMe heat-soak, zero OS PageCache bypass verification, exact greedy parity, causal shuffled-prefetch null control.

## 1. Models Under Test
- `OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf` (4,213,512,192 B, 16 layers, 64 experts, top-8) packed into `packed_models/OLMoE-1B-7B-0125-Instruct-Q4_K_M.moe.tfpack` (`3,219` objects, all expert tensors sliced along expert axis `ne[2]`).
- `Qwen3-30B-A3B-Q4_K_M.gguf` (18,556,685,824 B, 48 layers, 128 experts, top-8) packed into `packed_models/Qwen3-30B-A3B-Q4_K_M.moe.tfpack` (`18,867` objects, all expert tensors sliced along expert axis `ne[2]`).

## 2. Out-of-Core Budget & OS Measurement
- `OLMoE-1B-7B`: `--n-gpu-layers 99 --vram-mb 768 --ram-budget-mb 1152` (total governor budget `1,920 MiB = 2,013,265,920 B <= 47.8%` of `4.213 GB`).
- `Qwen3-30B-A3B`: `--n-gpu-layers 99 --vram-mb 2048 --ram-budget-mb 6144` (total governor budget `8,192 MiB = 8,589,934,592 B <= 46.3%` of `18.556 GB`).
- OS ground-truth verification reports `peak_system_used_delta_bytes`, `peak_process_working_set_bytes`, `peak_file_cache_delta_bytes`, `vram_governor_total_bytes`, and `ram_governor_total_bytes`.

## 3. Conditions (All Prompts: 20 Prompts x 32 Tokens, Greedy `temp=0`, `seed=42`, `n_ctx=1024`)
Right before timed MoE streaming conditions, run a 60-second continuous unbuffered NVMe heat-soak (`tf_io_bench.exe`) on `.moe.tfpack` to bring the NVMe controller into steady thermal state (eliminating SLC burst bias).
1. `stock_mmap_cpu_oracle`: `stock_mmap` (`n_gpu_layers=0`) as exact ground-truth parity reference and OS memory footprint baseline.
2. `tierflow_moe_lookahead`: `--prefetch lookahead --repin 1` (overlapped intra-layer sibling `Gate -> Up+Down` + causal $L \to L+1$ confidence-gated EAM prefetch).
3. `tierflow_moe_off`: `--prefetch off --repin 1` (no-prefetch ablation).
4. `tierflow_moe_shuffled`: `--prefetch shuffled --repin 1` (causal null control permuting predicted expert IDs at equal prefetch queue pressure).

## 4. Pre-Registered Acceptance Gates
- **G1 (No Quality Loss):** `20/20` exact generated token-ID match (`0` mismatches) against `stock_mmap` oracle on `OLMoE-1B-7B` and `Qwen3-30B-A3B`.
- **G2 (True Out-of-Core Memory):** `vram_governor + ram_governor <= 50%` of model file size, `vram_governor + peak_system_used_delta_bytes <= 50%` of model file size, and median `peak_file_cache_delta_bytes == 0`.
- **G3 (Prefetch Causality):** `tierflow_moe_lookahead` beats both `tierflow_moe_off` and `tierflow_moe_shuffled` on mean/median `decode_tok_s` and prefetch precision (`prefetch_hits / prefetch_submitted` and `eam_prefetch_hits / eam_prefetch_submitted`) post-60s NVMe heat-soak.
