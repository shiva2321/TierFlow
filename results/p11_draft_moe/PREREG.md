# Phase 11 Pre-Registration (`p11_draft_moe` — Universal Small Draft Model, `k <= 4` + Opt-In Tree Ablation)

**Registered:** 2026-11-13 (prior to running `p11` timed conditions)

## 1. Models & Budgets
- **Target Model:** `Qwen3-30B-A3B-Q4_K_M.gguf` (`packed_models/Qwen3-30B-A3B-Q4_K_M.moe.manifest.json`, 48 layers, 128 experts/layer, 8 active).
- **Draft Model:** `Qwen3-0.6B-Q4_K_M.gguf` (universal small draft model loaded via `LLAMA_LOAD_MODE_DIRECT_IO` inside the same TierFlow VRAM/RAM governor budget, `k <= 4` default cap).
- **Hybrid Hardware Budget:** `--n-gpu-layers 99 --vram-mb 2048 --ram-budget-mb 6144 --n-threads 6 --n-ctx 1024 --n-batch 512`.
- **Thermal Protocol:** 60-second unbuffered NVMe read-only heat-soak immediately before timed runs.

## 2. Primary Conditions (`prompts_20.jsonl`)
1. `qwen3_30b_cpu_off_ref`: `--mode tierflow --n-gpu-layers 0 --vram-mb 0 --ram-budget-mb 12288 --spec off --n-predict 16` (exact active-streaming CPU oracle across all 20 prompts).
2. `qwen3_30b_cpu_draft_k4`: `--mode tierflow --n-gpu-layers 0 --vram-mb 0 --ram-budget-mb 12288 --draft-model ... --spec fixed4 --spec-max-k 4 --n-predict 16` (exact CPU greedy parity check against `qwen3_30b_cpu_off_ref`).
3. `qwen3_30b_moe_off`: `--n-gpu-layers 99 --vram-mb 2048 --ram-budget-mb 6144 --spec off --n-predict 32` (`k = 0`, hybrid Vulkan + streamed MoE baseline).
4. `qwen3_30b_moe_draft_adaptive`: `--n-gpu-layers 99 --vram-mb 2048 --ram-budget-mb 6144 --draft-model ... --spec adaptive --spec-max-k 4 --n-predict 32` (union-aware controller `k <= 4`).
5. `qwen3_30b_moe_draft_fixed4`: `--n-gpu-layers 99 --vram-mb 2048 --ram-budget-mb 6144 --draft-model ... --spec fixed4 --spec-max-k 4 --n-predict 32` (fixed `k = 4` draft baseline).
6. `qwen3_30b_moe_spec_tree_k8` (**Opt-In Ablation**, excluded from exact parity gate): `--n-gpu-layers 99 --vram-mb 2048 --ram-budget-mb 6144 --draft-model ... --spec fixed4 --spec-max-k 8 --spec-tree 1 --n-predict 32` (tests whether `k > 4` increases expert-union SSD reads/token and causes greedy divergence).

## 3. Stochastic Equivalence Control (`temp = 0.8`, `500` draws x `4` tokens = `2,000` tokens per arm)
- Generate `500` i.i.d. 4-token trajectories (`--temp 0.8 --repeat 500 --n-predict 4`, distinct seed blocks `10000+` vs `50000+` on `prompts_20[0]`) under `qwen3_30b_moe_off` vs `qwen3_30b_moe_draft_fixed4` (exact Leviathan rejection + residual sampling).
- Compute empirical Pearson $\chi^2$ across `first_token`, `second_token` (under active verification), `third_token`, `bigram_12`, and `bigram_23`.

## 4. Pre-Registered Gates
- **G1 (Greedy Parity, CPU exact oracle):** `qwen3_30b_cpu_draft_k4` matches `qwen3_30b_cpu_off_ref` on `20/20` prompts (`0` exact mismatches, verifying draft KV rollback & verification math). Vulkan `n_tokens = 1` vs `n_tokens >= 2` exact match is reported transparently alongside.
- **G2 (Stochastic Distribution Parity):** `temp = 0.8` $\chi^2$ yields $p \ge 0.01$ across all marginal and bigram token tests between `spec off` and `spec fixed4`.
- **G3 (Speculative Efficiency & Ablation):** `qwen3_30b_moe_draft_adaptive` achieves positive acceptance (`spec_accept_rate > 0.30`), beats or matches `qwen3_30b_moe_off` throughput, and `k = 8` (`--spec-tree 1`) is evaluated honestly against `k <= 4` on `bytes_read_ssd_per_token`, acceptance, and exactness.
- **G4 (OS Memory & Cache Safety):** `peak_file_cache_delta_bytes` stays near zero (zero mmap cache bloat), and `ram_governor + vram_governor` stay within `--ram-budget-mb 6144` / `--vram-mb 2048` including the `Qwen3-0.6B` draft allocation.
