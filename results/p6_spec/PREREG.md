# Pre-Registration: `p6_spec` (Phase 6 Union-Aware Speculative Decoding)

Date: 2026-11-12 (UTC)
Experiment ID: `p6_spec`
Phase: Phase 6 (`GEMINI_IMPLEMENTATION_SPEC.md` Section 6)

## 1. Objective & Claims
Evaluate `tf_spec_scheduler` + prompt-lookup speculative verification (`n_ctx=2048`, `n_batch=512`, `n_threads=6`, `prompts_20.jsonl`, split into `n=10 general` and `n=10 code_overlap`) across:
1. **Part A-RAM (`1400 MiB`, `n_predict=128`, `temp=0.0`):**
   - `ram1400_spec_off` (`--spec off`)
   - `ram1400_spec_adaptive` (`--spec adaptive`, union-aware `choose_k`)
   - `ram1400_spec_fixed4` (`--spec fixed4`)
2. **Part A-Stream (`600 MiB`, `cache_slots=12`, `prefetch=lookahead`, `n_predict=32`, `temp=0.0`, `60s` NVMe cooldown between streaming arms):**
   - `stream600_spec_off`
   - `stream600_spec_adaptive`
   - `stream600_spec_fixed4`
   - `stream600_spec_naive_nounion` (`--spec naive_no_union`, ablation ignoring batch tensor union reuse)
3. **Part B (`2,000`-Sample Two-Sample Distribution Test, `1400 MiB`, `temp=0.8`, `n_predict=4`, `code_01`):**
   - `spec_off` (`--seed 10000`) vs. `spec_adaptive` (`--seed 50000`) using exact Leviathan rejection sampling, tested via Pearson chi-square at $\alpha = 0.01$ on `first_token`, `second_token`, `bigram_12`, and `bigram_23` (with counts < 5 pooled into an `__OTHER__` tail bin).

## 2. Falsifiers & Predictions
- **Stochastic Distribution Fidelity (Primary Exactness Criterion):** Under `temp=0.8` over $N=2000$ draws, chi-square p-values across `first_token`, `second_token`, `bigram_12`, `bigram_23` must satisfy $p \ge 0.01$.
- **Greedy Exactness Check:** Check token-for-token exactness of `adaptive`, `fixed4`, and `naive_no_union` vs. `spec_off`; note any batched-GEMM vs `n_tokens=1` GEMV floating-point tie flips cleanly (cf. Claim C5 `13/20`).
- **Union-Aware vs NaiveNoUnion:** On `code_overlap`, `stream600_spec_adaptive` must select `k* > 0` (`spec_k_mean > 0`) and outperform `stream600_spec_naive_nounion` (which degenerates to `k* = 0`), while keeping `general` regression bounded via online `alpha_ema` adaptation.

## 3. Frozen Protocol
- **Execution command:** `python tools/run_experiments.py p6`

