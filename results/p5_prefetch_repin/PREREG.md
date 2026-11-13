# Pre-Registration: `p5_prefetch_repin` (Phase 5 Prefetch, Cache Policy & Online Re-Pinning Ablations)

Date: 2026-11-12 (UTC)
Experiment ID: `p5_prefetch_repin`
Phase: Phase 5 (`GEMINI_IMPLEMENTATION_SPEC.md` Section 5)

## 1. Objective & Claims
Evaluate C++20 runtime streaming under `--ram-budget-mb 600 --cache-slots 12 --allow-slow-stream 1` across all 20 frozen prompts (`results/prompts_20.jsonl`, `n_predict=16`, `n_ctx=2048`, `n_batch=512`, `n_threads=6`, `temp=0.0`, `seed=42`) comparing 6 pre-registered configurations with a fixed `60 s` NVMe thermal cooldown between consecutive runs:
1. `lookahead_d1_wtinylfu`: `--cache-policy wtinylfu --prefetch lookahead --lookahead-depth 1 --repin 0`
2. `lookahead_d2_wtinylfu`: `--cache-policy wtinylfu --prefetch lookahead --lookahead-depth 2 --repin 0`
3. `lookahead_d1_repin`: `--cache-policy wtinylfu --prefetch lookahead --lookahead-depth 1 --repin 1`
4. `prefetch_off_wtinylfu`: `--cache-policy wtinylfu --prefetch off --repin 0`
5. `prefetch_shuffled_null`: `--cache-policy wtinylfu --prefetch shuffled --lookahead-depth 1 --repin 0` (null control)
6. `lookahead_d1_lru`: `--cache-policy lru --prefetch lookahead --lookahead-depth 1 --repin 0`

## 2. Falsifiers & Predictions
- **Exactness:** All 6 arms MUST match `lookahead_d1_wtinylfu` (`20/20` exact greedy tokens, `max_shared_top16_logit_diff = 0.0`) because prefetch/cache/repin never change arithmetic.
- **Look-ahead vs. Shuffled Null:** `lookahead_d1_wtinylfu` prefetch accuracy must exceed `0.90` and substantially outperform `prefetch_shuffled_null` in prefetch accuracy and wasted prefetch bytes.
- **Re-pinning on Dense Cyclic Access (Honest Scope):** With 12 cache slots across 97 dense FFN/Attn tensors accessed cyclically once per token, `--repin 1` pins up to `cap/3 = 4` objects (`4/97 = 4.1%` fewer SSD reads), while leaving 8 slots for the remaining 93 objects; Zipfian MoE re-pinning remains `UNVERIFIED` without a local MoE GGUF model.
- **Thermal Pacing Rationale:** Pre-registered `n_predict=16` (`320` steps, `~224 GB` SSD reads/arm) + `60s` inter-arm cooldown prevents multi-hour NVMe thermal runaway observed in `p4_planner_v2`.

## 3. Frozen Protocol
- **Execution command:** `python tools/run_experiments.py p5`
