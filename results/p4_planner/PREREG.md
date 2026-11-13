# Pre-Registration: `p4_planner` (Phase 4 Probe & Analytical Planner Validation)

Date: 2026-11-12 (UTC)
Experiment ID: `p4_planner`
Phase: Phase 4 (`GEMINI_IMPLEMENTATION_SPEC.md` Section 4)

## 1. Objective & Claims
Evaluate the current `tf_probe` (`runtime/tf_probe.cpp`) + `tf_planner` (`runtime/tf_planner.cpp`) analytical throughput predictions (`pred_tok_s` sustained and `pred_tok_s_burst`) against measured `decode_tok_s` (mean ± 95% CI, $n=20$ frozen prompts) across three memory budgets that span all operating regimes of `qwen2.5-1.5b-instruct-q4_k_m.gguf`:
1. `560 MiB` (`budget_560mb`): heavy SSD streaming regime.
2. `850 MiB` (`budget_850mb`): mixed RAM-resident + SSD streaming regime.
3. `1400 MiB` (`budget_1400mb`): 100% RAM-resident regime (`0 B` SSD stream).

## 2. Pre-Registered Pass / Falsifier Criterion
- **Falsifier:** Relative prediction error `|pred_tok_s - measured_decode_tok_s| / measured_decode_tok_s > 0.25` (25%) on any of the three tested budgets triggers a planner failure requiring diagnosis and a revised pre-registered model (`p4_planner_v2`), preserving both `p4_planner` and `p4_planner_v2`.
- Specifically record both `pred_tok_s` (sustained model) and `pred_tok_s_burst` alongside measured burst (`gen_01`) vs steady-state (`prompts[2..19]`) and overall mean across all 20 prompts.

## 3. Frozen Protocol
- **Model:** `E:\AI_Research\sophia\models\qwen2.5-1.5b-instruct-q4_k_m.gguf` + `e:\RAM_optimizer\packed_models\qwen2.5-1.5b-instruct-q4_k_m.manifest.json`.
- **Prompts:** `e:\RAM_optimizer\results\prompts_20.jsonl` ($n=20$), `n_ctx=2048`, `n_batch=512`, `n_threads=6`, `n_predict=64`, `temp=0.0`, `seed=42`, `--cache-slots 12`, `--prefetch lookahead --lookahead-depth 1`, `--allow-slow-stream 1`.
- **Execution command:** `python tools/run_experiments.py p4`.

