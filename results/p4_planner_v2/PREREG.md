# Pre-Registration: `p4_planner_v2` (Phase 4 Revised Probe & Analytical Planner Validation)

Date: 2026-11-12 (UTC)
Experiment ID: `p4_planner_v2`
Phase: Phase 4 Revision (follows `p4_planner` failure under 25% relative error threshold)

## 1. Diagnosis of `p4_planner` Failures (Preserved in `results/p4_planner/`)
In `p4_planner`, relative errors of `pred_tok_s` were:
- `budget_560mb`: `pred_tok_s = 3.449` vs `measured = 2.350 ± 0.752` (**46.75% error — FAILED**)
- `budget_850mb`: `pred_tok_s = 5.626` vs `measured = 2.986 ± 0.034` (**88.44% error — FAILED**)
- `budget_1400mb`: `pred_tok_s = 35.185` vs `measured = 44.240 ± 0.421` (**20.47% error**)

Root causes identified in the code:
1. **Per-token traffic inflation (`tf_llama_stream.cpp` + `tf_planner.cpp`):** `attn_*` projections had `freq = 1.5` used in `evaluate_placement_cost_sec` (charging 1.5× byte traffic for single-pass projections) and `token_embd.weight` (`131.3 MiB`) was charged as a full dense matrix scan (`freq = 1.0`) rather than a single-row lookup (`freq = 0.001`).
2. **Thread-spawn distortion in `tf_probe.cpp`:** `run_tf_probe()` started its timer before creating/joining 12 OS threads for a ~2.5 ms 128 MiB memory loop, inflating measured read time and depressing `cpu_gemv_effective_gbps`.
3. **Uncalibrated NVMe constant (`bw_ssd_sustained_gbps = 2.80`):** `tf_llama_stream.cpp` called `run_tf_probe()` without passing `manifest_.tfpack_path`, leaving `bw_ssd_sustained_gbps` fixed at QD=32 `2.80 GB/s` and ignoring low-QD (`QD = 1 + lookahead_depth`) post-pSLC NVMe sustained throughput (~`0.35×` of low-QD burst) plus per-object async I/O synchronization overhead (`0.20 ms`).

## 2. Changes Applied for `p4_planner_v2`
1. Separated `rank_priority` (for pinning order) from true per-token traffic fraction `freq` (`0.001` for `token_embd.weight`, `1.0` for dense matmuls/norms/head) in `PlannerObjectInput` and `tf_llama_stream.cpp`.
2. Updated `tf_probe.cpp` to use C++20 `std::barrier` synchronization across `cfg_.n_threads` threads for RAM read measurement (`cpu_gemv_effective_gbps = bw_ram_read_gbps * 0.90`) and live 16×8 MiB unbuffered `.tfpack` low-QD probe (`bw_ssd_burst_gbps`, `bw_ssd_sustained_gbps = max(0.25, burst * 0.35)`).
3. Added `ssd_obj_overhead_ms = 0.20 ms` per streamed `SSD_STREAM` object in `evaluate_placement_cost_sec` and `solve_exact_bruteforce`.

## 3. Pre-Registered Pass / Falsifier Criterion
- **Success criterion:** Relative prediction error `|pred_tok_s - measured_decode_tok_s| / measured_decode_tok_s <= 0.25` (25%) across **all three** budgets (`560 MiB`, `850 MiB`, `1400 MiB`) under the identical frozen 20-prompt protocol (`p4_planner`).
- Both `p4_planner` and `p4_planner_v2` artifacts are retained unmodified in `results/`.

## 4. Frozen Protocol
- **Execution command:** `python tools/run_experiments.py p4_v2`
- Same model, prompts, budgets (`560, 850, 1400 MiB`), and flags as `p4_planner`.

