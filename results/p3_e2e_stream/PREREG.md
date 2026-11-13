# Pre-Registration: `p3_e2e_stream` (Phase 3 End-to-End Streaming & Exactness)

Date: 2026-11-12 (UTC)
Experiment ID: `p3_e2e_stream`
Phase: Phase 3 (`GEMINI_IMPLEMENTATION_SPEC.md` Section 3)

## 1. Self-Consistency Calibration (Measured in `p3_selfcheck` Prior to Freeze)
- Stock `llama.cpp` (`--mode stock_mmap`, `--no-repack 0`): `20/20` exact token match, `max_shared_top16_logit_diff = 0.0` across two independent runs.
- Stock `llama.cpp` (`--mode stock_mmap`, `--no-repack 1`): `20/20` exact token match, `max_shared_top16_logit_diff = 0.0` across two independent runs.
- Difference solely from `CPU_REPACK` (`Q4_K_8x8` SIMD reduction order) vs native `Q4_K`: `8/20` exact token match, `max_logit_diff = 0.6507`, `mean_logit_diff = 0.3800`, wall time `34.07s` vs `34.09s`.

## 2. Claims & Headline Metrics
1. **Exactness under SSD streaming (`tierflow_stream600`):** At an artificial budget of `600 MiB` (which pins embeddings, output head, and norms in RAM and forces `717 MiB` of FFN+Attention tensors to stream on demand from `.tfpack` via `tf_io_win` into a `12`-slot governor-bounded buffer bank), greedy decoding on all 20 frozen prompts (`results/prompts_20.jsonl`) matches kernel-matched stock `llama.cpp` (`stock_mmap_norepack`) and full-RAM `tierflow_ram1400` **20/20** in token IDs with `max_shared_top16_logit_diff <= 1e-4`.
2. **System-wide memory & zero OS cache bloat:** System-wide delta RAM (`peak_system_used_delta_bytes`, Section 6.3) for `tierflow_stream600` stays below **`600 MiB + 10%`** (`~660 MiB` due to runtime stacks/heaps outside weight/slot/KV arenas), strictly lower than `stock_mmap` (`~1.6 GB`), with `peak_file_cache_delta_bytes` remaining near zero (`< 5 MiB`).
3. **Throughput vs baselines:** Report `decode_tok_s` and `ttft_ms` (mean ± 95% CI, min, max, $n=20$) of `tierflow_ram1400` and `tierflow_stream600` against `stock_mmap_repack`, `stock_mmap_norepack`, and `stock_dio_norepack`. Any streaming slowdown vs RAM-resident baselines is reported directly.

## 3. Falsifiers
- Exactness fails if `tierflow_stream600` vs `stock_mmap_norepack` or `tierflow_ram1400` is `< 20/20` exact token matches or `max_shared_top16_logit_diff > 1e-4`.
- Memory fails if `tierflow_stream600` `peak_system_used_delta_bytes > 660 MiB` (10% over 600 MiB) or OS file cache inflates with streamed tensor bytes.

## 4. Frozen Protocol
- **Model:** `E:\AI_Research\sophia\models\qwen2.5-1.5b-instruct-q4_k_m.gguf` + `.tfpack` (`e:\RAM_optimizer\packed_models\qwen2.5-1.5b-instruct-q4_k_m.manifest.json`, 339 SHA-256 verified objects).
- **Prompts:** `results/prompts_20.jsonl` ($n=20$: 10 general + 10 code), `n_predict=64`, `n_ctx=2048`, `n_batch=512`, `n_threads=6`, `temp=0.0`, `seed=42`. Note: `n_predict=64` matches `p0_dense_baseline` so 5 full sweeps complete cleanly within one session while providing 1,280 streamed token steps per streaming sweep.
- **Conditions (run sequentially via `python tools/run_experiments.py p3`):**
  1. `stock_mmap_repack` (`--mode stock_mmap --no-repack 0`)
  2. `stock_mmap_norepack` (`--mode stock_mmap --no-repack 1`)
  3. `stock_dio_norepack` (`--mode stock_dio --no-repack 1`)
  4. `tierflow_ram1400` (`--mode tierflow --ram-budget-mb 1400`)
  5. `tierflow_stream600` (`--mode tierflow --ram-budget-mb 600 --allow-slow-stream 1 --cache-slots 12 --prefetch lookahead --lookahead-depth 1`)
- **I/O cross-check:** Compare `tf_io` `bytes_read_ssd_total` against Windows PDH `\PhysicalDisk(*)\Disk Read Bytes/sec` accumulated bytes (`pdh_disk_read_bytes_total`).

