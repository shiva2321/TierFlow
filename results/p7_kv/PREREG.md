# Pre-Registration: `p7_kv` (Phase 7 Q8_0 KV Compression, 32K Context, & Disk Prefix Cache)

Date: 2026-11-12 (UTC)
Experiment ID: `p7_kv`
Phase: Phase 7 (`GEMINI_IMPLEMENTATION_SPEC.md` Section 7)

## 1. Objective & Claims
1. **Q8_0 KV parity vs F16 baseline (20 frozen prompts, `n_ctx=2048`):** Measure run-to-run `f16_a` vs `f16_b` consistency alongside `q8_0` vs `f16_a` exact token matches, first-step top-16 logit diff, RAM reservation (`kv_reserve`), and decode speed (`decode_tok_s`).
2. **Disk prefix cache exactness & TTFT speedup:** Compare cold prefill+save against warm disk-restored prefix state on both the 20 frozen prompts (`n_ctx=2048`) and a ~2,500-token long prompt (`n_ctx=32768`), requiring **exact 20/20** (`max_logit_diff == 0.0`) identity and reduced TTFT on warm restore.
3. **32K context within tight budget (`1700 MiB`):** Run `--n-ctx 32768 --kv-type q8_0 --ram-budget-mb 1700` end-to-end without SSD spilling (`0 B` weight streaming, full RAM speed) vs `--kv-type f16` (`952 MiB` KV, requiring `--ram-budget-mb 2200` to avoid spilling). Note scope: runtime 32K evaluation tests native `GGML_TYPE_Q8_0` KV compression; explicit paged SSD-KV eviction/reload under contention is covered in `tf_kv_store` C++ unit tests (`tf_tests.exe`).

## 2. Falsifiers
- Prefix cache fails if warm restore diverges from cold prefill (`< 20/20` or `max_shared_top16_logit_diff > 1e-4` or `warm_hit_count != 20`).
- `q8_0` KV fails if decode speed collapses (`< 20 tok/s`) or `32K` (`n_ctx=32768`) fails governor check or spills weights to SSD under `1700 MiB`.

## 3. Frozen Protocol
- **Execution command:** `python tools/run_experiments.py p7`
- **Model:** `qwen2.5-1.5b-instruct-q4_k_m.gguf` + `.tfpack` manifest, `n_threads=6`, `n_batch=512`, `temp=0.0`, `seed=42`.

