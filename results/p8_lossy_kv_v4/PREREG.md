# Pre-Registration: `p8_lossy_kv_v4` (Phase 8 Opt-In Q8_K + Q4_0_V Lossy KV Compression Ladder)

Date: 2026-11-12 (UTC)
Experiment ID: `p8_lossy_kv_v4`
Phase: Phase 8 (`GEMINI_IMPLEMENTATION_SPEC.md` Section 8 — opt-in, off by default)

## 1. Objective & Claims
Evaluate the opt-in asymmetric 4-bit V KV cache rung (`--kv-type q8_q4`, i.e., `GGML_TYPE_Q8_0` keys + `GGML_TYPE_Q4_0` values with `flash_attn`) against `f16` exact and `q8_0` default on all 20 frozen prompts (`n_ctx=2048, n_predict=64, ram_budget=1400 MiB`), reporting KV reservation reduction, decode throughput (`decode_tok_s`), exact greedy token agreement, and first-step top-16 logit deltas cleanly (including any quality degradation!).

Note: MoE-only Phase 8 components (`HOBBIT` miss-expert precision, cache-aware router bias, `REAP`, `TEAL`) remain `UNVERIFIED` because no MoE GGUF model or CUDA/Vulkan toolkit is available on this machine.

## 2. Falsifiers
- If `--kv-type q8_q4` fails to initialize in `llama.cpp` FlashAttention or corrupts output catastrophically (`top1_first_token_agree < 15/20` or severe throughput regression vs `f16`), report the failure and exact failure mode honestly.

## 3. Frozen Protocol
- **Execution command:** `python tools/run_experiments.py p8`
- **Model:** `qwen2.5-1.5b-instruct-q4_k_m.gguf` + `.tfpack` manifest, `n_ctx=2048`, `n_batch=512`, `n_threads=6`, `n_predict=64`, `temp=0.0`, `seed=42`.

