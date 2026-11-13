# Pre-Registration: `p0_dense_baseline`

Date: 2026-11-12 (UTC)

## 1. Claims & Headline Metrics
- **Claim 1 (Packer Round-Trip Identity on Real GGUF):** Packing `E:\AI_Research\sophia\models\qwen2.5-1.5b-instruct-q4_k_m.gguf` (1,117,320,736 bytes, 339 tensors) into `.tfpack` produces 339 page-aligned objects (4096-byte boundaries) where `verified_byte_identical_objects == 339` (100% SHA-256 match against source GGUF) and unbuffered direct I/O single-object retrieval passes.
- **Claim 2 (Exact Speculative Decoding Equivalence & Throughput):** Evaluating `qwen2.5-1.5b-instruct-q4_k_m.gguf` via `llama-cpp-python` (v0.3.21, CPU backend, 6 threads) across $n = 20$ held-out prompts (64 generated tokens per prompt, greedy `temperature=0.0`, `seed=42`):
  - Exactness metric: token ID sequence match between standard autoregressive decoding and prompt-lookup n-gram speculative decoding (`LlamaPromptLookupDecoding`) across all 20 prompts (`exact_match_count / 20` with Wilson 95% CI).
  - Throughput metric: `decode_tok_s` (mean $\pm$ 95% CI across 20 prompts) for standard decoding vs. prompt-lookup speculative decoding on general prompts and code/structured prompts.
  - Memory accounting metric: true system-wide memory (`peak_proc_private_bytes`, `peak_proc_working_set_bytes`, `delta_system_cache_bytes`) under `use_mmap=True` vs. `use_mmap=False`.

## 2. Falsifiers
- **Falsifier 1:** Any of the 339 tensors in `qwen2.5-1.5b-instruct-q4_k_m.gguf` fails SHA-256 round-trip verification or 4096-byte page alignment.
- **Falsifier 2:** Greedy speculative decoding diverges from standard greedy decoding on any prompt (`exact_match_count < 20`), or we fail to report both wins and slowdowns across prompt categories honestly.

## 3. Frozen Protocol
- Model: `E:\AI_Research\sophia\models\qwen2.5-1.5b-instruct-q4_k_m.gguf`
- Threads: `n_threads = 6` (matching 6 physical cores on AMD Ryzen 5 7600), `n_ctx = 1024`, `max_tokens = 64`, `temperature = 0.0`, `seed = 42`.
- Test set: $n = 20$ disjoint prompts (10 general knowledge/reasoning prompts + 10 code/structured refactoring prompts where n-gram overlap with the prompt exists).

