# Running large models on small memory: research report and build plan

Date: 2026-11-12. Scope: inference (not training) of large language models on machines with little RAM and VRAM.

## 1. The answer in short

- **"Almost no memory, full speed, full quality" is not reachable for dense models.** A dense model reads every weight for every token, so speed is capped by how fast the slowest place holding the weights can be read (section 2). No software removes that cap.
- **It is reachable, to a useful degree, for Mixture-of-Experts (MoE) models.** They touch only a few percent of their weights per token, so the rest can stay on SSD. Most large open models released in 2025–2026 are MoE, so this is where the opportunity is.
- **No single trick does it.** The practical gain comes from stacking five independent levers: fewer bits per weight, touching fewer weights per token, keeping only the hot part in fast memory, shrinking the context cache, and getting more tokens out of each pass over the weights.
- **What I propose building:** a tiered-residency runtime (working name **TierFlow**) that places every piece of a model in VRAM, RAM or SSD according to how often it is used, streams the rest on demand, and never changes the model's output in its default mode. Sections 5–8 give the design, algorithms and test plan for Gemini to implement.

I could not read the shared Gemini conversation: the link shows only a sign-in page in my browser, so this report does not build on it.

## 2. The limits that no program can remove

**Speed limit.** For one user generating one token at a time:

```
tokens/second  ≤  read speed of the tier holding the weights  ÷  bytes of weights touched per token
```

Worked example, a dense 70B model at 4 bits per weight (35 GB touched per token):

| Where the weights sit | Typical read speed | Ceiling |
|---|---|---|
| NVMe SSD | ~6 GB/s | ~0.17 tok/s |
| Dual-channel DDR5 RAM | ~60 GB/s | ~1.7 tok/s |
| RTX 3090 VRAM | ~750 GB/s measured | ~21 tok/s |

The SSD row matches what AirLLM users report for 70B on a Gen4 drive (0.175 tok/s). The read speeds for SSD and RAM are my typical-hardware figures; the VRAM figure is from the Strata project's measurement.

The same formula for an MoE model: Qwen3-30B-A3B touches about 31 MB per layer instead of 345 MB, an 11× drop, which is why a prototype reaches 4+ tok/s from SSD in about 2 GB of RAM.

**Quality limit.** Weights hold information, and there is a floor on bits per weight:

| Bits per weight | Effect on an already-trained model |
|---|---|
| 8 | Indistinguishable from the original |
| ~4–5 | Small loss; one test shows code-benchmark scores dropping from 56.1% to 51.8% |
| ~3.5 | Where a 2026 on-device study places the quality threshold |
| 2 | Best methods (QTIP, PV-Tuning) cost roughly +17–19% perplexity on Llama-2-7B, and reasoning tasks suffer more than perplexity shows |
| 1.58 | Only works if the model is trained that way from scratch (BitNet); proven at 2B parameters, not at frontier scale |

So about 4 bits per weight is the practical floor for "no quality compromise", which is a 4× saving against 16-bit weights and nothing more.

## 3. What exists, by lever

### Lever A — fewer bits per weight

| Technique | What it does | Status |
|---|---|---|
| Pruning (Optimal Brain Damage 1989, SparseGPT, Wanda) | Deletes weights | Unstructured sparsity rarely speeds up real hardware; largely superseded by quantization |
| Distillation | Trains a smaller model to copy a larger one | Works, but produces a different, smaller model |
| GPTQ, AWQ, GGUF K-quants with importance matrix | Rounds weights to 3–8 bits using calibration data | Mainstream; GGUF is the de facto format |
| QuIP#, AQLM, PV-Tuning, VPTQ, QTIP, Leech-lattice VQ | Vector and trellis coding for 2–3 bits | Best quality per bit; QTIP ships in ExLlamaV3 (EXL3). AQLM-family decoding is slow |
| MXFP4 / NVFP4 / FP8 | 4- and 8-bit float formats with hardware support on recent NVIDIA cards | Increasingly how vendors publish checkpoints |
| Mixed and nested precision (Any-Precision LLM, MatQuant, MatGPTQ, DP-LLM, FlexQuant) | Different bits for different layers; one file that can be read at several precisions | Research; per-layer allocation under a budget is the useful part |
| Low-rank plus quantized residual (LQER, EoRA, SVD-LLM) | Small correction matrices repair quantization error | Research; modest gains |
| Lossless coding (ZipNN, DFloat11, Huff-LLM) | Entropy-codes the bits exactly | DFloat11 gives ~30% on 16-bit weights with bit-identical output, decoded on the GPU. Gain on already-quantized weights is much smaller |
| BitNet b1.58 | Ternary weights, trained from scratch | 2B model at ~0.4 GB (excluding embeddings); needs its own runtime |

### Lever B — touch fewer weights per token

| Technique | What it does | Status |
|---|---|---|
| MoE routing | The model itself picks a few experts per token | The big one. Built into the model, exact, no quality cost |
| Activation sparsity (Deja Vu, Apple "LLM in a Flash", PowerInfer) | Predicts which neurons will be zero and skips them | Works well on ReLU models (Apple: 2.4% of feed-forward neurons loaded per token on OPT-6.7B). Modern models use SiLU/GELU and are not naturally sparse; ProSparse retrains them to ~89% sparsity, which means a modified model |
| Early exit / layer skipping (LayerSkip) | Stops before the last layer on easy tokens | Needs special training; changes output unless used only for drafting |

### Lever C — keep only the hot part in fast memory

| Technique | What it does | Status |
|---|---|---|
| mmap and OS paging (llama.cpp default) | The OS decides what stays in RAM | Works by accident; the OS cannot predict which expert comes next |
| Layer-by-layer loading (AirLLM, Accelerate disk offload) | One layer in memory at a time | Tiny memory, but hits the SSD ceiling: 0.03–2 tok/s |
| GPU/CPU layer split, MoE experts on CPU (llama.cpp, Fiddler) | Hot layers on GPU, rest in RAM | Mainstream |
| Expert caches (ExpertFlow, MoEpic, llama.cpp PR #27861) | VRAM as a cache of experts held in RAM | Predictive caches beat plain LRU |
| Expert streaming from SSD (FlashMoE, OD-MoE, llama.cpp PR #25294, TinyGiant, Strata, Micro-Expert-Router) | Experts stay on disk and are read when the router asks | Active in 2026; at least five independent implementations. PR #25294 was still unmerged at last report |

Useful measured details from this group: repacking each expert contiguously on disk cuts page faults 36×; calibrating which experts to pin from real text gives a 42% hit rate against 6% for a random profile; overlapping the next layer's read with the current layer's compute roughly doubles throughput.

### Lever D — shrink the context (KV) cache

A 7B model at 128K tokens needs about 64 GB of cache, more than the weights.

| Technique | What it does | Status |
|---|---|---|
| 8-bit cache (llama.cpp `--cache-type-k/v q8_0`) | Halves the cache | Community tests find no measurable loss |
| 4-bit cache | Quarters it | Model-dependent; one test shows severe damage on Qwen2.5-Coder. Keys are more sensitive than values |
| KIVI, TurboQuant | 2–3 bit cache with smarter coding | KIVI: 2.6× lower peak memory on Llama-2-7B. TurboQuant (Google, ICLR 2026) claims 6× with no loss; authors' figures |
| Token eviction (H2O, SnapKV, Ada-KV) | Drops unimportant tokens | Lossy; a critical study found research gains often do not carry to production |
| Architecture changes (grouped-query, latent and sliding-window attention) | Smaller cache by design | Built into newer models |

### Lever E — more tokens per pass over the weights

| Technique | What it does | Status |
|---|---|---|
| Speculative decoding | A small model drafts several tokens; the large model checks them in one pass | Exact output. Published self-speculative results are 1.8–2.6× |
| Batching | Many requests share one pass | Free throughput for servers; no help for a single user |
| Prefill in one pass | The whole prompt is processed in a single sweep | Standard |

Lever E matters more than usual here: when weights are streamed, one pass is expensive, so every extra token accepted per pass is a direct multiplier.

## 4. Gaps in what exists

1. **No one runtime combines the levers.** Expert streaming, cache tiers, KV compression and speculative decoding live in separate projects with separate flags.
2. **Placement is manual.** Users hand-tune how many layers go to the GPU. Nothing solves placement as an optimisation over measured device speeds.
3. **File layout is storage-oriented, not access-oriented.** GGUF interleaves experts, which is why a relayout tool gives a 36× page-fault cut.
4. **Memory claims are routinely overstated.** A 2026 paper (Stepanek, "Memory-Sovereign Inference") shows that process memory excludes the OS page cache, so "runs in 2 GB" often means "uses 2 GB plus 14 GB of cache". Several headline claims I found (a 26B model in 1.85 GB at 14 tok/s; 50–90 tok/s for a 125B model on 12 GB) come from single blogs or project pages and are unverified.
5. **Dense models have no good answer.** Streaming them stays slow; only lower bits or a faster tier helps.

## 5. Proposed design: TierFlow

One sentence: treat a model as a set of addressable objects, measure how often each is used and how fast each device is, place objects to maximise tokens per second inside a hard memory budget, and stream the rest ahead of need.

### 5.1 Rules

1. **Exact by default.** The default mode produces the same tokens as the unmodified quantized checkpoint. Predictors and caches may reorder work, never change results.
2. **Hard budgets.** The user states RAM and VRAM limits; the runtime stays under them including page cache, or refuses to start.
3. **Lossy options are opt-in and named**, each with a measured quality cost: lower-bit weights, 4-bit KV cache, token eviction.
4. **Honest reporting.** Every run prints bytes read per token per tier, cache hit rate and peak memory including page cache.

### 5.2 Components

**Packer (offline, run once per model).**
- Splits the model into objects: one per expert, one per attention block, embeddings, router weights.
- Writes each object contiguous and page-aligned (4 KiB) so one read fetches one object.
- Keeps the checkpoint's own precision; no widened copy. Optionally allocates bits per object under a size budget using sensitivity from calibration data (opt-in, lossy).
- Runs calibration text through the model and records, per object, how often it is used and which objects tend to follow which.
- Writes a manifest: object id, offset, size, format, usage frequency, checksum.

**Probe (at start-up).** Measures real read speed of SSD (direct I/O, queue depth 1 and 32), RAM copy speed, host-to-GPU transfer speed, and compute speed per format. Takes a few seconds; cached per machine.

**Planner.** Chooses a tier for every object (algorithm in 6.1) and prints the plan and predicted tokens per second before loading anything.

**Residency manager.** Holds pinned objects permanently, runs a segmented-LRU cache per tier for the rest, and reads from SSD with direct, asynchronous I/O (io_uring on Linux, unbuffered overlapped I/O on Windows) so the OS page cache is bypassed and the budget is real.

**Prefetcher.** For MoE layers the router result is known before the expert runs; the prefetcher issues reads the moment the router finishes and, using the follow-on statistics, speculatively reads likely experts for the next layer while the current one computes.

**Executor.** Runs layers on kernels that consume the stored format directly. Reuse ggml/llama.cpp kernels; do not write new ones in the first version.

**KV store.** 8-bit by default. Blocks of old context are kept in RAM and, past the budget, written to SSD and read back sequentially (attention reads the whole cache in order, which suits SSDs).

**Draft engine.** A small model that is fully resident proposes several tokens; the large model verifies them in one pass. Draft length adapts to the measured acceptance rate.

### 5.3 What each mode is expected to deliver

These are my projections from the formula in section 2 and the measurements cited, not results.

| Model type | Memory budget | Expected outcome |
|---|---|---|
| MoE, ~3B active of 30B | 4 GB RAM, no GPU | 3–5 tok/s, CPU-bound once cache hit rate passes ~85% |
| MoE, ~100B+ total | 16–32 GB RAM + 6–12 GB VRAM | Usable interactive speed if experts fit in RAM; I/O-dependent and slow if they do not |
| Dense, fits in RAM + VRAM | Same as today | Same speed as llama.cpp, plus the draft-engine gain |
| Dense, larger than RAM | SSD-streamed | Under 1 tok/s even with drafting; honest answer is to use fewer bits or an MoE model |

## 6. Algorithms

### 6.1 Placement

```
inputs:
  objects O, each with size s[o] and expected uses per token f[o]   (from the manifest)
  tiers T = {VRAM, RAM, SSD} with capacity C[t] and per-byte cost c[t]
      c[t] = seconds to make one byte available to compute from tier t   (from the probe)
  reserve: KV cache and activation memory subtracted from C before planning

goal: minimise expected seconds per token
      = Σ_o  f[o] · s[o] · c[tier(o)]        subject to  Σ_{o in t} s[o] ≤ C[t]

method (greedy, optimal for this fractional-knapsack form up to one object per tier):
  1. force-pin objects with f[o] = 1 that are small: router weights, norms, embeddings rows in use
  2. sort remaining objects by f[o] descending   (benefit per byte of promotion = f[o] · Δc)
  3. fill VRAM, then RAM, in that order; the remainder stays on SSD
  4. split each tier's remaining space: pinned set (top by f) and a cache region for the rest
  5. output plan + predicted tok/s; if predicted tok/s < user floor, say so and stop
```

For dense models every `f[o]` is 1, so the planner reduces to "fill the fastest tier first", which is correct.

### 6.2 Decode step

```
for each layer L:
    run attention on resident weights
    ids = router(L)                          # MoE: experts needed now
    need = ids not resident
    submit async reads for need              # direct I/O into pre-registered buffers
    guess = follow_on_stats(L, ids)          # likely experts for layer L+1
    submit low-priority reads for guess
    wait for need; compute experts on what is resident meanwhile
    cache.admit(need); cache.evict by segmented LRU, never evicting pinned objects
    cancel or keep guess reads according to remaining budget
```

A wrong guess costs only wasted I/O; results never depend on it.

### 6.3 Draft-and-verify under streaming

```
k = current draft length
draft k tokens with the small resident model
run ONE pass of the large model over all k positions   # one sweep of streamed weights
accept the longest prefix where the large model agrees (standard rejection rule for sampling)
adjust k toward the value maximising  accepted_tokens / (draft_time + sweep_time)
```

### 6.4 Online re-pinning

Every N tokens, compare observed usage with the manifest's calibration profile; if an object outside the pinned set has been used more than the least-used pinned object by a margin, swap them. This is what makes a long conversation get faster as it settles on a set of experts.

## 7. Language and stack

**Recommendation: C++20, built on ggml/llama.cpp, with the packer in Python.**

- The kernels, quantization formats, GGUF parser and every hardware backend already exist there; rewriting them would take longer than the whole project.
- The current community work on expert streaming is all converging on llama.cpp, so results are comparable and upstreamable.
- The new code (planner, residency manager, prefetcher, I/O layer) is ordinary systems code with no GPU dependency and can be tested without a model.
- Python for the packer because calibration and bit allocation use existing PyTorch tooling and run once.

Rust would be a sound choice for a standalone engine later; it does not pay off while the kernels are C/C++.

## 8. Build and test plan for Gemini

Each phase has a pass condition. Do not start the next phase until it is met.

| Phase | Build | Pass condition |
|---|---|---|
| 0. Baseline | Benchmark harness: tokens/s, peak memory including page cache, bytes read per token. Run stock llama.cpp on one MoE model (Qwen3-30B-A3B) and one dense model | Numbers reproducible within 5% across three runs |
| 1. Packer | Object splitter, contiguous page-aligned container, manifest, calibration profile | Round-trip: unpacked tensors are byte-identical to the source GGUF |
| 2. I/O layer | Direct async reads with registered buffers, Linux and Windows | Reading one expert touches only its own pages; throughput within 20% of the drive's measured sequential speed |
| 3. Residency + executor | Pinned set, segmented-LRU cache, streaming forward pass through ggml | **Exactness:** token ids and logits identical to stock llama.cpp over 1,000 tokens on 20 prompts. Memory stays under the stated budget with page cache counted |
| 4. Planner + probe | Section 6.1 | Predicted tok/s within 25% of measured on three different budgets |
| 5. Prefetcher | Section 6.2 and 6.4 | Cache hit rate ≥ 80% on held-out text; tokens/s improves over phase 3 with identical output |
| 6. Draft engine | Section 6.3 | Output distribution unchanged (identical tokens under greedy decoding); speed-up ≥ 1.5× on the streamed configuration |
| 7. KV store | 8-bit default, SSD spill | 32K-token context runs inside the budget; quality gap versus 16-bit cache within noise on a fixed evaluation set |
| 8. Lossy options | Budget-aware per-object bits, 4-bit KV | Each option ships with a measured quality table; none is on by default |

Evaluation set for every phase: perplexity on a fixed text, a code benchmark (HumanEval), a reasoning benchmark, and KL divergence of logits against the reference. Perplexity alone hides damage at low bits.

## 9. Open questions for you

1. What hardware should the first tests target (RAM, GPU, SSD model)? The plan assumes a Windows 11 machine with an NVMe drive.
2. Is an MoE-first scope acceptable? It is where the large gains are; dense models get only the smaller improvements.
3. Should the result be an upstream contribution to llama.cpp or a separate program?

## Sources

Quantization
- [QTIP (NeurIPS 2024)](https://arxiv.org/pdf/2406.11235)
- [PV-Tuning (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/091166620a04a289c555f411d8899049-Paper-Conference.pdf)
- [Leech Lattice Vector Quantization](https://arxiv.org/pdf/2603.11021)
- [QAM-W](https://arxiv.org/pdf/2605.26339)
- [ICQuant](https://arxiv.org/html/2505.00850v2)
- [VPTQ](https://arxiv.org/pdf/2409.17066)
- [BitNet b1.58 2B4T technical report](https://arxiv.org/html/2504.12285v1)
- [BitNet RAM review](https://wavect.io/blog/bitnet-cpu-llm-ram-requirements/)
- [On-device quantization threshold study](https://arxiv.org/html/2505.15030v5)
- [Quantization guide with HumanEval figures](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026)
- [EXL3 vs GGUF](https://atomic.chat/blog/guides/exl3-vs-gguf)
- [DP-LLM](https://arxiv.org/html/2508.06041v1), [MoBiQuant](https://arxiv.org/html/2602.20191v1), [FlexQuant](https://arxiv.org/pdf/2506.12024)
- [EoRA](https://arxiv.org/html/2410.21271v5), [SVD-LLM](https://arxiv.org/html/2403.07378v5)

Lossless compression
- [DFloat11 (NeurIPS 2025)](https://arxiv.org/html/2504.11651v2)
- [ZipNN](https://arxiv.org/html/2411.05239v2)
- [Huff-LLM](https://arxiv.org/pdf/2502.00922)
- [LEXI](https://arxiv.org/html/2603.15589v1)

Sparsity and flash
- [LLM in a Flash (Apple)](https://arxiv.org/html/2312.11514v3)
- [PowerInfer](https://arxiv.org/html/2312.12456v2)
- [ProSparse](https://arxiv.org/html/2402.13516v6)

Streaming and tiering
- [llama.cpp discussion #27149: expert-aware SSD streaming](https://github.com/ggml-org/llama.cpp/discussions/27149)
- [llama.cpp PR #25294: stream MoE experts from disk](https://github.com/ggml-org/llama.cpp/pull/25294)
- [llama.cpp issue #20757: two-tier expert cache](https://github.com/ggml-org/llama.cpp/issues/20757)
- [Strata (ro99)](https://github.com/ro99/strata)
- [Memory-Sovereign Inference (Stepanek, 2026)](https://arxiv.org/abs/2608.23805)
- [ExpertFlow](https://arxiv.org/html/2410.17954v2), [FlashMoE](https://arxiv.org/abs/2601.17063v1), [MoEpic](https://arxiv.org/html/2509.08342v1), [CPU-GPU collaborative MoE inference](https://arxiv.org/html/2512.16473)
- [Micro-Expert-Router (Rust)](https://dev.to/randyap8wq/i-built-a-rust-inference-engine-that-streams-moe-expert-weights-from-nvme-ssds-no-gpu-required-3bie)
- [AirLLM analysis](https://wavect.io/blog/airllm-layer-wise-inference-low-vram/), [AirLLM speed figures](https://ai-radar.it/article/il-grande-divario-hardware-dell-ia-airllm-e-il-sostituto-giusto-per-la-quantization)
- [awesome-low-memory-llm](https://github.com/notsointresting/awesome-low-memory-llm)

KV cache
- [KIVI (ICML 2024)](https://arxiv.org/html/2402.02750v2)
- [KV cache optimization survey (2026)](https://arxiv.org/html/2603.20397v1)
- [Keep the Cost Down review](https://arxiv.org/html/2407.18003v4)
- [TurboQuant coverage](https://helpnetsecurity.com/2026/03/25/google-turboquant-ai-model-compression)
- [KV cache quantization KL-divergence test](https://localbench.substack.com/p/kv-cache-quantization-benchmark)
- [Rethinking KV cache compression for serving](https://themoonlight.io/review/rethinking-key-value-cache-compression-techniques-for-large-language-model-serving)

Speed
- [LayerSkip](https://ai.meta.com/research/publications/layerskip-enabling-early-exit-inference-and-self-speculative-decoding/)

Context
- [Bloomberg: 2026 memory chip shortage](https://www.bloomberg.com/graphics/2026-ai-boom-memory-chip-shortage/)
- [The Register: DRAM prices expected to double](https://www.theregister.com/2026/02/02/dram_prices_expected_to_double/)
