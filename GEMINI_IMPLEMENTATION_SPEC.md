# TierFlow: implementation specification for Gemini

Companion to [FINAL_REPORT.md](FINAL_REPORT.md). Read sections 6–8 of the report first. This file says **what to build, in what order, and how to prove each piece works.**

Status: **nothing here is implemented.** Every phase ends in an exit criterion that must be met, with real measured output, before the next phase starts.

## 0. Ground rules (non-negotiable)

1. **Exact by default.** In `exact` mode, greedy decoding must produce token IDs identical to stock llama.cpp built from the same commit, on the same GGUF, with the same seed and threads. If it does not, it is a bug. Never loosen the comparison.
2. **No typed numbers.** Every number in a README, table, commit message or chat reply is copied from a results file written by a script under `results/<exp_id>/`.
3. **Pre-register before measuring.** Each benchmark experiment gets a `results/<exp_id>/PREREG.md` before it runs, covering:
   - claim, metric, falsifier
   - frozen configuration
   - prompts split into calibration / validation / test, disjoint
   - replicates (≥ 20 prompts, or ≥ 20 repeated runs where prompts do not apply)
   - baselines and null controls

   Any change after seeing results gets a new `exp_id`.
4. **Strong baselines, run in the same script.** Always include:
   - stock llama.cpp default (mmap)
   - llama.cpp with the best hand-tuned `-ngl` / `--n-cpu-moe` / `-ot` placement found within a fixed tuning budget
   - the upstream expert cache (`--moe-cache-mib`) if it exists at the pinned commit (check, do not assume)
5. **Honest memory.** Report system-wide memory, not only process memory (section 6.3).
6. **Claim levels:** `IMPLEMENTED` / `OBSERVED` / `REPRODUCED` / `CONTROLLED`, kept in `docs/CLAIMS.md`.

## 1. Target environment

- **Development machine** (inventory read 2026-11-12):
  - Windows 11 Home build 26200
  - AMD Ryzen 5 7600 (6 cores / 12 threads)
  - 31.7 GB RAM
  - NVIDIA RTX 3060 12 GB
  - Kingston KC3000 1 TB NVMe
  - Free space: E: 121 GB, F: 69 GB, D: 59 GB
- **Also supported:** Linux (io_uring path), CPU-only builds.
- **Models:**
  - *Development model:* Qwen3-30B-A3B GGUF Q4_K_M. It fits in RAM, so streaming is forced through **artificial budgets** (e.g. `--tf-ram-budget 4G --tf-vram-budget 2G`).
  - *Stretch model:* Qwen3-235B-A22B at ~3 bits per weight (≈ 88 GB, fits on E:).
  - *Dense control:* any 7B–8B dense GGUF, for exactness tests, plus Llama-3-70B Q4 if disk allows.

## 2. Repository layout

```
RAM_optimizer/
  FINAL_REPORT.md
  GEMINI_IMPLEMENTATION_SPEC.md
  tools/
    ceiling_model.py            # exists: analytic bounds
    trace_router.py             # Phase 0: drive llama.cpp, dump router top-k per token per layer
    cache_sim.py                # Phase 0: replay traces through policies (8.1/8.2 of report)
    pack.py                     # Phase 1: GGUF -> .tfpack + manifest.json
    bench.py                    # Phase 0+: tok/s, TTFT, memory, bytes read, exactness, writes results/
    quality_eval.py             # Phase 7+: KL vs reference logits, perplexity, task benchmarks
    sysmem_win.py / sysmem_linux.py  # system-wide memory sampler
  third_party/llama.cpp/        # git submodule pinned to a commit recorded in docs/PINNED.md
  runtime/                      # TierFlow C++ sources compiled into the llama.cpp fork
    tf_manifest.{h,cpp}
    tf_probe.{h,cpp}
    tf_planner.{h,cpp}
    tf_governor.{h,cpp}
    tf_io_win.cpp / tf_io_linux.cpp / tf_io.h
    tf_cache.{h,cpp}
    tf_prefetch.{h,cpp}
    tf_compute_place.{h,cpp}
    tf_spec.{h,cpp}
    tf_kv.{h,cpp}
    tests/                      # GoogleTest or doctest unit tests, no model required
  results/<exp_id>/             # PREREG.md, config.json, raw.jsonl, summary.json, meta.json
  docs/CLAIMS.md
  docs/PINNED.md
```

## 3. Phases

### Phase 0: measure before building (Python + stock llama.cpp only)

**Goal:** find out whether the design's central assumption (a skewed, cacheable expert working set) holds. Also produce the baseline numbers.

Tasks:

1. Pin a llama.cpp commit; build with CUDA on Windows. Record the commit hash and build flags in `docs/PINNED.md`.
2. Read and summarise in `docs/UPSTREAM_STATE.md`. Open the code; do not rely on memory:
   - how MoE is built (`build_moe_ffn` or equivalent)
   - how `ggml_mul_mat_id` consumes expert tensors
   - whether `--moe-cache-mib`, `--n-cpu-moe` and expert streaming (PR #25294 or successors) exist at the pinned commit
3. `trace_router.py`: use llama.cpp's eval-callback mechanism, or a minimal patch, to dump the selected expert IDs (and router weights) per token per layer.
   - Prompts: three disjoint sets of 20 each (cal / val / test), mixing chat, code and long-document text.
   - Record 512 generated tokens per prompt under greedy decoding.
4. `cache_sim.py`: replay val traces at cache sizes {5, 10, 15, 25, 40, 60}% of experts through:
   - Belady (oracle upper bound)
   - LRU
   - LFU-with-aging
   - W-TinyLFU + SLRU (report section 8.2)
   - static top-f pinning
   - random pinning (null control)
   - each of the above with router look-ahead prefetch (report section 8.3) using real `Router_{L+1}(h_L)` guesses; this needs hidden-state dumps for at least the val set
   - the same prefetch with shuffled guesses (null control)
5. Also measure:
   - distinct experts per layer for windows of k = 1, 2, 4, 8, 16, 50 tokens (compare with the uniform-routing numbers in report section 4)
   - the measured union growth U_L(k) used by report section 8.4
6. `bench.py` baselines on the dev model under budgets {4, 8, 16} GB RAM × {0, 2, 6} GB VRAM:
   - stock mmap
   - tuned `--n-cpu-moe` / `-ot`
   - upstream expert cache if present

**Exit criteria** (report all numbers from `results/p0_*`):

- Traces and simulator results exist for val and test sets. The policy ranking on test is reported whether or not it matches val.
- Baseline tok/s, TTFT and system memory are recorded, with 95% confidence intervals over 20 prompts.
- **Go/no-go:** if the best non-oracle policy at 25% cache size has a test hit rate below 0.5, stop. Report to the user, and re-scope toward the "experts fully in RAM" regime (planner + CPU/GPU compute placement + speculation only).

### Phase 1: packer and container

Tasks:

- `pack.py` reads GGUF and writes a `.tfpack` file plus `manifest.json`:
  - one object per expert (gate, up and down for one expert, contiguous), per layer's always-on block, embeddings and output head
  - 4 KiB-aligned offsets
  - xxHash64 per object
  - the original ggml type kept unchanged (no re-quantization)
- **Disk layout options:**
  - `--layout gguf-order` (control)
  - `--layout coact` (spectral co-activation order from the cal traces; report section 8.5)
- The manifest stores per-object `freq` from cal traces, layer, expert index, offset, size, type and checksum.

**Exit criteria:**

- Round-trip test: every tensor reconstructed from `.tfpack` is byte-identical to the source GGUF. This is a unit test in CI.
- Reading any single expert touches only its own pages, verified with an I/O trace (Windows: ETW / Process Monitor; Linux: blktrace or `strace -e pread64`).

### Phase 2: I/O layer

Tasks:

- `tf_io.h` interface:
  ```cpp
  struct tf_read_req { uint32_t object_id; uint64_t offset, size; void* dst; uint8_t priority; uint64_t tag; };
  class tf_io {
   public:
    virtual bool   open(const char* path) = 0;          // unbuffered
    virtual bool   submit(const tf_read_req&) = 0;       // non-blocking
    virtual int    poll(tf_read_req* done, int max) = 0; // completed requests
    virtual bool   cancel(uint64_t tag) = 0;             // best-effort for low-priority reads
    virtual size_t alignment() const = 0;                // sector/page alignment of dst/offset/size
  };
  ```
- **Windows:** `CreateFileW` with `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`, an I/O completion port, aligned buffers (`VirtualAlloc`), and queue depth up to 32. Optional second backend: IoRing (`CreateIoRing`, `BuildIoRingReadFile`) behind a runtime check.
- **Linux:** `O_DIRECT` + `io_uring` (liburing), registered buffers, optional SQPOLL.
- **Two priority queues.** Demand reads are never queued behind prefetch reads; prefetch reads are cancellable.

**Exit criteria:**

- Sequential throughput is within 20% of the drive's own measured sequential read speed (measure the drive with the same harness at QD32, 4 MiB reads).
- The system file-cache size does not grow during a 60-second read of 50 GB of experts, as reported by `sysmem_*`.
- Unit tests pass: alignment errors are rejected; cancel works; out-of-order completion is handled.

### Phase 3: residency cache, governor, and the streaming forward pass (first end-to-end)

Tasks:

- `tf_governor`: a tagged accounting API. Every allocation goes through `reserve(bytes, tag)` / `release`. A start-up check rejects plans over budget.
- `tf_cache`: a pinned set plus W-TinyLFU + SLRU (report section 8.2), with per-slot reference counts.
- **Integration with llama.cpp MoE compute**, chosen from Phase 0 findings and recorded in `docs/INTEGRATION.md`. Options to evaluate:
  - **(a) Indirection buffer.** The expert tensor for layer L is a fixed set of cache slots; a per-step slot map converts router expert IDs to slot IDs before `mul_mat_id`.
  - **(b) Custom op** that gathers resident experts.
  - **(c) Extend the upstream expert cache** if present.
- CPU compute of RAM-resident experts uses existing ggml CPU kernels.

**Exit criteria:**

- **Exactness:**
  - Dev model, budget 4 GB RAM / 2 GB VRAM: greedy token IDs identical to stock llama.cpp on 20 test prompts × 256 tokens.
  - Logits within the run-to-run tolerance that stock llama.cpp shows against itself. Measure that tolerance first and record it in PREREG.
- Peak system memory (section 6.3) stays within budget + 5% on all 20 prompts.
- Report tok/s against the Phase 0 baselines at identical budgets. **A slowdown is a valid result. Report it as such.**

### Phase 4: probe and planner

Tasks:

- `tf_probe`: SSD QD1/QD32 at 1, 4 and 16 MiB; RAM copy bandwidth; PCIe H2D bandwidth; CPU and GPU GEMV rates per ggml type. Cache results per machine.
- `tf_planner`: report section 8.1. Simulation inside the planner reuses the cache model from `cache_sim.py`, ported or called offline.
- The plan prints before loading: bytes per tier, predicted hit rate and predicted tok/s.

**Exit criteria:**

- Predicted tok/s is within 25% of measured tok/s at three different budgets. If not, report the error and fix the cost model, not the measurement.

### Phase 5: router look-ahead prefetch, online re-pinning, compute placement

Tasks: report sections 8.3, 8.9 and 8.6.

**Exit criteria (all on test prompts):**

- Output identical to Phase 3 (exactness test re-run).
- Measured hit rate and tok/s improve over Phase 3 at the same budget, with 95% CIs non-overlapping. Otherwise report "no improvement".
- **Null controls:**
  - With shuffled prefetch guesses, the gain disappears.
  - With re-pinning disabled, its contribution is quantified.

### Phase 6: union-aware speculative decoding

Tasks:

- Use the upstream speculative / MTP / EAGLE-3 code paths that exist at the pinned commit (check them).
- Add the k-selection policy (report section 8.4) and union-batched expert fetch in the verify pass.

**Exit criteria:**

- **Greedy:** tokens identical to non-speculative output.
- **Sampling:** the output distribution passes a two-sample test against non-speculative sampling. Use 2,000 samples on fixed prompts, a chi-square test on the first-token and bigram distributions, and α fixed in PREREG.
- Speed-up against Phase 5 is reported, together with:
  - fixed k = 4 (control)
  - k chosen ignoring the union (control)

### Phase 7: KV store

Tasks: report section 8.8. Paged q8_0, SSD spill, disk prefix cache, and MLA / hybrid-attention handling where llama.cpp supports the architecture.

**Exit criteria:**

- A 32K-token context runs inside the stated budget.
- Quality versus a 16-bit KV cache on a fixed evaluation set is within the noise band measured between two 16-bit runs.
- The prefix cache returns identical logits to a cold prefill.

### Phase 8: opt-in lossy ladder

Each item is a separate flag and a separate experiment:

- 4-bit V cache
- low-bit miss experts (HOBBIT-style shadow copies)
- cache-aware routing bias (prefer a resident expert only when its router score is within ε of a non-resident one)
- REAP-pruned pack
- TEAL thresholds
- dynamic low-bit experts

**Exit criteria (per item):**

- A quality table covering KL divergence of next-token distributions against the exact mode, perplexity on a fixed text, a code benchmark (HumanEval-style) and a reasoning benchmark, each with CIs.
- A speed / memory table.
- Off by default.

## 4. Metrics (written by `bench.py` to `results/<exp_id>/raw.jsonl`, one line per prompt × run)

`exp_id`, git commit + dirty flag, llama.cpp commit, model file hash, full config, prompt id, split, and:

- `ttft_ms`, `decode_tok_s`, `prefill_tok_s`
- `bytes_read_ssd_per_token`, `hit_rate_vram`, `hit_rate_ram`, `prefetch_accuracy`, `prefetch_wasted_bytes`
- `spec_k_mean`, `spec_accept_rate`
- `peak_process_private_bytes`, `peak_system_used_bytes` (see 6.3), `peak_file_cache_bytes`, `peak_gpu_used_bytes` (board-wide, from NVML)
- `exact_match_tokens` (bool + first divergence index)
- UTC timestamp and the exact command line

`summary.json` holds mean, SD, 95% CI, min, max and n per metric per configuration.

## 5. Unit tests that do not need a model (`runtime/tests`)

- **Cache:**
  - W-TinyLFU beats LRU on a synthetic Zipf + burst trace with known optimal order.
  - It never evicts a referenced slot.
  - Belady on a known trace gives the textbook miss count.
- **Planner:**
  - Dense model: fills the fastest tier first.
  - Known toy instance: the result matches brute force over all placements (≤ 12 objects).
- **Governor:** over-budget plan is rejected; tags sum correctly.
- **Spec scheduler:**
  - Returns k = 0 when α → 0.
  - Grows k when U_L(k) is flat (dense-like).
  - Matches the closed form `(1 − α^(k+1)) / (1 − α)` for accepted tokens.
- **I/O:** misaligned request rejected; cancelled request never written to its buffer after cancel acknowledgement.

## 6. Measurement details

### 6.1 Run hygiene

- Fixed thread count; high-performance power plan; no other heavy processes.
- **Cold-cache runs:** reboot, or flush the standby list with a documented tool, and say which.
- **Warm-cache runs:** labelled separately.

### 6.2 Replicates

20 test prompts per configuration. Report per-prompt values plus the CI. Exactness is a count, "x of 20", with a Wilson interval.

### 6.3 System-wide memory on Windows

Sample every 100 ms:

- `GetPerformanceInfo` (PhysicalTotal, PhysicalAvailable, SystemCache)
- `GetProcessMemoryInfo` (PrivateUsage, WorkingSetSize)
- performance counters `\Memory\Standby Cache Normal Priority Bytes`, `\Memory\Modified Page List Bytes`, `\Memory\Cache Bytes`
- NVML for GPU memory

`peak_system_used_bytes` = PhysicalTotal − PhysicalAvailable at peak, minus the same quantity measured idle before the run. Standby/cache growth is reported separately. **Never report only the process working set.**

### 6.4 Bytes read

Count bytes in `tf_io`. Cross-check against `\PhysicalDisk(*)\Disk Read Bytes/sec` once per experiment.

## 7. Definition of done for the whole project

TierFlow is "done" for a model and machine when all of the following hold, each at claim level `CONTROLLED`:

- exact-mode output identity on test prompts
- a memory budget honoured system-wide
- a measured tok/s against the strongest llama.cpp baseline at the same budget, with confidence intervals

If the measured result is slower than the baseline, the project still reports it. That outcome is valid and useful.

