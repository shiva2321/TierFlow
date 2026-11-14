# TierFlow v2 Optimization & Scaling Plan (Revised)

> **Purpose:** Engineering roadmap to evolve TierFlow from the Phase 0–8 single-SSD CPU streaming prototype (`0.710–2.085 tok/s` dense on 600 MiB RAM budget) to interactive **10–20+ tok/s** without compromising output quality, grounded strictly in measured Phase 0–8 data and realistic near-term scope.
>
> **Status Distinction:**
> - **Measured:** Verified from repository artifacts (`results/p0_*` through `results/p8_*`, `FINAL_REPORT.md` Section 11, `docs/CLAIMS.md`, `tf_tests.exe`) on `Windows 11`, `Ryzen 5 7600`, `RTX 3060 12GB`, `KINGSTON SKC3000S1024G` NVMe, `qwen2.5-1.5b-instruct-q4_k_m.gguf`.
> - **Literature:** Published 2024–2026 techniques reviewed for architectural fit.
> - **Planned:** TierFlow v2 workstreams and acceptance gates requiring implementation and empirical measurement.

---

## 1. Verified Codebase Bottlenecks & Measured Constraints

### 1.1 Code-Level Bottlenecks Observed
1. **CPU-Only Execution & Unwired GPU/MoE Cache ([runtime/CMakeLists.txt](file:///e:/RAM_optimizer/runtime/CMakeLists.txt), [runtime/tf_llama_stream.cpp](file:///e:/RAM_optimizer/runtime/tf_llama_stream.cpp#L654), [L758–759](file:///e:/RAM_optimizer/runtime/tf_llama_stream.cpp#L758-L759)):**
   - Links only `llama` and `ggml-cpu`; hardcodes `p_budget.vram_bytes = 0`, `mparams.n_gpu_layers = 0`, `mparams.use_extra_bufts = false`.
   - [choose_compute_target()](file:///e:/RAM_optimizer/runtime/tf_compute_place.h#L24-L53) and upstream [llama_moe_cache](file:///e:/RAM_optimizer/third_party/llama.cpp/src/llama-moe-cache.h#L13-L46) already exist in the tree but are not wired into `tf_llama_stream.cpp`.
2. **Whole-Tensor Packing Instead of MoE Expert Slicing ([tools/pack.py](file:///e:/RAM_optimizer/tools/pack.py#L198-L206), [runtime/tf_llama_stream.cpp](file:///e:/RAM_optimizer/runtime/tf_llama_stream.cpp#L602-L625)):**
   - `pack.py` packs each `blk.<i>.ffn_*_exps` tensor as one monolithic `"moe_experts"` blob, and `tf_llama_stream.cpp` only splits dense `ffn_gate/up/down` and `attn_*` roles. Streaming dense weights reads ~100% of evicted layers (`698.35 MB/tok` in `stream600_spec_off`).
3. **Synchronous Spin-Wait & Hard-Capped `QD=4` Prefetch ([runtime/tf_llama_stream.cpp](file:///e:/RAM_optimizer/runtime/tf_llama_stream.cpp#L350-L385), [L727](file:///e:/RAM_optimizer/runtime/tf_llama_stream.cpp#L727)):**
   - `acquire_object_sync()` polls in a `std::this_thread::yield()` loop when prefetch has not finished; `prefetch_inflight_cap = 4` and `(n_slots - 2)/2` cap concurrent lookahead.
4. **Uniform Worst-Case Slot Stride ([runtime/tf_llama_stream.cpp](file:///e:/RAM_optimizer/runtime/tf_llama_stream.cpp#L624)):**
   - Allocates every RAM cache slot at `max_stream_aligned`, wasting pool bytes on smaller attention/router objects and reducing cached slot count.
5. **Tight-Slot `--repin` Starvation ([runtime/tf_llama_stream.cpp](file:///e:/RAM_optimizer/runtime/tf_llama_stream.cpp#L469-L477)):**
   - Measured in `p5_prefetch_repin`: pinning `4 / 12` slots shrunk dynamic Probation capacity to `8` slots, halving prefetch submissions (`14,848` vs `30,314`), dropping RAM hit rate (`51.91%` vs `99.69%`), and slowing throughput (`1.208` vs `1.423 tok/s`).

### 1.2 Measured Guardrails From Phases 0–8 That Constrain v2 Design
1. **Batch-Verification Size vs. Greedy Identity & MoE Expert Explosion:**
   - In `p6_spec` (`C14`), `fixed4` ($k \le 4$) preserved **20 / 20** exact greedy tokens on both RAM and stream, whereas `adaptive` ($k \le 8$) flipped FP rounding ties mid-rollout (**6 / 20** exact on `ram1400`, **14 / 20** on `stream600`).
   - In `p0_cache_sim_v2` (`128` experts, `top-8`), distinct experts touched per layer grow from **`k=1: 8.0`** $\to$ **`k=4: 18.25`** $\to$ **`k=8: 28.12`**. Wide draft trees (`SpecExec` multi-branch batches of 20–100+ tokens) multiply MoE SSD reads (`3.5×+` more experts per pass at $k=8$ alone) and threaten bit/token equivalence.
2. **Router Visibility in Standard MoE Models:**
   - On standard architectures (Qwen3, Mixtral), a layer's MoE router takes that same layer's **post-attention** hidden state as input—so exact expert IDs for layer $L$ are not known while attention is running. Pre-attention exact routing only exists on bespoke architectures like SmallThinker (`arXiv:2507.20984`). Cross-layer early prefetch on standard MoE is **heuristic prediction**: safe (a miss wastes I/O without corrupting output), but hit rate must be empirically measured against shuffled-guess null controls.
3. **KV Quantization Compute Penalty on CPU:**
   - In `p7_kv` (`C10`), even `q8_0` KV at ~4.2K prompt length ran **~40% slower** than `f16` on CPU (`18.02 tok/s` vs `30.07 tok/s` decode; `63.57 tok/s` vs `201.57 tok/s` prefill) and changed greedy outputs on **15 / 20** prompts (`5 / 20` full 64-token matches). Memory compression via heavier schemes (`ROT_Q3_QJL`) is not the active bottleneck until long-context MoE GPU/RAM tests hit an actual KV capacity wall.
4. **Thermal Throttling & System Memory Accounting:**
   - Sustained SSD streaming (`p2_io_win_v2`, `p4_planner_v2`) throttles the `SKC3000` NVMe from `7.35 GB/s` burst $\to$ `2.8 GB/s` sustained $\to$ **`0.293 GB/s`** post-soak (`16.5×` collapse).
   - `ram_governor` undercounted total system delta in `p3_e2e_stream` (`598.9 MiB` governor vs **`679.1 MiB` `peak_system_used_delta_bytes`**, failing the `660 MiB` gate) and `p6_spec` (`598.9 MiB` vs **`748.4 MiB`**).
5. **Local Prerequisites Missing (`docs/PINNED.md`):**
   - Neither **CUDA Toolkit (`nvcc`)** nor **Vulkan SDK** is installed, and **no MoE GGUF model** is present on any local drive (`~121 GB` free on `E:`).

---

## 2. Physics Reality Check: Where 10–20+ tok/s Is Achievable

* **MoE Models (Target Route for 10–20+ tok/s at Low RAM):** Sparse routing activates ~3–10% of parameters per token. Pinning attention + routers + KV + top-frequency experts in GPU VRAM + host RAM yields high hit rates (~80–90% in simulation at 25% cache capacity with lookahead), keeping cold NVMe misses small enough for GPU-hybrid compute + universal $k \le 4$ speculative decoding to reach interactive speeds even on consumer hardware.
* **Dense Models:** Streaming a whole dense model (e.g., 70B `Q4` ~40 GB) from one SSD per token cannot hit 10–20 tok/s due to single-drive PCIe bandwidth physics (~0.08–0.7 tok/s autoregressive/small-spec ceiling). TierFlow keeps dense models useful primarily when **compressed weights fit hybrid `(VRAM + RAM)`** (avoiding per-token SSD reads via GGUF dynamic quants), plus optional `exact` prompt-lookup/small-draft speculation ($k \le 4$) and **strictly opt-in** lossy acceleration (`TEAL` / `SpecExec` trees) clearly separated from the default `exact` contract.

---

## 3. TierFlow v2 Plan (Tightened, Exact-Default Architecture)

### Workstream 1: GPU-Hybrid Execution & Pool Fixes (Pre-Requisite for Speed)
**Files:** `runtime/CMakeLists.txt`, `runtime/tf_llama_stream.cpp`, `runtime/tf_planner.cpp`, `runtime/tf_compute_place.h`, `third_party/llama.cpp/src/llama-moe-cache.h`

1. **Optional CUDA / Vulkan Build:** Add `TF_ENABLE_CUDA` / `TF_ENABLE_VULKAN` options in `CMakeLists.txt` (keeping CPU build default when SDKs are absent).
2. **VRAM Placement + `llama_moe_cache` Integration:** Probe free VRAM (`--vram-reserve-mib 512`), populate `p_budget.vram_bytes`, pin attention, norms, routers, shared experts, KV, and hot weights to GPU via `tensor_buft_overrides`, and connect `choose_compute_target()` so cold RAM/SSD expert misses evaluate CPU-in-place for $B \le 2$ vs GPU upload for larger batches (`C2` crossover).
3. **3-Size-Class Arena & `--repin` Guard (`tf_llama_stream.cpp`):** Replace uniform `slot_stride_` with 3 size classes (`attn/router`, `expert`, `large slab`) to eliminate wasted arena RAM, and disallow `--repin` when dynamic unpinned Probation slots would fall below `max(inflight_cap + 4, capacity / 2)`.
4. **Live `io_bw_ewma` & Non-Spin Wait (`tf_llama_stream.cpp`, `tf_spec.cpp`):** Feed rolling completed NVMe MB/s (`io_bw_ewma_gbps`) into `tf_spec_scheduler` so thermal drops (`2.8 -> 0.3 GB/s`) adjust decisions live; replace spin-`yield()` in `acquire_object_sync` with OS completion wait (`GetQueuedCompletionStatusEx` / IoRing event / `io_uring_wait_cqe`).

### Workstream 2: MoE Expert Slicing & Empirical Prefetch Pipeline (`exact`, 0 Quality Loss)
**Files:** `tools/pack.py`, `runtime/tf_manifest.{h,cpp}`, `runtime/tf_llama_stream.cpp`, `runtime/tf_prefetch.h`

1. **Contiguous Per-Expert Packing (`pack.py --slice-moe-experts`):** Split `blk.<i>.ffn_{gate,up,down}_exps` along the expert axis and co-pack each `(layer, expert_id)` `{gate, up, down}` triple into **one 4096-aligned `.tfpack` object** so one NVMe read fetches the whole expert.
2. **Raise Prefetch Concurrency & Testable Cross-Layer Prediction:** Remove the hardcoded `prefetch_inflight_cap = 4` ceiling in favor of hardware-probed QD (`16–32` for Gen4 NVMe). Queue layer $L$'s non-resident top-$k$ experts immediately after `ffn_moe_logits` finishes; treat any cross-layer $L \to L+1$ transition predictor (EAM table) strictly as an **overlap prefetch whose accuracy and wasted-byte overhead must beat a shuffled-guess null control**.
3. **Out-of-Order Expert Dispatch (Within Exact Floating-Point Tolerance or Bitwise Accumulation):** Compute RAM/VRAM-resident experts first while missing experts finish NVMe DMA in parallel, summing expert outputs in canonical deterministic index order so partial sums remain exact.

### Workstream 3: Universal Small Draft Model (`k <= 4`) Speculative Decoding (Exact Default)
**Files:** `runtime/tf_draft.{h,cpp}` (new), `runtime/tf_spec.{h,cpp}`, `runtime/tf_llama_stream.cpp`

1. **Universal Pinned Small Draft Model (Instead of EAGLE-3):** Pin any standard small GGUF model from the target family (`0.5B` Q4/Q8, ~350–500 MiB in VRAM/RAM) via `--draft-model <path>`. Unlike EAGLE-3 (which requires a separately trained per-model feature head), a standalone small GGUF works across **any model family**. *(EAGLE-3 remains optional only if pre-trained heads exist, never a core dependency.)*
2. **Strict `k <= 4` Draft Budget by Default:** Cap speculative batch verification at **`k <= 4`** to match our `C14` empirical finding (`20/20` greedy identity preserved at `fixed4`, while `k <= 8` dropped to `6/20` and bloated MoE distinct experts from `8 -> 28.12/layer`).
3. **Experimental Wide Trees & Lossy Ladder Kept Strictly Opt-In:**
   - **Wide SpecExec trees ($k > 4$):** Implemented behind an explicit experimental flag (`--spec-tree`) to measure greedy divergence + MoE union traffic inflation vs net speedup, **not** enabled by default.
   - **Lossy Opt-In Flags Only (Excluded from `exact` Gates):** `HOBBIT` low-bit shadow miss experts (`--miss-quant`), `TEAL` column-skipped activations (`--teal-sparsity`), and `REAP` offline expert pruning stay strictly opt-in under the `budget-lossy` ladder with separate perplexity/accuracy gates.

### Workstream 4: Cross-Platform Linux I/O & Deferred Scope
**Files:** `runtime/tf_io_linux.cpp`, `runtime/CMakeLists.txt`

1. **Near-Term Scope (`tf_io_linux.cpp`):** Complete `io_uring` `O_DIRECT` + registered buffers parity on Linux alongside Windows IOCP/IoRing (`tf_io_win.cpp`, `tf_io_ioring_win.cpp`), plus multi-drive `.tfpack` sharding only after single-NVMe MoE streaming is verified.
2. **Explicitly Deferred (Do Not Build Yet):**
   - **3-bit Rotated/Polar KV (`TurboQuant` / `ROT_Q3_QJL`):** Deferred until GPU/MoE benchmarks demonstrate KV capacity is an actual bottleneck (since `p7_kv` showed `q8_0` already slowed CPU prefill/decode and perturbed 15/20 greedy runs). Keep `f16` default (`20/20` exact) and existing `q8_0` / opt-in `q8_q4`.
   - **Server-Farm Transports (`GPUDirect Storage / cuFile`, `RDMA / NIXL`, `CXL Type-3`):** Deferred until single-node GPU+RAM+NVMe MoE streaming works end-to-end and clears Phase 10 gates.

---

## 4. Tightened Phased Roadmap & Pre-Registered Gates

Every phase keeps TierFlow's pre-registration rules (`PREREG.md`, null controls, heat-soaked NVMe pacing, and **both** `ram_governor` and OS `peak_system_used_delta_bytes` enforcement).

| Phase | Scope | Concrete Acceptance Gate (All Mandatory) |
| :--- | :--- | :--- |
| **Phase 9: Hybrid VRAM + Pool Fixes** | CUDA/Vulkan backend, VRAM attention/KV + `llama_moe_cache` hookup, 3-size-class pool, `--repin` fix, live `io_bw_ewma`, non-spin wait | Pass `tf_tests.exe` 100%; on `qwen2.5-1.5b` dense (`stream600` equivalent or VRAM+RAM split), zero `--repin` starvation, **`peak_system_used_delta_bytes <= budget * 1.10`**, greedy `20/20` exact vs stock GPU-offloaded `llama.cpp` |
| **Phase 10: Real MoE Slicing & Prefetch (Heat-Soaked Gate)** | `pack.py --slice-moe-experts`, per-expert objects, early post-router queueing + empirical EAM prefetch | Named target model (`Qwen3-30B-A3B-Q4_K_M`, ~18 GB, or `OLMoE-1B-7B-Q4_K_M` ~4 GB on local 121 GB free disk), executed **after 60s NVMe thermal heat-soak**, with an explicit forced out-of-core budget (e.g. combined `VRAM_cap + peak_system_used_delta_bytes <= 50%` of model file size: for `Qwen3-30B-A3B`, e.g., `VRAM <= 4.0 GiB` + `peak_system_used_delta_bytes <= 4.0 GiB`), beating `llama.cpp --mmap` under the **same system-RAM + VRAM ceiling** with **20 / 20 exact greedy parity** and crossing **>= 10 tok/s** post-soak; cross-layer prefetch must outperform a shuffled-guess null control |
| **Phase 11: Universal Small Draft Model ($k \le 4$) & Opt-In Experiments** | `--draft-model` (small pinned family GGUF, $k \le 4$ default) + union-aware `tf_spec_scheduler` live BW; `--spec-tree` & lossy `HOBBIT`/`TEAL` strictly opt-in ablations | Default $k \le 4$ draft on MoE & RAM-fit dense improves `general` prompt tok/s over autoregressive baseline while keeping **20 / 20** greedy match (and Pearson $\chi^2$ `p >= 0.01` at `temp=0.8`); separately log greedy divergence & SSD bytes/tok inflation for `--spec-tree` experiments |
| **Phase 12: Linux `io_uring` Parity** | Complete `tf_io_linux.cpp` (`io_uring` `O_DIRECT` + `IORING_REGISTER_BUFFERS`) + CMake cross-platform build | Linux `tf_io_bench` achieves `>= 80%` raw `liburing` bandwidth, `0` OS page-cache growth (`SystemCache` Linux `/proc/meminfo` PageCache control), and 100% cancel/alignment unit tests |

*(3-bit rotated KV and server-farm GDS/RDMA/CXL remain deferred beyond Phase 12 until Phases 9–12 establish live MoE bottlenecks.)*

---

## 5. Immediate Actionable Blockers Before Phase 9 Code Edits
1. Install **CUDA Toolkit (`nvcc`)** (or **Vulkan SDK**) to compile and benchmark `ggml-cuda`/`ggml-vulkan` on the local `RTX 3060 12GB`.
2. Download a compact MoE GGUF fitting inside the ~121 GB free space on `E:` (e.g., `Qwen3-30B-A3B-Q4_K_M` ~18 GB + `Qwen3-0.6B` draft ~0.4 GB, or `OLMoE-1B-7B` ~4 GB for rapid iteration).
