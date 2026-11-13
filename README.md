# TierFlow — Zero-`mmap`, Budget-Bounded Tiered LLM Inference Engine

TierFlow is a deterministic, memory-governed C++20 inference runtime built on top of a minimal surgical hook in pinned `llama.cpp` (`3ba1ce0e`). Instead of relying on OS `mmap()` virtual paging—which silently inflates the Windows/Linux page cache and stalls unpredictably on cold SSD reads—TierFlow explicitly owns every byte across **VRAM / host RAM / unbuffered NVMe SSD** under a strict allocation governor, async `FILE_FLAG_NO_BUFFERING` / `IoRing` / `O_DIRECT` I/O, a **W-TinyLFU + SLRU + Pinned** object cache, cross-layer look-ahead prefetching, disk-backed prefix KV caching, and union-aware speculative decoding.

> **Project Status (`main` — v1 Phases 0–8 Baseline):**
> Verified end-to-end on CPU (`Windows 11`, `Ryzen 5 7600`, `KINGSTON SKC3000S1024G` NVMe) using `qwen2.5-1.5b-instruct-q4_k_m.gguf`. Active `v2` development targets GPU-hybrid execution, fine-grained MoE expert slicing, and universal small-draft acceleration.

---

## Architecture Overview

```
+-------------------------------------------------------------------------+
|                           TierFlow Runtime                              |
|                                                                         |
|  [ tf_probe ] ---> HardwareProfile (DDR5 BW, PCIe H2D, low-QD NVMe)    |
|        |                                                                |
|        v                                                                |
|  [ tf_planner ] -> Exact 3^N (N<=14) / heuristic placement under        |
|        |           [ tf_governor ] hard byte ceiling                    |
|        v                                                                |
|  [ tf_llama_stream ] (via 37-line llama.cpp cb_eval / tensor hook)      |
|     |-- Pin Tier: Embds / Head / Norms / Hot Weights in RAM (or VRAM)   |
|     |-- Stream Arena: W-TinyLFU + SLRU + Pinned [ tf_cache ]            |
|     |-- Look-Ahead Prefetch: Dual-priority queue (Demand > Prefetch)    |
|     |-- KV Store [ tf_kv ]: f16 / q8_0 / opt-in q8_q4 + Disk Prefix     |
|     '-- Spec Decode [ tf_spec ]: Adaptive / fixed-k Leviathan verifier  |
|        |                                                                |
|        v                                                                |
|  [ tf_io ] ---> Page-aligned 4096B .tfpack via Win32 IOCP / IoRing /    |
|                 Linux O_DIRECT (0 bytes OS SystemCache growth)          |
+-------------------------------------------------------------------------+
```

### How TierFlow Differs From Mainstream `mmap` Runtimes

| Property | Stock `llama.cpp` (`mmap`) | TierFlow v1 (`main`, Phases 0–8) |
| :--- | :--- | :--- |
| **OS Page Cache Pollution** | Inflates OS `SystemCache` (`+511.9 MiB` on a 512 MiB buffered read control) | **0 MiB** OS cache growth (`FILE_FLAG_NO_BUFFERING` / `IoRing`, PDH `99.88%` match in `p2_io_win_v2`) |
| **I/O Scheduling** | Implicit OS demand page faults; prefetches compete head-on with compute | **Strict 2-priority queue**: demand reads (`0.708 ms` mean) stay **13.09× faster** than single-FIFO under prefetch storms (`9.269 ms`) |
| **Cold-Miss & Thrash Resistance** | LRU/FIFO OS eviction thrashes on scan traffic | **W-TinyLFU + SLRU + Pinned** cache (`99.69%` hit rate in `p5`, look-ahead accuracy `99.970%`, `2.94×` spec streaming speedup in `p6`) |
| **Container Alignment** | Raw GGUF byte offsets | **4096-byte page-aligned `.tfpack`** (`tools/pack.py`, 339/339 SHA-256 verified) |
| **Default Quality Contract** | Silent greedy divergence when weight repacking is enabled (`8/20` match) | **`exact` by default**: `tierflow_stream600` & `ram1400` match `stock_mmap_norepack` **20 / 20 greedy tokens** (`max_logit_diff = 0.0`); opt-in lossy modes isolated |

---

## Verified Measured Highlights (Phases 0–8)

All benchmarks below were measured directly on `Windows 11` (`Ryzen 5 7600`, `KINGSTON SKC3000S1024G` PCIe 4.0 NVMe) with full raw artifacts under [`results/`](results/) and claimed gates audited in [`docs/CLAIMS.md`](docs/CLAIMS.md) & [`FINAL_REPORT.md`](FINAL_REPORT.md).

1. **Async Unbuffered NVMe I/O (`p2_io_win_v2`):**
   - `tf_io_win` (IOCP): **`7.312 ± 0.024 GB/s`** (`4 MiB seq QD=32`); **`7.328 ± 0.006 GB/s`** (`1 MiB rand QD=32`).
   - `tf_io_ioring_win` (`IoRing MaxVersion=400`): **`7.331 ± 0.006 GB/s`** (`178.66 GB` sustained over 60s, `0` I/O errors, `-60.74 MiB` SystemCache delta vs idle).
2. **Exact Zero-`mmap` End-to-End Streaming (`p3_e2e_stream`, `qwen2.5-1.5b-instruct-q4_k_m`):**
   - `tierflow_ram1400`: **`43.23 ± 0.84 tok/s`**, **`20 / 20` exact** vs `stock_mmap_norepack` (`44.01 ± 0.36 tok/s`, `max_logit_diff = 0.0`).
   - `tierflow_stream600` (`598.9 MiB` governor, `0` mmap): **`1.66 ± 0.69 tok/s`**, **`20 / 20` exact**, `99.992%` look-ahead accuracy.
3. **Look-Ahead Prefetch vs. Shuffled Null Control (`p5_prefetch_repin`, `600 MiB`, `60s` thermal cooldown):**
   - `lookahead_d1_wtinylfu`: **`1.423 ± 0.598 tok/s`**, `223.52 GB` SSD read (`99.970%` accuracy).
   - `prefetch_shuffled_null` control: **`0.508 ± 0.269 tok/s`**, `425.19 GB` SSD read (`1.90×` extra traffic, `0.018%` accuracy).
4. **Union-Aware Speculative Decoding (`p6_spec`, `600 MiB` cold stream):**
   - `stream600_spec_adaptive` (**`2.085 ± 0.710 tok/s`**, **`2.94×`** overall, **`9.08×`** on `code_overlap` at `2.533 ± 0.804 tok/s`, `-43.3%` SSD reads) vs `stream600_spec_off` (`0.710 ± 0.486 tok/s`).
   - `stream600_spec_fixed4` ($k \le 4$): **`1.602 ± 0.574 tok/s`** with **`20 / 20` exact greedy match**.
   - Stochastic verification (`temp = 0.8`, `N = 2000`, `code_01`): passed all Pearson $\chi^2$ parity tests ($\alpha = 0.01$, $p = 0.21\text{--}0.64$).
5. **Prefix Caching & KV Compression (`p7_kv`, `p8_lossy_kv_v4`):**
   - Persistent disk prefix restore (`tf_kv_store`): **20 / 20 exact**, reducing warm TTFT **`12.36×`** at `n_ctx=2048` (`287.99 ms -> 23.30 ms`) and **`782.9×`** at `n_ctx=32768` (`67,136.5 ms -> 85.76 ms`).
   - `q8_0` KV halved KV memory (`31.2 MB` vs `58.7 MB` at 2K context, fit `32K` in `1,677.6 MB`), while opt-in `--kv-type q8_q4` cut KV allocation by **`-59.4%`** (`23.86 MB`).

### Known Hardware Constraints & Open Gaps on `main`
* **Single-SSD dense streaming physics & thermal throttling:** Sustained reads throttle the NVMe controller from **`7.35 GB/s`** burst $\to$ **`2.8 GB/s`** sustained $\to$ **`0.293 GB/s`** after extended heat-soak (`p4_planner_v2`), capping single-SSD dense-model streaming around `~1–2 tok/s`.
* **System-wide commit delta:** In `p3`/`p6`, `ram_governor` held `598.9 MiB`, while OS `peak_system_used_delta_bytes` reached `679.1 MiB` (`p3`) and `748.4 MiB` (`p6`). Both metrics are logged separately.
* **Not yet run on `main`:** Live GPU (`CUDA`/`Vulkan`) execution and live MoE GGUF streaming remain **UNVERIFIED** on `main` (addressed on branch `v2`).

---

## Hardware & OS Safety Guarantees

Because TierFlow operates close to OS memory and block storage (`VirtualAlloc`, `FILE_FLAG_NO_BUFFERING`, `IoRing`, PDH hardware counters), the runtime enforces strict safety invariants verified by `tf_tests`:
* **Strict alignment enforcement:** `tf_io` rejects any misaligned offset, byte count, or host buffer pointer (`offset % 4096`, `size % 4096`, `dst % 4096`) before issuing OS calls.
* **Post-cancel DMA safety (`tf_cache` refcounts):** Buffers remain pinned by reference count during async I/O; cancelled prefetch tags are guarded so kernel completions never overwrite reused cache slots (covered by unit test post-cancel zero-write assertions).
* **Bounded virtual/physical pool (`tf_governor`):** All slot pools, KV stores, and prefetch staging areas allocate within deterministic pre-checked byte quotas rather than unbounded heap growth.
* **Zero page-cache pollution:** Unbuffered flags prevent large streamed weights from evicting the host OS's active application working sets.

---

## Repository Layout

| Path | Description |
| :--- | :--- |
| [`runtime/`](runtime/) | C++20 core library (`tf_governor`, `tf_cache`, `tf_kv`, `tf_manifest`, `tf_probe`, `tf_planner`, `tf_spec`, `tf_io_{win,ioring_win,linux}`), `tf_llama_stream`, `tf_runner`, `tf_io_bench`, `tests/test_main.cpp` |
| [`tools/`](tools/) | Python utilities: `.tfpack` packer (`pack.py`), hardware crossover prober (`hw_probe.py`), Zipfian cache simulator (`cache_sim.py`), experiment runner (`run_experiments.py`), Windows memory inspector (`sysmem_win.py`), dependency setup (`setup_llama.py`) |
| [`third_party/patches/`](third_party/patches/) | Minimal surgical `37+/6-` patch (`llama_tierflow_v1.patch`) applied on top of pinned `llama.cpp` commit `3ba1ce0e` |
| [`docs/`](docs/) | [`CLAIMS.md`](docs/CLAIMS.md), [`INTEGRATION.md`](docs/INTEGRATION.md), [`PINNED.md`](docs/PINNED.md), [`UPSTREAM_STATE.md`](docs/UPSTREAM_STATE.md) |
| [`results/`](results/) | Complete reproducible Phase 0–8 experiment JSON/JSONL records and `PREREG.md` pre-registrations |

---

## Quick Start (Build & Test)

### 1. Bootstrap & Build Pinned `llama.cpp`
```powershell
python tools/setup_llama.py
```

### 2. Build TierFlow C++20 Runtime & Run Hardware-Safe Unit Tests
```powershell
cmake -S runtime -B runtime/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build runtime/build --config Release
.\runtime\build\tf_tests.exe
```

### 3. Pack Any GGUF Into a 4096-Aligned `.tfpack` & Run Zero-`mmap` Inference
```powershell
# Pack and SHA-256 verify round-trip identity
python tools/pack.py --input path\to\model.gguf --out-dir packed_models --verify

# Stream under a strict RAM budget
.\runtime\build\tf_runner.exe --model path\to\model.gguf --tfpack packed_models\model.tfpack --ram-mb 600 --prefetch 1 --spec adaptive --prompt "Explain KV caching cleanly:" -n 64
```

### 4. Reproduce Benchmarks (`p0`–`p8`)
```powershell
python tools/run_experiments.py p6
```

---

## Documentation
* [Claim Ledger (`docs/CLAIMS.md`)](docs/CLAIMS.md) — Verified C1–C14 quantitative evidence, controls, and negative findings
* [Final Technical Report (`FINAL_REPORT.md`)](FINAL_REPORT.md) — Full Phase 0–8 methodology, thermal curves, and analysis
* [Implementation Specification (`GEMINI_IMPLEMENTATION_SPEC.md`)](GEMINI_IMPLEMENTATION_SPEC.md) — Original phase contract
* [Integration Notes (`docs/INTEGRATION.md`)](docs/INTEGRATION.md) — `llama.cpp` callback and `llama_moe_cache` architecture

