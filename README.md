# TierFlow — Zero-`mmap`, Budget-Bounded Hybrid LLM Inference Engine

TierFlow is a deterministic, memory-governed C++20 inference engine built on top of a surgical patch in pinned `llama.cpp` (`3ba1ce0e`). Instead of relying on OS `mmap()` virtual paging—which silently bloats the Windows/Linux page cache and stalls unpredictably on cold SSD reads—TierFlow explicitly manages every byte across **GPU VRAM (Vulkan) / host DDR5 RAM / unbuffered NVMe SSD** under a strict allocation governor, async `FILE_FLAG_NO_BUFFERING` / `IoRing` (Windows) and syscall `io_uring` `O_DIRECT` (Linux) I/O, a **3-size-class W-TinyLFU + SLRU + Pinned** object cache, sliced per-expert MoE streaming with confidence-gated cross-layer lookahead, disk-backed prefix KV caching, and universal small-draft speculative decoding.

> **Branch Summary:**
> - `main` (`v1`, Phases 0–8): Verified end-to-end on CPU (`Windows 11`, `Ryzen 5 7600`, `KINGSTON SKC3000S1024G` NVMe) with `qwen2.5-1.5b-instruct-q4_k_m.gguf`.
> - `v2` (Phases 9–12): Adds **Vulkan GPU+RAM+SSD hybrid execution**, **3-size-class arena allocation & `--repin` starvation protection**, **per-expert MoE slicing** (`OLMoE-1B-7B` & `Qwen3-30B-A3B`) with **noisy-OR confidence-gated EAM lookahead**, **universal small-draft speculative decoding** (`Qwen3-0.6B` on `Qwen3-30B-A3B`, $k \le 4$ default + `--spec-tree` opt-in), and **Linux `io_uring` + POSIX KV spill parity**.

---

## Architecture Overview

```
+-------------------------------------------------------------------------+
|                         TierFlow v2 Runtime                             |
|                                                                         |
|  [ tf_probe ] ---> HardwareProfile (256 MiB DRAM / FMA GEMV, low-QD     |
|        |           NVMe, Vulkan VRAM & PCIe H2D crossover)              |
|        v                                                                |
|  [ tf_planner ] -> 3^N / heuristic placement + 3-size-class arenas      |
|        |           under [ tf_governor ] hard VRAM + RAM byte ceilings  |
|        v                                                                |
|  [ tf_llama_stream ] (via surgical llama.cpp cb_eval / tensor hooks)    |
|     |-- VRAM Tier: Attention, Routers, KV Cache, Hot Weights (Vulkan)   |
|     |-- 3-Class RAM Arena: W-TinyLFU + SLRU + Pinned [ tf_cache ]       |
|     |-- MoE Slicer & Prefetch: Dual-priority queue (Demand > Prefetch)  |
|     |   '-- Router-ready Gate -> Up+Down (98%+ hit) + EAM (>=0.65 conf) |
|     |-- KV Store [ tf_kv ]: f16 / q8_0 / opt-in q8_q4 + Disk Prefix     |
|     '-- Universal Draft [ tf_draft + tf_spec ]: Any family GGUF (k<=4)  |
|        |                                                                |
|        v                                                                |
|  [ tf_io ] ---> 4096B-aligned .tfpack via Win32 IOCP / IoRing /         |
|                 Linux io_uring O_DIRECT + registered buffers            |
+-------------------------------------------------------------------------+
```

### How TierFlow Differs From Stock `mmap` Runtimes

| Property | Stock `llama.cpp` (`mmap`) | TierFlow `v2` |
| :--- | :--- | :--- |
| **OS Page Cache Pollution** | Inflates Windows `SystemCache` (`+4.2 GiB` on OLMoE, `+8.5 GiB` on Qwen3-30B) & Linux PageCache | **0 MiB** OS cache bloat in unbuffered streaming (`FILE_FLAG_NO_BUFFERING` / `IoRing` / `io_uring O_DIRECT`, verified down to `0 B` in WSL2 rand QD=32) |
| **System RAM Footprint (`Qwen3-30B-A3B`)** | `+14,689 MB` system commit delta (`vulkan24`) | **`+6,011 MB`** system commit delta (`-59.1%`) under strict `6,144 MB` RAM + `2,048 MB` VRAM governors |
| **I/O Scheduling** | Implicit OS page faults; prefetches contend directly with demand reads | **Strict 2-priority queue**: demand reads (`0.708 ms`) stay **13.09× faster** than single-FIFO under prefetch storms (`9.269 ms`) |
| **MoE Streaming Granularity** | Reads monolithic layer/tensor blobs (`~100%` of evicted tensor) | **`tools/pack.py --slice-moe-experts`**: co-packs each `(layer, expert)` `{gate, up, down}` triple as one 4096B-aligned object |
| **Default Quality Contract** | Silent greedy divergence with CPU weight repacking (`8/20` exact) | **`exact` deterministic CPU path** (`20/20`, `max_logit_diff = 0.0` on `Qwen2.5-1.5B` and `OLMoE-1B-7B` `CPU`); opt-in lossy modes isolated |

---

## Verified Measured Results (Phases 0–12)

All Windows benchmarks were executed on `Windows 11` (`AMD Ryzen 5 7600`, `32 GiB DDR5`, `NVIDIA RTX 3060 12GB` Vulkan, `KINGSTON SKC3000S1024G` PCIe 4.0 NVMe). Linux I/O and unit tests were executed inside `WSL2` (`6.6.87.2-microsoft-standard-WSL2`, `g++ 13.3.0`). Full JSON/JSONL artifacts live in [`results/`](results/) and the audited ledger lives in [`docs/CLAIMS.md`](docs/CLAIMS.md).

### 1. Core Async Unbuffered I/O (`p2_io_win_v2` & Phase 12 Linux `io_uring`)
* **Windows Win32 IOCP / IoRing (`p2_io_win_v2`):** **`7.312 ± 0.024 GB/s`** (`IOCP` 4 MiB seq QD=32) and **`7.331 ± 0.006 GB/s`** (`IoRing`, **106.6%** of raw Win32 IOCP), **`7.328 ± 0.006 GB/s`** (1 MiB rand QD=32). Sustained 60s unbuffered read transferred **`178.66 GB`** (`99.88%` PDH PhysicalDisk match, `0` I/O errors, `-60.74 MiB` SystemCache delta vs idle) while a 512 MiB buffered control added `+511.94 MiB`.
* **Two-Priority Queue Protection:** Under a 64-req prefetch storm, 1 MiB demand latency dropped **`13.09×`** (**`0.708 ± 0.024 ms`** vs **`9.269 ± 0.112 ms`** FIFO null).
* **Linux `io_uring` + POSIX Spill (`Phase 12`):** Raw kernel syscall `linux_io_uring_direct_reg` (`O_DIRECT` + `IORING_REGISTER_BUFFERS`) + POSIX `O_DIRECT`/`pwrite`/`pread` `tf_kv_store` pass **100%** of unit tests. Inside WSL2 VHDX `ext4`, `tf_io_bench` measured **`6.580 ± 0.790 GB/s`** (`max 7.174`, 4 MiB seq QD=32, `/proc/meminfo` PageCache delta = **`0 B`**) and **`6.742 ± 0.472 GB/s`** (`max 7.073`, 1 MiB rand QD=32, PageCache delta = **`0 B`**).

### 2. Dense Streaming, Vulkan Hybrid, Prefix Cache & Speculative Decoding (Phases 3–9)
* **Zero-`mmap` CPU Exactness (`p3_e2e_stream`, `Qwen2.5-1.5B`):** `tierflow_ram1400` (**`43.23 ± 0.84 tok/s`**) and `tierflow_stream600` (**`1.66 ± 0.69 tok/s`**, `99.992%` prefetch acc) matched `stock_mmap_norepack` **`20 / 20`** (`max_logit_diff = 0.0`).
* **Vulkan Hybrid VRAM+Pool (`p9_vulkan_pool`, `Qwen2.5-1.5B`):** `tierflow_vulkan_full` reached **`167.551 ± 12.042 tok/s`** (**`20 / 20` exact** vs `stock_mmap_vulkan` `134.802 ± 3.803 tok/s`, `0.0 MB` file cache growth, `-38.5%` system RAM delta). Note: `stock_mmap_vulkan` vs `stock_mmap_cpu` itself matches `11/20` due to Vulkan FP16 vs CPU accumulation differences.
* **Lookahead vs. Shuffled Null (`p5_prefetch_repin`, 600 MiB, 60s soak):** `lookahead_d1_wtinylfu` (**`1.423 ± 0.598 tok/s`**, `99.970%` acc) beat `prefetch_shuffled_null` (**`0.508 ± 0.269 tok/s`**, `0.018%` acc, `1.90×` extra SSD reads).
* **Dense Cold-Stream Speculative Decoding (`p6_spec`, 600 MiB):** `--spec adaptive` accelerated streaming **`2.94×` overall** (**`2.085 ± 0.710`** vs **`0.710 ± 0.486 tok/s`**), **`9.08×` on `code_overlap`** (`2.533 tok/s`), cutting SSD reads **`-43.3%`**. Passing `k <= 4` (`fixed4`) kept **`20 / 20` exact** greedy parity while unconstrained `k <= 8` (`adaptive`) dropped to `6/20` on RAM.
* **Prefix Caching & KV Compression (`p7_kv`, `p8_lossy_kv_v4`):** Disk prefix cache hit `20 / 20` exact (`0.0` logit diff), cutting warm TTFT **`12.36×`** at 2K (`287.99 -> 23.30 ms`) and **`782.9×`** at 32K (`67,136.5 -> 85.76 ms`). `q8_0` fit 32K KV inside `1700 MiB`, and opt-in `--kv-type q8_q4` reduced KV bytes `-59.4%`.

### 3. Phase 10: Sliced MoE Streaming After 60s NVMe Heat-Soak (`p10_moe_stream`)
Heat-soak: `60.03 s` unbuffered read of **`99.41 GB` (`1.656 GB/s`)** prior to timed conditions:

| Model / Condition | `decode_tok_s` | RAM / VRAM Gov | Sys Δ / Cache Δ | SSD Read / Prefetch Acc | Exact Greedy Parity |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`OLMoE-1B-7B` `stock_mmap_cpu`** | `42.073 ± 1.255` | `0 / 0 MB` | `4604 / 4210 MB` | `0 GB` | Reference (CPU) |
| **`OLMoE-1B-7B` `tierflow_cpu`** (`1.5 GiB`) | **`3.114 ± 0.125`** | `1609 / 0 MB` | `1582 / 0.0 MB` | `201.6 GB` / **`98.8%`** | **`20 / 20`** (`0.0000` diff vs CPU) |
| **`OLMoE-1B-7B` `lookahead`** (`vram=768M, ram=1152M`) | **`2.362 ± 0.186`** | `1207 / 501.6 MB` | `1207 / 0.0 MB` | `223.1 GB` / **`98.15%`** (EAM `71.3%`) | `13 / 20` vs `stock_vulkan` (`0.488` Vulkan FP diff) |
| **`OLMoE-1B-7B` `off`** (no prefetch) | `1.919 ± 0.128` | `1207 / 501.6 MB` | `1186 / 29 MB` | `233.9 GB` | `13 / 20` vs `stock_vulkan` |
| **`OLMoE-1B-7B` `shuffled`** (null control) | `1.042 ± 0.012` | `1207 / 501.6 MB` | `1324 / 255 MB` | `411.1 GB` / `10.62%` | `13 / 20` vs `stock_vulkan` |
| **`Qwen3-30B-A3B` `stock_mmap_vulkan24`** | `23.448 ± 3.134` | `0 / 0 MB` | `14689 / 8510 MB` | `0 GB` | Reference (Vulkan24) |
| **`Qwen3-30B-A3B` `lookahead`** (`vram=2G, ram=6G`) | **`2.636 ± 0.211`** | `6442 / 1144 MB` | **`6011 / 11.5 MB`** | `327.2 GB` / **`98.19%`** (EAM `60.8%`) | `15 / 20` vs `stock_vulkan24` (`top1=19/20`) |
| **`Qwen3-30B-A3B` `off`** | `2.360 ± 0.178` | `6442 / 1144 MB` | `6230 / 25 MB` | `321.6 GB` | `15 / 20` vs `stock_vulkan24` |
| **`Qwen3-30B-A3B` `shuffled`** (null control) | `0.856 ± 0.087` | `6442 / 1144 MB` | `6883 / 361 MB` | `792.0 GB` / `5.70%` | `15 / 20` vs `stock_vulkan24` |

*(Causal validation: `lookahead` improves streaming throughput **+23.1%** on OLMoE and **+11.7%** on Qwen3-30B over `off`, and runs **2.27×** and **3.08×** faster than `shuffled` null controls, while all 3 prefetch variants produce identical token outputs.)*

### 4. Phase 11: Universal Small Draft Model (`Qwen3-0.6B` on `Qwen3-30B-A3B`, `p11_draft_moe`)
Heat-soak: `60.03 s` unbuffered read of **`120.04 GB` (`2.000 GB/s`)**:
* **Exact Stochastic Rejection Sampling (`temp = 0.8`, `500` draws × `4` tokens = `2,000` tokens/arm):** With `accept_rate = 62.20%` (`k_mean = 1.56`), `spec_fixed4` passed **5 / 5** Pearson $\chi^2$ equivalence tests ($\alpha = 0.01$): `first_token` ($p = 0.766$), `second_token` ($p = 0.982$), `third_token` ($p = 0.844$), `bigram_12` ($p = 0.805$), and `bigram_23` ($p = 0.705$).
* **Honest Negative/Mixed MoE Speculation Result (Why Wide Trees Hurt Single-SSD MoE):** On 128-expert top-8 `Qwen3-30B-A3B` (`vram=2G, ram=6G`), batching `k + 1` tokens widens the distinct active expert union per layer (`8 -> ~18.25 -> ~28.12`), increasing SSD reads per token monotonically:
  * `spec off` ($k = 0$): **`2.861 ± 0.219 tok/s`**, **`494.6 MB/tok`** SSD read
  * `spec adaptive` ($k_{\text{mean}} = 1.81$, `77.3%` accept): **`2.700 ± 0.191 tok/s`**, **`524.1 MB/tok`** (`+6.0%` SSD)
  * `spec fixed4` ($k_{\text{mean}} = 3.72$, `62.5%` accept): **`2.351 ± 0.228 tok/s`**, **`580.5 MB/tok`** (`+17.4%` SSD)
  * `--spec-tree 1` `k = 8` opt-in ablation ($k_{\text{mean}} = 6.87$, `55.2%` accept): **`2.023 ± 0.274 tok/s`**, **`658.0 MB/tok`** (`+33.0%` SSD)
* Greedy parity (`k <= 4`) achieved `17 / 20` exact (`top1 = 20/20`, `max_first_logit_diff = 0.0`) on CPU (`4.726` vs `5.498 tok/s`) and `15 / 20` on Vulkan (`top1 = 20/20`).

### Hardware Physics & Known Constraints
* **Single-SSD Heat-Soaked Ceiling:** Sustained NVMe streaming thermally derates single-drive bandwidth (`7.35 GB/s` burst $\to$ `2.8 GB/s` $\to$ `~1.66–2.00 GB/s` 60s soak $\to$ `0.293 GB/s` 50-min continuous saturation), capping out-of-core single-SSD MoE streaming at `~2.4–3.1 tok/s` (far below the aspirational 10+ tok/s out-of-core target when `~68%` of experts miss RAM/VRAM).
* **CPU vs. Vulkan FP16 Determinism:** Hybrid CPU+Vulkan splits differ on close greedy ties (`13–15 / 20` vs pure Vulkan, `11/20` stock Vulkan vs stock CPU) due to FP16 vs INT8/FMA summation order across backends, while pure CPU TierFlow matches `stock_mmap` CPU `20 / 20` bit-for-bit (`0.0` logit diff).

---

## Hardware & OS Safety Guarantees

Every boundary in `tf_io`, `tf_cache`, `tf_governor`, and `tf_kv` is verified by `tf_tests` (`Win32` and `WSL2 Linux`):
1. **Strict 4096-Byte Alignment Rejection:** `tf_io` rejects misaligned file offsets, transfer lengths, or host pointers before issuing `FILE_FLAG_NO_BUFFERING` / `O_DIRECT` I/O calls.
2. **Post-Cancel DMA Retirement & Refcount Guards:** On Windows IOCP/IoRing and Linux `io_uring`, `cancel(tag)` waits synchronously for kernel CQE retirement so OS DMA never mutates reused buffers (`Verified N cancelled prefetch buffers untouched after cancel ack`).
3. **Hard Governor Ceilings + `--repin` Starvation Guard:** Allocation requests fail cleanly before over-committing RAM/VRAM; `tf_llama_stream` blocks `--repin` if unpinned Probation slots would drop below safe I/O concurrency bounds.
4. **Auto-Cleaning Spill Files:** Windows `FILE_FLAG_DELETE_ON_CLOSE` and POSIX unlink-on-open (`tf_kv.cpp`) prevent orphaned temp SSD spill artifacts on process exit or crash.

---

## Repository Layout

| Path | Description |
| :--- | :--- |
| [`runtime/`](runtime/) | C++20 runtime: `tf_governor`, `tf_cache`, `tf_kv`, `tf_manifest`, `tf_probe`, `tf_planner`, `tf_spec`, `tf_draft.{h,cpp}`, `tf_io_{win,ioring_win,linux}`, `tf_llama_stream.{h,cpp}`, `tf_runner.cpp`, `tf_io_bench.cpp`, `tests/test_main.cpp` |
| [`tools/`](tools/) | Python utilities: `.tfpack` packer w/ `--slice-moe-experts` (`pack.py`), PCIe/SSD prober (`hw_probe.py`), Zipfian simulator (`cache_sim.py`), experiment suite `p0`–`p11` (`run_experiments.py`), Windows PDH/memory inspector (`sysmem_win.py`), pinned `llama.cpp` builder (`setup_llama.py`) |
| [`third_party/patches/`](third_party/patches/) | Minimal surgical `llama.cpp` patches (`llama_tierflow_v1.patch`, `llama_tierflow_v2.patch` incl. Direct I/O, host-weight buffer hooks, and CPU tiled `mul_mat_id` `src0_cur` streaming support) |
| [`docs/`](docs/) | [`CLAIMS.md`](docs/CLAIMS.md), [`TIERFLOW_V2_PLAN.md`](docs/TIERFLOW_V2_PLAN.md), [`INTEGRATION.md`](docs/INTEGRATION.md), [`PINNED.md`](docs/PINNED.md) |
| [`results/`](results/) | Complete raw JSON/JSONL records and `PREREG.md` files for Phases 0–11 |

---

## Quick Start (Build, Pack & Run)

### 1. Build Pinned `llama.cpp` & TierFlow (Windows + Vulkan SDK)
```powershell
python tools/setup_llama.py
cmake -S runtime -B runtime/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build runtime/build --config Release --parallel
.\runtime\build\tf_tests.exe
```

### 2. Build & Verify Linux `io_uring` + POSIX Spill Parity (WSL2 / Linux `g++ 13+`)
```bash
cd runtime
g++ -std=c++20 -O2 -Wall -Wextra -I. -pthread tf_governor.cpp tf_cache.cpp tf_kv.cpp tf_manifest.cpp tf_probe.cpp tf_planner.cpp tf_spec.cpp tf_io_linux.cpp tests/test_main.cpp -o /tmp/tf_tests_linux
g++ -std=c++20 -O2 -Wall -Wextra -I. -pthread tf_governor.cpp tf_cache.cpp tf_kv.cpp tf_manifest.cpp tf_probe.cpp tf_planner.cpp tf_spec.cpp tf_io_linux.cpp tf_io_bench.cpp -o /tmp/tf_io_bench_linux
/tmp/tf_tests_linux && /tmp/tf_io_bench_linux --reps 10
```

### 3. Pack a Dense or MoE GGUF & Run Budget-Bounded Inference
```powershell
# Pack MoE model with per-expert 4096B slicing and SHA-256 verification
python tools/pack.py --input path\to\MoE.gguf --out-dir packed_models --slice-moe-experts --verify

# Hybrid Vulkan + RAM + NVMe MoE inference under strict budgets
.\runtime\build\tf_runner.exe --mode tierflow --gguf path\to\MoE.gguf --manifest packed_models\MoE.moe.manifest.json `
  --n-gpu-layers 99 --vram-mb 2048 --ram-budget-mb 6144 --prefetch lookahead --repin 1 --prompt "Explain MoE routing:" --n-predict 64

# Optional universal small draft model (k <= 4 default; add --spec-tree 1 for k > 4 experiments)
.\runtime\build\tf_runner.exe --mode tierflow --gguf path\to\Qwen3-30B-A3B.gguf --manifest packed_models\Qwen3-30B-A3B.moe.manifest.json `
  --draft-model path\to\Qwen3-0.6B.gguf --n-gpu-layers 99 --vram-mb 2048 --ram-budget-mb 6144 --spec adaptive --spec-max-k 4 --n-predict 64
```

### 4. Reproduce Benchmarks & Self-Tests
```powershell
python tools/pack.py --self-test
python tools/cache_sim.py --self-test
python tools/sysmem_win.py --self-test
python tools/run_experiments.py p10   # Heat-soaked MoE streaming (OLMoE + Qwen3-30B)
python tools/run_experiments.py p11   # Heat-soaked universal draft MoE + chi-square parity
```
