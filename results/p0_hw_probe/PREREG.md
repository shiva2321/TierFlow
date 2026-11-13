# Pre-Registration: `p0_hw_probe`

Date: 2026-11-12 (UTC)

## 1. Claim & Headline Metrics
- **Claim 1 (Unbuffered Direct I/O bypasses OS page cache):** Reading a 256 MiB weight shard on Windows 11 with `FILE_FLAG_NO_BUFFERING` into `VirtualAlloc` page-aligned memory achieves multi-GB/s NVMe throughput while keeping `delta_system_cache_bytes` significantly below the 256 MiB file size (whereas buffered I/O grows or hits the OS page cache).
- **Claim 2 (Fiddler/KTransformers CPU-vs-PCIe crossover):** For a single Qwen3-30B-A3B expert MLP (`hidden=2048, inter=768`, 9.437 MiB in FP16) residing in host RAM, computing the expert in-place on the CPU and transferring only the output activation vector to the GPU (`cpu_inplace_ms_mean`) is faster than transferring the 9.437 MiB expert weights over PCIe to the GPU and computing on the GPU (`pcie_transfer_plus_gpu_ms_mean`) at batch size $B = 1$.

## 2. Falsifiers
- **Falsifier 1:** `nvme_seq_4mib_unbuffered.delta_system_cache_bytes >= 128 MiB` (indicating the OS page cache was not bypassed) OR unbuffered throughput is `< 0.5 GB/s`.
- **Falsifier 2:** At `batch_1`, `cpu_inplace_ms_mean >= pcie_transfer_plus_gpu_ms_mean`.

## 3. Frozen Protocol
- File size: 256 MiB temporary binary shard on `E:\RAM_optimizer\results\p0_hw_probe\_nvme_probe_temp.bin`.
- Block sizes: 4 MiB sequential (64 reads) and 1 MiB random permutation (256 reads, RNG seed = 42).
- Compute/memory repeats: $n = 20$ replicates per measurement.
- Baselines / Controls:
  1. Buffered OS read (`open(..., "rb")`) on the exact same 256 MiB file.
  2. Pageable vs. Pinned PCIe Host-to-Device transfer.
  3. GPU-resident expert execution (upper-bound speed control when expert is already cached in VRAM).

