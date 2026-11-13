# Pre-Registration: `p2_io_win` (Phase 2 C++20 Async Unbuffered I/O Verification)

Date: 2026-11-12 (UTC)
Experiment ID: `p2_io_win`
Phase: Phase 2 (`GEMINI_IMPLEMENTATION_SPEC.md` Section 3 & Section 5)

## 1. Claims & Headline Metrics
1. **Throughput parity:** `tf_io_win` (`FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED` + IOCP) and `tf_io_ioring_win` (Windows 11 IoRing, if runtime-supported) achieve 4 MiB QD=32 sequential read throughput within **20%** of the raw Win32 `ReadFile` IOCP baseline measured in the same binary (`GB/s`, mean ± 95% CI over $n=20$ independent 512 MiB passes).
2. **Sustained zero-cache-growth streaming:** During a 60-second sustained 1 MiB random expert-block read of $\ge 50\text{ GB}$ on the Kingston KC3000 NVMe drive, OS `SystemCache` (`GetPerformanceInfo`) and `Standby Cache Normal Priority Bytes` (`PDH`) do not inflate with streamed shard bytes, while `tf_io` bytes completed reconcile with Windows `\PhysicalDisk(_Total)\Disk Read Bytes/sec`.
3. **Strict two-priority demand protection:** Under a 64-request 1 MiB prefetch storm, demand read latency (`ms`, $n=20$ probes) with `strict_priority = true` (`prefetch_inflight_cap = 4`) is lower than with single-FIFO scheduling (`strict_priority = false`, `prefetch_inflight_cap = 32`).

## 2. Falsifiers
- Claim 1 is false if `tf_io_win` sequential 4 MiB QD=32 throughput falls below **80%** of raw Win32 overlapped `ReadFile` throughput.
- Claim 2 is false if OS `SystemCache` grows monotonically during the 60-second unbuffered run or fails to stay near idle compared to the 512 MiB buffered OS control, or if `< 50 GB` are transferred / `io_errors > 0`.
- Claim 3 is false if strict two-priority demand latency 95% CI overlaps or exceeds the single-FIFO null control under the same prefetch storm.

## 3. Frozen Protocol
- **Test Shard:** `E:\RAM_optimizer\.tmp_p2_io_shard.bin`, `2 GiB` (`2,147,483,648` bytes), populated via `FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH` with non-zero PRNG bytes (zero sparse cluster / OS writeback caching). Deleted after run.
- **Compiler:** MSVC `cl.exe` (C++20, `/O2` Release, Ninja generator).
- **Replicates:** `reps = 20` after `2` warmup passes per timed sweep; 60.0s sustained run sampled every 100 ms.
- **Baselines & Null Controls:**
  1. Raw Win32 `CreateFileW(FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED)` direct IOCP loop (`4 MiB seq QD=32`).
  2. Standard buffered Windows `CreateFileW` (`ReadFile` 512 MiB) control proving OS cache inflation.
  3. Single-FIFO queue null control (`strict_priority = false, prefetch_inflight_cap = 32`) under identical prefetch storm.
- **Unit Tests (`tf_tests.exe`):** Must pass 100% (`tf_governor`, `tf_cache`, `tf_io_win`, `tf_io_ioring_win` including misaligned rejection, out-of-order completion, and post-cancel zero-write invariant) before running benchmark.

