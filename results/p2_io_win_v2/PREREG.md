# Pre-Registration: `p2_io_win_v2` (Phase 2 Per-Second Timeline & Post-Write Cooldown Isolation)

Date: 2026-11-12 (UTC)
Experiment ID: `p2_io_win_v2` (follows `p2_io_win` after observing unexpected `2045.54 MiB` `system_cache` growth and `2.65 GB/s` sustained throughput in `p2_io_win`)

## 1. Hypotheses Under Test
1. **Post-write SSD settle:** In `p2_io_win`, `raw_win32_seq` immediately followed the 2 GiB unbuffered write (`4.69 GB/s` vs `6.91–7.31 GB/s` in subsequent tests). Adding a 3s settle wait after shard creation removes SLC-writeback/GC interference from the raw baseline.
2. **Cache-growth isolation:** In `p2_io_win`, `buffered_os_control` ran *before* `sustained_60s`, and `sustained_60s` saw `+2045 MiB` `system_cache` / `standby_normal` growth and `2.65 GB/s` mean read rate. Moving `buffered_os_control` to the **end** of the suite and logging **per-second timeline** `(gbps_1s, pdh_disk_gbps, delta_sys_cache_mib, delta_standby_mib)` over 60s isolates whether `tf_io` itself triggers any OS cache growth or whether an asynchronous OS/Defender background scan or thermal throttling occurred.

## 2. Frozen Protocol & Exit Criteria
- Identical hardware, 2 GiB shard on `E:`, `reps = 20`, `2` warmup passes, 60.0s sustained run (`>= 50 GB` 1 MiB random expert reads at QD=32).
- `tf_io_win` 4 MiB seq QD=32 mean GB/s within **20%** of raw Win32 overlapped IOCP baseline (`>= 0.80×`).
- During the 60s sustained unbuffered run, OS `system_cache` delta attributable to `tf_io` stays near zero (`< 64 MiB` above idle baseline), whereas post-run buffered OS control inflates `system_cache` by ~512 MiB.
- Strict two-priority storm control continues to show lower demand latency than single-FIFO control ($n=20$).

