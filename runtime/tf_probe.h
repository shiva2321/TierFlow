#pragma once

#include <cstdint>
#include <string>

namespace tierflow {

struct HardwareProfile {
    double bw_vram_gbps            = 329.66; // RTX 3060 measured copy traffic
    double bw_ram_copy_gbps        = 32.62;  // DDR5 copy traffic
    double bw_ram_read_gbps        = 48.24;  // DDR5 pure read
    double bw_pcie_h2d_gbps        = 26.52;  // Pinned H2D
    double bw_ssd_burst_gbps       = 4.64;   // Unbuffered NVMe low-QD burst
    double bw_ssd_sustained_gbps   = 1.55;   // Steady-state NVMe sustained (post-pSLC throttle, low-QD)
    double cpu_gemv_effective_gbps = 43.4;   // Multi-thread CPU streaming matvec bandwidth
    double gpu_gemv_effective_gbps = 280.0;  // Effective GPU streaming matvec bandwidth
    double pcie_launch_overhead_ms = 0.10;   // Fixed DMA + kernel launch latency floor
    double ssd_obj_overhead_ms     = 0.20;   // Per-streamed-object IOCP/callback synchronization floor
    bool   probed_live             = false;
};

// Measures live host RAM read/copy bandwidth using C++20 barrier-synchronized threads
// (excluding OS thread-spawn time) and calibrates NVMe low-QD burst + sustained post-pSLC
// bandwidth from `optional_nvme_probe_file` when provided.
HardwareProfile run_tf_probe(const std::string& optional_nvme_probe_file = "",
                             int n_threads_hint = 0,
                             int ssd_qd_hint = 2);

} // namespace tierflow
