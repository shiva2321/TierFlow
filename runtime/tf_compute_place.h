#pragma once

#include "tf_probe.h"

#include <algorithm>
#include <cstdint>

namespace tierflow {

enum class ComputeTarget : uint8_t {
    CPU_IN_PLACE = 0,
    GPU_UPLOAD   = 1,
    GPU_RESIDENT = 2,
};

struct ComputePlacementDecision {
    ComputeTarget target   = ComputeTarget::CPU_IN_PLACE;
    double        t_cpu_ms = 0.0;
    double        t_gpu_ms = 0.0;
};

// Section 8.6: For a RAM-resident expert (or VRAM-resident hit), chooses whether
// to compute in-place on CPU or upload over PCIe to GPU based on batch size B.
inline ComputePlacementDecision choose_compute_target(
    uint64_t weight_bytes,
    uint64_t flops_per_token,
    uint32_t batch_tokens,
    bool already_in_vram,
    const HardwareProfile& probe) {
    ComputePlacementDecision d{};
    if (already_in_vram) {
        d.target   = ComputeTarget::GPU_RESIDENT;
        d.t_gpu_ms = 1e3 * ((static_cast<double>(weight_bytes) / (std::max(1.0, probe.gpu_gemv_effective_gbps) * 1e9)) +
                            static_cast<double>(flops_per_token * std::max(1u, batch_tokens)) / (std::max(1.0, probe.gpu_gemv_effective_gbps) * 1e9 * 10.0));
        d.t_cpu_ms = d.t_gpu_ms * 10.0;
        return d;
    }

    const double b = static_cast<double>(std::max(1u, batch_tokens));
    // CPU compute time grows near-linearly with batch for moderate/high B while RAM read stays bounded
    const double t_ram_read = static_cast<double>(weight_bytes) / (std::max(1.0, probe.bw_ram_read_gbps) * 1e9);
    const double t_cpu_comp = b * (static_cast<double>(weight_bytes) / (std::max(1.0, probe.cpu_gemv_effective_gbps) * 1e9));
    d.t_cpu_ms = 1e3 * std::max(t_ram_read, t_cpu_comp);

    // GPU requires fixed PCIe weight transfer + DMA/kernel launch latency floor
    const double t_pcie = static_cast<double>(weight_bytes) / (std::max(1.0, probe.bw_pcie_h2d_gbps) * 1e9);
    const double t_gpu_comp = static_cast<double>(weight_bytes) / (std::max(1.0, probe.gpu_gemv_effective_gbps) * 1e9) +
                              b * (static_cast<double>(flops_per_token) / 1e12);
    d.t_gpu_ms = probe.pcie_launch_overhead_ms + 1e3 * (t_pcie + t_gpu_comp);

    d.target = (d.t_cpu_ms <= d.t_gpu_ms) ? ComputeTarget::CPU_IN_PLACE : ComputeTarget::GPU_UPLOAD;
    return d;
}

} // namespace tierflow
