#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tierflow {

constexpr size_t   kNumArenaClasses   = 3;
constexpr uint64_t kSmallClassMaxB    = 512ULL << 10; // 512 KiB
constexpr uint64_t kMediumClassMaxB   = 4ULL << 20;   // 4 MiB

struct ArenaClassSpec {
    size_t   object_count     = 0;
    uint64_t max_aligned_size = 0;
    size_t   stride_bytes     = 0;
    size_t   n_slots          = 0;
    uint64_t total_bytes      = 0;
    uint64_t total_obj_bytes  = 0;
};

struct ArenaLayoutPlan {
    std::array<ArenaClassSpec, kNumArenaClasses> classes{};
    uint64_t total_bytes  = 0;
    size_t   total_slots  = 0;
    bool     feasible     = true;
};

inline uint64_t align_up_4096(uint64_t v) {
    return (v + 4095ULL) & ~4095ULL;
}

/// Assign a 4 KiB-aligned object size into one of three fixed-stride size classes:
/// Class 0: <= 512 KiB, Class 1: <= 4 MiB, Class 2: > 4 MiB.
inline int assign_size_class(uint64_t page_aligned_bytes) {
    if (page_aligned_bytes <= kSmallClassMaxB) return 0;
    if (page_aligned_bytes <= kMediumClassMaxB) return 1;
    return 2;
}

/// Partition `ram_budget_bytes` across non-empty size classes cleanly.
/// Empty size classes receive 0 slots and 0 bytes; non-empty classes receive at least
/// `min_slots` slots each (if budget permits) plus proportional extra slots capped by `max_slots_per_class`.
inline ArenaLayoutPlan compute_arena_layout(const std::vector<uint64_t>& streamed_aligned_sizes,
                                            uint64_t                     ram_budget_bytes,
                                            size_t                       max_slots_per_class,
                                            size_t                       min_slots = 4) {
    ArenaLayoutPlan plan{};
    if (streamed_aligned_sizes.empty()) {
        return plan;
    }

    for (uint64_t sz : streamed_aligned_sizes) {
        const uint64_t asz = align_up_4096(std::max<uint64_t>(4096ULL, sz));
        const int c        = assign_size_class(asz);
        auto& sp           = plan.classes[static_cast<size_t>(c)];
        ++sp.object_count;
        sp.total_obj_bytes += asz;
        if (asz > sp.max_aligned_size) sp.max_aligned_size = asz;
    }

    uint64_t min_required = 0;
    uint64_t active_weight = 0;
    for (auto& sp : plan.classes) {
        if (sp.object_count == 0) continue;
        sp.stride_bytes = static_cast<size_t>(sp.max_aligned_size);
        min_required   += static_cast<uint64_t>(min_slots) * sp.stride_bytes;
        active_weight  += sp.total_obj_bytes;
    }

    if (min_required > ram_budget_bytes) {
        plan.feasible = false;
        return plan;
    }

    // Reserve minimum safe working slots per non-empty class first
    uint64_t spent = 0;
    for (auto& sp : plan.classes) {
        if (sp.object_count == 0) continue;
        sp.n_slots      = min_slots;
        sp.total_bytes  = static_cast<uint64_t>(sp.n_slots) * sp.stride_bytes;
        spent          += sp.total_bytes;
    }

    // Distribute leftover budget proportionally across classes needing more than min_slots
    const size_t cap = std::max<size_t>(min_slots, max_slots_per_class);
    uint64_t leftover = (ram_budget_bytes > spent) ? (ram_budget_bytes - spent) : 0ULL;
    for (int c = 0; c < static_cast<int>(kNumArenaClasses); ++c) {
        auto& sp = plan.classes[static_cast<size_t>(c)];
        if (sp.object_count == 0 || sp.stride_bytes == 0) continue;
        const size_t target_cap = std::min<size_t>(cap, sp.object_count);
        if (sp.n_slots >= target_cap) continue;
        const double frac = (active_weight > 0)
            ? (static_cast<double>(sp.total_obj_bytes) / static_cast<double>(active_weight))
            : 1.0;
        const uint64_t share = static_cast<uint64_t>(static_cast<double>(leftover) * frac);
        const size_t extra   = static_cast<size_t>(share / sp.stride_bytes);
        const size_t add     = std::min<size_t>(extra, target_cap - sp.n_slots);
        sp.n_slots     += add;
        sp.total_bytes  = static_cast<uint64_t>(sp.n_slots) * sp.stride_bytes;
        spent          += static_cast<uint64_t>(add) * sp.stride_bytes;
    }

    // Opportunistically allocate any small remainder slots where full stride still fits
    bool added_any = true;
    while (added_any) {
        added_any = false;
        for (int c = 0; c < static_cast<int>(kNumArenaClasses); ++c) {
            auto& sp = plan.classes[static_cast<size_t>(c)];
            if (sp.object_count == 0 || sp.stride_bytes == 0) continue;
            const size_t target_cap = std::min<size_t>(cap, sp.object_count);
            if (sp.n_slots < target_cap && (spent + sp.stride_bytes) <= ram_budget_bytes) {
                ++sp.n_slots;
                sp.total_bytes += sp.stride_bytes;
                spent          += sp.stride_bytes;
                added_any       = true;
            }
        }
    }

    plan.total_bytes = spent;
    for (const auto& sp : plan.classes) {
        plan.total_slots += sp.n_slots;
    }
    return plan;
}

/// Prevent `--repin` from pinning so many slots that prefetch + active eval deadlock.
/// Requires `arena_cap >= 12` and guarantees at least `(prefetch_inflight_cap + 4)` unpinned slots remain.
inline bool can_repin_slot(size_t arena_cap, size_t already_repinned, size_t prefetch_inflight_cap) {
    if (arena_cap < 12) return false;
    const size_t max_ratio_pins = std::max<size_t>(1, arena_cap / 3);
    if (already_repinned >= max_ratio_pins) return false;
    const size_t min_free_reserve = prefetch_inflight_cap + 4;
    if (arena_cap <= already_repinned + 1 + min_free_reserve) return false;
    return true;
}

/// Dynamic prefetch queue depth scaled by available unpinned cache slots (capped at [1, 16]).
inline size_t compute_dynamic_prefetch_cap(size_t unpinned_slots) {
    if (unpinned_slots < 4) return 0;
    return std::clamp<size_t>(unpinned_slots / 2, 1, 16);
}

/// Update rolling EWMA storage bandwidth (GB/s) from an observed asynchronous DMA completion.
inline double update_bw_ewma(double prev_gbps, uint64_t bytes, double elapsed_sec, double decay = 0.85) {
    if (bytes == 0 || elapsed_sec <= 1e-6) return prev_gbps;
    const double sample_gbps = (static_cast<double>(bytes) / 1e9) / elapsed_sec;
    if (prev_gbps <= 0.0) return sample_gbps;
    return decay * prev_gbps + (1.0 - decay) * sample_gbps;
}

} // namespace tierflow

