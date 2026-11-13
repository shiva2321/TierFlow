#include "tf_planner.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace tierflow {

namespace {

inline double tier_sec_per_byte(PlacementTier t, const HardwareProfile& p) {
    // Effective reading cost for active token step (1e9 bytes/GB)
    switch (t) {
        case PlacementTier::VRAM:
            return 1.0 / (std::max(1.0, p.gpu_gemv_effective_gbps) * 1e9);
        case PlacementTier::RAM:
            // In-place CPU streaming read rate
            return 1.0 / (std::max(1.0, p.cpu_gemv_effective_gbps) * 1e9);
        case PlacementTier::SSD_STREAM:
            return 1.0 / (std::max(0.1, p.bw_ssd_sustained_gbps) * 1e9);
    }
    return 1.0;
}

} // namespace

double evaluate_placement_cost_sec(const std::vector<PlannerObjectInput>& objects,
                                   const std::vector<PlacementTier>& assignment,
                                   const HardwareProfile& probe) {
    double total = 0.0;
    for (size_t i = 0; i < objects.size(); ++i) {
        const double f = std::max(0.0, objects[i].freq);
        total += f * static_cast<double>(objects[i].bytes) * tier_sec_per_byte(assignment[i], probe);
        if (assignment[i] == PlacementTier::SSD_STREAM && f > 0.0) {
            total += f * (probe.ssd_obj_overhead_ms * 1e-3);
        }
    }
    return total;
}

PlacementPlan solve_exact_bruteforce(const std::vector<PlannerObjectInput>& objects,
                                     const PlannerBudget& budget,
                                     const HardwareProfile& probe) {
    PlacementPlan plan{};
    const size_t n = objects.size();
    plan.tier_of.assign(n, PlacementTier::SSD_STREAM);

    if (n > 14) {
        plan.feasible = false;
        plan.reason   = "Exact solver only supports <= 14 objects";
        return plan;
    }

    // Sort indices by size/freq descending? Branch-and-bound checks all valid placements cleanly.
    std::vector<PlacementTier> cur(n, PlacementTier::SSD_STREAM);
    std::vector<PlacementTier> best_assign(n, PlacementTier::SSD_STREAM);
    double best_cost = std::numeric_limits<double>::infinity();

    // Suffix lower bound: remaining items at best possible tier cost (VRAM)
    std::vector<double> suffix_lb(n + 1, 0.0);
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        const double f = std::max(0.0, objects[i].freq);
        suffix_lb[i] = suffix_lb[i + 1] +
            f * static_cast<double>(objects[i].bytes) * tier_sec_per_byte(PlacementTier::VRAM, probe);
    }

    auto dfs = [&](auto& self, size_t idx, uint64_t vram_u, uint64_t ram_u, double cur_cost) -> void {
        if (cur_cost + suffix_lb[idx] >= best_cost) return;
        if (idx == n) {
            best_cost   = cur_cost;
            best_assign = cur;
            return;
        }

        const auto& obj = objects[idx];
        const double f  = std::max(0.0, obj.freq);

        // Try VRAM first
        if (vram_u + obj.bytes <= budget.vram_bytes) {
            cur[idx] = PlacementTier::VRAM;
            self(self, idx + 1, vram_u + obj.bytes, ram_u,
                 cur_cost + f * static_cast<double>(obj.bytes) * tier_sec_per_byte(PlacementTier::VRAM, probe));
        }
        // Try RAM
        if (ram_u + obj.bytes <= budget.ram_bytes) {
            cur[idx] = PlacementTier::RAM;
            self(self, idx + 1, vram_u, ram_u + obj.bytes,
                 cur_cost + f * static_cast<double>(obj.bytes) * tier_sec_per_byte(PlacementTier::RAM, probe));
        }
        // Try SSD_STREAM (allowed if expert OR allow_slow_dense_stream)
        if (!obj.always_on || budget.allow_slow_dense_stream) {
            cur[idx] = PlacementTier::SSD_STREAM;
            const double ssd_oh = (f > 0.0) ? (f * probe.ssd_obj_overhead_ms * 1e-3) : 0.0;
            self(self, idx + 1, vram_u, ram_u,
                 cur_cost + ssd_oh + f * static_cast<double>(obj.bytes) * tier_sec_per_byte(PlacementTier::SSD_STREAM, probe));
        }
    };

    dfs(dfs, 0, 0, 0, 0.0);

    if (!std::isfinite(best_cost)) {
        plan.feasible = false;
        plan.reason   = "No feasible placement fits always-on objects in VRAM+RAM budgets";
        return plan;
    }

    plan.feasible = true;
    plan.tier_of  = best_assign;
    double weighted_demand = 0.0, resident_demand = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double w = std::max(0.0, objects[i].freq) * static_cast<double>(objects[i].bytes);
        weighted_demand += w;
        if (best_assign[i] == PlacementTier::VRAM) {
            plan.vram_used_bytes += objects[i].bytes;
            resident_demand += w;
        } else if (best_assign[i] == PlacementTier::RAM) {
            plan.ram_used_bytes += objects[i].bytes;
            resident_demand += w;
        } else {
            plan.ssd_stream_bytes += objects[i].bytes;
        }
    }
    plan.expected_sec_per_tok = best_cost;
    plan.predicted_tok_s      = (best_cost > 0.0) ? (1.0 / best_cost) : 0.0;
    plan.predicted_hit_rate   = (weighted_demand > 0.0) ? (resident_demand / weighted_demand) : 1.0;
    return plan;
}

PlacementPlan build_placement_plan(const std::vector<PlannerObjectInput>& objects,
                                   const PlannerBudget& budget,
                                   const HardwareProfile& probe) {
    if (objects.size() <= 14) {
        return solve_exact_bruteforce(objects, budget, probe);
    }

    PlacementPlan best_plan{};
    best_plan.feasible             = false;
    best_plan.expected_sec_per_tok = std::numeric_limits<double>::infinity();

    std::vector<size_t> always_indices;
    std::vector<size_t> expert_indices;
    for (size_t i = 0; i < objects.size(); ++i) {
        if (objects[i].always_on) always_indices.push_back(i);
        else expert_indices.push_back(i);
    }

    // Place always-on objects fastest-tier-first (largest/highest impact into VRAM first, then RAM)
    std::sort(always_indices.begin(), always_indices.end(), [&](size_t a, size_t b) {
        return objects[a].bytes > objects[b].bytes;
    });

    std::vector<PlacementTier> base_assign(objects.size(), PlacementTier::SSD_STREAM);
    uint64_t base_vram = 0, base_ram = 0;
    for (size_t idx : always_indices) {
        const uint64_t sz = objects[idx].bytes;
        if (base_vram + sz <= budget.vram_bytes) {
            base_assign[idx] = PlacementTier::VRAM;
            base_vram += sz;
        } else if (base_ram + sz <= budget.ram_bytes) {
            base_assign[idx] = PlacementTier::RAM;
            base_ram += sz;
        } else if (budget.allow_slow_dense_stream) {
            base_assign[idx] = PlacementTier::SSD_STREAM;
        } else {
            best_plan.feasible = false;
            best_plan.reason   = "Always-on weights exceed VRAM+RAM budgets (enable allow_slow to stream)";
            return best_plan;
        }
    }

    // Rank dynamic/expert objects by pinning priority (or freq if rank_priority < 0, tie-break smallest size)
    auto prio_of = [&](size_t idx) -> double {
        return (objects[idx].rank_priority >= 0.0) ? objects[idx].rank_priority : objects[idx].freq;
    };
    std::sort(expert_indices.begin(), expert_indices.end(), [&](size_t a, size_t b) {
        const double pa = prio_of(a);
        const double pb = prio_of(b);
        if (pa != pb) return pa > pb;
        return objects[a].bytes < objects[b].bytes;
    });

    // Grid search pin_fraction p in {0.0, 0.1, ..., 0.9} (Section 8.1)
    for (int step = 0; step <= 9; ++step) {
        const double pin_frac = 0.1 * static_cast<double>(step);
        std::vector<PlacementTier> assign = base_assign;
        uint64_t vram_u = base_vram;
        uint64_t ram_u  = base_ram;

        const uint64_t vram_pin_cap = base_vram + static_cast<uint64_t>(pin_frac * static_cast<double>(budget.vram_bytes - base_vram));
        const uint64_t ram_pin_cap  = base_ram  + static_cast<uint64_t>(pin_frac * static_cast<double>(budget.ram_bytes - base_ram));

        for (size_t idx : expert_indices) {
            const uint64_t sz = objects[idx].bytes;
            if (vram_u + sz <= vram_pin_cap) {
                assign[idx] = PlacementTier::VRAM;
                vram_u += sz;
            } else if (ram_u + sz <= ram_pin_cap) {
                assign[idx] = PlacementTier::RAM;
                ram_u += sz;
            }
        }
        // Remaining dynamic cache capacity still accelerates non-pinned hot experts
        for (size_t idx : expert_indices) {
            if (assign[idx] != PlacementTier::SSD_STREAM) continue;
            const uint64_t sz = objects[idx].bytes;
            if (vram_u + sz <= budget.vram_bytes) {
                assign[idx] = PlacementTier::VRAM;
                vram_u += sz;
            } else if (ram_u + sz <= budget.ram_bytes) {
                assign[idx] = PlacementTier::RAM;
                ram_u += sz;
            }
        }

        const double cost = evaluate_placement_cost_sec(objects, assign, probe);
        if (cost < best_plan.expected_sec_per_tok) {
            best_plan.feasible             = true;
            best_plan.tier_of              = std::move(assign);
            best_plan.vram_used_bytes      = vram_u;
            best_plan.ram_used_bytes       = ram_u;
            best_plan.ssd_stream_bytes     = 0;
            double weighted_demand         = 0.0;
            double resident_demand         = 0.0;
            for (size_t i = 0; i < objects.size(); ++i) {
                const double w = std::max(0.0, objects[i].freq) * static_cast<double>(objects[i].bytes);
                weighted_demand += w;
                if (best_plan.tier_of[i] == PlacementTier::SSD_STREAM) {
                    best_plan.ssd_stream_bytes += objects[i].bytes;
                } else {
                    resident_demand += w;
                }
            }
            best_plan.best_pin_fraction    = pin_frac;
            best_plan.expected_sec_per_tok = cost;
            best_plan.predicted_tok_s      = (cost > 0.0) ? (1.0 / cost) : 0.0;
            best_plan.predicted_hit_rate   = (weighted_demand > 0.0) ? (resident_demand / weighted_demand) : 1.0;
        }
    }

    return best_plan;
}

bool validate_plan_with_governor(const PlacementPlan& plan,
                                 const tf_governor& gov,
                                 std::string* reason_out) {
    if (!plan.feasible) {
        if (reason_out) *reason_out = plan.reason;
        return false;
    }
    std::vector<PlanReservation> reqs{
        {MemoryTier::VRAM, plan.vram_used_bytes, "planner_vram"},
        {MemoryTier::RAM, plan.ram_used_bytes, "planner_ram"},
    };
    return gov.check_plan(reqs, reason_out);
}

} // namespace tierflow

