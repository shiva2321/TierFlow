#pragma once

#include "tf_governor.h"
#include "tf_manifest.h"
#include "tf_probe.h"

#include <cstdint>
#include <string>
#include <vector>

namespace tierflow {

enum class PlacementTier : uint8_t {
    VRAM       = 0,
    RAM        = 1,
    SSD_STREAM = 2,
};

struct PlannerBudget {
    uint64_t vram_bytes = 0;
    uint64_t ram_bytes  = 0;
    bool     allow_slow_dense_stream = false;
};

struct PlannerObjectInput {
    uint32_t   object_id     = 0;
    uint64_t   bytes         = 0;
    double     freq          = 0.0;  // expected fraction of bytes read per generated token
    bool       always_on     = false;
    double     rank_priority = -1.0; // pinning tie-break priority (< 0 => uses freq)
};

struct PlacementPlan {
    bool                       feasible           = false;
    std::string                reason;
    std::vector<PlacementTier> tier_of;           // indexed parallel to input objects
    uint64_t                   vram_used_bytes    = 0;
    uint64_t                   ram_used_bytes     = 0;
    uint64_t                   ssd_stream_bytes   = 0;
    double                     best_pin_fraction  = 0.0;
    double                     expected_sec_per_tok = 0.0;
    double                     predicted_tok_s    = 0.0;
    double                     predicted_hit_rate = 0.0; // RAM+VRAM byte-weighted coverage of demand
};

// Evaluates exact cost (s/token) for a candidate placement assignment.
double evaluate_placement_cost_sec(const std::vector<PlannerObjectInput>& objects,
                                   const std::vector<PlacementTier>& assignment,
                                   const HardwareProfile& probe);

// Exact branch-and-bound / brute-force solver for small object counts (<= 14 objects).
PlacementPlan solve_exact_bruteforce(const std::vector<PlannerObjectInput>& objects,
                                     const PlannerBudget& budget,
                                     const HardwareProfile& probe);

// Production planner: uses exact solver when objects.size() <= 14, else fast
// fastest-tier-first always-on placement + pin_fraction grid over calibration freqs.
PlacementPlan build_placement_plan(const std::vector<PlannerObjectInput>& objects,
                                   const PlannerBudget& budget,
                                   const HardwareProfile& probe);

// Validates a PlacementPlan against a live tf_governor.
bool validate_plan_with_governor(const PlacementPlan& plan,
                                 const tf_governor& gov,
                                 std::string* reason_out = nullptr);

} // namespace tierflow

