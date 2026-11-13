#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace tierflow {

enum class MemoryTier : uint8_t {
    RAM       = 0,
    VRAM      = 1,
    SSD_SPILL = 2,
};

struct PlanReservation {
    MemoryTier  tier = MemoryTier::RAM;
    uint64_t    bytes = 0;
    std::string tag;
};

class tf_governor {
public:
    tf_governor(uint64_t ram_budget_bytes,
                uint64_t vram_budget_bytes,
                uint64_t ssd_spill_budget_bytes = 0);

    // Reserve `bytes` under `tier` for `tag`. Fails atomically (returns false)
    // if `used(tier) + bytes > budget(tier)` or if `tag` is empty.
    bool reserve(MemoryTier tier, uint64_t bytes, const std::string& tag);

    // Release previously reserved `bytes` for `tag`. Returns false if `tag`
    // holds fewer than `bytes`.
    bool release(MemoryTier tier, uint64_t bytes, const std::string& tag);

    // Non-mutating start-up validator: checks total requested bytes per tier
    // against budgets without modifying reservations.
    bool check_plan(const std::vector<PlanReservation>& items,
                    std::string* reason_out = nullptr) const;

    uint64_t used(MemoryTier tier) const;
    uint64_t budget(MemoryTier tier) const;
    uint64_t remaining(MemoryTier tier) const;
    uint64_t sum_tags(MemoryTier tier) const;

    std::unordered_map<std::string, uint64_t> breakdown(MemoryTier tier) const;

private:
    static size_t tier_idx(MemoryTier tier) {
        return static_cast<size_t>(tier);
    }

    mutable std::mutex mu_;
    uint64_t           budgets_[3]{0, 0, 0};
    uint64_t           used_[3]{0, 0, 0};
    std::unordered_map<std::string, uint64_t> tags_[3];
};

} // namespace tierflow

