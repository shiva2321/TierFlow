#include "tf_governor.h"

#include <limits>
#include <sstream>

namespace tierflow {

tf_governor::tf_governor(uint64_t ram_budget_bytes,
                         uint64_t vram_budget_bytes,
                         uint64_t ssd_spill_budget_bytes) {
    budgets_[tier_idx(MemoryTier::RAM)]       = ram_budget_bytes;
    budgets_[tier_idx(MemoryTier::VRAM)]      = vram_budget_bytes;
    budgets_[tier_idx(MemoryTier::SSD_SPILL)] = ssd_spill_budget_bytes;
}

bool tf_governor::reserve(MemoryTier tier, uint64_t bytes, const std::string& tag) {
    if (tag.empty()) return false;
    std::lock_guard<std::mutex> lock(mu_);
    const size_t idx = tier_idx(tier);
    if (bytes > budgets_[idx] || used_[idx] > budgets_[idx] - bytes) {
        return false;
    }
    used_[idx] += bytes;
    tags_[idx][tag] += bytes;
    return true;
}

bool tf_governor::release(MemoryTier tier, uint64_t bytes, const std::string& tag) {
    if (tag.empty()) return false;
    std::lock_guard<std::mutex> lock(mu_);
    const size_t idx = tier_idx(tier);
    auto it = tags_[idx].find(tag);
    if (it == tags_[idx].end() || it->second < bytes || used_[idx] < bytes) {
        return false;
    }
    it->second -= bytes;
    used_[idx] -= bytes;
    if (it->second == 0) {
        tags_[idx].erase(it);
    }
    return true;
}

bool tf_governor::check_plan(const std::vector<PlanReservation>& items,
                             std::string* reason_out) const {
    std::lock_guard<std::mutex> lock(mu_);
    uint64_t planned[3]{0, 0, 0};

    for (const auto& item : items) {
        if (item.tag.empty()) {
            if (reason_out) *reason_out = "Plan item has empty tag";
            return false;
        }
        const size_t idx = tier_idx(item.tier);
        if (item.bytes > (std::numeric_limits<uint64_t>::max)() - planned[idx]) {
            if (reason_out) *reason_out = "Plan byte total overflow";
            return false;
        }
        planned[idx] += item.bytes;
    }

    const char* names[3] = {"RAM", "VRAM", "SSD_SPILL"};
    for (size_t i = 0; i < 3; ++i) {
        if (planned[i] > budgets_[i] || used_[i] > budgets_[i] - planned[i]) {
            if (reason_out) {
                std::ostringstream oss;
                oss << "Plan exceeds " << names[i] << " budget: used=" << used_[i]
                    << " + planned=" << planned[i] << " > budget=" << budgets_[i];
                *reason_out = oss.str();
            }
            return false;
        }
    }
    return true;
}

uint64_t tf_governor::used(MemoryTier tier) const {
    std::lock_guard<std::mutex> lock(mu_);
    return used_[tier_idx(tier)];
}

uint64_t tf_governor::budget(MemoryTier tier) const {
    std::lock_guard<std::mutex> lock(mu_);
    return budgets_[tier_idx(tier)];
}

uint64_t tf_governor::remaining(MemoryTier tier) const {
    std::lock_guard<std::mutex> lock(mu_);
    const size_t idx = tier_idx(tier);
    return budgets_[idx] - used_[idx];
}

uint64_t tf_governor::sum_tags(MemoryTier tier) const {
    std::lock_guard<std::mutex> lock(mu_);
    uint64_t s = 0;
    for (const auto& kv : tags_[tier_idx(tier)]) {
        s += kv.second;
    }
    return s;
}

std::unordered_map<std::string, uint64_t> tf_governor::breakdown(MemoryTier tier) const {
    std::lock_guard<std::mutex> lock(mu_);
    return tags_[tier_idx(tier)];
}

} // namespace tierflow

