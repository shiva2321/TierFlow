#pragma once

#include <algorithm>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace tierflow {

// Section 8.3 & 8.9 adaptive look-ahead prefetch controller
struct PrefetchConfig {
    int    k_top           = 8;
    int    init_m          = 2;
    int    max_m           = 6;
    int    init_depth      = 1;
    int    max_depth       = 2;
    double min_accuracy     = 0.35;
    double expand_threshold = 0.75;
    double ema_decay        = 0.90;
};

class tf_prefetch_controller {
public:
    explicit tf_prefetch_controller(const PrefetchConfig& cfg = {})
        : cfg_(cfg), cur_m_(cfg.init_m), cur_depth_(cfg.init_depth) {}

    // Filter guessed IDs excluding already resident or inflight objects.
    std::vector<uint32_t> select_prefetch_candidates(
        const std::vector<uint32_t>& router_guesses,
        const std::unordered_set<uint32_t>& resident,
        const std::unordered_set<uint32_t>& inflight) const {
        std::vector<uint32_t> out;
        const size_t limit = static_cast<size_t>(cfg_.k_top + cur_m_);
        for (size_t i = 0; i < router_guesses.size() && i < limit; ++i) {
            const uint32_t id = router_guesses[i];
            if (resident.find(id) == resident.end() && inflight.find(id) == inflight.end()) {
                out.push_back(id);
            }
        }
        return out;
    }

    void record_observation(const std::vector<uint32_t>& guessed,
                            const std::vector<uint32_t>& actual) {
        if (actual.empty()) return;
        std::unordered_set<uint32_t> gset(guessed.begin(), guessed.end());
        size_t overlap = 0;
        for (uint32_t a : actual) {
            if (gset.find(a) != gset.end()) ++overlap;
        }
        const double acc = static_cast<double>(overlap) / static_cast<double>(actual.size());
        accuracy_ema_ = cfg_.ema_decay * accuracy_ema_ + (1.0 - cfg_.ema_decay) * acc;

        if (accuracy_ema_ < cfg_.min_accuracy) {
            if (cur_m_ > 0) --cur_m_;
            else if (cur_depth_ > 0) --cur_depth_;
        } else if (accuracy_ema_ > cfg_.expand_threshold) {
            if (cur_m_ < cfg_.max_m) ++cur_m_;
            else if (cur_depth_ < cfg_.max_depth) ++cur_depth_;
        }
    }

    double accuracy_ema() const { return accuracy_ema_; }
    int    current_m() const { return cur_m_; }
    int    current_depth() const { return cur_depth_; }

private:
    PrefetchConfig cfg_;
    int            cur_m_        = 2;
    int            cur_depth_    = 1;
    double         accuracy_ema_ = 0.70;
};

} // namespace tierflow

