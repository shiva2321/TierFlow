#pragma once

#include <cstdint>
#include <vector>

namespace tierflow {

// Closed-form expected accepted tokens per verification step of draft depth k:
//   E_acc(k) = (1 - alpha^(k + 1)) / (1 - alpha), approaching (k + 1) as alpha -> 1.
double spec_expected_accepted(double alpha, int k);

struct SpecSchedulerConfig {
    int    k_max      = 8;
    double ema_decay  = 0.85;
};

struct SpecInputs {
    double              alpha_hat          = 0.7;
    double              c_always_sec       = 0.010;
    double              cost_per_expert_sec = 0.002;
    double              t_draft_sec        = 0.001;
    // u_total_by_k[k] = total expected distinct missing/loaded experts summed across layers
    // when verifying k+1 positions (k = 0..k_max). If shorter, last value is extrapolated linearly/constantly.
    std::vector<double> u_total_by_k;
};

struct SpecDecision {
    int    k_opt       = 0;
    double best_score  = 0.0; // expected tokens per second
    double base_score  = 0.0; // tokens per second at k = 0
};

class tf_spec_scheduler {
public:
    explicit tf_spec_scheduler(const SpecSchedulerConfig& cfg = {});

    SpecDecision choose_k(const SpecInputs& in) const;

    void   update_ema(int k_drafted, int tokens_accepted, double observed_union);
    double alpha_ema() const { return alpha_ema_; }

private:
    SpecSchedulerConfig cfg_;
    double              alpha_ema_ = 0.65;
};

} // namespace tierflow

