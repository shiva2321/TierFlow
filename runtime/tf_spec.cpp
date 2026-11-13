#include "tf_spec.h"

#include <algorithm>
#include <cmath>

namespace tierflow {

double spec_expected_accepted(double alpha, int k) {
    if (k <= 0) return 1.0;
    alpha = std::clamp(alpha, 0.0, 1.0);
    if (std::abs(1.0 - alpha) < 1e-9) {
        return static_cast<double>(k + 1);
    }
    return (1.0 - std::pow(alpha, static_cast<double>(k + 1))) / (1.0 - alpha);
}

tf_spec_scheduler::tf_spec_scheduler(const SpecSchedulerConfig& cfg) : cfg_(cfg) {}

SpecDecision tf_spec_scheduler::choose_k(const SpecInputs& in) const {
    SpecDecision out{};
    const int k_max = std::max(0, cfg_.k_max);

    auto u_at = [&](int k) -> double {
        if (in.u_total_by_k.empty()) return 0.0;
        if (static_cast<size_t>(k) < in.u_total_by_k.size()) return std::max(0.0, in.u_total_by_k[k]);
        return std::max(0.0, in.u_total_by_k.back());
    };

    double best_score = -1.0;
    int    best_k     = 0;

    for (int k = 0; k <= k_max; ++k) {
        const double e_acc   = spec_expected_accepted(in.alpha_hat, k);
        const double t_ver   = in.c_always_sec + u_at(k) * in.cost_per_expert_sec;
        const double t_dft   = static_cast<double>(k) * in.t_draft_sec;
        const double total_t = std::max(1e-12, t_ver + t_dft);
        const double score   = e_acc / total_t;

        if (k == 0) out.base_score = score;
        if (score > best_score) {
            best_score = score;
            best_k     = k;
        }
    }

    out.k_opt      = best_k;
    out.best_score = best_score;
    return out;
}

void tf_spec_scheduler::update_ema(int k_drafted, int tokens_accepted, double) {
    if (k_drafted <= 0) return;
    // Standard acceptance ratio excluding the guaranteed final token
    const int bonus_acc = std::clamp(tokens_accepted - 1, 0, k_drafted);
    const double obs_a  = static_cast<double>(bonus_acc) / static_cast<double>(k_drafted);
    alpha_ema_          = cfg_.ema_decay * alpha_ema_ + (1.0 - cfg_.ema_decay) * obs_a;
}

} // namespace tierflow

