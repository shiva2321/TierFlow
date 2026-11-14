#pragma once

#include "tf_llama_stream.h"

#include <cstdint>
#include <random>
#include <string>
#include <vector>

struct llama_context;
struct llama_model;

namespace tierflow {

struct DraftConfig {
    std::string  gguf_path;
    uint32_t     n_ctx         = 2048;
    uint32_t     n_batch       = 512;
    int32_t      n_threads     = 4;
    int32_t      n_gpu_layers  = 99;
    KvPrecision  kv_precision  = KvPrecision::Q8_0;
    bool         quiet_logs    = true;
};

struct DraftProposal {
    std::vector<int32_t>              tokens;
    std::vector<std::vector<double>>  probs; // non-empty when temp > 0.0f for Leviathan sampling
    double                            elapsed_sec = 0.0;
};

/// Universal small draft model wrapper (e.g. Qwen3-0.6B) for <=4-token
/// speculative decoding against TierFlow target models.
class tf_draft {
public:
    tf_draft() = default;
    ~tf_draft();

    tf_draft(const tf_draft&)            = delete;
    tf_draft& operator=(const tf_draft&) = delete;

    bool init(const DraftConfig& cfg, std::string* err_out = nullptr);
    void shutdown();

    /// Resets KV cache before starting a new prompt.
    void reset();

    /// Synchronizes draft KV state with `target_history` (truncating any rejected
    /// draft suffix via `llama_memory_seq_rm` and decoding accepted/bonus tokens),
    /// then proposes up to `k` draft tokens.
    DraftProposal propose(const std::vector<int32_t>& target_history,
                          int32_t                     k,
                          float                       temp,
                          std::mt19937&               rng);

    int32_t  n_vocab() const { return n_vocab_; }
    uint64_t model_bytes() const { return model_bytes_; }
    bool     is_ready() const { return model_ != nullptr && ctx_ != nullptr; }

private:
    int32_t greedy_from_logits(int32_t idx) const;
    int32_t sample_with_probs(int32_t idx, float temp, std::mt19937& rng, std::vector<double>* probs_out) const;

    DraftConfig           cfg_;
    llama_model*          model_       = nullptr;
    llama_context*        ctx_         = nullptr;
    int32_t               n_vocab_     = 0;
    uint64_t              model_bytes_ = 0;
    std::vector<int32_t>  history_; // exact token prefix currently decoded in draft KV
};

} // namespace tierflow

