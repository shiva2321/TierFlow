#include "tf_draft.h"

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace tierflow {

namespace {

using Clock = std::chrono::steady_clock;

void quiet_draft_log_cb(ggml_log_level level, const char* text, void* ud) {
    (void)level;
    (void)text;
    (void)ud;
}

} // namespace

tf_draft::~tf_draft() {
    shutdown();
}

void tf_draft::shutdown() {
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    history_.clear();
    n_vocab_     = 0;
    model_bytes_ = 0;
}

bool tf_draft::init(const DraftConfig& cfg, std::string* err_out) {
    shutdown();
    cfg_ = cfg;
    if (cfg_.gguf_path.empty()) {
        if (err_out) *err_out = "Empty draft GGUF path";
        return false;
    }
    if (cfg_.quiet_logs) {
        llama_log_set(quiet_draft_log_cb, nullptr);
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = cfg_.n_gpu_layers;
    mparams.load_mode          = LLAMA_LOAD_MODE_DIRECT_IO; // zero OS PageCache footprint
    mparams.use_extra_bufts    = true;

    model_ = llama_model_load_from_file(cfg_.gguf_path.c_str(), mparams);
    if (!model_) {
        if (err_out) *err_out = "Failed to load draft GGUF via Direct I/O: " + cfg_.gguf_path;
        return false;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                = cfg_.n_ctx;
    cparams.n_batch              = cfg_.n_batch;
    cparams.n_ubatch             = cfg_.n_batch;
    cparams.n_threads            = cfg_.n_threads;
    cparams.n_threads_batch      = cfg_.n_threads;
    cparams.flash_attn_type      = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.type_k               = (cfg_.kv_precision == KvPrecision::F16) ? GGML_TYPE_F16 : GGML_TYPE_Q8_0;
    cparams.type_v               = (cfg_.kv_precision == KvPrecision::F16)
        ? GGML_TYPE_F16
        : (cfg_.kv_precision == KvPrecision::Q8_K_Q4_V ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0);

    ctx_ = llama_init_from_model(model_, cparams);
    if (!ctx_) {
        shutdown();
        if (err_out) *err_out = "Failed to initialize draft llama_context";
        return false;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model_);
    n_vocab_     = vocab ? llama_vocab_n_tokens(vocab) : 0;
    model_bytes_ = llama_model_size(model_);
    history_.clear();
    return n_vocab_ > 0;
}

void tf_draft::reset() {
    if (ctx_) {
        llama_memory_clear(llama_get_memory(ctx_), true);
    }
    history_.clear();
}

int32_t tf_draft::greedy_from_logits(int32_t idx) const {
    const float* logits = llama_get_logits_ith(ctx_, idx);
    if (!logits || n_vocab_ <= 0) return 0;
    int32_t best_id = 0;
    float   best_v  = logits[0];
    for (int32_t v = 1; v < n_vocab_; ++v) {
        if (logits[v] > best_v) {
            best_v  = logits[v];
            best_id = v;
        }
    }
    return best_id;
}

int32_t tf_draft::sample_with_probs(int32_t idx, float temp, std::mt19937& rng, std::vector<double>* probs_out) const {
    if (temp <= 0.0f) {
        if (probs_out) probs_out->clear();
        return greedy_from_logits(idx);
    }
    const float* logits = llama_get_logits_ith(ctx_, idx);
    if (!logits || n_vocab_ <= 0) return 0;

    double max_l = logits[0];
    for (int32_t v = 1; v < n_vocab_; ++v) {
        if (logits[v] > max_l) max_l = logits[v];
    }
    std::vector<double> probs(static_cast<size_t>(n_vocab_));
    double sum = 0.0;
    for (int32_t v = 0; v < n_vocab_; ++v) {
        const double p = std::exp((logits[v] - max_l) / temp);
        probs[static_cast<size_t>(v)] = p;
        sum += p;
    }
    const double inv = (sum > 0.0) ? (1.0 / sum) : 0.0;
    for (int32_t v = 0; v < n_vocab_; ++v) {
        probs[static_cast<size_t>(v)] *= inv;
    }
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    const double target = u01(rng);
    double acc = 0.0;
    int32_t chosen = n_vocab_ - 1;
    for (int32_t v = 0; v < n_vocab_; ++v) {
        acc += probs[static_cast<size_t>(v)];
        if (acc >= target) {
            chosen = v;
            break;
        }
    }
    if (probs_out) {
        *probs_out = std::move(probs);
    }
    return chosen;
}

DraftProposal tf_draft::propose(const std::vector<int32_t>& target_history,
                                int32_t                     k,
                                float                       temp,
                                std::mt19937&               rng) {
    DraftProposal prop{};
    if (!ctx_ || !model_ || k <= 0 || target_history.empty()) {
        return prop;
    }

    const auto t0 = Clock::now();

    // Align KV state with target_history by finding longest matching prefix.
    const size_t min_len = std::min(history_.size(), target_history.size());
    size_t lcp = 0;
    while (lcp < min_len && history_[lcp] == target_history[lcp]) {
        ++lcp;
    }
    // We always need at least the final token of target_history decoded on this call
    // if lcp == target_history.size() so live logits at the end are present.
    if (lcp == target_history.size() && lcp > 0) {
        --lcp;
    }
    if (lcp < history_.size()) {
        llama_memory_seq_rm(llama_get_memory(ctx_), 0, static_cast<int32_t>(lcp), -1);
        history_.resize(lcp);
    }

    // Chunked decode of missing suffix so draft logits sit cleanly at target_history.back().
    const int32_t suffix_len = static_cast<int32_t>(target_history.size() - lcp);
    const int32_t batch_cap  = std::max<int32_t>(1, std::min<int32_t>(static_cast<int32_t>(cfg_.n_batch), suffix_len));
    llama_batch batch = llama_batch_init(batch_cap, 0, 1);

    int32_t offset = 0;
    while (offset < suffix_len) {
        const int32_t chunk = std::min<int32_t>(batch_cap, suffix_len - offset);
        batch.n_tokens = chunk;
        for (int32_t i = 0; i < chunk; ++i) {
            const size_t pos = lcp + static_cast<size_t>(offset + i);
            batch.token[i]     = target_history[pos];
            batch.pos[i]       = static_cast<int32_t>(pos);
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = (offset + i == suffix_len - 1) ? 1 : 0;
        }
        if (llama_decode(ctx_, batch) != 0) {
            llama_batch_free(batch);
            reset();
            return prop;
        }
        for (int32_t i = 0; i < chunk; ++i) {
            history_.push_back(batch.token[i]);
        }
        offset += chunk;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model_);
    int32_t last_logit_idx = batch.n_tokens - 1;
    llama_batch_free(batch);

    llama_batch step_batch = llama_batch_init(1, 0, 1);
    for (int32_t i = 0; i < k; ++i) {
        if (static_cast<uint32_t>(history_.size()) >= cfg_.n_ctx) {
            break;
        }
        std::vector<double> q_probs;
        const int32_t tok = (temp > 0.0f)
            ? sample_with_probs(last_logit_idx, temp, rng, &q_probs)
            : greedy_from_logits(last_logit_idx);

        prop.tokens.push_back(tok);
        if (temp > 0.0f) {
            prop.probs.push_back(std::move(q_probs));
        }

        if (i + 1 >= k || (vocab && llama_vocab_is_eog(vocab, tok))) {
            break;
        }

        // Advance draft model by tok to obtain logits for draft step i + 1
        step_batch.n_tokens     = 1;
        step_batch.token[0]     = tok;
        step_batch.pos[0]       = static_cast<int32_t>(history_.size());
        step_batch.n_seq_id[0]  = 1;
        step_batch.seq_id[0][0] = 0;
        step_batch.logits[0]    = 1;
        if (llama_decode(ctx_, step_batch) != 0) {
            break;
        }
        history_.push_back(tok);
        last_logit_idx = 0;
    }
    llama_batch_free(step_batch);

    prop.elapsed_sec = std::chrono::duration<double>(Clock::now() - t0).count();
    return prop;
}

} // namespace tierflow

