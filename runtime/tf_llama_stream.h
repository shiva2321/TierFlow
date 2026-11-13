#pragma once

#include "tf_cache.h"
#include "tf_governor.h"
#include "tf_io.h"
#include "tf_kv.h"
#include "tf_manifest.h"
#include "tf_planner.h"
#include "tf_prefetch.h"
#include "tf_probe.h"
#include "tf_spec.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct ggml_tensor;
struct llama_model;
struct llama_context;
struct llama_vocab;

namespace tierflow {

enum class RunMode : uint8_t {
    TierFlow  = 0,
    StockMmap = 1,
    StockDio  = 2,
};

enum class PrefetchMode : uint8_t {
    LookAhead = 0,
    Off       = 1,
    Shuffled  = 2,
};

enum class SpecKMode : uint8_t {
    Off          = 0,
    Adaptive     = 1,
    Fixed4       = 2,
    NaiveNoUnion = 3,
};

struct StreamEngineConfig {
    RunMode      mode              = RunMode::TierFlow;
    std::string  gguf_path;
    std::string  manifest_path;
    uint64_t     ram_budget_bytes  = 0;     // 0 = auto / unlimited in stock mode
    uint32_t     n_ctx             = 2048;
    uint32_t     n_batch           = 512;
    int32_t      n_threads         = 6;
    KvPrecision  kv_precision      = KvPrecision::Q8_0;
    CachePolicy  cache_policy      = CachePolicy::WTinyLFU_SLRU;
    size_t       cache_slots_max   = 12;    // dynamic slot bank size cap (actual capped by budget)
    PrefetchMode prefetch_mode     = PrefetchMode::LookAhead;
    int32_t      lookahead_depth   = 1;
    bool         repin_enabled     = false;
    bool         allow_slow_stream = false;
    bool         no_repack         = false; // match llama.cpp -no-repack in StockMmap/StockDio
    bool         use_ioring        = false; // default IOCP (win32_iocp) or optional IoRing
    SpecKMode    spec_k_mode       = SpecKMode::Off;
    std::string  prefix_cache_dir;
    bool         quiet_logs        = true;
};

struct StreamPromptResult {
    bool                 ok                        = false;
    std::string          error;
    double               ttft_ms                   = 0.0;
    double               prefill_tok_s             = 0.0;
    double               decode_tok_s              = 0.0;
    int32_t              n_prompt_tokens           = 0;
    int32_t              n_generated_tokens        = 0;
    std::vector<int32_t> generated_tokens;
    std::string          generated_text;
    std::vector<float>   first_step_top_logits;    // top 16 logits (first decode step)
    std::vector<int32_t> first_step_top_ids;

    // TierFlow runtime instrumentation
    uint64_t             cache_hits                = 0;
    uint64_t             cache_misses              = 0;
    double               hit_rate_ram              = 1.0;
    uint64_t             bytes_read_ssd            = 0;
    double               bytes_read_ssd_per_token  = 0.0;
    uint64_t             prefetch_hits             = 0;
    uint64_t             prefetch_submitted        = 0;
    uint64_t             prefetch_wasted_bytes     = 0;
    double               spec_k_mean               = 0.0;
    double               spec_accept_rate          = 0.0;
    bool                 prefix_cache_hit          = false;
};

struct StreamInitSummary {
    bool        ok                         = false;
    std::string error;
    uint64_t    model_total_bytes          = 0;
    uint64_t    ram_weights_bytes          = 0;
    uint64_t    ram_slot_bank_bytes        = 0;
    uint64_t    ram_kv_bytes               = 0;
    uint64_t    ram_compute_reserve_bytes  = 0;
    uint64_t    ram_governor_total_bytes   = 0;
    uint64_t    ssd_stream_bytes           = 0;
    size_t      n_streamed_tensors         = 0;
    size_t      n_cache_slots              = 0;
    uint64_t    slot_stride_bytes          = 0;
    double      predicted_hit_rate         = 1.0;
    double      predicted_tok_s            = 0.0;
    double      predicted_tok_s_burst      = 0.0;
    double      load_time_ms               = 0.0;
};

class tf_llama_stream {
public:
    explicit tf_llama_stream(const StreamEngineConfig& cfg);
    ~tf_llama_stream();

    tf_llama_stream(const tf_llama_stream&) = delete;
    tf_llama_stream& operator=(const tf_llama_stream&) = delete;

    StreamInitSummary init();

    // Greedy or stochastic generation on a text prompt.
    StreamPromptResult run_prompt(const std::string& prompt,
                                  int32_t n_predict,
                                  float temp = 0.0f,
                                  uint32_t seed = 42,
                                  bool save_or_use_prefix_cache = false);

    const StreamInitSummary& summary() const { return summary_; }
    const HardwareProfile&   hardware_profile() const { return probe_; }

private:
    struct StreamedTensorInfo {
        uint32_t    object_id          = 0;
        std::string name;
        uint64_t    tfpack_offset      = 0;
        uint64_t    exact_bytes        = 0;
        uint64_t    page_aligned_bytes = 0;
        int32_t     layer              = -1;
        ggml_tensor* tensor            = nullptr;
        uint64_t    access_count       = 0;
    };

    struct PendingIo {
        uint32_t object_id   = 0;
        int32_t  slot_id     = -1;
        uint64_t tag         = 0;
        bool     is_prefetch = false;
        uint64_t bytes       = 0;
    };

    // Callbacks invoked by llama.cpp & ggml_backend_sched
    static bool tf_is_streamed_cb(const char* name, void* ud);
    static void tf_set_tensor_data_cb(ggml_tensor* tensor, void* ud);
    static bool tf_eval_cb(ggml_tensor* node, bool ask, void* ud);

    bool handle_eval_node(ggml_tensor* node, bool ask);
    uint8_t* slot_ptr(int32_t slot_id) const;
    void poll_completed_async(bool block_one = false);
    int32_t acquire_object_sync(uint32_t obj_id, int32_t order_idx);
    void issue_lookahead_prefetch(int32_t current_order_idx);
    void drain_all_async();

    StreamEngineConfig cfg_;
    StreamInitSummary  summary_;
    HardwareProfile    probe_;
    TfManifest         manifest_;
    PlacementPlan      plan_;

    std::unique_ptr<tf_governor> gov_;
    std::unique_ptr<tf_io_ex>    io_;
    std::unique_ptr<tf_cache>    cache_;
    std::unique_ptr<tf_kv_store> kv_store_;
    void*                        slot_bank_mem_ = nullptr;
    size_t                       slot_bank_bytes_ = 0;
    size_t                       slot_stride_   = 0;

    // Mapping from manifest/tensor name & ggml_tensor*
    std::unordered_map<std::string, uint32_t>            name_to_obj_id_;
    std::unordered_set<std::string>                      streamed_names_;
    std::unordered_map<const ggml_tensor*, uint32_t>     tensor_to_obj_id_;
    std::vector<StreamedTensorInfo>                      streamed_objs_; // indexed by dense streamed idx
    std::unordered_map<uint32_t, size_t>                 obj_id_to_sidx_;
    std::unordered_set<std::string>                      loaded_ram_names_;

    // Recorded graph execution order for L -> L+1 look-ahead
    std::vector<uint32_t>                                exec_order_;
    std::unordered_map<uint32_t, int32_t>                obj_to_order_idx_;
    bool                                                 recording_order_ = true;

    // Async I/O state
    std::unordered_map<uint64_t, PendingIo>              inflight_by_tag_;
    std::unordered_map<uint32_t, uint64_t>               inflight_tag_by_obj_;
    std::vector<uint8_t>                                 slot_is_prefetch_loading_;
    std::vector<int32_t>                                 active_node_slots_;
    std::unordered_set<uint32_t>                         prefetched_unused_objs_;
    std::unordered_set<uint32_t>                         repinned_objs_;
    uint64_t                                             next_tag_ = 1;
    std::vector<uint32_t>                                shuffled_order_;
    uint64_t                                             shuffle_rng_ = 0x9e3779b97f4a7c15ULL;

    // Bounce buffer used only once during RAM-resident init load from .tfpack
    void*                                                init_bounce_ = nullptr;
    size_t                                               init_bounce_bytes_ = 0;

    // Counters
    uint64_t                                             run_hits_              = 0;
    uint64_t                                             run_misses_            = 0;
    uint64_t                                             run_bytes_ssd_         = 0;
    uint64_t                                             run_prefetch_sub_      = 0;
    uint64_t                                             run_prefetch_hits_     = 0;
    uint64_t                                             run_prefetch_wasted_b_ = 0;

    llama_model*   model_ = nullptr;
    llama_context* ctx_   = nullptr;
};

} // namespace tierflow
