#include "tf_llama_stream.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace tierflow {

namespace {

using Clock = std::chrono::steady_clock;

size_t round_up_4096_sz(size_t n) {
    return (n + 4095) & ~size_t(4095);
}

uint64_t fnv1a_64_tokens(const std::vector<llama_token>& tokens) {
    uint64_t h = 14695981039346656037ULL;
    for (llama_token tok : tokens) {
        uint32_t u = static_cast<uint32_t>(tok);
        for (int b = 0; b < 4; ++b) {
            h ^= (u >> (b * 8)) & 0xFFu;
            h *= 1099511628211ULL;
        }
    }
    return h;
}

void quiet_llama_log_cb(enum ggml_log_level level, const char* text, void* user_data) {
    (void)user_data;
    if (level >= GGML_LOG_LEVEL_ERROR) {
        std::fprintf(stderr, "%s", text);
    }
}

struct UnbufferedGgufReader {
    std::unique_ptr<tf_io_ex> io;
    uint64_t                  file_size   = 0;
    void*                     scratch     = nullptr;
    size_t                    scratch_sz  = 1ULL << 20; // 1 MiB aligned read chunk
    uint64_t                  cached_off  = UINT64_MAX;
    size_t                    cached_len  = 0;
};

size_t unbuffered_gguf_read_cb(void* userdata, void* output, uint64_t offset, size_t len) {
    auto* r = static_cast<UnbufferedGgufReader*>(userdata);
    if (!r || !r->io || offset >= r->file_size || len == 0) return 0;
    const size_t avail = static_cast<size_t>(std::min<uint64_t>(len, r->file_size - offset));
    auto* dst = static_cast<uint8_t*>(output);
    size_t copied = 0;
    uint64_t tag = 1;
    while (copied < avail) {
        const uint64_t cur_off = offset + copied;
        if (r->cached_off != UINT64_MAX &&
            cur_off >= r->cached_off &&
            cur_off < r->cached_off + r->cached_len) {
            const size_t in_off = static_cast<size_t>(cur_off - r->cached_off);
            const size_t take   = std::min(avail - copied, r->cached_len - in_off);
            std::memcpy(dst + copied, static_cast<const uint8_t*>(r->scratch) + in_off, take);
            copied += take;
            continue;
        }
        const uint64_t aligned_off = cur_off & ~uint64_t(4095);
        const size_t   want_min    = static_cast<size_t>(cur_off - aligned_off) + (avail - copied);
        // Prefetch up to scratch_sz on header scans to avoid re-reading 4 KiB per scalar field
        size_t req_sz = std::min(r->scratch_sz, round_up_4096_sz(std::max<size_t>(want_min, 512ULL << 10)));
        if (req_sz < 4096) req_sz = 4096;
        tf_read_req req{};
        req.object_id = 0;
        req.offset    = aligned_off;
        req.size      = req_sz;
        req.dst       = r->scratch;
        req.priority  = TF_PRIO_DEMAND;
        req.tag       = tag++;
        if (!r->io->submit(req)) {
            break;
        }
        tf_read_req done{};
        while (r->io->poll(&done, 1) == 0) {
        }
        r->cached_off = aligned_off;
        r->cached_len = static_cast<size_t>(std::min<uint64_t>(req_sz, r->file_size - aligned_off));
        if (cur_off >= r->cached_off + r->cached_len) break;
    }
    return copied;
}

gguf_context* load_gguf_meta_unbuffered(const std::string& gguf_path, std::string& err_out) {
    UnbufferedGgufReader reader;
    tf_io_options opts{};
    opts.max_inflight          = 4;
    opts.prefetch_inflight_cap = 1;
    opts.strict_priority       = true;
    opts.align_bytes           = 4096;
    reader.io                  = tf_io_create_iocp(opts);
    if (!reader.io || !reader.io->open(gguf_path.c_str())) {
        err_out = "Failed to open GGUF unbuffered for metadata: " + gguf_path;
        return nullptr;
    }
    reader.file_size = reader.io->file_size();
    reader.scratch   = tf_aligned_alloc(reader.scratch_sz, 4096);
    if (!reader.scratch) {
        err_out = "Failed to alloc scratch for unbuffered GGUF header";
        return nullptr;
    }
    gguf_init_params gp{};
    gp.no_alloc = true;
    gp.ctx      = nullptr;
    gguf_context* gctx = gguf_init_from_callback(
        unbuffered_gguf_read_cb, &reader, reader.scratch_sz - 4096, reader.file_size, gp);
    reader.io->close();
    tf_aligned_free(reader.scratch);
    if (!gctx) {
        err_out = "gguf_init_from_callback failed for: " + gguf_path;
    }
    return gctx;
}

uint64_t estimate_kv_cache_bytes(const llama_model* model, uint32_t n_ctx, KvPrecision prec) {
    const int32_t n_layer = llama_model_n_layer(model);
    const int32_t n_head_kv = llama_model_n_head_kv(model);
    const int32_t n_embd = llama_model_n_embd(model);
    const int32_t n_head = llama_model_n_head(model);
    const int32_t head_dim = (n_head > 0) ? (n_embd / n_head) : 128;
    const double elem_bytes = (prec == KvPrecision::F16) ? 2.0 : 1.0625;
    const double total = static_cast<double>(n_layer) * 2.0 * n_head_kv * head_dim * n_ctx * elem_bytes;
    return static_cast<uint64_t>(std::ceil(total));
}

std::vector<llama_token> prompt_lookup_draft(const std::vector<llama_token>& history,
                                             int32_t max_k,
                                             int32_t max_ngram = 4) {
    std::vector<llama_token> draft;
    if (max_k <= 0 || history.size() < 2) return draft;
    const size_t n = history.size();
    for (int32_t ng = std::min<int32_t>(max_ngram, static_cast<int32_t>(n - 1)); ng >= 1; --ng) {
        // Search backwards from latest earlier occurrence
        for (int64_t pos = static_cast<int64_t>(n) - ng - 1; pos >= 0; --pos) {
            bool match = true;
            for (int32_t j = 0; j < ng; ++j) {
                if (history[static_cast<size_t>(pos + j)] != history[n - ng + j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                const size_t start = static_cast<size_t>(pos + ng);
                for (size_t t = 0; t < static_cast<size_t>(max_k) && start + t < n - 1; ++t) {
                    draft.push_back(history[start + t]);
                }
                if (!draft.empty()) return draft;
            }
        }
    }
    return draft;
}

} // namespace

tf_llama_stream::tf_llama_stream(const StreamEngineConfig& cfg) : cfg_(cfg) {}

tf_llama_stream::~tf_llama_stream() {
    drain_all_async();
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    if (io_) {
        io_->close();
        io_.reset();
    }
    if (slot_bank_mem_) {
        tf_aligned_free(slot_bank_mem_);
        slot_bank_mem_ = nullptr;
    }
    if (init_bounce_) {
        tf_aligned_free(init_bounce_);
        init_bounce_ = nullptr;
    }
    kv_store_.reset();
    gov_.reset();
}

uint8_t* tf_llama_stream::slot_ptr(int32_t slot_id) const {
    return static_cast<uint8_t*>(slot_bank_mem_) + static_cast<size_t>(slot_id) * slot_stride_;
}

bool tf_llama_stream::tf_is_streamed_cb(const char* name, void* ud) {
    auto* self = static_cast<tf_llama_stream*>(ud);
    if (!self || !name) return false;
    return self->streamed_names_.count(name) > 0;
}

void tf_llama_stream::tf_set_tensor_data_cb(ggml_tensor* tensor, void* ud) {
    auto* self = static_cast<tf_llama_stream*>(ud);
    if (!self || !tensor) return;
    const char* name = ggml_get_name(tensor);
    if (!name) return;
    auto it = self->name_to_obj_id_.find(name);
    if (it == self->name_to_obj_id_.end()) return;
    const uint32_t obj_id = it->second;
    if (self->streamed_names_.count(name) > 0) {
        // Streamed tensor resides in 0-byte dummy buffer; give non-null sentinel pointer inside the
        // governor-reserved slot bank so ggml_gallocr skips allocation while cb_eval swaps real slots.
        tensor->data = self->slot_ptr(0);
        auto s_it = self->obj_id_to_sidx_.find(obj_id);
        if (s_it != self->obj_id_to_sidx_.end()) {
            self->streamed_objs_[s_it->second].tensor = tensor;
            self->tensor_to_obj_id_[tensor] = obj_id;
        }
        return;
    }
    if (!tensor->data) return;

    const auto& obj = self->manifest_.objects[obj_id];
    const size_t nbytes = ggml_nbytes(tensor);
    const size_t to_copy = std::min<size_t>(nbytes, obj.exact_bytes);
    const size_t aligned_read = static_cast<size_t>(obj.page_aligned_bytes);
    if (aligned_read > self->init_bounce_bytes_) {
        if (self->init_bounce_) tf_aligned_free(self->init_bounce_);
        self->init_bounce_bytes_ = aligned_read;
        self->init_bounce_ = tf_aligned_alloc(self->init_bounce_bytes_, 4096);
    }
    tf_read_req req{};
    req.object_id = obj_id;
    req.offset    = obj.tfpack_offset;
    req.size      = aligned_read;
    req.dst       = self->init_bounce_;
    req.priority  = TF_PRIO_DEMAND;
    req.tag       = 900000ULL + obj_id;
    if (!self->io_->submit(req)) {
        throw std::runtime_error("tf_io init submit failed for tensor " + std::string(name));
    }
    tf_read_req done{};
    while (self->io_->poll(&done, 1) == 0) {
    }
    std::memcpy(tensor->data, self->init_bounce_, to_copy);
    self->loaded_ram_names_.insert(name);
}

void tf_llama_stream::poll_completed_async(bool block_one) {
    tf_read_req done[16];
    int n = io_->poll(done, 16);
    while (block_one && n == 0 && !inflight_by_tag_.empty()) {
        std::this_thread::yield();
        n = io_->poll(done, 16);
    }
    for (int i = 0; i < n; ++i) {
        auto it = inflight_by_tag_.find(done[i].tag);
        if (it == inflight_by_tag_.end()) continue;
        PendingIo pio = it->second;
        inflight_by_tag_.erase(it);
        auto oit = inflight_tag_by_obj_.find(pio.object_id);
        if (oit != inflight_tag_by_obj_.end() && oit->second == pio.tag) {
            inflight_tag_by_obj_.erase(oit);
        }
        if (pio.slot_id >= 0 && static_cast<size_t>(pio.slot_id) < slot_is_prefetch_loading_.size()) {
            slot_is_prefetch_loading_[pio.slot_id] = 0;
            cache_->release_slot(pio.slot_id);
        }
    }
}

void tf_llama_stream::drain_all_async() {
    if (!io_) return;
    while (!inflight_by_tag_.empty()) {
        poll_completed_async(true);
    }
}

int32_t tf_llama_stream::acquire_object_sync(uint32_t obj_id, int32_t order_idx) {
    // Wait first if this object is currently in-flight from an async prefetch
    auto inf_it = inflight_tag_by_obj_.find(obj_id);
    if (inf_it != inflight_tag_by_obj_.end()) {
        const uint64_t wait_tag = inf_it->second;
        while (inflight_by_tag_.count(wait_tag) > 0) {
            poll_completed_async(true);
        }
    }

    auto res = cache_->access(obj_id);
    if (res.slot_id < 0) {
        // Fallback: drain async prefetches to free unreferenced Probation slots and retry
        drain_all_async();
        res = cache_->access(obj_id);
    }
    if (res.slot_id < 0) {
        throw std::runtime_error("tf_cache out of evictable slots during streaming");
    }
    if (res.evicted && prefetched_unused_objs_.erase(res.evicted_object_id) > 0) {
        auto s_it = obj_id_to_sidx_.find(res.evicted_object_id);
        if (s_it != obj_id_to_sidx_.end()) {
            run_prefetch_wasted_b_ += streamed_objs_[s_it->second].page_aligned_bytes;
        }
    }

    // If that slot happened to still have a prefetch hold (shouldn't since refcount==0 is required to evict), guard:
    cache_->acquire_slot(res.slot_id);

    if (res.hit) {
        ++run_hits_;
        if (prefetched_unused_objs_.erase(obj_id) > 0) {
            ++run_prefetch_hits_;
        }
    } else {
        ++run_misses_;
        const auto& obj = manifest_.objects[obj_id];
        const uint64_t tag = next_tag_++;
        tf_read_req req{};
        req.object_id = obj_id;
        req.offset    = obj.tfpack_offset;
        req.size      = obj.page_aligned_bytes;
        req.dst       = slot_ptr(res.slot_id);
        req.priority  = TF_PRIO_DEMAND;
        req.tag       = tag;
        if (!io_->submit(req)) {
            // Pump completions in case queue was full
            poll_completed_async(true);
            if (!io_->submit(req)) {
                cache_->release_slot(res.slot_id);
                throw std::runtime_error("tf_io demand submit failed");
            }
        }
        bool done_flag = false;
        while (!done_flag) {
            tf_read_req done[8];
            int n = io_->poll(done, 8);
            for (int i = 0; i < n; ++i) {
                if (done[i].tag == tag) {
                    done_flag = true;
                } else {
                    auto it = inflight_by_tag_.find(done[i].tag);
                    if (it != inflight_by_tag_.end()) {
                        PendingIo pio = it->second;
                        inflight_by_tag_.erase(it);
                        auto oit = inflight_tag_by_obj_.find(pio.object_id);
                        if (oit != inflight_tag_by_obj_.end() && oit->second == pio.tag) {
                            inflight_tag_by_obj_.erase(oit);
                        }
                        if (pio.slot_id >= 0 && static_cast<size_t>(pio.slot_id) < slot_is_prefetch_loading_.size()) {
                            slot_is_prefetch_loading_[pio.slot_id] = 0;
                            cache_->release_slot(pio.slot_id);
                        }
                    }
                }
            }
            if (!done_flag) std::this_thread::yield();
        }
        run_bytes_ssd_ += obj.page_aligned_bytes;
    }

    (void)order_idx;
    return res.slot_id;
}

void tf_llama_stream::issue_lookahead_prefetch(int32_t current_order_idx) {
    if (cfg_.prefetch_mode == PrefetchMode::Off || exec_order_.empty() || !cache_ || !io_) return;
    const size_t n_slots = cache_->capacity();
    if (n_slots < 3) return;
    const size_t max_prefetches = std::max<size_t>(1, std::min<size_t>(4, (n_slots - 2) / 2));
    if (inflight_by_tag_.size() >= max_prefetches) return;

    const int32_t depth = std::max<int32_t>(1, cfg_.lookahead_depth);
    const size_t total = exec_order_.size();
    for (int32_t d = 1; d <= depth; ++d) {
        if (inflight_by_tag_.size() >= max_prefetches) break;
        const size_t idx = (static_cast<size_t>(current_order_idx) + static_cast<size_t>(d)) % total;
        uint32_t target_obj = exec_order_[idx];
        if (cfg_.prefetch_mode == PrefetchMode::Shuffled && !shuffled_order_.empty()) {
            target_obj = shuffled_order_[idx % shuffled_order_.size()];
        }
        if (cache_->is_resident(target_obj) || inflight_tag_by_obj_.count(target_obj) > 0) {
            continue;
        }
        auto pres = cache_->prefetch(target_obj);
        if (pres.slot_id < 0) continue;
        if (pres.evicted && prefetched_unused_objs_.erase(pres.evicted_object_id) > 0) {
            auto s_it = obj_id_to_sidx_.find(pres.evicted_object_id);
            if (s_it != obj_id_to_sidx_.end()) {
                run_prefetch_wasted_b_ += streamed_objs_[s_it->second].page_aligned_bytes;
            }
        }
        cache_->acquire_slot(pres.slot_id);
        slot_is_prefetch_loading_[pres.slot_id] = 1;
        const auto& obj = manifest_.objects[target_obj];
        const uint64_t tag = next_tag_++;
        tf_read_req req{};
        req.object_id = target_obj;
        req.offset    = obj.tfpack_offset;
        req.size      = obj.page_aligned_bytes;
        req.dst       = slot_ptr(pres.slot_id);
        req.priority  = TF_PRIO_PREFETCH;
        req.tag       = tag;
        if (!io_->submit(req)) {
            slot_is_prefetch_loading_[pres.slot_id] = 0;
            cache_->release_slot(pres.slot_id);
            break;
        }
        PendingIo pio{target_obj, pres.slot_id, tag, true, obj.page_aligned_bytes};
        inflight_by_tag_[tag]        = pio;
        inflight_tag_by_obj_[target_obj] = tag;
        prefetched_unused_objs_.insert(target_obj);
        ++run_prefetch_sub_;
        run_bytes_ssd_ += obj.page_aligned_bytes;
    }
}

bool tf_llama_stream::handle_eval_node(ggml_tensor* node, bool ask) {
    if (!node || streamed_names_.empty()) return false;

    if (ask) {
        active_node_slots_.clear();
        bool uses_streamed = false;
        int32_t last_order_idx = -1;
        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            ggml_tensor* src = node->src[k];
            if (!src) continue;
            ggml_tensor* base = src->view_src ? src->view_src : src;
            auto it = tensor_to_obj_id_.find(base);
            if (it == tensor_to_obj_id_.end()) continue;
            uses_streamed = true;
            const uint32_t obj_id = it->second;
            int32_t order_idx = -1;
            auto oit = obj_to_order_idx_.find(obj_id);
            if (oit == obj_to_order_idx_.end()) {
                order_idx = static_cast<int32_t>(exec_order_.size());
                exec_order_.push_back(obj_id);
                obj_to_order_idx_[obj_id] = order_idx;
            } else {
                order_idx = oit->second;
            }
            last_order_idx = order_idx;

            const int32_t slot_id = acquire_object_sync(obj_id, order_idx);
            uint8_t* ptr = slot_ptr(slot_id);
            base->data = ptr;
            if (src != base) {
                src->data = ptr + src->view_offs;
            }
            active_node_slots_.push_back(slot_id);
            auto sit = obj_id_to_sidx_.find(obj_id);
            if (sit != obj_id_to_sidx_.end()) {
                ++streamed_objs_[sit->second].access_count;
                if (cfg_.repin_enabled && cache_ &&
                    repinned_objs_.size() < std::max<size_t>(1, cache_->capacity() / 3) &&
                    repinned_objs_.count(obj_id) == 0 &&
                    streamed_objs_[sit->second].access_count >= 2 &&
                    !slot_is_prefetch_loading_[slot_id]) {
                    if (cache_->pin(obj_id)) {
                        repinned_objs_.insert(obj_id);
                    }
                }
            }
        }
        if (uses_streamed && last_order_idx >= 0) {
            issue_lookahead_prefetch(last_order_idx);
            return true;
        }
        return false;
    } else {
        poll_completed_async(false);
        for (int32_t s : active_node_slots_) {
            cache_->release_slot(s);
        }
        active_node_slots_.clear();
        return true;
    }
}

bool tf_llama_stream::tf_eval_cb(ggml_tensor* node, bool ask, void* ud) {
    auto* self = static_cast<tf_llama_stream*>(ud);
    return self->handle_eval_node(node, ask);
}

StreamInitSummary tf_llama_stream::init() {
    summary_ = {};
    const auto t0 = Clock::now();
    if (cfg_.quiet_logs) {
        llama_log_set(quiet_llama_log_cb, nullptr);
    }
    llama_backend_init();

    if (cfg_.mode != RunMode::TierFlow) {
        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers       = 0;
        mparams.load_mode          = (cfg_.mode == RunMode::StockDio) ? LLAMA_LOAD_MODE_DIRECT_IO : LLAMA_LOAD_MODE_MMAP;
        mparams.use_extra_bufts    = !cfg_.no_repack;
        model_ = llama_model_load_from_file(cfg_.gguf_path.c_str(), mparams);
        if (!model_) {
            summary_.error = "Stock llama_model_load_from_file failed";
            return summary_;
        }
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx           = cfg_.n_ctx;
        cparams.n_batch         = cfg_.n_batch;
        cparams.n_ubatch        = cfg_.n_batch;
        cparams.n_threads       = cfg_.n_threads;
        cparams.n_threads_batch = cfg_.n_threads;
        cparams.type_k          = (cfg_.kv_precision == KvPrecision::F16) ? GGML_TYPE_F16 : GGML_TYPE_Q8_0;
        cparams.type_v          = (cfg_.kv_precision == KvPrecision::F16)
            ? GGML_TYPE_F16
            : (cfg_.kv_precision == KvPrecision::Q8_K_Q4_V ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0);
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        ctx_ = llama_init_from_model(model_, cparams);
        if (!ctx_) {
            summary_.error = "Stock llama_init_from_model failed";
            return summary_;
        }
        summary_.ok               = true;
        summary_.model_total_bytes = llama_model_size(model_);
        summary_.ram_kv_bytes     = estimate_kv_cache_bytes(model_, cfg_.n_ctx, cfg_.kv_precision);
        summary_.load_time_ms     = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        return summary_;
    }

    // TierFlow Mode: 1) Load manifest & probe machine hardware
    std::string err;
    if (!load_tf_manifest(cfg_.manifest_path, manifest_, &err)) {
        summary_.error = "load_tf_manifest failed: " + err;
        return summary_;
    }
    const int eff_qd = (cfg_.prefetch_mode == PrefetchMode::Off)
        ? 1
        : std::clamp(1 + cfg_.lookahead_depth, 1, 4);
    probe_ = run_tf_probe(manifest_.tfpack_path, cfg_.n_threads, eff_qd);

    std::string gerr;
    gguf_context* gctx = load_gguf_meta_unbuffered(cfg_.gguf_path, gerr);
    if (!gctx) {
        summary_.error = gerr;
        return summary_;
    }

    // Extract architectural KV dimensions from GGUF metadata before weight allocation
    auto get_u32_kv = [&](const char* suffix, int32_t fallback) -> int32_t {
        std::string key = manifest_.architecture + "." + suffix;
        int64_t kid = gguf_find_key(gctx, key.c_str());
        if (kid < 0) return fallback;
        return static_cast<int32_t>(gguf_get_val_u32(gctx, kid));
    };
    const int32_t n_layer   = std::max(1, manifest_.block_count);
    const int32_t n_head    = std::max(1, manifest_.head_count);
    const int32_t n_head_kv = std::max(1, manifest_.head_count_kv);
    const int32_t n_embd    = std::max(1, manifest_.embedding_length);
    const int32_t head_dim  = get_u32_kv("attention.key_length", n_embd / n_head);
    const double kv_elem_b  = (cfg_.kv_precision == KvPrecision::F16)
        ? 2.0
        : (cfg_.kv_precision == KvPrecision::Q8_K_Q4_V ? 0.8125 : 1.0625);
    const uint64_t kv_bytes = static_cast<uint64_t>(
        std::ceil(static_cast<double>(n_layer) * 2.0 * n_head_kv * head_dim * cfg_.n_ctx * kv_elem_b));
    const uint64_t compute_reserve = 64ULL << 20; // compute buffers + metadata + runtime headroom

    const uint64_t total_model_exact = manifest_.total_exact_bytes();
    summary_.model_total_bytes        = total_model_exact;
    summary_.ram_kv_bytes             = kv_bytes;
    summary_.ram_compute_reserve_bytes = compute_reserve;

    const uint64_t ram_budget = (cfg_.ram_budget_bytes > 0)
        ? cfg_.ram_budget_bytes
        : (total_model_exact + kv_bytes + compute_reserve + (128ULL << 20));
    if (ram_budget <= kv_bytes + compute_reserve) {
        gguf_free(gctx);
        summary_.error = "RAM budget too small even for KV cache + runtime compute reserve";
        return summary_;
    }
    const uint64_t avail_for_weights_and_cache = ram_budget - kv_bytes - compute_reserve;
    const bool needs_streaming = (avail_for_weights_and_cache < total_model_exact);

    // Prepare planner inputs cleanly
    name_to_obj_id_.clear();
    std::vector<PlannerObjectInput> p_inputs;
    p_inputs.reserve(manifest_.objects.size());
    uint64_t max_stream_aligned = 0;
    for (const auto& obj : manifest_.objects) {
        name_to_obj_id_[obj.name] = obj.object_id;
        // Classify FFN matrices (and Attn if needed) as streamable; keep small norms/biases & heads always_on
        const bool is_ffn_mat  = (obj.name.find(".ffn_gate.weight") != std::string::npos ||
                                  obj.name.find(".ffn_up.weight") != std::string::npos ||
                                  obj.name.find(".ffn_down.weight") != std::string::npos);
        const bool is_attn_mat = (obj.name.find(".attn_q.weight") != std::string::npos ||
                                  obj.name.find(".attn_k.weight") != std::string::npos ||
                                  obj.name.find(".attn_v.weight") != std::string::npos ||
                                  obj.name.find(".attn_output.weight") != std::string::npos);
        const bool streamable = is_ffn_mat || is_attn_mat;
        const bool is_sparse_embd = (obj.name == "token_embd.weight");
        if (streamable && obj.page_aligned_bytes > max_stream_aligned) {
            max_stream_aligned = obj.page_aligned_bytes;
        }
        // True per-token byte traffic multiplier (token_embd is a 1-row lookup; dense weights read 1.0x/tok)
        // Pinning preference (rank_priority) keeps attention tensors in RAM before FFN tensors.
        const double traffic_freq = is_sparse_embd ? 0.001 : 1.0;
        const double rank_prio    = streamable ? (is_attn_mat ? 1.5 : (1.0 - 1e-4 * std::max(0, obj.layer))) : 2.0;
        p_inputs.push_back({obj.object_id, obj.exact_bytes, traffic_freq, !streamable, rank_prio});
    }

    size_t n_slots = 0;
    uint64_t slot_bank_bytes = 0;
    if (needs_streaming) {
        slot_stride_ = static_cast<size_t>(round_up_4096_sz(max_stream_aligned));
        // Reserve dynamic slot bank within budget (minimum 4 slots required for safe async prefetch + active op)
        const uint64_t min_slots = 4;
        if (avail_for_weights_and_cache < min_slots * slot_stride_) {
            gguf_free(gctx);
            summary_.error = "RAM budget too small for minimum 4-slot streaming bank";
            return summary_;
        }
        // Allocate up to cache_slots_max slots, leaving room for pinned always-on weights!
        uint64_t always_on_bytes = 0;
        for (const auto& po : p_inputs) {
            if (po.always_on) always_on_bytes += po.bytes;
        }
        if (always_on_bytes + min_slots * slot_stride_ > avail_for_weights_and_cache && !cfg_.allow_slow_stream) {
            gguf_free(gctx);
            summary_.error = "Always-on weights + 4 streaming slots exceed RAM budget (enable --allow-slow-stream to proceed)";
            return summary_;
        }
        const uint64_t room_after_always = (avail_for_weights_and_cache > always_on_bytes)
            ? (avail_for_weights_and_cache - always_on_bytes)
            : (avail_for_weights_and_cache);
        const size_t affordable_slots = static_cast<size_t>(room_after_always / slot_stride_);
        n_slots = std::clamp<size_t>(affordable_slots, min_slots, std::max<size_t>(min_slots, cfg_.cache_slots_max));
        slot_bank_bytes = static_cast<uint64_t>(n_slots) * slot_stride_;
    }

    const uint64_t weight_pin_budget = (avail_for_weights_and_cache > slot_bank_bytes)
        ? (avail_for_weights_and_cache - slot_bank_bytes)
        : 0;
    PlannerBudget p_budget{};
    p_budget.vram_bytes             = 0;
    p_budget.ram_bytes              = weight_pin_budget;
    p_budget.allow_slow_dense_stream = cfg_.allow_slow_stream;
    plan_ = build_placement_plan(p_inputs, p_budget, probe_);
    if (!plan_.feasible) {
        gguf_free(gctx);
        summary_.error = "tf_planner rejected placement: " + plan_.reason;
        return summary_;
    }

    // Governor startup budget check & atomic reservations
    gov_ = std::make_unique<tf_governor>(ram_budget, 0, ram_budget * 2);
    std::vector<PlanReservation> startup_reqs{
        {MemoryTier::RAM, plan_.ram_used_bytes, "ram_weights"},
        {MemoryTier::RAM, slot_bank_bytes, "tf_cache_slots"},
        {MemoryTier::RAM, kv_bytes, "kv_cache"},
        {MemoryTier::RAM, compute_reserve, "compute_scratch"},
    };
    std::string gov_reason;
    if (!gov_->check_plan(startup_reqs, &gov_reason)) {
        gguf_free(gctx);
        summary_.error = "tf_governor rejected startup plan: " + gov_reason;
        return summary_;
    }
    for (const auto& r : startup_reqs) {
        if (r.bytes > 0 && !gov_->reserve(r.tier, r.bytes, r.tag)) {
            gguf_free(gctx);
            summary_.error = "tf_governor failed reserving " + r.tag;
            return summary_;
        }
    }

    streamed_names_.clear();
    streamed_objs_.clear();
    obj_id_to_sidx_.clear();
    for (size_t i = 0; i < manifest_.objects.size(); ++i) {
        if (plan_.tier_of[i] == PlacementTier::SSD_STREAM) {
            const auto& obj = manifest_.objects[i];
            streamed_names_.insert(obj.name);
            StreamedTensorInfo info{};
            info.object_id          = obj.object_id;
            info.name               = obj.name;
            info.tfpack_offset      = obj.tfpack_offset;
            info.exact_bytes        = obj.exact_bytes;
            info.page_aligned_bytes = obj.page_aligned_bytes;
            info.layer              = obj.layer;
            obj_id_to_sidx_[obj.object_id] = streamed_objs_.size();
            streamed_objs_.push_back(info);
            shuffled_order_.push_back(obj.object_id);
        }
    }
    std::mt19937_64 shuf_gen(20261112ULL);
    std::shuffle(shuffled_order_.begin(), shuffled_order_.end(), shuf_gen);

    // Burst prediction refinement alongside sustained planner prediction
    HardwareProfile burst_probe = probe_;
    burst_probe.bw_ssd_sustained_gbps = probe_.bw_ssd_burst_gbps;
    const double burst_cost = evaluate_placement_cost_sec(p_inputs, plan_.tier_of, burst_probe);

    summary_.ram_weights_bytes       = plan_.ram_used_bytes;
    summary_.ram_slot_bank_bytes     = slot_bank_bytes;
    summary_.ram_governor_total_bytes = gov_->used(MemoryTier::RAM);
    summary_.ssd_stream_bytes        = plan_.ssd_stream_bytes;
    summary_.n_streamed_tensors      = streamed_objs_.size();
    summary_.n_cache_slots           = n_slots;
    summary_.slot_stride_bytes       = slot_stride_;
    summary_.predicted_hit_rate      = plan_.predicted_hit_rate;
    summary_.predicted_tok_s         = plan_.predicted_tok_s;
    summary_.predicted_tok_s_burst   = (burst_cost > 0.0) ? (1.0 / burst_cost) : 0.0;

    // Open .tfpack via unbuffered async I/O
    tf_io_options io_opts{};
    io_opts.max_inflight          = 32;
    io_opts.prefetch_inflight_cap = 4;
    io_opts.strict_priority       = true;
    io_opts.align_bytes           = 4096;
    if (cfg_.use_ioring && tf_ioring_is_supported(nullptr)) {
        io_ = tf_io_create_ioring(io_opts);
    } else {
        io_ = tf_io_create_iocp(io_opts);
    }
    if (!io_ || !io_->open(manifest_.tfpack_path.c_str())) {
        gguf_free(gctx);
        summary_.error = "Failed to open .tfpack via tf_io unbuffered backend: " + manifest_.tfpack_path;
        return summary_;
    }

    if (n_slots > 0) {
        slot_bank_bytes_ = static_cast<size_t>(slot_bank_bytes);
        slot_bank_mem_   = tf_aligned_alloc(slot_bank_bytes_, 4096);
        if (!slot_bank_mem_) {
            gguf_free(gctx);
            summary_.error = "Failed to VirtualAlloc aligned slot bank";
            return summary_;
        }
        slot_is_prefetch_loading_.assign(n_slots, 0);
        CacheConfig ccfg{};
        ccfg.capacity_slots = n_slots;
        ccfg.policy         = cfg_.cache_policy;
        cache_              = std::make_unique<tf_cache>(ccfg);
    }

    // Construct llama_model using zero-mmap user callback loader + 0-byte streamed tensor dummy buffers
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = 0;
    mparams.use_extra_bufts    = false;
    mparams.tf_is_streamed_tensor = &tf_llama_stream::tf_is_streamed_cb;
    mparams.tf_is_streamed_ud     = this;

    try {
        model_ = llama_model_init_from_user(gctx, &tf_llama_stream::tf_set_tensor_data_cb, this, mparams);
    } catch (const std::exception& ex) {
        gguf_free(gctx);
        summary_.error = std::string("llama_model_init_from_user threw: ") + ex.what();
        return summary_;
    }
    gguf_free(gctx);
    if (!model_) {
        summary_.error = "llama_model_init_from_user returned nullptr";
        return summary_;
    }

    // Free initial unbuffered staging bounce buffer before running inference
    if (init_bounce_) {
        tf_aligned_free(init_bounce_);
        init_bounce_ = nullptr;
        init_bounce_bytes_ = 0;
    }

    // Phase 7 KV store initialization for prefix caching
    if (!cfg_.prefix_cache_dir.empty()) {
        KvConfig kcfg{};
        kcfg.n_layers   = static_cast<uint32_t>(n_layer);
        kcfg.n_kv_heads = static_cast<uint32_t>(n_head_kv);
        kcfg.head_dim_k = static_cast<uint32_t>(head_dim);
        kcfg.head_dim_v = static_cast<uint32_t>(head_dim);
        kcfg.precision  = cfg_.kv_precision;
        kcfg.prefix_dir = cfg_.prefix_cache_dir;
        kv_store_       = std::make_unique<tf_kv_store>(kcfg, *gov_);
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                = cfg_.n_ctx;
    cparams.n_batch              = cfg_.n_batch;
    cparams.n_ubatch             = cfg_.n_batch;
    cparams.n_threads            = cfg_.n_threads;
    cparams.n_threads_batch      = cfg_.n_threads;
    cparams.type_k               = (cfg_.kv_precision == KvPrecision::F16) ? GGML_TYPE_F16 : GGML_TYPE_Q8_0;
    cparams.type_v               = (cfg_.kv_precision == KvPrecision::F16)
        ? GGML_TYPE_F16
        : (cfg_.kv_precision == KvPrecision::Q8_K_Q4_V ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0);
    cparams.flash_attn_type      = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (!streamed_names_.empty()) {
        cparams.cb_eval           = &tf_llama_stream::tf_eval_cb;
        cparams.cb_eval_user_data = this;
    }

    ctx_ = llama_init_from_model(model_, cparams);
    if (!ctx_) {
        summary_.error = "llama_init_from_model failed";
        return summary_;
    }

    summary_.ok           = true;
    summary_.load_time_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    return summary_;
}

StreamPromptResult tf_llama_stream::run_prompt(const std::string& prompt,
                                               int32_t n_predict,
                                               float temp,
                                               uint32_t seed,
                                               bool save_or_use_prefix_cache) {
    StreamPromptResult out{};
    if (!ctx_ || !model_) {
        out.error = "Engine not initialized";
        return out;
    }

    llama_memory_clear(llama_get_memory(ctx_), true);
    run_hits_              = 0;
    run_misses_            = 0;
    run_bytes_ssd_         = 0;
    run_prefetch_sub_      = 0;
    run_prefetch_hits_     = 0;
    run_prefetch_wasted_b_ = 0;

    const llama_vocab* vocab = llama_model_get_vocab(model_);
    const int32_t n_vocab    = llama_vocab_n_tokens(vocab);

    int32_t n_prompt = -llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), nullptr, 0, true, true);
    if (n_prompt <= 0) {
        out.error = "Failed to tokenize prompt";
        return out;
    }
    std::vector<llama_token> tokens(static_cast<size_t>(n_prompt));
    if (llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                       tokens.data(), static_cast<int32_t>(tokens.size()), true, true) < 0) {
        out.error = "llama_tokenize failed";
        return out;
    }
    out.n_prompt_tokens = n_prompt;

    const uint64_t prefix_hash = (n_prompt >= 2)
        ? fnv1a_64_tokens(std::vector<llama_token>(tokens.begin(), tokens.end() - 1))
        : fnv1a_64_tokens(tokens);
    const auto t_prompt_start = Clock::now();
    bool restored_from_prefix = false;
    if (save_or_use_prefix_cache && kv_store_ && n_prompt >= 2) {
        std::vector<uint8_t> cached_state;
        if (kv_store_->load_prefix_block(prefix_hash, cached_state)) {
            if (llama_state_seq_set_data(ctx_, cached_state.data(), cached_state.size(), 0) > 0) {
                restored_from_prefix = true;
                out.prefix_cache_hit = true;
            }
        }
    }

    auto extract_top_logits = [&](int32_t idx) {
        const float* logits = llama_get_logits_ith(ctx_, idx);
        if (!logits) return;
        std::vector<std::pair<float, int32_t>> pairs;
        pairs.reserve(static_cast<size_t>(n_vocab));
        for (int32_t v = 0; v < n_vocab; ++v) {
            pairs.emplace_back(logits[v], v);
        }
        const size_t k_top = std::min<size_t>(16, pairs.size());
        std::partial_sort(pairs.begin(), pairs.begin() + k_top, pairs.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        out.first_step_top_logits.clear();
        out.first_step_top_ids.clear();
        for (size_t i = 0; i < k_top; ++i) {
            out.first_step_top_logits.push_back(pairs[i].first);
            out.first_step_top_ids.push_back(pairs[i].second);
        }
    };

    auto greedy_from_logits = [&](int32_t idx) -> llama_token {
        const float* logits = llama_get_logits_ith(ctx_, idx);
        int32_t best_id = 0;
        float best_val = logits[0];
        for (int32_t v = 1; v < n_vocab; ++v) {
            if (logits[v] > best_val) {
                best_val = logits[v];
                best_id  = v;
            }
        }
        return best_id;
    };

    llama_batch batch = llama_batch_init(std::max<int32_t>(cfg_.n_batch, n_prompt), 0, 1);

    int32_t n_cur = n_prompt;
    llama_token next_tok = LLAMA_TOKEN_NULL;
    if (!restored_from_prefix) {
        // Chunked prefill respecting n_batch; if saving prefix cache & n_prompt >= 2, save state before last token
        const int32_t first_part = (save_or_use_prefix_cache && kv_store_ && n_prompt >= 2) ? (n_prompt - 1) : n_prompt;
        int32_t processed = 0;
        while (processed < first_part) {
            const int32_t chunk = std::min<int32_t>(cfg_.n_batch, first_part - processed);
            batch.n_tokens = chunk;
            for (int32_t i = 0; i < chunk; ++i) {
                batch.token[i]     = tokens[static_cast<size_t>(processed + i)];
                batch.pos[i]       = processed + i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = ((processed + i) == n_prompt - 1) ? 1 : 0;
            }
            if (llama_decode(ctx_, batch) != 0) {
                llama_batch_free(batch);
                out.error = "llama_decode prefill failed";
                return out;
            }
            processed += chunk;
        }
        if (first_part < n_prompt) {
            const size_t sz = llama_state_seq_get_size(ctx_, 0);
            if (sz > 0) {
                std::vector<uint8_t> blob(sz);
                const size_t written = llama_state_seq_get_data(ctx_, blob.data(), sz, 0);
                if (written > 0) {
                    blob.resize(written);
                    kv_store_->save_prefix_block(prefix_hash, blob.data(), blob.size());
                }
            }
            batch.n_tokens     = 1;
            batch.token[0]     = tokens.back();
            batch.pos[0]       = n_prompt - 1;
            batch.n_seq_id[0]  = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0]    = 1;
            if (llama_decode(ctx_, batch) != 0) {
                llama_batch_free(batch);
                out.error = "llama_decode final prefill token failed";
                return out;
            }
        }
        extract_top_logits(batch.n_tokens - 1);
        next_tok = greedy_from_logits(batch.n_tokens - 1);
    } else {
        // Warm prefix restored for [0 .. n_prompt-2]; evaluate final prompt token to obtain live logits
        batch.n_tokens     = 1;
        batch.token[0]     = tokens.back();
        batch.pos[0]       = n_prompt - 1;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = 1;
        if (llama_decode(ctx_, batch) != 0) {
            llama_batch_free(batch);
            out.error = "llama_decode warm-prefix tail token failed";
            return out;
        }
        extract_top_logits(0);
        next_tok = greedy_from_logits(0);
    }

    const auto t_after_prefill = Clock::now();
    out.ttft_ms = std::chrono::duration<double, std::milli>(t_after_prefill - t_prompt_start).count();
    if (out.ttft_ms > 0.0) {
        out.prefill_tok_s = (1000.0 * n_prompt) / out.ttft_ms;
    }

    // Sampling helper for temp > 0 comparison if requested
    std::mt19937 rng(seed);
    auto sample_token = [&](int32_t idx) -> llama_token {
        if (temp <= 0.0f) return greedy_from_logits(idx);
        const float* logits = llama_get_logits_ith(ctx_, idx);
        double max_l = logits[0];
        for (int32_t v = 1; v < n_vocab; ++v) {
            if (logits[v] > max_l) max_l = logits[v];
        }
        std::vector<double> probs(static_cast<size_t>(n_vocab));
        double sum = 0.0;
        for (int32_t v = 0; v < n_vocab; ++v) {
            double p = std::exp((logits[v] - max_l) / temp);
            probs[static_cast<size_t>(v)] = p;
            sum += p;
        }
        std::uniform_real_distribution<double> u01(0.0, sum);
        double target = u01(rng);
        double acc = 0.0;
        for (int32_t v = 0; v < n_vocab; ++v) {
            acc += probs[static_cast<size_t>(v)];
            if (acc >= target) return v;
        }
        return n_vocab - 1;
    };

    // Note: if temp > 0, first token should match temp sampling
    const int32_t last_prefill_logit_idx = (restored_from_prefix || (save_or_use_prefix_cache && kv_store_ && n_prompt >= 2))
        ? 0
        : (batch.n_tokens - 1);
    if (temp > 0.0f) {
        next_tok = sample_token(last_prefill_logit_idx);
    }

    auto softmax_probs = [&](int32_t idx, std::vector<double>& probs) {
        const float* logits = llama_get_logits_ith(ctx_, idx);
        double max_l = logits[0];
        for (int32_t v = 1; v < n_vocab; ++v) {
            if (logits[v] > max_l) max_l = logits[v];
        }
        probs.resize(static_cast<size_t>(n_vocab));
        double sum = 0.0;
        for (int32_t v = 0; v < n_vocab; ++v) {
            const double p = std::exp((logits[v] - max_l) / temp);
            probs[static_cast<size_t>(v)] = p;
            sum += p;
        }
        const double inv = (sum > 0.0) ? (1.0 / sum) : 0.0;
        for (int32_t v = 0; v < n_vocab; ++v) {
            probs[static_cast<size_t>(v)] *= inv;
        }
    };

    std::vector<llama_token> full_history = tokens;
    const auto t_dec_start = Clock::now();
    int32_t generated = 0;

    // Speculative controller state (Phase 6)
    SpecSchedulerConfig scfg{};
    scfg.k_max = 8;
    tf_spec_scheduler spec_sched(scfg);
    uint64_t spec_draft_total = 0;
    uint64_t spec_accept_total = 0;
    uint64_t spec_steps = 0;
    uint64_t spec_k_sum = 0;

    if (n_predict > 0) {
        if (llama_vocab_is_eog(vocab, next_tok)) {
            out.generated_tokens.push_back(next_tok);
            generated = 1;
        } else {
            out.generated_tokens.push_back(next_tok);
            full_history.push_back(next_tok);
            generated = 1;
        }
    }

    while (generated < n_predict && !llama_vocab_is_eog(vocab, out.generated_tokens.back())) {
        const llama_token cur_tok = out.generated_tokens.back();
        if (cfg_.spec_k_mode == SpecKMode::Off) {
            // Non-speculative single-token decode (greedy or baseline stochastic)
            batch.n_tokens     = 1;
            batch.token[0]     = cur_tok;
            batch.pos[0]       = n_cur;
            batch.n_seq_id[0]  = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0]    = 1;
            if (llama_decode(ctx_, batch) != 0) {
                llama_batch_free(batch);
                out.error = "llama_decode step failed";
                return out;
            }
            ++n_cur;
            const llama_token nxt = (temp > 0.0f) ? sample_token(0) : greedy_from_logits(0);
            out.generated_tokens.push_back(nxt);
            full_history.push_back(nxt);
            ++generated;
        } else {
            // Phase 6 Speculative Verification with union-aware / ablation k-selection
            int32_t k_target = 0;
            if (cfg_.spec_k_mode == SpecKMode::Fixed4) {
                k_target = 4;
            } else {
                SpecInputs sin{};
                sin.alpha_hat    = spec_sched.alpha_ema();
                const double ram_gb = static_cast<double>(summary_.ram_weights_bytes) / 1e9;
                const double n_stream_f = static_cast<double>(summary_.n_streamed_tensors);
                sin.c_always_sec = ram_gb / std::max(1.0, probe_.cpu_gemv_effective_gbps);
                if (summary_.ssd_stream_bytes > 0 && summary_.n_streamed_tensors > 0) {
                    const double mean_obj_gb = (static_cast<double>(summary_.ssd_stream_bytes) / n_stream_f) / 1e9;
                    sin.cost_per_expert_sec  = mean_obj_gb / std::max(0.1, probe_.bw_ssd_sustained_gbps);
                } else {
                    sin.cost_per_expert_sec  = 0.0002;
                }
                sin.t_draft_sec = 0.0001; // prompt-lookup microsecond draft cost
                sin.u_total_by_k.resize(static_cast<size_t>(scfg.k_max + 1));
                const double base_u = std::max(1.0, n_stream_f);
                for (int k = 0; k <= scfg.k_max; ++k) {
                    if (cfg_.spec_k_mode == SpecKMode::NaiveNoUnion) {
                        // Naive model ignores batch tensor sharing across draft steps
                        sin.u_total_by_k[static_cast<size_t>(k)] = base_u * static_cast<double>(k + 1);
                    } else {
                        // Union-aware: batched forward loads each active tensor at most once per step
                        sin.u_total_by_k[static_cast<size_t>(k)] = base_u * (1.0 + 0.02 * static_cast<double>(k));
                    }
                }
                k_target = spec_sched.choose_k(sin).k_opt;
            }
            k_target = std::min<int32_t>(k_target, n_predict - generated - 1);
            auto draft = prompt_lookup_draft(full_history, k_target);
            const int32_t k_act = static_cast<int32_t>(draft.size());
            ++spec_steps;
            spec_k_sum += static_cast<uint64_t>(k_act);

            batch.n_tokens = 1 + k_act;
            batch.token[0]     = cur_tok;
            batch.pos[0]       = n_cur;
            batch.n_seq_id[0]  = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0]    = 1;
            for (int32_t i = 0; i < k_act; ++i) {
                batch.token[1 + i]     = draft[static_cast<size_t>(i)];
                batch.pos[1 + i]       = n_cur + 1 + i;
                batch.n_seq_id[1 + i]  = 1;
                batch.seq_id[1 + i][0] = 0;
                batch.logits[1 + i]    = 1;
            }
            if (llama_decode(ctx_, batch) != 0) {
                llama_batch_free(batch);
                out.error = "llama_decode speculative verify failed";
                return out;
            }

            int32_t accepted = 0;
            bool rejected_early = false;
            if (temp <= 0.0f) {
                llama_token final_tok = greedy_from_logits(0);
                for (int32_t i = 0; i < k_act; ++i) {
                    if (final_tok == draft[static_cast<size_t>(i)]) {
                        out.generated_tokens.push_back(final_tok);
                        full_history.push_back(final_tok);
                        ++generated;
                        ++accepted;
                        if (generated >= n_predict || llama_vocab_is_eog(vocab, final_tok)) {
                            rejected_early = true;
                            break;
                        }
                        final_tok = greedy_from_logits(i + 1);
                    } else {
                        out.generated_tokens.push_back(final_tok);
                        full_history.push_back(final_tok);
                        ++generated;
                        rejected_early = true;
                        break;
                    }
                }
                if (!rejected_early && generated < n_predict &&
                    (out.generated_tokens.empty() || !llama_vocab_is_eog(vocab, out.generated_tokens.back()))) {
                    out.generated_tokens.push_back(final_tok);
                    full_history.push_back(final_tok);
                    ++generated;
                }
            } else {
                // Exact Leviathan speculative rejection sampling for deterministic prompt-lookup q(x)=1[x=d_i]
                std::vector<double> probs;
                std::uniform_real_distribution<double> u01(0.0, 1.0);
                for (int32_t i = 0; i < k_act; ++i) {
                    softmax_probs(i, probs);
                    const llama_token d = draft[static_cast<size_t>(i)];
                    const double p_d = probs[static_cast<size_t>(d)];
                    if (u01(rng) < p_d) {
                        out.generated_tokens.push_back(d);
                        full_history.push_back(d);
                        ++generated;
                        ++accepted;
                        if (generated >= n_predict || llama_vocab_is_eog(vocab, d)) {
                            rejected_early = true;
                            break;
                        }
                    } else {
                        const double rem = std::max(0.0, 1.0 - p_d);
                        llama_token bonus = 0;
                        if (rem > 1e-12) {
                            double target = u01(rng) * rem;
                            double acc = 0.0;
                            for (int32_t v = 0; v < n_vocab; ++v) {
                                if (v == d) continue;
                                acc += probs[static_cast<size_t>(v)];
                                if (acc >= target) {
                                    bonus = v;
                                    break;
                                }
                            }
                            if (acc < target) {
                                bonus = (d == n_vocab - 1) ? (n_vocab - 2) : (n_vocab - 1);
                            }
                        } else {
                            bonus = d;
                        }
                        out.generated_tokens.push_back(bonus);
                        full_history.push_back(bonus);
                        ++generated;
                        rejected_early = true;
                        break;
                    }
                }
                if (!rejected_early && generated < n_predict &&
                    (out.generated_tokens.empty() || !llama_vocab_is_eog(vocab, out.generated_tokens.back()))) {
                    const llama_token final_tok = sample_token(k_act);
                    out.generated_tokens.push_back(final_tok);
                    full_history.push_back(final_tok);
                    ++generated;
                }
            }
            const int32_t new_pos_end = n_cur + 1 + accepted;
            if (accepted < k_act) {
                llama_memory_seq_rm(llama_get_memory(ctx_), 0, new_pos_end, -1);
            }
            n_cur = new_pos_end;
            if (k_act > 0) {
                spec_draft_total  += static_cast<uint64_t>(k_act);
                spec_accept_total += static_cast<uint64_t>(accepted);
                spec_sched.update_ema(k_act, accepted + 1, static_cast<double>(summary_.n_streamed_tensors));
            }
        }
    }

    drain_all_async();
    llama_batch_free(batch);

    const double dec_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_dec_start).count();
    out.n_generated_tokens = static_cast<int32_t>(out.generated_tokens.size());
    const int32_t dec_tokens = std::max<int32_t>(0, out.n_generated_tokens - 1);
    out.decode_tok_s = (dec_ms > 0.0 && dec_tokens > 0) ? (1000.0 * dec_tokens) / dec_ms : 0.0;

    for (llama_token id : out.generated_tokens) {
        char buf[256];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n > 0) out.generated_text.append(buf, static_cast<size_t>(n));
    }

    out.cache_hits               = run_hits_;
    out.cache_misses             = run_misses_;
    out.hit_rate_ram             = (run_hits_ + run_misses_ > 0)
        ? static_cast<double>(run_hits_) / static_cast<double>(run_hits_ + run_misses_)
        : 1.0;
    out.bytes_read_ssd           = run_bytes_ssd_;
    out.bytes_read_ssd_per_token = (out.n_generated_tokens > 0)
        ? static_cast<double>(run_bytes_ssd_) / static_cast<double>(out.n_generated_tokens)
        : 0.0;
    out.prefetch_hits            = run_prefetch_hits_;
    out.prefetch_submitted       = run_prefetch_sub_;
    out.prefetch_wasted_bytes    = run_prefetch_wasted_b_;
    out.spec_k_mean              = (spec_steps > 0)
        ? static_cast<double>(spec_k_sum) / static_cast<double>(spec_steps)
        : 0.0;
    out.spec_accept_rate         = (spec_draft_total > 0)
        ? static_cast<double>(spec_accept_total) / static_cast<double>(spec_draft_total)
        : 0.0;
    out.ok = true;
    return out;
}

} // namespace tierflow
