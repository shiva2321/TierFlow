#include "tf_llama_stream.h"
#include "tf_compute_place.h"
#include "tf_draft.h"
#include "tf_pool_plan.h"

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
    draft_.reset();
    for (void* p : current_moe_overflow_ptrs_) {
        if (p) tf_aligned_free(p);
    }
    current_moe_overflow_ptrs_.clear();
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
    for (auto& a : arenas_) {
        a.cache.reset();
        if (a.mem) {
            tf_aligned_free(a.mem);
            a.mem = nullptr;
        }
        a.mem_bytes = 0;
        a.n_slots = 0;
    }
    if (init_bounce_) {
        tf_aligned_free(init_bounce_);
        init_bounce_ = nullptr;
    }
    kv_store_.reset();
    gov_.reset();
}

uint8_t* tf_llama_stream::slot_ptr(int32_t packed_slot) const {
    if (packed_slot < 0) return nullptr;
    const int c = slot_class(packed_slot);
    const int32_t ls = slot_local(packed_slot);
    if (c < 0 || c >= static_cast<int>(kNumArenaClasses)) return nullptr;
    const auto& a = arenas_[static_cast<size_t>(c)];
    if (!a.mem || ls < 0 || static_cast<size_t>(ls) >= a.n_slots) return nullptr;
    return a.slot_ptr(ls);
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

    auto mit = self->parent_to_moe_idx_.find(name);
    if (mit != self->parent_to_moe_idx_.end()) {
        const auto& mp = self->moe_parents_[mit->second];
        if (self->streamed_names_.count(name) > 0) {
            tensor->data = self->arenas_[static_cast<size_t>(mp.class_idx)].slot_ptr(0);
            self->tensor_to_moe_parent_[tensor] = mp.parent_idx;
            return;
        }
        if (!tensor->data && (!tensor->buffer || ggml_backend_buffer_is_host(tensor->buffer))) return;
        const size_t n_exp = std::max<size_t>(1, mp.n_experts);
        const size_t expert_stride = (tensor->nb[2] > 0)
            ? static_cast<size_t>(tensor->nb[2])
            : (ggml_nbytes(tensor) / n_exp);
        const bool is_device = (tensor->buffer && !ggml_backend_buffer_is_host(tensor->buffer));
        for (int32_t e = 0; e < mp.n_experts; ++e) {
            const uint32_t obj_id = mp.slice_obj_ids[static_cast<size_t>(e)];
            const auto& obj = self->manifest_.objects[obj_id];
            const size_t to_copy = std::min<size_t>(expert_stride, obj.exact_bytes);
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
                throw std::runtime_error("tf_io init submit failed for MoE slice " + obj.name);
            }
            tf_read_req done{};
            while (self->io_->wait(&done, 1, 5) == 0) {
            }
            const size_t dst_offs = static_cast<size_t>(e) * expert_stride;
            if (is_device) {
                ggml_backend_tensor_set(tensor, self->init_bounce_, dst_offs, to_copy);
            } else {
                std::memcpy(static_cast<uint8_t*>(tensor->data) + dst_offs, self->init_bounce_, to_copy);
            }
        }
        self->loaded_ram_names_.insert(name);
        return;
    }

    auto it = self->name_to_obj_id_.find(name);
    if (it == self->name_to_obj_id_.end()) return;
    const uint32_t obj_id = it->second;
    if (self->streamed_names_.count(name) > 0) {
        // Streamed tensor resides in 0-byte dummy buffer; give non-null sentinel pointer inside the
        // governor-reserved size-class arena so ggml_gallocr skips allocation while cb_eval swaps real slots.
        auto s_it = self->obj_id_to_sidx_.find(obj_id);
        if (s_it != self->obj_id_to_sidx_.end()) {
            const int c = self->streamed_objs_[s_it->second].class_idx;
            tensor->data = self->arenas_[static_cast<size_t>(c)].slot_ptr(0);
            self->streamed_objs_[s_it->second].tensor = tensor;
            self->tensor_to_obj_id_[tensor] = obj_id;
        }
        return;
    }
    if (!tensor->data && (!tensor->buffer || ggml_backend_buffer_is_host(tensor->buffer))) return;

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
    while (self->io_->wait(&done, 1, 5) == 0) {
    }
    if (tensor->buffer && !ggml_backend_buffer_is_host(tensor->buffer)) {
        ggml_backend_tensor_set(tensor, self->init_bounce_, 0, to_copy);
    } else {
        std::memcpy(tensor->data, self->init_bounce_, to_copy);
    }
    self->loaded_ram_names_.insert(name);
}

void tf_llama_stream::finish_async_cqe(uint64_t tag, std::chrono::steady_clock::time_point now_tp) {
    auto it = inflight_by_tag_.find(tag);
    if (it == inflight_by_tag_.end()) return;
    PendingIo pio = it->second;
    inflight_by_tag_.erase(it);
    const double dt_s = std::chrono::duration<double>(now_tp - pio.submit_tp).count();
    io_bw_ewma_gbps_ = update_bw_ewma(io_bw_ewma_gbps_, pio.bytes, dt_s);
    auto oit = inflight_tag_by_obj_.find(pio.object_id);
    if (oit != inflight_tag_by_obj_.end() && oit->second == pio.tag) {
        inflight_tag_by_obj_.erase(oit);
    }
    if (pio.slot_id >= 0) {
        const int c = slot_class(pio.slot_id);
        const int32_t ls = slot_local(pio.slot_id);
        if (c >= 0 && c < static_cast<int>(kNumArenaClasses)) {
            auto& a = arenas_[static_cast<size_t>(c)];
            if (ls >= 0 && static_cast<size_t>(ls) < a.prefetch_loading.size()) {
                a.prefetch_loading[static_cast<size_t>(ls)] = 0;
                a.cache->release_slot(ls);
            }
        }
    }
}

void tf_llama_stream::poll_completed_async(bool block_one) {
    tf_read_req done[16];
    int n = (block_one && !inflight_by_tag_.empty())
        ? io_->wait(done, 16, 1)
        : io_->poll(done, 16);
    while (block_one && n == 0 && !inflight_by_tag_.empty()) {
        n = io_->wait(done, 16, 1);
    }
    const auto now_tp = Clock::now();
    for (int i = 0; i < n; ++i) {
        finish_async_cqe(done[i].tag, now_tp);
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

    const int c = streamed_objs_[obj_id_to_sidx_.at(obj_id)].class_idx;
    auto& arena = arenas_[static_cast<size_t>(c)];

    auto res = arena.cache->access(obj_id);
    if (res.slot_id < 0) {
        // Fallback: drain async prefetches to free unreferenced Probation slots and retry
        drain_all_async();
        res = arena.cache->access(obj_id);
    }
    if (res.slot_id < 0) {
        throw std::runtime_error("tf_cache out of evictable slots during streaming");
    }
    if (res.evicted) {
        eam_prefetched_unused_objs_.erase(res.evicted_object_id);
        if (prefetched_unused_objs_.erase(res.evicted_object_id) > 0) {
            auto s_it = obj_id_to_sidx_.find(res.evicted_object_id);
            if (s_it != obj_id_to_sidx_.end()) {
                run_prefetch_wasted_b_ += streamed_objs_[s_it->second].page_aligned_bytes;
            }
        }
    }

    arena.cache->acquire_slot(res.slot_id);
    const int32_t packed_slot = pack_slot(c, res.slot_id);

    if (res.hit) {
        ++run_hits_;
        if (prefetched_unused_objs_.erase(obj_id) > 0) {
            ++run_prefetch_hits_;
        }
        if (eam_prefetched_unused_objs_.erase(obj_id) > 0) {
            ++run_eam_prefetch_hits_;
        }
    } else {
        ++run_misses_;
        const auto& obj = manifest_.objects[obj_id];
        const uint64_t tag = next_tag_++;
        const auto sub_tp = Clock::now();
        tf_read_req req{};
        req.object_id = obj_id;
        req.offset    = obj.tfpack_offset;
        req.size      = obj.page_aligned_bytes;
        req.dst       = arena.slot_ptr(res.slot_id);
        req.priority  = TF_PRIO_DEMAND;
        req.tag       = tag;
        if (!io_->submit(req)) {
            poll_completed_async(true);
            if (!io_->submit(req)) {
                arena.cache->release_slot(res.slot_id);
                throw std::runtime_error("tf_io demand submit failed");
            }
        }
        bool done_flag = false;
        while (!done_flag) {
            tf_read_req done[8];
            int n = io_->wait(done, 8, 1);
            const auto now_tp = Clock::now();
            for (int i = 0; i < n; ++i) {
                if (done[i].tag == tag) {
                    const double dt_s = std::chrono::duration<double>(now_tp - sub_tp).count();
                    io_bw_ewma_gbps_ = update_bw_ewma(io_bw_ewma_gbps_, obj.page_aligned_bytes, dt_s);
                    done_flag = true;
                } else {
                    finish_async_cqe(done[i].tag, now_tp);
                }
            }
        }
        run_bytes_ssd_ += obj.page_aligned_bytes;
    }

    (void)order_idx;
    return packed_slot;
}

void tf_llama_stream::submit_prefetch_obj(uint32_t target_obj, bool is_eam) {
    if (!io_ || cfg_.prefetch_mode == PrefetchMode::Off) return;
    if (prefetch_inflight_cap_ == 0 || inflight_by_tag_.size() >= prefetch_inflight_cap_) return;
    auto s_target_it = obj_id_to_sidx_.find(target_obj);
    if (s_target_it == obj_id_to_sidx_.end()) return;
    const int c = streamed_objs_[s_target_it->second].class_idx;
    auto& arena = arenas_[static_cast<size_t>(c)];
    if (!arena.cache || arena.n_slots < 3) return;
    if (arena.cache->is_resident(target_obj) || inflight_tag_by_obj_.count(target_obj) > 0) {
        return;
    }
    auto pres = arena.cache->prefetch(target_obj);
    if (pres.slot_id < 0) return;
    if (pres.evicted) {
        eam_prefetched_unused_objs_.erase(pres.evicted_object_id);
        if (prefetched_unused_objs_.erase(pres.evicted_object_id) > 0) {
            auto s_it = obj_id_to_sidx_.find(pres.evicted_object_id);
            if (s_it != obj_id_to_sidx_.end()) {
                run_prefetch_wasted_b_ += streamed_objs_[s_it->second].page_aligned_bytes;
            }
        }
    }
    arena.cache->acquire_slot(pres.slot_id);
    arena.prefetch_loading[static_cast<size_t>(pres.slot_id)] = 1;
    const int32_t packed_slot = pack_slot(c, pres.slot_id);
    const auto& obj = manifest_.objects[target_obj];
    const uint64_t tag = next_tag_++;
    tf_read_req req{};
    req.object_id = target_obj;
    req.offset    = obj.tfpack_offset;
    req.size      = obj.page_aligned_bytes;
    req.dst       = arena.slot_ptr(pres.slot_id);
    req.priority  = TF_PRIO_PREFETCH;
    req.tag       = tag;
    if (!io_->submit(req)) {
        arena.prefetch_loading[static_cast<size_t>(pres.slot_id)] = 0;
        arena.cache->release_slot(pres.slot_id);
        return;
    }
    PendingIo pio{target_obj, packed_slot, tag, true, obj.page_aligned_bytes, Clock::now()};
    inflight_by_tag_[tag]            = pio;
    inflight_tag_by_obj_[target_obj] = tag;
    prefetched_unused_objs_.insert(target_obj);
    ++run_prefetch_sub_;
    if (is_eam) {
        eam_prefetched_unused_objs_.insert(target_obj);
        ++run_eam_prefetch_sub_;
    }
    run_bytes_ssd_ += obj.page_aligned_bytes;
}

void tf_llama_stream::issue_lookahead_prefetch(int32_t current_order_idx) {
    if (cfg_.prefetch_mode == PrefetchMode::Off || exec_order_.empty() || !io_) return;
    if (prefetch_inflight_cap_ == 0 || inflight_by_tag_.size() >= prefetch_inflight_cap_) return;

    const int32_t depth = std::max<int32_t>(1, cfg_.lookahead_depth);
    const size_t total = exec_order_.size();
    for (int32_t d = 1; d <= depth; ++d) {
        if (inflight_by_tag_.size() >= prefetch_inflight_cap_) break;
        const size_t idx = (static_cast<size_t>(current_order_idx) + static_cast<size_t>(d)) % total;
        uint32_t target_obj = exec_order_[idx];
        if (cfg_.prefetch_mode == PrefetchMode::Shuffled && !shuffled_order_.empty()) {
            target_obj = shuffled_order_[idx % shuffled_order_.size()];
        }
        submit_prefetch_obj(target_obj, false);
    }
}

bool tf_llama_stream::tf_prepare_experts_cb(const ggml_tensor* src,
                                            const int32_t* expert_ids,
                                            int64_t n_ids,
                                            void* ud) {
    auto* self = static_cast<tf_llama_stream*>(ud);
    return self ? self->prepare_moe_experts(src, expert_ids, n_ids) : false;
}

const void* tf_llama_stream::tf_resolve_expert_cb(const ggml_tensor* src,
                                                  int32_t expert_idx,
                                                  size_t* out_bytes,
                                                  void* ud) {
    auto* self = static_cast<tf_llama_stream*>(ud);
    return self ? self->resolve_moe_expert(src, expert_idx, out_bytes) : nullptr;
}

void tf_llama_stream::tf_release_experts_cb(const ggml_tensor* src, void* ud) {
    auto* self = static_cast<tf_llama_stream*>(ud);
    if (self) self->release_moe_experts(src);
}

bool tf_llama_stream::prepare_moe_experts(const ggml_tensor* src,
                                          const int32_t* expert_ids,
                                          int64_t n_ids) {
    if (!src) return false;
    const ggml_tensor* base = src->view_src ? src->view_src : src;
    auto mit = tensor_to_moe_parent_.find(base);
    if (mit == tensor_to_moe_parent_.end()) {
        return false;
    }

    release_moe_experts(base);
    const auto& mp = moe_parents_[mit->second];
    active_moe_parent_ = static_cast<int32_t>(mp.parent_idx);
    current_moe_slices_.assign(static_cast<size_t>(mp.n_experts), nullptr);

    std::vector<int32_t> uniq_ids;
    uniq_ids.reserve(static_cast<size_t>(std::max<int64_t>(0, n_ids)));
    std::vector<uint8_t> seen(static_cast<size_t>(mp.n_experts), 0);
    for (int64_t i = 0; i < n_ids; ++i) {
        const int32_t eid = expert_ids[i];
        if (eid >= 0 && eid < mp.n_experts && !seen[static_cast<size_t>(eid)]) {
            seen[static_cast<size_t>(eid)] = 1;
            uniq_ids.push_back(eid);
        }
    }

    if (uniq_ids.empty()) {
        return true;
    }

    // Wait for any requested experts that are currently in-flight from prefetch
    for (int32_t eid : uniq_ids) {
        const uint32_t obj_id = mp.slice_obj_ids[static_cast<size_t>(eid)];
        auto inf_it = inflight_tag_by_obj_.find(obj_id);
        while (inf_it != inflight_tag_by_obj_.end() && inflight_by_tag_.count(inf_it->second) > 0) {
            poll_completed_async(true);
            inf_it = inflight_tag_by_obj_.find(obj_id);
        }
    }

    auto& arena = arenas_[static_cast<size_t>(mp.class_idx)];
    struct PendingDemand {
        int32_t                               expert_idx = 0;
        uint32_t                              obj_id     = 0;
        uint64_t                              bytes      = 0;
        std::chrono::steady_clock::time_point submit_tp{};
    };
    std::unordered_map<uint64_t, PendingDemand> pending_demands;
    pending_demands.reserve(uniq_ids.size());

    auto reap_demand_or_prefetch = [&](bool block_one) {
        tf_read_req done[16];
        int n = block_one ? io_->wait(done, 16, 1) : io_->poll(done, 16);
        while (block_one && n == 0 && (!pending_demands.empty() || !inflight_by_tag_.empty())) {
            n = io_->wait(done, 16, 1);
        }
        const auto now_tp = Clock::now();
        for (int i = 0; i < n; ++i) {
            auto dit = pending_demands.find(done[i].tag);
            if (dit != pending_demands.end()) {
                const double dt_s = std::chrono::duration<double>(now_tp - dit->second.submit_tp).count();
                io_bw_ewma_gbps_ = update_bw_ewma(io_bw_ewma_gbps_, dit->second.bytes, dt_s);
                run_bytes_ssd_ += dit->second.bytes;
                pending_demands.erase(dit);
            } else {
                finish_async_cqe(done[i].tag, now_tp);
            }
        }
    };

    for (int32_t eid : uniq_ids) {
        const uint32_t obj_id = mp.slice_obj_ids[static_cast<size_t>(eid)];
        const auto& obj = manifest_.objects[obj_id];
        auto res = arena.cache ? arena.cache->access(obj_id) : CacheAccessResult{};
        if (res.slot_id < 0 && arena.cache) {
            drain_all_async();
            res = arena.cache->access(obj_id);
        }

        uint8_t* dst_ptr = nullptr;
        if (res.slot_id >= 0) {
            if (res.evicted) {
                eam_prefetched_unused_objs_.erase(res.evicted_object_id);
                if (prefetched_unused_objs_.erase(res.evicted_object_id) > 0) {
                    auto s_it = obj_id_to_sidx_.find(res.evicted_object_id);
                    if (s_it != obj_id_to_sidx_.end()) {
                        run_prefetch_wasted_b_ += streamed_objs_[s_it->second].page_aligned_bytes;
                    }
                }
            }
            arena.cache->acquire_slot(res.slot_id);
            current_moe_pinned_slots_.push_back(pack_slot(mp.class_idx, res.slot_id));
            dst_ptr = arena.slot_ptr(res.slot_id);

            auto sit = obj_id_to_sidx_.find(obj_id);
            if (sit != obj_id_to_sidx_.end()) {
                ++streamed_objs_[sit->second].access_count;
                if (cfg_.repin_enabled &&
                    can_repin_slot(arena.n_slots, arena.repinned_objs.size(), prefetch_inflight_cap_) &&
                    arena.repinned_objs.count(obj_id) == 0 &&
                    streamed_objs_[sit->second].access_count >= 2 &&
                    !arena.prefetch_loading[static_cast<size_t>(res.slot_id)]) {
                    if (arena.cache->pin(obj_id)) {
                        arena.repinned_objs.insert(obj_id);
                    }
                }
            }
        } else {
            // Emergency aligned overflow buffer when prefill batch touches more active experts than arena slots
            void* obuf = tf_aligned_alloc(static_cast<size_t>(obj.page_aligned_bytes), 4096);
            if (!obuf) {
                release_moe_experts(base);
                throw std::runtime_error("tf_aligned_alloc overflow failed during MoE batch");
            }
            current_moe_overflow_ptrs_.push_back(obuf);
            dst_ptr = static_cast<uint8_t*>(obuf);
        }

        current_moe_slices_[static_cast<size_t>(eid)] = dst_ptr;
        if (res.slot_id >= 0 && res.hit) {
            ++run_hits_;
            if (prefetched_unused_objs_.erase(obj_id) > 0) {
                ++run_prefetch_hits_;
            }
            if (eam_prefetched_unused_objs_.erase(obj_id) > 0) {
                ++run_eam_prefetch_hits_;
            }
        } else {
            ++run_misses_;
            const uint64_t tag = next_tag_++;
            const auto sub_tp = Clock::now();
            tf_read_req req{};
            req.object_id = obj_id;
            req.offset    = obj.tfpack_offset;
            req.size      = obj.page_aligned_bytes;
            req.dst       = dst_ptr;
            req.priority  = TF_PRIO_DEMAND;
            req.tag       = tag;
            while (!io_->submit(req)) {
                if (pending_demands.empty() && inflight_by_tag_.empty()) {
                    release_moe_experts(base);
                    throw std::runtime_error("tf_io MoE demand submit failed");
                }
                reap_demand_or_prefetch(true);
            }
            pending_demands[tag] = PendingDemand{eid, obj_id, obj.page_aligned_bytes, sub_tp};
        }
    }

    // Overlapped MoE pipeline: once Gate demand reads are queued in NVMe, immediately enqueue
    // deterministic sibling Up/Down + high-confidence L+1 EAM prefetches before awaiting Gate
    // completions so all slices execute concurrently at deep NVMe queue depth.
    const bool overlap_prefetch = (arena.n_slots >= 32);
    if (overlap_prefetch) {
        issue_moe_prefetch(mp.layer, mp.kind, uniq_ids);
    }

    while (!pending_demands.empty()) {
        reap_demand_or_prefetch(true);
    }

    if (!overlap_prefetch) {
        issue_moe_prefetch(mp.layer, mp.kind, uniq_ids);
    }
    return true;
}

const void* tf_llama_stream::resolve_moe_expert(const ggml_tensor* src,
                                                int32_t expert_idx,
                                                size_t* out_bytes) const {
    (void)src;
    if (active_moe_parent_ < 0 || static_cast<size_t>(active_moe_parent_) >= moe_parents_.size()) {
        return nullptr;
    }
    const auto& mp = moe_parents_[static_cast<size_t>(active_moe_parent_)];
    if (expert_idx < 0 || expert_idx >= mp.n_experts) {
        return nullptr;
    }
    if (out_bytes) {
        const uint32_t obj_id = mp.slice_obj_ids[static_cast<size_t>(expert_idx)];
        *out_bytes = manifest_.objects[obj_id].exact_bytes;
    }
    return current_moe_slices_[static_cast<size_t>(expert_idx)];
}

void tf_llama_stream::release_moe_experts(const ggml_tensor* src) {
    (void)src;
    for (int32_t packed : current_moe_pinned_slots_) {
        const int c = slot_class(packed);
        const int32_t ls = slot_local(packed);
        if (c >= 0 && c < static_cast<int>(kNumArenaClasses) && arenas_[static_cast<size_t>(c)].cache) {
            arenas_[static_cast<size_t>(c)].cache->release_slot(ls);
        }
    }
    current_moe_pinned_slots_.clear();
    for (void* p : current_moe_overflow_ptrs_) {
        if (p) tf_aligned_free(p);
    }
    current_moe_overflow_ptrs_.clear();
    active_moe_parent_ = -1;
}

void tf_llama_stream::issue_moe_prefetch(int32_t layer,
                                         MoeProjKind kind,
                                         const std::vector<int32_t>& active_ids) {
    if (layer < 0 || layer >= moe_n_layer_ || moe_n_experts_ <= 0) return;

    auto perm_exp = [&](int32_t e, int32_t l) -> int32_t {
        return static_cast<int32_t>(
            (static_cast<uint64_t>(e) * 2654435761ULL + static_cast<uint64_t>(l + 1) * 1013904223ULL + 17ULL) %
            static_cast<uint64_t>(moe_n_experts_));
    };

    // Update online causal EAM counts once per layer step (on Gate)
    if (kind == MoeProjKind::Gate) {
        if (layer > 0 && static_cast<size_t>(layer - 1) < eam_counts_.size() &&
            !prev_layer_experts_[static_cast<size_t>(layer - 1)].empty()) {
            auto& tbl = eam_counts_[static_cast<size_t>(layer - 1)];
            auto& row_tots = eam_row_totals_[static_cast<size_t>(layer - 1)];
            bool need_halve = false;
            for (int32_t pe : prev_layer_experts_[static_cast<size_t>(layer - 1)]) {
                if (pe < 0 || pe >= moe_n_experts_) continue;
                ++row_tots[static_cast<size_t>(pe)];
                if (row_tots[static_cast<size_t>(pe)] >= 32768u) {
                    need_halve = true;
                }
                for (int32_t ce : active_ids) {
                    if (ce < 0 || ce >= moe_n_experts_) continue;
                    const size_t idx = static_cast<size_t>(pe) * static_cast<size_t>(moe_n_experts_) + static_cast<size_t>(ce);
                    if (tbl[idx] < 65535u) {
                        ++tbl[idx];
                    }
                    if (tbl[idx] >= 32768u) {
                        need_halve = true;
                    }
                }
            }
            if (need_halve) {
                for (uint16_t& v : tbl) v >>= 1;
                for (uint32_t& rt : row_tots) rt >>= 1;
            }
        }
        prev_layer_experts_[static_cast<size_t>(layer)] = active_ids;
    }

    if (cfg_.prefetch_mode == PrefetchMode::Off || !io_) return;
    if (prefetch_inflight_cap_ == 0 || inflight_by_tag_.size() >= prefetch_inflight_cap_) return;

    // Step A: deterministic intra-layer sibling prefetch (Gate -> Up + Down, or Up -> Down)
    auto queue_layer_proj = [&](int32_t target_layer, MoeProjKind target_kind, int32_t e, bool is_eam) {
        if (target_layer < 0 || target_layer >= moe_n_layer_) return;
        const int32_t pid = moe_layer_parent_[static_cast<size_t>(target_layer)][static_cast<size_t>(target_kind)];
        if (pid < 0) return;
        const auto& tmp = moe_parents_[static_cast<size_t>(pid)];
        if (e < 0 || e >= tmp.n_experts) return;
        const uint32_t obj_id = tmp.slice_obj_ids[static_cast<size_t>(e)];
        submit_prefetch_obj(obj_id, is_eam);
    };

    if (kind == MoeProjKind::Gate) {
        for (int32_t e : active_ids) {
            if (inflight_by_tag_.size() >= prefetch_inflight_cap_) break;
            const int32_t pe = (cfg_.prefetch_mode == PrefetchMode::Shuffled) ? perm_exp(e, layer) : e;
            queue_layer_proj(layer, MoeProjKind::Up, pe, false);
        }
        for (int32_t e : active_ids) {
            if (inflight_by_tag_.size() >= prefetch_inflight_cap_) break;
            const int32_t pe = (cfg_.prefetch_mode == PrefetchMode::Shuffled) ? perm_exp(e, layer + 100) : e;
            queue_layer_proj(layer, MoeProjKind::Down, pe, false);
        }
    } else if (kind == MoeProjKind::Up) {
        for (int32_t e : active_ids) {
            if (inflight_by_tag_.size() >= prefetch_inflight_cap_) break;
            const int32_t pe = (cfg_.prefetch_mode == PrefetchMode::Shuffled) ? perm_exp(e, layer + 100) : e;
            queue_layer_proj(layer, MoeProjKind::Down, pe, false);
        }
    }

    // Step B: empirical confidence-gated L -> L+1 EAM prefetch on Gate
    if (kind == MoeProjKind::Gate && (layer + 1) < moe_n_layer_ &&
        static_cast<size_t>(layer) < eam_counts_.size() &&
        inflight_by_tag_.size() < prefetch_inflight_cap_) {
        const auto& tbl = eam_counts_[static_cast<size_t>(layer)];
        const auto& row_tots = eam_row_totals_[static_cast<size_t>(layer)];
        std::vector<double> miss_prob(static_cast<size_t>(moe_n_experts_), 1.0);
        for (int32_t pe : active_ids) {
            if (pe < 0 || pe >= moe_n_experts_) continue;
            const uint32_t rt = row_tots[static_cast<size_t>(pe)];
            if (rt < 4u) continue;
            const double inv_rt = 1.0 / static_cast<double>(rt);
            const size_t base = static_cast<size_t>(pe) * static_cast<size_t>(moe_n_experts_);
            for (int32_t ne = 0; ne < moe_n_experts_; ++ne) {
                const uint16_t cnt = tbl[base + static_cast<size_t>(ne)];
                if (cnt == 0) continue;
                const double p = std::clamp(static_cast<double>(cnt) * inv_rt, 0.0, 1.0);
                miss_prob[static_cast<size_t>(ne)] *= (1.0 - p);
            }
        }
        std::vector<std::pair<double, int32_t>> cand;
        cand.reserve(static_cast<size_t>(moe_n_experts_));
        for (int32_t ne = 0; ne < moe_n_experts_; ++ne) {
            const double conf = 1.0 - miss_prob[static_cast<size_t>(ne)];
            if (conf >= 0.65) {
                cand.emplace_back(conf, ne);
            }
        }
        if (!cand.empty()) {
            const size_t top_k = std::min<size_t>(cand.size(), static_cast<size_t>(std::min<int32_t>(2, std::max<int32_t>(1, moe_n_used_))));
            std::partial_sort(cand.begin(), cand.begin() + top_k, cand.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first; });
            for (size_t i = 0; i < top_k; ++i) {
                if (inflight_by_tag_.size() >= prefetch_inflight_cap_) break;
                const int32_t ne = (cfg_.prefetch_mode == PrefetchMode::Shuffled)
                    ? perm_exp(cand[i].second, layer + 1)
                    : cand[i].second;
                queue_layer_proj(layer + 1, MoeProjKind::Gate, ne, true);
            }
        }
    }
}

bool tf_llama_stream::handle_eval_node(ggml_tensor* node, bool ask) {
    if (!node || streamed_names_.empty()) return false;

    const char* nname = ggml_get_name(node);
    const bool is_topk = (has_streamed_moe_ && nname && std::strncmp(nname, "ffn_moe_topk-", 13) == 0);
    const bool is_host_node = (!node->buffer || ggml_backend_buffer_is_host(node->buffer));

    if (ask) {
        if (is_topk && is_host_node) {
            return true;
        }
        if (has_streamed_moe_ && node->op == GGML_OP_MUL_MAT_ID && node->src[0]) {
            ggml_tensor* base0 = node->src[0]->view_src ? node->src[0]->view_src : node->src[0];
            auto mit = tensor_to_moe_parent_.find(base0);
            if (mit != tensor_to_moe_parent_.end()) {
                ggml_tensor* ids = node->src[2];
                if (ids && ids->type == GGML_TYPE_I32 && ids->data &&
                    (!ids->buffer || ggml_backend_buffer_is_host(ids->buffer))) {
                    std::vector<int32_t> active_ids;
                    active_ids.reserve(static_cast<size_t>(ggml_nelements(ids)));
                    for (int64_t i1 = 0; i1 < ids->ne[1]; ++i1) {
                        for (int64_t i0 = 0; i0 < ids->ne[0]; ++i0) {
                            const int32_t id = *reinterpret_cast<const int32_t*>(
                                static_cast<const char*>(ids->data) + i1 * ids->nb[1] + i0 * ids->nb[0]);
                            active_ids.push_back(id);
                        }
                    }
                    if (prepare_moe_experts(base0,
                                            active_ids.empty() ? nullptr : active_ids.data(),
                                            static_cast<int64_t>(active_ids.size()))) {
                        const auto& mp = moe_parents_[mit->second];
                        cpu_slice_ptrs_.assign(static_cast<size_t>(mp.n_experts), nullptr);
                        for (int32_t e = 0; e < mp.n_experts; ++e) {
                            cpu_slice_ptrs_[static_cast<size_t>(e)] = current_moe_slices_[static_cast<size_t>(e)];
                        }
                        ggml_cpu_tf_set_active_experts(base0, cpu_slice_ptrs_.data(), mp.n_experts);
                        active_cpu_moe_src_ = base0;
                        return true;
                    }
                }
                return false;
            }
        }

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

            const int32_t packed_slot = acquire_object_sync(obj_id, order_idx);
            uint8_t* ptr = slot_ptr(packed_slot);
            base->data = ptr;
            if (src != base) {
                src->data = ptr + src->view_offs;
            }
            active_node_slots_.push_back(packed_slot);
            auto sit = obj_id_to_sidx_.find(obj_id);
            if (sit != obj_id_to_sidx_.end()) {
                ++streamed_objs_[sit->second].access_count;
                const int c = slot_class(packed_slot);
                const int32_t ls = slot_local(packed_slot);
                auto& arena = arenas_[static_cast<size_t>(c)];
                if (cfg_.repin_enabled && arena.cache &&
                    can_repin_slot(arena.n_slots, arena.repinned_objs.size(), prefetch_inflight_cap_) &&
                    arena.repinned_objs.count(obj_id) == 0 &&
                    streamed_objs_[sit->second].access_count >= 2 &&
                    !arena.prefetch_loading[static_cast<size_t>(ls)]) {
                    if (arena.cache->pin(obj_id)) {
                        arena.repinned_objs.insert(obj_id);
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
        if (is_topk && is_host_node && node->type == GGML_TYPE_I32 && node->data &&
            cfg_.prefetch_mode != PrefetchMode::Off && io_) {
            const int32_t il = std::atoi(nname + 13);
            if (il >= 0 && il < moe_n_layer_ && moe_n_experts_ > 0) {
                std::vector<uint8_t> seen(static_cast<size_t>(moe_n_experts_), 0);
                std::vector<int32_t> topk_ids;
                for (int64_t i1 = 0; i1 < node->ne[1]; ++i1) {
                    for (int64_t i0 = 0; i0 < node->ne[0]; ++i0) {
                        const int32_t id = *reinterpret_cast<const int32_t*>(
                            static_cast<const char*>(node->data) + i1 * node->nb[1] + i0 * node->nb[0]);
                        if (id >= 0 && id < moe_n_experts_ && !seen[static_cast<size_t>(id)]) {
                            seen[static_cast<size_t>(id)] = 1;
                            topk_ids.push_back(id);
                        }
                    }
                }
                for (int kidx = 0; kidx < 3; ++kidx) {
                    const int32_t pid = moe_layer_parent_[static_cast<size_t>(il)][static_cast<size_t>(kidx)];
                    if (pid < 0) continue;
                    const auto& tmp = moe_parents_[static_cast<size_t>(pid)];
                    for (int32_t e : topk_ids) {
                        if (inflight_by_tag_.size() >= prefetch_inflight_cap_) break;
                        const int32_t pe = (cfg_.prefetch_mode == PrefetchMode::Shuffled)
                            ? static_cast<int32_t>((uint64_t(e) * 2654435761ULL + uint64_t(il + kidx * 17) * 1013904223ULL) % static_cast<uint64_t>(tmp.n_experts))
                            : e;
                        if (pe >= 0 && pe < tmp.n_experts) {
                            submit_prefetch_obj(tmp.slice_obj_ids[static_cast<size_t>(pe)], false);
                        }
                    }
                }
            }
            poll_completed_async(false);
            return true;
        }
        if (active_cpu_moe_src_ != nullptr) {
            ggml_cpu_tf_set_active_experts(nullptr, nullptr, 0);
            release_moe_experts(active_cpu_moe_src_);
            active_cpu_moe_src_ = nullptr;
        }
        poll_completed_async(false);
        for (int32_t packed : active_node_slots_) {
            const int c = slot_class(packed);
            const int32_t ls = slot_local(packed);
            if (c >= 0 && c < static_cast<int>(kNumArenaClasses) && arenas_[static_cast<size_t>(c)].cache) {
                arenas_[static_cast<size_t>(c)].cache->release_slot(ls);
            }
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
        mparams.n_gpu_layers       = (cfg_.n_gpu_layers >= 0) ? cfg_.n_gpu_layers : 0;
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

    // 2) Probe Vulkan GPU device & partition hybrid VRAM vs CPU/SSD tiers under tf_governor
    uint64_t vram_total = 0;
    uint64_t vram_free  = 0;
    uint64_t vram_budget = 0;
    std::string gpu_name;
    if (cfg_.n_gpu_layers != 0) {
        ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (gpu_dev) {
            ggml_backend_dev_props props{};
            ggml_backend_dev_get_props(gpu_dev, &props);
            size_t f_b = 0, t_b = 0;
            ggml_backend_dev_memory(gpu_dev, &f_b, &t_b);
            vram_free  = static_cast<uint64_t>(f_b);
            vram_total = static_cast<uint64_t>(t_b);
            if (props.description && props.description[0]) gpu_name = props.description;
            else if (props.name && props.name[0]) gpu_name = props.name;
            const uint64_t usable = (vram_free > cfg_.vram_reserve_bytes)
                ? (vram_free - cfg_.vram_reserve_bytes) : 0ULL;
            vram_budget = (cfg_.vram_budget_bytes > 0)
                ? std::min<uint64_t>(cfg_.vram_budget_bytes, usable)
                : usable;
        }
    }

    const uint64_t orig_vram_budget = vram_budget;
    uint64_t draft_vram_bytes = 0;
    uint64_t draft_ram_bytes  = 0;
    bool     draft_use_gpu    = false;
    if (!cfg_.draft_gguf_path.empty()) {
        if (cfg_.spec_k_mode == SpecKMode::Off) {
            cfg_.spec_k_mode = SpecKMode::Adaptive;
        }
        std::error_code df_ec;
        const uint64_t df_size = std::filesystem::file_size(cfg_.draft_gguf_path, df_ec);
        if (df_ec || df_size == 0) {
            gguf_free(gctx);
            summary_.error = "Failed to stat draft model GGUF: " + cfg_.draft_gguf_path;
            return summary_;
        }
        const uint64_t draft_est = df_size + (48ULL << 20);
        if (vram_budget >= draft_est + (256ULL << 20)) {
            draft_use_gpu    = true;
            draft_vram_bytes = draft_est;
            vram_budget     -= draft_vram_bytes;
        } else {
            draft_use_gpu   = false;
            draft_ram_bytes = draft_est;
        }
    }

    std::vector<uint64_t> layer_bytes(static_cast<size_t>(n_layer), 0ULL);
    std::vector<uint64_t> layer_non_exp_bytes(static_cast<size_t>(n_layer), 0ULL);
    uint64_t output_head_bytes = 0;

    moe_parents_.clear();
    parent_to_moe_idx_.clear();
    tensor_to_moe_parent_.clear();
    moe_layer_parent_.assign(static_cast<size_t>(n_layer), {-1, -1, -1});
    moe_n_layer_   = n_layer;
    moe_n_experts_ = manifest_.expert_count;
    moe_n_used_    = std::max<int32_t>(1, manifest_.expert_used_count);

    std::unordered_map<std::string, uint64_t> parent_total_bytes;
    for (const auto& obj : manifest_.objects) {
        const bool is_moe_slice_obj = (manifest_.slice_moe_experts && obj.role == ObjectRole::EXPERT &&
                                       !obj.parent_tensor.empty() && obj.expert_idx >= 0);
        if (obj.layer >= 0 && obj.layer < n_layer) {
            layer_bytes[static_cast<size_t>(obj.layer)] += obj.exact_bytes;
            if (!is_moe_slice_obj) {
                layer_non_exp_bytes[static_cast<size_t>(obj.layer)] += obj.exact_bytes;
            }
        } else if (obj.name == "output.weight" || obj.name == "output_norm.weight") {
            output_head_bytes += obj.exact_bytes;
        }
        if (is_moe_slice_obj) {
            parent_total_bytes[obj.parent_tensor] += obj.exact_bytes;
            auto pit = parent_to_moe_idx_.find(obj.parent_tensor);
            uint32_t pid = 0;
            if (pit == parent_to_moe_idx_.end()) {
                pid = static_cast<uint32_t>(moe_parents_.size());
                parent_to_moe_idx_[obj.parent_tensor] = pid;
                MoeParentInfo mp{};
                mp.parent_idx  = pid;
                mp.parent_name = obj.parent_tensor;
                mp.layer       = obj.layer;
                mp.n_experts   = std::max<int32_t>(obj.n_experts, obj.expert_idx + 1);
                if (obj.parent_tensor.find("ffn_up") != std::string::npos) mp.kind = MoeProjKind::Up;
                else if (obj.parent_tensor.find("ffn_down") != std::string::npos) mp.kind = MoeProjKind::Down;
                else mp.kind = MoeProjKind::Gate;
                mp.class_idx = assign_size_class(obj.page_aligned_bytes);
                mp.slice_obj_ids.assign(static_cast<size_t>(mp.n_experts), 0);
                moe_parents_.push_back(std::move(mp));
                if (obj.layer >= 0 && obj.layer < n_layer) {
                    moe_layer_parent_[static_cast<size_t>(obj.layer)][static_cast<size_t>(moe_parents_.back().kind)] =
                        static_cast<int32_t>(pid);
                }
            } else {
                pid = pit->second;
            }
            auto& mp = moe_parents_[pid];
            if (obj.expert_idx >= mp.n_experts) {
                mp.n_experts = obj.expert_idx + 1;
                mp.slice_obj_ids.resize(static_cast<size_t>(mp.n_experts), 0);
            }
            mp.slice_obj_ids[static_cast<size_t>(obj.expert_idx)] = obj.object_id;
            if (mp.n_experts > moe_n_experts_) moe_n_experts_ = mp.n_experts;
        }
    }

    uint64_t max_exps_tensor_bytes = 0;
    for (const auto& kvp : parent_total_bytes) {
        if (kvp.second > max_exps_tensor_bytes) max_exps_tensor_bytes = kvp.second;
    }
    if (moe_n_layer_ > 1 && moe_n_experts_ > 0) {
        eam_counts_.assign(static_cast<size_t>(moe_n_layer_ - 1),
                           std::vector<uint16_t>(static_cast<size_t>(moe_n_experts_) * static_cast<size_t>(moe_n_experts_), 0));
        eam_row_totals_.assign(static_cast<size_t>(moe_n_layer_ - 1),
                               std::vector<uint32_t>(static_cast<size_t>(moe_n_experts_), 0));
    } else {
        eam_counts_.clear();
        eam_row_totals_.clear();
    }
    prev_layer_experts_.assign(static_cast<size_t>(std::max(1, moe_n_layer_)), {});

    int32_t  n_gpu_layers_active = 0;
    uint64_t vram_weights_bytes  = 0;
    uint64_t vram_kv_bytes       = 0;
    uint64_t vram_scratch_bytes  = 0;
    if (vram_budget > 0 && cfg_.n_gpu_layers != 0) {
        const int32_t max_layers = (cfg_.n_gpu_layers > 0)
            ? std::min<int32_t>(cfg_.n_gpu_layers, n_layer + 1)
            : (n_layer + 1);
        vram_scratch_bytes = std::min<uint64_t>(std::max<uint64_t>(64ULL << 20, vram_budget / 8), 256ULL << 20);
        if (manifest_.slice_moe_experts && max_exps_tensor_bytes > 0) {
            vram_scratch_bytes = std::max<uint64_t>(vram_scratch_bytes, max_exps_tensor_bytes + (64ULL << 20));
        }
        if (vram_budget > output_head_bytes + vram_scratch_bytes) {
            uint64_t vram_rem = vram_budget - output_head_bytes - vram_scratch_bytes;
            vram_weights_bytes += output_head_bytes;
            n_gpu_layers_active = 1; // output layer offloaded
            const uint64_t first_layer_need = (n_layer > 0)
                ? (manifest_.slice_moe_experts ? layer_non_exp_bytes[static_cast<size_t>(n_layer - 1)]
                                               : layer_bytes[static_cast<size_t>(n_layer - 1)])
                : 0ULL;
            if (vram_rem > kv_bytes + first_layer_need) {
                vram_kv_bytes = kv_bytes;
                vram_rem     -= kv_bytes;
            }
            for (int32_t il = n_layer - 1; il >= 0 && n_gpu_layers_active < max_layers; --il) {
                const uint64_t lb = manifest_.slice_moe_experts
                    ? layer_non_exp_bytes[static_cast<size_t>(il)]
                    : layer_bytes[static_cast<size_t>(il)];
                if (lb <= vram_rem) {
                    vram_rem           -= lb;
                    vram_weights_bytes += lb;
                    ++n_gpu_layers_active;
                } else {
                    break;
                }
            }
        } else {
            vram_scratch_bytes = 0;
        }
    }

    auto is_on_gpu = [&](const ManifestObject& obj) -> bool {
        if (n_gpu_layers_active <= 0) return false;
        if (manifest_.slice_moe_experts && obj.role == ObjectRole::EXPERT &&
            !obj.parent_tensor.empty() && obj.expert_idx >= 0) return false;
        if (obj.name == "token_embd.weight") return false; // input layer always CPU in llama.cpp
        if (obj.layer == -1 && (obj.name == "output.weight" || obj.name == "output_norm.weight")) return true;
        if (obj.layer >= 0 && obj.layer >= (n_layer + 1 - n_gpu_layers_active)) return true;
        return false;
    };

    const uint64_t cpu_kv_bytes = (vram_kv_bytes > 0) ? 0ULL : kv_bytes;
    const uint64_t total_model_exact = manifest_.total_exact_bytes();
    uint64_t cpu_model_exact = 0;
    for (const auto& obj : manifest_.objects) {
        if (!is_on_gpu(obj)) cpu_model_exact += obj.exact_bytes;
    }

    summary_.model_total_bytes         = total_model_exact;
    summary_.ram_kv_bytes              = cpu_kv_bytes;
    summary_.vram_kv_bytes             = vram_kv_bytes;
    summary_.vram_total_bytes          = vram_total;
    summary_.vram_free_bytes           = vram_free;
    summary_.vram_budget_bytes         = orig_vram_budget;
    summary_.vram_weights_bytes        = vram_weights_bytes + draft_vram_bytes;
    summary_.n_gpu_layers_active       = n_gpu_layers_active;
    summary_.gpu_name                  = gpu_name;
    summary_.ram_compute_reserve_bytes = compute_reserve;

    const uint64_t ram_budget = (cfg_.ram_budget_bytes > 0)
        ? cfg_.ram_budget_bytes
        : (cpu_model_exact + cpu_kv_bytes + compute_reserve + draft_ram_bytes + (128ULL << 20));
    if (ram_budget <= cpu_kv_bytes + compute_reserve + draft_ram_bytes) {
        gguf_free(gctx);
        summary_.error = "RAM budget too small even for CPU KV cache + runtime compute reserve";
        return summary_;
    }
    const uint64_t avail_for_weights_and_cache = ram_budget - cpu_kv_bytes - compute_reserve - draft_ram_bytes;
    const bool needs_streaming = (avail_for_weights_and_cache < cpu_model_exact);

    // Classify remaining CPU objects & partition 3-size-class arena budget cleanly
    name_to_obj_id_.clear();
    std::vector<PlannerObjectInput> p_inputs;
    p_inputs.reserve(manifest_.objects.size());
    std::vector<uint64_t> streamable_aligned_sizes;
    uint64_t always_on_bytes = 0;
    uint64_t first_stream_bytes = 0;

    for (const auto& obj : manifest_.objects) {
        name_to_obj_id_[obj.name] = obj.object_id;
        if (is_on_gpu(obj)) continue;

        const bool is_moe_slice = (manifest_.slice_moe_experts && obj.role == ObjectRole::EXPERT &&
                                   !obj.parent_tensor.empty() && obj.expert_idx >= 0);
        const bool is_ffn_mat   = (obj.name.find(".ffn_gate.weight") != std::string::npos ||
                                   obj.name.find(".ffn_up.weight") != std::string::npos ||
                                   obj.name.find(".ffn_down.weight") != std::string::npos);
        const bool is_attn_mat  = (obj.name.find(".attn_q.weight") != std::string::npos ||
                                   obj.name.find(".attn_k.weight") != std::string::npos ||
                                   obj.name.find(".attn_v.weight") != std::string::npos ||
                                   obj.name.find(".attn_output.weight") != std::string::npos);
        const bool streamable = manifest_.slice_moe_experts ? is_moe_slice : (is_ffn_mat || is_attn_mat);
        const bool is_sparse_embd = (obj.name == "token_embd.weight");
        if (streamable) {
            streamable_aligned_sizes.push_back(obj.page_aligned_bytes);
            if (first_stream_bytes == 0) first_stream_bytes = obj.exact_bytes;
        } else {
            always_on_bytes += obj.exact_bytes;
        }
        const double traffic_freq = is_sparse_embd ? 0.001 : (is_moe_slice ? 0.15 : 1.0);
        const double rank_prio    = streamable
            ? (is_moe_slice ? -1.0 : (is_attn_mat ? 1.5 : (1.0 - 1e-4 * std::max(0, obj.layer))))
            : 2.0;
        p_inputs.push_back({obj.object_id, obj.exact_bytes, traffic_freq, !streamable, rank_prio});
    }

    // Consult compute placement heuristic (DDR5 CPU GEMV vs PCIe + GPU) cleanly
    if (first_stream_bytes > 0) {
        const auto ct = choose_compute_target(
            first_stream_bytes, first_stream_bytes * 2ULL, 1u, false, probe_);
        (void)ct;
    }

    ArenaLayoutPlan arena_plan{};
    uint64_t slot_bank_bytes = 0;
    const size_t effective_slot_cap = manifest_.slice_moe_experts
        ? std::max<size_t>(cfg_.cache_slots_max, 65536ULL)
        : cfg_.cache_slots_max;
    if (needs_streaming && !streamable_aligned_sizes.empty()) {
        if (always_on_bytes >= avail_for_weights_and_cache && !cfg_.allow_slow_stream) {
            gguf_free(gctx);
            summary_.error = "Always-on CPU weights exceed RAM budget (enable --allow-slow-stream to proceed)";
            return summary_;
        }
        const uint64_t room_after_always = (avail_for_weights_and_cache > always_on_bytes)
            ? (avail_for_weights_and_cache - always_on_bytes)
            : avail_for_weights_and_cache;
        arena_plan = compute_arena_layout(streamable_aligned_sizes, room_after_always, effective_slot_cap, 4);
        if (!arena_plan.feasible) {
            gguf_free(gctx);
            summary_.error = "RAM budget too small for minimum 4-slot per-class streaming arenas";
            return summary_;
        }
        slot_bank_bytes = arena_plan.total_bytes;
    }

    uint64_t weight_pin_budget = (avail_for_weights_and_cache > slot_bank_bytes)
        ? (avail_for_weights_and_cache - slot_bank_bytes)
        : 0;
    if (manifest_.slice_moe_experts && needs_streaming) {
        weight_pin_budget = std::min<uint64_t>(always_on_bytes, weight_pin_budget);
    }
    PlannerBudget p_budget{};
    // Non-GPU objects remaining in p_inputs must never overflow back into VRAM tier
    p_budget.vram_bytes              = 0ULL;
    p_budget.ram_bytes               = weight_pin_budget;
    p_budget.allow_slow_dense_stream = cfg_.allow_slow_stream;
    plan_ = build_placement_plan(p_inputs, p_budget, probe_);
    if (!plan_.feasible) {
        gguf_free(gctx);
        summary_.error = "tf_planner rejected placement: " + plan_.reason;
        return summary_;
    }

    // Governor startup budget check & atomic reservations across RAM + VRAM
    gov_ = std::make_unique<tf_governor>(ram_budget, orig_vram_budget, ram_budget * 2);
    std::vector<PlanReservation> startup_reqs{
        {MemoryTier::RAM, plan_.ram_used_bytes, "ram_weights"},
        {MemoryTier::RAM, slot_bank_bytes, "tf_cache_slots"},
        {MemoryTier::RAM, cpu_kv_bytes, "kv_cache"},
        {MemoryTier::RAM, compute_reserve, "compute_scratch"},
    };
    if (draft_ram_bytes > 0)     startup_reqs.push_back({MemoryTier::RAM, draft_ram_bytes, "draft_ram"});
    if (vram_weights_bytes > 0)  startup_reqs.push_back({MemoryTier::VRAM, vram_weights_bytes, "vram_weights"});
    if (draft_vram_bytes > 0)    startup_reqs.push_back({MemoryTier::VRAM, draft_vram_bytes, "draft_vram"});
    if (vram_kv_bytes > 0)       startup_reqs.push_back({MemoryTier::VRAM, vram_kv_bytes, "vram_kv_cache"});
    if (vram_scratch_bytes > 0)  startup_reqs.push_back({MemoryTier::VRAM, vram_scratch_bytes, "vram_compute_scratch"});
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

    // Map dense index over p_inputs -> manifest_.objects[] safely!
    std::unordered_map<uint32_t, PlacementTier> tier_by_obj;
    for (size_t i = 0; i < p_inputs.size(); ++i) {
        tier_by_obj[p_inputs[i].object_id] = plan_.tier_of[i];
    }

    streamed_names_.clear();
    streamed_objs_.clear();
    obj_id_to_sidx_.clear();
    shuffled_order_.clear();
    has_streamed_dense_ = false;
    has_streamed_moe_   = false;
    for (const auto& obj : manifest_.objects) {
        auto tit = tier_by_obj.find(obj.object_id);
        const bool is_moe_slice = (manifest_.slice_moe_experts && obj.role == ObjectRole::EXPERT &&
                                   !obj.parent_tensor.empty() && obj.expert_idx >= 0);
        if (tit != tier_by_obj.end() && tit->second == PlacementTier::SSD_STREAM) {
            if (is_moe_slice && !obj.parent_tensor.empty()) {
                streamed_names_.insert(obj.parent_tensor);
                has_streamed_moe_ = true;
            } else {
                streamed_names_.insert(obj.name);
                has_streamed_dense_ = true;
            }
            StreamedTensorInfo info{};
            info.object_id          = obj.object_id;
            info.name               = obj.name;
            info.tfpack_offset      = obj.tfpack_offset;
            info.exact_bytes        = obj.exact_bytes;
            info.page_aligned_bytes = obj.page_aligned_bytes;
            info.layer              = obj.layer;
            info.class_idx          = assign_size_class(obj.page_aligned_bytes);
            obj_id_to_sidx_[obj.object_id] = streamed_objs_.size();
            streamed_objs_.push_back(info);
            shuffled_order_.push_back(obj.object_id);
        }
    }
    std::mt19937_64 shuf_gen(20261112ULL);
    std::shuffle(shuffled_order_.begin(), shuffled_order_.end(), shuf_gen);

    HardwareProfile burst_probe = probe_;
    burst_probe.bw_ssd_sustained_gbps = probe_.bw_ssd_burst_gbps;
    const double burst_cost = evaluate_placement_cost_sec(p_inputs, plan_.tier_of, burst_probe);

    size_t max_arena_stride = 0;
    for (size_t c = 0; c < kNumArenaClasses; ++c) {
        summary_.arena_slots[c]   = arena_plan.classes[c].n_slots;
        summary_.arena_strides[c] = arena_plan.classes[c].stride_bytes;
        if (arena_plan.classes[c].stride_bytes > max_arena_stride) {
            max_arena_stride = arena_plan.classes[c].stride_bytes;
        }
    }

    io_bw_ewma_gbps_                  = probe_.bw_ssd_sustained_gbps;
    summary_.ram_weights_bytes        = plan_.ram_used_bytes;
    summary_.ram_slot_bank_bytes      = slot_bank_bytes;
    summary_.ram_governor_total_bytes = gov_->used(MemoryTier::RAM);
    summary_.vram_governor_total_bytes = gov_->used(MemoryTier::VRAM);
    summary_.ssd_stream_bytes         = plan_.ssd_stream_bytes;
    summary_.n_streamed_tensors       = streamed_objs_.size();
    summary_.n_cache_slots            = arena_plan.total_slots;
    summary_.slot_stride_bytes        = max_arena_stride;
    summary_.predicted_hit_rate       = plan_.predicted_hit_rate;
    summary_.predicted_tok_s          = plan_.predicted_tok_s;
    summary_.predicted_tok_s_burst    = (burst_cost > 0.0) ? (1.0 / burst_cost) : 0.0;
    summary_.io_bw_ewma_gbps          = io_bw_ewma_gbps_;

    prefetch_inflight_cap_ = compute_dynamic_prefetch_cap(arena_plan.total_slots);
    if (prefetch_inflight_cap_ == 0) prefetch_inflight_cap_ = 1;

    // Open .tfpack via unbuffered async I/O with dynamic queue depth
    tf_io_options io_opts{};
    io_opts.max_inflight          = std::max<size_t>(has_streamed_moe_ ? 64 : 32, prefetch_inflight_cap_ * 2);
    io_opts.prefetch_inflight_cap = prefetch_inflight_cap_;
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

    for (size_t c = 0; c < kNumArenaClasses; ++c) {
        const auto& sp = arena_plan.classes[c];
        if (sp.n_slots == 0 || sp.stride_bytes == 0) continue;
        auto& arena = arenas_[c];
        arena.stride_bytes = sp.stride_bytes;
        arena.n_slots      = sp.n_slots;
        arena.mem_bytes    = static_cast<size_t>(sp.total_bytes);
        arena.mem          = tf_aligned_alloc(arena.mem_bytes, 4096);
        if (!arena.mem) {
            gguf_free(gctx);
            summary_.error = "Failed to VirtualAlloc aligned size-class arena";
            return summary_;
        }
        arena.prefetch_loading.assign(sp.n_slots, 0);
        CacheConfig ccfg{};
        ccfg.capacity_slots = sp.n_slots;
        ccfg.policy         = cfg_.cache_policy;
        arena.cache         = std::make_unique<tf_cache>(ccfg);
    }

    // Construct llama_model using zero-mmap user callback loader + 0-byte streamed tensor dummy buffers
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = n_gpu_layers_active;
    mparams.use_extra_bufts    = (n_gpu_layers_active > 0);
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
    cparams.offload_kqv          = (vram_kv_bytes > 0);
    cparams.op_offload           = !has_streamed_dense_;
    if (!streamed_names_.empty()) {
        // Dense CPU-streamed weights must run on CPU so cb_eval acquires slots before compute;
        // otherwise ggml_backend_sched op_offload copies NULL base->data for batches >= 32.
        // MoE exps slices keep op_offload=true and stream cleanly via sched_copy_experts hooks.
        cparams.cb_eval           = &tf_llama_stream::tf_eval_cb;
        cparams.cb_eval_user_data = this;
    }
    if (has_streamed_moe_) {
        cparams.tf_prepare_experts = &tf_llama_stream::tf_prepare_experts_cb;
        cparams.tf_resolve_expert  = &tf_llama_stream::tf_resolve_expert_cb;
        cparams.tf_release_experts = &tf_llama_stream::tf_release_experts_cb;
        cparams.tf_experts_ud      = this;
    }

    ctx_ = llama_init_from_model(model_, cparams);
    if (!ctx_) {
        summary_.error = "llama_init_from_model failed";
        return summary_;
    }

    if (!cfg_.draft_gguf_path.empty()) {
        draft_ = std::make_unique<tf_draft>();
        DraftConfig dcfg{};
        dcfg.gguf_path    = cfg_.draft_gguf_path;
        dcfg.n_ctx        = cfg_.n_ctx;
        dcfg.n_batch      = cfg_.n_batch;
        dcfg.n_threads    = cfg_.n_threads;
        dcfg.n_gpu_layers = draft_use_gpu ? 99 : 0;
        dcfg.kv_precision = cfg_.kv_precision;
        dcfg.quiet_logs   = cfg_.quiet_logs;
        std::string derr;
        if (!draft_->init(dcfg, &derr)) {
            summary_.error = "tf_draft init failed: " + derr;
            return summary_;
        }
        const int32_t target_nv = llama_vocab_n_tokens(llama_model_get_vocab(model_));
        if (draft_->n_vocab() != target_nv) {
            summary_.error = "Draft vocab mismatch: draft=" + std::to_string(draft_->n_vocab()) +
                             " target=" + std::to_string(target_nv);
            return summary_;
        }
        summary_.ram_weights_bytes += draft_ram_bytes;
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
    if (draft_) {
        draft_->reset();
    }
    run_hits_              = 0;
    run_misses_            = 0;
    run_bytes_ssd_         = 0;
    run_prefetch_sub_      = 0;
    run_prefetch_hits_     = 0;
    run_prefetch_wasted_b_ = 0;
    run_eam_prefetch_sub_  = 0;
    run_eam_prefetch_hits_ = 0;
    for (auto& pv : prev_layer_experts_) {
        pv.clear();
    }

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

    // Speculative controller state (Phase 6 + Phase 11 MoE draft model, k <= 4 unless --spec-tree opt-in)
    const int32_t max_allowed_k = cfg_.spec_tree
        ? std::clamp<int32_t>(cfg_.spec_max_k, 1, 16)
        : std::clamp<int32_t>(cfg_.spec_max_k, 1, 4);
    SpecSchedulerConfig scfg{};
    scfg.k_max = max_allowed_k;
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
            // Phase 6 / 11 Speculative Verification with union-aware / ablation k-selection
            int32_t k_target = 0;
            if (cfg_.spec_k_mode == SpecKMode::Fixed4) {
                k_target = max_allowed_k;
            } else {
                SpecInputs sin{};
                sin.alpha_hat    = spec_sched.alpha_ema();
                const double ram_gb = static_cast<double>(summary_.ram_weights_bytes) / 1e9;
                const double n_stream_f = static_cast<double>(summary_.n_streamed_tensors);
                sin.c_always_sec = ram_gb / std::max(1.0, probe_.cpu_gemv_effective_gbps);
                if (summary_.ssd_stream_bytes > 0 && summary_.n_streamed_tensors > 0) {
                    const double mean_obj_gb = (static_cast<double>(summary_.ssd_stream_bytes) / n_stream_f) / 1e9;
                    const double eff_bw = (io_bw_ewma_gbps_ > 0.05) ? io_bw_ewma_gbps_ : probe_.bw_ssd_sustained_gbps;
                    sin.cost_per_expert_sec  = mean_obj_gb / std::max(0.1, eff_bw);
                } else {
                    sin.cost_per_expert_sec  = 0.0002;
                }
                sin.t_draft_sec = draft_ ? last_draft_sec_ : 0.0001;
                sin.u_total_by_k.resize(static_cast<size_t>(scfg.k_max + 1));
                const double cur_miss = (run_hits_ + run_misses_ > 0)
                    ? static_cast<double>(run_misses_) / static_cast<double>(run_hits_ + run_misses_)
                    : (1.0 - std::clamp(summary_.predicted_hit_rate, 0.0, 0.95));
                const double base_u = has_streamed_moe_
                    ? std::max(1.0, static_cast<double>(moe_n_layer_) * 3.0 * static_cast<double>(moe_n_used_) * std::max(0.10, cur_miss))
                    : std::max(1.0, n_stream_f);
                for (int k = 0; k <= scfg.k_max; ++k) {
                    if (cfg_.spec_k_mode == SpecKMode::NaiveNoUnion) {
                        sin.u_total_by_k[static_cast<size_t>(k)] = base_u * static_cast<double>(k + 1);
                    } else if (has_streamed_moe_) {
                        // Correlated MoE union curve: 8-token batch widens distinct active experts (~3.5x vs 1 token)
                        sin.u_total_by_k[static_cast<size_t>(k)] = base_u * (1.0 + 0.35 * static_cast<double>(k));
                    } else {
                        sin.u_total_by_k[static_cast<size_t>(k)] = base_u * (1.0 + 0.02 * static_cast<double>(k));
                    }
                }
                k_target = spec_sched.choose_k(sin).k_opt;
            }
            k_target = std::min<int32_t>(k_target, n_predict - generated - 1);
            DraftProposal dprop{};
            if (draft_ && k_target > 0) {
                dprop = draft_->propose(full_history, k_target, temp, rng);
                if (dprop.elapsed_sec > 0.0 && !dprop.tokens.empty()) {
                    const double per_tok = dprop.elapsed_sec / static_cast<double>(dprop.tokens.size());
                    last_draft_sec_ = 0.7 * last_draft_sec_ + 0.3 * per_tok;
                }
            } else if (k_target > 0) {
                dprop.tokens = prompt_lookup_draft(full_history, k_target);
            }
            const auto& draft = dprop.tokens;
            const int32_t k_act = static_cast<int32_t>(draft.size());
            ++spec_steps;
            spec_k_sum += static_cast<uint64_t>(k_act);
            const uint64_t misses_before_step = run_misses_;

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
                // Exact Leviathan speculative rejection sampling (supports both neural q(x) and deterministic prompt-lookup q=1[x=d])
                std::vector<double> probs;
                std::uniform_real_distribution<double> u01(0.0, 1.0);
                const bool has_q = (dprop.probs.size() == static_cast<size_t>(k_act));
                for (int32_t i = 0; i < k_act; ++i) {
                    softmax_probs(i, probs);
                    const llama_token d = draft[static_cast<size_t>(i)];
                    const double p_d = probs[static_cast<size_t>(d)];
                    const double q_d = has_q ? dprop.probs[static_cast<size_t>(i)][static_cast<size_t>(d)] : 1.0;
                    const double accept_p = (q_d > 0.0) ? std::min(1.0, p_d / q_d) : ((p_d > 0.0) ? 1.0 : 0.0);
                    if (u01(rng) < accept_p) {
                        out.generated_tokens.push_back(d);
                        full_history.push_back(d);
                        ++generated;
                        ++accepted;
                        if (generated >= n_predict || llama_vocab_is_eog(vocab, d)) {
                            rejected_early = true;
                            break;
                        }
                    } else {
                        std::vector<double> res(static_cast<size_t>(n_vocab), 0.0);
                        double sum_res = 0.0;
                        if (has_q) {
                            const auto& qp = dprop.probs[static_cast<size_t>(i)];
                            for (int32_t v = 0; v < n_vocab; ++v) {
                                const double diff = std::max(0.0, probs[static_cast<size_t>(v)] - qp[static_cast<size_t>(v)]);
                                res[static_cast<size_t>(v)] = diff;
                                sum_res += diff;
                            }
                        } else {
                            for (int32_t v = 0; v < n_vocab; ++v) {
                                if (v == d) continue;
                                res[static_cast<size_t>(v)] = probs[static_cast<size_t>(v)];
                                sum_res += res[static_cast<size_t>(v)];
                            }
                        }
                        llama_token bonus = 0;
                        if (sum_res > 1e-12) {
                            double target = u01(rng) * sum_res;
                            double acc = 0.0;
                            for (int32_t v = 0; v < n_vocab; ++v) {
                                acc += res[static_cast<size_t>(v)];
                                if (acc >= target) {
                                    bonus = v;
                                    break;
                                }
                            }
                            if (acc < target) {
                                bonus = (d == n_vocab - 1) ? (n_vocab - 2) : (n_vocab - 1);
                            }
                        } else {
                            bonus = sample_token(i);
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
                const double obs_union = has_streamed_moe_
                    ? static_cast<double>(run_misses_ - misses_before_step)
                    : static_cast<double>(summary_.n_streamed_tensors);
                spec_sched.update_ema(k_act, accepted + 1, obs_union);
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
    out.eam_prefetch_hits        = run_eam_prefetch_hits_;
    out.eam_prefetch_submitted   = run_eam_prefetch_sub_;
    out.spec_k_mean              = (spec_steps > 0)
        ? static_cast<double>(spec_k_sum) / static_cast<double>(spec_steps)
        : 0.0;
    out.spec_accept_rate         = (spec_draft_total > 0)
        ? static_cast<double>(spec_accept_total) / static_cast<double>(spec_draft_total)
        : 0.0;
    out.io_bw_ewma_gbps          = io_bw_ewma_gbps_;
    out.ok = true;
    return out;
}

} // namespace tierflow
