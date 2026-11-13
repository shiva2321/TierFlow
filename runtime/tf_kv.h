#pragma once

#include "tf_governor.h"
#include "tf_io.h"

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace tierflow {

enum class KvPrecision : uint8_t {
    F16 = 0,
    Q8_0 = 1,
    Q8_K_Q4_V = 2, // Phase 8 opt-in lossy V compression
};

struct KvConfig {
    uint32_t    n_layers         = 28;
    uint32_t    n_kv_heads       = 2;
    uint32_t    head_dim_k       = 128;
    uint32_t    head_dim_v       = 128;
    uint32_t    page_tokens      = 16;
    KvPrecision precision        = KvPrecision::Q8_0;
    std::string spill_path;      // if non-empty, enables unbuffered SSD spill + reload
    std::string prefix_dir;      // if non-empty, enables disk prefix cache blocks
};

struct KvPageStats {
    uint64_t pages_resident_ram  = 0;
    uint64_t pages_spilled_ssd   = 0;
    uint64_t spill_writes        = 0;
    uint64_t spill_reads         = 0;
    uint64_t prefix_hits         = 0;
    uint64_t prefix_misses       = 0;
    uint64_t spill_bytes_written = 0;
    uint64_t spill_bytes_read    = 0;
};

// Low-level IEEE-754 half helpers + symmetric Q8_0 (32-element block, 34B) codec.
uint16_t f32_to_f16_bits(float x);
float    f16_bits_to_f32(uint16_t h);
void     quantize_row_q8_0(const float* src, void* dst, size_t n);
void     dequantize_row_q8_0(const void* src, float* dst, size_t n);
void     quantize_row_q4_0(const float* src, void* dst, size_t n);
void     dequantize_row_q4_0(const void* src, float* dst, size_t n);

class tf_kv_store {
public:
    tf_kv_store(const KvConfig& cfg, tf_governor& gov);
    ~tf_kv_store();

    tf_kv_store(const tf_kv_store&) = delete;
    tf_kv_store& operator=(const tf_kv_store&) = delete;

    // Logical byte size per page (exact encoded KV payload) and 4096-aligned spill slot size.
    size_t page_payload_bytes() const { return payload_bytes_; }
    size_t page_aligned_bytes() const { return aligned_bytes_; }

    // Append/store `n_tokens` (<= page_tokens) of float K and V vectors for `(seq_id, layer, page_idx)`:
    // k/v shape: [n_tokens, n_kv_heads * head_dim]. Spills oldest unpinned RAM page to SSD if
    // gov_->remaining(RAM) < payload_bytes_.
    bool put_page_f32(uint64_t seq_id,
                      uint32_t layer,
                      uint32_t page_idx,
                      uint32_t n_tokens,
                      const float* k_f32,
                      const float* v_f32);

    // Retrieve decoded float K and V buffers for `(seq_id, layer, page_idx)`. If spilled to SSD,
    // reloads from direct unbuffered SSD shard back into RAM (evicting/spilling another page if full).
    bool get_page_f32(uint64_t seq_id,
                      uint32_t layer,
                      uint32_t page_idx,
                      std::vector<float>& k_out,
                      std::vector<float>& v_out);

    // Store / load raw encoded page payload (bit-exact with prefix cache).
    bool save_prefix_block(uint64_t prefix_hash, const void* state_bytes, size_t bytes);
    bool load_prefix_block(uint64_t prefix_hash, std::vector<uint8_t>& state_out);

    void evict_sequence(uint64_t seq_id);
    KvPageStats stats() const;

private:
    struct PageKey {
        uint64_t seq_id   = 0;
        uint32_t layer    = 0;
        uint32_t page_idx = 0;
        bool operator==(const PageKey& o) const {
            return seq_id == o.seq_id && layer == o.layer && page_idx == o.page_idx;
        }
    };
    struct PageKeyHash {
        size_t operator()(const PageKey& k) const {
            uint64_t h = k.seq_id * 0x9e3779b97f4a7c15ULL;
            h ^= (uint64_t(k.layer) << 32) | k.page_idx;
            h ^= h >> 30;
            h *= 0xbf58476d1ce4e5b9ULL;
            h ^= h >> 27;
            return static_cast<size_t>(h);
        }
    };

    struct PageEntry {
        PageKey                    key;
        uint32_t                   n_tokens   = 0;
        bool                       in_ram     = false;
        bool                       on_ssd     = false;
        void*                      ram_buf    = nullptr; // 4096-aligned VirtualAlloc/aligned buffer
        uint64_t                   ssd_slot   = UINT64_MAX;
        std::list<PageKey>::iterator lru_it;
    };

    bool ensure_spill_open_locked();
    bool spill_one_lru_locked(const PageKey* avoid_key = nullptr);
    bool write_ssd_slot_locked(uint64_t slot, const void* src_aligned);
    bool read_ssd_slot_locked(uint64_t slot, void* dst_aligned);
    uint64_t alloc_ssd_slot_locked();

    KvConfig    cfg_;
    tf_governor& gov_;
    size_t      k_row_elems_    = 0;
    size_t      v_row_elems_    = 0;
    size_t      k_page_bytes_   = 0;
    size_t      v_page_bytes_   = 0;
    size_t      payload_bytes_  = 0;
    size_t      aligned_bytes_  = 0;

    mutable std::mutex mu_;
    std::unordered_map<PageKey, PageEntry, PageKeyHash> pages_;
    std::list<PageKey> ram_lru_;
    std::vector<uint64_t> free_ssd_slots_;
    uint64_t next_ssd_slot_ = 0;

    void* spill_handle_ = nullptr; // HANDLE on Windows or int fd wrapper
    KvPageStats stats_;
};

} // namespace tierflow

