#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace tierflow {

enum class CachePolicy : uint8_t {
    WTinyLFU_SLRU = 0,
    LRU           = 1,
};

struct CacheConfig {
    size_t      capacity_slots = 32;
    CachePolicy policy         = CachePolicy::WTinyLFU_SLRU;
    double      window_frac    = 0.15;
    double      protected_frac = 0.80;
    uint32_t    age_interval   = 2048;
    uint32_t    cms_width      = 512;
    uint32_t    cms_depth      = 4;
};

struct CacheAccessResult {
    bool     hit               = false;
    int32_t  slot_id           = -1; // -1 only if cache full and ALL slots are pinned or refcount > 0
    bool     evicted           = false;
    uint32_t evicted_object_id = 0;
};

class tf_cache {
public:
    explicit tf_cache(const CacheConfig& cfg);

    // Pin an object permanently into a slot (fails if no unpinned slots left).
    bool pin(uint32_t object_id, int32_t* slot_out = nullptr);

    // Demand access: records frequency, returns slot_id on hit or admits/evicts on miss.
    // Never evicts a pinned slot or any slot with refcount > 0.
    CacheAccessResult access(uint32_t object_id);

    // Non-mutating lookup returning resident slot_id or -1.
    int32_t find_slot(uint32_t object_id) const;

    // Low-priority prefetch admission: inserts into Probation only if space exists or an
    // unreferenced Probation victim can be replaced (never evicts Protected or Window items).
    CacheAccessResult prefetch(uint32_t object_id);

    // Refcount guard for in-flight async I/O or active GPU/CPU execution.
    bool acquire_slot(int32_t slot_id);
    bool release_slot(int32_t slot_id);
    uint32_t refcount(int32_t slot_id) const;

    bool     is_resident(uint32_t object_id) const;
    size_t   capacity() const { return cfg_.capacity_slots; }
    size_t   resident_count() const;
    uint64_t hits() const;
    uint64_t misses() const;
    double   hit_rate() const;

private:
    enum class Segment : uint8_t {
        FREE      = 0,
        PINNED    = 1,
        WINDOW    = 2,
        PROBATION = 3,
        PROTECTED = 4,
    };

    struct SlotMeta {
        uint32_t                   object_id = 0;
        Segment                    segment   = Segment::FREE;
        uint32_t                   refcount  = 0;
        std::list<int32_t>::iterator list_it;
    };

    void     recompute_segment_caps_locked();
    void     record_freq_locked(uint32_t object_id);
    uint32_t est_freq_locked(uint32_t object_id) const;
    int32_t  alloc_free_slot_locked();
    void     attach_to_segment_locked(int32_t slot_id, uint32_t object_id, Segment seg);
    void     detach_from_segment_locked(int32_t slot_id);

    // Finds oldest slot with refcount == 0 in a given list (from front = LRU end).
    int32_t find_unreffed_lru_locked(const std::list<int32_t>& lst) const;
    CacheAccessResult replace_window_victim_locked(int32_t win_victim, uint32_t new_object_id);

    CacheConfig                        cfg_;
    mutable std::mutex                 mu_;
    std::vector<SlotMeta>              slots_;
    std::unordered_map<uint32_t, int32_t> obj_to_slot_;

    std::list<int32_t>                 free_slots_;
    std::list<int32_t>                 window_lru_;
    std::list<int32_t>                 probation_lru_;
    std::list<int32_t>                 protected_lru_;

    size_t                             pinned_count_ = 0;
    size_t                             win_cap_      = 0;
    size_t                             prob_cap_     = 0;
    size_t                             prot_cap_     = 0;

    // 4-bit Count-Min Sketch (stored in uint8_t counters capped at 15, halved on age)
    std::vector<uint8_t>               cms_;
    uint64_t                           freq_steps_   = 0;
    uint64_t                           hits_         = 0;
    uint64_t                           misses_       = 0;
};

// Offline Belady optimal oracle for unit test verification & ceiling simulation.
uint64_t belady_oracle_misses(const std::vector<uint32_t>& trace, size_t capacity_slots);

} // namespace tierflow

