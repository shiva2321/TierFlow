#include "tf_cache.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace tierflow {

namespace {

inline uint32_t hash_row(uint32_t x, uint32_t row) {
    uint64_t z = static_cast<uint64_t>(x) + 0x9E3779B97F4A7C15ULL * (static_cast<uint64_t>(row) + 1ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return static_cast<uint32_t>(z);
}

} // namespace

tf_cache::tf_cache(const CacheConfig& cfg) : cfg_(cfg) {
    if (cfg_.cms_width < 16) cfg_.cms_width = 16;
    if (cfg_.cms_depth == 0) cfg_.cms_depth = 4;
    if (cfg_.age_interval == 0) cfg_.age_interval = 2048;

    slots_.resize(cfg_.capacity_slots);
    for (size_t i = 0; i < cfg_.capacity_slots; ++i) {
        slots_[i].segment = Segment::FREE;
        free_slots_.push_back(static_cast<int32_t>(i));
        slots_[i].list_it = std::prev(free_slots_.end());
    }
    cms_.assign(static_cast<size_t>(cfg_.cms_width) * cfg_.cms_depth, 0);
    recompute_segment_caps_locked();
}

void tf_cache::recompute_segment_caps_locked() {
    const size_t dyn_cap = (cfg_.capacity_slots > pinned_count_)
        ? (cfg_.capacity_slots - pinned_count_)
        : 0;

    if (dyn_cap == 0) {
        win_cap_ = prob_cap_ = prot_cap_ = 0;
        return;
    }
    if (cfg_.policy == CachePolicy::LRU || dyn_cap < 4) {
        win_cap_  = 0;
        prob_cap_ = dyn_cap;
        prot_cap_ = 0;
        return;
    }

    win_cap_  = std::max<size_t>(1, static_cast<size_t>(std::llround(static_cast<double>(dyn_cap) * cfg_.window_frac)));
    if (win_cap_ >= dyn_cap) win_cap_ = dyn_cap - 1;
    const size_t main_cap = dyn_cap - win_cap_;
    if (main_cap < 2) {
        prot_cap_ = 0;
        prob_cap_ = main_cap;
    } else {
        prot_cap_ = std::max<size_t>(1, static_cast<size_t>(std::llround(static_cast<double>(main_cap) * cfg_.protected_frac)));
        if (prot_cap_ >= main_cap) prot_cap_ = main_cap - 1;
        prob_cap_ = main_cap - prot_cap_;
    }
}

void tf_cache::record_freq_locked(uint32_t object_id) {
    ++freq_steps_;
    if ((freq_steps_ % cfg_.age_interval) == 0) {
        for (uint8_t& c : cms_) {
            c >>= 1;
        }
    }
    for (uint32_t r = 0; r < cfg_.cms_depth; ++r) {
        const size_t col = hash_row(object_id, r) % cfg_.cms_width;
        uint8_t& cell    = cms_[static_cast<size_t>(r) * cfg_.cms_width + col];
        if (cell < 15) ++cell;
    }
}

uint32_t tf_cache::est_freq_locked(uint32_t object_id) const {
    uint8_t min_v = 255;
    for (uint32_t r = 0; r < cfg_.cms_depth; ++r) {
        const size_t col = hash_row(object_id, r) % cfg_.cms_width;
        const uint8_t v  = cms_[static_cast<size_t>(r) * cfg_.cms_width + col];
        if (v < min_v) min_v = v;
    }
    return static_cast<uint32_t>(min_v);
}

int32_t tf_cache::alloc_free_slot_locked() {
    if (free_slots_.empty()) return -1;
    int32_t s = free_slots_.front();
    free_slots_.pop_front();
    return s;
}

void tf_cache::attach_to_segment_locked(int32_t slot_id, uint32_t object_id, Segment seg) {
    SlotMeta& m = slots_[static_cast<size_t>(slot_id)];
    m.object_id = object_id;
    m.segment   = seg;
    obj_to_slot_[object_id] = slot_id;

    switch (seg) {
        case Segment::FREE:
            free_slots_.push_back(slot_id);
            m.list_it = std::prev(free_slots_.end());
            break;
        case Segment::PINNED:
            break;
        case Segment::WINDOW:
            window_lru_.push_back(slot_id);
            m.list_it = std::prev(window_lru_.end());
            break;
        case Segment::PROBATION:
            probation_lru_.push_back(slot_id);
            m.list_it = std::prev(probation_lru_.end());
            break;
        case Segment::PROTECTED:
            protected_lru_.push_back(slot_id);
            m.list_it = std::prev(protected_lru_.end());
            break;
    }
}

void tf_cache::detach_from_segment_locked(int32_t slot_id) {
    SlotMeta& m = slots_[static_cast<size_t>(slot_id)];
    switch (m.segment) {
        case Segment::FREE:
            free_slots_.erase(m.list_it);
            break;
        case Segment::PINNED:
            break;
        case Segment::WINDOW:
            window_lru_.erase(m.list_it);
            break;
        case Segment::PROBATION:
            probation_lru_.erase(m.list_it);
            break;
        case Segment::PROTECTED:
            protected_lru_.erase(m.list_it);
            break;
    }
    if (m.segment != Segment::FREE) {
        obj_to_slot_.erase(m.object_id);
    }
    m.segment = Segment::FREE;
}

int32_t tf_cache::find_unreffed_lru_locked(const std::list<int32_t>& lst) const {
    for (int32_t slot_id : lst) {
        if (slots_[static_cast<size_t>(slot_id)].refcount == 0) {
            return slot_id;
        }
    }
    return -1;
}

bool tf_cache::pin(uint32_t object_id, int32_t* slot_out) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = obj_to_slot_.find(object_id);
    if (it != obj_to_slot_.end()) {
        const int32_t s = it->second;
        if (slots_[static_cast<size_t>(s)].segment == Segment::PINNED) {
            if (slot_out) *slot_out = s;
            return true;
        }
        detach_from_segment_locked(s);
        slots_[static_cast<size_t>(s)].segment   = Segment::PINNED;
        slots_[static_cast<size_t>(s)].object_id = object_id;
        obj_to_slot_[object_id] = s;
        ++pinned_count_;
        recompute_segment_caps_locked();
        if (slot_out) *slot_out = s;
        return true;
    }

    int32_t s = alloc_free_slot_locked();
    if (s < 0) {
        s = find_unreffed_lru_locked(probation_lru_);
        if (s < 0) s = find_unreffed_lru_locked(window_lru_);
        if (s < 0) s = find_unreffed_lru_locked(protected_lru_);
        if (s < 0) return false;
        detach_from_segment_locked(s);
    }
    slots_[static_cast<size_t>(s)].segment   = Segment::PINNED;
    slots_[static_cast<size_t>(s)].object_id = object_id;
    obj_to_slot_[object_id] = s;
    ++pinned_count_;
    recompute_segment_caps_locked();
    if (slot_out) *slot_out = s;
    return true;
}

void tf_cache::admit_to_probation_locked(uint32_t cand_obj, int32_t slot_hint,
                                         bool& did_evict, uint32_t& evicted_obj,
                                         int32_t& assigned_slot) {
    // slot_hint is an already-free slot (e.g. from window victim) if >= 0.
    // Wait: if we simply keep cand_obj in slot_hint vs swapping with probation victim:
    // Notice we only reach here if Window evicted `cand_obj` (releasing `slot_hint` which held
    // `cand_obj`, OR `slot_hint == -1` when win_cap == 0).
    if (prob_cap_ == 0 && prot_cap_ == 0) {
        if (slot_hint >= 0) {
            free_slots_.push_back(slot_hint);
            slots_[static_cast<size_t>(slot_hint)].list_it = std::prev(free_slots_.end());
        }
        assigned_slot = -1;
        return;
    }

    if (probation_lru_.size() + protected_lru_.size() < prob_cap_ + prot_cap_) {
        int32_t dst = (slot_hint >= 0) ? slot_hint : alloc_free_slot_locked();
        if (dst >= 0) {
            attach_to_segment_locked(dst, cand_obj, Segment::PROBATION);
            assigned_slot = dst;
            return;
        }
    }

    int32_t prob_victim = find_unreffed_lru_locked(probation_lru_);
    if (prob_victim < 0 && !protected_lru_.empty()) {
        int32_t prot_demote = find_unreffed_lru_locked(protected_lru_);
        if (prot_demote >= 0) {
            uint32_t demote_obj = slots_[static_cast<size_t>(prot_demote)].object_id;
            detach_from_segment_locked(prot_demote);
            attach_to_segment_locked(prot_demote, demote_obj, Segment::PROBATION);
            prob_victim = prot_demote;
        }
    }

    if (prob_victim < 0) {
        // Every main-cache slot is referenced (refcount > 0); cannot evict!
        if (slot_hint >= 0) {
            free_slots_.push_back(slot_hint);
            slots_[static_cast<size_t>(slot_hint)].list_it = std::prev(free_slots_.end());
        }
        assigned_slot = -1;
        return;
    }

    const uint32_t vic_obj = slots_[static_cast<size_t>(prob_victim)].object_id;
    if (cfg_.policy == CachePolicy::LRU || est_freq_locked(cand_obj) >= est_freq_locked(vic_obj)) {
        // Admit candidate into probation by evicting prob_victim
        did_evict   = true;
        evicted_obj = vic_obj;
        detach_from_segment_locked(prob_victim);
        if (slot_hint >= 0) {
            free_slots_.push_back(prob_victim);
            slots_[static_cast<size_t>(prob_victim)].list_it = std::prev(free_slots_.end());
            attach_to_segment_locked(slot_hint, cand_obj, Segment::PROBATION);
            assigned_slot = slot_hint;
        } else {
            attach_to_segment_locked(prob_victim, cand_obj, Segment::PROBATION);
            assigned_slot = prob_victim;
        }
    } else {
        // Rejected by TinyLFU admission filter: cand_obj leaves cache, vic_obj stays!
        did_evict   = true;
        evicted_obj = cand_obj;
        if (slot_hint >= 0) {
            free_slots_.push_back(slot_hint);
            slots_[static_cast<size_t>(slot_hint)].list_it = std::prev(free_slots_.end());
        }
        assigned_slot = -1;
    }
}

CacheAccessResult tf_cache::access(uint32_t object_id) {
    std::lock_guard<std::mutex> lock(mu_);
    record_freq_locked(object_id);

    CacheAccessResult res{};
    if (cfg_.capacity_slots == 0) {
        ++misses_;
        return res;
    }

    auto it = obj_to_slot_.find(object_id);
    if (it != obj_to_slot_.end()) {
        ++hits_;
        const int32_t s = it->second;
        SlotMeta& m     = slots_[static_cast<size_t>(s)];
        res.hit         = true;
        res.slot_id     = s;

        if (m.segment == Segment::WINDOW) {
            window_lru_.splice(window_lru_.end(), window_lru_, m.list_it);
        } else if (m.segment == Segment::PROTECTED) {
            protected_lru_.splice(protected_lru_.end(), protected_lru_, m.list_it);
        } else if (m.segment == Segment::PROBATION) {
            if (cfg_.policy == CachePolicy::LRU || prot_cap_ == 0) {
                probation_lru_.splice(probation_lru_.end(), probation_lru_, m.list_it);
            } else {
                // Promote Probation -> Protected
                detach_from_segment_locked(s);
                if (protected_lru_.size() >= prot_cap_) {
                    int32_t demote = find_unreffed_lru_locked(protected_lru_);
                    if (demote < 0) demote = protected_lru_.front();
                    uint32_t demote_obj = slots_[static_cast<size_t>(demote)].object_id;
                    detach_from_segment_locked(demote);
                    attach_to_segment_locked(demote, demote_obj, Segment::PROBATION);
                }
                attach_to_segment_locked(s, object_id, Segment::PROTECTED);
            }
        }
        return res;
    }

    // Cache miss
    ++misses_;

    if (cfg_.policy == CachePolicy::LRU || win_cap_ == 0) {
        int32_t dst = alloc_free_slot_locked();
        if (dst >= 0) {
            attach_to_segment_locked(dst, object_id, Segment::PROBATION);
            res.slot_id = dst;
            return res;
        }
        int32_t victim = find_unreffed_lru_locked(probation_lru_);
        if (victim < 0) victim = find_unreffed_lru_locked(protected_lru_);
        if (victim < 0) victim = find_unreffed_lru_locked(window_lru_);
        if (victim < 0) {
            res.slot_id = -1; // all slots pinned or referenced
            return res;
        }
        res.evicted           = true;
        res.evicted_object_id = slots_[static_cast<size_t>(victim)].object_id;
        detach_from_segment_locked(victim);
        attach_to_segment_locked(victim, object_id, Segment::PROBATION);
        res.slot_id = victim;
        return res;
    }

    // W-TinyLFU + SLRU miss path: enter Window LRU first
    if (window_lru_.size() < win_cap_) {
        int32_t free_s = alloc_free_slot_locked();
        if (free_s >= 0) {
            attach_to_segment_locked(free_s, object_id, Segment::WINDOW);
            res.slot_id = free_s;
            return res;
        }
    }

    int32_t win_victim = find_unreffed_lru_locked(window_lru_);
    if (win_victim < 0) {
        // Every window slot has refcount > 0; check if any free/probation slot exists
        int32_t free_s = alloc_free_slot_locked();
        if (free_s >= 0) {
            attach_to_segment_locked(free_s, object_id, Segment::WINDOW);
            res.slot_id = free_s;
            return res;
        }
        int32_t fallback = find_unreffed_lru_locked(probation_lru_);
        if (fallback < 0) fallback = find_unreffed_lru_locked(protected_lru_);
        if (fallback < 0) {
            res.slot_id = -1;
            return res;
        }
        res.evicted           = true;
        res.evicted_object_id = slots_[static_cast<size_t>(fallback)].object_id;
        detach_from_segment_locked(fallback);
        attach_to_segment_locked(fallback, object_id, Segment::PROBATION);
        res.slot_id = fallback;
        return res;
    }

    // Remove win_victim from Window and put new item `object_id` directly in Window!
    const uint32_t win_victim_obj = slots_[static_cast<size_t>(win_victim)].object_id;
    detach_from_segment_locked(win_victim);
    attach_to_segment_locked(win_victim, object_id, Segment::WINDOW);
    res.slot_id = win_victim;

    // Challenge win_victim_obj into Probation via TinyLFU
    int32_t unused_slot = -1;
    admit_to_probation_locked(win_victim_obj, -1, res.evicted, res.evicted_object_id, unused_slot);
    return res;
}

CacheAccessResult tf_cache::prefetch(uint32_t object_id) {
    std::lock_guard<std::mutex> lock(mu_);
    CacheAccessResult res{};
    auto it = obj_to_slot_.find(object_id);
    if (it != obj_to_slot_.end()) {
        res.hit     = true;
        res.slot_id = it->second;
        return res;
    }
    if (prob_cap_ == 0) {
        return res;
    }
    if (probation_lru_.size() + protected_lru_.size() < prob_cap_ + prot_cap_) {
        int32_t free_s = alloc_free_slot_locked();
        if (free_s >= 0) {
            attach_to_segment_locked(free_s, object_id, Segment::PROBATION);
            res.slot_id = free_s;
            return res;
        }
    }
    int32_t victim = find_unreffed_lru_locked(probation_lru_);
    if (victim >= 0) {
        res.evicted           = true;
        res.evicted_object_id = slots_[static_cast<size_t>(victim)].object_id;
        detach_from_segment_locked(victim);
        attach_to_segment_locked(victim, object_id, Segment::PROBATION);
        res.slot_id = victim;
    }
    return res;
}

int32_t tf_cache::find_slot(uint32_t object_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = obj_to_slot_.find(object_id);
    return (it == obj_to_slot_.end()) ? -1 : it->second;
}

bool tf_cache::acquire_slot(int32_t slot_id) {
    std::lock_guard<std::mutex> lock(mu_);
    if (slot_id < 0 || static_cast<size_t>(slot_id) >= slots_.size()) return false;
    SlotMeta& m = slots_[static_cast<size_t>(slot_id)];
    if (m.segment == Segment::FREE) return false;
    ++m.refcount;
    return true;
}

bool tf_cache::release_slot(int32_t slot_id) {
    std::lock_guard<std::mutex> lock(mu_);
    if (slot_id < 0 || static_cast<size_t>(slot_id) >= slots_.size()) return false;
    SlotMeta& m = slots_[static_cast<size_t>(slot_id)];
    if (m.refcount == 0) return false;
    --m.refcount;
    return true;
}

uint32_t tf_cache::refcount(int32_t slot_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (slot_id < 0 || static_cast<size_t>(slot_id) >= slots_.size()) return 0;
    return slots_[static_cast<size_t>(slot_id)].refcount;
}

bool tf_cache::is_resident(uint32_t object_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    return obj_to_slot_.find(object_id) != obj_to_slot_.end();
}

size_t tf_cache::resident_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return obj_to_slot_.size();
}

uint64_t tf_cache::hits() const {
    std::lock_guard<std::mutex> lock(mu_);
    return hits_;
}

uint64_t tf_cache::misses() const {
    std::lock_guard<std::mutex> lock(mu_);
    return misses_;
}

double tf_cache::hit_rate() const {
    std::lock_guard<std::mutex> lock(mu_);
    const uint64_t total = hits_ + misses_;
    return total == 0 ? 0.0 : static_cast<double>(hits_) / static_cast<double>(total);
}

uint64_t belady_oracle_misses(const std::vector<uint32_t>& trace, size_t capacity_slots) {
    if (capacity_slots == 0) return static_cast<uint64_t>(trace.size());

    const size_t n = trace.size();
    std::unordered_map<uint32_t, std::vector<size_t>> positions;
    for (size_t i = 0; i < n; ++i) {
        positions[trace[i]].push_back(i);
    }
    std::unordered_map<uint32_t, size_t> cursor;

    std::unordered_set<uint32_t> resident;
    uint64_t misses = 0;

    for (size_t i = 0; i < n; ++i) {
        const uint32_t item = trace[i];
        cursor[item] += 1;
        if (resident.find(item) != resident.end()) {
            continue;
        }
        ++misses;
        if (resident.size() >= capacity_slots) {
            uint32_t victim     = 0;
            size_t   furthest   = 0;
            bool     have_vic   = false;
            for (uint32_t r : resident) {
                const auto& pos = positions[r];
                const size_t c  = cursor[r];
                const size_t nxt = (c < pos.size()) ? pos[c] : (std::numeric_limits<size_t>::max)();
                if (!have_vic || nxt > furthest) {
                    furthest = nxt;
                    victim   = r;
                    have_vic = true;
                }
            }
            if (have_vic) resident.erase(victim);
        }
        resident.insert(item);
    }
    return misses;
}

} // namespace tierflow

