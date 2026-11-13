#include "tf_kv.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace tierflow {

namespace {

size_t round_up_4096(size_t n) {
    return (n + 4095) & ~size_t(4095);
}

size_t row_bytes_for(KvPrecision prec, bool is_v, size_t elems) {
    if (prec == KvPrecision::F16) {
        return elems * 2;
    }
    if (prec == KvPrecision::Q8_K_Q4_V && is_v) {
        // q4_0: 32 elems -> 2B scale + 16B nibbles = 18B
        return ((elems + 31) / 32) * 18;
    }
    // q8_0: 32 elems -> 2B scale + 32B int8 = 34B
    return ((elems + 31) / 32) * 34;
}

} // namespace

uint16_t f32_to_f16_bits(float x) {
    uint32_t u = 0;
    std::memcpy(&u, &x, sizeof(u));
    uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp = int32_t((u >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = u & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        uint32_t shift = uint32_t(14 - exp);
        return static_cast<uint16_t>(sign | ((mant + (1u << (shift - 1))) >> shift));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    uint32_t rounded = mant + 0x1000u;
    if (rounded & 0x800000u) {
        rounded = 0;
        ++exp;
        if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    }
    return static_cast<uint16_t>(sign | (uint32_t(exp) << 10) | (rounded >> 13));
}

float f16_bits_to_f32(uint16_t h) {
    uint32_t sign = uint32_t(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t u = 0;
    if (exp == 0) {
        if (mant == 0) {
            u = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03FFu;
            u = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7F800000u | (mant << 13);
    } else {
        u = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &u, sizeof(out));
    return out;
}

void quantize_row_q8_0(const float* src, void* dst, size_t n) {
    auto* out = static_cast<uint8_t*>(dst);
    const size_t nb = (n + 31) / 32;
    for (size_t b = 0; b < nb; ++b) {
        const size_t off = b * 32;
        const size_t cnt = std::min<size_t>(32, n - off);
        float amax = 0.0f;
        for (size_t i = 0; i < cnt; ++i) {
            amax = std::max(amax, std::fabs(src[off + i]));
        }
        const float d = amax / 127.0f;
        const float id = (d > 0.0f) ? (127.0f / amax) : 0.0f;
        uint16_t d16 = f32_to_f16_bits(d);
        std::memcpy(out + b * 34, &d16, 2);
        auto* qs = reinterpret_cast<int8_t*>(out + b * 34 + 2);
        for (size_t i = 0; i < 32; ++i) {
            if (i < cnt) {
                int q = static_cast<int>(std::llround(src[off + i] * id));
                qs[i] = static_cast<int8_t>(std::clamp(q, -127, 127));
            } else {
                qs[i] = 0;
            }
        }
    }
}

void dequantize_row_q8_0(const void* src, float* dst, size_t n) {
    const auto* in = static_cast<const uint8_t*>(src);
    const size_t nb = (n + 31) / 32;
    for (size_t b = 0; b < nb; ++b) {
        uint16_t d16 = 0;
        std::memcpy(&d16, in + b * 34, 2);
        const float d = f16_bits_to_f32(d16);
        const auto* qs = reinterpret_cast<const int8_t*>(in + b * 34 + 2);
        const size_t off = b * 32;
        const size_t cnt = std::min<size_t>(32, n - off);
        for (size_t i = 0; i < cnt; ++i) {
            dst[off + i] = static_cast<float>(qs[i]) * d;
        }
    }
}

void quantize_row_q4_0(const float* src, void* dst, size_t n) {
    auto* out = static_cast<uint8_t*>(dst);
    const size_t nb = (n + 31) / 32;
    for (size_t b = 0; b < nb; ++b) {
        const size_t off = b * 32;
        const size_t cnt = std::min<size_t>(32, n - off);
        float amax = 0.0f;
        for (size_t i = 0; i < cnt; ++i) {
            amax = std::max(amax, std::fabs(src[off + i]));
        }
        const float d = amax / 7.0f;
        const float id = (d > 0.0f) ? (7.0f / amax) : 0.0f;
        uint16_t d16 = f32_to_f16_bits(d);
        std::memcpy(out + b * 18, &d16, 2);
        uint8_t* qs = out + b * 18 + 2;
        for (size_t i = 0; i < 16; ++i) {
            float x0 = (off + i < n) ? src[off + i] : 0.0f;
            float x1 = (off + i + 16 < n) ? src[off + i + 16] : 0.0f;
            int q0 = std::clamp(static_cast<int>(std::llround(x0 * id)) + 8, 0, 15);
            int q1 = std::clamp(static_cast<int>(std::llround(x1 * id)) + 8, 0, 15);
            qs[i] = static_cast<uint8_t>((q0 & 0xF) | ((q1 & 0xF) << 4));
        }
    }
}

void dequantize_row_q4_0(const void* src, float* dst, size_t n) {
    const auto* in = static_cast<const uint8_t*>(src);
    const size_t nb = (n + 31) / 32;
    for (size_t b = 0; b < nb; ++b) {
        uint16_t d16 = 0;
        std::memcpy(&d16, in + b * 18, 2);
        const float d = f16_bits_to_f32(d16);
        const uint8_t* qs = in + b * 18 + 2;
        const size_t off = b * 32;
        for (size_t i = 0; i < 16; ++i) {
            const int q0 = (qs[i] & 0xF) - 8;
            const int q1 = ((qs[i] >> 4) & 0xF) - 8;
            if (off + i < n) dst[off + i] = static_cast<float>(q0) * d;
            if (off + i + 16 < n) dst[off + i + 16] = static_cast<float>(q1) * d;
        }
    }
}

tf_kv_store::tf_kv_store(const KvConfig& cfg, tf_governor& gov)
    : cfg_(cfg), gov_(gov) {
    k_row_elems_   = static_cast<size_t>(cfg_.n_kv_heads) * cfg_.head_dim_k;
    v_row_elems_   = static_cast<size_t>(cfg_.n_kv_heads) * cfg_.head_dim_v;
    const size_t k_elems_page = k_row_elems_ * cfg_.page_tokens;
    const size_t v_elems_page = v_row_elems_ * cfg_.page_tokens;
    k_page_bytes_  = row_bytes_for(cfg_.precision, false, k_elems_page);
    v_page_bytes_  = row_bytes_for(cfg_.precision, true, v_elems_page);
    payload_bytes_ = k_page_bytes_ + v_page_bytes_;
    aligned_bytes_ = round_up_4096(payload_bytes_);
}

tf_kv_store::~tf_kv_store() {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& [k, e] : pages_) {
        if (e.in_ram && e.ram_buf) {
            tf_aligned_free(e.ram_buf);
            gov_.release(MemoryTier::RAM, aligned_bytes_, "kv_pool");
        }
        if (e.on_ssd) {
            gov_.release(MemoryTier::SSD_SPILL, aligned_bytes_, "kv_spill");
        }
    }
    pages_.clear();
    ram_lru_.clear();
#ifdef _WIN32
    if (spill_handle_ && spill_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(static_cast<HANDLE>(spill_handle_));
        spill_handle_ = nullptr;
    }
#else
    if (spill_fd_ >= 0) {
        ::close(spill_fd_);
        spill_fd_ = -1;
    }
#endif
}

bool tf_kv_store::ensure_spill_open_locked() {
    if (cfg_.spill_path.empty()) return false;
#ifdef _WIN32
    if (spill_handle_ && spill_handle_ != INVALID_HANDLE_VALUE) return true;
    std::error_code ec;
    auto parent = std::filesystem::path(cfg_.spill_path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent, ec);
    HANDLE h = CreateFileA(
        cfg_.spill_path.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0,
        nullptr,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS | FILE_FLAG_DELETE_ON_CLOSE,
        nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return false;
    }
    spill_handle_ = h;
    return true;
#else
    if (spill_fd_ >= 0) return true;
    std::error_code ec;
    auto parent = std::filesystem::path(cfg_.spill_path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent, ec);
    int fd = ::open(cfg_.spill_path.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_DIRECT, 0600);
    if (fd < 0 && (errno == EINVAL || errno == EOPNOTSUPP)) {
        fd = ::open(cfg_.spill_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
    }
    if (fd < 0) {
        return false;
    }
    ::unlink(cfg_.spill_path.c_str());
    spill_fd_ = fd;
    return true;
#endif
}

uint64_t tf_kv_store::alloc_ssd_slot_locked() {
    if (!free_ssd_slots_.empty()) {
        uint64_t s = free_ssd_slots_.back();
        free_ssd_slots_.pop_back();
        return s;
    }
    return next_ssd_slot_++;
}

bool tf_kv_store::write_ssd_slot_locked(uint64_t slot, const void* src_aligned) {
    if (!ensure_spill_open_locked()) return false;
#ifdef _WIN32
    HANDLE h = static_cast<HANDLE>(spill_handle_);
    const uint64_t off = slot * aligned_bytes_;
    OVERLAPPED ov{};
    ov.Offset     = static_cast<DWORD>(off & 0xFFFFFFFFULL);
    ov.OffsetHigh = static_cast<DWORD>(off >> 32);
    DWORD written = 0;
    if (!WriteFile(h, src_aligned, static_cast<DWORD>(aligned_bytes_), &written, &ov) &&
        GetLastError() != ERROR_IO_PENDING) {
        return false;
    }
    if (!GetOverlappedResult(h, &ov, &written, TRUE) || written != aligned_bytes_) {
        return false;
    }
    ++stats_.spill_writes;
    stats_.spill_bytes_written += aligned_bytes_;
    return true;
#else
    const uint8_t* p = static_cast<const uint8_t*>(src_aligned);
    off_t off = static_cast<off_t>(slot * aligned_bytes_);
    size_t rem = aligned_bytes_;
    while (rem > 0) {
        ssize_t w = ::pwrite(spill_fd_, p, rem, off);
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (w == 0) return false;
        p += w;
        off += w;
        rem -= static_cast<size_t>(w);
    }
    ++stats_.spill_writes;
    stats_.spill_bytes_written += aligned_bytes_;
    return true;
#endif
}

bool tf_kv_store::read_ssd_slot_locked(uint64_t slot, void* dst_aligned) {
    if (!ensure_spill_open_locked()) return false;
#ifdef _WIN32
    HANDLE h = static_cast<HANDLE>(spill_handle_);
    const uint64_t off = slot * aligned_bytes_;
    OVERLAPPED ov{};
    ov.Offset     = static_cast<DWORD>(off & 0xFFFFFFFFULL);
    ov.OffsetHigh = static_cast<DWORD>(off >> 32);
    DWORD bytes_read = 0;
    if (!ReadFile(h, dst_aligned, static_cast<DWORD>(aligned_bytes_), &bytes_read, &ov) &&
        GetLastError() != ERROR_IO_PENDING) {
        return false;
    }
    if (!GetOverlappedResult(h, &ov, &bytes_read, TRUE) || bytes_read != aligned_bytes_) {
        return false;
    }
    ++stats_.spill_reads;
    stats_.spill_bytes_read += aligned_bytes_;
    return true;
#else
    uint8_t* p = static_cast<uint8_t*>(dst_aligned);
    off_t off = static_cast<off_t>(slot * aligned_bytes_);
    size_t rem = aligned_bytes_;
    while (rem > 0) {
        ssize_t r = ::pread(spill_fd_, p, rem, off);
        if (r < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (r == 0) return false;
        p += r;
        off += r;
        rem -= static_cast<size_t>(r);
    }
    ++stats_.spill_reads;
    stats_.spill_bytes_read += aligned_bytes_;
    return true;
#endif
}

bool tf_kv_store::spill_one_lru_locked(const PageKey* avoid_key) {
    if (ram_lru_.empty()) return false;
    PageKey victim{};
    bool found = false;
    for (auto it = ram_lru_.begin(); it != ram_lru_.end(); ++it) {
        if (avoid_key && *it == *avoid_key) continue;
        victim = *it;
        found = true;
        break;
    }
    if (!found) return false;
    auto p_it = pages_.find(victim);
    if (p_it == pages_.end()) return false;
    PageEntry& e = p_it->second;
    if (!gov_.reserve(MemoryTier::SSD_SPILL, aligned_bytes_, "kv_spill")) {
        return false;
    }
    const uint64_t slot = alloc_ssd_slot_locked();
    if (!write_ssd_slot_locked(slot, e.ram_buf)) {
        gov_.release(MemoryTier::SSD_SPILL, aligned_bytes_, "kv_spill");
        free_ssd_slots_.push_back(slot);
        return false;
    }
    ram_lru_.erase(e.lru_it);
    tf_aligned_free(e.ram_buf);
    e.ram_buf  = nullptr;
    e.in_ram   = false;
    e.on_ssd   = true;
    e.ssd_slot = slot;
    gov_.release(MemoryTier::RAM, aligned_bytes_, "kv_pool");
    if (stats_.pages_resident_ram > 0) --stats_.pages_resident_ram;
    ++stats_.pages_spilled_ssd;
    return true;
}

bool tf_kv_store::put_page_f32(uint64_t seq_id,
                               uint32_t layer,
                               uint32_t page_idx,
                               uint32_t n_tokens,
                               const float* k_f32,
                               const float* v_f32) {
    if (n_tokens == 0 || n_tokens > cfg_.page_tokens || !k_f32 || !v_f32) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mu_);
    PageKey key{seq_id, layer, page_idx};
    auto it = pages_.find(key);
    void* buf = nullptr;
    if (it != pages_.end() && it->second.in_ram) {
        buf = it->second.ram_buf;
        ram_lru_.erase(it->second.lru_it);
    } else {
        while (!gov_.reserve(MemoryTier::RAM, aligned_bytes_, "kv_pool")) {
            if (!spill_one_lru_locked(&key)) {
                return false;
            }
        }
        buf = tf_aligned_alloc(aligned_bytes_, 4096);
        if (!buf) {
            gov_.release(MemoryTier::RAM, aligned_bytes_, "kv_pool");
            return false;
        }
        std::memset(buf, 0, aligned_bytes_);
    }

    const size_t k_elems = size_t(n_tokens) * k_row_elems_;
    const size_t v_elems = size_t(n_tokens) * v_row_elems_;
    auto* base = static_cast<uint8_t*>(buf);

    if (cfg_.precision == KvPrecision::F16) {
        auto* kp = reinterpret_cast<uint16_t*>(base);
        auto* vp = reinterpret_cast<uint16_t*>(base + k_page_bytes_);
        for (size_t i = 0; i < k_elems; ++i) kp[i] = f32_to_f16_bits(k_f32[i]);
        for (size_t i = 0; i < v_elems; ++i) vp[i] = f32_to_f16_bits(v_f32[i]);
    } else if (cfg_.precision == KvPrecision::Q8_0) {
        quantize_row_q8_0(k_f32, base, k_elems);
        quantize_row_q8_0(v_f32, base + k_page_bytes_, v_elems);
    } else {
        quantize_row_q8_0(k_f32, base, k_elems);
        quantize_row_q4_0(v_f32, base + k_page_bytes_, v_elems);
    }

    ram_lru_.push_back(key);
    auto lru_it = std::prev(ram_lru_.end());

    if (it == pages_.end()) {
        PageEntry e;
        e.key      = key;
        e.n_tokens = n_tokens;
        e.in_ram   = true;
        e.ram_buf  = buf;
        e.lru_it   = lru_it;
        pages_.emplace(key, e);
        ++stats_.pages_resident_ram;
    } else {
        if (!it->second.in_ram) {
            ++stats_.pages_resident_ram;
            if (it->second.on_ssd) {
                free_ssd_slots_.push_back(it->second.ssd_slot);
                gov_.release(MemoryTier::SSD_SPILL, aligned_bytes_, "kv_spill");
                it->second.on_ssd   = false;
                it->second.ssd_slot = UINT64_MAX;
                if (stats_.pages_spilled_ssd > 0) --stats_.pages_spilled_ssd;
            }
        }
        it->second.n_tokens = n_tokens;
        it->second.in_ram   = true;
        it->second.ram_buf  = buf;
        it->second.lru_it   = lru_it;
    }
    return true;
}

bool tf_kv_store::get_page_f32(uint64_t seq_id,
                               uint32_t layer,
                               uint32_t page_idx,
                               std::vector<float>& k_out,
                               std::vector<float>& v_out) {
    std::lock_guard<std::mutex> lock(mu_);
    PageKey key{seq_id, layer, page_idx};
    auto it = pages_.find(key);
    if (it == pages_.end()) return false;
    PageEntry& e = it->second;

    if (!e.in_ram) {
        if (!e.on_ssd) return false;
        while (!gov_.reserve(MemoryTier::RAM, aligned_bytes_, "kv_pool")) {
            if (!spill_one_lru_locked(&key)) {
                return false;
            }
        }
        void* buf = tf_aligned_alloc(aligned_bytes_, 4096);
        if (!buf) {
            gov_.release(MemoryTier::RAM, aligned_bytes_, "kv_pool");
            return false;
        }
        if (!read_ssd_slot_locked(e.ssd_slot, buf)) {
            tf_aligned_free(buf);
            gov_.release(MemoryTier::RAM, aligned_bytes_, "kv_pool");
            return false;
        }
        e.ram_buf = buf;
        e.in_ram  = true;
        ram_lru_.push_back(key);
        e.lru_it = std::prev(ram_lru_.end());
        ++stats_.pages_resident_ram;
    } else {
        ram_lru_.splice(ram_lru_.end(), ram_lru_, e.lru_it);
        e.lru_it = std::prev(ram_lru_.end());
    }

    const size_t k_elems = size_t(e.n_tokens) * k_row_elems_;
    const size_t v_elems = size_t(e.n_tokens) * v_row_elems_;
    k_out.resize(k_elems);
    v_out.resize(v_elems);
    const auto* base = static_cast<const uint8_t*>(e.ram_buf);

    if (cfg_.precision == KvPrecision::F16) {
        const auto* kp = reinterpret_cast<const uint16_t*>(base);
        const auto* vp = reinterpret_cast<const uint16_t*>(base + k_page_bytes_);
        for (size_t i = 0; i < k_elems; ++i) k_out[i] = f16_bits_to_f32(kp[i]);
        for (size_t i = 0; i < v_elems; ++i) v_out[i] = f16_bits_to_f32(vp[i]);
    } else if (cfg_.precision == KvPrecision::Q8_0) {
        dequantize_row_q8_0(base, k_out.data(), k_elems);
        dequantize_row_q8_0(base + k_page_bytes_, v_out.data(), v_elems);
    } else {
        dequantize_row_q8_0(base, k_out.data(), k_elems);
        dequantize_row_q4_0(base + k_page_bytes_, v_out.data(), v_elems);
    }
    return true;
}

bool tf_kv_store::save_prefix_block(uint64_t prefix_hash, const void* state_bytes, size_t bytes) {
    if (cfg_.prefix_dir.empty() || !state_bytes || bytes == 0) return false;
    std::error_code ec;
    std::filesystem::create_directories(cfg_.prefix_dir, ec);
    char fname[64];
    std::snprintf(fname, sizeof(fname), "prefix_%016llx.bin", static_cast<unsigned long long>(prefix_hash));
    const auto path = (std::filesystem::path(cfg_.prefix_dir) / fname).string();
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) return false;
    ofs.write(static_cast<const char*>(state_bytes), static_cast<std::streamsize>(bytes));
    return ofs.good();
}

bool tf_kv_store::load_prefix_block(uint64_t prefix_hash, std::vector<uint8_t>& state_out) {
    if (cfg_.prefix_dir.empty()) {
        std::lock_guard<std::mutex> lock(mu_);
        ++stats_.prefix_misses;
        return false;
    }
    char fname[64];
    std::snprintf(fname, sizeof(fname), "prefix_%016llx.bin", static_cast<unsigned long long>(prefix_hash));
    const auto path = (std::filesystem::path(cfg_.prefix_dir) / fname).string();
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        std::lock_guard<std::mutex> lock(mu_);
        ++stats_.prefix_misses;
        return false;
    }
    const auto sz = ifs.tellg();
    if (sz <= 0) {
        std::lock_guard<std::mutex> lock(mu_);
        ++stats_.prefix_misses;
        return false;
    }
    state_out.resize(static_cast<size_t>(sz));
    ifs.seekg(0, std::ios::beg);
    ifs.read(reinterpret_cast<char*>(state_out.data()), sz);
    std::lock_guard<std::mutex> lock(mu_);
    if (!ifs.good()) {
        ++stats_.prefix_misses;
        return false;
    }
    ++stats_.prefix_hits;
    return true;
}

void tf_kv_store::evict_sequence(uint64_t seq_id) {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto it = pages_.begin(); it != pages_.end(); ) {
        if (it->first.seq_id == seq_id) {
            if (it->second.in_ram) {
                ram_lru_.erase(it->second.lru_it);
                tf_aligned_free(it->second.ram_buf);
                gov_.release(MemoryTier::RAM, aligned_bytes_, "kv_pool");
                if (stats_.pages_resident_ram > 0) --stats_.pages_resident_ram;
            }
            if (it->second.on_ssd) {
                free_ssd_slots_.push_back(it->second.ssd_slot);
                gov_.release(MemoryTier::SSD_SPILL, aligned_bytes_, "kv_spill");
                if (stats_.pages_spilled_ssd > 0) --stats_.pages_spilled_ssd;
            }
            it = pages_.erase(it);
        } else {
            ++it;
        }
    }
}

KvPageStats tf_kv_store::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    return stats_;
}

} // namespace tierflow
