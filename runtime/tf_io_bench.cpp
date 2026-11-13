#include "tf_io.h"
#include "tf_mem_win.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct SummaryStats {
    size_t n    = 0;
    double mean = 0.0;
    double sd   = 0.0;
    double ci95 = 0.0;
    double min  = 0.0;
    double max  = 0.0;
    double p50  = 0.0;
    double p95  = 0.0;
};

double t_crit_95(size_t df) {
    // Two-sided Student-t 97.5th percentile for exact small-sample CIs
    static const double tbl[] = {
        0.0,     12.7062, 4.3027, 3.1824, 2.7764, 2.5706, 2.4469, 2.3646, 2.3060, 2.2622,
        2.2281,  2.2010,  2.1788, 2.1604, 2.1448, 2.1314, 2.1199, 2.1098, 2.1009, 2.0930,
        2.0860,  2.0796,  2.0739, 2.0687, 2.0639, 2.0595, 2.0555, 2.0518, 2.0484, 2.0452};
    if (df == 0) return 0.0;
    if (df < sizeof(tbl) / sizeof(tbl[0])) return tbl[df];
    return 1.96;
}

SummaryStats compute_stats(std::vector<double> vals) {
    SummaryStats s{};
    if (vals.empty()) return s;
    s.n = vals.size();
    std::sort(vals.begin(), vals.end());
    s.min = vals.front();
    s.max = vals.back();
    s.p50 = vals[vals.size() / 2];
    size_t idx95 = static_cast<size_t>(std::ceil(0.95 * static_cast<double>(vals.size()))) - 1;
    if (idx95 >= vals.size()) idx95 = vals.size() - 1;
    s.p95 = vals[idx95];

    double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
    s.mean     = sum / static_cast<double>(s.n);
    if (s.n > 1) {
        double sq = 0.0;
        for (double v : vals) sq += (v - s.mean) * (v - s.mean);
        s.sd   = std::sqrt(sq / static_cast<double>(s.n - 1));
        s.ci95 = t_crit_95(s.n - 1) * s.sd / std::sqrt(static_cast<double>(s.n));
    }
    return s;
}

#ifdef _WIN32

bool create_unbuffered_shard(const std::string& path, uint64_t total_bytes) {
    std::printf("[SETUP] Creating %.2f GiB unbuffered NVMe test shard at %s ...\n",
                static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0), path.c_str());

    int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring wpath(static_cast<size_t>(wlen), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen);

    HANDLE hf = CreateFileW(
        wpath.c_str(),
        GENERIC_WRITE,
        0,
        nullptr,
        CREATE_ALWAYS,
        FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH,
        nullptr);
    if (hf == INVALID_HANDLE_VALUE) {
        std::fprintf(stderr, "CreateFileW write failed: err=%lu\n", GetLastError());
        return false;
    }

    constexpr uint64_t kChunk = 4ULL << 20;
    void* buf = tierflow::tf_aligned_alloc(kChunk, 4096);
    if (!buf) {
        CloseHandle(hf);
        return false;
    }

    // Fill with non-zero deterministic pattern so NTFS allocates actual non-zero flash blocks
    std::mt19937_64 rng(0x54696572466C6F77ULL);
    uint64_t* words = static_cast<uint64_t*>(buf);
    for (size_t i = 0; i < kChunk / 8; ++i) {
        words[i] = rng() | 1ULL; // ensure never zero
    }

    auto t0 = Clock::now();
    uint64_t written_total = 0;
    while (written_total < total_bytes) {
        DWORD to_write = static_cast<DWORD>(std::min<uint64_t>(kChunk, total_bytes - written_total));
        DWORD written  = 0;
        if (!WriteFile(hf, buf, to_write, &written, nullptr) || written != to_write) {
            std::fprintf(stderr, "WriteFile failed at %llu: err=%lu\n",
                         static_cast<unsigned long long>(written_total), GetLastError());
            tierflow::tf_aligned_free(buf);
            CloseHandle(hf);
            return false;
        }
        written_total += written;
        // Slightly perturb first word so every chunk is distinct
        words[0] ^= written_total;
    }

    FlushFileBuffers(hf);
    CloseHandle(hf);
    tierflow::tf_aligned_free(buf);
    double sec = std::chrono::duration<double>(Clock::now() - t0).count();
    std::printf("[SETUP] Shard ready in %.2f s (%.2f GB/s write)\n",
                sec, (static_cast<double>(total_bytes) / 1e9) / sec);
    return true;
}

// Raw Win32 IOCP QD=N baseline directly using ReadFile/OVERLAPPED without tf_io wrapper
SummaryStats run_raw_iocp_seq_baseline(const std::string& path, uint64_t file_size,
                                       uint64_t block_size, size_t qd,
                                       uint64_t bytes_per_rep, int reps) {
    int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring wpath(static_cast<size_t>(wlen), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen);

    HANDLE hf = CreateFileW(
        wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
        FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS, nullptr);
    HANDLE iocp = CreateIoCompletionPort(hf, nullptr, 0, 1);

    struct RawSlot {
        OVERLAPPED ov{};
        void*      buf = nullptr;
    };
    std::vector<RawSlot> slots(qd);
    for (auto& s : slots) s.buf = tierflow::tf_aligned_alloc(block_size, 4096);

    const uint64_t blocks_per_rep = bytes_per_rep / block_size;
    std::vector<double> gbps;

    for (int r = 0; r < reps + 2; ++r) { // +2 warmup passes
        uint64_t next_blk   = 0;
        uint64_t done_blks  = 0;
        size_t   inflight   = 0;
        uint64_t base_off   = (static_cast<uint64_t>(r) * bytes_per_rep) % (file_size - block_size);
        base_off            = (base_off / block_size) * block_size;

        auto issue = [&](size_t idx) {
            uint64_t off = (base_off + next_blk * block_size) % (file_size - block_size + 1);
            off = (off / block_size) * block_size;
            slots[idx].ov            = OVERLAPPED{};
            slots[idx].ov.Offset     = static_cast<DWORD>(off & 0xFFFFFFFFULL);
            slots[idx].ov.OffsetHigh = static_cast<DWORD>(off >> 32);
            DWORD br = 0;
            BOOL ok  = ReadFile(hf, slots[idx].buf, static_cast<DWORD>(block_size), &br, &slots[idx].ov);
            if (!ok && GetLastError() != ERROR_IO_PENDING) {
                std::fprintf(stderr, "Raw ReadFile failed err=%lu\n", GetLastError());
            }
            ++next_blk;
            ++inflight;
        };

        auto t0 = Clock::now();
        for (size_t i = 0; i < qd && next_blk < blocks_per_rep; ++i) issue(i);
        while (done_blks < blocks_per_rep) {
            DWORD br = 0;
            ULONG_PTR key = 0;
            OVERLAPPED* ov = nullptr;
            if (!GetQueuedCompletionStatus(iocp, &br, &key, &ov, INFINITE) || !ov) break;
            ++done_blks;
            --inflight;
            size_t idx = static_cast<size_t>(reinterpret_cast<RawSlot*>(ov) - slots.data());
            if (next_blk < blocks_per_rep) issue(idx);
        }
        double dt = std::chrono::duration<double>(Clock::now() - t0).count();
        if (r >= 2) {
            gbps.push_back((static_cast<double>(blocks_per_rep * block_size) / 1e9) / dt);
        }
    }

    for (auto& s : slots) tierflow::tf_aligned_free(s.buf);
    CloseHandle(iocp);
    CloseHandle(hf);
    return compute_stats(gbps);
}

struct BufferedControlResult {
    double gbps                    = 0.0;
    int64_t delta_system_cache     = 0;
    int64_t delta_standby_normal   = 0;
};

BufferedControlResult run_buffered_os_control(const std::string& path, uint64_t bytes_to_read,
                                              tierflow::tf_mem_sampler& mem) {
    BufferedControlResult res{};
    int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring wpath(static_cast<size_t>(wlen), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen);

    // Offset near middle of shard so we read fresh uncached blocks cleanly
    HANDLE hf = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    LARGE_INTEGER li{};
    li.QuadPart = 512ULL << 20;
    SetFilePointerEx(hf, li, nullptr, FILE_BEGIN);

    constexpr size_t kChunk = 4ULL << 20;
    std::vector<uint8_t> buf(kChunk);

    auto m0 = mem.sample();
    auto t0 = Clock::now();
    uint64_t got = 0;
    while (got < bytes_to_read) {
        DWORD nr = 0;
        if (!ReadFile(hf, buf.data(), static_cast<DWORD>(kChunk), &nr, nullptr) || nr == 0) break;
        got += nr;
    }
    double dt = std::chrono::duration<double>(Clock::now() - t0).count();
    auto m1   = mem.sample();
    CloseHandle(hf);

    res.gbps                 = (static_cast<double>(got) / 1e9) / dt;
    res.delta_system_cache   = static_cast<int64_t>(m1.system_cache_bytes) - static_cast<int64_t>(m0.system_cache_bytes);
    res.delta_standby_normal = static_cast<int64_t>(m1.standby_normal_bytes) - static_cast<int64_t>(m0.standby_normal_bytes);
    return res;
}

SummaryStats run_tf_io_workload(tierflow::tf_io_ex* io, const std::string& path,
                                uint64_t file_size, uint64_t block_size, size_t qd,
                                uint64_t bytes_per_rep, int reps, bool random_offsets,
                                int64_t* delta_sys_cache_out = nullptr,
                                tierflow::tf_mem_sampler* mem = nullptr) {
    io->open(path.c_str());
    std::vector<void*> bufs(qd);
    for (size_t i = 0; i < qd; ++i) bufs[i] = tierflow::tf_aligned_alloc(block_size, 4096);

    const uint64_t blocks_per_rep = bytes_per_rep / block_size;
    const uint64_t num_file_blks  = file_size / block_size;
    std::mt19937_64 rng(9001);
    std::vector<double> gbps;
    std::vector<tierflow::tf_read_req> done(qd);

    auto m0 = mem ? mem->sample() : tierflow::tf_mem_snapshot{};
    int64_t max_cache_delta = 0;

    for (int r = 0; r < reps + 2; ++r) {
        uint64_t next_i   = 0;
        uint64_t done_cnt = 0;
        uint64_t base_blk = (static_cast<uint64_t>(r) * blocks_per_rep) % num_file_blks;

        auto make_off = [&](uint64_t i) -> uint64_t {
            if (random_offsets) {
                return (rng() % num_file_blks) * block_size;
            }
            return ((base_blk + i) % num_file_blks) * block_size;
        };

        auto t0 = Clock::now();
        for (size_t s = 0; s < qd && next_i < blocks_per_rep; ++s) {
            uint64_t off = make_off(next_i);
            io->submit({static_cast<uint32_t>(next_i), off, block_size, bufs[s],
                        tierflow::TF_PRIO_DEMAND, 1000000ULL + s});
            ++next_i;
        }
        while (done_cnt < blocks_per_rep) {
            int n = io->poll(done.data(), static_cast<int>(qd));
            for (int k = 0; k < n; ++k) {
                ++done_cnt;
                if (next_i < blocks_per_rep) {
                    uint64_t off = make_off(next_i);
                    io->submit({static_cast<uint32_t>(next_i), off, block_size, done[k].dst,
                                tierflow::TF_PRIO_DEMAND, done[k].tag + qd});
                    ++next_i;
                }
            }
            if (n == 0) _mm_pause();
        }
        double dt = std::chrono::duration<double>(Clock::now() - t0).count();
        if (r >= 2) {
            gbps.push_back((static_cast<double>(blocks_per_rep * block_size) / 1e9) / dt);
        }
        if (mem) {
            auto m1 = mem->sample();
            int64_t dc = static_cast<int64_t>(m1.system_cache_bytes) - static_cast<int64_t>(m0.system_cache_bytes);
            if (dc > max_cache_delta) max_cache_delta = dc;
        }
    }

    if (delta_sys_cache_out) *delta_sys_cache_out = max_cache_delta;
    for (void* b : bufs) tierflow::tf_aligned_free(b);
    io->close();
    return compute_stats(gbps);
}

struct SustainedSecond {
    int     sec                    = 0;
    double  tf_gbps                = 0.0;
    double  pdh_disk_gbps          = 0.0;
    int64_t delta_sys_cache_bytes  = 0;
    int64_t delta_standby_bytes    = 0;
    int64_t delta_sys_used_bytes   = 0;
};

struct SustainedResult {
    double   duration_sec          = 0.0;
    uint64_t total_bytes_read      = 0;
    double   mean_gbps             = 0.0;
    double   pdh_mean_gbps         = 0.0;
    uint64_t pdh_est_total_bytes   = 0;
    int64_t  peak_delta_sys_cache  = 0;
    int64_t  end_delta_sys_cache   = 0;
    int64_t  peak_delta_standby    = 0;
    int64_t  peak_delta_sys_used   = 0;
    uint64_t peak_proc_private     = 0;
    size_t   mem_samples           = 0;
    uint64_t io_errors             = 0;
    std::vector<SustainedSecond> timeline;
};

SustainedResult run_sustained_60s(const std::string& path, uint64_t file_size,
                                  double target_seconds, uint64_t min_bytes,
                                  tierflow::tf_mem_sampler& mem) {
    SustainedResult res{};
    tierflow::tf_io_options opts{};
    opts.max_inflight          = 32;
    opts.prefetch_inflight_cap = 8;
    opts.strict_priority       = true;
    auto io = tierflow::tf_io_create_iocp(opts);
    io->open(path.c_str());

    constexpr uint64_t kBlock = 1ULL << 20; // 1 MiB expert-sized reads (matches Phase 2 ">=50 GB of experts")
    constexpr size_t   kQD    = 32;
    std::vector<void*> bufs(kQD);
    for (size_t i = 0; i < kQD; ++i) bufs[i] = tierflow::tf_aligned_alloc(kBlock, 4096);

    const auto idle_mem = mem.sample();
    tierflow::tf_disk_counter disk_ctr;
    std::vector<double> disk_samples;
    int64_t  peak_sys_cache = 0;
    int64_t  peak_standby   = 0;
    int64_t  peak_sys_used  = 0;
    uint64_t peak_priv      = idle_mem.process_private_bytes;
    size_t   sample_count   = 0;

    std::atomic<bool> stop_sampling{false};
    std::atomic<uint64_t> shared_bytes_done{0};
    auto t0 = Clock::now();

    std::thread sampler_thread([&]() {
        int next_sec_log      = 1;
        uint64_t prev_bytes   = 0;
        auto prev_t           = t0;
        double sec_disk_sum   = 0.0;
        int sec_disk_cnt      = 0;
        while (!stop_sampling.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            auto now = Clock::now();
            auto s   = mem.sample();
            int64_t dc = static_cast<int64_t>(s.system_cache_bytes) - static_cast<int64_t>(idle_mem.system_cache_bytes);
            int64_t ds = static_cast<int64_t>(s.standby_normal_bytes) - static_cast<int64_t>(idle_mem.standby_normal_bytes);
            int64_t du = static_cast<int64_t>(s.system_used_bytes) - static_cast<int64_t>(idle_mem.system_used_bytes);
            peak_sys_cache = std::max<int64_t>(peak_sys_cache, dc);
            peak_standby   = std::max<int64_t>(peak_standby, ds);
            peak_sys_used  = std::max<int64_t>(peak_sys_used, du);
            peak_priv      = std::max<uint64_t>(peak_priv, s.process_private_bytes);
            double dbps    = disk_ctr.sample_read_bytes_per_sec();
            if (dbps >= 0.0) {
                disk_samples.push_back(dbps);
                sec_disk_sum += dbps;
                ++sec_disk_cnt;
            }
            ++sample_count;

            double el = std::chrono::duration<double>(now - t0).count();
            if (el >= static_cast<double>(next_sec_log)) {
                uint64_t cur_b = shared_bytes_done.load(std::memory_order_relaxed);
                double dt      = std::chrono::duration<double>(now - prev_t).count();
                double tf_gbps = (dt > 0.0) ? (static_cast<double>(cur_b - prev_bytes) / 1e9) / dt : 0.0;
                double d_gbps  = (sec_disk_cnt > 0) ? (sec_disk_sum / static_cast<double>(sec_disk_cnt)) / 1e9 : 0.0;
                res.timeline.push_back({next_sec_log, tf_gbps, d_gbps, dc, ds, du});
                prev_bytes   = cur_b;
                prev_t       = now;
                sec_disk_sum = 0.0;
                sec_disk_cnt = 0;
                ++next_sec_log;
            }
        }
    });

    const uint64_t num_blks = file_size / kBlock;
    std::mt19937_64 rng(424242);
    uint64_t tag_seq = 1;

    for (size_t i = 0; i < kQD; ++i) {
        uint64_t off = (rng() % num_blks) * kBlock;
        io->submit({static_cast<uint32_t>(i), off, kBlock, bufs[i], tierflow::TF_PRIO_DEMAND, tag_seq++});
    }

    std::vector<tierflow::tf_read_req> done(kQD);
    uint64_t bytes_done = 0;
    while (true) {
        double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
        if (elapsed >= target_seconds && bytes_done >= min_bytes) break;

        int n = io->poll(done.data(), static_cast<int>(kQD));
        for (int k = 0; k < n; ++k) {
            bytes_done += done[k].size;
            shared_bytes_done.store(bytes_done, std::memory_order_relaxed);
            uint64_t off = (rng() % num_blks) * kBlock;
            io->submit({done[k].object_id, off, kBlock, done[k].dst, tierflow::TF_PRIO_DEMAND, tag_seq++});
        }
        if (n == 0) _mm_pause();
    }

    io->drain();
    double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
    stop_sampling.store(true);
    sampler_thread.join();

    auto end_mem = mem.sample();
    auto st      = io->stats();

    res.duration_sec         = elapsed;
    res.total_bytes_read     = st.bytes_completed;
    res.mean_gbps            = (static_cast<double>(res.total_bytes_read) / 1e9) / elapsed;
    if (!disk_samples.empty()) {
        double mean_bps          = std::accumulate(disk_samples.begin(), disk_samples.end(), 0.0) / static_cast<double>(disk_samples.size());
        res.pdh_mean_gbps        = mean_bps / 1e9;
        res.pdh_est_total_bytes  = static_cast<uint64_t>(mean_bps * elapsed);
    }
    res.peak_delta_sys_cache = peak_sys_cache;
    res.end_delta_sys_cache  = static_cast<int64_t>(end_mem.system_cache_bytes) - static_cast<int64_t>(idle_mem.system_cache_bytes);
    res.peak_delta_standby   = peak_standby;
    res.peak_delta_sys_used  = peak_sys_used;
    res.peak_proc_private    = peak_priv;
    res.mem_samples          = sample_count;
    res.io_errors            = st.io_errors;

    for (void* b : bufs) tierflow::tf_aligned_free(b);
    io->close();
    return res;
}

SummaryStats run_priority_storm_trial(const std::string& path, uint64_t file_size,
                                      bool strict_priority, int num_demand_probes) {
    tierflow::tf_io_options opts{};
    opts.max_inflight          = 32;
    opts.prefetch_inflight_cap = strict_priority ? 4 : 32;
    opts.strict_priority       = strict_priority;

    auto io = tierflow::tf_io_create_iocp(opts);
    io->open(path.c_str());

    constexpr uint64_t kBlock        = 1ULL << 20; // 1 MiB
    constexpr size_t   kPrefetchPool = 64;
    std::vector<void*> pbufs(kPrefetchPool);
    std::vector<void*> free_pbufs;
    for (size_t i = 0; i < kPrefetchPool; ++i) {
        pbufs[i] = tierflow::tf_aligned_alloc(kBlock, 4096);
        free_pbufs.push_back(pbufs[i]);
    }
    void* dbuf = tierflow::tf_aligned_alloc(kBlock, 4096);

    const uint64_t num_blks = file_size / kBlock;
    std::mt19937_64 rng(777 + (strict_priority ? 0 : 1));
    uint64_t next_tag = 1;

    auto topup_storm = [&]() {
        while (!free_pbufs.empty()) {
            void* b = free_pbufs.back();
            free_pbufs.pop_back();
            uint64_t off = (rng() % num_blks) * kBlock;
            io->submit({0, off, kBlock, b, tierflow::TF_PRIO_PREFETCH, next_tag++});
        }
    };

    topup_storm();
    // Let the storm reach steady-state queue saturation before measuring demand probes
    std::this_thread::sleep_for(std::chrono::milliseconds(15));

    std::vector<double> demand_lat_ms;
    std::vector<tierflow::tf_read_req> done(64);

    for (int p = 0; p < num_demand_probes; ++p) {
        int drain_n = io->poll(done.data(), static_cast<int>(done.size()));
        for (int k = 0; k < drain_n; ++k) {
            if (done[k].priority == tierflow::TF_PRIO_PREFETCH) free_pbufs.push_back(done[k].dst);
        }
        topup_storm();

        const uint64_t dtag = 9000000ULL + static_cast<uint64_t>(p);
        const uint64_t doff = (rng() % num_blks) * kBlock;
        auto td0 = Clock::now();
        io->submit({1, doff, kBlock, dbuf, tierflow::TF_PRIO_DEMAND, dtag});

        bool got_demand = false;
        while (!got_demand) {
            int n = io->poll(done.data(), static_cast<int>(done.size()));
            for (int k = 0; k < n; ++k) {
                if (done[k].tag == dtag) {
                    double ms = std::chrono::duration<double, std::milli>(Clock::now() - td0).count();
                    demand_lat_ms.push_back(ms);
                    got_demand = true;
                } else {
                    free_pbufs.push_back(done[k].dst);
                    topup_storm();
                }
            }
            if (n == 0) _mm_pause();
        }
    }

    io->drain();
    for (void* b : pbufs) tierflow::tf_aligned_free(b);
    tierflow::tf_aligned_free(dbuf);
    io->close();
    return compute_stats(demand_lat_ms);
}

std::string json_stats(const SummaryStats& s) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(4)
      << "{\"n\": " << s.n
      << ", \"mean\": " << s.mean
      << ", \"sd\": " << s.sd
      << ", \"ci95\": " << s.ci95
      << ", \"min\": " << s.min
      << ", \"p50\": " << s.p50
      << ", \"p95\": " << s.p95
      << ", \"max\": " << s.max << "}";
    return o.str();
}

#endif

} // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    std::string exp_id      = "p2_io_win";
    std::string shard_path  = "E:\\RAM_optimizer\\.tmp_p2_io_shard.bin";
    std::string out_json    = "E:\\RAM_optimizer\\results\\p2_io_win\\io_bench_results.json";
    uint64_t    shard_bytes = 2ULL << 30; // 2 GiB real non-zero test file
    double      sustained_s = 60.0;
    bool        keep_file   = false;
    bool        heat_soak_only = false;
    int         reps        = 20;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--file" && i + 1 < argc) shard_path = argv[++i];
        else if (a == "--exp-id" && i + 1 < argc) exp_id = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_json = argv[++i];
        else if (a == "--seconds" && i + 1 < argc) sustained_s = std::atof(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (a == "--keep-file") keep_file = true;
        else if (a == "--heat-soak-only") heat_soak_only = true;
    }

    std::filesystem::create_directories(std::filesystem::path(out_json).parent_path());

    if (heat_soak_only) {
        if (!std::filesystem::exists(shard_path)) {
            std::fprintf(stderr, "Heat-soak target missing: %s\n", shard_path.c_str());
            return 2;
        }
        const uint64_t fsz = std::filesystem::file_size(shard_path);
        tierflow::tf_mem_sampler mem;
        std::printf("[SOAK] Running %.0fs read-only unbuffered NVMe heat-soak on %s (%.2f GiB)...\n",
                    sustained_s, shard_path.c_str(), static_cast<double>(fsz) / (1024.0 * 1024.0 * 1024.0));
        auto sust = run_sustained_60s(shard_path, fsz, sustained_s, 0ULL, mem);
        std::printf("[SOAK] Done %.2fs: read=%.2f GB (%.3f GB/s), peak_delta_sys_cache=%.2f MiB, io_errors=%llu\n",
                    sust.duration_sec,
                    static_cast<double>(sust.total_bytes_read) / 1e9,
                    sust.mean_gbps,
                    static_cast<double>(sust.peak_delta_sys_cache) / (1024.0 * 1024.0),
                    static_cast<unsigned long long>(sust.io_errors));
        return (sust.io_errors == 0) ? 0 : 1;
    }

    if (!std::filesystem::exists(shard_path) || std::filesystem::file_size(shard_path) != shard_bytes) {
        if (!create_unbuffered_shard(shard_path, shard_bytes)) return 1;
        std::this_thread::sleep_for(std::chrono::seconds(3));
    }

    tierflow::tf_mem_sampler mem;
    auto idle = mem.sample();
    std::printf("[INFO] Idle physical_available=%.2f GiB, system_cache=%.2f MiB, standby_normal=%.2f MiB\n",
                static_cast<double>(idle.physical_available_bytes) / (1024.0 * 1024.0 * 1024.0),
                static_cast<double>(idle.system_cache_bytes) / (1024.0 * 1024.0),
                static_cast<double>(idle.standby_normal_bytes) / (1024.0 * 1024.0));

    // 1. Raw OS unbuffered baseline
    auto raw_seq = run_raw_iocp_seq_baseline(shard_path, shard_bytes, 4ULL << 20, 32, 512ULL << 20, reps);
    std::printf("[BENCH] Raw Win32 Overlapped IOCP 4MiB seq QD=32: %.3f +/- %.3f GB/s (min=%.3f max=%.3f, n=%zu)\n",
                raw_seq.mean, raw_seq.ci95, raw_seq.min, raw_seq.max, raw_seq.n);

    // 2. tf_io IOCP 4 MiB seq QD=32 + 1 MiB random QD=1,8,32
    int64_t iocp_seq_cache_delta = 0;
    auto iocp = tierflow::tf_io_create_iocp({32, 8, true, 4096});
    auto tf_iocp_seq = run_tf_io_workload(iocp.get(), shard_path, shard_bytes, 4ULL << 20, 32, 512ULL << 20, reps, false, &iocp_seq_cache_delta, &mem);
    double iocp_vs_raw_pct = 100.0 * tf_iocp_seq.mean / raw_seq.mean;
    std::printf("[BENCH] tf_io_win (IOCP) 4MiB seq QD=32:          %.3f +/- %.3f GB/s (%.1f%% of raw drive, delta_cache=%lld B)\n",
                tf_iocp_seq.mean, tf_iocp_seq.ci95, iocp_vs_raw_pct, static_cast<long long>(iocp_seq_cache_delta));

    auto tf_iocp_rnd_qd1  = run_tf_io_workload(iocp.get(), shard_path, shard_bytes, 1ULL << 20, 1,  64ULL << 20,  reps, true);
    auto tf_iocp_rnd_qd8  = run_tf_io_workload(iocp.get(), shard_path, shard_bytes, 1ULL << 20, 8,  128ULL << 20, reps, true);
    auto tf_iocp_rnd_qd32 = run_tf_io_workload(iocp.get(), shard_path, shard_bytes, 1ULL << 20, 32, 256ULL << 20, reps, true);
    std::printf("[BENCH] tf_io_win (IOCP) 1MiB rand: QD1=%.3f+/-%.3f, QD8=%.3f+/-%.3f, QD32=%.3f+/-%.3f GB/s\n",
                tf_iocp_rnd_qd1.mean, tf_iocp_rnd_qd1.ci95,
                tf_iocp_rnd_qd8.mean, tf_iocp_rnd_qd8.ci95,
                tf_iocp_rnd_qd32.mean, tf_iocp_rnd_qd32.ci95);

    // 3. IoRing backend if supported
    std::string ioring_reason;
    bool has_ioring = tierflow::tf_ioring_is_supported(&ioring_reason);
    SummaryStats tf_ioring_seq{}, tf_ioring_rnd32{};
    if (has_ioring) {
        auto ioring = tierflow::tf_io_create_ioring({32, 8, true, 4096});
        tf_ioring_seq   = run_tf_io_workload(ioring.get(), shard_path, shard_bytes, 4ULL << 20, 32, 512ULL << 20, reps, false);
        tf_ioring_rnd32 = run_tf_io_workload(ioring.get(), shard_path, shard_bytes, 1ULL << 20, 32, 256ULL << 20, reps, true);
        std::printf("[BENCH] tf_io_ioring_win 4MiB seq QD=32: %.3f +/- %.3f GB/s (%.1f%% of raw); 1MiB rand QD32: %.3f +/- %.3f GB/s\n",
                    tf_ioring_seq.mean, tf_ioring_seq.ci95, 100.0 * tf_ioring_seq.mean / raw_seq.mean,
                    tf_ioring_rnd32.mean, tf_ioring_rnd32.ci95);
    }

    // 4. Two-priority queue vs FIFO null control under prefetch storm
    auto prio_strict = run_priority_storm_trial(shard_path, shard_bytes, true, reps);
    auto prio_fifo   = run_priority_storm_trial(shard_path, shard_bytes, false, reps);
    std::printf("[BENCH] Demand latency under 48-req prefetch storm (n=%d): StrictPrio=%.2f+/-%.2f ms (p95=%.2f) vs FIFO_Null=%.2f+/-%.2f ms (p95=%.2f)\n",
                reps, prio_strict.mean, prio_strict.ci95, prio_strict.p95,
                prio_fifo.mean, prio_fifo.ci95, prio_fifo.p95);

    // 5. 60-second sustained >=50 GB unbuffered expert-block read + sysmem/PDH check
    std::printf("[BENCH] Running %.0fs sustained unbuffered read (>= 50 GB target)...\n", sustained_s);
    auto sust = run_sustained_60s(shard_path, shard_bytes, sustained_s, 50ULL * 1000 * 1000 * 1000ULL, mem);
    std::printf("[BENCH] Sustained %.2fs: read=%.2f GB (%.3f GB/s), PDH_PhysicalDisk=%.3f GB/s (~%.2f GB), peak_delta_sys_cache=%.2f MiB, end_delta_sys_cache=%.2f MiB, peak_delta_standby=%.2f MiB, io_errors=%llu\n",
                sust.duration_sec,
                static_cast<double>(sust.total_bytes_read) / 1e9,
                sust.mean_gbps,
                sust.pdh_mean_gbps,
                static_cast<double>(sust.pdh_est_total_bytes) / 1e9,
                static_cast<double>(sust.peak_delta_sys_cache) / (1024.0 * 1024.0),
                static_cast<double>(sust.end_delta_sys_cache) / (1024.0 * 1024.0),
                static_cast<double>(sust.peak_delta_standby) / (1024.0 * 1024.0),
                static_cast<unsigned long long>(sust.io_errors));

    // 6. Buffered OS read control (512 MiB) AFTER unbuffered sustained run
    auto buf_ctrl = run_buffered_os_control(shard_path, 512ULL << 20, mem);
    std::printf("[CTRL]  Buffered OS read 4MiB (512 MiB): %.3f GB/s, delta_system_cache=%.2f MiB, delta_standby=%.2f MiB\n",
                buf_ctrl.gbps,
                static_cast<double>(buf_ctrl.delta_system_cache) / (1024.0 * 1024.0),
                static_cast<double>(buf_ctrl.delta_standby_normal) / (1024.0 * 1024.0));

    // Write full JSON results artifact
    {
        std::ofstream f(out_json);
        f << "{\n";
        f << "  \"exp_id\": \"" << exp_id << "\",\n";
        f << "  \"shard_bytes\": " << shard_bytes << ",\n";
        f << "  \"reps\": " << reps << ",\n";
        f << "  \"ioring_supported\": " << (has_ioring ? "true" : "false") << ",\n";
        f << "  \"ioring_info\": \"" << ioring_reason << "\",\n";
        f << "  \"raw_win32_seq_4m_qd32_gbps\": " << json_stats(raw_seq) << ",\n";
        f << "  \"tf_iocp_seq_4m_qd32_gbps\": " << json_stats(tf_iocp_seq) << ",\n";
        f << "  \"tf_iocp_seq_pct_of_raw\": " << std::fixed << std::setprecision(2) << iocp_vs_raw_pct << ",\n";
        f << "  \"tf_iocp_rand_1m_qd1_gbps\": " << json_stats(tf_iocp_rnd_qd1) << ",\n";
        f << "  \"tf_iocp_rand_1m_qd8_gbps\": " << json_stats(tf_iocp_rnd_qd8) << ",\n";
        f << "  \"tf_iocp_rand_1m_qd32_gbps\": " << json_stats(tf_iocp_rnd_qd32) << ",\n";
        if (has_ioring) {
            f << "  \"tf_ioring_seq_4m_qd32_gbps\": " << json_stats(tf_ioring_seq) << ",\n";
            f << "  \"tf_ioring_seq_pct_of_raw\": " << std::fixed << std::setprecision(2) << (100.0 * tf_ioring_seq.mean / raw_seq.mean) << ",\n";
            f << "  \"tf_ioring_rand_1m_qd32_gbps\": " << json_stats(tf_ioring_rnd32) << ",\n";
        }
        f << "  \"buffered_os_control\": {\n"
          << "    \"read_bytes\": " << (512ULL << 20) << ",\n"
          << "    \"gbps\": " << std::setprecision(4) << buf_ctrl.gbps << ",\n"
          << "    \"delta_system_cache_bytes\": " << buf_ctrl.delta_system_cache << ",\n"
          << "    \"delta_standby_normal_bytes\": " << buf_ctrl.delta_standby_normal << "\n"
          << "  },\n";
        f << "  \"demand_latency_under_storm_ms\": {\n"
          << "    \"strict_priority\": " << json_stats(prio_strict) << ",\n"
          << "    \"fifo_null_control\": " << json_stats(prio_fifo) << "\n"
          << "  },\n";
        f << "  \"sustained_read_60s\": {\n"
          << "    \"duration_sec\": " << std::setprecision(3) << sust.duration_sec << ",\n"
          << "    \"total_bytes_read\": " << sust.total_bytes_read << ",\n"
          << "    \"mean_gbps\": " << std::setprecision(4) << sust.mean_gbps << ",\n"
          << "    \"pdh_physical_disk_mean_gbps\": " << sust.pdh_mean_gbps << ",\n"
          << "    \"pdh_est_total_bytes\": " << sust.pdh_est_total_bytes << ",\n"
          << "    \"peak_delta_system_cache_bytes\": " << sust.peak_delta_sys_cache << ",\n"
          << "    \"end_delta_system_cache_bytes\": " << sust.end_delta_sys_cache << ",\n"
          << "    \"peak_delta_standby_normal_bytes\": " << sust.peak_delta_standby << ",\n"
          << "    \"peak_delta_system_used_bytes\": " << sust.peak_delta_sys_used << ",\n"
          << "    \"peak_process_private_bytes\": " << sust.peak_proc_private << ",\n"
          << "    \"mem_samples\": " << sust.mem_samples << ",\n"
          << "    \"io_errors\": " << sust.io_errors << ",\n"
          << "    \"timeline\": [\n";
        for (size_t i = 0; i < sust.timeline.size(); ++i) {
            const auto& row = sust.timeline[i];
            f << "      {\"sec\": " << row.sec
              << ", \"tf_gbps\": " << std::setprecision(3) << row.tf_gbps
              << ", \"pdh_disk_gbps\": " << row.pdh_disk_gbps
              << ", \"delta_sys_cache_mib\": " << std::setprecision(2) << (static_cast<double>(row.delta_sys_cache_bytes) / (1024.0 * 1024.0))
              << ", \"delta_standby_mib\": " << (static_cast<double>(row.delta_standby_bytes) / (1024.0 * 1024.0))
              << ", \"delta_sys_used_mib\": " << (static_cast<double>(row.delta_sys_used_bytes) / (1024.0 * 1024.0))
              << "}" << (i + 1 < sust.timeline.size() ? "," : "") << "\n";
        }
        f << "    ]\n"
          << "  }\n";
        f << "}\n";
    }

    if (!keep_file) {
        std::error_code ec;
        std::filesystem::remove(shard_path, ec);
        std::printf("[CLEANUP] Removed temporary shard %s\n", shard_path.c_str());
    }
    std::printf("[DONE] Wrote %s\n", out_json.c_str());
    return 0;
#else
    std::string shard_path  = "/var/tmp/.tmp_tf_io_linux.bin";
    uint64_t    shard_bytes = 512ULL << 20;
    int         reps        = 10;
    bool        keep_file   = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--file" && i + 1 < argc) shard_path = argv[++i];
        else if (a == "--size-mb" && i + 1 < argc) shard_bytes = static_cast<uint64_t>(std::atoll(argv[++i])) << 20;
        else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (a == "--keep-file") keep_file = true;
    }

    auto read_meminfo_cache_bytes = []() -> int64_t {
        std::ifstream f("/proc/meminfo");
        std::string line;
        int64_t total_kb = 0;
        while (std::getline(f, line)) {
            long long val = 0;
            if (std::sscanf(line.c_str(), "Cached: %lld kB", &val) == 1 ||
                std::sscanf(line.c_str(), "Buffers: %lld kB", &val) == 1) {
                total_kb += static_cast<int64_t>(val);
            }
        }
        return total_kb * 1024LL;
    };

    bool created_temp = false;
    if (!std::filesystem::exists(shard_path)) {
        created_temp = true;
        int fd = ::open(shard_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0600);
        if (fd < 0 && (errno == EINVAL || errno == EOPNOTSUPP)) {
            fd = ::open(shard_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        }
        if (fd < 0) {
            std::fprintf(stderr, "Failed to create Linux benchmark file %s (errno=%d)\n", shard_path.c_str(), errno);
            return 1;
        }
        constexpr uint64_t kChunk = 4ULL << 20;
        void* buf = tierflow::tf_aligned_alloc(kChunk, 4096);
        auto* p = static_cast<uint8_t*>(buf);
        for (uint64_t i = 0; i < kChunk; i += 4096) p[i] = static_cast<uint8_t>((i >> 12) ^ 0x5Au);
        for (uint64_t off = 0; off < shard_bytes; off += kChunk) {
            const size_t n = static_cast<size_t>(std::min<uint64_t>(kChunk, shard_bytes - off));
            if (::write(fd, buf, n) != static_cast<ssize_t>(n)) {
                tierflow::tf_aligned_free(buf);
                ::close(fd);
                std::filesystem::remove(shard_path);
                return 1;
            }
        }
        ::fsync(fd);
        ::close(fd);
        tierflow::tf_aligned_free(buf);
    }

    const uint64_t fsz = std::filesystem::file_size(shard_path);
    std::string reason;
    const bool has_uring = tierflow::tf_ioring_is_supported(&reason);
    std::printf("[INFO] Linux io_uring supported=%s (%s), file=%s (%.2f MiB)\n",
                has_uring ? "true" : "false", reason.c_str(), shard_path.c_str(),
                static_cast<double>(fsz) / (1024.0 * 1024.0));

    auto bench_work = [&](size_t chunk_bytes, size_t qd, uint64_t total_bytes, bool random_off, int64_t* out_cache_delta) {
        std::vector<double> gbps_runs;
        auto io = tierflow::tf_io_create_ioring({32, 8, true, 4096});
        if (!io || !io->open(shard_path.c_str())) return compute_stats(gbps_runs);
        std::vector<void*> bufs(qd, nullptr);
        for (size_t i = 0; i < qd; ++i) bufs[i] = tierflow::tf_aligned_alloc(chunk_bytes, 4096);
        const uint64_t blocks = std::max<uint64_t>(1, (fsz - chunk_bytes) / chunk_bytes);
        const uint64_t num_reads = std::max<uint64_t>(1, total_bytes / chunk_bytes);
        std::mt19937_64 rng(42);
        const int64_t cache0 = read_meminfo_cache_bytes();
        for (int rep = 0; rep < reps; ++rep) {
            uint64_t submitted = 0, completed = 0, tag = 1;
            auto submit_one = [&](size_t slot) {
                uint64_t blk = random_off ? (rng() % blocks) : (submitted % blocks);
                io->submit({0, blk * chunk_bytes, chunk_bytes, bufs[slot], tierflow::TF_PRIO_DEMAND, tag++});
                ++submitted;
            };
            auto t0 = Clock::now();
            for (size_t s = 0; s < qd && submitted < num_reads; ++s) submit_one(s);
            tierflow::tf_read_req done[16];
            while (completed < submitted) {
                int n = io->wait(done, 16, 100);
                for (int i = 0; i < n; ++i) {
                    ++completed;
                    if (submitted < num_reads) submit_one(static_cast<size_t>((done[i].tag - 1) % qd));
                }
            }
            double dt = std::chrono::duration<double>(Clock::now() - t0).count();
            if (dt > 0.0) gbps_runs.push_back((static_cast<double>(num_reads * chunk_bytes) / 1e9) / dt);
        }
        if (out_cache_delta) *out_cache_delta = read_meminfo_cache_bytes() - cache0;
        for (void* b : bufs) tierflow::tf_aligned_free(b);
        io->close();
        return compute_stats(gbps_runs);
    };

    int64_t cache_delta_seq = 0, cache_delta_rnd = 0;
    auto seq_st = bench_work(4ULL << 20, 32, 256ULL << 20, false, &cache_delta_seq);
    auto rnd_st = bench_work(1ULL << 20, 32, 128ULL << 20, true, &cache_delta_rnd);
    std::printf("[BENCH] Linux seq 4MiB QD=32: %.3f +/- %.3f GB/s (min=%.3f max=%.3f, pagecache_delta=%lld B)\n",
                seq_st.mean, seq_st.ci95, seq_st.min, seq_st.max, static_cast<long long>(cache_delta_seq));
    std::printf("[BENCH] Linux rand 1MiB QD=32: %.3f +/- %.3f GB/s (min=%.3f max=%.3f, pagecache_delta=%lld B)\n",
                rnd_st.mean, rnd_st.ci95, rnd_st.min, rnd_st.max, static_cast<long long>(cache_delta_rnd));

    if (created_temp && !keep_file) {
        std::error_code ec;
        std::filesystem::remove(shard_path, ec);
    }
    return 0;
#endif
}
