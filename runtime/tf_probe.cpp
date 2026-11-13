#include "tf_probe.h"
#include "tf_io.h"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

namespace tierflow {

HardwareProfile run_tf_probe(const std::string& optional_nvme_probe_file,
                             int n_threads_hint,
                             int ssd_qd_hint) {
    HardwareProfile prof{};

    // Multi-threaded barrier-synchronized DDR5 read & GEMV probe (excludes thread-spawn latency)
    constexpr size_t kBytes = 128ULL << 20;
    const size_t n_floats   = kBytes / sizeof(float);
    std::vector<float> src(n_floats, 1.0001f);
    std::vector<float> dst(n_floats, 0.0f);
    std::memcpy(dst.data(), src.data(), kBytes);

    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    unsigned int n_threads = (n_threads_hint > 0)
        ? static_cast<unsigned int>(n_threads_hint)
        : std::min<unsigned int>(hw, 8u);
    n_threads = std::clamp<unsigned int>(n_threads, 1u, 16u);

    double best_copy = 0.0;
    for (int it = 0; it < 4; ++it) {
        std::barrier sync_pt(static_cast<std::ptrdiff_t>(n_threads + 1));
        std::vector<std::thread> workers;
        workers.reserve(n_threads);
        const size_t chunk = (kBytes / n_threads) & ~4095ULL;
        for (unsigned int t = 0; t < n_threads; ++t) {
            const size_t off = static_cast<size_t>(t) * chunk;
            const size_t len = (t + 1 == n_threads) ? (kBytes - off) : chunk;
            workers.emplace_back([&, off, len]() {
                sync_pt.arrive_and_wait();
                std::memcpy(reinterpret_cast<uint8_t*>(dst.data()) + off,
                            reinterpret_cast<const uint8_t*>(src.data()) + off, len);
                sync_pt.arrive_and_wait();
            });
        }
        sync_pt.arrive_and_wait();
        auto t0 = std::chrono::steady_clock::now();
        sync_pt.arrive_and_wait();
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        for (auto& w : workers) w.join();
        if (dt > 0.0) {
            double gbps = (static_cast<double>(kBytes) / 1e9) / dt;
            if (gbps > best_copy) best_copy = gbps;
        }
    }

    std::atomic<double> total_sink{0.0};
    double best_read = 0.0;
    for (int it = 0; it < 5; ++it) {
        std::barrier sync_pt(static_cast<std::ptrdiff_t>(n_threads + 1));
        std::vector<std::thread> workers;
        workers.reserve(n_threads);
        const size_t chunk = n_floats / n_threads;
        for (unsigned int t = 0; t < n_threads; ++t) {
            const size_t beg = static_cast<size_t>(t) * chunk;
            const size_t end = (t + 1 == n_threads) ? n_floats : (beg + chunk);
            workers.emplace_back([&, beg, end]() {
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
                sync_pt.arrive_and_wait();
                for (size_t i = beg; i + 7 < end; i += 8) {
                    s0 += src[i + 0];
                    s1 += src[i + 1];
                    s2 += src[i + 2];
                    s3 += src[i + 3];
                    s4 += src[i + 4];
                    s5 += src[i + 5];
                    s6 += src[i + 6];
                    s7 += src[i + 7];
                }
                sync_pt.arrive_and_wait();
                double prev = total_sink.load(std::memory_order_relaxed);
                total_sink.store(prev + static_cast<double>(s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7),
                                 std::memory_order_relaxed);
            });
        }
        sync_pt.arrive_and_wait();
        auto t0 = std::chrono::steady_clock::now();
        sync_pt.arrive_and_wait();
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        for (auto& w : workers) w.join();
        if (dt > 0.0) {
            double gbps = (static_cast<double>(kBytes) / 1e9) / dt;
            if (gbps > best_read) best_read = gbps;
        }
    }

    if (best_copy > 1.0) prof.bw_ram_copy_gbps = best_copy;
    if (best_read > 1.0) {
        prof.bw_ram_read_gbps        = best_read;
        prof.cpu_gemv_effective_gbps = best_read * 0.90;
    }

    if (!optional_nvme_probe_file.empty()) {
        const size_t qd = static_cast<size_t>(std::clamp(ssd_qd_hint, 1, 8));
        auto io = tf_io_create_iocp({16, qd, true, 4096});
        if (io && io->open(optional_nvme_probe_file.c_str()) && io->file_size() >= (64ULL << 20)) {
            constexpr uint64_t kBlk = 8ULL << 20; // tensor-scale 8 MiB aligned probe reads
            std::vector<void*> bufs(qd, nullptr);
            bool alloc_ok = true;
            for (size_t i = 0; i < qd; ++i) {
                bufs[i] = tf_aligned_alloc(kBlk, 4096);
                if (!bufs[i]) alloc_ok = false;
            }
            if (alloc_ok) {
                constexpr size_t kTotalReads = 16; // 128 MiB across .tfpack strides
                const uint64_t fsz = io->file_size();
                uint64_t submitted = 0;
                uint64_t completed = 0;
                uint64_t tag = 1;
                auto submit_next = [&](size_t slot) {
                    const uint64_t stride = std::max<uint64_t>(kBlk, (fsz - kBlk) / kTotalReads) & ~4095ULL;
                    const uint64_t off = std::min<uint64_t>(submitted * stride, (fsz - kBlk) & ~4095ULL);
                    if (io->submit({0, off, kBlk, bufs[slot], TF_PRIO_DEMAND, tag})) {
                        ++tag;
                        ++submitted;
                    }
                };
                auto t0 = std::chrono::steady_clock::now();
                for (size_t s = 0; s < qd && submitted < kTotalReads; ++s) {
                    submit_next(s);
                }
                tf_read_req done[8];
                while (completed < submitted) {
                    int n = io->poll(done, 8);
                    for (int i = 0; i < n; ++i) {
                        ++completed;
                        if (submitted < kTotalReads) {
                            const size_t slot = static_cast<size_t>((done[i].tag - 1) % qd);
                            submit_next(slot);
                        }
                    }
                    if (n == 0) std::this_thread::yield();
                }
                double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (dt > 0.0 && completed > 0) {
                    const double burst_lowqd = (static_cast<double>(completed * kBlk) / 1e9) / dt;
                    prof.bw_ssd_burst_gbps     = burst_lowqd;
                    // Post-pSLC NVMe sustained-to-burst derate factor (~0.35), never exceeding live burst
                    prof.bw_ssd_sustained_gbps = std::min(burst_lowqd, std::max(0.25, burst_lowqd * 0.35));
                }
            }
            for (void* p : bufs) {
                if (p) tf_aligned_free(p);
            }
            io->close();
        }
    }

    prof.probed_live = true;
    return prof;
}

} // namespace tierflow
