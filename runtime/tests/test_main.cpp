#include "tf_cache.h"
#include "tf_compute_place.h"
#include "tf_governor.h"
#include "tf_io.h"
#include "tf_kv.h"
#include "tf_manifest.h"
#include "tf_mem_win.h"
#include "tf_planner.h"
#include "tf_pool_plan.h"
#include "tf_prefetch.h"
#include "tf_probe.h"
#include "tf_spec.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

int g_failures = 0;

#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "[FAIL] %s:%d: %s (%s)\n", __FILE__, __LINE__, \
                         msg, #cond);                                         \
            ++g_failures;                                                     \
        }                                                                     \
    } while (0)

std::string make_temp_test_file(uint64_t size_bytes, uint64_t seed) {
    namespace fs = std::filesystem;
    fs::path p = fs::temp_directory_path() / ("tf_unit_io_" + std::to_string(seed) + ".bin");
    std::string sp = p.string();

    std::vector<uint8_t> buf(static_cast<size_t>(size_bytes));
    std::mt19937_64 rng(seed);
    for (size_t i = 0; i < buf.size(); i += 8) {
        uint64_t v = rng();
        std::memcpy(buf.data() + i, &v, std::min<size_t>(8, buf.size() - i));
    }

    std::FILE* f = std::fopen(sp.c_str(), "wb");
    if (!f) return "";
    std::fwrite(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    return sp;
}

void test_governor() {
    std::printf("[TEST] tf_governor...\n");
    constexpr uint64_t kRamBudget  = 1024ULL * 1024 * 100; // 100 MiB
    constexpr uint64_t kVramBudget = 1024ULL * 1024 * 50;  // 50 MiB

    tierflow::tf_governor gov(kRamBudget, kVramBudget);
    CHECK(gov.budget(tierflow::MemoryTier::RAM) == kRamBudget, "RAM budget mismatch");
    CHECK(gov.budget(tierflow::MemoryTier::VRAM) == kVramBudget, "VRAM budget mismatch");

    // Empty tag rejected
    CHECK(!gov.reserve(tierflow::MemoryTier::RAM, 4096, ""), "Empty tag must fail");

    // Exact reserves & tag sums
    CHECK(gov.reserve(tierflow::MemoryTier::RAM, 40ULL << 20, "experts_ram"), "Reserve 40 MiB");
    CHECK(gov.reserve(tierflow::MemoryTier::RAM, 25ULL << 20, "kv_ram"), "Reserve 25 MiB");
    CHECK(gov.reserve(tierflow::MemoryTier::RAM, 10ULL << 20, "experts_ram"), "Add 10 MiB to tag");
    CHECK(gov.reserve(tierflow::MemoryTier::VRAM, 30ULL << 20, "always_on"), "Reserve 30 MiB VRAM");

    CHECK(gov.used(tierflow::MemoryTier::RAM) == (75ULL << 20), "Used RAM exact");
    CHECK(gov.sum_tags(tierflow::MemoryTier::RAM) == gov.used(tierflow::MemoryTier::RAM), "RAM tag sum exact");
    CHECK(gov.sum_tags(tierflow::MemoryTier::VRAM) == gov.used(tierflow::MemoryTier::VRAM), "VRAM tag sum exact");

    // Over-budget reservation rejected atomically
    CHECK(!gov.reserve(tierflow::MemoryTier::RAM, 26ULL << 20, "overflow"), "Over-budget RAM must fail");
    CHECK(gov.used(tierflow::MemoryTier::RAM) == (75ULL << 20), "Failed reserve must not mutate used");
    CHECK(!gov.reserve(tierflow::MemoryTier::VRAM, 21ULL << 20, "overflow_v"), "Over-budget VRAM must fail");

    // Start-up plan check
    std::string reason;
    std::vector<tierflow::PlanReservation> bad_plan{
        {tierflow::MemoryTier::RAM, 30ULL << 20, "more_experts"},
    };
    CHECK(!gov.check_plan(bad_plan, &reason), "Over-budget plan rejected");
    CHECK(!reason.empty(), "Rejection reason populated");

    std::vector<tierflow::PlanReservation> good_plan{
        {tierflow::MemoryTier::RAM, 25ULL << 20, "more_experts"},
        {tierflow::MemoryTier::VRAM, 20ULL << 20, "hot_experts"},
    };
    CHECK(gov.check_plan(good_plan, &reason), "Exact-fit plan accepted");

    // Over-release fails
    CHECK(!gov.release(tierflow::MemoryTier::RAM, 99ULL << 20, "kv_ram"), "Over-release must fail");
    CHECK(gov.release(tierflow::MemoryTier::RAM, 25ULL << 20, "kv_ram"), "Exact release succeeds");
    CHECK(gov.used(tierflow::MemoryTier::RAM) == (50ULL << 20), "Used after release");
    CHECK(gov.sum_tags(tierflow::MemoryTier::RAM) == (50ULL << 20), "Tag sum after release");
}

void test_cache() {
    std::printf("[TEST] tf_cache...\n");

    // 1. Textbook Belady vs LRU check
    // Trace: [1,2,3,4,1,2,5,1,2,3,4,5], capacity = 3
    // Textbook: LRU = 10 misses (2 hits), Belady = 7 misses (5 hits)
    const std::vector<uint32_t> textbook = {1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5};
    CHECK(tierflow::belady_oracle_misses(textbook, 3) == 7, "Belady textbook miss count == 7");

    tierflow::CacheConfig lru_cfg{};
    lru_cfg.capacity_slots = 3;
    lru_cfg.policy         = tierflow::CachePolicy::LRU;
    tierflow::tf_cache lru_cache(lru_cfg);
    for (uint32_t x : textbook) lru_cache.access(x);
    CHECK(lru_cache.misses() == 10, "LRU textbook miss count == 10");
    CHECK(lru_cache.hits() == 2, "LRU textbook hit count == 2");

    // 2. Scan-resistance: W-TinyLFU+SLRU beats LRU on Zipf-like hot working set + cold burst
    constexpr size_t kCap = 24;
    tierflow::CacheConfig wt_cfg{};
    wt_cfg.capacity_slots = kCap;
    wt_cfg.policy         = tierflow::CachePolicy::WTinyLFU_SLRU;
    wt_cfg.age_interval   = 1024;
    tierflow::tf_cache wt_scan(wt_cfg);

    tierflow::CacheConfig lru_scan_cfg{};
    lru_scan_cfg.capacity_slots = kCap;
    lru_scan_cfg.policy         = tierflow::CachePolicy::LRU;
    tierflow::tf_cache lru_scan(lru_scan_cfg);

    std::vector<uint32_t> scan_trace;
    for (int round = 0; round < 6; ++round) {
        // Warm up 16 hot items heavily (10 passes = 160 accesses)
        for (int p = 0; p < 10; ++p) {
            for (uint32_t hot = 1; hot <= 16; ++hot) scan_trace.push_back(hot);
        }
        // Cold scan burst of 32 distinct items once
        const uint32_t cold_start = 1000u + static_cast<uint32_t>(round) * 100u;
        for (uint32_t cold = cold_start; cold < cold_start + 32u; ++cold) {
            scan_trace.push_back(cold);
        }
    }
    for (uint32_t x : scan_trace) {
        wt_scan.access(x);
        lru_scan.access(x);
    }
    std::printf("       Scan trace hits: W-TinyLFU=%llu vs LRU=%llu (total=%zu)\n",
                static_cast<unsigned long long>(wt_scan.hits()),
                static_cast<unsigned long long>(lru_scan.hits()),
                scan_trace.size());
    CHECK(wt_scan.hits() > lru_scan.hits(), "W-TinyLFU must beat LRU on hot+burst scan trace");

    // 3. Refcount protection: referenced slot is NEVER evicted
    tierflow::CacheConfig ref_cfg{};
    ref_cfg.capacity_slots = 8;
    ref_cfg.policy         = tierflow::CachePolicy::WTinyLFU_SLRU;
    tierflow::tf_cache ref_cache(ref_cfg);

    std::vector<int32_t> pinned_slots;
    for (uint32_t obj = 100; obj < 108; ++obj) {
        auto r = ref_cache.access(obj);
        CHECK(r.slot_id >= 0, "Initial fill allocates valid slot");
    }
    // Acquire refcount on objects 100..106 (7 out of 8 slots locked)
    for (uint32_t obj = 100; obj < 107; ++obj) {
        int32_t s = ref_cache.find_slot(obj);
        CHECK(s >= 0, "Object must be resident");
        CHECK(ref_cache.acquire_slot(s), "Acquire slot succeeds");
        pinned_slots.push_back(s);
    }
    // Hammer with 20 new misses: ONLY slot holding 107 may be churned; 100..106 MUST stay!
    for (uint32_t obj = 200; obj < 220; ++obj) {
        ref_cache.access(obj);
        for (uint32_t locked = 100; locked < 107; ++locked) {
            CHECK(ref_cache.is_resident(locked), "Referenced slot was unexpectedly evicted");
        }
    }
    // Acquire the 8th slot too -> all 8 slots now have refcount > 0 -> new miss must return slot_id == -1
    int32_t eighth_slot = -1;
    for (int32_t s = 0; s < 8; ++s) {
        if (ref_cache.refcount(s) == 0) eighth_slot = s;
    }
    CHECK(eighth_slot >= 0, "One unreferenced slot should exist");
    CHECK(ref_cache.acquire_slot(eighth_slot), "Acquire 8th slot");
    auto full_res = ref_cache.access(9999);
    CHECK(full_res.slot_id == -1, "Must refuse eviction when ALL slots are referenced");

    for (int32_t s : pinned_slots) CHECK(ref_cache.release_slot(s), "Release refcount");
    CHECK(ref_cache.release_slot(eighth_slot), "Release 8th slot");
}

void run_io_backend_tests(tierflow::tf_io_ex* io, const std::string& path,
                          uint64_t file_size, const std::vector<uint8_t>& ground_truth) {
    CHECK(!io->open("Z:\\nonexistent_tierflow_dir\\missing_file.bin"), "Non-existent file rejected");
    CHECK(io->open(path.c_str()), "Open temp file");
    std::printf("[TEST] tf_io (%s)...\n", io->backend_name());
    CHECK(io->file_size() == file_size, "File size matches");
    CHECK(io->alignment() == 4096, "Alignment == 4096");

    void* aligned_buf = tierflow::tf_aligned_alloc(4ULL << 20, 4096);
    CHECK(aligned_buf != nullptr, "Aligned buffer allocated");

    // 1. Misalignment & bounds rejections
    CHECK(!io->submit({0, 512, 4096, aligned_buf, tierflow::TF_PRIO_DEMAND, 1}),
          "Misaligned offset rejected");
    CHECK(!io->submit({0, 0, 4096 + 512, aligned_buf, tierflow::TF_PRIO_DEMAND, 2}),
          "Misaligned size rejected");
    CHECK(!io->submit({0, 0, 4096, static_cast<uint8_t*>(aligned_buf) + 64, tierflow::TF_PRIO_DEMAND, 3}),
          "Misaligned dst pointer rejected");
    CHECK(!io->submit({0, 0, 0, aligned_buf, tierflow::TF_PRIO_DEMAND, 4}),
          "Zero size rejected");
    CHECK(!io->submit({0, 0, 4096, nullptr, tierflow::TF_PRIO_DEMAND, 5}),
          "Null dst rejected");
    CHECK(!io->submit({0, file_size - 4096, 8192, aligned_buf, tierflow::TF_PRIO_DEMAND, 6}),
          "Past-EOF read rejected");

    // 2. Concurrent multi-offset varied-size reads (out-of-order completion check)
    struct ChunkJob {
        uint64_t off;
        uint64_t sz;
        uint64_t tag;
        void*    buf;
        bool     done = false;
    };
    std::vector<ChunkJob> jobs;
    const size_t kNumJobs = 20;
    for (size_t i = 0; i < kNumJobs; ++i) {
        // Mix varied sizes: 4 KiB, 64 KiB, 512 KiB, 1 MiB, in reversed offset order
        uint64_t sz = (i % 4 == 0) ? 1048576ULL : (i % 4 == 1) ? 4096ULL : (i % 4 == 2) ? 65536ULL : 524288ULL;
        uint64_t off = ((kNumJobs - 1 - i) * 1048576ULL) % (file_size - sz);
        off = (off / 4096ULL) * 4096ULL;
        void* b = tierflow::tf_aligned_alloc(sz, 4096);
        std::memset(b, 0xCC, sz);
        uint64_t tag = 100 + i;
        jobs.push_back({off, sz, tag, b, false});
        CHECK(io->submit({static_cast<uint32_t>(i), off, sz, b, tierflow::TF_PRIO_DEMAND, tag}),
              "Valid read submit");
    }

    size_t collected = 0;
    std::vector<tierflow::tf_read_req> done_vec(32);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (collected < kNumJobs && std::chrono::steady_clock::now() < deadline) {
        int n = io->poll(done_vec.data(), static_cast<int>(done_vec.size()));
        for (int k = 0; k < n; ++k) {
            size_t idx = static_cast<size_t>(done_vec[k].tag - 100);
            CHECK(idx < kNumJobs, "Returned tag in range");
            if (idx < kNumJobs) {
                CHECK(!jobs[idx].done, "No duplicate completion");
                jobs[idx].done = true;
                CHECK(std::memcmp(jobs[idx].buf, ground_truth.data() + jobs[idx].off, jobs[idx].sz) == 0,
                      "Unbuffered payload matches ground truth byte-for-byte");
                ++collected;
            }
        }
        if (n == 0) std::this_thread::yield();
    }
    CHECK(collected == kNumJobs, "All out-of-order jobs completed");
    for (auto& j : jobs) tierflow::tf_aligned_free(j.buf);

    // 3. Cancellation guarantee test:
    // Submit a wave of 32 prefetch reads (1 MiB each). Cancel half of them (queued and in-flight),
    // immediately overwrite their buffers with sentinel 0x5A right when cancel() returns true,
    // drain remaining reads + sleep, and assert ZERO bytes in cancelled buffers changed!
    constexpr size_t kPrefetchWave = 32;
    constexpr uint64_t kChunkSz    = 1048576ULL;
    struct CancelProbe {
        uint64_t tag;
        void*    buf;
        bool     cancelled = false;
        bool     checked   = false;
    };
    std::vector<CancelProbe> probes(kPrefetchWave);
    for (size_t i = 0; i < kPrefetchWave; ++i) {
        probes[i].tag = 5000 + i;
        probes[i].buf = tierflow::tf_aligned_alloc(kChunkSz, 4096);
        std::memset(probes[i].buf, 0xA5, kChunkSz);
        uint64_t off = ((i * 524288ULL) % (file_size - kChunkSz)) & ~4095ULL;
        CHECK(io->submit({static_cast<uint32_t>(i), off, kChunkSz, probes[i].buf,
                          tierflow::TF_PRIO_PREFETCH, probes[i].tag}),
              "Prefetch submit");
    }

    // Cancel from back (software queued) AND middle/front (likely kernel in-flight!)
    for (int i = static_cast<int>(kPrefetchWave) - 1; i >= 0; i -= 2) {
        if (io->cancel(probes[i].tag)) {
            probes[i].cancelled = true;
            // Stamp sentinel IMMEDIATELY upon cancel acknowledgement!
            std::memset(probes[i].buf, 0x5A, kChunkSz);
        }
    }

    // Drain remaining completions and wait briefly to catch any stray DMA writes
    deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        auto st = io->stats();
        if (st.inflight_prefetch == 0 && st.queued_prefetch == 0 && st.inflight_demand == 0 && st.queued_demand == 0) {
            break;
        }
        int n = io->poll(done_vec.data(), static_cast<int>(done_vec.size()));
        for (int k = 0; k < n; ++k) {
            size_t idx = static_cast<size_t>(done_vec[k].tag - 5000);
            CHECK(!probes[idx].cancelled, "Cancelled tag must not be returned by poll()");
        }
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    size_t cancel_verified = 0;
    for (size_t i = 0; i < kPrefetchWave; ++i) {
        if (probes[i].cancelled) {
            const uint8_t* p = static_cast<const uint8_t*>(probes[i].buf);
            bool intact = true;
            for (size_t b = 0; b < kChunkSz; ++b) {
                if (p[b] != 0x5A) {
                    intact = false;
                    break;
                }
            }
            CHECK(intact, "Buffer written AFTER cancel() acknowledgement!");
            ++cancel_verified;
        }
        tierflow::tf_aligned_free(probes[i].buf);
    }
    std::printf("       Verified %zu cancelled prefetch buffers untouched after cancel ack\n", cancel_verified);
    CHECK(cancel_verified > 0, "At least one prefetch request was cancelled and verified");

    tierflow::tf_aligned_free(aligned_buf);
    io->close();
}

void test_planner() {
    std::printf("[TEST] tf_planner...\n");
    tierflow::HardwareProfile probe{};

    // 1. Dense model: fills fastest tier (VRAM) first, then RAM
    std::vector<tierflow::PlannerObjectInput> dense_objs;
    for (uint32_t i = 0; i < 6; ++i) {
        dense_objs.push_back({i, 100ULL << 20, 1.0, true});
    }
    tierflow::PlannerBudget dense_budget{250ULL << 20, 400ULL << 20, false};
    auto dense_plan = tierflow::build_placement_plan(dense_objs, dense_budget, probe);
    CHECK(dense_plan.feasible, "Dense plan fits VRAM+RAM");
    CHECK(dense_plan.vram_used_bytes == (200ULL << 20), "Dense planner fills VRAM fastest tier first (200 MiB)");
    CHECK(dense_plan.ram_used_bytes == (400ULL << 20), "Dense planner places remainder in RAM (400 MiB)");
    CHECK(dense_plan.ssd_stream_bytes == 0, "Zero SSD spill when dense weights fit");

    // 2. Toy MoE <= 12 objects compared against exhaustive 3^N enumeration
    std::vector<tierflow::PlannerObjectInput> toy{
        {0, 100ULL << 20, 1.00, true},   // always-on attention/head
        {1,  80ULL << 20, 0.90, false},  // super-hot expert
        {2,  80ULL << 20, 0.70, false},
        {3,  80ULL << 20, 0.50, false},
        {4,  80ULL << 20, 0.30, false},
        {5,  80ULL << 20, 0.15, false},
        {6,  80ULL << 20, 0.05, false},
        {7,  80ULL << 20, 0.02, false},
    };
    tierflow::PlannerBudget toy_budget{180ULL << 20, 160ULL << 20, false};
    auto planner_res = tierflow::build_placement_plan(toy, toy_budget, probe);
    CHECK(planner_res.feasible, "Toy MoE plan feasible");

    // Exhaustive enumeration of all 3^8 = 6561 placements as independent oracle
    double oracle_best = 1e30;
    uint32_t limit = 1;
    for (size_t i = 0; i < toy.size(); ++i) limit *= 3;
    std::vector<tierflow::PlacementTier> cand(toy.size());
    for (uint32_t mask = 0; mask < limit; ++mask) {
        uint32_t tmp = mask;
        uint64_t vu = 0, ru = 0;
        bool ok = true;
        for (size_t i = 0; i < toy.size(); ++i) {
            uint32_t t = tmp % 3;
            tmp /= 3;
            cand[i] = static_cast<tierflow::PlacementTier>(t);
            if (cand[i] == tierflow::PlacementTier::VRAM) vu += toy[i].bytes;
            else if (cand[i] == tierflow::PlacementTier::RAM) ru += toy[i].bytes;
            else if (toy[i].always_on) ok = false;
        }
        if (!ok || vu > toy_budget.vram_bytes || ru > toy_budget.ram_bytes) continue;
        double c = tierflow::evaluate_placement_cost_sec(toy, cand, probe);
        if (c < oracle_best) oracle_best = c;
    }
    CHECK(std::abs(planner_res.expected_sec_per_tok - oracle_best) < 1e-12,
          "Planner matches exhaustive 3^N brute-force optimal cost");

    // 3. Budget governor enforcement
    tierflow::tf_governor strict_gov(100ULL << 20, 100ULL << 20);
    CHECK(!tierflow::validate_plan_with_governor(planner_res, strict_gov),
          "Governor rejects plan exceeding tighter runtime budgets");
}

void test_spec_scheduler() {
    std::printf("[TEST] tf_spec_scheduler...\n");

    // 1. Exact closed form (1 - alpha^(k+1)) / (1 - alpha)
    double alpha = 0.72;
    double manual_sum = 0.0;
    for (int k = 0; k <= 6; ++k) {
        manual_sum += std::pow(alpha, k);
        double cf = tierflow::spec_expected_accepted(alpha, k);
        CHECK(std::abs(cf - manual_sum) < 1e-12, "Closed form matches finite geometric sum");
    }

    tierflow::SpecSchedulerConfig cfg{};
    cfg.k_max = 8;
    tierflow::tf_spec_scheduler sched(cfg);

    // 2. alpha -> 0 returns k = 0
    tierflow::SpecInputs zero_a{};
    zero_a.alpha_hat           = 0.0;
    zero_a.c_always_sec        = 0.010;
    zero_a.cost_per_expert_sec = 0.002;
    zero_a.t_draft_sec         = 0.001;
    zero_a.u_total_by_k        = {8, 12, 18, 28, 42, 60, 80, 96, 110};
    auto dec_zero = sched.choose_k(zero_a);
    CHECK(dec_zero.k_opt == 0, "alpha -> 0 returns k_opt == 0");

    // 3. Flat U_L(k) (dense-like constant weight fetch) grows k when alpha is high
    tierflow::SpecInputs dense_like{};
    dense_like.alpha_hat           = 0.85;
    dense_like.c_always_sec        = 0.020;
    dense_like.cost_per_expert_sec = 0.001;
    dense_like.t_draft_sec         = 0.0005;
    dense_like.u_total_by_k        = {8, 8, 8, 8, 8, 8, 8, 8, 8};
    auto dec_dense = sched.choose_k(dense_like);

    // Independent-routing MoE (steep linear U_L(k)) forces smaller k* than flat dense-like!
    tierflow::SpecInputs moe_steep = dense_like;
    moe_steep.cost_per_expert_sec  = 0.002;
    moe_steep.u_total_by_k         = {8, 16, 24, 32, 40, 48, 56, 64, 72};
    auto dec_moe = sched.choose_k(moe_steep);
    std::printf("       Spec k*: alpha=0 -> %d, dense_flat -> %d, moe_steep -> %d\n",
                dec_zero.k_opt, dec_dense.k_opt, dec_moe.k_opt);
    CHECK(dec_dense.k_opt >= 4, "Flat U_L(k) with high alpha grows k");
    CHECK(dec_moe.k_opt < dec_dense.k_opt, "Steep MoE union growth reduces optimal k*");
}

void test_manifest_probe_compute_prefetch() {
    std::printf("[TEST] tf_manifest, tf_probe, tf_compute_place, tf_prefetch...\n");

    // Manifest real file verification
    const std::string mpath = "E:\\RAM_optimizer\\results\\p0_dense_baseline\\pack_verification\\qwen2.5-1.5b-instruct-q4_k_m.manifest.json";
    if (std::filesystem::exists(mpath)) {
        tierflow::TfManifest mf;
        std::string err;
        CHECK(tierflow::load_tf_manifest(mpath, mf, &err), "Load real manifest.json");
        CHECK(mf.architecture == "qwen2", "Parsed architecture == qwen2");
        CHECK(mf.block_count == 28, "Parsed block_count == 28");
        CHECK(mf.objects.size() == 339, "Parsed all 339 tensor objects");
        CHECK(mf.tfpack_size_bytes == 1111777280ULL, "Parsed exact tfpack_size_bytes");
    }

    auto probe = tierflow::run_tf_probe();
    CHECK(probe.probed_live, "Live hardware probe completed");
    CHECK(probe.bw_ram_copy_gbps > 1.0 && probe.bw_ram_read_gbps > 1.0, "Plausible live DDR5 GB/s");
    std::printf("       Live DDR5 probe: copy=%.2f GB/s, read=%.2f GB/s\n",
                probe.bw_ram_copy_gbps, probe.bw_ram_read_gbps);

    // Compute placement crossover check (9.4 MiB expert in host RAM)
    constexpr uint64_t kExpBytes = 9437184ULL;
    constexpr uint64_t kExpFlops = 2ULL * 2048 * 768 * 3;
    auto b1 = tierflow::choose_compute_target(kExpBytes, kExpFlops, 1, false, probe);
    auto b8 = tierflow::choose_compute_target(kExpBytes, kExpFlops, 8, false, probe);
    CHECK(b1.target == tierflow::ComputeTarget::CPU_IN_PLACE, "B=1 RAM expert runs on CPU");
    CHECK(b8.target == tierflow::ComputeTarget::GPU_UPLOAD, "B=8 RAM expert uploads to GPU");

    // Prefetch adaptation test: collapse accuracy -> shrinks m
    tierflow::tf_prefetch_controller pf{};
    int init_m = pf.current_m();
    for (int i = 0; i < 20; ++i) {
        pf.record_observation({1, 2, 3, 4}, {50, 51, 52, 53}); // 0% overlap
    }
    CHECK(pf.current_m() < init_m, "0% prefetch accuracy shrinks look-ahead margin m");
}

} // namespace

int main() {
    test_governor();
    test_cache();
    test_planner();
    test_spec_scheduler();
    test_manifest_probe_compute_prefetch();

    constexpr uint64_t kTestFileSize = 16ULL << 20; // 16 MiB seeded payload
    std::string temp_path = make_temp_test_file(kTestFileSize, 20261112ULL);
    CHECK(!temp_path.empty(), "Created temp test file");

    std::vector<uint8_t> truth(static_cast<size_t>(kTestFileSize));
    if (!temp_path.empty()) {
        std::FILE* f = std::fopen(temp_path.c_str(), "rb");
        if (f) {
            const size_t nread = std::fread(truth.data(), 1, truth.size(), f);
            (void)nread;
            std::fclose(f);
        }
    }

#ifdef _WIN32
    auto iocp = tierflow::tf_io_create_iocp();
    run_io_backend_tests(iocp.get(), temp_path, kTestFileSize, truth);
#else
    auto linux_io = tierflow::tf_io_create_auto();
    run_io_backend_tests(linux_io.get(), temp_path, kTestFileSize, truth);
#endif

    std::string ioring_reason;
    if (tierflow::tf_ioring_is_supported(&ioring_reason)) {
        std::printf("[INFO] %s\n", ioring_reason.c_str());
        auto ioring = tierflow::tf_io_create_ioring();
        run_io_backend_tests(ioring.get(), temp_path, kTestFileSize, truth);
    } else {
        std::printf("[INFO] Skipping IoRing test: %s\n", ioring_reason.c_str());
    }

    if (!temp_path.empty()) {
        std::error_code ec;
        std::filesystem::remove(temp_path, ec);
    }

    {
        std::printf("[TEST] tf_kv_store (q8_0 codec, governor spill, prefix cache)...\n");
        // 1. q8_0 roundtrip accuracy check
        constexpr size_t kDim = 256;
        std::vector<float> orig(kDim), back(kDim);
        for (size_t i = 0; i < kDim; ++i) {
            orig[i] = std::sin(0.17f * static_cast<float>(i)) * 2.5f;
        }
        std::vector<uint8_t> enc(((kDim + 31) / 32) * 34);
        tierflow::quantize_row_q8_0(orig.data(), enc.data(), kDim);
        tierflow::dequantize_row_q8_0(enc.data(), back.data(), kDim);
        double dot = 0.0, na = 0.0, nb = 0.0, max_err = 0.0;
        for (size_t i = 0; i < kDim; ++i) {
            dot += orig[i] * back[i];
            na  += orig[i] * orig[i];
            nb  += back[i] * back[i];
            max_err = std::max(max_err, std::abs(static_cast<double>(orig[i] - back[i])));
        }
        double cos_sim = dot / (std::sqrt(na) * std::sqrt(nb));
        CHECK(cos_sim > 0.9999, "q8_0 roundtrip cosine similarity > 0.9999");
        CHECK(max_err < 0.03, "q8_0 max abs error < 0.03");

        // 2. Governor-capped RAM + direct unbuffered SSD spill + prefix cache
        auto spill_file = (std::filesystem::temp_directory_path() / "tf_kv_spill_test.bin").string();
        auto prefix_dir = (std::filesystem::temp_directory_path() / "tf_kv_prefix_test").string();
        std::error_code ec;
        std::filesystem::remove_all(prefix_dir, ec);

        tierflow::KvConfig kcfg;
        kcfg.n_layers    = 2;
        kcfg.n_kv_heads  = 2;
        kcfg.head_dim_k  = 64;
        kcfg.head_dim_v  = 64;
        kcfg.page_tokens = 16;
        kcfg.precision   = tierflow::KvPrecision::Q8_0;
        kcfg.spill_path  = spill_file;
        kcfg.prefix_dir  = prefix_dir;

        // 1 page aligned is ~4096 bytes (`16 * 128 / 32 * 34 = 2176` x 2 = 4352 -> 8192 bytes).
        // Give RAM budget for ONLY 2 pages so inserting 5 pages forces 3 SSD spills!
        tierflow::tf_governor probe_gov(1ULL << 30, 0, 1ULL << 30);
        size_t page_sz = 0;
        {
            tierflow::tf_kv_store tmp_kv(kcfg, probe_gov);
            page_sz = tmp_kv.page_aligned_bytes();
        }
        tierflow::tf_governor kv_gov(page_sz * 2, 0, page_sz * 16);
        {
            tierflow::tf_kv_store kv(kcfg, kv_gov);
            const size_t row_e = size_t(kcfg.n_kv_heads) * kcfg.head_dim_k;
            const size_t pg_e  = row_e * kcfg.page_tokens;
            std::vector<std::vector<float>> ks(5, std::vector<float>(pg_e));
            std::vector<std::vector<float>> vs(5, std::vector<float>(pg_e));
            for (uint32_t p = 0; p < 5; ++p) {
                for (size_t i = 0; i < pg_e; ++i) {
                    ks[p][i] = std::cos(0.03f * float(i + p * 17));
                    vs[p][i] = std::sin(0.05f * float(i + p * 29));
                }
                CHECK(kv.put_page_f32(1, 0, p, kcfg.page_tokens, ks[p].data(), vs[p].data()),
                      "Put KV page succeeds with SSD spill");
            }
            auto st = kv.stats();
            CHECK(st.pages_resident_ram == 2, "RAM page count strictly <= 2-page governor budget");
            CHECK(st.pages_spilled_ssd == 3, "3 LRU pages spilled to SSD");
            CHECK(st.spill_writes >= 3, "Direct SSD spill writes recorded");

            // Read back spilled page 0 and verify q8_0 fidelity
            std::vector<float> k_got, v_got;
            CHECK(kv.get_page_f32(1, 0, 0, k_got, v_got), "Reload spilled page 0 from SSD");
            CHECK(kv.stats().spill_reads >= 1, "Direct SSD spill read recorded");
            double max_reload_err = 0.0;
            for (size_t i = 0; i < pg_e; ++i) {
                max_reload_err = std::max(max_reload_err, std::abs(double(ks[0][i] - k_got[i])));
            }
            CHECK(max_reload_err < 0.02, "Spilled+reloaded q8_0 page preserves values");

            // Prefix block save & bit-exact reload
            std::vector<uint8_t> blob{0xDE, 0xAD, 0xBE, 0xEF, 1, 2, 3, 4, 5};
            std::vector<uint8_t> loaded;
            CHECK(kv.save_prefix_block(0x123456789ABCDEF0ULL, blob.data(), blob.size()),
                  "Save prefix block");
            CHECK(kv.load_prefix_block(0x123456789ABCDEF0ULL, loaded),
                  "Load prefix block");
            CHECK(loaded == blob, "Loaded prefix block is bit-exact");
        }
        CHECK(kv_gov.used(tierflow::MemoryTier::RAM) == 0, "KV destructor releases all RAM budget");
        CHECK(kv_gov.used(tierflow::MemoryTier::SSD_SPILL) == 0, "KV destructor releases all SSD spill budget");
        std::filesystem::remove_all(prefix_dir, ec);
    }

    {
        std::printf("[TEST] Phase 9 3-size-class arena planner, repin starvation guard, BW EWMA...\n");
        CHECK(tierflow::assign_size_class(4096) == 0, "4 KiB -> small class 0");
        CHECK(tierflow::assign_size_class(512ULL << 10) == 0, "512 KiB -> small class 0");
        CHECK(tierflow::assign_size_class(516ULL << 10) == 1, "516 KiB -> medium class 1");
        CHECK(tierflow::assign_size_class(4ULL << 20) == 1, "4 MiB -> medium class 1");
        CHECK(tierflow::assign_size_class((4ULL << 20) + 4096) == 2, ">4 MiB -> large class 2");

        // Verify empty classes allocate 0 slots/bytes, non-empty fit strictly under budget
        std::vector<uint64_t> sizes{
            256ULL << 10, 256ULL << 10, 512ULL << 10, 256ULL << 10, // 4 small (stride 512 KiB)
            3ULL << 20, 3ULL << 20, 4ULL << 20, 3ULL << 20,         // 4 medium (stride 4 MiB)
        };
        const uint64_t budget = 64ULL << 20;
        auto aplan = tierflow::compute_arena_layout(sizes, budget, 16, 4);
        CHECK(aplan.feasible, "Arena layout feasible within 64 MiB");
        CHECK(aplan.classes[0].n_slots == 4, "4 small slots match 4 small objects");
        CHECK(aplan.classes[1].n_slots == 4, "4 medium slots match 4 medium objects");
        CHECK(aplan.classes[2].n_slots == 0 && aplan.classes[2].total_bytes == 0, "Empty large class allocates 0 bytes");
        CHECK(aplan.total_bytes <= budget, "Arena total bytes strictly <= budget");

        // Repin tight-slot deadlock guard
        CHECK(!tierflow::can_repin_slot(8, 0, 2), "Tight 8-slot pool forbids repinning");
        CHECK(tierflow::can_repin_slot(16, 0, 4), "16-slot pool allows first repin with 11 free slots >= 8");
        CHECK(!tierflow::can_repin_slot(16, 5, 4), "16-slot pool stops repinning at ratio cap");
        CHECK(!tierflow::can_repin_slot(16, 4, 8), "16-slot pool stops repinning when remaining < inflight+4");

        // Dynamic prefetch QD & EWMA bandwidth
        CHECK(tierflow::compute_dynamic_prefetch_cap(2) == 0, "<4 free slots disables prefetch");
        CHECK(tierflow::compute_dynamic_prefetch_cap(12) == 6, "12 free slots -> QD 6");
        CHECK(tierflow::compute_dynamic_prefetch_cap(64) == 16, "64 free slots caps QD at 16");

        double bw = tierflow::update_bw_ewma(1.0, 16ULL * 1000 * 1000, 0.004, 0.85); // sample = 4.0 GB/s
        CHECK(std::abs(bw - 1.45) < 1e-6, "BW EWMA blends 0.85*1.0 + 0.15*4.0 = 1.45 GB/s");

        // Regression test: W-TinyLFU window->probation promotion must never change physical slot bytes
        tierflow::CacheConfig scfg{};
        scfg.capacity_slots = 12;
        scfg.policy = tierflow::CachePolicy::WTinyLFU_SLRU;
        tierflow::tf_cache scache(scfg);
        std::vector<uint32_t> slot_contents(12, 0xFFFFFFFFu);
        for (uint32_t step = 0; step < 300; ++step) {
            const uint32_t obj = (step < 66) ? (step % 11u) : (step % 19u);
            auto r = scache.access(obj);
            CHECK(r.slot_id >= 0 && r.slot_id < 12, "Physical slot bounds valid");
            if (r.hit) {
                CHECK(slot_contents[static_cast<size_t>(r.slot_id)] == obj,
                      "Cache hit returns exact physical slot containing object");
            } else {
                slot_contents[static_cast<size_t>(r.slot_id)] = obj;
            }
        }
    }

    {
        std::printf("[TEST] Phase 10 sliced MoE manifest parsing & medium arena layout...\n");
        const auto m_path = std::filesystem::temp_directory_path() / "tf_test_moe_v2.json";
        {
            std::ofstream ofs(m_path, std::ios::binary | std::ios::trunc);
            ofs << R"({
  "format": "TierFlow-TFPack-v2",
  "tfpack_path": "dummy.tfpack",
  "slice_moe_experts": true,
  "gguf_metadata": {
    "general.architecture": "qwen3moe",
    "qwen3moe.block_count": 48,
    "qwen3moe.attention.head_count": 32,
    "qwen3moe.attention.head_count_kv": 4,
    "qwen3moe.embedding_length": 2048,
    "qwen3moe.expert_count": 128,
    "qwen3moe.expert_used_count": 8
  },
  "objects": [
    {"object_id": 0, "name": "blk.0.ffn_gate_exps.weight#e0", "parent_tensor": "blk.0.ffn_gate_exps.weight", "expert_idx": 0, "n_experts": 128, "role": "moe_experts", "layer": 0, "exact_bytes": 884736, "page_aligned_bytes": 884736, "tfpack_offset": 4096},
    {"object_id": 1, "name": "blk.0.ffn_down_exps.weight#e7", "parent_tensor": "blk.0.ffn_down_exps.weight", "expert_idx": 7, "n_experts": 128, "role": "moe_experts", "layer": 0, "exact_bytes": 1290240, "page_aligned_bytes": 1290240, "tfpack_offset": 888832}
  ]
})";
        }
        tierflow::TfManifest mf;
        std::string err;
        CHECK(tierflow::load_tf_manifest(m_path.string(), mf, &err), "Parse sliced MoE v2 manifest");
        std::filesystem::remove(m_path);
        CHECK(mf.slice_moe_experts, "slice_moe_experts flag parsed");
        CHECK(mf.expert_count == 128 && mf.expert_used_count == 8, "expert_count and expert_used_count parsed");
        CHECK(mf.objects.size() == 2, "2 MoE slice objects parsed");
        CHECK(mf.objects[0].role == tierflow::ObjectRole::EXPERT, "moe_experts role mapped");
        CHECK(mf.objects[0].parent_tensor == "blk.0.ffn_gate_exps.weight", "parent_tensor preserved");
        CHECK(mf.objects[1].expert_idx == 7 && mf.objects[1].n_experts == 128, "expert_idx & n_experts preserved");

        std::vector<uint64_t> moe_sizes(384, 1290240ULL);
        auto moe_plan = tierflow::compute_arena_layout(moe_sizes, 512ULL << 20, 65536ULL, 4);
        CHECK(moe_plan.feasible, "MoE 512 MiB Class-1 arena feasible");
        CHECK(moe_plan.classes[0].n_slots == 0 && moe_plan.classes[2].n_slots == 0,
              "Only Class-1 (512KiB..4MiB) slots allocated for Qwen3-30B-A3B slices");
        CHECK(moe_plan.classes[1].n_slots > 300, "Hundreds of hot expert slices fit in 512 MiB Class-1 arena");
        CHECK(moe_plan.total_bytes <= (512ULL << 20), "MoE arena total bytes strictly <= budget");
    }

    if (g_failures == 0) {
        std::printf("\nALL TIERFLOW UNIT TESTS PASSED.\n");
        return 0;
    }
    std::fprintf(stderr, "\n%d TIERFLOW UNIT TEST(S) FAILED.\n", g_failures);
    return 1;
}
