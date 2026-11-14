#include "tf_llama_stream.h"
#include "tf_mem_win.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    return out;
}

struct PromptItem {
    std::string id;
    std::string split;
    std::string prompt;
};

bool extract_json_str(const std::string& line, const char* key, std::string& out) {
    std::string pat = std::string("\"") + key + "\"";
    auto p = line.find(pat);
    if (p == std::string::npos) return false;
    p = line.find(':', p + pat.size());
    if (p == std::string::npos) return false;
    p = line.find('"', p + 1);
    if (p == std::string::npos) return false;
    ++p;
    out.clear();
    while (p < line.size()) {
        char c = line[p++];
        if (c == '"') return true;
        if (c == '\\' && p < line.size()) {
            char e = line[p++];
            if (e == 'n') out.push_back('\n');
            else if (e == 'r') out.push_back('\r');
            else if (e == 't') out.push_back('\t');
            else out.push_back(e);
        } else {
            out.push_back(c);
        }
    }
    return false;
}

std::vector<PromptItem> load_prompts(const std::string& path) {
    std::vector<PromptItem> items;
    std::ifstream ifs(path);
    std::string line;
    int idx = 0;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        PromptItem pi;
        if (line.front() == '{') {
            if (!extract_json_str(line, "prompt", pi.prompt)) continue;
            if (!extract_json_str(line, "id", pi.id)) pi.id = "p" + std::to_string(idx);
            if (!extract_json_str(line, "split", pi.split)) pi.split = "unknown";
        } else {
            pi.id     = "p" + std::to_string(idx);
            pi.split  = "text";
            pi.prompt = line;
        }
        items.push_back(std::move(pi));
        ++idx;
    }
    return items;
}

} // namespace

int main(int argc, char** argv) {
    tierflow::StreamEngineConfig cfg{};
    int32_t     n_predict      = 64;
    int32_t     repeat_count   = 1;
    float       temp           = 0.0f;
    uint32_t    seed           = 42;
    bool        prefix_use     = false;
    std::string single_prompt;
    std::string prompts_file;
    std::string out_jsonl;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--mode") {
            std::string v = need("--mode");
            if (v == "tierflow") cfg.mode = tierflow::RunMode::TierFlow;
            else if (v == "stock_mmap") cfg.mode = tierflow::RunMode::StockMmap;
            else if (v == "stock_dio") cfg.mode = tierflow::RunMode::StockDio;
            else { std::fprintf(stderr, "Unknown --mode %s\n", v.c_str()); return 2; }
        } else if (a == "--gguf") cfg.gguf_path = need("--gguf");
        else if (a == "--manifest") cfg.manifest_path = need("--manifest");
        else if (a == "--ram-budget-mb") cfg.ram_budget_bytes = std::stoull(need("--ram-budget-mb")) << 20;
        else if (a == "--vram-mb") cfg.vram_budget_bytes = std::stoull(need("--vram-mb")) << 20;
        else if (a == "--vram-reserve-mib") cfg.vram_reserve_bytes = std::stoull(need("--vram-reserve-mib")) << 20;
        else if (a == "--n-gpu-layers") cfg.n_gpu_layers = std::stoi(need("--n-gpu-layers"));
        else if (a == "--n-ctx") cfg.n_ctx = static_cast<uint32_t>(std::stoul(need("--n-ctx")));
        else if (a == "--n-batch") cfg.n_batch = static_cast<uint32_t>(std::stoul(need("--n-batch")));
        else if (a == "--n-threads") cfg.n_threads = std::stoi(need("--n-threads"));
        else if (a == "--n-predict") n_predict = std::stoi(need("--n-predict"));
        else if (a == "--repeat") repeat_count = std::max(1, std::stoi(need("--repeat")));
        else if (a == "--kv-type") {
            std::string v = need("--kv-type");
            if (v == "f16") cfg.kv_precision = tierflow::KvPrecision::F16;
            else if (v == "q8_q4" || v == "q8_k_q4_v") cfg.kv_precision = tierflow::KvPrecision::Q8_K_Q4_V;
            else cfg.kv_precision = tierflow::KvPrecision::Q8_0;
        } else if (a == "--cache-policy") {
            std::string v = need("--cache-policy");
            cfg.cache_policy = (v == "lru") ? tierflow::CachePolicy::LRU : tierflow::CachePolicy::WTinyLFU_SLRU;
        } else if (a == "--cache-slots") cfg.cache_slots_max = std::stoull(need("--cache-slots"));
        else if (a == "--prefetch") {
            std::string v = need("--prefetch");
            if (v == "lookahead") cfg.prefetch_mode = tierflow::PrefetchMode::LookAhead;
            else if (v == "off") cfg.prefetch_mode = tierflow::PrefetchMode::Off;
            else if (v == "shuffled") cfg.prefetch_mode = tierflow::PrefetchMode::Shuffled;
            else return 2;
        } else if (a == "--lookahead-depth") cfg.lookahead_depth = std::stoi(need("--lookahead-depth"));
        else if (a == "--repin") cfg.repin_enabled = (std::stoi(need("--repin")) != 0);
        else if (a == "--spec") {
            std::string v = need("--spec");
            if (v == "off") cfg.spec_k_mode = tierflow::SpecKMode::Off;
            else if (v == "adaptive") cfg.spec_k_mode = tierflow::SpecKMode::Adaptive;
            else if (v == "fixed4") cfg.spec_k_mode = tierflow::SpecKMode::Fixed4;
            else if (v == "naive_no_union") cfg.spec_k_mode = tierflow::SpecKMode::NaiveNoUnion;
            else return 2;
        } else if (a == "--draft-model") cfg.draft_gguf_path = need("--draft-model");
        else if (a == "--spec-max-k") cfg.spec_max_k = std::max(1, std::stoi(need("--spec-max-k")));
        else if (a == "--spec-tree") {
            cfg.spec_tree = (std::stoi(need("--spec-tree")) != 0);
            if (cfg.spec_tree && cfg.spec_max_k <= 4) {
                cfg.spec_max_k = 8;
            }
        } else if (a == "--allow-slow-stream") cfg.allow_slow_stream = (std::stoi(need("--allow-slow-stream")) != 0);
        else if (a == "--no-repack") cfg.no_repack = (std::stoi(need("--no-repack")) != 0);
        else if (a == "--use-ioring") cfg.use_ioring = (std::stoi(need("--use-ioring")) != 0);
        else if (a == "--temp") temp = std::stof(need("--temp"));
        else if (a == "--seed") seed = static_cast<uint32_t>(std::stoul(need("--seed")));
        else if (a == "--prefix-cache-dir") cfg.prefix_cache_dir = need("--prefix-cache-dir");
        else if (a == "--prefix-use") prefix_use = (std::stoi(need("--prefix-use")) != 0);
        else if (a == "--prompt") single_prompt = need("--prompt");
        else if (a == "--prompts-file") prompts_file = need("--prompts-file");
        else if (a == "--out-jsonl") out_jsonl = need("--out-jsonl");
    }

    if (cfg.gguf_path.empty()) {
        std::fprintf(stderr, "Usage: tf_runner --gguf <model.gguf> [--manifest <m.json>] ...\n");
        return 2;
    }

    std::vector<PromptItem> prompts;
    if (!prompts_file.empty()) {
        prompts = load_prompts(prompts_file);
    } else if (!single_prompt.empty()) {
        prompts.push_back({"inline_0", "inline", single_prompt});
    } else {
        prompts.push_back({"default_0", "inline", "The capital of France is"});
    }

    // Idle baseline + 100 ms background system-wide memory sampler (Section 6.3 & 6.4)
    tierflow::tf_mem_sampler sampler;
    tierflow::tf_disk_counter disk_ctr;
    if (disk_ctr.available()) {
        disk_ctr.sample_read_bytes_per_sec(); // prime PDH counter
    }
    const auto idle_snap = sampler.sample();

    std::atomic<bool> stop_mem{false};
    std::atomic<uint64_t> max_sys_used{idle_snap.system_used_bytes};
    std::atomic<uint64_t> max_priv{idle_snap.process_private_bytes};
    std::atomic<uint64_t> max_ws{idle_snap.process_working_set_bytes};
    std::atomic<uint64_t> max_sys_cache{idle_snap.system_cache_bytes};
    std::atomic<uint64_t> max_standby{idle_snap.standby_normal_bytes};
    std::atomic<uint64_t> max_gpu{idle_snap.gpu_used_bytes};
    std::atomic<double>   pdh_disk_read_bytes_accum{0.0};
    auto prev_disk_tp = std::chrono::steady_clock::now();
    std::thread mem_thread([&]() {
        while (!stop_mem.load()) {
            auto s = sampler.sample();
            uint64_t prev = max_sys_used.load();
            while (s.system_used_bytes > prev && !max_sys_used.compare_exchange_weak(prev, s.system_used_bytes)) {}
            prev = max_priv.load();
            while (s.process_private_bytes > prev && !max_priv.compare_exchange_weak(prev, s.process_private_bytes)) {}
            prev = max_ws.load();
            while (s.process_working_set_bytes > prev && !max_ws.compare_exchange_weak(prev, s.process_working_set_bytes)) {}
            prev = max_sys_cache.load();
            while (s.system_cache_bytes > prev && !max_sys_cache.compare_exchange_weak(prev, s.system_cache_bytes)) {}
            prev = max_standby.load();
            while (s.standby_normal_bytes > prev && !max_standby.compare_exchange_weak(prev, s.standby_normal_bytes)) {}
            prev = max_gpu.load();
            while (s.gpu_used_bytes > prev && !max_gpu.compare_exchange_weak(prev, s.gpu_used_bytes)) {}
            if (disk_ctr.available()) {
                auto now_tp = std::chrono::steady_clock::now();
                double dt = std::chrono::duration<double>(now_tp - prev_disk_tp).count();
                prev_disk_tp = now_tp;
                double bps = disk_ctr.sample_read_bytes_per_sec();
                if (bps >= 0.0 && dt > 0.0) {
                    pdh_disk_read_bytes_accum.store(pdh_disk_read_bytes_accum.load() + bps * dt);
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    tierflow::tf_llama_stream engine(cfg);
    auto init_res = engine.init();
    if (!init_res.ok) {
        stop_mem.store(true);
        mem_thread.join();
        std::fprintf(stderr, "Init failed: %s\n", init_res.error.c_str());
        return 1;
    }

    std::printf(
        "PLAN model_bytes=%llu ram_weights=%llu slot_bank=%llu kv_reserve=%llu ssd_stream=%llu "
        "vram_budget=%llu vram_weights=%llu vram_kv=%llu gpu_layers=%d arenas=[%zu,%zu,%zu] "
        "streamed_objs=%zu cache_slots=%zu pred_hit=%.4f pred_tok_s=%.3f pred_tok_s_burst=%.3f load_ms=%.2f\n",
        static_cast<unsigned long long>(init_res.model_total_bytes),
        static_cast<unsigned long long>(init_res.ram_weights_bytes),
        static_cast<unsigned long long>(init_res.ram_slot_bank_bytes),
        static_cast<unsigned long long>(init_res.ram_kv_bytes),
        static_cast<unsigned long long>(init_res.ssd_stream_bytes),
        static_cast<unsigned long long>(init_res.vram_budget_bytes),
        static_cast<unsigned long long>(init_res.vram_weights_bytes),
        static_cast<unsigned long long>(init_res.vram_kv_bytes),
        init_res.n_gpu_layers_active,
        init_res.arena_slots[0],
        init_res.arena_slots[1],
        init_res.arena_slots[2],
        init_res.n_streamed_tensors,
        init_res.n_cache_slots,
        init_res.predicted_hit_rate,
        init_res.predicted_tok_s,
        init_res.predicted_tok_s_burst,
        init_res.load_time_ms);

    std::ofstream ofs;
    if (!out_jsonl.empty()) {
        ofs.open(out_jsonl, std::ios::binary | std::ios::trunc);
    }

    for (int32_t rep = 0; rep < repeat_count; ++rep) {
      for (size_t i = 0; i < prompts.size(); ++i) {
        const uint32_t cur_seed = seed + static_cast<uint32_t>(rep * prompts.size() + i);
        auto r = engine.run_prompt(prompts[i].prompt, n_predict, temp, cur_seed, prefix_use);
        auto snap_now = sampler.sample();
        const int64_t peak_sys_used_delta =
            static_cast<int64_t>(std::max(max_sys_used.load(), snap_now.system_used_bytes)) -
            static_cast<int64_t>(idle_snap.system_used_bytes);
        const int64_t peak_cache_delta =
            static_cast<int64_t>(std::max(max_sys_cache.load(), snap_now.system_cache_bytes)) -
            static_cast<int64_t>(idle_snap.system_cache_bytes);
        const int64_t end_cache_delta =
            static_cast<int64_t>(snap_now.system_cache_bytes) -
            static_cast<int64_t>(idle_snap.system_cache_bytes);
        const int64_t peak_standby_delta =
            static_cast<int64_t>(std::max(max_standby.load(), snap_now.standby_normal_bytes)) -
            static_cast<int64_t>(idle_snap.standby_normal_bytes);
        const int64_t end_standby_delta =
            static_cast<int64_t>(snap_now.standby_normal_bytes) -
            static_cast<int64_t>(idle_snap.standby_normal_bytes);

        std::ostringstream js;
        js << "{";
        js << "\"id\":\"" << json_escape(prompts[i].id) << "\",";
        js << "\"split\":\"" << json_escape(prompts[i].split) << "\",";
        js << "\"ok\":" << (r.ok ? "true" : "false") << ",";
        js << "\"n_prompt_tokens\":" << r.n_prompt_tokens << ",";
        js << "\"n_generated_tokens\":" << r.n_generated_tokens << ",";
        js << "\"ttft_ms\":" << r.ttft_ms << ",";
        js << "\"prefill_tok_s\":" << r.prefill_tok_s << ",";
        js << "\"decode_tok_s\":" << r.decode_tok_s << ",";
        js << "\"predicted_tok_s\":" << init_res.predicted_tok_s << ",";
        js << "\"predicted_tok_s_burst\":" << init_res.predicted_tok_s_burst << ",";
        js << "\"predicted_hit_rate\":" << init_res.predicted_hit_rate << ",";
        js << "\"hit_rate_ram\":" << r.hit_rate_ram << ",";
        js << "\"cache_hits\":" << r.cache_hits << ",";
        js << "\"cache_misses\":" << r.cache_misses << ",";
        js << "\"bytes_read_ssd\":" << r.bytes_read_ssd << ",";
        js << "\"bytes_read_ssd_per_token\":" << r.bytes_read_ssd_per_token << ",";
        js << "\"prefetch_hits\":" << r.prefetch_hits << ",";
        js << "\"prefetch_submitted\":" << r.prefetch_submitted << ",";
        js << "\"prefetch_wasted_bytes\":" << r.prefetch_wasted_bytes << ",";
        js << "\"eam_prefetch_hits\":" << r.eam_prefetch_hits << ",";
        js << "\"eam_prefetch_submitted\":" << r.eam_prefetch_submitted << ",";
        js << "\"spec_k_mean\":" << r.spec_k_mean << ",";
        js << "\"spec_accept_rate\":" << r.spec_accept_rate << ",";
        js << "\"prefix_cache_hit\":" << (r.prefix_cache_hit ? "true" : "false") << ",";
        js << "\"peak_process_private_bytes\":" << std::max(max_priv.load(), snap_now.process_private_bytes) << ",";
        js << "\"peak_process_working_set_bytes\":" << std::max(max_ws.load(), snap_now.process_working_set_bytes) << ",";
        js << "\"peak_system_used_delta_bytes\":" << peak_sys_used_delta << ",";
        js << "\"peak_file_cache_delta_bytes\":" << peak_cache_delta << ",";
        js << "\"end_system_cache_delta_bytes\":" << end_cache_delta << ",";
        js << "\"peak_standby_delta_bytes\":" << peak_standby_delta << ",";
        js << "\"end_standby_delta_bytes\":" << end_standby_delta << ",";
        js << "\"peak_gpu_used_bytes\":" << std::max(max_gpu.load(), snap_now.gpu_used_bytes) << ",";
        js << "\"pdh_disk_read_bytes_total\":" << static_cast<uint64_t>(pdh_disk_read_bytes_accum.load()) << ",";
        js << "\"ram_governor_total_bytes\":" << init_res.ram_governor_total_bytes << ",";
        js << "\"vram_budget_bytes\":" << init_res.vram_budget_bytes << ",";
        js << "\"vram_governor_total_bytes\":" << init_res.vram_governor_total_bytes << ",";
        js << "\"n_gpu_layers_active\":" << init_res.n_gpu_layers_active << ",";
        js << "\"io_bw_ewma_gbps\":" << r.io_bw_ewma_gbps << ",";
        js << "\"ssd_stream_bytes\":" << init_res.ssd_stream_bytes << ",";
        js << "\"n_cache_slots\":" << init_res.n_cache_slots << ",";
        js << "\"generated_tokens\":[";
        for (size_t t = 0; t < r.generated_tokens.size(); ++t) {
            if (t) js << ",";
            js << r.generated_tokens[t];
        }
        js << "],\"first_step_top_ids\":[";
        for (size_t t = 0; t < r.first_step_top_ids.size(); ++t) {
            if (t) js << ",";
            js << r.first_step_top_ids[t];
        }
        js << "],\"first_step_top_logits\":[";
        for (size_t t = 0; t < r.first_step_top_logits.size(); ++t) {
            if (t) js << ",";
            js << r.first_step_top_logits[t];
        }
        js << "],\"generated_text\":\"" << json_escape(r.generated_text) << "\"}";

        const std::string line = js.str();
        if (ofs.is_open()) {
            ofs << line << "\n";
        }
        if (repeat_count == 1) {
            std::printf("%s\n", line.c_str());
        }
      }
    }

    stop_mem.store(true);
    mem_thread.join();
    return 0;
}
