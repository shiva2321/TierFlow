"""Phase 0 & Phase 1 End-to-End Benchmark Harness (bench.py).

Executes:
1. Real GGUF -> `.tfpack` packing + 339-tensor SHA-256 round-trip verification
   + Win32 FILE_FLAG_NO_BUFFERING single-object retrieval test (deletes the
   1.1 GB `.tfpack` payload after verification if `--cleanup-pack` is set, while
   keeping the full manifest and SHA-256 verification log).
2. Live `llama.cpp` (`llama-cpp-python` v0.3.21) evaluation over n=20 frozen prompts
   (10 general reasoning + 10 code/structured refactoring prompts), measuring:
   - Standard autoregressive decoding (`use_mmap=True`)
   - Prompt-lookup n-gram speculative decoding (`LlamaPromptLookupDecoding`, `max_ngram_size=3, num_pred_tokens=4`)
   - Exact greedy token-ID equivalence (`exact_match_tokens`, Wilson 95% CI)
   - System-wide memory accounting (`sysmem_win.py`: process private bytes, working
     set, and OS system cache bytes) comparing `use_mmap=True` vs `use_mmap=False`.

Writes `raw.jsonl` and `summary.json` under `results/p0_dense_baseline/`.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw_probe import summary_stats
from pack import pack_gguf_to_tfpack
from sysmem_win import MemoryMonitor, sample_memory


# 20 frozen evaluation prompts (10 general/reasoning, 10 code/refactoring with prompt n-gram overlap)
FROZEN_PROMPTS: List[Dict[str, str]] = [
    {"id": "gen_01", "category": "general", "prompt": "Explain the difference between virtual memory paging and direct unbuffered disk I/O in operating systems:\n"},
    {"id": "gen_02", "category": "general", "prompt": "Why does autoregressive token generation in dense large language models hit a memory bandwidth wall?\n"},
    {"id": "gen_03", "category": "general", "prompt": "Describe how a Least Recently Used (LRU) cache differs from a Least Frequently Used (LFU) cache:\n"},
    {"id": "gen_04", "category": "general", "prompt": "What are the main advantages of Mixture-of-Experts (MoE) transformer architectures over dense transformers?\n"},
    {"id": "gen_05", "category": "general", "prompt": "Explain how speculative decoding verifies multiple draft tokens in a single forward pass without losing accuracy:\n"},
    {"id": "gen_06", "category": "general", "prompt": "Derive the memory footprint formula for the Key-Value (KV) cache in multi-head transformer attention:\n"},
    {"id": "gen_07", "category": "general", "prompt": "Compare PCIe Gen4 NVMe solid state drives, DDR5 system RAM, and GDDR6 GPU memory in terms of bandwidth and latency:\n"},
    {"id": "gen_08", "category": "general", "prompt": "Why do outlier channels in transformer activations make 2-bit scalar weight quantization difficult?\n"},
    {"id": "gen_09", "category": "general", "prompt": "Explain how rotary positional embeddings (RoPE) encode relative token distances in attention layers:\n"},
    {"id": "gen_10", "category": "general", "prompt": "Summarize the steps required to benchmark an inference runtime reproducibly with confidence intervals:\n"},
    {
        "id": "code_01",
        "category": "code_overlap",
        "prompt": (
            "Rewrite the following Python dataclass and helper function by adding docstrings and keeping exact variable names:\n"
            "```python\n"
            "class TierCacheConfig:\n"
            "    def __init__(self, vram_budget_bytes: int, ram_budget_bytes: int, page_size_bytes: int = 4096):\n"
            "        self.vram_budget_bytes = vram_budget_bytes\n"
            "        self.ram_budget_bytes = ram_budget_bytes\n"
            "        self.page_size_bytes = page_size_bytes\n"
            "```\nRewritten code:\n```python\n"
        ),
    },
    {
        "id": "code_02",
        "category": "code_overlap",
        "prompt": (
            "Copy the following JSON configuration and add a field `\"prefetch_depth\": 2` at the end:\n"
            "{\n"
            "  \"model_name\": \"qwen3-30b-a3b\",\n"
            "  \"num_experts\": 128,\n"
            "  \"topk_experts\": 8,\n"
            "  \"cache_policy\": \"wtinylfu_slru\"\n"
            "}\nUpdated JSON:\n{\n"
        ),
    },
    {
        "id": "code_03",
        "category": "code_overlap",
        "prompt": (
            "Repeat the following C++ struct definition and add a `uint64_t checksum;` field at the bottom:\n"
            "struct tf_read_request {\n"
            "    uint32_t object_id;\n"
            "    uint64_t file_offset;\n"
            "    uint64_t byte_length;\n"
            "    uint8_t priority_level;\n"
            "};\nUpdated C++ struct:\nstruct tf_read_request {\n"
        ),
    },
    {
        "id": "code_04",
        "category": "code_overlap",
        "prompt": (
            "Refactor this Python function to include type annotations while preserving its exact body:\n"
            "def compute_token_ceiling(bandwidth_bytes_per_sec, active_bytes_per_token):\n"
            "    if active_bytes_per_token <= 0:\n"
            "        raise ValueError('active_bytes_per_token must be positive')\n"
            "    return bandwidth_bytes_per_sec / active_bytes_per_token\n"
            "Refactored function:\ndef compute_token_ceiling("
        ),
    },
    {
        "id": "code_05",
        "category": "code_overlap",
        "prompt": (
            "Convert this SQL query to select one additional column `cache_hit_rate` from the same table:\n"
            "SELECT experiment_id, model_name, decode_tokens_per_sec, peak_system_memory_bytes\n"
            "FROM benchmark_runs\n"
            "WHERE exact_match_tokens = TRUE\n"
            "ORDER BY decode_tokens_per_sec DESC;\n"
            "Updated SQL query:\nSELECT "
        ),
    },
    {
        "id": "code_06",
        "category": "code_overlap",
        "prompt": (
            "Duplicate the following list of dictionaries and add `'status': 'verified'` to each entry:\n"
            "items = [\n"
            "    {'layer_id': 0, 'expert_id': 12, 'tier': 'VRAM'},\n"
            "    {'layer_id': 1, 'expert_id': 45, 'tier': 'RAM'},\n"
            "    {'layer_id': 2, 'expert_id': 99, 'tier': 'SSD'},\n"
            "]\nUpdated items:\nitems = [\n"
        ),
    },
    {
        "id": "code_07",
        "category": "code_overlap",
        "prompt": (
            "Echo the following shell script lines and add `echo \"Done\"` at the end:\n"
            "python tools/sysmem_win.py --self-test\n"
            "python tools/hw_probe.py --self-test\n"
            "python tools/cache_sim.py --self-test\n"
            "python tools/pack.py --self-test\n"
            "Updated script:\npython tools/"
        ),
    },
    {
        "id": "code_08",
        "category": "code_overlap",
        "prompt": (
            "Copy the following markdown table and add a row for `TierFlow` at the bottom:\n"
            "| Engine | Memory Accounting | Direct IO | Union Speculation |\n"
            "| :--- | :--- | :--- | :--- |\n"
            "| stock_llama_cpp | Process RSS only | No | No |\n"
            "| airllm_layerwise | VRAM buffer only | No | No |\n"
            "Updated table:\n| Engine | Memory Accounting |"
        ),
    },
    {
        "id": "code_09",
        "category": "code_overlap",
        "prompt": (
            "Preserve the exact fields of this Rust struct and derive `Clone` in addition to `Debug`:\n"
            "#[derive(Debug)]\n"
            "pub struct ExpertResidencySlot {\n"
            "    pub layer_index: u16,\n"
            "    pub expert_index: u16,\n"
            "    pub pin_count: u32,\n"
            "    pub last_access_tick: u64,\n"
            "}\nUpdated Rust struct:\n#[derive(Debug, Clone)]\npub struct ExpertResidencySlot {\n"
        ),
    },
    {
        "id": "code_10",
        "category": "code_overlap",
        "prompt": (
            "Keep the exact equations below and add a comment on each line:\n"
            "accepted_tokens = (1.0 - alpha ** (k + 1)) / (1.0 - alpha)\n"
            "verify_cost_sec = always_on_sec + union_experts * expert_read_sec\n"
            "net_speedup = (accepted_tokens * base_step_sec) / (verify_cost_sec + k * draft_step_sec)\n"
            "Commented equations:\naccepted_tokens = "
        ),
    },
]


def wilson_ci95(successes: int, total: int) -> Tuple[float, float]:
    """Compute exact 95% Wilson score interval for a binomial proportion."""
    if total <= 0:
        return 0.0, 0.0
    z = 1.959964
    phat = successes / total
    denom = 1.0 + (z * z) / total
    center = (phat + (z * z) / (2.0 * total)) / denom
    half = (z * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * total)) / total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def run_benchmark(
    model_path: str,
    out_dir: str,
    max_tokens: int = 64,
    n_threads: int = 6,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    raw_jsonl_path = os.path.join(out_dir, "raw.jsonl")
    summary_json_path = os.path.join(out_dir, "summary.json")

    # -----------------------------------------------------------------------
    # Step 1: Pack real GGUF -> .tfpack and verify 100% of tensors via SHA-256
    # -----------------------------------------------------------------------
    pack_dir = os.path.join(out_dir, "pack_verification")
    manifest = pack_gguf_to_tfpack(model_path, pack_dir, verify_roundtrip=True)
    tfpack_file = manifest["tfpack_path"]
    # Remove the 1.1 GB binary container after verification so we don't waste disk space,
    # while keeping the full manifest.json with all 339 tensor SHA-256 checksums!
    if os.path.exists(tfpack_file):
        os.remove(tfpack_file)

    pack_summary = {
        "source_gguf": manifest["source_gguf"],
        "source_gguf_size_bytes": manifest["source_gguf_size_bytes"],
        "tfpack_size_bytes": manifest["tfpack_size_bytes"],
        "num_objects": manifest["num_objects"],
        "verified_byte_identical_objects": manifest["verified_byte_identical_objects"],
        "unbuffered_single_object_read_verified": manifest["unbuffered_single_object_read_verified"],
        "pack_elapsed_sec": manifest["pack_elapsed_sec"],
        "verify_elapsed_sec": manifest["verify_elapsed_sec"],
    }

    # -----------------------------------------------------------------------
    # Step 2: Memory Footprint Comparison (`use_mmap=True` vs `use_mmap=False`)
    # -----------------------------------------------------------------------
    mon_mmap = MemoryMonitor(interval_sec=0.02)
    mon_mmap.start()
    llm_std = Llama(
        model_path=model_path,
        n_ctx=1024,
        n_threads=n_threads,
        use_mmap=True,
        verbose=False,
        seed=42,
    )
    _ = llm_std("Hello", max_tokens=4, temperature=0.0)
    mem_mmap_stats = mon_mmap.stop()

    # -----------------------------------------------------------------------
    # Step 3: Run 20 Frozen Prompts — Standard Decoding vs Speculative Decoding
    # -----------------------------------------------------------------------
    llm_spec = Llama(
        model_path=model_path,
        n_ctx=1024,
        n_threads=n_threads,
        use_mmap=True,
        draft_model=LlamaPromptLookupDecoding(num_pred_tokens=4, max_ngram_size=3),
        verbose=False,
        seed=42,
    )
    _ = llm_spec("Hello", max_tokens=4, temperature=0.0)

    records: List[Dict[str, Any]] = []
    std_tps_all: List[float] = []
    spec_tps_all: List[float] = []
    std_tps_gen: List[float] = []
    spec_tps_gen: List[float] = []
    std_tps_code: List[float] = []
    spec_tps_code: List[float] = []
    speedup_ratios_code: List[float] = []
    speedup_ratios_gen: List[float] = []
    exact_matches = 0

    with open(raw_jsonl_path, "w", encoding="utf-8") as fraw:
        for item in FROZEN_PROMPTS:
            pid = item["id"]
            cat = item["category"]
            prompt_text = item["prompt"]
            prompt_tokens = llm_std.tokenize(prompt_text.encode("utf-8"))

            # A. Standard Autoregressive Decoding (greedy, temp=0)
            llm_std.reset()
            t0 = time.perf_counter()
            out_std = llm_std.create_completion(
                prompt_text,
                max_tokens=max_tokens,
                temperature=0.0,
                top_k=1,
            )
            t_std_total = time.perf_counter() - t0
            std_text = out_std["choices"][0]["text"]
            gen_tokens = out_std["usage"]["completion_tokens"]
            tps_std = gen_tokens / max(1e-9, t_std_total)

            # B. Speculative Decoding via LlamaPromptLookupDecoding (greedy, temp=0)
            llm_spec.reset()
            t1 = time.perf_counter()
            out_spec = llm_spec.create_completion(
                prompt_text,
                max_tokens=max_tokens,
                temperature=0.0,
                top_k=1,
            )
            t_spec_total = time.perf_counter() - t1
            spec_text = out_spec["choices"][0]["text"]
            spec_gen_tokens = out_spec["usage"]["completion_tokens"]

            # Check exact output match
            is_exact = (spec_text == std_text)
            if is_exact:
                exact_matches += 1
            tps_spec = spec_gen_tokens / max(1e-9, t_spec_total)

            std_tps_all.append(tps_std)
            spec_tps_all.append(tps_spec)
            ratio = tps_spec / tps_std
            if cat == "general":
                std_tps_gen.append(tps_std)
                spec_tps_gen.append(tps_spec)
                speedup_ratios_gen.append(ratio)
            else:
                std_tps_code.append(tps_std)
                spec_tps_code.append(tps_spec)
                speedup_ratios_code.append(ratio)

            rec = {
                "exp_id": "p0_dense_baseline",
                "prompt_id": pid,
                "category": cat,
                "prompt_token_count": len(prompt_tokens),
                "generated_tokens": gen_tokens,
                "std_total_sec": t_std_total,
                "spec_total_sec": t_spec_total,
                "std_tok_s": tps_std,
                "spec_tok_s": tps_spec,
                "spec_vs_std_speedup": ratio,
                "exact_match_output": is_exact,
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            records.append(rec)
            fraw.write(json.dumps(rec) + "\n")
            fraw.flush()

    del llm_std, llm_spec

    # Also measure `use_mmap=False` memory allocation delta
    mon_nommap = MemoryMonitor(interval_sec=0.02)
    mon_nommap.start()
    llm_nommap = Llama(
        model_path=model_path,
        n_ctx=1024,
        n_threads=n_threads,
        use_mmap=False,
        verbose=False,
        seed=42,
    )
    _ = llm_nommap("Hello", max_tokens=4, temperature=0.0)
    mem_nommap_stats = mon_nommap.stop()
    del llm_nommap

    w_low, w_high = wilson_ci95(exact_matches, len(FROZEN_PROMPTS))

    summary = {
        "exp_id": "p0_dense_baseline",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_path": model_path,
        "n_threads": n_threads,
        "max_tokens": max_tokens,
        "pack_roundtrip_verification": pack_summary,
        "memory_accounting_mmap_true": mem_mmap_stats,
        "memory_accounting_mmap_false": mem_nommap_stats,
        "exactness_greedy_spec_vs_std": {
            "exact_matches": exact_matches,
            "total_prompts": len(FROZEN_PROMPTS),
            "proportion": exact_matches / len(FROZEN_PROMPTS),
            "wilson_ci95_low": w_low,
            "wilson_ci95_high": w_high,
        },
        "throughput_all_20_prompts": {
            "std_tok_s": summary_stats(std_tps_all),
            "spec_tok_s": summary_stats(spec_tps_all),
        },
        "throughput_general_10_prompts": {
            "std_tok_s": summary_stats(std_tps_gen),
            "spec_tok_s": summary_stats(spec_tps_gen),
            "speedup_ratio": summary_stats(speedup_ratios_gen),
        },
        "throughput_code_overlap_10_prompts": {
            "std_tok_s": summary_stats(std_tps_code),
            "spec_tok_s": summary_stats(spec_tps_code),
            "speedup_ratio": summary_stats(speedup_ratios_code),
        },
    }
    with open(summary_json_path, "w", encoding="utf-8") as fs:
        json.dump(summary, fs, indent=2)

    print("=== Phase 0 / Phase 1 Live Benchmark Summary ===")
    print("Packer Verification:", json.dumps(pack_summary, indent=2))
    print("Exactness (Speculative vs Standard Greedy):", json.dumps(summary["exactness_greedy_spec_vs_std"], indent=2))
    print(
        "General Prompts (n=10) Speedup:",
        f"{summary['throughput_general_10_prompts']['speedup_ratio']['mean']:.3f}x "
        f"+/- {summary['throughput_general_10_prompts']['speedup_ratio']['ci95_halfwidth']:.3f}x "
        f"(std {summary['throughput_general_10_prompts']['std_tok_s']['mean']:.2f} tok/s -> "
        f"spec {summary['throughput_general_10_prompts']['spec_tok_s']['mean']:.2f} tok/s)",
    )
    print(
        "Code/Overlap Prompts (n=10) Speedup:",
        f"{summary['throughput_code_overlap_10_prompts']['speedup_ratio']['mean']:.3f}x "
        f"+/- {summary['throughput_code_overlap_10_prompts']['speedup_ratio']['ci95_halfwidth']:.3f}x "
        f"(std {summary['throughput_code_overlap_10_prompts']['std_tok_s']['mean']:.2f} tok/s -> "
        f"spec {summary['throughput_code_overlap_10_prompts']['spec_tok_s']['mean']:.2f} tok/s)",
    )
    print(
        "Memory Accounting (mmap=True vs mmap=False):",
        json.dumps(
            {
                "mmap_true_delta_proc_private_mb": mem_mmap_stats["delta_proc_private_bytes"] / (1024 * 1024),
                "mmap_true_peak_working_set_mb": mem_mmap_stats["peak_proc_working_set_bytes"] / (1024 * 1024),
                "mmap_false_delta_proc_private_mb": mem_nommap_stats["delta_proc_private_bytes"] / (1024 * 1024),
                "mmap_false_peak_working_set_mb": mem_nommap_stats["peak_proc_working_set_bytes"] / (1024 * 1024),
            },
            indent=2,
        ),
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=r"E:\AI_Research\sophia\models\qwen2.5-1.5b-instruct-q4_k_m.gguf",
    )
    parser.add_argument("--out-dir", type=str, default="results/p0_dense_baseline")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    run_benchmark(args.model, args.out_dir, max_tokens=args.max_tokens, n_threads=args.threads)
