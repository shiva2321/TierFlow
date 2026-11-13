"""Reproducible C++20 TierFlow Benchmark & Experiment Runner (Phases 3-8).

Drives `runtime/build/tf_runner.exe` sequentially under frozen PREREG protocols,
records raw JSONL records, computes 95% confidence intervals / Wilson intervals /
logit diffs / Pearson chi-square tests, and writes `summary.json` per experiment.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import math
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import FROZEN_PROMPTS, wilson_ci95
from hw_probe import summary_stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TF_RUNNER = os.path.join(REPO_ROOT, "runtime", "build", "tf_runner.exe")
GGUF_PATH = r"E:\AI_Research\sophia\models\qwen2.5-1.5b-instruct-q4_k_m.gguf"
MANIFEST_PATH = os.path.join(REPO_ROOT, "packed_models", "qwen2.5-1.5b-instruct-q4_k_m.manifest.json")
PROMPTS_20_PATH = os.path.join(REPO_ROOT, "results", "prompts_20.jsonl")


def git_info() -> Dict[str, Any]:
    def _run(cwd: str, args: List[str]) -> str:
        try:
            r = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True, timeout=5)
            return r.stdout.strip()
        except Exception:
            return ""
    llama_dir = os.path.join(REPO_ROOT, "third_party", "llama.cpp")
    return {
        "repo_has_commit": bool(_run(REPO_ROOT, ["rev-parse", "HEAD"])),
        "llama_cpp_commit": _run(llama_dir, ["rev-parse", "HEAD"]),
        "llama_cpp_dirty": bool(_run(llama_dir, ["status", "--short"])),
    }


def export_prompts() -> None:
    os.makedirs(os.path.dirname(PROMPTS_20_PATH), exist_ok=True)
    with open(PROMPTS_20_PATH, "w", encoding="utf-8", newline="\n") as f:
        for item in FROZEN_PROMPTS:
            rec = {
                "id": item["id"],
                "split": item["category"],
                "prompt": item["prompt"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Exported {len(FROZEN_PROMPTS)} prompts to {PROMPTS_20_PATH}")


def run_tf(
    args: List[str],
    out_jsonl: str,
) -> Tuple[str, List[Dict[str, Any]], float]:
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    cmd = [TF_RUNNER] + args + ["--out-jsonl", out_jsonl]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"tf_runner failed (rc={proc.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    rows: List[Dict[str, Any]] = []
    with open(out_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    plan_line = ""
    for line in proc.stdout.splitlines():
        if line.startswith("PLAN "):
            plan_line = line
            break
    return plan_line, rows, elapsed


def parse_plan_line(plan_line: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not plan_line.startswith("PLAN "):
        return out
    for tok in plan_line[5:].split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                out[k] = int(v) if v.isdigit() else float(v)
            except ValueError:
                out[k] = v
    return out


def compare_vs_ref(rows: List[Dict[str, Any]], ref_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    exact = 0
    total = min(len(rows), len(ref_rows))
    first_divs: List[Optional[int]] = []
    max_logit_diffs: List[float] = []
    top1_agree = 0
    for r, ref in zip(rows, ref_rows):
        gt = r["generated_tokens"]
        rt = ref["generated_tokens"]
        if gt == rt:
            exact += 1
            first_divs.append(None)
        else:
            div_idx = next((i for i, (a, b) in enumerate(zip(gt, rt)) if a != b), min(len(gt), len(rt)))
            first_divs.append(div_idx)

        r_ids = r.get("first_step_top_ids", [])
        r_log = r.get("first_step_top_logits", [])
        ref_ids = ref.get("first_step_top_ids", [])
        ref_log = ref.get("first_step_top_logits", [])
        if r_ids and ref_ids and r_ids[0] == ref_ids[0]:
            top1_agree += 1
        ref_map = dict(zip(ref_ids, ref_log))
        diffs = [abs(lv - ref_map[tid]) for tid, lv in zip(r_ids, r_log) if tid in ref_map]
        if diffs:
            max_logit_diffs.append(max(diffs))

    w_lo, w_hi = wilson_ci95(exact, total)
    return {
        "exact_token_matches": exact,
        "total_prompts": total,
        "proportion": exact / total if total else 0.0,
        "wilson_ci95": [w_lo, w_hi],
        "top1_first_token_agree": top1_agree,
        "first_divergence_indices": first_divs,
        "max_shared_top16_logit_diff": max(max_logit_diffs) if max_logit_diffs else None,
        "mean_shared_top16_logit_diff": (sum(max_logit_diffs) / len(max_logit_diffs)) if max_logit_diffs else None,
    }


def summarize_condition(name: str, rows: List[Dict[str, Any]], plan_line: str, cmd_args: List[str], elapsed_s: float) -> Dict[str, Any]:
    decode_s = [float(r["decode_tok_s"]) for r in rows]
    ttft_ms = [float(r["ttft_ms"]) for r in rows]
    prefill_s = [float(r["prefill_tok_s"]) for r in rows]
    hit_rates = [float(r["hit_rate_ram"]) for r in rows]
    ssd_per_tok = [float(r["bytes_read_ssd_per_token"]) for r in rows]
    spec_k = [float(r["spec_k_mean"]) for r in rows]
    spec_acc = [float(r["spec_accept_rate"]) for r in rows]

    pref_sub = sum(int(r["prefetch_submitted"]) for r in rows)
    pref_hits = sum(int(r["prefetch_hits"]) for r in rows)
    pref_waste = sum(int(r["prefetch_wasted_bytes"]) for r in rows)
    ssd_total = sum(int(r["bytes_read_ssd"]) for r in rows)
    last = rows[-1] if rows else {}
    return {
        "condition": name,
        "cmd": " ".join([TF_RUNNER] + cmd_args),
        "wall_elapsed_sec": elapsed_s,
        "plan": parse_plan_line(plan_line),
        "n_prompts": len(rows),
        "decode_tok_s": summary_stats(decode_s) if len(decode_s) >= 2 else {"mean": decode_s[0], "n": len(decode_s)},
        "ttft_ms": summary_stats(ttft_ms) if len(ttft_ms) >= 2 else {"mean": ttft_ms[0], "n": len(ttft_ms)},
        "prefill_tok_s": summary_stats(prefill_s) if len(prefill_s) >= 2 else {"mean": prefill_s[0], "n": len(prefill_s)},
        "hit_rate_ram": summary_stats(hit_rates) if len(hit_rates) >= 2 else {"mean": hit_rates[0], "n": len(hit_rates)},
        "bytes_read_ssd_per_token": summary_stats(ssd_per_tok) if len(ssd_per_tok) >= 2 else {"mean": ssd_per_tok[0], "n": len(ssd_per_tok)},
        "spec_k_mean": summary_stats(spec_k) if len(spec_k) >= 2 else {"mean": spec_k[0], "n": len(spec_k)},
        "spec_accept_rate": summary_stats(spec_acc) if len(spec_acc) >= 2 else {"mean": spec_acc[0], "n": len(spec_acc)},
        "prefetch_submitted_total": pref_sub,
        "prefetch_hits_total": pref_hits,
        "prefetch_accuracy": (pref_hits / pref_sub) if pref_sub > 0 else None,
        "prefetch_wasted_bytes_total": pref_waste,
        "bytes_read_ssd_total": ssd_total,
        "pdh_disk_read_bytes_total": max((int(r.get("pdh_disk_read_bytes_total", 0)) for r in rows), default=0),
        "peak_process_private_bytes": max((int(r["peak_process_private_bytes"]) for r in rows), default=0),
        "peak_process_working_set_bytes": max((int(r.get("peak_process_working_set_bytes", 0)) for r in rows), default=0),
        "peak_system_used_delta_bytes": max((int(r["peak_system_used_delta_bytes"]) for r in rows), default=0),
        "peak_file_cache_delta_bytes": max((int(r["peak_file_cache_delta_bytes"]) for r in rows), default=0),
        "end_system_cache_delta_bytes": int(last.get("end_system_cache_delta_bytes", 0)),
        "peak_standby_delta_bytes": max((int(r.get("peak_standby_delta_bytes", 0)) for r in rows), default=0),
        "end_standby_delta_bytes": int(last.get("end_standby_delta_bytes", 0)),
        "ram_governor_total_bytes": int(last.get("ram_governor_total_bytes", 0)),
        "vram_budget_bytes": int(last.get("vram_budget_bytes", 0)),
        "vram_governor_total_bytes": int(last.get("vram_governor_total_bytes", 0)),
        "peak_gpu_used_bytes": max((int(r.get("peak_gpu_used_bytes", 0)) for r in rows), default=0),
        "n_gpu_layers_active": int(last.get("n_gpu_layers_active", 0)),
        "io_bw_ewma_gbps": float(last.get("io_bw_ewma_gbps", 0.0)),
        "ssd_stream_bytes": int(last.get("ssd_stream_bytes", 0)),
        "n_cache_slots": int(last.get("n_cache_slots", 0)),
    }


def write_summary(out_dir: str, payload: Dict[str, Any]) -> None:
    payload["timestamp_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    payload["git"] = git_info()
    path = os.path.join(out_dir, "summary.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Phase 3 calibration: Stock run-to-run self-consistency & repack check
# ---------------------------------------------------------------------------
def run_selfcheck() -> None:
    out_dir = os.path.join(REPO_ROOT, "results", "p3_selfcheck")
    os.makedirs(out_dir, exist_ok=True)
    base_args = [
        "--gguf", GGUF_PATH,
        "--n-ctx", "2048",
        "--n-batch", "512",
        "--n-threads", "6",
        "--n-predict", "64",
        "--prompts-file", PROMPTS_20_PATH,
    ]
    _, r_mmap_a, dt_a = run_tf(["--mode", "stock_mmap", "--no-repack", "0"] + base_args, os.path.join(out_dir, "stock_mmap_a.jsonl"))
    _, r_mmap_b, dt_b = run_tf(["--mode", "stock_mmap", "--no-repack", "0"] + base_args, os.path.join(out_dir, "stock_mmap_b.jsonl"))
    _, r_norep_a, dt_c = run_tf(["--mode", "stock_mmap", "--no-repack", "1"] + base_args, os.path.join(out_dir, "stock_mmap_norepack_a.jsonl"))
    _, r_norep_b, dt_d = run_tf(["--mode", "stock_mmap", "--no-repack", "1"] + base_args, os.path.join(out_dir, "stock_mmap_norepack_b.jsonl"))

    res = {
        "exp_id": "p3_selfcheck",
        "stock_mmap_repack_a_vs_b": compare_vs_ref(r_mmap_b, r_mmap_a),
        "stock_mmap_norepack_a_vs_b": compare_vs_ref(r_norep_b, r_norep_a),
        "stock_mmap_norepack_vs_repack": compare_vs_ref(r_norep_a, r_mmap_a),
        "elapsed_sec": {"repack_a": dt_a, "repack_b": dt_b, "norepack_a": dt_c, "norepack_b": dt_d},
    }
    write_summary(out_dir, res)


# ---------------------------------------------------------------------------
# Phase 3: End-to-End Streaming & Exactness
# ---------------------------------------------------------------------------
def run_p3() -> None:
    out_dir = os.path.join(REPO_ROOT, "results", "p3_e2e_stream")
    common = [
        "--gguf", GGUF_PATH,
        "--n-ctx", "2048",
        "--n-batch", "512",
        "--n-threads", "6",
        "--n-predict", "64",
        "--prompts-file", PROMPTS_20_PATH,
    ]
    configs = [
        ("stock_mmap_repack", ["--mode", "stock_mmap", "--no-repack", "0"] + common),
        ("stock_mmap_norepack", ["--mode", "stock_mmap", "--no-repack", "1"] + common),
        ("stock_dio_norepack", ["--mode", "stock_dio", "--no-repack", "1"] + common),
        ("tierflow_ram1400", [
            "--mode", "tierflow", "--manifest", MANIFEST_PATH,
            "--ram-budget-mb", "1400",
        ] + common),
        ("tierflow_stream600", [
            "--mode", "tierflow", "--manifest", MANIFEST_PATH,
            "--ram-budget-mb", "600", "--allow-slow-stream", "1",
            "--cache-slots", "12", "--prefetch", "lookahead", "--lookahead-depth", "1",
        ] + common),
    ]
    runs: Dict[str, Tuple[str, List[Dict[str, Any]], float, List[str]]] = {}
    summaries: Dict[str, Any] = {}
    for name, args in configs:
        pl, rows, dt = run_tf(args, os.path.join(out_dir, f"{name}.jsonl"))
        runs[name] = (pl, rows, dt, args)
        summaries[name] = summarize_condition(name, rows, pl, args, dt)

    ref_norep = runs["stock_mmap_norepack"][1]
    ref_repack = runs["stock_mmap_repack"][1]
    exactness = {
        "tierflow_ram1400_vs_stock_norepack": compare_vs_ref(runs["tierflow_ram1400"][1], ref_norep),
        "tierflow_stream600_vs_stock_norepack": compare_vs_ref(runs["tierflow_stream600"][1], ref_norep),
        "tierflow_stream600_vs_tierflow_ram1400": compare_vs_ref(runs["tierflow_stream600"][1], runs["tierflow_ram1400"][1]),
        "tierflow_stream600_vs_stock_repack": compare_vs_ref(runs["tierflow_stream600"][1], ref_repack),
        "stock_dio_norepack_vs_stock_norepack": compare_vs_ref(runs["stock_dio_norepack"][1], ref_norep),
    }
    write_summary(out_dir, {
        "exp_id": "p3_e2e_stream",
        "conditions": summaries,
        "exactness": exactness,
    })


# ---------------------------------------------------------------------------
# Phase 4: Probe & Planner Validation Across 3 Budgets
# ---------------------------------------------------------------------------
def run_p4(exp_id: str = "p4_planner") -> None:
    out_dir = os.path.join(REPO_ROOT, "results", exp_id)
    common = [
        "--mode", "tierflow",
        "--gguf", GGUF_PATH,
        "--manifest", MANIFEST_PATH,
        "--n-ctx", "2048",
        "--n-batch", "512",
        "--n-threads", "6",
        "--n-predict", "64",
        "--allow-slow-stream", "1",
        "--prefetch", "lookahead",
        "--lookahead-depth", "1",
        "--prompts-file", PROMPTS_20_PATH,
    ]
    budgets_mb = [560, 850, 1400]
    results_by_budget: Dict[str, Any] = {}
    for b_mb in budgets_mb:
        name = f"budget_{b_mb}mb"
        args = ["--ram-budget-mb", str(b_mb), "--cache-slots", "12"] + common
        pl, rows, dt = run_tf(args, os.path.join(out_dir, f"{name}.jsonl"))
        s = summarize_condition(name, rows, pl, args, dt)
        meas = s["decode_tok_s"]["mean"]
        pred_sust = float(s["plan"].get("pred_tok_s", 0.0))
        pred_burst = float(s["plan"].get("pred_tok_s_burst", 0.0))
        s["prediction_error_sustained_rel"] = abs(pred_sust - meas) / meas if meas > 0 else None
        s["prediction_error_burst_rel"] = abs(pred_burst - meas) / meas if meas > 0 else None
        results_by_budget[name] = s
    write_summary(out_dir, {
        "exp_id": exp_id,
        "budgets_mb": budgets_mb,
        "conditions": results_by_budget,
    })


# ---------------------------------------------------------------------------
# Phase 5: Prefetch, Cache Policy & Online Re-Pinning Ablations
# ---------------------------------------------------------------------------
def run_p5() -> None:
    out_dir = os.path.join(REPO_ROOT, "results", "p5_prefetch_repin")
    os.makedirs(out_dir, exist_ok=True)
    common = [
        "--mode", "tierflow",
        "--gguf", GGUF_PATH,
        "--manifest", MANIFEST_PATH,
        "--ram-budget-mb", "600",
        "--allow-slow-stream", "1",
        "--cache-slots", "12",
        "--n-ctx", "2048",
        "--n-batch", "512",
        "--n-threads", "6",
        "--n-predict", "16",
        "--prompts-file", PROMPTS_20_PATH,
    ]
    configs = [
        ("lookahead_d1_wtinylfu", ["--cache-policy", "wtinylfu", "--prefetch", "lookahead", "--lookahead-depth", "1", "--repin", "0"] + common),
        ("lookahead_d2_wtinylfu", ["--cache-policy", "wtinylfu", "--prefetch", "lookahead", "--lookahead-depth", "2", "--repin", "0"] + common),
        ("lookahead_d1_repin",    ["--cache-policy", "wtinylfu", "--prefetch", "lookahead", "--lookahead-depth", "1", "--repin", "1"] + common),
        ("prefetch_off_wtinylfu", ["--cache-policy", "wtinylfu", "--prefetch", "off", "--repin", "0"] + common),
        ("prefetch_shuffled_null", ["--cache-policy", "wtinylfu", "--prefetch", "shuffled", "--lookahead-depth", "1", "--repin", "0"] + common),
        ("lookahead_d1_lru",      ["--cache-policy", "lru", "--prefetch", "lookahead", "--lookahead-depth", "1", "--repin", "0"] + common),
    ]
    cooldown_sec = 60
    runs: Dict[str, List[Dict[str, Any]]] = {}
    summaries: Dict[str, Any] = {}
    for idx, (name, args) in enumerate(configs):
        if idx > 0:
            time.sleep(cooldown_sec)
        pl, rows, dt = run_tf(args, os.path.join(out_dir, f"{name}.jsonl"))
        runs[name] = rows
        summaries[name] = summarize_condition(name, rows, pl, args, dt)

    ref = runs["lookahead_d1_wtinylfu"]
    exactness = {k: compare_vs_ref(v, ref) for k, v in runs.items() if k != "lookahead_d1_wtinylfu"}
    write_summary(out_dir, {
        "exp_id": "p5_prefetch_repin",
        "cooldown_sec_between_arms": cooldown_sec,
        "conditions": summaries,
        "exactness_vs_lookahead_d1": exactness,
    })


# ---------------------------------------------------------------------------
# Phase 6: Union-Aware Speculative Decoding (Greedy + 2000-Sample Chi-Square)
# ---------------------------------------------------------------------------
def chi_square_two_sample(counts_a: collections.Counter, counts_b: collections.Counter, min_expected: float = 5.0) -> Dict[str, Any]:
    # Group all categories with combined count < 2*min_expected into an "__OTHER__" bucket
    all_keys = set(counts_a.keys()) | set(counts_b.keys())
    na = sum(counts_a.values())
    nb = sum(counts_b.values())
    if na == 0 or nb == 0:
        return {"error": "empty counts"}
    kept_keys = [k for k in sorted(all_keys, key=lambda x: counts_a[x] + counts_b[x], reverse=True)
                 if (counts_a[k] + counts_b[k]) >= 2.0 * min_expected]
    kept_set = set(kept_keys)
    obs_a = [counts_a[k] for k in kept_keys]
    obs_b = [counts_b[k] for k in kept_keys]
    tail_a = sum(v for k, v in counts_a.items() if k not in kept_set)
    tail_b = sum(v for k, v in counts_b.items() if k not in kept_set)
    if tail_a + tail_b > 0:
        obs_a.append(tail_a)
        obs_b.append(tail_b)

    chi2 = 0.0
    df = max(1, len(obs_a) - 1)
    for a, b in zip(obs_a, obs_b):
        tot = a + b
        if tot == 0:
            continue
        ea = tot * (na / (na + nb))
        eb = tot * (nb / (na + nb))
        if ea > 0:
            chi2 += ((a - ea) ** 2) / ea
        if eb > 0:
            chi2 += ((b - eb) ** 2) / eb

    p_val: Optional[float] = None
    try:
        from scipy.stats import chi2 as sp_chi2
        p_val = float(sp_chi2.sf(chi2, df))
    except Exception:
        p_val = None

    return {
        "n_a": na,
        "n_b": nb,
        "num_bins": len(obs_a),
        "df": df,
        "chi2_stat": chi2,
        "p_value": p_val,
        "pass_alpha_0_01": (p_val >= 0.01) if p_val is not None else None,
        "counts_a": obs_a,
        "counts_b": obs_b,
    }


def run_p6() -> None:
    out_dir = os.path.join(REPO_ROOT, "results", "p6_spec")
    os.makedirs(out_dir, exist_ok=True)
    # Part A-RAM: Greedy exactness & speed across 20 prompts x 128 tokens at RAM (1400MB)
    # Part A-Stream: Greedy exactness & speed across 20 prompts x 32 tokens at SSD-Stream (600MB) with NVMe thermal cooldown
    ram_common = [
        "--mode", "tierflow",
        "--gguf", GGUF_PATH,
        "--manifest", MANIFEST_PATH,
        "--n-ctx", "2048",
        "--n-batch", "512",
        "--n-threads", "6",
        "--n-predict", "128",
        "--allow-slow-stream", "1",
        "--prompts-file", PROMPTS_20_PATH,
    ]
    stream_common = [
        "--mode", "tierflow",
        "--gguf", GGUF_PATH,
        "--manifest", MANIFEST_PATH,
        "--n-ctx", "2048",
        "--n-batch", "512",
        "--n-threads", "6",
        "--n-predict", "32",
        "--allow-slow-stream", "1",
        "--prompts-file", PROMPTS_20_PATH,
    ]
    configs = [
        ("ram1400_spec_off",          False, ["--ram-budget-mb", "1400", "--spec", "off"] + ram_common),
        ("ram1400_spec_adaptive",     False, ["--ram-budget-mb", "1400", "--spec", "adaptive"] + ram_common),
        ("ram1400_spec_fixed4",       False, ["--ram-budget-mb", "1400", "--spec", "fixed4"] + ram_common),
        ("stream600_spec_off",        True,  ["--ram-budget-mb", "600", "--cache-slots", "12", "--prefetch", "lookahead", "--spec", "off"] + stream_common),
        ("stream600_spec_adaptive",   True,  ["--ram-budget-mb", "600", "--cache-slots", "12", "--prefetch", "lookahead", "--spec", "adaptive"] + stream_common),
        ("stream600_spec_fixed4",     True,  ["--ram-budget-mb", "600", "--cache-slots", "12", "--prefetch", "lookahead", "--spec", "fixed4"] + stream_common),
        ("stream600_spec_naive_nounion", True, ["--ram-budget-mb", "600", "--cache-slots", "12", "--prefetch", "lookahead", "--spec", "naive_no_union"] + stream_common),
    ]
    cooldown_sec = 60
    runs: Dict[str, List[Dict[str, Any]]] = {}
    summaries: Dict[str, Any] = {}
    stream_idx = 0
    for name, is_stream, args in configs:
        if is_stream and stream_idx > 0:
            time.sleep(cooldown_sec)
        if is_stream:
            stream_idx += 1
        pl, rows, dt = run_tf(args, os.path.join(out_dir, f"{name}.jsonl"))
        runs[name] = rows
        s = summarize_condition(name, rows, pl, args, dt)
        s["by_split"] = {}
        for sp in ("general", "code_overlap"):
            sub = [r for r in rows if r.get("split") == sp]
            if sub:
                s["by_split"][sp] = {
                    "n": len(sub),
                    "decode_tok_s": summary_stats([float(r["decode_tok_s"]) for r in sub]),
                    "spec_k_mean": summary_stats([float(r["spec_k_mean"]) for r in sub]),
                    "spec_accept_rate": summary_stats([float(r["spec_accept_rate"]) for r in sub]),
                }
        summaries[name] = s

    exactness = {
        "ram1400_adaptive_vs_off": compare_vs_ref(runs["ram1400_spec_adaptive"], runs["ram1400_spec_off"]),
        "ram1400_fixed4_vs_off": compare_vs_ref(runs["ram1400_spec_fixed4"], runs["ram1400_spec_off"]),
        "stream600_adaptive_vs_off": compare_vs_ref(runs["stream600_spec_adaptive"], runs["stream600_spec_off"]),
        "stream600_fixed4_vs_off": compare_vs_ref(runs["stream600_spec_fixed4"], runs["stream600_spec_off"]),
        "stream600_naive_vs_off": compare_vs_ref(runs["stream600_spec_naive_nounion"], runs["stream600_spec_off"]),
    }

    # Part B: 2,000-sample stochastic two-sample chi-square distribution test (temp=0.8, 4-token draw on code_01)
    # Using code_01 with temperature=0.8 exercises real prompt-lookup acceptance + rejection sampling
    sample_prompt_path = os.path.join(out_dir, "stochastic_prompt.jsonl")
    with open(sample_prompt_path, "w", encoding="utf-8", newline="\n") as f:
        code01 = next(x for x in FROZEN_PROMPTS if x["id"] == "code_01")
        f.write(json.dumps({"id": code01["id"], "split": "spec_stochastic", "prompt": code01["prompt"]}) + "\n")

    stoch_common = [
        "--mode", "tierflow",
        "--gguf", GGUF_PATH,
        "--manifest", MANIFEST_PATH,
        "--ram-budget-mb", "1400",
        "--n-ctx", "1024",
        "--n-batch", "512",
        "--n-threads", "6",
        "--n-predict", "4",
        "--temp", "0.8",
        "--repeat", "2000",
        "--prompts-file", sample_prompt_path,
    ]
    _, s_off, dt_soff = run_tf(["--spec", "off", "--seed", "10000"] + stoch_common, os.path.join(out_dir, "stochastic_2000_off.jsonl"))
    _, s_adp, dt_sadp = run_tf(["--spec", "adaptive", "--seed", "50000"] + stoch_common, os.path.join(out_dir, "stochastic_2000_adaptive.jsonl"))

    c_tok_off = collections.Counter(str(r["generated_tokens"][0]) for r in s_off if len(r["generated_tokens"]) >= 1)
    c_tok_adp = collections.Counter(str(r["generated_tokens"][0]) for r in s_adp if len(r["generated_tokens"]) >= 1)
    # Also test token #2 and bigram (t1, t2) where speculative verification actively runs!
    c_t2_off = collections.Counter(str(r["generated_tokens"][1]) for r in s_off if len(r["generated_tokens"]) >= 2)
    c_t2_adp = collections.Counter(str(r["generated_tokens"][1]) for r in s_adp if len(r["generated_tokens"]) >= 2)
    c_bi_off = collections.Counter(f"{r['generated_tokens'][0]}_{r['generated_tokens'][1]}" for r in s_off if len(r["generated_tokens"]) >= 2)
    c_bi_adp = collections.Counter(f"{r['generated_tokens'][0]}_{r['generated_tokens'][1]}" for r in s_adp if len(r["generated_tokens"]) >= 2)
    c_bi23_off = collections.Counter(f"{r['generated_tokens'][1]}_{r['generated_tokens'][2]}" for r in s_off if len(r["generated_tokens"]) >= 3)
    c_bi23_adp = collections.Counter(f"{r['generated_tokens'][1]}_{r['generated_tokens'][2]}" for r in s_adp if len(r["generated_tokens"]) >= 3)

    stochastic_res = {
        "temp": 0.8,
        "n_samples_per_arm": len(s_off),
        "elapsed_sec": {"spec_off": dt_soff, "spec_adaptive": dt_sadp},
        "spec_adaptive_accept_rate_mean": sum(float(r["spec_accept_rate"]) for r in s_adp) / max(1, len(s_adp)),
        "spec_adaptive_k_mean": sum(float(r["spec_k_mean"]) for r in s_adp) / max(1, len(s_adp)),
        "first_token_chi2": chi_square_two_sample(c_tok_off, c_tok_adp),
        "second_token_chi2": chi_square_two_sample(c_t2_off, c_t2_adp),
        "bigram_12_chi2": chi_square_two_sample(c_bi_off, c_bi_adp),
        "bigram_23_chi2": chi_square_two_sample(c_bi23_off, c_bi23_adp),
    }

    write_summary(out_dir, {
        "exp_id": "p6_spec",
        "cooldown_sec_between_stream_arms": cooldown_sec,
        "conditions": summaries,
        "exactness": exactness,
        "stochastic_2000": stochastic_res,
    })


# ---------------------------------------------------------------------------
# Phase 7: KV Store (32K Context, Q8_0 vs F16 Quality, Prefix Cache Identity)
# ---------------------------------------------------------------------------
def run_p7() -> None:
    out_dir = os.path.join(REPO_ROOT, "results", "p7_kv")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Quality parity on 20 prompts (n_ctx=2048): f16_a vs f16_b vs q8_0
    common20 = [
        "--mode", "tierflow",
        "--gguf", GGUF_PATH,
        "--manifest", MANIFEST_PATH,
        "--ram-budget-mb", "1400",
        "--n-ctx", "2048",
        "--n-batch", "512",
        "--n-threads", "6",
        "--n-predict", "64",
        "--prompts-file", PROMPTS_20_PATH,
    ]
    _, f16_a, dt_f1 = run_tf(["--kv-type", "f16"] + common20, os.path.join(out_dir, "kv_f16_a.jsonl"))
    _, f16_b, dt_f2 = run_tf(["--kv-type", "f16"] + common20, os.path.join(out_dir, "kv_f16_b.jsonl"))
    _, q8_0, dt_q8 = run_tf(["--kv-type", "q8_0"] + common20, os.path.join(out_dir, "kv_q8_0.jsonl"))

    # 2. Disk prefix cache exactness + TTFT speedup (cold write vs warm restore on same 20 prompts)
    pcache_dir = os.path.join(out_dir, "prefix_blocks")
    if os.path.exists(pcache_dir):
        for fn in os.listdir(pcache_dir):
            os.remove(os.path.join(pcache_dir, fn))
    common_pref = common20 + ["--kv-type", "q8_0", "--prefix-cache-dir", pcache_dir, "--prefix-use", "1"]
    _, pref_cold, dt_pc = run_tf(common_pref, os.path.join(out_dir, "prefix_cold.jsonl"))
    _, pref_warm, dt_pw = run_tf(common_pref, os.path.join(out_dir, "prefix_warm.jsonl"))

    # 3. 32K-token context under tight RAM budget (1700 MB Q8_0 vs 2200 MB F16 control)
    #    plus cold vs warm prefix-cache TTFT speedup on a ~2,500-token long prompt!
    long_prompt_file = os.path.join(out_dir, "prompt_32k_ctx.jsonl")
    long_text = (
        "Summarize the technical architecture of hierarchical paged KV-cache tiering across GPU VRAM, "
        "host DDR5 RAM, and direct unbuffered NVMe SSD spill storage: "
    ) * 120  # ~2500+ tokens prefill inside 32768 context window
    with open(long_prompt_file, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"id": "ctx32k_01", "split": "long_ctx", "prompt": long_text}) + "\n")

    ctx32k_common = [
        "--mode", "tierflow",
        "--gguf", GGUF_PATH,
        "--manifest", MANIFEST_PATH,
        "--n-ctx", "32768",
        "--n-batch", "512",
        "--n-threads", "6",
        "--n-predict", "32",
        "--prompts-file", long_prompt_file,
    ]
    ctx32k_q8_args = ["--ram-budget-mb", "1700", "--kv-type", "q8_0"] + ctx32k_common
    ctx32k_f16_args = ["--ram-budget-mb", "2200", "--kv-type", "f16"] + ctx32k_common
    pl32, rows32, dt32 = run_tf(ctx32k_q8_args, os.path.join(out_dir, "ctx32k_q8_0.jsonl"))
    pl32_f16, rows32_f16, dt32_f16 = run_tf(ctx32k_f16_args, os.path.join(out_dir, "ctx32k_f16.jsonl"))

    long_pcache_dir = os.path.join(out_dir, "long_prefix_blocks")
    if os.path.exists(long_pcache_dir):
        for fn in os.listdir(long_pcache_dir):
            os.remove(os.path.join(long_pcache_dir, fn))
    long_pref_args = ["--ram-budget-mb", "1700", "--kv-type", "q8_0",
                      "--prefix-cache-dir", long_pcache_dir, "--prefix-use", "1"] + ctx32k_common
    pl_lc, rows_lc, dt_lc = run_tf(long_pref_args, os.path.join(out_dir, "long_prefix_cold.jsonl"))
    pl_lw, rows_lw, dt_lw = run_tf(long_pref_args, os.path.join(out_dir, "long_prefix_warm.jsonl"))

    write_summary(out_dir, {
        "exp_id": "p7_kv",
        "kv_quality_20_prompts": {
            "f16_a_vs_f16_b": compare_vs_ref(f16_b, f16_a),
            "q8_0_vs_f16_a": compare_vs_ref(q8_0, f16_a),
            "f16_summary": summarize_condition("kv_f16", f16_a, "", [], dt_f1),
            "q8_0_summary": summarize_condition("kv_q8_0", q8_0, "", [], dt_q8),
        },
        "prefix_cache_20_prompts": {
            "warm_vs_cold_exactness": compare_vs_ref(pref_warm, pref_cold),
            "warm_hit_count": sum(1 for r in pref_warm if r.get("prefix_cache_hit")),
            "cold_summary": summarize_condition("prefix_cold", pref_cold, "", [], dt_pc),
            "warm_summary": summarize_condition("prefix_warm", pref_warm, "", [], dt_pw),
        },
        "prefix_cache_long_2500tok": {
            "warm_vs_cold_exactness": compare_vs_ref(rows_lw, rows_lc),
            "warm_hit_count": sum(1 for r in rows_lw if r.get("prefix_cache_hit")),
            "cold_summary": summarize_condition("long_prefix_cold", rows_lc, pl_lc, long_pref_args, dt_lc),
            "warm_summary": summarize_condition("long_prefix_warm", rows_lw, pl_lw, long_pref_args, dt_lw),
        },
        "ctx_32768": {
            "q8_0_1700mb": summarize_condition("ctx32k_q8_0_1700mb", rows32, pl32, ctx32k_q8_args, dt32),
            "f16_2200mb": summarize_condition("ctx32k_f16_2200mb", rows32_f16, pl32_f16, ctx32k_f16_args, dt32_f16),
            "q8_0_vs_f16_exactness": compare_vs_ref(rows32, rows32_f16),
        },
    })


# ---------------------------------------------------------------------------
# Phase 8: Opt-In Lossy Ladder (4-bit V cache: Q8_K + Q4_0_V)
# ---------------------------------------------------------------------------
def run_p8() -> None:
    out_dir = os.path.join(REPO_ROOT, "results", "p8_lossy_kv_v4")
    common20 = [
        "--mode", "tierflow",
        "--gguf", GGUF_PATH,
        "--manifest", MANIFEST_PATH,
        "--ram-budget-mb", "1400",
        "--n-ctx", "2048",
        "--n-batch", "512",
        "--n-threads", "6",
        "--n-predict", "64",
        "--prompts-file", PROMPTS_20_PATH,
    ]
    pl_f16, f16_rows, dt_f16 = run_tf(["--kv-type", "f16"] + common20, os.path.join(out_dir, "kv_f16.jsonl"))
    pl_q8, q8_rows, dt_q8 = run_tf(["--kv-type", "q8_0"] + common20, os.path.join(out_dir, "kv_q8_0.jsonl"))
    pl_q8q4, q8q4_rows, dt_q8q4 = run_tf(["--kv-type", "q8_q4"] + common20, os.path.join(out_dir, "kv_q8_q4.jsonl"))

    # Compute first-step top-16 KL proxy / logit diff against F16 exact mode
    write_summary(out_dir, {
        "exp_id": "p8_lossy_kv_v4",
        "conditions": {
            "f16": summarize_condition("kv_f16", f16_rows, pl_f16, [], dt_f16),
            "q8_0": summarize_condition("kv_q8_0", q8_rows, pl_q8, [], dt_q8),
            "q8_k_q4_v": summarize_condition("kv_q8_k_q4_v", q8q4_rows, pl_q8q4, [], dt_q8q4),
        },
        "quality_vs_f16": {
            "q8_0_vs_f16": compare_vs_ref(q8_rows, f16_rows),
            "q8_k_q4_v_vs_f16": compare_vs_ref(q8q4_rows, f16_rows),
        },
    })


# ---------------------------------------------------------------------------
# Phase 9: Vulkan Hybrid Tiering + 3-Size-Class Arena Pool
# ---------------------------------------------------------------------------
def run_p9() -> None:
    out_dir = os.path.join(REPO_ROOT, "results", "p9_vulkan_pool")
    base_common = [
        "--gguf", GGUF_PATH,
        "--n-ctx", "2048",
        "--n-batch", "512",
        "--n-threads", "6",
        "--n-predict", "64",
        "--prompts-file", PROMPTS_20_PATH,
    ]
    a_cpu = ["--mode", "stock_mmap", "--n-gpu-layers", "0"] + base_common
    a_vk = ["--mode", "stock_mmap", "--n-gpu-layers", "99"] + base_common
    a_hyb9 = ["--mode", "stock_mmap", "--n-gpu-layers", "9"] + base_common
    a_tf_full = ["--mode", "tierflow", "--manifest", MANIFEST_PATH, "--vram-mb", "2048", "--n-gpu-layers", "99"] + base_common
    a_tf_split = [
        "--mode", "tierflow", "--manifest", MANIFEST_PATH,
        "--vram-mb", "512", "--ram-budget-mb", "512", "--n-gpu-layers", "99",
        "--allow-slow-stream", "1",
    ] + base_common

    pl_cpu, r_cpu, dt_cpu = run_tf(a_cpu, os.path.join(out_dir, "stock_mmap_cpu.jsonl"))
    pl_vk, r_vk, dt_vk = run_tf(a_vk, os.path.join(out_dir, "stock_mmap_vulkan.jsonl"))
    pl_hyb9, r_hyb9, dt_hyb9 = run_tf(a_hyb9, os.path.join(out_dir, "stock_mmap_hybrid9.jsonl"))
    pl_full, r_full, dt_full = run_tf(a_tf_full, os.path.join(out_dir, "tierflow_vulkan_full.jsonl"))
    pl_split, r_split, dt_split = run_tf(a_tf_split, os.path.join(out_dir, "tierflow_hybrid_512_512.jsonl"))

    split_plan = parse_plan_line(pl_split)
    slot_bank_b = int(split_plan.get("slot_bank", 0))
    n_slots = int(split_plan.get("cache_slots", 0))
    max_stride_b = int(r_split[-1].get("slot_stride_bytes", 0)) if r_split else 0
    if max_stride_b == 0:
        # Derive largest class stride directly from manifest (ffn_down page-aligned size)
        with open(MANIFEST_PATH, "r", encoding="utf-8") as mf:
            man = json.load(mf)
        max_stride_b = max((int(o["page_aligned_bytes"]) for o in man.get("objects", [])), default=0)
    uniform_equiv_b = n_slots * max_stride_b

    write_summary(out_dir, {
        "exp_id": "p9_vulkan_pool",
        "conditions": {
            "stock_mmap_cpu": summarize_condition("stock_mmap_cpu", r_cpu, pl_cpu, a_cpu, dt_cpu),
            "stock_mmap_vulkan": summarize_condition("stock_mmap_vulkan", r_vk, pl_vk, a_vk, dt_vk),
            "stock_mmap_hybrid9": summarize_condition("stock_mmap_hybrid9", r_hyb9, pl_hyb9, a_hyb9, dt_hyb9),
            "tierflow_vulkan_full": summarize_condition("tierflow_vulkan_full", r_full, pl_full, a_tf_full, dt_full),
            "tierflow_hybrid_512_512": summarize_condition("tierflow_hybrid_512_512", r_split, pl_split, a_tf_split, dt_split),
        },
        "exactness": {
            "tierflow_vulkan_full_vs_stock_mmap_vulkan": compare_vs_ref(r_full, r_vk),
            "tierflow_vulkan_full_vs_stock_mmap_cpu": compare_vs_ref(r_full, r_cpu),
            "tierflow_hybrid_512_512_vs_stock_mmap_hybrid9": compare_vs_ref(r_split, r_hyb9),
            "tierflow_hybrid_512_512_vs_stock_mmap_vulkan": compare_vs_ref(r_split, r_vk),
            "tierflow_hybrid_512_512_vs_stock_mmap_cpu": compare_vs_ref(r_split, r_cpu),
            "stock_mmap_vulkan_vs_stock_mmap_cpu": compare_vs_ref(r_vk, r_cpu),
        },
        "arena_efficiency": {
            "n_cache_slots": n_slots,
            "class_arena_slot_bank_bytes": slot_bank_b,
            "uniform_stride_equiv_bytes": uniform_equiv_b,
            "memory_reduction_ratio": (uniform_equiv_b / slot_bank_b) if slot_bank_b > 0 else None,
        },
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["export-prompts", "selfcheck", "p3", "p4", "p4_v2", "p5", "p6", "p7", "p8", "p9"])
    args = ap.parse_args()
    if args.phase == "export-prompts":
        export_prompts()
    elif args.phase == "selfcheck":
        run_selfcheck()
    elif args.phase == "p3":
        run_p3()
    elif args.phase == "p4":
        run_p4("p4_planner")
    elif args.phase == "p4_v2":
        run_p4("p4_planner_v2")
    elif args.phase == "p5":
        run_p5()
    elif args.phase == "p6":
        run_p6()
    elif args.phase == "p7":
        run_p7()
    elif args.phase == "p8":
        run_p8()
    elif args.phase == "p9":
        run_p9()


if __name__ == "__main__":
    main()

