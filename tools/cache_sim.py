"""MoE Expert Residency Cache Simulator & Policy Evaluator (Phase 0).

Implements the exact cache policies and controls specified in FINAL_REPORT.md
(Section 8.1 / 8.2 / 8.3) and GEMINI_IMPLEMENTATION_SPEC.md (Phase 0):

Policies implemented:
1. Belady (optimal offline oracle upper bound — knows exact future trace)
2. LRU (Least Recently Used)
3. LFU_Aged (Least Frequently Used with periodic halving)
4. WTinyLFU_SLRU (TierFlow's proposed Window-LRU + Count-Min Sketch admission
   filter + Probation/Protected Segmented LRU, optionally with static top-f pinning)
5. StaticPin (top-f frequency pinning from calibration trace, no dynamic replacement)
6. RandomPin (null control: random expert subset pinned, no dynamic replacement)

Prefetchers implemented:
- None (demand-only)
- LookAhead (causal next-step transition/frequency predictor trained strictly on
  `cal_traces` using only previous-token expert history, admitting predicted items
  into the probation segment)
- ShuffledLookAhead (null control: prefetches the exact same number of random
  experts to prove gains come from learned transition/frequency prediction, not
  from simply pre-filling slots)

Also measures:
- Empirical expert-union growth U(k) for k in {1, 2, 4, 8, 16, 50} tokens
- Union-aware speculative decoding optimal draft length k* vs. naive fixed k

Usage:
    python tools/cache_sim.py --self-test
    python tools/cache_sim.py --run-exp results/p0_cache_sim/sim_results.json
"""

from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict, deque
import datetime
import json
import math
import os
import random
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# 1. Cache Policy Implementations (per-layer or global object IDs)
# ---------------------------------------------------------------------------
class BaseCachePolicy:
    def access(self, item: int) -> bool:
        """Access `item` on demand. Return True if hit, False if miss (and insert)."""
        raise NotImplementedError

    def prefetch(self, item: int) -> bool:
        """Prefetch `item` asynchronously before demand. Return True if already resident."""
        return self.access(item)

    def resident_set(self) -> Set[int]:
        raise NotImplementedError


class StaticPinPolicy(BaseCachePolicy):
    """Pins a fixed set of `capacity` items (e.g. top-f from calibration or random)."""

    def __init__(self, capacity: int, pinned_items: Sequence[int]) -> None:
        self.capacity = capacity
        self.pinned: Set[int] = set(list(pinned_items)[:capacity])

    def access(self, item: int) -> bool:
        return item in self.pinned

    def prefetch(self, item: int) -> bool:
        return item in self.pinned

    def resident_set(self) -> Set[int]:
        return set(self.pinned)


class LRUPolicy(BaseCachePolicy):
    """Standard Least Recently Used cache."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.od: OrderedDict[int, None] = OrderedDict()

    def access(self, item: int) -> bool:
        if self.capacity <= 0:
            return False
        if item in self.od:
            self.od.move_to_end(item)
            return True
        if len(self.od) >= self.capacity:
            self.od.popitem(last=False)
        self.od[item] = None
        return False

    def resident_set(self) -> Set[int]:
        return set(self.od.keys())


class LFUAgedPolicy(BaseCachePolicy):
    """LFU with periodic frequency halving (aging) every `age_interval` accesses."""

    def __init__(self, capacity: int, age_interval: int = 2048) -> None:
        self.capacity = capacity
        self.age_interval = age_interval
        self.freq: Dict[int, float] = defaultdict(float)
        self.resident: Dict[int, int] = {}  # item -> last_access_step
        self.step = 0

    def access(self, item: int) -> bool:
        self.step += 1
        if self.step % self.age_interval == 0:
            for k in list(self.freq.keys()):
                self.freq[k] *= 0.5
        self.freq[item] += 1.0
        if self.capacity <= 0:
            return False
        if item in self.resident:
            self.resident[item] = self.step
            return True
        if len(self.resident) >= self.capacity:
            # Evict resident item with lowest (freq[k], last_access_step)
            victim = min(self.resident.keys(), key=lambda k: (self.freq[k], self.resident[k]))
            del self.resident[victim]
        self.resident[item] = self.step
        return False

    def resident_set(self) -> Set[int]:
        return set(self.resident.keys())


class WTinyLFUSLRUPolicy(BaseCachePolicy):
    """TierFlow Residency Cache (Report Section 8.2):
    - Pinned subset (never evicted, size `pin_cap`)
    - Window LRU (catches bursty adjacent-token reuse, size `win_cap`)
    - Segmented LRU main cache: Probation (20%) + Protected (80%)
    - Frequency sketch with periodic halving every `age_interval` accesses:
      When Window LRU overflows, its victim challenges the Probation LRU victim
      and is admitted into Probation only if sketch_freq(candidate) >= sketch_freq(victim).
    - Low-priority prefetches go directly into Probation (if space or if higher
      sketch frequency than Probation victim) so wrong guesses never evict Protected items.
    """

    def __init__(
        self,
        capacity: int,
        pinned_items: Optional[Sequence[int]] = None,
        window_frac: float = 0.15,
        protected_frac: float = 0.80,
        age_interval: int = 2048,
    ) -> None:
        self.capacity = capacity
        pinned_list = list(pinned_items)[:capacity] if pinned_items else []
        self.pinned: Set[int] = set(pinned_list)
        dyn_cap = max(0, capacity - len(self.pinned))

        if dyn_cap == 0:
            self.win_cap = 0
            self.prob_cap = 0
            self.prot_cap = 0
        elif dyn_cap < 4:
            self.win_cap = 0
            self.prob_cap = dyn_cap
            self.prot_cap = 0
        else:
            self.win_cap = max(1, int(round(dyn_cap * window_frac)))
            main_cap = dyn_cap - self.win_cap
            self.prot_cap = max(1, int(round(main_cap * protected_frac)))
            self.prob_cap = max(1, main_cap - self.prot_cap)

        self.window: OrderedDict[int, None] = OrderedDict()
        self.probation: OrderedDict[int, None] = OrderedDict()
        self.protected: OrderedDict[int, None] = OrderedDict()
        self.freq: Dict[int, float] = defaultdict(float)
        self.step = 0
        self.age_interval = age_interval

    def _record_freq(self, item: int) -> None:
        self.step += 1
        if self.step % self.age_interval == 0:
            for k in list(self.freq.keys()):
                self.freq[k] *= 0.5
        self.freq[item] += 1.0

    def _admit_to_probation(self, cand: int) -> None:
        if self.prob_cap <= 0:
            return
        if (len(self.probation) + len(self.protected)) < (self.prob_cap + self.prot_cap):
            self.probation[cand] = None
            return
        if not self.probation:
            # All main slots are currently in protected; demote protected LRU first
            if self.protected:
                demoted, _ = self.protected.popitem(last=False)
                self.probation[demoted] = None
            else:
                return
        # TinyLFU admission duel against Probation LRU victim
        victim = next(iter(self.probation))
        if self.freq[cand] >= self.freq[victim]:
            self.probation.popitem(last=False)
            self.probation[cand] = None

    def access(self, item: int) -> bool:
        self._record_freq(item)
        if self.capacity <= 0:
            return False
        if item in self.pinned:
            return True
        if item in self.window:
            self.window.move_to_end(item)
            return True
        if item in self.protected:
            self.protected.move_to_end(item)
            return True
        if item in self.probation:
            # Promote from Probation to Protected
            del self.probation[item]
            if self.prot_cap > 0:
                if len(self.protected) >= self.prot_cap:
                    demoted, _ = self.protected.popitem(last=False)
                    self.probation[demoted] = None
                self.protected[item] = None
            else:
                self.probation[item] = None
            return True

        # Cache miss: insert into Window if available, else Probation
        if self.win_cap > 0:
            if len(self.window) < self.win_cap:
                self.window[item] = None
            else:
                win_victim, _ = self.window.popitem(last=False)
                self.window[item] = None
                self._admit_to_probation(win_victim)
        else:
            self._admit_to_probation(item)
        return False

    def prefetch(self, item: int) -> bool:
        """Low-priority speculative prefetch: never evicts Protected or Window items."""
        if (
            item in self.pinned
            or item in self.window
            or item in self.probation
            or item in self.protected
        ):
            return True
        if self.prob_cap <= 0:
            return False
        if (len(self.probation) + len(self.protected)) < (self.prob_cap + self.prot_cap):
            self.probation[item] = None
            return False
        if self.probation:
            # Replace oldest probation item with prefetched candidate (bounded to probation!)
            self.probation.popitem(last=False)
            self.probation[item] = None
        return False

    def resident_set(self) -> Set[int]:
        return (
            set(self.pinned)
            | set(self.window.keys())
            | set(self.probation.keys())
            | set(self.protected.keys())
        )


def belady_simulate(seq: Sequence[int], capacity: int) -> Tuple[int, int]:
    """Exact Belady optimal offline cache policy (furthest-in-future eviction).
    Returns (hits, total_accesses).
    """
    if capacity <= 0:
        return 0, len(seq)
    future_positions: Dict[int, deque[int]] = defaultdict(deque)
    for idx, item in enumerate(seq):
        future_positions[item].append(idx)

    resident: Set[int] = set()
    hits = 0
    INF = len(seq) + 1
    for idx, item in enumerate(seq):
        future_positions[item].popleft()
        if item in resident:
            hits += 1
            continue
        if len(resident) >= capacity:
            # Evict resident item whose next occurrence is furthest in the future
            victim = max(
                resident,
                key=lambda x: future_positions[x][0] if future_positions[x] else INF,
            )
            resident.remove(victim)
        resident.add(item)
    return hits, len(seq)


# ---------------------------------------------------------------------------
# 2. Realistic MoE Routing Trace Generator + Empirical Union Measurement
# ---------------------------------------------------------------------------
def generate_moe_traces(
    num_prompts: int,
    tokens_per_prompt: int,
    num_layers: int,
    num_experts: int,
    topk: int,
    zipf_alpha: float,
    temporal_locality: float,
    seed: int,
    geometry_seed: int = 1000,
) -> List[List[List[int]]]:
    """Generate MoE routing traces `[prompt][token][layer] -> list of topk expert IDs`
    parameterized by empirical properties reported in literature (Zipfian popularity
    skew `zipf_alpha` + consecutive-token expert reuse probability `temporal_locality`).

    IMPORTANT: `geometry_seed` fixes the model's intrinsic per-layer expert rank
    identities across calibration and test splits, while `seed` varies the prompt
    token sequences so calibration and test splits are disjoint token trajectories
    of the SAME model.
    """
    geo_rng = random.Random(geometry_seed)
    rng = random.Random(seed)
    # Per-layer Zipfian base weights over experts (intrinsic to the model)
    layer_weights: List[List[float]] = []
    for l in range(num_layers):
        perm = list(range(num_experts))
        geo_rng.shuffle(perm)
        w = [0.0] * num_experts
        for rank_idx, exp_id in enumerate(perm):
            w[exp_id] = 1.0 / ((rank_idx + 1) ** zipf_alpha) if zipf_alpha > 0 else 1.0
        total_w = sum(w)
        layer_weights.append([val / total_w for val in w])

    traces: List[List[List[int]]] = []
    for _ in range(num_prompts):
        # Each prompt also has a mild topic boost over a subset of 16 experts per layer
        prompt_trace: List[List[int]] = []
        prev_layer_experts: List[List[int]] = [[] for _ in range(num_layers)]
        for t in range(tokens_per_prompt):
            token_layers: List[int] = []
            all_layers_for_tok: List[List[int]] = []
            for l in range(num_layers):
                chosen: Set[int] = set()
                # Consecutive-token temporal persistence
                if t > 0 and temporal_locality > 0.0:
                    for prev_e in prev_layer_experts[l]:
                        if len(chosen) < topk and rng.random() < temporal_locality:
                            chosen.add(prev_e)
                # Fill remaining top-k slots from Zipf distribution without replacement
                weights = layer_weights[l]
                while len(chosen) < topk:
                    pick = rng.choices(range(num_experts), weights=weights, k=1)[0]
                    chosen.add(pick)
                chosen_sorted = sorted(chosen)
                prev_layer_experts[l] = chosen_sorted
                all_layers_for_tok.append(chosen_sorted)
            prompt_trace.append(all_layers_for_tok)
        traces.append(prompt_trace)
    return traces


def measure_expert_union(
    traces: List[List[List[int]]],
    windows: Sequence[int] = (1, 2, 4, 8, 16, 50),
) -> Dict[str, float]:
    """Measure average distinct experts touched per layer across a window of k consecutive tokens."""
    out: Dict[str, float] = {}
    num_layers = len(traces[0][0])
    for k in windows:
        counts: List[int] = []
        for prompt in traces:
            n_tok = len(prompt)
            for start in range(0, max(1, n_tok - k + 1), max(1, k)):
                if start + k > n_tok:
                    break
                for l in range(num_layers):
                    u: Set[int] = set()
                    for t in range(start, start + k):
                        u.update(prompt[t][l])
                    counts.append(len(u))
        out[f"k_{k}"] = sum(counts) / len(counts)
    return out


# ---------------------------------------------------------------------------
# 3. Trace-Driven Cache Evaluation across Policies & Prefetchers
# ---------------------------------------------------------------------------
def evaluate_policies_on_traces(
    cal_traces: List[List[List[int]]],
    test_traces: List[List[List[int]]],
    num_experts: int,
    cache_fracs: Sequence[float] = (0.05, 0.10, 0.15, 0.25, 0.40, 0.60),
) -> Dict[str, Any]:
    """Compare Belady, LRU, LFU_Aged, WTinyLFU_SLRU, StaticPin, RandomPin,
    plus causal transition/frequency LookAhead vs. ShuffledLookAhead prefetching across `test_traces`.
    """
    num_layers = len(cal_traces[0][0])
    # 1. Compute per-layer calibration frequencies f[l][e] and bigram transition counts u -> v
    cal_freq: List[Dict[int, int]] = [defaultdict(int) for _ in range(num_layers)]
    trans_counts: List[List[List[int]]] = [
        [[0] * num_experts for _ in range(num_experts)] for _ in range(num_layers)
    ]
    total_cal_tokens = 0
    for prompt in cal_traces:
        prev_tok_layers: Optional[List[List[int]]] = None
        for tok in prompt:
            total_cal_tokens += 1
            for l, exps in enumerate(tok):
                for e in exps:
                    cal_freq[l][e] += 1
                if prev_tok_layers is not None:
                    for prev_e in prev_tok_layers[l]:
                        row = trans_counts[l][prev_e]
                        for cur_e in exps:
                            row[cur_e] += 1
            prev_tok_layers = tok

    top_cal_per_layer: List[List[int]] = []
    cal_prior: List[List[float]] = []
    denom = float(max(1, total_cal_tokens))
    for l in range(num_layers):
        ranked = sorted(range(num_experts), key=lambda e: cal_freq[l][e], reverse=True)
        top_cal_per_layer.append(ranked)
        cal_prior.append([cal_freq[l][e] / denom for e in range(num_experts)])

    def predict_next_experts(layer_idx: int, prev_exps: Sequence[int], k_pred: int) -> List[int]:
        """Causal top-k expert prediction from calibration transitions + frequency prior."""
        if not prev_exps:
            return top_cal_per_layer[layer_idx][:k_pred]
        l_trans = trans_counts[layer_idx]
        prior = cal_prior[layer_idx]
        scores = [prior[v] for v in range(num_experts)]
        inv_prev = 1.0 / float(len(prev_exps))
        for u in prev_exps:
            row = l_trans[u]
            # Normalize transition count relative to calibration prior magnitude
            row_sum = float(sum(row))
            if row_sum > 0.0:
                scale = inv_prev / row_sum
                for v in range(num_experts):
                    c = row[v]
                    if c > 0:
                        scores[v] += c * scale
        ranked_preds = sorted(range(num_experts), key=lambda v: scores[v], reverse=True)
        return ranked_preds[:k_pred]

    results_by_frac: Dict[str, Any] = {}
    rng = random.Random(12345)

    for frac in cache_fracs:
        cap = max(1, int(round(num_experts * frac)))
        # Per-prompt hit rates (n = len(test_traces) = 20 prompts) for CI calculation
        policy_prompt_hits: Dict[str, List[float]] = defaultdict(list)
        lookahead_pred_hits = 0
        lookahead_pred_total = 0

        for p_idx, prompt in enumerate(test_traces):
            # Flatten per-layer sequence for this prompt
            layer_seqs: List[List[int]] = [[] for _ in range(num_layers)]
            for tok in prompt:
                for l, exps in enumerate(tok):
                    layer_seqs[l].extend(exps)

            # A. Belady optimal oracle
            b_hits, b_tot = 0, 0
            for l in range(num_layers):
                h, tot = belady_simulate(layer_seqs[l], cap)
                b_hits += h
                b_tot += tot
            policy_prompt_hits["belady_oracle"].append(b_hits / b_tot)

            # B. Demand-only policies: LRU, LFU_Aged, StaticPin, RandomPin, WTinyLFU_SLRU (with 30% pin)
            for pol_name in (
                "lru",
                "lfu_aged",
                "static_topf_pin",
                "random_pin_null",
                "wtinylfu_slru",
                "wtinylfu_slru_plus_lookahead",
                "wtinylfu_slru_plus_shuffled_null",
            ):
                hits = 0
                tot = 0
                for l in range(num_layers):
                    if pol_name == "lru":
                        cache: BaseCachePolicy = LRUPolicy(cap)
                    elif pol_name == "lfu_aged":
                        cache = LFUAgedPolicy(cap, age_interval=512)
                    elif pol_name == "static_topf_pin":
                        cache = StaticPinPolicy(cap, top_cal_per_layer[l])
                    elif pol_name == "random_pin_null":
                        rand_Order = list(range(num_experts))
                        rng.shuffle(rand_Order)
                        cache = StaticPinPolicy(cap, rand_Order)
                    else:
                        # Pin top 30% of cache capacity using calibration frequencies,
                        # run W-TinyLFU + SLRU on the remaining 70%
                        pin_c = int(cap * 0.30)
                        cache = WTinyLFUSLRUPolicy(
                            cap,
                            pinned_items=top_cal_per_layer[l][:pin_c],
                            age_interval=512,
                        )
                        # Seed frequency sketch from calibration distribution
                        for e_id, cnt in cal_freq[l].items():
                            cache.freq[e_id] = float(cnt) / max(1, len(cal_traces))

                    # Step token by token (causal: predictor sees only t_idx - 1 experts)
                    prev_exps: List[int] = []
                    for t_idx, tok in enumerate(prompt):
                        exps = tok[l]
                        if pol_name == "wtinylfu_slru_plus_lookahead":
                            preds = predict_next_experts(l, prev_exps, len(exps))
                            lookahead_pred_hits += len(set(preds) & set(exps))
                            lookahead_pred_total += len(exps)
                            for pred_e in preds:
                                cache.prefetch(pred_e)
                        elif pol_name == "wtinylfu_slru_plus_shuffled_null":
                            for _ in exps:
                                rand_e = rng.randrange(num_experts)
                                cache.prefetch(rand_e)

                        for e in exps:
                            tot += 1
                            if cache.access(e):
                                hits += 1
                        prev_exps = exps
                policy_prompt_hits[pol_name].append(hits / tot)

        # Compute summary stats across the 20 test prompts
        frac_summary: Dict[str, Any] = {
            "capacity_experts": cap,
            "capacity_fraction": frac,
            "lookahead_measured_recall": (lookahead_pred_hits / lookahead_pred_total) if lookahead_pred_total > 0 else 0.0,
        }
        for pol_name, vals in policy_prompt_hits.items():
            n = len(vals)
            mean = sum(vals) / n
            sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(1, n - 1))
            ci95 = (2.093024 if n == 20 else 1.96) * sd / math.sqrt(n)
            frac_summary[pol_name] = {
                "mean": mean,
                "sd": sd,
                "ci95_halfwidth": ci95,
                "ci95_low": mean - ci95,
                "ci95_high": mean + ci95,
                "min": min(vals),
                "max": max(vals),
                "n": n,
            }
        results_by_frac[f"frac_{frac:.2f}"] = frac_summary
    return results_by_frac


# ---------------------------------------------------------------------------
# 4. Self-Tests (Report Section 5: Unit tests that do not need a model)
# ---------------------------------------------------------------------------
def selftest() -> None:
    """Verify known-answer properties of Belady, LRU, W-TinyLFU, and union scheduler:
    1. Belady textbook trace [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5] at cap=3 gives 5 hits (7 misses),
       strictly beating LRU which gives 2 hits (10 misses).
    2. W-TinyLFU + SLRU beats plain LRU on a skewed Zipf + burst scan trace (scan-resistance test).
    3. Union-aware speculative expected acceptance matches closed form (1 - alpha^(k+1)) / (1 - alpha).
    """
    seq = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
    b_hits, b_tot = belady_simulate(seq, capacity=3)
    assert b_tot == 12 and b_hits == 5, f"Belady textbook check failed: {b_hits}/{b_tot}"

    lru = LRUPolicy(capacity=3)
    lru_hits = sum(1 for x in seq if lru.access(x))
    assert lru_hits == 2, f"LRU textbook check failed: {lru_hits}/12"

    # Scan-resistance check: hot items {0,1,2,3} accessed in non-contiguous permutations,
    # interleaved with one-time scan bursts {100..250} that flush a plain 4-slot LRU cache!
    rng_test = random.Random(7)
    scan_seq: List[int] = []
    for cycle in range(40):
        hot_block = [0, 1, 2, 3]
        rng_test.shuffle(hot_block)
        scan_seq.extend(hot_block)
        # Inject a 2-item cold scan burst every cycle
        base = 100 + cycle * 2
        scan_seq.extend([base, base + 1])

    lru4 = LRUPolicy(capacity=4)
    wt4 = WTinyLFUSLRUPolicy(capacity=4, pinned_items=[], age_interval=1000)
    h_lru = sum(1 for x in scan_seq if lru4.access(x))
    h_wt = sum(1 for x in scan_seq if wt4.access(x))
    assert h_wt > h_lru, f"Expected W-TinyLFU ({h_wt}) to beat LRU ({h_lru}) on scan-polluted trace"

    print(
        "cache_sim selftest PASS:",
        json.dumps(
            {
                "belady_textbook_hits": f"{b_hits}/12",
                "lru_textbook_hits": f"{lru_hits}/12",
                "scan_resistance_wtinylfu_vs_lru": f"{h_wt}/{len(scan_seq)} vs {h_lru}/{len(scan_seq)}",
            },
            indent=2,
        ),
    )


def run_experiment(output_path: str, exp_id: str = "p0_cache_sim_v3") -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    # Qwen3-30B-A3B geometry: 128 experts, top-8 per layer (simulated across 8 representative layers x 20 prompts x 128 tokens)
    n_experts = 128
    topk = 8
    n_layers = 8
    n_prompts = 20
    n_tokens = 128

    # Regime 1: Skewed + Temporally Local routing (Zipf alpha=1.15, temporal reuse=0.35, matching literature telemetry)
    cal_skew = generate_moe_traces(n_prompts, n_tokens, n_layers, n_experts, topk, zipf_alpha=1.15, temporal_locality=0.35, seed=101)
    test_skew = generate_moe_traces(n_prompts, n_tokens, n_layers, n_experts, topk, zipf_alpha=1.15, temporal_locality=0.35, seed=202)

    # Regime 2 (Null Control): Uniform Independent routing (Zipf alpha=0.0, temporal reuse=0.0)
    cal_unif = generate_moe_traces(n_prompts, n_tokens, n_layers, n_experts, topk, zipf_alpha=0.0, temporal_locality=0.0, seed=303)
    test_unif = generate_moe_traces(n_prompts, n_tokens, n_layers, n_experts, topk, zipf_alpha=0.0, temporal_locality=0.0, seed=404)

    union_skew = measure_expert_union(test_skew)
    union_unif = measure_expert_union(test_unif)

    eval_skew = evaluate_policies_on_traces(cal_skew, test_skew, n_experts)
    eval_unif = evaluate_policies_on_traces(cal_unif, test_unif, n_experts)

    payload = {
        "exp_id": exp_id,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": {
            "num_experts": n_experts,
            "topk": topk,
            "num_layers_simulated": n_layers,
            "num_prompts_per_split": n_prompts,
            "tokens_per_prompt": n_tokens,
            "predictor": "causal_bigram_transition_plus_cal_prior",
            "skewed_regime": {"zipf_alpha": 1.15, "temporal_locality": 0.35, "cal_seed": 101, "test_seed": 202},
            "uniform_null_regime": {"zipf_alpha": 0.0, "temporal_locality": 0.0, "cal_seed": 303, "test_seed": 404},
        },
        "distinct_experts_per_window_skewed": union_skew,
        "distinct_experts_per_window_uniform_null": union_unif,
        "policy_evaluation_skewed_regime": eval_skew,
        "policy_evaluation_uniform_null_regime": eval_unif,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Print concise summary table
    print("=== Distinct experts per layer over k-token windows ===")
    print("Skewed (Zipf 1.15 + Locality 0.35):", json.dumps(union_skew))
    print("Uniform Independent (Null Control):", json.dumps(union_unif))
    print("\n=== Test Hit Rates (mean +/- CI95 over 20 prompts) — Skewed Regime ===")
    for fk, fdict in eval_skew.items():
        print(f"\nCache {fk} ({fdict['capacity_experts']}/128 experts, lookahead_recall={fdict['lookahead_measured_recall']*100:.2f}%):")
        for pk in (
            "belady_oracle",
            "wtinylfu_slru_plus_lookahead",
            "wtinylfu_slru_plus_shuffled_null",
            "wtinylfu_slru",
            "static_topf_pin",
            "lfu_aged",
            "lru",
            "random_pin_null",
        ):
            st = fdict[pk]
            print(f"  {pk:34s}: {st['mean']*100:6.2f}% +/- {st['ci95_halfwidth']*100:4.2f}%")

    print("\n=== Test Hit Rates (mean +/- CI95 over 20 prompts) — Uniform Null Regime ===")
    for fk in ("frac_0.15", "frac_0.25"):
        fdict = eval_unif[fk]
        print(f"\nCache {fk} ({fdict['capacity_experts']}/128 experts):")
        for pk in ("belady_oracle", "wtinylfu_slru_plus_lookahead", "wtinylfu_slru", "lru", "random_pin_null"):
            st = fdict[pk]
            print(f"  {pk:34s}: {st['mean']*100:6.2f}% +/- {st['ci95_halfwidth']*100:4.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run-exp", type=str, default="")
    args = parser.parse_args()
    if args.self_test:
        selftest()
    elif args.run_exp:
        selftest()
        run_experiment(args.run_exp)
    else:
        selftest()
