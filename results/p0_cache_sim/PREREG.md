# Pre-Registration: `p0_cache_sim`

Date: 2026-11-12 (UTC)

## 1. Claim & Headline Metrics
- **Claim 1 (W-TinyLFU + SLRU + LookAhead beats plain LRU on skewed MoE routing):** On synthetic traces matching empirical MoE routing telemetry (`num_experts=128, topk=8`, Zipf popularity skew $\alpha = 1.15$, consecutive-token reuse probability $0.35$), `wtinylfu_slru` achieves a higher mean test hit rate than `lru` at cache fractions $\in \{0.10, 0.15, 0.25\}$ with non-overlapping 95% confidence intervals across $n = 20$ disjoint test prompts, and adding router look-ahead prefetch (`wtinylfu_slru_plus_lookahead`) further increases hit rate whereas a shuffled look-ahead null control (`wtinylfu_slru_plus_shuffled_null`) does not.
- **Claim 2 (Negative / Boundary Result — Uniform Independent Routing Refutes Caching):** Under uniform independent routing (`zipf_alpha=0.0, temporal_locality=0.0`), all non-prefetching online policies (`wtinylfu_slru`, `lru`, `static_topf_pin`, `random_pin_null`) collapse to the cache capacity fraction ($\approx 15\%$ at `frac_0.15` and $\approx 25\%$ at `frac_0.25`), proving that MoE SSD streaming is viable **only** when router activations exhibit real Zipfian skew or temporal persistence.

## 2. Falsifiers
- **Falsifier 1:** `wtinylfu_slru` fails to beat `lru` at `frac_0.15` or `frac_0.25`, OR `wtinylfu_slru_plus_shuffled_null` matches `wtinylfu_slru_plus_lookahead`.
- **Falsifier 2:** `wtinylfu_slru` achieves $> 30\%$ hit rate at `frac_0.25` on the uniform-independent null control trace without look-ahead.

## 3. Frozen Protocol
- Disjoint calibration (`seed=101` / `seed=303`, 20 prompts $\times$ 128 tokens) and test (`seed=202` / `seed=404`, 20 prompts $\times$ 128 tokens) splits.
- Policies evaluated identically across all 20 test prompts:
  1. `belady_oracle` (offline future-knowing upper bound)
  2. `wtinylfu_slru_plus_lookahead` (85% top-k router look-ahead accuracy)
  3. `wtinylfu_slru_plus_shuffled_null` (null control: random expert IDs prefetched at same rate)
  4. `wtinylfu_slru` (30% static top-f pin + 70% Window-TinyLFU/SLRU)
  5. `static_topf_pin` (100% static calibration top-f pin)
  6. `lfu_aged` (LFU with halving every 512 accesses)
  7. `lru` (standard LRU)
  8. `random_pin_null` (null control: random subset pinned)

