# Pre-Registration: `p0_cache_sim_v2`

Date: 2026-11-12 (UTC)

## Reason for New Experiment ID (`p0_cache_sim_v2`)
In `p0_cache_sim`, `generate_moe_traces` used `seed` (`101` for calibration, `202` for test) to shuffle the model's intrinsic per-layer expert rank order `perm` *before* generating tokens. That meant the calibration trace and test trace belonged to two *different* random expert permutations rather than the same model, turning `static_topf_pin` and `wtinylfu_slru`'s 30% calibration pin into a random pin (`15.14%` hit rate at `frac_0.15`, identical to `random_pin_null`'s `14.40%`). Both `p0_cache_sim` (kept on record) and `p0_cache_sim_v2` (with fixed model `geometry_seed=1000` and disjoint prompt trajectory seeds `cal_seed=101, test_seed=202`) are reported.

## 1. Claims & Falsifiers (unchanged from `p0_cache_sim`)
- **Claim 1:** On skewed MoE routing (`zipf_alpha=1.15, temporal_locality=0.35`, fixed model `geometry_seed=1000`), `wtinylfu_slru` beats `lru` at cache fractions $\in \{0.10, 0.15, 0.25\}$ with non-overlapping 95% CIs across $n=20$ disjoint test prompts, and `wtinylfu_slru_plus_lookahead` beats `wtinylfu_slru_plus_shuffled_null`.
- **Claim 2 (Negative / Boundary Result):** Under uniform independent routing (`zipf_alpha=0.0, temporal_locality=0.0`), all non-prefetching online policies collapse to the cache capacity fraction ($\approx 15\%$ at `frac_0.15`, $\approx 25\%$ at `frac_0.25`).

