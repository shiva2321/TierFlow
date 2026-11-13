# Pre-Registration: `p0_cache_sim_v3`

## Reason for New Experiment ID (`p0_cache_sim_v3`)
In `p0_cache_sim_v2`, the look-ahead prefetcher simulated `lookahead_accuracy = 0.85` using `rng.random()` over future test tokens. To eliminate synthetic random outcome injection and evaluate a strictly causal prefetcher, `p0_cache_sim_v3` replaces the stochastic accuracy flip with a causal bigram-transition + calibration-prior predictor trained strictly on `cal_traces` (`predict_next_experts(l, prev_exps, k_pred)`), measuring empirical top-`k` recall and end-to-end hit rates on disjoint `test_traces`. Both `p0_cache_sim_v2` (historical baseline) and `p0_cache_sim_v3` are preserved on disk.

## Pre-Registered Claims & Null Controls
- **Hypothesis 1:** On skewed MoE traces (`zipf_alpha=1.15, temporal_locality=0.35, geometry_seed=1000`), causal `wtinylfu_slru_plus_lookahead` outperforms `wtinylfu_slru_plus_shuffled_null` and demand-only `wtinylfu_slru` at `frac_0.25` with non-overlapping 95% CIs ($n=20$ prompts).
- **Null Control (`zipf_alpha=0.0, temporal_locality=0.0`):** Under uniform independent routing, non-oracle policies remain bounded near the cache fraction and learned transition recall drops to ~random (`topk / num_experts = 8/128 = 6.25%`).
