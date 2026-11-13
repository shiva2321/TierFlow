# Upstream `llama.cpp` State (`docs/UPSTREAM_STATE.md`)

Inspected commit: `3ba1ce0e28d991800e1585b7d3533f77213939d6` (shallow clone in `third_party/llama.cpp/`, 2026-11-12).

## 1. MoE Graph Construction & `ggml_mul_mat_id`
- `src/llama-graph.cpp` (`llm_graph_context::build_moe_ffn`, lines `2204–2670`):
  - Computes router logits and top-$k$ `selected_experts` (`[n_expert_used, n_tokens]`).
  - Calls `build_moe_cache_slots(selected_experts, up_exps, gate_exps, down_exps, gate_up_exps, cpu_part, il)` (lines `2433, 2672`).
  - Performs expert matrix multiplication via `ggml_mul_mat_id(ctx0, cached ? moe_cache->get_experts(w) : w, cur_mm, slots)` (lines `1772–1773`).
- When `moe_cache` is active, `build_moe_cache_slots` appends a `ggml_get_rows(ctx0, slot_map, ids)` host-weight lookup (`line 2712`) which forces `ggml_backend_sched` to split the graph at each cached MoE layer and run `llama_moe_cache::copy(...)` (`src/llama-context.cpp:3167`).

## 2. Upstream Expert Cache & CLI Flags (`src/llama-moe-cache.{h,cpp}`, `common/arg.cpp`)
Already merged at `3ba1ce0e28d991800e1585b7d3533f77213939d6`:
- `-cmoe` / `--cpu-moe` and `-ncmoe` / `--n-cpu-moe` (`common/arg.cpp:2732–2747`): keep all or first $N$ layers' MoE experts on CPU host memory.
- `--moe-cache-mib N` (`common/arg.cpp:2749`): allocates GPU VRAM banks caching subsets of host-resident experts via `llama_moe_cache`.
- `--moe-cache-cpu-share F` (`common/arg.cpp:2759`): executes fraction `F` of cache misses in-place on CPU using `cpu_map` + `n_zero` dummy zero-slots in VRAM (`src/llama-moe-cache.cpp:226–266, 452`).
- **Cache Policy:** `moe_cache_lru` (`src/llama-moe-cache.cpp:25–287`) implements 8-bucket CRF ClassLRU (`crf_half_life = 128` tokens). Supports `LLAMA_MOE_CACHE_TRACE` binary trace output (`"MCT1"` format, line `642`).

## 3. Critical Gaps Remaining (Where TierFlow Adds Novel Value)
1. **Host-memory assumption (no direct unbuffered SSD streaming):** `llama_moe_cache_is_host_weight` (`src/llama-moe-cache.cpp:312–316`) requires all uncached experts to already reside in a host `ggml_backend_buffer_t` (`mmap` or RAM) and copies them from `(uint8_t *) w->data`. Models exceeding RAM hit OS `mmap` page faults and inflate OS Standby Cache.
2. **No hardware probe or DP tier planner:** `--moe-cache-mib` and `--moe-cache-cpu-share` must be hand-guessed.
3. **No pinned hot-expert tier or W-TinyLFU scan filter** over the SSD $\leftrightarrow$ RAM boundary.
4. **No cross-layer look-ahead prefetch ($L \to L+1$)** overlapping async NVMe reads with layer-$L$ attention/FFN compute.
5. **No union-aware speculative $k$-controller:** Speculative code paths exist in `common/arg.cpp` (`COMMON_SPECULATIVE_TYPE_DRAFT_MTP`, `EAGLE3`, `DFLASH`, `DSPARK`), but draft length does not model MoE expert-union bandwidth inflation.

