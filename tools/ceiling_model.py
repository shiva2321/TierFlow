"""Bandwidth-ceiling model for single-user (batch 1) LLM decoding under tiered memory.

This is an ANALYTIC MODEL, not a measurement. Every output is an upper bound
derived from the inputs below. Hardware bandwidths are ASSUMED typical values;
model architecture numbers are from public config files as recalled from memory
(UNVERIFIED -- check each against the model's config.json before relying on it).

Usage:  python tools/ceiling_model.py
"""

from dataclasses import dataclass

GB = 1e9


# ----------------------------------------------------------------------------
# Hardware profiles (ASSUMED effective bandwidths, bytes/second)
# ----------------------------------------------------------------------------
@dataclass
class Hardware:
    name: str
    vram_bytes: float
    vram_bw: float
    ram_bytes: float
    ram_bw: float
    ssd_bw: float  # effective for 1-4 MiB random reads, queue depth >= 8


MODERATE = Hardware("moderate: 16GB DDR5 + 8GB GPU + Gen4 NVMe",
                    vram_bytes=8 * GB, vram_bw=272 * GB,
                    ram_bytes=16 * GB, ram_bw=60 * GB, ssd_bw=5 * GB)
LOW = Hardware("low: 8GB DDR4, no GPU, Gen3 NVMe",
               vram_bytes=0, vram_bw=1,
               ram_bytes=8 * GB, ram_bw=35 * GB, ssd_bw=2.5 * GB)
# The user's machine (inventory read 2026-11-12): 31.7 GB RAM, RTX 3060 12 GB, KC3000 NVMe.
# Bandwidths are ASSUMED (spec-sheet class values), not probed.
THIS_PC = Hardware("this PC: 32GB DDR5 + RTX3060 12GB + KC3000 (bw assumed)",
                   vram_bytes=12 * GB, vram_bw=360 * GB,
                   ram_bytes=31.7 * GB, ram_bw=60 * GB, ssd_bw=5 * GB)


# ----------------------------------------------------------------------------
# Model descriptions. Params counted in units of parameters.
#   always_on : weights read for every token (attention, shared experts, norms,
#               router, dense FFN layers, output head)
#   routed_total / routed_active : routed-expert weights, total and per token
# ----------------------------------------------------------------------------
@dataclass
class Model:
    name: str
    always_on: float
    routed_total: float
    routed_active: float
    experts_per_layer: int = 0
    topk: int = 0

    @property
    def total(self):
        return self.always_on + self.routed_total

    @property
    def active(self):
        return self.always_on + self.routed_active


def moe(name, hidden, moe_inter, moe_layers, n_exp, topk, active_total):
    """Derive routed/always-on split from config dims (SwiGLU: 3 matrices per expert)."""
    per_expert = 3 * hidden * moe_inter
    routed_total = moe_layers * n_exp * per_expert
    routed_active = moe_layers * topk * per_expert
    return Model(name, always_on=active_total - routed_active,
                 routed_total=routed_total, routed_active=routed_active,
                 experts_per_layer=n_exp, topk=topk)


MODELS = {
    # Dense: everything is always-on.
    "llama3-70b-dense": Model("Llama-3-70B (dense)", always_on=70.6e9,
                              routed_total=0, routed_active=0),
    # Qwen3-30B-A3B: hidden 2048, moe_inter 768, 48 layers, 128 experts, top-8, ~3.3B active
    "qwen3-30b-a3b": moe("Qwen3-30B-A3B (MoE)", 2048, 768, 48, 128, 8, 3.3e9),
    # Qwen3-235B-A22B: hidden 4096, moe_inter 1536, 94 layers, 128 experts, top-8, ~22B active
    "qwen3-235b-a22b": moe("Qwen3-235B-A22B (MoE)", 4096, 1536, 94, 128, 8, 22e9),
    # DeepSeek-V3: hidden 7168, moe_inter 2048, 58 MoE layers, 256 experts, top-8, ~37B active
    "deepseek-v3": moe("DeepSeek-V3 671B (MoE)", 7168, 2048, 58, 256, 8, 37e9),
    # Kimi K3: ~2.8T total, ~104B active (search summary; split not public to me -> assume
    # 20% of active is always-on). Treated coarsely.
    "kimi-k3": Model("Kimi K3 ~2.8T (MoE, coarse)", always_on=0.2 * 104e9,
                     routed_total=2.8e12 - 0.2 * 104e9, routed_active=0.8 * 104e9,
                     experts_per_layer=896, topk=16),
}


# ----------------------------------------------------------------------------
# Core model: greedy placement + per-token time
# ----------------------------------------------------------------------------
def tokens_per_second(model, hw, bits, hit_rate, kv_reserve=1.0 * GB,
                      overlap=False, verbose=False):
    """Upper-bound tok/s.

    Placement: always-on weights go to VRAM first, then RAM, then SSD.
    Remaining fast memory caches routed experts. hit_rate = fraction of routed
    bytes per token served from RAM/VRAM cache (the rest come from SSD).
    For dense models hit_rate is ignored: what does not fit is streamed from SSD.
    overlap=False: tier times add (pessimistic). True: max of tier times (optimistic).
    """
    bpw = bits / 8.0
    vram_free = max(0.0, hw.vram_bytes - (kv_reserve if hw.vram_bytes > 0 else 0))
    ram_free = max(0.0, hw.ram_bytes - 2.0 * GB - (kv_reserve if hw.vram_bytes == 0 else 0))
    # 2 GB of RAM reserved for OS + runtime.

    t = {"vram": 0.0, "ram": 0.0, "ssd": 0.0}

    a = model.always_on * bpw
    in_vram = min(a, vram_free); vram_free -= in_vram; a -= in_vram
    in_ram = min(a, ram_free); ram_free -= in_ram; a -= in_ram
    in_ssd = a
    t["vram"] += in_vram / hw.vram_bw
    t["ram"] += in_ram / hw.ram_bw
    t["ssd"] += in_ssd / hw.ssd_bw

    r_tok = model.routed_active * bpw
    r_tot = model.routed_total * bpw
    cache_capacity = vram_free + ram_free
    fits_entirely = r_tot <= cache_capacity
    h = 1.0 if fits_entirely else hit_rate
    # Served-from-cache bytes: split between VRAM and RAM by cache share.
    if cache_capacity > 0:
        vshare = vram_free / cache_capacity
    else:
        vshare = 0.0
    hit_bytes = r_tok * h
    t["vram"] += hit_bytes * vshare / hw.vram_bw
    t["ram"] += hit_bytes * (1 - vshare) / hw.ram_bw
    miss = r_tok * (1 - h)
    t["ssd"] += miss / hw.ssd_bw
    t["ram"] += miss / hw.ram_bw  # missed expert is computed on CPU after landing in RAM

    sec = max(t.values()) if overlap else sum(t.values())
    if verbose:
        return 1.0 / sec, t, fits_entirely
    return 1.0 / sec


def expected_unique_experts(n_exp, topk, k_tokens, locality=0.0):
    """Expected distinct experts per layer touched by k tokens.

    Independent uniform routing gives n(1-(1-K/n)^k). locality in [0,1) shrinks
    the growth: an extra token reuses the previous token's experts with prob=locality.
    """
    if n_exp == 0:
        return 0.0
    eff_new = 1 + (k_tokens - 1) * (1 - locality)
    return n_exp * (1 - (1 - topk / n_exp) ** eff_new)


def spec_speedup(model, hw, bits, hit_rate, alpha, k_draft, locality=0.0,
                 draft_cost_frac=0.05):
    """Speedup from draft-and-verify vs plain decoding, bandwidth-bound regime.

    alpha: per-token acceptance probability. Accepted tokens per pass =
    (1 - alpha^(k+1)) / (1 - alpha)  (standard i.i.d. approximation).
    Verify pass cost: always-on bytes once + routed bytes x (unique experts / topk).
    Draft cost: draft_cost_frac of one plain step per drafted token.
    """
    base = 1.0 / tokens_per_second(model, hw, bits, hit_rate)
    if model.routed_active > 0:
        union = expected_unique_experts(model.experts_per_layer, model.topk,
                                        k_draft + 1, locality) / model.topk
        frac_routed = 1.0 / (1.0 + model.always_on / model.routed_active)
        # crude: scale the routed share of time by the union factor
        verify = base * ((1 - frac_routed) + frac_routed * union)
    else:
        verify = base
    accepted = (1 - alpha ** (k_draft + 1)) / (1 - alpha)
    cost = verify + k_draft * draft_cost_frac * base
    return accepted * base / cost


def selftest():
    """Known-answer checks computed independently by hand-derivable formulas."""
    hw = Hardware("t", vram_bytes=0, vram_bw=1, ram_bytes=100 * GB, ram_bw=50 * GB, ssd_bw=5 * GB)
    # 8e9 params dense at 8 bits = 8 GB, fully in RAM (98 GB usable) -> 50/8 = 6.25 tok/s
    d = Model("d", always_on=8e9, routed_total=0, routed_active=0)
    assert abs(tokens_per_second(d, hw, 8, 0.0, kv_reserve=0) - 6.25) < 1e-9
    # Uniform routing, 1 token -> exactly topk experts
    assert abs(expected_unique_experts(64, 4, 1) - 4) < 1e-9
    # Dense model, alpha -> accepted tokens; k=1, alpha=0.5, no draft cost -> 1.5x
    hw2 = hw
    s = spec_speedup(d, hw2, 8, 0.0, alpha=0.5, k_draft=1, draft_cost_frac=0.0)
    assert abs(s - 1.5) < 1e-9
    print("selftest: 3 known-answer checks passed")


def main():
    selftest()
    print("=" * 78)
    print("ANALYTIC UPPER BOUNDS (not measurements). Assumed hardware:")
    for hw in (MODERATE, LOW):
        print(f"  {hw.name}: VRAM {hw.vram_bytes/GB:.0f} GB @ {hw.vram_bw/GB:.0f} GB/s, "
              f"RAM {hw.ram_bytes/GB:.0f} GB @ {hw.ram_bw/GB:.0f} GB/s, SSD {hw.ssd_bw/GB:.1f} GB/s")
    print()

    print("1) Model sizes (derived from config dims; dims UNVERIFIED from memory)")
    for key, m in MODELS.items():
        print(f"  {m.name:34s} total {m.total/1e9:7.1f}B  active {m.active/1e9:6.1f}B  "
              f"always-on {m.always_on/1e9:5.1f}B  routed/token {m.routed_active/1e9:5.1f}B")
    print()

    print("2) Bytes touched per token at 4.5 bits/weight (Q4_K-class)")
    for key, m in MODELS.items():
        print(f"  {m.name:34s} {m.active*4.5/8/GB:7.2f} GB/token   "
              f"whole model {m.total*4.5/8/GB:8.1f} GB")
    print()

    print("3) Ceiling tok/s, MODERATE hardware, 4.5 bpw, tier times added (pessimistic) "
          "/ overlapped (optimistic)")
    for key, m in MODELS.items():
        for h in ([0.0, 0.65, 0.85, 0.95] if m.routed_active else [0.0]):
            lo, t, fits = tokens_per_second(m, MODERATE, 4.5, h, verbose=True)
            hi = tokens_per_second(m, MODERATE, 4.5, h, overlap=True)
            tag = "routed experts fit in RAM+VRAM" if (fits and m.routed_active) else f"hit={h:.2f}"
            print(f"  {m.name:34s} {tag:32s} {lo:7.3f} .. {hi:7.3f} tok/s")
            if fits and m.routed_active:
                break  # hit rate is irrelevant when everything is resident
    print()

    print("4) Same, LOW hardware (8 GB RAM, no GPU)")
    for key in ("qwen3-30b-a3b", "llama3-70b-dense", "deepseek-v3"):
        m = MODELS[key]
        for h in ([0.0, 0.65, 0.85] if m.routed_active else [0.0]):
            lo = tokens_per_second(m, LOW, 4.5, h)
            hi = tokens_per_second(m, LOW, 4.5, h, overlap=True)
            print(f"  {m.name:34s} hit={h:.2f}  {lo:7.3f} .. {hi:7.3f} tok/s")
    print()

    print("5) Check of the Gemini report's headline claims (its own assumptions, MODERATE hw)")
    # Claim A: 70B dense, 2-bit, 'top 18% hot neurons pinned', cold neurons streamed,
    # 80-90% fewer flash bytes thanks to activation sparsity. Be generous: 2.0 bpw,
    # only 10% of non-resident bytes read from SSD per token, the rest skipped.
    m = MODELS["llama3-70b-dense"]
    bytes_total = m.total * 2.0 / 8
    resident = 2.1 * GB + 0.8 * GB  # the report's stated RAM + VRAM for weights
    streamed = max(0.0, bytes_total - resident) * 0.10
    t = resident / MODERATE.ram_bw + streamed / MODERATE.ssd_bw
    print(f"  A) 70B dense @2.0 bpw: model {bytes_total/GB:.1f} GB, resident {resident/GB:.1f} GB, "
          f"SSD bytes/token (10% of rest) {streamed/GB:.2f} GB -> ceiling {1/t:.2f} tok/s "
          f"(report claims 12-25 tok/s)")
    # Claim B: 671B MoE at 2 bpw with 3.2 GB RAM + 1.4 GB VRAM resident.
    m = MODELS["deepseek-v3"]
    per_tok = m.active * 2.0 / 8
    resident = 3.2 * GB + 1.4 * GB
    for h in (0.0, 0.5, 0.8):
        always = m.always_on * 2.0 / 8
        routed = m.routed_active * 2.0 / 8
        ssd = max(0.0, always - resident) + routed * (1 - h)
        t = min(always, resident) / MODERATE.ram_bw + routed * h / MODERATE.ram_bw + ssd / MODERATE.ssd_bw
        print(f"  B) DeepSeek-V3 @2.0 bpw, {per_tok/GB:.2f} GB/token, resident {resident/GB:.1f} GB, "
              f"routed hit={h:.1f}: ceiling {1/t:.2f} tok/s")
    print()

    print("6) Draft-and-verify speedup ceiling vs. plain decoding (alpha=0.8, draft cost 5%/token)")
    for key, hwp, h in (("llama3-70b-dense", MODERATE, 0.0),
                        ("qwen3-30b-a3b", LOW, 0.65),
                        ("deepseek-v3", MODERATE, 0.65)):
        m = MODELS[key]
        row = []
        for k in (1, 2, 3, 4, 6, 8):
            row.append(f"k={k}: {spec_speedup(m, hwp, 4.5, h, 0.8, k):.2f}x")
        print(f"  {m.name:34s} no locality   " + "  ".join(row))
        if m.routed_active:
            row = [f"k={k}: {spec_speedup(m, hwp, 4.5, h, 0.8, k, locality=0.5):.2f}x"
                   for k in (1, 2, 3, 4, 6, 8)]
            print(f"  {m.name:34s} locality 0.5  " + "  ".join(row))
    print()

    print("7) Expected distinct experts per layer for k tokens (uniform routing)")
    for key in ("qwen3-30b-a3b", "deepseek-v3"):
        m = MODELS[key]
        vals = "  ".join(f"k={k}: {expected_unique_experts(m.experts_per_layer, m.topk, k):.1f}"
                         for k in (1, 2, 4, 8, 16, 50))
        print(f"  {m.name:34s} (n={m.experts_per_layer}, top-{m.topk})  {vals}")
    print()

    print("8) THIS PC, candidate test models (tier times added .. overlapped)")
    for key, bits in (("qwen3-30b-a3b", 4.5), ("qwen3-235b-a22b", 4.5),
                      ("qwen3-235b-a22b", 3.0), ("llama3-70b-dense", 4.5)):
        m = MODELS[key]
        for h in ([0.0, 0.65, 0.85] if m.routed_active else [0.0]):
            lo, t, fits = tokens_per_second(m, THIS_PC, bits, h, verbose=True)
            hi = tokens_per_second(m, THIS_PC, bits, h, overlap=True)
            tag = "fits in RAM+VRAM" if (fits and m.routed_active) else f"hit={h:.2f}"
            print(f"  {m.name:28s} {bits:.1f} bpw  file {m.total*bits/8/GB:6.1f} GB  "
                  f"{tag:16s} {lo:7.3f} .. {hi:7.3f} tok/s")
            if fits and m.routed_active:
                break


if __name__ == "__main__":
    main()
