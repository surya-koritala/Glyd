"""Output quality of Nemotron 3 Super (120B-A12B, bf16) when its routed experts are rounded to 2-4 bits (Linear GLY-131, second half; GLM-5.3-Flash's is glm_quality.py): the KL divergence of
each next-token distribution from the bf16 model's, the top token's agreement and the perplexity, over the held-out text of glm_skew.py (six domains, ~49k tokens), for the same seven
variants as glm_quality.py (ref, mix13, int4, hlm3, int3, cold50, hlm3g; same recipes, same item numbers) and, named in VARIANTS (a default run leaves them out), for its three Hummingbird ones, at most 3.0
bits/weight averaged over the experts (hlm2g: every expert hlm2 with GPTQ act-order; hb40: the 40% most-used experts of each layer hlm3g, the rest hlm2g; hb20: the 20% most-used int4, the rest hlm2g).
Beyond those, the variants of nemotron_hbr.py (GLY-98): hbr40, Hummingbird with neuron rows (the 40% most-used experts of each layer 3 bits, the rest 2, 2.99 bits/weight, one rotation of the latent a layer; the recipe
the engine ships, and --export DIR writes its codes) and hbr40p (the same with a rotation of its own for each expert).
Only the 20,480 routed experts of the 40 latent-MoE layers change; the Mamba-2 and attention layers, the latent projections, the shared experts, routers, embedding and head stay as released.
  plan   [--text]      shell lines for the launcher; with --text the variant table and its time budget (of the variants VARIANTS names, else of the default run)
  run    --weights DIR --data DIR --out DIR [--export DIR]   the pod job (GPUs): load, reference, routing counts, then one layer-by-layer pass per variant; --export: hbr40's codes (nemotron_hbr.py's format)
  report DIR           results.jsonl + meta.json -> quality.md and quality.json (numpy only; the job runs it after every variant)
  selftest             a tiny random NemotronH (Mamba-2, attention and latent-MoE layers) on the CPU: every variant end to end, the block codes checked against calib.py's; with GPUs also
                       the codes on each of them against the CPU's and the tiny model split over them as the real one is (SELFTEST_GPU=0: CPU only)
Everything but what is this model's is glm_quality.py's, imported: the block codes, the layer-by-layer pass (one layer after another over all held-out sequences, the residual streams on the
host between layers, the experts of a MoE layer read a chunk at a time, rounded and dropped), the scores, the deadline and the report. This file brings the model: how its blocks are
called, what its MoE block does around the routed experts, the experts' own arithmetic, the plan and the selftest.
What a Nemotron MoE block does (transformers 5.19.0, modeling_nemotron_h.py): the block's input x is the pre-norm of the residual stream; the router sees x at the full width (4096), a
sigmoid router with a load-balancing bias, top-22 of 512, the weights renormalised and scaled; z = fc1_latent_proj(x) is 1024 wide; each chosen expert computes down(relu2(up(z))) in
that latent width (no gate); the weighted sum goes through fc2_latent_proj back to 4096; the shared expert (a relu2 MLP, 5376 wide) on x is added. So the Hessians GPTQ needs, and the
tokens an expert is applied to, are z, while the router's choice is made on x.
Unlike GLM's, the model is released in bf16: there is no FP8 floor under the control, but the control still differs from the model's own forward (its grouped bf16 expert sums), so every variant is also scored against it. No token is used anywhere."""
import argparse, functools, json, os, sys, tempfile, time, types

import glm_quality as gq  # the block codes, the pass, scores, the job and the report
import glm_skew as gs     # the pins, the six text domains and their prep
import nemotron_hbr as hbr  # the neuron-row recipe (hbr40, hbr40p) and its export; registers its spec kinds with gq

np, torch, F = gq.np, gq.torch, gq.F  # (None without numpy / torch: plan and report need neither)

HID, LAT, INTER = 4096, 1024, 2688           # config.json: hidden_size, moe_latent_size, moe_intermediate_size
NEMO_MATS = [(INTER, LAT), (LAT, INTER)]     # rows x columns of an expert's matrices: up [I, lat] and down [lat, I]
# GLM's seven variants and its three Hummingbird ones (the same recipes and item numbers) with this model's minutes. The minutes are guesses until a first run. glm_skew.py's run on this pod class (4 x A100 80 GB) put the
# model's own forward at 1,237 tokens/s (torch Mamba-2, grouped experts): ~1 min for the 49k held-out tokens, with the streams' trips through the host ~1.5; scoring them against the 131k
# vocabulary ~1 min. The rounding is an op count: int's 16 clips are ~1 KB of memory traffic a weight, 113 TB for the 112.7 B expert weights, one card's 1.5 TB/s at a time (the layers go
# one after another) = 1.3 min and twice that in practice; hlm's sorts, 25 Lloyd steps and two Hadamard products about the same. GPTQ sends 7 x the tokens through (the calibration, ~5 more
# min) and adds the Cholesky factors and the column loops (~8 min). The routing counts for mix13 are 96 forwards at the model's speed, ~3 min.
# The first run (2026-10-10, 4 x A100 80 GB) measured hlm3g's pass at 818 s, the routing counts at 110 s and ref's pass at 39 s. hlm2g and hb20 have not run: one GPTQ call a chunk (for all experts, for the 80% hlm2g),
# so hlm3g's 14 min each. hb40 has not run: a chunk holds an hlm3g and an hlm2g group, two GPTQ calls and so two sets of column loops, so GPTQ's part of hlm3g's pass (818 s less the 168 sequences
# through the model, 7 x 39 s) twice, and the routing counts: 25 min.
# The Hummingbird run (2026-10-10, 4 x A100): hlm2g's pass 778 s, hb40's 907 s (and the routing counts 110 s), hb20's 699 s, ref's 35 s. hbr40 and hbr40p are GPTQ twice a chunk like hb40, with a full 1,024-point rotation (a matrix product),
# one row at a time through the down matrix, and for hbr40 the packing and writing of 1.05 GB a layer: guessed at hb40's pass and 3 min.
MIN = dict(ref=3, mix13=9, int4=5, hlm3=6, int3=5, cold50=6, hlm3g=20, hlm2g=14, hb40=25, hb20=14, hbr40=20, hbr40p=20)
VARIANTS = {n: dict(v, min=MIN[n]) for n, v in gq.VARIANTS.items()}
HBR = ["hbr40", "hbr40p"]  # not in the default run: named in --variants / VARIANTS
VARIANTS["hbr40"] = dict(item=10, what="Hummingbird, neuron rows, mixed by usage: the 40% most-used experts of each layer 3 bits, the rest 2; GPTQ act-order on both matrices, one rotation of the latent a layer (the export's recipe)",
                         share=40, hot="hbr3g", rest="hbr2g", min=MIN["hbr40"])
VARIANTS["hbr40p"] = dict(item=11, what="hbr40 with a rotation of the latent for each expert (what sharing the layer's costs)", share=40, hot="hbp3g", rest="hbp2g", min=MIN["hbr40p"])
VARIANTS["ref"]["what"] = "control: the bf16 experts through the rounding path unrounded (the identity: must reproduce the model)"
VARIANTS["mix13"]["what"] = "mixed by usage: the 13% most-used experts of each layer int4, the rest hlm3 (GLM's split, to compare; this model's 24 GB plan holds 32%)"
ORDER = list(gq.ORDER)
SETUP_MIN = 10   # minutes before the first pass: the skew run on this pod class took pyenv 112 s, download 206 s (in the background), load 60 s; plus the selftest, prep, check, reference


def experts_forward(e, Z, idx, wgt, specs, li, err, chunk, ctx):
    """The routed experts' output for the latent tokens Z (N, 1024) with routing idx, wgt (N, K), the experts rounded by specs: what NemotronHExperts' loop does (down(relu2(up(z))) times the
    router's weight, added in the weights' float32), with the weights read from the model's bf16 tensors a chunk of experts at a time and the same order of additions."""
    N, E = Z.shape[0], e.up_proj.shape[0]
    rt = gq.Route(idx, E)
    acc = torch.zeros(N, Z.shape[1], dtype=wgt.dtype, device=Z.device)
    ex = getattr(ctx, "export", None)
    exporting = ex is not None and ex.variant == getattr(ctx, "label", None)  # (an export needs every expert, whether or not a token was routed to it)
    for a in range(0, E, chunk):
        b = min(E, a + chunk)
        if not any(rt.cnt[a:b]) and not exporting:
            continue
        up, dn = gq.fetch(e, a, b)
        qup, qdn = gq.rounded(e, a, up, dn, specs, li, err, rt, Z, ctx)
        del up, dn
        for j in range(b - a):
            tk, ps = rt.of(a + j)
            if not len(tk):
                continue
            cur = F.linear(e.act_fn(F.linear(Z[tk], qup[j])), qdn[j]) * wgt[tk, ps, None]
            acc.index_add_(0, tk, cur.to(acc.dtype))
        del qup, qdn
    return acc.to(Z.dtype)


class NemoRun(gq.Run):
    """The model and what a pass needs of it: one residual stream [1, S, 4096] (no hyper-connections), blocks called with the mask the model builds for them, MoE blocks in the layer's
    `mixer`, the final norm `norm_f` and `lm_head`."""
    NAME = "nemotron"
    SLOT = "mixer"
    REF = "bf16"
    REF_BPW = 16.0
    CONTROL = True     # the control (experts unrounded, through this code) is KL 0.0047 from the model's own forward on the real checkpoint (2026-10-10, 4 x A100; ~1e-4 expected): the
    REF_MAX = 0.1      # grouped kernel's bf16 sums of 22 slots at a routed scale of 5 against this pass's float32, over 40 MoE layers; so each variant is also scored against the control
    KEPT = "the Mamba-2 and attention layers, the latent projections, the shared experts, the routers, embedding and head"
    CHUNK_CAP = 32     # an expert is 5.5 M weights (GLM's 25 M), so twice as many fit at a time, and GPTQ's column loops, which are bound by kernel launches, run half as often
    VARIANTS = VARIANTS
    ORDER = ORDER
    SETUP_MIN = SETUP_MIN

    def __init__(self, model):
        self.m, self.layers = model, model.model.layers
        self.moe = [i for i, l in enumerate(self.layers) if l.block_type == "moe"]
        self.E = self.block(self.moe[0]).experts.up_proj.shape[0]
        self.K = model.config.num_experts_per_tok
        self.eager = False  # (whether the model's own experts were switched to the eager loop: see robust)

    @property
    def width(self):
        return self.m.config.hidden_size

    @property
    def dev0(self):
        return self.m.model.embeddings.weight.device

    @property
    def head_dev(self):
        return self.m.model.norm_f.weight.device

    def embed(self, x):
        return self.m.model.embeddings(torch.from_numpy(x).long()[None].to(self.dev0)).cpu()

    def step(self, layer, h, carry, S, dev):
        """The block on the stream h (on dev, S positions) as NemotronHModel.forward calls it, with position ids and, for an attention block only, the causal mask the model builds (the Mamba-2
        blocks' padding mask is None for a batch without padding): the stream it gives back; nothing is carried on to the next block."""
        from transformers.masking_utils import create_causal_mask
        pos = torch.arange(S, device=dev)[None]
        mask = create_causal_mask(config=self.m.config, inputs_embeds=h, attention_mask=None, past_key_values=None, position_ids=pos) if layer.block_type == "full_attention" else None
        return layer(h, attention_mask=mask, position_ids=pos), None

    def normed(self, h):
        return self.m.model.norm_f(h.to(self.head_dev))

    def expert_input(self, moe, X, S):
        """z = fc1_latent_proj(x), a sequence at a time as the block calls it: what the experts take (and what GPTQ's Hessians are of); the router already chose on x."""
        return torch.cat([moe.fc1_latent_proj(X[i * S:(i + 1) * S]) for i in range(X.shape[0] // S)])

    def routed(self, e, Z, idx, w, specs, li, err, chunk, ctx):
        out = experts_forward(e, Z, idx, w, specs, li, err, chunk, ctx)
        ex = getattr(ctx, "export", None)
        if ex is not None and ex.variant == getattr(ctx, "label", None):
            ex.flush(li, self.moe, self.E)  # (the layer's experts are all in: its file is written; a retry after an out-of-memory never gets here)
        return out

    def finish(self, moe, routed, X, S):
        """fc2_latent_proj of the routed sum, plus the shared expert on x: the block's output a sequence at a time, as NemotronHMoE.forward adds them."""
        return [moe.fc2_latent_proj(routed[i * S:(i + 1) * S]).view(1, S, -1) + moe.shared_experts(X[i * S:(i + 1) * S].view(1, S, -1)) for i in range(X.shape[0] // S)]


def chosen():
    """The variants the job runs, in order: the launcher's VARIANTS (a comma list, which the job takes too) if set, else the default ORDER."""
    names = [n.strip() for n in os.environ.get("VARIANTS", "").split(",") if n.strip()] or ORDER
    bad = [n for n in names if n not in VARIANTS]
    if bad:
        sys.exit(f"VARIANTS {bad}: {list(VARIANTS)}")
    return names


def export_gb(share, E, L):
    """GB of hbr40's export of L layers of E experts: k = share% of a layer's experts at 3 bits, the rest at 2, each expert's two matrices as nemotron_hbr.py writes them (codes, scales, codebook)."""
    k = max(1, round(share * E / 100))
    eb = lambda bits: 2 * INTER * (LAT * bits // 8 + 2 * (LAT // gq.BLOCK) + 2 * (1 << bits))
    return L * (k * eb(3) + (E - k) * eb(2)) / 1e9


def plan(a):
    m = gs.MODELS["nemotron"]
    E, L = m["ne"], m["n_moe"]
    order = chosen()
    if a.text:
        jmm, n_w = int(os.environ.get("JOB_MAX_MIN", 128)), sum(r * c for r, c in NEMO_MATS)
        print(f"{L} MoE layers x {E} experts, top-{m['k']}; an expert is up [{INTER}, {LAT}] + down [{LAT}, {INTER}] on the {LAT}-wide latent z = fc1_latent_proj(x) of the {HID}-wide x the router sees = {n_w:,} weights; "
              f"{L * E * n_w / 1e9:.1f} B routed-expert weights")
        print("variants in the order they run (item = the GLY-127 list; bits/weight all in). The minutes are guesses, not measurements: the skew run's 1,237 tokens/s through the model, 60 s load, and an op count of the rounding.")
        print(f"The job measures its own speed and starts no pass that would not end 2 min before JOB_MAX_MIN {jmm}. The setup before the first (venv, download, pyenv, prep, selftest, load, check, bf16 reference) is ~{SETUP_MIN} min:")
        total = gq.plan_rows(NemoRun, NEMO_MATS, L, E, jmm, order)
        print(f"  total ~{total} min of passes after the setup (the routing counts for the mixed variants, 110 s on the first run, are inside the minutes of whichever of them runs first)")
        if "hbr40" in order:
            print(f"  --export DIR (EXPORT=1 in the launcher) writes hbr40's codes while its pass runs: {export_gb(VARIANTS['hbr40']['share'], E, L):.1f} GB in {L} files (one a MoE layer, ~{export_gb(VARIANTS['hbr40']['share'], E, 1):.2f} GB each) and meta.json")
        return
    gs.plan(argparse.Namespace(model="nemotron"))  # REPO REV POD_CLASS DISK K NE N_MOE CACHE_PCTS WEIGHTS_GB
    print(f'ORDER="{" ".join(order)}"')
    print(f'SETUP_MIN="{SETUP_MIN}"')
    print(f'PASSES_MIN="{sum(VARIANTS[n]["min"] for n in order)}"')
    print(f'EXPORT_GB="{export_gb(VARIANTS["hbr40"]["share"], E, L):.1f}"')


# -------------------------------------------------------------------------------------------------------------------------------------- selftest
def tiny_model(seed=0):
    """A random NemotronH of 8 layers as the real one's pattern, M E M E * E M E (3 Mamba-2, 1 attention, 4 latent-MoE: 16 experts, top-4, latent 64, intermediate 192, a shared expert),
    weights scaled so that rounding an expert is felt."""
    from transformers import NemotronHConfig, NemotronHForCausalLM
    torch.manual_seed(seed)
    cfg = NemotronHConfig(
        vocab_size=256, hidden_size=128, layers_block_type=["linear_attention", "moe", "linear_attention", "moe", "full_attention", "moe", "linear_attention", "moe"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, intermediate_size=256, mamba_num_heads=8, mamba_head_dim=32, ssm_state_size=16, n_groups=2, chunk_size=64,
        n_routed_experts=16, num_experts_per_tok=4, moe_intermediate_size=192, moe_shared_expert_intermediate_size=128, moe_latent_size=64, routed_scaling_factor=2.5,
        pad_token_id=0, eos_token_id=[1], bos_token_id=1)
    m = NemotronHForCausalLM(cfg).to(torch.bfloat16).eval()
    m.set_experts_implementation("eager")
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith(("experts.up_proj", "experts.down_proj")):
                p.copy_(torch.randn_like(p) * 0.06)
            elif n.endswith(("fc1_latent_proj.weight", "fc2_latent_proj.weight")):
                p.copy_(torch.randn_like(p) * 0.088)
            elif n.endswith("gate.weight"):
                p.copy_(torch.randn_like(p) * 0.05)
            elif n.endswith("embeddings.weight"):
                p.copy_(torch.randn_like(p))
            elif n == "lm_head.weight":
                p.copy_(torch.randn_like(p) * 0.18)
            elif "shared_experts" in n:
                p.mul_(0.5)
    return m


def device_map(nd, layers=8):
    """The tiny model's layers spread over nd GPUs, the head on the last: the shape of the real model's map (which "auto" makes for 247 GB, but would keep a tiny model on one card)."""
    return {"model.embeddings": 0, **{f"model.layers.{i}": i * nd // layers for i in range(layers)}, "model.norm_f": nd - 1, "lm_head": nd - 1}


def selftest(_=None):
    import shutil
    for pkg in ("fla", "causal_conv1d", "mamba_ssm"):
        sys.modules[pkg] = None  # (the tiny model is on the CPU: with these installed, transformers binds their Triton / CUDA kernels when the model code is imported, and they take no CPU tensors)
    from transformers.masking_utils import create_causal_mask  # noqa: F401 (the pass imports it when it runs: fail here if this version has moved it)
    torch.manual_seed(0)
    n_cuda = gq.cuda_check([f"cuda:{i}" for i in range(torch.cuda.device_count())]) if torch.cuda.is_available() and os.environ.get("SELFTEST_GPU", "1") != "0" else 0
    assert gq.cuda_check(["cpu"]) == 1  # (the check itself: the CPU against itself)
    gq.codes_selftest()  # the block codes against calib.py's own functions (the same code GLM's selftest runs)
    hbr.selftest()  # the neuron-row recipe: rotation, packing, rounding against gq's and against least squares, the export
    if n_cuda:
        hbr.cuda_check([f"cuda:{i}" for i in range(torch.cuda.device_count())])
    assert hbr.cuda_check(["cpu"]) == 1  # (the check itself: the CPU against itself)

    # ---- specs, bits: this model's matrices (up [2688, 1024], down [1024, 2688]; the codebook and the rotation's signs cost more of a narrower row)
    bpw = lambda s: gq.spec_bpw(s, NEMO_MATS, NemoRun.REF_BPW)
    assert abs(bpw("int4") - 4.5) < 1e-9 and abs(bpw("int3") - 3.5) < 1e-9 and bpw("ref") == 16.0 and 3.58 < bpw("hlm3") < 3.59 and 2.5 < bpw("hlm2") < 2.6
    assert abs(bpw("hlm3g") - bpw("hlm3")) < 1e-12 and abs(gq.variant_bpw(VARIANTS["mix13"], NEMO_MATS, 16.0) - (0.13 * 4.5 + 0.87 * bpw("hlm3"))) < 1e-9
    assert set(ORDER) | set(gq.HUMMINGBIRD) | set(HBR) == set(VARIANTS) and set(gq.VARIANTS) <= set(VARIANTS) and all(all(VARIANTS[n][k] == gq.VARIANTS[n][k] for k in ("item", "share", "hot", "rest")) for n in gq.VARIANTS), "not GLM's recipes"
    # Hummingbird at this model's shapes: hlm2 is 2 bits of codes + 1/2 of scale + 64/cols of codebook (4 entries of 16 bits a row) + 1/rows of the rotation's signs = 2.5629 (up, 2688 x 1024) and 2.5248 (down, 1024 x 2688), 2.5438 over an expert
    assert abs(bpw("hlm2g") - 2.5438) < 1e-4 and bpw("hlm2g") == bpw("hlm2")
    hb = {n: gq.variant_bpw(VARIANTS[n], NEMO_MATS, 16.0) for n in gq.HUMMINGBIRD}
    assert abs(hb["hb40"] - (0.4 * bpw("hlm3g") + 0.6 * bpw("hlm2g"))) < 1e-9 and abs(hb["hb20"] - (0.2 * 4.5 + 0.8 * bpw("hlm2g"))) < 1e-9 and max(hb.values()) <= 3.0, hb
    for n, k in (("hb40", 205), ("hb20", 102)):  # (the hot experts of a layer of 512: round(0.4 * 512), round(0.2 * 512))
        sp = gq.layer_specs(VARIANTS[n], [e < k for e in range(512)], 512)
        assert sp.count(VARIANTS[n]["hot"]) == k and sum(bpw(x) for x in sp) / 512 <= 3.0, n
    assert gq.layer_specs(VARIANTS["hlm2g"], None, 3) == ["hlm2g"] * 3 and gq.layer_specs(VARIANTS["hb40"], [True, False, False, True], 4) == ["hlm3g", "hlm2g", "hlm2g", "hlm3g"]
    assert gq.layer_specs(VARIANTS["hb20"], [False, True, False], 3) == ["hlm2g", "int4", "hlm2g"]
    # Hummingbird with neuron rows: both matrices are 2,688 rows of 1,024 values, so 3 + 16/32 + 8 * 16/1024 = 3.625 and 2 + 16/32 + 4 * 16/1024 = 2.5625 bits a weight, hbr40 0.4 * 3.625 + 0.6 * 2.5625 = 2.9875 (the 205 of 512
    # experts a layer that round(0.4 * 512) makes hot: 2.98792), a rotation an expert 1,024 sign bits an expert more
    hr = {n: gq.variant_bpw(VARIANTS[n], NEMO_MATS, 16.0) for n in HBR}
    assert abs(bpw("hbr3g") - 3.625) < 1e-12 and abs(bpw("hbr2g") - 2.5625) < 1e-12 and bpw("hbr2") == bpw("hbr2g") and abs(hr["hbr40"] - 2.9875) < 1e-12 and abs(hr["hbr40p"] - 2.9875 - 1024 / 5505024) < 1e-12 and max(hr.values()) <= 3.0, hr
    sp = gq.layer_specs(VARIANTS["hbr40"], [e < 205 for e in range(512)], 512)
    assert sp.count("hbr3g") == 205 and abs(sum(bpw(x) for x in sp) / 512 - (2.5625 + 205 / 512 * 1.0625)) < 1e-12 and sum(bpw(x) for x in sp) / 512 <= 3.0
    assert gq.parse_spec("hbr3g") == ("hbr", 3, True) and gq.parse_spec("hbp2") == ("hbp", 2, False) and abs(export_gb(40, 512, 40) - 42.1) < 0.05
    for bad in ("hbr", "int3g", "hbx3"):
        try:
            gq.parse_spec(bad)
            raise AssertionError(bad)
        except ValueError:
            pass

    # ---- the block codes at the real expert shapes: kept, finite, the ladder's errors (int4 0.0918, int3 0.1965 of gaussian weights), GPTQ no worse than nearest under its Hessian
    g = torch.Generator().manual_seed(7)
    for shp in ((2, INTER, LAT), (2, LAT, INTER)):
        W = (torch.randn(*shp, generator=g) * 0.05).bfloat16().float()
        rel = lambda q: float(((q.float() - W) ** 2).sum() / (W ** 2).sum()) ** 0.5
        q4, q3, h3 = gq.quant_int(W, 4), gq.quant_int(W, 3), gq.quant_hlm(W, 3, 5)
        assert q4.shape == q3.shape == h3.shape == W.shape and torch.isfinite(h3.float()).all() and 0.085 < rel(q4) < 0.10 and 0.19 < rel(q3) < 0.21 and rel(h3) < rel(q3), (shp, rel(q4), rel(q3), rel(h3))
        X = torch.randn(3072, shp[2], generator=g) * (1 + 4.0 * (torch.arange(shp[2]) % 17 == 0))
        H = X.T @ X / 3072
        q, ok = gq.gptq_hlm(W.clone(), [H.clone(), H.clone()], 3, 7)
        herr = lambda dq, j: float((((dq.float() - W[j]) @ H) * (dq.float() - W[j])).sum())
        assert ok.all() and all(herr(q[j], j) <= herr(h3[j], j) for j in range(2)), shp

    # ---- the experts' arithmetic is NemotronHExperts': the same bits, whatever the chunk
    m = tiny_model()
    e = m.model.layers[1].mixer.experts
    assert not e.has_gate and gq.expert_names(e) == ("up_proj", "down_proj") and NemoRun(m).mats() == [(192, 64), (64, 192)]
    Zt = torch.randn(40, 64).bfloat16()
    w_, idx_ = torch.rand(40, 4), torch.stack([torch.randperm(16)[:4] for _ in range(40)])
    with torch.inference_mode():
        want = e(Zt, idx_, w_)
        for chunk in (1, 3, 16):
            assert torch.equal(experts_forward(e, Zt, idx_, w_, ["ref"] * 16, 0, gq.Err(), chunk, types.SimpleNamespace()), want), ("NemotronHExperts arithmetic", chunk)
    assert not torch.equal(experts_forward(e, Zt, idx_, w_, ["int3"] * 16, 0, gq.Err(), 16, types.SimpleNamespace()), want), "int3 left the experts alone"
    # an export has every expert of the layer, those no token was routed to too (here 12 of 16 get none)
    dx = tempfile.mkdtemp()
    ex_ = hbr.Exporter(dx, "hbr40", "repo", "rev")
    ctx_ = types.SimpleNamespace(ncal=0, min_tok=1 << 30, done=0, fell=0, label="hbr40", export=ex_)
    with torch.inference_mode():
        experts_forward(e, Zt, torch.stack([torch.randperm(4) for _ in range(40)]), w_, ["hbr2g"] * 16, 3, gq.Err(), 4, ctx_)
    assert sorted(ex_.buf[3]) == list(range(16)) and ctx_.fell == 16, (sorted(ex_.buf[3]), ctx_.fell)
    ex_.flush(3, [3], 16)
    assert hbr.check_export(dx)["complete"]
    shutil.rmtree(dx)

    # ---- the whole job on a tiny random model: every variant, and the checks around it
    q_extra = {f"t_{s}": dict(item=-1, what=f"test: every expert {s}", share=None, hot=None, rest=s, min=1) for s in ("int5", "int2", "hlm4", "hlm2", "hbr2", "hbr2g", "hbr3g")}
    VARIANTS.update(q_extra)
    S, d, out = 128, tempfile.mkdtemp(), tempfile.mkdtemp()
    rng = np.random.default_rng(0)
    for dom in gs.DOMAINS:
        np.save(f"{d}/held_{dom}.npy", rng.integers(2, 256, (1, S)).astype(np.int32))
        np.save(f"{d}/calib_{dom}.npy", rng.integers(2, 256, (3, S)).astype(np.int32))
    names = ORDER + gq.HUMMINGBIRD + HBR + list(q_extra)
    ns = lambda **kw: types.SimpleNamespace(data=d, out=out, variants=names, held_seqs=1, count_seqs=2, gptq_seqs=2, gptq_min_tok=300, chunk=0, deadline=0, **kw)
    with torch.inference_mode():
        # the layer-by-layer pass with the model's own experts and the head: logits to the bit
        ids = np.stack([np.load(f"{d}/held_{dom}.npy")[0] for dom in gs.DOMAINS[:2]])
        run = NemoRun(m)
        assert run.moe == [1, 3, 5, 7] and run.E == 16 and run.K == 4 and run.width == 128
        hs, _, _ = gq.layerwise(run, ids)
        bits = []
        for h, x in zip(hs, ids):  # (the same to the bit where it was written; the tolerance, a bf16 step of the logits, is for a BLAS that orders a product differently for another number of rows)
            a_, b_ = run.m.lm_head(run.normed(h))[0].float(), m(input_ids=torch.from_numpy(x).long()[None], use_cache=False).logits[0].float()
            assert (a_ - b_).abs().max() <= 0.07, "layer-by-layer differs from the forward"
            bits.append(torch.equal(a_, b_))
        # the pass hands the experts what they take in the model's own forward: z (not x) and the router's choice made on x
        seen, cap, hooks = {}, {}, []

        class Spy(NemoRun):
            def routed(self, e, Z, idx, w, specs, li, err, chunk, ctx):
                seen[li] = (Z.clone(), idx.clone(), w.clone())
                return super().routed(e, Z, idx, w, specs, li, err, chunk, ctx)
        for li in run.moe:
            def on_experts(mod, args_, li=li):
                cap.setdefault(li, {})["z"] = args_[0].clone()

            def on_router(mod, args_, out_, li=li):
                cap.setdefault(li, {}).update(x=args_[0].clone(), route=(out_[2].clone(), out_[1].clone()))
            mo = m.model.layers[li].mixer
            hooks += [mo.experts.register_forward_pre_hook(on_experts), mo.gate.register_forward_hook(on_router)]
        m(input_ids=torch.from_numpy(ids[0]).long()[None], use_cache=False)
        for hk in hooks:
            hk.remove()
        gq.layerwise(Spy(m), ids[:1], lambda li: ["ref"] * run.E)
        for li in run.moe:
            z, idx, w = seen[li]
            assert z.shape[1] == 64 and cap[li]["z"].shape[1] == 64 and cap[li]["x"].shape[-1] == 128, "the experts take the 64-wide latent, the router the 128-wide input"
            assert torch.equal(z, cap[li]["z"]) and torch.equal(idx, cap[li]["route"][0]) and torch.equal(w, cap[li]["route"][1]), f"layer {li}: the pass hands the experts something else than the model's"
        try:
            gq.layerwise(run, ids, deadline=1, label="late")
            raise AssertionError("no deadline")
        except gq.Deadline:
            pass
        try:  # (a pass that cannot end before the deadline says so as soon as it knows, not at the deadline: here the 90 s kept for scoring alone are more than the 60 s left)
            gq.layerwise(run, ids, lambda li: ["int4"] * run.E, deadline=time.time() + 60, label="slow", args=types.SimpleNamespace(chunk=0, gptq_min_tok=0))
            raise AssertionError("no projection")
        except gq.Deadline as ex:
            assert "would end" in str(ex), ex
        # the routing counts are the router's choices, and glm_skew.py's analyze reads them
        held_d, cal_d = gq.load_data(d, 3, 1)
        cdir = tempfile.mkdtemp()
        counts = gq.count_routing(NemoRun(m), cal_d, held_d, cdir, 2)
        assert counts[0].sum() == 6 * 2 * S * run.K * len(run.moe) and counts[1].sum() == 6 * 1 * S * run.K * len(run.moe), "counts"
        by_hook = {}  # (the held-out wiki sequence through a plain forward, counted by a hook of our own on each router)
        hooks = [m.model.layers[li].mixer.gate.register_forward_hook(lambda mod, i, o, li=li: by_hook.__setitem__(li, np.bincount(o[2].reshape(-1).numpy(), minlength=16))) for li in run.moe]
        m(input_ids=torch.from_numpy(held_d["wiki"][0]).long()[None], use_cache=False)
        for hk in hooks:
            hk.remove()
        first = np.load(f"{cdir}/counts/held_wiki.npz")
        assert (first["counts"] == np.stack([by_hook[li] for li in run.moe])).all() and list(first["layers"]) == run.moe, "count_routing counts something else than the router chose"
        # an out-of-memory in the experts halves the chunk and changes nothing
        real, fired = globals()["experts_forward"], []

        def flaky(e, Z, idx, w, specs, li, err, chunk, ctx):
            if not fired:
                fired.append(chunk)
                raise torch.cuda.OutOfMemoryError("simulated")
            fired.append(chunk)
            return real(e, Z, idx, w, specs, li, err, chunk, ctx)
        a1 = ns()
        a1.variants = ["ref", "int4"]
        shutil.rmtree(out)
        os.makedirs(out)
        gq.run_all(m, a1, NemoRun)
        kl_plain = {r["variant"]: r for r in map(json.loads, open(f"{out}/results.jsonl"))}["int4"]["kl"]
        globals()["experts_forward"] = flaky
        try:
            shutil.rmtree(out)
            os.makedirs(out)
            a1.chunk = 8
            gq.run_all(m, a1, NemoRun)
        finally:
            globals()["experts_forward"] = real
        assert fired[0] == 8 and fired[1] == 4, fired
        assert abs({r["variant"]: r for r in map(json.loads, open(f"{out}/results.jsonl"))}["int4"]["kl"] - kl_plain) < 1e-5, "a smaller chunk changed the result"
        # the full job, with hbr40's codes exported; the weights the pass computes with are kept to read the export back against
        shutil.rmtree(out)
        os.makedirs(out)
        dexp = tempfile.mkdtemp()
        exp = hbr.Exporter(dexp, "hbr40", gs.MODELS["nemotron"]["repo"], gs.MODELS["nemotron"]["rev"], keep=True)
        used, real_round = {}, gq.ROWS["hbr"]

        def spy(e_, up_, dn_, ids_, X_, rt_, ctx_, li_, bits_, g_, kind_):
            qu, qd = real_round(e_, up_, dn_, ids_, X_, rt_, ctx_, li_, bits_, g_, kind_)
            if ctx_.export is not None and ctx_.export.variant == ctx_.label:
                used.update({(li_, i): (qu[j].clone(), qd[j].clone()) for j, i in enumerate(ids_)})
            return qu, qd
        gq.ROWS["hbr"] = spy
        t0 = time.time()
        try:
            gq.run_all(m, ns(exporter=exp), NemoRun)
        finally:
            gq.ROWS["hbr"] = real_round
        secs = time.time() - t0
        # time left: a variant that would not end in time is skipped, one cut off by the deadline is recorded
        a2 = ns()
        a2.out, a2.variants, a2.deadline = tempfile.mkdtemp(), ["int4", "int3"], time.time() + 3600
        VARIANTS["int4"]["min"], old = 600, VARIANTS["int4"]["min"]
        try:
            gq.run_all(m, a2, NemoRun)
        finally:
            VARIANTS["int4"]["min"] = old
        rows2 = {r["variant"]: r for r in map(json.loads, open(f"{a2.out}/results.jsonl"))}
        assert "skipped" in rows2["int4"] and "kl" in rows2["int3"], rows2
        # the model's own experts summed in another order (batched_mm: the grouped kernel's reshape and sum, not the loop's index_add_): ref differs from the model by float additions only
        m.set_experts_implementation("batched_mm")
        a3 = ns()
        a3.out, a3.variants = tempfile.mkdtemp(), ["ref", "int4"]
        try:
            gq.run_all(m, a3, NemoRun)
        finally:
            m.set_experts_implementation("eager")
        rows3 = {r["variant"]: r for r in map(json.loads, open(f"{a3.out}/results.jsonl"))}
        assert rows3["base"]["driver_check"]["kl"] < 1e-6 and rows3["ref"]["kl"] < 0.1 * rows3["int4"]["kl"] and rows3["ref"]["kl"] < NemoRun.REF_MAX, rows3  # (a tenth of the least rounding measured)
    # the tiny model as the pod loads the real one: saved (as the checkpoint's own layout: backbone.*, one tensor per expert), then from_pretrained with a device_map over the GPUs
    # (accelerate hooks, layers on different cards), through the same job
    from transformers import NemotronHForCausalLM
    ck, out_g = tempfile.mkdtemp(), tempfile.mkdtemp()
    tiny_model().save_pretrained(ck)
    from safetensors import safe_open
    with safe_open(f"{ck}/model.safetensors", "pt") as sf:
        assert "backbone.layers.1.mixer.experts.0.up_proj.weight" in sf.keys() and "backbone.layers.1.mixer.fc1_latent_proj.weight" in sf.keys(), "not the checkpoint's layout"
    nd = n_cuda
    dm = device_map(nd) if nd else None
    covered = lambda dmap: all(any(n == k or n.startswith(k + ".") for k in dmap) for n, _ in list(m.named_parameters()) + list(m.named_buffers()))
    assert covered(device_map(4)) and covered(device_map(1)), "the device map leaves a tensor out"
    if nd > 1:
        gq.log(f"direct GPU-to-GPU copies that came back wrong: {gq.host_hops() or 'none'} (the job's copies go through the host)")
    mg = NemotronHForCausalLM.from_pretrained(ck, dtype=torch.bfloat16, **({"device_map": dm} if dm else {})).eval()
    mg.set_experts_implementation("eager")
    if not nd:  # (on the CPU the reloaded model is the same model to the bit; on a GPU a product may round differently)
        with torch.inference_mode():
            x0 = torch.from_numpy(ids[0]).long()[None]
            assert torch.equal(mg(input_ids=x0, use_cache=False).logits, m(input_ids=x0, use_cache=False).logits), "the saved and reloaded model is another model"
    ag = ns()
    ag.out, ag.variants = out_g, ["ref", "int4", "mix13", "hlm3g", "hb40"]
    with torch.inference_mode():
        gq.run_all(mg, ag, NemoRun)
    rowsg = {r["variant"]: r for r in map(json.loads, open(f"{out_g}/results.jsonl"))}
    devs = {str(NemoRun(mg).dev(li)) for li in range(8)}
    # (loose: on a GPU the same product for another number of rows may round differently, and this model is chaotic; what is caught is a path that does not work)
    assert rowsg["base"]["driver_check"]["kl"] < 5e-3 and rowsg["ref"]["kl"] < 0.5 * rowsg["int4"]["kl"] and 0 < rowsg["int4"]["kl"] and "kl" in rowsg["mix13"] and "kl" in rowsg["hlm3g"] and "kl" in rowsg["hb40"], (rowsg, devs)
    assert len(devs) >= min(2, n_cuda), f"the tiny model sits on {devs}: the device_map did not split it"
    rows = {r["variant"]: r for r in map(json.loads, open(f"{out}/results.jsonl"))}
    assert set(rows) == {"base"} | set(names), set(names) - set(rows)
    kl = {n: rows[n]["kl"] for n in names}
    # the identity variant: KL 0 (exactly, where run; the bound is for a BLAS that orders a product differently), the same top token and perplexity
    assert kl["ref"] < 1e-6 and rows["ref"]["top1"] > 0.99 and abs(rows["ref"]["ppl"] / rows["base"]["ppl"] - 1) < 1e-4 and np.load(f"{out}/perpos/ref.npz")["kl"].max() < 1e-4, "the identity variant is not the identity"
    assert rows["base"]["driver_check"]["kl"] < 1e-6 and rows["base"]["driver_check"]["top1"] > 0.99
    assert 0 < kl["t_int5"] < kl["int4"] < kl["int3"] < kl["t_int2"], kl
    assert 0 < kl["t_hlm4"] < kl["hlm3"] < kl["t_hlm2"], kl
    assert kl["hlm3"] < kl["int3"] and kl["int4"] < kl["hlm3"] and kl["mix13"] < kl["hlm3"] and kl["int4"] < kl["cold50"], kl
    assert all(abs(rows[n]["kl_ctl"] - rows[n]["kl"]) < 1e-6 and abs(rows[n]["top1_ctl"] - rows[n]["top1"]) < 1e-9 for n in names if n != "ref"), "here the control is the model to the bit: the same numbers"
    assert "kl_ctl" not in rows["ref"] and "KL divergence from the control" in open(f"{out}/quality.md").read()
    assert kl["hlm3g"] < kl["hlm3"] * 1.05, kl
    # Hummingbird: GPTQ at 2 bits beats nearest; the hot experts' extra bits help (hb40, hb20 < hlm2g) and the rest's missing ones cost (hlm3g < hb40, int4 < hb20)
    assert kl["hlm2g"] < kl["t_hlm2"] and kl["hb40"] < kl["hlm2g"] and kl["hb20"] < kl["hlm2g"] and kl["hlm3g"] < kl["hb40"] and kl["int4"] < kl["hb20"], kl
    # Hummingbird with neuron rows: GPTQ at 2 bits beats nearest, the hot experts' third bit helps, and a rotation an expert is the same recipe (between the same two)
    assert kl["t_hbr2g"] < kl["t_hbr2"] and kl["hbr40"] < kl["t_hbr2g"] and kl["t_hbr3g"] < kl["hbr40"] and kl["hbr40p"] < kl["t_hbr2g"] and kl["t_hbr3g"] < kl["hbr40p"], kl
    gp = rows["hlm3g"]["gptq"]
    assert gp["experts_gptq"] > 0 and gp["experts_rtn_fallback"] > 0 and gp["experts_gptq"] + gp["experts_rtn_fallback"] == 4 * 16, gp
    gh = {n: rows[n]["gptq"] for n in gq.HUMMINGBIRD}  # (GPTQ's experts over the 4 MoE layers: all 16 of hlm2g's and of hb40's (6 hlm3g, 10 hlm2g), the 13 hlm2g of hb20's)
    assert all(x["experts_gptq"] > 0 for x in gh.values()) and [x["experts_gptq"] + x["experts_rtn_fallback"] for x in gh.values()] == [4 * 16, 4 * 16, 4 * 13], gh
    gr = {n: rows[n]["gptq"] for n in HBR}  # (4 layers x 16 experts: GPTQ or the fallback for each)
    assert all(x["experts_gptq"] > 0 and x["experts_gptq"] + x["experts_rtn_fallback"] == 4 * 16 for x in gr.values()), gr
    we = {n: rows[n]["weight_error"] for n in ("int4", "int3", "hlm3", "mix13", "cold50", "hb40", "hb20", "hbr40", "hbr40p")}
    assert 0.085 < we["int4"]["int4"]["rel_rmse"] < 0.10 and 0.19 < we["int3"]["int3"]["rel_rmse"] < 0.21 and we["hlm3"]["hlm3"]["rel_rmse"] < we["int3"]["int3"]["rel_rmse"], we  # (the ladder's: 0.0918 and 0.1965)
    assert we["mix13"]["int4"]["weights"] == 2 * 4 * 24_576 and we["mix13"]["hlm3"]["weights"] == 14 * 4 * 24_576 and we["cold50"]["hlm2"]["weights"] == 8 * 4 * 24_576, we
    assert we["hb40"]["hlm3g"]["weights"] == 6 * 4 * 24_576 and we["hb40"]["hlm2g"]["weights"] == 10 * 4 * 24_576 and we["hb20"]["int4"]["weights"] == 3 * 4 * 24_576 and we["hb20"]["hlm2g"]["weights"] == 13 * 4 * 24_576, we
    assert we["hbr40"]["hbr3g"]["weights"] == 6 * 4 * 24_576 and we["hbr40"]["hbr2g"]["weights"] == 10 * 4 * 24_576 and we["hbr40p"]["hbp3g"]["weights"] == 6 * 4 * 24_576 and we["hbr40p"]["hbp2g"]["weights"] == 10 * 4 * 24_576, we
    assert we["hbr40"]["hbr3g"]["rel_rmse"] < we["hbr40"]["hbr2g"]["rel_rmse"], we["hbr40"]
    # the export of hbr40: read back (from the files) bit for bit what the job computed, the saved signs undo to the weights the pass used, every layer there
    summ = hbr.check_export(dexp, expect=exp.keep, hash_files=True)
    mx = json.load(open(f"{dexp}/meta.json"))
    assert summ["complete"] and summ["layers"] == 4 and summ["matrices_read"] == 4 * 16 * 2 and summ["signs_regenerate_with_this_torch"] and mx["recipe"] == "hbr40" and mx["model"]["revision"] == gs.MODELS["nemotron"]["rev"], (summ, mx["model"])
    assert [r["block"] for r in mx["layers"]] == run.moe and all(sorted(r["expert_bits"]) == [2] * 10 + [3] * 6 and r["seed"] == hbr.layer_seed(r["block"]) for r in mx["layers"]) and mx["rows"] == 192 and mx["row_length"] == 64, mx["layers"][0]["expert_bits"]
    for rec in mx["layers"]:
        sg = torch.tensor(rec["signs_u8"], dtype=torch.float32) * 2 - 1
        for (e_, m_), t_ in hbr.read_layer(dexp, rec, 192, 64).items():
            w_dq = hbr.unrot(hbr.dequant(t_["codes"], t_["scales"], t_["codebook"]), sg).bfloat16()
            want_w = used[(rec["block"], e_)][0 if m_ == "up" else 1]
            assert torch.equal(w_dq if m_ == "up" else w_dq.T, want_w), (rec["block"], e_, m_)
    exp_bytes = sum(r["bytes"] for r in mx["layers"])
    meta = json.load(open(f"{out}/meta.json"))
    assert meta["ref"] == "bf16" and meta["control"] is True and meta["expert_weights"] == 24_576 and meta["moe_layers"] == 4 and meta["repo"] == gs.MODELS["nemotron"]["repo"], meta
    assert meta["hot"]["13"]["k"] == 2 and meta["hot"]["50"]["k"] == 8 and 0.1 < meta["hot"]["13"]["held"] < meta["hot"]["50"]["held"] <= 1, meta["hot"]
    assert meta["hot"]["20"]["k"] == 3 and meta["hot"]["40"]["k"] == 6, meta["hot"]
    md = open(f"{out}/quality.md").read()
    assert all(f"| {n} |" in md for n in names) and "### KL divergence from bf16, mean, by domain" in md and "fell back to nearest" in md and json.load(open(f"{out}/quality.json"))["meta"]["repo"] == meta["repo"]
    assert "FP8" not in md and "KL from control" in md and "top-1 = control" in md and "the experts as released, run through the same path unrounded" in md, "the report speaks of FP8, or has no control column"
    # what quality.md says when things went wrong: a control that does not reproduce the model, a base that never got made; and a bf16 model that cannot run stops the job
    broken = [r for r in map(json.loads, open(f"{out}/results.jsonl")) if r["variant"] in ("base", "ref", "int4")]
    broken[1] = {**broken[1], "error": "the control (ref) does not reproduce the bf16 model (KL 0.7)"}
    text = gq.markdown(meta, broken)
    assert "reproduce the bf16 model (KL 0.7)" in text and "Not measured: ref" in text and "the check that the path reproduces the model" not in text and "| int4 |" in text, text
    m_bad = tiny_model()
    m_bad.forward = lambda *a_, **k_: (_ for _ in ()).throw(RuntimeError("boom"))
    ab = ns()
    ab.out, ab.variants = tempfile.mkdtemp(), ["ref"]
    try:
        with torch.inference_mode():
            gq.run_all(m_bad, ab, NemoRun)
        raise AssertionError("a model that cannot run went on")
    except RuntimeError:
        pass
    assert "boom" in open(f"{ab.out}/quality.md").read() and not any("kl" in json.loads(l) for l in open(f"{ab.out}/results.jsonl")), "no row for the failed base"
    gs.MODELS["tiny-nemo"] = dict(gs.MODELS["nemotron"], ne=16, k=4, n_moe=4, caches=(13, 25, 50))
    R = gs.analyze(argparse.Namespace(model="tiny-nemo", dir=out, json=f"{out}/skew.json", md=f"{out}/skew.md", quiet=True))
    assert R["layers"] == 4 and R["static"]["50"]["per_layer_alloc"]["heldout"] > 0
    print(f"selftest ok: rounding a tiny random NemotronH ({secs:.0f} s for {len(names)} variants): KL {', '.join(f'{n} {kl[n]:.4f}' for n in ('ref', 't_int5', 'int4', 'int3', 't_int2', 'hlm3', 'mix13', 'cold50', 'hlm3g', 't_hlm2', 'hlm2g', 'hb40', 'hb20'))}; "
          f"weight error int4 {we['int4']['int4']['rel_rmse']:.4f} int3 {we['int3']['int3']['rel_rmse']:.4f} hlm3 {we['hlm3']['hlm3']['rel_rmse']:.4f}; GPTQ {gp['experts_gptq']} experts, {gp['experts_rtn_fallback']} fell back; "
          f"Hummingbird bits/weight on Nemotron's matrices {', '.join(f'{n} {hb[n]:.3f}' for n in gq.HUMMINGBIRD)}; neuron rows {', '.join(f'{n} {hr[n]:.4f}' for n in HBR)} "
          f"(KL hbr40 {kl['hbr40']:.4f} hbr40p {kl['hbr40p']:.4f}, 2-bit nearest {kl['t_hbr2']:.4f} GPTQ {kl['t_hbr2g']:.4f}, 3-bit GPTQ {kl['t_hbr3g']:.4f}); hbr40's export of 4 layers ({exp_bytes:,} B) read back bit for bit and its signs undo to the pass's weights; "
          f"the experts' input is the 64-wide latent and the router's choice the model's, the experts' arithmetic is NemotronHExperts' (chunks 1, 3, 16), with the grouped sum order ref KL {rows3['ref']['kl']:.1e}; "
          f"the layer-by-layer logits are the forward's {'to the bit' if all(bits) else 'within a bf16 step'}; real expert shapes ok; block codes checked on {n_cuda} GPUs, the tiny model reloaded from the checkpoint layout through run_all on {len(devs)} devices")
    for p_ in (d, out, a2.out, a3.out, out_g, ck, cdir, ab.out, dexp):
        shutil.rmtree(p_, ignore_errors=True)


def cmd_run(a):
    """gq.cmd_run for this model; --export DIR hands hbr40's pass an Exporter (nemotron_hbr.py)."""
    a.exporter = None
    if a.export:
        if "hbr40" not in [v.strip() for v in (a.variants or ",".join(ORDER)).split(",")]:
            sys.exit("--export writes the codes of hbr40: name it in --variants")
        m = gs.MODELS["nemotron"]
        a.exporter = hbr.Exporter(a.export, "hbr40", m["repo"], m["rev"])
    gq.cmd_run(a, run_cls=NemoRun)


def main():
    gq.main(plan, cmd_run, selftest, VARIANTS, __doc__, ORDER)


if __name__ == "__main__":
    main()
