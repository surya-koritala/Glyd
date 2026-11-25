//! `ceiling MODEL [--box B] [--gpus N] [--by-layers] [--pos P] [--hit H] [--fp8 PCT] [--bf16-pct PCT] [--eff E] [--t0-us U] [--t0-moe-us U] [--pair-ms M] [--fixed-us U] [--exchange-us U] [--stats FILE [--calibrate MODEL:FILE]]
//! [--measured MS]` (development): a
//! decode step's least time by the formula of docs/ceiling.md, from the checkpoint's own tensors (a directory's safetensors, or a Hub repo's headers
//! read by ranges: no weights are downloaded), the engine's layouts, cuts and slicing (src/dynamic/split.rs, tp.rs), and a box's spec. Beside each
//! term, the same with the rates our code was measured at (the source named), and the measured step where one is given.

use crate::args::Args;
use crate::dynamic::split::{CODED_PCT, PLAIN_PCT};
use crate::sizes::layer_of;

/// A bf16 matrix's bytes as packed, a share of the release's, measured: Kimi-K3's layer 0, its embedding and head (6.56 GiB of bf16) packed to 4.522 GiB
/// tiered and 4.935 GiB in the 12-bit layout (tpbench --one, 2026-10-22). The planner's TIERED_PCT and TWELVE_PCT (72, 78) are bounds for memory, not this.
const TIERED_ACTUAL: f64 = 0.689;
const TWELVE_ACTUAL: f64 = 0.752;
use crate::ffi::{err, Error, Result};
use glyd_gpu::json::{self, Value};
use std::path::Path;

/// A box: its GPUs' memory read rate (GB/s: the copy rate measured where there is one, else the spec sheet's), memory (GiB), the rate a GPU's copies
/// from host memory take when every GPU copies at once (GB/s), the host's memory read rate (GB/s), whether HBM, and our code's measured rates there.
struct Box {
    name: &'static str,
    gpus: usize,
    read: f64,
    read_src: &'static str,
    mem: f64,
    pcie: f64,
    pcie_alone: f64,
    pcie_src: &'static str,
    host: f64,
    hbm: bool,
    /// Our products' share of `read`: bf16 (12-bit on HBM, tiered elsewhere), FP8, FP4 experts; and the source.
    ours: (f64, f64, f64, &'static str),
    /// A small gather's time measured at 2, 4 and 8 GPUs (us; 0: not measured) and a host pair of a Kimi-K3 expert (ms; 0: not measured).
    gather_us: [f64; 3],
    /// A gather's time by its size, in the step graph (tpgather): for each count of GPUs, (KB a GPU, us) points, the low-latency kind's then the
    /// regular kind's (empty: not measured; then `gather_us`).
    gathers: &'static [(usize, &'static [(f64, f64)], &'static [(f64, f64)])],
    pair_ms: f64,
    /// A product's own time (us) beside its bytes, a dense one's and a MoE layer's experts' (0: in the shares above).
    t0_us: (f64, f64),
}

/// 8x A100 80 GB SXM4's gathers in the step graph (tpgather, research/kimi-k3-a100-2026-10-22b): 1 x 896, 1 x 4608, 16 x 768, 64 x 4608 and 1024 x
/// 4608 values a GPU.
const A100_GATHERS: &[(usize, &[(f64, f64)], &[(f64, f64)])] = &[
    (2, &[(1.8, 3.6), (9.2, 5.5), (24.6, 6.4), (589.8, 48.2), (9437.2, 227.4)], &[(1.8, 7.9), (9.2, 9.8), (24.6, 12.9), (589.8, 45.2), (9437.2, 228.6)]),
    (4, &[(1.8, 5.6), (9.2, 10.5), (24.6, 13.3), (589.8, 98.2), (9437.2, 482.4)], &[(1.8, 10.3), (9.2, 15.2), (24.6, 21.7), (589.8, 98.2), (9437.2, 483.0)]),
    (
        8,
        &[(1.8, 8.8), (9.2, 20.9), (24.6, 22.7), (589.8, 202.9), (9437.2, 1224.9)],
        &[(1.8, 15.4), (9.2, 24.1), (24.6, 32.8), (589.8, 200.3), (9437.2, 1210.9)],
    ),
];

/// A gather's time (us) of `kb` KB a GPU from measured points: straight lines between them, the first's time below it, the last two's line past them.
fn gather_time(pts: &[(f64, f64)], kb: f64) -> f64 {
    let i = pts.iter().position(|p| p.0 >= kb).unwrap_or(pts.len() - 1).max(1);
    if kb <= pts[0].0 {
        return pts[0].1;
    }
    let ((x0, y0), (x1, y1)) = (pts[i - 1], pts[i]);
    y0 + (y1 - y0) * (kb - x0) / (x1 - x0)
}

/// The boxes measured or planned. ponytail: a table in the source; a file of boxes when there are more than a handful.
const BOXES: &[Box] = &[
    Box {
        name: "4080s",
        gpus: 1,
        read: 702.0,
        read_src: "measured (bandwidth, 2026-10-17)",
        mem: 16.0,
        pcie: 25.0,
        pcie_alone: 25.0,
        pcie_src: "PCIe 4.0 x16, spec",
        host: 96.0,
        hbm: false,
        ours: (
            0.922,
            0.951,
            0.951,
            "measured at real widths: Kimi-K3 layer 0's bf16 products 92.2%, DeepSeek-V4's 3 layers 95.1% (tpbench --one --prof, 2026-10-22)",
        ),
        gather_us: [0.0; 3],
        gathers: &[],
        pair_ms: 0.45,
        t0_us: (0.0, 0.0),
    },
    Box {
        name: "a10",
        gpus: 1,
        read: 600.0,
        read_src: "spec sheet (A10 24 GB)",
        mem: 24.0,
        pcie: 25.0,
        pcie_alone: 25.0,
        pcie_src: "PCIe 4.0 x16, spec",
        host: 204.8,
        hbm: false,
        ours: (0.922, 0.951, 0.951, "the 4080 SUPER's (GDDR); not measured on an A10"),
        gather_us: [0.0; 3],
        gathers: &[],
        pair_ms: 0.0,
        t0_us: (0.0, 0.0),
    },
    Box {
        name: "a100",
        gpus: 1,
        read: 1555.0,
        read_src: "spec sheet (A100 SXM4 40 GB)",
        mem: 40.0,
        pcie: 23.3,
        pcie_alone: 23.3,
        pcie_src: "measured: one A100 SXM4 copying alone (a100x8's)",
        host: 204.8,
        hbm: true,
        ours: (0.80, 0.57, 0.57, "a100x8's (the same GPU)"),
        gather_us: [0.0; 3],
        gathers: &[],
        pair_ms: 0.0,
        t0_us: (0.0, 0.0),
    },
    Box {
        name: "a100x8",
        gpus: 8,
        read: 2039.0,
        read_src: "spec sheet (A100 SXM4 80 GB)",
        mem: 80.0,
        pcie: 12.3,
        pcie_alone: 23.3,
        pcie_src: "measured: 8 GPUs copying at once (one alone 23.3), 2 GPUs a PCIe switch",
        host: 409.6,
        hbm: true,
        ours: (
            0.80,
            0.57,
            0.57,
            "bf16: H100's 12-bit 80% (Nsight); FP8 and FP4 experts 57%, fitted to DeepSeek-V4-Flash's split-on-2 and across-2 runs (docs/ceiling.md)",
        ),
        gather_us: [8.9, 12.9, 19.5],
        gathers: A100_GATHERS,
        pair_ms: 0.76,
        t0_us: (0.0, 0.0),
    },
    Box {
        name: "h100x8",
        gpus: 8,
        read: 3350.0,
        read_src: "spec sheet (H100 SXM5)",
        mem: 80.0,
        pcie: 25.0,
        pcie_alone: 50.0,
        pcie_src: "assumed: half of PCIe 5.0 x16, not measured",
        host: 614.4,
        hbm: true,
        ours: (0.80, 0.80, 0.30, "Nsight on one H100, 2026-10-22; FP8 assumed 80%"),
        gather_us: [0.0; 3],
        gathers: &[],
        pair_ms: 0.0,
        t0_us: (0.0, 0.0),
    },
];

/// A JSON number as f64.
fn num(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.parse().ok(),
        _ => None,
    }
}

/// The config's value of the first of `keys` there (its text_config's first).
fn cfg_num(c: &Value, keys: &[&str]) -> Option<f64> {
    let t = c.get("text_config").unwrap_or(c);
    keys.iter().find_map(|k| t.get(k).or_else(|| c.get(k)).and_then(num))
}

/// The tensors of MODEL (name, dtype, shape, bytes) and its config: a directory's safetensors, else a Hub repo's headers.
fn tensors(model: &str) -> Result<(Value, Vec<(String, String, Vec<u64>, u64)>)> {
    let dir = Path::new(model);
    if dir.join("config.json").exists() {
        let cfg = json::parse(&std::fs::read_to_string(dir.join("config.json")).map_err(|e| Error(e.to_string()))?)
            .map_err(|e| Error(format!("config.json: {e}")))?;
        let ck = glyd_gpu::safetensors::Checkpoint::open(dir).map_err(|e| Error(format!("{model}: {e}")))?;
        let ts = ck.files.iter().flat_map(|f| f.tensors.iter().map(|(n, t)| (n.clone(), t.dtype.clone(), t.shape.clone(), t.bytes))).collect();
        return Ok((cfg, ts));
    }
    let hub = crate::hub::Hub::from_env();
    let remote = hub.list(model, "main").map_err(|e| Error(format!("{model}: {e}")))?;
    let cfg = json::parse(&hub.text(model, &remote.commit, "config.json").map_err(|e| Error(format!("{model}: {e}")))?)
        .map_err(|e| Error(format!("config.json: {e}")))?;
    let files: Vec<_> = remote.files.iter().filter(|e| e.path.ends_with(".safetensors") && !e.path.contains('/')).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out = std::sync::Mutex::new(Vec::new());
    let failed = std::sync::Mutex::new(None);
    std::thread::scope(|s| {
        for _ in 0..8 {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= files.len() {
                    break;
                }
                match hub.header(&remote, files[i]) {
                    Ok((h, _)) => out.lock().unwrap().extend(h.into_iter().map(|(n, d, sh, a, b)| (n, d, sh, b - a))),
                    Err(e) => {
                        failed.lock().unwrap().get_or_insert(format!("{}: {e}", files[i].path));
                        break;
                    }
                }
            });
        }
    });
    if let Some(e) = failed.into_inner().unwrap() {
        return err(e);
    }
    Ok((cfg, out.into_inner().unwrap()))
}

/// What a decode step reads, per GPU, and what it counts (src: the checkpoint's tensors; the engine's rules).
#[derive(Default, Debug)]
struct Step {
    dense: f64,
    /// of `dense`, the FP8 matrices'
    dense_f8: f64,
    sliced: f64,
    whole: f64,
    expert: f64,
    moe_layers: usize,
    exchanges: usize,
    /// the products a token runs on a GPU, sliced or whole (the experts' two a MoE layer apart)
    products: usize,
    /// each exchange's KB a GPU at one token (its slice of the product's rows, bf16)
    gather_kb: Vec<f64>,
    attn: f64,
    attn_note: String,
    /// every byte the GPUs hold (the embedding and every expert too)
    held: f64,
    /// one routed expert's bytes, whole
    per_expert: f64,
}

/// A matrix's product as the engine makes it, by its name without the weight's suffix: q, k and v one, gate and up one; Kimi-K3's (`mla`: its layer's
/// attention, None: another model's) as src/dynamic/kimi.rs packs them: the delta attention's q, k, v, output gate, f_a and beta one, the latent
/// attention's q_a, gate and kv_a one, a MoE layer's latent down projection with the shared experts' gate and up.
fn product_of(base: &str, mla: Option<bool>) -> String {
    let generic = || {
        base.replace("q_proj", "qkv")
            .replace("k_proj", "qkv")
            .replace("v_proj", "qkv")
            .replace("gate_proj", "gate_up")
            .replace("up_proj", "gate_up")
            .replace(".w1", ".w13")
            .replace(".w3", ".w13")
    };
    let (Some(mla), Some((head, last))) = (mla, base.rsplit_once('.')) else { return generic() };
    match last {
        "q_proj" | "k_proj" | "v_proj" | "g_proj" | "f_a_proj" | "b_proj" if !mla && head.ends_with("self_attn") => format!("{head}.qkv_gates"),
        "q_a_proj" | "g_proj" | "kv_a_proj_with_mqa" if mla => format!("{head}.qa_gate_kva"),
        "routed_expert_down_proj" => format!("{head}.latent_shared"),
        "gate_proj" | "up_proj" if head.ends_with(".shared_experts") => format!("{}.latent_shared", head.trim_end_matches(".shared_experts")),
        _ => generic(),
    }
}

/// The tensors' bytes as the engine holds them and reads them a token, over `n` GPUs (every layer across them; 1: one GPU, or a split's, whose
/// stages read in turn).
fn step(cfg: &Value, ts: &[(String, String, Vec<u64>, u64)], n: usize, bf16_share: f64, f8_pct: u64, pos: usize) -> Result<Step> {
    let nl = cfg_num(cfg, &["num_hidden_layers", "n_layer"]).ok_or_else(|| Error("config.json: no num_hidden_layers".into()))? as usize;
    let experts = cfg_num(cfg, &["num_experts", "n_routed_experts", "num_local_experts"]).unwrap_or(0.0) as usize;
    let topk = cfg_num(cfg, &["num_experts_per_token", "num_experts_per_tok", "moe_topk"]).unwrap_or(0.0);
    let slicer = crate::dynamic::tp::Slicer::of(n);
    let mut s = Step::default();
    let mut expert_layers = vec![0u64; nl];
    // (Kimi-K3: its layers' attention (full_attn_layers counts from 1), and its packs of several tensors sliced as one, rows padded to 512)
    let tc = cfg.get("text_config").unwrap_or(cfg);
    let kimi = tc.get("model_type").and_then(Value::as_str) == Some("kimi_linear");
    let full: Vec<u64> = tc.get("linear_attn_config").and_then(|la| la.get("full_attn_layers")).and_then(Value::as_u64s).unwrap_or_default();
    let kind = |l: Option<usize>| kimi.then(|| l.is_some_and(|l| full.contains(&(l as u64 + 1))));
    let base_of = |name: &str| name.trim_end_matches(".weight").trim_end_matches(".weight_packed").to_string();
    let mut packs: std::collections::BTreeMap<(usize, String), (u64, u64, usize)> = std::collections::BTreeMap::new();
    if kimi {
        for (name, _, shape, _) in ts {
            if shape.len() == 2 && !name.contains("scale") && !name.contains(".experts.") {
                let l = layer_of(name);
                let p = packs.entry((l.unwrap_or(usize::MAX), product_of(&base_of(name), kind(l)))).or_default();
                (p.0, p.1, p.2) = (p.0 + shape[0], shape[1], p.2 + 1);
            }
        }
    }
    let mut rows_of: std::collections::BTreeMap<(usize, String), u64> = std::collections::BTreeMap::new();
    // (the products of a layer, as the engine makes them (`product_of`); their names without the weight's suffix)
    let mut products: std::collections::BTreeSet<(usize, String)> = std::collections::BTreeSet::new();
    let mut every: std::collections::BTreeSet<(usize, String)> = std::collections::BTreeSet::new();
    for (name, dtype, shape, bytes) in ts {
        if ["mtp", "vision", "visual", "audio", "multi_modal"].iter().any(|w| name.contains(w)) {
            continue;
        }
        let l = layer_of(name);
        if l.is_some_and(|l| l >= nl) {
            continue;
        }
        let matrix = shape.len() == 2 && shape.iter().all(|&d| d >= 256);
        let held = match dtype.as_str() {
            "BF16" if matrix => *bytes as f64 * bf16_share,
            "F8_E4M3" if matrix => *bytes as f64 * f8_pct as f64 / 100.0,
            _ => *bytes as f64,
        };
        s.held += held;
        if name.contains(".experts.") && !name.contains("shared") {
            if let Some(l) = l {
                expert_layers[l] += held as u64;
            }
            continue;
        }
        if name.contains("embed_tokens") || name == "embed.weight" || name.ends_with(".embed.weight") {
            // (a token reads one row)
            s.dense += held / shape.first().copied().unwrap_or(1).max(1) as f64;
            continue;
        }
        // (the loaders slice any two-dimensional matrix the rule takes, whatever its size, Kimi-K3's packs as one; the layouts above count the big ones alone)
        let key = (l.unwrap_or(usize::MAX), product_of(&base_of(name), kind(l)));
        if shape.len() == 2 && !name.contains("scale") {
            *rows_of.entry(key.clone()).or_default() += shape[0];
        }
        let cut = shape.len() == 2
            && match packs.get(&key) {
                Some(&(rows, cols, parts)) => slicer.slices(if parts > 1 { rows.div_ceil(512) * 512 } else { rows } as usize, cols as usize),
                None => slicer.slices(shape[0] as usize, shape[1] as usize),
            };
        if shape.len() == 2 && !name.contains("scale") {
            every.insert(key.clone());
        }
        if dtype == "F8_E4M3" && matrix {
            s.dense_f8 += if cut { held / n as f64 } else { held };
        }
        if cut {
            s.sliced += held / n as f64;
            if !name.contains("scale") {
                products.insert(key);
            }
        } else {
            s.whole += held;
        }
    }
    s.dense += s.sliced + s.whole;
    s.moe_layers = expert_layers.iter().filter(|&&b| b > 0).count();
    if experts > 0 && s.moe_layers > 0 {
        // (each MoE layer reads top_k experts; an expert's bytes are its layer's over the experts; its rows sliced over the GPUs)
        let per_expert = expert_layers.iter().sum::<u64>() as f64 / (s.moe_layers * experts) as f64;
        s.per_expert = per_expert;
        s.expert = topk * s.moe_layers as f64 * per_expert / n as f64;
    }
    // (every sliced product gathers once; the experts' two products once each a MoE layer)
    s.exchanges = if n > 1 { products.len() + 2 * s.moe_layers } else { 0 };
    if n > 1 {
        for k in &products {
            let rows = rows_of.get(k).copied().unwrap_or(0);
            let rows = if packs.get(k).is_some_and(|p| p.2 > 1) { rows.div_ceil(512) * 512 } else { rows };
            s.gather_kb.push(rows as f64 / n as f64 * 2.0 / 1e3);
        }
        // (the experts' first product: top_k rows of gate and up, sliced; the second's weighted sum: the latent's or the model's width, sliced)
        let inter = cfg_num(cfg, &["moe_intermediate_size"]).unwrap_or(0.0);
        let out = cfg_num(cfg, &["routed_expert_hidden_size"]).or_else(|| cfg_num(cfg, &["hidden_size"])).unwrap_or(0.0);
        for _ in 0..s.moe_layers {
            s.gather_kb.push(topk * 2.0 * inter / n as f64 * 2.0 / 1e3);
            s.gather_kb.push(out / n as f64 * 2.0 / 1e3);
        }
    }
    s.products = every.len();
    // attention's reads a token (every GPU its own: attention is not sliced): the cache at `pos` and the linear attention's states, read and written
    let t = cfg.get("text_config").unwrap_or(cfg);
    let mt = t.get("model_type").and_then(Value::as_str).unwrap_or("");
    if let Some(la) = t.get("linear_attn_config") {
        let full = la.get("full_attn_layers").and_then(Value::as_u64s).map_or(0, |v| v.len());
        let kda = nl.saturating_sub(full);
        let heads = cfg_num(cfg, &["num_attention_heads"]).unwrap_or(0.0);
        let hd = la.get("head_dim").and_then(num).unwrap_or(128.0);
        let latent = cfg_num(cfg, &["kv_lora_rank"]).unwrap_or(0.0) + cfg_num(cfg, &["qk_rope_head_dim"]).unwrap_or(0.0);
        s.attn = kda as f64 * heads * hd * hd * 4.0 * 2.0 + full as f64 * pos as f64 * latent * 2.0;
        s.attn_note =
            format!("{kda} linear layers' states ({heads} x {hd} x {hd} fp32, read and written) + {full} MLA layers' cache at {pos} ({latent} values a token)");
    } else if mt == "deepseek_v4" {
        let window = cfg_num(cfg, &["sliding_window"]).unwrap_or(128.0);
        let hd = cfg_num(cfg, &["head_dim"]).unwrap_or(512.0);
        let seen = (pos as f64).min(window) + pos as f64 / 4.0;
        s.attn = nl as f64 * seen * hd * 2.0;
        s.attn_note = format!("{nl} layers' cache: the window's {window} and a quarter of {pos} positions compressed, {hd} values each (approximate)");
    } else {
        s.attn_note = "not modeled for this family".into();
    }
    Ok(s)
}

fn ms(bytes: f64, gbs: f64) -> f64 {
    if gbs > 0.0 {
        bytes / gbs / 1e6
    } else {
        f64::NAN
    }
}

/// One count of rows of a run through the expert cache (`storebench --out`'s line): the rows, the measured step, an expert's bytes as the cache holds it, and each MoE layer's statistics over `per` steps (a line
/// marked `totals` holds the timed steps' sums, `steps` of them; an older line their averages, rounded: `per` 1).
struct Run {
    rows: usize,
    step_ms: f64,
    expert_bytes: f64,
    per: f64,
    layers: Vec<crate::offload::LayerStats>,
}

impl Run {
    fn parse(line: &str) -> Result<Run> {
        let v = json::parse(line).map_err(|e| Error(format!("a line of --stats: {e}")))?;
        let n = |o: &Value, k: &str| o.get(k).and_then(num).ok_or_else(|| Error(format!("a line of --stats without {k}")));
        let layers = match v.get("layers") {
            Some(Value::Array(ls)) => ls
                .iter()
                .map(|l| {
                    Ok(crate::offload::LayerStats {
                        calls: n(l, "calls")? as u64,
                        whole: l.get("whole").and_then(num).unwrap_or(0.0) as u64,
                        whole_bytes: l.get("whole_bytes").and_then(num).unwrap_or(0.0) as u64,
                        rows: n(l, "rows")? as u64,
                        pairs: n(l, "pairs")? as u64,
                        reads: n(l, "reads")? as u64,
                        hits: n(l, "hits")? as u64,
                        copies: n(l, "copies")? as u64,
                        host: n(l, "host")? as u64,
                        host_pairs: n(l, "host_pairs")? as u64,
                        skipped: n(l, "skipped")? as u64,
                        waves: n(l, "waves")? as u64,
                        copy_bytes: n(l, "copy_bytes")? as u64,
                        host_bytes: n(l, "host_bytes")? as u64,
                        missed_bytes: l.get("missed_bytes").and_then(num).unwrap_or(0.0) as u64,
                        ahead: n(l, "ahead")? as u64,
                        ahead_bytes: n(l, "ahead_bytes")? as u64,
                        copy_ms: n(l, "copy_ms")?,
                        host_ms: n(l, "host_ms")?,
                        wait_ms: n(l, "wait_ms")?,
                        plan_ms: l.get("plan_ms").and_then(num).unwrap_or(0.0),
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            _ => return err("a line of --stats without its layers"),
        };
        let per = if v.get("totals") == Some(&Value::Bool(true)) { n(&v, "steps")?.max(1.0) } else { 1.0 };
        Ok(Run { rows: n(&v, "rows")? as usize, step_ms: n(&v, "step_ms")?, expert_bytes: n(&v, "expert_bytes")?, per, layers })
    }
}

/// A decode step of `r.rows` rows on one GPU whose experts are in the expert cache, from its layers' statistics (`--stats`, src/tiering.rs storebench): (ceiling, ours) in ms. The dense weights are read once
/// for the rows; each expert a layer runs on the GPU (the ones asked for less those the host computed) once; the attention once a row.
///
/// The ceiling: a layer's misses take the longer of their copies at the box's link rate and the host's reads at its memory rate (each missed expert read once), beside each other.
///
/// Ours follows the layers in order on the host, the copy stream and the GPU, at the rates the run measured (the layer's copy stream time a byte, the host's time): a call plans (`plan_ms`), then its copies start once
/// the copy stream has run what was queued before them (the last layer's promotions, which wait for that layer's products); its products run once its copies have, and its promotions (`ahead`) after them; the host's
/// pairs run from the plan on; the next layer starts once the products and the host's pairs are done and the rest of the layer has run (the dense weights and the attention, a layer's share, and `fixed`). A call
/// copied whole (`whole`: no host half, its misses in waves where they outnumber the places) runs its copies and products with the host waiting; a layer's calls of each kind are counted by their share. The step is
/// the second of two in a row (the copies a step leaves queued delay the next one's first).
fn cached_step(s: &Step, b: &Box, shares: (f64, f64, f64), r: &Run, fixed: Fixed) -> (f64, f64) {
    let (e_bf, e_f8, e_fp4) = shares;
    let x = |v: u64| v as f64 / r.per;
    let layers: Vec<&crate::offload::LayerStats> = r.layers.iter().filter(|l| l.calls > 0).collect();
    let on_gpu: f64 = layers.iter().map(|l| (x(l.reads) - x(l.host)) * r.expert_bytes).sum();
    let attn = r.rows as f64 * s.attn;
    let ceiling = ms(s.dense + on_gpu + attn, b.read) + layers.iter().map(|l| ms(x(l.copy_bytes), b.pcie_alone).max(ms(x(l.host_bytes), b.host))).sum::<f64>();
    let n = layers.len().max(1) as f64;
    // (the fixed time a layer goes with its products, which the copies behind them wait for; the time a row with the rest of the layer, the attention sequence by sequence)
    let rest = (ms(s.dense - s.dense_f8, b.read * e_bf) + ms(s.dense_f8, b.read * e_f8) + ms(attn, b.read)) / n + fixed.row * r.rows as f64;
    // (t: the host's clock at a call's start; pend: the copy stream's end of what is queued on it)
    let (mut t, mut pend, mut first) = (0.0f64, 0.0f64, 0.0f64);
    for step in 0..2 {
        if step == 1 {
            first = t;
        }
        for l in &layers {
            let all = x(l.copy_bytes) + x(l.ahead_bytes);
            // (the copy stream's measured time a byte; copies the GPU's own kernels made, the plan on the GPU's, have none: the box's link rate)
            let per_byte = if all > 0.0 && l.copy_ms > 0.0 { x_ms(l.copy_ms, r.per) / all } else { 1.0 / (b.pcie_alone * 1e6) };
            // (a step's calls of the layer, those copied whole a share `w` of them; each kind's own figures a call)
            let m = x(l.calls).round().max(1.0);
            let w = (l.whole as f64 / l.calls as f64).clamp(0.0, 1.0);
            let (nw, nh) = (w * m, (1.0 - w) * m);
            let of = |v: f64, k: f64| if k > 0.0 { v / k } else { 0.0 };
            let (reads, plan) = (x(l.reads) / m, x_ms(l.plan_ms, r.per) / m);
            let cw = of(x(l.whole_bytes), nw) * per_byte;
            let ch = of((x(l.copy_bytes) - x(l.whole_bytes)).max(0.0), nh) * per_byte;
            let (host, prom) = (of(x_ms(l.host_ms, r.per), nh), of(x(l.ahead_bytes), nh) * per_byte);
            let gw = ms(reads * r.expert_bytes, b.read * e_fp4) + fixed.layer / m;
            let gh = ms((reads - of(x(l.host), nh)).max(0.0) * r.expert_bytes, b.read * e_fp4) + fixed.layer / m;
            for _ in 0..m as usize {
                // a call copied whole: its copies behind the queue, then its products, the host waiting
                let ew = (t + plan).max(pend) + cw;
                // a call with a host half: its copies, its products after them and its promotions after those, beside the host's pairs
                let q = t + plan;
                let ph = q.max(pend) + ch + gh;
                t = w * (ew + gw) + (1.0 - w) * ph.max(q + host);
                pend = w * ew + (1.0 - w) * (ph + prom);
            }
            t += rest;
        }
    }
    (ceiling, t - first)
}

/// A run's milliseconds over its `per` steps, a step's.
fn x_ms(v: f64, per: f64) -> f64 {
    v / per
}

/// What a cached step takes besides its terms, for each MoE layer: `layer` ms, which goes with the layer's products (the copies queued behind them wait for it), and `row` ms a row, which goes with the rest of the
/// layer (the work a row adds a layer that the terms do not count: on the RTX 4080 SUPER (the box `4080s`), the launches of the operations a pass runs sequence by sequence), measured as runs' unexplained (`fit`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Fixed {
    layer: f64,
    row: f64,
}

impl Fixed {
    /// The least-squares line through the runs' unexplained a layer by their rows (one run: its unexplained, no time a row).
    fn fit(runs: &[(usize, f64)]) -> Fixed {
        let n = runs.len() as f64;
        let (mx, my) = (runs.iter().map(|r| r.0 as f64).sum::<f64>() / n, runs.iter().map(|r| r.1).sum::<f64>() / n);
        let sxx: f64 = runs.iter().map(|r| (r.0 as f64 - mx).powi(2)).sum();
        let row = if sxx > 0.0 { runs.iter().map(|r| (r.0 as f64 - mx) * (r.1 - my)).sum::<f64>() / sxx } else { 0.0 };
        Fixed { layer: my - row * mx, row }
    }
}

/// A line's unexplained a MoE layer: the fixed time a layer that makes ours its measured step (ours grows with it, so it is found by halving).
fn unexplained(s: &Step, b: &Box, shares: (f64, f64, f64), r: &Run) -> f64 {
    let at = |f: f64| cached_step(s, b, shares, r, Fixed { layer: f, row: 0.0 }).1;
    let span = r.step_ms.max(1e-3);
    let (mut lo, mut hi) = (-span, span);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if at(mid) < r.step_ms {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

pub fn cmd(a: &Args) -> Result<()> {
    let model = a.need(1, "MODEL")?;
    let name = a.flag("box").unwrap_or("a100x8");
    let b = BOXES
        .iter()
        .find(|b| b.name == name)
        .ok_or_else(|| Error(format!("--box {name}: one of {}", BOXES.iter().map(|b| b.name).collect::<Vec<_>>().join(", "))))?;
    let split = a.has("by-layers");
    let n: usize = a.num("gpus", b.gpus)?;
    let pos: usize = a.num("pos", 512)?;
    let (cfg, ts) = tensors(model)?;
    let mt = cfg.get("model_type").and_then(Value::as_str).unwrap_or("");
    // (--stats FILE: decode steps through the expert cache on one GPU, each line a count of rows' layer statistics and measured step (storebench --out); --fixed-us: ours' fixed time a step, else the
    // first line's unexplained, and the others predicted from it)
    if let Some(f) = a.flag("stats") {
        let text = std::fs::read_to_string(f).map_err(|e| Error(format!("{f}: {e}")))?;
        let runs: Vec<Run> = text.lines().filter(|l| !l.trim().is_empty()).map(Run::parse).collect::<Result<_>>()?;
        let f8_pct: u64 = a.num("fp8", if mt == "deepseek_v4" { 100 } else { CODED_PCT })?;
        let bf16_share: f64 = a.num("bf16-pct", 100.0 * if b.hbm { TWELVE_ACTUAL } else { TIERED_ACTUAL })? / 100.0;
        let s = step(&cfg, &ts, 1, bf16_share, f8_pct, pos)?;
        let shares = (b.ours.0, b.ours.1, b.ours.2);
        let first = runs.first().ok_or_else(|| Error(format!("{f}: no line")))?;
        let (fixed, how) = match (a.flag("fixed-us"), a.flag("calibrate")) {
            (Some(u), _) => {
                let per = u.parse::<f64>().map_err(|_| Error(format!("--fixed-us {u}: microseconds")))? / 1e3;
                (Fixed { layer: per, row: 0.0 }, "--fixed-us".to_string())
            }
            // (another run of the same box: its lines' unexplained, fitted; it gives its own tensors' terms)
            (None, Some(c)) => {
                let (cm, cf) = c.split_once(':').ok_or_else(|| Error(format!("--calibrate {c}: MODEL:FILE")))?;
                let (ccfg, cts) = tensors(cm)?;
                let cs = step(
                    &ccfg,
                    &cts,
                    1,
                    bf16_share,
                    a.num("fp8", if ccfg.get("model_type").and_then(Value::as_str) == Some("deepseek_v4") { 100 } else { CODED_PCT })?,
                    pos,
                )?;
                let ctext = std::fs::read_to_string(cf).map_err(|e| Error(format!("{cf}: {e}")))?;
                let cruns: Vec<Run> = ctext.lines().filter(|l| !l.trim().is_empty()).map(Run::parse).collect::<Result<_>>()?;
                let pts: Vec<(usize, f64)> = cruns.iter().map(|r| (r.rows, unexplained(&cs, b, shares, r))).collect();
                (Fixed::fit(&pts), format!("fitted to {cf}'s {} lines", pts.len()))
            }
            (None, None) => (Fixed::fit(&[(first.rows, unexplained(&s, b, shares, first))]), "the first line's unexplained".to_string()),
        };
        println!(
            "{model} on {}, one GPU, the experts in the expert cache (--stats {f}); ours' fixed time {:.3} ms a MoE layer and {:.4} ms a row a layer ({how})",
            b.name, fixed.layer, fixed.row
        );
        for r in &runs {
            let (ceiling, ours) = cached_step(&s, b, shares, r, fixed);
            let t = crate::offload::LayerStats::sum(&r.layers);
            println!(
                "  {:>3} rows: ceiling {ceiling:8.2} ms, ours {ours:8.2} ms, measured {:8.2} ms: ours {:+.1}% of it ({} experts a step on the GPU, {} copied, {} by the host)",
                r.rows,
                r.step_ms,
                100.0 * (ours / r.step_ms - 1.0),
                t.reads - t.host,
                t.copies,
                t.host
            );
        }
        return Ok(());
    }
    // (DeepSeek-V4's FP8 matrices are held exact, as released; elsewhere layout H's plain packs on HBM, coded on the others)
    let f8_pct: u64 = a.num(
        "fp8",
        if mt == "deepseek_v4" {
            100
        } else if b.hbm {
            PLAIN_PCT
        } else {
            CODED_PCT
        },
    )?;
    let across = if split { 1 } else { n };
    let bf16_share: f64 = a.num("bf16-pct", 100.0 * if b.hbm { TWELVE_ACTUAL } else { TIERED_ACTUAL })? / 100.0;
    let s = step(&cfg, &ts, across, bf16_share, f8_pct, pos)?;
    // (the experts all on the GPUs where everything fits them, as cli.rs's rooms count: each GPU's memory but 3 GiB and 5%; else the share the cache
    // finds there, measured: --hit)
    let held_total = s.held;
    let room = n as f64 * (b.mem * 0.95 - 3.0) * (1u64 << 30) as f64;
    let hit: f64 = match a.flag("hit") {
        Some(h) => h.parse().map_err(|_| Error(format!("--hit {h}: a share, 0 to 1")))?,
        None if held_total <= room => 1.0,
        None => return err(format!("the experts do not fit the {n} GPUs ({:.0} GiB held, {:.0} GiB of room): give --hit, the share the cache finds on them (Kimi-K3 on 8x A100: 0.81-0.88 measured)", held_total / (1u64 << 30) as f64, room / (1u64 << 30) as f64)),
    };
    // terms (ms): weights and attention by the GPUs' read rate; a split's stages read in turn, so their bytes add over the GPUs as one GPU's
    let w = s.dense + s.expert * hit;
    let missing = s.expert * across as f64 * (1.0 - hit);
    let fixed_us: f64 = a.num("fixed-us", 0.0)?;
    // (a product's own time beside its bytes: its start and its last wave, measured on the RTX 4080 SUPER (the box `4080s`) at the slices' sizes; the experts' product of a MoE layer more)
    let (t0, t0_moe): (f64, f64) = (a.num("t0-us", b.t0_us.0)?, a.num("t0-moe-us", b.t0_us.1)?);
    let t_products = (s.products as f64 * t0 + 2.0 * s.moe_layers as f64 * t0_moe) / 1e3;
    // (each gather at its size from the box's measured points, the low-latency kind's (--gathers regular: the other's); else one time for every gather:
    // --exchange-us, or tpgather's 1 x 896 a GPU at the GPUs' count)
    let ex_us: f64 = a.num(
        "exchange-us",
        b.gather_us[if n <= 2 {
            0
        } else if n <= 4 {
            1
        } else {
            2
        }],
    )?;
    let regular = a.flag("gathers") == Some("regular");
    let by_size = b.gathers.iter().find(|g| g.0 == n).filter(|_| a.flag("exchange-us").is_none()).map(|g| if regular { g.2 } else { g.1 });
    let ceil_ex_us = 2.0;
    let t_w = ms(w, b.read);
    let t_a = ms(s.attn, b.read);
    // (the missing experts are read from host memory either way, copied by PCIe or computed by the host: its read rate bounds them)
    let t_m = ms(missing, b.host);
    // (ours: the copies alone, every GPU's at once across them, one GPU's at a time split)
    let pcie = if split { b.pcie_alone } else { across as f64 * b.pcie };
    let t_x = s.exchanges as f64 * ceil_ex_us / 1e3;
    let (mut e_bf, mut e_f8, mut e_fp4, mut src) = b.ours;
    // (--eff: one share for every product, as fitted to a box's runs: docs/ceiling.md)
    if let Some(e) = a.flag("eff") {
        let e: f64 = e.parse().map_err(|_| Error(format!("--eff {e}: a share, 0 to 1")))?;
        (e_bf, e_f8, e_fp4, src) = (e, e, e, "--eff, fitted to this box's runs");
    }
    let ours_w = ms(s.dense - s.dense_f8, b.read * e_bf) + ms(s.dense_f8, b.read * e_f8) + ms(s.expert * hit, b.read * e_fp4) + t_products;
    // (ours: the plan shares the missing experts between the copies and the host's pairs, so that both end together: their rates add)
    let pair_ms: f64 = a.num("pair-ms", b.pair_ms)?;
    let host_ours = if pair_ms > 0.0 { s.per_expert / (pair_ms * 1e-3) / 1e9 } else { 0.0 };
    let ours_m = if missing > 0.0 { ms(missing, pcie + host_ours) } else { 0.0 };
    let ours_x = match by_size {
        Some(pts) => s.gather_kb.iter().map(|&kb| gather_time(pts, kb)).sum::<f64>() / 1e3,
        None => s.exchanges as f64 * ex_us / 1e3,
    };
    let gib = |x: f64| x / (1u64 << 30) as f64;
    println!(
        "{model} on {} ({}), {} GPU{} {}, position {pos}, experts found on the GPUs {:.0}%",
        b.name,
        b.read_src,
        n,
        if n == 1 { "" } else { "s" },
        if split {
            "split by layers"
        } else if n > 1 {
            "every layer across them"
        } else {
            ""
        },
        hit * 100.0
    );
    println!("  per GPU a token: weights {:.2} GiB ({:.2} dense: {:.2} sliced, {:.2} whole; {:.2} experts x {:.2} found), attention {:.3} GiB ({}); missing experts {:.2} GiB in all", gib(w), gib(s.dense), gib(s.sliced), gib(s.whole), gib(s.expert), hit, gib(s.attn), s.attn_note, gib(missing));
    println!("  {:<34} {:>10} {:>10}  {}", "term", "ceiling", "ours", "how");
    println!("  {:<34} {:>8.2}ms {:>8.2}ms  bytes / {:.0} GB/s (bf16 packed to {:.1}%); ours at {:.0}% bf16, {:.0}% FP8, {:.0}% FP4 experts ({src}), and {} products x {t0} us + {} experts' x {t0_moe} us = {t_products:.2} ms", "weights", t_w, ours_w, b.read, bf16_share * 100.0, e_bf * 100.0, e_f8 * 100.0, e_fp4 * 100.0, s.products, 2 * s.moe_layers);
    println!("  {:<34} {:>8.2}ms {:>8}  bytes / {:.0} GB/s", "attention's cache and states", t_a, "-", b.read);
    if missing > 0.0 {
        println!(
            "  {:<34} {:>8.2}ms {:>8.2}ms  host memory {:.0} GB/s bounds it; ours: copies {:.1} GB/s ({}) + the host's pairs {:.1} GB/s ({pair_ms} ms a pair)",
            "missing experts", t_m, ours_m, b.host, pcie, b.pcie_src, host_ours
        );
    }
    if s.exchanges > 0 {
        let how = match by_size {
            Some(_) => format!(
                "each at its size ({} kind, tpgather in the step graph): {:.1} us on average",
                if regular { "regular" } else { "low-latency" },
                ours_x * 1e3 / s.exchanges as f64
            ),
            None => format!("{ex_us} us each"),
        };
        println!(
            "  {:<34} {:>8.2}ms {:>8.2}ms  {} exchanges x {ceil_ex_us} us (one NVLink write and a kernel); ours {how}",
            "exchanges", t_x, ours_x, s.exchanges
        );
    }
    if fixed_us > 0.0 {
        println!(
            "  {:<34} {:>10} {:>8.2}ms  measured on the PC: the small kernels, and the cache's plan a MoE layer ({fixed_us} us a step)",
            "fixed",
            "-",
            fixed_us / 1e3
        );
    }
    let ceiling = t_w + t_a + t_m + t_x;
    let ours = ours_w + t_a + ours_m + ours_x + fixed_us / 1e3;
    println!("  {:<34} {:>8.2}ms {:>8.2}ms  ({:.1} and {:.1} tokens/s)", "total", ceiling, ours, 1e3 / ceiling, 1e3 / ours);
    if let Some(m) = a.flag("measured") {
        let m: f64 = m.parse().map_err(|_| Error(format!("--measured {m}: ms a token")))?;
        println!("  measured {m:.2} ms: {:.0}% of the ceiling; {:.2} ms the terms above do not explain", 100.0 * ceiling / m, m - ours);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A two-layer model: a dense layer (q, k, v, o, gate, up, down) and a MoE layer (o and four experts), across two GPUs: the products the engine
    /// gathers (q, k and v one, gate and up one, the experts two) and the bytes each GPU reads.
    #[test]
    fn the_terms_of_a_small_model() {
        let cfg = json::parse(r#"{"num_hidden_layers": 2, "num_experts": 4, "num_experts_per_tok": 2}"#).unwrap();
        let big = |name: &str| (name.to_string(), "BF16".to_string(), vec![4096u64, 4096], 4096 * 4096 * 2);
        let mut ts: Vec<_> = ["q_proj", "k_proj", "v_proj", "o_proj"].iter().map(|p| big(&format!("model.layers.0.self_attn.{p}.weight"))).collect();
        ts.extend(["gate_proj", "up_proj", "down_proj"].iter().map(|p| big(&format!("model.layers.0.mlp.{p}.weight"))));
        ts.push(big("model.layers.1.self_attn.o_proj.weight"));
        for e in 0..4 {
            ts.push((format!("model.layers.1.mlp.experts.{e}.w1.weight"), "U8".into(), vec![512, 2048], 512 * 2048));
        }
        ts.push(("model.layers.1.input_layernorm.weight".into(), "BF16".into(), vec![4096], 8192));
        ts.push(("model.layers.9.mlp.down_proj.weight".into(), "BF16".into(), vec![4096, 4096], 4096 * 4096 * 2));
        let s = step(&cfg, &ts, 2, TWELVE_ACTUAL, 100, 512).unwrap();
        assert_eq!(s.exchanges, 4 + 1 + 2);
        assert_eq!(s.moe_layers, 1);
        let m = (4096 * 4096 * 2) as f64 * TWELVE_ACTUAL;
        assert!((s.sliced - 8.0 * m / 2.0).abs() < 1.0 && (s.whole - 8192.0).abs() < 1.0, "{s:?}");
        // (two experts a token of 1 MiB each, over the two GPUs; the layer past the model's left out)
        assert!((s.expert - 2.0 * (512.0 * 2048.0) / 2.0).abs() < 1.0 && (s.per_expert - 512.0 * 2048.0).abs() < 1.0, "{s:?}");
        assert_eq!(step(&cfg, &ts, 1, TWELVE_ACTUAL, 100, 512).unwrap().exchanges, 0);
    }

    /// A line of storebench --out read back, and a cached step's terms: the dense bytes once, the experts the GPU ran once each, the attention a row, and each layer's misses the longer of its copies (at the run's
    /// measured rate) and the host's time; the fixed time added as given.
    #[test]
    fn a_cached_step_from_its_layer_statistics() {
        let line = r#"{"rows":8,"steps":4,"step_ms":30.0,"expert_bytes":1000000,"layers":[{"layer":0,"calls":1,"rows":8,"pairs":48,"reads":30,"hits":10,"copies":15,"host":5,"host_pairs":6,"skipped":0,"waves":0,"copy_bytes":15000000,"host_bytes":5000000,"ahead":0,"ahead_bytes":0,"copy_ms":0.75,"host_ms":2.0,"wait_ms":0.1}]}"#;
        let r = Run::parse(line).unwrap();
        assert_eq!((r.rows, r.layers.len(), r.layers[0].copies, r.layers[0].host_ms), (8, 1, 15, 2.0));
        let b = &BOXES[0];
        let s = Step { dense: 702e6, attn: 0.0, ..Step::default() };
        let (ceiling, ours) = cached_step(&s, b, (1.0, 1.0, 1.0), &r, Fixed { layer: 0.1, row: 0.05 });
        // (ours: the copies (0.75 ms), the products of 25 experts of 1 MB at 702 GB/s and the fixed 0.1 ms a layer run beside the host's 2 ms; then the 702 MB at 702 GB/s and 0.05 a row of 8)
        assert!((ours - (2.0 + 1.0 + 0.4)).abs() < 1e-9, "{ours}");
        // (a line of totals over 4 steps is the same step; promotions behind the call hold the next one's copies; a call copied whole has the host wait for its copies and products)
        let totals = line.replace(r#""steps":4,"#, r#""steps":4,"totals":true,"#).replace(r#""calls":1,"#, r#""calls":4,"#);
        let totals = totals
            .replace("\"reads\":30,", "\"reads\":120,")
            .replace("\"host\":5,", "\"host\":20,")
            .replace("15000000", "60000000")
            .replace("5000000,", "20000000,");
        let totals = totals.replace("\"copy_ms\":0.75", "\"copy_ms\":3.0").replace("\"host_ms\":2.0", "\"host_ms\":8.0");
        let t = Run::parse(&totals).unwrap();
        assert!((cached_step(&s, b, (1.0, 1.0, 1.0), &t, Fixed { layer: 0.1, row: 0.05 }).1 - ours).abs() < 1e-9);
        let whole = r#"{"rows":8,"steps":1,"totals":true,"step_ms":30.0,"expert_bytes":1000000,"layers":[{"layer":0,"calls":1,"whole":1,"whole_bytes":15000000,"rows":8,"pairs":48,"reads":30,"hits":10,"copies":15,"host":0,"host_pairs":0,"skipped":0,"waves":0,"copy_bytes":15000000,"host_bytes":0,"ahead":0,"ahead_bytes":0,"copy_ms":0.75,"host_ms":0.0,"wait_ms":0.1}]}"#;
        let w = cached_step(&s, b, (1.0, 1.0, 1.0), &Run::parse(whole).unwrap(), Fixed::default()).1;
        assert!((w - (0.75 + 30.0 / 702.0 + 1.0)).abs() < 1e-9, "{w}");
        // (the fixed time a layer and a row: a line through runs' unexplained)
        let f = Fixed::fit(&[(1, 0.15), (8, 0.5), (32, 1.7)]);
        assert!((f.layer - 0.1).abs() < 1e-9 && (f.row - 0.05).abs() < 1e-9, "{f:?}");
        assert_eq!(Fixed::fit(&[(4, 0.3)]), Fixed { layer: 0.3, row: 0.0 });
        // (the ceiling: the copies at PCIe 4's 25 GB/s (0.6 ms) against 5 MB at the host memory's 96 GB/s)
        assert!((ceiling - (1.0 + 25.0 / 702.0 + 0.6)).abs() < 1e-9, "{ceiling}");
        assert!(Run::parse(r#"{"rows":1}"#).is_err());
    }

    /// The cached step against the 4080 SUPER's measured runs (storebench, 2026-11-23: random DeepSeek-V4 of 16 layers of 64 experts with 32 places, Kimi-K3 of 3 layers of 256 with 128, 18 GB each), its fixed
    /// time fitted to each model's copy run and its auto run predicted: within 10% at 1 and 8 rows, and at Kimi-K3's 32 (the auto runs' copies queue behind the promotions of the layer before, which wait for its
    /// products; the formula before counted the copy stream's whole time once and was 13.4% under DeepSeek-V4's 8 rows). Not held: DeepSeek-V4 at 32 rows, which the formula predicts 21.1% under (225.9 ms against
    /// 286.5 measured; printed). These lines are averages without `whole`: at 32 rows a DeepSeek-V4 layer copied some of its calls whole in waves and computed others on the host, which their average does not
    /// say (storebench's lines of totals do).
    #[test]
    fn the_cached_step_predicts_the_measured_auto_runs() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/storebench");
        let b = BOXES.iter().find(|b| b.name == "4080s").unwrap();
        let shares = (b.ours.0, b.ours.1, b.ours.2);
        let runs = |f: &str| -> Vec<Run> { std::fs::read_to_string(format!("{dir}/{f}")).unwrap().lines().map(|l| Run::parse(l).unwrap()).collect() };
        let s = Step::default();
        let (mut bad, mut held) = (Vec::new(), 0);
        for m in ["dsv4", "kimi"] {
            let pts: Vec<(usize, f64)> = runs(&format!("{m}-4080s-copy.jsonl")).iter().map(|r| (r.rows, unexplained(&s, b, shares, r))).collect();
            let fixed = Fixed::fit(&pts);
            for r in runs(&format!("{m}-4080s-auto.jsonl")) {
                let ours = cached_step(&s, b, shares, &r, fixed).1;
                let e = ours / r.step_ms - 1.0;
                eprintln!("{m} auto {} rows: ours {ours:.2} ms, measured {:.2} ms ({:+.1}%), fixed {fixed:?}", r.rows, r.step_ms, 100.0 * e);
                // (every line but DeepSeek-V4's 32 rows, whose average hides the calls copied whole: see above)
                if r.rows <= 8 || m == "kimi" {
                    held += 1;
                    bad.extend((e.abs() >= 0.10).then(|| format!("{m} at {} rows: {:+.1}%", r.rows, 100.0 * e)));
                }
            }
        }
        assert!(bad.is_empty(), "{bad:?}");
        assert_eq!(held, 5, "the 1 and 8 row lines of both models and Kimi-K3's 32");
    }

    /// Kimi-K3's packs as one product each, sliced as one whatever each tensor's size: a delta attention layer's q, k, v, gate, f_a and beta, a latent
    /// attention's q_a, gate and kv_a, a MoE layer's latent down projection and the shared experts' gate and up.
    #[test]
    fn kimi_packs_are_one_product() {
        let cfg = json::parse(r#"{"model_type": "kimi_linear", "num_hidden_layers": 2, "num_experts": 4, "num_experts_per_token": 2, "linear_attn_config": {"full_attn_layers": [2]}}"#).unwrap();
        let m = |name: String, r: u64, c: u64| (name, "BF16".to_string(), vec![r, c], r * c * 2);
        let (a, b, f) = ("model.layers.0.self_attn", "model.layers.1.self_attn", "model.layers.1.block_sparse_moe");
        let mut ts: Vec<_> = ["q_proj", "k_proj", "v_proj", "g_proj", "o_proj"].iter().map(|p| m(format!("{a}.{p}.weight"), 4096, 4096)).collect();
        ts.extend([m(format!("{a}.f_a_proj.weight"), 128, 4096), m(format!("{a}.b_proj.weight"), 32, 4096), m(format!("{a}.f_b_proj.weight"), 4096, 128)]);
        ts.extend([
            m(format!("{b}.q_a_proj.weight"), 1536, 4096),
            m(format!("{b}.g_proj.weight"), 4096, 4096),
            m(format!("{b}.kv_a_proj_with_mqa.weight"), 576, 4096),
            m(format!("{b}.o_proj.weight"), 4096, 4096),
        ]);
        ts.extend([
            m(format!("{f}.routed_expert_down_proj.weight"), 2048, 4096),
            m(format!("{f}.shared_experts.gate_proj.weight"), 2048, 4096),
            m(format!("{f}.shared_experts.up_proj.weight"), 2048, 4096),
        ]);
        ts.extend([m(format!("{f}.shared_experts.down_proj.weight"), 4096, 2048), m(format!("{f}.routed_expert_up_proj.weight"), 4096, 2048)]);
        for e in 0..4 {
            ts.push((format!("{f}.experts.{e}.w1.weight"), "U8".into(), vec![512, 2048], 512 * 2048));
        }
        let s = step(&cfg, &ts, 2, TWELVE_ACTUAL, 100, 512).unwrap();
        // (layer 0: the six as one, f_b, o; layer 1: q_a, gate and kv_a as one, o, the latent's and the shared gate and up as one, the shared down, the latent up)
        assert_eq!(s.products, 3 + 5, "{s:?}");
        // (f_b too small to slice; the others gathered, and the experts' two)
        assert_eq!(s.exchanges, 2 + 5 + 2, "{s:?}");
        // (a gather a GPU's slice of a token's row: the delta attention's pack is 4 x 4096 + 128 + 32 rows, 16,896 padded, half of them a GPU)
        assert_eq!(s.gather_kb.len(), s.exchanges);
        assert!((s.gather_kb.iter().copied().fold(0.0, f64::max) - 16.896).abs() < 1e-9, "{:?}", s.gather_kb);
        assert!((gather_time(&[(1.0, 10.0), (3.0, 20.0)], 2.0) - 15.0).abs() < 1e-9 && gather_time(&[(1.0, 10.0), (3.0, 20.0)], 0.5) == 10.0);
    }
}
