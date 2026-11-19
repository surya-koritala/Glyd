//! `ceiling MODEL [--box B] [--gpus N] [--by-layers] [--pos P] [--hit H] [--fp8 PCT] [--bf16-pct PCT] [--eff E] [--t0-us U] [--t0-moe-us U] [--pair-ms M] [--fixed-us U] [--exchange-us U]
//! [--measured MS]` (development): a
//! decode step's least time by the formula of docs/ceiling.md, from the checkpoint's own tensors (a directory's safetensors, or a Hub repo's headers
//! read by ranges: no weights are downloaded), the engine's layouts, cuts and slicing (src/dynamic/split.rs, tp.rs), and a box's spec. Beside each
//! term, the same with the rates our code was measured at (the source named), and the measured step where one is given.

use crate::args::Args;
use crate::dynamic::split::{layer_of, CODED_PCT, PLAIN_PCT};

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
    pair_ms: f64,
    /// A product's own time (us) beside its bytes, a dense one's and a MoE layer's experts' (0: in the shares above).
    t0_us: (f64, f64),
}

/// The boxes measured or planned. ponytail: a table in the source; a file of boxes when there are more than a handful.
const BOXES: &[Box] = &[
    Box { name: "4080s", gpus: 1, read: 702.0, read_src: "measured (bandwidth, 2026-10-17)", mem: 16.0, pcie: 25.0, pcie_alone: 25.0, pcie_src: "PCIe 4.0 x16, spec", host: 96.0, hbm: false, ours: (0.922, 0.951, 0.951, "measured at real widths: Kimi-K3 layer 0's bf16 products 92.2%, DeepSeek-V4's 3 layers 95.1% (tpbench --one --prof, 2026-10-22)"), gather_us: [0.0; 3], pair_ms: 0.45, t0_us: (0.0, 0.0) },
    Box { name: "a100x8", gpus: 8, read: 2039.0, read_src: "spec sheet (A100 SXM4 80 GB)", mem: 80.0, pcie: 12.3, pcie_alone: 23.3, pcie_src: "measured: 8 GPUs copying at once (one alone 23.3), 2 GPUs a PCIe switch", host: 409.6, hbm: true, ours: (0.80, 0.57, 0.57, "bf16: H100's 12-bit 80% (Nsight); FP8 and FP4 experts 57%, fitted to DeepSeek-V4-Flash's split-on-2 and across-2 runs (docs/ceiling.md)"), gather_us: [8.9, 12.9, 19.5], pair_ms: 0.76, t0_us: (0.0, 0.0) },
    Box { name: "h100x8", gpus: 8, read: 3350.0, read_src: "spec sheet (H100 SXM5)", mem: 80.0, pcie: 25.0, pcie_alone: 50.0, pcie_src: "assumed: half of PCIe 5.0 x16, not measured", host: 614.4, hbm: true, ours: (0.80, 0.80, 0.30, "Nsight on one H100, 2026-10-22; FP8 assumed 80%"), gather_us: [0.0; 3], pair_ms: 0.0, t0_us: (0.0, 0.0) },
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
        let cfg = json::parse(&std::fs::read_to_string(dir.join("config.json")).map_err(|e| Error(e.to_string()))?).map_err(|e| Error(format!("config.json: {e}")))?;
        let ck = glyd_gpu::safetensors::Checkpoint::open(dir).map_err(|e| Error(format!("{model}: {e}")))?;
        let ts = ck.files.iter().flat_map(|f| f.tensors.iter().map(|(n, t)| (n.clone(), t.dtype.clone(), t.shape.clone(), t.bytes))).collect();
        return Ok((cfg, ts));
    }
    let hub = crate::hub::Hub::from_env();
    let remote = hub.list(model, "main").map_err(|e| Error(format!("{model}: {e}")))?;
    let cfg = json::parse(&hub.text(model, &remote.commit, "config.json").map_err(|e| Error(format!("{model}: {e}")))?).map_err(|e| Error(format!("config.json: {e}")))?;
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
    let generic = || base.replace("q_proj", "qkv").replace("k_proj", "qkv").replace("v_proj", "qkv").replace("gate_proj", "gate_up").replace("up_proj", "gate_up").replace(".w1", ".w13").replace(".w3", ".w13");
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
        s.attn_note = format!("{kda} linear layers' states ({heads} x {hd} x {hd} fp32, read and written) + {full} MLA layers' cache at {pos} ({latent} values a token)");
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
    if gbs > 0.0 { bytes / gbs / 1e6 } else { f64::NAN }
}

pub fn cmd(a: &Args) -> Result<()> {
    let model = a.need(1, "MODEL")?;
    let name = a.flag("box").unwrap_or("a100x8");
    let b = BOXES.iter().find(|b| b.name == name).ok_or_else(|| Error(format!("--box {name}: one of {}", BOXES.iter().map(|b| b.name).collect::<Vec<_>>().join(", "))))?;
    let split = a.has("by-layers");
    let n: usize = a.num("gpus", b.gpus)?;
    let pos: usize = a.num("pos", 512)?;
    let (cfg, ts) = tensors(model)?;
    let mt = cfg.get("model_type").and_then(Value::as_str).unwrap_or("");
    // (DeepSeek-V4's FP8 matrices are held exact, as released; elsewhere layout H's plain packs on HBM, coded on the others)
    let f8_pct: u64 = a.num("fp8", if mt == "deepseek_v4" { 100 } else if b.hbm { PLAIN_PCT } else { CODED_PCT })?;
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
    // (a product's own time beside its bytes: its start and its last wave, measured on the PC at the slices' sizes; the experts' product of a MoE layer more)
    let (t0, t0_moe): (f64, f64) = (a.num("t0-us", b.t0_us.0)?, a.num("t0-moe-us", b.t0_us.1)?);
    let t_products = (s.products as f64 * t0 + 2.0 * s.moe_layers as f64 * t0_moe) / 1e3;
    // (a decode step's gathers are small: tpgather's 1 x 896 a GPU, at the GPUs' count)
    let ex_us: f64 = a.num("exchange-us", b.gather_us[if n <= 2 { 0 } else if n <= 4 { 1 } else { 2 }])?;
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
    let ours_x = s.exchanges as f64 * ex_us / 1e3;
    let gib = |x: f64| x / (1u64 << 30) as f64;
    println!("{model} on {} ({}), {} GPU{} {}, position {pos}, experts found on the GPUs {:.0}%", b.name, b.read_src, n, if n == 1 { "" } else { "s" }, if split { "split by layers" } else if n > 1 { "every layer across them" } else { "" }, hit * 100.0);
    println!("  per GPU a token: weights {:.2} GiB ({:.2} dense: {:.2} sliced, {:.2} whole; {:.2} experts x {:.2} found), attention {:.3} GiB ({}); missing experts {:.2} GiB in all", gib(w), gib(s.dense), gib(s.sliced), gib(s.whole), gib(s.expert), hit, gib(s.attn), s.attn_note, gib(missing));
    println!("  {:<34} {:>10} {:>10}  {}", "term", "ceiling", "ours", "how");
    println!("  {:<34} {:>8.2}ms {:>8.2}ms  bytes / {:.0} GB/s (bf16 packed to {:.1}%); ours at {:.0}% bf16, {:.0}% FP8, {:.0}% FP4 experts ({src}), and {} products x {t0} us + {} experts' x {t0_moe} us = {t_products:.2} ms", "weights", t_w, ours_w, b.read, bf16_share * 100.0, e_bf * 100.0, e_f8 * 100.0, e_fp4 * 100.0, s.products, 2 * s.moe_layers);
    println!("  {:<34} {:>8.2}ms {:>8}  bytes / {:.0} GB/s", "attention's cache and states", t_a, "-", b.read);
    if missing > 0.0 {
        println!("  {:<34} {:>8.2}ms {:>8.2}ms  host memory {:.0} GB/s bounds it; ours: copies {:.1} GB/s ({}) + the host's pairs {:.1} GB/s ({pair_ms} ms a pair)", "missing experts", t_m, ours_m, b.host, pcie, b.pcie_src, host_ours);
    }
    if s.exchanges > 0 {
        println!("  {:<34} {:>8.2}ms {:>8.2}ms  {} exchanges x {ceil_ex_us} us (one NVLink write and a kernel); ours {ex_us} us each", "exchanges", t_x, ours_x, s.exchanges);
    }
    if fixed_us > 0.0 {
        println!("  {:<34} {:>10} {:>8.2}ms  measured on the PC: the small kernels, and the cache's plan a MoE layer ({fixed_us} us a step)", "fixed", "-", fixed_us / 1e3);
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

    /// Kimi-K3's packs as one product each, sliced as one whatever each tensor's size: a delta attention layer's q, k, v, gate, f_a and beta, a latent
    /// attention's q_a, gate and kv_a, a MoE layer's latent down projection and the shared experts' gate and up.
    #[test]
    fn kimi_packs_are_one_product() {
        let cfg = json::parse(r#"{"model_type": "kimi_linear", "num_hidden_layers": 2, "num_experts": 4, "num_experts_per_token": 2, "linear_attn_config": {"full_attn_layers": [2]}}"#).unwrap();
        let m = |name: String, r: u64, c: u64| (name, "BF16".to_string(), vec![r, c], r * c * 2);
        let (a, b, f) = ("model.layers.0.self_attn", "model.layers.1.self_attn", "model.layers.1.block_sparse_moe");
        let mut ts: Vec<_> = ["q_proj", "k_proj", "v_proj", "g_proj", "o_proj"].iter().map(|p| m(format!("{a}.{p}.weight"), 4096, 4096)).collect();
        ts.extend([m(format!("{a}.f_a_proj.weight"), 128, 4096), m(format!("{a}.b_proj.weight"), 32, 4096), m(format!("{a}.f_b_proj.weight"), 4096, 128)]);
        ts.extend([m(format!("{b}.q_a_proj.weight"), 1536, 4096), m(format!("{b}.g_proj.weight"), 4096, 4096), m(format!("{b}.kv_a_proj_with_mqa.weight"), 576, 4096), m(format!("{b}.o_proj.weight"), 4096, 4096)]);
        ts.extend([m(format!("{f}.routed_expert_down_proj.weight"), 2048, 4096), m(format!("{f}.shared_experts.gate_proj.weight"), 2048, 4096), m(format!("{f}.shared_experts.up_proj.weight"), 2048, 4096)]);
        ts.extend([m(format!("{f}.shared_experts.down_proj.weight"), 4096, 2048), m(format!("{f}.routed_expert_up_proj.weight"), 4096, 2048)]);
        for e in 0..4 {
            ts.push((format!("{f}.experts.{e}.w1.weight"), "U8".into(), vec![512, 2048], 512 * 2048));
        }
        let s = step(&cfg, &ts, 2, TWELVE_ACTUAL, 100, 512).unwrap();
        // (layer 0: the six as one, f_b, o; layer 1: q_a, gate and kv_a as one, o, the latent's and the shared gate and up as one, the shared down, the latent up)
        assert_eq!(s.products, 3 + 5, "{s:?}");
        // (f_b too small to slice; the others gathered, and the experts' two)
        assert_eq!(s.exchanges, 2 + 5 + 2, "{s:?}");
    }
}
