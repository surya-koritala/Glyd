//! FROZEN (R13, #234): see src/model_glm.rs. `random` (a random GLM-layout model) is also the dynamic route's logits tests', and stays with them.
//! GLM-5.3-Flash's kernels and layers on the GPU, their outputs written as files for tools/glm_ref.py to compare with its PyTorch reference (a development build's):
//!   `glyd-engine glmkernels IN OUT`   the kernels (kernels/glm.cu) on random inputs: tools/glm_ref.py `kernels gen`'s files, `kernels check` compares (tools/glm_emu.cpp runs the same sequence on the host); the Kimi delta
//!                                     attention's, the hyper-connections' and the latent attention's (mla.txt)
//!   `glyd-engine glmlayers DIR IO`    whole layers (the token mixer's half, Kimi delta or latent attention, and the feed-forward's) of the real model (tools/glm_ref.py `fetch` made DIR: their tensors by range from the Hub) or of the random tiny model
//!                                     `tiny` made, through the engine's own code, the cases of IO/spec.json
//!   `glyd-engine glmtoy DIR IO`       the random tiny model whole: prefill and decode steps, as a graph and not, the logits of every step
//!   `glyd-engine glmfp8 DIR OUT`      the FP8 matrices of DIR as `Checkpoint::read_bf16` reads them (the bf16 they dequantise to), for tools/glm_ref.py `fp8 check` to compare bit for bit with transformers'
//!   `glyd-engine glmdecode DIR OUT --layers L,L --level 6|5|4`   (no GPU) the matrices a mixture-of-experts layer's block holds as packs (the shared expert and the experts), packed from DIR at that level and decoded back to
//!                                     bf16 as the GPU's products read them, as a mini checkpoint that `layers check --decoded` takes: the reference then runs on the weights the engine multiplies by
//! Files are raw little-endian bf16 (u16) unless noted; *.txt hold numbers.

use crate::dev::{Buf, Dev, Graph};
use crate::ffi::{err, Error, Result};
use crate::glm::{ConfigGlm, GlmKind};
use crate::mla_glm::unpacked;
use crate::model_glm::{EngineGlm, Hc, KernGlm, OptsGlm, GM_D, GM_HC, GM_NP};
use crate::moe_glm::factory;
use crate::weights::Layout;
use glyd_gpu::json::{self, Value};
use glyd_gpu::safetensors::Checkpoint;
use std::path::Path;
use std::time::Instant;

fn io<T>(r: std::io::Result<T>) -> Result<T> {
    r.map_err(|e| Error(e.to_string()))
}

fn nums(dir: &Path, name: &str) -> Result<Vec<f64>> {
    let t = io(std::fs::read_to_string(dir.join(name)))?;
    t.split_whitespace().map(|x| x.parse::<f64>().map_err(|_| Error(format!("{name}: {x}: not a number")))).collect()
}

fn load(dir: &Path, name: &str) -> Result<Buf> {
    let b = io(std::fs::read(dir.join(name)))?;
    let buf = Buf::new(b.len())?;
    buf.write(0, &b)?;
    Ok(buf)
}

fn save(dir: &Path, name: &str, b: &Buf, offset: usize, bytes: usize) -> Result<()> {
    let mut v = vec![0u8; bytes];
    b.read(offset, &mut v)?;
    io(std::fs::write(dir.join(name), v))
}

fn save_bytes<T: Copy>(dir: &Path, name: &str, v: &[T]) -> Result<()> {
    io(std::fs::write(dir.join(name), crate::weights::as_bytes(v)))
}

/// A copy of `b` on the device.
fn copy(b: &Buf) -> Result<Buf> {
    let mut v = vec![0u8; b.bytes];
    b.read(0, &mut v)?;
    let c = Buf::new(b.bytes)?;
    c.write(0, &v)?;
    Ok(c)
}

/// The level the mixture-of-experts layers' matrices are held at for `bits` (the tiny model's `--level`: 6, 5 or 4; none: 6).
fn level_of(bits: Option<usize>) -> Result<Layout> {
    match bits {
        None | Some(6) => Ok(Layout::Int6),
        Some(5) => Ok(Layout::Int5),
        Some(4) => Ok(Layout::Int4),
        Some(b) => err(format!("a level of {b} bits: the mixture-of-experts layers are held in 6, 5 or 4")),
    }
}

// ---- glmkernels --------------------------------------------------------------------------------------------------------------------------------------------------------------------

pub fn kernels(ind: &Path, outd: &Path) -> Result<()> {
    io(std::fs::create_dir_all(outd))?;
    let dev = Dev::new(0)?;
    let k = KernGlm::load(&dev)?;
    let s = dev.stream;
    let m = nums(ind, "kda.txt")?;
    let (heads, n, split, bound, eps) = (m[0] as usize, m[1] as usize, m[2] as usize, m[3] as f32, m[4] as f32);
    let qd = heads * GM_D;
    let (xc, fg, bb, og, ow) = (load(ind, "xc.bin")?, load(ind, "fg.bin")?, load(ind, "bb.bin")?, load(ind, "og.bin")?, load(ind, "ow.bin")?);
    let (dtb, alog, state0) = (load(ind, "dtb.bin")?, load(ind, "alog.bin")?, load(ind, "state0.bin")?);
    let first = if split > 0 && split < n { split } else { n };
    // (first token, tokens, the delta rule in chunks) a call
    type Call = (usize, usize, bool);
    let cases: [(&str, Vec<Call>); 4] = [
        ("chunked", [vec![(0, first, true)], if first < n { vec![(first, n - first, true)] } else { vec![] }].concat()),
        ("recurrent", [vec![(0, first, false)], if first < n { vec![(first, n - first, false)] } else { vec![] }].concat()),
        ("steps", (0..n).map(|t| (t, 1, false)).collect()),
        ("mixed", [vec![(0, first, true)], (first..n).map(|t| (t, 1, false)).collect()].concat()),
    ];
    for (name, calls) in &cases {
        let (core, att, state) = (Buf::zeroed(n * qd * 2)?, Buf::zeroed(n * qd * 2)?, copy(&state0)?);
        dev.sync()?;
        let t0 = Instant::now();
        for &(a, len, chunked) in calls {
            // SAFETY: buffers made here of the sizes the files give; the stream orders the calls and is synchronized before they are read.
            unsafe {
                let (x, f, b) = (xc.ptr + (a * 3 * qd * 2) as u64, fg.ptr + (a * qd * 2) as u64, bb.ptr + (a * heads * 2) as u64);
                k.delta(s, chunked, x, f, dtb.ptr, alog.ptr, b, state.ptr, core.ptr + (a * qd * 2) as u64, len as i32, heads as i32, bound)?;
            }
        }
        dev.sync()?;
        println!("delta rule case {name}: {} calls, {:.1} ms", calls.len(), t0.elapsed().as_secs_f64() * 1e3);
        // SAFETY: as above.
        unsafe { k.delta_out(s, core.ptr, og.ptr, ow.ptr, false, att.ptr, n as u32, heads as i32, eps)? };
        dev.sync()?;
        save(outd, &format!("{name}_core.bin"), &core, 0, core.bytes)?;
        save(outd, &format!("{name}_att.bin"), &att, 0, att.bytes)?;
        save(outd, &format!("{name}_state.bin"), &state, 0, state.bytes)?;
    }
    // the convolution of q, k and v (qwen35.cu's, as the layers run it): three calls of different lengths from a state of zeros, as one sequence
    if ind.join("cx.bin").exists() {
        let cx = load(ind, "cx.bin")?;
        let (cw, cst) = (load(ind, "cw.bin")?, load(ind, "cst0.bin")?);
        let y = Buf::zeroed(n * 3 * qd * 2)?;
        let cuts = [0, n / 3, n / 3 + 1, n];
        for w in cuts.windows(2) {
            let len = w[1] - w[0];
            if len > 0 {
                // SAFETY: as above.
                unsafe {
                    k.conv(
                        s,
                        cx.ptr + (w[0] * 3 * qd * 2) as u64,
                        (3 * qd) as i32,
                        cw.ptr,
                        false,
                        cst.ptr,
                        y.ptr + (w[0] * 3 * qd * 2) as u64,
                        len as i32,
                        (3 * qd) as i32,
                    )?
                };
            }
        }
        dev.sync()?;
        save(outd, "conv.bin", &y, 0, y.bytes)?;
        save(outd, "cstate.bin", &cst, 0, cst.bytes)?;
    }
    // the streams
    let h = nums(ind, "hc.txt")?;
    let (d, n2, heps, hc_eps, iters) = (h[0] as usize, h[1] as usize, h[2] as f32, h[3] as f32, h[4] as i32);
    let hc = Hc { proj: load(ind, "hfn.bin")?, base: load(ind, "hbase.bin")?, scale: load(ind, "hscale.bin")? };
    let (hx, hnw, ho) = (load(ind, "hx.bin")?, load(ind, "hnw.bin")?, load(ind, "ho.bin")?);
    let (hh, hcp, post, head, spread) =
        (Buf::zeroed(n2 * d * 2)?, Buf::zeroed(n2 * GM_NP * 4)?, copy(&hx)?, Buf::zeroed(n2 * d * 2)?, Buf::zeroed(n2 * GM_HC * d * 2)?);
    // SAFETY: as above.
    unsafe {
        k.hc_mix(s, hx.ptr, &hc, hnw.ptr, hh.ptr, hcp.ptr, n2 as u32, d as i32, heps, hc_eps, iters)?;
        k.hc_post(s, post.ptr, ho.ptr, hcp.ptr, n2 as u32, d as i32)?;
        // (a decode step's: the first rows through the split projection)
        let m8 = n2.min(crate::model_glm::HC_FEW as usize);
        let (hcp_s, h_s) = (Buf::zeroed(m8 * GM_NP * 4)?, Buf::zeroed(m8 * d * 2)?);
        k.hc_mix(s, hx.ptr, &hc, hnw.ptr, h_s.ptr, hcp_s.ptr, m8 as u32, d as i32, heps, hc_eps, iters)?;
        dev.sync()?;
        save(outd, "hc_params_s.bin", &hcp_s, 0, hcp_s.bytes)?;
        save(outd, "hc_h_s.bin", &h_s, 0, h_s.bytes)?;
        k.hc_head(s, hx.ptr, hnw.ptr, head.ptr, n2 as u32, d as i32, heps)?;
        k.streams(s, ho.ptr, spread.ptr, n2 as u32, d as i32)?;
    }
    dev.sync()?;
    save(outd, "hc_params.bin", &hcp, 0, hcp.bytes)?;
    save(outd, "hc_h.bin", &hh, 0, hh.bytes)?;
    save(outd, "hc_post.bin", &post, 0, post.bytes)?;
    save(outd, "hc_head.bin", &head, 0, head.bytes)?;
    save(outd, "hc_streams.bin", &spread, 0, spread.bytes)?;
    println!("streams done");
    if ind.join("mla.txt").exists() {
        mla_kernels(&dev, &k, ind, outd)?;
    }
    Ok(())
}

/// The latent attention's kernels on mla.txt's inputs (tools/glm_ref.py `kernels gen`; tools/glm_emu.cpp runs the same sequence): the rows of a sequence in calls, each through the norms into the caches, the pool keys, the absorbed
/// queries, the indexer's scores and choice, the attention and the values; the caches and every row's results written as mla_*.bin for `kernels check`.
fn mla_kernels(dev: &Dev, k: &KernGlm, ind: &Path, outd: &Path) -> Result<()> {
    let m = nums(ind, "mla.txt")?;
    let [nh, nope, vd, kr, ih, id, kp, topk, n, nc] = [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(|i| m[i] as usize);
    let calls: Vec<usize> = m[10..10 + nc].iter().map(|&x| x as usize).collect();
    let eps = m[10 + nc] as f32;
    let tail = m.get(11 + nc).is_none_or(|&x| x != 0.0);
    let (ksel, tmax, pmax) = (topk / kp, topk + kp - 1, n / kp + 1);
    let ld = |name: &str| load(ind, &format!("mla_{name}.bin"));
    let (kvraw, kvw, ikraw, knw, knb, gsraw, ape, iq, wts, q, wkv) =
        (ld("kvraw")?, ld("kvw")?, ld("ikraw")?, ld("knw")?, ld("knb")?, ld("gsraw")?, ld("ape")?, ld("iq")?, ld("wts")?, ld("q")?, ld("wkv")?);
    let posb = Buf::new(n * 4)?;
    posb.write(0, &(0..n as i32).collect::<Vec<_>>())?;
    let (lat, ikc, gsc, pkc) = (Buf::zeroed(n * kr * 2)?, Buf::zeroed(n * id * 2)?, Buf::zeroed(n * id * 2)?, Buf::zeroed(pmax * id * 2)?);
    let (qa, scores, sel, cnt, olat, ov) = (
        Buf::zeroed(n * nh * kr * 4)?,
        Buf::zeroed(n * pmax * 4)?,
        Buf::zeroed(n * tmax * 4)?,
        Buf::zeroed(n * 4)?,
        Buf::zeroed(n * nh * kr * 4)?,
        Buf::zeroed(n * nh * vd * 2)?,
    );
    let (scale, iscale, wscale) = ((nope as f32).powf(-0.5), (id as f32).powf(-0.5), (ih as f32).powf(-0.5));
    let part = Buf::zeroed(crate::model_glm::HC_FEW as usize * nh * 8 * (kr + 2) * 4)?;
    let (s, t0) = (dev.stream, Instant::now());
    let mut a = 0usize;
    for &len in &calls {
        let (l, ps) = (len as u32, posb.ptr + (a * 4) as u64);
        // SAFETY: buffers made here of the sizes the files give; the stream orders the launches and is synchronized before the results are read.
        unsafe {
            k.rms2(s, kvraw.ptr + (a * kr * 2) as u64, kr as i32, kvw.ptr, lat.ptr, kr as i32, ps, kr as i32, eps, l)?;
            k.layernorm(s, ikraw.ptr + (a * id * 2) as u64, id as i32, knw.ptr, knb.ptr, ikc.ptr, id as i32, ps, id as i32, 1e-6, l)?;
            k.scatter(s, gsraw.ptr + (a * id * 2) as u64, id as i32, gsc.ptr, id as i32, ps, id as i32, l)?;
            k.kp_keys(s, ikc.ptr, gsc.ptr, ape.ptr, pkc.ptr, ps, len as i32, id as i32, kp as i32)?;
            k.absorb(
                s,
                q.ptr + (a * nh * nope * 2) as u64,
                (nh * nope) as i32,
                wkv.ptr,
                qa.ptr + (a * nh * kr * 4) as u64,
                len as i32,
                nh as i32,
                nope as i32,
                vd as i32,
                kr as i32,
            )?;
            k.idx_score(
                s,
                iq.ptr + (a * ih * id * 2) as u64,
                (ih * id) as i32,
                wts.ptr + (a * ih * 2) as u64,
                ih as i32,
                pkc.ptr,
                ps,
                scores.ptr,
                pmax as i32,
                ih as i32,
                id as i32,
                kp as i32,
                ksel as i32,
                iscale,
                wscale,
                l,
            )?;
            k.select(s, scores.ptr, pmax as i32, ps, sel.ptr + (a * tmax * 4) as u64, cnt.ptr + (a * 4) as u64, tmax as i32, kp as i32, ksel as i32, tail, l)?;
            k.mla_attn(
                s,
                qa.ptr + (a * nh * kr * 4) as u64,
                lat.ptr,
                sel.ptr + (a * tmax * 4) as u64,
                cnt.ptr + (a * 4) as u64,
                olat.ptr + (a * nh * kr * 4) as u64,
                part.ptr,
                tmax as i32,
                nh as i32,
                kr as i32,
                scale,
                l,
            )?;
            k.expand_v(
                s,
                olat.ptr + (a * nh * kr * 4) as u64,
                wkv.ptr,
                ov.ptr + (a * nh * vd * 2) as u64,
                len as i32,
                nh as i32,
                nope as i32,
                vd as i32,
                kr as i32,
            )?;
        }
        a += len;
    }
    dev.sync()?;
    println!("latent attention kernels: {n} rows in {nc} calls, {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
    for (name, b) in [("lat", &lat), ("ikc", &ikc), ("gsc", &gsc), ("pk", &pkc), ("qa", &qa), ("sel", &sel), ("cnt", &cnt), ("olat", &olat), ("ov", &ov)] {
        save(outd, &format!("mla_{name}.bin"), b, 0, b.bytes)?;
    }
    Ok(())
}

// ---- glmlayers ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// The rows `rows` (indices within a call) of `v`, which has `width` values a row.
fn pick_rows<T: Copy>(v: &[T], width: usize, rows: &[usize]) -> Vec<T> {
    rows.iter().flat_map(|&i| v[i * width..(i + 1) * width].iter().copied()).collect()
}

/// The stages of a case that has a "sample" (see `layers`) that are dumped whole whatever the windows are: the checks run the convolution and the delta rule over all of a sequence's rows (Kimi delta attention), and the
/// pooling of the indexer's keys and the attention over the latents (latent attention) from the caches of all of them.
const WHOLE_KDA: [&str; 4] = ["mix", "conv", "fg", "bb"];
const WHOLE_MLA: [&str; 3] = ["c", "ik", "gs"];

/// The cases of IO/spec.json, {"chunk": N, "level": 4 | 5 | 6, "cases": [{"name", "layer", "mode": "prefill" | "decode" | "mixed", "calls": [tokens a call] (prefill), "tokens": N (decode, mixed), "prefill": N (mixed),
/// "graph": whether single tokens are replayed from a CUDA graph, "chunked": whether a prompt's delta rule runs in chunks (default: yes)}]}: each case a layer on the first rows of DIR/x.bin (the streams, bf16 [rows, 4,
/// hidden]), both halves of it; "level": the bits of the level a mixture-of-experts layer's matrices are held at (6, 5 or 4; default 6). Written per case, the rows of every call one after another: `hcA` and `hcF` (fp32 [rows,
/// 24]: the mixing weights of the two halves), the token mixer's half: `h1` (the collapse, normed), `mix`, `conv`, `fg`, `bb`, `og`, `core`, `att`, `o1` (the mixer's output), `x1` (the streams after it); the feed-forward's: `h2`,
/// `gu` and `act` (a dense MLP's), `o2` (the block's output), `x2` (the streams after the layer), with the experts the router chose, `ids` int64 [rows, k], and their weights, `wts` fp32, for a mixture-of-experts layer; at the
/// end the state (`ssm` fp32, `cstate` bf16).
/// A case with "sample": [[first row, rows], ...] dumps those windows of the sequence only (a long sequence's dumps are too many bytes to keep whole), in ascending order, except the stages the checks need whole (`WHOLE_KDA`,
/// `WHOLE_MLA`, and the latent attention's final caches and pooled keys); the engine still runs every row.
pub fn layers(dir: &Path, outd: &Path) -> Result<()> {
    let spec = json::parse(&io(std::fs::read_to_string(outd.join("spec.json")))?).map_err(|e| Error(format!("spec.json: {e}")))?;
    let cases = spec.get("cases").and_then(Value::as_array).ok_or_else(|| Error("spec.json: no cases".into()))?;
    let get = |c: &Value, k: &str| c.get(k).and_then(Value::as_u64).map(|x| x as usize);
    let mut wanted_layers: Vec<usize> = cases.iter().filter_map(|c| get(c, "layer")).collect();
    wanted_layers.sort_unstable();
    wanted_layers.dedup();
    let xb = io(std::fs::read(dir.join("x.bin")))?;
    let hid = ConfigGlm::load(dir)?.hidden;
    let total = xb.len() / 2 / (GM_HC * hid);
    let o = OptsGlm { ctx: total + 64, chunk: get(&spec, "chunk").unwrap_or(512), graph: false, only: Some(wanted_layers.clone()), ..Default::default() };
    let make = factory(level_of(get(&spec, "level"))?);
    let make_mla = crate::mla_glm::factory();
    let mut e = EngineGlm::new(dir, &o, Some(&make), Some(&make_mla))?;
    let c = e.w.cfg.clone();
    println!("{}", e.describe());
    let x_words: Vec<u16> = xb.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect();
    let (qd, heads, s) = (c.qkv(), c.kda.heads, e.dev.stream);
    // (one step of each layer's two halves before the first case: the lazy initializations, and the scratch of the mixture-of-experts layers, are made outside a graph's capture; the states are then zeroed)
    e.x.write(0, &x_words[..GM_HC * hid])?;
    e.pos.write(0, &[0i32])?;
    for &l in &wanted_layers {
        let ly = e.w.layers[l].as_ref().expect("a loaded layer");
        // SAFETY: live buffers of the engine, sized for a chunk; the stream orders every step.
        unsafe {
            e.mixer_half(ly, 1, e.pos.ptr, 0)?;
            e.mlp_half(ly, 1)?;
        }
    }
    e.dev.sync()?;
    for case in cases {
        let name = case.get("name").and_then(Value::as_str).ok_or_else(|| Error("a case has no name".into()))?;
        let l = get(case, "layer").ok_or_else(|| Error(format!("{name}: no layer")))?;
        let mode = case.get("mode").and_then(Value::as_str).unwrap_or("prefill");
        let graph = case.get("graph").and_then(Value::as_bool).unwrap_or(false);
        let windows: Option<Vec<(usize, usize)>> = case
            .get("sample")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(|w| w.as_u64s().filter(|w| w.len() == 2).map(|w| (w[0] as usize, w[1] as usize))).collect());
        let sampled = |r: usize| windows.as_ref().is_none_or(|w| w.iter().any(|&(first, k)| r >= first && r < first + k));
        let (kind, moe) = e.w.layers[l].as_ref().map(|ly| (ly.kind, ly.moe.is_some())).ok_or_else(|| Error(format!("{name}: layer {l} is not loaded")))?;
        // the calls: (first row, rows)
        let mut calls: Vec<(usize, usize)> = Vec::new();
        let mut at = 0;
        let prefill_calls = |sizes: Vec<usize>, at: &mut usize, calls: &mut Vec<(usize, usize)>| {
            for n in sizes {
                calls.push((*at, n));
                *at += n;
            }
        };
        match mode {
            "prefill" => {
                prefill_calls(case.get("calls").and_then(Value::as_u64s).unwrap_or_default().into_iter().map(|v| v as usize).collect(), &mut at, &mut calls)
            }
            "decode" => prefill_calls(vec![1; get(case, "tokens").unwrap_or(1)], &mut at, &mut calls),
            "mixed" => {
                prefill_calls(vec![get(case, "prefill").unwrap_or(1)], &mut at, &mut calls);
                prefill_calls(vec![1; get(case, "tokens").unwrap_or(1).saturating_sub(at)], &mut at, &mut calls);
            }
            m => return err(format!("{name}: mode {m}")),
        }
        if at > total {
            return err(format!("{name}: {at} rows, where x.bin has {total}"));
        }
        if let Some(&(_, n)) = calls.iter().max_by_key(|c| c.1) {
            if n > e.chunk() {
                return err(format!("{name}: a call of {n} rows, where the chunk is {}", e.chunk()));
            }
        }
        e.reset()?;
        e.chunked = case.get("chunked").and_then(Value::as_bool).unwrap_or(true);
        // (a latent attention layer's calls of more rows than a block of its attention keep every block's stages)
        if kind == GlmKind::Mla {
            if let Some(b) = e.w.layers[l].as_ref().and_then(|ly| ly.mla.as_ref()) {
                b.collect_stages(true);
            }
        }
        // the stages dumped: (name, width of a row), and the engine's buffer each is read from
        let live_a: Vec<(&str, usize, &Buf)> = if kind == GlmKind::Kda {
            vec![
                ("h1", hid, &e.h),
                ("mix", 3 * qd, &e.mix),
                ("conv", 3 * qd, &e.xc),
                ("fg", qd, &e.fg),
                ("bb", heads, &e.bb),
                ("og", qd, &e.og),
                ("core", qd, &e.core),
                ("att", qd, &e.att),
                ("o1", hid, &e.o),
                ("x1", GM_HC * hid, &e.x),
            ]
        } else {
            vec![("h1", hid, &e.h), ("o1", hid, &e.o), ("x1", GM_HC * hid, &e.x)]
        };
        // (a latent attention layer's own stages, a call after another)
        let mut extra: std::collections::BTreeMap<&'static str, Vec<u8>> = Default::default();
        let live_f: Vec<(&str, usize, &Buf)> = if moe {
            vec![("h2", hid, &e.h), ("o2", hid, &e.o), ("x2", GM_HC * hid, &e.x)]
        } else {
            vec![("h2", hid, &e.h), ("gu", 2 * c.inter, &e.mix), ("act", c.inter, &e.xc), ("o2", hid, &e.o), ("x2", GM_HC * hid, &e.x)]
        };
        let mut got: Vec<Vec<u16>> = vec![Vec::new(); live_a.len() + live_f.len()];
        let mut params: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
        // (a mixture-of-experts layer's experts and their weights, a call after another)
        let mut chosen: (Vec<i64>, Vec<f32>) = (Vec::new(), Vec::new());
        let mut graphs: Option<[Graph; 2]> = None;
        let t0 = Instant::now();
        let mut spent = 0.0;
        for &(a, n) in &calls {
            let rows: Vec<usize> = (0..n).filter(|&i| sampled(a + i)).collect();
            e.x.write(0, &x_words[a * GM_HC * hid..(a + n) * GM_HC * hid])?;
            // (a call's two halves are run apart, so that the stages of each are read before the other's overwrite them)
            let run_half = |half: usize, e: &EngineGlm, pos: u64| -> Result<()> {
                let ly = e.w.layers[l].as_ref().ok_or_else(|| Error(format!("layer {l} is not loaded")))?;
                // SAFETY: live buffers of the engine, sized for a chunk; the stream orders every step.
                unsafe {
                    if half == 0 {
                        e.mixer_half(ly, n, pos, a)
                    } else {
                        e.mlp_half(ly, n)
                    }
                }
            };
            if n == 1 && graph {
                if graphs.is_none() {
                    // (nothing runs during a capture)
                    let ga = e.dev.capture(|| run_half(0, &e, e.st.ptr))?;
                    let gb = e.dev.capture(|| run_half(1, &e, e.st.ptr))?;
                    graphs = Some([ga, gb]);
                }
                e.st.write(0, &[a as i32])?;
            } else {
                let pos: Vec<i32> = (a..a + n).map(|p| p as i32).collect();
                e.pos.write(0, &pos)?;
            }
            for (half, p) in params.iter_mut().enumerate() {
                let t2 = Instant::now();
                if n == 1 && graph {
                    // SAFETY: the buffers and kernels this check captured are alive and unchanged since the capture.
                    unsafe { graphs.as_ref().expect("graphs")[half].launch(s) }?;
                } else {
                    run_half(half, &e, e.pos.ptr)?;
                }
                e.dev.sync()?;
                spent += t2.elapsed().as_secs_f64();
                if half == 0 && kind == GlmKind::Mla {
                    let block = e.w.layers[l]
                        .as_ref()
                        .and_then(|ly| ly.mla.as_ref())
                        .ok_or_else(|| Error(format!("{name}: layer {l} has no latent attention block")))?;
                    for (nm, bytes) in block.stages(&e, n, a)? {
                        let whole = rows.len() == n || WHOLE_MLA.contains(&nm) || n == 0 || bytes.len() % n != 0;
                        extra.entry(nm).or_default().extend(if whole { bytes } else { pick_rows(&bytes, bytes.len() / n, &rows) });
                    }
                }
                let (live, offset) = if half == 0 { (&live_a, 0) } else { (&live_f, live_a.len()) };
                for (i, (nm, width, buf)) in live.iter().enumerate() {
                    let mut v = vec![0u16; n * width];
                    buf.read(0, &mut v)?;
                    if rows.len() == n || (kind == GlmKind::Kda && WHOLE_KDA.contains(nm)) {
                        got[offset + i].extend(v);
                    } else {
                        got[offset + i].extend(pick_rows(&v, *width, &rows));
                    }
                }
                let mut v = vec![0f32; n * GM_NP];
                e.hcp[half].read(0, &mut v)?;
                p.extend(if rows.len() == n { v } else { pick_rows(&v, GM_NP, &rows) });
            }
            if moe {
                let sc = e.moe_scratch.get().ok_or_else(|| Error(format!("{name}: the mixture-of-experts layer made no scratch")))?;
                let (mut ids, mut wts) = (vec![0i64; n * c.experts.top_k], vec![0f32; n * c.experts.top_k]);
                sc.ids.read(0, &mut ids)?;
                sc.wts.read(0, &mut wts)?;
                let k = c.experts.top_k;
                if rows.len() == n {
                    chosen.0.extend(ids);
                    chosen.1.extend(wts);
                } else {
                    chosen.0.extend(pick_rows(&ids, k, &rows));
                    chosen.1.extend(pick_rows(&wts, k, &rows));
                }
            }
        }
        for (i, (nm, _, _)) in live_a.iter().chain(&live_f).enumerate() {
            save_bytes(outd, &format!("{name}.{nm}.bin"), &got[i])?;
        }
        save_bytes(outd, &format!("{name}.hcA.bin"), &params[0])?;
        save_bytes(outd, &format!("{name}.hcF.bin"), &params[1])?;
        if kind == GlmKind::Kda {
            let idx = e.w.layers[l].as_ref().expect("a loaded layer").idx;
            save(outd, &format!("{name}.ssm.bin"), &e.ssm, idx * heads * GM_D * GM_D * 4, heads * GM_D * GM_D * 4)?;
            save(outd, &format!("{name}.cstate.bin"), &e.convst, idx * 3 * 3 * qd * 2, 3 * 3 * qd * 2)?;
        } else {
            let block =
                e.w.layers[l].as_ref().and_then(|ly| ly.mla.as_ref()).ok_or_else(|| Error(format!("{name}: layer {l} has no latent attention block")))?;
            extra.extend(block.finish(&e, at)?);
            for (nm, bytes) in &extra {
                io(std::fs::write(outd.join(format!("{name}.{nm}.bin")), bytes))?;
            }
        }
        if moe {
            save_bytes(outd, &format!("{name}.ids.bin"), &chosen.0)?;
            save_bytes(outd, &format!("{name}.wts.bin"), &chosen.1)?;
        }
        let what =
            format!("{} and {}", if kind == GlmKind::Kda { "Kimi delta attention" } else { "latent attention" }, if moe { "experts" } else { "a dense MLP" });
        println!(
            "case {name}: layer {l} ({what}), {} calls, {at} rows: {:.1} ms in the layer ({:.3} ms a call), {:.1} ms with the dumps",
            calls.len(),
            spent * 1e3,
            spent * 1e3 / calls.len().max(1) as f64,
            t0.elapsed().as_secs_f64() * 1e3
        );
    }
    Ok(())
}

// ---- glmtoy ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// IO/toy.txt: the tokens to prefill, the tokens in all and the bits of the level the mixture-of-experts layers' matrices are held at (6, 5 or 4; default 6); IO/ids.txt: the token ids. The model of DIR (the random tiny
/// one: the layers of Kimi delta attention, a dense MLP and mixture-of-experts layers, the embedding, the head, the final norm) is run twice, with the decode step as a CUDA graph and without: the logits after the prefill and
/// after each step (the next input being the file's token) to IO/toy_graph_logits.bin and toy_eager_logits.bin (bf16 [steps, vocab]), the chosen tokens to *.txt.
pub fn toy(dir: &Path, outd: &Path) -> Result<()> {
    let t = nums(outd, "toy.txt")?;
    let (p, total) = (t[0] as usize, t[1] as usize);
    let level = level_of(t.get(2).map(|&b| b as usize))?;
    let ids: Vec<u32> = nums(outd, "ids.txt")?.into_iter().map(|x| x as u32).collect();
    if p == 0 || p > total || total > ids.len() {
        return err(format!("toy.txt: {p} of {total} tokens, ids.txt has {}", ids.len()));
    }
    for (graph, tag) in [(true, "graph"), (false, "eager")] {
        // (the prompt in two chunks, so that the delta rule's state and the convolution's pass from one to the next)
        let o = OptsGlm { ctx: total + 16, chunk: p.div_ceil(2).max(16), graph, chunked: true, ..Default::default() };
        let make = factory(level);
        let make_mla = crate::mla_glm::factory();
        let mut e = EngineGlm::new(dir, &o, Some(&make), Some(&make_mla))?;
        if graph != e.has_graph() {
            return err(format!("the decode graph was asked for: {graph}, the engine has one: {}", e.has_graph()));
        }
        println!("{tag}: {}", e.describe());
        // (the eager pass records, layer by layer and call by call, the experts the routers chose and the tokens the indexers picked: tools/glm_ref.py `toy` teacher-forces them into its reference)
        let traced: std::rc::Rc<std::cell::RefCell<std::collections::BTreeMap<String, Vec<u8>>>> = Default::default();
        if !graph {
            let sink = traced.clone();
            e.trace = Some(Box::new(move |e: &EngineGlm, l: usize, n: usize, start: usize| {
                let mut sink = sink.borrow_mut();
                let Some(ly) = e.w.layers[l].as_ref() else { return };
                if let Some(sc) = ly.moe.as_ref().and_then(|_| e.moe_scratch.get()) {
                    let mut ids = vec![0i64; n * e.w.cfg.experts.top_k];
                    if e.dev.sync().is_ok() && sc.ids.read(0, &mut ids).is_ok() {
                        sink.entry(format!("ids_L{l}")).or_default().extend(crate::weights::as_bytes(&ids));
                    }
                }
                if let Some(block) = ly.mla.as_ref() {
                    for (nm, bytes) in block.stages(e, n, start).unwrap_or_default() {
                        if nm == "sel" || nm == "cnt" {
                            sink.entry(format!("{nm}_L{l}")).or_default().extend(bytes);
                        }
                    }
                }
            }));
        }
        e.reset()?;
        let (mut logits, mut tokens) = (Vec::<u16>::new(), Vec::<String>::new());
        let grab = |e: &EngineGlm, logits: &mut Vec<u16>| -> Result<()> {
            logits.extend(e.logits()?.into_iter().map(|f| (f.to_bits() >> 16) as u16));
            Ok(())
        };
        let t0 = Instant::now();
        let s = e.prefill(&ids[..p])?;
        tokens.push(s.token.to_string());
        grab(&e, &mut logits)?;
        let pre = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        for &id in &ids[p..total] {
            let s = e.step(id)?;
            tokens.push(s.token.to_string());
            grab(&e, &mut logits)?;
        }
        println!(
            "{tag}: prefill of {p} tokens {:.1} ms, {} steps {:.2} ms each",
            pre * 1e3,
            total - p,
            t1.elapsed().as_secs_f64() * 1e3 / (total - p).max(1) as f64
        );
        save_bytes(outd, &format!("toy_{tag}_logits.bin"), &logits)?;
        io(std::fs::write(outd.join(format!("toy_{tag}_tokens.txt")), tokens.join(" ")))?;
        for (nm, bytes) in traced.borrow().iter() {
            io(std::fs::write(outd.join(format!("toy_{nm}.bin")), bytes))?;
        }
    }
    Ok(())
}

// ---- glmprof -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// `glmprof DIR LAYERS CTX REPS`: where a decode step of a layer spends its time. For each layer of DIR (the real model's mini checkpoint or the tiny one; tools/glm_ref.py `fetch` and `tiny` make them) the layer is run
/// over CTX positions (the rows of DIR/x.bin cycled, in chunks of 512), then one decode step at position CTX is captured into a CUDA graph with a marker event between its stages (`mark!`) and replayed REPS times: the
/// time between consecutive markers, averaged, is the stage's own in a graph's replay (its launch gaps included). A mixture-of-experts layer is held at six bits.
pub fn prof(dir: &Path, layers: &[usize], ctx: usize, reps: usize) -> Result<()> {
    let hid = ConfigGlm::load(dir)?.hidden;
    let xb = io(std::fs::read(dir.join("x.bin")))?;
    let x_words: Vec<u16> = xb.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect();
    let row = GM_HC * hid;
    let rows = x_words.len() / row;
    if rows == 0 {
        return err(format!("{}: no rows of streams in x.bin", dir.display()));
    }
    let chunk = 512.min(ctx.max(1));
    let o = OptsGlm { ctx: ctx + 64, chunk, graph: false, only: Some(layers.to_vec()), chunked: true, ..Default::default() };
    let make = factory(Layout::Int6);
    let make_mla = crate::mla_glm::factory();
    let mut e = EngineGlm::new(dir, &o, Some(&make), Some(&make_mla))?;
    println!("{}", e.describe());
    for &l in layers {
        e.reset()?;
        let (kind, moe) = e.w.layers[l].as_ref().map(|ly| (ly.kind, ly.moe.is_some())).ok_or_else(|| Error(format!("layer {l} is not loaded")))?;
        let ly = e.w.layers[l].as_ref().ok_or_else(|| Error(format!("layer {l} is not loaded")))?;
        let mut at = 0;
        while at < ctx {
            let n = chunk.min(ctx - at);
            let mut w = Vec::<u16>::with_capacity(n * row);
            for i in 0..n {
                let r = (at + i) % rows;
                w.extend_from_slice(&x_words[r * row..(r + 1) * row]);
            }
            e.x.write(0, &w)?;
            e.pos.write(0, &(at as i32..(at + n) as i32).collect::<Vec<_>>())?;
            // SAFETY: live buffers of the engine, sized for a chunk; the stream orders every step.
            unsafe {
                e.mixer_half(ly, n, e.pos.ptr, at)?;
                e.mlp_half(ly, n)?;
            }
            at += n;
        }
        e.dev.sync()?;
        let step = &x_words[..row];
        e.x.write(0, step)?;
        e.st.write(0, &[ctx as i32])?;
        *e.prof.borrow_mut() = Some(Vec::new());
        // SAFETY: as above; nothing runs during a capture.
        let g = e.dev.capture(|| unsafe {
            e.mixer_half(ly, 1, e.st.ptr, 0)?;
            e.mlp_half(ly, 1)
        })?;
        let marks = e.prof.borrow_mut().take().unwrap_or_default();
        let mut acc = vec![0f64; marks.len()];
        let mut whole = 0f64;
        for _ in 0..reps {
            e.x.write(0, step)?;
            // SAFETY: the buffers and kernels this check captured are alive and unchanged since the capture.
            unsafe { g.launch(e.dev.stream) }?;
            e.dev.sync()?;
            for i in 1..marks.len() {
                let ms = marks[i].1.since(&marks[i - 1].1)? as f64;
                acc[i] += ms;
                whole += ms;
            }
        }
        let per = |ms: f64| ms * 1e3 / reps.max(1) as f64;
        println!(
            "layer {l} ({} and {}), a decode step at position {ctx}: {:.1} us in a graph's replay",
            if kind == GlmKind::Kda { "Kimi delta attention" } else { "latent attention" },
            if moe { "experts" } else { "a dense MLP" },
            per(whole)
        );
        for i in 1..marks.len() {
            println!("   {:<36} {:>8.1} us {:>5.1} %", marks[i].0, per(acc[i]), 100.0 * acc[i] / whole.max(1e-9));
        }
    }
    Ok(())
}

// ---- glmmtp ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// `glmmtp DIR IO`: the MTP head on the random tiny model's hidden states. The model decodes IO/ids.txt (a prompt of one token, then a step a token) and the final normed hidden state of every position is kept; the head is
/// run on the rows (the hidden state of position j, the token j + 1) in calls of odd sizes, and writes IO/mtp_prev.bin (the hidden states, bf16 [rows, hidden]), mtp_out.bin (the head's normed hidden states) and
/// mtp_tok.txt (the token the LM head picks from each, and its logit) for `glm_ref.py mtp check`.
pub fn mtp(dir: &Path, outd: &Path) -> Result<()> {
    let t = nums(outd, "toy.txt")?;
    let level = level_of(t.get(2).map(|&b| b as usize))?;
    let ids: Vec<u32> = nums(outd, "ids.txt")?.into_iter().map(|x| x as u32).collect();
    let total = ids.len().min(120);
    let o = OptsGlm { ctx: total + 16, chunk: 64, graph: false, chunked: true, mtp: true, ..Default::default() };
    let make = factory(level);
    let make_mla = crate::mla_glm::factory();
    let mut e = EngineGlm::new(dir, &o, Some(&make), Some(&make_mla))?;
    println!("{}", e.describe());
    let head = e.mtp.take().ok_or_else(|| Error("the model has no MTP head (num_nextn_predict_layers)".into()))?;
    println!("the MTP head holds {:.3} GiB", head.bytes() as f64 / (1u64 << 30) as f64);
    // the model's hidden states, a position at a time
    e.reset()?;
    let hid = e.w.cfg.hidden;
    let mut hs: Vec<u16> = Vec::with_capacity(total * hid);
    e.prefill(&ids[..1])?;
    hs.extend(e.hidden()?);
    for &id in &ids[1..total - 1] {
        e.step(id)?;
        hs.extend(e.hidden()?);
    }
    let rows = total - 1;
    // the head on rows 0 .. rows in calls of these sizes (the first one a prompt's worth), the last row of each call and the token it picks kept
    let prev = Buf::new(rows * hid * 2)?;
    prev.write(0, &hs[..rows * hid])?;
    let (mut out, mut toks) = (Vec::<u16>::new(), Vec::<String>::new());
    let (mut at, sizes) = (0usize, [40usize, 1, 3, 1, 1, 7, 2, 1, 5, 1]);
    let mut k = 0;
    while at < rows {
        let n = sizes[k % sizes.len()].min(rows - at).min(e.chunk());
        k += 1;
        // SAFETY: `prev` holds the rows; the stream orders the head's launches and `picks` synchronizes.
        let hout = unsafe { head.forward(&e, prev.ptr + (at * hid * 2) as u64, &ids[at + 1..at + 1 + n], at)? };
        out.extend(head.read_hidden(&e, n)?);
        let mut r = 0;
        while r < n {
            let m = (n - r).min(crate::mtp_glm::ROWS);
            // SAFETY: the head's hidden states for these rows are in its scratch.
            let picks = unsafe { head.picks(&e, hout + (r * hid * 2) as u64, m)? };
            toks.extend(picks.iter().map(|(t, v)| format!("{t} {v}")));
            r += m;
        }
        at += n;
    }
    save_bytes(outd, "mtp_prev.bin", &hs[..rows * hid])?;
    save_bytes(outd, "mtp_out.bin", &out)?;
    io(std::fs::write(outd.join("mtp_tok.txt"), toks.join("\n")))?;
    println!("mtp: {rows} rows through the head");
    Ok(())
}

/// `glmmtpreal SNAPSHOT REF OUT LEVEL`: the real MTP head (the checkpoint's layer 45, its 288 experts at LEVEL bits, held alone on the GPU) on the reference's positions: the rows (the final normed hidden state of position
/// j that REF's `final_hidden` holds, the token j + 1) through the head in one call, the head's normed hidden states written to OUT/mtp_real_out.bin (bf16 [positions - 1, hidden]) for `glm_ref.py mtpreal` to turn into tokens
/// with the LM head and compare with the model's own picks (what a verified draft accepts) and with the text.
pub fn mtp_real(snap: &Path, refd: &Path, outd: &Path, level: Layout) -> Result<()> {
    io(std::fs::create_dir_all(outd))?;
    let fin = io(glyd_gpu::safetensors::File::open(&refd.join("final_ref128.safetensors")))?;
    let tokens: Vec<u32> = ref_tensor(&fin, "tokens")?.as_chunks::<4>().0.iter().map(|b| i32::from_le_bytes(*b) as u32).collect();
    let hidden: Vec<u16> = ref_tensor(&fin, "final_hidden")?.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect();
    let rows = tokens.len() - 1;
    let o = OptsGlm { ctx: rows + 16, chunk: rows, graph: false, chunked: true, layout: Some(level), only: Some(Vec::new()), mtp: true, ..Default::default() };
    let make = factory(level);
    let make_mla = crate::mla_glm::factory();
    let t0 = Instant::now();
    let mut e = EngineGlm::new(snap, &o, Some(&make), Some(&make_mla))?;
    let head = e.mtp.take().ok_or_else(|| Error("the model has no MTP head".into()))?;
    let hid = e.w.cfg.hidden;
    println!("the MTP head: {:.2} GiB, loaded in {:.1} s", head.bytes() as f64 / (1u64 << 30) as f64, t0.elapsed().as_secs_f64());
    let prev = Buf::new(rows * hid * 2)?;
    prev.write(0, &hidden[..rows * hid])?;
    let t1 = Instant::now();
    // SAFETY: `prev` holds the rows; the read synchronizes.
    unsafe { head.forward(&e, prev.ptr, &tokens[1..], 0)? };
    let out = head.read_hidden(&e, rows)?;
    println!("{rows} rows through the head in {:.1} ms", t1.elapsed().as_secs_f64() * 1e3);
    save_bytes(outd, "mtp_real_out.bin", &out)
}

// ---- glmspec -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// `glmspec DIR IO`: the verification of a draft in one pass and the rewind after it (`EngineGlm::verify`, `commit`, `generate_spec`) on the random tiny model: the tokens `generate` makes greedily, made again by rounds of
/// drafts that are the model's own continuation with a wrong guess put in by a schedule (none: every guess right; one at the place a round number says, so that the delta rules' states are rewound to each place in turn;
/// the first of every round; drafts of 0 to 3 tokens), must be the same tokens whatever the guesses are. IO/toy.txt and ids.txt as `toy`'s.
pub fn spec(dir: &Path, outd: &Path) -> Result<()> {
    let t = nums(outd, "toy.txt")?;
    let p = t[0] as usize;
    let level = level_of(t.get(2).map(|&b| b as usize))?;
    let ids: Vec<u32> = nums(outd, "ids.txt")?.into_iter().map(|x| x as u32).collect();
    let want = 56usize;
    if p == 0 || p + 1 > ids.len() {
        return err(format!("toy.txt: {p} tokens of prompt, ids.txt has {}", ids.len()));
    }
    let o = OptsGlm { ctx: p + want + 64, chunk: p.div_ceil(2).max(16), graph: false, chunked: true, spec: true, ..Default::default() };
    let make = factory(level);
    let make_mla = crate::mla_glm::factory();
    let mut e = EngineGlm::new(dir, &o, Some(&make), Some(&make_mla))?;
    println!("{}", e.describe());
    let prompt = &ids[..p];
    let plain: Vec<u32> = e.generate(prompt, want, true)?.iter().map(|s| s.token).collect();
    let vocab = e.w.cfg.vocab as u32;
    type Schedule = fn(usize) -> (usize, Option<usize>);
    let schedules: [(&str, Schedule); 5] = [
        ("every guess right, 4 a pass", |_| (4, None)),
        ("one wrong at 0, 1, 2, 3 in turn and then none, 4 a pass", |r| (4, Some(r % 5))),
        ("the first guess wrong every pass, 3 a pass", |_| (3, Some(0))),
        ("seven guesses a pass, one wrong at a place that moves", |r| (7, Some(r % 8))),
        ("drafts of 0 to 3 guesses, one wrong at the last place", |r| (r % 4, Some(r % 4).filter(|&k| k > 0).map(|k| k - 1))),
    ];
    let mut ok = true;
    for (name, schedule) in schedules {
        let mut round = 0usize;
        let mut draft = |all: &[u32]| -> Vec<u32> {
            let m = all.len() - prompt.len();
            let (k, wrong) = schedule(round);
            round += 1;
            let mut g: Vec<u32> = plain.iter().skip(m).take(k).copied().collect();
            if let Some(at) = wrong.filter(|&a| a < g.len()) {
                g[at] = (g[at] + 1) % vocab;
            }
            g
        };
        let (got, stats) = e.generate_spec(prompt, want, &mut draft)?;
        let got: Vec<u32> = got.iter().map(|s| s.token).collect();
        let same = got == plain;
        ok &= same;
        let first = got.iter().zip(&plain).position(|(a, b)| a != b);
        println!(
            "{} {name}: {} tokens {} generate's ({}), {} passes, {} guesses of which {} right",
            if same { "ok  " } else { "FAIL" },
            got.len(),
            if same { "are" } else { "are not" },
            first.map_or("no difference".to_string(), |i| format!("the first difference at token {i}: {} against {}", got[i], plain[i])),
            stats.passes,
            stats.drafted,
            stats.accepted
        );
    }
    // drafted by the MTP head (random weights: the guesses are mostly wrong, so the rewind is what is run; the tokens are still generate's)
    let o = OptsGlm { ctx: p + want + 64, chunk: p.div_ceil(2).max(16), graph: false, chunked: true, spec: true, mtp: true, ..Default::default() };
    drop(e);
    let mut e = EngineGlm::new(dir, &o, Some(&make), Some(&make_mla))?;
    for k in [1usize, 3, 7] {
        let (got, stats) = e.generate_mtp(prompt, want, k)?;
        let got: Vec<u32> = got.iter().map(|s| s.token).collect();
        let same = got == plain;
        ok &= same;
        let first = got.iter().zip(&plain).position(|(a, b)| a != b);
        println!(
            "{} the MTP head's drafts of {k}: {} tokens {} generate's ({}), {} passes, {} guesses of which {} right",
            if same { "ok  " } else { "FAIL" },
            got.len(),
            if same { "are" } else { "are not" },
            first.map_or("no difference".to_string(), |i| format!("the first difference at token {i}: {} against {}", got[i], plain[i])),
            stats.passes,
            stats.drafted,
            stats.accepted
        );
    }
    println!("spec: {}", if ok { "ok" } else { "FAILED" });
    if ok {
        Ok(())
    } else {
        err("a speculative generation differs from generate")
    }
}

// ---- glmreal -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// A tensor of the reference's safetensors file `f` as raw bytes.
fn ref_tensor(f: &glyd_gpu::safetensors::File, name: &str) -> Result<Vec<u8>> {
    let t = f.get(name).ok_or_else(|| Error(format!("{}: no tensor {name}", f.path.display())))?;
    io(f.read(t))
}

/// `glmreal SNAPSHOT REF OUT LAYERS LEVEL TEACHER`: the real model's layers `first .. end` one at a time on the GPU (each made from the checkpoint, run on the reference's positions and dropped, so that any layer fits a 16 GB
/// card: a layer of experts at six bits is 5.9 GB), against the reference of research/glm-data's `ref128` that REF (a directory with `final_ref128.safetensors`, `tokens` and layers/ref128_layerNN.safetensors, whose
/// `h_out` and `h_mid` are the streams after a layer and after its mixer) holds. A layer's input is the reference's output of the layer before it (`teacher`) or this run's own (a chain, from the embedding where `first` is
/// 0, and from the output of the layer before `first` that an earlier run left in OUT). The engine's `LEVEL` bits (6, 5 or 4) hold the experts and every other matrix of the layer. Written to OUT: real_L{l}.mid.bin and .out.bin (bf16 [positions, 4, hidden]: the streams after the mixer and after the
/// layer), real_L{l}.ids.bin (int64 [positions, k]) and .wts.bin (fp32) for a layer of experts, and times.txt.
pub fn real(snap: &Path, refd: &Path, outd: &Path, first: usize, end: usize, level: Layout, teacher: bool) -> Result<()> {
    io(std::fs::create_dir_all(outd))?;
    let cfg = ConfigGlm::load(snap)?;
    let (hid, row) = (cfg.hidden, GM_HC * cfg.hidden);
    let fin = io(glyd_gpu::safetensors::File::open(&refd.join("final_ref128.safetensors")))?;
    let tokens: Vec<usize> = ref_tensor(&fin, "tokens")?.as_chunks::<4>().0.iter().map(|b| i32::from_le_bytes(*b) as usize).collect();
    let rows = tokens.len();
    let words = |b: &[u8]| -> Vec<u16> { b.as_chunks::<2>().0.iter().map(|x| u16::from_le_bytes(*x)).collect() };
    let layer_ref = |l: usize| io(glyd_gpu::safetensors::File::open(&refd.join("layers").join(format!("ref128_layer{l:02}.safetensors"))));
    // the first layer's input: the embedding rows, each on the four streams; or the reference's output of the layer before
    let mut cur: Vec<u16> = if first == 0 {
        let ck = io(Checkpoint::open(snap))?;
        let name = format!("{}embed_tokens.weight", cfg.prefix);
        let (file, info) = ck.get(&name).ok_or_else(|| Error(format!("no tensor {name}")))?;
        let mut v = Vec::with_capacity(rows * row);
        let mut f = io(std::fs::File::open(&file.path))?;
        for &t in &tokens {
            use std::io::{Read, Seek, SeekFrom};
            let mut b = vec![0u8; hid * 2];
            io(f.seek(SeekFrom::Start(info.offset + (t * hid * 2) as u64)))?;
            io(f.read_exact(&mut b))?;
            let w = words(&b);
            for _ in 0..GM_HC {
                v.extend_from_slice(&w);
            }
        }
        v
    } else if !teacher && outd.join(format!("real_L{}.out.bin", first - 1)).exists() {
        // (a chain goes on from where an earlier run of it stopped)
        words(&io(std::fs::read(outd.join(format!("real_L{}.out.bin", first - 1))))?)
    } else {
        let f = layer_ref(first - 1)?;
        words(&ref_tensor(&f, "h_out")?)
    };
    let mut times = String::new();
    for l in first..end {
        if teacher && l > first {
            cur = words(&ref_tensor(&layer_ref(l - 1)?, "h_out")?);
        }
        let t0 = Instant::now();
        let o = OptsGlm { ctx: rows + 16, chunk: rows, graph: false, chunked: true, layout: Some(level), only: Some(vec![l]), ..Default::default() };
        let make = factory(level);
        let make_mla = crate::mla_glm::factory();
        let e = EngineGlm::new(snap, &o, Some(&make), Some(&make_mla))?;
        let loaded = t0.elapsed().as_secs_f64();
        let ly = e.w.layers[l].as_ref().ok_or_else(|| Error(format!("layer {l} is not loaded")))?;
        e.x.write(0, &cur)?;
        e.pos.write(0, &(0..rows as i32).collect::<Vec<_>>())?;
        let t1 = Instant::now();
        // SAFETY: live buffers of the engine, sized for a chunk of the reference's positions; the stream orders every step and the reads synchronize.
        let (mid, out) = unsafe {
            e.mixer_half(ly, rows, e.pos.ptr, 0)?;
            e.dev.sync()?;
            let mut mid = vec![0u16; rows * row];
            e.x.read(0, &mut mid)?;
            e.mlp_half(ly, rows)?;
            e.dev.sync()?;
            let mut out = vec![0u16; rows * row];
            e.x.read(0, &mut out)?;
            (mid, out)
        };
        let ran = t1.elapsed().as_secs_f64();
        save_bytes(outd, &format!("real_L{l}.mid.bin"), &mid)?;
        save_bytes(outd, &format!("real_L{l}.out.bin"), &out)?;
        if ly.moe.is_some() {
            if let Some(sc) = e.moe_scratch.get() {
                let (mut ids, mut wts) = (vec![0i64; rows * cfg.experts.top_k], vec![0f32; rows * cfg.experts.top_k]);
                sc.ids.read(0, &mut ids)?;
                sc.wts.read(0, &mut wts)?;
                save_bytes(outd, &format!("real_L{l}.ids.bin"), &ids)?;
                save_bytes(outd, &format!("real_L{l}.wts.bin"), &wts)?;
            }
        }
        let line = format!(
            "layer {l}: {} and {}: loaded in {loaded:.1} s, run on {rows} positions in {:.1} ms, {:.2} GiB of weights",
            if ly.kind == GlmKind::Kda { "Kimi delta attention" } else { "latent attention" },
            if ly.moe.is_some() { "experts" } else { "a dense MLP" },
            ran * 1e3,
            e.w.bytes as f64 / (1u64 << 30) as f64
        );
        println!("{line}");
        times.push_str(&line);
        times.push('\n');
        cur = out;
        drop(e);
    }
    io(std::fs::write(outd.join("times.txt"), times))?;
    Ok(())
}

// ---- glmfp8 ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// Every FP8 matrix of DIR whose name contains one of the comma-separated pieces of `only` (all where none), read by `Checkpoint::read_bf16` (the bf16 it dequantises to: the weights the packers pack) into OUT/NAME.bf16 (raw bf16 words, [rows, cols]); the
/// names are written to OUT/names.txt. (No GPU.)
pub fn fp8(dir: &Path, outd: &Path, only: Option<&str>) -> Result<()> {
    let ck = io(Checkpoint::open(dir))?;
    io(std::fs::create_dir_all(outd))?;
    let mut names: Vec<String> = ck
        .names()
        .filter(|n| ck.get(n).is_some_and(|(_, t)| t.dtype == "F8_E4M3") && only.is_none_or(|o| o.split(',').any(|p| n.contains(p))))
        .map(String::from)
        .collect();
    names.sort();
    let t0 = Instant::now();
    for n in &names {
        let (_, t) = ck.get(n).ok_or_else(|| Error(format!("no tensor {n}")))?;
        let mut w = vec![0u16; t.shape.iter().product::<u64>() as usize];
        io(ck.read_bf16(n, &mut w))?;
        save_bytes(outd, &format!("{n}.bf16"), &w)?;
    }
    io(std::fs::write(outd.join("names.txt"), names.join("\n")))?;
    println!("{} FP8 matrices of {} read as bf16 in {:.1} s", names.len(), dir.display(), t0.elapsed().as_secs_f64());
    Ok(())
}

// ---- glmdecode ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// OUT/model.safetensors: the tensors of `layers`' mixture-of-experts blocks that the engine holds as packs at `level` (the shared expert's gate, up and down, every expert's), in the checkpoint's names as bf16 (each pack
/// decoded and split back into its parts); OUT/config.json beside it. `layers check --decoded OUT` runs the reference on these. (No GPU.)
pub fn decode(dir: &Path, out: &Path, layers: &[usize], level: Layout) -> Result<()> {
    let cfg = ConfigGlm::load(dir)?;
    let ck = io(Checkpoint::open(dir))?;
    if let Some(&l) = layers.iter().find(|&&l| l >= cfg.layers || !cfg.moe[l]) {
        return err(format!("layer {l}: not a mixture-of-experts layer of this model"));
    }
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get().min(16));
    let layers: Vec<usize> = layers.iter().copied().collect::<std::collections::BTreeSet<_>>().into_iter().collect();
    let mut tensors: Vec<(String, String, Vec<u64>, Vec<u8>)> = Vec::new();
    for &l in &layers {
        let all = cfg.layer_packs(l);
        let packs: Vec<crate::qwen35::Pack35> = all[all.len() - 2..].iter().cloned().chain(cfg.expert_packs(l)).collect();
        crate::weights::ordered(packs.len(), threads, 2 * threads, &|k| unpacked(&ck, None, &packs[k], level), |k, w| {
            let pk = &packs[k];
            let mut at = 0;
            for (name, rows) in &pk.parts {
                let bytes: Vec<u8> = w[at * pk.cols..(at + rows) * pk.cols].iter().flat_map(|x| x.to_le_bytes()).collect();
                tensors.push((name.clone(), "BF16".into(), vec![*rows as u64, pk.cols as u64], bytes));
                at += rows;
            }
            Ok(())
        })?;
    }
    io(std::fs::create_dir_all(out))?;
    io(std::fs::copy(dir.join("config.json"), out.join("config.json")))?;
    let refs: Vec<(&str, &str, Vec<u64>, &[u8])> = tensors.iter().map(|(n, d, s, b)| (n.as_str(), d.as_str(), s.clone(), &b[..])).collect();
    crate::pack::write_safetensors(&out.join("model.safetensors"), &[("format", "pt".to_string())], &refs)?;
    println!(
        "{}: {} tensors of layers {layers:?} decoded from the packs of {level:?} ({:.0} MB)",
        out.display(),
        tensors.len(),
        tensors.iter().map(|t| t.3.len()).sum::<usize>() as f64 / 1e6
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tiny model's mixture-of-experts layer, packed at each level and decoded back (`decode`): every tensor the block holds as a pack is there in the checkpoint's name, as the weights the packer's own decode gives (the
    /// bf16 the FP8 matrices dequantise to, packed, unpacked), and the router and the layers' other tensors are not; layers that are not mixture-of-experts ones are refused.
    #[test]
    fn a_layer_is_decoded_from_the_packs_the_engine_makes() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-glmdecode-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let text = crate::glm::tests::tiny_text();
        let c = ConfigGlm::parse(text, "tiny", None).unwrap();
        crate::glm::tests::checkpoint(&dir, text, &c);
        let fp8 = Checkpoint::open(&dir.join("fp8")).unwrap();
        for level in [Layout::Int6, Layout::Int5, Layout::Int4] {
            let out = dir.join(format!("decoded-{}", level.name()));
            decode(&dir.join("fp8"), &out, &[1], level).unwrap();
            let got = Checkpoint::open(&out).unwrap();
            // the shared expert's three matrices and the four experts' three each
            assert_eq!(got.names().count(), 3 + 4 * 3);
            assert_eq!(std::fs::read(out.join("config.json")).unwrap(), std::fs::read(dir.join("fp8").join("config.json")).unwrap());
            let p = "model.language_model.layers.1.mlp";
            for name in [
                format!("{p}.shared_experts.gate_proj.weight"),
                format!("{p}.shared_experts.up_proj.weight"),
                format!("{p}.experts.2.up_proj.weight"),
                format!("{p}.experts.3.down_proj.weight"),
            ] {
                let (_, t) = got.get(&name).unwrap_or_else(|| panic!("no {name}"));
                assert_eq!(t.dtype, "BF16");
                // (the pack of the matrix on its own, decoded, is the same words)
                let part = name.strip_suffix(".weight").unwrap();
                let pk = c.layer_packs(1).into_iter().chain(c.expert_packs(1)).find(|k| k.parts.iter().any(|x| x.0 == name)).unwrap();
                let whole = unpacked(&fp8, None, &pk, level).unwrap();
                let at: usize = pk.parts.iter().take_while(|x| x.0 != name).map(|x| x.1).sum::<usize>() * pk.cols;
                let want: Vec<u8> = whole[at..at + t.shape.iter().product::<u64>() as usize].iter().flat_map(|x| x.to_le_bytes()).collect();
                assert!(got.read(&name).unwrap() == want, "{part} at {level:?}");
            }
            assert!(got.get(&format!("{p}.gate.weight")).is_none() && got.get("model.language_model.layers.1.self_attn.q_proj.weight").is_none());
        }
        assert!(decode(&dir.join("fp8"), &dir.join("x"), &[0], Layout::Int6).unwrap_err().msg.contains("not a mixture-of-experts layer"));
        assert!(decode(&dir.join("fp8"), &dir.join("y"), &[9], Layout::Int6).unwrap_err().msg.contains("layer 9"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The FP8 matrices of a checkpoint are written as the words `read_bf16` gives (the dequantised bf16), the others not at all, and `only` picks by name.
    #[test]
    fn the_fp8_matrices_are_written_as_bf16() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-glmfp8-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let text = crate::glm::tests::tiny_text();
        let c = ConfigGlm::parse(text, "tiny", None).unwrap();
        let words = crate::glm::tests::checkpoint(&dir, text, &c);
        let out = dir.join("out");
        fp8(&dir.join("fp8"), &out, Some("layers.0.mlp.")).unwrap();
        let names = std::fs::read_to_string(out.join("names.txt")).unwrap();
        let names: Vec<&str> = names.lines().collect();
        assert_eq!(
            names,
            [
                "model.language_model.layers.0.mlp.down_proj.weight",
                "model.language_model.layers.0.mlp.gate_proj.weight",
                "model.language_model.layers.0.mlp.up_proj.weight"
            ]
        );
        for n in &names {
            let got: Vec<u16> = std::fs::read(out.join(format!("{n}.bf16"))).unwrap().as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect();
            assert!(got == words[*n], "{n}");
        }
        // (two pieces: the first layer's MLP and one expert's matrices)
        fp8(&dir.join("fp8"), &dir.join("two"), Some("layers.0.mlp.,layers.1.mlp.experts.2.")).unwrap();
        assert_eq!(std::fs::read_to_string(dir.join("two").join("names.txt")).unwrap().lines().count(), 6);
        fp8(&dir.join("fp8"), &dir.join("all"), None).unwrap();
        assert!(std::fs::read_to_string(dir.join("all").join("names.txt")).unwrap().lines().count() > names.len());
        assert!(!dir.join("all").join("model.language_model.layers.0.self_attn.q_proj.weight.bf16").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The tiny model's level: 6 bits unless the files say 5 or 4; no other.
    #[test]
    fn a_levels_bits_are_six_five_or_four() {
        assert_eq!(level_of(None).unwrap(), Layout::Int6);
        assert_eq!(level_of(Some(6)).unwrap(), Layout::Int6);
        assert_eq!(level_of(Some(5)).unwrap(), Layout::Int5);
        assert_eq!(level_of(Some(4)).unwrap(), Layout::Int4);
        assert!(level_of(Some(3)).unwrap_err().msg.contains("3 bits"));
    }
}

/// `glmrandom OUT`: a random GLM-5.3-Flash of the toy config's shape (tests/fixtures/glm5_next_toy_config.json: eight layers, Kimi delta attention and latent attention, a dense MLP in the first three and
/// experts in the rest; its matrices bf16, no MTP head) in Hub format, for the dynamic engine to be checked against this engine on.
/// `fp8`: the routed experts in FP8 as the release has them (E4M3 with fp32 scales of 128 x 128 blocks, `_scale_inv`; their intermediate size 256, a multiple of the blocks), the rest bf16.
/// `real`: (layers, experts) at the release's widths (GLM-5.3-Flash's config: its heads, latents, indexer, experts' sizes and choices, vocabulary), every MLP a block of experts, the layers' attention
/// kinds its first ones' with a sparse attention layer last; for timing a step on one GPU (the experts in FP8 as released).
pub fn random(out: &Path, fp8: bool, real: Option<(usize, usize)>) -> Result<()> {
    let io = |r: std::io::Result<()>| r.map_err(|e| Error(e.to_string()));
    let fp8 = fp8 || real.is_some();
    let mut v = json::parse(if real.is_some() {
        include_str!("../tests/fixtures/glm5_next_config.json")
    } else {
        include_str!("../tests/fixtures/glm5_next_toy_config.json")
    })
    .map_err(|e| Error(e.to_string()))?;
    if let (Some((layers, experts)), Value::Object(o)) = (real, &mut v) {
        if let Some((_, Value::Object(t))) = o.iter_mut().find(|(k, _)| k == "text_config") {
            let kinds: Vec<Value> = (0..layers).map(|l| Value::from(if l + 1 == layers { "deepseek_sparse_attention" } else { "linear_attention" })).collect();
            for (k, x) in t.iter_mut() {
                match k.as_str() {
                    "num_hidden_layers" => *x = Value::from(layers as u64),
                    "n_routed_experts" => *x = Value::from(experts as u64),
                    "first_k_dense_replace" => *x = Value::from(0u64),
                    "layer_types" => *x = Value::Array(kinds.clone()),
                    "mlp_layer_types" => *x = Value::Array(vec![Value::from("sparse"); layers]),
                    _ => {}
                }
            }
            if let Some((_, Value::Object(la))) = t.iter_mut().find(|(k, _)| k == "linear_attn_config") {
                for (k, x) in la.iter_mut() {
                    match k.as_str() {
                        "kda_layers" => *x = Value::Array((0..layers - 1).map(|l| Value::from(l as u64)).collect()),
                        "full_attn_layers" => *x = Value::Array(vec![Value::from((layers - 1) as u64)]),
                        _ => {}
                    }
                }
            }
            for key in ["indexer_types"] {
                if let Some((_, x)) = t.iter_mut().find(|(k, _)| k == key) {
                    if let Value::Array(a) = x {
                        a.truncate(layers);
                        while a.len() < layers {
                            a.push(a.last().cloned().unwrap_or(Value::Null));
                        }
                    }
                }
            }
        }
    }
    if let Value::Object(o) = &mut v {
        if !fp8 {
            o.retain(|(k, _)| k != "quantization_config");
        }
        if let Some((_, Value::Object(t))) = o.iter_mut().find(|(k, _)| k == "text_config") {
            for (k, x) in t.iter_mut() {
                if k == "num_nextn_predict_layers" {
                    *x = Value::from(0u64);
                }
                if fp8 && real.is_none() && k == "moe_intermediate_size" {
                    *x = Value::from(256u64);
                }
            }
        }
    }
    let text = json::to_python(&v, 2);
    let c = ConfigGlm::parse(&text, "random", None)?;
    let mut r = 0x2026_1018_0000_0002u64;
    let mut uni = |a: f32, b: f32| {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        a + (b - a) * ((r >> 40) as f32 / (1u64 << 24) as f32)
    };
    let bf = |x: f32| {
        let b = x.to_bits();
        ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
    };
    let mut tensors: Vec<(String, &str, Vec<u64>, Vec<u8>)> = Vec::new();
    // (E4M3: a value rounded to it, then its byte)
    let code: std::collections::HashMap<u32, u8> =
        (0..=255u8).filter(|b| !glyd_gpu::fp8::E4M3[*b as usize].is_nan()).map(|b| (glyd_gpu::fp8::E4M3[b as usize].to_bits(), b)).collect();
    for pk in c.packs() {
        let a = (3.0 / pk.cols as f32).sqrt();
        for (name, rows) in &pk.parts {
            let cols = pk.cols;
            if fp8 && (name.contains(".mlp.experts.") || (real.is_some() && name.contains(".mlp.shared_experts."))) {
                let w: Vec<f32> = (0..rows * cols).map(|_| uni(-a, a)).collect();
                let (gr, gc) = (rows.div_ceil(128), cols.div_ceil(128));
                let mut sc = vec![0f32; gr * gc];
                for (i, x) in w.iter().enumerate() {
                    let b = &mut sc[(i / cols) / 128 * gc + (i % cols) / 128];
                    *b = b.max(x.abs() / 448.0);
                }
                let q: Vec<u8> =
                    w.iter().enumerate().map(|(i, x)| code[&crate::dsv4_ref::e4m3_round(x / sc[(i / cols) / 128 * gc + (i % cols) / 128]).to_bits()]).collect();
                tensors.push((name.clone(), "F8_E4M3", vec![*rows as u64, cols as u64], q));
                tensors.push((format!("{name}_scale_inv"), "F32", vec![gr as u64, gc as u64], sc.iter().flat_map(|x| x.to_le_bytes()).collect()));
                continue;
            }
            let w: Vec<u16> = (0..rows * cols).map(|_| bf(uni(-a, a))).collect();
            tensors.push((name.clone(), "BF16", vec![*rows as u64, cols as u64], crate::weights::as_bytes(&w).to_vec()));
        }
    }
    for t in c.kept() {
        let n: u64 = t.shape.iter().product();
        let last = *t.shape.last().unwrap_or(&1) as f32;
        let (lo, hi) = match t.name.as_str() {
            s if s.ends_with("k_norm.bias") => (-0.1, 0.1),
            s if s.ends_with("norm.weight") || s.ends_with("layernorm.weight") => (0.9, 1.1),
            s if s.contains("hc_") && s.ends_with("_fn") => (-0.02, 0.02),
            s if s.contains("hc_") && s.ends_with("_base") => (-0.5, 0.5),
            s if s.contains("hc_") && s.ends_with("_scale") => (0.5, 1.5),
            s if s.ends_with("A_log") => (0.0, 2.7),
            s if s.ends_with("dt_bias") => (-4.6, -2.3),
            s if s.contains("conv1d") => (-0.5, 0.5),
            s if s.ends_with("e_score_correction_bias") => (0.0, 0.1),
            s if s.ends_with("compress_ape") => (-0.5, 0.5),
            _ => {
                let a = (3.0 / last).sqrt();
                (-a, a)
            }
        };
        let bytes = if t.dtype == "F32" {
            (0..n).flat_map(|_| uni(lo, hi).to_le_bytes()).collect()
        } else {
            crate::weights::as_bytes(&(0..n).map(|_| bf(uni(lo, hi))).collect::<Vec<u16>>()).to_vec()
        };
        tensors.push((t.name.clone(), t.dtype, t.shape.clone(), bytes));
    }
    io(std::fs::create_dir_all(out))?;
    io(std::fs::write(out.join("config.json"), &text))?;
    io(std::fs::write(out.join("generation_config.json"), include_str!("../tests/fixtures/glm5_next_generation_config.json")))?;
    let refs: Vec<(&str, &str, Vec<u64>, &[u8])> = tensors.iter().map(|(n, d, s, b)| (n.as_str(), *d, s.clone(), &b[..])).collect();
    crate::pack::write_safetensors(&out.join("model.safetensors"), &[("format", "pt".to_string())], &refs)?;
    println!(
        "{}: a random GLM-5.3-Flash of the toy's shape, {} tensors, {:.0} MB",
        out.display(),
        tensors.len(),
        tensors.iter().map(|t| t.3.len()).sum::<usize>() as f64 / 1e6
    );
    Ok(())
}

/// `glmdyn DIR [--tokens N] [--old]`: GLM on the dynamic engine against this engine on the same tokens (a fixed prompt, then this engine's choices): each step's tokens and the largest logit difference; `--old`
/// runs this engine alone (its tokens and logits).
pub fn dyn_check(dir: &Path, n: usize, old_only: bool, level: Layout, cuts: &[usize], devices: &[i32]) -> Result<()> {
    // (the experts at `level`, the other matrices lossless, both engines alike)
    let (ctx, chunk) = (1024, 64);
    let moe = factory(level);
    let mla = crate::mla_glm::factory();
    let mut e = EngineGlm::new(dir, &OptsGlm { ctx, chunk, graph: true, layout: Some(Layout::Tiered), ..OptsGlm::default() }, Some(&moe), Some(&mla))?;
    let c = ConfigGlm::load(dir)?;
    println!("{}", e.describe());
    let toks: Vec<u32> = (0..n).map(|i| ((i as u32) * 37 + 11) % c.vocab as u32).collect();
    let best = |l: &[f32]| l.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
    let o = crate::dynamic::Opts { ctx, chunk, graph: true, device: 0, layout: Some(Layout::Tiered) };
    let owned = dir.to_path_buf();
    let make: crate::dynamic::split::MakeStage = std::sync::Arc::new(move |st, o: &crate::dynamic::Opts| crate::dynamic::glm::load_stage(&owned, o, level, st));
    let mut d = if old_only { None } else { Some(crate::dynamic::split::Run::new(c.layers, c.eos.clone(), cuts, devices, &o, make)?) };
    if let Some(d) = &d {
        println!("{}", d.describe());
    }
    let (mut worst, mut agree) = (0f32, 0usize);
    let (mut lo_last, mut ld_last) = (Vec::new(), Vec::new());
    for (i, &t) in toks.iter().enumerate() {
        if i == 0 {
            e.prefill(&[t])?;
        } else {
            e.step(t)?;
        }
        let lo = e.logits()?;
        let bo = best(&lo);
        lo_last = lo.clone();
        match d.as_mut() {
            None => println!("{i:4} token {t:5} -> {:5} ({:.4})", bo.0, bo.1),
            Some(d) => {
                if i == 0 {
                    d.prefill(&[t])?;
                } else {
                    d.step(t)?;
                }
                let ld = d.logits()?;
                let bd = best(&ld);
                ld_last = ld.clone();
                let diff = lo.iter().zip(&ld).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
                worst = worst.max(diff);
                agree += (bo.0 == bd.0) as usize;
                if i < 6 || bo.0 != bd.0 || diff > 0.0 {
                    println!("{i:4} token {t:5}: this engine {:5} ({:.4}), dynamic {:5} ({:.4}), largest difference {diff:.4}", bo.0, bo.1, bd.0, bd.1);
                }
            }
        }
    }
    if let Some(d) = d.as_mut() {
        println!("{agree} of {n} steps the same token; the largest logit difference {worst:.4}");
        // the same tokens as one prompt (in chunks): its last logits
        e.reset()?;
        d.reset()?;
        e.prefill(&toks)?;
        d.prefill(&toks)?;
        let (lo, ld) = (e.logits()?, d.logits()?);
        let diff = lo.iter().zip(&ld).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        println!("a prompt of {n} in chunks of {chunk}: this engine {}, dynamic {}, largest logit difference {diff:.4}", best(&lo).0, best(&ld).0);
        // (each engine's prompt against its own last decode step: the same tokens, so the same logits but for rounding)
        let gap = |a: &[f32], b: &[f32]| a.iter().zip(b).fold(0f32, |m, (x, y)| if (x - y).is_nan() { f32::INFINITY } else { m.max((x - y).abs()) });
        println!("each prompt against its decode: this engine {:.4}, dynamic {:.4}", gap(&lo, &lo_last), gap(&ld, &ld_last));
    }
    Ok(())
}
