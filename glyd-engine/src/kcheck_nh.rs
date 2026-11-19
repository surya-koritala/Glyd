//! Nemotron's kernels and layers on the GPU, their outputs written as files for tools/nemotron_ref.py to compare with its PyTorch reference (a development build's):
//!   `glyd-engine nhkernels IN OUT`   the kernels (kernels/nemotron.cu) on random inputs: tools/nemotron_ref.py `kernels gen`'s files, `kernels check` compares (tools/nh_emu.cpp runs the same sequence on the host)
//!   `glyd-engine nhlayers DIR IO`    layers of the real model (tools/nemotron_ref.py `fetch` made DIR: their tensors by range from the Hub) or of the random tiny model `moetoy` made, through the engine's own code, the
//!                                    cases of IO/spec.json
//!   `glyd-engine nhtoy DIR IO`       a model of a few of the real layers (DIR/toy, with part of the embedding and the head), or the random tiny model with mixture-of-experts layers: prefill and decode steps, as a graph
//!                                    and not, the logits of every step; DIR may be a glyd-v7 directory (`glyd-engine pack MODEL DIR --layout i6|i5|i4`), the whole model, with the level of toy.txt's third number
//!   `glyd-engine nhdecode DIR OUT`   (no GPU) the packs of some layers of a glyd-v7 directory decoded to bf16, with their small tensors and the embeddings of some tokens, as a mini checkpoint that tools/nemotron_ref.py's
//!                                    `layers` and `full` take: the reference then runs on the weights the engine multiplies by
//! Files are raw little-endian bf16 (u16) unless noted; *.txt hold numbers.

use crate::dev::{Buf, Dev};
use crate::ffi::{err, Error, Result, Words};
use crate::kcheck::top_vs_row;
use crate::kern::Kern;
use crate::model_nh::{EngineNh, KernNh, OptsNh, SsdWs, NH_N, NH_P};
use crate::moe_nh::factory;
use crate::nemotron::{pack_layout, ConfigNemotronH, NhKind};
use crate::offload::{CachePlan, HostMemory, Mode, Offload};
use crate::qwen35::Pack35;
use crate::sampler::Need;
use crate::weights::{read_pack, v7_manifest, Layout};
use glyd_gpu::json::{self, Value};
use glyd_gpu::pack::{rows_bits, transpose, unpack_int3, unpack_int4, unpack_int5, unpack_int6, unpack_rows, Tiered, INT4_TAG, INT5_TAG, INT6_TAG};
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

/// The level a mixture-of-experts layer's matrices are held at for `bits` (the toy's `moetoy --level`: 6, 5 or 4; none: 6): the experts in that many bits, and the dense matrices too but at 4, where they are six-bit codes.
fn level_of(bits: Option<usize>) -> Result<Layout> {
    match bits {
        None | Some(6) => Ok(Layout::Int6),
        Some(5) => Ok(Layout::Int5),
        Some(4) => Ok(Layout::Int4),
        Some(b) => err(format!("a level of {b} bits: the mixture-of-experts layers are held in 6, 5 or 4")),
    }
}

// ---- nhkernels --------------------------------------------------------------------------------------------------------------------------------------------------------------------

pub fn kernels(ind: &Path, outd: &Path) -> Result<()> {
    io(std::fs::create_dir_all(outd))?;
    let dev = Dev::new(0)?;
    let (kern, k) = (Kern::load(&dev)?, KernNh::load(&dev)?);
    let s = dev.stream;
    let m = nums(ind, "mamba.txt")?;
    let (nh, ng, n, eps, dt_min, split) = (m[0] as usize, m[1] as usize, m[2] as usize, m[3] as f32, m[4] as f32, m[5] as usize);
    let (di, cd) = (nh * NH_P, nh * NH_P + 2 * ng * NH_N);
    let w = di + cd + nh;
    let (mix, cw, cb, alog, dd, dtb, nw) = (load(ind, "mix.bin")?, load(ind, "conv_w.bin")?, load(ind, "conv_b.bin")?, load(ind, "a_log.bin")?, load(ind, "d.bin")?, load(ind, "dt_bias.bin")?, load(ind, "norm_w.bin")?);
    let (state0, cstate0) = (load(ind, "state0.bin")?, load(ind, "cstate0.bin")?);
    let ws = SsdWs::sized(nh, ng, n)?;
    let first = if split > 0 && split < n { split } else { n };
    // (first token, tokens, the scan in chunks) a call
    type Call = (usize, usize, bool);
    let cases: [(&str, Vec<Call>); 4] = [
        ("chunked", [vec![(0, first, true)], if first < n { vec![(first, n - first, true)] } else { vec![] }].concat()),
        ("recurrent", [vec![(0, first, false)], if first < n { vec![(first, n - first, false)] } else { vec![] }].concat()),
        ("steps", (0..n).map(|t| (t, 1, false)).collect()),
        ("mixed", [vec![(0, first, true)], (first..n).map(|t| (t, 1, false)).collect()].concat()),
    ];
    for (name, calls) in &cases {
        let (conv, y, cst, state) = (Buf::zeroed(n * cd * 2)?, Buf::zeroed(n * di * 2)?, copy(&cstate0)?, copy(&state0)?);
        let t0 = Instant::now();
        for &(a, len, chunked) in calls {
            let (x, dtr) = (mix.ptr + ((a * w + di) * 2) as u64, mix.ptr + ((a * w + di + cd) * 2) as u64);
            let (cv, yy) = (conv.ptr + (a * cd * 2) as u64, y.ptr + (a * di * 2) as u64);
            // SAFETY: buffers made here of the sizes the files give; the stream orders the calls and is synchronized before they are read.
            unsafe {
                k.conv(s, x, w as i32, cw.ptr, cb.ptr, cst.ptr, cv, len as i32, cd as i32)?;
                if chunked {
                    k.ssd(s, &ws, cv, cd as i32, dtr, w as i32, dtb.ptr, alog.ptr, dd.ptr, state.ptr, yy, len as i32, nh as i32, ng as i32, dt_min)?;
                } else {
                    k.ssm(s, cv, cd as i32, dtr, w as i32, dtb.ptr, alog.ptr, dd.ptr, state.ptr, yy, len as i32, nh as i32, ng as i32, dt_min)?;
                }
            }
        }
        dev.sync()?;
        println!("mamba case {name}: {} calls, {:.1} ms", calls.len(), t0.elapsed().as_secs_f64() * 1e3);
        let gn = copy(&y)?;
        // SAFETY: as above.
        unsafe { k.gated_norm(s, gn.ptr, mix.ptr, w as i32, nw.ptr, n as u32, di as i32, ng as i32, eps)? };
        dev.sync()?;
        save(outd, &format!("{name}_conv.bin"), &conv, 0, conv.bytes)?;
        save(outd, &format!("{name}_y.bin"), &y, 0, y.bytes)?;
        save(outd, &format!("{name}_gn.bin"), &gn, 0, gn.bytes)?;
        save(outd, &format!("{name}_state.bin"), &state, 0, state.bytes)?;
        save(outd, &format!("{name}_cstate.bin"), &cst, 0, cst.bytes)?;
    }
    // the attention: keys and values into the cache, a decode step a row, then the engine's prompt kernel
    let a = nums(ind, "attn.txt")?;
    let (nq, nk, na) = (a[0] as usize, a[1] as usize, a[2] as usize);
    let qs = (nq + 2 * nk) * 128;
    let qkv = load(ind, "qkv.bin")?;
    let pos = Buf::new(na * 4)?;
    pos.write(0, &(0..na as i32).collect::<Vec<_>>())?;
    let (kv, out, part, pre) = (Buf::zeroed(na * nk * 256 * 2)?, Buf::zeroed(na * nq * 128 * 2)?, Buf::zeroed(nq * crate::model_nh::NH_SPLITS * 130 * 4)?, Buf::zeroed(na * nq * 128 * 2)?);
    // SAFETY: as above.
    unsafe {
        k.kv_write(s, qkv.ptr, qs as i32, pos.ptr, kv.ptr, na as u32, nq as i32, nk as i32)?;
        for t in 0..na {
            k.attn_decode(s, qkv.ptr + (t * qs * 2) as u64, qs as i32, kv.ptr, pos.ptr + (t * 4) as u64, 1, nq as i32, nk as i32, part.ptr, out.ptr + (t * nq * 128 * 2) as u64, (nq * 128) as i32)?;
        }
        kern.prefill_attn(s, qkv.ptr, qs as i32, kv.ptr, pre.ptr, (nq * 128) as i32, na as i32, 0, nq as i32, nk as i32)?;
    }
    dev.sync()?;
    save(outd, "kv.bin", &kv, 0, kv.bytes)?;
    save(outd, "dec.bin", &out, 0, out.bytes)?;
    save(outd, "pre.bin", &pre, 0, pre.bytes)?;
    println!("attention done");
    Ok(())
}

// ---- nhlayers ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// The cases of IO/spec.json, {"chunk": N, "level": 4 | 5 | 6, "cases": [{"name", "layer", "mode": "prefill" | "decode" | "mixed", "calls": [tokens a call] (prefill), "tokens": N (decode, mixed), "prefill": N (mixed),
/// "graph": whether single tokens are replayed from a CUDA graph, "recurrent": the scan token by token}]}: each case a layer's mixer on the first rows of DIR/x.bin (the residual input, bf16 [rows, hidden]),
/// the layer's own norm first; "level": the bits of the level a mixture-of-experts layer's matrices are held at (6, 5 or 4; default 6). Written per case: stage outputs of every row (`mix`: the in_proj's output, `conv`,
/// `gn`: the gated norm's, `out` for a Mamba-2 layer; `qkv`, `attn`, `out` for an attention layer; `h`: the layer's normed input, `z`: the first latent projection's, `y`: the experts' sum, `out` for a
/// mixture-of-experts layer, with the experts the router chose, `ids` int64 [rows, k], and their weights, `wts` fp32), and at the end the state (`ssm` fp32, `cstate` bf16; the cache's `kv`).
pub fn layers(dir: &Path, outd: &Path) -> Result<()> {
    let spec = json::parse(&io(std::fs::read_to_string(outd.join("spec.json")))?).map_err(|e| Error(format!("spec.json: {e}")))?;
    let cases = spec.get("cases").and_then(Value::as_array).ok_or_else(|| Error("spec.json: no cases".into()))?;
    let get = |c: &Value, k: &str| c.get(k).and_then(Value::as_u64).map(|x| x as usize);
    let mut wanted: Vec<usize> = cases.iter().filter_map(|c| get(c, "layer")).collect();
    wanted.sort_unstable();
    wanted.dedup();
    let xb = io(std::fs::read(dir.join("x.bin")))?;
    let hid = ConfigNemotronH::load(dir)?.hidden;
    let total = xb.len() / 2 / hid;
    let o = OptsNh { ctx: total + 64, chunk: get(&spec, "chunk").unwrap_or(512), graph: false, only: Some(wanted), ..Default::default() };
    let make = factory(level_of(get(&spec, "level"))?);
    let mut e = EngineNh::new(dir, &o, Some(&make))?;
    let c = e.w.cfg.clone();
    println!("{}", e.describe());
    let x_words: Vec<u16> = xb.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect();
    let s = e.dev.stream;
    for case in cases {
        let name = case.get("name").and_then(Value::as_str).ok_or_else(|| Error("a case has no name".into()))?;
        let l = get(case, "layer").ok_or_else(|| Error(format!("{name}: no layer")))?;
        let mode = case.get("mode").and_then(Value::as_str).unwrap_or("prefill");
        let graph = case.get("graph").and_then(Value::as_bool).unwrap_or(false);
        let kind = e.w.layers[l].as_ref().map(|ly| ly.kind).ok_or_else(|| Error(format!("{name}: layer {l} is not loaded")))?;
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
            "prefill" => prefill_calls(case.get("calls").and_then(Value::as_u64s).unwrap_or_default().into_iter().map(|v| v as usize).collect(), &mut at, &mut calls),
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
        e.chunked = !case.get("recurrent").and_then(Value::as_bool).unwrap_or(false);
        let eps = c.eps;
        let (qkv_w, att_w) = ((c.heads + 2 * c.kv_heads) * c.head_dim, c.heads * c.head_dim);
        // the stages dumped: (name, width of a row), and the engine's buffer each is read from
        let live: Vec<(&str, usize, &crate::dev::Buf)> = match kind {
            NhKind::Mamba => vec![("mix", c.in_rows(), &e.mix), ("conv", c.conv_dim(), &e.xc), ("gn", c.inner(), &e.core), ("out", hid, &e.o)],
            NhKind::Attention => vec![("qkv", qkv_w, &e.mix), ("attn", att_w, &e.att), ("out", hid, &e.o)],
            NhKind::Moe => vec![("h", hid, &e.h), ("z", c.moe.latent, &e.mix), ("y", c.moe.latent, &e.xc), ("out", hid, &e.o)],
        };
        let mut got: Vec<Vec<u16>> = vec![Vec::new(); live.len()];
        // (a mixture-of-experts layer's experts and their weights, a call after another)
        let mut chosen: (Vec<i64>, Vec<f32>) = (Vec::new(), Vec::new());
        let mut graph_exec: Option<crate::dev::Graph> = None;
        let t0 = Instant::now();
        let mut spent = 0.0;
        for &(a, n) in &calls {
            e.x.write(0, &x_words[a * hid..(a + n) * hid])?;
            let ly = e.w.layers[l].as_ref().expect("a loaded layer");
            let t1 = Instant::now();
            if n == 1 && graph {
                if graph_exec.is_none() {
                    // (the scratch of the mixture-of-experts layers is made by the first call of one: before the capture and not in it, as the engine's warm-up step does)
                    if kind == NhKind::Moe && e.moe_scratch.get().is_none() {
                        // SAFETY: as below; the layer has no state, and the call is waited for.
                        unsafe {
                            e.kern.rmsnorm(s, e.x.ptr, ly.norm.ptr, e.h.ptr, 1, hid as i32, eps)?;
                            e.mixer(l, 1, e.st.ptr, 0)?;
                        }
                        e.dev.sync()?;
                    }
                    // SAFETY: live buffers of the engine, sized for a chunk; nothing runs during a capture.
                    let g = e.dev.capture(|| unsafe {
                        e.kern.rmsnorm(s, e.x.ptr, ly.norm.ptr, e.h.ptr, 1, hid as i32, eps)?;
                        e.mixer(l, 1, e.st.ptr, 0)
                    })?;
                    graph_exec = Some(g);
                }
                e.st.write(0, &[a as i32])?;
                graph_exec.as_ref().expect("a graph").launch(s)?;
            } else {
                let pos: Vec<i32> = (a..a + n).map(|p| p as i32).collect();
                e.pos.write(0, &pos)?;
                // SAFETY: as above; the stream orders every step.
                unsafe {
                    e.kern.rmsnorm(s, e.x.ptr, ly.norm.ptr, e.h.ptr, n as u32, hid as i32, eps)?;
                    e.mixer(l, n, e.pos.ptr, a)?;
                }
            }
            e.dev.sync()?;
            spent += t1.elapsed().as_secs_f64();
            for (i, (_, width, buf)) in live.iter().enumerate() {
                let mut v = vec![0u16; n * width];
                buf.read(0, &mut v)?;
                got[i].extend(v);
            }
            if kind == NhKind::Moe {
                let sc = e.moe_scratch.get().ok_or_else(|| Error(format!("{name}: the mixture-of-experts layer made no scratch")))?;
                let (mut ids, mut wts) = (vec![0i64; n * c.moe.top_k], vec![0f32; n * c.moe.top_k]);
                sc.ids.read(0, &mut ids)?;
                sc.wts.read(0, &mut wts)?;
                chosen.0.extend(ids);
                chosen.1.extend(wts);
            }
        }
        for (i, (nm, _, _)) in live.iter().enumerate() {
            save_bytes(outd, &format!("{name}.{nm}.bin"), &got[i])?;
        }
        let idx = e.w.layers[l].as_ref().expect("a loaded layer").idx;
        match kind {
            NhKind::Mamba => {
                save(outd, &format!("{name}.ssm.bin"), &e.ssm, idx * c.m_heads * NH_P * NH_N * 4, c.m_heads * NH_P * NH_N * 4)?;
                save(outd, &format!("{name}.cstate.bin"), &e.convst, idx * 3 * c.conv_dim() * 2, 3 * c.conv_dim() * 2)?;
            }
            NhKind::Attention => save(outd, &format!("{name}.kv.bin"), &e.kv, idx * e.ctx() * c.kv_heads * 256 * 2, at * c.kv_heads * 256 * 2)?,
            NhKind::Moe => {
                save_bytes(outd, &format!("{name}.ids.bin"), &chosen.0)?;
                save_bytes(outd, &format!("{name}.wts.bin"), &chosen.1)?;
            }
        }
        println!(
            "case {name}: layer {l} ({kind:?}), {} calls, {at} rows: {:.1} ms in the layer ({:.3} ms a call), {:.1} ms with the dumps",
            calls.len(),
            spent * 1e3,
            spent * 1e3 / calls.len().max(1) as f64,
            t0.elapsed().as_secs_f64() * 1e3
        );
    }
    Ok(())
}

// ---- nhtoy -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// IO/toy.txt: the tokens to prefill, the tokens in all and, where the model has mixture-of-experts layers, the bits of the level their matrices are held at (6, 5 or 4; default 6; a glyd-v7 directory's own); IO/ids.txt: the token ids.
/// The model of DIR (a few real layers, a part of the embedding and the head; or the random tiny model with mixture-of-experts layers) is run twice, with the decode step as a
/// CUDA graph and without: the logits after the prefill and after each step (the next input being the file's token) to IO/toy_graph_logits.bin and toy_eager_logits.bin (bf16 [steps, vocab]), the chosen tokens to *.txt.
pub fn toy(dir: &Path, outd: &Path, cache: Option<(&str, Offload)>) -> Result<()> {
    let t = nums(outd, "toy.txt")?;
    let (p, total) = (t[0] as usize, t[1] as usize);
    // (a glyd-v7 directory is at its own level, which toy.txt's third number need not say)
    let level = match v7_manifest(dir)? {
        Some((l, _)) => l,
        None => level_of(t.get(2).map(|&b| b as usize))?,
    };
    let ids: Vec<u32> = nums(outd, "ids.txt")?.into_iter().map(|x| x as u32).collect();
    if p == 0 || p > total || total > ids.len() {
        return err(format!("toy.txt: {p} of {total} tokens, ids.txt has {}", ids.len()));
    }
    // with an expert cache (its `--expert-cache` and how it serves a miss) the decode step is not a graph: the one eager run is the file's
    let places = match &cache {
        Some((spec, _)) => cache_for(Some(spec), dir, level, &ConfigNemotronH::load(dir)?)?,
        None => None,
    };
    let mut offload = [cache.map_or(Offload::default(), |c| c.1)];
    host_setup(places.as_ref(), &mut offload)?;
    for (graph, tag) in [(true, "graph"), (false, "eager")].into_iter().filter(|(graph, _)| *graph != places.is_some()) {
        let o = OptsNh { ctx: total + 16, chunk: 128, graph, ..Default::default() };
        let store = crate::offload::ExpertStore::new();
        store.set_offload(offload[0]);
        let make: Box<dyn Fn(usize, &ConfigNemotronH, &Checkpoint, Layout) -> Result<Box<dyn crate::model_nh::MoeBlock>>> = match places {
            Some(plan) => Box::new(crate::moe_nh::factory_cached(level, plan.slots, &store)),
            None => Box::new(factory(level)),
        };
        let mut e = EngineNh::new(dir, &o, Some(&*make))?;
        if graph != e.has_graph() {
            return err(format!("the decode graph was asked for: {graph}, the engine has one: {}", e.has_graph()));
        }
        println!("{tag}: {}", e.describe());
        e.reset()?;
        let (mut logits, mut tokens) = (Vec::<u16>::new(), Vec::<String>::new());
        // the sampler's candidates for each step, as `Gen::reply` asks for them with a top-p alone (the 64 largest logits and the sum of the exponentials, at temperature 0.7), against the row the step left
        e.top.ask = Some(Need { temperature: 0.7, mass: true });
        let mut tops = 0usize;
        let mut grab = |e: &EngineNh, logits: &mut Vec<u16>| -> Result<()> {
            let row: Vec<u16> = e.logits()?.into_iter().map(|f| (f.to_bits() >> 16) as u16).collect();
            if let Some(top) = e.top.read() {
                top_vs_row(&top, &row, 0.7).map_err(|m| Error(format!("{tag}: row {}: the sampler's candidates are not the logits' largest: {m}", logits.len() / row.len())))?;
                tops += 1;
            }
            logits.extend(row);
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
        println!("{tag}: prefill of {p} tokens {:.1} ms, {} steps {:.2} ms each", pre * 1e3, total - p, t1.elapsed().as_secs_f64() * 1e3 / (total - p).max(1) as f64);
        if places.is_some() {
            println!("{tag}: {}", store.report()?);
        }
        if tops == 0 {
            println!("{tag}: candidates: none made (the vocabulary of {} is not a multiple of 8 of at least 64)", e.w.cfg.vocab);
        } else {
            println!("{tag}: candidates: the 64 largest logits and the sum of their exponentials, as the sampler is given them, are the row's in all {tops} rows");
        }
        save_bytes(outd, &format!("toy_{tag}_logits.bin"), &logits)?;
        io(std::fs::write(outd.join(format!("toy_{tag}_tokens.txt")), tokens.join(" ")))?;
    }
    Ok(())
}

// ---- the expert cache's settings -------------------------------------------------------------------------------------------------------------------------------------------------

/// The expert cache an `--expert-cache` asks for (None: no cache): `auto`, what `run` would give this GPU on the memory it has free; `auto:GIB`, what a GPU with that much free would get (a card of 24 GB has about 23
/// GiB free, one of 32 GB about 31, one of 16 GB about 15); a number, that many GiB of the GPU's for the places. The error is the plan's own where the GPU cannot hold the rest of the model and the least cache.
pub fn cache_for(spec: Option<&str>, dir: &Path, level: Layout, cfg: &ConfigNemotronH) -> Result<Option<CachePlan>> {
    match spec {
        None => Ok(None),
        Some(g) if g == "auto" || g.starts_with("auto:") => {
            let (free, gpu) = match g.strip_prefix("auto:") {
                Some(n) => ((n.parse::<f64>().ok().filter(|v| v.is_finite() && *v > 0.0).ok_or_else(|| Error(format!("--expert-cache {g}: auto or auto:GIB")))? * (1u64 << 30) as f64) as u64, "the GPU".to_string()),
                None => {
                    let p = Dev::probe(0)?;
                    (p.free, p.name)
                }
            };
            crate::cli::expert_cache_for(dir, level, free, &gpu, crate::offload::host_memory()).map_err(|r| Error(r.to_string()))
        }
        Some(g) => {
            let gib: f64 = g.parse().ok().filter(|v: &f64| v.is_finite() && *v > 0.0).ok_or_else(|| Error(format!("--expert-cache {g}: not a number of GiB")))?;
            let plan = crate::moe_nh::cache_plan(cfg, level, (gib * (1u64 << 30) as f64) as u64, cfg.moe.top_k)?;
            Ok(Some(plan.ok_or_else(|| Error(format!("--expert-cache {g}: fewer than {} experts a layer fit", cfg.moe.top_k)))?))
        }
    }
}

/// `--host-memory` (src/cli.rs) applied to the plan of `generate`: the bytes of the experts the computer's memory holds (`plan.ram`), what was asked for or else what the computer has free less the reserve; only the Hummingbird
/// level keeps experts on the disk, and the others are refused less than all of them.
pub fn host_budget(a: &crate::args::Args, plan: &mut CachePlan) -> Result<()> {
    let asked = crate::cli::host_flag(a).map_err(|r| Error(r.to_string()))?;
    if plan.in_place {
        plan.ram = crate::cli::held_in_memory(plan.host_bytes(), asked, crate::offload::host_memory());
    } else if asked.is_some_and(|b| b < plan.host_bytes()) {
        return err("--host-memory: only the Hummingbird level keeps experts on the disk");
    }
    Ok(())
}

/// The cache said in a line on stderr (the places, the memory of each side, the computer's), and the offload variants set to what the computer's memory holds: the host's half (a second copy of the experts' first matrices,
/// half the experts' bytes) besides the experts, or not, where the mode `auto` copies and `cpu` and `split` are refused.
pub fn host_setup(plan: Option<&CachePlan>, variants: &mut [Offload]) -> Result<()> {
    let Some(p) = plan else { return Ok(()) };
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let host = crate::offload::host_memory();
    let said = host.map_or("none said".to_string(), |h| format!("{:.1} GiB", gib(h)));
    if p.ram < p.host_bytes() {
        eprintln!(
            "expert cache plan: {} of {} experts a layer on the GPU ({:.2} GiB of its memory), {:.1} GiB of experts in the computer's memory, {:.1} GiB read from the disk as needed ({said} to give)",
            p.slots,
            p.experts,
            gib(p.gpu_bytes()),
            gib(p.ram),
            gib(p.disk_bytes())
        );
    } else {
        eprintln!(
            "expert cache plan: {} of {} experts a layer on the GPU ({:.2} GiB of its memory), all {} in the computer's memory ({:.1} GiB pinned, and {:.1} GiB more for the host's half; {said} to give)",
            p.slots,
            p.experts,
            gib(p.gpu_bytes()),
            p.layers * p.experts,
            gib(p.host_bytes()),
            gib(p.host_half_bytes())
        );
    }
    if !p.host_half_fits(host, 6 << 30) {
        for v in variants.iter_mut() {
            match v.mode {
                Mode::Auto => {
                    eprintln!("the computer's memory holds the experts but not the host's half besides: auto copies");
                    v.mode = Mode::Copy;
                }
                Mode::Cpu | Mode::Split => return err(format!("--offload-mode {}: the host's half takes {:.1} GiB besides the experts' {:.1}, and this computer has {said} to give", v.mode.name(), gib(p.host_half_bytes()), gib(p.host_bytes()))),
                Mode::Copy => {}
            }
        }
    }
    Ok(())
}

/// How far a run with some experts computed on the host may be from the run with every expert on the GPU, as a share of the largest logit: two places of bf16 (a place is 2^-8 to 2^-7 of a value). The routed sum
/// the host takes part in differs from the GPU's by the order of an fp32 sum, seen through one rounding to bf16 (on the tiny model's grid weights, where the host's weights are the GPU's, `a_layer_with_misses_on_the_host_is_the_layer_all_there_to_a_bound`
/// finds the layer's output the same to the bit or within 6e-5 of its norm); the weights it does not round to bf16 are 1.5e-3 to 3.7e-3 of a pair's norm on weights off the grid.
pub const LOGITS_BOUND: f32 = 1.0 / 64.0;

/// Two runs' logits (`positions` rows of bf16 words, `got` the run to be held, `want` the run with every expert on the GPU): Err with the first thing wrong, else a line of what they are: every logit of `got` finite and within
/// `bound` of the largest logit of its row in `want`, and the token `got` picks (the largest logit) one that `want` has within `bound` of its own best.
pub fn logits_within(got: &[u16], want: &[u16], positions: usize, bound: f32) -> std::result::Result<String, String> {
    if positions == 0 || got.len() != want.len() || want.len() % positions != 0 {
        return Err(format!("{} logits against {}, for {positions} positions", got.len(), want.len()));
    }
    let (vocab, f) = (want.len() / positions, |w: u16| f32::from_bits((w as u32) << 16));
    let (mut worst, mut same, mut top1) = (0f32, 0usize, 0usize);
    for p in 0..positions {
        let (g, w) = (&got[p * vocab..(p + 1) * vocab], &want[p * vocab..(p + 1) * vocab]);
        if g.iter().any(|&x| !f(x).is_finite()) {
            return Err(format!("position {p}: a logit that is not a number"));
        }
        let top = w.iter().map(|&x| f(x).abs()).fold(0f32, f32::max).max(f32::MIN_POSITIVE);
        let dev = g.iter().zip(w).map(|(&a, &b)| (f(a) - f(b)).abs()).fold(0f32, f32::max) / top;
        if dev > bound {
            return Err(format!("position {p}: a logit {dev:.3e} of the largest away from the run with every expert on the GPU's (the bound is {bound:.3e})"));
        }
        let arg = |v: &[u16]| v.iter().enumerate().fold(0, |m, (i, &x)| if f(x) > f(v[m]) { i } else { m });
        let (a, b) = (arg(g), arg(w));
        let margin = (f(w[b]) - f(w[a])) / top;
        if margin > bound {
            return Err(format!("position {p}: the token picked ({a}) is {margin:.3e} of the largest logit below the best ({b}), past the bound {bound:.3e}"));
        }
        (worst, same, top1) = (worst.max(dev), same + (g == w) as usize, top1 + (a == b) as usize);
    }
    Ok(format!("within {worst:.2e} of the largest at most (bit for bit at {same}), the token picked the same at {top1}"))
}

// ---- nhoffload ---------------------------------------------------------------------------------------------------------------------------------------------------------------

/// IO/toy.txt and IO/ids.txt as `toy` reads them. The model of DIR (the random tiny model with mixture-of-experts layers, `nemotron_ref.py moetoy`'s toy/ directory) is run all there (eager, then as a graph) and then
/// through the expert cache with each of `slots` places a layer (src/offload.rs; fewer than the layer's choices makes a call of every layer more than one wave, and fewer than its experts makes a prompt's chunk
/// several), on two runs of it: the file's prompt in chunks of 128 tokens and then its steps (the products a few tokens and a chunk take), and a prompt of 400 tokens in one chunk and 8 steps (1,600 pairs of 16 experts:
/// enough for the library's tiled product, which takes 48 pairs an expert on most GPUs, 64 on an A100 and 96 on an H100). The logits after the prefill and after each step, bit for bit against the run with all of the experts there, on the first pass over the tokens, again on the warm cache, and
/// with hints that churn the places between the steps. A line a pass, `ok` or `FAIL`, then the cache's counts. False if anything differs, or if a cache of fewer places than experts never copied one in.
/// The experts' second matrices are in the layout the loader packs them in (`pack_layout`: the row layout, which runs the sparse product, and a place holds an expert's first matrix and its rows; with
/// `GLYD_ENGINE_DENSE_DOWN=1` the block layout, the library's dense product): the first line says which, and every comparison is between runs of that one layout. DIR may be a Hummingbird directory (`glyd-engine hbtoy`):
/// the places a layer has are then counted at the mean of its experts' widths and split between the two groups of experts (src/moe_hb.rs), each a layer of the cache, and the hints go to those.
///
/// `memory`: budgets of the computer's memory for the experts, in bytes (a Hummingbird directory's SSD tier, src/offload.rs `Tier`). For each, and each count of places below the layer's experts, the cache runs the passes
/// above twice, with every expert in the computer's memory and with no more than the budget of them (the rest read from the packed directory as they are asked for), with the planner's times held at their starting values
/// (`ExpertStore::hold_costs`: two runs of a mode plan alike whatever their timings), in the mode of `host`: every logit and token of the second the first's to the bit, whatever serves the misses (the memory changes where an expert's
/// bytes are and never what is computed), and experts read from the directory.
///
/// `host`: how the cache serves a miss (`Offload`; None, or the mode `Copy`: copies, and the comparison is bit for bit). With the host computing the pairs of experts that are not in places (`cpu`, `split`, `auto`) the
/// logits are not the GPU's to the bit (the order of an fp32 sum, one rounding of the routed sum that the GPU makes alone; the tiny model's weights are on the block layouts' grids, so that the host's weights are the GPU's)
/// and a run is held to `logits_within` the run with every expert on the GPU: every logit within two places of bf16 (2^-6) of the largest, the token it picks within that of the best. The host must have computed
/// pairs in a cache of fewer places than experts, and in `cpu` have started copies behind them (`promoted`).
pub fn offload(dir: &Path, outd: &Path, slots: &[usize], host: Option<Offload>, memory: &[u64]) -> Result<bool> {
    use crate::moe_nh::factory_cached;
    use crate::offload::ExpertStore;
    let tolerant = host.is_some_and(|o| o.mode != Mode::Copy);
    let t = nums(outd, "toy.txt")?;
    let (p, total) = (t[0] as usize, t[1] as usize);
    // (a Hummingbird directory is its own level, whatever toy.txt says: the experts are codebook rows of 3 and 2 bits, a layer's two widths two layers of the expert cache)
    let manifest = v7_manifest(dir)?;
    let hummingbird = matches!(manifest, Some((Layout::Hummingbird, _)));
    if !memory.is_empty() && !hummingbird {
        return err("--host-memory: only a Hummingbird directory keeps experts on the disk");
    }
    // (the bytes of all the directory's experts, which a budget is a share of)
    let all_experts = match &manifest {
        Some((Layout::Hummingbird, m)) if !memory.is_empty() => crate::moe_hb::experts_bytes(m, &ConfigNemotronH::load(dir)?)?,
        _ => 0,
    };
    let level = if hummingbird { Layout::Hummingbird } else { level_of(t.get(2).map(|&b| b as usize))? };
    let down = pack_layout(level, "model.layers.0.mixer.experts.0.down_proj");
    if hummingbird {
        println!("the experts: codebook rows (c3r, c2r) in the rotated basis of the layer, the sparse second product");
    } else {
        println!("the experts' second matrices: {} ({})", down.name(), if down.row_bits().is_some() { "the row layout: the sparse product" } else { "the block layout: the library's dense product" });
    }
    let ids: Vec<u32> = nums(outd, "ids.txt")?.into_iter().map(|x| x as u32).collect();
    if p == 0 || p > total || total > ids.len() {
        return err(format!("toy.txt: {p} of {total} tokens, ids.txt has {}", ids.len()));
    }
    // (what a run is, the chunk of its engine, the prompt, the tokens it steps through)
    let runs: Vec<(String, usize, Vec<u32>, Vec<u32>)> = vec![
        (format!("{p} tokens in chunks of 128, then {} steps", total - p), 128, ids[..p].to_vec(), ids[p..total].to_vec()),
        ("400 tokens in one chunk, then 8 steps".to_string(), 512, ids.iter().cycle().take(400).copied().collect(), ids[p..total].iter().take(8).copied().collect()),
    ];
    // the prefill and the steps: the logits after each (bf16 words) and the tokens chosen; `hints` (the store, the layers it holds, the experts of a layer) put experts in the places between steps
    let run = |e: &mut dyn Pass, prompt: &[u32], steps: &[u32], hints: Option<(&ExpertStore, usize, usize)>| -> Result<(Vec<u16>, Vec<u32>)> {
        e.reset()?;
        let (mut logits, mut tokens) = (Vec::<u16>::new(), Vec::<u32>::new());
        let grab = |e: &dyn Pass, logits: &mut Vec<u16>| -> Result<()> {
            logits.extend(e.logits()?.into_iter().map(|f| (f.to_bits() >> 16) as u16));
            Ok(())
        };
        tokens.push(e.prefill(prompt)?.token);
        grab(e, &mut logits)?;
        let mut r = 0x9E37_79B9_7F4A_7C15u64;
        for &id in steps {
            if let Some((store, layers, experts)) = hints {
                for l in 0..layers {
                    let want: Vec<u16> = (0..experts / 2)
                        .map(|_| {
                            r ^= r << 13;
                            r ^= r >> 7;
                            r ^= r << 17;
                            (r % experts as u64) as u16
                        })
                        .collect();
                    store.hint(l, &want)?;
                }
            }
            tokens.push(e.step(id)?.token);
            grab(e, &mut logits)?;
        }
        Ok((logits, tokens))
    };
    let verdict = |what: &str, got: &(Vec<u16>, Vec<u32>), want: &(Vec<u16>, Vec<u32>)| -> bool {
        if tolerant {
            return match logits_within(&got.0, &want.0, want.1.len(), LOGITS_BOUND) {
                Ok(m) => {
                    println!("ok   {what}: the logits of {} positions are {m}", want.1.len());
                    true
                }
                Err(m) => {
                    println!("FAIL {what}: {m}");
                    false
                }
            };
        }
        let same = got == want;
        if same {
            println!("ok   {what}: {} logits and {} tokens as the run with every expert there", got.0.len(), got.1.len());
        } else {
            let at = got.0.iter().zip(&want.0).position(|(a, b)| a != b);
            let worst = got.0.iter().zip(&want.0).map(|(a, b)| (f32::from_bits((*a as u32) << 16) - f32::from_bits((*b as u32) << 16)).abs()).fold(0f32, f32::max);
            println!("FAIL {what}: the logits differ from the first at {at:?} (of {} per step), by {worst} at most; tokens {}", want.0.len() / want.1.len().max(1), if got.1 == want.1 { "the same" } else { "differ" });
        }
        same
    };
    let mut ok = true;
    let mut host_used = false;
    let (cfg, whole) = (ConfigNemotronH::load(dir)?, |c: &ConfigNemotronH| crate::dynamic::Stage::whole(c.layers));
    for (what, chunk, prompt, steps) in &runs {
        println!("== {what}");
        let o = |graph| OptsNh { ctx: prompt.len() + steps.len() + 16, chunk: *chunk, graph, ..Default::default() };
        let dyo = crate::dynamic::Opts { ctx: prompt.len() + steps.len() + 16, chunk: *chunk, graph: true, device: 0, layout: None };
        let make = factory(level);
        let mut e = EngineNh::new(dir, &o(false), Some(&make))?;
        println!("all there: {}", e.describe());
        let reference = run(&mut e, prompt, steps, None)?;
        drop(e);
        let mut g = EngineNh::new(dir, &o(true), Some(&make))?;
        if !g.has_graph() {
            println!("FAIL all there: no decode graph was captured");
            ok = false;
        }
        ok &= verdict("all there, as a graph", &run(&mut g, prompt, steps, None)?, &reference);
        drop(g);
        // the dynamic engine (GLY-354), all there and over each cache below: the hand-written engine's logits
        let mut d = crate::dynamic::nemotron::load_with(dir, &dyo, &make, whole(&cfg))?;
        ok &= verdict("dynamic engine, all there", &run(&mut d, prompt, steps, None)?, &reference);
        drop(d);
        for &n in slots {
            let store = ExpertStore::new();
            if let Some(o) = host {
                store.set_offload(o);
            }
            let make = factory_cached(level, n, &store);
            let mut e = EngineNh::new(dir, &o(true), Some(&make))?;
            let (experts, layers) = (e.w.cfg.moe.experts, e.w.cfg.layers_of(NhKind::Moe) * if hummingbird { 2 } else { 1 });
            println!("{n} places a layer of {experts}: {}", e.describe());
            if n < experts && e.has_graph() {
                println!("FAIL {n} places: the decode step was captured in a graph with blocks that read the router's choices on the host");
                ok = false;
            }
            let hinted = n < experts;
            for (pass, hints) in [("first pass", false), ("second pass, the cache warm", false), ("with hints between the steps", hinted)] {
                let got = run(&mut e, prompt, steps, hints.then_some((&*store, layers, experts)))?;
                ok &= verdict(&format!("{n} places, {pass}"), &got, &reference);
            }
            if hinted {
                // a predictor that expects a layer's experts to be the ones the layer before it chose, and one layer ahead of that: places taken and copies queued while a call is under way
                store.set_predictor(move |l, chosen| vec![((l + 1) % layers, chosen.to_vec()), ((l + 2) % layers, chosen.iter().rev().copied().collect())]);
                let got = run(&mut e, prompt, steps, None)?;
                ok &= verdict(&format!("{n} places, with a predictor hinting the layers ahead"), &got, &reference);
            }
            if n < experts {
                let st = store.stats()?;
                println!("{}", store.report()?);
                if st.misses == 0 || st.hinted == 0 {
                    println!("FAIL {n} places: no expert was copied in ({} misses, {} for hints): the cache was not used", st.misses, st.hinted);
                    ok = false;
                }
                // (the layers have a host half where their second matrices are in the row layout, and at the Hummingbird level, whose codebook rows the host reads in the host tier; `cpu` keeps every miss, so
                // each count of places uses it; `split` and `auto` may copy every miss where copies are cheap, as the tiny model's are at 8 places of 16, so they must use it at one count at least: below)
                host_used |= st.cpu > 0;
                if tolerant && (hummingbird || down.row_bits().is_some()) && host.is_some_and(|o| o.mode == Mode::Cpu) {
                    if st.cpu == 0 || st.promoted == 0 {
                        println!("FAIL {n} places: the host computed {} experts ({} pairs) and {} were copied in behind them: the host's half was not used", st.cpu, st.cpu_pairs, st.promoted);
                        ok = false;
                    }
                }
            }
            if n < experts {
                for &budget in memory {
                    ok &= on_disk(dir, &o(true), level, n, (prompt, steps), (host.unwrap_or(Offload { mode: Mode::Copy, ..Offload::default() }), budget, all_experts), &run)?;
                }
            }
            drop(e);
            let store = ExpertStore::new();
            if let Some(o) = host {
                store.set_offload(o);
            }
            let make = factory_cached(level, n, &store);
            let mut d = crate::dynamic::nemotron::load_with(dir, &dyo, &make, whole(&cfg))?;
            if (n < experts) == d.has_graph() {
                println!("FAIL dynamic engine, {n} places: a decode graph {} (it is one where every expert is on the GPU alone)", if d.has_graph() { "was captured" } else { "was not captured" });
                ok = false;
            }
            for pass in ["first pass", "second pass, the cache warm"] {
                ok &= verdict(&format!("dynamic engine, {n} places, {pass}"), &run(&mut d, prompt, steps, None)?, &reference);
            }
        }
    }
    if tolerant && (hummingbird || down.row_bits().is_some()) && !host_used {
        println!("FAIL the host computed no expert at any count of places: the host's half was not used");
        ok = false;
    }
    println!("{}", if ok { "nhoffload: ok" } else { "nhoffload: FAILED" });
    Ok(ok)
}

/// The tiny model's passes (what `offload`'s `run` does with a prompt and steps).
type Run<'a> = &'a dyn Fn(&mut dyn Pass, &[u32], &[u32], Option<(&crate::offload::ExpertStore, usize, usize)>) -> Result<(Vec<u16>, Vec<u32>)>;

/// The cache of `n` places run through its passes twice (see `offload`'s `memory`): with every expert in the computer's memory and with `budget` bytes of the directory's `all` bytes of experts, in `mode` with the planner's times
/// held; true if every pass of the second is the first's to the bit and it read experts from the packed directory.
/// What a pass reads of an engine: the hand-written one or the dynamic one.
trait Pass {
    fn reset(&mut self) -> Result<()>;
    fn prefill(&mut self, t: &[u32]) -> Result<crate::model::Sample>;
    fn step(&mut self, t: u32) -> Result<crate::model::Sample>;
    fn logits(&self) -> Result<Vec<f32>>;
}

macro_rules! pass_for {
    ($t:ty) => {
        impl Pass for $t {
            fn reset(&mut self) -> Result<()> {
                <$t>::reset(self)
            }
            fn prefill(&mut self, t: &[u32]) -> Result<crate::model::Sample> {
                <$t>::prefill(self, t)
            }
            fn step(&mut self, t: u32) -> Result<crate::model::Sample> {
                <$t>::step(self, t)
            }
            fn logits(&self) -> Result<Vec<f32>> {
                <$t>::logits(self)
            }
        }
    };
}
pass_for!(EngineNh);
pass_for!(crate::dynamic::Dyn);

fn on_disk(dir: &Path, o: &OptsNh, level: Layout, n: usize, (prompt, steps): (&[u32], &[u32]), (mode, budget, all): (Offload, u64, u64), run: Run) -> Result<bool> {
    use crate::moe_nh::factory_cached;
    let mut results = Vec::new();
    for bytes in [None, Some(budget)] {
        let store = crate::offload::ExpertStore::new();
        store.set_offload(mode);
        store.hold_costs();
        store.set_host_memory(bytes.map(|bytes| HostMemory { bytes, total: all }));
        let make = factory_cached(level, n, &store);
        let mut e = EngineNh::new(dir, o, Some(&make))?;
        let (experts, layers) = (e.w.cfg.moe.experts, e.w.cfg.layers_of(NhKind::Moe) * 2);
        let mut passes = Vec::new();
        for (pass, hints) in [("first pass", false), ("second pass, the cache warm", false), ("with hints between the steps", true)] {
            passes.push((pass, run(&mut e, prompt, steps, hints.then_some((&*store, layers, experts)))?));
        }
        // (a predictor that expects a layer's experts to be the ones the layer before it chose: places taken and copies queued, and experts read, while a call is under way)
        store.set_predictor(move |l, chosen| vec![((l + 1) % layers, chosen.to_vec()), ((l + 2) % layers, chosen.iter().rev().copied().collect())]);
        passes.push(("with a predictor hinting the layers ahead", run(&mut e, prompt, steps, None)?));
        results.push((passes, store.stats()?, store.host_memory().is_some()));
    }
    let (want, got) = (&results[0], &results[1]);
    let (gib, mut ok) = (budget as f64 / (1u64 << 30) as f64, true);
    for ((pass, a), (_, b)) in want.0.iter().zip(&got.0) {
        if a == b {
            println!("ok   {n} places, {gib:.5} GiB of experts in the computer's memory, {pass}: {} logits and {} tokens as the run with every expert in memory, to the bit", b.0.len(), b.1.len());
        } else {
            let at = a.0.iter().zip(&b.0).position(|(x, y)| x != y);
            println!("FAIL {n} places, {gib:.5} GiB of experts in the computer's memory, {pass}: the logits differ from the run with every expert in memory from the first at {at:?} (tokens {})", if a.1 == b.1 { "the same" } else { "differ" });
            ok = false;
        }
    }
    let st = got.1;
    println!("{n} places, {gib:.5} GiB: {} experts read from the packed directory ({:.2} MB in {:.1} ms), {} found in the computer's memory when wanted; the run with every expert in memory read {}", st.disk_reads, st.disk_bytes as f64 / 1e6, st.disk_ms, st.ram_hits, want.1.disk_reads);
    if !got.2 || st.disk_reads == 0 || want.2 || want.1.disk_reads != 0 {
        println!("FAIL {n} places, {gib:.5} GiB: the disk tier was not used where it should be (tier on: {}, reads {}), or was where it should not be (tier on: {}, reads {})", got.2, st.disk_reads, want.2, want.1.disk_reads);
        ok = false;
    }
    Ok(ok)
}

/// `glyd-engine diskcheck DIR [--reads N] [--depth N]`: the disk the packed directory of a Hummingbird model is on, as the SSD tier reads it: `reads` experts of random layers and numbers read whole (the six tensors of its two
/// matrices, placed as a slot holds them, src/moe_hb.rs `HbDisk`) with the page cache bypassed, one at a time (the first 32) and then `depth` at a time (16: the queue the tier uses); the same experts again, which a file
/// system that kept them in the page cache would give back faster. Each line is the experts' MB a second and the experts a second; a consumer NVMe on PCIe 4 reads about 5 to 7 GB/s. No GPU.
pub fn disk(dir: &Path, reads: usize, depth: usize) -> Result<()> {
    use crate::disk::{Job, Pool};
    use crate::moe_hb::{index_layer, HbDisk};
    use std::cell::RefCell;
    use std::sync::Arc;
    let Some((Layout::Hummingbird, manifest)) = v7_manifest(dir)? else { return err(format!("{}: not a glyd-v7 directory at the Hummingbird level", dir.display())) };
    let c = ConfigNemotronH::load(dir)?;
    let ck = io(Checkpoint::open(dir))?;
    let layers: Vec<usize> = (0..c.layers).filter(|&l| c.kinds[l] == NhKind::Moe).collect();
    let disks = layers.iter().map(|&l| Ok(Arc::new(HbDisk::open(&ck, &c, l, &index_layer(&manifest, &c, l)?)?))).collect::<Result<Vec<_>>>()?;
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let picks: Vec<(usize, usize)> = (0..reads.max(1)).map(|_| ((next() % layers.len() as u64) as usize, (next() % c.moe.experts as u64) as usize)).collect();
    // (a thread's scratch holds an expert's bytes as a slot does)
    thread_local! {
        static SCRATCH: RefCell<(Vec<u8>, Vec<u8>)> = const { RefCell::new((Vec::new(), Vec::new())) };
    }
    let jobs = |from: &[(usize, usize)]| -> Vec<Job> {
        from.iter()
            .map(|&(l, e)| {
                let d = disks[l].clone();
                Box::new(move || {
                    SCRATCH.with(|s| {
                        let (data, meta) = &mut *s.borrow_mut();
                        d.read_sized(e, data, meta).map_err(|e| std::io::Error::other(e.0))
                    })
                }) as Job
            })
            .collect()
    };
    let say = |what: &str, n: usize, secs: f64, bytes: f64| println!("diskcheck {what}: {n} experts, {:.0} MB in {secs:.2} s: {:.0} MB/s, {:.0} experts/s, {:.2} ms an expert", bytes / 1e6, bytes / 1e6 / secs, n as f64 / secs, secs * 1e3 / n as f64);
    println!("diskcheck: {} experts of the {} layers of {}", picks.len(), layers.len(), dir.display());
    // (an expert's bytes are its group's place's: the sum is what the reads moved)
    let bytes_of = |from: &[(usize, usize)]| -> f64 { from.iter().map(|&(l, e)| disks[l].bytes_of(e) as f64).sum() };
    let alone = &picks[..picks.len().min(32)];
    let pool = Pool::new(1);
    let t0 = Instant::now();
    io(pool.run(jobs(alone)))?;
    say("one at a time", alone.len(), t0.elapsed().as_secs_f64(), bytes_of(alone));
    let pool = Pool::new(depth.max(1));
    let mut first = 0.0;
    for pass in ["first pass", "the same experts again"] {
        let t0 = Instant::now();
        io(pool.run(jobs(&picks)))?;
        let secs = t0.elapsed().as_secs_f64();
        say(&format!("{} at a time, {pass}", depth.max(1)), picks.len(), secs, bytes_of(&picks));
        if first == 0.0 {
            first = secs;
        } else if secs * 3.0 < first {
            println!("diskcheck: the experts read again came {:.1} times faster: the page cache is keeping them (this file system takes no direct reads?)", first / secs);
        }
    }
    println!("diskcheck: the shards are read {} (the page cache is {})", disks[0].mode(), if disks[0].mode() == "buffered" { "asked to drop what is read, after each read" } else { "bypassed" });
    Ok(())
}

// ---- nhdecode ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// A pack of the glyd-v7 directory `ck` (`v7`: its level and glyd.json) decoded to bf16 rows, as the library decodes it on the GPU.
fn unpacked(ck: &Checkpoint, v7: &(Layout, Value), pk: &Pack35) -> Result<Vec<u16>> {
    let p = read_pack(ck, Some(v7), pk, v7.0)?;
    let Words::Tiered { tiers, .. } = p.words else { return err(format!("{}: not in a block layout", pk.key)) };
    let t = Tiered { rows: p.rows, cols: p.cols, data: p.data, blocks: p.b1, block_base: Vec::new(), tiers };
    let mut w = vec![0u16; p.rows * p.cols];
    // (by the pack's own tag: the directory's level may hold packs in more than one layout)
    // (a row layout's pack is its weight's transpose: decoded, then turned back to [out, in])
    let rows = rows_bits(tiers[0] >> 24).is_some();
    let ok = match tiers[0] >> 24 {
        INT6_TAG => unpack_int6(&t, &mut w),
        INT5_TAG => unpack_int5(&t, &mut w),
        INT4_TAG => unpack_int4(&t, &mut w),
        _ if rows => unpack_rows(&t, &mut w),
        _ => unpack_int3(&t, &mut w),
    };
    if ok {
        Ok(if rows { transpose(&w, t.rows, t.cols) } else { w })
    } else {
        err(format!("{}: its buffers do not decode", pk.key))
    }
}

/// OUT as `fetch_mini` (tools/nemotron_ref.py) makes a mini checkpoint of `layers` of the model of DIR, a glyd-v7 directory: config.json; model.safetensors, every tensor of those layers in the checkpoint's names (the
/// packs decoded to bf16 on the CPU, a mixture-of-experts layer's experts too, the small tensors as DIR holds them); x.bin, bf16 [tokens, hidden], the embeddings of random tokens decoded from the embedding's pack
/// (the layers' residual input); tokens.json.
pub fn decode(dir: &Path, out: &Path, layers: &[usize], tokens: usize) -> Result<()> {
    let cfg = ConfigNemotronH::load(dir)?;
    let v7 = v7_manifest(dir)?.ok_or_else(|| Error(format!("{}: not a glyd-v7 directory (glyd-engine pack MODEL DIR --layout i6|i5|i4 makes one)", dir.display())))?;
    let ck = io(Checkpoint::open(dir))?;
    if let Some(&l) = layers.iter().find(|&&l| l >= cfg.layers) {
        return err(format!("layer {l}, where the model has {}", cfg.layers));
    }
    let bytes_of = |w: &[u16]| -> Vec<u8> { w.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let mut tensors: Vec<(String, String, Vec<u64>, Vec<u8>)> = Vec::new();
    let layers: Vec<usize> = layers.iter().copied().collect::<std::collections::BTreeSet<_>>().into_iter().collect();
    for &l in &layers {
        for pk in cfg.layer_packs(l).iter().chain(&cfg.expert_packs(l)) {
            let w = unpacked(&ck, &v7, pk)?;
            let mut at = 0;
            for (name, rows) in &pk.parts {
                tensors.push((name.clone(), "BF16".into(), vec![*rows as u64, pk.cols as u64], bytes_of(&w[at..at + rows * pk.cols])));
                at += rows * pk.cols;
            }
        }
        for t in cfg.layer_kept(l) {
            let (_, info) = ck.get(&t.name).ok_or_else(|| Error(format!("no tensor {}", t.name)))?;
            tensors.push((t.name.clone(), info.dtype.clone(), info.shape.clone(), io(ck.read(&t.name))?));
        }
    }
    // the embeddings of random tokens
    let (vocab, hid) = (cfg.vocab, cfg.hidden);
    let table = unpacked(&ck, &v7, &Pack35 { key: "embed".into(), parts: vec![(format!("{}embeddings.weight", cfg.prefix), vocab)], cols: hid })?;
    let (low, mut r) = (1000.min(vocab / 2), 0x9E37_79B9_7F4A_7C15u64);
    let ids: Vec<usize> = (0..tokens)
        .map(|_| {
            r ^= r << 13;
            r ^= r >> 7;
            r ^= r << 17;
            low + (r % (vocab - low) as u64) as usize
        })
        .collect();
    io(std::fs::create_dir_all(out))?;
    io(std::fs::write(out.join("x.bin"), bytes_of(&ids.iter().flat_map(|&i| table[i * hid..(i + 1) * hid].iter().copied()).collect::<Vec<u16>>())))?;
    io(std::fs::write(out.join("tokens.json"), format!("{{\"ids\": {ids:?}}}")))?;
    io(std::fs::copy(dir.join("config.json"), out.join("config.json")))?;
    let refs: Vec<(&str, &str, Vec<u64>, &[u8])> = tensors.iter().map(|(n, d, s, b)| (n.as_str(), d.as_str(), s.clone(), &b[..])).collect();
    crate::pack::write_safetensors(&out.join("model.safetensors"), &[("format", "pt".to_string())], &refs)?;
    println!("{}: {} tensors of layers {layers:?} decoded from the packs of the {:?} level ({:.0} MB), {tokens} embeddings", out.display(), tensors.len(), v7.0, tensors.iter().map(|t| t.3.len()).sum::<usize>() as f64 / 1e6);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Some layers of a tiny glyd-v7 directory decoded to a mini checkpoint, at swift and at wren (the experts four-bit codes and the dense matrices six-bit ones, each decoded by its own tag): every tensor the reference reads of
    /// a Mamba-2, a mixture-of-experts (its experts too) and an attention layer is there, each the very weights glyd.json hashes (the packer's own decode of its pack, or the tensor as saved), and x.bin holds the decoded
    /// embeddings of the tokens tokens.json names. A bf16 directory, and a layer it does not have, are refused.
    #[test]
    fn a_v7_directory_decodes_to_the_weights_it_hashes() {
        use sha2::{Digest, Sha256};
        use std::collections::HashMap;
        let dir = std::env::temp_dir().join(format!("glyd-engine-nhdecode-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let src = dir.join("src");
        let text = include_str!("../tests/fixtures/nemotron_h_moetoy_config.json");
        let c = ConfigNemotronH::parse(text, "moetoy", None).unwrap();
        crate::nemotron::tests::checkpoint(&src, text, &c);
        let bf16 = Checkpoint::open(&src).unwrap();
        for level in [Layout::Int5, Layout::Int4] {
            let (packed, mini) = (dir.join(format!("packed-{}", level.name())), dir.join(format!("mini-{}", level.name())));
            let plan = crate::pack::nemotron_plan(&c, level, &|n| bf16.get(n).is_some_and(|(_, t)| t.dtype == "F32"));
            glyd_gpu::save::save_planned(&glyd_gpu::save::Source::find(src.to_str().unwrap()).unwrap(), &packed, 2, 5 << 30, level, &plan).unwrap();
            decode(&packed, &mini, &[0, 1, 2], 40).unwrap();
            // the sha256 glyd.json holds of each tensor: a pack's tensors (decoded) and those saved as they are
            let manifest = json::parse(&std::fs::read_to_string(packed.join("glyd.json")).unwrap()).unwrap();
            let mut want: HashMap<String, String> = HashMap::new();
            for (_, e) in manifest.get("packs").and_then(Value::as_object).unwrap() {
                for t in e.get("tensors").and_then(Value::as_array).unwrap() {
                    want.insert(t.get("name").and_then(Value::as_str).unwrap().to_string(), t.get("sha256").and_then(Value::as_str).unwrap().to_string());
                }
            }
            for (n, h) in manifest.get("tensors").and_then(Value::as_object).unwrap() {
                want.insert(n.clone(), h.as_str().unwrap().to_string());
            }
            let sha = |b: &[u8]| -> String { Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect() };
            let got = Checkpoint::open(&mini).unwrap();
            // layer 0: in_proj, out_proj and 7 small tensors; layer 1: 4 matrices, 16 experts' two and 3 small tensors; layer 2: q, k, v, o and a norm
            assert_eq!(got.names().count(), 9 + (4 + 32 + 3) + 5);
            for n in got.names() {
                assert_eq!(Some(&sha(&got.read(n).unwrap())), want.get(n), "{n}");
            }
            assert_eq!(got.get("backbone.layers.1.mixer.experts.15.down_proj.weight").unwrap().1.shape, [128, 128]);
            assert!(got.get("backbone.layers.3.norm.weight").is_none());
            // the embeddings of the tokens: rows of the embedding's pack, decoded
            let ids = json::parse(&std::fs::read_to_string(mini.join("tokens.json")).unwrap()).unwrap().get("ids").and_then(Value::as_u64s).unwrap();
            let (packed_ck, v7) = (Checkpoint::open(&packed).unwrap(), v7_manifest(&packed).unwrap().unwrap());
            let table = unpacked(&packed_ck, &v7, &Pack35 { key: "embed".into(), parts: vec![("backbone.embeddings.weight".into(), c.vocab)], cols: c.hidden }).unwrap();
            let x = std::fs::read(mini.join("x.bin")).unwrap();
            assert_eq!((ids.len(), x.len()), (40, 40 * c.hidden * 2));
            for (i, &id) in ids.iter().enumerate() {
                let row: Vec<u8> = table[id as usize * c.hidden..(id as usize + 1) * c.hidden].iter().flat_map(|w| w.to_le_bytes()).collect();
                assert!(x[i * c.hidden * 2..(i + 1) * c.hidden * 2] == row[..], "token {i}");
            }
            assert_eq!(std::fs::read(mini.join("config.json")).unwrap(), std::fs::read(packed.join("config.json")).unwrap());
            assert!(decode(&src, &dir.join("x"), &[0], 4).unwrap_err().0.contains("not a glyd-v7 directory"));
            assert!(decode(&packed, &dir.join("y"), &[6], 4).unwrap_err().0.contains("layer 6, where the model has 6"));
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The toy's level: 6 bits unless the files say 5 or 4; no other.
    #[test]
    fn a_toys_level_is_six_five_or_four_bits() {
        assert_eq!(level_of(None).unwrap(), Layout::Int6);
        assert_eq!(level_of(Some(6)).unwrap(), Layout::Int6);
        assert_eq!(level_of(Some(5)).unwrap(), Layout::Int5);
        assert_eq!(level_of(Some(4)).unwrap(), Layout::Int4);
        assert!(level_of(Some(3)).unwrap_err().0.contains("3 bits"));
    }

    fn row(v: &[f32]) -> Vec<u16> {
        v.iter().map(|x| (x.to_bits() >> 16) as u16).collect()
    }

    /// A run held to the run with every expert on the GPU: the same logits are within it; a logit a place of bf16 away, or two, is; one further than the bound is not; a number that is not one is not; a token picked
    /// that the best has by less than the bound is; by more is not; and rows that are not the same shape are refused.
    #[test]
    fn logits_are_held_to_the_run_with_every_expert_on_the_gpu() {
        let want = [row(&[16.0, 15.0, -3.0, 0.5]), row(&[1.0, 2.0, 8.0, 7.0])].concat();
        let ok = logits_within(&want, &want, 2, 0.05).unwrap();
        assert!(ok.contains("bit for bit at 2") && ok.contains("the same at 2"), "{ok}");
        // a place of bf16 at the largest (16.0 to 16.125) and in another (7.0 to 7.03125), the same token
        let near = [row(&[16.125, 15.0, -3.0, 0.5]), row(&[1.0, 2.0, 8.0, 7.03125])].concat();
        let ok = logits_within(&near, &want, 2, 1.0 / 64.0).unwrap();
        assert!(ok.contains("bit for bit at 0") && ok.contains("the same at 2"), "{ok}");
        // 16.0 to 17.0 is 1/16 of the largest: past 1/64, within 1/8
        let far = [row(&[17.0, 15.0, -3.0, 0.5]), row(&[1.0, 2.0, 8.0, 7.0])].concat();
        assert!(logits_within(&far, &want, 2, 1.0 / 64.0).unwrap_err().contains("position 0"));
        assert!(logits_within(&far, &want, 2, 0.125).is_ok());
        let nan = [row(&[16.0, f32::NAN, -3.0, 0.5]), row(&[1.0, 2.0, 8.0, 7.0])].concat();
        assert!(logits_within(&nan, &want, 2, 0.5).unwrap_err().contains("not a number"));
        // another token picked: 15.3 and 15.75 against 16 and 15 (a 16th of the largest below the best, past 0.05)
        let flip = [row(&[15.3125, 15.75, -3.0, 0.5]), row(&[1.0, 2.0, 8.0, 7.0])].concat();
        let e = logits_within(&flip, &want, 2, 0.05).unwrap_err();
        assert!(e.contains("position 0") && e.contains("the token picked (1)"), "{e}");
        // the same flip is fine where the two logits are within the bound of each other in the reference
        let tied = [row(&[16.0, 15.9375, -3.0, 0.5]), row(&[1.0, 2.0, 8.0, 7.0])].concat();
        let flipped = [row(&[15.5, 16.0, -3.0, 0.5]), row(&[1.0, 2.0, 8.0, 7.0])].concat();
        let ok = logits_within(&flipped, &tied, 2, 0.05).unwrap();
        assert!(ok.contains("the same at 1"), "{ok}");
        assert!(logits_within(&want[..7], &want, 2, 0.1).is_err() && logits_within(&want, &want, 0, 0.1).is_err() && logits_within(&want, &want, 3, 0.1).is_err());
    }

    /// `diskcheck` reads the tiny model's directory (random whole experts, one at a time and several at a time, the same ones again) and says so; a directory that is not a Hummingbird one is refused, and a budget for the
    /// computer's memory is a share of the directory's experts (`--host-memory` of `generate`: more than they are is all of them, the plan's `ram`).
    #[test]
    fn diskcheck_reads_a_directory_and_the_budget_is_a_share_of_it() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-diskcheck-{}", std::process::id()));
        let (out, c) = crate::hbpack::tests::packed_toy(&dir, 1 << 30, 40);
        disk(&out, 48, 4).unwrap();
        let e = disk(&dir.join("src"), 4, 2).unwrap_err().0;
        assert!(e.contains("not a glyd-v7 directory at the Hummingbird level"), "{e}");
        // the tiny directory's experts: 6 at three bits and 10 at two in each of three layers
        let (_, m) = v7_manifest(&out).unwrap().unwrap();
        let total = crate::moe_hb::experts_bytes(&m, &c).unwrap();
        assert_eq!(total, 3 * (6 * crate::moe_hb::expert_bytes_at(&c, 3).unwrap() + 10 * crate::moe_hb::expert_bytes_at(&c, 2).unwrap()));
        let mut plan = CachePlan { layers: 3, experts: 16, slots: 4, expert_bytes: total / 48, in_place: true, ram: total };
        let args = |extra: &[&str]| crate::args::Args::from(["generate".to_string(), "x".to_string()].into_iter().chain(extra.iter().map(|s| s.to_string())));
        host_budget(&args(&["--host-memory", "0.0003"]), &mut plan).unwrap();
        assert_eq!(plan.ram, (0.0003 * (1u64 << 30) as f64) as u64);
        host_budget(&args(&["--host-memory", "5"]), &mut plan).unwrap();
        assert_eq!(plan.ram, plan.host_bytes(), "more than the experts is all of them");
        assert!(host_budget(&args(&["--host-memory", "lots"]), &mut plan).is_err());
        let mut wren = CachePlan { in_place: false, ram: total, ..plan };
        assert!(host_budget(&args(&["--host-memory", "0.0003"]), &mut wren).unwrap_err().0.contains("only the Hummingbird level"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// `nhrandom OUT [--latent N] [--experts E]`: a random Nemotron-H model at Nemotron 3.5 Lightning's widths (Mamba-2, attention and mixture-of-experts layers "M*ME", E experts (16) of 6 a token, a vocabulary of
/// 4,096; with a latent of N, Super's kind of layer), every tensor the loader reads in its dtype and shape (config.json, model.safetensors): where the real model does not fit, the two engines are
/// compared on it (`check --dynamic`).
pub fn random(out: &Path, latent: usize, experts: usize) -> Result<()> {
    let lat = if latent > 0 { latent.to_string() } else { "null".into() };
    let text = format!(
        r#"{{"architectures": ["NemotronHForCausalLM"], "model_type": "nemotron_h", "hidden_size": 2688, "num_hidden_layers": 4, "hybrid_override_pattern": "M*ME", "vocab_size": 4096,
        "layer_norm_epsilon": 1e-5, "max_position_embeddings": 8192, "mamba_num_heads": 64, "mamba_head_dim": 64, "n_groups": 8, "ssm_state_size": 128, "conv_kernel": 4, "num_attention_heads": 32,
        "num_key_value_heads": 2, "head_dim": 128, "n_routed_experts": {experts}, "num_experts_per_tok": 6, "moe_intermediate_size": 1856, "moe_shared_expert_intermediate_size": 3712, "moe_latent_size": {lat},
        "routed_scaling_factor": 2.5, "norm_topk_prob": true, "n_group": 1, "topk_group": 1, "n_shared_experts": 1, "mlp_hidden_act": "relu2", "tie_word_embeddings": false, "eos_token_id": 2}}"#
    );
    let c = ConfigNemotronH::parse(&text, "random", None)?;
    let mut r = 0x2026_1017_0000_0001u64;
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
    for pk in c.packs() {
        let a = (3.0 / pk.cols as f32).sqrt();
        for (name, rows) in &pk.parts {
            let w: Vec<u16> = (0..rows * pk.cols).map(|_| bf(uni(-a, a))).collect();
            tensors.push((name.clone(), "BF16", vec![*rows as u64, pk.cols as u64], crate::weights::as_bytes(&w).to_vec()));
        }
    }
    for t in c.kept() {
        let n: u64 = t.shape.iter().product();
        let (lo, hi) = match t.name.rsplit('.').next().unwrap_or("") {
            _ if t.name.ends_with("norm.weight") || t.name.ends_with("norm_f.weight") => (0.9, 1.1),
            "A_log" => (0.0, 2.7),
            "D" => (0.9, 1.1),
            "dt_bias" => (-4.6, -2.3),
            "bias" if t.name.contains("conv1d") => (-0.1, 0.1),
            "e_score_correction_bias" => (0.0, 0.1),
            _ if t.name.contains("conv1d") => (-0.5, 0.5),
            _ => {
                let a = (3.0 / *t.shape.last().unwrap_or(&1) as f32).sqrt();
                (-a, a)
            }
        };
        let bytes = if t.dtype == "F32" { (0..n).flat_map(|_| uni(lo, hi).to_le_bytes()).collect() } else { crate::weights::as_bytes(&(0..n).map(|_| bf(uni(lo, hi))).collect::<Vec<u16>>()).to_vec() };
        tensors.push((t.name.clone(), t.dtype, t.shape.clone(), bytes));
    }
    io(std::fs::create_dir_all(out))?;
    io(std::fs::write(out.join("config.json"), &text))?;
    let refs: Vec<(&str, &str, Vec<u64>, &[u8])> = tensors.iter().map(|(n, d, s, b)| (n.as_str(), *d, s.clone(), &b[..])).collect();
    crate::pack::write_safetensors(&out.join("model.safetensors"), &[("format", "pt".to_string())], &refs)?;
    println!("{}: a random Nemotron-H model, {} tensors, {:.0} MB", out.display(), tensors.len(), tensors.iter().map(|t| t.3.len()).sum::<usize>() as f64 / 1e6);
    Ok(())
}
