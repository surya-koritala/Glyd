//! `kindcheck [CASE...] [--prompt N] [--steps N] [--scratch DIR]` (R2, #135): every layer kind of the recent families run by the executor's CUDA
//! kernels against its CPU f32 reference (`runtime::reference`), kind by kind, on each logits case's random model (Gemma 4, the Qwen3.5 layout,
//! Nemotron-H, Muse Spark, Kimi-K3, GLM-5.3-Flash, DeepSeek-V4) at the lossless level.
//!
//! Each layer of the executor's description runs operation by operation on a prompt of `--prompt` rows, then `--steps` decode steps, each pass
//! from a residual made at random (the layer's caches and states carried from pass to pass, as a sequence's). A layer's operations are runs of
//! the residual's (the norms, the residual rule, mHC's mixing, the depth mix) and of the blocks' bodies (a mixer or a feed-forward); each is
//! compared where it ends:
//!
//! - a body's output slot against the reference's body (`RefLayer::body`) on the GPU's own input rows (H), so a body's error is its own;
//! - after the residual's operations, the residual (X, mHC's streams S, the depth mix's partial sum and completed groups) and the next body's
//!   input H against the reference's residual rule, advanced by the GPU's own body outputs, and its norms (the final norm after the last).
//!
//! A comparison is the largest difference over the reference's largest magnitude on a pass's rows; each line has its kind's bound (`bound`,
//! the table in `runtime::reference`), and names the rows left out as near a tie (`NEAR`). Exit status 1 where one is not kept.
//! GLYD_KINDCHECK_ROWS prints each body's rows one by one. DeepSeek-V4's simulated quantizations of the release (GLYD_V4_SIM) are turned
//! off: no kind names them.

use super::{Dyn, Op, Opts, Slot};
use crate::ffi::{err, Error, Result};
use crate::runtime::kinds::{AttnRounding, Block, Body, Ffn, LinearKind, Mixer, NormKind, NormWeight, Residual};
use crate::runtime::reference::model::{checkpoint_tensor, Depth, RefModel};
use crate::runtime::reference::{bf16, Hc};
use crate::weights::Layout;
use std::ops::Range;
use std::path::{Path, PathBuf};

/// The rows of a comparison's kind in the bounds' table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Class {
    Attention,
    Latent,
    Compressed,
    DeltaNet,
    KimiDelta,
    Mamba2,
    Mlp,
    Moe,
    Residual,
}

/// Each class's bound: the largest difference over the reference's largest magnitude a kernel may have (why each: `runtime::reference`).
pub fn bound(c: Class) -> f32 {
    match c {
        Class::Attention | Class::Latent | Class::Compressed => 2e-2,
        Class::DeltaNet | Class::KimiDelta | Class::Mamba2 => 3e-2,
        Class::Mlp => 1e-2,
        Class::Moe => 2e-2,
        Class::Residual => 2e-2,
    }
}

/// One comparison: where (case, layer, block), its kind, its worst on the prompt's pass and on the steps', and the rows left out as near a
/// tie (`NEAR`).
struct Line {
    case: &'static str,
    at: String,
    kind: String,
    class: Class,
    prompt: f32,
    steps: f32,
    near: Vec<String>,
}

/// A row past its bound whose choice (a router's experts, an indexer's picks) was nearer than this to another (`reference::ffn::margin`: the
/// gap over the largest score) is left out of its body's comparison and named: the GPU's scores of bf16 inputs can order such a pair the
/// other way, which changes the output by far more than its arithmetic does.
pub(crate) const NEAR: f32 = 1e-2;

/// The largest difference over `want`'s largest magnitude (infinite where `got` has a value that is not finite).
fn rel(got: &[f32], want: &[f32]) -> f32 {
    if got.len() != want.len() || got.iter().any(|v| !v.is_finite()) {
        return f32::INFINITY;
    }
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
    got.iter().zip(want).fold(0f32, |m, (a, b)| m.max((a - b).abs())) / scale
}

/// A block's kind as a line names it, and its class.
fn label(b: &Block) -> (Class, String) {
    match &b.body {
        Body::Mixer(Mixer::Attention(a)) => {
            let mut s = format!("attention: {} heads over {} of {}", a.heads, a.kv_heads, a.head_dim);
            if let Some(w) = a.window {
                s += &format!(", window {w}");
            }
            if a.rope.is_none() {
                s += ", no rope";
            } else if a.rope.as_ref().is_some_and(|r| r.freqs * 2 < r.dim || r.dim < a.head_dim) {
                s += ", partial rope";
            }
            if a.k_eq_v {
                s += ", K = V";
            }
            if let Some(g) = a.gate {
                s += &format!(", {:?} gate{}", g.from, if g.norm.is_some() { " after a norm" } else { "" });
            }
            if a.softcap.is_some() {
                s += ", soft-capped";
            }
            if a.sinks {
                s += ", sinks";
            }
            s += &format!(
                ", scale {:.4}, {}",
                a.scale,
                match a.rounding {
                    AttnRounding::F32 => "f32",
                    AttnRounding::Flash => "bf16 probabilities",
                    AttnRounding::Eager => "bf16 scores",
                    AttnRounding::Sdpa => "bf16 rope, f32 softmax",
                }
            );
            (Class::Attention, s)
        }
        Body::Mixer(Mixer::Mla(m)) => {
            let ix =
                m.indexer.as_ref().map_or(String::new(), |i| format!(", indexer of {} of {} keeping {} (pools of {})", i.heads, i.head_dim, i.top_k, i.block));
            (
                Class::Latent,
                format!("latent attention: {} heads, rank {}, rope {}{}{ix}", m.heads, m.kv_lora, m.rope_dim, if m.gate.is_some() { ", gated" } else { "" }),
            )
        }
        Body::Mixer(Mixer::Compressed(c)) => {
            let ix = c.indexer.as_ref().map_or(String::new(), |i| format!(", indexer keeping {}", i.top_k));
            (Class::Compressed, format!("compressed attention: window {}, ratio {}{}{ix}", c.window, c.ratio, if c.overlap { " overlapping" } else { "" }))
        }
        Body::Mixer(Mixer::Linear(l)) => match l.kind {
            LinearKind::GatedDeltaNet => (Class::DeltaNet, format!("Gated DeltaNet: {} key heads, {} value heads", l.k_heads, l.v_heads)),
            LinearKind::KimiDelta => (Class::KimiDelta, format!("Kimi delta attention: {} heads of {}", l.k_heads, l.k_dim)),
        },
        Body::Mixer(Mixer::Mamba2(m)) => (Class::Mamba2, format!("Mamba-2: {} heads of {}, state {}, {} groups", m.heads, m.head_dim, m.state, m.groups)),
        Body::Ffn(Ffn::Mlp(m)) => {
            (Class::Mlp, format!("MLP: {:?}{}{}", m.act, if m.gated { ", gated" } else { "" }, m.clamp.map_or(String::new(), |c| format!(", limit {c}"))))
        }
        Body::Ffn(Ffn::Moe(e)) => {
            let r = &e.router;
            let mut s = format!("MoE: {:?} router, top {} of {}", r.score, r.top_k, e.experts);
            if r.groups.is_some() {
                s += ", groups";
            }
            if r.hash {
                s += ", hash";
            }
            if e.shared.is_some() {
                s += if e.shared_gate { ", gated shared" } else { ", shared" };
            }
            if e.latent.is_some() {
                s += ", latent";
            }
            s += &format!(", {:?} experts", e.expert.act);
            (Class::Moe, s)
        }
        Body::Ngram(n) => (Class::Residual, format!("n-gram embedding: orders up to {}, {} tables each, {} streams", n.order, n.heads, n.streams)),
    }
}

/// The residual's kind before a block (or the final norm): its norm and its rule.
fn residual_label(b: Option<&Block>) -> String {
    let Some(b) = b else { return "the final norm".into() };
    let n = &b.norm_in;
    let norm = match (n.kind, n.weight) {
        (NormKind::Rms, NormWeight::PlusOne) => "RMS (1 + w)",
        (NormKind::Rms, NormWeight::InputPlusOne) => "RMS of x (1 + w)",
        (NormKind::Rms, _) => "RMS",
        (NormKind::Layer, _) => "LayerNorm",
        (NormKind::L2, _) => "l2",
    };
    let rule = match &b.residual {
        Residual::Add => "add".to_string(),
        Residual::AddScaled(s) => format!("add times {s}"),
        Residual::AddThenScale => "add, then the layer's scalar".into(),
        Residual::Gated(_) => "gated".into(),
        Residual::Streams(s) => format!("mHC over {} streams", s.streams),
        Residual::GatedStreams(g) => format!("gated residuals over {} streams", g.streams),
        Residual::DepthMix(_) => "depth mix".into(),
    };
    format!("residual ({rule}) and the {norm} norm{}", if b.norm_out.is_some() { ", with the block's post-norm" } else { "" })
}

/// Whether an operation is the residual's (it reads or writes X, mHC's streams or the depth mix's blocks, or it is a block's post-norm of its
/// output in place) and not a body's.
fn residual_op(op: &Op) -> bool {
    match *op {
        Op::AddNorm { .. } | Op::NormAdd { .. } | Op::EmbedNorm { .. } | Op::GatedResidual { .. } | Op::Nop => true,
        Op::RmsNorm { x, y, .. } => x == Slot::X || x == y,
        Op::Streams | Op::StreamsPre { .. } | Op::StreamsPost { .. } | Op::StreamsHead { .. } => true,
        Op::DepthMix(_) | Op::Add { .. } | Op::MoeEarly { .. } => true,
        _ => false,
    }
}

/// Where a body's last operation writes its output.
fn out_slot(op: &Op) -> Result<Slot> {
    match *op {
        Op::Linear { y, .. } | Op::F8Lin { y, .. } | Op::LinearAt { y, .. } => Ok(y),
        Op::Moe(m) => Ok(m.y),
        Op::Mla { .. } => Ok(Slot::O),
        ref o => err(format!("kindcheck: where {o:?} writes a body's output is not known")),
    }
}

/// A layer's operations as runs: (a body's, its range).
fn runs(ops: &[Op]) -> Vec<(bool, Range<usize>)> {
    let mut v: Vec<(bool, Range<usize>)> = Vec::new();
    for (i, op) in ops.iter().enumerate() {
        let body = !residual_op(op);
        match v.last_mut() {
            Some((b, r)) if *b == body => r.end = i + 1,
            _ => v.push((body, i..i + 1)),
        }
    }
    v
}

struct Rng(u64);

impl Rng {
    /// n bf16 values uniform in [-1, 1).
    fn v(&mut self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                bf16(2.0 * ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 1.0)
            })
            .collect()
    }
}

/// A token's residual on the reference: its stream(s), or the depth mix's state; and the mHC weights its block's `pre` gave its `post`.
struct Row {
    x: Vec<f32>,
    depth: Option<Depth>,
    hc: Option<Hc>,
}

impl Dyn {
    fn kc_width(&self, s: Slot) -> usize {
        self.big.d.widths[s as usize]
    }

    fn kc_put(&self, s: Slot, v: &[f32]) -> Result<()> {
        let w: Vec<u16> = v.iter().map(|&x| (bf16(x).to_bits() >> 16) as u16).collect();
        self.big.slots[s as usize].write(0, &w)
    }

    fn kc_get(&self, s: Slot, n: usize) -> Result<Vec<f32>> {
        self.dev.sync()?;
        let mut w = vec![0u16; n * self.kc_width(s)];
        self.big.slots[s as usize].read(0, &mut w)?;
        Ok(w.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect())
    }

    /// Operations `ops` of layer `l` on n rows at positions `at`.., the tokens `tok` (DeepSeek-V4's hash routing reads them).
    fn kc_ops(&self, l: usize, ops: Range<usize>, n: usize, at: usize, tok: &[u32]) -> Result<()> {
        self.dev.sync()?;
        self.pos.write(0, &(0..n).map(|i| (at + i) as i32).collect::<Vec<i32>>())?;
        if let Some(v) = &self.tok {
            v.write(0, &tok.iter().map(|&t| t as i32).collect::<Vec<i32>>())?;
        }
        self.start.set(at);
        let pp = self.pos.ptr;
        // (what `forward` makes before a pass's operations: the decode's rows' lengths and first tokens, which the attention's decode steps read)
        // SAFETY: as below.
        unsafe { self.pass_prepare(&self.big, n, (pp, pp))? };
        for i in ops {
            // SAFETY: the engine's own buffers, sized for a chunk (n rows at most a chunk); its slots hold the rows the operation reads (written above or by the
            // operations before it), and the stream orders the launches; it is synchronized before a slot is read.
            unsafe { self.op(&self.big, l, i, &self.big.d.layers[l][i], n, (pp, pp), 0)? };
        }
        self.dev.sync()
    }
}

/// What one pass of a layer compares: the comparisons' (where, kind, class, value, the rows near a tie).
type Got = Vec<(String, String, Class, f32, Vec<String>)>;

/// Layer `l` of the executor (blocks `per l ..`) on n rows at position `at`, from a residual made by `rng`.
fn pass(dy: &Dyn, m: &mut RefModel, l: usize, per: usize, n: usize, at: usize, rng: &mut Rng) -> Result<Got> {
    let (d, layers) = (m.desc.hidden, dy.big.d.layers.len());
    let tok: Vec<u32> = (0..n).map(|t| (((at + t) * 37 + 11) % m.desc.vocab) as u32).collect();
    let runs_l = runs(&dy.big.d.layers[l]);
    // (the residual: mHC's streams where the model has them, else X; the depth mix's completed groups in Blk)
    let state = if dy.kc_width(Slot::S) > 0 { Slot::S } else { Slot::X };
    let sw = dy.kc_width(state);
    let x0 = rng.v(n * sw);
    dy.kc_put(state, &x0)?;
    for s in [Slot::O, Slot::D] {
        if dy.kc_width(s) > 0 {
            dy.kc_put(s, &vec![0.0; n * dy.kc_width(s)])?;
        }
    }
    let depth = matches!(m.layers[per * l].block.residual, Residual::DepthMix(_));
    let (bw, nb) = (dy.kc_width(Slot::Blk), m.layers[..per * l].iter().filter(|b| matches!(b.block.residual, Residual::DepthMix(dm) if dm.starts)).count());
    let mut rows: Vec<Row> = Vec::with_capacity(n);
    let blk = if bw > 0 {
        let mut b = vec![0f32; n * bw];
        for t in 0..n {
            b[t * bw..][..nb * d].copy_from_slice(&rng.v(nb * d));
        }
        dy.kc_put(Slot::Blk, &b)?;
        b
    } else {
        Vec::new()
    };
    for t in 0..n {
        let x = x0[t * sw..][..sw].to_vec();
        let dp = depth.then(|| Depth { sources: (0..nb).map(|j| blk[t * bw + j * d..][..d].to_vec()).collect(), partial: Some(x.clone()) });
        rows.push(Row { x, depth: dp, hc: None });
    }
    let mut got: Got = Vec::new();
    let mut put = |at: String, kind: String, class: Class, v: f32, near: Vec<String>| got.push((at, kind, class, v, near));
    // the reference's input to block b (or the final norm's, b = None), from each row's residual: H as the reference makes it
    let pre = |m: &RefModel, rows: &mut Vec<Row>, b: Option<usize>| -> Result<Vec<f32>> {
        let mut h = Vec::with_capacity(n * d);
        for r in rows.iter_mut() {
            match b {
                None => h.extend(m.final_h(&r.x, r.depth.as_ref())?),
                Some(b) => {
                    let blk = &m.layers[b];
                    let input = match (&blk.block.residual, r.depth.as_mut()) {
                        (Residual::DepthMix(dm), Some(dp)) => dp.input(dm.starts, blk.p.vec("mix_norm")?, blk.p.vec("mix_proj")?),
                        _ => {
                            let (input, hc) = blk.pre(&r.x)?;
                            r.hc = hc;
                            input
                        }
                    };
                    h.extend(blk.norm_in(&input));
                }
            }
        }
        Ok(h)
    };
    // the residual rule of block b with the GPU's output o of its body
    let post = |m: &RefModel, rows: &mut Vec<Row>, b: usize, o: &[f32]| -> Result<()> {
        let blk = &m.layers[b];
        for (t, r) in rows.iter_mut().enumerate() {
            let ot = blk.norm_out(o[t * d..][..d].to_vec());
            match r.depth.as_mut() {
                Some(dp) => dp.add(&ot),
                None => blk.post(&mut r.x, &ot, r.hc.take())?,
            }
        }
        Ok(())
    };
    // the GPU's residual against the reference's
    let state_rel = |rows: &Vec<Row>| -> Result<f32> {
        let x = dy.kc_get(state, n)?;
        let want: Vec<f32> = rows.iter().flat_map(|r| r.depth.as_ref().map_or(r.x.clone(), |dp| dp.partial.clone().unwrap_or(vec![0.0; d]))).collect();
        let mut v = rel(&x, &want);
        if depth {
            let b = dy.kc_get(Slot::Blk, n)?;
            let (got, want): (Vec<f32>, Vec<f32>) = rows
                .iter()
                .enumerate()
                .flat_map(|(t, r)| {
                    let dp = r.depth.as_ref().expect("a depth mix");
                    dp.sources.iter().enumerate().flat_map(move |(j, s)| (0..d).map(move |i| (t, j, i, s[i])))
                })
                .map(|(t, j, i, w)| (b[t * bw + j * d + i], w))
                .unzip();
            v = v.max(rel(&got, &want));
        }
        Ok(v)
    };
    let mut pending: Option<(usize, Vec<f32>)> = None;
    let mut body = 0;
    let mut last_ran = false;
    for (k, (is_body, r)) in runs_l.iter().enumerate() {
        let b = per * l + body;
        if !is_body {
            dy.kc_ops(l, r.clone(), n, at, &tok)?;
            last_ran = true;
            if body == per {
                // (the trailing run: compared with the next layer's first run, below)
                continue;
            }
            if let Some((pb, o)) = pending.take() {
                post(m, &mut rows, pb, &o)?;
            }
            let h = pre(m, &mut rows, Some(b))?;
            put(
                format!("L{l} before B{b}"),
                residual_label(Some(&m.layers[b].block)),
                Class::Residual,
                state_rel(&rows)?.max(rel(&dy.kc_get(Slot::H, n)?, &h)),
                Vec::new(),
            );
            continue;
        }
        if body >= per {
            return err(format!("layer {l}: more bodies than its {per} blocks"));
        }
        if k == 0 {
            // (a layer that begins with a body reads the H the layer before made: the reference's, rounded as a slot holds it)
            let h = pre(m, &mut rows, Some(b))?;
            dy.kc_put(Slot::H, &h)?;
        }
        let h = dy.kc_get(Slot::H, n)?;
        dy.kc_ops(l, r.clone(), n, at, &tok)?;
        last_ran = false;
        let out = out_slot(&dy.big.d.layers[l][r.end - 1])?;
        if dy.kc_width(out) != d {
            return err(format!("layer {l}: a body's output slot {out:?} of {} values, not {d}", dy.kc_width(out)));
        }
        let o = dy.kc_get(out, n)?;
        let (mut want, mut margin) = (Vec::with_capacity(n * d), Vec::with_capacity(n));
        for t in 0..n {
            want.extend(m.layers[b].body(0, (at + t) as u32, tok[t], &h[t * d..][..d])?);
            margin.push(m.layers[b].margin);
        }
        // (each row's largest difference over the pass's scale; a row past the bound whose choice was near a tie is left out, and named)
        let (class, kind) = label(&m.layers[b].block);
        let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-30);
        let (mut worst, mut near) = (0f32, Vec::new());
        for t in 0..n {
            let r = rel(&o[t * d..][..d], &want[t * d..][..d]) * want[t * d..][..d].iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-30) / scale;
            // (GLYD_KINDCHECK_ROWS: each row's difference, a body at a time)
            if crate::tune!("GLYD_KINDCHECK_ROWS").is_some() {
                println!("     L{l} B{b} position {}: {r:.2e} (margin {:.1e})", at + t, margin[t]);
            }
            if margin[t] < NEAR && r > bound(class) {
                near.push(format!("position {} (margin {:.1e}, off by {r:.1e})", at + t, margin[t]));
            } else {
                worst = worst.max(r);
            }
        }
        put(format!("L{l} B{b}"), kind, class, worst, near);
        pending = Some((b, o));
        body += 1;
    }
    if body != per {
        return err(format!("layer {l}: {body} bodies, where it has {per} blocks"));
    }
    // the residual after the last body: the layer's last run and the next layer's first (its norm), or the final norm
    let next = (l + 1 < layers).then(|| runs(&dy.big.d.layers[l + 1]).into_iter().next().filter(|(b, _)| !b)).flatten();
    if let Some((_, r)) = &next {
        dy.kc_ops(l + 1, r.clone(), n, at, &tok)?;
    }
    if last_ran || next.is_some() {
        if let Some((pb, o)) = pending.take() {
            post(m, &mut rows, pb, &o)?;
        }
        let b = (l + 1 < layers).then_some(per * (l + 1));
        let h = pre(m, &mut rows, b)?;
        let what = b.map_or("the final norm".to_string(), |b| format!("B{b}"));
        put(
            format!("L{l} before {what}"),
            residual_label(b.map(|b| &m.layers[b].block)),
            Class::Residual,
            state_rel(&rows)?.max(rel(&dy.kc_get(Slot::H, n)?, &h)),
            Vec::new(),
        );
    }
    Ok(got)
}

/// A case's model made (its logits case's random one), loaded lossless, described and loaded on the reference; every layer through `pass`.
fn case(id: &'static str, dir: &Path, prompt: usize, steps: usize) -> Result<Vec<Line>> {
    let c = crate::logits::find_case(id)?;
    if !dir.join("config.json").exists() {
        match c.weights {
            crate::logits::Weights::Random(make) => make(dir)?,
            crate::logits::Weights::Hub(h) => return err(format!("{id}: a Hub model ({h}), not a random one")),
        }
    }
    let o = Opts { ctx: 256, chunk: 64, graph: false, device: 0, layout: Some(Layout::Tiered) };
    let mut dy = match id {
        // (its logits case runs the experts lossy; here the lossless level's)
        "glm" => super::glm::load(dir, &o, Layout::Tiered)?,
        "nemotron-h" => crate::logits::load_dyn("nemotron", dir, &o, "penguin")?,
        _ => crate::logits::load_dyn(id, dir, &o, "penguin")?,
    };
    dy.reset()?;
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).map_err(|e| Error(e.to_string()))?)
        .map_err(|e| Error(format!("config.json: {e}")))?;
    let d = crate::runtime::describe::describe(&cfg)?;
    let ck = glyd_gpu::safetensors::Checkpoint::open(dir).map_err(|e| Error(e.to_string()))?;
    let mut m = RefModel::load(&d, &|n: &str| checkpoint_tensor(&ck, n))?;
    let layers = dy.big.d.layers.len();
    if layers == 0 || m.layers.len() % layers != 0 {
        return err(format!("{id}: {} blocks over {layers} layers", m.layers.len()));
    }
    let per = m.layers.len() / layers;
    if prompt + steps + 1 > o.ctx || prompt > o.chunk {
        return err(format!("a prompt of {prompt} and {steps} steps: a chunk of {} and a cache of {}", o.chunk, o.ctx));
    }
    let mut out: Vec<Line> = Vec::new();
    for l in 0..layers {
        let mut rng = Rng(0x2026_1123_0000_0001 ^ ((l as u64 + 1) << 32) ^ id.bytes().fold(0u64, |h, b| h.wrapping_mul(31).wrapping_add(b as u64)));
        let first = pass(&dy, &mut m, l, per, prompt, 0, &mut rng)?;
        let mut lines: Vec<Line> =
            first.into_iter().map(|(at, kind, class, v, near)| Line { case: id, at, kind, class, prompt: v, steps: 0.0, near }).collect();
        for s in 0..steps {
            let g = pass(&dy, &mut m, l, per, 1, prompt + s, &mut rng)?;
            if g.len() != lines.len() {
                return err(format!("{id} layer {l}: a step compared {} things, the prompt {}", g.len(), lines.len()));
            }
            for (line, (_, _, _, v, near)) in lines.iter_mut().zip(g) {
                line.steps = line.steps.max(v);
                line.near.extend(near);
            }
        }
        for x in &lines {
            let ok = x.prompt <= bound(x.class) && x.steps <= bound(x.class);
            let near = if x.near.is_empty() { String::new() } else { format!("; past the bound near a tie, not compared: {}", x.near.join(", ")) };
            println!(
                "{} {:11} {:22} {}: prompt {:.2e}, steps {:.2e} (bound {:.0e}){near}",
                if ok { "ok  " } else { "FAIL" },
                x.case,
                x.at,
                x.kind,
                x.prompt,
                x.steps,
                bound(x.class)
            );
        }
        out.extend(lines);
    }
    Ok(out)
}

/// `kindcheck [CASE...] [--prompt N] [--steps N] [--scratch DIR]`: the cases (all unless named), then each class's worst against its bound.
pub fn run(a: &crate::args::Args) -> Result<()> {
    const ALL: [&str; 10] = ["gemma4", "qwen35", "qwen35-wide", "qwen35moe", "dense", "nemotron-h", "muse", "kimi", "glm", "deepseek-v4"];
    let want: Vec<&str> = a.pos.iter().skip(1).map(String::as_str).collect();
    if let Some(w) = want.iter().find(|w| !ALL.contains(w)) {
        return err(format!("no case {w}: {}", ALL.join(", ")));
    }
    let ids: Vec<&'static str> = ALL.iter().copied().filter(|c| want.is_empty() || want.contains(c)).collect();
    let (prompt, steps) = (a.num("prompt", 48usize)?, a.num("steps", 4usize)?);
    std::env::set_var("GLYD_V4_SIM", "0");
    let (scratch, own) = match a.flag("scratch") {
        Some(s) => (PathBuf::from(s), false),
        None => (std::env::temp_dir().join(format!("glyd-kindcheck-{}", std::process::id())), true),
    };
    let mut all: Vec<Line> = Vec::new();
    let mut failed: Vec<String> = Vec::new();
    for id in ids {
        match case(id, &scratch.join(id), prompt, steps) {
            Ok(l) => all.extend(l),
            Err(e) => {
                println!("FAIL {id}: {}", e.msg);
                failed.push(format!("{id}: {}", e.msg));
            }
        }
    }
    if own {
        let _ = std::fs::remove_dir_all(&scratch);
    }
    println!("\nthe worst of each kind (the largest difference over the reference's largest magnitude):");
    let mut classes: Vec<Class> = all.iter().map(|x| x.class).collect();
    classes.sort();
    classes.dedup();
    for c in classes {
        let of: Vec<&Line> = all.iter().filter(|x| x.class == c).collect();
        let worst = of.iter().fold(0f32, |m, x| m.max(x.prompt).max(x.steps));
        let near: usize = of.iter().map(|x| x.near.len()).sum();
        let mut cases: Vec<&str> = of.iter().map(|x| x.case).collect();
        cases.dedup();
        let near = if near > 0 { format!(" ({near} rows past the bound near a tie, not compared)") } else { String::new() };
        println!("  {:12} {worst:.2e} (bound {:.0e}) over {} comparisons{near}: {}", format!("{c:?}"), bound(c), of.len(), cases.join(", "));
    }
    failed.extend(all.iter().filter(|x| x.prompt > bound(x.class) || x.steps > bound(x.class)).map(|x| format!("{} {} {}", x.case, x.at, x.kind)));
    if !failed.is_empty() {
        return err(format!("kindcheck: {} not within bounds: {}", failed.len(), failed.join("; ")));
    }
    println!("kindcheck: PASS");
    Ok(())
}
