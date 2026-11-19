//! The lossless KV cache's tables, calibrated on the engine itself (the plugin's `glyd_gpu.kvcal` rule, VERSION 7: src/kvcal.rs has the host's half). Three passes of the model:
//!
//! 1. two sequences of the corpus (every kind of text at every depth) through the layers, the exponents' histograms of each kind's keys and values counted in each layer;
//! 2. the same text again, the values outside the windows that the *other* kinds' histograms make counted per kind and layer (the tables refuse a model whose worst kind leaves more than
//!    0.15% of either tensor outside), and the entries each page of 16 tokens would need beyond its heads' inline ones;
//! 3. every row of the embedding matrix as a one-token sequence: the exceptions the token has in each layer, from which the cache's reserves are sized.
//!
//! The keys and values are what the layer hands the cache: the key after its norm and rotary embedding, the value as it is (`Engine::forward`'s calibration hook).

use crate::chat::{self, Chat};
use crate::dev::Buf;
use crate::ffi::{err, Result};
use crate::kvcal::{self, Json, Seg};
use crate::model::Engine;
use crate::tok::Tok;
use std::time::Instant;

/// The calibration's result: the windows, the sweep's exceptions per token and layer, and the meta.
pub struct Tables {
    pub k: Vec<u8>,
    pub v: Vec<u8>,
    pub exc: Vec<i16>,
    pub rows: usize,
    pub layers: usize,
    pub heads: usize,
    pub channels: usize,
    pub meta: Json,
    pub refused: Option<String>,
}

pub struct Options {
    /// The positions the tables cover (one of kvcal::TIERS).
    pub positions: usize,
    /// The model's name for the meta.
    pub model: String,
}

/// The sequences' length for a table of `positions`: a prompt is at most one token shorter than the model's own window (vLLM's rule, kept for the same tables).
pub fn sequence_length(positions: usize, window: usize) -> usize {
    let mml = if window == 0 || positions + 64 <= window { positions + 64 } else { window };
    positions.min(mml - 1)
}

/// The text the tables are calibrated on (data/kvcorpus.bin: the text zlib-compressed, the file the Python package ships too), inflated once.
pub fn corpus() -> &'static str {
    static TEXT: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    TEXT.get_or_init(|| {
        use std::io::Read;
        let mut s = String::new();
        flate2::read::ZlibDecoder::new(&include_bytes!("../data/kvcorpus.bin")[..]).read_to_string(&mut s).expect("the corpus is inflated");
        s
    })
}

/// The corpus as token ids by document: (kind, name, ids). A chat section goes through the chat template (a model without one: its turns as plain lines).
pub fn corpus_docs(tok: &Tok, chat: Option<&Chat>) -> Result<Vec<(String, String, Vec<u32>)>> {
    let mut out = Vec::new();
    for (kind, name, body) in kvcal::parse_corpus(corpus()) {
        let ids = if kind == "chat" {
            let msgs = chat::parse_messages(&body)?;
            match chat {
                Some(c) => tok.encode_with(&c.render(&msgs, false, None)?, false)?,
                None => tok.encode(&msgs.iter().map(|m| format!("{}: {}", m.role, m.content)).collect::<Vec<_>>().join("\n"))?,
            }
        } else {
            tok.encode(&body)?
        };
        out.push((kind, name, ids));
    }
    Ok(out)
}

fn fnv(text: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in text.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

enum Mode {
    Hist,
    Eval,
    Sweep { row0: usize },
}

/// What the calibration holds on the GPU, and where it is in a pass: the layer hook's state.
struct Cal<'a> {
    e: &'a Engine,
    layers: usize,
    heads: usize,
    kinds: Vec<String>,
    mode: Mode,
    seq: usize,
    segs: Vec<Seg>,
    off: usize,
    // the staging cache of the sequence's keys and values, a layer after another: [layers, stage_tokens, heads, 256]
    stage: Buf,
    stage_tokens: usize,
    // pass 1: [kind][w][layer] counts [C, 128] int32
    hist: Buf,
    // the windows (uint8 [C]) a pass uses: pass 2 the other kinds' ([kind][w][layer]), pass 3 the final ones ([w][layer])
    bases: Buf,
    // pass 2: counts of values outside, [kind][w][layer] u64; the pages' exceptions [seq][layer][page][head] int32
    total: Buf,
    hp: Buf,
    max_pages: usize,
    // the pages pass 2 recorded: (sequence, kind, first page, end page)
    pages_done: Vec<(usize, usize, usize, usize)>,
    // the values counted per kind and layer (values outside / of all)
    numel: Vec<u64>,
    // pass 3: [rows, layers] int32 and int16
    need: Buf,
    exc: Buf,
}

impl Cal<'_> {
    fn kind_index(&self, kind: &str) -> usize {
        self.kinds.iter().position(|k| k == kind).unwrap()
    }

    fn c(&self) -> usize {
        self.heads * 128
    }

    fn hist_at(&self, kind: usize, w: usize, l: usize) -> u64 {
        self.hist.ptr + (((kind * 2 + w) * self.layers + l) * self.c() * 128 * 4) as u64
    }

    fn base_at(&self, kind: usize, w: usize, l: usize) -> u64 {
        self.bases.ptr + (((kind * 2 + w) * self.layers + l) * self.c()) as u64
    }

    fn total_at(&self, kind: usize, w: usize, l: usize) -> u64 {
        self.total.ptr + (((kind * 2 + w) * self.layers + l) * 8) as u64
    }

    fn hp_at(&self, seq: usize, l: usize) -> u64 {
        self.hp.ptr + (((seq * self.layers + l) * self.max_pages * self.heads) * 4) as u64
    }

    /// The layer hook: the keys and values of the layer's `n` tokens (rows `off` ..) are counted as the pass says, and the attention done.
    fn layer(&mut self, e: &Engine, l: usize, n: usize) -> Result<()> {
        let cfg = &e.w.cfg;
        let (heads, kvh, hd) = (cfg.heads, cfg.kv_heads, cfg.head_dim);
        let c = kvh * hd;
        let qkv_w = (heads + 2 * kvh) * hd;
        let s = e.dev.stream;
        let k = e.ko.ptr;
        let v = e.qkv.ptr + ((heads + kvh) * hd * 2) as u64;
        // SAFETY: live buffers of the engine and of this calibration, sized for the chunk; one stream orders every launch.
        unsafe {
            match self.mode {
                Mode::Sweep { .. } => e.kern.attn_self(s, v, qkv_w as i32, e.attn.ptr, (heads * hd) as i32, heads as i32, kvh as i32, n as u32)?,
                _ => {
                    let layer_stage = self.stage.ptr + (l * self.stage_tokens * kvh * 256 * 2) as u64;
                    e.lib.kv_stage_put(&e.kv.geom, k, c as i64, v, qkv_w as i64, e.slots.ptr, n as i64, layer_stage, (self.stage_tokens / 16) as i64, s)?;
                    e.kern.prefill_attn(s, e.qo.ptr, (heads * hd) as i32, layer_stage, e.attn.ptr, (heads * hd) as i32, n as i32, self.off as i32, heads as i32, kvh as i32)?;
                }
            }
            match self.mode {
                Mode::Hist => {
                    for (kind, s0, e0) in self.segs.clone() {
                        let (a, b) = (s0.max(self.off), e0.min(self.off + n));
                        if a >= b {
                            continue;
                        }
                        let ki = self.kind_index(&kind);
                        let r = a - self.off;
                        e.kern.kv_hist(s, k + (r * c * 2) as u64, c as i32, (b - a) as i32, c as i32, self.hist_at(ki, 0, l))?;
                        e.kern.kv_hist(s, v + (r * qkv_w * 2) as u64, qkv_w as i32, (b - a) as i32, c as i32, self.hist_at(ki, 1, l))?;
                    }
                }
                Mode::Eval => {
                    for (kind, s0, e0) in self.segs.clone() {
                        let (a, b) = (s0.max(self.off), e0.min(self.off + n));
                        if a >= b {
                            continue;
                        }
                        let ki = self.kind_index(&kind);
                        let r = a - self.off;
                        let (kp, vp) = (k + (r * c * 2) as u64, v + (r * qkv_w * 2) as u64);
                        e.kern.kv_outside_total(s, kp, c as i32, (b - a) as i32, c as i32, self.base_at(ki, 0, l), self.total_at(ki, 0, l))?;
                        e.kern.kv_outside_total(s, vp, qkv_w as i32, (b - a) as i32, c as i32, self.base_at(ki, 1, l), self.total_at(ki, 1, l))?;
                        self.numel[ki * self.layers + l] += ((b - a) * c) as u64;
                        // the pages of the sequence (16 tokens from its start) that lie in this segment and step
                        let (j0, j1) = (a.div_ceil(kvcal::PT), b / kvcal::PT);
                        if j1 > j0 {
                            let rows = j0 * kvcal::PT - self.off;
                            e.kern.kv_leaf_hp(s, k + (rows * c * 2) as u64, c as i32, v + (rows * qkv_w * 2) as u64, qkv_w as i32, self.base_at(ki, 0, l), self.base_at(ki, 1, l), kvh as i32, (j1 - j0) as u32, j0 as i32, self.hp_at(self.seq, l))?;
                            if l == 0 {
                                self.pages_done.push((self.seq, ki, j0, j1));
                            }
                        }
                    }
                }
                Mode::Sweep { row0 } => {
                    e.kern.kv_sweep(s, k, c as i32, v, qkv_w as i32, self.base_at(0, 0, l), self.base_at(0, 1, l), kvh as i32, l as i32, self.layers as i32, row0 as i32, n as u32, self.need.ptr, self.exc.ptr)?;
                }
            }
        }
        Ok(())
    }
}

/// Runs the model over the chunks of a sequence (or, in the sweep, over the rows of the embedding matrix), the hook counting.
fn run_sequence(cal: &mut Cal, ids: &[u32]) -> Result<()> {
    let e = cal.e;
    let mut off = 0;
    while off < ids.len() {
        let n = e.chunk.min(ids.len() - off);
        let pos: Vec<i32> = (off..off + n).map(|p| p as i32).collect();
        let slots: Vec<i64> = (off..off + n).map(|p| p as i64).collect();
        e.dev.sync()?; // (the arrays are read by the previous chunk's kernels until then)
        e.pos.write(0, &pos)?;
        e.slots.write(0, &slots)?;
        cal.off = off;
        // SAFETY: live buffers of the engine, sized for a chunk; the stream orders every step.
        unsafe {
            e.embed(&ids[off..off + n])?;
            let mut hook = |en: &Engine, l: usize, nn: usize| cal.layer(en, l, nn);
            e.forward(n, e.pos.ptr, e.slots.ptr, 0, 0, 0, false, None, Some(&mut hook))?;
        }
        off += n;
    }
    e.dev.sync()
}

fn read_i32(b: &Buf, n: usize) -> Result<Vec<i32>> {
    let mut v = vec![0i32; n];
    b.read(0, &mut v)?;
    Ok(v)
}

/// Makes a model's tables on `e` (an engine for calibration: weights, activations, no cache), `positions` of them, with the corpus tokenized by `tok`.
pub fn calibrate(e: &Engine, tok: &Tok, chat: Option<&Chat>, o: &Options, log: &dyn Fn(&str)) -> Result<Tables> {
    let t0 = Instant::now();
    let cfg = e.w.cfg.clone();
    let (layers, heads, hd) = (cfg.layers, cfg.kv_heads, cfg.head_dim);
    let c = heads * hd;
    if hd != 128 {
        return err(format!("head size {hd}: not supported"));
    }
    let positions = kvcal::check_positions(o.positions)?;
    let length = sequence_length(positions, cfg.max_pos);
    let docs = corpus_docs(tok, chat)?;
    let seqs = kvcal::sequences(&docs, length);
    let mut kinds: Vec<String> = docs.iter().map(|d| d.0.clone()).collect();
    kinds.sort();
    kinds.dedup();
    let nk = kinds.len();
    let stage_tokens = length.div_ceil(16) * 16;
    log(&if cfg!(feature = "dev") { format!("glyd kvcal: {} documents, sequences of {positions} tokens, {layers} layers", docs.len()) } else { format!("setting up for this model: {layers} layers") });
    let max_pages = stage_tokens / 16;
    let mut cal = Cal {
        e,
        layers,
        heads,
        kinds: kinds.clone(),
        mode: Mode::Hist,
        seq: 0,
        segs: Vec::new(),
        off: 0,
        stage: Buf::new(layers * stage_tokens * heads * 256 * 2)?,
        stage_tokens,
        hist: Buf::zeroed(nk * 2 * layers * c * 128 * 4)?,
        bases: Buf::zeroed(nk * 2 * layers * c)?,
        total: Buf::zeroed(nk * 2 * layers * 8)?,
        hp: Buf::zeroed(2 * layers * max_pages * heads * 4)?,
        max_pages,
        pages_done: Vec::new(),
        numel: vec![0; nk * layers],
        need: Buf::new(4)?,
        exc: Buf::new(2)?,
    };

    // pass 1
    let t = Instant::now();
    for (si, (ids, segs)) in seqs.iter().enumerate() {
        cal.seq = si;
        cal.segs = segs.clone();
        run_sequence(&mut cal, ids)?;
    }
    log(&if cfg!(feature = "dev") { format!("glyd kvcal: pass 1 of 3 ({:.0} s)", t.elapsed().as_secs_f64()) } else { format!("setting up: {:.0} s", t.elapsed().as_secs_f64()) });
    // the windows from the histograms: all kinds' (the tables), and each kind left out
    let mut hist = vec![0i32; nk * 2 * layers * c * 128];
    cal.hist.read(0, &mut hist)?;
    let h_at = |ki: usize, w: usize, l: usize| ((ki * 2 + w) * layers + l) * c * 128;
    let mut base = vec![0u8; 2 * layers * c]; // [w][layer][C]
    let mut loo = vec![0u8; nk * 2 * layers * c]; // [kind][w][layer][C]
    for w in 0..2 {
        for l in 0..layers {
            for ch in 0..c {
                let mut tot = [0i64; 128];
                for ki in 0..nk {
                    for (b, t) in tot.iter_mut().enumerate() {
                        *t += hist[h_at(ki, w, l) + ch * 128 + b] as i64;
                    }
                }
                base[(w * layers + l) * c + ch] = kvcal::best_base(&tot);
                for ki in 0..nk {
                    let mut x = tot;
                    for (b, t) in x.iter_mut().enumerate() {
                        *t -= hist[h_at(ki, w, l) + ch * 128 + b] as i64;
                    }
                    loo[((ki * 2 + w) * layers + l) * c + ch] = kvcal::best_base(&x);
                }
            }
        }
    }
    cal.bases.write(0, &loo)?;
    // pass 2
    let t = Instant::now();
    cal.mode = Mode::Eval;
    for (si, (ids, segs)) in seqs.iter().enumerate() {
        cal.seq = si;
        cal.segs = segs.clone();
        run_sequence(&mut cal, ids)?;
    }
    log(&if cfg!(feature = "dev") { format!("glyd kvcal: pass 2 of 3 ({:.0} s)", t.elapsed().as_secs_f64()) } else { format!("setting up: {:.0} s", t.elapsed().as_secs_f64()) });
    let mut total = vec![0u64; nk * 2 * layers];
    cal.total.read(0, &mut total)?;
    let hp = read_i32(&cal.hp, 2 * layers * max_pages * heads)?;
    // pass 3: every row of the embedding matrix, one token a sequence at position 0, the final windows
    let rows = cfg.vocab;
    let t1 = Instant::now();
    let free_arena = {
        let pb = e.lib.kv_page_bytes(heads as i64, 128, 0)?;
        kvcal::free_arena(pb, heads as i64)
    };
    let capacity = free_arena + kvcal::ND * kvcal::CH;
    cal.bases = Buf::zeroed(2 * layers * c)?; // (kind 0's slots: [w][layer], as base_at(0, w, l) reads them)
    cal.bases.write(0, &base)?;
    cal.need = Buf::zeroed(rows * layers * 4)?;
    cal.exc = Buf::zeroed(rows * layers * 2)?;
    cal.mode = Mode::Sweep { row0: 0 };
    let step = e.chunk.min(rows).max(64) / 64 * 64;
    let mut row0 = 0;
    while row0 < rows {
        let n = step.min(rows - row0);
        cal.mode = Mode::Sweep { row0 };
        e.dev.sync()?;
        e.pos.write(0, &vec![0i32; n])?;
        e.slots.write(0, &vec![0i64; n])?;
        // SAFETY: live buffers of the engine, sized for a chunk; the stream orders every step. Rows of the embedding matrix a multiple of 64 decoded straight into the first activations.
        unsafe {
            if n.is_multiple_of(64) && row0.is_multiple_of(64) {
                e.lib.unpack(&e.w.embed.pack, row0 as i64, n as i64, e.x.ptr, e.dev.stream)?;
            } else {
                e.embed(&(row0 as u32..(row0 + n) as u32).collect::<Vec<u32>>())?;
            }
            let mut hook = |en: &Engine, l: usize, nn: usize| cal.layer(en, l, nn);
            e.forward(n, e.pos.ptr, e.slots.ptr, 0, 0, 0, false, None, Some(&mut hook))?;
        }
        row0 += n;
    }
    e.dev.sync()?;
    let need = read_i32(&cal.need, rows * layers)?;
    let mut exc = vec![0i16; rows * layers];
    cal.exc.read(0, &mut exc)?;
    let (chunks, over, maxneed) = kvcal::sweep_stats(&need, layers, free_arena, capacity);
    log(&if cfg!(feature = "dev") { format!("glyd kvcal: pass 3 of 3 ({:.0} s)", t1.elapsed().as_secs_f64()) } else { format!("setting up: {:.0} s", t0.elapsed().as_secs_f64()) });

    // what the passes counted, as the plugin's meta
    let mut per_kind: Vec<(String, Json)> = Vec::new();
    let (mut worst_k, mut worst_v) = (0f64, 0f64);
    let mut per_tensor = [(0f64, String::new()), (0f64, String::new())];
    let names: Vec<String> = (0..layers).map(|l| format!("model.layers.{l}.self_attn.attn")).collect();
    for (ki, kind) in kinds.iter().enumerate() {
        let (mut ek, mut ev, mut n) = (0u64, 0u64, 0u64);
        for l in 0..layers {
            let (k_, v_, nn) = (total[(ki * 2) * layers + l], total[(ki * 2 + 1) * layers + l], cal.numel[ki * layers + l]);
            ek += k_;
            ev += v_;
            n += nn;
            for (w, x) in [(0usize, k_), (1usize, v_)] {
                let rate = x as f64 / nn.max(1) as f64;
                if rate > per_tensor[w].0 {
                    per_tensor[w] = (rate, format!("{} ({kind})", names[l]));
                }
            }
        }
        let (rk, rv) = (ek as f64 / n.max(1) as f64, ev as f64 / n.max(1) as f64);
        worst_k = worst_k.max(rk);
        worst_v = worst_v.max(rv);
        per_kind.push((kind.clone(), Json::obj(vec![("k".into(), Json::Num(rk)), ("v".into(), Json::Num(rv)), ("values".into(), Json::Num(n as f64))])));
    }
    // the pages' needs: from pass 2's per-page, per-head exceptions of every layer
    let (mut pt, mut pn, mut ppos) = (Vec::new(), Vec::new(), Vec::new());
    for l in 0..layers {
        for &(si, _ki, j0, j1) in &cal.pages_done {
            for j in j0..j1 {
                let row = &hp[((si * layers + l) * max_pages + j) * heads..][..heads];
                pt.push(row.iter().map(|&x| x as i64).sum::<i64>());
                pn.push(row.iter().map(|&x| (x as i64 - kvcal::INLINE).max(0)).sum::<i64>());
                ppos.push(j * kvcal::PT);
            }
        }
    }
    let mean = |x: &[i64]| x.iter().sum::<i64>() as f64 / x.len().max(1) as f64;
    let mut by_depth = Vec::new();
    for lo in (0..positions).step_by(4096) {
        let sel: Vec<i64> = pn.iter().zip(&ppos).filter(|(_, &p)| p >= lo && p < lo + 4096).map(|(&x, _)| x).collect();
        if !sel.is_empty() {
            by_depth.push(Json::obj(vec![
                ("from".into(), Json::Num(lo as f64)),
                ("leaves".into(), Json::Num(sel.len() as f64)),
                ("bay_need_mean".into(), Json::Num((mean(&sel) * 1000.0).round() / 1000.0)),
                ("bay_need_p999".into(), Json::Num(kvcal::quantile_int(&sel, 0.999) as f64)),
                ("bay_need_max".into(), Json::Num(*sel.iter().max().unwrap() as f64)),
            ]));
        }
    }
    let added = tok.added_ids();
    let reach = |i: usize| -> bool {
        if added.contains(&(i as u32)) {
            return false;
        }
        match tok.decode(&[i as u32]).and_then(|s| tok.encode_with(&s, false)) {
            Ok(ids) => ids == [i as u32],
            Err(_) => false,
        }
    };
    let decode = |i: usize| tok.decode(&[i as u32]).unwrap_or_default();
    let mut pool = kvcal::store_meta(&chunks, &over, &maxneed, free_arena, capacity, rows, &reach, &decode);
    pool.push(("seconds".into(), Json::Num((t1.elapsed().as_secs_f64() * 10.0).round() / 10.0)));
    let refused = worst_k.max(worst_v) > kvcal::REFUSED;
    let pool_json = Json::obj(pool);
    let n_i = |x: usize| Json::Num(x as f64);
    let meta = Json::obj(vec![
        ("positions".into(), n_i(positions)),
        ("held_out_rate".into(), Json::obj(vec![("k".into(), Json::Num(worst_k)), ("v".into(), Json::Num(worst_v))])),
        ("per_kind".into(), Json::obj(per_kind)),
        (
            "leaves".into(),
            Json::obj(vec![
                ("n".into(), n_i(pt.len())),
                ("total_mean".into(), Json::Num(mean(&pt))),
                ("total_p99".into(), Json::Num(kvcal::quantile_int(&pt, 0.99) as f64)),
                ("total_max".into(), Json::Num(pt.iter().copied().max().unwrap_or(0) as f64)),
                ("bay_need_p99".into(), Json::Num(kvcal::quantile_int(&pn, 0.99) as f64)),
                ("bay_need_p999".into(), Json::Num(kvcal::quantile_int(&pn, 0.999) as f64)),
                ("bay_need_max".into(), Json::Num(pn.iter().copied().max().unwrap_or(0) as f64)),
            ]),
        ),
        ("by_depth".into(), Json::Arr(by_depth)),
        ("per_tensor".into(), Json::obj(vec![("k".into(), Json::obj(vec![("rate".into(), Json::Num(per_tensor[0].0)), ("worst".into(), Json::Str(per_tensor[0].1.clone()))])), ("v".into(), Json::obj(vec![("rate".into(), Json::Num(per_tensor[1].0)), ("worst".into(), Json::Str(per_tensor[1].1.clone()))]))])),
        ("store".into(), pool_json),
        ("refused".into(), Json::Bool(refused)),
        ("model".into(), Json::Str(o.model.clone())),
        ("layers".into(), n_i(layers)),
        ("heads".into(), n_i(heads)),
        ("channels".into(), n_i(c / heads)),
        ("tokens".into(), n_i(seqs.iter().map(|s| s.0.len()).sum())),
        ("corpus".into(), Json::Str(fnv(corpus()))),
        ("version".into(), n_i(kvcal::VERSION as usize)),
        ("seconds".into(), Json::Num((t0.elapsed().as_secs_f64() * 10.0).round() / 10.0)),
        ("engine".into(), Json::Str(env!("CARGO_PKG_VERSION").to_string())),
    ]);
    let kb: Vec<u8> = base[..layers * c].to_vec();
    let vb: Vec<u8> = base[layers * c..].to_vec();
    log(&if cfg!(feature = "dev") {
        format!(
            "glyd kvcal: {positions} positions; values that compress poorly, worst kind left out: K {:.4}%, V {:.4}%; {rows} tokens checked: {} need the lossless cache's second extra memory{} ({:.0} s)",
            worst_k * 100.0,
            worst_v * 100.0,
            over.iter().filter(|&&o| o > 0).count(),
            if refused { format!("; REFUSED: {}", kvcal::refusal(worst_k, worst_v)) } else { String::new() },
            t0.elapsed().as_secs_f64()
        )
    } else {
        format!("setup finished ({:.0} s)", t0.elapsed().as_secs_f64())
    });
    Ok(Tables { k: kb, v: vb, exc, rows, layers, heads, channels: 128, meta, refused: refused.then(|| kvcal::refusal(worst_k, worst_v)) })
}

/// The tables as the engine's file: kv-tables.safetensors (k, v uint8 [layers, heads, channels]; exc int16 [rows, layers]; the meta and the layers' names in the header).
pub fn save(path: &std::path::Path, t: &Tables) -> Result<()> {
    let exc: Vec<u8> = t.exc.iter().flat_map(|x| x.to_le_bytes()).collect();
    let names = Json::Arr((0..t.layers).map(|l| Json::Str(format!("model.layers.{l}.self_attn.attn"))).collect());
    crate::pack::write_safetensors(
        path,
        &[("format", "glyd-kv-tables-v0".to_string()), ("meta", t.meta.to_string()), ("layers", names.to_string())],
        &[
            ("k", "U8", vec![t.layers as u64, t.heads as u64, t.channels as u64], &t.k[..]),
            ("v", "U8", vec![t.layers as u64, t.heads as u64, t.channels as u64], &t.v[..]),
            ("exc", "I16", vec![t.rows as u64, t.layers as u64], &exc[..]),
        ],
    )
}

/// The calibration's kernels on random data against the host's reading of the rule, and the prompt attention against a plain f64 attention: one line each.
#[cfg(feature = "dev")]
pub fn kernel_check(dev: &crate::dev::Dev) -> Result<()> {
    use crate::kern::Kern;
    let kern = Kern::load(dev)?;
    let s = dev.stream;
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let (t, h, d) = (96usize, 3usize, 128usize);
    let cc = h * d;
    // bf16 bits: exponents from a narrow band, now and then anywhere
    let bits = |rnd: &mut dyn FnMut() -> u64| -> u16 {
        let r = rnd();
        let e = if r.is_multiple_of(8) { (r >> 8) % 256 } else { 120 + (r >> 8) % 9 } as u16;
        (((r >> 20) & 1) as u16) << 15 | e << 7 | ((r >> 24) & 127) as u16
    };
    let k: Vec<u16> = (0..t * cc).map(|_| bits(&mut rnd)).collect();
    let v: Vec<u16> = (0..t * cc).map(|_| bits(&mut rnd)).collect();
    let (kd, vd) = (Buf::new(k.len() * 2)?, Buf::new(v.len() * 2)?);
    kd.write(0, &k)?;
    vd.write(0, &v)?;
    let bin = |x: u16| ((x >> 8) & 127) as usize;
    // the histogram
    let counts = Buf::zeroed(cc * 128 * 4)?;
    // SAFETY: live buffers of the sizes the kernels read and write.
    unsafe { kern.kv_hist(s, kd.ptr, cc as i32, t as i32, cc as i32, counts.ptr)? };
    dev.sync()?;
    let mut got = vec![0i32; cc * 128];
    counts.read(0, &mut got)?;
    let mut want = vec![0i32; cc * 128];
    for r in 0..t {
        for ch in 0..cc {
            want[ch * 128 + bin(k[r * cc + ch])] += 1;
        }
    }
    println!("kv_hist: {}", if got == want { "equal to the host's counts".to_string() } else { "DIFFERENT".to_string() });
    let mut ok = got == want;
    // windows from them, then the values outside
    let base: Vec<u8> = (0..cc).map(|ch| kvcal::best_base(&want[ch * 128..(ch + 1) * 128].iter().map(|&x| x as i64).collect::<Vec<_>>())).collect();
    let (bd, total) = (Buf::new(cc)?, Buf::zeroed(8)?);
    bd.write(0, &base)?;
    // SAFETY: as above.
    unsafe { kern.kv_outside_total(s, kd.ptr, cc as i32, t as i32, cc as i32, bd.ptr, total.ptr)? };
    dev.sync()?;
    let mut tg = [0u64; 1];
    total.read(0, &mut tg)?;
    let outside = |x: u16, b: u8| bin(x) < b as usize || bin(x) >= b as usize + 8;
    let tw: u64 = (0..t * cc).filter(|&i| outside(k[i], base[i % cc])).count() as u64;
    println!("kv_outside_total: {} of {} values outside ({})", tg[0], t * cc, if tg[0] == tw { "equal to the host's" } else { "DIFFERENT" });
    ok &= tg[0] == tw;
    // a page's exceptions per head
    let pages = t / 16;
    let hp = Buf::zeroed(pages * h * 4)?;
    // SAFETY: as above.
    unsafe { kern.kv_leaf_hp(s, kd.ptr, cc as i32, vd.ptr, cc as i32, bd.ptr, bd.ptr, h as i32, pages as u32, 0, hp.ptr)? };
    dev.sync()?;
    let mut hg = vec![0i32; pages * h];
    hp.read(0, &mut hg)?;
    let mut hw = vec![0i32; pages * h];
    for p in 0..pages {
        for hh in 0..h {
            for r in p * 16..p * 16 + 16 {
                for dd in 0..d {
                    let ch = hh * d + dd;
                    hw[p * h + hh] += outside(k[r * cc + ch], base[ch]) as i32 + outside(v[r * cc + ch], base[ch]) as i32;
                }
            }
        }
    }
    println!("kv_leaf_hp: {} pages x {h} heads {}", pages, if hg == hw { "equal to the host's" } else { "DIFFERENT" });
    ok &= hg == hw;
    // the sweep: need and exceptions per row
    let layers = 4usize;
    let (need, exc) = (Buf::zeroed(t * layers * 4)?, Buf::zeroed(t * layers * 2)?);
    // SAFETY: as above.
    unsafe { kern.kv_sweep(s, kd.ptr, cc as i32, vd.ptr, cc as i32, bd.ptr, bd.ptr, h as i32, 2, layers as i32, 0, t as u32, need.ptr, exc.ptr)? };
    dev.sync()?;
    let mut ng = vec![0i32; t * layers];
    let mut eg = vec![0i16; t * layers];
    need.read(0, &mut ng)?;
    exc.read(0, &mut eg)?;
    let mut sw_ok = true;
    for r in 0..t {
        let per_head: Vec<i64> = (0..h).map(|hh| (0..d).map(|dd| (outside(k[r * cc + hh * d + dd], base[hh * d + dd]) as i64) + outside(v[r * cc + hh * d + dd], base[hh * d + dd]) as i64).sum()).collect();
        sw_ok &= ng[r * layers + 2] == kvcal::sweep_need(&per_head) && eg[r * layers + 2] as i64 == per_head.iter().sum::<i64>() && ng[r * layers + 1] == 0;
    }
    println!("kv_sweep: {t} rows {}", if sw_ok { "equal to the host's" } else { "DIFFERENT" });
    ok &= sw_ok;
    // the prompt attention against a plain attention in f64: a chunk of 70 tokens after 33, 8 query heads on 2 key-value heads
    let (n, start, qh, kvh) = (70usize, 33usize, 8usize, 2usize);
    let kv_len = start + n;
    let mut seedf = 7u64;
    let mut gauss = move || {
        seedf ^= seedf << 13;
        seedf ^= seedf >> 7;
        seedf ^= seedf << 17;
        let (a, b) = ((seedf >> 11) as f64 / (1u64 << 53) as f64, ((seedf.wrapping_mul(0x9e3779b97f4a7c15)) >> 11) as f64 / (1u64 << 53) as f64);
        ((-2.0 * (a + 1e-12).ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()) as f32
    };
    let bf = |x: f32| -> u16 { (x.to_bits().wrapping_add(0x7fff + ((x.to_bits() >> 16) & 1)) >> 16) as u16 };
    let tof = |b: u16| f32::from_bits((b as u32) << 16);
    let q: Vec<u16> = (0..n * qh * 128).map(|_| bf(gauss() * 1.5)).collect();
    let kvs: Vec<u16> = (0..kv_len * kvh * 256).map(|i| bf(gauss() * if i % 256 < 128 { 1.5 } else { 1.0 })).collect();
    let (qd, kvd, od) = (Buf::new(q.len() * 2)?, Buf::new(kvs.len() * 2)?, Buf::zeroed(n * qh * 128 * 2)?);
    qd.write(0, &q)?;
    kvd.write(0, &kvs)?;
    // SAFETY: as above.
    unsafe { kern.prefill_attn(s, qd.ptr, (qh * 128) as i32, kvd.ptr, od.ptr, (qh * 128) as i32, n as i32, start as i32, qh as i32, kvh as i32)? };
    dev.sync()?;
    let mut og = vec![0u16; n * qh * 128];
    od.read(0, &mut og)?;
    let (mut se, mut sr, mut mxd) = (0f64, 0f64, 0f64);
    for i in 0..n {
        for hh in 0..qh {
            let hk = hh / (qh / kvh);
            let pos = start + i;
            let qv: Vec<f64> = (0..128).map(|dd| tof(q[i * qh * 128 + hh * 128 + dd]) as f64).collect();
            let sc: Vec<f64> = (0..=pos).map(|j| (0..128).map(|dd| qv[dd] * tof(kvs[(j * kvh + hk) * 256 + dd]) as f64).sum::<f64>() / (128f64).sqrt()).collect();
            let mx = sc.iter().cloned().fold(f64::MIN, f64::max);
            let ps: Vec<f64> = sc.iter().map(|x| (x - mx).exp()).collect();
            let l: f64 = ps.iter().sum();
            for dd in 0..128 {
                let r: f64 = (0..=pos).map(|j| ps[j] * tof(kvs[(j * kvh + hk) * 256 + 128 + dd]) as f64).sum::<f64>() / l;
                let g = tof(og[i * qh * 128 + hh * 128 + dd]) as f64;
                se += (g - r) * (g - r);
                sr += r * r;
                mxd = mxd.max((g - r).abs());
            }
        }
    }
    let rel = (se / sr).sqrt();
    println!("prefill_attn: {n} tokens after {start}: relative L2 error {rel:.2e} against a plain attention in f64, largest difference {mxd:.2e} {}", if rel < 3e-3 { "(within bf16's rounding)" } else { "TOO LARGE" });
    ok &= rel < 3e-3;
    if ok {
        Ok(())
    } else {
        Err(crate::ffi::Error("a calibration kernel check failed".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::KvTables;

    /// Tables as `calibrate` makes them (its meta's keys: the version, the positions, the store's figures), written by `save` and read back: current, and what the cache's sizing reads is there.
    #[test]
    fn what_calibrate_writes_is_what_the_cache_reads() {
        let (layers, heads) = (2usize, 8usize);
        let store = kvcal::store_meta(&[1.0, 2.5, 0.5], &[0, 0, 1], &[10, 20, 5000], 15, 15 + kvcal::ND * kvcal::CH, 3, &|_| true, &|i| format!("<{i}>"));
        let meta = Json::obj(vec![
            ("positions".into(), Json::Num(4096.0)),
            ("model".into(), Json::Str("m".into())),
            ("store".into(), Json::obj(store)),
            ("version".into(), Json::Num(kvcal::VERSION as f64)),
        ]);
        let t = Tables { k: vec![3; layers * heads * 128], v: vec![4; layers * heads * 128], exc: vec![0; 3 * layers], rows: 3, layers, heads, channels: 128, meta, refused: None };
        let dir = std::env::temp_dir().join(format!("glyd-engine-calib-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("kv-tables-4096.safetensors");
        save(&p, &t).unwrap();
        assert!(KvTables::is_current(&p));
        let r = KvTables::load(&p).unwrap();
        assert_eq!((r.layers, r.heads, r.positions, r.arena, r.heavy_tokens), (layers, heads, 4096, 15, 1));
        assert_eq!(r.worst_chunks, Some(2.5));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
