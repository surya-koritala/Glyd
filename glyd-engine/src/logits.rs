//! Per-model logits tests (R7): each model the engine runs has a fixture of reference logits on fixed tokens, and `logitscheck` compares the engine's logits with it on every merge (gpu-tests.yml).
//!
//! A fixture (`tests/logits/CASE.json.gz`, format glyd-logits/1) holds the fixed tokens (a prompt, decode steps fed one at a time, and a longer prompt that crosses the chunk size), and for the
//! last position of each pass a row: the 64 largest logits with their token ids, the log of the sum of exponentials over the vocabulary, and a hash of every logit's bits. It says where the
//! numbers came from (`provenance`: the CPU reference of src/*_ref.rs, transformers on a named GPU, or a release's own engine where no reference exists) and, for each level, the bound the
//! engine must stay within:
//!   bits: the row's hash is the reference's (every logit the same to the bit: where the code promises it, the engine against itself or against a CPU reference of the same arithmetic);
//!   kl:   the KL divergence of the reference's softmax from the engine's, over the reference's 64 largest tokens and the tail as one bin, at most `max` and `mean` on average, and the engine's
//!         best token the reference's at every position where the reference's lead over its second is past 0.1 (a lead within the roundings is a coin's toss), `top1` of them at least.
//!
//! `logitscheck [CASE...] [--fixtures DIR] [--cpu-ref]` runs the engine (or, with --cpu-ref, the CPU reference, to see the fixture is still what the reference makes), `logitsrecord CASE --from cpu|engine`
//! writes a fixture, tools/logits_ref_hf.py writes the ones whose reference is transformers. A development build's only.

use crate::dynamic::{Dyn, Opts as DynOpts};
use crate::ffi::{err, Error, Result};
use crate::weights::Layout;
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub const FORMAT: &str = "glyd-logits/1";
/// The largest logits a row keeps.
pub const TOPK: usize = 64;
/// A reference's lead of its best logit over its second that counts for the greedy-token check (the bf16 roundings of two near ties decide those).
pub const CLEAR_LEAD: f32 = 0.1;

fn io<T>(r: std::io::Result<T>) -> Result<T> {
    r.map_err(|e| Error(e.to_string()))
}

/// One pass's last position: the largest logits, the log-sum-exp of the whole row, and a hash of its bits.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub ids: Vec<u32>,
    pub logits: Vec<f32>,
    pub lse: f64,
    pub hash: u64,
}

fn lse_of(l: &[f32]) -> f64 {
    let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    m + l.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
}

/// FNV-1a over the bits of every logit.
fn hash_of(l: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for x in l {
        for b in x.to_bits().to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

impl Row {
    /// The row of a full logit vector: its 64 largest (a tie by the smaller token), the log-sum-exp, the hash. A logit that is not a number is an error.
    pub fn from_logits(l: &[f32]) -> Result<Row> {
        if let Some(i) = l.iter().position(|x| !x.is_finite()) {
            return err(format!("logit {i} is {}: not a finite number", l[i]));
        }
        let k = TOPK.min(l.len());
        let mut idx: Vec<u32> = (0..l.len() as u32).collect();
        let order = |a: &u32, b: &u32| l[*b as usize].total_cmp(&l[*a as usize]).then(a.cmp(b));
        if k < idx.len() {
            idx.select_nth_unstable_by(k - 1, order);
            idx.truncate(k);
        }
        idx.sort_unstable_by(order);
        Ok(Row { logits: idx.iter().map(|&i| l[i as usize]).collect(), ids: idx, lse: lse_of(l), hash: hash_of(l) })
    }
}

/// What the engine's logits at a level must be within.
#[derive(Clone, Debug, PartialEq)]
pub enum Bound {
    Bits,
    Kl { mean: f64, max: f64, top1: f64 },
}

#[derive(Clone, Debug)]
pub struct LevelBound {
    pub level: String,
    pub bound: Bound,
}

#[derive(Clone, Debug)]
pub struct Fixture {
    pub case: String,
    pub family: String,
    pub model: String,
    pub provenance: BTreeMap<String, String>,
    /// The GPU generation whose bits these are ("sm89"), where the bound is `bits` of an engine's own output: another generation's kernels differ in the last bits, so it skips them.
    pub gpu: Option<String>,
    pub bounds: Vec<LevelBound>,
    pub prompt: Vec<u32>,
    pub decode: Vec<u32>,
    pub long: Vec<u32>,
    /// The prompt's row, each decode step's, then the long prompt's (where there is one).
    pub rows: Vec<Row>,
}

impl Fixture {
    pub fn expected_rows(&self) -> usize {
        1 + self.decode.len() + (!self.long.is_empty()) as usize
    }
}

use serde_json::{json, Value};

fn row_json(r: &Row) -> Value {
    // (an f32 prints as its shortest text and reads back to the same bits)
    json!({"ids": r.ids, "logits": r.logits.iter().map(|&x| serde_json::Number::from_f64(x as f64).map_or(Value::Null, Value::Number)).collect::<Vec<_>>(), "lse": r.lse, "hash": format!("{:016x}", r.hash)})
}

fn bound_json(b: &LevelBound) -> Value {
    match &b.bound {
        Bound::Bits => json!({"level": b.level, "kind": "bits"}),
        Bound::Kl { mean, max, top1 } => json!({"level": b.level, "kind": "kl", "mean": mean, "max": max, "top1": top1}),
    }
}

impl Fixture {
    pub fn to_json(&self) -> Value {
        json!({
            "format": FORMAT, "case": self.case, "family": self.family, "model": self.model, "provenance": self.provenance, "gpu": self.gpu,
            "bounds": self.bounds.iter().map(bound_json).collect::<Vec<_>>(),
            "prompt": self.prompt, "decode": self.decode, "long": self.long,
            "rows": self.rows.iter().map(row_json).collect::<Vec<_>>(),
        })
    }

    pub fn from_json(v: &Value) -> Result<Fixture> {
        let bad = |what: &str| Error(format!("a logits fixture: {what}"));
        if v["format"] != FORMAT {
            return err(format!("a logits fixture of format {}, where {FORMAT} is read", v["format"]));
        }
        let s = |k: &str| v[k].as_str().map(str::to_string).ok_or_else(|| bad(k));
        let toks = |k: &str| -> Result<Vec<u32>> {
            v[k].as_array().ok_or_else(|| bad(k))?.iter().map(|x| x.as_u64().map(|n| n as u32).ok_or_else(|| bad(k))).collect()
        };
        let provenance = v["provenance"]
            .as_object()
            .ok_or_else(|| bad("provenance"))?
            .iter()
            .map(|(k, x)| (k.clone(), x.as_str().unwrap_or_default().to_string()))
            .collect();
        let bounds = v["bounds"]
            .as_array()
            .ok_or_else(|| bad("bounds"))?
            .iter()
            .map(|b| {
                let level = b["level"].as_str().ok_or_else(|| bad("bound level"))?.to_string();
                let f = |k: &str| b[k].as_f64().ok_or_else(|| bad(k));
                let bound = match b["kind"].as_str() {
                    Some("bits") => Bound::Bits,
                    Some("kl") => Bound::Kl { mean: f("mean")?, max: f("max")?, top1: f("top1")? },
                    _ => return Err(bad("bound kind")),
                };
                Ok(LevelBound { level, bound })
            })
            .collect::<Result<Vec<_>>>()?;
        let rows = v["rows"]
            .as_array()
            .ok_or_else(|| bad("rows"))?
            .iter()
            .map(|r| {
                let ids = r["ids"]
                    .as_array()
                    .ok_or_else(|| bad("row ids"))?
                    .iter()
                    .map(|x| x.as_u64().map(|n| n as u32).ok_or_else(|| bad("row id")))
                    .collect::<Result<Vec<_>>>()?;
                let logits = r["logits"]
                    .as_array()
                    .ok_or_else(|| bad("row logits"))?
                    .iter()
                    .map(|x| x.as_f64().map(|n| n as f32).ok_or_else(|| bad("row logit")))
                    .collect::<Result<Vec<_>>>()?;
                let hash = u64::from_str_radix(r["hash"].as_str().ok_or_else(|| bad("row hash"))?, 16).map_err(|_| bad("row hash"))?;
                if ids.len() != logits.len() || ids.is_empty() {
                    return Err(bad("a row's ids and logits differ in count"));
                }
                Ok(Row { ids, logits, lse: r["lse"].as_f64().ok_or_else(|| bad("row lse"))?, hash })
            })
            .collect::<Result<Vec<_>>>()?;
        let gpu = v["gpu"].as_str().map(str::to_string);
        let f = Fixture {
            case: s("case")?,
            family: s("family")?,
            model: s("model")?,
            provenance,
            gpu,
            bounds,
            prompt: toks("prompt")?,
            decode: toks("decode")?,
            long: toks("long")?,
            rows,
        };
        if f.rows.len() != f.expected_rows() {
            return err(format!(
                "{}: {} rows, where the prompt, {} steps and the long prompt make {}",
                f.case,
                f.rows.len(),
                f.decode.len(),
                f.expected_rows()
            ));
        }
        Ok(f)
    }

    pub fn read(path: &Path) -> Result<Fixture> {
        let raw = io(std::fs::read(path)).map_err(|e| Error(format!("{}: {}", path.display(), e.0)))?;
        let text = if raw.starts_with(&[0x1f, 0x8b]) {
            let mut s = String::new();
            io(flate2::read::GzDecoder::new(&raw[..]).read_to_string(&mut s))?;
            s
        } else {
            String::from_utf8(raw).map_err(|e| Error(e.to_string()))?
        };
        let v: Value = serde_json::from_str(&text).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        Fixture::from_json(&v).map_err(|e| Error(format!("{}: {}", path.display(), e.0)))
    }

    /// Written compressed (and with a name that ends .gz), one line.
    pub fn write(&self, path: &Path) -> Result<()> {
        let text = serde_json::to_string(&self.to_json()).map_err(|e| Error(e.to_string()))?;
        let mut z = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::best());
        io(z.write_all(text.as_bytes()))?;
        let bytes = io(z.finish())?;
        if let Some(d) = path.parent() {
            io(std::fs::create_dir_all(d))?;
        }
        io(std::fs::write(path, bytes))
    }
}

/// The result of one level's comparison.
#[derive(Debug, Default)]
pub struct Verdict {
    pub ok: bool,
    pub notes: Vec<String>,
    pub mean_kl: f64,
    pub max_kl: f64,
    /// (positions where the engine's best token is the reference's, positions with a clear best token)
    pub top1: (usize, usize),
    /// The largest difference of a logit among the reference's largest, and the rows whose every logit is the reference's to the bit.
    pub max_dlogit: f32,
    pub identical: usize,
    pub rows: usize,
}

/// KL(p || q) of the reference's softmax from the engine's, over the reference's largest tokens and the rest as one bin.
pub fn kl_of(want: &Row, got: &[f32]) -> f64 {
    let lg = lse_of(got);
    let (mut kl, mut p_sum, mut q_sum) = (0f64, 0f64, 0f64);
    for (&id, &l) in want.ids.iter().zip(&want.logits) {
        let lp = l as f64 - want.lse;
        let lq = got[id as usize] as f64 - lg;
        let p = lp.exp();
        kl += p * (lp - lq);
        p_sum += p;
        q_sum += lq.exp();
    }
    let (pt, qt) = ((1.0 - p_sum).max(0.0), (1.0 - q_sum).max(0.0));
    if pt > 1e-9 {
        kl += pt * (pt / qt.max(1e-300)).ln();
    }
    kl.max(0.0)
}

/// The engine's rows (full logit vectors, in the fixture's order) against the fixture's, within `bound`.
pub fn compare(fx: &Fixture, bound: &Bound, got: &[Vec<f32>]) -> Verdict {
    let mut v = Verdict { rows: fx.rows.len(), ..Default::default() };
    if got.len() != fx.rows.len() {
        v.notes.push(format!("{} rows from the engine, the fixture has {}", got.len(), fx.rows.len()));
        return v;
    }
    let (mut sum, mut worst_at) = (0f64, 0usize);
    for (i, (want, g)) in fx.rows.iter().zip(got).enumerate() {
        let row = match Row::from_logits(g) {
            Ok(r) => r,
            Err(e) => {
                v.notes.push(format!("row {i}: {}", e.0));
                return v;
            }
        };
        if row.hash == want.hash {
            v.identical += 1;
        }
        for (&id, &l) in want.ids.iter().zip(&want.logits) {
            v.max_dlogit = v.max_dlogit.max((g[id as usize] - l).abs());
        }
        let kl = kl_of(want, g);
        sum += kl;
        if kl > v.max_kl {
            v.max_kl = kl;
            worst_at = i;
        }
        if want.logits.len() > 1 && want.logits[0] - want.logits[1] > CLEAR_LEAD {
            v.top1.1 += 1;
            let agree = row.ids[0] == want.ids[0];
            v.top1.0 += agree as usize;
            if !agree {
                v.notes.push(format!(
                    "row {i}: the engine's best token is {} ({:.4}), the reference's {} ({:.4})",
                    row.ids[0], row.logits[0], want.ids[0], want.logits[0]
                ));
            }
        }
        if let Bound::Bits = bound {
            if row.hash != want.hash {
                v.notes.push(format!(
                    "row {i}: the logits are not the reference's to the bit (largest difference among its top {}: {:.5}, KL {kl:.3e})",
                    want.ids.len(),
                    (0..want.ids.len()).map(|k| (g[want.ids[k] as usize] - want.logits[k]).abs()).fold(0f32, f32::max)
                ));
            }
        }
    }
    v.mean_kl = sum / v.rows.max(1) as f64;
    v.ok = match bound {
        Bound::Bits => v.identical == v.rows,
        Bound::Kl { mean, max, top1 } => {
            if v.max_kl > *max {
                v.notes.push(format!("row {worst_at}: KL {:.3e} is past the bound {max:.3e}", v.max_kl));
            }
            if v.mean_kl > *mean {
                v.notes.push(format!("mean KL {:.3e} is past the bound {mean:.3e}", v.mean_kl));
            }
            let need = (v.top1.1 as f64 * top1).ceil() as usize;
            if v.top1.0 < need {
                v.notes.push(format!("the best token agrees at {} of {} positions with a clear lead, {need} are needed", v.top1.0, v.top1.1));
            }
            v.max_kl <= *max && v.mean_kl <= *mean && v.top1.0 >= need
        }
    };
    v
}

/// The levels' names, as the CLI's: the layout they pack in (None: the lossless one the GPU runs best).
pub fn level_layout(level: &str) -> Result<Option<Layout>> {
    match level {
        "penguin" => Ok(None),
        "kestrel" => Ok(Some(Layout::Int6)),
        "swift" => Ok(Some(Layout::Int5)),
        "wren" => Ok(Some(Layout::Int4)),
        other => err(format!("level {other}: penguin, kestrel, swift or wren")),
    }
}

// ---- what runs the tokens

/// An engine, or a CPU reference, fed tokens: one pass's last logits.
pub trait Run {
    fn describe(&self) -> String;
    fn reset(&mut self) -> Result<()>;
    fn prefill(&mut self, tokens: &[u32]) -> Result<()>;
    fn step(&mut self, token: u32) -> Result<()>;
    fn logits(&mut self) -> Result<Vec<f32>>;
}

impl Run for Dyn {
    fn describe(&self) -> String {
        Dyn::describe(self)
    }
    fn reset(&mut self) -> Result<()> {
        Dyn::reset(self)
    }
    fn prefill(&mut self, t: &[u32]) -> Result<()> {
        Dyn::prefill(self, t).map(|_| ())
    }
    fn step(&mut self, t: u32) -> Result<()> {
        Dyn::step(self, t).map(|_| ())
    }
    fn logits(&mut self) -> Result<Vec<f32>> {
        Dyn::logits(self)
    }
}

/// Qwen3 on the hand-written dense engine.
struct Dense(crate::model::Engine);

impl Run for Dense {
    fn describe(&self) -> String {
        self.0.describe()
    }
    fn reset(&mut self) -> Result<()> {
        self.0.reset();
        Ok(())
    }
    fn prefill(&mut self, t: &[u32]) -> Result<()> {
        self.0.prefill(t, false).map(|_| ())
    }
    fn step(&mut self, t: u32) -> Result<()> {
        self.0.step(t, false).map(|_| ())
    }
    fn logits(&mut self) -> Result<Vec<f32>> {
        self.0.read_logits()
    }
}

/// A CPU reference one token at a time: its `step` is every pass's way in, and a prompt is its tokens one after another.
struct CpuRef<F: FnMut(u32) -> Vec<f32>> {
    load: Box<dyn Fn() -> Result<F>>,
    m: Option<F>,
    last: Vec<f32>,
}

impl<F: FnMut(u32) -> Vec<f32>> Run for CpuRef<F> {
    fn describe(&self) -> String {
        "the CPU reference".into()
    }
    fn reset(&mut self) -> Result<()> {
        self.m = Some((self.load)()?);
        Ok(())
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        for &t in tokens {
            self.step(t)?;
        }
        Ok(())
    }
    fn step(&mut self, t: u32) -> Result<()> {
        if self.m.is_none() {
            self.reset()?;
        }
        self.last = (self.m.as_mut().unwrap())(t);
        Ok(())
    }
    fn logits(&mut self) -> Result<Vec<f32>> {
        Ok(self.last.clone())
    }
}

/// The prompt, each decode token, then (after a reset) the long prompt: every pass's last logits.
pub fn drive(run: &mut dyn Run, prompt: &[u32], decode: &[u32], long: &[u32]) -> Result<Vec<Vec<f32>>> {
    let mut rows = Vec::new();
    run.reset()?;
    run.prefill(prompt)?;
    rows.push(run.logits()?);
    for &t in decode {
        run.step(t)?;
        rows.push(run.logits()?);
    }
    if !long.is_empty() {
        run.reset()?;
        run.prefill(long)?;
        rows.push(run.logits()?);
    }
    Ok(rows)
}

// ---- the cases

/// Where a case's weights are from.
pub enum Weights {
    /// A Hub repo, read from the cache or downloaded into it (a recent release that fits a 16 GB card: none of the cases is one yet, as the models released in the last six months are all larger).
    #[allow(dead_code)]
    Hub(&'static str),
    /// A small model of the family's layout made at random (a fixed seed) by this program.
    Random(fn(&Path) -> Result<()>),
}

pub struct Case {
    pub id: &'static str,
    pub family: &'static str,
    pub weights: Weights,
    /// The engine at a level, on the model in the directory.
    pub engine: fn(&Path, &str) -> Result<Box<dyn Run>>,
    /// The family's CPU reference on the same directory, where it has one.
    pub cpu: Option<fn(&Path) -> Result<Box<dyn Run>>>,
    /// The vocabulary of a random model (its fixed tokens are made from it); a Hub model's tokens come from the reference tool.
    pub vocab: usize,
}

/// The fixed list the checks of src/*_ref.rs feed (a first token, then 37 i + 11 modulo the vocabulary).
pub fn token_list(n: usize, vocab: usize, offset: usize) -> Vec<u32> {
    (0..n).map(|i| if i + offset == 0 { 2 } else { (((i + offset - 1) as u32) * 37 + 11) % vocab as u32 }).collect()
}

fn standard_tokens(vocab: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    // (a prompt of 24, then 12 steps; a long prompt of 150 crosses the chunk of 64 twice)
    (token_list(24, vocab, 0), token_list(12, vocab, 24), token_list(150, vocab, 0))
}

fn dyn_opts(layout: Option<Layout>) -> DynOpts {
    DynOpts { ctx: 1024, chunk: 64, graph: true, device: 0, layout }
}

fn boxed<T: Run + 'static>(r: Result<T>) -> Result<Box<dyn Run>> {
    r.map(|x| Box::new(x) as Box<dyn Run>)
}

fn muse_engine(dir: &Path, level: &str) -> Result<Box<dyn Run>> {
    boxed(crate::dynamic::muse::load(dir, &dyn_opts(level_layout(level)?)))
}

fn muse_cpu(dir: &Path) -> Result<Box<dyn Run>> {
    let d = dir.to_path_buf();
    Ok(Box::new(CpuRef { load: Box::new(move || crate::muse_ref::RefMuse::load(&d).map(|mut m| move |t| m.step(t))), m: None, last: Vec::new() }))
}

fn kimi_engine(dir: &Path, level: &str) -> Result<Box<dyn Run>> {
    boxed(crate::dynamic::kimi::load(dir, &dyn_opts(level_layout(level)?)))
}

fn kimi_cpu(dir: &Path) -> Result<Box<dyn Run>> {
    let d = dir.to_path_buf();
    Ok(Box::new(CpuRef { load: Box::new(move || crate::kimi_ref::RefKimi::load(&d).map(|mut m| move |t| m.step(t))), m: None, last: Vec::new() }))
}

fn dsv4_engine(dir: &Path, level: &str) -> Result<Box<dyn Run>> {
    boxed(crate::dynamic::deepseek_v4::load(dir, &dyn_opts(level_layout(level)?)))
}

fn dsv4_cpu(dir: &Path) -> Result<Box<dyn Run>> {
    let d = dir.to_path_buf();
    Ok(Box::new(CpuRef { load: Box::new(move || crate::dsv4_ref::RefV4::load(&d, true).map(|mut m| move |t| m.step(t))), m: None, last: Vec::new() }))
}

fn nemotron_engine(dir: &Path, level: &str) -> Result<Box<dyn Run>> {
    let layout = level_layout(level)?;
    boxed(crate::dynamic::nemotron::load(dir, &dyn_opts(layout), layout.unwrap_or(Layout::Tiered)))
}

/// GLM-5.3-Flash has no lossless level (its experts are FP8 in the release): its levels are the lossy ones, the experts at six, five or four bits.
fn glm_engine(dir: &Path, level: &str) -> Result<Box<dyn Run>> {
    let Some(experts) = level_layout(level)? else { return err("GLM-5.3-Flash runs at kestrel, swift or wren") };
    boxed(crate::dynamic::glm::load(dir, &dyn_opts(Some(Layout::Tiered)), experts))
}

/// Gemma 4 on the hand-written dense engine.
/// Gemma 4 on the description executor, as `serve` runs it (no draft model).
fn gemma4_engine(dir: &Path, level: &str) -> Result<Box<dyn Run>> {
    let c = crate::gemma4::ConfigG4::load(dir)?;
    let o = DynOpts { ctx: 1024, chunk: 64, graph: false, device: 0, layout: level_layout(level)? };
    boxed(Dyn::new(dir, crate::dynamic::gemma4::describe(&c), None, &o))
}

/// A small Gemma 4 of the 12B's layer kinds (sliding layers, and a global one whose keys are its values), the heads the engine runs (256 and 512), a soft-capped tied head, at random (a fixed seed).
pub fn gemma4_random(out: &Path) -> Result<()> {
    let text = r#"{"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4", "eos_token_id": [1, 106],
        "text_config": {"model_type": "gemma4_text", "hidden_size": 512, "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2, "num_global_key_value_heads": 1,
        "attention_k_eq_v": true, "head_dim": 256, "global_head_dim": 512, "intermediate_size": 1024, "vocab_size": 4096, "rms_norm_eps": 1e-6, "sliding_window": 32,
        "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"], "final_logit_softcapping": 30.0, "max_position_embeddings": 4096,
        "enable_moe_block": false, "hidden_size_per_layer_input": 0, "hidden_activation": "gelu_pytorch_tanh",
        "rope_parameters": {"sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}, "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25, "rope_type": "proportional"}}}}"#;
    let c = crate::gemma4::ConfigG4::parse(text, None, false)?;
    let mut r = 0x2026_1120_0000_0004u64;
    let mut uni = move |a: f32, b: f32| {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        a + (b - a) * ((r >> 40) as f32 / (1u64 << 24) as f32)
    };
    let bf = |x: f32| {
        let b = x.to_bits();
        ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
    };
    let mut t: Vec<(String, Vec<u64>, Vec<u8>)> = Vec::new();
    let mut put = |name: String, shape: Vec<u64>, lo: f32, hi: f32| {
        let n: u64 = shape.iter().product();
        let bytes: Vec<u8> = (0..n).flat_map(|_| bf(uni(lo, hi)).to_le_bytes()).collect();
        t.push((name, shape, bytes));
    };
    let a = |cols: usize| (3.0 / cols as f32).sqrt();
    let (h, p) = (c.hidden, c.prefix.clone());
    put(format!("{p}embed_tokens.weight"), vec![c.vocab as u64, h as u64], -1.0, 1.0);
    put(format!("{p}norm.weight"), vec![h as u64], 0.5, 1.5);
    for (l, &k) in c.kinds.iter().enumerate() {
        let (d, _, _) = c.attn_of(k);
        let pl = format!("{p}layers.{l}");
        for n in ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"] {
            put(format!("{pl}.{n}.weight"), vec![h as u64], 0.5, 1.5);
        }
        put(format!("{pl}.self_attn.q_norm.weight"), vec![d as u64], 0.5, 1.5);
        put(format!("{pl}.self_attn.k_norm.weight"), vec![d as u64], 0.5, 1.5);
        put(format!("{pl}.layer_scalar"), vec![1], 0.5, 1.0);
        let (_, nk, has_v) = c.attn_of(k);
        put(format!("{pl}.self_attn.q_proj.weight"), vec![(c.heads * d) as u64, h as u64], -a(h), a(h));
        put(format!("{pl}.self_attn.k_proj.weight"), vec![(nk * d) as u64, h as u64], -a(h), a(h));
        if has_v {
            put(format!("{pl}.self_attn.v_proj.weight"), vec![(nk * d) as u64, h as u64], -a(h), a(h));
        }
        put(format!("{pl}.self_attn.o_proj.weight"), vec![h as u64, (c.heads * d) as u64], -a(c.heads * d), a(c.heads * d));
        put(format!("{pl}.mlp.gate_proj.weight"), vec![c.inter as u64, h as u64], -a(h), a(h));
        put(format!("{pl}.mlp.up_proj.weight"), vec![c.inter as u64, h as u64], -a(h), a(h));
        put(format!("{pl}.mlp.down_proj.weight"), vec![h as u64, c.inter as u64], -a(c.inter), a(c.inter));
    }
    io(std::fs::create_dir_all(out))?;
    io(std::fs::write(out.join("config.json"), text))?;
    let refs: Vec<(&str, &str, Vec<u64>, &[u8])> = t.iter().map(|(n, s, b)| (n.as_str(), "BF16", s.clone(), &b[..])).collect();
    crate::pack::write_safetensors(&out.join("model.safetensors"), &[("format", "pt".to_string())], &refs)?;
    println!("{}: a random Gemma 4, {} tensors, {:.0} MB", out.display(), t.len(), t.iter().map(|t| t.2.len()).sum::<usize>() as f64 / 1e6);
    Ok(())
}

/// Qwen3.8-Flash-Next on its own engine: a prompt in chunks (the last chunk's last row through the head), the logits of a step.
struct Qwen4x {
    e: crate::model_qwen4x::EngineQx,
    chunk: usize,
    last_chunk: usize,
}

impl Run for Qwen4x {
    fn describe(&self) -> String {
        self.e.describe()
    }
    fn reset(&mut self) -> Result<()> {
        self.last_chunk = 0;
        self.e.reset()
    }
    fn prefill(&mut self, t: &[u32]) -> Result<()> {
        for c in t.chunks(self.chunk) {
            self.e.prefill(c)?;
            self.last_chunk = c.len();
        }
        Ok(())
    }
    fn step(&mut self, t: u32) -> Result<()> {
        self.last_chunk = 0;
        self.e.step(t).map(|_| ())
    }
    fn logits(&mut self) -> Result<Vec<f32>> {
        if self.last_chunk == 0 {
            return self.e.logits();
        }
        let h = self.e.final_rows(self.last_chunk)?;
        let rows = self.e.head_rows(&h)?;
        let v = rows.len() / self.last_chunk;
        Ok(rows[rows.len() - v..].iter().map(|&w| f32::from_bits((w as u32) << 16)).collect())
    }
}

/// The experts of Qwen3.8-Flash-Next are held in six-bit codes at its kestrel level (the random model's are on that grid, as tools/qwen4x_ref.py makes them).
fn qwen4x_engine(dir: &Path, level: &str) -> Result<Box<dyn Run>> {
    let Some(experts) = level_layout(level)? else { return err("the tiny Qwen3.8-Flash-Next runs at kestrel, swift or wren") };
    let o = crate::model_qwen4x::OptsQx { ctx: 1024, chunk: 64, layout: Layout::Tiered, chunked: true, ..Default::default() };
    let make = crate::model_qwen4x::gpu_experts(experts);
    Ok(Box::new(Qwen4x { e: crate::model_qwen4x::EngineQx::new(dir, &o, Some(&make))?, chunk: 64, last_chunk: 0 }))
}

/// The random tiny Qwen3.8-Flash-Next of tools/qwen4x_ref.py (8 layers: three Gated DeltaNet and one indexed attention, twice; a layer with per-layer embeddings; 16 experts; hyper-connections): its tensors'
/// names, types and shapes from tests/fixtures/qwen4x_tiny_manifest.json (what that tool wrote), their values at random from a fixed seed by the tool's rules for each kind of tensor.
pub fn qwen4x_random(out: &Path) -> Result<()> {
    manifest_random(include_str!("../tests/fixtures/qwen4x_tiny_manifest.json"), out, 0x2026_1120_0000_0005, false, "Qwen3.8-Flash-Next of the tiny shape")
}

/// A small Qwen3 dense model of Qwen3-0.6B's layout (4 layers, 16 query and 8 key-value heads of 128, a vocabulary of 4096): the architecture of the dense engine, not a release (none of the Qwen3 dense releases is recent).
pub fn qwen3_random(out: &Path) -> Result<()> {
    manifest_random(include_str!("../tests/fixtures/qwen3_tiny_manifest.json"), out, 0x2026_1120_0000_0006, true, "Qwen3 dense")
}

/// A small Qwen3.5-family text model (Qwen3_5ForCausalLM: three Gated DeltaNet layers and a gated full-attention one, 1024 wide, a vocabulary of 4096): the layout of Qwen3.8-27B and the releases of its family at 0.8B's widths.
pub fn qwen35_random(out: &Path) -> Result<()> {
    manifest_random(include_str!("../tests/fixtures/qwen35_tiny_manifest.json"), out, 0x2026_1120_0000_0007, false, "Qwen3.5-family")
}

/// A random model of a manifest's tensors (names, types and shapes from a real checkpoint's header, cut down), their values from a fixed seed by the rules of tools/qwen4x_ref.py for each kind of tensor.
/// `dense_norms`: the norms' weights are the scales themselves (0.5 to 1.5) and not the (1 + w) of the other families (-0.5 to 0.5).
fn manifest_random(manifest: &str, out: &Path, seed: u64, dense_norms: bool, what: &str) -> Result<()> {
    let m: Value = serde_json::from_str(manifest).map_err(|e| Error(e.to_string()))?;
    let mut r = seed;
    let mut u01 = move || {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        ((r >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut gauss = {
        let mut spare: Option<f64> = None;
        move |u: &mut dyn FnMut() -> f64| -> f64 {
            if let Some(x) = spare.take() {
                return x;
            }
            let (a, b) = (u(), u());
            let rad = (-2.0 * a.ln()).sqrt();
            spare = Some(rad * (std::f64::consts::TAU * b).sin());
            rad * (std::f64::consts::TAU * b).cos()
        }
    };
    let bf = |x: f32| -> u16 {
        let b = x.to_bits();
        ((b.wrapping_add(0x7fff + ((b >> 16) & 1))) >> 16) as u16
    };
    // (rows x cols weights on the six-bit grid, a code of 31 in every run of 32 along a row: the tool's `grid`)
    let grid = |rows: usize, cols: usize, u: &mut dyn FnMut() -> f64, g: &mut dyn FnMut(&mut dyn FnMut() -> f64) -> f64| -> Vec<f32> {
        let std = 1.0 / (cols as f64).sqrt();
        let scale = 2f64.powf((3.0 * std / 31.0).log2().ceil());
        let mut v = Vec::with_capacity(rows * cols);
        for _ in 0..rows {
            for c in 0..cols {
                let k = if c % 32 == 0 {
                    if u() < 0.5 {
                        -31.0
                    } else {
                        31.0
                    }
                } else {
                    (g(u) * std / scale).round().clamp(-31.0, 31.0)
                };
                v.push((k * scale) as f32);
            }
        }
        v
    };
    let tied = !m["tensors"].as_array().is_some_and(|t| t.iter().any(|e| e[0] == "lm_head.weight"));
    let mut tensors: Vec<(String, &str, Vec<u64>, Vec<u8>)> = Vec::new();
    for e in m["tensors"].as_array().ok_or_else(|| Error("manifest".into()))? {
        let name = e[0].as_str().unwrap_or_default().to_string();
        let shape: Vec<u64> = e[2].as_array().map(|a| a.iter().filter_map(Value::as_u64).collect()).unwrap_or_default();
        let n: usize = shape.iter().product::<u64>() as usize;
        if e[1] == "I64" {
            let vals: Vec<i64> = e[3].as_array().map(|a| a.iter().filter_map(Value::as_i64).collect()).unwrap_or_default();
            tensors.push((name, "I64", shape, vals.iter().flat_map(|x| x.to_le_bytes()).collect()));
            continue;
        }
        let cols = *shape.last().unwrap_or(&1) as usize;
        let vals: Vec<f32> = if name.ends_with("norm.weight")
            || name.ends_with("layernorm.weight")
            || ["hc_norm", "norm_key", "norm_query", "norm_conv"].iter().any(|k| name.contains(k))
        {
            let plus = if name.contains(".linear_attn.norm.") || dense_norms { 1.0 } else { 0.0 };
            (0..n).map(|_| (u01() - 0.5 + plus) as f32).collect()
        } else if name.ends_with("A_log") {
            (0..n).map(|_| (u01() * 15.0 + 1.0).ln() as f32).collect()
        } else if name.ends_with("dt_bias") {
            (0..n).map(|_| gauss(&mut u01) as f32).collect()
        } else if name.contains("conv1d") {
            (0..n).map(|_| (gauss(&mut u01) * 0.4) as f32).collect()
        } else if name.contains("experts.gate_up_proj") || name.contains("experts.down_proj") {
            (0..shape[0]).flat_map(|_| grid(shape[1] as usize, shape[2] as usize, &mut u01, &mut gauss)).collect()
        } else if name.contains("shared_expert.") {
            grid(shape[0] as usize, shape[1] as usize, &mut u01, &mut gauss)
        } else if name.contains("ngram_embedding") || name.contains("embed_tokens") {
            // (an embedding that is the head too, as Qwen3.5's tied one is, is as wide as a head's rows: else its logits are tens apart and the softmax a point)
            let std = if tied && name.contains("embed_tokens") { 1.0 / (cols as f64).sqrt() } else { 1.0 };
            (0..n).map(|_| (gauss(&mut u01) * std) as f32).collect()
        } else if shape.len() == 2 {
            let s = 1.0 / (cols as f64).sqrt();
            (0..n).map(|_| (gauss(&mut u01) * s) as f32).collect()
        } else {
            return err(format!("no rule for {name} {shape:?}"));
        };
        tensors.push((name, "BF16", shape, vals.iter().flat_map(|&x| bf(x).to_le_bytes()).collect()));
    }
    io(std::fs::create_dir_all(out))?;
    io(std::fs::write(out.join("config.json"), m["config"].as_str().unwrap_or_default()))?;
    if dense_norms {
        // (the dense engine's pack carries the tokenizer: a word-level one over the vocabulary's ids, which no check reads)
        let vocab: serde_json::Map<String, Value> = (0..4096).map(|i| (format!("t{i}"), json!(i))).collect();
        let tok = json!({"version": "1.0", "truncation": null, "padding": null, "added_tokens": [], "normalizer": null, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "t0"}});
        io(std::fs::write(out.join("tokenizer.json"), tok.to_string()))?;
    }
    let refs: Vec<(&str, &str, Vec<u64>, &[u8])> = tensors.iter().map(|(n, d, s, b)| (n.as_str(), *d, s.clone(), &b[..])).collect();
    crate::pack::write_safetensors(&out.join("model.safetensors"), &[("format", "pt".to_string())], &refs)?;
    println!("{}: a random {what}, {} tensors, {:.0} MB", out.display(), tensors.len(), tensors.iter().map(|t| t.3.len()).sum::<usize>() as f64 / 1e6);
    Ok(())
}

fn qwen35_engine(dir: &Path, level: &str) -> Result<Box<dyn Run>> {
    boxed(crate::dynamic::qwen35::load(dir, &dyn_opts(level_layout(level)?)))
}

fn qwen3_engine(model: &Path, level: &str) -> Result<Box<dyn Run>> {
    let o = crate::model::Opts { ctx: 1024, chunk: 64, layout: level_layout(level)?, ..Default::default() };
    let p = crate::store::prepare(&model.to_string_lossy(), o.ctx, o.layout, &|s| eprintln!("{s}"))?;
    let o = crate::model::Opts { tables: Some(p.tables.clone()), ..o };
    Ok(Box::new(Dense(crate::model::Engine::new(&p.dir, &o)?)))
}

/// Every case: a model of the family's layout with its layer kinds, at random (a fixed seed) and small: only models released in the last six months are tested, and those that are recent are larger than a 16 GB card holds.
pub fn cases() -> Vec<Case> {
    vec![
        Case {
            id: "qwen3",
            family: "Qwen3 dense (the dense engine's architecture; no recent release uses it)",
            weights: Weights::Random(qwen3_random),
            engine: qwen3_engine,
            cpu: None,
            vocab: 4096,
        },
        Case {
            id: "qwen35",
            family: "Qwen3.5 family (Gated DeltaNet and gated attention: Qwen3.8-27B's layout)",
            weights: Weights::Random(qwen35_random),
            engine: qwen35_engine,
            cpu: None,
            vocab: 4096,
        },
        Case {
            id: "muse",
            family: "Muse Spark (sliding and full attention, mixture of experts)",
            weights: Weights::Random(crate::muse_ref::random),
            engine: muse_engine,
            cpu: Some(muse_cpu),
            vocab: 4096,
        },
        Case {
            id: "kimi",
            family: "Kimi-K3 (delta attention, latent attention, mixture of experts)",
            weights: Weights::Random(|d| crate::kimi_ref::random(d, 1, true, 8)),
            engine: kimi_engine,
            cpu: Some(kimi_cpu),
            vocab: 4096,
        },
        Case {
            id: "deepseek-v4",
            family: "DeepSeek-V4 (four residual streams, compressed attention, mixture of experts)",
            weights: Weights::Random(|d| crate::dsv4_ref::random(d, None)),
            engine: dsv4_engine,
            cpu: Some(dsv4_cpu),
            vocab: 4096,
        },
        Case {
            id: "gemma4",
            family: "Gemma 4 (sliding and global attention, K = V, soft-capped head)",
            weights: Weights::Random(gemma4_random),
            engine: gemma4_engine,
            cpu: None,
            vocab: 4096,
        },
        Case {
            id: "qwen4x",
            family: "Qwen3.8-Flash-Next (Gated DeltaNet, indexed attention, hyper-connections, per-layer embeddings, mixture of experts)",
            weights: Weights::Random(qwen4x_random),
            engine: qwen4x_engine,
            cpu: None,
            vocab: 2048,
        },
        Case {
            id: "nemotron-h",
            family: "Nemotron-H (Mamba-2, attention, mixture of experts)",
            weights: Weights::Random(|d| crate::kcheck_nh::random(d, 0, 16)),
            engine: nemotron_engine,
            cpu: None,
            vocab: 4096,
        },
        Case {
            id: "glm",
            family: "GLM-5.3 (delta attention, sparse attention, mixture of experts)",
            weights: Weights::Random(|d| crate::kcheck_glm::random(d, false, None)),
            engine: glm_engine,
            cpu: None,
            vocab: 2048,
        },
    ]
}

pub fn find_case(id: &str) -> Result<Case> {
    cases().into_iter().find(|c| c.id == id).ok_or_else(|| Error(format!("no case {id}: {}", cases().iter().map(|c| c.id).collect::<Vec<_>>().join(", "))))
}

/// The directory a case's model is in: the Hub's cache for a real checkpoint, else a random model made under `scratch`.
fn model_dir(c: &Case, scratch: &Path) -> Result<PathBuf> {
    match &c.weights {
        Weights::Hub(id) => {
            if let Ok(s) = glyd_gpu::save::Source::find(id) {
                return Ok(s.dir);
            }
            // (not in the cache: downloaded into it, as the Hub's files are for a run; the persistent volume of the GPU runner keeps them for the next)
            let ui = std::cell::RefCell::new(crate::ui::Ui::new());
            let (hub, remote, wanted) = crate::store::hub_plan_as(id, &ui, true).map_err(|r| Error(r.to_string()))?;
            crate::store::download(&hub, &remote, &wanted, id, &ui).map_err(|r| Error(r.to_string()))?;
            glyd_gpu::save::Source::find(id).map(|s| s.dir).map_err(|e| Error(format!("{id}: {e}")))
        }
        Weights::Random(make) => {
            let dir = scratch.join(c.id);
            if !dir.join("config.json").exists() {
                make(&dir)?;
            }
            Ok(dir)
        }
    }
}

/// The GPU generation as a fixture names it: sm89 for compute capability 8.9.
pub fn gpu_label() -> Result<String> {
    let (major, minor) = crate::dev::Dev::new(0)?.cc_pair();
    Ok(format!("sm{major}{minor}"))
}

/// A case's fixtures: CASE.json.gz (a reference's) and CASE.*.json.gz (an engine's own bits on a GPU generation).
fn fixture_files(dir: &Path, id: &str) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = io(std::fs::read_dir(dir))
        .map_err(|e| Error(format!("{}: {}", dir.display(), e.0)))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            let n = p.file_name().unwrap_or_default().to_string_lossy().to_string();
            n.ends_with(".json.gz") && (n == format!("{id}.json.gz") || n.starts_with(&format!("{id}.")))
        })
        .collect();
    v.sort();
    if v.is_empty() {
        return err(format!("{}: no fixture of {id}", dir.display()));
    }
    Ok(v)
}

/// Where the random models are made: --scratch DIR (the caller's own, kept), else a directory of this run alone (the process and the time in its name, 0700 where the system has modes), removed
/// when the run ends: a path another user could have made first, or one a stale run left, is never read back as a model.
struct Scratch {
    dir: PathBuf,
    own: bool,
}

impl Scratch {
    fn new(a: &crate::args::Args) -> Result<Scratch> {
        if let Some(d) = a.flag("scratch") {
            return Ok(Scratch { dir: PathBuf::from(d), own: false });
        }
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        let dir = std::env::temp_dir().join(format!("glyd-logits-{}-{nanos}", std::process::id()));
        io(std::fs::create_dir(&dir))?; // (an error where it exists)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            io(std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)))?;
        }
        Ok(Scratch { dir, own: true })
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if self.own {
            std::fs::remove_dir_all(&self.dir).ok();
        }
    }
}

fn fixtures_dir(a: &crate::args::Args) -> PathBuf {
    a.flag("fixtures").map(PathBuf::from).unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/logits"))
}

/// `logitscheck [CASE...] [--fixtures DIR] [--scratch DIR] [--cpu-ref] [--level L]`: each case's engine (or CPU reference) against its fixture at each level the fixture bounds. Exit 1 where one is not within.
pub fn check(a: &crate::args::Args) -> Result<()> {
    let fixtures = fixtures_dir(a);
    let want: Vec<&str> = a.pos.iter().skip(1).map(String::as_str).collect();
    let mut all = cases();
    if !want.is_empty() {
        all.retain(|c| want.contains(&c.id));
        if all.len() != want.len() {
            return err(format!("cases: {}", cases().iter().map(|c| c.id).collect::<Vec<_>>().join(", ")));
        }
    }
    if a.has("list-cases") {
        for c in &all {
            println!("{:12} {}", c.id, c.family);
        }
        return Ok(());
    }
    let (cpu, only_level) = (a.has("cpu-ref"), a.flag("level"));
    let scratch = Scratch::new(a)?;
    let (mut failed, mut ran) = (Vec::new(), 0usize);
    let t_all = std::time::Instant::now();
    let gpu = gpu_label().ok();
    for c in &all {
        let dir = model_dir(c, &scratch.dir)?;
        for path in fixture_files(&fixtures, c.id)? {
            let fx = Fixture::read(&path)?;
            if cpu && fx.gpu.is_some() {
                continue; // (an engine's own bits: not what a CPU reference makes)
            }
            if let (Some(want), Some(have)) = (&fx.gpu, &gpu) {
                if want != have && !cpu {
                    println!("SKIP {:12} {}: its bits are {want}'s, this GPU is {have}", c.id, path.file_name().unwrap_or_default().to_string_lossy());
                    continue;
                }
            }
            let levels: Vec<&LevelBound> = fx.bounds.iter().filter(|b| only_level.is_none_or(|l| l == b.level) && (!cpu || b.level == "penguin")).collect();
            for lb in levels {
                let t0 = std::time::Instant::now();
                let mut run = if cpu {
                    match c.cpu {
                        Some(f) => f(&dir)?,
                        None => continue,
                    }
                } else {
                    (c.engine)(&dir, &lb.level)?
                };
                let load = t0.elapsed().as_secs_f64();
                let got = drive(run.as_mut(), &fx.prompt, &fx.decode, &fx.long)?;
                let v = compare(&fx, &lb.bound, &got);
                ran += 1;
                let bound = match &lb.bound {
                    Bound::Bits => "bits".to_string(),
                    Bound::Kl { mean, max, top1 } => format!("KL mean <= {mean:.1e}, max <= {max:.1e}, top-1 >= {:.0}%", top1 * 100.0),
                };
                let name = path.file_name().unwrap_or_default().to_string_lossy().replace(".json.gz", "");
                println!(
                    "{} {:20} {:8} {:3} rows: bits-equal {}, KL mean {:.2e} max {:.2e}, best token {}/{} with a lead, largest logit difference {:.4}  [{bound}] ({load:.1} s to load, {:.1} s)",
                    if v.ok { "PASS" } else { "FAIL" },
                    name,
                    lb.level,
                    v.rows,
                    v.identical,
                    v.mean_kl,
                    v.max_kl,
                    v.top1.0,
                    v.top1.1,
                    v.max_dlogit,
                    t0.elapsed().as_secs_f64()
                );
                for n in v.notes.iter().take(6) {
                    println!("       {n}");
                }
                if !v.ok {
                    failed.push(format!("{name} at {}", lb.level));
                }
            }
        }
    }
    println!("{ran} comparisons in {:.1} s against {}", t_all.elapsed().as_secs_f64(), fixtures.display());
    if failed.is_empty() {
        println!("all logits within their bounds");
        Ok(())
    } else {
        println!("OFF: {}", failed.join(", "));
        drop(scratch); // (the process exits next: the private directory is removed first)
        std::process::exit(1);
    }
}

// ---- the hand-written engines against the description path (R3b, #137)

/// Nemotron-H's hand-written engine (`model_nh.rs`).
struct LegacyNh(crate::model_nh::EngineNh);

impl Run for LegacyNh {
    fn describe(&self) -> String {
        self.0.describe()
    }
    fn reset(&mut self) -> Result<()> {
        self.0.reset()
    }
    fn prefill(&mut self, t: &[u32]) -> Result<()> {
        self.0.prefill(t).map(|_| ())
    }
    fn step(&mut self, t: u32) -> Result<()> {
        self.0.step(t).map(|_| ())
    }
    fn logits(&mut self) -> Result<Vec<f32>> {
        self.0.logits()
    }
}

fn nemotron_legacy(dir: &Path, level: &str) -> Result<Box<dyn Run>> {
    let layout = level_layout(level)?;
    let o = crate::model_nh::OptsNh { ctx: 1024, chunk: 64, graph: false, device: 0, layout, chunked: true, only: None };
    let make = crate::moe_nh::factory(layout.unwrap_or(Layout::Tiered));
    Ok(Box::new(LegacyNh(crate::model_nh::EngineNh::new(dir, &o, Some(&make))?)))
}

/// The cases whose families still have a hand-written engine: (case, the hand-written engine, the description path).
type Pair = (&'static str, fn(&Path, &str) -> Result<Box<dyn Run>>, fn(&Path, &str) -> Result<Box<dyn Run>>);

fn legacy_pairs() -> Vec<Pair> {
    vec![("nemotron-h", nemotron_legacy, nemotron_engine)]
}

/// `paritycheck [CASE...] [--level L] [--scratch DIR]`: each family that still has a hand-written engine (Nemotron-H; Qwen3.5's and Gemma 4's went in R3b) on
/// it and on the description executor, the same random model, prompt and decode tokens as its R7 fixture, at each level the fixture bounds
/// that the hand-written engine runs: rows with the same bits, the same best token, KL of the description path from the hand-written one.
/// The record R3b deletes the engines on (docs/design/runtime.md §7: no old path survives its parity record). Exit 1 where a row's best
/// token differs or KL passes 1e-3.
pub fn parity(a: &crate::args::Args) -> Result<()> {
    let want: Vec<&str> = a.pos.iter().skip(1).map(String::as_str).collect();
    let pairs: Vec<Pair> = legacy_pairs().into_iter().filter(|p| want.is_empty() || want.contains(&p.0)).collect();
    if !want.is_empty() && pairs.len() != want.len() {
        return err(format!("cases: {}", legacy_pairs().iter().map(|p| p.0).collect::<Vec<_>>().join(", ")));
    }
    let (fixtures, scratch, only_level) = (fixtures_dir(a), Scratch::new(a)?, a.flag("level"));
    let mut failed = Vec::new();
    for (id, legacy, desc) in pairs {
        let c = find_case(id)?;
        let dir = model_dir(&c, &scratch.dir)?;
        let fx = Fixture::read(&fixture_files(&fixtures, id)?[0])?;
        let mut levels: Vec<String> = Vec::new();
        for p in fixture_files(&fixtures, id)? {
            for b in Fixture::read(&p)?.bounds {
                if !levels.contains(&b.level) && only_level.is_none_or(|l| l == b.level) {
                    levels.push(b.level);
                }
            }
        }
        for level in levels {
            let mut old = match legacy(&dir, &level) {
                Ok(r) => r,
                Err(e) => {
                    println!("SKIP {id:12} {level:8} the hand-written engine does not run it: {}", e.0);
                    continue;
                }
            };
            let old_rows = drive(old.as_mut(), &fx.prompt, &fx.decode, &fx.long)?;
            drop(old);
            let mut new = desc(&dir, &level)?;
            let new_rows = drive(new.as_mut(), &fx.prompt, &fx.decode, &fx.long)?;
            let arg = |v: &[f32]| v.iter().enumerate().fold(0, |b, (j, &x)| if x > v[b] { j } else { b });
            let (mut same_bits, mut same_best, mut max_kl, mut max_d) = (0usize, 0usize, 0f64, 0f32);
            for (o, n) in old_rows.iter().zip(&new_rows) {
                same_bits += (o == n) as usize;
                same_best += (arg(o) == arg(n)) as usize;
                max_kl = max_kl.max(kl_of(&Row::from_logits(o)?, n));
                max_d = o.iter().zip(n).fold(max_d, |m, (x, y)| m.max((x - y).abs()));
            }
            let rows = old_rows.len();
            let ok = same_best == rows && max_kl <= 1e-3;
            println!(
                "{} {id:12} {level:8} {rows:3} rows: bits-equal {same_bits}, best token {same_best}/{rows}, KL max {max_kl:.2e}, largest logit difference {max_d:.4}",
                if ok { "PASS" } else { "FAIL" }
            );
            if !ok {
                failed.push(format!("{id} at {level}"));
            }
        }
    }
    if failed.is_empty() {
        println!("the description path gives the hand-written engines' tokens");
        Ok(())
    } else {
        println!("OFF: {}", failed.join(", "));
        drop(scratch);
        std::process::exit(1);
    }
}

/// The per-level KL thresholds of a lossy level from the lossless one (mean, max over rows), per family's random model: Qwen3.5's and
/// Nemotron-H's are the bounds the qwen35 fixture holds for the engine against transformers in bf16 (tests/logits/qwen35.json.gz, about
/// twice a measured run); Gemma 4's random model moves more under every rounding, kestrel too (its levels have shipped since v0.30), so
/// its thresholds are about twice its own first measured run (RTX 4080 SUPER, 2026-11-21: mean/max KL kestrel 2.15e-2/1.03e-1, swift
/// 3.70e-2/1.01e-1, wren 9.55e-2/2.27e-1; Nemotron-H's wren 8.5e-4/6.2e-3, its experts alone four-bit).
pub const LEVEL_KL: [(&str, f64, f64); 3] = [("kestrel", 0.006, 0.015), ("swift", 0.025, 0.05), ("wren", 0.09, 0.14)];
pub const LEVEL_KL_GEMMA4: [(&str, f64, f64); 3] = [("kestrel", 0.045, 0.21), ("swift", 0.075, 0.21), ("wren", 0.19, 0.46)];

/// `levelcheck [CASE...] [--level L] [--scratch DIR]`: each family with lossy levels (Qwen3.5, Gemma 4, Nemotron-H: Nemotron 3.5
/// Lightning's layout) on its random model at penguin and at each lossy level, the same tokens as its R7 fixture, and the KL of the
/// lossy level's rows from penguin's against `LEVEL_KL` (`LEVEL_KL_GEMMA4` for Gemma 4). Exit 1 where one is past its threshold.
pub fn level_check(a: &crate::args::Args) -> Result<()> {
    let want: Vec<&str> = a.pos.iter().skip(1).map(String::as_str).collect();
    let ids: Vec<&str> = ["qwen35", "gemma4", "nemotron-h"].into_iter().filter(|c| want.is_empty() || want.contains(c)).collect();
    if !want.is_empty() && ids.len() != want.len() {
        return err("cases: qwen35, gemma4, nemotron-h");
    }
    let (fixtures, scratch, only_level) = (fixtures_dir(a), Scratch::new(a)?, a.flag("level"));
    let mut failed = Vec::new();
    for id in ids {
        let c = find_case(id)?;
        let dir = model_dir(&c, &scratch.dir)?;
        let fx = Fixture::read(&fixture_files(&fixtures, id)?[0])?;
        let engine = c.engine;
        let mut base = match engine(&dir, "penguin") {
            Ok(r) => r,
            Err(e) => {
                println!("SKIP {id:12} penguin, the base: {}", e.0);
                continue;
            }
        };
        let base_rows = drive(base.as_mut(), &fx.prompt, &fx.decode, &fx.long)?;
        drop(base);
        for (level, mean_bound, max_bound) in if id == "gemma4" { LEVEL_KL_GEMMA4 } else { LEVEL_KL } {
            if only_level.is_some_and(|l| l != level) {
                continue;
            }
            let mut run = match engine(&dir, level) {
                Ok(r) => r,
                Err(e) => {
                    println!("SKIP {id:12} {level:8} {}", e.0);
                    continue;
                }
            };
            let rows = drive(run.as_mut(), &fx.prompt, &fx.decode, &fx.long)?;
            let kls: Vec<f64> = base_rows.iter().zip(&rows).map(|(b, r)| Row::from_logits(b).map(|b| kl_of(&b, r))).collect::<Result<_>>()?;
            let (mean, max) = (kls.iter().sum::<f64>() / kls.len() as f64, kls.iter().cloned().fold(0f64, f64::max));
            let ok = mean <= mean_bound && max <= max_bound;
            println!(
                "{} {id:12} {level:8} {:3} rows: KL from penguin mean {mean:.2e} (at most {mean_bound:.1e}), max {max:.2e} (at most {max_bound:.1e})",
                if ok { "PASS" } else { "FAIL" },
                kls.len()
            );
            if !ok {
                failed.push(format!("{id} at {level}"));
            }
        }
    }
    if failed.is_empty() {
        println!("every lossy level within its KL threshold");
        Ok(())
    } else {
        println!("OFF: {}", failed.join(", "));
        drop(scratch);
        std::process::exit(1);
    }
}

/// `logitsmodel CASE DIR`: the case's random model made in DIR (a model to run the other checks on: `kvtest DIR`, `run DIR`).
pub fn make_model(a: &crate::args::Args) -> Result<()> {
    let c = find_case(a.need(1, "CASE")?)?;
    match c.weights {
        Weights::Random(make) => make(Path::new(a.need(2, "DIR")?)),
        Weights::Hub(id) => err(format!("{}: a real model ({id}), not a random one", c.id)),
    }
}

/// `logitsbound FIXTURE LEVEL=BOUND...`: the fixture's bounds replaced (bits, or kl:MEAN:MAX:TOP1), from what a run measured.
pub fn set_bounds(a: &crate::args::Args) -> Result<()> {
    let path = PathBuf::from(a.need(1, "FIXTURE")?);
    let mut fx = Fixture::read(&path)?;
    let mut bounds = Vec::new();
    for item in a.pos.iter().skip(2) {
        let (level, b) = item.split_once('=').ok_or_else(|| Error(format!("{item}: LEVEL=BOUND")))?;
        level_layout(level)?;
        bounds.push(LevelBound { level: level.into(), bound: parse_bound(b)? });
    }
    if bounds.is_empty() {
        return err("LEVEL=BOUND (bits, or kl:MEAN:MAX:TOP1)");
    }
    fx.bounds = bounds;
    fx.write(&path)
}

fn parse_bound(s: &str) -> Result<Bound> {
    let p: Vec<&str> = s.split(':').collect();
    match p.as_slice() {
        ["bits"] => Ok(Bound::Bits),
        ["kl", m, x, t] => {
            let f = |v: &str| v.parse::<f64>().map_err(|_| Error(format!("bound {s}: {v} is not a number")));
            Ok(Bound::Kl { mean: f(m)?, max: f(x)?, top1: f(t)? })
        }
        _ => err(format!("bound {s}: bits or kl:MEAN:MAX:TOP1")),
    }
}

fn git_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short=10", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

/// `logitsrecord CASE --from cpu|engine --out FILE|DIR [--level L] [--bound bits|kl:MEAN:MAX:TOP1] [--note TEXT] [--scratch DIR]`: a fixture from the CPU reference of src/*_ref.rs, or from this engine
/// (a release's own, where the family has no reference: say so in --note). The levels the fixture bounds: --bound gives the first's (the lossless one), `--levels kestrel=kl:1e-3:5e-3:0.9,...` more.
pub fn record(a: &crate::args::Args) -> Result<()> {
    let c = find_case(a.need(1, "CASE")?)?;
    let from = a.flag("from").unwrap_or("engine");
    let scratch = Scratch::new(a)?;
    let dir = model_dir(&c, &scratch.dir)?;
    let (prompt, decode, long) = match a.flag("tokens") {
        Some(f) => {
            let v: Value = serde_json::from_str(&io(std::fs::read_to_string(f))?).map_err(|e| Error(format!("{f}: {e}")))?;
            let t = |k: &str| v[k].as_array().map(|x| x.iter().filter_map(|n| n.as_u64().map(|n| n as u32)).collect::<Vec<_>>()).unwrap_or_default();
            (t("prompt"), t("decode"), t("long"))
        }
        None if c.vocab > 0 => standard_tokens(c.vocab),
        None => return err(format!("{}: a Hub model's tokens come from --tokens FILE (tools/logits_ref_hf.py)", c.id)),
    };
    let mut run = match from {
        "cpu" => (c.cpu.ok_or_else(|| Error(format!("{} has no CPU reference", c.id)))?)(&dir)?,
        "engine" => (c.engine)(&dir, a.flag("level").unwrap_or("penguin"))?,
        o => return err(format!("--from {o}: cpu or engine")),
    };
    let t0 = std::time::Instant::now();
    let got = drive(run.as_mut(), &prompt, &decode, &long)?;
    let rows = got.iter().map(|l| Row::from_logits(l)).collect::<Result<Vec<_>>>()?;
    let level = if from == "cpu" { "penguin" } else { a.flag("level").unwrap_or("penguin") };
    let mut bounds = vec![LevelBound { level: level.into(), bound: parse_bound(a.flag("bound").unwrap_or("bits"))? }];
    if let Some(l) = a.flag("levels") {
        for item in l.split(',').filter(|s| !s.is_empty()) {
            let (level, b) = item.split_once('=').ok_or_else(|| Error(format!("--levels {item}: LEVEL=BOUND")))?;
            level_layout(level)?;
            bounds.push(LevelBound { level: level.into(), bound: parse_bound(b)? });
        }
    }
    let mut provenance = BTreeMap::new();
    provenance.insert(
        "source".into(),
        match from {
            "cpu" => format!(
                "the CPU reference of src/{}_ref.rs, fp32 inside, on this case's random model (seed fixed in the generator)",
                c.id.replace("deepseek-", "ds").replace('-', "")
            ),
            _ => format!(
                "glyd-engine {} at {} ({}, level {}): the engine's own output, no reference of another implementation",
                env!("CARGO_PKG_VERSION"),
                git_sha(),
                run.describe(),
                a.flag("level").unwrap_or("penguin")
            ),
        },
    );
    provenance.insert("recorded".into(), a.flag("date").unwrap_or("").to_string());
    provenance.insert("note".into(), a.flag("note").unwrap_or("").to_string());
    provenance.insert(
        "tokens".into(),
        if a.flag("tokens").is_some() { "from the reference tool".into() } else { "the fixed list of src/logits.rs token_list".into() },
    );
    let model = match &c.weights {
        Weights::Hub(id) => id.to_string(),
        Weights::Random(_) => format!("random {} model, made by the generator of this case", c.id),
    };
    let gpu = if from == "engine" { Some(gpu_label()?) } else { None };
    let fx = Fixture { case: c.id.into(), family: c.family.into(), model, provenance, gpu, bounds, prompt, decode, long, rows };
    let out = PathBuf::from(a.flag("out").ok_or_else(|| Error("--out FILE".into()))?);
    let out = if out.is_dir() {
        out.join(match &fx.gpu {
            Some(g) => format!("{}.{g}.json.gz", c.id),
            None => format!("{}.json.gz", c.id),
        })
    } else {
        out
    };
    fx.write(&out)?;
    println!(
        "{}: {} rows from {from} in {:.1} s, {} bytes",
        out.display(),
        fx.rows.len(),
        t0.elapsed().as_secs_f64(),
        std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits(n: usize, seed: u32) -> Vec<f32> {
        (0..n).map(|i| (((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 8) % 4000) as f32 / 250.0 - 8.0).collect()
    }

    fn fixture(rows: Vec<Vec<f32>>) -> Fixture {
        Fixture {
            case: "t".into(),
            family: "t".into(),
            model: "t".into(),
            provenance: BTreeMap::new(),
            gpu: None,
            bounds: vec![LevelBound { level: "penguin".into(), bound: Bound::Bits }],
            prompt: vec![1, 2],
            decode: vec![3; rows.len() - 1],
            long: vec![],
            rows: rows.iter().map(|r| Row::from_logits(r).unwrap()).collect(),
        }
    }

    #[test]
    fn a_row_is_its_largest_and_a_tie_is_the_smaller_token() {
        let r = Row::from_logits(&[1.0, 5.0, 5.0, 2.0]).unwrap();
        assert_eq!(r.ids, vec![1, 2, 3, 0]);
        assert!(Row::from_logits(&[1.0, f32::NAN]).is_err());
    }

    #[test]
    fn a_fixture_reads_back_the_same_bits_through_its_file() {
        let rows = vec![logits(5000, 1), logits(5000, 2)];
        let fx = fixture(rows.clone());
        let path = std::env::temp_dir().join(format!("glyd-logits-test-{}.json.gz", std::process::id()));
        fx.write(&path).unwrap();
        let back = Fixture::read(&path).unwrap();
        std::fs::remove_file(&path).ok();
        for (x, y) in back.rows.iter().zip(&fx.rows) {
            assert_eq!((&x.ids, &x.logits, x.hash), (&y.ids, &y.logits, y.hash));
            assert!((x.lse - y.lse).abs() < 1e-9);
        }
        assert!(compare(&back, &Bound::Bits, &rows).ok);
    }

    #[test]
    fn the_same_logits_pass_and_one_wrong_logit_fails_bits_and_a_wrong_best_token_fails_kl() {
        let rows = vec![logits(5000, 1), logits(5000, 2), logits(5000, 3)];
        let fx = fixture(rows.clone());
        assert!(compare(&fx, &Bound::Bits, &rows).ok);
        // a kernel bug that moves one logit far down the vocabulary: the hash sees it
        let mut bug = rows.clone();
        let low = bug[1].iter().enumerate().fold((0, f32::INFINITY), |m, (i, &x)| if x < m.1 { (i, x) } else { m }).0;
        bug[1][low] += 0.5;
        let v = compare(&fx, &Bound::Bits, &bug);
        assert!(!v.ok && v.identical == 2, "{:?}", v.notes);
        // the best token swapped with another by a bug: the KL and the greedy check see it
        let kl = Bound::Kl { mean: 1e-3, max: 1e-3, top1: 1.0 };
        let mut bug = rows.clone();
        let best = fx.rows[2].ids[0] as usize;
        bug[2][best] -= 3.0;
        let v = compare(&fx, &kl, &bug);
        assert!(!v.ok && v.max_kl > 1e-3, "{:?}", v);
        // rounding-sized noise stays within
        let noisy: Vec<Vec<f32>> = rows.iter().map(|r| r.iter().enumerate().map(|(i, &x)| x + if i % 2 == 0 { 0.004 } else { -0.004 }).collect()).collect();
        let v = compare(&fx, &kl, &noisy);
        assert!(v.ok, "{:?}", v);
    }

    #[test]
    fn every_case_has_a_reference_fixture_or_an_engine_s_own_and_each_reads() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/logits");
        for c in cases() {
            let files = fixture_files(&dir, c.id).unwrap();
            for f in files {
                let fx = Fixture::read(&f).unwrap();
                assert_eq!(fx.case, c.id, "{}", f.display());
                assert!(!fx.provenance.get("source").map_or(true, |s| s.is_empty()), "{}: no provenance", f.display());
                assert!(!fx.bounds.is_empty());
                for lb in &fx.bounds {
                    level_layout(&lb.level).unwrap();
                    // (a fixture of an engine's own bits is Bits at the level it was made at and carries the GPU it was made on)
                    assert!(fx.gpu.is_none() || fx.bounds.len() == 1 && lb.bound == Bound::Bits, "{}", f.display());
                }
                assert!(
                    fx.rows.iter().all(|r| r.ids.len() == TOPK.min(r.ids.len()) && r.ids.len() >= 2 && r.logits.windows(2).all(|w| w[0] >= w[1])),
                    "{}",
                    f.display()
                );
                assert!(fx.prompt.len() >= 8 && !fx.decode.is_empty() && fx.long.len() > 64, "{}: tokens", f.display());
            }
        }
    }

    #[test]
    fn kl_is_zero_for_the_same_row_and_grows_with_a_shift_of_the_mass() {
        let l = logits(2000, 7);
        let r = Row::from_logits(&l).unwrap();
        assert!(kl_of(&r, &l) < 1e-12);
        let mut shifted = l.clone();
        for &i in &r.ids[..8] {
            shifted[i as usize] -= 1.0;
        }
        assert!(kl_of(&r, &shifted) > 1e-3);
    }
}

/// `usercheck CASE [--level L] [--users N] [--tokens T]` (gemma4, qwen35, muse, kimi, glm): the case's random model on the description executor, each of N prompts replied to alone as `run` replies (the reference),
/// then by the scheduler (src/dynamic/rows.rs) alone, then all at once. Alone through the scheduler every reply is the reference, token for token; at once too where GLYD_ENGINE_BATCH_PREFILL=0 (prompts
/// one a pass, decode steps of up to 8 rows: a step's sums). With prompts run together the replies that differ are counted, not refused (a product of more rows may take another route; a random
/// model's near ties turn on its last bits).
pub fn users_check(a: &crate::args::Args) -> Result<()> {
    use crate::conv::Gen;
    use crate::sched::{Ev, Finish, Reply, Sched, Submit};
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    let case = a.need(1, "CASE")?;
    let level = a.flag("level").unwrap_or("penguin");
    let users: usize = a.num("users", 4usize)?;
    let tokens: usize = a.num("tokens", 24usize)?;
    let dir = std::env::temp_dir().join(format!("glyd-usercheck-{case}-{}", std::process::id()));
    let o = dyn_opts(level_layout(level)?);
    let mut e: Dyn = match case {
        "gemma4" => {
            gemma4_random(&dir)?;
            let c = crate::gemma4::ConfigG4::load(&dir)?;
            Dyn::new(&dir, crate::dynamic::gemma4::describe(&c), None, &o)?
        }
        "qwen35" => {
            qwen35_random(&dir)?;
            crate::dynamic::qwen35::load(&dir, &o)?
        }
        "muse" => {
            crate::muse_ref::random(&dir)?;
            crate::dynamic::muse::load(&dir, &o)?
        }
        "kimi" => {
            crate::kimi_ref::random(&dir, 1, true, 8)?;
            crate::dynamic::kimi::load(&dir, &o)?
        }
        "glm" => {
            // (its experts at a lossy level, as its logits case: kestrel where penguin is asked)
            crate::kcheck_glm::random(&dir, false, None)?;
            crate::dynamic::glm::load(&dir, &dyn_opts(Some(Layout::Tiered)), level_layout(level)?.unwrap_or(Layout::Int6))?
        }
        other => return err(format!("{other}: gemma4, qwen35, muse, kimi or glm")),
    };
    let _ = std::fs::remove_dir_all(&dir);
    let vocab = e.desc().vocab as u32;
    let prompts: Vec<Vec<u32>> = (0..users).map(|k| (0..5 + 29 * k as u32).map(|i| 3 + (i * 7919 + k as u32 * 104_729) % (vocab - 3)).collect()).collect();
    let reply = Reply {
        params: crate::sampler::Params { temperature: 0.0, ..Default::default() },
        max_new: tokens,
        ignore_eos: true,
        logprobs: None,
        stop_ids: Vec::new(),
    };
    let reference: Vec<Vec<u32>> = prompts.iter().map(|p| e.reply(p, &reply, &mut |_| true).map(|r| r.0)).collect::<Result<_>>()?;
    let batchable = e.batchable();
    let Some((k, step)) = e.as_step(users)? else { return err(format!("{case}: no scheduler's step")) };
    let mut s = Sched::new(step, k);
    let ask = |s: &mut Sched<&mut dyn crate::sched::ModelStep>, ps: &[Vec<u32>]| -> Result<Vec<Vec<u32>>> {
        let rx: Vec<_> = ps
            .iter()
            .map(|p| {
                let (tx, rx) = std::sync::mpsc::channel();
                s.submit(Submit {
                    prompt: p.clone(),
                    reply: reply.clone(),
                    out: tx,
                    abort: Arc::new(AtomicBool::new(false)),
                    resumed: Default::default(),
                    images: Vec::new(),
                });
                rx
            })
            .collect();
        while !s.idle() {
            s.tick()?;
        }
        rx.iter()
            .map(|r| {
                let mut out = Vec::new();
                loop {
                    match r.recv().map_err(|_| Error("the scheduler dropped a sequence".into()))? {
                        Ev::Token(t) => out.push(t),
                        Ev::Lp(_) => {}
                        Ev::End(Ok(Finish::Length)) => return Ok(out),
                        Ev::End(f) => return err(format!("a sequence ended {f:?} before its {tokens} tokens")),
                    }
                }
            })
            .collect()
    };
    let mut fails = 0;
    for (i, p) in prompts.iter().enumerate() {
        let got = ask(&mut s, std::slice::from_ref(p))?.remove(0);
        let same = got == reference[i];
        fails += !same as usize;
        println!(
            "{} alone {i} ({} prompt tokens): {}",
            if same { "PASS" } else { "FAIL" },
            p.len(),
            if same { "the reply alone, token for token".into() } else { format!("{got:?} against {:?}", reference[i]) }
        );
    }
    let together = ask(&mut s, &prompts)?;
    let same = together.iter().zip(&reference).filter(|(a, b)| a == b).count();
    // (with a page budget too small for all of them at once (GLYD_ENGINE_PAGES), sequences give their blocks back and run again from their tokens, the generated ones as a prompt: the
    // replies are then counted, not required to be the same)
    let squeezed = std::env::var_os("GLYD_ENGINE_PAGES").is_some();
    let strict = std::env::var_os("GLYD_ENGINE_BATCH_PREFILL").is_some_and(|v| v == "0") && !squeezed;
    let steps: Vec<u64> = crate::sched::SHARED[18..27].iter().map(|x| x.load(std::sync::atomic::Ordering::Relaxed)).collect();
    let resumed = crate::sched::SHARED[2].load(std::sync::atomic::Ordering::Relaxed);
    println!(
        "at once: {same} of {} replies the reply alone (prompts {}); {k} sequences, rows in one pass: {batchable}; decode steps by rows {steps:?}; sequences sent back for room {resumed}",
        prompts.len(),
        if strict { "one a pass" } else { "together" }
    );
    if squeezed && resumed == 0 {
        println!("FAIL a page budget of GLYD_ENGINE_PAGES sent no sequence back");
        fails += 1;
    }
    if strict && same < prompts.len() {
        println!("FAIL replies at once differ from alone with prompts one a pass");
        fails += 1;
    }
    if batchable && k > 1 && steps[2..].iter().all(|&x| x == 0) {
        println!("FAIL no decode step had more than one row");
        fails += 1;
    }
    if fails > 0 {
        return err(format!("{case}: {fails} failed"));
    }
    println!("{case}: PASS");
    Ok(())
}
