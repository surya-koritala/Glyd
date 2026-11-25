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
/// The words of a case's engine that is not built yet (its fixture the CPU reference's): logitscheck skips its levels.
pub const NOT_ON_EXECUTOR: &str = "not on the executor yet";
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
        let raw = io(std::fs::read(path)).map_err(|e| Error(format!("{}: {}", path.display(), e.msg)))?;
        let text = if raw.starts_with(&[0x1f, 0x8b]) {
            let mut s = String::new();
            io(flate2::read::GzDecoder::new(&raw[..]).read_to_string(&mut s))?;
            s
        } else {
            String::from_utf8(raw).map_err(|e| Error(e.to_string()))?
        };
        let v: Value = serde_json::from_str(&text).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        Fixture::from_json(&v).map_err(|e| Error(format!("{}: {}", path.display(), e.msg)))
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
                v.notes.push(format!("row {i}: {}", e.msg));
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

/// A model directory's description (`runtime::describe`) on the CPU reference of the layer kinds, a token at a time: the reference of a family that has no hand-written one
/// (Nemotron-H). Its weights are read as they are stored (bf16 or f32).
fn described_cpu(dir: &Path) -> Result<Box<dyn Run>> {
    described_cpu_with(dir, |_| {})
}

/// `described_cpu` of the description `edit` changes (a test's: another rounding point, say).
fn described_cpu_with(dir: &Path, edit: fn(&mut crate::runtime::describe::Described)) -> Result<Box<dyn Run>> {
    use crate::runtime::reference::model::{checkpoint_tensor, RefModel};
    let d = dir.to_path_buf();
    let load = move || {
        let cfg: Value = serde_json::from_str(&io(std::fs::read_to_string(d.join("config.json")))?).map_err(|e| Error(format!("config.json: {e}")))?;
        let mut desc = crate::runtime::describe::describe(&cfg)?;
        edit(&mut desc);
        let ck = io(glyd_gpu::safetensors::Checkpoint::open(&d))?;
        let mut m = RefModel::load(&desc, &|n: &str| checkpoint_tensor(&ck, n))?;
        // (Engram's compressed vocabulary, where the model has Engram layers)
        if let Some(map) = engram_token_map(&d)? {
            for l in m
                .layers
                .iter_mut()
                .filter(|l| matches!(l.block.body, crate::runtime::kinds::Body::Ngram(g) if g.form == crate::runtime::kinds::NgramForm::Engram))
            {
                l.p.put("ngram_map", &[map.len()], map.clone());
            }
        }
        let mut pos = 0u32;
        Ok(move |t: u32| {
            let l = m.forward(&[t], pos).unwrap_or_else(|e| panic!("the reference at position {pos}: {}", e.msg)).pop().expect("a token's logits");
            pos += 1;
            l
        })
    };
    Ok(Box::new(CpuRef { load: Box::new(load), m: None, last: Vec::new() }))
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
    boxed(crate::dynamic::gemma4::lowered(dir, &c).and_then(|d| Dyn::new(dir, d, None, &o)))
}

/// A small Gemma 4 of the 12B's layer kinds (sliding layers, and a global one whose keys are its values), the heads the engine runs (256 and 512), a soft-capped tied head, at random (a fixed seed).
/// The random Gemma 4 model's config (`gemma4_random`): three sliding layers and a global one.
pub(crate) const GEMMA4_RANDOM_CONFIG: &str = r#"{"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4", "eos_token_id": [1, 106],
        "text_config": {"model_type": "gemma4_text", "hidden_size": 512, "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2, "num_global_key_value_heads": 1,
        "attention_k_eq_v": true, "head_dim": 256, "global_head_dim": 512, "intermediate_size": 1024, "vocab_size": 4096, "rms_norm_eps": 1e-6, "sliding_window": 32,
        "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"], "final_logit_softcapping": 30.0, "max_position_embeddings": 4096,
        "enable_moe_block": false, "hidden_size_per_layer_input": 0, "hidden_activation": "gelu_pytorch_tanh",
        "rope_parameters": {"sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}, "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25, "rope_type": "proportional"}}}}"#;

pub fn gemma4_random(out: &Path) -> Result<()> {
    let text = GEMMA4_RANDOM_CONFIG;
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

/// A random model of a Gemma 4 config `text`, every tensor at the shape its description gives (`Described::shapes`) and drawn from `seed` in the order of
/// their names: the embedding in [-1, 1], the q and k norms' weights in [0.1, 0.3], the other norms in [0.5, 1.5], layer scalars in [0.5, 1], every other
/// matrix in +-sqrt(3 / its columns). The sizes the E2B and E4B releases add (KV sharing, the double-wide MLP) come from the description, so each config's
/// model is its own shape. (Gemma 4's scores are not scaled: the q and k norms set them. With those weights in [0.5, 1.5] a random model's scores reach
/// about +-16 at heads of 256, its softmax is nearly one-hot and the engine's bf16 rounding can move a near tie: on an A10, eight-layer models of six seeds
/// were 2e-3 to 3.2e-1 in KL mean from the CPU reference, their largest logit differences up to 13, with KV sharing and without alike; at [0.1, 0.3] every
/// one was 6e-4 to 1.3e-3, at most 2.3e-3 a row and 0.37 a logit.)
pub fn gemma4_shaped_random(out: &Path, text: &str, seed: u64) -> Result<()> {
    let cfg: Value = serde_json::from_str(text).map_err(|e| Error(format!("config.json: {e}")))?;
    let d = crate::runtime::describe::describe(&cfg)?;
    let mut r = seed;
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
    for (name, shape) in &d.shapes {
        let (lo, hi) = if *name == d.embed {
            (-1.0, 1.0)
        } else if name.ends_with("layer_scalar") {
            (0.5, 1.0)
        } else if name.ends_with("q_norm.weight") || name.ends_with("k_norm.weight") {
            (0.1, 0.3)
        } else if shape.len() == 1 {
            (0.5, 1.5)
        } else {
            let a = (3.0 / shape[shape.len() - 1] as f32).sqrt();
            (-a, a)
        };
        let n: usize = shape.iter().product();
        t.push((name.clone(), shape.iter().map(|&x| x as u64).collect(), (0..n).flat_map(|_| bf(uni(lo, hi)).to_le_bytes()).collect()));
    }
    io(std::fs::create_dir_all(out))?;
    io(std::fs::write(out.join("config.json"), text))?;
    let refs: Vec<(&str, &str, Vec<u64>, &[u8])> = t.iter().map(|(n, s, b)| (n.as_str(), "BF16", s.clone(), &b[..])).collect();
    crate::pack::write_safetensors(&out.join("model.safetensors"), &[("format", "pt".to_string())], &refs)?;
    println!("{}: a random Gemma 4, {} tensors, {:.0} MB", out.display(), t.len(), t.iter().map(|t| t.2.len()).sum::<usize>() as f64 / 1e6);
    Ok(())
}

/// A small Gemma 4 with the E2B and E4B releases' KV sharing and double-wide MLP: eight layers, the last four (three sliding, one global) reading the
/// keys and values of the last sliding and the last global layer before them, their MLPs twice as wide; no K = V (the E sizes have none).
pub fn gemma4_kv_random(out: &Path) -> Result<()> {
    let text = r#"{"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4", "eos_token_id": [1, 106],
        "text_config": {"model_type": "gemma4_text", "hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 4, "num_key_value_heads": 1,
        "attention_k_eq_v": false, "head_dim": 256, "global_head_dim": 512, "intermediate_size": 512, "vocab_size": 4096, "rms_norm_eps": 1e-6, "sliding_window": 32,
        "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        "num_kv_shared_layers": 4, "use_double_wide_mlp": true, "final_logit_softcapping": 30.0, "max_position_embeddings": 4096,
        "enable_moe_block": false, "hidden_size_per_layer_input": 0, "hidden_activation": "gelu_pytorch_tanh",
        "rope_parameters": {"sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}, "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25, "rope_type": "proportional"}}}}"#;
    gemma4_shaped_random(out, text, 0x2026_1123_0000_00e2)
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

/// The experts of Qwen3.8-Flash-Next are held in six-bit codes at its kestrel level (the random model's are on that grid, as tools/qwen4x_ref.py makes them). At penguin:
/// the description executor (src/dynamic/qwen4x.rs: the other matrices lossless, the experts at six bits, which hold the random model's exactly); at the other levels
/// Condor's own engine. The penguin fixture's bound (tests/logits/qwen4x.json.gz: KL mean 1e-4, max 5e-4) is the cost of the routers' near ties: the random model's 16
/// experts, 4 a token, leave 15 choices in its 12 decode rows within 1e-2 of the next expert (`reference::ffn::margin`), which a bf16 engine may take either way, and
/// each taken the other way on the reference costs its row a KL of 7.8e-6 to 3.5e-4 (the most: row 7, the fourth layer's experts, margin 3.8e-3; an A100 running the
/// executor measured 3.52e-4 at that row, 2026-11-23).
fn qwen4x_engine(dir: &Path, level: &str) -> Result<Box<dyn Run>> {
    let Some(experts) = level_layout(level)? else { return boxed(crate::dynamic::qwen4x::load(dir, &dyn_opts(None))) };
    let o = crate::model_qwen4x::OptsQx { ctx: 1024, chunk: 64, layout: Layout::Tiered, chunked: true, ..Default::default() };
    let make = crate::model_qwen4x::gpu_experts(experts);
    Ok(Box::new(Qwen4x { e: crate::model_qwen4x::EngineQx::new(dir, &o, Some(&make))?, chunk: 64, last_chunk: 0 }))
}

/// The random tiny Qwen3.8-Flash-Next of tools/qwen4x_ref.py (8 layers: three Gated DeltaNet and one indexed attention, twice; a layer with per-layer embeddings; 16 experts; hyper-connections): its tensors'
/// names, types and shapes from tests/fixtures/qwen4x_tiny_manifest.json (what that tool wrote), their values at random from a fixed seed by the tool's rules for each kind of tensor.
pub fn qwen4x_random(out: &Path) -> Result<()> {
    manifest_random(include_str!("../tests/fixtures/qwen4x_tiny_manifest.json"), out, 0x2026_1120_0000_0005, false, "Qwen3.8-Flash-Next of the tiny shape")
}

/// The plain dense kind's small model (a pre-norm decoder of one attention and one gated SiLU MLP a layer: 4 layers, 16 query and 8 key-value heads of 128 with normed
/// queries and keys, a vocabulary of 4096): the kind's case, no release's.
pub fn dense_random(out: &Path) -> Result<()> {
    manifest_random(include_str!("../tests/fixtures/dense_tiny_manifest.json"), out, 0x2026_1120_0000_0006, true, "plain dense model")
}

/// A small Qwen3.5-family text model (Qwen3_5ForCausalLM: three Gated DeltaNet layers and a gated full-attention one, 1024 wide, a vocabulary of 4096): the layout of Qwen3.8-27B and the releases of its family at 0.8B's widths.
pub fn qwen35_random(out: &Path) -> Result<()> {
    manifest_random(include_str!("../tests/fixtures/qwen35_tiny_manifest.json"), out, 0x2026_1120_0000_0007, false, "Qwen3.5-family")
}

/// A small model of the Qwen3.5 layout with experts (`Qwen3_5MoeForCausalLM`; tests/fixtures/qwen35moe_tiny_config.json: the family's config keys at 1024 wide and four layers, three
/// Gated DeltaNet and a gated full-attention one, each followed by 32 experts of which a token takes 10 and the gated shared expert): every tensor of its description at its
/// shape, its values by `manifest_random`'s rules for each kind of tensor.
pub fn qwen35moe_random(out: &Path) -> Result<()> {
    let text = include_str!("../tests/fixtures/qwen35moe_tiny_config.json");
    let d = crate::runtime::describe::describe(&serde_json::from_str(text).map_err(|e| Error(e.to_string()))?)?;
    let tensors = d
        .names()
        .into_iter()
        .map(|n| d.shapes.get(&n).map(|s| json!([n, "BF16", s])).ok_or_else(|| Error(format!("the description has no shape for {n}"))))
        .collect::<Result<Vec<Value>>>()?;
    manifest_random(&json!({"config": text, "tensors": tensors}).to_string(), out, 0x2026_1124_0000_0333, false, "Qwen3.5-family mixture of experts")
}

/// The Qwen3.5 family's layout in the shape its larger checkpoints have and `qwen35_random` lacks (Qwen3_5ForConditionalGeneration: the text model under `text_config`, its
/// tensors under `model.language_model.`): three Gated DeltaNet layers and a gated full-attention one, twice; three value heads to a key head in the linear attention and
/// six query heads to a key-value head; a head of its own (no tied embedding); an MTP layer and a part of a vision tower in the checkpoint, which the engine counts or leaves.
pub fn qwen35_wide_random(out: &Path) -> Result<()> {
    manifest_random(
        include_str!("../tests/fixtures/qwen35_wide_tiny_manifest.json"),
        out,
        0x2026_1125_0000_0038,
        false,
        "Qwen3.5-family model of the wide shape",
    )
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
            || ["hc_norm", "norm_key", "norm_query", "norm_conv", "pre_fc_norm"].iter().any(|k| name.contains(k))
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
        // (a tokenizer: a word-level one over the vocabulary's ids, which the 12-bit KV tables' calibration reads the corpus through)
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

/// A small DeepSeek-V4.1 (tests/fixtures/deepseek_v41_tiny_config.json: six layers at DeepSeek-V4's kernels' widths, ratios 0, 2, 2, 1, 1, 1 sharing their
/// compressors, indexers and candidate blocks, Engram before layers 1 and 4, lagged mHC over four streams, one DSpark layer): every tensor of its description at
/// its shape, at random from a fixed seed (the matrices about U(-a, a), a = sqrt(3 / cols); the norms' weights about 1; the mixing projections small), bf16 but
/// the mixers' and the sinks' and choice biases' fp32 as released; and the compressed vocabulary Engram hashes (`engram_token_map.json`: a token's id mod its size).
pub fn deepseek_v41_random(out: &Path) -> Result<()> {
    let text = include_str!("../tests/fixtures/deepseek_v41_tiny_config.json");
    let cfg: Value = serde_json::from_str(text).map_err(|e| Error(e.to_string()))?;
    let d = crate::runtime::describe::describe(&cfg)?;
    let mut r = 0x2026_1124_0000_0041u64;
    let mut uni = move |a: f32, b: f32| {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        a + (b - a) * ((r >> 40) as f32 / (1u64 << 24) as f32)
    };
    let bf = |x: f32| -> u16 {
        let b = x.to_bits();
        ((b.wrapping_add(0x7fff + ((b >> 16) & 1))) >> 16) as u16
    };
    let mut tensors: Vec<(String, &str, Vec<u64>, Vec<u8>)> = Vec::new();
    for name in d.names() {
        let shape = d.shapes.get(&name).cloned().ok_or_else(|| Error(format!("DeepSeek-V4.1's description has no shape for {name}")))?;
        let n: usize = shape.iter().product();
        let cols = *shape.last().unwrap_or(&1) as f32;
        let leaf = name.rsplit('.').next().unwrap_or("");
        let (lo, hi) = if name.contains("hc_") && name.ends_with("_fn") {
            (-0.02, 0.02)
        } else if name.contains("hc_") && name.ends_with("_scale") {
            (0.5, 1.5)
        } else if name.contains("hc_") || name.ends_with("attn_sink") || name.ends_with("gate.bias") {
            (-0.5, 0.5)
        } else if name.contains("norm") || name.ends_with("q_weight") || name.ends_with("k_weight") {
            (0.9, 1.1)
        } else if name == d.embed || name.ends_with("engram.embed.weight") {
            (-1.0, 1.0)
        } else {
            let a = (3.0 / cols).sqrt();
            (-a, a)
        };
        let vals: Vec<f32> = (0..n).map(|_| uni(lo, hi)).collect();
        let f32s = name.contains("hc_") || name.ends_with("attn_sink") || name.ends_with("gate.bias") || leaf == "q_weight" || leaf == "k_weight";
        let bytes = if f32s { vals.iter().flat_map(|x| x.to_le_bytes()).collect() } else { vals.iter().flat_map(|&x| bf(x).to_le_bytes()).collect() };
        tensors.push((name, if f32s { "F32" } else { "BF16" }, shape.iter().map(|&x| x as u64).collect(), bytes));
    }
    io(std::fs::create_dir_all(out))?;
    io(std::fs::write(out.join("config.json"), text))?;
    let (vocab, map) = (d.desc.vocab, cfg["text_config"]["engram_compressed_vocab_size"].as_u64().unwrap_or(1) as usize);
    io(std::fs::write(out.join("engram_token_map.json"), serde_json::to_string(&(0..vocab).map(|t| t % map).collect::<Vec<_>>()).unwrap_or_default()))?;
    let refs: Vec<(&str, &str, Vec<u64>, &[u8])> = tensors.iter().map(|(n, t, s, b)| (n.as_str(), *t, s.clone(), &b[..])).collect();
    crate::pack::write_safetensors(&out.join("model.safetensors"), &[("format", "pt".to_string())], &refs)?;
    println!("{}: a random DeepSeek-V4.1, {} tensors, {:.0} MB", out.display(), tensors.len(), tensors.iter().map(|t| t.3.len()).sum::<usize>() as f64 / 1e6);
    Ok(())
}

/// The compressed vocabulary DeepSeek-V4.1's Engram hashes (`kinds::NgramForm::Engram`): the model directory's `engram_token_map.json` (a list, the compressed id of
/// each token), which the loader makes from the tokenizer; None where the directory has none.
pub fn engram_token_map(dir: &Path) -> Result<Option<Vec<f32>>> {
    let p = dir.join("engram_token_map.json");
    let Ok(text) = std::fs::read_to_string(&p) else { return Ok(None) };
    let v: Vec<u32> = serde_json::from_str(&text).map_err(|e| Error(format!("{}: {e}", p.display())))?;
    Ok(Some(v.into_iter().map(|x| x as f32).collect()))
}

fn qwen35_engine(dir: &Path, level: &str) -> Result<Box<dyn Run>> {
    boxed(crate::dynamic::qwen35::load(dir, &dyn_opts(level_layout(level)?)))
}

/// A plain dense model on the executor (src/dynamic/dense.rs): its description lowered from the config, packed here at the level (penguin: the GPU's lossless layout).
fn dense_engine(dir: &Path, level: &str) -> Result<Box<dyn Run>> {
    boxed(crate::dynamic::dense::load(dir, &dyn_opts(level_layout(level)?)))
}

/// Every case: a model of the family's layout with its layer kinds, at random (a fixed seed) and small: only models released in the last six months are tested, and those that are recent are larger than a 16 GB card holds.
pub fn cases() -> Vec<Case> {
    vec![
        Case {
            id: "dense",
            family: "the plain dense kind (a pre-norm decoder of one attention and one gated MLP a layer, on the executor)",
            weights: Weights::Random(dense_random),
            engine: dense_engine,
            cpu: Some(described_cpu),
            vocab: 4096,
        },
        Case {
            id: "qwen35",
            family: "Qwen3.5 family (Gated DeltaNet and gated attention: Qwen3.8-27B's layout)",
            weights: Weights::Random(qwen35_random),
            engine: qwen35_engine,
            cpu: Some(described_cpu),
            vocab: 4096,
        },
        Case {
            id: "qwen35-wide",
            family: "Qwen3.5 family, the wide shape (three value heads to a key head, a head of its own, the text model of a checkpoint with a vision tower)",
            weights: Weights::Random(qwen35_wide_random),
            engine: qwen35_engine,
            cpu: Some(described_cpu),
            vocab: 4096,
        },
        Case {
            id: "qwen35moe",
            family: "Qwen3.5 family with experts (Gated DeltaNet, gated attention, softmax-routed experts with a gated shared expert)",
            weights: Weights::Random(qwen35moe_random),
            engine: qwen35_engine,
            cpu: Some(described_cpu),
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
            id: "kimi-dsa",
            family: "Kimi-K3 family with the newer config's sparse attention (latent attention over the tokens its indexer picks, a key a token, RMS-normed)",
            weights: Weights::Random(crate::kimi_ref::random_dsa),
            engine: kimi_engine,
            cpu: Some(described_cpu),
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
            cpu: Some(described_cpu),
            vocab: 4096,
        },
        Case {
            id: "gemma4-kv",
            family: "Gemma 4 E2B/E4B layers (KV sharing across layers, the double-wide MLP)",
            weights: Weights::Random(gemma4_kv_random),
            engine: gemma4_engine,
            cpu: Some(described_cpu),
            vocab: 4096,
        },
        Case {
            id: "qwen4x",
            family: "Qwen3.8-Flash-Next (Gated DeltaNet, indexed attention, hyper-connections, per-layer embeddings, mixture of experts)",
            weights: Weights::Random(qwen4x_random),
            engine: qwen4x_engine,
            cpu: Some(described_cpu),
            vocab: 2048,
        },
        Case {
            id: "nemotron-h",
            family: "Nemotron-H (Mamba-2, attention, mixture of experts)",
            weights: Weights::Random(|d| crate::kcheck_nh::random(d, 0, 16)),
            engine: nemotron_engine,
            cpu: Some(described_cpu),
            vocab: 4096,
        },
        Case {
            id: "deepseek-v41",
            family: "DeepSeek-V4.1 (CSA2's shared compressed attention, lagged mHC, Engram, mixture of experts)",
            weights: Weights::Random(deepseek_v41_random),
            engine: |_, _| err(format!("DeepSeek-V4.1 is {NOT_ON_EXECUTOR}: its CPU reference alone (logitscheck --cpu-ref)")),
            cpu: Some(described_cpu),
            vocab: 4096,
        },
        Case {
            id: "glm",
            family: "GLM-5.3 (delta attention, sparse attention, mixture of experts)",
            weights: Weights::Random(|d| crate::glm::random(d, false, None)),
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
        .map_err(|e| Error(format!("{}: {}", dir.display(), e.msg)))?
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
                    match (c.engine)(&dir, &lb.level) {
                        // (a family whose CPU reference stands before its engine: its fixture is the reference's, checked with --cpu-ref)
                        Err(e) if e.msg.contains(NOT_ON_EXECUTOR) => {
                            println!("SKIP {:12} {}: {}", c.id, lb.level, e.msg);
                            continue;
                        }
                        r => r?,
                    }
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

/// The cases whose families still have a hand-written engine: (case, the hand-written engine, the description path).
type Pair = (&'static str, fn(&Path, &str) -> Result<Box<dyn Run>>, fn(&Path, &str) -> Result<Box<dyn Run>>);

fn legacy_pairs() -> Vec<Pair> {
    // (none: R3b retired Qwen3.5's, Gemma 4's and Nemotron-H's hand-written engines on their parity records, #410 and the R3b pull requests)
    Vec::new()
}

/// `paritycheck [CASE...] [--level L] [--scratch DIR]`: each family that still has a hand-written engine (none since R3b) on
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
                    println!("SKIP {id:12} {level:8} the hand-written engine does not run it: {}", e.msg);
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

/// The hummingbird level's KL thresholds from the lossless one (mean, max over rows), per mixture-of-experts family's random model: the
/// model's routed experts coded at 3 and 2 bits by use (src/hb_pack.rs) from a corpus pass over made-up documents (src/hb_usage.rs,
/// `synthetic`), every other matrix as released. Each is twice its family's first measured run, to two figures, as Gemma 4's above
/// (A10, 2026-11-25, the same figures in two runs: mean/max KL DeepSeek-V4 9.32e-2/1.30e-1, Muse Spark 2.10e-2/2.69e-2).
pub const LEVEL_KL_HB: [(&str, f64, f64); 2] = [("deepseek-v4", 0.19, 0.26), ("muse", 0.042, 0.054)];

/// The mixture-of-experts families that run at hummingbird (src/support.rs) and whose hummingbird directory this tree does not make or
/// load on a random model, with what stops it. The check runs each and holds it to that: one that runs after all fails the check with
/// its figures, to be given a threshold in `LEVEL_KL_HB` and taken out of here.
pub const LEVEL_HB_OPEN: [(&str, &str); 2] = [
    ("kimi", "the corpus pass counts none of its layers (src/dynamic/mod.rs record_routes reads a resident layer's router, which src/dynamic/kimi.rs KimiMoe is not)"),
    ("glm", "its directory does not load (src/glm.rs check_layers wants the experts' matrices, which the directory holds as codes)"),
];

/// The families held at the block levels of every weight (kestrel, swift, wren: `LEVEL_KL`); `LEVEL_KL_HB`'s are held at hummingbird.
const LEVEL_BLOCK_CASES: [&str; 3] = ["qwen35", "gemma4", "nemotron-h"];

/// A mixture-of-experts case's model in `dir` with every matrix as `dir` holds it: the release's lossless, or a hummingbird directory's
/// experts in their codes. (GLM-5.3-Flash's logits case runs its experts at a block level; here they are lossless, as kindcheck loads them.)
fn moe_as_held(id: &str, dir: &Path) -> Result<Box<dyn Run>> {
    match id {
        "glm" => boxed(crate::dynamic::glm::load(dir, &dyn_opts(Some(Layout::Tiered)), Layout::Tiered)),
        _ => (find_case(id)?.engine)(dir, "penguin"),
    }
}

/// The model in `dir` at the hummingbird level in `out`, as its two commands make it (`calibrate-experts --synthetic`, then
/// `pack --layout hummingbird --usage`): each expert's use counted on this GPU over made-up documents, its rows coded from the counts.
fn hummingbird_dir(dir: &Path, out: &Path) -> Result<()> {
    if out.join("glyd.json").exists() {
        return Ok(());
    }
    let (model, usage) = (dir.to_string_lossy(), out.with_extension("usage.json"));
    crate::hb_usage::run_with(&model, &usage, 512, &[0], &[], 256, true)?;
    crate::hb_pack::run(&model, out, std::thread::available_parallelism().map_or(8, |n| n.get()), &usage)
}

/// `levelcheck [CASE...] [--level L] [--scratch DIR]`: each family on its random model at penguin and at each lossy level it runs, the
/// same tokens as its R7 fixture, and the KL of the lossy level's rows from penguin's against the level's threshold: Qwen3.5, Gemma 4 and
/// the state-space case at kestrel, swift and wren (`LEVEL_KL`, `LEVEL_KL_GEMMA4` for Gemma 4); DeepSeek-V4 and Muse Spark at
/// hummingbird (`LEVEL_KL_HB`, the directory made here). Exit 1 where one is past its threshold, or does not load: a level that does
/// not run is not one within its threshold. Kimi-K3 and GLM-5.3-Flash at hummingbird are `LEVEL_HB_OPEN`'s: printed as not held, with
/// what stops each, and a failure the day one runs without a threshold.
pub fn level_check(a: &crate::args::Args) -> Result<()> {
    let want: Vec<&str> = a.pos.iter().skip(1).map(String::as_str).collect();
    let all: Vec<&str> = LEVEL_BLOCK_CASES.iter().copied().chain(LEVEL_KL_HB.iter().map(|x| x.0)).chain(LEVEL_HB_OPEN.iter().map(|x| x.0)).collect();
    let ids: Vec<&str> = all.iter().copied().filter(|c| want.is_empty() || want.contains(c)).collect();
    if !want.is_empty() && ids.len() != want.len() {
        return err(format!("cases: {}", all.join(", ")));
    }
    let (fixtures, scratch, only_level) = (fixtures_dir(a), Scratch::new(a)?, a.flag("level"));
    let (mut failed, mut not_held) = (Vec::new(), Vec::new());
    for id in ids {
        let open = LEVEL_HB_OPEN.iter().find(|x| x.0 == id).map(|x| x.1);
        let hb = LEVEL_KL_HB.iter().find(|x| x.0 == id).map(|x| (x.1, x.2)).or(open.map(|_| (f64::NAN, f64::NAN)));
        let levels: Vec<(&str, f64, f64)> = match hb {
            Some((mean, max)) => vec![("hummingbird", mean, max)],
            None if id == "gemma4" => LEVEL_KL_GEMMA4.to_vec(),
            None => LEVEL_KL.to_vec(),
        };
        let levels: Vec<(&str, f64, f64)> = levels.into_iter().filter(|l| only_level.is_none_or(|o| o == l.0)).collect();
        if levels.is_empty() {
            continue;
        }
        let c = find_case(id)?;
        let dir = model_dir(&c, &scratch.dir)?;
        let fx = Fixture::read(&fixture_files(&fixtures, id)?[0])?;
        let engine = c.engine;
        let lossless = if hb.is_some() { moe_as_held(id, &dir) } else { engine(&dir, "penguin") };
        let mut base = match lossless {
            Ok(r) => r,
            Err(e) => {
                println!("FAIL {id:12} penguin, the base, does not load: {}", e.msg);
                failed.push(format!("{id} at penguin"));
                continue;
            }
        };
        let base_rows = drive(base.as_mut(), &fx.prompt, &fx.decode, &fx.long)?;
        drop(base);
        for (level, mean_bound, max_bound) in levels {
            let lossy = if hb.is_some() {
                let out = scratch.dir.join(format!("{id}-hummingbird"));
                hummingbird_dir(&dir, &out).and_then(|()| moe_as_held(id, &out))
            } else {
                engine(&dir, level)
            };
            let mut run = match (lossy, open) {
                (Ok(r), _) => r,
                (Err(e), Some(why)) => {
                    println!("OPEN {id:12} {level:8} not held: {why}. It said: {}", e.msg.chars().take(200).collect::<String>());
                    not_held.push(format!("{id} at {level}"));
                    continue;
                }
                (Err(e), None) => {
                    println!("FAIL {id:12} {level:8} does not load: {}", e.msg);
                    failed.push(format!("{id} at {level}"));
                    continue;
                }
            };
            let rows = drive(run.as_mut(), &fx.prompt, &fx.decode, &fx.long)?;
            let kls: Vec<f64> = base_rows.iter().zip(&rows).map(|(b, r)| Row::from_logits(b).map(|b| kl_of(&b, r))).collect::<Result<_>>()?;
            let (mean, max) = (kls.iter().sum::<f64>() / kls.len() as f64, kls.iter().cloned().fold(0f64, f64::max));
            if open.is_some() {
                println!(
                    "FAIL {id:12} {level:8} {:3} rows: KL from penguin mean {mean:.2e}, max {max:.2e}: it runs now, so it takes a threshold in LEVEL_KL_HB and leaves LEVEL_HB_OPEN",
                    kls.len()
                );
                failed.push(format!("{id} at {level}"));
                continue;
            }
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
        let open = if not_held.is_empty() { String::new() } else { format!("; not held: {}", not_held.join(", ")) };
        println!("every lossy level held is within its KL threshold{open}");
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
        Weights::Random(make) => {
            let dir = Path::new(a.need(2, "DIR")?);
            make(dir)?;
            // (`--servable`: the files a server needs besides the weights, for a client to talk to it: a tokenizer, the chat template and the end token; the Qwen3.8 layout's, which is the one with a template made)
            if a.has("servable") {
                if c.id != "qwen35" {
                    return err(format!("--servable: only the qwen35 case has a chat template made (not {})", c.id));
                }
                write_serving_files(dir, include_str!("../tests/fixtures/chat_qwen38_tools.jinja"))?;
            }
            Ok(())
        }
        Weights::Hub(id) => err(format!("{}: a real model ({id}), not a random one", c.id)),
    }
}

/// The tokens of a servable random model: Qwen's own special ones in the places the random config's end token (id 2) wants them, then every printable ASCII character and the new line and tab, one each. A text with
/// another character is the unknown token's. A reply of such a model is random characters, and a grammar can spell JSON and a tool call in them.
fn tiny_tokens() -> Vec<(String, bool)> {
    let mut t: Vec<(String, bool)> = [
        ("<unk>", false),
        ("<|im_start|>", true),
        ("<|im_end|>", true),
        ("<|endoftext|>", true),
        ("<think>", false),
        ("</think>", false),
        ("<tool_call>", false),
        ("</tool_call>", false),
    ]
    .iter()
    .map(|(n, s)| (n.to_string(), *s))
    .collect();
    t.push(("\n".into(), false));
    t.push(("\t".into(), false));
    t.extend((32u8..127).map(|b| ((b as char).to_string(), false)));
    t
}

/// A random model's directory made servable: `tokenizer.json` (a character-level one over `tiny_tokens`), `tokenizer_config.json` with `template` as its chat template and `<|im_end|>` as its end token, and
/// `generation_config.json` (the end token's id).
pub fn write_serving_files(dir: &Path, template: &str) -> Result<()> {
    let io = |r: std::io::Result<()>| r.map_err(|e| Error(format!("{}: {e}", dir.display())));
    let tokens = tiny_tokens();
    let vocab: serde_json::Map<String, Value> = tokens.iter().enumerate().map(|(i, (n, _))| (n.clone(), json!(i))).collect();
    let added: Vec<Value> = tokens
        .iter()
        .enumerate()
        .filter(|(i, (n, _))| *i > 0 && n.starts_with('<') && n.len() > 1)
        .map(
            |(i, (n, special))| json!({"id": i, "content": n, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": special}),
        )
        .collect();
    let tok = json!({"version": "1.0", "truncation": null, "padding": null, "added_tokens": added, "normalizer": null,
        "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "[\\s\\S]"}, "behavior": "Isolated", "invert": false}, "post_processor": null, "decoder": {"type": "Fuse"},
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"}});
    io(std::fs::write(dir.join("tokenizer.json"), tok.to_string()))?;
    let config = json!({"chat_template": template, "eos_token": "<|im_end|>", "bos_token": null, "unk_token": "<unk>", "tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 4096});
    io(std::fs::write(dir.join("tokenizer_config.json"), config.to_string()))?;
    io(std::fs::write(dir.join("generation_config.json"), json!({"eos_token_id": 2, "do_sample": true, "temperature": 0.7, "top_p": 0.9}).to_string()))
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
            // (Nemotron-H has no reference of its own: its description on the reference of the layer kinds)
"cpu" if matches!(c.id, "nemotron-h" | "qwen4x" | "deepseek-v41" | "qwen35" | "qwen35-wide" | "dense" | "qwen35moe" | "kimi-dsa") => "the model's description (src/runtime/describe.rs) on the CPU reference of the layer kinds (src/runtime/reference), fp32 inside, on this case's random model (seed fixed in the generator)".to_string(),
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

    /// The CPU reference of each Gemma 4 random case against the fixtures the engine and the reference recorded, row by row: the plain case's against the
    /// engine's own bits (gemma4.sm89.json.gz), the KV-sharing case's against its fixture. Minutes on an unoptimized build: cargo test -- --ignored
    /// gemma4_reference_against.
    #[cfg(feature = "dev")]
    #[test]
    #[ignore]
    fn gemma4_reference_against_the_fixtures() {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/logits");
        for (case, file, make) in
            [("gemma4", "gemma4.sm89.json.gz", gemma4_random as fn(&Path) -> Result<()>), ("gemma4-kv", "gemma4-kv.json.gz", gemma4_kv_random)]
        {
            let dir = std::env::temp_dir().join(format!("glyd-g4ref-{case}-{}", std::process::id()));
            make(&dir).unwrap();
            let fx = Fixture::read(&fixtures.join(file)).unwrap();
            let got = drive(described_cpu(&dir).unwrap().as_mut(), &fx.prompt, &fx.decode, &fx.long).unwrap();
            let _ = std::fs::remove_dir_all(&dir);
            let kls: Vec<String> = fx.rows.iter().zip(&got).map(|(w, g)| format!("{:.1e}", kl_of(w, g))).collect();
            let v = compare(&fx, &Bound::Kl { mean: 1e-4, max: 2.5e-4, top1: 0.9 }, &got);
            println!(
                "{case} reference against {file}: KL mean {:.2e} max {:.2e}, best token {}/{}; rows {}",
                v.mean_kl,
                v.max_kl,
                v.top1.0,
                v.top1.1,
                kls.join(" ")
            );
            // (the reference with each norm's output rounded to bf16, as transformers' Gemma4RMSNorm casts it back to a bf16 input's type and the engine holds it)
            let once = |d: &mut crate::runtime::describe::Described| {
                use crate::runtime::kinds::Rounding;
                for b in d.desc.blocks.iter_mut() {
                    b.norm_in.rounding = Rounding::Once;
                    if let Some(n) = b.norm_out.as_mut() {
                        n.rounding = Rounding::Once;
                    }
                }
                d.desc.final_norm.rounding = Rounding::Once;
            };
            make(&dir).unwrap();
            let got2 = drive(described_cpu_with(&dir, once).unwrap().as_mut(), &fx.prompt, &fx.decode, &fx.long).unwrap();
            let _ = std::fs::remove_dir_all(&dir);
            let v = compare(&fx, &Bound::Kl { mean: 1e-4, max: 2.5e-4, top1: 0.9 }, &got2);
            println!("{case} reference, its norms' outputs rounded to bf16, against {file}: KL mean {:.2e} max {:.2e}", v.mean_kl, v.max_kl);
            // (the same logits through the engine's head arithmetic: the product's output bf16, then its soft-cap's three roundings, kernels/gemma4.cu g4_softcap)
            let b = crate::runtime::reference::bf16;
            let cap = 30.0f32;
            let r16: Vec<Vec<f32>> = got
                .iter()
                .map(|r| r.iter().map(|&l| cap * (l / cap).clamp(-1.0 + 1e-7, 1.0 - 1e-7).atanh()).map(|z| b(b(b(b(z) / cap).tanh()) * cap)).collect())
                .collect();
            let v = compare(&fx, &Bound::Kl { mean: 1e-4, max: 2.5e-4, top1: 0.9 }, &r16);
            println!("{case} reference, its head rounded as the engine's, against {file}: KL mean {:.2e} max {:.2e}", v.mean_kl, v.max_kl);
        }
    }

    /// The CPU's candidates usercheck holds the step's to: the largest first, equal logits by the smaller token, as the sampler's `sample_top` takes them.
    #[test]
    fn host_top_orders_by_logit_then_token() {
        let row = [0.5f32, 2.0, -1.0, 2.0, 7.0, f32::NEG_INFINITY, 0.5, 2.0];
        assert_eq!(host_top(&row, 5), vec![(4, 7.0), (1, 2.0), (3, 2.0), (7, 2.0), (0, 0.5)]);
        assert_eq!(host_top(&row, 64).len(), row.len());
        assert_eq!(host_top(&row, 64).last(), Some(&(5, f32::NEG_INFINITY)));
    }

    /// A step that refuses a chunk past its `chunk` tokens or not at the slot's place, as the executor's does, and makes its candidates from its rows' logits.
    struct ChunkedStep {
        chunk: usize,
        held: Vec<usize>,
        rows: usize,
    }

    impl crate::sched::ModelStep for ChunkedStep {
        fn window(&self) -> usize {
            1024
        }
        fn chunk(&self) -> usize {
            self.chunk
        }
        fn pages(&self) -> usize {
            0
        }
        fn slots(&self) -> Option<usize> {
            Some(self.held.len())
        }
        fn eos(&self) -> Vec<i64> {
            Vec::new()
        }
        fn prefill(&mut self, pages: &[i32], start: usize, tokens: &[u32], last: bool) -> Result<Option<crate::sched::Sample>> {
            let s = pages[0] as usize;
            if tokens.len() > self.chunk || start != self.held[s] {
                return err(format!("a chunk of {} tokens after {start}, where a chunk holds {} and the slot {}", tokens.len(), self.chunk, self.held[s]));
            }
            self.held[s] += tokens.len();
            self.rows = 1;
            Ok(last.then_some(crate::sched::Sample { token: 7, logit: 0.0, logprob: None }))
        }
        fn decode(&mut self, rows: &[crate::sched::Row]) -> Result<Vec<crate::sched::Sample>> {
            self.rows = rows.len();
            Ok(rows.iter().map(|_| crate::sched::Sample { token: 7, logit: 0.0, logprob: None }).collect())
        }
        fn logits(&mut self, row: usize) -> Result<Vec<f32>> {
            Ok(logits(128, row as u32))
        }
        fn top(&mut self, row: usize) -> Option<Vec<(u32, f32)>> {
            (row < self.rows).then(|| host_top(&logits(128, row as u32), crate::kern::TOPK))
        }
        fn release(&mut self, pages: &[i32]) -> Result<()> {
            pages.iter().for_each(|&p| self.held[p as usize] = 0);
            Ok(())
        }
    }

    /// usercheck's candidates' check runs each prompt in the step's chunks (a 92-token prompt where a chunk holds 64 was one pass, which the executor refuses).
    #[test]
    fn the_candidates_check_runs_prompts_in_chunks() {
        let mut step = ChunkedStep { chunk: 64, held: vec![0; 4], rows: 0 };
        let prompts: Vec<Vec<u32>> = [5usize, 64, 92, 200].iter().map(|&n| (0..n as u32).collect()).collect();
        assert_eq!(cands_check(&mut step, &prompts).unwrap(), (8, 0));
        assert!(step.held.iter().all(|&h| h == 0), "the slots given back");
    }

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

/// The case's model in `dir` on the description executor.
pub(crate) fn load_dyn(case: &str, dir: &Path, o: &DynOpts, level: &str) -> Result<Dyn> {
    Ok(match case {
        "gemma4" | "gemma4-kv" => {
            let c = crate::gemma4::ConfigG4::load(dir)?;
            Dyn::new(dir, crate::dynamic::gemma4::lowered(dir, &c)?, None, o)?
        }
        "qwen35" | "qwen35-wide" | "qwen35moe" => crate::dynamic::qwen35::load(dir, o)?,
        "dense" => crate::dynamic::dense::load(dir, o)?,
        "muse" => crate::dynamic::muse::load(dir, o)?,
        "kimi" | "kimi-dsa" => crate::dynamic::kimi::load(dir, o)?,
        // (its experts at a lossy level, as its logits case: kestrel where penguin is asked)
        "glm" => crate::dynamic::glm::load(dir, &DynOpts { layout: Some(Layout::Tiered), ..o.clone() }, level_layout(level)?.unwrap_or(Layout::Int6))?,
        // (its experts at the level asked: the lossless one where none is)
        "nemotron" => crate::dynamic::nemotron::load(dir, o, level_layout(level)?.unwrap_or(Layout::Tiered))?,
        "deepseek-v4" => crate::dynamic::deepseek_v4::load(dir, o)?,
        "qwen4x" => crate::dynamic::qwen4x::load(dir, o)?,
        other => {
            return err(format!(
                "{other}: gemma4, gemma4-kv, qwen35, qwen35-wide, qwen35moe, dense, muse, kimi, kimi-dsa, glm, nemotron, deepseek-v4 or qwen4x"
            ))
        }
    })
}

/// The case's random model, made in a directory of its own and loaded.
pub(crate) fn random_dyn(case: &str, level: &str, tag: &str) -> Result<Dyn> {
    random_dyn_at(case, level, level_layout(level)?, tag, None)
}

/// `random_dyn` with its packs in `layout` (None: the GPU's choice) and a context of `ctx` tokens (None: the cases' 1,024).
pub(crate) fn random_dyn_at(case: &str, level: &str, layout: Option<Layout>, tag: &str, ctx: Option<usize>) -> Result<Dyn> {
    let dir = std::env::temp_dir().join(format!("glyd-{tag}-{case}-{}", std::process::id()));
    let o = DynOpts { ctx: ctx.unwrap_or(1024), ..dyn_opts(layout) };
    let e: Result<Dyn> = (|| {
        random_model(case, &dir)?;
        load_dyn(case, &dir, &o, level)
    })();
    let _ = std::fs::remove_dir_all(&dir);
    e
}

/// The case's random model written to `dir`.
fn random_model(case: &str, dir: &Path) -> Result<()> {
    match case {
        "gemma4" => gemma4_random(dir),
        "gemma4-kv" => gemma4_kv_random(dir),
        "qwen35" => qwen35_random(dir),
        "qwen35-wide" => qwen35_wide_random(dir),
        "qwen35moe" => qwen35moe_random(dir),
        "dense" => dense_random(dir),
        "muse" => crate::muse_ref::random(dir),
        "kimi" => crate::kimi_ref::random(dir, 1, true, 8),
        "kimi-dsa" => crate::kimi_ref::random_dsa(dir),
        "glm" => crate::glm::random(dir, false, None),
        "nemotron" => crate::kcheck_nh::random(dir, 0, 16),
        "deepseek-v4" => crate::dsv4_ref::random(dir, None),
        "qwen4x" => qwen4x_random(dir),
        other => err(format!("{other}: gemma4, gemma4-kv, qwen35, qwen35-wide, qwen35moe, dense, muse, kimi, kimi-dsa, glm, nemotron, deepseek-v4 or qwen4x")),
    }
}

/// The context the planner gives the case's random model on this GPU for one user (`serve`'s plan of it), or why it gives none.
fn planned_context(case: &str) -> Result<usize> {
    let dir = std::env::temp_dir().join(format!("glyd-planned-{case}-{}", std::process::id()));
    let got = (|| -> Result<usize> {
        random_model(case, &dir)?;
        let words = ["serve".to_string(), dir.display().to_string(), "--users".into(), "1".into(), "--no-draft".into()];
        let f = crate::plan::planned(&crate::args::Args::from(words)).map_err(|r| Error(format!("the planner: {}", r.what)))?;
        f.plan.chosen.as_ref().map(|c| c.context).filter(|&c| c > 0).ok_or_else(|| Error(format!("the planner has no plan for the random {case} model")))
    })();
    let _ = std::fs::remove_dir_all(&dir);
    got
}

/// `splitcheck CASE [--split CUT,..] [--devices GPU,..] [--level L | --layout mma12] [--tokens T]` (development): the case's random model on the dynamic route as `run` opens it
/// (`split::open`: its family's loader, any family of that route), once on one GPU and once with its layers cut into stages (by default before its middle
/// layer, both stages on GPU 0: the hand-over through host memory checked on a machine with one GPU). A prompt of 24 tokens, then T greedy steps fed the
/// one-GPU run's tokens: every pass's logits the same bits both ways.
pub fn split_check(a: &crate::args::Args) -> Result<()> {
    let case = a.need(1, "CASE")?;
    let level = a.flag("level").unwrap_or("penguin");
    let tokens: usize = a.num("tokens", 12usize)?;
    let Weights::Random(make) = find_case(case)?.weights else { return err(format!("{case}: a real model, not a random one")) };
    let dir = std::env::temp_dir().join(format!("glyd-splitcheck-{case}-{}", std::process::id()));
    let r = (|| -> Result<()> {
        make(&dir)?;
        let text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| Error(e.to_string()))?;
        if crate::dynamic::split::family(&text).is_none() {
            return err(format!("{case}: its family is not on the dynamic route (`run` cuts no layers of it)"));
        }
        let config = |k: &str| crate::dynamic::split::config_num(&text, k).unwrap_or(0);
        let (nl, vocab) = (config("num_hidden_layers"), config("vocab_size"));
        let (cuts, devices) = match crate::dynamic::split::split_args(a)? {
            (c, _) if c.is_empty() => (vec![nl / 2], vec![0, 0]),
            x => x,
        };
        // (penguin's packs in the smallest layout, or the 12-bit one with --layout mma12, as an HBM GPU's plan holds them)
        let o = dyn_opts(a.layout()?.or(level_layout(level)?).or(Some(Layout::Tiered)));
        let prompt = token_list(24, vocab, 0);
        // (each pass's logits, the steps fed `feed` where given, else the run's own greedy tokens)
        let run = |cuts: &[usize], devices: &[i32], feed: Option<&[u32]>| -> Result<(Vec<Vec<f32>>, Vec<u32>)> {
            let mut e = crate::dynamic::split::open(&dir, &o, cuts, devices)?;
            let mut s = e.prefill(&prompt)?;
            let (mut rows, mut fed) = (vec![e.logits()?], Vec::new());
            for i in 0..tokens {
                let t = feed.map_or(s.token, |f| f[i]);
                fed.push(t);
                s = e.step(t)?;
                rows.push(e.logits()?);
            }
            Ok((rows, fed))
        };
        let (one, fed) = run(&[], &devices[..1], None)?;
        let (staged, _) = run(&cuts, &devices, Some(&fed))?;
        let differ: Vec<usize> = (0..one.len()).filter(|&i| one[i].iter().map(|x| x.to_bits()).ne(staged[i].iter().map(|x| x.to_bits()))).collect();
        println!(
            "{case} at {level}: {nl} layers, cut before {cuts:?} on GPUs {devices:?}; {} passes (a prompt of {}, {tokens} steps), {} with logits that differ from the one-GPU run's",
            one.len(),
            prompt.len(),
            differ.len()
        );
        if !differ.is_empty() {
            return err(format!("{case}: the passes {differ:?} differ between one GPU and the stages"));
        }
        Ok(())
    })();
    let _ = std::fs::remove_dir_all(&dir);
    r
}

/// `twomodels [A B] [--level L] [--tokens T]` (R16: two models in one process): two cases' random models (gemma4 and qwen35 unless named) loaded on this GPU in this process, each with a scheduler
/// in a runtime context of its own. Each of three prompts is replied to by each model alone (`reply`: the reference), then by both at once, their schedulers' rounds one after the other on
/// this thread, so that a step of one model runs between two steps of the other. Every reply is its reference, token for token, and each context counts its own model's prompt tokens and
/// not the other's.
pub fn two_models_check(a: &crate::args::Args) -> Result<()> {
    use crate::conv::Gen;
    use crate::runtime::context::Context;
    use crate::sched::{Ev, Finish, Reply, Sched, Submit};
    use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
    use std::sync::Arc;
    let (ca, cb) = (a.pos.get(1).map_or("gemma4", String::as_str), a.pos.get(2).map_or("qwen35", String::as_str));
    let level = a.flag("level").unwrap_or("penguin");
    let tokens: usize = a.num("tokens", 16usize)?;
    let (mut ea, mut eb) = (random_dyn(ca, level, "twomodels-a")?, random_dyn(cb, level, "twomodels-b")?);
    let reply = Reply {
        params: crate::sampler::Params { temperature: 0.0, ..Default::default() },
        max_new: tokens,
        ignore_eos: true,
        logprobs: None,
        stop_ids: Vec::new(),
    };
    let prompts = |vocab: u32, seed: u32| -> Vec<Vec<u32>> {
        (0..3u32).map(|k| (0..7 + 23 * k).map(|i| 3 + (i * 7919 + (k + seed) * 104_729) % (vocab - 3)).collect()).collect()
    };
    let (pa, pb) = (prompts(ea.desc().vocab as u32, 1), prompts(eb.desc().vocab as u32, 5));
    let alone = |e: &mut Dyn, ps: &[Vec<u32>]| -> Result<Vec<Vec<u32>>> { ps.iter().map(|p| e.reply(p, &reply, &mut |_| true).map(|r| r.0)).collect() };
    let (ra, rb) = (alone(&mut ea, &pa)?, alone(&mut eb, &pb)?);
    let (Some((_, sa)), Some((_, sb))) = (ea.as_step(1)?, eb.as_step(1)?) else { return err(format!("{ca} or {cb}: no scheduler's step")) };
    let (rt_a, rt_b) = (Context::new(), Context::new());
    let before = Context::process().sched[0].load(Relaxed);
    let (mut sa, mut sb) = (Sched::new(sa, 1).in_context(rt_a.clone()), Sched::new(sb, 1).in_context(rt_b.clone()));
    let submit = |s: &mut Sched<&mut dyn crate::sched::ModelStep>, p: &[u32]| {
        let (tx, rx) = std::sync::mpsc::channel();
        s.submit(Submit {
            prompt: p.to_vec(),
            reply: reply.clone(),
            out: tx,
            abort: Arc::new(AtomicBool::new(false)),
            resumed: Default::default(),
            cached: Default::default(),
            created: Default::default(),
            salt: 0,
            pin: 0,
            images: Vec::new(),
        });
        rx
    };
    let collect = |r: &std::sync::mpsc::Receiver<Ev>| -> Result<Vec<u32>> {
        let mut out = Vec::new();
        loop {
            match r.recv().map_err(|_| Error("a scheduler dropped a sequence".into()))? {
                Ev::Token(t) => out.push(t),
                Ev::Lp(_) => {}
                Ev::End(Ok(Finish::Length)) => return Ok(out),
                Ev::End(f) => return err(format!("a sequence ended {f:?} before its {tokens} tokens")),
            }
        }
    };
    let (mut fails, mut rounds) = (0, [0usize; 2]);
    for k in 0..3 {
        let (xa, xb) = (submit(&mut sa, &pa[k]), submit(&mut sb, &pb[k]));
        // (a round of one model, then a round of the other, until both are done)
        while !sa.idle() || !sb.idle() {
            if !sa.idle() {
                sa.tick()?;
                rounds[0] += 1;
            }
            if !sb.idle() {
                sb.tick()?;
                rounds[1] += 1;
            }
        }
        for (name, got, want) in [(ca, collect(&xa)?, &ra[k]), (cb, collect(&xb)?, &rb[k])] {
            let same = got == *want;
            fails += !same as usize;
            println!(
                "{} {name}, prompt {k} beside the other model: {}",
                if same { "PASS" } else { "FAIL" },
                if same { "the reply alone, token for token".into() } else { format!("{got:?} against {want:?}") }
            );
        }
    }
    // each context counts its own model's prompts
    let total = |ps: &[Vec<u32>]| ps.iter().map(Vec::len).sum::<usize>() as u64;
    let (na, nb) = (rt_a.sched[0].load(Relaxed), rt_b.sched[0].load(Relaxed));
    let own = na == total(&pa) && nb == total(&pb) && Context::process().sched[0].load(Relaxed) == before;
    println!(
        "{} the contexts' figures: {na} prompt tokens of {ca} ({} sent), {nb} of {cb} ({} sent); rounds {} and {}",
        if own { "PASS" } else { "FAIL" },
        total(&pa),
        total(&pb),
        rounds[0],
        rounds[1]
    );
    fails += !own as usize;
    if fails > 0 {
        return err(format!("twomodels {ca} {cb}: {fails} failed"));
    }
    println!("twomodels {ca} {cb}: PASS");
    Ok(())
}

/// One logits row against the reference's: the logits that differ, the largest difference, KL(reference || this) of their softmaxes, and whether the largest logit is the same token's.
fn row_diff(want: &[f32], got: &[f32]) -> (usize, f32, f64, bool) {
    let differ = want.iter().zip(got).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    let max = want.iter().zip(got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let (lw, lg) = (lse_of(want), lse_of(got));
    let kl: f64 = want.iter().zip(got).map(|(&a, &b)| ((a as f64 - lw).exp()) * ((a as f64 - lw) - (b as f64 - lg))).sum();
    let top = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|x| x.0);
    (differ, max, kl.max(0.0), top(want) == top(got))
}

/// Room in a step's slot for its sequence's first `upto` tokens, as the scheduler asks for it before a pass.
fn room(s: &mut dyn crate::sched::ModelStep, slot: i32, upto: usize) -> Result<()> {
    if s.grow(slot, upto)? {
        Ok(())
    } else {
        err(format!("no room for {upto} tokens in sequence {slot}"))
    }
}

/// Tokens `t` into a step's slot from position `start` on, alone, `by` tokens a pass; the logits after them where `last`. Room is asked for before every pass, for the tokens
/// the sequence holds after it, as the scheduler asks: a sliding layer's blocks are the window behind the newest token, so room for a whole long prompt at once is not room
/// for its first chunks.
fn fill_from(s: &mut dyn crate::sched::ModelStep, slot: i32, start: usize, t: &[u32], by: usize, last: bool) -> Result<Option<Vec<f32>>> {
    let n = t.len().div_ceil(by);
    for (i, part) in t.chunks(by).enumerate() {
        room(s, slot, start + i * by + part.len() + (i + 1 == n) as usize)?;
        s.prefill(&[slot], start + i * by, part, last && i + 1 == n)?;
    }
    if last {
        s.logits(0).map(Some)
    } else {
        Ok(None)
    }
}

/// `fill_from` from a sequence's first token with its full blocks named after every pass, as the scheduler names them (`keys`: the names of `t`'s full blocks).
fn fill_named(s: &mut dyn crate::sched::ModelStep, slot: i32, t: &[u32], by: usize, keys: &[u128]) -> Result<()> {
    let mut at = 0;
    for part in t.chunks(by) {
        room(s, slot, at + part.len())?;
        s.prefill(&[slot], at, part, false)?;
        let (lo, hi) = (at / crate::kv::PAGE, (at + part.len()) / crate::kv::PAGE);
        s.name(slot, lo, &keys[lo..hi])?;
        at += part.len();
    }
    Ok(())
}

/// A prompt alone in a step's slot from its first token (`fill_from`).
fn fill(s: &mut dyn crate::sched::ModelStep, slot: i32, t: &[u32], by: usize, last: bool) -> Result<Option<Vec<f32>>> {
    fill_from(s, slot, 0, t, by, last)
}

/// `roundcheck CASE [--level L] [--tokens T] [--users N] [--dir MODEL --ctx N --chunk N [--random-tokens]] [--kl BOUND]` (gemma4, qwen35, muse, kimi, glm; R4a's round that is one step): what a sequence's logits are by
/// what its step holds beside it. The case's random model (or the model in MODEL, a real one) as a scheduler's step, and one sequence whose tokens are given, not sampled, so that every way
/// runs the same tokens: its row alone in a step (the reference: a reply alone), with other sequences' rows (decode steps of 2 to 64 rows), and with another sequence's prompt chunk in the
/// same step (`ModelStep::round`), the step 8, 16, 32, 64 rows and on to a chunk's. For each size:
///   - the row's logits against the reference's: the logits that differ (their bits), the largest difference, the KL, whether the largest logit is the same token's;
///   - the row's logits with another prompt of the same length beside it: what a row computes may turn on how many rows its step has (a product's kernel is picked by them), never on what
///     the other sequence holds, so with no mixture of experts (`rows_exact`) not one bit may differ (a row that read another sequence's keys or state would);
///   - the prompt's side: the logits after its last chunk run beside the row, against the same chunks alone; and, as a yardstick of what the kernels' choice by rows alone does on this model,
///     the same prompt alone in chunks of that size against alone in whole chunks.
///   - prompts of different lengths in one pass (`prefill_many`, prompts that come in together): each prompt's logits after its last token and after the step that follows it, against
///     the prompt alone (every bit where the pass is 16 rows or fewer); and the first prompt beside other prompts of the same lengths: with no mixture of experts not a bit may differ.
///
/// Required: in a step of 16 rows or fewer every logit is the reference's, bit for bit (what the kernels give at those sizes on every family); the other sequence's content changes nothing
/// on a model with no mixture of experts; and every KL within --kl where one is given (a real model: a random model's logits are near ties that last bits move far). With
/// GLYD_ENGINE_BATCH_PREFILL=0 (every reply to be the reply alone) only the decode steps are run, and not a bit may differ at any size.
pub fn round_check(a: &crate::args::Args) -> Result<()> {
    use crate::conv::Gen;
    use crate::sched::{ModelStep, PrefillJob, Row};
    let case = a.need(1, "CASE")?;
    let level = a.flag("level").unwrap_or("penguin");
    let steps: usize = a.num("tokens", 8usize)?;
    let bound: Option<f64> = a.flag("kl").map(|b| b.parse().map_err(|_| Error(format!("--kl {b}: a number")))).transpose()?;
    let mut e: Dyn = match a.flag("dir") {
        Some(dir) => {
            let o = DynOpts { ctx: a.num("ctx", 4096usize)?, chunk: a.num("chunk", 512usize)?, graph: true, device: 0, layout: level_layout(level)? };
            load_dyn(case, Path::new(dir), &o, level)?
        }
        None => random_dyn(case, level, "roundcheck")?,
    };
    let (vocab, exact) = (e.desc().vocab as u32, e.rows_exact());
    // (a real model's sequence is a text's tokens, the tokens given after the prompt the text's own next ones: what a reply looks like; the bench's passage, begun again where it ends)
    let text: Option<Vec<u32>> = match a.flag("dir").filter(|_| !a.has("random-tokens")) {
        Some(dir) => {
            let mut t = crate::tok::Tok::load(Path::new(dir))?.encode(crate::bench::PASSAGE)?;
            // (the model's first token before the text, where its config names one: a text as the model was trained to read it)
            let cfg: Option<Value> = std::fs::read_to_string(Path::new(dir).join("config.json")).ok().and_then(|c| serde_json::from_str(&c).ok());
            if let Some(bos) = cfg.as_ref().and_then(|c| c["bos_token_id"].as_u64().or(c["text_config"]["bos_token_id"].as_u64())) {
                t.insert(0, bos as u32);
            }
            if t.len() < 64 {
                return err(format!("the passage is {} tokens of this model's", t.len()));
            }
            Some(t)
        }
        None => None,
    };
    let Some((k, s)) = e.as_step(a.num("users", 64usize)?)? else { return err(format!("{case}: no scheduler's step")) };
    if k < 2 {
        return err(format!("{case}: {k} sequences"));
    }
    // (GLYD_ENGINE_BATCH_PREFILL=0: the round is two passes and every reply is to be the reply alone, so only the steps of rows that decode are run, and at every size not a bit may differ)
    let strict = !s.one_step();
    let (chunk, window) = (s.chunk(), s.window());
    let toks = |n: usize, seed: u32| -> Vec<u32> { (0..n as u32).map(|i| 3 + (i * 7919 + seed * 104_729) % (vocab - 3)).collect() };
    let passage = |n: usize, from: usize| -> Option<Vec<u32>> { text.as_ref().map(|t| (0..n).map(|i| t[(from + i) % t.len()]).collect()) };
    let (pa, fa) = (passage(40, 0).unwrap_or_else(|| toks(40, 1)), passage(steps, 40).unwrap_or_else(|| toks(steps, 2)));
    // the reference: the sequence alone, a row a step
    fill(s, 0, &pa, chunk, false)?;
    let mut reference = Vec::new();
    for (t, &f) in fa.iter().enumerate() {
        room(s, 0, pa.len() + t + 2)?;
        s.decode(&[Row::new(f, pa.len() + t, &[0])])?;
        reference.push(s.logits(0)?);
    }
    s.release(&[0])?;
    let margin = reference
        .iter()
        .map(|l| {
            let mut v = l.clone();
            v.sort_by(|a, b| b.total_cmp(a));
            v[0] - v[1]
        })
        .fold(f32::INFINITY, f32::min);
    println!(
        "{case} at {level}: {steps} steps of a sequence after {} prompt tokens ({}); no mixture of experts: {exact}; a chunk {chunk}, a sequence {window}",
        pa.len(),
        if text.is_some() { "a text's tokens, the bench's passage" } else { "tokens by a rule, not a text's" }
    );
    println!(
        "the reference (the row alone): its two largest logits are {margin:e} apart at the closest step{}",
        text.as_ref().map_or(String::new(), |t| format!("; the text is {} tokens", t.len()))
    );
    let mut fails = 0;
    // (`bits`: not one logit may differ; None: counted)
    let mut report = |what: &str, bits: bool, pairs: &[(&[f32], &[f32])]| {
        let d: Vec<_> = pairs.iter().map(|(w, g)| row_diff(w, g)).collect();
        let (differ, max, kl, top) = (
            d.iter().map(|x| x.0).sum::<usize>(),
            d.iter().map(|x| x.1).fold(0f32, f32::max),
            d.iter().map(|x| x.2).fold(0f64, f64::max),
            d.iter().filter(|x| x.3).count(),
        );
        let mean = d.iter().map(|x| x.2).sum::<f64>() / d.len().max(1) as f64;
        let bad = (bits && differ > 0) || bound.is_some_and(|b| kl > b) || kl.is_nan();
        fails += bad as usize;
        println!(
            "{} {what}: {differ} of {} logits differ, the largest by {max:e}, KL at most {kl:e} (mean {mean:e}), the largest logit the same token's in {top} of {}",
            if bad {
                "FAIL"
            } else if bits || bound.is_some() {
                "PASS"
            } else {
                "    "
            },
            pairs.len() * vocab as usize,
            pairs.len()
        );
    };
    // with other sequences' rows: decode steps of 2, 8, 16, 32 and 64 rows
    for n in [2usize, 8, 16, 32, 48, 63, 64].into_iter().filter(|&n| n <= k) {
        fill(s, 0, &pa, chunk, false)?;
        let others: Vec<(Vec<u32>, Vec<u32>)> = (1..n).map(|j| (toks(9 + j % 5, 50 + j as u32), toks(steps, 100 + j as u32))).collect();
        for (j, (p, _)) in others.iter().enumerate() {
            fill(s, j as i32 + 1, p, chunk, false)?;
        }
        let slots: Vec<[i32; 1]> = (0..n as i32).map(|j| [j]).collect();
        let mut got = Vec::new();
        for t in 0..steps {
            let mut rows = vec![Row::new(fa[t], pa.len() + t, &slots[0])];
            room(s, 0, pa.len() + t + 2)?;
            for (j, (p, f)) in others.iter().enumerate() {
                room(s, j as i32 + 1, p.len() + t + 2)?;
                rows.push(Row::new(f[t], p.len() + t, &slots[j + 1]));
            }
            s.decode(&rows)?;
            got.push(s.logits(0)?);
        }
        s.release(&slots.concat())?;
        let pairs: Vec<(&[f32], &[f32])> = reference.iter().zip(&got).map(|(w, g)| (&w[..], &g[..])).collect();
        report(&format!("{n} rows a step, the row with {} other sequences' rows, against alone", n - 1), n <= 16 || strict, &pairs);
    }
    if strict {
        sizes_done(case, fails)?;
        return Ok(());
    }
    // with another sequence's prompt chunk in the step
    // (62 and 63: either side of the 64 rows at which a product's kernel changes)
    let mut sizes: Vec<usize> = [7, 15, 31, 47, 62, 63, 127, 255, 511, 1023].into_iter().filter(|&c| c < chunk && c < window).collect();
    if sizes.last().is_none_or(|&c| c != chunk - 1) && chunk - 1 < window {
        sizes.push(chunk - 1);
    }
    for c in sizes {
        // (as many steps as the other sequence's room holds chunks; a real model's other prompt is the text from its first token, as a prompt is, for as many whole chunks as the text has,
        // and tokens by a rule where it has not two)
        let n = steps.min((window - 1) / c);
        let (px, of_text) = match &text {
            Some(t) if t.len() >= 2 * c => (t[..n.min(t.len() / c) * c].to_vec(), true),
            _ => (toks(n * c, 5 + c as u32), false),
        };
        let n = px.len() / c;
        let rows = c + 1;
        // the row beside a prompt's chunks, and the prompt's logits after its last
        let beside = |s: &mut dyn ModelStep, prompt: &[u32]| -> Result<(Vec<Vec<f32>>, Vec<f32>)> {
            fill(s, 0, &pa, chunk, false)?;
            let (mut got, mut after) = (Vec::new(), Vec::new());
            for t in 0..n {
                room(s, 0, pa.len() + t + 2)?;
                room(s, 1, (t + 1) * c + (t + 1 == n) as usize)?;
                let last = t + 1 == n;
                s.round(&[Row::new(fa[t], pa.len() + t, &[0])], &[PrefillJob { pages: &[1], start: t * c, tokens: &prompt[t * c..(t + 1) * c], last }])?;
                got.push(s.logits(0)?);
                if last {
                    after = s.logits(1)?;
                }
            }
            s.release(&[0, 1])?;
            Ok((got, after))
        };
        let py = toks(n * c, 1005 + c as u32);
        let (gx, after) = beside(s, &px)?;
        let (gy, _) = beside(s, &py)?;
        let pairs: Vec<(&[f32], &[f32])> = reference.iter().zip(&gx).map(|(w, g)| (&w[..], &g[..])).collect();
        report(&format!("{rows} rows a step, the row with a prompt's chunk of {c}, against alone"), rows <= 16, &pairs);
        let kind = if of_text { "the text's" } else { "tokens by a rule" };
        let pairs: Vec<(&[f32], &[f32])> = gx.iter().zip(&gy).map(|(w, g)| (&w[..], &g[..])).collect();
        report(&format!("{rows} rows a step, the row with another prompt of {c} beside it, against the first"), exact, &pairs);
        // the prompt's side: the same chunks alone, and whole chunks alone
        let alone = fill(s, 1, &px, c, true)?.expect("the last chunk's logits");
        s.release(&[1])?;
        report(
            &format!("{rows} rows a step, a prompt of {} ({kind}) in chunks of {c} beside the row, against the chunks alone", px.len()),
            rows <= 16,
            &[(&alone[..], &after[..])],
        );
        let whole = fill(s, 1, &px, chunk, true)?.expect("the last chunk's logits");
        s.release(&[1])?;
        report(&format!("(yardstick) that prompt alone in chunks of {c}, against alone in chunks of {chunk}"), false, &[(&whole[..], &alone[..])]);
    }
    // prompts of different lengths in one pass (`prefill_many`: prompts that come in together), each prompt's logits after its last token and after the step that follows it
    // (its state and cache as the pass left them), against the prompt alone; the first prompt beside other prompts of the same lengths, against beside the first ones
    let alone_then_step = |s: &mut dyn ModelStep, p: &[u32]| -> Result<(Vec<f32>, Vec<f32>)> {
        let after = fill(s, 0, p, chunk, true)?.expect("the prompt's logits");
        room(s, 0, p.len() + 2)?;
        s.decode(&[Row { token: fa[0], pos: p.len(), pages: &[0], after: None }])?;
        let next = s.logits(0)?;
        s.release(&[0])?;
        Ok((after, next))
    };
    let together = |s: &mut dyn ModelStep, ps: &[Vec<u32>]| -> Result<Vec<(Vec<f32>, Vec<f32>)>> {
        let slots: Vec<[i32; 1]> = (0..ps.len() as i32).map(|j| [j]).collect();
        for (p, sl) in ps.iter().zip(&slots) {
            room(s, sl[0], p.len() + 2)?;
        }
        let jobs: Vec<PrefillJob> = ps.iter().zip(&slots).map(|(p, sl)| PrefillJob { pages: sl, start: 0, tokens: p, last: true }).collect();
        s.prefill_many(&jobs)?;
        let after: Vec<Vec<f32>> = (0..ps.len()).map(|j| s.logits(j)).collect::<Result<_>>()?;
        let mut out = Vec::new();
        for ((p, sl), a) in ps.iter().zip(&slots).zip(after) {
            s.decode(&[Row { token: fa[0], pos: p.len(), pages: sl, after: None }])?;
            out.push((a, s.logits(0)?));
        }
        s.release(&slots.concat())?;
        Ok(out)
    };
    // (15 rows in all: the products' kernels of a pass alone, so every bit is required; then 74 rows, another kernel's sums, counted)
    for lens in [&[7usize, 5, 3][..], &[37, 9, 23, 5]] {
        let lens: Vec<usize> = lens.iter().copied().take(k.min(s.prefill_rows())).filter(|&l| l + 2 < window).collect();
        if lens.len() < 2 || lens.iter().sum::<usize>() > chunk {
            continue;
        }
        let rows: usize = lens.iter().sum();
        let ps: Vec<Vec<u32>> = lens.iter().enumerate().map(|(j, &l)| toks(l, 3000 + j as u32)).collect();
        let alone: Vec<(Vec<f32>, Vec<f32>)> = ps.iter().map(|p| alone_then_step(s, p)).collect::<Result<_>>()?;
        let got = together(s, &ps)?;
        let pairs: Vec<(&[f32], &[f32])> = alone.iter().zip(&got).flat_map(|(w, g)| [(&w.0[..], &g.0[..]), (&w.1[..], &g.1[..])]).collect();
        report(
            &format!("prompts of {lens:?} in one pass ({rows} rows), each one's logits after it and after its next step, against alone"),
            rows <= crate::dynamic::EXACT_ROWS,
            &pairs,
        );
        let others: Vec<Vec<u32>> = std::iter::once(ps[0].clone()).chain(lens[1..].iter().enumerate().map(|(j, &l)| toks(l, 4000 + j as u32))).collect();
        let beside = together(s, &others)?;
        report(
            &format!("prompts of {lens:?} in one pass, the first beside other prompts of those lengths, against beside the first ones"),
            exact,
            &[(&got[0].0[..], &beside[0].0[..]), (&got[0].1[..], &beside[0].1[..])],
        );
    }
    sizes_done(case, fails)
}

/// roundcheck's last line: its failures, or that it passed.
fn sizes_done(case: &str, fails: usize) -> Result<()> {
    if fails > 0 {
        return err(format!("roundcheck {case}: {fails} failed"));
    }
    println!("roundcheck {case}: PASS");
    Ok(())
}

/// `keepcheck CASE [--level L] [--dir MODEL --ctx N --chunk N]` (gemma4, qwen35, muse, kimi, glm; R4c's first step): a finished sequence kept in its slot and gone on with, as a
/// conversation's next turn does (`ModelStep::resume` and `mark`, src/dynamic/rows.rs), on the case's random model, or the model in MODEL (a real one). Every comparison is of logits, bit for bit, against a second slot that ran the same tokens
/// in the same passes and was never resumed:
///   - the whole of a sequence (a prompt, then tokens a step at a time) gone on with a further chunk: every family;
///   - the sequence asked to go on from inside the tokens it wrote a step at a time: from there where the model has no recurrent state; from its prompt's end, which was
///     marked, where it has recurrent states (their copy put back; a latent cache is paged as any keys are); refused, the slot left empty, where its states are not kept at a
///     mark (DeepSeek-V4's compressed entries);
///   - a long sequence (9 chunks and 24 steps) cut back by 10 tokens and to its third chunk's end, or gone on from a mark at its third chunk's end: kept, or refused where the
///     sliding layers' blocks behind the place were given back, and where kept the bits too;
///   - a start shared with an empty slot (`share`): the full blocks of what the two have in common, or a recurrent model's mark with its states copied, or nothing where
///     the states are not kept at a mark; the sharer and the sequence it shared from both go on as sequences that shared nothing do, and the sharer too after the other has ended;
///   - through the scheduler: a conversation's second turn (the first's prompt and reply and more) goes on from all the first left, another's second turn that leaves the reply
///     out goes on from the prompt's end where the model can, and a conversation that begins as another does shares its start; the prompt tokens found are counted, and each
///     reply is, token for token, what the same passes give by hand.
pub fn keep_check(a: &crate::args::Args) -> Result<()> {
    use crate::conv::Gen;
    use crate::sched::{Ev, Finish, ModelStep, Reply, Row, Sched, Submit};
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    let case = a.need(1, "CASE")?;
    let level = a.flag("level").unwrap_or("penguin");
    let mut e: Dyn = match a.flag("dir") {
        Some(dir) => {
            let o = DynOpts { ctx: a.num("ctx", 4096usize)?, chunk: a.num("chunk", 256usize)?, graph: true, device: 0, layout: level_layout(level)? };
            load_dyn(case, Path::new(dir), &o, level)?
        }
        None => random_dyn(case, level, "keepcheck")?,
    };
    let (vocab, cuts, marks, window) = (e.desc().vocab as u32, e.cuts(), e.can_mark(), e.least_window());
    // (a model with recurrent states keeps them at boundaries (src/dynamic/snaps.rs); this check's prompts are short, so at boundaries 32 tokens apart, or as far apart as the
    // largest blocks: DeepSeek-V4's compressed entries' are found by name whole)
    if !cuts && marks {
        let unit = e.whole_start();
        e.ask_state_every(if unit % 32 == 0 { unit } else { 2 * unit });
    }
    let Some((k, s)) = e.as_step(3)? else { return err(format!("{case}: no scheduler's step")) };
    if k < 3 || !s.keeps() {
        return err(format!("{case}: {k} sequences, a step that keeps: {}", s.keeps()));
    }
    let chunk = s.chunk();
    // (the boundaries the step keeps recurrent states at, where the scheduler ends a prompt's pieces: 0 none)
    let every = s.snap_every();
    // a prompt's pieces from `start` as the scheduler runs them: a chunk's tokens at most, none across a boundary
    fn pieces(t: &[u32], start: usize, chunk: usize, every: usize) -> Vec<&[u32]> {
        let (mut out, mut at) = (Vec::new(), start);
        while at < t.len() {
            let mut n = (t.len() - at).min(chunk);
            if every > 0 {
                n = n.min((at / every + 1) * every - at);
            }
            out.push(&t[at..at + n]);
            at += n;
        }
        out
    }
    let toks = |n: usize, seed: u32| -> Vec<u32> { (0..n as u32).map(|i| 3 + (i * 7919 + seed * 104_729) % (vocab - 3)).collect() };
    // tokens a step at a time, each a row alone
    fn steps(s: &mut dyn ModelStep, slot: i32, start: usize, t: &[u32]) -> Result<()> {
        for (i, &tok) in t.iter().enumerate() {
            room(s, slot, start + i + 2)?;
            s.decode(&[Row::new(tok, start + i, &[slot])])?;
        }
        Ok(())
    }
    // (0 or 1: whether the logits differ from the other slot's)
    let same = |what: &str, want: &[f32], got: &[f32]| -> usize {
        let (differ, max, _, _) = row_diff(want, got);
        println!(
            "{} {what}: {differ} of {} logits differ from a sequence that ran the same tokens on its own, the largest by {max:e}",
            if differ == 0 { "PASS" } else { "FAIL" },
            want.len()
        );
        (differ > 0) as usize
    };
    // a slot that should hold nothing: a prompt begins in it as in another empty one
    let empty = |s: &mut dyn ModelStep, slot: i32, other: i32, what: &str, t: &[u32]| -> Result<usize> {
        let got = fill(s, slot, t, chunk, true)?.expect("the logits");
        let want = fill(s, other, t, chunk, true)?.expect("the logits");
        Ok(same(what, &want, &got))
    };
    let mut fails = 0;
    println!("{case} at {level}: a chunk {chunk}; a kept sequence can be cut back anywhere: {cuts}; its states are kept at a mark: {marks}");
    let (p, g, tail, alt) = (toks(40, 1), toks(24, 2), toks(20, 3), toks(30, 4));
    let n = p.len() + g.len();
    // the whole of it
    fill(s, 0, &p, chunk, false)?;
    steps(s, 0, p.len(), &g)?;
    let kept = s.resume(0, n)?;
    if kept != n {
        println!("FAIL the whole of a sequence of {n} tokens: {kept} kept");
        fails += 1;
    } else {
        let got = fill_from(s, 0, n, &tail, chunk, true)?.expect("the logits");
        fill(s, 1, &p, chunk, false)?;
        steps(s, 1, p.len(), &g)?;
        let want = fill_from(s, 1, n, &tail, chunk, true)?.expect("the logits");
        fails += same(&format!("the whole of a sequence of {n} tokens, then {} more", tail.len()), &want, &got);
    }
    s.release(&[0, 1])?;
    // inside the steps: from there, from the prompt's end (marked), or not at all
    let cut = p.len() + 10;
    fill(s, 0, &p, chunk, false)?;
    s.mark(0, p.len())?;
    steps(s, 0, p.len(), &g)?;
    let kept = s.resume(0, cut)?;
    let should = if cuts {
        cut
    } else if marks {
        p.len()
    } else {
        0
    };
    if kept != should {
        println!("FAIL a sequence of {n} tokens asked to go on within {cut}: {kept} kept, where {should} should be");
        fails += 1;
    } else if kept > 0 {
        let got = fill_from(s, 0, kept, &alt, chunk, true)?.expect("the logits");
        fill(s, 1, &p, chunk, false)?;
        steps(s, 1, p.len(), &g[..kept - p.len()])?;
        let want = fill_from(s, 1, kept, &alt, chunk, true)?.expect("the logits");
        let place = if kept == cut { "there".to_string() } else { format!("its prompt's end, {kept}, where its states were kept") };
        fails += same(&format!("a sequence of {n} tokens asked to go on within {cut}: from {place}, then {} others", alt.len()), &want, &got);
    } else {
        // refused, and nothing of it changed: it goes on from its end as one that was never asked
        let whole = s.resume(0, n)?;
        let got = fill_from(s, 0, n, &tail, chunk, true)?.expect("the logits");
        fill(s, 1, &p, chunk, false)?;
        steps(s, 1, p.len(), &g)?;
        let want = fill_from(s, 1, n, &tail, chunk, true)?.expect("the logits");
        let what = format!("a sequence of {n} tokens asked to go on within {cut}: refused (its states are what they are after the last token, and not kept at a mark) and as it was ({whole} held); from its end, {} more", tail.len());
        fails += same(&what, &want, &got) | (whole != n) as usize;
    }
    s.release(&[0, 1])?;
    // a long sequence: 9 chunks, then 24 steps
    let long = toks(9 * chunk, 7);
    let all = long.len() + g.len();
    if (cuts || marks) && all + alt.len() < s.window() {
        // (the place asked for, where it should go on from, the steps a sequence that stops there took, and whether it must be kept: a cut by fewer tokens than a chunk always is)
        let cases: Vec<(usize, usize, usize, bool)> = if cuts {
            vec![(long.len() + 14, long.len() + 14, 14, true), (3 * chunk, 3 * chunk, 0, false)]
        } else {
            vec![(3 * chunk + 5, 3 * chunk, 0, false)]
        };
        for (ask, to, by_hand, must) in cases {
            fill(s, 0, &long[..3 * chunk], chunk, false)?;
            s.mark(0, 3 * chunk)?;
            fill_from(s, 0, 3 * chunk, &long[3 * chunk..], chunk, false)?;
            steps(s, 0, long.len(), &g)?;
            let kept = s.resume(0, ask)?;
            if kept == to {
                let got = fill_from(s, 0, to, &alt, chunk, true)?.expect("the logits");
                fill(s, 1, &long[..to.min(long.len())], chunk, false)?;
                steps(s, 1, long.len(), &g[..by_hand])?;
                let want = fill_from(s, 1, to, &alt, chunk, true)?.expect("the logits");
                fails += same(&format!("a sequence of {all} tokens asked to go on within {ask}: from {to}, then {} others", alt.len()), &want, &got);
            } else if kept == 0 && !must {
                // refused, and nothing of it changed: it goes on from its end
                let whole = s.resume(0, all)?;
                let got = fill_from(s, 0, all, &alt, chunk, true)?.expect("the logits");
                fill(s, 1, &long, chunk, false)?;
                steps(s, 1, long.len(), &g)?;
                let want = fill_from(s, 1, all, &alt, chunk, true)?.expect("the logits");
                let what = format!(
                    "a sequence of {all} tokens asked to go on within {ask}: refused (the sliding layers' blocks behind it were given back) and as it was ({whole} held); from its end, {} more",
                    alt.len()
                );
                fails += same(&what, &want, &got) | (whole != all) as usize;
            } else {
                println!("FAIL a sequence of {all} tokens asked to go on within {ask}: {kept} kept");
                fails += 1;
            }
            s.release(&[0, 1])?;
        }
    }
    // a start shared with an empty slot: of a sequence of a prompt (marked at its end) and 24 steps, what a request that has its first 50 tokens can have
    {
        let should = if cuts {
            cut / 16 * 16
        } else if marks {
            p.len()
        } else {
            0
        };
        let begin = |s: &mut dyn ModelStep| -> Result<usize> {
            fill(s, 0, &p, chunk, false)?;
            s.mark(0, p.len())?;
            steps(s, 0, p.len(), &g)?;
            s.share(0, 1, cut)
        };
        let got = begin(s)?;
        if got != should {
            println!("FAIL a start of {cut} tokens shared from a sequence of {n}: {got} shared, where {should} should be");
            fails += 1;
        } else if got == 0 {
            // nothing shared: the slot it was asked for holds nothing still, and the sequence asked is as it was
            fails +=
                empty(s, 1, 2, "a start is not shared (its states are not kept at a mark); the slot it was asked for is empty still: a prompt in it", &alt)?;
            s.release(&[1, 2])?;
            let source = fill_from(s, 0, n, &tail, chunk, true)?.expect("the logits");
            fill(s, 1, &p, chunk, false)?;
            steps(s, 1, p.len(), &g)?;
            let alone = fill_from(s, 1, n, &tail, chunk, true)?.expect("the logits");
            fails += same(&format!("the sequence the start was asked of, gone on with {} more", tail.len()), &alone, &source);
        } else {
            // the sharer goes on, then the sequence it shared from; each against a slot that ran the same tokens and shared nothing
            let sharer = fill_from(s, 1, got, &alt, chunk, true)?.expect("the logits");
            let source = fill_from(s, 0, n, &tail, chunk, true)?.expect("the logits");
            fill(s, 2, &p, chunk, false)?;
            steps(s, 2, p.len(), &g[..got - p.len()])?;
            let want = fill_from(s, 2, got, &alt, chunk, true)?.expect("the logits");
            fails += same(&format!("a start of {cut} tokens shared from a sequence of {n}: {got} of them, then {} others", alt.len()), &want, &sharer);
            s.release(&[2])?;
            fill(s, 2, &p, chunk, false)?;
            steps(s, 2, p.len(), &g)?;
            let alone = fill_from(s, 2, n, &tail, chunk, true)?.expect("the logits");
            fails += same(&format!("the sequence the start was shared from, gone on with {} more", tail.len()), &alone, &source);
            s.release(&[0, 1, 2])?;
            // the sequence it shared from ends first: what the two held together stays the sharer's
            let again = begin(s)?;
            s.release(&[0])?;
            let after = fill_from(s, 1, again, &alt, chunk, true)?.expect("the logits");
            fails += same("the same start, after the sequence it was shared from has ended", &want, &after);
        }
        s.release(&[0, 1, 2])?;
        // of a long sequence: layers whose attention slides keep a window and a chunk behind the sequence's end and give back the blocks before, so a step at the start
        // reads blocks it no longer holds, and nothing is shared; a model whose layers all read the whole sequence shares it as before
        if cuts && all + alt.len() < s.window() {
            fill(s, 0, &long, chunk, false)?;
            let got = s.share(0, 1, cut)?;
            let m = cut / 16 * 16;
            let held = window.is_none_or(|w| long.len().saturating_sub(w + chunk) / 16 <= m.saturating_sub(w) / 16);
            let should = if held { m } else { 0 };
            if got != should {
                println!("FAIL a start of {cut} tokens shared from a sequence of {}: {got} shared, where {should} should be", long.len());
                fails += 1;
            } else if got == 0 {
                let what = format!("a start of {cut} tokens is not shared from a sequence of {} (its sliding layers gave back the blocks a step there reads); a prompt in the slot asked for", long.len());
                fails += empty(s, 1, 2, &what, &alt)?;
            } else {
                let sharer = fill_from(s, 1, got, &alt, chunk, true)?.expect("the logits");
                fill(s, 2, &long[..got], chunk, false)?;
                let want = fill_from(s, 2, got, &alt, chunk, true)?.expect("the logits");
                fails += same(
                    &format!("a start of {cut} tokens shared from a sequence of {}: {got} of them, then {} others", long.len(), alt.len()),
                    &want,
                    &sharer,
                );
            }
            s.release(&[0, 1, 2])?;
            // a sequence that shared a start holds its sliding layers' blocks from the window before that start on, and none earlier: asked to go on from before the
            // start's end (the place is the start of a block both hold) it is refused where a step there reads earlier blocks, and is as it was; else it goes on from there
            let at = 3 * chunk;
            fill(s, 0, &long[..at + 16], chunk, false)?;
            let got = s.share(0, 1, at + 8)?;
            fill_from(s, 1, got, &alt, chunk, false)?;
            let (ask, to, whole) = (at - 4, at - 16, got + alt.len());
            let reads = |u: usize| window.map_or(0, |w| u.saturating_sub(w) / 16);
            let should = if reads(to) < reads(at) { 0 } else { to };
            let kept = s.resume(1, ask)?;
            if got != at || kept != should {
                println!(
                    "FAIL a sequence that shared a start of {at} tokens ({got} shared), asked to go on within {ask}: {kept} kept, where {should} should be"
                );
                fails += 1;
            } else if kept == 0 {
                let held = s.resume(1, whole)?;
                let sharer = fill_from(s, 1, whole, &tail, chunk, true)?.expect("the logits");
                fill(s, 2, &long[..at], chunk, false)?;
                fill_from(s, 2, at, &alt, chunk, false)?;
                let want = fill_from(s, 2, whole, &tail, chunk, true)?.expect("the logits");
                let what = format!(
                    "a sequence that shared a start of {at} tokens, asked to go on within {ask}: refused (it holds its sliding layers' blocks from the window before that start, none earlier) and as it was ({held} held); from its end, {} more",
                    tail.len()
                );
                fails += same(&what, &want, &sharer) | (held != whole) as usize;
            } else {
                let sharer = fill_from(s, 1, to, &tail, chunk, true)?.expect("the logits");
                fill(s, 2, &long[..at], chunk, false)?;
                let cut = s.resume(2, to)?;
                let want = fill_from(s, 2, to, &tail, chunk, true)?.expect("the logits");
                let what = format!(
                    "a sequence that shared a start of {at} tokens, asked to go on within {ask}: from {to}, the start of the block both hold, then {} others",
                    tail.len()
                );
                fails += same(&what, &want, &sharer) | (cut != to) as usize;
            }
            s.release(&[0, 1, 2])?;
        }
    }
    // blocks found by name (a step that finds them so): tokens a sequence ran are had again whoever holds their blocks, or no one
    if s.shares_starts() {
        // (a scope of this check's own: what is named here stays in the pools, and the conversations below, in the scope of requests with no salt, are not to find it)
        let root = crate::kv::block_root(0x6b65_6570_6368_6563);
        let both: Vec<u32> = [&p[..], &g[..]].concat();
        let keys = crate::kv::block_names(root, &both);
        // a prompt and steps, its blocks named, and the slot emptied: three blocks of it found
        fill(s, 0, &p, chunk, false)?;
        steps(s, 0, p.len(), &g)?;
        s.name(0, 0, &keys)?;
        s.release(&[0])?;
        let got = s.adopt(1, &keys[..3])?;
        if s.snap_every() > 0 {
            // (a recurrent model's start is found at a boundary whose states were kept, and this one ended at none: the sequence named its blocks at its end alone)
            let ok = got == 0;
            fails += !ok as usize;
            println!(
                "{} a start of 48 tokens whose states were kept at no boundary, looked for by name after the sequence that ran it has ended: {got} found (it should 0)",
                if ok { "PASS" } else { "FAIL" }
            );
        } else if got != 48 {
            println!("FAIL a start of 48 tokens looked for by name after the sequence that ran it has ended: {got} found");
            fails += 1;
        } else {
            let finder = fill_from(s, 1, got, &alt, chunk, true)?.expect("the logits");
            fill(s, 2, &p, chunk, false)?;
            steps(s, 2, p.len(), &g[..got - p.len()])?;
            let want = fill_from(s, 2, got, &alt, chunk, true)?.expect("the logits");
            fails += same(&format!("a start of 48 tokens found by name after the sequence that ran it has ended, then {} others", alt.len()), &want, &finder);
        }
        s.release(&[0, 1, 2])?;
        // a long sequence, named as it grows: a start of six chunks is found though the sequence no longer holds all a step there reads (its sliding layers gave blocks
        // back as it grew: they wait by their names); and the sequence is unharmed
        if all + alt.len() < s.window() {
            let keys = crate::kv::block_names(root, &long);
            // (a recurrent model's start is found at a boundary)
            let at = (6 * chunk).next_multiple_of(every.max(1));
            fill_named(s, 0, &long, chunk, &keys)?;
            let got = s.adopt(1, &keys[..at / 16])?;
            if got != at {
                println!("FAIL a start of {at} tokens looked for by name while the sequence that ran it holds {}: {got} found", long.len());
                fails += 1;
            } else {
                let finder = fill_from(s, 1, at, &alt, chunk, true)?.expect("the logits");
                fill(s, 2, &long[..at], chunk, false)?;
                let want = fill_from(s, 2, at, &alt, chunk, true)?.expect("the logits");
                let what = format!(
                    "a start of {at} tokens found by name while the sequence that ran it holds {} (it gave back blocks a step there reads; they wait by their names), then {} others",
                    long.len(),
                    alt.len()
                );
                fails += same(&what, &want, &finder);
                s.release(&[1, 2])?;
                let source = fill_from(s, 0, long.len(), &tail, chunk, true)?.expect("the logits");
                fill(s, 2, &long, chunk, false)?;
                let alone = fill_from(s, 2, long.len(), &tail, chunk, true)?.expect("the logits");
                fails += same(&format!("the sequence that ran it, gone on with {} more", tail.len()), &alone, &source);
            }
            s.release(&[0, 1, 2])?;
        }
    }
    // through the scheduler: a conversation's second turn (the prompt, the reply and more), and another's that leaves the reply out
    let reply = Reply {
        params: crate::sampler::Params { temperature: 0.0, ..Default::default() },
        max_new: 16,
        ignore_eos: true,
        logprobs: None,
        stop_ids: Vec::new(),
    };
    let q = toks(40, 11);
    // (the other conversation is kept no longer than a chunk past the start the two share, so that sliding layers hold it still, and longer than twice the start, so that
    // it is not gone on in)
    let (xs, xa_more, yb_more) = (toks(50, 21), toks(40, 22), toks(40, 23));
    let (xa, yb): (Vec<u32>, Vec<u32>) = ([&xs[..], &xa_more[..]].concat(), [&xs[..], &yb_more[..]].concat());
    let (r1, p2, r2, hit, q2, rq, hit_q, rx, ry, hit_y) = {
        // (the step lent for the scheduler's time alone)
        let lent: &mut dyn ModelStep = &mut *s;
        let mut sched = Sched::new(lent, 2);
        let ask = |sched: &mut Sched<&mut dyn ModelStep>, prompt: &[u32]| -> Result<Vec<u32>> {
            let (tx, rx) = std::sync::mpsc::channel();
            sched.submit(Submit {
                prompt: prompt.to_vec(),
                reply: reply.clone(),
                out: tx,
                abort: Arc::new(AtomicBool::new(false)),
                resumed: Default::default(),
                cached: Default::default(),
                created: Default::default(),
                salt: 0,
                pin: 0,
                images: Vec::new(),
            });
            while !sched.idle() {
                sched.tick()?;
            }
            let mut out = Vec::new();
            loop {
                match rx.recv().map_err(|_| Error("the scheduler dropped a sequence".into()))? {
                    Ev::Token(t) => out.push(t),
                    Ev::Lp(_) => {}
                    Ev::End(Ok(Finish::Length)) => return Ok(out),
                    Ev::End(f) => return err(format!("a sequence ended {f:?}")),
                }
            }
        };
        let r1 = ask(&mut sched, &p)?;
        let p2: Vec<u32> = [&p[..], &r1[..], &toks(12, 9)[..]].concat();
        let r2 = ask(&mut sched, &p2)?;
        let hit = sched.stats.hit as usize;
        ask(&mut sched, &q)?;
        let q2: Vec<u32> = [&q[..], &toks(20, 12)[..]].concat();
        let rq = ask(&mut sched, &q2)?;
        let hit_q = sched.stats.hit as usize - hit;
        // a third conversation, then one that begins as it does (50 tokens of the 105 it left): a slot of its own, the start shared
        let rx = ask(&mut sched, &xa)?;
        let before = sched.stats.hit as usize;
        let ry = ask(&mut sched, &yb)?;
        (r1, p2, r2, hit, q2, rq, hit_q, rx, ry, sched.stats.hit as usize - before)
    };
    s.release(&[0, 1, 2])?;
    // the same passes by hand, each token the step's own choice (the tokens given are the scheduler's, so one that differs does not move the ones after it): the chunks one after
    // another from `start`, the last one's sample, then the reply's tokens a step at a time
    let replay_in = |s: &mut dyn ModelStep, slot: i32, start: usize, chunks: &[&[u32]], reply: &[u32]| -> Result<Vec<u32>> {
        let (mut at, mut first) = (start, None);
        for (i, c) in chunks.iter().enumerate() {
            room(s, slot, at + c.len() + 1)?;
            first = s.prefill(&[slot], at, c, i + 1 == chunks.len())?;
            at += c.len();
        }
        let mut out = vec![first.ok_or_else(|| Error("no sample after a chunk".into()))?.token];
        for (i, &t) in reply[..reply.len() - 1].iter().enumerate() {
            room(s, slot, at + i + 2)?;
            out.push(s.decode(&[Row::new(t, at + i, &[slot])])?[0].token);
        }
        Ok(out)
    };
    let replay = |s: &mut dyn ModelStep, start: usize, chunks: &[&[u32]], reply: &[u32]| replay_in(s, 0, start, chunks, reply);
    // (the first turn left the prompt and its reply but the last token in the slot)
    let held = p.len() + r1.len() - 1;
    let first = replay(s, 0, &pieces(&p, 0, chunk, every), &r1)? == r1;
    let second = replay(s, held, &pieces(&p2, held, chunk, every), &r2)?;
    s.release(&[0])?;
    let ok = hit == held && first && second == r2;
    fails += !ok as usize;
    println!(
        "{} a conversation's second turn through the scheduler: it went on from {hit} of its {} prompt tokens (the slot held {held}); its reply {}",
        if ok { "PASS" } else { "FAIL" },
        p2.len(),
        if second == r2 {
            "is the same passes' by hand, token for token".to_string()
        } else {
            format!("{r2:?} against {second:?} by hand (the first turn's the same: {first})")
        }
    );
    // the turn that leaves the reply out: from the prompt's end where the model can (two passes by hand: the prompt, the rest), else from its first token (one)
    let should = if cuts || marks { q.len() } else { 0 };
    let by_hand = if hit_q > 0 {
        replay(s, 0, &[pieces(&q, 0, chunk, every), pieces(&q2, q.len(), chunk, every)].concat(), &rq)?
    } else {
        replay(s, 0, &pieces(&q2, 0, chunk, every), &rq)?
    };
    s.release(&[0])?;
    let ok = hit_q == should && by_hand == rq;
    fails += !ok as usize;
    println!(
        "{} a second turn that leaves the reply out: it went on from {hit_q} of its {} prompt tokens (it should from {should}); its reply {}",
        if ok { "PASS" } else { "FAIL" },
        q2.len(),
        if by_hand == rq { "is the same passes' by hand, token for token".to_string() } else { format!("{rq:?} against {by_hand:?} by hand") }
    );
    // the conversation that begins as another does: the start's full blocks shared where the model has no state; a recurrent model's states kept at the newest boundary
    // within it (the other's mark is past what the two share); with no states kept (DeepSeek-V4): from its first token there. By hand: the other conversation as the
    // scheduler ran it, the start shared, the rest (a recurrent model: its own pieces from its first token, which give the states the other's kept); or the prompt whole.
    let should = if cuts {
        xs.len() / 16 * 16
    } else if every > 0 {
        xs.len() / every * every
    } else {
        0
    };
    let by_hand = if hit_y > 0 && !cuts {
        let first = replay(s, 0, &pieces(&xa, 0, chunk, every), &rx)? == rx;
        let y = replay_in(s, 1, 0, &pieces(&yb, 0, chunk, every), &ry)?;
        if first {
            y
        } else {
            Vec::new()
        }
    } else if hit_y > 0 {
        let chunks = pieces(&xa, 0, chunk, every);
        let first = replay(s, 0, &chunks, &rx)? == rx;
        let got = s.share(0, 1, xs.len())?;
        let y = replay_in(s, 1, got, &[&yb[got..]], &ry)?;
        if first && got == hit_y {
            y
        } else {
            Vec::new()
        }
    } else {
        replay(s, 0, &pieces(&yb, 0, chunk, every), &ry)?
    };
    s.release(&[0, 1])?;
    let ok = hit_y == should && by_hand == ry;
    fails += !ok as usize;
    println!(
        "{} a conversation that begins as another does, through the scheduler: {hit_y} of its {} prompt tokens shared (it should {should}); its reply {}",
        if ok { "PASS" } else { "FAIL" },
        yb.len(),
        if by_hand == ry { "is the same passes' by hand, token for token".to_string() } else { format!("{ry:?} against {by_hand:?} by hand") }
    );
    if fails > 0 {
        return err(format!("keepcheck {case}: {fails} failed"));
    }
    println!("keepcheck {case}: PASS");
    Ok(())
}

/// `usercheck CASE [--level L] [--users N] [--tokens T] [--dir DIR [--places N [--mode M] [--device-plan]] | --ctx C] [--prompt-tokens P] [--stagger D]` (gemma4, qwen35, muse, kimi, glm, nemotron): the case's random model on the
/// description executor (or the model in DIR, with `--places` its experts in the expert cache; with `--ctx` a context of C tokens; with `--prompt-tokens` prompts of P (k + 1) / N tokens), each of N prompts replied to alone as `run` replies (the reference), then by the scheduler (src/dynamic/rows.rs) alone, then all at once. Alone through
/// the scheduler every reply is the reference, token for token; at once too where GLYD_ENGINE_BATCH_PREFILL=0 (prompts one a pass, decode steps of the rows that decode: 16 or fewer here, whose bits
/// are each row's own pass's). With prompts run together the replies that differ are counted, not refused (a step of more than 16 rows is another kernel's; a random model's near ties turn on its last
/// bits: `roundcheck` measures the logits). The round is one step of the rows that decode and the prompts' chunks unless GLYD_ENGINE_ONE_STEP=0, and some step must then have held both.
/// Then the prompts at once twice more, with no graph at all (every pass eager) and with every graph on and every pass that is no whole graph's in pieces
/// (src/dynamic/pieces.rs): the replies the same token for token and each sample's logit to the bit, and where pieces can run, some pass replayed them. With
/// GLYD_ENGINE_OVERLAP=1, where steps are launched ahead, the prompts at once launched ahead and not: the same tokens and logits. With `--places` every run (each reference, alone,
/// at once, each side of each comparison) begins from the expert cache's cold state with its planner's times held (`ExpertStore::cold`, `hold_costs`), so two runs of the
/// same calls find the same experts in places and plan alike, in every mode.
/// With --stagger D the prompts then come D scheduler rounds apart, as a server's requests do: with prompts one a pass every reply is the reply alone, and with no mixture of
/// experts each reply is the same when every other prompt's tokens are changed (a row's sums may turn on its passes' row counts, never on the other rows' content).
///
/// With a page budget too small for all of them at once (GLYD_ENGINE_PAGES, R4d) the prompts are run at once three more times: sequences sent back for room swapped to the host wherever they can be,
/// swapped where that costs less than their recompute (the default), and run again from their tokens only; each run's replies against the references, its sequences sent back and swapped, and each
/// request's first token and the mean time between its tokens. Swapped sequences go on with the cache they had, bit for bit: with prompts one a pass their replies must be the references.
pub fn users_check(a: &crate::args::Args) -> Result<()> {
    use crate::conv::Gen;
    use crate::sched::{Ev, Finish, Reply, Sched, Submit, SwapMode};
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    let case = a.need(1, "CASE")?;
    let level = a.flag("level").unwrap_or("penguin");
    let users: usize = a.num("users", 4usize)?;
    let tokens: usize = a.num("tokens", 24usize)?;
    // (--ctx plan: the random model at the context the planner gives it on this GPU, each prompt filling it to within half a chunk with its reply: the long-context check)
    let planned = if a.flag("ctx") == Some("plan") { Some(planned_context(case)?) } else { None };
    if let Some(c) = planned {
        println!(
            "the random {case} model at the planner's context for this GPU: {c} tokens, prompts of up to {} and {tokens} made",
            c.saturating_sub(tokens + 32)
        );
    }
    let long: Option<usize> = match (a.flag("prompt-tokens"), planned) {
        (Some(v), _) => Some(v.parse().map_err(|_| Error(format!("--prompt-tokens {v}: not a number")))?),
        (None, Some(c)) => Some(c.saturating_sub(tokens + 32).max(1)),
        (None, None) => None,
    };
    let len = |k: usize| long.map_or(5 + 29 * k, |p| (p * (k + 1) / users).max(1));
    // (--layout mma|mma12: penguin's packs in that layout, where the GPU's choice is another)
    let layout = a.layout()?.or(level_layout(level)?);
    let places = a.num("places", 0usize)?;
    let mode = match a.flag("mode") {
        Some(m) => crate::offload::Mode::parse(m).ok_or_else(|| Error(format!("--mode {m}: copy, cpu, split or auto")))?,
        None => crate::offload::Mode::Copy,
    };
    let mut e: Dyn = match (a.flag("dir"), places) {
        // (`--places N`: the model in DIR with its experts in the expert cache, N places a layer and every expert in the computer's memory (src/offload.rs; `--mode`, copy by default, `--device-plan`: the
        // decode calls planned on the GPU, src/devplan.rs): a model larger than the GPU, its replies alone against its replies at once)
        (Some(d), places) if places > 0 => {
            let o = DynOpts { ctx: (len(users - 1) + tokens + 64).next_multiple_of(256), chunk: 256, graph: true, device: 0, layout };
            let offload = crate::offload::Offload { mode, device: a.has("device-plan"), ..Default::default() };
            let caching = crate::hbg_store::HbCaching { places, ram: None, offload, dense: 0 };
            match crate::dynamic::split::open_cached(Path::new(d), &o, &[], &[0], vec![caching])? {
                crate::dynamic::split::Run::One(e) => {
                    // (the planner's times held where they start, so that two runs from one state plan alike: each run below starts from the cold cache, `Sched::cold`)
                    e.stores().iter().for_each(|st| st.hold_costs());
                    let planned: usize = e.stores().iter().map(|st| st.device_layers()).sum();
                    println!(
                        "{d}: its experts in the expert cache, {places} places a layer, mode {}{}{}",
                        mode.name(),
                        if a.has("device-plan") { format!(", the decode calls of {planned} layers planned on the GPU") } else { String::new() },
                        if e.has_graph() { ", the decode step one graph" } else { "" }
                    );
                    e
                }
                _ => return err("the model on one GPU was asked for"),
            }
        }
        (Some(d), _) => {
            let o = DynOpts { ctx: (len(users - 1) + tokens + 64).next_multiple_of(256), chunk: 256, graph: true, device: 0, layout };
            load_dyn(case, Path::new(d), &o, level)?
        }
        // (--ctx N: the random model with a context of N tokens, for a long prompt's check: --prompt-tokens)
        (None, _) => random_dyn_at(
            case,
            level,
            layout,
            "usercheck",
            match planned {
                Some(c) => Some(c),
                None => a.flag("ctx").map(|v| v.parse().map_err(|_| Error(format!("--ctx {v}: not a number")))).transpose()?,
            },
        )?,
    };
    println!("{case} at {level}: layout {:?}", e.layout);
    let vocab = e.desc().vocab as u32;
    let prompts: Vec<Vec<u32>> = (0..users).map(|k| (0..len(k) as u32).map(|i| 3 + (i * 7919 + k as u32 * 104_729) % (vocab - 3)).collect()).collect();
    let reply = Reply {
        params: crate::sampler::Params { temperature: 0.0, ..Default::default() },
        max_new: tokens,
        ignore_eos: true,
        logprobs: None,
        stop_ids: Vec::new(),
    };
    // (each reply from the expert cache's cold state, as every run below: the same calls then find the same experts in places)
    let reference: Vec<Vec<u32>> =
        prompts.iter().map(|p| crate::sched::ModelStep::cold(&mut e).and_then(|()| e.reply(p, &reply, &mut |_| true).map(|r| r.0))).collect::<Result<_>>()?;
    // (--kv12: the scheduler's cache in 12-bit lossless pages, of tables made on these prompts; with GLYD_KV12_POOL's small pools the newest sequence is paused where the
    // raw slabs run short, and no page is made void)
    let kv12 = a.has("kv12");
    if kv12 {
        let t = crate::dynamic::kv12::calibrate(&mut e, &prompts, "usercheck")?;
        e.set_kv12(Some(t));
    }
    let (batchable, exact) = (e.batchable(), e.rows_exact());
    let Some((k, step)) = e.as_step(users)? else { return err(format!("{case}: no scheduler's step")) };
    let chunk = crate::sched::ModelStep::chunk(step);
    // (no finished sequence kept: the same prompts come again below, and each is to run from its first token, in passes with the others)
    let mut s = Sched::new(step, k).keeping(false).recording(true);
    // (the prompts `gap` scheduler rounds apart: 0, all of them before the first round; from the expert cache's cold state, its planner's times held, so that every run
    // begins from one state and the same calls find the same experts in places)
    let ask = |s: &mut Sched<&mut dyn crate::sched::ModelStep>, ps: &[Vec<u32>], gap: usize| -> Result<Vec<Vec<u32>>> {
        s.cold()?;
        if let Some(c) = s.chosen.as_mut() {
            c.clear();
        }
        let mut rx = Vec::new();
        for p in ps {
            let (tx, r) = std::sync::mpsc::channel();
            s.submit(Submit {
                prompt: p.clone(),
                reply: reply.clone(),
                out: tx,
                abort: Arc::new(AtomicBool::new(false)),
                resumed: Default::default(),
                cached: Default::default(),
                created: Default::default(),
                salt: 0,
                pin: 0,
                images: Vec::new(),
            });
            rx.push(r);
            for _ in 0..gap {
                if s.idle() {
                    break;
                }
                s.tick()?;
            }
        }
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
    // (the last run's sample logits, each sequence's in the order the prompts were given (its id rises with the submits), as bits)
    let seen = |s: &Sched<&mut dyn crate::sched::ModelStep>| -> Vec<Vec<u32>> {
        let mut by: std::collections::BTreeMap<u64, Vec<u32>> = Default::default();
        for &(id, _, l) in s.chosen.iter().flatten() {
            by.entry(id).or_default().push(l.to_bits());
        }
        by.into_values().collect()
    };
    let mut fails = 0;
    let mut alone_logits = Vec::new();
    for (i, p) in prompts.iter().enumerate() {
        let got = ask(&mut s, std::slice::from_ref(p), 0)?.remove(0);
        alone_logits.push(seen(&s).pop().unwrap_or_default());
        let same = got == reference[i];
        fails += !same as usize;
        println!(
            "{} alone {i} ({} prompt tokens): {}",
            if same { "PASS" } else { "FAIL" },
            p.len(),
            if same { "the reply alone, token for token".into() } else { format!("{got:?} against {:?}", reference[i]) }
        );
    }
    let together = ask(&mut s, &prompts, 0)?;
    let same = together.iter().zip(&reference).filter(|(a, b)| a == b).count();
    let same_logits = seen(&s).iter().zip(&alone_logits).filter(|(a, b)| a == b).count();
    // (with a page budget too small for all of them at once (GLYD_ENGINE_PAGES), sequences give their blocks back and run again from their tokens, the generated ones as a prompt: the
    // replies are then counted, not required to be the same)
    let squeezed = std::env::var_os("GLYD_ENGINE_PAGES").is_some();
    let one_a_pass = std::env::var_os("GLYD_ENGINE_BATCH_PREFILL").is_some_and(|v| v == "0");
    let strict = one_a_pass && !squeezed;
    let steps: Vec<u64> = s.context().sched[18..27].iter().map(|x| x.load(std::sync::atomic::Ordering::Relaxed)).collect();
    let resumed = s.context().sched[2].load(std::sync::atomic::Ordering::Relaxed);
    println!(
        "at once: {same} of {} replies the reply alone (prompts {}); {k} sequences, rows in one pass: {batchable}; decode steps by rows {steps:?}; sequences sent back for room {resumed}",
        prompts.len(),
        if strict { "one a pass" } else { "together" }
    );
    if squeezed && resumed == 0 {
        println!("FAIL a page budget of GLYD_ENGINE_PAGES sent no sequence back");
        fails += 1;
    }
    if squeezed {
        // (R4d: the prompts at once again, swapped wherever they can be, by cost, and recomputed only; each request's first token and the time between its tokens, as the request
        // sees them come)
        let timed = |s: &mut Sched<&mut dyn crate::sched::ModelStep>, ps: &[Vec<u32>]| -> Result<(Vec<Vec<u32>>, Vec<f64>, Vec<f64>, f64)> {
            let t0 = std::time::Instant::now();
            let mut seen = Vec::new();
            for p in ps {
                let (tx, rx) = std::sync::mpsc::channel();
                s.submit(Submit {
                    prompt: p.clone(),
                    reply: reply.clone(),
                    out: tx,
                    abort: Arc::new(AtomicBool::new(false)),
                    resumed: Default::default(),
                    cached: Default::default(),
                    created: Default::default(),
                    salt: 0,
                    pin: 0,
                    images: Vec::new(),
                });
                let at = std::time::Instant::now();
                seen.push(std::thread::spawn(move || {
                    let (mut out, mut when) = (Vec::new(), Vec::new());
                    loop {
                        match rx.recv() {
                            Ok(Ev::Token(t)) => {
                                out.push(t);
                                when.push(at.elapsed().as_secs_f64() * 1e3);
                            }
                            Ok(Ev::Lp(_)) => {}
                            Ok(Ev::End(f)) => return (out, when, f.map_err(|e| e.msg)),
                            Err(_) => return (out, when, Err("the scheduler dropped it".into())),
                        }
                    }
                }));
            }
            while !s.idle() {
                s.tick()?;
            }
            let wall = t0.elapsed().as_secs_f64();
            let (mut got, mut first, mut gap) = (Vec::new(), Vec::new(), Vec::new());
            for h in seen {
                let (out, when, f) = h.join().map_err(|_| Error("a request's thread panicked".into()))?;
                if f != Ok(Finish::Length) {
                    return err(format!("a sequence ended {f:?} after {} of its {tokens} tokens", out.len()));
                }
                first.push(when[0]);
                gap.push(if when.len() > 1 { (when[when.len() - 1] - when[0]) / (when.len() - 1) as f64 } else { 0.0 });
                got.push(out);
            }
            Ok((got, first, gap, wall))
        };
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
        let most = |v: &[f64]| v.iter().cloned().fold(0f64, f64::max);
        let mut times = Vec::new();
        for (mode, name) in [(SwapMode::Always, "swap always"), (SwapMode::ByCost, "swap by cost"), (SwapMode::Never, "recompute only")] {
            s.set_swapping(mode);
            let (back, swapped) = (s.stats.preempted, s.stats.swapped);
            let (got, first, gap, wall) = timed(&mut s, &prompts)?;
            let (back, swapped) = (s.stats.preempted - back, s.stats.swapped - swapped);
            let same = got.iter().zip(&reference).filter(|(a, b)| a == b).count();
            println!(
                "{name}: {same} of {} replies the reply alone; {back} sequences sent back for room, {swapped} of them swapped; first token {:.1} ms on average ({:.1} at most), {:.2} ms between tokens on average ({:.2} at most); {:.3} s in all",
                prompts.len(),
                mean(&first),
                most(&first),
                mean(&gap),
                most(&gap),
                wall
            );
            if back == 0 {
                println!("FAIL {name}: a page budget of GLYD_ENGINE_PAGES sent no sequence back");
                fails += 1;
            }
            if mode == SwapMode::Always && swapped == 0 {
                println!("FAIL swap always: no sequence was swapped");
                fails += 1;
            }
            // (swapped, a sequence goes on from the cache it had; recomputed, its generated tokens run again as a prompt, whose kernels round them otherwise)
            if mode == SwapMode::Always && one_a_pass && same < prompts.len() {
                println!("FAIL swap always: replies differ from alone with prompts one a pass");
                fails += 1;
            }
            times.push((name, mean(&first), mean(&gap), wall));
        }
        let (sw, re) = (times[1], times[2]);
        println!(
            "swap by cost against recompute only: first token {:.1} against {:.1} ms ({:+.1}%), between tokens {:.2} against {:.2} ms ({:+.1}%), in all {:.3} against {:.3} s",
            sw.1,
            re.1,
            100.0 * (sw.1 / re.1 - 1.0),
            sw.2,
            re.2,
            100.0 * (sw.2 / re.2 - 1.0),
            sw.3,
            re.3
        );
    }
    if kv12 {
        let back = s.stats.taken_back;
        println!("12-bit pages: {back} steps taken back for void pages, {} sequences ended for want of room, {resumed} sent back for room", s.stats.ended_lost);
        // (R4b's deterministic capacity: small pools (GLYD_KV12_POOL), values that do not compress (GLYD_KV12_SHIFT): the newest sequence is paused where the raw slabs
        // run short, no page is made void and no request fails)
        if std::env::var_os("GLYD_KV12_POOL").is_some() && (back > 0 || s.stats.ended_lost > 0) {
            println!("FAIL small pools (GLYD_KV12_POOL): a step taken back for a void page, or a request ended, where the newest sequence is to be paused");
            fails += 1;
        }
    }
    println!("  at once against alone, logit for logit: {same_logits} of {} replies' every sample logit the same bits", prompts.len());
    // (an expert cache whose mode gives the host pairs (cpu, split, auto): which of a call's misses the host computes, and which are copied or promoted, turns on how many pairs each
    // expert has in the call (xp_plan, `plan`), so a row's sums turn on the rows beside it and alone against at once is counted; runs of the same calls are held to the bit)
    let host_pairs = places > 0 && mode != crate::offload::Mode::Copy;
    if strict && host_pairs {
        println!("  (mode {}: the host's share of a call is planned by all of its rows' pairs, so at once is not alone's calls: counted)", mode.name());
    }
    if strict && !host_pairs && same < prompts.len() {
        println!("FAIL replies at once differ from alone with prompts one a pass");
        fails += 1;
    }
    if strict && !host_pairs && same == prompts.len() && same_logits < prompts.len() {
        println!("FAIL replies at once take other logits than alone with prompts one a pass");
        fails += 1;
    }
    // (--stagger D: the prompts D scheduler rounds apart, as requests reach a server, each joining while those before it decode (GLYD_ENGINE_STEP_TOKENS cuts a prompt
    // mid-way in the rounds that hold rows beside it); the replies against alone, then each against the same arrivals with every other prompt's tokens changed, their
    // lengths kept: a row's sums turn on how many rows its passes hold, never on what the others hold, so with no mixture of experts every reply is the same)
    if let Some(d) = a.flag("stagger") {
        let d: usize = d.parse().map_err(|_| Error(format!("--stagger {d}: a number of rounds")))?;
        let got = ask(&mut s, &prompts, d)?;
        let alone = got.iter().zip(&reference).filter(|(a, b)| a == b).count();
        let mut beside = 0;
        for i in 0..prompts.len() {
            let others: Vec<Vec<u32>> = prompts
                .iter()
                .enumerate()
                .map(|(j, p)| if j == i { p.clone() } else { p.iter().map(|&t| 3 + (t * 31 + 17) % (vocab - 3)).collect() })
                .collect();
            beside += (ask(&mut s, &others, d)?[i] == got[i]) as usize;
        }
        println!(
            "staggered {d} rounds apart: {alone} of {n} replies the reply alone, {beside} of {n} the same beside other prompts' tokens",
            n = prompts.len()
        );
        if strict && alone < prompts.len() {
            println!("FAIL staggered replies differ from alone with prompts one a pass");
            fails += 1;
        }
        if exact && !squeezed && beside < prompts.len() {
            println!("FAIL a staggered reply changed with what the other prompts hold");
            fails += 1;
        }
    }
    if batchable && k > 1 && steps[2..].iter().all(|&x| x == 0) {
        println!("FAIL no decode step had more than one row");
        fails += 1;
    }
    // (a round that is one step, unless GLYD_ENGINE_ONE_STEP=0 or the prompts go one a pass: the longest prompt joins after the first three are decoding, so some step holds both)
    if std::env::var_os("GLYD_ENGINE_ONE_STEP").is_none_or(|v| v != "0") && !strict && !squeezed {
        let both = s.context().sched[28].load(std::sync::atomic::Ordering::Relaxed);
        // (prompts that fit one step's chunk together can all run before any row decodes: a step that holds both is not asked of them)
        let fit_one = prompts.iter().map(Vec::len).sum::<usize>() <= chunk;
        println!(
            "steps of rows that decode and a prompt's chunk together: {both}{}",
            if fit_one { " (the prompts fit one step's chunk together: not asked)" } else { "" }
        );
        if batchable && k > 1 && both == 0 && !fit_one {
            println!("FAIL no step held both rows that decode and a prompt's chunk");
            fails += 1;
        }
    }
    // (the sampler's candidates the step makes on the GPU (src/dynamic/rows.rs `Cands`), against its rows' logits)
    let swapped = s.stats.swapped;
    drop(s);
    // (decode steps of several rows replay a CUDA graph of their padded rows where the description takes one (src/dynamic/rows.rs `rows_bucket`): with prompts one a pass, a
    // description with experts decodes a sequence a step)
    let (replays, late) = (e.rows_replays.get(), e.rows_late.get());
    // (steps launched ahead, GLYD_ENGINE_OVERLAP=1, run as passes of parts: no graph of rows and no pieces there yet)
    let ahead = crate::sched::ModelStep::overlaps(&e);
    if crate::sched::overlap_asked() {
        println!("steps launched before the step before lands (GLYD_ENGINE_OVERLAP=1): {}", if ahead { "yes" } else { "no, this model's steps land first" });
    }
    let graphs = e.rows_graphs() && !(one_a_pass && !e.rows_exact()) && !ahead;
    println!(
        "decode steps of several rows replayed from a CUDA graph: {replays}, graphs captured in a request's step: {late}{}",
        if graphs { "" } else { " (this description's steps of several rows run eager)" }
    );
    if graphs && batchable && k > 1 && steps[2..].iter().any(|&x| x > 0) && replays == 0 {
        println!("FAIL no decode step of several rows replayed a graph");
        fails += 1;
    }
    // (every size captured before the first request (rows.rs `capture_rows`): no step waits on a capture)
    if late > 0 {
        println!("FAIL {late} graphs of decode steps were captured in a request's step, not before the first");
        fails += 1;
    }
    // (graphs against eager: the prompts at once twice more, first with no graph at all (every pass eager: no one-row step's graph, no decode steps' graph, no pieces), then
    // with every graph on and every pass that is no whole graph's in pieces (src/dynamic/pieces.rs), each size captured the first time it is met: the same kernels on the
    // same rows in the same order, so the replies are the same token for token, and where pieces can run, passes replayed them. Not where sequences are sent back for room,
    // which turns on timings)
    let pieces_on = e.pieces_for_check(true, true);
    // (a run of the prompts at once, its replies and its sample logits: launched ahead or not where the step can be, `overlapping`)
    let run = |e: &mut Dyn, ahead: bool| -> Result<(Vec<Vec<u32>>, Vec<Vec<u32>>)> {
        let Some((_, step)) = e.as_step(users)? else { return err(format!("{case}: no scheduler's step")) };
        let mut s = Sched::new(step, k).keeping(false).recording(true).overlapping(ahead);
        let got = ask(&mut s, &prompts, 0)?;
        Ok((got, seen(&s)))
    };
    let same = |a: &(Vec<Vec<u32>>, Vec<Vec<u32>>), b: &(Vec<Vec<u32>>, Vec<Vec<u32>>)| {
        (a.0.iter().zip(&b.0).filter(|(x, y)| x == y).count(), a.1.iter().zip(&b.1).filter(|(x, y)| x == y).count())
    };
    if !squeezed && std::env::var_os("GLYD_KV12_POOL").is_none() && (pieces_on || e.graphs_taken()) {
        let eager = run(&mut e, ahead)?;
        e.pieces_for_check(true, false);
        let (before, rows_before) = (e.piece_replays.get(), e.rows_replays.get());
        let graphs = run(&mut e, ahead)?;
        let (played, rows_played) = (e.piece_replays.get() - before, e.rows_replays.get() - rows_before);
        e.pieces_for_check(false, false);
        let (same_t, same_l) = same(&graphs, &eager);
        let ok = same_t == prompts.len() && same_l == prompts.len() && (played > 0 || !pieces_on || ahead);
        println!(
            "{} graphs on against off: {same_t} of {} replies at once the eager passes', token for token, {same_l} logit for logit; {played} passes replayed their pieces, \
             {rows_played} decode steps their graph{}",
            if ok { "PASS" } else { "FAIL" },
            prompts.len(),
            if pieces_on {
                ""
            } else if !e.stores().is_empty() {
                " (no pieces with an expert cache: a prompt's chunk plans its calls on the host)"
            } else {
                " (no pieces here)"
            }
        );
        fails += !ok as usize;
    }
    // (launched ahead against not, GLYD_ENGINE_OVERLAP=1: the same steps, each landed before the next is launched in the second run)
    if ahead && !squeezed {
        let (on, off) = (run(&mut e, true)?, run(&mut e, false)?);
        let (same_t, same_l) = same(&on, &off);
        let ok = same_t == prompts.len() && same_l == prompts.len();
        println!(
            "{} launched ahead against not: {same_t} of {} replies at once token for token, {same_l} logit for logit",
            if ok { "PASS" } else { "FAIL" },
            prompts.len()
        );
        fails += !ok as usize;
    }
    if squeezed {
        // (the bytes over the link, each way)
        let (out, back) = e.swap_moved();
        println!("swaps: {swapped} sequences, {:.1} MiB copied out and {:.1} MiB back", out as f64 / (1 << 20) as f64, back as f64 / (1 << 20) as f64);
    }
    let Some((_, step)) = e.as_step(users)? else { return err(format!("{case}: no scheduler's step")) };
    let (rows, bad) = cands_check(step, &prompts)?;
    println!("{} the step's candidates of {rows} rows against their logits' {} largest", if bad == 0 { "PASS" } else { "FAIL" }, crate::kern::TOPK);
    fails += bad;
    if fails > 0 {
        return err(format!("{case}: {fails} failed"));
    }
    println!("{case}: PASS");
    Ok(())
}

/// A row's `k` largest logits, largest first, equal ones by the smaller token: the GPU's top-k (kern/mod.rs `topk_rows`) as the CPU makes it.
pub fn host_top(row: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut v: Vec<(u32, f32)> = row.iter().enumerate().map(|(i, &x)| (i as u32, x)).collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    v.truncate(k);
    v
}

/// `ModelStep::top` against the CPU's `host_top` of `ModelStep::logits`: after each prompt's last chunk (its slot's alone, row 0) and after one decode step of
/// every sequence (each row); empty slots taken and given back. The rows looked at, and those whose candidates differ.
fn cands_check(step: &mut dyn crate::sched::ModelStep, prompts: &[Vec<u32>]) -> Result<(usize, usize)> {
    let k = step.slots().unwrap_or(1).min(prompts.len());
    let (mut rows, mut bad) = (0, 0);
    // (none where the top-k kernel does not take the vocabulary, or GLYD_ENGINE_HOST_SAMPLING=1 asks for the row on the host)
    let host = crate::tune!("GLYD_ENGINE_HOST_SAMPLING").is_some();
    let mut check = |step: &mut dyn crate::sched::ModelStep, row: usize, what: &str| -> Result<()> {
        let logits = step.logits(row)?;
        let takes = !host && logits.len().is_multiple_of(8) && logits.len() >= crate::kern::TOPK;
        let want = takes.then(|| host_top(&logits, crate::kern::TOPK));
        let got = step.top(row);
        rows += 1;
        if got != want {
            bad += 1;
            let head = |c: Option<Vec<(u32, f32)>>| c.map(|c| c[..4.min(c.len())].to_vec());
            println!("FAIL candidates {what}: {:?} against {:?}", head(got), head(want));
        }
        Ok(())
    };
    let mut next = Vec::new();
    for (j, p) in prompts.iter().take(k).enumerate() {
        if !step.grow(j as i32, p.len() + 2)? {
            break;
        }
        // (in the step's chunks, as the scheduler runs a prompt: a pass holds `chunk` tokens at most)
        let mut s = None;
        for (i, c) in p.chunks(step.chunk().max(1)).enumerate() {
            s = step.prefill(&[j as i32], i * step.chunk().max(1), c, (i + 1) * step.chunk().max(1) >= p.len())?;
        }
        let s = s.ok_or_else(|| Error("a prompt's last chunk gave no sample".into()))?;
        check(step, 0, &format!("after prompt {j}"))?;
        next.push(s.token);
    }
    let pages: Vec<[i32; 1]> = (0..next.len()).map(|j| [j as i32]).collect();
    let decode: Vec<crate::sched::Row> = next.iter().enumerate().map(|(j, &t)| crate::sched::Row::new(t, prompts[j].len(), &pages[j])).collect();
    if !decode.is_empty() {
        step.decode(&decode)?;
        for r in 0..decode.len() {
            check(step, r, &format!("decode row {r} of {}", decode.len()))?;
        }
    }
    step.release(&(0..next.len() as i32).collect::<Vec<_>>())?;
    Ok((rows, bad))
}
