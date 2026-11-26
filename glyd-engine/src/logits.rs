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

use crate::cpu_cb::Fmt;
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

    /// What a comparison with this fixture shows (`Held`): a fixture with a GPU generation is the engine's own recorded output, a pin.
    pub fn held(&self) -> Held {
        if self.gpu.is_some() {
            Held::Pin
        } else {
            Held::Reference
        }
    }
}

/// What a fixture holds the engine to. `Reference`: another implementation's numbers (a CPU reference, transformers), within a bound: it shows the logits are right
/// to that bound. `Pin`: the engine's own output as one GPU generation recorded it, bit for bit: it shows the logits have not moved since, and nothing of whether
/// they were right when recorded. A case with pins alone is said to be pinned only, never within bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Held {
    Reference,
    Pin,
}

impl Held {
    /// The word a passing line starts with: PASS for a reference's bound kept, PIN for recorded bits unchanged.
    pub fn passed(self) -> &'static str {
        match self {
            Held::Reference => "PASS",
            Held::Pin => "PIN ",
        }
    }
    /// The tag a line's bound carries.
    pub fn tag(self) -> &'static str {
        match self {
            Held::Reference => "a reference",
            Held::Pin => "a pin: the engine's own recorded output",
        }
    }
}

/// A run's comparisons counted by what held them, and the cases nothing or a pin alone held.
#[derive(Default, Debug)]
pub struct Tally {
    pub refs: usize,
    pub pins: usize,
    /// Cases whose only comparisons were pins.
    pub pinned_only: Vec<String>,
    /// Cases with nothing compared, each with why: failures.
    pub none: Vec<(String, String)>,
    /// Cases whose engine is not built: named, not held.
    pub waits: Vec<String>,
}

impl Tally {
    pub fn add(&mut self, h: Held) {
        match h {
            Held::Reference => self.refs += 1,
            Held::Pin => self.pins += 1,
        }
    }
    /// A case's end: `refs` and `pins` its comparisons, `waiting` its engine not built, `others` the GPU generations whose bits it has and this GPU cannot be held to.
    /// The line to print for a case no reference held, if any.
    pub fn case(&mut self, id: &str, refs: usize, pins: usize, waiting: bool, others: &[String]) -> Option<String> {
        if refs > 0 {
            return None;
        }
        if pins > 0 {
            self.pinned_only.push(id.to_string());
            return Some(format!(
                "PIN  {id:12} pinned only: no reference holds this case; its fixtures are the engine's own recorded bits, which show a change and not that the logits are right"
            ));
        }
        if waiting {
            self.waits.push(id.to_string());
            return None;
        }
        if others.is_empty() {
            return None;
        }
        let why = format!("its fixtures are the bits of {}", others.join(", "));
        self.none.push((id.to_string(), why.clone()));
        Some(format!("FAIL {id:12} nothing compared: {why}, another GPU generation"))
    }
    /// The cases that fail for having nothing compared.
    pub fn failed(&self) -> Vec<String> {
        self.none.iter().map(|(c, why)| format!("{c}: nothing compared ({why})")).collect()
    }
    /// "N held to a reference, M pinned, K not held", then the cases pinned only and not held by name.
    pub fn summary(&self) -> String {
        let mut s = format!("{} held to a reference, {} pinned, {} not held", self.refs, self.pins, self.none.len() + self.waits.len());
        if !self.pinned_only.is_empty() {
            s += &format!("; pinned only (no reference): {}", self.pinned_only.join(", "));
        }
        if !self.none.is_empty() {
            s += &format!("; not held: {}", self.none.iter().map(|x| x.0.as_str()).collect::<Vec<_>>().join(", "));
        }
        if !self.waits.is_empty() {
            s += &format!("; not held, no engine yet: {}", self.waits.join(", "));
        }
        s
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
    /// The residual (every token's streams) after each layer of `tokens` as a prompt from a reset: an engine's view of each layer, where it has one.
    fn layers(&mut self, _tokens: &[u32]) -> Result<Option<Vec<Vec<f32>>>> {
        Ok(None)
    }
    /// The routers of the passes run between `choices(true)` and `choices(false)`, which returns them (`Routing`): for each mixture-of-experts layer
    /// in the model's order, each token's experts and the router's input the engine made, in the order the tokens ran. None: the engine has no view
    /// of them. (Traced passes run eager.)
    fn choices(&mut self, _on: bool) -> Result<Option<Routing>> {
        Ok(None)
    }
    /// Each mixture-of-experts block's routed experts as the GPU holds them, decoded on the host (the first products' rows, the second's), where
    /// the engine holds them all there in a lossless level's layout; None: no such blocks, or no view of them; an error: experts on the GPU in
    /// another layout (a lossy level's codes).
    fn held_experts(&mut self) -> Result<Option<Vec<(Vec<u16>, Vec<u16>)>>> {
        Ok(None)
    }
}

/// One row of an engine's indexer in a traced pass (`dynamic::PickTrace`): its position, its picks (the positions its query attends to, in order), and the
/// inputs they were made from as the engine held them: its queries [heads dim], its head weights' product [heads], and the keys of the complete entries
/// its pass's last row sees (its own the first (position + 1) / block of them); where it picks inside a candidate source's published blocks, the entries
/// they hold (`inside`).
pub struct PickRow {
    pub pos: usize,
    pub picks: Vec<usize>,
    pub q: Vec<f32>,
    pub w: Vec<f32>,
    pub keys: std::rc::Rc<Vec<Vec<f32>>>,
    pub inside: Option<Vec<usize>>,
}

/// An engine's picks: for each latent attention with an indexer, in the model's order, its rows in the order they ran.
pub type Picks = Vec<Vec<PickRow>>;

/// `Picks` of an engine's traced passes (`Dyn::trace_picks`), by layer.
fn picks_of(traces: Vec<crate::dynamic::PickTrace>) -> Picks {
    let mut by: BTreeMap<usize, Vec<PickRow>> = BTreeMap::new();
    for t in traces {
        let keys = std::rc::Rc::new(t.keys);
        let rows = t.pos.into_iter().zip(t.picks).zip(t.q.into_iter().zip(t.w)).zip(t.inside);
        by.entry(t.layer).or_default().extend(rows.map(|(((pos, picks), (q, w)), inside)| PickRow { pos, picks, q, w, keys: keys.clone(), inside }));
    }
    by.into_values().collect()
}

/// A row where an engine's picks are not the reference's own (in a run that took the engine's earlier choices): the token (in the order the rows ran), the
/// layer among the latent attentions with an indexer, the pools only the reference took and only the engine took, how far apart the two are on the
/// reference's own inputs (`Picked::margin_between`: information), and the reference's indexer on the engine's own inputs for that row (`mla::choose` on
/// its traced queries, head weights and keys): how far its choice is from the engine's there (0: the same).
pub struct PickApart {
    pub token: usize,
    pub layer: usize,
    pub reference: Vec<usize>,
    pub engine: Vec<usize>,
    pub own_margin: f32,
    pub margin: f32,
}

/// Where the engine's `picks` are not the reference's own choices `picked` (each layer's records, a token at a time, of the reference run that took the
/// engine's picks), each judged by the reference's indexer on the engine's own inputs (`PickApart`); `kinds` the layers' indexers.
pub fn picks_apart(picked: &[Vec<crate::runtime::reference::mla::Picked>], picks: &Picks, kinds: &[crate::runtime::kinds::Indexer]) -> Vec<PickApart> {
    use crate::runtime::reference::mla::choose_among;
    let mut v = Vec::new();
    for (l, ((r, e), ix)) in picked.iter().zip(picks).zip(kinds).enumerate() {
        let b = ix.block.max(1);
        for (t, (r, e)) in r.iter().zip(e).enumerate() {
            if r.own == e.picks {
                continue;
            }
            let on = choose_among(ix, &e.q, &e.w, &e.keys[..((e.pos + 1) / b).min(e.keys.len())], &(0..=e.pos).collect::<Vec<usize>>(), e.inside.as_deref());
            let pools = |a: &[usize], z: &[usize]| {
                let n = r.score.len().max(on.score.len());
                let mut p: Vec<usize> = a.iter().filter(|x| !z.contains(x) && **x < n * b).map(|x| x / b).collect();
                p.dedup();
                p
            };
            v.push(PickApart {
                token: t,
                layer: l,
                reference: pools(&r.own, &e.picks),
                engine: pools(&e.picks, &r.own),
                own_margin: r.margin_between(&e.picks),
                margin: if on.own == e.picks { 0.0 } else { on.margin_between(&e.picks) },
            });
        }
    }
    v
}

/// Each mixture-of-experts layer's choices, in the model's order: each token's experts, in the order the tokens ran.
pub type Choices = Vec<Vec<Vec<usize>>>;

/// An engine's routers in the passes a check traced: each mixture-of-experts layer's choices and, beside each token's, the router's input row as
/// the engine made it (the block's normed input, its bf16 values), in the same order; and each latent attention's indexer's picks with the inputs
/// they were made from (`Picks`: none where the model has no indexer of its own, or the engine gives no view of them).
pub struct Routing {
    pub choices: Choices,
    pub inputs: Vec<Vec<Vec<f32>>>,
    pub picks: Picks,
}

/// `Routing` of the calls an engine made (`dynamic::Routes`: the layer, the experts a row takes, the rows' experts [rows, k], the router's input
/// rows) in the order run.
fn choices_of(calls: Vec<crate::dynamic::Routes>) -> Result<Routing> {
    let mut by: BTreeMap<usize, (Vec<Vec<usize>>, Vec<Vec<f32>>)> = BTreeMap::new();
    for (l, k, ids, input) in calls {
        if k == 0 || ids.len() % k != 0 || ids.iter().any(|&e| e < 0) {
            return err(format!("layer {l}: {} choices of {k} experts a row, or one below 0", ids.len()));
        }
        let rows = ids.len() / k;
        if rows == 0 || input.len() % rows != 0 || input.is_empty() {
            return err(format!("layer {l}: a router's input of {} values for {rows} rows", input.len()));
        }
        let at = by.entry(l).or_default();
        at.0.extend(ids.chunks(k).map(|r| r.iter().map(|&e| e as usize).collect::<Vec<usize>>()));
        at.1.extend(input.chunks(input.len() / rows).map(|r| r.iter().map(|&h| f32::from_bits((h as u32) << 16)).collect::<Vec<f32>>()));
    }
    let (choices, inputs) = by.into_values().unzip();
    Ok(Routing { choices, inputs, picks: Vec::new() })
}

fn widen(v: Vec<Vec<u16>>) -> Vec<Vec<f32>> {
    v.into_iter().map(|r| r.into_iter().map(|h| f32::from_bits((h as u32) << 16)).collect()).collect()
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
    fn layers(&mut self, t: &[u32]) -> Result<Option<Vec<Vec<f32>>>> {
        Dyn::reset(self)?;
        self.trace_layers(true);
        let r = Dyn::prefill(self, t);
        let got = self.trace_layers(false);
        r?;
        Ok(Some(widen(got)))
    }
    fn choices(&mut self, on: bool) -> Result<Option<Routing>> {
        let picks = picks_of(self.trace_picks(on));
        choices_of(self.trace_routes(on)).map(|r| Some(Routing { picks, ..r }))
    }
    fn held_experts(&mut self) -> Result<Option<Vec<(Vec<u16>, Vec<u16>)>>> {
        Dyn::held_experts(self)
    }
}

/// A CPU reference one token at a time: its `step` is every pass's way in, and a prompt is its tokens one after another. `source`: which reference it is, in
/// the words a fixture's provenance gives it.
struct CpuRef<F: FnMut(u32) -> Vec<f32>> {
    load: Box<dyn Fn() -> Result<F>>,
    m: Option<F>,
    last: Vec<f32>,
    source: &'static str,
}

impl<F: FnMut(u32) -> Vec<f32>> Run for CpuRef<F> {
    fn describe(&self) -> String {
        self.source.into()
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
    /// Where Some: the engine is held to the case's CPU reference on its own choices, its routers' and its indexers' (`held_on_choices`), that reference
    /// run taking them by this (the layer kinds' `ref_drive`; DeepSeek-V4's hand-written one, `dsv4_drive`): asked by a case of a described model whose near
    /// ties a bf16 engine takes either way. Its rows against the reference's own are printed all the same.
    pub on_choices: Option<Forced>,
    /// Where Some: what the case holds beside its engine's rows, on each fixture and level of the reference's (`Also`).
    pub also: Option<Also>,
}

/// A CPU reference on a fixture's tokens (`ref_drive`'s order and record), taking an engine's experts and indexer picks where given.
pub type Forced = fn(&Path, &Fixture, Option<&Choices>, Option<&Picks>) -> Result<RefRun>;

/// A case's checks beside its engine's rows against its reference (a GPU's: not under `--cpu-ref`), for a fixture and its level: they print their lines, and
/// say whether they hold (a part that is information only holds). A part that fails fails the case.
pub type Also = fn(&Path, &Fixture, &LevelBound) -> Result<bool>;

/// The fixed list the checks of src/*_ref.rs feed (a first token, then 37 i + 11 modulo the vocabulary).
pub fn token_list(n: usize, vocab: usize, offset: usize) -> Vec<u32> {
    (0..n).map(|i| if i + offset == 0 { 2 } else { (((i + offset - 1) as u32) * 37 + 11) % vocab as u32 }).collect()
}

pub(crate) fn standard_tokens(vocab: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
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

fn kimi_engine(dir: &Path, level: &str) -> Result<Box<dyn Run>> {
    boxed(crate::dynamic::kimi::load(dir, &dyn_opts(level_layout(level)?)))
}

/// DeepSeek-V4's engine without the release's simulated quantizations (the held part of its case, `dsv4_also` says why), or with them as the product runs.
fn dsv4_engine(dir: &Path, level: &str) -> Result<Box<dyn Run>> {
    boxed(crate::dynamic::deepseek_v4::load_sim(dir, &dyn_opts(level_layout(level)?), false))
}

fn dsv4_engine_sim(dir: &Path, level: &str) -> Result<Box<dyn Run>> {
    boxed(crate::dynamic::deepseek_v4::load_sim(dir, &dyn_opts(level_layout(level)?), true))
}

/// DeepSeek-V4's CPU reference (src/dsv4_ref.rs) without the release's simulated quantizations, the function its case's engine computes.
fn dsv4_cpu(dir: &Path) -> Result<Box<dyn Run>> {
    let d = dir.to_path_buf();
    let load = move || crate::dsv4_ref::RefV4::load(&d, false).map(|mut m| move |t| m.step(t));
    Ok(Box::new(CpuRef { load: Box::new(load), m: None, last: Vec::new(), source: "the CPU reference of src/dsv4_ref.rs" }))
}

/// DeepSeek-V4's CPU reference (src/dsv4_ref.rs) on a fixture's tokens as `ref_drive` runs the kinds' (each sequence from a fresh load, a token at a time),
/// taking an engine's experts (`take`: every layer's) and its indexers' picks (`picks`: the entries each query reads, the indexed layers in order) where
/// given: its rows, each token's routing and each indexer's choice as the kinds' reference records them (an entry a pool of its own), and the indexers as
/// the picks' judge takes them (`picks_apart`: the description's, a pool an entry, a row's position its last entry, as `dynamic::Dyn::trace_comp_pick`
/// traces the engine's). Without the release's simulated quantizations (`dsv4_drive_sim` with them).
pub fn dsv4_drive(dir: &Path, fx: &Fixture, take: Option<&Choices>, picks: Option<&Picks>) -> Result<RefRun> {
    dsv4_drive_with(dir, fx, take, picks, false)
}

fn dsv4_drive_sim(dir: &Path, fx: &Fixture, take: Option<&Choices>, picks: Option<&Picks>) -> Result<RefRun> {
    dsv4_drive_with(dir, fx, take, picks, true)
}

/// `dsv4_drive` with the release's simulated quantizations (`sim`) or without them.
fn dsv4_drive_with(dir: &Path, fx: &Fixture, take: Option<&Choices>, picks: Option<&Picks>, sim: bool) -> Result<RefRun> {
    use crate::runtime::kinds::{Body, Indexer, Mixer};
    let first: Vec<u32> = fx.prompt.iter().chain(&fx.decode).copied().collect();
    let mut out = RefRun { rows: Vec::new(), routed: Vec::new(), picked: Vec::new(), indexers: Vec::new() };
    let mut at = 0usize;
    for (tokens, rows_from) in [(&first, fx.prompt.len().max(1) - 1), (&fx.long, fx.long.len().max(1) - 1)] {
        if tokens.is_empty() {
            continue;
        }
        let mut m = crate::dsv4_ref::RefV4::load(dir, sim)?;
        let (layers, csa) = (m.c.layers, m.c.ratios.iter().take(m.c.layers).filter(|&&r| r == 4).count());
        let short = |n: usize, l: &[usize]| n != l.len() || l.iter().any(|&x| x < at + tokens.len());
        if let Some(t) = take {
            if short(layers, &t.iter().map(Vec::len).collect::<Vec<_>>()) {
                return err(format!(
                    "choices of {} layers of {:?} tokens, where the model has {layers} layers of experts",
                    t.len(),
                    t.iter().map(Vec::len).collect::<Vec<_>>()
                ));
            }
            *m.take.borrow_mut() = t.iter().map(|l| l[at..at + tokens.len()].iter().cloned().collect()).collect();
        }
        if let Some(p) = picks {
            if short(csa, &p.iter().map(Vec::len).collect::<Vec<_>>()) {
                return err(format!(
                    "picks of {} layers of {:?} tokens, where the model has {csa} indexers",
                    p.len(),
                    p.iter().map(Vec::len).collect::<Vec<_>>()
                ));
            }
            m.take_picks = p.iter().map(|l| l[at..at + tokens.len()].iter().map(|r| r.picks.clone()).collect()).collect();
        }
        *m.chose.borrow_mut() = Some(Default::default());
        for (i, &t) in tokens.iter().enumerate() {
            let l = m.step(t);
            if i >= rows_from {
                out.rows.push(l);
            }
        }
        let (routed, picked) = m.chose.borrow_mut().take().unwrap_or_default();
        out.routed.resize_with(routed.len().max(out.routed.len()), Vec::new);
        out.routed.iter_mut().zip(routed).for_each(|(a, g)| a.extend(g));
        out.picked.resize_with(picked.len().max(out.picked.len()), Vec::new);
        out.picked.iter_mut().zip(picked).for_each(|(a, g)| a.extend(g));
        at += tokens.len();
    }
    // (the indexers as the judge takes them: the description's, each entry a pool of its own)
    out.indexers = ref_model(dir)?
        .layers
        .iter()
        .filter_map(|l| match &l.block.body {
            Body::Mixer(Mixer::Compressed(c)) => c.indexer.clone().map(|ix| Indexer { block: 1, ..ix }),
            _ => None,
        })
        .collect();
    Ok(out)
}

/// A model directory's description (`runtime::describe`) on the CPU reference of the layer kinds: its weights read as they are stored (bf16 or f32).
fn ref_model(d: &Path) -> Result<crate::runtime::reference::model::RefModel> {
    ref_model_with(d, |_| {})
}

/// `ref_model` of the description `edit` changes (a test's: another rounding point, say).
fn ref_model_with(d: &Path, edit: fn(&mut crate::runtime::describe::Described)) -> Result<crate::runtime::reference::model::RefModel> {
    use crate::runtime::reference::model::{checkpoint_tensor, RefModel};
    let cfg: Value = serde_json::from_str(&io(std::fs::read_to_string(d.join("config.json")))?).map_err(|e| Error(format!("config.json: {e}")))?;
    let mut desc = crate::runtime::describe::describe(&cfg)?;
    edit(&mut desc);
    let ck = io(glyd_gpu::safetensors::Checkpoint::open(d))?;
    let mut m = RefModel::load(&desc, &|n: &str| checkpoint_tensor(&ck, n))?;
    put_token_map(&mut m, d)?;
    Ok(m)
}

/// Engram's compressed vocabulary (`engram_token_map`) into the reference's Engram blocks (role `ngram_map`), where the model in `dir` has them.
pub(crate) fn put_token_map(m: &mut crate::runtime::reference::model::RefModel, dir: &Path) -> Result<()> {
    if let Some(map) = engram_token_map(dir)? {
        for l in
            m.layers.iter_mut().filter(|l| matches!(l.block.body, crate::runtime::kinds::Body::Ngram(g) if g.form == crate::runtime::kinds::NgramForm::Engram))
        {
            l.p.put("ngram_map", &[map.len()], map.clone());
        }
    }
    Ok(())
}

/// `ref_model` a token at a time: the reference of every case with no hand-written one.
fn described_cpu(dir: &Path) -> Result<Box<dyn Run>> {
    described_cpu_with(dir, |_| {})
}

/// `described_cpu` of the description `edit` changes (a test's: another rounding point, say).
fn described_cpu_with(dir: &Path, edit: fn(&mut crate::runtime::describe::Described)) -> Result<Box<dyn Run>> {
    let d = dir.to_path_buf();
    let load = move || {
        let mut m = ref_model_with(&d, edit)?;
        let mut pos = 0u32;
        Ok(move |t: u32| {
            let l = m.forward(&[t], pos).unwrap_or_else(|e| panic!("the reference at position {pos}: {}", e.msg)).pop().expect("a token's logits");
            pos += 1;
            l
        })
    };
    let source = "the model's description (src/runtime/describe.rs) on the CPU reference of the layer kinds (src/runtime/reference)";
    Ok(Box::new(CpuRef { load: Box::new(load), m: None, last: Vec::new(), source }))
}

/// The CPU reference of the layer kinds on a fixture's tokens, as `drive` runs an engine (the prompt and each decode token; then, from a reset, the long
/// prompt), a token a pass: each pass's last logits (`rows`), every token's routing in each mixture-of-experts block, in the model's order (`routed`), and
/// every token's choice in each latent attention with an indexer of its own (`picked`; their indexers, `indexers`).
pub struct RefRun {
    pub rows: Vec<Vec<f32>>,
    pub routed: Vec<Vec<crate::runtime::reference::ffn::Routed>>,
    pub picked: Vec<Vec<crate::runtime::reference::mla::Picked>>,
    pub indexers: Vec<crate::runtime::kinds::Indexer>,
}

/// `RefRun` of the model in `dir` on `fx`'s tokens. With `take`, each token's experts in each block are those (an engine's `Choices`; a token's empty
/// list: the router's own) in the place of the router's own choice; with `picks`, each token's picks in each latent attention with an indexer of its own
/// those (an engine's `Picks`) in the place of the indexer's own: the reference of the kinds on another engine's choices.
pub fn ref_drive(dir: &Path, fx: &Fixture, take: Option<&Choices>, picks: Option<&Picks>) -> Result<RefRun> {
    ref_drive_with(dir, fx, take, picks, |_| {})
}

/// `ref_drive` of the description `edit` changes (another rounding point, say).
fn ref_drive_with(
    dir: &Path,
    fx: &Fixture,
    take: Option<&Choices>,
    picks: Option<&Picks>,
    edit: fn(&mut crate::runtime::describe::Described),
) -> Result<RefRun> {
    use crate::runtime::kinds::{Body, Ffn, Indexer, Mixer};
    use crate::runtime::reference::{comp::CompState, mla::LatentCache, SeqState};
    let first: Vec<u32> = fx.prompt.iter().chain(&fx.decode).copied().collect();
    let mut out = RefRun { rows: Vec::new(), routed: Vec::new(), picked: Vec::new(), indexers: Vec::new() };
    let mut at = 0usize;
    for (tokens, rows_from) in [(&first, fx.prompt.len().max(1) - 1), (&fx.long, fx.long.len().max(1) - 1)] {
        if tokens.is_empty() {
            continue;
        }
        let mut m = ref_model_with(dir, edit)?;
        let moe: Vec<usize> = (0..m.layers.len()).filter(|&b| matches!(m.layers[b].block.body, Body::Ffn(Ffn::Moe(_)))).collect();
        if let Some(t) = take {
            if t.len() != moe.len() || t.iter().any(|l| l.len() < at + tokens.len()) {
                return err(format!(
                    "choices of {} layers of {:?} tokens, where the model has {} mixture-of-experts blocks and the fixture {} tokens",
                    t.len(),
                    t.iter().map(Vec::len).collect::<Vec<_>>(),
                    moe.len(),
                    first.len() + fx.long.len()
                ));
            }
        }
        out.routed.resize_with(moe.len(), Vec::new);
        for (i, &b) in moe.iter().enumerate() {
            m.layers[b].routed = Some(Vec::new());
            if let Some(t) = take {
                m.layers[b].take = t[i][at..at + tokens.len()].iter().cloned().collect();
            }
        }
        // (each latent attention with an indexer of its own, and each compressed attention that picks for itself (an index source: DeepSeek-V4.1's, V4's
        // own indexers): its state of the sequence records every query's choice, and takes the engine's picks; a compressed attention's indexer as the
        // picks' judge takes it, each entry a pool of its own)
        let mla: Vec<(usize, Indexer)> = (0..m.layers.len())
            .filter_map(|b| match &m.layers[b].block.body {
                Body::Mixer(Mixer::Mla(x)) => x.indexer.clone().filter(|ix| !ix.shared).map(|ix| (b, ix)),
                Body::Mixer(Mixer::Compressed(c)) if c.ratio > 0 && c.share.is_none_or(|s| s.picks.is_none()) => {
                    c.indexer.clone().map(|ix| (b, Indexer { block: 1, ..ix }))
                }
                _ => None,
            })
            .collect();
        if let Some(p) = picks {
            if p.len() != mla.len() || p.iter().any(|l| l.len() < at + tokens.len()) {
                return err(format!(
                    "picks of {} layers of {:?} tokens, where the model has {} latent attentions with an indexer and the fixture {} tokens",
                    p.len(),
                    p.iter().map(Vec::len).collect::<Vec<_>>(),
                    mla.len(),
                    first.len() + fx.long.len()
                ));
            }
        }
        out.picked.resize_with(mla.len(), Vec::new);
        out.indexers = mla.iter().map(|(_, ix)| ix.clone()).collect();
        for (i, &(b, _)) in mla.iter().enumerate() {
            let take = picks.map(|p| p[i][at..at + tokens.len()].iter().map(|r| r.picks.clone()).collect()).unwrap_or_default();
            let st = match m.layers[b].block.body {
                Body::Mixer(Mixer::Compressed(_)) => SeqState::Comp(CompState { take, picked: Some(Vec::new()), ..Default::default() }),
                _ => SeqState::Mla(LatentCache { take, picked: Some(Vec::new()), ..Default::default() }),
            };
            m.layers[b].seqs.insert(0, st);
        }
        for (i, &t) in tokens.iter().enumerate() {
            let l = m.forward(&[t], i as u32)?.pop().ok_or_else(|| Error("a token without logits".into()))?;
            if i >= rows_from {
                out.rows.push(l);
            }
        }
        for (i, &b) in moe.iter().enumerate() {
            out.routed[i].extend(m.layers[b].routed.take().unwrap_or_default());
        }
        for (i, &(b, _)) in mla.iter().enumerate() {
            match m.layers[b].seqs.get_mut(&0) {
                Some(SeqState::Mla(c)) => out.picked[i].extend(c.picked.take().unwrap_or_default()),
                Some(SeqState::Comp(c)) => out.picked[i].extend(c.picked.take().unwrap_or_default()),
                _ => {}
            }
        }
        at += tokens.len();
    }
    Ok(out)
}

/// The lossless level's experts on the GPU against the checkpoint's, bit for bit: each mixture-of-experts block's two stacks as an engine holds them
/// (`Run::held_experts`) against the description's own reading of the checkpoint (the CPU reference's parameters: the stored bf16 values). The words
/// that differ: 0 where the level is lossless for the experts. A level's name over another layout's experts is the fault this is for: a random model
/// whose experts sit on a lossy level's grid gives the same logits either way.
fn experts_apart(dir: &Path, held: &[(Vec<u16>, Vec<u16>)]) -> Result<usize> {
    use crate::runtime::kinds::{Body, Ffn};
    let m = ref_model(dir)?;
    let moe: Vec<_> = m.layers.iter().filter_map(|l| if let Body::Ffn(Ffn::Moe(k)) = &l.block.body { Some((l, k)) } else { None }).collect();
    if moe.len() != held.len() {
        return err(format!("{} mixture-of-experts blocks on the GPU, {} in the description", held.len(), moe.len()));
    }
    let mut apart = 0;
    for (b, ((l, k), (up, down))) in moe.iter().zip(held).enumerate() {
        let first = l.p.vec(if k.expert.gated { "experts_gate_up" } else { "experts_up" })?;
        for (what, want, got) in [("first", first, up), ("second", l.p.vec("experts_down")?, down)] {
            if want.len() != got.len() {
                return err(format!("block {b}: the experts' {what} products hold {} values on the GPU, {} in the checkpoint", got.len(), want.len()));
            }
            apart += want.iter().zip(got).filter(|(w, g)| w.to_bits() != (**g as u32) << 16).count();
        }
    }
    Ok(apart)
}

/// `experts_apart` of an engine at the lossless level, in words: None where the engine has no experts to read; else whether every word is the
/// checkpoint's, and the line that says so.
fn experts_line(dir: &Path, run: &mut dyn Run) -> Result<Option<(bool, String)>> {
    Ok(match run.held_experts() {
        Ok(None) => None,
        Ok(Some(h)) => {
            let n = experts_apart(dir, &h)?;
            let words: usize = h.iter().map(|(u, d)| u.len() + d.len()).sum();
            Some((n == 0, format!("the experts on the GPU at the lossless level: {} blocks, {n} of {words} words not the checkpoint's", h.len())))
        }
        Err(e) => Some((false, format!("the experts on the GPU at the lossless level are not a lossless level's: {}", e.msg))),
    })
}

/// The fixture row whose pass ran token `t` of `ref_drive`'s order, and that token's place in words.
fn row_of_token(fx: &Fixture, t: usize) -> (usize, String) {
    let (p, d) = (fx.prompt.len(), fx.decode.len());
    if t < p {
        (0, format!("row 0 (the prompt's token {t})"))
    } else if t < p + d {
        (t - p + 1, format!("row {}", t - p + 1))
    } else {
        (d + 1, format!("row {} (the long prompt's token {})", d + 1, t - p - d))
    }
}

/// An engine's choice of one token that is not the reference's own there: the token (in `ref_drive`'s order), the layer among the mixture-of-experts
/// layers, the experts only the reference took and only the engine took, and how far apart the two choices are by the reference's ranks
/// (`ffn::margin_between`).
pub struct Apart {
    pub token: usize,
    pub layer: usize,
    pub reference: Vec<usize>,
    pub engine: Vec<usize>,
    pub margin: f32,
}

/// Where the engine's `choices` are not the reference's own in `run`: against the reference's own choice at each token (in a run that took the engine's
/// choices, the choice it would have made after the engine's earlier ones).
pub fn apart(run: &RefRun, choices: &Choices) -> Vec<Apart> {
    let mut v = Vec::new();
    let tokens = run.routed.first().map_or(0, Vec::len);
    for t in 0..tokens {
        for (l, (r, c)) in run.routed.iter().zip(choices).enumerate() {
            let (Some(r), Some(c)) = (r.get(t), c.get(t)) else { continue };
            let only = |a: &[usize], b: &[usize]| a.iter().copied().filter(|e| !b.contains(e)).collect::<Vec<usize>>();
            let (reference, engine) = (only(&r.own, c), only(c, &r.own));
            if !reference.is_empty() || !engine.is_empty() {
                v.push(Apart { token: t, layer: l, margin: crate::runtime::reference::ffn::margin_between(&r.rank, &r.own, c), reference, engine });
            }
        }
    }
    v
}

/// What `held_on_choices` found: whether the reference's router, on the engine's own input, takes the engine's choice or one a near tie from it at
/// every choice apart (b1), whether the logits are within the bound of the reference on the engine's choices (b2), and the choices apart.
pub struct OnChoices {
    pub routing: bool,
    pub logits: bool,
    pub apart: Vec<Apart>,
}

/// An engine held to the CPU reference of the layer kinds on the engine's own routers' choices, where a difference past the bound may be a router's
/// choice taken the other way and not arithmetic. `rows` and `routing` are the engine's on `fx`'s tokens (`drive`, `Run::choices`); `own` the
/// reference's own run (`ref_drive` with no choices). Two things are required, and neither has a figure of this run in it:
///   - routing: wherever the engine's choice is not the reference's own (the reference's after the engine's earlier choices), the reference's router
///     (fp32) is run on the ENGINE's own input row for that token and layer (the block's normed input as the engine made it, `Routing::inputs`).
///     Its choice there must be the engine's, or differ from it by less than `kindcheck::NEAR` by its ranks on that same input. A choice that
///     differs by `NEAR` or more on the same input is a router that differs: a fault of the router, not of the arithmetic before it.
///     Why not the margin on the reference's own input: `NEAR` is kindcheck's, where every body runs on the reference's own input, so a choice
///     taken the other way can only come from the body's own arithmetic. In a whole model a later layer's router sees an input that already
///     carries the earlier layers' bf16 roundings (the first random case's fifth layer is 2.3e-2 from fp32 on both of the family's engines),
///     so how near the two choices are on the reference's own input says nothing of the router. That margin is printed, as information.
///   - logits: the engine's rows are within `bound` of the reference run again taking the engine's choices, every row: given the same choices,
///     everything else is the reference's math.
///
/// Each row's KL against the reference's own choices is printed beside the one against the engine's, and each choice apart with both margins.
#[allow(clippy::too_many_arguments)]
pub fn held_on_choices(
    who: &str,
    dir: &Path,
    fx: &Fixture,
    bound: &Bound,
    rows: &[Vec<f32>],
    routing: &Routing,
    own: &RefRun,
    forced: Forced,
) -> Result<OnChoices> {
    use crate::dynamic::kindcheck::NEAR;
    use crate::runtime::kinds::{Body, Ffn};
    use crate::runtime::reference::ffn::{margin_between, route_of};
    let choices = &routing.choices;
    let taken = forced(dir, fx, Some(choices), Some(&routing.picks))?;
    if routing.picks.len() != taken.indexers.len() {
        return err(format!("{who}: picks of {} latent attentions with an indexer, where the model has {}", routing.picks.len(), taken.indexers.len()));
    }
    let (diffs, raw_diffs) = (apart(&taken, choices), apart(own, choices).len());
    let on_taken = Fixture { rows: taken.rows.iter().map(|l| Row::from_logits(l)).collect::<Result<Vec<_>>>()?, ..fx.clone() };
    let v = compare(&on_taken, bound, rows);
    // (b1: the reference's router on the engine's own input row of each choice apart)
    let m = ref_model(dir)?;
    let moe: Vec<usize> = (0..m.layers.len()).filter(|&b| matches!(m.layers[b].block.body, Body::Ffn(Ffn::Moe(_)))).collect();
    let tokens: Vec<u32> = fx.prompt.iter().chain(&fx.decode).chain(&fx.long).copied().collect();
    let mut judged = Vec::with_capacity(diffs.len());
    for d in &diffs {
        let (Some(&b), Some(x), Some(engine), Some(&t)) = (
            moe.get(d.layer),
            routing.inputs.get(d.layer).and_then(|l| l.get(d.token)),
            choices.get(d.layer).and_then(|l| l.get(d.token)),
            tokens.get(d.token),
        ) else {
            return err(format!("{who}: no router's input of token {} in mixture-of-experts layer {}", d.token, d.layer));
        };
        let Body::Ffn(Ffn::Moe(k)) = &m.layers[b].block.body else { return err(format!("block {b} is not a mixture of experts")) };
        let r = route_of(k, &m.layers[b].p, x, t, None)?;
        judged.push((r.own.clone(), margin_between(&r.rank, &r.own, engine)));
    }
    let wrong = |s: f32| s.is_nan() || s >= NEAR;
    let (same, near, far) = (
        judged.iter().filter(|(_, s)| *s == 0.0).count(),
        judged.iter().filter(|(_, s)| *s > 0.0 && !wrong(*s)).count(),
        judged.iter().filter(|(_, s)| wrong(*s)).count(),
    );
    println!(
        "     {who} on its own routers' choices: {} of its choices are not the reference's own there ({raw_diffs} against the reference's own run); on the engine's own router inputs the reference's router takes the engine's choice at {same} of them, a near tie (below {NEAR:e}) at {near}, and differs by {NEAR:e} or more at {far}",
        diffs.len()
    );
    for (d, (on_x, s)) in diffs.iter().zip(&judged) {
        println!(
            "       {}, layer {}: the engine took {:?} where the reference takes {:?} (margin {:.2e} on the reference's own input, information); on the engine's input the reference's router takes {on_x:?}, margin {s:.2e}{}",
            row_of_token(fx, d.token).1,
            d.layer,
            d.engine,
            d.reference,
            d.margin,
            if wrong(*s) { ": a router that differs" } else { "" }
        );
    }
    // (the indexers' picks the same way: the reference's indexer on the engine's own queries, head weights and keys of each pick apart)
    let picks_apart = picks_apart(&taken.picked, &routing.picks, &taken.indexers);
    let pick_far = picks_apart.iter().filter(|d| wrong(d.margin)).count();
    if !taken.indexers.is_empty() {
        let raw: usize = own.picked.iter().zip(&routing.picks).map(|(r, e)| r.iter().zip(e).filter(|(r, e)| r.own != e.picks).count()).sum();
        let same = picks_apart.iter().filter(|d| d.margin == 0.0).count();
        println!(
            "     {who} on its own indexers' picks: {} of its picks are not the reference's own there ({raw} against the reference's own run); on the engine's own indexer inputs the reference's indexer takes the engine's picks at {same} of them, a near tie (below {NEAR:e}) at {}, and differs by {NEAR:e} or more at {pick_far}",
            picks_apart.len(),
            picks_apart.len() - same - pick_far
        );
        for d in &picks_apart {
            println!(
                "       {}, indexer {}: the engine took pools {:?} where the reference takes {:?} (margin {:.2e} on the reference's own input, information); on the engine's input, margin {:.2e}{}",
                row_of_token(fx, d.token).1,
                d.layer,
                d.engine,
                d.reference,
                d.own_margin,
                d.margin,
                if wrong(d.margin) { ": an indexer that differs" } else { "" }
            );
        }
    }
    println!(
        "     {who} against the reference taking its choices: KL mean {:.2e} max {:.2e}, best token {}/{} with a lead ({})",
        v.mean_kl,
        v.max_kl,
        v.top1.0,
        v.top1.1,
        if v.ok { "within the bound".to_string() } else { format!("PAST the bound: {}", v.notes.join("; ")) }
    );
    for (i, (g, (o, t))) in rows.iter().zip(fx.rows.iter().zip(&on_taken.rows)).enumerate() {
        let here: Vec<String> = diffs
            .iter()
            .filter(|d| row_of_token(fx, d.token).0 == i)
            .map(|d| format!("layer {} {:?} for {:?} ({:.1e})", d.layer, d.engine, d.reference, d.margin))
            .chain(
                picks_apart
                    .iter()
                    .filter(|d| row_of_token(fx, d.token).0 == i)
                    .map(|d| format!("indexer {} pools {:?} for {:?} ({:.1e})", d.layer, d.engine, d.reference, d.own_margin)),
            )
            .collect();
        println!(
            "       row {i:2}: KL {:.2e} against the reference's own choices, {:.2e} against the engine's; {}",
            kl_of(o, g),
            kl_of(t, g),
            match here.len() {
                0 => "no choice apart in its pass".to_string(),
                n if n > 4 => format!("{n} apart in its pass (each above): {}, ...", here[..4].join(", ")),
                n => format!("{n} apart in its pass: {}", here.join(", ")),
            }
        );
    }
    Ok(OnChoices { routing: far == 0 && pick_far == 0, logits: v.ok, apart: diffs })
}

/// DeepSeek-V4's case, held in three parts, none with a number set from a run:
///   1. its engine and its reference both WITHOUT the release's simulated quantizations (`dsv4_engine`, `dsv4_cpu`, `dsv4_drive`: the fixture is that reference's), on the
///      engine's own routers' and indexers' choices at the kind's bound (KL mean 1e-4, max 2.5e-4, best token 85%): the arithmetic of the model, whose reference's noise
///      floor is within a tenth of that bound (`tests::deepseek_v4_noise_floor`). That is the verdict of `check` on the case's line.
///   2. the operations the simulation is made of, here: each operation that quantizes as the release does, fed the reference's own input, bit for bit
///      (`dsv4_ref::qcheck_rows`). The case fails where one differs: part 1 leaves them out, and this is where a fault of theirs shows.
///   3. the whole model with them (the product's default), the engine against the reference that simulates them: its figures against the reference's own choices
///      and on the engine's, each choice apart, printed, never a pass or fail. A rounding before a quantizer moves a value near a grid point's edge to the next
///      grid point (an E4M3 step is 6 to 12% of a value, E2M1's more), and such values are a fixed share of every activation whatever its scale, so no draw
///      lowers it: the reference's own noise floor with the simulation, its rounding points against itself in fp32, is eight times the kind's bound
///      (the Mac, 2026-11-25: KL mean 7.75e-4, max 1.04e-3), and an engine that is not the reference bit for bit before each quantizer cannot be held to the
///      kind's bound by it. What the printed figures say is where it stands, not a verdict.
fn dsv4_also(dir: &Path, fx: &Fixture, lb: &LevelBound) -> Result<bool> {
    println!("     the quantizing operations on the reference's own input, bit for bit:");
    let ops = match crate::dsv4_ref::qcheck_rows(64) {
        Ok(()) => true,
        Err(e) => {
            println!("FAIL {}", e.msg);
            false
        }
    };
    let mut run = dsv4_engine_sim(dir, &lb.level)?;
    let own = dsv4_drive_sim(dir, fx, None, None)?;
    let fx = Fixture { rows: own.rows.iter().map(|r| Row::from_logits(r)).collect::<Result<_>>()?, ..fx.clone() };
    run.choices(true)?;
    let rows = drive(run.as_mut(), &fx.prompt, &fx.decode, &fx.long)?;
    let Some(routing) = run.choices(false)? else { return err("deepseek-v4: its engine gives no view of its routers' choices") };
    let v = compare(&fx, &lb.bound, &rows);
    println!(
        "     the whole model with the release's simulated quantizations (information, never a pass or fail): the engine against the reference that simulates them, on the reference's own choices KL mean {:.2e} max {:.2e}, best token {}/{} with a lead",
        v.mean_kl, v.max_kl, v.top1.0, v.top1.1
    );
    held_on_choices("the engine with the simulation", dir, &fx, &lb.bound, &rows, &routing, &own, dsv4_drive_sim)?;
    Ok(ops)
}

fn nemotron_engine(dir: &Path, level: &str) -> Result<Box<dyn Run>> {
    nemotron_at(dir, level_layout(level)?)
}

fn nemotron_at(dir: &Path, layout: Option<Layout>) -> Result<Box<dyn Run>> {
    boxed(crate::dynamic::nemotron::load(dir, &dyn_opts(layout), layout.unwrap_or(Layout::Tiered)))
}

/// The case's engine on the model in `dir` at `level` with its packs in `layout`: as the case's own engine (`Case::engine`) runs it, the layout given where the level
/// leaves it to the GPU (penguin: the tiered layout on Ada, the 12-bit one on other GPUs). `packcheck --engine` loads a saved directory and the model it was saved
/// from this way, so that both are in the layout the directory was saved in on any GPU. The cases of `packcheck::SAVED`.
pub(crate) fn engine_at(case: &str, dir: &Path, level: &str, layout: Layout) -> Result<Box<dyn Run>> {
    if let Some(l) = level_layout(level)?.filter(|&l| l != layout) {
        return err(format!("{case} at {level} is packed {}, not {}", l.name(), layout.name()));
    }
    let o = dyn_opts(Some(layout));
    match case {
        "gemma4" | "gemma4-kv" => gemma4_at(dir, Some(layout)),
        "qwen35" | "qwen35-wide" | "qwen35moe" => boxed(crate::dynamic::qwen35::load(dir, &o)),
        "dense" => boxed(crate::dynamic::dense::load(dir, &o)),
        "nemotron-h" => nemotron_at(dir, Some(layout)),
        other => err(format!("{other}: no saved format of this case is checked (packcheck::SAVED)")),
    }
}

/// GLM-5.3-Flash at a level: its experts at six, five or four bits (the release's levels, as its experts are FP8 there), or at penguin as the random model holds
/// them (bf16, the lossless layout `kindcheck` loads): the level the description's CPU reference is held to.
fn glm_engine(dir: &Path, level: &str) -> Result<Box<dyn Run>> {
    boxed(crate::dynamic::glm::load(dir, &dyn_opts(Some(Layout::Tiered)), level_layout(level)?.unwrap_or(Layout::Tiered)))
}

/// GLM-5.3's layout by its kinds alone (`dynamic::glm::load_plain`), the experts at the level asked (penguin's: bf16, as the random model holds them).
fn glm_dsa_engine(dir: &Path, level: &str) -> Result<Box<dyn Run>> {
    boxed(crate::dynamic::glm::load_plain(dir, &dyn_opts(level_layout(level)?)))
}

/// Gemma 4 on the description executor, as `serve` runs it (no draft model).
fn gemma4_engine(dir: &Path, level: &str) -> Result<Box<dyn Run>> {
    gemma4_at(dir, level_layout(level)?)
}

fn gemma4_at(dir: &Path, layout: Option<Layout>) -> Result<Box<dyn Run>> {
    let c = crate::gemma4::ConfigG4::load(dir)?;
    let o = DynOpts { ctx: 1024, chunk: 64, graph: false, device: 0, layout };
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
    gemma4_shaped_random(out, GEMMA4_RANDOM_CONFIG, 0x2026_1120_0000_0004)
}

/// A random model of a Gemma 4 config `text`, every tensor at the shape its description gives (`Described::shapes`) and drawn from `seed` in the order of
/// their names: the q and k norms' weights in [0.1, 0.3], the other norms in [0.5, 1.5], layer scalars in [0.5, 1], every matrix in +-sqrt(3 / its
/// columns), the embedding (the tied head) too. The sizes the E2B and E4B releases add (KV sharing, the double-wide MLP) come from the description, so
/// each config's model is its own shape. Both are the draw's conditioning (`tests::gemma4_reference_and_its_draws` measures it):
/// - Gemma 4's scores are not scaled: the q and k norms set them. With those weights in [0.5, 1.5] a random model's scores are about +-17 (RMS) at heads
///   of 256 and +-24 at 512, its softmax nearly one-hot, and a near tie moves under 2^-9 of rounding: on an A10, eight-layer models of six seeds were 2e-3
///   to 3.2e-1 in KL mean from the CPU reference, with KV sharing and without alike.
/// - The embedding in [-1, 1] made the scaled embedding's RMS 13 against the layers' outputs' 1, and the tied head's largest logit 30.00 after the
///   soft-cap (four seeds), where the release's bf16 logits are 1/8 apart: with the q and k norms in [0.1, 0.3] it alone left the CPU reference 8e-5
///   to 1.1e-4 in KL mean from itself with the engine's storage roundings (the norms' outputs and the embedding's scale bf16), 8e-6 to 1.2e-5 with an
///   f32 head. With both draws it was 3e-4 to 1.1e-3; with these ranges, 6e-6 to 1.1e-5.
pub fn gemma4_shaped_random(out: &Path, text: &str, seed: u64) -> Result<()> {
    gemma4_drawn(out, text, seed, (0.1, 0.3), None)
}

/// `gemma4_shaped_random` with the q and k norms' weights in `qk` and the embedding in +-`emb` where given (the draws `gemma4_shaped_random` replaced,
/// which the conditioning test measures).
fn gemma4_drawn(out: &Path, text: &str, seed: u64, qk: (f32, f32), emb: Option<f32>) -> Result<()> {
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
        let (lo, hi) = if let Some(e) = emb.filter(|_| *name == d.embed) {
            (-e, e)
        } else if name.ends_with("layer_scalar") {
            (0.5, 1.0)
        } else if name.ends_with("q_norm.weight") || name.ends_with("k_norm.weight") {
            qk
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

/// Qwen3.8-Flash-Next on the description executor (src/dynamic/qwen4x.rs) at every level, as the other cases' engines: the matrices at the level (penguin's lossless
/// packs where none is), the experts lossless at penguin and in a lossy level's codes at the others. Condor's own engine runs no level of the case: the family is held to
/// the CPU reference of the kinds, not to that engine's bits (its kestrel rows on an RTX 4080 SUPER are its record, tests/fixtures/qwen4x_condor_rows.json.gz, which
/// the reference's own test reads), and `paritycheck` runs the two side by side while that engine exists. The penguin fixture's bound is the kind's (the Qwen3.5
/// layout's fixture's: KL mean 1e-4, max 2.5e-4), and the case asks to be held on its routers' choices (`Case::on_choices`): the random model's 16 experts, 4 a token,
/// leave choices within 1e-2 of the next expert (`reference::ffn::margin`), which a bf16 engine may take either way, and one taken the other way moves its row and
/// the rows after it by more than the bound allows arithmetic. Such a choice is judged by the reference's router on the engine's own input row, and the logits are
/// held, at the same bound, to the reference taking the engine's choices (`held_on_choices`); the figures against the reference's own choices are printed beside them.
fn qwen4x_engine(dir: &Path, level: &str) -> Result<Box<dyn Run>> {
    boxed(crate::dynamic::qwen4x::load(dir, &dyn_opts(level_layout(level)?)))
}

/// The random tiny Qwen3.8-Flash-Next of tools/qwen4x_ref.py (8 layers: three Gated DeltaNet and one indexed attention, twice; a layer with per-layer embeddings; 16 experts; hyper-connections): its tensors'
/// names, types and shapes from tests/fixtures/qwen4x_tiny_manifest.json (what that tool wrote), their values at random from a fixed seed by the tool's rules for each kind of tensor.
pub fn qwen4x_random(out: &Path) -> Result<()> {
    manifest_random(include_str!("../tests/fixtures/qwen4x_tiny_manifest.json"), out, 0x2026_1120_0000_0005, false, "Qwen3.8-Flash-Next of the tiny shape")
}

/// A second random model of the Qwen3.8-Flash-Next family (`qwen4_exp`) with other counts than the release's: the tiny model's manifest with 2 residual streams,
/// index heads of 64 over pools of 2 and no n-gram embedding (its kernels take four streams yet); each tensor's shape the counts make (the gated residuals'
/// [2 hidden] norms, [rank, 2 hidden] and [2 hidden, rank] products, [2, 2 hidden] inject weights; the index product's (heads + 1) 64 rows, its norms of 64).
/// Its experts are drawn as its other matrices are, not on the six-bit grid the first case's sit on: only a lossless load holds them.
pub fn qwen4x_s2_random(out: &Path) -> Result<()> {
    manifest_random(&qwen4x_s2_manifest()?, out, 0x2026_1125_0000_0002, false, "Qwen3.8-Flash-Next family of 2 streams, index heads of 64 over pools of 2")
}

/// The places a layer the expert cache's random case keeps of its 8 experts: fewer than a token takes (4), so every call has misses the host computes.
const QWEN4X_CB_PLACES: usize = 3;

/// The random case of the expert cache (`qwen4x-cb`): `qwen4x_codes_random` with every expert two-bit (the "cb2" form, the release's earliest fast pack's).
pub fn qwen4x_cb_random(out: &Path) -> Result<()> {
    qwen4x_codes_random(out, codes_form("cb2")?)
}

/// The forms of a codes pack the random cases are made in, each its experts' formats (up, down) in turn over a layer's 8 experts: each width alone, a mix of an
/// expert's two matrices in two formats, and the release's packs' mixes (glyd/Qwen3.8-Flash-Next-hummingbird's meta.json, 49 layers of 512 experts each:
/// precise-v2 22,592 q3 and 2,496 q, best-v2 14,672 q3 and 10,416 q4, fast every one q, precise 21,632 cb3 and 3,456 cb2, best 17,696 cb3 and 7,392 cb4, the
/// earlier fast `pack` every one cb2: the "cb2" form).
pub const CODES_FORMS: [(&str, &[(Fmt, Fmt)]); 13] = [
    ("cb2", &[(Fmt::Cb2, Fmt::Cb2)]),
    ("cb3", &[(Fmt::Cb3, Fmt::Cb3)]),
    ("cb4", &[(Fmt::Cb4, Fmt::Cb4)]),
    ("tern", &[(Fmt::Tern, Fmt::Tern)]),
    ("q", &[(Fmt::Tq, Fmt::Tq)]),
    ("q3", &[(Fmt::Tq3, Fmt::Tq3)]),
    ("q4", &[(Fmt::Tq4, Fmt::Tq4)]),
    ("mixed", &[(Fmt::Cb4, Fmt::Tq), (Fmt::Tern, Fmt::Cb3), (Fmt::Tq3, Fmt::Cb2), (Fmt::Cb2, Fmt::Tq4)]),
    ("precise-v2", &[(Fmt::Tq3, Fmt::Tq3), (Fmt::Tq3, Fmt::Tq3), (Fmt::Tq3, Fmt::Tq3), (Fmt::Tq, Fmt::Tq)]),
    ("best-v2", &[(Fmt::Tq3, Fmt::Tq3), (Fmt::Tq4, Fmt::Tq4)]),
    ("fast", &[(Fmt::Tq, Fmt::Tq)]),
    ("precise", &[(Fmt::Cb3, Fmt::Cb3), (Fmt::Cb3, Fmt::Cb3), (Fmt::Cb3, Fmt::Cb3), (Fmt::Cb2, Fmt::Cb2)]),
    ("best", &[(Fmt::Cb3, Fmt::Cb3), (Fmt::Cb4, Fmt::Cb4)]),
];

/// The formats of the form named `name` (`CODES_FORMS`), or the refusal that lists them.
pub fn codes_form(name: &str) -> Result<&'static [(Fmt, Fmt)]> {
    CODES_FORMS
        .iter()
        .find(|f| f.0 == name)
        .map(|f| f.1)
        .ok_or_else(|| Error(format!("a codes form {name:?}: one of {}", CODES_FORMS.iter().map(|f| f.0).collect::<Vec<_>>().join(", "))))
}

/// A random model of the family at the codebook kernels' widths with its routed experts in a codes pack of the form `form` (`CODES_FORMS`: each expert's formats
/// in turn): the tiny model's config at hidden 2,560, experts of 640 neurons (`moe_cb::takes`), two layers (a Gated DeltaNet one, an indexed attention one),
/// 8 experts of which 4 a token, no n-gram embedding. Its routed experts are random experts of those formats (`cb_random::CodesExpert`) written as a pack in
/// `pack/`, and the checkpoint holds them in the model's basis, fp32 (`CodesExpert::weights`), which the CPU reference reads as stored; the shared expert is on the
/// grid the codes block holds it on (int8 rows with a scale each: a row k 2^-m with integers k, the largest |k| 127), so the block's copy is the checkpoint's. Every
/// other tensor of the description at random by the family's rules (`manifest_random`). A shard of each (`model.safetensors`, `experts.safetensors`) and the index
/// that names them. Two-bit experts are the draws of the form before the others were made (the `qwen4x-cb` case's model is the same bytes).
pub fn qwen4x_codes_random(out: &Path, form: &[(Fmt, Fmt)]) -> Result<()> {
    codes_model(out, form, false)
}

/// The release's form of a directory of the family (glyd/Qwen3.8-Flash-Next-hummingbird's `model/` and a pack of its experts): `qwen4x_codes_random` in the
/// form `precise-v2` (the release's default pack's formats) and a third layer, with an n-gram embedding before its second layer (a Gated DeltaNet one, as the
/// release's) at the release's row of 160 values (its `ple_embed_dim` the hidden width), the checkpoint slim (none of its routed experts: they are the pack's alone, `pack/`) and its n-gram table stored at 4 bits
/// (`tables_at_4_bits`), as the release stores it; its index says so in its metadata, in the release's words.
pub fn qwen4x_release_random(out: &Path) -> Result<()> {
    codes_model(out, codes_form("precise-v2")?, true)?;
    tables_at_4_bits(out, &|n| n.contains(".ngram_embedding."))
}

/// `qwen4x_codes_random`, or with `release` the release's form (`qwen4x_release_random`, but its table's form: written in bf16 here).
fn codes_model(out: &Path, form: &[(Fmt, Fmt)], release: bool) -> Result<()> {
    use crate::cb_random::{write_codes, CodesExpert, Rng};
    let m: Value = serde_json::from_str(include_str!("../tests/fixtures/qwen4x_tiny_manifest.json")).map_err(|e| Error(e.to_string()))?;
    let mut c: Value = serde_json::from_str(m["config"].as_str().unwrap_or("{}")).map_err(|e| Error(e.to_string()))?;
    // (the release's form: a third layer, its n-gram block before the second (a Gated DeltaNet one), as the release's `ple_layer_ids` [2])
    let types: &[&str] = if release { &["linear_attention", "linear_attention", "indexed_attention"] } else { &["linear_attention", "indexed_attention"] };
    let (hidden, inter, experts, layers) = (2560usize, 640usize, 8usize, types.len());
    let t = &mut c["text_config"];
    for (k, v) in [
        ("hidden_size", json!(hidden)),
        ("moe_intermediate_size", json!(inter)),
        ("shared_expert_intermediate_size", json!(inter)),
        ("num_experts", json!(experts)),
        ("num_hidden_layers", json!(layers)),
        ("layer_types", json!(types)),
        ("ple_layer_ids", if release { json!([2]) } else { json!([]) }),
    ] {
        t[k] = v;
    }
    // (the n-gram block's convolution states, the tiny model's 3, where it has one)
    match release {
        true => t["ple_embed_dim"] = json!(hidden),
        false => t["number_of_conv_states"] = json!(1),
    }
    let text = c.to_string();
    let d = crate::runtime::describe::describe(&c)?;
    let routed = |n: &str| n.contains(".mlp.experts.");
    let shared = |n: &str| n.contains(".mlp.shared_expert.");
    // (the n-gram hash's int64 tensors as the tiny manifest records them: its vocabulary, seed, base and heads are these)
    let i64s: BTreeMap<&str, &Value> =
        m["tensors"].as_array().into_iter().flatten().filter(|e| e[1] == "I64").filter_map(|e| Some((e[0].as_str()?, e))).collect();
    let rest: Vec<Value> = d
        .shapes
        .iter()
        .filter(|(n, _)| !routed(n) && !shared(n))
        .map(|(n, sh)| i64s.get(n.as_str()).map_or_else(|| json!([n, "BF16", sh]), |e| (*e).clone()))
        .collect();
    let manifest = json!({"config": text, "experts": OFF_GRID, "tensors": rest}).to_string();
    manifest_random(&manifest, out, 0x2026_1125_0000_00cb, false, "Qwen3.8-Flash-Next family at the codebook kernels' widths (all but its experts)")?;
    // (the routed experts: a pack's, and their weights in the model's basis; the shared expert on the int8 grid)
    let mut rng = Rng(0x2026_1125_0000_0cb2);
    let mut pack: Vec<(Vec<i8>, Vec<CodesExpert>)> = Vec::new();
    let mut shard: Vec<(String, &str, Vec<u64>, Vec<u8>)> = Vec::new();
    let le = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    for l in 0..layers {
        let signs: Vec<i8> = (0..hidden).map(|_| if rng.sym() < 0.0 { -1 } else { 1 }).collect();
        if form.is_empty() {
            return err("a codes form of no formats");
        }
        let ex: Vec<CodesExpert> = (0..experts).map(|e| CodesExpert::random(hidden, inter, form[e % form.len()], &mut rng)).collect();
        let (mut gu, mut dn) = (Vec::with_capacity(experts * 2 * inter * hidden), Vec::with_capacity(experts * hidden * inter));
        for e in &ex {
            let (g, w) = e.weights(&signs, hidden, inter);
            gu.extend(g);
            dn.extend(w);
        }
        let of = |what: &str| d.shapes.keys().find(|n| n.contains(&format!("layers.{l}.mlp.{what}"))).cloned();
        for (what, v) in [("experts.gate_up_proj", gu), ("experts.down_proj", dn)] {
            let name = of(what).ok_or_else(|| Error(format!("no {what} of layer {l} in the description")))?;
            let shape: Vec<u64> = d.shapes[&name].iter().map(|&x| x as u64).collect();
            if shape.iter().product::<u64>() as usize != v.len() {
                return err(format!("{name}: the description's {shape:?}, {} weights made", v.len()));
            }
            shard.push((name, "F32", shape, le(&v)));
        }
        for what in ["shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert.down_proj"] {
            let name = of(what).ok_or_else(|| Error(format!("no {what} of layer {l} in the description")))?;
            let sh = d.shapes[&name].clone();
            let (rows, cols) = (sh[0], sh[1]);
            // (each row integers in [-127, 127], one of them +-127, times a power of two near the row's usual size: exact in bf16 and in the block's int8 rows)
            let m = (3f32.sqrt() / (cols as f32).sqrt() / 127.0).log2().round();
            let mut b = Vec::with_capacity(rows * cols * 2);
            for _ in 0..rows {
                let mut k: Vec<f32> = (0..cols).map(|_| (rng.sym() * 127.0).round()).collect();
                k[(rng.next() % cols as u64) as usize] = if rng.sym() < 0.0 { -127.0 } else { 127.0 };
                b.extend(k.iter().flat_map(|&q| (((q * m.exp2()).to_bits() >> 16) as u16).to_le_bytes()));
            }
            shard.push((name, "BF16", sh.iter().map(|&x| x as u64).collect(), b));
        }
        pack.push((signs, ex));
    }
    // (the release's checkpoint is slim: its routed experts the pack's alone)
    shard.retain(|x| !(release && routed(&x.0)));
    let refs: Vec<(&str, &str, Vec<u64>, &[u8])> = shard.iter().map(|(n, t, s, b)| (n.as_str(), *t, s.clone(), &b[..])).collect();
    crate::pack::write_safetensors(&out.join("experts.safetensors"), &[("format", "pt".to_string())], &refs)?;
    let mut map = serde_json::Map::new();
    for n in d.shapes.keys().filter(|n| !(release && routed(n))) {
        map.insert(n.clone(), json!(if routed(n) || shared(n) { "experts.safetensors" } else { "model.safetensors" }));
    }
    let index = match release {
        true => json!({"metadata": {"glyd": "slim: routed experts in the Glyd pack; n-gram table at 4 bits (NF4, a bf16 scale a row)"}, "weight_map": map}),
        false => json!({"weight_map": map}),
    };
    io(std::fs::write(out.join("model.safetensors.index.json"), index.to_string()))?;
    write_codes(&out.join("pack"), hidden, inter, &pack)
}

/// A case's name and whether it asks for its tables read by rows stored at 4 bits: `CASE+q4` (`tables_at_4_bits`, every table its description holds on the
/// host: the n-gram table's shards and the embedding, for the family's cases).
pub fn q4_case(case: &str) -> (&str, bool) {
    case.strip_suffix("+q4").map_or((case, false), |b| (b, true))
}

/// The tables the description of the model in `dir` reads by rows on the host (`tables::host_names`) that `keep` keeps, stored at 4 bits in place of their
/// bf16 rows (`formats::q4_name`, as Glyd's slim checkpoints store the n-gram table): each row its scale (its largest magnitude, rounded to bf16), then each
/// value's nearest NF4 level of the scale's (`formats::nf4_encode`). Every file that holds one is written again, the index's names with it. The tables' bf16
/// rows are then the values their 4-bit rows decode to (`tables_decoded`).
pub fn tables_at_4_bits(dir: &Path, keep: &dyn Fn(&str) -> bool) -> Result<()> {
    let text = io(std::fs::read_to_string(dir.join("config.json")))?;
    let names: std::collections::BTreeSet<String> = crate::dynamic::tables::host_names(&text).into_iter().filter(|n| keep(n)).collect();
    if names.is_empty() {
        return err(format!("{}: its description holds no table read by rows on the host to store at 4 bits", dir.display()));
    }
    let n = rewrite_tensors(dir, &|name, dtype, shape, b| {
        if !names.contains(name) {
            return Ok(None);
        }
        let cols = shape.get(1).copied().unwrap_or(0) as usize;
        if dtype != "BF16" || shape.len() != 2 || cols % 2 != 0 {
            return err(format!("{name}: {dtype} {shape:?}, where a bf16 table of an even number of columns is stored at 4 bits"));
        }
        let q4 = crate::formats::q4_name(name).ok_or_else(|| Error(format!("{name}: no .weight to name its 4-bit form by")))?;
        let rows: Vec<u8> = b.chunks_exact(cols * 2).flat_map(crate::formats::nf4_encode).collect();
        Ok(Some((q4, "U8".into(), vec![shape[0], 2 + cols as u64 / 2], rows)))
    })?;
    if n != names.len() {
        return err(format!("{}: {n} of the {} tables on the host stored at 4 bits", dir.display(), names.len()));
    }
    Ok(())
}

/// `tables_at_4_bits`' tables back in bf16: each 4-bit table (`X.q4`) of the checkpoint in `dir` stored as `X.weight`, the bf16 rows it decodes to
/// (`formats::nf4_row`); its index's names with it. How many it wrote.
pub fn tables_decoded(dir: &Path) -> Result<usize> {
    rewrite_tensors(dir, &|name, dtype, shape, b| {
        if !name.ends_with(".q4") {
            return Ok(None);
        }
        let cols = (shape.get(1).copied().unwrap_or(2) as usize).saturating_sub(2) * 2;
        let rb = crate::formats::q4_row_bytes(dtype, shape, cols).ok_or_else(|| Error(format!("{name}: {dtype} {shape:?}, not a table at 4 bits")))?;
        let mut out = vec![0u8; b.len() / rb * cols * 2];
        for (q, o) in b.chunks_exact(rb).zip(out.chunks_exact_mut(cols * 2)) {
            crate::formats::nf4_row(q, o);
        }
        Ok(Some((crate::formats::stands_for(name).into_owned(), "BF16".into(), vec![shape[0], cols as u64], out)))
    })
}

/// A tensor's new name, type, shape and bytes, where a rewrite replaces it.
type Rewritten = Option<(String, String, Vec<u64>, Vec<u8>)>;

/// Each safetensors file of the checkpoint in `dir` that holds a tensor `f` replaces (`f(name, dtype, shape, bytes)`), written again with the replacements
/// in their places (its metadata `format: pt`), and the index's names with them; how many were replaced.
fn rewrite_tensors(dir: &Path, f: &dyn Fn(&str, &str, &[u64], &[u8]) -> Result<Rewritten>) -> Result<usize> {
    let ck = io(glyd_gpu::safetensors::Checkpoint::open(dir))?;
    let mut renamed: BTreeMap<String, String> = BTreeMap::new();
    for file in &ck.files {
        let (mut out, mut changed) = (Vec::new(), false);
        for (name, t) in &file.tensors {
            let b = io(file.read(t))?;
            match f(name, &t.dtype, &t.shape, &b)? {
                Some(x) => {
                    renamed.insert(name.clone(), x.0.clone());
                    out.push(x);
                    changed = true;
                }
                None => out.push((name.clone(), t.dtype.clone(), t.shape.clone(), b)),
            }
        }
        if changed {
            let refs: Vec<(&str, &str, Vec<u64>, &[u8])> = out.iter().map(|(n, d, sh, b)| (n.as_str(), d.as_str(), sh.clone(), &b[..])).collect();
            crate::pack::write_safetensors(&file.path, &[("format", "pt".to_string())], &refs)?;
        }
    }
    let idx = dir.join("model.safetensors.index.json");
    if idx.exists() && !renamed.is_empty() {
        let mut v: Value = serde_json::from_str(&io(std::fs::read_to_string(&idx))?).map_err(|e| Error(format!("{}: {e}", idx.display())))?;
        if let Some(map) = v.get_mut("weight_map").and_then(Value::as_object_mut) {
            for (old, new) in &renamed {
                if let Some(file) = map.remove(old) {
                    map.insert(new.clone(), file);
                }
            }
        }
        io(std::fs::write(&idx, v.to_string()))?;
    }
    Ok(renamed.len())
}

/// The random case's model on the expert cache (`dynamic::qwen4x::load_cached`): the matrices at the level, the routed experts from the case's pack (`PackMode::Best`,
/// which skips none) in `QWEN4X_CB_PLACES` places a layer, the misses computed by the host.
fn qwen4x_cb_load(dir: &Path, o: &DynOpts) -> Result<Dyn> {
    crate::dynamic::qwen4x::load_cached(dir, o, &qwen4x_cb_caching(dir))
}

/// The expert cache's random case's caching: its pack served as best (none skipped), `QWEN4X_CB_PLACES` places a layer, room for 4 sequences.
fn qwen4x_cb_caching(dir: &Path) -> crate::dynamic::qwen4x::Caching {
    use crate::dynamic::qwen4x::{Caching, PackMode};
    Caching { places: Some(QWEN4X_CB_PLACES), ..Caching::new(&dir.join("pack"), PackMode::Best, 4) }
}

fn qwen4x_cb_engine(dir: &Path, level: &str) -> Result<Box<dyn Run>> {
    boxed(qwen4x_cb_load(dir, &dyn_opts(level_layout(level)?)))
}

/// `glyd-engine codescheck [FORM..] [--places N] [--tokens N]`: the Moe kind's codes block (src/moe_cb.rs) in each form of a codes pack (`CODES_FORMS`, every one
/// where none is named: each width alone, an expert's two matrices in two formats, the release's packs' mixes), on the family's random model at the codebook
/// kernels' widths in that form (`qwen4x_codes_random`), its pack served with none of its pairs skipped. Its experts resident (as many places as a layer keeps:
/// every expert on the GPU) against the expert cache (`--places` a layer, 3 by default: fewer than a token's 4, so every call misses) with each miss copied into a
/// place, for a prompt of `--tokens` (40 by default: one pass of the per-pair products; the streaming prefill of a longer chunk sums its pairs with atomics) and 6
/// decode steps: every logit the same bits, as everywhere the GPU computes every pair (`BESIDE_KL`'s ruling). Then the cache with the host computing its misses
/// (src/cpu_cb.rs: the pack's arithmetic in another implementation, its own sums' order) against the resident run by the tree's causal rule (`Judge`): each
/// row by the first discrete choice the two runs made apart in its history (the routers' experts, measured by the model's router on each run's own input; the
/// indexer's pools, by their scores): none, the kind's lossless bound (`BESIDE_KL`) and the best token the same where the lead clears `CLEAR_LEAD`; a near
/// tie (below `kindcheck::NEAR`), reported; wider, failed. The GPU's decode of every format is the host's to that bound. An error where a form fails.
pub fn codes_check(a: &crate::args::Args) -> Result<()> {
    let named: Vec<&str> = a.pos.iter().skip(1).map(String::as_str).collect();
    let forms: Vec<&str> = if named.is_empty() { CODES_FORMS.iter().map(|f| f.0).collect() } else { named };
    let (places, n) = (a.num("places", QWEN4X_CB_PLACES)?, a.num("tokens", 40usize)?);
    let mut failed = Vec::new();
    for name in forms {
        let (ok, line) = codes_form_check(name, places, n)?;
        println!("{line}");
        if !ok {
            failed.push(name);
        }
    }
    if !failed.is_empty() {
        return err(format!("codescheck: {} failed", failed.join(", ")));
    }
    println!("PASS codescheck");
    Ok(())
}

/// The forms of the release's codes packs (glyd/Qwen3.8-Flash-Next-hummingbird: precise-v2, best-v2 and fast, which this engine serves, and the earlier
/// precise, best and fast `pack`, the "cb2" form), as `packcheck --engine` loads each (`codes_form_check`).
pub const RELEASED_CODES: [&str; 6] = ["precise-v2", "best-v2", "fast", "precise", "best", "cb2"];

/// `codes_check`'s comparisons of the form `name` (`places` a layer in the cache, a prompt of `n` tokens): whether it passes, and its line.
pub fn codes_form_check(name: &str, places: usize, n: usize) -> Result<(bool, String)> {
    use crate::dynamic::qwen4x::{load_cached, Caching, PackMode};
    use crate::dynamic::RowRoute;
    use crate::offload::Mode;
    use std::collections::HashMap;
    if n == 0 || n >= crate::moe_cb::STREAM_MIN {
        return err(format!("--tokens {n}: 1 to {} (one pass of the per-pair products)", crate::moe_cb::STREAM_MIN - 1));
    }
    let (prompt, decode) = (token_list(n, 2048, 0), token_list(6, 2048, n));
    let form = codes_form(name)?;
    let dir = std::env::temp_dir().join(format!("glyd-codescheck-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    qwen4x_codes_random(&dir, form)?;
    type Choices = HashMap<(usize, usize), Vec<RowRoute>>;
    // (the pack as best serves it: none of a call's pairs skipped; each run's discrete choices traced by row (`Dyn::trace_row_routes`: the routers' experts,
    // the indexer's pools), and the model's routers for the margins of a choice apart (`Routers`))
    let run = |places: usize, mode: Mode| -> Result<(Vec<Vec<f32>>, Choices, Option<Routers>)> {
        let mut k = Caching { places: Some(places), ..Caching::new(&dir.join("pack"), PackMode::Best, 1) };
        k.offload.mode = mode;
        let mut e = load_cached(&dir, &dyn_opts(None), &k)?;
        e.trace_row_routes(true);
        let rows = (|| -> Result<Vec<Vec<f32>>> {
            e.prefill(&prompt)?;
            let mut rows = vec![e.logits()?];
            for &t in &decode {
                e.step(t)?;
                rows.push(e.logits()?);
            }
            Ok(rows)
        })();
        let seen = e.trace_row_routes(false);
        let routers = if mode == Mode::Cpu { None } else { Some(Routers::load(&dir, &e)?) };
        Ok((rows?, seen, routers))
    };
    // (resident: as many places as a layer keeps, the form's random model's 8)
    let runs = (|| Ok::<_, Error>((run(8, Mode::Copy)?, run(places, Mode::Copy)?, run(places, Mode::Cpu)?)))();
    let _ = std::fs::remove_dir_all(&dir);
    let ((resident, seen_r, routers), (copied, ..), (host, seen_h, _)) = runs?;
    let differ: usize = resident.iter().zip(&copied).map(|(r, c)| row_diff(r, c).0).sum();
    let total: usize = resident.iter().map(Vec::len).sum();
    // (the host's run held to the resident run by the tree's causal rule (`Judge`): each row by the first discrete choice the two runs made apart in its
    // sequence's history up to it, the prompt's last position then each decode step's)
    let hist = |m: &Choices, at: usize| -> Option<Hist> {
        let slot = m.keys().map(|k| k.0).min()?;
        Some((0..=at).map(|p| m.get(&(slot, p)).cloned().unwrap_or_default()).collect())
    };
    let rows: Vec<Judged> = resident
        .iter()
        .zip(&host)
        .enumerate()
        .map(|(i, (r, h))| Judged { alone: r, beside: h, choices: (hist(&seen_r, n - 1 + i), hist(&seen_h, n - 1 + i)) })
        .collect();
    let j = Judge::of(&rows, routers.as_ref())?;
    let ok = differ == 0 && resident.len() == copied.len() && j.ok();
    let fmts: Vec<String> = form.iter().map(|&(u, d)| if u == d { u.name().to_string() } else { format!("{}+{}", u.name(), d.name()) }).collect();
    let notes: String = j.notes.iter().map(|x| format!("\n  {x}")).collect();
    Ok((
        ok,
        format!(
            "{} codes {name} ({}): {} rows; resident against the cache's {places} places, misses copied: {differ} of {total} logits differ; the host's misses against \
             resident: {}; {}{notes}",
            if ok { "PASS" } else { "FAIL" },
            fmts.join(" "),
            resident.len(),
            j.all.words(false),
            j.words()
        ),
    ))
}

/// `qwen4x_s2_random`'s manifest, from the tiny model's.
pub fn qwen4x_s2_manifest() -> Result<String> {
    let mut m: Value = serde_json::from_str(include_str!("../tests/fixtures/qwen4x_tiny_manifest.json")).map_err(|e| Error(e.to_string()))?;
    let (hc, idim, ratio) = (2usize, 64usize, 2usize);
    let mut c: Value = serde_json::from_str(m["config"].as_str().unwrap_or("{}")).map_err(|e| Error(e.to_string()))?;
    let t = &mut c["text_config"];
    let (hid, old_hc, old_idim, heads) = (
        t["hidden_size"].as_u64().unwrap_or(0) as usize,
        4usize,
        t["indexer_head_dim"].as_u64().unwrap_or(0) as usize,
        t["indexer_n_heads"].as_u64().unwrap_or(0) as usize,
    );
    t["hc_count"] = hc.into();
    t["indexer_head_dim"] = idim.into();
    t["indexer_compress_ratio"] = ratio.into();
    t["ple_layer_ids"] = serde_json::json!([]);
    t["number_of_conv_states"] = 1.into();
    let tensors = m["tensors"].as_array().cloned().unwrap_or_default();
    let mut kept = Vec::new();
    for mut x in tensors {
        let name = x[0].as_str().unwrap_or("").to_string();
        if name.contains(".ple.") {
            continue;
        }
        let shape: Vec<usize> = x[2].as_array().map(|a| a.iter().filter_map(|v| v.as_u64().map(|v| v as usize)).collect()).unwrap_or_default();
        let new: Vec<usize> = if name.contains("hyper_connection") {
            shape
                .iter()
                .map(|&d| {
                    if d == old_hc * hid {
                        hc * hid
                    } else if d == old_hc {
                        hc
                    } else {
                        d
                    }
                })
                .collect()
        } else if name.contains("indexer.index_qk_proj") {
            vec![(heads + 1) * idim, shape[1]]
        } else if name.contains("indexer.q_layernorm") || name.contains("indexer.k_layernorm") {
            vec![idim]
        } else {
            shape.clone()
        };
        let _ = old_idim;
        x[2] = serde_json::json!(new);
        kept.push(x);
    }
    m["tensors"] = Value::Array(kept);
    m["config"] = Value::String(c.to_string());
    // (its experts not on the six-bit grid: a lossy level's codes under the lossless level's name then move its logits and its experts' bits)
    m["experts"] = OFF_GRID.into();
    Ok(m.to_string())
}

/// A manifest's `experts` where `manifest_random` draws its experts as its other matrices, not on the six-bit grid.
const OFF_GRID: &str = "off the six-bit grid";

/// The family's random model with its draft head (`kinds::MtpInput::Streams`): the `qwen4x` case's tiny model, each of its tensors drawn as that case draws
/// it (the same seed, the same tensors first), and one MTP layer (an indexed attention and its experts, on the six-bit grid as the model's, in gated
/// residuals), the head's front (its two norms and two products) and its own final mixer, every tensor of the head at the shape its description says,
/// drawn after the model's by the same rules.
pub fn qwen4x_mtp_manifest() -> Result<String> {
    let mut m: Value = serde_json::from_str(include_str!("../tests/fixtures/qwen4x_tiny_manifest.json")).map_err(|e| Error(e.to_string()))?;
    let mut c: Value = serde_json::from_str(m["config"].as_str().unwrap_or("{}")).map_err(|e| Error(e.to_string()))?;
    c["text_config"]["mtp_num_hidden_layers"] = 1.into();
    let d = crate::runtime::describe::describe(&c)?;
    let (blocks, own) = d.mtp.first().ok_or_else(|| Error("the description names no draft head".into()))?;
    let mut names: Vec<String> = blocks.iter().flatten().chain(own).flat_map(|(_, s)| s.names()).collect();
    names.sort();
    let head = names
        .iter()
        .map(|n| d.shapes.get(n).map(|s| json!([n, "BF16", s])).ok_or_else(|| Error(format!("the description has no shape for {n}"))))
        .collect::<Result<Vec<Value>>>()?;
    m["tensors"].as_array_mut().ok_or_else(|| Error("manifest".into()))?.extend(head);
    m["config"] = Value::String(c.to_string());
    Ok(m.to_string())
}

/// `qwen4x_mtp_manifest`'s model at random: the `qwen4x` case's seed, so its model's weights are that case's to the bit.
pub fn qwen4x_mtp_random(out: &Path) -> Result<()> {
    manifest_random(&qwen4x_mtp_manifest()?, out, 0x2026_1120_0000_0005, false, "Qwen3.8-Flash-Next of the tiny shape with its draft head")
}

/// The CPU reference of the kinds on the model in `dir` over `tokens` (one sequence from position 0): each position's state a draft head of the description
/// reads there (`kinds::MtpInput`): the residual streams after the last block (before the head's collapse), or the final normed hidden state (a joined
/// head's, DeepSeek-V3's form).
pub fn ref_streams(dir: &Path, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
    let mut m = ref_model(dir)?;
    let joined = m.desc.mtp.first().is_some_and(|h| h.input == crate::runtime::kinds::MtpInput::Joined);
    let mut last = Vec::new();
    m.forward_seen(tokens, 0, &mut |x| last = x.to_vec())?;
    let w = last.len() / tokens.len().max(1);
    last.chunks(w.max(1)).map(|r| if joined { m.final_h(r, None) } else { Ok(r.to_vec()) }).collect()
}

/// A draft head's row as a check gives it to the CPU reference: its position, the token it reads, the state it continues (the model's last streams [n
/// hidden], or its final normed hidden state: `ref_streams`), and where the row's experts and its indexer's picks are taken in the place of its own (an
/// engine's, `held on its choices`).
#[derive(Clone, Debug)]
pub struct HeadRow {
    pub pos: u32,
    pub token: u32,
    pub streams: Vec<f32>,
    pub take: Option<Vec<usize>>,
    pub picks: Option<Vec<usize>>,
}

/// What `ref_head` gives for each row: its logits by the model's head, its state after the head (the next draft's), its experts as the head's router chose
/// them and as it took them (`ffn::Routed::own`, `ids`), and its indexers' own picks and the ones it took (empty where it has none).
pub struct HeadRef {
    pub logits: Vec<Vec<f32>>,
    pub states: Vec<Vec<f32>>,
    pub routes: Vec<(Vec<usize>, Vec<usize>)>,
    pub picks: Vec<(Vec<usize>, Vec<usize>)>,
}

/// The draft head of the model in `dir` (its description's first, as `edit` leaves it) on the CPU reference of the kinds over `rows` in order (one
/// sequence: a row at a position the head has run before takes its place, `RefMtp::step_streams` or `RefMtp::step` by the head's form).
pub fn ref_head(dir: &Path, rows: &[HeadRow], edit: fn(&mut crate::runtime::describe::Described)) -> Result<HeadRef> {
    use crate::runtime::kinds::{Body, Ffn, Mixer, MtpInput};
    use crate::runtime::reference::{mla::LatentCache, model::checkpoint_tensor, mtp::RefMtp, SeqState};
    let cfg: Value = serde_json::from_str(&io(std::fs::read_to_string(dir.join("config.json")))?).map_err(|e| Error(format!("config.json: {e}")))?;
    let mut desc = crate::runtime::describe::describe(&cfg)?;
    edit(&mut desc);
    let ck = io(glyd_gpu::safetensors::Checkpoint::open(dir))?;
    let src = |n: &str| checkpoint_tensor(&ck, n);
    let m = crate::runtime::reference::model::RefModel::load(&desc, &src)?;
    let mut h = RefMtp::load(&desc, 0, &src)?;
    let moe: Vec<usize> = (0..h.blocks.len()).filter(|&b| matches!(h.blocks[b].block.body, Body::Ffn(Ffn::Moe(_)))).collect();
    for &b in &moe {
        h.blocks[b].routed = Some(Vec::new());
    }
    // (its latent attentions with an indexer of their own: each row's picks recorded, and taken where the row gives them)
    let idx: Vec<usize> = (0..h.blocks.len())
        .filter(|&b| matches!(&h.blocks[b].block.body, Body::Mixer(Mixer::Mla(x)) if x.indexer.as_ref().is_some_and(|i| !i.shared)))
        .collect();
    let d = desc.desc.hidden;
    let mut out = HeadRef { logits: Vec::new(), states: Vec::new(), routes: Vec::new(), picks: Vec::new() };
    for r in rows {
        for &b in &moe {
            h.blocks[b].take = r.take.clone().into_iter().collect();
        }
        for &b in &idx {
            if let SeqState::Mla(c) = h.blocks[b].seqs.entry(0).or_insert_with(|| SeqState::Mla(LatentCache::default())) {
                c.take = r.picks.clone().into_iter().collect();
                c.picked.get_or_insert_with(Vec::new);
            }
        }
        let e = m.embed.get(r.token as usize * d..(r.token as usize + 1) * d).ok_or_else(|| Error(format!("token {} past the vocabulary", r.token)))?;
        let (x, y) = match h.head.input {
            MtpInput::Streams => h.step_streams(0, r.pos, r.token, e, &r.streams)?,
            MtpInput::Joined => {
                let y = h.step(0, r.pos, r.token, e, &r.streams)?;
                (y.clone(), y)
            }
        };
        out.logits
            .push(crate::runtime::reference::matvec(&m.head, d, &y).into_iter().map(|z| crate::runtime::reference::model::logit(&desc.desc.head, z)).collect());
        out.states.push(x);
        let c = moe.first().and_then(|&b| h.blocks[b].routed.as_mut().and_then(Vec::pop));
        out.routes.push(c.map_or((Vec::new(), Vec::new()), |c| (c.own, c.ids)));
        let pk = idx.first().and_then(|&b| match h.blocks[b].seqs.get_mut(&0) {
            Some(SeqState::Mla(c)) => c.picked.as_mut().and_then(Vec::pop),
            _ => None,
        });
        out.picks.push(pk.map_or((Vec::new(), Vec::new()), |p| (p.own, p.picks)));
    }
    Ok(out)
}

/// The DSpark drafter of the model in `dir` (its description's first draft head) on the CPU reference of the kinds (`RefDSpark::load`), and the model's
/// embedding and LM head (f32 [vocab, hidden]) its block reads.
pub fn ref_dspark(dir: &Path) -> Result<(Vec<f32>, Vec<f32>, crate::runtime::reference::mtp::RefDSpark)> {
    use crate::runtime::reference::{model::checkpoint_tensor, mtp::RefDSpark};
    let cfg: Value = serde_json::from_str(&io(std::fs::read_to_string(dir.join("config.json")))?).map_err(|e| Error(format!("config.json: {e}")))?;
    let desc = crate::runtime::describe::describe(&cfg)?;
    let ck = io(glyd_gpu::safetensors::Checkpoint::open(dir))?;
    let src = |n: &str| checkpoint_tensor(&ck, n);
    let t = |n: &str| src(n).map(|x| x.1).ok_or_else(|| Error(format!("the checkpoint has no tensor {n}")));
    let (embed, head) = (t(&desc.embed)?, t(desc.head.as_deref().unwrap_or(&desc.embed))?);
    Ok((embed, head, RefDSpark::load(&desc, &src)?))
}

/// The CPU reference of the kinds on the model in `dir` over `tokens` (one sequence from position 0): each position's taps for its DSpark drafter
/// (`kinds::DSpark::targets`), the mean of the streams each target layer's attention reads (after an n-gram block before it), the targets' side by side.
pub fn ref_taps(dir: &Path, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
    let mut m = ref_model(dir)?;
    let ds = m.desc.mtp.first().and_then(|h| h.dspark.clone()).ok_or_else(|| Error("the description has no DSpark drafter".into()))?;
    // (the block of each target layer's attention: a layer's mixer the l-th of the blocks')
    let mixers: Vec<usize> =
        m.desc.blocks.iter().enumerate().filter(|(_, b)| matches!(b.body, crate::runtime::kinds::Body::Mixer(_))).map(|(i, _)| i).collect();
    let at: Vec<usize> =
        ds.targets.iter().map(|&t| mixers.get(t).copied().ok_or_else(|| Error(format!("target layer {t} of {}", mixers.len())))).collect::<Result<_>>()?;
    let (n, d) = (m.desc.embed.streams.max(1), m.desc.hidden);
    let mut out = vec![vec![0f32; at.len() * d]; tokens.len()];
    let mut b = 0usize;
    m.forward_seen(tokens, 0, &mut |x| {
        for (k, _) in at.iter().enumerate().filter(|(_, &a)| a == b) {
            for (t, xt) in x.chunks(n * d).enumerate() {
                for c in 0..d {
                    out[t][k * d + c] = (0..n).map(|s| xt[s * d + c]).sum::<f32>() / n as f32;
                }
            }
        }
        b += 1;
    })?;
    Ok(out)
}

/// The `deepseek-v4` case's random model with its DSpark drafter (`dsv4_ref::random_dspark`: the case's model to the bit, two draft layers, a block of 8).
pub fn deepseek_v4_dspark_random(out: &Path) -> Result<()> {
    crate::dsv4_ref::random_dspark(out)
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
    qwen35moe_draw(include_str!("../tests/fixtures/qwen35moe_tiny_config.json"), out)
}

/// The Qwen3.5 layout with experts at its release's width (tests/fixtures/qwen35_configs.json's Qwen3.5-35B-A3B: 2,048 wide, experts of 512 neurons, 8 a token, the
/// shared expert of 512, its attention's and linear attention's heads, its kinds of layer in their order), `layers` layers of `experts` experts each, with the small
/// case's vocabulary and rotation: drawn as `qwen35moe_random` draws its model. A model whose experts exceed a GPU's memory, for the expert cache's checks
/// (`logitsmodel qwen35moe DIR --real --layers L --experts E`; `storecheck`).
pub fn qwen35moe_random_real(out: &Path, layers: usize, experts: usize) -> Result<()> {
    let js = |t: &str| serde_json::from_str::<Value>(t).map_err(|e| Error(e.to_string()));
    let mut c = js(include_str!("../tests/fixtures/qwen35moe_tiny_config.json"))?;
    let all = js(include_str!("../tests/fixtures/qwen35_configs.json"))?;
    let rel =
        all.get("Qwen/Qwen3.5-35B-A3B").map(|v| v.get("text_config").unwrap_or(v)).ok_or_else(|| Error("no Qwen3.5-35B-A3B config in the fixtures".into()))?;
    for k in [
        "hidden_size",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "num_experts_per_tok",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
    ] {
        c[k] = rel.get(k).cloned().ok_or_else(|| Error(format!("Qwen3.5-35B-A3B's config has no {k}")))?;
    }
    let kinds = rel["layer_types"].as_array().filter(|k| !k.is_empty()).ok_or_else(|| Error("Qwen3.5-35B-A3B's config has no layer_types".into()))?;
    (c["num_hidden_layers"], c["num_experts"]) = (json!(layers.max(1)), json!(experts.max(1)));
    c["layer_types"] = Value::Array((0..layers.max(1)).map(|l| kinds[l % kinds.len()].clone()).collect());
    qwen35moe_draw(&c.to_string(), out)
}

/// `qwen35moe_random`'s draw of the model of config `text`.
fn qwen35moe_draw(text: &str, out: &Path) -> Result<()> {
    let d = crate::runtime::describe::describe(&serde_json::from_str(text).map_err(|e| Error(e.to_string()))?)?;
    let tensors = d
        .names()
        .into_iter()
        .map(|n| d.shapes.get(&n).map(|s| json!([n, "BF16", s])).ok_or_else(|| Error(format!("the description has no shape for {n}"))))
        .collect::<Result<Vec<Value>>>()?;
    manifest_random(&json!({"config": text, "tensors": tensors}).to_string(), out, 0x2026_1124_0000_0333, false, "Qwen3.5-family mixture of experts")
}

/// GLM-5.3's layout as a small random model (tests/fixtures/glm_moe_dsa_toy_config.json, `glm_moe_dsa`: four layers 256 wide of the latent attention, two heads
/// of 64 + 64 rotated channels, its indexer of 4 heads of 128 keeping 8 tokens and turning the first 64 of each key and query, its own in the first two layers
/// and shared in the last two; a dense MLP, then 16 experts of which a token takes 4 and the shared one): every tensor of its description at its shape, drawn as
/// GLM-5.3-Flash's random case is (`glm::random`): each matrix U(+-sqrt(3 / its columns)), the embedding U(-1, 1), the head at half the matrices' scale, the
/// norms' weights about 1, the indexer's key norm's bias and the choice bias small (the choice bias fp32, as released), bf16 else.
pub fn glm_dsa_random(out: &Path) -> Result<()> {
    glm_dsa_draw(out, GLM_DSA_SEED, false, false)
}

/// The seed of `glm_dsa_random`'s draw.
pub const GLM_DSA_SEED: u64 = 0x2026_1125_0000_0053;

/// GLM-5.3's layout as the release holds it, in FP8 (the glm-dsa-f8 case): `glm_dsa_random`'s config with the release's latent attention a head (heads of
/// 192 + 64 rotated channels, values of 256, the key-value latent of 512) in a model 512 wide with a query latent of 256, so that the scales of every
/// FP8 product span several blocks of 128 both ways and kv_b_proj's rows of a head cross its blocks; its quantization_config the release's (fp8, e4m3,
/// blocks of 128 x 128, the head left out). Every matrix the description names FP8 (`Described::scales`: the attention's q_a, q_b, kv_a, kv_b and o, the
/// indexer's wq_b and wk, the dense MLP's, the experts' and the shared expert's) is E4M3 with its fp32 block scales (`X.weight_scale_inv`), each block's
/// scale its largest magnitude over 448 and each value the E4M3 nearest it over that scale (`glm::random`'s FP8 experts); the rest as `glm_dsa_random`'s.
pub fn glm_dsa_f8_random(out: &Path) -> Result<()> {
    glm_dsa_draw(out, GLM_DSA_SEED, true, false)
}

/// GLM-5.3's layout with its draft head, as the release has it (the draftcheck cases `glm-dsa` and `glm-dsa-f8`): `glm_dsa_random`'s model (`fp8`:
/// `glm_dsa_f8_random`'s) to the bit, and one MTP layer after its layers (`num_nextn_predict_layers` 1: a latent attention with an indexer of its own and
/// the experts, on a plain residual, the head's two norms, eh_proj and its final norm), every tensor of the head drawn after the model's by the same rules.
pub fn glm_dsa_mtp_random(out: &Path, fp8: bool) -> Result<()> {
    glm_dsa_draw(out, GLM_DSA_SEED, fp8, true)
}

/// `glm_dsa_random` (`fp8`: `glm_dsa_f8_random`; `mtp`: with its draft head, `glm_dsa_mtp_random`) drawn from `seed` (not 0).
pub fn glm_dsa_draw(out: &Path, seed: u64, fp8: bool, mtp: bool) -> Result<()> {
    let mut cfg: Value = serde_json::from_str(include_str!("../tests/fixtures/glm_moe_dsa_toy_config.json")).map_err(|e| Error(e.to_string()))?;
    if mtp {
        cfg["num_nextn_predict_layers"] = json!(1);
    }
    if fp8 {
        let wide = [
            ("hidden_size", 512),
            ("q_lora_rank", 256),
            ("kv_lora_rank", 512),
            ("qk_nope_head_dim", 192),
            ("head_dim", 192),
            ("qk_head_dim", 256),
            ("v_head_dim", 256),
            ("num_attention_heads", 4),
            ("num_key_value_heads", 4),
        ];
        for (k, v) in wide {
            cfg[k] = json!(v);
        }
        cfg["quantization_config"] = json!({"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8", "weight_block_size": [128, 128], "modules_to_not_convert": ["lm_head", "model.embed_tokens"]});
    }
    let text = serde_json::to_string_pretty(&cfg).map_err(|e| Error(e.to_string()))?;
    let d = crate::runtime::describe::describe(&cfg)?;
    if seed == 0 {
        return err("a seed of 0 draws nothing but zeros");
    }
    if fp8 && d.scales.is_empty() {
        return err("GLM-5.3's description in FP8 names no FP8 matrix");
    }
    let mut r = seed;
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
    // (E4M3: a value rounded to it, then its byte)
    let code: std::collections::HashMap<u32, u8> =
        (0..=255u8).filter(|b| !glyd_gpu::fp8::E4M3[*b as usize].is_nan()).map(|b| (glyd_gpu::fp8::E4M3[b as usize].to_bits(), b)).collect();
    let mut tensors: Vec<(String, &str, Vec<u64>, Vec<u8>)> = Vec::new();
    // (an FP8 matrix's scales are drawn with it; a draft head's tensors after the model's, so that the model is the one without a head to the bit)
    let scales: std::collections::BTreeSet<&String> = d.scales.values().collect();
    let head: std::collections::BTreeSet<String> = d.mtp.iter().flat_map(|(b, o)| b.iter().flatten().chain(o)).flat_map(|(_, s)| s.names()).collect();
    let (model, own): (Vec<String>, Vec<String>) = d.names().into_iter().filter(|n| !scales.contains(n)).partition(|n| !head.contains(n));
    for name in model.into_iter().chain(own) {
        let shape = d.shapes.get(&name).cloned().ok_or_else(|| Error(format!("GLM-5.3's description has no shape for {name}")))?;
        let n: usize = shape.iter().product();
        let a = (3.0 / *shape.last().unwrap_or(&1) as f32).sqrt();
        let (lo, hi) = if name.ends_with("e_score_correction_bias") {
            (0.0, 0.1)
        } else if name.ends_with("k_norm.bias") {
            (-0.1, 0.1)
        } else if name.ends_with("norm.weight") || name.ends_with("layernorm.weight") {
            (0.9, 1.1)
        } else if name == d.embed {
            (-1.0, 1.0)
        } else if d.head.as_ref() == Some(&name) {
            (-0.5 * a, 0.5 * a)
        } else {
            (-a, a)
        };
        if let (Some(sn), &[rows, cols]) = (d.scales.get(&name), &shape[..]) {
            let w: Vec<f32> = (0..n).map(|_| uni(lo, hi)).collect();
            let (gr, gc) = (rows.div_ceil(128), cols.div_ceil(128));
            let mut sc = vec![0f32; gr * gc];
            for (i, x) in w.iter().enumerate() {
                let b = &mut sc[(i / cols) / 128 * gc + (i % cols) / 128];
                *b = b.max(x.abs() / 448.0);
            }
            let q: Vec<u8> =
                w.iter().enumerate().map(|(i, x)| code[&crate::cpu_fp4::e4m3_round(x / sc[(i / cols) / 128 * gc + (i % cols) / 128]).to_bits()]).collect();
            tensors.push((name.clone(), "F8_E4M3", vec![rows as u64, cols as u64], q));
            tensors.push((sn.clone(), "F32", vec![gr as u64, gc as u64], sc.iter().flat_map(|x| x.to_le_bytes()).collect()));
            continue;
        }
        let (dtype, bytes): (&str, Vec<u8>) = if name.ends_with("e_score_correction_bias") {
            ("F32", (0..n).flat_map(|_| uni(lo, hi).to_le_bytes()).collect())
        } else {
            ("BF16", (0..n).flat_map(|_| bf(uni(lo, hi)).to_le_bytes()).collect())
        };
        tensors.push((name, dtype, shape.iter().map(|&x| x as u64).collect(), bytes));
    }
    io(std::fs::create_dir_all(out))?;
    io(std::fs::write(out.join("config.json"), text))?;
    let refs: Vec<(&str, &str, Vec<u64>, &[u8])> = tensors.iter().map(|(n, d, s, b)| (n.as_str(), *d, s.clone(), &b[..])).collect();
    crate::pack::write_safetensors(&out.join("model.safetensors"), &[("format", "pt".to_string())], &refs)?;
    println!(
        "{}: a random model of GLM-5.3's layout{}, {} tensors, {:.0} MB",
        out.display(),
        if fp8 { " in FP8 as released" } else { "" },
        tensors.len(),
        tensors.iter().map(|t| t.3.len()).sum::<usize>() as f64 / 1e6
    );
    Ok(())
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

/// A random model of a manifest's tensors (names and shapes from a real checkpoint's header, cut down), their values from a fixed seed by the rules of tools/qwen4x_ref.py for each kind of tensor.
/// Every drawn tensor is stored in bf16, the embedding too (the manifest's dtype is read for its int64 tensors alone), so no tensor is held wider than a release holds its matrices.
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
    let on_grid = m["experts"] != OFF_GRID;
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
        } else if on_grid && (name.contains("experts.gate_up_proj") || name.contains("experts.down_proj")) {
            (0..shape[0]).flat_map(|_| grid(shape[1] as usize, shape[2] as usize, &mut u01, &mut gauss)).collect()
        } else if on_grid && name.contains("shared_expert.") {
            grid(shape[0] as usize, shape[1] as usize, &mut u01, &mut gauss)
        } else if name.contains("ngram_embedding") || name.contains("embed_tokens") {
            // (an embedding that is the head too, as Qwen3.5's tied one is, is as wide as a head's rows: else its logits are tens apart and the softmax a point)
            let std = if tied && name.contains("embed_tokens") { 1.0 / (cols as f64).sqrt() } else { 1.0 };
            (0..n).map(|_| (gauss(&mut u01) * std) as f32).collect()
        } else if shape.len() == 2 || name.contains("experts.") {
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
/// compressors, indexers and candidate blocks, Engram of orders 2 to 4 before layers 1 and 4, lagged mHC over four streams, one DSpark layer; the release's
/// norm eps of 1e-20 and its formats): every tensor of its description at its shape, at random from a fixed seed, stored as the release stores it. The values: the
/// matrices about U(-a, a), a = sqrt(3 / cols); the norms' weights about 1; the mixing projections small. The formats (DeepSeek-V4.1-Flash's headers): the
/// matrices its config names FP8 E4M3 with an E8M0 scale a block of 32 x 32 (each block's scale its largest magnitude over 448 rounded up to a power of two;
/// the Engram tables a scale a row's run of 32), the routed experts FP4 (two E2M1 codes a byte, the low first, an E8M0 scale a run of 32 near sqrt(3 / cols) / 6,
/// one exponent either way: `dsv4_ref::random`'s draw), the mixers, sinks and choice biases fp32, the rest bf16; and the compressed vocabulary Engram hashes
/// (`engram_token_map.json`: a token's id mod its size).
pub fn deepseek_v41_random(out: &Path) -> Result<()> {
    deepseek_v41_draw(out, DEEPSEEK_V41_SEED)
}

/// The seed of `deepseek_v41_random`'s draw.
pub const DEEPSEEK_V41_SEED: u64 = 0x2026_1124_0000_0041;

/// `deepseek_v41_random` drawn from `seed` (not 0).
pub fn deepseek_v41_draw(out: &Path, seed: u64) -> Result<()> {
    let text = include_str!("../tests/fixtures/deepseek_v41_tiny_config.json");
    let cfg: Value = serde_json::from_str(text).map_err(|e| Error(e.to_string()))?;
    let d = crate::runtime::describe::describe(&cfg)?;
    let mut r = seed;
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
    let scale_names: std::collections::BTreeSet<&String> = d.scales.values().collect();
    let mut tensors: Vec<(String, &str, Vec<u64>, Vec<u8>)> = Vec::new();
    for name in d.names() {
        // (a quantized matrix's scales are written with it)
        if scale_names.contains(&name) {
            continue;
        }
        let shape = d.shapes.get(&name).cloned().ok_or_else(|| Error(format!("DeepSeek-V4.1's description has no shape for {name}")))?;
        let n: usize = shape.iter().product();
        let cols = *shape.last().unwrap_or(&1);
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
            let a = (3.0 / cols as f32).sqrt();
            (-a, a)
        };
        let dims: Vec<u64> = shape.iter().map(|&x| x as u64).collect();
        match d.scales.get(&name) {
            // (the routed experts as released: FP4)
            Some(sn) if name.contains(".experts.") => {
                let rows = shape[0];
                let q: Vec<u8> = (0..rows * cols / 2).map(|_| uni(0.0, 256.0) as u8).collect();
                let e0 = (((3.0 / cols as f32).sqrt() / 3.0).log2().round() as i32 + 126) as f32;
                let s: Vec<u8> = (0..rows * cols / 32).map(|_| (e0 + uni(-1.0, 1.0).round()) as u8).collect();
                tensors.push((name.clone(), "I8", vec![rows as u64, cols as u64 / 2], q));
                tensors.push((sn.clone(), "F8_E8M0", vec![rows as u64, cols as u64 / 32], s));
            }
            // (an FP8 matrix as released: E4M3 with E8M0 block scales)
            Some(sn) => {
                let rows = shape[0];
                let (br, bc) = if name.ends_with("engram.embed.weight") { (1, 32) } else { (32, 32) };
                let w: Vec<f32> = (0..n).map(|_| uni(lo, hi)).collect();
                let (gr, gc) = (rows.div_ceil(br), cols.div_ceil(bc));
                let (mut q, mut sc) = (vec![0u8; n], vec![0u8; gr * gc]);
                for bi in 0..gr {
                    for bj in 0..gc {
                        let at = || (bi * br..((bi + 1) * br).min(rows)).flat_map(move |i| (bj * bc..((bj + 1) * bc).min(cols)).map(move |j| i * cols + j));
                        let amax = at().fold(0f32, |m, k| m.max(w[k].abs()));
                        let s = crate::cpu_fp4::pow2_scale(amax.max(1e-4), 1.0 / 448.0);
                        sc[bi * gc + bj] = (s.to_bits() >> 23) as u8;
                        for k in at() {
                            q[k] = crate::dsv4_ref::e4m3_byte(crate::cpu_fp4::e4m3_round((w[k] / s).clamp(-448.0, 448.0)));
                        }
                    }
                }
                tensors.push((name.clone(), "F8_E4M3", dims, q));
                tensors.push((sn.clone(), "F8_E8M0", vec![gr as u64, gc as u64], sc));
            }
            None => {
                let vals: Vec<f32> = (0..n).map(|_| uni(lo, hi)).collect();
                let f32s = name.contains("hc_") || name.ends_with("attn_sink") || name.ends_with("gate.bias");
                let bytes = if f32s { vals.iter().flat_map(|x| x.to_le_bytes()).collect() } else { vals.iter().flat_map(|&x| bf(x).to_le_bytes()).collect() };
                tensors.push((name, if f32s { "F32" } else { "BF16" }, dims, bytes));
            }
        }
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
/// each token), which `deepseek_v41_random` writes for the random model (no loader makes one from a release's tokenizer); None where the directory has none.
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
            on_choices: None,
            also: None,
        },
        Case {
            id: "qwen35",
            family: "Qwen3.5 family (Gated DeltaNet and gated attention: Qwen3.8-27B's layout)",
            weights: Weights::Random(qwen35_random),
            engine: qwen35_engine,
            cpu: Some(described_cpu),
            vocab: 4096,
            on_choices: None,
            also: None,
        },
        Case {
            id: "qwen35-wide",
            family: "Qwen3.5 family, the wide shape (three value heads to a key head, a head of its own, the text model of a checkpoint with a vision tower)",
            weights: Weights::Random(qwen35_wide_random),
            engine: qwen35_engine,
            cpu: Some(described_cpu),
            vocab: 4096,
            on_choices: None,
            also: None,
        },
        Case {
            id: "qwen35moe",
            family: "Qwen3.5 family with experts (Gated DeltaNet, gated attention, softmax-routed experts with a gated shared expert)",
            weights: Weights::Random(qwen35moe_random),
            engine: qwen35_engine,
            cpu: Some(described_cpu),
            vocab: 4096,
            on_choices: Some(ref_drive),
            also: None,
        },
        Case {
            id: "muse",
            family: "Muse Spark (sliding and full attention, mixture of experts)",
            weights: Weights::Random(crate::muse_ref::random),
            engine: muse_engine,
            cpu: Some(described_cpu),
            vocab: 4096,
            on_choices: Some(ref_drive),
            also: None,
        },
        Case {
            id: "kimi",
            family: "Kimi-K3 (delta attention, latent attention, mixture of experts)",
            weights: Weights::Random(|d| crate::kimi_ref::random(d, 1, true, 8)),
            engine: kimi_engine,
            cpu: Some(described_cpu),
            vocab: 4096,
            on_choices: Some(ref_drive),
            also: None,
        },
        Case {
            id: "kimi-dsa",
            family: "Kimi-K3 family with the newer config's sparse attention (latent attention over the tokens its indexer picks, a key a token, RMS-normed)",
            weights: Weights::Random(crate::kimi_ref::random_dsa),
            engine: kimi_engine,
            cpu: Some(described_cpu),
            vocab: 4096,
            on_choices: Some(ref_drive),
            also: None,
        },
        Case {
            id: "glm-dsa",
            family: "GLM-5.3's layout (glm_moe_dsa: the latent attention with its rotary on every layer, indexers of their own and shared, the plain residual, a dense MLP and experts)",
            weights: Weights::Random(glm_dsa_random),
            engine: glm_dsa_engine,
            cpu: Some(described_cpu),
            vocab: 2048,
            on_choices: Some(ref_drive),
            also: None,
        },
        Case {
            id: "glm-dsa-f8",
            family: "GLM-5.3's layout as released in FP8 (the attention's, the indexer's, the dense MLP's and the experts' matrices E4M3 with fp32 block scales, held as released)",
            weights: Weights::Random(glm_dsa_f8_random),
            engine: glm_dsa_engine,
            cpu: Some(described_cpu),
            vocab: 2048,
            on_choices: Some(ref_drive),
            also: None,
        },
        Case {
            id: "deepseek-v4",
            family: "DeepSeek-V4 (four residual streams, compressed attention, mixture of experts)",
            weights: Weights::Random(|d| crate::dsv4_ref::random(d, None)),
            // (held in three parts, `dsv4_also` has the second and third: the engine and the reference both WITHOUT the release's simulated quantizations, on the
            // engine's choices at the kind's bound; the quantizing operations bit for bit on the reference's own input; and the whole model with them, printed)
            engine: dsv4_engine,
            cpu: Some(dsv4_cpu),
            vocab: 4096,
            on_choices: Some(dsv4_drive),
            also: Some(dsv4_also),
        },
        Case {
            id: "gemma4",
            family: "Gemma 4 (sliding and global attention, K = V, soft-capped head)",
            weights: Weights::Random(gemma4_random),
            engine: gemma4_engine,
            cpu: Some(described_cpu),
            vocab: 4096,
            on_choices: None,
            also: None,
        },
        Case {
            id: "gemma4-kv",
            family: "Gemma 4 E2B/E4B layers (KV sharing across layers, the double-wide MLP)",
            weights: Weights::Random(gemma4_kv_random),
            engine: gemma4_engine,
            cpu: Some(described_cpu),
            vocab: 4096,
            on_choices: None,
            also: None,
        },
        Case {
            id: "qwen4x",
            family: "Qwen3.8-Flash-Next (Gated DeltaNet, indexed attention, hyper-connections, per-layer embeddings, mixture of experts)",
            weights: Weights::Random(qwen4x_random),
            engine: qwen4x_engine,
            cpu: Some(described_cpu),
            vocab: 2048,
            on_choices: Some(ref_drive),
            also: None,
        },
        Case {
            id: "qwen4x-cb",
            family:
                "Qwen3.8-Flash-Next family through the expert cache (hidden 2,560, two-bit codebook experts of 640 neurons from a pack, 3 places of 8 a layer)",
            weights: Weights::Random(qwen4x_cb_random),
            engine: qwen4x_cb_engine,
            cpu: Some(described_cpu),
            vocab: 2048,
            on_choices: Some(ref_drive),
            also: None,
        },
        Case {
            id: "qwen4x-s2",
            family:
                "Qwen3.8-Flash-Next family with other counts (2 streams, index heads of 64 over pools of 2, no n-gram embedding, experts off the six-bit grid)",
            weights: Weights::Random(qwen4x_s2_random),
            engine: qwen4x_engine,
            cpu: Some(described_cpu),
            vocab: 2048,
            on_choices: Some(ref_drive),
            also: None,
        },
        Case {
            id: "nemotron-h",
            family: "Nemotron-H (Mamba-2, attention, mixture of experts)",
            weights: Weights::Random(|d| crate::kcheck_nh::random(d, 0, 16)),
            engine: nemotron_engine,
            cpu: Some(described_cpu),
            vocab: 4096,
            on_choices: None,
            also: None,
        },
        Case {
            id: "deepseek-v41",
            family: "DeepSeek-V4.1 (CSA2's shared compressed attention, lagged mHC, Engram, mixture of experts)",
            weights: Weights::Random(deepseek_v41_random),
            // (the compressed attention's executor, held to the kinds' reference on its own routers' and indexers' choices at the kind's bound. Without
            // simulated quantizations: the kernels simulate DeepSeek-V4's release's (FP8 in blocks of 128), and V4.1's release quantizes otherwise
            // (`lower_comp::lower`), so its engine runs its weights' arithmetic, which is what is held)
            engine: dsv4_engine,
            cpu: Some(described_cpu),
            vocab: 4096,
            on_choices: Some(ref_drive),
            also: None,
        },
        Case {
            id: "glm",
            family: "GLM-5.3 (delta attention, sparse attention, mixture of experts)",
            weights: Weights::Random(|d| crate::glm::random(d, false, None)),
            engine: glm_engine,
            cpu: Some(described_cpu),
            vocab: 2048,
            // (its indexers keep 2 pools of up to 6 and its routers 4 experts of 16, each weighing about 0.6: near ties that rounding alone takes the other
            // way, `tests::glm_reference_with_its_residual_rounded`)
            on_choices: Some(ref_drive),
            also: None,
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

/// `logitscheck [CASE...] [--fixtures DIR] [--scratch DIR] [--cpu-ref] [--level L]`: each case's engine (or CPU reference) against its fixture at each level the fixture bounds. Exit 1 where one is not within,
/// and where a case had no fixture this GPU can be held to (every one of them another GPU generation's bits): a case with nothing compared is not one within its bounds.
/// A line says what held it (`Held`): PASS a reference's bound kept, PIN the engine's own recorded bits unchanged. The last lines count them apart ("N held to a
/// reference, M pinned, K not held") and name the cases pinned only (no reference of another implementation: GLM-5.3's case today) and not held.
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
    let mut tally = Tally::default();
    let t_all = std::time::Instant::now();
    let gpu = gpu_label().ok();
    for c in &all {
        let dir = model_dir(c, &scratch.dir)?;
        let (mut others, mut waiting) = (Vec::new(), false);
        let (refs_before, pins_before) = (tally.refs, tally.pins);
        for path in fixture_files(&fixtures, c.id)? {
            let fx = Fixture::read(&path)?;
            if cpu && fx.gpu.is_some() {
                continue; // (an engine's own bits: not what a CPU reference makes)
            }
            if let (Some(want), Some(have)) = (&fx.gpu, &gpu) {
                if want != have && !cpu {
                    println!("SKIP {:12} {}: its bits are {want}'s, this GPU is {have}", c.id, path.file_name().unwrap_or_default().to_string_lossy());
                    others.push(want.clone());
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
                            waiting = true;
                            continue;
                        }
                        r => r?,
                    }
                };
                let load = t0.elapsed().as_secs_f64();
                let got = drive(run.as_mut(), &fx.prompt, &fx.decode, &fx.long)?;
                let v = compare(&fx, &lb.bound, &got);
                ran += 1;
                // (under --cpu-ref the reference runs against its own fixture: a reference's line)
                let held = fx.held();
                tally.add(held);
                // (a case that asks: the verdict is the engine's on its own routers' choices; the line below still has its rows against the reference's own)
                let chosen = if let (Some(forced), false, None, false) = (c.on_choices, cpu, &fx.gpu, lb.bound == Bound::Bits) {
                    run.choices(true)?;
                    let traced = drive(run.as_mut(), &fx.prompt, &fx.decode, &fx.long);
                    let h = match run.choices(false)? {
                        Some(ch) => held_on_choices("the engine", &dir, &fx, &lb.bound, &traced?, &ch, &forced(&dir, &fx, None, None)?, forced)?,
                        None => return err(format!("{}: its engine gives no view of its routers' choices", c.id)),
                    };
                    Some(h.routing && h.logits)
                } else {
                    None
                };
                // (the lossless level: the experts the engine holds are the checkpoint's, bit for bit)
                let experts = if lb.level == "penguin" && !cpu { experts_line(&dir, run.as_mut())? } else { None };
                // (what the case holds beside its rows, on the reference's fixture on a GPU: its lines come before the case's)
                let also = match (c.also, cpu, &fx.gpu) {
                    (Some(f), false, None) => f(&dir, &fx, lb)?,
                    _ => true,
                };
                let ok = chosen.unwrap_or(v.ok) && experts.as_ref().is_none_or(|e| e.0) && also;
                let bound = match &lb.bound {
                    Bound::Bits => "bits".to_string(),
                    Bound::Kl { mean, max, top1 } => format!("KL mean <= {mean:.1e}, max <= {max:.1e}, top-1 >= {:.0}%", top1 * 100.0),
                };
                let bound = format!("{bound}; {}", held.tag());
                let name = path.file_name().unwrap_or_default().to_string_lossy().replace(".json.gz", "");
                println!(
                    "{} {:20} {:8} {:3} rows: bits-equal {}, KL mean {:.2e} max {:.2e}, best token {}/{} with a lead, largest logit difference {:.4}  [{bound}{}] ({load:.1} s to load, {:.1} s)",
                    if ok { held.passed() } else { "FAIL" },
                    name,
                    lb.level,
                    v.rows,
                    v.identical,
                    v.mean_kl,
                    v.max_kl,
                    v.top1.0,
                    v.top1.1,
                    v.max_dlogit,
                    match chosen {
                        Some(h) => format!(
                            "; these figures are against the reference's own choices ({}), the verdict is on the engine's choices ({}): the lines above",
                            if v.ok { "within the bound" } else { "PAST the bound" },
                            if h { "on its own router inputs its choices apart the reference router's or near ties, its logits within the bound" } else { "not held" }
                        ),
                        None => String::new(),
                    },
                    t0.elapsed().as_secs_f64()
                );
                for n in v.notes.iter().take(6) {
                    println!("       {n}");
                }
                if let Some((good, line)) = &experts {
                    println!("       {}{line}", if *good { "" } else { "FAIL " });
                }
                if !ok {
                    failed.push(format!("{name} at {}", lb.level));
                }
            }
        }
        // (a case no reference held: pinned only; its engine not built; or nothing compared, every fixture it has being another GPU generation's bits)
        others.dedup();
        if let Some(line) = tally.case(c.id, tally.refs - refs_before, tally.pins - pins_before, waiting, &others) {
            let how = if line.starts_with("FAIL") {
                format!(
                    "; this GPU is {} (logitsrecord {} --from engine --out {} records this generation's pin)",
                    gpu.as_deref().unwrap_or("unknown"),
                    c.id,
                    fixtures.display()
                )
            } else {
                String::new()
            };
            println!("{line}{how}");
        }
    }
    failed.extend(tally.failed());
    println!("{ran} comparisons in {:.1} s against {}: {}", t_all.elapsed().as_secs_f64(), fixtures.display(), tally.summary());
    if failed.is_empty() {
        println!("every comparison made is kept: a reference's bound, or a pin's bits unchanged (a pin does not show the logits are right)");
        Ok(())
    } else {
        println!("OFF: {}", failed.join(", "));
        drop(scratch); // (the process exits next: the private directory is removed first)
        std::process::exit(1);
    }
}

// ---- the description path against the CPU reference on its own choices, layer by layer (R3b, #137; R3d, #140)

/// The cases `paritycheck` takes: (case, the description path). (R3b retired Qwen3.5's, Gemma 4's and Nemotron-H's hand-written engines on their parity
/// records, #410 and the R3b pull requests; R3d Condor's, the last, once the executor served its model.)
type ParityCase = (&'static str, fn(&Path, &str) -> Result<Box<dyn Run>>);

fn parity_cases() -> Vec<ParityCase> {
    vec![("qwen4x", qwen4x_desc), ("qwen4x-s2", qwen4x_desc)]
}

/// The experts of the random Qwen3.8-Flash-Next at a level: a lossy level's codes, lossless at penguin.
fn qwen4x_experts(level: &str) -> Result<Layout> {
    Ok(crate::dynamic::qwen4x::experts_at(level_layout(level)?))
}

/// The description executor (src/dynamic/qwen4x.rs) on the random model: the other matrices lossless, the experts at the level.
fn qwen4x_desc(dir: &Path, level: &str) -> Result<Box<dyn Run>> {
    boxed(crate::dynamic::qwen4x::load_level(dir, &dyn_opts(Some(Layout::Tiered)), qwen4x_experts(level)?))
}

/// `paritycheck [CASE...] [--level L] [--scratch DIR]`: each case of Qwen3.8-Flash-Next (qwen4x, qwen4x-s2) on the description executor: the same random
/// model, prompt and decode tokens as the case's fixture, at each level the fixtures bound, held to the CPU reference of the layer kinds (the case's
/// fixture). Required of the executor:
///   1. wherever its router's choice is not the reference's own, the reference's router on the executor's own input row for that token and layer
///      takes the executor's choice, or one less than `NEAR` from it by its ranks on that input (`held_on_choices`: why the same input);
///   2. its logits are within the kind's bound for the lossless level of the reference taking its choices, every row. The bound is the case's
///      fixture's, which must be the one the Qwen3.5 layout's fixture holds (tests/logits/qwen35.json.gz): a family has no bound of its own;
///   3. its best token is the reference's own on every row where the reference's lead clears `CLEAR_LEAD`; a row below it is printed, not required.
///
/// Printed beside them and not required: each row's KL against the reference's own choices (never left out); its residual after each layer against the
/// reference's; and what the executor's first choice apart leaves in the rows after it, by the kind of state that carries it (`carry_lines`). Exit 1
/// where a requirement fails. With `--choice ROW:LAYER` no engine runs: the reference alone takes the nearest other choice in that row's pass and layer,
/// and the same lines say what it leaves (any machine, no GPU). (Condor's own engine ran beside it here until it was deleted: its record is
/// tests/fixtures/qwen4x_condor_rows.json.gz, which `runtime::describe_tests` holds the reference to.)
pub fn parity(a: &crate::args::Args) -> Result<()> {
    let want: Vec<&str> = a.pos.iter().skip(1).map(String::as_str).collect();
    let pairs: Vec<ParityCase> = parity_cases().into_iter().filter(|p| want.is_empty() || want.contains(&p.0)).collect();
    if !want.is_empty() && pairs.len() != want.len() {
        return err(format!("cases: {}", parity_cases().iter().map(|p| p.0).collect::<Vec<_>>().join(", ")));
    }
    let (fixtures, scratch, only_level) = (fixtures_dir(a), Scratch::new(a)?, a.flag("level"));
    let kind_bound = |level: &str| -> Result<Bound> {
        let f = Fixture::read(&fixtures.join("qwen35.json.gz"))?;
        f.bounds.into_iter().find(|b| b.level == level).map(|b| b.bound).ok_or_else(|| Error(format!("the Qwen3.5 layout's fixture bounds no level {level}")))
    };
    // (--choice ROW:LAYER: no engine; the reference alone with the nearest other choice in that row's pass and layer)
    let choice = match a.flag("choice") {
        Some(c) => {
            let parse = |x: &str| x.parse::<usize>().map_err(|_| Error(format!("--choice {c}: ROW:LAYER")));
            let (row, layer) = c.split_once(':').ok_or_else(|| Error(format!("--choice {c}: ROW:LAYER")))?;
            Some((parse(row)?, parse(layer)?))
        }
        None => None,
    };
    let mut failed = Vec::new();
    for (id, desc) in pairs {
        let c = find_case(id)?;
        let dir = model_dir(&c, &scratch.dir)?;
        let files = fixture_files(&fixtures, id)?;
        let all = files.iter().map(|p| Fixture::read(p)).collect::<Result<Vec<_>>>()?;
        // (the CPU reference's rows: what the executor is held to)
        let Some(cf) = all.iter().find(|f| f.gpu.is_none()) else { return err(format!("{id}: no fixture of the CPU reference in {}", fixtures.display())) };
        if let Some((row, layer)) = choice {
            let own = ref_drive(&dir, cf, None, None)?;
            let d = other_choice(cf, &own, row, layer)?;
            println!(
                "{id}: the reference's choice in row {row}'s pass, layer {layer}: {:?}, the nearest other {:?} for {:?}, margin {:.2e}",
                own.routed[layer][d.token].own, d.engine, d.reference, d.margin
            );
            carry_lines(&dir, cf, &own, &d)?;
            continue;
        }
        let mut levels: Vec<String> = Vec::new();
        for b in all.iter().flat_map(|f| &f.bounds) {
            if !levels.contains(&b.level) && only_level.is_none_or(|l| l == b.level) {
                levels.push(b.level.clone());
            }
        }
        // (the bound: the lossless level's, the case's fixture's and the kind's the same)
        let Some(bound) = cf.bounds.iter().find(|b| b.level == "penguin").map(|b| b.bound.clone()) else {
            return err(format!("{id}: its reference fixture bounds no penguin level"));
        };
        let kind = kind_bound("penguin")?;
        let bound_ok = bound == kind;
        println!(
            "{id}: the bound {bound:?}{}",
            if bound_ok { ", the Qwen3.5 layout's fixture's".to_string() } else { format!(": NOT the Qwen3.5 layout's fixture's {kind:?}") }
        );
        let own = ref_drive(&dir, cf, None, None)?;
        let same = own.rows.iter().zip(&cf.rows).filter(|(l, r)| hash_of(l) == r.hash).count();
        let drift = own.rows.iter().zip(&cf.rows).map(|(l, r)| kl_of(r, l)).fold(0f64, f64::max);
        println!("     the reference run here against its fixture: {same}/{} rows the same bits, KL at most {drift:.2e}", cf.rows.len());
        let cpu_layers = cpu_layers(&dir, &cf.prompt)?;
        let arg = |v: &[f32]| v.iter().enumerate().fold(0, |b, (j, &x)| if x > v[b] { j } else { b });
        // (an engine's rows, its rows again with its routers' choices read (those passes eager) and the choices, its layers after the prompt)
        type Ran = (Vec<Vec<f32>>, Vec<Vec<f32>>, Option<Routing>, Option<Vec<Vec<f32>>>);
        let ran = |r: &mut dyn Run| -> Result<Ran> {
            let rows = drive(r, &cf.prompt, &cf.decode, &cf.long)?;
            let layers = r.layers(&cf.prompt)?;
            r.choices(true)?;
            let traced = drive(r, &cf.prompt, &cf.decode, &cf.long);
            let choices = r.choices(false)?;
            Ok((rows, traced?, choices, layers))
        };
        for level in levels {
            let mut ok = bound_ok;
            let mut why: Vec<String> = if bound_ok { Vec::new() } else { vec!["the fixture's bound is not the kind's".into()] };
            let mut new = desc(&dir, &level)?;
            let (new_rows, new_traced, new_choices, new_layers) = ran(new.as_mut())?;
            if level == "penguin" {
                if let Some((good, line)) = experts_line(&dir, new.as_mut())? {
                    println!("     {line}");
                    if !good {
                        ok = false;
                        why.push("the lossless level's experts on the GPU are not the checkpoint's".into());
                    }
                }
            }
            drop(new);
            if let Some(nl) = &new_layers {
                layer_lines(nl, &cpu_layers);
            }
            // (the executor against the reference's own rows: never left out)
            {
                let v = compare(cf, &bound, &new_rows);
                println!(
                    "     the executor against the reference's own rows: bits-equal {}/{}, KL mean {:.2e} max {:.2e}, best token {}/{} with a lead ({})",
                    v.identical,
                    v.rows,
                    v.mean_kl,
                    v.max_kl,
                    v.top1.0,
                    v.top1.1,
                    if v.ok { "within the bound" } else { "PAST the bound" }
                );
            }
            // 3. the best token where the reference's lead is clear; every row's lead, best tokens and KL
            let mut unclear = Vec::new();
            for (i, r) in cf.rows.iter().enumerate() {
                let lead = if r.logits.len() > 1 { r.logits[0] - r.logits[1] } else { f32::INFINITY };
                let n = arg(&new_rows[i]) as u32;
                println!("       row {i:2}: the reference's best {} by {lead:.4}; the executor's {n} KL {:.2e}", r.ids[0], kl_of(r, &new_rows[i]));
                if n != r.ids[0] {
                    if lead > CLEAR_LEAD {
                        ok = false;
                        why.push(format!("row {i}: the best token is {n}, the reference's {} leads by {lead:.4}", r.ids[0]));
                    } else {
                        unclear.push(format!("row {i} ({n} for {}, lead {lead:.4})", r.ids[0]));
                    }
                }
            }
            if !unclear.is_empty() {
                println!("     the executor's best token is not the reference's where its lead is {CLEAR_LEAD} or less (not required): {}", unclear.join(", "));
            }
            // 1. and 2. on the engine's own choices
            let traced_same = new_rows.iter().zip(&new_traced).filter(|(x, y)| x == y).count();
            println!("     the executor's rows with its choices read (every pass eager): {traced_same}/{} the same bits as without", new_rows.len());
            match &new_choices {
                Some(ch) => {
                    let h = held_on_choices("the executor", &dir, cf, &bound, &new_traced, ch, &own, ref_drive)?;
                    if !h.routing {
                        ok = false;
                        why.push("on its own input a router's choice differs from the reference router's by more than a near tie".into());
                    }
                    if !h.logits {
                        ok = false;
                        why.push("the logits are past the bound of the reference taking the executor's choices".into());
                    }
                    // (what the first choice apart leaves in the rows after it: a measurement, which decides nothing)
                    if let Some(Err(e)) = h.apart.first().map(|d| carry_lines(&dir, cf, &own, d)) {
                        println!("     what the first choice apart leaves was not measured: {}", e.msg);
                    }
                }
                None => {
                    ok = false;
                    why.push("the executor gave no view of its routers' choices".into());
                }
            }
            println!(
                "{} {id:12} {level:8} {} rows{}",
                if ok { "PASS" } else { "FAIL" },
                new_rows.len(),
                if ok { ": the routers' choices, the logits on them and the clear best tokens hold".to_string() } else { format!(": {}", why.join("; ")) }
            );
            if !ok {
                failed.push(format!("{id} at {level}"));
            }
        }
    }
    if failed.is_empty() {
        println!("the description executor is within the kinds' reference on every case");
        Ok(())
    } else {
        println!("OFF: {}", failed.join(", "));
        drop(scratch);
        std::process::exit(1);
    }
}

/// The nearest other choice to the reference's own in the pass of fixture row `row` (its last token) and mixture-of-experts layer `layer`: the best
/// expert the router left out for the last one it took, as an engine's choice apart there (`Apart`, its margin the choice's own).
fn other_choice(fx: &Fixture, own: &RefRun, row: usize, layer: usize) -> Result<Apart> {
    let (p, d) = (fx.prompt.len(), fx.decode.len());
    if row > d || p == 0 {
        return err(format!("row {row}: the prompt's row 0 or a decode step's, 1 to {d}"));
    }
    let token = p + row - 1;
    let r = own
        .routed
        .get(layer)
        .and_then(|l| l.get(token))
        .ok_or_else(|| Error(format!("layer {layer}: the model has {} mixture-of-experts layers", own.routed.len())))?;
    let by = |a: &usize, b: &usize| r.rank[*b].total_cmp(&r.rank[*a]).then(a.cmp(b));
    let mut left: Vec<usize> = (0..r.rank.len()).filter(|e| !r.own.contains(e)).collect();
    left.sort_by(by);
    let mut took = r.own.clone();
    took.sort_by(by);
    let (Some(&inn), Some(&out)) = (left.first(), took.last()) else { return err(format!("layer {layer}: its router leaves no expert out")) };
    Ok(Apart { token, layer, reference: vec![out], engine: vec![inn], margin: r.gap })
}

/// What one choice of an engine that is not the reference's own (`d`) leaves behind it, on the reference alone: the reference run again taking that
/// one choice and its own everywhere else, each row's KL from its own run from the choice's row on (that row, then the rows that read the states its
/// pass wrote). And which states carry it: the same again with the states after that pass taken from the reference's own run for every kind of block
/// but one (the linear attentions' states and convolutions; the attentions' keys, values and index keys; the n-gram embedding's history), so the rows
/// after it differ by that one kind's states alone.
fn carry_lines(dir: &Path, fx: &Fixture, own: &RefRun, d: &Apart) -> Result<()> {
    use crate::runtime::kinds::{Body, Ffn, Mixer};
    type Model = crate::runtime::reference::model::RefModel;
    let (p, n) = (fx.prompt.len(), fx.prompt.len() + fx.decode.len());
    if d.token >= n {
        println!("     the first choice apart is in the long prompt's pass, which no later row reads");
        return Ok(());
    }
    let first: Vec<u32> = fx.prompt.iter().chain(&fx.decode).copied().collect();
    // (the choice to take at that token: the reference's own experts there, with the engine's in the place of the ones it did not take)
    let mut choice: Vec<usize> = own.routed[d.layer][d.token].own.iter().copied().filter(|e| !d.reference.contains(e)).collect();
    choice.extend(&d.engine);
    let start = |flip: bool| -> Result<Model> {
        let mut m = ref_model(dir)?;
        let moe: Vec<usize> = (0..m.layers.len()).filter(|&b| matches!(m.layers[b].block.body, Body::Ffn(Ffn::Moe(_)))).collect();
        let Some(&b) = moe.get(d.layer) else { return err(format!("{} mixture-of-experts blocks, and a choice in layer {}", moe.len(), d.layer)) };
        if flip {
            m.layers[b].take = (0..=d.token).map(|t| if t == d.token { choice.clone() } else { Vec::new() }).collect();
        }
        Ok(m)
    };
    // (tokens `at` of the first sequence through `m`: the rows of the passes from the prompt's last on)
    let go = |m: &mut Model, at: std::ops::Range<usize>, rows: &mut Vec<Vec<f32>>| -> Result<()> {
        for i in at {
            let l = m.forward(&[first[i]], i as u32)?.pop().ok_or_else(|| Error("a token without logits".into()))?;
            if i + 1 >= p {
                rows.push(l);
            }
        }
        Ok(())
    };
    let from = row_of_token(fx, d.token).0;
    let kl = |rows: &[Vec<f32>]| -> Result<String> {
        let mut v = Vec::new();
        for (i, r) in rows.iter().enumerate().skip(from) {
            v.push(format!("row {i} {:.2e}", kl_of(&Row::from_logits(&own.rows[i])?, r)));
        }
        Ok(v.join(", "))
    };
    let (mut m, mut rows) = (start(true)?, Vec::new());
    go(&mut m, 0..n, &mut rows)?;
    println!(
        "     the reference taking that one choice ({}, layer {}: {:?} for {:?}) and its own everywhere else, each row's KL from its own run: {}",
        row_of_token(fx, d.token).1,
        d.layer,
        d.engine,
        d.reference,
        kl(&rows)?
    );
    let kinds: [(&str, &dyn Fn(&Body) -> bool); 3] = [
        ("the linear attentions' states and convolutions", &|b| matches!(b, Body::Mixer(Mixer::Linear(_)))),
        ("the attentions' keys, values and index keys", &|b| matches!(b, Body::Mixer(m) if !matches!(m, Mixer::Linear(_)))),
        ("the n-gram embedding's history", &|b| matches!(b, Body::Ngram(_))),
    ];
    for (what, which) in kinds {
        let (mut lender, mut m, mut rows, mut none) = (start(false)?, start(true)?, Vec::new(), Vec::new());
        go(&mut lender, 0..d.token + 1, &mut none)?;
        go(&mut m, 0..d.token + 1, &mut rows)?;
        let mut held = 0usize;
        for (a, b) in m.layers.iter_mut().zip(lender.layers.iter_mut()) {
            if which(&a.block.body) {
                held += 1;
            } else {
                std::mem::swap(&mut a.seqs, &mut b.seqs);
            }
        }
        go(&mut m, d.token + 1..n, &mut rows)?;
        println!("       with only {what} ({held} blocks) from that pass, every other state the reference's own: {}", kl(&rows)?);
    }
    Ok(())
}

/// The CPU f32 reference's residual after each layer of a prompt of the model in `dir` (from its description).
fn cpu_layers(dir: &Path, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
    use crate::runtime::reference::model::{checkpoint_tensor, RefModel};
    let cfg: Value = serde_json::from_str(&io(std::fs::read_to_string(dir.join("config.json")))?).map_err(|e| Error(format!("config.json: {e}")))?;
    let desc = crate::runtime::describe::describe(&cfg)?;
    let ck = io(glyd_gpu::safetensors::Checkpoint::open(dir))?;
    let mut m = RefModel::load(&desc, &|n: &str| checkpoint_tensor(&ck, n))?;
    m.record = Some(Vec::new());
    m.forward(tokens, 0)?;
    Ok(m.record.take().unwrap_or_default())
}

/// The executor's residual after each layer of a prompt against the CPU reference's: per layer the largest difference over the largest magnitude of the
/// reference's, and the values with the same bits.
fn layer_lines(new: &[Vec<f32>], cpu: &[Vec<f32>]) {
    if cpu.len() != new.len() {
        println!("     layers: the reference traced {}, the executor {}", cpu.len(), new.len());
    }
    for (l, (n, c)) in new.iter().zip(cpu).enumerate().filter(|(_, (n, c))| n.len() == c.len()) {
        let scale = c.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
        let d = n.iter().zip(c).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale;
        let same = n.iter().zip(c).filter(|(x, y)| x.to_bits() == y.to_bits()).count();
        println!("     layer {l}: against the CPU reference apart by {d:.2e}, {same}/{} values the same bits", n.len());
    }
}

/// The per-level KL thresholds of a lossy level from the lossless one (mean, max over rows), per family's random model: Qwen3.5's and
/// Nemotron-H's are the bounds the qwen35 fixture holds for the engine against transformers in bf16 (tests/logits/qwen35.json.gz, about
/// twice a measured run); Gemma 4's random model moved more under every rounding, kestrel too (its levels have shipped since v0.30), so
/// its thresholds are about twice its own first measured run (RTX 4080 SUPER, 2026-11-21: mean/max KL kestrel 2.15e-2/1.03e-1, swift
/// 3.70e-2/1.01e-1, wren 9.55e-2/2.27e-1; Nemotron-H's wren 8.5e-4/6.2e-3, its experts alone four-bit). (Measured on the draw before
/// `gemma4_shaped_random`'s, whose scores and head are well conditioned: they hold this one as upper bounds.)
pub const LEVEL_KL: [(&str, f64, f64); 3] = [("kestrel", 0.006, 0.015), ("swift", 0.025, 0.05), ("wren", 0.09, 0.14)];
pub const LEVEL_KL_GEMMA4: [(&str, f64, f64); 3] = [("kestrel", 0.045, 0.21), ("swift", 0.075, 0.21), ("wren", 0.19, 0.46)];

/// The hummingbird level's KL thresholds from the lossless one (mean, max over rows), per mixture-of-experts family's random model: the
/// model's routed experts coded at 3 and 2 bits by use (src/hb_pack.rs) from a corpus pass over made-up documents (src/hb_usage.rs,
/// `synthetic`), every other matrix as released. Each is twice its family's first measured run, to two figures, as Gemma 4's above
/// (A10, 2026-11-25, the same figures in two runs: mean/max KL DeepSeek-V4 9.32e-2/1.30e-1, Muse Spark 2.10e-2/2.69e-2, Kimi-K3
/// 1.46e-2/2.92e-2, its first run once the corpus pass counted its layers; GLM-5.3-Flash 2.90e-2/4.79e-2, its first run once its
/// directory loaded, by the executor's loader). Kimi-K3's was measured on the draw before `kimi_ref::random_with`'s well-conditioned one,
/// which it holds as an upper bound (an A10, 2026-11-25: 6.17e-3/9.87e-3).
pub const LEVEL_KL_HB: [(&str, f64, f64); 4] = [("deepseek-v4", 0.19, 0.26), ("muse", 0.042, 0.054), ("kimi", 0.029, 0.058), ("glm", 0.058, 0.096)];

/// The mixture-of-experts families that run at hummingbird (src/support.rs) and whose hummingbird directory this tree does not make or
/// load on a random model, with what stops it: none now. The check runs each and holds it to that: one that runs after all fails the
/// check with its figures, to be given a threshold in `LEVEL_KL_HB` and taken out of here.
pub const LEVEL_HB_OPEN: [(&str, &str); 0] = [];

/// The families held at the block levels of every weight (kestrel, swift, wren: `LEVEL_KL`); `LEVEL_KL_HB`'s are held at hummingbird.
const LEVEL_BLOCK_CASES: [&str; 3] = ["qwen35", "gemma4", "nemotron-h"];

/// A mixture-of-experts case's model in `dir` with every matrix as `dir` holds it: the release's lossless, or a hummingbird directory's
/// experts in their codes. (GLM-5.3-Flash's logits case runs its experts at a block level; here they are lossless, as kindcheck loads them.)
fn moe_as_held(id: &str, dir: &Path) -> Result<Box<dyn Run>> {
    match id {
        "glm" => boxed(crate::dynamic::glm::load(dir, &dyn_opts(Some(Layout::Tiered)), Layout::Tiered)),
        // (with the release's quantizations simulated, as the product runs it and as these thresholds were measured: its case's own engine is without them)
        "deepseek-v4" => dsv4_engine_sim(dir, "penguin"),
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
/// the state-space case at kestrel, swift and wren (`LEVEL_KL`, `LEVEL_KL_GEMMA4` for Gemma 4); DeepSeek-V4, Muse Spark, Kimi-K3 and
/// GLM-5.3-Flash at hummingbird (`LEVEL_KL_HB`, the directory made here). Exit 1 where one is past its threshold, or does not load: a
/// level that does not run is not one within its threshold. A family of `LEVEL_HB_OPEN` (none now) is printed as not held, with what
/// stops it, and fails the day it runs without a threshold.
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

/// `logitsmodel CASE DIR [--servable] [--real [--layers L] [--experts E]]`: the case's random model made in DIR (`--real`: the Qwen3.5 layout with experts at its release's width) (a model to run the other checks on: `kvtest DIR`, `run DIR`); `qwen4x-rel` the
/// family's release's form (`qwen4x_release_random`); `CASE+q4` with its tables on the host stored at 4 bits.
pub fn make_model(a: &crate::args::Args) -> Result<()> {
    let id = a.need(1, "CASE")?;
    let (base, q4) = q4_case(id);
    let dir = Path::new(a.need(2, "DIR")?);
    // (a case of the logits checks by its own maker; another random model `random_model` makes, the release's form among them; + q4: its tables at 4 bits)
    match find_case(base).map(|c| c.weights) {
        Ok(Weights::Hub(h)) => return err(format!("{id}: a real model ({h}), not a random one")),
        // (`--real --layers L --experts E`: the Qwen3.5 layout with experts at its release's width, `qwen35moe_random_real`)
        Ok(Weights::Random(_)) if a.has("real") => match base {
            "qwen35moe" => qwen35moe_random_real(dir, a.num("layers", 8usize)?, a.num("experts", 256usize)?)?,
            id => return err(format!("--real: the qwen35moe case's release width only (not {id})")),
        },
        Ok(Weights::Random(make)) => make(dir)?,
        Err(_) => random_model_of(base, dir)?,
    }
    if q4 {
        tables_at_4_bits(dir, &|_| true)?;
    }
    // (`--servable`: the files a server needs besides the weights, for a client to talk to it: a tokenizer, the chat template and the end token; the
    // family's chat template where one is made: Qwen's for the Qwen3.5 layout's and Qwen3.8-Flash-Next's cases, GLM-5.3's for its layout's and
    // GLM-5.3-Flash's, Kimi-K3's own (src/chat.rs `KIMI_CHAT`) for its cases, DeepSeek-V4.1's release's own (tests/fixtures/chat_deepseek_v41.jinja, its
    // chat_template.jinja) for its case)
    if a.has("servable") {
        let template = match base {
            "qwen35" | "qwen35-wide" | "qwen35moe" | "qwen4x" | "qwen4x-s2" | "qwen4x-cb" | "qwen4x-rel" => include_str!("../tests/fixtures/chat_qwen38_tools.jinja"),
            "glm" | "glm-dsa" | "glm-dsa-f8" => include_str!("../tests/fixtures/chat_glm53_tools.jinja"),
            "kimi" | "kimi-dsa" => crate::chat::KIMI_CHAT,
            "deepseek-v41" => include_str!("../tests/fixtures/chat_deepseek_v41.jinja"),
            id => {
                return err(format!(
                    "--servable: only the cases whose family's chat template is made (qwen35, qwen35-wide, qwen35moe, qwen4x, qwen4x-s2, qwen4x-cb, qwen4x-rel, glm, glm-dsa, glm-dsa-f8, kimi, kimi-dsa, deepseek-v41; not {id})"
                ))
            }
        };
        write_serving_files(dir, template)?;
    }
    Ok(())
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
    write_serving_files_with(dir, template, &[])
}

/// `write_serving_files` with `extra` special tokens after the others (a model that reads images: its image's markers and place).
pub fn write_serving_files_with(dir: &Path, template: &str, extra: &[&str]) -> Result<()> {
    let io = |r: std::io::Result<()>| r.map_err(|e| Error(format!("{}: {e}", dir.display())));
    let mut tokens = tiny_tokens();
    tokens.extend(extra.iter().map(|t| (t.to_string(), true)));
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
            // (the reference's own words: the description's on the layer kinds' reference, or a hand-written one's file; `CpuRef::source`)
            "cpu" => format!("{}, fp32 inside, on this case's random model (seed fixed in the generator)", run.describe()),
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
    use std::collections::VecDeque;

    /// A fixture's provenance is its CPU reference's own words, not a list of case ids: every case on the description's reference says so (the two Gemma 4
    /// cases too, which such a list had left out, so that gemma4-kv's fixture named a src/gemma4kv_ref.rs that never was), and a hand-written reference
    /// names its file, which is in the tree. (The references load on the first reset: nothing is read here.)
    #[test]
    fn a_cpu_references_provenance_is_its_own() {
        let described = "the model's description (src/runtime/describe.rs) on the CPU reference of the layer kinds (src/runtime/reference)";
        let mut on_description = Vec::new();
        for c in cases() {
            let Some(cpu) = c.cpu else { continue };
            let source = cpu(Path::new("no-such-model")).unwrap().describe();
            match source.strip_prefix("the CPU reference of ") {
                Some(file) => assert!(Path::new(env!("CARGO_MANIFEST_DIR")).join(file).is_file(), "{}: {source}", c.id),
                None => {
                    assert_eq!(source, described, "{}", c.id);
                    on_description.push(c.id);
                }
            }
        }
        for id in ["gemma4", "gemma4-kv", "dense", "qwen35", "nemotron-h"] {
            assert!(on_description.contains(&id), "{id}: {on_description:?}");
        }
        let kv = Fixture::read(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/logits/gemma4-kv.json.gz")).unwrap();
        assert!(kv.provenance["source"].starts_with(described), "{}", kv.provenance["source"]);
    }

    /// The storage roundings every engine makes and the CPU reference of the layer kinds does not, where the description keeps the value in f32:
    /// each block's norms' outputs and the final norm's rounded once (an engine holds its activations in bf16), and the embedding's scale its bf16.
    fn stored(d: &mut crate::runtime::describe::Described) {
        use crate::runtime::kinds::{Norm, Rounding};
        let once = |n: &mut Norm| {
            if n.rounding == Rounding::F32 {
                n.rounding = Rounding::Once;
            }
        };
        for b in d.desc.blocks.iter_mut() {
            once(&mut b.norm_in);
            if let Some(n) = b.norm_out.as_mut() {
                once(n);
            }
        }
        once(&mut d.desc.final_norm);
        d.desc.embed.scale = crate::runtime::reference::bf16(d.desc.embed.scale);
        // (and a draft head's: its front's norms, its blocks', its final mixer's)
        for h in d.desc.mtp.iter_mut() {
            (once(&mut h.norm_embed), once(&mut h.norm_hidden), once(&mut h.final_norm));
            for b in h.blocks.iter_mut() {
                once(&mut b.norm_in);
                if let Some(n) = b.norm_out.as_mut() {
                    once(n);
                }
            }
        }
    }

    /// The random checkpoint stores what its manifest lists as bf16 in bf16, the tied embedding first (a release stores it so), whatever the draw's scale: the
    /// `qwen35` case's floor (`NOISE_FLOOR_OPEN`) is not that of a tensor held wider than a release holds it. Read back from the file's header, on a manifest cut
    /// down to an embedding, a norm and two matrices; a tensor the manifest lists as int64 is stored as listed.
    #[test]
    fn the_random_checkpoint_stores_a_manifests_bf16_tensors_in_bf16() {
        let tensors = json!([
            ["model.embed_tokens.weight", "BF16", [64, 32]],
            ["model.layers.0.input_layernorm.weight", "BF16", [32]],
            ["model.layers.0.mlp.gate_proj.weight", "BF16", [48, 32]],
            ["model.layers.0.self_attn.q_proj.weight", "BF16", [32, 32]],
            ["model.layers.0.ngram_embedding.ids", "I64", [3], [1, 2, 3]],
        ]);
        let dir = std::env::temp_dir().join(format!("glyd-bf16-writer-{}", std::process::id()));
        manifest_random(&json!({"config": "{}", "tensors": tensors}).to_string(), &dir, 0x2026_1125_0000_0777, false, "bf16 writer").unwrap();
        let file = glyd_gpu::safetensors::File::open(&dir.join("model.safetensors")).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        for e in tensors.as_array().unwrap() {
            let (name, listed) = (e[0].as_str().unwrap(), e[1].as_str().unwrap());
            let held = file.get(name).unwrap_or_else(|| panic!("{name}: not written"));
            assert_eq!(held.dtype, listed, "{name}");
            assert_eq!(held.shape, e[2].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect::<Vec<_>>(), "{name}");
        }
        assert_eq!(file.get("model.embed_tokens.weight").unwrap().bytes, 64 * 32 * 2, "the embedding two bytes a value");
    }

    /// A case's noise floor: the CPU reference of the layer kinds against itself with `stored`'s roundings, on the fixture's tokens, as `compare`
    /// measures an engine against `bound`; where the case is held on its routers' choices (`Case::on_choices`), the rounded run takes the
    /// reference's own.
    fn noise_floor(dir: &Path, fx: &Fixture, on_choices: bool, bound: &Bound) -> Result<Verdict> {
        noise_floor_with(dir, fx, on_choices, bound, stored)
    }

    /// `noise_floor` with `edit`'s description in the place of `stored`'s (`ModelDesc::f32_arithmetic`: the floor against fp32).
    fn noise_floor_with(dir: &Path, fx: &Fixture, on_choices: bool, bound: &Bound, edit: fn(&mut crate::runtime::describe::Described)) -> Result<Verdict> {
        let base = ref_drive(dir, fx, None, None)?;
        let take: Option<Choices> = on_choices.then(|| base.routed.iter().map(|l| l.iter().map(|r| r.ids.clone()).collect()).collect());
        // (and its indexers' picks, where it has any)
        let picks: Option<Picks> = on_choices.then(|| {
            base.picked
                .iter()
                .map(|l| {
                    l.iter().map(|r| PickRow { pos: 0, picks: r.picks.clone(), q: Vec::new(), w: Vec::new(), keys: Default::default(), inside: None }).collect()
                })
                .collect()
        });
        let rounded = ref_drive_with(dir, fx, take.as_ref(), picks.as_ref(), edit)?;
        let own = Fixture { rows: base.rows.iter().map(|r| Row::from_logits(r)).collect::<Result<_>>()?, ..fx.clone() };
        Ok(compare(&own, bound, &rounded.rows))
    }

    /// The family's draft head (`qwen4x_mtp_random`: the `qwen4x` case's model with one MTP layer) is well conditioned for the bound the family is held to
    /// (the `qwen4x` fixture's): on the reference's own streams of the fixture's prompt and decode tokens, a row a position (the streams there and the token
    /// after it), the head with `stored`'s roundings (its norms' outputs rounded once) against the head without them, on the head's own routers' choices, is
    /// within a tenth of the bound; and the head is no copy of the model's next-token answer (its best token is not the model's at every row).
    #[cfg(feature = "dev")]
    #[test]
    fn the_draft_heads_case_is_well_conditioned() {
        let dir = std::env::temp_dir().join(format!("glyd-floor-qwen4x-mtp-{}", std::process::id()));
        qwen4x_mtp_random(&dir).unwrap();
        let fx = Fixture::read(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/logits/qwen4x.json.gz")).unwrap();
        let tenth = match fx.bounds[0].bound {
            Bound::Kl { mean, max, top1 } => Bound::Kl { mean: mean / 10.0, max: max / 10.0, top1 },
            Bound::Bits => panic!("a reference fixture of bits"),
        };
        let toks: Vec<u32> = fx.prompt.iter().chain(&fx.decode).copied().collect();
        let st = ref_streams(&dir, &toks).unwrap();
        let rows: Vec<HeadRow> =
            (0..toks.len() - 1).map(|i| HeadRow { pos: i as u32, token: toks[i + 1], streams: st[i].clone(), take: None, picks: None }).collect();
        let HeadRef { logits: base, routes, .. } = ref_head(&dir, &rows, |_| {}).unwrap();
        let taken: Vec<HeadRow> = rows.iter().zip(&routes).map(|(r, (own, _))| HeadRow { take: Some(own.clone()), ..r.clone() }).collect();
        let rounded = ref_head(&dir, &taken, stored).unwrap().logits;
        // (the model itself: its own next-token rows on the same tokens)
        let model = described_rows(&dir, &toks).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        let own = Fixture { rows: base.iter().map(|r| Row::from_logits(r)).collect::<Result<_>>().unwrap(), ..fx.clone() };
        let v = compare(&own, &tenth, &rounded);
        let Bound::Kl { mean, max, .. } = tenth else { unreachable!() };
        println!(
            "qwen4x-mtp head noise floor: KL mean {:.2e} max {:.2e}, best token {}/{}, largest logit difference {:.4} over {} rows; a tenth of the bound: mean {mean:.1e}, max {max:.1e}",
            v.mean_kl,
            v.max_kl,
            v.top1.0,
            v.top1.1,
            v.max_dlogit,
            base.len()
        );
        assert!(v.ok, "{:?}", v.notes);
        let best = |r: &[f32]| (0..r.len()).fold(0, |b, t| if r[t] > r[b] { t } else { b });
        let same = base.iter().zip(&model[1..]).filter(|(h, m)| best(h) == best(m)).count();
        assert!(same < base.len(), "the head's best token is the model's next at every row");
    }

    /// The DSpark drafters' cases (`draft_model`: deepseek-v4's two draft layers and a block of 8, deepseek-v41's release-shaped one) hold every tensor their
    /// descriptions read (`describe_dir`), and the CPU reference drafts a block on each: on the reference's own taps of a prompt of 20, the block after it of
    /// the drafter's size, its logits finite, the Markov head moving them, its blocks of experts each choosing for every row.
    #[cfg(feature = "dev")]
    #[test]
    fn the_dspark_drafters_cases_draft_on_the_reference() {
        for case in ["deepseek-v4", "deepseek-v41"] {
            let dir = std::env::temp_dir().join(format!("glyd-dspark-{case}-{}", std::process::id()));
            draft_model(case, &dir).unwrap();
            let (d, _) = crate::runtime::describe::describe_dir(&dir).unwrap_or_else(|e| panic!("{case}: {}", e.msg));
            let ds = d.desc.mtp[0].dspark.clone().unwrap();
            let toks: Vec<u32> = (0..20u32).map(|i| 3 + (i * 7919) % (d.desc.vocab as u32 - 3)).collect();
            let taps = ref_taps(&dir, &toks).unwrap();
            let (embed, head, mut r) = ref_dspark(&dir).unwrap();
            std::fs::remove_dir_all(&dir).ok();
            assert_eq!((taps.len(), taps[0].len()), (20, ds.targets.len() * d.desc.hidden), "{case}");
            for b in r.blocks.iter_mut().filter(|b| matches!(b.block.body, crate::runtime::kinds::Body::Ffn(_))) {
                b.routed = Some(Vec::new());
            }
            let out = r.draft(&taps[..19], toks[19], &embed, &head).unwrap();
            assert_eq!(out.tokens.len(), ds.block, "{case}");
            assert!(out.logits.iter().flatten().all(|x| x.is_finite()), "{case}");
            assert!(out.logits.iter().zip(&out.base).all(|(l, b)| l != b), "{case}: the Markov head moves no draft's logits");
            assert!(r.blocks.iter().filter_map(|b| b.routed.as_ref()).all(|x| x.len() == ds.block), "{case}");
            println!("{case}: the reference's block after a prompt of 20: drafts {:?}, confidence {:?}", out.tokens, out.confidence);
        }
    }

    /// GLM-5.3's draft head (`glm_dsa_mtp_random`: the glm-dsa and glm-dsa-f8 cases' models with one MTP layer, DeepSeek-V3's joined head) is well conditioned
    /// for the bound the family is held to (each case's fixture's): on the reference's own final normed states of the fixture's prompt and decode tokens, a
    /// row a position (the state there and the token after it), the head with its description's rounding points (its norms' `Twice`) against every one in
    /// fp32 (`ModelDesc::f32_arithmetic`), the fp32 run on the rounded run's experts and indexer picks, is within a tenth of the bound; and the head is no
    /// copy of the model's next-token answer.
    #[cfg(feature = "dev")]
    #[test]
    fn the_glm_draft_heads_case_is_well_conditioned() {
        for (case, fp8) in [("glm-dsa", false), ("glm-dsa-f8", true)] {
            let dir = std::env::temp_dir().join(format!("glyd-floor-{case}-mtp-{}", std::process::id()));
            glm_dsa_mtp_random(&dir, fp8).unwrap();
            let fx = Fixture::read(&Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/logits/{case}.json.gz"))).unwrap();
            let tenth = match fx.bounds[0].bound {
                Bound::Kl { mean, max, top1 } => Bound::Kl { mean: mean / 10.0, max: max / 10.0, top1 },
                Bound::Bits => panic!("a reference fixture of bits"),
            };
            let toks: Vec<u32> = fx.prompt.iter().chain(&fx.decode).copied().collect();
            let st = ref_streams(&dir, &toks).unwrap();
            let rows: Vec<HeadRow> =
                (0..toks.len() - 1).map(|i| HeadRow { pos: i as u32, token: toks[i + 1], streams: st[i].clone(), take: None, picks: None }).collect();
            let base = ref_head(&dir, &rows, |_| {}).unwrap();
            let taken: Vec<HeadRow> = rows
                .iter()
                .zip(base.routes.iter().zip(&base.picks))
                .map(|(r, ((experts, _), (picks, _)))| HeadRow { take: Some(experts.clone()), picks: Some(picks.clone()), ..r.clone() })
                .collect();
            let f32s = ref_head(&dir, &taken, |d| d.desc.f32_arithmetic()).unwrap().logits;
            let model = described_rows(&dir, &toks).unwrap();
            std::fs::remove_dir_all(&dir).ok();
            let own = Fixture { rows: base.logits.iter().map(|r| Row::from_logits(r)).collect::<Result<_>>().unwrap(), ..fx.clone() };
            let v = compare(&own, &tenth, &f32s);
            println!(
                "{case} draft head noise floor: KL mean {:.2e} max {:.2e}, best token {}/{}, largest logit difference {:.4} over {} rows (the indexer leaves positions out at {} of {})",
                v.mean_kl,
                v.max_kl,
                v.top1.0,
                v.top1.1,
                v.max_dlogit,
                base.logits.len(),
                base.picks.iter().enumerate().filter(|(i, (p, _))| p.len() <= *i).count(),
                rows.len()
            );
            assert!(v.ok, "{case}: {:?}", v.notes);
            let best = |r: &[f32]| (0..r.len()).fold(0, |b, t| if r[t] > r[b] { t } else { b });
            let same = base.logits.iter().zip(&model[1..]).filter(|(h, m)| best(h) == best(m)).count();
            assert!(same < base.logits.len(), "{case}: the head's best token is the model's next at every row");
        }
    }

    /// The model's rows on the CPU reference of the kinds over `tokens` from position 0, a row a token.
    #[cfg(feature = "dev")]
    fn described_rows(dir: &Path, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        ref_model(dir)?.forward(tokens, 0)
    }

    /// The cases past a tenth of their bound today, each with why: printed, not held to the line (one near it may read either side of it on another
    /// machine's libm). Its draw is its family's owner's to fix, never its bound.
    const NOISE_FLOOR_OPEN: [(&str, &str); 1] = [(
        "qwen35",
        "1.04e-5 in KL mean against 1.0e-5, four more seeds 8.3e-6 to 1.05e-5; its checkpoint holds nothing wider than a release does (the writer stores \
         the embedding and every drawn tensor in bf16: `the_random_checkpoint_stores_a_manifests_bf16_tensors_in_bf16`), so no f32 tensor to narrow: 70% of \
         it is layer 0's input, as Qwen3.5 adds its tied embedding unscaled and the first block's output is all of the residual; an embedding wide enough \
         to share the residual makes a token's own row the head's largest (19.9, the softmax a point), so no honest draw lowers it",
    )];

    /// Every case a CPU reference holds is well conditioned for its bound: its noise floor (`noise_floor`) is at most a tenth of its reference
    /// fixture's bound, in KL mean and max, the best token kept where the bound asks it. A random case whose draw cannot tell a fault from rounding
    /// (Gemma 4's were 3e-4 to 1.1e-3 against a bound of 1e-4, Muse's 1.2e-4 from its scores' scale) fails here, and its draw is fixed, not its
    /// bound (`NOISE_FLOOR_OPEN`).
    ///
    /// A case with routers is measured on its reference's own expert choices where it is held on its routers' choices (`Case::on_choices`), which
    /// is how a case with free routers must be held: a choice flips where the router's gap between the last expert chosen and the first left out is
    /// below the change the input's rounding (about 2^-9 of it) makes in the router's scores, and both scale with the router's weights, so no
    /// router draw clears it short of one that takes the same experts for every token (Muse, before it was held on choices: 187 of 744 choices
    /// apart under these roundings, a floor of 2.2e-3; qwen35moe 18, one row 1.9e-4).
    ///
    /// Each case's reference is its description's on the layer kinds (DeepSeek-V4's case is held to src/dsv4_ref.rs, without its simulated
    /// quantizations: that reference's floor is `deepseek_v4_noise_floor`). Where a description already rounds every norm (Kimi-K3's, GLM-5.3-Flash's) and
    /// scales no embedding, `stored` moves nothing: its floor is 0 and says nothing of its draw (theirs are the reference in fp32 against all its
    /// rounding points, `noise_floor_against_fp32`, `glm_noise_floor`). Not here: nemotron-h (an older family).
    #[cfg(feature = "dev")]
    #[test]
    fn every_case_held_to_a_reference_is_well_conditioned() {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/logits");
        let all: Vec<Case> = cases().into_iter().filter(|c| c.cpu.is_some() && c.id != "nemotron-h").collect();
        let floors: Vec<(&str, Result<(Verdict, Bound)>)> = std::thread::scope(|s| {
            let fixtures = &fixtures;
            let runs: Vec<_> = all
                .iter()
                .map(|c| {
                    s.spawn(move || {
                        let dir = std::env::temp_dir().join(format!("glyd-floor-{}-{}", c.id, std::process::id()));
                        let r = (|| {
                            let Weights::Random(make) = &c.weights else { return err("a Hub model") };
                            make(&dir)?;
                            let fx = Fixture::read(&fixtures.join(format!("{}.json.gz", c.id)))?;
                            let tenth = match fx.bounds[0].bound {
                                Bound::Kl { mean, max, top1 } => Bound::Kl { mean: mean / 10.0, max: max / 10.0, top1 },
                                Bound::Bits => return err("a reference fixture of bits"),
                            };
                            Ok((noise_floor(&dir, &fx, c.on_choices.is_some(), &tenth)?, tenth))
                        })();
                        std::fs::remove_dir_all(&dir).ok();
                        (c.id, r)
                    })
                })
                .collect();
            runs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut wrong = Vec::new();
        for (id, r) in &floors {
            let (v, tenth) = r.as_ref().unwrap_or_else(|e| panic!("{id}: {}", e.msg));
            let Bound::Kl { mean, max, .. } = tenth else { unreachable!() };
            let open = NOISE_FLOOR_OPEN.iter().find(|o| o.0 == *id);
            println!(
                "{} {id:12} noise floor: KL mean {:.2e} max {:.2e}, best token {}/{}, largest logit difference {:.4}; a tenth of its bound: mean {mean:.1e}, max {max:.1e}{}",
                if v.ok { "PASS" } else if open.is_some() { "OPEN" } else { "FAIL" },
                v.mean_kl,
                v.max_kl,
                v.top1.0,
                v.top1.1,
                v.max_dlogit,
                match open {
                    Some(o) if v.ok => format!(" (open, {}: within it here, off the list once it is on every machine)", o.1),
                    Some(o) => format!(" (open: {}, its draw its owner's to fix)", o.1),
                    None => String::new(),
                }
            );
            if !v.ok && open.is_none() {
                wrong.push(*id);
            }
        }
        assert!(wrong.is_empty(), "past a tenth of the bound: {wrong:?} (a draw to fix, not a bound)");
    }

    /// Muse Spark's two CPU references on the same draw and the same expert choices (the description's, which its case holds the engine to, and
    /// src/muse_ref.rs's): the same function term by term (router, gated residuals, attention, experts, head); muse_ref.rs rounds to bf16 at every
    /// operation of the release's eager bf16 arithmetic, the description's where its kinds say, so they part by those roundings alone: on the Mac,
    /// 2026-11-25, KL mean 8.6e-6 and 1.3e-5 at most, within a tenth of the kind's bound, every best token (on their own choices 2.3e-3: router
    /// flips). A minute: cargo test --profile ci -p glyd-engine --lib -- --ignored muse_references.
    #[cfg(feature = "dev")]
    #[test]
    #[ignore]
    fn muse_references_agree_on_the_same_choices() {
        let v = references_on_the_same_choices(
            "muse",
            crate::muse_ref::random,
            |_| {},
            &TENTH,
            |dir, take| {
                let mut m = crate::muse_ref::RefMuse::load(dir).unwrap();
                *m.take.borrow_mut() = take;
                Box::new(move |t| m.step(t))
            },
        );
        assert!(v.ok, "{:?}", v.notes);
    }

    /// A tenth of the kind's bound: how far two references of one function may part on the same choices by their roundings alone.
    const TENTH: Bound = Bound::Kl { mean: 1e-5, max: 2.5e-5, top1: 0.85 };

    /// Kimi-K3's hand-written reference (src/kimi_ref.rs) with its bf16 roundings off against the description's in fp32 (`ModelDesc::f32_arithmetic`),
    /// on the same draw and expert choices: the same function term by term (the depth mix, the delta attention and its gated norm, the latent attention,
    /// the latent block of experts with its norm, the situ activation, the head), their logits apart by fp32's roundings alone (KL mean at most 1e-9: a
    /// value of one operation off by 1e-3 of itself moves them by more). A minute: cargo test --profile ci -p glyd-engine --lib -- --ignored
    /// kimi_references_compute.
    #[cfg(feature = "dev")]
    #[test]
    #[ignore]
    fn kimi_references_compute_the_same_function() {
        crate::kimi_ref::PLAIN.with(|p| p.set(true));
        let same = Bound::Kl { mean: 1e-9, max: 1e-8, top1: 1.0 };
        let v = references_on_the_same_choices("kimi", |d| crate::kimi_ref::random(d, 1, true, 8), |d| d.desc.f32_arithmetic(), &same, kimi_hand);
        crate::kimi_ref::PLAIN.with(|p| p.set(false));
        assert!(v.ok, "{:?}", v.notes);
    }

    /// Kimi-K3's two CPU references with their roundings on the same draw and the same expert choices, as Muse Spark's: the description's, which its
    /// case holds the engine to, and src/kimi_ref.rs's, which rounds to bf16 wherever the release holds a bf16 tensor (every product's output, the depth
    /// mix and its partial sum, the delta attention's convolutions and output, the situ activations, the experts' sum) where the kinds' reference keeps
    /// fp32 between the points its description names. The same function (`kimi_references_compute_the_same_function`), so they part by those roundings
    /// alone: on the Mac, 2026-11-25, KL mean 1.08e-5 and 3.7e-5 at most, three times the description's floor (as Muse Spark's two are), spread
    /// over every class of the hand-written one's roundings (none alone moves it by more than a third; the largest, the partial sum's, given to the
    /// description too, moved it not at all: 1.13e-5). Held to the kind's bound, as an engine that rounds as the release does is. A minute: cargo test
    /// --profile ci -p glyd-engine --lib -- --ignored kimi_references_agree.
    #[cfg(feature = "dev")]
    #[test]
    #[ignore]
    fn kimi_references_agree_on_the_same_choices() {
        let kind = Bound::Kl { mean: 1e-4, max: 2.5e-4, top1: 0.85 };
        let v = references_on_the_same_choices("kimi", |d| crate::kimi_ref::random(d, 1, true, 8), |_| {}, &kind, kimi_hand);
        assert!(v.ok, "{:?}", v.notes);
    }

    /// src/kimi_ref.rs's reference on a directory, taking the experts given.
    fn kimi_hand(dir: &Path, take: Vec<VecDeque<Vec<usize>>>) -> Box<dyn FnMut(u32) -> Vec<f32>> {
        let mut m = crate::kimi_ref::RefKimi::load(dir).unwrap();
        *m.take.borrow_mut() = take;
        Box::new(move |t| m.step(t))
    }

    /// A hand-written CPU reference's rows on `fx`'s tokens as `drive` runs an engine (the prompt and each decode token; then, from a fresh load, the long
    /// prompt), taking `take`'s experts (each mixture-of-experts block's, in `ref_drive`'s order) in the place of its routers' own: `load` makes one with
    /// the experts of a sequence's tokens and gives its step.
    fn hand_rows(fx: &Fixture, take: &Choices, load: &dyn Fn(Vec<VecDeque<Vec<usize>>>) -> Box<dyn FnMut(u32) -> Vec<f32>>) -> Vec<Vec<f32>> {
        let first: Vec<u32> = fx.prompt.iter().chain(&fx.decode).copied().collect();
        let (mut rows, mut at) = (Vec::new(), 0usize);
        for (tokens, from) in [(&first, fx.prompt.len() - 1), (&fx.long, fx.long.len() - 1)] {
            let mut step = load(take.iter().map(|l| l[at..at + tokens.len()].iter().cloned().collect()).collect());
            for (i, &t) in tokens.iter().enumerate() {
                let l = step(t);
                if i >= from {
                    rows.push(l);
                }
            }
            at += tokens.len();
        }
        rows
    }

    /// A case's hand-written CPU reference (`hand`, taking the expert choices given in the place of its routers') against its description's (`edit`ed:
    /// another rounding point, say) on its draw, on the description's own choices: the verdict at `bound`, its figures and each row's KL printed.
    fn references_on_the_same_choices(
        case: &str,
        make: fn(&Path) -> Result<()>,
        edit: fn(&mut crate::runtime::describe::Described),
        bound: &Bound,
        hand: fn(&Path, Vec<VecDeque<Vec<usize>>>) -> Box<dyn FnMut(u32) -> Vec<f32>>,
    ) -> Verdict {
        let dir = std::env::temp_dir().join(format!("glyd-{case}-refs-{}", std::process::id()));
        make(&dir).unwrap();
        let fx = Fixture::read(&Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/logits/{case}.json.gz"))).unwrap();
        let desc = ref_drive_with(&dir, &fx, None, None, edit).unwrap();
        let take: Choices = desc.routed.iter().map(|l| l.iter().map(|r| r.ids.clone()).collect()).collect();
        let rows = hand_rows(&fx, &take, &|t| hand(&dir, t));
        let _ = std::fs::remove_dir_all(&dir);
        let own = Fixture { rows: desc.rows.iter().map(|r| Row::from_logits(r).unwrap()).collect(), ..fx.clone() };
        let v = compare(&own, bound, &rows);
        println!(
            "{case}: the description's reference against the hand-written one on its choices: KL mean {:.2e} max {:.2e}, best token {}/{}, largest logit difference {:.4}; rows {}",
            v.mean_kl,
            v.max_kl,
            v.top1.0,
            v.top1.1,
            v.max_dlogit,
            own.rows.iter().zip(&rows).map(|(w, g)| format!("{:.1e}", kl_of(w, g))).collect::<Vec<_>>().join(" ")
        );
        v
    }

    /// The noise floor of Kimi-K3's two cases against fp32: the CPU reference of the layer kinds with the rounding points its description names as the
    /// release's (KimiRMSNorm's `Twice` at every norm) against itself with every one of them in fp32 (`ModelDesc::f32_arithmetic`), on the same choices,
    /// as `glm_noise_floor` measures GLM-5.3-Flash's: at most a tenth of the kind's bound (KL mean 1e-5, max 2.5e-5, every best token with a lead), on
    /// each case's draw and three others of its shape (`kimi_ref::random_seeded`), so that the draw is well conditioned and not a lucky one. Kimi's
    /// description rounds every norm and scales no embedding, so the floor test's `stored` moves nothing; this is its draws' floor (why they are drawn
    /// as they are: `kimi_ref::random_with`). Minutes: cargo test --profile ci -p glyd-engine --lib -- --ignored noise_floor_against_fp32.
    #[cfg(feature = "dev")]
    #[test]
    #[ignore]
    fn noise_floor_against_fp32() {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/logits");
        let mut wrong = Vec::new();
        for (id, dsa) in [("kimi", false), ("kimi-dsa", true)] {
            let fx = Fixture::read(&fixtures.join(format!("{id}.json.gz"))).unwrap();
            for k in 0..4u64 {
                let dir = std::env::temp_dir().join(format!("glyd-fp32floor-{id}-{k}-{}", std::process::id()));
                crate::kimi_ref::random_seeded(&dir, dsa, crate::kimi_ref::RANDOM_SEED ^ (k << 32)).unwrap();
                let v = noise_floor_with(&dir, &fx, true, &TENTH, |d| d.desc.f32_arithmetic()).unwrap_or_else(|e| panic!("{id}: {}", e.msg));
                let _ = std::fs::remove_dir_all(&dir);
                println!(
                    "{} {id:9} seed {k}{}: the noise floor against fp32 KL mean {:.2e} max {:.2e}, best token {}/{}, largest logit difference {:.4}",
                    if v.ok { "PASS" } else { "FAIL" },
                    if k == 0 { " (the case's draw)" } else { "" },
                    v.mean_kl,
                    v.max_kl,
                    v.top1.0,
                    v.top1.1,
                    v.max_dlogit
                );
                if !v.ok {
                    wrong.push(format!("{id} seed {k}"));
                }
            }
        }
        assert!(wrong.is_empty(), "past a tenth of the kind's bound: {wrong:?}");
    }

    /// DeepSeek-V4.1's noise floor against fp32, on the reference its case is held to (the layer kinds'): the reference with the rounding points its description
    /// names as the release's (each RMSNorm's output and the streams' collapse and update rounded once: model.py's `.to(dtype)`) against itself with every one in
    /// fp32 (`ModelDesc::f32_arithmetic`), the fp32 run on the rounded run's experts and indexer picks: at most a tenth of the kind's bound (KL mean 1e-5, max
    /// 2.5e-5, every best token with a lead), on the case's draw and three others of its shape (`deepseek_v41_draw`), so that the draw is well conditioned and not
    /// a lucky one. Minutes: cargo test --profile ci -p glyd-engine --lib -- --ignored deepseek_v41_noise_floor.
    #[cfg(feature = "dev")]
    #[test]
    #[ignore]
    fn deepseek_v41_noise_floor() {
        let fx = Fixture::read(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/logits/deepseek-v41.json.gz")).unwrap();
        let mut wrong = Vec::new();
        for k in 0..4u64 {
            let dir = std::env::temp_dir().join(format!("glyd-fp32floor-v41-{k}-{}", std::process::id()));
            deepseek_v41_draw(&dir, DEEPSEEK_V41_SEED ^ (k << 32)).unwrap();
            let v = noise_floor_with(&dir, &fx, true, &TENTH, |d| d.desc.f32_arithmetic()).unwrap_or_else(|e| panic!("{}", e.msg));
            let base = ref_drive(&dir, &fx, None, None).unwrap();
            let _ = std::fs::remove_dir_all(&dir);
            let near = base.routed.iter().flatten().map(|r| r.gap).fold(f32::INFINITY, f32::min);
            let near_pick = base.picked.iter().flatten().map(|r| r.gap).fold(f32::INFINITY, f32::min);
            println!(
                "{} deepseek-v41 seed {k}{}: the noise floor against fp32 KL mean {:.2e} max {:.2e}, best token {}/{}, largest logit difference {:.4}; the nearest router choice {near:.1e}, indexer {near_pick:.1e}",
                if v.ok { "PASS" } else { "FAIL" },
                if k == 0 { " (the case's draw)" } else { "" },
                v.mean_kl,
                v.max_kl,
                v.top1.0,
                v.top1.1,
                v.max_dlogit
            );
            if !v.ok {
                wrong.push(format!("seed {k}"));
            }
        }
        assert!(wrong.is_empty(), "past a tenth of the kind's bound: {wrong:?}");
    }

    /// DeepSeek-V4's noise floor on the reference its case is held to (src/dsv4_ref.rs): the reference with the release's bf16 rounding points against itself
    /// with every one of them in fp32 (`dsv4_ref::PLAIN`), on the same expert choices and indexer picks, without its simulated quantizations (the case's held
    /// part: at most a tenth of the kind's bound, as the other cases' floors) and with them (`dsv4_also`'s third part, printed there and not held: it is past the
    /// bound). On the Mac, 2026-11-25: without them KL mean 7.42e-6, max 1.06e-5 (three more draws, 7.0e-6 to 8.3e-6 and 9.3e-6 to 1.2e-5 at most), with them
    /// 7.75e-4 and 1.04e-3. A rounding before a quantizer moves a value near a grid point's edge to the next one (an E4M3 step is 6 to 12% of the value, FP4's
    /// more), and such values are a fixed share of every activation whatever its scale. The draw's experts were 1.8 times the other matrices' scale before
    /// (`dsv4_ref::random`): the floor without the quantizations was 1.11e-5 and 1.76e-5 (four draws 1.08e-5 to 1.26e-5, one max at 2.62e-5), past a tenth
    /// of the bound whatever the draw. A minute: cargo test --profile ci -p glyd-engine --lib -- --ignored deepseek_v4_noise_floor.
    #[cfg(feature = "dev")]
    #[test]
    #[ignore]
    fn deepseek_v4_noise_floor() {
        let dir = std::env::temp_dir().join(format!("glyd-dsv4floor-{}", std::process::id()));
        crate::dsv4_ref::random(&dir, None).unwrap();
        let fx = Fixture::read(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/logits/deepseek-v4.json.gz")).unwrap();
        let floor = |sim: bool| -> Verdict {
            let base = dsv4_drive_with(&dir, &fx, None, None, sim).unwrap();
            let take: Choices = base.routed.iter().map(|l| l.iter().map(|r| r.ids.clone()).collect()).collect();
            let picks: Picks = base
                .picked
                .iter()
                .map(|l| {
                    l.iter().map(|r| PickRow { pos: 0, picks: r.picks.clone(), q: Vec::new(), w: Vec::new(), keys: Default::default(), inside: None }).collect()
                })
                .collect();
            crate::dsv4_ref::PLAIN.with(|p| p.set(true));
            let plain = dsv4_drive_with(&dir, &fx, Some(&take), Some(&picks), sim);
            crate::dsv4_ref::PLAIN.with(|p| p.set(false));
            let plain = plain.unwrap();
            let own = Fixture { rows: base.rows.iter().map(|r| Row::from_logits(r).unwrap()).collect(), ..fx.clone() };
            let v = compare(&own, &TENTH, &plain.rows);
            let near = base.routed.iter().flatten().map(|r| r.gap).fold(f32::INFINITY, f32::min);
            let near_pick = base.picked.iter().flatten().map(|r| r.gap).fold(f32::INFINITY, f32::min);
            println!(
                "deepseek-v4: the noise floor of src/dsv4_ref.rs against fp32, {}: KL mean {:.2e} max {:.2e}, best token {}/{}, largest logit difference {:.4}; rows {}; the nearest router choice {near:.1e}, indexer {near_pick:.1e}",
                if sim { "its quantizations simulated" } else { "without its simulated quantizations" },
                v.mean_kl,
                v.max_kl,
                v.top1.0,
                v.top1.1,
                v.max_dlogit,
                own.rows.iter().zip(&plain.rows).map(|(w, g)| format!("{:.1e}", kl_of(w, g))).collect::<Vec<_>>().join(" ")
            );
            v
        };
        let (with, without) = (floor(true), floor(false));
        let _ = std::fs::remove_dir_all(&dir);
        assert!(without.ok, "without the simulation, past a tenth of the kind's bound: {:?}", without.notes);
        assert!(with.mean_kl > 1e-4, "with the simulation, within the kind's bound: `dsv4_also` is to hold it then ({:.2e})", with.mean_kl);
    }

    /// Each Gemma 4 random case is well conditioned for the kind's bound, and its fixture is what the CPU reference makes. For each case: its attention
    /// scores' scale (each layer's queries against the prompt's keys, before the rope), and the CPU reference against itself with storage roundings the
    /// engine makes and the reference does not (each norm's output and the embedding's scale bf16), within the kind's bound; the draw the plain case
    /// had before (the q and k norms' weights in [0.5, 1.5], the embedding in [-1, 1]) is past it. Then the reference against the case's fixture, at its
    /// bound. A minute on an optimized build: cargo test --profile ci -p glyd-engine --lib -- --ignored gemma4_reference_and_its_draws.
    #[cfg(feature = "dev")]
    #[test]
    #[ignore]
    fn gemma4_reference_and_its_draws() {
        use crate::runtime::describe::Described;
        use crate::runtime::kinds::{Body, Mixer};
        use crate::runtime::reference::{dot, norm::norm};
        let kind = Bound::Kl { mean: 1e-4, max: 2.5e-4, top1: 0.85 };
        let (prompt, decode, long) = standard_tokens(4096);
        let rows_of = |dir: &Path, edit: fn(&mut Described)| drive(described_cpu_with(dir, edit).unwrap().as_mut(), &prompt, &decode, &long).unwrap();
        let as_fx = |rows: &[Vec<f32>]| Fixture {
            rows: rows.iter().map(|r| Row::from_logits(r).unwrap()).collect(),
            prompt: prompt.clone(),
            decode: decode.clone(),
            long: long.clone(),
            ..fixture(vec![vec![0.0; 2]])
        };
        // (each attention layer's scores over the prompt, before the rope: their RMS and the largest)
        let scores = |dir: &Path| -> String {
            let mut m = ref_model(dir).unwrap();
            m.record = Some(Vec::new());
            m.forward(&prompt, 0).unwrap();
            let rec = m.record.take().unwrap();
            let d = m.desc.hidden;
            let x0: Vec<f32> =
                prompt.iter().flat_map(|&t| m.embed[t as usize * d..][..d].iter().map(|&x| x * m.desc.embed.scale).collect::<Vec<_>>()).collect();
            let (mut out, mut keys) = (Vec::new(), BTreeMap::new());
            for l in 0..rec.len() {
                let la = &m.layers[2 * l];
                let Body::Mixer(Mixer::Attention(a)) = &la.block.body else { continue };
                let (x, hd) = (if l == 0 { &x0 } else { &rec[l - 1] }, a.head_dim);
                let heads = |role: &str, w: &str, n: &Option<crate::runtime::kinds::Norm>, xt: &[f32]| -> Vec<f32> {
                    la.p.prod(role, &la.norm_in(xt)).unwrap().chunks(hd).flat_map(|c| norm(n.as_ref().unwrap(), c, la.p.opt(w), None)).collect()
                };
                let qs: Vec<Vec<f32>> = x.chunks(d).map(|xt| heads("q", "q_norm", &a.q_norm, xt)).collect();
                // (a layer that shares an earlier one's keys: that layer's)
                if a.kv_from.is_none() {
                    keys.insert(2 * l, x.chunks(d).map(|xt| heads("k", "k_norm", &a.k_norm, xt)).collect::<Vec<_>>());
                }
                let ks = &keys[&a.kv_from.unwrap_or(2 * l)];
                let (mut s2, mut n, mut big) = (0f64, 0usize, 0f32);
                for i in 0..qs.len() {
                    for k in &ks[..=i] {
                        for h in 0..a.heads {
                            let s = dot(&qs[i][h * hd..][..hd], &k[h / (a.heads / a.kv_heads) * hd..][..hd]);
                            (s2, n, big) = (s2 + (s as f64).powi(2), n + 1, big.max(s.abs()));
                        }
                    }
                }
                out.push(format!("layer {l} (heads of {hd}) RMS {:.2}, largest {:.1}", (s2 / n as f64).sqrt(), big));
            }
            out.join("; ")
        };
        let conditioned = |what: &str, dir: &Path| -> Verdict {
            let v = compare(&as_fx(&rows_of(dir, |_| {})), &kind, &rows_of(dir, stored));
            println!("{what}: the scores {}", scores(dir));
            println!(
                "{what}: the reference against itself with the storage roundings: KL mean {:.2e} max {:.2e}, best token {}/{}, largest logit difference {:.4}",
                v.mean_kl, v.max_kl, v.top1.0, v.top1.1, v.max_dlogit
            );
            v
        };
        let tmp = |tag: &str| std::env::temp_dir().join(format!("glyd-g4draw-{tag}-{}", std::process::id()));
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/logits");
        for (case, make) in [("gemma4", gemma4_random as fn(&Path) -> Result<()>), ("gemma4-kv", gemma4_kv_random)] {
            let dir = tmp(case);
            make(&dir).unwrap();
            let v = conditioned(case, &dir);
            let fx = Fixture::read(&fixtures.join(format!("{case}.json.gz"))).unwrap();
            let r = compare(&fx, &fx.bounds[0].bound, &drive(described_cpu(&dir).unwrap().as_mut(), &fx.prompt, &fx.decode, &fx.long).unwrap());
            std::fs::remove_dir_all(&dir).ok();
            println!("{case}: the reference against {case}.json.gz: bits-equal {}/{}, KL mean {:.2e} max {:.2e}", r.identical, r.rows, r.mean_kl, r.max_kl);
            assert!(v.ok, "{case}: {:?}", v.notes);
            assert!(r.ok && fx.gpu.is_none(), "{case}: {:?}", r.notes);
        }
        let dir = tmp("before");
        gemma4_drawn(&dir, GEMMA4_RANDOM_CONFIG, 0x2026_1120_0000_0004, (0.5, 1.5), Some(1.0)).unwrap();
        let v = conditioned("gemma4 drawn as before", &dir);
        std::fs::remove_dir_all(&dir).ok();
        assert!(!v.ok, "the earlier draw is within the bound");
    }

    /// GLM-5.3-Flash's random case on its CPU reference with the residual rounded to bf16 before each block (the executor holds its streams bf16),
    /// against the reference as it is: how far that rounding alone moves the case's logits, for the case's model and five of other seeds, the
    /// indexer as drawn and keeping every key; each row with the nearest choice (an indexer's picks, a router's experts) of the tokens it ran.
    /// Minutes on an unoptimized build: cargo test -- --ignored glm_reference_with. What it showed (2026-11-25): KL mean 5.3e-3 (max 2.6e-2) for
    /// the case's model and 4.1e-4 to 7.0e-3 for the others; with every key kept 1.2e-5 for two seeds and 2.1e-4 to 7.0e-3 for four, whose rows
    /// moved where a router's choice at or before them was within 2e-3 of another. Rounding alone takes such choices the other way, so the case's
    /// engine is held to this reference on the engine's own choices, not on the reference's.
    #[cfg(feature = "dev")]
    #[test]
    #[ignore]
    fn glm_reference_with_its_residual_rounded() {
        use crate::runtime::kinds::{Body, Mixer};
        use crate::runtime::reference::{bf16, model::checkpoint_tensor, model::RefModel};
        fn step(m: &mut RefModel, pos: &mut u32, t: u32, round: bool, near: &mut f32) -> Vec<f32> {
            let l = m.forward_seen(&[t], *pos, &mut |x| x.iter_mut().filter(|_| round).for_each(|v| *v = bf16(*v))).unwrap().pop().unwrap();
            *pos += 1;
            *near = m.layers.iter().fold(*near, |a, l| a.min(l.margin));
            l
        }
        let (prompt, decode, long) = standard_tokens(2048);
        for k in 0..6u64 {
            let dir = std::env::temp_dir().join(format!("glyd-glmref-{k}-{}", std::process::id()));
            crate::glm::random_seeded(&dir, false, None, crate::glm::RANDOM_SEED ^ (k << 32)).unwrap();
            let cfg: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
            let ck = glyd_gpu::safetensors::Checkpoint::open(&dir).unwrap();
            for every in [false, true] {
                let mut d = crate::runtime::describe::describe(&cfg).unwrap();
                for b in d.desc.blocks.iter_mut() {
                    if let (true, Body::Mixer(Mixer::Mla(m))) = (every, &mut b.body) {
                        m.indexer.iter_mut().for_each(|ix| ix.top_k = 1 << 20);
                    }
                }
                // (as `drive`: the prompt, each decode token, then from a fresh model the long prompt; each pass's last logits and its nearest choice)
                let run = |round: bool| -> Vec<(Vec<f32>, f32)> {
                    let load = || RefModel::load(&d, &|n: &str| checkpoint_tensor(&ck, n)).unwrap();
                    let (mut m, mut pos, mut near, mut out) = (load(), 0u32, f32::INFINITY, Vec::new());
                    let mut last = Vec::new();
                    for &t in &prompt {
                        last = step(&mut m, &mut pos, t, round, &mut near);
                    }
                    out.push((last, std::mem::replace(&mut near, f32::INFINITY)));
                    for &t in &decode {
                        let l = step(&mut m, &mut pos, t, round, &mut near);
                        out.push((l, std::mem::replace(&mut near, f32::INFINITY)));
                    }
                    (m, pos) = (load(), 0);
                    let mut last = Vec::new();
                    for &t in &long {
                        last = step(&mut m, &mut pos, t, round, &mut near);
                    }
                    out.push((last, near));
                    out
                };
                let (a, b) = (run(false), run(true));
                let kls: Vec<f64> = a.iter().zip(&b).map(|((w, _), (g, _))| kl_of(&Row::from_logits(w).unwrap(), g)).collect();
                let best = |r: &[f32]| (0..r.len()).max_by(|&i, &j| r[i].total_cmp(&r[j])).unwrap();
                let same = a.iter().zip(&b).filter(|((w, _), (g, _))| best(w) == best(g)).count();
                let rows: Vec<String> = kls.iter().zip(&a).map(|(k, (_, n))| format!("{k:.1e}[{n:.0e}]")).collect();
                println!(
                    "glm seed {k}, the indexer {}: KL mean {:.2e} max {:.2e}, best token the same at {same}/{}; rows KL[nearest choice]: {}",
                    if every { "keeping every key" } else { "as drawn" },
                    kls.iter().sum::<f64>() / kls.len() as f64,
                    kls.iter().cloned().fold(0f64, f64::max),
                    a.len(),
                    rows.join(" ")
                );
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// GLM-5.3-Flash's noise floor: the CPU reference against itself with the rounding points the release's bf16 modeling has and the engine keeps
    /// (the description's: the streams' `Twice`, the RMS norms' `Twice`) and with every one of them in fp32 (`ModelDesc::f32_arithmetic`), on the same
    /// choices (the rounded run's experts and picks taken by the other), for the case's model and five of other seeds. Each must be at most a tenth of the
    /// lossless bound (KL mean 1e-5 against 1e-4): the draw is calibrated by the reference alone, never by an engine's output, so the bound stays a fixed
    /// test of an engine and a fault must move the logits well past the rounding noise to pass (`glm::random`). Seconds on an optimized build: cargo test
    /// --profile ci -- --ignored glm_noise_floor.
    #[cfg(feature = "dev")]
    #[test]
    #[ignore]
    fn glm_noise_floor() {
        let mut worst = 0f64;
        for k in 0..6u64 {
            let dir = std::env::temp_dir().join(format!("glyd-glmfloor-{k}-{}", std::process::id()));
            crate::glm::random_seeded(&dir, false, None, crate::glm::RANDOM_SEED ^ (k << 32)).unwrap();
            worst = worst.max(rounded_against_f32(&dir, &format!("glm seed {k}")));
            let _ = std::fs::remove_dir_all(&dir);
        }
        assert!(worst <= 1e-5, "a noise floor of KL {worst:.2e}, past a tenth of the bound");
    }

    /// GLM-5.3's layout (the glm-dsa and glm-dsa-f8 cases): its noise floor as `glm_noise_floor` measures GLM-5.3-Flash's (the rounding points its description
    /// gives, the RMS norms' `Twice`, against every one in fp32, on the same experts and picks), on the case's model and two of other seeds, in bf16 and in
    /// FP8 as released, each at most a tenth of the lossless bound. cargo test --profile ci -- --ignored glm_dsa_noise_floor.
    #[cfg(feature = "dev")]
    #[test]
    #[ignore]
    fn glm_dsa_noise_floor() {
        let mut worst = 0f64;
        // (and the same draws in FP8 as released, the glm-dsa-f8 case's)
        for (k, fp8) in (0..3u64).flat_map(|k| [(k, false), (k, true)]) {
            let dir = std::env::temp_dir().join(format!("glyd-glmdsafloor-{k}-{fp8}-{}", std::process::id()));
            glm_dsa_draw(&dir, GLM_DSA_SEED ^ (k << 32), fp8, false).unwrap();
            worst = worst.max(rounded_against_f32(&dir, &format!("glm-dsa{} seed {k}", if fp8 { "-f8" } else { "" })));
            let _ = std::fs::remove_dir_all(&dir);
        }
        assert!(worst <= 1e-5, "a noise floor of KL {worst:.2e}, past a tenth of the bound");
    }

    /// The CPU reference of the model in `dir` with its description's rounding points against itself with every one in fp32 (`ModelDesc::f32_arithmetic`),
    /// the f32 run taking the rounded run's experts and indexer picks, on the standard tokens (each sequence a token at a time from a fresh model): the KL
    /// mean over the rows, printed with its max (as `what`).
    #[cfg(feature = "dev")]
    fn rounded_against_f32(dir: &Path, what: &str) -> f64 {
        use crate::runtime::kinds::{Body, Ffn, Mixer};
        use crate::runtime::reference::{mla::LatentCache, model::checkpoint_tensor, model::RefModel, SeqState};
        let cfg: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
        let (prompt, decode, long) = standard_tokens(crate::runtime::describe::describe(&cfg).unwrap().desc.vocab);
        let ck = glyd_gpu::safetensors::Checkpoint::open(dir).unwrap();
        // (each sequence a token at a time from a fresh model: its rows, and every block's choices in the order taken)
        let run = |rounded: bool, take: Option<&Vec<Vec<Vec<usize>>>>| -> (Vec<Vec<f32>>, Vec<Vec<Vec<usize>>>) {
            let mut d = crate::runtime::describe::describe(&cfg).unwrap();
            if !rounded {
                d.desc.f32_arithmetic();
            }
            let (mut rows, mut chose) = (Vec::new(), Vec::new());
            for (i, seq) in [[prompt.clone(), decode.clone()].concat(), long.clone()].iter().enumerate() {
                let mut m = RefModel::load(&d, &|n: &str| checkpoint_tensor(&ck, n)).unwrap();
                let at = chose.first().map_or(0, |c: &Vec<Vec<usize>>| c.len());
                chose.resize_with(m.layers.len(), Vec::new);
                for (b, l) in m.layers.iter_mut().enumerate() {
                    let given: std::collections::VecDeque<Vec<usize>> = take.map(|t| t[b][at..at + seq.len()].iter().cloned().collect()).unwrap_or_default();
                    match &l.block.body {
                        Body::Ffn(Ffn::Moe(_)) => (l.take, l.routed) = (given, Some(Vec::new())),
                        Body::Mixer(Mixer::Mla(_)) => {
                            l.seqs.insert(0, SeqState::Mla(LatentCache { take: given, picked: Some(Vec::new()), ..Default::default() }));
                        }
                        _ => {}
                    }
                }
                for (t, &tok) in seq.iter().enumerate() {
                    let r = m.forward(&[tok], t as u32).unwrap().pop().unwrap();
                    if (i == 0 && t + 1 >= prompt.len()) || (i == 1 && t + 1 == seq.len()) {
                        rows.push(r);
                    }
                }
                for (b, l) in m.layers.iter_mut().enumerate() {
                    let got: Vec<Vec<usize>> = match (&l.block.body, l.seqs.get_mut(&0)) {
                        (Body::Ffn(Ffn::Moe(_)), _) => l.routed.take().unwrap_or_default().into_iter().map(|r| r.ids).collect(),
                        (_, Some(SeqState::Mla(c))) => c.picked.take().unwrap_or_default().into_iter().map(|p| p.picks).collect(),
                        _ => vec![Vec::new(); seq.len()],
                    };
                    // (a layer whose indexer is shared records no choice of its own: it takes the one before it, whatever is given)
                    chose[b].extend(if got.len() == seq.len() { got } else { vec![Vec::new(); seq.len()] });
                }
            }
            (rows, chose)
        };
        let (a, choices) = run(true, None);
        let (b, _) = run(false, Some(&choices));
        let kls: Vec<f64> = a.iter().zip(&b).map(|(w, g)| kl_of(&Row::from_logits(w).unwrap(), g)).collect();
        let mean = kls.iter().sum::<f64>() / kls.len() as f64;
        let rms = (a.iter().flatten().map(|&x| (x as f64).powi(2)).sum::<f64>() / a.iter().map(Vec::len).sum::<usize>() as f64).sqrt();
        println!(
            "{what}: the noise floor KL mean {mean:.2e} max {:.2e} over {} rows (the logits' RMS {rms:.2})",
            kls.iter().cloned().fold(0f64, f64::max),
            kls.len()
        );
        mean
    }

    /// The reference taking an engine's picks (`ref_drive` with `Picks`) on the glm case's model: its own picks taken give its own rows bit for bit;
    /// another choice of one query is attended to (the record keeps the indexer's own) and moves the rows from that token's on; picks of another
    /// count of layers are refused.
    #[cfg(feature = "dev")]
    #[test]
    fn the_reference_takes_an_engines_picks() {
        let dir = std::env::temp_dir().join(format!("glyd-glmpicks-{}", std::process::id()));
        crate::glm::random(&dir, false, None).unwrap();
        let (prompt, decode, long) = standard_tokens(2048);
        let fx = Fixture {
            case: "glm".into(),
            family: String::new(),
            model: String::new(),
            provenance: BTreeMap::new(),
            gpu: None,
            bounds: Vec::new(),
            prompt,
            decode,
            long,
            rows: Vec::new(),
        };
        let own = ref_drive(&dir, &fx, None, None).unwrap();
        let as_engine = |f: &dyn Fn(usize, usize, &crate::runtime::reference::mla::Picked) -> Vec<usize>| -> Picks {
            own.picked
                .iter()
                .enumerate()
                .map(|(l, v)| {
                    v.iter()
                        .enumerate()
                        .map(|(t, r)| PickRow { pos: 0, picks: f(l, t, r), q: Vec::new(), w: Vec::new(), keys: Default::default(), inside: None })
                        .collect()
                })
                .collect()
        };
        assert_eq!(own.indexers.len(), 2);
        let same = ref_drive(&dir, &fx, None, Some(&as_engine(&|_, _, r| r.own.clone()))).unwrap();
        assert_eq!(same.rows, own.rows, "its own picks taken");
        // (the first query of the prompt that chose: its weakest pool kept swapped for the best left out)
        let (l, t) = (0..2).flat_map(|l| (0..24).map(move |t| (l, t))).find(|&(l, t)| !own.picked[l][t].score.is_empty()).unwrap();
        let r = &own.picked[l][t];
        let (b, sc) = (r.block, &r.score);
        let mut kept: Vec<usize> = (0..sc.len()).filter(|&j| r.own.contains(&(j * b))).collect();
        let low = *kept.iter().min_by(|&&x, &&y| sc[x].total_cmp(&sc[y])).unwrap();
        let best = (0..sc.len()).filter(|j| !kept.contains(j)).max_by(|&x, &y| sc[x].total_cmp(&sc[y])).unwrap();
        kept.retain(|&j| j != low);
        kept.push(best);
        kept.sort_unstable();
        let mut other: Vec<usize> = kept.iter().flat_map(|&j| j * b..j * b + b).collect();
        other.extend(r.own.iter().copied().filter(|&e| e >= sc.len() * b));
        let moved = ref_drive(&dir, &fx, None, Some(&as_engine(&|ll, tt, x| if (ll, tt) == (l, t) { other.clone() } else { x.own.clone() }))).unwrap();
        assert_eq!((&moved.picked[l][t].picks, &moved.picked[l][t].own), (&other, &r.own));
        assert_ne!(moved.rows[0], own.rows[0], "the prompt's row after another choice");
        assert_eq!(moved.rows[14 - 1], own.rows[14 - 1], "the long prompt, from a reset, its own");
        assert!(ref_drive(&dir, &fx, None, Some(&Vec::new())).is_err(), "picks of no layer");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An engine's picks apart from the reference's own, each judged by the reference's indexer on the engine's own inputs: a choice that is the
    /// reference's there has margin 0 (whatever the reference's own inputs gave), a near tie there its small margin, a choice no score there explains
    /// a margin past `NEAR`; the margin on the reference's own inputs is kept beside it.
    #[test]
    fn picks_apart_are_judged_on_the_engines_inputs() {
        use crate::runtime::kinds::{IndexForm, Indexer};
        use crate::runtime::reference::mla::{choose, Picked};
        let ix = Indexer { heads: 2, head_dim: 4, top_k: 2, block: 1, shared: false, form: IndexForm::LAYER_KEYS, rope: 0 };
        // (a key a token, scores rising with its first channel: tokens 3 and 4 the best two of 0..=4, token 2 near token 3 where `near`)
        let row = |near: bool, picks: Vec<usize>| {
            let s = [0.1f32, 0.2, if near { 0.6999 } else { 0.3 }, 0.7, 0.9];
            let keys: Vec<Vec<f32>> = s.iter().map(|&x| vec![x, 0.0, 0.0, 0.0]).collect();
            PickRow { pos: 4, picks, q: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], w: vec![1.0, 1.0], keys: std::rc::Rc::new(keys), inside: None }
        };
        let seen: Vec<usize> = (0..5).collect();
        let own = |keys: &[Vec<f32>]| choose(&ix, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], &[1.0, 1.0], keys, &seen);
        // (the reference's own inputs: token 2 well below token 3; its own choice 3 and 4)
        let refk: Vec<Vec<f32>> = [0.1f32, 0.2, 0.3, 0.7, 0.9].iter().map(|&x| vec![x, 0.0, 0.0, 0.0]).collect();
        let r: Picked = own(&refk);
        assert_eq!(r.own, vec![3, 4]);
        let cases = [(row(true, vec![3, 4]), None), (row(true, vec![2, 4]), Some(false)), (row(false, vec![2, 4]), Some(true))];
        for (e, far) in cases {
            let taken = Picked { picks: e.picks.clone(), ..r.clone() };
            let got = picks_apart(&[vec![taken]], &vec![vec![e]], std::slice::from_ref(&ix));
            match far {
                None => assert!(got.is_empty(), "the reference's own choice"),
                Some(far) => {
                    assert_eq!((got.len(), got[0].reference.clone(), got[0].engine.clone()), (1, vec![3], vec![2]));
                    assert!(got[0].own_margin > 0.3, "{}", got[0].own_margin);
                    assert_eq!(got[0].margin >= crate::dynamic::kindcheck::NEAR, far, "{}", got[0].margin);
                }
            }
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

    /// logitscheck counts apart what a reference held and what a pin held. A case with pins alone is said to be pinned only; one with nothing compared is a failure
    /// where its fixtures are all another GPU generation's bits, and is named where its engine is not built: none of the three reads as a case within its bounds.
    #[test]
    fn a_case_no_reference_held_is_said_and_one_with_nothing_compared_fails() {
        let fx = |gpu: Option<&str>| Fixture { gpu: gpu.map(str::to_string), ..fixture(vec![logits(100, 1), logits(100, 2)]) };
        assert_eq!((fx(None).held(), fx(Some("sm89")).held()), (Held::Reference, Held::Pin));
        assert_eq!((Held::Reference.passed(), Held::Pin.passed()), ("PASS", "PIN "));
        let mut t = Tally::default();
        // a case a reference held, with a pin beside it: nothing to say
        for h in [Held::Reference, Held::Reference, Held::Pin] {
            t.add(h);
        }
        assert!(t.case("qwen35", 2, 1, false, &[]).is_none());
        // a case with pins alone
        t.add(Held::Pin);
        let line = t.case("gemma4", 0, 1, false, &[]).unwrap();
        assert!(line.starts_with("PIN  gemma4") && line.contains("pinned only") && !line.contains("within"), "{line}");
        // a case whose fixtures are another generation's bits, one whose engine is not built, one with no fixture at this level
        let line = t.case("glm", 0, 0, false, &["sm89".to_string()]).unwrap();
        assert!(line.starts_with("FAIL glm") && line.contains("nothing compared") && line.contains("sm89"), "{line}");
        assert!(t.case("a-later-family", 0, 0, true, &[]).is_none() && t.case("dense", 0, 0, false, &[]).is_none());
        assert_eq!(t.failed(), vec!["glm: nothing compared (its fixtures are the bits of sm89)".to_string()]);
        assert_eq!(
            t.summary(),
            "2 held to a reference, 2 pinned, 2 not held; pinned only (no reference): gemma4; not held: glm; not held, no engine yet: a-later-family"
        );
        assert_eq!(Tally::default().summary(), "0 held to a reference, 0 pinned, 0 not held");
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

    /// A random model of a case in a directory of this test's, removed when the guard goes.
    struct Made(PathBuf);
    impl Made {
        fn new(case: &str, tag: &str) -> Made {
            let d = std::env::temp_dir().join(format!("glyd-logits-test-{tag}-{case}-{}", std::process::id()));
            random_model(case, &d).unwrap();
            Made(d)
        }
    }
    impl Drop for Made {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A case's fixture with `prompt` prompt tokens, `decode` decode tokens and no long prompt: a short run of its tokens.
    fn short(case: &str, prompt: usize, decode: usize) -> Fixture {
        let f = Fixture::read(&Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/logits/{case}.json.gz"))).unwrap();
        Fixture { prompt: f.prompt[..prompt].to_vec(), decode: f.decode[..decode].to_vec(), long: Vec::new(), rows: Vec::new(), ..f }
    }

    /// An engine's calls (`dynamic::Routes`) become each layer's tokens in the order run, each token's experts beside its router's input row
    /// (widened from bf16), whatever the calls' sizes; a call whose input is not a whole number of rows is refused.
    #[test]
    fn an_engine_s_calls_are_each_token_s_choice_and_router_input() {
        let bf = |x: f32| (x.to_bits() >> 16) as u16;
        let calls = vec![
            (1, 2, vec![3, 1, 0, 2], vec![bf(1.0), bf(2.0), bf(3.0), bf(4.0)]),
            (0, 2, vec![5, 4], vec![bf(-1.0), bf(0.5)]),
            (1, 2, vec![2, 3], vec![bf(8.0), bf(-8.0)]),
        ];
        let r = choices_of(calls).unwrap();
        assert_eq!(r.choices, vec![vec![vec![5, 4]], vec![vec![3, 1], vec![0, 2], vec![2, 3]]]);
        assert_eq!(r.inputs, vec![vec![vec![-1.0, 0.5]], vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![8.0, -8.0]]]);
        assert!(choices_of(vec![(0, 2, vec![1, 2, 3, 4], vec![bf(1.0); 3])]).is_err());
        assert!(choices_of(vec![(0, 2, vec![1, -1], vec![bf(1.0); 2])]).is_err());
    }

    /// The reference of the experts kind taking choices: its own choices give its own rows, bit for bit, and none apart; the nearest other choice
    /// at one token of one layer leaves the rows before that token's as they were, moves that row, and is the first choice apart, by that
    /// choice's own margin. And what it leaves in the rows after it is carried by states (`carry_lines` runs).
    #[test]
    fn the_reference_takes_an_engine_s_choices() {
        let (m, fx) = (Made::new("qwen4x", "choices"), short("qwen4x", 6, 3));
        let own = ref_drive(&m.0, &fx, None, None).unwrap();
        assert_eq!((own.rows.len(), own.routed.len(), own.routed[0].len()), (4, 8, 9));
        let choices: Choices = own.routed.iter().map(|l| l.iter().map(|r| r.own.clone()).collect()).collect();
        // (the engine's order within a token is not the router's: the same experts are the same choice)
        let turned: Choices = choices.iter().map(|l| l.iter().map(|c| c.iter().rev().copied().collect()).collect()).collect();
        let same = ref_drive(&m.0, &fx, Some(&turned), None).unwrap();
        assert!(same.rows == own.rows && apart(&same, &turned).is_empty());
        let d = other_choice(&fx, &own, 2, 3).unwrap();
        assert_eq!(d.token, 7);
        let mut other = choices.clone();
        other[3][7] = choices[3][7].iter().map(|&e| if e == d.reference[0] { d.engine[0] } else { e }).collect();
        let moved = ref_drive(&m.0, &fx, Some(&other), None).unwrap();
        assert!(moved.rows[..2] == own.rows[..2] && moved.rows[2] != own.rows[2]);
        let got = apart(&moved, &other);
        assert_eq!((got[0].token, got[0].layer, &got[0].reference, &got[0].engine), (7, 3, &d.reference, &d.engine));
        assert_eq!(got[0].margin, own.routed[3][7].gap);
        assert!(got[0].margin > 0.0 && got.iter().all(|g| (g.token, g.layer) >= (7, 3)));
        // (a choice of another size is refused)
        other[3][7].pop();
        assert!(ref_drive(&m.0, &fx, Some(&other), None).is_err());
        carry_lines(&m.0, &fx, &own, &d).unwrap();
    }

    /// The reference on a fixture's tokens with the delta rule in its chunked form (`reference::linear::chunked`): the prompts a pass each, cut in
    /// chunks as the upstream prompt kernels cut them, the decode tokens on the recurrence (`form` None: every pass on the recurrence).
    fn chunked_rows(dir: &Path, fx: &Fixture, form: Option<crate::runtime::reference::linear::chunked::Rounded>) -> Vec<Vec<f32>> {
        use crate::runtime::kinds::{Body, Mixer};
        use crate::runtime::reference::linear::chunked;
        use crate::runtime::reference::model::RefModel;
        use crate::runtime::reference::SeqState;
        let pass = |m: &mut RefModel, tokens: &[u32], at: usize, form| -> Vec<f32> {
            chunked::FORM.set(form);
            let l = m.forward(tokens, at as u32);
            chunked::FORM.set(None);
            for b in &mut m.layers {
                if let Body::Mixer(Mixer::Linear(lin)) = &b.block.body {
                    for st in b.seqs.values_mut() {
                        if let SeqState::Linear(s) = st {
                            chunked::end(lin, s);
                        }
                    }
                }
            }
            l.unwrap().pop().unwrap()
        };
        let mut m = ref_model(dir).unwrap();
        let mut rows = vec![pass(&mut m, &fx.prompt, 0, form)];
        for (i, &t) in fx.decode.iter().enumerate() {
            rows.push(pass(&mut m, &[t], fx.prompt.len() + i, None));
        }
        let mut m = ref_model(dir).unwrap();
        rows.push(pass(&mut m, &fx.long, 0, form));
        rows
    }

    /// The delta rule's two forms on the family's random cases, each whole fixture's tokens: the chunked form in fp32 gives the recurrence's logits
    /// within the kind's bound for the lossless level (the Qwen3.5 layout's fixture's; it is the same sums in another order), and a prompt in one
    /// pass is the prompt a token a pass, bit for bit. Printed, and held to nothing: how far the upstream chunk kernels' roundings to bf16 move
    /// the logits from the fp32 recurrence (the matrix A, the rows u and w and the chunk's first state; then every value those kernels keep in bf16).
    #[test]
    fn the_delta_rule_s_chunked_form_on_the_family_s_random_cases() {
        use crate::runtime::reference::linear::chunked::Rounded;
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/logits");
        let kind = Fixture::read(&fixtures.join("qwen35.json.gz")).unwrap().bounds.into_iter().find(|b| b.level == "penguin").unwrap().bound;
        for case in ["qwen4x", "qwen4x-s2"] {
            let m = Made::new(case, "chunked");
            let fx = Fixture::read(&fixtures.join(format!("{case}.json.gz"))).unwrap();
            let own = ref_drive(&m.0, &fx, None, None).unwrap().rows;
            let recurrence = Fixture { rows: own.iter().map(|l| Row::from_logits(l).unwrap()).collect(), ..fx.clone() };
            assert!(chunked_rows(&m.0, &fx, None) == own, "{case}: a prompt in one pass is not the prompt a token a pass");
            for (what, form) in [
                ("fp32", Rounded::None),
                ("A, u, w and the chunk's first state rounded to bf16", Rounded::Four),
                ("every value the upstream chunk kernels keep in bf16 rounded", Rounded::Stored),
            ] {
                let rows = chunked_rows(&m.0, &fx, Some(form));
                let v = compare(&recurrence, &kind, &rows);
                let each: Vec<String> = recurrence.rows.iter().zip(&rows).map(|(r, g)| format!("{:.1e}", kl_of(r, g))).collect();
                println!(
                    "{case}: the chunked form, {what}, from the fp32 recurrence: KL mean {:.2e} max {:.2e}, best token {}/{} with a lead; rows {}",
                    v.mean_kl,
                    v.max_kl,
                    v.top1.0,
                    v.top1.1,
                    each.join(" ")
                );
                assert!(form != Rounded::None || v.ok, "{case}: the chunked form in fp32 is past the kind's bound of the recurrence: {:?}", v.notes);
            }
        }
    }

    /// The Qwen3.5 layout with experts is held at every level to the bound the Qwen3.5 layout's fixture gives that level (the dense layout's): a family has no
    /// bound of its own, and its lossy levels' were set from that fixture before any run of the case at them.
    #[test]
    fn the_qwen35_layout_with_experts_has_the_layouts_bounds() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/logits");
        let (moe, dense) = (Fixture::read(&dir.join("qwen35moe.json.gz")).unwrap(), Fixture::read(&dir.join("qwen35.json.gz")).unwrap());
        let levels = |f: &Fixture| f.bounds.iter().map(|b| (b.level.clone(), format!("{:?}", b.bound))).collect::<Vec<_>>();
        assert_eq!(levels(&moe), levels(&dense));
        assert_eq!(moe.bounds.iter().map(|b| b.level.as_str()).collect::<Vec<_>>(), ["penguin", "kestrel", "swift", "wren"]);
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

    /// A row beside others against the row alone: held bit for bit where every expert is on the GPU, at the kind's bound where the host computes experts. One
    /// last place apart passes the bound and fails the bits; a best token taken the other way at a clear lead fails both, at a near tie only the bits; a logit
    /// that is not a number fails both; and the words say which held and why.
    #[test]
    fn a_row_beside_others_is_held_bit_for_bit_or_at_the_kinds_bound_where_the_host_computes_experts() {
        let mut alone = logits(2000, 3);
        alone[7] = 40.0;
        alone[8] = 30.0;
        let mut ulp = alone.clone();
        ulp[100] = f32::from_bits(ulp[100].to_bits() + 1);
        let b = Beside::of(&[(&alone[..], &ulp[..])]);
        assert!((b.differ, b.rows, b.clear, b.top_same) == (1, 1, 1, 1) && b.kl_max < 1e-9 && b.bound());
        assert!(!b.held(false) && b.held(true));
        assert!(b.words(true).contains("experts computed on the host: bits not held, the kind's bound held") && !b.words(false).contains("host"));
        assert!(Beside::of(&[(&alone[..], &alone[..])]).held(false));
        // (the best token another at a lead of 10: past the bound, and the best token at a clear lead)
        let mut other = alone.clone();
        other[8] = 50.0;
        let b = Beside::of(&[(&alone[..], &other[..])]);
        assert!(!b.held(true) && b.top_same == 0 && b.words(true).contains("the kind's bound NOT held"));
        // (a near tie, 1e-3 apart, taken the other way by a last place: the best token is not asked there, and the KL is within the bound)
        let mut tie = alone.clone();
        tie[8] = 40.0 - 1e-3;
        let mut flipped = tie.clone();
        (flipped[7], flipped[8]) = (tie[8], tie[7]);
        let b = Beside::of(&[(&tie[..], &flipped[..])]);
        assert!(b.clear == 0 && b.top_all == 0 && b.kl_max < BESIDE_KL.1 && b.held(true) && !b.held(false));
        // (KL past the bound with the best token the same, a lead of 1 kept; and a logit that is not a number)
        let mut spread = logits(2000, 3);
        spread[7] = 9.0;
        let mut far = spread.clone();
        for x in far.iter_mut().skip(200).take(400) {
            *x += 0.5;
        }
        let b = Beside::of(&[(&spread[..], &far[..])]);
        assert!(b.top_same == 1 && b.clear == 1 && b.kl_max > BESIDE_KL.1 && !b.held(true), "{}", b.words(true));
        let mut nan = alone.clone();
        nan[5] = f32::NAN;
        let b = Beside::of(&[(&alone[..], &nan[..])]);
        assert!(!b.finite && !b.held(true) && !b.held(false));
    }

    /// The causal rule for rows where the host computes experts: the first discrete choice the two runs made apart in the row's history decides. None apart:
    /// the row is held to the kind's bound; the first apart a near tie on each run's own input: reported, whatever comes after it in the history; not a near
    /// tie: failed, whatever the logits; no history seen and past the bound: failed. An indexer's pick is measured by its pools' scores, a router's by its
    /// input through the description's router.
    #[test]
    fn the_first_choice_apart_in_a_rows_history_decides() {
        use crate::dynamic::{Choice, ChoiceFrom, ChoiceKind, RowRoute};
        use crate::runtime::kinds::{Act, Mlp, Moe, Router, RouterScore};
        let m = Moe {
            router: Router { score: RouterScore::Softmax, logit_scale: 1.0, bias: false, groups: None, top_k: 1, renorm: true, scale: 1.0, hash: false },
            experts: 2,
            expert: Mlp { act: Act::Silu, gated: true, bias: false, clamp: None },
            shared: None,
            shared_gate: false,
            latent: None,
        };
        // (a router of 2 experts on inputs of 2: the logits are the input's two values)
        let mut p = crate::runtime::reference::Params::new();
        p.put("router", &[2, 2], vec![1.0, 0.0, 0.0, 1.0]);
        let routers = Routers([(5usize, (m, p))].into_iter().collect());
        let bf = |a: f32, b: f32| vec![crate::weights::bf16_bits(a), crate::weights::bf16_bits(b)];
        let route =
            |e: i64, x: Vec<u16>| -> RowRoute { std::rc::Rc::new(Choice { layer: 5, kind: ChoiceKind::Route, chosen: vec![e], from: ChoiceFrom::Router(x) }) };
        let pick =
            |q: i64, s: Vec<f32>| -> RowRoute { std::rc::Rc::new(Choice { layer: 5, kind: ChoiceKind::Pick, chosen: vec![q], from: ChoiceFrom::Scores(s) }) };
        // (expert 0 against 1: a near tie at 1.0 / 0.996 either way; 0.0 / 1.0 is none. Pool 0 against 1: 1.0 / 0.999 a near tie, 1.0 / 0.5 none)
        let (same, tie_a, tie_b, far_b) = (route(0, bf(1.0, 0.996)), route(0, bf(1.0, 0.996)), route(1, bf(0.996, 1.0)), route(1, bf(0.0, 1.0)));
        let (pick_a, pick_tie, pick_far) = (pick(0, vec![1.0, 0.999]), pick(1, vec![0.999, 1.0]), pick(1, vec![0.5, 1.0]));
        let h = |v: Vec<Vec<RowRoute>>| Some(v);
        let mut alone = logits(64, 9);
        alone[3] = 20.0;
        let mut off = alone.clone();
        off[3] = 10.0;
        let rows = [
            Judged { alone: &alone, beside: &alone, choices: (h(vec![vec![same.clone()]]), h(vec![vec![same.clone()]])) },
            Judged {
                alone: &alone,
                beside: &off,
                choices: (h(vec![vec![same.clone()], vec![tie_a.clone()]]), h(vec![vec![same.clone()], vec![tie_b.clone()]])),
            },
            Judged { alone: &alone, beside: &off, choices: (h(vec![vec![same.clone()]]), h(vec![vec![same.clone()]])) },
            Judged { alone: &alone, beside: &alone, choices: (h(vec![vec![tie_a.clone()]]), h(vec![vec![far_b.clone()]])) },
            Judged { alone: &alone, beside: &off, choices: (None, None) },
            Judged { alone: &alone, beside: &off, choices: (h(vec![vec![pick_a.clone(), tie_a.clone()]]), h(vec![vec![pick_tie.clone(), far_b.clone()]])) },
            Judged {
                alone: &alone,
                beside: &off,
                choices: (h(vec![vec![pick_a.clone()], vec![tie_a.clone()]]), h(vec![vec![pick_far.clone()], vec![tie_b.clone()]])),
            },
        ];
        let j = Judge::of(&rows, Some(&routers)).unwrap();
        assert_eq!((j.held, j.reported, j.failed), (1, 2, 4), "{:?}", j.notes);
        let said = |i: usize, w: &str| j.notes.iter().any(|n| n.starts_with(&format!("row {i}: {w}")));
        assert!(
            said(1, "reported") && said(5, "reported") && said(2, "FAILED, past the kind's bound with no choice apart") && said(4, "FAILED"),
            "{:?}",
            j.notes
        );
        assert!(said(3, "FAILED, the first choice apart is not a near tie") && said(6, "FAILED, the first choice apart is not a near tie"), "{:?}", j.notes);
        assert!(j
            .notes
            .iter()
            .any(|n| n.starts_with("row 1:") && n.contains("position 1, layer 5, the router's experts: the first run chose [0], the second [1]")));
        assert!(j.notes.iter().any(|n| n.starts_with("row 5:") && n.contains("position 0, layer 5, the indexer's pools")));
        let j = Judge::of(&rows[..2], Some(&routers)).unwrap();
        assert!(j.ok() && (j.held, j.reported, j.failed) == (1, 1, 0) && j.words().contains("1 reported: a near tie taken the other way"));
    }

    /// Replies beside others against alone, row for row while their tokens are the same: the row where a token first differs is compared and said, with alone's
    /// lead there, and none after it.
    #[test]
    fn replies_beside_others_are_compared_row_for_row_until_a_token_parts() {
        let row = |seed: u32, best: usize, lead: f32| {
            let mut l = logits(64, seed);
            l[best] = 30.0;
            l[(best + 1) % 64] = 30.0 - lead;
            l
        };
        let seen = |l: Vec<f32>| (l, None);
        let alone: Vec<Seen> = vec![
            (vec![3, 4, 5], vec![seen(row(1, 3, 1.0)), seen(row(2, 4, 1.0)), seen(row(3, 5, 1.0))]),
            (vec![9, 9], vec![seen(row(4, 9, 1e-3)), seen(row(5, 9, 1.0))]),
        ];
        let mut beside = alone.clone();
        // (reply 1 takes the near tie's other token at its first row; its second row is on other tokens and far from alone's)
        beside[1].0[0] = 10;
        beside[1].1[1] = seen(row(6, 20, 1.0));
        let (rows, parted) = replies_beside(&alone, &beside);
        assert_eq!(rows.len(), 4);
        assert_eq!(parted, vec![(1, 0, lead_of(&alone[1].1[0].0))]);
        let j = Judge::of(&rows, None).unwrap();
        assert!(parted[0].2 < CLEAR_LEAD && j.ok() && (j.held, j.reported, j.failed) == (4, 0, 0), "{}", j.words());
    }
}

/// The case's model in `dir` on the description executor.
pub(crate) fn load_dyn(case: &str, dir: &Path, o: &DynOpts, level: &str) -> Result<Dyn> {
    Ok(match q4_case(case).0 {
        "gemma4" | "gemma4-kv" => {
            let c = crate::gemma4::ConfigG4::load(dir)?;
            Dyn::new(dir, crate::dynamic::gemma4::lowered(dir, &c)?, None, o)?
        }
        "qwen35" | "qwen35-wide" | "qwen35moe" => crate::dynamic::qwen35::load(dir, o)?,
        "dense" => crate::dynamic::dense::load(dir, o)?,
        "muse" => crate::dynamic::muse::load(dir, o)?,
        "kimi" | "kimi-dsa" => crate::dynamic::kimi::load(dir, o)?,
        "glm-dsa" | "glm-dsa-f8" => crate::dynamic::glm::load_plain(dir, o)?,
        // (its experts at a lossy level, as its logits case: kestrel where penguin is asked)
        "glm" => crate::dynamic::glm::load(dir, &DynOpts { layout: Some(Layout::Tiered), ..o.clone() }, level_layout(level)?.unwrap_or(Layout::Int6))?,
        // (its experts at the level asked: the lossless one where none is)
        "nemotron" => crate::dynamic::nemotron::load(dir, o, level_layout(level)?.unwrap_or(Layout::Tiered))?,
        // (DeepSeek-V4 and V4.1: one loader of the compressed attention's lowering)
        "deepseek-v4" | "deepseek-v41" => crate::dynamic::deepseek_v4::load(dir, o)?,
        "qwen4x" | "qwen4x-s2" => crate::dynamic::qwen4x::load(dir, o)?,
        "qwen4x-cb" | "qwen4x-rel" => qwen4x_cb_load(dir, o)?,
        other => {
            return err(format!(
                "{other}: gemma4, gemma4-kv, qwen35, qwen35-wide, qwen35moe, dense, muse, kimi, kimi-dsa, glm, glm-dsa, glm-dsa-f8, nemotron, deepseek-v4, deepseek-v41, qwen4x, qwen4x-s2, qwen4x-cb or qwen4x-rel"
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
    random_dyn_routed(case, level, layout, tag, ctx).map(|x| x.0)
}

/// `random_dyn_at`, and where the engine computes experts on the host (`host_experts`) its routers (`Routers`), read before the model's directory goes.
pub(crate) fn random_dyn_routed(case: &str, level: &str, layout: Option<Layout>, tag: &str, ctx: Option<usize>) -> Result<(Dyn, Option<Routers>)> {
    random_dyn_with(case, level, tag, &DynOpts { ctx: ctx.unwrap_or(1024), ..dyn_opts(layout) })
}

/// `random_dyn_routed` loaded with the options `o` (a plan's context, chunk and layout: `users_check --ctx plan`).
fn random_dyn_with(case: &str, level: &str, tag: &str, o: &DynOpts) -> Result<(Dyn, Option<Routers>)> {
    let dir = std::env::temp_dir().join(format!("glyd-{tag}-{case}-{}", std::process::id()));
    let o = o.clone();
    let e: Result<(Dyn, Option<Routers>)> = (|| {
        random_model(case, &dir)?;
        let e = load_dyn(case, &dir, &o, level)?;
        let r = routers_of(&dir, &e)?;
        Ok((e, r))
    })();
    let _ = std::fs::remove_dir_all(&dir);
    e
}

/// The routers of the model in `dir` where `e` computes experts on the host (`Routers`), else none. Routers that cannot be read are said, and none are
/// given: a row past the bound then fails, as one whose choices no one saw.
fn routers_of(dir: &Path, e: &Dyn) -> Result<Option<Routers>> {
    if !host_experts(e) {
        return Ok(None);
    }
    match Routers::load(dir, e) {
        Ok(r) => Ok(Some(r)),
        Err(x) => {
            println!("the routers of {} could not be read ({}): a row past the kind's bound fails, its choices unjudged", dir.display(), x.msg);
            Ok(None)
        }
    }
}

/// The case's random model written to `dir` (`CASE+q4`: with its tables read by rows on the host stored at 4 bits, `tables_at_4_bits`).
pub(crate) fn random_model(case: &str, dir: &Path) -> Result<()> {
    let (case, q4) = q4_case(case);
    random_model_of(case, dir)?;
    if q4 {
        tables_at_4_bits(dir, &|_| true)?;
    }
    Ok(())
}

fn random_model_of(case: &str, dir: &Path) -> Result<()> {
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
        "glm-dsa" => glm_dsa_random(dir),
        "glm-dsa-f8" => glm_dsa_f8_random(dir),
        "nemotron" => crate::kcheck_nh::random(dir, 0, 16),
        "deepseek-v4" => crate::dsv4_ref::random(dir, None),
        "deepseek-v41" => deepseek_v41_random(dir),
        "qwen4x" => qwen4x_random(dir),
        "qwen4x-s2" => qwen4x_s2_random(dir),
        "qwen4x-cb" => qwen4x_cb_random(dir),
        "qwen4x-rel" => qwen4x_release_random(dir),
        other => err(format!(
            "{other}: gemma4, gemma4-kv, qwen35, qwen35-wide, qwen35moe, dense, muse, kimi, kimi-dsa, glm, glm-dsa, glm-dsa-f8, nemotron, deepseek-v4, deepseek-v41, qwen4x, qwen4x-s2, qwen4x-cb or qwen4x-rel (each + q4: its tables on the host at 4 bits)"
        )),
    }
}

/// The draft heads' cases' random models with their heads (`draftcheck`'s): the case's model to the bit and its MTP layer or DSpark drafter.
fn draft_model(case: &str, dir: &Path) -> Result<()> {
    match case {
        "qwen4x" => qwen4x_mtp_random(dir),
        "glm-dsa" => glm_dsa_mtp_random(dir, false),
        "glm-dsa-f8" => glm_dsa_mtp_random(dir, true),
        "deepseek-v4" => deepseek_v4_dspark_random(dir),
        "deepseek-v41" => deepseek_v41_random(dir),
        other => err(format!("{other}: the draft heads' cases are qwen4x, glm-dsa, glm-dsa-f8, deepseek-v4 and deepseek-v41")),
    }
}

/// The model of a draft head's case in `dir` drafting with its head `k` tokens a round, by its family's loader.
fn load_drafting_dyn(case: &str, dir: &Path, o: &DynOpts, k: usize) -> Result<Dyn> {
    match case {
        "qwen4x" => crate::dynamic::qwen4x::load_drafting(dir, o, crate::dynamic::qwen4x::experts_at(o.layout), k),
        "glm-dsa" | "glm-dsa-f8" => crate::dynamic::glm::load_plain_drafting(dir, o, k),
        "deepseek-v4" | "deepseek-v41" => crate::dynamic::deepseek_v4::load_drafting(dir, o, k, None),
        other => err(format!("{other}: the draft heads' cases are qwen4x, glm-dsa, glm-dsa-f8, deepseek-v4 and deepseek-v41")),
    }
}

/// `usercheck CASE --draft K [--users N] [--tokens T]`: the case's model with its draft head (`draft_model`) drafting K tokens a round, its greedy replies held to
/// the model's own alone (its engine without the head), token for token: each reply alone (`Gen::reply`, drafted), then through the scheduler (a drafted model
/// serves one sequence at a time, each of its decode steps a drafting round: `ModelStep::draft`), one prompt at a time and all of them asked at once.
fn users_drafted(case: &str, level: &str, users: usize, tokens: usize, k: usize) -> Result<()> {
    use crate::conv::Gen;
    use crate::sched::{Ev, Finish, Reply, Sched, Submit};
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    let dir = std::env::temp_dir().join(format!("glyd-usercheck-draft-{case}-{}", std::process::id()));
    let o = DynOpts { ctx: 1024, ..dyn_opts(level_layout(level)?) };
    let r: Result<(Vec<Vec<u32>>, Dyn, Vec<Vec<u32>>)> = (|| {
        draft_model(case, &dir)?;
        let vocab = crate::runtime::describe::describe_dir(&dir)?.0.desc.vocab as u32;
        let prompts: Vec<Vec<u32>> = (0..users).map(|u| (0..5 + 29 * u as u32).map(|i| 3 + (i * 7919 + u as u32 * 104_729) % (vocab - 3)).collect()).collect();
        let reply = Reply {
            params: crate::sampler::Params { temperature: 0.0, ..Default::default() },
            max_new: tokens,
            ignore_eos: true,
            logprobs: None,
            stop_ids: Vec::new(),
        };
        // (the model's own replies, its engine without the head)
        let mut plain = load_dyn(case, &dir, &o, level)?;
        let want = prompts.iter().map(|p| plain.reply(p, &reply, &mut |_| true).map(|r| r.0)).collect::<Result<Vec<_>>>()?;
        drop(plain);
        Ok((want, load_drafting_dyn(case, &dir, &o, k)?, prompts))
    })();
    let _ = std::fs::remove_dir_all(&dir);
    let (want, mut e, prompts) = r?;
    let reply = Reply {
        params: crate::sampler::Params { temperature: 0.0, ..Default::default() },
        max_new: tokens,
        ignore_eos: true,
        logprobs: None,
        stop_ids: Vec::new(),
    };
    println!(
        "{case} at {level}: drafting {k} tokens a round with its draft head, {users} prompts of {:?} tokens, {tokens} tokens each",
        prompts.iter().map(Vec::len).collect::<Vec<_>>()
    );
    let mut fails = 0usize;
    for (i, p) in prompts.iter().enumerate() {
        let got = e.reply(p, &reply, &mut |_| true)?.0;
        let ok = got == want[i];
        fails += !ok as usize;
        println!(
            "{} drafted reply {i} alone: {}",
            if ok { "PASS" } else { "FAIL" },
            if ok { "the model's own, token for token".into() } else { format!("{got:?} against {:?}", want[i]) }
        );
    }
    let Some((slots, step)) = e.as_step(users)? else { return err(format!("{case}: no scheduler's step of the drafted model")) };
    let mut s = Sched::new(step, slots).keeping(false);
    let mut ask = |ps: &[Vec<u32>]| -> Result<Vec<Vec<u32>>> {
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
    for (i, p) in prompts.iter().enumerate() {
        let got = ask(std::slice::from_ref(p))?.remove(0);
        let ok = got == want[i];
        fails += !ok as usize;
        println!(
            "{} through the scheduler, drafted, reply {i} alone: {}",
            if ok { "PASS" } else { "FAIL" },
            if ok { "the model's own, token for token".into() } else { format!("{got:?} against {:?}", want[i]) }
        );
    }
    let together = ask(&prompts)?;
    let same = together.iter().zip(&want).filter(|(a, b)| a == b).count();
    fails += prompts.len() - same;
    println!(
        "{} through the scheduler, drafted, {} prompts at once ({slots} sequence at a time): {same} of {} replies the model's own, token for token",
        if same == prompts.len() { "PASS" } else { "FAIL" },
        prompts.len(),
        prompts.len()
    );
    if fails > 0 {
        return err(format!("usercheck {case} --draft {k}: {fails} replies not the model's own"));
    }
    println!("PASS usercheck {case} --draft {k}");
    Ok(())
}

/// The plan `serve` makes of the case's random model at `level` on this GPU for `users` sequences, its draft model off (`plan::planned`): the plan the load takes,
/// its context, prompt chunk, layout, sequences and their bytes; or why it gives none (a refusal is the planner's word that it would not load).
/// The plan `serve DIR --users N` makes of the dynamic route's model in `dir` on this machine (src/cli.rs `dynamic_plan`, what `serve` loads), with the run's flags:
/// `usercheck --dir DIR --ctx plan`'s.
fn planned_dir(dir: &Path, users: usize) -> Result<crate::plan::LoadPlan> {
    let text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| Error(format!("{}: {e}", dir.display())))?;
    let family = crate::dynamic::split::family(&text).ok_or_else(|| Error(format!("{}: not a model of the dynamic route", dir.display())))?;
    let words = ["serve".to_string(), dir.display().to_string(), "--users".into(), users.to_string()];
    crate::cli::dynamic_plan(&crate::args::Args::from(words), &crate::cli::short_name(&dir.display().to_string()), family, dir, false)
        .map(|p| p.3)
        .map_err(|r| Error(format!("the planner: {} ({})", r.what, r.fix)))
}

fn planned_for(case: &str, level: &str, users: usize) -> Result<crate::plan::LoadPlan> {
    let dir = std::env::temp_dir().join(format!("glyd-planned-{case}-{}", std::process::id()));
    let got = (|| -> Result<crate::plan::LoadPlan> {
        random_model(case, &dir)?;
        let words = ["serve".to_string(), format!("{}:{level}", dir.display()), "--users".into(), users.to_string(), "--no-draft".into()];
        let f = crate::plan::planned(&crate::args::Args::from(words)).map_err(|r| Error(format!("the planner: {}", r.what)))?;
        if let Some(r) = f.refusal {
            return err(format!("the planner refuses the random {case} model at {level} for {users} users: {}", r.what));
        }
        f.plan
            .chosen
            .and_then(|c| c.load)
            .filter(|l| l.context > 0)
            .ok_or_else(|| Error(format!("the planner has no plan of the loader for the random {case} model (an estimate alone)")))
    })();
    let _ = std::fs::remove_dir_all(&dir);
    got
}

/// `splitcheck CASE [--split CUT,..] [--devices GPU,..] [--level L | --layout mma12] [--tokens T] [--mode M] [--tables] [--decoded]` (development): the case's random model
/// on the dynamic route as `run` opens it (`split::open`: its family's loader, any family of that route; the expert cache's case, `qwen4x-cb`, its pack in an
/// expert cache a stage, its misses as `--mode` says: `split_open`), once on one GPU and once with its layers cut into stages (by default before its middle layer,
/// both stages on GPU 0 or on the two `--devices` names: the hand-over through host memory checked on a machine with one GPU). A prompt of 24 tokens and one
/// past two chunks (150), each then T greedy steps fed the one-GPU run's tokens: every pass's logits the same bits both ways; and each prompt's greedy reply
/// through the scheduler with the log-probabilities a server sends (the token's and its 5 likeliest): every one the same bits both ways. `--tables`: the one-GPU run with its
/// tables read by rows on the GPU (src/dynamic/tables.rs `force`), the other with them where its description holds them (on the host: refused where none
/// is), on one GPU unless `--split` cuts it: the rows copied up are the GPU's rows. `--decoded` (a case whose tables are at 4 bits: `CASE+q4`, `qwen4x-rel`):
/// the one-GPU run on the same model with those tables as the bf16 rows they decode to (`tables_decoded`), on one GPU unless `--split` cuts the other: the
/// rows decoded as they are gathered are the bf16 table's.
pub fn split_check(a: &crate::args::Args) -> Result<()> {
    let case = a.need(1, "CASE")?;
    let level = a.flag("level").unwrap_or("penguin");
    let tokens: usize = a.num("tokens", 12usize)?;
    let mode =
        a.flag("mode").map_or(Some(crate::offload::Mode::Cpu), crate::offload::Mode::parse).ok_or_else(|| Error("--mode: copy, cpu, split or auto".into()))?;
    let dir = std::env::temp_dir().join(format!("glyd-splitcheck-{}-{}", case.replace('+', "-"), std::process::id()));
    let bf16 = dir.with_extension("decoded");
    let decoded = a.has("decoded");
    let r = (|| -> Result<()> {
        random_model(case, &dir)?;
        // (--decoded: the same model again, its tables at 4 bits written as the bf16 rows they decode to, for the one-GPU run)
        if decoded {
            random_model(case, &bf16)?;
            if tables_decoded(&bf16)? == 0 {
                return err(format!("{case}: its checkpoint holds no table at 4 bits (--decoded compares it with its bf16 rows)"));
            }
        }
        let text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| Error(e.to_string()))?;
        if crate::dynamic::split::family(&text).is_none() {
            return err(format!("{case}: its family is not on the dynamic route (`run` cuts no layers of it)"));
        }
        let config = |k: &str| crate::dynamic::split::config_num(&text, k).unwrap_or(0);
        let (nl, vocab) = (config("num_hidden_layers"), config("vocab_size"));
        let tables = a.has("tables");
        let hosted = crate::dynamic::tables::host_names(&text).len();
        if tables && hosted == 0 {
            return err(format!("{case}: its description holds no table on the host (--tables compares the two places)"));
        }
        // (no cut asked: before the middle layer, on the two GPUs `--devices` names, else both on GPU 0; with --tables or --decoded, none)
        let (cuts, devices) = match crate::dynamic::split::split_args(a)? {
            (c, d) if c.is_empty() && (tables || decoded) => (c, d[..1].to_vec()),
            (c, d) if c.is_empty() => (vec![nl / 2], if d.len() == 2 { d } else { vec![0, 0] }),
            x => x,
        };
        // (penguin's packs in the smallest layout, or the 12-bit one with --layout mma12, as an HBM GPU's plan holds them)
        let o = dyn_opts(a.layout()?.or(level_layout(level)?).or(Some(Layout::Tiered)));
        let (short, _, long) = standard_tokens(vocab);
        let prompts = [short, long];
        // (each pass's logits, a prompt after the other in one sequence each, the steps fed `feed` where given, else the run's own greedy tokens; then each prompt's
        // greedy reply through the scheduler, with the log-probabilities a server sends: the token's and its 5 likeliest's)
        type Served = Vec<(Vec<u32>, Vec<crate::sampler::Lp>)>;
        let run = |dir: &Path, cuts: &[usize], devices: &[i32], feed: Option<&[u32]>| -> Result<(Vec<Vec<f32>>, Vec<u32>, Served)> {
            let mut e = split_open(case, mode, dir, &o, cuts, devices)?;
            let (mut rows, mut fed) = (Vec::new(), Vec::new());
            for prompt in &prompts {
                e.reset()?;
                let mut s = e.prefill(prompt)?;
                rows.push(e.logits()?);
                for _ in 0..tokens {
                    let t = feed.map_or(s.token, |f| f[fed.len()]);
                    fed.push(t);
                    s = e.step(t)?;
                    rows.push(e.logits()?);
                }
            }
            let served = prompts.iter().map(|p| crate::dynamic::split::served(&mut e, p, tokens, Some(5))).collect::<Result<Served>>()?;
            Ok((rows, fed, served))
        };
        crate::dynamic::tables::force(tables.then_some(crate::runtime::kinds::TableAt::Gpu));
        let one = run(if decoded { &bf16 } else { &dir }, &[], &devices[..1], None);
        crate::dynamic::tables::force(None);
        let (one, fed, one_lp) = one?;
        let (staged, _, staged_lp) = run(&dir, &cuts, &devices, Some(&fed))?;
        let differ: Vec<usize> = (0..one.len()).filter(|&i| one[i].iter().map(|x| x.to_bits()).ne(staged[i].iter().map(|x| x.to_bits()))).collect();
        // (a served token's log-probabilities to the bit: the token, its log-probability, and each of the likeliest tokens and theirs)
        let bits = |l: &crate::sampler::Lp| (l.token, l.logprob.to_bits(), l.top.iter().map(|&(t, x)| (t, x.to_bits())).collect::<Vec<_>>());
        let lps: usize = one_lp.iter().map(|(_, l)| l.len()).sum();
        let lp_differ = one_lp
            .iter()
            .zip(&staged_lp)
            .map(
                |((ta, la), (tb, lb))| {
                    if ta != tb || la.len() != lb.len() {
                        la.len().max(lb.len())
                    } else {
                        la.iter().zip(lb).filter(|(x, y)| bits(x) != bits(y)).count()
                    }
                },
            )
            .sum::<usize>();
        println!(
            "{case} at {level}{}{}{}: {nl} layers, cut before {cuts:?} on GPUs {devices:?}; {} passes (prompts of {} and {}, {tokens} steps after each), {} with logits that differ from the one-GPU run's; \
             {lps} served tokens' log-probabilities (the token's and its 5 likeliest, through the scheduler), {lp_differ} not the one-GPU run's bits",
            if cached(case) { format!(", the expert cache's misses by {}", mode.name()) } else { String::new() },
            if tables { format!(", its tables read by rows ({hosted} tensors) on the GPU in the one-GPU run and on the host in the other") } else { String::new() },
            if decoded { ", its tables at 4 bits in the other run and the bf16 rows they decode to in the one-GPU run" } else { "" },
            one.len(),
            prompts[0].len(),
            prompts[1].len(),
            differ.len()
        );
        if !differ.is_empty() {
            return err(format!("{case}: the passes {differ:?} differ between one GPU and the stages"));
        }
        if lp_differ > 0 || lps == 0 {
            return err(format!("{case}: {lp_differ} of {lps} served tokens' log-probabilities differ between one GPU and the stages"));
        }
        Ok(())
    })();
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&bf16);
    r
}

/// Whether the case's experts are a codes pack's in the expert cache (`qwen4x-cb`, the release's form `qwen4x-rel`).
fn cached(case: &str) -> bool {
    matches!(q4_case(case).0, "qwen4x-cb" | "qwen4x-rel")
}

/// The case's random model as `splitcheck` opens it: on the dynamic route by its family's loader (`split::open`), or for the expert cache's case (`qwen4x-cb`)
/// its pack in an expert cache a stage (`dynamic::qwen4x::open_cached`, the case's places a layer each), its misses served as `mode` says (the host computes
/// them in `cpu`, the case's own; `copy` copies each into a place first); on one GPU where `cuts` is empty.
fn split_open(case: &str, mode: crate::offload::Mode, dir: &Path, o: &DynOpts, cuts: &[usize], devices: &[i32]) -> Result<crate::dynamic::split::Run> {
    if cached(case) {
        let mut k = qwen4x_cb_caching(dir);
        k.offload.mode = mode;
        return crate::dynamic::qwen4x::open_cached(dir, o, &k, cuts, devices);
    }
    crate::dynamic::split::open(dir, o, cuts, devices)
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
/// A KL that is not a number (a logit that is not one) stays one.
fn row_diff(want: &[f32], got: &[f32]) -> (usize, f32, f64, bool) {
    let differ = want.iter().zip(got).filter(|(a, b)| a.to_bits() != b.to_bits()).count() + want.len().abs_diff(got.len());
    let max = want.iter().zip(got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let (lw, lg) = (lse_of(want), lse_of(got));
    let kl: f64 = want.iter().zip(got).map(|(&a, &b)| ((a as f64 - lw).exp()) * ((a as f64 - lw) - (b as f64 - lg))).sum();
    let top = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|x| x.0);
    (differ, max, if kl.is_nan() { kl } else { kl.max(0.0) }, top(want) == top(got))
}

/// A row's lead: its largest logit less its second.
fn lead_of(l: &[f32]) -> f32 {
    let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &x in l {
        if x > a {
            (a, b) = (x, a);
        } else if x > b {
            b = x;
        }
    }
    a - b
}

/// The kind's lossless bound (the experts kind's fixtures': KL mean, max) a row run beside others is held to against the row alone where the engine computes
/// experts on the host (`host_experts`), its best token the same wherever the alone row's lead clears `CLEAR_LEAD`.
pub const BESIDE_KL: (f64, f64) = (1e-4, 2.5e-4);

/// Whether the engine's blocks of experts have the host compute pairs of a call: an expert cache whose mode gives the host its misses (cpu, split, auto; the
/// codebook cache's misses). The host and the GPU sum a pair in different orders, and which share of a call the host takes turns on what shares the call and on
/// the places' state, so a row's last bits turn on the rows beside it and on the calls before it. There the batch checks (`usercheck` with prompts one a pass,
/// `keepcheck`, `roundcheck`) hold a row beside others to the row alone at the kind's lossless bound (`Beside::held`), not bit for bit; everywhere else (every
/// expert on the GPU, copy mode, resident) bit for bit (the lead's ruling, 2026-11-25).
pub(crate) fn host_experts(e: &Dyn) -> bool {
    e.stores().iter().any(|s| s.wants_cpu())
}

/// Rows run beside others (or after other calls, or in other passes) against the same rows alone, as (alone, beside) pairs of whole rows of logits: the logits
/// that differ, the largest difference, the KL (alone || beside) over the whole vocabulary at most and on average, the rows whose alone lead clears `CLEAR_LEAD`
/// and those of them whose best token is the same, and whether every logit is a number.
#[derive(Clone, Debug, Default)]
pub(crate) struct Beside {
    pub rows: usize,
    pub logits: usize,
    pub differ: usize,
    pub max: f32,
    pub kl_max: f64,
    pub kl_mean: f64,
    pub clear: usize,
    pub top_same: usize,
    /// The rows whose best token is the same, of all of them.
    pub top_all: usize,
    pub finite: bool,
}

impl Beside {
    pub(crate) fn of(pairs: &[(&[f32], &[f32])]) -> Beside {
        let mut b = Beside { rows: pairs.len(), finite: true, ..Default::default() };
        let mut sum = 0f64;
        for (w, g) in pairs {
            let (differ, max, kl, top) = row_diff(w, g);
            b.logits += w.len();
            b.differ += differ;
            b.max = b.max.max(max);
            b.finite &= kl.is_finite() && g.iter().all(|x| x.is_finite());
            b.kl_max = if kl.is_nan() { kl } else { b.kl_max.max(kl) };
            sum += kl;
            b.top_all += top as usize;
            if lead_of(w) > CLEAR_LEAD {
                b.clear += 1;
                b.top_same += top as usize;
            }
        }
        b.kl_mean = sum / pairs.len().max(1) as f64;
        b
    }

    /// Within the kind's lossless bound (`BESIDE_KL`), the best token the same at every clear lead.
    pub(crate) fn bound(&self) -> bool {
        self.finite && self.kl_max <= BESIDE_KL.1 && self.kl_mean <= BESIDE_KL.0 && self.top_same == self.clear
    }

    /// Held as the engine allows: bit for bit, or the kind's bound where it computes experts on the host (`host_experts`).
    pub(crate) fn held(&self, host: bool) -> bool {
        if host {
            self.bound()
        } else {
            self.finite && self.differ == 0
        }
    }

    /// What it found, in words: the logits that differ and the largest difference, the KL, the best tokens; where `host`, why the bits are not held and whether the
    /// kind's bound is.
    pub(crate) fn words(&self, host: bool) -> String {
        let mut s = format!(
            "{} of {} logits differ, the largest by {:e}, KL at most {:e} (mean {:e}), the largest logit the same token's in {} of {} (where the lead clears {CLEAR_LEAD}: {} of {})",
            self.differ, self.logits, self.max, self.kl_max, self.kl_mean, self.top_all, self.rows, self.top_same, self.clear
        );
        if !self.finite {
            s += "; a logit is not a number";
        }
        if host {
            s += &format!(
                "; experts computed on the host: bits not held, the kind's bound {} (KL mean <= {:e}, max <= {:e}, the best token the same where the lead clears {CLEAR_LEAD})",
                if self.bound() { "held" } else { "NOT held" },
                BESIDE_KL.0,
                BESIDE_KL.1
            );
        }
        s
    }
}

/// Each mixture-of-experts layer's router as the description reads it (its matrix, hash table and choice bias, f32, from the checkpoint), by the executor's layer
/// number: what a check measures a choice of experts by on an engine's own router input (`ffn::route_of`). Loaded only where the engine computes experts on the
/// host, for the batch checks' choices apart (`Judge`).
pub(crate) struct Routers(BTreeMap<usize, (crate::runtime::kinds::Moe, crate::runtime::reference::Params)>);

impl Routers {
    /// The routers of the model in `dir`, keyed by the layers of `e`'s description whose operations hold a block of experts (in order, the description's
    /// mixture-of-experts blocks in order).
    pub(crate) fn load(dir: &Path, e: &Dyn) -> Result<Routers> {
        use crate::runtime::kinds::{Body, Ffn};
        use crate::runtime::reference::{model, Params};
        let (desc, label) = crate::runtime::describe::describe_dir(dir)?;
        let ck = io(glyd_gpu::safetensors::Checkpoint::open(dir))?;
        let src = |n: &str| model::checkpoint_tensor(&ck, n);
        let layers: Vec<usize> =
            e.desc().layers.iter().enumerate().filter(|(_, ops)| ops.iter().any(|op| matches!(op, crate::dynamic::Op::Moe(_)))).map(|x| x.0).collect();
        let mut blocks = Vec::new();
        for (b, roles) in desc.desc.blocks.iter().zip(&desc.blocks) {
            let Body::Ffn(Ffn::Moe(m)) = &b.body else { continue };
            let mut p = Params::new();
            for (role, src_of) in
                roles.iter().filter(|(r, _)| matches!(r.as_str(), "router" | "router_bias" | "router_a" | "router_b" | "hash" | "choice_bias"))
            {
                let (shape, v) = model::load(&src, role, src_of).map_err(|e| e.context(&label))?;
                p.put(role, &shape, v);
            }
            blocks.push((m.clone(), p));
        }
        if blocks.len() != layers.len() {
            return err(format!("{label}: {} blocks of experts in the description, {} layers of the executor's with experts", blocks.len(), layers.len()));
        }
        Ok(Routers(layers.into_iter().zip(blocks).collect()))
    }

    /// How far apart choices `own` and `other` of one token are on the router input `x` of layer `layer` (`ffn::margin_between` on its ranks, unsigned); a
    /// hash-routed layer's choice is the token's, never a tie (infinite).
    fn margin(&self, layer: usize, x: &[f32], own: &[usize], other: &[usize]) -> Result<f32> {
        let (m, p) = self.0.get(&layer).ok_or_else(|| Error(format!("layer {layer}: no router of a block of experts")))?;
        if m.router.hash {
            return Ok(f32::INFINITY);
        }
        let r = crate::runtime::reference::ffn::route_of(m, p, x, 0, None)?;
        Ok(crate::runtime::reference::ffn::margin_between(&r.rank, own, other).abs())
    }
}

/// A sequence's history of discrete choices as a check takes it from a step (`ModelStep::row_choices`), position by position from its first token: each
/// position's choices (an indexer's picks, a router's experts, a layer each), empty where none was seen.
pub(crate) type Hist = Vec<Vec<crate::dynamic::RowRoute>>;

/// The first discrete choice two runs of one sequence made apart (`first_apart`), in causal order: its position, layer and kind, what each run chose there,
/// and how far apart the two are on each run's own input (`Routers::margin` for a router's experts, the pools' scores for an indexer's pick: below
/// `kindcheck::NEAR`, a near tie).
#[derive(Clone, Debug)]
pub(crate) struct FirstApart {
    pub pos: usize,
    pub layer: usize,
    pub kind: crate::dynamic::ChoiceKind,
    pub first: Vec<i64>,
    pub second: Vec<i64>,
    pub margins: (f32, f32),
}

impl FirstApart {
    fn near(&self) -> bool {
        self.margins.0 < crate::dynamic::kindcheck::NEAR && self.margins.1 < crate::dynamic::kindcheck::NEAR
    }

    fn words(&self) -> String {
        format!(
            "position {}, layer {}, {}: the first run chose {:?}, the second {:?}; margin {:.2e} on the first run's own input, {:.2e} on the second's",
            self.pos,
            self.layer,
            match self.kind {
                crate::dynamic::ChoiceKind::Pick => "the indexer's pools",
                crate::dynamic::ChoiceKind::Route => "the router's experts",
            },
            self.first,
            self.second,
            self.margins.0,
            self.margins.1
        )
    }
}

/// The first choice two runs' histories of one sequence hold apart: over the positions both saw, the earliest position, then the earliest layer, an
/// indexer's pick before the layer's experts; with the positions before it that one of the two runs did not see. A router's choice is measured by
/// `routers` (none: not measured, NaN, never a near tie), an indexer's by its pools' scores (`ffn::margin_between` on them).
fn first_apart(routers: Option<&Routers>, a: &Hist, b: &Hist) -> Result<(Option<FirstApart>, usize)> {
    use crate::dynamic::{ChoiceFrom, ChoiceKind};
    use crate::runtime::reference::ffn::margin_between;
    let mut unseen = 0;
    let ids = |v: &[i64]| v.iter().map(|&e| e as usize).collect::<Vec<usize>>();
    for (pos, (ca, cb)) in a.iter().zip(b).enumerate() {
        if ca.is_empty() || cb.is_empty() {
            unseen += 1;
            continue;
        }
        let mut keys: Vec<(usize, ChoiceKind)> = ca.iter().chain(cb).map(|c| (c.layer, c.kind)).collect();
        (keys.sort_unstable(), keys.dedup());
        for (layer, kind) in keys {
            let find = |h: &[crate::dynamic::RowRoute]| h.iter().find(|c| c.layer == layer && c.kind == kind).cloned();
            let (Some(x), Some(y)) = (find(ca), find(cb)) else {
                unseen += 1;
                continue;
            };
            let (mut u, mut v) = (x.chosen.clone(), y.chosen.clone());
            (u.sort_unstable(), v.sort_unstable());
            if u == v {
                continue;
            }
            let (sa, sb) = (ids(&x.chosen), ids(&y.chosen));
            let on = |c: &crate::dynamic::Choice, own: &[usize], other: &[usize]| -> Result<f32> {
                Ok(match (&c.from, kind) {
                    (ChoiceFrom::Router(input), ChoiceKind::Route) => match routers {
                        Some(r) => r.margin(layer, &input.iter().map(|&h| f32::from_bits((h as u32) << 16)).collect::<Vec<f32>>(), own, other)?,
                        None => f32::NAN,
                    },
                    (ChoiceFrom::Scores(s), ChoiceKind::Pick) => margin_between(s, own, other).abs(),
                    _ => f32::NAN,
                })
            };
            let margins = (on(&x, &sa, &sb)?, on(&y, &sb, &sa)?);
            return Ok((Some(FirstApart { pos, layer, kind, first: x.chosen.clone(), second: y.chosen.clone(), margins }), unseen));
        }
    }
    Ok((None, unseen))
}

/// One row run beside others and the same row alone, as a host-experts check holds them (`Judge`): the alone row's logits and the beside row's, and each run's
/// history of the row's sequence up to the row (None: not seen).
pub(crate) struct Judged<'a> {
    pub alone: &'a [f32],
    pub beside: &'a [f32],
    pub choices: (Option<Hist>, Option<Hist>),
}

/// Rows beside others held to the rows alone where the engine computes experts on the host (the coordinator's causal rule, 2026-11-25), row by row, by the
/// first discrete choice the two runs made apart in the history of the row's sequence up to the row (`first_apart`: experts' routes and indexers' picks,
/// every layer and position, in causal order):
///   - none apart: held to the kind's bound (KL at most `BESIDE_KL`'s max, the best token the same where the alone row's lead clears `CLEAR_LEAD`), always;
///     the held rows' KL mean within the bound's mean too;
///   - the first apart a near tie on each run's own input (below `NEAR`): reported, not failed, with that choice (position, layer, kind, both margins); every
///     later row of the sequence sees the same first choice, and is reported too;
///   - the first apart not a near tie: failed (a fault, not a tie); and a row past the bound whose history shows none apart: failed.
#[derive(Debug, Default)]
pub(crate) struct Judge {
    pub held: usize,
    pub reported: usize,
    pub failed: usize,
    held_kl: f64,
    /// The bound the rows with no choice apart are held to (KL mean, max): the kind's (`BESIDE_KL`), or a tenth of it for a draw's own floor.
    bound: (f64, f64),
    /// Every row's figures, held or not (the line's words).
    pub all: Beside,
    /// A line for each row reported or failed.
    pub notes: Vec<String>,
}

impl Judge {
    /// The rows judged at the kind's bound: `routers` measure a router's choice apart (None: such a choice is never a near tie).
    pub(crate) fn of(rows: &[Judged], routers: Option<&Routers>) -> Result<Judge> {
        Judge::of_at(rows, routers, BESIDE_KL)
    }

    /// `of` with the rows that show no choice apart held to `bound` (KL mean, max).
    pub(crate) fn of_at(rows: &[Judged], routers: Option<&Routers>, bound: (f64, f64)) -> Result<Judge> {
        let pairs: Vec<(&[f32], &[f32])> = rows.iter().map(|r| (r.alone, r.beside)).collect();
        let mut j = Judge { all: Beside::of(&pairs), bound, ..Default::default() };
        for (i, r) in rows.iter().enumerate() {
            let b = Beside::of(&[(r.alone, r.beside)]);
            let within = b.finite && b.kl_max <= bound.1 && b.top_same == b.clear;
            let (first, unseen) = match &r.choices {
                (Some(a), Some(c)) => first_apart(routers, a, c)?,
                _ => (None, usize::MAX),
            };
            let figures = format!("KL {:.3e}, the largest difference {:.3e}, the best token the same: {}", b.kl_max, b.max, b.top_all == 1);
            let seen = match unseen {
                0 => String::new(),
                usize::MAX => " (no history of choices seen)".to_string(),
                n => format!(" ({n} choices before it not seen in both runs)"),
            };
            match first {
                Some(f) if f.near() && b.finite => {
                    j.reported += 1;
                    j.notes.push(format!("row {i}: reported, a near tie taken the other way first at {}{seen}; {figures}", f.words()));
                }
                Some(f) => {
                    j.failed += 1;
                    j.notes.push(format!("row {i}: FAILED, the first choice apart is not a near tie: {}{seen}; {figures}", f.words()));
                }
                None if within => {
                    j.held += 1;
                    j.held_kl += b.kl_max;
                }
                None => {
                    j.failed += 1;
                    let past = if bound == BESIDE_KL { "the kind's bound" } else { "a tenth of the kind's bound" };
                    j.notes.push(format!("row {i}: FAILED, past {past} with no choice apart in its history{seen}; {figures}"));
                }
            }
        }
        Ok(j)
    }

    /// Whether the rows hold: none failed, and the held rows' KL mean within the bound's.
    pub(crate) fn ok(&self) -> bool {
        self.failed == 0 && self.held_kl / self.held.max(1) as f64 <= self.bound.0
    }

    /// The counts in words, and the held rows' KL mean.
    pub(crate) fn words(&self) -> String {
        format!(
            "{}: bits not held; {} rows held to {} (KL mean {:.3e}, at most {:e}; max at most {:e}; the best token where the lead clears {CLEAR_LEAD}), {} reported: a near tie taken the other way, {} failed",
            if self.bound == BESIDE_KL { "experts computed on the host" } else { "the experts' sums in another order" },
            self.held,
            if self.bound == BESIDE_KL { "the kind's bound" } else { "a tenth of the kind's bound" },
            self.held_kl / self.held.max(1) as f64,
            self.bound.0,
            self.bound.1,
            self.reported,
            self.failed
        )
    }

    /// The line, PASS or FAIL as `ok` says (`ok()`, and what else the check holds), and its notes printed.
    pub(crate) fn print(&self, what: &str, ok: bool) {
        println!("{} {what}: {}; {}", if ok { "PASS" } else { "FAIL" }, self.all.words(false), self.words());
        for n in &self.notes {
            println!("  {n}");
        }
    }
}

/// A reply as a check sees it: its tokens, and for each the row of logits it was chosen from with the history of discrete choices up to that row (None: not
/// seen).
pub(crate) type Seen = (Vec<u32>, Vec<(Vec<f32>, Option<Hist>)>);

/// Replies run beside others against the same replies alone (`Seen`): each reply's rows compared while its tokens are the alone reply's, the row where a token
/// first differs too (its inputs are the same), none after it (the tokens fed differ). The rows compared (`Judged`), and each reply that parts from alone: its
/// number, the token where, and the alone row's lead there.
pub(crate) fn replies_beside<'a>(alone: &'a [Seen], beside: &'a [Seen]) -> (Vec<Judged<'a>>, Vec<(usize, usize, f32)>) {
    let (mut rows, mut parted): (Vec<Judged>, Vec<(usize, usize, f32)>) = (Vec::new(), Vec::new());
    for (i, ((ta, ra), (tb, rb))) in alone.iter().zip(beside).enumerate() {
        for (j, (a, b)) in ra.iter().zip(rb).enumerate() {
            rows.push(Judged { alone: &a.0, beside: &b.0, choices: (a.1.clone(), b.1.clone()) });
            if ta.get(j) != tb.get(j) {
                parted.push((i, j, lead_of(&a.0)));
                break;
            }
        }
    }
    (rows, parted)
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

/// A row's history of discrete choices, where the step keeps them (`Hist`).
type Ch = Option<Hist>;

/// The history of the sequence in `slot` up to and with position `upto`, as the step keeps it now (`ModelStep::row_choices`, while it traces them): None where
/// it keeps none.
fn hist(s: &dyn crate::sched::ModelStep, slot: i32, upto: usize) -> Ch {
    let h: Vec<Option<Vec<crate::dynamic::RowRoute>>> = (0..=upto).map(|p| s.row_choices(slot, p)).collect();
    h.iter().any(Option::is_some).then(|| h.into_iter().map(Option::unwrap_or_default).collect())
}

/// `fill_from` to the logits after `t`, with the history of discrete choices up to its last row (where the step traces them, `ModelStep::trace_choices`).
fn fill_last(s: &mut dyn crate::sched::ModelStep, slot: i32, start: usize, t: &[u32], by: usize) -> Result<(Vec<f32>, Ch)> {
    let l = fill_from(s, slot, start, t, by, true)?.ok_or_else(|| Error("no logits after the last chunk".into()))?;
    Ok((l, hist(s, slot, start + t.len() - 1)))
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
/// GLYD_ENGINE_BATCH_PREFILL=0 (every reply to be the reply alone) only the decode steps are run, and not a bit may differ at any size. Where the engine computes experts on the
/// host (`host_experts`: which pairs the host takes turns on the call and the places' state, and the host sums a pair in another order than the GPU), what is required bit for
/// bit is held at the kind's lossless bound instead (`Beside`), and the line says why.
pub fn round_check(a: &crate::args::Args) -> Result<()> {
    use crate::conv::Gen;
    use crate::sched::{ModelStep, PrefillJob, Row};
    let case = a.need(1, "CASE")?;
    let level = a.flag("level").unwrap_or("penguin");
    let steps: usize = a.num("tokens", 8usize)?;
    let bound: Option<f64> = a.flag("kl").map(|b| b.parse().map_err(|_| Error(format!("--kl {b}: a number")))).transpose()?;
    let (mut e, routers): (Dyn, Option<Routers>) = match a.flag("dir") {
        Some(dir) => {
            let o = DynOpts { ctx: a.num("ctx", 4096usize)?, chunk: a.num("chunk", 512usize)?, graph: true, device: 0, layout: level_layout(level)? };
            let e = load_dyn(case, Path::new(dir), &o, level)?;
            let r = routers_of(Path::new(dir), &e)?;
            (e, r)
        }
        None => random_dyn_routed(case, level, level_layout(level)?, "roundcheck", None)?,
    };
    let (vocab, exact, host) = (e.desc().vocab as u32, e.rows_exact(), host_experts(&e));
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
    // (where the engine computes experts on the host, the routers' choices of every row kept by sequence and place: each compared row's in each run, taken right after
    // the run (`row_choices`), judge a row past the kind's bound)
    s.trace_choices(host);
    let chosen = |s: &dyn ModelStep, slot: i32, from: usize, n: usize| -> Vec<Ch> { (from..from + n).map(|p| hist(s, slot, p)).collect() };
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
    let ref_ch = chosen(s, 0, pa.len(), steps);
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
        "{case} at {level}: {steps} steps of a sequence after {} prompt tokens ({}); no mixture of experts: {exact}; experts computed on the host: {host}; a chunk {chunk}, a sequence {window}",
        pa.len(),
        if text.is_some() { "a text's tokens, the bench's passage" } else { "tokens by a rule, not a text's" }
    );
    println!(
        "the reference (the row alone): its two largest logits are {margin:e} apart at the closest step{}",
        text.as_ref().map_or(String::new(), |t| format!("; the text is {} tokens", t.len()))
    );
    let mut fails = 0;
    // (`held`: not one logit may differ, or where the engine computes experts on the host each row held to the kind's bound or reported for a near tie taken the other
    // way (`Judge`, by the rows' choices `ch`, each pair's); else counted)
    let mut report = |what: &str, held: bool, pairs: &[(&[f32], &[f32])], ch: &[(Ch, Ch)]| -> Result<()> {
        if host && held {
            let rows: Vec<Judged> =
                pairs.iter().enumerate().map(|(i, (w, g))| Judged { alone: w, beside: g, choices: ch.get(i).cloned().unwrap_or((None, None)) }).collect();
            let j = Judge::of(&rows, routers.as_ref())?;
            let bad = !j.ok() || bound.is_some_and(|k| j.all.kl_max > k);
            fails += bad as usize;
            j.print(what, !bad);
            return Ok(());
        }
        let b = Beside::of(pairs);
        let bad = (held && !b.held(host)) || bound.is_some_and(|k| b.kl_max > k) || !b.finite;
        fails += bad as usize;
        println!(
            "{} {what}: {}",
            if bad {
                "FAIL"
            } else if held || bound.is_some() {
                "PASS"
            } else {
                "    "
            },
            b.words(false)
        );
        Ok(())
    };
    // (each pair's choices: the reference's and another run's, row for row)
    let paired = |a: &[Ch], b: &[Ch]| -> Vec<(Ch, Ch)> { a.iter().cloned().zip(b.iter().cloned()).collect() };
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
        let got_ch = chosen(s, 0, pa.len(), steps);
        s.release(&slots.concat())?;
        let pairs: Vec<(&[f32], &[f32])> = reference.iter().zip(&got).map(|(w, g)| (&w[..], &g[..])).collect();
        report(&format!("{n} rows a step, the row with {} other sequences' rows, against alone", n - 1), n <= 16 || strict, &pairs, &paired(&ref_ch, &got_ch))?;
    }
    if strict {
        s.trace_choices(false);
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
        // the row beside a prompt's chunks, and the prompt's logits after its last; the row's choices and the prompt's last row's
        let beside = |s: &mut dyn ModelStep, prompt: &[u32]| -> Result<(Vec<Vec<f32>>, Vec<f32>, Vec<Ch>, Ch)> {
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
            let (ch, last) = (chosen(s, 0, pa.len(), n), hist(s, 1, n * c - 1));
            s.release(&[0, 1])?;
            Ok((got, after, ch, last))
        };
        let py = toks(n * c, 1005 + c as u32);
        let (gx, after, gx_ch, after_ch) = beside(s, &px)?;
        let (gy, _, _, _) = beside(s, &py)?;
        let pairs: Vec<(&[f32], &[f32])> = reference.iter().zip(&gx).map(|(w, g)| (&w[..], &g[..])).collect();
        report(&format!("{rows} rows a step, the row with a prompt's chunk of {c}, against alone"), rows <= 16, &pairs, &paired(&ref_ch, &gx_ch))?;
        let kind = if of_text { "the text's" } else { "tokens by a rule" };
        let pairs: Vec<(&[f32], &[f32])> = gx.iter().zip(&gy).map(|(w, g)| (&w[..], &g[..])).collect();
        report(&format!("{rows} rows a step, the row with another prompt of {c} beside it, against the first"), exact, &pairs, &[])?;
        // the prompt's side: the same chunks alone, and whole chunks alone
        let alone = fill(s, 1, &px, c, true)?.expect("the last chunk's logits");
        let alone_ch = hist(s, 1, px.len() - 1);
        s.release(&[1])?;
        report(
            &format!("{rows} rows a step, a prompt of {} ({kind}) in chunks of {c} beside the row, against the chunks alone", px.len()),
            rows <= 16,
            &[(&alone[..], &after[..])],
            &[(alone_ch, after_ch)],
        )?;
        let whole = fill(s, 1, &px, chunk, true)?.expect("the last chunk's logits");
        s.release(&[1])?;
        report(&format!("(yardstick) that prompt alone in chunks of {c}, against alone in chunks of {chunk}"), false, &[(&whole[..], &alone[..])], &[])?;
    }
    // prompts of different lengths in one pass (`prefill_many`: prompts that come in together), each prompt's logits after its last token and after the step that follows it
    // (its state and cache as the pass left them), against the prompt alone; the first prompt beside other prompts of the same lengths, against beside the first ones
    // (each with its two rows' choices: the prompt's last row's, the step's)
    type Two = (Vec<f32>, Vec<f32>, [Ch; 2]);
    let alone_then_step = |s: &mut dyn ModelStep, p: &[u32]| -> Result<Two> {
        let after = fill(s, 0, p, chunk, true)?.expect("the prompt's logits");
        room(s, 0, p.len() + 2)?;
        s.decode(&[Row { token: fa[0], pos: p.len(), pages: &[0], after: None }])?;
        let next = s.logits(0)?;
        let ch = [hist(s, 0, p.len() - 1), hist(s, 0, p.len())];
        s.release(&[0])?;
        Ok((after, next, ch))
    };
    let together = |s: &mut dyn ModelStep, ps: &[Vec<u32>]| -> Result<Vec<Two>> {
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
            let ch = [hist(s, sl[0], p.len() - 1), hist(s, sl[0], p.len())];
            out.push((a, s.logits(0)?, ch));
        }
        s.release(&slots.concat())?;
        Ok(out)
    };
    // (15 rows in all: the products' kernels of a pass alone, so every bit is required; then 56 rows, which a pass of 64 holds, and 74: another kernel's sums, counted.
    // A set this step cannot run is said, not passed over: a run that left one out did not hold it)
    for set in [&[7usize, 5, 3][..], &[23, 9, 19, 5], &[37, 9, 23, 5]] {
        let lens: Vec<usize> = set.iter().copied().take(k.min(s.prefill_rows())).filter(|&l| l + 2 < window).collect();
        if lens.len() < 2 || lens.iter().sum::<usize>() > chunk {
            println!(
                "SKIP prompts of {set:?} in one pass: {} of them fit ({k} sequences, {} prompts a pass, a sequence of {window}), and a pass holds {chunk} rows",
                lens.len(),
                s.prefill_rows()
            );
            continue;
        }
        let rows: usize = lens.iter().sum();
        let ps: Vec<Vec<u32>> = lens.iter().enumerate().map(|(j, &l)| toks(l, 3000 + j as u32)).collect();
        let alone: Vec<Two> = ps.iter().map(|p| alone_then_step(s, p)).collect::<Result<_>>()?;
        let got = together(s, &ps)?;
        let pairs: Vec<(&[f32], &[f32])> = alone.iter().zip(&got).flat_map(|(w, g)| [(&w.0[..], &g.0[..]), (&w.1[..], &g.1[..])]).collect();
        let ch: Vec<(Ch, Ch)> = alone.iter().zip(&got).flat_map(|(w, g)| [(w.2[0].clone(), g.2[0].clone()), (w.2[1].clone(), g.2[1].clone())]).collect();
        report(
            &format!("prompts of {lens:?} in one pass ({rows} rows), each one's logits after it and after its next step, against alone"),
            rows <= crate::dynamic::EXACT_ROWS,
            &pairs,
            &ch,
        )?;
        let others: Vec<Vec<u32>> = std::iter::once(ps[0].clone()).chain(lens[1..].iter().enumerate().map(|(j, &l)| toks(l, 4000 + j as u32))).collect();
        let beside = together(s, &others)?;
        report(
            &format!("prompts of {lens:?} in one pass, the first beside other prompts of those lengths, against beside the first ones"),
            exact,
            &[(&got[0].0[..], &beside[0].0[..]), (&got[0].1[..], &beside[0].1[..])],
            &[],
        )?;
    }
    s.trace_choices(false);
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

/// `keepcheck CASE [--level L] [--dir MODEL --ctx N --chunk N] [--kv12]` (gemma4, qwen35, muse, kimi, glm; R4c's first step; `--kv12`: the pools in 12-bit pages): a finished sequence kept in its slot and gone on with, as a
/// conversation's next turn does (`ModelStep::resume` and `mark`, src/dynamic/rows.rs), on the case's random model, or the model in MODEL (a real one). Every comparison is of logits, bit for bit, against a second slot that ran the same tokens
/// in the same passes and was never resumed (where the engine computes experts on the host, at the kind's lossless bound: `host_experts`, `Beside`):
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
///     reply is, token for token, what the same passes give by hand (where the engine computes experts on the host, a token apart only where the row's lead is a near tie).
pub fn keep_check(a: &crate::args::Args) -> Result<()> {
    use crate::conv::Gen;
    use crate::sched::{Ev, Finish, ModelStep, Reply, Row, Sched, Submit};
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    let case = a.need(1, "CASE")?;
    let level = a.flag("level").unwrap_or("penguin");
    let (mut e, routers): (Dyn, Option<Routers>) = match a.flag("dir") {
        Some(dir) => {
            let o = DynOpts { ctx: a.num("ctx", 4096usize)?, chunk: a.num("chunk", 256usize)?, graph: true, device: 0, layout: level_layout(level)? };
            let e = load_dyn(case, Path::new(dir), &o, level)?;
            let r = routers_of(Path::new(dir), &e)?;
            (e, r)
        }
        None => random_dyn_routed(case, level, level_layout(level)?, "keepcheck", None)?,
    };
    // (--kv12: the pools in 12-bit lossless pages, of tables made on prompts of the model's tokens, as a server's that takes them)
    if a.has("kv12") {
        let v = e.desc().vocab as u32;
        let prompts: Vec<Vec<u32>> = (0..4u32).map(|k| (0..320u32).map(|i| 3 + (i * 7919 + k * 104_729) % (v - 3)).collect()).collect();
        let t = crate::dynamic::kv12::calibrate(&mut e, &prompts, "keepcheck")?;
        e.set_kv12(Some(t));
    }
    let twelve = Gen::held(&e).is_some_and(|h| h.kv_type == crate::plan::kv_format_of(true));
    let (vocab, cuts, marks, window, host) = (e.desc().vocab as u32, e.cuts(), e.can_mark(), e.least_window(), host_experts(&e));
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
    // (where the engine computes experts on the host, the routers' choices of every row kept by sequence and place: each compared row's, taken right after its run
    // (`fill_last`), judge a row past the kind's bound)
    s.trace_choices(host);
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
    // (0 or 1: whether the logits are not the other slot's: bit for bit, or where the engine computes experts on the host the row held to the kind's bound or reported
    // for a near tie taken the other way (`Judge`, by the two runs' choices of the row))
    let same = |what: &str, want: &(Vec<f32>, Ch), got: &(Vec<f32>, Ch)| -> Result<usize> {
        let what = format!("{what}: against a sequence that ran the same tokens on its own");
        if host {
            let j = Judge::of(&[Judged { alone: &want.0, beside: &got.0, choices: (want.1.clone(), got.1.clone()) }], routers.as_ref())?;
            let ok = j.ok();
            j.print(&what, ok);
            return Ok((!ok) as usize);
        }
        let b = Beside::of(&[(&want.0[..], &got.0[..])]);
        let ok = b.held(false);
        println!("{} {what}, {}", if ok { "PASS" } else { "FAIL" }, b.words(false));
        Ok((!ok) as usize)
    };
    // a slot that should hold nothing: a prompt begins in it as in another empty one
    let empty = |s: &mut dyn ModelStep, slot: i32, other: i32, what: &str, t: &[u32]| -> Result<usize> {
        let got = fill_last(s, slot, 0, t, chunk)?;
        let want = fill_last(s, other, 0, t, chunk)?;
        same(what, &want, &got)
    };
    let mut fails = 0;
    println!(
        "{case} at {level}: a chunk {chunk}; a kept sequence can be cut back anywhere: {cuts}; its states are kept at a mark: {marks}; experts computed on the host: {host}"
    );
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
        let got = fill_last(s, 0, n, &tail, chunk)?;
        fill(s, 1, &p, chunk, false)?;
        steps(s, 1, p.len(), &g)?;
        let want = fill_last(s, 1, n, &tail, chunk)?;
        fails += same(&format!("the whole of a sequence of {n} tokens, then {} more", tail.len()), &want, &got)?;
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
        let got = fill_last(s, 0, kept, &alt, chunk)?;
        fill(s, 1, &p, chunk, false)?;
        steps(s, 1, p.len(), &g[..kept - p.len()])?;
        let want = fill_last(s, 1, kept, &alt, chunk)?;
        let place = if kept == cut { "there".to_string() } else { format!("its prompt's end, {kept}, where its states were kept") };
        fails += same(&format!("a sequence of {n} tokens asked to go on within {cut}: from {place}, then {} others", alt.len()), &want, &got)?;
    } else {
        // refused, and nothing of it changed: it goes on from its end as one that was never asked
        let whole = s.resume(0, n)?;
        let got = fill_last(s, 0, n, &tail, chunk)?;
        fill(s, 1, &p, chunk, false)?;
        steps(s, 1, p.len(), &g)?;
        let want = fill_last(s, 1, n, &tail, chunk)?;
        let what = format!("a sequence of {n} tokens asked to go on within {cut}: refused (its states are what they are after the last token, and not kept at a mark) and as it was ({whole} held); from its end, {} more", tail.len());
        fails += same(&what, &want, &got)? | (whole != n) as usize;
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
                let got = fill_last(s, 0, to, &alt, chunk)?;
                fill(s, 1, &long[..to.min(long.len())], chunk, false)?;
                steps(s, 1, long.len(), &g[..by_hand])?;
                let want = fill_last(s, 1, to, &alt, chunk)?;
                fails += same(&format!("a sequence of {all} tokens asked to go on within {ask}: from {to}, then {} others", alt.len()), &want, &got)?;
            } else if kept == 0 && !must {
                // refused, and nothing of it changed: it goes on from its end
                let whole = s.resume(0, all)?;
                let got = fill_last(s, 0, all, &alt, chunk)?;
                fill(s, 1, &long, chunk, false)?;
                steps(s, 1, long.len(), &g)?;
                let want = fill_last(s, 1, all, &alt, chunk)?;
                let what = format!(
                    "a sequence of {all} tokens asked to go on within {ask}: refused (the sliding layers' blocks behind it were given back) and as it was ({whole} held); from its end, {} more",
                    alt.len()
                );
                fails += same(&what, &want, &got)? | (whole != all) as usize;
            } else {
                println!("FAIL a sequence of {all} tokens asked to go on within {ask}: {kept} kept");
                fails += 1;
            }
            s.release(&[0, 1])?;
        }
    }
    // a start shared with an empty slot: of a sequence of a prompt (marked at its end) and 24 steps, what a request that has its first 50 tokens can have (none where the mark
    // is inside a block of rings in 12-bit pages, which `share` does not copy: the prompt runs whole)
    {
        let should = if cuts {
            cut / 16 * 16
        } else if marks && !(twelve && p.len() % 16 != 0) {
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
            let source = fill_last(s, 0, n, &tail, chunk)?;
            fill(s, 1, &p, chunk, false)?;
            steps(s, 1, p.len(), &g)?;
            let alone = fill_last(s, 1, n, &tail, chunk)?;
            fails += same(&format!("the sequence the start was asked of, gone on with {} more", tail.len()), &alone, &source)?;
        } else {
            // the sharer goes on, then the sequence it shared from; each against a slot that ran the same tokens and shared nothing
            let sharer = fill_last(s, 1, got, &alt, chunk)?;
            let source = fill_last(s, 0, n, &tail, chunk)?;
            fill(s, 2, &p, chunk, false)?;
            steps(s, 2, p.len(), &g[..got - p.len()])?;
            let want = fill_last(s, 2, got, &alt, chunk)?;
            fails += same(&format!("a start of {cut} tokens shared from a sequence of {n}: {got} of them, then {} others", alt.len()), &want, &sharer)?;
            s.release(&[2])?;
            fill(s, 2, &p, chunk, false)?;
            steps(s, 2, p.len(), &g)?;
            let alone = fill_last(s, 2, n, &tail, chunk)?;
            fails += same(&format!("the sequence the start was shared from, gone on with {} more", tail.len()), &alone, &source)?;
            s.release(&[0, 1, 2])?;
            // the sequence it shared from ends first: what the two held together stays the sharer's
            let again = begin(s)?;
            s.release(&[0])?;
            let after = fill_last(s, 1, again, &alt, chunk)?;
            fails += same("the same start, after the sequence it was shared from has ended", &want, &after)?;
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
                let sharer = fill_last(s, 1, got, &alt, chunk)?;
                fill(s, 2, &long[..got], chunk, false)?;
                let want = fill_last(s, 2, got, &alt, chunk)?;
                fails += same(
                    &format!("a start of {cut} tokens shared from a sequence of {}: {got} of them, then {} others", long.len(), alt.len()),
                    &want,
                    &sharer,
                )?;
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
                let sharer = fill_last(s, 1, whole, &tail, chunk)?;
                fill(s, 2, &long[..at], chunk, false)?;
                fill_from(s, 2, at, &alt, chunk, false)?;
                let want = fill_last(s, 2, whole, &tail, chunk)?;
                let what = format!(
                    "a sequence that shared a start of {at} tokens, asked to go on within {ask}: refused (it holds its sliding layers' blocks from the window before that start, none earlier) and as it was ({held} held); from its end, {} more",
                    tail.len()
                );
                fails += same(&what, &want, &sharer)? | (held != whole) as usize;
            } else {
                let sharer = fill_last(s, 1, to, &tail, chunk)?;
                fill(s, 2, &long[..at], chunk, false)?;
                let cut = s.resume(2, to)?;
                let want = fill_last(s, 2, to, &tail, chunk)?;
                let what = format!(
                    "a sequence that shared a start of {at} tokens, asked to go on within {ask}: from {to}, the start of the block both hold, then {} others",
                    tail.len()
                );
                fails += same(&what, &want, &sharer)? | (cut != to) as usize;
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
            let finder = fill_last(s, 1, got, &alt, chunk)?;
            fill(s, 2, &p, chunk, false)?;
            steps(s, 2, p.len(), &g[..got - p.len()])?;
            let want = fill_last(s, 2, got, &alt, chunk)?;
            fails += same(&format!("a start of 48 tokens found by name after the sequence that ran it has ended, then {} others", alt.len()), &want, &finder)?;
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
                let finder = fill_last(s, 1, at, &alt, chunk)?;
                fill(s, 2, &long[..at], chunk, false)?;
                let want = fill_last(s, 2, at, &alt, chunk)?;
                let what = format!(
                    "a start of {at} tokens found by name while the sequence that ran it holds {} (it gave back blocks a step there reads; they wait by their names), then {} others",
                    long.len(),
                    alt.len()
                );
                fails += same(&what, &want, &finder)?;
                s.release(&[1, 2])?;
                let source = fill_last(s, 0, long.len(), &tail, chunk)?;
                fill(s, 2, &long, chunk, false)?;
                let alone = fill_last(s, 2, long.len(), &tail, chunk)?;
                fails += same(&format!("the sequence that ran it, gone on with {} more", tail.len()), &alone, &source)?;
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
    // another from `start`, the last one's sample, then the reply's tokens a step at a time. Where the engine computes experts on the host the scheduler's passes and these
    // are not the same calls of the expert cache (its places' state differs), so a token may differ where the row's lead is a near tie: such a token is taken as the
    // scheduler's (`Picks`), and one past `CLEAR_LEAD` is not
    struct Picks {
        tokens: Vec<u32>,
        leads: Vec<f32>,
    }
    let replay_in = |s: &mut dyn ModelStep, slot: i32, start: usize, chunks: &[&[u32]], reply: &[u32]| -> Result<Picks> {
        let (mut at, mut first) = (start, None);
        for (i, c) in chunks.iter().enumerate() {
            room(s, slot, at + c.len() + 1)?;
            first = s.prefill(&[slot], at, c, i + 1 == chunks.len())?;
            at += c.len();
        }
        let mut out = Picks { tokens: vec![first.ok_or_else(|| Error("no sample after a chunk".into()))?.token], leads: Vec::new() };
        if host {
            out.leads.push(lead_of(&s.logits(0)?));
        }
        for (i, &t) in reply[..reply.len() - 1].iter().enumerate() {
            room(s, slot, at + i + 2)?;
            out.tokens.push(s.decode(&[Row::new(t, at + i, &[slot])])?[0].token);
            if host {
                out.leads.push(lead_of(&s.logits(0)?));
            }
        }
        Ok(out)
    };
    // (whether the scheduler's reply is the passes' by hand: token for token, or where the engine computes experts on the host every token apart a near tie of the row's;
    // and in words)
    let as_by_hand = |by_hand: &Picks, r: &[u32]| -> bool {
        by_hand.tokens.len() == r.len()
            && by_hand.tokens.iter().zip(r).enumerate().all(|(j, (a, b))| a == b || (host && by_hand.leads.get(j).is_some_and(|&l| l <= CLEAR_LEAD)))
    };
    let told = |by_hand: &Picks, r: &[u32]| -> String {
        let apart = by_hand.tokens.iter().zip(r).filter(|(a, b)| a != b).count();
        if by_hand.tokens == r {
            "is the same passes' by hand, token for token".to_string()
        } else if as_by_hand(by_hand, r) {
            format!(
                "is the same passes' by hand but at {apart} tokens, each where the row's lead is {CLEAR_LEAD} or less (experts computed on the host: bits not held, a near tie taken either way)"
            )
        } else {
            format!("{r:?} against {:?} by hand", by_hand.tokens)
        }
    };
    let replay = |s: &mut dyn ModelStep, start: usize, chunks: &[&[u32]], reply: &[u32]| replay_in(s, 0, start, chunks, reply);
    // (the first turn left the prompt and its reply but the last token in the slot)
    let held = p.len() + r1.len() - 1;
    let first = as_by_hand(&replay(s, 0, &pieces(&p, 0, chunk, every), &r1)?, &r1);
    let second = replay(s, held, &pieces(&p2, held, chunk, every), &r2)?;
    s.release(&[0])?;
    let ok = hit == held && first && as_by_hand(&second, &r2);
    fails += !ok as usize;
    println!(
        "{} a conversation's second turn through the scheduler: it went on from {hit} of its {} prompt tokens (the slot held {held}); its reply {}{}",
        if ok { "PASS" } else { "FAIL" },
        p2.len(),
        told(&second, &r2),
        if first { "" } else { " (the first turn's is not)" }
    );
    // the turn that leaves the reply out: from the prompt's end where the model can (two passes by hand: the prompt, the rest), else from its first token (one)
    let should = if cuts || marks { q.len() } else { 0 };
    let by_hand = if hit_q > 0 {
        replay(s, 0, &[pieces(&q, 0, chunk, every), pieces(&q2, q.len(), chunk, every)].concat(), &rq)?
    } else {
        replay(s, 0, &pieces(&q2, 0, chunk, every), &rq)?
    };
    s.release(&[0])?;
    let ok = hit_q == should && as_by_hand(&by_hand, &rq);
    fails += !ok as usize;
    println!(
        "{} a second turn that leaves the reply out: it went on from {hit_q} of its {} prompt tokens (it should from {should}); its reply {}",
        if ok { "PASS" } else { "FAIL" },
        q2.len(),
        told(&by_hand, &rq)
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
    let none = || Picks { tokens: Vec::new(), leads: Vec::new() };
    let by_hand = if hit_y > 0 && !cuts {
        let first = as_by_hand(&replay(s, 0, &pieces(&xa, 0, chunk, every), &rx)?, &rx);
        let y = replay_in(s, 1, 0, &pieces(&yb, 0, chunk, every), &ry)?;
        if first {
            y
        } else {
            none()
        }
    } else if hit_y > 0 {
        let chunks = pieces(&xa, 0, chunk, every);
        let first = as_by_hand(&replay(s, 0, &chunks, &rx)?, &rx);
        let got = s.share(0, 1, xs.len())?;
        let y = replay_in(s, 1, got, &[&yb[got..]], &ry)?;
        if first && got == hit_y {
            y
        } else {
            none()
        }
    } else {
        replay(s, 0, &pieces(&yb, 0, chunk, every), &ry)?
    };
    s.release(&[0, 1])?;
    let ok = hit_y == should && as_by_hand(&by_hand, &ry);
    fails += !ok as usize;
    println!(
        "{} a conversation that begins as another does, through the scheduler: {hit_y} of its {} prompt tokens shared (it should {should}); its reply {}",
        if ok { "PASS" } else { "FAIL" },
        yb.len(),
        told(&by_hand, &ry)
    );
    s.trace_choices(false);
    if fails > 0 {
        return err(format!("keepcheck {case}: {fails} failed"));
    }
    println!("keepcheck {case}: PASS");
    Ok(())
}

/// `samplecheck CASE [--users N] [--tokens T] [--level L] [--vocab V]` (development, R4a #138: sampling on the GPU): the case's random model served by the scheduler to N requests at
/// once whose draws differ (greedy and drawn with nothing on the row; a JSON schema's grammar greedy and drawn; the repetition, presence and frequency penalties; a bias; an allowed list;
/// a least length), the rows their draws see made on the GPU (`sampler::Proc`): every request's tokens and log-probabilities against the host's draws from the whole rows of the same
/// steps (GLYD_ENGINE_HOST_SAMPLING's), a grammar's document against its schema, the rows each request's draws read alone (none, but where a top-p or a min-p over the whole row cuts
/// past the 64 largest), and each request at once against its run alone: token for token where a step's rows are each their own pass's bits (prompts one a pass,
/// GLYD_ENGINE_BATCH_PREFILL=0, as `usercheck` holds them, and 16 users or fewer: the rows a pass keeps the bits of; more are counted here), else counted (a prompt run beside others
/// rounds its rows' values otherwise, #528, and a drawn token may then differ). Then the host's time a step both ways, on this model and for 8 and 64 rows of a vocabulary of V (262,144 by default: the rows of
/// requests with the penalties, a bias and a top-k, half of them an allowed list of half the tokens) read and drawn on the host, or made on the GPU and drawn from their candidates.
pub fn sample_check(a: &crate::args::Args) -> Result<()> {
    use crate::conv::Gen;
    use crate::sampler::{Lp, Params};
    use crate::sched::{Ev, Finish, Reply, Sched, Submit};
    let case = a.need(1, "CASE")?;
    let level = a.flag("level").unwrap_or("penguin");
    let users: usize = a.num("users", 8usize)?;
    let tokens: usize = a.num("tokens", 24usize)?;
    let (mut e, _) = random_dyn_routed(case, level, None, "samplecheck", None)?;
    let v = e.desc().vocab;
    // (the end tokens: the model's, else its last token, which every request then takes as a stop token)
    let ends: Vec<u32> = e.desc().eos.iter().filter(|&&t| t >= 0 && (t as usize) < v).map(|&t| t as u32).collect();
    let (ends, stop) = if ends.is_empty() { (vec![v as u32 - 1], vec![v as u32 - 1]) } else { (ends, Vec::new()) };
    // (a vocabulary for the grammar: the 256 bytes, then words of their own, the end tokens special)
    let mut bytes: Vec<Vec<u8>> = (0..v).map(|t| if t < 256 { vec![t as u8] } else { format!("w{t}").into_bytes() }).collect();
    for &t in &ends {
        bytes[t as usize] = [&[toktrie::TokTrie::SPECIAL_TOKEN_MARKER][..], b"<|end|>"].concat();
    }
    let g = crate::constrain::Grammars::new(bytes.clone(), &ends, v).map_err(Error)?;
    let schema = r#"{"type":"object","properties":{"city":{"enum":["Paris","Rome","Oslo"]},"days":{"type":"integer","minimum":1,"maximum":9},"sunny":{"type":"boolean"}},"required":["city","days","sunny"],"additionalProperties":false}"#;
    let c = g.compile(&crate::constrain::Spec::JsonSchema(schema.into())).map_err(Error)?;
    let allowed: Vec<u32> = (0..v as u32).step_by(5).chain(ends.iter().copied()).collect();
    let d = Params::default();
    let kinds: Vec<(&str, Params, Option<usize>)> = vec![
        ("greedy", Params { temperature: 0.0, ..d.clone() }, None),
        ("top-k 20", Params { temperature: 0.9, top_k: 20, seed: 1, ..d.clone() }, None),
        ("a JSON schema, greedy, log-probabilities", Params { temperature: 0.0, constraint: Some(c.clone()), ..d.clone() }, Some(3)),
        (
            "a JSON schema, top-p 0.95, repetition 1.1",
            Params { temperature: 0.8, top_p: 0.95, repetition_penalty: 1.1, constraint: Some(c.clone()), seed: 2, ..d.clone() },
            None,
        ),
        (
            "repetition 1.3, presence 0.5, top-p 0.9",
            Params { temperature: 1.0, top_p: 0.9, repetition_penalty: 1.3, presence_penalty: 0.5, seed: 3, ..d.clone() },
            None,
        ),
        (
            "a bias, frequency 0.4, greedy, log-probabilities",
            Params { temperature: 0.0, logit_bias: vec![(5, 4.0), (100, -100.0), (v as u32 - 9, 2.5)], frequency_penalty: 0.4, ..d.clone() },
            Some(2),
        ),
        ("an allowed list, min-p 0.1", Params { temperature: 1.0, min_p: 0.1, allowed: allowed.clone(), seed: 4, ..d.clone() }, None),
        ("a least length, repetition 1.05, the whole row", Params { temperature: 1.0, min_tokens: 10, repetition_penalty: 1.05, seed: 5, ..d.clone() }, None),
    ];
    let prompts: Vec<Vec<u32>> =
        (0..users).map(|k| (0..(5 + 7 * (k % 9)) as u32).map(|i| 3 + (i * 7919 + k as u32 * 104_729) % (v as u32 - 3)).collect()).collect();
    let reply = |u: usize| Reply {
        params: kinds[u % kinds.len()].1.clone(),
        max_new: tokens,
        ignore_eos: false,
        logprobs: kinds[u % kinds.len()].2,
        stop_ids: stop.clone(),
    };
    type Got = (Vec<u32>, Vec<Lp>, Finish);
    // the requests `which` through the scheduler at once: each one's tokens, log-probabilities and end, with the scheduler's figures
    let run = |e: &mut Dyn, which: &[usize]| -> Result<(Vec<Got>, crate::sched::Stats)> {
        let Some((k, step)) = e.as_step(users)? else { return err(format!("{case}: no scheduler's step")) };
        let mut s = Sched::new(step, k).keeping(false);
        let rxs: Vec<_> = which
            .iter()
            .map(|&u| {
                let (tx, rx) = std::sync::mpsc::channel();
                s.submit(Submit {
                    prompt: prompts[u].clone(),
                    reply: reply(u),
                    out: tx,
                    abort: Default::default(),
                    resumed: Default::default(),
                    cached: Default::default(),
                    created: Default::default(),
                    salt: 0,
                    pin: 0,
                    images: Vec::new(),
                });
                rx
            })
            .collect();
        while !s.idle() {
            s.tick()?;
        }
        let got = rxs
            .iter()
            .map(|r| {
                let (mut toks, mut lps) = (Vec::new(), Vec::new());
                loop {
                    match r.recv().map_err(|_| Error("the scheduler dropped a sequence".into()))? {
                        Ev::Token(t) => toks.push(t),
                        Ev::Lp(l) => lps.push(l),
                        Ev::End(f) => return f.map(|f| (toks, lps, f)).map_err(|e| Error(e.msg)),
                    }
                }
            })
            .collect::<Result<Vec<Got>>>()?;
        Ok((got, s.stats))
    };
    let all: Vec<usize> = (0..users).collect();
    e.top.off = false;
    let (gpu, st_gpu) = run(&mut e, &all)?;
    e.top.off = true;
    let (host, st_host) = run(&mut e, &all)?;
    e.top.off = false;
    let mut fails = 0;
    let strict = std::env::var_os("GLYD_ENGINE_BATCH_PREFILL").is_some_and(|v| v == "0") && users <= 16;
    // (the draws that never need their row: the largest, a top-k within the candidates, a draw over the whole row from its buckets)
    let never = |p: &Params| p.temperature <= 0.0 || (1..=crate::kern::TOPK).contains(&p.top_k) || (p.top_k == 0 && p.top_p >= 1.0 && p.min_p == 0.0);
    let (mut like_alone, mut read_alone) = (0, 0);
    let lp_off = |a: &Lp, b: &Lp| {
        let tops = a.top.iter().zip(&b.top).map(|(x, y)| if x.0 == y.0 { (x.1 - y.1).abs() } else { f64::INFINITY }).fold(0f64, f64::max);
        if a.token != b.token || a.top.len() != b.top.len() {
            f64::INFINITY
        } else {
            (a.logprob - b.logprob).abs().max(tops)
        }
    };
    for u in 0..users {
        let (alone, st_alone) = run(&mut e, &[u])?;
        let (name, g_, h_, a_) = (kinds[u % kinds.len()].0, &gpu[u], &host[u], &alone[0]);
        let worst = g_.1.iter().zip(&h_.1).map(|(x, y)| lp_off(x, y)).fold(0f64, f64::max);
        let mut why = Vec::new();
        like_alone += (g_.0 == a_.0) as usize;
        read_alone += st_alone.row_reads;
        if g_.0 != a_.0 && strict {
            why.push(format!("at once {:?} against alone {:?}", g_.0, a_.0));
        }
        if st_alone.row_reads > 0 && never(&kinds[u % kinds.len()].1) {
            why.push(format!("alone, {} of its draws read the row", st_alone.row_reads));
        }
        if g_.0 != h_.0 {
            why.push(format!("on the GPU {:?} against the host's {:?}", g_.0, h_.0));
        }
        if g_.1.len() != h_.1.len() || worst >= 1e-5 {
            why.push(format!("log-probabilities {} against {} of the host's, {worst:.2e} apart at most", g_.1.len(), h_.1.len()));
        }
        // (a grammar's reply that ended: its text is a document of the schema)
        if kinds[u % kinds.len()].1.constraint.is_some() && g_.2 == Finish::Stop {
            let text: Vec<u8> = g_.0.iter().flat_map(|&t| bytes[t as usize].clone()).collect();
            let ok = serde_json::from_slice::<serde_json::Value>(&text).ok().is_some_and(|j| {
                j["city"].as_str().is_some_and(|c| ["Paris", "Rome", "Oslo"].contains(&c))
                    && j["days"].as_i64().is_some_and(|d| (1..=9).contains(&d))
                    && j["sunny"].is_boolean()
                    && j.as_object().is_some_and(|o| o.len() == 3)
            });
            if !ok {
                why.push(format!("the grammar's text {:?} is no document of its schema", String::from_utf8_lossy(&text)));
            }
        }
        fails += !why.is_empty() as usize;
        println!(
            "{} request {u} ({name}; {} tokens, {:?}{}; alone {} rows read): {}",
            if why.is_empty() { "PASS" } else { "FAIL" },
            g_.0.len(),
            g_.2,
            if g_.1.is_empty() { String::new() } else { format!(", log-probabilities {worst:.1e} from the host's") },
            st_alone.row_reads,
            match (why.is_empty(), g_.0 == a_.0) {
                (true, true) => "at once the host's draws and the run alone".to_string(),
                (true, false) => "at once the host's draws; alone another reply (a step's rows rounded otherwise: counted)".to_string(),
                _ => why.join("; "),
            }
        );
    }
    println!(
        "{like_alone} of {users} replies at once the reply alone ({}); the requests alone read {read_alone} rows",
        if strict {
            "prompts one a pass and 16 users or fewer: each required"
        } else {
            "a prompt beside others or a step of more than 16 rows rounds the rows' values: counted"
        }
    );
    let per = |st: &crate::sched::Stats| (st.before_ns + st.after_ns) as f64 / 1e3 / st.steps.max(1) as f64;
    println!(
        "{users} requests at once on the random {case} model (a vocabulary of {v}): rows read {} with the rows made on the GPU, {} on the host; the host's time a step {:.1} us and {:.1} us ({} and {} steps)",
        st_gpu.row_reads,
        st_host.row_reads,
        per(&st_gpu),
        per(&st_host),
        st_gpu.steps,
        st_host.steps
    );
    if st_host.row_reads == 0 {
        println!("FAIL the host's run read no row (GLYD_ENGINE_HOST_SAMPLING's draws were not the host's)");
        fails += 1;
    }
    // the host's time a step for rows of a vocabulary's size (--vocab 0: none)
    let tv: usize = a.num("vocab", 262_144usize)?;
    for rows in [8usize, 64].into_iter().filter(|_| tv > 0) {
        let (host_ms, gpu_ms) = time_rows(&e, rows, tv)?;
        println!(
            "the host's time a step for {rows} rows of {tv} (the penalties over a prompt of 2,000, a bias, a top-k of 40; every other one an allowed list of half the tokens): {host_ms:.3} ms reading the rows and drawing on the host, {gpu_ms:.3} ms with the rows made on the GPU ({:.3} ms less)",
            host_ms - gpu_ms
        );
    }
    if fails > 0 {
        return err(format!("{fails} checks failed"));
    }
    println!("PASS samplecheck {case}: {users} requests");
    Ok(())
}

/// `samplecheck`'s time: for `rows` requests of a vocabulary of `v` on `e`'s GPU, each with the repetition, presence and frequency penalties over a prompt of 2,000 tokens, a bias and a
/// top-k of 40 (every other one an allowed list of half the tokens too, as a grammar's mask), the host's time a step (ms, the mean of 20 after 3): every row read from the GPU and drawn
/// on the host (`Sampler::sample`, as a step's rows are read with GLYD_ENGINE_HOST_SAMPLING=1), and the rows made on the GPU (`ProcGpu`, the model's candidates first as a step's) and
/// drawn from their candidates. The same random logits every step; the same tokens both ways.
fn time_rows(e: &Dyn, rows: usize, v: usize) -> Result<(f64, f64)> {
    use crate::dev::Buf;
    use crate::sampler::{Params, Sampler};
    let dev = &e.dev;
    let kern = &crate::kern::Kern::load(dev)?;
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    let mut rnd = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let bits: Vec<u16> = (0..rows * v).map(|_| ((((rnd() % 2000) as f32 / 100.0) - 10.0).to_bits() >> 16) as u16).collect();
    let logits = Buf::new(2 * rows * v)?;
    logits.write(0, &bits)?;
    let samplers = || -> Vec<Sampler> {
        (0..rows)
            .map(|r| {
                let prompt: Vec<u32> = (0..2000u32).map(|i| (i * 7919 + r as u32 * 104_729) % v as u32).collect();
                let p = Params {
                    temperature: 0.8,
                    top_k: 40,
                    repetition_penalty: 1.1,
                    presence_penalty: 0.3,
                    frequency_penalty: 0.2,
                    logit_bias: vec![(7, 3.0), (v as u32 / 2, -100.0)],
                    allowed: if r % 2 == 1 { (0..v as u32).step_by(2).collect() } else { Vec::new() },
                    seed: r as u64,
                    ..Params::default()
                };
                Sampler::new(p, &prompt)
            })
            .collect()
    };
    let (steps, warm) = (20usize, 3usize);
    // the host's: each row read and drawn
    let (mut host_s, mut host_toks) = (samplers(), Vec::new());
    let mut host_t = 0.0;
    for step in 0..steps + warm {
        let t = std::time::Instant::now();
        dev.sync()?;
        let mut toks = Vec::new();
        for (r, s) in host_s.iter_mut().enumerate() {
            let mut w = vec![0u16; v];
            logits.read(2 * r * v, &mut w)?;
            let row: Vec<f32> = w.into_iter().map(|b| f32::from_bits((b as u32) << 16)).collect();
            toks.push(s.sample(&row));
        }
        if step >= warm {
            host_t += t.elapsed().as_secs_f64();
        }
        host_toks.push(toks);
    }
    // the GPU's: the rows made there, their candidates read, drawn from them
    let (tok, val) = (Buf::zeroed(4 * rows * crate::kern::TOPK)?, Buf::zeroed(4 * rows * crate::kern::TOPK)?);
    let mut host = vec![0u8; 8 * rows * crate::kern::TOPK];
    let mut g = crate::kern::ProcGpu::new(v);
    let (mut gpu_s, mut gpu_t) = (samplers(), 0.0);
    for (step, want) in host_toks.iter().enumerate() {
        let t = std::time::Instant::now();
        let res: Vec<crate::sampler::Resolved> =
            gpu_s.iter_mut().map(|s| s.proc().map(|p| p.resolve(v))).collect::<Option<_>>().ok_or_else(|| Error("a request with no changes".into()))?;
        let list: Vec<(usize, &crate::sampler::Resolved)> = res.iter().enumerate().collect();
        // SAFETY: `rows` rows of v bf16 logits and TOPK candidates' room for each; the stream is synchronized before the candidates are read and before the next step's queue
        unsafe {
            kern.topk_rows(dev.stream, logits.ptr, v as i32, rows as u32, tok.ptr, val.ptr)?;
            g.queue(kern, dev, &list, logits.ptr, tok.ptr, val.ptr)?;
            crate::dev::copy_d2h_async(host.as_mut_ptr() as *mut _, tok.ptr, 4 * rows * crate::kern::TOPK, dev.stream)?;
            crate::dev::copy_d2h_async(host.as_mut_ptr().add(4 * rows * crate::kern::TOPK) as *mut _, val.ptr, 4 * rows * crate::kern::TOPK, dev.stream)?;
        }
        dev.sync()?;
        let mut toks = Vec::new();
        for (r, s) in gpu_s.iter_mut().enumerate() {
            // SAFETY: the candidates were copied above, the stream synchronized since
            let cands = unsafe { crate::kern::read_candidates(host.as_ptr(), 0, 4 * rows * crate::kern::TOPK, r) };
            toks.push(s.sample_proc(&cands, None).ok_or_else(|| Error("a draw from the candidates was not shown".into()))?);
        }
        if step >= warm {
            gpu_t += t.elapsed().as_secs_f64();
        }
        if &toks != want {
            return err(format!("step {step}: the tokens drawn from the GPU's candidates {toks:?}, the host's {want:?}"));
        }
    }
    Ok((host_t * 1e3 / steps as f64, gpu_t * 1e3 / steps as f64))
}

/// Replies' tokens and their sample logits' bits (each reply's in turn, its length first) as one SHA-256, in hex.
fn replies_digest(tokens: &[Vec<u32>], logits: &[Vec<u32>]) -> String {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    for v in tokens.iter().chain(logits) {
        h.update((v.len() as u64).to_le_bytes());
        v.iter().for_each(|x| h.update(x.to_le_bytes()));
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// `usercheck CASE [--level L] [--users N] [--tokens T] [--dir DIR [--places N [--mode M] [--device-plan] | --ctx plan]] | --ctx C] [--prompt-tokens P] [--stagger D] [--pack P [--precise]]` (gemma4, qwen35, muse, kimi, glm, qwen4x, qwen4x-s2): the case's random model on the
/// description executor (or the model in DIR, with `--places` its experts in the expert cache, with `--ctx plan` as `serve --users N` plans and loads it (a model larger than the GPU: its experts in the expert cache, every user's keys, values and states first), with `--pack` Condor's from its pack of codebook experts; with `--ctx` a context of C tokens; with `--prompt-tokens` prompts of P (k + 1) / N tokens), each of N prompts replied to alone as `run` replies (the reference), then by the scheduler (src/dynamic/rows.rs) alone, then all at once. Alone through
/// the scheduler every reply is the reference, token for token; at once too where GLYD_ENGINE_BATCH_PREFILL=0 (prompts one a pass, decode steps of the rows that decode: 16 or fewer here, whose bits
/// are each row's own pass's). With prompts run together the replies that differ are counted, not refused (a step of more than 16 rows is another kernel's; a random model's near ties turn on its last
/// bits: `roundcheck` measures the logits); their tokens and sample logits are printed as one digest, which runs held to the same bits compare (GLYD_TABLES=gpu against
/// host). The round is one step of the rows that decode and the prompts' chunks unless GLYD_ENGINE_ONE_STEP=0, and some step must then have held both.
/// Then the prompts at once twice more, with no graph at all (every pass eager) and with every graph on and every pass that is no whole graph's in pieces
/// (src/dynamic/pieces.rs): the replies the same token for token and each sample's logit to the bit, and where pieces can run, some pass replayed them. With
/// GLYD_ENGINE_OVERLAP=1, where steps are launched ahead, the prompts at once launched ahead and not: the same tokens and logits. With `--places` every run (each reference, alone,
/// at once, each side of each comparison) begins from the expert cache's cold state with its planner's times held (`ExpertStore::cold`, `hold_costs`), so two runs of the
/// same calls find the same experts in places and plan alike, in every mode. Where the engine computes experts on the host (`host_experts`: a cache whose mode gives the
/// host pairs, the codebook cache), at once is not alone's calls (the host's share of a call turns on what shares it and on the places' state, and the host sums a pair in
/// another order than the GPU): with prompts one a pass the replies at once are held to alone row for row at the kind's lossless bound (`replies_beside`), not bit for bit.
/// With --stagger D the prompts then come D scheduler rounds apart, as a server's requests do: with prompts one a pass every reply is the reply alone, and with no mixture of
/// experts each reply is the same when every other prompt's tokens are changed (a row's sums may turn on its passes' row counts, never on the other rows' content).
///
/// With a page budget too small for all of them at once (GLYD_ENGINE_PAGES, R4d) the prompts are run at once three more times: sequences sent back for room swapped to the host wherever they can be,
/// swapped where that costs less than their recompute (the default), and run again from their tokens only; each run's replies against the references, its sequences sent back and swapped, and each
/// request's first token and the mean time between its tokens. Swapped sequences go on with the cache they had, bit for bit: with prompts one a pass their replies must be the references.
///
/// With `--draft K` (the draft heads' cases: qwen4x, glm-dsa, glm-dsa-f8, deepseek-v4, deepseek-v41): `users_drafted`.
pub fn users_check(a: &crate::args::Args) -> Result<()> {
    use crate::conv::Gen;
    use crate::sched::{Ev, Finish, Reply, Sched, Submit, SwapMode};
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    let case = a.need(1, "CASE")?;
    let level = a.flag("level").unwrap_or("penguin");
    let users: usize = a.num("users", 4usize)?;
    let tokens: usize = a.num("tokens", 24usize)?;
    if let Some(k) = a.flag("draft") {
        let k = k.parse().map_err(|_| Error(format!("--draft {k}: not a number")))?;
        return users_drafted(case, level, users, tokens, k);
    }
    // (--ctx plan: the random model at the context the planner gives it on this GPU, each prompt filling it to within half a chunk with its reply: the long-context check;
    // with --kv12 the plan made for 12-bit pages wherever the planner makes one (GLYD_KV=lossless asks for them, and GLYD_KV12_ALWAYS takes their plan also where a
    // server's would be bf16's, `sizing::twelve_always`), so the pages are held to the bytes planned for them)
    let kv12 = a.has("kv12");
    if kv12 && a.flag("ctx") == Some("plan") {
        std::env::set_var("GLYD_KV", "lossless");
        std::env::set_var("GLYD_KV12_ALWAYS", "1");
    }
    // (--ctx plan: the plan `serve` makes of the random model for these users on this GPU, which the load takes (its context, chunk, layout, sequences and their bytes:
    // `take_plan`); the GPU's memory the load and the step's sequences take is held to the plan's figure, and the users to the plan's)
    // (with --dir DIR, the plan `serve DIR --users N` makes of that model on this machine, which the load takes too: its experts in the expert cache where the GPU
    // does not hold them, every user's keys, values and states first and the places what is left)
    let plan = match (a.flag("ctx") == Some("plan"), a.flag("dir")) {
        (true, Some(d)) => Some(planned_dir(Path::new(d), users)?),
        (true, None) => Some(planned_for(case, level, users)?),
        _ => None,
    };
    // (whether the plan is of the pages this run holds. The planner plans no 12-bit pages where the description has no ring in pools they take (heads of 64, the
    // compressed attention's entries: `rows::twelve_bit_rings`): there --kv12 runs them at the bf16 plan's context in pools of what the GPU has free, and its
    // memory is not held to that plan, which is of other pages)
    let own_plan = plan.as_ref().is_some_and(|lp| !kv12 || lp.kv_format() == "12-bit");
    let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
    if let Some(lp) = &plan {
        println!(
            "{} at the planner's plan for this GPU{}: {} tokens of context for each of {} users ({:.1} MiB of keys, values and states planned in {}{}), prompts of up to {} and {tokens} made",
            a.flag("dir").map_or(format!("the random {case} model"), |d| format!("the model in {d}")),
            if kv12 { " with GLYD_KV=lossless" } else { "" },
            lp.context,
            lp.users,
            mib(lp.kv),
            lp.kv_format(),
            if own_plan { "" } else { ": the planner plans no 12-bit pages for this route; the check runs them in pools of what the GPU has free" },
            lp.context.saturating_sub(tokens + 32)
        );
        println!("  the plan: {}", lp.text());
    }
    // (the GPU's free memory before the load, its context made: what the load and the sequences take of it is the plan's at most)
    let free_before = match &plan {
        Some(_) => Some(crate::dev::Dev::new(0)?.mem_info()?.0 as u64),
        None => None,
    };
    let planned = plan.as_ref().map(|p| p.context);
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
    let (mut e, routers): (Dyn, Option<Routers>) = match (a.flag("dir"), places) {
        // (`--dir DIR --ctx plan`: the model in DIR as `serve` loads it, by its plan (src/dynamic/split.rs `open_plan`: the plan's places, sequences and their bytes;
        // `--mode` the misses', copy by default): its GPU memory held to the plan below, its users to the plan's)
        (Some(d), _) if plan.is_some() => {
            let lp = plan.as_ref().expect("a plan");
            let offload = crate::offload::Offload { mode, device: a.has("device-plan"), ..Default::default() };
            match crate::dynamic::split::open_plan(Path::new(d), lp, offload)? {
                crate::dynamic::split::Run::One(e) => {
                    // (the planner's times held where they start, so that two runs from one state plan alike: each run below starts from the cold cache, `Sched::cold`)
                    e.stores().iter().for_each(|st| st.hold_costs());
                    println!(
                        "{d}: as `serve` loads it by its plan, {}, mode {}",
                        lp.experts.map_or("every expert on the GPU".to_string(), |c| format!(
                            "its experts in the expert cache, {} places a layer of {}",
                            c.slots, c.experts
                        )),
                        mode.name()
                    );
                    let r = routers_of(Path::new(d), &e)?;
                    (e, r)
                }
                _ => return err("the plan cuts the model over GPUs or across them: --dir with --ctx plan checks a plan on one GPU"),
            }
        }
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
                    let r = routers_of(Path::new(d), &e)?;
                    (e, r)
                }
                _ => return err("the model on one GPU was asked for"),
            }
        }
        (Some(d), _) => {
            let o = DynOpts { ctx: (len(users - 1) + tokens + 64).next_multiple_of(256), chunk: 256, graph: true, device: 0, layout };
            // (--pack P [--precise]: Condor's experts from its pack of codebook rows in the expert cache)
            let e = match a.flag("pack") {
                Some(p) if case == "qwen4x" => crate::dynamic::qwen4x::load_pack(Path::new(d), Path::new(p), a.has("precise"), users, &o)?,
                Some(_) => return err("--pack: the qwen4x case's pack of experts"),
                None => load_dyn(case, Path::new(d), &o, level)?,
            };
            let r = routers_of(Path::new(d), &e)?;
            (e, r)
        }
        // (--ctx plan: as the plan says, its sequences' bytes the plan's)
        (None, _) if plan.is_some() => {
            let lp = plan.as_ref().expect("a plan");
            let (mut e, r) =
                random_dyn_with(case, level, "usercheck", &DynOpts { ctx: lp.context, chunk: lp.chunk, graph: true, device: 0, layout: Some(lp.layout) })?;
            if own_plan {
                e.take_plan(lp.users, lp.kv);
            }
            (e, r)
        }
        // (--ctx N: the random model with a context of N tokens, for a long prompt's check: --prompt-tokens)
        (None, _) => random_dyn_routed(
            case,
            level,
            layout,
            "usercheck",
            a.flag("ctx").map(|v| v.parse().map_err(|_| Error(format!("--ctx {v}: not a number")))).transpose()?,
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
    if kv12 {
        let t = crate::dynamic::kv12::calibrate(&mut e, &prompts, "usercheck")?;
        e.set_kv12(Some(t));
    }
    // (what a server says of its keys and values, /v1/glyd/info's `precision.kv`, is read from the engine at load, before the pools are made: the format asked for, which the pools then hold)
    let kv_said = crate::conv::Gen::held(&e).map_or("", |h| h.kv_type);
    let (batchable, exact, host) = (e.batchable(), e.rows_exact(), host_experts(&e));
    // (the one-row steps that replayed their sequence's graph so far: the replies above, which are no scheduler's; the check below counts the scheduler's alone)
    let ones_before = e.one_replays.get();
    let Some((k, step)) = e.as_step(users)? else { return err(format!("{case}: no scheduler's step")) };
    let chunk = crate::sched::ModelStep::chunk(step);
    // (--ctx plan: the GPU memory the load and the step's sequences took, against the plan's (its weights, sequences and work: `planned_gpu_used_bytes`), and the users it holds)
    let held_to_plan = |when: &str| -> Result<bool> {
        let (Some(lp), Some(before)) = (&plan, free_before) else { return Ok(true) };
        if !own_plan {
            println!("  (GPU memory {when} is not held to the plan: the plan is of bf16 pages, this run's are 12-bit)");
            return Ok(true);
        }
        let used = before.saturating_sub(crate::dev::Dev::new(0)?.mem_info()?.0 as u64);
        let planned = lp.gpu_total() - lp.reserve;
        let ok = used <= planned;
        println!(
            "{} GPU memory {when}: {:.1} MiB taken, the plan {:.1} MiB ({:.1} of weights, {:.1} of keys, values and states, {:.1} of work)",
            if ok { "PASS" } else { "FAIL" },
            mib(used),
            mib(planned),
            mib(lp.weights_on_gpu()),
            mib(lp.kv),
            mib(lp.work)
        );
        Ok(ok)
    };
    let mut plan_fails = !held_to_plan("after the load and the step's sequences")? as usize;
    if let Some(lp) = plan.as_ref().filter(|lp| k < lp.users.min(users)) {
        println!("FAIL the step holds {k} sequences, where the plan gives {} of the {users} asked", lp.users);
        plan_fails += 1;
    }
    if let Some(lp) = plan.as_ref().filter(|lp| lp.users < users) {
        println!("FAIL the plan holds {} of the {users} users asked", lp.users);
        plan_fails += 1;
    }
    // (no finished sequence kept: the same prompts come again below, and each is to run from its first token, in passes with the others; where the engine computes experts
    // on the host, every row a token is chosen from recorded too, for the kind's bound)
    let mut s = Sched::new(step, k).keeping(false).recording(true).recording_rows(host);
    // (the prompts `gap` scheduler rounds apart: 0, all of them before the first round; from the expert cache's cold state, its planner's times held, so that every run
    // begins from one state and the same calls find the same experts in places)
    let ask = |s: &mut Sched<&mut dyn crate::sched::ModelStep>, ps: &[Vec<u32>], gap: usize| -> Result<Vec<Vec<u32>>> {
        s.cold()?;
        if let Some(c) = s.chosen.as_mut() {
            c.clear();
        }
        if let Some(r) = s.rows.as_mut() {
            r.clear();
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
    // (the last run's rows of logits each token was chosen from, each sequence's in the order the prompts were given: recorded where the engine computes experts on the host)
    type RowsOf = Vec<(Vec<f32>, Option<Hist>)>;
    let seen_rows = |s: &Sched<&mut dyn crate::sched::ModelStep>| -> Vec<RowsOf> {
        let mut by: std::collections::BTreeMap<u64, RowsOf> = Default::default();
        for (id, l, c) in s.rows.iter().flatten() {
            by.entry(*id).or_default().push((l.clone(), c.clone()));
        }
        by.into_values().collect()
    };
    // (where the engine computes experts on the host, replies beside others against alone row for row at the kind's bound (`replies_beside`): its line, and whether held;
    // each reply's rows must be as many as its tokens)
    let rows_held = |what: &str, alone: &[Seen], beside: &[Seen]| -> Result<bool> {
        let whole = |r: &[Seen]| r.len() == prompts.len() && r.iter().all(|(t, l)| t.len() == l.len());
        if !whole(alone) || !whole(beside) {
            println!("FAIL {what}: the rows of logits the tokens were chosen from were not recorded for every token");
            return Ok(false);
        }
        let (rows, parted) = replies_beside(alone, beside);
        let j = Judge::of(&rows, routers.as_ref())?;
        let ok = j.ok();
        j.print(&format!("{what}, row for row while the tokens are the same ({} rows)", rows.len()), ok);
        for (i, j, lead) in parted {
            println!(
                "  reply {i} parts from alone at its token {j}, where alone's lead is {lead:e}: {}",
                if lead > CLEAR_LEAD { "a clear lead" } else { "a near tie (the rest of the reply runs on other tokens: not compared)" }
            );
        }
        Ok(ok)
    };
    let mut fails = plan_fails;
    if kv_said != crate::plan::kv_format_of(kv12) {
        println!(
            "FAIL the engine says its keys and values are in {kv_said}, as the info reports them, where this run holds them in {}",
            crate::plan::kv_format_of(kv12)
        );
        fails += 1;
    }
    let mut alone_logits = Vec::new();
    let mut alone_rows: Vec<Seen> = Vec::new();
    for (i, p) in prompts.iter().enumerate() {
        let got = ask(&mut s, std::slice::from_ref(p), 0)?.remove(0);
        alone_logits.push(seen(&s).pop().unwrap_or_default());
        alone_rows.push((got.clone(), seen_rows(&s).pop().unwrap_or_default()));
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
    let together_logits = seen(&s);
    let same = together.iter().zip(&reference).filter(|(a, b)| a == b).count();
    let same_logits = together_logits.iter().zip(&alone_logits).filter(|(a, b)| a == b).count();
    let together_rows: Vec<Seen> = together.iter().cloned().zip(seen_rows(&s)).collect();
    // (with a page budget too small for all of them at once (GLYD_ENGINE_PAGES), sequences give their blocks back and run again from their tokens, the generated ones as a prompt: the
    // replies are then counted, not required to be the same)
    let squeezed = std::env::var_os("GLYD_ENGINE_PAGES").is_some();
    let one_a_pass = std::env::var_os("GLYD_ENGINE_BATCH_PREFILL").is_some_and(|v| v == "0");
    let strict = one_a_pass && !squeezed;
    let steps: Vec<u64> = s.context().sched[18..27].iter().map(|x| x.load(std::sync::atomic::Ordering::Relaxed)).collect();
    // (and the steps that held a prompt's chunk beside their rows that decode, which no graph of rows takes: `Dyn::pass` replays one for rows that decode alone)
    let mixed = s.context().sched[28].load(std::sync::atomic::Ordering::Relaxed);
    let resumed = s.context().sched[2].load(std::sync::atomic::Ordering::Relaxed);
    // (and of them those swapped to the host and copied back, whose copies' pages land in the pools' counts while others step)
    let swapped = s.context().sched[29].load(std::sync::atomic::Ordering::Relaxed);
    println!(
        "at once: {same} of {} replies the reply alone (prompts {}); {k} sequences, rows in one pass: {batchable}; decode steps by rows {steps:?}; sequences sent back for room {resumed} \
         ({swapped} swapped)",
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
            if let Some(r) = s.rows.as_mut() {
                r.clear();
            }
            let (got, first, gap, wall) = timed(&mut s, &prompts)?;
            let got_rows: Vec<Seen> = got.iter().cloned().zip(seen_rows(&s)).collect();
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
            if mode == SwapMode::Always
                && one_a_pass
                && host
                && !rows_held("swap always: replies against alone with prompts one a pass", &alone_rows, &got_rows)?
            {
                fails += 1;
            }
            if mode == SwapMode::Always && one_a_pass && !host && same < prompts.len() {
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
        // (and the pools did run short: with no sequence sent back for room the run shows nothing of what a short pool does)
        if std::env::var_os("GLYD_KV12_POOL").is_some() && resumed == 0 {
            println!("FAIL small pools (GLYD_KV12_POOL): no sequence was sent back for room, so the pools never ran short and the run does not show what they do then");
            fails += 1;
        }
    }
    println!("  at once against alone, logit for logit: {same_logits} of {} replies' every sample logit the same bits", prompts.len());
    // (the replies at once and their sample logits as one digest, which two runs held to the same bits compare: the tables an embedding kind reads by rows on the GPU
    // against on the host (GLYD_TABLES), whose rows are the same bits)
    println!("  at once, the replies' tokens and sample logits: digest {}", replies_digest(&together, &together_logits));
    // (where the engine computes experts on the host (an expert cache whose mode gives the host pairs, `host_experts`): which of a call's misses the host computes, and which are
    // copied or promoted, turns on the call's pairs and the places' state, and the host sums a pair in another order than the GPU, so a row's last bits turn on the rows beside
    // it: at once is held to alone at the kind's bound, row for row; runs of the same calls are held to the bit)
    if strict && host && !rows_held("replies at once against alone with prompts one a pass", &alone_rows, &together_rows)? {
        fails += 1;
    }
    if strict && !host && same < prompts.len() {
        println!("FAIL replies at once differ from alone with prompts one a pass");
        fails += 1;
    }
    if strict && !host && same == prompts.len() && same_logits < prompts.len() {
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
        let got_rows: Vec<Seen> = got.iter().cloned().zip(seen_rows(&s)).collect();
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
        if strict && host && !rows_held("staggered replies against alone with prompts one a pass", &alone_rows, &got_rows)? {
            fails += 1;
        }
        if strict && !host && alone < prompts.len() {
            println!("FAIL staggered replies differ from alone with prompts one a pass");
            fails += 1;
        }
        if exact && !squeezed && beside < prompts.len() {
            println!("FAIL a staggered reply changed with what the other prompts hold");
            fails += 1;
        }
    }
    // (decode steps of several rows where the replies' decoding can meet: the scheduler runs the prompts' chunks in the order they came (a step's `chunk` tokens, the
    // oldest prompt first: `Sched`'s picks), so a sequence decodes its tokens, one a pass, while the next prompt's passes run, all of them: where every next prompt takes
    // more passes than the tokens made (prompts that fill a long planned context, `--ctx plan`), no two of them decode together. On an A10, 8 users of qwen35 at its
    // planned 32,768 tokens (prompts of 4,089 to 32,712, each 4,089 past the one before) decoded one row a step, bf16 and 12-bit pages alike)
    let meet = (1..users).any(|k| len(k).div_ceil(chunk.max(1)) + 1 < tokens);
    if batchable && k > 1 && !meet {
        println!("decode steps of several rows: not asked (each prompt after the first takes more passes than the {tokens} tokens made)");
    }
    if batchable && k > 1 && meet && steps[2..].iter().all(|&x| x == 0) {
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
    // (--ctx plan: and after every run of the scheduler, what the step's graphs and swaps took too)
    fails += !held_to_plan("after the runs")? as usize;
    // (the sampler's candidates the step makes on the GPU (src/dynamic/rows.rs `Cands`), against its rows' logits)
    let swapped = s.stats.swapped;
    // (the scheduler's decode steps of one row in every run above)
    let one_steps = s.context().sched[19].load(std::sync::atomic::Ordering::Relaxed);
    s.stop_rows();
    drop(s);
    // (decode steps of several rows replay a CUDA graph of their padded rows where the description takes one (src/dynamic/rows.rs `rows_bucket`), every kind's (`capturable`),
    // up to the ladder's top (`rows_top`: the head's rows, or the expert cache's plan on the GPU's, 64, each row count its own); with prompts one a pass, a description with experts decodes a
    // sequence a step. A step of one row replays its sequence's own graph)
    let (replays, late, eager, top, ones) =
        (e.rows_replays.get(), e.rows_late.get(), e.rows_eager.get(), e.rows_graph_top(), e.one_replays.get() - ones_before);
    // (steps launched ahead, GLYD_ENGINE_OVERLAP=1, run as passes of parts: no graph of rows and no pieces there yet)
    let ahead = crate::sched::ModelStep::overlaps(&e);
    if crate::sched::overlap_asked() {
        println!("steps launched before the step before lands (GLYD_ENGINE_OVERLAP=1): {}", if ahead { "yes" } else { "no, this model's steps land first" });
    }
    let graphs = e.rows_graphs() && !(one_a_pass && !e.rows_exact()) && !ahead;
    // (where the engine computes experts on the host the runs above recorded every row's logits and traced the routers' choices, for the kind's bound (`recording_rows`), and a
    // traced pass runs eager (`Dyn::tracing`): no step of theirs replays a graph, so the run of graphs on below is the one counted)
    let traced = host;
    let by: Vec<String> = e.rows_replays_by_size().iter().map(|(r, n)| format!("{r}:{n}")).collect();
    println!(
        "decode steps of several rows replayed from a CUDA graph: {replays}, run eager: {eager} (graphs up to {top} rows; replays by the graph's rows {}), graphs captured \
         in a request's step: {late}{}; decode steps of one row replayed from their sequence's graph: {ones} of {one_steps}{}",
        by.join(" "),
        if graphs || k < 2 { "" } else { " (this description's steps of several rows run eager)" },
        if traced { " (runs that traced the routers' choices, every pass eager: the run of graphs on below is counted)" } else { "" }
    );
    // (a step of several rows that decode and nothing else: more of them than steps that held a prompt's chunk beside; else every one may have held one, as where each
    // prompt runs while the one before decodes, prompts that fill a long planned context)
    let alone_rows = steps[2..].iter().sum::<u64>() > mixed;
    if graphs && batchable && k > 1 && steps[2..].iter().any(|&x| x > 0) && !alone_rows {
        println!("decode steps of several rows alone: not asked (no more of them than the {mixed} steps that held a prompt's chunk beside)");
    }
    if !traced && graphs && batchable && k > 1 && alone_rows && replays == 0 {
        println!("FAIL no decode step of several rows replayed a graph");
        fails += 1;
    }
    // (every step of several rows that decode within the ladder replays its graph: where the users fit it, none runs eager)
    if !traced && graphs && k <= top && eager > 0 {
        println!("FAIL {eager} decode steps of several rows ran eager, where the graphs of up to {top} rows hold them");
        fails += 1;
    }
    // (a graph that could not be captured turns the description's graphs of several rows off: not a pass)
    if e.rows_lost.get() > 0 {
        println!("FAIL {} graphs of decode steps of several rows could not be captured (such steps then ran eager)", e.rows_lost.get());
        fails += 1;
    }
    if !traced && e.graphs_taken() && !ahead && steps[1] > 0 && ones == 0 {
        println!("FAIL no decode step of one row replayed its sequence's graph");
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
    // (a run of the prompts at once, its replies and its sample logits: launched ahead or not where the step can be, `overlapping`; its scheduler's decode steps of one row
    // in `run_ones`)
    let run_ones = std::cell::Cell::new(0u64);
    let run = |e: &mut Dyn, ahead: bool| -> Result<(Vec<Vec<u32>>, Vec<Vec<u32>>)> {
        let Some((_, step)) = e.as_step(users)? else { return err(format!("{case}: no scheduler's step")) };
        let mut s = Sched::new(step, k).keeping(false).recording(true).overlapping(ahead);
        let got = ask(&mut s, &prompts, 0)?;
        run_ones.set(s.context().sched[19].load(std::sync::atomic::Ordering::Relaxed));
        Ok((got, seen(&s)))
    };
    let same = |a: &(Vec<Vec<u32>>, Vec<Vec<u32>>), b: &(Vec<Vec<u32>>, Vec<Vec<u32>>)| {
        (a.0.iter().zip(&b.0).filter(|(x, y)| x == y).count(), a.1.iter().zip(&b.1).filter(|(x, y)| x == y).count())
    };
    if !squeezed && std::env::var_os("GLYD_KV12_POOL").is_none() && (pieces_on || e.graphs_taken()) {
        let eager = run(&mut e, ahead)?;
        e.pieces_for_check(true, false);
        let (before, rows_before, eager_before, ones_on_before) = (e.piece_replays.get(), e.rows_replays.get(), e.rows_eager.get(), e.one_replays.get());
        let on = run(&mut e, ahead)?;
        let (played, rows_played) = (e.piece_replays.get() - before, e.rows_replays.get() - rows_before);
        let (eager_on, ones_on, one_steps_on) = (e.rows_eager.get() - eager_before, e.one_replays.get() - ones_on_before, run_ones.get());
        e.pieces_for_check(false, false);
        let (same_t, same_l) = same(&on, &eager);
        // (pieces are of a full pass's rows, a step's chunk's: prompts that fit one chunk together make one pass of prompt tokens at most, which need take none (one-token
        // prompts are rows that decode: their pass is the decode steps' graph))
        let no_full = prompts.iter().map(Vec::len).sum::<usize>() <= chunk;
        let ok = same_t == prompts.len() && same_l == prompts.len() && (played > 0 || !pieces_on || ahead || no_full);
        println!(
            "{} graphs on against off: {same_t} of {} replies at once the eager passes', token for token, {same_l} logit for logit; {played} passes replayed their pieces, \
             {rows_played} decode steps their graph{}",
            if ok { "PASS" } else { "FAIL" },
            prompts.len(),
            if pieces_on && no_full {
                " (the prompts fit one step's chunk together: no pass need replay pieces)"
            } else if pieces_on {
                ""
            } else if !e.stores().is_empty() {
                " (no pieces with an expert cache: a prompt's chunk plans its calls on the host)"
            } else {
                " (no pieces here)"
            }
        );
        fails += !ok as usize;
        // (runs that traced the routers' choices: the run of graphs on held to what the checks above hold an untraced run to)
        if traced {
            println!(
                "  the run of graphs on: {rows_played} decode steps of several rows replayed their graph, {eager_on} ran eager; {ones_on} of its {one_steps_on} decode steps of \
                 one row replayed their sequence's graph"
            );
            if graphs && batchable && k > 1 && rows_played == 0 {
                println!("FAIL no decode step of several rows replayed a graph in the run of graphs on");
                fails += 1;
            }
            if graphs && k <= top && eager_on > 0 {
                println!("FAIL {eager_on} decode steps of several rows ran eager in the run of graphs on, where the graphs of up to {top} rows hold them");
                fails += 1;
            }
            if e.graphs_taken() && !ahead && one_steps_on > 0 && ones_on == 0 {
                println!("FAIL no decode step of one row replayed its sequence's graph in the run of graphs on");
                fails += 1;
            }
        }
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
    // (a sequence's room grown before each of its passes, as the scheduler grows it (`Sched::ensure_pages`): a sliding group holds the blocks the next pass reads and
    // writes, and a pass's rows are never written past the blocks their sequence holds. Where the step has no room for a pass (the 12-bit pages' raw slabs all taken,
    // where the scheduler pauses the newest sequence), that sequence is given up and the check takes no more: it runs on the sequences before it)
    'prompts: for (j, p) in prompts.iter().take(k).enumerate() {
        if !step.grow(j as i32, p.len() + 2)? {
            break;
        }
        // (in the step's chunks, as the scheduler runs a prompt: a pass holds `chunk` tokens at most)
        let (mut s, chunk) = (None, step.chunk().max(1));
        for (i, c) in p.chunks(chunk).enumerate() {
            if !step.grow(j as i32, i * chunk + c.len())? {
                step.release(&[j as i32])?;
                break 'prompts;
            }
            s = step.prefill(&[j as i32], i * chunk, c, (i + 1) * chunk >= p.len())?;
        }
        let s = s.ok_or_else(|| Error("a prompt's last chunk gave no sample".into()))?;
        check(step, 0, &format!("after prompt {j}"))?;
        next.push(s.token);
    }
    // (a decode row for each sequence whose room grows for it, likewise)
    let mut ready = Vec::new();
    for j in 0..next.len() {
        if step.grow(j as i32, prompts[j].len() + 1)? {
            ready.push(j);
        }
    }
    let pages: Vec<[i32; 1]> = ready.iter().map(|&j| [j as i32]).collect();
    let decode: Vec<crate::sched::Row> = ready.iter().zip(&pages).map(|(&j, page)| crate::sched::Row::new(next[j], prompts[j].len(), page)).collect();
    if !decode.is_empty() {
        step.decode(&decode)?;
        for r in 0..decode.len() {
            check(step, r, &format!("decode row {r} of {}", decode.len()))?;
        }
    }
    step.release(&(0..next.len() as i32).collect::<Vec<_>>())?;
    Ok((rows, bad))
}
