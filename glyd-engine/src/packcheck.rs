//! Pack-format compatibility (R10, #231): every pack format Glyd has written loads, on every merge, to the logits recorded for it. Two sets:
//!
//! - The saved checkpoints of `tests/data/packs` (glyd-gpu's `pack_compat.rs` says what each holds and who wrote it: a tiny Qwen3, one layer of 128, in every format version and layout,
//!   glyd-v1 to glyd-v7; and matrices alone for the formats no model of it holds). Each is read as a reader of the save reads it (glyd-gpu's `save::decoded`: every check
//!   `glyd verify` makes, then each pack decoded and cut into its tensors) and run on the CPU reference of the layer kinds (`runtime::reference`, the model's description
//!   from `TINY_CONFIG`). Every fixture's tensors are first held bit for bit to their sha256 in `tests/packs/tensors.json`, then a model's rows to `tests/packs/NAME.json.gz`
//!   within the bounds, both recorded from the same (`packcheck --record`). On the CPU on every merge (the unit test below,
//!   which ci.yml's engine tests run) and with the GPU library's decode on the GPU runner (`packcheck --gpu`).
//! - The formats the engine reads a saved directory in, as the glyd/* repos and the engine's cache hold them (`packcheck --engine`, the GPU job): glyd-v7 at penguin,
//!   kestrel, swift and wren (glyd-v4 and glyd-v3 with the dense Qwen3 engine's `engine.safetensors` went with that engine, R3c #139: such a cache is packed again). Each is
//!   made here from a logits case's random model of a family published in it (Qwen3.5's layout, Gemma 4, Nemotron-H; Qwen3 dense for the engine's cache) by the packer that made those repos (`pack::run`), loaded from the saved directory as
//!   `run` loads a pre-packed repo, and its rows held to the case's logits fixtures at that level (`tests/logits`, src/logits.rs): the same bits as the engine's own run
//!   of the model packed as it loads (`CASE.GPU[.LEVEL].json.gz`, on the GPU generation that recorded them) and within the bound of the case's reference. And glyd-v7 at
//!   hummingbird (glyd/nemotron-3-hummingbird: the experts in codebook rows), from a random Nemotron-H with latent experts and a random export of them: no level packs it as
//!   it loads, so its digest is the engine's own bits on the saved directory (`HB_DIGEST`).
//!
//! Comparable mechanisms (llama.cpp's GGUF: a type id is never reused and a file names its version, which a
//! reader refuses past what it knows; vLLM: a checkpoint's quantization_config picks the reader). Glyd's saves name
//! their format in glyd.json (`glyd-vN`) and each pack its layout; a change to a layout's bytes is a new format version with a new fixture, never an edit of one.

use crate::ffi::{err, Error, Result};
use crate::logits::{self, Bound, Fixture, LevelBound, Row};
use crate::runtime::reference::model::RefModel;
use glyd_gpu::save::Decoder;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// The saved checkpoints.
pub const PACKS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/data/packs");
/// Their recorded rows.
pub const DIGESTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/packs");

/// The tiny Qwen3 of tests/data/packs as transformers' config of it reads (the fixtures' own config.json, which a release's `glyd pack` copied, names the architecture and the
/// layers alone): 128 wide, 4 query heads and 2 key-value heads of 32, an MLP of 128, a vocabulary of 128, the output layer tied.
pub const TINY_CONFIG: &str = r#"{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 128, "intermediate_size": 128, "num_hidden_layers": 1,
"num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32, "vocab_size": 128, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "max_position_embeddings": 4096,
"tie_word_embeddings": true, "hidden_act": "silu"}"#;

/// What a fixture's rows on the CPU reference must be within of its recorded ones: the same arithmetic on the same weights, so a difference is the last bits of another
/// system's exp and sum order, and a reader that gives another weight is far past it. The tiny Qwen3's random weights make logits of a hundred or more and a softmax of one
/// token, where a KL tells little: each recorded logit is also held to `CPU_DLOGIT` of the row's largest magnitude.
pub const CPU_BOUND: Bound = Bound::Kl { mean: 1e-9, max: 1e-8, top1: 1.0 };
pub const CPU_DLOGIT: f32 = 1e-5;

/// `rows` against the digest `fx`: within `CPU_BOUND` and every recorded logit within `CPU_DLOGIT` of the row's largest magnitude.
pub fn within(fx: &Fixture, rows: &[Vec<f32>]) -> logits::Verdict {
    let mut v = logits::compare(fx, &CPU_BOUND, rows);
    let scale = fx.rows.iter().flat_map(|r| r.logits.iter()).fold(0f32, |m, x| m.max(x.abs()));
    if v.max_dlogit > CPU_DLOGIT * scale {
        v.ok = false;
        v.notes.push(format!("a logit {:.3e} from its recorded value, past {:.0e} of the rows' largest magnitude {scale:.3}", v.max_dlogit, CPU_DLOGIT));
    }
    v
}

/// The fixed tokens: a prompt of 12, then 4 decode steps (from the logits cases' list, over the vocabulary of 128).
fn tokens() -> (Vec<u32>, Vec<u32>) {
    (logits::token_list(12, 128, 0), logits::token_list(4, 128, 12))
}

/// Whether a saved checkpoint holds the tiny Qwen3 (its first layer's attention packed), rather than matrices alone.
fn holds_model(dir: &Path) -> bool {
    std::fs::read_to_string(dir.join("glyd.json")).is_ok_and(|t| t.contains("\"model.layers.0.self_attn.q_proj\""))
}

/// Every saved checkpoint of tests/data/packs, by name.
pub fn fixtures() -> Result<Vec<(String, PathBuf)>> {
    let mut v: Vec<(String, PathBuf)> = std::fs::read_dir(PACKS)
        .map_err(|e| Error(format!("{PACKS}: {e}")))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("glyd.json").is_file())
        .map(|e| (e.file_name().to_string_lossy().into_owned(), e.path()))
        .collect();
    v.sort();
    Ok(v)
}

/// A saved model on the CPU reference, a token at a time.
struct Cpu {
    tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
    m: Option<RefModel>,
    pos: u32,
    last: Vec<f32>,
}

impl logits::Run for Cpu {
    fn describe(&self) -> String {
        "the CPU reference of the layer kinds".into()
    }
    fn reset(&mut self) -> Result<()> {
        let cfg: serde_json::Value = serde_json::from_str(TINY_CONFIG).map_err(|e| Error(e.to_string()))?;
        let d = crate::runtime::describe::describe(&cfg)?;
        let t = &self.tensors;
        self.m = Some(RefModel::load(&d, &|n: &str| t.get(n).cloned())?);
        self.pos = 0;
        Ok(())
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        tokens.iter().try_for_each(|&t| self.step(t))
    }
    fn step(&mut self, t: u32) -> Result<()> {
        if self.m.is_none() {
            self.reset()?;
        }
        let m = self.m.as_mut().expect("loaded above");
        self.last = m.forward(&[t], self.pos)?.pop().ok_or_else(|| Error("no logits".into()))?;
        self.pos += 1;
        Ok(())
    }
    fn logits(&mut self) -> Result<Vec<f32>> {
        Ok(self.last.clone())
    }
}

/// A save's tensors as glyd-gpu's reader gives them (`save::decoded`: every check `glyd verify` makes, then each pack decoded, by `decoder`, and cut into its tensors).
pub type Decoded = Vec<(String, Vec<u64>, Vec<f32>)>;

pub fn read(dir: &Path, decoder: &Decoder) -> Result<Decoded> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get().min(8));
    glyd_gpu::save::decoded(dir, threads, decoder).map_err(|e| Error(format!("{}: {e}", dir.display())))
}

/// The tiny Qwen3 of `t` run on the CPU reference over the fixed tokens: each pass's last logits.
pub fn rows_of(t: Decoded) -> Result<Vec<Vec<f32>>> {
    let tensors = t.into_iter().map(|(n, s, v)| (n, (s.iter().map(|&d| d as usize).collect(), v))).collect();
    let (prompt, decode) = tokens();
    logits::drive(&mut Cpu { tensors, m: None, pos: 0, last: Vec::new() }, &prompt, &decode, &[])
}

/// The saved checkpoint `dir` read by glyd-gpu's reader (`decoder`: the CPU's, or the GPU library's) and run on the CPU reference over the fixed tokens: each pass's last logits.
pub fn cpu_rows(dir: &Path, decoder: &Decoder) -> Result<Vec<Vec<f32>>> {
    rows_of(read(dir, decoder)?)
}

/// Every fixture's tensors as `read` gives them, each its shape and the sha256 of its values (f32, little-endian): the decode is integer and bit operations and exact
/// products (a scale times a code, rounded to bf16 once), no function of a system's library, so these hold bit for bit on any machine and on the GPU's decode.
pub const TENSORS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/packs/tensors.json");

/// Each tensor of `t` by name: its shape and the sha256 of its values.
pub fn tensor_digests(t: &Decoded) -> BTreeMap<String, (Vec<u64>, String)> {
    use sha2::{Digest, Sha256};
    t.iter()
        .map(|(n, s, v)| {
            let mut h = Sha256::new();
            v.iter().for_each(|x| h.update(x.to_le_bytes()));
            (n.clone(), (s.clone(), h.finalize().iter().map(|b| format!("{b:02x}")).collect()))
        })
        .collect()
}

type FixtureDigests = BTreeMap<String, BTreeMap<String, (Vec<u64>, String)>>;

fn read_tensor_digests() -> Result<FixtureDigests> {
    let text = std::fs::read_to_string(TENSORS).map_err(|e| Error(format!("{TENSORS}: {e} (packcheck --record writes it)")))?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| Error(format!("{TENSORS}: {e}")))?;
    let bad = || Error(format!("{TENSORS}: not {{fixture: {{tensor: {{shape, sha256}}}}}}"));
    v.as_object()
        .ok_or_else(bad)?
        .iter()
        .map(|(f, ts)| {
            let ts = ts
                .as_object()
                .ok_or_else(bad)?
                .iter()
                .map(|(n, t)| {
                    let shape = t["shape"].as_array().ok_or_else(bad)?.iter().map(|d| d.as_u64().ok_or_else(bad)).collect::<Result<Vec<u64>>>()?;
                    Ok((n.clone(), (shape, t["sha256"].as_str().ok_or_else(bad)?.to_string())))
                })
                .collect::<Result<_>>()?;
            Ok((f.clone(), ts))
        })
        .collect()
}

fn write_tensor_digests(d: &FixtureDigests) -> Result<()> {
    let v: serde_json::Map<String, serde_json::Value> = d
        .iter()
        .map(|(f, ts)| {
            (
                f.clone(),
                serde_json::json!(ts.iter().map(|(n, (s, h))| (n.clone(), serde_json::json!({"shape": s, "sha256": h}))).collect::<serde_json::Map<_, _>>()),
            )
        })
        .collect();
    let text = serde_json::to_string_pretty(&serde_json::Value::Object(v)).map_err(|e| Error(e.to_string()))? + "\n";
    std::fs::write(TENSORS, text).map_err(|e| Error(format!("{TENSORS}: {e}")))
}

/// The first way `got` is not `want`, bit for bit: a tensor missing or not recorded, another shape, other values.
pub fn tensors_differ(want: &BTreeMap<String, (Vec<u64>, String)>, got: &BTreeMap<String, (Vec<u64>, String)>) -> Option<String> {
    if let Some(n) = want.keys().find(|n| !got.contains_key(*n)) {
        return Some(format!("{n}: recorded, not read"));
    }
    if let Some(n) = got.keys().find(|n| !want.contains_key(*n)) {
        return Some(format!("{n}: read, not recorded"));
    }
    want.iter().find_map(|(n, (s, h))| {
        let (gs, gh) = &got[n];
        (gs != s)
            .then(|| format!("{n}: shape {gs:?}, recorded {s:?}"))
            .or_else(|| (gh != h).then(|| format!("{n}: other values than recorded (sha256 {gh}, recorded {h})")))
    })
}

/// A fixture's recorded rows.
pub fn digest_path(name: &str) -> PathBuf {
    Path::new(DIGESTS).join(format!("{name}.json.gz"))
}

/// The rows of `name` recorded as its digest (`provenance` says how).
fn digest_of(name: &str, rows: &[Vec<f32>], what: &str) -> Result<Fixture> {
    let (prompt, decode) = tokens();
    let mut provenance = BTreeMap::new();
    provenance.insert("source".into(), format!("{what}, on the weights glyd-gpu's reader decodes from tests/data/packs/{name} (save::decoded), fp32 inside"));
    provenance.insert("tokens".into(), "src/logits.rs token_list over the vocabulary of 128: a prompt of 12, then 4 steps".into());
    Ok(Fixture {
        case: name.into(),
        family: "the tiny Qwen3 of tests/data/packs (src/packcheck.rs TINY_CONFIG)".into(),
        model: name.into(),
        provenance,
        gpu: None,
        bounds: vec![LevelBound { level: "penguin".into(), bound: CPU_BOUND }],
        prompt,
        decode,
        long: Vec::new(),
        rows: rows.iter().map(|r| Row::from_logits(r)).collect::<Result<_>>()?,
    })
}

/// One fixture's line: PASS or FAIL against its digest, and how near (its `n` tensors already bit for bit).
fn line(name: &str, n: usize, fx: &Fixture, rows: &[Vec<f32>]) -> (bool, String) {
    let v = within(fx, rows);
    let s = format!(
        "{} {name:16} {n} tensors bit for bit; {} rows: bits-equal {}, KL mean {:.2e} max {:.2e}, best token {}/{}, largest logit difference {:.2e}{}",
        if v.ok { "PASS" } else { "FAIL" },
        v.rows,
        v.identical,
        v.mean_kl,
        v.max_kl,
        v.top1.0,
        v.top1.1,
        v.max_dlogit,
        v.notes.iter().take(3).map(|n| format!("\n       {n}")).collect::<String>()
    );
    (v.ok, s)
}

/// Every saved checkpoint of tests/data/packs against its digests, its packs decoded by `decoder`: first each tensor bit for bit (`TENSORS`), then, where it holds the tiny
/// Qwen3, its logits on the CPU reference within the bounds (`within`). The lines, and the names that failed.
pub fn check_saved(decoder: &Decoder) -> Result<(Vec<String>, Vec<String>)> {
    let recorded = read_tensor_digests()?;
    let (mut lines, mut failed) = (Vec::new(), Vec::new());
    for (name, dir) in fixtures()? {
        let t = read(&dir, decoder)?;
        let got = tensor_digests(&t);
        let differ = match recorded.get(&name) {
            None => Some("no tensor digests recorded (packcheck --record)".to_string()),
            Some(want) => tensors_differ(want, &got),
        };
        if let Some(why) = differ {
            lines.push(format!("FAIL {name:16} its tensors are not the recorded ones, bit for bit: {why}"));
            failed.push(name);
            continue;
        }
        if !holds_model(&dir) {
            lines.push(format!("PASS {name:16} {} tensors bit for bit (matrices alone, no model to run)", got.len()));
            continue;
        }
        let fx = Fixture::read(&digest_path(&name)).map_err(|e| Error(format!("{name}: no digest ({}): packcheck --record writes it", e.msg)))?;
        let (ok, s) = line(&name, got.len(), &fx, &rows_of(t)?);
        lines.push(s);
        if !ok {
            failed.push(name);
        }
    }
    Ok((lines, failed))
}

/// `packcheck [--gpu [--device N]] [--record] [--engine [--scratch DIR]]`: every saved checkpoint of tests/data/packs on the CPU reference against its digest (its packs decoded on
/// the CPU, or by the GPU library with --gpu); --engine the formats the engine reads, each made from a random model, loaded from the saved directory and held to its fixtures
/// (`check_engine`). --record writes the digests instead (from the CPU's decode; with --engine, the Hummingbird level's from this GPU). Exit 1 where one is not within.
pub fn check(a: &crate::args::Args) -> Result<()> {
    if a.has("record") {
        let mut all = FixtureDigests::new();
        for (name, dir) in fixtures()? {
            let t = read(&dir, &Decoder::Cpu)?;
            all.insert(name.clone(), tensor_digests(&t));
            if !holds_model(&dir) {
                continue;
            }
            let fx = digest_of(&name, &rows_of(t)?, "the CPU reference of the layer kinds (src/runtime/reference)")?;
            fx.write(&digest_path(&name))?;
            println!("{}: {} rows", digest_path(&name).display(), fx.rows.len());
        }
        write_tensor_digests(&all)?;
        println!("{TENSORS}: the tensors of {} fixtures", all.len());
        if a.has("engine") {
            with_scratch(a, |scratch| hummingbird(scratch, true).map(|_| ()))?;
        }
        return Ok(());
    }
    let mut failed = Vec::new();
    let (lines, f) = if a.has("gpu") {
        let lib = glyd_gpu::Library::find().map_err(|e| Error(e.to_string()))?;
        let ctx = glyd_gpu::cuda::Context::new(a.num("device", 0i32)?).map_err(|e| Error(e.to_string()))?;
        check_saved(&Decoder::Gpu(&lib, &ctx))?
    } else {
        check_saved(&Decoder::Cpu)?
    };
    lines.iter().for_each(|l| println!("{l}"));
    failed.extend(f);
    if a.has("engine") {
        let fixtures = a.flag("fixtures").map(PathBuf::from).unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/logits"));
        failed.extend(with_scratch(a, |scratch| {
            let mut f = check_engine(scratch, &fixtures)?;
            if !hummingbird(scratch, false)? {
                f.push("nemotron-h-hb".into());
            }
            Ok(f)
        })?);
    }
    if failed.is_empty() {
        println!("every pack format loads to its recorded logits");
        Ok(())
    } else {
        println!("OFF: {}", failed.join(", "));
        std::process::exit(1);
    }
}

/// `f` on the models' directory: --scratch DIR (kept, and what it holds used again), else one of this run that is removed after.
fn with_scratch<T>(a: &crate::args::Args, f: impl FnOnce(&Path) -> Result<T>) -> Result<T> {
    let (scratch, own) = match a.flag("scratch") {
        Some(d) => (PathBuf::from(d), false),
        None => (std::env::temp_dir().join(format!("glyd-packcheck-{}", std::process::id())), true),
    };
    std::fs::create_dir_all(&scratch).map_err(|e| Error(e.to_string()))?;
    let r = f(&scratch);
    if own {
        let _ = std::fs::remove_dir_all(&scratch);
    }
    r
}

fn threads() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get().min(16))
}

/// The formats the engine reads a saved directory in, each as a published repo or the engine's cache holds it: the logits case of a family published in it, the level it
/// runs at, the layout `pack::run` saves it in, and what it is.
pub const SAVED: &[(&str, &str, &str, &str)] = &[
    ("qwen3", "penguin", "mma", "glyd-v7 at penguin: a Qwen3 model in the engine's cache (the executor's, src/dynamic/dense.rs)"),
    ("qwen35", "kestrel", "i6", "glyd-v7 i6: glyd/Qwen3.5-*, Qwen3.6-27B and Qwen3.8-27B -kestrel"),
    ("qwen35", "swift", "i5", "glyd-v7 i5: the same at swift"),
    ("qwen35", "wren", "i4", "glyd-v7 i4: the same at wren"),
    ("gemma4", "penguin", "mma", "glyd-v7 at penguin: glyd/gemma-4-12B-penguin, -31B-penguin"),
    ("gemma4", "kestrel", "i6", "glyd-v7 i6: Gemma 4 at kestrel"),
    ("gemma4", "swift", "i5", "glyd-v7 i5: Gemma 4 at swift"),
    ("gemma4", "wren", "i4", "glyd-v7 i4: Gemma 4 at wren (a checkpoint not trained to four bits)"),
    ("nemotron-h", "kestrel", "i6", "glyd-v7 i6: Nemotron-H at kestrel"),
    ("nemotron-h", "swift", "i5", "glyd-v7 i5: Nemotron-H at swift"),
    ("nemotron-h", "wren", "i4", "glyd-v7 i4: Nemotron-H at wren"),
];

/// The case's fixtures (`tests/logits`, or --fixtures DIR) that bound `level`, and the bound each gives it there.
fn level_fixtures(dir: &Path, case: &str, level: &str) -> Result<Vec<(PathBuf, Fixture, Bound)>> {
    let mut v = Vec::new();
    for e in std::fs::read_dir(dir).map_err(|e| Error(format!("{}: {e}", dir.display())))? {
        let p = e.map_err(|e| Error(e.to_string()))?.path();
        let n = p.file_name().unwrap_or_default().to_string_lossy().to_string();
        if !(n.ends_with(".json.gz") && (n == format!("{case}.json.gz") || n.starts_with(&format!("{case}.")))) {
            continue;
        }
        let fx = Fixture::read(&p)?;
        if let Some(b) = fx.bounds.iter().find(|b| b.level == level).map(|b| b.bound.clone()) {
            v.push((p, fx, b));
        }
    }
    v.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(v)
}

/// `--engine`: each of `SAVED` made from its case's random model (made once in `scratch`), packed by `pack::run` into a directory of its own, loaded from it by the case's engine at
/// the level, and its rows held to every fixture of the case in `fixtures` that bounds the level: the bits of one recorded on this GPU generation (one of another is skipped),
/// the bound of a reference's. A format with no such fixture fails: it has no digest. The names that failed.
fn check_engine(scratch: &Path, fixtures: &Path) -> Result<Vec<String>> {
    let gpu = logits::gpu_label()?;
    let mut failed = Vec::new();
    {
        for &(case, level, layout, what) in SAVED {
            let c = logits::find_case(case)?;
            let model = scratch.join(case);
            if !model.join("config.json").exists() {
                match c.weights {
                    logits::Weights::Random(make) => make(&model)?,
                    logits::Weights::Hub(id) => return err(format!("{case}: a Hub model ({id}), not a random one")),
                }
            }
            let out = scratch.join(format!("{case}-{layout}"));
            let l = glyd_gpu::save::Layout::by_name(layout).ok_or_else(|| Error(format!("layout {layout}")))?;
            if !out.join("glyd.json").exists() {
                crate::pack::run_with(&model.to_string_lossy(), &out, threads(), l, true)?;
            }
            let format = std::fs::read_to_string(out.join("glyd.json"))
                .ok()
                .and_then(|t| glyd_gpu::json::parse(&t).ok())
                .and_then(|m| m.get("format").and_then(glyd_gpu::json::Value::as_str).map(str::to_string))
                .unwrap_or_default();
            let fxs = level_fixtures(fixtures, case, level)?;
            let name = format!("{case}-{layout}");
            if fxs.is_empty() {
                println!(
                    "FAIL {name:16} {format} at {level}: no fixture of {case} bounds {level} (logitsrecord {case} --from engine --level {level} records one)"
                );
                failed.push(name);
                continue;
            }
            let mut run = (c.engine)(&out, level)?;
            let mut ran = 0;
            for (path, fx, bound) in fxs {
                if fx.gpu.as_ref().is_some_and(|g| *g != gpu) {
                    println!(
                        "SKIP {name:16} {}: its bits are {}'s, this GPU is {gpu}",
                        path.file_name().unwrap_or_default().to_string_lossy(),
                        fx.gpu.as_deref().unwrap_or("")
                    );
                    continue;
                }
                let rows = logits::drive(run.as_mut(), &fx.prompt, &fx.decode, &fx.long)?;
                let v = logits::compare(&fx, &bound, &rows);
                ran += 1;
                println!(
                    "{} {name:16} {format} at {level} ({what}) against {}: bits-equal {}/{}, KL mean {:.2e} max {:.2e}, best token {}/{}",
                    if v.ok { "PASS" } else { "FAIL" },
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    v.identical,
                    v.rows,
                    v.mean_kl,
                    v.max_kl,
                    v.top1.0,
                    v.top1.1
                );
                for n in v.notes.iter().take(4) {
                    println!("       {n}");
                }
                if !v.ok {
                    failed.push(name.clone());
                }
            }
            if ran == 0 {
                println!("FAIL {name:16} {format} at {level}: no fixture of this GPU generation or of a reference bounds it");
                failed.push(name);
            }
        }
    }
    Ok(failed)
}

/// Where a digest the check holds a format to is not recorded (and this is no recording): the line that says so, a failure, before any model is made or loaded.
fn digest_missing(path: &Path, record: bool) -> Option<String> {
    (!record && !path.exists()).then(|| {
        format!(
            "FAIL nemotron-h-hb     glyd-v7 at hummingbird: no digest recorded ({}): packcheck --engine --record on the GPU generation the GPU job runs on",
            path.file_name().unwrap_or_default().to_string_lossy()
        )
    })
}

/// The Hummingbird level's digest: the engine's own bits (no level packs it as it loads, its codes being an export's), on the GPU generation that recorded them.
pub const HB_DIGEST: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/packs/nemotron-h-hb.json.gz");

/// Nemotron-H at the Hummingbird level, as glyd/nemotron-3-hummingbird is saved: a random Nemotron-H with Super's latent experts (`kcheck_nh::random`, a latent of 1,024, 16
/// experts), a random export of its experts (`hbpack::toy_export`, as `hbtoy` makes one: 40 percent at three bits, the rest at two) and the glyd-v7 directory `hbpack::run`
/// saves from them, loaded as `run` loads the repo; its rows against `HB_DIGEST` (written where `record`). Whether it passed (a digest of another GPU generation: skipped).
fn hummingbird(scratch: &Path, record: bool) -> Result<bool> {
    use glyd_gpu::save::Layout;
    if let Some(line) = digest_missing(Path::new(HB_DIGEST), record) {
        println!("{line}");
        return Ok(false);
    }
    let (model, export, out) = (scratch.join("nemotron-h-latent"), scratch.join("nemotron-h-export"), scratch.join("nemotron-h-hb"));
    if !model.join("config.json").exists() {
        crate::kcheck_nh::random(&model, 1024, 16)?;
    }
    if !out.join("glyd.json").exists() {
        crate::hbpack::toy_export(&crate::nemotron::ConfigNemotronH::load(&model)?, &export, 1, 40)?;
        crate::hbpack::run(&model.to_string_lossy(), &out, threads(), &export)?;
    }
    let gpu = logits::gpu_label()?;
    let o = crate::dynamic::Opts { ctx: 1024, chunk: 64, graph: true, device: 0, layout: Some(Layout::Hummingbird) };
    let mut run = crate::dynamic::nemotron::load(&out, &o, Layout::Hummingbird)?;
    let (prompt, decode, long) = (logits::token_list(24, 4096, 0), logits::token_list(12, 4096, 24), logits::token_list(150, 4096, 0));
    let rows = logits::drive(&mut run, &prompt, &decode, &long)?;
    if record {
        let mut provenance = BTreeMap::new();
        provenance.insert(
            "source".into(),
            format!(
                "glyd-engine {} ({}): the engine's own output on the saved directory, no reference of another implementation",
                env!("CARGO_PKG_VERSION"),
                logits::Run::describe(&run)
            ),
        );
        provenance.insert("tokens".into(), "src/logits.rs token_list over the vocabulary of 4,096: a prompt of 24, 12 steps, a long prompt of 150".into());
        let fx = Fixture {
            case: "nemotron-h-hb".into(),
            family: "Nemotron-H with latent experts at the Hummingbird level (src/packcheck.rs hummingbird)".into(),
            model: "random (kcheck_nh::random, latent 1024, 16 experts), its experts from hbpack::toy_export (seed 1, 40 percent at 3 bits)".into(),
            provenance,
            gpu: Some(gpu),
            bounds: vec![LevelBound { level: "hummingbird".into(), bound: Bound::Bits }],
            prompt,
            decode,
            long,
            rows: rows.iter().map(|r| Row::from_logits(r)).collect::<Result<_>>()?,
        };
        fx.write(Path::new(HB_DIGEST))?;
        println!("{HB_DIGEST}: {} rows", fx.rows.len());
        return Ok(true);
    }
    let fx = Fixture::read(Path::new(HB_DIGEST))?;
    if fx.gpu.as_deref() != Some(gpu.as_str()) {
        println!("SKIP nemotron-h-hb     glyd-v7 at hummingbird: its digest's bits are {}'s, this GPU is {gpu}", fx.gpu.as_deref().unwrap_or("none"));
        return Ok(true);
    }
    let v = logits::compare(&fx, &Bound::Bits, &rows);
    println!(
        "{} nemotron-h-hb     glyd-v7 at hummingbird (glyd/nemotron-3-hummingbird: the experts in codebook rows c3r and c2r) against {}: bits-equal {}/{}, best token {}/{}",
        if v.ok { "PASS" } else { "FAIL" },
        Path::new(HB_DIGEST).file_name().unwrap_or_default().to_string_lossy(),
        v.identical,
        v.rows,
        v.top1.0,
        v.top1.1
    );
    for n in v.notes.iter().take(4) {
        println!("       {n}");
    }
    Ok(v.ok)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every saved checkpoint of tests/data/packs loads (every check of `glyd verify`, then its packs decoded on the CPU) and, where it holds the tiny Qwen3, its rows on the CPU
    /// reference are its recorded digest's: on every merge (ci.yml's engine tests).
    /// A format whose digest is not recorded fails the check with a line that says how to record it, and does not stop the other formats' checks.
    #[test]
    fn a_digest_not_recorded_fails_and_says_how_to_record_it() {
        let gone = std::env::temp_dir().join(format!("glyd-packcheck-no-digest-{}.json.gz", std::process::id()));
        let line = digest_missing(&gone, false).unwrap();
        assert!(line.starts_with("FAIL nemotron-h-hb") && line.contains("packcheck --engine --record"), "{line}");
        assert!(digest_missing(&gone, true).is_none() && digest_missing(Path::new(env!("CARGO_MANIFEST_DIR")), false).is_none());
    }

    #[test]
    fn every_saved_pack_format_loads_to_its_recorded_logits() {
        let (lines, failed) = check_saved(&Decoder::Cpu).unwrap();
        let models = lines.iter().filter(|l| l.contains(" rows: ")).count();
        assert!(lines.len() >= 16 && models >= 12, "{} fixtures, {models} of a model:\n{}", lines.len(), lines.join("\n"));
        assert!(failed.is_empty(), "{}", lines.join("\n"));
    }

    /// The digests tell the layouts apart: two fixtures' rows are within each other's digests exactly where their formats decode to the same weights (the lossless ones,
    /// glyd-v3 to glyd-v5 and glyd-v7 at penguin), and past them otherwise (a row layout's runs of 32 are along the outputs, the block layout's along the inputs: other weights).
    #[test]
    fn the_digests_tell_the_layouts_apart() {
        let all: Vec<(String, Vec<Vec<f32>>)> =
            fixtures().unwrap().into_iter().filter(|(_, d)| holds_model(d)).map(|(n, d)| (n.clone(), cpu_rows(&d, &Decoder::Cpu).unwrap())).collect();
        fn class(n: &str) -> &str {
            match n {
                "v0.33.1-mma" | "v0.33.1-mma12" | "v5-0.29.1-mma" | "main-penguin" => "lossless",
                _ => n,
            }
        }
        for (a, _) in &all {
            let fx = Fixture::read(&digest_path(a)).unwrap();
            for (b, rows) in &all {
                assert_eq!(within(&fx, rows).ok, class(a) == class(b), "{b}'s rows against {a}'s digest");
            }
        }
    }

    /// A row-layout pack read as its block layout's (its transpose taken for the weight): the reference's rows are past the bound. The check sees a reader that misplaces weights.
    #[test]
    fn a_misread_row_layout_is_past_the_bound() {
        let dir = Path::new(PACKS).join("main-i6r");
        let mut t: HashMap<String, (Vec<usize>, Vec<f32>)> =
            glyd_gpu::save::decoded(&dir, 2, &Decoder::Cpu).unwrap().into_iter().map(|(n, s, v)| (n, (s.iter().map(|&d| d as usize).collect(), v))).collect();
        let (shape, w) = t.get("model.layers.0.self_attn.o_proj.weight").cloned().unwrap();
        let (r, c) = (shape[0], shape[1]);
        let tr: Vec<f32> = (0..r * c).map(|i| w[(i % c) * c + i / c]).collect();
        t.insert("model.layers.0.self_attn.o_proj.weight".into(), (shape, tr));
        let (prompt, decode) = tokens();
        let rows = logits::drive(&mut Cpu { tensors: t.clone(), m: None, pos: 0, last: Vec::new() }, &prompt, &decode, &[]).unwrap();
        let fx = Fixture::read(&digest_path("main-i6r")).unwrap();
        assert!(!within(&fx, &rows).ok);
        // and its tensors are not the recorded ones, bit for bit, which the check sees first
        let d: Decoded = t.into_iter().map(|(n, (s, v))| (n, s.iter().map(|&x| x as u64).collect(), v)).collect();
        let why = tensors_differ(&read_tensor_digests().unwrap()["main-i6r"], &tensor_digests(&d)).expect("a transposed weight's values differ");
        assert!(why.contains("o_proj") && why.contains("other values"), "{why}");
    }
}
