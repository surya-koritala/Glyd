//! `glyd-engine intake MODEL [--record DIR]`: the first step of adding a model family, or of meeting a new release of one the engine has (#617).
//!
//! MODEL is a Hub repo (its config.json and its safetensors files' headers are read by ranges: nothing of the weights is downloaded), a directory, or a
//! recorded header (tests/fixtures/headers/NAME.json). It tries to describe the model (`runtime::describe`) and prints, in one pass:
//!
//!   - what the model is: its model type, its architecture's name, its tensors and their bytes;
//!   - where it describes: the layer kinds its blocks are, then every tensor of the checkpoint its description does not read and every tensor it reads that
//!     the checkpoint lacks, grouped by name with the layer and expert numbers masked;
//!   - where it does not: the builder's refusal, every config key it names as not understood, and, for a model type with no builder, every key of the text
//!     config that says something of the model's shape and every group of tensors with its type and shape: what a builder is written from;
//!   - signs of the layer kinds it needs (`SIGNS`: words in its config keys and tensor names), each with whether the engine has the kind. A sign is a name's
//!     word, not a reading of the release's code: it says where to look.
//!
//! `--record DIR` writes the recorded header DIR/NAME.json (a Hub repo's alone), the fixture the family's tests hold its description to.
//! It ends 0 where the model describes and its checkpoint is read whole, 3 where it does not (the page docs/guide/src/adding-a-family.md says what comes next).

use crate::args::Args;
use crate::ffi::{err, Error, Result};
use crate::hub;
use crate::runtime::config::ModelConfig;
use crate::runtime::describe::{self, Described};
use crate::runtime::kinds::{Body, Ffn, Mixer};
use crate::sizes::{self, Group, Table};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// A word in a config key or a tensor's name, the layer kind it is a sign of, and whether the engine has that kind (`kinds.rs`). A word is whole words of
/// the name (`word_in`): "conv" is in `short_conv_kernel_size` and not in `modules_to_not_convert`, "ssm" not in `ssmax_scaling`.
/// Kept short and plain: a sign sends a reader to the release's code, it does not replace reading it.
const SIGNS: &[(&str, &str, bool)] = &[
    ("expert", "a mixture of experts (Ffn::Moe)", true),
    ("moe", "a mixture of experts (Ffn::Moe)", true),
    ("kv_lora", "latent attention (Mixer::Mla)", true),
    ("q_lora", "latent attention (Mixer::Mla)", true),
    ("mamba", "a Mamba-2 state-space mixer (Mixer::Mamba2)", true),
    ("ssm", "a Mamba-2 state-space mixer (Mixer::Mamba2)", true),
    ("compress_ratio", "compressed attention (Mixer::Compressed)", true),
    ("indexer", "a sparse-attention indexer (Indexer)", true),
    ("index_topk", "a sparse-attention indexer (Indexer)", true),
    ("hc", "residual streams (Residual::Streams)", true),
    ("sliding_window", "an attention window by layer (Attn's window)", true),
    ("layer_types", "more than one kind of layer: a block's kind by layer", true),
    ("nextn_predict", "a draft head (MtpHead)", true),
    ("mtp", "a draft head (MtpHead)", true),
    ("ngram", "an n-gram embedding (Body::Ngram): check its form against the kind's", true),
    ("linear_attn", "linear attention (Mixer::Linear)", true),
    ("conv", "a short convolution (Mixer::ShortConv): check its form against the kind's", true),
    ("num_loops", "looped depth: no kind yet", false),
    ("loop", "looped depth: no kind yet", false),
    ("value_residual", "a value residual: no residual form yet", false),
    ("rel_pos", "relative-position attention scores: no option yet", false),
    ("vision", "a vision tower: read by few families' builders", false),
    ("audio", "an audio tower: not built", false),
];

/// Whether `name` (a config key, a tensor's masked name) has `word` as whole words: the name cut at `_`, `.` and `-`, in lower case, holds the word's
/// parts in a row, the last of them alone, in the plural (`experts`) or numbered (`conv1d`).
fn word_in(name: &str, word: &str) -> bool {
    let lower = name.to_lowercase();
    let parts: Vec<&str> = lower.split(['_', '.', '-']).filter(|p| !p.is_empty()).collect();
    let want: Vec<&str> = word.split('_').collect();
    let last =
        |part: &str, w: &str| part.strip_prefix(w).is_some_and(|rest| rest.is_empty() || rest == "s" || rest.chars().all(|c| c.is_ascii_digit() || c == 'd'));
    parts.windows(want.len()).any(|at| at[..want.len() - 1] == want[..want.len() - 1] && last(at[want.len() - 1], want[want.len() - 1]))
}

/// One group of tensors under a masked name with one type and shape: how many tensors, and their bytes in all.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Named {
    pub mask: String,
    pub tensors: usize,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub bytes: u64,
}

/// What `inspect` found.
#[derive(Debug, Default)]
pub struct Report {
    pub model_type: String,
    pub text_type: String,
    pub arch: String,
    pub tensors: usize,
    pub bytes: u64,
    /// The layer kinds of its blocks with how many blocks each (described models).
    pub kinds: Vec<(String, usize)>,
    /// The builder's refusal (None where the model describes).
    pub refusal: Option<String>,
    /// Whether the refusal is "no builder" (a new family) and not a builder's (a release of a family the engine has).
    pub no_builder: bool,
    /// Config keys the builder named as not understood; for a model type with no builder, every key of the text config that is not inert.
    pub keys: Vec<String>,
    /// For a model type with no builder: each of `keys` with its value, cut short.
    pub values: Vec<String>,
    /// Tensors of the checkpoint the description does not read, and tensors it reads that the checkpoint lacks (described models).
    pub not_read: Vec<Named>,
    pub missing: Vec<Named>,
    /// Every group of the checkpoint's tensors (a model type with no builder).
    pub groups: Vec<Named>,
    /// The signs found: the word, where (a config key or a tensor's masked name), the kind, whether the engine has it.
    pub signs: Vec<(String, String, String, bool)>,
}

impl Report {
    /// The model describes and its checkpoint is read whole but the parts its description leaves.
    pub fn ok(&self) -> bool {
        self.refusal.is_none() && self.not_read.is_empty() && self.missing.is_empty()
    }
}

fn group_names(names: impl Iterator<Item = String>, table: &Table) -> Vec<Named> {
    let rows: BTreeMap<&str, &sizes::Row> = table.rows.iter().map(|r| (r.name.as_str(), r)).collect();
    // (a line for each type and shape under a masked name: a family whose layers differ in width, as Domyn Edge's head size does, shows it here)
    let mut by: BTreeMap<(String, String, Vec<u64>), Named> = BTreeMap::new();
    for n in names {
        let mask = sizes::masked(&n);
        let row = rows.get(n.as_str());
        let (dtype, shape) = (row.map_or(String::new(), |r| r.dtype.clone()), row.map_or(Vec::new(), |r| r.shape.clone()));
        let e = by.entry((mask.clone(), dtype.clone(), shape.clone())).or_insert_with(|| Named { mask, tensors: 0, dtype, shape, bytes: 0 });
        e.tensors += 1;
        e.bytes += row.map_or(0, |r| r.bytes);
    }
    by.into_values().collect()
}

fn kinds_of(d: &Described) -> Vec<(String, usize)> {
    let mut n: BTreeMap<&'static str, usize> = BTreeMap::new();
    for b in &d.desc.blocks {
        let k = match &b.body {
            Body::Mixer(Mixer::Attention(_)) => "Attention",
            Body::Mixer(Mixer::Mla(_)) => "Mla (latent attention)",
            Body::Mixer(Mixer::Compressed(_)) => "Compressed attention",
            Body::Mixer(Mixer::Linear(_)) => "Linear attention",
            Body::Mixer(Mixer::Mamba2(_)) => "Mamba2",
            Body::Mixer(Mixer::ShortConv(_)) => "ShortConv (gated short convolution)",
            Body::Ffn(Ffn::Mlp(_)) => "Mlp",
            Body::Ffn(Ffn::Moe(_)) => "Moe",
            Body::Ngram(_) => "Ngram",
            Body::PerLayer(_) => "PerLayer",
        };
        *n.entry(k).or_default() += 1;
    }
    let mut out: Vec<(String, usize)> = n.into_iter().map(|(k, c)| (k.to_string(), c)).collect();
    if !d.desc.mtp.is_empty() {
        out.push(("draft head (MtpHead)".to_string(), d.desc.mtp.len()));
    }
    out
}

/// The signs in a config's keys (every level: a nested object's keys too) and in the tensors' masked names, each word once per kind.
fn signs_of(cfg: &serde_json::Value, groups: &[Group]) -> Vec<(String, String, String, bool)> {
    fn keys(v: &serde_json::Value, out: &mut BTreeSet<String>) {
        if let Some(m) = v.as_object() {
            for (k, v) in m {
                out.insert(k.clone());
                keys(v, out);
            }
        }
    }
    let mut ks = BTreeSet::new();
    keys(cfg, &mut ks);
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    let mut out = Vec::new();
    for (word, kind, has) in SIGNS {
        if seen.contains(kind) {
            continue;
        }
        let at = ks
            .iter()
            .find(|k| !describe::is_inert(k) && word_in(k, word))
            .map(|k| format!("config key {k}"))
            // (a tower's own tensors are signs of the tower alone: a vision encoder's patch convolution is not a convolution block of the language model)
            .or_else(|| {
                let tower =
                    |m: &str| word_in(m, "vision") || word_in(m, "audio") || m.to_lowercase().contains("vision_") || m.to_lowercase().contains("audio_");
                groups.iter().find(|g| word_in(&g.mask, word) && (matches!(*word, "vision" | "audio") || !tower(&g.mask))).map(|g| format!("tensor {}", g.mask))
            });
        if let Some(at) = at {
            seen.insert(kind);
            out.push((word.to_string(), at, kind.to_string(), *has));
        }
    }
    out
}

/// What a config.json's text and its checkpoint's table of tensors say of the model against the engine's description builders.
pub fn inspect(config: &str, table: &Table) -> Result<Report> {
    let cfg: serde_json::Value = serde_json::from_str(config).map_err(|e| Error(format!("config.json: {e}")))?;
    let c = ModelConfig::from_value(&cfg)?;
    let have: BTreeSet<String> = table.rows.iter().map(|r| r.name.clone()).collect();
    let groups = table.compact();
    let mut r = Report {
        model_type: c.model_type.clone(),
        text_type: c.text_type.clone(),
        arch: c.arch.clone(),
        tensors: table.rows.len(),
        bytes: table.total_bytes(),
        signs: signs_of(&cfg, &groups),
        ..Report::default()
    };
    match describe::describe_held(&cfg, &have) {
        Ok(d) => {
            let names = d.names();
            let left = |n: &str| d.unread.iter().any(|p| p.strip_prefix('*').map_or(n.starts_with(p.as_str()), |sfx| n.ends_with(sfx)));
            r.kinds = kinds_of(&d);
            r.not_read = group_names(have.iter().filter(|n| !names.contains(*n) && !left(n)).cloned(), table);
            r.missing = group_names(names.iter().filter(|n| !have.contains(*n)).cloned(), table);
        }
        Err(e) => {
            r.no_builder = e.msg.contains("has no description builder yet");
            r.keys = describe::keys_named(&e.msg);
            if r.no_builder {
                r.keys = c.text.keys().filter(|k| !describe::is_inert(k)).cloned().collect();
                let short = |v: &serde_json::Value| {
                    let t = v.to_string();
                    if t.chars().count() > 72 {
                        format!("{}...", t.chars().take(72).collect::<String>())
                    } else {
                        t
                    }
                };
                r.values = r.keys.iter().map(|k| short(&c.text[k])).collect();
                r.groups = group_names(have.iter().cloned(), table);
            }
            r.refusal = Some(e.msg);
        }
    }
    Ok(r)
}

fn gib(b: u64) -> String {
    format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64)
}

fn lines(title: &str, v: &[Named], out: &mut String) {
    let n: usize = v.iter().map(|g| g.tensors).sum();
    out.push_str(&format!("{title}: {n} tensor{} in {} group{}\n", if n == 1 { "" } else { "s" }, v.len(), if v.len() == 1 { "" } else { "s" }));
    for g in v {
        let what = if g.dtype.is_empty() { String::new() } else { format!("  {} {:?}", g.dtype, g.shape) };
        out.push_str(&format!("  {:>6} x  {}{what}\n", g.tensors, g.mask));
    }
}

/// The report as the command prints it.
pub fn render(name: &str, r: &Report) -> String {
    let mut s = format!("{name}: model type {}", if r.model_type.is_empty() { "(none)" } else { &r.model_type });
    if !r.text_type.is_empty() && r.text_type != r.model_type {
        s.push_str(&format!(", text model {}", r.text_type));
    }
    if !r.arch.is_empty() {
        s.push_str(&format!(", {}", r.arch));
    }
    s.push_str(&format!("; {} tensors, {}\n", r.tensors, gib(r.bytes)));
    match &r.refusal {
        None => {
            let kinds: Vec<String> = r.kinds.iter().map(|(k, n)| format!("{n} {k}")).collect();
            s.push_str(&format!("Describes. Blocks: {}\n", kinds.join(", ")));
            if r.ok() {
                s.push_str("Every tensor of the checkpoint is read, or is a part its description leaves; every tensor it reads is there.\n");
            }
            if !r.not_read.is_empty() {
                lines("In the checkpoint, not read by the description", &r.not_read, &mut s);
            }
            if !r.missing.is_empty() {
                lines("Read by the description, not in the checkpoint", &r.missing, &mut s);
            }
        }
        Some(why) => {
            s.push_str(&format!("Does not describe: {why}\n"));
            if r.no_builder {
                s.push_str(&format!(
                    "No builder takes this model type: a new family. Its text config's keys that say something of its shape ({}):\n",
                    r.keys.len()
                ));
                for (k, v) in r.keys.iter().zip(&r.values) {
                    s.push_str(&format!("  {k} = {v}\n"));
                }
                lines("Its checkpoint", &r.groups, &mut s);
            } else if !r.keys.is_empty() {
                s.push_str(&format!("Config keys its family's builder does not understand ({}):\n", r.keys.len()));
                for k in &r.keys {
                    s.push_str(&format!("  {k}\n"));
                }
            }
        }
    }
    // (a model that describes has its kinds above: signs are for the one that does not)
    if r.refusal.is_some() && !r.signs.is_empty() {
        s.push_str("Signs of layer kinds (a word in a name: read the release's code before building):\n");
        for (word, at, kind, has) in &r.signs {
            s.push_str(&format!("  {} {kind}  [\"{word}\" in {at}]\n", if *has { "has  " } else { "LACKS" }));
        }
    }
    s
}

/// The config's text and the table of MODEL: a recorded header's file, a directory, else a Hub repo by ranges (with its record, for `--record`).
fn read(model: &str) -> Result<(String, String, Table, Option<serde_json::Value>)> {
    let p = Path::new(model);
    if p.is_file() {
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(p).map_err(|e| Error(format!("{model}: {e}")))?).map_err(|e| Error(format!("{model}: {e}")))?;
        let groups = sizes::groups_of(&v).ok_or_else(|| Error(format!("{model}: not a recorded header (tests/fixtures/headers)")))?;
        let config = v["config"].as_str().ok_or_else(|| Error(format!("{model}: no config in the record")))?.to_string();
        let name = v["repo"].as_str().unwrap_or(model).to_string();
        return Ok((name, config, Table::expand(&groups), None));
    }
    if p.is_dir() {
        let config = std::fs::read_to_string(p.join("config.json")).map_err(|e| Error(format!("{}: {e}", p.join("config.json").display())))?;
        return Ok((model.to_string(), config, Table::from_dir(p).map_err(Error)?, None));
    }
    let hub = hub::Hub::from_env();
    let remote = hub.list(model, "main").map_err(|e| Error(format!("{model}: {e}")))?;
    let table = Table::from_hub(&hub, &remote).map_err(|e| Error(format!("{model}: {e}")))?;
    let record = sizes::header_record(&hub, &remote, &table);
    let config = record["config"].as_str().unwrap_or_default().to_string();
    if config.is_empty() {
        return err(format!("{model}: no config.json"));
    }
    Ok((model.to_string(), config, table, Some(record)))
}

/// `glyd-engine intake MODEL [--record DIR]`: prints the report; whether the model describes and its checkpoint is read whole.
pub fn run(a: &Args) -> Result<bool> {
    let model = a.need(1, "MODEL (a Hub repo, a directory, or a recorded header's file)")?;
    let (name, config, table, record) = read(model)?;
    let r = inspect(&config, &table)?;
    print!("{}", render(&name, &r));
    if let Some(dir) = a.flag("record") {
        if dir.is_empty() {
            return err("--record DIR: the folder to write the record into (glyd-engine/tests/fixtures/headers)");
        }
        let Some(v) = record else { return err("--record takes a Hub repo: a directory's or a record's header is already here") };
        let file = Path::new(dir).join(format!("{}.json", name.rsplit('/').next().unwrap_or(&name)));
        std::fs::create_dir_all(dir).map_err(|e| Error(format!("{dir}: {e}")))?;
        std::fs::write(&file, serde_json::to_string(&v).map_err(|e| Error(e.to_string()))?).map_err(|e| Error(format!("{}: {e}", file.display())))?;
        println!("Recorded {}", file.display());
    }
    Ok(r.ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(name: &str) -> (String, Table) {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/headers").join(name);
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
        (v["config"].as_str().unwrap().to_string(), Table::expand(&sizes::groups_of(&v).unwrap()))
    }

    /// A release the engine runs describes, and its checkpoint is read whole: nothing to do.
    #[test]
    fn a_supported_release_is_read_whole() {
        for name in ["gemma-4-12B-it.json", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.json", "Qwen3.8-27B.json"] {
            let (config, table) = record(name);
            let r = inspect(&config, &table).unwrap();
            assert!(r.ok(), "{name}: {:?} {:?} {:?}", r.refusal, r.not_read, r.missing);
            assert!(!r.kinds.is_empty() && r.tensors == table.rows.len(), "{name}");
            assert!(render(name, &r).contains("Describes. Blocks: "), "{name}");
        }
    }

    /// A release of a family the engine has whose config grew keys: every one of them is named at once, and a tensor the checkpoint gained or lost is named by
    /// its masked name with how many layers have it.
    #[test]
    fn a_changed_release_names_every_new_key_and_tensor() {
        let (config, table) = record("Qwen3.8-27B.json");
        let mut cfg: serde_json::Value = serde_json::from_str(&config).unwrap();
        let text = if cfg.get("text_config").is_some() { &mut cfg["text_config"] } else { &mut cfg };
        text["zz_new_gate"] = serde_json::json!(true);
        text["zz_other_width"] = serde_json::json!(64);
        let r = inspect(&cfg.to_string(), &table).unwrap();
        assert!(!r.ok() && !r.no_builder, "{:?}", r.refusal);
        assert_eq!(r.keys, ["zz_new_gate", "zz_other_width"], "{:?}", r.refusal);
        let page = render("m", &r);
        assert!(page.contains("does not understand (2)") && page.contains("  zz_new_gate\n") && page.contains("  zz_other_width\n"), "{page}");

        // (the same checkpoint with a tensor more in every layer and one of the description's gone from layer 0)
        let mut more = table.clone();
        let first = more.rows.iter().find(|r| r.name.contains("layers.0.") && r.name.ends_with(".weight")).unwrap().clone();
        let layers: BTreeSet<usize> = more.rows.iter().filter_map(|r| sizes::layer_of(&r.name)).collect();
        for l in &layers {
            more.rows.push(sizes::Row { name: format!("model.language_model.layers.{l}.zz_added.weight"), ..first.clone() });
        }
        more.rows.retain(|r| r.name != first.name);
        let r = inspect(&config, &more).unwrap();
        assert!(!r.ok() && r.refusal.is_none());
        assert_eq!(r.not_read.len(), 1, "{:?}", r.not_read);
        assert_eq!((r.not_read[0].mask.as_str(), r.not_read[0].tensors), ("model.language_model.layers.{L}.zz_added.weight", layers.len()));
        assert_eq!(r.missing.iter().map(|g| g.tensors).sum::<usize>(), 1, "{:?}", r.missing);
        assert!(render("m", &r).contains("In the checkpoint, not read by the description: "), "{}", render("m", &r));
    }

    /// A model type no builder takes: the refusal, the text config's keys that say something of its shape (the inert ones left out), the checkpoint's groups
    /// with types and shapes, and the signs its names give: a kind the engine has, and one it lacks.
    #[test]
    fn a_new_family_gets_its_keys_groups_and_signs() {
        let config = serde_json::json!({
            "architectures": ["NewThingForCausalLM"], "model_type": "new_thing", "hidden_size": 64, "num_hidden_layers": 2, "vocab_size": 100,
            "num_local_experts": 8, "conv_kernel": 3, "num_loops": 2, "bos_token_id": 1, "transformers_version": "5.0"
        })
        .to_string();
        let row = |name: &str, shape: &[u64]| sizes::Row {
            name: name.to_string(),
            dtype: "BF16".into(),
            shape: shape.to_vec(),
            bytes: shape.iter().product::<u64>() * 2,
        };
        let mut rows = vec![row("model.embed_tokens.weight", &[100, 64])];
        for l in 0..2 {
            // (the second layer's is another width: a line of its own)
            rows.push(row(&format!("model.layers.{l}.conv.weight"), &[64, 1, 3 + 2 * l]));
            for e in 0..8 {
                rows.push(row(&format!("model.layers.{l}.mlp.experts.{e}.up.weight"), &[128, 64]));
            }
        }
        rows.sort_by(|a, b| a.name.cmp(&b.name));
        let r = inspect(&config, &Table { rows }).unwrap();
        assert!(!r.ok() && r.no_builder, "{:?}", r.refusal);
        // (in the config's own order, the inert ones left out)
        assert_eq!(r.keys, ["hidden_size", "num_hidden_layers", "vocab_size", "num_local_experts", "conv_kernel", "num_loops"]);
        let groups: Vec<(&str, usize)> = r.groups.iter().map(|g| (g.mask.as_str(), g.tensors)).collect();
        assert_eq!(
            groups,
            [
                ("model.embed_tokens.weight", 1),
                ("model.layers.{L}.conv.weight", 1),
                ("model.layers.{L}.conv.weight", 1),
                ("model.layers.{L}.mlp.experts.{E}.up.weight", 16)
            ]
        );
        assert_eq!((&r.groups[1].shape[..], &r.groups[2].shape[..], &r.groups[3].shape[..]), (&[64, 1, 3][..], &[64, 1, 5][..], &[128, 64][..]));
        assert_eq!(r.values[3], "8");
        let has: Vec<(&str, bool)> = r.signs.iter().map(|(w, _, _, h)| (w.as_str(), *h)).collect();
        assert_eq!(has, [("expert", true), ("conv", true), ("num_loops", false)], "{:?}", r.signs);
        let page = render("org/new", &r);
        assert!(
            page.contains("a new family")
                && page.contains("LACKS looped depth")
                && page.contains("has   a short convolution")
                && page.contains("has   a mixture of experts"),
            "{page}"
        );
        assert!(page.contains("    16 x  model.layers.{L}.mlp.experts.{E}.up.weight  BF16 [128, 64]"), "{page}");
    }

    /// A sign is whole words of a name: the names that misled the first run on released configs do not match.
    #[test]
    fn a_sign_is_whole_words() {
        for (name, word, is) in [
            ("short_conv_kernel_size", "conv", true),
            ("model.layers.{L}.conv1d.weight", "conv", true),
            ("modules_to_not_convert", "conv", false),
            ("layers.{L}.attention.indexer.ssmax_scaling.b", "ssm", false),
            ("ssm_cfg", "ssm", true),
            ("n_routed_experts", "expert", true),
            ("kv_lora_rank", "kv_lora", true),
            ("num_nextn_predict_layers", "nextn_predict", true),
            ("vision_config", "vision", true),
            ("revision", "vision", false),
            ("hc_mult", "hc", true),
        ] {
            assert_eq!(word_in(name, word), is, "{word} in {name}");
        }
    }

    /// `--record DIR` takes its folder (elsewhere `--record` is a switch: `args::BY_COMMAND`).
    #[test]
    fn record_takes_a_folder() {
        let a = Args::from(["intake", "org/model", "--record", "some/dir"].map(String::from));
        assert_eq!((a.flag("record"), a.need(1, "MODEL").unwrap()), (Some("some/dir"), "org/model"));
    }

    /// The words of a refusal give its keys back, one or several, and another refusal gives none.
    #[test]
    fn a_refusals_keys_are_read_back() {
        assert_eq!(describe::keys_named("qwen3_5: config key some_key is not understood by the description builder"), ["some_key"]);
        assert_eq!(describe::keys_named("glm: config keys a_b, c_d, e are not understood by the description builder"), ["a_b", "c_d", "e"]);
        assert!(describe::keys_named("model_type x has no description builder yet").is_empty());
    }
}
