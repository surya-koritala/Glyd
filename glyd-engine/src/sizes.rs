//! Exact sizes from the files' headers (ROADMAP P2b): every tensor of a checkpoint, by name, type, shape and bytes, from the safetensors headers of a directory or,
//! where the model is not here yet, of the Hub's repo by two ranges a shard (its length, then its JSON): nothing of the weights is downloaded. A table is classified
//! by name into the parts a plan counts (`Part`: the embedding, the head, attention, dense MLPs, routed experts, shared experts, a vision tower, a multi-token-prediction
//! head), kept in the cache by repo and commit, and sizes the dynamic route's layers before its files are local (src/plan.rs `sizing::size_dynamic`) and the planner's shape
//! (`plan::Shape::with_table`) in place of an estimate from config.json.

use crate::hub;
use glyd_gpu::json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The file a planning directory made from a GGUF keeps the file's tensors in (`plan_dir`), as the cache keeps a repo's.
pub const GGUF_TENSORS: &str = "glyd-tensors.json";

/// One tensor of a checkpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub bytes: u64,
}

impl Row {
    /// Weights in the tensor, where its type says (a packed type's, as the release packs them, is not told by its shape: None).
    pub fn params(&self) -> Option<u64> {
        matches!(self.dtype.as_str(), "BF16" | "F16" | "F32" | "F64" | "F8_E4M3" | "F8_E5M2").then(|| self.shape.iter().product())
    }
}

/// What a tensor is of a model, for what a plan counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Part {
    Embedding,
    Head,
    /// Attention of every kind (latent, delta, state-space mixers' projections) and its norms.
    Attention,
    /// Dense MLPs, routers and the norms between layers.
    Mlp,
    RoutedExperts,
    SharedExperts,
    /// A vision, audio or multimodal tower: counted only where the model is run with one.
    Vision,
    /// A multi-token-prediction head: counted only where it is drafted with.
    Mtp,
    Other,
}

/// The layer a tensor's name is of (`...layers.12....`).
pub fn layer_of(name: &str) -> Option<usize> {
    let rest = &name[name.find("layers.")? + 7..];
    rest.split('.').next()?.parse().ok()
}

/// The part of a model the tensor `name` is, of a model of `layers` layers (a layer past them is a multi-token-prediction one).
pub fn classify(name: &str, layers: usize) -> Part {
    let has = |words: &[&str]| words.iter().any(|w| name.contains(w));
    if has(&["vision", "visual", "audio", "multi_modal", "mm_projector", "patch_embed", "image_newline"]) {
        return Part::Vision;
    }
    let layer = layer_of(name);
    if has(&["mtp", "nextn"]) || layer.is_some_and(|l| l >= layers) {
        return Part::Mtp;
    }
    if layer.is_none() {
        return if has(&["lm_head", "output.weight", "head.weight"]) {
            Part::Head
        } else if has(&["embed_tokens", "wte", "embeddings", "tok_embeddings", "embed.weight", "embedding"]) {
            Part::Embedding
        } else {
            Part::Other
        };
    }
    if has(&["shared_expert", "shared_experts"]) {
        Part::SharedExperts
    } else if has(&[".experts.", ".expert."]) {
        Part::RoutedExperts
    } else if has(&["attn", "attention", "mixer", "self_attn", "ssm", "conv1d"]) {
        Part::Attention
    } else {
        Part::Mlp
    }
}

/// A checkpoint's tensors.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Table {
    pub rows: Vec<Row>,
}

impl Table {
    /// A directory's safetensors files, by their headers.
    pub fn from_dir(dir: &Path) -> Result<Table, String> {
        // (the files' own headers first: a table file beside real weights is not believed. A planning directory made from a GGUF (`gguf::plan_dir`) has no weights, and holds its
        // tensors in a file of its own)
        let ck = match glyd_gpu::safetensors::Checkpoint::open(dir) {
            Ok(ck) => ck,
            Err(e) => return Table::read(&dir.join(GGUF_TENSORS)).ok_or_else(|| format!("{}: {e}", dir.display())),
        };
        let mut rows: Vec<Row> = ck
            .files
            .iter()
            .flat_map(|f| f.tensors.iter().map(|(name, t)| Row { name: name.clone(), dtype: t.dtype.clone(), shape: t.shape.clone(), bytes: t.bytes }))
            .collect();
        rows.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(Table { rows })
    }

    /// A GGUF file's tensors (its header's table) under the Hugging Face names the engine loads where it knows them, the shape as Hugging Face lists it and the bytes
    /// a run holds of each (bf16: what `materialize` makes of the file's types).
    pub fn from_gguf(g: &crate::gguf::Gguf) -> Table {
        let mut rows: Vec<Row> = g
            .tensors
            .iter()
            .map(|t| Row {
                name: crate::gguf::hf_name(&t.name).unwrap_or_else(|| t.name.clone()),
                // (a run holds every tensor of the file as bf16: src/gguf.rs `materialize`)
                dtype: "BF16".to_string(),
                shape: t.shape(),
                bytes: t.elements() * 2,
            })
            .collect();
        rows.sort_by(|a, b| a.name.cmp(&b.name));
        Table { rows }
    }

    /// The Hub repo `remote`'s safetensors files by their headers, read by ranges on up to 16 connections and kept in the cache by repo and commit.
    pub fn from_hub(hub: &hub::Hub, remote: &hub::Remote) -> Result<Table, String> {
        Table::from_hub_in(&crate::store::cache_root(), hub, remote)
    }

    pub(crate) fn from_hub_in(root: &Path, hub: &hub::Hub, remote: &hub::Remote) -> Result<Table, String> {
        let file = cache_file(root, &remote.repo, &remote.commit);
        if let Some(t) = Table::read(&file) {
            return Ok(t);
        }
        let shards: Vec<&hub::Entry> = remote.files.iter().filter(|e| e.path.ends_with(".safetensors") && !e.path.contains('/')).collect();
        if shards.is_empty() {
            return Err(format!("{} has no safetensors files", remote.repo));
        }
        eprintln!("Reading the file headers of {} by ranges (nothing of the weights is downloaded)...", remote.repo);
        let next = std::sync::atomic::AtomicUsize::new(0);
        let found: std::sync::Mutex<Vec<Result<Vec<Row>, String>>> = std::sync::Mutex::new(Vec::new());
        std::thread::scope(|s| {
            for _ in 0..shards.len().min(16) {
                s.spawn(|| loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(e) = shards.get(i) else { break };
                    let r = hub.header(remote, e).map_err(|err| format!("{}: {err}", e.path)).map(|(h, _)| {
                        h.into_iter().map(|(name, dtype, shape, a, b)| Row { name, dtype, shape, bytes: b.saturating_sub(a) }).collect::<Vec<Row>>()
                    });
                    if let Ok(mut f) = found.lock() {
                        f.push(r);
                    }
                });
            }
        });
        let mut rows = Vec::new();
        for r in found.into_inner().map_err(|e| e.to_string())? {
            rows.extend(r?);
        }
        rows.sort_by(|a, b| a.name.cmp(&b.name));
        let t = Table { rows };
        t.write(&file);
        Ok(t)
    }

    /// The table of MODEL where it can be had without downloading its weights: a directory, a model in the Hub's cache, else the Hub's by ranges (None where
    /// offline, where the Hub does not answer or has no such model).
    pub fn find(model: &str) -> Option<Table> {
        let name = crate::levels::split_tag(model).0;
        if Path::new(name).join("config.json").exists() {
            return Table::from_dir(Path::new(name)).ok();
        }
        if let Ok(s) = glyd_gpu::save::Source::find(name) {
            return Table::from_dir(&s.dir).ok();
        }
        if hub::offline() || name.matches('/').count() != 1 {
            return None;
        }
        let h = hub::Hub::from_env();
        let remote = h.list(name, "main").ok()?;
        Table::from_hub(&h, &remote).ok()
    }

    pub(crate) fn read(file: &Path) -> Option<Table> {
        let v: Value = glyd_gpu::json::parse(&std::fs::read_to_string(file).ok()?).ok()?;
        let rows = v
            .as_array()?
            .iter()
            .map(|r| {
                let r = r.as_array()?;
                Some(Row {
                    name: r.first()?.as_str()?.to_string(),
                    dtype: r.get(1)?.as_str()?.to_string(),
                    shape: r.get(2)?.as_u64s()?,
                    bytes: r.get(3)?.as_u64()?,
                })
            })
            .collect::<Option<Vec<Row>>>()?;
        (!rows.is_empty()).then_some(Table { rows })
    }

    /// A failure to keep the table is not the plan's.
    pub(crate) fn write(&self, file: &Path) {
        let rows: Vec<serde_json::Value> = self.rows.iter().map(|r| serde_json::json!([r.name, r.dtype, r.shape, r.bytes])).collect();
        if let Some(dir) = file.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(file, serde_json::Value::Array(rows).to_string());
    }

    /// The bytes of the checkpoint's tensors, all of them.
    pub fn total_bytes(&self) -> u64 {
        self.rows.iter().map(|r| r.bytes).sum()
    }

    /// The bytes of each part of a model of `layers` layers.
    pub fn by_part(&self, layers: usize) -> BTreeMap<Part, u64> {
        let mut m = BTreeMap::new();
        for r in &self.rows {
            *m.entry(classify(&r.name, layers)).or_insert(0) += r.bytes;
        }
        m
    }

    /// The weights of each part (the tensors whose type tells them; a packed type's bytes are `packed_bits` a weight): the parameters a plan counts.
    pub fn params_by_part(&self, layers: usize, packed_bits: f64) -> BTreeMap<Part, f64> {
        let mut m = BTreeMap::new();
        for r in &self.rows {
            let p = r.params().map_or(r.bytes as f64 * 8.0 / packed_bits, |n| n as f64);
            *m.entry(classify(&r.name, layers)).or_insert(0.0) += p;
        }
        m
    }

    /// The bytes of each layer of `layers` and the rest's, of the tensors of the parts `keep` takes (`Vision` and `Mtp` are not a layer's: the tensors no description loads).
    pub fn per_layer(&self, layers: usize, keep: &dyn Fn(&Row) -> bool, bytes: &dyn Fn(&Row) -> u64) -> (Vec<u64>, u64) {
        let (mut per, mut base) = (vec![0u64; layers], 0u64);
        for r in self.rows.iter().filter(|r| !matches!(classify(&r.name, layers), Part::Vision | Part::Mtp) && keep(r)) {
            match layer_of(&r.name) {
                Some(l) if l < layers => per[l] += bytes(r),
                Some(_) => {}
                None => base += bytes(r),
            }
        }
        (per, base)
    }
}

/// A table in a form small enough to commit as a fixture: the names with their layer and expert numbers masked (`model.layers.{L}.mlp.experts.{E}.w1.weight`), a
/// group for each distinct (mask, type, shape, bytes) with the layers it is in and how many experts it has. `expand` gives the table back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Group {
    pub mask: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub bytes: u64,
    /// The layers it is in (empty: no layer number in the name).
    pub layers: Vec<usize>,
    /// How many experts of each layer (0: the name has no expert number).
    pub experts: usize,
}

fn mask_numbers(name: &str) -> (String, Option<usize>, Option<usize>) {
    let (mut out, mut layer, mut expert) = (name.to_string(), None, None);
    for (key, slot, tag) in [("layers.", 0, "{L}"), (".experts.", 1, "{E}")] {
        if let Some(i) = out.find(key) {
            let from = i + key.len();
            let digits: String = out[from..].chars().take_while(char::is_ascii_digit).collect();
            if let Ok(n) = digits.parse::<usize>() {
                out.replace_range(from..from + digits.len(), tag);
                if slot == 0 {
                    layer = Some(n)
                } else {
                    expert = Some(n)
                }
            }
        }
    }
    (out, layer, expert)
}

impl Table {
    /// The table as groups (`Group`), in a fixed order.
    pub fn compact(&self) -> Vec<Group> {
        let mut m: BTreeMap<(String, String, Vec<u64>, u64), (std::collections::BTreeSet<usize>, std::collections::BTreeMap<usize, usize>)> = BTreeMap::new();
        for r in &self.rows {
            let (mask, layer, expert) = mask_numbers(&r.name);
            let e = m.entry((mask, r.dtype.clone(), r.shape.clone(), r.bytes)).or_default();
            let l = layer.unwrap_or(usize::MAX);
            e.0.insert(l);
            if expert.is_some() {
                *e.1.entry(l).or_insert(0) += 1;
            }
        }
        m.into_iter()
            .map(|((mask, dtype, shape, bytes), (layers, experts))| Group {
                experts: experts.values().copied().max().unwrap_or(0),
                layers: layers.into_iter().filter(|&l| l != usize::MAX).collect(),
                mask,
                dtype,
                shape,
                bytes,
            })
            .collect()
    }

    /// The table of `groups`.
    pub fn expand(groups: &[Group]) -> Table {
        let mut rows = Vec::new();
        for g in groups {
            let layers: Vec<Option<usize>> = if g.layers.is_empty() { vec![None] } else { g.layers.iter().map(|&l| Some(l)).collect() };
            for l in layers {
                let experts: Vec<Option<usize>> = if g.experts == 0 { vec![None] } else { (0..g.experts).map(Some).collect() };
                for e in experts {
                    let mut name = g.mask.clone();
                    if let Some(l) = l {
                        name = name.replace("{L}", &l.to_string());
                    }
                    if let Some(e) = e {
                        name = name.replace("{E}", &e.to_string());
                    }
                    rows.push(Row { name, dtype: g.dtype.clone(), shape: g.shape.clone(), bytes: g.bytes });
                }
            }
        }
        rows.sort_by(|a, b| a.name.cmp(&b.name));
        Table { rows }
    }
}

fn cache_file(root: &Path, repo: &str, commit: &str) -> PathBuf {
    let slug: String = format!("{repo}-{commit}").chars().map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '-' { c } else { '_' }).collect();
    root.join("headers").join(format!("{slug}.json"))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn tensors_are_classified_by_name() {
        let l = 61;
        let c = |n: &str| classify(n, l);
        assert_eq!(c("model.embed_tokens.weight"), Part::Embedding);
        assert_eq!(c("model.language_model.embed_tokens.weight"), Part::Embedding);
        assert_eq!(c("lm_head.weight"), Part::Head);
        assert_eq!(c("model.layers.3.self_attn.q_a_proj.weight"), Part::Attention);
        assert_eq!(c("model.layers.3.input_layernorm.weight"), Part::Mlp);
        assert_eq!(c("model.layers.3.mlp.gate_proj.weight"), Part::Mlp);
        assert_eq!(c("model.layers.4.mlp.experts.17.down_proj.weight"), Part::RoutedExperts);
        assert_eq!(c("model.layers.4.block_sparse_moe.experts.17.w1.weight"), Part::RoutedExperts);
        assert_eq!(c("model.layers.4.mlp.shared_experts.up_proj.weight"), Part::SharedExperts);
        assert_eq!(c("model.layers.4.mlp.gate.weight"), Part::Mlp);
        assert_eq!(c("backbone.layers.2.mixer.in_proj.weight"), Part::Attention);
        assert_eq!(c("model.visual.patch_embed.weight"), Part::Vision);
        assert_eq!(c("model.vision_tower.blocks.0.attn.qkv.weight"), Part::Vision);
        assert_eq!(c("mtp.fc.weight"), Part::Mtp);
        assert_eq!(c("model.layers.61.eh_proj.weight"), Part::Mtp, "a layer past the model's");
        assert_eq!(c("model.norm.weight"), Part::Other);
    }

    /// A table file in a directory that has weights is not believed: the weights' own headers are the sizes; a planning directory with none reads its table file.
    #[test]
    fn a_table_file_does_not_override_the_weights_headers() {
        let dir = std::env::temp_dir().join(format!("glyd-sizes-trust-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let fake = Table { rows: vec![Row { name: "model.embed_tokens.weight".into(), dtype: "BF16".into(), shape: vec![1], bytes: 2 }] };
        fake.write(&dir.join(GGUF_TENSORS));
        assert_eq!(Table::from_dir(&dir).unwrap(), fake, "no weights: the planning directory's table");
        std::fs::write(dir.join("model.safetensors"), safetensors(&[("model.embed_tokens.weight", "BF16", &[64, 32])])).unwrap();
        let t = Table::from_dir(&dir).unwrap();
        assert_eq!((t.rows.len(), t.rows[0].bytes), (1, 64 * 32 * 2), "the weights' headers, not the file beside them");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_table_is_kept_and_read_again() {
        let t = Table {
            rows: vec![
                Row { name: "a.weight".into(), dtype: "BF16".into(), shape: vec![4, 8], bytes: 64 },
                Row { name: "b.weight".into(), dtype: "U8".into(), shape: vec![3], bytes: 3 },
            ],
        };
        let root = std::env::temp_dir().join(format!("glyd-sizes-{}", std::process::id()));
        let file = cache_file(&root, "org/name", "abc123");
        t.write(&file);
        assert_eq!(Table::read(&file), Some(t.clone()));
        assert_eq!(t.rows[0].params(), Some(32));
        assert_eq!(t.rows[1].params(), None);
        assert_eq!(t.total_bytes(), 67);
        assert_eq!(Table::read(&cache_file(&root, "org/name", "other")), None);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A safetensors file of `tensors` (name, type, shape) with `fill` bytes of data each: its header and zeros.
    pub(crate) fn safetensors(tensors: &[(&str, &str, &[u64])]) -> Vec<u8> {
        let (mut header, mut at) = (String::from("{"), 0u64);
        for (i, (name, dtype, shape)) in tensors.iter().enumerate() {
            let w = match *dtype {
                "F32" => 4,
                "F8_E4M3" | "U8" => 1,
                _ => 2,
            };
            let n = shape.iter().product::<u64>() * w;
            header += &format!("{}\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":{shape:?},\"data_offsets\":[{at},{}]}}", if i == 0 { "" } else { "," }, at + n);
            at += n;
        }
        header += "}";
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend(header.as_bytes());
        out.resize(out.len() + at as usize, 0);
        out
    }

    /// A Hub repo's sizes come from its shards' headers by ranges alone: two small ranges a shard, no data, and the second time none at all (the cache).
    #[test]
    fn a_hub_repo_is_sized_by_ranges_alone() {
        let a = safetensors(&[("model.embed_tokens.weight", "BF16", &[64, 32]), ("model.layers.0.self_attn.q_proj.weight", "BF16", &[32, 32])]);
        let b = safetensors(&[("model.layers.1.mlp.experts.0.w1.weight", "F8_E4M3", &[64, 64]), ("lm_head.weight", "BF16", &[64, 32])]);
        let (a_len, b_len) = (a.len(), b.len());
        let (m, h, dir) = crate::hub::tests::mock_with(vec![
            ("config.json".into(), b"{}".to_vec()),
            ("model-00001-of-00002.safetensors".into(), a),
            ("model-00002-of-00002.safetensors".into(), b),
        ]);
        let remote = h.list("org/model", "main").unwrap();
        let root = std::env::temp_dir().join(format!("glyd-sizes-hub-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let t = Table::from_hub_in(&root, &h, &remote).unwrap();
        let names: Vec<(&str, u64)> = t.rows.iter().map(|r| (r.name.as_str(), r.bytes)).collect();
        assert_eq!(
            names,
            vec![
                ("lm_head.weight", 64 * 32 * 2),
                ("model.embed_tokens.weight", 64 * 32 * 2),
                ("model.layers.0.self_attn.q_proj.weight", 32 * 32 * 2),
                ("model.layers.1.mlp.experts.0.w1.weight", 64 * 64),
            ]
        );
        let ranged: Vec<u64> = m
            .log
            .lock()
            .unwrap()
            .iter()
            .filter(|(p, _, _)| p.starts_with("/store/"))
            .map(|(_, range, _)| {
                let r = range.as_deref().expect("every read of a shard is a range").trim_start_matches("bytes=").to_string();
                let (x, y) = r.split_once('-').unwrap();
                y.parse::<u64>().unwrap() - x.parse::<u64>().unwrap() + 1
            })
            .collect();
        assert_eq!(ranged.len(), 4, "two ranges of each of two shards: {ranged:?}");
        assert!(ranged.iter().all(|&n| n < (a_len.min(b_len) as u64)), "no read is a shard's data: {ranged:?}");
        let before = m.log.lock().unwrap().len();
        assert_eq!(Table::from_hub_in(&root, &h, &remote).unwrap(), t);
        assert_eq!(m.log.lock().unwrap().len(), before, "the second time is the cache's");
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The recorded headers: for each model the files the Hub listed (path, bytes), the index's total, the config.json and the table of tensors, compact
    /// (tests/fixtures/headers/NAME.json; `record_header_fixtures` writes them).
    const HEADER_FIXTURES: [(&str, &str); 4] = [
        ("gemma-4-12B-it", include_str!("../tests/fixtures/headers/gemma-4-12B-it.json")),
        ("Kimi-K3", include_str!("../tests/fixtures/headers/Kimi-K3.json")),
        ("DeepSeek-V4-Flash-0731", include_str!("../tests/fixtures/headers/DeepSeek-V4-Flash-0731.json")),
        ("Qwen3.8-27B", include_str!("../tests/fixtures/headers/Qwen3.8-27B.json")),
    ];

    fn groups_of(v: &serde_json::Value) -> Vec<Group> {
        v["groups"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| Group {
                mask: g["mask"].as_str().unwrap().to_string(),
                dtype: g["dtype"].as_str().unwrap().to_string(),
                shape: g["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect(),
                bytes: g["bytes"].as_u64().unwrap(),
                layers: g["layers"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect(),
                experts: g["experts"].as_u64().unwrap() as usize,
            })
            .collect()
    }

    /// Records the headers of the recent models by ranges from the Hub (the files' headers and the small files; no weights) into tests/fixtures/headers:
    /// `GLYD_RECORD_HEADERS=1 cargo test -p glyd-engine --lib -- --ignored record_header_fixtures`. Recorded and committed so far: four of the nine repos
    /// listed below (gemma-4-12B-it, Kimi-K3, DeepSeek-V4-Flash-0731, Qwen3.8-27B). The others (Muse Spark, and #333's releases: GLM-5.3, Qwen3.8-2.4T-A95B,
    /// DeepSeek-V4.1-Flash, DeepSeek-V4-Flash's DSpark drafter) are listed and not recorded: no header of theirs is in the tree, so
    /// `runtime::describe_tests::every_recorded_release_describes_or_says_what_it_lacks`, which holds each file here, holds none of them.
    #[test]
    #[ignore = "needs the network: GLYD_RECORD_HEADERS=1"]
    fn record_header_fixtures() {
        if std::env::var_os("GLYD_RECORD_HEADERS").is_none() {
            return;
        }
        let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/headers");
        std::fs::create_dir_all(&out).unwrap();
        let hub = hub::Hub::from_env();
        let root = std::env::temp_dir().join(format!("glyd-record-headers-{}", std::process::id()));
        for repo in [
            "google/gemma-4-12B-it",
            "moonshotai/Kimi-K3",
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            "meta-models/Muse-Spark-Open-816B-A42B",
            "Qwen/Qwen3.8-27B",
            "zai-org/GLM-5.3",
            "Qwen/Qwen3.8-2.4T-A95B",
            "deepseek-ai/DeepSeek-V4.1-Flash",
            "deepseek-ai/DeepSeek-V4-Flash-DSpark",
        ] {
            let remote = hub.list(repo, "main").unwrap_or_else(|e| panic!("{repo}: {e}"));
            let table = match Table::from_hub_in(&root, &hub, &remote) {
                Ok(t) => t,
                // (a gated repo's files answer 401 without the owner's token, which this does not look for: left out, and said)
                Err(e) if e.contains("401") => {
                    eprintln!("{repo}: skipped, the Hub answered 401 (gated: record it with HF_TOKEN set)");
                    continue;
                }
                Err(e) => panic!("{repo}: {e}"),
            };
            let groups = table.compact();
            assert_eq!(Table::expand(&groups), table, "{repo}: the compact table gives the table back");
            let listed: Vec<serde_json::Value> =
                remote.files.iter().filter(|e| e.path.ends_with(".safetensors")).map(|e| serde_json::json!({"path": e.path, "bytes": e.size})).collect();
            let index = hub.text(repo, &remote.commit, "model.safetensors.index.json").ok().and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok());
            let config = hub.text(repo, &remote.commit, "config.json").unwrap_or_default();
            let v = serde_json::json!({
                "repo": repo, "commit": remote.commit, "listed_safetensors": listed,
                "index_total_size": index.as_ref().and_then(|i| i["metadata"]["total_size"].as_u64()),
                "config": config,
                "groups": groups.iter().map(|g| serde_json::json!({"mask": g.mask, "dtype": g.dtype, "shape": g.shape, "bytes": g.bytes, "layers": g.layers, "experts": g.experts})).collect::<Vec<_>>(),
            });
            let name = repo.rsplit('/').next().unwrap();
            std::fs::write(out.join(format!("{name}.json")), serde_json::to_string(&v).unwrap()).unwrap();
            eprintln!("{repo}: {} tensors in {} groups, {} files", table.rows.len(), groups.len(), listed.len());
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    /// P2b's acceptance on recorded headers of the recent families (Gemma 4 12B, Kimi-K3, DeepSeek-V4, Muse Spark, Qwen3.8-27B): the tensors the headers list are the
    /// bytes the Hub lists for the files (their headers are all that differ: under 1%, and the index's total where there is one is exactly them); the model's
    /// shape counted from them is within 1% of the weights in them less the vision tower and a prediction head not drafted with; and the plan sized from them is made.
    #[test]
    fn recorded_headers_are_the_hubs_file_sizes() {
        use crate::plan::Shape;
        for (name, text) in HEADER_FIXTURES {
            let v: serde_json::Value = serde_json::from_str(text).unwrap();
            let table = Table::expand(&groups_of(&v));
            assert_eq!(table.compact(), groups_of(&v), "{name}: the table is its compact form");
            let listed: u64 = v["listed_safetensors"].as_array().unwrap().iter().map(|f| f["bytes"].as_u64().unwrap()).sum();
            let total = table.total_bytes();
            assert!(total <= listed && (listed - total) as f64 / (listed as f64) < 0.01, "{name}: {total} bytes of tensors in {listed} bytes of files");
            if let Some(idx) = v["index_total_size"].as_u64() {
                assert_eq!(total, idx, "{name}: the index's total size");
            }
            let sh = Shape::from_config(v["config"].as_str().unwrap()).unwrap().with_table(&table);
            let layers = sh.layers;
            let counted: f64 = table
                .rows
                .iter()
                .filter(|r| match classify(&r.name, layers) {
                    Part::Vision => false,
                    Part::Mtp => sh.mtp,
                    _ => true,
                })
                .map(|r| r.params().unwrap_or_else(|| r.bytes * 8 / sh.release_bits.max(1.0) as u64) as f64)
                .sum();
            assert!((sh.params() / counted - 1.0).abs() < 0.01, "{name}: {:.4e} parameters in the shape, {counted:.4e} in the tensors", sh.params());
        }
    }

    /// The recorded headers of Kimi-K3 and DeepSeek-V4 plan over the GPUs of a machine before any file is here: the layers' bytes and the rest's are the tensors' less the
    /// vision tower and the prediction head, and the plan fits what its GPUs have free.
    #[test]
    fn the_recorded_dynamic_families_are_planned_from_their_headers() {
        let m = crate::plan::Machine::preset("8xh100").unwrap();
        let devs: Vec<crate::plan::Target> = m.devices.iter().enumerate().map(|(i, d)| crate::plan::Target::of(d, i as i32, Some(m.ram_free))).collect();
        let mut planned = 0;
        for (name, text) in HEADER_FIXTURES.iter().filter(|f| f.0 == "Kimi-K3" || f.0.starts_with("DeepSeek")) {
            let v: serde_json::Value = serde_json::from_str(text).unwrap();
            let table = Table::expand(&groups_of(&v));
            let config = v["config"].as_str().unwrap();
            let family = if *name == "Kimi-K3" { "Kimi-K3" } else { "DeepSeek-V4" };
            let nl = crate::dynamic::split::config_num(config, "num_hidden_layers").unwrap();
            let (per, base) = crate::dynamic::split::layer_bytes_table(&table, nl, 100, 100, &|_| true);
            let counted: u64 = table.rows.iter().filter(|r| !matches!(classify(&r.name, nl), Part::Vision | Part::Mtp)).map(|r| r.bytes).sum();
            assert_eq!(per.iter().sum::<u64>() + base, counted, "{name}: every layer's bytes and the rest's");
            let args = crate::args::Args::from(["run".to_string(), name.to_string()]);
            match crate::plan::sizing::size_dynamic(&args, name, &table, None, config, family, &devs, Some(m.ram_free), &|_| true) {
                Ok(p) => {
                    assert!(p.gpu_total() <= p.free && p.spread.is_some(), "{name}");
                    // (each stage's share of the computer's memory: one a stage, adding up to all of it)
                    let sp = p.spread.as_ref().unwrap();
                    let stages = if sp.across { 1 } else { sp.cuts.len() + 1 };
                    assert_eq!(sp.host_per_stage.len(), stages, "{name}");
                    assert_eq!(sp.host_per_stage.iter().sum::<u64>(), sp.host_bytes, "{name}: {:?} of {}", sp.host_per_stage, sp.host_bytes);
                    assert!(sp.host_bytes == 0 || sp.host_per_stage.iter().filter(|&&b| b > 0).count() == stages.min(sp.places.len().max(1)), "{name}: {sp:?}");
                    planned += 1;
                }
                Err(r) => assert!(r.what.contains(name), "{name}: {}", r.what),
            }
        }
        assert!(planned >= 1, "one of them is planned on eight H100s");
    }
}
