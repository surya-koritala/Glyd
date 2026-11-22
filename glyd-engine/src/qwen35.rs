//! Qwen3.5 (2026) as the engine reads it: its config (the model with its text model under `text_config`, `qwen3_5`, or the text model
//! alone, `qwen3_5_text`), the order of its two kinds of layer, the tensors it loads and how they are packed, and what one sequence
//! holds. The design: glyd-engine/docs/design/specs/2026-10-06-engine-qwen35-design.md.

use crate::ffi::{err, Error, Result};
use crate::weights::{eos_ids, num};
use glyd_gpu::json::{self, Value};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// A layer's token mixer: Gated DeltaNet (`linear_attention`) or gated full attention (`full_attention`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Linear,
    Full,
}

/// What the engine reads of a Qwen3.5 config.json.
#[derive(Clone, Debug)]
pub struct Config35 {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub vocab: usize,
    pub eps: f32,
    pub rope_theta: f64,
    /// The dims of a head the rotary turns (the first ones; the rest pass through).
    pub rotary_dim: usize,
    pub max_pos: usize,
    pub tie: bool,
    pub eos: Vec<i64>,
    /// Each layer's kind, in order.
    pub kinds: Vec<Kind>,
    pub lin_k_heads: usize,
    pub lin_v_heads: usize,
    pub lin_k_dim: usize,
    pub lin_v_dim: usize,
    pub conv_kernel: usize,
    /// The layers of the MTP head (`mtp.*`), which this engine does not load yet.
    pub mtp_layers: usize,
    /// What the text model's tensor names begin with: `model.language_model.` (the model with a vision tower) or `model.`.
    pub prefix: &'static str,
    /// The checkpoint has a vision tower (`model.visual.*`), which this engine does not load yet.
    pub vision: bool,
}

impl Config35 {
    pub fn load(dir: &Path) -> Result<Config35> {
        let p = dir.join("config.json");
        let text = std::fs::read_to_string(&p).map_err(|e| Error(format!("{}: {e}", p.display())))?;
        let generation = std::fs::read_to_string(dir.join("generation_config.json")).ok();
        Ok(Config35::parse(&text, &p.display().to_string(), generation.as_deref())?.as_saved(dir))
    }

    /// A glyd-v7 directory (and a lossless glyd-v4 save, glyd/NAME-penguin's) holds the text model under the names transformers' causal LM gives it (`model.layers.*`,
    /// `model.embed_tokens`), whatever its copied config.json says of the checkpoint it came from, and no vision tower.
    pub fn as_saved(mut self, dir: &Path) -> Config35 {
        let m = std::fs::read_to_string(dir.join("glyd.json")).ok().and_then(|t| json::parse(&t).ok());
        let causal =
            |k: &str| m.as_ref().and_then(|m| m.get(k)).and_then(Value::as_object).is_some_and(|o| o.iter().any(|(n, _)| n.starts_with("model.layers.")));
        let saved =
            matches!(m.as_ref().and_then(|m| m.get("format")).and_then(Value::as_str), Some("glyd-v7" | "glyd-v4")) && (causal("packs") || causal("tensors"));
        if saved {
            (self.prefix, self.vision) = ("model.", false);
        }
        self
    }

    /// A config.json's text (and a generation_config.json's, where there is one): `label` names the file in an error.
    pub fn parse(text: &str, label: &str, generation: Option<&str>) -> Result<Config35> {
        let v = json::parse(text).map_err(|e| Error(format!("{label}: {e}")))?;
        let arch = v.get("architectures").and_then(Value::as_array).and_then(|a| a.first()).and_then(Value::as_str).unwrap_or("");
        let (t, prefix, vision) = match arch {
            "Qwen3_5ForConditionalGeneration" => {
                (v.get("text_config").ok_or_else(|| Error(format!("{label}: no text_config")))?, "model.language_model.", true)
            }
            "Qwen3_5ForCausalLM" => (&v, "model.", false),
            "Qwen3_5MoeForConditionalGeneration" | "Qwen3_5MoeForCausalLM" => {
                return err(format!("{label}: {arch}: a mixture of experts, which the engine runs after the dense Qwen3.5 models"))
            }
            _ => return err(format!("{label}: {arch}: not a Qwen3.5 model")),
        };
        let kind = t.get("model_type").and_then(Value::as_str).unwrap_or("");
        if kind != "qwen3_5_text" {
            return err(format!("{label}: the text model is {kind:?}, where a dense Qwen3.5 has qwen3_5_text"));
        }
        let get = |k: &str| t.get(k).ok_or_else(|| Error(format!("{label}: no {k}")));
        let u = |k: &str| -> Result<usize> { get(k)?.as_u64().map(|x| x as usize).ok_or_else(|| Error(format!("{label}: {k}: not an integer"))) };
        let refuse = |what: &str| err(format!("{label}: {what}, which the engine does not run"));
        if t.get("hidden_act").and_then(Value::as_str).is_some_and(|a| a != "silu") {
            return refuse("an activation other than silu");
        }
        if t.get("attention_bias").and_then(Value::as_bool) == Some(true) {
            return refuse("biases in the attention");
        }
        if t.get("attn_output_gate").and_then(Value::as_bool) == Some(false) {
            return refuse("attention without its output gate");
        }
        // the gated norm's gate: vLLM reads swish as silu and also takes sigmoid
        if let Some(g) = t.get("output_gate_type").and_then(Value::as_str).filter(|g| !matches!(*g, "silu" | "swish")) {
            return refuse(&format!("a linear-attention output gate of type {g}"));
        }
        if t.get("mamba_ssm_dtype").and_then(Value::as_str).is_some_and(|d| d != "float32") {
            return refuse("a linear-attention state other than float32");
        }
        if t.get("rope_scaling").is_some_and(|r| *r != Value::Null) {
            return refuse("rope scaling");
        }
        let heads = u("num_attention_heads")?;
        let kv_heads = u("num_key_value_heads")?;
        let head_dim = t.get("head_dim").and_then(Value::as_u64).map(|x| x as usize).unwrap_or(u("hidden_size")? / heads.max(1));
        if heads == 0 || kv_heads == 0 || !heads.is_multiple_of(kv_heads) {
            return err(format!("{label}: {heads} query heads to {kv_heads} KV heads"));
        }
        let rope = get("rope_parameters")?;
        let rope_type = rope.get("rope_type").and_then(Value::as_str).unwrap_or("default");
        if rope_type != "default" {
            return refuse(&format!("rope of type {rope_type}"));
        }
        let theta = num(rope.get("rope_theta").ok_or_else(|| Error(format!("{label}: no rope_theta")))?, "rope_theta")?;
        let factor = match rope.get("partial_rotary_factor") {
            Some(f) => num(f, "partial_rotary_factor")?,
            None => 0.25,
        };
        // (transformers: int(head_dim * partial_rotary_factor))
        let rotary_dim = (head_dim as f64 * factor) as usize;
        if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
            return err(format!("{label}: a rotary of {rotary_dim} of {head_dim} dims"));
        }
        if let Some(sec) = rope.get("mrope_section").and_then(Value::as_u64s) {
            // with text alone the three position streams are one, and the sections' frequencies are the plain rotary's
            // (the sections' sum, checked: a section past 2^63 or sums that wrap round to the right one are not a model's)
            if sec.iter().try_fold(0u64, |a, &x| a.checked_add(x)) != Some(rotary_dim as u64 / 2) {
                return err(format!("{label}: mrope_section {sec:?} does not cover the {rotary_dim} rotary dims"));
            }
        }
        let layers = u("num_hidden_layers")?;
        if layers > crate::plan::MAX_LAYERS {
            return err(format!("{label}: num_hidden_layers {layers}, where no model has more than {}", crate::plan::MAX_LAYERS));
        }
        let kinds = match t.get("layer_types").and_then(Value::as_array) {
            Some(a) => {
                if a.len() != layers {
                    return err(format!("{label}: {} layer_types for {layers} layers", a.len()));
                }
                a.iter()
                    .map(|k| match k.as_str() {
                        Some("linear_attention") => Ok(Kind::Linear),
                        Some("full_attention") => Ok(Kind::Full),
                        other => err(format!("{label}: a layer of type {other:?}")),
                    })
                    .collect::<Result<Vec<Kind>>>()?
            }
            None => {
                let every = t.get("full_attention_interval").and_then(Value::as_u64).unwrap_or(4).max(1) as usize;
                (0..layers).map(|i| if (i + 1).is_multiple_of(every) { Kind::Full } else { Kind::Linear }).collect()
            }
        };
        let (lin_k_heads, lin_v_heads) = (u("linear_num_key_heads")?, u("linear_num_value_heads")?);
        if lin_k_heads == 0 || !lin_v_heads.is_multiple_of(lin_k_heads) {
            return err(format!("{label}: {lin_v_heads} linear-attention value heads to {lin_k_heads} key heads"));
        }
        let conv_kernel = u("linear_conv_kernel_dim")?;
        if conv_kernel == 0 {
            return err(format!("{label}: linear_conv_kernel_dim 0"));
        }
        let mut eos = eos_ids(t, generation);
        if eos.is_empty() {
            eos = eos_ids(&v, None);
        }
        let tie = v.get("tie_word_embeddings").or_else(|| t.get("tie_word_embeddings")).and_then(Value::as_bool).unwrap_or(false);
        Ok(Config35 {
            hidden: u("hidden_size")?,
            layers,
            heads,
            kv_heads,
            head_dim,
            inter: u("intermediate_size")?,
            vocab: u("vocab_size")?,
            eps: num(get("rms_norm_eps")?, "rms_norm_eps")? as f32,
            rope_theta: theta,
            rotary_dim,
            max_pos: u("max_position_embeddings")?,
            tie,
            eos,
            kinds,
            lin_k_heads,
            lin_v_heads,
            lin_k_dim: u("linear_key_head_dim")?,
            lin_v_dim: u("linear_value_head_dim")?,
            conv_kernel,
            mtp_layers: t.get("mtp_num_hidden_layers").and_then(Value::as_u64).unwrap_or(0) as usize,
            prefix,
            vision,
        })
    }

    /// Each layer's kind and its index among the layers of its kind (a linear layer's state, a full layer's KV layer).
    pub fn schedule(&self) -> Vec<(Kind, usize)> {
        let (mut lin, mut full) = (0, 0);
        self.kinds
            .iter()
            .map(|&k| {
                let n = if k == Kind::Linear { &mut lin } else { &mut full };
                *n += 1;
                (k, *n - 1)
            })
            .collect()
    }

    pub fn linear_layers(&self) -> usize {
        self.kinds.iter().filter(|&&k| k == Kind::Linear).count()
    }

    pub fn full_layers(&self) -> usize {
        self.layers - self.linear_layers()
    }

    pub fn key_dim(&self) -> usize {
        self.lin_k_heads * self.lin_k_dim
    }

    pub fn value_dim(&self) -> usize {
        self.lin_v_heads * self.lin_v_dim
    }

    /// The channels of the causal convolution: the linear attention's queries, keys and values.
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }

    /// What one sequence holds besides its KV: every linear layer's fp32 state and bf16 convolution state.
    pub fn state_bytes(&self) -> u64 {
        let per = self.lin_v_heads * self.lin_k_dim * self.lin_v_dim * 4 + self.conv_dim() * (self.conv_kernel - 1) * 2;
        (self.linear_layers() * per) as u64
    }

    /// A token's keys and values in bf16, over the full-attention layers.
    pub fn kv_bytes_per_token(&self) -> u64 {
        (self.full_layers() * 2 * self.kv_heads * self.head_dim * 2) as u64
    }
}

/// One packed matrix: its parts (tensor names and rows) stacked in order, all of `cols` columns. `key` names it (the module of its first part).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pack35 {
    pub key: String,
    pub parts: Vec<(String, usize)>,
    pub cols: usize,
}

impl Pack35 {
    pub fn rows(&self) -> usize {
        self.parts.iter().map(|p| p.1).sum()
    }

    /// The name a tensor's pack is saved under in a glyd-v7 directory (`NAME.glyd_data`, `.glyd_blocks`, `.glyd_block_base`): the tensor without `.weight`. A save fuses a pack's parts
    /// as `Pack35` does for q, k, v and gate, up (the pack is named by the first part), but not the linear attention's qkv and z: those are two packs.
    pub fn saved(tensor: &str) -> &str {
        tensor.trim_end_matches(".weight")
    }

    /// Whether the packer packs it: every part's rows a multiple of 64, the columns of 16. glyd-gpu's `save` merges a group only when each
    /// member passes `save::packable` on its own, so a merge of parts under 64 rows is kept as its parts.
    pub fn packable(&self) -> bool {
        self.cols.is_multiple_of(16) && self.parts.iter().all(|p| p.1 > 0 && p.1.is_multiple_of(64))
    }
}

/// A tensor the engine loads as the checkpoint has it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor35 {
    pub name: String,
    pub dtype: &'static str,
    pub shape: Vec<u64>,
}

/// The tensors of a checkpoint this engine does not load yet: the vision tower's and the MTP head's.
pub fn skipped(name: &str) -> bool {
    name.starts_with("model.visual.") || name.starts_with("mtp.")
}

impl Config35 {
    /// The packed matrices in load order (the packable ones of `matrices`).
    pub fn packs(&self) -> Vec<Pack35> {
        self.matrices().into_iter().filter(Pack35::packable).collect()
    }

    /// Every matrix in load order, packable or not: each layer's (full: q with its gate, k and v as one, then o; linear: the queries, keys, values and z as one, b and a
    /// as one, then out; both: gate and up as one, then down), then the embedding and the LM head where the model has its own.
    fn matrices(&self) -> Vec<Pack35> {
        let (h, hd) = (self.hidden, self.head_dim);
        let mut out = Vec::new();
        let mut pack = |module: String, parts: &[(&str, usize)], cols: usize| {
            out.push(Pack35 {
                parts: parts.iter().map(|&(n, r)| (format!("{module}.{n}.weight"), r)).collect(),
                key: format!("{module}.{}", parts[0].0),
                cols,
            });
        };
        for (l, &kind) in self.kinds.iter().enumerate() {
            let p = format!("{}layers.{l}", self.prefix);
            match kind {
                Kind::Full => {
                    pack(format!("{p}.self_attn"), &[("q_proj", self.heads * hd * 2), ("k_proj", self.kv_heads * hd), ("v_proj", self.kv_heads * hd)], h);
                    pack(format!("{p}.self_attn"), &[("o_proj", h)], self.heads * hd);
                }
                Kind::Linear => {
                    pack(format!("{p}.linear_attn"), &[("in_proj_qkv", self.conv_dim()), ("in_proj_z", self.value_dim())], h);
                    pack(format!("{p}.linear_attn"), &[("in_proj_b", self.lin_v_heads), ("in_proj_a", self.lin_v_heads)], h);
                    pack(format!("{p}.linear_attn"), &[("out_proj", h)], self.value_dim());
                }
            }
            pack(format!("{p}.mlp"), &[("gate_proj", self.inter), ("up_proj", self.inter)], h);
            pack(format!("{p}.mlp"), &[("down_proj", h)], self.inter);
        }
        out.push(Pack35 { key: "embed".into(), parts: vec![(format!("{}embed_tokens.weight", self.prefix), self.vocab)], cols: h });
        if !self.tie {
            out.push(Pack35 { key: "lm_head".into(), parts: vec![("lm_head.weight".into(), self.vocab)], cols: h });
        }
        out
    }

    /// The tensors loaded as the checkpoint has them, in load order: each layer's norms and its linear attention's convolution, `A_log`, `dt_bias` and gated norm, the parts
    /// of a matrix the library cannot pack, and the final norm.
    pub fn kept(&self) -> Vec<Tensor35> {
        let (h, hd) = (self.hidden as u64, self.head_dim as u64);
        let t = |name: String, dtype: &'static str, shape: &[u64]| Tensor35 { name, dtype, shape: shape.to_vec() };
        let unpacked: Vec<Pack35> = self.matrices().into_iter().filter(|p| !p.packable()).collect();
        let mut out = Vec::new();
        for (l, &kind) in self.kinds.iter().enumerate() {
            let p = format!("{}layers.{l}", self.prefix);
            out.push(t(format!("{p}.input_layernorm.weight"), "BF16", &[h]));
            out.push(t(format!("{p}.post_attention_layernorm.weight"), "BF16", &[h]));
            match kind {
                Kind::Full => {
                    out.push(t(format!("{p}.self_attn.q_norm.weight"), "BF16", &[hd]));
                    out.push(t(format!("{p}.self_attn.k_norm.weight"), "BF16", &[hd]));
                }
                Kind::Linear => {
                    let nv = self.lin_v_heads as u64;
                    out.push(t(format!("{p}.linear_attn.conv1d.weight"), "BF16", &[self.conv_dim() as u64, 1, self.conv_kernel as u64]));
                    out.push(t(format!("{p}.linear_attn.A_log"), "F32", &[nv]));
                    out.push(t(format!("{p}.linear_attn.dt_bias"), "BF16", &[nv]));
                    out.push(t(format!("{p}.linear_attn.norm.weight"), "F32", &[self.lin_v_dim as u64]));
                }
            }
            for pk in unpacked.iter().filter(|pk| pk.key.starts_with(&format!("{p}."))) {
                out.extend(pk.parts.iter().map(|(n, r)| t(n.clone(), "BF16", &[*r as u64, pk.cols as u64])));
            }
        }
        out.push(t(format!("{}norm.weight", self.prefix), "BF16", &[h]));
        out
    }

    /// FP32 runtime parameters also accept BF16 checkpoint values; the loader widens them exactly.
    /// A checkpoint's tensors (name, dtype, shape) against what the engine loads: every one present with its dtype and shape, and nothing else but the classes it skips.
    /// Returns the number of tensors skipped. The tensors in `packed` (by `Pack35::saved`) are held as their three buffers, not as bf16 tensors.
    pub fn check(&self, have: &[(String, String, Vec<u64>)], packed: &HashSet<String>) -> Result<usize> {
        let mut want: Vec<Tensor35> = self
            .packs()
            .into_iter()
            .flat_map(|p| p.parts.into_iter().map(move |(n, r)| Tensor35 { name: n, dtype: "BF16", shape: vec![r as u64, p.cols as u64] }))
            .filter(|t| !packed.contains(Pack35::saved(&t.name)))
            .collect();
        want.extend(self.kept());
        let by: HashMap<&str, (&str, &[u64])> = have.iter().map(|(n, d, s)| (n.as_str(), (d.as_str(), s.as_slice()))).collect();
        let mut bad = Vec::new();
        for w in &want {
            match by.get(w.name.as_str()) {
                None => bad.push(format!("no {}", w.name)),
                Some((d, s)) if !(*d == w.dtype || (w.dtype == "F32" && *d == "BF16")) || *s != w.shape.as_slice() => {
                    bad.push(format!("{}: {d} {s:?}, where {} {:?} is wanted", w.name, w.dtype, w.shape))
                }
                _ => {}
            }
        }
        let wanted: std::collections::HashSet<&str> = want.iter().map(|w| w.name.as_str()).collect();
        let buffers: HashSet<String> = packed.iter().flat_map(|k| ["data", "blocks", "block_base"].map(|s| format!("{k}.glyd_{s}"))).collect();
        let mut left_out = 0;
        for (n, _, _) in have {
            if wanted.contains(n.as_str()) || buffers.contains(n) {
                continue;
            }
            if skipped(n) {
                left_out += 1;
            } else {
                bad.push(format!("{n}: not a tensor of this model"));
            }
        }
        if bad.is_empty() {
            Ok(left_out)
        } else {
            let more = if bad.len() > 8 { format!(" (and {} more)", bad.len() - 8) } else { String::new() };
            err(format!("{}{more}", bad[..bad.len().min(8)].join("; ")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn configs() -> Value {
        json::parse(include_str!("../tests/fixtures/qwen35_configs.json")).unwrap()
    }

    fn config(name: &str) -> String {
        json::to_compact(configs().get(name).unwrap())
    }

    fn parse(name: &str) -> Result<Config35> {
        Config35::parse(&config(name), name, None)
    }

    /// The 9B's config with `edit` applied to its text model, as text.
    fn edited(edit: impl Fn(&mut Vec<(String, Value)>)) -> String {
        let mut v = configs().get("Qwen/Qwen3.5-9B").unwrap().clone();
        if let Value::Object(top) = &mut v {
            if let Some((_, Value::Object(t))) = top.iter_mut().find(|(k, _)| k == "text_config") {
                edit(t);
            }
        }
        json::to_compact(&v)
    }

    fn set(t: &mut Vec<(String, Value)>, k: &str, v: &str) {
        let v = json::parse(v).unwrap();
        match t.iter_mut().find(|(n, _)| n == k) {
            Some(e) => e.1 = v,
            None => t.push((k.to_string(), v)),
        }
    }

    #[test]
    fn a_saved_directory_names_the_causal_model() {
        let d = std::env::temp_dir().join(format!("glyd-saved-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("config.json"), config("Qwen/Qwen3.5-9B")).unwrap();
        let c = Config35::load(&d).unwrap();
        assert_eq!((c.prefix, c.vision), ("model.language_model.", true));
        std::fs::write(d.join("glyd.json"), r#"{"format": "glyd-v7", "packs": {"model.layers.0.mlp.down_proj": {}}, "tensors": {"model.norm.weight": {}}}"#)
            .unwrap();
        let c = Config35::load(&d).unwrap();
        assert_eq!((c.prefix, c.vision), ("model.", false));
        // (a lossless glyd-v4 save, glyd/NAME-penguin's, names the causal model the same way; a save whose packs keep the checkpoint's names does not change the prefix)
        std::fs::write(d.join("glyd.json"), r#"{"format": "glyd-v4", "packs": {"model.layers.0.mlp.down_proj": {}}}"#).unwrap();
        assert_eq!(Config35::load(&d).unwrap().prefix, "model.");
        std::fs::write(d.join("glyd.json"), r#"{"format": "glyd-v4", "packs": {"model.language_model.layers.0.mlp.down_proj": {}}}"#).unwrap();
        assert_eq!(Config35::load(&d).unwrap().prefix, "model.language_model.");
        std::fs::remove_dir_all(&d).unwrap();
    }

    fn refused(text: &str, says: &str) {
        let e = Config35::parse(text, "x", None).unwrap_err().msg;
        assert!(e.contains(says), "{e:?} does not say {says:?}");
    }

    #[test]
    fn the_dense_configs() {
        let c = parse("Qwen/Qwen3.5-9B").unwrap();
        assert_eq!((c.hidden, c.layers, c.heads, c.kv_heads, c.head_dim, c.inter, c.vocab), (4096, 32, 16, 4, 256, 12288, 248320));
        assert_eq!((c.rotary_dim, c.rope_theta, c.eps, c.max_pos), (64, 1e7, 1e-6, 262144));
        assert_eq!((c.lin_k_heads, c.lin_v_heads, c.lin_k_dim, c.lin_v_dim, c.conv_kernel, c.mtp_layers), (16, 32, 128, 128, 4, 1));
        assert_eq!((c.prefix, c.vision, c.tie, c.eos.clone()), ("model.language_model.", true, false, vec![248044]));
        let c = parse("Qwen/Qwen3.5-0.8B").unwrap();
        assert_eq!((c.hidden, c.layers, c.heads, c.kv_heads, c.lin_k_heads, c.lin_v_heads, c.tie), (1024, 24, 8, 2, 16, 16, true));
        // the 27Bs: 24 query heads to 4 KV heads (6 to 1); Qwen3.6-27B and Qwen3.8-27B name their output gate swish
        for name in ["Qwen/Qwen3.5-27B", "Qwen/Qwen3.6-27B", "Qwen/Qwen3.8-27B"] {
            let c = parse(name).unwrap();
            assert_eq!((c.hidden, c.layers, c.heads, c.kv_heads, c.lin_v_heads, c.inter, c.tie), (5120, 64, 24, 4, 48, 17408, false), "{name}");
        }
        for name in ["Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B"] {
            let c = parse(name).unwrap();
            assert_eq!((c.head_dim, c.rotary_dim, c.layers % 4, c.full_layers() * 4), (256, 64, 0, c.layers));
        }
    }

    /// The text model alone (Qwen3_5ForCausalLM, qwen3_5_text): its tensors under `model.`, no vision tower.
    #[test]
    fn the_text_model_alone() {
        let v = configs();
        let t = v.get("Qwen/Qwen3.5-9B").unwrap().get("text_config").unwrap().clone();
        let Value::Object(mut t) = t else { panic!() };
        set(&mut t, "architectures", r#"["Qwen3_5ForCausalLM"]"#);
        let c = Config35::parse(&json::to_compact(&Value::Object(t)), "x", None).unwrap();
        assert_eq!((c.prefix, c.vision, c.layers, c.tie, c.eos), ("model.", false, 32, false, vec![248044]));
    }

    #[test]
    fn what_is_refused() {
        refused(&config("Qwen/Qwen3.5-35B-A3B"), "mixture of experts");
        refused(&config("Qwen/Qwen3.8-2.4T-A95B"), "mixture of experts");
        refused(r#"{"architectures": ["Qwen3ForCausalLM"]}"#, "not a Qwen3.5 model");
        refused(&edited(|t| set(t, "model_type", r#""qwen3_5_moe_text""#)), "qwen3_5_text");
        refused(&edited(|t| set(t, "rope_parameters", r#"{"rope_type": "yarn", "rope_theta": 1e7}"#)), "rope of type yarn");
        refused(&edited(|t| set(t, "rope_scaling", r#"{"type": "yarn"}"#)), "rope scaling");
        refused(&edited(|t| set(t, "attention_bias", "true")), "biases");
        refused(&edited(|t| set(t, "hidden_act", r#""gelu""#)), "silu");
        refused(&edited(|t| set(t, "attn_output_gate", "false")), "output gate");
        refused(&edited(|t| set(t, "mamba_ssm_dtype", r#""bfloat16""#)), "float32");
        refused(&edited(|t| set(t, "layer_types", r#"["linear_attention", "full_attention"]"#)), "2 layer_types for 32 layers");
        refused(&edited(|t| set(t, "linear_num_value_heads", "24")), "24 linear-attention value heads to 16");
        refused(
            &edited(|t| {
                set(t, "rope_parameters", r#"{"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 11]}"#)
            }),
            "mrope_section",
        );
        // sections that overflow were a panic of the sum (and, where it wrapped to the right one, a config accepted)
        for sec in ["[9223372036854775808]", "[9223372036854775808, 9223372036854775808]", "[18446744073709551615, 33]"] {
            let rope = format!(r#"{{"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25, "mrope_section": {sec}}}"#);
            refused(&edited(|t| set(t, "rope_parameters", &rope)), "mrope_section");
        }
        refused(&edited(|t| set(t, "num_key_value_heads", "3")), "16 query heads to 3 KV heads");
        refused(&edited(|t| set(t, "num_attention_heads", "0")), "0 query heads to 4 KV heads");
        refused(&edited(|t| set(t, "output_gate_type", r#""sigmoid""#)), "output gate of type sigmoid");
        // what is not refused: the layer pattern from full_attention_interval, and generation_config's end tokens over the config's
        let text = edited(|t| t.retain(|(k, _)| k != "layer_types"));
        let c = Config35::parse(&text, "x", Some(r#"{"eos_token_id": [248046, 248044]}"#)).unwrap();
        assert_eq!((c.full_layers(), c.eos), (8, vec![248046, 248044]));
        let c = Config35::parse(
            &edited(|t| {
                t.retain(|(k, _)| k != "layer_types");
                set(t, "full_attention_interval", "2")
            }),
            "x",
            None,
        )
        .unwrap();
        assert_eq!(c.full_layers(), 16);
    }

    #[test]
    fn the_schedule() {
        let c = parse("Qwen/Qwen3.5-9B").unwrap();
        let s = c.schedule();
        assert_eq!(s.len(), 32);
        for (l, &(k, i)) in s.iter().enumerate() {
            if l % 4 == 3 {
                assert_eq!((k, i), (Kind::Full, l / 4));
            } else {
                assert_eq!((k, i), (Kind::Linear, l / 4 * 3 + l % 4));
            }
        }
        assert_eq!((c.linear_layers(), c.full_layers()), (24, 8));
    }

    /// The spec's table: a sequence's states and a token's KV.
    #[test]
    fn what_a_sequence_holds() {
        let mib = 1u64 << 20;
        let c = parse("Qwen/Qwen3.5-9B").unwrap();
        assert_eq!((c.conv_dim(), c.state_bytes(), c.kv_bytes_per_token()), (8192, 48 * mib + 1_179_648, 32 << 10));
        let c = parse("Qwen/Qwen3.5-0.8B").unwrap();
        assert_eq!((c.conv_dim(), c.state_bytes(), c.kv_bytes_per_token()), (6144, 18 * mib + 663_552, 12 << 10));
        let c = parse("Qwen/Qwen3.8-27B").unwrap();
        assert_eq!((c.conv_dim(), c.state_bytes(), c.kv_bytes_per_token()), (10240, 144 * mib + 2_949_120, 64 << 10));
    }

    /// A checkpoint's index: (name, dtype, shape) of each tensor.
    type Index = Vec<(String, String, Vec<u64>)>;

    fn tensors(model: &str) -> Index {
        let v = json::parse(include_str!("../tests/fixtures/qwen35_tensors.json")).unwrap();
        v.get(model)
            .unwrap()
            .as_object()
            .unwrap()
            .iter()
            .map(|(n, e)| {
                let a = e.as_array().unwrap();
                (n.clone(), a[0].as_str().unwrap().to_string(), a[1].as_u64s().unwrap())
            })
            .collect()
    }

    #[test]
    fn the_weight_map_against_the_checkpoints() {
        let c = parse("Qwen/Qwen3.5-9B").unwrap();
        let have = tensors("Qwen/Qwen3.5-9B");
        assert_eq!(c.check(&have, &HashSet::new()).unwrap(), 333 + 15);
        let c = parse("Qwen/Qwen3.5-0.8B").unwrap();
        let have = tensors("Qwen/Qwen3.5-0.8B");
        assert_eq!(c.check(&have, &HashSet::new()).unwrap(), 153 + 15);
    }

    #[test]
    fn the_packs() {
        let c = parse("Qwen/Qwen3.5-9B").unwrap();
        let packs = c.packs();
        // 24 linear layers of 4 packs (b and a, 32 rows each, are kept), 8 full ones of 4, the embedding and the LM head
        assert_eq!(packs.len(), 24 * 4 + 8 * 4 + 2);
        assert!(packs.iter().all(Pack35::packable));
        let shape = |key: &str| packs.iter().find(|p| p.key == key).map(|p| (p.rows(), p.cols, p.packable())).unwrap();
        let pre = "model.language_model.layers";
        assert_eq!(shape(&format!("{pre}.3.self_attn.q_proj")), (10240, 4096, true));
        assert_eq!(shape(&format!("{pre}.3.self_attn.o_proj")), (4096, 4096, true));
        assert_eq!(shape(&format!("{pre}.0.linear_attn.in_proj_qkv")), (12288, 4096, true));
        assert!(packs.iter().all(|p| !p.key.ends_with(".in_proj_b")));
        let kept = c.kept();
        for (n, r) in [("in_proj_b", 32), ("in_proj_a", 32)] {
            assert!(kept.iter().any(|t| t.name == format!("{pre}.0.linear_attn.{n}.weight") && t.shape == [r, 4096]), "{n}");
        }
        assert_eq!(shape(&format!("{pre}.0.linear_attn.out_proj")), (4096, 4096, true));
        assert_eq!(shape(&format!("{pre}.0.mlp.gate_proj")), (24576, 4096, true));
        assert_eq!(shape(&format!("{pre}.0.mlp.down_proj")), (4096, 12288, true));
        assert_eq!(shape("embed"), (248320, 4096, true));
        assert_eq!(shape("lm_head"), (248320, 4096, true));
        let q = packs.iter().find(|p| p.key == format!("{pre}.3.self_attn.q_proj")).unwrap();
        assert_eq!(
            q.parts.iter().map(|(n, r)| (n.rsplit('.').nth(1).unwrap(), *r)).collect::<Vec<_>>(),
            [("q_proj", 8192), ("k_proj", 1024), ("v_proj", 1024)]
        );
        // a merge is packed only where each part is: 32 rows as one of 64 is not (the packer keeps both parts as they are)
        let ba = Pack35 { key: "x".into(), parts: vec![("b".into(), 32), ("a".into(), 32)], cols: 4096 };
        assert!(ba.rows() == 64 && !ba.packable());
        // the 0.8B: b and a kept; tied, no LM head of its own
        let c = parse("Qwen/Qwen3.5-0.8B").unwrap();
        assert!(c.kept().iter().any(|t| t.name == "model.language_model.layers.0.linear_attn.in_proj_a.weight" && t.shape == [16, 1024]));
        assert!(c.packs().iter().all(|p| p.key != "lm_head"));
    }

    #[test]
    fn a_checkpoint_that_is_not_the_config_is_refused() {
        let c = parse("Qwen/Qwen3.5-0.8B").unwrap();
        let have = tensors("Qwen/Qwen3.5-0.8B");
        let says = |edit: &dyn Fn(&mut Index), what: &str| {
            let mut h = have.clone();
            edit(&mut h);
            let e = c.check(&h, &HashSet::new()).unwrap_err().msg;
            assert!(e.contains(what), "{e:?} does not say {what:?}");
        };
        let at = |h: &Index, n: &str| h.iter().position(|t| t.0 == n).unwrap();
        let conv = "model.language_model.layers.0.linear_attn.conv1d.weight";
        says(
            &|h| {
                let i = at(h, conv);
                h.remove(i);
            },
            &format!("no {conv}"),
        );
        says(
            &|h| {
                let i = at(h, conv);
                h[i].2 = vec![6144, 1, 3];
            },
            "[6144, 1, 3], where BF16 [6144, 1, 4]",
        );
        let a_log = "model.language_model.layers.0.linear_attn.A_log";
        says(
            &|h| {
                let i = at(h, a_log);
                h[i].1 = "F16".into();
            },
            "where F32 [16]",
        );
        says(&|h| h.push(("lm_head.weight".into(), "BF16".into(), vec![248320, 1024])), "lm_head.weight: not a tensor of this model");
        // a config with more layers than the checkpoint
        let more = Config35 { layers: 25, kinds: [c.kinds.clone(), vec![Kind::Linear]].concat(), ..c.clone() };
        assert!(more.check(&have, &HashSet::new()).unwrap_err().msg.contains("no model.language_model.layers.24.linear_attn.in_proj_qkv.weight"));
    }
    /// A tensor held as its three saved buffers needs no bf16 tensor, and the buffers are no stray tensors.
    #[test]
    fn packed_matrices_are_held_as_their_buffers() {
        let c = parse("Qwen/Qwen3.5-9B").unwrap();
        let mut have = tensors("Qwen/Qwen3.5-9B");
        let keys: HashSet<String> = c
            .packs()
            .iter()
            .filter(|p| p.key.starts_with("model.language_model.layers.0.") || p.key == "embed")
            .flat_map(|p| p.parts.iter().map(|x| Pack35::saved(&x.0).to_string()))
            .collect();
        assert!(keys.iter().any(|k| k.ends_with("embed_tokens")) && keys.iter().any(|k| k.ends_with("in_proj_z")));
        have.retain(|t| !keys.contains(Pack35::saved(&t.0)));
        assert!(c.check(&have, &HashSet::new()).is_err());
        for k in &keys {
            have.extend(["data", "blocks", "block_base"].map(|s| (format!("{k}.glyd_{s}"), "U8".to_string(), vec![1u64])));
        }
        assert_eq!(c.check(&have, &keys).unwrap(), 333 + 15);
        assert!(c.check(&have, &HashSet::new()).is_err());
    }
    #[test]
    fn qwen38_pinned_bf16_inventory() {
        let c = Config35::parse(include_str!("../../research/lossy-2026/qwen38-config.json"), "Qwen/Qwen3.8-27B", None).unwrap();
        let v = json::parse(include_str!("../../research/lossy-2026/qwen38-inventory.json")).unwrap();
        let mut have: Vec<(String, String, Vec<u64>)> = v
            .as_array()
            .unwrap()
            .iter()
            .map(|t| {
                (
                    t.get("name").unwrap().as_str().unwrap().to_string(),
                    t.get("dtype").unwrap().as_str().unwrap().to_string(),
                    t.get("shape").unwrap().as_u64s().unwrap(),
                )
            })
            .collect();
        assert_eq!(c.check(&have, &HashSet::new()).unwrap(), 348);
        let i = have.iter().position(|(n, _, _)| n.ends_with("layers.0.linear_attn.A_log")).unwrap();
        have[i].1 = "F32".into();
        assert_eq!(c.check(&have, &HashSet::new()).unwrap(), 348);
        have[i].1 = "F16".into();
        assert!(c.check(&have, &HashSet::new()).is_err());
        have[i].1 = "BF16".into();
        have[i].2 = vec![47];
        assert!(c.check(&have, &HashSet::new()).is_err());
    }
}
