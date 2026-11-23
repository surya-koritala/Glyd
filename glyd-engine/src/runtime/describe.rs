//! Model descriptions from `config.json` (R3a, #136): a `ModelDesc` of layer kinds and the weight map that names each block role's
//! checkpoint tensors, with no per-model code. Families: Qwen3.8 dense and MoE (the Qwen3.5 architecture), gemma-4 text, GLM-5.3 and
//! GLM-5.3-Flash (transformers' keys or the release's, FP8 block-scaled matrices), Kimi-K3, Muse Spark, DeepSeek-V4, Nemotron 3
//! (`nemotron_h`), and the Llama-style standard architectures (Llama, Mistral, Qwen2, Qwen3). Weight maps name the released tensors;
//! a format detail such as Kimi-K3's MXFP4 experts (`.weight_packed` for `.weight`) is the loader's.
//!
//! A config key the builder does not know fails with the key named; keys that cannot change the math (ids, dtypes, training settings)
//! are listed in `INERT`. A decoder layer is two blocks: the mixer's, then the feed-forward's (a Nemotron layer is one block: its
//! pattern names a mixer or a feed-forward per layer).

use super::kinds::*;
use crate::ffi::{err, Error, Result};
use serde_json::{Map, Value};
use std::cell::RefCell;
use std::collections::BTreeSet;

/// Where a role's tensor comes from in the checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub enum Source {
    Tensor(String),
    /// Matrices stacked along their rows (a gate and an up projection as one `gate_up`).
    Rows(Vec<String>),
    /// One entry per expert (each its rows stacked), as [experts, rows, cols].
    Experts(Vec<Vec<String>>),
}

impl Source {
    fn names(&self) -> Vec<String> {
        match self {
            Source::Tensor(n) => vec![n.clone()],
            Source::Rows(v) => v.clone(),
            Source::Experts(v) => v.iter().flatten().cloned().collect(),
        }
    }
}

/// A description and its weight map.
#[derive(Clone, Debug)]
pub struct Described {
    pub desc: ModelDesc,
    pub embed: String,
    pub final_norm: String,
    /// None: the head is the embedding (tied).
    pub head: Option<String>,
    /// Per block, its roles' sources.
    pub blocks: Vec<Vec<(String, Source)>>,
    /// The model's own roles besides the embedding, final norm and head (Kimi-K3's `out_mix_norm`, `out_mix_proj`).
    pub extra: Vec<(String, Source)>,
    /// Draft layers in the checkpoint that the description does not use yet (`mtp_num_hidden_layers`).
    pub mtp_layers: usize,
    /// Matrices stored quantized (FP8 blocks as released, FP4 experts), each with the tensor of its scales. The weight map names them;
    /// decoding a stored format is the tensor source's job.
    pub scales: std::collections::BTreeMap<String, String>,
    /// Each tensor's shape as the config makes it, where the builder says it (Gemma 4's: its matrices, norms and scalars; empty for the builders that do not yet):
    /// what a lowering to the executor sizes its matrices by, for a packed directory too, whose tensors are not stored at their shapes.
    pub shapes: std::collections::BTreeMap<String, Vec<usize>>,
}

impl Described {
    /// Every checkpoint tensor the description reads.
    pub fn names(&self) -> BTreeSet<String> {
        let mut s: BTreeSet<String> = [self.embed.clone(), self.final_norm.clone()].into_iter().chain(self.head.clone()).collect();
        for (_, src) in &self.extra {
            s.extend(src.names());
        }
        s.extend(self.scales.values().cloned());
        for b in &self.blocks {
            for (_, src) in b {
                s.extend(src.names());
            }
        }
        s
    }
}

/// Keys that cannot change what a step computes.
const INERT: &[&str] = &[
    "architectures",
    "model_type",
    "dtype",
    "torch_dtype",
    "transformers_version",
    "initializer_range",
    "use_cache",
    "pad_token_id",
    "bos_token_id",
    "eos_token_id",
    "attention_dropout",
    "output_router_logits",
    "router_aux_loss_coef",
    "pretraining_tp",
    "_name_or_path",
    "output_attentions",
    "output_hidden_states",
    "return_dict",
    "chunk_size_feed_forward",
    "id2label",
    "label2id",
    "problem_type",
    "tokenizer_class",
    "is_encoder_decoder",
    "tie_encoder_decoder",
    "vocab_size_per_layer_input",
    "max_position_embeddings",
    "auto_map",
    "hidden_dropout",
    // (transformers' PretrainedConfig writes its generation and training defaults into some releases' configs: Kimi-K3's text config has them all; none is the model's shape)
    "add_cross_attention",
    "bad_words_ids",
    "begin_suppress_tokens",
    "cross_attention_hidden_size",
    "diversity_penalty",
    "do_sample",
    "early_stopping",
    "encoder_no_repeat_ngram_size",
    "exponential_decay_length_penalty",
    "finetuning_task",
    "is_decoder",
    "length_penalty",
    "max_length",
    "min_length",
    "no_repeat_ngram_size",
    "num_beam_groups",
    "num_beams",
    "num_return_sequences",
    "output_scores",
    "prefix",
    "pruned_heads",
    "remove_invalid_values",
    "repetition_penalty",
    "return_dict_in_generate",
    "suppress_tokens",
    "task_specific_params",
    "temperature",
    "tf_legacy_loss",
    "top_k",
    "top_p",
    "torchscript",
    "typical_p",
    "use_bfloat16",
    // (the loss's ignored label, and the text a prompt marks an image with: the training's and the processor's, not the language model's)
    "ignore_index",
    "image_placeholder",
];

/// A key that says nothing of the model's shape: one of INERT, or a special token's number (`decoder_start_token_id`, `image_token_index`).
fn inert(k: &str) -> bool {
    INERT.contains(&k) || k.ends_with("_token_id") || k.ends_with("_token_index")
}

/// A config's keys, each read at most through these getters; `finish` names a key nobody read.
struct Keys<'a> {
    m: &'a Map<String, Value>,
    used: RefCell<BTreeSet<String>>,
    what: String,
}

impl<'a> Keys<'a> {
    fn new(m: &'a Map<String, Value>, what: &str) -> Keys<'a> {
        Keys { m, used: RefCell::new(BTreeSet::new()), what: what.to_string() }
    }
    fn raw(&self, k: &str) -> Option<&'a Value> {
        self.used.borrow_mut().insert(k.to_string());
        self.m.get(k).filter(|v| !v.is_null())
    }
    fn usize(&self, k: &str) -> Result<usize> {
        self.opt_usize(k)?.ok_or_else(|| Error(format!("{}: no {k}", self.what)))
    }
    fn opt_usize(&self, k: &str) -> Result<Option<usize>> {
        self.raw(k).map(|v| v.as_u64().map(|n| n as usize).ok_or_else(|| Error(format!("{}: {k} is not a count", self.what)))).transpose()
    }
    fn f64(&self, k: &str) -> Result<f64> {
        self.opt_f64(k)?.ok_or_else(|| Error(format!("{}: no {k}", self.what)))
    }
    fn opt_f64(&self, k: &str) -> Result<Option<f64>> {
        self.raw(k).map(|v| v.as_f64().ok_or_else(|| Error(format!("{}: {k} is not a number", self.what)))).transpose()
    }
    fn bool(&self, k: &str, default: bool) -> Result<bool> {
        self.raw(k).map_or(Ok(default), |v| v.as_bool().ok_or_else(|| Error(format!("{}: {k} is not true or false", self.what))))
    }
    fn str(&self, k: &str) -> Result<Option<&'a str>> {
        self.raw(k).map(|v| v.as_str().ok_or_else(|| Error(format!("{}: {k} is not a string", self.what)))).transpose()
    }
    /// The key must be absent, null, or `want` (a setting the builder does not support otherwise).
    fn expect(&self, k: &str, want: &Value) -> Result<()> {
        match self.raw(k) {
            Some(v) if v != want => err(format!("{}: {k} = {v} is not supported (only {want})", self.what)),
            _ => Ok(()),
        }
    }
    fn finish(&self) -> Result<()> {
        let used = self.used.borrow();
        match self.m.keys().find(|k| !used.contains(*k) && !inert(k)) {
            Some(k) => err(format!("{}: config key {k} is not understood by the description builder", self.what)),
            None => Ok(()),
        }
    }
}

fn act(name: &str) -> Result<Act> {
    Ok(match name {
        "silu" | "swish" => Act::Silu,
        "gelu_pytorch_tanh" | "gelu_tanh" => Act::GeluTanh,
        "gelu" => Act::Gelu,
        "relu2" => Act::Relu2,
        "sigmoid" => Act::Sigmoid,
        a => return err(format!("activation {a} is not supported")),
    })
}

/// A rope from transformers' `rope_parameters` (or the older `rope_theta` + `rope_scaling`) for heads of `head_dim`.
fn rope(p: &Map<String, Value>, head_dim: usize, max_pos: Option<usize>, what: &str) -> Result<Rope> {
    let k = Keys::new(p, what);
    let theta = k.f64("rope_theta")?;
    let partial = k.opt_f64("partial_rotary_factor")?.unwrap_or(1.0);
    let dim = (head_dim as f64 * partial) as usize;
    let ty = k.str("rope_type")?.or(k.str("type")?).unwrap_or("default");
    let orig = || -> Result<u32> { Ok(k.usize("original_max_position_embeddings")? as u32) };
    let mut r = Rope::plain(theta, dim);
    let mrope = match k.raw("mrope_section") {
        Some(Value::Array(a)) if a.len() == 3 => Some([0, 1, 2].map(|i| a[i].as_u64().unwrap_or(0) as usize)),
        Some(v) => return err(format!("{what}: mrope_section {v}")),
        None => None,
    };
    k.bool("mrope_interleaved", false)?;
    r.mrope = mrope;
    r.scaling = match ty {
        "default" => RopeScaling::None,
        "proportional" => {
            r = Rope { mrope, ..Rope::proportional(theta, head_dim, partial) };
            match k.opt_f64("factor")? {
                Some(f) if f != 1.0 => RopeScaling::Linear { factor: f },
                _ => RopeScaling::None,
            }
        }
        "linear" => RopeScaling::Linear { factor: k.f64("factor")? },
        "dynamic" => RopeScaling::DynamicNtk {
            factor: k.f64("factor")?,
            max_positions: max_pos.ok_or_else(|| Error(format!("{what}: dynamic rope without max_position_embeddings")))? as u32,
        },
        "yarn" => RopeScaling::Yarn {
            factor: k.f64("factor")?,
            original_max: orig()?,
            beta_fast: k.opt_f64("beta_fast")?.unwrap_or(32.0),
            beta_slow: k.opt_f64("beta_slow")?.unwrap_or(1.0),
            mscale: k.opt_f64("mscale")?,
            mscale_all_dim: k.opt_f64("mscale_all_dim")?,
            attention_factor: k.opt_f64("attention_factor")?,
            truncate: k.bool("truncate", true)?,
        },
        "llama3" => RopeScaling::Llama3 {
            factor: k.f64("factor")?,
            low_freq_factor: k.f64("low_freq_factor")?,
            high_freq_factor: k.f64("high_freq_factor")?,
            original_max: orig()?,
        },
        "longrope" => {
            let fs = |n: &str| -> Result<Vec<f64>> {
                k.raw(n)
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(1.0)).collect())
                    .ok_or_else(|| Error(format!("{what}: no {n}")))
            };
            let max = max_pos.ok_or_else(|| Error(format!("{what}: longrope without max_position_embeddings")))? as u32;
            k.opt_f64("factor")?;
            RopeScaling::LongRope {
                short: fs("short_factor")?,
                long: fs("long_factor")?,
                original_max: orig()?,
                max_positions: max,
                attention_factor: k.opt_f64("attention_factor")?,
            }
        }
        t => return err(format!("{what}: rope_type {t} is not supported")),
    };
    k.finish()?;
    Ok(r)
}

/// The rope of a config without `rope_parameters`: `rope_theta` and the older `rope_scaling`.
fn legacy_rope(k: &Keys, head_dim: usize, max_pos: Option<usize>) -> Result<Rope> {
    let mut m = match k.raw("rope_scaling") {
        Some(Value::Object(o)) => o.clone(),
        _ => Map::new(),
    };
    m.insert("rope_theta".into(), Value::from(k.opt_f64("rope_theta")?.unwrap_or(10_000.0)));
    if let Some(p) = k.opt_f64("partial_rotary_factor")? {
        m.insert("partial_rotary_factor".into(), Value::from(p));
    }
    rope(&m, head_dim, max_pos, &format!("{} rope", k.what))
}

fn gated(act: Act) -> Mlp {
    Mlp { act, gated: true, bias: false, clamp: None }
}

fn t(n: String) -> Source {
    Source::Tensor(n)
}

/// The roles of a Llama-style attention and its gated MLP at `p` (`model.layers.N`).
fn attn_roles(p: &str, bias: bool, qk_norm: bool) -> Vec<(String, Source)> {
    let mut r = vec![("norm_in".to_string(), t(format!("{p}.input_layernorm.weight")))];
    for x in ["q", "k", "v", "o"] {
        r.push((x.to_string(), t(format!("{p}.self_attn.{x}_proj.weight"))));
        if bias && x != "o" {
            r.push((format!("{x}_bias"), t(format!("{p}.self_attn.{x}_proj.bias"))));
        }
    }
    if qk_norm {
        r.push(("q_norm".into(), t(format!("{p}.self_attn.q_norm.weight"))));
        r.push(("k_norm".into(), t(format!("{p}.self_attn.k_norm.weight"))));
    }
    r
}

fn mlp_roles(p: &str, norm: &str) -> Vec<(String, Source)> {
    let mut r = vec![("norm_in".to_string(), t(format!("{p}.{norm}.weight")))];
    for x in ["gate", "up", "down"] {
        r.push((x.to_string(), t(format!("{p}.mlp.{x}_proj.weight"))));
    }
    r
}

/// The text config (`text_config` of a multimodal one) and the prefix of its layers' tensor names.
fn text(cfg: &Value) -> Result<(&Map<String, Value>, &'static str)> {
    let top = cfg.as_object().ok_or_else(|| Error("config.json is not an object".into()))?;
    match top.get("text_config") {
        Some(Value::Object(t)) => Ok((t, "model.language_model.")),
        Some(_) => err("text_config is not an object"),
        None => Ok((top, "model.")),
    }
}

/// The description of the model whose `config.json` is `cfg`.
pub fn describe(cfg: &Value) -> Result<Described> {
    let (tc, prefix) = text(cfg)?;
    match cfg.get("model_type").and_then(|v| v.as_str()) {
        Some("kimi_k3") => return kimi(cfg, tc),
        Some("muse_spark") => return muse(tc, prefix),
        Some("deepseek_v4") => return deepseek_v4(tc),
        Some("nemotron_h") => return nemotron_h(tc),
        _ => {}
    }
    let mt = tc.get("model_type").and_then(|v| v.as_str()).unwrap_or("");
    // (a multimodal config's tie flag can sit at the top)
    let tied_top = cfg.get("tie_word_embeddings").and_then(|v| v.as_bool());
    match mt {
        "llama" | "mistral" | "qwen2" | "qwen3" => llama(tc, prefix, mt, tied_top),
        "qwen3_5_text" | "qwen3_5_moe_text" | "qwen3_5" | "qwen3_5_moe" => qwen35(tc, prefix, mt, tied_top),
        "gemma4_text" | "gemma4_unified_text" => gemma4(tc, prefix, mt, tied_top),
        "glm_moe_dsa" | "glm5_next_text" | "glm5_next" => glm(tc, prefix, mt, tied_top, cfg.get("quantization_config").or(tc.get("quantization_config"))),
        "" => err("config.json names no model_type"),
        m => err(format!("model_type {m} has no description builder yet")),
    }
}

fn common(k: &Keys, tied_top: Option<bool>) -> Result<(usize, usize, usize, f32, bool)> {
    let (d, layers, vocab) = (k.usize("hidden_size")?, k.usize("num_hidden_layers")?, k.usize("vocab_size")?);
    let eps = k.opt_f64("rms_norm_eps")?.unwrap_or(1e-6) as f32;
    let tied = k.bool("tie_word_embeddings", tied_top.unwrap_or(false))?;
    Ok((d, layers, vocab, eps, tied))
}

fn finish(k: &Keys, desc: ModelDesc, prefix: &str, blocks: Vec<Vec<(String, Source)>>, mtp_layers: usize) -> Result<Described> {
    k.finish()?;
    let tied = desc.head.tied;
    Ok(Described {
        desc,
        embed: format!("{prefix}embed_tokens.weight"),
        final_norm: format!("{prefix}norm.weight"),
        head: (!tied).then(|| "lm_head.weight".to_string()),
        blocks,
        extra: Vec::new(),
        mtp_layers,
        scales: Default::default(),
        shapes: Default::default(),
    })
}

fn llama(tc: &Map<String, Value>, prefix: &str, mt: &str, tied_top: Option<bool>) -> Result<Described> {
    let k = Keys::new(tc, mt);
    let (d, layers, vocab, eps, tied) = common(&k, tied_top)?;
    let (heads, kv) = (k.usize("num_attention_heads")?, k.opt_usize("num_key_value_heads")?.unwrap_or(k.usize("num_attention_heads")?));
    let hd = k.opt_usize("head_dim")?.unwrap_or(d / heads);
    let max_pos = k.opt_usize("max_position_embeddings")?;
    let rp = match k.raw("rope_parameters") {
        Some(Value::Object(o)) => rope(o, hd, max_pos, &format!("{mt} rope_parameters"))?,
        _ => legacy_rope(&k, hd, max_pos)?,
    };
    let bias = k.bool("attention_bias", mt == "qwen2")?;
    k.expect("mlp_bias", &Value::Bool(false))?;
    let act = act(k.str("hidden_act")?.unwrap_or("silu"))?;
    k.usize("intermediate_size")?;
    // sliding windows: Mistral's `sliding_window` for every layer; Qwen2's past `max_window_layers` when `use_sliding_window`
    let window = k.opt_usize("sliding_window")?;
    let qwen_sliding = k.bool("use_sliding_window", mt == "mistral")?;
    let max_window_layers = k.opt_usize("max_window_layers")?.unwrap_or(0);
    let layer_types: Option<Vec<String>> =
        k.raw("layer_types").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect());
    let qk_norm = mt == "qwen3";
    let mut blocks = Vec::new();
    let mut roles = Vec::new();
    for l in 0..layers {
        let sliding = match &layer_types {
            Some(lt) => lt.get(l).map(|s| s == "sliding_attention").unwrap_or(false),
            None => qwen_sliding && window.is_some() && (mt == "mistral" || l >= max_window_layers),
        };
        let a = Attn {
            heads,
            kv_heads: kv,
            head_dim: hd,
            v_head_dim: hd,
            window: if sliding { window } else { None },
            sinks: false,
            rope: Some(rp.clone()),
            q_norm: qk_norm.then(|| Norm::rms(eps)),
            k_norm: qk_norm.then(|| Norm::rms(eps)),
            v_norm: None,
            gate: None,
            softcap: None,
            bias,
            scale: (hd as f32).powf(-0.5),
            k_eq_v: false,
            indexer: None,
            rounding: AttnRounding::F32,
        };
        let p = format!("{prefix}layers.{l}");
        blocks.push(Block { norm_in: Norm::rms(eps), body: Body::Mixer(Mixer::Attention(a)), norm_out: None, residual: Residual::Add });
        roles.push(attn_roles(&p, bias, qk_norm));
        blocks.push(Block { norm_in: Norm::rms(eps), body: Body::Ffn(Ffn::Mlp(gated(act))), norm_out: None, residual: Residual::Add });
        roles.push(mlp_roles(&p, "post_attention_layernorm"));
    }
    let desc = ModelDesc {
        hidden: d,
        vocab,
        embed: Embed { scale: 1.0, norm: None, streams: 1 },
        blocks,
        final_norm: Norm::rms(eps),
        head: Head { tied, softcap: None, scale: 1.0, collapse: Collapse::None },
        mtp: Vec::new(),
        vision: None,
        rope: Some(rp),
    };
    finish(&k, desc, prefix, roles, 0)
}

fn qwen35(tc: &Map<String, Value>, prefix: &str, mt: &str, tied_top: Option<bool>) -> Result<Described> {
    let k = Keys::new(tc, mt);
    let (d, layers, vocab, eps, tied) = common(&k, tied_top)?;
    let (heads, kv, hd) = (k.usize("num_attention_heads")?, k.usize("num_key_value_heads")?, k.usize("head_dim")?);
    let max_pos = k.opt_usize("max_position_embeddings")?;
    k.opt_f64("partial_rotary_factor")?;
    let rp = match k.raw("rope_parameters") {
        Some(Value::Object(o)) => rope(o, hd, max_pos, &format!("{mt} rope_parameters"))?,
        _ => legacy_rope(&k, hd, max_pos)?,
    };
    k.expect("attention_bias", &Value::Bool(false))?;
    let gate = k.bool("attn_output_gate", true)?;
    let lin_gate = act(k.str("output_gate_type")?.unwrap_or("swish"))?;
    let act = act(k.str("hidden_act")?.unwrap_or("silu"))?;
    k.expect("mamba_ssm_dtype", &Value::from("float32"))?;
    k.opt_usize("full_attention_interval")?;
    let types: Vec<String> = k
        .raw("layer_types")
        .and_then(|v| v.as_array())
        .ok_or_else(|| Error(format!("{mt}: no layer_types")))?
        .iter()
        .map(|x| x.as_str().unwrap_or("").to_string())
        .collect();
    if types.len() != layers {
        return err(format!("{mt}: {} layer_types for {layers} layers", types.len()));
    }
    let lin = Linear {
        kind: LinearKind::GatedDeltaNet,
        k_heads: k.usize("linear_num_key_heads")?,
        v_heads: k.usize("linear_num_value_heads")?,
        k_dim: k.usize("linear_key_head_dim")?,
        v_dim: k.usize("linear_value_head_dim")?,
        conv: k.usize("linear_conv_kernel_dim")?,
        decay: Decay::Softplus,
        gate_act: lin_gate,
        norm_eps: eps,
        state_f32: true,
    };
    let mut dense_inter = None;
    let moe = if k.raw("num_experts").is_some() {
        let ne = k.usize("num_experts")?;
        k.usize("moe_intermediate_size")?;
        let shared = k.opt_usize("shared_expert_intermediate_size")?.is_some_and(|s| s > 0);
        Some(Moe {
            router: Router {
                score: RouterScore::Softmax,
                logit_scale: 1.0,
                bias: false,
                groups: None,
                top_k: k.usize("num_experts_per_tok")?,
                renorm: k.bool("norm_topk_prob", true)?,
                scale: 1.0,
                hash: false,
            },
            experts: ne,
            expert: gated(act),
            shared: shared.then(|| gated(act)),
            shared_gate: shared,
            latent: None,
        })
    } else {
        dense_inter = Some(k.usize("intermediate_size")?);
        None
    };
    k.expect("mlp_only_layers", &Value::Array(Vec::new()))?;
    let mtp_layers = k.opt_usize("mtp_num_hidden_layers")?.unwrap_or(0);
    k.bool("mtp_use_dedicated_embeddings", false)?;
    let pn = Norm::rms_plus_one(eps);
    let (mut blocks, mut roles) = (Vec::new(), Vec::new());
    for (l, ty) in types.iter().enumerate() {
        let p = format!("{prefix}layers.{l}");
        let (mixer, mut r) = match ty.as_str() {
            "linear_attention" => {
                let a = format!("{p}.linear_attn");
                let r: Vec<(String, Source)> = [
                    ("qkv", "in_proj_qkv.weight"),
                    ("gate", "in_proj_z.weight"),
                    ("beta", "in_proj_b.weight"),
                    ("decay", "in_proj_a.weight"),
                    ("conv", "conv1d.weight"),
                    ("a_log", "A_log"),
                    ("dt_bias", "dt_bias"),
                    ("norm", "norm.weight"),
                    ("o", "out_proj.weight"),
                ]
                .iter()
                .map(|(role, n)| (role.to_string(), t(format!("{a}.{n}"))))
                .collect();
                (Mixer::Linear(lin.clone()), r)
            }
            "full_attention" => {
                let a = Attn {
                    heads,
                    kv_heads: kv,
                    head_dim: hd,
                    v_head_dim: hd,
                    window: None,
                    sinks: false,
                    rope: Some(rp.clone()),
                    q_norm: Some(pn),
                    k_norm: Some(pn),
                    v_norm: None,
                    gate: gate.then_some(OutGate { act: Act::Sigmoid, from: GateFrom::InQuery, norm: None }),
                    softcap: None,
                    bias: false,
                    scale: (hd as f32).powf(-0.5),
                    k_eq_v: false,
                    indexer: None,
                    rounding: AttnRounding::Flash,
                };
                let mut r = attn_roles(&p, false, true);
                r.remove(0);
                (Mixer::Attention(a), r)
            }
            other => return err(format!("{mt}: layer type {other} at layer {l}")),
        };
        r.insert(0, ("norm_in".into(), t(format!("{p}.input_layernorm.weight"))));
        blocks.push(Block { norm_in: pn, body: Body::Mixer(mixer), norm_out: None, residual: Residual::Add });
        roles.push(r);
        match &moe {
            None => {
                blocks.push(Block { norm_in: pn, body: Body::Ffn(Ffn::Mlp(gated(act))), norm_out: None, residual: Residual::Add });
                roles.push(mlp_roles(&p, "post_attention_layernorm"));
            }
            Some(m) => {
                let f = format!("{p}.mlp");
                let mut r = vec![
                    ("norm_in".to_string(), t(format!("{p}.post_attention_layernorm.weight"))),
                    ("router".into(), t(format!("{f}.gate.weight"))),
                    (
                        "experts_gate_up".into(),
                        Source::Experts(
                            (0..m.experts).map(|e| vec![format!("{f}.experts.{e}.gate_proj.weight"), format!("{f}.experts.{e}.up_proj.weight")]).collect(),
                        ),
                    ),
                    ("experts_down".into(), Source::Experts((0..m.experts).map(|e| vec![format!("{f}.experts.{e}.down_proj.weight")]).collect())),
                ];
                if m.shared.is_some() {
                    for x in ["gate", "up", "down"] {
                        r.push((format!("shared_{x}"), t(format!("{f}.shared_expert.{x}_proj.weight"))));
                    }
                    r.push(("shared_router".into(), t(format!("{f}.shared_expert_gate.weight"))));
                }
                blocks.push(Block { norm_in: pn, body: Body::Ffn(Ffn::Moe(m.clone())), norm_out: None, residual: Residual::Add });
                roles.push(r);
            }
        }
    }
    let desc = ModelDesc {
        hidden: d,
        vocab,
        embed: Embed { scale: 1.0, norm: None, streams: 1 },
        blocks,
        final_norm: pn,
        head: Head { tied, softcap: None, scale: 1.0, collapse: Collapse::None },
        mtp: Vec::new(),
        vision: None,
        rope: Some(rp),
    };
    let mut out = finish(&k, desc, prefix, roles, mtp_layers)?;
    // (the shapes of a dense model's tensors, as its heads and widths make them; a mixture of experts' are not said yet)
    if let Some(inter) = dense_inter {
        let (kh, vh, kd, vd, conv) = (lin.k_heads, lin.v_heads, lin.k_dim, lin.v_dim, lin.conv);
        let mut shapes = std::collections::BTreeMap::new();
        let mut put = |n: String, s: Vec<usize>| {
            shapes.insert(n, s);
        };
        for (l, ty) in types.iter().enumerate() {
            let p = format!("{prefix}layers.{l}");
            if ty == "linear_attention" {
                let a = format!("{p}.linear_attn");
                let qkv = 2 * kh * kd + vh * vd;
                put(format!("{a}.in_proj_qkv.weight"), vec![qkv, d]);
                put(format!("{a}.in_proj_z.weight"), vec![vh * vd, d]);
                put(format!("{a}.in_proj_b.weight"), vec![vh, d]);
                put(format!("{a}.in_proj_a.weight"), vec![vh, d]);
                put(format!("{a}.conv1d.weight"), vec![qkv, 1, conv]);
                put(format!("{a}.A_log"), vec![vh]);
                put(format!("{a}.dt_bias"), vec![vh]);
                put(format!("{a}.norm.weight"), vec![vd]);
                put(format!("{a}.out_proj.weight"), vec![d, vh * vd]);
            } else {
                let a = format!("{p}.self_attn");
                put(format!("{a}.q_proj.weight"), vec![heads * hd * if gate { 2 } else { 1 }, d]);
                put(format!("{a}.k_proj.weight"), vec![kv * hd, d]);
                put(format!("{a}.v_proj.weight"), vec![kv * hd, d]);
                put(format!("{a}.o_proj.weight"), vec![d, heads * hd]);
                put(format!("{a}.q_norm.weight"), vec![hd]);
                put(format!("{a}.k_norm.weight"), vec![hd]);
            }
            put(format!("{p}.input_layernorm.weight"), vec![d]);
            put(format!("{p}.post_attention_layernorm.weight"), vec![d]);
            put(format!("{p}.mlp.gate_proj.weight"), vec![inter, d]);
            put(format!("{p}.mlp.up_proj.weight"), vec![inter, d]);
            put(format!("{p}.mlp.down_proj.weight"), vec![d, inter]);
        }
        put(out.embed.clone(), vec![vocab, d]);
        put(out.final_norm.clone(), vec![d]);
        if let Some(h) = out.head.clone() {
            put(h, vec![vocab, d]);
        }
        out.shapes = shapes;
    }
    Ok(out)
}

fn gemma4(tc: &Map<String, Value>, prefix: &str, mt: &str, tied_top: Option<bool>) -> Result<Described> {
    let k = Keys::new(tc, mt);
    // (transformers' Gemma 4 ties the head to the embedding unless the config says otherwise, where Llama and Qwen default to a head of their own)
    let (d, layers, vocab, eps, tied) = common(&k, tied_top.or(Some(true)))?;
    let (heads, kv, hd) = (k.usize("num_attention_heads")?, k.usize("num_key_value_heads")?, k.usize("head_dim")?);
    // the global layers' head dim and KV heads: the release's keys, or (transformers 5.18 writes these instead) `per_layer_config`
    let per = match k.raw("per_layer_config") {
        Some(Value::Object(o)) => Some(o),
        Some(v) => return err(format!("{mt}: per_layer_config {v}")),
        None => None,
    };
    let per_first = |key: &str| per.and_then(|o| o.values().find_map(|v| v.get(key).and_then(|x| x.as_u64()))).map(|x| x as usize);
    let ghd = k.opt_usize("global_head_dim")?.or(per_first("head_dim")).unwrap_or(hd);
    let k_eq_v = k.bool("attention_k_eq_v", false)?;
    let gkv = if k_eq_v { k.opt_usize("num_global_key_value_heads")?.or(per_first("num_key_value_heads")).unwrap_or(kv) } else { kv };
    k.opt_usize("num_global_key_value_heads")?;
    let window = k.usize("sliding_window")?;
    let max_pos = k.opt_usize("max_position_embeddings")?;
    k.expect("attention_bias", &Value::Bool(false))?;
    k.expect("enable_moe_block", &Value::Bool(false))?;
    k.expect("hidden_size_per_layer_input", &Value::from(0))?;
    k.expect("num_kv_shared_layers", &Value::from(0))?;
    k.expect("use_double_wide_mlp", &Value::Bool(false))?;
    for x in ["num_experts", "top_k_experts", "moe_intermediate_size"] {
        k.expect(x, &Value::Null)?;
    }
    if k.str("use_bidirectional_attention")? == Some("all") {
        return err(format!("{mt}: bidirectional attention for text is not supported"));
    }
    let inter = k.usize("intermediate_size")?;
    let act = act(k.str("hidden_activation")?.or(k.str("hidden_act")?).unwrap_or("gelu_pytorch_tanh"))?;
    let softcap = k.opt_f64("final_logit_softcapping")?.map(|v| v as f32);
    let rps = match k.raw("rope_parameters") {
        Some(Value::Object(o)) => o,
        _ => return err(format!("{mt}: no rope_parameters per layer type")),
    };
    let rope_of = |ty: &str, head: usize| -> Result<Rope> {
        match rps.get(ty) {
            Some(Value::Object(o)) => rope(o, head, max_pos, &format!("{mt} rope_parameters.{ty}")),
            _ => err(format!("{mt}: no rope_parameters for {ty}")),
        }
    };
    let types: Vec<String> = k
        .raw("layer_types")
        .and_then(|v| v.as_array())
        .ok_or_else(|| Error(format!("{mt}: no layer_types")))?
        .iter()
        .map(|x| x.as_str().unwrap_or("").to_string())
        .collect();
    if types.len() != layers {
        return err(format!("{mt}: {} layer_types for {layers} layers", types.len()));
    }
    let (sl, gl) = (rope_of("sliding_attention", hd)?, rope_of("full_attention", ghd)?);
    // transformers writes the global layers' head dim and KV heads again per layer; they must agree
    if let Some(per) = per {
        for (l, o) in per {
            let i: usize = l.parse().map_err(|_| Error(format!("{mt}: per_layer_config key {l}")))?;
            let (h, n) = (o.get("head_dim").and_then(|v| v.as_u64()), o.get("num_key_value_heads").and_then(|v| v.as_u64()));
            if types.get(i).map(String::as_str) != Some("full_attention")
                || h.is_some_and(|h| h as usize != ghd)
                || n.is_some_and(|n| n as usize != gkv)
                || o.as_object().is_none_or(|m| m.len() > 2)
            {
                return err(format!("{mt}: per_layer_config for layer {l} ({o:?}) is not the global layers' shape"));
            }
        }
    }
    let n = Norm::rms(eps);
    let (mut blocks, mut roles) = (Vec::new(), Vec::new());
    for (l, ty) in types.iter().enumerate() {
        let sliding = match ty.as_str() {
            "sliding_attention" => true,
            "full_attention" => false,
            o => return err(format!("{mt}: layer type {o} at layer {l}")),
        };
        let (h, kvh, kev) = if sliding { (hd, kv, false) } else { (ghd, gkv, k_eq_v) };
        let a = Attn {
            heads,
            kv_heads: kvh,
            head_dim: h,
            v_head_dim: h,
            window: sliding.then_some(window),
            sinks: false,
            rope: Some(if sliding { sl.clone() } else { gl.clone() }),
            q_norm: Some(n),
            k_norm: Some(n),
            v_norm: Some(Norm::rms_bare(eps)),
            gate: None,
            softcap: None,
            bias: false,
            scale: 1.0,
            k_eq_v: kev,
            indexer: None,
            rounding: AttnRounding::Eager,
        };
        let p = format!("{prefix}layers.{l}");
        let mut r = attn_roles(&p, false, true);
        if kev {
            r.retain(|(role, _)| role != "v");
        }
        r.push(("norm_out".into(), t(format!("{p}.post_attention_layernorm.weight"))));
        blocks.push(Block { norm_in: n, body: Body::Mixer(Mixer::Attention(a)), norm_out: Some(n), residual: Residual::Add });
        roles.push(r);
        let mut r = mlp_roles(&p, "pre_feedforward_layernorm");
        r.push(("norm_out".into(), t(format!("{p}.post_feedforward_layernorm.weight"))));
        r.push(("res_scale".into(), t(format!("{p}.layer_scalar"))));
        blocks.push(Block { norm_in: n, body: Body::Ffn(Ffn::Mlp(gated(act))), norm_out: Some(n), residual: Residual::AddThenScale });
        roles.push(r);
    }
    let desc = ModelDesc {
        hidden: d,
        vocab,
        embed: Embed { scale: (d as f32).sqrt(), norm: None, streams: 1 },
        blocks,
        final_norm: n,
        head: Head { tied, softcap, scale: 1.0, collapse: Collapse::None },
        mtp: Vec::new(),
        vision: None,
        rope: None,
    };
    let mut out = finish(&k, desc, prefix, roles, 0)?;
    // (the shapes: each matrix [rows, columns] as the layer's heads and widths make it, each norm and scalar its width)
    let mut shapes = std::collections::BTreeMap::new();
    for (l, ty) in types.iter().enumerate() {
        let (h, kvh) = if ty == "sliding_attention" { (hd, kv) } else { (ghd, gkv) };
        let p = format!("{prefix}layers.{l}");
        let a = format!("{p}.self_attn");
        for (x, s) in [
            ("q_proj", vec![heads * h, d]),
            ("k_proj", vec![kvh * h, d]),
            ("v_proj", vec![kvh * h, d]),
            ("o_proj", vec![d, heads * h]),
            ("q_norm", vec![h]),
            ("k_norm", vec![h]),
        ] {
            shapes.insert(format!("{a}.{x}.weight"), s);
        }
        for (x, s) in [("gate_proj", vec![inter, d]), ("up_proj", vec![inter, d]), ("down_proj", vec![d, inter])] {
            shapes.insert(format!("{p}.mlp.{x}.weight"), s);
        }
        for x in ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"] {
            shapes.insert(format!("{p}.{x}.weight"), vec![d]);
        }
        shapes.insert(format!("{p}.layer_scalar"), vec![1]);
    }
    shapes.insert(out.embed.clone(), vec![vocab, d]);
    shapes.insert(out.final_norm.clone(), vec![d]);
    let named = out.names();
    shapes.retain(|n, _| named.contains(n));
    out.shapes = shapes;
    Ok(out)
}

/// FP8 block quantization as released (`quantization_config`): which matrices hold FP8 and their block.
struct Fp8 {
    block: usize,
    skip: Vec<String>,
}

impl Fp8 {
    fn parse(q: Option<&Value>, what: &str) -> Result<Option<Fp8>> {
        let Some(q) = q.and_then(|v| v.as_object()) else { return Ok(None) };
        let method = q.get("quant_method").and_then(|v| v.as_str()).unwrap_or("fp8");
        let fmt = q.get("fmt").and_then(|v| v.as_str()).unwrap_or("e4m3");
        if method != "fp8" || fmt != "e4m3" {
            return err(format!("{what}: quantization {method} {fmt} is not supported"));
        }
        let block = match q.get("weight_block_size") {
            Some(Value::Array(a)) if a.len() == 2 && a[0] == a[1] => a[0].as_u64().unwrap_or(128) as usize,
            Some(v) => return err(format!("{what}: weight_block_size {v}")),
            // (no block size: one scale a tensor or a channel, not the block scales the loaders read; refused, as the GLM loader refuses it)
            None => return err(format!("{what}: FP8 with no weight_block_size (scales a tensor or a channel) is not supported: block scales are")),
        };
        let skip = q
            .get("modules_to_not_convert")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
            .unwrap_or_default();
        Ok(Some(Fp8 { block, skip }))
    }

    /// Whether the matrix `w` (`x.weight`, the language model's prefix dropped as the config lists modules) is FP8.
    fn holds(&self, w: &str) -> bool {
        let m = w.strip_suffix(".weight").unwrap_or(w).replace("language_model.", "");
        !self.skip.iter().any(|s| m == *s || m.ends_with(&format!(".{s}")) || m.starts_with(&format!("{s}.")))
    }
}

/// The layer kinds of a GLM config, from transformers' keys (`layer_types`) or the release's (`linear_attn_config`).
fn glm_types(k: &Keys, layers: usize, mt: &str) -> Result<Vec<bool>> {
    if let Some(Value::Array(a)) = k.raw("layer_types") {
        let t: Vec<bool> = a
            .iter()
            .map(|x| match x.as_str() {
                Some("linear_attention") => Ok(true),
                Some("indexed_attention" | "deepseek_sparse_attention" | "full_attention") => Ok(false),
                o => err(format!("{mt}: layer type {o:?}")),
            })
            .collect::<Result<_>>()?;
        if t.len() < layers {
            return err(format!("{mt}: {} layer_types for {layers} layers", t.len()));
        }
        return Ok(t[..layers].to_vec());
    }
    Ok(vec![false; layers])
}

fn glm(tc: &Map<String, Value>, prefix: &str, mt: &str, tied_top: Option<bool>, quant: Option<&Value>) -> Result<Described> {
    let k = Keys::new(tc, mt);
    let flash = mt != "glm_moe_dsa";
    let (d, layers, vocab, eps, tied) = common(&k, tied_top)?;
    let max_pos = k.opt_usize("max_position_embeddings")?;
    k.expect("attention_bias", &Value::Bool(false))?;
    k.expect("mlp_bias", &Value::Bool(false))?;
    k.raw("quantization_config");
    let fp8 = Fp8::parse(quant, mt)?;
    let act = act(k.str("hidden_act")?.unwrap_or("silu"))?;
    if let Some(sf) = k.str("scoring_func")? {
        if sf != "sigmoid" {
            return err(format!("{mt}: scoring_func {sf}"));
        }
    }
    if let Some(tm) = k.str("topk_method")? {
        if tm != "noaux_tc" {
            return err(format!("{mt}: topk_method {tm}"));
        }
    }
    k.expect("moe_router_dtype", &Value::from("float32"))?;
    k.expect("moe_layer_freq", &Value::from(1))?;
    k.opt_usize("ep_size")?;
    let types = glm_types(&k, layers, mt)?;
    // the release's linear-attention block (else transformers' keys)
    let lac = match k.raw("linear_attn_config") {
        Some(Value::Object(o)) => Some(o),
        _ => None,
    };
    let lin_num = |a: &str, b: &str| -> Result<Option<usize>> {
        match lac.and_then(|o| o.get(a)).and_then(|v| v.as_u64()) {
            Some(v) => Ok(Some(v as usize)),
            None => k.opt_usize(b),
        }
    };
    let types = if lac.is_some() && k.m.get("layer_types").is_none() {
        let kda: BTreeSet<u64> =
            lac.and_then(|o| o.get("kda_layers")).and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|x| x.as_u64()).collect()).unwrap_or_default();
        (0..layers).map(|l| kda.contains(&(l as u64))).collect()
    } else {
        types
    };
    let bound = match lac.and_then(|o| o.get("gate_lower_bound")) {
        Some(v) => v.as_f64(),
        None => k.opt_f64("linear_lower_bound")?,
    };
    let lin = Linear {
        kind: LinearKind::KimiDelta,
        k_heads: lin_num("num_heads", "linear_num_heads")?.unwrap_or(0),
        v_heads: lin_num("num_heads", "linear_num_heads")?.unwrap_or(0),
        k_dim: lin_num("head_dim", "linear_head_dim")?.unwrap_or(0),
        v_dim: lin_num("head_dim", "linear_head_dim")?.unwrap_or(0),
        conv: lin_num("short_conv_kernel_size", "linear_conv_kernel_dim")?.unwrap_or(4),
        decay: match bound {
            Some(b) => Decay::Bounded(b as f32),
            None => Decay::Softplus,
        },
        gate_act: Act::Sigmoid,
        norm_eps: eps,
        state_f32: true,
    };
    if types.iter().any(|&t| t) && lin.k_heads == 0 {
        return err(format!("{mt}: linear-attention layers without their heads"));
    }
    // the latent attention
    let (heads, nope, rd, vd, kvl) = (
        k.usize("num_attention_heads")?,
        k.usize("qk_nope_head_dim")?,
        k.opt_usize("qk_rope_head_dim")?.unwrap_or(0),
        k.usize("v_head_dim")?,
        k.usize("kv_lora_rank")?,
    );
    k.opt_usize("num_key_value_heads")?;
    k.opt_usize("qk_head_dim")?;
    k.opt_usize("head_dim")?;
    let q_lora = k.opt_usize("q_lora_rank")?;
    let nope_only = k.bool("mla_use_nope", rd == 0)?;
    let rope = if rd > 0 && !nope_only {
        let mut r = match k.raw("rope_parameters") {
            Some(Value::Object(o)) => rope(o, rd, max_pos, &format!("{mt} rope_parameters"))?,
            _ => legacy_rope(&k, rd, max_pos)?,
        };
        r.style = RopeStyle::Interleaved;
        Some(r)
    } else {
        k.raw("rope_parameters");
        k.raw("rope_theta");
        None
    };
    let mut scale = ((nope + rd) as f32).powf(-0.5);
    if let Some(Rope { scaling: RopeScaling::Yarn { factor, mscale_all_dim: Some(m), .. }, .. }) = &rope {
        let ms = if *factor <= 1.0 { 1.0 } else { 0.1 * m * factor.ln() + 1.0 };
        scale *= (ms * ms) as f32;
    }
    let mla_eps = if flash { eps } else { 1e-6 };
    let (ih, idim, itop) = (k.opt_usize("index_n_heads")?, k.opt_usize("index_head_dim")?, k.opt_usize("index_topk")?);
    let pool = k.opt_usize("index_kpool")?.unwrap_or(1);
    for x in ["index_kpool_always_select_tail", "index_kpool_compress", "index_share_for_mtp_iteration", "indexer_rope_interleave"] {
        k.bool(x, true)?;
    }
    let itypes: Vec<String> =
        k.raw("indexer_types").and_then(|v| v.as_array()).map(|a| a.iter().map(|x| x.as_str().unwrap_or("full").to_string()).collect()).unwrap_or_default();
    // the feed-forward
    let first_dense = k.opt_usize("first_k_dense_replace")?;
    let mlp_types: Option<Vec<bool>> = k.raw("mlp_layer_types").and_then(|v| v.as_array()).map(|a| a.iter().map(|x| x.as_str() == Some("sparse")).collect());
    let dense_inter = k.usize("intermediate_size")?;
    k.usize("moe_intermediate_size")?;
    let clamp = if flash { k.opt_f64("swiglu_limit")?.map(|v| v as f32) } else { None };
    let mlp = Mlp { act, gated: true, bias: false, clamp };
    let ne = k.usize("n_routed_experts")?;
    let ng = k.opt_usize("n_group")?.unwrap_or(1);
    let tg = k.opt_usize("topk_group")?.unwrap_or(1);
    let moe = Moe {
        router: Router {
            score: RouterScore::Sigmoid,
            logit_scale: 1.0,
            bias: true,
            groups: (ng > 1).then_some((ng, tg)),
            top_k: k.usize("num_experts_per_tok")?,
            renorm: k.bool("norm_topk_prob", true)?,
            scale: k.opt_f64("routed_scaling_factor")?.unwrap_or(1.0) as f32,
            hash: false,
        },
        experts: ne,
        expert: mlp,
        shared: (k.opt_usize("n_shared_experts")?.unwrap_or(0) > 0).then_some(mlp),
        shared_gate: false,
        latent: None,
    };
    // the residual streams (GLM-5.3-Flash)
    let streams = if flash && k.bool("mhc", true)? {
        Some(StreamsP {
            streams: k.opt_usize("hc_mult")?.unwrap_or(4),
            sinkhorn_iters: k.opt_usize("hc_sinkhorn_iters")?.unwrap_or(20),
            eps: k.opt_f64("hc_eps")?.unwrap_or(1e-6) as f32,
            norm_eps: eps,
        })
    } else {
        None
    };
    let mtp_layers = k.opt_usize("num_nextn_predict_layers")?.unwrap_or(0);
    let n = Norm::rms(eps);
    let fp8w = |w: String, out: &mut BTreeSet<String>| -> Source {
        if fp8.as_ref().is_some_and(|f| f.holds(&w)) {
            out.insert(w.clone());
        }
        Source::Tensor(w)
    };
    let mut fp8set = BTreeSet::new();
    let mut dense_shapes = Vec::new();
    let (mut blocks, mut roles) = (Vec::new(), Vec::new());
    for (l, &is_lin) in types.iter().enumerate() {
        let p = format!("{prefix}layers.{l}");
        let a = format!("{p}.self_attn");
        let residual = |sub: &str, r: &mut Vec<(String, Source)>| -> Residual {
            match streams {
                Some(s) => {
                    for (role, x) in [("hc_fn", "fn"), ("hc_base", "base"), ("hc_scale", "scale")] {
                        r.push((role.to_string(), t(format!("{p}.hc_{sub}_{x}"))));
                    }
                    Residual::Streams(s)
                }
                None => Residual::Add,
            }
        };
        let mut r = vec![("norm_in".to_string(), t(format!("{p}.input_layernorm.weight")))];
        let mixer = if is_lin {
            for (role, x) in [
                ("q", "q_proj"),
                ("k", "k_proj"),
                ("v", "v_proj"),
                ("beta", "b_proj"),
                ("decay_a", "f_a_proj"),
                ("decay_b", "f_b_proj"),
                ("gate_a", "g_a_proj"),
                ("gate_b", "g_b_proj"),
                ("o", "o_proj"),
            ] {
                r.push((role.to_string(), fp8w(format!("{a}.{x}.weight"), &mut fp8set)));
            }
            r.push(("conv".into(), Source::Rows(["q", "k", "v"].iter().map(|x| format!("{a}.{x}_conv1d.weight")).collect())));
            for (role, x) in [("a_log", "A_log"), ("dt_bias", "dt_bias"), ("norm", "o_norm.weight")] {
                r.push((role.to_string(), t(format!("{a}.{x}"))));
            }
            Mixer::Linear(lin.clone())
        } else {
            match q_lora {
                Some(_) => {
                    r.push(("q_a".into(), fp8w(format!("{a}.q_a_proj.weight"), &mut fp8set)));
                    r.push(("q_a_norm".into(), t(format!("{a}.q_a_layernorm.weight"))));
                    r.push(("q_b".into(), fp8w(format!("{a}.q_b_proj.weight"), &mut fp8set)));
                }
                None => r.push(("q".into(), fp8w(format!("{a}.q_proj.weight"), &mut fp8set))),
            }
            r.push(("kv_a".into(), fp8w(format!("{a}.kv_a_proj_with_mqa.weight"), &mut fp8set)));
            r.push(("kv_a_norm".into(), t(format!("{a}.kv_a_layernorm.weight"))));
            r.push(("kv_b".into(), fp8w(format!("{a}.kv_b_proj.weight"), &mut fp8set)));
            r.push(("o".into(), fp8w(format!("{a}.o_proj.weight"), &mut fp8set)));
            let shared = itypes.get(l).is_some_and(|s| s == "shared");
            let indexer = match (ih, idim, itop) {
                (Some(h), Some(dim), Some(top)) => {
                    if !shared {
                        let ix = format!("{a}.indexer");
                        let mut names = vec![
                            ("indexer_wq_b", "wq_b.weight"),
                            ("indexer_wk", "wk.weight"),
                            ("indexer_k_norm", "k_norm.weight"),
                            ("indexer_k_norm_bias", "k_norm.bias"),
                            ("indexer_weights", "weights_proj.weight"),
                        ];
                        if flash {
                            names.extend([("indexer_pool_ape", "index_kpool_compress_ape"), ("indexer_pool_gate", "index_kpool_compress_gate")]);
                        }
                        for (role, x) in names {
                            r.push((role.to_string(), t(format!("{ix}.{x}"))));
                        }
                    }
                    Some(Indexer { heads: h, head_dim: dim, top_k: top, block: pool, shared })
                }
                _ => None,
            };
            Mixer::Mla(Mla {
                heads,
                q_lora,
                kv_lora: kvl,
                nope,
                rope_dim: rd,
                v_dim: vd,
                rope: rope.clone(),
                q_norm: Norm::rms(mla_eps),
                kv_norm: Norm::rms(mla_eps),
                gate: None,
                scale,
                absorbed: true,
                indexer,
            })
        };
        let res = residual("attn", &mut r);
        blocks.push(Block { norm_in: n, body: Body::Mixer(mixer), norm_out: None, residual: res });
        roles.push(r);
        let sparse = match (&mlp_types, first_dense) {
            (Some(t), _) => t.get(l).copied().unwrap_or(true),
            (None, Some(f)) => l >= f,
            (None, None) => true,
        };
        let f = format!("{p}.mlp");
        let mut r = vec![("norm_in".to_string(), t(format!("{p}.post_attention_layernorm.weight")))];
        let body = if sparse {
            r.push(("router".into(), t(format!("{f}.gate.weight"))));
            r.push(("choice_bias".into(), t(format!("{f}.gate.e_score_correction_bias"))));
            let ex = |x: &str, out: &mut BTreeSet<String>| -> Vec<String> {
                let w = format!("{f}.experts.0.{x}_proj.weight");
                let _ = fp8w(w, out);
                (0..ne).map(|e| format!("{f}.experts.{e}.{x}_proj.weight")).collect()
            };
            let (g, u, dn) = (ex("gate", &mut fp8set), ex("up", &mut fp8set), ex("down", &mut fp8set));
            if fp8set.contains(&format!("{f}.experts.0.gate_proj.weight")) {
                fp8set.extend(g.iter().chain(&u).chain(&dn).cloned());
            }
            r.push(("experts_gate_up".into(), Source::Experts(g.into_iter().zip(u).map(|(a, b)| vec![a, b]).collect())));
            r.push(("experts_down".into(), Source::Experts(dn.into_iter().map(|x| vec![x]).collect())));
            if moe.shared.is_some() {
                for x in ["gate", "up", "down"] {
                    r.push((format!("shared_{x}"), fp8w(format!("{f}.shared_experts.{x}_proj.weight"), &mut fp8set)));
                }
            }
            Body::Ffn(Ffn::Moe(moe.clone()))
        } else {
            for x in ["gate", "up", "down"] {
                r.push((x.to_string(), fp8w(format!("{f}.{x}_proj.weight"), &mut fp8set)));
            }
            // (a dense MLP's shapes, which the lowering to the executor reads its width from)
            for (x, s) in [("gate", vec![dense_inter, d]), ("up", vec![dense_inter, d]), ("down", vec![d, dense_inter])] {
                dense_shapes.push((format!("{f}.{x}_proj.weight"), s));
            }
            Body::Ffn(Ffn::Mlp(mlp))
        };
        let res = residual("ffn", &mut r);
        blocks.push(Block { norm_in: n, body, norm_out: None, residual: res });
        roles.push(r);
    }
    let desc = ModelDesc {
        hidden: d,
        vocab,
        embed: Embed { scale: 1.0, norm: None, streams: streams.map_or(1, |s| s.streams) },
        blocks,
        final_norm: n,
        head: Head { tied, softcap: None, scale: 1.0, collapse: if streams.is_some() { Collapse::Mean } else { Collapse::None } },
        mtp: Vec::new(),
        vision: None,
        rope,
    };
    let mut out = finish(&k, desc, prefix, roles, mtp_layers)?;
    out.scales = fp8set.into_iter().map(|w| (w.clone(), format!("{w}_scale_inv"))).collect();
    out.shapes = dense_shapes.into_iter().collect();
    let _ = fp8.map(|f| f.block);
    Ok(out)
}

/// The top-level keys of a multimodal config that the text model does not read.
fn top_keys(cfg: &Value, what: &str, known: &[&str]) -> Result<()> {
    for k in cfg.as_object().map(|m| m.keys().collect::<Vec<_>>()).unwrap_or_default() {
        if !known.contains(&k.as_str()) && !inert(k) {
            return err(format!("{what}: config key {k} is not understood by the description builder"));
        }
    }
    Ok(())
}

/// Kimi-K3 (`kimi_k3`, a `kimi_linear` text model): Kimi delta attention and gated NoPE MLA, the depth mix over groups of
/// `attn_res_block_size` layers, a dense first layer then the latent MoE, situ activations, KimiRMSNorm (two roundings).
fn kimi(cfg: &Value, tc: &Map<String, Value>) -> Result<Described> {
    let mt = "kimi_linear";
    top_keys(cfg, "kimi_k3", &["text_config", "vision_config", "quantization_config", "tie_word_embeddings"])?;
    let k = Keys::new(tc, mt);
    if tc.get("model_type").and_then(|v| v.as_str()) != Some(mt) {
        return err("kimi_k3: the text model is not kimi_linear");
    }
    let (d, layers, vocab, eps, tied) = common(&k, None)?;
    if k.str("hidden_act")? != Some("situ") {
        return err(format!("{mt}: only situ is supported"));
    }
    let situ = Act::Situ { beta: k.f64("activation_situ_beta")? as f32, linear: k.f64("activation_situ_linear_beta")? as f32 };
    let mlp = Mlp { act: situ, gated: true, bias: false, clamp: None };
    let twice = |e: f32| Norm { rounding: Rounding::Twice, ..Norm::rms(e) };
    for (x, want) in [("mla_use_nope", true), ("mla_use_output_gate", true), ("latent_moe_use_norm", true), ("moe_renormalize", true)] {
        if k.bool(x, want)? != want {
            return err(format!("{mt}: {x} = {} is not supported", !want));
        }
    }
    k.expect("moe_router_activation_func", &Value::from("sigmoid"))?;
    k.expect("num_expert_group", &Value::from(1))?;
    k.expect("topk_group", &Value::from(1))?;
    k.expect("topk_method", &Value::from("noaux_tc"))?;
    // (grouped top-k over one group of all the experts, which the two above make it, is the plain top-k)
    let _ = k.raw("use_grouped_topk");
    k.expect("moe_layer_freq", &Value::from(1))?;
    k.expect("rope_scaling", &Value::Null)?;
    k.opt_usize("max_position_embeddings")?;
    let la = match k.raw("linear_attn_config") {
        Some(Value::Object(o)) => o,
        _ => return err(format!("{mt}: no linear_attn_config")),
    };
    let lk = Keys::new(la, "linear_attn_config");
    let list = |x: &str| -> Result<BTreeSet<u64>> {
        lk.raw(x)
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
            .ok_or_else(|| Error(format!("{mt}: no linear_attn_config.{x}")))
    };
    let (kda_l, full_l) = (list("kda_layers")?, list("full_attn_layers")?);
    if (1..=layers as u64).any(|n| kda_l.contains(&n) == full_l.contains(&n)) {
        return err(format!("{mt}: kda_layers and full_attn_layers are not the {layers} layers (1-based) split in two"));
    }
    if !lk.bool("use_full_rank_gate", false)? {
        return err(format!("{mt}: a low-rank output gate in the delta attention is not supported"));
    }
    let (kh, kd) = (lk.usize("num_heads")?, lk.usize("head_dim")?);
    let lin = Linear {
        kind: LinearKind::KimiDelta,
        k_heads: kh,
        v_heads: kh,
        k_dim: kd,
        v_dim: kd,
        conv: lk.usize("short_conv_kernel_size")?,
        decay: Decay::Bounded(lk.f64("gate_lower_bound")? as f32),
        gate_act: Act::Sigmoid,
        norm_eps: 1e-5,
        state_f32: true,
    };
    lk.finish()?;
    let heads = k.usize("num_attention_heads")?;
    k.expect("num_key_value_heads", &Value::from(heads))?;
    let (nope, rd) = (k.usize("qk_nope_head_dim")?, k.usize("qk_rope_head_dim")?);
    let mla = Mla {
        heads,
        q_lora: Some(k.usize("q_lora_rank")?),
        kv_lora: k.usize("kv_lora_rank")?,
        nope,
        rope_dim: rd,
        v_dim: k.usize("v_head_dim")?,
        rope: None,
        q_norm: twice(1e-6),
        kv_norm: twice(1e-6),
        gate: Some(OutGate { act: Act::Sigmoid, from: GateFrom::Own, norm: None }),
        scale: ((nope + rd) as f32).powf(-0.5),
        absorbed: true,
        indexer: None,
    };
    let dense = k.usize("first_k_dense_replace")?;
    let inter = k.usize("intermediate_size")?;
    let moe_inter = k.usize("moe_intermediate_size")?;
    let ne = k.usize("num_experts")?;
    // (the routed experts as released: MXFP4, 4-bit floats in groups of 32 with a byte's scale each, as compressed-tensors packs them; every other matrix bf16)
    let mxfp4 = match k.raw("quantization_config") {
        None => false,
        Some(q) => {
            kimi_mxfp4(q)?;
            true
        }
    };
    let mut scales = std::collections::BTreeMap::new();
    let n_shared = k.usize("num_shared_experts")?;
    let shared = n_shared > 0;
    let lat = k.usize("routed_expert_hidden_size")?;
    let moe = Moe {
        router: Router {
            score: RouterScore::Sigmoid,
            logit_scale: 1.0,
            bias: true,
            groups: None,
            top_k: k.usize("num_experts_per_token")?,
            renorm: true,
            scale: k.f64("routed_scaling_factor")? as f32,
            hash: false,
        },
        experts: ne,
        expert: mlp,
        shared: shared.then_some(mlp),
        shared_gate: false,
        latent: Some(Latent { dim: lat, norm_in: None, expert_norm: None, norm_out: Some(twice(eps)) }),
    };
    let group = k.usize("attn_res_block_size")?;
    let prefix = "language_model.model.";
    let n = twice(eps);
    let (mut blocks, mut roles, mut shapes) = (Vec::new(), Vec::new(), Vec::new());
    for l in 0..layers {
        let p = format!("{prefix}layers.{l}");
        let a = format!("{p}.self_attn");
        let mut r = vec![
            ("norm_in".to_string(), t(format!("{p}.input_layernorm.weight"))),
            ("mix_norm".into(), t(format!("{p}.self_attention_res_norm.weight"))),
            ("mix_proj".into(), t(format!("{p}.self_attention_res_proj.weight"))),
        ];
        let mixer = if kda_l.contains(&(l as u64 + 1)) {
            for (role, x) in [
                ("q", "q_proj"),
                ("k", "k_proj"),
                ("v", "v_proj"),
                ("beta", "b_proj"),
                ("decay_a", "f_a_proj"),
                ("decay_b", "f_b_proj"),
                ("gate", "g_proj"),
                ("o", "o_proj"),
            ] {
                r.push((role.to_string(), t(format!("{a}.{x}.weight"))));
            }
            r.push(("conv".into(), Source::Rows(["q", "k", "v"].iter().map(|x| format!("{a}.{x}_conv1d.weight")).collect())));
            // (the forget gate's low rank: the delta heads' size, which the lowering to the executor sizes its slot by)
            shapes.push((format!("{a}.f_a_proj.weight"), vec![kd, d]));
            for (role, x) in [("a_log", "A_log"), ("dt_bias", "dt_bias"), ("norm", "o_norm.weight")] {
                r.push((role.to_string(), t(format!("{a}.{x}"))));
            }
            Mixer::Linear(lin.clone())
        } else {
            for (role, x) in [
                ("q_a", "q_a_proj.weight"),
                ("q_a_norm", "q_a_layernorm.weight"),
                ("q_b", "q_b_proj.weight"),
                ("kv_a", "kv_a_proj_with_mqa.weight"),
                ("kv_a_norm", "kv_a_layernorm.weight"),
                ("kv_b", "kv_b_proj.weight"),
                ("g", "g_proj.weight"),
                ("o", "o_proj.weight"),
            ] {
                r.push((role.to_string(), t(format!("{a}.{x}"))));
            }
            Mixer::Mla(mla.clone())
        };
        blocks.push(Block { norm_in: n, body: Body::Mixer(mixer), norm_out: None, residual: Residual::DepthMix(DepthMixP { starts: l % group == 0 }) });
        roles.push(r);
        let mut r = vec![
            ("norm_in".to_string(), t(format!("{p}.post_attention_layernorm.weight"))),
            ("mix_norm".into(), t(format!("{p}.mlp_res_norm.weight"))),
            ("mix_proj".into(), t(format!("{p}.mlp_res_proj.weight"))),
        ];
        let body = if l < dense {
            for x in ["gate", "up", "down"] {
                r.push((x.to_string(), t(format!("{p}.mlp.{x}_proj.weight"))));
            }
            shapes.push((format!("{p}.mlp.gate_proj.weight"), vec![inter, d]));
            Body::Ffn(Ffn::Mlp(mlp))
        } else {
            let f = format!("{p}.block_sparse_moe");
            for (role, x) in [
                ("router", "gate.weight"),
                ("choice_bias", "gate.e_score_correction_bias"),
                ("latent_down", "routed_expert_down_proj.weight"),
                ("latent_up", "routed_expert_up_proj.weight"),
                ("latent_norm_out", "routed_expert_norm.weight"),
            ] {
                r.push((role.to_string(), t(format!("{f}.{x}"))));
            }
            let mut ex = |e: usize, w: &str| {
                let n = format!("{f}.experts.{e}.{w}.{}", if mxfp4 { "weight_packed" } else { "weight" });
                if mxfp4 {
                    scales.insert(n.clone(), n.replace(".weight_packed", ".weight_scale"));
                }
                n
            };
            let gate_up: Vec<Vec<String>> = (0..ne).map(|e| vec![ex(e, "w1"), ex(e, "w3")]).collect();
            let down: Vec<Vec<String>> = (0..ne).map(|e| vec![ex(e, "w2")]).collect();
            r.push(("experts_gate_up".into(), Source::Experts(gate_up)));
            r.push(("experts_down".into(), Source::Experts(down)));
            if shared {
                for x in ["gate", "up", "down"] {
                    r.push((format!("shared_{x}"), t(format!("{f}.shared_experts.{x}_proj.weight"))));
                }
                shapes.push((format!("{f}.shared_experts.gate_proj.weight"), vec![n_shared * moe_inter, d]));
            }
            Body::Ffn(Ffn::Moe(moe.clone()))
        };
        blocks.push(Block { norm_in: n, body, norm_out: None, residual: Residual::DepthMix(DepthMixP { starts: false }) });
        roles.push(r);
    }
    // (the checkpoint's next-token-prediction layers, which the description does not use yet)
    let mtp_layers = k.opt_usize("num_nextn_predict_layers")?.unwrap_or(0);
    k.finish()?;
    Ok(Described {
        desc: ModelDesc {
            hidden: d,
            vocab,
            embed: Embed { scale: 1.0, norm: None, streams: 1 },
            blocks,
            final_norm: n,
            head: Head { tied, softcap: None, scale: 1.0, collapse: Collapse::DepthMix },
            mtp: Vec::new(),
            vision: None,
            rope: None,
        },
        embed: format!("{prefix}embed_tokens.weight"),
        final_norm: format!("{prefix}norm.weight"),
        head: (!tied).then(|| "language_model.lm_head.weight".to_string()),
        blocks: roles,
        extra: vec![
            ("out_mix_norm".into(), t(format!("{prefix}output_attn_res_norm.weight"))),
            ("out_mix_proj".into(), t(format!("{prefix}output_attn_res_proj.weight"))),
        ],
        mtp_layers,
        scales,
        shapes: shapes.into_iter().collect(),
    })
}

/// Kimi-K3's `quantization_config` as released: compressed-tensors' `mxfp4-pack-quantized`, 4-bit floats in symmetric groups of 32, the attention, the shared experts, the
/// dense layer's MLP and the head left out (so only the routed experts are packed). Anything else is refused with what differs.
fn kimi_mxfp4(q: &Value) -> Result<()> {
    let s = |v: &Value, k: &str| v.get(k).and_then(Value::as_str).map(str::to_string);
    let what = "kimi_linear: quantization_config";
    if s(q, "quant_method").as_deref() != Some("compressed-tensors") || s(q, "format").as_deref() != Some("mxfp4-pack-quantized") {
        return err(format!("{what}: {:?} {:?}, where the description reads compressed-tensors' mxfp4-pack-quantized", s(q, "quant_method"), s(q, "format")));
    }
    let groups = q.get("config_groups").and_then(Value::as_object).ok_or_else(|| Error(format!("{what}: no config_groups")))?;
    if groups.len() != 1 {
        return err(format!("{what}: {} config groups, where the description reads one", groups.len()));
    }
    let w = groups.values().next().and_then(|g| g.get("weights")).ok_or_else(|| Error(format!("{what}: a config group with no weights")))?;
    let n = |k: &str| w.get(k).and_then(Value::as_u64);
    if n("num_bits") != Some(4)
        || n("group_size") != Some(32)
        || s(w, "type").as_deref() != Some("float")
        || s(w, "strategy").as_deref() != Some("group")
        || w.get("symmetric").and_then(Value::as_bool) != Some(true)
    {
        return err(format!("{what}: weights {w}, where the description reads 4-bit floats in symmetric groups of 32"));
    }
    let ignore: Vec<String> =
        q.get("ignore").and_then(Value::as_array).map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default();
    for part in ["self_attn", "shared_experts", "mlp", "lm_head"] {
        if !ignore.iter().any(|x| x.contains(part)) {
            return err(format!("{what}: the {part} matrices are packed too, where the description reads them in bf16"));
        }
    }
    Ok(())
}

/// Muse Spark (`muse_spark`): GQA with weightless q/k norms, rope and a window on the sliding layers only, a gated output with a per-head
/// norm; a normed embedding; sandwich norms with Muse's gated residual; the latent MoE in every layer; a soft-capped head.
fn muse(tc: &Map<String, Value>, prefix: &str) -> Result<Described> {
    let mt = "muse_spark_text";
    let k = Keys::new(tc, mt);
    let (d, layers, vocab, eps, tied) = common(&k, None)?;
    let pe = k.f64("post_norm_eps")? as f32;
    k.expect("hidden_act", &Value::from("silu"))?;
    k.expect("attention_bias", &Value::Bool(false))?;
    k.expect("num_shared_experts", &Value::from(0))?;
    k.opt_usize("max_position_embeddings")?;
    k.usize("intermediate_size")?;
    let (heads, kv, hd) = (k.usize("num_attention_heads")?, k.usize("num_key_value_heads")?, k.usize("head_dim")?);
    let window = k.usize("sliding_window")?;
    let strs = |x: &str| -> Result<Vec<String>> {
        Ok(k.raw(x).and_then(|v| v.as_array()).ok_or_else(|| Error(format!("{mt}: no {x}")))?.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect())
    };
    let (lt, mlt) = (strs("layer_types")?, strs("mlp_layer_types")?);
    if lt.len() != layers || mlt.iter().any(|m| m != "sparse") {
        return err(format!("{mt}: layer types other than one per layer, every MLP sparse"));
    }
    let thetas: Vec<f64> = k
        .raw("layer_rope_theta")
        .and_then(|v| v.as_array())
        .ok_or_else(|| Error(format!("{mt}: no layer_rope_theta")))?
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0))
        .collect();
    if let Some(Value::Object(rp)) = k.raw("rope_parameters") {
        if rp.get("rope_type").and_then(|v| v.as_str()).is_some_and(|t| t != "default") {
            return err(format!("{mt}: rope scaling is not supported"));
        }
    }
    let qk = k.f64("qk_scale_factor")? as f32;
    let out_mult = k.f64("output_multiplier")? as f32;
    let softcap = k.opt_f64("final_logit_softcapping")?.map(|v| v as f32);
    let temperature = k.f64("residual_gate_temperature")? as f32;
    let (ne, lat) = (k.usize("num_local_experts")?, k.usize("moe_hidden_size")?);
    k.usize("moe_intermediate_size")?;
    let moe = Moe {
        router: Router {
            score: RouterScore::Sigmoid,
            logit_scale: out_mult,
            bias: true,
            groups: None,
            top_k: k.usize("num_experts_per_tok")?,
            renorm: true,
            scale: k.f64("routed_scaling_factor")? as f32,
            hash: false,
        },
        experts: ne,
        expert: Mlp { act: Act::Silu, gated: true, bias: false, clamp: None },
        shared: None,
        shared_gate: false,
        latent: Some(Latent {
            dim: lat,
            norm_in: Some(Norm::rms_plus_one(eps)),
            expert_norm: Some(Norm { weight: NormWeight::InputPlusOne, ..Norm::rms(pe) }),
            norm_out: None,
        }),
    };
    let bare = Norm::rms_bare(eps);
    let (mut blocks, mut roles) = (Vec::new(), Vec::new());
    for l in 0..layers {
        let sliding = lt[l] == "sliding_attention";
        let theta = thetas.get(l).copied().unwrap_or(0.0);
        if sliding != (theta > 0.0) {
            return err(format!("{mt}: layer {l}: a rotary that is not on exactly the window's layers"));
        }
        let a = Attn {
            heads,
            kv_heads: kv,
            head_dim: hd,
            v_head_dim: hd,
            window: sliding.then_some(window),
            sinks: false,
            rope: sliding.then(|| Rope::plain(theta, hd)),
            q_norm: Some(bare),
            k_norm: Some(bare),
            v_norm: None,
            gate: Some(OutGate { act: Act::Sigmoid, from: GateFrom::Own, norm: Some(bare) }),
            softcap: None,
            bias: false,
            scale: qk / (hd as f32).sqrt(),
            k_eq_v: false,
            indexer: None,
            rounding: AttnRounding::Sdpa,
        };
        let p = format!("{prefix}layers.{l}");
        let mut r =
            vec![("norm_in".to_string(), t(format!("{p}.input_layernorm.weight"))), ("res_gate".into(), t(format!("{p}.post_attention_residual_gate.gate")))];
        for (role, x) in [("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("g", "gate_proj"), ("o", "o_proj")] {
            r.push((role.to_string(), t(format!("{p}.self_attn.{x}.weight"))));
        }
        blocks.push(Block {
            norm_in: Norm::rms_plus_one(eps),
            body: Body::Mixer(Mixer::Attention(a)),
            norm_out: Some(Norm::rms_bare(pe)),
            residual: Residual::Gated(GateP { temperature }),
        });
        roles.push(r);
        let m = format!("{p}.mlp");
        let mut r = vec![
            ("norm_in".to_string(), t(format!("{p}.pre_feedforward_layernorm.weight"))),
            ("norm_out".into(), t(format!("{p}.post_feedforward_layernorm.weight"))),
            ("res_gate".into(), t(format!("{p}.post_feedforward_residual_gate.gate"))),
        ];
        for (role, x) in [
            ("router", "gate.weight"),
            ("choice_bias", "gate.e_score_correction_bias"),
            ("latent_down", "pre_expert_proj.weight"),
            ("latent_norm_in", "pre_expert_norm.weight"),
            ("experts_gate_up", "experts.gate_up_proj"),
            ("experts_down", "experts.down_proj"),
            ("expert_norm", "experts.post_expert_norm.weight"),
            ("latent_up", "post_expert_proj.weight"),
        ] {
            r.push((role.to_string(), t(format!("{m}.{x}"))));
        }
        blocks.push(Block {
            norm_in: Norm::rms_plus_one(eps),
            body: Body::Ffn(Ffn::Moe(moe.clone())),
            norm_out: Some(Norm { weight: NormWeight::InputPlusOne, ..Norm::rms(pe) }),
            residual: Residual::Gated(GateP { temperature }),
        });
        roles.push(r);
    }
    let desc = ModelDesc {
        hidden: d,
        vocab,
        embed: Embed { scale: 1.0, norm: Some(bare), streams: 1 },
        blocks,
        final_norm: Norm::rms(eps),
        head: Head { tied, softcap, scale: out_mult, collapse: Collapse::None },
        mtp: Vec::new(),
        vision: None,
        rope: None,
    };
    finish(&k, desc, prefix, roles, 0)
}

/// DeepSeek-V4 (`deepseek_v4`): the compressed attention (`kinds::Comp`), MoE with hash routing in the first layers then sqrt-softplus
/// with a choice bias, SwiGLU limits, 4 mHC streams and the weighted hyper-head. Two config flavours: the release's (`compress_ratios`
/// per layer, a ratio of 4 being the overlapping windows with the indexer; `num_hash_layers`; YaRN `rope_scaling` and
/// `compress_rope_theta` on the compressing layers; FP8 matrices and FP4 experts with `.scale` tensors; names without a prefix) and
/// transformers' (`layer_types`, `compress_rates`, `rope_parameters` main and compress, `mlp_layer_types`; names under `model.`).
fn deepseek_v4(tc: &Map<String, Value>) -> Result<Described> {
    let mt = "deepseek_v4";
    let k = Keys::new(tc, mt);
    let (d, layers, vocab, eps, tied) = common(&k, None)?;
    let max_pos = k.opt_usize("max_position_embeddings")?;
    k.expect("hidden_act", &Value::from("silu"))?;
    k.expect("attention_bias", &Value::Bool(false))?;
    k.expect("mlp_bias", &Value::Bool(false))?;
    k.expect("scoring_func", &Value::from("sqrtsoftplus"))?;
    k.expect("topk_method", &Value::from("noaux_tc"))?;
    k.expect("num_key_value_heads", &Value::from(1))?;
    k.expect("n_shared_experts", &Value::from(1))?;
    k.expect("norm_topk_prob", &Value::Bool(true))?;
    k.expect("router_jitter_noise", &Value::from(0.0))?;
    k.raw("architectures");
    let released = k.m.contains_key("compress_ratios");
    let fp4 = k.str("expert_dtype")? == Some("fp4");
    // (the DSpark drafter's settings: its block of draft tokens, its noise token, the layers it reads and its rank. A drafter, not this model's computation; drafters are #333's)
    for x in ["dspark_block_size", "dspark_noise_token_id", "dspark_target_layer_ids", "dspark_markov_rank"] {
        let _ = k.raw(x);
    }
    let fp8 = Fp8::parse(k.raw("quantization_config"), mt)?.is_some();
    let (heads, hd, rd) = (k.usize("num_attention_heads")?, k.usize("head_dim")?, k.usize("qk_rope_head_dim")?);
    k.opt_f64("partial_rotary_factor")?;
    let (q_lora, o_lora, o_groups, window) =
        (k.usize("q_lora_rank")?, k.usize("o_lora_rank")?, k.usize("o_groups")?, k.opt_usize("sliding_window")?.unwrap_or(128));
    let (ih, idim, itop) = (k.usize("index_n_heads")?, k.usize("index_head_dim")?, k.usize("index_topk")?);
    let (ne, top_k) = (k.usize("n_routed_experts")?, k.usize("num_experts_per_tok")?);
    k.usize("moe_intermediate_size")?;
    let limit = k.opt_f64("swiglu_limit")?.filter(|&l| l > 0.0).map(|l| l as f32);
    let scale = k.opt_f64("routed_scaling_factor")?.unwrap_or(1.0) as f32;
    let streams = StreamsP {
        streams: k.usize("hc_mult")?,
        sinkhorn_iters: k.opt_usize("hc_sinkhorn_iters")?.unwrap_or(20),
        eps: k.opt_f64("hc_eps")?.unwrap_or(1e-6) as f32,
        norm_eps: eps,
    };
    let mtp_layers = k.opt_usize("num_nextn_predict_layers")?.unwrap_or(0);
    // each layer's compression ratio and whether it is CSA (overlap and indexer), its rope, and whether its MoE is hash-routed
    let (ratios, csa, hash, main_rope, comp_rope): (Vec<usize>, Vec<bool>, Vec<bool>, Rope, Rope) = if released {
        let r: Vec<usize> =
            k.raw("compress_ratios").and_then(|v| v.as_array()).map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect()).unwrap_or_default();
        if r.len() < layers {
            return err(format!("{mt}: {} compress_ratios for {layers} layers", r.len()));
        }
        let r = r[..layers].to_vec();
        let nh = k.opt_usize("num_hash_layers")?.unwrap_or(0);
        let mut main = Rope::plain(k.opt_f64("rope_theta")?.unwrap_or(10_000.0), rd);
        main.style = RopeStyle::Interleaved;
        let mut comp_m = Map::new();
        if let Some(Value::Object(o)) = k.raw("rope_scaling") {
            comp_m = o.clone();
        }
        comp_m.insert("rope_theta".into(), Value::from(k.opt_f64("compress_rope_theta")?.unwrap_or(160_000.0)));
        // (the release's model.py stretches the frequencies only: YaRN without the attention factor transformers puts on cos and sin)
        if comp_m.contains_key("factor") && !comp_m.contains_key("attention_factor") {
            comp_m.insert("attention_factor".into(), Value::from(1.0));
        }
        let mut comp = rope(&comp_m, rd, max_pos, &format!("{mt} rope_scaling"))?;
        comp.style = RopeStyle::Interleaved;
        (r.clone(), r.iter().map(|&x| x == 4).collect(), (0..layers).map(|l| l < nh).collect(), main, comp)
    } else {
        let rates = match k.raw("compress_rates") {
            Some(Value::Object(o)) => o.clone(),
            _ => Map::new(),
        };
        let rate = |t: &str| rates.get(t).and_then(|v| v.as_u64()).map(|x| x as usize);
        let types: Vec<String> = k
            .raw("layer_types")
            .and_then(|v| v.as_array())
            .ok_or_else(|| Error(format!("{mt}: no layer_types")))?
            .iter()
            .map(|x| x.as_str().unwrap_or("").to_string())
            .collect();
        let mut r = Vec::new();
        for t in types.iter().take(layers) {
            r.push(match t.as_str() {
                "sliding_attention" => 0,
                "compressed_sparse_attention" | "heavily_compressed_attention" => rate(t).ok_or_else(|| Error(format!("{mt}: no compress_rates.{t}")))?,
                o => return err(format!("{mt}: layer type {o}")),
            });
        }
        let mlp: Vec<String> =
            k.raw("mlp_layer_types").and_then(|v| v.as_array()).map(|a| a.iter().map(|x| x.as_str().unwrap_or("").to_string()).collect()).unwrap_or_default();
        let rp = match k.raw("rope_parameters") {
            Some(Value::Object(o)) => o.clone(),
            _ => return err(format!("{mt}: no rope_parameters")),
        };
        let one = |t: &str| -> Result<Rope> {
            let o = rp.get(t).and_then(|v| v.as_object()).ok_or_else(|| Error(format!("{mt}: no rope_parameters.{t}")))?;
            let mut o = o.clone();
            o.insert("partial_rotary_factor".into(), Value::from(1.0));
            let mut r = rope(&o, rd, max_pos, &format!("{mt} rope_parameters.{t}"))?;
            r.style = RopeStyle::Interleaved;
            Ok(r)
        };
        k.opt_f64("rope_theta")?;
        k.opt_f64("compress_rope_theta")?;
        (
            r,
            types.iter().take(layers).map(|t| t == "compressed_sparse_attention").collect(),
            (0..layers).map(|l| mlp.get(l).is_some_and(|m| m == "hash_moe")).collect(),
            one("main")?,
            one("compress")?,
        )
    };
    let (p0, names): (&str, [&str; 4]) = if released {
        ("", ["embed.weight", "norm.weight", "head.weight", "hc_head_"])
    } else {
        ("model.", ["model.embed_tokens.weight", "model.norm.weight", "head.weight", "model.hc_head.hc_"])
    };
    let kv_norm = if released { "kv_norm" } else { "norm" };
    let mut scales = std::collections::BTreeMap::new();
    let q8 = |w: String, scales: &mut std::collections::BTreeMap<String, String>| -> Source {
        if fp8 && released {
            scales.insert(w.clone(), w.replace(".weight", ".scale"));
        }
        t(w)
    };
    let mlp = Mlp { act: Act::Silu, gated: true, bias: false, clamp: limit };
    let (mut blocks, mut roles) = (Vec::new(), Vec::new());
    for l in 0..layers {
        let p = format!("{p0}layers.{l}");
        let a = format!("{p}.attn");
        let (ratio, is_csa) = (ratios[l], csa[l]);
        let mut r = vec![("norm_in".to_string(), t(format!("{p}.attn_norm.weight")))];
        for (role, x) in [("q_a", "wq_a"), ("q_b", "wq_b"), ("kv", "wkv"), ("o_a", "wo_a"), ("o_b", "wo_b")] {
            r.push((role.to_string(), q8(format!("{a}.{x}.weight"), &mut scales)));
        }
        r.push(("q_a_norm".into(), t(format!("{a}.q_norm.weight"))));
        r.push(("kv_norm".into(), t(format!("{a}.{kv_norm}.weight"))));
        r.push(("sinks".into(), t(format!("{a}.attn_sink"))));
        let pool = |pre: &str, at: &str, r: &mut Vec<(String, Source)>| {
            for (role, x) in [("kv", "wkv.weight"), ("gate", "wgate.weight"), ("ape", "ape"), ("norm", "norm.weight")] {
                r.push((format!("{pre}_{role}"), t(format!("{at}.{x}"))));
            }
        };
        if ratio > 0 {
            pool("comp", &format!("{a}.compressor"), &mut r);
        }
        if ratio > 0 && is_csa {
            pool("idx", &format!("{a}.indexer.compressor"), &mut r);
            r.push(("idx_q_b".into(), q8(format!("{a}.indexer.wq_b.weight"), &mut scales)));
            r.push(("idx_weights".into(), t(format!("{a}.indexer.weights_proj.weight"))));
        }
        for (role, x) in [("hc_fn", "fn"), ("hc_base", "base"), ("hc_scale", "scale")] {
            r.push((role.to_string(), t(format!("{p}.hc_attn_{x}"))));
        }
        let c = Comp {
            heads,
            head_dim: hd,
            q_lora,
            o_groups,
            o_lora,
            window,
            ratio,
            overlap: ratio > 0 && is_csa,
            rope: if ratio > 0 { comp_rope.clone() } else { main_rope.clone() },
            indexer: (ratio > 0 && is_csa).then_some(Indexer { heads: ih, head_dim: idim, top_k: itop, block: ratio, shared: false }),
            norm_eps: eps,
        };
        blocks.push(Block { norm_in: Norm::rms(eps), body: Body::Mixer(Mixer::Compressed(c)), norm_out: None, residual: Residual::Streams(streams) });
        roles.push(r);
        let f = format!("{p}.ffn");
        let mut r = vec![("norm_in".to_string(), t(format!("{p}.ffn_norm.weight"))), ("router".into(), t(format!("{f}.gate.weight")))];
        if hash[l] {
            r.push(("hash".into(), t(format!("{f}.gate.tid2eid"))));
        } else {
            r.push(("choice_bias".into(), t(format!("{f}.gate.bias"))));
        }
        let ex = |w: &str, scales: &mut std::collections::BTreeMap<String, String>| -> Vec<String> {
            (0..ne)
                .map(|e| {
                    let n = format!("{f}.experts.{e}.{w}.weight");
                    if fp4 && released {
                        scales.insert(n.clone(), n.replace(".weight", ".scale"));
                    }
                    n
                })
                .collect()
        };
        let (w1, w3, w2) = (ex("w1", &mut scales), ex("w3", &mut scales), ex("w2", &mut scales));
        r.push(("experts_gate_up".into(), Source::Experts(w1.into_iter().zip(w3).map(|(g, u)| vec![g, u]).collect())));
        r.push(("experts_down".into(), Source::Experts(w2.into_iter().map(|x| vec![x]).collect())));
        for (role, x) in [("shared_gate", "w1"), ("shared_up", "w3"), ("shared_down", "w2")] {
            let w = format!("{f}.shared_experts.{x}.weight");
            if fp8 && released {
                scales.insert(w.clone(), w.replace(".weight", ".scale"));
            }
            r.push((role.to_string(), t(w)));
        }
        for (role, x) in [("hc_fn", "fn"), ("hc_base", "base"), ("hc_scale", "scale")] {
            r.push((role.to_string(), t(format!("{p}.hc_ffn_{x}"))));
        }
        let moe = Moe {
            router: Router { score: RouterScore::SqrtSoftplus, logit_scale: 1.0, bias: !hash[l], groups: None, top_k, renorm: true, scale, hash: hash[l] },
            experts: ne,
            expert: mlp,
            shared: Some(mlp),
            shared_gate: false,
            latent: None,
        };
        blocks.push(Block { norm_in: Norm::rms(eps), body: Body::Ffn(Ffn::Moe(moe)), norm_out: None, residual: Residual::Streams(streams) });
        roles.push(r);
    }
    for x in [
        "expert_dtype",
        "quantization_config",
        "rope_scaling",
        "num_hash_layers",
        "compress_ratios",
        "rope_theta",
        "compress_rope_theta",
        "layer_types",
        "mlp_layer_types",
        "compress_rates",
        "rope_parameters",
    ] {
        k.raw(x);
    }
    k.finish()?;
    Ok(Described {
        desc: ModelDesc {
            hidden: d,
            vocab,
            embed: Embed { scale: 1.0, norm: None, streams: streams.streams },
            blocks,
            final_norm: Norm::rms(eps),
            head: Head { tied, softcap: None, scale: 1.0, collapse: Collapse::Weighted { eps: streams.eps, norm_eps: eps } },
            mtp: Vec::new(),
            vision: None,
            rope: None,
        },
        embed: names[0].to_string(),
        final_norm: names[1].to_string(),
        head: (!tied).then(|| names[2].to_string()),
        blocks: roles,
        extra: ["fn", "base", "scale"].iter().map(|x| (format!("out_hc_{x}"), t(format!("{}{x}", names[3])))).collect(),
        mtp_layers,
        scales,
        shapes: Default::default(),
    })
}

/// Nemotron 3 (`nemotron_h`, `NemotronHForCausalLM`): a layer is one block, `x + mixer(norm(x))`, its mixer named by the pattern:
/// Mamba-2 (`M`), attention without any position embedding (`*`), or a mixture of experts (`E`) whose routed experts run in a latent
/// (two projections around them; none where `moe_latent_size` is null) with a shared expert on the full vector. The router is
/// DeepSeek-V3's: sigmoid scores, a bias for the choice, the chosen scores normed and times `routed_scaling_factor`. The pattern is
/// `hybrid_override_pattern`, or a list of types (`layers_block_type`). The draft head (`mtp.*`) is counted and not described yet.
fn nemotron_h(tc: &Map<String, Value>) -> Result<Described> {
    let k = Keys::new(tc, "nemotron_h");
    let (d, vocab) = (k.usize("hidden_size")?, k.usize("vocab_size")?);
    let tied = k.bool("tie_word_embeddings", false)?;
    // (one eps for the layers' norms, the final norm and the Mamba-2 gated norm; a config with two that differ does not say which is whose)
    let eps = match (k.opt_f64("layer_norm_epsilon")?, k.opt_f64("norm_eps")?) {
        (Some(a), Some(b)) if a != b => return err(format!("nemotron_h: layer_norm_epsilon {a} and norm_eps {b} differ")),
        (a, b) => a.or(b).unwrap_or(1e-5) as f32,
    };
    let kinds: Vec<char> = if let Some(p) = k.str("hybrid_override_pattern")? {
        p.chars().collect()
    } else if let Some(a) = k.raw("layers_block_type").or(k.raw("layer_types")).and_then(|v| v.as_array()) {
        a.iter()
            .map(|x| match x.as_str() {
                Some("mamba" | "linear_attention") => Ok('M'),
                Some("attention" | "full_attention") => Ok('*'),
                Some("moe") => Ok('E'),
                other => err(format!("nemotron_h: a layer of type {other:?} is not supported (mamba, attention, moe)")),
            })
            .collect::<Result<_>>()?
    } else {
        return err("nemotron_h: no hybrid_override_pattern");
    };
    (k.raw("layers_block_type"), k.raw("layer_types"));
    if let Some(c) = kinds.iter().find(|c| !"M*E".contains(**c)) {
        return err(format!("nemotron_h: a layer {c:?} in hybrid_override_pattern is not supported (M, * and E)"));
    }
    if kinds.is_empty() || k.opt_usize("num_hidden_layers")?.is_some_and(|n| n != kinds.len()) {
        return err(format!("nemotron_h: {} layers in the pattern, num_hidden_layers {:?}", kinds.len(), tc.get("num_hidden_layers")));
    }
    // settings the description does not take otherwise
    for (key, want) in [
        ("use_bias", Value::Bool(false)),
        ("mlp_bias", Value::Bool(false)),
        ("attention_bias", Value::Bool(false)),
        ("mamba_proj_bias", Value::Bool(false)),
        ("use_conv_bias", Value::Bool(true)),
        ("residual_in_fp32", Value::Bool(false)),
        ("mamba_hidden_act", Value::from("silu")),
        ("mamba_ssm_cache_dtype", Value::from("float32")),
    ] {
        k.expect(key, &want)?;
    }
    // (when the shared experts' work overlaps the routed experts' exchange between GPUs: a schedule, not arithmetic; the releases say true, the transformers
    // default; either is read)
    k.bool("moe_shared_expert_overlap", true)?;
    if k.raw("sliding_window").is_some() {
        return err("nemotron_h: a sliding window is not supported");
    }
    // (read and without effect: the attention has no position embedding; the time steps and the rescaling are how the weights were
    // first set; the kernels and the logits kept are transformers' own choices; no layer of the pattern is a plain MLP)
    for x in [
        "rope_theta",
        "partial_rotary_factor",
        "time_step_floor",
        "time_step_max",
        "time_step_min",
        "time_step_rank",
        "time_step_scale",
        "time_step_limit",
        "rescale_prenorm_residual",
        "use_mamba_kernels",
        "num_logits_to_keep",
        "intermediate_size",
        "mtp_hybrid_override_pattern",
    ] {
        k.raw(x);
    }
    let has = |c: char| kinds.contains(&c);
    let mamba = if has('M') {
        let (heads, head_dim, groups) = (k.usize("mamba_num_heads")?, k.usize("mamba_head_dim")?, k.usize("n_groups")?);
        if heads == 0 || groups == 0 || !heads.is_multiple_of(groups) {
            return err(format!("nemotron_h: {heads} Mamba-2 heads in {groups} groups"));
        }
        // (the inner width is the heads' channels; `expand` is not what sets it: the toy model of the GPU check has 64 heads of 64 on a width of 256)
        k.raw("expand");
        Some(Mamba2 {
            heads,
            head_dim,
            state: k.usize("ssm_state_size")?,
            groups,
            conv: k.usize("conv_kernel")?,
            chunk: k.opt_usize("chunk_size")?.unwrap_or(128),
            norm_eps: eps,
        })
    } else {
        for x in ["mamba_num_heads", "mamba_head_dim", "n_groups", "expand", "ssm_state_size", "conv_kernel", "chunk_size"] {
            k.raw(x);
        }
        None
    };
    let attn = {
        let (heads, kv) = (k.usize("num_attention_heads")?, k.usize("num_key_value_heads")?);
        if heads == 0 || kv == 0 || !heads.is_multiple_of(kv) {
            return err(format!("nemotron_h: {heads} query heads to {kv} KV heads"));
        }
        let hd = k.opt_usize("head_dim")?.unwrap_or(d / heads);
        Attn {
            heads,
            kv_heads: kv,
            head_dim: hd,
            v_head_dim: hd,
            window: None,
            sinks: false,
            rope: None,
            q_norm: None,
            k_norm: None,
            v_norm: None,
            gate: None,
            softcap: None,
            bias: false,
            scale: (hd as f32).powf(-0.5),
            k_eq_v: false,
            indexer: None,
            rounding: AttnRounding::Flash,
        }
    };
    let moe_keys =
        ["n_routed_experts", "num_experts_per_tok", "moe_intermediate_size", "moe_shared_expert_intermediate_size", "moe_latent_size", "mlp_hidden_act"];
    let moe = if has('E') {
        let (experts, top_k) = (k.usize("n_routed_experts")?, k.usize("num_experts_per_tok")?);
        if top_k == 0 || top_k > experts {
            return err(format!("nemotron_h: {top_k} experts a token of {experts}"));
        }
        for g in ["n_group", "topk_group", "n_shared_experts"] {
            k.expect(g, &Value::from(1))?;
        }
        (k.usize("moe_intermediate_size")?, k.usize("moe_shared_expert_intermediate_size")?);
        let mlp = Mlp { act: act(k.str("mlp_hidden_act")?.unwrap_or("relu2"))?, gated: false, bias: false, clamp: None };
        Some(Moe {
            router: Router {
                score: RouterScore::Sigmoid,
                logit_scale: 1.0,
                bias: true,
                groups: None,
                top_k,
                renorm: k.bool("norm_topk_prob", true)?,
                scale: k.opt_f64("routed_scaling_factor")?.unwrap_or(1.0) as f32,
                hash: false,
            },
            experts,
            expert: mlp,
            shared: Some(mlp),
            shared_gate: false,
            latent: k.opt_usize("moe_latent_size")?.filter(|&l| l > 0).map(|dim| Latent { dim, norm_in: None, expert_norm: None, norm_out: None }),
        })
    } else {
        for x in moe_keys.iter().chain(&["n_group", "topk_group", "n_shared_experts", "norm_topk_prob", "routed_scaling_factor"]) {
            k.raw(x);
        }
        None
    };
    let mtp_layers = k.opt_usize("num_nextn_predict_layers")?.unwrap_or(0);
    let n = Norm::rms(eps);
    let (mut blocks, mut roles) = (Vec::new(), Vec::new());
    for (l, c) in kinds.iter().enumerate() {
        let m = format!("backbone.layers.{l}.mixer");
        let mut r = vec![("norm_in".to_string(), t(format!("backbone.layers.{l}.norm.weight")))];
        let mut put = |role: &str, x: &str| r.push((role.to_string(), t(format!("{m}.{x}"))));
        let body = match (c, &mamba, &moe) {
            ('M', Some(mb), _) => {
                for (role, x) in [
                    ("in", "in_proj.weight"),
                    ("conv", "conv1d.weight"),
                    ("conv_bias", "conv1d.bias"),
                    ("a_log", "A_log"),
                    ("d", "D"),
                    ("dt_bias", "dt_bias"),
                    ("norm", "norm.weight"),
                    ("o", "out_proj.weight"),
                ] {
                    put(role, x);
                }
                Body::Mixer(Mixer::Mamba2(mb.clone()))
            }
            ('E', _, Some(me)) => {
                put("router", "gate.weight");
                put("choice_bias", "gate.e_score_correction_bias");
                if me.latent.is_some() {
                    put("latent_down", "fc1_latent_proj.weight");
                    put("latent_up", "fc2_latent_proj.weight");
                }
                put("shared_up", "shared_experts.up_proj.weight");
                put("shared_down", "shared_experts.down_proj.weight");
                for x in ["up", "down"] {
                    r.push((format!("experts_{x}"), Source::Experts((0..me.experts).map(|e| vec![format!("{m}.experts.{e}.{x}_proj.weight")]).collect())));
                }
                Body::Ffn(Ffn::Moe(me.clone()))
            }
            _ => {
                for x in ["q", "k", "v", "o"] {
                    put(x, &format!("{x}_proj.weight"));
                }
                Body::Mixer(Mixer::Attention(attn.clone()))
            }
        };
        blocks.push(Block { norm_in: n, body, norm_out: None, residual: Residual::Add });
        roles.push(r);
    }
    k.finish()?;
    Ok(Described {
        desc: ModelDesc {
            hidden: d,
            vocab,
            embed: Embed { scale: 1.0, norm: None, streams: 1 },
            blocks,
            final_norm: n,
            head: Head { tied, softcap: None, scale: 1.0, collapse: Collapse::None },
            mtp: Vec::new(),
            vision: None,
            rope: None,
        },
        embed: "backbone.embeddings.weight".into(),
        final_norm: "backbone.norm_f.weight".into(),
        head: (!tied).then(|| "lm_head.weight".to_string()),
        blocks: roles,
        extra: Vec::new(),
        mtp_layers,
        scales: Default::default(),
        shapes: Default::default(),
    })
}
