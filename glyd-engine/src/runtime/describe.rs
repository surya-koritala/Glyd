//! Model descriptions from `config.json` (R3a, #136): a `ModelDesc` of layer kinds and the weight map that names each block role's
//! checkpoint tensors, with no per-model code. Families: Qwen3.8 dense and MoE (the Qwen3.5 architecture), gemma-4 text, and the
//! Llama-style standard architectures (Llama, Mistral, Qwen2, Qwen3).
//!
//! A config key the builder does not know fails with the key named; keys that cannot change the math (ids, dtypes, training settings)
//! are listed in `INERT`. A decoder layer is two blocks: the mixer's, then the feed-forward's.

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
    /// Draft layers in the checkpoint that the description does not use yet (`mtp_num_hidden_layers`).
    pub mtp_layers: usize,
}

impl Described {
    /// Every checkpoint tensor the description reads.
    pub fn names(&self) -> BTreeSet<String> {
        let mut s: BTreeSet<String> = [self.embed.clone(), self.final_norm.clone()].into_iter().chain(self.head.clone()).collect();
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
];

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
        match self.m.keys().find(|k| !used.contains(*k) && !INERT.contains(&k.as_str())) {
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
    let mt = tc.get("model_type").and_then(|v| v.as_str()).unwrap_or("");
    // (a multimodal config's tie flag can sit at the top)
    let tied_top = cfg.get("tie_word_embeddings").and_then(|v| v.as_bool());
    match mt {
        "llama" | "mistral" | "qwen2" | "qwen3" => llama(tc, prefix, mt, tied_top),
        "qwen3_5_text" | "qwen3_5_moe_text" | "qwen3_5" | "qwen3_5_moe" => qwen35(tc, prefix, mt, tied_top),
        "gemma4_text" | "gemma4_unified_text" => gemma4(tc, prefix, mt, tied_top),
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
        mtp_layers,
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
        embed: Embed { scale: 1.0 },
        blocks,
        final_norm: Norm::rms(eps),
        head: Head { tied, softcap: None, scale: 1.0 },
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
        k.usize("intermediate_size")?;
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
        embed: Embed { scale: 1.0 },
        blocks,
        final_norm: pn,
        head: Head { tied, softcap: None, scale: 1.0 },
        mtp: Vec::new(),
        vision: None,
        rope: Some(rp),
    };
    finish(&k, desc, prefix, roles, mtp_layers)
}

fn gemma4(tc: &Map<String, Value>, prefix: &str, mt: &str, tied_top: Option<bool>) -> Result<Described> {
    let k = Keys::new(tc, mt);
    let (d, layers, vocab, eps, tied) = common(&k, tied_top)?;
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
    k.usize("intermediate_size")?;
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
        embed: Embed { scale: (d as f32).sqrt() },
        blocks,
        final_norm: n,
        head: Head { tied, softcap, scale: 1.0 },
        mtp: Vec::new(),
        vision: None,
        rope: None,
    };
    finish(&k, desc, prefix, roles, 0)
}
