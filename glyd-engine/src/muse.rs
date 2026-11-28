//! Muse Spark (`MuseSparkForConditionalGeneration`: Muse-Spark-Open-816B-A42B, 2026-09) as the engine reads it, its text model: layers of attention over a sliding window (with the rotary) or over every
//! position (without), queries and keys normed per head, the output normed per head and gated, then a mixture of experts working in a latent (a projection in, a norm, the experts, a norm of each expert's
//! output, the weighted sum, a projection out), each block's output let into the residual stream by a learned gate. All bf16 but the routers and the gates' factors (transformers keeps those modules fp32).
//! The reference: transformers' modeling_muse_spark.py (the arithmetic and where it rounds; src/muse_ref.rs is ours).

use crate::ffi::{err, Error, Result};
use crate::runtime::config::ModelConfig;
use crate::weights::eos_ids;
use serde_json::{Map, Value};
use std::path::Path;

#[derive(Clone, Debug)]
pub struct ConfigMuse {
    pub hidden: usize,
    pub layers: usize,
    pub vocab: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub window: usize,
    /// each layer: attention over the window with the rotary (true), or over every position without
    pub sliding: Vec<bool>,
    pub rope_theta: f64,
    pub experts: usize,
    pub top_k: usize,
    /// the experts' latent width and their intermediate size
    pub latent: usize,
    pub inter: usize,
    pub eps: f32,
    pub post_eps: f32,
    pub qk_scale: f32,
    pub out_mult: f32,
    pub softcap: f32,
    pub gate_temp: f32,
    pub route_scale: f32,
    pub max_pos: usize,
    pub eos: Vec<i64>,
    pub bos: Option<u32>,
    /// the tensors' prefix (`model.language_model.`)
    pub prefix: String,
}

/// Muse Spark's own keys of its text model (the shared ones are `runtime::config::Shared`'s), typed from the one parse of its config.json; its numbers are
/// `number`'s.
#[derive(Default, serde::Deserialize)]
#[serde(default)]
struct Keys {
    hidden_act: Option<String>,
    attention_bias: Option<bool>,
    num_shared_experts: Option<usize>,
    mlp_layer_types: Option<Vec<String>>,
    layer_rope_theta: Option<Vec<f64>>,
    rope_parameters: Option<RopeKeys>,
    bos_token_id: Option<serde_json::Value>,
    eos_token_id: Option<serde_json::Value>,
}

/// A rotary's parameters as a config gives them: its type.
#[derive(Default, serde::Deserialize)]
#[serde(default)]
struct RopeKeys {
    rope_type: Option<String>,
}

pub fn is_muse(text: &str) -> bool {
    crate::runtime::config::family_of(text) == Some(crate::support::Family::Muse)
}

/// Muse Spark's own numbers and their defaults where its config leaves one out, transformers' (configuration_muse_spark.py, `MuseSparkTextConfig`):
/// `rms_norm_eps: float = 1e-5`, `post_norm_eps: float = 1e-8`, `qk_scale_factor: float = 5.4730064863875`, `output_multiplier: float = 0.1767766952966369`,
/// `final_logit_softcapping: float = 20.0`, `residual_gate_temperature: float = 0.3`, `routed_scaling_factor: float = 1.0`. The description
/// (`runtime::describe`) and this reader read them by `number`, one reading.
pub const NUMBERS: [(&str, f64); 7] = [
    ("rms_norm_eps", 1e-5),
    ("post_norm_eps", 1e-8),
    ("qk_scale_factor", 5.4730064863875),
    ("output_multiplier", 0.1767766952966369),
    ("final_logit_softcapping", 20.0),
    ("residual_gate_temperature", 0.3),
    ("routed_scaling_factor", 1.0),
];

/// Number `key` of Muse Spark's text config `text` (one of `NUMBERS`): its default where the key is absent; refused in words where it is null or not a number
/// (transformers' typed config refuses those: the field is a float), and the soft-cap or the gates' temperature where not above 0 (each divides).
pub fn number(text: &Map<String, Value>, key: &str, label: &str) -> Result<f32> {
    let default = NUMBERS.iter().find(|n| n.0 == key).map(|n| n.1).ok_or_else(|| Error(format!("{key}: not one of Muse Spark's numbers")))?;
    let v = match text.get(key) {
        None => default,
        Some(v) => v.as_f64().ok_or_else(|| Error(format!("{label}: {key} = {v}, where transformers reads a number (its default {default})")))?,
    };
    if matches!(key, "final_logit_softcapping" | "residual_gate_temperature") && !(v > 0.0 && v.is_finite()) {
        return err(format!("{label}: {key} = {v}, where transformers divides by it: a number above 0 (its default {default})"));
    }
    Ok(v as f32)
}

/// A Muse Spark config's `quantization_config` (the top level's, or its text model's) refused in words that name the format: the engine reads the release's
/// bf16 weights (its routers fp32), and a quantized checkpoint's experts (ModelOpt's NVFP4: codes, block scales and a global scale beside each matrix) would
/// otherwise be half read.
pub fn quantized(top: &Map<String, Value>, text: &Map<String, Value>, label: &str) -> Result<()> {
    let Some(q) = top.get("quantization_config").or(text.get("quantization_config")).filter(|q| !q.is_null()) else { return Ok(()) };
    let word = |k: &str| q.get(k).and_then(Value::as_str);
    let format = [word("quant_method"), word("quant_algo")].into_iter().flatten().collect::<Vec<_>>().join(" ");
    err(format!(
        "{label}: a Muse Spark checkpoint quantized as {}: this engine reads Muse Spark's bf16 release (its quantization_config is not read)",
        if format.is_empty() { q.to_string() } else { format }
    ))
}

impl ConfigMuse {
    pub fn load(dir: &Path) -> Result<ConfigMuse> {
        ModelConfig::read(dir, ConfigMuse::of)
    }

    pub fn parse(text: &str, label: &str, generation: Option<&str>) -> Result<ConfigMuse> {
        ConfigMuse::of(&ModelConfig::parse(text).map_err(|e| e.context(label))?, label, generation)
    }

    /// `parse` of a config.json parsed once (`ModelConfig`): the shared keys typed there (`Shared`), Muse Spark's own typed here (`Keys`).
    pub fn of(c: &ModelConfig, label: &str, generation: Option<&str>) -> Result<ConfigMuse> {
        if c.family() != Some(crate::support::Family::Muse) {
            return err(format!("{label}: not a Muse Spark model"));
        }
        if !c.nested {
            return err(format!("{label}: no text_config"));
        }
        quantized(&c.top, &c.text, label)?;
        let (s, k): (&crate::runtime::config::Shared, Keys) = (&c.shared, c.own(label)?);
        let need = |v: Option<usize>, key: &str| ModelConfig::need(v, label, key);
        let refuse = |what: &str| err(format!("{label}: {what}, which the engine does not run"));
        if k.hidden_act.as_deref().is_some_and(|a| a != "silu") {
            return refuse("an activation other than silu");
        }
        if k.attention_bias == Some(true) {
            return refuse("biases in the attention");
        }
        if k.num_shared_experts.unwrap_or(0) != 0 {
            return refuse("shared experts");
        }
        let layers = need(s.num_hidden_layers, "num_hidden_layers")?;
        if layers > crate::plan::MAX_LAYERS {
            return err(format!("{label}: num_hidden_layers {layers}, where no model has more than {}", crate::plan::MAX_LAYERS));
        }
        let types = s.layer_types.clone().ok_or_else(|| Error(format!("{label}: no layer_types")))?;
        let mlp = k.mlp_layer_types.ok_or_else(|| Error(format!("{label}: no mlp_layer_types")))?;
        if types.len() != layers || mlp.len() != layers || mlp.iter().any(|m| m != "sparse") {
            return refuse("layer types other than one per layer, every MLP sparse");
        }
        let thetas = k.layer_rope_theta.ok_or_else(|| Error(format!("{label}: no layer_rope_theta")))?;
        let sliding: Vec<bool> = types.iter().map(|s| s == "sliding_attention").collect();
        let theta = thetas.iter().copied().find(|&x| x > 0.0).unwrap_or(0.0);
        // (the rotary is on the window's layers and off on the others, at one base)
        if thetas.len() != layers
            || sliding.iter().zip(&thetas).any(|(&s, &th)| s != (th > 0.0) || (th > 0.0 && th != theta))
            || types.iter().any(|s| s != "sliding_attention" && s != "full_attention")
        {
            return refuse("a rotary that is not on exactly the window's layers, at one base");
        }
        if k.rope_parameters.and_then(|r| r.rope_type).is_some_and(|r| r != "default") {
            return refuse("a rotary scaling");
        }
        let (heads, kv_heads, head_dim) =
            (need(s.num_attention_heads, "num_attention_heads")?, need(s.num_key_value_heads, "num_key_value_heads")?, need(s.head_dim, "head_dim")?);
        if kv_heads == 0 || heads % kv_heads != 0 {
            return err(format!("{label}: {heads} heads over {kv_heads} key-value heads"));
        }
        let (latent, hidden) = (need(s.moe_hidden_size, "moe_hidden_size")?, need(s.hidden_size, "hidden_size")?);
        let mut eos = eos_ids(&c.text, generation);
        if eos.is_empty() {
            eos = k.eos_token_id.as_ref().and_then(serde_json::Value::as_i64).into_iter().collect();
        }
        let num = |key: &str| number(&c.text, key, label);
        Ok(ConfigMuse {
            hidden,
            layers,
            vocab: need(s.vocab_size, "vocab_size")?,
            heads,
            kv_heads,
            head_dim,
            window: need(s.sliding_window, "sliding_window")?,
            sliding,
            rope_theta: theta,
            experts: need(s.num_local_experts, "num_local_experts")?,
            top_k: need(s.num_experts_per_tok, "num_experts_per_tok")?,
            latent,
            inter: need(s.moe_intermediate_size, "moe_intermediate_size")?,
            eps: num("rms_norm_eps")?,
            post_eps: num("post_norm_eps")?,
            qk_scale: num("qk_scale_factor")?,
            out_mult: num("output_multiplier")?,
            softcap: num("final_logit_softcapping")?,
            gate_temp: num("residual_gate_temperature")?,
            route_scale: num("routed_scaling_factor")?,
            max_pos: need(s.max_position_embeddings, "max_position_embeddings")?,
            eos,
            bos: k.bos_token_id.as_ref().and_then(serde_json::Value::as_u64).map(|b| b as u32),
            prefix: "model.language_model.".to_string(),
        })
    }

    /// The rotary's frequencies (transformers' default: base^(-2i / head_dim) in fp32).
    pub fn freqs(&self) -> Vec<f32> {
        (0..self.head_dim / 2).map(|i| 1.0 / (self.rope_theta as f32).powf((2 * i) as f32 / self.head_dim as f32)).collect()
    }
}

/// `x` rounded to bf16 (to nearest, ties to even), as an f32.
pub fn bf(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let b = x.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

/// The residual gate's add of one value, as transformers' `MuseSparkResidualGate` makes it: its factors alpha and beta fp32 (the gate is an fp32 module,
/// `_keep_in_fp32_modules_strict`; `runtime::reference::residual_gate` makes them), the residual x and the block's output h bf16 values, the products and
/// their sum in fp32, rounded to bf16 once (`(alpha * residual + beta * hidden_states).to(residual.dtype)`). kernels/residual.cu `res_gate` is its GPU form.
pub fn gated(x: f32, h: f32, alpha: f32, beta: f32) -> f32 {
    bf(alpha * x + beta * h)
}

/// A centered norm's factors: bf16(1 + w) for each channel, as transformers adds them (a bf16 tensor plus 1).
pub fn plus_one(w: &[f32]) -> Vec<f32> {
    w.iter().map(|&x| bf(1.0 + x)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The residual gate as transformers' fp32 module makes it (`MuseSparkResidualGate`): its factors fp32, none rounded to bf16, its add of bf16 values
    /// rounded once. g = 0 lets half of the output through (beta 0.5, alpha sqrt(0.5 x 1.5)); at g = 0.7 and T = 0.3 alpha's bf16 rounding would move it by
    /// 2.2e-3 of itself, and the add of x = 2.109375 and h = -1.9609375 made with bf16 factors and three roundings is a bf16 step from transformers' -0.921875.
    #[test]
    fn the_residual_gate_is_transformers_fp32_one_rounding() {
        let (a, b) = crate::runtime::reference::residual_gate(&[0.0, 0.7], 0.3);
        assert_eq!((a[0], b[0]), (0.75f32.sqrt(), 0.5));
        assert!(a[1] != bf(a[1]) && b[1] != bf(b[1]), "fp32 factors: {} {}", a[1], b[1]);
        let (x, h) = (2.109375, -1.9609375);
        assert_eq!(gated(x, h, a[1], b[1]), -0.921875);
        assert_ne!(bf(bf(bf(a[1]) * x) + bf(bf(b[1]) * h)), -0.921875, "bf16 factors and three roundings");
        assert_eq!(plus_one(&[0.0, 1.0]), vec![1.0, 2.0]);
    }
}

// (the random model's config is the development build's: `muse_ref`)
#[cfg(all(test, feature = "dev"))]
mod reader_tests {
    use super::*;

    /// Muse Spark's reader takes its keys from the one parse of config.json, typed: the random model's config reads; a key of another type than its field's is
    /// refused with its name, and a shared key the model needs, missing or null, is named as missing.
    #[test]
    fn the_keys_are_read_typed_from_the_one_parse() {
        let c = ConfigMuse::parse(crate::muse_ref::RANDOM_CONFIG, "random", None).unwrap();
        assert_eq!((c.hidden, c.layers, c.sliding.len(), c.prefix.as_str()), (1024, 4, 4, "model.language_model."));
        let mut v: serde_json::Value = serde_json::from_str(crate::muse_ref::RANDOM_CONFIG).unwrap();
        v["text_config"]["rms_norm_eps"] = serde_json::json!("small");
        assert!(ConfigMuse::parse(&v.to_string(), "c", None).unwrap_err().msg.contains("rms_norm_eps"));
        let mut v: serde_json::Value = serde_json::from_str(crate::muse_ref::RANDOM_CONFIG).unwrap();
        v["text_config"]["hidden_size"] = serde_json::Value::Null;
        assert_eq!(ConfigMuse::parse(&v.to_string(), "c", None).unwrap_err().msg, "c: no hidden_size");
    }

    /// Muse Spark's numbers are read once, as transformers reads them (`NUMBERS`), by the description and by this reader alike: each absent key its default
    /// there (the soft-cap 20, and the output multiplier with it, rms_norm_eps 1e-5, the query scale 5.473, the gates' temperature 0.3, post_norm_eps 1e-8, the
    /// routers' scaling 1); a null or a string, and a soft-cap or a temperature of 0, refused by both with the key named; a quantization_config (the top
    /// level's: ModelOpt's NVFP4 release form) refused by both, its format named.
    #[test]
    fn the_numbers_are_transformers_and_read_once() {
        use crate::runtime::describe::describe;
        use crate::runtime::kinds::{Body, Ffn, GateP, Mixer, Residual};
        let base: serde_json::Value = serde_json::from_str(crate::muse_ref::RANDOM_CONFIG).unwrap();
        let mut v = base.clone();
        for (k, _) in NUMBERS {
            v["text_config"].as_object_mut().unwrap().remove(k);
        }
        let c = ConfigMuse::parse(&v.to_string(), "c", None).unwrap();
        assert_eq!((c.eps, c.post_eps, c.qk_scale, c.out_mult), (1e-5, 1e-8, 5.4730064863875, 0.1767766952966369));
        assert_eq!((c.softcap, c.gate_temp, c.route_scale), (20.0, 0.3, 1.0));
        let d = describe(&v).unwrap_or_else(|e| panic!("{}", e.msg));
        assert_eq!((d.desc.head.softcap, d.desc.head.scale, d.desc.final_norm.eps), (Some(c.softcap), c.out_mult, c.eps));
        let (att, moe) = (&d.desc.blocks[0], &d.desc.blocks[1]);
        let (Body::Mixer(Mixer::Attention(a)), Body::Ffn(Ffn::Moe(m))) = (&att.body, &moe.body) else { panic!("{:?}", att.body) };
        assert_eq!((att.norm_in.eps, att.norm_out.map(|n| n.eps), a.scale), (c.eps, Some(c.post_eps), c.qk_scale / 8.0));
        assert_eq!((m.router.logit_scale, m.router.scale, &moe.residual), (c.out_mult, c.route_scale, &Residual::Gated(GateP { temperature: c.gate_temp })));
        for (key, value) in [
            ("final_logit_softcapping", serde_json::Value::Null),
            ("final_logit_softcapping", serde_json::json!(0.0)),
            ("residual_gate_temperature", serde_json::json!(0)),
            ("qk_scale_factor", serde_json::Value::Null),
            ("rms_norm_eps", serde_json::json!("small")),
        ] {
            let mut v = base.clone();
            v["text_config"][key] = value.clone();
            let (r, d) = (ConfigMuse::parse(&v.to_string(), "c", None).unwrap_err().msg, describe(&v).unwrap_err().msg);
            assert!(r.contains(key) && d.contains(key), "{key} = {value}: {r} | {d}");
        }
        let mut v = base.clone();
        v["quantization_config"] = serde_json::json!({"quant_method": "modelopt", "quant_algo": "NVFP4", "group_size": 16});
        let (r, d) = (ConfigMuse::parse(&v.to_string(), "c", None).unwrap_err().msg, describe(&v).unwrap_err().msg);
        assert!(r.contains("quantized as modelopt NVFP4") && d.contains("quantized as modelopt NVFP4"), "{r} | {d}");
    }
}
