//! Muse Spark (`MuseSparkForConditionalGeneration`: Muse-Spark-Open-816B-A42B, 2026-09) as the engine reads it, its text model: layers of attention over a sliding window (with the rotary) or over every
//! position (without), queries and keys normed per head, the output normed per head and gated, then a mixture of experts working in a latent (a projection in, a norm, the experts, a norm of each expert's
//! output, the weighted sum, a projection out), each block's output let into the residual stream by a learned gate. All bf16 but the routers (fp32). The reference: transformers'
//! modeling_muse_spark.py (the arithmetic and where it rounds; src/muse_ref.rs is ours).

use crate::ffi::{err, Error, Result};
use crate::runtime::config::ModelConfig;
use crate::weights::eos_ids;
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

/// Muse Spark's own keys of its text model (the shared ones are `runtime::config::Shared`'s), typed from the one parse of its config.json.
#[derive(Default, serde::Deserialize)]
#[serde(default)]
struct Keys {
    hidden_act: Option<String>,
    attention_bias: Option<bool>,
    num_shared_experts: Option<usize>,
    mlp_layer_types: Option<Vec<String>>,
    layer_rope_theta: Option<Vec<f64>>,
    rope_parameters: Option<RopeKeys>,
    rms_norm_eps: Option<f64>,
    post_norm_eps: Option<f64>,
    qk_scale_factor: Option<f64>,
    output_multiplier: Option<f64>,
    final_logit_softcapping: Option<f64>,
    residual_gate_temperature: Option<f64>,
    routed_scaling_factor: Option<f64>,
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
        let or = |v: Option<f64>, d: f64| v.unwrap_or(d) as f32;
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
            eps: or(k.rms_norm_eps, 1e-5),
            post_eps: or(k.post_norm_eps, 1e-8),
            qk_scale: or(k.qk_scale_factor, 1.0),
            out_mult: or(k.output_multiplier, 1.0),
            softcap: or(k.final_logit_softcapping, 0.0),
            gate_temp: or(k.residual_gate_temperature, 1.0),
            route_scale: or(k.routed_scaling_factor, 1.0),
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

/// A residual gate's two factors for each channel, from its parameter (bf16), in transformers' bf16 arithmetic (each step rounded): w = g / T, beta = sigmoid(w), alpha = sqrt(max(sigmoid(-w) (1 + beta),
/// 1e-3)); the block's output becomes alpha residual + beta output.
pub fn residual_gate(g: &[f32], temp: f32) -> (Vec<f32>, Vec<f32>) {
    let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
    g.iter()
        .map(|&g| {
            let w = bf(g / temp);
            let beta = bf(sig(w));
            let prod = bf(bf(sig(-w)) * bf(1.0 + beta));
            let alpha = bf(prod.max(bf(1e-3)).sqrt());
            (alpha, beta)
        })
        .unzip()
}

/// A centered norm's factors: bf16(1 + w) for each channel, as transformers adds them (a bf16 tensor plus 1).
pub fn plus_one(w: &[f32]) -> Vec<f32> {
    w.iter().map(|&x| bf(1.0 + x)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_residual_gate_of_zero_lets_half_of_each_through() {
        // (g = 0: beta = 0.5, alpha = sqrt(0.5 x 1.5) = 0.866)
        let (a, b) = residual_gate(&[0.0], 0.3);
        assert_eq!(b[0], 0.5);
        assert_eq!(a[0], bf(0.75f32.sqrt()));
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
}
