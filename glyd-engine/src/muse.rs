//! Muse Spark (`MuseSparkForConditionalGeneration`: Muse-Spark-Open-816B-A42B, 2026-09) as the engine reads it, its text model: layers of attention over a sliding window (with the rotary) or over every
//! position (without), queries and keys normed per head, the output normed per head and gated, then a mixture of experts working in a latent (a projection in, a norm, the experts, a norm of each expert's
//! output, the weighted sum, a projection out), each block's output let into the residual stream by a learned gate. All bf16 but the routers (fp32). The reference: transformers'
//! modeling_muse_spark.py (the arithmetic and where it rounds; src/muse_ref.rs is ours).

use crate::ffi::{err, Error, Result};
use crate::weights::{eos_ids, num};
use glyd_gpu::json::{self, Value};
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

pub fn is_muse(text: &str) -> bool {
    json::parse(text).ok().and_then(|v| v.get("architectures").and_then(Value::as_array).and_then(|a| a.first().and_then(Value::as_str).map(|s| s == "MuseSparkForConditionalGeneration"))).unwrap_or(false)
}

impl ConfigMuse {
    pub fn load(dir: &Path) -> Result<ConfigMuse> {
        let p = dir.join("config.json");
        let text = std::fs::read_to_string(&p).map_err(|e| Error(format!("{}: {e}", p.display())))?;
        let generation = std::fs::read_to_string(dir.join("generation_config.json")).ok();
        ConfigMuse::parse(&text, &p.display().to_string(), generation.as_deref())
    }

    pub fn parse(text: &str, label: &str, generation: Option<&str>) -> Result<ConfigMuse> {
        let v = json::parse(text).map_err(|e| Error(format!("{label}: {e}")))?;
        if !is_muse(text) {
            return err(format!("{label}: not a Muse Spark model"));
        }
        let t = v.get("text_config").ok_or_else(|| Error(format!("{label}: no text_config")))?;
        let get = |k: &str| t.get(k).ok_or_else(|| Error(format!("{label}: no {k}")));
        let u = |k: &str| -> Result<usize> { get(k)?.as_u64().map(|x| x as usize).ok_or_else(|| Error(format!("{label}: {k}: not an integer"))) };
        let f = |k: &str| -> Result<f64> { num(get(k)?, k) };
        let f_or = |k: &str, d: f64| -> Result<f64> { if t.get(k).is_some_and(|x| *x != Value::Null) { f(k) } else { Ok(d) } };
        let refuse = |what: &str| err(format!("{label}: {what}, which the engine does not run"));
        if t.get("hidden_act").and_then(Value::as_str).is_some_and(|a| a != "silu") {
            return refuse("an activation other than silu");
        }
        if t.get("attention_bias").and_then(Value::as_bool) == Some(true) {
            return refuse("biases in the attention");
        }
        if u("num_shared_experts").unwrap_or(0) != 0 {
            return refuse("shared experts");
        }
        let layers = u("num_hidden_layers")?;
        let strs = |k: &str| -> Result<Vec<String>> { Ok(get(k)?.as_array().ok_or_else(|| Error(format!("{label}: {k}: not a list")))?.iter().map(|x| x.as_str().unwrap_or("").to_string()).collect()) };
        let types = strs("layer_types")?;
        let mlp = strs("mlp_layer_types")?;
        if types.len() != layers || mlp.len() != layers || mlp.iter().any(|m| m != "sparse") {
            return refuse("layer types other than one per layer, every MLP sparse");
        }
        let thetas: Vec<f64> = get("layer_rope_theta")?.as_array().ok_or_else(|| Error(format!("{label}: layer_rope_theta: not a list")))?.iter().map(|x| num(x, "layer_rope_theta")).collect::<Result<_>>()?;
        let sliding: Vec<bool> = types.iter().map(|s| s == "sliding_attention").collect();
        let theta = thetas.iter().copied().find(|&x| x > 0.0).unwrap_or(0.0);
        // (the rotary is on the window's layers and off on the others, at one base)
        if thetas.len() != layers || sliding.iter().zip(&thetas).any(|(&s, &th)| s != (th > 0.0) || (th > 0.0 && th != theta)) || types.iter().any(|s| s != "sliding_attention" && s != "full_attention") {
            return refuse("a rotary that is not on exactly the window's layers, at one base");
        }
        if t.get("rope_parameters").and_then(|r| r.get("rope_type")).and_then(Value::as_str).is_some_and(|r| r != "default") {
            return refuse("a rotary scaling");
        }
        let (heads, kv_heads, head_dim) = (u("num_attention_heads")?, u("num_key_value_heads")?, u("head_dim")?);
        if kv_heads == 0 || heads % kv_heads != 0 {
            return err(format!("{label}: {heads} heads over {kv_heads} key-value heads"));
        }
        let (latent, hidden) = (u("moe_hidden_size")?, u("hidden_size")?);
        let mut eos = eos_ids(t, generation);
        if eos.is_empty() {
            eos = t.get("eos_token_id").and_then(Value::as_i64).into_iter().collect();
        }
        Ok(ConfigMuse {
            hidden,
            layers,
            vocab: u("vocab_size")?,
            heads,
            kv_heads,
            head_dim,
            window: u("sliding_window")?,
            sliding,
            rope_theta: theta,
            experts: u("num_local_experts")?,
            top_k: u("num_experts_per_tok")?,
            latent,
            inter: u("moe_intermediate_size")?,
            eps: f_or("rms_norm_eps", 1e-5)? as f32,
            post_eps: f_or("post_norm_eps", 1e-8)? as f32,
            qk_scale: f_or("qk_scale_factor", 1.0)? as f32,
            out_mult: f_or("output_multiplier", 1.0)? as f32,
            softcap: f_or("final_logit_softcapping", 0.0)? as f32,
            gate_temp: f_or("residual_gate_temperature", 1.0)? as f32,
            route_scale: f_or("routed_scaling_factor", 1.0)? as f32,
            max_pos: u("max_position_embeddings")?,
            eos,
            bos: t.get("bos_token_id").and_then(Value::as_u64).map(|b| b as u32),
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
