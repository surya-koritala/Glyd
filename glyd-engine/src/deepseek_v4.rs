//! DeepSeek-V4 (`DeepseekV4ForCausalLM`: V4-Flash, 2026-04) as the engine reads it: its config, its tensors' names, and the checkpoint's weights decoded exactly. The model: four residual streams mixed by learned
//! weights around each layer's two blocks (mHC, as GLM-5.3-Flash); the attention one shared key-value head of 512 (its last 64 channels rotary) over a window of the last 128 tokens and, in most layers, a compressed
//! cache: every `ratio` tokens pooled into one entry (ratio 4: overlapping windows, and an indexer picks the entries a query reads; ratio 128: all of them), a learned sink in the softmax, the query through a low rank
//! and normed per head, the output de-rotated and projected through a grouped low rank; then a mixture of experts (sqrt-softplus scores, the first layers routed by a token's id) with a shared expert. The checkpoint:
//! FP8 E4M3 with a power-of-two (E8M0) scale per 128 x 128 block for the dense matrices, FP4 E2M1 with an E8M0 scale per 32 weights along a row for the experts, bf16 or fp32 for the rest. Both decode exactly: an
//! E4M3 value times a power of two is a bf16, and an E2M1 value times two is an integer of -12 to 12, so a run of 32 is a five-bit run (`pack_int5_runs_checked`) with the power of two as its scale. The reference:
//! the release's inference/model.py and kernel.py (the arithmetic, and where it rounds; src/dsv4_ref.rs is ours).

use crate::ffi::{err, Error, Result};
use crate::weights::{eos_ids, num};
use glyd_gpu::json::{self, Value};
use std::path::Path;

#[derive(Clone, Debug)]
pub struct ConfigV4 {
    pub hidden: usize,
    pub layers: usize,
    pub vocab: usize,
    pub eps: f32,
    pub max_pos: usize,
    pub eos: Vec<i64>,
    /// the start token (bos_token_id), which a raw prompt begins with
    pub bos: Option<u32>,
    /// query heads, all of `head_dim` channels (the last `rope_dim` rotary); one key-value head of `head_dim`
    pub heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub q_lora: usize,
    /// the output's low rank: `o_groups` groups of heads, each to `o_lora` channels
    pub o_lora: usize,
    pub o_groups: usize,
    /// the window of keys every attention reads
    pub window: usize,
    /// each layer's compression: 0 (the window alone), 4 (overlapping pools and an indexer) or another (pools of that many, all read)
    pub ratios: Vec<usize>,
    pub idx_heads: usize,
    pub idx_dim: usize,
    pub idx_topk: usize,
    pub experts: usize,
    pub top_k: usize,
    pub inter: usize,
    /// the first layers route by the token's id (`tid2eid`)
    pub hash_layers: usize,
    pub route_scale: f32,
    pub swiglu_limit: f32,
    pub hc_mult: usize,
    pub hc_iters: usize,
    pub hc_eps: f32,
    /// the window-only layers' rotary base (no YaRN), and the compressing layers' (with YaRN)
    pub rope_theta: f64,
    pub compress_theta: f64,
    pub yarn: Yarn,
    pub mtp: usize,
}

/// YaRN's stretch of the rotary frequencies (model.py's precompute_freqs_cis).
#[derive(Clone, Copy, Debug)]
pub struct Yarn {
    pub factor: f64,
    pub orig: usize,
    pub beta_fast: f64,
    pub beta_slow: f64,
}

pub fn is_v4(text: &str) -> bool {
    json::parse(text).ok().and_then(|v| v.get("architectures").and_then(Value::as_array).and_then(|a| a.first().and_then(Value::as_str).map(|s| s == "DeepseekV4ForCausalLM"))).unwrap_or(false)
}

impl ConfigV4 {
    pub fn load(dir: &Path) -> Result<ConfigV4> {
        let p = dir.join("config.json");
        let text = std::fs::read_to_string(&p).map_err(|e| Error(format!("{}: {e}", p.display())))?;
        let generation = std::fs::read_to_string(dir.join("generation_config.json")).ok();
        ConfigV4::parse(&text, &p.display().to_string(), generation.as_deref())
    }

    pub fn parse(text: &str, label: &str, generation: Option<&str>) -> Result<ConfigV4> {
        let v = json::parse(text).map_err(|e| Error(format!("{label}: {e}")))?;
        let arch = v.get("architectures").and_then(Value::as_array).and_then(|a| a.first()).and_then(Value::as_str).unwrap_or("");
        if arch != "DeepseekV4ForCausalLM" {
            return err(format!("{label}: {arch}: not a DeepSeek-V4 model"));
        }
        let get = |k: &str| v.get(k).ok_or_else(|| Error(format!("{label}: no {k}")));
        let u = |k: &str| -> Result<usize> { get(k)?.as_u64().map(|x| x as usize).ok_or_else(|| Error(format!("{label}: {k}: not an integer"))) };
        let f = |k: &str| -> Result<f64> { num(get(k)?, k) };
        let u_or = |k: &str, d: usize| -> Result<usize> { if v.get(k).is_some_and(|x| *x != Value::Null) { u(k) } else { Ok(d) } };
        let f_or = |k: &str, d: f64| -> Result<f64> { if v.get(k).is_some_and(|x| *x != Value::Null) { f(k) } else { Ok(d) } };
        let text_is = |k: &str, want: &str| v.get(k).and_then(Value::as_str).is_none_or(|a| a == want);
        let refuse = |what: &str| err(format!("{label}: {what}, which the engine does not run"));
        if !text_is("hidden_act", "silu") {
            return refuse("an activation other than silu");
        }
        if v.get("attention_bias").and_then(Value::as_bool) == Some(true) {
            return refuse("biases in the attention");
        }
        if !text_is("scoring_func", "sqrtsoftplus") {
            return refuse("routing scores other than sqrtsoftplus");
        }
        if !text_is("topk_method", "noaux_tc") {
            return refuse("a selection other than noaux_tc");
        }
        if u_or("n_shared_experts", 1)? != 1 || u_or("num_key_value_heads", 1)? != 1 {
            return refuse("other than one shared expert and one key-value head");
        }
        if v.get("norm_topk_prob").and_then(Value::as_bool) == Some(false) {
            return refuse("routing weights not renormalized");
        }
        if v.get("tie_word_embeddings").and_then(Value::as_bool) == Some(true) {
            return refuse("a tied LM head");
        }
        let layers = u("num_hidden_layers")?;
        let ratios: Vec<usize> = match v.get("compress_ratios").and_then(Value::as_u64s) {
            Some(r) => r.iter().map(|&x| x as usize).collect(),
            None => return err(format!("{label}: no compress_ratios")),
        };
        // (the release lists one more ratio than layers: the multi-token prediction layer's, which reads the window alone)
        if ratios.len() < layers {
            return err(format!("{label}: {} compress_ratios for {layers} layers", ratios.len()));
        }
        let ratios = ratios[..layers].to_vec();
        let window = u_or("sliding_window", 128)?;
        if ratios.iter().any(|&r| r != 0 && (r < 2 || window % r != 0 && r % window != 0)) {
            return refuse(&format!("compress_ratios {ratios:?} with a window of {window}"));
        }
        if let Some(q) = v.get("quantization_config").filter(|q| **q != Value::Null) {
            let method = q.get("quant_method").and_then(Value::as_str).unwrap_or("");
            let fmt = q.get("fmt").and_then(Value::as_str).unwrap_or("e4m3");
            let sfmt = q.get("scale_fmt").and_then(Value::as_str).unwrap_or("");
            let block = q.get("weight_block_size").and_then(Value::as_u64s);
            if method != "fp8" || fmt != "e4m3" || sfmt != "ue8m0" || block.as_deref() != Some(&[128, 128]) {
                return refuse(&format!("a quantization of {method} {fmt}, scales {sfmt:?}, blocks {block:?} (the release's is fp8 e4m3, ue8m0 scales of 128 x 128 blocks)"));
            }
        }
        if v.get("expert_dtype").and_then(Value::as_str).is_some_and(|d| d != "fp4") {
            return refuse("experts other than FP4");
        }
        let rs = v.get("rope_scaling").filter(|r| **r != Value::Null);
        let rsf = |k: &str, d: f64| -> Result<f64> { rs.and_then(|r| r.get(k)).filter(|x| **x != Value::Null).map_or(Ok(d), |x| num(x, k)) };
        if rs.and_then(|r| r.get("type").or_else(|| r.get("rope_type"))).and_then(Value::as_str).is_some_and(|t| t != "yarn") {
            return refuse("a rotary scaling other than yarn");
        }
        let yarn = Yarn { factor: rsf("factor", 1.0)?, orig: rsf("original_max_position_embeddings", 0.0)? as usize, beta_fast: rsf("beta_fast", 32.0)?, beta_slow: rsf("beta_slow", 1.0)? };
        let (heads, head_dim, rope_dim) = (u("num_attention_heads")?, u("head_dim")?, u("qk_rope_head_dim")?);
        let (o_groups, idx_dim) = (u("o_groups")?, u("index_head_dim")?);
        if heads % o_groups != 0 || rope_dim % 2 != 0 || rope_dim > head_dim || rope_dim > idx_dim || head_dim % 64 != 0 || !idx_dim.is_power_of_two() {
            return err(format!("{label}: {heads} heads in {o_groups} groups, heads of {head_dim} with {rope_dim} rotary, indexer heads of {idx_dim}"));
        }
        let mut eos = eos_ids(&v, generation);
        eos.dedup();
        Ok(ConfigV4 {
            hidden: u("hidden_size")?,
            layers,
            vocab: u("vocab_size")?,
            eps: f_or("rms_norm_eps", 1e-6)? as f32,
            max_pos: u("max_position_embeddings")?,
            eos,
            bos: v.get("bos_token_id").and_then(Value::as_u64).map(|b| b as u32),
            heads,
            head_dim,
            rope_dim,
            q_lora: u("q_lora_rank")?,
            o_lora: u("o_lora_rank")?,
            o_groups,
            window,
            ratios,
            idx_heads: u("index_n_heads")?,
            idx_dim,
            idx_topk: u("index_topk")?,
            experts: u("n_routed_experts")?,
            top_k: u("num_experts_per_tok")?,
            inter: u("moe_intermediate_size")?,
            hash_layers: u_or("num_hash_layers", 0)?,
            route_scale: f_or("routed_scaling_factor", 1.0)? as f32,
            swiglu_limit: f_or("swiglu_limit", 0.0)? as f32,
            hc_mult: u("hc_mult")?,
            hc_iters: u_or("hc_sinkhorn_iters", 20)?,
            hc_eps: f_or("hc_eps", 1e-6)? as f32,
            rope_theta: f_or("rope_theta", 10000.0)?,
            compress_theta: f_or("compress_rope_theta", 160000.0)?,
            yarn,
            mtp: u_or("num_nextn_predict_layers", 0)?,
        })
    }

    /// The rotary frequencies of layer l (`rope_dim / 2` of them): a compressing layer's base with YaRN, a window-only layer's without.
    pub fn freqs(&self, l: usize) -> Vec<f32> {
        if self.ratios[l] != 0 {
            freqs(self.rope_dim, self.compress_theta, Some(self.yarn))
        } else {
            freqs(self.rope_dim, self.rope_theta, None)
        }
    }
}

/// model.py's precompute_freqs_cis's frequencies, in its arithmetic: the base's powers in fp32, the correction range in f64 (Python's math), the ramp and the blend in fp32.
pub fn freqs(dim: usize, base: f64, yarn: Option<Yarn>) -> Vec<f32> {
    let half = dim / 2;
    let mut f: Vec<f32> = (0..half).map(|i| 1.0 / (base as f32).powf((2 * i) as f32 / dim as f32)).collect();
    let Some(y) = yarn.filter(|y| y.orig > 0) else { return f };
    let corr = |rot: f64| dim as f64 * (y.orig as f64 / (rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln());
    let low = corr(y.beta_fast).floor().max(0.0);
    let high = corr(y.beta_slow).ceil().min(dim as f64 - 1.0);
    let (lo, hi) = (low as f32, if low == high { high as f32 + 0.001 } else { high as f32 });
    for (i, v) in f.iter_mut().enumerate() {
        let smooth = 1.0 - ((i as f32 - lo) / (hi - lo)).clamp(0.0, 1.0);
        *v = *v / y.factor as f32 * (1.0 - smooth) + *v * smooth;
    }
    f
}

/// An E8M0 scale's value: 2^(b - 127); 255 is NaN.
pub fn e8m0(b: u8) -> f32 {
    match b {
        255 => f32::NAN,
        // (2^-127 is an f32 subnormal)
        0 => f32::from_bits(0x0040_0000),
        _ => f32::from_bits((b as u32) << 23),
    }
}

/// E2M1's values, the low three bits the magnitude and the fourth the sign.
pub const E2M1: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

/// An FP4 matrix [rows, cols] (two weights a byte, the first in the low four bits; `bytes` rows x cols / 2) with its E8M0 scales (one per 32 weights of a row) as five-bit runs: each weight's code (twice its
/// E2M1 value, -12 to 12) and each run's bf16 scale (half its E8M0 scale), so that code x scale is the weight exactly. An Err where a scale is NaN or past bf16.
pub fn fp4_runs(bytes: &[u8], scales: &[u8], rows: usize, cols: usize) -> Result<(Vec<i8>, Vec<u16>)> {
    if cols % 32 != 0 || bytes.len() * 2 != rows * cols || scales.len() * 32 != rows * cols {
        return err(format!("{} bytes and {} scales for a {rows} x {cols} FP4 matrix", bytes.len(), scales.len()));
    }
    const CODE: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
    let code: Vec<i8> = bytes.iter().flat_map(|&b| [CODE[(b & 15) as usize], CODE[(b >> 4) as usize]]).collect();
    let scale = scales
        .iter()
        .enumerate()
        .map(|(i, &e)| match e {
            255 => err(format!("run {i}'s scale is NaN")),
            // 2^(e - 128): bf16's exponent e - 1, or below its normal range a subnormal (2^-133 times 2^(e + 5))
            2.. => Ok(((e - 1) as u16) << 7),
            _ => Ok(1u16 << (e + 5)),
        })
        .collect::<Result<_>>()?;
    Ok((code, scale))
}

/// An FP8 matrix [rows, cols] with its E8M0 block scales (128 x 128 blocks) as bf16 words, exactly (`glyd_gpu::fp8::dequant`).
pub fn fp8_bf16(q: &[u8], scales: &[u8], rows: usize, cols: usize) -> Result<Vec<u16>> {
    let s: Vec<f32> = scales.iter().map(|&b| e8m0(b)).collect();
    let mut out = vec![0u16; rows * cols];
    glyd_gpu::fp8::dequant(q, &s, rows, cols, [128, 128], &mut out).map_err(Error)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp4_runs_are_the_weights_exactly() {
        for e in [0u8, 1, 2, 100, 120, 127, 128, 200, 254] {
            let bytes: Vec<u8> = (0..16u8).map(|i| i | (15 - i) << 4).collect();
            let (code, scale) = fp4_runs(&bytes, &[e], 1, 32).unwrap();
            let s = f32::from_bits((scale[0] as u32) << 16);
            for (k, &c) in code.iter().enumerate() {
                let nib = if k % 2 == 0 { bytes[k / 2] & 15 } else { bytes[k / 2] >> 4 };
                assert_eq!(c as f32 * s, E2M1[nib as usize] * e8m0(e), "scale {e}, weight {k}");
            }
        }
        assert!(fp4_runs(&[0; 16], &[255], 1, 32).is_err());
    }

    #[test]
    fn freqs_are_yarns() {
        // (the window-only layers': plain powers)
        let f = freqs(64, 10000.0, None);
        assert_eq!(f[0], 1.0);
        assert!((f[31] as f64 - 10000f64.powf(-62.0 / 64.0)).abs() < 1e-9);
        // (the stretched: the fast end unchanged, the slow end divided by the factor)
        let y = Yarn { factor: 16.0, orig: 65536, beta_fast: 32.0, beta_slow: 1.0 };
        let (a, b) = (freqs(64, 160000.0, None), freqs(64, 160000.0, Some(y)));
        assert_eq!(a[0], b[0]);
        assert!((b[31] / a[31] - 1.0 / 16.0).abs() < 1e-6);
    }
}
