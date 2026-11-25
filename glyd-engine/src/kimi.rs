//! Kimi-K3 (`KimiK3ForConditionalGeneration`, 2026-06; its text model `kimi_linear`) as the engine reads it: layers of Kimi delta attention (GLM-5.3-Flash's,
//! with a full-rank output gate) or of multi-head latent attention with no rotary (a 64-wide key part shared by the heads, a sigmoid gate on its output); a
//! dense MLP in layer 0 and a mixture of experts in the rest (896 MXFP4 experts on a latent of 3,584, an RMS norm on their sum, two shared experts as one MLP);
//! the "situ" activation in every MLP; and no residual stream: each block's input is a softmax over depth of the embedding and the sums of the blocks of 12
//! layers before it (`res_block`). The math: the spec written from the release's own files (research/kimi-k3.md).

use crate::ffi::{err, Error, Result};
use crate::glm::{GlmKind, KdaDims};
use crate::runtime::config::ModelConfig;
use crate::weights::{eos_ids, num};
use std::path::Path;

/// The latent attention: `heads` heads, queries from a latent of `q_rank`, keys and values from one of `kv_rank`; a head's key is its `nope` channels and the
/// `rope` channels every head shares (not rotated: this model has no positional encoding), its value `v`.
#[derive(Clone, Debug, PartialEq)]
pub struct KimiMla {
    pub heads: usize,
    pub q_rank: usize,
    pub kv_rank: usize,
    pub nope: usize,
    pub rope: usize,
    pub v: usize,
}

/// The newer config's sparse attention (`dsa_config`): on the latent attention layers it names (`layers`, from 0) an indexer of a key a token, `heads` query
/// heads of `dim` channels, the best `top_k` tokens kept; its keys RMS-normed, or LayerNorm-ed with a bias (`layer_norm`).
#[derive(Clone, Debug, PartialEq)]
pub struct KimiDsa {
    pub layers: Vec<usize>,
    pub heads: usize,
    pub dim: usize,
    pub top_k: usize,
    pub layer_norm: bool,
}

/// The mixture of experts: the router picks `top_k` of `experts` by sigmoid scores plus a bias (the chosen scores divided by their sum where `renorm`, times
/// `scale`); each expert maps the `latent` the layer's input is projected to through `inter` and back; the `shared` experts are one MLP of `shared` x `inter`.
#[derive(Clone, Debug, PartialEq)]
pub struct KimiMoe {
    pub experts: usize,
    pub top_k: usize,
    pub inter: usize,
    pub latent: usize,
    pub shared: usize,
    pub scale: f32,
    pub renorm: bool,
}

/// What the engine reads of a Kimi-K3 config.json.
#[derive(Clone, Debug)]
pub struct ConfigKimi {
    pub hidden: usize,
    pub layers: usize,
    pub vocab: usize,
    pub eps: f32,
    pub max_pos: usize,
    pub eos: Vec<i64>,
    pub bos: Option<u32>,
    /// Each layer's mixer, in order.
    pub kinds: Vec<GlmKind>,
    /// Each layer's feed-forward: a mixture of experts, else the dense MLP of `inter`.
    pub moe: Vec<bool>,
    pub inter: usize,
    pub kda: KdaDims,
    pub mla: KimiMla,
    /// The sparse attention of the latent attention layers it names, where the config has one.
    pub dsa: Option<KimiDsa>,
    pub experts: KimiMoe,
    /// The situ activation's two caps: gate a, up b, 4 tanh(a / beta) sigmoid(a) x 25 tanh(b / linear) for beta 4 and linear 25.
    pub situ: (f32, f32),
    /// The layers a block of the depth mix sums (12: a block starts at 0, 12, 24, ...).
    pub res_block: usize,
    /// The checkpoint has a vision tower (`vision_tower.*`, `mm_projector.*`), which this engine does not load.
    pub vision: bool,
    /// What the text model's tensor names begin with.
    pub prefix: &'static str,
}

/// Whether a config.json is a Kimi-K3's.
pub fn is_kimi(text: &str) -> bool {
    crate::runtime::config::family_of(text) == Some(crate::support::Family::Kimi)
}

impl ConfigKimi {
    pub fn load(dir: &Path) -> Result<ConfigKimi> {
        let p = dir.join("config.json");
        let text = std::fs::read_to_string(&p).map_err(|e| Error(format!("{}: {e}", p.display())))?;
        let generation = std::fs::read_to_string(dir.join("generation_config.json")).ok();
        ConfigKimi::parse(&text, &p.display().to_string(), generation.as_deref())
    }

    /// A config.json's text (and a generation_config.json's, where there is one): `label` names the file in an error. What this engine does not run is refused
    /// by name: another activation, a rotary, grouped routing, experts outside the latent, a model with its residual stream.
    pub fn parse(text: &str, label: &str, generation: Option<&str>) -> Result<ConfigKimi> {
        ConfigKimi::of(&ModelConfig::parse(text).map_err(|e| e.context(label))?, label, generation)
    }

    /// `parse` of a config.json parsed once (`ModelConfig`).
    pub fn of(c: &ModelConfig, label: &str, generation: Option<&str>) -> Result<ConfigKimi> {
        use serde_json::Value;
        let v = &c.top;
        if v.get("model_type").and_then(Value::as_str) != Some("kimi_k3") {
            return err(format!("{label}: not a Kimi-K3 model (model_type kimi_k3)"));
        }
        let t = v.get("text_config").and_then(Value::as_object).ok_or_else(|| Error(format!("{label}: no text_config")))?;
        if t.get("model_type").and_then(Value::as_str) != Some("kimi_linear") {
            return err(format!("{label}: the text model is not kimi_linear"));
        }
        let get = |k: &str| t.get(k).filter(|x| **x != Value::Null).ok_or_else(|| Error(format!("{label}: no {k}")));
        let u = |k: &str| -> Result<usize> { get(k)?.as_u64().map(|x| x as usize).ok_or_else(|| Error(format!("{label}: {k}: not an integer"))) };
        let f = |k: &str| -> Result<f64> { num(get(k)?, k) };
        let flag = |k: &str| t.get(k).and_then(Value::as_bool);
        let refuse = |what: &str| err(format!("{label}: {what}, which the engine does not run"));
        if t.get("hidden_act").and_then(Value::as_str) != Some("situ") {
            return refuse("an activation other than situ");
        }
        if t.get("rope_scaling").is_some_and(|x| *x != Value::Null) || flag("mla_use_nope") == Some(false) {
            return refuse("a rotary in the latent attention");
        }
        if flag("mla_use_output_gate") != Some(true) || flag("latent_moe_use_norm") != Some(true) {
            return refuse("a latent attention without its output gate, or experts without the latent's norm");
        }
        if t.get("num_expert_group").and_then(Value::as_u64).unwrap_or(1) != 1 {
            return refuse("grouped routing");
        }
        if t.get("moe_router_activation_func").and_then(Value::as_str) != Some("sigmoid") {
            return refuse("a router other than sigmoid's");
        }
        let layers = u("num_hidden_layers")?;
        if layers > crate::plan::MAX_LAYERS {
            return err(format!("{label}: num_hidden_layers {layers}, where no model has more than {}", crate::plan::MAX_LAYERS));
        }
        let heads = u("num_attention_heads")?;
        if u("num_key_value_heads")? != heads {
            return refuse("fewer key and value heads than query heads");
        }
        // the mixers: linear_attn_config's lists are of 1-based layer numbers (the reference tests layer_idx + 1); a layer not in kda_layers is latent attention
        let attn = t.get("linear_attn_config").ok_or_else(|| Error(format!("{label}: no linear_attn_config")))?;
        let list = |k: &str| attn.get(k).and_then(crate::runtime::config::u64s).ok_or_else(|| Error(format!("{label}: no linear_attn_config.{k}")));
        let (kda_l, full_l) = (list("kda_layers")?, list("full_attn_layers")?);
        let kinds: Vec<GlmKind> = (1..=layers as u64).map(|n| if kda_l.contains(&n) { GlmKind::Kda } else { GlmKind::Mla }).collect();
        let mla_n: Vec<u64> = (1..=layers as u64).filter(|n| !kda_l.contains(n)).collect();
        if full_l != mla_n || kda_l.iter().any(|&n| n == 0 || n > layers as u64) {
            return err(format!("{label}: linear_attn_config's kda_layers and full_attn_layers are not the {layers} layers (1-based) split in two"));
        }
        let ka = |k: &str| -> Result<usize> {
            attn.get(k).and_then(Value::as_u64).map(|x| x as usize).ok_or_else(|| Error(format!("{label}: linear_attn_config.{k}")))
        };
        if attn.get("use_full_rank_gate").and_then(Value::as_bool) != Some(true) {
            return refuse("a low-rank output gate in the delta attention");
        }
        let bound = num(attn.get("gate_lower_bound").ok_or_else(|| Error(format!("{label}: no linear_attn_config.gate_lower_bound")))?, "gate_lower_bound")?;
        let kda = KdaDims { heads: ka("num_heads")?, dim: ka("head_dim")?, conv_kernel: ka("short_conv_kernel_size")?, bound: bound as f32 };
        let dense = u("first_k_dense_replace")?;
        if u("moe_layer_freq")? != 1 {
            return refuse("experts every other layer");
        }
        let moe: Vec<bool> = (0..layers).map(|l| l >= dense).collect();
        let mla = KimiMla {
            heads,
            q_rank: u("q_lora_rank")?,
            kv_rank: u("kv_lora_rank")?,
            nope: u("qk_nope_head_dim")?,
            rope: u("qk_rope_head_dim")?,
            v: u("v_head_dim")?,
        };
        // (the newer config's sparse attention: the latent attention layers it names, from 1; what of it this engine does not run the description builder refuses)
        let dsa = match t.get("dsa_config") {
            None | Some(Value::Null) => None,
            Some(Value::Object(o)) => {
                let du = |k: &str| o.get(k).and_then(Value::as_u64).map(|x| x as usize).ok_or_else(|| Error(format!("{label}: dsa_config: no {k}")));
                let named = o.get("dsa_layers").or_else(|| o.get("layers")).and_then(Value::as_array);
                let named = named.ok_or_else(|| Error(format!("{label}: dsa_config names no layers")))?;
                let on: Vec<usize> = named.iter().filter_map(Value::as_u64).filter(|&l| l >= 1).map(|l| l as usize - 1).collect();
                let layer_norm = match o.get("index_k_norm_type").and_then(Value::as_str).unwrap_or("rmsnorm") {
                    "rmsnorm" => false,
                    "layernorm" => true,
                    other => return err(format!("{label}: dsa_config.index_k_norm_type {other}: rmsnorm or layernorm")),
                };
                Some(KimiDsa { layers: on, heads: du("index_num_heads")?, dim: du("index_head_dim")?, top_k: du("index_topk")?, layer_norm })
            }
            Some(x) => return err(format!("{label}: dsa_config {x}: an object")),
        };
        let inter_e = u("moe_intermediate_size")?;
        let experts = KimiMoe {
            experts: u("num_experts")?,
            top_k: u("num_experts_per_token")?,
            inter: inter_e,
            latent: u("routed_expert_hidden_size")?,
            shared: u("num_shared_experts")?,
            scale: f("routed_scaling_factor")? as f32,
            renorm: flag("moe_renormalize").unwrap_or(true),
        };
        if experts.top_k == 0 || experts.top_k > experts.experts || experts.latent == 0 {
            return err(format!("{label}: {} of {} experts on a latent of {}", experts.top_k, experts.experts, experts.latent));
        }
        let situ = (f("activation_situ_beta")? as f32, f("activation_situ_linear_beta")? as f32);
        let eos = eos_ids(t, generation);
        Ok(ConfigKimi {
            hidden: u("hidden_size")?,
            layers,
            vocab: u("vocab_size")?,
            eps: f("rms_norm_eps")? as f32,
            max_pos: u("max_position_embeddings")?,
            eos,
            bos: t.get("bos_token_id").and_then(Value::as_u64).map(|b| b as u32),
            kinds,
            moe,
            inter: u("intermediate_size")?,
            kda,
            mla,
            dsa,
            experts,
            situ,
            res_block: u("attn_res_block_size")?,
            vision: v.get("vision_config").is_some(),
            prefix: "language_model.model.",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The release's config.json (moonshotai/Kimi-K3), its text model's keys the engine reads, the vision tower's left out.
    pub const K3: &str = r#"{"architectures": ["KimiK3ForConditionalGeneration"], "model_type": "kimi_k3", "bos_token_id": 163584, "eos_token_id": 163586, "vision_config": {"vt_num_hidden_layers": 27},
        "text_config": {"model_type": "kimi_linear", "hidden_act": "situ", "activation_situ_beta": 4.0, "activation_situ_linear_beta": 25.0, "attn_res_block_size": 12, "bos_token_id": 163584,
        "eos_token_id": 163586, "first_k_dense_replace": 1, "hidden_size": 7168, "intermediate_size": 33792, "kv_lora_rank": 512, "latent_moe_use_norm": true,
        "linear_attn_config": {"full_attn_layers": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 93], "gate_lower_bound": -5.0, "head_dim": 128,
        "kda_layers": [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 45, 46, 47, 49, 50, 51, 53, 54, 55, 57, 58, 59, 61, 62, 63, 65, 66, 67, 69, 70, 71, 73, 74, 75, 77, 78, 79, 81, 82, 83, 85, 86, 87, 89, 90, 91],
        "num_heads": 96, "short_conv_kernel_size": 4, "use_full_rank_gate": true}, "max_position_embeddings": 1048576, "mla_use_nope": true, "mla_use_output_gate": true, "moe_intermediate_size": 3072,
        "moe_layer_freq": 1, "moe_renormalize": true, "moe_router_activation_func": "sigmoid", "num_attention_heads": 96, "num_expert_group": 1, "num_experts": 896, "num_experts_per_token": 16,
        "num_hidden_layers": 93, "num_key_value_heads": 96, "num_shared_experts": 2, "q_lora_rank": 1536, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "rms_norm_eps": 1e-05,
        "routed_expert_hidden_size": 3584, "routed_scaling_factor": 1.0, "topk_group": 1, "topk_method": "noaux_tc", "v_head_dim": 128, "vocab_size": 163840}}"#;

    #[test]
    fn the_releases_config() {
        assert!(is_kimi(K3));
        let c = ConfigKimi::parse(K3, "config.json", None).unwrap();
        assert_eq!(
            (c.hidden, c.layers, c.vocab, c.eps, c.max_pos, c.eos.clone(), c.bos, c.inter, c.res_block, c.situ, c.vision),
            (7168, 93, 163840, 1e-5, 1_048_576, vec![163586], Some(163584), 33792, 12, (4.0, 25.0), true)
        );
        // (1-based lists: the latent attention at 0-based 3, 7, ..., 91 and 92; the 69 others delta attention; the dense MLP in layer 0 alone)
        let mla: Vec<usize> = (0..93).filter(|&l| c.kinds[l] == GlmKind::Mla).collect();
        assert_eq!(mla, (3..92).step_by(4).chain([92]).collect::<Vec<_>>());
        assert_eq!(c.kinds.iter().filter(|&&k| k == GlmKind::Kda).count(), 69);
        assert_eq!((c.moe[0], c.moe[1], c.moe[92]), (false, true, true));
        assert_eq!(c.kda, KdaDims { heads: 96, dim: 128, conv_kernel: 4, bound: -5.0 });
        assert_eq!(c.mla, KimiMla { heads: 96, q_rank: 1536, kv_rank: 512, nope: 128, rope: 64, v: 128 });
        assert_eq!(c.experts, KimiMoe { experts: 896, top_k: 16, inter: 3072, latent: 3584, shared: 2, scale: 1.0, renorm: true });
    }

    #[test]
    fn what_it_does_not_run_is_refused() {
        let refused = |edit: (&str, &str), what: &str| {
            let e = ConfigKimi::parse(&K3.replace(edit.0, edit.1), "config.json", None).unwrap_err().msg;
            assert!(e.contains(what), "{e}");
        };
        refused((r#""hidden_act": "situ""#, r#""hidden_act": "silu""#), "an activation other than situ");
        refused((r#""num_expert_group": 1"#, r#""num_expert_group": 8"#), "grouped routing");
        refused((r#""use_full_rank_gate": true"#, r#""use_full_rank_gate": false"#), "a low-rank output gate");
        refused((r#""mla_use_nope": true"#, r#""mla_use_nope": false"#), "a rotary");
        refused((r#""full_attn_layers": [4, "#, r#""full_attn_layers": [3, 4, "#), "split in two");
        assert!(!is_kimi(r#"{"model_type": "kimi_linear"}"#));
    }
}
