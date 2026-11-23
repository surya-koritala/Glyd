//! config.json parsed once (R3a, #136): `ModelConfig` holds the top level and the text model's keys (a multimodal release's `text_config`), which family the engine
//! tells the model apart as, and the keys the families share typed (`Shared`, serde), with their aliases in one place. The planner (`plan::Shape`), the refusals
//! (`support::family`) and the description builder (`describe`) read a model through it; a family's own keys stay in the text model's map, which the builder
//! reads key by key and names any it does not know.

use crate::ffi::{err, Error, Result};
use crate::support::Family;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::{Map, Value};
use std::path::Path;

/// A width that a few configs give per layer (vLLM's Nemotron-H reads `intermediate_size` as a list).
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Width {
    One(usize),
    PerLayer(Vec<usize>),
}

/// What a release says of its stored format (`quantization_config`): the method and, for compressed-tensors, its format.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct Quant {
    pub quant_method: Option<String>,
    pub format: Option<String>,
}

/// The keys the families share, typed: the text model's, else the top level's (and `quantization_config` the top level's first, where releases put it).
/// A key of another type than its field's is an error that names it.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct Shared {
    pub model_type: Option<String>,
    pub hidden_size: Option<usize>,
    pub num_hidden_layers: Option<usize>,
    pub vocab_size: Option<usize>,
    pub num_attention_heads: Option<usize>,
    pub num_key_value_heads: Option<usize>,
    pub head_dim: Option<usize>,
    pub intermediate_size: Option<Width>,
    pub max_position_embeddings: Option<usize>,
    pub tie_word_embeddings: Option<bool>,
    pub sliding_window: Option<usize>,
    pub layer_types: Option<Vec<String>>,
    /// Nemotron-H's layer pattern, a character a layer.
    pub hybrid_override_pattern: Option<String>,
    pub attn_output_gate: Option<bool>,
    pub global_head_dim: Option<usize>,
    pub num_global_key_value_heads: Option<usize>,
    // latent attention
    pub q_lora_rank: Option<usize>,
    pub kv_lora_rank: Option<usize>,
    pub qk_nope_head_dim: Option<usize>,
    pub qk_rope_head_dim: Option<usize>,
    pub v_head_dim: Option<usize>,
    // linear attention and Mamba-2
    pub linear_num_key_heads: Option<usize>,
    pub linear_key_head_dim: Option<usize>,
    pub linear_num_value_heads: Option<usize>,
    pub linear_value_head_dim: Option<usize>,
    pub mamba_num_heads: Option<usize>,
    pub mamba_head_dim: Option<usize>,
    pub ssm_state_size: Option<usize>,
    pub n_groups: Option<usize>,
    // mixtures of experts: the count and the experts a token under each family's name
    pub num_experts: Option<usize>,
    pub num_local_experts: Option<usize>,
    pub n_routed_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    pub num_experts_per_token: Option<usize>,
    pub moe_intermediate_size: Option<usize>,
    pub shared_expert_intermediate_size: Option<usize>,
    pub moe_shared_expert_intermediate_size: Option<usize>,
    pub n_shared_experts: Option<usize>,
    pub first_k_dense_replace: Option<usize>,
    /// The width the routed experts read where it is not the hidden size, under each family's name.
    pub moe_latent_size: Option<usize>,
    pub routed_expert_hidden_size: Option<usize>,
    pub moe_hidden_size: Option<usize>,
    pub expert_dtype: Option<String>,
    pub quantization_config: Option<Quant>,
    // draft layers
    pub mtp_num_hidden_layers: Option<usize>,
    pub num_nextn_predict_layers: Option<usize>,
}

impl Shared {
    /// The routed experts: the largest of the families' names for them (0: none).
    pub fn experts(&self) -> usize {
        [self.num_experts, self.num_local_experts, self.n_routed_experts].into_iter().flatten().max().unwrap_or(0)
    }

    /// The experts a token takes.
    pub fn top_k(&self) -> usize {
        [self.num_experts_per_tok, self.num_experts_per_token].into_iter().flatten().max().unwrap_or(0)
    }

    /// The experts' width where they run in a latent (Nemotron 3 Super's `moe_latent_size`, Kimi-K3's `routed_expert_hidden_size`, Muse Spark's `moe_hidden_size`).
    pub fn latent(&self) -> Option<usize> {
        self.moe_latent_size.or(self.routed_expert_hidden_size).or(self.moe_hidden_size)
    }

    /// The draft layers the checkpoint holds (`mtp_num_hidden_layers`, DeepSeek's `num_nextn_predict_layers`).
    pub fn mtp_layers(&self) -> usize {
        self.mtp_num_hidden_layers.or(self.num_nextn_predict_layers).unwrap_or(0)
    }
}

/// A config.json, parsed once.
#[derive(Clone, Debug)]
pub struct ModelConfig {
    /// The top level, and the text model's keys (`text_config`; the top level's where there is none).
    pub top: Map<String, Value>,
    pub text: Map<String, Value>,
    /// Whether the text model is a `text_config` (a multimodal release).
    pub nested: bool,
    /// `architectures[0]`, the top level's `model_type` and the text model's ("" where a config has none).
    pub arch: String,
    pub model_type: String,
    pub text_type: String,
    pub shared: Shared,
}

/// `T` from the object `m`; an error names the key whose value is not of its type.
fn typed<T: DeserializeOwned>(m: &Map<String, Value>, what: &str) -> Result<T> {
    let all = Value::Object(m.clone());
    T::deserialize(&all).map_err(|e| {
        let key = m.iter().find(|(k, v)| T::deserialize(&Value::Object(Map::from_iter([((*k).clone(), (*v).clone())]))).is_err());
        match key {
            Some((k, v)) => Error(format!("{what}: {k} = {v}: {e}")),
            None => Error(format!("{what}: {e}")),
        }
    })
}

impl ModelConfig {
    /// `dir`'s config.json.
    pub fn load(dir: &Path) -> Result<ModelConfig> {
        let p = dir.join("config.json");
        let text = std::fs::read_to_string(&p).map_err(|e| Error(format!("{}: {e}", p.display())))?;
        ModelConfig::parse(&text).map_err(|e| e.context(p.display()))
    }

    /// A config.json's text.
    pub fn parse(text: &str) -> Result<ModelConfig> {
        let v: Value = serde_json::from_str(text).map_err(|e| Error(format!("config.json: {e}")))?;
        ModelConfig::from_value(&v)
    }

    pub fn from_value(v: &Value) -> Result<ModelConfig> {
        let top = v.as_object().ok_or_else(|| Error("config.json is not an object".into()))?.clone();
        let (text, nested) = match top.get("text_config") {
            Some(Value::Object(t)) => (t.clone(), true),
            Some(_) => return err("config.json: text_config is not an object"),
            None => (top.clone(), false),
        };
        let s = |m: &Map<String, Value>, k: &str| m.get(k).and_then(Value::as_str).unwrap_or("").to_string();
        let arch = top.get("architectures").and_then(Value::as_array).and_then(|a| a.first()).and_then(Value::as_str).unwrap_or("").to_string();
        // (the shared keys: the text model's over the top level's, a null not hiding the top level's; the release's format the top level's first)
        let mut merged = top.clone();
        merged.extend(text.iter().filter(|(_, v)| !v.is_null()).map(|(k, v)| (k.clone(), v.clone())));
        if let Some(q) = top.get("quantization_config").filter(|q| !q.is_null()) {
            merged.insert("quantization_config".into(), q.clone());
        }
        let shared = typed(&merged, "config.json")?;
        Ok(ModelConfig { arch, model_type: s(&top, "model_type"), text_type: s(&text, "model_type"), shared, top, text, nested })
    }

    /// The text model's tensor names' prefix as transformers saves the model: `model.language_model.` under a `text_config`, else `model.`.
    pub fn prefix(&self) -> &'static str {
        if self.nested {
            "model.language_model."
        } else {
            "model."
        }
    }

    /// The family the engine tells this model apart as (`support::TABLE`), None where it is none of them.
    pub fn family(&self) -> Option<Family> {
        family_by(&self.arch, &self.model_type, self.nested.then_some(self.text_type.as_str()))
    }
}

/// A family by the names a release's config gives its architecture and model type (and its text model's, `nested`), in the order the engine has always asked
/// them. One rule for every reader: the refusals, the loaders and each family's own parser.
fn family_by(arch: &str, model_type: &str, nested: Option<&str>) -> Option<Family> {
    let (a, m) = (arch, model_type);
    Some(match () {
        _ if a == "MuseSparkForConditionalGeneration" => Family::Muse,
        _ if m == "kimi_k3" => Family::Kimi,
        _ if a == "Glm5NextForConditionalGeneration" => Family::Glm,
        _ if a == "NemotronHForCausalLM" => Family::NemotronH,
        _ if m == "deepseek_v4" => Family::DeepSeekV4,
        _ if m == "qwen4_exp" => Family::FlashNext,
        _ if m.starts_with("gemma4") || nested.is_some_and(|t| t.starts_with("gemma4")) => Family::Gemma4,
        _ if a == "Qwen3ForCausalLM" => Family::Qwen3,
        _ if a == "Qwen3_5ForCausalLM" || a == "Qwen3_5ForConditionalGeneration" => Family::Qwen35,
        _ => return None,
    })
}

/// The family of a config.json's text, None where it is none the engine tells apart or the text is no JSON object. Only the names are read: a key of
/// another type than the shared keys' is the parse's to name (`ModelConfig::parse`), not a reason to see another family.
pub fn family_of(text: &str) -> Option<Family> {
    let v: Value = serde_json::from_str(text).ok()?;
    let s = |v: &Value, k: &str| v.get(k).and_then(Value::as_str).unwrap_or("").to_string();
    let arch = v.get("architectures").and_then(Value::as_array).and_then(|a| a.first()).and_then(Value::as_str).unwrap_or("");
    let nested = v.get("text_config").filter(|t| t.is_object()).map(|t| s(t, "model_type"));
    family_by(arch, &s(&v, "model_type"), nested.as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> String {
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)).unwrap()
    }

    /// The text model's keys over the top level's, the release's format the top level's; the families' names for one thing read as one.
    #[test]
    fn shared_keys_from_the_text_model_and_the_top() {
        let g = ModelConfig::parse(&fixture("glm5_next_config.json")).unwrap();
        assert_eq!((g.nested, g.prefix(), g.family()), (true, "model.language_model.", Some(Family::Glm)));
        assert_eq!(g.shared.quantization_config.as_ref().and_then(|q| q.quant_method.as_deref()), Some("fp8"));
        assert_eq!((g.shared.experts(), g.shared.top_k(), g.shared.mtp_layers()), (288, 8, 1));
        let n = ModelConfig::parse(&fixture("nemotron_h_config.json")).unwrap();
        assert_eq!((n.nested, n.prefix(), n.family(), n.shared.latent()), (false, "model.", Some(Family::NemotronH), Some(1024)));
        // (a null in the text model does not hide the top level's value)
        let c = ModelConfig::parse(r#"{"tie_word_embeddings": true, "text_config": {"tie_word_embeddings": null, "hidden_size": 8}}"#).unwrap();
        assert_eq!((c.shared.tie_word_embeddings, c.shared.hidden_size), (Some(true), Some(8)));
    }

    /// A shared key of another type is an error that names the key; a config that is not an object is refused.
    #[test]
    fn a_key_of_another_type_is_named() {
        let e = ModelConfig::parse(r#"{"hidden_size": 8, "num_hidden_layers": "eight"}"#).unwrap_err().msg;
        assert!(e.contains("num_hidden_layers") && e.contains("eight"), "{e}");
        assert!(ModelConfig::parse("[1]").unwrap_err().msg.contains("not an object"));
        assert_eq!(ModelConfig::parse(r#"{"intermediate_size": [1, 2]}"#).unwrap().shared.intermediate_size, Some(Width::PerLayer(vec![1, 2])));
    }
}
