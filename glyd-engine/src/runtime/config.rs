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

/// What a release says of its stored format (`quantization_config`): the method and, for compressed-tensors, its format, and the routed experts' dtype where the
/// release names it here (DeepSeek-V4.1's) and not at the top (DeepSeek-V4's).
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct Quant {
    pub quant_method: Option<String>,
    pub format: Option<String>,
    pub expert_dtype: Option<String>,
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
    /// a sparse attention's indexer (GLM-5.3's, DeepSeek-V4's): its heads' width
    pub index_head_dim: Option<usize>,
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

    /// The experts' width where they run in a latent (the Nemotron-H layout's `moe_latent_size`, Kimi-K3's `routed_expert_hidden_size`, Muse Spark's `moe_hidden_size`).
    pub fn latent(&self) -> Option<usize> {
        self.moe_latent_size.or(self.routed_expert_hidden_size).or(self.moe_hidden_size)
    }

    /// The draft layers the checkpoint holds (`mtp_num_hidden_layers`, DeepSeek's `num_nextn_predict_layers`).
    pub fn mtp_layers(&self) -> usize {
        self.mtp_num_hidden_layers.or(self.num_nextn_predict_layers).unwrap_or(0)
    }
}

/// A key that a reader tells apart missing, null and given (`Option` takes null as missing): a field of a family's keys, `#[serde(default)]`.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Given<T> {
    #[default]
    Missing,
    Null,
    Some(T),
}

impl<T: Clone> Given<T> {
    /// The value where one is given.
    pub fn value(&self) -> Option<T> {
        match self {
            Given::Some(v) => Some(v.clone()),
            _ => None,
        }
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Given<T> {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        Option::<T>::deserialize(d).map(|o| o.map_or(Given::Null, Given::Some))
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

    /// Whether the text model is one of the types the description builder's `llama` reading takes: the plain dense kind.
    pub fn is_dense(&self) -> bool {
        super::describe::LLAMA.contains(&self.text_type.as_str())
    }

    /// A key's number (an integer or not), the top level's first, else the text model's.
    pub fn num(&self, key: &str) -> Option<usize> {
        self.top.get(key).or(self.text.get(key))?.as_u64().map(|x| x as usize)
    }

    /// `dir`'s generation_config.json, where it has one.
    pub fn generation(dir: &Path) -> Option<String> {
        std::fs::read_to_string(dir.join("generation_config.json")).ok()
    }

    /// A family's reader of the model in `dir`, from this one parse of its config.json: `of` takes the parse, the file's path for the words of an error and the
    /// generation_config.json beside it.
    pub fn read<T>(dir: &Path, of: impl FnOnce(&ModelConfig, &str, Option<&str>) -> Result<T>) -> Result<T> {
        of(&ModelConfig::load(dir)?, &dir.join("config.json").display().to_string(), ModelConfig::generation(dir).as_deref())
    }
}

/// A family by the names a release's config gives its architecture and model type (and its text model's, `nested`), in the order the engine has always asked
/// them. One rule for every reader: the refusals, the loaders and each family's own parser.
fn family_by(arch: &str, model_type: &str, nested: Option<&str>) -> Option<Family> {
    let (a, m) = (arch, model_type);
    Some(match () {
        _ if a == "MuseSparkForConditionalGeneration" => Family::Muse,
        // (the Kimi-K3 layout: a multimodal wrapper over a `kimi_linear` text model, whatever model type the wrapper's release gives itself: kimi_k3, kimi_k31)
        _ if nested == Some("kimi_linear") => Family::Kimi,
        _ if a == "Glm5NextForConditionalGeneration" => Family::Glm,
        _ if m == "glm_moe_dsa" => Family::GlmDsa,
        _ if a == "NemotronHForCausalLM" => Family::NemotronH,
        _ if m == "deepseek_v4" => Family::DeepSeekV4,
        _ if m == "deepseek_v41" => Family::DeepSeekV41,
        _ if m == "qwen4_exp" => Family::FlashNext,
        _ if m.starts_with("gemma4") || nested.is_some_and(|t| t.starts_with("gemma4")) => Family::Gemma4,
        // (the plain dense kind: every model type the description builder's `llama` reading takes, whatever the release calls its architecture)
        _ if super::describe::LLAMA.contains(&nested.unwrap_or(m)) => Family::Qwen3,
        // (the Qwen3.5 layout, dense or with experts: one family, its loader the one for both, src/dynamic/qwen35.rs)
        _ if matches!(a, "Qwen3_5ForCausalLM" | "Qwen3_5ForConditionalGeneration" | "Qwen3_5MoeForCausalLM" | "Qwen3_5MoeForConditionalGeneration") => {
            Family::Qwen35
        }
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

/// A key's number (an integer or not), else an error that names it: the families' parsers' reading of a float (`rms_norm_eps`, `rope_theta`).
pub fn num(v: &Value, what: &str) -> Result<f64> {
    v.as_f64().ok_or_else(|| Error(format!("config.json: {what}: not a number")))
}

/// An array of unsigned integers, else None.
pub fn u64s(v: &Value) -> Option<Vec<u64>> {
    v.as_array()?.iter().map(Value::as_u64).collect()
}

/// The end-of-sequence ids: generation_config.json's (a number or a list) where it gives any, else the config object's (`config`: the top level, or the text
/// model's).
pub fn eos_ids(config: &Map<String, Value>, generation: Option<&str>) -> Vec<i64> {
    let of = |v: Option<&Value>| match v {
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_i64).collect(),
        Some(x) => x.as_i64().into_iter().collect(),
        None => Vec::new(),
    };
    let g: Option<Value> = generation.and_then(|g| serde_json::from_str(g).ok());
    let eos: Vec<i64> = of(g.as_ref().and_then(|g| g.get("eos_token_id")));
    if eos.is_empty() {
        of(config.get("eos_token_id"))
    } else {
        eos
    }
}

impl ModelConfig {
    /// A family's own keys of the text model, typed by its reader's struct (`T`, serde, its fields `Option` and `#[serde(default)]`): read from this parse, a key
    /// of another type than its field's an error that names it. The keys the families share are `shared`'s.
    pub fn own<T: DeserializeOwned>(&self, what: &str) -> Result<T> {
        typed(&self.text, what)
    }

    /// Whether the release has a vision tower (`vision_config` at the top level).
    pub fn has_vision(&self) -> bool {
        self.top.get("vision_config").is_some()
    }

    /// A family's own keys of the top level (a multimodal release's beside its text model), typed as `own`'s.
    pub fn own_top<T: DeserializeOwned>(&self, what: &str) -> Result<T> {
        typed(&self.top, what)
    }

    /// A shared key a family needs: its value, else an error that names it.
    pub fn need(v: Option<usize>, label: &str, k: &str) -> Result<usize> {
        v.ok_or_else(|| Error(format!("{label}: no {k}")))
    }
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
        let kimi = serde_json::from_str::<Value>(&fixture("headers/Kimi-K3.json")).unwrap()["config"].as_str().unwrap().to_string();
        let k = ModelConfig::parse(&kimi).unwrap();
        assert_eq!((k.nested, k.prefix(), k.family(), k.shared.latent()), (true, "model.language_model.", Some(Family::Kimi), Some(3584)));
        // (the plain dense kind is told by the model type the description builder reads, any of them, and not by what a release calls its architecture)
        for model_type in super::super::describe::LLAMA {
            let d = ModelConfig::parse(&format!(r#"{{"architectures": ["AnyNameForCausalLM"], "model_type": "{model_type}"}}"#)).unwrap();
            assert_eq!((d.nested, d.prefix(), d.family()), (false, "model.", Some(Family::Qwen3)), "{model_type}");
        }
        assert_eq!(family_of(r#"{"architectures": ["Qwen3ForCausalLM"], "model_type": "other"}"#), None);
        // (a null in the text model does not hide the top level's value)
        let c = ModelConfig::parse(r#"{"tie_word_embeddings": true, "text_config": {"tie_word_embeddings": null, "hidden_size": 8}}"#).unwrap();
        assert_eq!((c.shared.tie_word_embeddings, c.shared.hidden_size), (Some(true), Some(8)));
    }

    /// Every family's own reader takes the one parse (R3a, #136: each `parse` is its `of` on `ModelConfig::parse`, so the two are one path and are not compared
    /// here): each reads its fixture's config, and a shared key of another type that the family does not read itself (`n_groups`, Mamba-2's, set to a word) is
    /// refused by each, named with its value, where all but Nemotron-H's reader once passed it by.
    #[test]
    fn every_family_reads_the_one_parse() {
        fn reads<T>(name: &str, text: &str, parse: fn(&str) -> Result<T>) {
            if let Err(e) = parse(text) {
                panic!("{name}: {}", e.msg);
            }
            let mut v: Value = serde_json::from_str(text).unwrap();
            v["n_groups"] = "eight".into();
            let e = parse(&v.to_string()).err().unwrap_or_else(|| panic!("{name}: n_groups \"eight\" passed by")).msg;
            assert!(e.contains("n_groups = \"eight\""), "{name}: {e}");
        }
        let header = |f: &str| serde_json::from_str::<Value>(&fixture(&format!("headers/{f}"))).unwrap()["config"].as_str().unwrap().to_string();
        let q35 = serde_json::from_str::<Value>(&fixture("qwen35_configs.json")).unwrap()["Qwen/Qwen3.5-0.8B"].to_string();
        // (the plain dense kind's reader, on a shape that is no release's by name: `model_type` is the description builder's input)
        let dense = r#"{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 4096, "intermediate_size": 12288, "num_hidden_layers": 36,
            "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936, "max_position_embeddings": 40960, "rms_norm_eps": 1e-6,
            "rope_theta": 1000000, "eos_token_id": 151645}"#;
        reads("plain dense", dense, |t| crate::weights::Config::parse(t, "c", None));
        reads("Qwen3.5", &q35, |t| crate::qwen35::Config35::parse(t, "c", None));
        reads("Gemma 4", &fixture("gemma4_12b_config.json"), |t| crate::gemma4::ConfigG4::parse(t, None, false));
        reads("GLM-5.3-Flash", &fixture("glm5_next_config.json"), |t| crate::glm::ConfigGlm::parse(t, "c", None));
        reads("Kimi-K3", &header("Kimi-K3.json"), |t| crate::kimi::ConfigKimi::parse(t, "c", None));
        reads("DeepSeek-V4", &header("DeepSeek-V4-Flash-0731.json"), |t| crate::deepseek_v4::ConfigV4::parse(t, "c", None));
        #[cfg(feature = "dev")]
        reads("Muse Spark", crate::muse_ref::RANDOM_CONFIG, |t| crate::muse::ConfigMuse::parse(t, "c", None));
        reads("Qwen3.8-Flash-Next", &fixture("qwen4x_config.json"), |t| crate::qwen4x::ConfigQx::parse(t, "c", None));
        // (Nemotron-H reads n_groups itself: refused before as now)
        reads("Nemotron-H", &fixture("nemotron_h_config.json"), |t| crate::nemotron::ConfigNemotronH::parse(t, "c", None));
    }

    /// A number is the top level's, else the text model's; the plain dense kind is the text model's type; a family's reader of a directory takes the one parse.
    #[test]
    fn numbers_kind_and_the_directory_read_from_the_one_parse() {
        let c =
            ModelConfig::parse(r#"{"vocab_size": 5, "model_type": "x", "text_config": {"model_type": "qwen3", "vocab_size": 7, "hidden_size": 8}}"#).unwrap();
        assert_eq!((c.num("vocab_size"), c.num("hidden_size"), c.num("none"), c.is_dense()), (Some(5), Some(8), None, true));
        let dir = std::env::temp_dir().join(format!("glyd-config-read-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), r#"{"model_type": "other", "vocab_size": 9}"#).unwrap();
        std::fs::write(dir.join("generation_config.json"), "{}").unwrap();
        let seen = ModelConfig::read(&dir, |c, label, generation| Ok((c.num("vocab_size"), label.to_string(), generation.map(str::to_string)))).unwrap();
        assert_eq!(seen, (Some(9), dir.join("config.json").display().to_string(), Some("{}".to_string())));
        std::fs::remove_file(dir.join("config.json")).unwrap();
        assert!(ModelConfig::read(&dir, |c, _, _| Ok(c.num("vocab_size"))).unwrap_err().msg.contains("config.json"));
        std::fs::remove_dir_all(&dir).unwrap();
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
