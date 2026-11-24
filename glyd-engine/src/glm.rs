//! GLM-5.3-Flash (`Glm5NextForConditionalGeneration`, 2026) as the engine reads it: its config (layers of Kimi delta attention or of multi-head latent attention over the keys an indexer picks, a dense MLP in the first three and a
//! mixture of experts in the rest, and four residual streams mixed by learned weights around every layer's two blocks), the tensors it loads and how they are packed, and the block scales of its FP8 source. The reference: transformers
//! 5.19.0's modeling_glm5_next.py (research/glm53-flash-offload.md has the numbers). The checkpoint's matrices are FP8 (E4M3 with 128 x 128 block scales) for the experts, the MLPs and four of the latent attention's, bf16 for the
//! rest; the engine reads either (`glyd_gpu::fp8`) and packs the bf16 a matrix dequantises to.

use crate::ffi::{err, Error, Result};
use crate::qwen35::{Pack35, Tensor35};
use crate::runtime::config::ModelConfig;
use crate::weights::{eos_ids, num, saved_parts};
use glyd_gpu::fp8;
use glyd_gpu::safetensors::Checkpoint;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// A layer's token mixer: Kimi delta attention (`linear_attention`) or multi-head latent attention over the keys an indexer picks (`deepseek_sparse_attention`; transformers calls it `indexed_attention`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlmKind {
    Kda,
    Mla,
}

/// Kimi delta attention: `heads` heads of `dim` channels for queries, keys and values alike, a causal convolution of `conv_kernel` taps on each of the three, and the forget gate's lower bound (the gate is `bound` times a sigmoid).
#[derive(Clone, Debug, PartialEq)]
pub struct KdaDims {
    pub heads: usize,
    pub dim: usize,
    pub conv_kernel: usize,
    pub bound: f32,
}

/// Multi-head latent attention, with no positional encoding: `heads` heads, queries from a latent of `q_rank`, keys and values from a latent of `kv_rank` (a head's key is `nope` wide, its value `v`).
#[derive(Clone, Debug, PartialEq)]
pub struct MlaDims {
    pub heads: usize,
    pub q_rank: usize,
    pub kv_rank: usize,
    pub nope: usize,
    pub v: usize,
}

/// The indexer of each latent attention layer: `heads` heads of `dim` scoring the keys pooled `kpool` at a time (the pool's key a softmax-weighted mean of its `kpool` keys), the `topk` tokens' worth of best pools attended
/// to, and the incomplete last pool too where `tail`.
#[derive(Clone, Debug, PartialEq)]
pub struct IndexerDims {
    pub heads: usize,
    pub dim: usize,
    pub topk: usize,
    pub kpool: usize,
    pub tail: bool,
}

/// The mixture-of-experts layers: the router picks `top_k` of `experts` by sigmoid scores plus a bias (the chosen scores themselves divided by their sum where `norm_topk`, times `scale`), each expert a SwiGLU of `inter` clamped
/// at `limit`, and a shared expert of `shared_inter` added.
#[derive(Clone, Debug, PartialEq)]
pub struct GlmMoeDims {
    pub experts: usize,
    pub top_k: usize,
    pub inter: usize,
    pub shared_inter: usize,
    pub scale: f32,
    pub norm_topk: bool,
    pub limit: f32,
}

/// The hyper-connections (mHC): `mult` residual streams, the mixing weights' Sinkhorn normalisation of `iters` rounds with `eps` in its divisions.
#[derive(Clone, Debug, PartialEq)]
pub struct HcDims {
    pub mult: usize,
    pub eps: f32,
    pub iters: usize,
}

/// What the engine reads of a GLM-5.3-Flash config.json.
#[derive(Clone, Debug)]
pub struct ConfigGlm {
    pub hidden: usize,
    pub layers: usize,
    pub vocab: usize,
    pub eps: f32,
    pub max_pos: usize,
    pub tie: bool,
    pub eos: Vec<i64>,
    /// Each layer's mixer, in order.
    pub kinds: Vec<GlmKind>,
    /// Each layer's feed-forward: a mixture of experts, else a dense MLP of `inter`.
    pub moe: Vec<bool>,
    pub inter: usize,
    pub kda: KdaDims,
    pub mla: MlaDims,
    pub indexer: IndexerDims,
    pub experts: GlmMoeDims,
    pub hc: HcDims,
    /// The layers of the MTP head (`layers.N` from N = `layers`), which this engine does not load.
    pub mtp_layers: usize,
    /// The checkpoint has a vision tower (`model.visual.*`), which this engine does not load.
    pub vision: bool,
    /// config.json's `quantization_config.weight_block_size` (rows, columns) of the block scales of the FP8 matrices, where the model has them.
    pub fp8_block: Option<[usize; 2]>,
    /// What the tensor names of the layers, the embedding and the final norm begin with.
    pub prefix: &'static str,
}

/// Whether a config.json is a GLM-5.3-Flash's (what the dispatcher asks first).
pub fn is_glm(text: &str) -> bool {
    crate::runtime::config::family_of(text) == Some(crate::support::Family::Glm)
}

impl ConfigGlm {
    pub fn load(dir: &Path) -> Result<ConfigGlm> {
        let p = dir.join("config.json");
        let text = std::fs::read_to_string(&p).map_err(|e| Error(format!("{}: {e}", p.display())))?;
        let generation = std::fs::read_to_string(dir.join("generation_config.json")).ok();
        ConfigGlm::parse(&text, &p.display().to_string(), generation.as_deref())
    }

    /// A config.json's text (and a generation_config.json's, where there is one): `label` names the file in an error. A key the config lacks that transformers gives a default is read as that default (the layer schedule, the
    /// experts' normalisation, the swiglu clamp, the hyper-connections' numbers); the model's sizes and the router's scale are required.
    pub fn parse(text: &str, label: &str, generation: Option<&str>) -> Result<ConfigGlm> {
        ConfigGlm::of(&ModelConfig::parse(text).map_err(|e| e.context(label))?, label, generation)
    }

    /// `parse` of a config.json parsed once (`ModelConfig`).
    pub fn of(c: &ModelConfig, label: &str, generation: Option<&str>) -> Result<ConfigGlm> {
        use serde_json::Value;
        let v = &c.top;
        let arch = v.get("architectures").and_then(Value::as_array).and_then(|a| a.first()).and_then(Value::as_str).unwrap_or("");
        if arch != "Glm5NextForConditionalGeneration" {
            return err(format!("{label}: {arch}: not a GLM-5.3-Flash model"));
        }
        if let Some(kind) = v.get("model_type").and_then(Value::as_str).filter(|k| *k != "glm5_next") {
            return err(format!("{label}: the model is {kind:?}, where GLM-5.3-Flash has glm5_next"));
        }
        let t = v.get("text_config").and_then(Value::as_object).ok_or_else(|| Error(format!("{label}: no text_config")))?;
        let kind = t.get("model_type").and_then(Value::as_str).unwrap_or("");
        if kind != "glm5_next_text" {
            return err(format!("{label}: the text model is {kind:?}, where GLM-5.3-Flash has glm5_next_text"));
        }
        let get = |k: &str| t.get(k).ok_or_else(|| Error(format!("{label}: no {k}")));
        let u = |k: &str| -> Result<usize> { get(k)?.as_u64().map(|x| x as usize).ok_or_else(|| Error(format!("{label}: {k}: not an integer"))) };
        let f = |k: &str| -> Result<f64> { num(get(k)?, k) };
        // a key that may be left out: its value, else the default
        let u_or = |k: &str, d: usize| -> Result<usize> {
            if t.get(k).is_some_and(|x| *x != Value::Null) {
                u(k)
            } else {
                Ok(d)
            }
        };
        let f_or = |k: &str, d: f64| -> Result<f64> {
            if t.get(k).is_some_and(|x| *x != Value::Null) {
                f(k)
            } else {
                Ok(d)
            }
        };
        let flag = |k: &str, d: bool| t.get(k).and_then(Value::as_bool).unwrap_or(d);
        let text_is = |k: &str, want: &str| t.get(k).and_then(Value::as_str).is_none_or(|a| a == want);
        let refuse = |what: &str| err(format!("{label}: {what}, which the engine does not run"));
        if !text_is("hidden_act", "silu") {
            return refuse("an activation other than silu");
        }
        if flag("attention_bias", false) {
            return refuse("biases in the attention");
        }
        if t.get("mhc").and_then(Value::as_bool) == Some(false) {
            return refuse("a model without its hyper-connections");
        }
        if u_or("qk_rope_head_dim", 0)? != 0 {
            return refuse("a rotary in the latent attention");
        }
        let layers = u("num_hidden_layers")?;
        if layers > crate::plan::MAX_LAYERS {
            return err(format!("{label}: num_hidden_layers {layers}, where no model has more than {}", crate::plan::MAX_LAYERS));
        }
        let (heads, kv_heads) = (u("num_attention_heads")?, u_or("num_key_value_heads", u("num_attention_heads")?)?);
        if heads == 0 || kv_heads != heads {
            return err(format!("{label}: {heads} query heads to {kv_heads} KV heads, where the latent attention has as many"));
        }
        // the schedule of the mixers: transformers' layer_types (the checkpoint's name for latent attention is deepseek_sparse_attention), else every fourth layer from the fourth
        let kinds: Vec<GlmKind> = match t.get("layer_types").and_then(Value::as_array) {
            Some(a) => {
                if a.len() != layers {
                    return err(format!("{label}: {} layer_types for {layers} layers", a.len()));
                }
                a.iter()
                    .map(|k| match k.as_str() {
                        Some("linear_attention") => Ok(GlmKind::Kda),
                        Some("deepseek_sparse_attention" | "indexed_attention" | "full_attention") => Ok(GlmKind::Mla),
                        other => err(format!("{label}: a layer of type {other:?}, where this model has linear_attention and deepseek_sparse_attention")),
                    })
                    .collect::<Result<_>>()?
            }
            None => (0..layers).map(|i| if i % 4 == 3 { GlmKind::Mla } else { GlmKind::Kda }).collect(),
        };
        let attn = t.get("linear_attn_config");
        let listed = |k: &str| attn.and_then(|a| a.get(k)).and_then(crate::runtime::config::u64s);
        for (key, kind) in [("kda_layers", GlmKind::Kda), ("full_attn_layers", GlmKind::Mla)] {
            let want: Vec<u64> = (0..layers as u64).filter(|&i| kinds[i as usize] == kind).collect();
            if listed(key).is_some_and(|l| l != want) {
                return err(format!("{label}: linear_attn_config.{key} is not the layers layer_types gives"));
            }
        }
        // the schedule of the feed-forwards: mlp_layer_types, and first_k_dense_replace where both are given
        let dense = t.get("first_k_dense_replace").and_then(Value::as_u64).map(|n| n as usize);
        let moe: Vec<bool> = match t.get("mlp_layer_types").and_then(Value::as_array) {
            Some(a) => {
                if a.len() != layers {
                    return err(format!("{label}: {} mlp_layer_types for {layers} layers", a.len()));
                }
                let moe: Vec<bool> = a
                    .iter()
                    .map(|k| match k.as_str() {
                        Some("dense") => Ok(false),
                        Some("sparse") => Ok(true),
                        other => err(format!("{label}: a feed-forward of type {other:?}, where this model has dense and sparse")),
                    })
                    .collect::<Result<_>>()?;
                if dense.is_some_and(|d| moe != (0..layers).map(|i| i >= d).collect::<Vec<_>>()) {
                    return err(format!("{label}: mlp_layer_types and first_k_dense_replace give other dense layers"));
                }
                moe
            }
            None => (0..layers).map(|i| i >= dense.unwrap_or(3)).collect(),
        };
        // Kimi delta attention: linear_attn_config's numbers (transformers' flat keys where it has none); the forget gate's lower bound, the safe gate's -5 where it is null
        let ka = |k: &str, flat: &str, d: usize| -> Result<usize> {
            match attn.and_then(|a| a.get(k)).filter(|x| **x != Value::Null) {
                Some(x) => x.as_u64().map(|x| x as usize).ok_or_else(|| Error(format!("{label}: linear_attn_config.{k}: not an integer"))),
                None => u_or(flat, d),
            }
        };
        let bound = match attn.and_then(|a| a.get("gate_lower_bound")).filter(|x| **x != Value::Null) {
            Some(b) => num(b, "gate_lower_bound")?,
            None if attn.and_then(|a| a.get("safe_gate")).and_then(Value::as_bool) == Some(false) => return refuse("a forget gate without its lower bound"),
            None if attn.is_none() && t.get("linear_lower_bound").is_some_and(|x| *x == Value::Null) => return refuse("a forget gate without its lower bound"),
            None => f_or("linear_lower_bound", -5.0)?,
        };
        let kda = KdaDims {
            heads: ka("num_heads", "linear_num_heads", 64)?,
            dim: ka("head_dim", "linear_head_dim", 128)?,
            conv_kernel: ka("short_conv_kernel_size", "linear_conv_kernel_dim", 4)?,
            bound: bound as f32,
        };
        if kda.heads == 0 || kda.dim == 0 || kda.conv_kernel == 0 {
            return err(format!("{label}: Kimi delta attention of {} heads of {} channels, a convolution of {} taps", kda.heads, kda.dim, kda.conv_kernel));
        }
        let mla = MlaDims { heads, q_rank: u("q_lora_rank")?, kv_rank: u("kv_lora_rank")?, nope: u("qk_nope_head_dim")?, v: u("v_head_dim")? };
        let indexer = IndexerDims {
            heads: u("index_n_heads")?,
            dim: u("index_head_dim")?,
            topk: u("index_topk")?,
            kpool: u("index_kpool")?,
            tail: flag("index_kpool_always_select_tail", true),
        };
        if indexer.kpool == 0 || !indexer.topk.is_multiple_of(indexer.kpool) {
            return err(format!("{label}: index_topk {} does not divide by index_kpool {}", indexer.topk, indexer.kpool));
        }
        // indexer_types says per layer whether the layer runs its indexer (full) or reuses the one before it (shared); only the first is run
        if let Some(a) = t.get("indexer_types").and_then(Value::as_array) {
            if a.len() != layers {
                return err(format!("{label}: {} indexer_types for {layers} layers", a.len()));
            }
            if a.iter().zip(&kinds).any(|(k, kind)| *kind == GlmKind::Mla && k.as_str() != Some("full")) {
                return refuse("a latent attention layer that shares another's indexer");
            }
        }
        let shared = u_or("n_shared_experts", 1)?;
        let experts = GlmMoeDims {
            experts: u("n_routed_experts")?,
            top_k: u("num_experts_per_tok")?,
            inter: u("moe_intermediate_size")?,
            shared_inter: u("moe_intermediate_size")? * shared,
            scale: f("routed_scaling_factor")? as f32,
            norm_topk: flag("norm_topk_prob", true),
            limit: f_or("swiglu_limit", 10.0)? as f32,
        };
        if moe.contains(&true) {
            if experts.top_k == 0 || experts.top_k > experts.experts {
                return err(format!("{label}: {} experts a token of {}", experts.top_k, experts.experts));
            }
            if u_or("n_group", 1)? != 1 || u_or("topk_group", 1)? != 1 {
                return refuse("group-limited routing");
            }
            if shared == 0 {
                return refuse("mixture-of-experts layers without a shared expert");
            }
            if !text_is("scoring_func", "sigmoid") {
                return refuse("routing scores other than sigmoid");
            }
            if !text_is("topk_method", "noaux_tc") {
                return refuse("a selection other than noaux_tc");
            }
            if !text_is("moe_router_dtype", "float32") {
                return refuse("a router that is not computed in float32");
            }
        }
        let hc = HcDims { mult: u("hc_mult")?, eps: f_or("hc_eps", 1e-6)? as f32, iters: u_or("hc_sinkhorn_iters", 20)? };
        if hc.mult == 0 {
            return err(format!("{label}: hc_mult 0"));
        }
        // the FP8 source: block-scaled E4M3 weights (any other quantization is refused); activations are quantised at run time in the original, which a dequantised matrix leaves out
        let fp8_block = match v.get("quantization_config").filter(|q| **q != Value::Null) {
            None => None,
            Some(q) => {
                let method = q.get("quant_method").and_then(Value::as_str).unwrap_or("");
                if method != "fp8" {
                    return refuse(&format!("a quantization of type {method:?}"));
                }
                if let Some(fmt) = q.get("fmt").and_then(Value::as_str).filter(|f| *f != "e4m3") {
                    return refuse(&format!("FP8 of format {fmt}"));
                }
                match q.get("weight_block_size").and_then(crate::runtime::config::u64s).as_deref() {
                    Some(&[m, n]) if m > 0 && n > 0 => Some([m as usize, n as usize]),
                    _ => return err(format!("{label}: an FP8 quantization with no weight_block_size of two sizes")),
                }
            }
        };
        let mut eos = eos_ids(t, generation);
        if eos.is_empty() {
            // (the config's own end tokens are a list here, which eos_ids reads in a generation config alone)
            eos = t
                .get("eos_token_id")
                .or_else(|| v.get("eos_token_id"))
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_i64).collect())
                .unwrap_or_default();
        }
        Ok(ConfigGlm {
            hidden: u("hidden_size")?,
            layers,
            vocab: u("vocab_size")?,
            eps: num(get("rms_norm_eps")?, "rms_norm_eps")? as f32,
            max_pos: u("max_position_embeddings")?,
            tie: v.get("tie_word_embeddings").or_else(|| t.get("tie_word_embeddings")).and_then(Value::as_bool).unwrap_or(false),
            eos,
            kinds,
            moe,
            inter: u("intermediate_size")?,
            kda,
            mla,
            indexer,
            experts,
            hc,
            mtp_layers: u_or("num_nextn_predict_layers", 0)?,
            vision: v.get("vision_config").is_some_and(|x| *x != Value::Null),
            fp8_block,
            prefix: "model.language_model.",
        })
    }

    /// Each layer's mixer and its index among the layers of its kind (a Kimi delta layer's state, a latent attention layer's cache).
    pub fn schedule(&self) -> Vec<(GlmKind, usize)> {
        let (mut kda, mut mla) = (0, 0);
        self.kinds
            .iter()
            .map(|&k| {
                let n = if k == GlmKind::Kda { &mut kda } else { &mut mla };
                *n += 1;
                (k, *n - 1)
            })
            .collect()
    }

    pub fn layers_of(&self, kind: GlmKind) -> usize {
        self.kinds.iter().filter(|&&k| k == kind).count()
    }

    pub fn moe_layers(&self) -> usize {
        self.moe.iter().filter(|&&m| m).count()
    }

    /// The width of q, k and v in a Kimi delta layer: all its heads' channels.
    pub fn qkv(&self) -> usize {
        self.kda.heads * self.kda.dim
    }

    /// The rows of a mixing layer's projection of the streams, `(2 + mult) * mult`: the weights that collapse them (`mult`), spread a block's output over them (`mult`) and mix them among themselves (`mult` squared).
    pub fn hc_rows(&self) -> usize {
        (2 + self.hc.mult) * self.hc.mult
    }

    fn layer_name(&self, l: usize) -> String {
        format!("{}layers.{l}", self.prefix)
    }

    /// Whether `l` is a layer of the model or of its MTP head (the layers after the model's own).
    pub fn has_layer(&self, l: usize) -> bool {
        l < self.layers + self.mtp_layers
    }

    /// The mixer of layer `l`; the MTP head's layer (`l` from `layers` on) is a latent attention layer.
    pub fn kind_of(&self, l: usize) -> GlmKind {
        self.kinds.get(l).copied().unwrap_or(GlmKind::Mla)
    }

    /// Whether layer `l`'s feed-forward is a mixture of experts; the MTP head's layer's is.
    pub fn moe_of(&self, l: usize) -> bool {
        self.moe.get(l).copied().unwrap_or(self.has_layer(l))
    }

    /// One layer's matrices besides a mixture-of-experts layer's routed experts, packable or not, in load order. Kimi delta attention: q, k and v as one, o, the forget gate's two (the second the wide one), the output gate's
    /// two, and the input gate `b_proj`; latent attention: q's down and up, k and v's down and up, o; then the MLP: gate and up as one, down (a mixture-of-experts layer's are the shared expert's). The indexer's matrices
    /// are not among them: the indexer's choice of keys is a ranking, and its 82 million weights are kept as the checkpoint has them (bf16).
    pub fn layer_packs(&self, l: usize) -> Vec<Pack35> {
        let (h, p) = (self.hidden, self.layer_name(l));
        let mut out = Vec::new();
        let mut pack = |module: &str, parts: &[(&str, usize)], cols: usize| {
            out.push(Pack35 {
                parts: parts.iter().map(|&(n, r)| (format!("{module}.{n}.weight"), r)).collect(),
                key: format!("{module}.{}", parts[0].0),
                cols,
            });
        };
        let a = format!("{p}.self_attn");
        match self.kind_of(l) {
            GlmKind::Kda => {
                let (qkv, d) = (self.qkv(), self.kda.dim);
                pack(&a, &[("q_proj", qkv), ("k_proj", qkv), ("v_proj", qkv)], h);
                pack(&a, &[("o_proj", h)], qkv);
                pack(&a, &[("f_a_proj", d)], h);
                pack(&a, &[("f_b_proj", qkv)], d);
                pack(&a, &[("g_a_proj", d)], h);
                pack(&a, &[("g_b_proj", qkv)], d);
                pack(&a, &[("b_proj", self.kda.heads)], h);
            }
            GlmKind::Mla => {
                let m = &self.mla;
                pack(&a, &[("q_a_proj", m.q_rank)], h);
                pack(&a, &[("q_b_proj", m.heads * m.nope)], m.q_rank);
                pack(&a, &[("kv_a_proj_with_mqa", m.kv_rank)], h);
                pack(&a, &[("kv_b_proj", m.heads * (m.nope + m.v))], m.kv_rank);
                pack(&a, &[("o_proj", h)], m.heads * m.v);
            }
        }
        let (mlp, inter) = if self.moe_of(l) { (format!("{p}.mlp.shared_experts"), self.experts.shared_inter) } else { (format!("{p}.mlp"), self.inter) };
        pack(&mlp, &[("gate_proj", inter), ("up_proj", inter)], h);
        pack(&mlp, &[("down_proj", h)], inter);
        out
    }

    /// A mixture-of-experts layer's routed experts: each a pack of its gate and up as one (`...experts.E.gate_proj`) and a pack of its down, in the checkpoint's names, expert by expert.
    pub fn expert_packs(&self, l: usize) -> Vec<Pack35> {
        if !self.moe_of(l) {
            return Vec::new();
        }
        let (m, h, p) = (&self.experts, self.hidden, format!("{}.mlp.experts", self.layer_name(l)));
        (0..m.experts)
            .flat_map(|e| {
                [
                    Pack35 {
                        key: format!("{p}.{e}.gate_proj"),
                        parts: vec![(format!("{p}.{e}.gate_proj.weight"), m.inter), (format!("{p}.{e}.up_proj.weight"), m.inter)],
                        cols: h,
                    },
                    Pack35 { key: format!("{p}.{e}.down_proj"), parts: vec![(format!("{p}.{e}.down_proj.weight"), h)], cols: m.inter },
                ]
            })
            .collect()
    }

    /// The embedding and, where the model has its own, the LM head.
    fn top_matrices(&self) -> Vec<Pack35> {
        let mut out = vec![Pack35 { key: "embed".into(), parts: vec![(format!("{}embed_tokens.weight", self.prefix), self.vocab)], cols: self.hidden }];
        if !self.tie {
            out.push(Pack35 { key: "lm_head".into(), parts: vec![("lm_head.weight".into(), self.vocab)], cols: self.hidden });
        }
        out
    }

    /// The packed matrices in load order (the packable ones of the layers' matrices and experts, then the embedding and the LM head): packed at a level.
    pub fn packs(&self) -> Vec<Pack35> {
        (0..self.layers).flat_map(|l| self.layer_packs(l).into_iter().chain(self.expert_packs(l))).chain(self.top_matrices()).filter(Pack35::packable).collect()
    }

    /// One layer's tensors that are loaded as the checkpoint has them: the two norms and the hyper-connections' parameters (the projection bf16; the base and the three scales fp32); Kimi delta attention's convolutions
    /// (bf16), `A_log` and `dt_bias` (fp32) and gated norm; latent attention's two norms and its indexer, whole; a mixture-of-experts layer's router (bf16, used in fp32) and its score-correction bias (fp32).
    pub fn layer_kept(&self, l: usize) -> Vec<Tensor35> {
        let (h, p) = (self.hidden as u64, self.layer_name(l));
        let t = |name: String, dtype: &'static str, shape: &[u64]| Tensor35 { name, dtype, shape: shape.to_vec() };
        let mut out = vec![t(format!("{p}.input_layernorm.weight"), "BF16", &[h]), t(format!("{p}.post_attention_layernorm.weight"), "BF16", &[h])];
        let rows = self.hc_rows() as u64;
        for block in ["attn", "ffn"].into_iter().filter(|_| l < self.layers) {
            // (the MTP head's layer has a plain residual stream: no mixing weights)
            out.push(t(format!("{p}.hc_{block}_fn"), "BF16", &[rows, self.hc.mult as u64 * h]));
            out.push(t(format!("{p}.hc_{block}_base"), "F32", &[rows]));
            out.push(t(format!("{p}.hc_{block}_scale"), "F32", &[3]));
        }
        let a = format!("{p}.self_attn");
        match self.kind_of(l) {
            GlmKind::Kda => {
                let (k, qkv) = (&self.kda, self.qkv() as u64);
                for c in ["q", "k", "v"] {
                    out.push(t(format!("{a}.{c}_conv1d.weight"), "BF16", &[qkv, 1, k.conv_kernel as u64]));
                }
                out.push(t(format!("{a}.A_log"), "F32", &[k.heads as u64]));
                out.push(t(format!("{a}.dt_bias"), "F32", &[qkv]));
                out.push(t(format!("{a}.o_norm.weight"), "BF16", &[k.dim as u64]));
            }
            GlmKind::Mla => {
                let (m, i) = (&self.mla, &self.indexer);
                out.push(t(format!("{a}.q_a_layernorm.weight"), "BF16", &[m.q_rank as u64]));
                out.push(t(format!("{a}.kv_a_layernorm.weight"), "BF16", &[m.kv_rank as u64]));
                let (heads, dim, ix) = (i.heads as u64, i.dim as u64, format!("{a}.indexer"));
                out.push(t(format!("{ix}.wq_b.weight"), "BF16", &[heads * dim, m.q_rank as u64]));
                out.push(t(format!("{ix}.wk.weight"), "BF16", &[dim, h]));
                out.push(t(format!("{ix}.k_norm.weight"), "BF16", &[dim]));
                out.push(t(format!("{ix}.k_norm.bias"), "BF16", &[dim]));
                out.push(t(format!("{ix}.weights_proj.weight"), "BF16", &[heads, h]));
                out.push(t(format!("{ix}.index_kpool_compress_ape"), "BF16", &[i.kpool as u64, dim]));
                out.push(t(format!("{ix}.index_kpool_compress_gate"), "BF16", &[dim, h]));
            }
        }
        if self.moe_of(l) {
            out.push(t(format!("{p}.mlp.gate.weight"), "BF16", &[self.experts.experts as u64, h]));
            out.push(t(format!("{p}.mlp.gate.e_score_correction_bias"), "F32", &[self.experts.experts as u64]));
        }
        out
    }

    /// The MTP head's own tensors besides its layer's (`layer_kept`, `layer_packs`, `expert_packs` of `layers`): the two norms of the token's embedding and of the previous hidden state, the projection of the two together
    /// back to the model's width, and the head's final norm (its LM head is the model's).
    pub fn mtp_kept(&self) -> Vec<Tensor35> {
        let (h, p) = (self.hidden as u64, self.layer_name(self.layers));
        let t = |name: String, shape: &[u64]| Tensor35 { name, dtype: "BF16", shape: shape.to_vec() };
        vec![
            t(format!("{p}.enorm.weight"), &[h]),
            t(format!("{p}.hnorm.weight"), &[h]),
            t(format!("{p}.eh_proj.weight"), &[h, 2 * h]),
            t(format!("{p}.shared_head.norm.weight"), &[h]),
        ]
    }

    /// Every tensor loaded as the checkpoint has it, in load order: each layer's (`layer_kept`) and the parts of its matrices the library cannot pack, then the final norm and any such part of the embedding or the head.
    pub fn kept(&self) -> Vec<Tensor35> {
        let t = |name: String, dtype: &'static str, shape: &[u64]| Tensor35 { name, dtype, shape: shape.to_vec() };
        let unpacked = |packs: Vec<Pack35>| -> Vec<Tensor35> {
            packs.into_iter().filter(|p| !p.packable()).flat_map(|p| p.parts.into_iter().map(move |(n, r)| t(n, "BF16", &[r as u64, p.cols as u64]))).collect()
        };
        let mut out = Vec::new();
        for l in 0..self.layers {
            out.extend(self.layer_kept(l));
            out.extend(unpacked(self.layer_packs(l).into_iter().chain(self.expert_packs(l)).collect()));
        }
        out.push(t(format!("{}norm.weight", self.prefix), "BF16", &[self.hidden as u64]));
        out.extend(unpacked(self.top_matrices()));
        out
    }

    /// Whether a checkpoint's tensor is one the engine leaves out: the vision tower's and the MTP head's (the layers past the model's own).
    pub fn skipped(&self, name: &str) -> bool {
        let past = |n: &str| {
            n.strip_prefix(&format!("{}layers.", self.prefix))
                .and_then(|r| r.split('.').next())
                .and_then(|i| i.parse::<usize>().ok())
                .is_some_and(|i| i >= self.layers && i < self.layers + self.mtp_layers)
        };
        name.starts_with("model.visual.") || past(name)
    }

    /// The matrices `ck` holds as saved packs (a glyd-v7 directory's), by their parts' names (`Pack35::saved`): what `check` takes as held in their three buffers.
    pub fn saved(&self, ck: &Checkpoint) -> HashSet<String> {
        self.packs().iter().filter(|p| !saved_parts(ck, p).is_empty()).flat_map(|p| p.parts.iter().map(|x| Pack35::saved(&x.0).to_string())).collect()
    }

    fn report(bad: Vec<String>) -> Result<()> {
        if bad.is_empty() {
            return Ok(());
        }
        let more = if bad.len() > 8 { format!(" (and {} more)", bad.len() - 8) } else { String::new() };
        err(format!("{}{more}", bad[..bad.len().min(8)].join("; ")))
    }

    /// The matrices of `packs` among the tensors `by` (name to dtype and shape): each present as bf16 or as FP8 with its block scales (`NAME_scale_inv`, F32 of the grid of the config's blocks), except those held as
    /// saved packs (`packed`, by `Pack35::saved`). The names wanted go into `wanted`, the problems into `bad`.
    fn matrices_present(
        &self,
        packs: &[Pack35],
        by: &HashMap<&str, (&str, &[u64])>,
        packed: &HashSet<String>,
        wanted: &mut HashSet<String>,
        bad: &mut Vec<String>,
    ) {
        for p in packs {
            for (n, rows) in &p.parts {
                if packed.contains(Pack35::saved(n)) {
                    continue;
                }
                wanted.insert(n.clone());
                let shape = [*rows as u64, p.cols as u64];
                match by.get(n.as_str()) {
                    None => bad.push(format!("no {n}")),
                    Some(&(d, s)) if s != shape => bad.push(format!("{n}: {d} {s:?}, where {shape:?} is wanted")),
                    Some(&("BF16", _)) => {}
                    Some(&("F8_E4M3", _)) => {
                        let scale = format!("{n}_scale_inv");
                        wanted.insert(scale.clone());
                        match (self.fp8_block, by.get(scale.as_str())) {
                            (None, _) => bad.push(format!("{n}: F8_E4M3, and config.json gives no weight_block_size for its block scales")),
                            (_, None) => bad.push(format!("no {scale}, the block scales of {n}")),
                            (Some(block), Some(&(d, s))) => {
                                let grid = fp8::grid(*rows, p.cols, block).map(|g| g as u64);
                                if d != "F32" || s != grid {
                                    bad.push(format!("{scale}: {d} {s:?}, where F32 {grid:?} is wanted"));
                                }
                            }
                        }
                    }
                    Some(&(d, _)) => bad.push(format!("{n}: {d}, where BF16 or F8_E4M3 is wanted")),
                }
            }
        }
    }

    /// The tensors of `kept` among `by`: each present with its dtype (an fp32 one may be held as bf16) and shape. The names wanted go into `wanted`, the problems into `bad`.
    fn kept_present(kept: Vec<Tensor35>, by: &HashMap<&str, (&str, &[u64])>, wanted: &mut HashSet<String>, bad: &mut Vec<String>) {
        for w in kept {
            wanted.insert(w.name.clone());
            match by.get(w.name.as_str()) {
                None => bad.push(format!("no {}", w.name)),
                Some(&(d, s)) if !(d == w.dtype || (w.dtype == "F32" && d == "BF16")) || s != w.shape.as_slice() => {
                    bad.push(format!("{}: {d} {s:?}, where {} {:?} is wanted", w.name, w.dtype, w.shape))
                }
                _ => {}
            }
        }
    }

    /// A checkpoint's tensors (name, dtype, shape) against what the engine loads: each matrix present as bf16 or as FP8 with its block scales (`NAME_scale_inv`, F32 of the grid of the config's blocks), each tensor kept present
    /// with its dtype (an fp32 one may be held as bf16) and shape, and nothing else but the vision tower and the MTP head. Returns the number of tensors skipped. The tensors in `packed` (by `Pack35::saved`) are held as
    /// their three buffers, not as tensors.
    pub fn check(&self, have: &[(String, String, Vec<u64>)], packed: &HashSet<String>) -> Result<usize> {
        let by: HashMap<&str, (&str, &[u64])> = have.iter().map(|(n, d, s)| (n.as_str(), (d.as_str(), s.as_slice()))).collect();
        let (mut bad, mut wanted) = (Vec::new(), HashSet::new());
        self.matrices_present(&self.packs(), &by, packed, &mut wanted, &mut bad);
        Self::kept_present(self.kept(), &by, &mut wanted, &mut bad);
        Self::report(bad)?;
        let buffers: HashSet<String> = packed.iter().flat_map(|k| ["data", "blocks", "block_base"].map(|s| format!("{k}.glyd_{s}"))).collect();
        let (mut left_out, mut bad) = (0, Vec::new());
        for (n, _, _) in have {
            if wanted.contains(n) || buffers.contains(n) {
                continue;
            }
            if self.skipped(n) {
                left_out += 1;
            } else {
                bad.push(format!("{n}: not a tensor of this model"));
            }
        }
        Self::report(bad).map(|()| left_out)
    }

    /// Just these layers' tensors against a checkpoint that may hold nothing else (a few layers read from the Hub by range): their matrices (a layer's experts too) and small tensors each present with its dtype and
    /// shape, the matrices in `packed` (as for `check`) as their saved buffers.
    pub fn check_layers(&self, have: &[(String, String, Vec<u64>)], layers: &[usize], packed: &HashSet<String>) -> Result<()> {
        if let Some(&l) = layers.iter().find(|&&l| l >= self.layers) {
            return err(format!("layer {l}, where the model has {}", self.layers));
        }
        let by: HashMap<&str, (&str, &[u64])> = have.iter().map(|(n, d, s)| (n.as_str(), (d.as_str(), s.as_slice()))).collect();
        let (mut bad, mut wanted) = (Vec::new(), HashSet::new());
        let packs: Vec<Pack35> = layers.iter().flat_map(|&l| self.layer_packs(l).into_iter().chain(self.expert_packs(l))).collect();
        self.matrices_present(&packs, &by, packed, &mut wanted, &mut bad);
        Self::kept_present(layers.iter().flat_map(|&l| self.layer_kept(l)).collect(), &by, &mut wanted, &mut bad);
        Self::report(bad)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::nemotron::tests::{bf16_bytes, weights};
    use glyd_gpu::fp8::E4M3;
    use glyd_gpu::json::{self, Value};

    pub(crate) fn text() -> &'static str {
        include_str!("../tests/fixtures/glm5_next_config.json")
    }

    pub(crate) fn tiny_text() -> &'static str {
        include_str!("../tests/fixtures/glm5_next_tiny_config.json")
    }

    /// The real config with the Hub's generation_config.json beside it, as the directory has them.
    pub(crate) fn parse() -> ConfigGlm {
        ConfigGlm::parse(text(), "glm", Some(include_str!("../tests/fixtures/glm5_next_generation_config.json"))).unwrap()
    }

    /// The config with `edit` applied to its text model's keys (`top`: to the top level's instead), as text.
    fn edited(top: bool, edit: impl Fn(&mut Vec<(String, Value)>)) -> String {
        let mut v = json::parse(text()).unwrap();
        if let Value::Object(o) = &mut v {
            match o.iter_mut().find(|(k, _)| k == "text_config") {
                Some((_, Value::Object(t))) if !top => edit(t),
                _ if top => edit(o),
                _ => unreachable!(),
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

    fn refused(text: &str, says: &str) {
        let e = ConfigGlm::parse(text, "x", None).err().unwrap_or_else(|| panic!("not refused: {says:?}")).msg;
        assert!(e.contains(says), "{e:?} does not say {says:?}");
    }

    /// The checkpoint's tensors as the fixture holds them (made by `tools/glm_inventory.py` from the shards' headers of the pinned revision): every name, the experts of a layer as a run.
    pub(crate) type Index = Vec<(String, String, Vec<u64>)>;

    pub(crate) fn tensors() -> Index {
        let v = json::parse(include_str!("../tests/fixtures/glm5_next_tensors.json")).unwrap();
        let one = |n: &str, e: &Value| {
            let a = e.as_array().unwrap();
            (n.to_string(), a[0].as_str().unwrap().to_string(), a[1].as_u64s().unwrap())
        };
        let mut out: Index = v.get("tensors").unwrap().as_object().unwrap().iter().map(|(n, e)| one(n, e)).collect();
        for run in v.get("runs").unwrap().as_array().unwrap() {
            let prefix = run.get("prefix").unwrap().as_str().unwrap();
            for e in 0..run.get("count").unwrap().as_u64().unwrap() {
                for (n, t) in run.get("tensors").unwrap().as_object().unwrap() {
                    out.push(one(&format!("{prefix}{e}.{n}"), t));
                }
            }
        }
        out
    }

    /// The MTP head's layer (the one after the model's 45) is a latent attention layer with experts on a plain residual stream, and the checkpoint holds every tensor of it that the engine names: its layer's matrices and
    /// experts, its norms, the indexer, the router, and the head's own four (the norms of the embedding and of the hidden state, the projection of the two, the final norm), no mixing weights.
    #[test]
    fn the_mtp_heads_tensors_are_the_checkpoints() {
        let c = parse();
        assert_eq!((c.mtp_layers, c.kind_of(45), c.moe_of(45), c.kind_of(44), c.moe_of(2)), (1, GlmKind::Mla, true, GlmKind::Kda, false));
        let have = tensors();
        let by: HashMap<&str, (&str, &[u64])> = have.iter().map(|(n, d, s)| (n.as_str(), (d.as_str(), s.as_slice()))).collect();
        // the layer's matrices (the latent attention's five, the shared expert's two) and its 288 experts' two each, by name and shape
        let packs: Vec<Pack35> = c.layer_packs(45).into_iter().chain(c.expert_packs(45)).collect();
        assert_eq!(packs.len(), 5 + 2 + 576);
        for p in &packs {
            for (n, rows) in &p.parts {
                let (_, s) = by.get(n.as_str()).unwrap_or_else(|| panic!("no {n}"));
                assert_eq!(*s, [*rows as u64, p.cols as u64], "{n}");
            }
        }
        // the tensors kept: the norms, the indexer, the router and its bias, then the head's own; nothing of the mixing weights
        let kept: Vec<Tensor35> = c.layer_kept(45).into_iter().chain(c.mtp_kept()).collect();
        assert!(kept.iter().all(|t| !t.name.contains(".hc_")));
        for t in &kept {
            let (d, s) = by.get(t.name.as_str()).unwrap_or_else(|| panic!("no {}", t.name));
            assert!(*d == t.dtype && *s == t.shape.as_slice(), "{}: {d} {s:?}, where {} {:?} is wanted", t.name, t.dtype, t.shape);
        }
        assert_eq!(kept.len(), 2 + 2 + 7 + 2 + 4);
        // and the engine names every tensor of the layer (the experts' and the FP8 scales aside)
        let named: std::collections::HashSet<&str> =
            packs.iter().flat_map(|p| p.parts.iter().map(|x| x.0.as_str())).chain(kept.iter().map(|t| t.name.as_str())).collect();
        let rest: Vec<&str> =
            have.iter().map(|t| t.0.as_str()).filter(|n| n.contains(".layers.45.") && !n.ends_with("_scale_inv") && !named.contains(n)).collect();
        assert!(rest.is_empty(), "tensors of the MTP layer that nothing names: {rest:?}");
    }

    /// Whether the real checkpoint holds this matrix as FP8: the MLPs' (dense, shared and routed) and, in a latent attention layer, q's two, k and v's down and o. The rest are bf16.
    pub(crate) fn is_fp8(c: &ConfigGlm, name: &str) -> bool {
        let layer = name.split('.').nth(3).and_then(|i| i.parse::<usize>().ok());
        name.contains(".mlp.") || (layer.is_some_and(|i| c.kinds[i] == GlmKind::Mla) && name.contains("self_attn.") && !name.contains("kv_b_proj"))
    }

    /// The E4M3 byte nearest to `x` (a tie to the even byte), saturating at 448.
    fn encode(x: f32) -> u8 {
        let pos = &E4M3[..=0x7E];
        let a = x.abs().min(448.0);
        let i = pos.partition_point(|&v| v < a);
        let code = if i == 0 {
            0
        } else {
            let (below, above) = (a - pos[i - 1], pos[i] - a);
            if below < above || (below == above && (i - 1) % 2 == 0) {
                i - 1
            } else {
                i
            }
        };
        code as u8 | if x.is_sign_negative() { 0x80 } else { 0 }
    }

    /// An f32 rounded to bf16, to nearest and ties to even.
    fn rne(x: f32) -> u16 {
        let b = x.to_bits();
        (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
    }

    /// A [rows, cols] matrix of bf16 words as E4M3 bytes and f32 block scales (each block's largest magnitude over 448, as a block-scaled quantiser makes them; the last blocks short where the matrix does not divide),
    /// and the bf16 words those dequantise to, each worked out a weight at a time.
    fn quantise(w: &[u16], rows: usize, cols: usize, block: [usize; 2]) -> (Vec<u8>, Vec<f32>, Vec<u16>) {
        let [gr, gc] = fp8::grid(rows, cols, block);
        let f = |x: u16| f32::from_bits((x as u32) << 16);
        let scales: Vec<f32> = (0..gr * gc)
            .map(|b| {
                let (r0, c0) = ((b / gc) * block[0], (b % gc) * block[1]);
                let amax = (r0..(r0 + block[0]).min(rows))
                    .flat_map(|r| (c0..(c0 + block[1]).min(cols)).map(move |c| (r, c)))
                    .map(|(r, c)| f(w[r * cols + c]).abs())
                    .fold(0f32, f32::max);
                if amax > 0.0 {
                    amax / 448.0
                } else {
                    1.0
                }
            })
            .collect();
        let scale = |i: usize| scales[(i / cols / block[0]) * gc + i % cols / block[1]];
        let q: Vec<u8> = (0..rows * cols).map(|i| encode(f(w[i]) / scale(i))).collect();
        let deq = (0..rows * cols).map(|i| rne(E4M3[q[i] as usize] * scale(i))).collect();
        (q, scales, deq)
    }

    /// The tiny model `c` (the config `text`) as two Hub-format directories under `dir`: `fp8/`, laid out as the real checkpoint is (the matrices `is_fp8` names as E4M3 bytes with their block scales, the others bf16, the fp32
    /// parameters fp32 with low bits a bf16 cannot hold), and `bf16/`, the same with each FP8 matrix as the bf16 it dequantises to. Both hold a tensor of the MTP head and one of the vision tower, which the engine does not
    /// read. Returns the bf16 words of every matrix by name (the dequantised ones).
    pub(crate) fn checkpoint(dir: &Path, text: &str, c: &ConfigGlm) -> HashMap<String, Vec<u16>> {
        type Owned = (String, &'static str, Vec<u64>, Vec<u8>);
        let (mut fp8, mut twin, mut by): (Vec<Owned>, Vec<Owned>, HashMap<String, Vec<u16>>) = (Vec::new(), Vec::new(), HashMap::new());
        let mut seed = 100u64;
        let block = c.fp8_block.expect("a config with FP8 blocks");
        for p in c.packs() {
            for (name, rows) in &p.parts {
                seed += 1;
                let (shape, w) = (vec![*rows as u64, p.cols as u64], weights(rows * p.cols, seed));
                if is_fp8(c, name) {
                    let (q, scales, deq) = quantise(&w, *rows, p.cols, block);
                    let grid = fp8::grid(*rows, p.cols, block).map(|g| g as u64).to_vec();
                    fp8.push((name.clone(), "F8_E4M3", shape.clone(), q));
                    fp8.push((format!("{name}_scale_inv"), "F32", grid, scales.iter().flat_map(|s| s.to_le_bytes()).collect()));
                    twin.push((name.clone(), "BF16", shape, bf16_bytes(&deq)));
                    by.insert(name.clone(), deq);
                } else {
                    fp8.push((name.clone(), "BF16", shape.clone(), bf16_bytes(&w)));
                    twin.push((name.clone(), "BF16", shape, bf16_bytes(&w)));
                    by.insert(name.clone(), w);
                }
            }
        }
        let h = c.hidden as u64;
        let mut small: Vec<(String, &'static str, Vec<u64>)> = c.kept().into_iter().map(|t| (t.name, t.dtype, t.shape)).collect();
        small.push((l(c.layers, "enorm.weight"), "BF16", vec![h]));
        small.push((l(c.layers, "eh_proj.weight"), "BF16", vec![h, 2 * h]));
        small.push(("model.visual.patch_embed.proj.weight".into(), "BF16", vec![64, 64]));
        for (name, dtype, shape) in small {
            seed += 1;
            let n = shape.iter().product::<u64>() as usize;
            let w = weights(n, seed);
            // (an fp32 parameter: the bf16 word as its high half, and low bits of its own)
            let bytes = if dtype == "F32" {
                (0..n).flat_map(|i| (((w[i] as u32) << 16) | ((i as u32).wrapping_mul(2654435761) >> 16)).to_le_bytes()).collect()
            } else {
                bf16_bytes(&w)
            };
            fp8.push((name.clone(), dtype, shape.clone(), bytes.clone()));
            twin.push((name, dtype, shape, bytes));
        }
        for (sub, ts) in [("fp8", &fp8), ("bf16", &twin)] {
            let d = dir.join(sub);
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("config.json"), text).unwrap();
            let tensors: Vec<(&str, &str, Vec<u64>, &[u8])> = ts.iter().map(|(n, t, s, b)| (n.as_str(), *t, s.clone(), &b[..])).collect();
            crate::pack::write_safetensors(&d.join("model.safetensors"), &[("format", "pt".to_string())], &tensors).unwrap();
        }
        by
    }

    /// The model's config as the pinned revision has it: 45 layers, 34 of Kimi delta attention and 11 of latent attention at 3, 7, ..., 43; a dense MLP in the first three; the FP8 source's blocks.
    #[test]
    fn the_flash_config() {
        let c = parse();
        assert_eq!(
            (c.hidden, c.layers, c.vocab, c.eps, c.max_pos, c.tie, c.eos.clone()),
            (4096, 45, 154880, 1e-5, 1_048_576, false, vec![154820, 154827, 154829])
        );
        assert_eq!(c.kda, KdaDims { heads: 64, dim: 128, conv_kernel: 4, bound: -5.0 });
        assert_eq!(c.mla, MlaDims { heads: 64, q_rank: 1536, kv_rank: 512, nope: 256, v: 256 });
        assert_eq!(c.indexer, IndexerDims { heads: 32, dim: 128, topk: 2048, kpool: 4, tail: true });
        assert_eq!(c.experts, GlmMoeDims { experts: 288, top_k: 8, inter: 2048, shared_inter: 2048, scale: 2.5, norm_topk: true, limit: 10.0 });
        assert_eq!(c.hc, HcDims { mult: 4, eps: 1e-6, iters: 20 });
        assert_eq!((c.inter, c.mtp_layers, c.vision, c.fp8_block, c.prefix), (12288, 1, true, Some([128, 128]), "model.language_model."));
        assert_eq!((c.qkv(), c.hc_rows()), (8192, 24));
        // the end tokens are the config's list when there is no generation_config.json, and generation_config.json's where there is one
        assert_eq!(ConfigGlm::parse(text(), "x", None).unwrap().eos, c.eos);
        assert_eq!(ConfigGlm::parse(text(), "x", Some(r#"{"eos_token_id": [7, 8]}"#)).unwrap().eos, vec![7, 8]);
        assert!(is_glm(text()) && !is_glm(r#"{"architectures": ["Qwen3ForCausalLM"]}"#) && !is_glm("{}"));
    }

    #[test]
    fn the_schedule() {
        let c = parse();
        let s = c.schedule();
        assert_eq!(s.len(), 45);
        assert_eq!((c.layers_of(GlmKind::Kda), c.layers_of(GlmKind::Mla), c.moe_layers()), (34, 11, 42));
        let mla: Vec<usize> = (0..45).filter(|&l| s[l].0 == GlmKind::Mla).collect();
        assert_eq!(mla, [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]);
        assert_eq!(
            (s[0], s[2], s[3], s[4], s[43], s[44]),
            ((GlmKind::Kda, 0), (GlmKind::Kda, 2), (GlmKind::Mla, 0), (GlmKind::Kda, 3), (GlmKind::Mla, 10), (GlmKind::Kda, 33))
        );
        assert_eq!(c.moe.iter().position(|&m| m), Some(3));
        assert!(c.moe[3..].iter().all(|&m| m) && !c.moe[..3].iter().any(|&m| m));
        for kind in [GlmKind::Kda, GlmKind::Mla] {
            let idx: Vec<usize> = s.iter().filter(|x| x.0 == kind).map(|x| x.1).collect();
            assert_eq!(idx, (0..idx.len()).collect::<Vec<_>>());
        }
    }

    #[test]
    fn what_is_refused() {
        refused(r#"{"architectures": ["Qwen3_5ForConditionalGeneration"]}"#, "not a GLM-5.3-Flash model");
        refused(&edited(true, |t| set(t, "model_type", r#""glm5""#)), "glm5_next");
        refused(&edited(false, |t| set(t, "model_type", r#""glm5_next""#)), "glm5_next_text");
        refused(&edited(true, |t| t.retain(|(k, _)| k != "text_config")), "no text_config");
        refused(&edited(false, |t| t.retain(|(k, _)| k != "hidden_size")), "no hidden_size");
        refused(&edited(false, |t| set(t, "hidden_act", r#""gelu""#)), "an activation other than silu");
        refused(&edited(false, |t| set(t, "attention_bias", "true")), "biases in the attention");
        refused(&edited(false, |t| set(t, "mhc", "false")), "without its hyper-connections");
        refused(&edited(false, |t| set(t, "qk_rope_head_dim", "64")), "a rotary in the latent attention");
        refused(&edited(false, |t| set(t, "num_key_value_heads", "8")), "64 query heads to 8 KV heads");
        refused(&edited(false, |t| set(t, "layer_types", r#"["linear_attention"]"#)), "1 layer_types for 45 layers");
        refused(
            &edited(false, |t| set(t, "layer_types", &format!("[{}]", vec![r#""sliding_attention""#; 45].join(",")))),
            "a layer of type Some(\"sliding_attention\")",
        );
        refused(
            &edited(false, |t| set(t, "linear_attn_config", r#"{"kda_layers": [0, 1, 2], "num_heads": 64, "head_dim": 128, "short_conv_kernel_size": 4}"#)),
            "linear_attn_config.kda_layers is not the layers layer_types gives",
        );
        refused(&edited(false, |t| set(t, "mlp_layer_types", r#"["dense"]"#)), "1 mlp_layer_types for 45 layers");
        refused(&edited(false, |t| set(t, "mlp_layer_types", &format!("[{}]", vec![r#""moe""#; 45].join(",")))), "a feed-forward of type Some(\"moe\")");
        refused(&edited(false, |t| set(t, "first_k_dense_replace", "2")), "mlp_layer_types and first_k_dense_replace give other dense layers");
        let shared_at = |i: usize| format!("[{}]", (0..45).map(|l| if l == i { "\"shared\"" } else { "\"full\"" }).collect::<Vec<_>>().join(","));
        refused(&edited(false, |t| set(t, "indexer_types", &shared_at(7))), "a latent attention layer that shares another's indexer");
        refused(&edited(false, |t| set(t, "indexer_types", r#"["full"]"#)), "1 indexer_types for 45 layers");
        refused(&edited(false, |t| set(t, "index_topk", "2050")), "index_topk 2050 does not divide by index_kpool 4");
        refused(&edited(false, |t| set(t, "n_group", "8")), "group-limited routing");
        refused(&edited(false, |t| set(t, "topk_group", "2")), "group-limited routing");
        refused(&edited(false, |t| set(t, "scoring_func", r#""softmax""#)), "routing scores other than sigmoid");
        refused(&edited(false, |t| set(t, "topk_method", r#""greedy""#)), "a selection other than noaux_tc");
        refused(&edited(false, |t| set(t, "moe_router_dtype", r#""bfloat16""#)), "not computed in float32");
        refused(&edited(false, |t| set(t, "num_experts_per_tok", "289")), "289 experts a token of 288");
        refused(&edited(false, |t| set(t, "n_shared_experts", "0")), "without a shared expert");
        refused(&edited(false, |t| set(t, "index_kpool", "0")), "index_topk 2048 does not divide by index_kpool 0");
        refused(
            &edited(false, |t| {
                set(t, "linear_attn_config", r#"{"safe_gate": false, "gate_lower_bound": null, "num_heads": 64, "head_dim": 128, "short_conv_kernel_size": 4}"#)
            }),
            "a forget gate without its lower bound",
        );
        // (transformers' flat key, with no linear_attn_config to fill it, left null is the gate without a bound too)
        refused(
            &edited(false, |t| {
                t.retain(|(k, _)| k != "linear_attn_config");
                set(t, "linear_lower_bound", "null")
            }),
            "a forget gate without its lower bound",
        );
        refused(
            &edited(false, |t| set(t, "linear_attn_config", r#"{"num_heads": 0, "head_dim": 128, "short_conv_kernel_size": 4}"#)),
            "Kimi delta attention of 0 heads",
        );
        refused(&edited(true, |t| set(t, "quantization_config", r#"{"quant_method": "compressed-tensors"}"#)), "a quantization of type \"compressed-tensors\"");
        refused(
            &edited(true, |t| set(t, "quantization_config", r#"{"quant_method": "fp8", "fmt": "e5m2", "weight_block_size": [128, 128]}"#)),
            "FP8 of format e5m2",
        );
        refused(
            &edited(true, |t| set(t, "quantization_config", r#"{"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128]}"#)),
            "no weight_block_size of two sizes",
        );
        refused(&edited(false, |t| set(t, "hc_mult", "0")), "hc_mult 0");
        // what is not refused: the rope fields the latent attention never reads, a quantization config that is not there (a bf16 model), a gate bound left null with the safe gate on (-5, transformers'), and the
        // layer schedules as transformers writes them (indexed_attention or full_attention for the latent layers; none: every fourth layer, and the first three dense)
        let ok = |text: &str| ConfigGlm::parse(text, "x", None).unwrap();
        ok(&edited(false, |t| set(t, "indexer_rope_interleave", "false")));
        assert_eq!(ok(&edited(true, |t| t.retain(|(k, _)| k != "quantization_config"))).fp8_block, None);
        assert_eq!(
            ok(&edited(false, |t| set(
                t,
                "linear_attn_config",
                r#"{"gate_lower_bound": null, "num_heads": 64, "head_dim": 128, "short_conv_kernel_size": 4}"#
            )))
            .kda
            .bound,
            -5.0
        );
        let c = parse();
        let named = |name: &str| {
            format!(
                "[{}]",
                c.kinds
                    .iter()
                    .map(|k| if *k == GlmKind::Kda { "\"linear_attention\"".to_string() } else { format!("\"{name}\"") })
                    .collect::<Vec<_>>()
                    .join(",")
            )
        };
        for name in ["indexed_attention", "full_attention"] {
            assert_eq!(ok(&edited(false, |t| set(t, "layer_types", &named(name)))).kinds, c.kinds);
        }
        let by_default =
            ok(&edited(false, |t| t.retain(|(k, _)| !["layer_types", "mlp_layer_types", "indexer_types", "linear_attn_config"].contains(&k.as_str()))));
        assert_eq!((by_default.kinds.clone(), by_default.moe.clone(), by_default.kda.heads), (c.kinds.clone(), c.moe.clone(), 64));
        // (a `shared` indexer at a layer of Kimi delta attention is never read)
        ok(&edited(false, |t| set(t, "indexer_types", &shared_at(44))));
        assert_eq!(
            ok(&edited(false, |t| {
                set(t, "linear_attn_config", r#"{"head_dim": 32}"#);
                set(t, "linear_num_heads", "4");
                t.retain(|(k, _)| k != "layer_types")
            }))
            .kda,
            KdaDims { heads: 4, dim: 32, conv_kernel: 4, bound: -5.0 }
        );
    }

    fn l(n: usize, rest: &str) -> String {
        format!("model.language_model.layers.{n}.{rest}")
    }

    /// The weight map against the pinned checkpoint's 76,108 tensors: every tensor the engine wants is there with its dtype and shape (each matrix bf16 or FP8 with its block scales), each is accounted for once, and the only
    /// ones it does not take are the vision tower's 347 and the MTP layer's 1,760.
    #[test]
    fn the_weight_map_against_the_checkpoint() {
        let c = parse();
        let have = tensors();
        assert_eq!(have.len(), 76_108);
        assert_eq!(c.check(&have, &HashSet::new()).unwrap(), 347 + 1_760);
        let skipped: Vec<&str> = have.iter().filter(|t| c.skipped(&t.0)).map(|t| t.0.as_str()).collect();
        assert_eq!(
            (skipped.iter().filter(|n| n.starts_with("model.visual.")).count(), skipped.iter().filter(|n| n.starts_with(&l(45, ""))).count()),
            (347, 1_760)
        );
        // every tensor once: the matrices' parts and their block scales, the tensors kept, the skipped; their number is the checkpoint's
        let by: HashMap<&str, (&str, &[u64])> = have.iter().map(|(n, d, s)| (n.as_str(), (d.as_str(), s.as_slice()))).collect();
        assert_eq!(by.len(), have.len());
        let (packs, kept) = (c.packs(), c.kept());
        let mut names: Vec<String> = packs.iter().flat_map(|p| p.parts.iter().map(|x| x.0.clone())).collect();
        let fp8: Vec<String> = names.iter().filter(|n| by[n.as_str()].0 == "F8_E4M3").cloned().collect();
        names.extend(fp8.iter().map(|n| format!("{n}_scale_inv")));
        names.extend(kept.iter().map(|t| t.name.clone()));
        let once: HashSet<&str> = names.iter().map(String::as_str).collect();
        assert_eq!((names.len(), once.len(), names.len() + skipped.len()), (74_001, 74_001, 76_108));
        assert!(once.iter().all(|n| by.contains_key(n) && !c.skipped(n)));
        // the dtypes that are the checkpoint's own: the small F32 parameters (not cast to bf16 in the plan), the router in bf16, the mHC projection in bf16
        let dtype = |n: &str| by[n].0;
        assert_eq!(
            [
                dtype(&l(0, "self_attn.A_log")),
                dtype(&l(0, "self_attn.dt_bias")),
                dtype(&l(3, "mlp.gate.e_score_correction_bias")),
                dtype(&l(0, "hc_attn_base")),
                dtype(&l(0, "hc_ffn_scale")),
                dtype(&l(3, "mlp.gate.weight")),
                dtype(&l(0, "hc_attn_fn")),
                dtype(&l(0, "self_attn.q_conv1d.weight"))
            ],
            ["F32", "F32", "F32", "F32", "F32", "BF16", "BF16", "BF16"]
        );
        assert!(kept.iter().all(|t| dtype(&t.name) == t.dtype), "a kept tensor's dtype is not the checkpoint's");
    }

    /// The plan: which matrices are packed (the experts' gate and up as one pack, a pack of the down; the Kimi delta attention's q, k and v as one, the latent attention's, the MLPs, the embedding and the head), which are
    /// kept (norms, convolutions, the fp32 parameters, the routers, the mHC projections and the whole indexer), and what the FP8 source holds: the experts, the MLPs and four of the latent attention's, 307.0 billion
    /// parameters, the rest of the matrices (6.13 billion) bf16.
    #[test]
    fn the_packs() {
        let c = parse();
        let have = tensors();
        let dtype: HashMap<&str, &str> = have.iter().map(|t| (t.0.as_str(), t.1.as_str())).collect();
        let packs = c.packs();
        // Kimi delta layers have 7 packs, latent attention layers 5, each with its MLP's 2; a mixture-of-experts layer's 288 experts 2 each; the embedding and the head
        assert_eq!(packs.len(), 3 * 9 + 31 * (7 + 2 + 576) + 11 * (5 + 2 + 576) + 2);
        assert!(packs.iter().all(Pack35::packable));
        assert_eq!(packs.iter().map(|p| p.parts.len()).sum::<usize>(), 36_786);
        let shape = |key: &str| packs.iter().find(|p| p.key == key).map(|p| (p.rows(), p.cols)).unwrap_or_else(|| panic!("no pack {key}"));
        assert_eq!(
            [
                shape(&l(0, "self_attn.q_proj")),
                shape(&l(0, "self_attn.o_proj")),
                shape(&l(0, "self_attn.f_a_proj")),
                shape(&l(0, "self_attn.f_b_proj")),
                shape(&l(0, "self_attn.g_a_proj")),
                shape(&l(0, "self_attn.g_b_proj")),
                shape(&l(0, "self_attn.b_proj"))
            ],
            [(24576, 4096), (4096, 8192), (128, 4096), (8192, 128), (128, 4096), (8192, 128), (64, 4096)]
        );
        assert_eq!(
            [
                shape(&l(3, "self_attn.q_a_proj")),
                shape(&l(3, "self_attn.q_b_proj")),
                shape(&l(3, "self_attn.kv_a_proj_with_mqa")),
                shape(&l(3, "self_attn.kv_b_proj")),
                shape(&l(3, "self_attn.o_proj"))
            ],
            [(1536, 4096), (16384, 1536), (512, 4096), (32768, 512), (4096, 16384)]
        );
        assert_eq!(
            [
                shape(&l(0, "mlp.gate_proj")),
                shape(&l(2, "mlp.down_proj")),
                shape(&l(3, "mlp.shared_experts.gate_proj")),
                shape(&l(44, "mlp.shared_experts.down_proj")),
                shape(&l(3, "mlp.experts.287.gate_proj")),
                shape(&l(44, "mlp.experts.0.down_proj"))
            ],
            [(24576, 4096), (4096, 12288), (4096, 4096), (4096, 2048), (4096, 4096), (4096, 2048)]
        );
        assert_eq!((shape("embed"), shape("lm_head")), ((154880, 4096), (154880, 4096)));
        let parts = |key: &str| {
            packs.iter().find(|p| p.key == key).unwrap().parts.iter().map(|(n, r)| (n.rsplit('.').nth(1).unwrap().to_string(), *r)).collect::<Vec<_>>()
        };
        assert_eq!(parts(&l(0, "self_attn.q_proj")), [("q_proj".to_string(), 8192), ("k_proj".to_string(), 8192), ("v_proj".to_string(), 8192)]);
        assert_eq!(parts(&l(3, "mlp.experts.5.gate_proj")), [("gate_proj".to_string(), 2048), ("up_proj".to_string(), 2048)]);
        assert_eq!(parts(&l(1, "mlp.gate_proj")), [("gate_proj".to_string(), 12288), ("up_proj".to_string(), 12288)]);
        // the FP8 source: the experts, the shared experts, the dense MLPs and four of a latent attention layer's five, every other matrix bf16; what the fixture holds sums to the checkpoint's own counts
        let (mut fp8_params, mut bf16_params, mut fp8_parts) = (0u64, 0u64, 0);
        for p in &packs {
            for (n, r) in &p.parts {
                let params = (*r * p.cols) as u64;
                let fp8 = dtype[n.as_str()] == "F8_E4M3";
                assert_eq!(fp8, is_fp8(&c, n), "{n}: {}", dtype[n.as_str()]);
                (fp8_params, bf16_params, fp8_parts) =
                    if fp8 { (fp8_params + params, bf16_params, fp8_parts + 1) } else { (fp8_params, bf16_params + params, fp8_parts) };
            }
        }
        assert_eq!((fp8_parts, fp8_params, bf16_params), (36_467, 307_023_052_800, 6_132_596_736));
        // with the MTP layer's 7,373,586,432 FP8 parameters they are the Hub's F8_E4M3 count
        assert_eq!(fp8_params + 7_373_586_432, 314_396_639_232);
        // kept: 748 tensors, 290 fp32 (the router's bias, the mHC base and scales, A_log, dt_bias), the rest bf16; 170.87 million bf16 parameters of them (82.19 million the indexer's) and 295,230 fp32
        let kept = c.kept();
        assert_eq!((kept.len(), kept.iter().filter(|t| t.dtype == "F32").count()), (748, 290));
        let params = |dtype: &str, f: &dyn Fn(&str) -> bool| {
            kept.iter().filter(|t| t.dtype == dtype && f(&t.name)).map(|t| t.shape.iter().product::<u64>()).sum::<u64>()
        };
        assert_eq!((params("BF16", &|_| true), params("F32", &|_| true), params("BF16", &|n| n.contains(".indexer."))), (170_867_200, 295_230, 82_190_592));
        let k = |name: &str| kept.iter().find(|t| t.name == name).map(|t| (t.dtype, t.shape.clone())).unwrap_or_else(|| panic!("{name} is not kept"));
        assert_eq!(k(&l(0, "self_attn.q_conv1d.weight")), ("BF16", vec![8192, 1, 4]));
        assert_eq!(k(&l(0, "self_attn.A_log")), ("F32", vec![64]));
        assert_eq!(k(&l(0, "self_attn.dt_bias")), ("F32", vec![8192]));
        assert_eq!(k(&l(0, "self_attn.o_norm.weight")), ("BF16", vec![128]));
        assert_eq!(k(&l(0, "hc_attn_fn")), ("BF16", vec![24, 16384]));
        assert_eq!((k(&l(7, "hc_ffn_base")), k(&l(7, "hc_ffn_scale"))), (("F32", vec![24]), ("F32", vec![3])));
        assert_eq!((k(&l(3, "self_attn.q_a_layernorm.weight")), k(&l(3, "self_attn.kv_a_layernorm.weight"))), (("BF16", vec![1536]), ("BF16", vec![512])));
        assert_eq!(
            (k(&l(3, "self_attn.indexer.wq_b.weight")), k(&l(3, "self_attn.indexer.wk.weight")), k(&l(3, "self_attn.indexer.weights_proj.weight"))),
            (("BF16", vec![4096, 1536]), ("BF16", vec![128, 4096]), ("BF16", vec![32, 4096]))
        );
        assert_eq!(
            (
                k(&l(3, "self_attn.indexer.index_kpool_compress_ape")),
                k(&l(3, "self_attn.indexer.index_kpool_compress_gate")),
                k(&l(3, "self_attn.indexer.k_norm.bias"))
            ),
            (("BF16", vec![4, 128]), ("BF16", vec![128, 4096]), ("BF16", vec![128]))
        );
        assert_eq!(
            (k(&l(3, "mlp.gate.weight")), k(&l(3, "mlp.gate.e_score_correction_bias")), k("model.language_model.norm.weight")),
            (("BF16", vec![288, 4096]), ("F32", vec![288]), ("BF16", vec![4096]))
        );
        // the dense layers have no router, the Kimi delta layers no indexer, the latent attention layers no convolution
        assert!(!kept.iter().any(|t| t.name.starts_with(&l(0, "mlp.gate."))
            || t.name.starts_with(&l(0, "self_attn.indexer"))
            || t.name.starts_with(&l(3, "self_attn.q_conv1d"))));
        // nothing is in both lists
        let names: HashSet<&str> = packs.iter().flat_map(|p| p.parts.iter().map(|x| x.0.as_str())).chain(kept.iter().map(|t| t.name.as_str())).collect();
        assert_eq!(names.len(), packs.iter().map(|p| p.parts.len()).sum::<usize>() + kept.len());
        // a matrix of rows the packer cannot take is kept as the checkpoint has it (here the experts', of an intermediate size of 2,088)
        let odd = ConfigGlm { experts: GlmMoeDims { inter: 2088, ..c.experts.clone() }, ..c.clone() };
        assert!(odd.packs().iter().all(|p| !p.key.contains(".mlp.experts.") || p.key.ends_with("down_proj") && p.cols == 2088 && p.rows() == 4096));
        assert!(odd.kept().iter().any(|t| t.name == l(3, "mlp.experts.0.gate_proj.weight") && t.shape == [2088, 4096]));
        // the MTP layer's tensors are in neither list, nor the vision tower's
        assert!(!names.iter().any(|n| n.starts_with(&l(45, "")) || n.starts_with("model.visual.")));
    }

    #[test]
    fn a_checkpoint_that_is_not_the_config_is_refused() {
        let c = parse();
        let have = tensors();
        let says = |edit: &dyn Fn(&mut Index), what: &str| {
            let mut h = have.clone();
            edit(&mut h);
            let e = c.check(&h, &HashSet::new()).unwrap_err().msg;
            assert!(e.contains(what), "{e:?} does not say {what:?}");
        };
        let at = |h: &Index, n: &str| h.iter().position(|t| t.0 == n).unwrap_or_else(|| panic!("{n}"));
        let conv = l(0, "self_attn.q_conv1d.weight");
        says(
            &|h| {
                let i = at(h, &conv);
                h.remove(i);
            },
            &format!("no {conv}"),
        );
        says(
            &|h| {
                let i = at(h, &conv);
                h[i].2 = vec![8192, 1, 3];
            },
            "[8192, 1, 3], where BF16 [8192, 1, 4]",
        );
        says(
            &|h| {
                let i = at(h, &l(0, "self_attn.A_log"));
                h[i].1 = "F16".into();
            },
            "where F32 [64]",
        );
        says(
            &|h| {
                let i = at(h, &l(3, "mlp.experts.287.up_proj.weight"));
                h.remove(i);
            },
            &format!("no {}", l(3, "mlp.experts.287.up_proj.weight")),
        );
        // a matrix: FP8 without its scales, with scales of another grid or dtype, of another dtype or shape
        let up = l(3, "mlp.experts.7.up_proj.weight");
        says(
            &|h| {
                let i = at(h, &format!("{up}_scale_inv"));
                h.remove(i);
            },
            &format!("no {up}_scale_inv, the block scales of {up}"),
        );
        says(
            &|h| {
                let i = at(h, &format!("{up}_scale_inv"));
                h[i].2 = vec![16, 31];
            },
            &format!("{up}_scale_inv: F32 [16, 31], where F32 [16, 32] is wanted"),
        );
        says(
            &|h| {
                let i = at(h, &format!("{up}_scale_inv"));
                h[i].1 = "BF16".into();
            },
            &format!("{up}_scale_inv: BF16 [16, 32], where F32 [16, 32] is wanted"),
        );
        says(
            &|h| {
                let i = at(h, &up);
                h[i].1 = "F16".into();
            },
            &format!("{up}: F16, where BF16 or F8_E4M3 is wanted"),
        );
        says(
            &|h| {
                let i = at(h, &up);
                h[i].2 = vec![2048, 4000];
            },
            &format!("{up}: F8_E4M3 [2048, 4000], where [2048, 4096] is wanted"),
        );
        // (the same matrix bf16: no scales wanted, a scale left in the checkpoint is not a tensor of the model)
        let mut h = have.clone();
        let i = at(&h, &up);
        h[i].1 = "BF16".into();
        assert!(c.check(&h, &HashSet::new()).unwrap_err().msg.contains(&format!("{up}_scale_inv: not a tensor of this model")));
        h.retain(|t| t.0 != format!("{up}_scale_inv"));
        assert_eq!(c.check(&h, &HashSet::new()).unwrap(), 2_107);
        // a config without the FP8 blocks cannot take FP8 tensors
        let bf16 = ConfigGlm { fp8_block: None, ..c.clone() };
        assert!(bf16.check(&have, &HashSet::new()).unwrap_err().msg.contains("config.json gives no weight_block_size for its block scales"));
        // an FP8 model's tensors against another's: a tensor of no layer, a layer past the MTP head, more layers than the checkpoint has
        says(&|h| h.push(("model.embed_tokens.weight".into(), "BF16".into(), vec![154880, 4096])), "model.embed_tokens.weight: not a tensor of this model");
        says(
            &|h| h.push((l(46, "input_layernorm.weight"), "BF16".into(), vec![4096])),
            &format!("{}: not a tensor of this model", l(46, "input_layernorm.weight")),
        );
        let more = ConfigGlm { layers: 46, kinds: [c.kinds.clone(), vec![GlmKind::Kda]].concat(), moe: [c.moe.clone(), vec![true]].concat(), ..c.clone() };
        assert!(more.check(&have, &HashSet::new()).unwrap_err().msg.contains(&format!("no {}", l(45, "self_attn.q_proj.weight"))));
        // a packed matrix is held as its buffers, a fused one under its first part's name; its scales are not saved
        let key = l(3, "mlp.experts.7.gate_proj");
        let mut h = have.clone();
        h.retain(|t| !t.0.starts_with(&key) || !(t.0.ends_with("gate_proj.weight") || t.0.ends_with("gate_proj.weight_scale_inv")));
        h.retain(|t| t.0 != format!("{}_scale_inv", l(3, "mlp.experts.7.up_proj.weight")) && t.0 != l(3, "mlp.experts.7.up_proj.weight"));
        assert!(c.check(&h, &HashSet::new()).is_err());
        h.extend(["data", "blocks", "block_base"].map(|s| (format!("{key}.glyd_{s}"), "U8".to_string(), vec![1u64])));
        let held: HashSet<String> = [key.clone(), l(3, "mlp.experts.7.up_proj")].into();
        assert_eq!(c.check(&h, &held).unwrap(), 2_107);
        assert!(c.check(&h, &HashSet::new()).is_err());
    }

    /// The whole model's bytes in the two block layouts the engine runs, from the real config (`cargo test glm::tests::the_sizes -- --nocapture` prints the table): what is packed by family (313.16 billion matrix weights),
    /// what stays as the checkpoint has it (bf16 and fp32), and the routed experts' share.
    #[test]
    fn the_sizes_at_both_levels() {
        use glyd_gpu::pack::{INT5_STEP_BYTES, INT6_STEP_BYTES};
        let c = parse();
        // a pack's bytes: its steps (64 rows by 16 columns) of data, 128 bytes of scales to each pair of steps, one base word more than steps
        let bytes = |p: &Pack35, step: usize| {
            let steps = (p.rows() / 64 * (p.cols / 16)) as u64;
            steps * step as u64 + steps / 2 * 128 + (steps + 1) * 4
        };
        let family = |key: &str| -> &'static str {
            let layer = key.split('.').nth(3).and_then(|i| i.parse::<usize>().ok());
            match (key, layer) {
                ("embed", _) => "embedding",
                ("lm_head", _) => "lm_head",
                _ if key.contains(".mlp.experts.") => "routed experts",
                _ if key.contains(".mlp.shared_experts.") => "shared experts",
                _ if key.contains(".mlp.") => "dense MLPs",
                (_, Some(l)) if c.kinds[l] == GlmKind::Kda => "Kimi delta attention",
                _ => "latent attention",
            }
        };
        let mut table: Vec<(&str, u64, u64, u64)> = Vec::new();
        for p in c.packs() {
            let (f, w) = (family(&p.key), (p.rows() * p.cols) as u64);
            let (a, b) = (bytes(&p, INT6_STEP_BYTES), bytes(&p, INT5_STEP_BYTES));
            match table.iter_mut().find(|r| r.0 == f) {
                Some(r) => (r.1, r.2, r.3) = (r.1 + w, r.2 + a, r.3 + b),
                None => table.push((f, w, a, b)),
            }
        }
        let kept = c.kept();
        let kept_bytes =
            |dtype: &str| kept.iter().filter(|t| t.dtype == dtype).map(|t| t.shape.iter().product::<u64>() * if dtype == "F32" { 4 } else { 2 }).sum::<u64>();
        let (bf16, f32) = (kept_bytes("BF16"), kept_bytes("F32"));
        let (w, i6, i5) = table.iter().fold((0, 0, 0), |a, r| (a.0 + r.1, a.1 + r.2, a.2 + r.3));
        println!("{:24} {:>16} {:>16} {:>16}", "", "weights", "i6 (kestrel)", "i5 (swift)");
        for r in &table {
            println!("{:24} {:>16} {:>16} {:>16}", r.0, r.1, r.2, r.3);
        }
        println!("{:24} {:>16} {:>16} {:>16}", "packed", w, i6, i5);
        println!("{:24} {:>16} {:>16} {:>16}", "kept bf16, fp32", bf16, f32, "");
        println!("{:24} {:>16} {:>16} {:>16}", "the model", "", i6 + bf16 + f32, i5 + bf16 + f32);
        let by = |name: &str| *table.iter().find(|r| r.0 == name).unwrap();
        assert_eq!(by("Kimi delta attention"), ("Kimi delta attention", 4_679_270_400, 3_820_186_552, 3_235_277_752));
        assert_eq!(by("latent attention"), ("latent attention", 1_291_845_632, 1_054_671_068, 893_190_364));
        assert_eq!(by("dense MLPs"), ("dense MLPs", 452_984_832, 369_819_672, 313_196_568));
        assert_eq!(by("shared experts"), ("shared experts", 1_056_964_608, 862_912_848, 730_792_272));
        assert_eq!(by("routed experts"), ("routed experts", 304_405_807_104, 248_518_900_224, 210_468_174_336));
        assert_eq!(by("embedding"), ("embedding", 634_388_480, 517_918_724, 438_620_164));
        assert_eq!((w, i6, i5), (313_155_649_536, 255_662_327_812, 216_517_871_620));
        assert_eq!((bf16, f32), (341_734_400, 1_180_920));
        // (the bits a weight of the packed matrices take, all in, are sizing::bits's 6.53 and 5.53; the routed experts are 97.1% of the model at either level; an expert's two packs take 20.5 and 17.4 MB)
        assert_eq!(((i6 as f64 * 8.0 / w as f64 * 100.0).round(), (i5 as f64 * 8.0 / w as f64 * 100.0).round()), (653.0, 553.0));
        let e = by("routed experts");
        assert!((e.2 as f64 / (i6 + bf16 + f32) as f64 - 0.9708).abs() < 1e-4 && (e.3 as f64 / (i5 + bf16 + f32) as f64 - 0.9705).abs() < 1e-4);
        let expert: Vec<Pack35> = c.expert_packs(3);
        assert_eq!(
            (
                expert.len(),
                expert.iter().map(|p| bytes(p, INT6_STEP_BYTES)).sum::<u64>() * 42,
                expert.iter().map(|p| bytes(p, INT5_STEP_BYTES)).sum::<u64>() * 42
            ),
            (2 * 288, e.2, e.3)
        );
        assert_eq!(
            (bytes(&expert[0], INT6_STEP_BYTES) + bytes(&expert[1], INT6_STEP_BYTES), bytes(&expert[0], INT5_STEP_BYTES) + bytes(&expert[1], INT5_STEP_BYTES)),
            (20_545_544, 17_399_816)
        );
    }
}
