//! Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`, model_type qwen4_exp, 2026) as the engine reads it: the config (12 times three Gated DeltaNet layers and one Qwen Sparse Attention layer, a mixture of 512
//! experts in every layer, four residual streams joined by Gated Residuals, the n-gram embedding at layer 1, one MTP layer) and the tensors it loads. The reference: transformers 5.19.0's modeling_qwen4_exp.py;
//! research/qwen38-flash-next.md has the numbers.

use crate::ffi::{err, Error, Result};
use crate::qwen35::Pack35;
use crate::weights::{eos_ids, num};
use glyd_gpu::json::{self, Value};
use std::path::Path;

/// A layer's token mixer: Gated DeltaNet (`linear_attention`) or Qwen Sparse Attention (`full_attention`, transformers' `indexed_attention`: gated attention over the tokens its indexer picks).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QxKind {
    Gdn,
    Attn,
}

/// The n-gram embedding (PLE) of one layer: `ngram` - 1 orders (bigrams, trigrams) of `heads` hashed heads each, rows of `dim / (heads (ngram - 1))` values; the dilated convolution's kernel.
#[derive(Clone, Debug, PartialEq)]
pub struct PleDims {
    pub layer: usize,
    pub dim: usize,
    pub kernel: usize,
    pub ngram: usize,
    pub heads: usize,
}

/// What the engine reads of a Qwen3.8-Flash-Next config.json.
#[derive(Clone, Debug)]
pub struct ConfigQx {
    pub hidden: usize,
    pub layers: usize,
    pub vocab: usize,
    pub eps: f32,
    pub max_pos: usize,
    /// The tokens that end a reply (generation_config's, else the config's).
    pub eos: Vec<i64>,
    /// The config's own end token: what the n-gram hashes put where a context token is missing.
    pub eos_ngram: u32,
    pub kinds: Vec<QxKind>,
    /// Gated DeltaNet: key heads, value heads (128 channels each), the convolution's taps, the output gate a sigmoid (else silu).
    pub nk: usize,
    pub nv: usize,
    pub conv: usize,
    pub gate_sigmoid: bool,
    /// Attention: query heads, key-value heads, a head's channels, the rotary's channels, its base.
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary: usize,
    pub theta: f64,
    /// The indexer: query heads of `ix_dim`, tokens attended (`ix_budget`, in blocks of `ix_block`).
    pub ix_heads: usize,
    pub ix_dim: usize,
    pub ix_budget: usize,
    pub ix_block: usize,
    /// The Gated Residuals: streams, the mixing weights' bottleneck.
    pub hc: usize,
    pub hc_rank: usize,
    /// The experts: count, per token, an expert's neurons, the shared expert's, the chosen weights renormalised.
    pub experts: usize,
    pub top_k: usize,
    pub inter: usize,
    pub shared_inter: usize,
    pub norm_topk: bool,
    pub ple: Option<PleDims>,
    pub mtp_layers: usize,
    pub prefix: &'static str,
}

/// Whether a config.json is a Qwen3.8-Flash-Next's.
pub fn is_qx(text: &str) -> bool {
    crate::runtime::config::family_of(text) == Some(crate::support::Family::FlashNext)
}

impl ConfigQx {
    pub fn load(dir: &Path) -> Result<ConfigQx> {
        let p = dir.join("config.json");
        let text = std::fs::read_to_string(&p).map_err(|e| Error(format!("{}: {e}", p.display())))?;
        let generation = std::fs::read_to_string(dir.join("generation_config.json")).ok();
        ConfigQx::parse(&text, &p.display().to_string(), generation.as_deref())
    }

    pub fn parse(text: &str, label: &str, generation: Option<&str>) -> Result<ConfigQx> {
        let v = json::parse(text).map_err(|e| Error(format!("{label}: {e}")))?;
        if v.get("model_type").and_then(Value::as_str) != Some("qwen4_exp") {
            return err(format!("{label}: not a qwen4_exp model"));
        }
        let t = v.get("text_config").ok_or_else(|| Error(format!("{label}: no text_config")))?;
        let get = |k: &str| t.get(k).filter(|x| **x != Value::Null).ok_or_else(|| Error(format!("{label}: no {k}")));
        let u = |k: &str| -> Result<usize> { get(k)?.as_u64().map(|x| x as usize).ok_or_else(|| Error(format!("{label}: {k}: not an integer"))) };
        let u_or = |k: &str, d: usize| -> Result<usize> {
            if t.get(k).is_some_and(|x| *x != Value::Null) {
                u(k)
            } else {
                Ok(d)
            }
        };
        let text_is = |k: &str, want: &str| t.get(k).and_then(Value::as_str).is_none_or(|a| a == want);
        let refuse = |what: &str| err(format!("{label}: {what}, which the engine does not run"));
        if !text_is("hidden_act", "silu") {
            return refuse("an activation other than silu");
        }
        if t.get("attention_bias").and_then(Value::as_bool) == Some(true) {
            return refuse("biases in the attention");
        }
        let layers = u("num_hidden_layers")?;
        if layers > crate::plan::MAX_LAYERS {
            return err(format!("{label}: num_hidden_layers {layers}, where no model has more than {}", crate::plan::MAX_LAYERS));
        }
        let kinds: Vec<QxKind> = match t.get("layer_types").and_then(Value::as_array) {
            Some(a) if a.len() == layers => a
                .iter()
                .map(|k| match k.as_str() {
                    Some("linear_attention") => Ok(QxKind::Gdn),
                    Some("full_attention" | "indexed_attention") => Ok(QxKind::Attn),
                    other => err(format!("{label}: a layer of type {other:?}")),
                })
                .collect::<Result<_>>()?,
            Some(a) => return err(format!("{label}: {} layer_types for {layers} layers", a.len())),
            None => {
                (0..layers).map(|i| if (i + 1) % u_or("full_attention_interval", 4).unwrap_or(4).max(1) == 0 { QxKind::Attn } else { QxKind::Gdn }).collect()
            }
        };
        let rope = t.get("rope_parameters");
        let rp = |k: &str| rope.and_then(|r| r.get(k)).filter(|x| **x != Value::Null);
        let theta = match rp("rope_theta").or_else(|| t.get("rope_theta")) {
            Some(x) => num(x, "rope_theta")?,
            None => return err(format!("{label}: no rope_theta")),
        };
        let factor = match rp("partial_rotary_factor").or_else(|| t.get("partial_rotary_factor")) {
            Some(x) => num(x, "partial_rotary_factor")?,
            None => 1.0,
        };
        if rp("rope_type").and_then(Value::as_str).is_some_and(|r| r != "default") {
            return refuse("a rotary other than the default");
        }
        let head_dim = u("head_dim")?;
        let rotary = (head_dim as f64 * factor) as usize;
        let gate = t.get("output_gate_type").and_then(Value::as_str).unwrap_or("silu");
        let ple = match t.get("ple_layer_ids").and_then(Value::as_u64s) {
            Some(ids) if ids.len() > 1 => return refuse("n-gram embeddings at more than one layer"),
            Some(ids) if ids.len() == 1 => {
                let layer = ids[0] as usize;
                if layer == 0 || layer > layers || kinds[layer - 1] != QxKind::Gdn {
                    return err(format!("{label}: ple_layer_ids {layer}: not a linear attention layer"));
                }
                let (ngram, heads) = (u_or("ngram_size", 3)?, u_or("heads_per_ngram", 8)?);
                let dim = u_or("ple_embed_dim", u("hidden_size")?)?;
                // (the numbers are a file's: an n-gram size of 0, or sizes whose product does not fit, is refused in words like any other)
                let over = heads.checked_mul(ngram.saturating_sub(1)).filter(|&n| ngram >= 2 && n > 0);
                if over.is_none_or(|n| !dim.is_multiple_of(n)) {
                    return err(format!("{label}: an n-gram embedding of {dim} over {heads} heads for each of n-grams of {ngram}"));
                }
                Some(PleDims { layer: layer - 1, dim, kernel: u_or("ple_conv_kernel_size", 4)?, ngram, heads })
            }
            _ => None,
        };
        let eos_ngram = match t.get("eos_token_id") {
            Some(Value::Array(a)) => a.first().and_then(Value::as_u64),
            Some(x) => x.as_u64(),
            None => None,
        }
        .ok_or_else(|| Error(format!("{label}: no eos_token_id")))? as u32;
        let mut eos = eos_ids(t, generation);
        if eos.is_empty() {
            eos.push(eos_ngram as i64);
        }
        let mtp_layers = match t.get("mtp_num_hidden_layers") {
            Some(Value::Null) | None => t.get("mtp").and_then(|m| m.get("num_hidden_layers")).and_then(Value::as_u64).unwrap_or(0) as usize,
            Some(x) => x.as_u64().unwrap_or(0) as usize,
        };
        let c = ConfigQx {
            hidden: u("hidden_size")?,
            layers,
            vocab: u("vocab_size")?,
            eps: num(get("rms_norm_eps")?, "rms_norm_eps")? as f32,
            max_pos: u("max_position_embeddings")?,
            eos,
            eos_ngram,
            kinds,
            nk: u("linear_num_key_heads")?,
            nv: u("linear_num_value_heads")?,
            conv: u_or("linear_conv_kernel_dim", 4)?,
            gate_sigmoid: gate == "sigmoid",
            heads: u("num_attention_heads")?,
            kv_heads: u("num_key_value_heads")?,
            head_dim,
            rotary,
            theta,
            ix_heads: u("indexer_n_heads")?,
            ix_dim: u("indexer_head_dim")?,
            ix_budget: u("indexer_budget")?,
            ix_block: u("indexer_compress_ratio")?,
            hc: u_or("hc_count", 4)?,
            hc_rank: u_or("hc_lowrank", 320)?,
            experts: u("num_experts")?,
            top_k: u("num_experts_per_tok")?,
            inter: u("moe_intermediate_size")?,
            shared_inter: u("shared_expert_intermediate_size")?,
            norm_topk: t.get("norm_topk_prob").and_then(Value::as_bool).unwrap_or(true),
            ple,
            mtp_layers,
            prefix: "model.language_model.",
        };
        // (what the kernels hold to: model_qwen4x.rs's checks name the limits)
        if u_or("linear_key_head_dim", 128)? != 128 || u_or("linear_value_head_dim", 128)? != 128 || c.conv != 4 || c.nk == 0 || !c.nv.is_multiple_of(c.nk) {
            return refuse("Gated DeltaNet heads other than 128 channels with a convolution of 4");
        }
        if c.head_dim != 256 || c.kv_heads == 0 || !c.heads.is_multiple_of(c.kv_heads) || c.heads / c.kv_heads > 16 || c.rotary > 128 || c.rotary % 2 != 0 {
            return refuse(&format!("attention of {} heads of {} over {} key-value heads, a rotary of {}", c.heads, c.head_dim, c.kv_heads, c.rotary));
        }
        if c.ix_dim != 128
            || u_or("indexer_kv_heads", 1)? != 1
            || c.ix_block != 4
            || c.ix_budget == 0
            || !c.ix_budget.is_multiple_of(c.ix_block)
            || c.ix_budget / c.ix_block > 1024
            || c.rotary > c.ix_dim
        {
            return refuse("an indexer other than one key head of 128 over blocks of 4 tokens");
        }
        if c.hc != 4 || c.hidden > 4096 || !c.hidden.is_multiple_of(64) {
            return refuse(&format!("{} streams of {}", c.hc, c.hidden));
        }
        if c.ple.as_ref().is_some_and(|p| p.ngram != 3 || p.kernel != 4) {
            return refuse("an n-gram embedding other than bigrams and trigrams with a convolution of 4");
        }
        Ok(c)
    }

    /// What this engine reads of a model, from the model's description (`runtime::describe`: its layer kinds, and the experts' widths from the
    /// shapes it says), with what no description holds from `own` (this engine's reading of the same config: the end tokens, with the generation
    /// config's, the longest context, the tensors' prefix). Held to `own` (`agrees`), it is the check that the description says everything the
    /// engine runs on, before the engine's work moves onto the description's kinds.
    pub fn lowered(d: &crate::runtime::describe::Described, own: &ConfigQx) -> Result<ConfigQx> {
        use crate::runtime::describe::Source;
        use crate::runtime::kinds::{Body, Ffn, Mixer, Residual};
        let m = &d.desc;
        let mixers: Vec<&Body> = m.blocks.iter().map(|b| &b.body).filter(|b| matches!(b, Body::Mixer(_))).collect();
        let kinds: Vec<QxKind> = mixers
            .iter()
            .map(|b| match b {
                Body::Mixer(Mixer::Linear(_)) => Ok(QxKind::Gdn),
                Body::Mixer(Mixer::Attention(_)) => Ok(QxKind::Attn),
                b => err(format!("Qwen3.8-Flash-Next: a {b:?}, where a layer is Gated DeltaNet or attention")),
            })
            .collect::<Result<_>>()?;
        let lin = mixers.iter().find_map(|b| if let Body::Mixer(Mixer::Linear(l)) = b { Some(l) } else { None });
        let attn = mixers.iter().find_map(|b| if let Body::Mixer(Mixer::Attention(a)) = b { Some(a) } else { None });
        let (lin, attn) = lin.zip(attn).ok_or_else(|| Error("Qwen3.8-Flash-Next: no Gated DeltaNet layer or no attention layer".into()))?;
        let (ix, rope) =
            attn.indexer.as_ref().zip(attn.rope.as_ref()).ok_or_else(|| Error("Qwen3.8-Flash-Next: an attention without its indexer or rotary".into()))?;
        let moe = m.blocks.iter().find_map(|b| if let Body::Ffn(Ffn::Moe(e)) = &b.body { Some(e) } else { None });
        let gr = m.blocks.iter().find_map(|b| if let Residual::GatedStreams(g) = b.residual { Some(g) } else { None });
        let (moe, gr) = moe.zip(gr).ok_or_else(|| Error("Qwen3.8-Flash-Next: no experts or no gated residuals".into()))?;
        // (the experts' widths: the shapes said of the first layer's tensors of each role)
        let width = |role: &str, at: usize| -> Result<usize> {
            d.blocks
                .iter()
                .flatten()
                .find(|(r, _)| r == role)
                .and_then(|(_, s)| if let Source::Tensor(n) = s { d.shapes.get(n) } else { None })
                .and_then(|s| s.get(at).copied())
                .ok_or_else(|| Error(format!("Qwen3.8-Flash-Next: no shape said of {role}")))
        };
        let mut layer = 0;
        let mut ple = None;
        for b in &m.blocks {
            match &b.body {
                Body::Ngram(n) => ple = Some(PleDims { layer, dim: n.dim, kernel: n.conv, ngram: n.order, heads: n.heads }),
                Body::Mixer(_) => layer += 1,
                _ => {}
            }
        }
        let ngram = m.blocks.iter().find_map(|b| if let Body::Ngram(n) = &b.body { Some(n.eos) } else { None });
        Ok(ConfigQx {
            hidden: m.hidden,
            layers: kinds.len(),
            vocab: m.vocab,
            eps: m.final_norm.eps,
            max_pos: own.max_pos,
            eos: own.eos.clone(),
            eos_ngram: ngram.unwrap_or(own.eos_ngram),
            kinds,
            nk: lin.k_heads,
            nv: lin.v_heads,
            conv: lin.conv,
            gate_sigmoid: lin.gate_act == crate::runtime::kinds::Act::Sigmoid,
            heads: attn.heads,
            kv_heads: attn.kv_heads,
            head_dim: attn.head_dim,
            rotary: rope.dim,
            theta: rope.theta,
            ix_heads: ix.heads,
            ix_dim: ix.head_dim,
            ix_budget: ix.top_k,
            ix_block: ix.block,
            hc: gr.streams,
            hc_rank: gr.lowrank,
            experts: moe.experts,
            top_k: moe.router.top_k,
            inter: width("experts_down", 2)?,
            shared_inter: if moe.shared.is_some() { width("shared_down", 1)? } else { 0 },
            norm_topk: moe.router.renorm,
            ple,
            mtp_layers: d.mtp_layers,
            prefix: own.prefix,
        })
    }

    /// `own` where its config's description, lowered (`lowered`), reads the same model: else the first difference named.
    pub fn agrees(own: ConfigQx, d: &crate::runtime::describe::Described) -> Result<ConfigQx> {
        let (a, b) = (format!("{own:?}"), format!("{:?}", ConfigQx::lowered(d, &own)?));
        if a == b {
            return Ok(own);
        }
        let at = a.bytes().zip(b.bytes()).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
        let near = |s: &str| s.get(at.saturating_sub(40)..(at + 40).min(s.len())).unwrap_or("").to_string();
        err(format!(
            "Qwen3.8-Flash-Next: its description and this engine's reading of its config differ (at {:?}, where the engine has {:?})",
            near(&b),
            near(&a)
        ))
    }

    pub fn layers_of(&self, kind: QxKind) -> usize {
        self.kinds.iter().filter(|&&k| k == kind).count()
    }

    /// Each layer's index among the layers of its kind (a GDN layer's state, an attention layer's cache).
    pub fn schedule(&self) -> Vec<(QxKind, usize)> {
        let (mut g, mut a) = (0, 0);
        self.kinds
            .iter()
            .map(|&k| {
                let n = if k == QxKind::Gdn { &mut g } else { &mut a };
                *n += 1;
                (k, *n - 1)
            })
            .collect()
    }

    /// Layer `l`'s tensor names' start: the model's layers, then the MTP head's (`l` = `layers`).
    pub fn layer_name(&self, l: usize) -> String {
        if l < self.layers {
            format!("{}layers.{l}", self.prefix)
        } else {
            format!("mtp.layers.{}", l - self.layers)
        }
    }

    pub fn kind_of(&self, l: usize) -> QxKind {
        self.kinds.get(l).copied().unwrap_or(QxKind::Attn)
    }

    /// The GDN's convolution width (q, k and v) and value width.
    pub fn conv_dim(&self) -> usize {
        (2 * self.nk + self.nv) * 128
    }

    pub fn value_dim(&self) -> usize {
        self.nv * 128
    }

    /// An attention layer's q, k and v product row: each query head's query and gate, then the keys, then the values.
    pub fn qkv_dim(&self) -> usize {
        2 * self.heads * self.head_dim + 2 * self.kv_heads * self.head_dim
    }

    /// The indexer's product row: its query heads, then the one key.
    pub fn ix_row(&self) -> usize {
        (self.ix_heads + 1) * self.ix_dim
    }

    /// Tokens a query attends to at most past the budget: the budget's blocks and an incomplete block's tail.
    pub fn ix_tokens(&self) -> usize {
        self.ix_budget + self.ix_block - 1
    }

    /// One layer's matrices (the token mixer's, then the shared expert's two), in the forward pass's order. GDN: q, k, v and z as one, b and a as one, out; attention: q, k and v as one, o, the indexer's.
    pub fn layer_packs(&self, l: usize) -> Vec<Pack35> {
        let (h, p) = (self.hidden, self.layer_name(l));
        let pk = |module: &str, parts: &[(&str, usize)], cols: usize| Pack35 {
            parts: parts.iter().map(|&(n, r)| (format!("{module}.{n}.weight"), r)).collect(),
            key: format!("{module}.{}", parts[0].0),
            cols,
        };
        let mut out = match self.kind_of(l) {
            QxKind::Gdn => {
                let m = format!("{p}.linear_attn");
                vec![
                    pk(&m, &[("in_proj_qkv", self.conv_dim()), ("in_proj_z", self.value_dim())], h),
                    pk(&m, &[("in_proj_b", self.nv), ("in_proj_a", self.nv)], h),
                    pk(&m, &[("out_proj", h)], self.value_dim()),
                ]
            }
            QxKind::Attn => {
                let m = format!("{p}.self_attn");
                let kv = self.kv_heads * self.head_dim;
                vec![
                    pk(&m, &[("q_proj", 2 * self.heads * self.head_dim), ("k_proj", kv), ("v_proj", kv)], h),
                    pk(&m, &[("o_proj", h)], self.heads * self.head_dim),
                    pk(&format!("{m}.indexer"), &[("index_qk_proj", self.ix_row())], h),
                ]
            }
        };
        let s = format!("{p}.mlp.shared_expert");
        out.push(pk(&s, &[("gate_proj", self.shared_inter), ("up_proj", self.shared_inter)], h));
        out.push(pk(&s, &[("down_proj", h)], self.shared_inter));
        out
    }

    /// A Gated Residual's matrices: its bottleneck's two; `inject` [hc, hc hidden] is kept as it is.
    pub fn gr_packs(&self, module: &str) -> Vec<Pack35> {
        let w = self.hc * self.hidden;
        vec![
            Pack35 { key: format!("{module}.input_mix_weight_down"), parts: vec![(format!("{module}.input_mix_weight_down.weight"), self.hc_rank)], cols: w },
            Pack35 { key: format!("{module}.input_mix_weight_up"), parts: vec![(format!("{module}.input_mix_weight_up.weight"), w)], cols: self.hc_rank },
        ]
    }

    /// The final mixer's module (the model's, or the MTP head's).
    pub fn mixer_name(&self, mtp: bool) -> String {
        if mtp {
            "mtp.hyper_connection_mixer".into()
        } else {
            format!("{}hyper_connection_mixer", self.prefix)
        }
    }

    /// The PLE layer's key and value products as one.
    pub fn ple_pack(&self) -> Option<Pack35> {
        let p = self.ple.as_ref()?;
        let m = format!("{}.ple", self.layer_name(p.layer));
        Some(Pack35 {
            key: format!("{m}.key_proj"),
            parts: vec![(format!("{m}.key_proj.weight"), self.hc * self.hidden), (format!("{m}.value_proj.weight"), self.hidden)],
            cols: p.dim,
        })
    }
}

/// Which of Condor's packs of experts a directory serves as (`qwenget` makes them: DIR/pack or DIR/fast, DIR/precise, DIR/best): precise where asked (`--precise`), best
/// where the directory's name starts with "best", else fast. Each takes its own sidecars and its own skipping of the misses' tail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackMode {
    Precise,
    Best,
    Fast,
}

impl PackMode {
    pub fn of(pack: &Path, precise: bool) -> PackMode {
        if precise {
            PackMode::Precise
        } else if pack.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("best")) {
            PackMode::Best
        } else {
            PackMode::Fast
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            PackMode::Precise => "precise",
            PackMode::Best => "best",
            PackMode::Fast => "fast",
        }
    }

    /// The sidecar whose copies of some experts the GPU's places hold in place of the pack's, where the pack has one: precise its `hot` (its cb3 and cb4 experts as
    /// they are, the hottest at three bits); else its `fast` (every cb3 and cb4 expert at two bits: smaller places, more of them; MMLU-Pro 124/150 as with the
    /// pack's own), else its `hot`.
    pub fn side(self, pack: &Path) -> Option<std::path::PathBuf> {
        let names: &[&str] = if self == PackMode::Precise { &["hot"] } else { &["fast", "hot"] };
        names.iter().map(|x| pack.join(x)).find(|d| d.join("meta.json").exists())
    }

    /// The offload's `skip_rank`: precise 1, the most skipping that keeps its pack's numbers against the bf16 model (Strata's set: top-64 KL 0.137, top-1 88.1%,
    /// against 0.152 and 87.0% at 3); best 0, none skipped (its pack's 0.099 and 89.6%; 0.105 at 1, 0.117 at 3); fast 3.
    pub fn skip_rank(self) -> usize {
        match self {
            PackMode::Precise => 1,
            PackMode::Best => 0,
            PackMode::Fast => 3,
        }
    }
}

#[cfg(test)]
mod tests {
    /// A pack's mode by the flag and the directory's name, and each mode's sidecar where the pack has it.
    #[test]
    fn a_packs_mode_takes_its_sidecar_and_skipping() {
        let root = std::env::temp_dir().join(format!("glyd-packmode-{}", std::process::id()));
        let (best, fast) = (root.join("best-v2"), root.join("fast"));
        for d in [best.join("hot"), fast.join("fast"), fast.join("hot")] {
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("meta.json"), "{}").unwrap();
        }
        assert_eq!([PackMode::of(&best, false), PackMode::of(&fast, false), PackMode::of(&best, true)], [PackMode::Best, PackMode::Fast, PackMode::Precise]);
        assert_eq!(PackMode::Fast.side(&fast), Some(fast.join("fast")));
        assert_eq!(PackMode::Precise.side(&fast), Some(fast.join("hot")));
        assert_eq!(PackMode::Best.side(&best), Some(best.join("hot")));
        assert_eq!(PackMode::Fast.side(&root), None);
        assert_eq!([PackMode::Precise, PackMode::Best, PackMode::Fast].map(PackMode::skip_rank), [1, 0, 3]);
        let _ = std::fs::remove_dir_all(&root);
    }

    use super::*;

    pub(crate) fn real() -> ConfigQx {
        ConfigQx::parse(include_str!("../tests/fixtures/qwen4x_config.json"), "qx", Some(include_str!("../tests/fixtures/qwen4x_generation_config.json")))
            .unwrap()
    }

    /// The pinned config is read as research/qwen38-flash-next.md has it: 36 GDN and 12 attention layers, the PLE at layer 1, the indexer's 2,048 tokens in blocks of 4, a rotary of 64.
    #[test]
    fn the_real_config() {
        let c = real();
        assert_eq!((c.hidden, c.layers, c.vocab, c.layers_of(QxKind::Gdn), c.layers_of(QxKind::Attn)), (2560, 48, 248320, 36, 12));
        assert_eq!((c.nk, c.nv, c.conv_dim(), c.value_dim(), c.gate_sigmoid), (16, 48, 10240, 6144, true));
        assert_eq!((c.heads, c.kv_heads, c.head_dim, c.rotary, c.theta), (24, 2, 256, 64, 1e7));
        assert_eq!((c.ix_heads, c.ix_dim, c.ix_budget, c.ix_block, c.ix_tokens(), c.ix_row()), (4, 128, 2048, 4, 2051, 640));
        assert_eq!((c.hc, c.hc_rank, c.experts, c.top_k, c.inter, c.shared_inter), (4, 320, 512, 10, 640, 640));
        assert_eq!(c.ple, Some(PleDims { layer: 1, dim: 2560, kernel: 4, ngram: 3, heads: 8 }));
        assert_eq!((c.eos_ngram, c.eos.clone(), c.mtp_layers), (248044, vec![248046, 248044], 1));
        assert!(c.kinds.iter().enumerate().all(|(i, k)| (*k == QxKind::Attn) == (i % 4 == 3)));
        // the matrices, by shape: a GDN layer's three and an attention layer's three, then the shared expert's two
        let shapes = |l: usize| c.layer_packs(l).iter().map(|p| (p.rows(), p.cols)).collect::<Vec<_>>();
        assert_eq!(shapes(0), [(16384, 2560), (96, 2560), (2560, 6144), (1280, 2560), (2560, 640)]);
        assert_eq!(shapes(3), [(13312, 2560), (2560, 6144), (640, 2560), (1280, 2560), (2560, 640)]);
        assert_eq!(c.layer_name(48), "mtp.layers.0");
        assert_eq!(c.ple_pack().map(|p| (p.rows(), p.cols)), Some((12800, 2560)));
    }

    /// The released config's description, lowered, is every number this engine reads of the config (R3d): the layers' kinds, the heads, the
    /// indexer, the streams, the experts and their widths, the n-gram embedding's layer and sizes, the MTP layer; the tiny model's of the
    /// `qwen4x` logits case too. A description of another model is told apart, the difference named.
    #[test]
    fn the_description_lowers_to_this_engines_reading() {
        let text = include_str!("../tests/fixtures/qwen4x_config.json");
        let d = crate::runtime::describe::describe(&serde_json::from_str(text).unwrap()).unwrap();
        let c = ConfigQx::agrees(real(), &d).unwrap();
        assert_eq!(c.ple.as_ref().map(|p| p.layer), Some(1));
        let m: serde_json::Value = serde_json::from_str(include_str!("../tests/fixtures/qwen4x_tiny_manifest.json")).unwrap();
        let tiny = m["config"].as_str().unwrap();
        let own = ConfigQx::parse(tiny, "tiny", None).unwrap();
        assert!(ConfigQx::agrees(own.clone(), &crate::runtime::describe::describe(&serde_json::from_str(tiny).unwrap()).unwrap()).is_ok());
        let e = ConfigQx::agrees(own, &d).unwrap_err().to_string();
        assert!(e.contains("differ") && e.contains("hidden"), "{e}");
    }

    /// The n-gram embedding's sizes are refused in words where they make no embedding: a size of 0 or 1, no heads, sizes whose product does not fit a word (the fuzzer's
    /// find: an n-gram size of 0 was a subtraction below zero in the refusal itself).
    #[test]
    fn n_gram_sizes_that_make_no_embedding_are_refused() {
        let text = include_str!("../tests/fixtures/qwen4x_config.json");
        // (the fixture's own: "ngram_size": 3 and "heads_per_ngram": 8)
        let with = |key: &str, value: &str| {
            let had = format!("\"{key}\": {}", if key == "ngram_size" { 3 } else { 8 });
            assert!(text.contains(&had), "{had}");
            ConfigQx::parse(&text.replace(&had, &format!("\"{key}\": {value}")), "qx", None)
        };
        for (key, value) in [
            ("ngram_size", "0"),
            ("ngram_size", "1"),
            ("heads_per_ngram", "0"),
            ("heads_per_ngram", "18446744073709551615"),
            ("ngram_size", "18446744073709551615"),
        ] {
            let e = with(key, value).err().map(|e| e.to_string()).unwrap_or_default();
            assert!(e.contains("n-gram embedding"), "{key} {value}: {e}");
        }
        assert!(with("ngram_size", "3").is_ok());
    }
}
