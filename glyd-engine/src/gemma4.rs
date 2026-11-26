//! Gemma 4 (2026) text models' config (`ConfigG4`) and packed weights (its kernel table is src/kern/gemma4.rs's `KernG4`), which the description executor runs (src/dynamic/gemma4.rs),
//! and the development commands that drive it (`g4check`, `g4score`, `g4prefill`, `g4spec`, ...). The hand-written engine was removed in R3b (#137), after `paritycheck`
//! found the executor's rows the same bits. The mixture-of-experts 26B-A4B and the E models (per-layer embeddings, shared key-value layers) are refused.
//!
//! A layer (research/gemma4-for-the-engine.md): h = norm_in(x); x += norm_post_attn(attention(h)); h = norm_pre_ff(x); x += norm_post_ff(down(gelu(gate h) * up h)); x *= scalar.
//! Sliding layers (5 of 6): head 256, keys and values of the last `window` positions (a ring), the full rotary at theta 1e4. Global layers: head 512, the keys' projection
//! as the values too where the config says K = V, the proportional rotary (64 of 256 pairs) at theta 1e6. Attention scale 1. The logits soft-capped (30 tanh(l / 30)).

use crate::dev::Buf;
use crate::ffi::{err, Error, Result};
#[cfg(feature = "dev")]
use crate::gpu_ops::GpuOps;
use crate::runtime::config::ModelConfig;
use crate::weights::{choose_layout, ordered, pack_item_map, upload_bytes, upload_packed, Item, Layout, PackBuf};
#[cfg(feature = "dev")]
use crate::{ffi::Lib, sched::Sample};
#[cfg(feature = "dev")]
use glyd_gpu::json::{self, Value};
use glyd_gpu::safetensors::Checkpoint;
use std::path::Path;

/// 64-row groups of the embedding decoded at once for a prompt chunk.
pub(crate) const EMBED_GROUPS: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KindG4 {
    Sliding,
    Global,
}

/// Gemma 4's text config (config.json's `text_config`).
#[derive(Clone, Debug)]
pub struct ConfigG4 {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    /// Key-value heads of the sliding layers, and of the global ones (the latter `num_global_key_value_heads` where K = V, else the same).
    pub kv_heads: usize,
    pub global_kv_heads: usize,
    pub head_dim: usize,
    pub global_head_dim: usize,
    pub k_eq_v: bool,
    pub inter: usize,
    pub vocab: usize,
    pub eps: f32,
    pub window: usize,
    pub kinds: Vec<KindG4>,
    pub sliding_theta: f64,
    pub global_theta: f64,
    /// Rotating pairs of a global head (partial_rotary_factor * head / 2).
    pub global_rot: usize,
    pub softcap: f32,
    pub max_pos: usize,
    pub eos: Vec<i64>,
    pub prefix: String,
}

/// Gemma 4's bytes by its typed config, which the tests hold the planner's reading of its description to (src/plan.rs `sizing::level_of`).
#[cfg(test)]
impl ConfigG4 {
    /// The bytes on the GPU of the weights at `layout` and of the bf16 model: the matrices and the small tensors (each layer's four norms, its q and k norms of
    /// its kind's heads and its scalar; the final norm).
    pub fn weight_bytes(&self, layout: Layout) -> (u64, u64) {
        let (mut w, mut bf16) = (0u64, 0u64);
        for it in self.items() {
            let b = (it.rows * it.cols * 2) as u64;
            bf16 += b;
            w += (b as f64 * crate::plan::sizing::bits(layout) / 16.0) as u64;
        }
        let small = 2 * (self.kinds.iter().map(|&k| 4 * self.hidden + 2 * self.attn_of(k).0 + 1).sum::<usize>() + self.hidden) as u64;
        (w + small, bf16 + small)
    }

    /// The keys and values: (the sliding layers' rings, fixed: the window and a chunk each, or the context where it is shorter; a position's bytes in the global
    /// layers). `twelve`: in the executor's 12-bit lossless pages where they take a layer's ring, else bf16.
    pub fn kv_bytes(&self, chunk: usize, ctx: usize, twelve: bool) -> (u64, u64) {
        let cap_s = (crate::dynamic::seq_blocks(Some(self.window + chunk), ctx, 16) * 16) as u64;
        let (mut fixed, mut per) = (0u64, 0u64);
        for &k in &self.kinds {
            let (d, nk, _) = self.attn_of(k);
            let b = match self.ring12(k).filter(|_| twelve) {
                Some(s) => crate::plan::sizing::kv12_token_bytes(s.heads, s.channels, s.one),
                None => (nk * d * 2 * 2) as u64,
            };
            match k {
                KindG4::Sliding => fixed += cap_s * b,
                KindG4::Global => per += b,
            }
        }
        (fixed, per)
    }

    /// A layer kind's ring as the executor's 12-bit pages take it, None where they do not.
    fn ring12(&self, k: KindG4) -> Option<crate::dynamic::kv12::KvShape> {
        let (d, nk, _) = self.attn_of(k);
        let window = if k == KindG4::Sliding { self.window } else { 0 };
        crate::dynamic::kv12::ring_shape(nk, d, self.heads, window, false)
    }

    /// Its layers' rings in the executor's 12-bit pages.
    pub fn rings12(&self) -> Vec<crate::dynamic::kv12::KvShape> {
        self.kinds.iter().filter_map(|&k| self.ring12(k)).collect()
    }

    /// A token's keys and values in bf16 in each of its pools' groups, the sliding layers' and the global ones'.
    pub fn group_token_bytes(&self) -> [u64; 2] {
        let of = |kind: KindG4| -> u64 {
            let (d, nk, _) = self.attn_of(kind);
            self.kinds.iter().filter(|&&k| k == kind).count() as u64 * (nk * d * 2 * 2) as u64
        };
        [of(KindG4::Sliding), of(KindG4::Global)]
    }
}

impl ConfigG4 {
    pub fn load(dir: &Path) -> Result<ConfigG4> {
        ConfigG4::load_for(dir, false)
    }

    /// `draft`: a Gemma 4 "-assistant" draft model (its layers all read the big model's keys and values: no key or value projections; its tensors under "model.").
    pub fn load_for(dir: &Path, draft: bool) -> Result<ConfigG4> {
        ConfigG4::load_for_of(dir, &ModelConfig::load(dir)?, draft)
    }

    /// `load_for` of a config.json parsed once (`m`: `dir`'s).
    pub fn load_for_of(dir: &Path, m: &ModelConfig, draft: bool) -> Result<ConfigG4> {
        m.read_with(dir, |c, _, generation| ConfigG4::of(c, generation, draft))
    }

    /// `load_for` from the texts of config.json and generation_config.json.
    pub fn parse(text: &str, gen: Option<&str>, draft: bool) -> Result<ConfigG4> {
        ConfigG4::of(&ModelConfig::parse(text).map_err(|e| e.context("config.json"))?, gen, draft)
    }

    /// `parse` of a config.json parsed once (`ModelConfig`).
    pub fn of(m: &ModelConfig, gen: Option<&str>, draft: bool) -> Result<ConfigG4> {
        use serde_json::Value;
        let top = &m.top;
        let c = &m.text;
        let num = |k: &str| c.get(k).and_then(Value::as_u64).map(|x| x as usize);
        let need = |k: &str| num(k).ok_or_else(|| Error(format!("config.json: no {k}")));
        let mt = c.get("model_type").and_then(Value::as_str).unwrap_or("");
        if !mt.starts_with("gemma4") {
            return err(format!("config.json: model_type {mt}, not Gemma 4"));
        }
        if c.get("enable_moe_block").and_then(Value::as_bool) == Some(true) {
            return err("Gemma 4's mixture of experts (26B-A4B) is not in this engine yet");
        }
        // (shared key-value layers and the double-wide MLP are the layer kinds' (`Attn::kv_from`): the executor runs them from the description's lowering)
        if num("hidden_size_per_layer_input").unwrap_or(0) > 0 {
            return err("Gemma 4's per-layer embeddings (E2B, E4B) are not in this engine yet");
        }
        let kinds: Vec<KindG4> = c
            .get("layer_types")
            .and_then(Value::as_array)
            .ok_or_else(|| Error("config.json: no layer_types".into()))?
            .iter()
            .map(|v| match v.as_str() {
                Some("sliding_attention") => Ok(KindG4::Sliding),
                Some("full_attention") => Ok(KindG4::Global),
                x => err(format!("a layer type {x:?}")),
            })
            .collect::<Result<_>>()?;
        let rope = c.get("rope_parameters").ok_or_else(|| Error("config.json: no rope_parameters".into()))?;
        let f = |v: Option<&Value>| v.and_then(Value::as_f64);
        let sliding_theta = f(rope.get("sliding_attention").and_then(|r| r.get("rope_theta"))).unwrap_or(10_000.0);
        let g = rope.get("full_attention");
        let global_theta = f(g.and_then(|r| r.get("rope_theta"))).unwrap_or(1_000_000.0);
        let partial = f(g.and_then(|r| r.get("partial_rotary_factor"))).unwrap_or(1.0);
        let gtype = g.and_then(|r| r.get("rope_type")).and_then(Value::as_str).unwrap_or("default");
        if gtype != "proportional" && gtype != "default" {
            return err(format!("global rotary {gtype}: not supported"));
        }
        let head_dim = need("head_dim")?;
        let global_head_dim = num("global_head_dim").unwrap_or(head_dim);
        if head_dim != 256 || global_head_dim != 512 {
            return err(format!("heads {head_dim} / {global_head_dim}: this engine runs Gemma 4's 256 / 512"));
        }
        let k_eq_v = c.get("attention_k_eq_v").and_then(Value::as_bool).unwrap_or(false);
        let kv_heads = need("num_key_value_heads")?;
        let global_kv_heads = if k_eq_v { num("num_global_key_value_heads").unwrap_or(kv_heads) } else { kv_heads };
        let eos = match top.get("eos_token_id").or_else(|| c.get("eos_token_id")) {
            Some(Value::Array(a)) => a.iter().filter_map(Value::as_u64).map(|x| x as i64).collect(),
            Some(v) => v.as_u64().map(|x| vec![x as i64]).unwrap_or_default(),
            None => vec![1, 106],
        };
        let mut eos = eos;
        // (generation_config adds <|tool_response> (50): the model hands a tool call back to its caller)
        if let Some(gtext) = gen {
            // (a file that is there and does not parse is an error: read as none, its end tokens would be dropped and a reply would not stop at them)
            let gc = serde_json::from_str::<Value>(gtext).map_err(|e| Error(format!("generation_config.json: {e}")))?;
            if let Some(Value::Array(a)) = gc.get("eos_token_id") {
                for x in a.iter().filter_map(Value::as_u64) {
                    if !eos.contains(&(x as i64)) {
                        eos.push(x as i64);
                    }
                }
            }
        }
        let layers = need("num_hidden_layers")?;
        if kinds.len() != layers {
            return err(format!("{} layer types for {layers} layers", kinds.len()));
        }
        Ok(ConfigG4 {
            hidden: need("hidden_size")?,
            layers,
            heads: need("num_attention_heads")?,
            kv_heads,
            global_kv_heads,
            head_dim,
            global_head_dim,
            k_eq_v,
            inter: need("intermediate_size")?,
            vocab: need("vocab_size")?,
            eps: f(c.get("rms_norm_eps")).unwrap_or(1e-6) as f32,
            window: need("sliding_window")?,
            kinds,
            sliding_theta,
            global_theta,
            global_rot: if gtype == "proportional" { ((partial * global_head_dim as f64) as usize) / 2 } else { global_head_dim / 2 },
            softcap: f(c.get("final_logit_softcapping")).unwrap_or(0.0) as f32,
            max_pos: num("max_position_embeddings").unwrap_or(131_072),
            eos,
            prefix: if draft { "model." } else { "model.language_model." }.to_string(),
        })
    }

    /// (head dim, key-value heads, whether the layer has its own value projection) of a kind.
    pub fn attn_of(&self, k: KindG4) -> (usize, usize, bool) {
        match k {
            KindG4::Sliding => (self.head_dim, self.kv_heads, true),
            KindG4::Global => (self.global_head_dim, self.global_kv_heads, !self.k_eq_v),
        }
    }

    /// The projection row's width of a kind: queries, keys and (has_v) values.
    pub fn proj_width(&self, k: KindG4) -> usize {
        let (d, nk, has_v) = self.attn_of(k);
        self.heads * d + nk * d * if has_v { 2 } else { 1 }
    }

    /// The bf16 of sqrt(hidden), as transformers casts the embedding's scale.
    pub fn embed_scale(&self) -> f32 {
        let s = (self.hidden as f32).sqrt();
        f32::from_bits((s.to_bits().wrapping_add(0x7fff + ((s.to_bits() >> 16) & 1))) & 0xffff_0000)
    }
}

/// A layer's matrices (the projection, o, gate and up, down) and small tensors.
pub struct LayerG4 {
    pub kind: KindG4,
    pub proj: PackBuf,
    pub o: PackBuf,
    pub gate_up: PackBuf,
    pub down: PackBuf,
    pub in_norm: Buf,
    pub post_attn: Buf,
    pub pre_ff: Buf,
    pub post_ff: Buf,
    pub q_norm: Buf,
    pub k_norm: Buf,
    pub scalar: Buf,
}

pub struct WeightsG4 {
    pub cfg: ConfigG4,
    pub layout: Layout,
    pub layers: Vec<LayerG4>,
    pub embed: PackBuf,
    pub norm: Buf,
    /// Rotary tables (bf16 [positions, pairs]): the sliding layers' (128 pairs) and the global layers' (their rotating pairs).
    pub cos_s: Buf,
    pub sin_s: Buf,
    pub cos_g: Buf,
    pub sin_g: Buf,
    pub bytes: usize,
}

/// cos and sin (bf16 [max_pos, pairs]) of pairs i < `pairs` of a head of `dim`: inv_freq_i = theta^(-2i / dim), as transformers' default and proportional rotaries (the
/// latter's other pairs are cos 1, sin 0: not stored, the kernel leaves them).
pub(crate) fn rotary(theta: f64, dim: usize, pairs: usize, max_pos: usize) -> (Vec<u16>, Vec<u16>) {
    let inv: Vec<f32> = (0..pairs).map(|i| 1.0f32 / (theta as f32).powf((2 * i) as f32 / dim as f32)).collect();
    let (mut c, mut s) = (Vec::with_capacity(max_pos * pairs), Vec::with_capacity(max_pos * pairs));
    for p in 0..max_pos {
        for &f in &inv {
            let a = p as f32 * f;
            c.push(crate::weights::bf16_bits(a.cos()));
            s.push(crate::weights::bf16_bits(a.sin()));
        }
    }
    (c, s)
}

impl ConfigG4 {
    /// The matrices the engine packs, in load order: per layer the projection (q, k[, v]), o, gate + up, down; then the embedding (the tied LM head).
    pub(crate) fn items(&self) -> Vec<Item> {
        let (p, h) = (&self.prefix, self.hidden);
        let mut items = Vec::new();
        for (l, &k) in self.kinds.iter().enumerate() {
            let a = format!("{p}layers.{l}.self_attn");
            let (d, _, has_v) = self.attn_of(k);
            let mut m = vec![format!("{a}.q_proj.weight"), format!("{a}.k_proj.weight")];
            if has_v {
                m.push(format!("{a}.v_proj.weight"));
            }
            items.push(Item { members: m, rows: self.proj_width(k), cols: h });
            items.push(Item { members: vec![format!("{a}.o_proj.weight")], rows: h, cols: self.heads * d });
            let mlp = format!("{p}layers.{l}.mlp");
            items.push(Item { members: vec![format!("{mlp}.gate_proj.weight"), format!("{mlp}.up_proj.weight")], rows: 2 * self.inter, cols: h });
            items.push(Item { members: vec![format!("{mlp}.down_proj.weight")], rows: h, cols: self.inter });
        }
        items.push(Item { members: vec![format!("{p}embed_tokens.weight")], rows: self.vocab, cols: h });
        items
    }
}

impl WeightsG4 {
    pub fn load(dir: &Path, max_pos: usize, threads: usize, layout: Option<Layout>, cc: (i32, i32), total_mem: u64) -> Result<WeightsG4> {
        let cfg = ConfigG4::load(dir)?;
        let ck = Checkpoint::open(dir).map_err(|e| Error(e.to_string()))?;
        let p = cfg.prefix.clone();
        let items = cfg.items();
        let matrix_bytes: u64 = items.iter().map(|it| (it.rows * it.cols * 2) as u64).sum();
        // a glyd-v7 directory (`glyd-engine pack MODEL OUT --layout i4x|i6|...`, src/pack.rs gemma4_plan): its packs read as saved (each fused under its first member's name), in its layout
        let v7 = crate::weights::v7_manifest(dir)?;
        let layout = match &v7 {
            Some((l, _)) => *l,
            None => choose_layout(layout, cc, total_mem, matrix_bytes, 0).0,
        };
        let one = |k: usize| -> Result<crate::weights::Packed> {
            let it = &items[k];
            match &v7 {
                Some((Layout::Tiered, m)) => crate::weights::read_lossless(&ck, m, it.members[0].trim_end_matches(".weight"), it.rows, it.cols),
                Some((l, m)) => crate::weights::read_v7(&ck, m, *l, &[(it.members[0].trim_end_matches(".weight"), it.rows)], it.cols),
                None => pack_item_map(&ck, it, layout, &|_| {}),
            }
        };
        let mut packed: Vec<Option<PackBuf>> = (0..items.len()).map(|_| None).collect();
        ordered(items.len(), threads, 2 * threads.max(1), &one, |k, pk| {
            crate::status::STATUS.beat();
            packed[k] = Some(upload_packed(pk)?);
            Ok(())
        })?;
        let read = |name: &str| -> Result<Buf> {
            let (_, t) = ck.get(name).ok_or_else(|| Error(format!("no tensor {name}")))?;
            if t.dtype != "BF16" {
                return err(format!("{name}: {} where bf16 is wanted", t.dtype));
            }
            upload_bytes(&ck.read(name).map_err(|e| Error(e.to_string()))?)
        };
        let mut it = packed.into_iter().map(|x| x.expect("every item packed"));
        let mut layers = Vec::with_capacity(cfg.layers);
        for (l, &kind) in cfg.kinds.iter().enumerate() {
            let pl = format!("{p}layers.{l}");
            layers.push(LayerG4 {
                kind,
                proj: it.next().expect("an item"),
                o: it.next().expect("an item"),
                gate_up: it.next().expect("an item"),
                down: it.next().expect("an item"),
                in_norm: read(&format!("{pl}.input_layernorm.weight"))?,
                post_attn: read(&format!("{pl}.post_attention_layernorm.weight"))?,
                pre_ff: read(&format!("{pl}.pre_feedforward_layernorm.weight"))?,
                post_ff: read(&format!("{pl}.post_feedforward_layernorm.weight"))?,
                q_norm: read(&format!("{pl}.self_attn.q_norm.weight"))?,
                k_norm: read(&format!("{pl}.self_attn.k_norm.weight"))?,
                scalar: read(&format!("{pl}.layer_scalar"))?,
            });
        }
        let embed = it.next().expect("the embedding");
        let norm = read(&format!("{p}norm.weight"))?;
        let mp = max_pos.min(cfg.max_pos);
        let (cs, ss) = rotary(cfg.sliding_theta, cfg.head_dim, cfg.head_dim / 2, mp);
        let (cg, sg) = rotary(cfg.global_theta, cfg.global_head_dim, cfg.global_rot, mp);
        let up = |v: &[u16]| upload_bytes(crate::weights::as_bytes(v));
        let (cos_s, sin_s, cos_g, sin_g) = (up(&cs)?, up(&ss)?, up(&cg)?, up(&sg)?);
        let mut bytes = embed.bytes + norm.bytes + cos_s.bytes + sin_s.bytes + cos_g.bytes + sin_g.bytes;
        for ly in &layers {
            bytes += ly.proj.bytes + ly.o.bytes + ly.gate_up.bytes + ly.down.bytes;
        }
        Ok(WeightsG4 { cfg, layout, layers, embed, norm, cos_s, sin_s, cos_g, sin_g, bytes })
    }
}

/// A draft model's layer: the query projection alone (the keys and values are the big model's), o, gate and up, down, and its small tensors.
pub struct DraftLayer {
    pub kind: KindG4,
    pub q: PackBuf,
    pub o: PackBuf,
    pub gate_up: PackBuf,
    pub down: PackBuf,
    pub in_norm: Buf,
    pub post_attn: Buf,
    pub pre_ff: Buf,
    pub post_ff: Buf,
    pub q_norm: Buf,
    pub scalar: Buf,
}

/// A Gemma 4 "-assistant" draft model (transformers' Gemma4AssistantForCausalLM, drafted as its SinglePositionMultiTokenCandidateGenerator does): its input the last
/// token's scaled embedding and the big model's final hidden state for the position before, projected down (pre); a few layers whose attention reads the big model's
/// last sliding and last global layers' keys and values at the last token's position; its own LM head (tied to its embedding) for the draft token, and the hidden
/// state projected back up (post) for the next draft. Packed lossy (the big model checks every draft token, so a draft's precision moves only how many are kept).
pub struct DraftG4 {
    pub cfg: ConfigG4,
    pub layers: Vec<DraftLayer>,
    pub pre: PackBuf,
    pub post: PackBuf,
    pub head: PackBuf,
    pub norm: Buf,
    pub bytes: usize,
}

impl DraftG4 {
    pub fn load(dir: &Path, big: &ConfigG4, threads: usize, layout: Layout) -> Result<DraftG4> {
        let cfg = ConfigG4::load_for(dir, true)?;
        if cfg.heads != big.heads || cfg.kv_heads != big.kv_heads || cfg.global_kv_heads != big.global_kv_heads || cfg.vocab != big.vocab {
            return err("the draft model's heads or vocabulary are not the big model's: it is not this model's assistant");
        }
        let ck = Checkpoint::open(dir).map_err(|e| Error(e.to_string()))?;
        let (p, h, b) = (cfg.prefix.clone(), cfg.hidden, big.hidden);
        let mut items = Vec::new();
        for (l, &k) in cfg.kinds.iter().enumerate() {
            let (a, d) = (format!("{p}layers.{l}.self_attn"), cfg.attn_of(k).0);
            items.push(Item { members: vec![format!("{a}.q_proj.weight")], rows: cfg.heads * d, cols: h });
            items.push(Item { members: vec![format!("{a}.o_proj.weight")], rows: h, cols: cfg.heads * d });
            let mlp = format!("{p}layers.{l}.mlp");
            items.push(Item { members: vec![format!("{mlp}.gate_proj.weight"), format!("{mlp}.up_proj.weight")], rows: 2 * cfg.inter, cols: h });
            items.push(Item { members: vec![format!("{mlp}.down_proj.weight")], rows: h, cols: cfg.inter });
        }
        items.push(Item { members: vec!["pre_projection.weight".into()], rows: h, cols: 2 * b });
        items.push(Item { members: vec!["post_projection.weight".into()], rows: b, cols: h });
        items.push(Item { members: vec![format!("{p}embed_tokens.weight")], rows: cfg.vocab, cols: h });
        let mut packed: Vec<Option<PackBuf>> = (0..items.len()).map(|_| None).collect();
        ordered(items.len(), threads, 2 * threads.max(1), &|k| pack_item_map(&ck, &items[k], layout, &|_| {}), |k, pk| {
            packed[k] = Some(upload_packed(pk)?);
            Ok(())
        })?;
        let read = |name: &str| -> Result<Buf> {
            ck.get(name).ok_or_else(|| Error(format!("no tensor {name}")))?;
            upload_bytes(&ck.read(name).map_err(|e| Error(e.to_string()))?)
        };
        let mut it = packed.into_iter().map(|x| x.expect("every item packed"));
        let mut layers = Vec::with_capacity(cfg.layers);
        for (l, &kind) in cfg.kinds.iter().enumerate() {
            let pl = format!("{p}layers.{l}");
            layers.push(DraftLayer {
                kind,
                q: it.next().expect("an item"),
                o: it.next().expect("an item"),
                gate_up: it.next().expect("an item"),
                down: it.next().expect("an item"),
                in_norm: read(&format!("{pl}.input_layernorm.weight"))?,
                post_attn: read(&format!("{pl}.post_attention_layernorm.weight"))?,
                pre_ff: read(&format!("{pl}.pre_feedforward_layernorm.weight"))?,
                post_ff: read(&format!("{pl}.post_feedforward_layernorm.weight"))?,
                q_norm: read(&format!("{pl}.self_attn.q_norm.weight"))?,
                scalar: read(&format!("{pl}.layer_scalar"))?,
            });
        }
        let (pre, post, head) = (it.next().expect("pre"), it.next().expect("post"), it.next().expect("the head"));
        let norm = read(&format!("{p}norm.weight"))?;
        let bytes = pre.bytes + post.bytes + head.bytes + layers.iter().map(|l| l.q.bytes + l.o.bytes + l.gate_up.bytes + l.down.bytes).sum::<usize>();
        Ok(DraftG4 { cfg, layers, pre, post, head, norm, bytes })
    }
}

/// What the development commands need of a Gemma 4 run: the description executor, or the same split over GPUs (`g4check --split`).
#[cfg(feature = "dev")]
trait Pass {
    fn reset(&mut self) -> Result<()>;
    fn prefill(&mut self, t: &[u32]) -> Result<Sample>;
    fn step(&mut self, t: u32) -> Result<Sample>;
    fn logits(&self) -> Result<Vec<f32>>;
    fn prof_take(&self) -> Result<Vec<(&'static str, f64)>>;
    fn spec(&mut self, last: u32) -> Result<Vec<u32>>;
    fn pos(&self) -> usize;
    fn ctx(&self) -> usize;
    fn eos(&self) -> Vec<i64>;
}

#[cfg(feature = "dev")]
impl Pass for crate::dynamic::Dyn {
    fn reset(&mut self) -> Result<()> {
        crate::dynamic::Dyn::reset(self)
    }
    fn prefill(&mut self, t: &[u32]) -> Result<Sample> {
        crate::dynamic::Dyn::prefill(self, t)
    }
    fn step(&mut self, t: u32) -> Result<Sample> {
        crate::dynamic::Dyn::step(self, t)
    }
    fn logits(&self) -> Result<Vec<f32>> {
        crate::dynamic::Dyn::logits(self)
    }
    fn prof_take(&self) -> Result<Vec<(&'static str, f64)>> {
        crate::dynamic::Dyn::prof_take(self)
    }
    fn spec(&mut self, last: u32) -> Result<Vec<u32>> {
        crate::dynamic::Dyn::spec_round(self, last, None)
    }
    fn pos(&self) -> usize {
        self.n
    }
    fn ctx(&self) -> usize {
        self.capacity()
    }
    fn eos(&self) -> Vec<i64> {
        self.desc().eos.clone()
    }
}

#[cfg(feature = "dev")]
impl Pass for crate::dynamic::split::Split {
    fn reset(&mut self) -> Result<()> {
        crate::dynamic::split::Split::reset(self)
    }
    fn prefill(&mut self, t: &[u32]) -> Result<Sample> {
        crate::dynamic::split::Split::prefill(self, t)
    }
    fn step(&mut self, t: u32) -> Result<Sample> {
        crate::dynamic::split::Split::step(self, t)
    }
    fn logits(&self) -> Result<Vec<f32>> {
        crate::dynamic::split::Split::logits(self)
    }
    fn prof_take(&self) -> Result<Vec<(&'static str, f64)>> {
        Ok(Vec::new())
    }
    fn spec(&mut self, _: u32) -> Result<Vec<u32>> {
        err("drafting with a model split across GPUs: not yet")
    }
    fn pos(&self) -> usize {
        self.n
    }
    fn ctx(&self) -> usize {
        self.capacity()
    }
    fn eos(&self) -> Vec<i64> {
        crate::dynamic::split::Split::eos(self).to_vec()
    }
}

/// `--split 24[,36] [--devices 0,1[,2]]`: the dynamic engine with the layers cut before each given layer, stage k on GPU k of the list (all on GPU 0 where none is given).
#[cfg(feature = "dev")]
fn split_of(a: &crate::args::Args, dir: &Path, desc: crate::dynamic::ModelDesc, o: &crate::dynamic::Opts) -> Result<Option<crate::dynamic::split::Split>> {
    let Some(cuts) = a.flag("split") else { return Ok(None) };
    let num = |x: &str| x.trim().parse::<usize>().map_err(|_| Error(format!("--split/--devices {x}: not a number")));
    let cuts: Vec<usize> = cuts.split(',').map(num).collect::<Result<_>>()?;
    let devices: Vec<i32> = match a.flag("devices") {
        Some(d) => d.split(',').map(|x| num(x).map(|v| v as i32)).collect::<Result<_>>()?,
        None => vec![0; cuts.len() + 1],
    };
    Ok(Some(crate::dynamic::split::Split::new(dir, desc, &cuts, &devices, o)?))
}

/// Gemma 4 on the description executor, for the development commands (the hand-written engine was removed in R3b, #137).
#[cfg(feature = "dev")]
fn load_dyn(dir: &Path, ctx: usize, chunk: usize, graph: bool, layout: Option<Layout>, draft: Option<(&Path, usize)>) -> Result<crate::dynamic::Dyn> {
    let c = ConfigG4::load(dir)?;
    let spec = match draft {
        Some((d, k)) => {
            Some(crate::dynamic::DraftSpec { dir: d, desc: crate::dynamic::gemma4::describe_draft(&ConfigG4::load_for(d, true)?, &c), k, experts: None })
        }
        None => None,
    };
    crate::dynamic::Dyn::new(
        dir,
        &ModelConfig::load(dir)?,
        crate::dynamic::gemma4::lowered(dir, &c)?,
        spec,
        &crate::dynamic::Opts { ctx, chunk, graph, device: 0, layout },
    )
}

/// `g4check MODEL REF.json [--layout L] [--ctx N] [--chunk N] [--no-graph] [--split ...]`: the description executor against transformers' reference
/// (research/gemma4-engine/ref_gemma4.py): each prompt read as a prompt, then its reference continuation fed token by token; at every position from the prompt's last on,
/// the executor's top token against the reference's, and the reference's top-8 log-probabilities against the executor's for the same tokens (a lower bound of the KL
/// divergence over them).
#[cfg(feature = "dev")]
pub fn check(a: &crate::args::Args) -> Result<()> {
    let dir = Path::new(a.need(1, "MODEL")?);
    let refp = a.need(2, "REF")?;
    let text = std::fs::read_to_string(refp).map_err(|e| Error(format!("{refp}: {e}")))?;
    let r = json::parse(&text).map_err(|e| Error(format!("{refp}: {e}")))?;
    let layout = a.layout()?;
    let (ctx, chunk) = (a.num("ctx", 8192usize)?, a.num("chunk", 1024usize)?);
    let t0 = std::time::Instant::now();
    let prof = a.has("prof");
    let graph = !a.has("no-graph") && !prof;
    let mut e: Box<dyn Pass> = if a.flag("split").is_some() {
        let c = ConfigG4::load(dir)?;
        let so = crate::dynamic::Opts { ctx, chunk, graph, device: 0, layout };
        let sp = split_of(a, dir, crate::dynamic::gemma4::lowered(dir, &c)?, &so)?.expect("--split given");
        eprintln!("loaded in {:.1} s: {}", t0.elapsed().as_secs_f64(), sp.describe());
        Box::new(sp)
    } else {
        let mut d = load_dyn(dir, ctx, chunk, graph, layout, None)?;
        if prof {
            d.prof = Some(Default::default());
        }
        eprintln!("loaded in {:.1} s: {}", t0.elapsed().as_secs_f64(), d.describe());
        Box::new(d)
    };
    let mut stages: Vec<(&'static str, f64)> = Vec::new();
    let mut prof_steps = 0usize;
    let (mut agree, mut total, mut kl_sum) = (0usize, 0usize, 0f64);
    for p in r.get("prompts").and_then(Value::as_array).unwrap_or(&[]) {
        let ids: Vec<u32> = p.get("ids").and_then(Value::as_u64s).unwrap_or_default().into_iter().map(|x| x as u32).collect();
        let cont: Vec<u32> = p.get("continuation").and_then(Value::as_u64s).unwrap_or_default().into_iter().map(|x| x as u32).collect();
        let top_ids = p.get("top_ids").and_then(Value::as_array).unwrap_or(&[]);
        let top_lp = p.get("top_lp").and_then(Value::as_array).unwrap_or(&[]);
        e.reset()?;
        let t1 = std::time::Instant::now();
        let mut s = e.prefill(&ids)?;
        e.prof_take()?;
        let prompt_ms = t1.elapsed().as_secs_f64() * 1e3;
        let (mut a_p, mut n_p, mut kl_p, mut step_ms) = (0usize, 0usize, 0f64, 0f64);
        for (i, &next) in std::iter::once(&0u32).chain(cont.iter()).enumerate() {
            if i > 0 {
                let ts = std::time::Instant::now();
                s = e.step(next)?;
                step_ms += ts.elapsed().as_secs_f64() * 1e3;
                for (k, ms) in e.prof_take()? {
                    match stages.iter_mut().find(|(x, _)| *x == k) {
                        Some(x) => x.1 += ms,
                        None => stages.push((k, ms)),
                    }
                }
                prof_steps += 1;
            }
            let at = ids.len() - 1 + i;
            if at >= top_ids.len() || i >= cont.len() {
                break;
            }
            let rid: Vec<u64> = top_ids[at].as_u64s().unwrap_or_default();
            let rlp: Vec<f64> = top_lp[at]
                .as_array()
                .unwrap_or(&[])
                .iter()
                .filter_map(|v| match v {
                    Value::Number(n) => n.parse().ok(),
                    _ => None,
                })
                .collect();
            let lg = e.logits()?;
            let m = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            let lse = m + lg.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln();
            for (j, (&tid, &lp)) in rid.iter().zip(&rlp).enumerate() {
                let elp = lg[tid as usize] as f64 - lse;
                kl_p += lp.exp() * (lp - elp);
                if j == 0 && s.token as u64 == tid {
                    a_p += 1;
                }
            }
            n_p += 1;
        }
        let steps = n_p.saturating_sub(1).max(1);
        eprintln!(
            "{} prompt tokens in {prompt_ms:.0} ms, then {:.2} ms a token ({:.1} tok/s): top-1 agrees {a_p}/{n_p}, KL (top-8) {:.5}",
            ids.len(),
            step_ms / steps as f64,
            1000.0 * steps as f64 / step_ms.max(1e-9),
            kl_p / n_p.max(1) as f64
        );
        agree += a_p;
        total += n_p;
        kl_sum += kl_p;
    }
    println!("top-1 agrees {agree}/{total}, mean KL over the reference's top 8: {:.5}", kl_sum / total.max(1) as f64);
    if prof_steps > 0 {
        let all: f64 = stages.iter().map(|x| x.1).sum();
        println!("a decode step's stages (GPU time, {prof_steps} steps): {:.2} ms", all / prof_steps as f64);
        for (k, ms) in &stages {
            println!("  {:>22} {:7.3} ms  {:5.1}%", k, ms / prof_steps as f64, 100.0 * ms / all);
        }
    }
    Ok(())
}

/// `g4score MODEL CHUNKS.json OUT.f32 [--layout L] [--ctx N]`: the engine's logits for KL divergence against a reference (research/gemma4-kld/kld_compare.py, the
/// rows kld_ref_g4.py writes): each chunk's first n/2 + 1 tokens read as a prompt, the rest fed one at a time by decode steps (the path generation takes); the logits
/// (after the soft-cap) at positions n/2 .. n-2 as float32 rows [chunks * (n/2 - 1), vocab], and OUT.json.
#[cfg(feature = "dev")]
pub fn score(a: &crate::args::Args) -> Result<()> {
    let dir = Path::new(a.need(1, "MODEL")?);
    let tokp = a.need(2, "CHUNKS")?;
    let out = a.need(3, "OUT")?;
    let text = std::fs::read_to_string(tokp).map_err(|e| Error(format!("{tokp}: {e}")))?;
    let docs: Vec<Vec<u32>> = json::parse(&text)
        .map_err(|e| Error(format!("{tokp}: {e}")))?
        .get("docs")
        .and_then(Value::as_array)
        .unwrap_or(&[])
        .iter()
        .map(|d| d.as_u64s().unwrap_or_default().into_iter().map(|x| x as u32).collect())
        .collect();
    let layout = a.layout()?;
    let t0 = std::time::Instant::now();
    let mut e = load_dyn(dir, a.num("ctx", 1024usize)?, a.num("chunk", 512usize)?, true, layout, None)?;
    eprintln!("loaded in {:.1} s: {}", t0.elapsed().as_secs_f64(), e.describe());
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(out).map_err(|e| Error(format!("{out}: {e}")))?);
    let mut per = 0;
    for (c, d) in docs.iter().enumerate() {
        let n = d.len();
        e.reset()?;
        e.prefill(&d[..n / 2 + 1])?;
        let mut rows = 0;
        loop {
            for x in e.logits()? {
                f.write_all(&x.to_le_bytes()).map_err(|e| Error(e.to_string()))?;
            }
            rows += 1;
            let at = n / 2 + rows;
            if at > n - 2 {
                break;
            }
            e.step(d[at])?;
        }
        per = rows;
        eprintln!("chunk {}/{} ({:.0} s)", c + 1, docs.len(), t0.elapsed().as_secs_f64());
    }
    f.flush().map_err(|e| Error(e.to_string()))?;
    let meta = format!(
        "{{\"vocab\": {}, \"rows_per_chunk\": {per}, \"model\": {:?}, \"chunks\": {}, \"engine\": \"glyd-engine g4score\", \"layout\": \"{:?}\"}}",
        e.desc().vocab,
        dir.display().to_string(),
        docs.len(),
        layout
    );
    std::fs::write(format!("{out}.json"), meta).map_err(|e| Error(e.to_string()))?;
    Ok(())
}

/// `g4prefill MODEL [--tokens N] [--ctx N] [--chunk N] [--layout L]`: a prompt of N tokens (a fixed pattern) read once to warm up and once timed, with each
/// stage's share of the GPU time (the same marks as g4check --prof): where a long prompt's time goes.
#[cfg(feature = "dev")]
pub fn prefill_prof(a: &crate::args::Args) -> Result<()> {
    let dir = Path::new(a.need(1, "MODEL")?);
    let n = a.num("tokens", 16384usize)?;
    let mut e = load_dyn(dir, a.num("ctx", n + 64)?, a.num("chunk", 512usize)?, false, a.layout()?, None)?;
    eprintln!("{}", e.describe());
    let toks: Vec<u32> = std::iter::once(2u32).chain((1..n).map(|i| 1000 + (i as u32 * 7919) % 200_000)).collect();
    e.reset()?;
    e.prefill(&toks[..n.min(2048)])?;
    e.prof = Some(Default::default());
    e.reset()?;
    let t0 = std::time::Instant::now();
    e.prefill(&toks)?;
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let st = e.prof_take()?;
    let all: f64 = st.iter().map(|x| x.1).sum();
    println!("{n} tokens in {:.2} s ({:.0} tokens/s), GPU time {:.2} s:", ms / 1e3, n as f64 * 1e3 / ms, all / 1e3);
    for (k, v) in &st {
        println!("  {:>22} {:8.1} ms  {:5.1}%", k, v, 100.0 * v / all);
    }
    // --dump FILE: the logits after the prompt (f32), and 32 greedy tokens from there, to compare kernels
    if let Some(f) = a.flag("dump") {
        let lg = e.logits()?;
        std::fs::write(f, lg.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).map_err(|e| Error(e.to_string()))?;
        e.prof = None;
        let mut t = crate::sampler::argmax(&lg);
        let mut out = vec![t];
        for _ in 0..31 {
            t = e.step(t)?.token;
            out.push(t);
        }
        println!("greedy after the prompt: {out:?}");
    }
    Ok(())
}

/// `g4tok MODEL REF.json`: the chat template and tokenizer (src/chat.rs, src/tok.rs) against transformers' on the reference prompts (research/gemma4-engine/ref_gemma4.py renders
/// each as a user turn, add_generation_prompt, enable_thinking false, then tokenizes without special tokens): the ids the same, and each continuation decoded as transformers decoded it. No GPU.
#[cfg(feature = "dev")]
pub fn tok_check(a: &crate::args::Args) -> Result<()> {
    let dir = Path::new(a.need(1, "MODEL")?);
    let refp = a.need(2, "REF")?;
    let text = std::fs::read_to_string(refp).map_err(|e| Error(format!("{refp}: {e}")))?;
    let r = json::parse(&text).map_err(|e| Error(format!("{refp}: {e}")))?;
    let tok = crate::tok::Tok::load(dir)?;
    let chat = crate::chat::Chat::load(dir)?.ok_or_else(|| Error("no chat template".into()))?;
    let (mut ok, mut n) = (0, 0);
    for p in r.get("prompts").and_then(Value::as_array).unwrap_or(&[]) {
        let prompt = p.get("prompt").and_then(Value::as_str).unwrap_or("");
        let want: Vec<u32> = p.get("ids").and_then(Value::as_u64s).unwrap_or_default().into_iter().map(|x| x as u32).collect();
        let rendered = chat.render_with(&[crate::chat::Message::new("user", prompt)], &crate::chat::Options::new(true).thinking(Some(false)))?;
        let got = tok.encode_with(&rendered, false)?;
        let cont: Vec<u32> = p.get("continuation").and_then(Value::as_u64s).unwrap_or_default().into_iter().map(|x| x as u32).collect();
        let dec = tok.decode_with(&cont, false)?;
        let want_text = p.get("text").and_then(Value::as_str).unwrap_or("");
        let same = got == want;
        n += 1;
        ok += (same && dec == want_text) as usize;
        println!(
            "{} ids ({} vs {}), decode {}: {prompt:?}",
            if same { "SAME" } else { "DIFFER" },
            got.len(),
            want.len(),
            if dec == want_text { "same" } else { "DIFFERS" }
        );
        if !same {
            println!("  ours   {got:?}\n  theirs {want:?}\n  rendered {rendered:?}");
        }
        if dec != want_text {
            println!("  ours   {dec:?}\n  theirs {want_text:?}");
        }
    }
    println!("{ok} of {n} prompts the same as transformers");
    if ok != n {
        return err("the chat template or tokenizer differs from transformers'");
    }
    Ok(())
}

/// `g4spec MODEL REF.json --draft DIR [--k 4] [--tokens 128] [--layout L] [--ctx N] [--chunk N]`: drafting (Gemma 4's "-assistant" model) against plain greedy steps on each
/// reference prompt: both generate the same number of tokens from the same prompt; the speeds, the tokens a round, whether the two outputs are the same token for token, and
/// how far each agrees with the reference's continuation (transformers' greedy).
#[cfg(feature = "dev")]
pub fn spec_check(a: &crate::args::Args) -> Result<()> {
    let dir = Path::new(a.need(1, "MODEL")?);
    let refp = a.need(2, "REF")?;
    let text = std::fs::read_to_string(refp).map_err(|e| Error(format!("{refp}: {e}")))?;
    let r = json::parse(&text).map_err(|e| Error(format!("{refp}: {e}")))?;
    let draft = a.flag("draft").ok_or_else(|| Error("--draft DIR: the draft model".into()))?;
    let k = a.num("k", 4usize)?;
    let want = a.num("tokens", 128usize)?;
    let t0 = std::time::Instant::now();
    let d = load_dyn(dir, a.num("ctx", 8192usize)?, a.num("chunk", 512usize)?, true, a.layout()?, Some((Path::new(draft), k)))?;
    eprintln!("loaded in {:.1} s: {}", t0.elapsed().as_secs_f64(), d.describe());
    let mut e: Box<dyn Pass> = Box::new(d);
    let (mut plain_ms, mut spec_ms, mut plain_n, mut spec_n, mut rounds, mut same_all) = (0f64, 0f64, 0usize, 0usize, 0usize, true);
    for p in r.get("prompts").and_then(Value::as_array).unwrap_or(&[]) {
        let ids: Vec<u32> = p.get("ids").and_then(Value::as_u64s).unwrap_or_default().into_iter().map(|x| x as u32).collect();
        let cont: Vec<u32> = p.get("continuation").and_then(Value::as_u64s).unwrap_or_default().into_iter().map(|x| x as u32).collect();
        let eos = e.eos();
        let stop = |t: u32| eos.contains(&(t as i64));
        // plain greedy steps
        e.reset()?;
        let mut tok = e.prefill(&ids)?.token;
        let mut plain = vec![tok];
        let ts = std::time::Instant::now();
        while plain.len() < want && !stop(tok) && e.pos() + 2 < e.ctx() {
            tok = e.step(tok)?.token;
            plain.push(tok);
        }
        let pm = ts.elapsed().as_secs_f64() * 1e3;
        // drafted
        e.reset()?;
        let first = e.prefill(&ids)?.token;
        let mut spec = vec![first];
        let ts = std::time::Instant::now();
        let mut rr = 0usize;
        while spec.len() < want && !stop(*spec.last().expect("a token")) && e.pos() + k + 2 < e.ctx() {
            let got = e.spec(*spec.last().expect("a token"))?;
            rr += 1;
            for t in got {
                spec.push(t);
                if stop(t) || spec.len() >= want {
                    break;
                }
            }
        }
        let sm = ts.elapsed().as_secs_f64() * 1e3;
        let common = plain.iter().zip(&spec).take_while(|(x, y)| x == y).count();
        let n_cmp = plain.len().min(spec.len());
        let same = common == n_cmp;
        same_all &= same;
        let agree = |v: &[u32]| v.iter().zip(&cont).take_while(|(x, y)| x == y).count();
        eprintln!(
            "{} prompt tokens: plain {:.1} tok/s ({} tokens), drafted {:.1} tok/s ({} tokens, {:.2} a round), {} (first {common} of {n_cmp} the same); the reference's first {} / {} / {} (plain / drafted / its length)",
            ids.len(),
            1000.0 * (plain.len() - 1) as f64 / pm.max(1e-9),
            plain.len(),
            1000.0 * (spec.len() - 1) as f64 / sm.max(1e-9),
            spec.len(),
            (spec.len() - 1) as f64 / rr.max(1) as f64,
            if same { "SAME" } else { "DIFFER" },
            agree(&plain),
            agree(&spec),
            cont.len()
        );
        plain_ms += pm;
        spec_ms += sm;
        plain_n += plain.len() - 1;
        spec_n += spec.len() - 1;
        rounds += rr;
    }
    println!(
        "plain {:.1} tok/s, drafted {:.1} tok/s ({:.2}x), {:.2} tokens a round, outputs {}",
        1000.0 * plain_n as f64 / plain_ms.max(1e-9),
        1000.0 * spec_n as f64 / spec_ms.max(1e-9),
        (spec_n as f64 / spec_ms.max(1e-9)) / (plain_n as f64 / plain_ms.max(1e-9)).max(1e-9),
        spec_n as f64 / rounds.max(1) as f64,
        if same_all { "the same" } else { "DIFFER (see above)" }
    );
    Ok(())
}

/// `i4xcheck MODEL [--layers N]`: the exact four-bit layout on this GPU against the checkpoint: each matrix of the first N layers (and the
/// embedding's first 4,096 rows) packed as the engine packs it (`--layout i4x`), decoded by the GPU (the library's unpack, the Int4H decode the
/// products use) and compared with the checkpoint's bf16 bit for bit. Says which matrices were not four-bit-trained (packed lossless instead).
#[cfg(feature = "dev")]
pub fn i4x_check(a: &crate::args::Args) -> Result<()> {
    let dir = Path::new(a.need(1, "MODEL")?);
    let cfg = ConfigG4::load(dir)?;
    let ck = Checkpoint::open(dir).map_err(|e| Error(e.to_string()))?;
    let dev = crate::dev::Dev::new(0)?;
    let lib = Lib::find()?;
    let p = &cfg.prefix;
    let mut names: Vec<(String, usize)> = Vec::new();
    for l in 0..a.num("layers", 2usize)?.min(cfg.layers) {
        let mut add = |n: &str| {
            let full = format!("{p}layers.{l}.{n}.weight");
            if ck.get(&full).is_some() {
                names.push((full, usize::MAX));
            }
        };
        for n in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"] {
            add(n);
        }
    }
    names.push((format!("{p}embed_tokens.weight"), 4096));
    let (mut exact, mut fallback, mut bad) = (0, 0, 0);
    for (name, cap) in &names {
        let (_, t) = ck.get(name).ok_or_else(|| Error(format!("no tensor {name}")))?;
        let (rows, cols) = ((t.shape[0] as usize).min(*cap), t.shape[1] as usize);
        let mut w = vec![0u16; t.shape[0] as usize * cols];
        ck.read_bf16(name, &mut w).map_err(|e| Error(e.to_string()))?;
        w.truncate(rows * cols);
        let pk = match glyd_gpu::pack::pack_int4h_checked(&w, rows, cols) {
            Some(pk) => pk,
            None => {
                println!("lossless {name}: not four-bit-trained (the engine keeps it in the smallest lossless layout)");
                fallback += 1;
                continue;
            }
        };
        let pb = upload_packed(crate::weights::Packed {
            data: pk.data,
            b1: pk.blocks,
            b2: pk.block_base.iter().flat_map(|x| x.to_le_bytes()).collect(),
            words: crate::ffi::Words::Tiered { blocks: 0, block_base: 0, tiers: pk.tiers },
            rows,
            cols,
        })?;
        let out = Buf::new(rows * cols * 2)?;
        // SAFETY: the pack and the buffer are live and sized for `rows` rows.
        unsafe { lib.unpack(&pb.pack, 0, rows as i64, out.ptr, dev.stream)? };
        dev.sync()?;
        let mut back = vec![0u16; rows * cols];
        out.read(0, &mut back)?;
        let differ = back.iter().zip(&w).filter(|(x, y)| x != y).count();
        if differ == 0 {
            exact += 1;
            println!("exact    {name} [{rows} x {cols}]");
        } else {
            bad += 1;
            println!("DIFFERS  {name}: {differ} of {} weights", w.len());
        }
    }
    println!("{exact} exact on the GPU, {fallback} lossless instead, {bad} that differ");
    if bad > 0 {
        return err("the GPU's i4x decode differs from the checkpoint");
    }
    Ok(())
}

#[cfg(all(test, feature = "dev"))]
mod tests {
    use super::ConfigG4;

    /// generation_config.json's end tokens join config.json's; a file that is there and does not parse is refused by name, not passed over.
    #[test]
    fn a_generation_config_is_read_or_refused() {
        let text = crate::logits::GEMMA4_RANDOM_CONFIG;
        let c = ConfigG4::parse(text, Some(r#"{"eos_token_id": [1, 106, 50]}"#), false).unwrap();
        assert_eq!(c.eos, [1, 106, 50]);
        assert_eq!(ConfigG4::parse(text, None, false).unwrap().eos, [1, 106]);
        let e = ConfigG4::parse(text, Some(r#"{"eos_token_id": [1, 106, 50"#), false).unwrap_err().msg;
        assert!(e.contains("generation_config.json"), "{e}");
    }
}
