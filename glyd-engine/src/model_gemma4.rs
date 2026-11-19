//! Gemma 4 (2026) text models on the GPU, one sequence: 31B, 12B (dense) at every level (the matrices packed by the library's layouts), the forward pass of its two kinds of
//! layer (kernels/gemma4.cu), a prompt in chunks and a decode step as one CUDA graph, greedy tokens. The mixture-of-experts 26B-A4B and the E models (per-layer embeddings,
//! shared key-value layers) are refused here (their parts come later).
//!
//! A layer (research/gemma4-for-the-engine.md): h = norm_in(x); x += norm_post_attn(attention(h)); h = norm_pre_ff(x); x += norm_post_ff(down(gelu(gate h) * up h)); x *= scalar.
//! Sliding layers (5 of 6): head 256, keys and values of the last `window` positions (a ring), the full rotary at theta 1e4. Global layers: head 512, the keys' projection
//! as the values too where the config says K = V, the proportional rotary (64 of 256 pairs) at theta 1e6. Attention scale 1. The logits soft-capped (30 tanh(l / 30)).

use crate::dev::{copy_d2h_async, Buf, Dev, Ev, Graph, Kernel, Pinned};
use crate::ffi::{err, Error, Lib, Result};
use crate::kern::{norm_threads, Kern};
use crate::launch;
use crate::model::{embed_plan, Sample, TopGpu};
use crate::weights::{choose_layout, ordered, pack_item_map, upload_bytes, upload_packed, Item, Layout, PackBuf};
use glyd_gpu::json::{self, Value};
use glyd_gpu::safetensors::Checkpoint;
use std::path::Path;

#[cfg(feature = "dev")]
macro_rules! kname {
    ($n:literal) => {
        $n
    };
}
#[cfg(not(feature = "dev"))]
include!(concat!(env!("OUT_DIR"), "/kernel_names.rs"));

/// Keys a tile of the attention kernels (G4_KB in kernels/gemma4.cu).
pub(crate) const KB: usize = 32;
/// The parts a decode step's attention splits a row's keys into.
pub const SPLITS: usize = 16;
/// The warps of a decode step's attention block (g4_attn*_parts): the G that score its row and more that only load the key and value tiles, so a short context is not
/// one warp pair's loads in a row (GLY-312: 0.75 ms of a 12B step at 30 tokens with 2 warps).
pub(crate) const PART_WARPS: i32 = 16;
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

impl ConfigG4 {
    pub fn load(dir: &Path) -> Result<ConfigG4> {
        ConfigG4::load_for(dir, false)
    }

    /// `draft`: a Gemma 4 "-assistant" draft model (its layers all read the big model's keys and values: no key or value projections; its tensors under "model.").
    pub fn load_for(dir: &Path, draft: bool) -> Result<ConfigG4> {
        let text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| Error(format!("{}: {e}", dir.join("config.json").display())))?;
        ConfigG4::parse(&text, std::fs::read_to_string(dir.join("generation_config.json")).ok().as_deref(), draft)
    }

    /// `load_for` from the texts of config.json and generation_config.json.
    pub fn parse(text: &str, gen: Option<&str>, draft: bool) -> Result<ConfigG4> {
        let top = json::parse(text).map_err(|e| Error(format!("config.json: {e}")))?;
        let c = top.get("text_config").unwrap_or(&top);
        let num = |k: &str| c.get(k).and_then(Value::as_u64).map(|x| x as usize);
        let need = |k: &str| num(k).ok_or_else(|| Error(format!("config.json: no {k}")));
        let mt = c.get("model_type").and_then(Value::as_str).unwrap_or("");
        if !mt.starts_with("gemma4") {
            return err(format!("config.json: model_type {mt}, not Gemma 4"));
        }
        if c.get("enable_moe_block").and_then(Value::as_bool) == Some(true) {
            return err("Gemma 4's mixture of experts (26B-A4B) is not in this engine yet");
        }
        if num("hidden_size_per_layer_input").unwrap_or(0) > 0 || (!draft && num("num_kv_shared_layers").unwrap_or(0) > 0) {
            return err("Gemma 4's E models (per-layer embeddings, shared key-value layers) are not in this engine yet");
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
        let f = |v: Option<&Value>| match v {
            Some(Value::Number(n)) => n.parse::<f64>().ok(),
            _ => None,
        };
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
            if let Ok(gc) = json::parse(gtext) {
                if let Some(Value::Array(a)) = gc.get("eos_token_id") {
                    for x in a.iter().filter_map(Value::as_u64) {
                        if !eos.contains(&(x as i64)) {
                            eos.push(x as i64);
                        }
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

fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// cos and sin (bf16 [max_pos, pairs]) of pairs i < `pairs` of a head of `dim`: inv_freq_i = theta^(-2i / dim), as transformers' default and proportional rotaries (the
/// latter's other pairs are cos 1, sin 0: not stored, the kernel leaves them).
pub(crate) fn rotary(theta: f64, dim: usize, pairs: usize, max_pos: usize) -> (Vec<u16>, Vec<u16>) {
    let inv: Vec<f32> = (0..pairs).map(|i| 1.0f32 / (theta as f32).powf((2 * i) as f32 / dim as f32)).collect();
    let (mut c, mut s) = (Vec::with_capacity(max_pos * pairs), Vec::with_capacity(max_pos * pairs));
    for p in 0..max_pos {
        for &f in &inv {
            let a = p as f32 * f;
            c.push(bf16_bits(a.cos()));
            s.push(bf16_bits(a.sin()));
        }
    }
    (c, s)
}

/// Whether config.json's text is a Gemma 4 model's (its model_type, or its text config's).
pub fn is_gemma4(text: &str) -> bool {
    let Ok(v) = json::parse(text) else { return false };
    let gemma = |c: &Value| c.get("model_type").and_then(Value::as_str).is_some_and(|m| m.starts_with("gemma4"));
    gemma(&v) || v.get("text_config").is_some_and(gemma)
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

    /// The bytes on the GPU of the weights at `layout` (`saved`: a glyd-v7 directory's, read as saved) and of the bf16 model: the matrices and the small tensors (norms, scalars).
    pub fn weight_bytes(&self, layout: Layout, saved: Option<&Checkpoint>) -> (u64, u64) {
        let (mut w, mut bf16) = (0u64, 0u64);
        for it in self.items() {
            let b = (it.rows * it.cols * 2) as u64;
            bf16 += b;
            let key = it.members[0].trim_end_matches(".weight");
            w += saved.and_then(|ck| crate::weights::packed_bytes(ck, key)).unwrap_or((b as f64 * crate::fit::bits(layout) / 16.0) as u64);
        }
        let small = (self.layers * (4 * self.hidden + 2 * self.global_head_dim + 1) + self.hidden) as u64 * 2;
        (w + small, bf16 + small)
    }

    /// The keys and values: (the sliding layers' rings, fixed: the window and a chunk each, or the context where it is shorter; a position's bytes in the global layers).
    pub fn kv_bytes(&self, chunk: usize, ctx: usize) -> (u64, u64) {
        let cap_s = (self.window + chunk).min(ctx) as u64;
        let (mut fixed, mut per) = (0u64, 0u64);
        for &k in &self.kinds {
            let (d, nk, _) = self.attn_of(k);
            let b = (nk * d * 2 * 2) as u64;
            match k {
                KindG4::Sliding => fixed += cap_s * b,
                KindG4::Global => per += b,
            }
        }
        (fixed, per)
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

/// The draft model and its buffers in an engine (`OptsG4::draft`).
struct Drafter {
    m: DraftG4,
    /// Draft tokens a round.
    k: usize,
    /// The input [2 big hidden], the big model's hidden state for the next draft [big hidden], the residual and the normed [hidden], the queries, the attention's
    /// output, o, gate-up, the product, down, the logits, the token and its value, and the positions [the queries', the last key's].
    din: Buf,
    hid: Buf,
    x: Buf,
    h: Buf,
    q: Buf,
    qo: Buf,
    att: Buf,
    o: Buf,
    gu: Buf,
    m_: Buf,
    d: Buf,
    logits: Buf,
    tok: Buf,
    val: Buf,
    pos: Buf,
}

/// kernels/gemma4.cu with typed launches.
pub struct KernG4 {
    pub(crate) norm: Kernel,
    pub(crate) embed_norm: Kernel,
    pub(crate) add_norm: Kernel,
    pub(crate) embed_norm_x32: Kernel,
    pub(crate) add_norm_x32: Kernel,
    pub(crate) prep256: Kernel,
    pub(crate) prep512: Kernel,
    pub(crate) attn256: Kernel,
    pub(crate) attn512: Kernel,
    pub(crate) parts256: Kernel,
    pub(crate) parts512: Kernel,
    pub(crate) merge: Kernel,
    pub(crate) gelu: Kernel,
    pub(crate) softcap: Kernel,
    pub(crate) draft_in: Kernel,
    pub(crate) fa256: Kernel,
    pub(crate) fa512: Kernel,
    pub(crate) fa512b: Kernel,
    /// Whether prompt chunks take the tensor-core attention (g4_fa256, g4_fa512b; GLYD_G4_FA=0: g4_attn_t).
    pub(crate) fa: bool,
}

/// Dynamic shared memory of g4_fa512b: K and V [32][520] bf16 and the score shares [2][4][16][33] floats.
pub(crate) const FA512B_SMEM: u32 = (2 * 32 * 520 * 2 + 2 * 4 * 16 * 33 * 4) as u32;

/// Dynamic shared memory of a tensor-core attention block (g4_fa_t): Q [BM][D + 8], K [32][D + 8], V [32][DV + 8] bf16.
pub(crate) fn fa_smem(d: usize, dv: usize, bm: usize) -> u32 {
    ((bm * (d + 8) + 32 * (d + 8) + 32 * (dv + 8)) * 2) as u32
}

/// Dynamic shared memory of an attention block of head d and r g warps.
pub(crate) fn attn_smem(d: usize, rg: usize) -> u32 {
    ((KB * (d / 2 + 1) + KB * d / 2 + rg * d / 2) * 4) as u32
}

impl KernG4 {
    pub fn load(dev: &Dev) -> Result<KernG4> {
        let k = KernG4 {
            norm: dev.kernel(kname!("g4_rmsnorm"))?,
            embed_norm: dev.kernel(kname!("g4_embed_norm"))?,
            add_norm: dev.kernel(kname!("g4_add_norm"))?,
            embed_norm_x32: dev.kernel(kname!("g4_embed_norm_x32"))?,
            add_norm_x32: dev.kernel(kname!("g4_add_norm_x32"))?,
            prep256: dev.kernel(kname!("g4_prep256"))?,
            prep512: dev.kernel(kname!("g4_prep512"))?,
            attn256: dev.kernel(kname!("g4_attn256"))?,
            attn512: dev.kernel(kname!("g4_attn512"))?,
            parts256: dev.kernel(kname!("g4_attn256_parts"))?,
            parts512: dev.kernel(kname!("g4_attn512_parts"))?,
            merge: dev.kernel(kname!("g4_attn_merge"))?,
            gelu: dev.kernel(kname!("g4_gelu_mul"))?,
            softcap: dev.kernel(kname!("g4_softcap"))?,
            draft_in: dev.kernel(kname!("g4_draft_in"))?,
            fa256: dev.kernel(kname!("g4_fa256"))?,
            fa512: dev.kernel(kname!("g4_fa512"))?,
            fa512b: dev.kernel(kname!("g4_fa512b"))?,
            // (on: against transformers bf16 on 31B QAT, wikitext 32 x 512, KLD 0.0192 where g4_attn_t's is 0.0225, and a 28K prompt 2x faster;
            // GLYD_G4_FA=0 for g4_attn_t)
            fa: {
                // (the switch back, a development build's alone: a release build reads no GLYD_ variable it does not list)
                #[cfg(feature = "dev")]
                let on = std::env::var("GLYD_G4_FA").map_or(true, |v| v != "0");
                #[cfg(not(feature = "dev"))]
                let on = true;
                on
            },
        };
        k.fa256.allow_smem(fa_smem(256, 256, 64))?;
        k.fa512.allow_smem(fa_smem(512, 256, 32))?;
        k.fa512b.allow_smem(FA512B_SMEM)?;
        for (kern, d) in [(&k.attn256, 256), (&k.parts256, 256), (&k.attn512, 512), (&k.parts512, 512)] {
            kern.allow_smem(attn_smem(d, 16))?;
        }
        Ok(k)
    }
}

pub struct OptsG4 {
    pub ctx: usize,
    pub chunk: usize,
    pub graph: bool,
    pub device: i32,
    pub layout: Option<Layout>,
    /// A draft model's directory (Gemma 4's "-assistant") and its draft tokens a round (`spec_round`).
    pub draft: Option<(std::path::PathBuf, usize)>,
}

impl Default for OptsG4 {
    fn default() -> OptsG4 {
        OptsG4 { ctx: 8192, chunk: 1024, graph: true, device: 0, layout: None, draft: None }
    }
}

/// kernels/gemma4.cu's G4Pf: a small kernel's spare blocks prefetch into L2 the start of the first-wave blocks of the skinny product that comes next (GLY-312).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct G4Pf {
    pub(crate) data: u64,
    pub(crate) scales: u64,
    pub(crate) dstride: i64,
    pub(crate) sstride: i64,
    pub(crate) dhead: i32,
    pub(crate) shead: i32,
    pub(crate) regions: i32,
    pub(crate) shift: i32,
}

pub struct EngineG4 {
    pub dev: Dev,
    lib: Lib,
    kern: Kern,
    kg: KernG4,
    /// The skinny products (kernels/skinny.cu) for 1 to 16 rows (a decode step and a verification of 8 drafts: the same sums row for row) of the Int4, Int5, Int6 and exact four-bit layouts: one block a row block over all of K, no parts to sum (GLY-312);
    /// GLYD_G4_SKINNY=0 (a development build's) leaves them to the library.
    skinny: crate::skinny::SkinnyKern,
    skinny_on: bool,
    pub w: WeightsG4,
    chunk: usize,
    ctx: usize,
    /// The sliding layers' ring (positions it holds: the window and a chunk).
    cap_s: usize,
    x: Buf,
    /// GLYD_G4_PRECISE=1 (a development build's): the residual in fp32 [chunk, hidden] (g4_add_norm_x32), else None (bf16 in x, transformers' roundings).
    xf: Option<Buf>,
    /// GLYD_G4_ABLATE=stage,stage (a development build's): the stages a step leaves out, to time the rest (the output is not the model's); empty in a release build.
    ablate: String,
    /// GLYD_G4_PF "a,n,g": the KiB of each first-wave block of the next product that the attention's (a), the norms' (n) and gelu's (g) spare blocks prefetch into L2.
    pf_kib: [usize; 3],
    /// GLYD_G4_PF_SHIFT: a prefetch every 2^shift bytes (7: a line of 128).
    pf_shift: i32,
    h: Buf,
    proj: Buf,
    qo: Buf,
    att: Buf,
    o: Buf,
    gu: Buf,
    m: Buf,
    d: Buf,
    logits: Buf,
    scratch: Buf,
    embed_groups: Buf,
    embed_pairs: Buf,
    pos: Buf,
    st: Buf,
    tok: Buf,
    val: Buf,
    lse: Buf,
    pub top: TopGpu,
    part: Buf,
    ws: Buf,
    done: Buf,
    /// Each layer's key and value ring (bytes into kc and vc).
    kv_off: Vec<u64>,
    pub kc: Buf,
    pub vc: Buf,
    graph: Option<Graph>,
    host: Pinned,
    pub n: usize,
    /// Passes of up to this many rows take the decode step's split attention (1, or a drafted round's k + 1).
    split_rows: usize,
    drafter: Option<Drafter>,
    /// `--prof` (no graph): an event after each stage of a pass, to time the stages (`prof_take`).
    pub prof: Option<std::cell::RefCell<Vec<(&'static str, Ev)>>>,
}

impl EngineG4 {
    pub fn new(dir: &Path, o: &OptsG4) -> Result<EngineG4> {
        let dev = Dev::new(o.device)?;
        let lib = Lib::find()?;
        let kern = Kern::load(&dev)?;
        let kg = KernG4::load(&dev)?;
        let skinny = crate::skinny::SkinnyKern::load(&dev)?;
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get().min(16));
        let w = WeightsG4::load(dir, o.ctx + o.chunk + 16, threads, o.layout, dev.cc_pair(), dev.total_mem as u64)?;
        let c = &w.cfg;
        let ch = o.chunk.max(1);
        let ctx = o.ctx.div_ceil(16) * 16;
        let cap_s = (c.window + ch).min(ctx);
        let act = |cols: usize, rows: usize| Buf::new(rows * cols * 2);
        let hid = c.hidden;
        let proj_w = c.kinds.iter().map(|&k| c.proj_width(k)).max().unwrap_or(0);
        let q_w = c.heads * c.global_head_dim;
        let (x, h) = (act(hid, ch)?, act(hid, ch)?);
        let xf = if crate::tune!("GLYD_G4_PRECISE").is_some_and(|v| v == "1") { Some(Buf::new(ch * hid * 4)?) } else { None };
        let proj = act(proj_w, ch)?;
        let (qo, att) = (act(q_w, ch)?, act(q_w, ch)?);
        let (o_, gu, m, d) = (act(hid, ch)?, act(2 * c.inter, ch)?, act(c.inter, ch)?, act(hid, ch)?);
        let rows_out = o.draft.as_ref().map_or(1, |d| d.1 + 1);
        if rows_out > ch {
            return err(format!("{} draft tokens a round: the chunk ({ch}) must hold them and the last token", rows_out - 1));
        }
        let logits = act(c.vocab, rows_out)?;
        let scratch = Buf::new(64 * hid * 2)?;
        let embed_groups = Buf::new(EMBED_GROUPS * 64 * hid * 2)?;
        let embed_pairs = Buf::new(ch * 8)?;
        let (pos, st) = (Buf::zeroed(4 * ch)?, Buf::zeroed(16)?);
        let (tok, val, lse) = (Buf::zeroed(4 * rows_out)?, Buf::zeroed(4 * rows_out)?, Buf::zeroed(4)?);
        let top = TopGpu::new(c.vocab)?;
        // (a decode step's attention parts; a drafted round's k + 1 rows take the same kernel, so its tokens are what steps would give, bit for bit)
        let part = Buf::zeroed(rows_out * c.heads * SPLITS * (c.global_head_dim + 2) * 4)?;
        let mut need = 0usize;
        let mats: Vec<&PackBuf> = w.layers.iter().flat_map(|l| [&l.proj, &l.o, &l.gate_up, &l.down]).collect();
        let mut shapes: Vec<(i64, i64, bool)> = mats.iter().map(|p| (p.pack.rows, p.pack.cols, p.pack.is_twelve())).collect();
        shapes.sort_unstable();
        shapes.dedup();
        // (every row count a product takes: a prompt's last chunk, a drafted round's k + 1; the workspace is not the most at 1 or a chunk on every GPU)
        for m_ in 1..=ch as i64 {
            for &(oo, kk, twelve) in &shapes {
                need = need.max(lib.linear_workspace(twelve, oo, kk, m_)?);
            }
        }
        need = need.max(lib.linear_workspace(w.embed.pack.is_twelve(), w.embed.pack.rows, w.embed.pack.cols, rows_out as i64)?);
        let drafter = match &o.draft {
            None => None,
            Some((ddir, k)) => {
                let dm = DraftG4::load(ddir, c, threads, Layout::Int6)?;
                let dl = &dm.layers;
                for p_ in dl.iter().flat_map(|l| [&l.q, &l.o, &l.gate_up, &l.down]).chain([&dm.pre, &dm.post, &dm.head]) {
                    need = need.max(lib.linear_workspace(p_.pack.is_twelve(), p_.pack.rows, p_.pack.cols, 1)?);
                }
                let dc = &dm.cfg;
                let qd = dc.heads * dc.global_head_dim;
                Some(Drafter {
                    k: *k,
                    din: act(2 * hid, 1)?,
                    hid: act(hid, 1)?,
                    x: act(dc.hidden, 1)?,
                    h: act(dc.hidden, 1)?,
                    q: act(qd, 1)?,
                    qo: act(qd, 1)?,
                    att: act(qd, 1)?,
                    o: act(dc.hidden, 1)?,
                    gu: act(2 * dc.inter, 1)?,
                    m_: act(dc.inter, 1)?,
                    d: act(dc.hidden, 1)?,
                    logits: act(dc.vocab, 1)?,
                    tok: Buf::zeroed(4)?,
                    val: Buf::zeroed(4)?,
                    pos: Buf::zeroed(8)?,
                    m: dm,
                })
            }
        };
        let ws = Buf::zeroed(need)?;
        let done = Buf::zeroed(4 << 18)?;
        let mut kv_off = Vec::with_capacity(c.layers);
        let mut at = 0u64;
        for &k in &c.kinds {
            kv_off.push(at);
            let (dd, nk, _) = c.attn_of(k);
            let cap = if k == KindG4::Sliding { cap_s } else { ctx };
            at += (cap * nk * dd * 2) as u64;
        }
        let (kc, vc) = (Buf::zeroed(at as usize)?, Buf::zeroed(at as usize)?);
        let host = Pinned::new(4096)?;
        let mut e = EngineG4 {
            dev, lib, kern, kg, skinny, skinny_on: crate::tune!("GLYD_G4_SKINNY").map_or(true, |v| v != "0"), w, chunk: ch, ctx, cap_s, x, xf, ablate: crate::tune!("GLYD_G4_ABLATE").unwrap_or_default(), pf_kib: pf_kib(crate::tune!("GLYD_G4_PF").as_deref())?, pf_shift: crate::tune!("GLYD_G4_PF_SHIFT").and_then(|v| v.parse().ok()).filter(|v| (5..=7).contains(v)).unwrap_or(7), h, proj, qo, att, o: o_, gu, m, d, logits, scratch, embed_groups, embed_pairs, pos, st, tok, val, lse, top, part, ws, done, kv_off, kc, vc,
            graph: None, host, n: 0, split_rows: rows_out, drafter, prof: None,
        };
        if o.graph {
            e.capture_graph()?;
        }
        Ok(e)
    }

    pub fn describe(&self) -> String {
        let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
        let (free, total) = self.dev.mem_info().unwrap_or((0, 0));
        format!(
            "{} (sm_{}{}): Gemma 4 {} layers, weights {:.3} GiB (layout {:?}), KV {:.3} GiB ({} positions, {} in the sliding layers), {:.2} of {:.2} GiB in use",
            self.dev.name, self.dev.cc.0, self.dev.cc.1, self.w.cfg.layers, gib(self.w.bytes), self.w.layout, gib(self.kc.bytes + self.vc.bytes), self.ctx, self.cap_s, gib(total - free), gib(total)
        )
    }

    pub fn reset(&mut self) -> Result<()> {
        self.n = 0;
        self.top.ask = None;
        self.dev.sync()
    }

    /// The last forward pass's logits (soft-capped), widened.
    pub fn logits(&self) -> Result<Vec<f32>> {
        self.dev.sync()?;
        let mut words = vec![0u16; self.w.cfg.vocab];
        self.logits.read(0, &mut words)?;
        Ok(words.into_iter().map(|w| f32::from_bits((w as u32) << 16)).collect())
    }

    /// The end of a stage (where profiling): an event recorded on the stream.
    /// Whether a step runs `stage` (all of them, unless GLYD_G4_ABLATE names it).
    fn on(&self, stage: &str) -> bool {
        self.ablate.is_empty() || !self.ablate.split(',').any(|s| s == stage)
    }

    fn mark(&self, what: &'static str) -> Result<()> {
        if let Some(p) = &self.prof {
            let e = Ev::new()?;
            e.record(self.dev.stream)?;
            p.borrow_mut().push((what, e));
        }
        Ok(())
    }

    /// The stages' times since the last call (ms, summed by stage, in the order first seen), the events cleared.
    pub fn prof_take(&self) -> Result<Vec<(&'static str, f64)>> {
        let Some(p) = &self.prof else { return Ok(Vec::new()) };
        self.dev.sync()?;
        let evs = std::mem::take(&mut *p.borrow_mut());
        let mut out: Vec<(&'static str, f64)> = Vec::new();
        for w in evs.windows(2) {
            let ms = w[1].1.since(&w[0].1)? as f64;
            match out.iter_mut().find(|(k, _)| *k == w[1].0) {
                Some(x) => x.1 += ms,
                None => out.push((w[1].0, ms)),
            }
        }
        Ok(out)
    }

    fn skinny_takes(&self, p: &PackBuf, m: usize) -> bool {
        self.skinny_on && (1..=16).contains(&m) && p.pack.rows >= 2048 && crate::skinny::SkinnyKern::takes(&p.pack)
    }

    /// The prefetch of `kib` KiB of each first-wave block of `p` (where `lin` takes it skinny), and the spare blocks of `threads` that issue it (4 lines a thread; two
    /// blocks an SM or more measured slower, 11.26-11.32 ms against 11.22-11.24).
    fn pf(&self, p: &PackBuf, m: usize, kib: usize, threads: u32) -> (G4Pf, u32) {
        let Some((data, scales, ds, ss)) = crate::skinny::SkinnyKern::regions(&p.pack).filter(|_| kib > 0 && self.skinny_takes(p, m)) else { return (G4Pf::default(), 0) };
        let dhead = ((kib as u64 * 1024).min(ds) / 128 * 128) as i32;
        let shead = ((dhead as u64 * ss / ds).div_ceil(128) * 128).min(ss) as i32;
        let regions = ((p.pack.rows / 64) as i32).min(2 * self.dev.sms);
        let lines = regions as u32 * ((dhead + shead) >> self.pf_shift) as u32;
        (G4Pf { data, scales, dstride: ds as i64, sstride: ss as i64, dhead, shead, regions, shift: self.pf_shift }, lines.div_ceil(4 * threads))
    }

    unsafe fn lin(&self, p: &PackBuf, x: u64, m: usize, y: u64) -> Result<()> {
        if self.skinny_takes(p, m) {
            return self.skinny.gemv(self.dev.stream, &p.pack, x, y, m);
        }
        self.lib.linear(&p.pack, x, m as i64, y, self.ws.ptr, self.ws.bytes, self.done.ptr, self.dev.stream)
    }

    unsafe fn embed(&self, tokens: &[u32]) -> Result<()> {
        let hid = self.w.cfg.hidden;
        let s = self.dev.stream;
        if let Some(&t) = tokens.iter().find(|&&t| t as usize >= self.w.cfg.vocab) {
            return err(format!("token {t}: the vocabulary has {}", self.w.cfg.vocab));
        }
        if tokens.len() == 1 {
            let t = tokens[0] as i64;
            let row0 = (t / 64) * 64;
            self.lib.unpack(&self.w.embed.pack, row0, 64, self.scratch.ptr, s)?;
            return self.kern.copy_row(s, self.scratch.ptr, (t - row0) as i32, hid as i32, self.x.ptr);
        }
        let (groups, by_batch) = embed_plan(tokens, EMBED_GROUPS);
        let counts: Vec<usize> = by_batch.iter().map(Vec::len).collect();
        let pairs: Vec<i32> = by_batch.iter().flatten().flat_map(|&(d, s_)| [d, s_]).collect();
        self.embed_pairs.write(0, &pairs)?;
        let mut done = 0usize;
        for (b, batch) in groups.chunks(EMBED_GROUPS).enumerate() {
            for (j, &g) in batch.iter().enumerate() {
                self.lib.unpack(&self.w.embed.pack, g * 64, 64, self.embed_groups.ptr + (j * 64 * hid * 2) as u64, s)?;
            }
            self.kern.copy_rows(s, self.embed_groups.ptr, self.embed_pairs.ptr + (done * 8) as u64, counts[b] as i32, hid as i32, self.x.ptr)?;
            done += counts[b];
        }
        Ok(())
    }

    /// The layers on the n rows in x (positions at `pos`), then the final norm and, on the last `head` rows, the LM head, the soft-cap and the argmax (logits, tok and val
    /// a row each).
    unsafe fn forward(&self, n: usize, pos: u64, head: usize) -> Result<()> {
        let c = &self.w.cfg;
        let s = self.dev.stream;
        let (hid, nq) = (c.hidden as i32, c.heads as i32);
        let (nu, eps) = (n as u32, c.eps);
        let nt = norm_threads(hid)?;
        let null = 0u64;
        let first = &self.w.layers[0];
        self.mark("start")?;
        match &self.xf {
            Some(xf) => launch!(self.kg.embed_norm_x32, [nu, 1, 1], [nt, 1, 1], s; self.x.ptr, (c.hidden as f32).sqrt(), first.in_norm.ptr, xf.ptr, self.h.ptr, hid, eps)?,
            None => launch!(self.kg.embed_norm, [nu, 1, 1], [nt, 1, 1], s; self.x.ptr, c.embed_scale(), first.in_norm.ptr, self.h.ptr, hid, eps)?,
        }
        for (l, ly) in self.w.layers.iter().enumerate() {
            let (d, nk, has_v) = c.attn_of(ly.kind);
            let (di, nki) = (d as i32, nk as i32);
            let (kc, vc) = (self.kc.ptr + self.kv_off[l], self.vc.ptr + self.kv_off[l]);
            let (cap, window, cos, sin, rot) = match ly.kind {
                KindG4::Sliding => (self.cap_s as i32, c.window as i32, self.w.cos_s.ptr, self.w.sin_s.ptr, (c.head_dim / 2) as i32),
                KindG4::Global => (self.ctx as i32, 0i32, self.w.cos_g.ptr, self.w.sin_g.ptr, c.global_rot as i32),
            };
            if self.on("qkv") {
                self.lin(&ly.proj, self.h.ptr, n, self.proj.ptr)?;
            }
            self.mark(if ly.kind == KindG4::Sliding { "qkv (sliding)" } else { "qk (global)" })?;
            let stride = c.proj_width(ly.kind) as i32;
            let prep = if d == 256 { &self.kg.prep256 } else { &self.kg.prep512 };
            if self.on("prep") {
            let (pf, pfb) = self.pf(&ly.o, n, self.pf_kib[0], (d / 2) as u32);
            let slice = nu * (nq + nki) as u32;
            launch!(*prep, [nu, (nq + nki) as u32, 1 + pfb.div_ceil(slice)], [(d / 2) as u32, 1, 1], s; self.proj.ptr, stride, ly.q_norm.ptr, ly.k_norm.ptr, cos, sin, rot, pos, cap, self.qo.ptr, kc, vc, nq, nki, has_v as i32, eps, pf)?;
            }
            self.mark("prep")?;
            let g = nq / nki;
            if !self.on("attention") {
            } else if n <= self.split_rows {
                let parts = if d == 256 { &self.kg.parts256 } else { &self.kg.parts512 };
                let splits = SPLITS as i32;
                launch!(*parts, [nu, nki as u32, SPLITS as u32], [(32 * g.max(PART_WARPS)) as u32, 1, 1], smem attn_smem(d, g as usize), s; self.qo.ptr, kc, vc, pos, n as i32, nq, nki, window, cap, self.part.ptr, splits)?;
                if self.on("merge") {
                    launch!(self.kg.merge, [nu, nq as u32, 1], [di as u32, 1, 1], s; self.part.ptr, splits, self.att.ptr, nq, di)?;
                }
            } else if self.kg.fa {
                // (a prompt chunk: the tensor cores; a 512-wide head's values in two passes)
                if d == 256 {
                    launch!(self.kg.fa256, [(n.div_ceil(64)) as u32, nq as u32, 1], [128, 1, 1], smem fa_smem(256, 256, 64), s; self.qo.ptr, kc, vc, pos, self.att.ptr, n as i32, nq, nki, window, cap)?;
                } else {
                    launch!(self.kg.fa512b, [(n.div_ceil(32)) as u32, nq as u32, 1], [256, 1, 1], smem FA512B_SMEM, s; self.qo.ptr, kc, vc, pos, self.att.ptr, n as i32, nq, nki, window, cap)?;
                }
            } else {
                let r = (16 / g).max(1);
                let attn = if d == 256 { &self.kg.attn256 } else { &self.kg.attn512 };
                launch!(*attn, [((n as i32 + r - 1) / r) as u32, nki as u32, 1], [(32 * r * g) as u32, 1, 1], smem attn_smem(d, (r * g) as usize), s; self.qo.ptr, kc, vc, pos, self.att.ptr, n as i32, nq, nki, r, window, cap)?;
            }
            self.mark(if ly.kind == KindG4::Sliding { "attention (sliding)" } else { "attention (global)" })?;
            if self.on("o") {
                self.lin(&ly.o, self.att.ptr, n, self.o.ptr)?;
            }
            self.mark("o")?;
            match &self.xf {
                _ if !self.on("norms") => {}
                Some(xf) => launch!(self.kg.add_norm_x32, [nu, 1, 1], [nt, 1, 1], s; xf.ptr, self.o.ptr, ly.post_attn.ptr, null, ly.pre_ff.ptr, self.h.ptr, hid, eps)?,
                None => {
                    let (pf, pfb) = self.pf(&ly.gate_up, n, self.pf_kib[1], nt);
                    launch!(self.kg.add_norm, [nu, 1 + pfb.div_ceil(nu), 1], [nt, 1, 1], s; self.x.ptr, self.o.ptr, ly.post_attn.ptr, null, ly.pre_ff.ptr, self.h.ptr, hid, eps, pf)?
                }
            }
            self.mark("norms")?;
            if self.on("gate_up") {
                self.lin(&ly.gate_up, self.h.ptr, n, self.gu.ptr)?;
            }
            self.mark("gate_up")?;
            let inter = c.inter as i32;
            if self.on("gelu") {
            let gx = (c.inter as u32).div_ceil(256).min(64);
            let (pf, pfb) = self.pf(&ly.down, n, self.pf_kib[2], 256);
            launch!(self.kg.gelu, [gx, nu, 1 + pfb.div_ceil(gx * nu)], [256, 1, 1], s; self.gu.ptr, self.m.ptr, inter, pf)?;
            }
            self.mark("gelu")?;
            if self.on("down") {
                self.lin(&ly.down, self.m.ptr, n, self.d.ptr)?;
            }
            self.mark("down")?;
            let next = if l + 1 < self.w.layers.len() { self.w.layers[l + 1].in_norm.ptr } else { self.w.norm.ptr };
            match &self.xf {
                _ if !self.on("norms") => {}
                Some(xf) => launch!(self.kg.add_norm_x32, [nu, 1, 1], [nt, 1, 1], s; xf.ptr, self.d.ptr, ly.post_ff.ptr, ly.scalar.ptr, next, self.h.ptr, hid, eps)?,
                None => {
                    let to = if l + 1 < self.w.layers.len() { &self.w.layers[l + 1].proj } else { &self.w.embed };
                    let (pf, pfb) = self.pf(to, if l + 1 < self.w.layers.len() { n } else { head }, self.pf_kib[1], nt);
                    launch!(self.kg.add_norm, [nu, 1 + pfb.div_ceil(nu), 1], [nt, 1, 1], s; self.x.ptr, self.d.ptr, ly.post_ff.ptr, ly.scalar.ptr, next, self.h.ptr, hid, eps, pf)?
                }
            }
            self.mark("norms")?;
        }
        if head > 0 {
            let first = ((n - head) * c.hidden * 2) as u64;
            self.lin(&self.w.embed, self.h.ptr + first, head, self.logits.ptr)?;
            if c.softcap > 0.0 {
                launch!(self.kg.softcap, [64, 1, 1], [256, 1, 1], s; self.logits.ptr, (head * c.vocab) as i32, c.softcap)?;
            }
            self.kern.argmax_rows(s, self.logits.ptr, c.vocab as i32, head as u32, self.tok.ptr, self.val.ptr)?;
            self.mark("head")?;
        }
        Ok(())
    }

    fn capture_graph(&mut self) -> Result<()> {
        self.st.write(0, &[0i32])?;
        // SAFETY: live buffers of the engine; the stream is synchronized after.
        unsafe {
            self.embed(&[0u32])?;
            self.forward(1, self.st.ptr, 1)?;
        }
        self.dev.sync()?;
        let g = self.dev.capture(|| {
            // SAFETY: as above.
            unsafe { self.forward(1, self.st.ptr, 1) }
        })?;
        self.graph = Some(g);
        self.reset()
    }

    fn read_sample(&mut self) -> Result<Sample> {
        // SAFETY: the pinned buffer holds 4096 bytes; the copies complete at the synchronize.
        unsafe {
            copy_d2h_async(self.host.ptr, self.tok.ptr, 4, self.dev.stream)?;
            copy_d2h_async((self.host.ptr as *mut u8).add(8) as *mut _, self.val.ptr, 4, self.dev.stream)?;
            self.top.queue(&self.kern, &self.dev, self.logits.ptr, self.val.ptr)?;
        }
        self.dev.sync()?;
        // SAFETY: the pinned bytes were just written.
        let (token, logit) = unsafe { (*(self.host.ptr as *const i32), *((self.host.ptr as *const u8).add(8) as *const f32)) };
        Ok(Sample { token: token as u32, logit, logprob: None })
    }

    /// The prompt after the tokens already in the cache, in chunks; the token that follows it.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Sample> {
        if tokens.is_empty() {
            return err("an empty prompt");
        }
        if self.n + tokens.len() + 1 > self.ctx {
            return err(format!("{} tokens and {} more, where the cache holds {}", self.n, tokens.len(), self.ctx));
        }
        let mut at = 0;
        while at < tokens.len() {
            let n = self.chunk.min(tokens.len() - at);
            let pos: Vec<i32> = (0..n).map(|i| (self.n + i) as i32).collect();
            self.dev.sync()?;
            self.pos.write(0, &pos)?;
            let last = at + n == tokens.len();
            // SAFETY: live buffers of the engine, sized for a chunk.
            unsafe {
                self.embed(&tokens[at..at + n])?;
                self.forward(n, self.pos.ptr, last as usize)?;
                if let (true, Some(dr)) = (last, &self.drafter) {
                    // (the draft model's first input: the hidden state of the prompt's last position)
                    self.kern.copy_row(self.dev.stream, self.h.ptr, n as i32 - 1, self.w.cfg.hidden as i32, dr.hid.ptr)?;
                }
            }
            self.n += n;
            at += n;
        }
        self.read_sample()
    }

    /// One token through the model (the graph where there is one); the token after it.
    pub fn step(&mut self, token: u32) -> Result<Sample> {
        if self.n + 1 > self.ctx {
            return err("the cache is full");
        }
        self.dev.sync()?;
        self.st.write(0, &[self.n as i32])?;
        // SAFETY: live buffers; the graph was captured on them.
        unsafe {
            self.embed(&[token])?;
            match &self.graph {
                Some(g) => g.launch(self.dev.stream)?,
                None => self.forward(1, self.st.ptr, 1)?,
            }
        }
        self.n += 1;
        self.read_sample()
    }

    pub fn capacity(&self) -> usize {
        self.ctx
    }

    /// Whether a draft model is loaded (`spec_round`).
    pub fn drafting(&self) -> bool {
        self.drafter.is_some()
    }

    /// One draft token: the draft model on x's first row (the raw embedding of the token drafted from) and its hidden-state input, the queries at position
    /// pos[0], the keys to pos[1]; the token to its `tok`, the next hidden state to its input.
    unsafe fn draft_one(&self, dr: &Drafter) -> Result<()> {
        let (c, dc) = (&self.w.cfg, &dr.m.cfg);
        let s = self.dev.stream;
        let (hd, eps) = (dc.hidden as i32, dc.eps);
        let nt = norm_threads(hd)?;
        let nq = dc.heads as i32;
        let lin = |p: &PackBuf, x: u64, y: u64| self.lin(p, x, 1, y);
        launch!(self.kg.draft_in, [((2 * c.hidden) as u32).div_ceil(256).min(64), 1, 1], [256, 1, 1], s; self.x.ptr, c.embed_scale(), dr.hid.ptr, dr.din.ptr, c.hidden as i32)?;
        lin(&dr.m.pre, dr.din.ptr, dr.x.ptr)?;
        launch!(self.kg.norm, [1, 1, 1], [nt, 1, 1], s; dr.x.ptr, dr.m.layers[0].in_norm.ptr, dr.h.ptr, hd, eps)?;
        // the big model's last layer of each kind: the keys and values the draft model reads
        let last_of = |k: KindG4| c.kinds.iter().rposition(|&x| x == k).expect("a layer of each kind");
        let (qpos, kpos) = (dr.pos.ptr, dr.pos.ptr + 4);
        for (l, ly) in dr.m.layers.iter().enumerate() {
            let src = last_of(ly.kind);
            let (d, nk, _) = c.attn_of(ly.kind);
            let nki = nk as i32;
            let (kc, vc) = (self.kc.ptr + self.kv_off[src], self.vc.ptr + self.kv_off[src]);
            let (cap, window, cos, sin, rot) = match ly.kind {
                KindG4::Sliding => (self.cap_s as i32, c.window as i32, self.w.cos_s.ptr, self.w.sin_s.ptr, (c.head_dim / 2) as i32),
                KindG4::Global => (self.ctx as i32, 0i32, self.w.cos_g.ptr, self.w.sin_g.ptr, c.global_rot as i32),
            };
            lin(&ly.q, dr.h.ptr, dr.q.ptr)?;
            let prep = if d == 256 { &self.kg.prep256 } else { &self.kg.prep512 };
            // (queries alone: no key or value heads, so nothing is written to the big model's cache)
            launch!(*prep, [1, nq as u32, 1], [(d / 2) as u32, 1, 1], s; dr.q.ptr, nq * d as i32, ly.q_norm.ptr, ly.q_norm.ptr, cos, sin, rot, qpos, cap, dr.qo.ptr, kc, vc, nq, 0i32, 0i32, eps, G4Pf::default())?;
            let g = nq / nki;
            let parts = if d == 256 { &self.kg.parts256 } else { &self.kg.parts512 };
            launch!(*parts, [1, nki as u32, SPLITS as u32], [(32 * g.max(PART_WARPS)) as u32, 1, 1], smem attn_smem(d, g as usize), s; dr.qo.ptr, kc, vc, kpos, 1i32, nq, nki, window, cap, self.part.ptr, SPLITS as i32)?;
            launch!(self.kg.merge, [1, nq as u32, 1], [d as u32, 1, 1], s; self.part.ptr, SPLITS as i32, dr.att.ptr, nq, d as i32)?;
            lin(&ly.o, dr.att.ptr, dr.o.ptr)?;
            launch!(self.kg.add_norm, [1, 1, 1], [nt, 1, 1], s; dr.x.ptr, dr.o.ptr, ly.post_attn.ptr, 0u64, ly.pre_ff.ptr, dr.h.ptr, hd, eps, G4Pf::default())?;
            lin(&ly.gate_up, dr.h.ptr, dr.gu.ptr)?;
            launch!(self.kg.gelu, [(dc.inter as u32).div_ceil(256).min(64), 1, 1], [256, 1, 1], s; dr.gu.ptr, dr.m_.ptr, dc.inter as i32, G4Pf::default())?;
            lin(&ly.down, dr.m_.ptr, dr.d.ptr)?;
            let next = if l + 1 < dr.m.layers.len() { dr.m.layers[l + 1].in_norm.ptr } else { dr.m.norm.ptr };
            launch!(self.kg.add_norm, [1, 1, 1], [nt, 1, 1], s; dr.x.ptr, dr.d.ptr, ly.post_ff.ptr, ly.scalar.ptr, next, dr.h.ptr, hd, eps, G4Pf::default())?;
        }
        lin(&dr.m.post, dr.h.ptr, dr.hid.ptr)?;
        lin(&dr.m.head, dr.h.ptr, dr.logits.ptr)?;
        self.kern.argmax_rows(s, dr.logits.ptr, dc.vocab as i32, 1, dr.tok.ptr, dr.val.ptr)
    }

    /// One round of drafting (greedy): from `last`, the token after the cache's (not in it yet), the draft model proposes k tokens, then the big model reads `last` and
    /// them in one pass and keeps the drafts up to its first disagreement, and its own token there. The tokens it returns (1 to k + 1) are the big model's greedy tokens,
    /// as `step` would give them one by one; the cache then holds `last` and the kept drafts.
    pub fn spec_round(&mut self, last: u32) -> Result<Vec<u32>> {
        self.spec_round_sampled(last, None)
    }

    /// `spec_round` sampling (`sampler`: the request's): each draft token is checked by `Sampler::check_draft` on the big model's logits for its position, in order, the first one
    /// it replaces ending the round; where all are kept the round's last token is drawn from the last row. The tokens are distributed as steps' draws would be.
    pub fn spec_round_sampled(&mut self, last: u32, sampler: Option<&mut crate::sampler::Sampler>) -> Result<Vec<u32>> {
        let Some(dr) = self.drafter.take() else { return err("no draft model") };
        let r = self.spec_round_with(&dr, last, sampler);
        self.drafter = Some(dr);
        r
    }

    fn spec_round_with(&mut self, dr: &Drafter, last: u32, sampler: Option<&mut crate::sampler::Sampler>) -> Result<Vec<u32>> {
        let n0 = self.n;
        let k = dr.k.min(self.ctx.saturating_sub(n0 + 1));
        if n0 == 0 {
            return err("spec_round before a prompt");
        }
        let mut seq = vec![last];
        self.dev.sync()?;
        dr.pos.write(0, &[n0 as i32, n0 as i32 - 1])?;
        for _ in 0..k {
            let t = *seq.last().expect("a token");
            // SAFETY: live buffers of the engine and the drafter; the token is read back after the synchronize.
            unsafe {
                self.embed(&[t])?;
                self.draft_one(dr)?;
                copy_d2h_async(self.host.ptr, dr.tok.ptr, 4, self.dev.stream)?;
            }
            self.dev.sync()?;
            // SAFETY: the pinned bytes were just written.
            seq.push(unsafe { *(self.host.ptr as *const i32) } as u32);
        }
        let n = seq.len();
        let pos: Vec<i32> = (0..n).map(|i| (n0 + i) as i32).collect();
        self.pos.write(0, &pos)?;
        // SAFETY: live buffers sized for k + 1 rows (OptsG4::draft); the tokens are read after the synchronize.
        unsafe {
            self.embed(&seq)?;
            self.forward(n, self.pos.ptr, n)?;
            copy_d2h_async(self.host.ptr, self.tok.ptr, 4 * n, self.dev.stream)?;
        }
        self.dev.sync()?;
        // SAFETY: the pinned buffer holds n ints, just written.
        let big: Vec<u32> = unsafe { std::slice::from_raw_parts(self.host.ptr as *const i32, n) }.iter().map(|&t| t as u32).collect();
        let (out, kept) = match sampler {
            None => {
                let kept = (0..k).take_while(|&i| seq[i + 1] == big[i]).count();
                (big[..=kept].to_vec(), kept)
            }
            Some(sm) => {
                let v = self.w.cfg.vocab;
                let mut words = vec![0u16; n * v];
                self.logits.read(0, &mut words)?;
                let row = |i: usize| -> Vec<f32> { words[i * v..(i + 1) * v].iter().map(|&w| f32::from_bits((w as u32) << 16)).collect() };
                let mut out = Vec::with_capacity(n);
                for i in 0..=k {
                    if i == k {
                        out.push(sm.sample(&row(i)));
                        break;
                    }
                    let (t, same) = sm.check_draft(&row(i), seq[i + 1]);
                    out.push(t);
                    if !same {
                        break;
                    }
                }
                let kept = out.len() - 1;
                (out, kept)
            }
        };
        // SAFETY: h holds the n rows' final hidden states.
        unsafe { self.kern.copy_row(self.dev.stream, self.h.ptr, kept as i32, self.w.cfg.hidden as i32, dr.hid.ptr)? };
        self.n = n0 + kept + 1;
        Ok(out)
    }
}

impl crate::conv::Gen for EngineG4 {
    fn capacity(&self) -> usize {
        self.ctx
    }

    fn reply(&mut self, prompt: &[u32], o: &crate::model::Reply, each: &mut dyn FnMut(u32) -> bool) -> Result<(Vec<u32>, crate::model::Finish)> {
        use crate::model::Finish;
        if prompt.is_empty() || prompt.len() + 1 > self.ctx {
            return err(format!("{} prompt tokens, where the context holds {}", prompt.len(), self.ctx));
        }
        self.reset()?;
        let greedy = o.params.greedy();
        let mut sampler = crate::sampler::Sampler::new(o.params.clone(), prompt);
        self.top.ask = sampler.top_need();
        let mut s = self.prefill(prompt)?;
        let mut out = Vec::new();
        if self.drafting() {
            // drafted: the big model's tokens a round at a time (the last of a round is not in the cache yet: the next round starts from it); sampled replies check each draft by the request's sampler
            let first = if greedy { s.token } else { sampler.sample_with(self.top.read().as_ref(), || self.logits())? };
            let mut pending = std::collections::VecDeque::from([first]);
            while let Some(tok) = pending.pop_front() {
                if !o.ignore_eos && (self.w.cfg.eos.contains(&(tok as i64)) || o.stop_ids.contains(&tok)) {
                    return Ok((out, Finish::Stop));
                }
                out.push(tok);
                if !each(tok) {
                    return Ok((out, Finish::Stopped));
                }
                if out.len() >= o.max_new || self.n + 1 > self.ctx {
                    return Ok((out, Finish::Length));
                }
                if pending.is_empty() {
                    pending.extend(self.spec_round_sampled(tok, if greedy { None } else { Some(&mut sampler) })?);
                }
            }
        }
        loop {
            let tok = if greedy { s.token } else { sampler.sample_with(self.top.read().as_ref(), || self.logits())? };
            if !o.ignore_eos && (self.w.cfg.eos.contains(&(tok as i64)) || o.stop_ids.contains(&tok)) {
                return Ok((out, Finish::Stop));
            }
            out.push(tok);
            if !each(tok) {
                return Ok((out, Finish::Stopped));
            }
            if out.len() >= o.max_new || self.n + 1 > self.ctx {
                return Ok((out, Finish::Length));
            }
            self.top.ask = sampler.top_need();
            s = self.step(tok)?;
        }
    }
}

/// `g4check MODEL REF.json [--layout L] [--ctx N] [--chunk N] [--no-graph]`: the engine against transformers' reference (research/gemma4-engine/ref_gemma4.py): each prompt
/// read as a prompt, then its reference continuation fed token by token; at every position from the prompt's last on, the engine's top token against the reference's, and
/// the reference's top-8 log-probabilities against the engine's for the same tokens (a lower bound of the KL divergence over them).
/// What g4check needs of an engine: today's (EngineG4) or the dynamic one's (`--dynamic`, GLY-326), to compare them on the same prompts.
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
impl Pass for EngineG4 {
    fn reset(&mut self) -> Result<()> {
        EngineG4::reset(self)
    }
    fn prefill(&mut self, t: &[u32]) -> Result<Sample> {
        EngineG4::prefill(self, t)
    }
    fn step(&mut self, t: u32) -> Result<Sample> {
        EngineG4::step(self, t)
    }
    fn logits(&self) -> Result<Vec<f32>> {
        EngineG4::logits(self)
    }
    fn prof_take(&self) -> Result<Vec<(&'static str, f64)>> {
        EngineG4::prof_take(self)
    }
    fn spec(&mut self, last: u32) -> Result<Vec<u32>> {
        EngineG4::spec_round(self, last)
    }
    fn pos(&self) -> usize {
        self.n
    }
    fn ctx(&self) -> usize {
        self.ctx
    }
    fn eos(&self) -> Vec<i64> {
        self.w.cfg.eos.clone()
    }
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

#[cfg(feature = "dev")]
pub fn check(a: &crate::args::Args) -> Result<()> {
    let dir = Path::new(a.need(1, "MODEL")?);
    let refp = a.need(2, "REF")?;
    let text = std::fs::read_to_string(refp).map_err(|e| Error(format!("{refp}: {e}")))?;
    let r = json::parse(&text).map_err(|e| Error(format!("{refp}: {e}")))?;
    let layout = a.layout()?;
    let o = OptsG4 { ctx: a.num("ctx", 8192usize)?, chunk: a.num("chunk", 1024usize)?, graph: !a.has("no-graph"), layout, ..OptsG4::default() };
    let t0 = std::time::Instant::now();
    let prof = a.has("prof");
    let o = OptsG4 { graph: o.graph && !prof, ..o };
    let mut e: Box<dyn Pass> = if a.flag("split").is_some() {
        let c = ConfigG4::load(dir)?;
        let so = crate::dynamic::Opts { ctx: o.ctx, chunk: o.chunk, graph: o.graph, device: o.device, layout: o.layout };
        let sp = split_of(a, dir, crate::dynamic::gemma4::describe(&c), &so)?.expect("--split given");
        eprintln!("loaded in {:.1} s: {}", t0.elapsed().as_secs_f64(), sp.describe());
        Box::new(sp)
    } else if a.has("dynamic") {
        let c = ConfigG4::load(dir)?;
        let mut d = crate::dynamic::Dyn::new(dir, crate::dynamic::gemma4::describe(&c), None, &crate::dynamic::Opts { ctx: o.ctx, chunk: o.chunk, graph: o.graph, device: o.device, layout: o.layout })?;
        if prof {
            d.prof = Some(Default::default());
        }
        eprintln!("loaded in {:.1} s: {}", t0.elapsed().as_secs_f64(), d.describe());
        Box::new(d)
    } else {
        let mut g = EngineG4::new(dir, &o)?;
        if prof {
            g.prof = Some(Default::default());
        }
        eprintln!("loaded in {:.1} s: {}", t0.elapsed().as_secs_f64(), g.describe());
        Box::new(g)
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
            let rlp: Vec<f64> = top_lp[at].as_array().unwrap_or(&[]).iter().filter_map(|v| match v {
                Value::Number(n) => n.parse().ok(),
                _ => None,
            }).collect();
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
        eprintln!("{} prompt tokens in {prompt_ms:.0} ms, then {:.2} ms a token ({:.1} tok/s): top-1 agrees {a_p}/{n_p}, KL (top-8) {:.5}", ids.len(), step_ms / steps as f64, 1000.0 * steps as f64 / step_ms.max(1e-9), kl_p / n_p.max(1) as f64);
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
    let docs: Vec<Vec<u32>> = json::parse(&text).map_err(|e| Error(format!("{tokp}: {e}")))?.get("docs").and_then(Value::as_array).unwrap_or(&[]).iter().map(|d| d.as_u64s().unwrap_or_default().into_iter().map(|x| x as u32).collect()).collect();
    let o = OptsG4 { ctx: a.num("ctx", 1024usize)?, chunk: a.num("chunk", 512usize)?, layout: a.layout()?, ..OptsG4::default() };
    let t0 = std::time::Instant::now();
    let mut e = EngineG4::new(dir, &o)?;
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
    let meta = format!("{{\"vocab\": {}, \"rows_per_chunk\": {per}, \"model\": {:?}, \"chunks\": {}, \"engine\": \"glyd-engine g4score\", \"layout\": \"{:?}\"}}", e.w.cfg.vocab, dir.display().to_string(), docs.len(), e.w.layout);
    std::fs::write(format!("{out}.json"), meta).map_err(|e| Error(e.to_string()))?;
    Ok(())
}

/// `g4prefill MODEL [--tokens N] [--ctx N] [--chunk N] [--layout L]`: a prompt of N tokens (a fixed pattern) read once to warm up and once timed, with each
/// stage's share of the GPU time (the same marks as g4check --prof): where a long prompt's time goes.
#[cfg(feature = "dev")]
pub fn prefill_prof(a: &crate::args::Args) -> Result<()> {
    let dir = Path::new(a.need(1, "MODEL")?);
    let n = a.num("tokens", 16384usize)?;
    let o = OptsG4 { ctx: a.num("ctx", n + 64)?, chunk: a.num("chunk", 512usize)?, graph: false, layout: a.layout()?, ..OptsG4::default() };
    let mut e = EngineG4::new(dir, &o)?;
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

/// `g4fatest [--rows N] [--keys K] [--window W]`: the tensor-core attention (g4_fa256, g4_fa512) against g4_attn_t on random queries, keys and values (Gemma 4 31B's heads:
/// 32 query heads, 16 key-value heads at 256 and 4 at 512): the largest difference of the outputs and the rows where they differ by more than 0.05.
#[cfg(feature = "dev")]
pub fn fa_test(a: &crate::args::Args) -> Result<()> {
    let dev = Dev::new(0)?;
    let kg = KernG4::load(&dev)?;
    let (n, keys, window) = (a.num("rows", 257usize)?, a.num("keys", 0usize)?, a.num("window", 0i32)?);
    let scale = a.num("scale", 0.5f32)?;
    let s = dev.stream;
    let mut seed = 0x9e3779b97f4a7c15u64;
    let mut rnd = |m: usize| -> Vec<u16> {
        (0..m).map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            bf16_bits(((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale)
        }).collect()
    };
    for (d, nk) in [(256usize, 16usize), (512, 4)] {
        let nq = 32;
        let cap = (keys + n).max(n) + 16;
        let q = upload_bytes(crate::weights::as_bytes(&rnd(n * nq * d)))?;
        let kc = upload_bytes(crate::weights::as_bytes(&rnd(cap * nk * d)))?;
        let vc = upload_bytes(crate::weights::as_bytes(&rnd(cap * nk * d)))?;
        let pos: Vec<i32> = (0..n).map(|i| (keys + i) as i32).collect();
        let pb = Buf::new(4 * n)?;
        pb.write(0, &pos)?;
        let (o1, o2) = (Buf::zeroed(n * nq * d * 2)?, Buf::zeroed(n * nq * d * 2)?);
        let g = nq / nk;
        let r = (16 / g).max(1);
        let (attn, fa, bm) = if d == 256 { (&kg.attn256, &kg.fa256, 64usize) } else { (&kg.attn512, &kg.fa512, 32usize) };
        let (nqi, nki) = (nq as i32, nk as i32);
        // SAFETY: buffers sized for n rows and cap positions.
        unsafe {
            launch!(*attn, [((n + r - 1) / r) as u32, nk as u32, 1], [(32 * r * g) as u32, 1, 1], smem attn_smem(d, r * g), s; q.ptr, kc.ptr, vc.ptr, pb.ptr, o1.ptr, n as i32, nqi, nki, r as i32, window, cap as i32)?;
            if d == 512 && a.has("b") {
                launch!(kg.fa512b, [(n.div_ceil(32)) as u32, nq as u32, 1], [256, 1, 1], smem FA512B_SMEM, s; q.ptr, kc.ptr, vc.ptr, pb.ptr, o2.ptr, n as i32, nqi, nki, window, cap as i32)?;
            } else {
                launch!(*fa, [(n.div_ceil(bm)) as u32, nq as u32, (d / 256) as u32], [(bm / 16 * 32) as u32, 1, 1], smem fa_smem(d, 256, bm), s; q.ptr, kc.ptr, vc.ptr, pb.ptr, o2.ptr, n as i32, nqi, nki, window, cap as i32)?;
            }
        }
        dev.sync()?;
        let (mut x, mut y) = (vec![0u16; n * nq * d], vec![0u16; n * nq * d]);
        o1.read(0, &mut x)?;
        o2.read(0, &mut y)?;
        let f = |w: u16| f32::from_bits((w as u32) << 16);
        let mut worst = (0f32, 0usize);
        let mut bad_rows = std::collections::BTreeSet::new();
        for i in 0..x.len() {
            let dd = (f(x[i]) - f(y[i])).abs();
            if dd > worst.0 { worst = (dd, i); }
            if dd > 0.05 { bad_rows.insert(i / (nq * d)); }
        }
        let wi = worst.1;
        println!("head {d}: largest difference {:.4} (row {}, head {}, dim {}); rows differing by > 0.05: {} of {n} {:?}", worst.0, wi / (nq * d), (wi / d) % nq, wi % d, bad_rows.len(), bad_rows.iter().take(12).collect::<Vec<_>>());
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
        println!("{} ids ({} vs {}), decode {}: {prompt:?}", if same { "SAME" } else { "DIFFER" }, got.len(), want.len(), if dec == want_text { "same" } else { "DIFFERS" });
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
    let o = OptsG4 {
        ctx: a.num("ctx", 8192usize)?,
        chunk: a.num("chunk", 512usize)?,
        layout: a.layout()?,
        draft: Some((std::path::PathBuf::from(draft), k)),
        ..OptsG4::default()
    };
    let t0 = std::time::Instant::now();
    let mut e: Box<dyn Pass> = if a.has("dynamic") {
        let (c, dc) = (ConfigG4::load(dir)?, ConfigG4::load_for(Path::new(draft), true)?);
        let spec = crate::dynamic::DraftSpec { dir: Path::new(draft), desc: crate::dynamic::gemma4::describe_draft(&dc, &c), k };
        let d = crate::dynamic::Dyn::new(dir, crate::dynamic::gemma4::describe(&c), Some(spec), &crate::dynamic::Opts { ctx: o.ctx, chunk: o.chunk, graph: true, device: o.device, layout: o.layout })?;
        eprintln!("loaded in {:.1} s: {}", t0.elapsed().as_secs_f64(), d.describe());
        Box::new(d)
    } else {
        let g = EngineG4::new(dir, &o)?;
        let dbytes = g.drafter.as_ref().map_or(0, |d| d.m.bytes);
        eprintln!("loaded in {:.1} s: {}; draft model {:.3} GiB, {k} draft tokens a round", t0.elapsed().as_secs_f64(), g.describe(), dbytes as f64 / (1u64 << 30) as f64);
        Box::new(g)
    };
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
    let dev = Dev::new(0)?;
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
        let pb = upload_packed(crate::weights::Packed { data: pk.data, b1: pk.blocks, b2: pk.block_base.iter().flat_map(|x| x.to_le_bytes()).collect(), words: crate::ffi::Words::Tiered { blocks: 0, block_base: 0, tiers: pk.tiers }, rows, cols })?;
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

/// GLYD_G4_PF: "a,n,g" KiB (unset: 128,32,8, the best measured on an RTX 4080 SUPER; 0,0,0 none).
pub(crate) fn pf_kib(v: Option<&str>) -> Result<[usize; 3]> {
    let Some(v) = v.filter(|v| !v.is_empty()) else { return Ok([128, 32, 8]) };
    let k: Vec<Option<usize>> = v.split(',').map(|x| x.trim().parse().ok()).collect();
    match k[..] {
        [Some(a), Some(n), Some(g)] => Ok([a, n, g]),
        _ => err(format!("the prefetch setting {v}: three numbers of KiB, a,n,g")),
    }
}
