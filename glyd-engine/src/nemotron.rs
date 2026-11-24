//! Nemotron 3 Super (`NemotronHForCausalLM`, 2026) as the engine reads it: its config (a hybrid of Mamba-2 layers, attention layers without any positional encoding, and latent
//! mixture-of-experts layers, in the order of `hybrid_override_pattern`), the tensors it loads and how they are packed, and what a sequence holds. The reference: transformers 5.19.0's
//! modeling_nemotron_h.py and vLLM's nemotron_h.py (research/nemotron3-super-offload.md has the numbers); the layers' math is tools/nemotron_ref.py, the layers on the GPU src/kern_nh.rs.

use crate::ffi::{err, Error, Result};
use crate::qwen35::{Pack35, Tensor35};
use crate::weights::{eos_ids, num, saved_parts, Layout};
use glyd_gpu::json::{self, Value};
use glyd_gpu::safetensors::Checkpoint;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// A layer's mixer: Mamba-2 (`M`), attention (`*`) or the latent mixture of experts (`E`). Every layer is `x + mixer(norm(x))`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NhKind {
    Mamba,
    Attention,
    Moe,
}

/// The mixture-of-experts layers' numbers: the router sees the full hidden vector (sigmoid scores plus a bias for the choice of `top_k` of `experts`, the weights of the chosen scores normed and times
/// `scale`), the experts work on a `latent`-wide vector (up `inter` x `latent`, relu2, down `latent` x `inter`; no gate), and a shared expert (`shared_inter` wide, relu2, on the full vector) is added.
#[derive(Clone, Debug)]
pub struct MoeDims {
    pub experts: usize,
    pub top_k: usize,
    pub inter: usize,
    pub shared_inter: usize,
    pub latent: usize,
    pub scale: f32,
    pub norm_topk: bool,
}

impl MoeDims {
    /// The width the experts read: the latent, or the model's (`hidden`) where there is none.
    pub fn width(&self, hidden: usize) -> usize {
        if self.latent > 0 {
            self.latent
        } else {
            hidden
        }
    }
}

/// What the engine reads of a NemotronH config.json.
#[derive(Clone, Debug)]
pub struct ConfigNemotronH {
    pub hidden: usize,
    pub layers: usize,
    pub vocab: usize,
    /// `layer_norm_epsilon`: the layers' norms, the final norm and the Mamba-2 gated norm.
    pub eps: f32,
    pub max_pos: usize,
    pub tie: bool,
    pub eos: Vec<i64>,
    /// Each layer's mixer, in order.
    pub kinds: Vec<NhKind>,
    /// Mamba-2: heads of `m_head_dim` channels (the inner width is their product), a state of `m_state` a channel, `m_groups` groups of B and C (shared by the heads of a group), the convolution's kernel,
    /// the chunk of the reference's scan.
    pub m_heads: usize,
    pub m_head_dim: usize,
    pub m_state: usize,
    pub m_groups: usize,
    pub conv_kernel: usize,
    pub chunk: usize,
    /// Attention: query heads, key-value heads, the head's width; full attention, scale head_dim^-0.5, no positional encoding.
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub moe: MoeDims,
    /// The layers of the MTP head (`mtp.*`), which this engine does not load.
    pub mtp_layers: usize,
    /// What the tensor names of the layers, the embedding and the final norm begin with.
    pub prefix: &'static str,
}

/// The tensors of a checkpoint this engine does not load: the MTP head's.
pub fn skipped(name: &str) -> bool {
    name.starts_with("mtp.")
}

/// The layout the matrix `key` (a `Pack35`'s: `...mixer.experts.E.up_proj`, `...mixer.in_proj`, `embed`) is held in at the level `level`: the level's, but at the four-bit level (wren) only the routed experts (112.7
/// of the 120.6 billion matrix weights) are four-bit codes, and every other matrix (Mamba-2's, attention's, the latent projections', the shared experts', the embedding and the head) stays in six-bit ones. A routed
/// expert's down matrix is held in the row layout of its bits (92% of its inputs are zero after relu squared: src/moe_rows.rs reads its rows for the ones that are not); in a development build
/// `GLYD_ENGINE_DENSE_DOWN` keeps it in the block layout, for the dense product to compare with. A packer saves each pack in this layout (src/pack.rs) and the loader packs a bf16 checkpoint's matrices in it; a glyd-v7
/// directory is read in the layout each entry names (`weights::read_v7`): one with a block layout's down matrices runs the dense product. At the Hummingbird level the routed experts are the level's own (the codebook rows
/// of the calibration, c3r or c2r by expert: src/moe_hb.rs reads them from the directory, nothing packs them from bf16) and every other matrix is six-bit, as at wren.
pub fn pack_layout(level: Layout, key: &str) -> Layout {
    let own = if matches!(level, Layout::Int4 | Layout::Hummingbird) && !key.contains(".mixer.experts.") { Layout::Int6 } else { level };
    if key.contains(".mixer.experts.") && key.ends_with(".down_proj") && crate::tune!("GLYD_ENGINE_DENSE_DOWN").is_none() {
        own.row_of().unwrap_or(own)
    } else {
        own
    }
}

/// Whether a config.json is a NemotronH model's (what the dispatcher asks first).
pub fn is_nemotron_h(text: &str) -> bool {
    crate::runtime::config::family_of(text) == Some(crate::support::Family::NemotronH)
}

impl ConfigNemotronH {
    pub fn load(dir: &Path) -> Result<ConfigNemotronH> {
        let p = dir.join("config.json");
        let text = std::fs::read_to_string(&p).map_err(|e| Error(format!("{}: {e}", p.display())))?;
        let generation = std::fs::read_to_string(dir.join("generation_config.json")).ok();
        ConfigNemotronH::parse(&text, &p.display().to_string(), generation.as_deref())
    }

    /// A config.json's text (and a generation_config.json's, where there is one): `label` names the file in an error.
    pub fn parse(text: &str, label: &str, generation: Option<&str>) -> Result<ConfigNemotronH> {
        let v = json::parse(text).map_err(|e| Error(format!("{label}: {e}")))?;
        let arch = v.get("architectures").and_then(Value::as_array).and_then(|a| a.first()).and_then(Value::as_str).unwrap_or("");
        if arch != "NemotronHForCausalLM" {
            return err(format!("{label}: {arch}: not a NemotronH model"));
        }
        if let Some(kind) = v.get("model_type").and_then(Value::as_str).filter(|k| *k != "nemotron_h") {
            return err(format!("{label}: the model is {kind:?}, where a NemotronH has nemotron_h"));
        }
        let get = |k: &str| v.get(k).ok_or_else(|| Error(format!("{label}: no {k}")));
        let u = |k: &str| -> Result<usize> { get(k)?.as_u64().map(|x| x as usize).ok_or_else(|| Error(format!("{label}: {k}: not an integer"))) };
        let refuse = |what: &str| err(format!("{label}: {what}, which the engine does not run"));
        let text_of = |k: &str, want: &str| v.get(k).and_then(Value::as_str).is_some_and(|a| a != want);
        if text_of("mamba_hidden_act", "silu") {
            return refuse("a Mamba-2 activation other than silu");
        }
        if text_of("mlp_hidden_act", "relu2") {
            return refuse("an expert activation other than relu2");
        }
        for k in ["use_bias", "mlp_bias", "attention_bias", "mamba_proj_bias"] {
            if v.get(k).and_then(Value::as_bool) == Some(true) {
                return refuse(&format!("biases ({k})"));
            }
        }
        if v.get("use_conv_bias").and_then(Value::as_bool) == Some(false) {
            return refuse("a convolution without its bias");
        }
        if v.get("sliding_window").is_some_and(|w| *w != Value::Null) {
            return refuse("a sliding window");
        }
        if text_of("mamba_ssm_cache_dtype", "float32") {
            return refuse("a Mamba-2 state other than float32");
        }
        // the layer pattern: M Mamba-2, * attention, E latent mixture of experts, from the string or the list of types (transformers' layers_block_type)
        let kinds: Vec<NhKind> = if let Some(p) = v.get("hybrid_override_pattern").and_then(Value::as_str) {
            p.chars()
                .map(|c| match c {
                    'M' => Ok(NhKind::Mamba),
                    '*' => Ok(NhKind::Attention),
                    'E' => Ok(NhKind::Moe),
                    other => err(format!("{label}: a layer {other:?} in hybrid_override_pattern, where this model has M, * and E")),
                })
                .collect::<Result<_>>()?
        } else if let Some(a) = v.get("layers_block_type").or_else(|| v.get("layer_types")).and_then(Value::as_array) {
            a.iter()
                .map(|k| match k.as_str() {
                    // (Nemotron 3.5's configs name them mamba and attention)
                    Some("linear_attention" | "mamba") => Ok(NhKind::Mamba),
                    Some("full_attention" | "attention") => Ok(NhKind::Attention),
                    Some("moe") => Ok(NhKind::Moe),
                    other => {
                        err(format!("{label}: a layer of type {other:?}, where this model has mamba (linear_attention), attention (full_attention) and moe"))
                    }
                })
                .collect::<Result<_>>()?
        } else {
            return err(format!("{label}: no hybrid_override_pattern"));
        };
        let layers = kinds.len();
        if layers == 0 || v.get("num_hidden_layers").and_then(Value::as_u64).is_some_and(|n| n as usize != layers) {
            return err(format!("{label}: {layers} layers in the pattern, num_hidden_layers {:?}", v.get("num_hidden_layers").and_then(Value::as_u64)));
        }
        let (m_heads, m_head_dim, m_groups) = (u("mamba_num_heads")?, u("mamba_head_dim")?, u("n_groups")?);
        if m_heads == 0 || m_groups == 0 || !m_heads.is_multiple_of(m_groups) {
            return err(format!("{label}: {m_heads} Mamba-2 heads to {m_groups} groups"));
        }
        let (heads, kv_heads) = (u("num_attention_heads")?, u("num_key_value_heads")?);
        if heads == 0 || kv_heads == 0 || !heads.is_multiple_of(kv_heads) {
            return err(format!("{label}: {heads} query heads to {kv_heads} KV heads"));
        }
        let head_dim = v.get("head_dim").and_then(Value::as_u64).map(|x| x as usize).unwrap_or(u("hidden_size")? / heads);
        let moe = MoeDims {
            experts: u("n_routed_experts")?,
            top_k: u("num_experts_per_tok")?,
            inter: u("moe_intermediate_size")?,
            shared_inter: u("moe_shared_expert_intermediate_size")?,
            latent: match v.get("moe_latent_size") {
                Some(l) if *l != Value::Null => l.as_u64().map(|x| x as usize).ok_or_else(|| Error(format!("{label}: moe_latent_size: not an integer")))?,
                _ => 0,
            },
            scale: v.get("routed_scaling_factor").map_or(Ok(1.0), |s| num(s, "routed_scaling_factor"))? as f32,
            norm_topk: v.get("norm_topk_prob").and_then(Value::as_bool).unwrap_or(true),
        };
        if kinds.contains(&NhKind::Moe) {
            if moe.top_k == 0 || moe.top_k > moe.experts {
                return err(format!("{label}: {} experts a token of {}", moe.top_k, moe.experts));
            }
            if v.get("n_group").and_then(Value::as_u64).is_some_and(|g| g != 1) || v.get("topk_group").and_then(Value::as_u64).is_some_and(|g| g != 1) {
                return refuse("group-limited routing");
            }
            if v.get("n_shared_experts").and_then(Value::as_u64).is_some_and(|g| g != 1) {
                return refuse("other than one shared expert");
            }
        }
        let eos = eos_ids(&v, generation);
        let mut eps = 1e-5;
        if let Some(e) = v.get("layer_norm_epsilon").or_else(|| v.get("norm_eps")) {
            eps = num(e, "layer_norm_epsilon")? as f32;
        }
        Ok(ConfigNemotronH {
            hidden: u("hidden_size")?,
            layers,
            vocab: u("vocab_size")?,
            eps,
            max_pos: u("max_position_embeddings")?,
            tie: v.get("tie_word_embeddings").and_then(Value::as_bool).unwrap_or(false),
            eos,
            kinds,
            m_heads,
            m_head_dim,
            m_state: u("ssm_state_size")?,
            m_groups,
            conv_kernel: u("conv_kernel")?,
            chunk: v.get("chunk_size").and_then(Value::as_u64).map_or(128, |c| c as usize),
            heads,
            kv_heads,
            head_dim,
            moe,
            mtp_layers: v.get("num_nextn_predict_layers").and_then(Value::as_u64).unwrap_or(0) as usize,
            prefix: "backbone.",
        })
    }

    /// Each layer's kind and its index among the layers of its kind (a Mamba-2 layer's state, an attention layer's cache, a mixture-of-experts layer's place).
    pub fn schedule(&self) -> Vec<(NhKind, usize)> {
        let mut count = [0usize; 3];
        self.kinds
            .iter()
            .map(|&k| {
                let n = &mut count[k as usize];
                *n += 1;
                (k, *n - 1)
            })
            .collect()
    }

    pub fn layers_of(&self, kind: NhKind) -> usize {
        self.kinds.iter().filter(|&&k| k == kind).count()
    }

    /// The Mamba-2 layers' inner width: all their heads' channels (the width of z and of the scan's output).
    pub fn inner(&self) -> usize {
        self.m_heads * self.m_head_dim
    }

    /// The channels of the causal convolution: the scan's inputs x (the inner width), B and C.
    pub fn conv_dim(&self) -> usize {
        self.inner() + 2 * self.m_groups * self.m_state
    }

    /// The rows of `in_proj`: z, then the convolution's channels, then dt (a value a head). ponytail: dt's 128 rows are packed with the rest of the matrix, which a lossy level rounds; dt is
    /// what softplus and exp amplify, so a level may want them kept bf16 (a row range of the tensor read apart, or a saver that splits it) once a level's quality asks.
    pub fn in_rows(&self) -> usize {
        self.inner() + self.conv_dim() + self.m_heads
    }

    fn layer_name(&self, l: usize) -> String {
        format!("{}layers.{l}", self.prefix)
    }

    /// One layer's matrices besides a mixture-of-experts layer's experts, packable or not, in load order: Mamba-2 in_proj, out_proj; attention q, k and v as one, then o; a mixture-of-experts layer's
    /// latent projections and shared expert (up, down).
    pub fn layer_packs(&self, l: usize) -> Vec<Pack35> {
        let (h, p) = (self.hidden, format!("{}.mixer", self.layer_name(l)));
        let mut out = Vec::new();
        let mut pack = |module: &str, parts: &[(&str, usize)], cols: usize| {
            out.push(Pack35 {
                parts: parts.iter().map(|&(n, r)| (format!("{module}.{n}.weight"), r)).collect(),
                key: format!("{module}.{}", parts[0].0),
                cols,
            });
        };
        match self.kinds[l] {
            NhKind::Mamba => {
                pack(&p, &[("in_proj", self.in_rows())], h);
                pack(&p, &[("out_proj", h)], self.inner());
            }
            NhKind::Attention => {
                pack(&p, &[("q_proj", self.heads * self.head_dim), ("k_proj", self.kv_heads * self.head_dim), ("v_proj", self.kv_heads * self.head_dim)], h);
                pack(&p, &[("o_proj", h)], self.heads * self.head_dim);
            }
            NhKind::Moe => {
                let m = &self.moe;
                // (no latent, Nemotron 3.5 Lightning: the experts read the model's width, without the two projections)
                if m.latent > 0 {
                    pack(&p, &[("fc1_latent_proj", m.latent)], h);
                    pack(&p, &[("fc2_latent_proj", h)], m.latent);
                }
                pack(&format!("{p}.shared_experts"), &[("up_proj", m.shared_inter)], h);
                pack(&format!("{p}.shared_experts"), &[("down_proj", h)], m.shared_inter);
            }
        }
        out
    }

    /// A mixture-of-experts layer's experts, each of its two matrices a pack of its own in the checkpoint's names (`...experts.E.up_proj`, `...experts.E.down_proj`), up before down, expert by expert.
    pub fn expert_packs(&self, l: usize) -> Vec<Pack35> {
        if self.kinds[l] != NhKind::Moe {
            return Vec::new();
        }
        let (m, p, w) = (&self.moe, format!("{}.mixer.experts", self.layer_name(l)), self.moe.width(self.hidden));
        (0..m.experts)
            .flat_map(|e| {
                [
                    Pack35 { key: format!("{p}.{e}.up_proj"), parts: vec![(format!("{p}.{e}.up_proj.weight"), m.inter)], cols: w },
                    Pack35 { key: format!("{p}.{e}.down_proj"), parts: vec![(format!("{p}.{e}.down_proj.weight"), w)], cols: m.inter },
                ]
            })
            .collect()
    }

    /// The packed matrices in load order (the packable ones of the layers' matrices and experts, then the embedding and the LM head where the model has its own): packed at a level.
    pub fn packs(&self) -> Vec<Pack35> {
        self.matrices().into_iter().filter(Pack35::packable).collect()
    }

    fn matrices(&self) -> Vec<Pack35> {
        let mut out: Vec<Pack35> = (0..self.layers).flat_map(|l| self.layer_packs(l).into_iter().chain(self.expert_packs(l))).collect();
        out.push(Pack35 { key: "embed".into(), parts: vec![(format!("{}embeddings.weight", self.prefix), self.vocab)], cols: self.hidden });
        if !self.tie {
            out.push(Pack35 { key: "lm_head".into(), parts: vec![("lm_head.weight".into(), self.vocab)], cols: self.hidden });
        }
        out
    }

    /// One layer's tensors that are loaded as the checkpoint has them: the norm; Mamba-2's convolution (weights and bias), `A_log`, `D`, `dt_bias` and gated norm (all bf16 in this checkpoint, and
    /// read as bf16: the kernels widen them); a mixture-of-experts layer's router (kept bf16 and used in fp32) and its score-correction bias (fp32).
    pub fn layer_kept(&self, l: usize) -> Vec<Tensor35> {
        let (h, p) = (self.hidden as u64, self.layer_name(l));
        let t = |name: String, dtype: &'static str, shape: &[u64]| Tensor35 { name, dtype, shape: shape.to_vec() };
        let mut out = vec![t(format!("{p}.norm.weight"), "BF16", &[h])];
        match self.kinds[l] {
            NhKind::Mamba => {
                let nh = self.m_heads as u64;
                out.push(t(format!("{p}.mixer.conv1d.weight"), "BF16", &[self.conv_dim() as u64, 1, self.conv_kernel as u64]));
                out.push(t(format!("{p}.mixer.conv1d.bias"), "BF16", &[self.conv_dim() as u64]));
                out.push(t(format!("{p}.mixer.A_log"), "BF16", &[nh]));
                out.push(t(format!("{p}.mixer.D"), "BF16", &[nh]));
                out.push(t(format!("{p}.mixer.dt_bias"), "BF16", &[nh]));
                out.push(t(format!("{p}.mixer.norm.weight"), "BF16", &[self.inner() as u64]));
            }
            NhKind::Attention => {}
            NhKind::Moe => {
                out.push(t(format!("{p}.mixer.gate.weight"), "BF16", &[self.moe.experts as u64, h]));
                out.push(t(format!("{p}.mixer.gate.e_score_correction_bias"), "F32", &[self.moe.experts as u64]));
            }
        }
        out
    }

    /// Every tensor loaded as the checkpoint has it, in load order: each layer's (`layer_kept`), the parts of a matrix the library cannot pack, and the final norm.
    pub fn kept(&self) -> Vec<Tensor35> {
        let t = |name: String, dtype: &'static str, shape: &[u64]| Tensor35 { name, dtype, shape: shape.to_vec() };
        let unpacked: Vec<Pack35> = self.matrices().into_iter().filter(|p| !p.packable()).collect();
        let mut out = Vec::new();
        for l in 0..self.layers {
            out.extend(self.layer_kept(l));
            let p = format!("{}.", self.layer_name(l));
            for pk in unpacked.iter().filter(|pk| pk.key.starts_with(&p)) {
                out.extend(pk.parts.iter().map(|(n, r)| t(n.clone(), "BF16", &[*r as u64, pk.cols as u64])));
            }
        }
        out.push(t(format!("{}norm_f.weight", self.prefix), "BF16", &[self.hidden as u64]));
        out
    }

    /// A checkpoint's tensors (name, dtype, shape) against `want`: every one present with its dtype (an fp32 one may be held as bf16) and shape. The first problems, in words.
    fn missing(want: &[Tensor35], have: &[(String, String, Vec<u64>)]) -> Vec<String> {
        let by: HashMap<&str, (&str, &[u64])> = have.iter().map(|(n, d, s)| (n.as_str(), (d.as_str(), s.as_slice()))).collect();
        let mut bad = Vec::new();
        for w in want {
            match by.get(w.name.as_str()) {
                None => bad.push(format!("no {}", w.name)),
                Some((d, s)) if !(*d == w.dtype || (w.dtype == "F32" && *d == "BF16")) || *s != w.shape.as_slice() => {
                    bad.push(format!("{}: {d} {s:?}, where {} {:?} is wanted", w.name, w.dtype, w.shape))
                }
                _ => {}
            }
        }
        bad
    }

    fn report(bad: Vec<String>) -> Result<()> {
        if bad.is_empty() {
            return Ok(());
        }
        let more = if bad.len() > 8 { format!(" (and {} more)", bad.len() - 8) } else { String::new() };
        err(format!("{}{more}", bad[..bad.len().min(8)].join("; ")))
    }

    fn wanted_matrices(packs: Vec<Pack35>, packed: &HashSet<String>) -> Vec<Tensor35> {
        packs
            .into_iter()
            .flat_map(|p| p.parts.into_iter().map(move |(n, r)| Tensor35 { name: n, dtype: "BF16", shape: vec![r as u64, p.cols as u64] }))
            .filter(|t| !packed.contains(Pack35::saved(&t.name)))
            .collect()
    }

    /// The matrices `ck` holds as saved packs (a glyd-v7 directory's), by their parts' names (`Pack35::saved`): what `check` and `check_layers` take as held in their three buffers.
    pub fn saved(&self, ck: &Checkpoint) -> HashSet<String> {
        self.packs().iter().filter(|p| !saved_parts(ck, p).is_empty()).flat_map(|p| p.parts.iter().map(|x| Pack35::saved(&x.0).to_string())).collect()
    }

    /// A checkpoint's tensors against what the engine loads: every one present with its dtype and shape, and nothing else but the class it skips. Returns the number skipped. The tensors in
    /// `packed` (by `Pack35::saved`) are held as their three buffers, not as bf16 tensors.
    pub fn check(&self, have: &[(String, String, Vec<u64>)], packed: &HashSet<String>) -> Result<usize> {
        let mut want = Self::wanted_matrices(self.packs(), packed);
        want.extend(self.kept());
        Self::report(Self::missing(&want, have))?;
        let wanted: HashSet<&str> = want.iter().map(|w| w.name.as_str()).collect();
        let buffers: HashSet<String> = packed.iter().flat_map(|k| ["data", "blocks", "block_base", "codebook"].map(|s| format!("{k}.glyd_{s}"))).collect();
        // (a Hummingbird directory also holds the signs of each mixture-of-experts layer's rotation: `glyd_rot_signs`, saved beside the codebook rows)
        let signs: HashSet<String> =
            (0..self.layers).filter(|&l| self.kinds[l] == NhKind::Moe).map(|l| format!("{}.mixer.glyd_rot_signs", self.layer_name(l))).collect();
        let (mut left_out, mut bad) = (0, Vec::new());
        for (n, _, _) in have {
            if wanted.contains(n.as_str()) || buffers.contains(n) || signs.contains(n) {
                continue;
            }
            if skipped(n) {
                left_out += 1;
            } else {
                bad.push(format!("{n}: not a tensor of this model"));
            }
        }
        Self::report(bad).map(|()| left_out)
    }

    /// Just these layers' tensors (their matrices and small tensors, not a mixture-of-experts layer's experts) against a checkpoint that may hold nothing else (a few layers read from the Hub by
    /// range): every one present with its dtype and shape, the matrices in `packed` (as for `check`) as their saved buffers.
    pub fn check_layers(&self, have: &[(String, String, Vec<u64>)], layers: &[usize], packed: &HashSet<String>) -> Result<()> {
        if let Some(&l) = layers.iter().find(|&&l| l >= self.layers) {
            return err(format!("layer {l}, where the model has {}", self.layers));
        }
        let mut want = Self::wanted_matrices(layers.iter().flat_map(|&l| self.layer_packs(l)).collect(), packed);
        want.extend(layers.iter().flat_map(|&l| self.layer_kept(l)));
        Self::report(Self::missing(&want, have))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn text() -> &'static str {
        include_str!("../tests/fixtures/nemotron_h_config.json")
    }

    pub(crate) fn parse() -> ConfigNemotronH {
        ConfigNemotronH::parse(text(), "nemotron", None).unwrap()
    }

    /// The config with `edit` applied to its top-level keys, as text.
    fn edited(edit: impl Fn(&mut Vec<(String, Value)>)) -> String {
        let mut v = json::parse(text()).unwrap();
        if let Value::Object(top) = &mut v {
            edit(top);
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
        let e = ConfigNemotronH::parse(text, "x", None).unwrap_err().msg;
        assert!(e.contains(says), "{e:?} does not say {says:?}");
    }

    /// The checkpoint's tensors as the fixture holds them (made by `tools/nemotron_ref.py inventory` from the shards' headers of the pinned revision): every name, the experts of a layer as a run.
    pub(crate) type Index = Vec<(String, String, Vec<u64>)>;

    pub(crate) fn tensors() -> Index {
        let v = json::parse(include_str!("../tests/fixtures/nemotron_h_tensors.json")).unwrap();
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

    /// n bf16 words of a spread like a trained matrix's, from a seed.
    pub(crate) fn weights(n: usize, seed: u64) -> Vec<u16> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let v = ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2;
                ((v.to_bits().wrapping_add(0x7fff + ((v.to_bits() >> 16) & 1))) >> 16) as u16
            })
            .collect()
    }

    pub(crate) fn bf16_bytes(w: &[u16]) -> Vec<u8> {
        w.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// The model `c` (the config `text`) as a Hub-format directory: every tensor the engine reads (`packs` and `kept`), bf16 of a seed apiece but the score-correction biases, fp32 as the checkpoint has them, and a
    /// tensor of the MTP head, which it does not. Returns the weights by name.
    pub(crate) fn checkpoint(dir: &Path, text: &str, c: &ConfigNemotronH) -> HashMap<String, Vec<u16>> {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), text).unwrap();
        let mut named: Vec<(String, &str, Vec<u64>)> = Vec::new();
        named.extend(c.packs().iter().flat_map(|p| p.parts.iter().map(|x| (x.0.clone(), "BF16", vec![x.1 as u64, p.cols as u64]))));
        named.extend(c.kept().into_iter().map(|t| (t.name, t.dtype, t.shape)));
        named.push(("mtp.layers.0.norm.weight".into(), "BF16", vec![c.hidden as u64]));
        let mut by: HashMap<String, Vec<u16>> = HashMap::new();
        let data: Vec<(String, &str, Vec<u64>, Vec<u8>)> = named
            .into_iter()
            .enumerate()
            .map(|(i, (name, dtype, shape))| {
                let n = shape.iter().product::<u64>() as usize;
                let w = weights(n, 100 + i as u64);
                // (an fp32 tensor: the bf16 words widened, so that the same values are the test's weights)
                let bytes = if dtype == "F32" { w.iter().flat_map(|&x| ((x as u32) << 16).to_le_bytes()).collect() } else { bf16_bytes(&w) };
                by.insert(name.clone(), w);
                (name, dtype, shape, bytes)
            })
            .collect();
        let tensors: Vec<(&str, &str, Vec<u64>, &[u8])> = data.iter().map(|(n, d, s, b)| (n.as_str(), *d, s.clone(), &b[..])).collect();
        crate::pack::write_safetensors(&dir.join("model.safetensors"), &[("format", "pt".to_string())], &tensors).unwrap();
        by
    }

    #[test]
    fn the_super_config() {
        let c = parse();
        assert_eq!((c.hidden, c.layers, c.vocab, c.eps, c.max_pos, c.tie, c.eos.clone()), (4096, 88, 131072, 1e-5, 262144, false, vec![2]));
        assert_eq!((c.m_heads, c.m_head_dim, c.m_state, c.m_groups, c.conv_kernel, c.chunk), (128, 64, 128, 8, 4, 128));
        assert_eq!((c.heads, c.kv_heads, c.head_dim), (32, 2, 128));
        assert_eq!(
            (c.moe.experts, c.moe.top_k, c.moe.inter, c.moe.shared_inter, c.moe.latent, c.moe.scale, c.moe.norm_topk),
            (512, 22, 2688, 5376, 1024, 5.0, true)
        );
        assert_eq!((c.mtp_layers, c.prefix), (1, "backbone."));
        assert_eq!((c.inner(), c.conv_dim(), c.in_rows()), (8192, 10240, 18560));
        // generation_config.json's end tokens win over the config's
        let c = ConfigNemotronH::parse(text(), "x", Some(r#"{"eos_token_id": [2, 11]}"#)).unwrap();
        assert_eq!(c.eos, vec![2, 11]);
        // the layer types as the list transformers names them, for the same model
        let kinds = parse().kinds;
        let list: Vec<String> = kinds
            .iter()
            .map(|k| {
                format!(
                    "\"{}\"",
                    ["linear_attention", "full_attention", "moe"][match k {
                        NhKind::Mamba => 0,
                        NhKind::Attention => 1,
                        NhKind::Moe => 2,
                    }]
                )
            })
            .collect();
        let t = edited(|t| {
            t.retain(|(k, _)| k != "hybrid_override_pattern");
            set(t, "layers_block_type", &format!("[{}]", list.join(",")));
        });
        assert_eq!(ConfigNemotronH::parse(&t, "x", None).unwrap().kinds, kinds);
        assert!(is_nemotron_h(text()) && !is_nemotron_h(r#"{"architectures": ["Qwen3ForCausalLM"]}"#) && !is_nemotron_h("{}"));
    }

    /// 88 layers: 40 Mamba-2, 40 latent mixtures of experts, 8 attention at 7, 16, 25, 36, 47, 58, 69 and 78.
    #[test]
    fn the_schedule() {
        let c = parse();
        let s = c.schedule();
        assert_eq!(s.len(), 88);
        assert_eq!((c.layers_of(NhKind::Mamba), c.layers_of(NhKind::Moe), c.layers_of(NhKind::Attention)), (40, 40, 8));
        let attn: Vec<usize> = (0..88).filter(|&l| s[l].0 == NhKind::Attention).collect();
        assert_eq!(attn, [7, 16, 25, 36, 47, 58, 69, 78]);
        assert_eq!(
            (s[0], s[1], s[2], s[7], s[8], s[16], s[87]),
            ((NhKind::Mamba, 0), (NhKind::Moe, 0), (NhKind::Mamba, 1), (NhKind::Attention, 0), (NhKind::Moe, 3), (NhKind::Attention, 1), (NhKind::Moe, 39))
        );
        // each kind's indices run 0, 1, 2, ... in order
        for kind in [NhKind::Mamba, NhKind::Attention, NhKind::Moe] {
            let idx: Vec<usize> = s.iter().filter(|x| x.0 == kind).map(|x| x.1).collect();
            assert_eq!(idx, (0..idx.len()).collect::<Vec<_>>());
        }
    }

    /// The research note's table: 160 MiB of fp32 state and 2.34 MiB of convolution state in all, 8 KiB of KV a token.
    #[test]
    fn what_a_sequence_holds() {
        let c = parse();
        assert_eq!(c.state_bytes(), 40 * (128 * 64 * 128 * 4 + 10240 * 3 * 2));
        assert_eq!(c.state_bytes(), 160 * (1 << 20) + 40 * 61_440);
        assert_eq!(c.kv_bytes_per_token(), 8 << 10);
    }

    #[test]
    fn what_is_refused() {
        refused(r#"{"architectures": ["Qwen3ForCausalLM"]}"#, "not a NemotronH model");
        refused(&edited(|t| set(t, "model_type", r#""nemotron_h_nano""#)), "nemotron_h");
        refused(&edited(|t| set(t, "hybrid_override_pattern", r#""ME-*""#)), "a layer '-'");
        refused(&edited(|t| set(t, "num_hidden_layers", "87")), "88 layers in the pattern");
        refused(&edited(|t| set(t, "mamba_hidden_act", r#""gelu""#)), "silu");
        refused(&edited(|t| set(t, "mlp_hidden_act", r#""silu""#)), "relu2");
        refused(&edited(|t| set(t, "use_bias", "true")), "biases (use_bias)");
        refused(&edited(|t| set(t, "attention_bias", "true")), "biases (attention_bias)");
        refused(&edited(|t| set(t, "use_conv_bias", "false")), "without its bias");
        refused(&edited(|t| set(t, "sliding_window", "4096")), "sliding window");
        refused(&edited(|t| set(t, "mamba_ssm_cache_dtype", r#""bfloat16""#)), "float32");
        refused(&edited(|t| set(t, "n_groups", "7")), "128 Mamba-2 heads to 7 groups");
        refused(&edited(|t| set(t, "num_key_value_heads", "3")), "32 query heads to 3 KV heads");
        // (no latent, as Nemotron 3.5 Lightning: the experts read the model's width; a layer's dense matrices are the shared expert's two)
        let flat = ConfigNemotronH::parse(&edited(|t| set(t, "moe_latent_size", "null")), "x", None).unwrap();
        let l = flat.kinds.iter().position(|&k| k == NhKind::Moe).unwrap();
        assert_eq!((flat.moe.latent, flat.moe.width(flat.hidden), flat.layer_packs(l).len()), (0, flat.hidden, 2));
        assert_eq!((flat.expert_packs(l)[0].cols, flat.expert_packs(l)[1].parts[0].1), (flat.hidden, flat.hidden));
        refused(&edited(|t| set(t, "n_group", "8")), "group-limited routing");
        refused(&edited(|t| set(t, "num_experts_per_tok", "513")), "513 experts a token of 512");
        refused(&edited(|t| t.retain(|(k, _)| k != "hybrid_override_pattern")), "no hybrid_override_pattern");
        refused(&edited(|t| t.retain(|(k, _)| k != "hidden_size")), "no hidden_size");
        // what is not refused: the rope fields the modeling code never reads, and a pattern of attention and Mamba-2 alone (no experts, so no router settings)
        assert!(ConfigNemotronH::parse(
            &edited(|t| {
                set(t, "rope_theta", "5");
                set(t, "hybrid_override_pattern", r#""M*M""#);
                set(t, "moe_latent_size", "null");
                set(t, "num_hidden_layers", "3")
            }),
            "x",
            None
        )
        .is_ok());
    }

    /// The weight map against the pinned checkpoint's 42,683 tensors: every tensor the engine wants is there with its dtype and shape, and the only ones it does not take are the MTP head's.
    #[test]
    fn the_weight_map_against_the_checkpoint() {
        let c = parse();
        let have = tensors();
        assert_eq!(have.len(), 42_683);
        assert_eq!(c.check(&have, &HashSet::new()).unwrap(), 1040);
        // the dtype of the small tensors is the checkpoint's own: A_log, D and dt_bias are bf16 there, the score-correction bias fp32
        let dtype = |n: &str| have.iter().find(|t| t.0 == n).unwrap().1.clone();
        assert_eq!(
            (dtype("backbone.layers.0.mixer.A_log"), dtype("backbone.layers.1.mixer.gate.e_score_correction_bias")),
            ("BF16".to_string(), "F32".to_string())
        );
    }

    #[test]
    fn the_packs() {
        let c = parse();
        let packs = c.packs();
        // 40 Mamba-2 layers of 2 packs, 8 attention layers of 2 (q, k, v as one; o), 40 mixtures of experts of 4 and 1,024 experts, the embedding and the LM head
        assert_eq!(packs.len(), 40 * 2 + 8 * 2 + 40 * (4 + 1024) + 2);
        assert!(packs.iter().all(Pack35::packable));
        let shape = |key: &str| packs.iter().find(|p| p.key == key).map(|p| (p.rows(), p.cols)).unwrap();
        let l = |n: usize, rest: &str| format!("backbone.layers.{n}.{rest}");
        assert_eq!(shape(&l(0, "mixer.in_proj")), (18560, 4096));
        assert_eq!(shape(&l(0, "mixer.out_proj")), (4096, 8192));
        assert_eq!(shape(&l(7, "mixer.q_proj")), (4096 + 256 + 256, 4096));
        assert_eq!(shape(&l(7, "mixer.o_proj")), (4096, 4096));
        assert_eq!(shape(&l(1, "mixer.fc1_latent_proj")), (1024, 4096));
        assert_eq!(shape(&l(1, "mixer.fc2_latent_proj")), (4096, 1024));
        assert_eq!(shape(&l(1, "mixer.shared_experts.up_proj")), (5376, 4096));
        assert_eq!(shape(&l(1, "mixer.shared_experts.down_proj")), (4096, 5376));
        assert_eq!(shape(&l(1, "mixer.experts.511.up_proj")), (2688, 1024));
        assert_eq!(shape(&l(1, "mixer.experts.0.down_proj")), (1024, 2688));
        assert_eq!((shape("embed"), shape("lm_head")), ((131072, 4096), (131072, 4096)));
        let q = packs.iter().find(|p| p.key == l(7, "mixer.q_proj")).unwrap();
        assert_eq!(q.parts.iter().map(|(n, r)| (n.rsplit('.').nth(1).unwrap(), *r)).collect::<Vec<_>>(), [("q_proj", 4096), ("k_proj", 256), ("v_proj", 256)]);
        // the packed parameters are the matrices (120.58 B), the kept ones the small tensors and the routers (86.7 M); with the MTP head's 2.94 B they are the checkpoint's 123,611,033,088
        let packed: u64 = packs.iter().map(|p| (p.rows() * p.cols) as u64).sum();
        assert_eq!(packed, 120_582_045_696);
        assert_eq!(c.kept().iter().map(|t| t.shape.iter().product::<u64>()).sum::<u64>(), 86_662_144);
        assert_eq!(packed + 86_662_144 + 2_942_325_248, 123_611_033_088);
        // kept: the norms and Mamba-2 small tensors, the routers (bf16, used in fp32) and their biases (fp32), the final norm
        let kept = c.kept();
        assert_eq!(kept.len(), 88 + 40 * 6 + 40 * 2 + 1);
        let k = |name: &str| kept.iter().find(|t| t.name == name).map(|t| (t.dtype, t.shape.clone())).unwrap();
        assert_eq!(k(&l(0, "mixer.conv1d.weight")), ("BF16", vec![10240, 1, 4]));
        assert_eq!(k(&l(0, "mixer.conv1d.bias")), ("BF16", vec![10240]));
        assert_eq!(k(&l(0, "mixer.A_log")), ("BF16", vec![128]));
        assert_eq!(k(&l(0, "mixer.norm.weight")), ("BF16", vec![8192]));
        assert_eq!(k(&l(1, "mixer.gate.weight")), ("BF16", vec![512, 4096]));
        assert_eq!(k(&l(1, "mixer.gate.e_score_correction_bias")), ("F32", vec![512]));
        assert_eq!(k("backbone.norm_f.weight"), ("BF16", vec![4096]));
        // nothing is in both lists
        let names: HashSet<&str> = packs.iter().flat_map(|p| p.parts.iter().map(|x| x.0.as_str())).chain(kept.iter().map(|t| t.name.as_str())).collect();
        assert_eq!(names.len(), packs.iter().map(|p| p.parts.len()).sum::<usize>() + kept.len());
        // a matrix of rows the packer cannot take is kept as the checkpoint has it
        let odd = ConfigNemotronH { moe: MoeDims { shared_inter: 5400, ..c.moe.clone() }, ..c.clone() };
        assert!(odd.packs().iter().all(|p| !p.key.ends_with("shared_experts.up_proj")));
        assert!(odd.kept().iter().any(|t| t.name == l(1, "mixer.shared_experts.up_proj.weight") && t.shape == [5400, 4096]));
    }

    /// The layout of each pack at each level, on the pinned checkpoint's config: at kestrel and swift every pack is the level's, but for the routed experts' down matrices (20,480 packs), which are in the level's row
    /// layout; at wren the routed experts (40,960 packs, 112.7 billion weights) alone are four-bit codes (their down matrices in the four-bit row layout), and the other 258 packs (Mamba-2's, attention's, the latent
    /// projections', the shared experts', the embedding and the head: 7.84 billion weights) are six-bit ones. The model comes to about 70 GB, as the plan sizes it (a row layout's bytes are its block layout's).
    #[test]
    fn the_experts_alone_are_four_bit_codes_at_wren() {
        let c = parse();
        let experts: HashSet<String> = (0..c.layers).flat_map(|l| c.expert_packs(l)).map(|p| p.key).collect();
        assert_eq!(experts.len(), 40 * 512 * 2);
        let down = |key: &str| key.ends_with(".down_proj") && experts.contains(key);
        for level in [Layout::Int6, Layout::Int5] {
            assert!(c.packs().iter().all(|p| pack_layout(level, &p.key) == if down(&p.key) { level.row_of().unwrap() } else { level }), "{level:?}");
        }
        let (mut weights, mut packs) = (HashMap::new(), HashMap::new());
        for p in c.packs() {
            let layout = pack_layout(Layout::Int4, &p.key);
            assert_eq!(
                layout,
                if down(&p.key) {
                    Layout::Int4Row
                } else if experts.contains(&p.key) {
                    Layout::Int4
                } else {
                    Layout::Int6
                },
                "{}",
                p.key
            );
            // (the weights and packs of each bit width: a row layout's are its block layout's)
            let bits = layout.name().trim_end_matches('r').to_string();
            *weights.entry(bits.clone()).or_insert(0u64) += (p.rows() * p.cols) as u64;
            *packs.entry(bits).or_insert(0usize) += 1;
        }
        assert_eq!((packs["i4"], packs["i6"]), (40_960, 80 + 16 + 160 + 2));
        assert_eq!((weights["i4"], weights["i6"]), (112_742_891_520, 7_839_154_176));
        // (bits a weight as the planner has them (plan.rs `sizing::bits`): the experts 4.53, the rest 6.53, and the small tensors as they are)
        let gb = (weights["i4"] as f64 * 4.53 + weights["i6"] as f64 * 6.53) / 8.0 / 1e9;
        assert!((70.0..70.5).contains(&(gb + 86_662_144.0 * 2.0 / 1e9)), "{gb} GB of packs");
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
        let at = |h: &Index, n: &str| h.iter().position(|t| t.0 == n).unwrap();
        let conv = "backbone.layers.0.mixer.conv1d.weight";
        says(
            &|h| {
                let i = at(h, conv);
                h.remove(i);
            },
            &format!("no {conv}"),
        );
        says(
            &|h| {
                let i = at(h, conv);
                h[i].2 = vec![10240, 1, 3];
            },
            "[10240, 1, 3], where BF16 [10240, 1, 4]",
        );
        says(
            &|h| {
                let i = at(h, "backbone.layers.1.mixer.gate.e_score_correction_bias");
                h[i].1 = "F16".into();
            },
            "where F32 [512]",
        );
        says(
            &|h| {
                let i = at(h, "backbone.layers.1.mixer.experts.300.up_proj.weight");
                h.remove(i);
            },
            "no backbone.layers.1.mixer.experts.300.up_proj.weight",
        );
        says(&|h| h.push(("model.embed_tokens.weight".into(), "BF16".into(), vec![131072, 4096])), "model.embed_tokens.weight: not a tensor of this model");
        // a config with more layers than the checkpoint
        let more = ConfigNemotronH { layers: 89, kinds: [c.kinds.clone(), vec![NhKind::Mamba]].concat(), ..c.clone() };
        assert!(more.check(&have, &HashSet::new()).unwrap_err().msg.contains("no backbone.layers.88.mixer.in_proj.weight"));
        // a packed matrix is held as its three buffers
        let key = "backbone.layers.0.mixer.in_proj";
        let mut h = have.clone();
        h.retain(|t| t.0 != format!("{key}.weight"));
        assert!(c.check(&h, &HashSet::new()).is_err());
        h.extend(["data", "blocks", "block_base"].map(|s| (format!("{key}.glyd_{s}"), "U8".to_string(), vec![1u64])));
        assert_eq!(c.check(&h, &HashSet::from([key.to_string()])).unwrap(), 1040);
        assert!(c.check(&h, &HashSet::new()).is_err());
    }

    /// The toy model tools/nemotron_ref.py builds for a pod check (a few real layers renumbered 0, 1, ..., a part of the embedding and of the head) is a checkpoint of the config it writes: this one's
    /// pattern `M*` and a vocabulary of 2,048, with layer 0 and layer 7 of the real model as its two layers.
    #[test]
    fn a_toy_model_of_real_layers() {
        let real = tensors();
        let mut have: Index = Vec::new();
        for (src, dst) in [(0, 0), (7, 1)] {
            let from = format!("backbone.layers.{src}.");
            for t in real.iter().filter(|t| t.0.starts_with(&from)) {
                have.push((t.0.replacen(&from, &format!("backbone.layers.{dst}."), 1), t.1.clone(), t.2.clone()));
            }
        }
        have.push(("backbone.embeddings.weight".into(), "BF16".into(), vec![2048, 4096]));
        have.push(("lm_head.weight".into(), "BF16".into(), vec![2048, 4096]));
        have.push(("backbone.norm_f.weight".into(), "BF16".into(), vec![4096]));
        let toy = edited(|t| {
            set(t, "hybrid_override_pattern", r#""M*""#);
            set(t, "num_hidden_layers", "2");
            set(t, "vocab_size", "2048");
        });
        let c = ConfigNemotronH::parse(&toy, "toy", None).unwrap();
        assert_eq!((c.layers, c.vocab, c.kinds.clone()), (2, 2048, vec![NhKind::Mamba, NhKind::Attention]));
        assert_eq!(c.check(&have, &HashSet::new()).unwrap(), 0);
    }

    /// A few layers read from the Hub by range hold only those layers' tensors: the check of just them.
    #[test]
    fn some_layers_alone() {
        let c = parse();
        let mut have = tensors();
        have.retain(|t| ["backbone.layers.0.", "backbone.layers.7."].iter().any(|p| t.0.starts_with(p)));
        assert_eq!(have.len(), 9 + 5);
        let none = HashSet::new();
        c.check_layers(&have, &[0, 7], &none).unwrap();
        assert!(c.check_layers(&have, &[0, 1], &none).unwrap_err().msg.contains("no backbone.layers.1.norm.weight"));
        assert!(c.check_layers(&have, &[88], &none).unwrap_err().msg.contains("layer 88"));
        // a layer's matrix held as its saved buffers (a glyd-v7 directory's) is not wanted as a bf16 tensor
        let held: HashSet<String> =
            ["backbone.layers.7.mixer.q_proj", "backbone.layers.7.mixer.k_proj", "backbone.layers.7.mixer.v_proj"].map(String::from).into();
        have.retain(|t| t.0 != "backbone.layers.7.mixer.v_proj.weight");
        assert!(c.check_layers(&have, &[0, 7], &none).unwrap_err().msg.contains("no backbone.layers.7.mixer.v_proj.weight"));
        have.retain(|t| !held.contains(Pack35::saved(&t.0)));
        assert!(c.check_layers(&have, &[0, 7], &none).unwrap_err().msg.contains("no backbone.layers.7.mixer.q_proj.weight"));
        c.check_layers(&have, &[0, 7], &held).unwrap();
    }
}
