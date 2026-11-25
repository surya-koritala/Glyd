//! GLM-5.3-Flash's weights, which the executor's description of it loads by (src/dynamic/glm.rs) and Condor's engine (src/model_qwen4x.rs) shares (the kernel table of
//! kernels/glm.cu and kernels/mla.cu, `KernGlm`, is src/kern/glm.rs's): the layers' weights
//! (`WeightsGlm`: a layer's norms, mixing weights and Kimi delta or dense MLP matrices, packed as they load, and the blocks a caller makes of its mixture-of-experts layers),
//! and a latent attention layer's (`MlaWeights`). The config and tensor names are src/glm.rs's. (GLM's own development engine, which these came out of, is gone: R13 #234.)

use crate::dev::Buf;
use crate::ffi::{err, Error, Result, Words};
use crate::glm::{ConfigGlm, GlmKind};
use crate::kern35::Mat;
use crate::moe::MoeLayer;
use crate::qwen35::Pack35;
use crate::weights::{as_bytes, choose_layout, ordered, read_pack, upload_bytes, upload_packed, v7_manifest, Layout, PackBuf};
use glyd_gpu::json::Value;
use glyd_gpu::pack::{unpack_int4, unpack_int5, unpack_int6, Tiered, INT4_TAG, INT5_TAG, INT6_TAG};
use glyd_gpu::safetensors::Checkpoint;
use std::collections::HashMap;
use std::path::Path;

pub use crate::kern::glm::{mla_positions, mla_smem, Hc, KernGlm, GM_CHUNK_SMEM, GM_D, GM_HC, GM_L, GM_NP, HC_FEW};

/// A layer's buffers, by kind: its matrices (Kimi delta attention: qkv, o, f_a, f_b, g_a, g_b, b; then a dense MLP's gate and up, down; a latent attention layer's are `MlaWeights`, a mixture of experts' its
/// layer) and small tensors; a mixture-of-experts layer's experts' layer, where the loader was given a maker of it.
pub struct LayerGlm {
    pub kind: GlmKind,
    /// Its index among the layers of its kind (a Kimi delta layer's state, a latent attention layer's cache).
    pub idx: usize,
    /// The norms before the two blocks.
    pub norm_a: Buf,
    pub norm_f: Buf,
    /// The mixing weights of the token mixer's block and of the feed-forward's.
    pub hc: [Hc; 2],
    pub mats: Vec<Mat>,
    /// Kimi delta attention: the three convolutions' weights as one [3 heads 128, 4], A_log, dt_bias (fp32), the gated norm's weight.
    pub small: Vec<Buf>,
    pub moe: Option<Box<MoeLayer>>,
}

/// The model's weights in GPU memory. A layer not asked for (`only`) is None; so are the embedding, the LM head and the final norm then.
pub struct WeightsGlm {
    pub cfg: ConfigGlm,
    pub layout: Layout,
    pub layers: Vec<Option<LayerGlm>>,
    pub embed: Option<PackBuf>,
    pub head: Option<PackBuf>,
    pub norm_f: Option<Buf>,
    /// Device bytes of the weights.
    pub bytes: usize,
}

/// A tensor's bytes as fp32 (a checkpoint's fp32 parameters, which a glyd-v7 directory may hold as bf16: widened exactly).
pub(crate) fn f32_bytes(ck: &Checkpoint, name: &str) -> Result<Vec<u8>> {
    let (_, t) = ck.get(name).ok_or_else(|| Error(format!("no tensor {name}")))?;
    let bytes = ck.read(name).map_err(|e| Error(format!("{name}: {e}")))?;
    match t.dtype.as_str() {
        "F32" => Ok(bytes),
        "BF16" => Ok(bytes.as_chunks::<2>().0.iter().flat_map(|b| ((u16::from_le_bytes(*b) as u32) << 16).to_le_bytes()).collect()),
        other => err(format!("{name} is {other}, where F32 or BF16 is wanted")),
    }
}

/// The matrices `WeightsGlm::load` holds for layer `l`, in the order of `LayerGlm::mats`: a Kimi delta layer's seven (q, k and v as one, o, the forget gate's two, the output gate's two, the input gate's) and, where the
/// feed-forward is a dense MLP, its gate and up as one and its down. A latent attention layer's are its block's; a mixture-of-experts layer's (the shared expert's two and the experts) are its block's.
pub(crate) fn layer_matrices(cfg: &ConfigGlm, l: usize) -> Vec<Pack35> {
    let packs = cfg.layer_packs(l);
    let (attn, mlp) = packs.split_at(packs.len() - 2);
    attn.iter().filter(|_| cfg.kinds[l] == GlmKind::Kda).chain(mlp.iter().filter(|_| !cfg.moe[l])).cloned().collect()
}

/// Makes layer `l`'s experts' layer (src/moe_glm.rs `layers_cached`): the layer's number, the config, the checkpoint, the layout of the packs.
pub type Make<'a> = &'a dyn Fn(usize, &ConfigGlm, &Checkpoint, Layout) -> Result<Box<MoeLayer>>;

impl WeightsGlm {
    /// A Hub-format directory (config.json, the safetensors: FP8 with block scales or bf16) or a glyd-v7 directory (`glyd-engine pack MODEL OUT --layout i6|i5`): the layers of `only` (all, with the embedding, the head and the
    /// final norm, where None) checked against the weight map, their matrices packed here by `threads` workers in `layout` (None: the GPU's) and copied to the GPU as they are done. A glyd-v7 directory is at one level
    /// (`layout` None, or the same): the packs it holds saved are read as they are, the rest (a matrix the packer does not take) as the Hub's. A latent attention layer holds its norms and mixing weights (its own weights are
    /// `MlaWeights`); a mixture-of-experts layer the same and `moe`'s experts' layer, which gets the checkpoint and the layout and loads what it needs of the layer's tensors.
    /// The ends are asked for apart from the layers: `ends` (the embedding; the LM head and the final norm) for a stage of a model split across GPUs, None for a whole load's (with the whole model, or with no
    /// layer: the MTP head's).
    #[allow(clippy::too_many_arguments)]
    pub fn load_ends(
        dir: &Path,
        threads: usize,
        layout: Option<Layout>,
        cc: (i32, i32),
        total_mem: u64,
        only: Option<&[usize]>,
        ends: Option<(bool, bool)>,
        moe: Option<Make<'_>>,
    ) -> Result<WeightsGlm> {
        let cfg = ConfigGlm::load(dir)?;
        let ck = Checkpoint::open(dir).map_err(|e| Error(e.to_string()))?;
        let have: Vec<(String, String, Vec<u64>)> = ck
            .names()
            .map(|n| {
                let (_, t) = ck.get(n).expect("a tensor the checkpoint names");
                (n.to_string(), t.dtype.clone(), t.shape.clone())
            })
            .collect();
        let v7 = v7_manifest(dir)?;
        if let (Some((got, _)), Some(asked)) = (&v7, layout) {
            if *got != asked {
                return err(format!("{}: packed in the {got:?} layout, where {asked:?} is asked", dir.display()));
            }
        }
        let mut wanted: Vec<usize> = only.map_or_else(|| (0..cfg.layers).collect(), |l| l.to_vec());
        wanted.sort_unstable();
        wanted.dedup();
        let held = if v7.is_some() { cfg.saved(&ck) } else { Default::default() };
        match only {
            None => {
                cfg.check(&have, &held)?;
            }
            Some(l) => cfg.check_layers(&have, l, &held)?,
        }
        // the matrices: packed where the packer takes them, else as the checkpoint has them (as the bf16 an FP8 matrix dequantises to). The blocks load their own (a latent attention layer's, the experts and the shared
        // expert of a mixture-of-experts layer): here a layer's Kimi delta matrices and a dense MLP's.
        let mut items: Vec<Pack35> = Vec::new();
        let mut dense: Vec<(String, Vec<String>, usize, usize)> = Vec::new();
        for &l in &wanted {
            for p in layer_matrices(&cfg, l) {
                if p.packable() {
                    items.push(p);
                } else {
                    dense.push((p.key.clone(), p.parts.iter().map(|x| x.0.clone()).collect(), p.rows(), p.cols));
                }
            }
        }
        // (the embedding and the LM head: of a whole model, and of an engine with no layer of its own, which is the MTP head's alone; or a stage's ends)
        let (first, last) = ends.unwrap_or_else(|| {
            let all = only.is_none() || wanted.is_empty();
            (all, all)
        });
        items.extend(cfg.packs().into_iter().filter(|p| (p.key == "embed" && (first || (last && cfg.tie))) || (p.key == "lm_head" && last)));
        let matrix_bytes: u64 = items.iter().map(|p| (p.rows() * p.cols * 2) as u64).sum();
        let layout = match &v7 {
            Some((l, _)) => *l,
            None => choose_layout(layout, cc, total_mem, matrix_bytes, 0).0,
        };
        let mut packed: HashMap<String, PackBuf> = HashMap::new();
        // (a GPU of a group: its slice of the rows of each big matrix but the embedding, src/dynamic/tp.rs)
        let sl = crate::dynamic::tp::slicer();
        let cut: Vec<Option<(usize, usize)>> =
            items.iter().map(|p| sl.filter(|s| p.key != "embed" && s.slices(p.rows(), p.cols)).map(|s| s.rows(p.rows()))).collect();
        ordered(items.len(), threads, 2 * threads.max(1), &|k| crate::weights::read_pack_rows(&ck, v7.as_ref(), &items[k], layout, cut[k]), |k, p| {
            crate::status::STATUS.beat(); // (a sign of life for the program that started the engine)
            let mut b = upload_packed(p)?;
            if cut[k].is_some() {
                b.tp_rows = items[k].rows();
            }
            packed.insert(items[k].key.clone(), b);
            Ok(())
        })?;
        let read = |name: &str| -> Result<Buf> { upload_bytes(&ck.read(name).map_err(|e| Error(format!("{name}: {e}")))?) };
        let read_f32 = |name: &str| -> Result<Buf> { upload_bytes(&f32_bytes(&ck, name)?) };
        let schedule = cfg.schedule();
        let mut layers: Vec<Option<LayerGlm>> = (0..cfg.layers).map(|_| None).collect();
        let mut bytes = 0usize;
        for &l in &wanted {
            crate::status::STATUS.beat(); // (a mixture-of-experts layer's block takes seconds to read)
            let (kind, idx) = schedule[l];
            let mut mats = Vec::new();
            for p in layer_matrices(&cfg, l) {
                mats.push(match packed.remove(&p.key) {
                    Some(b) => Mat::Packed(b),
                    None => {
                        let (_, members, rows, cols) = dense.iter().find(|d| d.0 == p.key).expect("a matrix kept as it is");
                        let mut w = vec![0u16; rows * cols];
                        let mut at = 0;
                        for (m, r) in members.iter().zip(&p.parts) {
                            ck.read_bf16(m, &mut w[at..at + r.1 * cols]).map_err(|e| Error(format!("{m}: {e}")))?;
                            at += r.1 * cols;
                        }
                        Mat::Dense(upload_bytes(crate::weights::as_bytes(&w))?, *rows, *cols)
                    }
                });
            }
            // (layer_kept's order: the two norms, the mixing weights of the two blocks, then Kimi delta attention's three convolutions, A_log, dt_bias and gated norm)
            let kept = cfg.layer_kept(l);
            let (norm_a, norm_f) = (read(&kept[0].name)?, read(&kept[1].name)?);
            let hc = |i: usize| -> Result<Hc> {
                Ok(Hc { proj: read(&kept[2 + 3 * i].name)?, base: read_f32(&kept[3 + 3 * i].name)?, scale: read_f32(&kept[4 + 3 * i].name)? })
            };
            let hc = [hc(0)?, hc(1)?];
            let small = if kind == GlmKind::Kda {
                let mut conv = Vec::new();
                for t in &kept[8..11] {
                    conv.extend(ck.read(&t.name).map_err(|e| Error(format!("{}: {e}", t.name)))?);
                }
                vec![upload_bytes(&conv)?, read_f32(&kept[11].name)?, read_f32(&kept[12].name)?, read(&kept[13].name)?]
            } else {
                Vec::new()
            };
            let moe_block = match &moe {
                Some(make) if cfg.moe[l] => Some(make(l, &cfg, &ck, layout)?),
                _ => None,
            };
            bytes += norm_a.bytes
                + norm_f.bytes
                + hc.iter().map(|h| h.proj.bytes + h.base.bytes + h.scale.bytes).sum::<usize>()
                + mats.iter().map(Mat::bytes).sum::<usize>()
                + small.iter().map(|b| b.bytes).sum::<usize>();
            bytes += moe_block.as_ref().map_or(0, |b| b.weights_bytes());
            layers[l] = Some(LayerGlm { kind, idx, norm_a, norm_f, hc, mats, small, moe: moe_block });
        }
        let embed = if first || (last && cfg.tie) { Some(packed.remove("embed").ok_or_else(|| Error("the embedding is not packed".into()))?) } else { None };
        let head = if last && !cfg.tie { Some(packed.remove("lm_head").ok_or_else(|| Error("the LM head is not packed".into()))?) } else { None };
        let norm_f = if last { Some(read(&format!("{}norm.weight", cfg.prefix))?) } else { None };
        bytes += embed.as_ref().map_or(0, |b| b.bytes) + head.as_ref().map_or(0, |h| h.bytes) + norm_f.as_ref().map_or(0, |b| b.bytes);
        if let Some(k) = packed.keys().next() {
            return err(format!("a pack the forward pass does not take: {k}"));
        }
        Ok(WeightsGlm { cfg, layout, layers, embed, head, norm_f, bytes })
    }

    pub fn lm_head(&self) -> Option<&PackBuf> {
        self.head.as_ref().or(self.embed.as_ref())
    }
}

/// A latent attention layer's weights on the GPU: the library's products (q's two, the key-value latent's, o; packed as they load), kv_b_proj as the bf16 it is (the
/// absorbed form multiplies by its rows), the two norms' weights and the indexer's tensors as the checkpoint has them. The executor's latent attention (src/dynamic/glm.rs:
/// `Op::Mla`'s matrices and small tensors) loads them here.
pub(crate) struct MlaWeights {
    pub q_a: Mat,
    pub q_b: Mat,
    pub kv_a: Mat,
    pub o: Mat,
    pub q_a_ln: Buf,
    pub kv_a_ln: Buf,
    pub kv_b: Buf,
    pub wq_b: Buf,
    pub wk: Buf,
    pub k_norm_w: Buf,
    pub k_norm_b: Buf,
    pub wproj: Buf,
    pub ape: Buf,
    pub ix_gate: Buf,
}

/// A matrix of the layer as the engine multiplies by it: packed in `layout` (from the FP8 or bf16 checkpoint or a glyd-v7 directory's saved pack) where the library packs it, else as the bf16 it is.
fn load_mat(ck: &Checkpoint, v7: Option<&(Layout, Value)>, pk: &Pack35, layout: Layout) -> Result<Mat> {
    if pk.packable() {
        // (a GPU of a group: its slice of the rows where the matrix is a big one, src/dynamic/tp.rs)
        let cut = crate::dynamic::tp::slicer().filter(|s| s.slices(pk.rows(), pk.cols)).map(|s| s.rows(pk.rows()));
        let mut b = upload_packed(crate::weights::read_pack_rows(ck, v7, pk, layout, cut)?)?;
        if cut.is_some() {
            b.tp_rows = pk.rows();
        }
        return Ok(Mat::Packed(b));
    }
    let (rows, cols) = (pk.rows(), pk.cols);
    let mut w = vec![0u16; rows * cols];
    let mut at = 0;
    for (name, r) in &pk.parts {
        ck.read_bf16(name, &mut w[at..at + r * cols]).map_err(|e| Error(format!("{name}: {e}")))?;
        at += r * cols;
    }
    Ok(Mat::Dense(upload_bytes(as_bytes(&w))?, rows, cols))
}

/// A tensor of the checkpoint, bf16 as it is, on the GPU.
fn tensor(ck: &Checkpoint, name: &str, elements: usize) -> Result<Buf> {
    let (_, t) = ck.get(name).ok_or_else(|| Error(format!("no tensor {name}")))?;
    if t.dtype != "BF16" || t.shape.iter().product::<u64>() as usize != elements {
        return err(format!("{name} is {} {:?}, where {elements} BF16 values are wanted", t.dtype, t.shape));
    }
    upload_bytes(&ck.read(name).map_err(|e| Error(format!("{name}: {e}")))?)
}

impl MlaWeights {
    /// Layer `l`'s of the model `c` from the checkpoint `ck`, its library matrices packed in `layout`. `kv_b_proj` is the checkpoint's bf16 (from a glyd-v7 directory, which holds it packed, the pack decoded: the
    /// layer's numbers are then the pack's).
    pub(crate) fn load(l: usize, c: &ConfigGlm, ck: &Checkpoint, layout: Layout) -> Result<MlaWeights> {
        if !c.has_layer(l) || c.kind_of(l) != crate::glm::GlmKind::Mla {
            return err(format!("layer {l} is not a latent attention layer"));
        }
        let (m, ix) = (&c.mla, &c.indexer);
        let v7 = v7_manifest(&ck.dir)?;
        if let Some((got, _)) = &v7 {
            if *got != layout {
                return err(format!("{}: packed in the {got:?} layout, where {layout:?} is asked", ck.dir.display()));
            }
        }
        let packs = c.layer_packs(l);
        let find =
            |suffix: &str| -> Result<&Pack35> { packs.iter().find(|p| p.key.ends_with(suffix)).ok_or_else(|| Error(format!("layer {l}: no pack {suffix}"))) };
        let a = format!("{}layers.{l}.self_attn", c.prefix);
        let (q_a, q_b, kv_a, o) = (
            load_mat(ck, v7.as_ref(), find(".q_a_proj")?, layout)?,
            load_mat(ck, v7.as_ref(), find(".q_b_proj")?, layout)?,
            load_mat(ck, v7.as_ref(), find(".kv_a_proj_with_mqa")?, layout)?,
            load_mat(ck, v7.as_ref(), find(".o_proj")?, layout)?,
        );
        // kv_b_proj: the checkpoint's bf16; where the directory holds it as a pack (a glyd-v7 one), that pack decoded
        let kv_b_name = format!("{a}.kv_b_proj.weight");
        let kv_b = if ck.get(&kv_b_name).is_some() {
            let mut w = vec![0u16; m.heads * (m.nope + m.v) * m.kv_rank];
            ck.read_bf16(&kv_b_name, &mut w).map_err(|e| Error(format!("{kv_b_name}: {e}")))?;
            upload_bytes(as_bytes(&w))?
        } else {
            let pk = find(".kv_b_proj")?;
            upload_bytes(as_bytes(&unpacked(ck, v7.as_ref(), pk, layout)?))?
        };
        let (h, qr) = (c.hidden, m.q_rank);
        let ixn = format!("{a}.indexer");
        Ok(MlaWeights {
            q_a,
            q_b,
            kv_a,
            o,
            q_a_ln: tensor(ck, &format!("{a}.q_a_layernorm.weight"), qr)?,
            kv_a_ln: tensor(ck, &format!("{a}.kv_a_layernorm.weight"), m.kv_rank)?,
            kv_b,
            wq_b: tensor(ck, &format!("{ixn}.wq_b.weight"), ix.heads * ix.dim * qr)?,
            wk: tensor(ck, &format!("{ixn}.wk.weight"), ix.dim * h)?,
            k_norm_w: tensor(ck, &format!("{ixn}.k_norm.weight"), ix.dim)?,
            k_norm_b: tensor(ck, &format!("{ixn}.k_norm.bias"), ix.dim)?,
            wproj: tensor(ck, &format!("{ixn}.weights_proj.weight"), ix.heads * h)?,
            ape: tensor(ck, &format!("{ixn}.index_kpool_compress_ape"), ix.kpool * ix.dim)?,
            ix_gate: tensor(ck, &format!("{ixn}.index_kpool_compress_gate"), ix.dim * h)?,
        })
    }
}

/// A pack of `ck` in the `level` layout (a glyd-v7 directory's saved one where `v7` is its manifest and it holds the pack, else packed from the Hub-format tensors) decoded to bf16 rows, as the library decodes it on the GPU.
pub(crate) fn unpacked(ck: &Checkpoint, v7: Option<&(Layout, Value)>, pk: &Pack35, level: Layout) -> Result<Vec<u16>> {
    let p = read_pack(ck, v7, pk, level)?;
    let Words::Tiered { tiers, .. } = p.words else { return err(format!("{}: not in a block layout", pk.key)) };
    let t = Tiered { rows: p.rows, cols: p.cols, data: p.data, blocks: p.b1, block_base: Vec::new(), tiers };
    let mut w = vec![0u16; p.rows * p.cols];
    let ok = match tiers[0] >> 24 {
        INT6_TAG => unpack_int6(&t, &mut w),
        INT5_TAG => unpack_int5(&t, &mut w),
        INT4_TAG => unpack_int4(&t, &mut w),
        tag => return err(format!("{}: a layout of tag {tag}", pk.key)),
    };
    if ok {
        Ok(w)
    } else {
        err(format!("{}: its buffers do not decode", pk.key))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &str = include_str!("../kernels/glm.cu");
    const MLA_SRC: &str = include_str!("../kernels/mla.cu");

    /// The constants this file names are the kernels' (`#define`s of kernels/glm.cu), and the shared memory of the chunked rule is what its arrays take.
    #[test]
    fn the_kernels_constants_are_the_sources() {
        let define = |n: &str| -> Option<usize> {
            SRC.lines().find_map(|l| l.strip_prefix(&format!("#define {n} "))).and_then(|v| v.split_whitespace().next()).and_then(|v| v.parse().ok())
        };
        assert_eq!((define("GM_D"), define("GM_L")), (Some(GM_D), Some(GM_L)));
        let (bv, ks, ls) = (define("GM_BV").unwrap(), define("GM_KS").unwrap(), define("GM_LS").unwrap());
        assert_eq!((3 * GM_L * ks + GM_D * bv + GM_L * bv + 2 * GM_L * ls + GM_L) * 4, GM_CHUNK_SMEM as usize);
        // (it fits the 99 KB a block of an Ampere, Ada or Blackwell consumer card may ask for)
        const { assert!(GM_CHUNK_SMEM <= 101_376) };
        // the latent attention's (kernels/mla.cu, the layer kind's): the limits the launches hold to and the rows a block takes are the kernels' defines
        let define_mla = |n: &str| -> Option<usize> {
            MLA_SRC.lines().find_map(|l| l.strip_prefix(&format!("#define {n} "))).and_then(|v| v.split_whitespace().next()).and_then(|v| v.parse().ok())
        };
        assert_eq!((define_mla("GM_NOPE_MAX"), define_mla("GM_IDX_W"), define_mla("GM_RB"), define_mla("GM_VB")), (Some(256), Some(8), Some(8), Some(4)));
        for name in ["rms2_rows", "layernorm_rows", "scatter_rows", "mla_absorb", "idx_pool", "idx_score", "idx_pick", "mla_attn", "mla_expand_v"] {
            assert!(
                MLA_SRC.contains(&format!("extern \"C\" __global__ void {name}(")) || MLA_SRC.contains(&format!(") {name}(")),
                "no kernel {name} in kernels/mla.cu"
            );
        }
        for name in ["glm_kda_rec", "glm_kda_chunk", "glm_kda_out", "glm_kda_out_w32"] {
            assert!(
                SRC.contains(&format!("extern \"C\" __global__ void {name}(")) || SRC.contains(&format!(") {name}(")),
                "no kernel {name} in kernels/glm.cu"
            );
        }
    }

    /// The real model's numbers are ones the kernels run.
    #[test]
    fn the_real_config_is_one_the_engine_runs() {
        let c = crate::glm::tests::parse();
        assert!(
            c.kda.dim == GM_D
                && c.kda.conv_kernel == 4
                && c.hc.mult == GM_HC
                && c.hidden.is_multiple_of(8)
                && c.hidden <= 16 * 512
                && c.inter.is_multiple_of(8)
        );
        assert_eq!(c.hc_rows(), GM_NP);
        // the qkv product's rows are the convolutions' channels, and its state the delta rule's: 34 layers of 4 MiB
        assert_eq!((3 * c.qkv(), c.layers_of(GlmKind::Kda) * c.kda.heads * GM_D * GM_D * 4), (24576, 34 * 4 * 1024 * 1024));
    }

    /// The random tiny model `tools/glm_ref.py tiny` builds (its config is the fixture the script's selftest holds it to) is a model these kernels run: the real model's pattern in eight layers (three of Kimi delta attention with a
    /// dense MLP, a latent attention layer, three of Kimi delta attention, a latent attention layer; experts from the fourth), the numbers of its kernels, every matrix of its layers in whole blocks of the three block layouts.
    #[test]
    fn the_random_tiny_model_is_one_the_engine_takes() {
        use crate::glm::{IndexerDims, MlaDims};
        let c = ConfigGlm::parse(include_str!("../tests/fixtures/glm5_next_toy_config.json"), "toy", None).unwrap();
        let (k, m) = (GlmKind::Kda, GlmKind::Mla);
        assert_eq!(c.kinds, [k, k, k, m, k, k, k, m]);
        assert_eq!(c.moe, [false, false, false, true, true, true, true, true]);
        assert!(
            c.kda.dim == GM_D
                && c.kda.conv_kernel == 4
                && c.hc.mult == GM_HC
                && c.hidden.is_multiple_of(8)
                && c.hidden <= 16 * 512
                && c.inter.is_multiple_of(8)
                && c.kda.heads == 2
        );
        assert_eq!(
            (c.hidden, c.inter, c.experts.experts, c.experts.top_k, c.experts.inter, c.experts.shared_inter, c.vocab),
            (256, 384, 16, 4, 192, 192, 2048)
        );
        // latent attention: 2 heads of 64 with latents of 64 and a query latent of 64; an indexer of 4 heads of 32 channels that picks 2 pools of 4 tokens (so that a prompt of 12 tokens or more prunes)
        assert_eq!(c.mla, MlaDims { heads: 2, q_rank: 64, kv_rank: 64, nope: 64, v: 64 });
        assert_eq!(c.indexer, IndexerDims { heads: 4, dim: 32, topk: 8, kpool: 4, tail: true });
        crate::moe_glm::moe_cfg(&c).check().unwrap();
        for l in 0..c.layers {
            assert!(c.layer_packs(l).iter().filter(|p| !p.key.ends_with(".b_proj")).all(Pack35::packable), "layer {l}");
            // (the input gate's product has 2 rows, one for a head: the engine multiplies by it as a small dense matrix; the latent attention has none)
            assert_eq!(c.layer_packs(l).iter().find(|p| p.key.ends_with(".b_proj")).map(Pack35::packable), (c.kinds[l] == k).then_some(false), "layer {l}");
        }
        // (the shared expert's and the experts' matrices of a Kimi delta layer and of a latent attention layer)
        for (l, attn) in [(4, 7), (3, 5)] {
            for p in c.layer_packs(l).into_iter().skip(attn).chain(c.expert_packs(l)) {
                for tag in [glyd_gpu::pack::INT6_TAG, glyd_gpu::pack::INT5_TAG, glyd_gpu::pack::INT4_TAG] {
                    assert!(crate::moe::expert_strides(tag, p.rows(), p.cols).is_ok(), "{} is {} x {}", p.key, p.rows(), p.cols);
                }
            }
        }
        // the FP8 blocks divide the dense MLP's matrices (as the real model's divide its own), which transformers' dequantisation wants
        let [bm, bn] = c.fp8_block.unwrap();
        assert!(c.layer_packs(0).iter().skip(7).all(|p| p.parts.iter().all(|x| x.1 % bm == 0) && p.cols % bn == 0));
        // the kernels' limits: heads and latents the latent attention kernels take
        assert!(
            c.mla.nope <= 256
                && c.mla.kv_rank <= 512
                && c.mla.kv_rank.is_multiple_of(64)
                && c.indexer.dim <= 256
                && c.indexer.kpool <= 8
                && c.indexer.topk / c.indexer.kpool <= 1024
        );
    }

    /// What the loader holds of each layer: nine matrices for a Kimi delta layer with a dense MLP (the delta attention's seven, the MLP's two), seven where the feed-forward is a block of experts, none for a latent attention
    /// layer (its block's); and in the order the executor's description indexes them by (src/dynamic/glm.rs).
    #[test]
    fn the_matrices_a_layer_holds() {
        let c = crate::glm::tests::parse();
        let keys = |l: usize| -> Vec<String> {
            layer_matrices(&c, l).iter().map(|p| p.key.rsplit("layers.").next().unwrap().split_once('.').unwrap().1.to_string()).collect()
        };
        assert_eq!(
            keys(1),
            [
                "self_attn.q_proj",
                "self_attn.o_proj",
                "self_attn.f_a_proj",
                "self_attn.f_b_proj",
                "self_attn.g_a_proj",
                "self_attn.g_b_proj",
                "self_attn.b_proj",
                "mlp.gate_proj",
                "mlp.down_proj"
            ]
        );
        assert_eq!(keys(4), keys(1)[..7]);
        assert_eq!(keys(44), keys(1)[..7]);
        assert!(layer_matrices(&c, 3).is_empty() && layer_matrices(&c, 43).is_empty());
        assert_eq!((0..c.layers).map(|l| layer_matrices(&c, l).len()).sum::<usize>(), 3 * 9 + 31 * 7);
        // (rows and columns: the products' shapes, as the delta attention's products take them)
        let shapes: Vec<(usize, usize)> = layer_matrices(&c, 0).iter().map(|p| (p.rows(), p.cols)).collect();
        assert_eq!(shapes, [(24576, 4096), (4096, 8192), (128, 4096), (8192, 128), (128, 4096), (8192, 128), (64, 4096), (24576, 4096), (4096, 12288)]);
    }

    /// `layer_kept`'s order is the one the loader reads it in: the two norms, the mixing weights of the two blocks, then a Kimi delta layer's three convolutions, A_log, dt_bias and gated norm.
    #[test]
    fn the_kept_tensors_come_in_the_order_the_loader_reads() {
        let c = crate::glm::tests::parse();
        let names: Vec<String> = c.layer_kept(4).into_iter().map(|t| t.name.rsplit("layers.4.").next().unwrap().to_string()).collect();
        assert_eq!(
            names[..14],
            [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "hc_attn_fn",
                "hc_attn_base",
                "hc_attn_scale",
                "hc_ffn_fn",
                "hc_ffn_base",
                "hc_ffn_scale",
                "self_attn.q_conv1d.weight",
                "self_attn.k_conv1d.weight",
                "self_attn.v_conv1d.weight",
                "self_attn.A_log",
                "self_attn.dt_bias",
                "self_attn.o_norm.weight"
            ]
        );
        // a layer's packs: the attention's, then the MLP's two; a Kimi delta layer's attention has seven
        assert_eq!((c.layer_packs(4).len(), c.layer_packs(3).len()), (9, 7));
        assert!(c.layer_packs(4)[7].key.ends_with("shared_experts.gate_proj") && c.layer_packs(0)[7].key.ends_with("mlp.gate_proj"));
    }

    /// An fp32 parameter is read as fp32 from an fp32 tensor and, widened exactly, from a bf16 one (a glyd-v7 directory may hold it so); any other dtype is refused.
    #[test]
    fn an_fp32_parameter_is_read_from_fp32_or_bf16() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-f32-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let w = [0x3f80u16, 0xbf00, 0x3dcd];
        let bf16: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
        let widened: Vec<u8> = w.iter().flat_map(|&x| ((x as u32) << 16).to_le_bytes()).collect();
        crate::pack::write_safetensors(
            &dir.join("model.safetensors"),
            &[],
            &[("a", "F32", vec![3], &widened[..]), ("b", "BF16", vec![3], &bf16[..]), ("c", "F16", vec![3], &bf16[..])],
        )
        .unwrap();
        let ck = Checkpoint::open(&dir).unwrap();
        assert_eq!(f32_bytes(&ck, "a").unwrap(), widened);
        assert_eq!(f32_bytes(&ck, "b").unwrap(), widened);
        assert!(f32_bytes(&ck, "c").unwrap_err().msg.contains("F32 or BF16"));
        assert!(f32_bytes(&ck, "d").unwrap_err().msg.contains("no tensor d"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
