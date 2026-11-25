//! GLM-5.3-Flash's kernel table (kernels/glm.cu and kernels/mla.cu, `KernGlm`: src/kern/glm.rs's), which the executor runs (src/dynamic/glm.rs), and a reader of
//! a checkpoint's fp32 parameters. The executor loads GLM's weights by its generic loader (`Dyn::new_placed`, from
//! the lowered description: `lower::glm`) and its experts as the Moe kind's block (`dynamic::experts`). The config and tensor names are src/glm.rs's.

use crate::ffi::{err, Error, Result};
use glyd_gpu::safetensors::Checkpoint;

pub use crate::kern::glm::{mla_positions, mla_smem, Hc, KernGlm, GM_CHUNK_SMEM, GM_D, GM_HC, GM_L, GM_NP};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glm::{ConfigGlm, GlmKind};
    use crate::qwen35::Pack35;

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
        // (its experts' core layer as the Moe kind's block makes it)
        let kinds = crate::runtime::describe::describe(&serde_json::from_str(include_str!("../tests/fixtures/glm5_next_toy_config.json")).unwrap()).unwrap();
        for (_, m, _) in crate::dynamic::experts::of_layers(&kinds, (0, c.layers), 2) {
            crate::dynamic::experts::cfg(m, c.hidden, c.experts.inter).unwrap().check().unwrap();
        }
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

    /// `layer_kept`'s order (the random model writes them so, `ConfigGlm::check` holds a checkpoint to them): the two norms, the mixing weights of the two blocks, then a Kimi
    /// delta layer's three convolutions, A_log, dt_bias and gated norm.
    #[test]
    fn the_kept_tensors_come_in_the_config_readers_order() {
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
