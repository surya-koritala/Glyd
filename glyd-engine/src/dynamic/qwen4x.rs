//! Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`, Condor's model) on the executor (R3d, #140): its config's layer kinds lowered to the shared operations
//! (`lower::qwen4x`) and loaded here: the matrices packed at the level asked (penguin's lossless packs where none is), the ones the library does not pack and the table of n-grams
//! held as saved (bf16), the small tensors as saved but A_log in fp32 (the delta rule reads it so), each layer's experts on the GPU at kestrel's six bits (Condor's
//! block, `QxMoe`, handed over as a `ResidentMoe`). Condor's engine (src/model_qwen4x.rs) keeps serving the release: its experts in the computer's memory and their
//! cache, its draft head and the table of n-grams read from the disk are not on the executor yet, so a model whose experts or table do not fit this GPU is refused.

use super::{Dyn, Loaded, ModelDesc, Op, Opts, ResidentMoe, Stage};
use crate::dev::{Buf, Dev};
use crate::ffi::{err, Error, Result};
use crate::kern35::Mat;
use crate::kern_nh::MoeBlock;
use crate::weights::{pack_item_map, upload_bytes, upload_packed, Layout};
use glyd_gpu::safetensors::Checkpoint;
use std::path::Path;

/// The executor description's family name.
pub const FAMILY: &str = "Qwen3.8-Flash-Next";

/// A matrix whose rows or columns are not a multiple of this is held as saved and multiplied by the small product (the library packs those that are): the inject
/// weights' 4 rows, the b and a gates' 2 value heads.
const PACKED_BY: usize = 64;

/// The executor's description of the model in `dir` for sequences of up to `ctx` positions: its kinds lowered (`lower::qwen4x`), held to Condor's own reading of
/// the config (`ConfigQx::agrees`).
pub fn lowered(dir: &Path, ctx: usize) -> Result<ModelDesc> {
    let (kinds, label) = crate::runtime::describe::describe_dir(dir)?;
    let c = crate::qwen4x::ConfigQx::agrees(crate::qwen4x::ConfigQx::load(dir)?, &kinds).map_err(|e| e.context(&label))?;
    super::lower::qwen4x(&kinds, c.eos.clone(), ctx.min(c.max_pos)).map_err(|e| e.context(&label))
}

/// The model in `dir` on the executor, its always-on matrices at `o.layout` (lossless where none), its experts at kestrel's six bits.
pub fn load(dir: &Path, o: &Opts) -> Result<Dyn> {
    let c = crate::qwen4x::ConfigQx::load(dir)?;
    let ctx = o.ctx + o.chunk + 16;
    let d = lowered(dir, ctx)?;
    let dev = Dev::new(o.device)?;
    let ck = Checkpoint::open(dir).map_err(|e| Error(e.to_string()))?;
    let layout = o.layout.unwrap_or(Layout::Tiered);
    // (what the GPU would hold: the experts at six bits and the table of n-grams as saved; Condor's engine serves a model past that)
    let experts = (c.layers * c.experts * 3 * c.inter * c.hidden) as u64 * 3 / 4;
    let table: u64 = held_as_saved(&d).iter().map(|&k| (d.mats[k].rows * d.mats[k].cols * 2) as u64).sum();
    let free = dev.mem_info().map(|(f, _)| f as u64).unwrap_or(0);
    if experts + table > free {
        return err(format!(
            "{FAMILY}: its experts ({:.1} GB at six bits) and its table of n-grams ({:.1} GB) do not fit this GPU's {:.1} GB free: Condor's engine serves it (condor/README.md)",
            experts as f64 / 1e9,
            table as f64 / 1e9,
            free as f64 / 1e9
        ));
    }
    let saved = held_as_saved(&d);
    let mats: Vec<Mat> = (0..d.mats.len())
        .map(|k| {
            let it = &d.mats[k];
            if saved.contains(&k) || it.rows % PACKED_BY != 0 || it.cols % PACKED_BY != 0 {
                let w = crate::weights::item_words(&ck, it)?;
                return Ok(Mat::Dense(upload_bytes(crate::weights::as_bytes(&w))?, it.rows, it.cols));
            }
            Ok(Mat::Packed(upload_packed(pack_item_map(&ck, it, layout, &|_| {})?)?))
        })
        .collect::<Result<_>>()?;
    let fp32 = a_logs(&d);
    let small: Vec<Buf> = d
        .small
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let (_, t) = ck.get(name).ok_or_else(|| Error(format!("no tensor {name}")))?;
            let b = ck.read(name).map_err(|e| Error(format!("{name}: {e}")))?;
            match (t.dtype.as_str(), fp32.contains(&i)) {
                ("BF16", true) => {
                    let v: Vec<f32> = b.chunks_exact(2).map(|x| f32::from_bits((u16::from_le_bytes([x[0], x[1]]) as u32) << 16)).collect();
                    upload_bytes(crate::weights::as_bytes(&v))
                }
                ("F32", true) | ("BF16", false) | ("I64", false) => upload_bytes(&b),
                (dt, _) => {
                    err(format!("{name}: {dt}, where {} is read", if fp32.contains(&i) { "bf16 or fp32" } else { "bf16, or int64 for the hash's numbers" }))
                }
            }
        })
        .collect::<Result<_>>()?;
    let moe: Vec<Box<dyn MoeBlock>> = (0..c.layers)
        .map(|l| Ok(Box::new(ResidentMoe(crate::model_qwen4x::QxMoe::load(l, &c, &ck, Layout::Int6)?.into_layer())) as Box<dyn MoeBlock>))
        .collect::<Result<_>>()?;
    let ropes = d
        .ropes
        .iter()
        .map(|r| {
            let (cs, sn) = crate::gemma4::rotary(r.theta, r.dim, r.pairs, d.max_pos);
            Ok((upload_bytes(crate::weights::as_bytes(&cs))?, upload_bytes(crate::weights::as_bytes(&sn))?))
        })
        .collect::<Result<_>>()?;
    let st = Stage::whole(d.layers.len());
    Dyn::from_parts(dev, d, Loaded { mats, small, moe, ..Loaded::default() }, ropes, layout, st, o)
}

/// The matrices held as saved: the tables of n-grams (`Op::NgramRows` reads their rows).
fn held_as_saved(d: &ModelDesc) -> Vec<usize> {
    d.layers.iter().flatten().filter_map(|op| if let Op::NgramRows(x) = op { Some(x.table) } else { None }).collect()
}

/// The small tensors the delta rule reads in fp32 (`Op::GdnDelta`'s A_log).
fn a_logs(d: &ModelDesc) -> Vec<usize> {
    d.layers.iter().flatten().filter_map(|op| if let Op::GdnDelta { a_log, .. } = op { Some(*a_log) } else { None }).collect()
}

#[cfg(test)]
mod tests {
    use super::super::{HeadMix, IdxKeys, Op, Slot, StreamsMix};
    use crate::runtime::describe::describe;

    fn tiny() -> serde_json::Value {
        let m: serde_json::Value = serde_json::from_str(include_str!("../../tests/fixtures/qwen4x_tiny_manifest.json")).unwrap();
        serde_json::from_str(m["config"].as_str().unwrap()).unwrap()
    }

    /// The tiny model of the `qwen4x` logits case, lowered: eight layers of gated residuals around each block (a gated mix before it, the gated update after),
    /// the n-gram embedding's three operations before the second layer's, Gated DeltaNet with the sigmoid gate, the sparse attention's index product, its
    /// queries and pools, its unweighted pick of 2 pools of 4 and the tail, the attention over the picks with its keys and values in states, the experts, and
    /// the final mixer on the last layer; every matrix and small tensor named is the checkpoint's, and every slot an operation reads is sized.
    #[test]
    fn condors_tiny_model_lowers_to_the_shared_operations() {
        let d = describe(&tiny()).unwrap();
        let m = super::super::lower::qwen4x(&d, vec![2000], 1100).unwrap();
        assert!(matches!(m.first, Op::Streams));
        assert_eq!((m.layers.len(), m.hidden, m.rings.len(), m.ropes.len()), (8, 256, 0, 1));
        let kinds: Vec<Vec<&'static str>> = m
            .layers
            .iter()
            .map(|ops| {
                ops.iter()
                    .map(|op| match op {
                        Op::StreamsPre { mix: StreamsMix::Gated { .. }, .. } => "pre",
                        Op::StreamsPost { gated: true, .. } => "post",
                        Op::StreamsHead { mix: HeadMix::Gated { .. }, .. } => "head",
                        Op::NgramRows(_) => "ngram rows",
                        Op::NgramAdd(_) => "ngram add",
                        Op::Linear { .. } => "linear",
                        Op::GdnConv { .. } => "conv",
                        Op::GdnDelta { .. } => "delta",
                        Op::GatedNorm { sigmoid: true, .. } => "sigmoid norm",
                        Op::IdxMean { .. } => "index",
                        Op::IdxPick(x) if !x.weighted && matches!(x.keys, IdxKeys::Pools(_)) && (x.ratio, x.top_k, x.heads) == (4, 2, 4) => "pick",
                        Op::Attn(a) if a.picked.is_some() && (a.nq, a.nk, a.d) == (24, 2, 256) => "picked attention",
                        Op::Moe(_) => "experts",
                        o => panic!("{o:?}"),
                    })
                    .collect()
            })
            .collect();
        let gdn = ["pre", "linear", "linear", "conv", "delta", "sigmoid norm", "linear", "post", "pre", "experts", "post"];
        let attn = ["pre", "linear", "linear", "index", "pick", "picked attention", "linear", "post", "pre", "experts", "post"];
        assert_eq!(kinds[0], gdn);
        assert_eq!(kinds[1][..3], ["ngram rows", "linear", "ngram add"]);
        assert_eq!(kinds[1][3..], gdn);
        assert_eq!(kinds[3], attn);
        assert_eq!(kinds[7][..attn.len()], attn);
        assert_eq!(kinds[7][attn.len()..], ["head"]);
        // (each matrix's members are the checkpoint's tensors, its rows theirs stacked)
        let names = d.names();
        assert!(m.mats.iter().all(|it| it.members.iter().all(|n| names.contains(n))) && m.small.iter().all(|n| names.contains(n)));
        let rows: Vec<(usize, usize)> = m.mats.iter().map(|it| (it.rows, it.cols)).collect();
        // (layer 0: the gated residual's three, qkvz, ba, out, the experts' gated residual's three)
        assert_eq!(rows[..9], [(64, 1024), (1024, 64), (4, 1024), (2048, 256), (12, 256), (256, 768), (64, 1024), (1024, 64), (4, 1024)]);
        let table = m.layers[1].iter().find_map(|op| if let Op::NgramRows(x) = op { Some(*x) } else { None }).unwrap();
        assert_eq!((rows[table.table], table.per, table.row, table.eos), ((128 * 132, 16), 8, 16, 2000));
        // (the states: per GDN layer its convolution's and recurrence's; the n-gram's tokens and convolution; per attention layer its keys and values, the indexer's keys and pools)
        assert_eq!(m.states.iter().filter(|&&b| b == 1100 * 2 * 256 * 2).count(), 4);
        assert_eq!(m.states.iter().filter(|&&b| b == 8).count(), 1);
        for (s, w) in
            [(Slot::S, 1024), (Slot::Hq, 1024), (Slot::Mid, 64), (Slot::Mix, 12288 + 1024), (Slot::Qr, 640), (Slot::Iq, 512), (Slot::Gu, 1280), (Slot::Kv, 4)]
        {
            assert_eq!(m.width(s), w, "{s:?}");
        }
        // (A_log the delta rule's, read in fp32)
        assert_eq!(super::a_logs(&m).len(), 6);
        assert!(super::a_logs(&m).iter().all(|&i| m.small[i].ends_with("linear_attn.A_log")));
    }

    /// What the shared operations do not take is refused in words: a linear attention's silu gate, gated residuals over another count of streams.
    #[test]
    fn what_the_kernels_do_not_take_is_refused() {
        let mut c = tiny();
        c["text_config"]["output_gate_type"] = "silu".into();
        let e = super::super::lower::qwen4x(&describe(&c).unwrap(), vec![2000], 1100).unwrap_err().msg;
        assert!(e.contains("sigmoid gate"), "{e}");
        let mut c = tiny();
        c["text_config"]["hc_count"] = 2.into();
        let e = super::super::lower::qwen4x(&describe(&c).unwrap(), vec![2000], 1100).unwrap_err().msg;
        assert!(e.contains("streams"), "{e}");
    }
}
