//! Dense layers in the computer's memory (R12, #232; P2d's rung after the experts in RAM; docs/design/runtime.md §6: a dense layer is one expert every row routes to). A stage's chosen layers keep their packed linears
//! (`Mat::Packed`, the description's `Op::Linear`s) in pinned memory, and one slot on the GPU, the largest such layer's bytes, holds the layer being run: `Dyn::forward` copies a layer into the slot on the engine's stream
//! before its ops (`DenseRam::fetch`), so a captured step's graph replays the copy. Every step reads each such layer whole (there is no hit rate), so the step grows by their bytes over the link. Blocks that hold their
//! own weights (latent attention, the shared expert of a mixture of experts, FP8 matrices, bf16 `Mat::Dense`) stay on the GPU.

use crate::dev::{copy_h2d_async, Buf, Pinned};
use crate::ffi::{err, Pack, Result, Stream, Words};
use crate::kern35::Mat;
use crate::weights::{upload_packed, PackBuf, Packed};

/// Each array of a matrix in the slot starts at a multiple of this (the kernels' vector loads).
const ALIGN: usize = 256;

/// Which layers of a stage keep their dense matrices in the computer's memory: layers `from..end`, the stage's last ones.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Placer {
    from: usize,
    end: usize,
}

impl Placer {
    /// The last `count` of the stage's layers `first..end` (all of them where it has fewer).
    pub fn last(first: usize, end: usize, count: usize) -> Placer {
        Placer { from: end.saturating_sub(count).max(first), end }
    }

    pub fn in_ram(&self, l: usize) -> bool {
        (self.from..self.end).contains(&l)
    }

    /// How many layers it puts in the computer's memory.
    pub fn count(&self) -> usize {
        self.end - self.from
    }
}

/// A stage's layers in the computer's memory keep each matrix packed from the bf16 that a matrix released in FP8 dequantises to, where the model on the GPU holds it as
/// released (the E4M3 values with their block scales, `f8::released_shape`): another product, so a matrix of the layers kept there (`ram`: the tensors of each) that is so
/// released is refused, of whichever family. `layers` of the stage `family` are to be there.
pub fn refuse_released_f8(ck: &glyd_gpu::safetensors::Checkpoint, family: &str, layers: usize, ram: &[&[String]]) -> Result<()> {
    if ram.iter().any(|members| crate::f8::released_shape(ck, members).is_some()) {
        return err(format!(
            "{family}'s layers keep their matrices on the GPU: {layers} layers of them in the computer's memory would be packed from the bf16 its FP8 matrices dequantise to"
        ));
    }
    Ok(())
}

/// Where each matrix's three arrays (its data and its two others: blocks and their bases, or exceptions and theirs) go in a layer's bytes, for matrices of `sizes`, and the layer's bytes: each array at a multiple of
/// `ALIGN`, the matrices in order.
pub fn layout(sizes: &[[usize; 3]]) -> (Vec<[usize; 3]>, usize) {
    let mut at = 0usize;
    let offsets = sizes
        .iter()
        .map(|s| {
            let mut o = [0usize; 3];
            for k in 0..3 {
                o[k] = at;
                at = (at + s[k]).next_multiple_of(ALIGN);
            }
            o
        })
        .collect();
    (offsets, at)
}

/// A family's loader's matrices of the layers in the computer's memory, kept with their bytes until `finish` points them into the slot.
#[derive(Default)]
pub struct RamBuilder {
    placer: Placer,
    layers: Vec<(usize, Vec<(usize, Packed)>)>,
}

impl RamBuilder {
    pub fn new(placer: Placer) -> RamBuilder {
        RamBuilder { placer, layers: Vec::new() }
    }

    pub fn in_ram(&self, l: usize) -> bool {
        self.placer.in_ram(l)
    }

    /// Matrix `p` of layer `l` pushed onto `mats` (its index returned): on the GPU where the layer is not in the computer's memory, else a matrix of its shape whose bytes `finish` places.
    pub(crate) fn push(&mut self, l: usize, mats: &mut Vec<Mat>, p: Packed) -> Result<usize> {
        let i = mats.len();
        if !self.placer.in_ram(l) {
            mats.push(Mat::Packed(upload_packed(p)?));
            return Ok(i);
        }
        mats.push(Mat::Packed(PackBuf::view(Pack { data: 0, words: p.words, rows: p.rows as i64, cols: p.cols as i64 })?));
        match self.layers.last_mut() {
            Some((at, v)) if *at == l => v.push((i, p)),
            _ => self.layers.push((l, vec![(i, p)])),
        }
        Ok(i)
    }

    /// The layers in the computer's memory: each laid out (`layout`) in a pinned buffer of its own, the slot on the GPU for the largest, and their matrices in `mats` pointed into the slot. None where no layer is.
    pub fn finish(self, mats: &mut [Mat]) -> Result<Option<DenseRam>> {
        if self.layers.is_empty() {
            return Ok(None);
        }
        let plans: Vec<(Vec<[usize; 3]>, usize)> =
            self.layers.iter().map(|(_, v)| layout(&v.iter().map(|(_, p)| [p.data.len(), p.b1.len(), p.b2.len()]).collect::<Vec<_>>())).collect();
        let slot = Buf::zeroed(plans.iter().map(|p| p.1).max().unwrap_or(0).max(ALIGN))?;
        let mut layers = Vec::with_capacity(self.layers.len());
        for ((l, v), (offsets, total)) in self.layers.into_iter().zip(plans) {
            let host = Pinned::new(total.max(ALIGN))?;
            // SAFETY: the pinned buffer holds `total` bytes, written here alone.
            let dst = unsafe { std::slice::from_raw_parts_mut(host.ptr as *mut u8, total) };
            dst.fill(0);
            for ((i, p), o) in v.iter().zip(&offsets) {
                for (k, b) in [&p.data, &p.b1, &p.b2].into_iter().enumerate() {
                    dst[o[k]..o[k] + b.len()].copy_from_slice(b);
                }
                let at = |k: usize| slot.ptr + o[k] as u64;
                let words = match p.words {
                    Words::Tiered { tiers, .. } => Words::Tiered { blocks: at(1), block_base: at(2), tiers },
                    Words::Twelve { sym, .. } => Words::Twelve { exc: at(1), exc_base: at(2), sym },
                };
                let Some(Mat::Packed(_)) = mats.get(*i) else { return err(format!("matrix {i} of layer {l} in the computer's memory is not a packed one")) };
                mats[*i] = Mat::Packed(PackBuf::view(Pack { data: at(0), words, rows: p.rows as i64, cols: p.cols as i64 })?);
            }
            layers.push((l, host, total));
        }
        Ok(Some(DenseRam { slot, layers }))
    }
}

/// The dense layers of a stage in the computer's memory: the slot on the GPU, and each layer's bytes (pinned) as the slot holds them while it runs.
pub struct DenseRam {
    slot: Buf,
    layers: Vec<(usize, Pinned, usize)>,
}

impl DenseRam {
    /// Layer `l`'s matrices copied into the slot on `s` before its ops (nothing where the layer is on the GPU).
    ///
    /// # Safety
    /// `s` is the stream the layer's ops run on, so the copy waits for the slot's last reader.
    pub unsafe fn fetch(&self, l: usize, s: Stream) -> Result<()> {
        match self.layers.iter().find(|x| x.0 == l) {
            Some((_, host, total)) => copy_h2d_async(self.slot.ptr, host.ptr as *const _, *total, s),
            None => Ok(()),
        }
    }

    /// The GPU's bytes (the slot) and the computer's (every layer's).
    pub fn bytes(&self) -> (usize, usize) {
        (self.slot.bytes, self.layers.iter().map(|x| x.2).sum())
    }

    /// The layers it holds.
    pub fn layers(&self) -> Vec<usize> {
        self.layers.iter().map(|x| x.0).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A layer kept in the computer's memory packs its matrices from bf16: refused where one of them is released in FP8 (E4M3 with fp32 block scales), by the matrix's format and
    /// in the words the GLM-5.3 loader used, whichever family's it is; a bf16 matrix, and an FP8 one with no block scales beside it (held as bf16 on the GPU too), are not.
    #[test]
    fn a_matrix_released_in_fp8_is_refused_in_the_computers_memory() {
        let dir = std::env::temp_dir().join(format!("glyd-dense-ram-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file = crate::sizes::tests::safetensors(&[
            ("f.weight", "F8_E4M3", &[256, 128]),
            ("f.weight_scale_inv", "F32", &[2, 1]),
            ("n.weight", "F8_E4M3", &[128, 128]),
            ("b.weight", "BF16", &[4, 4]),
        ]);
        std::fs::write(dir.join("model.safetensors"), file).unwrap();
        let ck = glyd_gpu::safetensors::Checkpoint::open(&dir).unwrap();
        let m = |n: &str| vec![n.to_string()];
        let (f8, no_scales, bf16) = (m("f.weight"), m("n.weight"), m("b.weight"));
        assert!(refuse_released_f8(&ck, "Any family", 2, &[&bf16, &no_scales]).is_ok());
        let e = refuse_released_f8(&ck, "GLM-5.3", 2, &[&bf16, &f8]).unwrap_err().msg;
        assert_eq!(e, "GLM-5.3's layers keep their matrices on the GPU: 2 layers of them in the computer's memory would be packed from the bf16 its FP8 matrices dequantise to");
        assert!(refuse_released_f8(&ck, "Another", 1, &[&f8]).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A layer's matrices laid out one after another, each array at a multiple of 256 bytes and none overlapping; the stage's last layers are the ones in the computer's memory.
    #[test]
    fn a_layers_matrices_are_laid_out_apart_and_aligned() {
        let sizes = [[1280 * 4, 300, 20], [0, 0, 0], [1536 * 2, 17, 12]];
        let (o, total) = layout(&sizes);
        let mut spans: Vec<(usize, usize)> = o.iter().zip(&sizes).flat_map(|(o, s)| (0..3).map(move |k| (o[k], o[k] + s[k]))).collect();
        assert!(o.iter().flatten().all(|x| x % ALIGN == 0));
        spans.sort();
        assert!(spans.windows(2).all(|w| w[0].1 <= w[1].0), "{spans:?}");
        assert!(spans.iter().all(|s| s.1 <= total) && total % ALIGN == 0);
        let p = Placer::last(10, 20, 3);
        assert_eq!((p.in_ram(16), p.in_ram(17), p.in_ram(19), p.in_ram(20), p.count()), (false, true, true, false, 3));
        assert_eq!(Placer::last(10, 20, 50).count(), 10);
        assert_eq!(Placer::default().count(), 0);
    }
}
