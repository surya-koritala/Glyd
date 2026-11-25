//! The executor's input stage of images, for a description with a vision tower (`runtime::kinds::VisionDesc`; src/vision.rs, whatever the family): each
//! sequence's images (`set`) run through the tower when a pass first reads their rows, the tower's rows written over their tokens' embedding rows in X
//! (`Dyn::stage_images`), and every row of the sequence turned by its M-RoPE position (`vision::positions`, transformers' `get_rope_index`): text from position p
//! takes (p, p, p), an image's tokens (p, p + row, p + column), the text after it p + the larger of the image's rows and columns, the rows past the prompt
//! going on from its last position. A pass's rows' positions are given to the kernels that turn rows by their position (the attention's preparation, the
//! mean-pooled indexer's queries and pools: `Dyn::rope3`): [rows, 6] i32, a row's (t, h, w) and its pool's first token's (the pool's key turns by it), each
//! rotary angle taking its row of the plain table from the position its section turns by (`kinds::Rope::mrope_axis`): the release's engine's own rows
//! (`vision::rope_rows`, which it wrote into per-sequence tables), bit for bit.

use super::{Dyn, ModelDesc, Op, Part, Slot};
use crate::dev::{copy_h2d_async, Buf};
use crate::ffi::{err, Error, Result};
use crate::vision::{Image, Prep, Tower};
use std::collections::HashMap;
use std::sync::Arc;

/// The most rotary angles the kernels' map takes (two bits an angle in a u64).
const MOST_ANGLES: usize = 32;

/// The images of a model with a vision tower: the tower, how images are read, the M-RoPE map, each sequence's positions and images, and the pass's rows' positions.
pub struct Images {
    pub tower: Tower,
    pub prep: Prep,
    /// Two bits an angle of the rotary: the position (0 t, 1 h, 2 w) its section turns by.
    axes: u64,
    /// The mean-pooled indexer's pools' tokens (1: no such indexer): a pool's key turns by its first token's position.
    block: usize,
    /// The pass's rows' positions and their pools' first tokens' ([rows, 6] i32), and the rows it holds.
    rpos: Buf,
    rows: usize,
    /// GPU bytes the tower takes for an image of the most pixels (kept free beside the sequences' memory).
    pub reserve: usize,
    /// Each sequence's (by slot): its prompt's positions and the position after them, and its images not yet read whole: the row each begins at, the image, and
    /// its rows once the tower has run (bf16 [tokens, out]).
    seqs: HashMap<usize, SeqImages>,
    /// The images of the next reply that is not the scheduler's (`conv::Gen::see`, `Dyn::reply`).
    pub next: Vec<Arc<Image>>,
    /// A check's: every row turned at its own position, the M-RoPE positions off (`Dyn::plain_positions`).
    plain: std::cell::Cell<bool>,
}

struct SeqImages {
    pos: Vec<[u32; 3]>,
    next: u32,
    images: Vec<(usize, Arc<Image>, Option<Vec<u16>>)>,
}

/// The most pixels an image is resized to (the processor's top lowered: `--image-pixels`, 1,048,576 by default: up to 1,024 tokens an image of the release's tower),
/// set before a model is loaded.
pub static MAX_PIXELS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

impl Images {
    /// The images of the model in `dir` whose executor's description is `d` (its rotary tables and operations), for passes of up to `rows` rows: None where its
    /// description has no vision tower. A tower this engine cannot run, or a model whose rows the M-RoPE map does not reach, is refused in words.
    pub fn open(dir: &std::path::Path, d: &ModelDesc, rows: usize, dev: &crate::dev::Dev) -> Result<Option<Images>> {
        // (a directory whose config.json names no tower: nothing more read)
        let has = std::fs::read_to_string(dir.join("config.json")).is_ok_and(|t| t.contains("\"vision_config\""));
        if !has {
            return Ok(None);
        }
        let (k, label) = crate::runtime::describe::describe_dir(dir)?;
        let (Some(v), Some(roles)) = (k.desc.vision.as_ref(), k.vision.as_ref()) else { return Ok(None) };
        // (the text's rotary: every attention's the same M-RoPE, whose angles the kernels' map holds)
        let ropes: Vec<&crate::runtime::kinds::Rope> = k
            .desc
            .blocks
            .iter()
            .filter_map(|b| if let crate::runtime::kinds::Body::Mixer(crate::runtime::kinds::Mixer::Attention(a)) = &b.body { a.rope.as_ref() } else { None })
            .collect();
        let rope = ropes.first().copied().ok_or_else(|| Error(format!("{label}: a vision tower and no attention whose rotary turns its tokens")))?;
        let pairs = rope.freqs;
        if ropes.iter().any(|r| *r != rope) || rope.mrope.is_none() || pairs > MOST_ANGLES || d.ropes.iter().any(|r| r.pairs != pairs) {
            return err(format!("{label}: a vision tower with rotaries of {ropes:?}: the executor's images take one M-RoPE of at most {MOST_ANGLES} angles"));
        }
        let axes = (0..pairs).map(|i| (rope.mrope_axis(i) as u64) << (2 * i)).sum();
        let blocks: Vec<usize> = d.layers.iter().flatten().filter_map(|op| if let Op::IdxMean { block, .. } = op { Some(*block) } else { None }).collect();
        if blocks.windows(2).any(|w| w[0] != w[1]) {
            return err(format!("{label}: indexers of pools of {blocks:?} tokens"));
        }
        let prep = Prep::load(dir, v, Some(MAX_PIXELS.get().copied().unwrap_or(1 << 20)))?;
        let tower = Tower::open(dir, v, roles, dev).map_err(|e| e.context(&label))?;
        let reserve = Tower::bytes(v, prep.max_pixels / (v.patch * v.patch));
        Ok(Some(Images {
            tower,
            prep,
            axes,
            block: blocks.first().copied().unwrap_or(1),
            rpos: Buf::zeroed(rows.max(1) * 24)?,
            rows: rows.max(1),
            reserve,
            seqs: HashMap::new(),
            next: Vec::new(),
            plain: std::cell::Cell::new(false),
        }))
    }

    /// The sequence in `slot` begins with `prompt`, whose images are `images` in order (each one's place in it as many image tokens as its rows): its rows'
    /// positions, its images read when a pass reaches their rows. No image: its rows at their own positions.
    pub fn set(&mut self, slot: usize, prompt: &[u32], images: &[Arc<Image>]) -> Result<()> {
        if images.is_empty() {
            self.seqs.remove(&slot);
            return Ok(());
        }
        let t = self.prep.image_token;
        let grids: Vec<(usize, usize)> = images.iter().map(|i| i.grid()).collect();
        let (pos, next) = crate::vision::positions(prompt, t, &grids)?;
        let starts = (0..prompt.len()).filter(|&i| prompt[i] == t && (i == 0 || prompt[i - 1] != t));
        let images = starts.zip(images).map(|(at, img)| (at, img.clone(), None)).collect();
        self.seqs.insert(slot, SeqImages { pos, next, images });
        Ok(())
    }

    /// The sequence in `slot` gone: its images forgotten.
    pub fn forget(&mut self, slot: usize) {
        self.seqs.remove(&slot);
    }

    /// The position (t, h, w) row `p` of the sequence in `slot` turns by.
    fn at(&self, slot: usize, p: usize) -> [u32; 3] {
        match self.seqs.get(&slot) {
            Some(s) if p < s.pos.len() => s.pos[p],
            Some(s) => [s.next + (p - s.pos.len()) as u32; 3],
            None => [p as u32; 3],
        }
    }
}

impl Dyn {
    /// The pass's rows' positions for the kernels (`Images`), each row (sequence, position) in the pass's order (a negative position: a padding row, none), queued
    /// on the stream before the pass's work. Nothing where the model reads no images.
    pub(super) fn rope_write(&self, rows: &[(usize, i64)]) -> Result<()> {
        let Some(v) = self.vis.as_ref() else { return Ok(()) };
        if rows.len() > v.rows {
            return err(format!("a pass of {} rows, where the images' positions hold {}", rows.len(), v.rows));
        }
        let mut w: Vec<i32> = Vec::with_capacity(rows.len() * 6);
        for &(slot, p) in rows {
            if p < 0 {
                w.extend([0; 6]);
                continue;
            }
            let p = p as usize;
            let (own, first) = (v.at(slot, p), v.at(slot, p - p % v.block));
            w.extend(own.iter().chain(&first).map(|&x| x as i32));
        }
        // SAFETY: the buffer holds `v.rows` rows of 6 (checked above); the pageable vector is copied before the call returns, and the copy is ordered before the
        // pass's work on the stream (and after the passes before it, which read the buffer).
        unsafe { copy_h2d_async(v.rpos.ptr, w.as_ptr() as *const _, w.len() * 4, self.dev.stream) }
    }

    /// The rotary positions the kernels of part `p` read for the operation running now (the rows of the sequence it runs on in a pass of several), and the angles'
    /// map: (0, 0) where the model reads no images or the part is not the model's (a draft's rows turn by their own positions).
    pub(super) fn rope3(&self, p: &Part) -> (u64, u64) {
        match &self.vis {
            Some(v) if std::ptr::eq(p, &self.big) && !v.plain.get() => (v.rpos.ptr + self.rows_before(12), v.axes),
            _ => (0, 0),
        }
    }

    /// A check's: the keys the selected sequence's first `n` positions left that turn by their positions, each layer's in order: an attention's keys ([n, kv
    /// heads, head], its pairs (i, i + span / 2) of i < `pairs` turned) and a mean-pooled indexer's pools' keys ([n / block, dim], the first 2 `pairs` channels
    /// turned): (what, a row's width, a head's width, span, pairs, the theta of its table, the tokens a row covers, bf16 rows). The sequence's own rooms (a
    /// paged engine's pools are not read here).
    #[cfg(feature = "dev")]
    pub fn turned_rows(&self, n: usize) -> Result<Vec<(String, usize, usize, usize, usize, f64, usize, Vec<u16>)>> {
        if self.paged.is_some() {
            return err("turned_rows: a paged engine's keys are in its pools");
        }
        let mut out = Vec::new();
        for (l, op) in self.big.d.layers.iter().enumerate() {
            for op in op {
                let (what, ring, width, head, span, pairs, theta, covers) = match op {
                    Op::Attn(a) => match a.rope {
                        Some(r) => (format!("layer {l}'s attention keys"), a.ring, a.nk * a.d, a.d, r.span, r.pairs, self.big.d.ropes[r.table].theta, 1),
                        None => continue,
                    },
                    Op::IdxMean { dim, block, rope, pools, .. } => {
                        let r = &self.big.d.ropes[*rope];
                        (format!("layer {l}'s indexer pools"), *pools, *dim, *dim, 2 * r.pairs, r.pairs, r.theta, *block)
                    }
                    _ => continue,
                };
                let rows = n / covers;
                if rows > self.ring_cap[ring] {
                    return err(format!("{rows} rows of a ring of {}", self.ring_cap[ring]));
                }
                let mut v = vec![0u16; rows * width];
                // SAFETY: the ring's first `rows` rows (a room of its own: row p at p), read after the stream is synchronized.
                unsafe { crate::dev::copy_d2h_async(v.as_mut_ptr() as *mut _, self.cache(ring).0, v.len() * 2, self.dev.stream)? };
                self.dev.sync()?;
                out.push((what, width, head, span, pairs, theta, covers, v));
            }
        }
        Ok(out)
    }

    /// A check's: every row of the model's passes turned at its own position (true), the M-RoPE positions off; the step graph captured again with it.
    #[cfg(feature = "dev")]
    pub fn plain_positions(&mut self, on: bool) {
        if let Some(v) = &self.vis {
            v.plain.set(on);
        }
        if self.graph.is_some() {
            let _ = self.capture_graph();
        }
    }

    /// The rows of the images among a pass's prompt rows written over their tokens' embedding rows in X: `spans` each a sequence's rows of the pass (its slot, its
    /// first row in the pass, its first position, its rows). An image's rows are made by the tower the first time a pass reaches them, and let go once the pass
    /// that reads its last row is queued.
    pub(super) fn stage_images(&mut self, spans: &[(usize, usize, usize, usize)]) -> Result<()> {
        let Some(v) = self.vis.as_mut() else { return Ok(()) };
        let (dev, kern, s) = (&self.dev, &self.kern, self.dev.stream);
        let hid = self.big.d.hidden;
        let x = self.big.slot(Slot::X);
        for &(slot, row0, start, n) in spans {
            let Some(seq) = v.seqs.get_mut(&slot) else { continue };
            for (at, img, rows) in seq.images.iter_mut() {
                let (lo, hi) = (start.max(*at), (start + n).min(*at + img.tokens()));
                if lo >= hi {
                    continue;
                }
                if rows.is_none() {
                    let t0 = std::time::Instant::now();
                    let r = v.tower.run(dev, kern, img)?;
                    if r.len() != img.tokens() * hid {
                        return err(format!("the vision tower made {} rows of {hid} for an image of {} tokens", r.len() / hid, img.tokens()));
                    }
                    eprintln!(
                        "an image of {} x {} patches, {} tokens: the vision tower {:.0} ms",
                        img.gh,
                        img.gw,
                        img.tokens(),
                        t0.elapsed().as_secs_f64() * 1e3
                    );
                    *rows = Some(r);
                }
                let r = rows.as_ref().expect("made above");
                // SAFETY: X holds the pass's rows (row0 + n at most, a chunk's); the image's rows lo..hi are its rows lo - at..hi - at; the pageable rows are copied
                // before the call returns, ordered after the embedding's rows on the stream.
                unsafe {
                    copy_h2d_async(x + ((row0 + lo - start) * hid * 2) as u64, r[(lo - *at) * hid..].as_ptr() as *const _, (hi - lo) * hid * 2, s)?;
                }
            }
            // (an image whose last row this pass reads: its rows not wanted again)
            seq.images.retain(|(at, img, _)| at + img.tokens() > start + n);
        }
        Ok(())
    }
}
