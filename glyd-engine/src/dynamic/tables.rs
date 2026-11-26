//! The tables an embedding kind reads by rows, held on the host where its description says so (`kinds::TableAt::Host`: the token embedding, an n-gram
//! block's tables): the checkpoint's files mapped (src/mapped.rs, one mapping a file), the table's bytes read at random, so that a row the system's page
//! cache does not hold is read alone (with the read-ahead a long prompt's n-gram rows filled the page cache with about 128 KB of neighbours a row, tens of GB
//! for a prompt of 32K tokens, and pushed the expert pack out of it). Each pass's rows are gathered by the host into pinned memory and copied up with the
//! pass: the embedding's into X where the pass begins (`Dyn::embed`), an n-gram block's into its slot where its operation runs (`Op::NgramRows`: staged with
//! the pass's tokens, the copy a node of a step's graph), their ids from the pass's tokens and each sequence's last two tokens as its state on the GPU holds
//! them (kernels/qwen4x.cu `qx_ngram_rows`'s hash, the same ids). Nothing of such a table is on the GPU, and the planner counts its bytes in the computer's
//! memory (`off_gpu`). GLYD_TABLES=gpu or =host (a development build's, or `force` in a check) holds every such table on the GPU, or on the host.

use super::ngram::NgramRows;
use super::{Dyn, ModelDesc, Op, Part};
use crate::dev::{copy_d2h_async, copy_h2d_async, Pinned};
use crate::ffi::{err, Error, Result};
use crate::mapped::Mapping;
use crate::runtime::kinds::{Body, TableAt};
use crate::weights::{Item, ZERO_ROWS};
use glyd_gpu::safetensors::Checkpoint;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

/// A check's placement of every table read by rows (0: the description's, 1: the GPU, 2: the host).
static FORCED: AtomicU8 = AtomicU8::new(0);

/// Every table read by rows held where `at` says (None: where the description says), for the models loaded and planned after the call.
pub fn force(at: Option<TableAt>) {
    FORCED.store(at.map_or(0, |a| if a == TableAt::Gpu { 1 } else { 2 }), Ordering::Relaxed);
}

/// The placement a check (`force`) or GLYD_TABLES (gpu, host) asks of every table read by rows; None: the description's.
pub fn forced() -> Option<TableAt> {
    match FORCED.load(Ordering::Relaxed) {
        1 => Some(TableAt::Gpu),
        2 => Some(TableAt::Host),
        _ => match crate::tune!("GLYD_TABLES").as_deref() {
            Some("gpu") => Some(TableAt::Gpu),
            Some("host") => Some(TableAt::Host),
            _ => None,
        },
    }
}

/// The matrices of a lowered description held on the host: its embedding (`embed`, where it is not the head `lm`) as the kinds' `m.embed.table` says, and each
/// n-gram block's table as its kind says (`ngram`: the table's matrix and the kind's placement), unless `forced`.
pub(super) fn held(m: &crate::runtime::kinds::ModelDesc, embed: usize, lm: usize, ngram: &[(usize, TableAt)]) -> Vec<usize> {
    let at = |t: TableAt| forced().unwrap_or(t) == TableAt::Host;
    let e = (embed != lm && at(m.embed.table)).then_some(embed);
    e.into_iter().chain(ngram.iter().filter(|x| at(x.1)).map(|x| x.0)).collect()
}

/// The checkpoint's tensors the description of the config.json `text` holds on the host (`held`'s, by name): its embedding's, its n-gram tables'. Empty for a
/// config no description reads.
pub fn host_names(text: &str) -> Vec<String> {
    let Ok(d) = serde_json::from_str(text).map_err(|e| Error(e.to_string())).and_then(|v| crate::runtime::describe::describe(&v)) else {
        return Vec::new();
    };
    let at = |t: TableAt| forced().unwrap_or(t) == TableAt::Host;
    let mut out: Vec<String> = Vec::new();
    if d.head.is_some() && at(d.desc.embed.table) {
        out.push(d.embed.clone());
    }
    for (b, roles) in d.desc.blocks.iter().zip(&d.blocks) {
        if let Body::Ngram(n) = &b.body {
            if at(n.table) {
                out.extend(roles.iter().filter(|(r, _)| r == "ngram").flat_map(|(_, s)| s.names()));
            }
        }
    }
    out
}

/// The tensors of `table` the GPUs hold (the planner's table less the tables the description of `text` holds on the host, `host_names`), and the bytes of
/// those in the computer's memory: the checkpoint's files mapped, the page cache's.
pub fn off_gpu(table: &crate::sizes::Table, text: &str) -> (crate::sizes::Table, u64) {
    let names: std::collections::HashSet<String> = host_names(text).into_iter().collect();
    let (host, gpu): (Vec<_>, Vec<_>) = table.rows.iter().cloned().partition(|r| names.contains(&r.name));
    (crate::sizes::Table { rows: gpu }, host.iter().map(|r| r.bytes).sum())
}

/// A table read by rows on the host: each member's rows in its file mapped (the mapping, the offset of its first row, the table's row it begins at), zeros past
/// them (an item padded with zero rows).
pub struct HostTable {
    pieces: Vec<(Arc<Mapping>, usize, usize)>,
    held: usize,
    pub rows: usize,
    pub row_bytes: usize,
}

impl HostTable {
    /// Item `it` of `ck` (its members bf16 matrices of its columns, stacked), the files mapped through `maps` (one mapping a file), their bytes read at random.
    pub(super) fn open(ck: &Checkpoint, it: &Item, maps: &mut HashMap<PathBuf, Arc<Mapping>>) -> Result<HostTable> {
        let row_bytes = it.cols * 2;
        let (mut pieces, mut at) = (Vec::new(), 0usize);
        for (i, m) in it.members.iter().enumerate() {
            if m == ZERO_ROWS {
                if i + 1 != it.members.len() {
                    return err(format!("{}: zero rows before its last member", it.members.join(" + ")));
                }
                continue;
            }
            let (f, t) = ck.get(m).ok_or_else(|| Error(format!("{}: no tensor {m}", ck.dir.display())))?;
            if t.dtype != "BF16" || t.shape.len() != 2 || t.shape[1] as usize != it.cols {
                return err(format!("{m}: {} {:?}, where a bf16 table of {} columns is read by rows on the host", t.dtype, t.shape, it.cols));
            }
            let map = match maps.get(&f.path) {
                Some(x) => x.clone(),
                None => {
                    let x = Arc::new(Mapping::open(&f.path)?);
                    maps.insert(f.path.clone(), x.clone());
                    x
                }
            };
            let (off, rows) = (t.offset as usize, t.shape[0] as usize);
            if off + rows * row_bytes > map.len {
                return err(format!("{m}: past the end of {}", f.path.display()));
            }
            map.random(off, rows * row_bytes);
            pieces.push((map, off, at));
            at += rows;
        }
        if pieces.is_empty() || at > it.rows {
            return err(format!("{}: {at} rows, where the table has {}", it.members.join(" + "), it.rows));
        }
        Ok(HostTable { pieces, held: at, rows: it.rows, row_bytes })
    }

    /// Row `r` as mapped; None past the members' rows (a zero row).
    fn row(&self, r: usize) -> Option<&[u8]> {
        if r >= self.held {
            return None;
        }
        let (m, off, at) = &self.pieces[self.pieces.partition_point(|p| p.2 <= r) - 1];
        // SAFETY: inside the mapping (checked at `open`), read-only, alive with `self`.
        Some(unsafe { std::slice::from_raw_parts((m.ptr as *const u8).add(off + (r - at) * self.row_bytes), self.row_bytes) })
    }

    /// Rows `ids` one after another into `out`.
    pub(super) fn gather(&self, ids: &[u64], out: &mut [u8]) -> Result<()> {
        if let Some(&r) = ids.iter().find(|&&r| r as usize >= self.rows) {
            return err(format!("row {r} of a table of {}", self.rows));
        }
        let rb = self.row_bytes;
        fill(&mut out[..ids.len() * rb], rb, |i, o| match self.row(ids[i] as usize) {
            Some(r) => o.copy_from_slice(r),
            None => o.fill(0),
        });
        Ok(())
    }
}

/// `out` (items of `size` bytes) filled by `f(i, item i)`, by up to 128 threads for many items: a row of a mapped table not in the page cache is a read from
/// the disk, and many reads in flight keep the disk busy (v0.34.2's figures: a prompt of 32K tokens' 512K n-gram rows one at a time ~20 s from the disk, by
/// 32 threads 0.8-1.0 s, by 128 0.4-0.5 s).
fn fill(out: &mut [u8], size: usize, f: impl Fn(usize, &mut [u8]) + Sync) {
    let n = out.len() / size.max(1);
    let threads = (n / 64).clamp(1, 128);
    if threads == 1 {
        out.chunks_mut(size).enumerate().for_each(|(i, o)| f(i, o));
        return;
    }
    let per = n.div_ceil(threads);
    std::thread::scope(|sc| {
        for (k, part) in out.chunks_mut(per * size).enumerate() {
            let f = &f;
            sc.spawn(move || part.chunks_mut(size).enumerate().for_each(|(i, o)| f(k * per + i, o)));
        }
    });
}

/// The table rows of `tokens`, n rows of one sequence, as kernels/qwen4x.cu's `qx_ngram_rows` reads them: row t's token, the one before it and the one before
/// that (the pass's, else the sequence's last two before the pass, `hist`: each token + 1, oldest first, 0 none: the end token), the third cut to the end token
/// where the second is the end token; the bigram's hash for the first `per` tables, the trigram's for the others, each mod its table's size at its offset.
pub(super) fn ngram_ids(tokens: &[u32], hist: [i32; 2], mult: [u64; 3], sizes: &[u64], offsets: &[u64], per: usize, eos: u32) -> Vec<u64> {
    let before = |h: i32| if h > 0 { (h - 1) as u32 } else { eos };
    let mut out = Vec::with_capacity(tokens.len() * sizes.len());
    for t in 0..tokens.len() {
        let h1 = if t >= 1 { tokens[t - 1] } else { before(hist[1]) };
        let h2 = match t {
            0 => before(hist[0]),
            1 => before(hist[1]),
            _ => tokens[t - 2],
        };
        let c2 = if h1 == eos { eos } else { h2 };
        let bi = (tokens[t] as u64).wrapping_mul(mult[0]) ^ (h1 as u64).wrapping_mul(mult[1]);
        let tri = bi ^ (c2 as u64).wrapping_mul(mult[2]);
        out.extend((0..sizes.len()).map(|h| (if h < per { bi } else { tri }) % sizes[h] + offsets[h]));
    }
    out
}

/// What a stage reads of its tables on the host each pass: the embedding's rows (a chunk's, pinned) where it holds the embedding there, and each n-gram block's
/// whose table is there (`Staged`).
pub(super) struct Hosted {
    embed: Option<Pinned>,
    ngram: Vec<Staged>,
}

/// An n-gram block's rows from its table on the host: its operation, the hash's numbers (the checkpoint's int64 tensors), and the pass's rows (a chunk's, pinned).
struct Staged {
    op: NgramRows,
    mult: [u64; 3],
    sizes: Vec<u64>,
    offsets: Vec<u64>,
    rows: Pinned,
}

impl Hosted {
    /// The pinned rows of part `p`'s tables on the host for stage `layers` (and the embedding's where `first`), passes of up to `chunk` rows; None where it holds
    /// none there.
    pub(super) fn new(p: &Part, layers: (usize, usize), first: bool, chunk: usize) -> Result<Option<Hosted>> {
        let embed = match p.host_table(p.d.embed) {
            Some(t) if first => Some(Pinned::new(chunk * t.row_bytes)?),
            _ => None,
        };
        let mut ngram = Vec::new();
        for op in p.d.layers[layers.0..layers.1].iter().flatten() {
            let Op::NgramRows(x) = op else { continue };
            let Some(t) = p.host_table(x.table) else { continue };
            let read = |i: usize| -> Result<Vec<u64>> {
                let b = &p.small[x.hash[i]];
                let mut v = vec![0u64; b.bytes / 8];
                b.read(0, &mut v)?;
                Ok(v)
            };
            let (mult, sizes, offsets) = (read(0)?, read(1)?, read(2)?);
            let heads = 2 * x.per;
            if mult.len() < 3 || sizes.len() < heads || offsets.len() < heads || sizes[..heads].contains(&0) {
                return err(format!(
                    "an n-gram block's hash of {} multipliers, {} sizes and {} offsets, where {heads} tables are read",
                    mult.len(),
                    sizes.len(),
                    offsets.len()
                ));
            }
            let row = heads * t.row_bytes;
            ngram.push(Staged {
                op: *x,
                mult: [mult[0], mult[1], mult[2]],
                sizes: sizes[..heads].to_vec(),
                offsets: offsets[..heads].to_vec(),
                rows: Pinned::new(chunk * row)?,
            });
        }
        Ok((embed.is_some() || !ngram.is_empty()).then_some(Hosted { embed, ngram }))
    }
}

impl Part {
    /// Matrix `k`'s table where it is held on the host.
    pub(super) fn host_table(&self, k: usize) -> Option<&HostTable> {
        self.host.iter().find(|x| x.0 == k).map(|x| &x.1)
    }
}

impl Dyn {
    /// Whether this stage reads a table on the host (its rows need each pass's tokens on the host: no step is launched before the tokens it reads are there).
    pub(super) fn reads_host_tables(&self) -> bool {
        self.hosted.is_some()
    }

    /// The tokens' embedding rows from its table on the host into X (`tokens` in a chunk's rows): gathered into the pinned rows, copied up on the stream, which
    /// the pinned rows are read by before the next pass writes them (waited for here); false where the embedding is the GPU's.
    pub(super) unsafe fn embed_host(&self, tokens: &[u32]) -> Result<bool> {
        let (Some(t), Some(rows)) = (self.big.host_table(self.big.d.embed), self.hosted.as_ref().and_then(|h| h.embed.as_ref())) else { return Ok(false) };
        let bytes = tokens.len() * t.row_bytes;
        if bytes > rows.bytes {
            return err(format!("{} rows of the embedding, where a pass stages {}", tokens.len(), rows.bytes / t.row_bytes));
        }
        self.dev.sync()?;
        let ids: Vec<u64> = tokens.iter().map(|&x| x as u64).collect();
        t.gather(&ids, std::slice::from_raw_parts_mut(rows.ptr as *mut u8, bytes))?;
        copy_h2d_async(self.big.slot(super::Slot::X), rows.ptr, bytes, self.dev.stream)?;
        Ok(true)
    }

    /// The pass's n-gram rows from their tables on the host (`tokens`: the pass's, each sequence's rows in a pass of several), staged in pinned memory for the
    /// blocks' operations (`ngram_rows_host`): each sequence's last two tokens read back from its state (the passes before it waited for), its rows' ids, the
    /// rows gathered.
    pub(super) fn stage_ngram(&self, tokens: &[u32]) -> Result<()> {
        let Some(h) = self.hosted.as_ref().filter(|h| !h.ngram.is_empty()) else { return Ok(()) };
        self.dev.sync()?;
        let seqs: Vec<(usize, usize, Option<&[u64]>)> =
            if self.parts.is_empty() { vec![(0, tokens.len(), None)] } else { self.parts.iter().map(|p| (p.row0, p.rows, Some(&p.states[..]))).collect() };
        for g in &h.ngram {
            let t = self.big.host_table(g.op.table).ok_or_else(|| Error("an n-gram block's table is not on the host".into()))?;
            let rb = g.sizes.len() * t.row_bytes;
            for &(row0, n, states) in &seqs {
                if row0 + n > tokens.len() || (row0 + n) * rb > g.rows.bytes {
                    return err(format!("rows {row0}..{} of a pass of {} tokens, where a pass stages {}", row0 + n, tokens.len(), g.rows.bytes / rb));
                }
                let mut hist = [0i32; 2];
                let at = states.map_or(self.states[g.op.hist].ptr, |s| s[g.op.hist]);
                // SAFETY: the state's two ints, read after the passes before it (synchronized above; this copy waited for below).
                unsafe { copy_d2h_async(hist.as_mut_ptr() as *mut _, at, 8, self.dev.stream)? };
                self.dev.sync()?;
                let ids = ngram_ids(&tokens[row0..row0 + n], hist, g.mult, &g.sizes, &g.offsets, g.op.per, g.op.eos);
                // SAFETY: the pinned rows hold a chunk's (checked above); no copy reads them now (the stream is synchronized).
                t.gather(&ids, unsafe { std::slice::from_raw_parts_mut((g.rows.ptr as *mut u8).add(row0 * rb), n * rb) })?;
            }
        }
        Ok(())
    }

    /// `Op::NgramRows` of a table on the host on n rows of one sequence: its staged rows (`stage_ngram`) copied into `e`; false where its table is the GPU's.
    pub(super) unsafe fn ngram_rows_host(&self, x: &NgramRows, e: u64, n: usize) -> Result<bool> {
        let Some(g) = self.hosted.as_ref().and_then(|h| h.ngram.iter().find(|g| g.op.table == x.table)) else { return Ok(false) };
        let rb = g.sizes.len() * x.row * 2;
        copy_h2d_async(e, (g.rows.ptr as *const u8).add(self.rows_before(rb / 2) as usize) as *const _, n * rb, self.dev.stream)?;
        Ok(true)
    }
}

/// Whether `d`'s matrix `k` is a table it holds on the host.
pub(super) fn on_host(d: &ModelDesc, k: usize) -> bool {
    d.host.contains(&k)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host's ids are the CPU reference's (`reference::ngram::context`, `ids`) for every row of passes cut anywhere in a sequence, the end token inside it
    /// cutting the trigrams after it: the GPU's hash, which the reference holds the kernel to, read on the host.
    #[test]
    fn the_hosts_ids_are_the_references() {
        use crate::runtime::kinds::{Ngram, NgramForm, Norm};
        use crate::runtime::reference::ngram as r;
        let n = Ngram {
            order: 3,
            heads: 8,
            dim: 256,
            base: 1000,
            vocab: 2048,
            seed: 1234,
            index: 0,
            eos: 7,
            streams: 4,
            conv: 4,
            dilation: 3,
            norm: Norm::rms_plus_one(1e-6),
            form: NgramForm::Segmented,
            table: TableAt::Host,
        };
        let mult = r::multipliers(&n).unwrap();
        let (sizes, offsets) = r::tables(&n);
        let seq: Vec<u32> = [3, 1900, 7, 7, 12, 7, 5, 2047, 0, 7, 9, 44].to_vec();
        let want: Vec<u64> = (0..seq.len()).flat_map(|i| r::ids(&n, &mult, &sizes, &offsets, &r::context(&n, &seq[..i], seq[i]))).collect();
        for cut in 0..seq.len() {
            let hist = |k: usize| if cut >= k { seq[cut - k] as i32 + 1 } else { 0 };
            let got = ngram_ids(&seq[cut..], [hist(2), hist(1)], [mult[0], mult[1], mult[2]], &sizes, &offsets, n.heads, n.eos);
            assert_eq!(got, want[cut * 16..], "a pass from token {cut}");
        }
    }
}
