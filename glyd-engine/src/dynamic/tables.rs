//! The tables an embedding kind reads by rows, held on the host where its description says so (`kinds::TableAt::Host`: the token embedding, an n-gram
//! block's tables): the checkpoint's files mapped (src/mapped.rs, one mapping a file), the table's bytes read at random, so that a row the system's page
//! cache does not hold is read alone (with the read-ahead a long prompt's n-gram rows filled the page cache with about 128 KB of neighbours a row, tens of GB
//! for a prompt of 32K tokens, and pushed the expert pack out of it). Each pass's rows are gathered by the host into pinned memory and copied up with the
//! pass: the embedding's into X where the pass begins (`Dyn::embed`), an n-gram block's into its slot where its operation runs (`Op::NgramRows`: staged with
//! the pass's tokens, the copy a node of a step's graph), their ids from the pass's tokens and each sequence's last two tokens as its state on the GPU holds
//! them (kernels/qwen4x.cu `qx_ngram_rows`'s hash, the same ids). Nothing of such a table is on the GPU, and the planner counts its bytes in the computer's
//! memory (`off_gpu`), at the bytes it is stored in. A table may be stored at 4 bits (`formats::q4_name`: NF4 codes and a bf16 scale a row, Glyd's slim
//! checkpoints' n-gram table), its rows decoded as they are gathered: the rows staged are bf16 either way (v0.34.2's decode). GLYD_TABLES=gpu or =host (a
//! development build's, or `force` in a check) holds every such table on the GPU (a table at 4 bits as the bf16 it decodes to), or on the host.

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

/// The checkpoint's tensors the description of the config.json `c` (parsed once) holds on the host (`held`'s, by name): its embedding's, its n-gram tables'.
/// Empty for a config no description reads.
pub fn host_names(c: &crate::runtime::config::ModelConfig) -> Vec<String> {
    let Ok(d) = crate::runtime::describe::describe_config(c) else {
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

/// The tensors of `table` the GPUs hold (the planner's table less the tables the description of `c` holds on the host, `host_names`, in either stored
/// form: a table at 4 bits stands for its bf16 name, `formats::stands_for`), and the bytes of those in the computer's memory as stored: the checkpoint's
/// files mapped, the page cache's.
pub fn off_gpu(table: &crate::sizes::Table, c: &crate::runtime::config::ModelConfig) -> (crate::sizes::Table, u64) {
    let names: std::collections::HashSet<String> = host_names(c).into_iter().collect();
    let (host, gpu): (Vec<_>, Vec<_>) = table.rows.iter().cloned().partition(|r| names.contains(crate::formats::stands_for(&r.name).as_ref()));
    (crate::sizes::Table { rows: gpu }, host.iter().map(|r| r.bytes).sum())
}

/// A table read by rows on the host: each member's rows in its file mapped (`Piece`), zeros past them (an item padded with zero rows). A member is stored as
/// the description names it (bf16 rows), or at 4 bits beside that name (`formats::q4_beside`: `X.q4` for `X.weight`, a row its bf16 scale and its NF4
/// codes), whose rows are decoded as they are gathered (`formats::nf4_row`): the rows staged are bf16 either way, `row_bytes` each.
pub struct HostTable {
    pieces: Vec<Piece>,
    held: usize,
    pub rows: usize,
    pub row_bytes: usize,
}

/// A member of a table on the host: its mapping, the offset of its first row, the table's row it begins at, its bytes a row as stored and whether at 4 bits.
struct Piece {
    map: Arc<Mapping>,
    off: usize,
    at: usize,
    stride: usize,
    q4: bool,
}

impl HostTable {
    /// Item `it` of `ck` (its members matrices of its columns, stacked: each bf16, or at 4 bits beside its name), the files mapped through `maps` (one mapping
    /// a file), their bytes read at random.
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
            let q4 = crate::formats::q4_beside(ck, m);
            let name = q4.as_deref().unwrap_or(m);
            let (f, t) = ck.get(name).ok_or_else(|| Error(format!("{}: no tensor {m}", ck.dir.display())))?;
            let stride = match &q4 {
                Some(_) => crate::formats::q4_row_bytes(&t.dtype, &t.shape, it.cols).ok_or_else(|| {
                    Error(format!("{name}: {} {:?}, where a table of {} columns at 4 bits is U8 [rows, {}]", t.dtype, t.shape, it.cols, 2 + it.cols / 2))
                })?,
                None if t.dtype == "BF16" && t.shape.len() == 2 && t.shape[1] as usize == it.cols => row_bytes,
                None => return err(format!("{m}: {} {:?}, where a bf16 table of {} columns is read by rows on the host", t.dtype, t.shape, it.cols)),
            };
            let map = match maps.get(&f.path) {
                Some(x) => x.clone(),
                None => {
                    let x = Arc::new(Mapping::open(&f.path)?);
                    maps.insert(f.path.clone(), x.clone());
                    x
                }
            };
            let (off, rows) = (t.offset as usize, t.shape[0] as usize);
            if off + rows * stride > map.len {
                return err(format!("{name}: past the end of {}", f.path.display()));
            }
            map.random(off, rows * stride);
            pieces.push(Piece { map, off, at, stride, q4: q4.is_some() });
            at += rows;
        }
        if pieces.is_empty() || at > it.rows {
            return err(format!("{}: {at} rows, where the table has {}", it.members.join(" + "), it.rows));
        }
        Ok(HostTable { pieces, held: at, rows: it.rows, row_bytes })
    }

    /// Row `r` as mapped (as stored) and whether at 4 bits; None past the members' rows (a zero row).
    fn row(&self, r: usize) -> Option<(&[u8], bool)> {
        if r >= self.held {
            return None;
        }
        let p = &self.pieces[self.pieces.partition_point(|p| p.at <= r) - 1];
        // SAFETY: inside the mapping (checked at `open`), read-only, alive with `self`.
        Some((unsafe { std::slice::from_raw_parts((p.map.ptr as *const u8).add(p.off + (r - p.at) * p.stride), p.stride) }, p.q4))
    }

    /// Rows `ids` one after another into `out`, bf16 (a row at 4 bits decoded).
    pub(super) fn gather(&self, ids: &[u64], out: &mut [u8]) -> Result<()> {
        if let Some(&r) = ids.iter().find(|&&r| r as usize >= self.rows) {
            return err(format!("row {r} of a table of {}", self.rows));
        }
        let rb = self.row_bytes;
        fill(&mut out[..ids.len() * rb], rb, |i, o| match self.row(ids[i] as usize) {
            Some((q, true)) => crate::formats::nf4_row(q, o),
            Some((r, false)) => o.copy_from_slice(r),
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

/// The table rows of `tokens`, n rows of one sequence, as kernels/qwen4x.cu's `qx_ngram_rows` reads them (`x`'s order, segments and end token): each row's
/// order positions p the token p places back (each its compressed vocabulary's id where `map` is given), the pass's or, before its first row, the sequence's
/// last ids (`hist`: each + 1, oldest first, 0 none: the pad, the end token's id); past an end token inside a segment the end token; each order's hash the
/// multipliers' XOR of them, mod each of its `per` tables' sizes at their offsets.
pub(super) fn ngram_ids(x: &NgramRows, tokens: &[u32], hist: &[i32], mult: &[u64], sizes: &[u64], offsets: &[u64], map: Option<&[i32]>) -> Vec<u64> {
    let id = |t: u32| map.map_or(t, |m| m[t as usize] as u32);
    let (eos, pad) = (x.eos, id(x.eos));
    let mut out = Vec::with_capacity(tokens.len() * sizes.len());
    for t in 0..tokens.len() {
        let (mut cut, mut m, mut mixed) = (false, 0u64, [0u64; 4]);
        for p in 0..x.order {
            let v = if cut {
                eos
            } else if p <= t {
                id(tokens[t - p])
            } else {
                // (hist[order - 2] is one back from the pass's first row)
                let k = (x.order - 1).checked_sub(p - t);
                k.and_then(|k| hist.get(k).copied()).filter(|&h| h > 0).map_or(pad, |h| (h - 1) as u32)
            };
            cut |= p > 0 && x.segments && v == eos;
            m ^= (v as u64).wrapping_mul(mult[p]);
            mixed[p] = m;
        }
        out.extend((0..sizes.len()).map(|h| mixed[h / x.per + 1] % sizes[h] + offsets[h]));
    }
    out
}

/// What a stage reads of its tables on the host each pass: the embedding's rows (a chunk's, pinned) where it holds the embedding there, and each n-gram block's
/// whose table is there (`Staged`).
pub(super) struct Hosted {
    embed: Option<Pinned>,
    ngram: Vec<Staged>,
}

/// An n-gram block's rows from its table on the host: its operation, the hash's numbers (the checkpoint's int64 tensors), its compressed vocabulary where it has
/// one, and the pass's rows (a chunk's, pinned).
struct Staged {
    op: NgramRows,
    mult: Vec<u64>,
    sizes: Vec<u64>,
    offsets: Vec<u64>,
    map: Option<Vec<i32>>,
    rows: Pinned,
}

impl Hosted {
    /// The pinned rows of part `p`'s tables on the host for stage `layers` (and the embedding's where the part holds it: the first stage, and the last where it
    /// drafts with a head of the model, which reads the next token's row), passes of up to `chunk` rows; None where it holds none there.
    pub(super) fn new(p: &Part, layers: (usize, usize), chunk: usize) -> Result<Option<Hosted>> {
        let embed = p.host_table(p.d.embed).map(|t| Pinned::new(chunk * t.row_bytes)).transpose()?;
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
            let heads = (x.order - 1) * x.per;
            if !(2..=4).contains(&x.order) || mult.len() < x.order || sizes.len() < heads || offsets.len() < heads || sizes[..heads].contains(&0) {
                return err(format!(
                    "an n-gram block of order {}'s hash of {} multipliers, {} sizes and {} offsets, where {heads} tables are read",
                    x.order,
                    mult.len(),
                    sizes.len(),
                    offsets.len()
                ));
            }
            let map = match x.map {
                Some(m) => {
                    let b = &p.small[m];
                    let mut v = vec![0i32; b.bytes / 4];
                    b.read(0, &mut v)?;
                    Some(v)
                }
                None => None,
            };
            let (row, offsets) = (heads * t.row_bytes, offsets[..heads].to_vec());
            ngram.push(Staged { op: *x, mult, sizes: sizes[..heads].to_vec(), offsets, map, rows: Pinned::new(chunk * row)? });
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
                let mut hist = vec![0i32; g.op.order - 1];
                let at = states.map_or(self.states[g.op.hist].ptr, |s| s[g.op.hist]);
                // SAFETY: the state's `order` - 1 ints, read after the passes before it (synchronized above; this copy waited for below).
                unsafe { copy_d2h_async(hist.as_mut_ptr() as *mut _, at, hist.len() * 4, self.dev.stream)? };
                self.dev.sync()?;
                if let Some(&t) = g.map.as_ref().and_then(|m| tokens[row0..row0 + n].iter().find(|&&t| t as usize >= m.len())) {
                    return err(format!("token {t}: the n-gram block's vocabulary map holds {}", g.map.as_ref().map_or(0, Vec::len)));
                }
                let ids = ngram_ids(&g.op, &tokens[row0..row0 + n], &hist, &g.mult, &g.sizes, &g.offsets, g.map.as_deref());
                // SAFETY: the pinned rows hold a chunk's (checked above); no copy reads them now (the stream is synchronized).
                t.gather(&ids, unsafe { std::slice::from_raw_parts_mut((g.rows.ptr as *mut u8).add(row0 * rb), n * rb) })?;
            }
        }
        Ok(())
    }

    /// `Op::NgramRows` of a table on the host on n rows of one sequence, or on a step's first n rows that decode (`ngram_rows_lead`: outside a sequence's own
    /// operation the rows from the pass's first): their staged rows (`stage_ngram`) copied into `e`; false where its table is the GPU's.
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

/// `tablerows DIR OUT [--tokens N] [--seed S]` (development): the rows the host stages for each n-gram block of the model in DIR (its description's roles; a
/// table at 4 bits decoded as it is gathered) for one sequence of N tokens (4,000 by default) from its start: random ids of its vocabulary from the seed, the
/// end token every 97th (the segments cut), the hash's ids as a pass from the sequence's start reads them (`ngram_ids`, no tokens before it) and the rows
/// gathered (`HostTable::gather`), each block's after the other, written to OUT (bf16); the tokens to OUT.tokens (u32, little-endian), for another engine to
/// stage the same rows from. A block of a compressed vocabulary (Engram's) is refused: its map is the tokenizer's.
#[cfg(feature = "dev")]
pub fn rows_cmd(a: &crate::args::Args) -> Result<()> {
    use crate::runtime::describe::Source;
    use crate::runtime::kinds::NgramForm;
    let (dir, out) = (std::path::Path::new(a.need(1, "DIR")?), std::path::Path::new(a.need(2, "OUT")?));
    let (n, mut r): (usize, u64) = (a.num("tokens", 4000usize)?, a.num("seed", 5u64)?.max(1));
    let (d, _) = crate::runtime::describe::describe_dir(dir)?;
    let ck = Checkpoint::open(dir).map_err(|e| Error(format!("{}: {e}", dir.display())))?;
    let i64s = |name: &str| -> Result<Vec<u64>> {
        let b = ck.read(name).map_err(|e| Error(format!("{name}: {e}")))?;
        Ok(b.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap_or_default())).collect())
    };
    let (mut rows, mut blocks, mut maps) = (Vec::new(), 0usize, HashMap::new());
    let mut tokens = Vec::new();
    for (b, roles) in d.desc.blocks.iter().zip(&d.blocks) {
        let Body::Ngram(g) = &b.body else { continue };
        if g.form != NgramForm::Segmented {
            return err(format!("an n-gram block of {:?}: its compressed vocabulary is the tokenizer's map", g.form));
        }
        let one = |role: &str| match roles.iter().find(|x| x.0 == role).map(|x| &x.1) {
            Some(Source::Tensor(t)) => Ok(t.clone()),
            other => err(format!("the n-gram block's role {role} is {other:?}, where one tensor is")),
        };
        let Some(Source::Rows(members)) = roles.iter().find(|x| x.0 == "ngram").map(|x| &x.1) else { return err("the n-gram block has no table of shards") };
        let shape = |m: &String| d.shapes.get(m).cloned().ok_or_else(|| Error(format!("{m}: no shape in the description")));
        let cols = shape(&members[0])?[1];
        let it = Item { members: members.clone(), rows: members.iter().map(|m| shape(m).map(|s| s[0])).sum::<Result<usize>>()?, cols };
        let table = HostTable::open(&ck, &it, &mut maps)?;
        let heads = (g.order - 1) * g.heads;
        let (mult, sizes, offsets) = (i64s(&one("ngram_mult")?)?, i64s(&one("ngram_sizes")?)?, i64s(&one("ngram_offsets")?)?);
        if tokens.is_empty() {
            tokens = (0..n)
                .map(|i| {
                    r ^= r << 13;
                    r ^= r >> 7;
                    r ^= r << 17;
                    if i % 97 == 96 {
                        g.eos
                    } else {
                        (r % d.desc.vocab as u64) as u32
                    }
                })
                .collect();
        }
        let x = NgramRows { table: 0, hash: [0; 3], hist: 0, per: g.heads, row: cols, eos: g.eos, order: g.order, map: None, segments: true };
        let ids = ngram_ids(&x, &tokens, &vec![0; g.order - 1], &mult, &sizes[..heads], &offsets[..heads], None);
        let at = rows.len();
        rows.resize(at + ids.len() * table.row_bytes, 0);
        table.gather(&ids, &mut rows[at..])?;
        blocks += 1;
    }
    if blocks == 0 {
        return err(format!("{}: its description has no n-gram block", dir.display()));
    }
    let io = |p: &std::path::Path, b: &[u8]| std::fs::write(p, b).map_err(|e| Error(format!("{}: {e}", p.display())));
    io(out, &rows)?;
    io(&out.with_extension("tokens"), &tokens.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>())?;
    println!("{}: {blocks} n-gram block(s), {} tokens, {} bytes of rows staged into {}", dir.display(), tokens.len(), rows.len(), out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A table stored at 4 bits beside its name (two shards, one bf16 and one `X.q4`, then zero rows) gathers on the host the bf16 rows the 4-bit rows decode
    /// to (`formats::nf4_row`) and the bf16 shard's as stored, zeros past them; the GPU's reader of the same item (`weights::item_words`) gives those words
    /// too. A 4-bit tensor of another width is refused in words.
    #[test]
    fn a_table_at_4_bits_gathers_the_rows_it_decodes_to() {
        let dir = std::env::temp_dir().join(format!("glyd-tables-q4-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (rows, cols) = (37usize, 20usize);
        let bf16: Vec<u8> = (0..2 * rows * cols).flat_map(|i| (((((i * 7919 % 211) as f32 - 105.0) / 64.0).to_bits() >> 16) as u16).to_le_bytes()).collect();
        let (a, b) = bf16.split_at(rows * cols * 2);
        let q: Vec<u8> = b.chunks_exact(cols * 2).flat_map(crate::formats::nf4_encode).collect();
        let shape = |w: usize| vec![rows as u64, w as u64];
        let wide = vec![0u8; rows * (3 + cols / 2)];
        let t =
            [("t.shard_0.weight", "BF16", shape(cols), a), ("t.shard_1.q4", "U8", shape(2 + cols / 2), &q[..]), ("u.q4", "U8", shape(3 + cols / 2), &wide[..])];
        crate::pack::write_safetensors(&dir.join("model.safetensors"), &[], &t).unwrap();
        let ck = Checkpoint::open(&dir).unwrap();
        let it = Item { members: vec!["t.shard_0.weight".into(), "t.shard_1.weight".into(), ZERO_ROWS.into()], rows: 2 * rows + 5, cols };
        let table = HostTable::open(&ck, &it, &mut HashMap::new()).unwrap();
        let mut want = a.to_vec();
        for r in q.chunks_exact(2 + cols / 2) {
            let mut o = vec![0u8; cols * 2];
            crate::formats::nf4_row(r, &mut o);
            want.extend(o);
        }
        want.resize(it.rows * cols * 2, 0);
        let ids: Vec<u64> = (0..it.rows as u64).rev().collect();
        let mut got = vec![0u8; ids.len() * cols * 2];
        table.gather(&ids, &mut got).unwrap();
        for (k, &r) in ids.iter().enumerate() {
            let r = r as usize;
            assert_eq!(got[k * cols * 2..(k + 1) * cols * 2], want[r * cols * 2..(r + 1) * cols * 2], "row {r}");
        }
        let words = crate::weights::item_words(&ck, &it).unwrap();
        assert!(words.iter().flat_map(|w| w.to_le_bytes()).eq(want.iter().copied()));
        let odd = Item { members: vec!["u.weight".into()], rows, cols };
        let e = HostTable::open(&ck, &odd, &mut HashMap::new()).err().unwrap().msg;
        assert!(e.contains("u.q4") && e.contains("at 4 bits is U8"), "{e}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The host's ids are the CPU reference's (`reference::ngram::context`, `ids`) for every row of passes cut anywhere in a sequence, read as the GPU's hash
    /// reads them (the sequence's last ids its state, + 1): Condor's trigrams, the end token inside the sequence cutting the n-grams after it, and Engram's
    /// order 4 over its compressed vocabulary with no segments (the pad's own id before the sequence).
    #[test]
    fn the_hosts_ids_are_the_references() {
        use crate::runtime::kinds::{Ngram, NgramForm, Norm};
        use crate::runtime::reference::ngram as r;
        let seg = Ngram {
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
        let engram = Ngram { order: 4, heads: 2, dim: 48, base: 50, vocab: 600, seed: 10007 * 3, conv: 0, dilation: 1, form: NgramForm::Engram, ..seg };
        let map: Vec<i32> = (0..2048).map(|t| (t * 7 % 600) as i32).collect();
        let seq: Vec<u32> = [3, 1900, 7, 7, 12, 7, 5, 2047, 0, 7, 9, 44].to_vec();
        for (n, m) in [(seg, None), (engram, Some(&map[..]))] {
            let (mult, (sizes, offsets)) = (r::multipliers(&n).unwrap(), r::tables(&n));
            let id = |t: u32| m.map_or(t, |m| m[t as usize] as u32);
            let ids: Vec<u32> = seq.iter().map(|&t| id(t)).collect();
            let want: Vec<u64> = (0..seq.len()).flat_map(|i| r::ids(&n, &mult, &sizes, &offsets, &r::context(&n, &ids[..i], ids[i], id(n.eos)))).collect();
            let x = NgramRows {
                table: 0,
                hash: [0; 3],
                hist: 0,
                per: n.heads,
                row: 1,
                eos: n.eos,
                order: n.order,
                map: m.map(|_| 0),
                segments: n.form == NgramForm::Segmented,
            };
            let h = sizes.len();
            for cut in 0..seq.len() {
                let hist: Vec<i32> = (0..n.order - 1).map(|k| cut.checked_sub(n.order - 1 - k).map_or(0, |i| ids[i] as i32 + 1)).collect();
                let got = ngram_ids(&x, &seq[cut..], &hist, &mult, &sizes, &offsets, m);
                assert_eq!(got, want[cut * h..], "{:?}: a pass from token {cut}", n.form);
            }
        }
    }
}
