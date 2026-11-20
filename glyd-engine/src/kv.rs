//! The lossless 12-bit KV cache of the library (glyd_gpu.h's glyd_gpu_kv_*): per layer a pool of pages (16 tokens of
//! every KV head, K and V, 12 bits a value, read back bit for bit), one escape pool and one raw pool for the pages that
//! hold values outside their layer's windows, and the tables (the windows) made for the model once.
//!
//! One user: the block table is the identity (block i is tokens 16 i to 16 i + 15) and a token's slot is its position.
//!
//! A value the cache cannot hold is never kept quietly: where the extra pools have no room the write kernel marks the page (void: what it holds is not what was written) and names its block in a ring of the status words, in the
//! step that wrote it. `Kv::drain` reads the rings after every step; the scheduler (sched.rs) takes the step's tokens back from the sequences that hold such a block and runs them again, or ends them with an error in words.

use crate::dev::Buf;
use crate::ffi::{err, Error, KvGeom, Lib, Result};
use glyd_gpu::json::{self, Value};
use glyd_gpu::safetensors::File;
use std::collections::HashSet;
use std::path::Path;

const CH: f64 = 64.0; // entries in an escape chunk
const POOL_MARGIN: f64 = 1.0;
const POOL_MIN: i64 = 1024; // chunks the escape pool has at least
const POOL_PERCENT: f64 = 1.3; // its share of the cache where the tables have no sweep
const POOL_MIN_PERCENT: f64 = 0.5;
const POOL_MAX_PERCENT: f64 = 15.0;
const RAW_K: f64 = 1.5; // the raw pool: layer-pages' planes of every block, for a model with tokens whose pages are raw
const RAW_K_NONE: f64 = 0.25;
const RAW_MIN: i64 = 256; // slabs the raw pool has at least
/// Places of each of the status's two rings at the least; a cache made for a prompt chunk of `c` tokens and `MAX_ROWS` sequences has `ring_for(c)` (a step names a block for each 16 tokens it writes, and two for each sequence).
pub const RING: i64 = 256;

pub fn ring_for(chunk: usize) -> i64 {
    RING.max((chunk / 16 + 4 * crate::model::MAX_ROWS + 64) as i64)
}

/// The status words before the rings (glyd_gpu.h): [0] entries that found no place, [4] extra pool chunks taken now, [5] the most taken at once, [8] raw slabs taken now, [9] the most, [10] pages made raw, [11] pages made void,
/// [12] the void ring's head (blocks named so far), [13] the raw ring's head.
pub const HEAD_WORDS: usize = 16;

/// What a model's KV tables hold: the windows (uint8 [layers, heads, channels], keys and values) and the figures that size the pools.
pub struct KvTables {
    pub layers: usize,
    pub heads: usize,
    pub channels: usize,
    pub k: Vec<u8>,
    pub v: Vec<u8>,
    pub positions: u64,
    pub arena: i64,
    pub worst_chunks: Option<f64>,
    pub heavy_tokens: u64,
    pub model: String,
}

impl KvTables {
    /// Whether `path` is a tables file of this engine's rule (`kvcal::VERSION`): a file of another version is made again.
    pub fn is_current(path: &Path) -> bool {
        let meta = File::open(path).ok().and_then(|f| f.metadata.iter().find(|(k, _)| k == "meta").map(|(_, v)| v.clone()));
        meta.and_then(|m| json::parse(&m).ok()).and_then(|m| m.get("version").and_then(Value::as_u64)) == Some(crate::kvcal::VERSION as u64)
    }

    /// `kv-tables.safetensors` (tools/npz_to_tables.py makes it from the plugin's tables).
    pub fn load(path: &Path) -> Result<KvTables> {
        let f = File::open(path).map_err(|e| Error(format!("{e} (this model's setup file: the first run makes it)")))?;
        let meta = f.metadata.iter().find(|(k, _)| k == "meta").map(|(_, v)| v.clone()).ok_or_else(|| Error(format!("{}: no meta", path.display())))?;
        let m = json::parse(&meta).map_err(|e| Error(format!("{}: meta: {e}", path.display())))?;
        let tk = f.get("k").ok_or_else(|| Error(format!("{}: no k", path.display())))?;
        let tv = f.get("v").ok_or_else(|| Error(format!("{}: no v", path.display())))?;
        if tk.shape.len() != 3 || tk.shape != tv.shape {
            return err(format!("{}: k {:?} and v {:?}: not [layers, heads, channels]", path.display(), tk.shape, tv.shape));
        }
        let pool = m.get("store");
        let get = |k: &str| pool.and_then(|p| p.get(k));
        let num = |v: Option<&Value>| match v {
            Some(Value::Number(n)) => n.parse::<f64>().ok(),
            _ => None,
        };
        Ok(KvTables {
            layers: tk.shape[0] as usize,
            heads: tk.shape[1] as usize,
            channels: tk.shape[2] as usize,
            k: f.read(tk).map_err(|e| Error(e.to_string()))?,
            v: f.read(tv).map_err(|e| Error(e.to_string()))?,
            positions: m.get("positions").and_then(Value::as_u64).unwrap_or(0),
            arena: num(get("bay")).map(|a| a as i64).unwrap_or(15),
            worst_chunks: num(get("worst_chunks_per_layer_leaf")),
            heavy_tokens: num(get("hard")).map(|a| a as u64).unwrap_or(0),
            model: m.get("model").and_then(Value::as_str).unwrap_or("").to_string(),
        })
    }
}

/// What the cache lost or made heavy in the steps since it was last asked: blocks with a void layer-page (what they hold is not what was written), blocks with a raw layer-page (held exactly, in the scarce second
/// extra memory: not to be kept for a later prompt), and whether more blocks were named than the status's rings hold (then any block may be lost).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Lost {
    pub void: Vec<i32>,
    pub raw: Vec<i32>,
    pub overflow: bool,
}

impl Lost {
    pub fn is_empty(&self) -> bool {
        self.void.is_empty() && self.raw.is_empty() && !self.overflow
    }

    pub fn merge(&mut self, o: Lost) {
        self.void.extend(o.void);
        self.raw.extend(o.raw);
        self.overflow |= o.overflow;
    }
}

pub struct Kv {
    pub geom: KvGeom,
    pub layers: usize,
    pub group: i64,
    pub nblocks: i64,
    pub pools: Vec<Buf>,
    pub kbase: Vec<Buf>,
    pub vbase: Vec<Buf>,
    pub status: Buf,
    /// The write kernel's bitmap of blocks that have a void or a raw layer-page (two bits a block: it names each block once while it stays so), and its bits.
    pub lost: Buf,
    pub nlost: i64,
    /// Places of each of the status's rings.
    pub ring: i64,
    pub block_table: Buf,
    _escape: Buf,
    _raw: Buf,
    /// Device bytes of the whole cache: the pools, the extra pools and the windows.
    pub bytes: usize,
    pub chunks: i64,
    pub slabs: i64,
    pub page_bytes: i64,
    /// The positions the tables were made for: the longest sequence the cache serves (0: not recorded).
    pub tables_positions: u64,
    /// Blocks named void and not cleared or rewritten since (what they hold is not what was written), and the status's header as the last step left it.
    pub void: HashSet<i32>,
    pub head: [u32; HEAD_WORDS],
    seen_void: u32,
    seen_raw: u32,
    /// Pages named void since the start, and the steps in which the rings overflowed.
    pub void_named: u64,
    pub overflows: u64,
}

/// The bytes a token takes in every layer's pool and the shares of the extra pools (for sizing a cache to a budget).
pub fn bytes_per_token(lib: &Lib, t: &KvTables) -> Result<f64> {
    let page = lib.kv_page_bytes(t.heads as i64, t.channels as i64, t.arena)? as f64;
    let (pool, raw) = shares(lib, t)?;
    Ok(t.layers as f64 * page / 16.0 * (1.0 + pool / 100.0 + raw / 100.0))
}

/// The escape pool's and the raw pool's shares of the cache (percent), from the tables' sweep.
fn shares(lib: &Lib, t: &KvTables) -> Result<(f64, f64)> {
    let h = t.heads as i64;
    let page = lib.kv_page_bytes(h, t.channels as i64, t.arena)? as f64;
    let pool = match t.worst_chunks {
        Some(w) => ((100.0 * w * POOL_MARGIN * (CH * 4.0 + 4.0) / page).clamp(POOL_MIN_PERCENT, POOL_MAX_PERCENT) * 100.0).round() / 100.0,
        None => POOL_PERCENT,
    };
    let k = if t.heavy_tokens > 0 { RAW_K } else { RAW_K_NONE };
    let raw = (100.0 * k * h as f64 * 2048.0 / (t.layers as f64 * page) * 100.0).round() / 100.0;
    Ok((pool, raw))
}

/// The extra pools' sizes (chunks of the escape pool, slabs of the raw pool) for a cache of `total` bytes of pages, as the plugin sizes them: the shares of it that `shares` gives, with floors. For measuring, GLYD_ENGINE_RESERVE_1 and
/// GLYD_ENGINE_RESERVE_2 set the shares (percent of the cache) and GLYD_ENGINE_RESERVE_1_UNITS and GLYD_ENGINE_RESERVE_2_UNITS the sizes themselves, as the plugin's GLYD_KV_RESERVE_* do: a pool too small for what a text needs is how
/// the engine's handling of a full pool is tried.
fn pools_for(lib: &Lib, t: &KvTables, total: f64) -> Result<(i64, i64)> {
    let (mut pool_pct, mut raw_pct) = shares(lib, t)?;
    let num = |v: Option<String>| v.and_then(|v| v.trim().parse::<f64>().ok());
    if let Some(p) = num(crate::tune!("GLYD_ENGINE_RESERVE_1")) {
        pool_pct = p.clamp(0.0, 50.0);
    }
    if let Some(p) = num(crate::tune!("GLYD_ENGINE_RESERVE_2")) {
        raw_pct = p.clamp(0.0, 50.0);
    }
    let h = t.heads as f64;
    let mut chunks = if pool_pct > 0.0 { POOL_MIN.max((pool_pct / 100.0 * total / (CH * 4.0 + 4.0)) as i64) } else { 0 };
    let mut slabs = if raw_pct > 0.0 { RAW_MIN.max((raw_pct / 100.0 * total / (h * 2048.0)) as i64) } else { 0 };
    if let Some(u) = num(crate::tune!("GLYD_ENGINE_RESERVE_1_UNITS")) {
        chunks = (u as i64).clamp(0, 1 << 30);
    }
    if let Some(u) = num(crate::tune!("GLYD_ENGINE_RESERVE_2_UNITS")) {
        slabs = (u as i64).clamp(0, 1 << 30);
    }
    Ok((chunks, slabs))
}

impl Kv {
    /// No cache at all (an engine that is calibrating: its keys and values go to the calibration, not to pages).
    pub fn empty(heads: usize, group: usize, layers: usize) -> Result<Kv> {
        let z = Buf::new(16)?;
        let geom = KvGeom { heads: heads as i64, dim: 128, arena: 0, page_stride: 0, escape: z.ptr, nchunks: 0, raw: z.ptr, nslabs: 0 };
        Ok(Kv {
            geom,
            layers,
            group: group as i64,
            nblocks: 0,
            pools: Vec::new(),
            kbase: Vec::new(),
            vbase: Vec::new(),
            status: Buf::zeroed(4 * (HEAD_WORDS + 2 * RING as usize))?,
            lost: Buf::zeroed(16)?,
            nlost: 0,
            ring: RING,
            block_table: Buf::new(16)?,
            _escape: z,
            _raw: Buf::new(16)?,
            bytes: 0,
            chunks: 0,
            slabs: 0,
            page_bytes: 0,
            tables_positions: 0,
            void: HashSet::new(),
            head: [0; HEAD_WORDS],
            seen_void: 0,
            seen_raw: 0,
            void_named: 0,
            overflows: 0,
        })
    }

    /// A cache of `tokens` tokens (rounded up to whole pages) for a model with these tables.
    pub fn new(lib: &Lib, t: &KvTables, heads: usize, group: usize, layers: usize, tokens: usize, ring: i64) -> Result<Kv> {
        if t.heads != heads || t.layers != layers || t.channels != 128 {
            return err("the tables are not this model's");
        }
        if group == 0 || group > 8 {
            return err(format!("{group} query heads to a KV head: not supported"));
        }
        let (h, d, a) = (heads as i64, 128i64, t.arena);
        let page = lib.kv_page_bytes(h, d, a)?;
        let nblocks = tokens.div_ceil(16) as i64;
        let total = nblocks as f64 * page as f64 * layers as f64;
        let (chunks, slabs) = pools_for(lib, t, total)?;
        let escape = Buf::zeroed(lib.kv_escape_bytes(chunks)? as usize)?;
        let raw = Buf::zeroed(lib.kv_raw_bytes(slabs, h)? as usize)?;
        let mut pools = Vec::with_capacity(layers);
        let (mut kbase, mut vbase) = (Vec::new(), Vec::new());
        let hd = (h * d) as usize;
        let mut bytes = escape.bytes + raw.bytes;
        for l in 0..layers {
            let p = Buf::new(nblocks as usize * page as usize)?;
            bytes += p.bytes;
            pools.push(p);
            let kb = Buf::new(hd)?;
            kb.write(0, &t.k[l * hd..(l + 1) * hd])?;
            let vb = Buf::new(hd)?;
            vb.write(0, &t.v[l * hd..(l + 1) * hd])?;
            bytes += 2 * hd;
            kbase.push(kb);
            vbase.push(vb);
        }
        let status = Buf::zeroed(4 * (HEAD_WORDS + 2 * ring as usize))?;
        let nlost = 2 * nblocks;
        let lost = Buf::zeroed(4 * (nlost as usize).div_ceil(32).max(1))?;
        bytes += status.bytes + lost.bytes;
        let table: Vec<i32> = (0..nblocks as i32).collect();
        let block_table = Buf::new(4 * nblocks as usize)?;
        block_table.write(0, &table)?;
        bytes += block_table.bytes;
        let geom = KvGeom { heads: h, dim: d, arena: a, page_stride: page, escape: escape.ptr, nchunks: chunks, raw: raw.ptr, nslabs: slabs };
        Ok(Kv {
            geom,
            layers,
            group: group as i64,
            nblocks,
            pools,
            kbase,
            vbase,
            status,
            lost,
            nlost,
            ring,
            block_table,
            _escape: escape,
            _raw: raw,
            bytes,
            chunks,
            slabs,
            page_bytes: page,
            tables_positions: t.positions,
            void: HashSet::new(),
            head: [0; HEAD_WORDS],
            seen_void: 0,
            seen_raw: 0,
            void_named: 0,
            overflows: 0,
        })
    }

    /// The block numbers the status's ring of `kind` (0: void pages, 1: raw pages) names between the heads `seen` and `head` (read now: the work queued must be done), and whether it named more than it holds.
    fn ring_blocks(&self, kind: usize, head: u32, seen: u32) -> Result<(Vec<i32>, bool)> {
        let n = head.wrapping_sub(seen) as u64;
        if n == 0 {
            return Ok((Vec::new(), false));
        }
        let ring = self.ring as u64;
        let take = n.min(ring);
        let mut buf = vec![0u32; ring as usize];
        self.status.read(4 * (HEAD_WORDS + kind * ring as usize), &mut buf)?;
        let first = head.wrapping_sub(take as u32);
        let mut blocks: Vec<i32> =
            (0..take as u32).filter_map(|j| Some(buf[(first.wrapping_add(j) as u64 % ring) as usize]).filter(|&e| e != 0).map(|e| e as i32 - 1)).collect();
        blocks.sort_unstable();
        blocks.dedup();
        Ok((blocks, n > ring))
    }

    /// The blocks the rings have named since the last call, from the status's header `w` as the work queued so far left it (the caller read it after the stream was synchronized). Void blocks stay in `self.void` until they are
    /// cleared or rewritten.
    pub fn drain(&mut self, w: &[u32; HEAD_WORDS]) -> Result<Lost> {
        self.head = *w;
        let (void, ov) = self.ring_blocks(0, w[12], self.seen_void)?;
        let (raw, or) = self.ring_blocks(1, w[13], self.seen_raw)?;
        self.seen_void = w[12];
        self.seen_raw = w[13];
        self.void.extend(void.iter().copied());
        self.void_named += void.len() as u64;
        if ov || or {
            self.overflows += 1;
        }
        Ok(Lost { void, raw, overflow: ov || or })
    }

    /// Blocks written from their first token (a new life: the page was cleared) or cleared: no longer void.
    pub fn renew(&mut self, blocks: &[i32]) {
        if !self.void.is_empty() {
            for b in blocks {
                self.void.remove(b);
            }
        }
    }

    /// The longest sequence the tables cover.
    pub fn max_seq(&self) -> usize {
        if self.tables_positions > 0 {
            self.tables_positions as usize
        } else {
            self.tokens()
        }
    }

    pub fn tokens(&self) -> usize {
        self.nblocks as usize * 16
    }

    /// The write kernel's status words (glyd_gpu.h): [0] entries that found no place, [10] pages made raw, [11] pages made void (read now: the work queued must be done).
    pub fn status_words(&self) -> Result<Vec<u32>> {
        let mut w = vec![0u32; HEAD_WORDS];
        self.status.read(0, &mut w)?;
        Ok(w)
    }

    /// After the single-sequence paths' steps: an error in words where a page was lost since the last call (read now: the work queued must be done).
    pub fn check_status(&mut self) -> Result<()> {
        let mut w = [0u32; HEAD_WORDS];
        self.status.read(0, &mut w)?;
        self.check_head(&w)
    }

    /// `check_status` from a header already read.
    pub fn check_head(&mut self, w: &[u32; HEAD_WORDS]) -> Result<()> {
        let l = self.drain(w)?;
        if !l.void.is_empty() || l.overflow {
            return err(LOST_MESSAGE);
        }
        Ok(())
    }
}

/// What a request is told where the cache could not hold what it needs: its keys and values did not all find a place in the cache's extra memory, so the reply is not to be trusted.
pub const LOST_MESSAGE: &str = "the lossless KV cache has no room for this text, so the reply could not be kept exact: use a shorter text";

/// What a request is told where it cannot be held even when it runs alone.
pub const ALONE_MESSAGE: &str = "the lossless KV cache cannot hold this request, even alone: send a shorter text";

/// Tokens in a page.
pub const PAGE: usize = 16;

/// A cached page's place in a chain: the key of its tokens after those before it, the key before, and its tokens (to tell a collision).
struct Entry {
    key: u64,
    prev: u64,
    toks: [u32; PAGE],
}

/// The key of a page of tokens that follows the pages whose chain ends in `prev` (0 where it is the first).
pub fn chain_key(prev: u64, toks: &[u32]) -> u64 {
    let mut h = prev ^ 0x9E37_79B9_7F4A_7C15;
    for &t in toks {
        h = (h ^ t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    }
    h | 1
}

/// The pages of the cache lent to sequences: a free list, the number of holders of every page, and the full pages of sequences that have ended kept by what their tokens were (a chain of keys), so a
/// prompt that begins with the same tokens finds them. Of the cached pages no one holds, the oldest is the first taken back, a chain's last page before its first.
/// A page the cache found lost (void) or heavy (a raw layer-page, held in the scarce second extra memory) is out of the cache and is not kept for a later prompt: it is cleared as soon as no one holds it.
pub struct Pager {
    free: Vec<i32>,
    refs: Vec<u32>,
    entry: Vec<Option<Entry>>,
    heavy: Vec<bool>,
    cache: std::collections::HashMap<u64, i32>,
    idle: std::collections::BTreeMap<u64, i32>,
    age: Vec<u64>,
    tick: u64,
}

impl Pager {
    pub fn new(pages: usize) -> Pager {
        Pager {
            free: (0..pages as i32).rev().collect(),
            refs: vec![0; pages],
            entry: (0..pages).map(|_| None).collect(),
            heavy: vec![false; pages],
            cache: Default::default(),
            idle: Default::default(),
            age: vec![0; pages],
            tick: 0,
        }
    }

    pub fn pages(&self) -> usize {
        self.refs.len()
    }

    /// Pages that can be had now: the free and the cached that no one holds.
    pub fn available(&self) -> usize {
        self.free.len() + self.idle.len()
    }

    /// A page for a new holder (the free first, else the oldest cached that no one holds), None where there is none.
    pub fn alloc(&mut self) -> Option<i32> {
        let p = match self.free.pop() {
            Some(p) => p,
            None => {
                let (&age, &p) = self.idle.iter().next()?;
                self.idle.remove(&age);
                if let Some(e) = self.entry[p as usize].take() {
                    self.cache.remove(&e.key);
                }
                p
            }
        };
        self.refs[p as usize] = 1;
        self.heavy[p as usize] = false;
        Some(p)
    }

    /// The run of cached pages `tokens` begins with (at most `max` of them), each now held by the caller.
    #[allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]
    pub fn lookup(&mut self, tokens: &[u32], max: usize) -> Vec<i32> {
        let mut out = Vec::new();
        let mut prev = 0;
        for chunk in tokens.chunks_exact(PAGE).take(max) {
            let key = chain_key(prev, chunk);
            let Some(&p) = self.cache.get(&key) else { break };
            match &self.entry[p as usize] {
                Some(e) if e.prev == prev && e.toks[..] == *chunk => {}
                _ => break,
            }
            if self.refs[p as usize] == 0 {
                self.idle.remove(&self.age[p as usize]);
            }
            self.refs[p as usize] += 1;
            out.push(p);
            prev = key;
        }
        out
    }

    /// A page written in full with `toks`, after the pages whose chain ends in `prev`: kept under its key (unless the same tokens are cached on another page, or the page is lost or heavy). Returns the key, for the next page's chain.
    pub fn publish(&mut self, page: i32, prev: u64, toks: &[u32]) -> u64 {
        let key = chain_key(prev, toks);
        if !self.heavy[page as usize] && !self.cache.contains_key(&key) && self.entry[page as usize].is_none() {
            let mut t = [0u32; PAGE];
            t.copy_from_slice(toks);
            self.entry[page as usize] = Some(Entry { key, prev, toks: t });
            self.cache.insert(key, page);
        }
        key
    }

    /// A holder gives its pages back: the cached ones that no one holds now wait to be taken back, the others (a last page not full, a page whose tokens another page holds, a lost or heavy page) are returned: the caller clears those on the GPU.
    pub fn release(&mut self, pages: &[i32]) -> Vec<i32> {
        let mut freed = Vec::new();
        for &p in pages.iter().rev() {
            let r = &mut self.refs[p as usize];
            debug_assert!(*r > 0, "page {p} released twice");
            *r = r.saturating_sub(1);
            if *r == 0 {
                if self.entry[p as usize].is_some() && !self.heavy[p as usize] {
                    self.tick += 1;
                    self.age[p as usize] = self.tick;
                    self.idle.insert(self.tick, p);
                } else {
                    if let Some(e) = self.entry[p as usize].take() {
                        self.cache.remove(&e.key);
                    }
                    self.heavy[p as usize] = false;
                    self.free.push(p);
                    freed.push(p);
                }
            }
        }
        freed
    }

    /// A page the cache found lost or heavy: out of the cache, so that no prompt finds it, and, where `heavy`, cleared as soon as no one holds it (not kept for a later prompt). True where no one held it and it was waiting
    /// in the cache: it is free now, and the caller clears it on the GPU.
    pub fn evict(&mut self, page: i32, heavy: bool) -> bool {
        let p = page as usize;
        if p >= self.refs.len() {
            return false;
        }
        if let Some(e) = self.entry[p].take() {
            self.cache.remove(&e.key);
        }
        if heavy {
            self.heavy[p] = true;
        }
        if self.refs[p] == 0 && self.idle.get(&self.age[p]) == Some(&page) {
            self.idle.remove(&self.age[p]);
            self.heavy[p] = false;
            self.free.push(page);
            return true;
        }
        false
    }

    /// Every cached page that no one holds taken out of the cache and made free: returned, for the caller to clear on the GPU.
    pub fn drop_cache(&mut self) -> Vec<i32> {
        let idle: Vec<i32> = self.idle.values().copied().collect();
        self.idle.clear();
        for &p in &idle {
            if let Some(e) = self.entry[p as usize].take() {
                self.cache.remove(&e.key);
            }
            self.heavy[p as usize] = false;
            self.free.push(p);
        }
        idle
    }

    /// Pages that are cached and that no one holds.
    pub fn idle_pages(&self) -> usize {
        self.idle.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pages lent out, kept and taken back: a run of cached pages is found by the tokens it was written for, shared pages go back when their last holder does, and the oldest idle page is taken first, a chain's end before its start.
    #[test]
    fn pager() {
        let mut g = Pager::new(6);
        assert_eq!((g.pages(), g.available()), (6, 6));
        let toks: Vec<u32> = (0..48).collect();
        // a sequence of three full pages and a partial one, published as it goes
        let held: Vec<i32> = (0..4).map(|_| g.alloc().unwrap()).collect();
        let mut prev = 0;
        for (i, c) in toks.chunks(PAGE).enumerate() {
            prev = g.publish(held[i], prev, c);
        }
        assert_eq!(g.available(), 2);
        // a second prompt that begins with the same 40 tokens (two full pages, then others) finds two pages and holds them with the first
        let mut other: Vec<u32> = toks[..40].to_vec();
        other.extend(100..110);
        let got = g.lookup(&other, 10);
        assert_eq!(got, held[..2].to_vec());
        // the first ends: its partial page goes back, the cached ones that the second still holds stay held, the third waits
        let freed = g.release(&held);
        assert_eq!(freed, vec![held[3]]);
        assert_eq!(g.available(), 2 + 1 + 1);
        // the second ends too: all three cached pages wait; the oldest idle is the chain's last
        assert!(g.release(&got).is_empty());
        assert_eq!(g.available(), 6);
        let p1 = g.alloc().unwrap();
        let p2 = g.alloc().unwrap();
        let p3 = g.alloc().unwrap();
        assert!(!held[..3].contains(&p1) && !held[..3].contains(&p2) && !held[..3].contains(&p3), "the free pages come first");
        // now the cached ones: the chain's last page first
        assert_eq!(g.alloc(), Some(held[2]));
        assert_eq!(g.lookup(&toks, 10), held[..2].to_vec());
        assert_eq!(g.alloc(), None, "the two cached pages that are held again are not for the taking");
        // the same tokens published on another page are not cached twice
        g.release(&held[..2]);
        let dup = g.release(&[p1]);
        assert_eq!(dup, vec![p1]);
        let q = g.alloc().unwrap();
        assert_eq!(q, p1);
        let k = g.publish(q, 0, &toks[..16]);
        assert_eq!(k, chain_key(0, &toks[..16]));
        assert_eq!(g.lookup(&toks, 10), vec![held[0], held[1]], "the first page of the run stays the cached one");
    }

    /// A page found lost or heavy is out of the cache and is never kept for a later prompt: a held page is cleared when its holder goes, an idle one at once, and every idle cached page can be dropped.
    #[test]
    fn lost_and_heavy_pages_are_not_kept() {
        let mut g = Pager::new(8);
        let toks: Vec<u32> = (0..64).collect();
        let held: Vec<i32> = (0..4).map(|_| g.alloc().unwrap()).collect();
        let mut prev = 0;
        for (i, c) in toks.chunks(PAGE).enumerate() {
            prev = g.publish(held[i], prev, c);
        }
        // the second page is found heavy while its sequence still holds it: out of the cache, still held
        assert!(!g.evict(held[1], true));
        let found = g.lookup(&toks, 4);
        assert_eq!(found, vec![held[0]], "the run of cached pages ends where a page left the cache");
        g.release(&found);
        // a heavy page is not published again
        g.publish(held[1], chain_key(0, &toks[..16]), &toks[16..32]);
        assert_eq!(g.lookup(&toks, 4), vec![held[0]]);
        g.release(&[held[0]]);
        // its holder gives it back: it goes back to be cleared, the others wait in the cache
        let freed = g.release(&held);
        assert_eq!(freed, vec![held[1]]);
        assert_eq!((g.idle_pages(), g.available()), (3, 8));
        // an idle cached page found lost is free at once, and returned for clearing
        assert!(g.evict(held[2], true));
        assert_eq!((g.idle_pages(), g.available()), (2, 8));
        let found = g.lookup(&toks, 4);
        assert_eq!(found, vec![held[0]]);
        g.release(&found);
        // dropping the cache frees every idle cached page
        let mut dropped = g.drop_cache();
        dropped.sort_unstable();
        let mut want = vec![held[0], held[3]];
        want.sort_unstable();
        assert_eq!(dropped, want);
        assert_eq!((g.idle_pages(), g.available()), (0, 8));
        assert!(g.lookup(&toks, 4).is_empty());
        // a page that is free again is a page like any other: published and found
        let p = g.alloc().unwrap();
        g.publish(p, 0, &toks[..16]);
        g.release(&[p]);
        assert_eq!(g.lookup(&toks, 4), vec![p]);
    }

    /// The tables' sidecar as tools/npz_to_tables.py writes it, read back: the windows and the figures that size the pools.
    #[test]
    fn tables_file() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-kv-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("kv-tables.safetensors");
        let (k, v): (Vec<u8>, Vec<u8>) = ((0..2 * 8 * 128).map(|i| (i % 120) as u8).collect(), (0..2 * 8 * 128).map(|i| ((i + 7) % 120) as u8).collect());
        let meta = r#"{"positions":4096,"model":"m","version":8,"store":{"bay":15,"worst_chunks_per_layer_leaf":2.8611,"hard":0}}"#;
        crate::pack::write_safetensors(
            &p,
            &[("format", "glyd-kv-tables-v0".to_string()), ("meta", meta.to_string())],
            &[("k", "U8", vec![2, 8, 128], &k[..]), ("v", "U8", vec![2, 8, 128], &v[..])],
        )
        .unwrap();
        assert!(KvTables::is_current(&p));
        let t = KvTables::load(&p).unwrap();
        assert_eq!((t.layers, t.heads, t.channels, t.positions, t.arena, t.heavy_tokens), (2, 8, 128, 4096, 15, 0));
        let older = dir.join("older.safetensors");
        crate::pack::write_safetensors(
            &older,
            &[("format", "glyd-kv-tables-v0".to_string()), ("meta", meta.replace("\"version\":8", "\"version\":7"))],
            &[("k", "U8", vec![2, 8, 128], &t.k[..]), ("v", "U8", vec![2, 8, 128], &t.v[..])],
        )
        .unwrap();
        assert!(!KvTables::is_current(&older) && !KvTables::is_current(&dir.join("none.safetensors")));
        assert_eq!(t.worst_chunks, Some(2.8611));
        assert_eq!((t.k, t.v), (k, v));
        assert!(KvTables::load(&dir.join("none.safetensors")).err().unwrap().to_string().contains("the first run makes it"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
