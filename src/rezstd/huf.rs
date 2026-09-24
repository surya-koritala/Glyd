//! Huffman coding of the literals as zstd 1.5.5's `huf_compress.c`
//! does it: the bucket sort with its quicksort on the large buckets
//! (the tie order decides the tree), the tree, the height limit
//! (`HUF_setMaxHeight`), the canonical codes, the weights written
//! through FSE or as nibbles, the estimates that choose between a new
//! table and the previous block's, and the one- and four-stream
//! layouts (`HUF_compress_internal`).

use super::fse::{self, highbit, BitWriter, Repeat};

pub const TABLELOG_MAX: u32 = 12;
pub const SYMBOLVALUE_MAX: usize = 255;

/// `HUF_CElt` table: a code length and value per symbol; symbols past
/// the block's alphabet stay at zero, which `validate` relies on.
#[derive(Clone, Copy)]
pub struct HufTable {
    pub log: u32,
    pub nbits: [u8; 256],
    pub val: [u16; 256],
}

impl HufTable {
    pub const fn blank() -> HufTable {
        HufTable { log: 0, nbits: [0; 256], val: [0; 256] }
    }
}

#[derive(Clone, Copy, Default)]
struct Node {
    count: u32,
    parent: u16,
    byte: u8,
    nbits: u8,
}

const RANK_POSITION_TABLE_SIZE: usize = 192;
const RANK_POSITION_MAX_COUNT_LOG: u32 = 32;
const RANK_POSITION_LOG_BUCKETS_BEGIN: u32 = (RANK_POSITION_TABLE_SIZE as u32 - 1) - RANK_POSITION_MAX_COUNT_LOG - 1;

/// `HUF_getIndex`: distinct buckets for small counts, log2 buckets above.
fn bucket(count: u32) -> usize {
    let cutoff = RANK_POSITION_LOG_BUCKETS_BEGIN + highbit(RANK_POSITION_LOG_BUCKETS_BEGIN);
    (if count < cutoff { count } else { highbit(count) + RANK_POSITION_LOG_BUCKETS_BEGIN }) as usize
}

/// `HUF_insertionSort`: descending by count.
fn insertion_sort(a: &mut [Node]) {
    for i in 1..a.len() {
        let key = a[i];
        let mut j = i as isize - 1;
        while j >= 0 && a[j as usize].count < key.count {
            a[j as usize + 1] = a[j as usize];
            j -= 1;
        }
        a[(j + 1) as usize] = key;
    }
}

/// `HUF_quickSortPartition`: the rightmost element as pivot.
fn partition(a: &mut [Node], low: isize, high: isize) -> isize {
    let pivot = a[high as usize].count;
    let mut i = low - 1;
    for j in low..high {
        if a[j as usize].count > pivot {
            i += 1;
            a.swap(i as usize, j as usize);
        }
    }
    a.swap((i + 1) as usize, high as usize);
    i + 1
}

/// `HUF_simpleQuickSort`: descending, recursing on the smaller side.
fn quick_sort(a: &mut [Node], mut low: isize, mut high: isize) {
    if high - low < 8 {
        if high > low {
            insertion_sort(&mut a[low as usize..=high as usize]);
        }
        return;
    }
    while low < high {
        let idx = partition(a, low, high);
        if idx - low < high - idx {
            quick_sort(a, low, idx - 1);
            low = idx + 1;
        } else {
            quick_sort(a, idx + 1, high);
            high = idx - 1;
        }
    }
}

/// `HUF_sort`: `nodes[..=max_symbol]` by decreasing count.
fn sort(nodes: &mut [Node], count: &[u32], max_symbol: usize) {
    let mut base = [0u16; RANK_POSITION_TABLE_SIZE];
    let mut curr = [0u16; RANK_POSITION_TABLE_SIZE];
    for &c in &count[..=max_symbol] {
        base[bucket(c)] += 1;
    }
    for n in (1..RANK_POSITION_TABLE_SIZE).rev() {
        base[n - 1] += base[n];
        curr[n - 1] = base[n - 1];
    }
    for (n, &c) in count[..=max_symbol].iter().enumerate() {
        let r = bucket(c) + 1;
        let pos = curr[r] as usize;
        curr[r] += 1;
        nodes[pos].count = c;
        nodes[pos].byte = n as u8;
    }
    let cutoff = (RANK_POSITION_LOG_BUCKETS_BEGIN + highbit(RANK_POSITION_LOG_BUCKETS_BEGIN)) as usize;
    for n in cutoff..RANK_POSITION_TABLE_SIZE - 1 {
        let size = (curr[n] - base[n]) as isize;
        if size > 1 {
            quick_sort(&mut nodes[base[n] as usize..], 0, size - 1);
        }
    }
}

const STARTNODE: usize = SYMBOLVALUE_MAX + 1;

/// `HUF_buildTree` on `nodes`, where `nodes[i + 1]` is the reference's
/// `huffNode[i]` and `nodes[0]` its barrier; returns the last non-null
/// rank.
fn build_tree(nodes: &mut [Node], max_symbol: usize) -> usize {
    let h = 1usize; // huffNode = huffNode0 + 1
    let mut non_null_rank = max_symbol;
    while nodes[h + non_null_rank].count == 0 {
        non_null_rank -= 1;
    }
    let mut low_s = non_null_rank as isize;
    let mut node_nb = STARTNODE;
    let node_root = node_nb + low_s as usize - 1;
    let mut low_n = node_nb;
    nodes[h + node_nb].count = nodes[h + low_s as usize].count + nodes[h + low_s as usize - 1].count;
    nodes[h + low_s as usize].parent = node_nb as u16;
    nodes[h + low_s as usize - 1].parent = node_nb as u16;
    node_nb += 1;
    low_s -= 2;
    for n in node_nb..=node_root {
        nodes[h + n].count = 1 << 30;
    }
    nodes[0].count = 1 << 31; // fake entry, strong barrier
    while node_nb <= node_root {
        let pick = |nodes: &[Node], low_s: &mut isize, low_n: &mut usize| -> usize {
            if nodes[(h as isize + *low_s) as usize].count < nodes[h + *low_n].count {
                let r = *low_s;
                *low_s -= 1;
                (h as isize + r) as usize
            } else {
                let r = *low_n;
                *low_n += 1;
                h + r
            }
        };
        let n1 = pick(nodes, &mut low_s, &mut low_n);
        let n2 = pick(nodes, &mut low_s, &mut low_n);
        nodes[h + node_nb].count = nodes[n1].count + nodes[n2].count;
        nodes[n1].parent = node_nb as u16;
        nodes[n2].parent = node_nb as u16;
        node_nb += 1;
    }
    nodes[h + node_root].nbits = 0;
    for n in (STARTNODE..node_root).rev() {
        nodes[h + n].nbits = nodes[h + nodes[h + n].parent as usize].nbits + 1;
    }
    for n in 0..=non_null_rank {
        nodes[h + n].nbits = nodes[h + nodes[h + n].parent as usize].nbits + 1;
    }
    non_null_rank
}

/// `HUF_setMaxHeight`: the tree in `nodes` (offset as in `build_tree`)
/// limited to `target` bits; returns the largest length after.
fn set_max_height(nodes: &mut [Node], last_non_null: usize, target: u32) -> u32 {
    let h = 1isize;
    let at = |i: isize| (h + i) as usize;
    let largest_bits = nodes[at(last_non_null as isize)].nbits as u32;
    if largest_bits <= target {
        return largest_bits;
    }
    let mut total_cost: i32 = 0;
    let base_cost = 1i32 << (largest_bits - target);
    let mut n = last_non_null as isize;
    while nodes[at(n)].nbits as u32 > target {
        total_cost += base_cost - (1 << (largest_bits - nodes[at(n)].nbits as u32));
        nodes[at(n)].nbits = target as u8;
        n -= 1;
    }
    while nodes[at(n)].nbits as u32 == target {
        n -= 1;
    }
    total_cost >>= largest_bits - target;
    const NO_SYMBOL: u32 = 0xF0F0F0F0;
    let mut rank_last = [NO_SYMBOL; TABLELOG_MAX as usize + 2];
    {
        let mut current_nb_bits = target;
        let mut pos = n;
        while pos >= 0 {
            if nodes[at(pos)].nbits as u32 >= current_nb_bits {
                pos -= 1;
                continue;
            }
            current_nb_bits = nodes[at(pos)].nbits as u32;
            rank_last[(target - current_nb_bits) as usize] = pos as u32;
            pos -= 1;
        }
    }
    while total_cost > 0 {
        let mut nb_bits_to_decrease = highbit(total_cost as u32) + 1;
        while nb_bits_to_decrease > 1 {
            let high_pos = rank_last[nb_bits_to_decrease as usize];
            let low_pos = rank_last[nb_bits_to_decrease as usize - 1];
            if high_pos == NO_SYMBOL {
                nb_bits_to_decrease -= 1;
                continue;
            }
            if low_pos == NO_SYMBOL {
                break;
            }
            let high_total = nodes[at(high_pos as isize)].count;
            let low_total = 2 * nodes[at(low_pos as isize)].count;
            if high_total <= low_total {
                break;
            }
            nb_bits_to_decrease -= 1;
        }
        while nb_bits_to_decrease <= TABLELOG_MAX && rank_last[nb_bits_to_decrease as usize] == NO_SYMBOL {
            nb_bits_to_decrease += 1;
        }
        total_cost -= 1 << (nb_bits_to_decrease - 1);
        let d = nb_bits_to_decrease as usize;
        nodes[at(rank_last[d] as isize)].nbits += 1;
        if rank_last[d - 1] == NO_SYMBOL {
            rank_last[d - 1] = rank_last[d];
        }
        if rank_last[d] == 0 {
            rank_last[d] = NO_SYMBOL;
        } else {
            rank_last[d] -= 1;
            if nodes[at(rank_last[d] as isize)].nbits as u32 != target - nb_bits_to_decrease {
                rank_last[d] = NO_SYMBOL;
            }
        }
    }
    while total_cost < 0 {
        if rank_last[1] == NO_SYMBOL {
            while nodes[at(n)].nbits as u32 == target {
                n -= 1;
            }
            nodes[at(n + 1)].nbits -= 1;
            rank_last[1] = (n + 1) as u32;
            total_cost += 1;
            continue;
        }
        nodes[at(rank_last[1] as isize + 1)].nbits -= 1;
        rank_last[1] += 1;
        total_cost += 1;
    }
    target
}

/// `HUF_buildCTable_wksp`: the table for `count`, at most `max_nbits`
/// deep; None where the reference errors (a tree deeper than 12).
pub fn build_ctable(count: &[u32], max_symbol: usize, max_nbits: u32) -> Option<HufTable> {
    let mut nodes = [Node::default(); 2 * (SYMBOLVALUE_MAX + 1)];
    sort(&mut nodes[1..], count, max_symbol);
    let non_null_rank = build_tree(&mut nodes, max_symbol);
    let max_nbits = set_max_height(&mut nodes, non_null_rank, max_nbits);
    if max_nbits > TABLELOG_MAX {
        return None;
    }
    // HUF_buildCTableFromTree.
    let huff = &nodes[1..];
    let mut nb_per_rank = [0u16; TABLELOG_MAX as usize + 1];
    let mut val_per_rank = [0u16; TABLELOG_MAX as usize + 1];
    for node in &huff[..=non_null_rank] {
        nb_per_rank[node.nbits as usize] += 1;
    }
    let mut min = 0u16;
    for n in (1..=max_nbits as usize).rev() {
        val_per_rank[n] = min;
        min += nb_per_rank[n];
        min >>= 1;
    }
    let mut t = HufTable::blank();
    t.log = max_nbits;
    for node in &huff[..=max_symbol] {
        t.nbits[node.byte as usize] = node.nbits;
    }
    for s in 0..=max_symbol {
        let n = t.nbits[s] as usize;
        if n > 0 {
            t.val[s] = val_per_rank[n];
        }
        val_per_rank[n] += 1;
    }
    Some(t)
}

/// `HUF_estimateCompressedSize`: bytes for `count` under `t`.
fn estimate(t: &HufTable, count: &[u32], max_symbol: usize) -> usize {
    let bits: u64 = (0..=max_symbol).map(|s| t.nbits[s] as u64 * count[s] as u64).sum();
    (bits >> 3) as usize
}

/// `HUF_validateCTable`: every symbol of `count` has a code in `t`.
fn validate(t: &HufTable, count: &[u32], max_symbol: usize) -> bool {
    !(0..=max_symbol).any(|s| count[s] != 0 && t.nbits[s] == 0)
}

/// `HUF_compressWeights`: the weights through FSE; a result of one
/// byte stands for the reference's "1" (a single weight value), an
/// empty one for its 0 (not compressible); None for its errors.
fn compress_weights(cap: usize, weights: &[u8]) -> Option<Vec<u8>> {
    let n = weights.len();
    if n <= 1 {
        return Some(Vec::new());
    }
    let mut count = [0u32; TABLELOG_MAX as usize + 1];
    let (max_symbol, max_count) = fse::histogram(&mut count, TABLELOG_MAX as usize, weights);
    if max_count as usize == n {
        return Some(vec![0]);
    }
    if max_count == 1 {
        return Some(Vec::new());
    }
    let table_log = fse::optimal_table_log(6, n, max_symbol as u32, 2);
    let mut norm = [0i16; TABLELOG_MAX as usize + 1];
    fse::normalize_count(&mut norm, table_log, &count, n, max_symbol, false)?;
    let mut out = fse::write_ncount(cap, &norm, max_symbol, table_log)?;
    let table = fse::CTable::build(&norm, max_symbol, table_log);
    match fse::compress(cap - out.len(), weights, &table) {
        Some(bits) => out.extend_from_slice(&bits),
        None => return Some(Vec::new()),
    }
    Some(out)
}

/// `HUF_writeCTable_wksp`: the table description (weights FSE-coded
/// or as nibbles); None for the reference's errors.
fn write_ctable(cap: usize, t: &HufTable, max_symbol: usize, huff_log: u32) -> Option<Vec<u8>> {
    let mut bits_to_weight = [0u8; TABLELOG_MAX as usize + 1];
    for (n, w) in bits_to_weight.iter_mut().enumerate().skip(1).take(huff_log as usize) {
        *w = (huff_log + 1 - n as u32) as u8;
    }
    let mut weights = [0u8; SYMBOLVALUE_MAX + 1];
    for n in 0..max_symbol {
        weights[n] = bits_to_weight[t.nbits[n] as usize];
    }
    if cap < 1 {
        return None;
    }
    let h = compress_weights(cap - 1, &weights[..max_symbol])?;
    if h.len() > 1 && h.len() < max_symbol / 2 {
        let mut out = vec![h.len() as u8];
        out.extend_from_slice(&h);
        return Some(out);
    }
    // Raw weights as nibbles.
    if max_symbol > 128 || max_symbol.div_ceil(2) + 1 > cap {
        return None;
    }
    let mut out = vec![(128 + max_symbol - 1) as u8];
    for n in (0..max_symbol).step_by(2) {
        out.push((weights[n] << 4) + weights[n + 1]);
    }
    Some(out)
}

/// `HUF_compress1X_usingCTable`: one stream, symbols from the last;
/// None where the reference returns 0.
fn compress_1x(cap: usize, src: &[u8], t: &HufTable) -> Option<Vec<u8>> {
    let mut w = BitWriter::new(cap)?;
    for &b in src.iter().rev() {
        w.add(t.val[b as usize] as u64, t.nbits[b as usize] as u32);
    }
    w.close()
}

/// `HUF_compress4X_usingCTable`: four streams behind a jump table.
fn compress_4x(cap: usize, src: &[u8], t: &HufTable) -> Option<Vec<u8>> {
    if cap < 6 + 1 + 1 + 1 + 8 || src.len() < 12 {
        return None;
    }
    let segment = src.len().div_ceil(4);
    let mut out = vec![0u8; 6];
    for i in 0..4 {
        let part = if i < 3 { &src[i * segment..(i + 1) * segment] } else { &src[3 * segment..] };
        let c = compress_1x(cap - out.len(), part, t)?;
        if c.is_empty() || c.len() > 65535 {
            return None;
        }
        if i < 3 {
            out[2 * i..2 * i + 2].copy_from_slice(&(c.len() as u16).to_le_bytes());
        }
        out.extend_from_slice(&c);
    }
    Some(out)
}

/// `HUF_compressCTable_internal`: the streams appended to `out` (the
/// table description, if any); None where the reference returns 0.
fn compress_streams(mut out: Vec<u8>, cap: usize, src: &[u8], four: bool, t: &HufTable) -> Option<Vec<u8>> {
    let c = if four { compress_4x(cap - out.len(), src, t)? } else { compress_1x(cap - out.len(), src, t)? };
    out.extend_from_slice(&c);
    if out.len() >= src.len() - 1 {
        return None;
    }
    Some(out)
}

/// `HUF_compress_internal` with `HUF_SYMBOLVALUE_MAX` and `LitHufLog`:
/// the coded literals in `cap` bytes, or an empty result where the
/// reference returns 0 (kept raw), or one byte for its RLE signal;
/// `Err` for its error codes (the caller keeps the literals raw too).
/// `table` is the previous block's table, replaced when a new one is
/// written; `repeat` says whether it may be reused and comes back as
/// `Repeat::None` when it was not.
pub fn compress(cap: usize, src: &[u8], four: bool, table: &mut HufTable, repeat: &mut Repeat, prefer_repeat: bool, suspect_uncompressible: bool) -> Result<Vec<u8>, ()> {
    const HUFF_LOG: u32 = 11;
    let n = src.len();
    if n == 0 || cap == 0 {
        return Ok(Vec::new());
    }
    let streams = |out: Vec<u8>, t: &HufTable| Ok(compress_streams(out, cap, src, four, t).unwrap_or_default());
    if prefer_repeat && *repeat == Repeat::Valid {
        return streams(Vec::new(), table);
    }
    let mut count = [0u32; SYMBOLVALUE_MAX + 1];
    const SAMPLE: usize = 4096;
    if suspect_uncompressible && n >= SAMPLE * 10 {
        let (_, largest_begin) = fse::histogram(&mut count, SYMBOLVALUE_MAX, &src[..SAMPLE]);
        let (_, largest_end) = fse::histogram(&mut count, SYMBOLVALUE_MAX, &src[n - SAMPLE..]);
        if (largest_begin + largest_end) as usize <= ((2 * SAMPLE) >> 7) + 4 {
            return Ok(Vec::new());
        }
    }
    let (max_symbol, largest) = fse::histogram(&mut count, SYMBOLVALUE_MAX, src);
    if largest as usize == n {
        return Ok(vec![src[0]]);
    }
    if largest as usize <= (n >> 7) + 4 {
        return Ok(Vec::new());
    }
    if *repeat == Repeat::Check && !validate(table, &count, max_symbol) {
        *repeat = Repeat::None;
    }
    if prefer_repeat && *repeat != Repeat::None {
        return streams(Vec::new(), table);
    }
    let huff_log = fse::optimal_table_log(HUFF_LOG, n, max_symbol as u32, 1);
    let new = build_ctable(&count, max_symbol, huff_log).ok_or(())?;
    let header = write_ctable(cap, &new, max_symbol, new.log).ok_or(())?;
    if *repeat != Repeat::None {
        let old_size = estimate(table, &count, max_symbol);
        let new_size = estimate(&new, &count, max_symbol);
        if old_size <= header.len() + new_size || header.len() + 12 >= n {
            return streams(Vec::new(), table);
        }
    }
    if header.len() + 12 >= n {
        return Ok(Vec::new());
    }
    *repeat = Repeat::None;
    *table = new;
    streams(header, &new)
}
