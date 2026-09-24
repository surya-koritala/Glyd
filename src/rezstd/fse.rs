//! Finite-state entropy coding as zstd 1.5.5's `fse_compress.c` does
//! it: the histogram, the normalized counts with their rounding rule
//! and the second-chance normalization, the table description
//! (`FSE_writeNCount`), the coding table (`FSE_buildCTable_wksp`) and
//! the two-state bitstream (`FSE_compress_usingCTable`). The forward
//! bit writer (`BIT_CStream_t`) lives here too: the Huffman and the
//! sequence bitstreams use it as well, and its capacity rule (a stream
//! that fills its buffer to the last eight bytes is dropped) is part of
//! what the reference writes.

/// `ZSTD_highbit32`: the index of the highest set bit; `v` is never 0
/// where the reference calls it.
#[inline]
pub fn highbit(v: u32) -> u32 {
    debug_assert!(v != 0);
    31 - v.leading_zeros()
}

pub const MIN_TABLELOG: u32 = 5;
pub const MAX_TABLELOG: u32 = 12;
pub const DEFAULT_TABLELOG: u32 = 11;
/// The largest alphabet an FSE table here codes (zstd's `MaxSeq`, 52,
/// for the match lengths; the Huffman weights use 13 symbols).
pub const MAX_SYMBOLS: usize = 53;
/// The largest table log of the tables here (zstd's `MaxFSELog`).
const MAX_TABLE_SIZE: usize = 1 << 9;

/// Whether an entropy table from the previous block may be reused
/// (`FSE_repeat` / `HUF_repeat`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Repeat {
    /// No usable previous table.
    None,
    /// A previous table exists but must be checked against the block.
    Check,
    /// A previous table known to cover the block (dictionaries only).
    Valid,
}

/// `BIT_CStream_t`: bits are added at increasing positions and whole
/// bytes flushed; the reader takes them from the end.
pub struct BitWriter {
    acc: u64,
    bits: u32,
    out: Vec<u8>,
    cap: usize,
}

impl BitWriter {
    /// `BIT_initCStream`: None when the destination cannot hold a
    /// register (`dstCapacity <= 8`).
    pub fn new(cap: usize) -> Option<BitWriter> {
        if cap <= 8 {
            return None;
        }
        Some(BitWriter { acc: 0, bits: 0, out: Vec::new(), cap })
    }

    /// `BIT_addBits`: the low `nbits` bits of `value` (at most 31).
    #[inline]
    pub fn add(&mut self, value: u64, nbits: u32) {
        self.acc |= (value & ((1u64 << nbits) - 1)) << self.bits;
        self.bits += nbits;
        while self.bits >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.bits -= 8;
        }
    }

    /// `BIT_closeCStream`: the end mark, then the bytes; None when the
    /// stream reached the last eight bytes of its buffer (the reference
    /// reports "not storable" and its callers give up on the stream).
    pub fn close(mut self) -> Option<Vec<u8>> {
        self.add(1, 1);
        if self.out.len() >= self.cap - 8 {
            return None;
        }
        if self.bits > 0 {
            self.out.push(self.acc as u8);
        }
        Some(self.out)
    }
}

/// `HIST_count_simple`: the histogram of `src` over symbols up to
/// `max_in`; returns the highest symbol present and the largest count.
pub fn histogram(count: &mut [u32], max_in: usize, src: &[u8]) -> (usize, u32) {
    for c in count[..=max_in].iter_mut() {
        *c = 0;
    }
    if src.is_empty() {
        return (0, 0);
    }
    for &b in src {
        count[b as usize] += 1;
    }
    let mut max = max_in;
    while count[max] == 0 {
        max -= 1;
    }
    (max, *count[..=max].iter().max().unwrap())
}

/// `FSE_minTableLog`.
fn min_table_log(src_size: usize, max_symbol: u32) -> u32 {
    let min_bits_src = highbit(src_size as u32) + 1;
    let min_bits_symbols = highbit(max_symbol) + 2;
    min_bits_src.min(min_bits_symbols)
}

/// `FSE_optimalTableLog_internal`: `minus` is 2 for FSE, 1 for the
/// Huffman depth (`HUF_optimalTableLog` without the optimal-depth flag).
pub fn optimal_table_log(max_table_log: u32, src_size: usize, max_symbol: u32, minus: u32) -> u32 {
    let max_bits_src = highbit((src_size - 1) as u32).wrapping_sub(minus);
    let mut table_log = if max_table_log == 0 { DEFAULT_TABLELOG } else { max_table_log };
    let min_bits = min_table_log(src_size, max_symbol);
    if max_bits_src < table_log {
        table_log = max_bits_src;
    }
    if min_bits > table_log {
        table_log = min_bits;
    }
    table_log.clamp(MIN_TABLELOG, MAX_TABLELOG)
}

/// `FSE_normalizeM2`: the second normalization, used when the first
/// over-distributes; None for the reference's error return.
fn normalize_m2(norm: &mut [i16], table_log: u32, count: &[u32], mut total: u64, max_symbol: usize, low_prob: i16) -> Option<()> {
    const NOT_YET_ASSIGNED: i16 = -2;
    let mut distributed = 0u32;
    let low_threshold = (total >> table_log) as u32;
    let mut low_one = ((total * 3) >> (table_log + 1)) as u32;
    for s in 0..=max_symbol {
        if count[s] == 0 {
            norm[s] = 0;
            continue;
        }
        if count[s] <= low_threshold {
            norm[s] = low_prob;
            distributed += 1;
            total -= count[s] as u64;
            continue;
        }
        if count[s] <= low_one {
            norm[s] = 1;
            distributed += 1;
            total -= count[s] as u64;
            continue;
        }
        norm[s] = NOT_YET_ASSIGNED;
    }
    let mut to_distribute = (1u32 << table_log) - distributed;
    if to_distribute == 0 {
        return Some(());
    }
    if (total / to_distribute as u64) > low_one as u64 {
        // Risk of rounding to zero.
        low_one = ((total * 3) / (to_distribute as u64 * 2)) as u32;
        for s in 0..=max_symbol {
            if norm[s] == NOT_YET_ASSIGNED && count[s] <= low_one {
                norm[s] = 1;
                distributed += 1;
                total -= count[s] as u64;
            }
        }
        to_distribute = (1u32 << table_log) - distributed;
    }
    if distributed as usize == max_symbol + 1 {
        // All values are poor: the remaining points go to the largest.
        let (mut max_v, mut max_c) = (0usize, 0u32);
        for s in 0..=max_symbol {
            if count[s] > max_c {
                max_v = s;
                max_c = count[s];
            }
        }
        norm[max_v] += to_distribute as i16;
        return Some(());
    }
    if total == 0 {
        // Every symbol was low enough for lowOne or lowThreshold.
        let mut s = 0usize;
        while to_distribute > 0 {
            if norm[s] > 0 {
                to_distribute -= 1;
                norm[s] += 1;
            }
            s = (s + 1) % (max_symbol + 1);
        }
        return Some(());
    }
    let v_step_log = 62 - table_log as u64;
    let mid = (1u64 << (v_step_log - 1)) - 1;
    let r_step = (((1u64 << v_step_log) * to_distribute as u64) + mid) / total;
    let mut tmp_total = mid;
    for s in 0..=max_symbol {
        if norm[s] == NOT_YET_ASSIGNED {
            let end = tmp_total + count[s] as u64 * r_step;
            let s_start = (tmp_total >> v_step_log) as u32;
            let s_end = (end >> v_step_log) as u32;
            let weight = s_end - s_start;
            if weight < 1 {
                return None;
            }
            norm[s] = weight as i16;
            tmp_total = end;
        }
    }
    Some(())
}

/// `FSE_normalizeCount`: `norm` over `count` summing to `1 << table_log`
/// (-1 marks a symbol below the low threshold when `use_low_prob`, 1
/// otherwise). Returns 0 when one symbol is all of the input (the
/// caller uses RLE), None for the reference's errors.
pub fn normalize_count(norm: &mut [i16], table_log: u32, count: &[u32], total: usize, max_symbol: usize, use_low_prob: bool) -> Option<u32> {
    let table_log = if table_log == 0 { DEFAULT_TABLELOG } else { table_log };
    if !(MIN_TABLELOG..=MAX_TABLELOG).contains(&table_log) || table_log < min_table_log(total, max_symbol as u32) {
        return None;
    }
    const RTB: [u32; 8] = [0, 473195, 504333, 520860, 550000, 700000, 750000, 830000];
    let low_prob: i16 = if use_low_prob { -1 } else { 1 };
    let scale = 62 - table_log as u64;
    let step = (1u64 << 62) / total as u64;
    let v_step = 1u64 << (scale - 20);
    let mut still_to_distribute = 1i32 << table_log;
    let mut largest = 0usize;
    let mut largest_p: i16 = 0;
    let low_threshold = (total >> table_log) as u32;
    for s in 0..=max_symbol {
        if count[s] as usize == total {
            return Some(0);
        }
        if count[s] == 0 {
            norm[s] = 0;
            continue;
        }
        if count[s] <= low_threshold {
            norm[s] = low_prob;
            still_to_distribute -= 1;
        } else {
            let scaled = count[s] as u64 * step;
            let mut proba = (scaled >> scale) as i16;
            if proba < 8 {
                let rest_to_beat = v_step * RTB[proba as usize] as u64;
                proba += (scaled - ((proba as u64) << scale) > rest_to_beat) as i16;
            }
            if proba > largest_p {
                largest_p = proba;
                largest = s;
            }
            norm[s] = proba;
            still_to_distribute -= proba as i32;
        }
    }
    if -still_to_distribute >= (norm[largest] as i32 >> 1) {
        normalize_m2(norm, table_log, count, total as u64, max_symbol, low_prob)?;
    } else {
        norm[largest] += still_to_distribute as i16;
    }
    Some(table_log)
}

/// `FSE_writeNCount`: the table description; None when it does not
/// fit `cap` (the reference's `dstSize_tooSmall`).
pub fn write_ncount(cap: usize, norm: &[i16], max_symbol: usize, table_log: u32) -> Option<Vec<u8>> {
    if !(MIN_TABLELOG..=MAX_TABLELOG).contains(&table_log) {
        return None;
    }
    let mut out = Vec::new();
    let table_size = 1i32 << table_log;
    let mut bit_stream: u32 = table_log - MIN_TABLELOG;
    let mut bit_count: u32 = 4;
    let mut remaining = table_size + 1; // +1 for extra accuracy
    let mut threshold = table_size;
    let mut nb_bits = table_log + 1;
    let alphabet_size = max_symbol + 1;
    let mut symbol = 0usize;
    let mut previous_is_0 = false;
    // `out > oend - 2`: two bytes must fit before each write.
    let room = |out: &Vec<u8>| out.len() as isize <= cap as isize - 2;
    while symbol < alphabet_size && remaining > 1 {
        if previous_is_0 {
            let mut start = symbol;
            while symbol < alphabet_size && norm[symbol] == 0 {
                symbol += 1;
            }
            if symbol == alphabet_size {
                break;
            }
            while symbol >= start + 24 {
                start += 24;
                bit_stream = bit_stream.wrapping_add(0xFFFFu32 << bit_count);
                if !room(&out) {
                    return None;
                }
                out.extend_from_slice(&(bit_stream as u16).to_le_bytes());
                bit_stream >>= 16;
            }
            while symbol >= start + 3 {
                start += 3;
                bit_stream = bit_stream.wrapping_add(3 << bit_count);
                bit_count += 2;
            }
            bit_stream = bit_stream.wrapping_add(((symbol - start) as u32) << bit_count);
            bit_count += 2;
            if bit_count > 16 {
                if !room(&out) {
                    return None;
                }
                out.extend_from_slice(&(bit_stream as u16).to_le_bytes());
                bit_stream >>= 16;
                bit_count -= 16;
            }
        }
        {
            let mut count = norm[symbol] as i32;
            symbol += 1;
            let max = (2 * threshold - 1) - remaining;
            remaining -= count.abs();
            count += 1; // +1 for extra accuracy
            if count >= threshold {
                count += max;
            }
            bit_stream = bit_stream.wrapping_add((count as u32) << bit_count);
            bit_count += nb_bits;
            bit_count -= (count < max) as u32;
            previous_is_0 = count == 1;
            if remaining < 1 {
                return None;
            }
            while remaining < threshold {
                nb_bits -= 1;
                threshold >>= 1;
            }
        }
        if bit_count > 16 {
            if !room(&out) {
                return None;
            }
            out.extend_from_slice(&(bit_stream as u16).to_le_bytes());
            bit_stream >>= 16;
            bit_count -= 16;
        }
    }
    if remaining != 1 {
        return None;
    }
    if !room(&out) {
        return None;
    }
    let tail = (bit_stream as u16).to_le_bytes();
    out.extend_from_slice(&tail[..bit_count.div_ceil(8) as usize]);
    Some(out)
}

/// `FSE_CTable`: the next-state table and each symbol's transform.
#[derive(Clone, Copy)]
pub struct CTable {
    pub log: u32,
    pub max_symbol: u32,
    next_state: [u16; MAX_TABLE_SIZE],
    delta_find_state: [i32; MAX_SYMBOLS],
    delta_nb_bits: [u32; MAX_SYMBOLS],
}

impl CTable {
    fn blank() -> CTable {
        CTable { log: 0, max_symbol: 0, next_state: [0; MAX_TABLE_SIZE], delta_find_state: [0; MAX_SYMBOLS], delta_nb_bits: [0; MAX_SYMBOLS] }
    }

    /// `FSE_buildCTable_rle`: a table coding one symbol in zero bits.
    pub fn rle(symbol: u8) -> CTable {
        let mut t = CTable::blank();
        t.max_symbol = symbol as u32;
        t
    }

    /// `FSE_buildCTable_wksp` from normalized counts.
    pub fn build(norm: &[i16], max_symbol: usize, table_log: u32) -> CTable {
        let mut t = CTable::blank();
        t.log = table_log;
        t.max_symbol = max_symbol as u32;
        let table_size = 1usize << table_log;
        let table_mask = table_size - 1;
        let step = (table_size >> 1) + (table_size >> 3) + 3;
        let mut cumul = [0u32; MAX_SYMBOLS + 1];
        let mut symbols = vec![0u8; table_size];
        let mut high_threshold = table_size - 1;
        // Symbol start positions; low-probability symbols at the top.
        for u in 1..=max_symbol + 1 {
            if norm[u - 1] == -1 {
                cumul[u] = cumul[u - 1] + 1;
                symbols[high_threshold] = (u - 1) as u8;
                high_threshold -= 1;
            } else {
                cumul[u] = cumul[u - 1] + norm[u - 1] as u32;
            }
        }
        cumul[max_symbol + 1] = table_size as u32 + 1;
        // Spread symbols.
        let mut position = 0usize;
        for s in 0..=max_symbol {
            for _ in 0..norm[s].max(0) {
                symbols[position] = s as u8;
                position = (position + step) & table_mask;
                while position > high_threshold {
                    position = (position + step) & table_mask;
                }
            }
        }
        debug_assert_eq!(position, 0);
        for (u, &s) in symbols.iter().enumerate() {
            t.next_state[cumul[s as usize] as usize] = (table_size + u) as u16;
            cumul[s as usize] += 1;
        }
        // Symbol transformation table.
        let mut total = 0u32;
        for s in 0..=max_symbol {
            match norm[s] {
                0 => t.delta_nb_bits[s] = ((table_log + 1) << 16) - table_size as u32,
                -1 | 1 => {
                    t.delta_nb_bits[s] = (table_log << 16) - table_size as u32;
                    t.delta_find_state[s] = total as i32 - 1;
                    total += 1;
                }
                n => {
                    let n = n as u32;
                    let max_bits_out = table_log - highbit(n - 1);
                    let min_state_plus = n << max_bits_out;
                    t.delta_nb_bits[s] = (max_bits_out << 16) - min_state_plus;
                    t.delta_find_state[s] = total as i32 - n as i32;
                    total += n;
                }
            }
        }
        t
    }
}

/// `FSE_CState_t`: one encoder state over a table.
pub struct CState<'a> {
    table: &'a CTable,
    value: u64,
}

impl<'a> CState<'a> {
    /// `FSE_initCState2`: the state starts on `symbol` at the smallest
    /// value, so that symbol costs nothing.
    pub fn new(table: &'a CTable, symbol: u8) -> CState<'a> {
        let s = symbol as usize;
        let nb_bits_out = (table.delta_nb_bits[s] + (1 << 15)) >> 16;
        let value = (nb_bits_out << 16).wrapping_sub(table.delta_nb_bits[s]) as u64;
        let value = table.next_state[((value >> nb_bits_out) as i64 + table.delta_find_state[s] as i64) as usize] as u64;
        CState { table, value }
    }

    /// `FSE_encodeSymbol`.
    #[inline]
    pub fn encode(&mut self, w: &mut BitWriter, symbol: u8) {
        let s = symbol as usize;
        let nb_bits_out = ((self.value + self.table.delta_nb_bits[s] as u64) >> 16) as u32;
        w.add(self.value, nb_bits_out);
        self.value = self.table.next_state[((self.value >> nb_bits_out) as i64 + self.table.delta_find_state[s] as i64) as usize] as u64;
    }

    /// `FSE_flushCState`.
    pub fn flush(&self, w: &mut BitWriter) {
        w.add(self.value, self.table.log);
    }
}

/// `FSE_compress_usingCTable`: `src` coded with two interleaved states;
/// None where the reference returns 0 (too short, or no room).
pub fn compress(cap: usize, src: &[u8], table: &CTable) -> Option<Vec<u8>> {
    let n = src.len();
    if n <= 2 {
        return None;
    }
    let mut w = BitWriter::new(cap)?;
    let mut ip = n;
    let (mut c1, mut c2);
    if n & 1 != 0 {
        c1 = CState::new(table, src[ip - 1]);
        c2 = CState::new(table, src[ip - 2]);
        c1.encode(&mut w, src[ip - 3]);
        ip -= 3;
    } else {
        c2 = CState::new(table, src[ip - 1]);
        c1 = CState::new(table, src[ip - 2]);
        ip -= 2;
    }
    if (n - 2) & 2 != 0 {
        c2.encode(&mut w, src[ip - 1]);
        c1.encode(&mut w, src[ip - 2]);
        ip -= 2;
    }
    while ip > 0 {
        c2.encode(&mut w, src[ip - 1]);
        c1.encode(&mut w, src[ip - 2]);
        c2.encode(&mut w, src[ip - 3]);
        c1.encode(&mut w, src[ip - 4]);
        ip -= 4;
    }
    c2.flush(&mut w);
    c1.flush(&mut w);
    w.close()
}
