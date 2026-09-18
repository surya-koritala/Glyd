//! Table ANS (FSE-style) for the sequence code streams: at most 64
//! symbols, table log 10. The decoder reads a forward LSB-first stream:
//! the initial state (TL bits), then per symbol the table's `nbits` bits.
//! The encoder therefore processes symbols last-to-first and emits the
//! chunks in reverse, so no bit-reversal is needed anywhere.
use crate::bits::{BitReader, BitWriter};

pub const TL: u32 = 10;
pub const L: usize = 1 << TL;
pub const MAX_SYMBOLS: usize = 64;

/// Scale a histogram to counts summing to L. Present symbols get >= 1.
pub fn normalize(hist: &[u32], n_symbols: usize) -> Vec<u16> {
    assert!(n_symbols <= MAX_SYMBOLS && hist.len() >= n_symbols);
    let total: u64 = hist[..n_symbols].iter().map(|&h| h as u64).sum();
    let mut counts = vec![0u16; n_symbols];
    if total == 0 {
        counts[0] = L as u16;
        return counts;
    }
    let mut sum = 0usize;
    let mut largest = 0usize;
    for s in 0..n_symbols {
        let h = hist[s] as u64;
        if h == 0 {
            continue;
        }
        let mut c = ((h * L as u64) / total) as usize;
        if c == 0 {
            c = 1;
        }
        counts[s] = c as u16;
        sum += c;
        if hist[s] > hist[largest] {
            largest = s;
        }
    }
    // Put the rounding error on the most frequent symbol.
    let cl = counts[largest] as isize + (L as isize - sum as isize);
    assert!(cl >= 1, "normalization underflow");
    counts[largest] = cl as u16;
    counts
}

fn check(counts: &[u16]) -> bool {
    counts.len() >= 1 && counts.len() <= MAX_SYMBOLS && counts.iter().map(|&c| c as usize).sum::<usize>() == L
}

/// zstd's spread: symbol occurrences placed at stride (5/8)L + 3.
fn spread(counts: &[u16]) -> Vec<u8> {
    let step = (L >> 1) + (L >> 3) + 3;
    let mask = L - 1;
    let mut table = vec![0u8; L];
    let mut pos = 0usize;
    for (s, &c) in counts.iter().enumerate() {
        for _ in 0..c {
            table[pos] = s as u8;
            pos = (pos + step) & mask;
        }
    }
    debug_assert_eq!(pos, 0);
    table
}

fn highbit(x: u32) -> u32 {
    31 - x.leading_zeros()
}

#[derive(Clone, Copy)]
pub struct DecodeEntry {
    pub sym: u8,
    pub nbits: u8,
    pub base: u16,
}

pub struct DecodeTable {
    pub entries: Vec<DecodeEntry>,
}

impl DecodeTable {
    pub fn build(counts: &[u16]) -> Option<DecodeTable> {
        if !check(counts) {
            return None;
        }
        let sp = spread(counts);
        let mut next: Vec<u32> = counts.iter().map(|&c| c as u32).collect();
        let mut entries = vec![DecodeEntry { sym: 0, nbits: 0, base: 0 }; L];
        for i in 0..L {
            let s = sp[i] as usize;
            let x = next[s];
            next[s] += 1;
            let nbits = TL - highbit(x);
            entries[i] = DecodeEntry { sym: s as u8, nbits: nbits as u8, base: ((x << nbits) - L as u32) as u16 };
        }
        Some(DecodeTable { entries })
    }
}

pub struct EncodeTable {
    /// Indexed by cumulative position; holds the next state (L..2L).
    state_table: Vec<u16>,
    /// Per symbol: (delta_nbits, delta_find_state).
    sym: Vec<(u32, i32)>,
}

impl EncodeTable {
    pub fn build(counts: &[u16]) -> Option<EncodeTable> {
        if !check(counts) {
            return None;
        }
        let sp = spread(counts);
        let mut cumul = vec![0u32; counts.len() + 1];
        for s in 0..counts.len() {
            cumul[s + 1] = cumul[s] + counts[s] as u32;
        }
        let mut fill = cumul.clone();
        let mut state_table = vec![0u16; L];
        for i in 0..L {
            let s = sp[i] as usize;
            state_table[fill[s] as usize] = (L + i) as u16;
            fill[s] += 1;
        }
        let mut sym = Vec::with_capacity(counts.len());
        for s in 0..counts.len() {
            let c = counts[s] as u32;
            if c == 0 {
                sym.push((0, 0));
                continue;
            }
            let max_bits_out = TL - highbit(c);
            let min_state_plus = c << max_bits_out;
            let delta_nbits = (max_bits_out << 16).wrapping_sub(min_state_plus);
            let delta_find_state = cumul[s] as i32 - c as i32;
            sym.push((delta_nbits, delta_find_state));
        }
        Some(EncodeTable { state_table, sym })
    }
}

pub struct Encoder<'a> {
    t: &'a EncodeTable,
    syms: Vec<u8>,
}

impl<'a> Encoder<'a> {
    pub fn new(t: &'a EncodeTable) -> Self {
        Encoder { t, syms: Vec::new() }
    }
    #[inline(always)]
    pub fn push(&mut self, sym: u8) {
        self.syms.push(sym);
    }
    /// Encode last-to-first, then write the chunks in decode order.
    pub fn finish(self) -> Vec<u8> {
        let mut chunks: Vec<(u32, u8)> = Vec::with_capacity(self.syms.len());
        // Initial encoder state: any value in [L, 2L); use the first
        // symbol's smallest legal state so the decoder's first state is
        // well defined.
        let mut state: u32 = L as u32;
        for &s in self.syms.iter().rev() {
            let (delta_nbits, delta_find) = self.t.sym[s as usize];
            let nbits_out = (state.wrapping_add(delta_nbits)) >> 16;
            let low = state & ((1u32 << nbits_out) - 1);
            chunks.push((low, nbits_out as u8));
            let idx = ((state >> nbits_out) as i32 + delta_find) as usize;
            state = self.t.state_table[idx] as u32;
        }
        let mut w = BitWriter::new();
        w.put((state - L as u32) as u64, TL);
        for &(v, n) in chunks.iter().rev() {
            if n > 0 {
                w.put(v as u64, n as u32);
            }
        }
        w.finish()
    }
}

pub struct Decoder<'a> {
    t: &'a [DecodeEntry],
    r: BitReader,
    state: u32,
}

impl<'a> Decoder<'a> {
    pub fn new(t: &'a DecodeTable, stream: &[u8]) -> Self {
        let mut r = BitReader::new(stream);
        let state = r.get(TL) as u32;
        Decoder { t: t.entries.as_slice(), r, state }
    }
    #[inline(always)]
    pub fn next(&mut self) -> u8 {
        let e = self.t[self.state as usize];
        self.r.refill();
        let bits = self.r.peek(e.nbits as u32) as u32;
        self.r.consume(e.nbits as u32);
        self.state = e.base as u32 + bits;
        e.sym
    }
    pub fn overrun(&self) -> bool {
        self.r.overrun()
    }
}
