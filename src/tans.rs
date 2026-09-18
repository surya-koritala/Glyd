//! Table ANS (FSE-style) for the sequence code streams: at most 64
//! symbols, table log 10. The decoder reads a forward LSB-first stream:
//! the initial state (TL bits), then per symbol the table's `nbits` bits.
//! The encoder therefore processes symbols last-to-first and emits the
//! chunks in reverse, so no bit-reversal is needed anywhere.
use crate::bits::{split_streams, write_streams, BitReader, BitWriter, FastReader, MAX_PUT};

pub const TL: u32 = 10;
pub const L: usize = 1 << TL;
pub const MAX_SYMBOLS: usize = 64;
/// A chunk packs the state under `nbits << 16`, and `encode8_into`
/// concatenates four chunks per put.
const _: () = assert!(TL < 16 && 4 * TL <= MAX_PUT);

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
    // Holds for any histogram: of the <= 64 present symbols, m got bumped
    // from 0 to 1 (each overshooting L by less than 1) and b = n - m share
    // the rest, largest among them; m + b <= 64 caps m * b <= 1024 = L
    // (AM-GM), which is enough to keep the bumps' overshoot under largest's
    // own share, so cl never drops below 1.
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

/// Decode-table entry packed into a `u32`: `sym | nbits << 8 | base << 16`.
/// A plain scalar measured faster than an equivalent 4-byte
/// `{sym: u8, nbits: u8, base: u16}` struct in the 8-stream hot loop
/// (tans8_speed: 0.873 -> see CHANGELOG-BENCH.md) -- one less live value's
/// worth of register pressure per stream across 8 unrolled streams, even
/// though both are 4 bytes; `base` is < L <= 1024 so it fits in the top 16
/// bits with no truncation.
#[inline(always)]
fn pack(sym: u8, nbits: u8, base: u16) -> u32 {
    sym as u32 | (nbits as u32) << 8 | (base as u32) << 16
}

#[inline(always)]
fn unpack_nbits(e: u32) -> u32 {
    (e >> 8) & 0xFF
}

#[inline(always)]
fn unpack_base(e: u32) -> u32 {
    e >> 16
}

#[inline(always)]
fn unpack_sym(e: u32) -> u8 {
    e as u8
}

pub struct DecodeTable {
    entries: Vec<u32>,
}

impl DecodeTable {
    pub fn build(counts: &[u16]) -> Option<DecodeTable> {
        if !check(counts) {
            return None;
        }
        let sp = spread(counts);
        let mut next: Vec<u32> = counts.iter().map(|&c| c as u32).collect();
        let mut entries = vec![0u32; L];
        for i in 0..L {
            let s = sp[i] as usize;
            let x = next[s];
            next[s] += 1;
            let nbits = TL - highbit(x);
            let base = (x << nbits) - L as u32;
            entries[i] = pack(s as u8, nbits as u8, base as u16);
        }
        Some(DecodeTable { entries })
    }
}

pub struct EncodeTable {
    /// Indexed by cumulative position; holds the next state (L..2L).
    state_table: Vec<u16>,
    /// Per symbol: (delta_nbits, delta_find_state as u32, added wrapping).
    /// 256 entries so a `u8` indexes it with no bounds check; a symbol the
    /// table does not hold (absent, or past `MAX_SYMBOLS`) sits on the
    /// `(0, 0)` placeholder, which sends `step` to `state_table[state]`,
    /// out of bounds and caught by that lookup's check -- the panic a
    /// zero-count symbol always got (see `v7_encode::close`).
    sym: [(u32, u32); 256],
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
        let mut sym = [(0u32, 0u32); 256];
        for s in 0..counts.len() {
            let c = counts[s] as u32;
            if c == 0 {
                continue;
            }
            let max_bits_out = TL - highbit(c);
            let min_state_plus = c << max_bits_out;
            let delta_nbits = (max_bits_out << 16).wrapping_sub(min_state_plus);
            let delta_find_state = cumul[s].wrapping_sub(c);
            sym[s] = (delta_nbits, delta_find_state);
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
        let mut chunks: Vec<u32> = Vec::with_capacity(self.syms.len());
        let mut state: u32 = L as u32;
        for &s in self.syms.iter().rev() {
            let (c, next) = step(self.t, state, s);
            chunks.push(c);
            state = next;
        }
        let mut w = BitWriter::new();
        w.put((state - L as u32) as u64, TL);
        for &c in chunks.iter().rev() {
            let (v, n) = unpack_chunk(c);
            w.put(v as u64, n);
        }
        w.finish()
    }
}

pub struct Decoder<'a> {
    t: &'a [u32],
    r: BitReader<'a>,
    state: u32,
}

impl<'a> Decoder<'a> {
    pub fn new(t: &'a DecodeTable, stream: &'a [u8]) -> Self {
        let mut r = BitReader::new(stream);
        let state = r.get(TL) as u32;
        Decoder { t: t.entries.as_slice(), r, state }
    }
    #[inline(always)]
    pub fn next(&mut self) -> u8 {
        let e = self.t[self.state as usize];
        self.r.refill();
        let nbits = unpack_nbits(e);
        let bits = self.r.peek(nbits) as u32;
        self.r.consume(nbits);
        self.state = unpack_base(e) + bits;
        unpack_sym(e)
    }
    pub fn overrun(&self) -> bool {
        self.r.overrun()
    }
}

pub use crate::bits::STREAMS;
/// 4 symbols/stream per refill: 4 * TL = 40 bits <= the 56-bit refill
/// guarantee (nbits <= TL for every table entry, see `DecodeTable::build`).
const PER_ITER: usize = 4 * STREAMS;

/// One encoder step: the chunk (`state | nbits << 16`: the low `nbits`
/// of the state are the bits to emit, masked by the reader of the chunk,
/// where there is slack; nbits <= TL < 16) and the next state.
#[inline(always)]
fn step(t: &EncodeTable, state: u32, s: u8) -> (u32, u32) {
    let (delta_nbits, delta_find) = t.sym[s as usize];
    let plus = state.wrapping_add(delta_nbits);
    let nbits = plus >> 16;
    let idx = (state >> nbits).wrapping_add(delta_find) as usize;
    (state | (plus & 0xFFFF_0000), t.state_table[idx] as u32)
}

/// A chunk's bits and their count.
#[inline(always)]
fn unpack_chunk(c: u32) -> (u32, u32) {
    let nbits = c >> 16;
    (c & !(u32::MAX << nbits), nbits)
}

/// Append the 8-stream section (size table + padded streams) for `syms`
/// (symbol i in stream i % 8) to `out`; `chunks` is scratch, grown to
/// `syms.len()`. Two passes: last-to-first over all symbols with the 8
/// stream states side by side (independent chains, so they overlap in the
/// pipeline instead of serialising on one state's table lookup), packing
/// each symbol's chunk into `chunks[i]`; then per stream, first-to-last,
/// the initial state and the chunks, four per `put` (4 x TL <= MAX_PUT).
pub fn encode8_into(syms: &[u8], t: &EncodeTable, chunks: &mut Vec<u32>, out: &mut Vec<u8>) {
    let n = syms.len();
    chunks.clear();
    chunks.resize(n, 0);
    // Initial encoder state: any value in [L, 2L); use the first symbol's
    // smallest legal state so the decoder's first state is well defined.
    let mut st = [L as u32; STREAMS];
    let full = n & !(STREAMS - 1);
    for i in (full..n).rev() {
        let (c, s) = step(t, st[i - full], syms[i]);
        chunks[i] = c;
        st[i - full] = s;
    }
    for (sg, cg) in syms[..full].chunks_exact(STREAMS).zip(chunks[..full].chunks_exact_mut(STREAMS)).rev() {
        for k in 0..STREAMS {
            let (c, s) = step(t, st[k], sg[k]);
            cg[k] = c;
            st[k] = s;
        }
    }
    let max_bits = (TL as usize) * (1 + n.div_ceil(STREAMS));
    write_streams(out, max_bits, |k, w| {
        let c = &chunks[k.min(n)..];
        let mut i = 0;
        // SAFETY: TL bits of state plus at most ceil(n / 8) chunks of at
        // most TL bits each go into this stream: `max_bits`.
        unsafe {
            w.put((st[k] - L as u32) as u64, TL);
            while i + 3 * STREAMS < c.len() {
                let (a, na) = unpack_chunk(c[i]);
                let (b, nb) = unpack_chunk(c[i + STREAMS]);
                let (d, nd) = unpack_chunk(c[i + 2 * STREAMS]);
                let (e, ne) = unpack_chunk(c[i + 3 * STREAMS]);
                let ab = a | b << na;
                let de = d | e << nd;
                w.put((ab as u64) | (de as u64) << (na + nb), na + nb + nd + ne);
                i += 4 * STREAMS;
            }
            while i < c.len() {
                let (a, na) = unpack_chunk(c[i]);
                w.put(a as u64, na);
                i += STREAMS;
            }
        }
    });
}

/// The 8 streams as separate vectors (tests).
pub fn encode8(syms: &[u8], t: &EncodeTable) -> Vec<Vec<u8>> {
    let mut section = Vec::new();
    encode8_into(syms, t, &mut Vec::new(), &mut section);
    split_streams(&section)
}

/// Decode `n` symbols from 8 interleaved tANS streams (symbol i in stream
/// i % STREAMS). Structured like `huff8::decode`: a clamped, accounted
/// `BitReader` carries 6 fields/stream, which spills registers across 8
/// unrolled streams, so the hot loop runs on `FastReader` (3 fields/stream,
/// unclamped and unaccounted) instead. `safe_refills` proves, from each
/// stream's remaining real bytes, how many `PER_ITER`-symbol batches can
/// run before that stream's reader might need the clamp; the outer loop
/// takes the minimum across streams and re-evaluates every pass. Whatever
/// is left once some stream runs low on margin -- normally under one
/// `PER_ITER` batch, but can be more if one stream is short -- goes
/// through the clamped, accounted, per-symbol path, which is also what
/// makes `overrun` exact for corrupt/truncated streams.
///
/// (A fix-round attempt split this loop's 8 streams into two sequential
/// groups of 4, to bring the per-iteration live set -- `FastReader`'s 3
/// fields + tANS's own `st[k]`, x 8 streams -- under the ~31 GPRs
/// available. It measured slower, not faster, and was reverted; see
/// CHANGELOG-BENCH.md.)
pub fn decode8(t: &DecodeTable, streams: &[&[u8]; STREAMS], n: usize, out: &mut [u8]) -> Result<(), ()> {
    assert!(out.len() >= n);
    let e = t.entries.as_slice();
    let mut rs: [BitReader; STREAMS] = std::array::from_fn(|k| BitReader::new(streams[k]));
    let mut st = [0u32; STREAMS];
    for k in 0..STREAMS {
        st[k] = rs[k].get(TL) as u32;
    }
    let lasts: [*const u8; STREAMS] = std::array::from_fn(|k| rs[k].last());
    let mut fast: [FastReader; STREAMS] = std::array::from_fn(|k| rs[k].to_fast());

    let mut o = 0usize;
    let mut remaining = n;
    loop {
        let mut iters = remaining / PER_ITER;
        for k in 0..STREAMS {
            iters = iters.min(fast[k].safe_refills(lasts[k]));
        }
        if iters == 0 {
            break;
        }
        for _ in 0..iters {
            // One bounds check per batch, not one per symbol.
            let batch: &mut [u8; PER_ITER] = (&mut out[o..o + PER_ITER]).try_into().unwrap();
            for k in 0..STREAMS {
                // SAFETY: `iters` <= every stream's safe_refills(lasts[k]),
                // proving this refill's starting p is <= lasts[k].
                unsafe {
                    fast[k].refill();
                }
            }
            for j in 0..4 {
                for k in 0..STREAMS {
                    // SAFETY: st[k] < L because base + bits < L for a valid
                    // table and st[k] is initialised from get(TL), which is
                    // < L (this holds inductively regardless of which bits
                    // a corrupt/overrun stream produces: nbits <= TL and
                    // base + (2^nbits - 1) < L are properties of the table
                    // alone, so base + bits < L for any bits < 2^nbits).
                    let d = unsafe { *e.get_unchecked(st[k] as usize) };
                    let nbits = unpack_nbits(d);
                    let bits = fast[k].peek(nbits) as u32;
                    fast[k].consume(nbits);
                    st[k] = unpack_base(d) + bits;
                    batch[j * STREAMS + k] = unpack_sym(d);
                }
            }
            o += PER_ITER;
        }
        remaining -= PER_ITER * iters;
    }

    for (k, f) in fast.into_iter().enumerate() {
        rs[k].resume(f);
    }

    // Tail: fewer than PER_ITER symbols left, or some stream ran low on
    // safe margin early. Back to the clamped per-symbol path, in stream
    // order i % STREAMS -- also what makes `overrun` below exact.
    for i in o..n {
        let k = i % STREAMS;
        let d = e[st[k] as usize];
        rs[k].refill();
        let nbits = unpack_nbits(d);
        let bits = rs[k].peek(nbits) as u32;
        rs[k].consume(nbits);
        st[k] = unpack_base(d) + bits;
        out[i] = unpack_sym(d);
    }
    if rs.iter().any(|r| r.overrun()) {
        return Err(());
    }
    Ok(())
}
