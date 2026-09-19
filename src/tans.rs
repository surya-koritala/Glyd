//! Table ANS (FSE-style) for the sequence code streams: at most 64
//! symbols, table log 10. The decoder reads a forward LSB-first stream:
//! the initial state (TL bits), then per symbol the table's `nbits` bits.
//! The encoder therefore processes symbols last-to-first and emits the
//! chunks in reverse, so no bit-reversal is needed anywhere.
use crate::bits::{split_streams, write_section, BitReader, BitWriter, Framing, Stream, MAX_PUT, PAD};

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

/// Decode-table entry packed into a `u64`: `nbits | sym << 8 | mask << 16
/// | base << 32`, with `mask = (1 << nbits) - 1` (`MASK_NBITS` holds the
/// symbol-independent part). Laid out for the 8-stream hot loop on
/// AArch64: `nbits` in the low byte is the shift amount as it stands (a
/// shift reads its low six bits) and the position advance is one `add`
/// with a `uxtb` operand; the mask below bit 32 and the base above it
/// make the bits one 32-bit `and` with a shifted operand and the next
/// state one `add` with a shifted operand. (A `u32` entry without the
/// mask cost a shift and a subtract per symbol to build it.) `nbits <=
/// TL < 16`, so mask and base each fit 16 bits.
#[inline(always)]
fn unpack_nbits(e: u64) -> u32 {
    e as u32 & 0xFF
}

#[inline(always)]
fn unpack_mask(e: u64) -> u32 {
    e as u32 >> 16
}

#[inline(always)]
fn unpack_base(e: u64) -> u32 {
    (e >> 32) as u32
}

#[inline(always)]
fn unpack_sym(e: u64) -> u8 {
    (e >> 8) as u8
}

#[derive(Clone)]
pub struct DecodeTable {
    entries: [u64; L],
}

impl DecodeTable {
    pub fn build(counts: &[u16]) -> Option<DecodeTable> {
        let mut t = DecodeTable::empty();
        if t.rebuild(counts) {
            Some(t)
        } else {
            None
        }
    }

    /// A placeholder to `rebuild` into (every entry yields symbol 0 and
    /// stays at state 0).
    pub fn empty() -> DecodeTable {
        DecodeTable { entries: [0; L] }
    }

    /// Overwrite with the table for `counts`, in place (the decoder keeps
    /// one per stream across blocks; 8 KB is not worth moving per block).
    /// False, contents unspecified, unless the counts sum to L. Two passes
    /// on the stack, no allocation: the spread, then the entries in table
    /// order (the state an entry leads to is the rank of that occurrence
    /// among its symbol's in table order -- what the encoder's
    /// `state_table` is filled by -- so the fill cannot follow the
    /// spread's order). Each pass is written for throughput, not chains:
    /// the spread computes every position from its index (no running
    /// `pos`), and the fill's per-entry work is a 256-entry counter (a
    /// `u8` indexes it unchecked), one `clz`, one shift and two ORs with
    /// the mask and count from a 16-entry table. 2.2 -> 1.5 us per table
    /// on the M1 Max.
    pub fn rebuild(&mut self, counts: &[u16]) -> bool {
        if !check(counts) {
            return false;
        }
        let step = (L >> 1) + (L >> 3) + 3;
        let mut sp = [0u8; L];
        let mut k = 0usize;
        for (s, &c) in counts.iter().enumerate() {
            for j in k..k + c as usize {
                sp[j * step & (L - 1)] = s as u8;
            }
            k += c as usize;
        }
        let mut next = [0u32; 256];
        for (s, &c) in counts.iter().enumerate() {
            next[s] = c as u32;
        }
        for i in 0..L {
            let s = sp[i];
            let x = next[s as usize];
            next[s as usize] = x + 1;
            // x < 2L, so x's top bit is at TL - nbits.
            let nbits = x.leading_zeros() - (31 - TL);
            let base = ((x as u64) << nbits) - L as u64;
            self.entries[i] = MASK_NBITS[nbits as usize] | (s as u64) << 8 | base << 32;
        }
        true
    }
}

/// `mask << 16 | nbits` per `nbits` (0..=TL): a decode entry minus its
/// symbol and base.
const MASK_NBITS: [u64; TL as usize + 1] = {
    let mut t = [0u64; TL as usize + 1];
    let mut n = 0;
    while n <= TL as usize {
        t[n] = ((1u64 << n) - 1) << 16 | n as u64;
        n += 1;
    }
    t
};

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

#[doc(hidden)]
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

#[doc(hidden)]
pub struct Decoder<'a> {
    t: &'a [u64],
    r: BitReader<'a>,
    state: u32,
}

impl<'a> Decoder<'a> {
    pub fn new(t: &'a DecodeTable, stream: &'a [u8]) -> Self {
        let mut r = BitReader::new(stream);
        let state = r.get(TL) as u32;
        Decoder { t: &t.entries[..], r, state }
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
/// 4 symbols/stream per batch: 4 * TL = 40 bits, from one 8-byte load
/// shifted by a sub-byte position (>= 57 valid bits).
const PER_ITER: usize = 4 * STREAMS;
/// Bytes a batch can advance a stream's load address: 40 bits.
const BATCH_BYTES: usize = (4 * TL / 8) as usize;
const _: () = assert!(4 * TL <= 57 && 4 * TL % 8 == 0);

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
pub fn encode8_into(syms: &[u8], t: &EncodeTable, chunks: &mut Vec<u32>, framing: Framing, out: &mut Vec<u8>) {
    let n = syms.len();
    chunks.clear();
    chunks.resize(n, 0);
    if framing == Framing::Single {
        // One state over every symbol, backwards; the stream holds the
        // final state then the chunks in symbol order.
        let mut st = L as u32;
        for i in (0..n).rev() {
            let (c, s) = step(t, st, syms[i]);
            chunks[i] = c;
            st = s;
        }
        write_section(out, framing, TL as usize * (1 + n), |_, w| unsafe {
            w.put((st - L as u32) as u64, TL);
            for &c in chunks.iter() {
                let (a, na) = unpack_chunk(c);
                w.put(a as u64, na);
            }
        });
        return;
    }
    // Initial encoder state (the decoder's final one): `L`. Any value in
    // [L, 2L) would do; the decoder never checks where it ends.
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
    write_section(out, framing, max_bits, |k, w| {
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
#[doc(hidden)]
pub fn encode8(syms: &[u8], t: &EncodeTable) -> Vec<Vec<u8>> {
    let mut section = Vec::new();
    encode8_into(syms, t, &mut Vec::new(), Framing::Wide, &mut section);
    split_streams(&section)
}

/// Batches the fast loop can take on one stream before a load might
/// start past `last`: the next load is at `at`, each later one at most
/// BATCH_BYTES further.
#[inline(always)]
fn safe_batches(at: usize, last: usize) -> usize {
    if at > last {
        0
    } else {
        (last - at) / BATCH_BYTES + 1
    }
}

/// Decode `n` symbols from 8 interleaved tANS streams (symbol i in stream
/// i % STREAMS) into `out[..n]`.
pub fn decode8(t: &DecodeTable, streams: &[Stream; STREAMS], n: usize, out: &mut [u8]) -> Result<(), ()> {
    decode8_rows(t, streams, n, STREAMS, out)
}

/// `decode8` with symbol i landing at `out[i / 8 * row + i % 8]`: rows of
/// eight at a stride (`row >= 8`; `row == 8` is `decode8`). v7 decodes
/// its three code streams at `row = 24` into one 24-byte group per eight
/// sequences, so its walk reads them through one pointer.
///
/// Structured like `v7_decode::sequences`' walk: the hot
/// state per stream is an absolute bit address (`ptr * 8 + bit`) and a
/// window of bits loaded from it -- 2 live values per stream, so the 8
/// streams and their 8 states stay in registers (a reader with pointer,
/// accumulator and count is 3, and with the states that spilled) -- and per batch one unaligned 8-byte
/// load per stream, shifted by its sub-byte position, holds the 40 bits
/// four symbols can take. `safe_batches` proves, from each stream's
/// remaining real bytes, how many batches can run before a load might
/// leave the stream; the outer loop takes the minimum across streams and
/// re-evaluates every pass. Whatever is left -- normally under one batch,
/// more if one stream is short -- goes through the clamped, accounted
/// `BitReader`s started at the positions the fast loop reached
/// (`BitReader::new_at`), which is also what makes `overrun` exact for
/// corrupt or truncated streams.
#[cfg_attr(target_arch = "x86_64", inline(always))]
pub fn decode8_rows(t: &DecodeTable, streams: &[Stream; STREAMS], n: usize, row: usize, out: &mut [u8]) -> Result<(), ()> {
    decode8_rows_with(t, streams, n, row, false, out)
}

/// `decode8_rows`; with `single`, every symbol comes from `streams[0]`
/// through one state on the clamped path (a compact block's short
/// section).
pub fn decode8_rows_with(t: &DecodeTable, streams: &[Stream; STREAMS], n: usize, row: usize, single: bool, out: &mut [u8]) -> Result<(), ()> {
    assert!(row >= STREAMS);
    if single {
        assert!(n == 0 || out.len() > (n - 1) / STREAMS * row + (n - 1) % STREAMS);
        let e = &t.entries;
        let mut r = BitReader::new_at(streams[0], 0);
        r.refill();
        let mut st = r.peek(TL) as u32;
        r.consume(TL);
        for i in 0..n {
            // SAFETY: st < L, as in `sym`: base + bits < L for a valid table.
            let d = unsafe { *e.get_unchecked(st as usize) };
            r.refill();
            let nbits = unpack_nbits(d);
            let bits = r.peek(nbits) as u32;
            r.consume(nbits);
            st = unpack_base(d) + bits;
            out[i / STREAMS * row + i % STREAMS] = unpack_sym(d);
        }
        return if r.overrun() { Err(()) } else { Ok(()) };
    }
    // Every index `i / 8 * row + i % 8` for i < n is in bounds from here on.
    assert!(n == 0 || out.len() > (n - 1) / STREAMS * row + (n - 1) % STREAMS);
    let e = &t.entries;
    for s in streams {
        assert!(s.bytes.len() >= PAD, "stream shorter than its padding");
    }
    // Initial states: the first TL bits of each stream (a valid stream
    // always holds them; a shorter one reads padding and overruns below).
    let mut st: [u32; STREAMS] = std::array::from_fn(|k| {
        // SAFETY: bytes.len() >= PAD = 8 bytes, asserted above.
        unsafe { std::ptr::read_unaligned(streams[k].bytes.as_ptr() as *const u64) as u32 & (L as u32 - 1) }
    });
    let mut b: [usize; STREAMS] = std::array::from_fn(|k| streams[k].bytes.as_ptr() as usize * 8 + TL as usize);
    let lasts: [usize; STREAMS] = std::array::from_fn(|k| streams[k].bytes.as_ptr() as usize + streams[k].bytes.len() - PAD);

    let mut o = 0usize;
    let mut remaining = n;
    loop {
        let mut iters = remaining / PER_ITER;
        for k in 0..STREAMS {
            iters = iters.min(safe_batches(b[k] >> 3, lasts[k]));
        }
        if iters == 0 {
            break;
        }
        for _ in 0..iters {
            // The batch's four rows. SAFETY: o + PER_ITER <= n (iters is
            // bounded by remaining / PER_ITER), so every store below is
            // at an index the length assert above covers.
            let r0 = unsafe { out.as_mut_ptr().add(o / STREAMS * row) };
            let rows = [r0, r0.wrapping_add(row), r0.wrapping_add(2 * row), r0.wrapping_add(3 * row)];
            let mut w = [0u64; STREAMS];
            for k in 0..STREAMS {
                // SAFETY: `iters` <= every stream's safe_batches at the
                // start of this run and each batch advances a load address
                // by at most BATCH_BYTES, so this load starts at or before
                // lasts[k]: its 8 bytes are inside stream k.
                w[k] = unsafe { std::ptr::read_unaligned((b[k] >> 3) as *const u64) } >> (b[k] & 7);
            }
            // Written out: the 32 symbols as straight-line code (a `for j`
            // over the four rows kept a counter, which was the register
            // that tipped two positions onto the stack). `$consume` shifts
            // the window past the symbol; the last row has nothing left
            // to read from it.
            #[allow(unused_macros)]
            macro_rules! sym {
                ($j:literal, $k:literal, $nbits:ident, $consume:block) => {{
                    // SAFETY: st[k] < L because base + bits < L for a valid
                    // table and st[k] is initialised masked to < L (this
                    // holds inductively regardless of which bits a
                    // corrupt/overrun stream produces: nbits <= TL and
                    // base + (2^nbits - 1) < L are properties of the table
                    // alone, so base + bits < L for any bits < 2^nbits).
                    let d = unsafe { *e.get_unchecked(st[$k] as usize) };
                    let $nbits = unpack_nbits(d);
                    let bits = w[$k] as u32 & unpack_mask(d);
                    $consume
                    b[$k] += $nbits as usize;
                    st[$k] = unpack_base(d) + bits;
                    unsafe { *rows[$j].add($k) = unpack_sym(d) };
                }};
            }
            #[allow(unused_macros)]
            macro_rules! row {
                ($j:literal, shift) => {
                    sym!($j, 0, n, { w[0] >>= n });
                    sym!($j, 1, n, { w[1] >>= n });
                    sym!($j, 2, n, { w[2] >>= n });
                    sym!($j, 3, n, { w[3] >>= n });
                    sym!($j, 4, n, { w[4] >>= n });
                    sym!($j, 5, n, { w[5] >>= n });
                    sym!($j, 6, n, { w[6] >>= n });
                    sym!($j, 7, n, { w[7] >>= n });
                };
                ($j:literal, last) => {
                    sym!($j, 0, n, {});
                    sym!($j, 1, n, {});
                    sym!($j, 2, n, {});
                    sym!($j, 3, n, {});
                    sym!($j, 4, n, {});
                    sym!($j, 5, n, {});
                    sym!($j, 6, n, {});
                    sym!($j, 7, n, {});
                };
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                row!(0, shift);
                row!(1, shift);
                row!(2, shift);
                row!(3, last);
            }
            // Stream-major on x86-64 (16 general registers): one stream's
            // four symbols with its state, position and window live, the
            // eight chains overlapped by the core rather than the register
            // file. Same table walk as `sym`.
            #[cfg(target_arch = "x86_64")]
            {
                macro_rules! stream {
                    ($k:literal) => {{
                        let mut v = w[$k];
                        // Position and state through memory (see the
                        // walk in v7_decode): 16 registers do not hold
                        // eight of each plus the window and the table.
                        let mut p = unsafe { std::ptr::read_volatile(&b[$k]) };
                        let mut x = unsafe { std::ptr::read_volatile(&st[$k]) };
                        // SAFETY: as in `sym`, x < L by the table's construction.
                        let d0 = unsafe { *e.get_unchecked(x as usize) };
                        x = unpack_base(d0) + (v as u32 & unpack_mask(d0));
                        v >>= unpack_nbits(d0);
                        p += unpack_nbits(d0) as usize;
                        let d1 = unsafe { *e.get_unchecked(x as usize) };
                        x = unpack_base(d1) + (v as u32 & unpack_mask(d1));
                        v >>= unpack_nbits(d1);
                        p += unpack_nbits(d1) as usize;
                        let d2 = unsafe { *e.get_unchecked(x as usize) };
                        x = unpack_base(d2) + (v as u32 & unpack_mask(d2));
                        v >>= unpack_nbits(d2);
                        p += unpack_nbits(d2) as usize;
                        let d3 = unsafe { *e.get_unchecked(x as usize) };
                        unsafe {
                            std::ptr::write_volatile(&mut st[$k], unpack_base(d3) + (v as u32 & unpack_mask(d3)));
                            std::ptr::write_volatile(&mut b[$k], p + unpack_nbits(d3) as usize);
                        }
                        unsafe {
                            *rows[0].add($k) = unpack_sym(d0);
                            *rows[1].add($k) = unpack_sym(d1);
                            *rows[2].add($k) = unpack_sym(d2);
                            *rows[3].add($k) = unpack_sym(d3);
                        }
                    }};
                }
                stream!(0);
                stream!(1);
                stream!(2);
                stream!(3);
                stream!(4);
                stream!(5);
                stream!(6);
                stream!(7);
            }
            o += PER_ITER;
        }
        remaining -= PER_ITER * iters;
    }

    let mut rs: [BitReader; STREAMS] = std::array::from_fn(|k| BitReader::new_at(streams[k], b[k] - streams[k].bytes.as_ptr() as usize * 8));

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
        out[i / STREAMS * row + i % STREAMS] = unpack_sym(d);
    }
    if rs.iter().any(|r| r.overrun()) {
        return Err(());
    }
    Ok(())
}
