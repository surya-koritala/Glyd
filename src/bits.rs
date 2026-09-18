//! LSB-first bitstreams for the v7 entropy coders.
//!
//! The writer mirrors the reader: a 64-bit accumulator holding under 8
//! bits between calls, and every `put` stores the whole accumulator with
//! one unaligned 8-byte write at the cursor and advances by the whole
//! bytes it completed, branch-free. The reader keeps a 64-bit accumulator
//! and refills it with one unaligned 8-byte load, branch-free (Giesen's
//! scheme): `cnt` is the number of valid low bits; after `refill` it is
//! at least 56. The load pointer is clamped to `end - 8`, so a corrupt
//! stream can make the reader return zeros but never read outside its
//! slice; `overrun` reports that.

pub const PAD: usize = 8;

/// Most bits one `put` may carry: 7 banked + 56 <= 63 keeps every shift
/// in range.
pub const MAX_PUT: u32 = 56;

/// Write cursor over a pre-reserved buffer, held by value in the caller's
/// frame so `acc`/`n`/`p` stay in registers (the `finder::Cursors`
/// lesson: state behind `&mut` that a raw store may alias goes through
/// memory on every call). Every `put` writes 8 bytes at `p` and advances
/// by the whole bytes completed, so a stream of `B` bits touches at most
/// `B / 8 + 8` bytes past its start, and `finish` adds the partial byte
/// and `PAD` zeros: `B / 8 + 16` bytes of room is always enough.
pub struct BitCursor {
    acc: u64,
    n: u32,
    p: *mut u8,
}

impl BitCursor {
    /// Append the low `nbits` (0..=`MAX_PUT`) of `value`, least
    /// significant first. `value` must be below `1 << nbits` (no mask
    /// here: every caller builds its values that way).
    ///
    /// # Safety
    /// The buffer behind `p` must have room for this stream's bit bound
    /// as described on the type (`write_streams` reserves it).
    #[inline(always)]
    pub unsafe fn put(&mut self, value: u64, nbits: u32) {
        debug_assert!(nbits <= MAX_PUT && value >> nbits == 0);
        self.acc |= value << self.n;
        self.n += nbits;
        std::ptr::write_unaligned(self.p as *mut u64, self.acc.to_le());
        let bytes = self.n >> 3;
        self.p = self.p.add(bytes as usize);
        self.acc >>= bytes * 8;
        self.n &= 7;
    }

    /// Close the stream: the partial byte, then `PAD` zero bytes. Returns
    /// the cursor just past the padding.
    ///
    /// # Safety
    /// As `put`.
    #[inline(always)]
    unsafe fn finish(self) -> *mut u8 {
        // The last `put` already stored the partial byte's bits here with
        // zeros above; the store below also covers the no-`put` case.
        *self.p = self.acc as u8;
        let p = self.p.add((self.n > 0) as usize);
        std::ptr::write_unaligned(p as *mut u64, 0u64);
        p.add(PAD)
    }
}

pub const STREAMS: usize = 8;

/// The v7 section layout: 8 sub-streams behind 8 u32 LE byte sizes, each
/// closed with `PAD` zeros. `fill(k, cursor)` writes sub-stream `k` with
/// at most `max_bits` bits, which is what the reservation is proven from
/// (see `BitCursor`).
pub fn write_streams(out: &mut Vec<u8>, max_bits: usize, mut fill: impl FnMut(usize, &mut BitCursor)) {
    let table = out.len();
    out.resize(table + 4 * STREAMS, 0);
    for k in 0..STREAMS {
        // SAFETY: `fill` puts at most `max_bits` bits, and a stream of B
        // bits plus its `finish` writes within B / 8 + 16 bytes of its
        // start (see `BitCursor`), which is reserved here; `set_len`
        // publishes only bytes those writes initialised.
        out.reserve(max_bits / 8 + 16);
        let start = out.len();
        unsafe {
            let base = out.as_mut_ptr().add(start);
            let mut c = BitCursor { acc: 0, n: 0, p: base };
            fill(k, &mut c);
            let end = c.finish();
            out.set_len(start + end.offset_from(base) as usize);
        }
        let len = (out.len() - start) as u32;
        out[table + 4 * k..table + 4 * k + 4].copy_from_slice(&len.to_le_bytes());
    }
}

/// Split a `write_streams` section back into its 8 streams (test helper).
pub fn split_streams(section: &[u8]) -> Vec<Vec<u8>> {
    let mut pos = 4 * STREAMS;
    (0..STREAMS)
        .map(|k| {
            let n = u32::from_le_bytes(section[4 * k..4 * k + 4].try_into().unwrap()) as usize;
            pos += n;
            section[pos - n..pos].to_vec()
        })
        .collect()
}

/// Growable writer for one stream (tests and the single-stream tANS
/// encoder); the hot paths use `write_streams`.
pub struct BitWriter {
    acc: u64,
    n: u32,
    out: Vec<u8>,
}

impl BitWriter {
    pub fn new() -> Self {
        BitWriter { acc: 0, n: 0, out: Vec::new() }
    }

    /// Append the low `nbits` (0..=`MAX_PUT`) of `value`, least
    /// significant first. Panics above `MAX_PUT`: this is the safe entry
    /// point to a raw cursor whose reservation is sized for one put, so
    /// the contract is checked in release too (it is not a hot path).
    #[inline(always)]
    pub fn put(&mut self, value: u64, nbits: u32) {
        assert!(nbits <= MAX_PUT, "put of more than MAX_PUT bits");
        self.out.reserve(16);
        let len = self.out.len();
        // SAFETY: 16 bytes of room past `len`, more than one put's 8-byte
        // store; `set_len` publishes only the whole bytes it completed.
        unsafe {
            let base = self.out.as_mut_ptr().add(len);
            let mut c = BitCursor { acc: self.acc, n: self.n, p: base };
            c.put(value & ((1u64 << nbits) - 1), nbits);
            self.out.set_len(len + c.p.offset_from(base) as usize);
            self.acc = c.acc;
            self.n = c.n;
        }
    }

    /// Flush the partial byte and append `PAD` zero bytes.
    pub fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.out.push(self.acc as u8);
        }
        self.out.extend_from_slice(&[0u8; PAD]);
        self.out
    }
}

pub struct BitReader<'a> {
    p: *const u8,
    /// Last address a full 8-byte load may start at.
    last: *const u8,
    bits: u64,
    cnt: u32,
    /// `cnt` immediately after the last `refill`, i.e. the start of the
    /// current fill window. `filled - cnt` is the bits consumed within
    /// that window, folded into `budget` the next time `refill` runs
    /// instead of on every `consume` call.
    filled: u32,
    /// Bits left in the stream, counting down from `total_bits` once per
    /// `refill` (not once per `consume`). Signed so it can go negative:
    /// that's the overrun signal, and folding `total_bits` into its
    /// initial value keeps this reader at the same field count as an
    /// explicit `consumed` + `total_bits` pair.
    budget: i64,
    /// Ties this reader to `src`'s lifetime: `p`/`last` are raw pointers
    /// into it, so nothing here is a real borrow without this marker,
    /// and safe code could otherwise outlive the buffer (use-after-free).
    _src: std::marker::PhantomData<&'a [u8]>,
}

impl<'a> BitReader<'a> {
    /// `src` must end with `PAD` bytes (as `BitWriter::finish` produces).
    ///
    /// The returned reader borrows `src`, so a buffer that doesn't outlive
    /// it is a compile error, not a use-after-free:
    ///
    /// ```compile_fail
    /// use simd_stream_codec::bits::{BitReader, BitWriter};
    /// let r = { let v = BitWriter::new().finish(); BitReader::new(&v) };
    /// let _ = r.overrun();
    /// ```
    pub fn new(src: &'a [u8]) -> BitReader<'a> {
        assert!(src.len() >= PAD, "stream shorter than its padding");
        let p = src.as_ptr();
        let mut r = BitReader {
            p,
            last: unsafe { p.add(src.len() - PAD) },
            bits: 0,
            cnt: 0,
            filled: 0,
            budget: ((src.len() - PAD) * 8) as i64,
            _src: std::marker::PhantomData,
        };
        r.refill();
        r
    }

    #[inline(always)]
    pub fn refill(&mut self) {
        // Fold in whatever was consumed since the last refill, once per
        // refill instead of once per consume (this loop calls refill 4x
        // less often than consume, since it decodes 4 symbols per stream
        // between refills).
        self.budget -= (self.filled - self.cnt) as i64;
        unsafe {
            let p = if self.p > self.last { self.last } else { self.p };
            self.bits |= std::ptr::read_unaligned(p as *const u64) << self.cnt;
            self.p = p.add(((63 - self.cnt) >> 3) as usize);
            self.cnt |= 56;
        }
        self.filled = self.cnt;
    }

    #[inline(always)]
    pub fn peek(&self, n: u32) -> u64 {
        self.bits & ((1u64 << n) - 1)
    }

    #[inline(always)]
    pub fn consume(&mut self, n: u32) {
        self.bits >>= n;
        self.cnt -= n;
    }

    /// Read `n` (0..=32) bits; refills first.
    #[inline(always)]
    pub fn get(&mut self, n: u32) -> u64 {
        self.refill();
        let v = self.peek(n);
        self.consume(n);
        v
    }

    /// More bits consumed than the stream holds: the data was corrupt.
    /// `budget` covers every fully-closed fill window; subtract whatever
    /// has been consumed from the still-open one too.
    pub fn overrun(&self) -> bool {
        let open = (self.filled - self.cnt) as i64;
        self.budget - open < 0
    }

    /// End-of-safe-region pointer, for `FastReader::safe_refills`.
    pub fn last(&self) -> *const u8 {
        self.last
    }

    /// Snapshot the live state into an unclamped, unaccounted `FastReader`
    /// for a hot loop. Pairs with `resume`.
    pub fn to_fast(&self) -> FastReader {
        FastReader { p: self.p, bits: self.bits, cnt: self.cnt }
    }

    /// Write a `FastReader`'s state back and fold in the bits it consumed,
    /// so `overrun` stays exact across the excursion. `f` must be a value
    /// this same reader produced via `to_fast` (mixing readers would mix
    /// up the accounting below).
    ///
    /// `f` never clamped (the caller proved every one of its refills
    /// started at or before `last`), so the bits it loaded are exactly
    /// `(f.p - self.p) * 8` (the same identity `refill` relies on: each
    /// call's `p` advance in bytes equals the bits it newly loaded). The
    /// window still open when `to_fast` was called started with `filled`
    /// bits banked (not `cnt`: bits consumed between the last `refill`
    /// and `to_fast`, i.e. `filled - cnt` at that point, haven't been
    /// folded into `budget` yet either, and must not be lost when this
    /// closes that window out) and ends with `f.cnt` banked, so the bits
    /// consumed over the whole window, excursion included, is
    /// `consumed = self.filled + loaded - f.cnt`.
    pub fn resume(&mut self, f: FastReader) {
        let loaded = (f.p as usize - self.p as usize) as i64 * 8;
        let consumed = self.filled as i64 + loaded - f.cnt as i64;
        self.budget -= consumed;
        self.p = f.p;
        self.bits = f.bits;
        self.cnt = f.cnt;
        self.filled = f.cnt;
    }
}

/// Unclamped reader for hot loops: a caller must prove, from the stream's
/// remaining bytes, that every refill it performs starts at or before
/// `last` (see `safe_refills`). `p` advances by at most 7 bytes per refill.
/// Fields are private -- get one only from `BitReader::to_fast`.
pub struct FastReader {
    p: *const u8,
    bits: u64,
    cnt: u32,
}

impl FastReader {
    /// SAFETY (caller must uphold): `self.p` must be `<= last` for whatever
    /// `last` the reader was proved safe against (see `safe_refills`) --
    /// otherwise this can read past the end of the stream's allocation.
    #[inline(always)]
    pub unsafe fn refill(&mut self) {
        self.bits |= std::ptr::read_unaligned(self.p as *const u64) << self.cnt;
        self.p = self.p.add(((63 - self.cnt) >> 3) as usize);
        self.cnt |= 56;
    }

    #[inline(always)]
    pub fn peek(&self, n: u32) -> u64 {
        self.bits & ((1u64 << n) - 1)
    }

    #[inline(always)]
    pub fn consume(&mut self, n: u32) {
        self.bits >>= n;
        self.cnt -= n;
    }

    /// How many refills can run unclamped from here: each advances p by <= 7.
    #[inline(always)]
    pub fn safe_refills(&self, last: *const u8) -> usize {
        if self.p > last {
            0
        } else {
            (last as usize - self.p as usize) / 7 + 1
        }
    }
}
