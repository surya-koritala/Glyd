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

const _: () = assert!(usize::BITS >= 64, "v7 bit-position readers use absolute bit addresses");

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

    /// Close the stream: the partial byte, if any. Returns the cursor just
    /// past it. (The section's `PAD` zeros come after the last stream.)
    ///
    /// # Safety
    /// As `put`.
    #[inline(always)]
    unsafe fn finish(self) -> *mut u8 {
        // The last `put` already stored the partial byte's bits here with
        // zeros above; the store below also covers the no-`put` case.
        *self.p = self.acc as u8;
        self.p.add((self.n > 0) as usize)
    }
}

/// Bytes of the size table at the head of a section: seven 24-bit sizes,
/// the eighth stream's being what remains before the padding.
pub const SIZES_BYTES: usize = 3 * (STREAMS - 1);

/// A sub-stream as the decoders take it: `bytes` runs from its start to
/// the end of its section (at least `PAD` long, so a load starting at or
/// before `bytes.len() - PAD` is inside the section) and `len` says how
/// many of them are its own bits' (the overrun budget).
#[derive(Clone, Copy)]
pub struct Stream<'a> {
    pub bytes: &'a [u8],
    pub len: usize,
}

impl<'a> Stream<'a> {
    /// The eight streams of a section written by `write_streams`, or
    /// None if its size table does not fit the section.
    pub fn split(section: &'a [u8]) -> Option<[Stream<'a>; STREAMS]> {
        if section.len() < SIZES_BYTES + PAD {
            return None;
        }
        let data = SIZES_BYTES..section.len() - PAD;
        let mut out = [Stream { bytes: &section[data.end..], len: 0 }; STREAMS];
        let mut pos = data.start;
        for k in 0..STREAMS - 1 {
            let n = u32::from_le_bytes([section[3 * k], section[3 * k + 1], section[3 * k + 2], 0]) as usize;
            if n > data.end - pos {
                return None;
            }
            out[k] = Stream { bytes: &section[pos..], len: n };
            pos += n;
        }
        out[STREAMS - 1] = Stream { bytes: &section[pos..], len: data.end - pos };
        Some(out)
    }

    /// A stream that stands alone with its own padding (tests).
    pub fn whole(bytes: &'a [u8]) -> Stream<'a> {
        assert!(bytes.len() >= PAD);
        Stream { bytes, len: bytes.len() - PAD }
    }
}

pub const STREAMS: usize = 8;

/// The section layout: `SIZES_BYTES` of 24-bit LE sizes for streams 0-6,
/// the 8 sub-streams back to back, then `PAD` zero bytes; the eighth
/// stream's size is what the section has left. `fill(k, cursor)` writes
/// sub-stream `k` with at most `max_bits` bits, which is what the
/// reservation is proven from (see `BitCursor`).
pub fn write_streams(out: &mut Vec<u8>, max_bits: usize, mut fill: impl FnMut(usize, &mut BitCursor)) {
    let table = out.len();
    out.resize(table + SIZES_BYTES, 0);
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
        let len = out.len() - start;
        if k < STREAMS - 1 {
            assert!(len < 1 << 24, "sub-stream over 16 MB");
            out[table + 3 * k..table + 3 * k + 3].copy_from_slice(&len.to_le_bytes()[..3]);
        }
    }
    out.extend_from_slice(&[0u8; PAD]);
}

/// How a section is laid out: the v8 layout, or the compact one with
/// eight streams or, for a section of few symbols, one.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Framing {
    Wide,
    Compact,
    Single,
}

impl Framing {
    /// The framing of a compact block's section of `symbols` symbols.
    pub fn compact_for(symbols: usize) -> Framing {
        if symbols <= SINGLE_MAX_SYMBOLS {
            Framing::Single
        } else {
            Framing::Compact
        }
    }

    /// Streams a section of this framing has; symbol i goes to stream
    /// `i % streams`.
    pub fn streams(self) -> usize {
        if self == Framing::Single {
            1
        } else {
            STREAMS
        }
    }
}

/// A compact section of at most this many symbols is one stream: the
/// decoders' per-symbol path is slower per symbol than the eight-stream
/// batches, but on a section this short the 8-way framing (a size table,
/// eight byte-aligned tails) costs more than the speed is worth.
pub const SINGLE_MAX_SYMBOLS: usize = 1024;

/// A section in the given framing; `fill(k, cursor)` writes stream `k`
/// (only k = 0 for `Single`).
pub fn write_section(out: &mut Vec<u8>, framing: Framing, max_bits: usize, fill: impl FnMut(usize, &mut BitCursor)) {
    match framing {
        Framing::Wide => write_streams(out, max_bits, fill),
        Framing::Compact => write_streams_compact(out, STREAMS, max_bits, fill),
        Framing::Single => write_streams_compact(out, 1, max_bits, fill),
    }
}

/// Bytes a section's framing takes beyond its streams' bits: the size
/// table and the padding (compact: a stream count and the varints, ~1
/// byte each; the payload's one padding is counted in the block).
pub fn section_frame_bytes(framing: Framing) -> usize {
    match framing {
        Framing::Wide => SIZES_BYTES + PAD,
        Framing::Compact => STREAMS,
        Framing::Single => 1,
    }
}

/// The compact section layout (v9 blocks): a stream count (1 or 8), for
/// 8 seven varint sizes (7 bits a byte, low first, the top bit marking
/// more), the streams back to back, and no padding of its own:
/// `section` must run on to the end of the payload, whose last `PAD`
/// bytes are zero.
pub fn write_streams_compact(out: &mut Vec<u8>, n_streams: usize, max_bits: usize, mut fill: impl FnMut(usize, &mut BitCursor)) {
    debug_assert!(n_streams == 1 || n_streams == STREAMS);
    let mut lens = [0usize; STREAMS];
    let mut data: Vec<u8> = Vec::with_capacity(n_streams * (max_bits / 8 + 16));
    out.push(n_streams as u8);
    for k in 0..n_streams {
        data.reserve(max_bits / 8 + 16);
        let start = data.len();
        // SAFETY: as in `write_streams`.
        unsafe {
            let base = data.as_mut_ptr().add(start);
            let mut c = BitCursor { acc: 0, n: 0, p: base };
            fill(k, &mut c);
            let end = c.finish();
            data.set_len(start + end.offset_from(base) as usize);
        }
        lens[k] = data.len() - start;
    }
    if n_streams == STREAMS {
        for &n in &lens[..STREAMS - 1] {
            let mut v = n;
            while v >= 128 {
                out.push((v & 127) as u8 | 128);
                v >>= 7;
            }
            out.push(v as u8);
        }
    }
    out.extend_from_slice(&data);
}

impl<'a> Stream<'a> {
    /// The streams of a compact section, and whether it is a single one
    /// (then only `[0]` holds symbols; the rest are empty): `section` runs
    /// from the stream count to the end of the payload (`section_len`
    /// bytes are the section's own), so every stream's bytes reach the
    /// payload's padding.
    pub fn split_compact(section: &'a [u8], section_len: usize) -> Option<([Stream<'a>; STREAMS], bool)> {
        if section_len > section.len() - PAD.min(section.len()) || section_len < 1 {
            return None;
        }
        let n_streams = section[0] as usize;
        let mut pos = 1usize;
        if n_streams == 1 {
            let mut out = [Stream { bytes: &section[section_len..], len: 0 }; STREAMS];
            out[0] = Stream { bytes: &section[pos..], len: section_len - pos };
            return Some((out, true));
        }
        if n_streams != STREAMS {
            return None;
        }
        let mut lens = [0usize; STREAMS];
        for l in lens[..STREAMS - 1].iter_mut() {
            let (mut v, mut shift) = (0usize, 0u32);
            loop {
                let b = *section.get(pos)?;
                pos += 1;
                v |= ((b & 127) as usize) << shift;
                if b < 128 {
                    break;
                }
                shift += 7;
                if shift > 28 {
                    return None;
                }
            }
            *l = v;
        }
        let data_start = pos;
        let sum: usize = lens[..STREAMS - 1].iter().sum();
        if data_start + sum > section_len {
            return None;
        }
        lens[STREAMS - 1] = section_len - data_start - sum;
        let mut out = [Stream { bytes: &section[section_len..], len: 0 }; STREAMS];
        for k in 0..STREAMS {
            if section.len() - pos < PAD {
                return None;
            }
            out[k] = Stream { bytes: &section[pos..], len: lens[k] };
            pos += lens[k];
        }
        Some((out, false))
    }
}

/// Split a `write_streams` section back into its 8 streams' own bytes,
/// each followed by `PAD` zeros as a stream on its own (test helper).
pub fn split_streams(section: &[u8]) -> Vec<Vec<u8>> {
    let streams = Stream::split(section).expect("a write_streams section");
    streams
        .iter()
        .map(|s| {
            let mut v = s.bytes[..s.len].to_vec();
            v.extend_from_slice(&[0u8; PAD]);
            v
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
    /// use glyd::bits::{BitReader, BitWriter};
    /// let r = { let v = BitWriter::new().finish(); BitReader::new(&v) };
    /// let _ = r.overrun();
    /// ```
    pub fn new(src: &'a [u8]) -> BitReader<'a> {
        Self::new_at(Stream::whole(src), 0)
    }

    /// A reader positioned at absolute bit `pos` of `src` (0 = its first
    /// bit), for a caller that walked the bits before it some other way
    /// (v7's extra-bits fast loop). The accounting counts `pos` as
    /// consumed, so `overrun` is exact from here: a `pos` past the stream
    /// reports it at once, and the first load is clamped like every other.
    pub fn new_at(src: Stream<'a>, pos: usize) -> BitReader<'a> {
        assert!(src.bytes.len() >= PAD && src.len <= src.bytes.len() - PAD, "stream without its padding");
        let p = src.bytes.as_ptr();
        let bit = (pos & 7) as u32;
        let mut r = BitReader {
            p: unsafe { p.add((pos >> 3).min(src.bytes.len() - PAD)) },
            last: unsafe { p.add(src.bytes.len() - PAD) },
            bits: 0,
            cnt: 0,
            filled: 0,
            // The whole bytes of `pos` here; its `bit` sub-byte bits are
            // consumed below into the open window, so `overrun` sees both.
            budget: (src.len * 8) as i64 - (pos - bit as usize) as i64,
            _src: std::marker::PhantomData,
        };
        r.refill();
        r.consume(bit);
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
}
