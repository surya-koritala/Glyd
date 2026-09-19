//! 8-stream interleaved Huffman for byte streams. Symbol i lives in
//! sub-stream i % 8, LSB-first with bit-reversed canonical codes, so the
//! decoder's table is indexed by the next `TB` bits directly. Measured on
//! the M1 Max: 0.61 ns/symbol (tests/v7_codecs.rs, huff8_speed_silesia).
use crate::bits::{split_streams, write_streams, BitReader, Stream, MAX_PUT, PAD};
use crate::huffman::{build_codes, build_lengths, MAX_CODE_LEN};

pub use crate::bits::STREAMS;
pub const TB: u32 = MAX_CODE_LEN;
pub const TABLE_BYTES: usize = 128;
/// `encode_into` concatenates four codes per put.
const _: () = assert!(4 * TB <= MAX_PUT);

fn reverse_bits(code: u16, len: u8) -> u16 {
    let mut r = 0u16;
    for i in 0..len {
        r |= ((code >> i) & 1) << (len - 1 - i);
    }
    r
}

pub fn lengths_for(hist: &[u64; 256]) -> [u8; 256] {
    let present = hist.iter().filter(|&&c| c > 0).count();
    if present <= 1 {
        // One symbol: give it a 1-bit code so the stream is well formed.
        let mut l = [0u8; 256];
        if let Some(s) = hist.iter().position(|&c| c > 0) {
            l[s] = 1;
        }
        return l;
    }
    build_lengths(hist)
}

/// Bytes the coded stream will occupy, plus the packed table.
pub fn coded_size(hist: &[u64; 256], lengths: &[u8; 256]) -> usize {
    let bits: u64 = (0..256).map(|s| hist[s] * lengths[s] as u64).sum();
    (bits / 8) as usize + TABLE_BYTES
}

/// Packed decode table: entry = sym | (len << 8).
pub struct Table {
    entries: Vec<u16>,
}

impl Table {
    /// None unless the lengths form a prefix code that fits `TB` bits.
    pub fn build(lengths: &[u8; 256]) -> Option<Table> {
        let mut kraft: u64 = 0; // in units of 2^-TB
        for &l in lengths.iter() {
            if l as u32 > TB {
                return None;
            }
            if l > 0 {
                kraft += 1u64 << (TB - l as u32);
            }
        }
        // kraft == 0 means no symbols at all (e.g. empty input): a legitimate
        // degenerate case, not an invalid code. decode() with n = 0 never
        // indexes the table, so an all-zero entries table is fine to return.
        if kraft > (1u64 << TB) {
            return None;
        }
        let codes = build_codes(lengths);
        let mut entries = vec![0u16; 1 << TB];
        for s in 0..256 {
            let l = lengths[s];
            if l == 0 {
                continue;
            }
            let r = reverse_bits(codes[s], l) as usize;
            let step = 1usize << l;
            let mut i = r;
            while i < (1 << TB) {
                entries[i] = s as u16 | ((l as u16) << 8);
                i += step;
            }
        }
        Some(Table { entries })
    }
}

/// Append the 8-stream section (size table + padded streams) for `data`
/// to `out`. Sub-stream k is the stride-8 walk from `data[k]`, four
/// symbols per `put` (their codes concatenated off the accumulator's
/// dependency chain: 4 x TB <= MAX_PUT) from tables of reversed codes
/// and lengths (two loads beat one load plus the unpacking ALU ops).
pub fn encode_into(data: &[u8], lengths: &[u8; 256], out: &mut Vec<u8>) {
    let codes = build_codes(lengths);
    let rev: [u32; 256] = std::array::from_fn(|s| reverse_bits(codes[s], lengths[s]) as u32);
    let len: [u32; 256] = std::array::from_fn(|s| lengths[s] as u32);
    debug_assert!(data.iter().all(|&b| (1..=TB as u8).contains(&lengths[b as usize])), "symbol without a code, or one longer than TB");
    let max_bits = data.len().div_ceil(STREAMS) * TB as usize;
    write_streams(out, max_bits, |k, w| {
        let s = &data[k.min(data.len())..];
        let mut i = 0;
        // SAFETY: at most ceil(len / 8) symbols of at most TB bits each go
        // into this stream, the `max_bits` the section was reserved for.
        // No length exceeds TB: `build_codes` above indexes its
        // `[_; MAX_CODE_LEN + 1]` count table by every length, so a
        // longer one has already panicked there.
        unsafe {
            while i + 3 * STREAMS < s.len() {
                let (a, b, c, d) = (s[i] as usize, s[i + STREAMS] as usize, s[i + 2 * STREAMS] as usize, s[i + 3 * STREAMS] as usize);
                let lab = len[a] + len[b];
                let ab = rev[a] | rev[b] << len[a];
                let cd = rev[c] | rev[d] << len[c];
                w.put((ab as u64) | (cd as u64) << lab, lab + len[c] + len[d]);
                i += 4 * STREAMS;
            }
            while i < s.len() {
                let a = s[i] as usize;
                w.put(rev[a] as u64, len[a]);
                i += STREAMS;
            }
        }
    });
}

/// The 8 streams as separate vectors (tests).
#[doc(hidden)]
pub fn encode(data: &[u8], lengths: &[u8; 256]) -> Vec<Vec<u8>> {
    let mut section = Vec::new();
    encode_into(data, lengths, &mut section);
    split_streams(&section)
}

/// One symbol from a clamped reader (the tail).
#[inline(always)]
fn sym(r: &mut BitReader<'_>, t: &[u16]) -> u8 {
    // SAFETY: peek(TB) masks to < 1 << TB == t.len() (Table::build allocates
    // exactly `1 << TB` entries), so the index is always in bounds.
    let e = unsafe { *t.get_unchecked(r.peek(TB) as usize) };
    r.consume((e >> 8) as u32);
    e as u8
}

/// 4 symbols/stream per batch: 4 * TB = 44 bits, from one 8-byte load
/// shifted by a sub-byte position (>= 57 valid bits).
const PER_ITER: usize = 4 * STREAMS;
/// Bytes a batch can advance a stream's load address: ceil(44 / 8).
const BATCH_BYTES: usize = (4 * TB as usize).div_ceil(8);
const _: () = assert!(4 * TB <= 57);

/// Batches the fast loop can take on one stream before a load might
/// start past `last` (the next load is at `at`, each later one at most
/// BATCH_BYTES further).
#[inline(always)]
fn safe_batches(at: usize, last: usize) -> usize {
    if at > last {
        0
    } else {
        (last - at) / BATCH_BYTES + 1
    }
}

/// Decode `n` symbols into `out[..n]`. Err if any sub-stream overran.
///
/// The hot loop runs on absolute bit addresses (`ptr * 8 + bit`), one per
/// stream, with a window of bits loaded from it per batch of four
/// symbols -- 2 live values per stream, so the 8 streams stay in
/// registers (a reader with pointer, accumulator and count is 3, and
/// those spilled); a symbol is one
/// `and` for the index, the table load, a shift of the window by the
/// code length, an add of it to the position, and the byte store.
/// `safe_batches` proves, from each stream's remaining real bytes, how
/// many batches can run before a load might leave the stream; the outer
/// loop takes the minimum across streams (and caps at whole batches of
/// the symbols still wanted) and re-evaluates every pass, so streams with
/// short codes (whose address creeps forward slowly) just cost more,
/// cheap passes instead of one big one. Whatever is left once some stream
/// runs low on margin -- normally under one batch, more if one stream is
/// short -- goes through the clamped, accounted `BitReader`s started at
/// the positions the fast loop reached (`BitReader::new_at`), which is
/// also what makes `overrun` exact for corrupt/truncated streams.
#[cfg_attr(target_arch = "x86_64", inline(always))]
pub fn decode<'b>(table: &Table, streams: &[Stream<'b>; STREAMS], n: usize, out: &mut [u8]) -> Result<(), ()> {
    assert!(out.len() >= n);
    let t = table.entries.as_slice();
    for s in streams {
        assert!(s.bytes.len() >= PAD, "stream shorter than its padding");
    }
    let mut b: [usize; STREAMS] = std::array::from_fn(|k| streams[k].bytes.as_ptr() as usize * 8);
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
            // One bounds check per batch, not one per symbol.
            let batch: &mut [u8; PER_ITER] = (&mut out[o..o + PER_ITER]).try_into().unwrap();
            let mut w = [0u64; STREAMS];
            for k in 0..STREAMS {
                // SAFETY: `iters` <= every stream's safe_batches at the
                // start of this run and each batch advances a load address
                // by at most BATCH_BYTES, so this load starts at or before
                // lasts[k]: its 8 bytes are inside stream k.
                w[k] = unsafe { std::ptr::read_unaligned((b[k] >> 3) as *const u64) } >> (b[k] & 7);
            }
            // The 32 symbols written out; `$consume` shifts the window past
            // the symbol, and the last row has nothing left to read from it.
            #[allow(unused_macros)]
            macro_rules! sym {
                ($j:literal, $k:literal, $len:ident, $consume:block) => {{
                    // SAFETY: the index is masked to < 1 << TB == t.len().
                    let e = unsafe { *t.get_unchecked(w[$k] as usize & ((1 << TB) - 1)) };
                    let $len = (e >> 8) as u32;
                    $consume
                    b[$k] += $len as usize;
                    batch[$j * STREAMS + $k] = e as u8;
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
            // x86-64 has 16 general registers: the row-major order above
            // keeps 8 windows and 8 positions live and spills a third of
            // its instructions to the stack. Stream-major order decodes one
            // stream's four symbols with a single window live, and lets
            // the core overlap the eight independent chains itself.
            #[cfg(target_arch = "x86_64")]
            {
                macro_rules! stream {
                    ($k:literal) => {{
                        let mut p = b[$k];
                        let mut v = w[$k];
                        // SAFETY: as in `sym`, every index is masked to < 1 << TB.
                        let e0 = unsafe { *t.get_unchecked(v as usize & ((1 << TB) - 1)) };
                        v >>= e0 >> 8;
                        p += (e0 >> 8) as usize;
                        let e1 = unsafe { *t.get_unchecked(v as usize & ((1 << TB) - 1)) };
                        v >>= e1 >> 8;
                        p += (e1 >> 8) as usize;
                        let e2 = unsafe { *t.get_unchecked(v as usize & ((1 << TB) - 1)) };
                        v >>= e2 >> 8;
                        p += (e2 >> 8) as usize;
                        let e3 = unsafe { *t.get_unchecked(v as usize & ((1 << TB) - 1)) };
                        b[$k] = p + (e3 >> 8) as usize;
                        batch[$k] = e0 as u8;
                        batch[STREAMS + $k] = e1 as u8;
                        batch[2 * STREAMS + $k] = e2 as u8;
                        batch[3 * STREAMS + $k] = e3 as u8;
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
    // safe margin early. Either way, the clamped per-symbol path.
    for i in o..n {
        let k = i % STREAMS;
        rs[k].refill();
        out[i] = sym(&mut rs[k], t);
    }
    if rs.iter().any(|r| r.overrun()) {
        return Err(());
    }
    Ok(())
}
