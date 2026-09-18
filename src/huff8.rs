//! 8-stream interleaved Huffman for byte streams. Symbol i lives in
//! sub-stream i % 8, LSB-first with bit-reversed canonical codes, so the
//! decoder's table is indexed by the next `TB` bits directly. Measured on
//! the M1 Max: 0.61 ns/symbol (tests/v7_codecs.rs, huff8_speed_silesia).
use crate::bits::{split_streams, write_streams, BitReader, FastReader, MAX_PUT};
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
pub fn encode(data: &[u8], lengths: &[u8; 256]) -> Vec<Vec<u8>> {
    let mut section = Vec::new();
    encode_into(data, lengths, &mut section);
    split_streams(&section)
}

struct St<'a> {
    r: BitReader<'a>,
}

#[inline(always)]
fn sym(s: &mut St<'_>, t: &[u16]) -> u8 {
    // SAFETY: peek(TB) masks to < 1 << TB == t.len() (Table::build allocates
    // exactly `1 << TB` entries), so the index is always in bounds.
    let e = unsafe { *t.get_unchecked(s.r.peek(TB) as usize) };
    s.r.consume((e >> 8) as u32);
    e as u8
}

/// Same as `sym`, for the unclamped hot-loop reader.
#[inline(always)]
fn fast_sym(s: &mut FastReader, t: &[u16]) -> u8 {
    // SAFETY: peek(TB) masks to < 1 << TB == t.len() (Table::build allocates
    // exactly `1 << TB` entries), so the index is always in bounds.
    let e = unsafe { *t.get_unchecked(s.peek(TB) as usize) };
    s.consume((e >> 8) as u32);
    e as u8
}

const PER_ITER: usize = 4 * STREAMS; // 4 symbols per stream per refill: 4 * 11 <= 56

/// Decode `n` symbols into `out[..n]`. Err if any sub-stream overran.
///
/// Hot loop runs on `FastReader`: 3 live values per stream (as in
/// examples/huff_spike.rs), no clamp and no accounting, so 8 of them fit
/// in registers instead of spilling `BitReader`'s 6 fields/stream to the
/// stack. `safe_refills` proves, from each stream's remaining real bytes,
/// how many refills can run before its reader might need the clamp; the
/// outer loop takes the minimum across streams (and caps at whole
/// PER_ITER batches of the symbols still wanted) and re-evaluates every
/// pass, so streams with short codes (whose pointer creeps forward slowly)
/// just cost more, cheap passes instead of one big one. Whatever is left
/// once some stream runs low on margin -- normally under one PER_ITER
/// batch, but can be more if one stream is short -- goes through the
/// original clamped, accounted, per-symbol path, which is also what makes
/// `overrun` exact for corrupt/truncated streams.
pub fn decode<'b>(table: &Table, streams: &[&'b [u8]; STREAMS], n: usize, out: &mut [u8]) -> Result<(), ()> {
    assert!(out.len() >= n);
    let t = table.entries.as_slice();
    let mut st: [St<'b>; STREAMS] = std::array::from_fn(|k| St { r: BitReader::new(streams[k]) });
    let lasts: [*const u8; STREAMS] = std::array::from_fn(|k| st[k].r.last());
    let mut fast: [FastReader; STREAMS] = std::array::from_fn(|k| st[k].r.to_fast());

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
                    batch[j * STREAMS + k] = fast_sym(&mut fast[k], t);
                }
            }
            o += PER_ITER;
        }
        remaining -= PER_ITER * iters;
    }

    for (k, f) in fast.into_iter().enumerate() {
        st[k].r.resume(f);
    }

    // Tail: fewer than PER_ITER symbols left, or some stream ran low on
    // safe margin early. Either way, back to the clamped per-symbol path.
    for i in o..n {
        let k = i % STREAMS;
        st[k].r.refill();
        out[i] = sym(&mut st[k], t);
    }
    if st.iter().any(|s| s.r.overrun()) {
        return Err(());
    }
    Ok(())
}
