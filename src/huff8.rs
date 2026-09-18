//! 8-stream interleaved Huffman for byte streams. Symbol i lives in
//! sub-stream i % 8, LSB-first with bit-reversed canonical codes, so the
//! decoder's table is indexed by the next `TB` bits directly. Measured on
//! the M1 Max (examples/huff_spike.rs): 0.47 ns/symbol.
use crate::bits::{BitReader, BitWriter};
use crate::huffman::{build_codes, build_lengths, MAX_CODE_LEN};

pub const STREAMS: usize = 8;
pub const TB: u32 = MAX_CODE_LEN;
pub const TABLE_BYTES: usize = 128;

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
    pub entries: Vec<u16>,
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

pub fn encode(data: &[u8], lengths: &[u8; 256]) -> Vec<Vec<u8>> {
    let codes = build_codes(lengths);
    let rev: Vec<u16> = (0..256).map(|s| reverse_bits(codes[s], lengths[s])).collect();
    let mut writers: Vec<BitWriter> = (0..STREAMS).map(|_| BitWriter::new()).collect();
    for (i, &b) in data.iter().enumerate() {
        let l = lengths[b as usize] as u32;
        debug_assert!(l > 0, "symbol without a code");
        writers[i % STREAMS].put(rev[b as usize] as u64, l);
    }
    writers.into_iter().map(|w| w.finish()).collect()
}

struct St {
    r: BitReader,
}

#[inline(always)]
fn sym(s: &mut St, t: &[u16]) -> u8 {
    let e = t[s.r.peek(TB) as usize];
    s.r.consume((e >> 8) as u32);
    e as u8
}

/// Decode `n` symbols into `out[..n]`. Err if any sub-stream overran.
pub fn decode(table: &Table, streams: &[&[u8]; STREAMS], n: usize, out: &mut [u8]) -> Result<(), ()> {
    assert!(out.len() >= n);
    let t = table.entries.as_slice();
    let mut st: [St; STREAMS] = std::array::from_fn(|k| St { r: BitReader::new(streams[k]) });
    let per_iter = 4 * STREAMS; // 4 symbols per stream per refill: 4 * 11 <= 56
    let full = n / per_iter;
    let mut o = 0usize;
    for _ in 0..full {
        for k in 0..STREAMS {
            st[k].r.refill();
        }
        for j in 0..4 {
            for k in 0..STREAMS {
                out[o + j * STREAMS + k] = sym(&mut st[k], t);
            }
        }
        o += per_iter;
    }
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
