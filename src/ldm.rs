//! Long-distance matcher: repeats of at least `MIN_LEN` bytes anywhere
//! in the last 128 MB, which the block-local finders, whose tables
//! cover 8 MB and forget most of that, cannot see. JSON event streams
//! and log archives repeat whole records across hundreds of megabytes;
//! zstd's `--long` gains 21% on an hour of GitHub events from the same
//! idea, at half its speed.
//!
//! One pass over the input (a unit) before it is parsed: content-defined
//! anchors (positions whose 4-byte hash has `ANCHOR_BITS` high zero
//! bits, one in 16 on average, found 16 positions at a time with NEON
//! or AVX2) put the hash of the 32 bytes following them in a table of
//! `TABLE_BITS` entries, later anchors replacing earlier ones. At each
//! anchor the table's entry is a candidate; its bytes are compared, the
//! match extended forward and back, and kept when it is at least
//! `MIN_LEN` long. The matches, in position order and without overlaps,
//! are handed to the parse (`Matches::cursor`), which may take a match
//! from any position inside it (a copy of a contiguous region is one
//! from any of its bytes).
//!
//! The pass runs at 1.2-2 GB/s on one core; the max level's parse at
//! 0.25-0.8 GB/s. So the max level gates it: after 4 MB and 16 MB it
//! stops on inputs whose repeats are too few, or too near, to pay
//! (`GATE_*`); the ultra level always runs it whole.
use crate::v7_format::MAX_OFFSET_BITS;

/// Shortest repeat worth a far offset (its code costs up to 26 extra bits).
pub const MIN_LEN: usize = 32;
const ANCHOR_BITS: u32 = 4;
/// Table entries (log2) for an input of up to 64 MB; larger inputs
/// get a slot per 16 bytes (`find` sizes the table to the input).
const TABLE_BITS: u32 = 22;
const TABLE_BITS_MAX: u32 = 25;
const POS_BITS: u32 = MAX_OFFSET_BITS;
const CHECK_BITS: u32 = 32 - POS_BITS;
const HASH_LEN: usize = 32;
/// Positions per anchor-gathering chunk.
const CHUNK: usize = 1 << 10;
/// The gated pass decides after this many bytes (or half the input) ...
const GATE_AT: usize = 16 << 20;
/// ... whether repeats at least this far back ...
const GATE_OFFSET: usize = 1 << 20;
/// ... cover at least 1/GATE_YIELD of what it scanned; earlier, after
/// `EARLY_AT` bytes, it gives up on data with almost no repeats at all
/// (media, Parquet), which the parse stores as it is.
const GATE_YIELD: usize = 32;
const EARLY_AT: usize = 4 << 20;
const EARLY_YIELD: usize = 64;

#[inline(always)]
unsafe fn prefetch(p: *const u32) {
    #[cfg(target_arch = "aarch64")]
    std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, preserves_flags, readonly));
    #[cfg(target_arch = "x86_64")]
    std::arch::x86_64::_mm_prefetch(p as *const i8, std::arch::x86_64::_MM_HINT_T0);
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = p;
}

/// A far match: `len` bytes at `start` equal the bytes `off` back.
#[derive(Clone, Copy, Debug)]
pub struct Far {
    pub start: u32,
    pub len: u32,
    pub off: u32,
}

/// The far matches of an input, in position order, non-overlapping.
pub struct Matches {
    pub list: Vec<Far>,
}

#[inline(always)]
fn h4(w: u32) -> u32 {
    w.wrapping_mul(0x9E37_79B1)
}

/// 1 when the 4 bytes `w` anchor: one position in 2^ANCHOR_BITS does.
#[inline(always)]
fn anchor(w: u32) -> u32 {
    (h4(w) >> (32 - ANCHOR_BITS) == 0) as u32
}

/// Hash of the `HASH_LEN` bytes at `p` (the caller guarantees them):
/// `TABLE_BITS` + `CHECK_BITS` bits, the four words mixed in parallel.
#[inline(always)]
unsafe fn h32p(p: *const u8) -> u32 {
    let a = std::ptr::read_unaligned(p as *const u64);
    let b = std::ptr::read_unaligned(p.add(8) as *const u64);
    let c = std::ptr::read_unaligned(p.add(16) as *const u64);
    let d = std::ptr::read_unaligned(p.add(24) as *const u64);
    let h = (a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
        .wrapping_add(c.wrapping_mul(0x1656_67B1_9E37_79F9) ^ d.wrapping_mul(0x27D4_EB2F_1656_67C5));
    let h = (h ^ (h >> 29)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (h >> (64 - TABLE_BITS_MAX - CHECK_BITS)) as u32
}

/// The anchors among 16 positions as a bit mask (bit k for `p + k`).
trait Scan {
    unsafe fn mask16(p: *const u8) -> u32;
}

/// The fallback where no vector step applies (x86 without AVX2).
#[allow(dead_code)]
struct Scalar;
impl Scan for Scalar {
    #[inline(always)]
    unsafe fn mask16(p: *const u8) -> u32 {
        let mut m = 0;
        for k in 0..16 {
            m |= anchor(std::ptr::read_unaligned(p.add(k) as *const u32)) << k;
        }
        m
    }
}

#[cfg(target_arch = "aarch64")]
struct Neon;
#[cfg(target_arch = "aarch64")]
impl Scan for Neon {
    #[inline(always)]
    unsafe fn mask16(p: *const u8) -> u32 {
        use std::arch::aarch64::*;
        // Vector j holds the words at p + 4i + j; lane i's bit is 4i + j.
        let v0 = vld1q_u8(p);
        let v1 = vld1q_u8(p.add(16));
        let k = vdupq_n_u32(0x9E37_79B1);
        let lim = vdupq_n_u32(1 << (32 - ANCHOR_BITS));
        let bits = vld1q_u32([1u32, 1 << 4, 1 << 8, 1 << 12].as_ptr());
        let a0 = vreinterpretq_u32_u8(v0);
        let a1 = vreinterpretq_u32_u8(vextq_u8::<1>(v0, v1));
        let a2 = vreinterpretq_u32_u8(vextq_u8::<2>(v0, v1));
        let a3 = vreinterpretq_u32_u8(vextq_u8::<3>(v0, v1));
        let m0 = vandq_u32(vcltq_u32(vmulq_u32(a0, k), lim), bits);
        let m1 = vandq_u32(vcltq_u32(vmulq_u32(a1, k), lim), vshlq_n_u32::<1>(bits));
        let m2 = vandq_u32(vcltq_u32(vmulq_u32(a2, k), lim), vshlq_n_u32::<2>(bits));
        let m3 = vandq_u32(vcltq_u32(vmulq_u32(a3, k), lim), vshlq_n_u32::<3>(bits));
        vaddvq_u32(vorrq_u32(vorrq_u32(m0, m1), vorrq_u32(m2, m3)))
    }
}

#[cfg(target_arch = "x86_64")]
struct Avx2;
#[cfg(target_arch = "x86_64")]
impl Scan for Avx2 {
    #[inline(always)]
    unsafe fn mask16(p: *const u8) -> u32 {
        use std::arch::x86_64::*;
        // Vector j holds the words at p + 4i + j (i < 4 in its low
        // half); lane i's bit is 4i + j.
        let k = _mm256_set1_epi32(0x9E37_79B1u32 as i32);
        let zero = _mm256_setzero_si256();
        let mut m = 0u32;
        for j in 0..4 {
            let v = _mm256_loadu_si256(p.add(j) as *const __m256i);
            let h = _mm256_srli_epi32::<{ 32 - ANCHOR_BITS as i32 }>(_mm256_mullo_epi32(v, k));
            let bits = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(h, zero))) as u32;
            m |= _pdep_u32(bits & 15, 0x1111 << j);
        }
        m
    }
}

/// The set bits of a byte, as bit indexes (the rest zero).
static BIT_INDEXES: [[u8; 8]; 256] = {
    let mut t = [[0u8; 8]; 256];
    let mut m = 0;
    while m < 256 {
        let (mut b, mut k) = (0, 0);
        while b < 8 {
            if m >> b & 1 == 1 {
                t[m][k] = b;
                k += 1;
            }
            b += 1;
        }
        m += 1;
    }
    t
};

/// Anchors of the positions `pos..end` (their table slots prefetched)
/// appended to `anchors`. The anchors' positions are first extracted
/// without a branch per anchor (a stray branch mispredicts once per
/// 16 bytes), then hashed in a loop of known length.
#[inline(always)]
unsafe fn gather<S: Scan>(src: *const u8, mut pos: usize, end: usize, table: *const u32, table_shift: u32, positions: &mut [u32; CHUNK + 16], anchors: &mut Vec<(u32, u32)>) {
    let mut count = 0usize;
    while pos + 16 <= end {
        let m = S::mask16(src.add(pos)) as usize;
        for half in 0..2 {
            let byte = (m >> (8 * half)) & 0xFF;
            let idx = BIT_INDEXES.get_unchecked(byte);
            let base = (pos + 8 * half) as u32;
            let out = positions.as_mut_ptr().add(count);
            for i in 0..8 {
                *out.add(i) = base + *idx.get_unchecked(i) as u32;
            }
            count += byte.count_ones() as usize;
        }
        pos += 16;
    }
    while pos < end {
        let w = std::ptr::read_unaligned(src.add(pos) as *const u32);
        if anchor(w) != 0 {
            *positions.get_unchecked_mut(count) = pos as u32;
            count += 1;
        }
        pos += 1;
    }
    for &pos in positions.get_unchecked(..count) {
        let w = std::ptr::read_unaligned(src.add(pos as usize) as *const u32);
        // A run of one byte anchors nowhere: its every position would
        // hash alike.
        if w != w.rotate_left(8) {
            let h = h32p(src.add(pos as usize));
            prefetch(table.add((h >> (CHECK_BITS + table_shift)) as usize));
            anchors.push((pos, h));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi2")]
unsafe fn gather_avx2(src: *const u8, pos: usize, end: usize, table: *const u32, table_shift: u32, positions: &mut [u32; CHUNK + 16], anchors: &mut Vec<(u32, u32)>) {
    gather::<Avx2>(src, pos, end, table, table_shift, positions, anchors)
}

/// The gather for this machine.
fn gather_fn() -> unsafe fn(*const u8, usize, usize, *const u32, u32, &mut [u32; CHUNK + 16], &mut Vec<(u32, u32)>) {
    #[cfg(target_arch = "aarch64")]
    {
        gather::<Neon>
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("bmi2") {
            return gather_avx2;
        }
        gather::<Scalar>
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        gather::<Scalar>
    }
}

impl Matches {
    /// Far matches over `input` (positions are absolute). Inputs under
    /// `MIN_LEN` * 2 yield none. With `gated`, the pass gives up early
    /// on inputs whose first `GATE_AT` bytes (or first half) hold under
    /// `GATE_YIELD` of repeats beyond `GATE_OFFSET`: the local finders
    /// see the rest, and the pass costs as much as the max level's parse
    /// (it stays on for JSON and logs, off for sources, media, Parquet).
    pub fn find(input: &[u8], gated: bool) -> Matches {
        let mut list = Vec::new();
        let n = input.len();
        if n < 2 * MIN_LEN + HASH_LEN {
            return Matches { list };
        }
        // An entry is a position's low `POS_BITS` bits under `CHECK_BITS`
        // of its hash: a mismatching check skips the (usually missing)
        // read of the candidate's bytes. The candidate is the nearest
        // position with those low bits (the window is 2^POS_BITS).
        // A slot per 16 bytes of input, at least the base size: the
        // table would otherwise forget the start of a 128 MB unit.
        let table_bits = (usize::BITS - (n / 16).leading_zeros()).clamp(TABLE_BITS, TABLE_BITS_MAX);
        let table_shift = TABLE_BITS_MAX - table_bits;
        let mut table = vec![u32::MAX; 1 << table_bits];
        let src = input.as_ptr();
        let end = n - HASH_LEN;
        let mut gate_at = if gated { (n / 2).min(GATE_AT) } else { usize::MAX };
        let mut early_at = if gated { (n / 2).min(EARLY_AT) } else { usize::MAX };
        let mut covered = 0usize; // no match may start before this
        let mut yield_all = 0usize; // bytes in matches
        let mut yield_far = 0usize; // bytes in matches beyond GATE_OFFSET
        // Anchors are gathered a chunk at a time and their table slots
        // prefetched (the table is 16 MB: a random slot per anchor, one
        // anchor in 16 bytes, is what the pass would otherwise wait on).
        let mut anchors: Vec<(u32, u32)> = Vec::with_capacity(CHUNK / 4);
        let mut positions = Box::new([0u32; CHUNK + 16]);
        let mut chunk_start = 0usize;
        let gather = gather_fn();
        // SAFETY: every load is at a position < end = n - HASH_LEN, so its
        // 4 (anchor) or 32 (hash) bytes are inside `input`; extensions
        // are bounded by `max` and `covered`.
        unsafe {
            while chunk_start < end {
                if chunk_start >= early_at {
                    if yield_all < early_at / EARLY_YIELD {
                        break;
                    }
                    early_at = usize::MAX;
                }
                if chunk_start >= gate_at {
                    if yield_far < gate_at / GATE_YIELD {
                        break;
                    }
                    gate_at = usize::MAX;
                }
                let chunk_end = (chunk_start + CHUNK).min(end);
                anchors.clear();
                gather(src, chunk_start.max(covered), chunk_end, table.as_ptr(), table_shift, &mut positions, &mut anchors);
                for &(pos, h) in &anchors {
                    let pos = pos as usize;
                    let slot = table.get_unchecked_mut((h >> (CHECK_BITS + table_shift)) as usize);
                    let entry = *slot;
                    let check = (h & ((1 << CHECK_BITS) - 1)) << POS_BITS;
                    *slot = check | (pos as u32 & ((1 << POS_BITS) - 1));
                    // Inside a match found from an earlier anchor: the
                    // table still learns it.
                    if pos < covered || entry >> POS_BITS != check >> POS_BITS {
                        continue;
                    }
                    let off = (pos - (entry & ((1 << POS_BITS) - 1)) as usize) & ((1 << POS_BITS) - 1);
                    if off < MIN_LEN || off > pos {
                        continue;
                    }
                    let c = pos - off;
                    let max = n - pos;
                    let mut len = 0usize;
                    while len + 8 <= max {
                        let a = std::ptr::read_unaligned(src.add(pos + len) as *const u64);
                        let b = std::ptr::read_unaligned(src.add(c + len) as *const u64);
                        if a != b {
                            len += ((a ^ b).trailing_zeros() / 8) as usize;
                            break;
                        }
                        len += 8;
                    }
                    if len + 8 > max {
                        while len < max && *src.add(pos + len) == *src.add(c + len) {
                            len += 1;
                        }
                    }
                    if len < MIN_LEN {
                        continue;
                    }
                    // Back to the match's true start, 8 bytes at a time.
                    let mut start = pos;
                    let mut sp = c;
                    let floor = covered.max(off);
                    while start >= floor + 8 {
                        let a = std::ptr::read_unaligned(src.add(start - 8) as *const u64);
                        let b = std::ptr::read_unaligned(src.add(sp - 8) as *const u64);
                        if a != b {
                            let k = ((a ^ b).leading_zeros() / 8) as usize;
                            start -= k;
                            sp -= k;
                            len += k;
                            break;
                        }
                        start -= 8;
                        sp -= 8;
                        len += 8;
                    }
                    while start > floor && *src.add(start - 1) == *src.add(sp - 1) {
                        start -= 1;
                        sp -= 1;
                        len += 1;
                    }
                    list.push(Far { start: start as u32, len: len as u32, off: off as u32 });
                    covered = start + len;
                    yield_all += len;
                    if off >= GATE_OFFSET {
                        yield_far += len;
                    }
                }
                chunk_start = chunk_end;
            }
        }
        Matches { list }
    }

    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    /// A cursor over the matches for a parse walking forward: one
    /// compare per position until the next match starts.
    pub fn cursor(&self, from: usize) -> Cursor<'_> {
        let i = self.list.partition_point(|m| (m.start + m.len) as usize <= from);
        let mut c = Cursor { list: &self.list, i, start: usize::MAX, end: 0, off: 0 };
        c.load();
        c
    }
}

pub struct Cursor<'a> {
    list: &'a [Far],
    i: usize,
    start: usize,
    end: usize,
    off: usize,
}

impl<'a> Cursor<'a> {
    #[inline(always)]
    fn load(&mut self) {
        match self.list.get(self.i) {
            Some(m) => {
                self.start = m.start as usize;
                self.end = self.start + m.len as usize;
                self.off = m.off as usize;
            }
            None => {
                self.start = usize::MAX;
                self.end = usize::MAX;
            }
        }
    }

    /// The far match covering `pos` as (length from `pos`, offset).
    #[inline(always)]
    pub fn at(&mut self, pos: usize) -> Option<(usize, usize)> {
        if pos < self.start {
            return None;
        }
        while pos >= self.end {
            self.i += 1;
            self.load();
            if pos < self.start {
                return None;
            }
        }
        Some((self.end - pos, self.off))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_far_repeats_and_verifies_them() {
        let mut x = 1u64;
        let mut rnd = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        };
        let block: Vec<u8> = (0..5000).map(|_| rnd()).collect();
        let mut data = block.clone();
        data.extend((0..(20 << 20)).map(|_| rnd()));
        data.extend_from_slice(&block);
        let m = Matches::find(&data, false);
        let far: Vec<&Far> = m.list.iter().filter(|f| f.off as usize > 10 << 20).collect();
        assert!(!far.is_empty(), "the 5000-byte repeat 20 MB back should be found: {:?}", m.list.len());
        for f in &m.list {
            let (s, l, o) = (f.start as usize, f.len as usize, f.off as usize);
            assert!(data[s..s + l] == data[s - o..s - o + l]);
        }
        let f = far[0];
        let mut c = m.cursor(0);
        assert_eq!(c.at(f.start as usize + 10).map(|(l, _)| l), Some(f.len as usize - 10));
        assert_eq!(c.at(f.start as usize + f.len as usize), None);
    }
}
