//! Long-distance matcher: repeats of at least `MIN_LEN` bytes anywhere
//! in the last `MAX_WINDOW` (128 MB), which the block-local finders,
//! whose tables cover 8 MB, cannot see. JSON event streams and log
//! archives repeat whole records across hundreds of megabytes; zstd's
//! `--long` gains 23% on an hour of GitHub events from the same idea.
//!
//! One pass over the input (a unit) before it is parsed: content-defined
//! anchors (positions whose 4-byte hash has `ANCHOR_BITS` low zero bits,
//! one in 32 on average) put the hash of the 32 bytes following them in
//! a table of `TABLE_BITS` entries, later anchors replacing earlier
//! ones. At each anchor the table's entry is a candidate; its bytes are
//! compared, the match extended forward and back, and kept when it is
//! at least `MIN_LEN` long. The matches, in position order and without
//! overlaps, are handed to the parse (`Matches::at`), which may take a
//! match from any position inside it (a copy of a contiguous region is
//! one from any of its bytes).
use crate::v7_format::MAX_WINDOW;

/// Shortest repeat worth a far offset (its code costs up to 26 extra bits).
pub const MIN_LEN: usize = 32;
const ANCHOR_BITS: u32 = 4;
const TABLE_BITS: u32 = 22;
const HASH_LEN: usize = 32;

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

#[inline(always)]
fn h32(p: &[u8]) -> u32 {
    let mut h = 0x9E37_79B9_7F4A_7C15u64;
    for c in p[..HASH_LEN].chunks_exact(8) {
        h = (h ^ u64::from_le_bytes(c.try_into().unwrap())).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29);
    }
    (h >> (64 - TABLE_BITS)) as u32
}

impl Matches {
    /// Far matches over `input` (positions are absolute). Inputs under
    /// `MIN_LEN` * 2 yield none.
    pub fn find(input: &[u8]) -> Matches {
        let mut list = Vec::new();
        let n = input.len();
        if n < 2 * MIN_LEN + HASH_LEN {
            return Matches { list };
        }
        let mut table = vec![u32::MAX; 1 << TABLE_BITS];
        let mut pos = 0usize;
        let end = n - HASH_LEN;
        let mut covered = 0usize; // no match may start before this
        while pos < end {
            let w = u32::from_le_bytes(input[pos..pos + 4].try_into().unwrap());
            if h4(w) >> (32 - ANCHOR_BITS) != 0 {
                pos += 1;
                continue;
            }
            let h = h32(&input[pos..]) as usize;
            let cand = table[h];
            table[h] = pos as u32;
            if cand == u32::MAX {
                pos += 1;
                continue;
            }
            let c = cand as usize;
            let off = pos - c;
            if off >= MAX_WINDOW as usize || off < MIN_LEN {
                pos += 1;
                continue;
            }
            // Verify and extend.
            let mut len = 0usize;
            let max = n - pos;
            while len + 8 <= max {
                let a = u64::from_le_bytes(input[pos + len..pos + len + 8].try_into().unwrap());
                let b = u64::from_le_bytes(input[c + len..c + len + 8].try_into().unwrap());
                if a != b {
                    len += ((a ^ b).trailing_zeros() / 8) as usize;
                    break;
                }
                len += 8;
            }
            if len + 8 > max {
                while len < max && input[pos + len] == input[c + len] {
                    len += 1;
                }
            }
            let mut start = pos;
            let mut src = c;
            while start > covered && src > 0 && input[start - 1] == input[src - 1] {
                start -= 1;
                src -= 1;
                len += 1;
            }
            if len >= MIN_LEN {
                list.push(Far { start: start as u32, len: len as u32, off: off as u32 });
                covered = start + len;
                // No matches start inside this one, but its anchors still
                // go into the table (the newest copy of a region is the
                // one a later repeat should point at).
                let stop = (start + len).min(end);
                pos += 1;
                while pos < stop {
                    let w = u32::from_le_bytes(input[pos..pos + 4].try_into().unwrap());
                    if h4(w) >> (32 - ANCHOR_BITS) == 0 {
                        table[h32(&input[pos..]) as usize] = pos as u32;
                    }
                    pos += 1;
                }
                pos = stop.max(pos);
                continue;
            }
            pos += 1;
        }
        Matches { list }
    }

    /// The far match covering `pos`, as (length from `pos`, offset);
    /// `i` is the caller's cursor into the list, advanced past matches
    /// that end at or before `pos`.
    #[inline(always)]
    pub fn at(&self, i: &mut usize, pos: usize) -> Option<(usize, usize)> {
        while *i < self.list.len() {
            let m = self.list[*i];
            let end = m.start as usize + m.len as usize;
            if end <= pos {
                *i += 1;
                continue;
            }
            if (m.start as usize) <= pos {
                return Some((end - pos, m.off as usize));
            }
            return None;
        }
        None
    }

    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
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
        let m = Matches::find(&data);
        let far: Vec<&Far> = m.list.iter().filter(|f| f.off as usize > 10 << 20).collect();
        assert!(!far.is_empty(), "the 5000-byte repeat 20 MB back should be found: {:?}", m.list.len());
        for f in &m.list {
            let (s, l, o) = (f.start as usize, f.len as usize, f.off as usize);
            assert!(data[s..s + l] == data[s - o..s - o + l]);
        }
        let mut i = 0;
        let f = far[0];
        assert_eq!(m.at(&mut i, f.start as usize + 10).map(|(l, _)| l), Some(f.len as usize - 10));
    }
}
