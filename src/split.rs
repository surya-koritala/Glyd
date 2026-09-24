//! Where the next block should end: at the byte statistics' seam, not
//! the 256 KB mark, when that pays. Before the parse, from the bytes
//! ahead (as zstd 1.5.7's pre-splitter): sixteen bytes of every 256 in
//! the window, in 16 KB segments, make a histogram each; a cut is taken
//! where the two parts coded on their own statistics beat the whole by
//! more than a few blocks' overhead, each part charged for describing
//! its table. Binaries with sections of different content (Silesia
//! mozilla) code 0.9% smaller; text and logs are unmoved. The caller
//! stops asking after a run of windows without a cut.
use crate::fixlog::log2_fast_q16;

/// Segment the window is sampled in; cuts fall on segment ends.
const SEG: usize = 16 << 10;
/// `RUN` bytes of every `STRIDE` are counted: a sixteenth of the bytes
/// from a quarter of the cache lines.
const STRIDE: usize = 256;
const RUN: usize = 16;
/// Windows without a cut after which the caller stops asking
/// (`Splitter`): on text and logs no window cuts, and the sampling is
/// then a few percent of the parse for nothing. Eight was too few:
/// mozilla's seams come every few MB and half were missed; at 32 every
/// one the always-on splitter finds is found.
pub const GIVE_UP_AFTER: u32 = 32;
/// A part is never shorter than this: every block costs the coder its
/// tables and the decoder its tails.
const MIN_PART: usize = 32 << 10;
/// What a cut must save, in 1/65536 bit of the sampled counts: a block's
/// framing and tables come to ~250 bytes; four times that, over the
/// sampling ratio.
const THRESHOLD_Q16: u64 = ((4 * 250 * 8 * RUN / STRIDE) as u64) << 16;

/// Order-0 cost of `h` coded with its own table, plus the table's
/// description (the MDL penalty: half a log of the count per symbol
/// used), in 1/65536 bit. Without the penalty two halves always look
/// cheaper than the whole by about that much, and every window cuts.
fn cost(h: &[u32; 256]) -> u64 {
    let t: u32 = h.iter().sum();
    if t == 0 {
        return 0;
    }
    let log_t = log2_fast_q16(t);
    let mut used = 0u64;
    let mut bits = 0u64;
    for &c in h.iter() {
        if c > 0 {
            used += 1;
            bits += c as u64 * (log_t - log2_fast_q16(c));
        }
    }
    bits + ((used.saturating_sub(1) * (32 - t.leading_zeros() as u64)) << 16) / 2
}

/// The length of the block starting at `start`: at most `max`, cut
/// short where the statistics change.
pub fn block_len(input: &[u8], start: usize, max: usize) -> usize {
    let n = (input.len() - start).min(max);
    if n < 2 * MIN_PART {
        return n;
    }
    let segs = (n / SEG).min(max / SEG);
    let mut h = [[0u32; 256]; 32];
    debug_assert!(segs <= h.len());
    let bytes = &input[start..start + n];
    for (s, hist) in h[..segs].iter_mut().enumerate() {
        let a = s * SEG;
        let b = if s + 1 == segs { n } else { a + SEG };
        let mut i = a;
        while i + RUN <= b {
            for &x in &bytes[i..i + RUN] {
                hist[x as usize] += 1;
            }
            i += STRIDE;
        }
    }
    let mut total = [0u32; 256];
    for hist in &h[..segs] {
        for (t, &c) in total.iter_mut().zip(hist.iter()) {
            *t += c;
        }
    }
    let whole = cost(&total);
    let mut best = (whole.saturating_sub(THRESHOLD_Q16), 0usize);
    let mut pre = [0u32; 256];
    for s in 0..segs - 1 {
        for (p, &c) in pre.iter_mut().zip(h[s].iter()) {
            *p += c;
        }
        let cut = (s + 1) * SEG;
        if cut < MIN_PART || n - cut < MIN_PART {
            continue;
        }
        let mut suf = [0u32; 256];
        for ((x, &t), &p) in suf.iter_mut().zip(total.iter()).zip(pre.iter()) {
            *x = t - p;
        }
        let c = cost(&pre) + cost(&suf);
        if c < best.0 {
            best = (c, cut);
        }
    }
    if best.1 != 0 { best.1 } else { n }
}

/// The block loop's view of the splitter: asks `block_len` until
/// `GIVE_UP_AFTER` windows in a row went uncut, then hands out full
/// blocks.
pub struct Splitter {
    misses: u32,
}

impl Splitter {
    pub fn new() -> Splitter {
        Splitter { misses: 0 }
    }

    /// The next block's length from `start`, at most `max`.
    pub fn next(&mut self, input: &[u8], start: usize, max: usize) -> usize {
        let n = (input.len() - start).min(max);
        if self.misses >= GIVE_UP_AFTER {
            return n;
        }
        let len = block_len(input, start, max);
        if len < n {
            self.misses = 0;
        } else {
            self.misses += 1;
        }
        len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuts_at_a_seam_and_nowhere_else() {
        // 128 KB of text-like bytes then 128 KB of noise: the cut lands
        // on the seam; uniform content is one block.
        let mut x = 5u64;
        let mut rnd = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x };
        let mut d = Vec::new();
        for _ in 0..(128 << 10) { d.push(b"etaoin shrdlu"[(rnd() % 13) as usize]); }
        for _ in 0..(128 << 10) { d.push(rnd() as u8); }
        let cut = block_len(&d, 0, 256 << 10);
        assert_eq!(cut, 128 << 10, "cut at {cut}");
        assert_eq!(block_len(&d, cut, 256 << 10), 128 << 10);
        let text: Vec<u8> = (0..(256 << 10)).map(|_| b"etaoin shrdlu"[(rnd() % 13) as usize]).collect();
        assert_eq!(block_len(&text, 0, 256 << 10), 256 << 10);
        assert_eq!(block_len(&text, 0, 100), 100);
    }
}
