//! The tiered layout as glyd 0.20 to 0.27 packed it (glyd-v1 and glyd-v2 saves), kept to read those saves and to pack them
//! again in the layout of [`crate::pack`]. A pack of it has 0xFF in the top byte of `tiers[0]` (the library's decode
//! unpacks it, as before, but takes no product from it). Tier 3 holds three exponents and a digit 3 of it escapes to
//! the exponent's byte, which follows in the step's block.

use crate::pack::{by_count, gather, histogram, places, sign_mantissa, steps, weights_of, STEP};

/// The top byte of `tiers[0]` of a pack in this layout.
pub const TAG: u32 = 0xFF;

/// W in the tiered layout of glyd 0.27 and before.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tiered {
    pub rows: usize,
    pub cols: usize,
    /// [steps][1280]: a step's tier-1 digits (two words a lane, word-major), then its 1024 bytes.
    pub data: Vec<u8>,
    /// A step's escapes from block_base[step] to block_base[step + 1]: 128 bytes before the first, 256 after the last.
    pub blocks: Vec<u8>,
    /// [steps + 1].
    pub block_base: Vec<i32>,
    /// Tier k's three exponents in bytes 0-2 of word k (byte 3: 0xFF).
    pub tiers: [u32; 3],
}

/// A step in the tiered layout: its 1280 bytes into `d`, its block after `blocks`' end (the escapes' digits and
/// bytes); `esc` scratch.
fn tiered_step(u: &[u16; STEP], rank: &[u8; 256], d: &mut [u8], blocks: &mut Vec<u8>, esc: &mut [Vec<u8>; 3]) {
    // tier-1 digits: two words a lane, stored word-major ([2][32 lanes]); the tier-1 escapes' places, in order
    let (mut at, mut n1) = ([0u16; STEP], 0);
    for lane in 0..32 {
        let v: &[u16; 32] = u[lane * 32..lane * 32 + 32].try_into().unwrap();
        let x: [u8; 32] = std::array::from_fn(|j| rank[(v[j] >> 7) as u8 as usize]);
        let (mut w0, mut w1, mut escapes) = (0u32, 0u32, 0u32);
        for j in 0..16 {
            w0 |= (x[j].min(3) as u32) << (2 * j);
            w1 |= (x[16 + j].min(3) as u32) << (2 * j);
        }
        for (j, &r) in x.iter().enumerate() {
            escapes |= ((r >= 3) as u32) << j;
        }
        while escapes != 0 {
            at[n1] = (lane * 32) as u16 + escapes.trailing_zeros() as u16;
            n1 += 1;
            escapes &= escapes - 1;
        }
        d[lane * 4..lane * 4 + 4].copy_from_slice(&w0.to_le_bytes());
        d[128 + lane * 4..128 + lane * 4 + 4].copy_from_slice(&w1.to_le_bytes());
    }
    sign_mantissa(u, &mut d[256..]);
    // The step's block: tier-3 digits of its tier-2 escapes, then bytes of its tier-3 escapes; at its end the
    // tier-2 digits of its tier-1 escapes, words of 16 back from it.
    for e in esc.iter_mut() {
        e.clear();
    }
    for &q in &at[..n1] {
        let e = (u[q as usize] >> 7) as u8;
        let x = rank[e as usize];
        esc[0].push((x - 3).min(3));
        if x >= 6 {
            esc[1].push((x - 6).min(3));
            if x >= 9 {
                esc[2].push(e);
            }
        }
    }
    let (t1, t2, t3) = (n1, esc[1].len(), esc[2].len());
    let r0 = (2 * t2 + 7) >> 3;
    let size = r0 + t3 + 4 * t1.div_ceil(16);
    let start = blocks.len();
    blocks.resize(start + size, 0);
    let b = &mut blocks[start..];
    for (k, &d3) in esc[1].iter().enumerate() {
        b[k >> 2] |= d3 << (2 * (k & 3));
    }
    b[r0..r0 + t3].copy_from_slice(&esc[2]);
    for (k, &d2) in esc[0].iter().enumerate() {
        b[size - 4 * ((k >> 4) + 1) + ((k & 15) >> 2)] |= d2 << (2 * (k & 3));
    }
}

/// A step back from the tiered layout: its 1280 bytes `d`, its block `b`, the nine exponents by rank; None where
/// its block does not hold its escapes.
fn tiered_unstep(d: &[u8], b: &[u8], sym: &[u8; 9], out: &mut [u16; STEP], esc: &mut Vec<u16>) -> Option<()> {
    let mut exp = [0u8; STEP];
    let first = [sym[0], sym[1], sym[2], 0];
    let mut n1 = 0;
    esc.clear();
    esc.resize(STEP, 0);
    for lane in 0..32 {
        for h in 0..2 {
            let word = u32::from_le_bytes(d[h * 128 + lane * 4..h * 128 + lane * 4 + 4].try_into().unwrap());
            let e: &mut [u8; 16] = (&mut exp[lane * 32 + 16 * h..lane * 32 + 16 * h + 16]).try_into().unwrap();
            for (j, x) in e.iter_mut().enumerate() {
                *x = first[((word >> (2 * j)) & 3) as usize];
            }
            let mut escapes = word & (word >> 1) & 0x5555_5555; // digit 3: bit 2j
            while escapes != 0 {
                esc[n1] = (lane * 32 + 16 * h) as u16 + (escapes.trailing_zeros() / 2) as u16;
                n1 += 1;
                escapes &= escapes - 1;
            }
        }
    }
    // tier 2: each tier-1 escape's digit from the words at the block's end; tier 3: each tier-2 escape's from its
    // start; past it, the bytes after them
    let size = b.len();
    if 4 * n1.div_ceil(16) > size {
        return None;
    }
    let mut n2 = 0;
    for k in 0..n1 {
        let q = esc[k] as usize;
        let d2 = (b[size - 4 * ((k >> 4) + 1) + ((k & 15) >> 2)] >> (2 * (k & 3))) & 3;
        if d2 < 3 {
            exp[q] = sym[3 + d2 as usize];
        } else {
            esc[n2] = q as u16; // (the tier-2 escapes, in order, over the tier-1 ones already read)
            n2 += 1;
        }
    }
    let r0 = (2 * n2 + 7) >> 3;
    let mut t3 = 0;
    for (k, &q) in esc[..n2].iter().enumerate() {
        let q = q as usize;
        let d3 = (*b.get(k >> 2)? >> (2 * (k & 3))) & 3;
        exp[q] = if d3 < 3 {
            sym[6 + d3 as usize]
        } else {
            t3 += 1;
            *b.get(r0 + t3 - 1)?
        };
    }
    weights_of(&d[256..], &exp, out);
    Some(())
}

fn tiered(w: &[u16], rows: usize, cols: usize, check: bool) -> Option<Tiered> {
    crate::pack::check_shape(w, rows, cols);
    let order = by_count(&histogram(w));
    let tiers: [u32; 3] = std::array::from_fn(|k| order[3 * k] as u32 | (order[3 * k + 1] as u32) << 8 | (order[3 * k + 2] as u32) << 16 | 0xFF << 24);
    let mut rank = [9u8; 256];
    for (r, &e) in order[..9].iter().enumerate() {
        rank[e as usize] = r as u8;
    }
    let sym: [u8; 9] = std::array::from_fn(|r| order[r]);
    let n = rows * cols / STEP;
    let places = places(cols);
    let mut data = vec![0u8; n * 1280];
    let mut blocks = vec![0u8; 128];
    let mut block_base = Vec::with_capacity(n + 1);
    block_base.push(128i32);
    let (mut u, mut back) = ([0u16; STEP], [0u16; STEP]);
    let mut esc = [Vec::with_capacity(STEP), Vec::with_capacity(STEP), Vec::with_capacity(STEP)];
    let mut scratch = Vec::with_capacity(STEP);
    for (s, at) in steps(cols, 0, n) {
        gather(w, at, &places, &mut u);
        let start = blocks.len();
        tiered_step(&u, &rank, &mut data[s * 1280..(s + 1) * 1280], &mut blocks, &mut esc);
        block_base.push(i32::try_from(blocks.len()).expect("a matrix too large to pack"));
        if check {
            tiered_unstep(&data[s * 1280..(s + 1) * 1280], &blocks[start..], &sym, &mut back, &mut scratch)?;
            if back != u {
                return None;
            }
        }
    }
    blocks.resize(blocks.len() + 256, 0);
    assert!(blocks.len() < 1 << 31, "a matrix too large to pack");
    Some(Tiered { rows, cols, data, blocks, block_base, tiers })
}

/// W [rows, cols] in the tiered layout of glyd 0.27 and before, as its pack_mma made it (here for the tests of the
/// conversion; nothing packs it today).
pub fn pack(w: &[u16], rows: usize, cols: usize) -> Tiered {
    tiered(w, rows, cols, false).unwrap()
}

/// W back from the tiered layout, into out [rows, cols]; false where its buffers do not hold it (a step's block
/// past `blocks`, or shorter than its escapes).
pub fn unpack(p: &Tiered, out: &mut [u16]) -> bool {
    let n = p.rows * p.cols / STEP;
    if out.len() != p.rows * p.cols || p.data.len() != n * 1280 || p.block_base.len() != n + 1 || p.tiers.iter().any(|t| t >> 24 != TAG) {
        return false;
    }
    let sym: [u8; 9] = std::array::from_fn(|r| (p.tiers[r / 3] >> (8 * (r % 3))) as u8);
    let places = places(p.cols);
    let mut u = [0u16; STEP];
    let mut esc = Vec::with_capacity(STEP);
    for (s, at) in steps(p.cols, 0, n) {
        let (start, end) = (p.block_base[s] as usize, p.block_base[s + 1] as usize);
        if p.block_base[s] < 0
            || start > end
            || end > p.blocks.len()
            || tiered_unstep(&p.data[s * 1280..(s + 1) * 1280], &p.blocks[start..end], &sym, &mut u, &mut esc).is_none()
        {
            return false;
        }
        for (q, &v) in u.iter().enumerate() {
            out[at + places[q]] = v;
        }
    }
    true
}

/// Whether a pack's buffers are as its packer makes them wherever the GPU's decode reads by them: `data`
/// [steps][1280], `block_base` [steps + 1] from 128 on, rising, 256 bytes of `blocks` past the last; each step's
/// block exactly as long as its escapes take (the tier-3 digits and bytes from its start, the tier-2 digits' words
/// back from its end, the counts read from the step's digits): the decode then reads inside its buffers. Its escapes'
/// counts alone, not its weights (a decode does that).
pub fn blocks_hold(data: &[u8], blocks: &[u8], block_base: &[i32]) -> bool {
    let (n, b) = (data.len() / 1280, block_base);
    // (rising from 128 first: b[n] is then no negative number cast)
    if data.len() != n * 1280 || b.len() != n + 1 || b[0] < 128 || b.windows(2).any(|w| w[1] < w[0]) || b[n] as usize + 256 > blocks.len() {
        return false;
    }
    for s in 0..n {
        let d = &data[s * 1280..s * 1280 + 256];
        let (start, end) = (b[s] as usize, b[s + 1] as usize);
        let block = &blocks[start..end];
        let size = block.len();
        let t1: usize = d
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| {
                let w = u32::from_le_bytes(*w);
                (w & (w >> 1) & 0x5555_5555).count_ones() as usize
            })
            .sum();
        let digits = 4 * t1.div_ceil(16);
        if digits > size {
            return false;
        }
        let t2 = (0..t1).filter(|&k| (block[size - 4 * ((k >> 4) + 1) + ((k & 15) >> 2)] >> (2 * (k & 3))) & 3 == 3).count();
        let r0 = (2 * t2 + 7) >> 3;
        if r0 > size {
            return false;
        }
        let t3 = (0..t2).filter(|&k| (block[k >> 2] >> (2 * (k & 3))) & 3 == 3).count();
        if r0 + t3 + digits != size {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack::tests::weights;

    #[test]
    fn round_trips_and_holds() {
        for &(o, k, wild) in &[(64, 16, 0.0), (64, 64, 0.5), (128, 48, 0.02), (192, 1040, 0.001), (256, 4096, 0.1), (64, 16, 1.0)] {
            let w = weights(o * k, wild, (o * k) as u64);
            let t = pack(&w, o, k);
            assert!(tiered(&w, o, k, true).as_ref() == Some(&t), "v1 {o}x{k} {wild}: checked");
            let mut back = vec![0u16; o * k];
            assert!(unpack(&t, &mut back) && back == w, "v1 {o}x{k} {wild}");
            assert_eq!((t.data.len(), t.block_base.len()), (o * k / 1024 * 1280, o * k / 1024 + 1));
            assert!(blocks_hold(&t.data, &t.blocks, &t.block_base), "v1 {o}x{k} {wild}: its blocks");
        }
    }

    /// A pack whose data or blocks were changed where the decode reads by them: its blocks no longer hold it (and
    /// offsets that fall to a negative one are refused before any sum of them, in a debug build too).
    #[test]
    fn blocks_hold_their_escapes() {
        let w = weights(128 * 256, 0.2, 5);
        let t = pack(&w, 128, 256);
        let hold = |p: &Tiered| blocks_hold(&p.data, &p.blocks, &p.block_base);
        assert!(hold(&t));
        let mut more = t.clone();
        more.data[..256].fill(0xFF); // every weight of step 0 a tier-1 escape: its block too short for their digits
        assert!(!hold(&more));
        let mut shorter = t.clone();
        shorter.block_base[1] -= 1; // step 0's block a byte short, step 1's a byte long
        assert!(!hold(&shorter));
        let mut end = t.clone();
        end.blocks.truncate(end.blocks.len() - 1); // 255 bytes past the last block
        assert!(!hold(&end));
        assert!(!blocks_hold(&[0; 1280], &[0; 1024], &[128, -1])); // offsets falling to a negative one (no overflow)
    }
}
