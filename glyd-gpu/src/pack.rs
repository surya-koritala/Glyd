//! The mma layouts packed and decoded on the CPU, byte for byte as glyd.gpu's
//! pack_mma and pack_mma12 make them on the GPU (bindings/python/glyd/gpu/
//! kernels.py; glyd_gpu.h has the layouts).
//!
//! W [O, K] (O a multiple of 64, K of 16) goes as O K / 1024 warp steps of
//! 64 rows by 16 columns, row block by row block, a step's 1024 weights in
//! the order the tensor cores take their operand: lane l = 4g + t and its
//! weight i = 4n + 2jh + jl hold W[64 rb + 8n + g][16 ks + 8jh + 2t + jl].
//! In the tiered layout a weight's sign and mantissa is a byte (its mantissa
//! above the sign of its pair's other weight, i ^ 1), its exponent 2-bit
//! digits over three tiers of the matrix's ten commonest exponents (tier 3
//! holds four; digit 3 of tiers 1 and 2 goes on to the next tier; an exponent
//! past the tenth is a literal, an entry in the step's list). In the 12-bit
//! layout (split byte) a weight's low byte is kept as it is, its high byte
//! (sign, exponent >> 1) a 4-bit code: the sign and an offset 0-7 from the
//! matrix's base hb (0-120, the first of the 8 values of exponent >> 1 that
//! hold the most weights); any other weight's offset is 0 and the byte to XOR
//! into its high byte is in the step's exception list. [`pack_smallest`] takes
//! the tiered layout for a matrix, the 12-bit one where that is smaller or the
//! tiered one's steps mostly take the general decode. The tiered layout of
//! glyd 0.27 and before is [`crate::v1`]'s.

/// Weights a step (64 rows by 16 columns).
pub const STEP: usize = 1024;

/// Where each of a step's 1024 weights is in W, from the step's first
/// (row 64 rb, column 16 ks), for rows `k` weights long.
pub(crate) fn places(k: usize) -> Vec<usize> {
    (0..STEP)
        .map(|p| {
            let (lane, i) = (p / 32, p % 32);
            let (g, t, n, jh, jl) = (lane / 4, lane % 4, i / 4, (i / 2) % 2, i % 2);
            (8 * n + g) * k + 8 * jh + 2 * t + jl
        })
        .collect()
}

/// The exponents' counts.
pub fn histogram(w: &[u16]) -> [u64; 256] {
    let mut h = [[0u64; 256]; 4];
    let (quads, rest) = w.as_chunks::<4>();
    for q in quads {
        h[0][(q[0] >> 7) as u8 as usize] += 1;
        h[1][(q[1] >> 7) as u8 as usize] += 1;
        h[2][(q[2] >> 7) as u8 as usize] += 1;
        h[3][(q[3] >> 7) as u8 as usize] += 1;
    }
    for &v in rest {
        h[0][(v >> 7) as u8 as usize] += 1;
    }
    std::array::from_fn(|e| h[0][e] + h[1][e] + h[2][e] + h[3][e])
}

/// The exponents by count, most first, ties by the smaller exponent (torch's
/// stable argsort, descending).
pub(crate) fn by_count(h: &[u64; 256]) -> [u8; 256] {
    let mut e: [u8; 256] = std::array::from_fn(|i| i as u8);
    e.sort_by(|a, b| h[*b as usize].cmp(&h[*a as usize]).then(a.cmp(b)));
    e
}

/// A step's sign-and-mantissa bytes: weight i's in its half i / 16 of 512
/// bytes, at 16 lane + i % 16 (each 16-byte load of a warp: 512 contiguous
/// bytes).
pub(crate) fn sign_mantissa(u: &[u16; STEP], out: &mut [u8]) {
    for lane in 0..32 {
        let v = &u[lane * 32..lane * 32 + 32];
        for half in 0..2 {
            for (b, o) in out[half * 512 + lane * 16..half * 512 + lane * 16 + 16].iter_mut().enumerate() {
                let i = half * 16 + b;
                *o = (((v[i] & 0x7F) << 1) as u8) | (v[i ^ 1] >> 15) as u8;
            }
        }
    }
}

/// A step's weights, the inverse of `sign_mantissa`, with their exponents.
pub(crate) fn weights_of(sm: &[u8], exp: &[u8; STEP], out: &mut [u16; STEP]) {
    for lane in 0..32 {
        let (lo, hi) = (&sm[lane * 16..lane * 16 + 16], &sm[512 + lane * 16..512 + lane * 16 + 16]);
        for i in 0..32 {
            let (b, pair) = if i < 16 { (lo[i], lo[i ^ 1]) } else { (hi[i - 16], hi[(i ^ 1) - 16]) };
            out[lane * 32 + i] = ((pair & 1) as u16) << 15 | (exp[lane * 32 + i] as u16) << 7 | (b >> 1) as u16;
        }
    }
}

pub(crate) fn check_shape(w: &[u16], rows: usize, cols: usize) {
    assert!(rows.is_multiple_of(64) && cols.is_multiple_of(16) && w.len() == rows * cols, "a matrix this layout does not take");
}

/// A step's weights gathered from W (its first at `at`).
pub(crate) fn gather(w: &[u16], at: usize, places: &[usize], u: &mut [u16; STEP]) {
    for (x, &p) in u.iter_mut().zip(places) {
        *x = w[at + p];
    }
}

/// Steps [s0, s1) of a matrix `cols` wide: each step's index and its first weight's place in W.
pub(crate) fn steps(cols: usize, s0: usize, s1: usize) -> impl Iterator<Item = (usize, usize)> {
    let ks = cols / 16;
    (s0..s1).map(move |s| (s, (s / ks) * 64 * cols + (s % ks) * 16))
}

/// The top byte of `tiers[0]` of a pack in this layout (the layout of glyd 0.27 and before has 0xFF there: [`crate::v1`]).
pub const TIERED_TAG: u32 = 2;

/// Tier-1 escapes a step's general decode takes past (the other decode reads their digits, a word a lane, 32 words).
pub const HOT_ESCAPES: usize = 512;
/// Literals a step has, at most, where its decode is the other one's (one entry a lane).
pub const HOT_LITERALS: usize = 32;
/// The bits of a step's block the other decode reads through the shuffles of its first 128 bytes, less one word.
pub const HOT_WINDOW: usize = 992;
/// The share of a matrix's steps that may take the general decode: past it the 12-bit layout holds the matrix.
pub const COLD_SHARE: (usize, usize) = (1, 4);

/// W in the tiered layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tiered {
    pub rows: usize,
    pub cols: usize,
    /// [steps][1280]: a step's tier-1 digits (two words a lane, word-major), then its 1024 bytes.
    pub data: Vec<u8>,
    /// A step's block from block_start(block_base[step]) to block_start(block_base[step + 1]): 128 bytes before the
    /// first, 256 after the last. A block: the tier-3 digits of its steps' weights of ranks 6-9 (2 bits each, from its
    /// start, to a byte), the list of its literals (ranks 10 and on: an entry is its place among the step's tier-1
    /// escapes, 10 bits, and its exponent's byte, 18 bits in all, packed from a byte's start, low bits first), and
    /// the tier-2 digits of its tier-1 escapes (words of 16, the c-th word back from its end).
    pub blocks: Vec<u8>,
    /// [steps + 1]: a step's block start, with bit 31 of its entry set where the step takes the general decode
    /// ([`is_cold`]); the last entry is the end of the last block.
    pub block_base: Vec<i32>,
    /// Tier k's three exponents in bytes 0-2 of word k, by count, most first; byte 3: [`TIERED_TAG`] of word 0, 0xFF
    /// of word 1, tier 3's fourth exponent of word 2.
    pub tiers: [u32; 3],
}

/// The ten exponents by rank of a pack's `tiers`: tier 1's three, tier 2's, tier 3's four.
pub fn tier_symbols(tiers: &[u32; 3]) -> [u8; 10] {
    std::array::from_fn(|r| if r < 9 { (tiers[r / 3] >> (8 * (r % 3))) as u8 } else { (tiers[2] >> 24) as u8 })
}

/// A block's start in `blocks`: a `block_base` entry without its flag.
pub fn block_start(x: i32) -> usize {
    (x & 0x7FFF_FFFF) as usize
}

/// Whether a step takes the general decode, from where its block starts and what it holds: t1 tier-1 escapes (the
/// weights of rank 3 and on), t2 tier-3 digits (ranks 6-9), nl literals (rank 10 and on). The other decode (the one
/// the packer's flag keeps) takes a step with at most 512 escapes, 32 literals, its tier-3 digits and its last
/// literal's entry inside the first 124 bytes of the 128 the block's first word begins (o: its bit offset in that word).
pub fn is_cold(start: usize, t1: usize, t2: usize, nl: usize) -> bool {
    let (o, r0) = (8 * (start & 3), (2 * t2 + 7) >> 3);
    t1 > HOT_ESCAPES || nl > HOT_LITERALS || o + 2 * t2 + 32 >= HOT_WINDOW || (nl > 0 && o + 8 * r0 + 18 * (nl - 1) >= HOT_WINDOW)
}

/// The steps of a pack that take the general decode.
pub fn cold_steps(block_base: &[i32]) -> usize {
    block_base.iter().take(block_base.len().saturating_sub(1)).filter(|&&x| x < 0).count()
}

/// The tiered layout's arrays and words, in bytes (as glyd.gpu's Mma.nbytes counts them).
pub fn tiered_bytes(p: &Tiered) -> u64 {
    (p.data.len() + p.blocks.len() + 4 * p.block_base.len() + 12) as u64
}

/// A step's scratch.
struct Scratch {
    d2: Vec<u8>,
    d3: Vec<u8>,
    lit: Vec<u32>,
}

/// Bits [at, at + n) (n up to 25) of `b`, low bits first (bytes past its end read as 0).
fn bits(b: &[u8], at: usize, n: u32) -> u32 {
    let byte = |i: usize| b.get(at / 8 + i).copied().unwrap_or(0) as u64;
    ((byte(0) | byte(1) << 8 | byte(2) << 16 | byte(3) << 24) >> (at % 8)) as u32 & ((1 << n) - 1)
}

/// A step in the tiered layout: its 1280 bytes into `d`, its block after `blocks`' end; its counts of tier-1 escapes,
/// tier-3 digits and literals.
fn tiered_step(u: &[u16; STEP], rank: &[u8; 256], d: &mut [u8], blocks: &mut Vec<u8>, s: &mut Scratch) -> (usize, usize, usize) {
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
    // each tier-1 escape's tier-2 digit (a literal's: 0), the tier-3 digit of the weights of ranks 6-9, the list's entries
    s.d2.clear();
    s.d3.clear();
    s.lit.clear();
    for (k, &q) in at[..n1].iter().enumerate() {
        let e = (u[q as usize] >> 7) as u8;
        let x = rank[e as usize];
        if x >= 10 {
            s.d2.push(0);
            s.lit.push(k as u32 | (e as u32) << 10);
        } else {
            s.d2.push((x - 3).min(3));
            if x >= 6 {
                s.d3.push(x - 6);
            }
        }
    }
    let (t2, nl) = (s.d3.len(), s.lit.len());
    let r0 = (2 * t2 + 7) >> 3;
    let size = r0 + (18 * nl).div_ceil(8) + 4 * n1.div_ceil(16);
    let start = blocks.len();
    blocks.resize(start + size, 0);
    let b = &mut blocks[start..];
    for (k, &d3) in s.d3.iter().enumerate() {
        b[k >> 2] |= d3 << (2 * (k & 3));
    }
    for (j, &x) in s.lit.iter().enumerate() {
        let p = 8 * r0 + 18 * j;
        let v = (x as u64) << (p % 8);
        for i in 0..4 {
            let byte = (v >> (8 * i)) as u8;
            if byte != 0 {
                b[p / 8 + i] |= byte;
            }
        }
    }
    for (k, &d2) in s.d2.iter().enumerate() {
        b[size - 4 * ((k >> 4) + 1) + ((k & 15) >> 2)] |= d2 << (2 * (k & 3));
    }
    (n1, t2, nl)
}

/// A step back from the tiered layout: its 1280 bytes `d`, its block `b`, the ten exponents by rank; None where its
/// block is not what a packer makes of the step's escapes (a decode of any step, whatever its flag).
fn tiered_unstep(d: &[u8], b: &[u8], sym: &[u8; 10], out: &mut [u16; STEP], at: &mut Vec<u16>) -> Option<()> {
    let mut exp = [0u8; STEP];
    let first = [sym[0], sym[1], sym[2], 0];
    let mut n1 = 0;
    at.clear();
    at.resize(STEP, 0);
    for lane in 0..32 {
        for h in 0..2 {
            let word = u32::from_le_bytes(d[h * 128 + lane * 4..h * 128 + lane * 4 + 4].try_into().unwrap());
            let e: &mut [u8; 16] = (&mut exp[lane * 32 + 16 * h..lane * 32 + 16 * h + 16]).try_into().unwrap();
            for (j, x) in e.iter_mut().enumerate() {
                *x = first[((word >> (2 * j)) & 3) as usize];
            }
            let mut escapes = word & (word >> 1) & 0x5555_5555; // digit 3: bit 2j
            while escapes != 0 {
                at[n1] = (lane * 32 + 16 * h) as u16 + (escapes.trailing_zeros() / 2) as u16;
                n1 += 1;
                escapes &= escapes - 1;
            }
        }
    }
    // tier 2: each tier-1 escape's digit from the words at the block's end; tier 3: each tier-2 escape's digit from the
    // block's start; the list after them: each literal its escape's exponent
    let size = b.len();
    let words = 4 * n1.div_ceil(16);
    if words > size {
        return None;
    }
    let mut t2 = 0;
    for k in 0..n1 {
        let q = at[k] as usize;
        let d2 = (b[size - 4 * ((k >> 4) + 1) + ((k & 15) >> 2)] >> (2 * (k & 3))) & 3;
        if d2 < 3 {
            exp[q] = sym[3 + d2 as usize];
        } else {
            t2 += 1;
        }
    }
    let r0 = (2 * t2 + 7) >> 3;
    let list = (size - words).checked_sub(r0)?;
    let mut j = 0;
    for k in 0..n1 {
        if (b[size - 4 * ((k >> 4) + 1) + ((k & 15) >> 2)] >> (2 * (k & 3))) & 3 == 3 {
            exp[at[k] as usize] = sym[6 + ((*b.get(j >> 2)? >> (2 * (j & 3))) & 3) as usize];
            j += 1;
        }
    }
    let nl = 8 * list / 18;
    if (18 * nl).div_ceil(8) != list {
        return None;
    }
    for j in 0..nl {
        let x = bits(&b[..r0 + list], 8 * r0 + 18 * j, 18);
        let k = (x & 0x3FF) as usize;
        if k >= n1 {
            return None;
        }
        exp[at[k] as usize] = (x >> 10) as u8;
    }
    weights_of(&d[256..], &exp, out);
    Some(())
}

fn tiered(w: &[u16], rows: usize, cols: usize, check: bool) -> Option<Tiered> {
    let n = rows * cols / STEP;
    tiered_on(w, rows, cols, check, if n >= PAR_STEPS { std::thread::available_parallelism().map_or(1, |t| t.get()).min(32) } else { 1 })
}

/// `tiered` on `threads` threads, each a run of the steps: the same bytes whatever the count.
fn tiered_on(w: &[u16], rows: usize, cols: usize, check: bool, threads: usize) -> Option<Tiered> {
    check_shape(w, rows, cols);
    let order = by_count(&histogram(w));
    let tiers: [u32; 3] = [
        order[0] as u32 | (order[1] as u32) << 8 | (order[2] as u32) << 16 | TIERED_TAG << 24,
        order[3] as u32 | (order[4] as u32) << 8 | (order[5] as u32) << 16 | 0xFF << 24,
        order[6] as u32 | (order[7] as u32) << 8 | (order[8] as u32) << 16 | (order[9] as u32) << 24,
    ];
    let mut rank = [10u8; 256];
    for (r, &e) in order[..10].iter().enumerate() {
        rank[e as usize] = r as u8;
    }
    let sym: [u8; 10] = std::array::from_fn(|r| order[r]);
    let n = rows * cols / STEP;
    let places = places(cols);
    let mut data = vec![0u8; n * 1280];
    // the steps in runs, a thread each where the matrix is large (a mixture of experts' stacked experts, an embedding): a step's block is the same bytes
    // wherever it lands, so the runs' blocks are joined after, and each step's general-decode flag, which reads where its block starts, is set then
    let per = n.div_ceil(threads.max(1)).max(1);
    // (a run's blocks, and each of its steps' start in them and counts)
    type Run = (Vec<u8>, Vec<(usize, usize, usize, usize)>);
    let run = |s0: usize, d: &mut [u8]| -> Option<Run> {
        let (mut blocks, mut marks) = (Vec::new(), Vec::with_capacity(per));
        let (mut u, mut back) = ([0u16; STEP], [0u16; STEP]);
        let mut scratch = Scratch { d2: Vec::with_capacity(STEP), d3: Vec::with_capacity(STEP), lit: Vec::with_capacity(STEP) };
        let mut at = Vec::with_capacity(STEP);
        for (s, first) in steps(cols, s0, s0 + d.len() / 1280) {
            gather(w, first, &places, &mut u);
            let start = blocks.len();
            let ds = &mut d[(s - s0) * 1280..(s - s0 + 1) * 1280];
            let (t1, t2, nl) = tiered_step(&u, &rank, ds, &mut blocks, &mut scratch);
            marks.push((start, t1, t2, nl));
            if check {
                tiered_unstep(ds, &blocks[start..], &sym, &mut back, &mut at)?;
                if back != u {
                    return None;
                }
            }
        }
        Some((blocks, marks))
    };
    let runs: Vec<Option<Run>> = if per >= n {
        vec![run(0, &mut data)]
    } else {
        std::thread::scope(|sc| {
            let hs: Vec<_> = data.chunks_mut(per * 1280).enumerate().map(|(k, d)| sc.spawn(move || run(k * per, d))).collect();
            hs.into_iter().map(|h| h.join().expect("a packing thread")).collect()
        })
    };
    let mut blocks = vec![0u8; 128];
    let mut block_base = Vec::with_capacity(n + 1);
    for r in runs {
        let (b, marks) = r?;
        let base = blocks.len();
        for (start, t1, t2, nl) in marks {
            let at = base + start;
            block_base.push(i32::try_from(at).expect("a matrix too large to pack") | if is_cold(at, t1, t2, nl) { i32::MIN } else { 0 });
        }
        blocks.extend_from_slice(&b);
    }
    block_base.push(i32::try_from(blocks.len()).expect("a matrix too large to pack"));
    blocks.resize(blocks.len() + 256, 0);
    assert!(blocks.len() < 1 << 31, "a matrix too large to pack");
    Some(Tiered { rows, cols, data, blocks, block_base, tiers })
}

/// Steps from which `tiered` packs a matrix on several threads (about 256 MB of bf16): below it the callers' own threads, a matrix each, are enough.
const PAR_STEPS: usize = 1 << 17;

/// W [rows, cols] in the tiered layout, as pack_mma makes it.
pub fn pack_tiered(w: &[u16], rows: usize, cols: usize) -> Tiered {
    tiered(w, rows, cols, false).unwrap()
}

/// The same, each step decoded back as it is made and compared with its weights (the check glyd.gpu's pack makes
/// of each pack on the GPU): None where one is not its weights, bit for bit.
pub fn pack_tiered_checked(w: &[u16], rows: usize, cols: usize) -> Option<Tiered> {
    tiered(w, rows, cols, true)
}

/// W back from the tiered layout, into out [rows, cols] by the general decode of every step; false where its buffers
/// do not hold it (a step's block past `blocks`, or not what its escapes take).
/// The compact tiered layout's tag (tiers[0]'s top byte): weights of three fraction bits (Candidate A's values), the
/// tiered layout with each sign-and-mantissa byte kept as a nibble (its bits 5-7 and 0): a step's 256 bytes of digits
/// as they are, then its 512 nibble bytes, two bytes each in order (768 bytes a step; glyd_gpu.cu's Compact).
pub const COMPACT_TAG: u32 = 3;
pub const COMPACT_STEP_BYTES: usize = 768;

/// bf16 rounded to three fraction bits, half to even (Candidate A, as the quality screens' round_candidate_a rounds):
/// a finite word rounded past the largest is the largest (0x7F70 with its sign); a non-finite word as it is.
pub fn round_a(w: u16) -> u16 {
    if w & 0x7F80 == 0x7F80 {
        return w;
    }
    let r = ((w as u32 + 7 + ((w as u32 >> 4) & 1)) & 0xFFF0) as u16;
    if r & 0x7FFF >= 0x7F80 {
        (w & 0x8000) | 0x7F70
    } else {
        r
    }
}

fn nibble(byte: u8) -> u8 {
    ((byte >> 4) & 14) | (byte & 1)
}
fn denibble(n: u8) -> u8 {
    ((n & 14) << 4) | (n & 1)
}

/// W, its finite weights of three fraction bits (round_a's values), in the compact tiered layout, checked by
/// decoding; None where a finite weight is not such a value. A matrix holding a non-finite weight whose low bits are
/// set stays in the tiered layout (tiers[0]'s tag TIERED_TAG).
pub fn pack_compact_checked(w: &[u16], rows: usize, cols: usize) -> Option<Tiered> {
    if w.iter().any(|&x| x & 0x7F80 != 0x7F80 && x & 15 != 0) {
        return None;
    }
    let base = pack_tiered_checked(w, rows, cols)?;
    if !w.iter().all(|&x| x & 15 == 0) {
        return Some(base);
    }
    let mut data = Vec::with_capacity(base.data.len() / 1280 * COMPACT_STEP_BYTES);
    for s in base.data.chunks_exact(1280) {
        data.extend_from_slice(&s[..256]);
        for pair in s[256..].chunks_exact(2) {
            if (pair[0] | pair[1]) & 30 != 0 {
                return None;
            }
            data.push(nibble(pair[0]) | nibble(pair[1]) << 4);
        }
    }
    let mut tiers = base.tiers;
    tiers[0] = (tiers[0] & 0x00FF_FFFF) | COMPACT_TAG << 24;
    let p = Tiered { data, tiers, ..base };
    let mut back = vec![0u16; w.len()];
    (expand_compact(&p).is_some_and(|t| unpack_tiered(&t, &mut back)) && back == w).then_some(p)
}

/// A compact pack in the tiered layout (its bytes restored, its tag TIERED_TAG); None where its data is not its shape's.
pub fn expand_compact(p: &Tiered) -> Option<Tiered> {
    let steps = p.rows * p.cols / STEP;
    if p.tiers[0] >> 24 != COMPACT_TAG || p.data.len() != steps * COMPACT_STEP_BYTES {
        return None;
    }
    let mut data = Vec::with_capacity(steps * 1280);
    for s in p.data.chunks_exact(COMPACT_STEP_BYTES) {
        data.extend_from_slice(&s[..256]);
        for &v in &s[256..] {
            data.extend_from_slice(&[denibble(v), denibble(v >> 4)]);
        }
    }
    let mut tiers = p.tiers;
    tiers[0] = (tiers[0] & 0x00FF_FFFF) | TIERED_TAG << 24;
    Some(Tiered { rows: p.rows, cols: p.cols, data, blocks: p.blocks.clone(), block_base: p.block_base.clone(), tiers })
}

pub fn unpack_tiered(p: &Tiered, out: &mut [u16]) -> bool {
    let n = p.rows * p.cols / STEP;
    if out.len() != p.rows * p.cols || p.data.len() != n * 1280 || p.block_base.len() != n + 1 || p.tiers[0] >> 24 != TIERED_TAG {
        return false;
    }
    let sym = tier_symbols(&p.tiers);
    let places = places(p.cols);
    let mut u = [0u16; STEP];
    let mut at = Vec::with_capacity(STEP);
    for (s, first) in steps(p.cols, 0, n) {
        let (start, end) = (block_start(p.block_base[s]), block_start(p.block_base[s + 1]));
        if start > end || end > p.blocks.len() || tiered_unstep(&p.data[s * 1280..(s + 1) * 1280], &p.blocks[start..end], &sym, &mut u, &mut at).is_none() {
            return false;
        }
        for (q, &v) in u.iter().enumerate() {
            out[first + places[q]] = v;
        }
    }
    true
}

/// Whether a tiered pack's buffers are as its packer makes them wherever the GPU's decode reads by them: `data`
/// [steps][1280], `block_base` [steps + 1] from 128 on (the flags apart), rising, 256 bytes of `blocks` past the
/// last; each step's block exactly as long as its escapes and literals take (the tier-3 digits from its start, the
/// list after them, the tier-2 digits' words back from its end, the counts read from the step's digits): the
/// decode then reads inside its buffers. Its counts alone, not its weights (a decode does that), nor its flag.
pub fn tiered_blocks_hold(data: &[u8], blocks: &[u8], block_base: &[i32]) -> bool {
    let (n, b) = (data.len() / 1280, block_base);
    if data.len() != n * 1280
        || b.len() != n + 1
        || b[n] < 0
        || b[0] & 0x7FFF_FFFF < 128
        || b.windows(2).any(|w| block_start(w[1]) < block_start(w[0]))
        || b[n] as usize + 256 > blocks.len()
    {
        return false;
    }
    for s in 0..n {
        let d = &data[s * 1280..s * 1280 + 256];
        let (start, end) = (block_start(b[s]), block_start(b[s + 1]));
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
        let Some(list) = (size - digits).checked_sub(r0) else { return false };
        if (18 * (8 * list / 18)).div_ceil(8) != list {
            return false;
        }
    }
    true
}

/// W [rows, cols] in the 12-bit layout (split byte).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Twelve {
    pub rows: usize,
    pub cols: usize,
    /// [steps][1536]: a step's codes (four words a lane, lane-major: word q holds weights 8q to 8q + 7, byte j weight
    /// 8q + j's sign in bit 7 and offset in bits 0-2, weight 8q + 4 + j's sign in bit 3, and weight 8q + 4 + (j + 1)
    /// mod 4's offset in bits 4-6), then its 1024 low bytes (weight i of lane l at 512 (i / 16) + 16 l + i mod 16).
    pub data: Vec<u8>,
    /// The exceptions, a weight's place in its step (bits 0-9) and the byte to XOR into its high byte, hb ^
    /// (exponent >> 1) (bits 16-23), a step's from exc_base[step] to exc_base[step + 1]; zeros after, to a multiple of
    /// 4 (1 at least).
    pub exc: Vec<i32>,
    /// [steps + 1].
    pub exc_base: Vec<i32>,
    /// The C API's words: the base hb (0-120) in each byte of the first, then 0, 0, 0 (`twelve_words`).
    pub sym: [u32; 4],
}

impl Twelve {
    /// Its base hb (0-120), the first of the 8 values of exponent >> 1 its offsets count from.
    pub fn hb(&self) -> u8 {
        self.sym[0] as u8
    }
}

/// The C API's four words of a 12-bit pack whose base is `hb`.
pub fn twelve_words(hb: u8) -> [u32; 4] {
    [hb as u32 * 0x0101_0101, 0, 0, 0]
}

/// The 12-bit layout's base: the smallest x0 in 0..=120 whose window of 8 values of exponent >> 1 (x0 to x0 + 7) holds
/// the most weights (pack_mma12's argmax: the first maximum).
fn twelve_base(h: &[u64; 256]) -> u8 {
    let pairs: [u64; 128] = std::array::from_fn(|x| h[2 * x] + h[2 * x + 1]);
    let window = |x0: usize| pairs[x0..x0 + 8].iter().sum::<u64>();
    (0..=120).fold(0, |best, x0| if window(x0) > window(best) { x0 } else { best }) as u8
}

/// A 12-bit step decoded: each weight's high byte its sign and hb + its offset, an exception's then XORed with its
/// entry's byte (bits 16-23; its place in the step, bits 0-9), above its low byte.
fn twelve_unstep(d: &[u8], exc: &[i32], hb: u8, out: &mut [u16; STEP]) {
    let mut high = [0u8; STEP];
    for lane in 0..32 {
        for q in 0..4 {
            let word = u32::from_le_bytes(d[lane * 16 + q * 4..lane * 16 + q * 4 + 4].try_into().unwrap());
            for j in 0..4 {
                let (b, rotated) = ((word >> (8 * j)) as u8, (word >> (8 * ((j + 3) % 4) + 4)) as u8);
                high[lane * 32 + 8 * q + j] = (b & 0x80) | (hb + (b & 7));
                high[lane * 32 + 8 * q + 4 + j] = (b & 8) << 4 | (hb + (rotated & 7));
            }
        }
    }
    for &x in exc {
        high[(x & 1023) as usize] ^= (x >> 16) as u8;
    }
    for lane in 0..32 {
        for i in 0..32 {
            out[lane * 32 + i] = (high[lane * 32 + i] as u16) << 8 | d[512 + (i / 16) * 512 + lane * 16 + i % 16] as u16;
        }
    }
}

fn twelve(w: &[u16], rows: usize, cols: usize, check: bool) -> Option<Twelve> {
    check_shape(w, rows, cols);
    let hb = twelve_base(&histogram(w));
    // a weight's code (sign in bit 3, offset in bits 0-2) and whether it is an exception
    let code = |v: u16| -> (u32, bool) {
        let off = ((v >> 8) & 0x7F) as i32 - hb as i32;
        let esc = !(0..=7).contains(&off);
        (((v >> 15) as u32) << 3 | if esc { 0 } else { off as u32 }, esc)
    };
    let n = rows * cols / STEP;
    let places = places(cols);
    let mut data = vec![0u8; n * 1536];
    let (mut exc, mut exc_base) = (Vec::new(), Vec::with_capacity(n + 1));
    exc_base.push(0i32);
    let (mut u, mut back) = ([0u16; STEP], [0u16; STEP]);
    for (s, at) in steps(cols, 0, n) {
        gather(w, at, &places, &mut u);
        let first = exc.len();
        let d = &mut data[s * 1536..(s + 1) * 1536];
        for lane in 0..32 {
            let v = &u[lane * 32..lane * 32 + 32];
            for q in 0..4 {
                let mut word = 0u32;
                for j in 0..4 {
                    let ((a, _), (b, _)) = (code(v[8 * q + j]), code(v[8 * q + 4 + j]));
                    word |= ((a >> 3) << 7 | (a & 7)) << (8 * j) | (b >> 3) << (8 * j + 3) | (b & 7) << (8 * ((j + 3) % 4) + 4);
                }
                d[lane * 16 + q * 4..lane * 16 + q * 4 + 4].copy_from_slice(&word.to_le_bytes());
            }
            for (i, &x) in v.iter().enumerate() {
                d[512 + (i / 16) * 512 + lane * 16 + i % 16] = x as u8;
                if code(x).1 {
                    exc.push((lane * 32 + i) as i32 | ((hb ^ ((x >> 8) & 0x7F) as u8) as i32) << 16);
                }
            }
        }
        exc_base.push(i32::try_from(exc.len()).expect("a matrix too large to pack"));
        if check {
            twelve_unstep(&data[s * 1536..(s + 1) * 1536], &exc[first..], hb, &mut back);
            if back != u {
                return None;
            }
        }
    }
    exc.resize(exc.len() + 4 - exc.len() % 4, 0);
    Some(Twelve { rows, cols, data, exc, exc_base, sym: twelve_words(hb) })
}

/// W [rows, cols] in the 12-bit layout, as pack_mma12 makes it.
pub fn pack_twelve(w: &[u16], rows: usize, cols: usize) -> Twelve {
    twelve(w, rows, cols, false).unwrap()
}

/// The same, each step decoded back as it is made and compared with its weights: None where one is not its
/// weights, bit for bit.
pub fn pack_twelve_checked(w: &[u16], rows: usize, cols: usize) -> Option<Twelve> {
    twelve(w, rows, cols, true)
}

/// W back from the 12-bit layout, into out [rows, cols]; false where its buffers or words do not hold it.
pub fn unpack_twelve(p: &Twelve, out: &mut [u16]) -> bool {
    let n = p.rows * p.cols / STEP;
    if out.len() != p.rows * p.cols || p.data.len() != n * 1536 || p.exc_base.len() != n + 1 || p.hb() > 120 || p.sym != twelve_words(p.hb()) {
        return false;
    }
    let places = places(p.cols);
    let mut u = [0u16; STEP];
    for (s, at) in steps(p.cols, 0, n) {
        let (a, b) = (p.exc_base[s] as usize, p.exc_base[s + 1] as usize);
        if p.exc_base[s] < 0 || a > b || b > p.exc.len() {
            return false;
        }
        twelve_unstep(&p.data[s * 1536..(s + 1) * 1536], &p.exc[a..b], p.hb(), &mut u);
        for (q, &v) in u.iter().enumerate() {
            out[at + places[q]] = v;
        }
    }
    true
}

/// The six-bit block layout's number (tiers[0]'s top byte) and a step's bytes (gpu/glyd_gpu.cu's Int6).
pub const INT6_TAG: u32 = 4;
pub const INT6_STEP_BYTES: usize = 768;

/// The five-bit block layout's number (tiers[0]'s top byte) and a step's bytes (gpu/glyd_gpu.cu's Int5).
pub const INT5_TAG: u32 = 5;
pub const INT5_STEP_BYTES: usize = 640;

/// The four-bit block layout's number (tiers[0]'s top byte) and a step's bytes (gpu/glyd_gpu.cu's Int4).
pub const INT4_TAG: u32 = 6;
pub const INT4_STEP_BYTES: usize = 512;

/// The exact four-bit block layout's number (gpu/glyd_gpu.cu's Int4H): the four-bit layout's steps and places, a scale of its own form.
pub const INT4H_TAG: u32 = 13;

/// The exact FP4 block layout's number (gpu/glyd_gpu.cu's Fp4): the four-bit layout's steps and places, each code an E2M1 value, a run's scale one E8M0 byte.
pub const FP4_TAG: u32 = 14;

/// The exact FP4 layout with packed scales (gpu/glyd_gpu.cu's Fp4R): Fp4's codes, a run's scale its row's base byte plus two bits.
pub const FP4R_TAG: u32 = 15;

/// The three-bit block layout's number (tiers[0]'s top byte) and a step's bytes (gpu/glyd_gpu.cu's Int3).
pub const INT3_TAG: u32 = 7;
pub const INT3_STEP_BYTES: usize = 384;

fn bf16_f32(w: u16) -> f32 {
    f32::from_bits((w as u32) << 16)
}

/// A finite f32 rounded to bf16, to nearest and ties to even.
fn f32_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

/// The 16 ratios of a run's largest magnitude the scale fit tries: torch.linspace(0.70, 1.0, 16) in float32, as torch makes it (the step
/// (1 - 0.70) / 15; 0.70 + step i for i < 8, 1.0 - step (15 - i) for the rest).
const FIT_RATIOS: [f32; 16] = {
    let step = (1.0f32 - 0.70f32) / 15.0f32;
    let mut r = [0f32; 16];
    let mut i = 0;
    while i < 16 {
        r[i] = if i < 8 { 0.70f32 + step * i as f32 } else { 1.0f32 - step * (15 - i) as f32 };
        i += 1;
    }
    r
};

/// The weight a six-bit code u (0-63) stands for under the bf16 scale s: (u - 32) s, rounded to bf16 once (the product is exact in f32).
pub fn int6_value(u: u8, s: u16) -> u16 {
    f32_bf16((u as i32 - 32) as f32 * bf16_f32(s))
}

/// The scale and the codes of one run of 32 weights: s the study's mse fit (research/lossy-2026/kld_study.py): of the bf16 of amax r / 31 for
/// each r of [`FIT_RATIOS`] (a scale of 0 skipped), the first with the least squared error of codes times s against the weights, in f32 (0 for
/// a run of zeros, or with no scale). Its sums run in order, not in torch's order on a GPU, so a near-tie may resolve differently. A weight's
/// code clamp(round_half_even(w / s), -31, 31) + 32 (1 to 63). A candidate that cannot be the first of the least is not summed: a sum of
/// non-negative terms is no less than any of them, so one whose term for the run's largest weight is over the least sum so far (the last
/// candidate's, summed first, to begin with) is left out; the scale and the codes are those of summing every candidate.
fn int6_run(run: &[u16]) -> (u16, [u8; 32]) {
    let w: [f32; 32] = std::array::from_fn(|i| bf16_f32(run[i]));
    let peak = w.iter().fold(0f32, |p, &x| if x.abs() > p.abs() { x } else { p });
    let amax = peak.abs();
    let term = |x: f32, cf: f32| {
        let d = rne_clamp(x / cf, 31.0) * cf - x;
        d * d
    };
    let sum = |cf: f32| w.iter().fold(0f32, |e, &x| e + term(x, cf));
    let c15 = f32_bf16(amax * FIT_RATIOS[15] / 31.0);
    let e15 = if c15 == 0 { f32::INFINITY } else { sum(bf16_f32(c15)) };
    let (mut s, mut least, mut bound) = (0u16, 0f32, e15);
    for (i, r) in FIT_RATIOS.into_iter().enumerate() {
        let c = f32_bf16(amax * r / 31.0);
        if c == 0 {
            continue;
        }
        let cf = bf16_f32(c);
        if term(peak, cf) > bound {
            continue;
        }
        let err = if i == 15 { e15 } else { sum(cf) };
        debug_assert!(term(peak, cf) <= err);
        bound = bound.min(err);
        if s == 0 || err < least {
            (s, least) = (c, err);
        }
    }
    let mut codes = [32u8; 32];
    if s != 0 {
        let sf = bf16_f32(s);
        for (c, &x) in codes.iter_mut().zip(&w) {
            *c = (rne_clamp(x / sf, 31.0) as i32 + 32) as u8;
        }
    }
    (s, codes)
}

/// Lane l (4g + t) of a step's weight i (4n + j, n 0-7, j 0-3): its row and column within the step (the mma layout's places).
fn lane_place(l: usize, i: usize) -> (usize, usize) {
    let (g, t, n, j) = (l / 4, l % 4, i / 4, i % 4);
    (8 * n + g, 8 * (j >> 1) + 2 * t + (j & 1))
}

/// W [rows, cols] (rows a multiple of 64, cols of 32) in the six-bit block layout (gpu/glyd_gpu.cu's Int6): `data` the steps' 768 bytes,
/// `blocks` the scales (128 bytes a pair of steps), `block_base` steps + 1 zeros (not read), tiers [INT6_TAG << 24, 0, 0]. Lossy: each
/// weight becomes [`int6_value`] of its code and its run's scale. None where a weight is not finite or the shape is not one.
pub fn pack_int6(w: &[u16], rows: usize, cols: usize) -> Option<Tiered> {
    if rows == 0 || rows % 64 != 0 || cols == 0 || cols % 32 != 0 || w.len() != rows * cols || w.iter().any(|&x| x & 0x7F80 == 0x7F80) {
        return None;
    }
    let runs = cols / 32;
    let mut scale = vec![0u16; rows * runs];
    let mut code = vec![0u8; rows * cols];
    for r in 0..rows {
        for k in 0..runs {
            let (s, c) = int6_run(&w[r * cols + 32 * k..r * cols + 32 * k + 32]);
            scale[r * runs + k] = s;
            code[r * cols + 32 * k..r * cols + 32 * k + 32].copy_from_slice(&c);
        }
    }
    Some(int6_layout(&code, &scale, rows, cols))
}

/// The codes (a six-bit u, 1-63, rows x cols) and bf16 scales (rows x cols / 32) of W laid out as the GPU's Int6 reads them: what [`pack_int6`] does once its fit has made them.
fn int6_layout(code: &[u8], scale: &[u16], rows: usize, cols: usize) -> Tiered {
    let runs = cols / 32;
    let (rbs, ks) = (rows / 64, cols / 16);
    let mut data = vec![0u8; rbs * ks * INT6_STEP_BYTES];
    let mut blocks = vec![0u8; rbs * ks / 2 * 128];
    for rb in 0..rbs {
        for k in 0..ks {
            let step = rb * ks + k;
            let d = &mut data[step * INT6_STEP_BYTES..(step + 1) * INT6_STEP_BYTES];
            for l in 0..32 {
                let (mut hi, mut lo) = ([0u32; 2], [0u32; 4]);
                for i in 0..32 {
                    let (rr, cc) = lane_place(l, i);
                    let u = code[(rb * 64 + rr) * cols + 16 * k + cc] as u32;
                    let (pr, h) = (i / 2, i % 2);
                    hi[pr / 8] |= (u >> 4) << (2 * (pr % 8) + 16 * h);
                    lo[pr / 4] |= (u & 15) << (4 * (pr % 4) + 16 * h);
                }
                for (q, v) in hi.iter().enumerate() {
                    d[4 * (32 * q + l)..4 * (32 * q + l) + 4].copy_from_slice(&v.to_le_bytes());
                }
                for (q, v) in lo.iter().enumerate() {
                    let at = 256 + 8 * (l + 32 * (q / 2)) + 4 * (q % 2);
                    d[at..at + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
            if k % 2 == 0 {
                let b = &mut blocks[step / 2 * 128..step / 2 * 128 + 128];
                for g in 0..8 {
                    for n in 0..8 {
                        let s = scale[(rb * 64 + 8 * n + g) * runs + k / 2];
                        b[2 * (8 * g + n)..2 * (8 * g + n) + 2].copy_from_slice(&s.to_le_bytes());
                    }
                }
            }
        }
    }
    Tiered { rows, cols, data, blocks, block_base: vec![0; rbs * ks + 1], tiers: [INT6_TAG << 24, 0, 0] }
}

/// The same, decoded back and compared with [`int6_value`] of each weight's own code and its run's scale (the check [`pack_tiered_checked`]
/// makes of the tiered layout): None where one is not those, bit for bit, or where pack_int6 is None.
pub fn pack_int6_checked(w: &[u16], rows: usize, cols: usize) -> Option<Tiered> {
    let p = pack_int6(w, rows, cols)?;
    int6_agrees(w, &p).then_some(p)
}

fn int6_agrees(w: &[u16], p: &Tiered) -> bool {
    let mut back = vec![0u16; w.len()];
    unpack_int6(p, &mut back)
        && w.chunks(32).zip(back.chunks(32)).all(|(run, got)| {
            let (s, c) = int6_run(run);
            got.iter().zip(&c).all(|(&g, &u)| g == int6_value(u, s))
        })
}

/// W [rows, cols] given as its runs' codes and scales, not made by [`pack_int6`]'s fit (a calibration's: research/lossy-2026/calib.py's GPTQ): `code` the six-bit codes -31 to 31 (rows x cols), `scale` the bf16
/// scales (rows x cols / 32), laid out as [`pack_int6`] lays out its own and kept bit for bit. Each weight becomes [`int6_value`] of its code and its run's scale, decoded back and compared with those (the check
/// [`pack_int6_checked`] makes). None where a code is past 31, a scale is not finite, the sizes are not a six-bit layout's, or the decode is not those.
pub fn pack_int6_runs_checked(code: &[i8], scale: &[u16], rows: usize, cols: usize) -> Option<Tiered> {
    if rows == 0
        || rows % 64 != 0
        || cols == 0
        || cols % 32 != 0
        || code.len() != rows * cols
        || scale.len() != rows * cols / 32
        || code.iter().any(|&c| c.unsigned_abs() > 31)
        || scale.iter().any(|&s| s & 0x7F80 == 0x7F80)
    {
        return None;
    }
    let u: Vec<u8> = code.iter().map(|&c| (c + 32) as u8).collect();
    let p = int6_layout(&u, scale, rows, cols);
    int6_holds(&u, scale, &p).then_some(p)
}

/// The pack decoded back is [`int6_value`] of each weight's code `u` (1-63) and its run's scale.
fn int6_holds(code: &[u8], scale: &[u16], p: &Tiered) -> bool {
    let mut back = vec![0u16; code.len()];
    unpack_int6(p, &mut back) && back.chunks(32).zip(code.chunks(32)).zip(scale).all(|((got, u), &s)| got.iter().zip(u).all(|(&g, &u)| g == int6_value(u, s)))
}

/// A six-bit block pack back to bf16 (row-major), as the GPU's Int6 decodes it; false where the pack is not its shape's.
pub fn unpack_int6(p: &Tiered, out: &mut [u16]) -> bool {
    let (rows, cols) = (p.rows, p.cols);
    let (rbs, ks) = (rows / 64, cols / 16);
    if p.tiers[0] >> 24 != INT6_TAG
        || rows % 64 != 0
        || cols % 32 != 0
        || out.len() != rows * cols
        || p.data.len() != rbs * ks * INT6_STEP_BYTES
        || p.blocks.len() != rbs * ks / 2 * 128
    {
        return false;
    }
    let word = |b: &[u8], at: usize| u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]);
    for rb in 0..rbs {
        for k in 0..ks {
            let step = rb * ks + k;
            let d = &p.data[step * INT6_STEP_BYTES..(step + 1) * INT6_STEP_BYTES];
            let b = &p.blocks[step / 2 * 128..step / 2 * 128 + 128];
            for l in 0..32 {
                let hi = [word(d, 4 * l), word(d, 4 * (32 + l))];
                let lo = [word(d, 256 + 8 * l), word(d, 256 + 8 * l + 4), word(d, 256 + 8 * (l + 32)), word(d, 256 + 8 * (l + 32) + 4)];
                for i in 0..32 {
                    let (pr, h) = (i / 2, i % 2);
                    let u = (((hi[pr / 8] >> (2 * (pr % 8) + 16 * h)) & 3) << 4 | ((lo[pr / 4] >> (4 * (pr % 4) + 16 * h)) & 15)) as u8;
                    let (rr, cc) = lane_place(l, i);
                    let (g, n) = (l / 4, i / 4);
                    let s = u16::from_le_bytes([b[2 * (8 * g + n)], b[2 * (8 * g + n) + 1]]);
                    out[(rb * 64 + rr) * cols + 16 * k + cc] = int6_value(u, s);
                }
            }
        }
    }
    true
}

/// The weight a five-bit code u (0-31) stands for under the bf16 scale s: (u - 16) s, rounded to bf16 once (the product is exact in f32).
pub fn int5_value(u: u8, s: u16) -> u16 {
    f32_bf16((u as i32 - 16) as f32 * bf16_f32(s))
}

/// The scale and the codes of one run of 32 weights: s the study's mse fit (research/lossy-2026/kld_study.py): of the bf16 of amax r / 15 for
/// each r of [`FIT_RATIOS`] (a scale of 0 skipped), the first with the least squared error of codes times s against the weights, in f32 (0 for
/// a run of zeros, or with no scale). Its sums run in order, not in torch's order on a GPU, so a near-tie may resolve differently. A weight's
/// code clamp(round_half_even(w / s), -15, 15) + 16 (1 to 31). A candidate that cannot be the first of the least is not summed: a sum of
/// non-negative terms is no less than any of them, so one whose term for the run's largest weight is over the least sum so far (the last
/// candidate's, summed first, to begin with) is left out; the scale and the codes are those of summing every candidate.
fn int5_run(run: &[u16]) -> (u16, [u8; 32]) {
    let w: [f32; 32] = std::array::from_fn(|i| bf16_f32(run[i]));
    let peak = w.iter().fold(0f32, |p, &x| if x.abs() > p.abs() { x } else { p });
    let amax = peak.abs();
    let term = |x: f32, cf: f32| {
        let d = rne_clamp(x / cf, 15.0) * cf - x;
        d * d
    };
    let sum = |cf: f32| w.iter().fold(0f32, |e, &x| e + term(x, cf));
    let c15 = f32_bf16(amax * FIT_RATIOS[15] / 15.0);
    let e15 = if c15 == 0 { f32::INFINITY } else { sum(bf16_f32(c15)) };
    let (mut s, mut least, mut bound) = (0u16, 0f32, e15);
    for (i, r) in FIT_RATIOS.into_iter().enumerate() {
        let c = f32_bf16(amax * r / 15.0);
        if c == 0 {
            continue;
        }
        let cf = bf16_f32(c);
        if term(peak, cf) > bound {
            continue;
        }
        let err = if i == 15 { e15 } else { sum(cf) };
        debug_assert!(term(peak, cf) <= err);
        bound = bound.min(err);
        if s == 0 || err < least {
            (s, least) = (c, err);
        }
    }
    let mut codes = [16u8; 32];
    if s != 0 {
        let sf = bf16_f32(s);
        for (c, &x) in codes.iter_mut().zip(&w) {
            *c = (rne_clamp(x / sf, 15.0) as i32 + 16) as u8;
        }
    }
    (s, codes)
}

/// W [rows, cols] (rows a multiple of 64, cols of 32) in the five-bit block layout (gpu/glyd_gpu.cu's Int5): `data` the steps' 640 bytes,
/// `blocks` the scales (128 bytes a pair of steps), `block_base` steps + 1 zeros (not read), tiers [INT5_TAG << 24, 0, 0]. Lossy: each
/// weight becomes [`int5_value`] of its code and its run's scale. None where a weight is not finite or the shape is not one.
pub fn pack_int5(w: &[u16], rows: usize, cols: usize) -> Option<Tiered> {
    if rows == 0 || rows % 64 != 0 || cols == 0 || cols % 32 != 0 || w.len() != rows * cols || w.iter().any(|&x| x & 0x7F80 == 0x7F80) {
        return None;
    }
    let runs = cols / 32;
    let mut scale = vec![0u16; rows * runs];
    let mut code = vec![0u8; rows * cols];
    for r in 0..rows {
        for k in 0..runs {
            let (s, c) = int5_run(&w[r * cols + 32 * k..r * cols + 32 * k + 32]);
            scale[r * runs + k] = s;
            code[r * cols + 32 * k..r * cols + 32 * k + 32].copy_from_slice(&c);
        }
    }
    Some(int5_layout(&code, &scale, rows, cols))
}

/// The codes (a five-bit u, 1-31, rows x cols) and bf16 scales (rows x cols / 32) of W laid out as the GPU's Int5 reads them: what [`pack_int5`] does once its fit has made them.
fn int5_layout(code: &[u8], scale: &[u16], rows: usize, cols: usize) -> Tiered {
    let runs = cols / 32;
    let (rbs, ks) = (rows / 64, cols / 16);
    let mut data = vec![0u8; rbs * ks * INT5_STEP_BYTES];
    let mut blocks = vec![0u8; rbs * ks / 2 * 128];
    for rb in 0..rbs {
        for k in 0..ks {
            let step = rb * ks + k;
            let d = &mut data[step * INT5_STEP_BYTES..(step + 1) * INT5_STEP_BYTES];
            for l in 0..32 {
                let (mut hi, mut lo) = (0u32, [0u32; 4]);
                for i in 0..32 {
                    let (rr, cc) = lane_place(l, i);
                    let u = code[(rb * 64 + rr) * cols + 16 * k + cc] as u32;
                    let (pr, h) = (i / 2, i % 2);
                    hi |= (u >> 4) << (pr + 16 * h);
                    lo[pr / 4] |= (u & 15) << (4 * (pr % 4) + 16 * h);
                }
                d[4 * l..4 * l + 4].copy_from_slice(&hi.to_le_bytes());
                for (q, v) in lo.iter().enumerate() {
                    let at = 128 + 8 * (l + 32 * (q / 2)) + 4 * (q % 2);
                    d[at..at + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
            if k % 2 == 0 {
                let b = &mut blocks[step / 2 * 128..step / 2 * 128 + 128];
                for g in 0..8 {
                    for n in 0..8 {
                        let s = scale[(rb * 64 + 8 * n + g) * runs + k / 2];
                        b[2 * (8 * g + n)..2 * (8 * g + n) + 2].copy_from_slice(&s.to_le_bytes());
                    }
                }
            }
        }
    }
    Tiered { rows, cols, data, blocks, block_base: vec![0; rbs * ks + 1], tiers: [INT5_TAG << 24, 0, 0] }
}

/// The same, decoded back and compared with [`int5_value`] of each weight's own code and its run's scale (the check [`pack_tiered_checked`]
/// makes of the tiered layout): None where one is not those, bit for bit, or where pack_int5 is None.
pub fn pack_int5_checked(w: &[u16], rows: usize, cols: usize) -> Option<Tiered> {
    let p = pack_int5(w, rows, cols)?;
    int5_agrees(w, &p).then_some(p)
}

fn int5_agrees(w: &[u16], p: &Tiered) -> bool {
    let mut back = vec![0u16; w.len()];
    unpack_int5(p, &mut back)
        && w.chunks(32).zip(back.chunks(32)).all(|(run, got)| {
            let (s, c) = int5_run(run);
            got.iter().zip(&c).all(|(&g, &u)| g == int5_value(u, s))
        })
}

/// W [rows, cols] given as its runs' codes and scales, not made by [`pack_int5`]'s fit (a calibration's): `code` the five-bit codes -15 to 15 (rows x cols), `scale` the bf16 scales (rows x cols / 32), laid out as
/// [`pack_int5`] lays out its own and kept bit for bit. Each weight becomes [`int5_value`] of its code and its run's scale, decoded back and compared with those (the check [`pack_int5_checked`] makes). None where
/// a code is past 15, a scale is not finite, the sizes are not a five-bit layout's, or the decode is not those.
pub fn pack_int5_runs_checked(code: &[i8], scale: &[u16], rows: usize, cols: usize) -> Option<Tiered> {
    if rows == 0
        || rows % 64 != 0
        || cols == 0
        || cols % 32 != 0
        || code.len() != rows * cols
        || scale.len() != rows * cols / 32
        || code.iter().any(|&c| c.unsigned_abs() > 15)
        || scale.iter().any(|&s| s & 0x7F80 == 0x7F80)
    {
        return None;
    }
    let u: Vec<u8> = code.iter().map(|&c| (c + 16) as u8).collect();
    let p = int5_layout(&u, scale, rows, cols);
    int5_holds(&u, scale, &p).then_some(p)
}

/// The pack decoded back is [`int5_value`] of each weight's code `u` (1-31) and its run's scale.
fn int5_holds(code: &[u8], scale: &[u16], p: &Tiered) -> bool {
    let mut back = vec![0u16; code.len()];
    unpack_int5(p, &mut back) && back.chunks(32).zip(code.chunks(32)).zip(scale).all(|((got, u), &s)| got.iter().zip(u).all(|(&g, &u)| g == int5_value(u, s)))
}

/// A five-bit block pack back to bf16 (row-major), as the GPU's Int5 decodes it; false where the pack is not its shape's.
pub fn unpack_int5(p: &Tiered, out: &mut [u16]) -> bool {
    let (rows, cols) = (p.rows, p.cols);
    let (rbs, ks) = (rows / 64, cols / 16);
    if p.tiers[0] >> 24 != INT5_TAG
        || rows % 64 != 0
        || cols % 32 != 0
        || out.len() != rows * cols
        || p.data.len() != rbs * ks * INT5_STEP_BYTES
        || p.blocks.len() != rbs * ks / 2 * 128
    {
        return false;
    }
    let word = |b: &[u8], at: usize| u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]);
    for rb in 0..rbs {
        for k in 0..ks {
            let step = rb * ks + k;
            let d = &p.data[step * INT5_STEP_BYTES..(step + 1) * INT5_STEP_BYTES];
            let b = &p.blocks[step / 2 * 128..step / 2 * 128 + 128];
            for l in 0..32 {
                let hi = word(d, 4 * l);
                let lo = [word(d, 128 + 8 * l), word(d, 128 + 8 * l + 4), word(d, 128 + 8 * (l + 32)), word(d, 128 + 8 * (l + 32) + 4)];
                for i in 0..32 {
                    let (pr, h) = (i / 2, i % 2);
                    let u = (((hi >> (pr + 16 * h)) & 1) << 4 | ((lo[pr / 4] >> (4 * (pr % 4) + 16 * h)) & 15)) as u8;
                    let (rr, cc) = lane_place(l, i);
                    let (g, n) = (l / 4, i / 4);
                    let s = u16::from_le_bytes([b[2 * (8 * g + n)], b[2 * (8 * g + n) + 1]]);
                    out[(rb * 64 + rr) * cols + 16 * k + cc] = int5_value(u, s);
                }
            }
        }
    }
    true
}

/// The weight a four-bit code u (0-15) stands for under the bf16 scale s: (u - 8) s, rounded to bf16 once (the product is exact in f32).
pub fn int4_value(u: u8, s: u16) -> u16 {
    f32_bf16((u as i32 - 8) as f32 * bf16_f32(s))
}

/// The weight a three-bit code u (0-7) stands for under the bf16 scale s: (u - 4) s, rounded to bf16 once (the product is exact in f32).
pub fn int3_value(u: u8, s: u16) -> u16 {
    f32_bf16((u as i32 - 4) as f32 * bf16_f32(s))
}

/// The scale and the codes of one run of 32 weights for the codes -q to q: [`int6_run`]'s and [`int5_run`]'s rule with q 7 (four bits) or 3 (three) in
/// place of 31 and 15: s the study's mse fit (research/lossy-2026/kld_study.py: of the bf16 of amax r / q for each r of [`FIT_RATIOS`] (a scale of 0
/// skipped), the first with the least squared error of codes times s against the weights, in f32; 0 for a run of zeros, or with no scale; a candidate
/// that cannot be the first of the least is not summed, as there), a weight's code clamp(round_half_even(w / s), -q, q) + q + 1 (1 to 2 q + 1). The
/// codes are symmetric, as the study's (q = 2^(bits - 1) - 1) and the six- and five-bit layouts': the code 0 is left out, and the decode still reads
/// it, as -(q + 1) s.
/// x rounded to the nearest integer, ties to even, then clamped to [-q, q] (q <= 2^22): the clamp first, then 1.5 * 2^23 added and taken away,
/// which rounds in the FPU's default mode (nearest, ties to even). The same value as `x.round_ties_even().clamp(-q, q)` for every x (a zero's
/// sign aside: every use multiplies it or makes it an integer), without the software call `round_ties_even` becomes where the target has no
/// rounding instruction (x86-64 without SSE4.1, MSVC: a block layout's packing took 30 times as long).
#[inline]
fn rne_clamp(x: f32, q: f32) -> f32 {
    const M: f32 = 12_582_912.0;
    (x.clamp(-q, q) + M) - M
}

fn int_run(run: &[u16], q: f32) -> (u16, [u8; 32]) {
    let w: [f32; 32] = std::array::from_fn(|i| bf16_f32(run[i]));
    let peak = w.iter().fold(0f32, |p, &x| if x.abs() > p.abs() { x } else { p });
    let amax = peak.abs();
    let term = |x: f32, cf: f32| {
        let d = rne_clamp(x / cf, q) * cf - x;
        d * d
    };
    let sum = |cf: f32| w.iter().fold(0f32, |e, &x| e + term(x, cf));
    let c15 = f32_bf16(amax * FIT_RATIOS[15] / q);
    let e15 = if c15 == 0 { f32::INFINITY } else { sum(bf16_f32(c15)) };
    let (mut s, mut least, mut bound) = (0u16, 0f32, e15);
    for (i, r) in FIT_RATIOS.into_iter().enumerate() {
        let c = f32_bf16(amax * r / q);
        if c == 0 {
            continue;
        }
        let cf = bf16_f32(c);
        if term(peak, cf) > bound {
            continue;
        }
        let err = if i == 15 { e15 } else { sum(cf) };
        debug_assert!(term(peak, cf) <= err);
        bound = bound.min(err);
        if s == 0 || err < least {
            (s, least) = (c, err);
        }
    }
    let mut codes = [q as u8 + 1; 32];
    if s != 0 {
        let sf = bf16_f32(s);
        for (c, &x) in codes.iter_mut().zip(&w) {
            *c = (rne_clamp(x / sf, q) as i32 + q as i32 + 1) as u8;
        }
    }
    (s, codes)
}

fn int4_run(run: &[u16]) -> (u16, [u8; 32]) {
    int_run(run, 7.0)
}

fn int3_run(run: &[u16]) -> (u16, [u8; 32]) {
    int_run(run, 3.0)
}

/// W [rows, cols] (rows a multiple of 64, cols of 32) in the four-bit block layout (gpu/glyd_gpu.cu's Int4): `data` the steps' 512 bytes (the
/// six- and five-bit layouts' low four bits alone: lane l's four words), `blocks` the scales (128 bytes a pair of steps), `block_base` steps + 1
/// zeros (not read), tiers [INT4_TAG << 24, 0, 0]. Lossy: each weight becomes [`int4_value`] of its code (u = c + 8, c from -7 to 7) and its
/// run's scale. None where a weight is not finite or the shape is not one.
pub fn pack_int4(w: &[u16], rows: usize, cols: usize) -> Option<Tiered> {
    if rows == 0 || rows % 64 != 0 || cols == 0 || cols % 32 != 0 || w.len() != rows * cols || w.iter().any(|&x| x & 0x7F80 == 0x7F80) {
        return None;
    }
    let runs = cols / 32;
    let mut scale = vec![0u16; rows * runs];
    let mut code = vec![0u8; rows * cols];
    for r in 0..rows {
        for k in 0..runs {
            let (s, c) = int4_run(&w[r * cols + 32 * k..r * cols + 32 * k + 32]);
            scale[r * runs + k] = s;
            code[r * cols + 32 * k..r * cols + 32 * k + 32].copy_from_slice(&c);
        }
    }
    Some(int4_layout(&code, &scale, rows, cols, [INT4_TAG << 24, 0, 0]))
}

/// The four-bit layouts' bytes of codes u (0-15, row-major) and a 16-bit scale word a run (row-major, cols / 32 a row): Int4's and Int4H's.
/// The four-bit layouts' steps: each weight's code (0-15, row-major) at its place.
fn int4_steps(code: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    int4_steps_by(|i| code[i], rows, cols)
}

/// `int4_steps` of the code `code(i)` gives weight i (row-major): an MX FP4 matrix's nibbles read where they are, no copy of the codes made.
fn int4_steps_by(code: impl Fn(usize) -> u8, rows: usize, cols: usize) -> Vec<u8> {
    let mut data = vec![0u8; rows / 64 * (cols / 16) * INT4_STEP_BYTES];
    int4_steps_into(code, rows, cols, &mut data);
    data
}

/// `int4_steps_by` into the caller's bytes (every one of them written: a step's 32 lanes' four words).
fn int4_steps_into(code: impl Fn(usize) -> u8, rows: usize, cols: usize, data: &mut [u8]) {
    let (rbs, ks) = (rows / 64, cols / 16);
    for rb in 0..rbs {
        for k in 0..ks {
            let step = rb * ks + k;
            let d = &mut data[step * INT4_STEP_BYTES..(step + 1) * INT4_STEP_BYTES];
            for l in 0..32 {
                let mut lo = [0u32; 4];
                for i in 0..32 {
                    let (rr, cc) = lane_place(l, i);
                    let u = code((rb * 64 + rr) * cols + 16 * k + cc) as u32;
                    let (pr, h) = (i / 2, i % 2);
                    lo[pr / 4] |= u << (4 * (pr % 4) + 16 * h);
                }
                for (q, v) in lo.iter().enumerate() {
                    let at = 8 * (l + 32 * (q / 2)) + 4 * (q % 2);
                    d[at..at + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
        }
    }
}

/// `int4_steps_into` of an MX FP4 matrix's codes `q` (two a byte, row-major), a step's 64 rows by 16 columns read into a tile first: the same bytes, without
/// a code's place worked out from its index (an FP4 expert's pack was most of Kimi-K3's load).
fn fp4_steps_into(q: &[u8], rows: usize, cols: usize, data: &mut [u8]) {
    let (rbs, ks) = (rows / 64, cols / 16);
    let mut tile = [[0u8; 16]; 64];
    for rb in 0..rbs {
        for k in 0..ks {
            for (rr, row) in tile.iter_mut().enumerate() {
                let src = &q[((rb * 64 + rr) * cols + 16 * k) / 2..][..8];
                for b in 0..8 {
                    (row[2 * b], row[2 * b + 1]) = (src[b] & 15, src[b] >> 4);
                }
            }
            let d = &mut data[(rb * ks + k) * INT4_STEP_BYTES..(rb * ks + k + 1) * INT4_STEP_BYTES];
            for l in 0..32 {
                let mut lo = [0u32; 4];
                for i in 0..32 {
                    let (rr, cc) = lane_place(l, i);
                    let (pr, h) = (i / 2, i % 2);
                    lo[pr / 4] |= (tile[rr][cc] as u32) << (4 * (pr % 4) + 16 * h);
                }
                for (qq, v) in lo.iter().enumerate() {
                    let at = 8 * (l + 32 * (qq / 2)) + 4 * (qq % 2);
                    d[at..at + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
        }
    }
}

/// A four-bit layout's code of weight (row, col) from its steps.
fn int4_code(data: &[u8], cols: usize, row: usize, col: usize) -> u8 {
    let (ks, rb, rr, k, cc) = (cols / 16, row / 64, row % 64, col / 16, col % 16);
    let (g, n) = (rr % 8, rr / 8);
    let (t, j) = ((cc % 8) / 2, (cc / 8) * 2 + cc % 2);
    let (l, i) = (4 * g + t, 4 * n + j);
    let (pr, h) = (i / 2, i % 2);
    let q = pr / 4;
    let at = (rb * ks + k) * INT4_STEP_BYTES + 8 * (l + 32 * (q / 2)) + 4 * (q % 2);
    let w = u32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]]);
    ((w >> (4 * (pr % 4) + 16 * h)) & 15) as u8
}

fn int4_layout(code: &[u8], scale: &[u16], rows: usize, cols: usize, tiers: [u32; 3]) -> Tiered {
    let runs = cols / 32;
    let (rbs, ks) = (rows / 64, cols / 16);
    let data = int4_steps(code, rows, cols);
    let mut blocks = vec![0u8; rbs * ks / 2 * 128];
    for rb in 0..rbs {
        for k in 0..ks {
            let step = rb * ks + k;
            if k % 2 == 0 {
                let b = &mut blocks[step / 2 * 128..step / 2 * 128 + 128];
                for g in 0..8 {
                    for n in 0..8 {
                        let s = scale[(rb * 64 + 8 * n + g) * runs + k / 2];
                        b[2 * (8 * g + n)..2 * (8 * g + n) + 2].copy_from_slice(&s.to_le_bytes());
                    }
                }
            }
        }
    }
    Tiered { rows, cols, data, blocks, block_base: vec![0; rbs * ks + 1], tiers }
}

/// The same, decoded back and compared with [`int4_value`] of each weight's own code and its run's scale (the check [`pack_tiered_checked`]
/// makes of the tiered layout): None where one is not those, bit for bit, or where pack_int4 is None.
pub fn pack_int4_checked(w: &[u16], rows: usize, cols: usize) -> Option<Tiered> {
    let p = pack_int4(w, rows, cols)?;
    int4_agrees(w, &p).then_some(p)
}

/// An exact four-bit pack's scale word as the f32 it stands for: bit 15 the sign, bits 11-14 the exponent over `base` (the pack's tiers[1]: the
/// least biased f32 exponent of its scales), bits 0-10 the fraction's top 11 bits (a 12-bit significand).
pub fn int4h_scale(h: u16, base: u32) -> f32 {
    let h = h as u32;
    f32::from_bits(((h & 0x8000) << 16) | ((((h >> 11) & 15) + base) << 23) | ((h & 0x7FF) << 12))
}

/// The weight a code u (0-15) stands for under an exact four-bit pack's scale word: (u - 8) s in f32 (exact: 4 bits by 12), rounded to bf16 once.
pub fn int4h_value(u: u8, h: u16, base: u32) -> u16 {
    f32_bf16((u as i32 - 8) as f32 * int4h_scale(h, base))
}

/// A run of 32 bf16 weights as codes k (-8 to 7) and one scale s with a 12-bit significand such that bf16(k s) is each weight, bit for bit:
/// None where there is none. The weights of a checkpoint trained to four bits (Gemma 4's QAT) are such runs; k is round(w / (amax / d)) for
/// d from 8 down (and 7.5, 6.5), s every 12-bit value in the interval the weights' rounding cells leave (a power of two's cell is half as wide
/// toward zero), checked in f32; under a negative scale (codes -k, zeros -0) where the codes or the zeros' sign ask for it. A run of zeros: s 0
/// (its sign the zeros').
fn int4h_run(run: &[u16]) -> Option<(f32, [i8; 32])> {
    let v: [f64; 32] = std::array::from_fn(|i| bf16_f32(run[i]) as f64);
    let m = v.iter().fold(0f64, |a, &x| a.max(x.abs()));
    if m == 0.0 {
        let neg = run[0] == 0x8000;
        return run.iter().all(|&x| x == run[0]).then_some((if neg { -0.0 } else { 0.0 }, [0; 32]));
    }
    let fits = |k: &[i8; 32], c: f32| run.iter().zip(k).all(|(&w, &ki)| f32_bf16(ki as f32 * c) == w);
    for den in [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 7.5, 6.5] {
        let mut k = [0i8; 32];
        let (mut s_lo, mut s_hi, mut ok) = (0f64, f64::INFINITY, true);
        for i in 0..32 {
            let q = (v[i] / (m / den)).round_ties_even();
            if q.abs() > 8.0 || (q == 0.0) != (v[i] == 0.0) {
                ok = false;
                break;
            }
            k[i] = q as i8;
            if q == 0.0 {
                continue;
            }
            let ulp = 2f64.powi(v[i].abs().log2().floor() as i32 - 7);
            let down = if run[i] & 0x7F == 0 { 0.25 * ulp } else { 0.5 * ulp };
            let (lo, hi) = if v[i] > 0.0 { (v[i] - down, v[i] + 0.5 * ulp) } else { (v[i] - 0.5 * ulp, v[i] + down) };
            let (a, b) = if q > 0.0 { (lo / q, hi / q) } else { (hi / q, lo / q) };
            s_lo = s_lo.max(a);
            s_hi = s_hi.min(b);
        }
        if !ok || s_lo > s_hi || s_lo <= 0.0 {
            continue;
        }
        // under a negative scale the codes are -k (they must run -8 to 7) and the zeros -0
        let (lo_k, hi_k) = (k.iter().min().copied().unwrap_or(0), k.iter().max().copied().unwrap_or(0));
        let (pos0, neg0) = (run.contains(&0), run.contains(&0x8000));
        for sign in [1.0f32, -1.0] {
            let fit = if sign > 0.0 { lo_k >= -8 && hi_k <= 7 && !neg0 } else { lo_k >= -7 && !pos0 };
            if !fit {
                continue;
            }
            let k = if sign < 0.0 { k.map(|x| -x) } else { k };
            for e in [s_hi.log2().floor() as i32, s_lo.log2().floor() as i32] {
                let q = 2f64.powi(e - 11);
                let (a, b) = ((s_lo / q).floor() as i64 - 1, (s_hi / q).ceil() as i64 + 1);
                for n in a.max(2048)..=b.min(4095) {
                    let c = sign * (n as f64 * q) as f32;
                    if fits(&k, c) {
                        return Some((c, k));
                    }
                }
            }
        }
    }
    None
}

/// W [rows, cols] (rows a multiple of 64, cols of 32) in the exact four-bit block layout (gpu/glyd_gpu.cu's Int4H): the four-bit layout's
/// steps and places, each run's scale a word of [`int4h_scale`]'s form, tiers [INT4H_TAG << 24, base, 0]. Lossless where it is made: every
/// weight is [`int4h_value`] of its code and its run's word, bit for bit; None where a run is not codes times one such scale (any weights not
/// trained to four bits), where the scales' exponents span more than 16 binades, or where the shape is not one.
pub fn pack_int4h(w: &[u16], rows: usize, cols: usize) -> Option<Tiered> {
    if rows == 0 || rows % 64 != 0 || cols == 0 || cols % 32 != 0 || w.len() != rows * cols {
        return None;
    }
    let runs: Vec<(f32, [i8; 32])> = w.chunks(32).map(int4h_run).collect::<Option<_>>()?;
    let exps = runs.iter().filter(|r| r.0 != 0.0).map(|r| (r.0.to_bits() >> 23) & 0xFF);
    let (lo, hi) = exps.fold((255u32, 0u32), |(a, b), e| (a.min(e), b.max(e)));
    let base = if lo > hi { 127 } else { lo };
    if lo <= hi && (hi - lo > 15 || lo == 0) {
        return None;
    }
    let mut scale = vec![0u16; runs.len()];
    let mut code = vec![0u8; rows * cols];
    for (j, (s, k)) in runs.iter().enumerate() {
        let b = s.to_bits();
        scale[j] = ((b >> 16) & 0x8000) as u16 | if *s == 0.0 { 0 } else { ((((b >> 23) & 0xFF) - base) << 11 | ((b >> 12) & 0x7FF)) as u16 };
        for (i, &x) in k.iter().enumerate() {
            code[32 * j + i] = (x + 8) as u8;
        }
    }
    Some(int4_layout(&code, &scale, rows, cols, [INT4H_TAG << 24, base, 0]))
}

/// The same, decoded back ([`unpack_int4`], the GPU's decode) and compared with W bit for bit: None where they differ or pack_int4h is None.
pub fn pack_int4h_checked(w: &[u16], rows: usize, cols: usize) -> Option<Tiered> {
    let p = pack_int4h(w, rows, cols)?;
    let mut back = vec![0u16; w.len()];
    (unpack_int4(&p, &mut back) && back == w).then_some(p)
}

/// An E2M1 code (bit 3 its sign) times 2^(e - 127) as bf16, exactly (gpu/glyd_gpu.cu's Fp4; e of 2 to 128).
pub fn fp4_value(c: u8, e: u8) -> u16 {
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = MAG[(c & 7) as usize] * f32::from_bits((e as u32) << 23);
    ((if c & 8 != 0 { -v } else { v }).to_bits() >> 16) as u16
}

/// An MX FP4 matrix [rows, cols] (rows a multiple of 64, cols of 32) as released, `q` its E2M1 codes two a byte (the first in the low nibble, row-major) and `e` its E8M0 scale bytes
/// (one a run of 32 of a row), in the exact FP4 block layout (gpu/glyd_gpu.cu's Fp4): `data` Int4's steps of the codes, `blocks` the scales (64 bytes a pair of steps, a row's at
/// 8 g + n as Int4's), tiers [FP4_TAG << 24, 0, 0]. 4.25 bits a weight. Decoded back and compared with [`fp4_value`] of each code and scale; None where a scale is outside 2 to 128
/// (a product would leave bf16's normal range) or the shape is not one.
pub fn pack_fp4_checked(q: &[u8], e: &[u8], rows: usize, cols: usize) -> Option<Tiered> {
    let (db, sb) = fp4_bytes(rows, cols, false);
    let (mut data, mut blocks) = (vec![0u8; db], vec![0u8; sb]);
    pack_fp4_into(q, e, rows, cols, &mut data, &mut blocks).then(|| Tiered {
        rows,
        cols,
        data,
        blocks,
        block_base: vec![0; rows / 64 * (cols / 16) + 1],
        tiers: [FP4_TAG << 24, 0, 0],
    })
}

/// The bytes an exact FP4 matrix [rows, cols] takes: its steps and its scales (`packed`: the Fp4R layout's).
pub fn fp4_bytes(rows: usize, cols: usize, packed: bool) -> (usize, usize) {
    let (rbs, ks) = (rows / 64, cols / 16);
    (rbs * ks * INT4_STEP_BYTES, if packed { rbs * (64 + ks * 8) } else { rbs * ks / 2 * 64 })
}

/// [`pack_fp4_checked`] into the caller's bytes (`data` and `blocks` of [`fp4_bytes`]' sizes, every byte written): where a matrix is held among others (an
/// expert's place in a host tier), checked there; false where the layout does not hold it exactly or the shape is not one.
pub fn pack_fp4_into(q: &[u8], e: &[u8], rows: usize, cols: usize, data: &mut [u8], blocks: &mut [u8]) -> bool {
    if rows == 0
        || rows % 64 != 0
        || cols == 0
        || cols % 32 != 0
        || q.len() * 2 != rows * cols
        || e.len() * 32 != rows * cols
        || e.iter().any(|&x| !(2..=128).contains(&x))
        || (data.len(), blocks.len()) != fp4_bytes(rows, cols, false)
    {
        return false;
    }
    let (runs, rbs, ks) = (cols / 32, rows / 64, cols / 16);
    fp4_steps_into(q, rows, cols, data);
    for rb in 0..rbs {
        for k in (0..ks).step_by(2) {
            let b = &mut blocks[(rb * ks + k) / 2 * 64..(rb * ks + k) / 2 * 64 + 64];
            for g in 0..8 {
                for n in 0..8 {
                    b[8 * g + n] = e[(rb * 64 + 8 * n + g) * runs + k / 2];
                }
            }
        }
    }
    fp4_holds_in(data, blocks, rows, cols, false, q, e)
}

/// An MX FP4 matrix as [`pack_fp4_checked`] takes it, in the exact FP4 layout with packed scales (gpu/glyd_gpu.cu's Fp4R): Int4's steps of the codes; `blocks` a row block's scales,
/// its 64 rows' bases (each row's smallest scale, a row's at 8 g + n), then a pair of steps' offsets from them, two bits a run in 16 bytes (a row's at bit 2 n of the u16 at 2 g); tiers
/// [FP4R_TAG << 24, 0, 0]. Decoded back and compared with [`fp4_value`]; None where a row's scales span more than 4 values, a scale is outside 2 to 128, or the shape is not one.
pub fn pack_fp4r_checked(q: &[u8], e: &[u8], rows: usize, cols: usize) -> Option<Tiered> {
    let (db, sb) = fp4_bytes(rows, cols, true);
    let (mut data, mut blocks) = (vec![0u8; db], vec![0u8; sb]);
    pack_fp4r_into(q, e, rows, cols, &mut data, &mut blocks).then(|| Tiered {
        rows,
        cols,
        data,
        blocks,
        block_base: vec![0; rows / 64 * (cols / 16) + 1],
        tiers: [FP4R_TAG << 24, 0, 0],
    })
}

/// [`pack_fp4r_checked`] into the caller's bytes, as [`pack_fp4_into`].
pub fn pack_fp4r_into(q: &[u8], e: &[u8], rows: usize, cols: usize, data: &mut [u8], blocks: &mut [u8]) -> bool {
    if rows == 0
        || rows % 64 != 0
        || cols == 0
        || cols % 32 != 0
        || q.len() * 2 != rows * cols
        || e.len() * 32 != rows * cols
        || e.iter().any(|&x| !(2..=128).contains(&x))
        || (data.len(), blocks.len()) != fp4_bytes(rows, cols, true)
    {
        return false;
    }
    let (runs, rbs, ks) = (cols / 32, rows / 64, cols / 16);
    // (each row's smallest scale, and every scale within 3 of it)
    let base = |r: usize| *e[r * runs..(r + 1) * runs].iter().min().unwrap();
    if (0..rows).any(|r| {
        let b = base(r);
        e[r * runs..(r + 1) * runs].iter().any(|&x| x - b > 3)
    }) {
        return false;
    }
    fp4_steps_into(q, rows, cols, data);
    let per = 64 + ks * 8;
    for rb in 0..rbs {
        let b = &mut blocks[rb * per..(rb + 1) * per];
        let bases: Vec<u8> = (0..64).map(|rr| base(rb * 64 + rr)).collect();
        for g in 0..8 {
            for n in 0..8 {
                b[8 * g + n] = bases[8 * n + g];
            }
        }
        for k in (0..ks).step_by(2) {
            for g in 0..8 {
                let mut o = 0u16;
                for n in 0..8 {
                    let r = rb * 64 + 8 * n + g;
                    o |= ((e[r * runs + k / 2] - bases[8 * n + g]) as u16) << (2 * n);
                }
                let at = 64 + k / 2 * 16 + 2 * g;
                b[at..at + 2].copy_from_slice(&o.to_le_bytes());
            }
        }
    }
    fp4_holds_in(data, blocks, rows, cols, true, q, e)
}

/// Weight i's code of an MX FP4 matrix's bytes (two a byte, the first in the low nibble).
fn fp4_code(q: &[u8], i: usize) -> u8 {
    (q[i / 2] >> (4 * (i % 2))) & 15
}

/// Whether an exact FP4 pack (either layout) decodes as `unpack_fp4` does to the codes `q` and scales `e` it was made of: each weight's code and its run's scale
/// read back where the GPU reads them and compared (a weight's value is a function of the two: [`fp4_value`]), no values made (a matrix's would be twice its codes).
fn fp4_holds_in(data: &[u8], blocks: &[u8], rows: usize, cols: usize, packed: bool, q: &[u8], e: &[u8]) -> bool {
    if (data.len(), blocks.len()) != fp4_bytes(rows, cols, packed) || q.len() * 2 != rows * cols || e.len() * 32 != rows * cols {
        return false;
    }
    let (rbs, ks, runs) = (rows / 64, cols / 16, cols / 32);
    if !(0..rows).all(|r| (0..ks).all(|k| fp4_block_scale(blocks, cols, packed, r, k) == e[r * runs + k / 2])) {
        return false;
    }
    // (the codes: each step's words read back into a tile of its 64 rows by 16 columns, their eight nibbles each where `int4_code` reads them: lane l = 4 g + t,
    // word q of the lane, nibble 4 p + 16 h of it is pair 4 q + p's half h, weight i = 2 pair + h = 4 n + j of the lane; row 8 n + g, column 8 (j / 2) + 2 t
    // + j % 2 of the step; then each row of the tile against the source's 8 bytes there)
    let mut tile = [[0u8; 16]; 64];
    for rb in 0..rbs {
        for k in 0..ks {
            let d = &data[(rb * ks + k) * INT4_STEP_BYTES..][..INT4_STEP_BYTES];
            for l in 0..32 {
                let (g, t) = (l / 4, l % 4);
                for qq in 0..4 {
                    let at = 8 * (l + 32 * (qq / 2)) + 4 * (qq % 2);
                    let w = u32::from_le_bytes([d[at], d[at + 1], d[at + 2], d[at + 3]]);
                    for x in 0..8 {
                        let (p4, h) = (x % 4, x / 4);
                        let i = 2 * (4 * qq + p4) + h;
                        let (n, j) = (i / 4, i % 4);
                        tile[8 * n + g][8 * (j / 2) + 2 * t + j % 2] = ((w >> (4 * p4 + 16 * h)) & 15) as u8;
                    }
                }
            }
            for (rr, row) in tile.iter().enumerate() {
                let src = &q[((rb * 64 + rr) * cols + 16 * k) / 2..][..8];
                if (0..8).any(|b| src[b] != row[2 * b] | (row[2 * b + 1] << 4)) {
                    return false;
                }
            }
        }
    }
    true
}

/// Whether `p` is an exact FP4 pack of its shape (either layout).
fn fp4_shape(p: &Tiered) -> bool {
    let (rows, cols) = (p.rows, p.cols);
    let (rbs, ks) = (rows / 64, cols / 16);
    let packed = p.tiers[0] >> 24 == FP4R_TAG;
    let scales = if packed { rbs * (64 + ks * 8) } else { rbs * ks / 2 * 64 };
    (packed || p.tiers[0] >> 24 == FP4_TAG) && rows % 64 == 0 && cols % 32 == 0 && p.data.len() == rbs * ks * INT4_STEP_BYTES && p.blocks.len() == scales
}

/// The scale an exact FP4 pack gives row r's step k (16 weights), as the GPU reads it.
fn fp4_scale(p: &Tiered, r: usize, k: usize) -> u8 {
    fp4_block_scale(&p.blocks, p.cols, p.tiers[0] >> 24 == FP4R_TAG, r, k)
}

/// [`fp4_scale`] from a pack's scale bytes as they are (`packed`: the Fp4R layout's), for code that reads the pack where it is held (the host's experts).
pub fn fp4_block_scale(blocks: &[u8], cols: usize, packed: bool, r: usize, k: usize) -> u8 {
    let ks = cols / 16;
    let (rb, rr) = (r / 64, r % 64);
    let (g, n) = (rr % 8, rr / 8);
    if packed {
        let b = &blocks[rb * (64 + ks * 8)..];
        let o = u16::from_le_bytes([b[64 + k / 2 * 16 + 2 * g], b[64 + k / 2 * 16 + 2 * g + 1]]);
        b[8 * g + n].wrapping_add(((o >> (2 * n)) & 3) as u8)
    } else {
        blocks[(rb * ks + k) / 2 * 64 + 8 * g + n]
    }
}

/// An exact FP4 block pack back to bf16 (row-major), as the GPU's Fp4 decodes it; false where the pack is not its shape's.
pub fn unpack_fp4(p: &Tiered, out: &mut [u16]) -> bool {
    let (rows, cols) = (p.rows, p.cols);
    if !fp4_shape(p) || out.len() != rows * cols {
        return false;
    }
    for r in 0..rows {
        for c in 0..cols {
            out[r * cols + c] = fp4_value(int4_code(&p.data, cols, r, c), fp4_scale(p, r, c / 16));
        }
    }
    true
}

fn int4_agrees(w: &[u16], p: &Tiered) -> bool {
    let mut back = vec![0u16; w.len()];
    unpack_int4(p, &mut back)
        && w.chunks(32).zip(back.chunks(32)).all(|(run, got)| {
            let (s, c) = int4_run(run);
            got.iter().zip(&c).all(|(&g, &u)| g == int4_value(u, s))
        })
}

/// A four-bit block pack (Int4 or Int4H) back to bf16 (row-major), as the GPU decodes it; false where the pack is not its shape's.
pub fn unpack_int4(p: &Tiered, out: &mut [u16]) -> bool {
    let (rows, cols) = (p.rows, p.cols);
    let (rbs, ks) = (rows / 64, cols / 16);
    let exact = p.tiers[0] >> 24 == INT4H_TAG;
    if !(exact || p.tiers[0] >> 24 == INT4_TAG)
        || rows % 64 != 0
        || cols % 32 != 0
        || out.len() != rows * cols
        || p.data.len() != rbs * ks * INT4_STEP_BYTES
        || p.blocks.len() != rbs * ks / 2 * 128
    {
        return false;
    }
    let word = |b: &[u8], at: usize| u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]);
    for rb in 0..rbs {
        for k in 0..ks {
            let step = rb * ks + k;
            let d = &p.data[step * INT4_STEP_BYTES..(step + 1) * INT4_STEP_BYTES];
            let b = &p.blocks[step / 2 * 128..step / 2 * 128 + 128];
            for l in 0..32 {
                let lo = [word(d, 8 * l), word(d, 8 * l + 4), word(d, 8 * (l + 32)), word(d, 8 * (l + 32) + 4)];
                for i in 0..32 {
                    let (pr, h) = (i / 2, i % 2);
                    let u = ((lo[pr / 4] >> (4 * (pr % 4) + 16 * h)) & 15) as u8;
                    let (rr, cc) = lane_place(l, i);
                    let (g, n) = (l / 4, i / 4);
                    let s = u16::from_le_bytes([b[2 * (8 * g + n)], b[2 * (8 * g + n) + 1]]);
                    out[(rb * 64 + rr) * cols + 16 * k + cc] = if exact { int4h_value(u, s, p.tiers[1]) } else { int4_value(u, s) };
                }
            }
        }
    }
    true
}

/// W [rows, cols] (rows a multiple of 64, cols of 32) in the three-bit block layout (gpu/glyd_gpu.cu's Int3): `data` the steps' 384 bytes (128
/// bytes of the codes' high bit as the five-bit layout's, then 256 of their low two bits as the six-bit layout's high two), `blocks` the scales
/// (128 bytes a pair of steps), `block_base` steps + 1 zeros (not read), tiers [INT3_TAG << 24, 0, 0]. Lossy: each weight becomes [`int3_value`]
/// of its code (u = c + 4, c from -3 to 3: seven of the eight codes, as the six- and five-bit layouts leave the lowest out) and its run's scale.
/// None where a weight is not finite or the shape is not one.
pub fn pack_int3(w: &[u16], rows: usize, cols: usize) -> Option<Tiered> {
    if rows == 0 || rows % 64 != 0 || cols == 0 || cols % 32 != 0 || w.len() != rows * cols || w.iter().any(|&x| x & 0x7F80 == 0x7F80) {
        return None;
    }
    let runs = cols / 32;
    let mut scale = vec![0u16; rows * runs];
    let mut code = vec![0u8; rows * cols];
    for r in 0..rows {
        for k in 0..runs {
            let (s, c) = int3_run(&w[r * cols + 32 * k..r * cols + 32 * k + 32]);
            scale[r * runs + k] = s;
            code[r * cols + 32 * k..r * cols + 32 * k + 32].copy_from_slice(&c);
        }
    }
    let (rbs, ks) = (rows / 64, cols / 16);
    let mut data = vec![0u8; rbs * ks * INT3_STEP_BYTES];
    let mut blocks = vec![0u8; rbs * ks / 2 * 128];
    for rb in 0..rbs {
        for k in 0..ks {
            let step = rb * ks + k;
            let d = &mut data[step * INT3_STEP_BYTES..(step + 1) * INT3_STEP_BYTES];
            for l in 0..32 {
                let (mut hi, mut lo) = (0u32, [0u32; 2]);
                for i in 0..32 {
                    let (rr, cc) = lane_place(l, i);
                    let u = code[(rb * 64 + rr) * cols + 16 * k + cc] as u32;
                    let (pr, h) = (i / 2, i % 2);
                    hi |= (u >> 2) << (pr + 16 * h);
                    lo[pr / 8] |= (u & 3) << (2 * (pr % 8) + 16 * h);
                }
                d[4 * l..4 * l + 4].copy_from_slice(&hi.to_le_bytes());
                for (q, v) in lo.iter().enumerate() {
                    let at = 128 + 4 * (32 * q + l);
                    d[at..at + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
            if k % 2 == 0 {
                let b = &mut blocks[step / 2 * 128..step / 2 * 128 + 128];
                for g in 0..8 {
                    for n in 0..8 {
                        let s = scale[(rb * 64 + 8 * n + g) * runs + k / 2];
                        b[2 * (8 * g + n)..2 * (8 * g + n) + 2].copy_from_slice(&s.to_le_bytes());
                    }
                }
            }
        }
    }
    Some(Tiered { rows, cols, data, blocks, block_base: vec![0; rbs * ks + 1], tiers: [INT3_TAG << 24, 0, 0] })
}

/// The same, decoded back and compared with [`int3_value`] of each weight's own code and its run's scale (the check [`pack_tiered_checked`]
/// makes of the tiered layout): None where one is not those, bit for bit, or where pack_int3 is None.
pub fn pack_int3_checked(w: &[u16], rows: usize, cols: usize) -> Option<Tiered> {
    let p = pack_int3(w, rows, cols)?;
    int3_agrees(w, &p).then_some(p)
}

fn int3_agrees(w: &[u16], p: &Tiered) -> bool {
    let mut back = vec![0u16; w.len()];
    unpack_int3(p, &mut back)
        && w.chunks(32).zip(back.chunks(32)).all(|(run, got)| {
            let (s, c) = int3_run(run);
            got.iter().zip(&c).all(|(&g, &u)| g == int3_value(u, s))
        })
}

/// A three-bit block pack back to bf16 (row-major), as the GPU's Int3 decodes it; false where the pack is not its shape's.
pub fn unpack_int3(p: &Tiered, out: &mut [u16]) -> bool {
    let (rows, cols) = (p.rows, p.cols);
    let (rbs, ks) = (rows / 64, cols / 16);
    if p.tiers[0] >> 24 != INT3_TAG
        || rows % 64 != 0
        || cols % 32 != 0
        || out.len() != rows * cols
        || p.data.len() != rbs * ks * INT3_STEP_BYTES
        || p.blocks.len() != rbs * ks / 2 * 128
    {
        return false;
    }
    let word = |b: &[u8], at: usize| u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]);
    for rb in 0..rbs {
        for k in 0..ks {
            let step = rb * ks + k;
            let d = &p.data[step * INT3_STEP_BYTES..(step + 1) * INT3_STEP_BYTES];
            let b = &p.blocks[step / 2 * 128..step / 2 * 128 + 128];
            for l in 0..32 {
                let hi = word(d, 4 * l);
                let lo = [word(d, 128 + 4 * l), word(d, 128 + 4 * (32 + l))];
                for i in 0..32 {
                    let (pr, h) = (i / 2, i % 2);
                    let u = (((hi >> (pr + 16 * h)) & 1) << 2 | ((lo[pr / 8] >> (2 * (pr % 8) + 16 * h)) & 3)) as u8;
                    let (rr, cc) = lane_place(l, i);
                    let (g, n) = (l / 4, i / 4);
                    let s = u16::from_le_bytes([b[2 * (8 * g + n)], b[2 * (8 * g + n) + 1]]);
                    out[(rb * 64 + rr) * cols + 16 * k + cc] = int3_value(u, s);
                }
            }
        }
    }
    true
}

/// W [rows, cols] transposed: [cols, rows] (a weight [out, in] as the row layouts pack it, one input's weights a row).
pub fn transpose(w: &[u16], rows: usize, cols: usize) -> Vec<u16> {
    let mut t = vec![0u16; w.len()];
    for r0 in (0..rows).step_by(32) {
        for c0 in (0..cols).step_by(32) {
            for r in r0..(r0 + 32).min(rows) {
                for c in c0..(c0 + 32).min(cols) {
                    t[c * rows + r] = w[r * cols + c];
                }
            }
        }
    }
    t
}

/// A run's fit: the scale and the codes of 32 weights.
type RunFit = fn(&[u16]) -> (u16, [u8; 32]);

/// The six-, five- and four-bit row layouts' numbers (tiers[0]'s top byte).
pub const INT6_ROW_TAG: u32 = 8;
pub const INT5_ROW_TAG: u32 = 9;
pub const INT4_ROW_TAG: u32 = 10;

/// The three- and two-bit codebook row layouts' numbers (tiers[0]'s top byte): [`crate::codebook`]'s neuron rows (Hummingbird), codes that index a codebook of the row's own.
pub const CB3_ROW_TAG: u32 = 11;
pub const CB2_ROW_TAG: u32 = 12;

/// The bytes a row of `cols` values takes in the row layout of `bits` (4, 5 or 6) a code: its codes, `cols / 32` runs of 16 bytes (a run's 32 codes' low four bits) then, at five and six bits, 4 or 8 bytes a run
/// (their high bits), rounded up to 16 so that every row starts on 16 bytes. A row's scales are `cols / 32` bf16.
pub fn rows_stride(bits: u32, cols: usize) -> usize {
    (cols / 32 * (16 + 4 * (bits as usize - 4))).next_multiple_of(16)
}

/// The bits of a code in a row layout, by its tag.
pub fn rows_bits(tag: u32) -> Option<u32> {
    match tag {
        INT6_ROW_TAG => Some(6),
        INT5_ROW_TAG => Some(5),
        INT4_ROW_TAG => Some(4),
        _ => None,
    }
}

/// The row layout's run, 32 codes, in its row at run k of `runs`: the four words of the codes' low four bits at 16 k (the block layouts' lane words: pair p, weights 2p and 2p + 1, at bits 4 (p mod 4) and 16 + 4 (p mod 4) of
/// word p / 4), then at five bits the high bit in one word (pair p at bits p and 16 + p) and at six bits the high two in two (pair p at bits 2 (p mod 8) and 16 + 2 (p mod 8) of word p / 8), at 16 `runs` + 4 (bits - 4) k.
fn rows_place(bits: u32, c: &[u8; 32], k: usize, runs: usize, row: &mut [u8]) {
    let (mut lo, mut hi) = ([0u32; 4], [0u32; 2]);
    for (i, &u) in c.iter().enumerate() {
        let (pr, h, u) = (i / 2, i % 2, u as u32);
        lo[pr / 4] |= (u & 15) << (4 * (pr % 4) + 16 * h);
        match bits {
            6 => hi[pr / 8] |= (u >> 4) << (2 * (pr % 8) + 16 * h),
            5 => hi[0] |= (u >> 4) << (pr + 16 * h),
            _ => {}
        }
    }
    for (q, v) in lo.iter().enumerate() {
        row[16 * k + 4 * q..16 * k + 4 * q + 4].copy_from_slice(&v.to_le_bytes());
    }
    let hb = 4 * (bits as usize - 4);
    for (q, v) in hi.iter().take(hb / 4).enumerate() {
        let at = 16 * runs + hb * k + 4 * q;
        row[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }
}

/// W [rows, cols] (cols a multiple of 32, rows x cols of 1,024) in the row layout of `bits` (6, 5 or 4): the codes and scales of [`pack_int6`], [`pack_int5`] and [`pack_int4`] (each run of 32 of a row of W, the same fit and
/// rounding), placed a row at a time, not in the tensor cores' steps: `data` the rows' codes, [`rows_stride`] bytes each, in the order of [`rows_place`]; `blocks` the rows' scales, `cols / 32` bf16 each; `block_base` rows x cols / 1,024 +
/// 1 zeros (not read); tiers [tag << 24, 0, 0]. A row is read alone, by a kernel that takes the rows it needs (a matrix stored transposed, one neuron's weights a row: a down matrix whose zero activations are not read).
/// None where a weight is not finite or the shape is not one.
pub fn pack_rows(bits: u32, w: &[u16], rows: usize, cols: usize) -> Option<Tiered> {
    if !(4..=6).contains(&bits)
        || rows == 0
        || cols == 0
        || !cols.is_multiple_of(32)
        || w.len() != rows * cols
        || !(rows * cols).is_multiple_of(STEP)
        || w.iter().any(|&x| x & 0x7F80 == 0x7F80)
    {
        return None;
    }
    let (fit, tag): (RunFit, u32) = match bits {
        6 => (int6_run, INT6_ROW_TAG),
        5 => (int5_run, INT5_ROW_TAG),
        _ => (int4_run, INT4_ROW_TAG),
    };
    let (runs, stride) = (cols / 32, rows_stride(bits, cols));
    let (mut data, mut blocks) = (vec![0u8; rows * stride], vec![0u8; rows * runs * 2]);
    for r in 0..rows {
        let row = &mut data[r * stride..(r + 1) * stride];
        for k in 0..runs {
            let (s, c) = fit(&w[r * cols + 32 * k..r * cols + 32 * k + 32]);
            blocks[(r * runs + k) * 2..(r * runs + k) * 2 + 2].copy_from_slice(&s.to_le_bytes());
            rows_place(bits, &c, k, runs, row);
        }
    }
    Some(Tiered { rows, cols, data, blocks, block_base: vec![0; rows * cols / STEP + 1], tiers: [tag << 24, 0, 0] })
}

/// A row-layout pack back to bf16 (row-major), the weights the kernel that reads its rows computes with; false where the pack is not a row layout's of its shape.
pub fn unpack_rows(p: &Tiered, out: &mut [u16]) -> bool {
    let (rows, cols) = (p.rows, p.cols);
    let Some(bits) = rows_bits(p.tiers[0] >> 24) else { return false };
    let (runs, stride) = (cols / 32, rows_stride(bits, cols));
    if rows == 0 || cols == 0 || !cols.is_multiple_of(32) || out.len() != rows * cols || p.data.len() != rows * stride || p.blocks.len() != rows * runs * 2 {
        return false;
    }
    let value = match bits {
        6 => int6_value,
        5 => int5_value,
        _ => int4_value,
    };
    let word = |row: &[u8], at: usize| u32::from_le_bytes([row[at], row[at + 1], row[at + 2], row[at + 3]]);
    let hb = 4 * (bits as usize - 4);
    for r in 0..rows {
        let row = &p.data[r * stride..(r + 1) * stride];
        for k in 0..runs {
            let s = u16::from_le_bytes([p.blocks[(r * runs + k) * 2], p.blocks[(r * runs + k) * 2 + 1]]);
            let lo: [u32; 4] = std::array::from_fn(|q| word(row, 16 * k + 4 * q));
            let hi: [u32; 2] = std::array::from_fn(|q| if 4 * q < hb { word(row, 16 * runs + hb * k + 4 * q) } else { 0 });
            for i in 0..32 {
                let (pr, h) = (i / 2, i % 2);
                let mut u = (lo[pr / 4] >> (4 * (pr % 4) + 16 * h)) & 15;
                u |= match bits {
                    6 => ((hi[pr / 8] >> (2 * (pr % 8) + 16 * h)) & 3) << 4,
                    5 => ((hi[0] >> (pr + 16 * h)) & 1) << 4,
                    _ => 0,
                };
                out[r * cols + 32 * k + i] = value(u as u8, s);
            }
        }
    }
    true
}

/// [`pack_rows`], decoded back and compared with the value of each weight's own code and its run's scale (the check [`pack_int4_checked`] and its kin make of the block layouts): None where one is not those, bit for
/// bit, or where pack_rows is None.
pub fn pack_rows_checked(bits: u32, w: &[u16], rows: usize, cols: usize) -> Option<Tiered> {
    let p = pack_rows(bits, w, rows, cols)?;
    let (fit, value): (RunFit, fn(u8, u16) -> u16) = match bits {
        6 => (int6_run, int6_value),
        5 => (int5_run, int5_value),
        _ => (int4_run, int4_value),
    };
    let mut back = vec![0u16; w.len()];
    let agrees = unpack_rows(&p, &mut back)
        && w.chunks(32).zip(back.chunks(32)).all(|(run, got)| {
            let (s, c) = fit(run);
            got.iter().zip(&c).all(|(&g, &u)| g == value(u, s))
        });
    agrees.then_some(p)
}

/// W in either layout ([`pack_smallest`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Packed {
    Tiered(Tiered),
    Twelve(Twelve),
}

/// The 12-bit layout's arrays and words of W [rows, cols] in bytes, as glyd.gpu's Mma12.nbytes counts them, from the
/// exponents' counts alone: the exceptions are the weights outside the 8 values of exponent >> 1 from the base.
pub fn twelve_bytes(h: &[u64; 256], rows: usize, cols: usize) -> u64 {
    let (steps, hb) = ((rows * cols / STEP) as u64, twelve_base(h) as usize);
    let held: u64 = (hb..hb + 8).map(|x| h[2 * x] + h[2 * x + 1]).sum();
    let exceptions = (rows * cols) as u64 - held;
    1536 * steps + 4 * (exceptions + 4 - exceptions % 4) + 4 * (steps + 1) + 16
}

fn smallest(w: &[u16], rows: usize, cols: usize, check: bool) -> Option<Packed> {
    let t = tiered(w, rows, cols, check)?;
    let steps = rows * cols / STEP;
    let fits = tiered_bytes(&t) <= twelve_bytes(&histogram(w), rows, cols) && cold_steps(&t.block_base) * COLD_SHARE.1 <= steps * COLD_SHARE.0;
    if fits {
        Some(Packed::Tiered(t))
    } else {
        Some(Packed::Twelve(twelve(w, rows, cols, check)?))
    }
}

/// W [rows, cols] in the smaller of its two layouts: the tiered one, unless the 12-bit one takes fewer bytes or more
/// than a quarter of the tiered one's steps take the general decode.
pub fn pack_smallest(w: &[u16], rows: usize, cols: usize) -> Packed {
    smallest(w, rows, cols, false).unwrap()
}

/// The same, each step decoded back as it is made: None where one is not its weights, bit for bit.
pub fn pack_smallest_checked(w: &[u16], rows: usize, cols: usize) -> Option<Packed> {
    smallest(w, rows, cols, true)
}

#[cfg(test)]
mod compact_tests {
    use super::*;

    /// The time of one Kimi-K3 expert's first matrix (6144 x 3584) through the packed-scale FP4 pack and its check (`cargo test --release -- --ignored fp4r_pack_time --nocapture`).
    #[test]
    #[ignore]
    fn fp4r_pack_time() {
        let (rows, cols) = (6144usize, 3584usize);
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let q: Vec<u8> = (0..rows * cols / 2).map(|_| rnd() as u8).collect();
        let e: Vec<u8> = (0..rows * cols / 32).map(|_| 118 + (rnd() % 3) as u8).collect();
        let t = std::time::Instant::now();
        let code: Vec<u8> = q.iter().flat_map(|&b| [b & 15, b >> 4]).collect();
        let t_code = t.elapsed();
        let data = int4_steps(&code, rows, cols);
        let t_steps = t.elapsed();
        let p = pack_fp4r_checked(&q, &e, rows, cols).unwrap();
        let t_all = t.elapsed();
        assert_eq!(p.data, data);
        let mut back = vec![0u16; rows * cols];
        let t1 = std::time::Instant::now();
        assert!(unpack_fp4(&p, &mut back));
        let t_unpack = t1.elapsed();
        println!("codes {:?}, steps {:?}, whole pack and check {:?}, of it the unpack {:?}", t_code, t_steps - t_code, t_all - t_steps, t_unpack);
    }

    /// rne_clamp is round_ties_even then the clamp, value for value: every half, the clamp edges, the specials and 4 million random quotients.
    #[test]
    fn rne_clamp_is_round_ties_even_clamped() {
        let same = |x: f32, q: f32| {
            let (a, b) = (x.round_ties_even().clamp(-q, q), rne_clamp(x, q));
            assert!(a == b || (a.is_nan() && b.is_nan()), "{x} at {q}: {a} vs {b}");
        };
        let mut seed = 0x5EED_2026u64;
        for q in [3.0f32, 7.0, 15.0, 31.0, 63.0] {
            for k in -200..=200 {
                for f in [0.0f32, 0.25, 0.5, 0.75, 0.4999999, 0.5000001] {
                    same(k as f32 + f, q);
                    same(-(k as f32) - f, q);
                }
            }
            for x in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN, f32::MAX, f32::MIN, f32::MIN_POSITIVE, -0.0, 0.0, 1e-30, -1e-30] {
                same(x, q);
            }
            for _ in 0..800_000 {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                same(f32::from_bits(seed as u32) % 100.0, q);
                same(((seed >> 32) as u32 as f32 / u32::MAX as f32 - 0.5) * 4.0 * q, q);
            }
        }
    }

    /// The tiered layout packed on several threads is the one thread's, byte for byte (blocks, starts, general-decode flags), and decodes to its weights.
    #[test]
    fn tiered_on_threads_is_the_same_bytes() {
        let (rows, cols) = (192, 2048);
        let mut seed = 0x2026_1017u64;
        let w: Vec<u16> = (0..rows * cols)
            .map(|i| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                // (a bell of exponents with a tail, and a few non-finite words, so that steps take escapes, literals and the general decode)
                let e = 118 + (seed % 12) as u16 + if i % 997 == 0 { (seed >> 20) as u16 % 100 } else { 0 };
                ((seed >> 32) as u16 & 0x8000) | (e.min(255) << 7) | ((seed >> 40) as u16 & 0x7F)
            })
            .collect();
        let one = tiered_on(&w, rows, cols, true, 1).unwrap();
        for t in [2, 3, 7, 64] {
            let many = tiered_on(&w, rows, cols, true, t).unwrap();
            assert!(one.data == many.data && one.blocks == many.blocks && one.block_base == many.block_base && one.tiers == many.tiers, "{t} threads");
        }
        assert!(cold_steps(&one.block_base) > 0, "some steps take the general decode");
        let mut back = vec![0u16; w.len()];
        assert!(unpack_tiered(&one, &mut back) && back == w);
    }

    /// round_a as round_candidate_a: ties to even, the largest past the top, non-finite words kept; then every rounded
    /// finite word packed compact and back, and a matrix of noise; 768 bytes a step; an unrounded weight refused.
    #[test]
    fn compact_round_trips_every_rounded_word() {
        assert_eq!(
            [round_a(0x3F88), round_a(0x3F98), round_a(0x3F87), round_a(0x7F7F), round_a(0xFF7F), round_a(0x7FC1), round_a(0x7F80)],
            [0x3F80, 0x3FA0, 0x3F80, 0x7F70, 0xFF70, 0x7FC1, 0x7F80]
        );
        let mut words: Vec<u16> = (0..=u16::MAX).map(round_a).filter(|w| w & 0x7F80 != 0x7F80).collect();
        words.truncate(words.len() / 1024 * 1024);
        for (rows, cols) in [(64, words.len() / 64), (words.len() / 16, 16)] {
            let w = &words[..rows * cols];
            let p = pack_compact_checked(w, rows, cols).expect("compact");
            assert_eq!((p.tiers[0] >> 24, p.data.len()), (COMPACT_TAG, rows * cols / STEP * COMPACT_STEP_BYTES));
            let mut back = vec![0u16; w.len()];
            assert!(unpack_tiered(&expand_compact(&p).unwrap(), &mut back) && back == w);
        }
        let mut seed = 11u64;
        let noise: Vec<u16> = (0..256 * 512)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                round_a(((seed >> 20) as u16 & 0x80FF) | ((112 + (seed >> 40) % 16) as u16) << 7)
            })
            .collect();
        let p = pack_compact_checked(&noise, 256, 512).unwrap();
        assert!(p.tiers[0] >> 24 == COMPACT_TAG && expand_compact(&p).is_some());
        assert!(pack_compact_checked(&[0x3F81; 1024], 64, 16).is_none());
    }
}

#[cfg(test)]
mod int6_tests {
    use super::*;

    /// Every weight back as int6_value of its own code and its run's scale (the study's mse fit, computed here from the weights, apart from
    /// the packer: of the 16 ratios of torch.linspace(0.70, 1.0, 16) in float32, the first whose bf16 scale of amax r / 31 leaves the least
    /// squared error), the largest error half a step, a clamp's excess and the BF16 rounding; a run of zeros back as zeros; the shape and
    /// non-finite weights refused.
    #[test]
    fn int6_round_trips_as_its_runs_say() {
        let (rows, cols) = (128, 96);
        let w = tests::weights(rows * cols, 0.0, 3);
        let p = pack_int6(&w, rows, cols).expect("finite weights pack");
        assert_eq!((p.tiers[0] >> 24, p.data.len(), p.blocks.len()), (INT6_TAG, rows * cols / STEP * INT6_STEP_BYTES, rows * cols / 32 * 2));
        let mut back = vec![0u16; w.len()];
        assert!(unpack_int6(&p, &mut back));
        let step = (1.0f32 - 0.70f32) / 15.0f32;
        let mut fitted = 0;
        for r in 0..rows {
            for k in 0..cols / 32 {
                let run = &w[r * cols + 32 * k..r * cols + 32 * k + 32];
                let amax = run.iter().map(|&x| bf16_f32(x).abs()).fold(0f32, f32::max);
                let (mut s, mut least, mut pick) = (0f32, f32::INFINITY, 0);
                for i in 0..16 {
                    let ratio = if i < 8 { 0.70f32 + step * i as f32 } else { 1.0f32 - step * (15 - i) as f32 };
                    let cand = bf16_f32(f32_bf16(amax * ratio / 31.0));
                    let mut err = 0f32;
                    for &x in run {
                        let d = rne_clamp(bf16_f32(x) / cand, 31.0) * cand - bf16_f32(x);
                        err += d * d;
                    }
                    if err < least {
                        (s, least, pick) = (cand, err, i);
                    }
                }
                fitted += (pick != 15) as usize;
                for (c, &x) in run.iter().enumerate() {
                    let got = bf16_f32(back[r * cols + 32 * k + c]);
                    let code = rne_clamp(bf16_f32(x) / s, 31.0);
                    assert_eq!(got, bf16_f32(f32_bf16(code * s)), "row {r} run {k} place {c}");
                    assert!(
                        (got - bf16_f32(x)).abs() <= 0.5 * s * 1.0001 + (bf16_f32(x).abs() - 31.0 * s).max(0.0) + amax * 3.0 / 512.0,
                        "row {r} run {k} place {c}"
                    ); // (half a step; a clamp at 31 and the bf16 rounding)
                }
            }
        }
        assert!(fitted > 0, "no run takes a ratio under 1.0, so this does not check the fit");
        let zeros = vec![0u16; 64 * 32];
        let z = pack_int6(&zeros, 64, 32).unwrap();
        let mut zb = vec![1u16; zeros.len()];
        assert!(unpack_int6(&z, &mut zb) && zb.iter().all(|&x| x == 0));
        assert!(pack_int6(&w[..64 * 48], 64, 48).is_none() && pack_int6(&w[..32 * 96], 32, 96).is_none());
        let mut inf = w.clone();
        inf[5] = 0x7F80;
        assert!(pack_int6(&inf, rows, cols).is_none());
    }

    /// The bf16 the decode's two instructions make, (128 + u) - 160 then times s, rounded once, is int6_value for every code and a spread of
    /// scales (the decode's arithmetic, done here in f32 as fma.rn.bf16x2 does it: exactly, then one rounding).
    #[test]
    fn int6_value_is_the_decodes_arithmetic() {
        for s in [0x3C00u16, 0x3A1F, 0x0080, 0x0001, 0x7F7F, 0xBF80, 0x2C55] {
            for u in 0..64u8 {
                let x = bf16_f32(0x4300 | u as u16) - 160.0;
                assert_eq!(x, u as f32 - 32.0);
                assert_eq!(int6_value(u, s), f32_bf16(x * bf16_f32(s)));
            }
        }
    }
}

#[cfg(test)]
mod int5_tests {
    use super::*;

    /// Every weight back as int5_value of its own code and its run's scale (the study's mse fit, computed here from the weights, apart from
    /// the packer: of the 16 ratios of torch.linspace(0.70, 1.0, 16) in float32, the first whose bf16 scale of amax r / 15 leaves the least
    /// squared error), the largest error half a step, a clamp's excess and the BF16 rounding; a run of zeros back as zeros; the shape and
    /// non-finite weights refused, and a pack of another tag or sizes not unpacked.
    #[test]
    fn int5_round_trips_as_its_runs_say() {
        let (rows, cols) = (128, 96);
        let w = tests::weights(rows * cols, 0.0, 3);
        let p = pack_int5(&w, rows, cols).expect("finite weights pack");
        assert_eq!((p.tiers[0] >> 24, p.data.len(), p.blocks.len()), (INT5_TAG, rows * cols / STEP * INT5_STEP_BYTES, rows * cols / 32 * 2));
        let mut back = vec![0u16; w.len()];
        assert!(unpack_int5(&p, &mut back));
        let step = (1.0f32 - 0.70f32) / 15.0f32;
        let mut fitted = 0;
        for r in 0..rows {
            for k in 0..cols / 32 {
                let run = &w[r * cols + 32 * k..r * cols + 32 * k + 32];
                let amax = run.iter().map(|&x| bf16_f32(x).abs()).fold(0f32, f32::max);
                let (mut s, mut least, mut pick) = (0f32, f32::INFINITY, 0);
                for i in 0..16 {
                    let ratio = if i < 8 { 0.70f32 + step * i as f32 } else { 1.0f32 - step * (15 - i) as f32 };
                    let cand = bf16_f32(f32_bf16(amax * ratio / 15.0));
                    let mut err = 0f32;
                    for &x in run {
                        let d = rne_clamp(bf16_f32(x) / cand, 15.0) * cand - bf16_f32(x);
                        err += d * d;
                    }
                    if err < least {
                        (s, least, pick) = (cand, err, i);
                    }
                }
                fitted += (pick != 15) as usize;
                for (c, &x) in run.iter().enumerate() {
                    let got = bf16_f32(back[r * cols + 32 * k + c]);
                    let code = rne_clamp(bf16_f32(x) / s, 15.0);
                    assert_eq!(got, bf16_f32(f32_bf16(code * s)), "row {r} run {k} place {c}");
                    assert!(
                        (got - bf16_f32(x)).abs() <= 0.5 * s * 1.0001 + (bf16_f32(x).abs() - 15.0 * s).max(0.0) + amax * 3.0 / 512.0,
                        "row {r} run {k} place {c}"
                    ); // (half a step; a clamp at 15 and the bf16 rounding)
                }
            }
        }
        assert!(fitted > 0, "no run takes a ratio under 1.0, so this does not check the fit");
        let zeros = vec![0u16; 64 * 32];
        let z = pack_int5(&zeros, 64, 32).unwrap();
        let mut zb = vec![1u16; zeros.len()];
        assert!(unpack_int5(&z, &mut zb) && zb.iter().all(|&x| x == 0));
        assert!(pack_int5(&w[..64 * 48], 64, 48).is_none() && pack_int5(&w[..32 * 96], 32, 96).is_none());
        let mut inf = w.clone();
        inf[5] = 0x7F80;
        assert!(pack_int5(&inf, rows, cols).is_none());
        let (mut other, mut long, mut few) = (p.clone(), p.clone(), p.clone());
        other.tiers[0] = INT6_TAG << 24;
        long.data.push(0);
        few.blocks.pop();
        assert!(!unpack_int5(&other, &mut back) && !unpack_int5(&long, &mut back) && !unpack_int5(&few, &mut back) && !unpack_int5(&p, &mut back[1..]));
    }

    /// The bf16 the decode's two instructions make, (128 + u) - 144 then times s, rounded once, is int5_value for every code and a spread of
    /// scales (the decode's arithmetic, done here in f32 as fma.rn.bf16x2 does it: exactly, then one rounding).
    #[test]
    fn int5_value_is_the_decodes_arithmetic() {
        for s in [0x3C00u16, 0x3A1F, 0x0080, 0x0001, 0x7F7F, 0xBF80, 0x2C55] {
            for u in 0..32u8 {
                let x = bf16_f32(0x4300 | u as u16) - 144.0;
                assert_eq!(x, u as f32 - 16.0);
                assert_eq!(int5_value(u, s), f32_bf16(x * bf16_f32(s)));
            }
        }
    }
}

#[cfg(test)]
mod int4_int3_tests {
    use super::*;

    type Pack = fn(&[u16], usize, usize) -> Option<Tiered>;
    type Unpack = fn(&Tiered, &mut [u16]) -> bool;
    /// A layout: its packer, its decode, the value of a code, q (the codes run -q to q), its tag and a step's bytes.
    type Case = (Pack, Unpack, fn(u8, u16) -> u16, f32, u32, usize);
    const LAYOUTS: [Case; 2] =
        [(pack_int4, unpack_int4, int4_value, 7.0, INT4_TAG, INT4_STEP_BYTES), (pack_int3, unpack_int3, int3_value, 3.0, INT3_TAG, INT3_STEP_BYTES)];

    /// Every weight back as the value of its own code and its run's scale (the study's mse fit, computed here from the weights, apart from
    /// the packer: of the 16 ratios of torch.linspace(0.70, 1.0, 16) in float32, the first whose bf16 scale of amax r / q leaves the least
    /// squared error), the largest error half a step, a clamp's excess and the BF16 rounding; a run of zeros back as zeros; the shape and
    /// non-finite weights refused, and a pack of another tag or sizes not unpacked.
    #[test]
    fn four_and_three_bit_packs_round_trip_as_their_runs_say() {
        for (pack, unpack, _, q, tag, step_bytes) in LAYOUTS {
            let (rows, cols) = (128, 96);
            let w = tests::weights(rows * cols, 0.0, 3);
            let p = pack(&w, rows, cols).expect("finite weights pack");
            assert_eq!((p.tiers[0] >> 24, p.data.len(), p.blocks.len()), (tag, rows * cols / STEP * step_bytes, rows * cols / 32 * 2));
            let mut back = vec![0u16; w.len()];
            assert!(unpack(&p, &mut back));
            let step = (1.0f32 - 0.70f32) / 15.0f32;
            let mut fitted = 0;
            for r in 0..rows {
                for k in 0..cols / 32 {
                    let run = &w[r * cols + 32 * k..r * cols + 32 * k + 32];
                    let amax = run.iter().map(|&x| bf16_f32(x).abs()).fold(0f32, f32::max);
                    let (mut s, mut least, mut pick) = (0f32, f32::INFINITY, 0);
                    for i in 0..16 {
                        let ratio = if i < 8 { 0.70f32 + step * i as f32 } else { 1.0f32 - step * (15 - i) as f32 };
                        let cand = bf16_f32(f32_bf16(amax * ratio / q));
                        let mut err = 0f32;
                        for &x in run {
                            let d = rne_clamp(bf16_f32(x) / cand, q) * cand - bf16_f32(x);
                            err += d * d;
                        }
                        if err < least {
                            (s, least, pick) = (cand, err, i);
                        }
                    }
                    fitted += (pick != 15) as usize;
                    for (c, &x) in run.iter().enumerate() {
                        let got = bf16_f32(back[r * cols + 32 * k + c]);
                        let code = rne_clamp(bf16_f32(x) / s, q);
                        assert_eq!(got, bf16_f32(f32_bf16(code * s)), "q {q} row {r} run {k} place {c}");
                        assert!(
                            (got - bf16_f32(x)).abs() <= 0.5 * s * 1.0001 + (bf16_f32(x).abs() - q * s).max(0.0) + amax * 3.0 / 512.0,
                            "q {q} row {r} run {k} place {c}"
                        ); // (half a step; a clamp at q and the bf16 rounding)
                    }
                }
            }
            assert!(fitted > 0, "q {q}: no run takes a ratio under 1.0, so this does not check the fit");
            let zeros = vec![0u16; 64 * 32];
            let z = pack(&zeros, 64, 32).unwrap();
            let mut zb = vec![1u16; zeros.len()];
            assert!(unpack(&z, &mut zb) && zb.iter().all(|&x| x == 0));
            assert!(pack(&w[..64 * 48], 64, 48).is_none() && pack(&w[..32 * 96], 32, 96).is_none());
            let mut inf = w.clone();
            inf[5] = 0x7F80;
            assert!(pack(&inf, rows, cols).is_none());
            let (mut other, mut long, mut few) = (p.clone(), p.clone(), p.clone());
            other.tiers[0] = INT6_TAG << 24;
            long.data.push(0);
            few.blocks.pop();
            assert!(!unpack(&other, &mut back) && !unpack(&long, &mut back) && !unpack(&few, &mut back) && !unpack(&p, &mut back[1..]));
        }
        // (each takes the other's pack for no pack of its own: another tag, other sizes)
        let w = tests::weights(64 * 32, 0.0, 3);
        let (mut back, four, three) = (vec![0u16; w.len()], pack_int4(&w, 64, 32).unwrap(), pack_int3(&w, 64, 32).unwrap());
        assert!(!unpack_int4(&three, &mut back) && !unpack_int3(&four, &mut back) && !unpack_int5(&four, &mut back) && !unpack_int6(&three, &mut back));
    }

    /// The bf16 the decode's two instructions make, (128 + u) - (128 + 2^(bits - 1)) then times s, rounded once, is the value for every code and a
    /// spread of scales (the decode's arithmetic, done here in f32 as fma.rn.bf16x2 does it: exactly, then one rounding); the lowest code, which
    /// the packers leave out, stands for -(q + 1) s.
    #[test]
    fn four_and_three_bit_values_are_the_decodes_arithmetic() {
        for (value, bits, off) in [(int4_value as fn(u8, u16) -> u16, 4, 136.0f32), (int3_value, 3, 132.0)] {
            for s in [0x3C00u16, 0x3A1F, 0x0080, 0x0001, 0x7F7F, 0xBF80, 0x2C55] {
                for u in 0..1u8 << bits {
                    let x = bf16_f32(0x4300 | u as u16) - off;
                    assert_eq!(x, u as f32 - (1 << (bits - 1)) as f32);
                    assert_eq!(value(u, s), f32_bf16(x * bf16_f32(s)));
                }
            }
        }
        assert_eq!(
            [int4_value(0, 0x3F80), int4_value(8, 0x3F80), int4_value(15, 0x3F80), int3_value(0, 0x3F80), int3_value(4, 0x3F80), int3_value(7, 0x3F80)],
            [0xC100, 0, 0x40E0, 0xC080, 0, 0x4040]
        );
    }
}

#[cfg(test)]
mod fit_tests {
    use super::*;

    /// The ratios are torch.linspace(0.70, 1.0, 16) in float32: the first 0.70f32, the last 1.0f32, each above the one before and within a
    /// float32's rounding of 0.70 + 0.02 i.
    #[test]
    fn the_ratios_run_from_0_70_to_1_0() {
        assert_eq!((FIT_RATIOS[0], FIT_RATIOS[15]), (0.70f32, 1.0f32));
        assert!(FIT_RATIOS.windows(2).all(|p| p[0] < p[1]));
        for (i, &r) in FIT_RATIOS.iter().enumerate() {
            assert!((r as f64 - (0.70 + 0.02 * i as f64)).abs() < 1e-6, "ratio {i}: {r}");
        }
    }

    /// Over 20000 runs of a fixed seed (bell-shaped, a spread of 0.005 to 0.08, every second run with one weight 5 to 45 times the spread), the
    /// fit's scale leaves no more squared error than amax / q's (the candidate of ratio 1.0), for six, five, four and three bits, and less in a tenth
    /// of the runs at least (a fit that is amax / q would pass the first).
    #[test]
    fn the_fit_is_never_worse_than_amax_over_q() {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut unif = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / (1u32 << 24) as f32 - 0.5
        };
        for (q, fit) in [(31.0f32, int6_run as fn(&[u16]) -> (u16, [u8; 32])), (15.0, int5_run), (7.0, int4_run), (3.0, int3_run)] {
            let (runs, mut better) = (20_000, 0);
            for n in 0..runs {
                let spread = 0.02 * 2f32.powf(4.0 * unif());
                let mut run = [0u16; 32];
                for w in run.iter_mut() {
                    *w = f32_bf16(spread * (unif() + unif() + unif() + unif()));
                }
                if n % 2 == 1 {
                    run[n % 32] = f32_bf16(spread * (5.0 + 40.0 * (unif() + 0.5)) * if unif() < 0.0 { -1.0 } else { 1.0 });
                }
                let amax = run.iter().map(|&w| bf16_f32(w).abs()).fold(0f32, f32::max);
                let err = |s: f32| {
                    run.iter().fold(0f32, |e, &w| {
                        let w = bf16_f32(w);
                        let d = rne_clamp(w / s, q) * s - w;
                        e + d * d
                    })
                };
                let (fitted, plain) = (err(bf16_f32(fit(&run).0)), err(bf16_f32(f32_bf16(amax / q))));
                assert!(fitted <= plain, "q {q} run {n}: {fitted} over {plain}");
                better += (fitted < plain) as usize;
            }
            assert!(better * 10 > runs, "q {q}: less error than amax / q in {better} of {runs} runs");
        }
    }

    /// The edges of the rule: a run of zeros, and one whose every scale rounds to 0 in bf16 (its largest the least subnormal), has scale 0 and
    /// the zero code; one whose lower ratios give scale 0 and the rest the least subnormal has that scale; one whose every error is infinite
    /// (weights of 2^97 and over, the squares overflow) has its first candidate's scale, the first of equals.
    #[test]
    fn the_fit_at_its_edges() {
        for (fit, q, zero, small) in
            [(int6_run as fn(&[u16]) -> (u16, [u8; 32]), 31.0f32, 32u8, 18u16), (int5_run, 15.0, 16, 9), (int4_run, 7.0, 8, 4), (int3_run, 3.0, 4, 2)]
        {
            assert_eq!(fit(&[0; 32]), (0, [zero; 32]));
            assert_eq!(fit(&[1; 32]), (0, [zero; 32]));
            let (mut run, mut codes) = ([0u16; 32], [zero; 32]);
            (run[7], codes[7]) = (small, zero + small as u8);
            assert_eq!(fit(&run), (1, codes));
            let big: [u16; 32] = std::array::from_fn(|i| 0x7000 | ((3 * i as u16) % 128));
            let amax = big.iter().map(|&w| bf16_f32(w)).fold(0f32, f32::max);
            assert_eq!(fit(&big).0, f32_bf16(amax * 0.70f32 / q));
        }
    }

    /// The bound changes no run: int6_run, int5_run, int4_run and int3_run give the scale and the codes of [`tests::unpruned_run`] (every candidate summed in
    /// full) for 35000 runs a layout of a fixed seed, in seven kinds: bell-shaped, the same with an outlier, one to four weights, bf16
    /// subnormals, weights of 2^97 and over (the sums overflow), one magnitude of both signs, and a grid of multiples of 2^-6 up to q.
    #[test]
    fn the_bound_changes_no_run() {
        let mut x = 0xD1B5_4A32_D192_ED03u64;
        let mut unif = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / (1u32 << 24) as f32 - 0.5
        };
        for (q, fit) in [(31.0f32, int6_run as fn(&[u16]) -> (u16, [u8; 32])), (15.0, int5_run), (7.0, int4_run), (3.0, int3_run)] {
            for n in 0..35_000 {
                let spread = 0.02 * 2f32.powf(4.0 * unif());
                let mut run = [0u16; 32];
                match n % 7 {
                    0 | 1 => {
                        for w in run.iter_mut() {
                            *w = f32_bf16(spread * (unif() + unif() + unif() + unif()));
                        }
                        if n % 7 == 1 {
                            run[n % 32] = f32_bf16(spread * (5.0 + 40.0 * (unif() + 0.5)) * if unif() < 0.0 { -1.0 } else { 1.0 });
                        }
                    }
                    2 => {
                        for k in 0..1 + n / 7 % 4 {
                            run[(n + 5 * k) % 32] = f32_bf16(spread * 4.0 * unif());
                        }
                    }
                    3 => {
                        for w in run.iter_mut() {
                            *w = ((unif() + 0.5) * 128.0) as u16 | if unif() < 0.0 { 0x8000 } else { 0 };
                        }
                    }
                    4 => {
                        for w in run.iter_mut() {
                            *w = (0x7000 + ((unif() + 0.5) * 3968.0) as u16) | if unif() < 0.0 { 0x8000 } else { 0 };
                        }
                    }
                    5 => {
                        for w in run.iter_mut() {
                            *w = f32_bf16(spread) | if unif() < 0.0 { 0x8000 } else { 0 };
                        }
                    }
                    _ => {
                        for w in run.iter_mut() {
                            *w = f32_bf16((2.0 * q * unif()).round() / 64.0);
                        }
                        run[n % 32] = f32_bf16(q / 64.0);
                    }
                }
                assert_eq!(fit(&run), tests::unpruned_run(&run, q), "q {q} run {n}: {run:04x?}");
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// bf16 weights of a trained matrix's spread: most exponents near 2^-6, `wild` of them far off (every exponent
    /// 0-255 among them, NaNs and infinities too), from a fixed seed.
    pub(crate) fn weights(n: usize, wild: f64, seed: u64) -> Vec<u16> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        (0..n)
            .map(|_| {
                let r = next();
                let sign = (r & 1) as u16;
                if ((r >> 8) % 1_000_000) as f64 / 1e6 < wild {
                    sign << 15 | ((r >> 32) as u16 & 0x7FFF)
                } else {
                    let e = 115 + ((r >> 40) % 12) as u16 + ((r >> 48) % 3) as u16;
                    sign << 15 | e << 7 | ((r >> 20) as u16 & 0x7F)
                }
            })
            .collect()
    }

    /// The scale and the codes of a run as the rule says (q 31 for six bits, 15 for five, 7 for four, 3 for three), every candidate summed in full: what the
    /// layouts' fits (int6_run, int5_run, int4_run, int3_run) give without their bound.
    pub(crate) fn unpruned_run(run: &[u16], q: f32) -> (u16, [u8; 32]) {
        let w: Vec<f32> = run.iter().map(|&x| bf16_f32(x)).collect();
        let amax = w.iter().fold(0f32, |m, x| m.max(x.abs()));
        let (mut s, mut least) = (0u16, 0f32);
        for r in FIT_RATIOS {
            let c = f32_bf16(amax * r / q);
            if c == 0 {
                continue;
            }
            let cf = bf16_f32(c);
            let err = w.iter().fold(0f32, |e, &x| {
                let d = rne_clamp(x / cf, q) * cf - x;
                e + d * d
            });
            if s == 0 || err < least {
                (s, least) = (c, err);
            }
        }
        let mut codes = [q as u8 + 1; 32];
        if s != 0 {
            let sf = bf16_f32(s);
            for (c, &x) in codes.iter_mut().zip(&w) {
                *c = (rne_clamp(x / sf, q) as i32 + q as i32 + 1) as u8;
            }
        }
        (s, codes)
    }

    /// W [64 (rb + ballast), 16 ks] whose step s of the first rb row blocks has `t3(s)` weights of ranks 6-9, `lit(s)`
    /// weights past rank 9 and the rest of `t1(s)` escapes of ranks 3-5 (the other weights of ranks 0-2); the ballast
    /// row blocks hold exponents 100-105 alone, 100-102 three times as often as 103-105, so that those are the
    /// six commonest, then 106-109 the next four (the ranks the steps count by). No random numbers (the Python
    /// packer's test makes the same matrices): in step s the weight of place q (0-1023, a lane's 32 in turn) is
    /// the class-th q' of (389 q + 5 s) mod 1024, q' = 0.. in the order literals, ranks 6-9, ranks 3-5, ranks 0-2.
    fn profile(rb: usize, ks: usize, t3: impl Fn(usize) -> usize, lit: impl Fn(usize) -> usize, t1: impl Fn(usize) -> usize, ballast: usize) -> Vec<u16> {
        let (rows, cols) = (64 * (rb + ballast), 16 * ks);
        let places = places(cols);
        let mut w = vec![0u16; rows * cols];
        for s in 0..(rb + ballast) * ks {
            let first = (s / ks) * 64 * cols + (s % ks) * 16;
            let (a, b, c) = (t1(s).min(STEP), t3(s), lit(s));
            for q in 0..STEP {
                let e: usize = if s >= rb * ks {
                    match (q + 3 * s) % 12 {
                        k @ 0..=8 => 100 + k % 3,
                        k => 94 + k,
                    }
                } else if q < c {
                    150 + (7 * q + s) % 100
                } else if q < c + b {
                    106 + (q + s) % 4
                } else if q < a.max(c + b) {
                    103 + (q + s) % 3
                } else {
                    100 + (q + s) % 3
                };
                w[first + places[(389 * q + 5 * s) % STEP]] = (((q + s) & 1) as u16) << 15 | (e as u16) << 7 | ((13 * q + 5 * s) & 0x7F) as u16;
            }
        }
        w
    }

    /// What the decode of the steps without the flag makes of a step: its reads through the 32 words of the
    /// 128 bytes from the block's first word (a shuffle of word k + 1 where k is 31 reads word 0: any lane may), the
    /// digits a lane's word of tier-2 digits has, the tier-3 digits from `o`, the literals one a lane from the list.
    fn hot_exponents(d: &[u8], blocks: &[u8], blk: usize, end: usize, sym: &[u8; 10]) -> [u8; STEP] {
        let word = |at: usize| u32::from_le_bytes(blocks[at..at + 4].try_into().unwrap());
        let win: [u32; 32] = std::array::from_fn(|l| word((blk & !3) + 4 * l));
        let win_bits = |pos: usize| (((win[((pos >> 5) + 1) % 32] as u64) << 32 | win[(pos >> 5) % 32] as u64) >> (pos & 31)) as u32;
        let escapes = |x: u32| x & (x >> 1) & 0x5555_5555;
        let pd: [[u32; 2]; 32] = std::array::from_fn(|l| {
            [u32::from_le_bytes(d[l * 4..l * 4 + 4].try_into().unwrap()), u32::from_le_bytes(d[128 + l * 4..128 + l * 4 + 4].try_into().unwrap())]
        });
        let n1: [usize; 32] = std::array::from_fn(|l| (escapes(pd[l][0]).count_ones() + escapes(pd[l][1]).count_ones()) as usize);
        let t1: usize = n1.iter().sum();
        assert!(t1 <= HOT_ESCAPES);
        let words = t1.div_ceil(16);
        let da: [u32; 32] = std::array::from_fn(|l| if l < words { word(end - 4 * (l + 1)) } else { 0 });
        let na: [usize; 32] = std::array::from_fn(|l| escapes(da[l]).count_ones() as usize);
        let t2: usize = na.iter().sum();
        let o = 8 * (blk & 3);
        let mut scratch = [0u8; 1040];
        let mut before = 0;
        for l in 0..32 {
            let d3 = win_bits(o + 2 * before);
            let mut k3 = 0;
            for q in 0..16 {
                let digit = (da[l] >> (2 * q)) & 3;
                scratch[16 * l + q] = if digit < 3 {
                    sym[3 + digit as usize]
                } else {
                    k3 += 1;
                    sym[6 + ((d3 >> (2 * (k3 - 1))) & 3) as usize]
                };
            }
            before += na[l];
        }
        let r0 = (2 * t2 + 7) >> 3;
        let nl = 8 * (end - blk - r0 - 4 * words) / 18;
        assert!(nl <= HOT_LITERALS);
        for l in 0..nl {
            let x = win_bits(o + 8 * r0 + 18 * l);
            scratch[(x & 0x3FF) as usize] = (x >> 10) as u8;
        }
        let mut exp = [0u8; STEP];
        let mut c = 0;
        for l in 0..32 {
            for h in 0..2 {
                for j in 0..16 {
                    let digit = (pd[l][h] >> (2 * j)) & 3;
                    exp[l * 32 + 16 * h + j] = if digit < 3 {
                        sym[digit as usize]
                    } else {
                        c += 1;
                        scratch[c - 1]
                    };
                }
            }
        }
        exp
    }

    /// A step's counts of tier-1 escapes, tier-3 digits and literals, read from its pack.
    fn counts(p: &Tiered, s: usize) -> (usize, usize, usize) {
        let d = &p.data[s * 1280..s * 1280 + 256];
        let block = &p.blocks[block_start(p.block_base[s])..block_start(p.block_base[s + 1])];
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
        let t2 = (0..t1).filter(|&k| (block[size - 4 * ((k >> 4) + 1) + ((k & 15) >> 2)] >> (2 * (k & 3))) & 3 == 3).count();
        let list = size - 4 * t1.div_ceil(16) - ((2 * t2 + 7) >> 3);
        (t1, t2, 8 * list / 18)
    }

    /// Every step of the pack that has no flag, through the reads of the other decode: its exponents, and so its
    /// weights, are the matrix's. The steps with the flag by what it is for (more than 512 escapes, more than 32
    /// literals, tier-3 digits past the window, a literal's entry past it) and the steps without, the nearest of them
    /// to the window's edge in bits.
    fn flags_are_right(w: &[u16], rows: usize, cols: usize) -> ([usize; 4], usize, usize) {
        let p = pack_tiered(w, rows, cols);
        let sym = tier_symbols(&p.tiers);
        let (places, mut why, mut hot, mut nearest) = (places(cols), [0; 4], 0, usize::MAX);
        let mut u = [0u16; STEP];
        for (s, first) in steps(cols, 0, rows * cols / STEP) {
            let (t1, t2, nl) = counts(&p, s);
            let (start, o) = (block_start(p.block_base[s]), 8 * (block_start(p.block_base[s]) & 3));
            assert_eq!(p.block_base[s] < 0, is_cold(start, t1, t2, nl), "step {s}: the flag is is_cold's");
            if p.block_base[s] < 0 {
                let r0 = (2 * t2 + 7) >> 3;
                for (i, reason) in [t1 > HOT_ESCAPES, nl > HOT_LITERALS, o + 2 * t2 + 32 >= HOT_WINDOW, nl > 0 && o + 8 * r0 + 18 * (nl - 1) >= HOT_WINDOW]
                    .into_iter()
                    .enumerate()
                {
                    why[i] += reason as usize;
                }
                continue;
            }
            hot += 1;
            let r0 = (2 * t2 + 7) >> 3;
            nearest = nearest.min(HOT_WINDOW - (o + 2 * t2 + 32)).min(if nl > 0 { HOT_WINDOW - (o + 8 * r0 + 18 * (nl - 1)) } else { usize::MAX });
            gather(w, first, &places, &mut u);
            let e = hot_exponents(&p.data[s * 1280..(s + 1) * 1280], &p.blocks, start, block_start(p.block_base[s + 1]), &sym);
            for (i, &v) in u.iter().enumerate() {
                assert_eq!(e[i], (v >> 7) as u8, "step {s}, weight {i}: a step without the flag decodes to another exponent");
            }
        }
        (why, hot, nearest)
    }

    #[test]
    fn round_trips() {
        for &(o, k, wild) in &[(64, 16, 0.0), (64, 64, 0.5), (128, 48, 0.02), (192, 1040, 0.001), (256, 4096, 0.1), (64, 16, 1.0)] {
            let w = weights(o * k, wild, (o * k) as u64);
            let t = pack_tiered(&w, o, k);
            assert!(pack_tiered_checked(&w, o, k).as_ref() == Some(&t), "tiered {o}x{k} {wild}: checked");
            let mut back = vec![0u16; o * k];
            assert!(unpack_tiered(&t, &mut back) && back == w, "tiered {o}x{k} {wild}");
            assert_eq!((t.data.len(), t.block_base.len()), (o * k / 1024 * 1280, o * k / 1024 + 1));
            assert!(tiered_blocks_hold(&t.data, &t.blocks, &t.block_base), "tiered {o}x{k} {wild}: its blocks");
            assert_eq!(t.tiers[0] >> 24, TIERED_TAG);
            let q = pack_twelve(&w, o, k);
            assert!(pack_twelve_checked(&w, o, k).as_ref() == Some(&q), "12-bit {o}x{k} {wild}: checked");
            let mut back = vec![0u16; o * k];
            assert!(unpack_twelve(&q, &mut back) && back == w, "12-bit {o}x{k} {wild}");
            assert!(q.exc.len().is_multiple_of(4) && q.exc.len() > *q.exc_base.last().unwrap() as usize);
        }
    }

    /// A tiered pack whose data or blocks were changed where the decode reads by them: its blocks no longer hold it
    /// (and offsets that fall to a negative one are refused before any sum of them, in a debug build too).
    #[test]
    fn blocks_hold_their_escapes() {
        let w = weights(128 * 256, 0.2, 5);
        let t = pack_tiered(&w, 128, 256);
        let hold = |p: &Tiered| tiered_blocks_hold(&p.data, &p.blocks, &p.block_base);
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
        let mut flagged_end = t.clone();
        *flagged_end.block_base.last_mut().unwrap() |= i32::MIN; // the end of the last block has no flag
        assert!(!hold(&flagged_end));
        assert!(!tiered_blocks_hold(&[0; 1280], &[0; 1024], &[128, -1])); // offsets falling to a negative one (no overflow)
    }

    /// The flag against what the other decode reads: steps at the edges of its limits (the window's 992 bits, 32
    /// literals, 512 escapes), every start of a block mod 4, and matrices of the usual spread.
    #[test]
    fn a_step_without_the_flag_decodes_in_the_window() {
        let (mut why, mut hot, mut nearest) = ([0usize; 4], 0, usize::MAX);
        let mut run = |w: &[u16], rows, cols| {
            let (y, h, n) = flags_are_right(w, rows, cols);
            (0..4).for_each(|i| why[i] += y[i]);
            hot += h;
            nearest = nearest.min(n);
        };
        // tier-3 digits up to the window's edge (t2 sweeping 400-511 of 512 escapes), a few literals
        let w = profile(1, 96, |s| 440 + s * 3 / 4, |s| s % 3, |_| 512, 3);
        run(&w, 64 * 4, 16 * 96);
        // literals sweeping 0-40 among tier-3 digits of 150-250
        let w = profile(1, 90, |s| 150 + s, |s| s * 2 / 5, |_| 420, 3);
        run(&w, 64 * 4, 16 * 90);
        // escapes around 512 (t1: the steps' rank 3 and on)
        let w = profile(1, 64, |s| s, |_| 0, |s| 480 + s, 3);
        run(&w, 64 * 4, 16 * 64);
        // literals up to the window's edge, among a few tier-3 digits (rank 6-9 weights a few: the list starts early)
        let w = profile(1, 100, |_| 20, |s| 10 + s % 23, |_| 300, 3);
        run(&w, 64 * 4, 16 * 100);
        for &(o, k, wild) in &[(64, 64, 0.0), (128, 48, 0.02), (192, 1040, 0.001), (256, 4096, 0.1), (64, 16, 1.0), (256, 1024, 0.3)] {
            let w = weights(o * k, wild, (o * k) as u64 + 3);
            run(&w, o, k);
        }
        eprintln!("the flag: {} for more than 512 escapes, {} for more than 32 literals, {} for tier-3 digits past the window, {} for a literal past it; {hot} steps without, the nearest {nearest} bits from the window's edge", why[0], why[1], why[2], why[3]);
        assert!(why.iter().all(|&n| n > 0) && hot > 100 && nearest < 64, "the test reaches every reason for the flag, and steps just inside the window");
    }

    /// The layout a matrix is packed in: the tiered one of a trained matrix's spread, the 12-bit one where its
    /// exponents are spread wide (it takes fewer bytes) and where a quarter of the steps take the general decode
    /// whatever the sizes (here a tiered pack smaller than the 12-bit one, 600 escapes a step).
    #[test]
    fn smallest_by_size_and_by_the_general_decode() {
        let which = |w: &[u16], o, k| match pack_smallest_checked(w, o, k).unwrap() {
            Packed::Tiered(t) => (true, tiered_bytes(&t), cold_steps(&t.block_base)),
            Packed::Twelve(_) => (false, 0, 0),
        };
        let w = profile(4, 64, |_| 5, |_| 1, |_| 70, 0);
        let (tiered, bytes, cold) = which(&w, 256, 1024);
        assert!(
            tiered && cold == 0 && bytes < twelve_bytes(&histogram(&w), 256, 1024),
            "a peaked spread of exponents: tiered ({bytes} bytes, {cold} steps with the flag)"
        );
        let w = weights(256 * 1024, 1.0, 9);
        assert!(!which(&w, 256, 1024).0, "every exponent alike: the 12-bit layout takes fewer bytes");
        // 600 escapes in every step (ranks 3-5, with ballast for the six commonest), so a pack of it is under 12 bits: all of them take the flag
        let w = profile(4, 64, |_| 0, |_| 0, |_| 600, 3);
        let t = pack_tiered(&w, 448, 1024);
        assert!(
            tiered_bytes(&t) < twelve_bytes(&histogram(&w), 448, 1024),
            "the tiered pack is the smaller ({} bytes against {})",
            tiered_bytes(&t),
            twelve_bytes(&histogram(&w), 448, 1024)
        );
        assert!(cold_steps(&t.block_base) * 4 > 7 * 64, "and more than a quarter of its steps have the flag ({} of 448)", cold_steps(&t.block_base));
        assert!(!which(&w, 448, 1024).0, "and the 12-bit layout is the pick");
        // the 12-bit layout's size from the counts is the packer's
        for &(o, k, wild) in &[(64, 16, 0.0), (64, 64, 0.5), (128, 48, 0.02), (192, 1040, 0.001), (256, 4096, 0.1), (64, 16, 1.0)] {
            let w = weights(o * k, wild, (o * k) as u64);
            let q = pack_twelve(&w, o, k);
            assert_eq!(twelve_bytes(&histogram(&w), o, k), (q.data.len() + 4 * q.exc.len() + 4 * q.exc_base.len() + 16) as u64, "{o}x{k} {wild}");
        }
    }

    /// The tiered packer against kernels.pack_mma, byte for byte: the sha256 of each pack's data, blocks, block_base and
    /// tiers (little-endian), the steps that have the flag and the length of its blocks, as the Python packer
    /// (glyd-gpu-py/tests/test_gpu.py has the same table: it packs the same matrices on the CPU) makes them: a trained
    /// matrix's spread with 0-100% wild weights, every bf16 bit pattern, and matrices whose steps are at the edges of
    /// the flag's limits.
    #[test]
    fn tiered_is_pack_mma() {
        use sha2::{Digest, Sha256};
        let every: Vec<u16> = (0..65536u32).map(|i| i as u16 ^ 0x8000).collect();
        let near = |f: fn(u16) -> u16| -> Vec<u16> { every.iter().enumerate().map(|(i, &v)| if i % 10 != 0 { f(v) } else { v }).collect() };
        let mut cases: Vec<(Vec<u16>, usize, usize)> = [(64, 16, 0.0), (64, 64, 0.5), (128, 48, 0.02), (192, 1040, 0.001), (256, 4096, 0.1), (64, 16, 1.0)]
            .iter()
            .map(|&(o, k, wild)| (weights(o * k, wild, (o * k) as u64), o, k))
            .collect();
        cases.push((every.clone(), 256, 256));
        cases.push((near(|v| v & 0x807F), 256, 256));
        cases.push((near(|v| v | 0x7F00), 256, 256));
        cases.push((profile(1, 96, |s| 440 + s * 3 / 4, |s| s % 3, |_| 512, 3), 256, 16 * 96));
        cases.push((profile(1, 90, |s| 150 + s, |s| s * 2 / 5, |_| 420, 3), 256, 16 * 90));
        cases.push((profile(1, 64, |s| s, |_| 0, |s| 480 + s, 3), 256, 16 * 64));
        cases.push((profile(1, 100, |_| 20, |s| 10 + s % 23, |_| 300, 3), 256, 16 * 100));
        let mut got = Vec::new();
        for (w, o, k) in &cases {
            let p = pack_tiered(w, *o, *k);
            let mut h = Sha256::new();
            h.update(&p.data);
            h.update(&p.blocks);
            p.block_base.iter().for_each(|x| h.update(x.to_le_bytes()));
            p.tiers.iter().for_each(|x| h.update(x.to_le_bytes()));
            got.push((cold_steps(&p.block_base), p.blocks.len(), h.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>()));
        }
        let want = [
            ("weights(1024, 0.0, 1024)", 1, 977, "b3600cb10d2363283b301f3d46c851733a7e789d0504cae9c9870e32cf993349"),
            ("weights(4096, 0.5, 4096)", 4, 6674, "d59d2f07b5b94191bc6362bfbfb830cbfe1a3e065cb57e445a513db2d229dbaf"),
            ("weights(6144, 0.02, 6144)", 6, 4604, "76ab1ebc4c36559143b24140d832b27dd06f447371d90e7de953a9b0d7d0e57a"),
            ("weights(199680, 0.001, 199680)", 195, 130862, "d7e06f0db432b53e01e3116eb1af0f77b71a8d537241efc59e82e4ff41b0d245"),
            ("weights(1048576, 0.1, 1048576)", 1024, 870909, "2d2985701f83455f599b0764592c3fc6c936a52d10ef77581016a0a2bf8e1300"),
            ("weights(1024, 1.0, 1024)", 1, 2742, "354f42722d6718934a449f54a1cb98ceb0572d2622f0e2ca15beffcd3db4b81d"),
            ("every pattern", 64, 158528, "441de428786dc3bb2ed61c92dbc1bef88078c0ccba3643b92182f45cc7d286c0"),
            ("near v & 0x807F", 64, 16396, "8f57b9d5727595b3ebd047cfa7a3e0282eb7f6d125a2d9ae9f77889b358a020b"),
            ("near v | 0x7F00", 64, 16400, "4b30cf8613f689d2f3ceb23f360cbb33c6ab307b0c451d6af624fcfa9cbf7a4a"),
            ("profile 440 + s * 3 / 4, s % 3, 512", 50, 43094, "2ad49cf6d8cbf0c9cef2b07514713cedc65b7bba37534560918c386a80d0a7a5"),
            ("profile 150 + s, s * 2 / 5, 420", 14, 35621, "174601964156b6998b701967f4014d58c6c4b1da204a83571edf46bf41517123"),
            ("profile s, 0, 480 + s", 31, 21696, "7ef96bd486edfa18de91bb349f74b8bde9ea768904b41b29ff9a96dc5d31bd3a"),
            ("profile 20, 10 + s % 23, 300", 0, 32610, "80b5e600dcd410ef985feb5e354287e9c2bf4e0804affc27e9fb081d4e7a82ad"),
        ];
        assert_eq!(got.len(), want.len());
        for (g, (name, cold, len, sha)) in got.iter().zip(want) {
            assert_eq!((g.0, g.1, g.2.as_str()), (cold, len, sha), "{name}");
        }
    }

    /// The 12-bit packer against kernels.pack_mma12, byte for byte: each matrix's base, its exceptions, and the sha256
    /// of its data, exc and exc_base (little-endian), as the format helper's CPU emulation of pack_mma12 packs the
    /// same matrices (benchmarks/gpu/splitbyte-2026-09-29/emulate.py's pack12, with `weights` ported): a trained
    /// matrix's spread with 0-100% wild weights, every bf16 bit pattern, and the base at 0 and at 120.
    #[test]
    fn twelve_is_pack_mma12s() {
        use sha2::{Digest, Sha256};
        let every: Vec<u16> = (0..65536u32).map(|i| i as u16 ^ 0x8000).collect();
        let near = |f: fn(u16) -> u16| -> Vec<u16> { every.iter().enumerate().map(|(i, &v)| if i % 10 != 0 { f(v) } else { v }).collect() };
        let mut cases: Vec<(Vec<u16>, usize, usize)> = [(64, 16, 0.0), (64, 64, 0.5), (128, 48, 0.02), (192, 1040, 0.001), (256, 4096, 0.1), (64, 16, 1.0)]
            .iter()
            .map(|&(o, k, wild)| (weights(o * k, wild, (o * k) as u64), o, k))
            .collect();
        cases.push((every.clone(), 256, 256));
        cases.push((near(|v| v & 0x807F), 256, 256));
        cases.push((near(|v| v | 0x7F00), 256, 256));
        let want = [
            ("weights(1024, 0.0, 1024)", 57, 0, "27b2b8c015802cdb75e55b50ba1a9c1040fc8d6c7932014ff01e570825f395a7"),
            ("weights(4096, 0.5, 4096)", 57, 1941, "4301f3d82554a2bc384fe7095e8b0b447d8b2641f2767dc7442dd271e45398b0"),
            ("weights(6144, 0.02, 6144)", 57, 107, "8bfd176a4d52e5cb02204828855bc2029693de97ca8baaf461fd6fc6ff2f2cd8"),
            ("weights(199680, 0.001, 199680)", 57, 214, "0734426ff83a7af220955e402b5f9b4e60d76f0f39015e1289c324cbd92666db"),
            ("weights(1048576, 0.1, 1048576)", 57, 98349, "359001272bd4eeb92edc4a14ce524e727b943a97e017c593cfd823eb51d0b695"),
            ("weights(1024, 1.0, 1024)", 75, 935, "3a1e9bfb33218a4d619672e02d84b412bef679176ff2a7335802545dc95ab361"),
            ("every pattern", 0, 61440, "081fe3fd1fc08ec32ec60f972655405fc5aa90aeb55d8e829d78c0b2e2077c40"),
            ("hb 0", 0, 6144, "d917573e7bc816d5ed90312be7f176fbcbd9cb609cae292d497b3587ce4befff"),
            ("hb 120", 120, 6144, "f5ae65b3a161a39651d47e0d2f77e30febb5eab4a280300c1c9ac66323ebec0d"),
        ];
        for ((w, o, k), (name, hb, n, sha)) in cases.iter().zip(want) {
            let q = pack_twelve(w, *o, *k);
            let mut h = Sha256::new();
            h.update(&q.data);
            q.exc.iter().chain(&q.exc_base).for_each(|x| h.update(x.to_le_bytes()));
            let got = h.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>();
            assert_eq!((q.hb(), *q.exc_base.last().unwrap() as usize, got.as_str(), q.sym), (hb, n, sha, twelve_words(hb)), "{name}");
        }
    }

    #[test]
    fn by_count_is_torchs_stable_argsort() {
        let mut h = [0u64; 256];
        h[5] = 3;
        h[9] = 7;
        h[2] = 3;
        let o = by_count(&h);
        assert_eq!(&o[..5], &[9, 2, 5, 0, 1]);
    }
}

#[cfg(test)]
mod int_checked_tests {
    use super::*;

    /// A random matrix packs checked in the block layouts, to what pack_int6 / pack_int5 / pack_int4 / pack_int3 make; a flipped byte of the scales or the codes is None.
    #[test]
    fn int_checked_packs_match_and_refuse_a_corrupted_pack() {
        let (rows, cols) = (128, 96);
        let w = tests::weights(rows * cols, 0.0, 5);
        assert!(pack_int6_checked(&w, rows, cols) == pack_int6(&w, rows, cols) && pack_int6_checked(&w, rows, cols).is_some());
        assert!(pack_int5_checked(&w, rows, cols) == pack_int5(&w, rows, cols) && pack_int5_checked(&w, rows, cols).is_some());
        assert!(pack_int4_checked(&w, rows, cols) == pack_int4(&w, rows, cols) && pack_int4_checked(&w, rows, cols).is_some());
        assert!(pack_int3_checked(&w, rows, cols) == pack_int3(&w, rows, cols) && pack_int3_checked(&w, rows, cols).is_some());
        assert!(pack_int6_checked(&w[..64 * 48], 64, 48).is_none() && pack_int5_checked(&w[..32 * 96], 32, 96).is_none());
        assert!(pack_int4_checked(&w[..64 * 48], 64, 48).is_none() && pack_int3_checked(&w[..32 * 96], 32, 96).is_none());
        for (a, b) in [(&mut pack_int6(&w, rows, cols).unwrap(), 0), (&mut pack_int6(&w, rows, cols).unwrap(), 300)] {
            assert!(int6_agrees(&w, a));
            a.blocks[b] ^= 0x40;
            assert!(!int6_agrees(&w, a));
        }
        for (a, b) in [(&mut pack_int5(&w, rows, cols).unwrap(), 0), (&mut pack_int5(&w, rows, cols).unwrap(), 300)] {
            assert!(int5_agrees(&w, a));
            a.blocks[b] ^= 0x40;
            assert!(!int5_agrees(&w, a));
        }
        for (a, b) in [(&mut pack_int4(&w, rows, cols).unwrap(), 0), (&mut pack_int4(&w, rows, cols).unwrap(), 300)] {
            assert!(int4_agrees(&w, a));
            a.blocks[b] ^= 0x40;
            assert!(!int4_agrees(&w, a));
            a.blocks[b] ^= 0x40;
            a.data[b] ^= 0x10;
            assert!(!int4_agrees(&w, a));
        }
        for (a, b) in [(&mut pack_int3(&w, rows, cols).unwrap(), 0), (&mut pack_int3(&w, rows, cols).unwrap(), 300)] {
            assert!(int3_agrees(&w, a));
            a.blocks[b] ^= 0x40;
            assert!(!int3_agrees(&w, a));
            a.blocks[b] ^= 0x40;
            a.data[b] ^= 0x04;
            assert!(!int3_agrees(&w, a));
        }
    }

    /// Codes and scales no fit would make (every code -q to q, scales of both signs, zero, subnormal and the largest) packed as they are: every weight decodes to int_value of its code and its run's scale, bit
    /// for bit; where they are the fit's, the pack is the one pack_int makes; refused: a code past q, a scale not finite, sizes that are not the layout's; a changed byte of the codes or the scales does not hold.
    #[test]
    fn given_codes_and_scales_are_kept_bit_for_bit() {
        type Pack = fn(&[i8], &[u16], usize, usize) -> Option<Tiered>;
        let (rows, cols) = (128, 96);
        let w = tests::weights(rows * cols, 0.0, 5);
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for (q, pack, unpack, value, fit, tag) in [
            (
                31i8,
                pack_int6_runs_checked as Pack,
                unpack_int6 as fn(&Tiered, &mut [u16]) -> bool,
                int6_value as fn(u8, u16) -> u16,
                int6_run as fn(&[u16]) -> (u16, [u8; 32]),
                INT6_TAG,
            ),
            (15, pack_int5_runs_checked, unpack_int5, int5_value, int5_run, INT5_TAG),
        ] {
            let code: Vec<i8> = (0..rows * cols).map(|_| (next() % (2 * q as u64 + 1)) as i8 - q).collect();
            let special = [0x0000u16, 0x8000, 0x0001, 0x7F7F, 0xFF7F, 0x3C00, 0xBF80];
            let scale: Vec<u16> = (0..rows * cols / 32).map(|i| special.get(i).copied().unwrap_or_else(|| next() as u16 & 0xBF7F | 0x3000)).collect();
            assert!(scale.iter().all(|&s| s & 0x7F80 != 0x7F80));
            let p = pack(&code, &scale, rows, cols).expect("codes and scales pack");
            assert_eq!(p.tiers[0] >> 24, tag);
            let mut back = vec![0u16; code.len()];
            assert!(unpack(&p, &mut back));
            for (i, (&got, &c)) in back.iter().zip(&code).enumerate() {
                assert_eq!(got, value((c + q + 1) as u8, scale[i / 32]), "weight {i}");
            }
            // the fit's own codes and scales are packed as pack_int packs the weights
            let fitted: Vec<(u16, [u8; 32])> = w.chunks(32).map(fit).collect();
            let (c, s): (Vec<i8>, Vec<u16>) =
                (fitted.iter().flat_map(|f| f.1.iter().map(|&u| u as i8 - (q + 1))).collect(), fitted.iter().map(|f| f.0).collect());
            assert_eq!(pack(&c, &s, rows, cols), if q == 31 { pack_int6(&w, rows, cols) } else { pack_int5(&w, rows, cols) });
            // refused
            for bad in [q + 1, -(q + 1), i8::MIN, i8::MAX] {
                let mut c = code.clone();
                c[7] = bad;
                assert!(pack(&c, &scale, rows, cols).is_none(), "code {bad}");
            }
            for bad in [0x7F80u16, 0xFF80, 0x7FC0, 0x7F81] {
                let mut s = scale.clone();
                s[3] = bad;
                assert!(pack(&code, &s, rows, cols).is_none(), "scale {bad:#06x}");
            }
            assert!(pack(&code[..code.len() - 32], &scale, rows, cols).is_none() && pack(&code, &scale[1..], rows, cols).is_none());
            assert!(pack(&code[..64 * 48], &scale[..64 * 48 / 32], 64, 48).is_none() && pack(&code[..32 * cols], &scale[..32 * cols / 32], 32, cols).is_none());
            // a changed byte of the codes or of the scales is not what was given
            let u: Vec<u8> = code.iter().map(|&c| (c + q + 1) as u8).collect();
            let holds = |p: &Tiered| if q == 31 { int6_holds(&u, &scale, p) } else { int5_holds(&u, &scale, p) };
            assert!(holds(&p));
            let (mut bad_codes, mut bad_scales) = (p.clone(), p.clone());
            bad_codes.data[0] ^= 0x40;
            bad_scales.blocks[300] ^= 0x40;
            assert!(!holds(&bad_codes) && !holds(&bad_scales));
        }
    }
}

#[cfg(test)]
mod rows_tests {
    use super::*;

    /// (bits, q, the block layout's packer and decode, its step's bytes, the row layout's tag)
    type Case = (u32, f32, fn(&[u16], usize, usize) -> Option<Tiered>, fn(&Tiered, &mut [u16]) -> bool, u32);
    const CASES: [Case; 3] =
        [(6, 31.0, pack_int6, unpack_int6, INT6_ROW_TAG), (5, 15.0, pack_int5, unpack_int5, INT5_ROW_TAG), (4, 7.0, pack_int4, unpack_int4, INT4_ROW_TAG)];

    /// A row layout's codes (u) and scales read from its bytes as the layout's doc says, apart from the packer: a run's four words of low nibbles, then (five and six bits) its high bits' word or words.
    fn row_codes(bits: u32, p: &Tiered) -> (Vec<u8>, Vec<u16>) {
        let (runs, stride, hb) = (p.cols / 32, rows_stride(bits, p.cols), 4 * (bits as usize - 4));
        let word = |r: usize, at: usize| u32::from_le_bytes(p.data[r * stride + at..r * stride + at + 4].try_into().unwrap());
        let (mut u, mut s) = (Vec::new(), Vec::new());
        for r in 0..p.rows {
            for k in 0..runs {
                s.push(u16::from_le_bytes([p.blocks[(r * runs + k) * 2], p.blocks[(r * runs + k) * 2 + 1]]));
                for i in 0..32 {
                    let (pair, half) = (i / 2, i % 2);
                    let lo_bit = 4 * (pair % 4) + 16 * half;
                    let mut code = (word(r, 16 * k + 4 * (pair / 4)) >> lo_bit) & 15;
                    if bits == 5 {
                        code |= ((word(r, 16 * runs + hb * k) >> (pair + 16 * half)) & 1) << 4;
                    }
                    if bits == 6 {
                        code |= ((word(r, 16 * runs + hb * k + 4 * (pair / 8)) >> (2 * (pair % 8) + 16 * half)) & 3) << 4;
                    }
                    u.push(code as u8);
                }
            }
        }
        (u, s)
    }

    /// A block layout's codes and scales read from its steps (the unpack's loops, returning the codes and the scales they would be multiplied by): row-major codes [rows x cols], scales [rows x cols / 32].
    fn block_codes(bits: u32, p: &Tiered) -> (Vec<u8>, Vec<u16>) {
        let (rows, cols) = (p.rows, p.cols);
        let (rbs, ks) = (rows / 64, cols / 16);
        let step_bytes = [INT4_STEP_BYTES, INT5_STEP_BYTES, INT6_STEP_BYTES][bits as usize - 4];
        let word = |b: &[u8], at: usize| u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]);
        let (mut u, mut s) = (vec![0u8; rows * cols], vec![0u16; rows * cols / 32]);
        for rb in 0..rbs {
            for k in 0..ks {
                let step = rb * ks + k;
                let d = &p.data[step * step_bytes..(step + 1) * step_bytes];
                let b = &p.blocks[step / 2 * 128..step / 2 * 128 + 128];
                for l in 0..32 {
                    // (the nibbles' words at 8 l + 4 q and 8 (l + 32) + 4 q after the high bits': five bits 128 bytes, six 256)
                    let base = [0, 128, 256][bits as usize - 4];
                    let lo = [word(d, base + 8 * l), word(d, base + 8 * l + 4), word(d, base + 8 * (l + 32)), word(d, base + 8 * (l + 32) + 4)];
                    for i in 0..32 {
                        let (pr, h) = (i / 2, i % 2);
                        let mut code = (lo[pr / 4] >> (4 * (pr % 4) + 16 * h)) & 15;
                        if bits == 5 {
                            code |= ((word(d, 4 * l) >> (pr + 16 * h)) & 1) << 4;
                        }
                        if bits == 6 {
                            let hi = [word(d, 4 * l), word(d, 4 * (32 + l))];
                            code |= ((hi[pr / 8] >> (2 * (pr % 8) + 16 * h)) & 3) << 4;
                        }
                        let (rr, cc) = lane_place(l, i);
                        u[(rb * 64 + rr) * cols + 16 * k + cc] = code as u8;
                        let (g, n) = (l / 4, i / 4);
                        // (a pair of steps share their 64 rows' scales, those of the run k / 2)
                        s[(rb * 64 + 8 * n + g) * (cols / 32) + k / 2] = u16::from_le_bytes([b[2 * (8 * g + n)], b[2 * (8 * g + n) + 1]]);
                    }
                }
            }
        }
        (u, s)
    }

    /// The row layout holds the block layout's very codes and scales (the same fit and rounding, placed a row at a time): for weights of a trained matrix's spread, with runs of zeros, the extreme words and a row of
    /// one value, at the three widths; the codes and the scales read from the row bytes are the block layout's, and each run's are the fit's (`unpruned_run`, apart from the packers); every weight decodes to the block
    /// layout's, bit for bit; the sizes are a row's stride and a row's scales; the pack checked is the pack.
    #[test]
    fn rows_hold_the_block_layouts_codes_and_scales() {
        let (rows, cols) = (192, 96);
        let mut w = tests::weights(rows * cols, 0.0, 17);
        w[32..64].fill(0); // (a run of zeros: scale 0, every code the middle one)
        w[cols..2 * cols].fill(0x3F80); // (a row of ones)
        w[2 * cols] = 0x7F7F; // (the largest finite)
        w[2 * cols + 1] = 0xFF7F;
        w[3 * cols] = 0x0001; // (the smallest subnormal)
        for (bits, q, pack, unpack, tag) in CASES {
            let rp = pack_rows(bits, &w, rows, cols).expect("rows pack");
            let bp = pack(&w, rows, cols).unwrap();
            assert_eq!(
                (rp.tiers, rp.data.len(), rp.blocks.len(), rp.block_base.len()),
                ([tag << 24, 0, 0], rows * rows_stride(bits, cols), rows * cols / 32 * 2, rows * cols / STEP + 1),
                "{bits} bits"
            );
            assert_eq!(rows_bits(tag), Some(bits));
            let (ru, rs) = row_codes(bits, &rp);
            let (bu, bs) = block_codes(bits, &bp);
            assert_eq!(rs, bs, "{bits} bits: the scales");
            assert_eq!(ru, bu, "{bits} bits: the codes");
            for (i, run) in w.chunks(32).enumerate() {
                let (s, c) = tests::unpruned_run(run, q);
                assert_eq!((rs[i], &ru[32 * i..32 * i + 32]), (s, &c[..]), "{bits} bits: run {i}");
            }
            let (mut a, mut b) = (vec![0u16; w.len()], vec![0u16; w.len()]);
            assert!(unpack_rows(&rp, &mut a) && unpack(&bp, &mut b));
            assert!(a == b, "{bits} bits: the weights decoded");
            assert_eq!(pack_rows_checked(bits, &w, rows, cols), Some(rp.clone()));
            // (a pack of rows is not a block pack, and neither is read as the other)
            assert!(!unpack(&rp, &mut b) && !unpack_rows(&bp, &mut a));
        }
    }

    /// The layout's bytes pinned: one run of codes -q to q (every second one, from the middle of the range) of scale 2^-6, in the layout of each width, as the doc says (the bytes stated apart from the packer, tools/
    /// moe_emu_test.cpp's kernels read the very same). A row is `rows_stride` bytes: 16 at four bits, 32 at five and six (a 20- and a 24-byte row, to 16).
    #[test]
    fn the_layouts_bytes() {
        let lo: [u8; 16] = [0x51, 0xd9, 0x73, 0xfb, 0x51, 0xd9, 0x73, 0xfb, 0x62, 0xea, 0x84, 0x0c, 0x62, 0xea, 0x84, 0x1c];
        let wants: [(u32, Vec<u8>); 3] = [
            (4, vec![0x51, 0xd9, 0x73, 0xfb, 0x62, 0xea, 0x84, 0x1c, 0x73, 0xfb, 0x95, 0x2d, 0x84, 0x1c, 0xa6, 0x3e]),
            (5, [&lo[..], &[0xf0, 0xf0, 0xf0, 0x78], &[0u8; 12]].concat()),
            (
                6,
                [
                    0x51, 0xd9, 0x73, 0xfb, 0x51, 0xd9, 0x73, 0xfb, 0x51, 0xd9, 0x73, 0xfb, 0x51, 0xd9, 0x73, 0xfb, 0x00, 0x55, 0x00, 0x55, 0xaa, 0xff, 0xaa,
                    0xff, 0, 0, 0, 0, 0, 0, 0, 0,
                ]
                .to_vec(),
            ),
        ];
        for (bits, want) in wants {
            let q = (1i32 << (bits - 1)) - 1;
            // 32 rows of 32 weights (1,024): row 0 is the run, row r the run turned by r places
            let k: Vec<i32> = (0..32).map(|i| (2 * i) % (2 * q + 1) - q).collect();
            let w: Vec<u16> = (0..32 * 32).map(|i| f32_bf16(k[(i % 32 + i / 32) % 32] as f32 / 64.0)).collect();
            let p = pack_rows(bits, &w, 32, 32).unwrap();
            let stride = rows_stride(bits, 32);
            assert_eq!((stride, want.len()), (want.len(), want.len()), "{bits} bits");
            assert_eq!(&p.data[..stride], &want[..], "{bits} bits: row 0");
            assert_eq!(&p.blocks[..2], &[0x80, 0x3c], "{bits} bits: the scale 2^-6");
            // row r is the run turned: its codes are the first row's shifted
            let (u, s) = row_codes(bits, &p);
            for r in 0..32 {
                assert_eq!(s[r], 0x3c80);
                for i in 0..32 {
                    assert_eq!(u[r * 32 + i] as i32, k[(i + r) % 32] + q + 1, "{bits} bits row {r} place {i}");
                }
            }
        }
        // the strides of the real shapes: a down matrix of Nemotron 3 Super (1,024 outputs a neuron)
        assert_eq!([rows_stride(4, 1024), rows_stride(5, 1024), rows_stride(6, 1024)], [512, 640, 768]);
        assert_eq!([rows_stride(4, 64), rows_stride(5, 64), rows_stride(6, 64)], [32, 48, 48]);
    }

    /// The transpose is the matrix turned (a shape that is not a multiple of its tile too), and turned twice is the matrix.
    #[test]
    fn transpose_turns_a_matrix() {
        for (rows, cols) in [(1, 1), (3, 70), (70, 3), (33, 65), (64, 64), (2688, 96)] {
            let w: Vec<u16> = (0..rows * cols).map(|i| (i * 7 + 3) as u16).collect();
            let t = transpose(&w, rows, cols);
            assert!((0..rows).all(|r| (0..cols).all(|c| t[c * rows + r] == w[r * cols + c])), "{rows} x {cols}");
            assert_eq!(transpose(&t, cols, rows), w);
        }
    }

    /// What is not a row layout's matrix is refused (a shape, a non-finite weight, bits other than 4 to 6), and a pack that is not one is not decoded (another tag, other sizes, a buffer short); a row of zeros decodes to zeros.
    #[test]
    fn rows_refuse_what_they_cannot_hold() {
        let w = tests::weights(64 * 64, 0.0, 5);
        for bits in [4u32, 5, 6] {
            assert!(pack_rows(bits, &w, 64, 64).is_some());
            assert!(pack_rows(bits, &w[..64 * 48], 64, 48).is_none(), "48 columns are not runs of 32");
            assert!(pack_rows(bits, &w[..33 * 32], 33, 32).is_none(), "33 x 32 is not whole thousands of weights");
            assert!(pack_rows(bits, &w[..96 * 32], 96, 32).is_some(), "96 x 32 is");
            assert!(pack_rows(bits, &w[..63 * 64], 63, 64).is_none());
            assert!(pack_rows(bits, &w[1..], 64, 64).is_none() && pack_rows(bits, &[], 0, 64).is_none() && pack_rows(bits, &[], 64, 0).is_none());
            let mut inf = w.clone();
            inf[9] = 0x7F80;
            assert!(pack_rows(bits, &inf, 64, 64).is_none());
            inf[9] = 0xFFC0;
            assert!(pack_rows(bits, &inf, 64, 64).is_none() && pack_rows_checked(bits, &inf, 64, 64).is_none());
            let p = pack_rows(bits, &w, 64, 64).unwrap();
            let mut back = vec![0u16; w.len()];
            assert!(unpack_rows(&p, &mut back));
            let (mut other, mut long, mut few, mut base) = (p.clone(), p.clone(), p.clone(), p.clone());
            other.tiers[0] = INT4_TAG << 24;
            long.data.push(0);
            few.blocks.pop();
            base.cols = 32;
            assert!(
                !unpack_rows(&other, &mut back)
                    && !unpack_rows(&long, &mut back)
                    && !unpack_rows(&few, &mut back)
                    && !unpack_rows(&base, &mut back)
                    && !unpack_rows(&p, &mut back[1..])
            );
            let z = pack_rows(bits, &vec![0u16; 64 * 64], 64, 64).unwrap();
            let mut zb = vec![1u16; 64 * 64];
            assert!(unpack_rows(&z, &mut zb) && zb.iter().all(|&x| x == 0));
        }
        for bits in [0u32, 3, 7, 8] {
            assert!(pack_rows(bits, &w, 64, 64).is_none(), "{bits} bits");
        }
        assert_eq!([rows_bits(INT6_TAG), rows_bits(INT3_TAG), rows_bits(TIERED_TAG), rows_bits(0)], [None; 4]);
    }
}

#[cfg(test)]
mod int4h_tests {
    use super::*;

    /// A splitmix64 stream (the test's own, so that the weights are the same on every run).
    struct Mix(u64);
    impl Mix {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    /// Weights as a checkpoint trained to four bits holds them: bf16(k s), k from -8 to 7, s a 12-bit significand over 10 binades (Gemma 4 31B
    /// QAT's scales span 9), some runs with a negative scale (their codes -7 to 8 as positive ones) and some all zero.
    fn qat_like(rows: usize, cols: usize, seed: u64) -> Vec<u16> {
        let mut r = Mix(seed);
        let mut w = vec![0u16; rows * cols];
        for run in w.chunks_mut(32) {
            let kind = r.next() % 16;
            if kind == 0 {
                continue;
            }
            let s = (2048 + r.next() % 2048) as f32 * 2f32.powi(-25 - (r.next() % 10) as i32) * if kind == 1 { -1.0 } else { 1.0 };
            for x in run.iter_mut() {
                *x = f32_bf16(((r.next() % 16) as f32 - 8.0) * s);
            }
        }
        w
    }

    /// Four-bit-trained weights packed and decoded back bit for bit (the decode is the GPU's), at 4.5 bits a weight; other weights refused.
    #[test]
    fn exact_four_bit_pack_holds_trained_weights_bit_for_bit() {
        let (rows, cols) = (128, 256);
        let w = qat_like(rows, cols, 7);
        let p = pack_int4h_checked(&w, rows, cols).expect("four-bit-trained weights pack exactly");
        assert_eq!(p.tiers[0] >> 24, INT4H_TAG);
        assert_eq!((p.data.len() + p.blocks.len()) * 8, rows * cols * 9 / 2);
        let mut back = vec![0u16; w.len()];
        assert!(unpack_int4(&p, &mut back));
        assert_eq!(back, w);
        // a bf16 weight that is no code times its run's scale: refused, as are the shapes
        let mut r = Mix(9);
        let other: Vec<u16> = (0..rows * cols).map(|_| f32_bf16(((r.next() % 20001) as f32 - 10000.0) * 1e-6)).collect();
        assert!(pack_int4h(&other, rows, cols).is_none());
        assert!(pack_int4h(&w[..64 * 256], 32, 512).is_none());
        // scales spanning more than 16 binades: refused
        let mut wide = w.clone();
        for x in wide[..32].iter_mut() {
            *x = f32_bf16(bf16_f32(*x) * 2f32.powi(-20));
        }
        if wide[..32].iter().any(|&x| x & 0x7FFF != 0) {
            assert!(pack_int4h(&wide, rows, cols).is_none());
        }
    }
}

#[cfg(test)]
mod fp4_tests {
    use super::*;

    /// The exact FP4 layout holds an MX FP4 matrix bit for bit: each weight back as its E2M1 value times 2^(e - 127) (the five-bit runs' value of twice the code and half the scale), and refuses scales outside 2 to 128 (the decode's 2^(e - 1)) or whose
    /// products would leave bf16's normal range.
    #[test]
    fn an_fp4_matrix_packs_and_decodes_exactly() {
        let (rows, cols) = (128, 96);
        let mut r = 0x2026_1018_f4u64;
        let mut next = || {
            r ^= r << 13;
            r ^= r >> 7;
            r ^= r << 17;
            r
        };
        let q: Vec<u8> = (0..rows * cols / 2).map(|_| next() as u8).collect();
        let e: Vec<u8> = (0..rows * cols / 32).map(|_| 89 + (next() % 40) as u8).collect();
        let p = pack_fp4_checked(&q, &e, rows, cols).expect("packs");
        assert_eq!((p.data.len() + p.blocks.len()) * 8, rows * cols * 17 / 4, "4.25 bits a weight");
        let mut back = vec![0u16; rows * cols];
        assert!(unpack_fp4(&p, &mut back));
        const TWICE: [f32; 16] = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, -0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0];
        for (i, &v) in back.iter().enumerate() {
            let c = if i % 2 == 0 { q[i / 2] & 15 } else { q[i / 2] >> 4 };
            let want = TWICE[c as usize] * f32::from_bits(((e[i / 32] as u32) - 1) << 23);
            assert_eq!(v, (want.to_bits() >> 16) as u16, "weight {i}");
        }
        let r2 = pack_fp4r_checked(&q, &e, rows, cols);
        assert!(r2.is_none(), "these rows' scales span 40 values: not two bits");
        let narrow: Vec<u8> = e.iter().enumerate().map(|(i, &x)| 120 + (x + i as u8 / 7) % 4).collect();
        let p2 = pack_fp4r_checked(&q, &narrow, rows, cols).expect("packs with two-bit scales");
        assert_eq!((p2.data.len() + p2.blocks.len()) * 8, rows * cols * 4 + rows * cols / 16 + rows / 64 * 512, "4 bits, 2 a run and a base a row");
        let mut b2 = vec![0u16; rows * cols];
        assert!(unpack_fp4(&p2, &mut b2));
        assert!(b2.iter().enumerate().all(|(i, &v)| v == fp4_value(if i % 2 == 0 { q[i / 2] & 15 } else { q[i / 2] >> 4 }, narrow[i / 32])));
        let mut low = e.clone();
        low[3] = 1;
        assert!(pack_fp4_checked(&q, &low, rows, cols).is_none());
        let mut high = e.clone();
        high[5] = 129;
        assert!(pack_fp4_checked(&q, &high, rows, cols).is_none(), "the decode's 2^(e - 1) is a bf16 only to 128");
        assert!(pack_fp4_checked(&q, &e, 100, cols).is_none());
        // (the check reads every code and scale back: one nibble or one scale changed in the pack, it says no)
        let (mut d, b) = (p2.data.clone(), p2.blocks.clone());
        assert!(fp4_holds_in(&d, &b, rows, cols, true, &q, &narrow));
        for at in [0usize, 77, d.len() - 1] {
            d[at] ^= 0x10;
            assert!(!fp4_holds_in(&d, &b, rows, cols, true, &q, &narrow), "a nibble changed at byte {at}");
            d[at] ^= 0x10;
        }
        let mut b3 = b.clone();
        b3[3] ^= 1;
        assert!(!fp4_holds_in(&d, &b3, rows, cols, true, &q, &narrow), "a row's base changed");
        assert!((0..rows).all(|r| (0..cols).all(|c| int4_code(&d, cols, r, c) == fp4_code(&q, r * cols + c))), "the codes where int4_code reads them");
    }
}
