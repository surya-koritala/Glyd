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
//! digits over three tiers of the matrix's nine commonest exponents (digit 3
//! on to the next tier; past the third, the exponent's byte). In the 12-bit
//! layout (split byte) a weight's low byte is kept as it is, its high byte
//! (sign, exponent >> 1) a 4-bit code: the sign and an offset 0-7 from the
//! matrix's base hb (0-120, the first of the 8 values of exponent >> 1 that
//! hold the most weights); any other weight's offset is 0 and the byte to XOR
//! into its high byte is in the step's exception list.

/// Weights a step (64 rows by 16 columns).
pub const STEP: usize = 1024;

/// Where each of a step's 1024 weights is in W, from the step's first
/// (row 64 rb, column 16 ks), for rows `k` weights long.
fn places(k: usize) -> Vec<usize> {
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
fn by_count(h: &[u64; 256]) -> [u8; 256] {
    let mut e: [u8; 256] = std::array::from_fn(|i| i as u8);
    e.sort_by(|a, b| h[*b as usize].cmp(&h[*a as usize]).then(a.cmp(b)));
    e
}

/// A step's sign-and-mantissa bytes: weight i's in its half i / 16 of 512
/// bytes, at 16 lane + i % 16 (each 16-byte load of a warp: 512 contiguous
/// bytes).
fn sign_mantissa(u: &[u16; STEP], out: &mut [u8]) {
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
fn weights_of(sm: &[u8], exp: &[u8; STEP], out: &mut [u16; STEP]) {
    for lane in 0..32 {
        let (lo, hi) = (&sm[lane * 16..lane * 16 + 16], &sm[512 + lane * 16..512 + lane * 16 + 16]);
        for i in 0..32 {
            let (b, pair) = if i < 16 { (lo[i], lo[i ^ 1]) } else { (hi[i - 16], hi[(i ^ 1) - 16]) };
            out[lane * 32 + i] = ((pair & 1) as u16) << 15 | (exp[lane * 32 + i] as u16) << 7 | (b >> 1) as u16;
        }
    }
}

/// W in the tiered layout.
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

fn check_shape(w: &[u16], rows: usize, cols: usize) {
    assert!(rows.is_multiple_of(64) && cols.is_multiple_of(16) && w.len() == rows * cols, "an mma layout's matrix: rows a multiple of 64, columns of 16");
}

/// A step's weights gathered from W (its first at `at`).
fn gather(w: &[u16], at: usize, places: &[usize], u: &mut [u16; STEP]) {
    for (x, &p) in u.iter_mut().zip(places) {
        *x = w[at + p];
    }
}

/// Steps [s0, s1) of a matrix `cols` wide: each step's index and its first weight's place in W.
fn steps(cols: usize, s0: usize, s1: usize) -> impl Iterator<Item = (usize, usize)> {
    let ks = cols / 16;
    (s0..s1).map(move |s| (s, (s / ks) * 64 * cols + (s % ks) * 16))
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
    check_shape(w, rows, cols);
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
        block_base.push(i32::try_from(blocks.len()).expect("blocks past 2^31 bytes"));
        if check {
            tiered_unstep(&data[s * 1280..(s + 1) * 1280], &blocks[start..], &sym, &mut back, &mut scratch)?;
            if back != u {
                return None;
            }
        }
    }
    blocks.resize(blocks.len() + 256, 0);
    assert!(blocks.len() < 1 << 31, "blocks past 2^31 bytes");
    Some(Tiered { rows, cols, data, blocks, block_base, tiers })
}

/// W [rows, cols] in the tiered layout, as pack_mma makes it.
pub fn pack_tiered(w: &[u16], rows: usize, cols: usize) -> Tiered {
    tiered(w, rows, cols, false).unwrap()
}

/// The same, each step decoded back as it is made and compared with its weights (the check glyd.gpu's pack makes
/// of each pack on the GPU): None where one is not its weights, bit for bit.
pub fn pack_tiered_checked(w: &[u16], rows: usize, cols: usize) -> Option<Tiered> {
    tiered(w, rows, cols, true)
}

/// W back from the tiered layout, into out [rows, cols]; false where its buffers do not hold it (a step's block
/// past `blocks`, or shorter than its escapes).
pub fn unpack_tiered(p: &Tiered, out: &mut [u16]) -> bool {
    let n = p.rows * p.cols / STEP;
    if out.len() != p.rows * p.cols || p.data.len() != n * 1280 || p.block_base.len() != n + 1 {
        return false;
    }
    let sym: [u8; 9] = std::array::from_fn(|r| (p.tiers[r / 3] >> (8 * (r % 3))) as u8);
    let places = places(p.cols);
    let mut u = [0u16; STEP];
    let mut esc = Vec::with_capacity(STEP);
    for (s, at) in steps(p.cols, 0, n) {
        let (start, end) = (p.block_base[s] as usize, p.block_base[s + 1] as usize);
        if start > end || end > p.blocks.len() || tiered_unstep(&p.data[s * 1280..(s + 1) * 1280], &p.blocks[start..end], &sym, &mut u, &mut esc).is_none() {
            return false;
        }
        for (q, &v) in u.iter().enumerate() {
            out[at + places[q]] = v;
        }
    }
    true
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
        exc_base.push(i32::try_from(exc.len()).expect("exceptions past 2^31"));
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

/// Whether a tiered pack's buffers are as its packer makes them wherever the GPU's decode reads by them: `data`
/// [steps][1280], `block_base` [steps + 1] from 128 on, rising, 256 bytes of `blocks` past the last; each step's
/// block exactly as long as its escapes take (the tier-3 digits and bytes from its start, the tier-2 digits' words
/// back from its end, the counts read from the step's digits): the decode then reads inside its buffers. Its escapes'
/// counts alone, not its weights (a decode does that).
pub fn tiered_blocks_hold(data: &[u8], blocks: &[u8], block_base: &[i32]) -> bool {
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
        let t1: usize = d.as_chunks::<4>().0.iter().map(|w| {
            let w = u32::from_le_bytes(*w);
            (w & (w >> 1) & 0x5555_5555).count_ones() as usize
        }).sum();
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
        assert!(!tiered_blocks_hold(&[0; 1280], &[0; 1024], &[128, -1])); // offsets falling to a negative one (no overflow)
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
        let mut cases: Vec<(Vec<u16>, usize, usize)> = [(64, 16, 0.0), (64, 64, 0.5), (128, 48, 0.02), (192, 1040, 0.001), (256, 4096, 0.1), (64, 16, 1.0)].iter().map(|&(o, k, wild)| (weights(o * k, wild, (o * k) as u64), o, k)).collect();
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
