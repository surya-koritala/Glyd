//! Length-limited canonical Huffman for the byte-symbol streams.
//!
//! Codes are capped at `MAX_CODE_LEN` bits so decoding is a single table
//! lookup: peek that many bits, read (symbol, length), consume length.
//!
//! The measured entropy of the token stream is 5.344 bits per symbol against
//! the 8 it costs today, which is where the T1.2 ratio gap is closed.

pub const MAX_CODE_LEN: u32 = 11;
pub const TABLE_BITS: u32 = MAX_CODE_LEN;
pub const TABLE_SIZE: usize = 1 << TABLE_BITS;
/// 256 symbols, one nibble of code length each.
pub const LENGTHS_BYTES: usize = 128;

/// Build code lengths from a histogram, capped at MAX_CODE_LEN.
///
/// Uses ordinary Huffman construction, then halves the counts and rebuilds if
/// any code came out too long. Scaling shortens the tail without disturbing the
/// common symbols, and converges in a few rounds.
pub fn build_lengths(hist: &[u64; 256]) -> [u8; 256] {
    let mut counts: Vec<u64> = hist.to_vec();
    loop {
        let lengths = huffman_lengths(&counts);
        if lengths.iter().all(|&l| l as u32 <= MAX_CODE_LEN) {
            return lengths;
        }
        for c in counts.iter_mut() {
            if *c > 1 {
                *c = (*c + 1) / 2;
            }
        }
    }
}

fn huffman_lengths(counts: &[u64]) -> [u8; 256] {
    let mut lengths = [0u8; 256];
    let used: Vec<usize> = (0..256).filter(|&i| counts[i] > 0).collect();
    if used.is_empty() {
        return lengths;
    }
    if used.len() == 1 {
        lengths[used[0]] = 1; // a lone symbol still needs one bit
        return lengths;
    }

    // Node pool: leaves then internal nodes.
    let n = used.len();
    let mut weight: Vec<u64> = used.iter().map(|&s| counts[s]).collect();
    let mut left: Vec<i32> = vec![-1; n];
    let mut right: Vec<i32> = vec![-1; n];
    let mut live: Vec<usize> = (0..n).collect();

    while live.len() > 1 {
        // Two lightest nodes.
        live.sort_by_key(|&i| std::cmp::Reverse(weight[i]));
        let a = live.pop().unwrap();
        let b = live.pop().unwrap();
        let w = weight[a] + weight[b];
        weight.push(w);
        left.push(a as i32);
        right.push(b as i32);
        live.push(weight.len() - 1);
    }

    // Walk down assigning depths.
    let root = live[0];
    let mut stack = vec![(root, 0u32)];
    while let Some((node, d)) = stack.pop() {
        if left[node] < 0 {
            lengths[used[node]] = d.max(1) as u8;
        } else {
            stack.push((left[node] as usize, d + 1));
            stack.push((right[node] as usize, d + 1));
        }
    }
    lengths
}

/// Canonical codes from code lengths.
pub fn build_codes(lengths: &[u8; 256]) -> [u16; 256] {
    let mut bl_count = [0u32; (MAX_CODE_LEN + 1) as usize];
    for &l in lengths.iter() {
        if l > 0 {
            bl_count[l as usize] += 1;
        }
    }
    let mut next = [0u16; (MAX_CODE_LEN + 1) as usize];
    let mut code: u32 = 0;
    for bits in 1..=MAX_CODE_LEN as usize {
        code = (code + bl_count[bits - 1] as u32) << 1;
        next[bits] = code as u16;
    }
    let mut codes = [0u16; 256];
    for sym in 0..256 {
        let l = lengths[sym] as usize;
        if l > 0 {
            codes[sym] = next[l];
            next[l] += 1;
        }
    }
    codes
}

/// Pack 256 code lengths as nibbles. Lengths are <= 11 so they fit.
pub fn pack_lengths(lengths: &[u8; 256], out: &mut Vec<u8>) {
    for i in (0..256).step_by(2) {
        out.push((lengths[i] & 0x0F) | ((lengths[i + 1] & 0x0F) << 4));
    }
}

pub fn unpack_lengths(src: &[u8]) -> [u8; 256] {
    let mut l = [0u8; 256];
    for i in 0..128 {
        l[i * 2] = src[i] & 0x0F;
        l[i * 2 + 1] = src[i] >> 4;
    }
    l
}

/// MSB-first bit writer.
pub struct BitWriter {
    acc: u64,
    nbits: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self { acc: 0, nbits: 0 }
    }
    #[inline(always)]
    pub fn put(&mut self, code: u16, len: u8, out: &mut Vec<u8>) {
        self.acc = (self.acc << len as u32) | code as u64;
        self.nbits += len as u32;
        while self.nbits >= 8 {
            self.nbits -= 8;
            out.push((self.acc >> self.nbits) as u8);
        }
    }
    pub fn finish(&mut self, out: &mut Vec<u8>) {
        if self.nbits > 0 {
            out.push((self.acc << (8 - self.nbits)) as u8);
            self.nbits = 0;
        }
        self.acc = 0;
    }
}

/// Single-lookup decode table: index by the next TABLE_BITS bits.
pub struct DecodeTable {
    pub sym: [u8; TABLE_SIZE],
    pub len: [u8; TABLE_SIZE],
}

impl DecodeTable {
    pub fn build(lengths: &[u8; 256]) -> Option<Box<DecodeTable>> {
        let codes = build_codes(lengths);
        let mut t = Box::new(DecodeTable {
            sym: [0u8; TABLE_SIZE],
            len: [0u8; TABLE_SIZE],
        });
        for sym in 0..256usize {
            let l = lengths[sym] as u32;
            if l == 0 {
                continue;
            }
            if l > MAX_CODE_LEN {
                return None;
            }
            let shift = TABLE_BITS - l;
            let base = (codes[sym] as usize) << shift;
            let count = 1usize << shift;
            if base + count > TABLE_SIZE {
                return None;
            }
            for i in 0..count {
                t.sym[base + i] = sym as u8;
                t.len[base + i] = l as u8;
            }
        }
        Some(t)
    }
}

/// Decode `n` symbols from `src` using `table`. Returns bytes consumed, or None
/// on a malformed stream.
pub fn decode_into(table: &DecodeTable, src: &[u8], n: usize, out: &mut [u8]) -> Option<usize> {
    let mut acc: u64 = 0;
    let mut nbits: u32 = 0;
    let mut pos = 0usize;
    for i in 0..n {
        while nbits < TABLE_BITS {
            let byte = if pos < src.len() { src[pos] } else { 0 };
            pos += 1;
            acc = (acc << 8) | byte as u64;
            nbits += 8;
        }
        let idx = ((acc >> (nbits - TABLE_BITS)) & ((TABLE_SIZE as u64) - 1)) as usize;
        let l = table.len[idx];
        if l == 0 {
            return None; // no code with this prefix
        }
        nbits -= l as u32;
        *out.get_mut(i)? = table.sym[idx];
    }
    // Bytes actually consumed, rounding the partial accumulator back.
    Some(pos - (nbits / 8) as usize)
}
