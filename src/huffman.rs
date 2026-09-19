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
    let mut counts = *hist;
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

/// Plain Huffman over the used symbols, on fixed arrays and in linear
/// time after one sort (this runs two or three times per block: as a
/// stable re-sort per merge it was 150 us per block on Silesia, more than
/// the rest of the literal coder).
fn huffman_lengths(counts: &[u64; 256]) -> [u8; 256] {
    let mut lengths = [0u8; 256];
    let mut used = [0u8; 256];
    let mut n = 0usize;
    for s in 0..256 {
        if counts[s] > 0 {
            used[n] = s as u8;
            n += 1;
        }
    }
    if n == 0 {
        return lengths;
    }
    if n == 1 {
        lengths[used[0] as usize] = 1; // a lone symbol still needs one bit
        return lengths;
    }

    // Node pool: leaves 0..n, then internal nodes n..2n-1.
    let mut weight = [0u64; 511];
    for i in 0..n {
        weight[i] = counts[used[i] as usize];
    }
    let mut child = [[0u16; 2]; 511];
    // Two queues (Van Leeuwen): leaves by ascending weight, and merged
    // nodes in creation order, whose weights never decrease. The lightest
    // node is at the head of one of them. Ties reproduce the original
    // stable-sort version exactly -- it merged the two *last* nodes of a
    // list ordered by descending weight, then leaves by ascending symbol,
    // then merged nodes by creation -- so a leaf tie takes the highest
    // symbol first, a leaf/merged tie takes the merged node, and a tie
    // among merged nodes takes the newest, i.e. the last of the
    // equal-weight run at the head of that queue.
    let mut leaves = [0u16; 256];
    for i in 0..n {
        leaves[i] = i as u16;
    }
    leaves[..n].sort_unstable_by_key(|&i| (weight[i as usize], std::cmp::Reverse(i)));
    // Queue state: `leaves[lh..n]`, `merged[mh..mt]`.
    struct Q {
        leaves: [u16; 256],
        merged: [u16; 256],
        lh: usize,
        mh: usize,
        mt: usize,
        n: usize,
    }
    fn take(q: &mut Q, weight: &[u64; 511]) -> usize {
        let from_merged = q.mh < q.mt && (q.lh >= q.n || weight[q.merged[q.mh] as usize] <= weight[q.leaves[q.lh] as usize]);
        if from_merged {
            let w = weight[q.merged[q.mh] as usize];
            let mut e = q.mh + 1;
            while e < q.mt && weight[q.merged[e] as usize] == w {
                e += 1;
            }
            let node = q.merged[e - 1] as usize;
            q.merged.copy_within(q.mh..e - 1, q.mh + 1);
            q.mh += 1;
            node
        } else {
            q.lh += 1;
            q.leaves[q.lh - 1] as usize
        }
    }
    let mut q = Q { leaves, merged: [0u16; 256], lh: 0, mh: 0, mt: 0, n };
    let mut next = n;
    while next < 2 * n - 1 {
        let a = take(&mut q, &weight);
        let b = take(&mut q, &weight);
        weight[next] = weight[a] + weight[b];
        child[next] = [a as u16, b as u16];
        q.merged[q.mt] = next as u16;
        q.mt += 1;
        next += 1;
    }
    let root = next - 1;

    // Walk down assigning depths.
    let mut stack = [(0u16, 0u8); 512];
    let mut top = 1;
    stack[0] = (root as u16, 0);
    while top > 0 {
        top -= 1;
        let (node, d) = stack[top];
        if (node as usize) < n {
            lengths[used[node as usize] as usize] = d.max(1);
        } else {
            stack[top] = (child[node as usize][0], d + 1);
            stack[top + 1] = (child[node as usize][1], d + 1);
            top += 2;
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

/// The v8 table: nibbles, a length as itself and a run of 1-16 unused
/// symbols as 0 then the run minus one; the last nibble of an odd count
/// is a zero pad. Returns the bytes written.
pub fn pack_lengths_v8(lengths: &[u8; 256], out: &mut Vec<u8>) -> usize {
    let mut nibbles = Vec::with_capacity(256);
    let mut i = 0;
    while i < 256 {
        if lengths[i] == 0 {
            let mut run = 0;
            while i < 256 && lengths[i] == 0 && run < 16 {
                run += 1;
                i += 1;
            }
            nibbles.push(0);
            nibbles.push(run as u8 - 1);
        } else {
            debug_assert!(lengths[i] <= 15);
            nibbles.push(lengths[i]);
            i += 1;
        }
    }
    let start = out.len();
    for pair in nibbles.chunks(2) {
        out.push(pair[0] | pair.get(1).map_or(0, |&n| n << 4));
    }
    out.len() - start
}

/// Bytes `pack_lengths_v8` writes for `lengths`.
pub fn packed_lengths_v8_size(lengths: &[u8; 256]) -> usize {
    let mut n = 0usize;
    let mut i = 0;
    while i < 256 {
        if lengths[i] == 0 {
            let mut run = 0;
            while i < 256 && lengths[i] == 0 && run < 16 {
                run += 1;
                i += 1;
            }
            n += 2;
        } else {
            n += 1;
            i += 1;
        }
    }
    n.div_ceil(2)
}

/// Read a v8 table from the front of `src`: the lengths and the bytes it
/// took, or None if it runs past `src` or past 256 symbols.
pub fn unpack_lengths_v8(src: &[u8]) -> Option<([u8; 256], usize)> {
    let mut l = [0u8; 256];
    let (mut i, mut k) = (0usize, 0usize);
    let nibble = |k: usize| -> Option<u8> { src.get(k / 2).map(|&b| if k % 2 == 0 { b & 0x0F } else { b >> 4 }) };
    while i < 256 {
        let n = nibble(k)?;
        k += 1;
        if n == 0 {
            let run = nibble(k)? as usize + 1;
            k += 1;
            if i + run > 256 {
                return None;
            }
            i += run;
        } else {
            l[i] = n;
            i += 1;
        }
    }
    Some((l, k.div_ceil(2)))
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
