//! zlib's Huffman trees (trees.c), built from a block's tokens the way
//! zlib builds them, so a dynamic block's header can be predicted:
//! `build_tree` with its heap and its `depth` tie-break, `gen_bitlen`
//! with the overflow fix at 15 bits (7 for the code-length code), the
//! forced pair of codes, `scan_tree` and `send_tree`'s run lengths, and
//! `_tr_flush_block`'s choice between stored, fixed and dynamic.

use super::{dist_code, len_code, Header, Token, CLEN_ORDER, LEN_EXTRA};

const L_CODES: usize = 286;
const D_CODES: usize = 30;
const BL_CODES: usize = 19;
const HEAP_SIZE: usize = 2 * L_CODES + 1;
const END_BLOCK: usize = 256;
const REP_3_6: usize = 16;
const REPZ_3_10: usize = 17;
const REPZ_11_138: usize = 18;
const DIST_EXTRA_BITS: [u8; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const BL_EXTRA_BITS: [u8; 19] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 7];

/// A tree under construction: frequency, then code length, and the
/// parent while building.
#[derive(Clone, Copy, Default)]
struct Node {
    freq: u32,
    len: u16,
    dad: u16,
}

struct Builder {
    heap: [u16; HEAP_SIZE],
    heap_len: usize,
    heap_max: usize,
    depth: [u8; HEAP_SIZE],
    bl_count: [u16; 16],
    opt_len: u64,
    static_len: u64,
}

impl Builder {
    fn new() -> Self {
        Builder { heap: [0; HEAP_SIZE], heap_len: 0, heap_max: HEAP_SIZE, depth: [0; HEAP_SIZE], bl_count: [0; 16], opt_len: 0, static_len: 0 }
    }

    #[inline]
    fn smaller(&self, tree: &[Node], n: usize, m: usize) -> bool {
        tree[n].freq < tree[m].freq || (tree[n].freq == tree[m].freq && self.depth[n] <= self.depth[m])
    }

    fn pqdownheap(&mut self, tree: &[Node], mut k: usize) {
        let v = self.heap[k];
        let mut j = k << 1;
        while j <= self.heap_len {
            if j < self.heap_len && self.smaller(tree, self.heap[j + 1] as usize, self.heap[j] as usize) {
                j += 1;
            }
            if self.smaller(tree, v as usize, self.heap[j] as usize) {
                break;
            }
            self.heap[k] = self.heap[j];
            k = j;
            j <<= 1;
        }
        self.heap[k] = v;
    }

    /// trees.c's gen_bitlen: code lengths from the tree, at most
    /// `max_length`, the overflow moved down.
    fn gen_bitlen(&mut self, tree: &mut [Node], max_code: usize, stree: Option<&[u8]>, extra: &[u8], base: usize, max_length: usize) {
        self.bl_count = [0; 16];
        tree[self.heap[self.heap_max] as usize].len = 0;
        let mut overflow = 0i32;
        let mut h = self.heap_max + 1;
        while h < HEAP_SIZE {
            let n = self.heap[h] as usize;
            let mut bits = tree[tree[n].dad as usize].len as usize + 1;
            if bits > max_length {
                bits = max_length;
                overflow += 1;
            }
            tree[n].len = bits as u16;
            h += 1;
            if n > max_code {
                continue;
            }
            self.bl_count[bits] += 1;
            let xbits = if n >= base { extra[n - base] as u64 } else { 0 };
            let f = tree[n].freq as u64;
            self.opt_len += f * (bits as u64 + xbits);
            if let Some(stree) = stree {
                self.static_len += f * (stree[n] as u64 + xbits);
            }
        }
        if overflow == 0 {
            return;
        }
        loop {
            let mut bits = max_length - 1;
            while self.bl_count[bits] == 0 {
                bits -= 1;
            }
            self.bl_count[bits] -= 1;
            self.bl_count[bits + 1] += 2;
            self.bl_count[max_length] -= 1;
            overflow -= 2;
            if overflow <= 0 {
                break;
            }
        }
        let mut h = HEAP_SIZE;
        for bits in (1..=max_length).rev() {
            let mut n = self.bl_count[bits];
            while n != 0 {
                h -= 1;
                let m = self.heap[h] as usize;
                if m > max_code {
                    continue;
                }
                if tree[m].len as usize != bits {
                    self.opt_len = (self.opt_len as i64 + (bits as i64 - tree[m].len as i64) * tree[m].freq as i64) as u64;
                    tree[m].len = bits as u16;
                }
                n -= 1;
            }
        }
    }

    /// trees.c's build_tree: the lengths of `tree` (its first `elems`
    /// entries hold frequencies); the largest code with a frequency.
    fn build_tree(&mut self, tree: &mut [Node], elems: usize, stree: Option<&[u8]>, extra: &[u8], base: usize, max_length: usize) -> usize {
        self.heap_len = 0;
        self.heap_max = HEAP_SIZE;
        let mut max_code: i32 = -1;
        for n in 0..elems {
            if tree[n].freq != 0 {
                self.heap_len += 1;
                self.heap[self.heap_len] = n as u16;
                max_code = n as i32;
                self.depth[n] = 0;
            } else {
                tree[n].len = 0;
            }
        }
        while self.heap_len < 2 {
            let node = if max_code < 2 {
                max_code += 1;
                max_code as usize
            } else {
                0
            };
            self.heap_len += 1;
            self.heap[self.heap_len] = node as u16;
            tree[node].freq = 1;
            self.depth[node] = 0;
            self.opt_len = self.opt_len.wrapping_sub(1);
            if let Some(stree) = stree {
                self.static_len = self.static_len.wrapping_sub(stree[node] as u64);
            }
        }
        let max_code = max_code as usize;
        let mut n = self.heap_len / 2;
        while n >= 1 {
            self.pqdownheap(tree, n);
            n -= 1;
        }
        let mut node = elems;
        loop {
            let n = self.heap[1] as usize;
            self.heap[1] = self.heap[self.heap_len];
            self.heap_len -= 1;
            self.pqdownheap(tree, 1);
            let m = self.heap[1] as usize;
            self.heap_max -= 1;
            self.heap[self.heap_max] = n as u16;
            self.heap_max -= 1;
            self.heap[self.heap_max] = m as u16;
            tree[node].freq = tree[n].freq + tree[m].freq;
            self.depth[node] = self.depth[n].max(self.depth[m]).wrapping_add(1);
            tree[n].dad = node as u16;
            tree[m].dad = node as u16;
            self.heap[1] = node as u16;
            node += 1;
            self.pqdownheap(tree, 1);
            if self.heap_len < 2 {
                break;
            }
        }
        self.heap_max -= 1;
        self.heap[self.heap_max] = self.heap[1];
        self.gen_bitlen(tree, max_code, stree, extra, base, max_length);
        max_code
    }
}

fn fixed_lit_lens() -> Vec<u8> {
    super::fixed_lit_lens()
}

/// scan_tree and send_tree in one: the run-length symbols of the
/// lengths `lens[..=max_code]`, and their counts into `bl_freq` when
/// asked.
fn runs(lens: &[u8], max_code: usize, bl_freq: Option<&mut [u32; BL_CODES]>, out: Option<&mut Vec<(u8, u8)>>) {
    let mut bl_freq = bl_freq;
    let mut out = out;
    let mut prevlen: i32 = -1;
    let mut nextlen = lens[0] as i32;
    let mut count = 0;
    let (mut max_count, mut min_count) = if nextlen == 0 { (138, 3) } else { (7, 4) };
    for n in 0..=max_code {
        let curlen = nextlen;
        nextlen = if n + 1 <= max_code { lens[n + 1] as i32 } else { 0xffff };
        count += 1;
        if count < max_count && curlen == nextlen {
            continue;
        } else if count < min_count {
            if let Some(f) = bl_freq.as_deref_mut() {
                f[curlen as usize] += count as u32;
            }
            if let Some(o) = out.as_deref_mut() {
                for _ in 0..count {
                    o.push((curlen as u8, 0));
                }
            }
        } else if curlen != 0 {
            if curlen != prevlen {
                if let Some(f) = bl_freq.as_deref_mut() {
                    f[curlen as usize] += 1;
                }
                if let Some(o) = out.as_deref_mut() {
                    o.push((curlen as u8, 0));
                }
                count -= 1;
            }
            if let Some(f) = bl_freq.as_deref_mut() {
                f[REP_3_6] += 1;
            }
            if let Some(o) = out.as_deref_mut() {
                o.push((REP_3_6 as u8, (count - 3) as u8));
            }
        } else if count <= 10 {
            if let Some(f) = bl_freq.as_deref_mut() {
                f[REPZ_3_10] += 1;
            }
            if let Some(o) = out.as_deref_mut() {
                o.push((REPZ_3_10 as u8, (count - 3) as u8));
            }
        } else {
            if let Some(f) = bl_freq.as_deref_mut() {
                f[REPZ_11_138] += 1;
            }
            if let Some(o) = out.as_deref_mut() {
                o.push((REPZ_11_138 as u8, (count - 11) as u8));
            }
        }
        count = 0;
        prevlen = curlen;
        if nextlen == 0 {
            max_count = 138;
            min_count = 3;
        } else if curlen == nextlen {
            max_count = 6;
            min_count = 3;
        } else {
            max_count = 7;
            min_count = 4;
        }
    }
}

/// What zlib would write for a block of `tokens`: the dynamic header,
/// and the lengths the three encodings would take, from which
/// `kind_of` picks the block type.
pub struct Trees {
    pub header: Header,
    pub opt_len: u64,
    pub static_len: u64,
}

pub fn build(tokens: &[Token]) -> Trees {
    let mut lit = [Node::default(); HEAP_SIZE];
    let mut dist = [Node::default(); HEAP_SIZE];
    lit[END_BLOCK].freq = 1;
    for t in tokens {
        match *t {
            Token::Lit(b) => lit[b as usize].freq += 1,
            Token::Ref { len, dist: d } => {
                lit[len_code(len).0 as usize].freq += 1;
                dist[dist_code(d).0 as usize].freq += 1;
            }
        }
    }
    let mut b = Builder::new();
    let fixed = fixed_lit_lens();
    let l_max = b.build_tree(&mut lit, L_CODES, Some(&fixed), &LEN_EXTRA, 257, 15);
    let d_max = b.build_tree(&mut dist, D_CODES, Some(&[5u8; 30]), &DIST_EXTRA_BITS, 0, 15);
    let lit_lens: Vec<u8> = lit[..L_CODES].iter().map(|n| n.len as u8).collect();
    let dist_lens: Vec<u8> = dist[..D_CODES].iter().map(|n| n.len as u8).collect();
    let mut bl_freq = [0u32; BL_CODES];
    runs(&lit_lens, l_max, Some(&mut bl_freq), None);
    runs(&dist_lens, d_max, Some(&mut bl_freq), None);
    let mut bl = [Node::default(); HEAP_SIZE];
    for i in 0..BL_CODES {
        bl[i].freq = bl_freq[i];
    }
    b.build_tree(&mut bl, BL_CODES, None, &BL_EXTRA_BITS, 0, 7);
    let mut clen = [0u8; BL_CODES];
    for i in 0..BL_CODES {
        clen[i] = bl[i].len as u8;
    }
    let mut max_blindex = BL_CODES - 1;
    while max_blindex >= 3 && clen[CLEN_ORDER[max_blindex]] == 0 {
        max_blindex -= 1;
    }
    b.opt_len += 3 * (max_blindex as u64 + 1) + 5 + 5 + 4;
    let mut symbols = Vec::new();
    runs(&lit_lens, l_max, None, Some(&mut symbols));
    runs(&dist_lens, d_max, None, Some(&mut symbols));
    let header = Header {
        hlit: l_max as u16 + 1,
        hdist: d_max as u8 + 1,
        hclen: max_blindex as u8 + 1,
        clen,
        symbols,
        lit_lens: lit_lens[..=l_max].to_vec(),
        dist_lens: dist_lens[..=d_max].to_vec(),
    };
    Trees { header, opt_len: b.opt_len, static_len: b.static_len }
}

/// `_tr_flush_block`'s choice: 0 stored, 1 fixed, 2 dynamic, for a
/// block of `stored_len` bytes.
pub fn kind_of(t: &Trees, stored_len: usize, z_fixed: bool) -> u8 {
    let mut opt_lenb = (t.opt_len + 3 + 7) >> 3;
    let static_lenb = (t.static_len + 3 + 7) >> 3;
    if static_lenb <= opt_lenb || z_fixed {
        opt_lenb = static_lenb;
    }
    if stored_len as u64 + 4 <= opt_lenb && stored_len <= 65535 {
        0
    } else if static_lenb == opt_lenb {
        1
    } else {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::super::{parse, tests::{text, zlib_deflate}, Kind};
    use super::*;

    /// Every dynamic header of zlib's streams, at every level, comes
    /// out of the tokens as zlib wrote it, and so does the block type.
    #[test]
    fn headers_and_kinds_predicted_from_the_tokens() {
        let plain = text(1 << 20);
        for level in 1..=9 {
            let stream = zlib_deflate(&plain, level, "Z_DEFAULT_STRATEGY");
            let s = parse(&stream).unwrap();
            for (i, b) in s.blocks.iter().enumerate() {
                if let Kind::Dynamic(h, tokens) = &b.kind {
                    let t = build(tokens);
                    assert!(t.header == *h, "level {level} block {i}: header\n  got {:?} {:?} {:?}\n  want {:?} {:?} {:?}", t.header.hlit, t.header.hdist, t.header.hclen, h.hlit, h.hdist, h.hclen);
                    let bytes: usize = tokens.iter().map(|t| match t { Token::Lit(_) => 1, Token::Ref { len, .. } => *len as usize }).sum();
                    assert_eq!(kind_of(&t, bytes, false), 2, "level {level} block {i}: kind");
                }
            }
        }
        let stream = zlib_deflate(&plain, 6, "Z_FIXED");
        for b in &parse(&stream).unwrap().blocks {
            if let Kind::Fixed(tokens) = &b.kind {
                let bytes: usize = tokens.iter().map(|t| match t { Token::Lit(_) => 1, Token::Ref { len, .. } => *len as usize }).sum();
                assert_eq!(kind_of(&build(tokens), bytes, true), 1);
            }
        }
    }
}
