//! A binary arithmetic coder with adaptive probabilities, for the
//! corrections: every decision is a bit under a context that learns.

/// A probability of a 1 bit, 12 bits, adapting at rate 1/32.
#[derive(Clone, Copy)]
pub struct Bit(u16);

impl Default for Bit {
    fn default() -> Self {
        Bit(2048)
    }
}

impl Bit {
    #[inline]
    fn update(&mut self, bit: u32) {
        if bit != 0 {
            self.0 += (4096 - self.0) >> 5;
        } else {
            self.0 -= self.0 >> 5;
        }
    }
}

/// Carry-free range coder over [x1, x2] (the one in `cm.rs`).
pub struct Encoder {
    x1: u32,
    x2: u32,
    pub out: Vec<u8>,
}

impl Encoder {
    pub fn new() -> Self {
        Encoder { x1: 0, x2: 0xffff_ffff, out: Vec::new() }
    }

    #[inline]
    pub fn bit(&mut self, m: &mut Bit, bit: u32) {
        let p = (m.0 as u32).clamp(1, 4095);
        let xmid = self.x1 + ((self.x2 - self.x1) >> 12) * p;
        if bit != 0 {
            self.x2 = xmid;
        } else {
            self.x1 = xmid + 1;
        }
        m.update(bit);
        while (self.x1 ^ self.x2) & 0xff00_0000 == 0 {
            self.out.push((self.x2 >> 24) as u8);
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 255;
        }
    }

    /// `v` in `bits` bits, most significant first, each under its own
    /// context of the tree `m` (which holds `1 << bits` entries).
    pub fn tree(&mut self, m: &mut [Bit], bits: u32, v: u32) {
        let mut node = 1usize;
        for i in (0..bits).rev() {
            let b = (v >> i) & 1;
            self.bit(&mut m[node], b);
            node = node * 2 + b as usize;
        }
    }

    /// A count with no fixed size: its bit length in unary under `m[0..]`,
    /// then the bits below the top one under `m[32..]`.
    pub fn count(&mut self, m: &mut [Bit; 64], v: u32) {
        let n = 32 - v.leading_zeros();
        for i in 0..n {
            self.bit(&mut m[i as usize], 1);
        }
        self.bit(&mut m[n as usize], 0);
        for i in (0..n.saturating_sub(1)).rev() {
            self.bit(&mut m[32 + i as usize], (v >> i) & 1);
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        self.out.extend_from_slice(&self.x1.to_be_bytes());
        self.out
    }
}

pub struct Decoder<'a> {
    x1: u32,
    x2: u32,
    x: u32,
    data: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let mut d = Decoder { x1: 0, x2: 0xffff_ffff, x: 0, data, pos: 0 };
        for _ in 0..4 {
            d.x = (d.x << 8) | d.byte();
        }
        d
    }

    #[inline]
    fn byte(&mut self) -> u32 {
        let b = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b as u32
    }

    #[inline]
    pub fn bit(&mut self, m: &mut Bit) -> u32 {
        let p = (m.0 as u32).clamp(1, 4095);
        let xmid = self.x1 + ((self.x2 - self.x1) >> 12) * p;
        let bit = if self.x <= xmid { 1 } else { 0 };
        if bit != 0 {
            self.x2 = xmid;
        } else {
            self.x1 = xmid + 1;
        }
        m.update(bit);
        while (self.x1 ^ self.x2) & 0xff00_0000 == 0 {
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 255;
            self.x = (self.x << 8) | self.byte();
        }
        bit
    }

    pub fn tree(&mut self, m: &mut [Bit], bits: u32) -> u32 {
        let mut node = 1usize;
        for _ in 0..bits {
            let b = self.bit(&mut m[node]);
            node = node * 2 + b as usize;
        }
        node as u32 - (1 << bits)
    }

    pub fn count(&mut self, m: &mut [Bit; 64]) -> u32 {
        let mut n = 0u32;
        while n < 32 && self.bit(&mut m[n as usize]) == 1 {
            n += 1;
        }
        if n == 0 {
            return 0;
        }
        let mut v = 1u32 << (n - 1);
        for i in (0..n - 1).rev() {
            v |= self.bit(&mut m[32 + i as usize]) << i;
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_trees_and_counts_round_trip() {
        let mut e = Encoder::new();
        let mut m = [Bit::default(); 512];
        let mut c = [Bit::default(); 64];
        let values: Vec<u32> = (0..5000u32).map(|i| (i * 2654435761u32.wrapping_mul(i + 1)) % 300).collect();
        for &v in &values {
            e.bit(&mut m[0], v & 1);
            e.tree(&mut m[256..], 8, v % 256);
            e.count(&mut c, v);
        }
        let out = e.finish();
        let mut d = Decoder::new(&out);
        let mut m = [Bit::default(); 512];
        let mut c = [Bit::default(); 64];
        for &v in &values {
            assert_eq!(d.bit(&mut m[0]), v & 1);
            assert_eq!(d.tree(&mut m[256..], 8), v % 256);
            assert_eq!(d.count(&mut c), v);
        }
    }
}
