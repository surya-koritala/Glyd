//! The binary arithmetic coder of the JPEG model: a range coder with
//! a 32-bit range and 16-bit probabilities (LZMA's construction, the
//! carry kept in a cache byte), a shorter chain of dependent
//! operations per decision than `reflate::coder`'s.

pub use crate::reflate::coder::Prob;

const TOP: u32 = 1 << 24;

pub struct Encoder {
    low: u64,
    range: u32,
    cache: u8,
    cache_size: u64,
    pub out: Vec<u8>,
}

impl Encoder {
    pub fn new() -> Self {
        Encoder { low: 0, range: 0xffff_ffff, cache: 0, cache_size: 1, out: Vec::new() }
    }

    #[inline(never)]
    fn shift_low(&mut self) {
        if (self.low as u32) < 0xff00_0000 || (self.low >> 32) != 0 {
            let carry = (self.low >> 32) as u8;
            let mut temp = self.cache;
            loop {
                self.out.push(temp.wrapping_add(carry));
                temp = 0xff;
                self.cache_size -= 1;
                if self.cache_size == 0 {
                    break;
                }
            }
            self.cache = (self.low >> 24) as u8;
        }
        self.cache_size += 1;
        self.low = (self.low & 0x00ff_ffff) << 8;
    }

    #[inline(always)]
    pub fn bit<P: Prob>(&mut self, m: &mut P, bit: u32) {
        let bound = (self.range >> 16) * m.p();
        // Branch-free, as the decoder: on real coefficients a
        // mispredicted branch here costs more than the decision.
        let mask = 0u32.wrapping_sub(bit & 1);
        self.low += (bound & !mask) as u64;
        self.range = (bound & mask) | ((self.range - bound) & !mask);
        m.update(bit);
        while self.range < TOP {
            self.range <<= 8;
            self.shift_low();
        }
    }

    /// A bit at even odds with nothing to adapt.
    #[inline(always)]
    pub fn raw(&mut self, bit: u32) {
        self.range >>= 1;
        self.low += (self.range & (bit & 1).wrapping_sub(1)) as u64;
        if self.range < TOP {
            self.range <<= 8;
            self.shift_low();
        }
    }

    /// `v` in `bits` bits, most significant first, each under its own
    /// context of the tree `m` (which holds `1 << bits` entries).
    #[inline]
    pub fn tree<P: Prob>(&mut self, m: &mut [P], bits: u32, v: u32) {
        let mut node = 1usize;
        for i in (0..bits).rev() {
            let b = (v >> i) & 1;
            self.bit(&mut m[node], b);
            node = node * 2 + b as usize;
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        for _ in 0..5 {
            self.shift_low();
        }
        self.out
    }
}

pub struct Decoder<'a> {
    range: u32,
    code: u32,
    data: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let mut d = Decoder { range: 0xffff_ffff, code: 0, data, pos: 0 };
        for _ in 0..5 {
            d.code = (d.code << 8) | d.byte();
        }
        d
    }

    #[inline]
    fn byte(&mut self) -> u32 {
        let b = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b as u32
    }

    #[inline(always)]
    pub fn bit<P: Prob>(&mut self, m: &mut P) -> u32 {
        let bound = (self.range >> 16) * m.p();
        let bit = (self.code < bound) as u32;
        // Branch-free: a mispredicted branch here would cost more than
        // the decision itself.
        let mask = 0u32.wrapping_sub(bit);
        self.range = (bound & mask) | ((self.range - bound) & !mask);
        self.code -= bound & !mask;
        m.update(bit);
        while self.range < TOP {
            self.range <<= 8;
            self.code = (self.code << 8) | self.byte();
        }
        bit
    }

    #[inline(always)]
    pub fn raw(&mut self) -> u32 {
        self.range >>= 1;
        let bit = (self.code < self.range) as u32;
        self.code -= self.range & (bit.wrapping_sub(1));
        if self.range < TOP {
            self.range <<= 8;
            self.code = (self.code << 8) | self.byte();
        }
        bit
    }

    #[inline]
    pub fn tree<P: Prob>(&mut self, m: &mut [P], bits: u32) -> u32 {
        let mut node = 1usize;
        for _ in 0..bits {
            let b = self.bit(&mut m[node]);
            node = node * 2 + b as usize;
        }
        node as u32 - (1 << bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpg::model::Bit;

    #[test]
    fn bits_raw_and_trees_round_trip() {
        let mut e = Encoder::new();
        let mut m = vec![Bit::default(); 64];
        let values: Vec<u32> = (0..20000u32).map(|i| (i.wrapping_mul(2654435761) >> 7) % 300).collect();
        for &v in &values {
            e.bit(&mut m[0], v & 1);
            e.raw(v >> 1 & 1);
            e.tree(&mut m[..32], 5, v % 32);
        }
        let out = e.finish();
        let mut d = Decoder::new(&out);
        let mut m = vec![Bit::default(); 64];
        for &v in &values {
            assert_eq!(d.bit(&mut m[0]), v & 1);
            assert_eq!(d.raw(), v >> 1 & 1);
            assert_eq!(d.tree(&mut m[..32], 5), v % 32);
        }
    }
}
