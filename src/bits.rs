//! LSB-first bitstreams for the v7 entropy coders.
//!
//! The reader keeps a 64-bit accumulator and refills it with one unaligned
//! 8-byte load, branch-free (Giesen's scheme): `cnt` is the number of valid
//! low bits; after `refill` it is at least 56. The load pointer is clamped
//! to `end - 8`, so a corrupt stream can make the reader return zeros but
//! never read outside its slice; `overrun` reports that.

pub const PAD: usize = 8;

pub struct BitWriter {
    acc: u64,
    n: u32,
    out: Vec<u8>,
}

impl BitWriter {
    pub fn new() -> Self {
        BitWriter { acc: 0, n: 0, out: Vec::new() }
    }

    /// Append the low `nbits` (1..=32) of `value`, least significant first.
    #[inline(always)]
    pub fn put(&mut self, value: u64, nbits: u32) {
        debug_assert!(nbits >= 1 && nbits <= 32);
        self.acc |= (value & ((1u64 << nbits) - 1)) << self.n;
        self.n += nbits;
        while self.n >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.n -= 8;
        }
    }

    pub fn bits_written(&self) -> usize {
        self.out.len() * 8 + self.n as usize
    }

    /// Flush the partial byte and append `PAD` zero bytes.
    pub fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.out.push(self.acc as u8);
        }
        self.out.extend_from_slice(&[0u8; PAD]);
        self.out
    }
}

pub struct BitReader {
    p: *const u8,
    /// Last address a full 8-byte load may start at.
    last: *const u8,
    bits: u64,
    cnt: u32,
    consumed: u64,
    total_bits: u64,
}

impl BitReader {
    /// `src` must end with `PAD` bytes (as `BitWriter::finish` produces).
    pub fn new(src: &[u8]) -> Self {
        assert!(src.len() >= PAD, "stream shorter than its padding");
        let p = src.as_ptr();
        let mut r = BitReader {
            p,
            last: unsafe { p.add(src.len() - PAD) },
            bits: 0,
            cnt: 0,
            consumed: 0,
            total_bits: ((src.len() - PAD) * 8) as u64,
        };
        r.refill();
        r
    }

    #[inline(always)]
    pub fn refill(&mut self) {
        unsafe {
            let p = if self.p > self.last { self.last } else { self.p };
            self.bits |= std::ptr::read_unaligned(p as *const u64) << self.cnt;
            self.p = self.p.add(((63 - self.cnt) >> 3) as usize);
            self.cnt |= 56;
        }
    }

    #[inline(always)]
    pub fn peek(&self, n: u32) -> u64 {
        self.bits & ((1u64 << n) - 1)
    }

    #[inline(always)]
    pub fn consume(&mut self, n: u32) {
        self.bits >>= n;
        self.cnt -= n;
        self.consumed += n as u64;
    }

    /// Read `n` (0..=32) bits; refills first.
    #[inline(always)]
    pub fn get(&mut self, n: u32) -> u64 {
        self.refill();
        let v = self.peek(n);
        self.consume(n);
        v
    }

    /// More bits consumed than the stream holds: the data was corrupt.
    pub fn overrun(&self) -> bool {
        self.consumed > self.total_bits
    }
}
