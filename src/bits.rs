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
    /// `cnt` immediately after the last `refill`, i.e. the start of the
    /// current fill window. `filled - cnt` is the bits consumed within
    /// that window, folded into `budget` the next time `refill` runs
    /// instead of on every `consume` call.
    filled: u32,
    /// Bits left in the stream, counting down from `total_bits` once per
    /// `refill` (not once per `consume`). Signed so it can go negative:
    /// that's the overrun signal, and folding `total_bits` into its
    /// initial value keeps this reader at the same field count as an
    /// explicit `consumed` + `total_bits` pair.
    budget: i64,
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
            filled: 0,
            budget: ((src.len() - PAD) * 8) as i64,
        };
        r.refill();
        r
    }

    #[inline(always)]
    pub fn refill(&mut self) {
        // Fold in whatever was consumed since the last refill, once per
        // refill instead of once per consume (this loop calls refill 4x
        // less often than consume, since it decodes 4 symbols per stream
        // between refills).
        self.budget -= (self.filled - self.cnt) as i64;
        unsafe {
            let p = if self.p > self.last { self.last } else { self.p };
            self.bits |= std::ptr::read_unaligned(p as *const u64) << self.cnt;
            self.p = p.add(((63 - self.cnt) >> 3) as usize);
            self.cnt |= 56;
        }
        self.filled = self.cnt;
    }

    #[inline(always)]
    pub fn peek(&self, n: u32) -> u64 {
        self.bits & ((1u64 << n) - 1)
    }

    #[inline(always)]
    pub fn consume(&mut self, n: u32) {
        self.bits >>= n;
        self.cnt -= n;
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
    /// `budget` covers every fully-closed fill window; subtract whatever
    /// has been consumed from the still-open one too.
    pub fn overrun(&self) -> bool {
        let open = (self.filled - self.cnt) as i64;
        self.budget - open < 0
    }
}
