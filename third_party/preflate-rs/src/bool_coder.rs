/*---------------------------------------------------------------------------------------------
 *  Copyright (c) 2026, Surya Koritala.
 *
 *  Glyd's own implementation of the boolean entropy coder that RFC 6386 (section 7)
 *  specifies, plus the adaptive probability context described in Glyd's functional
 *  specification of the corrections stream. It replaces the `cabac` crate
 *  (LGPL-3.0-or-later), which this crate no longer uses: it was written from that
 *  specification and the RFC alone, and writes and reads, bit for bit, the streams
 *  that crate wrote, so everything earlier releases wrote still opens.
 *
 *  Licensed under the BSD 3-Clause License or the GNU General Public License
 *  version 2, at your option, like the rest of Glyd's codec (LICENSE and COPYING at
 *  the repository root).
 *--------------------------------------------------------------------------------------------*/

//! The coder for preflate's corrections: every decision is a bit coded with
//! the 8-bit range coder of RFC 6386 section 7, at a probability a context
//! learns from the bits it has coded.
//!
//! What is contractual is the byte string. Everything below that touches it
//! is written as the specification has it:
//!
//! - a context is a pair of counts, (1, 1) at the start; the probability of
//!   a zero is `256 * n0 / (n0 + n1)`, read before the bit is coded, and the
//!   counts are updated after (`VP8Context::update`);
//! - the stream starts with a marker, a zero at probability 128 that the
//!   reader decodes and drops, so the first byte is always below 0x80 and a
//!   carry never runs off the front;
//! - the end is not the RFC's four-byte flush: zeros are coded (at the
//!   "bypass" split, `1 + range / 2`) until nothing is left in `bottom`, so
//!   the bytes that carry no information are not written, and the reader
//!   reads a missing byte as 0x00, as far as it is asked to go.
//!
//! The reader is bounded: a unary code longer than `MAX_RUN`, or a literal
//! wider than that, is an error (`InvalidData`, `InvalidInput`) where the
//! crate this replaces looped for ever on some malformed input (the single
//! byte 0xff and a unary read). preflate's own codes are at most 33
//! decisions long; no stream it wrote is affected.

use std::io::{self, Read, Write};

#[cfg(test)]
mod tests;
#[cfg(test)]
mod vs_cabac;

/// The longest unary code, in decisions, and the widest literal, in bits,
/// that are read or written. preflate codes the bit length of a `u32`:
/// at most 32 ones and a zero.
const MAX_RUN: usize = 4096;

/// The decision's split of the range at probability `prob` of a zero
/// (RFC 6386 section 7.2): `1 <= split <= range - 1` for `128 <= range <= 255`.
#[inline(always)]
fn split_of(range: u32, prob: u32) -> u32 {
    1 + (((range - 1) * prob) >> 8)
}

/// The split of a decision that has no context: half the range, rounded up
/// and one more. It is not `split_of(range, 128)` where the range is even.
#[inline(always)]
fn bypass_split(range: u32) -> u32 {
    1 + (range >> 1)
}

/// The probability a bit is coded with: how many zeros and ones the context
/// has seen, halved when one count reaches its ceiling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VP8Context {
    n0: u8,
    n1: u8,
}

impl Default for VP8Context {
    #[inline]
    fn default() -> Self {
        Self { n0: 1, n1: 1 }
    }
}

impl VP8Context {
    /// 256 times the probability of a zero, `1..=255` (both counts are
    /// at least one and at most 255).
    #[inline(always)]
    fn prob(self) -> u32 {
        let (n0, n1) = (u32::from(self.n0), u32::from(self.n1));
        (n0 << 8) / (n0 + n1)
    }

    /// Counts the coded bit. Its count goes up to 255 and stays there while
    /// the other is 1; at 255 with the other count above 1, the other count
    /// is halved (rounding up) and this one is set to 129.
    #[inline(always)]
    fn update(&mut self, bit: bool) {
        let (x, y) = if bit {
            (&mut self.n1, &mut self.n0)
        } else {
            (&mut self.n0, &mut self.n1)
        };
        if *x < 255 {
            *x += 1;
        } else if *y > 1 {
            *y = y.div_ceil(2);
            *x = 129;
        }
    }
}

/// The encoding side of the coder: decisions in, a stream out.
pub trait CabacWriter<Context> {
    /// Codes `bit` at the probability of `ctx`, then counts it in `ctx`.
    fn put(&mut self, bit: bool, ctx: &mut Context) -> io::Result<()>;

    /// Ends the stream and hands the bytes to the sink. Call it once, after
    /// the last decision.
    fn finish(&mut self) -> io::Result<()>;

    /// `v` ones and a zero, the decision `i` of them under
    /// `contexts[min(i, A - 1)]`.
    fn put_unary_encoded<const A: usize>(
        &mut self,
        v: usize,
        contexts: &mut [Context; A],
    ) -> io::Result<()> {
        const { assert!(A > 0) };
        if v >= MAX_RUN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "unary code too long",
            ));
        }
        for i in 0..v {
            self.put(true, &mut contexts[i.min(A - 1)])?;
        }
        self.put(false, &mut contexts[v.min(A - 1)])
    }

    /// The low `num_bits` bits of `bits`, the highest first, bit `i` under
    /// `contexts[min(i, A - 1)]`. Bits above the 64th are zeros.
    fn put_n_bits<const A: usize>(
        &mut self,
        bits: u64,
        num_bits: usize,
        contexts: &mut [Context; A],
    ) -> io::Result<()> {
        const { assert!(A > 0) };
        if num_bits > MAX_RUN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "literal too wide",
            ));
        }
        for i in (0..num_bits).rev() {
            let bit = i < 64 && (bits >> i) & 1 != 0;
            self.put(bit, &mut contexts[i.min(A - 1)])?;
        }
        Ok(())
    }
}

/// The decoding side: a stream in, decisions out.
pub trait CabacReader<Context> {
    /// The next decision at the probability of `ctx`, which then counts it.
    fn get(&mut self, ctx: &mut Context) -> io::Result<bool>;

    /// A unary code: the number of ones before the first zero, the decision
    /// `i` of them under `contexts[min(i, A - 1)]`. A code of `MAX_RUN` ones
    /// or more is an error, so that no input makes this loop for ever.
    fn get_unary_encoded<const A: usize>(
        &mut self,
        contexts: &mut [Context; A],
    ) -> io::Result<usize> {
        const { assert!(A > 0) };
        let mut count = 0;
        while self.get(&mut contexts[count.min(A - 1)])? {
            count += 1;
            if count >= MAX_RUN {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unary code too long",
                ));
            }
        }
        Ok(count)
    }

    /// `num_bits` bits, the highest first, bit `i` under
    /// `contexts[min(i, A - 1)]`. Bits above the 64th are read and dropped.
    fn get_n_bits<const A: usize>(
        &mut self,
        num_bits: usize,
        contexts: &mut [Context; A],
    ) -> io::Result<u64> {
        const { assert!(A > 0) };
        if num_bits > MAX_RUN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "literal too wide",
            ));
        }
        let mut v = 0u64;
        for i in (0..num_bits).rev() {
            let bit = self.get(&mut contexts[i.min(A - 1)])?;
            if i < 64 {
                v |= u64::from(bit) << i;
            }
        }
        Ok(v)
    }
}

/// Adds one to the number the bytes are, the first byte the highest: the
/// trailing 0xff bytes become 0x00 and the byte before them goes up by one.
/// A carry out of the first byte cannot happen (the marker keeps the stream
/// below 0.5), and is dropped.
fn carry(out: &mut [u8]) {
    for b in out.iter_mut().rev() {
        if *b == 0xff {
            *b = 0;
        } else {
            *b += 1;
            return;
        }
    }
}

/// Encoder. The bytes are kept until `finish`: a carry can reach back
/// through every 0xff byte since the last other byte, and a sink cannot be
/// taken back.
pub struct VP8Writer<W: Write> {
    sink: W,
    out: Vec<u8>,
    /// The left end of the interval: the low 8 bits are RFC 6386's `bottom`;
    /// above them are the `24 - bit_count` bits shifted out of it and not yet
    /// in `out`, then the one bit a carry out of those can set.
    bottom: u32,
    /// `128..=255` between decisions.
    range: u32,
    /// Shifts until the next byte comes out of `bottom`: 24 at the start,
    /// then 8, 7, ..., 1.
    bit_count: u32,
}

impl<W: Write> VP8Writer<W> {
    pub fn new(sink: W) -> io::Result<Self> {
        let mut w = Self {
            sink,
            out: Vec::new(),
            bottom: 0,
            range: 255,
            bit_count: 24,
        };
        // the marker: a zero at probability 128 (range 255 -> 128)
        w.code(false, split_of(255, 128));
        Ok(w)
    }

    /// A decision at `split` of the range; the interval narrows, and
    /// shifts until the range is at least 128.
    #[inline(always)]
    fn code(&mut self, bit: bool, split: u32) {
        if bit {
            self.bottom += split;
            self.range -= split;
        } else {
            self.range = split;
        }
        let shift = self.range.leading_zeros() - 24;
        if shift != 0 {
            self.normalize(shift);
        }
    }

    /// `shift` (1 to 7) shifts of `range` and `bottom` in one step. A byte
    /// comes out at the shift that takes `bit_count` to 0: first the carry,
    /// the bit that is in the one-bit slot, and that shift pushes out of the
    /// register, goes to the bytes already out; then the top byte of the
    /// register; then what is left of `shift` is done on the register with
    /// that byte taken off.
    #[inline]
    fn normalize(&mut self, shift: u32) {
        self.range <<= shift;
        let c = self.bit_count;
        if shift < c {
            self.bottom <<= shift;
            self.bit_count = c - shift;
        } else {
            if (self.bottom >> (32 - c)) & 1 != 0 {
                carry(&mut self.out);
            }
            let shifted = self.bottom << c;
            self.out.push((shifted >> 24) as u8);
            self.bottom = (shifted & 0x00ff_ffff) << (shift - c);
            self.bit_count = 8 - (shift - c);
        }
    }

    /// A decision with no context: it is the one the end of the stream is
    /// made of.
    pub(crate) fn put_bypass(&mut self, bit: bool) {
        self.code(bit, bypass_split(self.range));
    }
}

impl<W: Write> CabacWriter<VP8Context> for VP8Writer<W> {
    #[inline]
    fn put(&mut self, bit: bool, ctx: &mut VP8Context) -> io::Result<()> {
        self.code(bit, split_of(self.range, ctx.prob()));
        ctx.update(bit);
        Ok(())
    }

    /// Zeros are coded until `bottom` is empty, so the last byte written is
    /// the one that holds the last set bit; calling it again writes nothing.
    fn finish(&mut self) -> io::Result<()> {
        while self.bottom != 0 {
            self.put_bypass(false);
        }
        let out = std::mem::take(&mut self.out);
        self.sink.write_all(&out)
    }
}

/// Decoder. A byte the source does not have is 0x00, for as long as it is
/// asked: the writer does not write the zeros at the end.
pub struct VP8Reader<R: Read> {
    src: R,
    /// The top 8 bits are the window the decisions are compared in (RFC
    /// 6386's `value`, high byte), below them the `avail - 8` bits that
    /// follow in the stream, below those zeros. Bits shifted out of the top
    /// are dropped.
    value: u64,
    /// `128..=255` between decisions.
    range: u32,
    /// How many of the top bits of `value` are stream bits (or the zeros
    /// past its end): at least 16 before a decision, so that 7 shifts leave
    /// the window whole.
    avail: u32,
    /// The source has said it has no more.
    eof: bool,
}

impl<R: Read> VP8Reader<R> {
    pub fn new(src: R) -> io::Result<Self> {
        let mut r = Self {
            src,
            value: 0,
            range: 255,
            avail: 0,
            eof: false,
        };
        r.refill()?;
        // the marker, dropped
        r.decode(split_of(255, 128));
        Ok(r)
    }

    /// Brings in as many whole bytes as fit below the bits already in
    /// `value`; a source that has ended gives zeros.
    #[inline(never)]
    fn refill(&mut self) -> io::Result<()> {
        let want = ((64 - self.avail) / 8) as usize;
        let mut buf = [0u8; 8];
        if !self.eof {
            let mut got = 0;
            while got < want {
                match self.src.read(&mut buf[got..want]) {
                    Ok(0) => {
                        self.eof = true;
                        break;
                    }
                    Ok(n) => got += n,
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                    Err(e) => return Err(e),
                }
            }
        }
        // `buf` is the bytes, first one on top, then zeros
        self.value |= u64::from_be_bytes(buf) >> self.avail;
        self.avail += 8 * want as u32;
        Ok(())
    }

    /// A decision at `split` of the range (RFC 6386 7.3, `read_bool`): a
    /// one if the window is at least the split (equal counts).
    #[inline(always)]
    fn decode(&mut self, split: u32) -> bool {
        let s = u64::from(split) << 56;
        let bit = self.value >= s;
        if bit {
            self.value -= s;
            self.range -= split;
        } else {
            self.range = split;
        }
        let shift = self.range.leading_zeros() - 24;
        self.value <<= shift;
        self.range <<= shift;
        self.avail -= shift;
        bit
    }

    /// The decision with no context (see `VP8Writer::put_bypass`).
    #[cfg(test)]
    pub(crate) fn get_bypass(&mut self) -> io::Result<bool> {
        if self.avail < 16 {
            self.refill()?;
        }
        Ok(self.decode(bypass_split(self.range)))
    }
}

impl<R: Read> CabacReader<VP8Context> for VP8Reader<R> {
    #[inline]
    fn get(&mut self, ctx: &mut VP8Context) -> io::Result<bool> {
        if self.avail < 16 {
            self.refill()?;
        }
        let bit = self.decode(split_of(self.range, ctx.prob()));
        ctx.update(bit);
        Ok(bit)
    }
}

/// A coder that writes what it is asked to code as it is, with a number for
/// every context, so that a test finds a decision that is read under another
/// context than it was written under. Its output is not a stream of the
/// coder above and is never written to a file.
#[cfg(test)]
pub mod debug {
    use std::io::{self, Read, Write};

    use super::{CabacReader, CabacWriter};

    #[derive(Clone, Copy, Debug, Default)]
    pub struct DebugContext {
        value: u32,
    }

    /// A decision not under a context.
    const BYPASS_MARK: u32 = 0xdead;

    pub struct DebugWriter<W: Write> {
        sink: W,
        counter: u32,
    }

    impl<W: Write> DebugWriter<W> {
        pub fn new(sink: W) -> io::Result<Self> {
            Ok(Self { sink, counter: 100 })
        }
    }

    impl<W: Write> CabacWriter<DebugContext> for DebugWriter<W> {
        fn put(&mut self, bit: bool, ctx: &mut DebugContext) -> io::Result<()> {
            if ctx.value == 0 {
                self.counter += 1;
                ctx.value = self.counter;
            }
            self.sink.write_all(&ctx.value.to_le_bytes())?;
            self.counter += 1;
            ctx.value = self.counter;
            self.sink.write_all(&[u8::from(bit)])
        }

        fn finish(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<W: Write> DebugWriter<W> {
        #[allow(dead_code)]
        pub fn put_bypass(&mut self, bit: bool) -> io::Result<()> {
            self.sink.write_all(&BYPASS_MARK.to_le_bytes())?;
            self.sink.write_all(&[u8::from(bit)])
        }
    }

    pub struct DebugReader<R: Read> {
        src: R,
        counter: u32,
    }

    impl<R: Read> DebugReader<R> {
        pub fn new(src: R) -> io::Result<Self> {
            Ok(Self { src, counter: 100 })
        }
    }

    impl<R: Read> CabacReader<DebugContext> for DebugReader<R> {
        fn get(&mut self, ctx: &mut DebugContext) -> io::Result<bool> {
            if ctx.value == 0 {
                self.counter += 1;
                ctx.value = self.counter;
            }
            let mut word = [0u8; 4];
            self.src.read_exact(&mut word)?;
            assert_eq!(
                u32::from_le_bytes(word),
                ctx.value,
                "a decision is read under another context than it was written under"
            );
            self.counter += 1;
            ctx.value = self.counter;
            let mut bit = [0u8; 1];
            self.src.read_exact(&mut bit)?;
            Ok(bit[0] != 0)
        }
    }
}
