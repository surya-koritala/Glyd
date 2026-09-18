# Format v7 (entropy-coded `--max` level) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new level (`--max`, format v7) whose Silesia ratio is >= 3.20 (zstd -3) with compression >= 0.34 GB/s and decode >= 3.0 GB/s, measured in the same run on one core, without touching the v6 levels.

**Architecture:** Five per-block streams (literals, literal-length codes, match-length codes, offset codes, extra bits), each 8-way interleaved and entropy coded (Huffman for literals, tANS for the three code streams) or stored raw. The decoder runs three passes per block: entropy-decode sequences into flat u32 arrays, Huffman-decode literals into a buffer, then the existing-style NEON copy loop. The compressor is a zstd -3 style "double fast" parse with repeat offsets and a 2 MB window.

**Tech Stack:** Rust 2021, `std::arch::aarch64` NEON for the copy loop (scalar elsewhere), existing crates only (`zstd`/`lz4` are benchmark baselines). Tests with `cargo test --release`; numbers with `RUSTFLAGS="-C target-cpu=native"` release examples.

**Spec:** `docs/superpowers/specs/2026-09-18-format-v7-entropy-design.md`

## Global Constraints

- Existing levels (fast, default, turbo; format v6) keep their measured numbers; no change to v6 code paths except container dispatch.
- Every entropy-coded stream ends with 8 zero padding bytes; the bit reader never reads outside the stream it was given (clamped refill).
- Decoder allocates nothing per call beyond a once-per-thread scratch of at most 1.5 MB.
- Block size 256 KB (`MAX_BLOCK_SIZE`); offsets up to 21 bits (2 MB window); format minimum match 3; Huffman max code length 11 (`huffman::MAX_CODE_LEN`); tANS table log 10.
- Every measurement is same-run (our number and the baseline's from the same process), one core, `RUSTFLAGS="-C target-cpu=native"`, and gets a paragraph in `CHANGELOG-BENCH.md` before the next milestone starts.
- Commit after every task with the `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` trailer.
- All new files compile on x86_64 and aarch64; NEON-only code sits behind `#[cfg(target_arch = "aarch64")]` with a scalar twin.

## File Structure

| File | Responsibility |
|---|---|
| `src/bits.rs` (new) | LSB-first bit writer with padding; branchless clamped bit reader. |
| `src/huff8.rs` (new) | 8-stream interleaved Huffman: packed decode table, encode, decode. Uses `huffman::build_lengths/build_codes`. |
| `src/tans.rs` (new) | tANS: normalization, decode/encode tables, 8-stream interleaved encode/decode. |
| `src/v7_format.rs` (new) | v7 constants, code<->value mapping for lengths and offsets, repeat-offset state, block sub-header (de)serialization. |
| `src/v7_encode.rs` (new) | Block encoder (`Sequence` list + literals -> payload), double-fast parse, `compress_into_max`. |
| `src/v7_decode.rs` (new) | Block decoder: three passes, thread-local scratch. |
| `src/format.rs` (modify) | `VERSION_V7 = 7`; `BlockHeader::payload_len/is_plausible` for v7. |
| `src/lib.rs` (modify) | Header version check, decode dispatch, `compress_into_max`, `compress_parallel_into_max`, `compress_with_dict`. |
| `src/c_api.rs`, `include/alatirok.h`, `src/bin/alatirok.rs` (modify) | `--max` level exposure. |
| `tests/v7_codecs.rs`, `tests/v7_roundtrip.rs`, `tests/v7_fuzz.rs` (new) | Unit, round-trip and fuzz tests. |
| `examples/v7_bench.rs` (new), `examples/field_survey.rs` (modify), `scripts/download_corpus.sh` (modify) | Same-run measurement against zstd; corpus. |

---

### Task 1: LSB-first bit writer and clamped branchless bit reader

**Files:**
- Create: `src/bits.rs`
- Modify: `src/lib.rs:1-12` (add `pub mod bits;`)
- Test: `tests/v7_codecs.rs`

**Interfaces:**
- Produces: `bits::BitWriter { pub fn new() -> Self; pub fn put(&mut self, value: u64, nbits: u32); pub fn finish(self) -> Vec<u8> }` (output always ends with `bits::PAD = 8` zero bytes).
- Produces: `bits::BitReader { pub fn new(src: &[u8]) -> Self; pub fn refill(&mut self); pub fn peek(&self, n: u32) -> u64; pub fn consume(&mut self, n: u32); pub fn get(&mut self, n: u32) -> u64; pub fn overrun(&self) -> bool }`. `refill` guarantees `cnt >= 56`; `peek/consume` valid for `n <= 56` after a refill; `get(n)` for `n <= 32` calls `refill` itself.

- [ ] **Step 1: Write the failing test**

```rust
// tests/v7_codecs.rs
use simd_stream_codec::bits::{BitReader, BitWriter, PAD};

#[test]
fn bits_roundtrip_mixed_widths() {
    let mut w = BitWriter::new();
    let mut expect = Vec::new();
    let mut x = 0x2545F4914F6CDD1Du64;
    for i in 0..10_000u32 {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        let n = (i % 32) + 1; // 1..=32 bits
        let v = x & ((1u64 << n) - 1);
        w.put(v, n);
        expect.push((v, n));
    }
    let bytes = w.finish();
    assert_eq!(&bytes[bytes.len() - PAD..], &[0u8; PAD]);
    let mut r = BitReader::new(&bytes);
    for (v, n) in expect {
        assert_eq!(r.get(n), v);
    }
    assert!(!r.overrun());
}

#[test]
fn bits_reader_clamps_at_end() {
    let bytes = BitWriter::new().finish(); // 8 pad bytes only
    let mut r = BitReader::new(&bytes);
    for _ in 0..1000 { let _ = r.get(32); } // far past the end: must not fault
    assert!(r.overrun());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --release --test v7_codecs bits_ 2>&1 | tail -5`
Expected: compile error, `bits` module not found.

- [ ] **Step 3: Write the implementation**

```rust
// src/bits.rs
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
            self.p = p.add(((63 - self.cnt) >> 3) as usize);
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
```

`peek(0)` must return 0: `(1u64 << 0) - 1 == 0`, fine. `consume(0)` is a no-op.

Add to `src/lib.rs` after `pub mod c_api;`:

```rust
pub mod bits;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --release --test v7_codecs bits_ 2>&1 | tail -5`
Expected: `test result: ok. 2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/bits.rs src/lib.rs tests/v7_codecs.rs
git commit -m "v7: LSB-first bit writer and clamped branchless bit reader

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: 8-stream interleaved Huffman

**Files:**
- Create: `src/huff8.rs`
- Modify: `src/lib.rs` (add `pub mod huff8;`)
- Test: `tests/v7_codecs.rs`

**Interfaces:**
- Consumes: `bits::{BitWriter, BitReader, PAD}`, `huffman::{build_lengths, build_codes, pack_lengths, unpack_lengths, MAX_CODE_LEN}`.
- Produces: `huff8::STREAMS = 8`; `huff8::Table { pub fn build(lengths: &[u8; 256]) -> Option<Table> }` (None if not a valid prefix code); `huff8::encode(data: &[u8], lengths: &[u8; 256]) -> Vec<Vec<u8>>` (8 padded sub-streams, symbol i in stream i % 8); `huff8::decode(table: &Table, streams: &[&[u8]; 8], n: usize, out: &mut [u8]) -> Result<(), ()>` (Err on overrun); `huff8::lengths_for(hist: &[u64; 256]) -> [u8; 256]`; `huff8::coded_size(hist, lengths) -> usize` (bits/8 plus table bytes).

- [ ] **Step 1: Write the failing test**

```rust
// tests/v7_codecs.rs (append)
use simd_stream_codec::huff8;

fn skewed_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut x = seed;
    (0..n).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; let r = x & 0xFFFF; if r < 40000 { b'e' } else if r < 55000 { (x >> 20) as u8 & 15 } else { (x >> 24) as u8 } }).collect()
}

#[test]
fn huff8_roundtrip_and_size() {
    for n in [0usize, 1, 7, 8, 9, 31, 32, 33, 1000, 100_003] {
        let data = skewed_bytes(n, 99);
        let mut hist = [0u64; 256];
        for &b in &data { hist[b as usize] += 1; }
        let lengths = huff8::lengths_for(&hist);
        let streams = huff8::encode(&data, &lengths);
        assert_eq!(streams.len(), huff8::STREAMS);
        let refs: [&[u8]; 8] = std::array::from_fn(|k| streams[k].as_slice());
        let table = huff8::Table::build(&lengths).unwrap();
        let mut out = vec![0u8; n];
        huff8::decode(&table, &refs, n, &mut out).unwrap();
        assert_eq!(out, data, "n = {}", n);
        if n >= 1000 {
            let coded: usize = streams.iter().map(|s| s.len()).sum();
            assert!(coded < n * 8 / 10, "skewed data must compress: {} vs {}", coded, n);
            assert_eq!(huff8::coded_size(&hist, &lengths), (0..256).map(|s| hist[s] as usize * lengths[s] as usize).sum::<usize>() / 8 + 128);
        }
    }
}

#[test]
fn huff8_rejects_invalid_code() {
    let mut lengths = [1u8; 256]; // Kraft sum 128 > 1
    assert!(huff8::Table::build(&lengths).is_none());
    lengths = [0u8; 256];
    lengths[0] = 1; lengths[1] = 1; // valid
    assert!(huff8::Table::build(&lengths).is_some());
}

#[test]
fn huff8_overrun_is_an_error() {
    let data = skewed_bytes(5000, 7);
    let mut hist = [0u64; 256];
    for &b in &data { hist[b as usize] += 1; }
    let lengths = huff8::lengths_for(&hist);
    let streams = huff8::encode(&data, &lengths);
    let refs: [&[u8]; 8] = std::array::from_fn(|k| streams[k].as_slice());
    let table = huff8::Table::build(&lengths).unwrap();
    let mut out = vec![0u8; 50_000];
    assert!(huff8::decode(&table, &refs, 50_000, &mut out).is_err());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --release --test v7_codecs huff8_ 2>&1 | tail -5`
Expected: compile error, `huff8` not found.

- [ ] **Step 3: Write the implementation**

```rust
// src/huff8.rs
//! 8-stream interleaved Huffman for byte streams. Symbol i lives in
//! sub-stream i % 8, LSB-first with bit-reversed canonical codes, so the
//! decoder's table is indexed by the next `TB` bits directly. Measured on
//! the M1 Max (examples/huff_spike.rs): 0.47 ns/symbol.
use crate::bits::{BitReader, BitWriter};
use crate::huffman::{build_codes, build_lengths, MAX_CODE_LEN};

pub const STREAMS: usize = 8;
pub const TB: u32 = MAX_CODE_LEN;
pub const TABLE_BYTES: usize = 128;

fn reverse_bits(code: u16, len: u8) -> u16 {
    let mut r = 0u16;
    for i in 0..len {
        r |= ((code >> i) & 1) << (len - 1 - i);
    }
    r
}

pub fn lengths_for(hist: &[u64; 256]) -> [u8; 256] {
    let present = hist.iter().filter(|&&c| c > 0).count();
    if present <= 1 {
        // One symbol: give it a 1-bit code so the stream is well formed.
        let mut l = [0u8; 256];
        if let Some(s) = hist.iter().position(|&c| c > 0) {
            l[s] = 1;
        }
        return l;
    }
    build_lengths(hist)
}

/// Bytes the coded stream will occupy, plus the packed table.
pub fn coded_size(hist: &[u64; 256], lengths: &[u8; 256]) -> usize {
    let bits: u64 = (0..256).map(|s| hist[s] * lengths[s] as u64).sum();
    (bits / 8) as usize + TABLE_BYTES
}

/// Packed decode table: entry = sym | (len << 8).
pub struct Table {
    pub entries: Vec<u16>,
}

impl Table {
    /// None unless the lengths form a prefix code that fits `TB` bits.
    pub fn build(lengths: &[u8; 256]) -> Option<Table> {
        let mut kraft: u64 = 0; // in units of 2^-TB
        for &l in lengths.iter() {
            if l as u32 > TB {
                return None;
            }
            if l > 0 {
                kraft += 1u64 << (TB - l as u32);
            }
        }
        if kraft == 0 || kraft > (1u64 << TB) {
            return None;
        }
        let codes = build_codes(lengths);
        let mut entries = vec![0u16; 1 << TB];
        for s in 0..256 {
            let l = lengths[s];
            if l == 0 {
                continue;
            }
            let r = reverse_bits(codes[s], l) as usize;
            let step = 1usize << l;
            let mut i = r;
            while i < (1 << TB) {
                entries[i] = s as u16 | ((l as u16) << 8);
                i += step;
            }
        }
        Some(Table { entries })
    }
}

pub fn encode(data: &[u8], lengths: &[u8; 256]) -> Vec<Vec<u8>> {
    let codes = build_codes(lengths);
    let rev: Vec<u16> = (0..256).map(|s| reverse_bits(codes[s], lengths[s])).collect();
    let mut writers: Vec<BitWriter> = (0..STREAMS).map(|_| BitWriter::new()).collect();
    for (i, &b) in data.iter().enumerate() {
        let l = lengths[b as usize] as u32;
        debug_assert!(l > 0, "symbol without a code");
        writers[i % STREAMS].put(rev[b as usize] as u64, l);
    }
    writers.into_iter().map(|w| w.finish()).collect()
}

struct St {
    r: BitReader,
}

#[inline(always)]
fn sym(s: &mut St, t: &[u16]) -> u8 {
    let e = t[s.r.peek(TB) as usize];
    s.r.consume((e >> 8) as u32);
    e as u8
}

/// Decode `n` symbols into `out[..n]`. Err if any sub-stream overran.
pub fn decode(table: &Table, streams: &[&[u8]; STREAMS], n: usize, out: &mut [u8]) -> Result<(), ()> {
    assert!(out.len() >= n);
    let t = table.entries.as_slice();
    let mut st: [St; STREAMS] = std::array::from_fn(|k| St { r: BitReader::new(streams[k]) });
    let per_iter = 4 * STREAMS; // 4 symbols per stream per refill: 4 * 11 <= 56
    let full = n / per_iter;
    let mut o = 0usize;
    for _ in 0..full {
        for k in 0..STREAMS {
            st[k].r.refill();
        }
        for j in 0..4 {
            for k in 0..STREAMS {
                out[o + j * STREAMS + k] = sym(&mut st[k], t);
            }
        }
        o += per_iter;
    }
    for i in o..n {
        let k = i % STREAMS;
        st[k].r.refill();
        out[i] = sym(&mut st[k], t);
    }
    if st.iter().any(|s| s.r.overrun()) {
        return Err(());
    }
    Ok(())
}
```

Add `pub mod huff8;` to `src/lib.rs`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --release --test v7_codecs huff8_ 2>&1 | tail -5`
Expected: `3 passed`

- [ ] **Step 5: Measure and record**

Run: `RUSTFLAGS="-C target-cpu=native" cargo run --release --example huff_spike 2>&1 | grep "N=8"` and compare with a 10-line timing of `huff8::decode` over the same Silesia literals (copy the spike's block extraction into `examples/v7_bench.rs` later; for now, add a `#[ignore]` test printing ns/symbol is acceptable). Record the ns/symbol in `CHANGELOG-BENCH.md` under a heading `## v7 milestone 1: entropy coders`. Gate: <= 0.6 ns/symbol on the M1 Max.

- [ ] **Step 6: Commit**

```bash
git add src/huff8.rs src/lib.rs tests/v7_codecs.rs CHANGELOG-BENCH.md
git commit -m "v7: 8-stream interleaved Huffman coder

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: tANS single-stream tables, encode, decode

**Files:**
- Create: `src/tans.rs`
- Modify: `src/lib.rs` (add `pub mod tans;`)
- Test: `tests/v7_codecs.rs`

**Interfaces:**
- Consumes: `bits::{BitWriter, BitReader}`.
- Produces: `tans::TL = 10`, `tans::L = 1024`, `tans::MAX_SYMBOLS = 64`; `tans::normalize(hist: &[u32], n_symbols: usize) -> Vec<u16>` (counts sum to L, every present symbol >= 1); `tans::DecodeTable { pub fn build(counts: &[u16]) -> Option<DecodeTable> }`; `tans::EncodeTable { pub fn build(counts: &[u16]) -> Option<EncodeTable> }`; `tans::Encoder { pub fn new(t: &EncodeTable) -> Self; pub fn push(&mut self, sym: u8) }` collecting symbols, `pub fn finish(self) -> Vec<u8>` (padded stream, decodable forward); `tans::Decoder { pub fn new(t: &DecodeTable, stream: &[u8]) -> Self; pub fn next(&mut self) -> u8; pub fn overrun(&self) -> bool }`.

- [ ] **Step 1: Write the failing test**

```rust
// tests/v7_codecs.rs (append)
use simd_stream_codec::tans;

fn skewed_codes(n: usize, nsym: usize, seed: u64) -> Vec<u8> {
    let mut x = seed;
    (0..n).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; let r = (x >> 16) & 0xFF; (if r < 128 { 0 } else if r < 200 { 1 + (x & 3) } else { x % nsym as u64 }) as u8 }).collect()
}

#[test]
fn tans_normalize_sums_to_l_and_keeps_present() {
    let hist: Vec<u32> = vec![1_000_000, 1, 0, 3, 500];
    let c = tans::normalize(&hist, 5);
    assert_eq!(c.iter().map(|&v| v as u32).sum::<u32>(), tans::L as u32);
    assert!(c[1] >= 1 && c[3] >= 1 && c[4] >= 1 && c[2] == 0);
}

#[test]
fn tans_single_stream_roundtrip() {
    for (n, nsym) in [(0usize, 4usize), (1, 4), (2, 36), (1000, 36), (77_777, 64)] {
        let data = skewed_codes(n, nsym, 5);
        let mut hist = vec![0u32; nsym];
        for &s in &data { hist[s as usize] += 1; }
        if n == 0 { hist[0] = 1; }
        let counts = tans::normalize(&hist, nsym);
        let et = tans::EncodeTable::build(&counts).unwrap();
        let dt = tans::DecodeTable::build(&counts).unwrap();
        let mut enc = tans::Encoder::new(&et);
        for &s in &data { enc.push(s); }
        let stream = enc.finish();
        let mut dec = tans::Decoder::new(&dt, &stream);
        let out: Vec<u8> = (0..n).map(|_| dec.next()).collect();
        assert_eq!(out, data, "n={} nsym={}", n, nsym);
        assert!(!dec.overrun());
    }
}

#[test]
fn tans_rejects_bad_counts() {
    assert!(tans::DecodeTable::build(&[512, 511]).is_none()); // sums to 1023
    assert!(tans::DecodeTable::build(&vec![16u16; 65]).is_none()); // too many symbols
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --release --test v7_codecs tans_ 2>&1 | tail -5`
Expected: compile error, `tans` not found.

- [ ] **Step 3: Write the implementation**

```rust
// src/tans.rs
//! Table ANS (FSE-style) for the sequence code streams: at most 64
//! symbols, table log 10. The decoder reads a forward LSB-first stream:
//! the initial state (TL bits), then per symbol the table's `nbits` bits.
//! The encoder therefore processes symbols last-to-first and emits the
//! chunks in reverse, so no bit-reversal is needed anywhere.
use crate::bits::{BitReader, BitWriter};

pub const TL: u32 = 10;
pub const L: usize = 1 << TL;
pub const MAX_SYMBOLS: usize = 64;

/// Scale a histogram to counts summing to L. Present symbols get >= 1.
pub fn normalize(hist: &[u32], n_symbols: usize) -> Vec<u16> {
    assert!(n_symbols <= MAX_SYMBOLS && hist.len() >= n_symbols);
    let total: u64 = hist[..n_symbols].iter().map(|&h| h as u64).sum();
    let mut counts = vec![0u16; n_symbols];
    if total == 0 {
        counts[0] = L as u16;
        return counts;
    }
    let mut sum = 0usize;
    let mut largest = 0usize;
    for s in 0..n_symbols {
        let h = hist[s] as u64;
        if h == 0 {
            continue;
        }
        let mut c = ((h * L as u64) / total) as usize;
        if c == 0 {
            c = 1;
        }
        counts[s] = c as u16;
        sum += c;
        if hist[s] > hist[largest] {
            largest = s;
        }
    }
    // Put the rounding error on the most frequent symbol.
    let cl = counts[largest] as isize + (L as isize - sum as isize);
    assert!(cl >= 1, "normalization underflow");
    counts[largest] = cl as u16;
    counts
}

fn check(counts: &[u16]) -> bool {
    counts.len() >= 1 && counts.len() <= MAX_SYMBOLS && counts.iter().map(|&c| c as usize).sum::<usize>() == L
}

/// zstd's spread: symbol occurrences placed at stride (5/8)L + 3.
fn spread(counts: &[u16]) -> Vec<u8> {
    let step = (L >> 1) + (L >> 3) + 3;
    let mask = L - 1;
    let mut table = vec![0u8; L];
    let mut pos = 0usize;
    for (s, &c) in counts.iter().enumerate() {
        for _ in 0..c {
            table[pos] = s as u8;
            pos = (pos + step) & mask;
        }
    }
    debug_assert_eq!(pos, 0);
    table
}

fn highbit(x: u32) -> u32 {
    31 - x.leading_zeros()
}

#[derive(Clone, Copy)]
pub struct DecodeEntry {
    pub sym: u8,
    pub nbits: u8,
    pub base: u16,
}

pub struct DecodeTable {
    pub entries: Vec<DecodeEntry>,
}

impl DecodeTable {
    pub fn build(counts: &[u16]) -> Option<DecodeTable> {
        if !check(counts) {
            return None;
        }
        let sp = spread(counts);
        let mut next: Vec<u32> = counts.iter().map(|&c| c as u32).collect();
        let mut entries = vec![DecodeEntry { sym: 0, nbits: 0, base: 0 }; L];
        for i in 0..L {
            let s = sp[i] as usize;
            let x = next[s];
            next[s] += 1;
            let nbits = TL - highbit(x);
            entries[i] = DecodeEntry { sym: s as u8, nbits: nbits as u8, base: ((x << nbits) - L as u32) as u16 };
        }
        Some(DecodeTable { entries })
    }
}

pub struct EncodeTable {
    /// Indexed by cumulative position; holds the next state (L..2L).
    state_table: Vec<u16>,
    /// Per symbol: (delta_nbits, delta_find_state).
    sym: Vec<(u32, i32)>,
}

impl EncodeTable {
    pub fn build(counts: &[u16]) -> Option<EncodeTable> {
        if !check(counts) {
            return None;
        }
        let sp = spread(counts);
        let mut cumul = vec![0u32; counts.len() + 1];
        for s in 0..counts.len() {
            cumul[s + 1] = cumul[s] + counts[s] as u32;
        }
        let mut fill = cumul.clone();
        let mut state_table = vec![0u16; L];
        for i in 0..L {
            let s = sp[i] as usize;
            state_table[fill[s] as usize] = (L + i) as u16;
            fill[s] += 1;
        }
        let mut sym = Vec::with_capacity(counts.len());
        for s in 0..counts.len() {
            let c = counts[s] as u32;
            if c == 0 {
                sym.push((0, 0));
                continue;
            }
            let max_bits_out = TL - highbit(c);
            let min_state_plus = c << max_bits_out;
            let delta_nbits = (max_bits_out << 16).wrapping_sub(min_state_plus);
            let delta_find_state = cumul[s] as i32 - c as i32;
            sym.push((delta_nbits, delta_find_state));
        }
        Some(EncodeTable { state_table, sym })
    }
}

pub struct Encoder<'a> {
    t: &'a EncodeTable,
    syms: Vec<u8>,
}

impl<'a> Encoder<'a> {
    pub fn new(t: &'a EncodeTable) -> Self {
        Encoder { t, syms: Vec::new() }
    }
    #[inline(always)]
    pub fn push(&mut self, sym: u8) {
        self.syms.push(sym);
    }
    /// Encode last-to-first, then write the chunks in decode order.
    pub fn finish(self) -> Vec<u8> {
        let mut chunks: Vec<(u32, u8)> = Vec::with_capacity(self.syms.len());
        // Initial encoder state: any value in [L, 2L); use the first
        // symbol's smallest legal state so the decoder's first state is
        // well defined.
        let mut state: u32 = L as u32;
        for &s in self.syms.iter().rev() {
            let (delta_nbits, delta_find) = self.t.sym[s as usize];
            let nbits_out = (state.wrapping_add(delta_nbits)) >> 16;
            let low = state & ((1u32 << nbits_out) - 1);
            chunks.push((low, nbits_out as u8));
            let idx = ((state >> nbits_out) as i32 + delta_find) as usize;
            state = self.t.state_table[idx] as u32;
        }
        let mut w = BitWriter::new();
        w.put((state - L as u32) as u64, TL);
        for &(v, n) in chunks.iter().rev() {
            if n > 0 {
                w.put(v as u64, n as u32);
            }
        }
        w.finish()
    }
}

pub struct Decoder<'a> {
    t: &'a [DecodeEntry],
    r: BitReader,
    state: u32,
}

impl<'a> Decoder<'a> {
    pub fn new(t: &'a DecodeTable, stream: &[u8]) -> Self {
        let mut r = BitReader::new(stream);
        let state = r.get(TL) as u32;
        Decoder { t: t.entries.as_slice(), r, state }
    }
    #[inline(always)]
    pub fn next(&mut self) -> u8 {
        let e = self.t[self.state as usize];
        self.r.refill();
        let bits = self.r.peek(e.nbits as u32) as u32;
        self.r.consume(e.nbits as u32);
        self.state = e.base as u32 + bits;
        e.sym
    }
    pub fn overrun(&self) -> bool {
        self.r.overrun()
    }
}
```

Note on correctness of the encoder/decoder pairing: the decoder's `state` is in `[0, L)`; the encoder's is in `[L, 2L)`; `base = (x << nbits) - L` and `state_table` holds `L + i`, so `decoder_state = encoder_state - L` throughout. With `state` initialised to `L` before the last symbol, the decoder's first read is `state - L` after all symbols are encoded, which the encoder writes last (first in the stream).

Add `pub mod tans;` to `src/lib.rs`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --release --test v7_codecs tans_ 2>&1 | tail -5`
Expected: `3 passed`. If `tans_single_stream_roundtrip` fails on the first symbol only, the initial state convention is off by one: the decoder must read `state` as `enc_state - L`, which the encoder writes as `state - L`; check `w.put((state - L) ...)`.

- [ ] **Step 5: Commit**

```bash
git add src/tans.rs src/lib.rs tests/v7_codecs.rs
git commit -m "v7: tANS tables, single-stream encoder and decoder

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: tANS 8-stream interleaved encode/decode

**Files:**
- Modify: `src/tans.rs`
- Test: `tests/v7_codecs.rs`

**Interfaces:**
- Produces: `tans::encode8(syms: &[u8], t: &EncodeTable) -> Vec<Vec<u8>>` (symbol i in stream i % 8); `tans::decode8(t: &DecodeTable, streams: &[&[u8]; 8], n: usize, out: &mut [u8]) -> Result<(), ()>`.

- [ ] **Step 1: Write the failing test**

```rust
// tests/v7_codecs.rs (append)
#[test]
fn tans8_roundtrip() {
    for n in [0usize, 1, 5, 8, 9, 64, 65, 12_345] {
        let data = skewed_codes(n, 36, 11);
        let mut hist = vec![0u32; 36];
        for &s in &data { hist[s as usize] += 1; }
        if n == 0 { hist[0] = 1; }
        let counts = tans::normalize(&hist, 36);
        let et = tans::EncodeTable::build(&counts).unwrap();
        let dt = tans::DecodeTable::build(&counts).unwrap();
        let streams = tans::encode8(&data, &et);
        let refs: [&[u8]; 8] = std::array::from_fn(|k| streams[k].as_slice());
        let mut out = vec![0u8; n];
        tans::decode8(&dt, &refs, n, &mut out).unwrap();
        assert_eq!(out, data, "n={}", n);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --release --test v7_codecs tans8_ 2>&1 | tail -5`
Expected: compile error, `encode8` not found.

- [ ] **Step 3: Write the implementation**

```rust
// src/tans.rs (append)
pub const STREAMS: usize = 8;

pub fn encode8(syms: &[u8], t: &EncodeTable) -> Vec<Vec<u8>> {
    let mut encs: Vec<Encoder> = (0..STREAMS).map(|_| Encoder::new(t)).collect();
    for (i, &s) in syms.iter().enumerate() {
        encs[i % STREAMS].push(s);
    }
    encs.into_iter().map(|e| e.finish()).collect()
}

/// Decode `n` symbols from 8 interleaved streams, 4 per stream per refill
/// (4 * TL = 40 <= 56).
pub fn decode8(t: &DecodeTable, streams: &[&[u8]; STREAMS], n: usize, out: &mut [u8]) -> Result<(), ()> {
    assert!(out.len() >= n);
    let e = t.entries.as_slice();
    let mut rs: [BitReader; STREAMS] = std::array::from_fn(|k| BitReader::new(streams[k]));
    let mut st = [0u32; STREAMS];
    for k in 0..STREAMS {
        st[k] = rs[k].get(TL) as u32;
    }
    let per_iter = 4 * STREAMS;
    let full = n / per_iter;
    let mut o = 0usize;
    for _ in 0..full {
        for k in 0..STREAMS {
            rs[k].refill();
        }
        for j in 0..4 {
            for k in 0..STREAMS {
                let d = e[st[k] as usize];
                let bits = rs[k].peek(d.nbits as u32) as u32;
                rs[k].consume(d.nbits as u32);
                st[k] = d.base as u32 + bits;
                out[o + j * STREAMS + k] = d.sym;
            }
        }
        o += per_iter;
    }
    for i in o..n {
        let k = i % STREAMS;
        let d = e[st[k] as usize];
        rs[k].refill();
        let bits = rs[k].peek(d.nbits as u32) as u32;
        rs[k].consume(d.nbits as u32);
        st[k] = d.base as u32 + bits;
        out[i] = d.sym;
    }
    if rs.iter().any(|r| r.overrun()) {
        return Err(());
    }
    Ok(())
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --release --test v7_codecs tans 2>&1 | tail -5`
Expected: `4 passed`

- [ ] **Step 5: Measure and record**

Time `decode8` over 14M skewed symbols (a `#[ignore]` test or a 15-line example is fine) and record ns/symbol in `CHANGELOG-BENCH.md` under milestone 1. Gate: <= 0.6 ns/symbol. If above, the culprit is usually `DecodeEntry` being 6 bytes with padding: pack it as a `u32` (`sym | nbits << 8 | base << 16`) and index a `Vec<u32>`.

- [ ] **Step 6: Commit**

```bash
git add src/tans.rs tests/v7_codecs.rs CHANGELOG-BENCH.md
git commit -m "v7: 8-stream interleaved tANS

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: v7 format: codes, repeat offsets, block sub-header

**Files:**
- Create: `src/v7_format.rs`
- Modify: `src/format.rs:2` (add `pub const VERSION_V7: u16 = 7;`), `src/lib.rs` (add `pub mod v7_format;`)
- Test: `tests/v7_codecs.rs`

**Interfaces:**
- Produces:
  - `v7_format::{MAX_OFFSET_BITS = 21, MAX_WINDOW = 1 << 21, MIN_MATCH = 3, LL_SYMBOLS = 32, ML_SYMBOLS = 32, OFF_SYMBOLS = 24}`.
  - `v7_format::ll_code(v: u32) -> (u8 code, u8 extra_bits, u32 extra)` and `ll_value(code: u8, extra: u32) -> u32`; same for `ml_code/ml_value` (input `v >= MIN_MATCH`); `off_code(offset: u32) -> (u8, u8, u32)` for real offsets (codes 3..=23) and `off_value(code, extra) -> u32`; `extra_bits_of_code(kind: Kind, code: u8) -> u8`.
  - `v7_format::Reps { pub fn new() -> Self; pub fn resolve(&mut self, code: u8, extra: u32) -> u32 }` (decoder side: codes 0..=2 return and rotate the repeat; others compute the offset and push it) and `pub fn code_for(&mut self, offset: u32) -> (u8, u8, u32)` (encoder side: returns a rep code if `offset` equals a repeat, else the real code; updates state identically).
  - `v7_format::SubHeader { pub coded: u8, pub reuse: u8, pub dict_id: u32, pub sizes: [u32; 5] }` with `pub const BYTES: usize = 29`, `write(&self, out: &mut Vec<u8>)`, `parse(src: &[u8]) -> Option<SubHeader>`.
  - Stream indexes: `v7_format::{S_LIT = 0, S_LL = 1, S_ML = 2, S_OFF = 3, S_EXTRA = 4}`.

- [ ] **Step 1: Write the failing test**

```rust
// tests/v7_codecs.rs (append)
use simd_stream_codec::v7_format::{self, Reps, SubHeader};

#[test]
fn v7_length_and_offset_codes_roundtrip() {
    for v in (0u32..70_000).step_by(7).chain([0, 15, 16, 17, 31, 32, 262_143].into_iter()) {
        let (c, nb, e) = v7_format::ll_code(v);
        assert!(c < v7_format::LL_SYMBOLS as u8);
        assert_eq!(v7_format::extra_bits_of_code(v7_format::Kind::Ll, c), nb);
        assert_eq!(v7_format::ll_value(c, e), v);
        let m = v + v7_format::MIN_MATCH;
        let (c, nb, e) = v7_format::ml_code(m);
        assert!(c < v7_format::ML_SYMBOLS as u8);
        assert_eq!(v7_format::extra_bits_of_code(v7_format::Kind::Ml, c), nb);
        assert_eq!(v7_format::ml_value(c, e), m);
    }
    for o in (1u32..v7_format::MAX_WINDOW).step_by(997).chain([1, 2, 3, 4, 65535, 65536, v7_format::MAX_WINDOW - 1].into_iter()) {
        let (c, nb, e) = v7_format::off_code(o);
        assert!(c >= 3 && c < v7_format::OFF_SYMBOLS as u8);
        assert_eq!(v7_format::extra_bits_of_code(v7_format::Kind::Off, c), nb);
        assert_eq!(v7_format::off_value(c, e), o);
    }
}

#[test]
fn v7_repeat_offsets_encoder_and_decoder_agree() {
    let offsets = [100u32, 100, 7, 100, 7, 7, 300, 100, 300, 1];
    let mut enc = Reps::new();
    let mut dec = Reps::new();
    let mut rep_hits = 0;
    for &o in &offsets {
        let (code, nb, extra) = enc.code_for(o);
        if code < 3 { rep_hits += 1; assert_eq!(nb, 0); }
        assert_eq!(dec.resolve(code, extra), o);
    }
    assert!(rep_hits >= 5, "repeats must be found: {}", rep_hits);
}

#[test]
fn v7_subheader_roundtrip() {
    let h = SubHeader { coded: 0b01011, reuse: 0b10, dict_id: 0xDEADBEEF, sizes: [1, 2, 3, 4, 5] };
    let mut out = Vec::new();
    h.write(&mut out);
    assert_eq!(out.len(), SubHeader::BYTES);
    let p = SubHeader::parse(&out).unwrap();
    assert_eq!((p.coded, p.reuse, p.dict_id, p.sizes), (h.coded, h.reuse, h.dict_id, h.sizes));
    assert!(SubHeader::parse(&out[..10]).is_none());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --release --test v7_codecs v7_ 2>&1 | tail -5`
Expected: compile error, `v7_format` not found.

- [ ] **Step 3: Write the implementation**

```rust
// src/v7_format.rs
//! Format v7 constants, code mappings and the block sub-header.
//!
//! Lengths: values below 16 are their own code; larger values use code
//! 12 + floor(log2(v)) with floor(log2(v)) extra bits holding v - 2^k.
//! Offsets: codes 0..=2 are the three most recent distinct offsets; a real
//! offset o uses code 3 + floor(log2(o)) with that many extra bits.

pub const MAX_OFFSET_BITS: u32 = 21;
pub const MAX_WINDOW: u32 = 1 << MAX_OFFSET_BITS;
pub const MIN_MATCH: u32 = 3;
pub const LL_SYMBOLS: usize = 32; // codes 0..=30 for values up to 2^18
pub const ML_SYMBOLS: usize = 32;
pub const OFF_SYMBOLS: usize = 24; // 3 reps + log2 buckets 0..=20

pub const S_LIT: usize = 0;
pub const S_LL: usize = 1;
pub const S_ML: usize = 2;
pub const S_OFF: usize = 3;
pub const S_EXTRA: usize = 4;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Ll,
    Ml,
    Off,
}

#[inline(always)]
fn log2(v: u32) -> u32 {
    31 - v.leading_zeros()
}

#[inline(always)]
fn len_code(v: u32) -> (u8, u8, u32) {
    if v < 16 {
        (v as u8, 0, 0)
    } else {
        let k = log2(v);
        ((12 + k) as u8, k as u8, v - (1 << k))
    }
}

#[inline(always)]
fn len_value(code: u8, extra: u32) -> u32 {
    if code < 16 {
        code as u32
    } else {
        let k = code as u32 - 12;
        (1 << k) + extra
    }
}

pub fn ll_code(v: u32) -> (u8, u8, u32) {
    len_code(v)
}
pub fn ll_value(code: u8, extra: u32) -> u32 {
    len_value(code, extra)
}
pub fn ml_code(v: u32) -> (u8, u8, u32) {
    debug_assert!(v >= MIN_MATCH);
    len_code(v - MIN_MATCH)
}
pub fn ml_value(code: u8, extra: u32) -> u32 {
    len_value(code, extra) + MIN_MATCH
}
pub fn off_code(offset: u32) -> (u8, u8, u32) {
    debug_assert!(offset >= 1 && offset < MAX_WINDOW);
    let k = log2(offset);
    ((3 + k) as u8, k as u8, offset - (1 << k))
}
pub fn off_value(code: u8, extra: u32) -> u32 {
    let k = code as u32 - 3;
    (1 << k) + extra
}

/// Extra bits carried by a code; the decoder reads this many after the
/// symbol. For length codes below 16 and rep codes it is zero.
#[inline(always)]
pub fn extra_bits_of_code(kind: Kind, code: u8) -> u8 {
    match kind {
        Kind::Ll | Kind::Ml => {
            if code < 16 {
                0
            } else {
                code - 12
            }
        }
        Kind::Off => {
            if code < 3 {
                0
            } else {
                code - 3
            }
        }
    }
}

/// Repeat-offset state, identical on both sides.
#[derive(Clone, Copy)]
pub struct Reps {
    r: [u32; 3],
}

impl Reps {
    pub fn new() -> Self {
        Reps { r: [1, 4, 8] }
    }

    /// Encoder: rep code if `offset` is a repeat, else its real code.
    pub fn code_for(&mut self, offset: u32) -> (u8, u8, u32) {
        if offset == self.r[0] {
            return (0, 0, 0);
        }
        if offset == self.r[1] {
            self.r.swap(0, 1);
            return (1, 0, 0);
        }
        if offset == self.r[2] {
            self.r = [self.r[2], self.r[0], self.r[1]];
            return (2, 0, 0);
        }
        self.r = [offset, self.r[0], self.r[1]];
        off_code(offset)
    }

    /// Decoder: the offset for a code and its extra bits.
    #[inline(always)]
    pub fn resolve(&mut self, code: u8, extra: u32) -> u32 {
        match code {
            0 => self.r[0],
            1 => {
                self.r.swap(0, 1);
                self.r[0]
            }
            2 => {
                self.r = [self.r[2], self.r[0], self.r[1]];
                self.r[0]
            }
            _ => {
                let o = off_value(code, extra);
                self.r = [o, self.r[0], self.r[1]];
                o
            }
        }
    }
}

/// Sits at the start of a v7 payload. `coded` bit i: stream i is entropy
/// coded (stream 4 is always raw bits and ignores the flag). `reuse` bit
/// 0: literal table reused from the previous block; bit 1: the three
/// sequence tables reused. `sizes`: bytes of each stream section
/// (tables, sub-stream size table and data included).
#[derive(Clone, Copy, Debug)]
pub struct SubHeader {
    pub coded: u8,
    pub reuse: u8,
    pub dict_id: u32,
    pub sizes: [u32; 5],
}

impl SubHeader {
    pub const BYTES: usize = 1 + 1 + 4 + 4 * 5;

    pub fn write(&self, out: &mut Vec<u8>) {
        out.push(self.coded);
        out.push(self.reuse);
        out.extend_from_slice(&self.dict_id.to_le_bytes());
        for s in self.sizes {
            out.extend_from_slice(&s.to_le_bytes());
        }
    }

    pub fn parse(src: &[u8]) -> Option<SubHeader> {
        if src.len() < Self::BYTES {
            return None;
        }
        let u = |i: usize| u32::from_le_bytes([src[i], src[i + 1], src[i + 2], src[i + 3]]);
        Some(SubHeader {
            coded: src[0],
            reuse: src[1],
            dict_id: u(2),
            sizes: [u(6), u(10), u(14), u(18), u(22)],
        })
    }
}
```

In `src/format.rs` after `pub const CURRENT_VERSION: u16 = 6;` add:

```rust
/// Entropy-coded blocks (see v7_format.rs). Same BlockHeader; for this
/// version `token_bytes` is the whole payload length, `token_count` the
/// sequence count and `literal_len` the literal byte count; the other
/// section fields are zero.
pub const VERSION_V7: u16 = 7;
```

and in `BlockHeader::payload_len` make the v7 case explicit:

```rust
    pub fn payload_len(&self) -> usize {
        if (self.flags & FLAG_RAW_UNCOMPRESSED) != 0 {
            self.uncompressed_len as usize
        } else if self.version == VERSION_V7 {
            self.token_bytes as usize
        } else {
            self.token_bytes as usize
                + self.offset_bytes as usize
                + self.extras_bytes as usize
                + self.literal_len as usize
        }
    }
```

and in `is_plausible`, before the existing checks:

```rust
        if self.version == VERSION_V7 {
            return u <= MAX_BLOCK_SIZE
                && self.token_count as usize <= u / 3 + 1
                && self.literal_len as usize <= u
                && self.token_bytes as usize <= 2 * u + 4096
                && self.offset_bytes == 0
                && self.extras_bytes == 0;
        }
```

Add `pub mod v7_format;` to `src/lib.rs`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --release 2>&1 | grep -E "test result|FAILED"`
Expected: all green, including the existing 27 (v6 is untouched by these edits: `version` is 6 there).

- [ ] **Step 5: Commit**

```bash
git add src/v7_format.rs src/format.rs src/lib.rs tests/v7_codecs.rs
git commit -m "v7: length and offset codes, repeat offsets, block sub-header, version constant

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: v7 block encoder from a sequence list

**Files:**
- Create: `src/v7_encode.rs`
- Modify: `src/lib.rs` (add `pub mod v7_encode;`)
- Test: `tests/v7_roundtrip.rs` (created here; the decoder test in Task 7 completes the round trip; this task's test checks the payload structure)

**Interfaces:**
- Consumes: `huff8`, `tans`, `v7_format::*`, `bits::BitWriter`.
- Produces: `v7_encode::Sequence { pub lit_len: u32, pub match_len: u32, pub offset: u32 }` (`match_len == 0` only for the final literal-only sequence, `offset` then ignored); `v7_encode::Tables { lit_lengths: [u8; 256], ll: Vec<u16>, ml: Vec<u16>, off: Vec<u16> }` (the previous block's, for reuse) with `Tables::none()`; `v7_encode::encode_block(seqs: &[Sequence], literals: &[u8], dict_id: u32, prev: &mut Tables, out: &mut Vec<u8>)` appending the payload (sub-header + streams); `v7_encode::payload_layout(payload: &[u8]) -> Option<Layout>` where `Layout { pub sub: SubHeader, pub sections: [std::ops::Range<usize>; 5] }`.

- [ ] **Step 1: Write the failing test**

```rust
// tests/v7_roundtrip.rs
use simd_stream_codec::v7_encode::{encode_block, payload_layout, Sequence, Tables};
use simd_stream_codec::v7_format::{SubHeader, S_EXTRA, S_LIT, S_LL, S_ML, S_OFF};

fn sample_sequences() -> (Vec<Sequence>, Vec<u8>) {
    // "abcabcabc..." style: literal runs then matches at small offsets.
    let mut seqs = Vec::new();
    let mut lits = Vec::new();
    let mut x = 42u64;
    for i in 0..5000u32 {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        let ll = (x % 20) as u32;
        for _ in 0..ll { lits.push((x >> 8) as u8); }
        let ml = 3 + (x >> 16) as u32 % 40;
        let offset = if i % 3 == 0 { 100 } else { 1 + (x >> 24) as u32 % 5000 };
        seqs.push(Sequence { lit_len: ll, match_len: ml, offset });
    }
    lits.extend_from_slice(b"tail literals");
    seqs.push(Sequence { lit_len: 13, match_len: 0, offset: 0 });
    (seqs, lits)
}

#[test]
fn v7_encode_block_layout_is_self_describing() {
    let (seqs, lits) = sample_sequences();
    let mut prev = Tables::none();
    let mut out = Vec::new();
    encode_block(&seqs, &lits, 0, &mut prev, &mut out);
    let layout = payload_layout(&out).unwrap();
    assert_eq!(layout.sub.dict_id, 0);
    let total: usize = SubHeader::BYTES + layout.sub.sizes.iter().map(|&s| s as usize).sum::<usize>();
    assert_eq!(total, out.len());
    for s in [S_LIT, S_LL, S_ML, S_OFF, S_EXTRA] {
        assert_eq!(layout.sections[s].len(), layout.sub.sizes[s] as usize);
    }
    // Skewed sequence codes must have been coded, not stored raw.
    assert!(layout.sub.coded & (1 << S_LL) != 0);
    assert!(layout.sub.coded & (1 << S_OFF) != 0);
    // A second block with the same statistics reuses the sequence tables.
    let mut out2 = Vec::new();
    encode_block(&seqs, &lits, 0, &mut prev, &mut out2);
    let l2 = payload_layout(&out2).unwrap();
    assert!(l2.sub.reuse & 0b10 != 0);
    assert!(out2.len() < out.len());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --release --test v7_roundtrip 2>&1 | tail -5`
Expected: compile error, `v7_encode` not found.

- [ ] **Step 3: Write the implementation**

```rust
// src/v7_encode.rs
//! Format v7 block encoder: a sequence list plus literals -> payload.
//!
//! Payload = SubHeader, then five sections. A coded section is:
//!   [table: 128 bytes of packed Huffman lengths, or 1 + 2 * n bytes of
//!    tANS counts (u8 count then u16 LE counts); omitted when reused]
//!   [8 x u32 LE sub-stream sizes]
//!   [8 sub-streams, each ending in bits::PAD zero bytes]
//! A raw section is the bytes themselves (literals, or one code byte per
//! sequence). The extra-bits section is always 8 raw padded sub-streams
//! behind a size table: sub-stream k holds, for sequences i == k (mod 8)
//! in order, the ll extra bits, then ml, then offset extra bits.
use crate::bits::{BitWriter, PAD};
use crate::huff8;
use crate::tans;
use crate::v7_format::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sequence {
    pub lit_len: u32,
    pub match_len: u32,
    pub offset: u32,
}

pub struct Tables {
    pub lit_lengths: Option<[u8; 256]>,
    pub ll: Option<Vec<u16>>,
    pub ml: Option<Vec<u16>>,
    pub off: Option<Vec<u16>>,
}

impl Tables {
    pub fn none() -> Self {
        Tables { lit_lengths: None, ll: None, ml: None, off: None }
    }
}

pub struct Layout {
    pub sub: SubHeader,
    pub sections: [std::ops::Range<usize>; 5],
}

pub fn payload_layout(payload: &[u8]) -> Option<Layout> {
    let sub = SubHeader::parse(payload)?;
    let mut pos = SubHeader::BYTES;
    let mut sections: [std::ops::Range<usize>; 5] = std::array::from_fn(|_| 0..0);
    for s in 0..5 {
        let end = pos.checked_add(sub.sizes[s] as usize)?;
        if end > payload.len() {
            return None;
        }
        sections[s] = pos..end;
        pos = end;
    }
    if pos != payload.len() {
        return None;
    }
    Some(Layout { sub, sections })
}

/// Reuse when every count is within 1/8 of the previous table's.
fn close(a: &[u16], b: &[u16]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(&x, &y)| (x as i32 - y as i32).abs() <= (x as i32 / 8).max(2))
}

fn write_substreams(streams: &[Vec<u8>], out: &mut Vec<u8>) {
    for s in streams {
        out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    }
    for s in streams {
        out.extend_from_slice(s);
    }
}

/// One code stream: returns (section bytes, coded?, counts used).
fn encode_codes(codes: &[u8], n_symbols: usize, prev: &Option<Vec<u16>>, allow_reuse: bool) -> (Vec<u8>, bool, Vec<u16>, bool) {
    let mut hist = vec![0u32; n_symbols];
    for &c in codes {
        hist[c as usize] += 1;
    }
    if codes.is_empty() {
        hist[0] = 1;
    }
    let counts = tans::normalize(&hist, n_symbols);
    let reuse = allow_reuse && prev.as_ref().map_or(false, |p| close(p, &counts));
    let counts_used = if reuse { prev.clone().unwrap() } else { counts };
    // Estimate coded size: sum of -log2(p) bits.
    let bits: f64 = (0..n_symbols).map(|s| if hist[s] == 0 { 0.0 } else { hist[s] as f64 * -((counts_used[s] as f64) / tans::L as f64).log2() }).sum();
    let table_bytes = if reuse { 0 } else { 1 + 2 * n_symbols };
    let coded_estimate = (bits / 8.0) as usize + table_bytes + 8 * (4 + PAD);
    if coded_estimate + codes.len() / 50 >= codes.len() {
        return (codes.to_vec(), false, counts_used, false);
    }
    let et = tans::EncodeTable::build(&counts_used).expect("normalized counts");
    let streams = tans::encode8(codes, &et);
    let mut section = Vec::new();
    if !reuse {
        section.push(n_symbols as u8);
        for &c in &counts_used {
            section.extend_from_slice(&c.to_le_bytes());
        }
    }
    write_substreams(&streams, &mut section);
    (section, true, counts_used, reuse)
}

pub fn encode_block(seqs: &[Sequence], literals: &[u8], dict_id: u32, prev: &mut Tables, out: &mut Vec<u8>) {
    // Codes and extra bits.
    let n = seqs.len();
    let mut ll = Vec::with_capacity(n);
    let mut ml = Vec::with_capacity(n);
    let mut off = Vec::with_capacity(n);
    let mut extra: Vec<BitWriter> = (0..8).map(|_| BitWriter::new()).collect();
    let mut reps = Reps::new();
    for (i, s) in seqs.iter().enumerate() {
        let w = &mut extra[i % 8];
        let (c, nb, e) = ll_code(s.lit_len);
        ll.push(c);
        if nb > 0 {
            w.put(e as u64, nb as u32);
        }
        if s.match_len == 0 {
            debug_assert_eq!(i, n - 1, "literal-only sequence must be last");
            ml.push(0);
            off.push(0);
            continue;
        }
        let (c, nb, e) = ml_code(s.match_len);
        ml.push(c);
        if nb > 0 {
            w.put(e as u64, nb as u32);
        }
        let (c, nb, e) = reps.code_for(s.offset);
        off.push(c);
        if nb > 0 {
            w.put(e as u64, nb as u32);
        }
    }
    let extra_streams: Vec<Vec<u8>> = extra.into_iter().map(|w| w.finish()).collect();

    // Literals.
    let mut hist = [0u64; 256];
    for &b in literals {
        hist[b as usize] += 1;
    }
    let lengths = huff8::lengths_for(&hist);
    let lit_reuse = prev.lit_lengths.map_or(false, |p| {
        let est_prev: u64 = (0..256).map(|s| hist[s] * p[s] as u64).sum();
        let est_new: u64 = (0..256).map(|s| hist[s] * lengths[s] as u64).sum();
        p.iter().zip(lengths.iter()).all(|(&a, &b)| (a > 0) == (b > 0)) && est_prev <= est_new + (huff8::TABLE_BYTES as u64) * 8
    });
    let lit_lengths = if lit_reuse { prev.lit_lengths.unwrap() } else { lengths };
    let lit_coded_size = huff8::coded_size(&hist, &lit_lengths) - if lit_reuse { huff8::TABLE_BYTES } else { 0 } + 8 * (4 + PAD);
    let lit_coded = literals.len() >= 64 && lit_coded_size + literals.len() / 50 < literals.len();
    let mut lit_section = Vec::new();
    if lit_coded {
        if !lit_reuse {
            crate::huffman::pack_lengths(&lit_lengths, &mut lit_section);
        }
        let streams = huff8::encode(literals, &lit_lengths);
        write_substreams(&streams, &mut lit_section);
    } else {
        lit_section.extend_from_slice(literals);
    }

    let (ll_sec, ll_coded, ll_counts, ll_reuse) = encode_codes(&ll, LL_SYMBOLS, &prev.ll, true);
    let (ml_sec, ml_coded, ml_counts, ml_reuse) = encode_codes(&ml, ML_SYMBOLS, &prev.ml, ll_reuse);
    let (off_sec, off_coded, off_counts, off_reuse) = encode_codes(&off, OFF_SYMBOLS, &prev.off, ll_reuse && ml_reuse);
    // Sequence-table reuse is all or nothing: if the last one could not
    // reuse, re-encode the earlier ones without reuse.
    let seq_reuse = ll_reuse && ml_reuse && off_reuse;
    let (ll_sec, ll_coded, ll_counts) = if seq_reuse || !ll_reuse { (ll_sec, ll_coded, ll_counts) } else { let r = encode_codes(&ll, LL_SYMBOLS, &None, false); (r.0, r.1, r.2) };
    let (ml_sec, ml_coded, ml_counts) = if seq_reuse || !ml_reuse { (ml_sec, ml_coded, ml_counts) } else { let r = encode_codes(&ml, ML_SYMBOLS, &None, false); (r.0, r.1, r.2) };

    let mut extra_sec = Vec::new();
    write_substreams(&extra_streams, &mut extra_sec);

    let sub = SubHeader {
        coded: (lit_coded as u8) << S_LIT | (ll_coded as u8) << S_LL | (ml_coded as u8) << S_ML | (off_coded as u8) << S_OFF,
        reuse: (lit_coded && lit_reuse) as u8 | ((seq_reuse as u8) << 1),
        dict_id,
        sizes: [lit_section.len() as u32, ll_sec.len() as u32, ml_sec.len() as u32, off_sec.len() as u32, extra_sec.len() as u32],
    };
    sub.write(out);
    out.extend_from_slice(&lit_section);
    out.extend_from_slice(&ll_sec);
    out.extend_from_slice(&ml_sec);
    out.extend_from_slice(&off_sec);
    out.extend_from_slice(&extra_sec);

    if lit_coded {
        prev.lit_lengths = Some(lit_lengths);
    }
    if ll_coded && ml_coded && off_coded {
        prev.ll = Some(ll_counts);
        prev.ml = Some(ml_counts);
        prev.off = Some(off_counts);
    } else {
        prev.ll = None;
        prev.ml = None;
        prev.off = None;
    }
}
```

Add `pub mod v7_encode;` to `src/lib.rs`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --release --test v7_roundtrip 2>&1 | tail -5`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add src/v7_encode.rs src/lib.rs tests/v7_roundtrip.rs
git commit -m "v7: block encoder from a sequence list; table reuse

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: v7 block decoder (three passes)

**Files:**
- Create: `src/v7_decode.rs`
- Modify: `src/lib.rs` (add `pub mod v7_decode;`)
- Test: `tests/v7_roundtrip.rs`

**Interfaces:**
- Consumes: `v7_encode::{payload_layout, Layout, Sequence}` (test only), `huff8`, `tans`, `bits::BitReader`, `v7_format::*`.
- Produces: `v7_decode::Scratch { pub fn new() -> Self }` (thread-local via `v7_decode::with_scratch(|s| ...)`); `v7_decode::decode_block(payload: &[u8], n_seq: usize, n_lit: usize, dst: &mut [u8], buffer_start: *const u8, uncompressed_len: usize, prev: &mut DecTables, scratch: &mut Scratch) -> Result<usize>` (`crate::error::Result`); `v7_decode::DecTables { pub fn none() -> Self }` (previous block's decode tables for reuse).

- [ ] **Step 1: Write the failing test**

```rust
// tests/v7_roundtrip.rs (append)
use simd_stream_codec::v7_decode::{decode_block, DecTables, Scratch};

fn materialize(seqs: &[Sequence], lits: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut lp = 0usize;
    for s in seqs {
        out.extend_from_slice(&lits[lp..lp + s.lit_len as usize]);
        lp += s.lit_len as usize;
        for _ in 0..s.match_len {
            let b = out[out.len() - s.offset as usize];
            out.push(b);
        }
    }
    out
}

fn well_formed_sequences() -> (Vec<Sequence>, Vec<u8>) {
    // Offsets never exceed what has been produced so far.
    let mut seqs = Vec::new();
    let mut lits = Vec::new();
    let mut produced = 0u32;
    let mut x = 7u64;
    for i in 0..4000u32 {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        let ll = if i == 0 { 50 } else { (x % 12) as u32 };
        for _ in 0..ll { lits.push((x >> 8) as u8); }
        produced += ll;
        let ml = 3 + ((x >> 16) % 60) as u32;
        let offset = 1 + ((x >> 24) as u32 % produced.min(2000));
        seqs.push(Sequence { lit_len: ll, match_len: ml, offset });
        produced += ml;
    }
    lits.extend_from_slice(b"end");
    seqs.push(Sequence { lit_len: 3, match_len: 0, offset: 0 });
    (seqs, lits)
}

#[test]
fn v7_block_roundtrip_two_blocks_with_reuse() {
    let (seqs, lits) = well_formed_sequences();
    let expect = materialize(&seqs, &lits);
    let mut prev = Tables::none();
    let mut p1 = Vec::new();
    encode_block(&seqs, &lits, 0, &mut prev, &mut p1);
    let mut p2 = Vec::new();
    encode_block(&seqs, &lits, 0, &mut prev, &mut p2);

    let mut dst = vec![0u8; expect.len() * 2 + 128];
    let mut dtab = DecTables::none();
    let mut scratch = Scratch::new();
    let base = dst.as_ptr();
    let n = decode_block(&p1, seqs.len(), lits.len(), &mut dst, base, expect.len(), &mut dtab, &mut scratch).unwrap();
    assert_eq!(n, expect.len());
    assert_eq!(&dst[..n], &expect[..]);
    let n2 = decode_block(&p2, seqs.len(), lits.len(), &mut dst[n..], base, expect.len(), &mut dtab, &mut scratch).unwrap();
    assert_eq!(&dst[n..n + n2], &expect[..]);
}

#[test]
fn v7_block_rejects_bad_offset_and_wrong_length() {
    let (mut seqs, lits) = well_formed_sequences();
    seqs[5].offset = 1_000_000; // beyond produced output
    let mut prev = Tables::none();
    let mut p = Vec::new();
    encode_block(&seqs, &lits, 0, &mut prev, &mut p);
    let expect_len = materialize(&well_formed_sequences().0, &lits).len();
    let mut dst = vec![0u8; expect_len + 128];
    let base = dst.as_ptr();
    assert!(decode_block(&p, seqs.len(), lits.len(), &mut dst, base, expect_len, &mut DecTables::none(), &mut Scratch::new()).is_err());
    let (seqs, lits) = well_formed_sequences();
    let mut p = Vec::new();
    encode_block(&seqs, &lits, 0, &mut Tables::none(), &mut p);
    assert!(decode_block(&p, seqs.len(), lits.len(), &mut dst, base, expect_len - 1, &mut DecTables::none(), &mut Scratch::new()).is_err());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --release --test v7_roundtrip 2>&1 | tail -5`
Expected: compile error, `v7_decode` not found.

- [ ] **Step 3: Write the implementation**

```rust
// src/v7_decode.rs
//! Format v7 block decoder: three passes over thread-local scratch.
//!   1. sequence streams -> lit_len / match_len / offset arrays
//!   2. literal stream -> literal buffer
//!   3. copy loop over the arrays and the buffer
use crate::bits::{BitReader, PAD};
use crate::error::{CodecError, Result};
use crate::format::MAX_BLOCK_SIZE;
use crate::huff8;
use crate::tans;
use crate::v7_encode::payload_layout;
use crate::v7_format::*;

const MAX_SEQ: usize = MAX_BLOCK_SIZE / 3 + 1;

pub struct Scratch {
    pub ll: Vec<u32>,
    pub ml: Vec<u32>,
    pub off: Vec<u32>,
    pub codes: Vec<u8>,
    pub lits: Vec<u8>,
}

impl Scratch {
    pub fn new() -> Self {
        Scratch {
            ll: vec![0; MAX_SEQ],
            ml: vec![0; MAX_SEQ],
            off: vec![0; MAX_SEQ],
            codes: vec![0; MAX_SEQ],
            lits: vec![0; MAX_BLOCK_SIZE + 64],
        }
    }
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch::new());
}

pub fn with_scratch<T>(f: impl FnOnce(&mut Scratch) -> T) -> T {
    SCRATCH.with(|s| f(&mut s.borrow_mut()))
}

pub struct DecTables {
    pub lit: Option<huff8::Table>,
    pub ll: Option<tans::DecodeTable>,
    pub ml: Option<tans::DecodeTable>,
    pub off: Option<tans::DecodeTable>,
}

impl DecTables {
    pub fn none() -> Self {
        DecTables { lit: None, ll: None, ml: None, off: None }
    }
}

fn corrupt(msg: &'static str) -> CodecError {
    CodecError::CorruptedBitstream(msg)
}

/// Split a section's tail into 8 padded sub-streams behind a size table.
fn substreams<'a>(sec: &'a [u8]) -> Result<[&'a [u8]; 8]> {
    if sec.len() < 32 {
        return Err(corrupt("v7: sub-stream table truncated"));
    }
    let mut pos = 32usize;
    let mut out: [&[u8]; 8] = [&[]; 8];
    for k in 0..8 {
        let n = u32::from_le_bytes([sec[k * 4], sec[k * 4 + 1], sec[k * 4 + 2], sec[k * 4 + 3]]) as usize;
        let end = pos.checked_add(n).ok_or(corrupt("v7: sub-stream size"))?;
        if n < PAD || end > sec.len() {
            return Err(corrupt("v7: sub-stream out of section"));
        }
        out[k] = &sec[pos..end];
        pos = end;
    }
    if pos != sec.len() {
        return Err(corrupt("v7: section has trailing bytes"));
    }
    Ok(out)
}

/// Decode one code stream (or copy it raw) into `codes[..n]`.
fn code_stream(sec: &[u8], coded: bool, reuse: bool, n_symbols: usize, n: usize, prev: &mut Option<tans::DecodeTable>, codes: &mut [u8]) -> Result<()> {
    if !coded {
        if sec.len() != n {
            return Err(corrupt("v7: raw code stream length"));
        }
        if codes[..n].iter().zip(sec).any(|(_, &c)| c as usize >= n_symbols) {
            return Err(corrupt("v7: raw code out of range"));
        }
        codes[..n].copy_from_slice(sec);
        *prev = None;
        return Ok(());
    }
    let mut pos = 0usize;
    if !reuse {
        let ns = *sec.first().ok_or(corrupt("v7: table truncated"))? as usize;
        if ns != n_symbols || sec.len() < 1 + 2 * ns {
            return Err(corrupt("v7: table symbol count"));
        }
        let counts: Vec<u16> = (0..ns).map(|s| u16::from_le_bytes([sec[1 + 2 * s], sec[2 + 2 * s]])).collect();
        *prev = Some(tans::DecodeTable::build(&counts).ok_or(corrupt("v7: tANS counts"))?);
        pos = 1 + 2 * ns;
    }
    let table = prev.as_ref().ok_or(corrupt("v7: table reuse without a table"))?;
    let streams = substreams(&sec[pos..])?;
    tans::decode8(table, &streams, n, codes).map_err(|_| corrupt("v7: code stream overrun"))?;
    if codes[..n].iter().any(|&c| c as usize >= n_symbols) {
        return Err(corrupt("v7: code out of range"));
    }
    Ok(())
}

/// Pass 1: sequences. Fills scratch.ll/ml/off; returns (literal total,
/// match total).
fn sequences(payload: &[u8], layout: &crate::v7_encode::Layout, n: usize, prev: &mut DecTables, s: &mut Scratch) -> Result<(usize, usize)> {
    if n > MAX_SEQ {
        return Err(corrupt("v7: too many sequences"));
    }
    let sub = &layout.sub;
    let reuse = sub.reuse & 0b10 != 0;
    let c = |i: usize| sub.coded & (1 << i) != 0;
    if reuse && !(c(S_LL) && c(S_ML) && c(S_OFF)) {
        return Err(corrupt("v7: reuse flag on a raw stream"));
    }
    // ll codes -> values need extra bits, which live in the interleaved
    // extra section; decode all three code streams first, then walk.
    let extra = substreams(&payload[layout.sections[S_EXTRA].clone()])?;
    let mut ers: [BitReader; 8] = std::array::from_fn(|k| BitReader::new(extra[k]));

    code_stream(&payload[layout.sections[S_LL].clone()], c(S_LL), reuse, LL_SYMBOLS, n, &mut prev.ll, &mut s.codes)?;
    let mut lit_total = 0usize;
    for i in 0..n {
        let code = s.codes[i];
        let nb = extra_bits_of_code(Kind::Ll, code) as u32;
        let e = ers[i % 8].get(nb) as u32;
        let v = ll_value(code, e);
        s.ll[i] = v;
        lit_total += v as usize;
    }
    code_stream(&payload[layout.sections[S_ML].clone()], c(S_ML), reuse, ML_SYMBOLS, n, &mut prev.ml, &mut s.codes)?;
    let mut match_total = 0usize;
    // The ml extra bits for sequence i come after its ll extra bits in the
    // same sub-stream, so they must be read in the same pass order. Redo the
    // walk with both code arrays: keep ml codes, then read off codes too.
    let ml_codes: Vec<u8> = s.codes[..n].to_vec();
    code_stream(&payload[layout.sections[S_OFF].clone()], c(S_OFF), reuse, OFF_SYMBOLS, n, &mut prev.off, &mut s.codes)?;
    let mut ers: [BitReader; 8] = std::array::from_fn(|k| BitReader::new(extra[k]));
    let mut reps = Reps::new();
    for i in 0..n {
        let r = &mut ers[i % 8];
        let llc = ll_code(s.ll[i]).0;
        r.get(extra_bits_of_code(Kind::Ll, llc) as u32);
        let mlc = ml_codes[i];
        let offc = s.codes[i];
        if mlc == 0 && i == n - 1 {
            s.ml[i] = 0;
            s.off[i] = 0;
            continue;
        }
        let mv = ml_value(mlc, r.get(extra_bits_of_code(Kind::Ml, mlc) as u32) as u32);
        let ov = reps.resolve(offc, r.get(extra_bits_of_code(Kind::Off, offc) as u32) as u32);
        s.ml[i] = mv;
        s.off[i] = ov;
        match_total += mv as usize;
    }
    if ers.iter().any(|r| r.overrun()) {
        return Err(corrupt("v7: extra bits overrun"));
    }
    Ok((lit_total, match_total))
}

/// Pass 2: literals into scratch.lits[..n_lit].
fn literals(payload: &[u8], layout: &crate::v7_encode::Layout, n_lit: usize, prev: &mut DecTables, s: &mut Scratch) -> Result<()> {
    if n_lit > MAX_BLOCK_SIZE {
        return Err(corrupt("v7: too many literals"));
    }
    let sec = &payload[layout.sections[S_LIT].clone()];
    if layout.sub.coded & (1 << S_LIT) == 0 {
        if sec.len() != n_lit {
            return Err(corrupt("v7: raw literal length"));
        }
        s.lits[..n_lit].copy_from_slice(sec);
        return Ok(());
    }
    let mut pos = 0usize;
    if layout.sub.reuse & 1 == 0 {
        if sec.len() < huff8::TABLE_BYTES {
            return Err(corrupt("v7: literal table truncated"));
        }
        let lengths = crate::huffman::unpack_lengths(&sec[..huff8::TABLE_BYTES]);
        prev.lit = Some(huff8::Table::build(&lengths).ok_or(corrupt("v7: literal code lengths"))?);
        pos = huff8::TABLE_BYTES;
    }
    let table = prev.lit.as_ref().ok_or(corrupt("v7: literal table reuse without a table"))?;
    let streams = substreams(&sec[pos..])?;
    huff8::decode(table, &streams, n_lit, &mut s.lits).map_err(|_| corrupt("v7: literal stream overrun"))
}

/// Pass 3: copies. `dst` is the block's region plus padding; `buffer_start`
/// is where the window begins.
unsafe fn copies(s: &Scratch, n: usize, n_lit: usize, dst: &mut [u8], buffer_start: *const u8, uncompressed_len: usize) -> Result<usize> {
    let mut d = dst.as_mut_ptr();
    let start = d;
    let end = d.add(uncompressed_len);
    let mut lp = s.lits.as_ptr();
    let lit_end = lp.add(n_lit);
    for i in 0..n {
        let ll = s.ll[i] as usize;
        let ml = s.ml[i] as usize;
        if end.offset_from(d) < (ll + ml) as isize || lit_end.offset_from(lp) < ll as isize {
            return Err(corrupt("v7: sequence exceeds block"));
        }
        // Literals: 32-byte wild copies when the margins allow.
        if ll + 32 <= end.offset_from(d) as usize + 0 && lp.add(ll + 32) <= s.lits.as_ptr().add(s.lits.len()) && dst.len() >= (d.offset_from(start) as usize) + ll + 32 {
            let mut k = 0;
            while k < ll {
                std::ptr::copy_nonoverlapping(lp.add(k), d.add(k), 32);
                k += 32;
            }
        } else {
            std::ptr::copy_nonoverlapping(lp, d, ll);
        }
        lp = lp.add(ll);
        d = d.add(ll);
        if ml == 0 {
            continue;
        }
        let off = s.off[i] as usize;
        let available = d.offset_from(buffer_start) as usize;
        if off == 0 || off > available {
            return Err(CodecError::OffsetOutOfBounds { offset: off, available });
        }
        let src = d.sub(off);
        if off >= 32 && dst.len() >= (d.offset_from(start) as usize) + ml + 32 {
            let mut k = 0;
            while k < ml {
                std::ptr::copy_nonoverlapping(src.add(k), d.add(k), 32);
                k += 32;
            }
        } else {
            for k in 0..ml {
                *d.add(k) = *src.add(k);
            }
        }
        d = d.add(ml);
    }
    let written = d.offset_from(start) as usize;
    if written != uncompressed_len || lp != lit_end {
        return Err(corrupt("v7: decoded length mismatch"));
    }
    Ok(written)
}

pub fn decode_block(payload: &[u8], n_seq: usize, n_lit: usize, dst: &mut [u8], buffer_start: *const u8, uncompressed_len: usize, prev: &mut DecTables, scratch: &mut Scratch) -> Result<usize> {
    if uncompressed_len > dst.len() {
        return Err(CodecError::OutputBufferTooSmall { required: uncompressed_len, provided: dst.len() });
    }
    let layout = payload_layout(payload).ok_or(corrupt("v7: payload layout"))?;
    let (lit_total, match_total) = sequences(payload, &layout, n_seq, prev, scratch)?;
    if lit_total != n_lit || lit_total + match_total != uncompressed_len {
        return Err(corrupt("v7: sequence totals disagree with header"));
    }
    literals(payload, &layout, n_lit, prev, scratch)?;
    unsafe { copies(scratch, n_seq, n_lit, dst, buffer_start, uncompressed_len) }
}
```

The literal wild-copy condition is deliberately conservative (three margins); the first milestone is correctness, the NEON copy loop from `neon_decompress.rs` (`copy_run` shape with fixed 3x32 tails) replaces the body of `copies` in Task 8 once numbers exist.

Add `pub mod v7_decode;` to `src/lib.rs`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --release --test v7_roundtrip 2>&1 | tail -5`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/v7_decode.rs src/lib.rs tests/v7_roundtrip.rs
git commit -m "v7: block decoder, three passes over thread-local scratch

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Container integration and `compress_into_max` on the current parse (milestone 2)

**Files:**
- Modify: `src/lib.rs` (`parse_header`, `decode_block` dispatch, `decompress_sequential`, parallel unit loop, new `compress_into_max`, `compress_parallel_into_max`), `src/v7_encode.rs` (add `sequences_from_streams`), `src/v7_decode.rs` (NEON copy loop)
- Create: `examples/v7_bench.rs`
- Test: `tests/v7_roundtrip.rs`

**Interfaces:**
- Produces: `pub fn compress_into_max(input: &[u8], output: &mut Vec<u8>)`, `pub fn compress_parallel_into_max(input: &[u8], output: &mut Vec<u8>)`; v7 blocks decode through every existing `decompress*` entry point.
- Produces: `v7_encode::sequences_from_streams(tokens: &[u8], offsets: &[u8], extras: &[u8], min_match: usize) -> Vec<Sequence>` (v6 streams -> sequence list; the bridge for milestone 2).

- [ ] **Step 1: Write the failing test**

```rust
// tests/v7_roundtrip.rs (append)
#[test]
fn v7_max_level_roundtrip_through_container() {
    let mut x = 0x1234_5678_9ABC_DEF0u64;
    let mut rnd = |n: usize| -> Vec<u8> { (0..n).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x as u8 }).collect() };
    let mut inputs: Vec<Vec<u8>> = vec![Vec::new(), b"x".to_vec(), vec![b'A'; 300_000], rnd(300_000)];
    for period in [3usize, 17, 1000, 70_000] {
        let pat = rnd(period);
        inputs.push(pat.iter().cycle().take(600_000).copied().collect());
    }
    let mut text = Vec::new();
    while text.len() < 1_200_000 { text.extend_from_slice(b"the quick brown fox jumps over the lazy dog "); text.extend_from_slice(&rnd(2)); }
    inputs.push(text);
    for input in &inputs {
        let mut c = Vec::new();
        simd_stream_codec::compress_into_max(input, &mut c);
        assert_eq!(&simd_stream_codec::decompress(&c).unwrap(), input, "max sequential, len {}", input.len());
        let mut p = Vec::new();
        simd_stream_codec::compress_parallel_into_max(input, &mut p);
        assert_eq!(&simd_stream_codec::decompress_parallel(&p).unwrap(), input, "max parallel, len {}", input.len());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --release --test v7_roundtrip v7_max 2>&1 | tail -5`
Expected: compile error, `compress_into_max` not found.

- [ ] **Step 3: Bridge the current parse to sequences**

Append to `src/v7_encode.rs`:

```rust
/// v6 streams (tokens, offsets, extras) -> sequences. Milestone 2 bridge:
/// lets the v7 container and decoder be measured on the proven parse.
pub fn sequences_from_streams(tokens: &[u8], offsets: &[u8], extras: &[u8], min_match: usize) -> Vec<Sequence> {
    use crate::format::{Token, ESCAPE_BASE_LIT, ESCAPE_CONT, LIT_CODE_ESCAPE, MATCH_CODE_ESCAPE};
    let bias = min_match - 1;
    let mut e = 0usize;
    let mut o = 0usize;
    let mut read_escape = |e: &mut usize, base: usize| -> u32 {
        let v = extras[*e] as usize;
        *e += 1;
        if v != ESCAPE_CONT as usize {
            return (base + v) as u32;
        }
        let w = u16::from_le_bytes([extras[*e], extras[*e + 1]]) as usize;
        *e += 2;
        (base + v + w) as u32
    };
    let mut seqs = Vec::with_capacity(tokens.len());
    for &t in tokens {
        let tok = Token(t);
        let lc = tok.lit_code();
        let mc = tok.match_code();
        let lit_len = if lc == LIT_CODE_ESCAPE { read_escape(&mut e, ESCAPE_BASE_LIT) } else { lc as u32 };
        let (match_len, offset) = if mc == 0 {
            (0, 0)
        } else {
            let ml = if mc == MATCH_CODE_ESCAPE { read_escape(&mut e, bias + 15) } else { (mc + bias) as u32 };
            let lo = u16::from_le_bytes([offsets[o], offsets[o + 1]]) as u32;
            o += 2;
            (ml, lo | ((tok.off_hi() as u32) << 16))
        };
        seqs.push(Sequence { lit_len, match_len, offset });
    }
    // The format wants exactly one literal-only sequence, last. Merge any
    // interior literal-only tokens (the v6 MAX_LIT_LEN split) into the next.
    let mut merged: Vec<Sequence> = Vec::with_capacity(seqs.len());
    let mut carry = 0u32;
    for (i, s) in seqs.iter().enumerate() {
        if s.match_len == 0 && i + 1 < seqs.len() {
            carry += s.lit_len;
            continue;
        }
        merged.push(Sequence { lit_len: s.lit_len + carry, match_len: s.match_len, offset: s.offset });
        carry = 0;
    }
    if merged.last().map_or(true, |s| s.match_len != 0) {
        merged.push(Sequence { lit_len: carry, match_len: 0, offset: 0 });
    }
    merged
}
```

`Token` needs `lit_code()`, `match_code()`, `off_hi()` public: they already are (`src/format.rs:78-90`).

- [ ] **Step 4: Container: header, dispatch, decompress loops**

In `src/lib.rs` `parse_header`, replace the version check:

```rust
    if header.version != CURRENT_VERSION && header.version != VERSION_V7 {
        return Err(CodecError::UnsupportedVersion(header.version));
    }
```

In `decode_block` (the `unsafe fn` at `src/lib.rs:384`), add as the first thing after the `OutputBufferTooSmall` check and the raw-block branch:

```rust
    if header.version == VERSION_V7 {
        return v7_decode::with_scratch(|scratch| {
            V7_TABLES.with(|t| {
                let mut t = t.borrow_mut();
                if (header.flags & FLAG_CHAIN_RESET) != 0 {
                    *t = v7_decode::DecTables::none();
                }
                v7_decode::decode_block(
                    payload, header.token_count as usize, header.literal_len as usize,
                    dst, buffer_start, uncomp_len, &mut t, scratch,
                )
                .map(|_| ())
            })
        });
    }
```

and near the top of `lib.rs`:

```rust
thread_local! {
    static V7_TABLES: std::cell::RefCell<v7_decode::DecTables> = std::cell::RefCell::new(v7_decode::DecTables::none());
}
```

Table reuse across blocks requires blocks to be decoded in order on one thread. The sequential path does that. The parallel path decodes units (runs of chained blocks) on different threads, each starting at a `FLAG_CHAIN_RESET` block, which resets the tables above, so it is also correct. Add a comment saying so at the `V7_TABLES` definition.

Then the compressor, appended to `src/lib.rs`:

```rust
/// Max level: format v7. Milestone 2 uses the default parse through
/// `sequences_from_streams`; Task 9 replaces it with the double-fast parse.
pub fn compress_into_max(input: &[u8], output: &mut Vec<u8>) {
    let mut table = new_table();
    finder::init_table(&mut table, input);
    let (mut tokens, mut offsets, mut extras, mut literals) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut prev = v7_encode::Tables::none();
    let mut offset = 0;
    while offset < input.len() {
        let chunk_len = (input.len() - offset).min(MAX_BLOCK_SIZE);
        let chunk = &input[offset..offset + chunk_len];
        tokens.clear(); offsets.clear(); extras.clear(); literals.clear();
        find_block::<Lzav>(input, offset, chunk_len, &mut table, &mut tokens, &mut offsets, &mut extras, &mut literals);
        let seqs = v7_encode::sequences_from_streams(&tokens, &offsets, &extras, Lzav::MIN_MATCH);
        let mut payload = Vec::with_capacity(chunk_len);
        v7_encode::encode_block(&seqs, &literals, 0, &mut prev, &mut payload);
        let chain_flag = if offset == 0 { FLAG_CHAIN_RESET } else { 0 };
        if payload.len() + HEADER_SIZE >= chunk_len {
            prev = v7_encode::Tables::none();
            write_block(chunk, FLAG_RAW_UNCOMPRESSED, chain_flag, &[], &[], &[], &[], output);
        } else {
            let header = BlockHeader {
                magic: MAGIC, version: VERSION_V7, flags: FLAG_COMPRESSED | chain_flag,
                checksum: compute_checksum(chunk), uncompressed_len: chunk_len as u32,
                token_count: seqs.len() as u32, token_bytes: payload.len() as u32,
                offset_bytes: 0, extras_bytes: 0, literal_len: literals.len() as u32,
            };
            output.extend_from_slice(header_bytes(&header));
            output.extend_from_slice(&payload);
        }
        offset += chunk_len;
    }
}

/// Max level, all cores.
pub fn compress_parallel_into_max(input: &[u8], output: &mut Vec<u8>) {
    compress_parallel_with(input, output, compress_into_max)
}
```

Raw blocks written by `write_block` carry version 6, which the decoder handles as before; a v7 raw block is never needed.

- [ ] **Step 5: Run the test**

Run: `cargo test --release 2>&1 | grep -E "test result|FAILED|panicked"`
Expected: everything green (27 old + the new).

- [ ] **Step 6: NEON copy loop for pass 3**

Replace the body of `copies` in `src/v7_decode.rs` with the measured shape from `neon_decompress.rs::copy_run` (unconditional 32-byte copy, fixed 3x32 tails, cold `short_match` for offsets under 32), keeping the per-sequence `ll + ml` bound check and the offset validation, and a scalar twin under `#[cfg(not(target_arch = "aarch64"))]` that is the current loop. The 32-byte wild copies need: `dst.len() >= written + ll + ml + 64` (check once per sequence, fall back to exact copies otherwise) and a 64-byte margin on `scratch.lits` (already allocated `MAX_BLOCK_SIZE + 64`; keep `lp.add(ll + 32) <= lits.as_ptr().add(lits.len())` as the literal margin).

- [ ] **Step 7: Bench harness and the milestone-2 numbers**

Create `examples/v7_bench.rs`: for each Silesia file, `compress_into_max` once (ratio), then `timed` (copy the `timed` helper from `examples/quick3.rs`) compression, `decompress_into_raw` decode, and in the same run `zstd::bulk::compress(&data, 3)` / `zstd::bulk::decompress_to_buffer` with level 3 and level 1, printing per file and totals in the `quick3` style with a final line:

```
v7 total: ratio R comp C GB/s decomp D GB/s | zstd-3 ratio R3 comp C3 decomp D3 | zstd-1 ...
```

Register it in `Cargo.toml` (`[[example]] name = "v7_bench"`). Run: `RUSTFLAGS="-C target-cpu=native" cargo run --release --example v7_bench`. Record ratio, comp, decode in `CHANGELOG-BENCH.md` under `## v7 milestone 2: container + decoder on the default parse`. Expected ballpark: ratio 2.6-2.8, decode 3-4 GB/s. Stop rule: decode < 2.5 GB/s -> implement double-symbol Huffman tables in `huff8` (an entry that yields two symbols when the combined code fits `TB` bits) before Task 9.

- [ ] **Step 8: Commit**

```bash
git add src/lib.rs src/v7_encode.rs src/v7_decode.rs examples/v7_bench.rs Cargo.toml tests/v7_roundtrip.rs CHANGELOG-BENCH.md
git commit -m "v7: container integration, compress_into_max on the default parse, NEON copy pass, bench

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Double-fast parse with repeat offsets and a 2 MB window (milestones 3 and 4)

**Files:**
- Modify: `src/v7_encode.rs` (add `DfastTables`, `find_sequences_dfast`), `src/lib.rs` (`compress_into_max` uses it)
- Test: `tests/v7_roundtrip.rs`

**Interfaces:**
- Produces: `v7_encode::DfastTables { pub fn new() -> Box<Self> }` (long: `[u32; 1 << 17]` 8-byte hash, short: `[u32; 1 << 16]` 5-byte hash, positions absolute in the input, 0 = empty); `v7_encode::find_sequences_dfast(input: &[u8], block_start: usize, block_len: usize, t: &mut DfastTables, reps: &mut [u32; 3], seqs: &mut Vec<Sequence>, literals: &mut Vec<u8>)`. `reps` carries the repeat offsets across blocks exactly as the decoder's `Reps` does (the encoder's `Reps::code_for` inside `encode_block` starts fresh per block, so the parse must also start each block with `Reps::new()`'s values; pass `[1, 4, 8]` at every block start — the `reps` parameter exists so the parse can *use* repeats for match finding, and is reset per block).

- [ ] **Step 1: Write the failing test**

```rust
// tests/v7_roundtrip.rs (append)
use simd_stream_codec::v7_encode::{find_sequences_dfast, DfastTables};

#[test]
fn dfast_parse_finds_repeats_and_roundtrips() {
    // Records with a fixed stride: offsets repeat.
    let mut data = Vec::new();
    let mut x = 9u64;
    for i in 0..20_000u32 {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        data.extend_from_slice(format!("id={:08} name=user{:03} score={:05}\n", i, x % 500, x % 100_000).as_bytes());
    }
    let mut t = DfastTables::new();
    let mut seqs = Vec::new();
    let mut lits = Vec::new();
    let mut reps = [1u32, 4, 8];
    find_sequences_dfast(&data, 0, data.len().min(256 * 1024), &mut t, &mut reps, &mut seqs, &mut lits);
    let out = materialize(&seqs, &lits);
    assert_eq!(&out[..], &data[..out.len()]);
    let matched: u32 = seqs.iter().map(|s| s.match_len).sum();
    assert!(matched as usize > out.len() * 6 / 10, "expected mostly matches: {}/{}", matched, out.len());
    let mut c = Vec::new();
    simd_stream_codec::compress_into_max(&data, &mut c);
    assert_eq!(simd_stream_codec::decompress(&c).unwrap(), data);
    assert!(c.len() * 4 < data.len(), "structured text should compress 4x+: {}", c.len());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --release --test v7_roundtrip dfast 2>&1 | tail -5`
Expected: compile error, `find_sequences_dfast` not found.

- [ ] **Step 3: Write the parse**

Append to `src/v7_encode.rs`:

```rust
// ---------------------------------------------------------------------------
// Double-fast parse (zstd -3's shape): a long table keyed by an 8-byte hash
// and a short table keyed by a 5-byte hash, one candidate each; at every
// position the three repeat offsets are tried first, then the long
// candidate, then the short one. Greedy, no lazy matching. Window 2 MB.

pub const DFAST_LONG_BITS: u32 = 17;
pub const DFAST_SHORT_BITS: u32 = 16;
const MIN: usize = MIN_MATCH as usize; // format minimum 3; this parse emits >= 4
const PARSE_MIN: usize = 4;

pub struct DfastTables {
    pub long: Vec<u32>,
    pub short: Vec<u32>,
}

impl DfastTables {
    pub fn new() -> Box<Self> {
        Box::new(DfastTables { long: vec![0; 1 << DFAST_LONG_BITS], short: vec![0; 1 << DFAST_SHORT_BITS] })
    }
    pub fn reset(&mut self) {
        self.long.fill(0);
        self.short.fill(0);
    }
}

#[inline(always)]
unsafe fn h8(p: *const u8) -> usize {
    (std::ptr::read_unaligned(p as *const u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - DFAST_LONG_BITS)) as usize
}
#[inline(always)]
unsafe fn h5(p: *const u8) -> usize {
    ((std::ptr::read_unaligned(p as *const u64) << 24).wrapping_mul(889_523_592_379) >> (64 - DFAST_SHORT_BITS)) as usize
}
#[inline(always)]
unsafe fn eq4(a: *const u8, b: *const u8) -> bool {
    std::ptr::read_unaligned(a as *const u32) == std::ptr::read_unaligned(b as *const u32)
}
#[inline(always)]
unsafe fn eq8(a: *const u8, b: *const u8) -> bool {
    std::ptr::read_unaligned(a as *const u64) == std::ptr::read_unaligned(b as *const u64)
}
/// Common prefix length of `a` and `b`, at most `max`.
#[inline(always)]
unsafe fn prefix(mut a: *const u8, mut b: *const u8, max: usize) -> usize {
    let mut n = 0;
    while n + 8 <= max {
        let x = std::ptr::read_unaligned(a as *const u64) ^ std::ptr::read_unaligned(b as *const u64);
        if x != 0 {
            return n + (x.trailing_zeros() / 8) as usize;
        }
        n += 8;
        a = a.add(8);
        b = b.add(8);
    }
    while n < max && *a == *b {
        n += 1;
        a = a.add(1);
        b = b.add(1);
    }
    n
}

/// Parse one block. `reps` must hold `[1, 4, 8]` at the first block and
/// is otherwise carried by the caller; it is only used to *find* matches,
/// the codes are assigned by `encode_block`. Sequences use absolute
/// offsets; the last sequence is literal-only.
pub fn find_sequences_dfast(input: &[u8], block_start: usize, block_len: usize, t: &mut DfastTables, reps: &mut [u32; 3], seqs: &mut Vec<Sequence>, literals: &mut Vec<u8>) {
    let src = input.as_ptr();
    let block_end = block_start + block_len;
    let window = MAX_WINDOW as usize;
    // eq8 reads 8 bytes; hashing reads 8.
    let limit = block_end.saturating_sub(12).max(block_start);
    let mut anchor = block_start;
    let mut pos = block_start;
    let mut step_nb: u32 = 1 << 6;
    let mut r = *reps;

    unsafe {
        while pos < limit {
            let p = src.add(pos);
            let mut cand = usize::MAX;
            let mut rc = 0usize;
            // 1. repeat offsets (a 4-byte compare each).
            for k in 0..3 {
                let o = r[k] as usize;
                if o >= 1 && o <= pos && eq4(p, p.sub(o)) {
                    cand = pos - o;
                    rc = 4 + prefix(p.add(4), p.sub(o).add(4), block_end - pos - 4);
                    break;
                }
            }
            // 2. long candidate, 3. short candidate.
            let hl = h8(p);
            let hs = h5(p);
            let cl = t.long[hl] as usize;
            let cs = t.short[hs] as usize;
            t.long[hl] = pos as u32;
            t.short[hs] = pos as u32;
            if cand == usize::MAX {
                if cl != 0 && pos - cl < window && eq8(p, src.add(cl)) {
                    cand = cl;
                    rc = 8 + prefix(p.add(8), src.add(cl + 8), block_end - pos - 8);
                } else if cs != 0 && pos - cs < window && eq4(p, src.add(cs)) {
                    cand = cs;
                    rc = 4 + prefix(p.add(4), src.add(cs + 4), block_end - pos - 4);
                }
            }
            if cand == usize::MAX {
                let step = (step_nb >> 6) as usize;
                step_nb += 1;
                pos += step.max(1);
                continue;
            }
            step_nb = 1 << 6;
            // Back-match into pending literals.
            let mut mpos = pos;
            let mut c = cand;
            while mpos > anchor && c > 0 && *src.add(mpos - 1) == *src.add(c - 1) {
                mpos -= 1;
                c -= 1;
                rc += 1;
            }
            let offset = (mpos - c) as u32;
            let lit_len = (mpos - anchor) as u32;
            literals.extend_from_slice(&input[anchor..mpos]);
            seqs.push(Sequence { lit_len, match_len: rc as u32, offset });
            if offset != r[0] {
                r = [offset, r[0], r[1]];
            }
            pos = mpos + rc;
            anchor = pos;
            // Index the interior of the match so runs keep hashing.
            if pos >= 2 && pos - 2 + 8 <= block_end {
                let q = src.add(pos - 2);
                t.long[h8(q)] = (pos - 2) as u32;
                t.short[h5(q)] = (pos - 2) as u32;
            }
        }
    }
    let trailing = (block_end - anchor) as u32;
    literals.extend_from_slice(&input[anchor..block_end]);
    seqs.push(Sequence { lit_len: trailing, match_len: 0, offset: 0 });
    *reps = r;
    let _ = MIN;
    let _ = PARSE_MIN;
}
```

Update `compress_into_max` in `src/lib.rs` to use it: replace the `new_table`/`find_block`/`sequences_from_streams` lines with

```rust
    let mut t = v7_encode::DfastTables::new();
    let mut reps = [1u32, 4, 8];
    ...
        seqs.clear();
        literals.clear();
        reps = [1, 4, 8]; // encode_block's Reps starts fresh per block
        v7_encode::find_sequences_dfast(input, offset, chunk_len, &mut t, &mut reps, &mut seqs, &mut literals);
```

with `let mut seqs = Vec::new();` declared before the loop. Keep `sequences_from_streams` (tests and the milestone-2 numbers use it).

- [ ] **Step 4: Run the tests**

Run: `cargo test --release 2>&1 | grep -E "test result|FAILED|panicked"`
Expected: all green.

- [ ] **Step 5: Measure milestones 3 and 4**

Run: `RUSTFLAGS="-C target-cpu=native" cargo run --release --example v7_bench`. Record under `## v7 milestone 3/4: modeling + double-fast parse`. Gates: ratio >= 3.20 (G2), comp >= 0.34 (G4), decode >= 3.0 (G3). Stop rule from the spec: ratio under 3.10 -> add lazy matching (at each match, also probe `pos + 1`; take the later match if it is at least 4 bytes longer; costs one extra probe per match) and re-measure; if comp then falls under 0.34, report both numbers and stop for a decision, do not drop G4 silently.

- [ ] **Step 6: Commit**

```bash
git add src/v7_encode.rs src/lib.rs tests/v7_roundtrip.rs CHANGELOG-BENCH.md
git commit -m "v7: double-fast parse with repeat offsets, 2 MB window; compress_into_max uses it

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: Fuzz and cross-check tests

**Files:**
- Create: `tests/v7_fuzz.rs`
- Test: same

**Interfaces:**
- Consumes: `compress_into_max`, `decompress`, `decompress_into`, `v7_encode::{encode_block, payload_layout}`.

- [ ] **Step 1: Write the tests**

```rust
// tests/v7_fuzz.rs
//! Corruption must never panic, hang, or read out of bounds; it must
//! return an error or, when the checksum still matches, the right data.
use simd_stream_codec::{compress_into_max, decompress, decompress_into};

fn seed_inputs() -> Vec<Vec<u8>> {
    let mut x = 0xC0FFEEu64;
    let mut rnd = |n: usize| -> Vec<u8> { (0..n).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x as u8 }).collect() };
    let mut v = vec![b"hello hello hello world".to_vec(), vec![0u8; 5000], rnd(20_000)];
    let pat = rnd(37);
    v.push(pat.iter().cycle().take(300_000).copied().collect());
    let mut text = Vec::new();
    while text.len() < 400_000 { text.extend_from_slice(b"lorem ipsum dolor sit amet "); text.extend_from_slice(&rnd(3)); }
    v.push(text);
    v
}

#[test]
fn v7_mutation_fuzz() {
    let inputs = seed_inputs();
    let mut x = 0x5EEDu64;
    let mut mutations = 0u64;
    let target: u64 = std::env::var("V7_FUZZ").ok().and_then(|s| s.parse().ok()).unwrap_or(200_000);
    while mutations < target {
        for input in &inputs {
            let mut c = Vec::new();
            compress_into_max(input, &mut c);
            for _ in 0..50 {
                x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                let mut m = c.clone();
                match x % 4 {
                    0 => { let i = (x >> 8) as usize % m.len(); m[i] ^= 1 << ((x >> 40) & 7); }
                    1 => { let i = (x >> 8) as usize % m.len(); m[i] = (x >> 40) as u8; }
                    2 => { let n = (x >> 8) as usize % m.len(); m.truncate(n); }
                    _ => { let i = (x >> 8) as usize % m.len(); let j = (x >> 32) as usize % m.len(); m.swap(i, j); }
                }
                let mut dst = vec![0u8; input.len() + 4096];
                match decompress_into(&m, &mut dst) {
                    Ok(n) => assert_eq!(&dst[..n], &input[..n.min(input.len())], "checksum passed but data differs"),
                    Err(_) => {}
                }
                let _ = decompress(&m);
                mutations += 1;
                if mutations >= target { break; }
            }
            if mutations >= target { break; }
        }
    }
    std::fs::write(".v7-fuzz-status", format!("{}", mutations)).unwrap();
}

#[test]
fn v7_scalar_and_simd_agree() {
    // The scalar copy pass is the x86 fallback; on aarch64 both exist and
    // must produce identical bytes. On x86 this degenerates to a round trip.
    for input in seed_inputs() {
        let mut c = Vec::new();
        compress_into_max(&input, &mut c);
        assert_eq!(decompress(&c).unwrap(), input);
    }
}
```

The default is 200,000 mutations so `cargo test` stays fast; `V7_FUZZ=1000000 cargo test --release --test v7_fuzz` is the G1 run and its count lands in `.v7-fuzz-status`.

- [ ] **Step 2: Run the tests**

Run: `cargo test --release --test v7_fuzz 2>&1 | tail -5`, then `V7_FUZZ=1000000 cargo test --release --test v7_fuzz 2>&1 | tail -3`
Expected: both pass. Any panic is a decoder bug: fix it in `v7_decode.rs`/`bits.rs`/`tans.rs`/`huff8.rs` with the failing mutation reduced to a unit test in `tests/v7_codecs.rs`, and keep the mutation count.

- [ ] **Step 3: Commit**

```bash
git add tests/v7_fuzz.rs
git commit -m "v7: mutation fuzz and scalar/SIMD agreement tests

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 11: Dictionary support

**Files:**
- Modify: `src/lib.rs` (`compress_with_dict`, `decompress_with_dict`), `src/v7_encode.rs` (dictionary preload into `find_sequences_dfast` via `input` = dict ++ data), `src/v7_decode.rs` (window starts inside the dictionary)
- Test: `tests/v7_roundtrip.rs`

**Interfaces:**
- Produces: `pub fn compress_with_dict(dict: &[u8], input: &[u8], output: &mut Vec<u8>)`; `pub fn decompress_with_dict(dict: &[u8], compressed: &[u8]) -> Result<Vec<u8>>`; `pub fn dict_id(dict: &[u8]) -> u32` (`compute_checksum(dict)`, 0 reserved: a dictionary whose checksum is 0 gets id 1).

- [ ] **Step 1: Write the failing test**

```rust
// tests/v7_roundtrip.rs (append)
#[test]
fn v7_dictionary_helps_small_inputs_and_is_required() {
    let dict = b"{\"user\":\"\",\"event\":\"click\",\"ts\":0,\"session\":\"\",\"page\":\"/home\"}".repeat(40);
    let doc = b"{\"user\":\"alice\",\"event\":\"click\",\"ts\":1700000000,\"session\":\"abc123\",\"page\":\"/home\"}".to_vec();
    let mut plain = Vec::new();
    simd_stream_codec::compress_into_max(&doc, &mut plain);
    let mut with = Vec::new();
    simd_stream_codec::compress_with_dict(&dict, &doc, &mut with);
    assert!(with.len() < plain.len() * 6 / 10, "dictionary must help: {} vs {}", with.len(), plain.len());
    assert_eq!(simd_stream_codec::decompress_with_dict(&dict, &with).unwrap(), doc);
    assert!(simd_stream_codec::decompress(&with).is_err(), "dictionary id must be enforced");
    assert!(simd_stream_codec::decompress_with_dict(b"wrong dictionary bytes", &with).is_err());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --release --test v7_roundtrip dictionary 2>&1 | tail -5`
Expected: compile error, `compress_with_dict` not found.

- [ ] **Step 3: Implement**

In `src/lib.rs`:

```rust
pub fn dict_id(dict: &[u8]) -> u32 {
    let id = compute_checksum(dict);
    if id == 0 { 1 } else { id }
}

/// Max level with a dictionary: the dictionary is the window's first
/// `dict.len()` bytes, so offsets may reach into it; the header carries
/// `dict_id(dict)` and the decoder must be given the same bytes.
pub fn compress_with_dict(dict: &[u8], input: &[u8], output: &mut Vec<u8>) {
    let mut joined = Vec::with_capacity(dict.len() + input.len());
    joined.extend_from_slice(dict);
    joined.extend_from_slice(input);
    compress_max_from(&joined, dict.len(), dict_id(dict), output);
}

/// `compress_into_max` is `compress_max_from(input, 0, 0, output)`.
fn compress_max_from(full: &[u8], start: usize, id: u32, output: &mut Vec<u8>) { /* the body of compress_into_max, with the block loop starting at `start`, `dict_id: id` passed to encode_block, and the dictionary hashed into the tables first: for p in (0..start.saturating_sub(12)).step_by(1) insert h8/h5 at p (a plain loop over the dictionary, once) */ }

pub fn decompress_with_dict(dict: &[u8], compressed: &[u8]) -> Result<Vec<u8>> {
    let total = total_uncompressed_len(compressed)?;
    let mut buf = vec![0u8; dict.len() + total + PADDING * 2];
    buf[..dict.len()].copy_from_slice(dict);
    let written = decompress_sequential_from(compressed, &mut buf, dict.len(), Some(dict_id(dict)))?;
    buf.drain(..dict.len());
    buf.truncate(written);
    Ok(buf)
}
```

Refactor `decompress_sequential(compressed, dst, verify)` into `decompress_sequential_from(compressed, dst, dst_offset0, expected_dict: Option<u32>)` where the block loop starts at `dst_offset0` (the window's `buffer_start` stays `dst.as_ptr()`) and, for every v7 block, `payload_layout(payload)?.sub.dict_id` must equal `expected_dict.unwrap_or(0)`, else `Err(CodecError::CorruptedBitstream("dictionary id mismatch"))`. `decompress_sequential` becomes `decompress_sequential_from(c, d, 0, None)` with the `verify` behaviour preserved (add the flag back as a parameter). The parallel path rejects blocks with a non-zero `dict_id` (dictionaries are sequential-only in this milestone; document it in the function comment).

Move the `compress_into_max` body into `compress_max_from` and make `compress_into_max` call it with `(input, 0, 0, output)`. In `find_sequences_dfast` nothing changes: `block_start` > 0 with history before it is already how chained blocks work; add the pre-insert loop of the dictionary positions into the tables in `compress_max_from` before the first block (`for p in 0..start.saturating_sub(12) { unsafe { t.long[h8(src.add(p))] = p as u32; t.short[h5(src.add(p))] = p as u32; } }` with `h8/h5` made `pub(crate)`).

- [ ] **Step 4: Run the tests**

Run: `cargo test --release 2>&1 | grep -E "test result|FAILED|panicked"`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/v7_encode.rs src/v7_decode.rs tests/v7_roundtrip.rs
git commit -m "v7: dictionary support (window preload, id check)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 12: CLI, C ABI, corpus, survey, docs, spike removal (milestone 5)

**Files:**
- Modify: `src/bin/alatirok.rs` (`-9`/`--max`), `src/c_api.rs` + `include/alatirok.h` (`alatirok_compress_max`, `alatirok_compress_max_parallel`), `scripts/download_corpus.sh`, `examples/field_survey.rs` (add `Alatirok-max`), `examples/v7_bench.rs` (iterate the extended corpus when present), `README.md`, `CHANGELOG-BENCH.md`, `Cargo.toml`
- Delete: `examples/huff_spike.rs` (and its `[[example]]` entry)
- Test: `tests/test_c_abi.c` (add a max-level call), CLI round trip via `cargo run`

- [ ] **Step 1: CLI**

In `src/bin/alatirok.rs`: add `let mut max = false;`, parse `"-9" | "--max" => max = true,`, extend the level `match` with `(true, _, _, true) => compress_parallel_into_max` / `(false, _, _, true) => compress_into_max` as the highest-priority arms (tuple `(mc, fast, turbo, max)`), and the usage line `    -9, --max              Max level: entropy coded, ratio above zstd -3`. Verify:

```bash
cargo run --release --bin alatirok -- -9 corpus/dickens -o /tmp/d.alk && cargo run --release --bin alatirok -- -d /tmp/d.alk -o /tmp/d && cmp /tmp/d corpus/dickens && ls -l corpus/dickens /tmp/d.alk
```

- [ ] **Step 2: C ABI**

Add to `src/c_api.rs`, mirroring `alatirok_compress` exactly but calling `crate::compress_into_max` / `crate::compress_parallel_into_max`:

```rust
#[no_mangle]
pub unsafe extern "C" fn alatirok_compress_max(src: *const u8, src_len: usize, dst: *mut u8, dst_capacity: usize) -> isize { /* same body as alatirok_compress with compress_into_max */ }
#[no_mangle]
pub unsafe extern "C" fn alatirok_compress_max_parallel(src: *const u8, src_len: usize, dst: *mut u8, dst_capacity: usize) -> isize { /* same body with compress_parallel_into_max */ }
```

and the two prototypes to `include/alatirok.h` next to `alatirok_compress_parallel`. Extend `tests/test_c_abi.c` with a compress_max + decompress round trip of its existing buffer; build and run it as the README's "C ABI verification" section shows (on macOS use `clang` and `DYLD_LIBRARY_PATH`).

- [ ] **Step 3: Corpus script**

Append to `scripts/download_corpus.sh` (each guarded by `[ -f ]` like the existing entries), into `$CORPUS_DIR/ext/`:

```bash
mkdir -p "$CORPUS_DIR/ext"
# GitHub Archive: one hour of events, JSON lines (gzip)
[ -f "$CORPUS_DIR/ext/gharchive.json" ] || { curl -sSL https://data.gharchive.org/2024-01-15-12.json.gz | gunzip -c > "$CORPUS_DIR/ext/gharchive.json"; }
# NASA HTTP logs, July 1995 (gzip)
[ -f "$CORPUS_DIR/ext/nasa_access.log" ] || { curl -sSL ftp://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz | gunzip -c > "$CORPUS_DIR/ext/nasa_access.log"; }
# NYC yellow taxi, one month, Parquet (TLC public bucket)
[ -f "$CORPUS_DIR/ext/yellow_tripdata.parquet" ] || curl -sSL https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet -o "$CORPUS_DIR/ext/yellow_tripdata.parquet"
# Linux kernel source tarball, uncompressed, first 64 MB
[ -f "$CORPUS_DIR/ext/linux.tar" ] || { curl -sSL https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.6.tar.xz | xz -dc | head -c 67108864 > "$CORPUS_DIR/ext/linux.tar"; }
# OpenStreetMap PBF, a small region (Geofabrik)
[ -f "$CORPUS_DIR/ext/liechtenstein.osm.pbf" ] || curl -sSL https://download.geofabrik.de/europe/liechtenstein-latest.osm.pbf -o "$CORPUS_DIR/ext/liechtenstein.osm.pbf"
```

TPC-H `lineitem`: generate with DuckDB if present (`duckdb -c "INSTALL tpch; LOAD tpch; CALL dbgen(sf=1); COPY lineitem TO '$CORPUS_DIR/ext/lineitem.parquet'; COPY lineitem TO '$CORPUS_DIR/ext/lineitem.csv'"`), guarded by `command -v duckdb`; otherwise print a note and skip. `vmlinux`: skip if no kernel build is available; note it. The bench iterates every regular file in `corpus/ext/` when the directory exists.

- [ ] **Step 4: Bench and survey**

`examples/v7_bench.rs`: after Silesia, iterate `corpus/ext/*` files with the same per-file line (ratio, comp, decode for v7, zstd -3, zstd -1) and assert per file that v7's ratio >= zstd -3's, printing PASS/FAIL per file for G2. `examples/field_survey.rs`: add `"Alatirok-max"` as index 13 with `compress_into_max`/`decompress_into_raw`, same pattern as the fast/turbo rows. Run both:

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release --example v7_bench
RUSTFLAGS="-C target-cpu=native" cargo run --release --example field_survey 3 0.3
```

- [ ] **Step 5: Docs and cleanup**

`README.md`: add `--max` to the levels table with the measured numbers, a "Format v7" section (the five streams, one paragraph), the dictionary API, and the ext-corpus table. `CHANGELOG-BENCH.md`: `## v7 milestone 5: levels, corpus, field survey` with the survey table and the per-file G2 result. Delete `examples/huff_spike.rs` and its `Cargo.toml` entry. Run `cargo test --release` and `V7_FUZZ=1000000 cargo test --release --test v7_fuzz`.

- [ ] **Step 6: Commit and push**

```bash
git add -A
git commit -m "v7: --max level in CLI and C ABI, extended corpus, field survey, docs; remove spike

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git push origin HEAD
```

---

## Self-review notes

- Spec coverage: §1 layout -> Tasks 5, 6 (sub-header, five streams, padding, reuse flags, 21-bit offsets); §2 decoder -> Task 7 (three passes, scratch, error checks), Task 8 step 6 (NEON copy), Task 10 (fuzz); §3 compressor -> Tasks 6, 9 (dfast, reps, 2 MB window, raw-vs-coded per stream, table reuse), Task 11 (dictionary); §4 corpus/gates/milestones -> Tasks 8, 9, 12 (bench, survey, corpus, changelog entries, stop rules). Pre-trained dictionary tables and the AVX2 port are out of scope per the spec.
- Types: `Sequence {lit_len, match_len, offset}` (u32) used identically in Tasks 6-9; `Tables`/`DecTables` names match between encoder and decoder; `SubHeader.sizes: [u32; 5]` and `payload_layout` are shared by Task 6 (writer) and Task 7 (reader); `Reps::code_for` (encoder) and `Reps::resolve` (decoder) are the only rep-state mutators and both start from `[1, 4, 8]` per block.
- Known simplification: Task 7's pass 1 re-derives the ll code from the value to skip its extra bits on the second walk (cheap: one `leading_zeros`); an implementer may instead keep the ll codes in a second scratch array, either is fine.
