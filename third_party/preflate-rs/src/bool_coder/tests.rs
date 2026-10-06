//! The coder against the specification's own vectors, against a literal
//! model of its encoder (the specification's words, one shift at a time, over
//! a plain `Vec`), and on inputs made to break it. Nothing here needs another
//! coder: `vs_cabac.rs` compares with the crate this one replaces.

use std::io::{self, Cursor, Read};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use super::*;

/// How many carries went back through 0, 1, 2, 3 and 4 or more 0xff bytes
/// (the writer's own routine counts them: a run over a long workload says
/// how often the carry that the specification could not settle happens).
pub(super) static CARRIES: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

pub(super) fn note_carry(out: &[u8]) {
    let run = out.iter().rev().take_while(|&&b| b == 0xff).count();
    CARRIES[run.min(4)].fetch_add(1, Relaxed);
}

/// More work in a release build, which is how CI runs the tests.
fn scale() -> usize {
    if cfg!(debug_assertions) { 1 } else { 20 }
}

/// Bytes from hexadecimal, spaces or none between them.
pub(super) fn hex(s: &str) -> Vec<u8> {
    let digits: Vec<u8> = s.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    digits
        .chunks(2)
        .map(|d| u8::from_str_radix(std::str::from_utf8(d).unwrap(), 16).unwrap())
        .collect()
}

/// SplitMix64.
pub(super) struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    pub fn chance(&mut self, one_in: u64) -> bool {
        self.below(one_in) == 0
    }
}

// ---- the model: the specification's words, taken literally ----

/// Section 2: two counts, plain integers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct ModelCtx {
    pub n0: u32,
    pub n1: u32,
}

impl ModelCtx {
    pub fn new() -> Self {
        Self { n0: 1, n1: 1 }
    }

    pub fn p0(&self) -> u32 {
        256 * self.n0 / (self.n0 + self.n1)
    }

    pub fn update(&mut self, bit: bool) {
        let (mut x, mut y) = if bit { (self.n1, self.n0) } else { (self.n0, self.n1) };
        if x < 255 {
            x += 1;
        } else if y >= 2 {
            y = (y + 1) / 2;
            x = 129;
        }
        if bit {
            self.n1 = x;
            self.n0 = y;
        } else {
            self.n0 = x;
            self.n1 = y;
        }
    }
}

/// Sections 5.1 to 5.6: the encoder, a shift at a time, its carry done on
/// the bytes already out.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct ModelEnc {
    pub range: u32,
    pub bottom: u32,
    pub bit_count: u32,
    pub out: Vec<u8>,
    /// the longest run of 0xff bytes a carry has gone back through
    pub max_ff_run_carried: usize,
}

impl ModelEnc {
    pub fn new() -> Self {
        let mut e = Self { range: 255, bottom: 0, bit_count: 24, out: Vec::new(), max_ff_run_carried: 0 };
        e.put(false, &mut ModelCtx { n0: 1, n1: 1 }); // the marker
        e
    }

    /// Section 5.3.
    pub fn step(&mut self, bit: bool, split: u32) {
        assert!((1..self.range).contains(&split));
        if bit {
            self.bottom += split;
            self.range -= split;
        } else {
            self.range = split;
        }
        while self.range < 128 {
            self.range *= 2;
            if self.bottom & 0x8000_0000 != 0 {
                let run = self.out.iter().rev().take_while(|&&b| b == 0xff).count();
                self.max_ff_run_carried = self.max_ff_run_carried.max(run);
                add_one(&mut self.out);
            }
            self.bottom = self.bottom.wrapping_mul(2);
            self.bit_count -= 1;
            if self.bit_count == 0 {
                self.out.push((self.bottom >> 24) as u8);
                self.bottom %= 1 << 24;
                self.bit_count = 8;
            }
        }
    }

    pub fn put(&mut self, bit: bool, ctx: &mut ModelCtx) {
        let split = 1 + (self.range - 1) * ctx.p0() / 256;
        self.step(bit, split);
        ctx.update(bit);
    }

    pub fn bypass(&mut self, bit: bool) {
        self.step(bit, 1 + self.range / 2);
    }

    /// Section 5.6.
    pub fn finish(mut self) -> Vec<u8> {
        while self.bottom != 0 {
            self.bypass(false);
        }
        self.out
    }

    /// The "equivalent formulation" of 5.6: the RFC's flush, then the
    /// trailing zero bytes the padding would not have written.
    pub fn finish_by_flush(mut self) -> Vec<u8> {
        let e0 = self.out.len();
        let c = self.bit_count;
        let mut v = self.bottom;
        if v & (1 << (32 - c)) != 0 {
            add_one(&mut self.out);
        }
        v <<= c & 7;
        for _ in 0..(c >> 3) {
            v <<= 8;
        }
        for _ in 0..4 {
            self.out.push((v >> 24) as u8);
            v <<= 8;
        }
        while self.out.len() > e0 && self.out.last() == Some(&0) {
            self.out.pop();
        }
        self.out
    }
}

/// `add_one_to_output` of the RFC, here the one the model has.
fn add_one(out: &mut Vec<u8>) {
    let mut i = out.len();
    loop {
        i -= 1;
        if out[i] == 255 {
            out[i] = 0;
        } else {
            out[i] += 1;
            break;
        }
    }
}

// ---- scripts ----

#[derive(Clone, Copy, Debug)]
pub(super) enum Op {
    /// a bit under context `1` of the bank
    Put(bool, usize),
    Bypass(bool),
}

pub(super) fn encode(ops: &[Op]) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut w = VP8Writer::new(&mut buf).unwrap();
    let mut ctx = [VP8Context::default(); 8];
    for &op in ops {
        match op {
            Op::Put(b, i) => w.put(b, &mut ctx[i]).unwrap(),
            Op::Bypass(b) => w.put_bypass(b),
        }
    }
    w.finish().unwrap();
    buf
}

fn model_encode(ops: &[Op]) -> (Vec<u8>, Vec<u8>) {
    let mut e = ModelEnc::new();
    let mut ctx = [ModelCtx::new(); 8];
    for &op in ops {
        match op {
            Op::Put(b, i) => e.put(b, &mut ctx[i]),
            Op::Bypass(b) => e.bypass(b),
        }
    }
    (e.clone().finish(), e.finish_by_flush())
}

/// What the reader says to a script: the bits, in order.
pub(super) fn decode(stream: &[u8], ops: &[Op]) -> Vec<bool> {
    let mut r = VP8Reader::new(Cursor::new(stream)).unwrap();
    let mut ctx = [VP8Context::default(); 8];
    ops.iter()
        .map(|&op| match op {
            Op::Put(_, i) => r.get(&mut ctx[i]).unwrap(),
            Op::Bypass(_) => r.get_bypass().unwrap(),
        })
        .collect()
}

fn puts(bits: &[bool]) -> Vec<Op> {
    bits.iter().map(|&b| Op::Put(b, 0)).collect()
}

// ---- section 2: the context ----

fn after(history: &[(bool, usize)]) -> VP8Context {
    let mut c = VP8Context::default();
    for &(bit, n) in history {
        for _ in 0..n {
            c.update(bit);
        }
    }
    c
}

fn counts(c: VP8Context) -> (u32, u32, u32) {
    (c.counts().0, c.counts().1, c.prob())
}

#[test]
fn context_transitions_of_the_specification() {
    let (f, t) = (false, true);
    let cases: [(&[(bool, usize)], (u32, u32, u32)); 16] = [
        (&[], (1, 1, 128)),
        (&[(f, 1)], (2, 1, 170)),
        (&[(t, 1)], (1, 2, 85)),
        (&[(f, 254)], (255, 1, 255)),
        (&[(f, 255)], (255, 1, 255)),
        (&[(f, 300)], (255, 1, 255)),
        (&[(f, 300), (t, 1)], (255, 2, 254)),
        (&[(f, 300), (t, 1), (f, 1)], (129, 1, 254)),
        (&[(f, 300), (t, 2), (f, 1)], (129, 2, 252)),
        (&[(t, 254)], (1, 255, 1)),
        (&[(t, 300)], (1, 255, 1)),
        (&[(t, 300), (f, 1)], (2, 255, 1)),
        (&[(t, 300), (f, 1), (t, 1)], (1, 129, 1)),
        (&[(f, 254), (t, 254), (f, 1)], (129, 128, 128)),
        (&[(f, 254), (t, 254), (t, 1)], (128, 129, 127)),
        (&[(t, 99), (f, 254), (f, 1)], (129, 50, 184)),
    ];
    for (history, want) in cases {
        assert_eq!(counts(after(history)), want, "{history:?}");
    }
    assert_eq!(counts(after(&[(f, 99), (t, 254), (t, 1)])), (50, 129, 71));
}

#[test]
fn context_matches_its_integer_model_on_runs_of_every_length() {
    let mut rng = Rng(11);
    for _ in 0..200 * scale() {
        let (mut a, mut b) = (VP8Context::default(), ModelCtx::new());
        let mut bit = rng.chance(2);
        for _ in 0..rng.below(40) {
            // runs from 1 to 700: past the ceiling both ways, and the rescale
            let longest = 1 << rng.below(10);
            for _ in 0..1 + rng.below(longest) {
                assert_eq!(a.counts(), (b.n0, b.n1));
                assert_eq!(a.prob(), b.p0());
                a.update(bit);
                b.update(bit);
            }
            bit = !bit;
        }
    }
}

#[test]
fn probability_of_every_state_is_in_range_and_is_the_formula() {
    for n0 in 1..=255u32 {
        for n1 in 1..=255u32 {
            let c = VP8Context::with_counts(n0, n1);
            let p = c.prob();
            assert_eq!(p, 256 * n0 / (n0 + n1));
            assert!((1..=255).contains(&p), "({n0}, {n1}) -> {p}");
        }
    }
}

// ---- sections 9.2 and 9.3: the encoder's vectors ----

fn ones(n: usize) -> Vec<Op> {
    vec![Op::Put(true, 0); n]
}

fn bypass(bit: bool, n: usize) -> Vec<Op> {
    vec![Op::Bypass(bit); n]
}

#[test]
fn encoder_vectors() {
    use Op::{Bypass, Put};
    let e11 = [Put(true, 0), Put(false, 1), Put(true, 0), Put(true, 1), Put(false, 0)];
    let cases: Vec<(&str, Vec<Op>, &str)> = vec![
        ("E1", vec![], ""),
        ("E2", vec![Put(false, 0)], ""),
        ("E3", ones(1), "40"),
        ("E4", vec![Bypass(true)], "41"),
        ("E5", vec![Bypass(false)], ""),
        ("E6", ones(2), "55 80"),
        ("E7", ones(3), "60 40"),
        ("E8", ones(4), "66 a0"),
        ("E9", ones(8), "72"),
        ("E10", vec![Put(true, 0), Put(false, 0), Put(true, 0)], "4a c0"),
        ("E11", e11.to_vec(), "58 e0"),
        ("E12/16", ones(16), "78 a8"),
        ("E12/17", ones(17), "79 10"),
        ("E12/40", ones(40), "7d 08"),
        ("E12/100", ones(100), "7e ee"),
        ("E13/2", bypass(true, 2), "60 c0"),
        ("E13/7", bypass(true, 7), "7f 10"),
        ("E13/8", bypass(true, 8), "7f 89"),
        ("E13/9", bypass(true, 9), "7f c5"),
        ("E14/16", bypass(true, 16), "7f ff 91"),
        ("E14/24", bypass(true, 24), "7f ff ff 99"),
        ("E14/40", bypass(true, 40), "7f ff ff ff ff a9"),
        ("E15/1", bypass(false, 1), ""),
        ("E15/23", bypass(false, 23), ""),
        ("E15/24", bypass(false, 24), "00"),
        ("E15/25", bypass(false, 25), "00"),
        ("E15/32", bypass(false, 32), "00 00"),
        ("E15/100", bypass(false, 100), "00 00 00 00 00 00 00 00 00 00"),
        ("E16", vec![Put(false, 0); 17], ""),
        ("E17", vec![Put(false, 0); 300], ""),
    ];
    for (id, ops, want) in cases {
        assert_eq!(encode(&ops), hex(want), "{id}");
        let (spec, flush) = model_encode(&ops);
        assert_eq!(spec, hex(want), "{id}: model");
        assert_eq!(flush, hex(want), "{id}: model, by the flush");
    }
}

#[test]
fn encoder_traces_of_the_specification() {
    // E9, step by step: (range after, shifts, bottom after)
    let mut w = VP8Writer::new(Vec::new()).unwrap();
    assert_eq!((w.range, w.bottom, w.bit_count), (128, 0, 24), "after the marker");
    let mut c = VP8Context::default();
    let trace = [
        (128, 0x80), (170, 0x156), (254, 0x302), (203, 0x335),
        (169, 0x357), (145, 0x36f), (252, 0x704), (224, 0x720),
    ];
    for (i, (range, bottom)) in trace.into_iter().enumerate() {
        w.put(true, &mut c).unwrap();
        assert_eq!((w.range, w.bottom), (range, bottom), "E9 step {}", i + 1);
    }
    assert_eq!(c.counts(), (1, 9));
}

// ---- section 9.4: the decoder's vectors ----

#[test]
fn decoder_vectors() {
    use Op::{Bypass, Put};
    let t = true;
    let f = false;
    let cases: Vec<(&str, &str, Vec<Op>, Vec<bool>)> = vec![
        ("D1", "", vec![Put(f, 0), Put(f, 1), Bypass(f), Put(f, 0), Bypass(f)], vec![f; 5]),
        ("D2", "40", puts(&[f; 4]), vec![t, f, f, f]),
        ("D3", "55 80", puts(&[f; 3]), vec![t, t, f]),
        ("D4", "4a c0", puts(&[f; 4]), vec![t, f, t, f]),
        ("D5", "72", puts(&[f; 10]), vec![t, t, t, t, t, t, t, t, f, f]),
        (
            "D6",
            "58 e0",
            [0, 1, 0, 1, 0, 1, 1].iter().map(|&i| Op::Put(f, i)).collect(),
            vec![t, f, t, t, f, f, f],
        ),
        ("D7", "41", bypass(f, 3), vec![t, f, f]),
        ("D8", "7f 89", bypass(f, 10), vec![t, t, t, t, t, t, t, t, f, f]),
        ("D9", "7f ff ff 99", bypass(f, 26), [vec![t; 24], vec![f; 2]].concat()),
        ("D10", "00", puts(&[f; 6]), vec![f; 6]),
    ];
    for (id, stream, ops, want) in cases {
        assert_eq!(decode(&hex(stream), &ops), want, "{id}");
    }
}

// ---- section 9.5: the helpers ----

type Bank = [VP8Context; 8];

fn state(bank: &Bank) -> Vec<(u8, u8)> {
    bank.iter().map(|c| (c.counts().0 as u8, c.counts().1 as u8)).collect()
}

fn bank_of(states: &[(u8, u8)]) -> Vec<(u8, u8)> {
    let mut v = vec![(1, 1); 8];
    v[..states.len()].copy_from_slice(states);
    v
}

#[test]
fn helper_vectors() {
    // (the calls, the stream, the unary bank, the literal bank)
    type Calls = Box<dyn Fn(&mut VP8Writer<&mut Vec<u8>>, &mut Bank, &mut Bank)>;
    let cases: Vec<(&str, Calls, &str, Vec<(u8, u8)>, Vec<(u8, u8)>)> = vec![
        ("H1", Box::new(|w, u, _| w.put_unary_encoded(0, u).unwrap()), "", bank_of(&[(2, 1)]), bank_of(&[])),
        ("H2", Box::new(|w, u, _| w.put_unary_encoded(1, u).unwrap()), "40", bank_of(&[(1, 2), (2, 1)]), bank_of(&[])),
        (
            "H3",
            Box::new(|w, u, _| w.put_unary_encoded(3, u).unwrap()),
            "70",
            bank_of(&[(1, 2), (1, 2), (1, 2), (2, 1)]),
            bank_of(&[]),
        ),
        (
            "H4",
            Box::new(|w, u, _| w.put_unary_encoded(10, u).unwrap()),
            "7f c0 80",
            bank_of(&[(1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (2, 4)]),
            bank_of(&[]),
        ),
        (
            "H5",
            Box::new(|w, _, b| w.put_n_bits(5, 3, b).unwrap()),
            "50",
            bank_of(&[]),
            bank_of(&[(1, 2), (2, 1), (1, 2)]),
        ),
        (
            "H6",
            Box::new(|w, _, b| {
                w.put_n_bits(5, 3, b).unwrap();
                w.put_n_bits(5, 3, b).unwrap();
            }),
            "57 b8",
            bank_of(&[]),
            bank_of(&[(1, 3), (3, 1), (1, 3)]),
        ),
        (
            "H7",
            Box::new(|w, u, b| {
                w.put_unary_encoded(3, u).unwrap();
                w.put_n_bits(1, 2, b).unwrap();
            }),
            "72",
            bank_of(&[(1, 2), (1, 2), (1, 2), (2, 1)]),
            bank_of(&[(1, 2), (2, 1)]),
        ),
        (
            "H8",
            Box::new(|w, u, b| {
                w.put_unary_encoded(12, u).unwrap();
                w.put_n_bits(100, 6, b).unwrap();
            }),
            "7f d9 20",
            bank_of(&[(1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (2, 6)]),
            bank_of(&[(2, 1), (2, 1), (1, 2), (2, 1), (2, 1), (1, 2)]),
        ),
    ];
    for (id, calls, stream, unary, literal) in cases {
        let mut buf = Vec::new();
        let mut w = VP8Writer::new(&mut buf).unwrap();
        let (mut u, mut b) = ([VP8Context::default(); 8], [VP8Context::default(); 8]);
        calls(&mut w, &mut u, &mut b);
        w.finish().unwrap();
        assert_eq!(buf, hex(stream), "{id}");
        assert_eq!(state(&u), unary, "{id}: unary contexts");
        assert_eq!(state(&b), literal, "{id}: literal contexts");
    }
}

#[test]
fn helper_decoding_vectors() {
    let fresh = || ([VP8Context::default(); 8], [VP8Context::default(); 8]);
    let reader = |s: &str| VP8Reader::new(Cursor::new(hex(s))).unwrap();

    let (mut u, _) = fresh();
    let mut r = reader("70");
    assert_eq!((r.get_unary_encoded(&mut u).unwrap(), r.get_unary_encoded(&mut u).unwrap()), (3, 0));

    let (mut u, _) = fresh();
    let mut r = reader("7f c0 80");
    assert_eq!((r.get_unary_encoded(&mut u).unwrap(), r.get_unary_encoded(&mut u).unwrap()), (10, 0));

    let (_, mut b) = fresh();
    let mut r = reader("50");
    assert_eq!((r.get_n_bits(3, &mut b).unwrap(), r.get_n_bits(3, &mut b).unwrap()), (5, 0));

    let (mut u, mut b) = fresh();
    let mut r = reader("72");
    assert_eq!((r.get_unary_encoded(&mut u).unwrap(), r.get_n_bits(2, &mut b).unwrap()), (3, 1));

    let (mut u, mut b) = fresh();
    let mut r = reader("7f d9 20");
    assert_eq!((r.get_unary_encoded(&mut u).unwrap(), r.get_n_bits(6, &mut b).unwrap()), (12, 36));
}

// ---- section 9.6: the pseudo-random workloads ----

const THRESHOLDS: [u64; 8] = [8388608, 2097152, 14680064, 524288, 16252928, 4194304, 16384, 16760832];

/// The workload of the specification: decision `i` is a bit under one of
/// eight contexts, the context and the bit from a 64-bit LCG.
pub(super) fn workload_calls(seed: u64, n: usize) -> impl Iterator<Item = (usize, bool)> {
    let mut x = seed;
    (0..n).map(move |_| {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let k = (x >> 61) as usize;
        let u = (x >> 32) & 0xff_ffff;
        (k, u < THRESHOLDS[k])
    })
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100000001b3))
}

fn encode_workload(seed: u64, n: usize) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut w = VP8Writer::new(&mut buf).unwrap();
    let mut ctx = [VP8Context::default(); 8];
    for (k, bit) in workload_calls(seed, n) {
        w.put(bit, &mut ctx[k]).unwrap();
    }
    w.finish().unwrap();
    buf
}

fn check_workload(seed: u64, n: usize, len: usize, hash: u64, first: &str, last: &str) {
    let stream = encode_workload(seed, n);
    assert_eq!(stream.len(), len, "seed {seed}, {n} decisions: length");
    assert_eq!(fnv1a(&stream), hash, "seed {seed}, {n} decisions: hash");
    assert_eq!(stream[..stream.len().min(16)], hex(first)[..], "first bytes");
    assert_eq!(stream[stream.len().saturating_sub(16)..], hex(last)[..], "last bytes");
    // and back
    let mut r = VP8Reader::new(Cursor::new(&stream)).unwrap();
    let mut ctx = [VP8Context::default(); 8];
    for (i, (k, bit)) in workload_calls(seed, n).enumerate() {
        assert_eq!(r.get(&mut ctx[k]).unwrap(), bit, "decision {i}");
    }
}

#[test]
fn workloads_of_the_specification() {
    check_workload(1, 1, 0, 0xcbf29ce484222325, "", "");
    check_workload(1, 8, 2, 0x07e3c607b4a804f6, "2bf8", "2bf8");
    check_workload(1, 24, 4, 0xa12c4a961656eb34, "2c0d6680", "2c0d6680");
    check_workload(2, 50, 6, 0xf857e4d3e7a57fd1, "2d203bafdba0", "2d203bafdba0");
    check_workload(1, 1000, 52, 0xe07370d2c71c2ec7, "2c0d791417823eaa13313d8c6af01ff4", "1c7fce6d39423e2341f7866c5c92e8b1");
    check_workload(1, 100000, 5184, 0x064bb862c43a5f85, "2c0d791417823eaa13313d8c6af01ff4", "91d933092cc654e2b7af2a7499ebee30");
}

#[test]
#[cfg_attr(debug_assertions, ignore = "slow without optimisation")]
fn workload_of_three_million_decisions() {
    check_workload(1, 3_000_000, 156323, 0x00b09cf39c9506e2, "2c0d791417823eaa13313d8c6af01ff4", "ccc5aa08787b4a16a4f71dcba01525b0");
}

/// 6 carries each into bytes already out, one across a 0xff byte (decision
/// 10,385,777 of the first, 40,037,187 of the second).
#[test]
#[cfg_attr(debug_assertions, ignore = "slow without optimisation")]
fn workloads_with_carries_across_ff() {
    check_workload(18, 10_400_000, 543050, 0x3fad1bdaffa55da9, "4d546e4bffe0856e0ba6f9953219e73a", "573f95d879700c56fc07f19395bf05c0");
    check_workload(13, 40_100_000, 2094580, 0x193091ab722539ce, "427e74cf84891a62341a78dd1fe8b553", "0ba60adba5301b111105211069a5fa80");
}

// ---- the model, and a carry made by hand ----

fn random_ops(rng: &mut Rng, n: usize) -> Vec<Op> {
    let skew = [2, 3, 8, 40, 500][rng.below(5) as usize];
    (0..n)
        .map(|_| {
            if rng.chance(12) {
                Op::Bypass(rng.chance(2))
            } else {
                // a skewed bit, mostly under few contexts, so they saturate
                let k = if rng.chance(3) { rng.below(8) as usize } else { 0 };
                Op::Put(!rng.chance(skew), k)
            }
        })
        .collect()
}

#[test]
fn writer_matches_the_model_and_its_flush_formulation() {
    let mut rng = Rng(5);
    for i in 0..300 * scale() {
        let n = if i % 50 == 0 { 5000 } else { rng.below(120) as usize };
        let ops = random_ops(&mut rng, n);
        let (spec, flush) = model_encode(&ops);
        assert_eq!(encode(&ops), spec, "case {i}");
        assert_eq!(flush, spec, "case {i}: the two formulations of the end");
    }
}

#[test]
fn round_trip_of_scripts_with_zero_bytes_added_and_taken() {
    let mut rng = Rng(6);
    for i in 0..300 * scale() {
        let n = rng.below(300) as usize;
        let ops = random_ops(&mut rng, n);
        let stream = encode(&ops);
        let want: Vec<bool> = ops
            .iter()
            .map(|&o| match o {
                Op::Put(b, _) | Op::Bypass(b) => b,
            })
            .collect();
        assert_eq!(decode(&stream, &ops), want, "case {i}");
        // zero bytes at the end change nothing
        let mut longer = stream.clone();
        longer.extend_from_slice(&[0; 9]);
        assert_eq!(decode(&longer, &ops), want, "case {i}: zeros added");
        // the first byte is below 0x80
        assert!(stream.first().is_none_or(|&b| b < 0x80), "case {i}");
    }
}

#[test]
fn carry_on_bytes() {
    let cases: [(&[u8], &[u8]); 7] = [
        (&[0x12, 0xff, 0xff, 0xff], &[0x13, 0, 0, 0]),
        (&[0x12, 0xff], &[0x13, 0]),
        (&[0x12], &[0x13]),
        (&[0x00, 0xff, 0xff, 0xff, 0xff, 0xff], &[0x01, 0, 0, 0, 0, 0]),
        (&[0x7f, 0xff, 0x12, 0xff], &[0x7f, 0xff, 0x13, 0x00]),
        (&[0x7e, 0xfe, 0xff, 0xff], &[0x7e, 0xff, 0, 0]),
        (&[], &[]),
    ];
    for (before, after) in cases {
        let mut v = before.to_vec();
        carry(&mut v);
        assert_eq!(v, after, "{before:02x?}");
        // and the same as the model's
        if !before.is_empty() {
            let mut m = before.to_vec();
            add_one(&mut m);
            assert_eq!(m, v);
        }
    }
}

/// A valid register at a random point in the byte cycle: the carry slot set
/// or not, the bits under it all ones often, the bytes already out ending in
/// a run of 0xff.
fn random_state(rng: &mut Rng) -> ModelEnc {
    loop {
        let bit_count = if rng.chance(4) { 1 + rng.below(24) } else { 1 + rng.below(8) } as u32;
        let pending_bits = 24 - bit_count;
        let pending = match rng.below(3) {
            0 => (1u64 << pending_bits) - 1,
            1 => (1u64 << pending_bits) - 1 - rng.below(3).min((1 << pending_bits) - 1),
            _ => rng.next() & ((1u64 << pending_bits) - 1),
        };
        let slot = u64::from(rng.chance(3));
        let active = rng.below(256);
        let bottom = (slot << (32 - bit_count)) | (pending << 8) | active;
        // a decision adds at most 254: it must not carry out of the slot
        if bottom + 254 >= 1 << (33 - bit_count).min(33) {
            continue;
        }
        let mut out: Vec<u8> = (0..rng.below(12)).map(|_| rng.next() as u8).collect();
        let run = rng.below(9) as usize;
        out.extend(std::iter::repeat_n(0xff, run));
        if out.first().is_none_or(|&b| b >= 0x80) {
            out.insert(0, rng.below(0x80) as u8);
        }
        return ModelEnc { range: 128 + rng.below(128) as u32, bottom: bottom as u32, bit_count, out, max_ff_run_carried: 0 };
    }
}

fn writer_in(state: &ModelEnc) -> VP8Writer<Vec<u8>> {
    VP8Writer {
        sink: Vec::new(),
        out: state.out.clone(),
        bottom: state.bottom,
        range: state.range,
        bit_count: state.bit_count,
    }
}

fn same(w: &VP8Writer<Vec<u8>>, m: &ModelEnc) -> bool {
    (w.range, w.bottom, w.bit_count, &w.out) == (m.range, m.bottom, m.bit_count, &m.out)
}

/// The writer's multi-shift step, in every state a register can be in,
/// against the model that does the shifts one at a time: the same bytes,
/// the same carries through runs of 0xff, the same register after every
/// decision, and the same end.
#[test]
fn writer_matches_the_model_state_by_state() {
    let mut rng = Rng(8);
    let mut carries_through_ff = [0usize; 10];
    for case in 0..2000 * scale() {
        let mut m = random_state(&mut rng);
        let mut w = writer_in(&m);
        for step in 0..rng.below(40) {
            let bit = rng.chance(2);
            let split = if rng.chance(2) {
                1 + rng.below(u64::from(m.range - 1)) as u32
            } else {
                // the extremes of a probability
                [1, m.range - 1, 1 + (m.range - 1) / 2][rng.below(3) as usize]
            };
            // a step that would carry out of the slot is not one a coder makes
            if bit && u64::from(m.bottom) + u64::from(split) >= 1 << (33 - m.bit_count) {
                continue;
            }
            let before = m.out.clone();
            m.step(bit, split);
            w.code(bit, split);
            assert!(
                same(&w, &m),
                "case {case} step {step}: {:?} against {m:?}",
                (w.range, w.bottom, w.bit_count, &w.out)
            );
            // how many bytes a carry changed: itself, and the 0xff bytes
            // it went back through
            let common = before.iter().zip(&m.out).take_while(|(a, b)| a == b).count();
            carries_through_ff[(before.len() - common).min(9)] += 1;
        }
        w.finish().unwrap();
        assert_eq!(w.sink, m.finish(), "case {case}: the end");
    }
    // the generator reaches carries across runs of 0xff of every length up to eight
    for (n, &seen) in carries_through_ff.iter().enumerate().skip(1) {
        assert!(seen > 0, "no carry changed {n} bytes: {carries_through_ff:?}");
    }
}

/// `out` ends in 0x3f 0xff 0xff 0xff 0xff; a one-bit slot is set and all
/// below it is zero: the next byte out is 0x00, and the carry makes the
/// run 0x40 0x00 0x00 0x00 0x00.
#[test]
fn carry_through_four_ff_bytes_by_hand() {
    let mut w = VP8Writer {
        sink: Vec::new(),
        out: vec![0x12, 0x3f, 0xff, 0xff, 0xff, 0xff],
        bottom: 0x8000_005a,
        range: 128,
        bit_count: 1,
    };
    let model = {
        let mut m = ModelEnc { range: 128, bottom: 0x8000_005a, bit_count: 1, out: w.out.clone(), max_ff_run_carried: 0 };
        m.step(false, 1);
        assert_eq!(m.max_ff_run_carried, 4);
        m
    };
    // a zero at the extreme: range 128 -> 1, seven shifts; the first
    // brings the byte out, and with it the carry
    w.code(false, 1);
    assert_eq!(w.out, [0x12, 0x40, 0, 0, 0, 0, 0x00]);
    assert_eq!((w.range, w.bit_count), (128, 2));
    assert_eq!(w.bottom, (0x5a << 1 & 0xff_ffff) << 6);
    assert!(same(&w, &model));
    w.finish().unwrap();
    // what is left, 0x5a shifted 7 places, comes out three bytes on: 0x00 0x00 0xb4
    assert_eq!(w.sink, [0x12, 0x40, 0, 0, 0, 0, 0x00, 0x00, 0x00, 0xb4]);
    assert_eq!(w.sink, model.finish());
}

// ---- the reader: hardening, and what it does with every kind of source ----

/// A source that hands out `n` bytes at a time, and interrupts in between.
struct Stuttering<'a> {
    data: &'a [u8],
    n: usize,
    toggle: bool,
}

impl Read for Stuttering<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.toggle = !self.toggle;
        if self.toggle {
            return Err(io::ErrorKind::Interrupted.into());
        }
        let k = self.n.min(buf.len()).min(self.data.len());
        buf[..k].copy_from_slice(&self.data[..k]);
        self.data = &self.data[k..];
        Ok(k)
    }
}

/// A source that fails after `ok` bytes.
struct Failing<'a> {
    data: &'a [u8],
    ok: usize,
}

impl Read for Failing<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.ok == 0 {
            return Err(io::Error::other("the source failed"));
        }
        let k = self.ok.min(buf.len()).min(self.data.len());
        buf[..k].copy_from_slice(&self.data[..k]);
        self.data = &self.data[k..];
        self.ok -= k;
        Ok(k)
    }
}

#[test]
fn reader_gives_the_same_decisions_from_any_kind_of_source() {
    let mut rng = Rng(9);
    for i in 0..100 * scale() {
        let n = 50 + rng.below(2000) as usize;
        let ops = random_ops(&mut rng, n);
        let stream = encode(&ops);
        let want = decode(&stream, &ops);
        for n in [1, 2, 3, 7] {
            let mut r = VP8Reader::new(Stuttering { data: &stream, n, toggle: false }).unwrap();
            let mut ctx = [VP8Context::default(); 8];
            let got: Vec<bool> = ops
                .iter()
                .map(|&op| match op {
                    Op::Put(_, k) => r.get(&mut ctx[k]).unwrap(),
                    Op::Bypass(_) => r.get_bypass().unwrap(),
                })
                .collect();
            assert_eq!(got, want, "case {i}, {n} bytes at a time");
        }
    }
}

#[test]
fn reader_returns_its_source_errors_and_never_panics_on_them() {
    // the constructor reads ahead: an error there is the constructor's
    assert!(VP8Reader::new(Failing { data: &[1, 2, 3, 4, 5, 6, 7, 8, 9], ok: 0 }).is_err());
    // a stream longer than the read-ahead whose source fails later
    let stream: Vec<u8> = (0..200).map(|i| (i * 7 % 120) as u8).collect();
    let mut r = VP8Reader::new(Failing { data: &stream, ok: 30 }).unwrap();
    let mut ctx = VP8Context::default();
    let mut got_error = false;
    for _ in 0..100_000 {
        if r.get(&mut ctx).is_err() {
            got_error = true;
            break;
        }
    }
    assert!(got_error);
}

#[test]
fn reader_past_the_end_reads_zeros_for_ever() {
    let mut ctx = [VP8Context::default(); 3];
    for stream in [vec![], vec![0u8], vec![0u8; 5]] {
        let mut r = VP8Reader::new(Cursor::new(stream)).unwrap();
        for i in 0..1_000_000 {
            assert!(!r.get(&mut ctx[i % 3]).unwrap());
        }
        assert!(!r.get_bypass().unwrap());
    }
    // a stream's own end: the last decisions of E9 and then zeros
    let mut r = VP8Reader::new(Cursor::new(vec![0x72])).unwrap();
    let mut c = VP8Context::default();
    let got: Vec<bool> = (0..64).map(|_| r.get(&mut c).unwrap()).collect();
    assert_eq!(got[..8], [true; 8]);
    assert!(got[8..].iter().all(|&b| !b));
}

/// Counts the decisions a reader is asked for, so a test can see how many a
/// call made.
struct Counting<R> {
    inner: R,
    gets: usize,
}

impl<R: CabacReader<VP8Context>> CabacReader<VP8Context> for Counting<R> {
    fn get(&mut self, ctx: &mut VP8Context) -> io::Result<bool> {
        self.gets += 1;
        self.inner.get(ctx)
    }
}

#[test]
fn the_single_byte_ff_does_not_hang_a_unary_read() {
    // the stream the crate this replaces looped on: every decision is a one
    for stream in [vec![0xffu8], vec![0xff, 0, 0, 0], vec![0xff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] {
        let mut r = Counting { inner: VP8Reader::new(Cursor::new(stream.clone())).unwrap(), gets: 0 };
        let mut ctx = [VP8Context::default(); 32];
        let e = r.get_unary_encoded(&mut ctx).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::InvalidData, "{stream:02x?}");
        assert_eq!(r.gets, MAX_RUN);
    }
    // streams of ones: whatever they decode to, the read returns
    for len in 0..70 {
        let mut r = Counting { inner: VP8Reader::new(Cursor::new(vec![0xffu8; len])).unwrap(), gets: 0 };
        let mut ctx = [VP8Context::default(); 32];
        let _ = r.get_unary_encoded(&mut ctx);
        assert!(r.gets <= MAX_RUN);
    }
}

#[test]
fn the_limits_of_unary_codes_and_literals() {
    let mut u = [VP8Context::default(); 32];
    let mut b = [VP8Context::default(); 32];
    // the longest code that is accepted is read back
    let mut buf = Vec::new();
    let mut w = VP8Writer::new(&mut buf).unwrap();
    w.put_unary_encoded(MAX_RUN - 1, &mut u).unwrap();
    assert_eq!(w.put_unary_encoded(MAX_RUN, &mut u).unwrap_err().kind(), io::ErrorKind::InvalidInput);
    assert_eq!(w.put_n_bits(0, MAX_RUN + 1, &mut b).unwrap_err().kind(), io::ErrorKind::InvalidInput);
    // a literal wider than 64 bits has zeros above the 64th
    w.put_n_bits(u64::MAX, 70, &mut b).unwrap();
    w.put_n_bits(5, 0, &mut b).unwrap(); // nothing
    w.finish().unwrap();
    let mut r = VP8Reader::new(Cursor::new(&buf)).unwrap();
    let (mut u, mut b) = ([VP8Context::default(); 32], [VP8Context::default(); 32]);
    assert_eq!(r.get_unary_encoded(&mut u).unwrap(), MAX_RUN - 1);
    assert_eq!(r.get_n_bits(70, &mut b).unwrap(), u64::MAX);
    assert_eq!(r.get_n_bits(0, &mut b).unwrap(), 0);
    assert_eq!(r.get_n_bits(MAX_RUN + 1, &mut b).unwrap_err().kind(), io::ErrorKind::InvalidInput);
}

#[test]
fn contexts_beyond_the_last_are_the_last() {
    // a unary code longer than the array reuses the last context, and a
    // literal wider than the array does too: round trip, and the counts
    let mut buf = Vec::new();
    let mut w = VP8Writer::new(&mut buf).unwrap();
    let mut u = [VP8Context::default(); 4];
    let mut b = [VP8Context::default(); 4];
    for v in [0, 1, 5, 12, 40] {
        w.put_unary_encoded(v, &mut u).unwrap();
        w.put_n_bits(v as u64 * 3 + 1, 9, &mut b).unwrap();
    }
    w.finish().unwrap();
    let mut r = VP8Reader::new(Cursor::new(&buf)).unwrap();
    let (mut u2, mut b2) = ([VP8Context::default(); 4], [VP8Context::default(); 4]);
    for v in [0usize, 1, 5, 12, 40] {
        assert_eq!(r.get_unary_encoded(&mut u2).unwrap(), v);
        assert_eq!(r.get_n_bits(9, &mut b2).unwrap(), v as u64 * 3 + 1);
    }
    assert_eq!(u, u2);
    assert_eq!(b, b2);
}

/// Any bytes, any calls: every call returns, within its bound, and none
/// panics.
#[test]
fn fuzz_the_reader_with_random_bytes_and_calls() {
    let mut rng = Rng(10);
    let started = std::time::Instant::now();
    let mut errors = 0usize;
    for case in 0..3000 * scale() {
        let len = match rng.below(4) {
            0 => rng.below(4),
            1 => rng.below(64),
            _ => rng.below(600),
        } as usize;
        let mut bytes: Vec<u8> = (0..len)
            .map(|_| match rng.below(6) {
                0 => 0xff,
                1 => 0x00,
                _ => rng.next() as u8,
            })
            .collect();
        if rng.chance(5) && !bytes.is_empty() {
            bytes[0] = 0xff; // the first byte no stream starts with
        }
        let mut r = Counting { inner: VP8Reader::new(Cursor::new(bytes)).unwrap(), gets: 0 };
        let mut ctx = [VP8Context::default(); 32];
        let mut single = VP8Context::default();
        for _ in 0..rng.below(30) {
            let before = r.gets;
            let result = match rng.below(5) {
                0 => r.get(&mut single).map(|_| ()),
                1 => r.inner.get_bypass().map(|_| ()),
                2 => r.get_unary_encoded(&mut ctx).map(|_| ()),
                3 => r.get_n_bits(rng.below(70) as usize, &mut ctx).map(|_| ()),
                _ => r.get_n_bits([0, 1, 33, 64, 65, MAX_RUN, MAX_RUN + 1][rng.below(7) as usize], &mut ctx).map(|_| ()),
            };
            errors += usize::from(result.is_err());
            assert!(r.gets - before <= MAX_RUN, "case {case}");
        }
    }
    assert!(errors > 0, "no call was refused: the inputs are not hostile enough");
    assert!(started.elapsed().as_secs() < 60, "slow: {:?}", started.elapsed());
}

// ---- the writer, the sink, and what the types allow ----

#[test]
fn finish_twice_writes_the_stream_once() {
    let mut buf = Vec::new();
    let mut w = VP8Writer::new(&mut buf).unwrap();
    let mut c = VP8Context::default();
    for _ in 0..40 {
        w.put(true, &mut c).unwrap();
    }
    w.finish().unwrap();
    w.finish().unwrap();
    assert_eq!(buf, hex("7d 08"));
}

#[test]
fn a_sink_that_fails_makes_finish_fail() {
    struct Full;
    impl io::Write for Full {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::other("full"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut w = VP8Writer::new(Full).unwrap();
    let mut c = VP8Context::default();
    for _ in 0..100 {
        w.put(true, &mut c).unwrap();
    }
    assert!(w.finish().is_err());
    // nothing was written before: an empty stream never reaches the sink
    let mut w = VP8Writer::new(Full).unwrap();
    assert!(w.finish().is_ok());
}

/// The writer holds the sink's borrow only for as long as it is used: no
/// `Drop`, so the buffer can be read while the writer is still in scope
/// (preflate's `decompress` does).
#[test]
fn the_borrow_of_the_sink_ends_with_the_last_call() {
    let mut buf = Vec::new();
    let mut w = VP8Writer::new(&mut buf).unwrap();
    let mut c = VP8Context::default();
    w.put(true, &mut c).unwrap();
    w.finish().unwrap();
    assert_eq!(buf, [0x40]); // `w` is still in scope, not used again
}

/// `VP8Writer<W>` and `VP8Reader<R>` each implement the trait for exactly
/// one context type, so a caller that never names it has it inferred.
#[test]
fn the_context_type_is_inferred() {
    fn write<W: CabacWriter<C>, C: Default>(w: &mut W) -> io::Result<()> {
        w.put(true, &mut C::default())
    }
    fn read<R: CabacReader<C>, C: Default>(r: &mut R) -> io::Result<bool> {
        r.get(&mut C::default())
    }
    let mut buf = Vec::new();
    let mut w = VP8Writer::new(&mut buf).unwrap();
    write(&mut w).unwrap();
    w.finish().unwrap();
    let mut r = VP8Reader::new(Cursor::new(&buf)).unwrap();
    assert!(read(&mut r).unwrap());
}

#[test]
fn the_debug_coder_finds_a_context_mix_up() {
    use super::debug::{DebugContext, DebugReader, DebugWriter};
    let mut buf = Vec::new();
    let mut w = DebugWriter::new(&mut buf).unwrap();
    let (mut a, mut b) = (DebugContext::default(), DebugContext::default());
    w.put(true, &mut a).unwrap();
    w.put(false, &mut b).unwrap();
    w.put(true, &mut a).unwrap();
    w.finish().unwrap();
    let mut r = DebugReader::new(Cursor::new(&buf)).unwrap();
    let (mut a, mut b) = (DebugContext::default(), DebugContext::default());
    assert_eq!(
        [r.get(&mut a).unwrap(), r.get(&mut b).unwrap(), r.get(&mut a).unwrap()],
        [true, false, true]
    );
    let swapped = std::panic::catch_unwind(|| {
        let mut r = DebugReader::new(Cursor::new(buf.clone())).unwrap();
        let (mut a, mut b) = (DebugContext::default(), DebugContext::default());
        r.get(&mut a).unwrap();
        r.get(&mut a).unwrap(); // read under the first context what was written under the second
        let _ = &mut b;
    });
    assert!(swapped.is_err());
}
