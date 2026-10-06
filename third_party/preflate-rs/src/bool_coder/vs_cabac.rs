//! The coder against the `cabac` 0.15.0 crate it replaces, used as a black
//! box: its writer and reader are called, nothing else of it is read or
//! copied. Both coders are driven by the same calls, and what is compared is
//! what a stream is made of: the bytes the writers produce, the decisions the
//! readers return, and what both do with input that no writer made.
//!
//! This file is the only place the crate is named. It is a dev-dependency of
//! the crate: the tests link it, no library or binary does.

use std::io::{self, Cursor, Read, Write};

use cabac::vp8::{VP8Context as OldContext, VP8Reader as OldReader, VP8Writer as OldWriter};
use cabac::{CabacReader as _, CabacWriter as _};

use super::tests::{ModelCtx, ModelEnc, Rng};
use super::*;

fn scale() -> usize {
    if cfg!(debug_assertions) { 1 } else { 20 }
}

// ---- scripts: the calls preflate makes, and the others the traits have ----

#[derive(Clone, Copy, Debug)]
enum Call {
    /// a bit under context `k` of eight
    Put(bool, usize),
    /// a bit with no context
    Bypass(bool),
    /// a unary code on one of two arrays of 32 contexts
    Unary(usize, usize),
    /// `n` bits (1 to 64) of a value on one of two arrays of 32
    Bits(u64, usize, usize),
    Unary4(usize),
    Bits4(u64, usize),
    Unary1(usize),
}

/// What a call reads back from a stream a writer made of it.
fn expected(call: Call) -> u64 {
    match call {
        Call::Put(b, _) | Call::Bypass(b) => u64::from(b),
        Call::Unary(v, _) | Call::Unary4(v) | Call::Unary1(v) => v as u64,
        Call::Bits(x, n, _) | Call::Bits4(x, n) => {
            if n >= 64 {
                x
            } else {
                x & ((1u64 << n) - 1)
            }
        }
    }
}

/// Every context a script touches, for either coder's context type.
#[derive(Default)]
struct Banks<C> {
    single: [C; 8],
    unary: [[C; 32]; 2],
    bits: [[C; 32]; 2],
    unary4: [C; 4],
    bits4: [C; 4],
    unary1: [C; 1],
}

type NewW<'a> = VP8Writer<&'a mut Vec<u8>>;
type OldW<'a> = OldWriter<&'a mut Vec<u8>>;

fn write_new(call: Call, w: &mut NewW, b: &mut Banks<VP8Context>) {
    match call {
        Call::Put(bit, k) => w.put(bit, &mut b.single[k]).unwrap(),
        Call::Bypass(bit) => w.put_bypass(bit),
        Call::Unary(v, a) => w.put_unary_encoded(v, &mut b.unary[a]).unwrap(),
        Call::Bits(x, n, a) => w.put_n_bits(x, n, &mut b.bits[a]).unwrap(),
        Call::Unary4(v) => w.put_unary_encoded(v, &mut b.unary4).unwrap(),
        Call::Bits4(x, n) => w.put_n_bits(x, n, &mut b.bits4).unwrap(),
        Call::Unary1(v) => w.put_unary_encoded(v, &mut b.unary1).unwrap(),
    }
}

fn write_old(call: Call, w: &mut OldW, b: &mut Banks<OldContext>) {
    match call {
        Call::Put(bit, k) => w.put(bit, &mut b.single[k]).unwrap(),
        Call::Bypass(bit) => w.put_bypass(bit).unwrap(),
        Call::Unary(v, a) => w.put_unary_encoded(v, &mut b.unary[a]).unwrap(),
        Call::Bits(x, n, a) => w.put_n_bits(x, n, &mut b.bits[a]).unwrap(),
        Call::Unary4(v) => w.put_unary_encoded(v, &mut b.unary4).unwrap(),
        Call::Bits4(x, n) => w.put_n_bits(x, n, &mut b.bits4).unwrap(),
        Call::Unary1(v) => w.put_unary_encoded(v, &mut b.unary1).unwrap(),
    }
}

fn read_new<R: Read>(call: Call, r: &mut VP8Reader<R>, b: &mut Banks<VP8Context>) -> u64 {
    match call {
        Call::Put(_, k) => u64::from(r.get(&mut b.single[k]).unwrap()),
        Call::Bypass(_) => u64::from(r.get_bypass().unwrap()),
        Call::Unary(_, a) => r.get_unary_encoded(&mut b.unary[a]).unwrap() as u64,
        Call::Bits(_, n, a) => r.get_n_bits(n, &mut b.bits[a]).unwrap(),
        Call::Unary4(_) => r.get_unary_encoded(&mut b.unary4).unwrap() as u64,
        Call::Bits4(_, n) => r.get_n_bits(n, &mut b.bits4).unwrap(),
        Call::Unary1(_) => r.get_unary_encoded(&mut b.unary1).unwrap() as u64,
    }
}

fn read_old<R: Read>(call: Call, r: &mut OldReader<R>, b: &mut Banks<OldContext>) -> u64 {
    match call {
        Call::Put(_, k) => u64::from(r.get(&mut b.single[k]).unwrap()),
        Call::Bypass(_) => u64::from(r.get_bypass().unwrap()),
        Call::Unary(_, a) => r.get_unary_encoded(&mut b.unary[a]).unwrap() as u64,
        Call::Bits(_, n, a) => r.get_n_bits(n, &mut b.bits[a]).unwrap(),
        Call::Unary4(_) => r.get_unary_encoded(&mut b.unary4).unwrap() as u64,
        Call::Bits4(_, n) => r.get_n_bits(n, &mut b.bits4).unwrap(),
        Call::Unary1(_) => r.get_unary_encoded(&mut b.unary1).unwrap() as u64,
    }
}

fn stream_new(calls: &[Call]) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut w = VP8Writer::new(&mut buf).unwrap();
    let mut b = Banks::default();
    for &c in calls {
        write_new(c, &mut w, &mut b);
    }
    w.finish().unwrap();
    buf
}

fn stream_old(calls: &[Call]) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut w = OldWriter::new(&mut buf).unwrap();
    let mut b = Banks::default();
    for &c in calls {
        write_old(c, &mut w, &mut b);
    }
    w.finish().unwrap();
    buf
}

/// What both readers return for `calls` read from `stream`.
fn values_of(stream: &[u8], calls: &[Call]) -> (Vec<u64>, Vec<u64>) {
    let mut r = VP8Reader::new(Cursor::new(stream)).unwrap();
    let mut b = Banks::default();
    let new = calls.iter().map(|&c| read_new(c, &mut r, &mut b)).collect();
    let mut r = OldReader::new(Cursor::new(stream)).unwrap();
    let mut b = Banks::default();
    let old = calls.iter().map(|&c| read_old(c, &mut r, &mut b)).collect();
    (new, old)
}

/// A value of the kind preflate codes: mostly small, sometimes large.
fn correction(rng: &mut Rng) -> u64 {
    match rng.below(12) {
        0..=5 => rng.below(4),
        6..=8 => rng.below(100),
        9 => rng.below(100_000),
        10 => rng.below(1 << 31),
        _ => rng.below(u64::from(u32::MAX)),
    }
}

fn bit_length(x: u64) -> usize {
    (64 - x.leading_zeros()) as usize
}

/// `n` calls drawn the way a stream of corrections draws them, and the
/// others the trait has.
fn script(rng: &mut Rng, n: usize) -> Vec<Call> {
    let biases = [2u64, 3, 5, 16, 100, 700];
    let bias: Vec<u64> = (0..8).map(|_| biases[rng.below(6) as usize]).collect();
    let mut calls = Vec::with_capacity(n + 8);
    while calls.len() < n {
        match rng.below(20) {
            0..=6 => {
                let k = rng.below(8) as usize;
                calls.push(Call::Put(!rng.chance(bias[k]), k));
            }
            7 => calls.push(Call::Bypass(rng.chance(2))),
            8..=14 => {
                // an exp-coded value, as preflate writes it
                let x = correction(rng);
                let (bl, a) = (bit_length(x), rng.below(2) as usize);
                calls.push(Call::Unary(bl, a));
                if bl > 1 {
                    calls.push(Call::Bits(x, bl - 1, a));
                }
            }
            15 | 16 => {
                // a unary code of any length, now and then past the array
                let v = match rng.below(20) {
                    0 => rng.below(3000) as usize,
                    1..=4 => rng.below(60) as usize,
                    _ => rng.below(6) as usize,
                };
                calls.push(match rng.below(4) {
                    0 => Call::Unary4(v),
                    1 => Call::Unary1(v),
                    _ => Call::Unary(v, rng.below(2) as usize),
                });
            }
            _ => {
                let widest = if rng.chance(8) { 64 } else { 12 };
                let w = 1 + rng.below(widest) as usize;
                let x = rng.next() >> rng.below(64);
                calls.push(if rng.chance(4) { Call::Bits4(x, w) } else { Call::Bits(x, w, rng.below(2) as usize) });
            }
        }
    }
    calls
}

// ---- the writers ----

#[test]
fn writers_make_the_same_bytes_from_random_scripts() {
    let mut rng = Rng(21);
    let mut decisions = 0usize;
    for i in 0..300 * scale() {
        let n = if i % 30 == 0 { 20_000 } else { rng.below(400) as usize };
        let calls = script(&mut rng, n);
        let (new, old) = (stream_new(&calls), stream_old(&calls));
        assert_eq!(new, old, "script {i} ({} calls)", calls.len());
        decisions += calls.len();
    }
    assert!(decisions > 10_000);
}

/// A million-decision stream of every kind of call, once.
#[test]
fn writers_make_the_same_bytes_from_a_long_script() {
    let calls = script(&mut Rng(22), 1_000_000 * scale().min(5));
    assert_eq!(stream_new(&calls), stream_old(&calls));
}

#[test]
fn writers_make_the_same_bytes_for_the_workloads_of_the_specification() {
    // the specification's 9.6, with the seeds the table has and many others
    for seed in (1..200u64).chain([18, 13]) {
        let n = if seed == 13 || seed == 18 { 0 } else { 1 + (seed as usize * 7919) % 200_000 };
        let calls: Vec<Call> = super::tests::workload_calls(seed, n)
            .map(|(k, bit)| Call::Put(bit, k))
            .collect();
        assert_eq!(stream_new(&calls), stream_old(&calls), "seed {seed}");
    }
}

/// Every pair of counts a context can have, reached by counting up, then
/// three decisions under it, after a few that put the range somewhere else.
#[test]
fn every_state_of_a_context_codes_the_same() {
    let step = if cfg!(debug_assertions) { 6 } else { 1 };
    let mut counts: Vec<u32> = (1..=255).step_by(step).collect();
    counts.extend([2, 3, 4, 126, 127, 128, 129, 130, 253, 254, 255]);
    counts.sort();
    counts.dedup();
    let shapers: &[&[bool]] = &[&[], &[true], &[false, true, true], &[true, true, false, false, true, false]];
    for &n0 in &counts {
        for &n1 in &counts {
            for shaper in shapers {
                let mut calls: Vec<Call> = shaper.iter().map(|&b| Call::Put(b, 1)).collect();
                calls.extend((1..n0).map(|_| Call::Put(false, 0)));
                calls.extend((1..n1).map(|_| Call::Put(true, 0)));
                calls.extend([Call::Put(true, 0), Call::Put(false, 0), Call::Put(true, 0), Call::Put(true, 0)]);
                assert_eq!(stream_new(&calls), stream_old(&calls), "state ({n0}, {n1}) after {shaper:?}");
            }
        }
    }
}

// ---- the readers ----

#[test]
fn readers_return_what_was_written_and_the_same_as_each_other() {
    let mut rng = Rng(23);
    for i in 0..300 * scale() {
        let n = if i % 30 == 0 { 20_000 } else { rng.below(400) as usize };
        let calls = script(&mut rng, n);
        let stream = stream_old(&calls);
        let want: Vec<u64> = calls.iter().map(|&c| expected(c)).collect();
        let (new, old) = values_of(&stream, &calls);
        assert_eq!(old, want, "script {i}: the old reader");
        assert_eq!(new, want, "script {i}: the new reader");
    }
}

/// Past the end of a stream, and past the end of a stream with its last
/// zero bytes cut or more added: the same values, read as zeros.
#[test]
fn readers_agree_past_the_end() {
    let mut rng = Rng(24);
    for i in 0..200 * scale() {
        let n = rng.below(300) as usize;
        let calls = script(&mut rng, n);
        let stream = stream_new(&calls);
        let extra = 1 + rng.below(200) as usize;
        let more = script(&mut rng, extra);
        let both: Vec<Call> = calls.iter().chain(&more).copied().collect();
        for variant in 0..3 {
            let mut s = stream.clone();
            match variant {
                1 => s.extend_from_slice(&[0; 11]),
                2 => {
                    // a prefix of the stream: not what the writer made, the
                    // reader still reads zeros for the rest
                    s.truncate(s.len() / 2);
                }
                _ => {}
            }
            let (new, old) = values_of(&s, &both);
            assert_eq!(new, old, "script {i}, variant {variant}");
            if variant < 2 {
                assert_eq!(new[..calls.len()], calls.iter().map(|&c| expected(c)).collect::<Vec<_>>()[..]);
            }
        }
    }
}

#[test]
fn readers_read_the_empty_stream_as_zeros() {
    let calls: Vec<Call> = (0..2000)
        .map(|i| match i % 4 {
            0 => Call::Put(false, i % 8),
            1 => Call::Bypass(false),
            2 => Call::Unary(0, 0),
            _ => Call::Bits(0, 1 + i % 20, 1),
        })
        .collect();
    let (new, old) = values_of(&[], &calls);
    assert_eq!(new, old);
    assert!(new.iter().all(|&v| v == 0));
}

/// Input no writer made: any bytes, the first one often 0x80 or more. The
/// decisions are the same (the 16-bit window of the old reader is the
/// 64-bit register of this one with its top bits dropped). The old reader
/// can loop for ever on some of it in a unary read, so a unary read is made
/// of its `get`s, up to this crate's bound.
#[test]
fn readers_agree_on_input_no_writer_made() {
    let mut rng = Rng(25);
    let mut refused = 0usize;
    for case in 0..4000 * scale() {
        let len = match rng.below(4) {
            0 => rng.below(3),
            1 => rng.below(20),
            _ => rng.below(200),
        } as usize;
        let mut bytes: Vec<u8> = (0..len)
            .map(|_| match rng.below(8) {
                0 => 0xff,
                1 => 0x00,
                2 => 0x80,
                _ => rng.next() as u8,
            })
            .collect();
        if let Some(first) = bytes.first_mut() {
            *first = match rng.below(4) {
                0 => 0xff,
                1 => 0x80 + rng.below(0x80) as u8,
                _ => *first,
            };
        }
        let mut new = VP8Reader::new(Cursor::new(bytes.clone())).unwrap();
        let mut old = OldReader::new(Cursor::new(bytes.clone())).unwrap();
        let mut nb: Banks<VP8Context> = Banks::default();
        let mut ob: Banks<OldContext> = Banks::default();
        for step in 0..rng.below(80) {
            let at = format!("case {case} step {step}, bytes {bytes:02x?}");
            match rng.below(6) {
                0 | 1 => {
                    let k = rng.below(8) as usize;
                    let (a, b) = (new.get(&mut nb.single[k]).unwrap(), old.get(&mut ob.single[k]).unwrap());
                    assert_eq!(a, b, "{at}");
                }
                2 => assert_eq!(new.get_bypass().unwrap(), old.get_bypass().unwrap(), "{at}"),
                3 => {
                    let n = 1 + rng.below(64) as usize;
                    let a = new.get_n_bits(n, &mut nb.bits[0]).unwrap();
                    let b = old.get_n_bits(n, &mut ob.bits[0]).unwrap();
                    assert_eq!(a, b, "{at}");
                }
                _ => {
                    // the unary read, on the old reader as its `get`s
                    let mut count = 0usize;
                    let capped = loop {
                        if !old.get(&mut ob.unary[1][count.min(31)]).unwrap() {
                            break false;
                        }
                        count += 1;
                        if count >= MAX_RUN {
                            break true;
                        }
                    };
                    match new.get_unary_encoded(&mut nb.unary[1]) {
                        Ok(v) => assert!(!capped && v == count, "{at}: {v} against {count}"),
                        Err(e) => {
                            assert!(capped, "{at}: refused at {count}");
                            assert_eq!(e.kind(), io::ErrorKind::InvalidData);
                            refused += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(refused > 0, "no unary read was refused: the inputs are not hostile enough");
}

// ---- carries across runs of 0xff ----

/// Decisions that make the left end of the interval creep up under a round
/// number from below, and cross it: those of the arithmetic decoder reading
/// `target` (a number with few bits set, then zeros). The bytes out while it
/// creeps are 0x3f, 0xff, 0xff, ... and the carry that comes makes them
/// 0x40, 0x00, 0x00, ... however many there are.
fn creeping_decisions(seed: u64, target: &[u8], n: usize) -> Vec<(usize, bool)> {
    let mut rng = Rng(seed);
    let mut r = VP8Reader::new(Cursor::new(target.to_vec())).unwrap();
    let mut ctx = [VP8Context::default(); 8];
    (0..n)
        .map(|_| {
            let k = rng.below(8) as usize;
            (k, r.get(&mut ctx[k]).unwrap())
        })
        .collect()
}

/// The coders on the decisions of `creeping_decisions`, and the longest run
/// of 0xff bytes a carry went through (by the model).
fn creeping_case(seed: u64, target: &[u8], n: usize) -> usize {
    let decisions = creeping_decisions(seed, target, n);
    let calls: Vec<Call> = decisions.iter().map(|&(k, b)| Call::Put(b, k)).collect();
    let (new, old) = (stream_new(&calls), stream_old(&calls));
    assert_eq!(new, old, "seed {seed}, target {target:02x?}, {n} decisions");
    let mut m = ModelEnc::new();
    let mut ctx = [ModelCtx::new(); 8];
    for &(k, b) in &decisions {
        m.put(b, &mut ctx[k]);
    }
    let run = m.max_ff_run_carried;
    assert_eq!(m.finish(), new, "seed {seed}: the model");
    run
}

#[test]
fn carries_across_many_ff_bytes_are_the_same() {
    let mut longest_by_count = [0usize; 40];
    let mut with_a_carry = 0usize;
    let targets: [&[u8]; 6] = [
        &[0x40],
        &[0x20],
        &[0x01],
        &[0x40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        &[0x00, 0x80],
        &[0x3f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x80],
    ];
    for seed in 1..=200 * scale() as u64 {
        let target = targets[(seed % 6) as usize];
        let n = 100 + (seed as usize * 37) % 900;
        let run = creeping_case(seed, target, n);
        longest_by_count[run.min(39)] += 1;
        with_a_carry += usize::from(run > 0);
    }
    // the point of the test: it did reach carries across 0xff, runs of
    // several, of a dozen and more
    assert!(with_a_carry > 20, "{longest_by_count:?}");
    for run in [1, 2, 3, 4, 6, 8] {
        let seen: usize = longest_by_count[run..].iter().sum();
        assert!(seen > 0, "no carry went through {run} 0xff bytes: {longest_by_count:?}");
    }
    eprintln!("carries by the longest run of 0xff bytes they went through: {longest_by_count:?}");
}

// ---- preflate's own encoder and decoder, on real deflate streams ----

/// The old crate's writer and reader behind this crate's traits, so that
/// preflate's own code (generic over the coder) runs on either.
struct OldCoder<C>(C);

impl<W: Write> CabacWriter<OldContext> for OldCoder<OldWriter<W>> {
    fn put(&mut self, bit: bool, ctx: &mut OldContext) -> io::Result<()> {
        self.0.put(bit, ctx)
    }

    fn finish(&mut self) -> io::Result<()> {
        self.0.finish()
    }

    // the old crate's own helpers, not this crate's
    fn put_unary_encoded<const A: usize>(&mut self, v: usize, contexts: &mut [OldContext; A]) -> io::Result<()> {
        self.0.put_unary_encoded(v, contexts)
    }

    fn put_n_bits<const A: usize>(&mut self, bits: u64, num_bits: usize, contexts: &mut [OldContext; A]) -> io::Result<()> {
        self.0.put_n_bits(bits, num_bits, contexts)
    }
}

impl<R: Read> CabacReader<OldContext> for OldCoder<OldReader<R>> {
    fn get(&mut self, ctx: &mut OldContext) -> io::Result<bool> {
        self.0.get(ctx)
    }

    fn get_unary_encoded<const A: usize>(&mut self, contexts: &mut [OldContext; A]) -> io::Result<usize> {
        self.0.get_unary_encoded(contexts)
    }

    fn get_n_bits<const A: usize>(&mut self, num_bits: usize, contexts: &mut [OldContext; A]) -> io::Result<u64> {
        self.0.get_n_bits(num_bits, contexts)
    }
}

mod preflate_level {
    use super::*;
    use crate::{
        cabac_codec::{PredictionDecoderCabac, PredictionEncoderCabac},
        deflate::{deflate_reader::parse_deflate_whole, deflate_writer::DeflateWriter},
        estimator::preflate_parameter_estimator::estimate_preflate_parameters,
        preflate_input::PreflateInput,
        statistical_codec::{PredictionDecoder, PredictionEncoder},
        stream_processor::{predict_blocks, recreate_blocks},
        token_predictor::TokenPredictor,
    };

    /// The stream recreated by preflate's decoder, over a coder of the
    /// caller's choice.
    fn recreate<D: PredictionDecoder>(params: &crate::TokenPredictorParameters, plain: &crate::PlainText, mut dec: D) -> Vec<u8> {
        let mut input = PreflateInput::new(plain);
        let mut tp = TokenPredictor::new(params);
        let mut w = DeflateWriter::new();
        recreate_blocks(&mut tp, &mut dec, &mut w, &mut input).unwrap();
        w.flush();
        w.detach_output()
    }

    /// One stream: its corrections by preflate's encoder over each coder,
    /// the same; and the stream recreated from either by preflate's decoder
    /// over each coder, the stream. False when preflate does not open it.
    fn check(name: &str, deflate: &[u8]) -> bool {
        let Ok((contents, plain)) = parse_deflate_whole(deflate) else {
            return false;
        };
        let Ok(params) = estimate_preflate_parameters(&contents, &plain) else {
            return false;
        };
        let mut new = Vec::new();
        {
            let mut enc = PredictionEncoderCabac::new(VP8Writer::new(&mut new).unwrap());
            let mut input = PreflateInput::new(&plain);
            let mut tp = TokenPredictor::new(&params);
            if predict_blocks(&contents.blocks, &mut tp, &mut enc, &mut input).is_err() {
                return false;
            }
            enc.finish();
        }
        let mut old = Vec::new();
        {
            let mut enc = PredictionEncoderCabac::new(OldCoder(OldWriter::new(&mut old).unwrap()));
            let mut input = PreflateInput::new(&plain);
            let mut tp = TokenPredictor::new(&params);
            predict_blocks(&contents.blocks, &mut tp, &mut enc, &mut input).unwrap();
            enc.finish();
        }
        assert_eq!(new, old, "{name}: the corrections");
        let stream = &deflate[..contents.compressed_size];
        for bytes in [&new, &old] {
            let by_new = recreate(&params, &plain, PredictionDecoderCabac::new(VP8Reader::new(Cursor::new(bytes)).unwrap()));
            assert!(by_new == stream, "{name}: recreated by the new reader");
            let by_old = recreate(&params, &plain, PredictionDecoderCabac::new(OldCoder(OldReader::new(Cursor::new(bytes)).unwrap())));
            assert!(by_old == stream, "{name}: recreated by the old reader");
        }
        true
    }

    fn text(n: usize, seed: u64) -> Vec<u8> {
        let mut rng = Rng(seed);
        let mut v = Vec::with_capacity(n);
        while v.len() < n {
            let w = rng.below(3000);
            v.extend_from_slice(format!("word{w} line {} value {}\n", rng.below(1000), rng.below(7)).as_bytes());
        }
        v.truncate(n);
        v
    }

    fn noise(n: usize, seed: u64) -> Vec<u8> {
        let mut rng = Rng(seed);
        (0..n).map(|_| rng.next() as u8).collect()
    }

    /// Texts, noise, runs, and mixes, at every level of the compressor the
    /// dev-dependencies have.
    #[test]
    fn corrections_are_the_same_on_generated_streams() {
        let size = if cfg!(debug_assertions) { 40_000 } else { 300_000 };
        let mut mixed = text(size / 2, 3);
        mixed.extend(noise(size / 2, 4));
        mixed.extend(vec![b'a'; size / 8]);
        mixed.extend(text(size / 2, 5));
        let inputs = [text(size, 1), noise(size / 2, 2), vec![0u8; size], mixed];
        let mut checked = 0;
        for (i, input) in inputs.iter().enumerate() {
            for level in 0..=10 {
                let deflate = miniz_oxide::deflate::compress_to_vec(input, level);
                checked += usize::from(check(&format!("input {i} level {level}"), &deflate));
            }
        }
        eprintln!("{checked} generated streams: same corrections, recreated by either reader");
        assert!(checked >= 20, "only {checked} streams could be opened");
    }

    /// The deflate streams of the files the repository keeps from earlier
    /// releases (gzip members and zip entries), as they were written.
    #[test]
    fn corrections_are_the_same_on_the_repository_fixtures() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/legacy");
        let Ok(entries) = std::fs::read_dir(&dir) else {
            eprintln!("no fixtures at {dir:?}: skipped");
            return;
        };
        let mut checked = 0;
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if name.ends_with(".glyd") {
                continue;
            }
            let data = std::fs::read(e.path()).unwrap();
            for (at, stream) in deflate_streams(&data) {
                checked += usize::from(check(&format!("{name} at {at}"), stream));
            }
        }
        eprintln!("{checked} streams of the repository's fixtures: same corrections, recreated by either reader");
        assert!(checked >= 3, "only {checked} streams from {dir:?}");
    }

    /// The deflate streams in a gzip file's first member or in a zip's entries,
    /// from where they start (what follows is left to the parser).
    fn deflate_streams(data: &[u8]) -> Vec<(usize, &[u8])> {
        let mut found = Vec::new();
        if data.starts_with(&[0x1f, 0x8b, 8]) && data.len() > 18 {
            let flags = data[3];
            let mut at = 10;
            if flags & 4 != 0 {
                at += 2 + usize::from(u16::from_le_bytes([data[at], data[at + 1]]));
            }
            for bit in [8, 16] {
                if flags & bit != 0 {
                    while data[at] != 0 {
                        at += 1;
                    }
                    at += 1;
                }
            }
            if flags & 2 != 0 {
                at += 2;
            }
            found.push((at, &data[at..]));
        } else if data.starts_with(b"PK\x03\x04") {
            let le16 = |p: usize| usize::from(u16::from_le_bytes([data[p], data[p + 1]]));
            let le32 = |p: usize| u32::from_le_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]]) as usize;
            let mut at = 0;
            while at + 30 <= data.len() && &data[at..at + 4] == b"PK\x03\x04" {
                let (method, csize) = (le16(at + 8), le32(at + 18));
                let start = at + 30 + le16(at + 26) + le16(at + 28);
                if method == 8 && csize > 0 {
                    found.push((start, &data[start..start + csize]));
                }
                at = start + csize;
            }
        }
        found
    }
}

// ---- speed, and a long run (on demand) ----

fn mix(seed: u64, n: usize) -> Vec<(usize, bool)> {
    super::tests::workload_calls(seed, n).collect()
}

/// `cargo test --release -- --ignored --nocapture speed`
#[test]
#[ignore = "a measurement"]
fn speed_of_the_two_coders() {
    use std::time::Instant;
    let n = 20_000_000;
    let decisions = mix(1, n);
    let time = |f: &mut dyn FnMut()| {
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            f();
            best = best.min(t.elapsed().as_secs_f64());
        }
        best
    };
    let mut stream = Vec::new();
    let new_w = time(&mut || {
        let mut buf = Vec::with_capacity(1 << 20);
        let mut w = VP8Writer::new(&mut buf).unwrap();
        let mut ctx = [VP8Context::default(); 8];
        for &(k, bit) in &decisions {
            w.put(bit, &mut ctx[k]).unwrap();
        }
        w.finish().unwrap();
        stream = buf;
    });
    let old_w = time(&mut || {
        let mut buf = Vec::with_capacity(1 << 20);
        let mut w = OldWriter::new(&mut buf).unwrap();
        let mut ctx: [OldContext; 8] = Default::default();
        for &(k, bit) in &decisions {
            w.put(bit, &mut ctx[k]).unwrap();
        }
        w.finish().unwrap();
        assert_eq!(buf, stream);
    });
    let mut sink = 0usize;
    let new_r = time(&mut || {
        let mut r = VP8Reader::new(Cursor::new(&stream)).unwrap();
        let mut ctx = [VP8Context::default(); 8];
        for &(k, _) in &decisions {
            sink += usize::from(r.get(&mut ctx[k]).unwrap());
        }
    });
    let old_r = time(&mut || {
        let mut r = OldReader::new(Cursor::new(&stream)).unwrap();
        let mut ctx: [OldContext; 8] = Default::default();
        for &(k, _) in &decisions {
            sink += usize::from(r.get(&mut ctx[k]).unwrap());
        }
    });
    let m = |s: f64| n as f64 / s / 1e6;
    eprintln!(
        "{n} decisions. write: new {:.0} M/s, old {:.0} M/s ({:.2}x). read: new {:.0} M/s, old {:.0} M/s ({:.2}x) [{sink}]",
        m(new_w),
        m(old_w),
        old_w / new_w,
        m(new_r),
        m(old_r),
        old_r / new_r,
    );
}

/// Both readers on the stream of `decisions`: every decision read back.
fn read_back(stream: &[u8], decisions: &[(usize, bool)]) {
    let mut new = VP8Reader::new(Cursor::new(stream)).unwrap();
    let mut old = OldReader::new(Cursor::new(stream)).unwrap();
    let mut nc = [VP8Context::default(); 8];
    let mut oc: [OldContext; 8] = Default::default();
    for (i, &(k, bit)) in decisions.iter().enumerate() {
        assert_eq!(new.get(&mut nc[k]).unwrap(), bit, "decision {i}: the new reader");
        assert_eq!(old.get(&mut oc[k]).unwrap(), bit, "decision {i}: the old reader");
    }
}

/// Both readers on `len` bytes no writer made, `n` decisions, past the end
/// too: the same decisions, one by one.
fn read_noise(seed: u64, len: usize, n: usize) {
    let mut rng = Rng(seed);
    let mut bytes: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
    if seed % 2 == 0 {
        bytes[0] &= 0x7f; // a first byte a writer could have made
    }
    let mut new = VP8Reader::new(Cursor::new(&bytes)).unwrap();
    let mut old = OldReader::new(Cursor::new(&bytes)).unwrap();
    let mut nc = [VP8Context::default(); 8];
    let mut oc: [OldContext; 8] = Default::default();
    for i in 0..n {
        let k = rng.below(8) as usize;
        assert_eq!(new.get(&mut nc[k]).unwrap(), old.get(&mut oc[k]).unwrap(), "seed {seed}, decision {i}");
    }
}

/// Blocks of decisions on every core, new against old, until `DECISIONS`
/// (default two billion) are done: the streams the writers make, what each
/// reader reads back from them, and what the readers make of noise.
/// `DECISIONS=1000000000000 cargo test --release -- --ignored --nocapture
/// long_differential`. The carries the new writer made, by the run of 0xff
/// bytes they went back through, are counted and printed.
#[test]
#[ignore = "on demand: minutes to hours"]
fn long_differential() {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    let total: u64 = std::env::var("DECISIONS").ok().and_then(|s| s.parse().ok()).unwrap_or(2_000_000_000);
    let block = 5_000_000usize;
    let blocks = total / block as u64;
    let next = AtomicU64::new(0);
    let done = AtomicU64::new(0);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let started = std::time::Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let b = next.fetch_add(1, Relaxed);
                if b >= blocks {
                    break;
                }
                let seed = 1_000_003 + b;
                let decisions = mix(seed, block);
                let calls: Vec<Call> = decisions.iter().map(|&(k, bit)| Call::Put(bit, k)).collect();
                let (new, old) = (stream_new(&calls), stream_old(&calls));
                assert_eq!(new, old, "block {b}, seed {seed}");
                if b % 4 == 0 {
                    read_back(&new, &decisions);
                    read_noise(seed, 3_000_000, 20_000_000);
                }
                let d = done.fetch_add(1, Relaxed) + 1;
                if d % 2000 == 0 {
                    eprintln!("{} blocks, {:.0} s", d, started.elapsed().as_secs_f64());
                }
            });
        }
    });
    let c: Vec<u64> = super::tests::CARRIES.iter().map(|c| c.load(Relaxed)).collect();
    eprintln!(
        "{} decisions in {} blocks on {threads} threads, {:.0} s: every stream identical. Carries by the 0xff bytes they went back through (0, 1, 2, 3, 4+): {c:?}",
        done.load(Relaxed) * block as u64,
        done.load(Relaxed),
        started.elapsed().as_secs_f64(),
    );
}
