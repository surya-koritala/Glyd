//! Model files: safetensors and GGUF files coded tensor by tensor (src/weights), every byte back.
//! The tensors are made here (weights from a normal distribution in every element type, quantised
//! the way ggml quantises them, or filled with the fields of a block); the real files of the corpus
//! are read when `GLYD_WEIGHTS_CORPUS` names a directory of them.
#![cfg(feature = "deflate")]

const MAGIC: &[u8; 8] = b"GLYDWGT1";

// ------------------------------------------------------------------ data

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (a, b) = (self.unit().max(1e-300), self.unit());
        (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

fn gauss(r: &mut Rng, n: usize, std: f64) -> Vec<f32> {
    (0..n).map(|_| (r.normal() * std) as f32).collect()
}

fn bf16_bits(f: f32) -> u16 {
    let u = f.to_bits();
    ((u as u64 + 0x7FFF + ((u >> 16) & 1) as u64) >> 16) as u16
}

/// A binary16 from `f`, rounded to nearest even (overflow to infinity).
fn f16_bits(f: f32) -> u16 {
    let u = f.to_bits();
    let sign = ((u >> 16) & 0x8000) as u16;
    let exp = ((u >> 23) & 0xFF) as i32 - 127;
    let man = u & 0x7F_FFFF;
    if exp > 15 {
        return sign | 0x7C00;
    }
    if exp >= -14 {
        let half = ((exp + 15) as u32) << 10 | man >> 13;
        let rest = man & 0x1FFF;
        let up = rest > 0x1000 || (rest == 0x1000 && half & 1 == 1);
        return sign | (half + up as u32) as u16;
    }
    if exp >= -24 {
        let m = man | 0x80_0000;
        let shift = (-exp - 1) as u32;
        let half = m >> shift;
        let rest = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let up = rest > mid || (rest == mid && half & 1 == 1);
        return sign | (half + up as u32) as u16;
    }
    sign
}

/// The e4m3 (bias 7, no infinities) or e5m2 (bias 15) byte nearest `f` (saturating).
fn fp8_bits(f: f32, e4m3: bool) -> u8 {
    let (ebits, mbits, bias): (u32, u32, i32) = if e4m3 { (4, 3, 7) } else { (5, 2, 15) };
    let max = if e4m3 { 448.0 } else { 57344.0 };
    let (sign, a) = (if f < 0.0 { 0x80u8 } else { 0 }, f.abs().min(max));
    // all non-negative values of the format, nearest wins
    let mut best = (f32::MAX, 0u8);
    for code in 0..(1u32 << (ebits + mbits)) {
        let e = (code >> mbits) as i32;
        let m = (code & ((1 << mbits) - 1)) as f32;
        if e4m3 && e == 15 && m == 7.0 {
            continue;
        }
        if !e4m3 && e == 31 {
            continue;
        }
        let v = if e == 0 { m / (1 << mbits) as f32 * 2f32.powi(1 - bias) } else { (1.0 + m / (1 << mbits) as f32) * 2f32.powi(e - bias) };
        if (v - a).abs() < best.0 {
            best = ((v - a).abs(), code as u8);
        }
    }
    sign | best.1
}

fn e4m3_value(b: u8) -> f32 {
    let (e, m) = (((b >> 3) & 15) as i32, (b & 7) as f32);
    let v = if e == 0 { m / 8.0 * 2f32.powi(-6) } else { (1.0 + m / 8.0) * 2f32.powi(e - 7) };
    if b & 0x80 != 0 { -v } else { v }
}

fn as_bf16(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|&x| bf16_bits(x).to_le_bytes()).collect()
}
fn as_f16(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|&x| f16_bits(x).to_le_bytes()).collect()
}
fn as_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn as_f64(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|&x| (x as f64).to_le_bytes()).collect()
}
fn as_fp8(v: &[f32], e4m3: bool) -> Vec<u8> {
    v.iter().map(|&x| fp8_bits(x, e4m3)).collect()
}

// ggml quantisers (the reference ones, for the block types a model file is made of)

fn block_32(v: &[f32], block: impl Fn(&[f32]) -> Vec<u8>) -> Vec<u8> {
    v.chunks_exact(32).flat_map(|c| block(c)).collect()
}

fn q8_0(v: &[f32]) -> Vec<u8> {
    block_32(v, |c| {
        let amax = c.iter().fold(0f32, |m, &x| m.max(x.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut b = f16_bits(d).to_le_bytes().to_vec();
        b.extend(c.iter().map(|&x| (x * id).round() as i8 as u8));
        b
    })
}

fn q4_0(v: &[f32]) -> Vec<u8> {
    block_32(v, |c| {
        let max = c.iter().fold(0f32, |m, &x| if x.abs() > m.abs() { x } else { m });
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let q: Vec<u8> = c.iter().map(|&x| ((x * id + 8.5) as i32).min(15) as u8).collect();
        let mut b = f16_bits(d).to_le_bytes().to_vec();
        b.extend((0..16).map(|j| q[j] | q[j + 16] << 4));
        b
    })
}

fn q4_1(v: &[f32]) -> Vec<u8> {
    block_32(v, |c| {
        let (min, max) = c.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &x| (lo.min(x), hi.max(x)));
        let d = (max - min) / 15.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let q: Vec<u8> = c.iter().map(|&x| (((x - min) * id + 0.5) as i32).min(15) as u8).collect();
        let mut b = f16_bits(d).to_le_bytes().to_vec();
        b.extend(f16_bits(min).to_le_bytes());
        b.extend((0..16).map(|j| q[j] | q[j + 16] << 4));
        b
    })
}

fn q5(v: &[f32], with_min: bool) -> Vec<u8> {
    block_32(v, |c| {
        let (min, max) = c.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &x| (lo.min(x), hi.max(x)));
        let big = c.iter().fold(0f32, |m, &x| if x.abs() > m.abs() { x } else { m });
        let (d, id, off) = if with_min {
            let d = (max - min) / 31.0;
            (d, if d != 0.0 { 1.0 / d } else { 0.0 }, min)
        } else {
            let d = big / -16.0;
            (d, if d != 0.0 { 1.0 / d } else { 0.0 }, 0.0)
        };
        let q: Vec<u8> = c
            .iter()
            .map(|&x| if with_min { (((x - off) * id + 0.5) as i32).min(31) as u8 } else { ((x * id + 16.5) as i32).min(31) as u8 })
            .collect();
        let mut b = f16_bits(d).to_le_bytes().to_vec();
        if with_min {
            b.extend(f16_bits(min).to_le_bytes());
        }
        let mut qh = 0u32;
        for j in 0..16 {
            qh |= ((q[j] >> 4) as u32) << j | ((q[j + 16] >> 4) as u32) << (j + 16);
        }
        b.extend(qh.to_le_bytes());
        b.extend((0..16).map(|j| (q[j] & 15) | (q[j + 16] & 15) << 4));
        b
    })
}

/// `n` blocks of `size` bytes whose fields are drawn plausibly: a small half-precision scale where
/// `scale_at` says, bell-shaped bytes (two 4-bit codes) elsewhere.
fn blocks(r: &mut Rng, size: usize, n: usize, scale_at: &[usize]) -> Vec<u8> {
    let mut out = Vec::with_capacity(size * n);
    for _ in 0..n {
        let mut b = vec![0u8; size];
        for x in b.iter_mut() {
            let (lo, hi) = (r.normal() * 2.4 + 7.5, r.normal() * 2.4 + 7.5);
            *x = (lo.round().clamp(0.0, 15.0) as u8) | (hi.round().clamp(0.0, 15.0) as u8) << 4;
        }
        for &at in scale_at {
            let d = f16_bits((0.001 + r.unit() as f32 * 0.002) * if r.below(2) == 0 { 1.0 } else { 2.0 });
            b[at..at + 2].copy_from_slice(&d.to_le_bytes());
        }
        out.extend(b);
    }
    out
}

// ------------------------------------------------------------------ files

fn safetensors(tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) -> Vec<u8> {
    let mut h = String::from("{\"__metadata__\":{\"format\":\"pt\"}");
    let mut off = 0usize;
    for (name, dtype, shape, data) in tensors {
        h += &format!(",\"{}\":{{\"dtype\":\"{}\",\"shape\":{:?},\"data_offsets\":[{},{}]}}", name, dtype, shape, off, off + data.len());
        off += data.len();
    }
    h.push('}');
    while h.len() % 8 != 0 {
        h.push(' ');
    }
    let mut f = (h.len() as u64).to_le_bytes().to_vec();
    f.extend_from_slice(h.as_bytes());
    for (_, _, _, d) in tensors {
        f.extend_from_slice(d);
    }
    f
}

fn gguf_string(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// A GGUF v3 file: a few key-values (a vocabulary among them), the tensor infos and the data, aligned.
/// Tensors are (name, ggml type, elements in a row, rows, bytes).
fn gguf(tensors: &[(&str, u32, u64, u64, Vec<u8>)]) -> Vec<u8> {
    let mut h = b"GGUF".to_vec();
    h.extend(3u32.to_le_bytes());
    h.extend((tensors.len() as u64).to_le_bytes());
    h.extend(4u64.to_le_bytes());
    gguf_string(&mut h, "general.architecture");
    h.extend(8u32.to_le_bytes());
    gguf_string(&mut h, "llama");
    gguf_string(&mut h, "general.alignment");
    h.extend(4u32.to_le_bytes());
    h.extend(32u32.to_le_bytes());
    gguf_string(&mut h, "tokenizer.ggml.tokens");
    h.extend(9u32.to_le_bytes());
    h.extend(8u32.to_le_bytes());
    h.extend(300u64.to_le_bytes());
    for i in 0..300 {
        gguf_string(&mut h, &format!("▁tok{}", i * 7));
    }
    gguf_string(&mut h, "tokenizer.ggml.scores");
    h.extend(9u32.to_le_bytes());
    h.extend(6u32.to_le_bytes());
    h.extend(300u64.to_le_bytes());
    for i in 0..300 {
        h.extend((-(i as f32)).to_le_bytes());
    }
    let mut off = 0u64;
    let mut data = Vec::new();
    for (name, ty, cols, rows, bytes) in tensors {
        gguf_string(&mut h, name);
        h.extend(2u32.to_le_bytes());
        h.extend(cols.to_le_bytes());
        h.extend(rows.to_le_bytes());
        h.extend(ty.to_le_bytes());
        h.extend(off.to_le_bytes());
        data.extend_from_slice(bytes);
        let pad = (32 - data.len() % 32) % 32;
        data.resize(data.len() + pad, 0);
        off = data.len() as u64;
    }
    h.resize(h.len().div_ceil(32) * 32, 0);
    h.extend(data);
    h
}

// ------------------------------------------------------------------ round trips

type Compressor = (&'static str, fn(&[u8], &mut Vec<u8>));

fn compressors() -> Vec<Compressor> {
    vec![
        ("max", glyd::compress_into_max),
        ("max long", glyd::compress_into_max_long),
        ("parallel max", glyd::compress_parallel_into_max),
        ("records max", glyd::compress_records_into_max),
        ("ultra", glyd::compress_into_ultra),
        ("max dense", glyd::compress_into_max_dense),
    ]
}

fn all_ways_back(c: &[u8], original: &[u8], what: &str) {
    assert_eq!(glyd::decompress(c).unwrap(), original, "{what}: decompress");
    assert_eq!(glyd::decompress_parallel(c).unwrap(), original, "{what}: parallel");
    assert_eq!(glyd::decompressed_len(c).unwrap(), original.len(), "{what}: length");
    let mut dst = vec![0u8; original.len()];
    assert_eq!(glyd::decompress_into(c, &mut dst).unwrap(), original.len(), "{what}: into");
    assert_eq!(dst, original, "{what}: into bytes");
    let mut streamed = Vec::new();
    glyd::decompress_stream(c, |b| {
        streamed.extend_from_slice(b);
        Ok(())
    })
    .unwrap();
    assert_eq!(streamed, original, "{what}: stream");
}

fn round_trip(file: &[u8], what: &str) -> Vec<u8> {
    let mut first = Vec::new();
    for (name, f) in compressors() {
        let mut c = Vec::new();
        f(file, &mut c);
        all_ways_back(&c, file, &format!("{what} at {name}"));
        if first.is_empty() {
            first = c;
        }
    }
    first
}

// ------------------------------------------------------------------ tests

#[test]
fn every_safetensors_dtype_comes_back() {
    let mut r = Rng::new(1);
    let n = 40_000;
    let w = gauss(&mut r, n, 0.02);
    let ints: Vec<u8> = (0..n * 8).map(|i| if i % 8 < 2 { (r.next() % 200) as u8 } else { 0 }).collect();
    let f = safetensors(&[
        ("a.bf16", "BF16", vec![200, 200], as_bf16(&w)),
        ("a.f16", "F16", vec![200, 200], as_f16(&w)),
        ("a.f32", "F32", vec![200, 200], as_f32(&w)),
        ("a.f64", "F64", vec![200, 200], as_f64(&w)),
        ("a.e4m3", "F8_E4M3", vec![200, 200], as_fp8(&w.iter().map(|x| x * 100.0).collect::<Vec<_>>(), true)),
        ("a.e5m2", "F8_E5M2", vec![200, 200], as_fp8(&w.iter().map(|x| x * 100.0).collect::<Vec<_>>(), false)),
        ("a.i8", "I8", vec![n], ints[..n].to_vec()),
        ("a.u8", "U8", vec![n], ints[n..2 * n].to_vec()),
        ("a.i16", "I16", vec![n], ints[..2 * n].to_vec()),
        ("a.u16", "U16", vec![n], ints[2 * n..4 * n].to_vec()),
        ("a.i32", "I32", vec![n], ints[..4 * n].to_vec()),
        ("a.u32", "U32", vec![n], ints[4 * n..8 * n].to_vec()),
        ("a.i64", "I64", vec![n], ints[..8 * n].to_vec()),
        ("a.u64", "U64", vec![n], ints[..8 * n].to_vec()),
        ("a.bool", "BOOL", vec![n], (0..n).map(|_| (r.next() % 2) as u8).collect()),
        ("a.other", "F8_E8M0", vec![n], (0..n).map(|_| 120 + (r.next() % 5) as u8).collect()),
        ("a.norm", "BF16", vec![64], as_bf16(&w[..64])),
    ]);
    let c = round_trip(&f, "every dtype");
    assert!(c.starts_with(MAGIC), "the model-file mode was not used");
    assert!(c.len() < f.len() * 7 / 10, "{} of {}", c.len(), f.len());
}

#[test]
fn every_ggml_type_comes_back() {
    let mut r = Rng::new(2);
    let (cols, rows) = (1024u64, 64u64);
    let w = gauss(&mut r, (cols * rows) as usize, 0.03);
    let blk = |size: usize, scale_at: &[usize], elems: u64| blocks(&mut Rng::new(size as u64), size, (cols * rows / elems) as usize, scale_at);
    let _ = &mut r;
    let f = gguf(&[
        ("t.f32", 0, cols, rows, as_f32(&w)),
        ("t.f16", 1, cols, rows, as_f16(&w)),
        ("t.bf16", 30, cols, rows, as_bf16(&w)),
        ("t.q4_0", 2, cols, rows, q4_0(&w)),
        ("t.q4_1", 3, cols, rows, q4_1(&w)),
        ("t.q5_0", 6, cols, rows, q5(&w, false)),
        ("t.q5_1", 7, cols, rows, q5(&w, true)),
        ("t.q8_0", 8, cols, rows, q8_0(&w)),
        ("t.q4_k", 12, cols, rows, blk(144, &[0, 2], 256)),
        ("t.q5_k", 13, cols, rows, blk(176, &[0, 2], 256)),
        ("t.q6_k", 14, cols, rows, blk(210, &[208], 256)),
        ("t.iq4_nl", 20, cols, rows, blk(18, &[0], 32)),
        ("t.iq4_xs", 23, cols, rows, blk(136, &[0], 256)),
        ("t.mxfp4", 39, cols, rows, blk(17, &[], 32)),
        ("t.i8", 24, cols, rows, (0..cols * rows).map(|_| 100 + (r.next() % 9) as u8).collect()),
        ("t.i16", 25, cols, rows, (0..cols * rows * 2).map(|i| if i % 2 == 0 { (r.next() % 50) as u8 } else { 0 }).collect()),
        ("t.i32", 26, cols, rows, (0..cols * rows * 4).map(|i| if i % 4 == 0 { (r.next() % 50) as u8 } else { 0 }).collect()),
        ("t.i64", 27, cols, rows, (0..cols * rows * 8).map(|i| if i % 8 == 0 { (r.next() % 50) as u8 } else { 0 }).collect()),
        ("t.f64", 28, cols, rows, as_f64(&w)),
        // a type this does not code (IQ2_XXS: 66 bytes of 256 elements): left to the level
        ("t.iq2xxs", 16, cols, rows, (0..cols * rows / 256 * 66).map(|_| r.next() as u8).collect()),
        ("t.norm", 0, 256, 1, as_f32(&w[..256])),
    ]);
    let c = round_trip(&f, "every ggml type");
    assert!(c.starts_with(MAGIC), "the model-file mode was not used");
    assert!(c.len() < f.len() * 9 / 10, "{} of {}", c.len(), f.len());
}

#[test]
fn each_type_pays_what_its_entropy_says() {
    let mut r = Rng::new(3);
    let n = 1 << 20;
    let w = gauss(&mut r, n, 0.02);
    let saved = |bytes: Vec<u8>, dtype: &str| {
        let f = safetensors(&[("w", dtype, vec![bytes.len()], bytes)]);
        let mut c = Vec::new();
        glyd::compress_into_max(&f, &mut c);
        assert_eq!(glyd::decompress(&c).unwrap(), f);
        1.0 - c.len() as f64 / f.len() as f64
    };
    // bf16: sign 1, mantissa 7, exponent 2.5 bits of 16; half: the same with a mantissa of 10 bits
    assert!(saved(as_bf16(&w), "BF16") > 0.33);
    assert!(saved(as_f16(&w), "F16") > 0.12);
    // an 8-bit float has about 6.5 bits
    assert!(saved(as_fp8(&w.iter().map(|x| x * 50.0).collect::<Vec<_>>(), true), "F8_E4M3") > 0.15);
    // bf16 values from an fp8 grid times a scale: the mantissa is 3 bits
    let grid: Vec<f32> = w.iter().map(|&x| e4m3_value(fp8_bits(x * 50.0, true)) * 0.00071).collect();
    assert!(saved(as_bf16(&grid), "BF16") > 0.45);
    let q8 = q8_0(&w);
    let f = gguf(&[("q", 8, 4096, (n / 4096) as u64, q8)]);
    let mut c = Vec::new();
    glyd::compress_into_max(&f, &mut c);
    assert_eq!(glyd::decompress(&c).unwrap(), f);
    assert!(c.len() * 100 < f.len() * 95, "q8_0 {} of {}", c.len(), f.len());
}

#[test]
fn quantised_weights_with_few_scales_pay_more() {
    // a Q8_0 whose block scales are a handful of values (as from an fp8 checkpoint) and whose codes follow them
    let mut r = Rng::new(4);
    let mut b = Vec::new();
    for _ in 0..40_000 {
        let scale = [0.0021f32, 0.0023, 0.0026][r.below(3)];
        b.extend(f16_bits(scale).to_le_bytes());
        let c = r.below(8) as f64 / 8.0 + 1.0;
        b.extend((0..32).map(|i| if i == 5 { 127u8 } else { ((r.normal() * 30.0 * c) as i32).clamp(-126, 126) as i8 as u8 }));
    }
    let f = gguf(&[("q", 8, 4096, (40_000 * 32 / 4096) as u64, b)]);
    let c = round_trip(&f, "q8_0 scales");
    assert!(c.starts_with(MAGIC));
}

#[test]
fn bytes_outside_tensors_are_kept() {
    // padding, a trailing tail, a tensor the header leaves out, and bytes between tensors
    let mut r = Rng::new(5);
    let w = gauss(&mut r, 100_000, 0.02);
    let mut f = safetensors(&[("a", "BF16", vec![100_000], as_bf16(&w)), ("b", "F32", vec![25_000], as_f32(&w[..25_000]))]);
    f.extend((0..777).map(|i| (i * 31) as u8));
    let c = round_trip(&f, "a tail");
    assert!(c.starts_with(MAGIC));
    // every region of a GGUF file followed by garbage the tensor infos do not name
    let g = gguf(&[("a", 30, 1000, 100, as_bf16(&w))]);
    let mut g2 = g.clone();
    g2.extend((0..4097).map(|i| (i * 7) as u8));
    round_trip(&g2, "gguf with a tail");
}

#[test]
fn what_is_not_a_model_file_goes_the_old_way() {
    let mut r = Rng::new(6);
    for len in [0usize, 1, 7, 8, 9, 100, 5000, 70_000] {
        let mut v: Vec<u8> = (0..len).map(|_| r.next() as u8).collect();
        for prefix in [&b"GGUF"[..], &8u64.to_le_bytes()[..], b"{\"a\"", b""] {
            let n = prefix.len().min(v.len());
            v[..n].copy_from_slice(&prefix[..n]);
            let c = round_trip(&v, &format!("random bytes of {len}"));
            assert!(!c.starts_with(MAGIC));
        }
    }
}

fn mutations(file: &[u8], header_end: usize, r: &mut Rng, count: usize) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    for k in 0..count {
        let mut m = file.to_vec();
        match k % 6 {
            0 => {
                let i = r.below(header_end);
                m[i] = r.next() as u8;
            }
            1 => {
                let i = r.below(header_end.min(m.len()));
                m[i] ^= 1 << r.below(8);
            }
            2 => m.truncate(r.below(header_end + 100)),
            3 => {
                let i = r.below(header_end.min(m.len() - 8));
                m[i..i + 8].copy_from_slice(&u64::MAX.to_le_bytes());
            }
            4 => {
                let i = r.below(header_end.min(m.len() - 4));
                m[i..i + 4].copy_from_slice(&(r.next() as u32).to_le_bytes());
            }
            _ => {
                let (a, b) = (r.below(header_end), r.below(header_end));
                m.swap(a, b);
            }
        }
        out.push(m);
    }
    out
}

#[test]
fn malformed_safetensors_headers_come_back() {
    let mut r = Rng::new(7);
    let w = gauss(&mut r, 30_000, 0.02);
    let f = safetensors(&[("a", "BF16", vec![30_000], as_bf16(&w)), ("b", "F16", vec![30_000], as_f16(&w)), ("c", "F8_E4M3", vec![30_000], as_fp8(&w, true))]);
    let header_end = 8 + u64::from_le_bytes(f[..8].try_into().unwrap()) as usize;
    for (k, m) in mutations(&f, header_end, &mut r, 300).iter().enumerate() {
        let mut c = Vec::new();
        glyd::compress_into_max(m, &mut c);
        assert_eq!(glyd::decompress(&c).unwrap(), *m, "mutation {k}");
    }
    // the named cases: lengths past the end, overlapping and backwards offsets, a header with no end
    let cases = [
        "{\"a\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,9999999999]}}",
        "{\"a\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[100,0]}}",
        "{\"a\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,60000]},\"b\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[30000,90000]}}",
        "{\"a\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,60000]",
        "{\"a\":{\"dtype\":\"\u{0}\",\"shape\":[1],\"data_offsets\":[0,60000]}}",
        "{\"a\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[18446744073709551615,18446744073709551615]}}",
        "{}",
    ];
    for h in cases {
        let mut file = (h.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(h.as_bytes());
        file.extend(as_bf16(&w));
        round_trip(&file, h);
    }
    // header lengths: nothing, everything, past the end
    for n in [0u64, 1, 2, 8, 100, u64::MAX, 1 << 40, 100_000_000] {
        let mut file = n.to_le_bytes().to_vec();
        file.extend(&f[8..]);
        round_trip(&file, &format!("header length {n}"));
    }
}

#[test]
fn malformed_gguf_headers_come_back() {
    let mut r = Rng::new(8);
    let w = gauss(&mut r, 64 * 1024, 0.03);
    let f = gguf(&[("a", 8, 1024, 64, q8_0(&w)), ("b", 30, 1024, 64, as_bf16(&w)), ("c", 12, 1024, 64, blocks(&mut Rng::new(1), 144, 256, &[0, 2]))]);
    let header_end = f.windows(4).position(|x| x == b"t.ab").unwrap_or(2000).min(2000);
    for (k, m) in mutations(&f, header_end.max(600), &mut r, 400).iter().enumerate() {
        let mut c = Vec::new();
        glyd::compress_into_max(m, &mut c);
        assert_eq!(glyd::decompress(&c).unwrap(), *m, "mutation {k}");
    }
    // counts past what the file can hold, and version numbers other than 2 and 3
    for (at, v) in [(8usize, u64::MAX), (8, 1 << 40), (16, u64::MAX), (16, 1 << 40), (8, 0), (16, 0)] {
        let mut m = f.clone();
        m[at..at + 8].copy_from_slice(&v.to_le_bytes());
        round_trip(&m, &format!("count at {at} = {v}"));
    }
    for ver in [0u32, 1, 4, 99, u32::MAX] {
        let mut m = f.clone();
        m[4..8].copy_from_slice(&ver.to_le_bytes());
        round_trip(&m, &format!("version {ver}"));
    }
}

#[test]
fn truncated_files_come_back() {
    let mut r = Rng::new(9);
    let w = gauss(&mut r, 24_000, 0.02);
    let st = safetensors(&[("a", "BF16", vec![20_000], as_bf16(&w[..20_000])), ("b", "F32", vec![20_000], as_f32(&w[..20_000]))]);
    let gg = gguf(&[("a", 8, 1024, 20, q8_0(&w[..20 * 1024])), ("b", 30, 1024, 19, as_bf16(&w[..19 * 1024]))]);
    for file in [&st, &gg] {
        let step = (file.len() / 60).max(1);
        let mut cut = 0;
        while cut < file.len() {
            let t = &file[..cut];
            let mut c = Vec::new();
            glyd::compress_into_max(t, &mut c);
            assert_eq!(glyd::decompress(&c).unwrap(), t, "cut at {cut}");
            cut += step + r.below(97);
        }
        // and a byte short of the whole
        let t = &file[..file.len() - 1];
        let mut c = Vec::new();
        glyd::compress_into_max(t, &mut c);
        assert_eq!(glyd::decompress(&c).unwrap(), t);
    }
}

/// A file of some tensors of random types, sizes and fills.
fn random_model_file(r: &mut Rng) -> Vec<u8> {
    let kinds = 3 + r.below(5);
    let fill = |r: &mut Rng, n: usize, width: usize| -> Vec<u8> {
        match r.below(6) {
            0 => (0..n * width).map(|_| r.next() as u8).collect(),
            1 => vec![r.next() as u8; n * width],
            2 => (0..n * width).map(|i| (i / width) as u8).collect(),
            3 => as_bf16(&gauss(r, n * width / 2, 0.05)),
            4 => (0..n * width).map(|_| if r.below(10) == 0 { r.next() as u8 } else { 0 }).collect(),
            _ => (0..n * width).map(|_| (r.normal() * 20.0 + 128.0) as u8).collect(),
        }
    };
    if r.below(2) == 0 {
        let dtypes = [("BF16", 2usize), ("F16", 2), ("F32", 4), ("F8_E4M3", 1), ("I64", 8), ("U8", 1)];
        let t: Vec<(String, &str, Vec<usize>, Vec<u8>)> = (0..kinds)
            .map(|i| {
                let (d, w) = dtypes[r.below(dtypes.len())];
                let n = 1 + r.below(30_000);
                (format!("t{i}"), d, vec![n], fill(r, n, w))
            })
            .collect();
        let t: Vec<(&str, &str, Vec<usize>, Vec<u8>)> = t.iter().map(|(a, b, c, d)| (a.as_str(), *b, c.clone(), d.clone())).collect();
        safetensors(&t)
    } else {
        // (type, bytes per block, elements per block)
        let types = [(0u32, 4usize, 1u64), (1, 2, 1), (30, 2, 1), (8, 34, 32), (2, 18, 32), (12, 144, 256), (14, 210, 256), (39, 17, 32), (3, 20, 32), (6, 22, 32), (23, 136, 256)];
        let t: Vec<(String, u32, u64, u64, Vec<u8>)> = (0..kinds)
            .map(|i| {
                let (ty, bytes, elems) = types[r.below(types.len())];
                let rows = 1 + r.below(40) as u64;
                let cols = elems * (1 + r.below(30) as u64);
                let nblocks = (cols / elems * rows) as usize;
                (format!("t{i}"), ty, cols, rows, fill(r, nblocks, bytes))
            })
            .collect();
        let t: Vec<(&str, u32, u64, u64, Vec<u8>)> = t.iter().map(|(a, b, c, d, e)| (a.as_str(), *b, *c, *d, e.clone())).collect();
        gguf(&t)
    }
}

#[test]
fn random_model_files_come_back() {
    let mut r = Rng::new(10);
    for k in 0..120 {
        let f = random_model_file(&mut r);
        let mut c = Vec::new();
        glyd::compress_into_max(&f, &mut c);
        assert_eq!(glyd::decompress(&c).unwrap(), f, "file {k}");
        assert_eq!(glyd::decompress_parallel(&c).unwrap(), f, "file {k} (parallel)");
        let mut d = Vec::new();
        glyd::compress_parallel_into_max(&f, &mut d);
        assert_eq!(glyd::decompress(&d).unwrap(), f, "file {k} (parallel max)");
    }
}

#[test]
fn damaged_streams_are_refused_or_read_never_trusted() {
    let mut r = Rng::new(11);
    let w = gauss(&mut r, 80_000, 0.02);
    let f = gguf(&[("a", 8, 1024, 78, q8_0(&w[..78 * 1024])), ("b", 30, 1024, 78, as_bf16(&w[..78 * 1024])), ("c", 12, 1024, 40, blocks(&mut Rng::new(4), 144, 160, &[0, 2]))]);
    let mut c = Vec::new();
    glyd::compress_into_max(&f, &mut c);
    assert!(c.starts_with(MAGIC));
    assert_eq!(glyd::decompress(&c).unwrap(), f);
    let mut errors = 0;
    for k in 0..4000 {
        let mut m = c.clone();
        match k % 4 {
            0 => {
                let i = r.below(m.len());
                m[i] ^= 1 << r.below(8);
            }
            1 => {
                let i = r.below(m.len());
                m[i] = r.next() as u8;
            }
            2 => m.truncate(r.below(m.len())),
            _ => {
                let at = r.below(m.len());
                m.insert(at, r.next() as u8);
            }
        }
        // an error, or the bytes it was given (a checksum covers every chunk, and a damaged header leaves the file unreadable)
        match glyd::decompress(&m) {
            Ok(out) => assert!(out == f || m != c, "an undamaged stream gave other bytes"),
            Err(_) => errors += 1,
        }
        let _ = glyd::decompress_parallel(&m);
        let _ = glyd::decompressed_len(&m);
    }
    assert!(errors > 3000, "only {errors} of 4000 damaged streams were refused");
    // an envelope of a version this does not know
    for magic in [&b"GLYDWGT2"[..], b"GLYDWGT0", b"GLYDWGT\0"] {
        let mut m = magic.to_vec();
        m.extend_from_slice(&c[8..]);
        assert!(glyd::decompress(&m).is_err());
    }
}

#[test]
fn a_file_of_many_small_tensors_comes_back() {
    let mut r = Rng::new(12);
    let w = gauss(&mut r, 1 << 14, 0.02);
    let names: Vec<String> = (0..600).map(|i| format!("model.layers.{i}.weight")).collect();
    let t: Vec<(&str, &str, Vec<usize>, Vec<u8>)> = names.iter().enumerate().map(|(i, n)| (n.as_str(), "BF16", vec![5000 + i], as_bf16(&w[..5000 + i]))).collect();
    let f = safetensors(&t);
    let c = round_trip(&f, "many tensors");
    assert!(c.starts_with(MAGIC));
}

#[test]
fn the_corpus_comes_back() {
    let Some(dir) = std::env::var_os("GLYD_WEIGHTS_CORPUS") else {
        eprintln!("GLYD_WEIGHTS_CORPUS not set: the real files of the corpus are not read");
        return;
    };
    let mut files: Vec<_> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).filter(|p| matches!(p.extension().and_then(|e| e.to_str()), Some("safetensors") | Some("gguf"))).collect();
    files.sort();
    assert!(!files.is_empty());
    for p in files {
        let f = std::fs::read(&p).unwrap();
        let mut c = Vec::new();
        glyd::compress_into_max(&f, &mut c);
        assert!(c.starts_with(MAGIC), "{}", p.display());
        all_ways_back(&c, &f, &p.display().to_string());
        assert!(c.len() < f.len() * 95 / 100, "{}: {} of {}", p.display(), c.len(), f.len());
    }
}

#[test]
fn earlier_releases_files_still_decode() {
    // written by v0.28.0 at the max level: a safetensors file as byte planes (the container route) and a GGUF file as it was
    let dir = format!("{}/tests/data/weights", env!("CARGO_MANIFEST_DIR"));
    for name in ["small.safetensors", "small.gguf"] {
        let original = std::fs::read(format!("{dir}/{name}")).unwrap();
        let packed = std::fs::read(format!("{dir}/{name}.v0280.glyd")).unwrap();
        assert!(!packed.starts_with(MAGIC), "{name}: the fixture is meant to be an earlier release's");
        assert_eq!(glyd::decompress(&packed).unwrap(), original, "{name}");
        assert_eq!(glyd::decompress_parallel(&packed).unwrap(), original, "{name}");
    }
}

/// Writes the two files `earlier_releases_files_still_decode` reads (their packed forms are made by the released
/// glyd, `glyd --max`, v0.28.0): `GLYD_WRITE_FIXTURES=tests/data/weights cargo test --test weights write_fixtures -- --ignored`.
#[test]
#[ignore]
fn write_fixtures() {
    let dir = std::env::var("GLYD_WRITE_FIXTURES").expect("GLYD_WRITE_FIXTURES names the directory");
    let mut r = Rng::new(28);
    let w = gauss(&mut r, 24_000, 0.02);
    let st = safetensors(&[("small.weight", "BF16", vec![12_000], as_bf16(&w[..12_000])), ("small.norm", "F32", vec![6_000], as_f32(&w[12_000..18_000])), ("small.q", "F8_E4M3", vec![6_000], as_fp8(&w[18_000..].iter().map(|x| x * 80.0).collect::<Vec<_>>(), true))]);
    std::fs::write(format!("{dir}/small.safetensors"), st).unwrap();
    let gg = gguf(&[("blk.0.attn_q.weight", 8, 1024, 8, q8_0(&w[..8 * 1024])), ("blk.0.ffn.weight", 30, 1024, 8, as_bf16(&w[8 * 1024..16 * 1024])), ("output_norm.weight", 0, 1024, 1, as_f32(&w[16 * 1024..17 * 1024]))]);
    std::fs::write(format!("{dir}/small.gguf"), gg).unwrap();
}
