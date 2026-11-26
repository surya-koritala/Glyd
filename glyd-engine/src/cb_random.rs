//! Random experts in the codes pack's format (glyd-x/1: `cb_pack` reads it, kernels/qwen4x.cu runs it) and the same experts' weights in the model's own basis,
//! for the CPU reference of the kinds: what a random case of the expert cache (`dynamic::qwen4x::load_cached`) is made of. An expert's matrices are each of a
//! format of the pack's (`Fmt`): codebook rows of two, three or four bits or ternary (a row its codes and its own table of entries), or format q's trellis rows of
//! two, three or four bits a weight (a row its words and one scale).
//!
//! A pack's rows are in the basis of its layer's rotation R x = 512^-0.5 (I (x) H_512) (s * x): natural-order Walsh-Hadamard transforms of 512 values, s the
//! layer's signs (`qx_cb_rot_in`). A first product the kernels make as <u, R x> is <R^T u, x>, and a second product's output R^T (sum of act_n d_n) is the sum
//! of act_n R^T d_n: so an expert's weights in the model's basis are R^T of each of its rows (`weights`).

use crate::cpu_cb::Fmt;
use crate::ffi::{Error, Result};
use std::path::Path;

/// 512^-0.5 as the kernels have it.
const SCALE: f32 = 0.044_194_173_824_159_216;

/// One 512-point Walsh-Hadamard transform in natural order, unscaled, in place: the kernels' butterflies (`qc_wht512`) in their order.
fn wht512(v: &mut [f32]) {
    let mut h = 1;
    while h < 512 {
        for r in 0..256 {
            let i = (r / h) * 2 * h + r % h;
            let (a, c) = (v[i], v[i + h]);
            v[i] = a + c;
            v[i + h] = a - c;
        }
        h <<= 1;
    }
}

/// R x of one row (its length a multiple of 512, `s` its signs).
pub fn rotate(x: &[f32], s: &[i8]) -> Vec<f32> {
    let mut v: Vec<f32> = x.iter().zip(s).map(|(&a, &b)| a * b as f32).collect();
    for blk in v.chunks_mut(512) {
        wht512(blk);
        blk.iter_mut().for_each(|y| *y *= SCALE);
    }
    v
}

/// R^T y of one row: each block transformed, then signed and scaled (`qx_cb_sum_rot`).
pub fn rotate_back(y: &[f32], s: &[i8]) -> Vec<f32> {
    let mut v = y.to_vec();
    for blk in v.chunks_mut(512) {
        wht512(blk);
    }
    v.iter().zip(s).map(|(&a, &b)| a * b as f32 * SCALE).collect()
}

/// One expert of two-bit rows: `up` [2 inter, hidden] (the gate's rows, then the up's) and `down` [inter, hidden] (row n the weights neuron n's activation
/// multiplies, in the rotated basis), each as its codes ([rows, hidden / 4]: byte b holds codes 4b..4b+3, the low bits first) and its rows' codebooks (four
/// bf16 entries a row).
pub struct Cb2Expert {
    pub up_codes: Vec<u8>,
    pub up_books: Vec<u16>,
    pub down_codes: Vec<u8>,
    pub down_books: Vec<u16>,
}

/// A small generator of random numbers (xorshift), from a seed.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform in [-1, 1).
    pub fn sym(&mut self) -> f32 {
        2.0 * ((self.next() >> 40) as f32 / (1u64 << 24) as f32) - 1.0
    }
}

fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// `rows` two-bit rows of `cols` codes at random: each row's four entries uniform in +-sqrt(3) `std` (bf16, ascending), its codes uniform.
fn random_rows(rows: usize, cols: usize, std: f32, rng: &mut Rng) -> (Vec<u8>, Vec<u16>) {
    let a = std * 3f32.sqrt();
    let mut books = Vec::with_capacity(rows * 4);
    for _ in 0..rows {
        let mut e: Vec<f32> = (0..4).map(|_| a * rng.sym()).collect();
        e.sort_by(f32::total_cmp);
        books.extend(e.iter().map(|&x| bf16_bits(x)));
    }
    let codes = (0..rows * cols / 4).map(|_| (rng.next() >> 32) as u8).collect();
    (codes, books)
}

impl Cb2Expert {
    /// An expert of `inter` neurons over `hidden` at random, its rows' entries of a matrix's usual size (1 / sqrt of the inputs a row takes).
    pub fn random(hidden: usize, inter: usize, rng: &mut Rng) -> Cb2Expert {
        let (up_codes, up_books) = random_rows(2 * inter, hidden, (hidden as f32).powf(-0.5), rng);
        let (down_codes, down_books) = random_rows(inter, hidden, (inter as f32).powf(-0.5), rng);
        Cb2Expert { up_codes, up_books, down_codes, down_books }
    }

    /// Row `r` of a matrix of these codes and books, in the rotated basis.
    pub fn row(codes: &[u8], books: &[u16], cols: usize, r: usize) -> Vec<f32> {
        let (c, b) = (&codes[r * cols / 4..][..cols / 4], &books[r * 4..][..4]);
        (0..cols).map(|j| f32::from_bits((b[((c[j / 4] >> (2 * (j % 4))) & 3) as usize] as u32) << 16)).collect()
    }

    /// The expert's weights in the model's basis for a layer of signs `s`, fp32: gate and up [2 inter, hidden], down [hidden, inter].
    pub fn weights(&self, s: &[i8], hidden: usize, inter: usize) -> (Vec<f32>, Vec<f32>) {
        let gate_up: Vec<f32> = (0..2 * inter).flat_map(|r| rotate_back(&Self::row(&self.up_codes, &self.up_books, hidden, r), s)).collect();
        let mut down = vec![0f32; hidden * inter];
        for n in 0..inter {
            for (i, w) in rotate_back(&Self::row(&self.down_codes, &self.down_books, hidden, n), s).into_iter().enumerate() {
                down[i * inter + n] = w;
            }
        }
        (gate_up, down)
    }
}

/// One expert of rows of any of the pack's formats (`Fmt`): `up` [2 inter, hidden] (the gate's rows, then the up's) and `down` [inter, hidden] (row n the
/// weights neuron n's activation multiplies, in the rotated basis), each its format, its codes ([rows, `Fmt::row_bytes`]) and its rows' entries (`Fmt::entries`
/// bf16 a row: a codebook's, or format q's scale).
pub struct CodesExpert {
    pub up: Fmt,
    pub down: Fmt,
    pub up_codes: Vec<u8>,
    pub up_books: Vec<u16>,
    pub down_codes: Vec<u8>,
    pub down_books: Vec<u16>,
}

/// `rows` rows of `fmt` over `cols` values at random, of the usual size `std`: a codebook row's entries uniform in +-sqrt(3) std (bf16, ascending) and its codes
/// uniform (a ternary byte one of the 243 the pack writes); a format q row's words uniform and its scale std times a factor in [0.75, 1.25] (its states' values
/// are of unit variance, `tq_value` times `TQ_K`). Two-bit rows are `random_rows`' draws.
fn rows_of(fmt: Fmt, rows: usize, cols: usize, std: f32, rng: &mut Rng) -> (Vec<u8>, Vec<u16>) {
    let rb = fmt.row_bytes(cols);
    match fmt {
        Fmt::Cb2 => random_rows(rows, cols, std, rng),
        Fmt::Tq | Fmt::Tq3 | Fmt::Tq4 => {
            let books = (0..rows).map(|_| bf16_bits(std * (1.0 + 0.25 * rng.sym()))).collect();
            ((0..rows * rb).map(|_| (rng.next() >> 32) as u8).collect(), books)
        }
        _ => {
            let (k, a) = (fmt.entries(), std * 3f32.sqrt());
            let mut books = Vec::with_capacity(rows * k);
            for _ in 0..rows {
                let mut e: Vec<f32> = (0..k).map(|_| a * rng.sym()).collect();
                e.sort_by(f32::total_cmp);
                books.extend(e.iter().map(|&x| bf16_bits(x)));
            }
            let byte = |rng: &mut Rng| if fmt == Fmt::Tern { ((rng.next() >> 32) % 243) as u8 } else { (rng.next() >> 32) as u8 };
            ((0..rows * rb).map(|_| byte(rng)).collect(), books)
        }
    }
}

/// Value `t` of row `r` of a matrix of `fmt` over `cols` values (its codes and its rows' entries), in the rotated basis, as the kernels read it: a codebook
/// row's entry of the code at t (two bits: byte t / 4, the low bits first; three: codes 8 g..8 g + 7 the 24-bit little-endian integer of bytes 3 g..3 g + 2; four:
/// byte t / 2, the low nibble first; ternary: byte t / 5, digit t % 5 of c0 + 3 c1 + 9 c2 + 27 c3 + 81 c4), a format q row's state at t (`tq_state_k`) its value
/// (`tq_value`) times `TQ_K` times the row's scale.
pub fn value(fmt: Fmt, codes: &[u8], books: &[u16], cols: usize, r: usize, t: usize) -> f32 {
    let (c, k) = (&codes[r * fmt.row_bytes(cols)..][..fmt.row_bytes(cols)], fmt.entries());
    let entry = |i: usize| f32::from_bits((books[r * k + i] as u32) << 16);
    match fmt {
        Fmt::Cb2 => entry(((c[t / 4] >> (2 * (t % 4))) & 3) as usize),
        Fmt::Cb3 => {
            let g = 3 * (t / 8);
            let v = c[g] as u32 | (c[g + 1] as u32) << 8 | (c[g + 2] as u32) << 16;
            entry(((v >> (3 * (t % 8))) & 7) as usize)
        }
        Fmt::Cb4 => entry(((c[t / 2] >> (4 * (t % 2))) & 15) as usize),
        Fmt::Tern => entry(crate::cpu_simd::tern_digit(c[t / 5], t % 5)),
        Fmt::Tq | Fmt::Tq3 | Fmt::Tq4 => {
            let words: Vec<u32> = c.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            let s = crate::cpu_simd::tq_state_k(&words, fmt.tq_bits().unwrap_or(2), t);
            (crate::cpu_simd::tq_value(s) as f64 * crate::cpu_simd::TQ_K as f64 * entry(0) as f64) as f32
        }
    }
}

impl CodesExpert {
    /// An expert of `inter` neurons over `hidden` at random, its up of `fmts.0` and its down of `fmts.1`, its rows' values of a matrix's usual size (1 / sqrt of
    /// the inputs a row takes). Two-bit matrices are `Cb2Expert::random`'s draws.
    pub fn random(hidden: usize, inter: usize, fmts: (Fmt, Fmt), rng: &mut Rng) -> CodesExpert {
        let (up_codes, up_books) = rows_of(fmts.0, 2 * inter, hidden, (hidden as f32).powf(-0.5), rng);
        let (down_codes, down_books) = rows_of(fmts.1, inter, hidden, (inter as f32).powf(-0.5), rng);
        CodesExpert { up: fmts.0, down: fmts.1, up_codes, up_books, down_codes, down_books }
    }

    /// Row `r` of a matrix of `fmt` (its codes and entries), in the rotated basis: `value` of each (a format q row's words read once).
    pub fn row(fmt: Fmt, codes: &[u8], books: &[u16], cols: usize, r: usize) -> Vec<f32> {
        let Some(k) = fmt.tq_bits() else { return (0..cols).map(|t| value(fmt, codes, books, cols, r, t)).collect() };
        let rb = fmt.row_bytes(cols);
        let words: Vec<u32> = codes[r * rb..][..rb].chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let scale = f32::from_bits((books[r] as u32) << 16) as f64;
        (0..cols).map(|t| (crate::cpu_simd::tq_value(crate::cpu_simd::tq_state_k(&words, k, t)) as f64 * crate::cpu_simd::TQ_K as f64 * scale) as f32).collect()
    }

    /// The expert's weights in the model's basis for a layer of signs `s`, fp32: gate and up [2 inter, hidden], down [hidden, inter].
    pub fn weights(&self, s: &[i8], hidden: usize, inter: usize) -> (Vec<f32>, Vec<f32>) {
        let gate_up: Vec<f32> = (0..2 * inter).flat_map(|r| rotate_back(&Self::row(self.up, &self.up_codes, &self.up_books, hidden, r), s)).collect();
        let mut down = vec![0f32; hidden * inter];
        for n in 0..inter {
            for (i, w) in rotate_back(&Self::row(self.down, &self.down_codes, &self.down_books, hidden, n), s).into_iter().enumerate() {
                down[i * inter + n] = w;
            }
        }
        (gate_up, down)
    }

    /// The pack's name of its formats: one letter where both are of one format, else "X+Y" (X the up's, Y the down's: `cb_pack`'s `parse_fmts`).
    pub fn name(&self) -> String {
        if self.up == self.down {
            self.up.name().to_string()
        } else {
            format!("{}+{}", self.up.name(), self.down.name())
        }
    }
}

impl From<&Cb2Expert> for CodesExpert {
    fn from(x: &Cb2Expert) -> CodesExpert {
        let (up_codes, up_books, down_codes, down_books) = (x.up_codes.clone(), x.up_books.clone(), x.down_codes.clone(), x.down_books.clone());
        CodesExpert { up: Fmt::Cb2, down: Fmt::Cb2, up_codes, up_books, down_codes, down_books }
    }
}

/// `dir` written as a pack of the layers given (each: its rotation's signs and its experts, all kept), every expert two-bit: `write_codes`.
pub fn write_pack(dir: &Path, hidden: usize, inter: usize, layers: &[(Vec<i8>, Vec<Cb2Expert>)]) -> Result<()> {
    let layers: Vec<(Vec<i8>, Vec<CodesExpert>)> = layers.iter().map(|(s, ex)| (s.clone(), ex.iter().map(CodesExpert::from).collect())).collect();
    write_codes(dir, hidden, inter, &layers)
}

/// `dir` written as a pack of the layers given (each: its rotation's signs and its experts, all kept): `meta.json` (format glyd-x/1, each layer's file, signs as 1
/// and 0, each expert's formats by the pack's letters, `CodesExpert::name`; where a matrix is format q's, the version of its rows' constants, 2 where one is
/// q3 or q4, else 1), and a layer's file every expert's entries and then every expert's codes, each expert's down before its up, as `cb_pack::CbPack::map_layer`
/// reads a layer (of two-bit experts alone: end to end; of other formats: placed expert by expert from the header).
pub fn write_codes(dir: &Path, hidden: usize, inter: usize, layers: &[(Vec<i8>, Vec<CodesExpert>)]) -> Result<()> {
    let io = |e: std::io::Error| Error(format!("{}: {e}", dir.display()));
    std::fs::create_dir_all(dir).map_err(io)?;
    let mut meta = serde_json::Map::new();
    let all = || layers.iter().flat_map(|(_, ex)| ex.iter().flat_map(|x| [x.up, x.down]));
    for (l, (signs, experts)) in layers.iter().enumerate() {
        let file = format!("layer-{l:03}.safetensors");
        let s: Vec<u8> = signs.iter().map(|&x| (x > 0) as u8).collect();
        // (a format by the pack's letters: "a" the two-bit rows, `Fmt::parse`)
        let f: Vec<String> = experts.iter().map(CodesExpert::name).collect();
        meta.insert(l.to_string(), serde_json::json!({"file": file, "s": s, "f": f}));
        write_layer(&dir.join(&file), hidden, inter, &experts.iter().enumerate().collect::<Vec<_>>())?;
    }
    let mut v = serde_json::json!({"format": "glyd-x/1", "layers": meta});
    if all().any(Fmt::is_tq) {
        v["q"] = serde_json::json!(if all().any(|f| matches!(f, Fmt::Tq3 | Fmt::Tq4)) { 2 } else { 1 });
    }
    std::fs::write(dir.join("meta.json"), v.to_string()).map_err(io)
}

/// `dir` written as a sidecar of a pack (`cb_pack::CbPack::with_side`: other copies of some of its experts): for each layer given (its number in the pack, and
/// its experts there by their numbers), a file of those experts as a pack's layer holds them (`write_layer`, each named by its number in the pack), and
/// `meta.json` (format glyd-x/1) naming each layer's file, its experts and their formats.
pub fn write_side(dir: &Path, hidden: usize, inter: usize, layers: &[(usize, Vec<(usize, CodesExpert)>)]) -> Result<()> {
    let io = |e: std::io::Error| Error(format!("{}: {e}", dir.display()));
    std::fs::create_dir_all(dir).map_err(io)?;
    let mut meta = serde_json::Map::new();
    for (l, experts) in layers {
        let file = format!("layer-{l:03}.safetensors");
        write_layer(&dir.join(&file), hidden, inter, &experts.iter().map(|(e, x)| (*e, x)).collect::<Vec<_>>())?;
        let (numbers, fmts): (Vec<usize>, Vec<String>) = experts.iter().map(|(e, x)| (*e, x.name())).unzip();
        meta.insert(l.to_string(), serde_json::json!({"file": file, "experts": numbers, "fmts": fmts}));
    }
    std::fs::write(dir.join("meta.json"), serde_json::json!({"format": "glyd-x/1", "layers": meta}).to_string()).map_err(io)
}

/// A layer's file of a pack or a sidecar at `path`: every expert's entries (books), then every expert's codes, each expert's down before its up, each named
/// by its number `e` (`eNNNN.d.1`, `.u.1`, `.d.0`, `.u.0`), as `cb_pack::CbPack::map_layer` reads a layer (of two-bit experts alone: end to end; of other
/// formats: placed expert by expert from the header).
fn write_layer(path: &Path, hidden: usize, inter: usize, experts: &[(usize, &CodesExpert)]) -> Result<()> {
    let as_bytes = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let books: Vec<(String, Vec<u8>, [u64; 2])> = experts
        .iter()
        .flat_map(|&(e, x)| {
            [
                (format!("e{e:04}.d.1"), as_bytes(&x.down_books), [inter as u64, x.down.entries() as u64]),
                (format!("e{e:04}.u.1"), as_bytes(&x.up_books), [2 * inter as u64, x.up.entries() as u64]),
            ]
        })
        .collect();
    let codes: Vec<(String, &[u8], [u64; 2])> = experts
        .iter()
        .flat_map(|&(e, x)| {
            [
                (format!("e{e:04}.d.0"), &x.down_codes[..], [inter as u64, x.down.row_bytes(hidden) as u64]),
                (format!("e{e:04}.u.0"), &x.up_codes[..], [2 * inter as u64, x.up.row_bytes(hidden) as u64]),
            ]
        })
        .collect();
    let mut tensors: Vec<(&str, &str, Vec<u64>, &[u8])> = books.iter().map(|(n, b, sh)| (n.as_str(), "BF16", sh.to_vec(), &b[..])).collect();
    tensors.extend(codes.iter().map(|(n, b, sh)| (n.as_str(), "U8", sh.to_vec(), *b)));
    crate::pack::write_safetensors(path, &[], &tensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rotation is orthonormal and R^T undoes R, in fp32's last places.
    #[test]
    fn the_rotation_is_undone_by_its_transpose() {
        let mut rng = Rng(7);
        let s: Vec<i8> = (0..1024).map(|_| if rng.sym() < 0.0 { -1 } else { 1 }).collect();
        let x: Vec<f32> = (0..1024).map(|_| rng.sym()).collect();
        let r = rotate(&x, &s);
        let back = rotate_back(&r, &s);
        let norm = |v: &[f32]| v.iter().map(|a| a * a).sum::<f32>().sqrt();
        assert!((norm(&r) - norm(&x)).abs() < 1e-4 * norm(&x));
        assert!(x.iter().zip(&back).all(|(a, b)| (a - b).abs() < 1e-5), "R^T R x is not x");
    }

    /// A pack this module writes is read by the pack's reader as the expert cache reads it, byte for byte, in every format alone and in experts of two (each
    /// expert's two matrices of their own formats), and the host's computation of an expert from those bytes in the rotated basis (`cpu_cb`, the expert cache's
    /// misses) is the expert of the weights in the model's basis (`CodesExpert::weights`, every value as `value` decodes it), rotated back: the two agree but for
    /// the activations' bf16 rounding taking a last-bit difference of the two products' sums the other way (a few neurons at most, each 2^-8 of its activation),
    /// within 1e-3 of the output's largest magnitude, decided before the run. A ternary row's groups of 160 values need a width of them: 2,560, the pack's.
    #[test]
    fn a_written_pack_is_the_host_s_experts_of_the_weights_in_the_model_s_basis() {
        use crate::cpu_cb::{experts, GlmExpert, GlmMat, GlmPair};
        use crate::cpu_experts::Z;
        let (hidden, inter) = (2560, 64);
        let mut rng = Rng(0x2026_1125_0000_00cb);
        let s: Vec<i8> = (0..hidden).map(|_| if rng.sym() < 0.0 { -1 } else { 1 }).collect();
        let (a, b, c, t, q, q3, q4) = (Fmt::Cb2, Fmt::Cb3, Fmt::Cb4, Fmt::Tern, Fmt::Tq, Fmt::Tq3, Fmt::Tq4);
        let forms = [(a, a), (b, b), (c, c), (t, t), (q, q), (q3, q3), (q4, q4), (c, q), (t, b), (q3, a), (a, q4)];
        let layers = vec![(s.clone(), forms.iter().map(|&f| CodesExpert::random(hidden, inter, f, &mut rng)).collect::<Vec<_>>())];
        let dir = std::env::temp_dir().join(format!("glyd-cb-random-{}", std::process::id()));
        write_codes(&dir, hidden, inter, &layers).unwrap();
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
        assert_eq!((meta["q"].as_u64(), meta["layers"]["0"]["f"][7].as_str()), (Some(2), Some("c+q")), "format q's version, an expert of two formats");
        let pack = crate::cb_pack::CbPack::open_dims(&dir, hidden, inter).unwrap();
        assert_eq!((pack.kept(0).unwrap().len(), pack.signs(0).unwrap()), (forms.len(), &s[..]));
        let lm = pack.map_layer(0).unwrap();
        let ex = &layers[0].1;
        let x: Vec<f32> = (0..hidden).map(|_| f32::from_bits((bf16_bits(rng.sym()) as u32) << 16)).collect();
        let z = rotate(&x, &s);
        let bytes = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        for (e, d) in lm.descs.iter().enumerate() {
            let x_ = &ex[e];
            assert_eq!((d.up, d.down), (x_.up, x_.down), "expert {e}'s formats");
            let (cl, bl) = d.bytes();
            // SAFETY: the descriptors point into the layer's mapping (checked by `map_layer`), which `lm` keeps.
            let (codes, books) = unsafe { (std::slice::from_raw_parts(d.codes, cl), std::slice::from_raw_parts(d.books, bl)) };
            let (dc, uc) = codes.split_at(inter * x_.down.row_bytes(hidden));
            let (db, ub) = books.split_at(inter * x_.down.entries() * 2);
            assert!(dc == &x_.down_codes[..] && uc == &x_.up_codes[..], "expert {e}'s codes");
            assert!(db == &bytes(&x_.down_books)[..] && ub == &bytes(&x_.up_books)[..], "expert {e}'s entries");
            let up = GlmMat { fmt: x_.up, rows: 2 * inter, cols: hidden, codes: uc, book: ub };
            let down = GlmMat { fmt: x_.down, rows: inter, cols: hidden, codes: dc, book: db };
            let ge = GlmExpert::mixed(up, down, f32::MAX).unwrap();
            let mut out = vec![0f32; hidden];
            experts(Z::F32(&z), &[GlmPair { expert: &ge, token: 0, weight: 1.0 }], &mut out, 1).unwrap();
            let host = rotate_back(&out, &s);
            // (the expert in the model's basis: the gate's and the up's products, the activation rounded to bf16 once, the second product)
            let (gu, dn) = x_.weights(&s, hidden, inter);
            let dot = |w: &[f32]| w.iter().zip(&x).map(|(a, b)| a * b).sum::<f32>();
            let act: Vec<f32> = (0..inter)
                .map(|n| {
                    let (g, u) = (dot(&gu[n * hidden..][..hidden]), dot(&gu[(inter + n) * hidden..][..hidden]));
                    crate::cpu_experts::bf16_round(g / (1.0 + (-g).exp()) * u)
                })
                .collect();
            let want: Vec<f32> = (0..hidden).map(|i| (0..inter).map(|n| dn[i * inter + n] * act[n]).sum()).collect();
            let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
            let off = host.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs())) / scale;
            assert!(scale > 0.0 && off < 1e-3, "expert {e} ({}): {off:.2e} of the output's largest magnitude", x_.name());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A row of any format is `value` of each of its values (a format q row's words read once for the row).
    #[test]
    fn a_row_is_each_of_its_values() {
        let mut rng = Rng(9);
        for f in [Fmt::Cb2, Fmt::Cb3, Fmt::Cb4, Fmt::Tern, Fmt::Tq, Fmt::Tq3, Fmt::Tq4] {
            let x = CodesExpert::random(320, 64, (f, f), &mut rng);
            for r in [0, 17, 127] {
                let row = CodesExpert::row(f, &x.up_codes, &x.up_books, 320, r);
                assert!((0..320).all(|t| row[t].to_bits() == value(f, &x.up_codes, &x.up_books, 320, r, t).to_bits()), "{f:?} row {r}");
            }
        }
    }

    /// Two-bit experts are the draws they were before the other formats were made, and their pack the same bytes: the expert cache's random case
    /// (`logits::qwen4x_cb_random`) is the model its fixtures were recorded from.
    #[test]
    fn two_bit_experts_are_drawn_and_written_as_before() {
        let (hidden, inter) = (1024, 64);
        let (mut r1, mut r2) = (Rng(42), Rng(42));
        let s: Vec<i8> = (0..hidden).map(|i| if i % 3 == 0 { -1 } else { 1 }).collect();
        let old: Vec<Cb2Expert> = (0..3).map(|_| Cb2Expert::random(hidden, inter, &mut r1)).collect();
        let new: Vec<CodesExpert> = (0..3).map(|_| CodesExpert::random(hidden, inter, (Fmt::Cb2, Fmt::Cb2), &mut r2)).collect();
        assert_eq!(r1.0, r2.0, "the same draws");
        let root = std::env::temp_dir().join(format!("glyd-cb-random-two-{}", std::process::id()));
        let (a, b) = (root.join("a"), root.join("b"));
        write_pack(&a, hidden, inter, &[(s.clone(), old)]).unwrap();
        write_codes(&b, hidden, inter, &[(s, new)]).unwrap();
        for f in ["meta.json", "layer-000.safetensors"] {
            assert!(std::fs::read(a.join(f)).unwrap() == std::fs::read(b.join(f)).unwrap(), "{f}");
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A sidecar this module writes (`write_side`) is read by the pack's reader (`CbPack::with_side`): the experts it names are its copies, in its formats and
    /// its bytes, the others the pack's own.
    #[test]
    fn a_written_sidecar_stands_over_the_packs_experts_it_names() {
        let (hidden, inter) = (2560, 64);
        let mut rng = Rng(7);
        let s: Vec<i8> = (0..hidden).map(|i| if i % 5 == 0 { -1 } else { 1 }).collect();
        let own: Vec<CodesExpert> = (0..4).map(|_| CodesExpert::random(hidden, inter, (Fmt::Cb2, Fmt::Cb2), &mut rng)).collect();
        let side: Vec<(usize, CodesExpert)> = [1, 3].map(|e| (e, CodesExpert::random(hidden, inter, (Fmt::Cb3, Fmt::Cb3), &mut rng))).into();
        let root = std::env::temp_dir().join(format!("glyd-cb-side-{}", std::process::id()));
        write_codes(&root, hidden, inter, &[(s, own)]).unwrap();
        write_side(&root.join("hot"), hidden, inter, &[(0, side)]).unwrap();
        let pack = crate::cb_pack::CbPack::open_dims(&root, hidden, inter).unwrap().with_side(&root.join("hot")).unwrap();
        let lm = pack.map_layer(0).unwrap();
        let fmts: Vec<Fmt> = lm.descs.iter().map(|d| d.up).collect();
        assert_eq!(fmts, [Fmt::Cb2, Fmt::Cb3, Fmt::Cb2, Fmt::Cb3]);
        let _ = std::fs::remove_dir_all(&root);
    }
}
