//! Random experts in the codes pack's format (glyd-x/1, every matrix two-bit: `cb_pack` reads it, kernels/qwen4x.cu runs it) and the same experts' weights in the
//! model's own basis, for the CPU reference of the kinds: what a random case of the expert cache (`dynamic::qwen4x::load_cached`) is made of.
//!
//! A pack's rows are in the basis of its layer's rotation R x = 512^-0.5 (I (x) H_512) (s * x): natural-order Walsh-Hadamard transforms of 512 values, s the
//! layer's signs (`qx_cb_rot_in`). A first product the kernels make as <u, R x> is <R^T u, x>, and a second product's output R^T (sum of act_n d_n) is the sum
//! of act_n R^T d_n: so an expert's weights in the model's basis are R^T of each of its rows (`weights`).

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

/// `dir` written as a pack of the layers given (each: its rotation's signs and its experts, all kept), every expert two-bit: `meta.json` (format glyd-x/1, each
/// layer's file, signs as 1 and 0, each expert's format "a"), and a layer's file every expert's codebooks and then every expert's codes, each expert's down before its up, as
/// `cb_pack::CbPack::map_layer` reads a layer of two-bit experts.
pub fn write_pack(dir: &Path, hidden: usize, inter: usize, layers: &[(Vec<i8>, Vec<Cb2Expert>)]) -> Result<()> {
    let io = |e: std::io::Error| Error(format!("{}: {e}", dir.display()));
    std::fs::create_dir_all(dir).map_err(io)?;
    let mut meta = serde_json::Map::new();
    for (l, (signs, experts)) in layers.iter().enumerate() {
        let file = format!("layer-{l:03}.safetensors");
        let s: Vec<u8> = signs.iter().map(|&x| (x > 0) as u8).collect();
        // (a format by the pack's letter: "a" the two-bit rows, `Fmt::parse`)
        meta.insert(l.to_string(), serde_json::json!({"file": file, "s": s, "f": vec!["a"; experts.len()]}));
        let as_bytes = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let books: Vec<(String, Vec<u8>, u64)> = experts
            .iter()
            .enumerate()
            .flat_map(|(e, x)| {
                [(format!("e{e:04}.d.1"), as_bytes(&x.down_books), inter as u64), (format!("e{e:04}.u.1"), as_bytes(&x.up_books), 2 * inter as u64)]
            })
            .collect();
        let codes: Vec<(String, &[u8], u64)> = experts
            .iter()
            .enumerate()
            .flat_map(|(e, x)| [(format!("e{e:04}.d.0"), &x.down_codes[..], inter as u64), (format!("e{e:04}.u.0"), &x.up_codes[..], 2 * inter as u64)])
            .collect();
        let mut tensors: Vec<(&str, &str, Vec<u64>, &[u8])> = books.iter().map(|(n, b, r)| (n.as_str(), "BF16", vec![*r, 4], &b[..])).collect();
        tensors.extend(codes.iter().map(|(n, b, r)| (n.as_str(), "U8", vec![*r, hidden as u64 / 4], *b)));
        crate::pack::write_safetensors(&dir.join(&file), &[], &tensors)?;
    }
    let text = serde_json::json!({"format": "glyd-x/1", "layers": meta}).to_string();
    std::fs::write(dir.join("meta.json"), text).map_err(io)
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

    /// A pack this module writes is read by the pack's reader as the expert cache reads it, byte for byte, and the host's computation of an expert from those
    /// bytes in the rotated basis (`cpu_cb`, the expert cache's misses) is the expert of the weights in the model's basis (`Cb2Expert::weights`), rotated back:
    /// the two agree but for the activations' bf16 rounding taking a last-bit difference of the two products' sums the other way (a few neurons at most, each
    /// 2^-8 of its activation), within 1e-3 of the output's largest magnitude, decided before the run.
    #[test]
    fn a_written_pack_is_the_host_s_experts_of_the_weights_in_the_model_s_basis() {
        use crate::cpu_cb::{experts, Fmt, GlmExpert, GlmMat, GlmPair};
        use crate::cpu_experts::Z;
        let (hidden, inter) = (1024, 64);
        let mut rng = Rng(0x2026_1125_0000_00cb);
        let s: Vec<i8> = (0..hidden).map(|_| if rng.sym() < 0.0 { -1 } else { 1 }).collect();
        let layers = vec![(s.clone(), (0..3).map(|_| Cb2Expert::random(hidden, inter, &mut rng)).collect::<Vec<_>>())];
        let dir = std::env::temp_dir().join(format!("glyd-cb-random-{}", std::process::id()));
        write_pack(&dir, hidden, inter, &layers).unwrap();
        let pack = crate::cb_pack::CbPack::open_dims(&dir, hidden, inter).unwrap();
        assert_eq!((pack.kept(0).unwrap(), pack.signs(0).unwrap()), (&[0usize, 1, 2][..], &s[..]));
        let lm = pack.map_layer(0).unwrap();
        let ex = &layers[0].1;
        let x: Vec<f32> = (0..hidden).map(|_| f32::from_bits((bf16_bits(rng.sym()) as u32) << 16)).collect();
        let z = rotate(&x, &s);
        for (e, d) in lm.descs.iter().enumerate() {
            let (cl, bl) = d.bytes();
            // SAFETY: the descriptors point into the layer's mapping (checked by `map_layer`), which `lm` keeps.
            let (codes, books) = unsafe { (std::slice::from_raw_parts(d.codes, cl), std::slice::from_raw_parts(d.books, bl)) };
            let bytes = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
            let (dc, uc) = codes.split_at(inter * hidden / 4);
            let (db, ub) = books.split_at(inter * 8);
            assert!(dc == &ex[e].down_codes[..] && uc == &ex[e].up_codes[..], "expert {e}'s codes");
            assert!(db == &bytes(&ex[e].down_books)[..] && ub == &bytes(&ex[e].up_books)[..], "expert {e}'s codebooks");
            let up = GlmMat { fmt: Fmt::Cb2, rows: 2 * inter, cols: hidden, codes: uc, book: ub };
            let down = GlmMat { fmt: Fmt::Cb2, rows: inter, cols: hidden, codes: dc, book: db };
            let ge = GlmExpert::mixed(up, down, f32::MAX).unwrap();
            let mut out = vec![0f32; hidden];
            experts(Z::F32(&z), &[GlmPair { expert: &ge, token: 0, weight: 1.0 }], &mut out, 1).unwrap();
            let host = rotate_back(&out, &s);
            // (the expert in the model's basis: the gate's and the up's products, the activation rounded to bf16 once, the second product)
            let (gu, dn) = ex[e].weights(&s, hidden, inter);
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
            assert!(scale > 0.0 && off < 1e-3, "expert {e}: {off:.2e} of the output's largest magnitude");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
