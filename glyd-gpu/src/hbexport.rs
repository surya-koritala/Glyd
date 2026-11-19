//! A Hummingbird calibration export (`glyd-hbr-codes/1`, research/lossy-2026/nemotron_hbr.py's `--export DIR`): the codes of a mixture-of-experts model's routed experts as neuron rows, ready to be packed as they are.
//! `DIR/meta.json` has the format, the shape of the rows (`rows` neurons of `row_length` values), the experts of a layer, and for each MoE layer (`layers`, in order) its number among the MoE layers (`moe_layer`), the
//! model's block it is (`block`), the seed and the signs of its rotation (`signs_u8`: 1 is +1, 0 is -1; the saved signs are the authority, a seed may draw otherwise on another torch), the bits of each expert
//! (`expert_bits`, 2 or 3), its file and the file's bytes; `complete` says whether every layer is there. `DIR/layer-{L:03d}.safetensors` (L the MoE layer's number) holds for each expert e and matrix m (`up`, `downT`, the
//! transpose of the down matrix: its neurons' rows) `e{e:03d}.{m}.codes` (U8 [rows, row_length bits / 8], the layout of [`crate::codebook`]: the codes' bitstream, packed), `.scales` (BF16 [rows, row_length / 32]) and
//! `.codebook` (BF16 [rows, 2^bits]). The rows are in the rotated basis of the layer: the packs are these bytes, unchanged.

use crate::codebook::{cb_strides, CbRows};
use crate::json::{self, Value};
use crate::safetensors::File;
use std::io;
use std::path::{Path, PathBuf};

/// meta.json's `format`.
pub const FORMAT: &str = "glyd-hbr-codes/1";

fn bad(why: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, why.to_string())
}

/// A matrix of an expert: its first matrix (`up`, as the model has it) or its second one transposed (`downT`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Matrix {
    Up,
    DownT,
}

impl Matrix {
    pub fn name(self) -> &'static str {
        match self {
            Matrix::Up => "up",
            Matrix::DownT => "downT",
        }
    }
}

/// A MoE layer of an export.
pub struct HbLayer {
    /// Its number among the model's MoE layers, and the model's block it is (`backbone.layers.{block}`).
    pub moe_layer: usize,
    pub block: usize,
    pub seed: u64,
    /// The rotation's signs, +1 or -1 (meta.json's `signs_u8`: 1 is +1, 0 is -1).
    pub signs: Vec<i8>,
    /// The bits (2 or 3) of each expert's codes.
    pub expert_bits: Vec<u32>,
    file: File,
}

/// An export opened: meta.json read and held to what the model has, each layer's file's header read.
pub struct HbExport {
    pub dir: PathBuf,
    pub meta: Value,
    pub recipe: String,
    /// Neuron rows of a matrix and values in a row.
    pub rows: usize,
    pub cols: usize,
    pub experts: usize,
    pub layers: Vec<HbLayer>,
}

impl HbExport {
    /// The export in `dir`: meta.json of the format, complete, its layers in order each with a file of the size meta.json says and the tensors of every expert's two matrices (their dtypes and shapes held to the layout), signs of
    /// 1,024 (any `row_length`: a power of two) as 0 and 1 (255 is read as -1 too), and bits of 2 or 3 for each expert.
    pub fn open(dir: &Path) -> io::Result<HbExport> {
        let path = dir.join("meta.json");
        let text = std::fs::read_to_string(&path).map_err(|e| bad(format!("{}: {e}", path.display())))?;
        let meta = json::parse(&text).map_err(|e| bad(format!("{}: {e}", path.display())))?;
        let what = |why: String| bad(format!("{}: {why}", path.display()));
        match meta.get("format").and_then(Value::as_str) {
            Some(FORMAT) => {}
            other => return Err(what(format!("format {other:?}, where this reads {FORMAT:?}"))),
        }
        if meta.get("complete").and_then(Value::as_bool) != Some(true) {
            return Err(what("not complete (a layer of the export is not written yet)".into()));
        }
        let num = |k: &str| meta.get(k).and_then(Value::as_u64).map(|x| x as usize).filter(|&x| x > 0).ok_or_else(|| what(format!("no {k}")));
        let (rows, cols, experts) = (num("rows")?, num("row_length")?, num("experts")?);
        if !cols.is_multiple_of(32) || !cols.is_power_of_two() {
            return Err(what(format!("rows of {cols} values: a power of two in runs of 32 is wanted")));
        }
        let recs = meta.get("layers").and_then(Value::as_array).ok_or_else(|| what("no layers".into()))?;
        let mut layers = Vec::with_capacity(recs.len());
        for (i, r) in recs.iter().enumerate() {
            let n = |k: &str| r.get(k).and_then(Value::as_u64).ok_or_else(|| what(format!("layer {i}: no {k}")));
            let (moe_layer, block, seed) = (n("moe_layer")? as usize, n("block")? as usize, n("seed")?);
            if moe_layer != i {
                return Err(what(format!("layer {i}: the layers are not in order (moe_layer {moe_layer})")));
            }
            let signs: Vec<i8> = r
                .get("signs_u8")
                .and_then(Value::as_u64s)
                .filter(|s| s.len() == cols)
                .ok_or_else(|| what(format!("layer {i}: signs_u8 are not {cols} numbers")))?
                .into_iter()
                .map(|b| match b {
                    1 => Ok(1i8),
                    0 | 255 => Ok(-1i8),
                    other => Err(what(format!("layer {i}: a sign of {other}, where 1 (+1) and 0 (-1) are wanted"))),
                })
                .collect::<io::Result<_>>()?;
            let expert_bits: Vec<u32> = r
                .get("expert_bits")
                .and_then(Value::as_u64s)
                .filter(|b| b.len() == experts && b.iter().all(|&x| x == 2 || x == 3))
                .ok_or_else(|| what(format!("layer {i}: expert_bits are not {experts} numbers of 2 or 3")))?
                .into_iter()
                .map(|b| b as u32)
                .collect();
            let name = r.get("file").and_then(Value::as_str).filter(|f| !f.is_empty() && !f.contains('/') && *f != "..").ok_or_else(|| what(format!("layer {i}: no file")))?;
            let file = File::open(&dir.join(name))?;
            if r.get("bytes").and_then(Value::as_u64).is_some_and(|b| std::fs::metadata(&file.path).map(|m| m.len()).ok() != Some(b)) {
                return Err(what(format!("layer {i}: {name} is not the {} bytes meta.json says", r.get("bytes").and_then(Value::as_u64).unwrap_or(0))));
            }
            let layer = HbLayer { moe_layer, block, seed, signs, expert_bits, file };
            for e in 0..experts {
                for m in [Matrix::Up, Matrix::DownT] {
                    layer.check(e, m, rows, cols)?;
                }
            }
            layers.push(layer);
        }
        if layers.is_empty() {
            return Err(what("no layers".into()));
        }
        let recipe = meta.get("recipe").and_then(Value::as_str).unwrap_or("").to_string();
        Ok(HbExport { dir: dir.to_path_buf(), meta, recipe, rows, cols, experts, layers })
    }

    /// meta.json without the signs (a layer's 1,024 numbers each): what a saved directory records of how its codes were made.
    pub fn meta_brief(&self) -> Value {
        let Value::Object(top) = &self.meta else { return self.meta.clone() };
        Value::Object(
            top.iter()
                .map(|(k, v)| match (k.as_str(), v) {
                    ("layers", Value::Array(ls)) => (
                        k.clone(),
                        Value::Array(
                            ls.iter()
                                .map(|l| match l {
                                    Value::Object(o) => Value::Object(o.iter().filter(|(k, _)| k != "signs_u8" && k != "expert_bits").cloned().collect()),
                                    other => other.clone(),
                                })
                                .collect(),
                        ),
                    ),
                    _ => (k.clone(), v.clone()),
                })
                .collect(),
        )
    }

    /// The experts of each bit width, over all the layers: (two-bit, three-bit).
    pub fn bit_counts(&self) -> (usize, usize) {
        let threes = self.layers.iter().flat_map(|l| l.expert_bits.iter()).filter(|&&b| b == 3).count();
        (self.layers.len() * self.experts - threes, threes)
    }

    /// Layer `moe_layer`'s expert `e`'s matrix `m` as a pack's arrays (the file's bytes as they are).
    pub fn matrix(&self, moe_layer: usize, e: usize, m: Matrix) -> io::Result<CbRows> {
        let l = self.layers.get(moe_layer).ok_or_else(|| bad(format!("{}: no MoE layer {moe_layer} (it has {})", self.dir.display(), self.layers.len())))?;
        l.read(e, m, self.rows, self.cols)
    }
}

impl HbLayer {
    fn names(e: usize, m: Matrix) -> [String; 3] {
        ["codes", "scales", "codebook"].map(|k| format!("e{e:03}.{}.{k}", m.name()))
    }

    /// The three tensors of expert `e`'s matrix `m` in the file and their dtypes and shapes held to the layout.
    fn check(&self, e: usize, m: Matrix, rows: usize, cols: usize) -> io::Result<()> {
        let bits = *self.expert_bits.get(e).ok_or_else(|| bad(format!("expert {e} of {}", self.expert_bits.len())))?;
        let (c, _, _) = cb_strides(bits, cols);
        let want = [("U8", vec![rows as u64, c as u64]), ("BF16", vec![rows as u64, cols as u64 / 32]), ("BF16", vec![rows as u64, 1 << bits])];
        for (name, (dtype, shape)) in Self::names(e, m).iter().zip(want) {
            let t = self.file.get(name).ok_or_else(|| bad(format!("{}: no tensor {name}", self.file.path.display())))?;
            if t.dtype != dtype || t.shape != shape {
                return Err(bad(format!("{}: {name} is {} {:?}, where {dtype} {shape:?} is wanted (a {bits}-bit expert)", self.file.path.display(), t.dtype, t.shape)));
            }
        }
        Ok(())
    }

    fn read(&self, e: usize, m: Matrix, rows: usize, cols: usize) -> io::Result<CbRows> {
        self.check(e, m, rows, cols)?;
        let bits = self.expert_bits[e];
        let [codes, scales, book] = Self::names(e, m).map(|n| self.file.get(&n).and_then(|t| self.file.read(t).ok()));
        let what = || bad(format!("{}: expert {e} {}: its arrays cannot be read, or are not finite", self.file.path.display(), m.name()));
        CbRows::new(bits, rows, cols, codes.ok_or_else(what)?, scales.ok_or_else(what)?, book.ok_or_else(what)?).ok_or_else(what)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::codebook::{pack_codebook_rows, CbRows};
    use crate::json::Value;

    /// A safetensors file of the named tensors (name, dtype, shape, bytes), back to back in the order given.
    fn write(path: &Path, tensors: &[(String, &str, Vec<u64>, Vec<u8>)]) {
        let mut at = 0u64;
        let mut parts = vec!["\"__metadata__\":{\"format\":\"pt\"}".to_string()];
        for (n, d, s, b) in tensors {
            parts.push(format!("\"{n}\":{{\"dtype\":\"{d}\",\"shape\":[{}],\"data_offsets\":[{at},{}]}}", s.iter().map(u64::to_string).collect::<Vec<_>>().join(","), at + b.len() as u64));
            at += b.len() as u64;
        }
        let mut h = format!("{{{}}}", parts.join(",")).into_bytes();
        h.resize(h.len().next_multiple_of(8), b' ');
        let mut out = (h.len() as u64).to_le_bytes().to_vec();
        out.extend(&h);
        for (_, _, _, b) in tensors {
            out.extend(b);
        }
        std::fs::write(path, out).unwrap();
    }

    /// A synthetic export of `layers` MoE layers (blocks 1, 3, 5, ...) of `experts` experts of `rows` x `cols`, expert e at `bits(layer, e)` bits, written as nemotron_hbr.py's Exporter writes it; returns the matrices it
    /// holds (`[layer][expert][up, downT]`) as the packer makes them from the codes, scales and codebooks it drew.
    pub(crate) fn synthetic(dir: &Path, layers: usize, experts: usize, rows: usize, cols: usize, bits: &dyn Fn(usize, usize) -> u32, seed: u64) -> Vec<Vec<[CbRows; 2]>> {
        std::fs::create_dir_all(dir).unwrap();
        let mut x = seed | 1;
        let mut next = move || {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };
        let bf = |v: f32| ((v.to_bits() + 0x7FFF + ((v.to_bits() >> 16) & 1)) >> 16) as u16;
        let (mut held, mut recs) = (Vec::new(), Vec::new());
        for l in 0..layers {
            let mut tensors = Vec::new();
            let mut mats = Vec::new();
            for e in 0..experts {
                let b = bits(l, e);
                let mut pair = Vec::new();
                for m in ["up", "downT"] {
                    let codes: Vec<u8> = (0..rows * cols).map(|_| (next() % (1 << b)) as u8).collect();
                    let scales: Vec<u16> = (0..rows * cols / 32).map(|_| bf((next() >> 40) as f32 / (1u64 << 24) as f32 * 0.04 - 0.01)).collect();
                    let mut book: Vec<u16> = (0..rows << b).map(|_| bf((next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0)).collect();
                    book.chunks_mut(1 << b).for_each(|r| r.sort_by(|p, q| f32::from_bits((*p as u32) << 16).total_cmp(&f32::from_bits((*q as u32) << 16))));
                    let p = pack_codebook_rows(&codes, &scales, &book, b, rows, cols).unwrap();
                    tensors.push((format!("e{e:03}.{m}.codes"), "U8", vec![rows as u64, (cols * b as usize / 8) as u64], p.codes.clone()));
                    tensors.push((format!("e{e:03}.{m}.scales"), "BF16", vec![rows as u64, cols as u64 / 32], p.scales.clone()));
                    tensors.push((format!("e{e:03}.{m}.codebook"), "BF16", vec![rows as u64, 1 << b], p.codebook.clone()));
                    pair.push(p);
                }
                let [up, down]: [CbRows; 2] = pair.try_into().unwrap();
                mats.push([up, down]);
            }
            let name = format!("layer-{l:03}.safetensors");
            write(&dir.join(&name), &tensors);
            let signs: Vec<String> = (0..cols).map(|_| (next() & 1).to_string()).collect();
            recs.push(format!(
                "{{\"moe_layer\":{l},\"block\":{},\"seed\":{},\"signs_u8\":[{}],\"expert_bits\":[{}],\"file\":\"{name}\",\"bytes\":{},\"sha256\":\"00\"}}",
                2 * l + 1,
                l * 8 + 2,
                signs.join(","),
                (0..experts).map(|e| bits(l, e).to_string()).collect::<Vec<_>>().join(","),
                std::fs::metadata(dir.join(&name)).unwrap().len()
            ));
            held.push(mats);
        }
        std::fs::write(
            dir.join("meta.json"),
            format!("{{\"format\":\"{FORMAT}\",\"recipe\":\"hbr40\",\"complete\":true,\"model\":{{\"repo\":null,\"revision\":null}},\"rows\":{rows},\"row_length\":{cols},\"block\":32,\"experts\":{experts},\"moe_layers\":{layers},\"layers\":[{}]}}", recs.join(",")),
        )
        .unwrap();
        held
    }

    /// A synthetic export is opened and read back: each matrix is the packer's, bit for bit, of the codes and scales and codebook drawn; the signs are +-1 from the 0s and 1s; the bit counts and the brief meta.
    #[test]
    fn an_export_is_read_back() {
        let dir = std::env::temp_dir().join(format!("glyd-gpu-hbexport-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let bits = |l: usize, e: usize| if (e + l) % 3 == 0 { 3 } else { 2 };
        let held = synthetic(&dir, 3, 6, 64, 128, &bits, 11);
        let ex = HbExport::open(&dir).unwrap();
        assert_eq!((ex.layers.len(), ex.experts, ex.rows, ex.cols, ex.recipe.as_str()), (3, 6, 64, 128, "hbr40"));
        for (l, mats) in held.iter().enumerate() {
            assert_eq!((ex.layers[l].block, ex.layers[l].seed), (2 * l + 1, l as u64 * 8 + 2));
            assert!(ex.layers[l].signs.len() == 128 && ex.layers[l].signs.iter().all(|&s| s == 1 || s == -1) && ex.layers[l].signs.contains(&1) && ex.layers[l].signs.contains(&-1));
            for (e, pair) in mats.iter().enumerate() {
                assert_eq!(ex.layers[l].expert_bits[e], bits(l, e));
                assert_eq!(ex.matrix(l, e, Matrix::Up).unwrap(), pair[0]);
                assert_eq!(ex.matrix(l, e, Matrix::DownT).unwrap(), pair[1]);
            }
        }
        let threes = (0..3).flat_map(|l| (0..6).map(move |e| (l, e))).filter(|&(l, e)| bits(l, e) == 3).count();
        assert_eq!(ex.bit_counts(), (18 - threes, threes));
        let brief = ex.meta_brief();
        let l0 = brief.get("layers").and_then(Value::as_array).unwrap()[0].clone();
        assert!(l0.get("signs_u8").is_none() && l0.get("expert_bits").is_none() && l0.get("seed").is_some() && brief.get("recipe").is_some());
        assert!(ex.matrix(3, 0, Matrix::Up).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// What an export may not be: another format, an incomplete one, rows that are not a power of two, a sign that is neither 0 nor 1, bits that are not 2 or 3, a tensor of another dtype or shape than its bits give,
    /// a file of other bytes than meta.json says, layers out of order.
    #[test]
    fn what_is_refused() {
        let dir = std::env::temp_dir().join(format!("glyd-gpu-hbexport-bad-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        synthetic(&dir, 2, 3, 32, 64, &|_, e| 2 + (e % 2) as u32, 5);
        let good = std::fs::read_to_string(dir.join("meta.json")).unwrap();
        let says = |meta: &str, why: &str| {
            std::fs::write(dir.join("meta.json"), meta).unwrap();
            let e = HbExport::open(&dir).err().map(|e| e.to_string()).unwrap_or_default();
            assert!(e.contains(why), "{why:?} not in {e:?}");
        };
        std::fs::write(dir.join("meta.json"), &good).unwrap();
        assert!(HbExport::open(&dir).is_ok());
        says(&good.replace(FORMAT, "glyd-hbr-codes/2"), "format");
        says(&good.replace("\"complete\":true", "\"complete\":false"), "not complete");
        says(&good.replace("\"row_length\":64", "\"row_length\":96"), "a power of two");
        says(&good.replacen("\"signs_u8\":[", "\"signs_u8\":[7,", 1), "signs_u8");
        says(&good.replacen("\"expert_bits\":[2,3,2]", "\"expert_bits\":[2,4,2]", 1), "expert_bits");
        says(&good.replacen("\"moe_layer\":1", "\"moe_layer\":0", 1), "in order");
        says(&good.replacen("\"bytes\":", "\"bytes\":1", 1), "bytes meta.json says");
        // a tensor of the wrong bits: expert 1 is 3 bits in the file but meta.json says 2
        says(&good.replacen("\"expert_bits\":[2,3,2]", "\"expert_bits\":[2,2,2]", 1), "a 2-bit expert");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
