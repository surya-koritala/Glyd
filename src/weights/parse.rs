//! The tensors of a model file as runs of bytes the coder knows the shape of: a
//! safetensors file's (the dtype its header gives) and a GGUF file's (the type
//! its tensor info gives). A header that does not parse, a tensor that does not
//! fit the file, or a type this knows nothing of leaves those bytes to the
//! generic path; nothing here can make a file unreadable.

use super::kinds::{Kind, Spec};

/// A run of bytes of a file and what it is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Cand {
    pub start: usize,
    pub len: usize,
    pub spec: Spec,
}

/// Runs shorter than this are left to the generic path.
pub(crate) const MIN_RUN: usize = 4096;

pub(crate) fn is_gguf(input: &[u8]) -> bool {
    input.len() >= 24 && &input[..4] == b"GGUF" && matches!(u32::from_le_bytes(input[4..8].try_into().unwrap()), 2 | 3)
}

/// The runs of a safetensors or GGUF file, in file order and apart; `None` for any other file.
pub(crate) fn candidates(input: &[u8]) -> Option<Vec<Cand>> {
    let mut v = if is_gguf(input) { gguf(input)? } else { safetensors(input)? };
    v.retain(|c| c.len >= MIN_RUN);
    v.sort_by_key(|c| c.start);
    if v.windows(2).any(|w| w[0].start + w[0].len > w[1].start) {
        return None;
    }
    Some(v)
}

fn cand(start: usize, len: usize, spec: Spec) -> Option<Cand> {
    let ub = spec.unit_bytes();
    let len = len - len % ub;
    (len > 0).then_some(Cand { start, len, spec })
}

fn safetensors(input: &[u8]) -> Option<Vec<Cand>> {
    let mut out = Vec::new();
    for (t, dtype) in crate::safetensors::tensors_typed(input)? {
        let spec = match dtype.as_slice() {
            b"BF16" => Spec::new(Kind::Bf16),
            b"F16" => Spec::new(Kind::F16),
            b"F32" | b"I32" | b"U32" => Spec::planes(4),
            b"F64" | b"I64" | b"U64" => Spec::planes(8),
            b"I16" | b"U16" => Spec::planes(2),
            // the 8-bit floats, integers and anything this does not know: bytes
            _ => Spec::new(Kind::Bytes),
        };
        out.extend(cand(t.start, t.end - t.start, spec));
    }
    Some(out)
}

// ------------------------------------------------------------------ GGUF

struct Cur<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.b.get(self.p..self.p.checked_add(n)?)?;
        self.p += n;
        Some(s)
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Option<&'a [u8]> {
        let n = usize::try_from(self.u64()?).ok()?;
        self.take(n)
    }
    /// Past a value of metadata type `ty`.
    fn skip(&mut self, ty: u32, depth: u32) -> Option<()> {
        if depth > 4 {
            return None;
        }
        let scalar = |ty: u32| -> Option<usize> {
            Some(match ty {
                0 | 1 | 7 => 1,
                2 | 3 => 2,
                4 | 5 | 6 => 4,
                10 | 11 | 12 => 8,
                _ => return None,
            })
        };
        match ty {
            8 => {
                self.string()?;
            }
            9 => {
                let et = self.u32()?;
                let n = usize::try_from(self.u64()?).ok()?;
                if let Some(w) = scalar(et) {
                    self.take(n.checked_mul(w)?)?;
                } else {
                    // strings and arrays: at least 8 bytes each
                    if n > self.b.len() / 8 {
                        return None;
                    }
                    for _ in 0..n {
                        self.skip(et, depth + 1)?;
                    }
                }
            }
            t => {
                self.take(scalar(t)?)?;
            }
        }
        Some(())
    }
}

/// (kind, elements per block, bytes per block) of a ggml type this codes.
fn ggml(ty: u32) -> Option<(Spec, u64, u64)> {
    Some(match ty {
        0 => (Spec::planes(4), 1, 4),
        1 => (Spec::new(Kind::F16), 1, 2),
        2 => (Spec::new(Kind::Q4_0), 32, 18),
        3 => (Spec::new(Kind::Q4_1), 32, 20),
        6 => (Spec::new(Kind::Q5_0), 32, 22),
        7 => (Spec::new(Kind::Q5_1), 32, 24),
        8 => (Spec::new(Kind::Q8_0), 32, 34),
        12 => (Spec::new(Kind::Q4K), 256, 144),
        13 => (Spec::new(Kind::Q5K), 256, 176),
        14 => (Spec::new(Kind::Q6K), 256, 210),
        20 => (Spec::new(Kind::Iq4Nl), 32, 18),
        23 => (Spec::new(Kind::Iq4Xs), 256, 136),
        24 => (Spec::new(Kind::Bytes), 1, 1),
        25 => (Spec::planes(2), 1, 2),
        26 => (Spec::planes(4), 1, 4),
        27 | 28 => (Spec::planes(8), 1, 8),
        30 => (Spec::new(Kind::Bf16), 1, 2),
        39 => (Spec::new(Kind::Mxfp4), 32, 17),
        _ => return None,
    })
}

fn gguf(input: &[u8]) -> Option<Vec<Cand>> {
    let mut c = Cur { b: input, p: 8 };
    let n_tensors = c.u64()?;
    let n_kv = c.u64()?;
    // every entry takes at least 8 bytes
    if n_tensors > (input.len() / 8) as u64 || n_kv > (input.len() / 8) as u64 {
        return None;
    }
    let mut align = 32u64;
    for _ in 0..n_kv {
        let key = c.string()?;
        let ty = c.u32()?;
        if key == b"general.alignment" && ty == 4 {
            align = u32::from_le_bytes(c.b.get(c.p..c.p + 4)?.try_into().unwrap()) as u64;
            if align == 0 || align > 1 << 20 {
                return None;
            }
        }
        c.skip(ty, 0)?;
    }
    let mut infos = Vec::new();
    for _ in 0..n_tensors {
        c.string()?;
        let nd = c.u32()?;
        if nd == 0 || nd > 8 {
            return None;
        }
        let mut elems = 1u64;
        let mut first = 0u64;
        for d in 0..nd {
            let x = c.u64()?;
            if d == 0 {
                first = x;
            }
            elems = elems.checked_mul(x)?;
        }
        let ty = c.u32()?;
        let off = c.u64()?;
        infos.push((first, elems, ty, off));
    }
    let data = (c.p as u64).checked_add(align - 1)? / align * align;
    let mut out = Vec::new();
    for (first, elems, ty, off) in infos {
        let Some((spec, blck, tsz)) = ggml(ty) else { continue };
        if first % blck != 0 || elems % blck != 0 {
            continue;
        }
        let Some(len) = (elems / blck).checked_mul(tsz) else { continue };
        let Some(start) = data.checked_add(off) else { continue };
        let Some(end) = start.checked_add(len) else { continue };
        if end > input.len() as u64 {
            continue;
        }
        out.extend(cand(start as usize, len as usize, spec));
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn st_file(tensors: &[(&str, &str, &[usize], usize)]) -> Vec<u8> {
        // (name, dtype, shape, bytes)
        let mut h = String::from("{\"__metadata__\":{\"format\":\"pt\"}");
        let mut off = 0usize;
        for (name, dt, shape, n) in tensors {
            h += &format!(",\"{}\":{{\"dtype\":\"{}\",\"shape\":{:?},\"data_offsets\":[{},{}]}}", name, dt, shape, off, off + n);
            off += n;
        }
        h += "}";
        while h.len() % 8 != 0 {
            h.push(' ');
        }
        let mut f = (h.len() as u64).to_le_bytes().to_vec();
        f.extend_from_slice(h.as_bytes());
        f.resize(f.len() + off, 7);
        f
    }

    #[test]
    fn safetensors_runs() {
        let f = st_file(&[("a", "BF16", &[4096], 8192), ("b", "F32", &[1024], 4096), ("c", "F8_E4M3", &[10000], 10000), ("d", "BF16", &[8], 16)]);
        let c = candidates(&f).unwrap();
        assert_eq!(c.len(), 3, "the 16-byte tensor is too small");
        assert_eq!(c[0].spec, Spec::new(Kind::Bf16));
        assert_eq!(c[1].spec, Spec::planes(4));
        assert_eq!(c[2].spec, Spec::new(Kind::Bytes));
        assert_eq!(c[1].start, c[0].start + 8192);
    }

    #[test]
    fn what_is_not_a_model_file() {
        assert!(candidates(b"hello world, this is not a model").is_none());
        assert!(candidates(&[]).is_none());
        let mut g = b"GGUF".to_vec();
        g.extend_from_slice(&3u32.to_le_bytes());
        g.extend_from_slice(&u64::MAX.to_le_bytes());
        g.extend_from_slice(&0u64.to_le_bytes());
        assert!(candidates(&g).is_none());
    }
}
