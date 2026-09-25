//! Model weights in safetensors form: an 8-byte little-endian header
//! length, a JSON header naming each tensor's dtype and byte range in
//! the data that follows. Each tensor of 2-, 4- or 8-byte elements is
//! opened as byte planes (all first bytes, then all second bytes, ...):
//! a float's exponent bytes together, its mantissa bytes together, so
//! the codec sees each plane's own statistics. No dependency: the
//! header reader knows the JSON the format uses and nothing more.

/// A tensor's data: `start..end` in the file, `width` bytes an element.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tensor {
    pub start: usize,
    pub end: usize,
    pub width: usize,
}

/// Headers past this are not read (the format's own writers cap theirs
/// at 100 MB).
const HEADER_LIMIT: usize = 100 << 20;

pub fn is_safetensors(input: &[u8]) -> bool {
    let Some(n) = input.get(..8).map(|b| u64::from_le_bytes(b.try_into().unwrap()) as usize) else { return false };
    n >= 2 && n <= HEADER_LIMIT && 8 + n <= input.len() && input[8] == b'{'
}

/// The tensors of `input`, in file order, non-overlapping and inside
/// the data; `None` for anything else.
pub fn tensors(input: &[u8]) -> Option<Vec<Tensor>> {
    if !is_safetensors(input) {
        return None;
    }
    let n = u64::from_le_bytes(input[..8].try_into().unwrap()) as usize;
    let data_at = 8 + n;
    let mut p = Json { b: &input[8..data_at], i: 0 };
    let mut out = Vec::new();
    p.expect(b'{')?;
    if !p.eat(b'}') {
        loop {
            let name = p.string()?;
            p.expect(b':')?;
            if name == b"__metadata__" {
                p.skip_value(0)?;
            } else {
                let (dtype, offsets) = p.tensor()?;
                let width = match dtype.as_slice() {
                    b"F64" | b"I64" | b"U64" => 8,
                    b"F32" | b"I32" | b"U32" => 4,
                    b"F16" | b"BF16" | b"I16" | b"U16" => 2,
                    _ => 1,
                };
                let (s, e) = offsets;
                let (start, end) = (data_at.checked_add(s)?, data_at.checked_add(e)?);
                if s > e || end > input.len() {
                    return None;
                }
                out.push(Tensor { start, end, width });
            }
            if p.eat(b',') {
                continue;
            }
            p.expect(b'}')?;
            break;
        }
    }
    out.sort_unstable_by_key(|t| t.start);
    out.windows(2).all(|w| w[0].end <= w[1].start).then_some(out)
}

struct Json<'a> {
    b: &'a [u8],
    i: usize,
}

impl Json<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn eat(&mut self, c: u8) -> bool {
        self.ws();
        if self.b.get(self.i) == Some(&c) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, c: u8) -> Option<()> {
        self.eat(c).then_some(())
    }
    /// A string's raw bytes (escapes left as they are: names are compared
    /// only against plain ASCII).
    fn string(&mut self) -> Option<Vec<u8>> {
        self.expect(b'"')?;
        let from = self.i;
        loop {
            match *self.b.get(self.i)? {
                b'\\' => self.i += 2,
                b'"' => break,
                _ => self.i += 1,
            }
        }
        let s = self.b.get(from..self.i)?.to_vec();
        self.i += 1;
        Some(s)
    }
    fn number(&mut self) -> Option<usize> {
        self.ws();
        let from = self.i;
        let mut v = 0usize;
        while let Some(d) = self.b.get(self.i).filter(|c| c.is_ascii_digit()) {
            v = v.checked_mul(10)?.checked_add((d - b'0') as usize)?;
            self.i += 1;
        }
        (self.i > from).then_some(v)
    }
    /// Any value, passed over.
    fn skip_value(&mut self, depth: u32) -> Option<()> {
        if depth > 64 {
            return None;
        }
        self.ws();
        match *self.b.get(self.i)? {
            b'"' => {
                self.string()?;
            }
            b'{' | b'[' => {
                let close = if self.b[self.i] == b'{' { b'}' } else { b']' };
                self.i += 1;
                if self.eat(close) {
                    return Some(());
                }
                loop {
                    if close == b'}' {
                        self.string()?;
                        self.expect(b':')?;
                    }
                    self.skip_value(depth + 1)?;
                    if self.eat(b',') {
                        continue;
                    }
                    self.expect(close)?;
                    break;
                }
            }
            _ => {
                let from = self.i;
                while self.i < self.b.len() && !matches!(self.b[self.i], b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r') {
                    self.i += 1;
                }
                if self.i == from {
                    return None;
                }
            }
        }
        Some(())
    }
    /// A tensor's entry: its dtype and data offsets.
    fn tensor(&mut self) -> Option<(Vec<u8>, (usize, usize))> {
        self.expect(b'{')?;
        let (mut dtype, mut offsets) = (None, None);
        if !self.eat(b'}') {
            loop {
                let key = self.string()?;
                self.expect(b':')?;
                match key.as_slice() {
                    b"dtype" => dtype = Some(self.string()?),
                    b"data_offsets" => {
                        self.expect(b'[')?;
                        let s = self.number()?;
                        self.expect(b',')?;
                        let e = self.number()?;
                        self.expect(b']')?;
                        offsets = Some((s, e));
                    }
                    _ => self.skip_value(0)?,
                }
                if self.eat(b',') {
                    continue;
                }
                self.expect(b'}')?;
                break;
            }
        }
        Some((dtype?, offsets?))
    }
}

/// `data` (whole elements of `w` bytes) as byte planes.
pub fn split(data: &[u8], w: usize, out: &mut Vec<u8>) {
    let n = data.len() / w;
    let at = out.len();
    out.resize(at + n * w, 0);
    let dst = &mut out[at..];
    for (i, e) in data.chunks_exact(w).enumerate() {
        for (j, &b) in e.iter().enumerate() {
            dst[j * n + i] = b;
        }
    }
}

/// Byte planes back to elements of `w` bytes.
pub fn join(planes: &[u8], w: usize) -> Option<Vec<u8>> {
    if w == 0 || planes.len() % w != 0 {
        return None;
    }
    let n = planes.len() / w;
    let mut out = vec![0u8; planes.len()];
    for (i, e) in out.chunks_exact_mut(w).enumerate() {
        for (j, b) in e.iter_mut().enumerate() {
            *b = planes[j * n + i];
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(header: &str, data: &[u8]) -> Vec<u8> {
        let mut f = (header.len() as u64).to_le_bytes().to_vec();
        f.extend_from_slice(header.as_bytes());
        f.extend_from_slice(data);
        f
    }

    #[test]
    fn reads_the_header() {
        let h = r#"{"__metadata__":{"format":"pt","a":"{\"x\":1}"},"b":{"dtype":"BF16","shape":[2],"data_offsets":[8,12]}, "a" : {"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let f = file(h, &[0u8; 12]);
        let at = 8 + h.len();
        assert_eq!(tensors(&f).unwrap(), vec![Tensor { start: at, end: at + 8, width: 4 }, Tensor { start: at + 8, end: at + 12, width: 2 }]);
    }

    #[test]
    fn refuses_what_is_not_one() {
        assert!(tensors(b"not a safetensors file at all").is_none());
        let h = r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,80]}}"#;
        assert!(tensors(&file(h, &[0u8; 8])).is_none(), "data past the end");
        let h = r#"{"a":{"dtype":"F32","data_offsets":[0,8]},"b":{"dtype":"F32","data_offsets":[4,12]}}"#;
        assert!(tensors(&file(h, &[0u8; 12])).is_none(), "overlapping tensors");
        for cut in 1..40 {
            let h = r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
            let _ = tensors(&file(&h[..cut], &[0u8; 8]));
        }
    }

    #[test]
    fn planes_come_back() {
        let data: Vec<u8> = (0..4000u32).map(|i| (i * 2654435761u32 >> 13) as u8).collect();
        for w in [2, 4, 8] {
            let mut p = Vec::new();
            split(&data, w, &mut p);
            assert_eq!(join(&p, w).unwrap(), data);
        }
    }
}
