// Scratch: model weights in safetensors form, transformed so a codec can
// see their structure; the output files are then compressed by any CLI.
//   ai_probe info  <file.safetensors>
//   ai_probe planes <file.safetensors> <out>        every tensor as byte planes
//   ai_probe entropy <file.safetensors>   order-0 and conditioned entropy per plane
//   ai_probe xor <prev.safetensors> <cur.safetensors> <out>
//                     every tensor XOR the same tensor of `prev`, as byte planes
// The header (8-byte length and JSON) is kept as it is ahead of the data.
use std::collections::HashMap;

struct Tensor {
    name: String,
    dtype: String,
    start: usize,
    end: usize,
}

fn width(dtype: &str) -> usize {
    match dtype {
        "F64" | "I64" | "U64" => 8,
        "F32" | "I32" | "U32" => 4,
        "F16" | "BF16" | "I16" | "U16" => 2,
        _ => 1,
    }
}

/// The tensors of a safetensors file (a minimal reader of its JSON header:
/// an object of objects with "dtype" and "data_offsets"), and the data's
/// offset in the file.
fn tensors(file: &[u8]) -> (Vec<Tensor>, usize) {
    let n = u64::from_le_bytes(file[..8].try_into().unwrap()) as usize;
    let h = std::str::from_utf8(&file[8..8 + n]).unwrap();
    let b = h.as_bytes();
    let mut out = Vec::new();
    let mut i = 1usize; // past the outer '{'
    loop {
        while i < b.len() && (b[i] == b',' || b[i].is_ascii_whitespace()) {
            i += 1;
        }
        if i >= b.len() || b[i] == b'}' {
            break;
        }
        // "name": { ... }
        let k0 = i + 1;
        let k1 = k0 + h[k0..].find('"').unwrap();
        let name = h[k0..k1].to_string();
        let o0 = k1 + h[k1..].find('{').unwrap();
        let mut depth = 0;
        let mut j = o0;
        let mut in_str = false;
        while j < b.len() {
            let c = b[j];
            if in_str {
                if c == b'\\' {
                    j += 1;
                } else if c == b'"' {
                    in_str = false;
                }
            } else if c == b'"' {
                in_str = true;
            } else if c == b'{' {
                depth += 1;
            } else if c == b'}' {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            j += 1;
        }
        let obj = &h[o0..=j];
        i = j + 1;
        if name == "__metadata__" {
            continue;
        }
        let dtype = obj.split("\"dtype\"").nth(1).and_then(|s| s.split('"').nth(1)).unwrap().to_string();
        let offs = obj.split("\"data_offsets\"").nth(1).unwrap();
        let offs = &offs[offs.find('[').unwrap() + 1..offs.find(']').unwrap()];
        let v: Vec<usize> = offs.split(',').map(|x| x.trim().parse().unwrap()).collect();
        out.push(Tensor { name, dtype, start: v[0], end: v[1] });
    }
    out.sort_by_key(|t| t.start);
    (out, 8 + n)
}

fn planes(data: &[u8], w: usize, out: &mut Vec<u8>) {
    let n = data.len() / w;
    for j in 0..w {
        for i in 0..n {
            out.push(data[i * w + j]);
        }
    }
    out.extend_from_slice(&data[n * w..]);
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a[1].as_str() {
        "info" => {
            let f = std::fs::read(&a[2]).unwrap();
            let (ts, data_at) = tensors(&f);
            let mut by: HashMap<String, (usize, usize)> = HashMap::new();
            for t in &ts {
                let e = by.entry(t.dtype.clone()).or_insert((0, 0));
                e.0 += 1;
                e.1 += t.end - t.start;
            }
            println!("{}: {} B, header {} B, {} tensors, by dtype {:?}", a[2].rsplit('/').next().unwrap(), f.len(), data_at, ts.len(), by);
        }
        "planes" => {
            let f = std::fs::read(&a[2]).unwrap();
            let (ts, data_at) = tensors(&f);
            let mut out = Vec::with_capacity(f.len());
            out.extend_from_slice(&f[..data_at]);
            let mut at = data_at;
            for t in &ts {
                out.extend_from_slice(&f[at..data_at + t.start]);
                planes(&f[data_at + t.start..data_at + t.end], width(&t.dtype), &mut out);
                at = data_at + t.end;
            }
            out.extend_from_slice(&f[at..]);
            std::fs::write(&a[3], &out).unwrap();
        }
        "xor" => {
            let p = std::fs::read(&a[2]).unwrap();
            let f = std::fs::read(&a[3]).unwrap();
            let (pts, pdata) = tensors(&p);
            let prev: HashMap<&str, &Tensor> = pts.iter().map(|t| (t.name.as_str(), t)).collect();
            let (ts, data_at) = tensors(&f);
            let mut out = Vec::with_capacity(f.len());
            out.extend_from_slice(&f[..data_at]);
            let mut at = data_at;
            let (mut matched, mut unmatched) = (0usize, 0usize);
            for t in &ts {
                out.extend_from_slice(&f[at..data_at + t.start]);
                let cur = &f[data_at + t.start..data_at + t.end];
                match prev.get(t.name.as_str()).filter(|q| q.end - q.start == cur.len() && q.dtype == t.dtype) {
                    Some(q) => {
                        let old = &p[pdata + q.start..pdata + q.end];
                        let x: Vec<u8> = cur.iter().zip(old).map(|(a, b)| a ^ b).collect();
                        planes(&x, width(&t.dtype), &mut out);
                        matched += cur.len();
                    }
                    None => {
                        planes(cur, width(&t.dtype), &mut out);
                        unmatched += cur.len();
                    }
                }
                at = data_at + t.end;
            }
            out.extend_from_slice(&f[at..]);
            std::fs::write(&a[4], &out).unwrap();
            eprintln!("xor: {matched} B against the previous tensors, {unmatched} B without one");
        }
        "split" => {
            // Each byte plane of every tensor, concatenated, to its own file <out>.<j>.
            let f = std::fs::read(&a[2]).unwrap();
            let (ts, data_at) = tensors(&f);
            let w_all = ts.iter().map(|t| width(&t.dtype)).max().unwrap();
            let mut outs = vec![Vec::new(); w_all];
            for t in &ts {
                let w = width(&t.dtype);
                let d = &f[data_at + t.start..data_at + t.end];
                for (i, b) in d.iter().enumerate() {
                    outs[i % w].push(*b);
                }
            }
            for (j, o) in outs.iter().enumerate() {
                std::fs::write(format!("{}.{}", a[3], j), o).unwrap();
            }
        }
        "exponents" => {
            // bf16 only: bits a weight needs as sign + mantissa (8 bits
            // raw) + its exponent coded (a) by per-tensor entropy, (b) as
            // a fixed k-bit index into the tensor's 2^k - 1 most common
            // exponents with an escape to the full 8 bits.
            let f = std::fs::read(&a[2]).unwrap();
            let (ts, data_at) = tensors(&f);
            let (mut n_all, mut ent) = (0f64, 0f64);
            let mut fixed = [0f64; 7]; // k = 2..=8
            for t in ts.iter().filter(|t| t.dtype == "BF16") {
                let d = &f[data_at + t.start..data_at + t.end];
                let mut h = [0u64; 256];
                for e in d.chunks_exact(2) {
                    let v = u16::from_le_bytes([e[0], e[1]]);
                    h[((v >> 7) & 0xff) as usize] += 1;
                }
                let n: u64 = h.iter().sum();
                n_all += n as f64;
                ent += h.iter().filter(|&&c| c > 0).map(|&c| -(c as f64) * (c as f64 / n as f64).log2()).sum::<f64>();
                let mut sorted = h;
                sorted.sort_unstable_by(|a, b| b.cmp(a));
                for (i, k) in (2..=8).enumerate() {
                    let covered: u64 = sorted[..(1usize << k) - 1].iter().sum();
                    fixed[i] += (n as f64) * k as f64 + (n - covered) as f64 * 8.0;
                }
            }
            println!("{}: {:.0}M bf16 weights; exponent entropy {:.2} bits, so {:.2} bits a weight", a[2].rsplit('/').next().unwrap(), n_all / 1e6, ent / n_all, 8.0 + ent / n_all);
            for (i, k) in (2..=8).enumerate() {
                println!("  fixed {k}-bit exponent code + escape: {:.2} bits a weight", 8.0 + fixed[i] / n_all);
            }
        }
        "entropy" => {
            // Bytes an ideal per-tensor coder would need: each byte plane
            // order-0; the top plane given the previous top byte; each lower
            // plane given the plane above it at the same value.
            let f = std::fs::read(&a[2]).unwrap();
            let (ts, data_at) = tensors(&f);
            fn h(counts: &[u64]) -> f64 {
                let n: u64 = counts.iter().sum();
                counts.iter().filter(|&&c| c > 0).map(|&c| -(c as f64) * (c as f64 / n as f64).log2()).sum::<f64>() / 8.0
            }
            let mut o0 = [0f64; 8];
            let mut cond = [0f64; 8];
            let mut w_all = 0;
            for t in &ts {
                let w = width(&t.dtype);
                w_all = w_all.max(w);
                let d = &f[data_at + t.start..data_at + t.end];
                let n = d.len() / w;
                for j in 0..w {
                    let mut c0 = vec![0u64; 256];
                    let mut c1 = vec![0u64; 65536];
                    for i in 0..n {
                        let b = d[i * w + j] as usize;
                        c0[b] += 1;
                        let ctx = if j + 1 < w { d[i * w + j + 1] as usize } else if i > 0 { d[(i - 1) * w + j] as usize } else { 0 };
                        c1[ctx * 256 + b] += 1;
                    }
                    o0[j] += h(&c0);
                    cond[j] += c1.chunks(256).map(h).sum::<f64>();
                }
            }
            let t0: f64 = o0.iter().sum();
            let t1: f64 = cond.iter().sum();
            println!("{}: {} B; order-0 per plane {:.0} B ({:?}); conditioned {:.0} B ({:?})", a[2].rsplit('/').next().unwrap(), f.len(), t0,
                o0[..w_all].iter().map(|x| (*x / 1e6).round() as u64).collect::<Vec<_>>(), t1, cond[..w_all].iter().map(|x| (*x / 1e6).round() as u64).collect::<Vec<_>>());
        }
        _ => panic!("info | planes | entropy | xor"),
    }
}
