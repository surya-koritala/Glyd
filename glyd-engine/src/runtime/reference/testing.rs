//! Test helpers: weights both this crate and `tests/runtime/record.py` generate from a tensor's name, and the transformers fixtures.
//!
//! A tensor's values are xorshift64 from the FNV-1a hash of its name (or'd with 1), each lo + (hi - lo) u with u = (r >> 40) / 2^24, in
//! f32 arithmetic; the recorder does the same in numpy float32, so a fixture holds only each tensor's name, shape and range.

use std::path::PathBuf;

pub fn seed(name: &str) -> u64 {
    name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3)) | 1
}

pub fn fill(name: &str, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut r = seed(name);
    (0..n)
        .map(|_| {
            r ^= r << 13;
            r ^= r >> 7;
            r ^= r << 17;
            lo + (hi - lo) * ((r >> 40) as f32 / (1u64 << 24) as f32)
        })
        .collect()
}

/// The largest difference over the largest magnitude of `want`.
pub fn rel(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    assert!(got.iter().all(|v| v.is_finite()), "a value that is not finite");
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
    got.iter().zip(want).fold(0f32, |m, (a, b)| m.max((a - b).abs())) / scale
}

pub fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    let r = rel(got, want);
    assert!(r <= tol, "{what}: off by {r:.3e} of the scale (tolerance {tol:.0e})");
}

/// A transformers fixture (`tests/runtime/NAME.json`, written by `tests/runtime/record.py`).
pub struct Fixture {
    pub j: serde_json::Value,
}

impl Fixture {
    pub fn load(name: &str) -> Fixture {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/runtime").join(format!("{name}.json"));
        let text = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        Fixture { j: serde_json::from_str(&text).unwrap() }
    }

    pub fn usize(&self, key: &str) -> usize {
        self.j[key].as_u64().unwrap_or_else(|| panic!("no {key}")) as usize
    }

    pub fn f32s(&self, key: &str) -> Vec<f32> {
        self.j[key].as_array().unwrap_or_else(|| panic!("no {key}")).iter().map(|v| v.as_f64().unwrap() as f32).collect()
    }

    pub fn positions(&self) -> Vec<u32> {
        self.j["positions"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect()
    }

    /// A generated tensor by its transformers name: (shape, values).
    pub fn tensor(&self, name: &str) -> (Vec<usize>, Vec<f32>) {
        let t = &self.j["tensors"][name];
        assert!(t.is_object(), "the fixture has no tensor {name}");
        let shape: Vec<usize> = t["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let n = shape.iter().product();
        if let Some(ints) = t["ints"].as_array() {
            return (shape, ints.iter().map(|v| v.as_f64().unwrap() as f32).collect());
        }
        (shape, fill(name, n, t["lo"].as_f64().unwrap() as f32, t["hi"].as_f64().unwrap() as f32))
    }

    pub fn input(&self) -> Vec<f32> {
        let t = &self.j["input"];
        fill("input", self.usize("tokens") * self.usize("hidden"), t["lo"].as_f64().unwrap() as f32, t["hi"].as_f64().unwrap() as f32)
    }
}

#[test]
fn the_generator_is_stable() {
    // (the recorder checks the same values)
    assert_eq!(seed("input"), 0x1ebb_ae8f_5810_b65b);
    let v = fill("input", 3, -1.0, 1.0);
    assert!(v.iter().all(|x| (-1.0..1.0).contains(x)) && v[0] != v[1]);
}
