//! The lossless KV cache's per-model tables, calibrated on the engine itself: the host's half (the corpus and its sequences, the windows from the
//! exponents' histograms, the sweep's figures into the pool's sizing, the tables' meta). The plugin's `glyd_gpu.kvcal` does the same on vLLM; this is its
//! rule (VERSION 8) and the tables it makes are the plugin's: `k` and `v` uint8 [layers, kv heads, channels], `exc` int16 [rows of the embedding matrix, layers] and the meta.
//! The GPU's half (the histograms, the values outside the windows, the sweep) is in kernels/engine.cu and src/calib.rs.

use crate::ffi::{err, Error, Result};

pub const VERSION: u32 = 8; // the tables' format and rule (the plugin's: 8, the file's keys under neutral names)
pub const TIERS: [usize; 5] = [4096, 8192, 16384, 32768, 40960]; // the positions a table is calibrated for
pub const WINDOW: usize = 8; // values of exponent >> 1 a base covers
pub const REFUSED: f64 = 0.0015; // the most of either tensor's values a kind of text may leave outside the windows
pub const INLINE: i64 = 3; // exceptions a head-page holds in its header
pub const CH: i64 = 64; // entries in a chunk of the escape pool
pub const ND: i64 = 16; // the chunks a page's header names
pub const CHUNK: usize = 512; // tokens of a kind of text a sequence takes at a time
pub const LAP: usize = 211; // the offset a stream is read again from when the text is shorter than a sequence
pub const PT: usize = 16; // tokens a page

/// The corpus's sections (the text of kvcorpus.bin: `@@@ kind name` then the body): (kind, name, body).
pub fn parse_corpus(text: &str) -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    for part in text.split("@@@ ").skip(1) {
        let (head, body) = part.split_once('\n').unwrap_or((part, ""));
        let (kind, name) = head.split_once(' ').unwrap_or((head, ""));
        out.push((kind.to_string(), name.to_string(), body.trim().to_string()));
    }
    out
}

/// A kind's segment of a sequence: tokens `start` .. `end`.
pub type Seg = (String, usize, usize);

/// Two sequences of `length` tokens, each a document stream's chunks in turn (a kind's documents one after the other, CHUNK tokens of a kind at a time, the kinds round and round):
/// A in the order of the kinds by name, B in the reverse with each kind's stream entered half a chunk late, so that the same text is at other depths. A text shorter than the
/// sequence is read again from another offset a lap. Returns each sequence's ids and the segments each kind's chunks fill.
pub fn sequences(docs: &[(String, String, Vec<u32>)], length: usize) -> Vec<(Vec<u32>, Vec<Seg>)> {
    let mut streams: Vec<(String, Vec<u32>)> = Vec::new();
    for (kind, _, ids) in docs {
        match streams.iter_mut().find(|s| &s.0 == kind) {
            Some(s) => s.1.extend_from_slice(ids),
            None => streams.push((kind.clone(), ids.clone())),
        }
    }
    let mut names: Vec<String> = streams.iter().map(|s| s.0.clone()).collect();
    names.sort();
    let stream = |k: &str| &streams.iter().find(|s| s.0 == k).unwrap().1;
    let mut out = Vec::new();
    for (order, shift) in [(names.clone(), 0usize), (names.iter().rev().cloned().collect::<Vec<_>>(), CHUNK / 2)] {
        let (mut seq, mut segs, mut lap) = (Vec::new(), Vec::new(), 0usize);
        while seq.len() < length {
            let mut ptr: Vec<(String, usize)> = names
                .iter()
                .map(|k| (k.clone(), if lap == 0 { shift } else { (shift + lap * LAP) % stream(k).len().max(1) }))
                .collect();
            let had = seq.len();
            while seq.len() < length && ptr.iter().any(|(k, p)| *p < stream(k).len()) {
                for kind in &order {
                    let p = ptr.iter_mut().find(|(k, _)| k == kind).unwrap();
                    let s = stream(kind);
                    let from = p.1.min(s.len());
                    let part = &s[from..(p.1 + CHUNK).min(s.len())];
                    let part = &part[..part.len().min(length - seq.len())];
                    if !part.is_empty() {
                        segs.push((kind.clone(), seq.len(), seq.len() + part.len()));
                        seq.extend_from_slice(part);
                        p.1 += CHUNK;
                    }
                }
            }
            if seq.len() == had {
                break;
            }
            lap += 1;
        }
        out.push((seq, segs));
    }
    out
}

/// The start of the window of WINDOW consecutive exponents (of exponent >> 1) holding the most values (the first such) of a histogram of 128 of them.
pub fn best_base(hist: &[i64]) -> u8 {
    debug_assert_eq!(hist.len(), 128);
    let mut sum: i64 = hist[..WINDOW].iter().sum();
    let (mut best, mut at) = (sum, 0usize);
    for s in 1..=(128 - WINDOW) {
        sum += hist[s + WINDOW - 1] - hist[s - 1];
        if sum > best {
            best = sum;
            at = s;
        }
    }
    at as u8
}

/// The entries past its heads' inline ones that a page of 16 of a token needs, from the token's exceptions per head (keys' and values' together): the plugin's `sweep_needs`.
pub fn sweep_need(per_head: &[i64]) -> i32 {
    per_head.iter().map(|&hp| (hp * 16 - INLINE).max(0)).sum::<i64>() as i32
}

/// The sweep's figures per token from need [rows, layers] (the entries a page of 16 of each token needs past its heads' inline ones, in each layer): the chunks a layer-page of the
/// token takes on average over the layers (a page that needs more than `capacity` is made raw: no chunks), the layers in which it needs more than a page holds, and the most entries it
/// needs in any layer. The plugin's `sweep_stats`.
pub fn sweep_stats(need: &[i32], layers: usize, arena: i64, capacity: i64) -> (Vec<f32>, Vec<i32>, Vec<i32>) {
    let rows = need.len() / layers;
    let (mut chunks, mut over, mut maxneed) = (Vec::with_capacity(rows), Vec::with_capacity(rows), Vec::with_capacity(rows));
    for r in 0..rows {
        let (mut sum, mut o, mut mx) = (0i64, 0i32, i32::MIN);
        for &n in &need[r * layers..(r + 1) * layers] {
            let n64 = n as i64;
            if n64 > capacity {
                o += 1;
            } else {
                sum += ((n64 - arena).max(0) + CH - 1) / CH;
            }
            mx = mx.max(n);
        }
        chunks.push(sum as f32 / layers as f32);
        over.push(o);
        maxneed.push(mx);
    }
    (chunks, over, maxneed)
}

/// Python's round(x, 4) of a float32 widened to a double.
pub fn round4(x: f32) -> f64 {
    format!("{:.4}", x as f64).parse().unwrap_or(x as f64)
}

/// torch.quantile of integers as float32, linear: the value at rank p (n - 1) of the sorted ones, the neighbours' interpolation, truncated to an integer (the meta's figures).
pub fn quantile_int(x: &[i64], p: f32) -> i64 {
    if x.is_empty() {
        return 0;
    }
    let mut v: Vec<f32> = x.iter().map(|&a| a as f32).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let rank = p * (v.len() - 1) as f32;
    let (lo, hi) = (rank.floor() as usize, rank.ceil() as usize);
    let w = rank - rank.floor();
    let (a, b) = (v[lo], v[hi]);
    let r = if w < 0.5 { a + w * (b - a) } else { b - (b - a) * (1.0 - w) };
    r as i64
}

/// The store's meta from the sweep's figures (chunks, over, maxneed per row): the plugin's `store_meta` (without its `top` list). `reach(i)`: the tokenizer makes token i from text;
/// `decode(i)`: its string.
#[allow(clippy::too_many_arguments)]
pub fn store_meta(chunks: &[f32], over: &[i32], maxneed: &[i32], arena: i64, capacity: i64, rows: usize, reach: &dyn Fn(usize) -> bool, decode: &dyn Fn(usize) -> String) -> Vec<(String, Json)> {
    let heavy: Vec<usize> = (0..over.len()).filter(|&i| over[i] > 0).collect();
    let text: Vec<usize> = heavy.iter().copied().filter(|&i| reach(i)).collect();
    let ids: Vec<usize> = heavy.iter().copied().filter(|i| !text.contains(i)).collect();
    let (mut worst, mut wi) = (f32::MIN, 0usize);
    for (i, &c) in chunks.iter().enumerate() {
        if c > worst {
            worst = c;
            wi = i;
        }
    }
    let mut sorted = chunks.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[(sorted.len() - 1) / 2];
    let n = |x: usize| Json::Num(x as f64);
    let list = |v: &[usize], k: usize| Json::Arr(v.iter().take(k).map(|&x| n(x)).collect());
    vec![
        ("swept".into(), n(rows)),
        ("bay".into(), Json::Num(arena as f64)),
        ("capacity".into(), Json::Num(capacity as f64)),
        ("worst_chunks_per_layer_leaf".into(), Json::Num(round4(worst))),
        ("worst_token".into(), n(wi)),
        ("median_chunks_per_layer_leaf".into(), Json::Num(round4(median))),
        ("hard".into(), n(heavy.len())),
        ("hard_text".into(), n(text.len())),
        ("hard_ids".into(), n(ids.len())),
        ("hard_text_ids".into(), list(&text, 32)),
        ("hard_text_tokens".into(), Json::Arr(text.iter().take(8).map(|&i| Json::Str(decode(i))).collect())),
        ("hard_ids_sample".into(), list(&ids, 64)),
        ("hard_worst_leaf_need".into(), Json::Num(heavy.iter().map(|&i| maxneed[i]).max().unwrap_or(0) as f64)),
        ("second_layers".into(), Json::Num(over.iter().copied().max().unwrap_or(0) as f64)),
    ]
}

/// A JSON value for the tables' meta (keys written in order given, so callers sort them as Python's sort_keys does).
#[derive(Clone, Debug)]
pub enum Json {
    Num(f64),
    Str(String),
    Bool(bool),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn obj(mut fields: Vec<(String, Json)>) -> Json {
        fields.sort_by(|a, b| a.0.cmp(&b.0));
        Json::Obj(fields)
    }

    pub fn write(&self, out: &mut String) {
        match self {
            Json::Num(x) => {
                if x.fract() == 0.0 && x.abs() < 1e15 {
                    out.push_str(&format!("{}", *x as i64));
                } else {
                    out.push_str(&format!("{x}"));
                }
            }
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Str(s) => {
                out.push('"');
                for c in s.chars() {
                    match c {
                        '"' => out.push_str("\\\""),
                        '\\' => out.push_str("\\\\"),
                        '\n' => out.push_str("\\n"),
                        '\r' => out.push_str("\\r"),
                        '\t' => out.push_str("\\t"),
                        c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                            let mut b = [0u16; 2];
                            for u in c.encode_utf16(&mut b) {
                                out.push_str(&format!("\\u{u:04x}"));
                            }
                        }
                        c => out.push(c),
                    }
                }
                out.push('"');
            }
            Json::Arr(v) => {
                out.push('[');
                for (i, x) in v.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    x.write(out);
                }
                out.push(']');
            }
            Json::Obj(v) => {
                out.push('{');
                for (i, (k, x)) in v.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    Json::Str(k.clone()).write(out);
                    out.push_str(": ");
                    x.write(out);
                }
                out.push('}');
            }
        }
    }

}

impl std::fmt::Display for Json {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        self.write(&mut s);
        f.write_str(&s)
    }
}

/// The arena a page of `h` heads holds without growing: what its header leaves up to a multiple of 128 bytes (15 for 8 heads), from the page's bytes with no arena (the library's).
pub fn free_arena(page_bytes_no_arena: i64, h: i64) -> i64 {
    (page_bytes_no_arena - h * 6144 - 16 * h) / 4 - 1 - ND
}

/// Why a model's tables refuse it: its values fall outside 12 bits too often.
pub fn refusal(k: f64, v: f64) -> String {
    if cfg!(feature = "dev") {
        format!("the model's values compress too poorly (worst kind of text left out: K {:.3}%, V {:.3}%, the limit {:.2}%)", k * 100.0, v * 100.0, REFUSED * 100.0)
    } else {
        "this model's values do not suit the engine's cache, so the engine does not run it".to_string()
    }
}

pub fn check_positions(p: usize) -> Result<usize> {
    TIERS.iter().copied().find(|&t| t >= p).ok_or_else(|| Error(format!("{p} positions: the tables cover up to {}", TIERS[TIERS.len() - 1]))).or_else(|e| if p == 0 { err("0 positions") } else { Err(e) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use glyd_gpu::json::{self, Value};

    fn fixtures() -> Value {
        json::parse(include_str!("../tests/fixtures/kvcal.json")).unwrap()
    }

    fn ints(v: &Value) -> Vec<i64> {
        v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect()
    }

    #[test]
    fn windows_of_the_most_values() {
        let f = fixtures();
        let b = f.get("best_base").unwrap();
        let (h, base) = (b.get("hist").unwrap().as_array().unwrap(), b.get("base").unwrap().as_array().unwrap());
        let mut n = 0;
        for (hi, bi) in h.iter().zip(base) {
            for (hd, bd) in hi.as_array().unwrap().iter().zip(bi.as_array().unwrap()) {
                assert_eq!(best_base(&ints(hd)) as i64, bd.as_i64().unwrap());
                n += 1;
            }
        }
        assert_eq!(n, 12);
    }

    #[test]
    fn the_sweeps_figures_and_the_pools_meta() {
        let f = fixtures();
        let s = f.get("sweep").unwrap();
        let need: Vec<i32> = s.get("need").unwrap().as_array().unwrap().iter().flat_map(ints).map(|x| x as i32).collect();
        let (arena, capacity) = (s.get("bay").unwrap().as_i64().unwrap(), s.get("capacity").unwrap().as_i64().unwrap());
        assert_eq!(arena, free_arena(256 + 8 * 6144, 8));
        assert_eq!(capacity, arena + ND * CH);
        let (chunks, over, maxneed) = sweep_stats(&need, 6, arena, capacity);
        let want = s.get("chunks").unwrap().as_array().unwrap();
        for (a, b) in chunks.iter().zip(want) {
            let b = match b {
                Value::Number(n) => n.parse::<f32>().unwrap(),
                _ => panic!(),
            };
            assert!((a - b).abs() < 1e-6, "{a} {b}");
        }
        assert_eq!(over.iter().map(|&x| x as i64).collect::<Vec<_>>(), ints(s.get("over").unwrap()));
        assert_eq!(maxneed.iter().map(|&x| x as i64).collect::<Vec<_>>(), ints(s.get("maxneed").unwrap()));
        let pool = store_meta(&chunks, &over, &maxneed, arena, capacity, 60, &|i| i % 3 != 0, &|i| format!("<{i}>"));
        let want = s.get("store").unwrap();
        for (k, v) in &pool {
            let w = want.get(k).unwrap_or_else(|| panic!("no {k}"));
            let mut got = String::new();
            v.write(&mut got);
            match (v, w) {
                (Json::Num(x), Value::Number(n)) => assert!((x - n.parse::<f64>().unwrap()).abs() < 1e-9, "{k}: {x} {n}"),
                (Json::Arr(a), Value::Array(b)) => {
                    assert_eq!(a.len(), b.len(), "{k}");
                    for (x, y) in a.iter().zip(b) {
                        match (x, y) {
                            (Json::Num(x), Value::Number(n)) => assert_eq!(*x, n.parse::<f64>().unwrap(), "{k}"),
                            (Json::Str(x), Value::String(y)) => assert_eq!(x, y, "{k}"),
                            _ => panic!("{k}"),
                        }
                    }
                }
                _ => panic!("{k}: {got}"),
            }
        }
    }

    #[test]
    fn quantiles_as_torch() {
        let f = fixtures();
        let q = f.get("quantile").unwrap();
        let x = ints(q.get("x").unwrap());
        for (p, key) in [(0.99f32, "0.99"), (0.999f32, "0.999")] {
            assert_eq!(quantile_int(&x, p), q.get("q").unwrap().get(key).unwrap().as_i64().unwrap());
        }
    }

    #[test]
    fn the_sequences_of_a_calibration() {
        let f = fixtures();
        let s = f.get("sequences").unwrap();
        let docs: Vec<(String, String, Vec<u32>)> = s.get("docs").unwrap().as_array().unwrap().iter().map(|d| {
            let d = d.as_array().unwrap();
            (d[0].as_str().unwrap().to_string(), d[1].as_str().unwrap().to_string(), ints(&d[2]).into_iter().map(|x| x as u32).collect())
        }).collect();
        for (len, want) in s.get("by_length").unwrap().as_object().unwrap() {
            let got = sequences(&docs, len.parse().unwrap());
            let want = want.as_array().unwrap();
            assert_eq!(got.len(), want.len());
            for ((ids, segs), w) in got.iter().zip(want) {
                let w = w.as_array().unwrap();
                assert_eq!(ids.iter().map(|&x| x as i64).collect::<Vec<_>>(), ints(&w[0]), "ids at {len}");
                let ws: Vec<(String, i64, i64)> = w[1].as_array().unwrap().iter().map(|s| {
                    let s = s.as_array().unwrap();
                    (s[0].as_str().unwrap().to_string(), s[1].as_i64().unwrap(), s[2].as_i64().unwrap())
                }).collect();
                assert_eq!(segs.iter().map(|(k, a, b)| (k.clone(), *a as i64, *b as i64)).collect::<Vec<_>>(), ws, "segments at {len}");
            }
        }
    }

    /// The engine's copy of the corpus is the plugin's (where the plugin's file is beside it).
    #[test]
    fn the_corpus_is_the_plugins() {
        if let Ok(t) = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../glyd-gpu-py/glyd_gpu/kvcorpus.bin")) {
            assert!(t == include_bytes!("../data/kvcorpus.bin"), "glyd-engine/data/kvcorpus.bin differs from glyd-gpu-py/glyd_gpu/kvcorpus.bin: copy it again");
        }
    }

    #[test]
    fn the_corpus_parses_as_the_plugin_reads_it() {
        let f = fixtures();
        let want = f.get("corpus").unwrap().get("sections").unwrap().as_array().unwrap();
        let got = parse_corpus(crate::calib::corpus());
        assert_eq!(got.len(), want.len());
        for ((k, n, b), w) in got.iter().zip(want) {
            let w = w.as_array().unwrap();
            assert_eq!((k.as_str(), n.as_str(), b.chars().count() as i64), (w[0].as_str().unwrap(), w[1].as_str().unwrap(), w[2].as_i64().unwrap()));
        }
        let c = f.get("constants").unwrap();
        assert_eq!(c.get("window").unwrap().as_i64().unwrap() as usize, WINDOW);
        assert_eq!(c.get("chunk").unwrap().as_i64().unwrap() as usize, CHUNK);
        assert_eq!(c.get("lap").unwrap().as_i64().unwrap() as usize, LAP);
        assert_eq!(c.get("version").unwrap().as_i64().unwrap() as u32, VERSION);
    }

    #[test]
    fn json_as_python_writes_it() {
        let j = Json::obj(vec![("b".into(), Json::Num(1.0)), ("a".into(), Json::Arr(vec![Json::Num(0.5), Json::Str("\u{e9}\"x".into()), Json::Bool(true)]))]);
        assert_eq!(j.to_string(), format!("{{\"a\": [0.5, \"{}u00e9\\\"x\", true], \"b\": 1}}", '\\'));
    }
}
