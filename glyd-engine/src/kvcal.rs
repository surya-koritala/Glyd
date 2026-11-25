//! The host's half of calibrating the 12-bit KV pages' tables (src/dynamic/kv12.rs) on the engine itself: the corpus and the sequences made of it, and the windows from the
//! exponents' histograms. The plugin's `glyd_gpu.kvcal` reads the same corpus the same way. The legacy dense engine's tables (their sweep, meta and file format) went with
//! it (R3c, #139); what was left of them here, used by nothing, went with `work_gen` (R4a, #138).

pub const TIERS: [usize; 5] = [4096, 8192, 16384, 32768, 40960]; // the positions a table is calibrated for
pub const WINDOW: usize = 8; // values of exponent >> 1 a base covers
pub const CHUNK: usize = 512; // tokens of a kind of text a sequence takes at a time
pub const LAP: usize = 211; // the offset a stream is read again from when the text is shorter than a sequence

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
            let mut ptr: Vec<(String, usize)> =
                names.iter().map(|k| (k.clone(), if lap == 0 { shift } else { (shift + lap * LAP) % stream(k).len().max(1) })).collect();
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
    fn the_sequences_of_a_calibration() {
        let f = fixtures();
        let s = f.get("sequences").unwrap();
        let docs: Vec<(String, String, Vec<u32>)> = s
            .get("docs")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|d| {
                let d = d.as_array().unwrap();
                (d[0].as_str().unwrap().to_string(), d[1].as_str().unwrap().to_string(), ints(&d[2]).into_iter().map(|x| x as u32).collect())
            })
            .collect();
        for (len, want) in s.get("by_length").unwrap().as_object().unwrap() {
            let got = sequences(&docs, len.parse().unwrap());
            let want = want.as_array().unwrap();
            assert_eq!(got.len(), want.len());
            for ((ids, segs), w) in got.iter().zip(want) {
                let w = w.as_array().unwrap();
                assert_eq!(ids.iter().map(|&x| x as i64).collect::<Vec<_>>(), ints(&w[0]), "ids at {len}");
                let ws: Vec<(String, i64, i64)> = w[1]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|s| {
                        let s = s.as_array().unwrap();
                        (s[0].as_str().unwrap().to_string(), s[1].as_i64().unwrap(), s[2].as_i64().unwrap())
                    })
                    .collect();
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
    }
}
