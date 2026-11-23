//! The 12-bit lossless KV cache's tables for the description executor (R4f, #332): for each attention layer a window base per (kv head, channel)
//! of its keys and one of its values (none where its keys are its values, as in Gemma 4's global layers), in that layer's own shape (Gemma 4's
//! are 256 and 512 channels), made from the keys and values the model itself writes. `calibrate` runs prompts through the engine and reads them
//! back from its cache; the rule is kvcal's (`kvcal::best_base` over histograms of exponent >> 1), and each layer's share of values outside its
//! windows is the escape pool's load. The tables of the legacy engine (kv.rs, one shape for every layer) are not these.
use super::{Dyn, Op};
use crate::dev::copy_d2h_async;
use crate::ffi::{err, Error, Result};
use crate::kvcal::{best_base, WINDOW};
use std::path::Path;

pub const FORMAT: &str = "glyd-kv-tables-v1";
pub const VERSION: u64 = 9;

/// One attention layer's windows: `heads` kv heads of `channels`, the keys' [heads, channels] and the values' (empty where `one`: keys that are the values).
#[derive(Clone, Debug, PartialEq)]
pub struct LayerTable {
    pub heads: usize,
    pub channels: usize,
    pub one: bool,
    pub k: Vec<u8>,
    pub v: Vec<u8>,
}

/// A model's tables: one for each layer that has keys and values (None for the others), the positions they were made over, and each layer's share
/// of values (keys and values together) outside its windows on the prompts they were made from.
#[derive(Clone, Debug, PartialEq)]
pub struct Tables {
    pub layers: Vec<Option<LayerTable>>,
    pub positions: u64,
    pub model: String,
    pub outside: Vec<f64>,
    /// Each layer's escape chunks a page of 16 tokens took on those prompts (`page_chunks`): on average, and at most.
    pub chunks: Vec<(f64, u32)>,
}

/// A 12-bit page's bytes for `heads` kv heads of `channels` (two streams, the arena the executor gives a page), by the GPU library's layout
/// (glyd_gpu.cu: a header of 16 bytes a head-page of 128 channels, the page's used word, 16 directory words and the arena, up to a multiple of 128;
/// then 6,144 bytes a head-page). For the planner, which has no library to ask; `Lib::kv_page_bytes` is the same number.
pub fn page_bytes(heads: usize, channels: usize) -> u64 {
    let hp = (heads * channels / 128) as u64;
    let hdr = (16 * hp + 4 * (1 + 16 + super::kvpages::ARENA as u64)).div_ceil(128) * 128;
    hdr + hp * 6144
}

/// The bucket of a bf16 value kvcal's windows are over: its exponent halved.
fn bucket(x: u16) -> usize {
    ((x >> 8) & 127) as usize
}

/// A layer's histograms: [channels of all heads][128] for the keys and for the values.
#[derive(Clone)]
pub struct Hist {
    heads: usize,
    channels: usize,
    one: bool,
    k: Vec<i64>,
    v: Vec<i64>,
}

impl Hist {
    pub fn new(heads: usize, channels: usize, one: bool) -> Hist {
        let c = heads * channels;
        Hist { heads, channels, one, k: vec![0; c * 128], v: if one { Vec::new() } else { vec![0; c * 128] } }
    }

    /// Rows [n][heads channels] of the keys and of the values (none where `one`) counted.
    pub fn add(&mut self, k: &[u16], v: &[u16]) {
        let c = self.heads * self.channels;
        for row in k.chunks_exact(c) {
            for (ch, &x) in row.iter().enumerate() {
                self.k[ch * 128 + bucket(x)] += 1;
            }
        }
        if !self.one {
            for row in v.chunks_exact(c) {
                for (ch, &x) in row.iter().enumerate() {
                    self.v[ch * 128 + bucket(x)] += 1;
                }
            }
        }
    }

    /// The windows: each channel's base, the start of the WINDOW buckets that hold the most of its values.
    pub fn table(&self) -> LayerTable {
        let bases = |h: &[i64]| h.chunks_exact(128).map(best_base).collect::<Vec<u8>>();
        LayerTable { heads: self.heads, channels: self.channels, one: self.one, k: bases(&self.k), v: if self.one { Vec::new() } else { bases(&self.v) } }
    }

    /// The share of the counted values outside `t`'s windows (keys and values together; 0 where none was counted).
    pub fn outside(&self, t: &LayerTable) -> f64 {
        let count = |h: &[i64], b: &[u8]| -> (i64, i64) {
            h.chunks_exact(128).zip(b).fold((0, 0), |(o, n), (row, &base)| {
                let inside: i64 = row[base as usize..(base as usize + WINDOW).min(128)].iter().sum();
                let all: i64 = row.iter().sum();
                (o + all - inside, n + all)
            })
        };
        let (ko, kn) = count(&self.k, &t.k);
        let (vo, vn) = if self.one { (0, 0) } else { count(&self.v, &t.v) };
        if kn + vn == 0 {
            0.0
        } else {
            (ko + vo) as f64 / (kn + vn) as f64
        }
    }
}

impl Tables {
    pub fn save(&self, path: &Path) -> Result<()> {
        let meta = serde_json::json!({
            "version": VERSION, "positions": self.positions, "model": self.model, "outside": self.outside,
            "chunks": self.chunks.iter().map(|c| serde_json::json!([c.0, c.1])).collect::<Vec<_>>(),
            "layers": self.layers.iter().map(|l| l.as_ref().map(|t| serde_json::json!({"heads": t.heads, "channels": t.channels, "one": t.one}))).collect::<Vec<_>>(),
        });
        let mut tensors: Vec<(String, Vec<u64>, &[u8])> = Vec::new();
        for (l, t) in self.layers.iter().enumerate() {
            if let Some(t) = t {
                tensors.push((format!("k.{l}"), vec![t.heads as u64, t.channels as u64], &t.k[..]));
                if !t.one {
                    tensors.push((format!("v.{l}"), vec![t.heads as u64, t.channels as u64], &t.v[..]));
                }
            }
        }
        let refs: Vec<(&str, &str, Vec<u64>, &[u8])> = tensors.iter().map(|(n, s, d)| (n.as_str(), "U8", s.clone(), *d)).collect();
        crate::pack::write_safetensors(path, &[("format", FORMAT.to_string()), ("meta", meta.to_string())], &refs)
    }

    pub fn load(path: &Path) -> Result<Tables> {
        let f = glyd_gpu::safetensors::File::open(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let get = |k: &str| f.metadata.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone());
        if get("format").as_deref() != Some(FORMAT) {
            return err(format!("{}: not {FORMAT}", path.display()));
        }
        let meta: serde_json::Value = serde_json::from_str(&get("meta").unwrap_or_default()).map_err(|e| Error(format!("{}: meta: {e}", path.display())))?;
        if meta["version"].as_u64() != Some(VERSION) {
            return err(format!("{}: tables of version {}, where this engine reads {VERSION}", path.display(), meta["version"]));
        }
        let mut layers = Vec::new();
        for (l, s) in meta["layers"].as_array().ok_or_else(|| Error(format!("{}: no layers", path.display())))?.iter().enumerate() {
            if s.is_null() {
                layers.push(None);
                continue;
            }
            let (heads, channels, one) =
                (s["heads"].as_u64().unwrap_or(0) as usize, s["channels"].as_u64().unwrap_or(0) as usize, s["one"].as_bool().unwrap_or(false));
            let read = |name: String| -> Result<Vec<u8>> {
                let t = f.get(&name).ok_or_else(|| Error(format!("{}: no {name}", path.display())))?;
                if t.shape != [heads as u64, channels as u64] {
                    return err(format!("{}: {name} is {:?}, where the meta says [{heads}, {channels}]", path.display(), t.shape));
                }
                f.read(t).map_err(|e| Error(e.to_string()))
            };
            let k = read(format!("k.{l}"))?;
            let v = if one { Vec::new() } else { read(format!("v.{l}"))? };
            layers.push(Some(LayerTable { heads, channels, one, k, v }));
        }
        let outside = meta["outside"].as_array().map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect()).unwrap_or_default();
        let chunks = meta["chunks"]
            .as_array()
            .map(|a| a.iter().map(|c| (c[0].as_f64().unwrap_or(0.0), c[1].as_u64().unwrap_or(0) as u32)).collect())
            .unwrap_or_default();
        Ok(Tables { layers, positions: meta["positions"].as_u64().unwrap_or(0), model: meta["model"].as_str().unwrap_or("").to_string(), outside, chunks })
    }
}

/// An attention layer's keys and values as the executor writes them: its ring, kv heads, channels, whether it has values of its own, and its
/// sliding window (0: global).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KvShape {
    pub ring: usize,
    pub heads: usize,
    pub channels: usize,
    pub one: bool,
    pub window: usize,
}

/// Each layer's keys and values, where the description writes them in rows of [kv heads][channels] (Gemma 4's preparation; others: None for now).
pub fn shapes(d: &super::ModelDesc) -> Vec<Option<KvShape>> {
    d.layers
        .iter()
        .map(|ops| {
            let window = ops.iter().find_map(|op| if let Op::Attention { window, .. } = *op { Some(window) } else { None }).unwrap_or(0);
            ops.iter().find_map(|op| match *op {
                // (both streams always: where a layer has no values of its own (Gemma 4's global layers) they are its keys' projection normed without
                // a weight and with no rotary, which the cache holds apart from the keys)
                Op::QkvPrep { d, nk, ring, .. } => Some(KvShape { ring, heads: nk, channels: d, one: false, window }),
                _ => None,
            })
        })
        .collect()
}

impl Dyn {
    /// The selected sequence's keys and values of `s.ring` at its tokens 0 .. n as the cache holds them: rows [tokens][kv heads channels] (the
    /// values none where the keys are the values), and the first token they start at (a sliding layer's window, or its ring's, holds the last alone).
    pub(crate) fn kv_rows(&self, s: &KvShape, n: usize) -> Result<(Vec<u16>, Vec<u16>, usize)> {
        let w = s.heads * s.channels;
        let cap = self.ring_cap[s.ring];
        let keep = if s.window > 0 { s.window.min(cap) } else { cap };
        let first = n.saturating_sub(keep);
        let (kc, vc) = self.cache(s.ring);
        // the row of token p: its block's in a paged ring (as the GPU's table names it, which the kernels read), else p modulo the ring
        self.dev.sync()?;
        let table: Option<Vec<u32>> = self.paged.as_ref().map(|pg| pg.gpu_table(self.seq, s.ring)).transpose()?;
        let row_of = |p: usize| -> Result<usize> {
            match &table {
                Some(t) => t.get(p / 16).map(|&b| b as usize * 16 + p % 16).ok_or_else(|| Error(format!("token {p}: past the sequence's blocks"))),
                None => Ok(p % cap),
            }
        };
        self.dev.sync()?;
        let mut k = vec![0u16; (n - first) * w];
        let mut v = if s.one { Vec::new() } else { vec![0u16; (n - first) * w] };
        for (i, p) in (first..n).enumerate() {
            let r = row_of(p)? as u64;
            // SAFETY: row r of the ring's [rows][heads][channels] bf16 (a block of the sequence's, or a place of its ring), into host rows of w values;
            // the stream is synchronized after.
            unsafe {
                copy_d2h_async(k[i * w..].as_mut_ptr() as *mut _, kc + r * (w * 2) as u64, w * 2, self.dev.stream)?;
                if !s.one {
                    copy_d2h_async(v[i * w..].as_mut_ptr() as *mut _, vc + r * (w * 2) as u64, w * 2, self.dev.stream)?;
                }
            }
        }
        self.dev.sync()?;
        Ok((k, v, first))
    }
}

/// The escape chunks each page of 16 tokens of a layer's keys and values takes against table `t`, by the library's rule (glyd_gpu.h): a value outside
/// its window is an entry of its head-page (128 channels), three of a head-page's are inline, the page's arena holds `arena` more, the rest take
/// chunks of 64; a page past 16 chunks is raw (counted 17). Rows [tokens][heads channels] from token `first` (pages at multiples of 16 of the position).
pub fn page_chunks(k: &[u16], v: &[u16], first: usize, t: &LayerTable, arena: usize) -> Vec<u32> {
    let w = t.heads * t.channels;
    let n = k.len() / w.max(1);
    let out = |x: u16, base: u8| {
        let b = bucket(x) as i32 - base as i32;
        !(0..WINDOW as i32).contains(&b)
    };
    let mut pages = Vec::new();
    let mut p = first;
    while p < first + n {
        let end = ((p / 16 + 1) * 16).min(first + n);
        let mut per = vec![0usize; w / 128];
        for row in (p - first)..(end - first) {
            for c in 0..w {
                per[c / 128] += out(k[row * w + c], t.k[c]) as usize + (!t.one && out(v[row * w + c], t.v[c])) as usize;
            }
        }
        let past = per.iter().map(|&e| e.saturating_sub(3)).sum::<usize>().saturating_sub(arena);
        pages.push(past.div_ceil(64).min(17) as u32);
        p = end;
    }
    pages
}

/// The tables of the model `e` runs, from `prompts` (each run as one reply of a token, its keys and values read back from the cache), with each
/// layer's share of values outside its windows. Refused where the description writes no keys this reads (see `shapes`).
pub fn calibrate(e: &mut Dyn, prompts: &[Vec<u32>], model: &str) -> Result<Tables> {
    let shapes = shapes(&e.big.d);
    if shapes.iter().all(Option::is_none) {
        return err(format!("{}: no attention layer whose keys and values this reads (Gemma 4's preparation)", e.big.d.family));
    }
    let mut hists: Vec<Option<Hist>> = shapes.iter().map(|s| s.map(|s| Hist::new(s.heads, s.channels, s.one))).collect();
    // (the pages' chunks are counted on tokens the windows were not made from: windows from each prompt's first half, counted on its second half;
    // the corpus makes few prompts, whatever their length)
    let mut first: Vec<Option<Hist>> = hists.clone();
    let reply = crate::sched::Reply {
        params: crate::sampler::Params { temperature: 0.0, ..Default::default() },
        max_new: 1,
        ignore_eos: true,
        logprobs: None,
        stop_ids: Vec::new(),
    };
    let mut positions = 0;
    for p in prompts {
        crate::conv::Gen::reply(e, p, &reply, &mut |_| true)?;
        positions = positions.max(p.len());
        for ((s, h), f) in shapes.iter().zip(hists.iter_mut()).zip(first.iter_mut()) {
            if let (Some(s), Some(h), Some(f)) = (s, h, f) {
                let (k, v, _) = e.kv_rows(s, p.len())?;
                h.add(&k, &v);
                let cut = (k.len() / (s.heads * s.channels)) / 2 * s.heads * s.channels;
                f.add(&k[..cut], if v.is_empty() { &v } else { &v[..cut] });
            }
        }
    }
    let layers: Vec<Option<LayerTable>> = hists.iter().map(|h| h.as_ref().map(Hist::table)).collect();
    let outside = hists.iter().zip(&layers).map(|(h, t)| h.as_ref().zip(t.as_ref()).map_or(0.0, |(h, t)| h.outside(t))).collect();
    // (the pages' escape chunks with the tables made: the prompts again; values outside their windows gather in a few channels, so a page's need is
    // counted, not taken from the share)
    let held_out: Vec<Option<LayerTable>> = first.iter().map(|h| h.as_ref().map(Hist::table)).collect();
    let mut taken: Vec<Vec<u32>> = vec![Vec::new(); shapes.len()];
    for p in prompts {
        crate::conv::Gen::reply(e, p, &reply, &mut |_| true)?;
        for (l, s) in shapes.iter().enumerate() {
            if let (Some(s), Some(t)) = (s, &held_out[l]) {
                let (k, v, first) = e.kv_rows(s, p.len())?;
                let w = s.heads * s.channels;
                let half = k.len() / w / 2;
                let v2 = if v.is_empty() { &v[..] } else { &v[half * w..] };
                taken[l].extend(page_chunks(&k[half * w..], v2, first + half, t, super::kvpages::ARENA as usize));
            }
        }
    }
    let chunks = taken.iter().map(|c| (c.iter().sum::<u32>() as f64 / c.len().max(1) as f64, c.iter().copied().max().unwrap_or(0))).collect();
    Ok(Tables { layers, positions: positions as u64, model: model.to_string(), outside, chunks })
}

/// A model's tables, from the cache (made for its directory, version and shapes), else made now from the calibration corpus's first `PROMPTS`
/// sequences of `POSITIONS` tokens (the context allowing) and kept there: true where they were made.
pub fn tables_for(e: &mut Dyn, dir: &Path, model: &str) -> Result<(Tables, bool)> {
    const POSITIONS: usize = 4096;
    const PROMPTS: usize = 8;
    let key = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf()).display().to_string();
    let hash = key.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3));
    let path = crate::store::cache_root().join("kv12").join(format!("{}-{hash:016x}.safetensors", model.replace(['/', ':'], "_")));
    let want = shapes(&e.big.d);
    let fits = |t: &Tables| {
        t.layers.len() == want.len()
            && t.layers.iter().zip(&want).all(|(t, s)| match (t, s) {
                (Some(t), Some(s)) => (t.heads, t.channels) == (s.heads, s.channels),
                (None, None) => true,
                _ => false,
            })
    };
    if let Ok(t) = Tables::load(&path) {
        if fits(&t) {
            return Ok((t, false));
        }
    }
    let docs = crate::calib::corpus_docs(&crate::tok::Tok::load(dir)?, None)?;
    let positions = POSITIONS.min(e.ctx.saturating_sub(64));
    let seqs: Vec<Vec<u32>> = crate::kvcal::sequences(&docs, positions).into_iter().map(|(s, _)| s).filter(|s| s.len() == positions).take(PROMPTS).collect();
    if seqs.is_empty() {
        return err(format!("the calibration corpus makes no sequence of {positions} tokens"));
    }
    let t = calibrate(e, &seqs, model)?;
    std::fs::create_dir_all(path.parent().expect("a file in a directory")).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    t.save(&path)?;
    Ok((t, true))
}

/// `glyd-engine kv12cal --dir DIR [--case gemma4] [--level L] [--positions N] [--prompts K] [--out FILE]`: the tables of a model the executor runs, from K
/// sequences of N tokens of the calibration corpus, and each layer's share of values outside its windows (what the 12-bit pages' escape pool holds).
#[cfg(feature = "dev")]
pub fn cmd(a: &crate::args::Args) -> Result<()> {
    let dir = Path::new(a.flag("dir").ok_or_else(|| Error("--dir DIR: the model".into()))?);
    let (case, level) = (a.flag("case").unwrap_or("gemma4"), a.flag("level").unwrap_or("penguin"));
    let (positions, prompts) = (a.num("positions", 2048usize)?, a.num("prompts", 8usize)?);
    let o = super::Opts { ctx: positions + 64, chunk: 256, graph: true, device: 0, layout: crate::logits::level_layout(level)? };
    let mut e = crate::logits::load_dyn(case, dir, &o, level)?;
    let tok = crate::tok::Tok::load(dir)?;
    let docs = crate::calib::corpus_docs(&tok, None)?;
    let seqs: Vec<Vec<u32>> = crate::kvcal::sequences(&docs, positions).into_iter().map(|(s, _)| s).filter(|s| s.len() == positions).take(prompts).collect();
    if seqs.is_empty() {
        return err(format!("the corpus makes no sequence of {positions} tokens"));
    }
    let t0 = std::time::Instant::now();
    let t = calibrate(&mut e, &seqs, &dir.display().to_string())?;
    let shapes = shapes(&e.big.d);
    let mut worst = 0f64;
    for (l, (lt, o)) in t.layers.iter().zip(&t.outside).enumerate() {
        if let (Some(lt), Some(s)) = (lt, shapes[l]) {
            println!(
                "layer {l:>2}: {} heads x {}{}, {}: {:.4}% of its values outside their windows; escape chunks a page {:.2} on average, {} at most",
                lt.heads,
                lt.channels,
                if lt.one { " (keys are the values)" } else { "" },
                if s.window > 0 { format!("window {}", s.window) } else { "global".into() },
                100.0 * o,
                t.chunks.get(l).map_or(0.0, |c| c.0),
                t.chunks.get(l).map_or(0, |c| c.1)
            );
            worst = worst.max(*o);
        }
    }
    println!("{} sequences of {positions} tokens in {:.1} s; the most outside in a layer {:.4}%", seqs.len(), t0.elapsed().as_secs_f64(), 100.0 * worst);
    if let Some(out) = a.flag("out") {
        t.save(Path::new(out))?;
        let back = Tables::load(Path::new(out))?;
        println!("{}: saved and read back {}", out, if back == t { "the same" } else { "DIFFERENT" });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A layer's windows are each channel's busiest WINDOW buckets, the values outside them counted; tables of two shapes (and a layer with none)
    /// save and load as they were.
    #[test]
    fn windows_are_the_busiest_buckets_and_tables_save_as_they_are() {
        let bf = |e: u16| (e << 7) | 0x11; // (a bf16 of exponent e)
        let mut h = Hist::new(1, 2, false);
        // channel 0: exponents 120..127 (buckets 60..63) mostly, one at 200 (bucket 100); channel 1: 10..17
        let mut k = Vec::new();
        for e in 120..128u16 {
            k.extend([bf(e), bf(e - 110)]);
        }
        k.extend([bf(200), bf(12)]);
        h.add(&k, &k);
        let t = h.table();
        // (channel 0's buckets 60..63 lie in every window from 56 to 60, channel 1's 5..8 in those from 1 to 5: the first is taken)
        assert_eq!(t.k, vec![56, 1], "the first busiest window of each channel");
        let o = h.outside(&t);
        assert!((o - 2.0 / 36.0).abs() < 1e-12, "{o}: the two values at exponent 200, of 36");
        let tables = Tables {
            layers: vec![
                Some(t.clone()),
                None,
                Some(LayerTable { heads: 2, channels: 512, one: true, k: (0..1024).map(|i| (i % 7) as u8).collect(), v: Vec::new() }),
            ],
            positions: 4096,
            model: "m".into(),
            outside: vec![o, 0.0, 0.01],
            chunks: vec![(0.5, 3), (0.0, 0), (1.25, 17)],
        };
        let dir = std::env::temp_dir().join(format!("glyd-kv12-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.safetensors");
        tables.save(&p).unwrap();
        assert_eq!(Tables::load(&p).unwrap(), tables);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The page sizes the GPU library gave on an RTX 4080 SUPER (kvshapes, kv12check): 8 heads of 256 (Gemma 4's sliding layers), one of 512
    /// (its global ones), two of 512.
    #[test]
    fn page_bytes_are_the_librarys() {
        assert_eq!((page_bytes(8, 256), page_bytes(1, 512), page_bytes(2, 512)), (98_688, 24_832, 49_408));
    }

    /// A page's chunks by the library's rule: a head-page's first three entries inline, then the arena, then chunks of 64; past 16 chunks raw (17).
    #[test]
    fn a_pages_chunks_are_its_entries_past_inline_and_arena() {
        let t = LayerTable { heads: 1, channels: 256, one: false, k: vec![60; 256], v: vec![60; 256] };
        let inside = (60u16 << 8) | 0x11; // (bucket 60: inside every window from 60)
        let outv = (100u16 << 8) | 0x11;
        let rows = |outs: &[(usize, usize)]| {
            let mut k = vec![inside; 16 * 256];
            for &(row, c) in outs {
                k[row * 256 + c] = outv;
            }
            k
        };
        let v = vec![inside; 16 * 256];
        assert_eq!(page_chunks(&rows(&[]), &v, 0, &t, 15), vec![0]);
        // 3 inline in head-page 0, 3 in head-page 1, 15 in the arena: 21 entries take no chunk; the 22nd takes one
        let fit: Vec<(usize, usize)> = (0..16).map(|r| (r, 0)).chain((0..5).map(|r| (r, 128))).collect();
        assert_eq!(page_chunks(&rows(&fit), &v, 0, &t, 15), vec![0]);
        let one: Vec<(usize, usize)> = fit.iter().copied().chain([(5, 128)]).collect();
        assert_eq!(page_chunks(&rows(&one), &v, 0, &t, 15), vec![1]);
        // every value outside: past 16 chunks, raw
        let all: Vec<(usize, usize)> = (0..16).flat_map(|r| (0..256).map(move |c| (r, c))).collect();
        assert_eq!(page_chunks(&rows(&all), &v, 0, &t, 15), vec![17]);
        // (pages at multiples of 16 of the position: rows from token 8 are two pages' halves)
        assert_eq!(page_chunks(&rows(&[]), &v, 8, &t, 15).len(), 2);
    }
}
