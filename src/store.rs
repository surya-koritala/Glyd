//! A store that compresses across the objects it holds. The redundancy
//! of object storage is between objects — builds, snapshots, dumps and
//! releases that are near-copies of earlier ones — not inside them. On
//! `put`, the object's fingerprints (one sparse anchor in 4 KB) are
//! looked up in the store's table; the stored object sharing the most
//! is its base, and the object is kept as a delta against it (base
//! mode) when that saves a fifth or more of its own size, else alone
//! at the max level. Chains are at most `MAX_DEPTH` long: past that the
//! chain's root is the base, so a read is at most `MAX_DEPTH + 1`
//! decodes. Measured on a 39 GB bucket of images, releases, dumps and
//! events: 4.6x fewer bytes than zstd -3 per object
//! (experiments/research/README.md, section H).
//!
//! On disk, a directory: `objects/<id>` (the stream), `objects/<id>.fp`
//! (the fingerprints, for the table at open) and `index` (one line per
//! object: id, base id or -, depth, raw length, stored length, name).

use std::collections::HashMap;
use std::io::{Error, ErrorKind, Result, Write};
use std::path::{Path, PathBuf};

/// Chains are at most this deep.
pub const MAX_DEPTH: usize = 4;
/// Holders of one fingerprint the table keeps (the most recent).
const HOLDERS: usize = 8;
/// A base is kept when the delta is at most this share of the object
/// compressed alone.
const WORTH_NUM: usize = 4;
const WORTH_DEN: usize = 5;
/// Fingerprints shared with the base as a share of the object's, below
/// which no base is tried.
const MIN_SHARE: f64 = 0.02;

#[derive(Clone, Debug)]
pub struct Entry {
    pub id: u32,
    pub name: String,
    pub base: Option<u32>,
    pub depth: usize,
    pub raw_len: u64,
    pub stored_len: u64,
}

/// The most recent holders of a fingerprint.
#[derive(Clone, Copy, Default)]
struct Holders {
    ids: [u32; HOLDERS],
    len: u8,
}

impl Holders {
    fn push(&mut self, id: u32) {
        if self.len as usize == HOLDERS {
            self.ids.copy_within(1.., 0);
            self.ids[HOLDERS - 1] = id;
        } else {
            self.ids[self.len as usize] = id;
            self.len += 1;
        }
    }
    fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.ids[..self.len as usize].iter().copied()
    }
}

pub struct Store {
    dir: PathBuf,
    entries: Vec<Entry>,
    // ponytail: an in-memory table, ~40 bytes per fingerprint (one per
    // 4 KB stored); an on-disk table when a store passes a terabyte.
    table: HashMap<u64, Holders>,
}

fn bad(msg: &str) -> Error {
    Error::new(ErrorKind::InvalidData, format!("store: {msg}"))
}

fn codec(e: crate::error::CodecError) -> Error {
    Error::new(ErrorKind::InvalidData, e)
}

/// The fingerprints of an object: its sparse anchors whose hash has two
/// more zero bits.
fn fingerprints(data: &[u8]) -> Vec<u64> {
    let mut anchors = Vec::new();
    crate::ldm::sparse_anchors(data, 0, &mut anchors);
    anchors.into_iter().filter(|&(h, _)| h >> 62 == 0).map(|(h, _)| h).collect()
}

impl Store {
    /// Open the store at `dir`, creating it when it does not exist.
    pub fn open(dir: impl AsRef<Path>) -> Result<Store> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(dir.join("objects"))?;
        let mut store = Store { dir, entries: Vec::new(), table: HashMap::new() };
        let index = store.dir.join("index");
        if index.exists() {
            let text = std::fs::read_to_string(&index)?;
            for line in text.lines() {
                let f: Vec<&str> = line.splitn(6, '\t').collect();
                if f.len() != 6 {
                    return Err(bad("index line"));
                }
                let parse = |s: &str| s.parse::<u64>().map_err(|_| bad("index number"));
                let entry = Entry {
                    id: parse(f[0])? as u32,
                    base: if f[1] == "-" { None } else { Some(parse(f[1])? as u32) },
                    depth: parse(f[2])? as usize,
                    raw_len: parse(f[3])?,
                    stored_len: parse(f[4])?,
                    name: f[5].to_string(),
                };
                if entry.id as usize != store.entries.len() {
                    return Err(bad("index order"));
                }
                let fp = std::fs::read(store.fp_path(entry.id))?;
                for h in fp.chunks_exact(8) {
                    store.table.entry(u64::from_le_bytes(h.try_into().unwrap())).or_default().push(entry.id);
                }
                store.entries.push(entry);
            }
        }
        Ok(store)
    }

    fn object_path(&self, id: u32) -> PathBuf {
        self.dir.join("objects").join(id.to_string())
    }

    fn fp_path(&self, id: u32) -> PathBuf {
        self.dir.join("objects").join(format!("{id}.fp"))
    }

    /// The objects, in id order.
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// Raw and stored bytes over the store.
    pub fn stats(&self) -> (u64, u64) {
        self.entries.iter().fold((0, 0), |(r, s), e| (r + e.raw_len, s + e.stored_len))
    }

    /// The stored object the data shares the most fingerprints with,
    /// and the share, when there is one worth trying.
    fn candidate(&self, prints: &[u64]) -> Option<(u32, f64)> {
        let mut hits: HashMap<u32, f64> = HashMap::new();
        for h in prints {
            if let Some(holders) = self.table.get(h) {
                let n = holders.len as f64;
                for id in holders.iter() {
                    *hits.entry(id).or_insert(0.0) += 1.0 / n;
                }
            }
        }
        let top = hits.values().cloned().fold(0.0f64, f64::max);
        // Among those within 5% of the best, the most recent: a copy of
        // a copy shares its fingerprints with both.
        let (id, score) = hits.iter().filter(|(_, &s)| s >= top * 0.95).max_by_key(|(&id, _)| id).map(|(&id, &s)| (id, s))?;
        let share = score / prints.len().max(1) as f64;
        if share < MIN_SHARE {
            return None;
        }
        Some((id, share))
    }

    /// Store `data` under `name`; its id. The object is kept as a delta
    /// against the stored object it most resembles when that pays, else
    /// alone at the max level.
    pub fn put(&mut self, name: &str, data: &[u8]) -> Result<u32> {
        let prints = fingerprints(data);
        let mut alone = Vec::with_capacity(data.len() / 4 + 1024);
        crate::compress_parallel_into_max(data, &mut alone);
        let mut stored = alone;
        let mut base = None;
        if let Some((mut id, _)) = self.candidate(&prints) {
            if self.entries[id as usize].depth >= MAX_DEPTH {
                while let Some(p) = self.entries[id as usize].base {
                    id = p;
                }
            }
            let base_data = self.get(id)?;
            let mut delta = Vec::with_capacity(data.len() / 16 + 1024);
            crate::compress_with_base(&base_data, data, &mut delta, false);
            if delta.len() * WORTH_DEN <= stored.len() * WORTH_NUM {
                stored = delta;
                base = Some(id);
            }
        }
        let id = self.entries.len() as u32;
        let depth = base.map_or(0, |b| self.entries[b as usize].depth + 1);
        std::fs::write(self.object_path(id), &stored)?;
        let mut fp = Vec::with_capacity(prints.len() * 8);
        for h in &prints {
            fp.extend_from_slice(&h.to_le_bytes());
        }
        std::fs::write(self.fp_path(id), &fp)?;
        let entry = Entry { id, name: name.replace(['\t', '\n'], " "), base, depth, raw_len: data.len() as u64, stored_len: stored.len() as u64 };
        let mut index = std::fs::OpenOptions::new().create(true).append(true).open(self.dir.join("index"))?;
        writeln!(index, "{}\t{}\t{}\t{}\t{}\t{}", entry.id, entry.base.map_or("-".to_string(), |b| b.to_string()), entry.depth, entry.raw_len, entry.stored_len, entry.name)?;
        for h in prints {
            self.table.entry(h).or_default().push(id);
        }
        self.entries.push(entry);
        Ok(id)
    }

    /// Object `id` back: its base first, when it has one.
    pub fn get(&self, id: u32) -> Result<Vec<u8>> {
        let entry = self.entries.get(id as usize).ok_or_else(|| bad("no such object"))?;
        let stored = std::fs::read(self.object_path(id))?;
        let data = match entry.base {
            Some(b) => {
                let base = self.get(b)?;
                crate::decompress_with_base(&base, &stored).map_err(codec)?
            }
            None => crate::decompress(&stored).map_err(codec)?,
        };
        if data.len() as u64 != entry.raw_len {
            return Err(bad("object length"));
        }
        Ok(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(x: &mut u64) -> u64 {
        *x ^= *x << 13;
        *x ^= *x >> 7;
        *x ^= *x << 17;
        *x
    }

    fn wordy(n: usize, seed: u64) -> Vec<u8> {
        let words: Vec<Vec<u8>> = (0..400).map(|i| format!("w{}{} ", i, "abcdefghij".repeat(i % 5)).into_bytes()).collect();
        let mut x = seed;
        let mut v = Vec::with_capacity(n + 64);
        while v.len() < n {
            let r = rnd(&mut x);
            v.extend_from_slice(&words[(r >> 8) as usize % words.len()]);
        }
        v.truncate(n);
        v
    }

    fn edited(old: &[u8], edits: usize, seed: u64) -> Vec<u8> {
        let mut x = seed;
        let mut v = old.to_vec();
        for _ in 0..edits {
            let at = (rnd(&mut x) as usize) % v.len().max(1);
            let len = (rnd(&mut x) as usize) % 300;
            let ins = wordy(len, rnd(&mut x));
            let end = (at + len / 2).min(v.len());
            v.splice(at..end, ins);
        }
        v
    }

    #[test]
    fn versions_are_found_and_come_back() {
        let dir = std::env::temp_dir().join(format!("glyd-store-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let base = wordy(6 << 20, 1);
        let mut versions = vec![base.clone()];
        for i in 1..8 {
            let prev = versions[i - 1].clone();
            versions.push(edited(&prev, 40, 10 + i as u64));
        }
        let other = wordy(2 << 20, 99);
        {
            let mut store = Store::open(&dir).unwrap();
            for (i, v) in versions.iter().enumerate() {
                let id = store.put(&format!("v{i}"), v).unwrap();
                assert_eq!(id as usize, i);
            }
            store.put("other", &other).unwrap();
            let e = store.entries();
            assert!(e[0].base.is_none() && e[8].base.is_none(), "the first version and an unrelated object stand alone");
            for i in 1..8 {
                assert!(e[i].base.is_some(), "version {i} should have a base");
                assert!(e[i].depth <= MAX_DEPTH, "depth {}", e[i].depth);
                assert!(e[i].stored_len * 10 < e[0].stored_len, "a version should cost a tenth of the first: {} vs {}", e[i].stored_len, e[0].stored_len);
            }
            assert!(e.iter().any(|x| x.depth == MAX_DEPTH) && e.iter().all(|x| x.depth <= MAX_DEPTH));
        }
        // Reopened: the table is rebuilt, every object comes back, a new version still finds its base.
        let mut store = Store::open(&dir).unwrap();
        for (i, v) in versions.iter().enumerate() {
            assert!(store.get(i as u32).unwrap() == *v, "version {i}");
        }
        assert!(store.get(8).unwrap() == other);
        let v8 = edited(&versions[7], 40, 77);
        let id = store.put("v8", &v8).unwrap();
        assert!(store.entries()[id as usize].base.is_some());
        assert!(store.get(id).unwrap() == v8);
        assert!(store.get(99).is_err());
        let (raw, stored) = store.stats();
        assert!(stored * 3 < raw / 2, "{stored} of {raw}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
