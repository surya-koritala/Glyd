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
//! Small objects (under `SMALL`) have nothing to fingerprint and would
//! cost their whole size alone; they are gathered into packs (`pack`,
//! record mode where it pays) of about `PACK_SIZE`, one stored object
//! each, and read back by decoding the pack and slicing.
//!
//! On disk, a directory: `table` (an open-addressing hash table of the
//! fingerprints to the objects holding them, mapped in place so a
//! store's memory does not grow with its size), `index` (one line per
//! object: id, base id or -, depth, raw length, stored length, pack id
//! and position or -, name; a later line for an id replaces an earlier
//! one; `D` lines delete), and the objects' streams under `objects/`
//! through a `Backend`: the directory itself, or an S3 bucket
//! (`S3Backend`, over HTTPS; also any S3-compatible service through
//! `AWS_ENDPOINT_URL`). Metadata stays local, but every object's index
//! lines ride beside it as `<id>.index` in the backend (a pack's carry
//! its members'), so a lost metadata directory is rebuilt from the
//! objects (`Store::rebuild_with`): the index from the sidecars, the
//! table by reading every object back.
//!
//! This crate is the store; the codec it builds on is the `glyd` crate
//! (BSD-3-Clause OR GPL-2.0). The store is under the Business Source License 1.1.

pub mod c_api;

use glyd::mmap::Mapping;
use std::collections::HashMap;
use std::io::{Error, ErrorKind, Result, Write};
use std::sync::Mutex;
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
/// Objects under this size go into packs.
pub const SMALL: usize = 256 << 10;
/// A pack is closed when it holds about this much.
const PACK_SIZE: usize = 2 << 20;
/// Decoded objects kept in memory for the next put's base.
const CACHE_BYTES: usize = 4 << 30;

#[derive(Clone, Debug)]
pub struct Entry {
    pub id: u32,
    pub name: String,
    pub base: Option<u32>,
    pub depth: usize,
    pub raw_len: u64,
    /// Bytes on disk; 0 for an object inside a pack (the pack's entry
    /// carries them).
    pub stored_len: u64,
    /// For a small object: the pack it is in and its position there.
    pub pack: Option<(u32, u32)>,
    /// Deleted: `get` refuses it; its bytes stay on disk while a live
    /// object's chain runs through it, until `compact`.
    pub deleted: bool,
}

/// The level objects stored alone (and packs) are compressed at; deltas
/// take the max level, or the ultra level under `Ultra`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Max,
    Ultra,
    Cold,
}

/// The fingerprint table on disk: open addressing, a 12-byte slot per
/// holder (the fingerprint and the object id), the home slot from the
/// fingerprint's low bits, linear probing; a fingerprint keeps its last
/// `HOLDERS` holders, the oldest replaced. The file is mapped into
/// memory, so the store's own memory stays flat whatever it holds (12
/// bytes per 4 KB stored on disk, at most half full, so probe runs stay
/// short) and the page cache decides what is resident.
struct Table {
    path: PathBuf,
    map: Mapping,
    /// Slots (a power of two) and slots in use.
    capacity: u64,
    count: u64,
}

const SLOT: usize = 12;
const EMPTY: u32 = u32::MAX;
const TABLE_HEADER: usize = 16;

impl Table {
    fn open(path: PathBuf) -> Result<Table> {
        let file = std::fs::OpenOptions::new().read(true).write(true).create(true).open(&path)?;
        let len = file.metadata()?.len() as usize;
        if len >= TABLE_HEADER {
            let map = Mapping::read_write(&file, len)?;
            let b = map.bytes();
            let capacity = u64::from_le_bytes(b[..8].try_into().unwrap());
            let count = u64::from_le_bytes(b[8..16].try_into().unwrap());
            if !capacity.is_power_of_two() || len != TABLE_HEADER + capacity as usize * SLOT {
                return Err(bad("table header"));
            }
            return Ok(Table { path, map, capacity, count });
        }
        Self::create(path, 1 << 16)
    }

    /// An empty table of `capacity` slots at `path`.
    fn create(path: PathBuf, capacity: u64) -> Result<Table> {
        let file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&path)?;
        let len = TABLE_HEADER + capacity as usize * SLOT;
        file.set_len(len as u64)?;
        let mut map = Mapping::read_write(&file, len)?;
        let b = map.bytes_mut();
        b[..8].copy_from_slice(&capacity.to_le_bytes());
        for i in 0..capacity as usize {
            b[TABLE_HEADER + i * SLOT + 8..TABLE_HEADER + i * SLOT + 12].copy_from_slice(&EMPTY.to_le_bytes());
        }
        Ok(Table { path, map, capacity, count: 0 })
    }

    fn sync(&mut self) {
        let count = self.count;
        self.map.bytes_mut()[8..16].copy_from_slice(&count.to_le_bytes());
        self.map.sync();
    }

    #[inline]
    fn slot(&self, i: u64) -> (u64, u32) {
        let at = TABLE_HEADER + i as usize * SLOT;
        let b = &self.map.bytes()[at..at + SLOT];
        (u64::from_le_bytes(b[..8].try_into().unwrap()), u32::from_le_bytes(b[8..].try_into().unwrap()))
    }

    #[inline]
    fn set(&mut self, i: u64, hash: u64, id: u32) {
        let at = TABLE_HEADER + i as usize * SLOT;
        let b = &mut self.map.bytes_mut()[at..at + SLOT];
        b[..8].copy_from_slice(&hash.to_le_bytes());
        b[8..].copy_from_slice(&id.to_le_bytes());
    }

    /// The objects holding `hash`.
    fn lookup(&self, hash: u64, out: &mut Vec<u32>) {
        let mask = self.capacity - 1;
        let mut i = hash & mask;
        loop {
            let (h, id) = self.slot(i);
            if id == EMPTY {
                return;
            }
            if h == hash {
                out.push(id);
            }
            i = (i + 1) & mask;
        }
    }

    /// `id` as a holder of `hash`: in the first empty slot of the probe
    /// run, or in place of the fingerprint's oldest holder when it has
    /// `HOLDERS` already. The table grows when it is half full.
    fn insert(&mut self, hash: u64, id: u32) -> Result<()> {
        if self.count * 2 >= self.capacity {
            self.grow()?;
        }
        let mask = self.capacity - 1;
        let mut i = hash & mask;
        let mut holders = 0usize;
        let mut oldest: Option<(u64, u32)> = None;
        loop {
            let (h, sid) = self.slot(i);
            if sid == EMPTY {
                if holders >= HOLDERS {
                    self.set(oldest.unwrap().0, hash, id);
                } else {
                    self.count += 1;
                    self.set(i, hash, id);
                }
                return Ok(());
            }
            if h == hash {
                holders += 1;
                if oldest.map_or(true, |(_, o)| sid < o) {
                    oldest = Some((i, sid));
                }
            }
            i = (i + 1) & mask;
        }
    }

    /// Twice the slots, every holder re-inserted oldest first (so the
    /// newest survive the cap), in a fresh file swapped into place.
    fn grow(&mut self) -> Result<()> {
        let mut holders: Vec<(u64, u32)> = (0..self.capacity).map(|i| self.slot(i)).filter(|&(_, id)| id != EMPTY).collect();
        holders.sort_by_key(|&(_, id)| id);
        let tmp = self.path.with_extension("tmp");
        let mut fresh = Table::create(tmp.clone(), self.capacity * 2)?;
        for (hash, id) in holders {
            fresh.insert(hash, id)?;
        }
        fresh.sync();
        drop(fresh);
        std::fs::rename(&tmp, &self.path)?;
        let opened = Table::open(self.path.clone())?;
        // The old mapping is unmapped as it is replaced.
        *self = opened;
        Ok(())
    }
}

/// Where a store keeps its objects' bytes.
pub trait Backend: Send + Sync {
    fn read(&self, key: &str) -> Result<Vec<u8>>;
    fn write(&self, key: &str, data: &[u8]) -> Result<()>;
    fn remove(&self, key: &str) -> Result<()>;
    fn exists(&self, key: &str) -> bool;
    /// Bytes of the object at `key`, when it exists.
    fn len(&self, key: &str) -> Option<u64>;
    /// Every (key, bytes) held, in key order.
    fn list(&self) -> Result<Vec<(String, u64)>>;
}

/// Objects as files under a directory.
pub struct LocalBackend {
    dir: PathBuf,
}

impl LocalBackend {
    pub fn new(dir: impl AsRef<Path>) -> Result<LocalBackend> {
        std::fs::create_dir_all(dir.as_ref())?;
        Ok(LocalBackend { dir: dir.as_ref().to_path_buf() })
    }
}

impl Backend for LocalBackend {
    fn read(&self, key: &str) -> Result<Vec<u8>> {
        std::fs::read(self.dir.join(key))
    }
    fn write(&self, key: &str, data: &[u8]) -> Result<()> {
        std::fs::write(self.dir.join(key), data)
    }
    fn remove(&self, key: &str) -> Result<()> {
        std::fs::remove_file(self.dir.join(key))
    }
    fn exists(&self, key: &str) -> bool {
        self.dir.join(key).exists()
    }
    fn len(&self, key: &str) -> Option<u64> {
        std::fs::metadata(self.dir.join(key)).ok().map(|m| m.len())
    }
    fn list(&self) -> Result<Vec<(String, u64)>> {
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            if let (Some(name), Ok(m)) = (entry.file_name().to_str(), entry.metadata()) {
                if m.is_file() {
                    out.push((name.to_string(), m.len()));
                }
            }
        }
        out.sort();
        Ok(out)
    }
}

pub mod s3;
pub use s3::S3Backend;

pub struct Store {
    dir: PathBuf,
    objects: Box<dyn Backend>,
    entries: Vec<Entry>,
    table: Table,
    level: Level,
    /// The latest id under each name.
    names: HashMap<String, u32>,
    /// Small objects waiting for their pack: (id, data).
    pending: Vec<(u32, Vec<u8>)>,
    pending_bytes: usize,
    /// The last pack decoded, for reads of its objects.
    pack_cache: std::cell::RefCell<Option<(u32, Vec<Vec<u8>>)>>,
    /// The last large object put or rebuilt: the likeliest base of the
    /// next put, and a read of it costs nothing.
    /// Objects decoded lately, newest last, up to `CACHE_BYTES`: the
    /// next base is usually one of them, and a chain's root serves
    /// every delta on it.
    cache: std::cell::RefCell<Vec<(u32, std::sync::Arc<Vec<u8>>)>>,
}

fn bad(msg: &str) -> Error {
    Error::new(ErrorKind::InvalidData, format!("store: {msg}"))
}

fn codec(e: glyd::error::CodecError) -> Error {
    Error::new(ErrorKind::InvalidData, e)
}

/// The fingerprints of an object: its sparse anchors whose hash has two
/// more zero bits.
fn fingerprints(data: &[u8]) -> Vec<u64> {
    // Every position is tested on its own, so the scan splits across
    // the cores: each chunk reads 64 bytes past its end for the hashes
    // at its last positions and keeps only the positions it owns.
    const CHUNK: usize = 64 << 20;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let chunks: Vec<(usize, usize)> = (0..data.len().max(1)).step_by(CHUNK).map(|a| (a, (a + CHUNK).min(data.len()))).collect();
    let slots: Vec<Mutex<Vec<u64>>> = chunks.iter().map(|_| Mutex::new(Vec::new())).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads.min(chunks.len()) {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= chunks.len() {
                    break;
                }
                let (a, b) = chunks[i];
                let mut anchors = Vec::new();
                glyd::ldm::sparse_anchors(&data[a..(b + 64).min(data.len())], 0, &mut anchors);
                let own = (b - a) as u64;
                *slots[i].lock().unwrap() = anchors.into_iter().filter(|&(h, pos)| pos < own && h >> 62 == 0).map(|(h, _)| h).collect();
            });
        }
    });
    slots.into_iter().flat_map(|m| m.into_inner().unwrap()).collect()
}

impl Store {
    /// Open the store at `dir`, creating it when it does not exist; the
    /// objects live under `dir/objects`.
    pub fn open(dir: impl AsRef<Path>) -> Result<Store> {
        let objects = LocalBackend::new(dir.as_ref().join("objects"))?;
        Self::open_with(dir, Box::new(objects))
    }

    /// Open the store whose metadata is at `dir` and whose objects are
    /// in `objects` (a directory, an S3 bucket, ...).
    pub fn open_with(dir: impl AsRef<Path>, objects: Box<dyn Backend>) -> Result<Store> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        let table = Table::open(dir.join("table"))?;
        let mut store = Store { dir, objects, entries: Vec::new(), table, level: Level::Max, names: HashMap::new(), pending: Vec::new(), pending_bytes: 0, pack_cache: std::cell::RefCell::new(None), cache: std::cell::RefCell::new(Vec::new()) };
        let index = store.dir.join("index");
        if index.exists() {
            let text = std::fs::read_to_string(&index)?;
            let mut lines: Vec<Entry> = Vec::new();
            let mut deleted: Vec<u32> = Vec::new();
            for line in text.lines() {
                if let Some(id) = line.strip_prefix("D\t") {
                    deleted.push(id.parse().map_err(|_| bad("index deletion"))?);
                    continue;
                }
                let f: Vec<&str> = line.splitn(7, '\t').collect();
                if f.len() != 7 {
                    return Err(bad("index line"));
                }
                let parse = |s: &str| s.parse::<u64>().map_err(|_| bad("index number"));
                let pack = if f[5] == "-" {
                    None
                } else {
                    let (p, i) = f[5].split_once(':').ok_or_else(|| bad("index pack"))?;
                    Some((parse(p)? as u32, parse(i)? as u32))
                };
                let entry = Entry {
                    id: parse(f[0])? as u32,
                    base: if f[1] == "-" { None } else { Some(parse(f[1])? as u32) },
                    depth: parse(f[2])? as usize,
                    raw_len: parse(f[3])?,
                    stored_len: parse(f[4])?,
                    pack,
                    name: f[6].to_string(),
                    deleted: false,
                };
                lines.push(entry);
            }
            // Members of a pack are written at the flush, after objects
            // put in between: entries go by id, and an id never written
            // (a small object whose store died before its flush) is lost.
            lines.sort_by_key(|e| e.id);
            let last = lines.last().map_or(0, |e| e.id as usize + 1);
            let mut by_id: Vec<Option<Entry>> = vec![None; last];
            for e in lines {
                let id = e.id as usize;
                by_id[id] = Some(e);
            }
            store.entries = by_id.into_iter().enumerate().map(|(id, e)| e.unwrap_or(Entry { id: id as u32, name: "(lost: not flushed)".to_string(), base: None, depth: 0, raw_len: 0, stored_len: 0, pack: Some((u32::MAX, u32::MAX)), deleted: true })).collect();
            for id in deleted {
                if let Some(e) = store.entries.get_mut(id as usize) {
                    e.deleted = true;
                }
            }
            for e in &store.entries {
                if !e.deleted && !e.name.starts_with("pack of ") {
                    store.names.insert(e.name.clone(), e.id);
                }
            }
        }
        Ok(store)
    }

    /// The store whose metadata directory is gone: `dir` is made anew
    /// from the objects in `objects`, the index from their sidecars and
    /// the table by reading every object back. Objects that fail to
    /// read are listed, not fatal.
    pub fn rebuild_with(dir: impl AsRef<Path>, objects: Box<dyn Backend>) -> Result<(Store, Vec<(u32, String)>)> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        for f in ["index", "table"] {
            let _ = std::fs::remove_file(dir.join(f));
        }
        let keys: Vec<String> = objects.list()?.into_iter().map(|(k, _)| k).filter(|k| k.ends_with(".index")).collect();
        let sidecars: Mutex<Vec<(u32, String)>> = Mutex::new(Vec::with_capacity(keys.len()));
        let next = std::sync::atomic::AtomicUsize::new(0);
        let failed: Mutex<Option<Error>> = Mutex::new(None);
        std::thread::scope(|scope| {
            for _ in 0..16.min(keys.len().max(1)) {
                scope.spawn(|| loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= keys.len() {
                        break;
                    }
                    let id: u32 = match keys[i].trim_end_matches(".index").parse() {
                        Ok(id) => id,
                        Err(_) => continue,
                    };
                    match objects.read(&keys[i]).map(|b| String::from_utf8_lossy(&b).to_string()) {
                        Ok(text) => sidecars.lock().unwrap().push((id, text)),
                        Err(e) => *failed.lock().unwrap() = Some(e),
                    }
                });
            }
        });
        if let Some(e) = failed.into_inner().unwrap() {
            return Err(e);
        }
        let mut sidecars = sidecars.into_inner().unwrap();
        sidecars.sort();
        let text: String = sidecars.iter().map(|(_, t)| t.as_str()).collect();
        std::fs::write(dir.join("index"), text)?;
        let mut store = Self::open_with(&dir, objects)?;
        let mut failures = Vec::new();
        for id in 0..store.entries.len() as u32 {
            let e = &store.entries[id as usize];
            if e.deleted || e.pack.is_some() || e.name.starts_with("pack of ") {
                continue;
            }
            match store.get(id) {
                Ok(data) => {
                    for h in fingerprints(&data) {
                        store.table.insert(h, id)?;
                    }
                }
                Err(err) => failures.push((id, err.to_string())),
            }
        }
        store.table.sync();
        Ok((store, failures))
    }

    /// `rebuild_with` for a store whose objects are under `dir/objects`.
    pub fn rebuild(dir: impl AsRef<Path>) -> Result<(Store, Vec<(u32, String)>)> {
        let objects = LocalBackend::new(dir.as_ref().join("objects"))?;
        Self::rebuild_with(dir, Box::new(objects))
    }

    fn key(id: u32) -> String {
        id.to_string()
    }

    fn line(entry: &Entry) -> String {
        format!("{}\t{}\t{}\t{}\t{}\t{}\t{}\n", entry.id, entry.base.map_or("-".to_string(), |b| b.to_string()), entry.depth, entry.raw_len, entry.stored_len, entry.pack.map_or("-".to_string(), |(p, i)| format!("{p}:{i}")), entry.name)
    }

    /// The index lines of object `id`, kept beside it in the backend.
    fn write_sidecar(&self, id: u32, lines: &str) -> Result<()> {
        self.objects.write(&format!("{id}.index"), lines.as_bytes())
    }

    /// The objects, in id order.
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// Raw bytes of the live objects and bytes on disk (an open pack's
    /// objects count as raw only until `flush`; a deleted object's file
    /// counts until `compact`).
    pub fn stats(&self) -> (u64, u64) {
        let raw = self.entries.iter().filter(|e| !e.deleted).map(|e| e.raw_len).sum();
        let stored = self.entries.iter().filter(|e| !e.deleted || e.stored_len > 0 && self.objects.exists(&Self::key(e.id))).map(|e| e.stored_len).sum();
        (raw, stored)
    }

    /// The level objects stored alone take from now on.
    pub fn set_level(&mut self, level: Level) {
        self.level = level;
    }

    /// The latest live object put under `name`.
    pub fn id_of(&self, name: &str) -> Option<u32> {
        self.names.get(name).copied()
    }

    /// Mark `id` deleted: `get` refuses it from now on; its bytes leave
    /// the disk at `compact`, once no live object's chain needs them.
    pub fn delete(&mut self, id: u32) -> Result<()> {
        let entry = self.entries.get_mut(id as usize).ok_or_else(|| bad("no such object"))?;
        if entry.deleted {
            return Ok(());
        }
        entry.deleted = true;
        let name = entry.name.clone();
        if self.names.get(&name) == Some(&id) {
            self.names.remove(&name);
        }
        let mut index = std::fs::OpenOptions::new().create(true).append(true).open(self.dir.join("index"))?;
        writeln!(index, "D\t{id}")?;
        // A pack member's line is in its pack's sidecar; its own carries
        // the deletion alone.
        let entry = &self.entries[id as usize];
        let lines = if entry.pack.is_some() { format!("D\t{id}\n") } else { format!("{}D\t{id}\n", Self::line(entry)) };
        self.write_sidecar(id, &lines)
    }

    /// Remove from disk what no live object needs: deleted objects no
    /// live chain runs through, packs whose members are all deleted.
    /// Returns the bytes freed.
    pub fn compact(&mut self) -> Result<u64> {
        let n = self.entries.len();
        let mut needed = vec![false; n];
        for e in &self.entries {
            if e.deleted || e.pack.map_or(false, |(p, _)| p == u32::MAX) {
                continue;
            }
            let mut at = Some(e.id);
            while let Some(i) = at {
                needed[i as usize] = true;
                at = self.entries[i as usize].base;
            }
            if let Some((p, _)) = e.pack {
                needed[p as usize] = true;
            }
        }
        let mut freed = 0u64;
        for i in 0..n {
            if needed[i] || self.entries[i].pack.is_some() {
                continue;
            }
            let key = Self::key(i as u32);
            if let Some(n) = self.objects.len(&key) {
                freed += n;
                self.objects.remove(&key)?;
                self.entries[i].stored_len = 0;
            }
        }
        self.cache.borrow_mut().retain(|(id, _)| needed[*id as usize]);
        Ok(freed)
    }

    /// Every live object read back and checked (lengths and the
    /// streams' own checksums): the count that passed and the failures.
    pub fn verify(&self) -> (usize, Vec<(u32, String)>) {
        let mut ok = 0usize;
        let mut failed = Vec::new();
        for e in &self.entries {
            if e.deleted || e.name.starts_with("pack of ") {
                continue;
            }
            match self.get(e.id) {
                Ok(_) => ok += 1,
                Err(err) => failed.push((e.id, err.to_string())),
            }
        }
        (ok, failed)
    }

    fn append_index(&self, entry: &Entry) -> Result<()> {
        let mut index = std::fs::OpenOptions::new().create(true).append(true).open(self.dir.join("index"))?;
        index.write_all(Self::line(entry).as_bytes())
    }

    /// The stored objects the data shares the most fingerprints with,
    /// best first with their shares: the best, and a second when it
    /// scores at least half as much (`put` tries both on a sample).
    fn candidates(&self, prints: &[u64]) -> Result<Vec<(u32, f64)>> {
        let mut hits: HashMap<u32, f64> = HashMap::new();
        let mut holders = Vec::with_capacity(HOLDERS);
        for &h in prints {
            holders.clear();
            self.table.lookup(h, &mut holders);
            let n = holders.len() as f64;
            for &id in &holders {
                *hits.entry(id).or_insert(0.0) += 1.0 / n;
            }
        }
        hits.retain(|&id, _| self.entries.get(id as usize).map_or(false, |e| !e.deleted));
        let top = hits.values().cloned().fold(0.0f64, f64::max);
        // Among those within 5% of the best, the most recent: a copy of
        // a copy shares its fingerprints with both.
        let Some((first, score)) = hits.iter().filter(|(_, &s)| s >= top * 0.95).max_by_key(|(&id, _)| id).map(|(&id, &s)| (id, s)) else {
            return Ok(Vec::new());
        };
        let n = prints.len().max(1) as f64;
        if score / n < MIN_SHARE {
            return Ok(Vec::new());
        }
        let mut out = vec![(first, score / n)];
        let mut rest: Vec<(u32, f64)> = hits.iter().filter(|(&id, &s)| id != first && s * 2.0 >= score).map(|(&id, &s)| (id, s / n)).collect();
        rest.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(b.0.cmp(&a.0)));
        out.extend(rest.into_iter().take(1));
        Ok(out)
    }

    /// The chain root of `id` when `id` sits at the depth cap, else `id`.
    fn within_cap(&self, mut id: u32) -> u32 {
        if self.entries[id as usize].depth >= MAX_DEPTH {
            while let Some(p) = self.entries[id as usize].base {
                id = p;
            }
        }
        id
    }

    /// Store `data` under `name`; its id. A large object is kept as a
    /// delta against the stored object it most resembles when that
    /// pays, else alone at the max level (record mode where that pays);
    /// a small one waits in the open pack (`flush` writes it).
    pub fn put(&mut self, name: &str, data: &[u8]) -> Result<u32> {
        let id = self.entries.len() as u32;
        let name = name.replace(['\t', '\n'], " ");
        if data.len() < SMALL {
            self.names.insert(name.clone(), id);
            let entry = Entry { id, name, base: None, depth: 0, raw_len: data.len() as u64, stored_len: 0, pack: Some((u32::MAX, self.pending.len() as u32)), deleted: false };
            self.entries.push(entry);
            self.pending.push((id, data.to_vec()));
            self.pending_bytes += data.len();
            if self.pending_bytes >= PACK_SIZE {
                self.flush()?;
            }
            return Ok(id);
        }
        // GLYD_STORE_TIMING=1 prints where a put's time goes.
        let timing = std::env::var_os("GLYD_STORE_TIMING").is_some();
        let mut laps: Vec<(&str, f64)> = Vec::new();
        let mut clock = std::time::Instant::now();
        let mut lap = |what: &'static str, laps: &mut Vec<(&str, f64)>| {
            if timing {
                laps.push((what, clock.elapsed().as_secs_f64()));
                clock = std::time::Instant::now();
            }
        };
        let prints = fingerprints(data);
        lap("fingerprints", &mut laps);
        let alone = |data: &[u8]| {
            let mut out = Vec::with_capacity(data.len() / 4 + 1024);
            match self.level {
                Level::Max => glyd::compress_records_into_max(data, &mut out),
                Level::Ultra => glyd::compress_records_into_ultra(data, &mut out),
                Level::Cold => glyd::compress_records_into_cold(data, &mut out),
            }
            out
        };
        let mut base = None;
        let mut stored = Vec::new();
        let scored = self.candidates(&prints)?;
        if timing {
            eprintln!("  candidates {id}: {}", scored.iter().map(|(c, s)| format!("{c} ({:.3})", s)).collect::<Vec<_>>().join(", "));
        }
        let mut candidates: Vec<u32> = scored.into_iter().map(|(id, _)| self.within_cap(id)).collect();
        candidates.dedup();
        lap("candidates", &mut laps);
        if !candidates.is_empty() {
            // Two candidates: the one whose delta of the first 32 MB is
            // smaller wins the whole object.
            let mut bid = candidates[0];
            let mut base_data = self.fetch(bid)?;
            lap("base fetched", &mut laps);
            if candidates.len() > 1 {
                let sample = &data[..data.len().min(32 << 20)];
                let other_data = self.fetch(candidates[1])?;
                let (mut a, mut b) = (Vec::new(), Vec::new());
                glyd::compress_with_base(&base_data, sample, &mut a, false);
                glyd::compress_with_base(&other_data, sample, &mut b, false);
                if b.len() < a.len() {
                    bid = candidates[1];
                    base_data = other_data;
                }
            }
            lap("second candidate", &mut laps);
            // Whether the delta pays is judged against the object alone.
            // Shared fingerprints do not settle it (hours of events share
            // half their fingerprints and gain nothing), so the first
            // 32 MB decide first: a delta of the sample against the
            // sample alone, and only a delta that pays there is taken
            // in full. Then the whole is estimated from the first 64 MB
            // when that settles it either way (a version's delta is a
            // few percent; an unrelated object's, about the whole), and
            // compressed in full otherwise.
            let sample = data.len().min(32 << 20);
            let mut trial = Vec::new();
            glyd::compress_with_base(&base_data, &data[..sample], &mut trial, false);
            let trial_alone = alone(&data[..sample]).len();
            lap("sample", &mut laps);
            // A version's sample delta is a few percent of the sample
            // alone; half is the bar, so that hours of events sharing
            // half their fingerprints are not tried in full.
            if trial.len() * 2 <= trial_alone {
                let mut delta = Vec::with_capacity(data.len() / 16 + 1024);
                glyd::compress_with_base(&base_data, data, &mut delta, self.level == Level::Ultra);
                lap("delta", &mut laps);
                let estimate = trial_alone as u64 * data.len() as u64 / sample.max(1) as u64;
                let d = delta.len() as u64;
                let alone_len = if d * 2 <= estimate || d * 5 >= estimate * 6 { estimate } else {
                    stored = alone(data);
                    lap("alone in full", &mut laps);
                    stored.len() as u64
                };
                if d * WORTH_DEN as u64 <= alone_len * WORTH_NUM as u64 {
                    stored = delta;
                    base = Some(bid);
                }
            }
        }
        if base.is_none() && stored.is_empty() {
            stored = alone(data);
            lap("alone", &mut laps);
        }
        let depth = base.map_or(0, |b| self.entries[b as usize].depth + 1);
        self.objects.write(&Self::key(id), &stored)?;
        lap("write", &mut laps);
        self.names.insert(name.clone(), id);
        let entry = Entry { id, name, base, depth, raw_len: data.len() as u64, stored_len: stored.len() as u64, pack: None, deleted: false };
        self.append_index(&entry)?;
        self.write_sidecar(id, &Self::line(&entry))?;
        for h in prints {
            self.table.insert(h, id)?;
        }
        lap("table insert", &mut laps);
        self.table.sync();
        lap("table sync", &mut laps);
        self.entries.push(entry);
        self.remember(id, std::sync::Arc::new(data.to_vec()));
        lap("cache", &mut laps);
        if timing {
            let total: f64 = laps.iter().map(|(_, t)| t).sum();
            eprintln!("  timing {id}: {}  total {total:.2} s ({:.0} MB/s)", laps.iter().map(|(w, t)| format!("{w} {t:.2}")).collect::<Vec<_>>().join(", "), data.len() as f64 / total / 1e6);
        }
        Ok(id)
    }

    /// Write the open pack: the small objects put since the last flush,
    /// as one stored object in record mode where that pays.
    pub fn flush(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let pack_id = self.entries.len() as u32;
        let objects: Vec<&[u8]> = self.pending.iter().map(|(_, d)| d.as_slice()).collect();
        let mut stored = Vec::new();
        let level: fn(&[u8], &mut Vec<u8>) = match self.level {
            Level::Max => glyd::compress_into_max,
            Level::Ultra => glyd::compress_into_ultra,
            Level::Cold => glyd::compress_into_cold,
        };
        glyd::compress_pack(&objects, &mut stored, level);
        self.objects.write(&Self::key(pack_id), &stored)?;
        let pack = Entry { id: pack_id, name: format!("pack of {}", self.pending.len()), base: None, depth: 0, raw_len: 0, stored_len: stored.len() as u64, pack: None, deleted: false };
        // The members' entries, now that their pack has an id.
        for (i, (member, _)) in self.pending.iter().enumerate() {
            let e = &mut self.entries[*member as usize];
            e.pack = Some((pack_id, i as u32));
        }
        let members: Vec<Entry> = self.pending.iter().map(|(m, _)| self.entries[*m as usize].clone()).collect();
        for e in &members {
            self.append_index(e)?;
        }
        self.append_index(&pack)?;
        self.write_sidecar(pack_id, &members.iter().chain(std::iter::once(&pack)).map(Self::line).collect::<String>())?;
        self.entries.push(pack);
        self.pending.clear();
        self.pending_bytes = 0;
        Ok(())
    }

    /// Object `id` back: its base first when it has one; from its pack
    /// (decoded and kept for the next read) when it is small.
    pub fn get(&self, id: u32) -> Result<Vec<u8>> {
        let entry = self.entries.get(id as usize).ok_or_else(|| bad("no such object"))?;
        if entry.deleted {
            return Err(bad("object deleted"));
        }
        if let Some(data) = self.cached(id) {
            return Ok((*data).clone());
        }
        if let Some((pack, i)) = entry.pack {
            if pack == u32::MAX {
                // Still in the open pack.
                return self.pending.iter().find(|(m, _)| *m == id).map(|(_, d)| d.clone()).ok_or_else(|| bad("object not flushed"));
            }
            let mut cache = self.pack_cache.borrow_mut();
            if cache.as_ref().map_or(true, |(p, _)| *p != pack) {
                let stored = self.objects.read(&Self::key(pack))?;
                *cache = Some((pack, glyd::decompress_pack(&stored).map_err(codec)?));
            }
            let data = cache.as_ref().unwrap().1.get(i as usize).ok_or_else(|| bad("pack member"))?.clone();
            if data.len() as u64 != entry.raw_len {
                return Err(bad("object length"));
            }
            return Ok(data);
        }
        let data = self.fetch(id)?;
        if data.len() as u64 != entry.raw_len {
            return Err(bad("object length"));
        }
        Ok((*data).clone())
    }

    fn cached(&self, id: u32) -> Option<std::sync::Arc<Vec<u8>>> {
        self.cache.borrow().iter().find(|(i, _)| *i == id).map(|(_, d)| d.clone())
    }

    /// `data` into the cache, the oldest out until it fits.
    fn remember(&self, id: u32, data: std::sync::Arc<Vec<u8>>) {
        if data.len() > CACHE_BYTES {
            return;
        }
        let mut cache = self.cache.borrow_mut();
        cache.retain(|(i, _)| *i != id);
        let mut held: usize = cache.iter().map(|(_, d)| d.len()).sum();
        while held + data.len() > CACHE_BYTES && !cache.is_empty() {
            held -= cache.remove(0).1.len();
        }
        cache.push((id, data));
    }

    /// A large object's bytes, decoded through its chain and remembered
    /// (deleted or not: a chain stays on disk until `compact`).
    fn fetch(&self, id: u32) -> Result<std::sync::Arc<Vec<u8>>> {
        if let Some(data) = self.cached(id) {
            return Ok(data);
        }
        let data = std::sync::Arc::new(self.base_data(id)?);
        self.remember(id, data.clone());
        Ok(data)
    }

    /// A base's bytes: deleted or not, its chain is still on disk until
    /// `compact` (which only removes what no live object needs).
    fn base_data(&self, id: u32) -> Result<Vec<u8>> {
        let entry = self.entries.get(id as usize).ok_or_else(|| bad("no such object"))?;
        if let Some(data) = self.cached(id) {
            return Ok((*data).clone());
        }
        let stored = self.objects.read(&Self::key(id))?;
        let data = match entry.base {
            Some(b) => {
                let base = self.fetch(b)?;
                glyd::decompress_with_base(&base, &stored).map_err(codec)?
            }
            None => glyd::decompress(&stored).map_err(codec)?,
        };
        if data.len() as u64 != entry.raw_len {
            return Err(bad("object length"));
        }
        Ok(data)
    }

    /// Store `id` again alone (depth 0), so a read of it is one decode:
    /// for an object read often that sits deep in a chain. Its
    /// dependants keep working (their chains still run through it).
    pub fn rebase(&mut self, id: u32) -> Result<()> {
        let entry = self.entries.get(id as usize).ok_or_else(|| bad("no such object"))?.clone();
        if entry.deleted || entry.pack.is_some() || entry.base.is_none() {
            return Ok(());
        }
        let data = self.get(id)?;
        let mut stored = Vec::with_capacity(data.len() / 4 + 1024);
        match self.level {
            Level::Max => glyd::compress_records_into_max(&data, &mut stored),
            Level::Ultra => glyd::compress_records_into_ultra(&data, &mut stored),
            Level::Cold => glyd::compress_records_into_cold(&data, &mut stored),
        }
        self.objects.write(&Self::key(id), &stored)?;
        let e = &mut self.entries[id as usize];
        e.base = None;
        e.depth = 0;
        e.stored_len = stored.len() as u64;
        let e = e.clone();
        self.append_index(&e)?;
        self.write_sidecar(id, &Self::line(&e))
    }
}

impl Drop for Store {
    fn drop(&mut self) {
        let _ = self.flush();
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
        // Small objects: packed, read back before and after the flush,
        // and after a reopen; a large object put in between keeps ids.
        let mut smalls: Vec<Vec<u8>> = Vec::new();
        let mut x = 5u64;
        for i in 0..300 {
            let mut o = Vec::new();
            for _ in 0..12 {
                let r = rnd(&mut x);
                o.extend_from_slice(format!("{{\"ts\": {}, \"host\": \"h{}\", \"n\": {}}}\n", 1_700_000_000 + i * 7, r % 20, r % 1000).as_bytes());
            }
            smalls.push(o);
        }
        let first_small = store.entries().len() as u32;
        for (i, o) in smalls.iter().enumerate() {
            store.put(&format!("s{i}"), o).unwrap();
            if i == 100 {
                store.put("big-in-between", &wordy(1 << 20, 123)).unwrap();
            }
        }
        assert!(store.get(first_small + 5).unwrap() == smalls[5], "from the open pack");
        store.flush().unwrap();
        let total_small: u64 = smalls.iter().map(|o| o.len() as u64).sum();
        let packs: u64 = store.entries().iter().filter(|e| e.name.starts_with("pack of")).map(|e| e.stored_len).sum();
        assert!(packs * 6 < total_small, "packed events should shrink sixfold: {packs} of {total_small}");
        drop(store);
        let store = Store::open(&dir).unwrap();
        for (i, o) in smalls.iter().enumerate() {
            let id = first_small + i as u32 + if i > 100 { 1 } else { 0 };
            assert!(store.get(id).unwrap() == *o, "small {i}");
        }
        // Names, deletion, compaction, verification.
        let mut store = Store::open(&dir).unwrap();
        assert_eq!(store.id_of("v3"), Some(3));
        assert_eq!(store.id_of("nope"), None);
        let (ok, failed) = store.verify();
        assert!(failed.is_empty() && ok > 300, "{ok} ok, {failed:?}");
        let (raw_before, disk_before) = store.stats();
        // v2 is v3's base: deleting it frees nothing until v3 goes too.
        store.delete(2).unwrap();
        assert!(store.get(2).is_err());
        assert!(store.get(3).unwrap() == versions[3], "v3 still rebuilds through its deleted base");
        assert_eq!(store.compact().unwrap(), 0);
        let chain: Vec<u32> = (0..versions.len() as u32).filter(|&i| { let mut at = Some(i); while let Some(j) = at { if j == 2 { return true; } at = store.entries()[j as usize].base; } false }).collect();
        for &i in &chain {
            store.delete(i).unwrap();
        }
        let freed = store.compact().unwrap();
        assert!(freed > 0, "the chain through v2 should leave the disk");
        assert!(!store.objects.exists("2"));
        let (raw_after, disk_after) = store.stats();
        assert!(raw_after < raw_before && disk_after < disk_before);
        assert_eq!(store.id_of("v2"), None);
        // A small object's deletion: the pack stays while a sibling lives.
        store.delete(first_small).unwrap();
        assert!(store.get(first_small).is_err());
        assert!(store.get(first_small + 1).unwrap() == smalls[1]);
        assert_eq!(store.compact().unwrap(), 0);
        let (ok, failed) = store.verify();
        assert!(failed.is_empty() && ok > 300, "{ok} ok, {failed:?}");
        // A new version after the deletions still finds a live base.
        let v9 = edited(&versions[7], 30, 88);
        let id = store.put("v9", &v9).unwrap();
        assert!(store.get(id).unwrap() == v9);
        drop(store);
        let store = Store::open(&dir).unwrap();
        assert!(store.get(2).is_err() && store.get(first_small).is_err());
        assert!(store.get(first_small + 1).unwrap() == smalls[1]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rebuild_from_objects() {
        let dir = std::env::temp_dir().join(format!("glyd-store-rebuild-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let a = wordy(3 << 20, 1);
        let b = edited(&a, 200, 2);
        let c = edited(&b, 200, 3);
        let smalls: Vec<Vec<u8>> = (0..5).map(|i| wordy(50_000, 10 + i)).collect();
        let (before, originals) = {
            let mut store = Store::open(&dir).unwrap();
            let ia = store.put("a", &a).unwrap();
            let mut ids = vec![ia];
            for (i, s) in smalls.iter().enumerate() {
                ids.push(store.put(&format!("small {i}"), s).unwrap());
            }
            let ib = store.put("b", &b).unwrap();
            let ic = store.put("c", &c).unwrap();
            ids.extend([ib, ic]);
            store.flush().unwrap();
            store.delete(ids[2]).unwrap();
            store.rebase(ic).unwrap();
            let mut originals: Vec<(u32, Vec<u8>)> = vec![(ia, a.clone()), (ib, b.clone()), (ic, c.clone())];
            for (i, s) in smalls.iter().enumerate() {
                originals.push((ids[1 + i], s.clone()));
            }
            (store.entries().to_vec(), originals)
        };
        std::fs::remove_file(dir.join("index")).unwrap();
        std::fs::remove_file(dir.join("table")).unwrap();
        let (mut store, failures) = Store::rebuild(&dir).unwrap();
        assert!(failures.is_empty(), "{failures:?}");
        let after = store.entries().to_vec();
        assert_eq!(after.len(), before.len());
        for (x, y) in before.iter().zip(&after) {
            assert_eq!((x.id, x.base, x.depth, x.raw_len, x.stored_len, x.pack, &x.name, x.deleted), (y.id, y.base, y.depth, y.raw_len, y.stored_len, y.pack, &y.name, y.deleted));
        }
        for (id, data) in &originals {
            if after[*id as usize].deleted {
                assert!(store.get(*id).is_err());
            } else {
                assert_eq!(&store.get(*id).unwrap(), data, "object {id}");
            }
        }
        // The rebuilt table finds bases again.
        let d = edited(&c, 100, 4);
        let id = store.put("d", &d).unwrap();
        assert!(store.entries()[id as usize].base.is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
