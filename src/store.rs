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
//! On disk, a directory: `objects/<id>` (the stream), `objects/<id>.fp`
//! (the fingerprints), `table` (an open-addressing hash table of the
//! fingerprints to the objects holding them, read and written in place
//! so a store's memory does not grow with its size) and `index` (one
//! line per object: id, base id or -, depth, raw length, stored length,
//! pack id and position or -, name).

use std::collections::HashMap;
use std::fs::File;
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
/// Objects under this size go into packs.
pub const SMALL: usize = 256 << 10;
/// A pack is closed when it holds about this much.
const PACK_SIZE: usize = 2 << 20;

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

/// A file mapped read-write (libc's mmap; the library carries no crate
/// for it).
struct Mapping {
    ptr: *mut u8,
    len: usize,
}

extern "C" {
    fn mmap(addr: *mut std::ffi::c_void, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut std::ffi::c_void;
    fn munmap(addr: *mut std::ffi::c_void, len: usize) -> i32;
    fn msync(addr: *mut std::ffi::c_void, len: usize, flags: i32) -> i32;
}
const PROT_READ: i32 = 1;
const PROT_WRITE: i32 = 2;
const MAP_SHARED: i32 = 1;
const MS_SYNC: i32 = 0x10;

impl Mapping {
    fn of(file: &File, len: usize) -> Result<Mapping> {
        use std::os::unix::io::AsRawFd;
        let ptr = unsafe { mmap(std::ptr::null_mut(), len, PROT_READ | PROT_WRITE, MAP_SHARED, file.as_raw_fd(), 0) };
        if ptr as isize == -1 {
            return Err(Error::new(ErrorKind::Other, "store: mmap of the table failed"));
        }
        Ok(Mapping { ptr: ptr as *mut u8, len })
    }
    fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    fn bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
    fn sync(&self) {
        unsafe {
            msync(self.ptr as *mut _, self.len, MS_SYNC);
        }
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        unsafe {
            munmap(self.ptr as *mut _, self.len);
        }
    }
}

impl Table {
    fn open(path: PathBuf) -> Result<Table> {
        let file = std::fs::OpenOptions::new().read(true).write(true).create(true).open(&path)?;
        let len = file.metadata()?.len() as usize;
        if len >= TABLE_HEADER {
            let map = Mapping::of(&file, len)?;
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
        let mut map = Mapping::of(&file, len)?;
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

pub struct Store {
    dir: PathBuf,
    entries: Vec<Entry>,
    table: Table,
    /// Small objects waiting for their pack: (id, data).
    pending: Vec<(u32, Vec<u8>)>,
    pending_bytes: usize,
    /// The last pack decoded, for reads of its objects.
    pack_cache: std::cell::RefCell<Option<(u32, Vec<Vec<u8>>)>>,
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
        let table = Table::open(dir.join("table"))?;
        let mut store = Store { dir, entries: Vec::new(), table, pending: Vec::new(), pending_bytes: 0, pack_cache: std::cell::RefCell::new(None) };
        let index = store.dir.join("index");
        if index.exists() {
            let text = std::fs::read_to_string(&index)?;
            let mut lines: Vec<Entry> = Vec::new();
            for line in text.lines() {
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
            store.entries = by_id.into_iter().enumerate().map(|(id, e)| e.unwrap_or(Entry { id: id as u32, name: "(lost: not flushed)".to_string(), base: None, depth: 0, raw_len: 0, stored_len: 0, pack: Some((u32::MAX, u32::MAX)) })).collect();
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

    /// Raw and stored bytes over the store (an open pack's objects count
    /// as raw only until `flush`).
    pub fn stats(&self) -> (u64, u64) {
        self.entries.iter().fold((0, 0), |(r, s), e| (r + e.raw_len, s + e.stored_len))
    }

    fn append_index(&self, entry: &Entry) -> Result<()> {
        let mut index = std::fs::OpenOptions::new().create(true).append(true).open(self.dir.join("index"))?;
        writeln!(index, "{}\t{}\t{}\t{}\t{}\t{}\t{}", entry.id, entry.base.map_or("-".to_string(), |b| b.to_string()), entry.depth, entry.raw_len, entry.stored_len, entry.pack.map_or("-".to_string(), |(p, i)| format!("{p}:{i}")), entry.name)
    }

    /// The stored object the data shares the most fingerprints with,
    /// and the share, when there is one worth trying.
    fn candidate(&self, prints: &[u64]) -> Result<Option<(u32, f64)>> {
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
        let top = hits.values().cloned().fold(0.0f64, f64::max);
        // Among those within 5% of the best, the most recent: a copy of
        // a copy shares its fingerprints with both.
        let Some((id, score)) = hits.iter().filter(|(_, &s)| s >= top * 0.95).max_by_key(|(&id, _)| id).map(|(&id, &s)| (id, s)) else {
            return Ok(None);
        };
        let share = score / prints.len().max(1) as f64;
        Ok(if share < MIN_SHARE { None } else { Some((id, share)) })
    }

    /// Store `data` under `name`; its id. A large object is kept as a
    /// delta against the stored object it most resembles when that
    /// pays, else alone at the max level (record mode where that pays);
    /// a small one waits in the open pack (`flush` writes it).
    pub fn put(&mut self, name: &str, data: &[u8]) -> Result<u32> {
        let id = self.entries.len() as u32;
        let name = name.replace(['\t', '\n'], " ");
        if data.len() < SMALL {
            let entry = Entry { id, name, base: None, depth: 0, raw_len: data.len() as u64, stored_len: 0, pack: Some((u32::MAX, self.pending.len() as u32)) };
            self.entries.push(entry);
            self.pending.push((id, data.to_vec()));
            self.pending_bytes += data.len();
            if self.pending_bytes >= PACK_SIZE {
                self.flush()?;
            }
            return Ok(id);
        }
        let prints = fingerprints(data);
        let mut stored = Vec::with_capacity(data.len() / 4 + 1024);
        crate::compress_records_into_max(data, &mut stored);
        let mut base = None;
        if let Some((mut bid, _)) = self.candidate(&prints)? {
            if self.entries[bid as usize].depth >= MAX_DEPTH {
                while let Some(p) = self.entries[bid as usize].base {
                    bid = p;
                }
            }
            let base_data = self.get(bid)?;
            let mut delta = Vec::with_capacity(data.len() / 16 + 1024);
            crate::compress_with_base(&base_data, data, &mut delta, false);
            if delta.len() * WORTH_DEN <= stored.len() * WORTH_NUM {
                stored = delta;
                base = Some(bid);
            }
        }
        let depth = base.map_or(0, |b| self.entries[b as usize].depth + 1);
        std::fs::write(self.object_path(id), &stored)?;
        let mut fp = Vec::with_capacity(prints.len() * 8);
        for h in &prints {
            fp.extend_from_slice(&h.to_le_bytes());
        }
        std::fs::write(self.fp_path(id), &fp)?;
        let entry = Entry { id, name, base, depth, raw_len: data.len() as u64, stored_len: stored.len() as u64, pack: None };
        self.append_index(&entry)?;
        for h in prints {
            self.table.insert(h, id)?;
        }
        self.table.sync();
        self.entries.push(entry);
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
        crate::compress_pack(&objects, &mut stored, crate::compress_into_max);
        std::fs::write(self.object_path(pack_id), &stored)?;
        let raw: u64 = self.pending.iter().map(|(_, d)| d.len() as u64).sum();
        let pack = Entry { id: pack_id, name: format!("pack of {}", self.pending.len()), base: None, depth: 0, raw_len: 0, stored_len: stored.len() as u64, pack: None };
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
        self.entries.push(pack);
        let _ = raw;
        self.pending.clear();
        self.pending_bytes = 0;
        Ok(())
    }

    /// Object `id` back: its base first when it has one; from its pack
    /// (decoded and kept for the next read) when it is small.
    pub fn get(&self, id: u32) -> Result<Vec<u8>> {
        let entry = self.entries.get(id as usize).ok_or_else(|| bad("no such object"))?;
        if let Some((pack, i)) = entry.pack {
            if pack == u32::MAX {
                // Still in the open pack.
                return self.pending.iter().find(|(m, _)| *m == id).map(|(_, d)| d.clone()).ok_or_else(|| bad("object not flushed"));
            }
            let mut cache = self.pack_cache.borrow_mut();
            if cache.as_ref().map_or(true, |(p, _)| *p != pack) {
                let stored = std::fs::read(self.object_path(pack))?;
                *cache = Some((pack, crate::decompress_pack(&stored).map_err(codec)?));
            }
            let data = cache.as_ref().unwrap().1.get(i as usize).ok_or_else(|| bad("pack member"))?.clone();
            if data.len() as u64 != entry.raw_len {
                return Err(bad("object length"));
            }
            return Ok(data);
        }
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
        let _ = std::fs::remove_dir_all(&dir);
    }
}
