//! Prepared dictionaries for small objects: a window of content the
//! first block may match into, plus the entropy tables of a typical
//! object, so a block that reuses them writes none of its own (a fresh
//! set is ~300 bytes, most of a small object's output). The decoder
//! side keeps the tables built, so decoding an object does no table work.
//!
//! `Dict::train` picks the content from samples by cover (zstd's idea):
//! the 64-byte segments whose 8-byte strings recur most across samples,
//! each taken segment covering its strings so the next pick adds new
//! ones; the most valuable last (nearest offsets). Then it parses the
//! samples against that content for the tables.
use crate::v7_decode::{DecTables, Slot};
use crate::v7_encode::{DictTables, Sequence, Tables};
use crate::v7_format::*;
use crate::{huff8, tans};

const MAGIC: &[u8; 8] = b"GLYDDICT";
const VERSION: u16 = 1;

/// Zero bytes kept after the content: the decoder's copies from a
/// dictionary run in 32-byte steps past a match's end.
pub const CONTENT_PAD: usize = 64;

/// The trainer's segment length (see `select_content`): 1 KB against
/// zstd's optimized 50-2000; measured 256/512/1024/2048 on JSON events,
/// access logs and a source tree, 1024 best or within noise.
const SEGMENT: usize = 1024;

pub struct Dict {
    /// The content, then `CONTENT_PAD` zeros.
    content: Vec<u8>,
    id: u32,
    tables: Tables,
    dec: DecTables<'static>,
    /// The max-level finder's tables seeded on the content, read by
    /// every object's parse (`content` never moves: `Dict` owns it and
    /// this points into it).
    finder: DictTables,
}

// `finder.content` points into `content`, which is never reallocated.
unsafe impl Send for Dict {}
unsafe impl Sync for Dict {}

impl Dict {
    /// A dictionary from `content` alone: the window, and tables from a
    /// max-level parse of `samples` against it (`content` itself when
    /// there are none).
    pub fn from_content(content: &[u8], samples: &[&[u8]]) -> Dict {
        let owned;
        let samples: &[&[u8]] = if samples.is_empty() {
            owned = [content];
            &owned
        } else {
            samples
        };
        let tables = tables_for(content, samples);
        Self::assemble(content.to_vec(), tables)
    }

    /// Train on `samples`: up to `content_bytes` of the segments that
    /// recur across them, then the tables.
    pub fn train(samples: &[&[u8]], content_bytes: usize) -> Dict {
        let content = select_content(samples, content_bytes);
        let tables = tables_for(&content, samples);
        Self::assemble(content, tables)
    }

    /// The serialized form: magic, version, the content, the tables.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.content().len() + 512);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(self.content().len() as u32).to_le_bytes());
        out.extend_from_slice(self.content());
        crate::huffman::pack_lengths_v8(self.tables.lit_lengths.as_ref().unwrap(), &mut out);
        for t in [&self.tables.ll, &self.tables.ml, &self.tables.off] {
            crate::v7_encode::write_tans_table(t.as_ref().unwrap(), &mut out);
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Dict> {
        if bytes.len() < 14 || &bytes[..8] != MAGIC || u16::from_le_bytes([bytes[8], bytes[9]]) != VERSION {
            return None;
        }
        let n = u32::from_le_bytes([bytes[10], bytes[11], bytes[12], bytes[13]]) as usize;
        let content = bytes.get(14..14 + n)?.to_vec();
        let mut pos = 14 + n;
        let (lengths, used) = crate::huffman::unpack_lengths_v8(bytes.get(pos..)?)?;
        pos += used;
        let mut tables = Tables { lit_lengths: Some(lengths), lit_codes: None, ll: None, ml: None, off: None, seq_enc: None };
        for (slot, n_symbols) in [(&mut tables.ll, LL_SYMBOLS), (&mut tables.ml, ML_SYMBOLS), (&mut tables.off, OFF_SYMBOLS)] {
            let (counts, used) = crate::v7_decode::read_tans_table(bytes.get(pos..)?, n_symbols)?;
            pos += used;
            *slot = Some(counts.into());
        }
        if pos != bytes.len() {
            return None;
        }
        let dict = Self::assemble(content, tables);
        Some(dict)
    }

    fn assemble(mut content: Vec<u8>, mut tables: Tables) -> Dict {
        content.extend_from_slice(&[0u8; CONTENT_PAD]);
        // Built once here; every object's `Tables` clone shares them.
        tables.seq_enc();
        tables.lit_codes = Some(std::sync::Arc::new(huff8::Codes::build(tables.lit_lengths.as_ref().unwrap())));
        fn own<T>(t: Option<T>) -> Slot<'static, T> {
            t.map_or(Slot::None, Slot::Own)
        }
        let dec = DecTables {
            lit: own(huff8::Table::build(tables.lit_lengths.as_ref().unwrap())),
            ll: own(tans::DecodeTable::build(tables.ll.as_ref().unwrap())),
            ml: own(tans::DecodeTable::build(tables.ml.as_ref().unwrap())),
            off: own(tans::DecodeTable::build(tables.off.as_ref().unwrap())),
        };
        debug_assert!(!dec.lit.is_none() && !dec.ll.is_none() && !dec.ml.is_none() && !dec.off.is_none());
        let mut finder_tables = crate::v7_encode::DfastTables::new();
        let content_len = content.len() - CONTENT_PAD;
        finder_tables.seed(&content, content_len);
        let finder = DictTables { tables: finder_tables, content: content.as_ptr(), len: content_len };
        let mut d = Dict { content, id: 0, tables, dec, finder };
        // The id covers the tables too: a block coded against them must
        // not be decoded with another set. 0 means "no dictionary".
        let id = crate::compute_checksum(&d.to_bytes());
        d.id = if id == 0 { 1 } else { id };
        d
    }

    /// The content (the window an object's first block sees).
    pub fn content(&self) -> &[u8] {
        &self.content[..self.content.len() - CONTENT_PAD]
    }

    /// The content with its padding: what the decoder reads from.
    pub(crate) fn content_padded(&self) -> &[u8] {
        &self.content
    }

    pub fn id(&self) -> u32 {
        self.id
    }

    pub(crate) fn tables(&self) -> Tables {
        self.tables.clone()
    }

    /// The decoder's tables, borrowing this dictionary's.
    pub(crate) fn dec_tables(&self) -> DecTables<'_> {
        DecTables::borrowing(&self.dec)
    }

    pub(crate) fn finder(&self) -> &DictTables {
        &self.finder
    }
}

/// Tables from a max-level parse of `samples` with `content` as history:
/// literal code lengths from the literal bytes, tANS counts from the
/// codes, every symbol given at least one count so any object can reuse
/// them.
fn tables_for(content: &[u8], samples: &[&[u8]]) -> Tables {
    let mut lit = [1u64; 256];
    let (mut ll, mut ml, mut off) = ([1u32; LL_SYMBOLS], [1u32; ML_SYMBOLS], [1u32; OFF_SYMBOLS]);
    let mut t = crate::v7_encode::DfastTables::new();
    let mut scratch = crate::v7_encode::EncScratch::new();
    let (mut seqs, mut lits): (Vec<Sequence>, Vec<u8>) = (Vec::new(), Vec::new());
    let mut joined = Vec::with_capacity(content.len() + crate::format::MAX_BLOCK_SIZE);
    for s in samples {
        joined.clear();
        joined.extend_from_slice(content);
        joined.extend_from_slice(&s[..s.len().min(crate::format::MAX_BLOCK_SIZE)]);
        t.clear();
        t.seed(&joined, content.len());
        seqs.clear();
        lits.clear();
        let mut reps = [1u32, 4, 8];
        crate::v7_encode::find_sequences_dfast(&joined, content.len(), joined.len() - content.len(), &mut t, &mut reps, &mut seqs, &mut lits, &mut scratch);
        for &b in &lits {
            lit[b as usize] += 1;
        }
        let mut r = Reps::new();
        for q in &seqs {
            ll[ll_code(q.lit_len).0 as usize] += 1;
            if q.match_len == 0 {
                continue;
            }
            ml[ml_code(q.match_len).0 as usize] += 1;
            off[r.code_for(q.offset).0 as usize] += 1;
        }
    }
    Tables {
        lit_lengths: Some(huff8::lengths_for(&lit)),
        lit_codes: None,
        ll: Some(tans::normalize(&ll, LL_SYMBOLS).into()),
        ml: Some(tans::normalize(&ml, ML_SYMBOLS).into()),
        off: Some(tans::normalize(&off, OFF_SYMBOLS).into()),
        seq_enc: None,
    }
}

/// The dictionary content, by cover (zstd's FASTCOVER shape): every
/// 8-byte string (a "dmer") in the samples is counted in a hashed table;
/// the samples' concatenation is cut into as many epochs as the budget
/// has segments, and each epoch contributes its best `SEGMENT`-byte
/// window (the sum of its dmers' counts, kept as a running sum), whose
/// dmers are then zeroed so later picks cover new strings. Taken
/// segments go in from the end, so the most valuable (the early epochs',
/// picked while every count was whole) sit at the shortest offsets.
/// Linear in the sample bytes.
fn select_content(samples: &[&[u8]], budget: usize) -> Vec<u8> {
    let segment = SEGMENT;
    const D: usize = 8;
    const FBITS: u32 = 20;
    let hash = |w: &[u8]| (u64::from_le_bytes(w.try_into().unwrap()).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - FBITS)) as usize;
    let joined: Vec<u8> = samples.concat();
    let n_segments = budget / segment;
    if n_segments == 0 || joined.len() < segment {
        return Vec::new();
    }
    let mut freq = vec![0u32; 1 << FBITS];
    for w in joined.windows(D) {
        freq[hash(w)] += 1;
    }
    let epoch = (joined.len() / n_segments).max(segment);
    let per_seg = segment - D + 1;
    // A window's score counts each distinct dmer once (a run of one
    // string is worth one string): `inside` holds how many times each
    // dmer is in the window, and a dmer's count is added when it enters
    // and taken away when its last copy leaves.
    let mut inside = vec![0u16; 1 << FBITS];
    let mut taken: Vec<usize> = Vec::new(); // offsets into `joined`, first epoch first
    let mut start = 0;
    while start + segment <= joined.len() {
        let e = &joined[start..(start + epoch).min(joined.len())];
        let dmers = e.len() - D + 1;
        let mut sum = 0u64;
        for i in 0..per_seg {
            let h = hash(&e[i..i + D]);
            if inside[h] == 0 {
                sum += freq[h] as u64;
            }
            inside[h] += 1;
        }
        let mut best = (sum, 0usize);
        for off in 1..=dmers.saturating_sub(per_seg) {
            let h = hash(&e[off + per_seg - 1..off + per_seg - 1 + D]);
            if inside[h] == 0 {
                sum += freq[h] as u64;
            }
            inside[h] += 1;
            let h = hash(&e[off - 1..off - 1 + D]);
            inside[h] -= 1;
            if inside[h] == 0 {
                sum -= freq[h] as u64;
            }
            if sum > best.0 {
                best = (sum, off);
            }
        }
        // Empty the window's counts for the next epoch.
        let last = dmers.saturating_sub(per_seg);
        for i in last..last + per_seg {
            inside[hash(&e[i..i + D])] -= 1;
        }
        if best.0 > 0 {
            for w in e[best.1..best.1 + segment].windows(D) {
                freq[hash(w)] = 0;
            }
            taken.push(start + best.1);
        }
        start += epoch;
    }
    taken.truncate(n_segments);
    let mut content = Vec::with_capacity(taken.len() * segment);
    for &off in taken.iter().rev() {
        content.extend_from_slice(&joined[off..off + segment]);
    }
    content
}
