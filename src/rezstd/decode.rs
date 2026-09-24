//! A plain zstd frame decoder (RFC 8878): one frame, no dictionary,
//! the checksum verified when present. It exists so that `reproduce` can get a
//! frame's content and so the round trips are checked against the
//! format rather than the encoder alone. Where the format's edge rules
//! decide how many symbols a stream holds (the Huffman weights' FSE
//! stream ends by running out of bits), the reference decoder's
//! bit-reader is followed.

use super::block::{LL_BITS, LL_DEFAULT_NORM, LL_DEFAULT_NORM_LOG, MAX_LL, MAX_ML, MAX_OFF, ML_BITS, ML_DEFAULT_NORM, ML_DEFAULT_NORM_LOG, OF_DEFAULT_NORM, OF_DEFAULT_NORM_LOG};
use super::fse::highbit;
use super::xxh64::xxh64;
use super::{BLOCK_SIZE_MAX, MAGIC};

/// `BIT_DStream_t`: bits taken from the end of a stream, the last
/// byte's highest set bit marking where they start.
struct BitReader<'a> {
    s: &'a [u8],
    /// Bytes below this index are not in the container yet.
    ptr: usize,
    container: u64,
    /// Bits of the container already taken, from its top.
    consumed: u32,
}

impl<'a> BitReader<'a> {
    fn new(s: &'a [u8]) -> Option<BitReader<'a>> {
        let last = *s.last()?;
        if last == 0 {
            return None;
        }
        let mark = 8 - highbit(last as u32);
        if s.len() >= 8 {
            let ptr = s.len() - 8;
            Some(BitReader { s, ptr, container: u64::from_le_bytes(s[ptr..].try_into().unwrap()), consumed: mark })
        } else {
            let mut bytes = [0u8; 8];
            bytes[..s.len()].copy_from_slice(s);
            Some(BitReader { s, ptr: 0, container: u64::from_le_bytes(bytes), consumed: mark + (8 - s.len() as u32) * 8 })
        }
    }

    /// `BIT_reloadDStream`: earlier bytes pulled in as far as room and
    /// input allow.
    fn refill(&mut self) {
        while self.consumed >= 8 && self.ptr > 0 {
            self.ptr -= 1;
            self.consumed -= 8;
            self.container = (self.container << 8) | self.s[self.ptr] as u64;
        }
    }

    /// The next `n` bits (at most 32) without taking them; bits past
    /// the start read as zero.
    fn peek(&mut self, n: u32) -> u64 {
        self.refill();
        if n == 0 {
            return 0;
        }
        self.container.wrapping_shl(self.consumed) >> (64 - n)
    }

    fn read(&mut self, n: u32) -> u64 {
        let v = self.peek(n);
        self.consumed += n;
        v
    }

    /// `BIT_DStream_overflow`: more bits taken than the stream had.
    fn overflowed(&mut self) -> bool {
        self.refill();
        self.consumed > 64
    }

    /// `BIT_endOfDStream`: every bit taken, none beyond.
    fn finished(&mut self) -> bool {
        self.refill();
        self.ptr == 0 && self.consumed == 64
    }
}

/// `FSE_readNCount`: the normalized counts of a table description;
/// returns them with the highest symbol, the table log and the bytes
/// read.
fn read_ncount(src: &[u8], max_symbol: usize) -> Option<(Vec<i16>, usize, u32, usize)> {
    if src.is_empty() {
        return None;
    }
    let bit = |pos: usize| -> u64 { src.get(pos >> 3).map_or(0, |&b| (b >> (pos & 7)) as u64 & 1) };
    let peek = |pos: usize, n: u32| -> u64 { (0..n).map(|i| bit(pos + i as usize) << i).sum() };
    let mut pos = 0usize;
    let log = (peek(0, 4) + 5) as u32;
    pos += 4;
    if log > 15 {
        return None;
    }
    let mut norm = vec![0i16; max_symbol + 1];
    let mut remaining = (1i32 << log) + 1;
    let mut threshold = 1i32 << log;
    let mut nb_bits = log + 1;
    let mut charnum = 0usize;
    let mut previous0 = false;
    loop {
        if previous0 {
            while peek(pos, 2) == 3 {
                charnum += 3;
                pos += 2;
                if charnum > max_symbol + 1 {
                    return None;
                }
            }
            charnum += peek(pos, 2) as usize;
            pos += 2;
            if charnum > max_symbol {
                break;
            }
        }
        let max = (2 * threshold - 1) - remaining;
        let v = peek(pos, nb_bits) as i32;
        let mut count;
        if (v & (threshold - 1)) < max {
            count = v & (threshold - 1);
            pos += nb_bits as usize - 1;
        } else {
            count = v & (2 * threshold - 1);
            if count >= threshold {
                count -= max;
            }
            pos += nb_bits as usize;
        }
        count -= 1;
        remaining -= count.abs();
        norm[charnum] = count as i16;
        charnum += 1;
        previous0 = count == 0;
        if remaining < threshold {
            if remaining <= 1 {
                break;
            }
            nb_bits = highbit(remaining as u32) + 1;
            threshold = 1 << (nb_bits - 1);
        }
        if charnum > max_symbol {
            break;
        }
    }
    if remaining != 1 || charnum > max_symbol + 1 {
        return None;
    }
    let bytes = pos.div_ceil(8);
    if bytes > src.len() {
        return None;
    }
    Some((norm, charnum - 1, log, bytes))
}

/// An FSE decoding table entry: the symbol, the bits to read and the
/// base of the next state.
#[derive(Clone, Copy, Default)]
struct DEntry {
    symbol: u8,
    nbits: u8,
    next: u16,
}

/// `FSE_DTable`.
#[derive(Clone)]
struct FseDTable {
    log: u32,
    entries: Vec<DEntry>,
}

impl FseDTable {
    /// `FSE_buildDTable`.
    fn build(norm: &[i16], max_symbol: usize, log: u32) -> Option<FseDTable> {
        let size = 1usize << log;
        let mask = size - 1;
        let step = (size >> 1) + (size >> 3) + 3;
        let mut entries = vec![DEntry::default(); size];
        let mut symbol_next = vec![0u16; max_symbol + 1];
        let mut high_threshold = size - 1;
        for s in 0..=max_symbol {
            if norm[s] == -1 {
                entries[high_threshold].symbol = s as u8;
                high_threshold = high_threshold.checked_sub(1)?;
                symbol_next[s] = 1;
            } else {
                symbol_next[s] = norm[s] as u16;
            }
        }
        let mut position = 0usize;
        for s in 0..=max_symbol {
            for _ in 0..norm[s].max(0) {
                entries[position].symbol = s as u8;
                position = (position + step) & mask;
                while position > high_threshold {
                    position = (position + step) & mask;
                }
            }
        }
        if position != 0 {
            return None;
        }
        for u in 0..size {
            let s = entries[u].symbol as usize;
            let next = symbol_next[s];
            symbol_next[s] += 1;
            if next == 0 {
                return None;
            }
            let nbits = log - highbit(next as u32);
            entries[u].nbits = nbits as u8;
            entries[u].next = ((next as u32) << nbits).wrapping_sub(size as u32) as u16;
        }
        Some(FseDTable { log, entries })
    }

    fn rle(symbol: u8) -> FseDTable {
        FseDTable { log: 0, entries: vec![DEntry { symbol, nbits: 0, next: 0 }] }
    }
}

/// A Huffman decoding table: (symbol, code length) by the next `log` bits.
struct HufDTable {
    log: u32,
    entries: Vec<(u8, u8)>,
}

/// The Huffman weights coded through FSE, as `FSE_decompress_wksp`
/// reads them: symbols alternate between two states until the
/// bitstream runs dry.
fn fse_decode_weights(src: &[u8]) -> Option<Vec<u8>> {
    let (norm, max_symbol, log, used) = read_ncount(src, 255)?;
    if log > 6 {
        return None;
    }
    let table = FseDTable::build(&norm, max_symbol, log)?;
    let mut r = BitReader::new(&src[used..])?;
    let mut state1 = r.read(log) as usize;
    let mut state2 = r.read(log) as usize;
    let mut out = Vec::new();
    let decode = |r: &mut BitReader, state: &mut usize| -> u8 {
        let e = table.entries[*state];
        *state = e.next as usize + r.read(e.nbits as u32) as usize;
        e.symbol
    };
    loop {
        if out.len() > 253 {
            return None;
        }
        out.push(decode(&mut r, &mut state1));
        if r.overflowed() {
            out.push(table.entries[state2].symbol);
            break;
        }
        if out.len() > 253 {
            return None;
        }
        out.push(decode(&mut r, &mut state2));
        if r.overflowed() {
            out.push(table.entries[state1].symbol);
            break;
        }
    }
    Some(out)
}

/// `HUF_readCTable` as a decoding table; returns the bytes it took.
fn read_huf_table(src: &[u8]) -> Option<(HufDTable, usize)> {
    let hb = *src.first()? as usize;
    let (mut weights, used) = if hb >= 128 {
        let n = hb - 127;
        let used = 1 + n.div_ceil(2);
        if src.len() < used {
            return None;
        }
        let w: Vec<u8> = (0..n).map(|i| if i % 2 == 0 { src[1 + i / 2] >> 4 } else { src[1 + i / 2] & 15 }).collect();
        (w, used)
    } else {
        if src.len() < 1 + hb {
            return None;
        }
        (fse_decode_weights(&src[1..1 + hb])?, 1 + hb)
    };
    if weights.iter().any(|&w| w > 12) {
        return None;
    }
    let total: u32 = weights.iter().map(|&w| (1u32 << w) >> 1).sum();
    if total == 0 {
        return None;
    }
    let log = highbit(total) + 1;
    if log > 12 {
        return None;
    }
    let rest = (1u32 << log) - total;
    if rest & (rest - 1) != 0 {
        return None;
    }
    weights.push((highbit(rest) + 1) as u8);
    if weights.len() > 256 {
        return None;
    }
    let nbits: Vec<u32> = weights.iter().map(|&w| if w > 0 { log + 1 - w as u32 } else { 0 }).collect();
    let mut nb_per_rank = [0u32; 14];
    for &n in &nbits {
        nb_per_rank[n as usize] += 1;
    }
    let mut val_per_rank = [0u32; 14];
    let mut min = 0u32;
    for n in (1..=log as usize).rev() {
        val_per_rank[n] = min;
        min += nb_per_rank[n];
        min >>= 1;
    }
    let mut entries = vec![(0u8, 0u8); 1 << log];
    for (s, &n) in nbits.iter().enumerate() {
        if n == 0 {
            continue;
        }
        let val = val_per_rank[n as usize];
        val_per_rank[n as usize] += 1;
        let start = (val << (log - n)) as usize;
        for e in &mut entries[start..start + (1 << (log - n))] {
            *e = (s as u8, n as u8);
        }
    }
    Some((HufDTable { log, entries }, used))
}

/// One Huffman stream of `n` symbols.
fn huf_decode(src: &[u8], n: usize, t: &HufDTable, out: &mut Vec<u8>) -> Option<()> {
    let mut r = BitReader::new(src)?;
    for _ in 0..n {
        let (symbol, nbits) = t.entries[r.peek(t.log) as usize];
        r.consumed += nbits as u32;
        out.push(symbol);
    }
    r.finished().then_some(())
}

/// The tables a block leaves for the next (`ZSTD_entropyDTables_t`).
struct Tables {
    huf: Option<HufDTable>,
    ll: Option<FseDTable>,
    of: Option<FseDTable>,
    ml: Option<FseDTable>,
    rep: [u32; 3],
}

/// The literals section: its bytes and what follows it.
fn literals<'a>(src: &'a [u8], t: &mut Tables) -> Option<(Vec<u8>, &'a [u8])> {
    let b0 = *src.first()? as usize;
    let kind = b0 & 3;
    let size_format = (b0 >> 2) & 3;
    if kind < 2 {
        let (hdr, size) = match size_format {
            0 | 2 => (1, b0 >> 3),
            1 => (2, (b0 >> 4) | (*src.get(1)? as usize) << 4),
            _ => (3, (b0 >> 4) | (*src.get(1)? as usize) << 4 | (*src.get(2)? as usize) << 12),
        };
        if size > BLOCK_SIZE_MAX {
            return None;
        }
        return if kind == 0 {
            Some((src.get(hdr..hdr + size)?.to_vec(), &src[hdr + size..]))
        } else {
            Some((vec![*src.get(hdr)?; size], &src[hdr + 1..]))
        };
    }
    let mut v = [0u8; 8];
    let hdr = [3, 3, 4, 5][size_format];
    v[..hdr].copy_from_slice(src.get(..hdr)?);
    let v = u64::from_le_bytes(v) as usize;
    let (streams, regen, comp) = match size_format {
        0 => (1, (v >> 4) & 0x3ff, (v >> 14) & 0x3ff),
        1 => (4, (v >> 4) & 0x3ff, (v >> 14) & 0x3ff),
        2 => (4, (v >> 4) & 0x3fff, (v >> 18) & 0x3fff),
        _ => (4, (v >> 4) & 0x3ffff, (v >> 22) & 0x3ffff),
    };
    let mut data = src.get(hdr..hdr + comp)?;
    if kind == 2 {
        let (table, used) = read_huf_table(data)?;
        t.huf = Some(table);
        data = &data[used..];
    }
    let table = t.huf.as_ref()?;
    let mut lits = Vec::with_capacity(regen);
    if streams == 1 {
        huf_decode(data, regen, table, &mut lits)?;
    } else {
        if data.len() < 6 {
            return None;
        }
        let sizes: Vec<usize> = (0..3).map(|i| u16::from_le_bytes([data[2 * i], data[2 * i + 1]]) as usize).collect();
        let segment = regen.div_ceil(4);
        if 3 * segment > regen {
            return None;
        }
        let mut at = 6usize;
        for (i, &size) in sizes.iter().chain(std::iter::once(&(data.len() - 6 - sizes.iter().sum::<usize>()))).enumerate() {
            let n = if i < 3 { segment } else { regen - 3 * segment };
            let end = at.checked_add(size)?;
            huf_decode(data.get(at..end)?, n, table, &mut lits)?;
            at = end;
        }
    }
    Some((lits, &src[hdr + comp..]))
}

/// A sequence table by its coding mode; returns the bytes it took.
fn seq_table(src: &[u8], mode: u8, max_symbol: usize, max_log: u32, default: (&[i16], u32), current: &mut Option<FseDTable>) -> Option<usize> {
    match mode {
        0 => {
            *current = Some(FseDTable::build(default.0, default.0.len() - 1, default.1)?);
            Some(0)
        }
        1 => {
            let symbol = *src.first()?;
            if symbol as usize > max_symbol {
                return None;
            }
            *current = Some(FseDTable::rle(symbol));
            Some(1)
        }
        2 => {
            let (norm, max, log, used) = read_ncount(src, max_symbol)?;
            if log > max_log {
                return None;
            }
            *current = Some(FseDTable::build(&norm, max, log)?);
            Some(used)
        }
        _ => current.as_ref().map(|_| 0),
    }
}

/// The base value of each literal length code.
fn ll_base() -> [u32; MAX_LL + 1] {
    let mut b = [0u32; MAX_LL + 1];
    for c in 1..=MAX_LL {
        b[c] = b[c - 1] + (1 << LL_BITS[c - 1]);
    }
    b
}

/// The base value of each match length code.
fn ml_base() -> [u32; MAX_ML + 1] {
    let mut b = [3u32; MAX_ML + 1];
    for c in 1..=MAX_ML {
        b[c] = b[c - 1] + (1 << ML_BITS[c - 1]);
    }
    b
}

/// A compressed block appended to `out`.
fn block(src: &[u8], out: &mut Vec<u8>, t: &mut Tables) -> Option<()> {
    let (lits, rest) = literals(src, t)?;
    let b0 = *rest.first()? as usize;
    let (nb_seq, mut at) = if b0 < 128 {
        (b0, 1)
    } else if b0 < 255 {
        (((b0 - 128) << 8) + *rest.get(1)? as usize, 2)
    } else {
        (u16::from_le_bytes([*rest.get(1)?, *rest.get(2)?]) as usize + 0x7F00, 3)
    };
    if nb_seq == 0 {
        if at != rest.len() {
            return None;
        }
        out.extend_from_slice(&lits);
        return Some(());
    }
    let modes = *rest.get(at)?;
    at += 1;
    at += seq_table(&rest[at..], modes >> 6, MAX_LL, 9, (&LL_DEFAULT_NORM, LL_DEFAULT_NORM_LOG), &mut t.ll)?;
    at += seq_table(&rest[at..], (modes >> 4) & 3, MAX_OFF, 8, (&OF_DEFAULT_NORM, OF_DEFAULT_NORM_LOG), &mut t.of)?;
    at += seq_table(&rest[at..], (modes >> 2) & 3, MAX_ML, 9, (&ML_DEFAULT_NORM, ML_DEFAULT_NORM_LOG), &mut t.ml)?;
    let (ll, of, ml) = (t.ll.as_ref()?, t.of.as_ref()?, t.ml.as_ref()?);
    let (ll_base, ml_base) = (ll_base(), ml_base());
    let mut r = BitReader::new(rest.get(at..)?)?;
    let mut ll_state = r.read(ll.log) as usize;
    let mut of_state = r.read(of.log) as usize;
    let mut ml_state = r.read(ml.log) as usize;
    let mut lit_at = 0usize;
    for i in 0..nb_seq {
        let (le, oe, me) = (*ll.entries.get(ll_state)?, *of.entries.get(of_state)?, *ml.entries.get(ml_state)?);
        let of_code = oe.symbol as u32;
        let off_base = (1u32 << of_code) + r.read(of_code) as u32;
        let match_len = ml_base[me.symbol as usize] as usize + r.read(ML_BITS[me.symbol as usize] as u32) as usize;
        let lit_len = ll_base[le.symbol as usize] as usize + r.read(LL_BITS[le.symbol as usize] as u32) as usize;
        let rep = &mut t.rep;
        let offset = if off_base > 3 {
            let o = off_base - 3;
            rep[2] = rep[1];
            rep[1] = rep[0];
            rep[0] = o;
            o
        } else {
            let r = off_base as usize - 1 + (lit_len == 0) as usize;
            if r == 0 {
                rep[0]
            } else {
                let o = if r == 3 { rep[0].wrapping_sub(1) } else { rep[r] };
                if r >= 2 {
                    rep[2] = rep[1];
                }
                rep[1] = rep[0];
                rep[0] = o;
                o
            }
        } as usize;
        if std::env::var("REZSTD_DUMP").is_ok() { eprintln!("seq {} ll {} off {} ml {} at {}", i, lit_len, offset, match_len, out.len() + lit_len); }
        out.extend_from_slice(lits.get(lit_at..lit_at + lit_len)?);
        lit_at += lit_len;
        if offset == 0 || offset > out.len() {
            return None;
        }
        let from = out.len() - offset;
        for k in 0..match_len {
            let b = out[from + k];
            out.push(b);
        }
        if i + 1 < nb_seq {
            ll_state = le.next as usize + r.read(le.nbits as u32) as usize;
            ml_state = me.next as usize + r.read(me.nbits as u32) as usize;
            of_state = oe.next as usize + r.read(oe.nbits as u32) as usize;
        }
    }
    if !r.finished() {
        return None;
    }
    out.extend_from_slice(&lits[lit_at..]);
    Some(())
}

/// The content of a zstd frame, or None for a malformed one, one that
/// needs a dictionary, or bytes after the frame.
pub fn decompress(frame: &[u8]) -> Option<Vec<u8>> {
    decompress_blocks(frame).map(|(out, _)| out)
}

/// `decompress` with the size of each block's content as well.
pub(crate) fn decompress_blocks(frame: &[u8]) -> Option<(Vec<u8>, Vec<usize>)> {
    if frame.len() < 5 || u32::from_le_bytes(frame[..4].try_into().unwrap()) != MAGIC {
        return None;
    }
    let fhd = frame[4];
    let single = (fhd >> 5) & 1 == 1;
    if (fhd >> 3) & 1 != 0 || fhd & 3 != 0 {
        return None;
    }
    let checksum = (fhd >> 2) & 1 == 1;
    let mut p = 5;
    if !single {
        p += 1;
    }
    let take = |p: &mut usize, n: usize| -> Option<u64> {
        let mut b = [0u8; 8];
        b[..n].copy_from_slice(frame.get(*p..*p + n)?);
        *p += n;
        Some(u64::from_le_bytes(b))
    };
    let content_size = match fhd >> 6 {
        0 => {
            if single {
                Some(take(&mut p, 1)?)
            } else {
                None
            }
        }
        1 => Some(take(&mut p, 2)? + 256),
        2 => Some(take(&mut p, 4)?),
        _ => Some(take(&mut p, 8)?),
    };
    let mut out = Vec::with_capacity(content_size.unwrap_or(0).min(1 << 30) as usize);
    let mut t = Tables { huf: None, ll: None, of: None, ml: None, rep: [1, 4, 8] };
    let mut blocks = Vec::new();
    loop {
        let h = take(&mut p, 3)? as usize;
        let last = h & 1 == 1;
        let size = h >> 3;
        let before = out.len();
        match (h >> 1) & 3 {
            0 => {
                out.extend_from_slice(frame.get(p..p + size)?);
                p += size;
            }
            1 => {
                if size > BLOCK_SIZE_MAX {
                    return None;
                }
                out.resize(out.len() + size, *frame.get(p)?);
                p += 1;
            }
            2 => {
                if size > BLOCK_SIZE_MAX {
                    return None;
                }
                block(frame.get(p..p + size)?, &mut out, &mut t)?;
                if out.len() - before > BLOCK_SIZE_MAX {
                    return None;
                }
                p += size;
            }
            _ => return None,
        }
        blocks.push(out.len() - before);
        if last {
            break;
        }
    }
    if checksum {
        let want = u32::from_le_bytes(frame.get(p..p + 4)?.try_into().unwrap());
        if want != xxh64(&out) as u32 {
            return None;
        }
        p += 4;
    }
    if p != frame.len() || content_size.is_some_and(|n| n != out.len() as u64) {
        return None;
    }
    Some((out, blocks))
}
