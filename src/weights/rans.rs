//! Static rANS for the model-file mode: 32-bit states, 16-bit renormalisation,
//! eight interleaved lanes, tables of up to 4096 slots chosen per context.
//!
//! A stream of `n` symbols is cut into eight lanes of `n / 8` symbols (the last
//! lane takes the remainder); the decoder steps the lanes in turn and reads
//! its 16-bit words from one forward cursor, the encoder runs the same
//! sequence backwards. A context function maps a position and the symbol
//! before it in the same lane (0 at a lane's start) to one of the stream's
//! tables, so a stream is coded under any number of contexts at the cost of
//! the tables only.
//!
//! Stream: eight `u32` initial states, then the words. A decoder checks that
//! every lane ends on the state the encoder began with and that every word
//! was read, so a stream that is cut or altered is refused.

use crate::error::{CodecError, Result};
use crate::record::{get_varint, put_varint};

pub(crate) const LANES: usize = 8;
const RANS_L: u32 = 1 << 16;
/// Symbols at most 255 and tables at most 4096 slots.
pub(crate) const MAX_BITS: u8 = 12;
const MIN_BITS: u8 = 8;
/// Contexts under this many symbols share a table, whatever it costs.
const MIN_CTX_SYMBOLS: u64 = 96;
/// Bits a table entry costs in the serialised form, on average.
const ENTRY_BITS: f64 = 20.0;

fn bad(msg: &'static str) -> CodecError {
    CodecError::CorruptedBitstream(msg)
}

/// The tables of one stream: `freq` tables over `alphabet` symbols, each summing
/// to `1 << bits`, and the table of every context.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Tables {
    pub bits: u8,
    pub alphabet: usize,
    pub map: Vec<u16>,
    pub freq: Vec<Vec<u16>>,
}

/// `h` (counts over `h.len()` symbols) scaled to sum to `1 << bits`: a present
/// symbol gets at least 1 and at most `(1 << bits) - 1`.
fn normalize(h: &[u32], bits: u8) -> Vec<u16> {
    let m = 1u64 << bits;
    let total: u64 = h.iter().map(|&c| c as u64).sum();
    let mut f = vec![0u16; h.len()];
    let nnz = h.iter().filter(|&&c| c > 0).count();
    if total == 0 {
        f[0] = (m - 1) as u16;
        if f.len() > 1 {
            f[1] = 1;
        }
        return f;
    }
    if nnz == 1 {
        let s = h.iter().position(|&c| c > 0).unwrap();
        f[s] = (m - 1) as u16;
        f[(s + 1) % h.len()] = 1;
        return f;
    }
    let mut sum = 0u64;
    for (s, &c) in h.iter().enumerate() {
        if c > 0 {
            let v = ((c as u64 * m) / total).max(1);
            f[s] = v as u16;
            sum += v;
        }
    }
    while sum != m {
        if sum > m {
            // the symbols that lose least by one slot
            let mut cand: Vec<(f64, usize)> = (0..h.len())
                .filter(|&s| f[s] > 1)
                .map(|s| (h[s] as f64 * ((f[s] as f64).ln() - ((f[s] - 1) as f64).ln()), s))
                .collect();
            cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let mut d = (sum - m) as usize;
            for &(_, s) in cand.iter().take(d.min(cand.len())) {
                f[s] -= 1;
                sum -= 1;
                d -= 1;
            }
            let _ = d;
        } else {
            let mut cand: Vec<(f64, usize)> = (0..h.len())
                .filter(|&s| h[s] > 0 && (f[s] as u64) < m - 1)
                .map(|s| (h[s] as f64 * (((f[s] + 1) as f64).ln() - (f[s] as f64).ln()), s))
                .collect();
            cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let d = (m - sum) as usize;
            for &(_, s) in cand.iter().take(d.min(cand.len())) {
                f[s] += 1;
                sum += 1;
            }
        }
    }
    f
}

/// Bits `h` costs under `f` (both over the same symbols), without a table.
fn cost(h: &[u32], f: &[u16], bits: u8) -> f64 {
    let mut c = 0.0;
    for (s, &n) in h.iter().enumerate() {
        if n > 0 {
            c += n as f64 * (bits as f64 - (f[s] as f64).log2());
        }
    }
    c
}

fn table_bits(f: &[u16]) -> f64 {
    24.0 + ENTRY_BITS * f.iter().filter(|&&x| x > 0).count() as f64
}

/// Tables for the histograms `hist` (`nctx` rows of `alphabet` counts): contexts too
/// thin to pay for a table of their own share one. Returns the tables, the bits
/// the symbols will take under them and the bits the tables take.
pub(crate) fn plan(hist: &[u32], nctx: usize, alphabet: usize, bits: u8) -> (Tables, f64, f64) {
    debug_assert!(hist.len() == nctx * alphabet && (MIN_BITS..=MAX_BITS).contains(&bits) && alphabet >= 2 && alphabet <= 256);
    let row = |c: usize| &hist[c * alphabet..(c + 1) * alphabet];
    let totals: Vec<u64> = (0..nctx).map(|c| row(c).iter().map(|&x| x as u64).sum()).collect();
    // First guess: every context with enough symbols has a table; the rest share one.
    let mut own: Vec<bool> = totals.iter().map(|&t| t >= MIN_CTX_SYMBOLS).collect();
    // Is a table of its own worth what it costs, against the shared one? Two rounds.
    for _ in 0..2 {
        let mut shared = vec![0u32; alphabet];
        for c in 0..nctx {
            if !own[c] {
                for s in 0..alphabet {
                    shared[s] = shared[s].saturating_add(row(c)[s]);
                }
            }
        }
        let shared_total: u64 = shared.iter().map(|&x| x as u64).sum();
        let fshared = if shared_total > 0 { Some(normalize(&shared, bits)) } else { None };
        let mut changed = false;
        for c in 0..nctx {
            if !own[c] || totals[c] == 0 {
                continue;
            }
            // A context whose symbols the shared table cannot code, or that is thin, must stay or go on merit.
            let f = normalize(row(c), bits);
            let own_cost = cost(row(c), &f, bits) + table_bits(&f);
            let in_shared = match &fshared {
                Some(fs) if (0..alphabet).all(|s| row(c)[s] == 0 || fs[s] > 0) => Some(cost(row(c), fs, bits)),
                _ => None,
            };
            if let Some(sc) = in_shared {
                if sc <= own_cost {
                    own[c] = false;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    let mut shared = vec![0u32; alphabet];
    let mut any_shared = false;
    for c in 0..nctx {
        if !own[c] && totals[c] > 0 {
            any_shared = true;
            for s in 0..alphabet {
                shared[s] = shared[s].saturating_add(row(c)[s]);
            }
        }
    }
    let mut freq: Vec<Vec<u16>> = Vec::new();
    let mut map = vec![0u16; nctx];
    let (mut data_bits, mut tab_bits) = (0.0, 0.0);
    if any_shared || own.iter().all(|&o| !o) {
        let f = normalize(&shared, bits);
        data_bits += cost(&shared, &f, bits);
        tab_bits += table_bits(&f);
        freq.push(f);
    }
    let shared_idx = if freq.is_empty() { None } else { Some(0u16) };
    for c in 0..nctx {
        if own[c] && totals[c] > 0 {
            let f = normalize(row(c), bits);
            data_bits += cost(row(c), &f, bits);
            tab_bits += table_bits(&f);
            map[c] = freq.len() as u16;
            freq.push(f);
        } else if let Some(i) = shared_idx {
            map[c] = i;
        } else {
            // an empty context with no shared table: any table will do, never used
            map[c] = 0;
        }
    }
    if freq.is_empty() {
        freq.push(normalize(&vec![0u32; alphabet], bits));
    }
    (Tables { bits, alphabet, map, freq }, data_bits, tab_bits)
}

pub(crate) fn write_tables(t: &Tables, out: &mut Vec<u8>) {
    out.push(t.bits);
    put_varint(out, t.map.len() as u64);
    put_varint(out, t.freq.len() as u64);
    if t.freq.len() == t.map.len() && t.map.iter().enumerate().all(|(i, &m)| m as usize == i) {
        out.push(0);
    } else {
        out.push(1);
        for &m in &t.map {
            put_varint(out, m as u64);
        }
    }
    for f in &t.freq {
        let nnz = f.iter().filter(|&&x| x > 0).count();
        put_varint(out, nnz as u64);
        let mut next = 0usize;
        for (s, &x) in f.iter().enumerate() {
            if x > 0 {
                put_varint(out, (s - next) as u64);
                put_varint(out, x as u64);
                next = s + 1;
            }
        }
    }
}

pub(crate) fn read_tables(src: &[u8], pos: &mut usize, alphabet: usize, nctx: usize) -> Result<Tables> {
    let bits = *src.get(*pos).ok_or(bad("weights: truncated tables"))?;
    *pos += 1;
    if !(MIN_BITS..=MAX_BITS).contains(&bits) {
        return Err(bad("weights: table precision"));
    }
    let nc = get_varint(src, pos)? as usize;
    let ntab = get_varint(src, pos)? as usize;
    if nc != nctx || ntab == 0 || ntab > nctx {
        return Err(bad("weights: table counts"));
    }
    let identity = *src.get(*pos).ok_or(bad("weights: truncated tables"))? == 0;
    *pos += 1;
    let mut map = Vec::with_capacity(nctx);
    if identity {
        if ntab != nctx {
            return Err(bad("weights: table map"));
        }
        map.extend((0..nctx).map(|i| i as u16));
    } else {
        for _ in 0..nctx {
            let m = get_varint(src, pos)? as usize;
            if m >= ntab {
                return Err(bad("weights: table map"));
            }
            map.push(m as u16);
        }
    }
    let m = 1u32 << bits;
    let mut freq = Vec::with_capacity(ntab);
    for _ in 0..ntab {
        let nnz = get_varint(src, pos)? as usize;
        if nnz == 0 || nnz > alphabet {
            return Err(bad("weights: table entries"));
        }
        let mut f = vec![0u16; alphabet];
        let (mut next, mut sum) = (0usize, 0u32);
        for _ in 0..nnz {
            let gap = get_varint(src, pos)? as usize;
            let x = get_varint(src, pos)?;
            let s = next.checked_add(gap).ok_or(bad("weights: table symbol"))?;
            if s >= alphabet || x == 0 || x >= m as u64 {
                return Err(bad("weights: table entry"));
            }
            f[s] = x as u16;
            sum += x as u32;
            next = s + 1;
        }
        if sum != m {
            return Err(bad("weights: table sum"));
        }
        freq.push(f);
    }
    Ok(Tables { bits, alphabet, map, freq })
}

/// Decoding tables: for every table and slot, the symbol, the frequency
/// and the slot's offset into the symbol's range, in one word.
pub(crate) struct DecTables {
    pub bits: u8,
    /// `map.len() - 1`: every context index is masked with it, so none can be out of range.
    pub ctx_mask: usize,
    map: Vec<u32>,
    entries: Vec<u32>,
}

impl DecTables {
    pub fn new(t: &Tables) -> DecTables {
        let slots = 1usize << t.bits;
        let mut entries = vec![0u32; t.freq.len() * slots];
        for (ti, f) in t.freq.iter().enumerate() {
            let e = &mut entries[ti * slots..(ti + 1) * slots];
            let mut at = 0usize;
            for (s, &x) in f.iter().enumerate() {
                let x = x as usize;
                for k in 0..x {
                    e[at + k] = s as u32 | (x as u32) << 8 | (k as u32) << 20;
                }
                at += x;
            }
        }
        // the table of a context, as the offset of its first entry; contexts past the last are the first's
        let mut map: Vec<u32> = t.map.iter().map(|&m| (m as u32) << t.bits).collect();
        map.resize(t.map.len().next_power_of_two(), 0);
        DecTables { bits: t.bits, ctx_mask: map.len() - 1, map, entries }
    }
}

/// The table a position's symbol is coded under. A lane carries a byte of state (0 at its
/// start): `at` names the table from it, `next` gives it after a symbol is decoded. The
/// byte is the symbol before for a context from that symbol, or flags that a block's
/// symbols set and its end clears.
pub(crate) trait Ctx {
    fn at(&self, i: usize, st: u8) -> usize;
    fn next(&self, i: usize, sym: u8, st: u8) -> u8;
}

/// One table for everything.
pub(crate) struct NoCtx;
impl Ctx for NoCtx {
    #[inline(always)]
    fn at(&self, _: usize, _: u8) -> usize {
        0
    }
    #[inline(always)]
    fn next(&self, _: usize, _: u8, _: u8) -> u8 {
        0
    }
}

/// Encoder side of one table: (frequency, start) of every symbol, and the reciprocal of the
/// frequency that divides without a division (the high half of `x * (2^64 / f)` is `x / f` or one
/// less, for every `x` below 2^32, and the remainder says which).
struct EncTables {
    bits: u8,
    nctx: usize,
    map: Vec<u32>,
    fc: Vec<u32>,
    rcp: Vec<u64>,
}

impl EncTables {
    fn new(t: &Tables) -> EncTables {
        let mut fc = vec![0u32; t.freq.len() * 256];
        let mut rcp = vec![0u64; t.freq.len() * 256];
        for (ti, f) in t.freq.iter().enumerate() {
            let mut at = 0u32;
            for (s, &x) in f.iter().enumerate() {
                fc[ti * 256 + s] = x as u32 | at << 16;
                if x > 0 {
                    rcp[ti * 256 + s] = u64::MAX / x as u64;
                }
                at += x as u32;
            }
        }
        EncTables { bits: t.bits, nctx: t.map.len(), map: t.map.iter().map(|&m| m as u32 * 256).collect(), fc, rcp }
    }
}

/// `syms` coded under `t`; `ctx[i]` names the table of symbol `i` (none: the first context).
/// Every symbol must be one the tables cover.
pub(crate) fn encode(syms: &[u8], ctx: Option<&[u16]>, t: &Tables, out: &mut Vec<u8>) {
    let e = EncTables::new(t);
    let n = syms.len();
    let seg = n / LANES;
    let rem = n - seg * LANES;
    let mut x = [RANS_L; LANES];
    // The words, written from the back of the stream to the front as they come: at most one for every symbol.
    let mut words = vec![0u16; n + 8];
    let mut wp = n + 8;
    let bits = e.bits as u32;
    let nctx = e.nctx;
    macro_rules! put {
        ($lane:expr, $i:expr) => {{
            let i = $i;
            let c = ctx.map_or(0, |c| (c[i] as usize).min(nctx - 1));
            let k = e.map[c] as usize + syms[i] as usize;
            let (fc, m) = (e.fc[k], e.rcp[k]);
            let (f, start) = (fc & 0xFFFF, fc >> 16);
            debug_assert!(f > 0, "symbol outside the table");
            let mut s = x[$lane];
            let emit = (s >= f << (32 - bits)) as usize;
            // written either way, kept (the cursor moves) only when the state has to give up its low bits
            words[wp - 1] = s as u16;
            wp -= emit;
            s >>= 16 * emit as u32;
            let q0 = ((m as u128 * s as u128) >> 64) as u32;
            let r0 = s - q0 * f;
            let up = (r0 >= f) as u32;
            x[$lane] = ((q0 + up) << bits) + (r0 - up * f) + start;
        }};
    }
    for r in (0..rem).rev() {
        put!(LANES - 1, LANES * seg + r);
    }
    for t in (0..seg).rev() {
        put!(7, 7 * seg + t);
        put!(6, 6 * seg + t);
        put!(5, 5 * seg + t);
        put!(4, 4 * seg + t);
        put!(3, 3 * seg + t);
        put!(2, 2 * seg + t);
        put!(1, seg + t);
        put!(0, t);
    }
    for s in x {
        out.extend_from_slice(&s.to_le_bytes());
    }
    for w in &words[wp..n + 8] {
        out.extend_from_slice(&w.to_le_bytes());
    }
}

/// `n` symbols of `data` into `out`, each under the table `c` names. (Not inlined: the loop is the
/// hot path of every stream, and a copy of it for each context function in each caller is slower.)
#[inline(never)]
pub(crate) fn decode<C: Ctx>(data: &[u8], n: usize, t: &DecTables, c: &C, out: &mut [u8]) -> Result<()> {
    if out.len() != n || data.len() < 4 * LANES {
        return Err(bad("weights: stream length"));
    }
    let bits = t.bits as u32;
    let mask = (1u32 << bits) - 1;
    let seg = n / LANES;
    let rem = n - seg * LANES;
    let mut x = [0u32; LANES];
    for (l, s) in x.iter_mut().enumerate() {
        *s = u32::from_le_bytes(data[4 * l..4 * l + 4].try_into().unwrap());
        if *s < RANS_L {
            return Err(bad("weights: stream state"));
        }
    }
    let real = data.len() - 4 * LANES;
    // The words, and two zero bytes after them for a stream that asks for more than it has
    // (refused below): the loop needs no check of its own.
    let mut padded = Vec::with_capacity(real + 2);
    padded.extend_from_slice(&data[4 * LANES..]);
    padded.extend_from_slice(&[0, 0]);
    let words = padded.as_ptr();
    let mut w = 0usize;
    let mut st = [0u8; LANES];
    let ctx_mask = t.ctx_mask;
    let entries = &t.entries[..];
    let map = &t.map[..];
    macro_rules! step {
        ($lane:expr, $i:expr) => {{
            let i = $i;
            let cx = c.at(i, st[$lane]) & ctx_mask;
            // SAFETY: `map` entries are table offsets inside `entries` and the slot is below 1 << bits.
            let e = unsafe { *entries.get_unchecked((*map.get_unchecked(cx)) as usize + (x[$lane] & mask) as usize) };
            let s = e as u8;
            // SAFETY: `i < n == out.len()`.
            unsafe { *out.get_unchecked_mut(i) = s };
            st[$lane] = c.next(i, s, st[$lane]);
            let v = ((e >> 8) & 0xFFF) * (x[$lane] >> bits) + (e >> 20);
            // Refill without a branch (a mispredicted one costs more than the whole step): the word is read
            // either way and used when the state has run low.
            let need = (v < RANS_L) as u32;
            // SAFETY: `w <= real` as long as the stream is well formed, and a read at `real` takes the padding; a
            // stream that reads on is refused below, after the loop, before any use of what it made.
            let word = unsafe { u16::from_le(std::ptr::read_unaligned(words.add(w.min(real)) as *const u16)) } as u32;
            x[$lane] = (v << (need << 4)) | (word & 0u32.wrapping_sub(need));
            w += (need as usize) << 1;
        }};
    }
    for t in 0..seg {
        step!(0, t);
        step!(1, seg + t);
        step!(2, 2 * seg + t);
        step!(3, 3 * seg + t);
        step!(4, 4 * seg + t);
        step!(5, 5 * seg + t);
        step!(6, 6 * seg + t);
        step!(7, 7 * seg + t);
    }
    for r in 0..rem {
        step!(LANES - 1, LANES * seg + r);
    }
    if w != real || x.iter().any(|&s| s != RANS_L) {
        return Err(bad("weights: stream does not close"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: u64) -> impl FnMut() -> u64 {
        let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
        move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        }
    }

    struct PrevCtx(u8);
    impl Ctx for PrevCtx {
        fn at(&self, _: usize, st: u8) -> usize {
            (st >> self.0) as usize
        }
        fn next(&self, _: usize, sym: u8, _: u8) -> u8 {
            sym
        }
    }

    fn lane_ctx(syms: &[u8], shift: u8) -> Vec<u16> {
        let n = syms.len();
        let seg = n / LANES;
        (0..n)
            .map(|i| {
                let start = i == 0 || (seg > 0 && i % seg == 0 && i / seg < LANES);
                if start { 0 } else { (syms[i - 1] >> shift) as u16 }
            })
            .collect()
    }

    fn roundtrip(syms: &[u8], alphabet: usize, nctx: usize, shift: Option<u8>) -> usize {
        let ctx = shift.map(|s| lane_ctx(syms, s));
        let mut hist = vec![0u32; nctx * alphabet];
        for (i, &s) in syms.iter().enumerate() {
            let c = ctx.as_ref().map_or(0, |c| c[i] as usize);
            hist[c * alphabet + s as usize] += 1;
        }
        let (t, _, _) = plan(&hist, nctx, alphabet, 12);
        let mut ser = Vec::new();
        write_tables(&t, &mut ser);
        let mut p = 0;
        let t2 = read_tables(&ser, &mut p, alphabet, nctx).unwrap();
        assert_eq!(p, ser.len());
        assert_eq!(t, t2);
        let mut enc = Vec::new();
        encode(syms, ctx.as_deref(), &t2, &mut enc);
        let d = DecTables::new(&t2);
        let mut out = vec![0u8; syms.len()];
        match shift {
            None => decode(&enc, syms.len(), &d, &NoCtx, &mut out).unwrap(),
            Some(s) => decode(&enc, syms.len(), &d, &PrevCtx(s), &mut out).unwrap(),
        }
        assert_eq!(out, syms);
        enc.len()
    }

    #[test]
    fn round_trips_every_length() {
        let mut r = rng(1);
        for n in (0..70usize).chain([255, 256, 257, 1000, 4097, 100_003]) {
            let skew: Vec<u8> = (0..n).map(|_| ((r() % 7) * (r() % 5)) as u8).collect();
            roundtrip(&skew, 64, 1, None);
            roundtrip(&skew, 64, 64, Some(0));
        }
    }

    #[test]
    fn near_entropy() {
        let mut r = rng(2);
        let n = 1 << 20;
        let syms: Vec<u8> = (0..n).map(|_| (r() % 4 + r() % 4) as u8).collect();
        let bytes = roundtrip(&syms, 16, 1, None);
        // the entropy of the sum of two dice with four faces: 1/16, 2/16, 3/16, 4/16, 3/16, 2/16, 1/16
        let h = 2.65565f64;
        let ideal = (h * n as f64 / 8.0) as usize;
        assert!(bytes < ideal + ideal / 200 + 64, "{} vs {}", bytes, ideal);
    }

    #[test]
    fn contexts_pay() {
        // each symbol is the one before it, mostly
        let mut r = rng(3);
        let mut syms = vec![0u8; 400_000];
        for i in 1..syms.len() {
            syms[i] = if r() % 16 == 0 { (r() % 8) as u8 } else { syms[i - 1] };
        }
        let plain = roundtrip(&syms, 8, 1, None);
        let ctx = roundtrip(&syms, 8, 8, Some(0));
        assert!(ctx * 2 < plain, "{} vs {}", ctx, plain);
    }

    #[test]
    fn one_symbol_streams() {
        for n in [0usize, 1, 5, 4096] {
            roundtrip(&vec![7u8; n], 16, 1, None);
            roundtrip(&vec![7u8; n], 16, 16, Some(0));
        }
    }

    #[test]
    fn refuses_what_does_not_close() {
        let mut r = rng(4);
        let syms: Vec<u8> = (0..5000).map(|_| (r() % 6) as u8).collect();
        let mut hist = vec![0u32; 8];
        for &s in &syms {
            hist[s as usize] += 1;
        }
        let (t, _, _) = plan(&hist, 1, 8, 12);
        let mut enc = Vec::new();
        encode(&syms, None, &t, &mut enc);
        let d = DecTables::new(&t);
        let mut out = vec![0u8; syms.len()];
        assert!(decode(&enc, syms.len(), &d, &NoCtx, &mut out).is_ok());
        for cut in [1usize, 2, 17, enc.len() / 2] {
            assert!(decode(&enc[..enc.len() - cut], syms.len(), &d, &NoCtx, &mut out).is_err());
        }
        let mut longer = enc.clone();
        longer.extend_from_slice(&[0, 0]);
        assert!(decode(&longer, syms.len(), &d, &NoCtx, &mut out).is_err());
        // flipped bits: an error or other symbols, never a panic
        for k in 0..enc.len() {
            let mut bad = enc.clone();
            bad[k] ^= 0x10;
            let _ = decode(&bad, syms.len(), &d, &NoCtx, &mut out);
        }
    }

    fn cpu_ns() -> u64 {
        let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
        ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
    }

    #[test]
    #[ignore]
    fn speed() {
        let mut r = rng(9);
        let n = 1 << 24;
        // an exponent-like distribution, about 2.55 bits a symbol
        let syms: Vec<u8> = (0..n).map(|_| { let a = (r() % 64) as u32; (118 + (a.leading_zeros() - 26).min(7) + (r() % 3 == 0) as u32) as u8 }).collect();
        let mut hist = vec![0u32; 256];
        for &s in &syms {
            hist[s as usize] += 1;
        }
        let (t, _, _) = plan(&hist, 1, 256, 12);
        let mut enc = Vec::new();
        let t0 = cpu_ns();
        encode(&syms, None, &t, &mut enc);
        let te = cpu_ns() - t0;
        let d = DecTables::new(&t);
        let mut out = vec![0u8; n];
        let t0 = cpu_ns();
        for _ in 0..4 {
            decode(&enc, n, &d, &NoCtx, &mut out).unwrap();
        }
        let td = (cpu_ns() - t0) / 4;
        assert_eq!(out, syms);
        println!("no ctx: {} -> {} bytes ({:.3} bits), encode {:.2} ns/sym, decode {:.2} ns/sym", n, enc.len(), enc.len() as f64 * 8.0 / n as f64, te as f64 / n as f64, td as f64 / n as f64);
        // 64 contexts from the symbol before
        let ctx = lane_ctx(&syms, 2);
        let mut h2 = vec![0u32; 64 * 256];
        for (i, &s) in syms.iter().enumerate() {
            h2[(ctx[i] as usize).min(63) * 256 + s as usize] += 1;
        }
        let (t, _, _) = plan(&h2, 64, 256, 12);
        let mut enc = Vec::new();
        let t0 = cpu_ns();
        encode(&syms, Some(&ctx), &t, &mut enc);
        let te = cpu_ns() - t0;
        let d = DecTables::new(&t);
        let t0 = cpu_ns();
        for _ in 0..4 {
            decode(&enc, n, &d, &PrevCtx(2), &mut out).unwrap();
        }
        let td = (cpu_ns() - t0) / 4;
        assert_eq!(out, syms);
        println!("64 ctx: {} bytes, encode {:.2} ns/sym, decode {:.2} ns/sym", enc.len(), te as f64 / n as f64, td as f64 / n as f64);
        // 448 contexts, as a bf16 mantissa's: the symbol before (halved) and a class from another array
        struct Two<'a>(&'a [u8]);
        impl Ctx for Two<'_> {
            #[inline(always)]
            fn at(&self, i: usize, prev: u8) -> usize {
                ((prev & 0x7E) as usize) << 2 | self.0[i] as usize
            }
            #[inline(always)]
            fn next(&self, _: usize, sym: u8, _: u8) -> u8 {
                sym
            }
        }
        let cls: Vec<u8> = (0..n).map(|_| (r() % 7) as u8).collect();
        let sy7: Vec<u8> = syms.iter().map(|&s| s & 127).collect();
        let seg = n / LANES;
        let ctx2: Vec<u16> = (0..n).map(|i| { let st = i == 0 || (i % seg == 0 && i / seg < LANES); ((if st { 0 } else { sy7[i - 1] >> 1 }) as usize * 8 + cls[i] as usize) as u16 }).collect();
        for bits in [12u8, 10] {
            let mut h3 = vec![0u32; 512 * 128];
            for (i, &s) in sy7.iter().enumerate() {
                h3[ctx2[i] as usize * 128 + s as usize] += 1;
            }
            let (t, _, _) = plan(&h3, 512, 128, bits);
            let mut enc = Vec::new();
            encode(&sy7, Some(&ctx2), &t, &mut enc);
            let d = DecTables::new(&t);
            let mut out7 = vec![0u8; n];
            let t0 = cpu_ns();
            for _ in 0..4 {
                decode(&enc, n, &d, &Two(&cls), &mut out7).unwrap();
            }
            let td = (cpu_ns() - t0) / 4;
            assert_eq!(out7, sy7);
            println!("448 ctx bits {}: {} tables, {} bytes, decode {:.2} ns/sym", bits, t.freq.len(), enc.len(), td as f64 / n as f64);
        }
    }

    #[test]
    #[ignore]
    fn speed_by_tables() {
        // the same distribution under `ntab` identical tables, picked by the context
        let mut r = rng(11);
        let n = 1 << 24;
        let syms: Vec<u8> = (0..n).map(|_| { let a = (r() % 64) as u32; (118 + (a.leading_zeros() - 26).min(7) + (r() % 3 == 0) as u32) as u8 }).collect();
        let mut hist = vec![0u32; 256];
        for &s in &syms {
            hist[s as usize] += 1;
        }
        let cls0: Vec<u8> = (0..n + 1024).map(|_| (r() % 64) as u8).collect();
        for off in [0usize, 64, 1000] {
        let cls = &cls0[off..off + n];
        println!("offset {}", off);
        for bits in [12u8] {
            let (t1, _, _) = plan(&hist, 1, 256, bits);
            for ntab in [1usize, 4, 16, 64, 256] {
                let t = Tables { bits, alphabet: 256, map: (0..256).map(|c| (c % ntab) as u16).collect(), freq: vec![t1.freq[0].clone(); ntab] };
                let ctx: Vec<u16> = cls.iter().map(|&c| c as u16 * 4 % 256).collect();
                struct Cl<'a>(&'a [u8]);
                impl Ctx for Cl<'_> {
                    #[inline(always)]
                    fn at(&self, i: usize, _: u8) -> usize {
                        (self.0[i] as usize) * 4 % 256
                    }
                    #[inline(always)]
                    fn next(&self, _: usize, _: u8, _: u8) -> u8 {
                        0
                    }
                }
                let mut enc = Vec::new();
                encode(&syms, Some(&ctx), &t, &mut enc);
                let d = DecTables::new(&t);
                let mut out = vec![0u8; n];
                let t0 = cpu_ns();
                for _ in 0..3 {
                    decode(&enc, n, &d, &Cl(cls), &mut out).unwrap();
                }
                let td = (cpu_ns() - t0) / 3;
                assert_eq!(out, syms);
                println!("bits {} tables {:3}: decode {:.2} ns/sym", bits, ntab, td as f64 / n as f64);
            }
        }
        }
    }

    #[test]
    fn tables_survive_garbage() {
        let mut r = rng(5);
        for _ in 0..2000 {
            let len = (r() % 40) as usize;
            let b: Vec<u8> = (0..len).map(|_| r() as u8).collect();
            let mut p = 0;
            let _ = read_tables(&b, &mut p, 8, 4);
        }
    }
}
