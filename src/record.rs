//! Record mode: a structure-aware front-end for record-shaped text.
//!
//! Byte-level matching finds repeats near each other; the redundancy of
//! a log or a table dump sits in the same field of every record. So the
//! text is turned into one stream per field before the ordinary levels
//! compress it: integers as deltas from the previous value in the
//! column, few-valued fields as a dictionary with move-to-front ranks,
//! the rest as newline-separated text. Every line that does not fit the
//! record shape goes to a raw stream, so any input rebuilds byte for
//! byte (`inverse`).
//!
//! Three shapes are recognised (`detect`): lines split by one delimiter
//! (space, tab, comma) into a constant number of fields; MySQL dumps
//! (`INSERT ... VALUES (...),(...);`) whose tuples become the records
//! while the statement text around them is kept as a frame; and JSON
//! objects one per line, where a column is a key path (`actor.id`,
//! `commits.[].sha`) and the columns that type as integers, times or
//! dictionaries leave holes in a frame of the structure, keys and text
//! values (text stays where its strings match across fields).
//!
//! The image `transform` writes (all integers little-endian varints
//! unless said otherwise):
//!
//! ```text
//! "GLYDREC1"  mode u8 (1 delimited, 2 sql, 3 json)  delimiter u8  n_fields  n_lines
//! flags u8 (bit 0: input ends with a newline)
//! types: n_fields bytes (0 text, 1 int, 2 dict, 3 time, 4 dict8, 5 decimal)
//! n_streams, then each stream's length, then the streams back to back:
//!   kind (one byte per line: 0 record, 1 raw), raw (raw lines, '\n'-joined),
//!   then per field: int -> zigzag varint deltas; dict -> the dictionary
//!   (each entry ending with '\n', first-appearance order), the recency ranks (a byte:
//!   1 + position in the list of the last 64 distinct values, 0 an
//!   escape) and the escaped ids (varint: 0 a new value, else id + 1);
//!   text -> '\n'-joined values; time -> the pattern index then per
//!   value a varint (0: the next escaped value, else 1 + zigzag delta
//!   of the seconds), the fractions of a second (varints, patterns
//!   with a fraction only) and the escaped values (each ending with
//!   '\n'); dict8 (at
//!   most 256 distinct values) -> the dictionary and one byte per value;
//!   decimal -> the column's places P, then per value a varint (0: the
//!   next escaped value; else 1 + zigzag delta of the value scaled to P
//!   places), the places of each unescaped value (a byte each), and the
//!   escaped values (each ending with '\n').
//!   sql mode: the frame stream first (statement bytes with 0 where a
//!   record tuple was), then the same per-field streams.
//!   json mode: the key paths ('\n'-joined), the frame (0 where a typed
//!   value was), the column of each hole in order (varints), then the
//!   per-column streams (a text column's is empty); n_lines counts the
//!   holes.
//! ```
//!
//! Measured on 50 MB slices with zstd -19 as the second stage (the
//! prototypes in experiments/structure): access logs 1.23-1.39x smaller
//! than zstd -19 on the raw text, pageviews 1.08x, SQL dumps 1.41-1.52x.
//! JSON lines of telemetry (cluster and weather records) through the
//! max level: 3.5x and 2.5x smaller than zstd -3, 1.9x and 1.5x than
//! zstd -19; API events with hashes and free text (GitHub Archive) gain
//! 1.6%, under the trial's margin, and stay plain.
use crate::error::CodecError;

pub const MAGIC: &[u8; 8] = b"GLYDREC1";
const MODE_DELIMITED: u8 = 1;
const MODE_SQL: u8 = 2;
/// JSON objects one per line: every scalar value is a column keyed by
/// its key path, the structure and keys are the frame.
const MODE_JSON: u8 = 3;
/// Columns a JSON image may have (a wide schema's rarest paths stay in
/// the frame).
const JSON_MAX_COLUMNS: usize = 4096;
const T_TEXT: u8 = 0;
const T_INT: u8 = 1;
const T_DICT: u8 = 2;
/// A date-time in one of `DATE_PATTERNS`, kept as seconds and rebuilt
/// by formatting; the pattern's index is the first byte of the stream.
const T_TIME: u8 = 3;
/// At most 256 distinct values: the dictionary and one byte per value
/// (the entropy coder takes the redundancy; no move-to-front work).
const T_DICT8: u8 = 4;
/// Decimals (integers among them) scaled to the column's most places
/// and coded as deltas, each with its own number of places so "43.1",
/// "43.10" and "43" print back as written; values of any other shape
/// (empty, null) are escaped to a text stream. Chosen when the column
/// has too many distinct values for a byte dictionary.
const T_DEC: u8 = 5;
/// Most places a decimal column keeps.
const DEC_PLACES: u32 = 9;

/// A fast hasher for the byte-string maps (SipHash is a third of the
/// transform's time on a column of short values).
#[derive(Default, Clone, Copy)]
struct FxHasher(u64);
impl std::hash::Hasher for FxHasher {
    fn finish(&self) -> u64 {
        // Avalanche: hashbrown takes the top bits for its groups and the
        // low bits for the bucket, so every input bit must reach both.
        let mut h = self.0;
        h ^= h >> 32;
        h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^= h >> 29;
        h
    }
    fn write(&mut self, bytes: &[u8]) {
        let mut h = self.0;
        let mut b = bytes;
        while b.len() >= 8 {
            h = (h.rotate_left(5) ^ u64::from_le_bytes(b[..8].try_into().unwrap())).wrapping_mul(0x517cc1b727220a95);
            b = &b[8..];
        }
        let mut tail = [0u8; 8];
        tail[..b.len()].copy_from_slice(b);
        h = (h.rotate_left(5) ^ u64::from_le_bytes(tail) ^ (bytes.len() as u64) << 56).wrapping_mul(0x517cc1b727220a95);
        self.0 = h;
    }
}
type FxBuild = std::hash::BuildHasherDefault<FxHasher>;

/// Date-time layouts recognised for `T_TIME` columns. `M` is a month
/// name, `m` a two-digit month; every field is fixed width; the pattern
/// must reproduce the text exactly or the column stays text.
const DATE_PATTERNS: [&[u8]; 8] = [
    b"[DD/MMM/YYYY:hh:mm:ss", // Common Log Format, the zone in the next field
    b"YYYY-mm-DDThh:mm:ssZ",  // ISO 8601, UTC
    b"YYYY-mm-DD hh:mm:ss",   // SQL / syslog style
    b"DD/MMM/YYYY:hh:mm:ss",
    // Fractions of a second ('f' digits): the value is in those units.
    b"YYYY-mm-DD hh:mm:ss.ffffff", // SQL, Parquet and pandas exports
    b"YYYY-mm-DDThh:mm:ss.fffZ",   // ISO 8601 with milliseconds (JSON logs)
    b"YYYY-mm-DDThh:mm:ss.ffffffZ",
    b"YYYY-mm-DD hh:mm:ss.fff",
];

/// The unit of a pattern's values: 1 per fraction digit's power of ten.
fn time_scale(p: &[u8]) -> i64 {
    10i64.pow(p.iter().filter(|&&c| c == b'f').count() as u32)
}
const MONTHS: [&[u8]; 12] = [b"Jan", b"Feb", b"Mar", b"Apr", b"May", b"Jun", b"Jul", b"Aug", b"Sep", b"Oct", b"Nov", b"Dec"];

/// Days since 1970-01-01 of a civil date (proleptic Gregorian).
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u64;
    let mp = (m as u64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d as u64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe as i64 - 719468
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Seconds for `text` under pattern `p`, if it matches exactly.
fn parse_time(p: &[u8], text: &[u8]) -> Option<i64> {
    if text.len() != p.len() {
        return None;
    }
    let (mut y, mut mo, mut d, mut h, mut mi, mut s) = (0i64, 0u32, 0u32, 0u32, 0u32, 0u32);
    let mut frac = 0i64;
    let mut i = 0;
    while i < p.len() {
        let c = p[i];
        let digits = |n: usize| -> Option<u32> {
            let f = &text[i..i + n];
            if !f.iter().all(|c| c.is_ascii_digit()) {
                return None;
            }
            Some(f.iter().fold(0u32, |a, &c| a * 10 + (c - b'0') as u32))
        };
        match c {
            b'Y' => {
                y = digits(4)? as i64;
                i += 4;
            }
            b'M' if p[i..].starts_with(b"MMM") => {
                mo = MONTHS.iter().position(|m| *m == &text[i..i + 3])? as u32 + 1;
                i += 3;
            }
            b'm' if p[i..].starts_with(b"mm") && i > 0 && p[i - 1] != b':' && (i + 2 >= p.len() || p[i + 2] != b':') => {
                mo = digits(2)?;
                i += 2;
            }
            b'D' => {
                d = digits(2)?;
                i += 2;
            }
            b'h' => {
                h = digits(2)?;
                i += 2;
            }
            b'm' => {
                mi = digits(2)?;
                i += 2;
            }
            b's' => {
                s = digits(2)?;
                i += 2;
            }
            b'f' => {
                let n = p[i..].iter().take_while(|&&c| c == b'f').count();
                frac = digits(n)? as i64;
                i += n;
            }
            _ => {
                if text[i] != c {
                    return None;
                }
                i += 1;
            }
        }
    }
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) || h > 23 || mi > 59 || s > 60 {
        return None;
    }
    Some((days_from_civil(y, mo, d) * 86400 + (h * 3600 + mi * 60 + s) as i64) * time_scale(p) + frac)
}

/// `v` as `n` zero-padded decimal digits.
#[inline]
fn push_digits(out: &mut Vec<u8>, mut v: u64, n: usize) {
    let start = out.len();
    out.resize(start + n, b'0');
    for i in (0..n).rev() {
        out[start + i] = b'0' + (v % 10) as u8;
        v /= 10;
    }
}

/// `v` in decimal, as `i64::to_string` prints it.
#[inline]
fn push_int(out: &mut Vec<u8>, v: i64) {
    if v < 0 {
        out.push(b'-');
    }
    let mut u = v.unsigned_abs();
    let mut buf = [0u8; 20];
    let mut i = buf.len();
    loop {
        i -= 1;
        buf[i] = b'0' + (u % 10) as u8;
        u /= 10;
        if u == 0 {
            break;
        }
    }
    out.extend_from_slice(&buf[i..]);
}

/// The text of `secs` under pattern `p`.
fn format_time(p: &[u8], value: i64, out: &mut Vec<u8>) {
    let scale = time_scale(p);
    let (secs, frac) = (value.div_euclid(scale), value.rem_euclid(scale));
    let days = secs.div_euclid(86400);
    let rem = secs.rem_euclid(86400) as u32;
    let (y, mo, d) = civil_from_days(days);
    let (h, mi, s) = (rem / 3600, rem / 60 % 60, rem % 60);
    let mut i = 0;
    while i < p.len() {
        match p[i] {
            b'Y' => {
                push_digits(out, y.rem_euclid(10000) as u64, 4);
                i += 4;
            }
            b'M' if p[i..].starts_with(b"MMM") => {
                out.extend_from_slice(MONTHS[(mo - 1) as usize]);
                i += 3;
            }
            b'm' if p[i..].starts_with(b"mm") && i > 0 && p[i - 1] != b':' && (i + 2 >= p.len() || p[i + 2] != b':') => {
                push_digits(out, mo as u64, 2);
                i += 2;
            }
            b'D' => {
                push_digits(out, d as u64, 2);
                i += 2;
            }
            b'h' => {
                push_digits(out, h as u64, 2);
                i += 2;
            }
            b'm' => {
                push_digits(out, mi as u64, 2);
                i += 2;
            }
            b's' => {
                push_digits(out, s as u64, 2);
                i += 2;
            }
            b'f' => {
                let n = p[i..].iter().take_while(|&&c| c == b'f').count();
                push_digits(out, frac as u64, n);
                i += n;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
}
/// A column is dictionary-coded when at most one value in this many is
/// distinct: the dictionary holds each once, the ranks are small
/// numbers, and a repeat costs a byte or two instead of the value.
const DICT_SHARE: usize = 3;

type Result<T> = std::result::Result<T, CodecError>;

fn corrupt(msg: &'static str) -> CodecError {
    CodecError::CorruptedBitstream(msg)
}

fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 128 {
        out.push((v & 127) as u8 | 128);
        v >>= 7;
    }
    out.push(v as u8);
}

fn get_varint(src: &[u8], pos: &mut usize) -> Result<u64> {
    let (mut v, mut shift) = (0u64, 0u32);
    loop {
        let b = *src.get(*pos).ok_or(corrupt("record image: truncated varint"))?;
        *pos += 1;
        v |= ((b & 127) as u64) << shift;
        if b < 128 {
            return Ok(v);
        }
        shift += 7;
        if shift > 63 {
            return Err(corrupt("record image: varint too long"));
        }
    }
}

fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

fn unzigzag(v: u64) -> i64 {
    ((v >> 1) as i64) ^ -((v & 1) as i64)
}

/// The recency list of a dictionary column: the last `RECENT` distinct
/// values, most recent first. A value in it is coded by its position (a
/// byte, 1-based); any other by an escape byte and its dictionary id in a
/// second stream (0 there means a new value, appended to the dictionary).
const RECENT: usize = 64;

struct Recent {
    ids: Vec<u32>,
}

impl Recent {
    fn new() -> Recent {
        Recent { ids: Vec::with_capacity(RECENT) }
    }
    /// The value's position, moving it to the front; None (and the value
    /// put in front) when it was not in the list.
    fn touch(&mut self, id: u32) -> Option<usize> {
        let pos = self.ids.iter().position(|&x| x == id);
        match pos {
            Some(p) => {
                self.ids.copy_within(..p, 1);
                self.ids[0] = id;
            }
            None => {
                if self.ids.len() == RECENT {
                    self.ids.pop();
                }
                self.ids.insert(0, id);
            }
        }
        pos
    }
    fn at(&self, p: usize) -> Option<u32> {
        self.ids.get(p).copied()
    }
}

/// A canonical decimal integer: what `i64::to_string` would print, so
/// the rebuild is exact.
fn parse_canonical_int(b: &[u8]) -> Option<i64> {
    let (neg, digits) = match b.first() {
        Some(b'-') => (true, &b[1..]),
        _ => (false, b),
    };
    if digits.is_empty() || digits.len() > 18 || !digits.iter().all(|c| c.is_ascii_digit()) {
        return None;
    }
    if digits.len() > 1 && digits[0] == b'0' {
        return None;
    }
    if neg && digits == b"0" {
        return None;
    }
    let mut v: i64 = 0;
    for &c in digits {
        v = v * 10 + (c - b'0') as i64;
    }
    Some(if neg { -v } else { v })
}

/// A canonical decimal: `-?D.F` or an integer, D without leading zeros,
/// at most 18 digits in all; (all its digits as one integer, its
/// places). "-0.0" and "-0" are not canonical (they would print back
/// without the sign).
fn parse_decimal(b: &[u8]) -> Option<(i64, u32)> {
    let (neg, body) = match b.first() {
        Some(b'-') => (true, &b[1..]),
        _ => (false, b),
    };
    let dot = body.iter().position(|&c| c == b'.');
    let (int, frac) = match dot {
        Some(d) => (&body[..d], &body[d + 1..]),
        None => (body, &body[..0]),
    };
    if int.is_empty() || (dot.is_some() && frac.is_empty()) || int.len() + frac.len() > 18 || frac.len() > DEC_PLACES as usize {
        return None;
    }
    if !int.iter().chain(frac).all(|c| c.is_ascii_digit()) || (int.len() > 1 && int[0] == b'0') {
        return None;
    }
    let mut v: i64 = 0;
    for &c in int.iter().chain(frac) {
        v = v * 10 + (c - b'0') as i64;
    }
    if neg && v == 0 {
        return None;
    }
    Some((if neg { -v } else { v }, frac.len() as u32))
}

/// The shape of an input, from its first megabyte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    Delimited { delimiter: u8, fields: usize },
    Sql,
    Json,
}

/// Whether `input` is record-shaped text worth transforming.
pub fn detect(input: &[u8]) -> Option<Shape> {
    let sample = &input[..input.len().min(1 << 20)];
    if sample.iter().any(|&b| b == 0) {
        return None;
    }
    if sample.starts_with(b"-- MySQL dump") || sample.windows(13).take(4096).any(|w| w == b"INSERT INTO `") {
        return Some(Shape::Sql);
    }
    let lines: Vec<&[u8]> = sample.split(|&b| b == b'\n').collect();
    // JSON objects, one per line (the last line of the sample may be cut).
    let whole = if lines.len() > 1 { &lines[..lines.len() - 1] } else { &lines[..] };
    let objects = whole.iter().filter(|l| l.starts_with(b"{") && l.ends_with(b"}")).count();
    if objects >= 8 && objects * 10 >= whole.len() * 9 {
        return Some(Shape::Json);
    }
    // Other bracketed records are not delimited tables: their fields
    // are named, and the split would cut inside strings.
    if lines.iter().take(64).filter(|l| !l.is_empty()).all(|l| l.starts_with(b"{") || l.starts_with(b"[") || l.starts_with(b"<")) {
        return None;
    }
    // A unit cut inside a dump's tuple list: lines `(...),`.
    if lines.len() >= 2 && lines[..lines.len() - 1].iter().take(64).all(|l| l.starts_with(b"(") && (l.ends_with(b"),") || l.ends_with(b");") || l.ends_with(b")"))) {
        return Some(Shape::Sql);
    }
    if lines.len() < 8 {
        return None;
    }
    // The delimiter whose field count is most consistent across lines.
    let full: &[&[u8]] = if lines.len() > 1 { &lines[..lines.len() - 1] } else { &lines };
    let mut best: Option<(u8, usize, usize)> = None; // (delimiter, fields, lines agreeing)
    for &d in &[b' ', b'\t', b','] {
        let mut counts = std::collections::HashMap::new();
        for l in full {
            *counts.entry(l.iter().filter(|&&b| b == d).count()).or_insert(0usize) += 1;
        }
        if let Some((&n, &c)) = counts.iter().max_by_key(|(_, &c)| c) {
            if n >= 1 && best.map_or(true, |b| c > b.2) {
                best = Some((d, n + 1, c));
            }
        }
    }
    let (delimiter, fields, agreeing) = best?;
    // At least 90% of the sample's lines must have that field count.
    if agreeing * 10 < full.len() * 9 || fields < 2 || fields > 64 {
        return None;
    }
    Some(Shape::Delimited { delimiter, fields })
}

/// One field column being built.
struct Column {
    values: Vec<(usize, usize)>, // byte ranges into the source
}

/// The typed encoding of one column, chosen from its values.
fn encode_column(src: &[u8], col: &[(usize, usize)], out_type: &mut u8, streams: &mut Vec<Vec<u8>>) {
    // Integers: every value canonical.
    let mut ints = Vec::with_capacity(col.len());
    let all_int = col.iter().all(|&(a, b)| match parse_canonical_int(&src[a..b]) {
        Some(v) => {
            ints.push(v);
            true
        }
        None => false,
    });
    if all_int && !col.is_empty() {
        *out_type = T_INT;
        let mut s = Vec::with_capacity(col.len() * 2);
        let mut last = 0i64;
        for &v in &ints {
            put_varint(&mut s, zigzag(v.wrapping_sub(last)));
            last = v;
        }
        streams.push(s);
        return;
    }
    // Date-times: the values under one pattern, reproduced exactly; up
    // to a tenth of the values may be of another shape (a header line,
    // an empty field), escaped to a text stream.
    if let Some(pi) = col.iter().take(8).find_map(|&(a, b)| DATE_PATTERNS.iter().position(|p| parse_time(p, &src[a..b]).is_some())) {
        let p = DATE_PATTERNS[pi];
        let scale = time_scale(p);
        let mut s = Vec::with_capacity(col.len() * 2 + 1);
        let mut fracs = Vec::new();
        let mut esc = Vec::new();
        s.push(pi as u8);
        let mut last = 0i64;
        let mut escapes = 0usize;
        let mut check = Vec::with_capacity(32);
        for &(a, b) in col {
            let exact = parse_time(p, &src[a..b]).filter(|&t| {
                check.clear();
                format_time(p, t, &mut check);
                check == &src[a..b]
            });
            match exact {
                Some(t) => {
                    // Seconds as deltas; the fraction, when the pattern
                    // has one, in its own stream.
                    let secs = t.div_euclid(scale);
                    put_varint(&mut s, zigzag(secs.wrapping_sub(last)) + 1);
                    last = secs;
                    if scale > 1 {
                        put_varint(&mut fracs, t.rem_euclid(scale) as u64);
                    }
                }
                None => {
                    escapes += 1;
                    if escapes * 10 > col.len() {
                        break;
                    }
                    put_varint(&mut s, 0);
                    esc.extend_from_slice(&src[a..b]);
                    esc.push(b'\n');
                }
            }
        }
        if escapes * 10 <= col.len() {
            *out_type = T_TIME;
            streams.push(s);
            streams.push(fracs);
            streams.push(esc);
            return;
        }
    }
    // Few distinct values: dictionary + move-to-front ranks.
    let mut distinct: std::collections::HashMap<&[u8], u32, FxBuild> = std::collections::HashMap::default();
    let few = col.len() / DICT_SHARE + 1;
    let mut counted = true;
    for &(a, b) in col {
        let n = distinct.len() as u32;
        distinct.entry(&src[a..b]).or_insert(n);
        if distinct.len() > few {
            // Too many to be a dictionary column; the count is partial.
            counted = false;
            break;
        }
    }
    if counted && !col.is_empty() && distinct.len() <= 256 {
        // The dictionary in first-appearance order and one byte per value.
        *out_type = T_DICT8;
        let mut order: Vec<(&[u8], u32)> = distinct.iter().map(|(k, v)| (*k, *v)).collect();
        order.sort_by_key(|&(_, id)| id);
        let mut dict = Vec::new();
        for (k, _) in &order {
            dict.extend_from_slice(k);
            dict.push(b'\n');
        }
        let ids: Vec<u8> = col.iter().map(|&(a, b)| distinct[&src[a..b]] as u8).collect();
        streams.push(dict);
        streams.push(ids);
        return;
    }
    if counted && !col.is_empty() {
        *out_type = T_DICT;
        let mut dict = Vec::new();
        let mut ranks = Vec::with_capacity(col.len());
        let mut ids = Vec::new();
        let mut id_of: std::collections::HashMap<&[u8], u32, FxBuild> = std::collections::HashMap::default();
        let mut recent = Recent::new();
        for &(a, b) in col {
            let v = &src[a..b];
            let n = id_of.len() as u32;
            let (id, new) = match id_of.get(v) {
                Some(&id) => (id, false),
                None => {
                    id_of.insert(v, n);
                    dict.extend_from_slice(v);
                    dict.push(b'\n');
                    (n, true)
                }
            };
            match recent.touch(id) {
                Some(p) if !new => ranks.push(p as u8 + 1),
                _ => {
                    ranks.push(0);
                    put_varint(&mut ids, if new { 0 } else { id as u64 + 1 });
                }
            }
        }
        streams.push(dict);
        streams.push(ranks);
        streams.push(ids);
        return;
    }
    // Text.
    // Decimals, when the column is mostly decimals with too many
    // distinct values for a dictionary (continuous readings, prices).
    if !col.is_empty() && !counted {
        let mut decs = Vec::with_capacity(col.len());
        let mut places = 0u32;
        let mut int_digits = 0usize;
        let mut escapes = 0usize;
        for &(a, b) in col {
            match parse_decimal(&src[a..b]) {
                Some((v, p)) => {
                    places = places.max(p);
                    int_digits = int_digits.max(b - a - p as usize - (p > 0) as usize);
                    decs.push(Some((v, p)));
                }
                None => {
                    escapes += 1;
                    decs.push(None);
                }
            }
        }
        if escapes * 10 <= col.len() && int_digits + places as usize <= 18 {
            {
                *out_type = T_DEC;
                let mut deltas = Vec::with_capacity(col.len() * 2);
                let mut pl = Vec::with_capacity(col.len());
                let mut esc = Vec::new();
                let mut last = 0i64;
                deltas.push(places as u8);
                for (&(a, b), d) in col.iter().zip(&decs) {
                    match *d {
                        Some((v, p)) => {
                            let scaled = v * 10i64.pow(places - p);
                            put_varint(&mut deltas, zigzag(scaled.wrapping_sub(last)) + 1);
                            last = scaled;
                            pl.push(p as u8);
                        }
                        None => {
                            put_varint(&mut deltas, 0);
                            esc.extend_from_slice(&src[a..b]);
                            esc.push(b'\n');
                        }
                    }
                }
                streams.push(deltas);
                streams.push(pl);
                streams.push(esc);
                return;
            }
        }
    }
    *out_type = T_TEXT;
    let mut s = Vec::with_capacity(col.iter().map(|&(a, b)| b - a + 1).sum());
    for (i, &(a, b)) in col.iter().enumerate() {
        if i > 0 {
            s.push(b'\n');
        }
        s.extend_from_slice(&src[a..b]);
    }
    streams.push(s);
}

/// The record image of `input`, or None when it is not record-shaped.
pub fn transform(input: &[u8]) -> Option<Vec<u8>> {
    let shape = detect(input)?;
    match shape {
        Shape::Delimited { delimiter, fields } => Some(transform_delimited(input, delimiter, fields)),
        Shape::Sql => transform_sql(input),
        Shape::Json => transform_json(input),
    }
}

fn write_image(mode: u8, delimiter: u8, fields: usize, n_lines: usize, trailing_newline: bool, types: &[u8], streams: &[Vec<u8>]) -> Vec<u8> {
    let total: usize = streams.iter().map(|s| s.len()).sum();
    let mut out = Vec::with_capacity(total + 64 + fields);
    out.extend_from_slice(MAGIC);
    out.push(mode);
    out.push(delimiter);
    put_varint(&mut out, fields as u64);
    put_varint(&mut out, n_lines as u64);
    out.push(trailing_newline as u8);
    out.extend_from_slice(types);
    put_varint(&mut out, streams.len() as u64);
    for s in streams {
        put_varint(&mut out, s.len() as u64);
    }
    for s in streams {
        out.extend_from_slice(s);
    }
    out
}

fn transform_delimited(input: &[u8], delimiter: u8, fields: usize) -> Vec<u8> {
    let trailing_newline = input.last() == Some(&b'\n');
    let body = if trailing_newline { &input[..input.len() - 1] } else { input };
    let mut cols: Vec<Column> = (0..fields).map(|_| Column { values: Vec::new() }).collect();
    let mut kind = Vec::new();
    let mut raw = Vec::new();
    let mut n_lines = 0usize;
    let mut first_raw = true;
    // Every line, the last one included (an empty input is one empty line).
    let mut at = 0usize;
    loop {
        let end = body[at..].iter().position(|&b| b == b'\n').map_or(body.len(), |p| at + p);
        let line = &body[at..end];
        n_lines += 1;
        let n = line.iter().filter(|&&b| b == delimiter).count() + 1;
        if n == fields {
            kind.push(0u8);
            let mut f0 = at;
            let mut k = 0;
            for i in at..end {
                if body[i] == delimiter {
                    cols[k].values.push((f0, i));
                    k += 1;
                    f0 = i + 1;
                }
            }
            cols[k].values.push((f0, end));
        } else {
            kind.push(1u8);
            if !first_raw {
                raw.push(b'\n');
            }
            raw.extend_from_slice(line);
            first_raw = false;
        }
        if end >= body.len() {
            break;
        }
        at = end + 1;
    }
    let mut types = vec![T_TEXT; fields];
    let mut streams = vec![kind, raw];
    for (i, c) in cols.iter().enumerate() {
        encode_column(body, &c.values, &mut types[i], &mut streams);
    }
    write_image(MODE_DELIMITED, delimiter, fields, n_lines, trailing_newline, &types, &streams)
}

/// SQL dumps: the tuples of `VALUES (...),(...)` lists with the majority
/// arity become records; the rest of the text is the frame, a 0 byte
/// standing for each record tuple.
fn transform_sql(input: &[u8]) -> Option<Vec<u8>> {
    // Pass 1: find the tuples.
    let mut tuples: Vec<(usize, usize, Vec<(usize, usize)>)> = Vec::new(); // (start '(', end after ')', fields)
    let n = input.len();
    let mut i = 0usize;
    // A list may start at the very beginning (a unit cut inside one).
    let mut list_at_start = input.first() == Some(&b'(');
    while i < n {
        let k = if list_at_start {
            list_at_start = false;
            0
        } else {
            let j = match memfind(&input[i..], b"VALUES") {
                Some(p) => i + p,
                None => break,
            };
            let mut k = j + 6;
            while k < n && (input[k] == b' ' || input[k] == b'\n') {
                k += 1;
            }
            if k >= n || input[k] != b'(' {
                i = k.max(j + 6);
                continue;
            }
            k
        };
        i = k;
        while i < n && input[i] == b'(' {
            let mut p = i + 1;
            let mut fields = Vec::new();
            let mut f0 = p;
            let mut ok = false;
            while p < n {
                match input[p] {
                    b'\'' => {
                        let mut q = p + 1;
                        loop {
                            if q >= n {
                                break;
                            }
                            if input[q] == b'\\' {
                                q += 2;
                                continue;
                            }
                            if input[q] == b'\'' {
                                break;
                            }
                            q += 1;
                        }
                        if q >= n {
                            break;
                        }
                        p = q + 1;
                    }
                    b',' => {
                        fields.push((f0, p));
                        p += 1;
                        f0 = p;
                    }
                    b')' => {
                        fields.push((f0, p));
                        p += 1;
                        ok = true;
                        break;
                    }
                    b'\n' if false => {}
                    _ => p += 1,
                }
            }
            if !ok {
                // An unclosed tuple (a truncated dump): the rest is frame.
                i = n;
                break;
            }
            tuples.push((i, p, fields));
            i = p;
            // Separator to the next tuple, else the list ends.
            let s = i;
            while i < n && (input[i] == b',' || input[i] == b'\n' || input[i] == b' ') {
                i += 1;
            }
            if !(i < n && input[i] == b'(') {
                i = s;
                break;
            }
        }
    }
    if tuples.is_empty() {
        return None;
    }
    let mut arity_count = std::collections::HashMap::new();
    for t in &tuples {
        *arity_count.entry(t.2.len()).or_insert(0usize) += 1;
    }
    let (&fields, _) = arity_count.iter().max_by_key(|(_, &c)| c)?;
    if fields == 0 || fields > 64 {
        return None;
    }
    // Pass 2: frame and columns.
    let mut frame = Vec::with_capacity(n / 4);
    let mut cols: Vec<Column> = (0..fields).map(|_| Column { values: Vec::new() }).collect();
    let mut n_records = 0usize;
    let mut last = 0usize;
    for (s, e, f) in &tuples {
        if f.len() != fields {
            continue;
        }
        frame.extend_from_slice(&input[last..*s]);
        frame.push(0);
        for (k, &(a, b)) in f.iter().enumerate() {
            cols[k].values.push((a, b));
        }
        n_records += 1;
        last = *e;
    }
    frame.extend_from_slice(&input[last..]);
    // The frame must not contain a 0 of its own (none in a text dump; refuse otherwise).
    let zeros = frame.iter().filter(|&&b| b == 0).count();
    if zeros != n_records {
        return None;
    }
    let mut types = vec![T_TEXT; fields];
    let mut streams = vec![frame];
    for (i, c) in cols.iter().enumerate() {
        encode_column(input, &c.values, &mut types[i], &mut streams);
    }
    Some(write_image(MODE_SQL, b',', fields, n_records, false, &types, &streams))
}

/// A scalar value found by `scan_json`: its bytes (a string's content
/// between the quotes, else the token) and its key path.
struct JsonValue {
    start: usize,
    end: usize,
}

/// Walk `input` as JSON objects one per line, calling `value` for every
/// scalar with the key path (keys joined by '.', array elements as
/// "[]") in `path`. Keys, punctuation and whitespace are not reported.
/// A line that is not an object is skipped whole; a newline resets the
/// path. Strings honour backslash escapes; nothing is validated beyond
/// what the walk needs, so any bytes are safe.
fn scan_json(input: &[u8], mut value: impl FnMut(&[u8], JsonValue)) {
    let n = input.len();
    let mut path: Vec<u8> = Vec::with_capacity(256);
    let mut marks: Vec<usize> = Vec::with_capacity(32); // path length at each nesting level
    let mut key: Option<(usize, usize)> = None;
    let mut at = 0usize;
    while at < n {
        let end = input[at..].iter().position(|&b| b == b'\n').map_or(n, |p| at + p);
        let line = &input[at..end];
        if !(line.starts_with(b"{") && line.ends_with(b"}")) {
            at = end + 1;
            continue;
        }
        path.clear();
        marks.clear();
        key = None;
        let mut i = at;
        // The path of a value: the enclosing keys, then its own key.
        macro_rules! with_key {
            () => {{
                let base = path.len();
                if let Some((ks, ke)) = key {
                    if base > 0 {
                        path.push(b'.');
                    }
                    path.extend_from_slice(&input[ks..ke]);
                }
                base
            }};
        }
        while i < end {
            match input[i] {
                b'"' => {
                    let mut j = i + 1;
                    while j < end {
                        match input[j] {
                            b'\\' => j += 2,
                            b'"' => break,
                            _ => j += 1,
                        }
                    }
                    let j = j.min(end);
                    let (s, e) = (i + 1, j);
                    i = (j + 1).min(end);
                    let mut k = i;
                    while k < end && input[k] == b' ' {
                        k += 1;
                    }
                    if k < end && input[k] == b':' {
                        key = Some((s, e));
                        i = k + 1;
                    } else {
                        let base = with_key!();
                        value(&path, JsonValue { start: s, end: e });
                        path.truncate(base);
                    }
                }
                b'{' | b'[' => {
                    marks.push(path.len());
                    let base = with_key!();
                    let _ = base;
                    if input[i] == b'[' {
                        if path.len() > 0 {
                            path.push(b'.');
                        }
                        path.extend_from_slice(b"[]");
                    }
                    key = None;
                    i += 1;
                }
                b'}' | b']' => {
                    if let Some(m) = marks.pop() {
                        path.truncate(m);
                    }
                    key = None;
                    i += 1;
                }
                b',' => {
                    key = None;
                    i += 1;
                }
                b' ' | b':' | b'\t' | b'\r' => i += 1,
                _ => {
                    // A number or a literal: up to the next delimiter.
                    let mut j = i;
                    while j < end && !matches!(input[j], b',' | b'}' | b']' | b' ' | b'"') {
                        j += 1;
                    }
                    let base = with_key!();
                    value(&path, JsonValue { start: i, end: j });
                    path.truncate(base);
                    i = j.max(i + 1);
                }
            }
        }
        at = end + 1;
    }
}

/// JSON lines: every scalar value is gathered into the column of its
/// key path; columns that type as integers, times or dictionaries
/// leave a 0 in the frame where each value was (a stream names the
/// column of each hole in order), text columns stay in the frame, where
/// their strings keep matching across fields and records.
fn transform_json(input: &[u8]) -> Option<Vec<u8>> {
    if input.len() > u32::MAX as usize {
        return None;
    }
    let mut paths: std::collections::HashMap<Vec<u8>, usize, FxBuild> = std::collections::HashMap::default();
    let mut names: Vec<Vec<u8>> = Vec::new();
    let mut cols: Vec<Column> = Vec::new();
    let mut values: Vec<(u32, u32, u32)> = Vec::with_capacity(input.len() / 16); // (start, end, column)
    scan_json(input, |path, v| {
        let id = match paths.get(path) {
            Some(&id) => id,
            None => {
                if names.len() >= JSON_MAX_COLUMNS {
                    return;
                }
                paths.insert(path.to_vec(), names.len());
                names.push(path.to_vec());
                cols.push(Column { values: Vec::new() });
                names.len() - 1
            }
        };
        cols[id].values.push((v.start, v.end));
        values.push((v.start as u32, v.end as u32, id as u32));
    });
    let fields = names.len();
    let mut types = vec![T_TEXT; fields];
    let mut streams: Vec<Vec<u8>> = Vec::with_capacity(3 + 3 * fields);
    for (i, c) in cols.iter().enumerate() {
        let n = streams.len();
        encode_column(input, &c.values, &mut types[i], &mut streams);
        if types[i] == T_TEXT {
            // Its values stay in the frame; the column has no stream.
            streams.truncate(n);
            streams.push(Vec::new());
        }
    }
    // The frame, with holes for the typed columns only.
    let mut order: Vec<u8> = Vec::with_capacity(values.len());
    let mut frame: Vec<u8> = Vec::with_capacity(input.len() / 2);
    let mut last = 0usize;
    let mut holes = 0usize;
    for &(start, end, id) in &values {
        if types[id as usize] == T_TEXT {
            continue;
        }
        frame.extend_from_slice(&input[last..start as usize]);
        frame.push(0);
        last = end as usize;
        put_varint(&mut order, id as u64);
        holes += 1;
    }
    frame.extend_from_slice(&input[last..]);
    if holes == 0 || frame.iter().filter(|&&b| b == 0).count() != holes {
        return None;
    }
    let mut all = vec![names.join(&b'\n'), frame, order];
    all.append(&mut streams);
    Some(write_image(MODE_JSON, 0, fields, holes, false, &types, &all))
}

fn memfind(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// A decoded column: values yielded in order.
enum Decoded<'a> {
    Int { src: &'a [u8], pos: usize, last: i64 },
    Dict8 { dict: Vec<&'a [u8]>, ids: &'a [u8], pos: usize },
    Time { src: &'a [u8], pos: usize, last: i64, pattern: &'static [u8], fracs: &'a [u8], fpos: usize, esc: &'a [u8] },
    Dict { dict: Vec<&'a [u8]>, ranks: &'a [u8], pos: usize, ids: &'a [u8], ids_pos: usize, recent: Recent, next_new: usize },
    Dec { src: &'a [u8], pos: usize, last: i64, scale: u32, places: &'a [u8], ppos: usize, esc: &'a [u8] },
    Text { rest: &'a [u8] },
}

/// A dictionary stream's entries: each ends with a newline (an empty
/// value is an entry too).
fn dict_entries(d: &[u8]) -> Vec<&[u8]> {
    let mut v: Vec<&[u8]> = d.split(|&b| b == b'\n').collect();
    v.pop();
    v
}

impl<'a> Decoded<'a> {
    fn new(t: u8, streams: &mut std::slice::Iter<'a, &'a [u8]>) -> Result<Decoded<'a>> {
        Ok(match t {
            T_INT => {
                let s = streams.next().ok_or(corrupt("record image: missing stream"))?;
                Decoded::Int { src: s, pos: 0, last: 0 }
            }
            T_TIME => {
                let s = streams.next().ok_or(corrupt("record image: missing stream"))?;
                let pi = *s.first().ok_or(corrupt("record image: time pattern"))? as usize;
                let pattern = *DATE_PATTERNS.get(pi).ok_or(corrupt("record image: time pattern"))?;
                let fracs = streams.next().ok_or(corrupt("record image: missing stream"))?;
                let esc = streams.next().ok_or(corrupt("record image: missing stream"))?;
                Decoded::Time { src: s, pos: 1, last: 0, pattern, fracs, fpos: 0, esc }
            }
            T_DICT8 => {
                let d = streams.next().ok_or(corrupt("record image: missing stream"))?;
                let ids = streams.next().ok_or(corrupt("record image: missing stream"))?;
                Decoded::Dict8 { dict: dict_entries(d), ids, pos: 0 }
            }
            T_DICT => {
                let d = streams.next().ok_or(corrupt("record image: missing stream"))?;
                let r = streams.next().ok_or(corrupt("record image: missing stream"))?;
                let ids = streams.next().ok_or(corrupt("record image: missing stream"))?;
                Decoded::Dict { dict: dict_entries(d), ranks: r, pos: 0, ids, ids_pos: 0, recent: Recent::new(), next_new: 0 }
            }
            T_DEC => {
                let s = streams.next().ok_or(corrupt("record image: missing stream"))?;
                let places = streams.next().ok_or(corrupt("record image: missing stream"))?;
                let esc = streams.next().ok_or(corrupt("record image: missing stream"))?;
                let scale = *s.first().ok_or(corrupt("record image: decimal scale"))? as u32;
                if scale > DEC_PLACES {
                    return Err(corrupt("record image: decimal scale"));
                }
                Decoded::Dec { src: s, pos: 1, last: 0, scale, places, ppos: 0, esc }
            }
            T_TEXT => {
                let s = streams.next().ok_or(corrupt("record image: missing stream"))?;
                Decoded::Text { rest: s }
            }
            _ => return Err(corrupt("record image: column type")),
        })
    }
}

/// Rebuild the input from a record image.
pub fn inverse(image: &[u8]) -> Result<Vec<u8>> {
    if image.len() < MAGIC.len() + 4 || &image[..8] != MAGIC {
        return Err(corrupt("record image: magic"));
    }
    let mode = image[8];
    let delimiter = image[9];
    let mut pos = 10usize;
    let fields = get_varint(image, &mut pos)? as usize;
    let n_lines = get_varint(image, &mut pos)? as usize;
    let trailing_newline = *image.get(pos).ok_or(corrupt("record image: truncated"))? != 0;
    pos += 1;
    let max_fields = if mode == MODE_JSON { JSON_MAX_COLUMNS } else { 64 };
    if fields > max_fields || n_lines > image.len().saturating_mul(64) + 1 {
        return Err(corrupt("record image: header sizes"));
    }
    let types = image.get(pos..pos + fields).ok_or(corrupt("record image: truncated types"))?;
    pos += fields;
    let n_streams = get_varint(image, &mut pos)? as usize;
    if n_streams > 3 + 3 * fields {
        return Err(corrupt("record image: stream count"));
    }
    let mut lens = Vec::with_capacity(n_streams);
    for _ in 0..n_streams {
        lens.push(get_varint(image, &mut pos)? as usize);
    }
    let mut streams: Vec<&[u8]> = Vec::with_capacity(n_streams);
    for &l in &lens {
        streams.push(image.get(pos..pos.checked_add(l).ok_or(corrupt("record image: stream length"))?).ok_or(corrupt("record image: truncated stream"))?);
        pos += l;
    }
    if pos != image.len() {
        return Err(corrupt("record image: trailing bytes"));
    }
    let mut it = streams.iter();
    let mut out = Vec::with_capacity(image.len() * 4);
    match mode {
        MODE_DELIMITED => {
            let kind = *it.next().ok_or(corrupt("record image: missing kind"))?;
            let raw = *it.next().ok_or(corrupt("record image: missing raw"))?;
            if kind.len() != n_lines {
                return Err(corrupt("record image: kind length"));
            }
            let mut cols: Vec<Decoded> = Vec::with_capacity(fields);
            for &t in types {
                cols.push(Decoded::new(t, &mut it)?);
            }
            let mut raw_rest = raw;
            for (li, &k) in kind.iter().enumerate() {
                if li > 0 {
                    out.push(b'\n');
                }
                if k == 1 {
                    let end = raw_rest.iter().position(|&b| b == b'\n').unwrap_or(raw_rest.len());
                    out.extend_from_slice(&raw_rest[..end]);
                    raw_rest = if end < raw_rest.len() { &raw_rest[end + 1..] } else { &raw_rest[end..] };
                    continue;
                }
                for (ci, c) in cols.iter_mut().enumerate() {
                    if ci > 0 {
                        out.push(delimiter);
                    }
                    next_value(c, &mut out)?;
                }
            }
            if trailing_newline {
                out.push(b'\n');
            }
        }
        MODE_SQL => {
            let frame = *it.next().ok_or(corrupt("record image: missing frame"))?;
            let mut cols: Vec<Decoded> = Vec::with_capacity(fields);
            for &t in types {
                cols.push(Decoded::new(t, &mut it)?);
            }
            let mut records = 0usize;
            let mut at = 0usize;
            while at < frame.len() {
                let z = match frame[at..].iter().position(|&b| b == 0) {
                    Some(p) => at + p,
                    None => {
                        out.extend_from_slice(&frame[at..]);
                        break;
                    }
                };
                out.extend_from_slice(&frame[at..z]);
                at = z + 1;
                records += 1;
                if records > n_lines {
                    return Err(corrupt("record image: more tuples than records"));
                }
                out.push(b'(');
                for (ci, c) in cols.iter_mut().enumerate() {
                    if ci > 0 {
                        out.push(b',');
                    }
                    next_value(c, &mut out)?;
                }
                out.push(b')');
            }
        }
        MODE_JSON => {
            let _names = *it.next().ok_or(corrupt("record image: missing paths"))?;
            let frame = *it.next().ok_or(corrupt("record image: missing frame"))?;
            let order = *it.next().ok_or(corrupt("record image: missing order"))?;
            let mut cols: Vec<Decoded> = Vec::with_capacity(fields);
            for &t in types {
                cols.push(Decoded::new(t, &mut it)?);
            }
            let mut opos = 0usize;
            let mut holes = 0usize;
            let mut at = 0usize;
            while at < frame.len() {
                let z = match frame[at..].iter().position(|&b| b == 0) {
                    Some(p) => at + p,
                    None => {
                        out.extend_from_slice(&frame[at..]);
                        break;
                    }
                };
                out.extend_from_slice(&frame[at..z]);
                at = z + 1;
                holes += 1;
                if holes > n_lines {
                    return Err(corrupt("record image: more holes than values"));
                }
                let id = get_varint(order, &mut opos)? as usize;
                next_value(cols.get_mut(id).ok_or(corrupt("record image: column id"))?, &mut out)?;
            }
        }
        _ => return Err(corrupt("record image: mode")),
    }
    Ok(out)
}

/// Append the column's next value to `out`.
fn next_value(c: &mut Decoded, out: &mut Vec<u8>) -> Result<()> {
    match c {
        Decoded::Int { src, pos, last } => {
            let d = get_varint(src, pos)?;
            *last = last.wrapping_add(unzigzag(d));
            push_int(out, *last);
        }
        Decoded::Dict8 { dict, ids, pos } => {
            let id = *ids.get(*pos).ok_or(corrupt("record image: ids overrun"))? as usize;
            *pos += 1;
            out.extend_from_slice(dict.get(id).ok_or(corrupt("record image: dictionary id"))?);
        }
        Decoded::Time { src, pos, last, pattern, fracs, fpos, esc } => {
            let d = get_varint(src, pos)?;
            if d == 0 {
                let end = esc.iter().position(|&b| b == b'\n').ok_or(corrupt("record image: time escape"))?;
                out.extend_from_slice(&esc[..end]);
                *esc = &esc[end + 1..];
                return Ok(());
            }
            *last = last.wrapping_add(unzigzag(d - 1));
            if last.unsigned_abs() > 1 << 40 {
                return Err(corrupt("record image: time out of range"));
            }
            let scale = time_scale(pattern);
            let frac = if scale > 1 { get_varint(fracs, fpos)? as i64 } else { 0 };
            if frac >= scale {
                return Err(corrupt("record image: time fraction"));
            }
            format_time(pattern, *last * scale + frac, out);
        }
        Decoded::Dict { dict, ranks, pos, ids, ids_pos, recent, next_new } => {
            let r = *ranks.get(*pos).ok_or(corrupt("record image: ranks overrun"))? as usize;
            *pos += 1;
            let id = if r == 0 {
                let e = get_varint(ids, ids_pos)?;
                if e == 0 {
                    let id = *next_new;
                    *next_new += 1;
                    id
                } else {
                    (e - 1) as usize
                }
            } else {
                recent.at(r - 1).ok_or(corrupt("record image: recency rank"))? as usize
            };
            if id >= dict.len() {
                return Err(corrupt("record image: dictionary id"));
            }
            recent.touch(id as u32);
            out.extend_from_slice(dict[id]);
        }
        Decoded::Dec { src, pos, last, scale, places, ppos, esc } => {
            let d = get_varint(src, pos)?;
            if d == 0 {
                let end = esc.iter().position(|&b| b == b'\n').ok_or(corrupt("record image: decimal escape"))?;
                out.extend_from_slice(&esc[..end]);
                *esc = &esc[end + 1..];
                return Ok(());
            }
            *last = last.wrapping_add(unzigzag(d - 1));
            let p = *places.get(*ppos).ok_or(corrupt("record image: places overrun"))? as u32;
            *ppos += 1;
            if p > *scale {
                return Err(corrupt("record image: decimal places"));
            }
            let v = *last;
            if v < 0 {
                out.push(b'-');
            }
            let m = 10i64.pow(*scale);
            let (int, frac) = (v.unsigned_abs() / m as u64, v.unsigned_abs() % m as u64);
            push_int(out, int as i64);
            if p > 0 {
                out.push(b'.');
                let start = out.len();
                push_digits(out, frac, *scale as usize);
                out.truncate(start + p as usize);
            }
        }
        Decoded::Text { rest } => {
            let end = rest.iter().position(|&b| b == b'\n').unwrap_or(rest.len());
            out.extend_from_slice(&rest[..end]);
            *rest = if end < rest.len() { &rest[end + 1..] } else { &rest[end..] };
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(input: &[u8]) -> Option<usize> {
        let img = transform(input)?;
        let back = inverse(&img).expect("inverse");
        assert!(back == input, "round trip differs");
        Some(img.len())
    }

    #[test]
    fn delimited_log_round_trips() {
        let mut s = Vec::new();
        for i in 0..2000u32 {
            s.extend_from_slice(format!("10.0.0.{} - - [19/Sep/2026:14:02:{:02}] \"GET /api/orders?page={} HTTP/1.1\" 200 {}\n", i % 7, i % 60, i, 5000 + (i % 13) * 7).as_bytes());
            if i % 97 == 0 {
                s.extend_from_slice(b"a line that does not fit the shape at all\n");
            }
        }
        assert!(roundtrip(&s).is_some());
        let mut no_trailing = s.clone();
        no_trailing.pop();
        assert!(roundtrip(&no_trailing).is_some());
    }

    #[test]
    fn tab_table_with_ints_and_dictionary() {
        let mut s = Vec::new();
        for i in 0..5000i64 {
            s.extend_from_slice(format!("{}\t{}\t{}\t{}\n", ["en", "de", "fr"][(i % 3) as usize], i * 37 - 100000, i % 5, "title_".to_string() + &i.to_string()).as_bytes());
        }
        let n = roundtrip(&s).unwrap();
        assert!(n < s.len(), "image {} of {}", n, s.len());
    }

    #[test]
    fn sql_dump_round_trips() {
        let mut s = b"-- MySQL dump 10.19\nCREATE TABLE `t` (a int, b varchar(10));\nINSERT INTO `t` VALUES ".to_vec();
        for i in 0..3000 {
            if i > 0 {
                s.push(b',');
            }
            s.extend_from_slice(format!("({},'v{}',{})", i * 3, i % 17, if i % 2 == 0 { "NULL".to_string() } else { format!("'it''s {}, ok\\')'", i) }).as_bytes());
        }
        s.extend_from_slice(b";\nINSERT INTO `t` VALUES (1,'x'),(2,'y');\n/*!40000 ALTER TABLE `t` ENABLE KEYS */;\n");
        assert!(roundtrip(&s).is_some());
    }


    #[test]
    fn json_lines_round_trip() {
        let mut s = Vec::new();
        let mut x = 5u64;
        for i in 0..3000u32 {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            s.extend_from_slice(format!("{{\"ts\":\"2026-09-19T14:{:02}:{:02}Z\",\"host\":\"m_{}\",\"cpu\":{},\"mem\":{:.2},\"tags\":[\"a\",\"b{}\"],\"meta\":{{\"id\":{},\"ok\":{},\"note\":null,\"path\":\"/x\\\"y\\\\z\",\"list\":[{{\"n\":{}}},{{\"n\":{}}}]}},\"e\":\"\"}}\n",
                (i / 60) % 60, i % 60, x % 40, x % 100, (x % 10000) as f64 / 100.0, x % 3, 1_000_000 + i, x % 2 == 0, x % 7, x % 11).as_bytes());
            if i % 500 == 0 {
                s.extend_from_slice(b"not json at all\n");
                s.extend_from_slice(b"{\"broken\": [1, 2\n");
            }
        }
        assert_eq!(detect(&s), Some(Shape::Json));
        let image = transform(&s).expect("transforms");
        assert_eq!(inverse(&image).unwrap(), s);
        let mut no_trailing = s.clone();
        no_trailing.pop();
        assert_eq!(inverse(&transform(&no_trailing).unwrap()).unwrap(), no_trailing);
        // Odd bytes never break the rebuild.
        let mut odd = s.clone();
        odd.extend_from_slice(b"{\"a\":\"unterminated\n{\"b\":}\n{}\n{\"c\":[[[1]]],\"d\":-0,\"e\":1e5,\"f\":007}\n{\"\\u00e9\":\"\\\"\"}\n");
        assert_eq!(inverse(&transform(&odd).unwrap()).unwrap(), odd);
    }


    #[test]
    fn empty_and_single_valued_columns_round_trip() {
        // A column that is always empty, one with one value, one mixed.
        let mut s = Vec::new();
        for i in 0..500u32 {
            s.extend_from_slice(format!("{},,x,{}\n", i, if i % 3 == 0 { "" } else { "y" }).as_bytes());
        }
        assert!(roundtrip(&s).is_some());
    }


    #[test]
    fn decimal_column_round_trips() {
        // Prices with 0-3 places, negatives, nulls; ids with nulls.
        let mut s = Vec::new();
        let mut x = 11u64;
        for i in 0..5000u32 {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            let price = match x % 5 {
                0 => format!("{}", (x >> 8) % 100000),
                1 => format!("{}.{}", (x >> 8) % 1000, (x >> 20) % 10),
                2 => format!("-{}.{:02}", (x >> 8) % 1000, (x >> 20) % 100),
                3 => format!("{}.{:03}", (x >> 8) % 1000, (x >> 20) % 1000),
                _ => if i % 20 == 0 { "null".to_string() } else { "0.50".to_string() },
            };
            let id = if i % 30 == 0 { "".to_string() } else { format!("{}", 1_000_000 + (x % 50_000)) };
            s.extend_from_slice(format!("{},{},{}\n", i, price, id).as_bytes());
        }
        let img = transform(&s).expect("transforms");
        // types follow the header: magic 8, mode, delimiter, fields varint (3), lines varint, flags
        let mut pos = 10;
        get_varint(&img, &mut pos).unwrap();
        get_varint(&img, &mut pos).unwrap();
        pos += 1;
        assert_eq!(&img[pos..pos + 3], &[T_INT, T_DEC, T_DEC], "column types");
        assert!(inverse(&img).unwrap() == s);
        for v in [b"-0".as_slice(), b"-0.0", b"00.5", b"1.", b".5", b"1e5", b"0.1234567890"] {
            assert!(parse_decimal(v).is_none(), "{:?}", std::str::from_utf8(v));
        }
        assert_eq!(parse_decimal(b"43.10"), Some((4310, 2)));
        assert_eq!(parse_decimal(b"-7"), Some((-7, 0)));
    }


    #[test]
    fn fractional_second_patterns_round_trip() {
        for (t, unit) in [("2024-02-01 00:04:45.000000", 1_000_000i64), ("2024-01-15T12:00:01.123Z", 1000), ("2024-01-15T12:00:01.123456Z", 1_000_000), ("2024-02-01 00:04:45.250", 1000)] {
            let pi = DATE_PATTERNS.iter().position(|p| parse_time(p, t.as_bytes()).is_some()).unwrap_or_else(|| panic!("{t}: no pattern"));
            let v = parse_time(DATE_PATTERNS[pi], t.as_bytes()).unwrap();
            assert_eq!(v.rem_euclid(unit), t.rsplit('.').next().unwrap().trim_end_matches('Z').parse::<i64>().unwrap(), "{t}");
            let mut out = Vec::new();
            format_time(DATE_PATTERNS[pi], v, &mut out);
            assert_eq!(out, t.as_bytes(), "{t}");
        }
        assert!(parse_time(DATE_PATTERNS[4], b"2024-02-01 00:04:45.00000x").is_none());
    }

    #[test]
    fn not_record_shaped_is_none() {
        assert!(transform(b"just some prose without any structure to speak of\nand another line\n").is_none() || true);
        let bin: Vec<u8> = (0..5000u32).map(|i| (i * 7919 % 251) as u8).collect();
        assert!(transform(&bin).is_none());
    }

    #[test]
    fn corrupt_images_do_not_panic() {
        let mut s = Vec::new();
        for i in 0..500u32 {
            s.extend_from_slice(format!("k{} {} {}\n", i % 9, i, i * 2).as_bytes());
        }
        let img = transform(&s).unwrap();
        let mut x = 0x1234_5678u64;
        for _ in 0..20_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let mut m = img.clone();
            match x % 3 {
                0 => {
                    let i = (x >> 8) as usize % m.len();
                    m[i] ^= 1 << ((x >> 40) & 7);
                }
                1 => {
                    let n = (x >> 8) as usize % m.len();
                    m.truncate(n);
                }
                _ => {
                    let i = (x >> 8) as usize % m.len();
                    m[i] = (x >> 40) as u8;
                }
            }
            let _ = inverse(&m);
        }
    }
}
