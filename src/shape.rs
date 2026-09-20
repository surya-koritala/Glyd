//! Shape dictionaries: record mode's gains for small objects. A single
//! event or a small log file has nothing to learn a schema from and
//! would pay for its column dictionaries whole, so it gets the plain
//! path today. A `ShapeDict` is trained once on a sample of the data
//! and carries what a small object cannot: the shape (delimited fields,
//! JSON lines, or log templates), the frames lines take (a line's text
//! with a hole where each value was, each hole tied to a column), the
//! columns' types, the values dictionary columns usually hold (so an
//! object's values are ranks, not text), and a prepared LZ dictionary
//! (`Dict`) trained on such objects' images for the residual.
//!
//! An object is coded as a compact image — a varint of rows, then per
//! row its frame (or the raw line, for one that fits no frame), then
//! every column's values in row order, each column self-delimiting
//! given its count — and that image through `compress_with_dict`. The
//! reader needs the same dictionary, as with a zstd dictionary.
//!
//! Serialized (`to_bytes`): "GLYDSHP1", the kind, the frames, the
//! columns with their seeds, then the LZ dictionary.

use crate::error::Result;
use crate::record::{self, Recent, RECENT};
use crate::Dict;
use std::collections::HashMap;

const MAGIC: &[u8; 8] = b"GLYDSHP1";
/// Values of a column that stay in its dictionary seeds, most frequent first.
const MAX_SEEDS: usize = 4096;
/// Frames kept from training, most frequent first.
const MAX_FRAMES: usize = 4096;
const MAX_COLUMNS: usize = 4096;
/// The LZ dictionary's content.
const LZ_CONTENT: usize = 64 << 10;
/// Training cuts the sample into objects of about this size for the LZ
/// dictionary's samples.
const TRAIN_OBJECT: usize = 2048;

const K_DELIMITED: u8 = 1;
const K_JSON: u8 = 2;
const K_TEMPLATE: u8 = 3;

const C_TEXT: u8 = 0;
const C_INT: u8 = 1;
const C_TIME: u8 = 2;
const C_DEC: u8 = 3;
const C_DICT: u8 = 4;
/// A column with one value in the sample: an object lists only the
/// rows that differ.
const C_CONST: u8 = 5;
/// Distinct recent values an integer column codes by rank.
const INT_RING: usize = 16;

#[derive(Clone)]
enum Kind {
    Delimited { delimiter: u8, fields: usize },
    Json,
    Template,
}

struct Frame {
    /// The line with a 0 at each hole.
    text: Vec<u8>,
    /// The column of each hole.
    cols: Vec<u32>,
}

struct Column {
    kind: u8,
    /// The time pattern's index, or the decimal places.
    param: u8,
    /// Dictionary seeds, most frequent first; their ids are their index.
    seeds: Vec<Vec<u8>>,
    seed_ids: HashMap<Vec<u8>, u32>,
    /// The recency ring with the most frequent seeds in front.
    recent: Recent,
}

/// The shape side of a dictionary: what makes an object's image.
struct Model {
    kind: Kind,
    frames: Vec<Frame>,
    frame_ids: HashMap<Vec<u8>, u32>,
    columns: Vec<Column>,
}

pub struct ShapeDict {
    model: Model,
    lz: Dict,
}

/// A row of an object: a frame's holes filled by value spans, or a raw
/// line that fits no frame.
enum Row {
    Shaped(u32),
    Raw(usize, usize),
}

/// A line's frame and value spans under a kind: None when the line does
/// not take the kind's shape at all.
fn frame_of(kind: &Kind, line: &[u8], text: &mut Vec<u8>, spans: &mut Vec<(usize, usize)>, keys: &mut Vec<Vec<u8>>) -> bool {
    text.clear();
    spans.clear();
    keys.clear();
    match *kind {
        Kind::Delimited { delimiter, fields } => {
            let mut at = 0usize;
            loop {
                let end = line[at..].iter().position(|&b| b == delimiter).map_or(line.len(), |p| at + p);
                spans.push((at, end));
                if end == line.len() {
                    break;
                }
                at = end + 1;
            }
            if spans.len() != fields {
                return false;
            }
            for i in 0..fields {
                if i > 0 {
                    text.push(delimiter);
                }
                text.push(0);
            }
            true
        }
        Kind::Json => {
            if !(line.starts_with(b"{") && line.ends_with(b"}")) {
                return false;
            }
            record::scan_json(line, |path, v| {
                spans.push((v.start, v.end));
                keys.push(path.to_vec());
            });
            if spans.is_empty() {
                return false;
            }
            let mut last = 0usize;
            for &(a, b) in spans.iter() {
                text.extend_from_slice(&line[last..a]);
                text.push(0);
                last = b;
            }
            text.extend_from_slice(&line[last..]);
            true
        }
        Kind::Template => {
            // The line's skeleton: every run of letters and digits is a
            // hole (so "blk_-1608999687919862906" is two holes, an
            // address four), and the punctuation between them is the
            // frame. Words are cheap as dictionary values, and the
            // skeletons of a log are few.
            let mut i = 0usize;
            while i < line.len() {
                let c = line[i];
                if c.is_ascii_alphanumeric() {
                    let start = i;
                    while i < line.len() && line[i].is_ascii_alphanumeric() {
                        i += 1;
                    }
                    text.push(0);
                    spans.push((start, i));
                } else {
                    text.push(c);
                    i += 1;
                }
            }
            !spans.is_empty() && !text.contains(&b'\n')
        }
    }
}

/// The lines of `input` (the last one may lack its newline).
fn lines(input: &[u8]) -> (Vec<(usize, usize)>, bool) {
    let mut v = Vec::new();
    let mut at = 0usize;
    while at < input.len() {
        let end = input[at..].iter().position(|&b| b == b'\n').map_or(input.len(), |p| at + p);
        v.push((at, end));
        at = end + 1;
    }
    (v, input.last() == Some(&b'\n'))
}

impl ShapeDict {
    /// Train on `sample` (a few megabytes of the data the objects will
    /// come from): None when the sample is not record-shaped.
    pub fn train(sample: &[u8]) -> Option<ShapeDict> {
        let kind = match record::detect(sample)? {
            record::Shape::Delimited { delimiter, fields } => Kind::Delimited { delimiter, fields },
            record::Shape::Json => Kind::Json,
            record::Shape::Template => Kind::Template,
            record::Shape::Sql => return None,
        };
        // Frames and their holes' columns; the values of every column.
        let (line_spans, _) = lines(sample);
        let mut frame_count: HashMap<Vec<u8>, (u32, Vec<u32>)> = HashMap::new();
        let mut col_ids: HashMap<Vec<u8>, u32> = HashMap::new();
        let mut col_values: Vec<Vec<(usize, usize)>> = Vec::new();
        let (mut text, mut spans, mut keys) = (Vec::new(), Vec::new(), Vec::new());
        for &(a, b) in &line_spans {
            let line = &sample[a..b];
            if !frame_of(&kind, line, &mut text, &mut spans, &mut keys) {
                continue;
            }
            let entry = frame_count.entry(text.clone()).or_insert_with(|| (0, Vec::new()));
            if entry.0 == 0 {
                // A new frame: its holes' columns.
                let mut cols = Vec::with_capacity(spans.len());
                for (i, _) in spans.iter().enumerate() {
                    let key: Vec<u8> = match kind {
                        Kind::Delimited { .. } => i.to_string().into_bytes(),
                        Kind::Json => keys[i].clone(),
                        Kind::Template => {
                            let mut k = text.clone();
                            k.extend_from_slice(&i.to_string().into_bytes());
                            k
                        }
                    };
                    let n = col_ids.len() as u32;
                    let id = *col_ids.entry(key).or_insert(n);
                    if id as usize >= col_values.len() {
                        col_values.push(Vec::new());
                    }
                    cols.push(id);
                }
                if col_values.len() > MAX_COLUMNS {
                    continue;
                }
                entry.1 = cols;
            }
            entry.0 += 1;
            for (i, &(s, e)) in spans.iter().enumerate() {
                col_values[entry.1[i] as usize].push((a + s, a + e));
            }
        }
        if frame_count.is_empty() {
            return None;
        }
        // Frames by frequency, at most MAX_FRAMES.
        let mut by_freq: Vec<(&Vec<u8>, &(u32, Vec<u32>))> = frame_count.iter().collect();
        by_freq.sort_by(|x, y| y.1 .0.cmp(&x.1 .0).then(x.0.cmp(y.0)));
        by_freq.truncate(MAX_FRAMES);
        let mut frames = Vec::with_capacity(by_freq.len());
        let mut frame_ids = HashMap::new();
        for (text, (_, cols)) in by_freq {
            frame_ids.insert(text.clone(), frames.len() as u32);
            frames.push(Frame { text: text.clone(), cols: cols.clone() });
        }
        // Columns typed on the sample's values.
        let columns: Vec<Column> = col_values.iter().map(|vals| type_column(sample, vals)).collect();
        let model = Model { kind, frames, frame_ids, columns };
        // The LZ dictionary: trained on the images of sample objects.
        let mut images: Vec<Vec<u8>> = Vec::new();
        let mut at = 0usize;
        while at < sample.len() {
            let mut end = (at + TRAIN_OBJECT).min(sample.len());
            if end < sample.len() {
                end = sample[end..].iter().position(|&b| b == b'\n').map_or(sample.len(), |p| end + p + 1);
            }
            images.push(model.image(&sample[at..end]));
            at = end;
        }
        let refs: Vec<&[u8]> = images.iter().map(|v| v.as_slice()).collect();
        let lz = Dict::train(&refs, LZ_CONTENT);
        Some(ShapeDict { model, lz })
    }

    /// `input` compressed with this dictionary.
    pub fn compress(&self, input: &[u8], output: &mut Vec<u8>) {
        let image = self.model.image(input);
        crate::compress_with_dict(&self.lz, &image, output);
    }

    /// The object's image before the LZ stage (for measurement).
    pub fn image_bytes(&self, input: &[u8]) -> Vec<u8> {
        self.model.image(input)
    }

    /// The bytes each column takes in the object's image, for measurement.
    pub fn column_bytes(&self, input: &[u8]) -> Vec<usize> {
        let m = &self.model;
        let (line_spans, _) = lines(input);
        let (mut text, mut spans, mut keys) = (Vec::new(), Vec::new(), Vec::new());
        let mut per_col: Vec<Vec<(usize, usize)>> = vec![Vec::new(); m.columns.len()];
        for &(a, b) in &line_spans {
            if frame_of(&m.kind, &input[a..b], &mut text, &mut spans, &mut keys) {
                if let Some(&f) = m.frame_ids.get(&text) {
                    for (i, &(s, e)) in spans.iter().enumerate() {
                        per_col[m.frames[f as usize].cols[i] as usize].push((a + s, a + e));
                    }
                }
            }
        }
        m.columns.iter().zip(&per_col).map(|(c, vals)| { let mut out = Vec::new(); encode_values(input, c, vals, &mut out); out.len() }).collect()
    }

    /// One line per column: its kind and seed count, for measurement.
    pub fn describe(&self) -> String {
        let mut s = format!("{} frames; columns:", self.model.frames.len());
        for (i, c) in self.model.columns.iter().enumerate() {
            s += &format!(" {}:{}{}", i, ["text", "int", "time", "dec", "dict", "const"][c.kind as usize], if c.kind == C_DICT { format!("({})", c.seeds.len()) } else { String::new() });
        }
        s
    }

    /// The object back, from `compress`'s output.
    pub fn decompress(&self, compressed: &[u8]) -> Result<Vec<u8>> {
        let image = crate::decompress_with_dict(&self.lz, compressed)?;
        self.model.rebuild(&image)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        let m = &self.model;
        match m.kind {
            Kind::Delimited { delimiter, fields } => {
                out.push(K_DELIMITED);
                out.push(delimiter);
                record::put_varint(&mut out, fields as u64);
            }
            Kind::Json => out.push(K_JSON),
            Kind::Template => out.push(K_TEMPLATE),
        }
        record::put_varint(&mut out, m.frames.len() as u64);
        for f in &m.frames {
            record::put_varint(&mut out, f.text.len() as u64);
            out.extend_from_slice(&f.text);
            for &c in &f.cols {
                record::put_varint(&mut out, c as u64);
            }
        }
        record::put_varint(&mut out, m.columns.len() as u64);
        for c in &m.columns {
            out.push(c.kind);
            out.push(c.param);
            record::put_varint(&mut out, c.seeds.len() as u64);
            for s in &c.seeds {
                record::put_varint(&mut out, s.len() as u64);
                out.extend_from_slice(s);
            }
        }
        out.extend_from_slice(&self.lz.to_bytes());
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<ShapeDict> {
        if bytes.len() < 10 || &bytes[..8] != MAGIC {
            return None;
        }
        let mut pos = 8usize;
        let varint = |pos: &mut usize| record::get_varint(bytes, pos).ok();
        let kind = match bytes[pos] {
            K_DELIMITED => {
                let delimiter = *bytes.get(pos + 1)?;
                pos += 2;
                Kind::Delimited { delimiter, fields: varint(&mut pos)? as usize }
            }
            K_JSON => {
                pos += 1;
                Kind::Json
            }
            K_TEMPLATE => {
                pos += 1;
                Kind::Template
            }
            _ => return None,
        };
        let n = varint(&mut pos)? as usize;
        if n > MAX_FRAMES {
            return None;
        }
        let mut frames = Vec::with_capacity(n);
        let mut frame_ids = HashMap::new();
        for i in 0..n {
            let len = varint(&mut pos)? as usize;
            let text = bytes.get(pos..pos.checked_add(len)?)?.to_vec();
            pos += len;
            let holes = text.iter().filter(|&&b| b == 0).count();
            let mut cols = Vec::with_capacity(holes);
            for _ in 0..holes {
                cols.push(varint(&mut pos)? as u32);
            }
            frame_ids.insert(text.clone(), i as u32);
            frames.push(Frame { text, cols });
        }
        let n = varint(&mut pos)? as usize;
        if n > MAX_COLUMNS {
            return None;
        }
        let mut columns = Vec::with_capacity(n);
        for _ in 0..n {
            let kind = *bytes.get(pos)?;
            let param = *bytes.get(pos + 1)?;
            pos += 2;
            let k = varint(&mut pos)? as usize;
            if k > MAX_SEEDS || kind > C_CONST || (kind == C_CONST && k != 1) || (kind == C_TIME && param as usize >= record::DATE_PATTERNS.len()) || (kind == C_DEC && param > 18) {
                return None;
            }
            let mut seeds = Vec::with_capacity(k);
            for _ in 0..k {
                let len = varint(&mut pos)? as usize;
                seeds.push(bytes.get(pos..pos.checked_add(len)?)?.to_vec());
                pos += len;
            }
            columns.push(seeded(kind, param, seeds));
        }
        if frames.iter().any(|f| f.cols.iter().any(|&c| c as usize >= columns.len())) {
            return None;
        }
        let lz = Dict::from_bytes(bytes.get(pos..)?)?;
        Some(ShapeDict { model: Model { kind, frames, frame_ids, columns }, lz })
    }
}

impl Model {
    /// The object's compact image.
    fn image(&self, input: &[u8]) -> Vec<u8> {
        let (line_spans, trailing) = lines(input);
        let (mut text, mut spans, mut keys) = (Vec::new(), Vec::new(), Vec::new());
        let mut rows: Vec<Row> = Vec::with_capacity(line_spans.len());
        let mut per_col: Vec<Vec<(usize, usize)>> = vec![Vec::new(); self.columns.len()];
        for &(a, b) in &line_spans {
            let line = &input[a..b];
            let frame = if frame_of(&self.kind, line, &mut text, &mut spans, &mut keys) { self.frame_ids.get(&text).copied() } else { None };
            match frame {
                Some(f) => {
                    let cols = &self.frames[f as usize].cols;
                    for (i, &(s, e)) in spans.iter().enumerate() {
                        per_col[cols[i] as usize].push((a + s, a + e));
                    }
                    rows.push(Row::Shaped(f));
                }
                None => rows.push(Row::Raw(a, b)),
            }
        }
        let mut out = Vec::with_capacity(input.len() / 4 + 16);
        record::put_varint(&mut out, rows.len() as u64 * 2 + trailing as u64);
        for r in &rows {
            match *r {
                Row::Shaped(frame) => record::put_varint(&mut out, frame as u64 + 1),
                Row::Raw(a, b) => {
                    out.push(0);
                    record::put_varint(&mut out, (b - a) as u64);
                    out.extend_from_slice(&input[a..b]);
                }
            }
        }
        for (c, vals) in self.columns.iter().zip(&per_col) {
            encode_values(input, c, vals, &mut out);
        }
        out
    }

    fn rebuild(&self, image: &[u8]) -> Result<Vec<u8>> {
        let mut pos = 0usize;
        let head = record::get_varint(image, &mut pos)?;
        let (n_rows, trailing) = ((head / 2) as usize, head & 1 == 1);
        if n_rows > image.len() {
            return Err(record::corrupt("shape image: rows"));
        }
        let mut rows: Vec<(u32, usize, usize)> = Vec::with_capacity(n_rows); // frame + 1, or 0 with a raw span
        let mut counts = vec![0usize; self.columns.len()];
        for _ in 0..n_rows {
            let f = record::get_varint(image, &mut pos)?;
            if f == 0 {
                let len = record::get_varint(image, &mut pos)? as usize;
                let a = pos;
                pos = pos.checked_add(len).filter(|&p| p <= image.len()).ok_or(record::corrupt("shape image: raw line"))?;
                rows.push((0, a, pos));
            } else {
                let frame = self.frames.get(f as usize - 1).ok_or(record::corrupt("shape image: frame"))?;
                for &c in &frame.cols {
                    counts[c as usize] += 1;
                }
                rows.push((f as u32, 0, 0));
            }
        }
        let mut cols: Vec<Values> = Vec::with_capacity(self.columns.len());
        for (c, &n) in self.columns.iter().zip(&counts) {
            cols.push(decode_values(image, &mut pos, c, n)?);
        }
        if pos != image.len() {
            return Err(record::corrupt("shape image: trailing bytes"));
        }
        let mut next = vec![0usize; self.columns.len()];
        let mut out = Vec::with_capacity(image.len() * 4);
        for (i, &(f, a, b)) in rows.iter().enumerate() {
            if f == 0 {
                out.extend_from_slice(&image[a..b]);
            } else {
                let frame = &self.frames[f as usize - 1];
                let mut hole = 0usize;
                for &t in &frame.text {
                    if t == 0 {
                        let c = frame.cols[hole] as usize;
                        let v = cols[c].get(next[c]).ok_or(record::corrupt("shape image: values"))?;
                        out.extend_from_slice(v);
                        next[c] += 1;
                        hole += 1;
                    } else {
                        out.push(t);
                    }
                }
            }
            if i + 1 < n_rows || trailing {
                out.push(b'\n');
            }
        }
        Ok(out)
    }

}

/// A canonical integer of up to 19 digits (record mode stops at 18):
/// block and object ids are often random 63-bit numbers.
fn parse_int(b: &[u8]) -> Option<i64> {
    if b.len() <= 18 {
        return record::parse_canonical_int(b);
    }
    let (neg, digits) = match b.first() {
        Some(b'-') => (true, &b[1..]),
        _ => (false, b),
    };
    if digits.len() != 19 || !digits.iter().all(|c| c.is_ascii_digit()) || digits[0] == b'0' {
        return None;
    }
    let mut v: i64 = 0;
    for &c in digits {
        v = v.checked_mul(10)?.checked_add((c - b'0') as i64)?;
    }
    Some(if neg { -v } else { v })
}

/// A column with its seeds indexed and its recency ring filled.
fn seeded(kind: u8, param: u8, seeds: Vec<Vec<u8>>) -> Column {
    let seed_ids: HashMap<Vec<u8>, u32> = seeds.iter().enumerate().map(|(i, s)| (s.clone(), i as u32)).collect();
    let mut recent = Recent::new();
    for i in (0..seeds.len().min(RECENT)).rev() {
        recent.push_front(i as u32);
    }
    Column { kind, param, seeds, seed_ids, recent }
}

/// A column's type and seeds from the sample's values: the rules of
/// record mode's `encode_column`, with a tenth of the values allowed to
/// escape (the compact image escapes any value).
fn type_column(src: &[u8], vals: &[(usize, usize)]) -> Column {
    let n = vals.len().max(1);
    let ints = vals.iter().filter(|&&(a, b)| parse_int(&src[a..b]).is_some()).count();
    if ints * 10 >= n * 9 && !vals.is_empty() {
        return seeded(C_INT, 0, Vec::new());
    }
    if let Some(pi) = vals.iter().take(8).find_map(|&(a, b)| record::DATE_PATTERNS.iter().position(|p| record::parse_time(p, &src[a..b]).is_some())) {
        let p = record::DATE_PATTERNS[pi];
        let mut check = Vec::new();
        let exact = vals
            .iter()
            .filter(|&&(a, b)| {
                record::parse_time(p, &src[a..b]).map_or(false, |t| {
                    check.clear();
                    record::format_time(p, t, &mut check);
                    check == &src[a..b]
                })
            })
            .count();
        if exact * 10 >= n * 9 {
            return seeded(C_TIME, pi as u8, Vec::new());
        }
    }
    let mut counts: HashMap<&[u8], u32> = HashMap::new();
    for &(a, b) in vals {
        *counts.entry(&src[a..b]).or_insert(0) += 1;
    }
    if counts.len() == 1 {
        return seeded(C_CONST, 0, vec![counts.into_keys().next().unwrap().to_vec()]);
    }
    if counts.len() <= n / 3 + 1 && !vals.is_empty() {
        let mut order: Vec<(&[u8], u32)> = counts.into_iter().collect();
        order.sort_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(y.0)));
        order.truncate(MAX_SEEDS);
        return seeded(C_DICT, 0, order.into_iter().map(|(v, _)| v.to_vec()).collect());
    }
    let mut places = 0u32;
    let mut int_digits = 0usize;
    let decs = vals
        .iter()
        .filter(|&&(a, b)| match record::parse_decimal(&src[a..b]) {
            Some((_, p)) => {
                places = places.max(p);
                int_digits = int_digits.max(b - a - p as usize - (p > 0) as usize);
                true
            }
            None => false,
        })
        .count();
    if decs * 10 >= n * 9 && !vals.is_empty() && int_digits + places as usize <= 18 {
        return seeded(C_DEC, places as u8, Vec::new());
    }
    seeded(C_TEXT, 0, Vec::new())
}

/// A column's values into the image, self-delimiting given their count:
/// integers as deltas (a varint of zigzag + 1; 0 escapes the value to
/// a text list after them); times likewise, their fractions after;
/// decimals likewise with their places; dictionary values as recency
/// ranks (1..=64) or 0 with an id in a list after them (a seed's or a
/// new entry's, 0, whose text follows in a third list); text as
/// newline-terminated values.
fn encode_values(src: &[u8], c: &Column, vals: &[(usize, usize)], out: &mut Vec<u8>) {
    let mut esc: Vec<u8> = Vec::new();
    match c.kind {
        C_INT => {
            // A value among the last INT_RING distinct ones is its rank
            // (1..=INT_RING, front first); else INT_RING + 1 + the
            // zigzag delta from the previous value; 0 escapes to text.
            let mut last = 0i64;
            let mut ring: Vec<i64> = Vec::with_capacity(INT_RING);
            for &(a, b) in vals {
                match parse_int(&src[a..b]) {
                    Some(v) => {
                        match ring.iter().position(|&r| r == v) {
                            Some(p) => {
                                record::put_varint(out, p as u64 + 1);
                                ring.remove(p);
                            }
                            None => {
                                record::put_varint(out, record::zigzag(v.wrapping_sub(last)) + INT_RING as u64 + 1);
                                if ring.len() == INT_RING {
                                    ring.pop();
                                }
                            }
                        }
                        ring.insert(0, v);
                        last = v;
                    }
                    None => {
                        out.push(0);
                        esc.extend_from_slice(&src[a..b]);
                        esc.push(b'\n');
                    }
                }
            }
            out.extend_from_slice(&esc);
        }
        C_CONST => {
            // The rows whose value is not the seed: how many, then each
            // as its distance from the last such row and its text.
            let seed = &c.seeds[0];
            let mut k = 0u64;
            let mut last = 0usize;
            for (i, &(a, b)) in vals.iter().enumerate() {
                if &src[a..b] != seed.as_slice() {
                    k += 1;
                    record::put_varint(&mut esc, (i - last) as u64);
                    last = i;
                    esc.extend_from_slice(&src[a..b]);
                    esc.push(b'\n');
                }
            }
            if !vals.is_empty() {
                record::put_varint(out, k);
                out.extend_from_slice(&esc);
            }
        }
        C_TIME => {
            let p = record::DATE_PATTERNS[c.param as usize];
            let scale = record::time_scale(p);
            let mut fracs = Vec::new();
            let mut last = 0i64;
            let mut check = Vec::new();
            for &(a, b) in vals {
                let exact = record::parse_time(p, &src[a..b]).filter(|&t| {
                    check.clear();
                    record::format_time(p, t, &mut check);
                    check == &src[a..b]
                });
                match exact {
                    Some(t) => {
                        let secs = t.div_euclid(scale);
                        record::put_varint(out, record::zigzag(secs.wrapping_sub(last)) + 1);
                        last = secs;
                        if scale > 1 {
                            record::put_varint(&mut fracs, t.rem_euclid(scale) as u64);
                        }
                    }
                    None => {
                        out.push(0);
                        esc.extend_from_slice(&src[a..b]);
                        esc.push(b'\n');
                    }
                }
            }
            out.extend_from_slice(&fracs);
            out.extend_from_slice(&esc);
        }
        C_DEC => {
            let places = c.param as u32;
            let mut pl = Vec::new();
            let mut last = 0i64;
            for &(a, b) in vals {
                match record::parse_decimal(&src[a..b]).filter(|&(_, p)| p <= places) {
                    Some((v, p)) => {
                        let scaled = v * 10i64.pow(places - p);
                        record::put_varint(out, record::zigzag(scaled.wrapping_sub(last)) + 1);
                        last = scaled;
                        pl.push(p as u8);
                    }
                    None => {
                        out.push(0);
                        esc.extend_from_slice(&src[a..b]);
                        esc.push(b'\n');
                    }
                }
            }
            out.extend_from_slice(&pl);
            out.extend_from_slice(&esc);
        }
        C_DICT => {
            let mut recent = c.recent.clone();
            let mut ids = Vec::new();
            let mut new_ids: HashMap<&[u8], u32> = HashMap::new();
            let mut n_new = c.seeds.len() as u32;
            for &(a, b) in vals {
                let v = &src[a..b];
                let (id, new) = match c.seed_ids.get(v).or_else(|| new_ids.get(v)) {
                    Some(&id) => (id, false),
                    None => {
                        new_ids.insert(v, n_new);
                        n_new += 1;
                        esc.extend_from_slice(v);
                        esc.push(b'\n');
                        (n_new - 1, true)
                    }
                };
                match recent.touch(id) {
                    Some(p) if !new => out.push(p as u8 + 1),
                    _ => {
                        out.push(0);
                        record::put_varint(&mut ids, if new { 0 } else { id as u64 + 1 });
                    }
                }
            }
            out.extend_from_slice(&ids);
            out.extend_from_slice(&esc);
        }
        _ => {
            for &(a, b) in vals {
                out.extend_from_slice(&src[a..b]);
                out.push(b'\n');
            }
        }
    }
}

/// A decoded column: its values' bytes and ends.
struct Values {
    bytes: Vec<u8>,
    ends: Vec<usize>,
}

impl Values {
    fn get(&self, i: usize) -> Option<&[u8]> {
        let end = *self.ends.get(i)?;
        let start = if i == 0 { 0 } else { self.ends[i - 1] };
        Some(&self.bytes[start..end])
    }
    fn push(&mut self, v: &[u8]) {
        self.bytes.extend_from_slice(v);
        self.ends.push(self.bytes.len());
    }
}

/// A newline-terminated value at `pos`.
fn line_at<'a>(image: &'a [u8], pos: &mut usize) -> Result<&'a [u8]> {
    let end = image[*pos..].iter().position(|&b| b == b'\n').map(|p| *pos + p).ok_or(record::corrupt("shape image: value"))?;
    let v = &image[*pos..end];
    *pos = end + 1;
    Ok(v)
}

fn decode_values(image: &[u8], pos: &mut usize, c: &Column, n: usize) -> Result<Values> {
    let mut out = Values { bytes: Vec::new(), ends: Vec::with_capacity(n) };
    if n > image.len() + 1 {
        return Err(record::corrupt("shape image: count"));
    }
    match c.kind {
        C_INT => {
            let mut codes = Vec::with_capacity(n);
            for _ in 0..n {
                codes.push(record::get_varint(image, pos)?);
            }
            let mut last = 0i64;
            let mut ring: Vec<i64> = Vec::with_capacity(INT_RING);
            for code in codes {
                if code == 0 {
                    let v = line_at(image, pos)?;
                    out.push(v);
                } else {
                    let v = if code <= INT_RING as u64 {
                        let p = code as usize - 1;
                        if p >= ring.len() {
                            return Err(record::corrupt("shape image: int rank"));
                        }
                        ring.remove(p)
                    } else {
                        if ring.len() == INT_RING {
                            ring.pop();
                        }
                        last.wrapping_add(record::unzigzag(code - INT_RING as u64 - 1))
                    };
                    ring.insert(0, v);
                    last = v;
                    record::push_int(&mut out.bytes, v);
                    out.ends.push(out.bytes.len());
                }
            }
        }
        C_CONST => {
            if n == 0 {
                return Ok(out);
            }
            let k = record::get_varint(image, pos)? as usize;
            if k > n {
                return Err(record::corrupt("shape image: constants"));
            }
            let mut rows = Vec::with_capacity(k);
            let mut at = 0usize;
            for _ in 0..k {
                at = at.checked_add(record::get_varint(image, pos)? as usize).filter(|&r| r < n).ok_or(record::corrupt("shape image: constant row"))?;
                rows.push((at, line_at(image, pos)?));
            }
            let mut ri = 0usize;
            for i in 0..n {
                if ri < rows.len() && rows[ri].0 == i {
                    out.push(rows[ri].1);
                    ri += 1;
                } else {
                    out.push(&c.seeds[0]);
                }
            }
        }
        C_TIME => {
            let p = record::DATE_PATTERNS[c.param as usize];
            let scale = record::time_scale(p);
            let mut codes = Vec::with_capacity(n);
            for _ in 0..n {
                codes.push(record::get_varint(image, pos)?);
            }
            let exact = codes.iter().filter(|&&c| c != 0).count();
            let mut fracs = Vec::with_capacity(exact);
            if scale > 1 {
                for _ in 0..exact {
                    fracs.push(record::get_varint(image, pos)?);
                }
            }
            let mut last = 0i64;
            let mut fi = 0usize;
            for code in codes {
                if code == 0 {
                    let v = line_at(image, pos)?;
                    out.push(v);
                } else {
                    let secs = last.wrapping_add(record::unzigzag(code - 1));
                    last = secs;
                    let frac = if scale > 1 { fracs[fi] as i64 } else { 0 };
                    fi += 1;
                    if frac >= scale {
                        return Err(record::corrupt("shape image: fraction"));
                    }
                    record::format_time(p, secs.wrapping_mul(scale).wrapping_add(frac), &mut out.bytes);
                    out.ends.push(out.bytes.len());
                }
            }
        }
        C_DEC => {
            let places = c.param as u32;
            let mut codes = Vec::with_capacity(n);
            for _ in 0..n {
                codes.push(record::get_varint(image, pos)?);
            }
            let exact = codes.iter().filter(|&&c| c != 0).count();
            let pl = image.get(*pos..*pos + exact).ok_or(record::corrupt("shape image: places"))?.to_vec();
            *pos += exact;
            let mut last = 0i64;
            let mut pi = 0usize;
            for code in codes {
                if code == 0 {
                    let v = line_at(image, pos)?;
                    out.push(v);
                } else {
                    let scaled = last.wrapping_add(record::unzigzag(code - 1));
                    last = scaled;
                    let p = pl[pi] as u32;
                    pi += 1;
                    if p > places {
                        return Err(record::corrupt("shape image: places"));
                    }
                    let div = 10i64.pow(places - p);
                    if scaled % div != 0 {
                        return Err(record::corrupt("shape image: decimal"));
                    }
                    let v = scaled / div;
                    push_decimal(&mut out.bytes, v, p);
                    out.ends.push(out.bytes.len());
                }
            }
        }
        C_DICT => {
            let ranks = image.get(*pos..*pos + n).ok_or(record::corrupt("shape image: ranks"))?.to_vec();
            *pos += n;
            let misses = ranks.iter().filter(|&&r| r == 0).count();
            let mut ids = Vec::with_capacity(misses);
            for _ in 0..misses {
                ids.push(record::get_varint(image, pos)?);
            }
            let news = ids.iter().filter(|&&i| i == 0).count();
            let mut entries: Vec<&[u8]> = Vec::with_capacity(news);
            for _ in 0..news {
                entries.push(line_at(image, pos)?);
            }
            let mut recent = c.recent.clone();
            let (mut ii, mut ei) = (0usize, 0usize);
            let mut fresh: Vec<&[u8]> = Vec::new();
            for r in ranks {
                let id = if r == 0 {
                    let code = ids[ii];
                    ii += 1;
                    let id = if code == 0 {
                        fresh.push(entries[ei]);
                        ei += 1;
                        (c.seeds.len() + fresh.len() - 1) as u32
                    } else {
                        (code - 1) as u32
                    };
                    recent.push_front(id);
                    id
                } else {
                    if r as usize > RECENT {
                        return Err(record::corrupt("shape image: rank"));
                    }
                    recent.touch_at(r as usize - 1)
                };
                let v: &[u8] = match c.seeds.get(id as usize) {
                    Some(s) => s,
                    None => fresh.get(id as usize - c.seeds.len()).copied().ok_or(record::corrupt("shape image: id"))?,
                };
                out.push(v);
            }
        }
        _ => {
            for _ in 0..n {
                let v = line_at(image, pos)?;
                out.push(v);
            }
        }
    }
    Ok(out)
}

/// `v` with `places` decimal places (as `parse_decimal` read it).
fn push_decimal(out: &mut Vec<u8>, v: i64, places: u32) {
    if places == 0 {
        record::push_int(out, v);
        return;
    }
    let neg = v < 0;
    let mag = v.unsigned_abs();
    let div = 10u64.pow(places);
    if neg {
        out.push(b'-');
    }
    record::push_int(out, (mag / div) as i64);
    out.push(b'.');
    let frac = (mag % div).to_string();
    for _ in frac.len()..places as usize {
        out.push(b'0');
    }
    out.extend_from_slice(frac.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logs(n: usize, seed: u64) -> Vec<u8> {
        let mut x = seed;
        let mut v = Vec::new();
        let mut t = 1_700_000_000u64;
        while v.len() < n {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            t += x % 7;
            v.extend_from_slice(format!("host{},{},/item/{}/page{}.html,{},{}.{:02}\n", x % 40, t, (x >> 8) % 5000, (x >> 20) % 9, if x % 11 == 0 { 404 } else { 200 }, (x >> 30) % 1000, (x >> 40) % 100).as_bytes());
        }
        v
    }

    #[test]
    fn small_objects_round_trip_and_shrink() {
        let sample = logs(2 << 20, 1);
        let dict = ShapeDict::train(&sample).expect("record-shaped");
        let dict = ShapeDict::from_bytes(&dict.to_bytes()).expect("serialized form");
        let objects = logs(400 << 10, 2);
        let mut at = 0usize;
        let (mut raw, mut with_shape, mut with_dict) = (0usize, 0usize, 0usize);
        let lz = Dict::train(&sample.chunks(1024).collect::<Vec<_>>(), 64 << 10);
        while at < objects.len() {
            let end = (at + 1024).min(objects.len());
            let end = objects[end..].iter().position(|&b| b == b'\n').map_or(objects.len(), |p| end + p + 1);
            let o = &objects[at..end];
            let mut c = Vec::new();
            dict.compress(o, &mut c);
            assert!(dict.decompress(&c).unwrap() == o, "round trip");
            raw += o.len();
            with_shape += c.len();
            let mut d = Vec::new();
            crate::compress_with_dict(&lz, o, &mut d);
            with_dict += d.len();
            at = end;
        }
        assert!(with_shape * 5 < with_dict * 4, "the shape dictionary should beat the plain dictionary by a fourth: {with_shape} vs {with_dict} ({raw} raw)");
        // JSON lines and a log of varying shape take their own kinds.
        let mut json = Vec::new();
        let mut log = Vec::new();
        let mut x = 9u64;
        for i in 0..40000u64 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            json.extend_from_slice(format!("{{\"id\": {}, \"host\": \"h{}\", \"ok\": {}, \"v\": {}.{}}}\n", i * 3, x % 50, x % 3 == 0, x % 100, x % 10).as_bytes());
            match x % 3 {
                0 => log.extend_from_slice(format!("081109 {:06} {} INFO dfs.DataNode$DataXceiver: Receiving block blk_-{} src: /10.250.{}.{}:{}\n", i, x % 200, 1000000000000000000 + (x >> 4) % 1000, x % 256, x % 200, x % 65536).as_bytes()),
                1 => log.extend_from_slice(format!("081109 {:06} {} INFO dfs.DataNode$PacketResponder: PacketResponder {} for block blk_{} terminating\n", i, x % 200, x % 3, 1000000000000000000 + (x >> 4) % 1000).as_bytes()),
                _ => log.extend_from_slice(format!("081109 {:06} {} INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.250.{}.{}:50010 is added to blk_{} size {}\n", i, x % 200, x % 256, x % 200, 1000000000000000000 + (x >> 4) % 1000, x % 70000000).as_bytes()),
            }
        }
        for data in [json, log] {
            let dict = ShapeDict::train(&data[..data.len() / 2]).expect("record-shaped");
            let dict = ShapeDict::from_bytes(&dict.to_bytes()).expect("serialized form");
            let (mut raw, mut with_shape) = (0usize, 0usize);
            let mut at = data.len() / 2;
            while at < data.len() {
                let end = (at + 2048).min(data.len());
                let end = data[end..].iter().position(|&b| b == b'\n').map_or(data.len(), |p| end + p + 1);
                let o = &data[at..end];
                let mut c = Vec::new();
                dict.compress(o, &mut c);
                assert!(dict.decompress(&c).unwrap() == o, "round trip");
                raw += o.len();
                with_shape += c.len();
                at = end;
            }
            assert!(with_shape * 6 < raw, "2 KB objects should shrink sixfold: {with_shape} of {raw}");
        }
        // Lines that fit no frame, an empty object, no trailing newline, odd bytes.
        for o in [&b""[..], b"x", b"not,a,row\n", b"host1,1700000000,/item/1/page1.html,200,1.00", b"a,b,c,d,e,f\n\n\x00\xff\n"] {
            let mut c = Vec::new();
            dict.compress(o, &mut c);
            assert!(dict.decompress(&c).unwrap() == o, "round trip of {:?}", o);
        }
    }
}
