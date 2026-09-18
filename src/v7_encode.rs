//! Format v7 block encoder: a sequence list plus literals -> payload.
//!
//! Payload = SubHeader, then five sections. A coded section is:
//!   [table: 128 bytes of packed Huffman lengths, or 1 + 2 * n bytes of
//!    tANS counts (u8 count then u16 LE counts); omitted when reused]
//!   [8 x u32 LE sub-stream sizes]
//!   [8 sub-streams, each ending in bits::PAD zero bytes]
//! A raw section is the bytes themselves (literals, or one code byte per
//! sequence). The extra-bits section is always 8 raw padded sub-streams
//! behind a size table: sub-stream k holds, for sequences i == k (mod 8)
//! in order, the ll extra bits, then ml, then offset extra bits.
use crate::bits::{BitWriter, PAD};
use crate::huff8;
use crate::tans;
use crate::v7_format::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sequence {
    pub lit_len: u32,
    pub match_len: u32,
    pub offset: u32,
}

/// The previous block's entropy tables, carried forward so the next block
/// can reuse them instead of writing a fresh one.
pub struct Tables {
    pub lit_lengths: Option<[u8; 256]>,
    pub ll: Option<Vec<u16>>,
    pub ml: Option<Vec<u16>>,
    pub off: Option<Vec<u16>>,
}

impl Tables {
    pub fn none() -> Self {
        Tables { lit_lengths: None, ll: None, ml: None, off: None }
    }
}

pub struct Layout {
    pub sub: SubHeader,
    pub sections: [std::ops::Range<usize>; 5],
}

pub fn payload_layout(payload: &[u8]) -> Option<Layout> {
    let sub = SubHeader::parse(payload)?;
    let mut pos = SubHeader::BYTES;
    let mut sections: [std::ops::Range<usize>; 5] = std::array::from_fn(|_| 0..0);
    for s in 0..5 {
        let end = pos.checked_add(sub.sizes[s] as usize)?;
        if end > payload.len() {
            return None;
        }
        sections[s] = pos..end;
        pos = end;
    }
    if pos != payload.len() {
        return None;
    }
    Some(Layout { sub, sections })
}

/// Reuse when every count is within 1/8 of the previous table's.
fn close(a: &[u16], b: &[u16]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(&x, &y)| (x as i32 - y as i32).abs() <= (x as i32 / 8).max(2))
}

fn write_substreams(streams: &[Vec<u8>], out: &mut Vec<u8>) {
    for s in streams {
        out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    }
    for s in streams {
        out.extend_from_slice(s);
    }
}

fn code_hist(codes: &[u8], n_symbols: usize) -> Vec<u32> {
    let mut hist = vec![0u32; n_symbols];
    for &c in codes {
        hist[c as usize] += 1;
    }
    hist
}

/// tANS-code `codes` against `counts` (either freshly normalized for this
/// block, or the previous block's when `reusing`), or leave the codes raw
/// if coding would not pay for itself. Returns the section bytes and
/// whether it was coded; a raw section never carries reuse.
fn encode_codes(codes: &[u8], hist: &[u32], counts: &[u16], reusing: bool) -> (Vec<u8>, bool) {
    let n_symbols = counts.len();
    let bits: f64 = (0..n_symbols)
        .map(|s| if hist[s] == 0 { 0.0 } else { hist[s] as f64 * -((counts[s] as f64) / tans::L as f64).log2() })
        .sum();
    let table_bytes = if reusing { 0 } else { 1 + 2 * n_symbols };
    let coded_estimate = (bits / 8.0) as usize + table_bytes + 8 * (4 + PAD);
    if coded_estimate + codes.len() / 50 >= codes.len() {
        return (codes.to_vec(), false);
    }
    let et = tans::EncodeTable::build(counts).expect("normalized counts sum to L");
    let streams = tans::encode8(codes, &et);
    let mut section = Vec::new();
    if !reusing {
        section.push(n_symbols as u8);
        for &c in counts {
            section.extend_from_slice(&c.to_le_bytes());
        }
    }
    write_substreams(&streams, &mut section);
    (section, true)
}

pub fn encode_block(seqs: &[Sequence], literals: &[u8], dict_id: u32, prev: &mut Tables, out: &mut Vec<u8>) {
    // Codes and extra bits.
    let n = seqs.len();
    let mut ll = Vec::with_capacity(n);
    let mut ml = Vec::with_capacity(n);
    let mut off = Vec::with_capacity(n);
    let mut extra: Vec<BitWriter> = (0..8).map(|_| BitWriter::new()).collect();
    let mut reps = Reps::new();
    for (i, s) in seqs.iter().enumerate() {
        let w = &mut extra[i % 8];
        let (c, nb, e) = ll_code(s.lit_len);
        ll.push(c);
        if nb > 0 {
            w.put(e as u64, nb as u32);
        }
        if s.match_len == 0 {
            debug_assert_eq!(i, n - 1, "literal-only sequence must be last");
            ml.push(0);
            off.push(0);
            continue;
        }
        let (c, nb, e) = ml_code(s.match_len);
        ml.push(c);
        if nb > 0 {
            w.put(e as u64, nb as u32);
        }
        let (c, nb, e) = reps.code_for(s.offset);
        off.push(c);
        if nb > 0 {
            w.put(e as u64, nb as u32);
        }
    }
    let extra_streams: Vec<Vec<u8>> = extra.into_iter().map(|w| w.finish()).collect();

    // Literals.
    let mut hist = [0u64; 256];
    for &b in literals {
        hist[b as usize] += 1;
    }
    let lengths = huff8::lengths_for(&hist);
    let lit_reuse = prev.lit_lengths.map_or(false, |p| {
        let est_prev: u64 = (0..256).map(|s| hist[s] * p[s] as u64).sum();
        let est_new: u64 = (0..256).map(|s| hist[s] * lengths[s] as u64).sum();
        p.iter().zip(lengths.iter()).all(|(&a, &b)| (a > 0) == (b > 0)) && est_prev <= est_new + (huff8::TABLE_BYTES as u64) * 8
    });
    let lit_lengths = if lit_reuse { prev.lit_lengths.unwrap() } else { lengths };
    let lit_coded_size = huff8::coded_size(&hist, &lit_lengths) - if lit_reuse { huff8::TABLE_BYTES } else { 0 } + 8 * (4 + PAD);
    let lit_coded = literals.len() >= 64 && lit_coded_size + literals.len() / 50 < literals.len();
    let mut lit_section = Vec::new();
    if lit_coded {
        if !lit_reuse {
            crate::huffman::pack_lengths(&lit_lengths, &mut lit_section);
        }
        let streams = huff8::encode(literals, &lit_lengths);
        write_substreams(&streams, &mut lit_section);
    } else {
        lit_section.extend_from_slice(literals);
    }

    // Sequence code streams: one up-front decision (not a per-stream, then
    // shadowed re-encode) -- reuse the previous block's three tables only
    // when all three are present and each is close to this block's fresh
    // counts; otherwise recompute and write fresh tables for all three.
    let ll_hist = code_hist(&ll, LL_SYMBOLS);
    let ml_hist = code_hist(&ml, ML_SYMBOLS);
    let off_hist = code_hist(&off, OFF_SYMBOLS);
    let ll_fresh = tans::normalize(&ll_hist, LL_SYMBOLS);
    let ml_fresh = tans::normalize(&ml_hist, ML_SYMBOLS);
    let off_fresh = tans::normalize(&off_hist, OFF_SYMBOLS);
    let seq_reuse = matches!(
        (&prev.ll, &prev.ml, &prev.off),
        (Some(a), Some(b), Some(c)) if close(a, &ll_fresh) && close(b, &ml_fresh) && close(c, &off_fresh)
    );
    let ll_counts = if seq_reuse { prev.ll.clone().unwrap() } else { ll_fresh };
    let ml_counts = if seq_reuse { prev.ml.clone().unwrap() } else { ml_fresh };
    let off_counts = if seq_reuse { prev.off.clone().unwrap() } else { off_fresh };
    let (ll_sec, ll_coded) = encode_codes(&ll, &ll_hist, &ll_counts, seq_reuse);
    let (ml_sec, ml_coded) = encode_codes(&ml, &ml_hist, &ml_counts, seq_reuse);
    let (off_sec, off_coded) = encode_codes(&off, &off_hist, &off_counts, seq_reuse);
    let seq_coded = ll_coded && ml_coded && off_coded;

    let mut extra_sec = Vec::new();
    write_substreams(&extra_streams, &mut extra_sec);

    let sub = SubHeader {
        coded: (lit_coded as u8) << S_LIT | (ll_coded as u8) << S_LL | (ml_coded as u8) << S_ML | (off_coded as u8) << S_OFF,
        reuse: (lit_coded && lit_reuse) as u8 | (((seq_reuse && seq_coded) as u8) << 1),
        dict_id,
        sizes: [lit_section.len() as u32, ll_sec.len() as u32, ml_sec.len() as u32, off_sec.len() as u32, extra_sec.len() as u32],
    };
    sub.write(out);
    out.extend_from_slice(&lit_section);
    out.extend_from_slice(&ll_sec);
    out.extend_from_slice(&ml_sec);
    out.extend_from_slice(&off_sec);
    out.extend_from_slice(&extra_sec);

    if lit_coded {
        prev.lit_lengths = Some(lit_lengths);
    }
    if seq_coded {
        prev.ll = Some(ll_counts);
        prev.ml = Some(ml_counts);
        prev.off = Some(off_counts);
    } else {
        prev.ll = None;
        prev.ml = None;
        prev.off = None;
    }
}
