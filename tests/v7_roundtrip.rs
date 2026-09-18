use simd_stream_codec::v7_encode::{encode_block, payload_layout, Sequence, Tables};
use simd_stream_codec::v7_format::{SubHeader, S_EXTRA, S_LIT, S_LL, S_ML, S_OFF};

fn sample_sequences() -> (Vec<Sequence>, Vec<u8>) {
    // "abcabcabc..." style: literal runs then matches at small offsets.
    let mut seqs = Vec::new();
    let mut lits = Vec::new();
    let mut x = 42u64;
    for i in 0..5000u32 {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        let ll = (x % 20) as u32;
        for _ in 0..ll { lits.push((x >> 8) as u8); }
        let ml = 3 + (x >> 16) as u32 % 40;
        let offset = if i % 3 == 0 { 100 } else { 1 + (x >> 24) as u32 % 5000 };
        seqs.push(Sequence { lit_len: ll, match_len: ml, offset });
    }
    lits.extend_from_slice(b"tail literals");
    seqs.push(Sequence { lit_len: 13, match_len: 0, offset: 0 });
    (seqs, lits)
}

#[test]
fn v7_encode_block_layout_is_self_describing() {
    let (seqs, lits) = sample_sequences();
    let mut prev = Tables::none();
    let mut out = Vec::new();
    encode_block(&seqs, &lits, 0, &mut prev, &mut out);
    let layout = payload_layout(&out).unwrap();
    assert_eq!(layout.sub.dict_id, 0);
    let total: usize = SubHeader::BYTES + layout.sub.sizes.iter().map(|&s| s as usize).sum::<usize>();
    assert_eq!(total, out.len());
    for s in [S_LIT, S_LL, S_ML, S_OFF, S_EXTRA] {
        assert_eq!(layout.sections[s].len(), layout.sub.sizes[s] as usize);
    }
    // Skewed sequence codes must have been coded, not stored raw.
    assert!(layout.sub.coded & (1 << S_LL) != 0);
    assert!(layout.sub.coded & (1 << S_OFF) != 0);
    // A second block with the same statistics reuses the sequence tables.
    let mut out2 = Vec::new();
    encode_block(&seqs, &lits, 0, &mut prev, &mut out2);
    let l2 = payload_layout(&out2).unwrap();
    assert!(l2.sub.reuse & 0b10 != 0);
    assert!(out2.len() < out.len());
}

/// A symbol absent from the previous block's table (count 0) but present,
/// even rarely, in this block's data must not pass the reuse check: a
/// reused table that assigns it zero probability sends the tANS encoder
/// out of bounds (see `close`). Block 2 introduces ll code 20 (lit_len
/// 300, via log2 bucketing), which never appears in `sample_sequences()`.
#[test]
fn v7_encode_reuse_with_new_symbol_does_not_panic() {
    let (seqs1, lits1) = sample_sequences();
    let mut seqs2 = seqs1.clone();
    let mut lits2 = lits1.clone();
    let added: u32 = (300 - seqs2[10].lit_len) + (300 - seqs2[11].lit_len);
    seqs2[10].lit_len = 300;
    seqs2[11].lit_len = 300;
    lits2.extend(std::iter::repeat(b'x').take(added as usize)); // keep lit_lens summing to lits.len()

    let mut prev = Tables::none();
    let mut out1 = Vec::new();
    encode_block(&seqs1, &lits1, 0, &mut prev, &mut out1);
    let mut out2 = Vec::new();
    encode_block(&seqs2, &lits2, 0, &mut prev, &mut out2); // must not panic

    let l2 = payload_layout(&out2).expect("second block payload must parse");
    assert!(l2.sub.reuse & 0b10 == 0, "must not reuse ll/ml/off tables when a new symbol appears");
}

/// The up-front reuse decision is optimistic (based on a closeness check
/// against fresh counts); if one of the three streams then decides raw
/// beats coding after all, the other two must not be left with an
/// omitted table under a header that ends up reporting "not reused" --
/// every coded, non-reused section must carry its own table. Block 1 (400
/// sequences) makes coding pay for all three streams; block 2 (192 of the
/// same shape) shrinks the win enough that `ml` alone falls back to raw
/// while reusing `ll`/`off`'s tables still looks attractive up front.
#[test]
fn v7_encode_reuse_falls_back_when_a_stream_goes_raw() {
    fn repeated_matches(n: u32) -> (Vec<Sequence>, Vec<u8>) {
        let mut seqs: Vec<Sequence> =
            (0..n).map(|i| Sequence { lit_len: 0, match_len: 3 + (i % 16), offset: 1 }).collect();
        seqs.push(Sequence { lit_len: 0, match_len: 0, offset: 0 });
        (seqs, Vec::new())
    }
    let (seqs1, lits1) = repeated_matches(400);
    let (seqs2, lits2) = repeated_matches(192);

    let mut prev = Tables::none();
    let mut out1 = Vec::new();
    encode_block(&seqs1, &lits1, 0, &mut prev, &mut out1);
    let mut out2 = Vec::new();
    encode_block(&seqs2, &lits2, 0, &mut prev, &mut out2);

    let l2 = payload_layout(&out2).expect("second block payload must parse");
    for (s, n_symbols) in [(S_LL, 32u8), (S_ML, 32u8), (S_OFF, 24u8)] {
        if l2.sub.coded & (1 << s) != 0 && l2.sub.reuse & 0b10 == 0 {
            let sec = &out2[l2.sections[s].clone()];
            assert_eq!(sec[0], n_symbols, "coded, non-reused stream {} must carry its own table", s);
        }
    }
}

// ---- Task 7: block decoder ----

use simd_stream_codec::error::CodecError;
use simd_stream_codec::v7_decode::{decode_block, DecTables, Scratch};

/// Oracle: materialize sequences + literals into the bytes they describe.
fn materialize(seqs: &[Sequence], lits: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut lp = 0usize;
    for s in seqs {
        out.extend_from_slice(&lits[lp..lp + s.lit_len as usize]);
        lp += s.lit_len as usize;
        for _ in 0..s.match_len {
            let b = out[out.len() - s.offset as usize];
            out.push(b);
        }
    }
    out
}

fn rnd(x: &mut u64) -> u64 { *x ^= *x << 13; *x ^= *x >> 7; *x ^= *x << 17; *x }

fn well_formed_sequences() -> (Vec<Sequence>, Vec<u8>) {
    // Offsets never exceed what has been produced so far. Literal bytes are
    // text-like (16 skewed symbols) so the literal stream gets coded, and
    // vary within a run so a literal-ordering bug cannot hide.
    let mut seqs = Vec::new();
    let mut lits = Vec::new();
    let mut produced = 0u32;
    let mut x = 7u64;
    for i in 0..4000u32 {
        let r = rnd(&mut x);
        let ll = if i == 0 { 50 } else { (r % 12) as u32 };
        for _ in 0..ll { lits.push(b"aaaabbbcdefghijk"[(rnd(&mut x) >> 8) as usize % 16]); }
        produced += ll;
        let ml = 3 + ((r >> 16) % 60) as u32;
        let offset = 1 + ((r >> 24) as u32 % produced.min(2000));
        seqs.push(Sequence { lit_len: ll, match_len: ml, offset });
        produced += ml;
    }
    lits.extend_from_slice(b"end");
    seqs.push(Sequence { lit_len: 3, match_len: 0, offset: 0 });
    (seqs, lits)
}

#[test]
fn v7_block_roundtrip_two_blocks_with_reuse() {
    let (seqs, lits) = well_formed_sequences();
    let expect = materialize(&seqs, &lits);
    let mut prev = Tables::none();
    let mut p1 = Vec::new();
    encode_block(&seqs, &lits, 0, &mut prev, &mut p1);
    let mut p2 = Vec::new();
    encode_block(&seqs, &lits, 0, &mut prev, &mut p2);
    assert_eq!(payload_layout(&p1).unwrap().sub.coded, 0b1111, "block 1 must code every stream");
    assert_eq!(payload_layout(&p2).unwrap().sub.reuse, 0b11, "block 2 must reuse the literal and sequence tables");

    let mut dst = vec![0u8; expect.len() * 2 + 128];
    let mut dtab = DecTables::none();
    let mut scratch = Scratch::new();
    let base = dst.as_ptr();
    let n = unsafe { decode_block(&p1, seqs.len(), lits.len(), &mut dst, base, expect.len(), &mut dtab, &mut scratch) }.unwrap();
    assert_eq!(n, expect.len());
    assert_eq!(&dst[..n], &expect[..]);
    let n2 = unsafe { decode_block(&p2, seqs.len(), lits.len(), &mut dst[n..], base, expect.len(), &mut dtab, &mut scratch) }.unwrap();
    assert_eq!(&dst[n..n + n2], &expect[..]);
}

#[test]
fn v7_block_rejects_bad_offset_and_wrong_length() {
    let (mut seqs, lits) = well_formed_sequences();
    seqs[5].offset = 1_000_000; // beyond produced output
    let mut prev = Tables::none();
    let mut p = Vec::new();
    encode_block(&seqs, &lits, 0, &mut prev, &mut p);
    let expect_len = materialize(&well_formed_sequences().0, &lits).len();
    let mut dst = vec![0u8; expect_len + 128];
    let base = dst.as_ptr();
    let r = unsafe { decode_block(&p, seqs.len(), lits.len(), &mut dst, base, expect_len, &mut DecTables::none(), &mut Scratch::new()) };
    assert!(matches!(r, Err(CodecError::OffsetOutOfBounds { offset: 1_000_000, .. })), "{:?}", r);
    let (seqs, lits) = well_formed_sequences();
    let mut p = Vec::new();
    encode_block(&seqs, &lits, 0, &mut Tables::none(), &mut p);
    let r = unsafe { decode_block(&p, seqs.len(), lits.len(), &mut dst, base, expect_len - 1, &mut DecTables::none(), &mut Scratch::new()) };
    assert!(matches!(r, Err(CodecError::CorruptedBitstream(_))), "{:?}", r);
}

/// Corrupt input comes back as an error, never a panic: byte mutations
/// and truncations of a valid payload, then header counts that disagree
/// with a valid payload.
#[test]
fn v7_block_decode_never_panics_on_mutations() {
    let (seqs, lits) = well_formed_sequences();
    let expect = materialize(&seqs, &lits);
    let mut p = Vec::new();
    encode_block(&seqs, &lits, 0, &mut Tables::none(), &mut p);
    let mut dst = vec![0u8; expect.len() + 128];
    let base = dst.as_ptr();
    let mut scratch = Scratch::new();
    let mut x = 12345u64;
    for it in 0..20_000 {
        let mut q = p.clone();
        match it % 4 {
            0 => { let i = rnd(&mut x) as usize % q.len(); q[i] = rnd(&mut x) as u8; }
            1 => { let i = rnd(&mut x) as usize % q.len(); q[i] ^= 1 << (rnd(&mut x) % 8); }
            2 => { q.truncate(rnd(&mut x) as usize % q.len()); }
            _ => { for _ in 0..8 { let i = rnd(&mut x) as usize % q.len(); q[i] = rnd(&mut x) as u8; } }
        }
        let _ = unsafe { decode_block(&q, seqs.len(), lits.len(), &mut dst, base, expect.len(), &mut DecTables::none(), &mut scratch) };
    }
    for _ in 0..2_000 {
        let n_seq = rnd(&mut x) as usize % (seqs.len() * 2);
        let n_lit = rnd(&mut x) as usize % (lits.len() * 2);
        let len = rnd(&mut x) as usize % (expect.len() * 2);
        let _ = unsafe { decode_block(&p, n_seq, n_lit, &mut dst, base, len, &mut DecTables::none(), &mut scratch) };
    }
}

/// Uniform random literal bytes are incompressible, so the literal
/// section is stored raw; a short block with near-uniform codes stores
/// all three code streams raw too (the coded estimate cannot beat one
/// byte per code once the table and padding overhead is paid). Both raw
/// paths must decode, and the decoder's table state must track the
/// encoder's across raw sections: blocks 1-2 raw, block 3 coded with a
/// fresh literal table, block 4 raw again, block 5 reusing block 3's
/// literal table (the encoder keeps it across raw-literal blocks).
#[test]
fn v7_block_decode_with_raw_literals_and_raw_codes() {
    let mut seqs = Vec::new();
    let mut lits = Vec::new();
    let mut produced = 0u32;
    let mut x = 99u64;
    for i in 0..200u32 {
        let ll = if i == 0 { 40 } else { (rnd(&mut x) % 16) as u32 };
        for _ in 0..ll { lits.push((rnd(&mut x) >> 24) as u8); }
        produced += ll;
        let ml = 3 + (rnd(&mut x) % 16) as u32;
        let offset = 1 + (rnd(&mut x) as u32 % produced.min(4000));
        seqs.push(Sequence { lit_len: ll, match_len: ml, offset });
        produced += ml;
    }
    seqs.push(Sequence { lit_len: 0, match_len: 0, offset: 0 }); // empty literal-only tail
    let expect = materialize(&seqs, &lits);

    let mut prev = Tables::none();
    let mut p1 = Vec::new();
    encode_block(&seqs, &lits, 0, &mut prev, &mut p1);
    assert_eq!(payload_layout(&p1).unwrap().sub.coded, 0, "every section of this block must be raw");
    let mut p2 = Vec::new();
    encode_block(&seqs, &lits, 0, &mut prev, &mut p2);
    let (seqs3, lits3) = well_formed_sequences();
    let expect3 = materialize(&seqs3, &lits3);
    let mut p3 = Vec::new();
    encode_block(&seqs3, &lits3, 0, &mut prev, &mut p3);
    assert!(payload_layout(&p3).unwrap().sub.coded & (1 << S_LIT) != 0, "block 3 literals must be coded");
    let mut p4 = Vec::new();
    encode_block(&seqs, &lits, 0, &mut prev, &mut p4);
    assert_eq!(payload_layout(&p4).unwrap().sub.coded, 0);
    let mut p5 = Vec::new();
    encode_block(&seqs3, &lits3, 0, &mut prev, &mut p5);
    assert!(payload_layout(&p5).unwrap().sub.reuse & 1 != 0, "block 5 must reuse block 3's literal table");

    let mut dst = vec![0u8; expect.len() * 3 + expect3.len() * 2 + 128];
    let mut dtab = DecTables::none();
    let mut scratch = Scratch::new();
    let base = dst.as_ptr();
    let n1 = unsafe { decode_block(&p1, seqs.len(), lits.len(), &mut dst, base, expect.len(), &mut dtab, &mut scratch) }.unwrap();
    assert_eq!(&dst[..n1], &expect[..]);
    let n2 = unsafe { decode_block(&p2, seqs.len(), lits.len(), &mut dst[n1..], base, expect.len(), &mut dtab, &mut scratch) }.unwrap();
    assert_eq!(&dst[n1..n1 + n2], &expect[..]);
    let n3 = unsafe { decode_block(&p3, seqs3.len(), lits3.len(), &mut dst[n1 + n2..], base, expect3.len(), &mut dtab, &mut scratch) }.unwrap();
    assert_eq!(&dst[n1 + n2..n1 + n2 + n3], &expect3[..]);
    let at = n1 + n2 + n3;
    let n4 = unsafe { decode_block(&p4, seqs.len(), lits.len(), &mut dst[at..], base, expect.len(), &mut dtab, &mut scratch) }.unwrap();
    assert_eq!(&dst[at..at + n4], &expect[..]);
    let at = at + n4;
    let n5 = unsafe { decode_block(&p5, seqs3.len(), lits3.len(), &mut dst[at..], base, expect3.len(), &mut dtab, &mut scratch) }.unwrap();
    assert_eq!(&dst[at..at + n5], &expect3[..]);
}
