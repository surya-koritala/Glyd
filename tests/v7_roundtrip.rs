use glyd::v7_encode::{encode_block, payload_layout, Sequence, Tables};
use glyd::v7_format::{SubHeader, S_EXTRA, S_LIT, S_LL, S_ML, S_OFF};

/// The blocks of a stream: each header with its payload.
fn blocks_of(c: &[u8]) -> Vec<(glyd::format::BlockHeader, &[u8])> {
    let (mut cursor, mut v) = (0usize, Vec::new());
    while cursor < c.len() {
        let (h, used) = glyd::format::BlockHeader::read(&c[cursor..]).unwrap();
        let end = cursor + used + h.payload_len();
        v.push((h, &c[cursor + used..end]));
        cursor = end;
    }
    v
}

fn layout_of(h: &glyd::format::BlockHeader, body: &[u8]) -> glyd::v7_encode::Layout {
    glyd::v7_encode::payload_layout_of(h, body).unwrap()
}

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

use glyd::error::CodecError;
use glyd::v7_decode::{decode_block, DecTables, Scratch};

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
    let n = unsafe { decode_block(&p1, true, false, false, seqs.len(), lits.len(), &mut dst, base, expect.len(), &mut dtab, &mut scratch, None) }.unwrap();
    assert_eq!(n, expect.len());
    assert_eq!(&dst[..n], &expect[..]);
    let n2 = unsafe { decode_block(&p2, true, false, false, seqs.len(), lits.len(), &mut dst[n..], base, expect.len(), &mut dtab, &mut scratch, None) }.unwrap();
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
    let r = unsafe { decode_block(&p, true, false, false, seqs.len(), lits.len(), &mut dst, base, expect_len, &mut DecTables::none(), &mut Scratch::new(), None) };
    assert!(matches!(r, Err(CodecError::OffsetOutOfBounds { offset: 1_000_000, .. })), "{:?}", r);
    let (seqs, lits) = well_formed_sequences();
    let mut p = Vec::new();
    encode_block(&seqs, &lits, 0, &mut Tables::none(), &mut p);
    let r = unsafe { decode_block(&p, true, false, false, seqs.len(), lits.len(), &mut dst, base, expect_len - 1, &mut DecTables::none(), &mut Scratch::new(), None) };
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
        let _ = unsafe { decode_block(&q, true, false, false, seqs.len(), lits.len(), &mut dst, base, expect.len(), &mut DecTables::none(), &mut scratch, None) };
    }
    for _ in 0..2_000 {
        let n_seq = rnd(&mut x) as usize % (seqs.len() * 2);
        let n_lit = rnd(&mut x) as usize % (lits.len() * 2);
        let len = rnd(&mut x) as usize % (expect.len() * 2);
        let _ = unsafe { decode_block(&p, true, false, false, n_seq, n_lit, &mut dst, base, len, &mut DecTables::none(), &mut scratch, None) };
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
    // Few enough sequences that a code table (45 bytes) plus the section
    // framing cannot beat the raw codes.
    for i in 0..60u32 {
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
    let n1 = unsafe { decode_block(&p1, true, false, false, seqs.len(), lits.len(), &mut dst, base, expect.len(), &mut dtab, &mut scratch, None) }.unwrap();
    assert_eq!(&dst[..n1], &expect[..]);
    let n2 = unsafe { decode_block(&p2, true, false, false, seqs.len(), lits.len(), &mut dst[n1..], base, expect.len(), &mut dtab, &mut scratch, None) }.unwrap();
    assert_eq!(&dst[n1..n1 + n2], &expect[..]);
    let n3 = unsafe { decode_block(&p3, true, false, false, seqs3.len(), lits3.len(), &mut dst[n1 + n2..], base, expect3.len(), &mut dtab, &mut scratch, None) }.unwrap();
    assert_eq!(&dst[n1 + n2..n1 + n2 + n3], &expect3[..]);
    let at = n1 + n2 + n3;
    let n4 = unsafe { decode_block(&p4, true, false, false, seqs.len(), lits.len(), &mut dst[at..], base, expect.len(), &mut dtab, &mut scratch, None) }.unwrap();
    assert_eq!(&dst[at..at + n4], &expect[..]);
    let at = at + n4;
    let n5 = unsafe { decode_block(&p5, true, false, false, seqs3.len(), lits3.len(), &mut dst[at..], base, expect3.len(), &mut dtab, &mut scratch, None) }.unwrap();
    assert_eq!(&dst[at..at + n5], &expect3[..]);
}

/// A `dst` with no slack past `uncompressed_len` (the parallel path's
/// per-block slices) must decode correctly with nothing written past its
/// end: the wild copies fall back to exact ones there. The worst shape
/// for the fixed 3x32 tails is a 33-byte run copied as 128 bytes, 95
/// past its end: one ends 65 bytes before the block end, so a margin
/// under 96 would write 30 bytes past `dst`. The empty block (one
/// literal-only sequence of length 0, `dst` of length 0) is the shape
/// where a saturating margin check let the unconditional 32-byte
/// literal copy through.
#[test]
fn v7_block_decode_into_exact_dst_writes_nothing_past_it() {
    for (seqs, lits) in [
        well_formed_sequences(),
        (
            vec![
                Sequence { lit_len: 200, match_len: 3, offset: 1 },
                Sequence { lit_len: 0, match_len: 33, offset: 50 },
                Sequence { lit_len: 65, match_len: 0, offset: 0 },
            ],
            (0..265u32).map(|i| b'a' + (i % 26) as u8).collect(),
        ),
        (vec![Sequence { lit_len: 0, match_len: 0, offset: 0 }], Vec::new()),
        // Shorter than the wild margin: every sequence takes the exact path.
        (
            vec![
                Sequence { lit_len: 10, match_len: 5, offset: 3 },
                Sequence { lit_len: 0, match_len: 20, offset: 7 },
                Sequence { lit_len: 5, match_len: 0, offset: 0 },
            ],
            (0..15u32).map(|i| b'a' + (i % 26) as u8).collect(),
        ),
        // Repeat offsets across the switch from wild to exact copies,
        // which falls in the middle of a group of eight (sequence 14 of
        // 22 is the first to end within the margin): the exact loop must
        // continue the rep state, not re-apply the switching sequence's.
        // That sequence pushes a new offset and the next one swaps in the
        // previous (code 1), which a double push would have lost; the
        // matches cover the literals, so the two offsets copy different
        // bytes.
        (
            std::iter::once(Sequence { lit_len: 12, match_len: 12, offset: 12 })
                .chain(std::iter::repeat(Sequence { lit_len: 2, match_len: 12, offset: 12 }).take(13))
                .chain(std::iter::once(Sequence { lit_len: 2, match_len: 12, offset: 24 }))
                .chain(std::iter::repeat(Sequence { lit_len: 2, match_len: 12, offset: 12 }).take(6))
                .chain(std::iter::once(Sequence { lit_len: 3, match_len: 0, offset: 0 }))
                .collect(),
            (0..55u32).map(|i| b'a' + (i % 26) as u8).collect(),
        ),
    ] {
        let expect = materialize(&seqs, &lits);
        let mut p = Vec::new();
        encode_block(&seqs, &lits, 0, &mut Tables::none(), &mut p);
        let mut dst = vec![0xEEu8; expect.len() + 256];
        let base = dst.as_ptr();
        let n = unsafe { decode_block(&p, true, false, false, seqs.len(), lits.len(), &mut dst[..expect.len()], base, expect.len(), &mut DecTables::none(), &mut Scratch::new(), None) }.unwrap();
        assert_eq!(n, expect.len());
        assert_eq!(&dst[..n], &expect[..]);
        assert!(dst[n..].iter().all(|&b| b == 0xEE), "bytes past dst were written");
    }
}

// ---- Task 8: container integration ----

/// Max level through every container entry point: empty, one byte, a
/// constant run, incompressible bytes (stored raw), periodic data at
/// four periods (short offsets, long matches), 1.2 MB of text and 1.2 MB
/// of word salad (several blocks, tables reused across them).
#[test]
fn v7_max_level_roundtrip_through_container() {
    let mut x = 0x1234_5678_9ABC_DEF0u64;
    let mut rnd = |n: usize| -> Vec<u8> { (0..n).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x as u8 }).collect() };
    let mut inputs: Vec<Vec<u8>> = vec![Vec::new(), b"x".to_vec(), vec![b'A'; 300_000], rnd(300_000)];
    for period in [3usize, 17, 1000, 70_000] {
        let pat = rnd(period);
        inputs.push(pat.iter().cycle().take(600_000).copied().collect());
    }
    let mut text = Vec::new();
    while text.len() < 1_200_000 { text.extend_from_slice(b"the quick brown fox jumps over the lazy dog "); text.extend_from_slice(&rnd(2)); }
    inputs.push(text);
    // Word salad from a fixed vocabulary: stationary literal statistics
    // with the same support block after block, so the literal table gets
    // reused across blocks (through the container's table carry), which
    // the periodic inputs above never trigger. Its sequence tables are
    // not: under the double-fast parse a rare code (a literal run of 5,
    // an offset bucket) flickers between blocks, and reuse is decided by
    // cost (`table_cost`), so the old table's extra bits for that block's
    // histogram outweigh a fresh table's saving.
    let vocab: Vec<Vec<u8>> = (0..300).map(|_| { let n = 3 + rnd(1)[0] as usize % 8; rnd(n).iter().map(|b| b'a' + b % 26).collect() }).collect();
    let mut salad = Vec::new();
    while salad.len() < 1_200_000 { salad.extend_from_slice(&vocab[u16::from_le_bytes(rnd(2).try_into().unwrap()) as usize % 300]); salad.push(b' '); }
    inputs.push(salad);
    // 64-byte records: every block parses to the same few codes, so the
    // sequence tables get reused block after block; the literals (the
    // random bytes) stay raw.
    inputs.push(reuse_records());
    // (version, coded bits, reuse bits) per block, so an all-raw stream
    // cannot pass vacuously and cross-block table reuse is known to run.
    fn versions(c: &[u8]) -> Vec<(u16, u8, u8)> {
        let mut v = Vec::new();
        for (h, body) in blocks_of(c) {
            let sub = if h.flags & glyd::format::FLAG_RAW_UNCOMPRESSED != 0 { (0, 0) } else { let l = layout_of(&h, body).sub; (l.coded, l.reuse) };
            v.push((h.version, sub.0, sub.1));
        }
        v
    }
    for (k, input) in inputs.iter().enumerate() {
        let mut c = Vec::new();
        glyd::compress_into_max(input, &mut c);
        let v = versions(&c);
        assert_eq!(v.len(), (input.len() + 256 * 1024 - 1) / (256 * 1024), "block count, len {}", input.len());
        if k >= 4 {
            assert!(v.iter().all(|&(ver, _, _)| ver == glyd::format::VERSION_V9), "input {} should be all coded (compact) blocks: {:?}", k, v);
            assert!(c.len() < input.len() / 2, "input {} ratio: {} -> {}", k, input.len(), c.len());
        }
        if k == inputs.len() - 2 {
            assert!(v[1..].iter().any(|&(_, _, reuse)| reuse & 1 != 0), "salad should reuse the literal table: {:?}", v);
        }
        if k == inputs.len() - 1 {
            assert!(v[1..].iter().any(|&(_, _, reuse)| reuse & 2 != 0), "records should reuse the sequence tables: {:?}", v);
        }
        if k == 3 {
            assert!(v.iter().all(|&(ver, _, _)| ver == 6), "random input must be stored raw: {:?}", v);
        }
        assert_eq!(&glyd::decompress(&c).unwrap(), input, "max sequential, len {}", input.len());
        let mut p = Vec::new();
        glyd::compress_parallel_into_max(input, &mut p);
        assert_eq!(&glyd::decompress_parallel(&p).unwrap(), input, "max parallel, len {}", input.len());
    }
}

use glyd::v7_encode::{find_sequences_dfast, DfastTables, EncScratch};

/// Records with a fixed stride: offsets repeat. Shared with
/// `v7_max_matches_reference_block_encoder`.
/// 64-byte records whose sequence statistics are the same in every
/// block: a big-endian counter (its last byte changes every record,
/// so no repeat offset picks the record up a byte or two on), then 60
/// fixed bytes matched 64 back — a real offset for the block's first
/// record, repeats after.
fn reuse_records() -> Vec<u8> {
    let mut x = 5u64;
    let fixed: Vec<u8> = (0..60).map(|_| rnd(&mut x) as u8).collect();
    let mut records = Vec::new();
    for i in 0..16384u32 {
        records.extend_from_slice(&i.to_be_bytes());
        records.extend_from_slice(&fixed);
    }
    records
}

fn records_input() -> Vec<u8> {
    let mut data = Vec::new();
    let mut x = 9u64;
    for i in 0..20_000u32 {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        data.extend_from_slice(format!("id={:08} name=user{:03} score={:05}\n", i, x % 500, x % 100_000).as_bytes());
    }
    data
}

#[test]
fn dfast_parse_finds_repeats_and_roundtrips() {
    let data = records_input();
    let mut t = DfastTables::new();
    let mut seqs = Vec::new();
    let mut lits = Vec::new();
    let mut reps = [1u32, 4, 8];
    find_sequences_dfast(&data, 0, data.len().min(256 * 1024), &mut t, &mut reps, &mut seqs, &mut lits, &mut EncScratch::new());
    let out = materialize(&seqs, &lits);
    assert_eq!(&out[..], &data[..out.len()]);
    let matched: u32 = seqs.iter().map(|s| s.match_len).sum();
    assert!(matched as usize > out.len() * 6 / 10, "expected mostly matches: {}/{}", matched, out.len());
    let mut c = Vec::new();
    glyd::compress_into_max(&data, &mut c);
    assert_eq!(glyd::decompress(&c).unwrap(), data);
    assert!(c.len() * 4 < data.len(), "structured text should compress 4x+: {}", c.len());
}

/// `encode_block_with` (the path a caller with its own `Sequence` list
/// takes) must write byte-for-byte what `compress_into_max` puts in the
/// container, which instead codes directly from `find_sequences_dfast`'s
/// scratch (`encode_block_coded`) -- for every block the container did
/// not store raw. The container clears its parse table per call, so it
/// parses from the same empty state as the `DfastTables::new` here.
#[test]
fn v7_max_matches_reference_block_encoder() {
    use glyd::format::{MAX_BLOCK_SIZE, VERSION_V9};
    use glyd::v7_encode::encode_block_with;

    // One entry per container block, in order; None for a raw block.
    fn block_payloads(c: &[u8]) -> Vec<Option<(bool, &[u8])>> {
        blocks_of(c).into_iter().map(|(h, body)| if h.flags & glyd::format::FLAG_RAW_UNCOMPRESSED != 0 { None } else { Some((h.version == VERSION_V9, body)) }).collect()
    }

    let mut text = Vec::new();
    while text.len() < 600_000 { text.extend_from_slice(b"the quick brown fox jumps over the lazy dog. "); }
    let mut mixed = Vec::new();
    let mut x = 99u64;
    while mixed.len() < 600_000 { mixed.extend_from_slice(b"the quick brown fox jumps over the lazy dog "); mixed.extend_from_slice(&rnd(&mut x).to_le_bytes()[..3]); }

    for input in [records_input(), text, mixed] {
        let mut c = Vec::new();
        glyd::compress_into_max(&input, &mut c);
        let reference = block_payloads(&c);
        assert!(reference.iter().any(|p| p.is_some()), "expected at least one coded block");

        let mut t = DfastTables::new();
        let mut prev = Tables::none();
        let mut scratch = EncScratch::new();
        let (mut seqs, mut literals) = (Vec::new(), Vec::new());
        let mut block_start = 0usize;
        for expected in reference {
            let block_len = (input.len() - block_start).min(MAX_BLOCK_SIZE);
            let mut reps = [1u32, 4, 8];
            seqs.clear();
            literals.clear();
            find_sequences_dfast(&input, block_start, block_len, &mut t, &mut reps, &mut seqs, &mut literals, &mut scratch);
            match expected {
                Some((compact, payload)) => {
                    let mut out = Vec::new();
                    encode_block_with(&seqs, &literals, 0, &mut prev, &mut scratch, compact, &mut out);
                    assert_eq!(&out[..], payload, "block at {} payload mismatch", block_start);
                }
                // The container drops its tables when a block is
                // stored raw (see `compress_max_from`).
                None => prev = Tables::none(),
            }
            block_start += block_len;
        }
        assert_eq!(block_start, input.len());
    }
}

// ---- Task 11: dictionaries ----

/// A dictionary of 64 distinct JSON fields (as one trained from samples
/// would hold) and a small object using 16 of them once each with fresh
/// values: nothing repeats inside the doc, so without the dictionary it
/// is nearly all literals. Measured 0.65 of the plain size (a v7 block's
/// fixed ~158 bytes -- header, sub-header, the eight padded extra-bit
/// streams -- are paid either way, which is also why an input under
/// ~200 bytes is stored raw with or without a dictionary, carrying no id).
#[test]
fn v7_dictionary_helps_small_inputs_and_is_required() {
    use glyd::{compress_with_dict, decompress, decompress_parallel, decompress_with_dict};
    let mut x = 1u64;
    let fields: Vec<Vec<u8>> = (0..64).map(|i| { let r = rnd(&mut x); format!("\"field_{:02}\":\"{}-{:x}-value-of-field-{:02}\",", i, ["alpha", "beta", "gamma", "delta"][(r % 4) as usize], r >> 40, i).into_bytes() }).collect();
    let dict = glyd::Dict::from_content(&fields.concat(), &[]);
    let mut doc = b"{".to_vec();
    for i in 0..16 { doc.extend_from_slice(&fields[(i * 7) % 64]); doc.extend_from_slice(format!("\"n{}\":{},", i, rnd(&mut x) % 100_000).as_bytes()); }
    doc.push(b'}');
    let mut plain = Vec::new();
    glyd::compress_into_max(&doc, &mut plain);
    let mut with = Vec::new();
    compress_with_dict(&dict, &doc, &mut with);
    assert!(with.len() * 4 < plain.len() * 3, "dictionary must help: {} vs {}", with.len(), plain.len());
    assert_eq!(decompress_with_dict(&dict, &with).unwrap(), doc);
    assert_eq!(decompress(&with), Err(CodecError::CorruptedBitstream("dictionary id mismatch")));
    assert_eq!(decompress_with_dict(&glyd::Dict::from_content(b"wrong dictionary bytes", &[]), &with), Err(CodecError::CorruptedBitstream("dictionary id mismatch")));
    // The serialized form round-trips and names the same id.
    let again = glyd::Dict::from_bytes(&dict.to_bytes()).unwrap();
    assert_eq!(again.id(), dict.id());
    assert_eq!(decompress_with_dict(&again, &with).unwrap(), doc);
    assert_eq!(decompress_parallel(&with), Err(CodecError::CorruptedBitstream("dictionary streams are sequential-only")));
    // Several blocks, matches into the dictionary and across blocks; an
    // incompressible doc is stored raw (no id) and still round-trips.
    let mut big = Vec::new();
    while big.len() < 600_000 { big.extend_from_slice(&fields[(rnd(&mut x) % 64) as usize]); big.extend_from_slice(&rnd(&mut x).to_le_bytes()[..3]); }
    let noise: Vec<u8> = (0..5000).map(|_| rnd(&mut x) as u8).collect();
    for input in [big, noise, Vec::new()] {
        let mut c = Vec::new();
        compress_with_dict(&dict, &input, &mut c);
        assert_eq!(decompress_with_dict(&dict, &c).unwrap(), input);
    }
}

// ---- Decoder table state is per call ----

/// A v7 block that reuses the previous block's tables, cut out of its
/// stream and decoded on its own, must fail the same way whatever the
/// thread decoded before: the decoder's table carry is reset at the
/// start of every call, not only at `FLAG_CHAIN_RESET` blocks.
#[test]
fn v7_decode_does_not_carry_tables_across_calls() {
    // 64-byte records: every block after the first reuses the sequence
    // tables (see `v7_max_level_roundtrip_through_container`).
    let records = reuse_records();
    let mut c = Vec::new();
    glyd::compress_into_max(&records, &mut c);
    let mut block = None;
    for (h, body) in blocks_of(&c) {
        if h.flags & glyd::format::FLAG_RAW_UNCOMPRESSED == 0 && layout_of(&h, body).sub.reuse & 0b10 != 0 {
            let start = body.as_ptr() as usize - c.as_ptr() as usize;
            let head = start - h.header_len();
            block = Some(c[head..start + body.len()].to_vec());
            break;
        }
    }
    let block = block.expect("a block reusing the sequence tables");
    // Warm thread: this one has just decoded the whole stream.
    assert_eq!(glyd::decompress(&c).unwrap(), records);
    let warm = glyd::decompress(&block);
    let b = block.clone();
    let fresh = std::thread::spawn(move || glyd::decompress(&b)).join().unwrap();
    assert!(matches!(fresh, Err(CodecError::CorruptedBitstream(_))), "{:?}", fresh);
    assert!(matches!(warm, Err(CodecError::CorruptedBitstream(_))), "{:?}", warm);
    assert_eq!(warm, fresh);
}

/// `--max` output depends only on the input: the parse's thread-local
/// tables are cleared at the start of every call, so a thread that has
/// compressed something else before writes the same bytes as a fresh
/// thread, through both the sequential and the parallel entry points.
///
/// The input is built so stale entries would show: S is 20 KB of random
/// bytes. B holds S twice (the second copy is one long match, so nothing
/// in it is indexed), then 30 pieces of S's middle, each behind 3 loose
/// bytes, then text to 1.2 MB. A (the warm-up) parses S at B's second
/// copy's positions, behind a 20 KB prefix whose tail is a match and 6
/// loose bytes, so it indexes S at other positions than B's first copy
/// (the skip phase differs). From cleared tables every piece matches the
/// first copy of S at the first position B indexed there; from A's
/// leftovers about half of them match the second copy instead (where
/// A's indexed position comes first), a different offset.
#[test]
fn v7_max_output_is_deterministic() {
    let mut x = 77u64;
    let mut bytes = |n: usize| -> Vec<u8> { (0..n).map(|_| rnd(&mut x) as u8).collect() };
    let s = bytes(20_000);
    let y = bytes(9_997);
    let a = [y.clone(), y, bytes(6), s.clone()].concat();
    let mut b = [s.clone(), s.clone()].concat();
    for i in 0..30 { b.extend_from_slice(&bytes(3)); let k = 1000 + 600 * i; b.extend_from_slice(&s[k..k + 200]); }
    while b.len() < 1_200_000 { b.extend_from_slice(b"the quick brown fox jumps over the lazy dog "); b.extend_from_slice(&bytes(3)); }
    // A in every 256 KB chunk (zero padding indexes nothing), so the
    // parallel path's pool threads are warmed too.
    let a_chunks: Vec<u8> = (0..8).flat_map(|_| { let mut c = a.clone(); c.resize(256 * 1024, 0); c }).collect();
    for level in [glyd::compress_into_max as fn(&[u8], &mut Vec<u8>), glyd::compress_parallel_into_max] {
        let run = move |input: &[u8]| { let mut c = Vec::new(); level(input, &mut c); c };
        run(&a_chunks);
        run(&a);
        let warm1 = run(&b);
        let warm2 = run(&b);
        let bb = b.clone();
        let fresh = std::thread::spawn(move || run(&bb)).join().unwrap();
        assert!(warm1 == fresh, "warm thread differs from a fresh one");
        assert!(warm2 == fresh, "second warm run differs from a fresh one");
    }
}
