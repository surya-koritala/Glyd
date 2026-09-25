//! PyTorch checkpoints (torch.save, PyTorch 2.14: two layers' weights, a
//! bf16 and an int64 tensor, AdamW's state; steps 2 and 3 of one run):
//! their storages opened as byte planes, and the later one against the
//! earlier as deltas, every byte back.

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!("{}/tests/data/torch/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

#[test]
fn storages_are_found_with_their_widths() {
    let s = glyd::deflate::torch_storages(&fixture("ckpt-3.pt")).unwrap();
    // fp32 weights and moments (4 bytes), the bf16 tensor (2), the int64 one (8).
    assert!(s.iter().any(|(_, w, _)| *w == 4) && s.iter().any(|(_, w, _)| *w == 2) && s.iter().any(|(_, w, _)| *w == 8), "{s:?}");
    assert!(s.iter().all(|(name, _, _)| name.starts_with(b"data/")));
    assert!(glyd::deflate::torch_storages(b"PK\x05\x06 not a checkpoint").is_none());
}

#[test]
fn a_checkpoint_comes_back() {
    let a = fixture("ckpt-3.pt");
    let mut c = Vec::new();
    glyd::compress_into_max(&a, &mut c);
    assert_eq!(glyd::decompress(&c).unwrap(), a);
    assert!(c.len() < a.len() * 9 / 10, "{} of {}", c.len(), a.len());
}

#[test]
fn a_checkpoint_against_the_one_before() {
    let (a, b) = (fixture("ckpt-2.pt"), fixture("ckpt-3.pt"));
    let mut d = Vec::new();
    glyd::compress_with_base(&a, &b, &mut d, false);
    assert!(glyd::needs_base(&d));
    assert_eq!(glyd::decompress_with_base(&a, &d).unwrap(), b);
    let mut alone = Vec::new();
    glyd::compress_into_max(&b, &mut alone);
    assert!(d.len() < alone.len(), "the delta {} against {} alone", d.len(), alone.len());
}
