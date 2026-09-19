//! Files written by earlier releases decode unchanged: v0.2.0's default
//! (format v6), max and ultra (format v7) output of tests/data/v7-sample.bin.
//! v8 (this release on) widened the window and compacted the section
//! layout; every earlier format stays readable.

fn check(name: &str) {
    let plain = std::fs::read("tests/data/v7-sample.bin").unwrap();
    let c = std::fs::read(format!("tests/data/{name}")).unwrap();
    let d = glyd::decompress(&c).unwrap_or_else(|e| panic!("{name}: {e:?}"));
    assert!(d == plain, "{name}: decoded bytes differ");
    let mut exact = vec![0u8; plain.len()];
    assert_eq!(glyd::decompress_into(&c, &mut exact).unwrap(), plain.len(), "{name}: exact-size decode");
    assert!(exact == plain);
}

#[test]
fn v6_default_from_v0_2_0() {
    check("v6-sample-default.glyd");
}

#[test]
fn v7_max_from_v0_2_0() {
    check("v7-sample-max.glyd");
}

#[test]
fn v7_ultra_from_v0_2_0() {
    check("v7-sample-ultra.glyd");
}

/// The current writer produces v8 blocks, which are denser than the v7
/// blocks of the same input (the 8 MB window cannot matter at 85 KB; the
/// section layout does).
#[test]
fn v8_is_written_now_and_is_denser() {
    use glyd::format::{BlockHeader, HEADER_SIZE, VERSION_V8};
    let plain = std::fs::read("tests/data/v7-sample.bin").unwrap();
    let old = std::fs::read("tests/data/v7-sample-max.glyd").unwrap();
    let mut now = Vec::new();
    glyd::compress_into_max(&plain, &mut now);
    let h: BlockHeader = unsafe { std::ptr::read_unaligned(now.as_ptr() as *const BlockHeader) };
    let version = h.version;
    assert_eq!(version, VERSION_V8);
    assert!(now.len() < old.len(), "v8 {} vs v7 {}", now.len(), old.len());
    assert!(now.len() >= HEADER_SIZE);
    assert_eq!(glyd::decompress(&now).unwrap(), plain);
}
