//! Files written by earlier releases decode unchanged: v0.2.0's default
//! (format v6), max and ultra (format v7) output of tests/data/v7-sample.bin,
//! v0.3.0's max and ultra (format v8), and v0.4.0's compact blocks (format
//! v9) and dictionary objects. Every earlier format stays readable; a
//! fixture is never regenerated.

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

#[test]
fn v8_max_from_v0_3_0() {
    check("v8-sample-max.glyd");
}

#[test]
fn v8_ultra_from_v0_3_0() {
    check("v8-sample-ultra.glyd");
}

fn check_small(name: &str) {
    let plain = std::fs::read("tests/data/v7-sample.bin").unwrap();
    let c = std::fs::read(format!("tests/data/{name}")).unwrap();
    assert_eq!(c[0], glyd::format::COMPACT_MARKER, "{name}: a compact (v9) block");
    let d = glyd::decompress(&c).unwrap_or_else(|e| panic!("{name}: {e:?}"));
    assert!(d == plain[..20_000], "{name}: decoded bytes differ");
}

#[test]
fn v9_compact_max_from_v0_4_0() {
    check_small("v9-small-max.glyd");
}

#[test]
fn v9_compact_ultra_from_v0_4_0() {
    check_small("v9-small-ultra.glyd");
}

/// A serialized dictionary and objects compressed with it, from v0.4.0.
#[test]
fn v9_dictionary_objects_from_v0_4_0() {
    let plain = std::fs::read("tests/data/v7-sample.bin").unwrap();
    let dict = glyd::Dict::from_bytes(&std::fs::read("tests/data/v9-dict.glyddict").unwrap()).expect("dictionary parses");
    for name in ["v9-object-dict.glyd", "v9-object-dict-ultra.glyd"] {
        let c = std::fs::read(format!("tests/data/{name}")).unwrap();
        let d = glyd::decompress_with_dict(&dict, &c).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        assert!(d == plain[..3000], "{name}: decoded bytes differ");
        assert!(glyd::decompress(&c).is_err(), "{name}: decodes without its dictionary");
    }
}

/// The current writer produces compact (v9) blocks, denser than the v7
/// and v8 blocks of the same input (the 8 MB window cannot matter at
/// 85 KB; the section layout and the framing do).
#[test]
fn v9_is_written_now_and_is_denser() {
    let plain = std::fs::read("tests/data/v7-sample.bin").unwrap();
    let v7 = std::fs::read("tests/data/v7-sample-max.glyd").unwrap();
    let v8 = std::fs::read("tests/data/v8-sample-max.glyd").unwrap();
    let mut now = Vec::new();
    glyd::compress_into_max(&plain, &mut now);
    assert_eq!(now[0], glyd::format::COMPACT_MARKER);
    assert!(now.len() < v8.len() && v8.len() < v7.len(), "v9 {} vs v8 {} vs v7 {}", now.len(), v8.len(), v7.len());
    assert_eq!(glyd::decompress(&now).unwrap(), plain);
}
