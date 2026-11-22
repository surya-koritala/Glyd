//! Pack-format compatibility: every format this glyd reads, held as small saved checkpoints in `tests/data/packs/` (a tiny Qwen3 packed by a release, or by the
//! packer of an old one), each of which must load and decode bit for bit, on the CPU, on every merge.
//!
//! `committed_packs_verify` is the guard: `verify` decodes every pack of each fixture and compares each decoded tensor with the sha256 the release that saved it recorded in
//! glyd.json (the weights themselves for a lossless layout, the decoded ones for a lossy one), so a change to a decoder that reads an old file to other bytes fails here.
//! `the_current_packer_writes_the_released_files` is the other direction: the packer of this glyd, run on the same tiny checkpoint, writes the fixtures' files byte for byte;
//! a deliberate change of a layout's bytes fails it, and wants a new format version (and a new fixture) rather than an edit of the old.
//!
//! The fixtures: `v0.33.1-LAYOUT` was written by the `glyd pack` of release 0.33.1 (the tiny checkpoint `tiny_source` writes, `--layout LAYOUT`); `v1-0.27.0-mma` is a glyd-v1
//! save as glyd 0.27's `pack_mma` made it (the tiered layout of glyd 0.20 to 0.27: the Python packer is the only one that wrote it, so its layout is `crate::v1`'s, held as the
//! one matrix `legacy` makes). To add the next release's: build it, run `glyd pack` on the checkpoint `GLYD_PACK_SRC=DIR cargo test -p glyd-gpu write_pack_sources -- --ignored`
//! writes, and put the output in `tests/data/packs/vX.Y.Z-LAYOUT`.
use super::*;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/data/packs");
/// The layouts `glyd pack --layout` takes, by the name a fixture's directory ends in.
const LAYOUTS: &[(&str, Layout)] = &[
    ("mma", Layout::Tiered),
    ("mma12", Layout::Twelve),
    ("a", Layout::Compact),
    ("i6", Layout::Int6),
    ("i5", Layout::Int5),
    ("i4", Layout::Int4),
    ("i3", Layout::Int3),
];

/// The tiny Qwen3 the fixtures hold the packs of: one layer, 128-wide, random bf16 weights from a fixed seed, the output layer tied.
fn tiny_source(src: &Path) {
    std::fs::create_dir_all(src).unwrap();
    std::fs::write(src.join("config.json"), r#"{"architectures": ["Qwen3ForCausalLM"], "num_hidden_layers": 1, "tie_word_embeddings": true}"#).unwrap();
    std::fs::write(src.join("tokenizer.json"), "{}").unwrap();
    let mut seed = 7u64;
    let mut t = |name: &str, shape: &[u64]| {
        let n: u64 = shape.iter().product();
        let w: Vec<u16> = (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                ((seed >> 20) as u16 & 0x807F) | ((116 + (seed >> 40) % 14) as u16) << 7
            })
            .collect();
        Saved { name: name.into(), dtype: "BF16", shape: shape.to_vec(), bytes: u16_bytes(&w), sha256: None }
    };
    let mut ts = vec![t("model.embed_tokens.weight", &[128, 128]), t("model.norm.weight", &[128])];
    for (n, s) in [
        ("self_attn.q_proj.weight", &[128, 128][..]),
        ("self_attn.k_proj.weight", &[64, 128]),
        ("self_attn.v_proj.weight", &[64, 128]),
        ("self_attn.o_proj.weight", &[128, 128]),
        ("self_attn.q_norm.weight", &[32]),
        ("self_attn.k_norm.weight", &[32]),
        ("mlp.gate_proj.weight", &[128, 128]),
        ("mlp.up_proj.weight", &[128, 128]),
        ("mlp.down_proj.weight", &[128, 128]),
        ("input_layernorm.weight", &[128]),
        ("post_attention_layernorm.weight", &[128]),
    ] {
        ts.push(t(&format!("model.layers.0.{n}"), s));
    }
    write_safetensors(&src.join("model.safetensors"), &mut ts).unwrap();
}

/// The glyd-v1 save of one matrix, as glyd 0.27 saved it (`v1::pack`; its tiers and its sha256 in glyd.json).
fn legacy(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    let (rows, cols) = (128, 192);
    let w = pack::tests::weights(rows * cols, 0.01, 17);
    let p = crate::v1::pack(&w, rows, cols);
    let steps = (rows * cols / pack::STEP) as u64;
    let saved = |name: &str, dtype: &'static str, n: u64, bytes: Vec<u8>| Saved { name: name.into(), dtype, shape: vec![n], bytes, sha256: None };
    let mut ts = vec![
        saved("m.o_proj.glyd_data", "U8", steps * 1280, p.data.clone()),
        saved("m.o_proj.glyd_blocks", "U8", p.blocks.len() as u64, p.blocks.clone()),
        saved("m.o_proj.glyd_block_base", "I32", steps + 1, le_bytes(&p.block_base, i32::to_le_bytes)),
    ];
    write_safetensors(&dir.join("model.safetensors"), &mut ts).unwrap();
    let tiers = p.tiers.iter().map(u32::to_string).collect::<Vec<_>>().join(", ");
    std::fs::write(
        dir.join("glyd.json"),
        format!(
            "{{\"format\": \"glyd-v1\", \"glyd\": \"0.27.0\", \"source\": {{\"repo\": null, \"revision\": null}}, \"layout\": \"mma\", \"packs\": {{\"m.o_proj\": {{\"layout\": \"mma\", \"shape\": [{rows}, {cols}], \"tiers\": [{tiers}], \"tensors\": [{{\"name\": \"m.o_proj.weight\", \"shape\": [{rows}, {cols}], \"sha256\": \"{}\"}}]}}}}, \"tensors\": {{}}}}",
            sha256_u16(&w)
        ),
    )
    .unwrap();
}

/// Every fixture directory, by name.
fn fixtures() -> Vec<(String, PathBuf)> {
    let mut v: Vec<(String, PathBuf)> = std::fs::read_dir(FIXTURES)
        .unwrap_or_else(|e| panic!("{FIXTURES}: {e}"))
        .map(|e| e.unwrap())
        .filter(|e| e.path().join("glyd.json").is_file())
        .map(|e| (e.file_name().to_string_lossy().into_owned(), e.path()))
        .collect();
    v.sort();
    v
}

/// Every committed pack loads and decodes, on the CPU, to the sha256 of the release that saved it.
#[test]
fn committed_packs_verify() {
    let all = fixtures();
    assert!(all.len() >= 8, "the fixtures of tests/data/packs are missing ({} found)", all.len());
    for (name, dir) in &all {
        let v = verify(dir, 2, &Decoder::Cpu).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(v.packed > 0 && v.unchecked == 0, "{name}: {} packs, {} tensors unchecked", v.packed, v.unchecked);
    }
}

/// A fixture is refused when a byte of its pack is changed: the check above is not one that passes whatever it reads.
#[test]
fn a_changed_byte_of_a_committed_pack_is_refused() {
    let tmp = std::env::temp_dir().join(format!("glyd-pack-compat-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    for (name, dir) in fixtures() {
        let copy = tmp.join(&name);
        std::fs::create_dir_all(&copy).unwrap();
        for f in std::fs::read_dir(&dir).unwrap() {
            let f = f.unwrap();
            std::fs::copy(f.path(), copy.join(f.file_name())).unwrap();
        }
        let c = Checkpoint::open(&copy).unwrap();
        let first = c.names().find(|n| n.contains("glyd_data")).unwrap_or_else(|| panic!("{name}: no pack")).to_string();
        let (f, info) = c.get(&first).unwrap();
        let mut bytes = std::fs::read(&f.path).unwrap();
        bytes[info.offset as usize + 5] ^= 0x10;
        std::fs::write(&f.path, bytes).unwrap();
        assert!(verify(&copy, 2, &Decoder::Cpu).is_err(), "{name}: a changed byte of {first} was not refused");
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

/// The packer of this glyd writes the tiny checkpoint's packs as each release's fixture holds them (the weights' files, byte for byte; glyd.json's packs and
/// tensors; not its `glyd` or `source`, which name the saving glyd and where the checkpoint was).
#[test]
fn the_current_packer_writes_the_released_files() {
    let tmp = std::env::temp_dir().join(format!("glyd-pack-compat-src-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let src = tmp.join("src");
    tiny_source(&src);
    let source = Source::find(src.to_str().unwrap()).unwrap();
    let mut checked = 0;
    for (name, dir) in fixtures() {
        let Some(layout) = LAYOUTS.iter().find(|(l, _)| name.starts_with('v') && name.ends_with(&format!("-{l}")) && !name.starts_with("v1-")).map(|(_, l)| *l) else { continue };
        let out = tmp.join(&name);
        save(&source, &out, 2, SHARD_BYTES, true, layout).unwrap_or_else(|e| panic!("{name}: {e}"));
        for f in std::fs::read_dir(&dir).unwrap() {
            let f = f.unwrap();
            let file = f.file_name().to_string_lossy().into_owned();
            if file == "glyd.json" {
                let (a, b) = (json::parse(&std::fs::read_to_string(f.path()).unwrap()).unwrap(), json::parse(&std::fs::read_to_string(out.join(&file)).unwrap()).unwrap());
                for k in ["format", "layout", "level", "packs", "tensors"] {
                    assert_eq!(format!("{:?}", a.get(k)), format!("{:?}", b.get(k)), "{name}: glyd.json's {k} is not what the release wrote");
                }
            } else {
                assert!(std::fs::read(f.path()).unwrap() == std::fs::read(out.join(&file)).unwrap(), "{name}: {file} is not the bytes the release wrote");
            }
        }
        checked += 1;
    }
    assert!(checked >= 7, "only {checked} fixtures of a release's packer were found");
    let _ = std::fs::remove_dir_all(&tmp);
}

/// Writes what the fixtures are made from: the tiny checkpoint to `$GLYD_PACK_SRC/src` (for the release's `glyd pack`), and the glyd-v1 fixture into tests/data/packs.
#[test]
#[ignore]
fn write_pack_sources() {
    let dir = PathBuf::from(std::env::var("GLYD_PACK_SRC").expect("GLYD_PACK_SRC: a folder"));
    tiny_source(&dir.join("src"));
    legacy(&Path::new(FIXTURES).join("v1-0.27.0-mma"));
}
