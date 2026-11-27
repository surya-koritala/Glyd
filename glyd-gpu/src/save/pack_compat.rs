//! Pack-format compatibility: every format this glyd reads, held as small saved checkpoints in `tests/data/packs/` (a tiny dense checkpoint packed by a release, or by the
//! packer of an old one), each of which must load and decode bit for bit, on the CPU, on every merge.
//!
//! `committed_packs_verify` is the guard: `verify` decodes every pack of each fixture and compares each decoded tensor with the sha256 the release that saved it recorded in
//! glyd.json (the weights themselves for a lossless layout, the decoded ones for a lossy one), so a change to a decoder that reads an old file to other bytes fails here.
//! `the_current_packer_writes_the_released_files` is the other direction: the packer of this glyd, run on the same tiny checkpoint, writes the fixtures' files byte for byte;
//! a deliberate change of a layout's bytes fails it, and wants a new format version (and a new fixture) rather than an edit of the old.
//!
//! The fixtures: `main-NAME` hold what no release's `glyd pack` writes of the tiny dense checkpoint (the row layouts i6r, i5r and i4r; a Hummingbird save with its codebook rows c3r and c2r;
//! glyd-v7 at penguin and the exact four-bit i4x, which `glyd-engine pack` writes for Gemma 4 alone, the second from the tiny dense checkpoint trained to four bits, `tiny_source_as`), made
//! by this glyd (`write_pack_sources`): they say what a decoder must keep reading; the release that first writes each replaces them with its own. `v0.33.1-LAYOUT` was written
//! by the `glyd pack` of release 0.33.1 (the tiny checkpoint `tiny_source` writes, `--layout LAYOUT`). `v1-0.27.0-mma` and `v2-0.27.0-mma` are a glyd-v1 save of one matrix
//! and a glyd-v2 save of a layer's experts as glyd 0.27's `save_pretrained` made them (the tiered layout of glyd 0.20 to 0.27: the Python packer is the only one that wrote
//! it, so its layout is `crate::v1`'s); `v5-0.29.1-mma` the glyd-v5 save of the tiny dense checkpoint as 0.29.1's `save_pretrained(tables=True)` made it (the embedding packed as a
//! table). To add the next release's: build it, run `glyd pack` on the checkpoint `GLYD_PACK_SRC=DIR cargo test -p glyd-gpu write_pack_sources -- --ignored` writes, and
//! put the output in `tests/data/packs/vX.Y.Z-LAYOUT`. glyd-engine's `packcheck.rs` runs each that holds the tiny dense checkpoint on its CPU reference against a recorded digest.
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

/// The tiny dense checkpoint's tensors in the order their values are drawn: one layer, 128-wide (4 query heads and 2 key-value heads of 32), a vocabulary of 128, the output layer tied.
fn tiny_tensors() -> Vec<(String, Vec<u64>)> {
    let mut ts = vec![("model.embed_tokens.weight".to_string(), vec![128, 128]), ("model.norm.weight".to_string(), vec![128])];
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
        ts.push((format!("model.layers.0.{n}"), s.to_vec()));
    }
    ts
}

/// The tiny dense checkpoint the fixtures hold the packs of: random bf16 weights from a fixed seed (its config.json names `Qwen3ForCausalLM` only because
/// the released fixtures' do: the model is no release's). `qat`: its matrices as a checkpoint trained to four bits holds them
/// (each run of 32 of a row a code from -8 to 7 times one scale of a 12-bit significand, rounded to bf16 once: what the exact four-bit layout, i4x, holds bit
/// for bit), its norms as the other's.
fn tiny_source_as(src: &Path, qat: bool) {
    std::fs::create_dir_all(src).unwrap();
    std::fs::write(src.join("config.json"), r#"{"architectures": ["Qwen3ForCausalLM"], "num_hidden_layers": 1, "tie_word_embeddings": true}"#).unwrap();
    std::fs::write(src.join("tokenizer.json"), "{}").unwrap();
    let mut seed = 7u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let bf16 = |x: f32| (x.to_bits().wrapping_add(0x7fff + ((x.to_bits() >> 16) & 1)) >> 16) as u16;
    let mut ts: Vec<Saved> = tiny_tensors()
        .into_iter()
        .map(|(name, shape)| {
            let n = shape.iter().product::<u64>() as usize;
            let w: Vec<u16> = if qat && shape.len() == 2 {
                let mut w = Vec::with_capacity(n);
                while w.len() < n {
                    let s = (2048 + next() % 2048) as f32 * 2f32.powi(-17 - (next() % 3) as i32);
                    w.extend((0..32).map(|_| bf16(((next() % 16) as f32 - 8.0) * s)));
                }
                w
            } else {
                (0..n)
                    .map(|_| {
                        let x = next();
                        ((x >> 20) as u16 & 0x807F) | ((116 + (x >> 40) % 14) as u16) << 7
                    })
                    .collect()
            };
            Saved { name, dtype: "BF16", shape, bytes: u16_bytes(&w), sha256: None }
        })
        .collect();
    write_safetensors(&src.join("model.safetensors"), &mut ts).unwrap();
}

fn tiny_source(src: &Path) {
    tiny_source_as(src, false)
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

/// The glyd-v2 save of a layer's experts, as glyd 0.27's save_pretrained (format.py) made one: a mixture of experts' weight (its experts' matrices stacked, [E out, in]) as
/// one pack in the tiered layout of glyd 0.20 to 0.27 (`v1::pack`) under the module that holds it (`MODULE.glyd_WEIGHT_data`), its entry naming its experts and whether the model
/// holds their matrices transposed ([E, in, out]: gate_up_proj here; down_proj as [E, out, in]); the sha256 of each weight as the model holds it.
fn legacy_experts(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    let (e, out, inn) = (2usize, 64usize, 128usize);
    let m = "model.layers.0.mlp.experts";
    let (mut ts, mut packs) = (Vec::new(), Vec::new());
    for (weight, transposed, seed) in [("gate_up_proj", true, 19u64), ("down_proj", false, 23)] {
        // the pack's matrix: each expert's [out, in] by rows; the weight as the model holds it, each expert's transposed where it says so
        let w = pack::tests::weights(e * out * inn, 0.01, seed);
        let held: Vec<u16> = if transposed {
            (0..e).flat_map(|x| (0..inn).flat_map(move |c| (0..out).map(move |r| (x, r, c)))).map(|(x, r, c)| w[x * out * inn + r * inn + c]).collect()
        } else {
            w.clone()
        };
        let p = crate::v1::pack(&w, e * out, inn);
        let steps = (e * out * inn / pack::STEP) as u64;
        let key = format!("{m}.glyd_{weight}_");
        ts.push(Saved { name: format!("{key}data"), dtype: "U8", shape: vec![steps * 1280], bytes: p.data.clone(), sha256: None });
        ts.push(Saved { name: format!("{key}blocks"), dtype: "U8", shape: vec![p.blocks.len() as u64], bytes: p.blocks.clone(), sha256: None });
        ts.push(Saved {
            name: format!("{key}block_base"),
            dtype: "I32",
            shape: vec![steps + 1],
            bytes: le_bytes(&p.block_base, i32::to_le_bytes),
            sha256: None,
        });
        let shape = if transposed { [e, inn, out] } else { [e, out, inn] };
        let tiers = p.tiers.iter().map(u32::to_string).collect::<Vec<_>>().join(", ");
        packs.push(format!(
            "\"{m}.{weight}\": {{\"layout\": \"mma\", \"shape\": [{}, {inn}], \"tiers\": [{tiers}], \"tensors\": [{{\"name\": \"{m}.{weight}\", \"shape\": [{}, {}, {}], \"sha256\": \"{}\"}}], \"experts\": {e}, \"transposed\": {transposed}}}",
            e * out,
            shape[0],
            shape[1],
            shape[2],
            sha256_u16(&held)
        ));
    }
    write_safetensors(&dir.join("model.safetensors"), &mut ts).unwrap();
    std::fs::write(
        dir.join("glyd.json"),
        format!(
            "{{\"format\": \"glyd-v2\", \"glyd\": \"0.27.0\", \"source\": {{\"repo\": null, \"revision\": null}}, \"layout\": \"mma\", \"packs\": {{{}}}, \"tensors\": {{}}}}",
            packs.join(", ")
        ),
    )
    .unwrap();
}

/// glyd.json's top-level `key` set to `v` (added where it has none).
fn set(m: &mut Value, key: &str, v: Value) {
    let Value::Object(o) = m else { panic!("glyd.json is not an object") };
    match o.iter_mut().find(|(k, _)| k == key) {
        Some((_, x)) => *x = v,
        None => o.push((key.into(), v)),
    }
}

/// The glyd-v5 save of the tiny dense checkpoint, as glyd 0.29.1's save_pretrained(tables=True) made one: the save of glyd-v4 with the embedding (the table the output layer is
/// tied to) packed as a Linear is, its entry marked "table". (format.py packs a table of 2^20 weights or more; the tiny model's 16,384 are packed to hold the same entry.)
fn tables(tmp: &Path) {
    let out = Path::new(FIXTURES).join("v5-0.29.1-mma");
    let src = tmp.join("src");
    save_planned(&Source::find(src.to_str().unwrap()).unwrap(), &out, 2, SHARD_BYTES, Layout::Tiered, &plan_of(&src, None)).unwrap();
    let path = out.join("glyd.json");
    let mut m = json::parse(&std::fs::read_to_string(&path).unwrap()).unwrap();
    set(&mut m, "format", Value::from("glyd-v5"));
    set(&mut m, "glyd", Value::from("0.29.1"));
    set(&mut m, "source", Value::Object(vec![("repo".into(), Value::from("tiny-qwen3")), ("revision".into(), Value::Null)]));
    let Some(Value::Object(packs)) = (match &mut m {
        Value::Object(o) => o.iter_mut().find(|(k, _)| k == "packs").map(|(_, v)| v),
        _ => None,
    }) else {
        panic!("no packs")
    };
    let (_, Value::Object(embed)) = packs.iter_mut().find(|(k, _)| k == "model.embed_tokens").expect("the embedding packed") else { panic!("an entry") };
    embed.push(("table".into(), Value::Bool(true)));
    std::fs::write(&path, json::to_python(&m, 1)).unwrap();
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
    assert!(all.len() >= 16, "the fixtures of tests/data/packs are missing ({} found)", all.len());
    for (name, dir) in &all {
        let v = verify(dir, 2, &Decoder::Cpu).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(v.packed > 0 && v.unchecked == 0, "{name}: {} packs, {} tensors unchecked", v.packed, v.unchecked);
    }
}

/// `decoded` gives every tensor of each fixture once, as the model holds it: a packed one at glyd.json's sha256 (an experts' weight transposed where its entry says so), its
/// values as many as its shape's; the tensors saved as they are beside them.
#[test]
fn decoded_gives_each_tensor_as_the_model_holds_it() {
    for (name, dir) in fixtures() {
        let m = json::parse(&std::fs::read_to_string(dir.join("glyd.json")).unwrap()).unwrap();
        let want: HashMap<String, (Vec<u64>, String)> = m
            .get("packs")
            .and_then(Value::as_object)
            .unwrap()
            .iter()
            .flat_map(|(_, e)| e.get("tensors").and_then(Value::as_array).unwrap().iter())
            .map(|t| {
                (
                    t.get("name").and_then(Value::as_str).unwrap().into(),
                    (u64s(t.get("shape")).unwrap(), t.get("sha256").and_then(Value::as_str).unwrap().into()),
                )
            })
            .collect();
        let got = decoded(&dir, 2, &Decoder::Cpu).unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut seen = HashSet::new();
        for (t, shape, v) in &got {
            assert!(seen.insert(t.clone()), "{name}: {t} twice");
            assert_eq!(shape.iter().product::<u64>() as usize, v.len(), "{name}: {t}");
            if let Some((s, sha)) = want.get(t) {
                let bf: Vec<u16> = v.iter().map(|x| (x.to_bits() >> 16) as u16).collect();
                assert_eq!((s, &sha256_u16(&bf)), (shape, sha), "{name}: {t}");
            }
        }
        assert!(want.keys().all(|k| seen.contains(k)), "{name}: a packed tensor missing");
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
        let first = c.names().find(|n| n.contains(".glyd_") && n.ends_with("data")).unwrap_or_else(|| panic!("{name}: no pack")).to_string();
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
        // (a release's: vX.Y.Z-LAYOUT; v1-, v2- and v5- name the format of an earlier glyd's save, made by this one as that glyd wrote it)
        let release = name.split_once('-').is_some_and(|(v, _)| v.starts_with('v') && v.matches('.').count() == 2);
        let Some(layout) = LAYOUTS.iter().find(|(l, _)| release && name.ends_with(&format!("-{l}"))).map(|(_, l)| *l) else {
            continue;
        };
        let out = tmp.join(&name);
        save(&source, &out, 2, SHARD_BYTES, true, layout).unwrap_or_else(|e| panic!("{name}: {e}"));
        for f in std::fs::read_dir(&dir).unwrap() {
            let f = f.unwrap();
            let file = f.file_name().to_string_lossy().into_owned();
            if file == "glyd.json" {
                let (a, b) = (
                    json::parse(&std::fs::read_to_string(f.path()).unwrap()).unwrap(),
                    json::parse(&std::fs::read_to_string(out.join(&file)).unwrap()).unwrap(),
                );
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

/// Writes what the fixtures are made from: the tiny checkpoint to `$GLYD_PACK_SRC/src` (for the release's `glyd pack`; its four-bit-trained form to `qat`), and into tests/data/packs
/// the fixtures this glyd makes (glyd-v1, glyd-v2, glyd-v5 and the `main-*` ones) that are not there yet: a fixture is never written over.
#[test]
#[ignore]
fn write_pack_sources() {
    let dir = PathBuf::from(std::env::var("GLYD_PACK_SRC").expect("GLYD_PACK_SRC: a folder"));
    tiny_source(&dir.join("src"));
    tiny_source_as(&dir.join("qat"), true);
    let new = |name: &str| !Path::new(FIXTURES).join(name).join("glyd.json").exists();
    if new("v1-0.27.0-mma") {
        legacy(&Path::new(FIXTURES).join("v1-0.27.0-mma"));
    }
    if new("v2-0.27.0-mma") {
        legacy_experts(&Path::new(FIXTURES).join("v2-0.27.0-mma"));
    }
    if new("v5-0.29.1-mma") {
        tables(&dir);
    }
    main_fixtures(&dir, &new);
}

/// The plan of the tiny dense checkpoint as the block layouts' (q, k, v and gate, up merged, the embedding a linear too), with `layout: ROW` on its one-member linears (o_proj, down_proj) where
/// `row` names one: the weight's transpose a row at a time (a merged group cannot be).
fn plan_of(src: &Path, row: Option<&str>) -> String {
    let config = json::parse(&std::fs::read_to_string(src.join("config.json")).unwrap()).unwrap();
    let puts = Plan::new(&Checkpoint::open(src).unwrap(), &config, true, true).unwrap().puts;
    let items: Vec<String> = puts
        .iter()
        .map(|p| match p {
            Put::Linear { members, .. } => {
                let names = members.iter().map(|m| format!("\"{m}\"")).collect::<Vec<_>>().join(", ");
                if let Some(row) = row.filter(|_| members.len() == 1 && (members[0].ends_with("o_proj") || members[0].ends_with("down_proj"))) {
                    format!("{{\"linear\": [{names}], \"layout\": \"{row}\"}}")
                } else {
                    format!("{{\"linear\": [{names}]}}")
                }
            }
            Put::Copy { name, from, .. } => format!("{{\"copy\": \"{name}\", \"from\": \"{from}\"}}"),
            Put::Experts { .. } | Put::Codebook { .. } | Put::Signs { .. } => unreachable!(),
        })
        .collect();
    format!("{{\"puts\": [{}]}}", items.join(", "))
}

/// The glyd.json of a save with the temporary folders' names taken out (`source.repo` and the like name where this ran).
fn scrub(out: &Path, names: &[&Path]) {
    let path = out.join("glyd.json");
    let mut text = std::fs::read_to_string(&path).unwrap();
    for n in names {
        text = text.replace(n.to_str().unwrap(), "tmp");
    }
    std::fs::write(path, text).unwrap();
}

/// The `main-*` fixtures (each where `new` says it is not there yet): the tiny dense checkpoint with its o_proj and down_proj in each row layout (the save at the block layout of the same bits); a
/// Hummingbird save from a synthetic export of two experts (one at three bits, c3r; one at two, c2r) beside a dense linear at six bits; the tiny dense checkpoint at penguin as Gemma 4's
/// lossless repos are saved (glyd-v7 at `mma`, every pack read as saved: `save_penguin`); and its four-bit-trained form at the exact four-bit layout, as Gemma 4's
/// quantization-aware checkpoints are saved at wren (glyd-v7, `i4x`).
fn main_fixtures(tmp: &Path, new: &dyn Fn(&str) -> bool) {
    let src = tmp.join("src");
    let source = Source::find(src.to_str().unwrap()).unwrap();
    for (name, block) in [("i6r", Layout::Int6), ("i5r", Layout::Int5), ("i4r", Layout::Int4)] {
        let out = Path::new(FIXTURES).join(format!("main-{name}"));
        if !new(&format!("main-{name}")) {
            continue;
        }
        save_planned(&source, &out, 2, SHARD_BYTES, block, &plan_of(&src, Some(name))).unwrap();
        scrub(&out, &[&src, tmp]);
    }
    if new("main-penguin") {
        let out = Path::new(FIXTURES).join("main-penguin");
        save_penguin(&source, &out, 2, SHARD_BYTES, &plan_of(&src, None)).unwrap();
        scrub(&out, &[&src, tmp]);
    }
    if new("main-i4x") {
        let (qat, out) = (tmp.join("qat"), Path::new(FIXTURES).join("main-i4x"));
        save(&Source::find(qat.to_str().unwrap()).unwrap(), &out, 2, SHARD_BYTES, true, Layout::Int4H).unwrap();
        scrub(&out, &[&qat, tmp]);
    }
    if !new("main-hb") {
        return;
    }
    // the Hummingbird save: what a_save_at_the_hummingbird_level_from_an_export builds
    let (rows, cols, experts) = (64usize, 128usize, 2usize);
    let export = tmp.join("export");
    crate::hbexport::tests::synthetic(&export, 1, experts, rows, cols, &|_l, e| if e == 0 { 3 } else { 2 }, 77);
    let hsrc = tmp.join("hsrc");
    std::fs::create_dir_all(&hsrc).unwrap();
    std::fs::write(hsrc.join("config.json"), "{}").unwrap();
    let mut x = 99u64;
    let mut bf = |n: usize| -> Vec<u8> {
        (0..n)
            .flat_map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.1).to_bits() >> 16).to_le_bytes()[..2].to_vec()
            })
            .collect()
    };
    let m = "m.layers.1.mixer";
    let t = |name: String, shape: Vec<u64>, bytes: Vec<u8>| Saved { name, dtype: "BF16", shape, bytes, sha256: None };
    let mut ts = vec![t("m.fc.weight".into(), vec![64, 128], bf(64 * 128))];
    for e in 0..experts {
        ts.push(t(format!("{m}.experts.{e}.up_proj.weight"), vec![rows as u64, cols as u64], bf(rows * cols)));
        ts.push(t(format!("{m}.experts.{e}.down_proj.weight"), vec![cols as u64, rows as u64], bf(rows * cols)));
    }
    ts.push(t("m.norm.weight".into(), vec![64], bf(64)));
    write_safetensors(&hsrc.join("model.safetensors"), &mut ts).unwrap();
    let mut puts = vec!["{\"linear\": [\"m.fc\"], \"layout\": \"i6\"}".to_string()];
    for e in 0..experts {
        puts.push(format!("{{\"codebook\": \"{m}.experts.{e}.up_proj\", \"layer\": 0, \"expert\": {e}, \"matrix\": \"up\"}}"));
        puts.push(format!("{{\"codebook\": \"{m}.experts.{e}.down_proj\", \"layer\": 0, \"expert\": {e}, \"matrix\": \"downT\"}}"));
    }
    puts.push(format!("{{\"signs\": \"{m}.glyd_rot_signs\", \"layer\": 0}}"));
    puts.push("{\"copy\": \"m.norm.weight\"}".to_string());
    let plan = format!("{{\"puts\": [{}]}}", puts.join(", "));
    let out = Path::new(FIXTURES).join("main-hb");
    save_hummingbird(&Source::find(hsrc.to_str().unwrap()).unwrap(), &out, 2, SHARD_BYTES, &plan, &export).unwrap();
    scrub(&out, &[&hsrc, &export, tmp]);
}
