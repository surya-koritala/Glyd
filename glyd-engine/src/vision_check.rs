//! `glyd-engine visioncheck` (development): the vision tower kind (`runtime::kinds::VisionDesc`) on the executor, held to its CPU reference
//! (src/runtime/reference/vision.rs) and to the release's engine (v0.34.2's, which a GPU run builds beside this one).
//!
//!   visioncheck make DIR                     the release's shape with its vision tower at random: the expert cache's random case (`logits::qwen4x_cb_random`: Qwen3.8-Flash-Next at the
//!                                            release's widths, two layers, eight experts and their pack), every expert taken a token and every pool kept (no discrete choice for two
//!                                            engines to make apart), the release's vision tower and image token, preprocessor_config.json, a tokenizer and chat template with the image's
//!                                            tokens
//!   visioncheck tower DIR IMAGE... [--out D]  the tower's rows against the CPU reference: each stage on the engine's own input at its kinds' bound (the patches' product and positions and
//!                                            the merger 1e-2, a block 2e-2: kindcheck's rule, the largest difference over the reference's largest magnitude), the whole tower and the
//!                                            reference's own noise floor (its rounding points against f32) printed; then a second random tower of other shapes; D: each image's rows,
//!                                            patches and grid as the release's `qxvision` writes them
//!   visioncheck logits DIR IMAGE --out P     the executor's logits of an image prompt, teacher-forced as the release's `qwenserve` scores (`score_from`), of the same text with no image,
//!                                            and of the image prompt with its rows turned at their own positions (no M-RoPE), as f32 rows in P.image.f32, P.text.f32 and P.plain.f32;
//!                                            P.requests.jsonl: the release's requests for the first two
//!   visioncheck compare A B --vocab V [--bound bits|kl:MEAN:MAX:TOP1] [--expect-apart]   two files of logit rows (A the one held to): the rows the same to the bit, KL, the best
//!                                            token, at the bound (the kind's: kl:1e-4:2.5e-4:0.85); --expect-apart: a pass where they are past it (a check that must tell them apart)
//!   visioncheck sched DIR IMAGE IMAGE [--tokens N]   prompts with and without images through the common scheduler at once, each against itself alone (`conv::Gen::reply`), and a second
//!                                            turn about an image already read; GLYD_ENGINE_BATCH_PREFILL=0 holds them to the bit

use crate::args::Args;
use crate::dev::Dev;
use crate::ffi::{err, Error, Result};
use crate::kern::Kern;
use crate::runtime::describe::VisionRoles;
use crate::runtime::kinds::{Act, Rounding, VisionDesc};
use crate::runtime::reference::vision::RefTower;
use crate::vision::{Image, Prep, Tower};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The image's markers and its place in the random model's tokenizer (after `logits::tiny_tokens`' 105).
const VISION_TOKENS: [&str; 3] = ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>"];
const IMAGE_TOKEN: u32 = 107;

pub fn run(a: &Args) -> Result<()> {
    match a.need(1, "make|tower|logits|compare|sched")? {
        "make" => make(Path::new(a.need(2, "DIR")?)),
        "tower" => tower(a),
        "logits" => logits(a),
        "compare" => compare(a),
        "sched" => sched(a),
        x => err(format!("visioncheck {x}: make, tower, logits, compare or sched")),
    }
}

fn io<T>(r: std::io::Result<T>, what: &Path) -> Result<T> {
    r.map_err(|e| Error(format!("{}: {e}", what.display())))
}

/// A seeded stream of numbers: uniform in [0, 1) and Gaussian.
struct Draw(u64, Option<f64>);

impl Draw {
    fn u(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn g(&mut self) -> f64 {
        if let Some(x) = self.1.take() {
            return x;
        }
        let (a, b) = (self.u(), self.u());
        let r = (-2.0 * a.ln()).sqrt();
        self.1 = Some(r * (std::f64::consts::TAU * b).sin());
        r * (std::f64::consts::TAU * b).cos()
    }
}

fn bf(x: f32) -> u16 {
    let b = x.to_bits();
    ((b.wrapping_add(0x7fff + ((b >> 16) & 1))) >> 16) as u16
}

/// A tower's tensors at random (`seed`), each at the shape the description says: the norms' weights about 1, the biases small, the position table about 0.5,
/// every product's weights of a deviation 1 / sqrt(its inputs). Written to `file`; their names returned.
fn tower_tensors(roles: &VisionRoles, shapes: &std::collections::BTreeMap<String, Vec<usize>>, file: &Path, seed: u64) -> Result<Vec<String>> {
    let mut d = Draw(seed, None);
    let mut tensors: Vec<(String, Vec<u64>, Vec<u8>)> = Vec::new();
    for (_, src) in roles.own.iter().chain(roles.blocks.iter().flatten()) {
        for name in src.names() {
            let shape = shapes.get(&name).ok_or_else(|| Error(format!("no shape for {name}")))?.clone();
            let n: usize = shape.iter().product();
            let ins: usize = shape[1..].iter().product::<usize>().max(1);
            let leaf = name.rsplit('.').next().unwrap_or("");
            let norm = name.contains("norm");
            let vals: Vec<f32> = (0..n)
                .map(|_| {
                    (if norm && leaf == "weight" {
                        1.0 + 0.2 * (d.u() - 0.5)
                    } else if leaf == "bias" {
                        0.02 * d.g()
                    } else if name.contains("pos_embed") {
                        0.5 * d.g()
                    } else {
                        d.g() / (ins as f64).sqrt()
                    }) as f32
                })
                .collect();
            tensors.push((name, shape.iter().map(|&x| x as u64).collect(), vals.iter().flat_map(|&x| bf(x).to_le_bytes()).collect()));
        }
    }
    let refs: Vec<(&str, &str, Vec<u64>, &[u8])> = tensors.iter().map(|(n, s, b)| (n.as_str(), "BF16", s.clone(), &b[..])).collect();
    crate::pack::write_safetensors(file, &[("format", "pt".to_string())], &refs)?;
    Ok(tensors.into_iter().map(|t| t.0).collect())
}

/// The processor's settings the release's checkpoint has (patches of 16 in blocks of 2 x 2, two frames, mean and deviation 0.5).
const PREPROCESSOR: &str = r#"{"do_convert_rgb": true, "do_normalize": true, "do_rescale": true, "do_resize": true, "image_mean": [0.5, 0.5, 0.5], "image_processor_type": "Qwen2VLImageProcessorFast", "image_std": [0.5, 0.5, 0.5], "merge_size": 2, "patch_size": 16, "resample": 3, "rescale_factor": 0.00392156862745098, "size": {"longest_edge": 16777216, "shortest_edge": 65536}, "temporal_patch_size": 2}"#;

/// `make`: see the module's words.
fn make(dir: &Path) -> Result<()> {
    crate::logits::qwen4x_cb_random(dir)?;
    let p = dir.join("config.json");
    let mut c: Value = serde_json::from_str(&io(std::fs::read_to_string(&p), &p)?).map_err(|e| Error(e.to_string()))?;
    let rel: Value = serde_json::from_str(include_str!("../tests/fixtures/qwen4x_config.json")).map_err(|e| Error(e.to_string()))?;
    // (every expert a token and every pool kept: the two engines make no choice apart, so their logits are held to the kind's bound)
    let t = &mut c["text_config"];
    t["num_experts_per_tok"] = t["num_experts"].clone();
    t["indexer_budget"] = json!(2048);
    c["vision_config"] = rel["vision_config"].clone();
    for (k, v) in [("image_token_id", IMAGE_TOKEN), ("vision_start_token_id", 105), ("vision_end_token_id", 106), ("video_token_id", 108)] {
        c[k] = json!(v);
    }
    io(std::fs::write(&p, serde_json::to_string_pretty(&c).map_err(|e| Error(e.to_string()))?), &p)?;
    // (the routed experts bf16 in the checkpoint, as a release holds them: the loader of experts held whole (`glyd run`, `serve`) reads bf16; the pack, which the
    // expert cache and the release's engine read, is as made)
    let ef = dir.join("experts.safetensors");
    let ck = glyd_gpu::safetensors::Checkpoint::open(dir).map_err(|e| Error(e.to_string()))?;
    let mut experts: Vec<(String, Vec<u64>, Vec<u8>)> = Vec::new();
    let ip = dir.join("model.safetensors.index.json");
    let index: Value = serde_json::from_str(&io(std::fs::read_to_string(&ip), &ip)?).map_err(|e| Error(e.to_string()))?;
    for (n, f) in index["weight_map"].as_object().ok_or_else(|| Error("an index with no weight map".into()))? {
        if f.as_str() == Some("experts.safetensors") {
            let (shape, vals) = crate::runtime::reference::model::checkpoint_tensor(&ck, n).ok_or_else(|| Error(format!("{n}: not read")))?;
            experts.push((n.clone(), shape.iter().map(|&x| x as u64).collect(), vals.iter().flat_map(|&x| bf(x).to_le_bytes()).collect()));
        }
    }
    drop(ck);
    let refs: Vec<(&str, &str, Vec<u64>, &[u8])> = experts.iter().map(|(n, s, b)| (n.as_str(), "BF16", s.clone(), &b[..])).collect();
    crate::pack::write_safetensors(&ef, &[("format", "pt".to_string())], &refs)?;
    let d = crate::runtime::describe::describe(&c)?;
    let roles = d.vision.clone().ok_or_else(|| Error("the random model's description has no vision tower".into()))?;
    let names = tower_tensors(&roles, &d.shapes, &dir.join("vision.safetensors"), 0x2026_1125_0000_0b4c)?;
    let mut index = index;
    for n in &names {
        index["weight_map"][n] = json!("vision.safetensors");
    }
    io(std::fs::write(&ip, index.to_string()), &ip)?;
    io(std::fs::write(dir.join("preprocessor_config.json"), PREPROCESSOR), dir)?;
    crate::logits::write_serving_files_with(dir, include_str!("../tests/fixtures/chat_qwen38_tools.jinja"), &VISION_TOKENS)?;
    let (v, _) = crate::vision::described(dir)?.ok_or_else(|| Error("no vision tower read back".into()))?;
    println!(
        "{}: the release's shape's random model with its vision tower ({} blocks of {} heads over {}, rows of {}; image token {}), {} tower tensors",
        dir.display(),
        v.blocks,
        v.heads,
        v.hidden,
        v.out,
        v.image_token,
        names.len()
    );
    Ok(())
}

/// The largest difference of `got` (bf16) from `want` over `want`'s largest magnitude.
fn far(got: &[u16], want: &[f32]) -> f32 {
    let scale = want.iter().fold(0f32, |m, x| m.max(x.abs())).max(f32::MIN_POSITIVE);
    got.iter().zip(want).map(|(&g, &w)| (f32::from_bits((g as u32) << 16) - w).abs()).fold(0f32, f32::max) / scale
}

fn far_f(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(0f32, |m, x| m.max(x.abs())).max(f32::MIN_POSITIVE);
    got.iter().zip(want).map(|(g, w)| (g - w).abs()).fold(0f32, f32::max) / scale
}

fn f32s(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&x| f32::from_bits((x as u32) << 16)).collect()
}

/// The kinds' bounds (src/runtime/reference/mod.rs, kindcheck's table): a product with its activation (the MLP kind), and a block of attention and an MLP with their
/// residuals and norms (the attention and residual kinds').
const PRODUCT_BOUND: f32 = 1e-2;
const BLOCK_BOUND: f32 = 2e-2;

/// One tower held to its reference on `images`: true where every stage is within its bound.
fn tower_case(what: &str, dir: &Path, v: &VisionDesc, roles: &VisionRoles, images: &[String], out: Option<&Path>) -> Result<bool> {
    let dev = Dev::new(0)?;
    let kern = Kern::load(&dev)?;
    let tower = Tower::open(dir, v, roles, &dev)?;
    let ck = glyd_gpu::safetensors::Checkpoint::open(dir).map_err(|e| Error(e.to_string()))?;
    let reference = RefTower::load(v, roles, &|n| crate::runtime::reference::model::checkpoint_tensor(&ck, n))?;
    let prep = Prep { min_pixels: 65536, max_pixels: 1 << 20, image_token: v.image_token, patch: v.patch, merge: v.merge, frames: v.frames };
    println!(
        "{what}: {} blocks of {} heads of {} ({}), {} neurons, a table of {} a side, rows of {}",
        v.blocks,
        v.heads,
        v.head_dim(),
        v.hidden,
        v.inter,
        v.table,
        v.out
    );
    let mut ok = true;
    for path in images {
        let bytes = io(std::fs::read(path), Path::new(path))?;
        let img = prep.image(&bytes)?;
        let mut stages: Vec<Vec<u16>> = Vec::new();
        let t0 = std::time::Instant::now();
        let rows = tower.run_seen(
            &dev,
            &kern,
            &img,
            Some(&mut |_, r| {
                stages.push(r);
                Ok(())
            }),
        )?;
        let gpu_ms = t0.elapsed().as_secs_f64() * 1e3;
        let px = f32s(&img.patches);
        let t1 = std::time::Instant::now();
        let once = reference.run(&px, img.gh, img.gw, Rounding::Once)?;
        let f = reference.run(&px, img.gh, img.gw, Rounding::F32)?;
        let ref_ms = t1.elapsed().as_secs_f64() * 1e3;
        // each stage on the engine's own input
        let mut worst = (0f32, String::new());
        let mut judge = |name: String, x: f32, bound: f32| {
            if x > bound {
                ok = false;
                println!("FAIL {what} {path}: {name}: {x:.3e} past its bound {bound:.0e}");
            }
            if x / bound > worst.0 {
                worst = (x / bound, format!("{name} {x:.3e} (bound {bound:.0e})"));
            }
        };
        judge("the patches' product and positions".into(), far(&stages[0], &once.embedded), PRODUCT_BOUND);
        for l in 0..v.blocks {
            let want = reference.block_at(l, &f32s(&stages[l]), img.gh, img.gw, Rounding::Once)?;
            judge(format!("block {l}"), far(&stages[l + 1], &want), BLOCK_BOUND);
        }
        judge("the merger".into(), far(&rows, &reference.merge_at(&f32s(&stages[v.blocks]), Rounding::Once)), PRODUCT_BOUND);
        let (whole, floor) = (far(&rows, &once.out), far_f(&once.out, &f.out));
        let bits = rows.iter().zip(&once.out).filter(|(&g, &w)| g == bf(w)).count();
        println!(
            "{what} {path}: {} x {} patches, {} tokens; each stage on the engine's own input within its bound, the nearest {:.2} of it ({}); the whole tower {whole:.3e} from the reference ({bits} of {} values the same bits), the reference's own noise floor (its rounding points against f32) {floor:.3e}; the tower {gpu_ms:.0} ms, the reference {ref_ms:.0} ms",
            img.gh,
            img.gw,
            img.tokens(),
            worst.0,
            worst.1,
            rows.len()
        );
        if let Some(out) = out {
            let name = Path::new(path).file_stem().map_or("image".into(), |s| s.to_string_lossy().to_string());
            let bytes_of = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
            io(std::fs::create_dir_all(out), out)?;
            io(std::fs::write(out.join(format!("{name}.patches.bf16")), bytes_of(&img.patches)), out)?;
            io(std::fs::write(out.join(format!("{name}.rows.bf16")), bytes_of(&rows)), out)?;
            io(std::fs::write(out.join(format!("{name}.json")), format!("{{\"grid_thw\":[1,{},{}],\"tokens\":{}}}\n", img.gh, img.gw, img.tokens())), out)?;
        }
    }
    Ok(ok)
}

/// `tower`: see the module's words.
fn tower(a: &Args) -> Result<()> {
    let dir = Path::new(a.need(2, "DIR")?);
    let images: Vec<String> = a.pos.iter().skip(3).cloned().collect();
    if images.is_empty() {
        return err("visioncheck tower DIR IMAGE...: no image");
    }
    let (v, roles) = crate::vision::described(dir)?.ok_or_else(|| Error(format!("{}: no vision tower", dir.display())))?;
    let first = tower_case("the release's tower", dir, &v, &roles, &images, a.flag("out").map(Path::new))?;
    // (a tower of other shapes: four heads of 64, three blocks, 640 neurons, a table of 16 a side, rows of 512, the MLP's GELU erf's: no shape of the kernels' is the release's alone)
    let v2 = VisionDesc { blocks: 3, hidden: 256, heads: 4, inter: 640, act: Act::Gelu, table: 16, out: 512, image_token: 0, ..v.clone() };
    let tmp = std::env::temp_dir().join(format!("glyd-visioncheck-{}", std::process::id()));
    io(std::fs::create_dir_all(&tmp), &tmp)?;
    let at = crate::runtime::names::part::VISION_AT;
    let roles2 = VisionRoles {
        own: crate::runtime::names::roles(crate::runtime::names::part::VISION, at, 0, crate::runtime::names::all),
        blocks: (0..v2.blocks)
            .map(|l| crate::runtime::names::roles(crate::runtime::names::part::VISION_BLOCK, &format!("{at}.blocks.{l}"), 0, crate::runtime::names::all))
            .collect(),
    };
    let (h, m) = (v2.hidden, v2.hidden * v2.merge * v2.merge);
    let mut shapes = std::collections::BTreeMap::new();
    for (r, s) in [
        ("patch", vec![h, v2.channels, v2.frames, v2.patch, v2.patch]),
        ("patch_bias", vec![h]),
        ("pos", vec![v2.table * v2.table, h]),
        ("merge_norm", vec![h]),
        ("merge_norm_bias", vec![h]),
        ("merge_up", vec![m, m]),
        ("merge_up_bias", vec![m]),
        ("merge_down", vec![v2.out, m]),
        ("merge_down_bias", vec![v2.out]),
    ] {
        shapes.insert(roles2.tensor(None, r)?.to_string(), s);
    }
    for l in 0..v2.blocks {
        for (r, s) in [
            ("norm1", vec![h]),
            ("norm1_bias", vec![h]),
            ("norm2", vec![h]),
            ("norm2_bias", vec![h]),
            ("qkv", vec![3 * h, h]),
            ("qkv_bias", vec![3 * h]),
            ("o", vec![h, h]),
            ("o_bias", vec![h]),
            ("up", vec![v2.inter, h]),
            ("up_bias", vec![v2.inter]),
            ("down", vec![h, v2.inter]),
            ("down_bias", vec![h]),
        ] {
            shapes.insert(roles2.tensor(Some(l), r)?.to_string(), s);
        }
    }
    tower_tensors(&roles2, &shapes, &tmp.join("model.safetensors"), 0x2026_1125_0000_0b4d)?;
    let second = tower_case("a tower of other shapes", &tmp, &v2, &roles2, &images[..1], None);
    std::fs::remove_dir_all(&tmp).ok();
    if !(first && second?) {
        return err("visioncheck tower: a stage past its bound");
    }
    println!("PASS visioncheck tower");
    Ok(())
}

/// The executor on the model in `dir` as its server loads it (src/dynamic/qwen4x.rs `load_cached`): the matrices lossless, its experts from its pack with a place for
/// each (none computed by the host), up to `users` sequences.
fn engine(dir: &Path, users: usize) -> Result<crate::dynamic::Dyn> {
    use crate::dynamic::qwen4x::{load_cached, lowered, Caching, PackMode};
    let o = crate::dynamic::Opts { ctx: 4096, chunk: 128, graph: true, device: 0, layout: None };
    let k = Caching::new(&dir.join("pack"), PackMode::Best, &lowered(dir, o.ctx + o.chunk + 16)?, users);
    let e = load_cached(dir, &o, &k)?;
    if !e.sees() {
        return err(format!("{}: the executor opened no vision tower", dir.display()));
    }
    Ok(e)
}

/// The prompt of an image and a question and an answer to be scored, its ids with the image's one place, and where the answer begins; with no image, the same text.
fn prompt_ids(dir: &Path, image: bool) -> Result<(Vec<u32>, usize)> {
    let tok = crate::tok::Tok::load(dir)?;
    let place = if image { crate::tools::IMAGE_PLACE } else { "" };
    let ask = format!("<|im_start|>user\n{place}What is in this picture?<|im_end|>\n<|im_start|>assistant\n");
    let ids = tok.encode_with(&format!("{ask}It is a red square on white."), false)?;
    Ok((ids, tok.encode_with(&ask, false)?.len()))
}

/// The engine's logits of `ids` from `from` on, teacher-forced (the release's `score_from`): the row after `ids[..from]`, then after each id up to the last but one.
fn scored(e: &mut crate::dynamic::Dyn, ids: &[u32], images: &[Arc<Image>], from: usize) -> Result<Vec<Vec<f32>>> {
    e.reset()?;
    e.set_images(ids, images)?;
    let mut rows = Vec::new();
    e.prefill(&ids[..from])?;
    rows.push(e.logits()?);
    for &t in &ids[from..ids.len() - 1] {
        e.step(t)?;
        rows.push(e.logits()?);
    }
    Ok(rows)
}

fn write_rows(path: &Path, rows: &[Vec<f32>]) -> Result<()> {
    io(std::fs::write(path, rows.iter().flatten().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()), path)
}

/// `logits`: see the module's words.
fn logits(a: &Args) -> Result<()> {
    let dir = Path::new(a.need(2, "DIR")?);
    let image = a.need(3, "IMAGE")?;
    let out = PathBuf::from(a.flag("out").ok_or_else(|| Error("visioncheck logits: --out PREFIX".into()))?);
    let mut e = engine(dir, 1)?;
    let prep = crate::vision::PREP.get().cloned().ok_or_else(|| Error("no image settings".into()))?;
    let img = Arc::new(prep.image(&io(std::fs::read(image), Path::new(image))?)?);
    let (ids, from) = prompt_ids(dir, true)?;
    let expanded = crate::vision::expand(&ids, prep.image_token, std::slice::from_ref(&img))?;
    let from_x = from - 1 + img.tokens();
    let (text, from_t) = prompt_ids(dir, false)?;
    let name = |what: &str| PathBuf::from(format!("{}.{what}", out.display()));
    // the release's requests: the image prompt with its one place (the release's engine makes it as long as the image's tokens), the text alone
    let req = |ids: &[u32], images: &[&str], from: usize, dump: &Path| {
        json!({"ids": ids, "images": images, "score_from": from, "dump": dump.display().to_string()}).to_string()
    };
    let lines = [req(&ids, &[image], from_x, &name("old.image.f32")), req(&text, &[], from_t, &name("old.text.f32"))];
    io(std::fs::write(name("requests.jsonl"), lines.join("\n") + "\n"), &out)?;
    let t0 = std::time::Instant::now();
    let rows = scored(&mut e, &expanded, std::slice::from_ref(&img), from_x)?;
    write_rows(&name("image.f32"), &rows)?;
    println!(
        "the image prompt: {} ids ({} the image's: {} x {} patches), {} rows scored from {from_x} in {:.0} ms",
        expanded.len(),
        img.tokens(),
        img.gh,
        img.gw,
        rows.len(),
        t0.elapsed().as_secs_f64() * 1e3
    );
    write_rows(&name("text.f32"), &scored(&mut e, &text, &[], from_t)?)?;
    // (the image prompt with every row at its own position: the M-RoPE positions off, what the comparison with the release's engine must tell apart)
    e.plain_positions(true);
    write_rows(&name("plain.f32"), &scored(&mut e, &expanded, &[img], from_x)?)?;
    e.plain_positions(false);
    println!("vocabulary {}: {}.{{image,text,plain}}.f32 and the release's requests in {}", e.desc().vocab, out.display(), name("requests.jsonl").display());
    Ok(())
}

fn read_rows(path: &Path, vocab: usize) -> Result<Vec<Vec<f32>>> {
    let b = io(std::fs::read(path), path)?;
    if b.is_empty() || b.len() % (4 * vocab) != 0 {
        return err(format!("{}: {} bytes are not rows of {vocab} f32", path.display(), b.len()));
    }
    Ok(b.chunks(4 * vocab).map(|r| r.chunks(4).map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]])).collect()).collect())
}

/// `compare`: see the module's words.
fn compare(a: &Args) -> Result<()> {
    use crate::logits::{kl_of, Row, CLEAR_LEAD};
    let (pa, pb) = (Path::new(a.need(2, "A")?), Path::new(a.need(3, "B")?));
    let vocab = a.num("vocab", 0usize)?;
    let (want, got) = (read_rows(pa, vocab)?, read_rows(pb, vocab)?);
    if want.len() != got.len() {
        return err(format!("{} rows in {}, {} in {}", want.len(), pa.display(), got.len(), pb.display()));
    }
    let (mut same, mut sum, mut max, mut top, mut clear, mut dmax) = (0usize, 0f64, 0f64, 0usize, 0usize, 0f32);
    for (w, g) in want.iter().zip(&got) {
        same += w.iter().zip(g).all(|(x, y)| x.to_bits() == y.to_bits()) as usize;
        let row = Row::from_logits(w)?;
        let kl = kl_of(&row, g);
        (sum, max) = (sum + kl, max.max(kl));
        dmax = row.ids.iter().fold(dmax, |m, &i| m.max((w[i as usize] - g[i as usize]).abs()));
        if row.logits.len() > 1 && row.logits[0] - row.logits[1] > CLEAR_LEAD {
            clear += 1;
            let best = g.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0;
            top += (best == row.ids[0] as usize) as usize;
        }
    }
    let n = want.len();
    let mean = sum / n as f64;
    let bound = a.flag("bound").unwrap_or("kl:1e-4:2.5e-4:0.85");
    let held = match bound.split(':').collect::<Vec<_>>().as_slice() {
        ["bits"] => same == n,
        ["kl", m, x, t] => {
            let f = |v: &str| v.parse::<f64>().map_err(|_| Error(format!("--bound {bound}")));
            mean <= f(m)? && max <= f(x)? && top as f64 >= (clear as f64 * f(t)?).ceil()
        }
        _ => return err(format!("--bound {bound}: bits or kl:MEAN:MAX:TOP1")),
    };
    let apart = a.has("expect-apart");
    let pass = held != apart;
    println!(
        "{} {} against {}: {n} rows, {same} the same to the bit; KL mean {mean:.3e} max {max:.3e}; the best token the same at {top} of {clear} rows with a clear lead; the largest logit difference among its top {:.4}; {} the bound {bound}{}",
        if pass { "PASS" } else { "FAIL" },
        pb.display(),
        pa.display(),
        dmax,
        if held { "within" } else { "past" },
        if apart { " (expected past: a check that tells them apart)" } else { "" }
    );
    if !pass {
        return err("visioncheck compare: not as expected");
    }
    Ok(())
}

/// `sched`: see the module's words.
fn sched(a: &Args) -> Result<()> {
    use crate::conv::Gen;
    use crate::sched::{Ev, Finish, Reply, Sched, Submit};
    let dir = Path::new(a.need(2, "DIR")?);
    let (ia, ib) = (a.need(3, "IMAGE")?, a.need(4, "IMAGE")?);
    let tokens = a.num("tokens", 12usize)?;
    let tok = crate::tok::Tok::load(dir)?;
    let mut e = engine(dir, 4)?;
    let prep = crate::vision::PREP.get().cloned().ok_or_else(|| Error("no image settings".into()))?;
    let read = |p: &str| -> Result<Arc<Image>> { Ok(Arc::new(prep.image(&io(std::fs::read(p), Path::new(p))?)?)) };
    let (a_img, b_img) = (read(ia)?, read(ib)?);
    let place = crate::tools::IMAGE_PLACE;
    let ids = |text: &str, imgs: &[Arc<Image>]| -> Result<Vec<u32>> { crate::vision::expand(&tok.encode_with(text, false)?, prep.image_token, imgs) };
    let turn = |q: &str| format!("<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n");
    let prompts: Vec<(String, Vec<u32>, Vec<Arc<Image>>)> = vec![
        ("an image".into(), ids(&turn(&format!("{place}What is this?")), std::slice::from_ref(&a_img))?, vec![a_img.clone()]),
        ("another image".into(), ids(&turn(&format!("Describe {place} briefly.")), std::slice::from_ref(&b_img))?, vec![b_img.clone()]),
        (
            "two images".into(),
            ids(&turn(&format!("First {place} then {place}: which is red?")), &[b_img.clone(), a_img.clone()])?,
            vec![b_img.clone(), a_img.clone()],
        ),
        ("text alone".into(), ids(&turn("What is a picture?"), &[])?, Vec::new()),
    ];
    let reply = Reply {
        params: crate::sampler::Params { temperature: 0.0, ..Default::default() },
        max_new: tokens,
        ignore_eos: true,
        logprobs: None,
        stop_ids: Vec::new(),
    };
    // each alone, on the engine's own sequence (`conv::Gen::reply`)
    let mut alone = Vec::new();
    for (_, p, imgs) in &prompts {
        e.see(imgs.clone())?;
        alone.push(Gen::reply(&mut e, p, &reply, &mut |_| true)?.0);
    }
    // a second turn about the first image (its first turn, its reply, a question): the scheduler goes on in the first's kept slot
    let second = [prompts[0].1.clone(), alone[0].clone(), ids(&format!("<|im_end|>\n{}", turn("And its color?")), &[])?].concat();
    e.see(vec![a_img.clone()])?;
    let alone_second = Gen::reply(&mut e, &second, &reply, &mut |_| true)?.0;
    drop(e);
    let mut e = engine(dir, 4)?;
    let Some((n, step)) = Gen::as_step(&mut e, 4)? else { return err("visioncheck sched: the engine is not a scheduler's step") };
    let mut s = Sched::new(step, n);
    fn submit<B: crate::sched::ModelStep>(
        s: &mut Sched<B>,
        reply: &Reply,
        p: &[u32],
        imgs: &[Arc<Image>],
    ) -> (std::sync::mpsc::Receiver<Ev>, Arc<std::sync::atomic::AtomicU32>) {
        let (tx, rx) = std::sync::mpsc::channel();
        let cached = Arc::new(std::sync::atomic::AtomicU32::new(0));
        s.submit(Submit {
            prompt: p.to_vec(),
            reply: reply.clone(),
            out: tx,
            abort: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            resumed: Default::default(),
            cached: cached.clone(),
            created: Default::default(),
            salt: 0,
            pin: 0,
            images: imgs.to_vec(),
        });
        (rx, cached)
    }
    let collect = |rx: std::sync::mpsc::Receiver<Ev>| -> Result<Vec<u32>> {
        let mut out = Vec::new();
        while let Ok(ev) = rx.recv() {
            match ev {
                Ev::Token(t) => out.push(t),
                Ev::Lp(_) => {}
                Ev::End(Ok(Finish::Length)) => break,
                Ev::End(f) => return err(format!("a sequence ended {f:?}")),
            }
        }
        Ok(out)
    };
    let waits: Vec<_> = prompts.iter().map(|(_, p, imgs)| submit(&mut s, &reply, p, imgs)).collect();
    while !s.idle() {
        s.tick()?;
    }
    let mut ok = true;
    for ((what, p, _), ((rx, _), want)) in prompts.iter().zip(waits.into_iter().zip(&alone)) {
        let got = collect(rx)?;
        let same = got == *want;
        ok &= same;
        println!(
            "{} {what} ({} ids) through the scheduler beside the others: its {tokens} tokens {} its own alone{}",
            if same { "PASS" } else { "FAIL" },
            p.len(),
            if same { "are" } else { "are NOT" },
            if same { String::new() } else { format!(" (here {got:?}, alone {want:?})") }
        );
    }
    let (rx, cached) = submit(&mut s, &reply, &second, std::slice::from_ref(&a_img));
    while !s.idle() {
        s.tick()?;
    }
    let got = collect(rx)?;
    let found = cached.load(std::sync::atomic::Ordering::Relaxed);
    let same = got == alone_second;
    ok &= found > 0;
    println!(
        "{} a second turn about the first image ({} ids): {found} of them found in the first's kept slot; its tokens {} the whole prompt's alone (not judged: the kept slot goes on from the first prompt's end, the prompt alone in chunks from its start){}",
        if found > 0 { "PASS" } else { "FAIL" },
        second.len(),
        if same { "are" } else { "are not" },
        if same { String::new() } else { format!(" (here {got:?}, alone {alone_second:?})") }
    );
    if !ok {
        return err("visioncheck sched: a sequence through the scheduler is not its own alone");
    }
    println!("PASS visioncheck sched");
    Ok(())
}
