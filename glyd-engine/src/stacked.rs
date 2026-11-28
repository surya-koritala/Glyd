//! Experts stacked in one tensor a matrix ([experts, rows, cols]: the form transformers holds a layer's experts in, and the one Qwen3.8-2.4T-A95B is
//! released in), as a source the Moe kind reads like a tensor an expert's matrix (#333): the description held to a checkpoint's index names the stacked
//! tensors (`describe::describe_held`), and every reader takes expert e's rows of one by their range of the file alone (src/dynamic/experts.rs `Stacked`:
//! the kind's block resident and in the expert cache; the lossy route's packer, `pack::plan`; the hummingbird packer, src/hb_pack.rs). Here:
//! a checkpoint written again in the stacked form (`restack`: the random maker's `--stacked`) and the check that the two forms are one model
//! (`check`: `glyd-engine stackcheck`).

use crate::ffi::{err, Error, Result};
use crate::runtime::describe::{stacked_rows, Described};
use glyd_gpu::safetensors::Checkpoint;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn io<T>(r: std::io::Result<T>) -> Result<T> {
    r.map_err(|e| Error(e.to_string()))
}

/// The checkpoint in `src` (each expert's matrices a tensor; `named` its description) written to `out` in the form `held` describes (the same model's
/// description held to a checkpoint that stacks its experts, `describe::describe_held`): every tensor `held` reads, each role's experts stacked as one tensor
/// [experts, rows, cols] whose expert e's rows are its matrices' (`describe::stacked_rows`), every other tensor its bytes as they are; and `src`'s other files
/// (config, tokenizer, templates) copied. The same weights to the bit; a tensor of `src` that neither form reads, or a stacked tensor's row no matrix fills,
/// is refused.
pub fn restack(src: &Path, out: &Path, named: &Described, held: &Described) -> Result<()> {
    let ck = io(Checkpoint::open(src))?;
    let rows = stacked_rows(named, held)?;
    if rows.is_empty() {
        return err(format!("{}: no experts the stacked form holds otherwise", src.display()));
    }
    let mut parts: BTreeMap<&str, Vec<(&str, usize, usize)>> = BTreeMap::new();
    for (m, (t, a, b)) in &rows {
        parts.entry(t.as_str()).or_default().push((m.as_str(), *a, *b));
    }
    let names = held.names();
    if let Some(n) = ck.names().find(|n| !names.contains(*n) && !rows.contains_key(*n)) {
        return err(format!("{}: {n}, which neither form of the model reads", src.display()));
    }
    let mut tensors: Vec<(String, String, Vec<u64>, Vec<u8>)> = Vec::new();
    for n in &names {
        match parts.get(n.as_str()) {
            Some(ps) => {
                let shape = held.shapes.get(n).ok_or_else(|| Error(format!("the description has no shape for {n}")))?;
                let [e, r, c] = shape[..] else { return err(format!("{n}: {shape:?}, where experts stacked are three-dimensional")) };
                let mut bytes = vec![0u8; e * r * c * 2];
                let mut filled = 0;
                for &(m, a, b) in ps {
                    let (f, t) = ck.get(m).ok_or_else(|| Error(format!("{}: no tensor {m}", src.display())))?;
                    if t.dtype != "BF16" || t.shape != [(b - a) as u64, c as u64] {
                        return err(format!("{m}: {} {:?}, where bf16 rows {a}..{b} of {n}'s {c} columns are", t.dtype, t.shape));
                    }
                    io(f.read_into(t, &mut bytes[a * c * 2..b * c * 2]))?;
                    filled += b - a;
                }
                if filled != e * r {
                    return err(format!("{n}: {filled} of its {} rows held by the experts' matrices", e * r));
                }
                tensors.push((n.clone(), "BF16".into(), shape.iter().map(|&x| x as u64).collect(), bytes));
            }
            None => {
                let (f, t) = ck.get(n).ok_or_else(|| Error(format!("{}: no tensor {n}", src.display())))?;
                tensors.push((n.clone(), t.dtype.clone(), t.shape.clone(), io(f.read(t))?));
            }
        }
    }
    io(std::fs::create_dir_all(out))?;
    let view: Vec<(&str, &str, Vec<u64>, &[u8])> = tensors.iter().map(|(n, d, s, b)| (n.as_str(), d.as_str(), s.clone(), &b[..])).collect();
    crate::pack::write_safetensors(&out.join("model.safetensors"), &[("format", "pt".to_string())], &view)?;
    for e in io(std::fs::read_dir(src))? {
        let p = io(e)?.path();
        let name = p.file_name().unwrap_or_default().to_string_lossy().to_string();
        if p.is_file() && !name.ends_with(".safetensors") && name != "model.safetensors.index.json" {
            io(std::fs::copy(&p, out.join(&name)))?;
        }
    }
    Ok(())
}

/// Each file of `dir` (its name and bytes), the directories in it none.
fn files_of(dir: &Path) -> Result<BTreeMap<String, Vec<u8>>> {
    let mut out = BTreeMap::new();
    for e in io(std::fs::read_dir(dir))? {
        let p = io(e)?.path();
        if p.is_file() {
            out.insert(p.file_name().unwrap_or_default().to_string_lossy().to_string(), io(std::fs::read(&p))?);
        }
    }
    Ok(out)
}

/// Whether the saves `a` and `b` are the same directory: every file's name and bytes (glyd.json, the shards, the index, the copied files), but for the
/// source each glyd.json names (`sources`: `a`'s, read as `b`'s there). None where they are; else the first file that differs, in words.
pub fn same_files(a: &Path, b: &Path, sources: (&str, &str)) -> Result<Option<String>> {
    use glyd_gpu::json::{to_compact, to_python, Value};
    let (mut fa, fb) = (files_of(a)?, files_of(b)?);
    if let Some(g) = fa.get_mut("glyd.json") {
        // (each source as glyd.json quotes it, by either of its writers: a Windows path's backslashes escaped)
        let quoted = |s: &str| [to_compact(&Value::from(s)), to_python(&Value::from(s), 0)];
        let mut text = String::from_utf8_lossy(g).into_owned();
        for (from, to) in quoted(sources.0).iter().zip(quoted(sources.1).iter()) {
            text = text.replace(from, to);
        }
        *g = text.into_bytes();
    }
    if fa.keys().ne(fb.keys()) {
        return Ok(Some(format!("files {:?} and {:?}", fa.keys().collect::<Vec<_>>(), fb.keys().collect::<Vec<_>>())));
    }
    Ok(fa.iter().find(|(n, x)| fb.get(*n) != Some(*x)).map(|(n, x)| format!("{n}: {} and {} bytes, not the same", x.len(), fb[n].len())))
}

/// The peak and the present memory of this process (Linux's VmHWM and VmRSS, bytes), where it says.
fn rss() -> Option<(u64, u64)> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let kb = |k: &str| s.lines().find_map(|l| l.strip_prefix(k)?.trim().strip_suffix("kB")?.trim().parse::<u64>().ok()).map(|v| v * 1024);
    Some((kb("VmHWM:")?, kb("VmRSS:")?))
}

fn gib(b: u64) -> String {
    format!("{:.2} GiB", b as f64 / (1u64 << 30) as f64)
}

/// `glyd-engine stackcheck [--dirs EACH,STACKED] [--levels penguin,kestrel,linnet,wren] [--places N] [--tokens N] [--scratch DIR]`: the Qwen3.5 layout's random
/// model with experts (the qwen35moe case, or the two directories given: the same weights a tensor an expert's matrix and stacked, `logitsmodel ... --stacked`)
/// held in its two forms on GPU 0: at each level its logits of a prompt and four decode steps the same bits in both, resident and through the expert cache
/// (`--places` of a layer's experts, fewer than it has; copy mode, the resident layer's bits); and at each lossy level the save of each form (`pack::run_with`)
/// the same files byte for byte. `--load DIR [--layout L] [--places N]`: the model of DIR loaded through the expert cache alone, and the process's memory
/// after (Linux): its peak against the experts' packed bytes the cache holds and the largest tensor of the checkpoint (a stack of experts read whole would
/// add one).
pub fn check(a: &crate::args::Args) -> Result<()> {
    use crate::weights::Layout;
    let places = a.num("places", 4usize)?;
    let n = a.num("tokens", 16usize)?.max(6);
    if let Some(dir) = a.flag("load") {
        return load_only(Path::new(dir), a.layout()?, places, n);
    }
    let own = a.flag("scratch").is_none();
    let scratch = a.flag("scratch").map(PathBuf::from).unwrap_or_else(|| std::env::temp_dir().join(format!("glyd-stackcheck-{}", std::process::id())));
    let (each, stacked) = match a.flag("dirs") {
        Some(d) => match d.split_once(',') {
            Some((x, y)) => (PathBuf::from(x), PathBuf::from(y)),
            None => return err("--dirs EACH,STACKED"),
        },
        None => {
            let (x, y) = (scratch.join("each"), scratch.join("stacked"));
            if !x.join("config.json").exists() {
                crate::logits::qwen35moe_random(&x)?;
            }
            if !y.join("config.json").exists() {
                crate::logits::random_model("qwen35moe-st", &y)?;
            }
            (x, y)
        }
    };
    let levels: Vec<String> = a.flag("levels").unwrap_or("penguin,kestrel,linnet,wren").split(',').map(str::to_string).collect();
    let vocab = crate::runtime::config::ModelConfig::load(&each)?.num("vocab_size").unwrap_or(1000).max(16);
    let tokens: Vec<u32> = (0..n).map(|i| ((i * 7919 + 13) % (vocab - 4) + 2) as u32).collect();
    let run = |dir: &Path, layout: Option<Layout>, cached: bool| -> Result<Vec<Vec<f32>>> {
        let o = crate::dynamic::Opts { ctx: n + 64, chunk: 256, graph: false, device: 0, layout };
        let mut e = if cached {
            let caching = crate::hbg_store::HbCaching {
                places,
                ram: None,
                offload: crate::offload::Offload { mode: crate::offload::Mode::Copy, ..Default::default() },
                dense: 0,
            };
            crate::dynamic::split::open_cached(dir, &o, &[], &[0], vec![caching])?
        } else {
            crate::dynamic::split::open(dir, &o, &[], &[0])?
        };
        let mut out = Vec::new();
        e.prefill(&tokens[..n - 4])?;
        out.push(e.logits()?);
        for &t in &tokens[n - 4..] {
            e.step(t)?;
            out.push(e.logits()?);
        }
        Ok(out)
    };
    let differ =
        |x: &[Vec<f32>], y: &[Vec<f32>]| -> usize { x.iter().zip(y).map(|(p, q)| p.iter().zip(q).filter(|(u, v)| u.to_bits() != v.to_bits()).count()).sum() };
    let mut failed = Vec::new();
    for level in &levels {
        let layout = crate::logits::level_layout(level)?;
        let (re, rs) = (run(&each, layout, false)?, run(&stacked, layout, false)?);
        let (ce, cs) = (run(&each, layout, true)?, run(&stacked, layout, true)?);
        let (d_res, d_cache, d_own) = (differ(&re, &rs), differ(&ce, &cs), differ(&rs, &cs));
        let ok = d_res == 0 && d_cache == 0 && d_own == 0;
        println!(
            "{level}: resident {} rows of {} logits, stacked vs per expert {d_res} differ; through the expert cache ({places} places, copy) {d_cache} differ; stacked cached vs resident {d_own} differ: {}",
            re.len(),
            re.first().map_or(0, Vec::len),
            if ok { "PASS" } else { "FAIL" }
        );
        if !ok {
            failed.push(format!("{level} logits"));
        }
        if let Some(l) = layout.filter(|l| matches!(l, Layout::Int6 | Layout::Int5 | Layout::Int4)) {
            let save = glyd_gpu::save::Layout::by_name(match l {
                Layout::Int6 => "i6",
                Layout::Int5 => "i5",
                _ => "i4",
            })
            .ok_or_else(|| Error("a save layout".into()))?;
            let (oe, os) = (scratch.join(format!("each-{level}")), scratch.join(format!("stacked-{level}")));
            for (m, o) in [(&each, &oe), (&stacked, &os)] {
                let _ = std::fs::remove_dir_all(o);
                crate::pack::run_with(&m.to_string_lossy(), o, 8, save, true)?;
            }
            match same_files(&oe, &os, (&each.to_string_lossy(), &stacked.to_string_lossy()))? {
                None => println!("{level}: the save of the stacked form is the per-expert form's, every file byte for byte: PASS"),
                Some(w) => {
                    println!("{level}: the saves differ: {w}: FAIL");
                    failed.push(format!("{level} save"));
                }
            }
        }
    }
    if own {
        let _ = std::fs::remove_dir_all(&scratch);
    }
    if failed.is_empty() {
        println!("stackcheck: PASS ({} levels)", levels.len());
        Ok(())
    } else {
        err(format!("stackcheck: FAIL: {}", failed.join(", ")))
    }
}

/// `stackcheck --load DIR`: the model of DIR through the expert cache alone (copy mode, `places` a layer), a prompt of `n` tokens run, and the process's
/// memory said: before the load, its peak after, the experts' packed bytes the cache holds, and the checkpoint's largest tensor.
fn load_only(dir: &Path, layout: Option<crate::weights::Layout>, places: usize, n: usize) -> Result<()> {
    let before = rss();
    let ck = io(Checkpoint::open(dir))?;
    let largest = ck.files.iter().flat_map(|f| f.tensors.iter()).map(|(n, t)| (t.bytes, n.clone())).max().unwrap_or_default();
    let (d, _) = crate::runtime::describe::describe_dir(dir)?;
    let moe = crate::dynamic::experts::of_layers(&d, (0, d.desc.blocks.len() / 2), 2).count();
    let experts = crate::dynamic::experts::of_layers(&d, (0, d.desc.blocks.len() / 2), 2).next().map_or(0, |(_, m, _)| m.experts);
    drop(ck);
    let o = crate::dynamic::Opts { ctx: n + 64, chunk: 256, graph: false, device: 0, layout };
    let caching = crate::hbg_store::HbCaching {
        places,
        ram: None,
        offload: crate::offload::Offload { mode: crate::offload::Mode::Copy, ..Default::default() },
        dense: 0,
    };
    let t0 = std::time::Instant::now();
    let mut e = crate::dynamic::split::open_cached(dir, &o, &[], &[0], vec![caching])?;
    let loaded = t0.elapsed().as_secs_f64();
    let after = rss();
    let per = match &e {
        crate::dynamic::split::Run::One(x) => x.stores().first().map_or(0, |s| s.expert_bytes()),
        _ => 0,
    };
    let packed = per * (experts * moe) as u64;
    let tokens: Vec<u32> = (0..n as u32).map(|i| (i * 7919 + 13) % 4000 + 2).collect();
    e.prefill(&tokens)?;
    e.logits()?;
    println!("load: {} through the expert cache ({places} places a layer, copy) in {loaded:.1} s: {moe} layers of {experts} experts", dir.display());
    println!(
        "experts packed in the computer's memory: {} ({} an expert); largest tensor of the checkpoint: {} ({})",
        gib(packed),
        per,
        gib(largest.0),
        largest.1
    );
    match (before, after, rss()) {
        (Some((_, r0)), Some((peak, r1)), Some((peak2, _))) => println!(
            "memory: before the load {}; after it {} (peak {}); after a prompt peak {}; peak less before {}, less the experts packed {}",
            gib(r0),
            gib(r1),
            gib(peak),
            gib(peak2),
            gib(peak2.saturating_sub(r0)),
            gib(peak2.saturating_sub(r0).saturating_sub(packed))
        ),
        _ => println!("memory: not read here (no /proc/self/status)"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    /// The Qwen3.5 layout's random model with experts written with each layer's experts stacked (the release's form) is the same weights: each stacked tensor's
    /// expert rows are the per-expert tensors' bytes, read back by their ranges (as every reader of the kind reads them); and the lossy route's save of it at
    /// a block level is the per-expert form's, every file byte for byte (its packs and glyd.json named each expert's, read from the stacked rows).
    #[test]
    fn the_stacked_form_is_the_same_model_and_saves_the_same() {
        use glyd_gpu::safetensors::Checkpoint;
        let dir = std::env::temp_dir().join(format!("glyd-stacked-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (each, stacked) = (dir.join("each"), dir.join("stacked"));
        crate::logits::qwen35moe_random(&each).unwrap();
        crate::logits::random_model("qwen35moe-st", &stacked).unwrap();
        let (named, _) = crate::runtime::describe::describe_dir(&each).unwrap();
        let (held, _) = crate::runtime::describe::describe_dir(&stacked).unwrap();
        let rows = crate::runtime::describe::stacked_rows(&named, &held).unwrap();
        assert_eq!(rows.len(), 4 * 32 * 3, "four layers of 32 experts: each one's gate, up and down");
        let (a, b) = (Checkpoint::open(&each).unwrap(), Checkpoint::open(&stacked).unwrap());
        assert!(b.get("model.layers.0.mlp.experts.gate_up_proj").is_some() && b.get("model.layers.0.mlp.experts.0.gate_proj.weight").is_none());
        for (m, (t, r0, r1)) in &rows {
            let (f, ti) = a.get(m).unwrap();
            let want = f.read(ti).unwrap();
            let mut got = vec![0u16; (r1 - r0) * ti.shape[1] as usize];
            b.read_rows_bf16(t, *r0, *r1, &mut got).unwrap();
            assert_eq!(got.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>(), want, "{m}");
        }
        for n in a.names().filter(|n| !rows.contains_key(*n)) {
            assert_eq!(a.read(n).unwrap(), b.read(n).unwrap(), "{n}");
        }
        for level in [glyd_gpu::save::Layout::Int6, glyd_gpu::save::Layout::Int4] {
            let (oa, ob) = (dir.join(format!("each-{}", level.name())), dir.join(format!("stacked-{}", level.name())));
            crate::pack::run_with(each.to_str().unwrap(), &oa, 4, level, true).unwrap();
            crate::pack::run_with(stacked.to_str().unwrap(), &ob, 4, level, true).unwrap();
            assert_eq!(super::same_files(&oa, &ob, (each.to_str().unwrap(), stacked.to_str().unwrap())).unwrap(), None, "{}", level.name());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
