//! A model ready to run, made on first use: the Hub checkpoint packed into the engine's cache (its level's layout), once.
//!
//! The cache is `$GLYD_CACHE`, else `$XDG_CACHE_HOME/glyd`, else `~/.cache/glyd`; a packed model lives in `packed/<model>-<revision>-<layout>/`, made beside the final name and
//! renamed into it, so a run that stops leaves nothing half made. The executor's 12-bit KV tables are kept in `kv12/` (src/dynamic/kv12.rs).

use crate::ffi::{err, Error, Result};
use crate::hub::{self, Entry, Hub, HubError, Remote};
use crate::ui::{fmt_time, gb, Bar, Done, Refusal, Ui};
use crate::weights::Layout;
use glyd_gpu::save::Source;
use std::cell::RefCell;
use std::path::{Path, PathBuf};

pub fn cache_root() -> PathBuf {
    if let Some(p) = std::env::var_os("GLYD_CACHE") {
        return p.into();
    }
    if let Some(p) = std::env::var_os("XDG_CACHE_HOME") {
        return Path::new(&p).join("glyd");
    }
    Path::new(&crate::os::home()).join(".cache").join("glyd")
}

fn io<T>(r: std::io::Result<T>) -> Result<T> {
    r.map_err(|e| Error(e.to_string()))
}

/// The directory name of a packed model: the repo (or directory) name, the revision's first eight characters (where it has one) and the layout (the compact one, which glyd.json names as the tiered, `a`).
pub fn packed_name(source: &glyd_gpu::save::Source, layout: Layout) -> String {
    let base: String = source.repo.trim_end_matches('/').rsplit('/').take(2).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("--");
    let rev = source.revision.as_deref().map_or(String::new(), |r| format!("-{}", &r[..r.len().min(8)]));
    format!("{base}{rev}-{}", if layout == Layout::Compact { "a" } else { layout.name() })
}

/// The new packed directory `tmp` put in the place of the old one `dir`.
fn replace_old(dir: &Path, tmp: &Path) -> Result<()> {
    io(std::fs::remove_dir_all(dir))?;
    io(std::fs::rename(tmp, dir))
}

// ---- the pieces of a first run, in the order `glyd run` takes them ------------------------------------------------------------------------------------------

const HUB: &str = "https://huggingface.co";

/// A Hub repo id: ORG/NAME, no more.
pub fn looks_like_repo(model: &str) -> bool {
    let ok = |p: &str| !p.is_empty() && p.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    let mut it = model.split('/');
    matches!((it.next(), it.next(), it.next()), (Some(a), Some(b), None) if ok(a) && ok(b))
}

/// The Hub's refusal in words.
pub fn hub_refusal(model: &str) -> Refusal {
    Refusal::new(
        format!("the Hugging Face Hub would not show {model}: the name is wrong, or it is a gated or private model"),
        format!("Check the name (ORG/NAME, as in google/gemma-4-12B-it). For a gated model, accept its licence at {HUB}/{model}, then run: glyd login   (it takes a token from {HUB}/settings/tokens)"),
    )
}

/// A checkpoint that is not bf16 (the Hub counts the parameters of each dtype; `weights` is the safetensors' bytes), refused in words: Glyd packs bf16 weights. `packed`: a repo whose glyd.json says
/// glyd-v7, whose buffers are bytes and words of packs, not weights.
pub fn check_dtypes(model: &str, weights: u64, dtypes: &[(String, u64)], packed: bool) -> Option<Refusal> {
    if packed || dtypes.is_empty() {
        return None;
    }
    let n = |k: &str| dtypes.iter().find(|(d, _)| d == k).map_or(0, |(_, n)| *n);
    let bf16 = 2 * (n("BF16") + n("F16"));
    if weights > bf16 && (weights - bf16) as f64 >= 0.05 * weights as f64 {
        if n("F32") * 4 > weights / 2 {
            return Some(Refusal::unsupported(
                format!("{model} is stored in 32-bit floats, and Glyd packs bf16 weights"),
                "Run a bf16 version of the model if there is one (most models publish one)",
            ));
        }
        let kind = if n("F8_E4M3") > n("U8").max(n("I8")) { "FP8" } else { "4-bit" };
        return Some(Refusal::unsupported(
            format!("{model} is already quantized ({kind}), and Glyd packs bf16 weights"),
            "Run the model's bf16 version: usually the same name without -FP8, -AWQ or -GPTQ",
        ));
    }
    if n("F16") > n("BF16") {
        return Some(Refusal::unsupported(
            format!("{model} is stored in float16, and Glyd packs bfloat16 weights"),
            "Use the model's bf16 version if it has one",
        ));
    }
    None
}

/// The model's files on the Hub, or why not: the repo listed, its weights counted by dtype (a checkpoint that is not bf16 is refused here, before any download).
pub fn hub_plan(model: &str, ui: &RefCell<Ui>) -> Done<(Hub, Remote, Vec<Entry>)> {
    hub_plan_as(model, ui, false)
}

/// `hub_plan` of a release the dynamic engine runs as it is (`exact`: an FP8 or FP4 checkpoint is not refused; GLY-354).
pub fn hub_plan_as(model: &str, ui: &RefCell<Ui>, exact: bool) -> Done<(Hub, Remote, Vec<Entry>)> {
    if !looks_like_repo(model) {
        return Err(Refusal::new(
            format!("{model}: no such directory, and not a Hugging Face name"),
            "Give a directory with the model, or its Hub name (ORG/NAME, as in google/gemma-4-12B-it)",
        ));
    }
    if hub::offline() {
        return Err(Refusal::new(format!("{model} is not downloaded, and HF_HUB_OFFLINE is set"), "Unset HF_HUB_OFFLINE to download it"));
    }
    let hub = Hub::from_env();
    ui.borrow_mut().status("Reading the model's details...");
    let remote = match hub.list(model, "main") {
        Ok(r) => r,
        Err(HubError::Denied(_)) => return Err(hub_refusal(model)),
        Err(e) => return Err(Refusal::new(format!("cannot read {model} from the Hugging Face Hub ({e})"), "Check the network connection and run this again")),
    };
    let wanted = hub::wanted(&remote);
    let weights: u64 = wanted.iter().filter(|e| e.path.ends_with(".safetensors")).map(|e| e.size).sum();
    if weights == 0 {
        return Err(Refusal::unsupported(
            format!("{model} has no safetensors weights"),
            "Glyd reads .safetensors checkpoints, which most models on the Hub have",
        ));
    }
    let packed = remote.files.iter().any(|e| e.path == "glyd.json") && hub.text(model, "main", "glyd.json").ok().is_some_and(|t| glyd_v7_level(&t).is_some());
    if let Some(r) = check_dtypes(model, weights, &remote.dtypes, packed || exact) {
        return Err(r);
    }
    Ok((hub, remote, wanted))
}

/// The level a glyd.json (text) names where it is a glyd-v7 save's.
fn glyd_v7_level(glyd_json: &str) -> Option<String> {
    let m = glyd_gpu::json::parse(glyd_json).ok()?;
    (m.get("format")?.as_str()? == "glyd-v7").then(|| m.get("level")?.as_str().map(str::to_string))?
}

/// The pre-packed repo `repo` of a model at `level` (kestrel or swift), planned as `hub_plan` plans a model, or None where the Hub has no such repo or it is not a glyd-v7 save of that level
/// (the caller packs the bf16 model here instead). A repo the Hub would not show is none: it answers 401 for a name that is not there as for a private one.
pub fn hub_plan_packed(repo: &str, level: &str, ui: &RefCell<Ui>) -> Done<Option<(Hub, Remote, Vec<Entry>)>> {
    if !looks_like_repo(repo) || hub::offline() {
        return Ok(None);
    }
    let hub = Hub::from_env();
    ui.borrow_mut().status("Looking for a pre-packed download...");
    let remote = match hub.list(repo, "main") {
        Ok(r) => r,
        Err(HubError::Denied(_)) => return Ok(None),
        Err(e) => return Err(Refusal::new(format!("cannot read {repo} from the Hugging Face Hub ({e})"), "Check the network connection and run this again")),
    };
    let wanted = packed_files(&remote, hub.text(repo, "main", "glyd.json").ok().as_deref(), level);
    Ok(wanted.map(|w| (hub, remote, w)))
}

/// The files to download of a repo whose glyd.json (text, where it has one) is a glyd-v7 save of `level`, else None.
fn packed_files(remote: &Remote, glyd_json: Option<&str>, level: &str) -> Option<Vec<Entry>> {
    (remote.files.iter().any(|e| e.path == "glyd.json") && glyd_v7_level(glyd_json?).as_deref() == Some(level)).then(|| hub::wanted(remote))
}

/// `entries` of `remote` in the cache, with one bar; where they are all there already, nothing is read.
pub fn download(hub: &Hub, remote: &Remote, entries: &[Entry], model: &str, ui: &RefCell<Ui>) -> Done<()> {
    let need = hub.missing(remote, entries);
    let total: u64 = entries.iter().map(|e| e.size).sum();
    if need > 0 {
        let mut probe = hub.cache().to_path_buf();
        while !probe.exists() && probe.pop() {}
        if let Some(free) = hub::disk_free(&probe) {
            if free < need + (1 << 30) {
                return Err(Refusal::new(
                    format!("the disk holding {} has {} left, and {} are to be downloaded", hub.cache().display(), gb(free), gb(need)),
                    format!("Free some space, or keep the models on another drive: HF_HOME=/path/on/that/drive glyd run {model}"),
                ));
            }
        }
    }
    let name = model.rsplit('/').next().unwrap_or(model);
    let bar = RefCell::new(Bar::new(&format!("Downloading {name}"), total));
    let shown = std::cell::Cell::new(false);
    let r = hub.fetch(remote, entries, &|done, _| {
        if need > 0 {
            shown.set(true);
            bar.borrow_mut().update(&mut ui.borrow_mut(), done, false);
        }
    });
    match r {
        Ok(_) => {}
        Err(HubError::Denied(_)) => return Err(hub_refusal(model)),
        Err(HubError::Network(e)) => {
            return Err(Refusal::new(
                format!("the download of {model} stopped ({e})"),
                "Check the network connection and run the same command again: it resumes where it stopped",
            ))
        }
        Err(HubError::Other(e)) => return Err(Refusal::new(e, "")),
    }
    hub.pin(remote)?;
    if shown.get() {
        bar.borrow_mut().update(&mut ui.borrow_mut(), total, true);
        ui.borrow_mut().line(&format!("Downloaded {name} ({}).", gb(total)));
    } else {
        ui.borrow_mut().line(&format!("{name} is downloaded already."));
    }
    Ok(())
}

/// What a run that was stopped left in `dir` (the lock is held, so no other run is making anything): the entries whose names start with `prefix`.
fn remove_stale(dir: &Path, prefix: &str) {
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        if e.file_name().to_string_lossy().starts_with(prefix) {
            let p = e.path();
            let _ = if p.is_dir() { std::fs::remove_dir_all(&p) } else { std::fs::remove_file(&p) };
        }
    }
}

/// The bytes in the files of a directory (not below it).
fn dir_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir).map_or(0, |d| d.flatten().filter_map(|e| e.metadata().ok()).filter(|m| m.is_file()).map(|m| m.len()).sum())
}

/// The engine's cache of packed models.
pub fn packed_root() -> PathBuf {
    cache_root().join("engine")
}

/// Where `source` packed in `layout` is (or would be) in the cache.
pub fn packed_dir(source: &Source, layout: Layout) -> PathBuf {
    packed_root().join(packed_name(source, layout))
}

/// Whether `dir` holds a packed model of the layout of this glyd (one packed by glyd 0.27 and before is made again, once).
pub fn is_packed(dir: &Path) -> bool {
    dir.join("engine.safetensors").exists() && crate::weights::earlier_format(dir).is_none()
}

/// Whether `dir` holds a glyd-v7 save in `layout` (a lossy level's pack in the cache: it is there whole or not at all).
pub fn cached_pack(dir: &Path, layout: Layout) -> bool {
    matches!(crate::weights::v7_manifest(dir), Ok(Some((l, _))) if l == layout)
}

/// A failure of what the engine does for itself (its cache, its packing, its tables), which the other runner does not need: the status that hands the model to it.
fn cannot(e: impl std::fmt::Display) -> Refusal {
    Refusal::unsupported(e.to_string(), "")
}

/// `source` packed for this GPU into the cache (once; a run that stops leaves nothing half made), with a bar by the bytes written. `expect`: about what it comes to.
pub fn pack(source: &Source, model: &str, layout: Layout, expect: u64, ui: &RefCell<Ui>) -> Done<PathBuf> {
    let root = packed_root();
    let io = |r: std::io::Result<()>| {
        r.map_err(|e| Refusal::unsupported(format!("{}: {e}", root.display()), "Check that the folder can be written, or set GLYD_CACHE to another"))
    };
    io(std::fs::create_dir_all(&root))?;
    let dir = packed_dir(source, layout);
    // (a packed copy of a format this glyd does not read is made again, once; a level's is a glyd-v7 directory, which a rename puts in its place whole: penguin's too, the
    // executor's, where the 12-bit layout and the compact one are glyd-v4 saves with the embedding beside them)
    let v7 = !matches!(layout, Layout::Twelve | Layout::Compact);
    let have = |d: &Path| if v7 { cached_pack(d, layout) } else { is_packed(d) };
    if have(&dir) {
        return Ok(dir);
    }
    if let Some(free) = hub::disk_free(&root) {
        if free < expect + (1 << 30) {
            return Err(Refusal::unsupported(
                format!("the disk holding {} has {} left, and the packed model takes {}", root.display(), gb(free), gb(expect)),
                "Free some space, or put the cache on another drive: GLYD_CACHE=/path/on/that/drive",
            ));
        }
    }
    // one process makes it at a time
    let lock = std::fs::File::create(root.join(".lock")).map_err(cannot)?;
    lock.lock().map_err(cannot)?;
    if have(&dir) {
        return Ok(dir);
    }
    let old = dir.join("glyd.json").exists();
    let name = model.rsplit('/').next().unwrap_or(model);
    if old {
        ui.borrow_mut().line(&format!(
            "The packed copy of {name} is of a format this glyd does not read (the earlier dense engine's, or glyd 0.27's): packing it again for this GPU ({}), once: {}",
            layout.name(),
            dir.display()
        ));
    } else {
        ui.borrow_mut().line(&format!("Packing {name} for this GPU ({}), once: {}", layout.name(), dir.display()));
    }
    remove_stale(&root, ".packing-");
    let tmp = root.join(format!(".packing-{}", std::process::id()));
    let t0 = std::time::Instant::now();
    let mut bar = Bar::new(&format!("Packing {name}"), expect);
    let result = std::thread::scope(|s| {
        let work = s.spawn(|| crate::pack::run_with(model, &tmp, 16, layout, true));
        while !work.is_finished() {
            bar.update(&mut ui.borrow_mut(), dir_bytes(&tmp), false);
            std::thread::sleep(std::time::Duration::from_millis(250));
        }
        work.join().unwrap_or_else(|_| err("the packing thread panicked"))
    });
    if let Err(e) = result {
        let _ = std::fs::remove_dir_all(&tmp);
        return Err(cannot(format!("packing {model} stopped: {e}")));
    }
    bar.update(&mut ui.borrow_mut(), expect, true);
    if old {
        replace_old(&dir, &tmp).map_err(cannot)?;
    } else {
        io(std::fs::rename(&tmp, &dir))?;
    }
    ui.borrow_mut().line(&format!("Packed in {} ({}).", fmt_time(t0.elapsed().as_secs_f64()), gb(dir_bytes(&dir))));
    Ok(dir)
}

/// The nearest folder of `path` that exists (a disk is asked about that one).
pub fn existing_ancestor(path: &Path) -> PathBuf {
    let mut p = path.to_path_buf();
    while !p.exists() && p.pop() {}
    p
}

/// Whether two folders that exist are on one filesystem.
#[cfg(unix)]
pub fn same_filesystem(a: &Path, b: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    matches!((std::fs::metadata(a), std::fs::metadata(b)), (Ok(x), Ok(y)) if x.dev() == y.dev())
}

/// Whether two folders that exist are on one volume (Windows: the same drive or share at the start of their full paths).
#[cfg(windows)]
pub fn same_filesystem(a: &Path, b: &Path) -> bool {
    let root = |p: &Path| std::fs::canonicalize(p).ok().and_then(|c| c.components().next().map(|x| x.as_os_str().to_ascii_lowercase()));
    matches!((root(a), root(b)), (Some(x), Some(y)) if x == y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoints_that_are_not_bf16_are_refused() {
        let d = |v: &[(&str, u64)]| v.iter().map(|(k, n)| (k.to_string(), *n)).collect::<Vec<_>>();
        // Qwen3-8B: 8.19 B parameters in bf16, 16.4 GB
        assert!(check_dtypes("Qwen/Qwen3-8B", 16_381_470_720, &d(&[("BF16", 8_190_735_360)]), false).is_none());
        assert!(check_dtypes("x", 1, &[], false).is_none());
        let fp8 = check_dtypes("Qwen/Qwen3-8B-FP8", 8_900_000_000, &d(&[("F8_E4M3", 6_950_000_000), ("BF16", 1_240_000_000)]), false).unwrap();
        assert!(fp8.what.contains("FP8"), "{}", fp8.what);
        let q4 = check_dtypes("x-AWQ", 5_700_000_000, &d(&[("I32", 1_100_000_000), ("F16", 600_000_000)]), false).unwrap();
        assert!(q4.what.contains("4-bit"), "{}", q4.what);
        let f32 = check_dtypes("x", 32_000_000_000, &d(&[("F32", 8_000_000_000)]), false).unwrap();
        assert!(f32.what.contains("32-bit"), "{}", f32.what);
        let f16 = check_dtypes("x", 16_000_000_000, &d(&[("F16", 8_000_000_000)]), false).unwrap();
        assert!(f16.what.contains("float16"), "{}", f16.what);
    }

    #[test]
    fn what_a_stopped_run_left_goes() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-stale-{}", std::process::id()));
        std::fs::create_dir_all(dir.join(".packing-1234")).unwrap();
        std::fs::write(dir.join(".packing-1234").join("shard"), b"x").unwrap();
        std::fs::write(dir.join(".tables-77"), b"x").unwrap();
        std::fs::create_dir_all(dir.join("Qwen--Qwen3-8B-mma")).unwrap();
        remove_stale(&dir, ".packing-");
        assert!(!dir.join(".packing-1234").exists() && dir.join(".tables-77").exists() && dir.join("Qwen--Qwen3-8B-mma").exists());
        remove_stale(&dir, ".tables-");
        assert!(!dir.join(".tables-77").exists());
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(
            looks_like_repo("Qwen/Qwen3-8B")
                && looks_like_repo("a-b/c_d.e")
                && !looks_like_repo("Qwen3-8B")
                && !looks_like_repo("a/b/c")
                && !looks_like_repo("/models/x")
        );
    }

    #[test]
    fn names_of_packed_models() {
        let s = glyd_gpu::save::Source { dir: "/x".into(), repo: "Qwen/Qwen3-8B".into(), revision: Some("b968826d9c46dd6066d109eabc6255188de91218".into()) };
        assert_eq!(packed_name(&s, Layout::Tiered), "Qwen--Qwen3-8B-b968826d-mma");
        let d = glyd_gpu::save::Source { dir: "/models/mine".into(), repo: "/models/mine".into(), revision: None };
        assert_eq!(packed_name(&d, Layout::Twelve), "models--mine-mma12");
        // each layout its own directory: the compact one is not the tiered one's, and the lossy levels have theirs
        let names: Vec<String> = [Layout::Tiered, Layout::Twelve, Layout::Compact, Layout::Int6, Layout::Int5, Layout::Int4, Layout::Int3]
            .iter()
            .map(|&l| packed_name(&s, l))
            .collect();
        let want = ["mma", "mma12", "a", "i6", "i5", "i4", "i3"].map(|l| format!("Qwen--Qwen3-8B-b968826d-{l}"));
        assert_eq!(names, want);
    }

    /// A glyd-v7 repo is taken whole (its buffers are not weights); the same counts of an unknown 4-bit repo are still refused.
    #[test]
    fn a_pre_packed_repo_is_planned_and_its_buffers_pass() {
        let d = |v: &[(&str, u64)]| v.iter().map(|(k, n)| (k.to_string(), *n)).collect::<Vec<_>>();
        let u8s = d(&[("U8", 7_000_000_000), ("I32", 300_000_000), ("BF16", 1_000_000_000)]);
        assert!(check_dtypes("glyd/Qwen3.5-9B-swift", 6_000_000_000, &u8s, true).is_none());
        let q4 = check_dtypes("x-AWQ", 5_700_000_000, &d(&[("I32", 1_100_000_000), ("F16", 600_000_000)]), false).unwrap();
        assert!(q4.what.contains("4-bit"), "{}", q4.what);
        let e = |p: &str, size| Entry { path: p.to_string(), size, blob: p.to_string(), sha256: None };
        let remote = |files: Vec<Entry>| Remote { repo: "glyd/m-swift".into(), revision: "main".into(), commit: "c".repeat(40), files, dtypes: vec![] };
        let files = vec![e("glyd.json", 9), e("config.json", 9), e("model-00001-of-00002.safetensors", 100), e("sub/x.safetensors", 5)];
        let v7 = |level: &str| format!("{{\"format\": \"glyd-v7\", \"layout\": \"i5\", \"level\": \"{level}\"}}");
        let paths = |w: Option<Vec<Entry>>| w.map(|w| w.iter().map(|e| e.path.clone()).collect::<Vec<_>>());
        assert_eq!(
            paths(packed_files(&remote(files.clone()), Some(&v7("swift")), "swift")),
            Some(vec!["glyd.json".to_string(), "config.json".into(), "model-00001-of-00002.safetensors".into()])
        );
        // the Hub's repo falls back to the bf16 model: another level, another format, no glyd.json, one that cannot be read
        assert!(packed_files(&remote(files.clone()), Some(&v7("kestrel")), "swift").is_none());
        assert!(packed_files(&remote(files.clone()), Some("{\"format\": \"glyd-v4\", \"level\": \"swift\"}"), "swift").is_none());
        assert!(packed_files(&remote(files[1..].to_vec()), Some(&v7("swift")), "swift").is_none());
        assert!(packed_files(&remote(files), None, "swift").is_none());
    }

    /// A lossy level's pack is in the cache when its directory is a glyd-v7 save of that layout, and not for the other level's.
    #[test]
    fn a_cached_lossy_pack_is_found_by_its_layout() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-cached-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!cached_pack(&dir, Layout::Int5));
        std::fs::write(dir.join("glyd.json"), "{\"format\": \"glyd-v7\", \"layout\": \"i5\", \"level\": \"swift\", \"packs\": {}}").unwrap();
        assert!(cached_pack(&dir, Layout::Int5) && !cached_pack(&dir, Layout::Int6) && !cached_pack(&dir, Layout::Int4));
        std::fs::write(dir.join("glyd.json"), "{\"format\": \"glyd-v7\", \"layout\": \"i4\", \"level\": \"wren\", \"packs\": {}}").unwrap();
        assert!(cached_pack(&dir, Layout::Int4) && !cached_pack(&dir, Layout::Int5) && !cached_pack(&dir, Layout::Int3));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn folders_and_filesystems() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-fs-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(existing_ancestor(&dir.join("a").join("b")), dir);
        assert_eq!(existing_ancestor(&dir), dir);
        assert!(same_filesystem(&dir, &std::env::temp_dir()));
        assert!(!same_filesystem(&dir, &dir.join("missing")));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_checkpoint_that_does_not_run_is_for_the_other_runner() {
        let d = |v: &[(&str, u64)]| v.iter().map(|(k, n)| (k.to_string(), *n)).collect::<Vec<_>>();
        for r in [
            check_dtypes("x-FP8", 8_900_000_000, &d(&[("F8_E4M3", 6_950_000_000), ("BF16", 1_240_000_000)]), false).unwrap(),
            check_dtypes("x", 32_000_000_000, &d(&[("F32", 8_000_000_000)]), false).unwrap(),
            check_dtypes("x", 16_000_000_000, &d(&[("F16", 8_000_000_000)]), false).unwrap(),
        ] {
            assert_eq!(r.code, crate::ui::CANNOT, "{r}");
        }
    }
}
