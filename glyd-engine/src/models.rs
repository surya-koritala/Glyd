//! Model names, and `glyd pull`, `ls` and `rm` over the Hub cache.
//!
//! A model is a Hub id (`Qwen/Qwen3.5-9B`), a directory, a GGUF file, or a short name from the manifest (models.json: versioned, in the repository and in the
//! program; `$GLYD_MODELS` or `<cache>/models.json` names another). The grammar of a short name is `NAME[:SIZE][:LEVEL]` (`qwen3.5`, `qwen3.5:27b`,
//! `qwen3.5:27b:swift`); a tag of letters only is a level (levels.rs), so `:27b` and `:35b-a3b` are sizes. NAME and SIZE are matched without case, and NAME without
//! its `-` and `_`. Where SIZE is left out the family's default is taken; choosing the size that fits this computer is `glyd fit`'s (P4b).
//!
//! The copies of a model are in the Hub's cache (hub.rs; `$HF_HUB_CACHE`) as huggingface_hub keeps them, so what vLLM or the Python tool downloaded is here too,
//! and in the engine's cache of packed copies (store.rs, `$GLYD_CACHE`): `glyd rm` takes a model out of both.

use crate::args::Args;
use crate::levels;
use crate::store;
use crate::ui;
use std::path::{Path, PathBuf};

/// The manifest's version this program reads.
pub const MANIFEST_VERSION: u64 = 1;

/// The manifest as the repository holds it.
const EMBEDDED: &str = include_str!("../models.json");

#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub size: String,
    pub repo: String,
    pub default: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Family {
    pub name: String,
    pub aliases: Vec<String>,
    pub models: Vec<Entry>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Manifest {
    pub families: Vec<Family>,
}

/// A name without case and without `-`, `_`, `.` and spaces: `Qwen-3.5`, `qwen3.5` and `qwen35` are one.
fn norm(s: &str) -> String {
    s.chars().filter(|c| !matches!(c, '-' | '_' | '.' | ' ')).flat_map(char::to_lowercase).collect()
}

impl Manifest {
    /// A manifest from its text: the version it must have, every family named once, every model of it with a size and a Hub id (ORG/NAME), one default and no size twice.
    pub fn parse(text: &str) -> Result<Manifest, String> {
        let v: serde_json::Value = serde_json::from_str(text).map_err(|e| format!("not JSON: {e}"))?;
        let version = v.get("version").and_then(|x| x.as_u64()).ok_or("no \"version\"")?;
        if version != MANIFEST_VERSION {
            return Err(format!("version {version}, and this program reads version {MANIFEST_VERSION}: update glyd"));
        }
        let mut families: Vec<Family> = Vec::new();
        for f in v.get("families").and_then(|x| x.as_array()).ok_or("no \"families\"")? {
            let name = f.get("name").and_then(|x| x.as_str()).filter(|n| !n.is_empty()).ok_or("a family has no \"name\"")?.to_string();
            let aliases: Vec<String> =
                f.get("aliases").and_then(|x| x.as_array()).map_or(Vec::new(), |a| a.iter().filter_map(|x| x.as_str()).map(str::to_string).collect());
            let mut models: Vec<Entry> = Vec::new();
            for m in f.get("models").and_then(|x| x.as_array()).ok_or_else(|| format!("{name}: no \"models\""))? {
                let size = m
                    .get("size")
                    .and_then(|x| x.as_str())
                    .filter(|s| !s.is_empty() && !s.contains(':'))
                    .ok_or_else(|| format!("{name}: a model has no \"size\""))?
                    .to_ascii_lowercase();
                let repo = m
                    .get("repo")
                    .and_then(|x| x.as_str())
                    .filter(|r| store::looks_like_repo(r))
                    .ok_or_else(|| format!("{name}:{size}: \"repo\" is not a Hub id (ORG/NAME)"))?
                    .to_string();
                if models.iter().any(|e| e.size == size) {
                    return Err(format!("{name}:{size} is there twice"));
                }
                models.push(Entry { size, repo, default: m.get("default").and_then(|x| x.as_bool()).unwrap_or(false) });
            }
            if models.is_empty() || models.iter().filter(|e| e.default).count() != 1 {
                return Err(format!("{name}: it needs models, and exactly one of them \"default\""));
            }
            let fam = Family { name, aliases, models };
            for key in std::iter::once(&fam.name).chain(fam.aliases.iter()).map(|s| norm(s)) {
                if families.iter().any(|o| std::iter::once(&o.name).chain(o.aliases.iter()).any(|s| norm(s) == key)) || key.is_empty() {
                    return Err(format!("the name {key} is taken twice"));
                }
            }
            families.push(fam);
        }
        Ok(Manifest { families })
    }

    /// The manifest this program has: the file `$GLYD_MODELS` names, else `<cache>/models.json` where there is one, else the repository's; a file that is not a manifest is said and left.
    pub fn load() -> Manifest {
        let file = std::env::var_os("GLYD_MODELS")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .or_else(|| Some(store::cache_root().join("models.json")).filter(|p| p.is_file()));
        if let Some(p) = file {
            match std::fs::read_to_string(&p).map_err(|e| e.to_string()).and_then(|t| Manifest::parse(&t)) {
                Ok(m) => return m,
                Err(e) => eprintln!("glyd: {} is not a model manifest ({e}): the one in this program is used", p.display()),
            }
        }
        Manifest::parse(EMBEDDED).expect("the manifest in the repository (glyd-engine/models.json) parses")
    }

    fn family(&self, name: &str) -> Option<&Family> {
        let key = norm(name);
        self.families.iter().find(|f| norm(&f.name) == key || f.aliases.iter().any(|a| norm(a) == key))
    }

    /// The names a refusal offers: the families, each with its sizes.
    pub fn names(&self) -> String {
        self.families
            .iter()
            .map(|f| format!("{} ({})", f.name, f.models.iter().map(|m| m.size.as_str()).collect::<Vec<_>>().join(", ")))
            .collect::<Vec<_>>()
            .join("; ")
    }

    /// The family that is the nearest to `name` where it is a slip of one letter (`gemma`, `qwem3`), for "did you mean".
    fn nearest(&self, name: &str) -> Option<&str> {
        let key = norm(name);
        if key.is_empty() {
            return None;
        }
        self.families.iter().map(|f| (edits(&norm(&f.name), &key), f)).filter(|(d, _)| *d <= 1).min_by_key(|(d, _)| *d).map(|(_, f)| f.name.as_str())
    }

    /// The short name of a Hub repo, where the manifest has it (`qwen3.5:9b`).
    pub fn name_of(&self, repo: &str) -> Option<String> {
        self.families.iter().find_map(|f| f.models.iter().find(|m| m.repo == repo).map(|m| format!("{}:{}", f.name, m.size)))
    }
}

/// The edit distance of two short words, capped at 2.
fn edits(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    if a.len().abs_diff(b.len()) > 1 {
        return 2;
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i];
        for j in 1..=b.len() {
            cur.push((prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + usize::from(a[i - 1] != b[j - 1])));
        }
        prev = cur;
    }
    prev[b.len()].min(2)
}

/// What a short name came to.
#[derive(Debug, PartialEq)]
pub struct Resolved {
    /// The model as the family and size name it (`qwen3.5:9b`).
    pub name: String,
    /// Its Hub repo (`Qwen/Qwen3.5-9B`).
    pub repo: String,
    /// The level the user tagged, lowercase.
    pub level: Option<String>,
}

impl Resolved {
    /// The model as the rest of the engine takes it: `Qwen/Qwen3.5-9B` or `Qwen/Qwen3.5-9B:swift`.
    pub fn model(&self) -> String {
        match &self.level {
            Some(l) => format!("{}:{l}", self.repo),
            None => self.repo.clone(),
        }
    }
}

/// Whether `arg` is something other than a short name: a Hub id (it has a `/`), a path, a GGUF file, a Windows drive, or a file or directory here (`exists`).
fn is_location(arg: &str, exists: &dyn Fn(&str) -> bool) -> bool {
    arg.contains('/') || arg.contains('\\') || arg.starts_with(['.', '~']) || arg.to_ascii_lowercase().ends_with(".gguf") || exists(arg)
}

/// `arg` as a short name of the manifest: Ok(None) where it is not one (a Hub id, a path: the rest of the engine takes it as it is), Err where it looks like a name and is not in the
/// manifest, or its size or level is not.
pub fn resolve(arg: &str, m: &Manifest) -> Result<Option<Resolved>, String> {
    resolve_with(arg, m, &|p| Path::new(p).exists())
}

/// `resolve`, with what is a file or directory here given (a test's).
fn resolve_with(arg: &str, m: &Manifest, exists: &dyn Fn(&str) -> bool) -> Result<Option<Resolved>, String> {
    if is_location(arg, exists) {
        return Ok(None);
    }
    let (rest, tag) = levels::split_tag(arg);
    let mut parts = rest.splitn(2, ':');
    let (name, size) = (parts.next().unwrap_or(""), parts.next());
    let Some(fam) = m.family(name) else {
        // (a name that is no family: a word with no digit may be a directory that is not there, which the old answer said; one the manifest nearly has says so)
        let near = m.nearest(name).map(|n| format!(" Did you mean {n}?")).unwrap_or_default();
        return Err(format!(
            "{arg}: no such directory, no such model name, and not a Hub id (ORG/NAME).{near} The names: {}. A Hub id is ORG/NAME, as in Qwen/Qwen3.5-9B.",
            m.names()
        ));
    };
    let entry = match size {
        None => fam.models.iter().find(|e| e.default),
        Some(s) => {
            let s = s.to_ascii_lowercase();
            let exact = fam.models.iter().find(|e| e.size == s);
            // (`35b` for `35b-a3b`, where only one size starts so)
            let by_prefix: Vec<&Entry> = fam.models.iter().filter(|e| e.size.starts_with(&format!("{s}-"))).collect();
            exact.or(if by_prefix.len() == 1 { Some(by_prefix[0]) } else { None })
        }
    };
    let Some(entry) = entry else {
        let sizes: Vec<&str> = fam.models.iter().map(|e| e.size.as_str()).collect();
        return Err(format!("{arg}: {} has no size {}. Its sizes: {}.", fam.name, size.unwrap_or(""), sizes.join(", ")));
    };
    if let Some(t) = &tag {
        if levels::by_name(t).is_none() {
            return Err(format!("{arg}: :{t} is not a level. The levels: {}.", levels::names_now().join(", ")));
        }
    }
    Ok(Some(Resolved { name: format!("{}:{}", fam.name, entry.size), repo: entry.repo.clone(), level: tag }))
}

/// The model word of a command's arguments made a Hub id where it is a short name (one line says which). Err: the refusal, for the caller to print; exit 1.
pub fn resolve_args(a: &mut Args) -> Result<(), String> {
    const TAKES_MODEL: [&str; 11] = ["run", "serve", "fetch", "pull", "prepare", "preflight", "supports", "costs", "fit", "verify", "rm"];
    if !a.pos.first().is_some_and(|c| TAKES_MODEL.contains(&c.as_str())) {
        return Ok(());
    }
    let Some(arg) = a.pos.get(1).cloned() else { return Ok(()) };
    // (`verify PATH` and `rm`'s Hub ids are taken as they are: only a name the manifest has changes)
    match resolve(&arg, &Manifest::load()) {
        Ok(Some(r)) => {
            eprintln!("glyd: {arg} is {}{}", r.repo, r.level.as_deref().map(|l| format!(" at level {l}")).unwrap_or_default());
            a.pos[1] = r.model();
            Ok(())
        }
        Ok(None) => Ok(()),
        // `verify` takes any file name and `rm` any Hub id: a word that is no name is left to them
        Err(_) if matches!(a.pos[0].as_str(), "verify") => Ok(()),
        Err(e) => Err(e),
    }
}

// ---- what is in the cache ------------------------------------------------------------------------------------------------------------------------------------

/// A model's copy in the Hub's cache.
#[derive(Debug, PartialEq)]
pub struct Local {
    pub repo: String,
    /// The bytes its files take.
    pub bytes: u64,
    /// The level its repo is (`glyd/Qwen3.5-9B-swift`: swift); penguin for a release's own weights (lossless: what the first run packs).
    pub level: String,
    /// The short name of the manifest, where the repo is one of its.
    pub name: Option<String>,
}

/// The size of a directory's files, symlinks not followed (the Hub's cache links its snapshots to its blobs: the blobs are counted once, in blobs/).
fn dir_bytes(dir: &Path) -> u64 {
    let Ok(rd) = std::fs::read_dir(dir) else { return 0 };
    rd.flatten()
        .map(|e| match e.file_type() {
            Ok(t) if t.is_dir() => dir_bytes(&e.path()),
            Ok(t) if t.is_file() => e.metadata().map_or(0, |m| m.len()),
            _ => 0,
        })
        .sum()
}

/// The repo a Hub cache directory is of (`models--Qwen--Qwen3.5-9B`: Qwen/Qwen3.5-9B); None for a dataset's or a stray.
fn repo_of_dir(name: &str) -> Option<String> {
    let rest = name.strip_prefix("models--")?;
    let (org, model) = rest.split_once("--")?;
    (!org.is_empty() && !model.is_empty()).then(|| format!("{org}/{model}"))
}

fn dir_of_repo(repo: &str) -> String {
    format!("models--{}", repo.replace('/', "--"))
}

/// A repo's level, and the manifest name of the model it is a copy of (a pre-packed repo is its model's, at its level).
fn describe(repo: &str, m: &Manifest) -> (String, Option<String>) {
    if let Some(l) = levels::repo_level(repo) {
        let level = levels::name_of(l).to_string();
        for f in &m.families {
            for e in &f.models {
                if levels::packed_repo(&e.repo, &level) == repo {
                    return (level, Some(format!("{}:{}", f.name, e.size)));
                }
            }
        }
        return (level, None);
    }
    ("penguin".to_string(), m.name_of(repo))
}

/// The models in the Hub's cache `cache`, the least repo first.
pub fn scan(cache: &Path, m: &Manifest) -> Vec<Local> {
    let Ok(rd) = std::fs::read_dir(cache) else { return Vec::new() };
    let mut out: Vec<Local> = rd
        .flatten()
        .filter_map(|e| {
            let repo = repo_of_dir(&e.file_name().to_string_lossy())?;
            // (a snapshot with nothing in it: a download that stopped before its first file)
            let bytes = dir_bytes(&e.path().join("blobs"));
            let (level, name) = describe(&repo, m);
            Some(Local { repo, bytes, level, name })
        })
        .collect();
    out.sort_by(|a, b| a.repo.to_ascii_lowercase().cmp(&b.repo.to_ascii_lowercase()));
    out
}

/// Whether `dir` (a name in the cache of packed copies) is the packed copy of `repo`: `ORG--NAME`, the revision's first eight characters where it has one, and the layout
/// (store::packed_name).
pub fn is_packed_of(dir: &str, repo: &str) -> bool {
    let Some(rest) = dir.strip_prefix(&format!("{}-", repo.replace('/', "--"))) else { return false };
    let layout = |w: &str| w == "a" || levels::layout_of(w).is_some();
    match rest.split_once('-') {
        Some((rev, lay)) if rev.len() == 8 && rev.bytes().all(|b| b.is_ascii_hexdigit()) => layout(lay),
        _ => layout(rest),
    }
}

/// The packed copies in `root` of `repo`, with their sizes.
pub fn packed_of(root: &Path, repo: &str) -> Vec<(PathBuf, u64)> {
    let Ok(rd) = std::fs::read_dir(root) else { return Vec::new() };
    rd.flatten().filter(|e| is_packed_of(&e.file_name().to_string_lossy(), repo)).map(|e| (e.path(), dir_bytes(&e.path()))).collect()
}

// ---- the commands --------------------------------------------------------------------------------------------------------------------------------------------

/// Bytes as a person reads a size on disk: GiB from one GiB, else MiB.
fn size(n: u64) -> String {
    if n >= 1 << 30 {
        ui::gib(n)
    } else {
        format!("{:.1} MiB", n as f64 / (1u64 << 20) as f64)
    }
}

fn json_str(s: &str) -> String {
    serde_json::Value::String(s.to_string()).to_string()
}

/// `glyd ls [--all] [--json]`: the models downloaded (name, level, size on disk, repo), and the packed copies made for this GPU; `--all` adds the names the manifest has that are not downloaded.
pub fn ls(a: &Args) -> i32 {
    let m = Manifest::load();
    let (cache, packed) = (crate::hub::hub_cache(), store::packed_root());
    let local = scan(&cache, &m);
    let packed_bytes: u64 = std::fs::read_dir(&packed)
        .map_or(0, |rd| rd.flatten().filter(|e| !e.file_name().to_string_lossy().starts_with('.')).map(|e| dir_bytes(&e.path())).sum());
    let missing: Vec<(String, String)> = if a.has("all") {
        m.families
            .iter()
            .flat_map(|f| f.models.iter().map(move |e| (format!("{}:{}", f.name, e.size), e.repo.clone())))
            .filter(|(_, r)| !local.iter().any(|l| &l.repo == r))
            .collect()
    } else {
        Vec::new()
    };
    if a.has("json") {
        let mut items: Vec<String> = local
            .iter()
            .map(|l| {
                format!(
                    "{{\"name\":{},\"repo\":{},\"level\":{},\"bytes\":{},\"downloaded\":true}}",
                    l.name.as_deref().map_or("null".into(), json_str),
                    json_str(&l.repo),
                    json_str(&l.level),
                    l.bytes
                )
            })
            .collect();
        items.extend(
            missing
                .iter()
                .map(|(n, r)| format!("{{\"name\":{},\"repo\":{},\"level\":\"penguin\",\"bytes\":0,\"downloaded\":false}}", json_str(n), json_str(r))),
        );
        println!(
            "{{\"models\":[{}],\"packed_bytes\":{packed_bytes},\"cache\":{},\"packed\":{}}}",
            items.join(","),
            json_str(&cache.to_string_lossy()),
            json_str(&packed.to_string_lossy())
        );
        return 0;
    }
    if local.is_empty() && missing.is_empty() {
        println!("No models downloaded ({}). glyd pull NAME downloads one (glyd ls --all: the names).", cache.display());
        return 0;
    }
    let w = local.iter().map(|l| l.name.as_deref().unwrap_or("-").len()).chain(missing.iter().map(|(n, _)| n.len())).max().unwrap_or(4).max(4);
    println!("{:<w$}  {:<11}  {:>10}  REPO", "NAME", "LEVEL", "SIZE");
    for l in &local {
        println!("{:<w$}  {:<11}  {:>10}  {}", l.name.as_deref().unwrap_or("-"), l.level, size(l.bytes), l.repo);
    }
    for (n, r) in &missing {
        println!("{:<w$}  {:<11}  {:>10}  {r}", n, "-", "not pulled");
    }
    if packed_bytes > 0 {
        println!("\nPacked for this GPU: {} in {}", size(packed_bytes), packed.display());
    }
    0
}

/// `glyd pull NAME[:LEVEL]`: the model downloaded into the Hub's cache (the level's pre-packed repo where there is one; where there is none the release's own weights, which the first run packs).
pub fn pull(a: &Args) -> i32 {
    let Ok(arg) = a.need(1, "MODEL") else { return 1 };
    // (a short name was made a Hub id by resolve_args; a tag on it is the level)
    let (repo, tag) = levels::split_tag(arg);
    let level = tag.as_deref().and_then(levels::by_name).filter(|l| l.layout.is_some());
    let mut b = a.clone();
    if let Some(l) = level {
        let packed = levels::packed_repo(repo, l.name);
        let ui = std::cell::RefCell::new(ui::Ui::new());
        let has = matches!(store::hub_plan_packed(&packed, l.name, &ui), Ok(Some(_)));
        if !has {
            eprintln!("glyd: no pre-packed {} download of {repo}: downloading its own weights, which the first run packs at that level.", l.name);
            b.pos[1] = repo.to_string();
        }
    }
    crate::cli::fetch(&b)
}

/// `glyd rm NAME[:LEVEL] | ORG/NAME`: the model's copies out of the Hub's cache and the packed copies made from them. A name with no level takes every level of the model.
pub fn rm(a: &Args) -> i32 {
    rm_in(a, &Manifest::load(), &crate::hub::hub_cache(), &store::packed_root())
}

/// `rm` over the Hub's cache `cache` and the packed copies in `packed_root`.
fn rm_in(a: &Args, m: &Manifest, cache: &Path, packed_root: &Path) -> i32 {
    let Ok(arg) = a.need(1, "MODEL") else { return 1 };
    let (repo, tag) = levels::split_tag(arg);
    let level = tag.as_deref().and_then(levels::by_name);
    if tag.is_some() && level.is_none() {
        eprintln!("glyd rm: :{} is not a level. The levels: {}.", tag.unwrap_or_default(), levels::names_now().join(", "));
        return 1;
    }
    if !store::looks_like_repo(repo) {
        eprintln!("glyd rm: {repo}: a model name or a Hub id (ORG/NAME); glyd ls shows what is downloaded");
        return 1;
    }
    // the repos to take out: the one named, at its level (penguin: the release's own; another: its pre-packed repo), or without a level all of them
    let mut repos: Vec<String> = Vec::new();
    match level {
        Some(l) if l.layout.is_some() => repos.push(levels::packed_repo(repo, l.name)),
        Some(_) => repos.push(repo.to_string()),
        None => {
            repos.push(repo.to_string());
            repos.extend(levels::LEVELS.iter().filter(|l| l.layout.is_some()).map(|l| levels::packed_repo(repo, l.name)));
        }
    }
    let local = scan(cache, m);
    let mut removed = 0u64;
    let mut found = false;
    for r in &repos {
        let dir = cache.join(dir_of_repo(r));
        // (only a directory of the Hub's cache for this very repo: never a path made of the argument)
        if local.iter().any(|l| &l.repo == r) && dir.parent() == Some(cache) {
            let bytes = local.iter().find(|l| &l.repo == r).map_or(0, |l| l.bytes);
            if let Err(e) = std::fs::remove_dir_all(&dir) {
                eprintln!("glyd rm: {}: {e}", dir.display());
                return 1;
            }
            println!("Removed {r} ({})", size(bytes));
            removed += bytes;
            found = true;
        }
        for (p, bytes) in packed_of(packed_root, r) {
            if p.parent() == Some(packed_root) && std::fs::remove_dir_all(&p).is_ok() {
                println!("Removed the packed copy {} ({})", p.file_name().unwrap_or_default().to_string_lossy(), size(bytes));
                removed += bytes;
                found = true;
            }
        }
    }
    if !found {
        eprintln!("glyd rm: {arg} is not downloaded (glyd ls shows what is)");
        return 1;
    }
    println!("{} freed", size(removed));
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> Manifest {
        Manifest::parse(EMBEDDED).unwrap()
    }

    #[test]
    fn the_manifest_in_the_repository_is_well_formed() {
        let m = manifest();
        assert!(m.families.len() >= 6);
        // every family has its default, every repo is a Hub id, no name is another's
        for f in &m.families {
            assert_eq!(f.models.iter().filter(|e| e.default).count(), 1, "{}", f.name);
            for e in &f.models {
                assert!(store::looks_like_repo(&e.repo), "{}", e.repo);
                // a size is never a level (letters only would be read as one)
                assert!(!e.size.bytes().all(|b| b.is_ascii_alphabetic()) || e.size.len() < 3 || levels::by_name(&e.size).is_none(), "{}:{}", f.name, e.size);
                assert!(levels::split_tag(&format!("{}:{}", f.name, e.size)).1.is_none(), "{}:{} would be read as a level", f.name, e.size);
            }
        }
    }

    #[test]
    fn manifests_that_are_wrong_are_refused_with_the_reason() {
        let bad = |s: &str| Manifest::parse(s).unwrap_err();
        assert!(bad("{").contains("not JSON"));
        assert!(bad("{\"families\":[]}").contains("version"));
        assert!(bad("{\"version\":2,\"families\":[]}").contains("update glyd"));
        let fam = |models: &str| format!("{{\"version\":1,\"families\":[{{\"name\":\"x\",\"models\":{models}}}]}}");
        assert!(bad(&fam("[]")).contains("exactly one"));
        assert!(bad(&fam("[{\"size\":\"1b\",\"repo\":\"a/b\"}]")).contains("exactly one"));
        assert!(bad(&fam("[{\"size\":\"1b\",\"repo\":\"ab\",\"default\":true}]")).contains("not a Hub id"));
        assert!(bad(&fam("[{\"size\":\"1b\",\"repo\":\"a/b\",\"default\":true},{\"size\":\"1B\",\"repo\":\"a/c\"}]")).contains("twice"));
        assert!(bad("{\"version\":1,\"families\":[{\"name\":\"X\",\"models\":[{\"size\":\"1b\",\"repo\":\"a/b\",\"default\":true}]},{\"name\":\"x\",\"models\":[{\"size\":\"1b\",\"repo\":\"a/b\",\"default\":true}]}]}").contains("taken twice"));
    }

    #[test]
    fn a_name_without_a_size_is_the_familys_default() {
        let m = manifest();
        let r = resolve("qwen3.5", &m).unwrap().unwrap();
        assert_eq!((r.name.as_str(), r.repo.as_str(), r.level.as_deref()), ("qwen3.5:9b", "Qwen/Qwen3.5-9B", None));
        assert_eq!(resolve("qwen3", &m).unwrap().unwrap().repo, "Qwen/Qwen3-8B");
        assert_eq!(resolve("gemma4", &m).unwrap().unwrap().repo, "google/gemma-4-12B-it");
    }

    #[test]
    fn sizes_and_levels_are_told_apart() {
        let m = manifest();
        let r = resolve("qwen3.5:27b", &m).unwrap().unwrap();
        assert_eq!((r.repo.as_str(), r.level.as_deref()), ("Qwen/Qwen3.5-27B", None));
        let r = resolve("qwen3.5:27b:swift", &m).unwrap().unwrap();
        assert_eq!((r.repo.as_str(), r.level.as_deref(), r.model().as_str()), ("Qwen/Qwen3.5-27B", Some("swift"), "Qwen/Qwen3.5-27B:swift"));
        // letters only after the name is a level; a size has a digit
        let r = resolve("qwen3.5:swift", &m).unwrap().unwrap();
        assert_eq!((r.name.as_str(), r.level.as_deref()), ("qwen3.5:9b", Some("swift")));
        // a size with a dash is one size, and the part before the dash finds it where only one starts so
        assert_eq!(resolve("qwen3.5:35b-a3b", &m).unwrap().unwrap().repo, "Qwen/Qwen3.5-35B-A3B");
        assert_eq!(resolve("qwen3.5:35b", &m).unwrap().unwrap().repo, "Qwen/Qwen3.5-35B-A3B");
        assert_eq!(resolve("qwen3.5:35b-a3b:kestrel", &m).unwrap().unwrap().level.as_deref(), Some("kestrel"));
        // case, dashes and aliases
        for name in ["Qwen3.5", "qwen-3.5", "QWEN_3.5:9B", "qwen35"] {
            assert_eq!(resolve(name, &m).unwrap().unwrap().name, "qwen3.5:9b", "{name}");
        }
        assert_eq!(resolve("nemotron", &m).unwrap().unwrap().repo, "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16");
    }

    #[test]
    fn hub_ids_paths_and_ggufs_are_not_names() {
        let m = manifest();
        for arg in
            ["Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-9B:swift", "glyd/Qwen3.5-9B-swift", "./qwen3.5", "/models/qwen3.5", "~/m", "model.gguf", "C:\\models\\m", "dir/"]
        {
            assert_eq!(resolve(arg, &m), Ok(None), "{arg}");
        }
        // an existing directory is a directory even where its name is a family's
        assert_eq!(resolve_with("qwen3", &m, &|p| p == "qwen3"), Ok(None));
        assert!(resolve_with("qwen3", &m, &|_| false).unwrap().is_some());
    }

    #[test]
    fn what_is_wrong_is_said_with_the_choices() {
        let m = manifest();
        let e = resolve("qwen3.5:99b", &m).unwrap_err();
        assert!(e.contains("no size 99b") && e.contains("9b") && e.contains("27b"), "{e}");
        let e = resolve("qwen3.5:9b:fast", &m).unwrap_err();
        assert!(e.contains(":fast is not a level") && e.contains("penguin"), "{e}");
        let e = resolve("qwem3.5", &m).unwrap_err();
        assert!(e.contains("Did you mean qwen3.5?") && e.contains("qwen3.5 (0.8b"), "{e}");
        let e = resolve("llama", &m).unwrap_err();
        assert!(e.contains("no such model name") && e.contains("ORG/NAME") && !e.contains("Did you mean"), "{e}");
    }

    #[test]
    fn resolve_args_changes_the_model_word_of_the_commands_that_take_one() {
        let words = |v: &[&str]| Args::from(v.iter().map(|s| s.to_string()));
        let mut a = words(&["pull", "qwen3.5:27b:swift", "--hf-token", "x"]);
        resolve_args(&mut a).unwrap();
        assert_eq!(a.pos, ["pull", "Qwen/Qwen3.5-27B:swift"]);
        assert_eq!(a.flag("hf-token"), Some("x"));
        let mut a = words(&["run", "Qwen/Qwen3-8B"]);
        resolve_args(&mut a).unwrap();
        assert_eq!(a.pos[1], "Qwen/Qwen3-8B");
        let mut a = words(&["doctor", "qwen3"]); // (a command with no model)
        resolve_args(&mut a).unwrap();
        assert_eq!(a.pos[1], "qwen3");
        assert!(resolve_args(&mut words(&["run", "nothing-here"])).is_err());
        // verify takes a file
        resolve_args(&mut words(&["verify", "nothing-here"])).unwrap();
    }

    /// A Hub cache with the given repos in it: each a `models--ORG--NAME` directory with a blob of `bytes` bytes and a snapshot that links to it.
    fn fake_cache(name: &str, repos: &[(&str, usize)]) -> PathBuf {
        let d = std::env::temp_dir().join(format!("glyd-models-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        for (repo, bytes) in repos {
            let r = d.join(dir_of_repo(repo));
            std::fs::create_dir_all(r.join("blobs")).unwrap();
            std::fs::create_dir_all(r.join("snapshots/abc")).unwrap();
            std::fs::write(r.join("blobs/aa11"), vec![0u8; *bytes]).unwrap();
            #[cfg(unix)]
            std::os::unix::fs::symlink("../../blobs/aa11", r.join("snapshots/abc/model.safetensors")).unwrap();
        }
        std::fs::create_dir_all(d.join("datasets--a--b")).unwrap();
        d
    }

    #[test]
    fn the_cache_is_read_as_huggingface_hub_keeps_it() {
        let m = manifest();
        let d = fake_cache("scan", &[("Qwen/Qwen3.5-9B", 3000), ("glyd/Qwen3.5-9B-swift", 1000), ("someone/else", 10), ("glyd/Qwen3.5-27B-wren", 5)]);
        let got = scan(&d, &m);
        let row = |r: &str| got.iter().find(|l| l.repo == r).unwrap();
        // a blob is counted once, though the snapshot links to it
        assert_eq!(
            (row("Qwen/Qwen3.5-9B").bytes, row("Qwen/Qwen3.5-9B").level.as_str(), row("Qwen/Qwen3.5-9B").name.as_deref()),
            (3000, "penguin", Some("qwen3.5:9b"))
        );
        let s = row("glyd/Qwen3.5-9B-swift");
        assert_eq!((s.bytes, s.level.as_str(), s.name.as_deref()), (1000, "swift", Some("qwen3.5:9b")));
        assert_eq!(row("someone/else").name, None);
        assert_eq!(row("glyd/Qwen3.5-27B-wren").name.as_deref(), Some("qwen3.5:27b"));
        assert_eq!(got.len(), 4, "a dataset is not a model");
        std::fs::remove_dir_all(d).ok();
    }

    #[test]
    fn a_packed_copy_belongs_to_exactly_its_repo() {
        // store::packed_name: ORG--NAME, the revision's first eight characters where it has one, the layout
        assert!(is_packed_of("Qwen--Qwen3-8B-0123abcd-mma", "Qwen/Qwen3-8B"));
        assert!(is_packed_of("Qwen--Qwen3-8B-0123abcd-i5", "Qwen/Qwen3-8B"));
        assert!(is_packed_of("Qwen--Qwen3-8B-a", "Qwen/Qwen3-8B"));
        assert!(is_packed_of("glyd--Qwen3.5-9B-swift-deadbeef-i5", "glyd/Qwen3.5-9B-swift"));
        // another model whose name starts with this one's
        assert!(!is_packed_of("Qwen--Qwen3-8B-FP8-0123abcd-mma", "Qwen/Qwen3-8B"));
        assert!(!is_packed_of("Qwen--Qwen3-80B-0123abcd-mma", "Qwen/Qwen3-8B"));
        assert!(!is_packed_of("Qwen--Qwen3-8B", "Qwen/Qwen3-8B"));
        assert!(!is_packed_of(".packing-123", "Qwen/Qwen3-8B"));
    }

    #[test]
    fn rm_takes_out_the_model_its_levels_and_its_packed_copies_and_nothing_else() {
        let m = manifest();
        let cache = fake_cache("rm", &[("Qwen/Qwen3-8B", 100), ("glyd/Qwen3-8B-swift", 50), ("Qwen/Qwen3-32B", 7)]);
        let packed = std::env::temp_dir().join(format!("glyd-models-{}-rm-packed", std::process::id()));
        let _ = std::fs::remove_dir_all(&packed);
        for d in ["Qwen--Qwen3-8B-0123abcd-mma", "Qwen--Qwen3-8B-FP8-0123abcd-mma", "Qwen--Qwen3-32B-0123abcd-mma"] {
            std::fs::create_dir_all(packed.join(d)).unwrap();
            std::fs::write(packed.join(d).join("engine.safetensors"), [0u8; 4]).unwrap();
        }
        let rm = |v: &[&str]| rm_in(&Args::from(v.iter().map(|s| s.to_string())), &m, &cache, &packed);
        assert_eq!(rm(&["rm", "Qwen/Qwen3-8B:swift"]), 0);
        assert!(!cache.join("models--glyd--Qwen3-8B-swift").exists() && cache.join("models--Qwen--Qwen3-8B").exists());
        assert_eq!(rm(&["rm", "Qwen/Qwen3-8B"]), 0);
        assert!(!cache.join("models--Qwen--Qwen3-8B").exists());
        assert!(cache.join("models--Qwen--Qwen3-32B").exists(), "another model's files stay");
        assert!(!packed.join("Qwen--Qwen3-8B-0123abcd-mma").exists());
        assert!(
            packed.join("Qwen--Qwen3-8B-FP8-0123abcd-mma").exists() && packed.join("Qwen--Qwen3-32B-0123abcd-mma").exists(),
            "the packed copies of other models stay"
        );
        // what is not there, and what is no model
        assert_eq!(rm(&["rm", "Qwen/Qwen3-8B"]), 1);
        assert_eq!(rm(&["rm", "../etc"]), 1);
        assert_eq!(rm(&["rm", "Qwen/Qwen3-8B:fast"]), 1);
        assert_eq!(rm(&["rm"]), 1);
        std::fs::remove_dir_all(cache).ok();
        std::fs::remove_dir_all(packed).ok();
    }
}
