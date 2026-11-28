//! bump-version against what scripts/bump_version.py did (the script through v0.35.0), recorded from it once: tests/bump/NAME.txt is one run of the script on the
//! small tree of tests/bump/base.txt with the case's own files over it (IN, or NOT: no such file), and holds its arguments, every file it wrote (OUT) or removed
//! (GONE), what it printed and its exit code. GIT: the tree is a repository, and these files were added by commits at these times, in this order (record.rs: the
//! records' form). The task must do the same, byte for byte; of its messages, only the
//! commands they name differ (cargo xtask, where the script named itself). Exit 2 is the command line's own error, whose text (argparse's) is not compared.

use super::*;
use crate::record;
use std::collections::BTreeMap;
use std::path::PathBuf;

type Files = BTreeMap<String, String>;

#[derive(Default)]
struct Case {
    args: Vec<String>,
    git: Option<Vec<(String, String)>>,
    files: BTreeMap<String, Option<String>>, // IN and NOT
    out: BTreeMap<String, Option<String>>,   // OUT and GONE
    stdout: String,
    stderr: String,
    exit: i32,
}

fn fixtures() -> PathBuf {
    record::folder("bump")
}

fn sections(name: &str) -> Vec<(String, String)> {
    record::read(&fixtures().join(format!("{name}.txt"))).0
}

fn case(name: &str) -> Case {
    let mut c = Case::default();
    for (head, body) in sections(name) {
        let lines = || body.lines().map(str::to_string);
        match head.split_once(' ').unwrap_or((&head, "")) {
            ("ARGS", _) => c.args = lines().collect(),
            ("GIT", _) => c.git = Some(lines().map(|l| l.split_once(' ').map(|(t, p)| (t.to_string(), p.to_string())).unwrap()).collect()),
            ("IN", p) => drop(c.files.insert(p.to_string(), Some(body))),
            ("NOT", p) => drop(c.files.insert(p.to_string(), None)),
            ("OUT", p) => drop(c.out.insert(p.to_string(), Some(body))),
            ("GONE", p) => drop(c.out.insert(p.to_string(), None)),
            ("STDOUT", _) => c.stdout = body,
            ("STDERR", _) => c.stderr = body,
            ("EXIT", _) => c.exit = body.trim().parse().unwrap(),
            _ => panic!("{name}: section {head}"),
        }
    }
    c
}

fn tree(root: &Path) -> Files {
    fn walk(root: &Path, dir: &Path, into: &mut Files) {
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                if p.file_name().unwrap() != ".git" {
                    walk(root, &p, into);
                }
            } else {
                let rel = p.strip_prefix(root).unwrap().components().map(|c| c.as_os_str().to_str().unwrap()).collect::<Vec<_>>().join("/");
                into.insert(rel, String::from_utf8(std::fs::read(&p).unwrap()).unwrap());
            }
        }
    }
    let mut all = Files::new();
    walk(root, root, &mut all);
    all
}

fn git(root: &Path, when: &str, args: &[&str]) {
    let date = format!("{when} +0000");
    let mut run = Command::new("git");
    run.arg("-C").arg(root).args(["-c", "commit.gpgsign=false", "-c", "core.autocrlf=false", "-c", "core.hooksPath=/dev/null"]).args(args);
    let ids = [("GIT_AUTHOR_NAME", "t"), ("GIT_AUTHOR_EMAIL", "t@example.com"), ("GIT_COMMITTER_NAME", "t"), ("GIT_COMMITTER_EMAIL", "t@example.com")];
    let out = run.envs(ids).env("GIT_AUTHOR_DATE", &date).env("GIT_COMMITTER_DATE", &date).output().expect("git runs");
    assert!(out.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&out.stderr));
}

/// The case's tree made in a new folder, the task run on it; what differs from the record.
fn differs(name: &str) -> Vec<String> {
    let c = case(name);
    let mut files: BTreeMap<String, Option<String>> =
        sections("base").into_iter().map(|(head, body)| (head.strip_prefix("IN ").unwrap().to_string(), Some(body))).collect();
    files.extend(c.files.clone());
    let before: Files = files.into_iter().filter_map(|(p, t)| Some((p, t?))).collect();
    let root = std::env::temp_dir().join(format!("xtask-bump-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    for (p, t) in &before {
        std::fs::create_dir_all(root.join(p).parent().unwrap()).unwrap();
        std::fs::write(root.join(p), t).unwrap();
    }
    if let Some(commits) = &c.git {
        git(&root, "1700000000", &["init", "-q"]);
        let first: Vec<&str> = before.keys().map(String::as_str).filter(|p| !commits.iter().any(|(_, later)| later == p)).collect();
        git(&root, "1700000000", &[&["add", "--"], &first[..]].concat());
        git(&root, "1700000000", &["commit", "-q", "-m", "base"]);
        for (when, p) in commits {
            git(&root, when, &["add", "--", p]);
            git(&root, when, &["commit", "-q", "-m", p]);
        }
    }
    let mut stdout = String::new();
    // (the repository given is no tree at all: --root is the one worked on)
    let args: Vec<String> = c.args.iter().cloned().chain(["--root".to_string(), root.to_str().unwrap().to_string()]).collect();
    let done = main(Path::new("/nonexistent"), &args, &mut stdout);
    let after = tree(&root);
    std::fs::remove_dir_all(&root).unwrap();

    let named = |s: &str| {
        s.replace("python3 scripts/bump_version.py", "cargo xtask bump-version").replace("python3 scripts/bump_formula.py", "cargo xtask bump-formula")
    };
    let mut bad = Vec::new();
    let (code, stderr) = match &done {
        Ok(()) => (0, String::new()),
        Err(e) => (e.code, format!("{}\n", e.msg)),
    };
    if code != c.exit {
        bad.push(format!("exit {code} ({stderr:?}), the script's {}", c.exit));
    }
    if c.exit != 2 && (stdout != named(&c.stdout) || stderr != named(&c.stderr)) {
        bad.push(format!("printed {stdout:?} and {stderr:?}, the script {:?} and {:?}", named(&c.stdout), named(&c.stderr)));
    }
    let mut out: BTreeMap<String, Option<String>> = BTreeMap::new();
    for p in before.keys().chain(after.keys()) {
        if before.get(p) != after.get(p) {
            out.insert(p.clone(), after.get(p).cloned());
        }
    }
    if out != c.out {
        let all: std::collections::BTreeSet<&String> = out.keys().chain(c.out.keys()).collect();
        bad.extend(all.into_iter().filter(|p| out.get(*p) != c.out.get(*p)).map(|p| format!("{p}: {:?}, the script's {:?}", out.get(p), c.out.get(p))));
    }
    bad
}

/// Every recorded run of the script.
#[test]
fn does_what_the_script_did() {
    let mut names: Vec<String> = std::fs::read_dir(fixtures()).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
    names.sort();
    let names: Vec<&str> = names.iter().filter_map(|n| n.strip_suffix(".txt")).filter(|n| *n != "base").collect();
    assert!(names.len() >= 50, "{} cases", names.len());
    for kind in ["pre", "release", "release-by-commit", "release-crlf", "release-no-folder", "check-slips", "error-heading-unended"] {
        assert!(names.contains(&kind), "the {kind} case is gone");
    }
    let bad: Vec<String> = names.iter().flat_map(|n| differs(n).into_iter().map(move |b| format!("{n}: {b}"))).collect();
    assert!(bad.is_empty(), "{} differences from the script:\n{}", bad.len(), bad.join("\n"));
}

#[test]
fn versions_in_both_spellings() {
    for (s, same) in
        [("0.26.0rc2", "0.26.0-rc.2"), ("0.26.0rc2", "v0.26.0-rc2"), ("0.26.0a1", "0.26.0-alpha.1"), ("0.26.0", "v0.26.0"), ("0.26.0b2", "0.26.0.beta.2")]
    {
        assert_eq!(parse(s).unwrap(), parse(same).unwrap(), "{s} {same}");
    }
    assert_eq!(spell(&parse("0.26.0-alpha.1").unwrap(), true), "0.26.0a1");
    assert_eq!(spell(&parse("0.26.0b2").unwrap(), false), "0.26.0-beta.2");
    for s in ["0.26", "0.26.0.dev1", "0.26.0rc", "0.26.0-rc.1+x", "0.26.0.post1", "rc1", "", "v", "0.26.0-", "0.26.0rc1 "] {
        assert!(parse(s).is_err(), "{s:?} was taken for a version");
    }
}

#[test]
fn days() {
    assert_eq!(Date::parse("2031-02-03").map(|d| (d.iso(), d.month())), Some(("2031-02-03".to_string(), "February 2031".to_string())));
    assert!(
        Date::parse("2032-02-29").is_some() && Date::parse("2100-02-29").is_none() && Date::parse("2031-13-01").is_none() && Date::parse("2031-2-3").is_none()
    );
    assert!(Date::today().y >= 2026);
}

/// This checkout's own files: they agree, and a pre-release and a release of a copy of them do what the script's own test asked (its --selftest, on the real files).
#[test]
fn this_tree() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let now = current(repo).unwrap();
    assert_eq!(problems(repo, &now).unwrap(), Vec::<String>::new());
    let root = std::env::temp_dir().join(format!("xtask-bump-{}-this-tree", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    for p in carriers() {
        std::fs::create_dir_all(root.join(p).parent().unwrap()).unwrap();
        std::fs::copy(repo.join(p), root.join(p)).unwrap();
    }
    // (whatever the real CHANGELOG says now: every heading a released one, and the release to make on top)
    let log: Vec<String> = read(&root, CHANGELOG)
        .unwrap()
        .split('\n')
        .map(|l| if l.starts_with("## ") && l.ends_with("(Unreleased)") { l.replace(" (Unreleased)", " — 2000-01-01") } else { l.to_string() })
        .collect();
    py::write(&root.join(CHANGELOG), &format!("## v9.9.9 (Unreleased)\n\n{}", log.join("\n").replace("\n## Unreleased\n", "\n## v0.0.0 — 2000-01-01\n")))
        .unwrap();
    let docs: Vec<String> = DOCS.iter().map(|d| read(&root, d.0).unwrap()).collect();
    let day = Date::parse("2031-02-03").unwrap();
    let text = |p: &str| read(&root, p).unwrap();

    bump(&root, &parse("9.9.9rc1").unwrap(), day).unwrap(); // a pre-release: the code only, in each ecosystem's spelling
    assert!(problems(&root, &current(&root).unwrap()).unwrap().is_empty() && problems(&root, &parse("v9.9.9-rc.1").unwrap()).unwrap().is_empty());
    assert!(!problems(&root, &parse("9.9.9rc2").unwrap()).unwrap().is_empty());
    assert!(text("Cargo.toml").contains("version = \"9.9.9-rc.1\"") && text("Cargo.lock").matches("version = \"9.9.9-rc.1\"").count() == 4);
    assert!(
        text("bindings/python/pyproject.toml").contains("gpu = [\"glyd-gpu[gpu]==9.9.9rc1\"]") && text("scripts/install.ps1").contains("else { '9.9.9rc1' }")
    );
    assert!(DOCS.iter().zip(&docs).all(|(d, t)| text(d.0) == *t) && text(CHANGELOG).contains("## v9.9.9 (Unreleased)"));

    std::fs::create_dir_all(root.join(FRAGMENTS)).unwrap();
    for (name, entry) in [("README.md", "how\n"), ("a-thing.md", "- A thing.\n"), ("b-other.md", "- Another thing.\n\n")] {
        py::write(&root.join(FRAGMENTS).join(name), entry).unwrap();
    }
    let changed = bump(&root, &parse("9.9.9").unwrap(), day).unwrap(); // the release: the docs and the heading too, and changelog.d's two entries
    assert_eq!(changed.len(), 16, "{changed:?}");
    assert!(problems(&root, &parse("9.9.9").unwrap()).unwrap().is_empty());
    assert!(text(CHANGELOG).contains("## v9.9.9 — 2031-02-03\n\n- A thing.\n- Another thing.\n\n") && !text(CHANGELOG).contains("(Unreleased)"));
    assert_eq!(fragments(&root), Vec::<String>::new());
    assert!(root.join(FRAGMENTS).join("README.md").exists());
    assert!(text("ROADMAP.md").contains("(v9.9.9, 2031-02-03)") && text("docs/index.html").contains("CLI; v9.9.9, February 2031."));
    assert_eq!(bump(&root, &parse("9.9.9").unwrap(), day).unwrap(), Vec::<String>::new()); // a second run changes nothing
    assert!(!problems(&root, &parse("9.9.10").unwrap()).unwrap().is_empty());
    std::fs::remove_dir_all(&root).unwrap();
}
