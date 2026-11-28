//! Set, or check, the version in every file that carries it.
//!
//!     cargo xtask bump-version 0.26.0            # a release: the code's versions, the docs', the CHANGELOG heading dated
//!     cargo xtask bump-version 0.26.0rc3         # a pre-release: the code's versions only (the docs stay at the last release)
//!     cargo xtask bump-version --check           # every file says the same version (ci.yml)
//!     cargo xtask bump-version --check v0.26.0   # ... and it is this tag's (release.yml, before anything is built)
//!     cargo xtask bump-version --files           # the files that carry it, one a line (scripts/release_ready.sh)
//!     cargo xtask bump-version --py v0.26.0-rc.3 # the version in Python's spelling: 0.26.0rc3
//!
//! `--root DIR` works on the tree at DIR instead of this checkout's (release_ready.sh: a commit's own files); `--date YYYY-MM-DD` is a release's date (default: today's, here).
//!
//! PEP 440 (Python: 0.26.0rc3) and semver (Cargo: 0.26.0-rc.3) spell a pre-release differently and name the same version. The argument or tag may be in either
//! spelling (a leading v is allowed); each file gets its own ecosystem's, and --check fails on a file in the other's. Only X.Y.Z and its alpha, beta and rc
//! pre-releases (X.Y.Zrc3, X.Y.Za1, X.Y.Zb2) exist here.
//!
//! What carries it: the four Cargo.toml (and glyd-store's dependency on glyd, and glyd's on glyd-engine), the four glyd entries in Cargo.lock, pyproject.toml (and its
//! extras' pins of glyd-gpu), glyd/__init__.py, glyd-gpu-py's pyproject.toml and glyd_gpu/__init__.py and scripts/install.sh and scripts/install.ps1 (the release they
//! install), in every version. In a release (X.Y.Z) also ROADMAP.md and docs/index.html, where they name the current release, and CHANGELOG.md's heading
//! (`## Unreleased` or `## vX.Y.Z (Unreleased)` becomes `## vX.Y.Z` with the date), with the entries waiting in changelog.d/ (one file per pull request) put under
//! it, newest first, and their files removed. Formula/glyd.rb is not here: it needs the release's files to exist.
//!
//! This is scripts/bump_version.py in Rust (the script through v0.35.0): the files it writes are that script's byte for byte, and the tests hold the outputs
//! recorded from it (tests/bump/). The patterns below are its regular expressions, written out as text matching; `pat` keeps each expression for the messages.

use crate::{fail, py, usage, Res};
use std::path::Path;
use std::process::Command;

const DASH: &str = "—";
const CHANGELOG: &str = "CHANGELOG.md";
const FRAGMENTS: &str = "changelog.d"; // one file per pull request (FRAGMENTS/README.md says how); a release puts them in CHANGELOG.md

/// X.Y.Z (as written, without leading zeros) and a pre-release's letter (a, b, rc) and number.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Version {
    n: [String; 3],
    pre: Option<(&'static str, String)>,
}

/// `^HEAD(?P<v>[^STOP]+)TAIL` on any line (re.M): the version is what follows one of `heads` at a line's start, up to `stop`, with `tail` after it.
struct Code {
    path: &'static str,
    py: bool,
    pat: &'static str,
    heads: &'static [&'static str],
    stop: char,
    tail: &'static str,
    n: usize,
}

const fn code(path: &'static str, py: bool, pat: &'static str, heads: &'static [&'static str], stop: char, tail: &'static str, n: usize) -> Code {
    Code { path, py, pat, heads, stop, tail, n }
}

const CARGO_VERSION: &str = r#"^version = "(?P<v>[^"]+)""#;

// (file, ecosystem, the pattern as the messages name it, its text, matches expected)
const CODE: &[Code] = &[
    code("Cargo.toml", false, CARGO_VERSION, &["version = \""], '"', "\"", 1),
    code("glyd-store/Cargo.toml", false, CARGO_VERSION, &["version = \""], '"', "\"", 1),
    code("glyd-store/Cargo.toml", false, r#"^glyd = \{ version = "(?P<v>[^"]+)""#, &["glyd = { version = \""], '"', "\"", 1),
    // (the engine linked into glyd: cargo-deny wants a version on a path dependency of a public crate)
    code(
        "Cargo.toml",
        false,
        r#"^glyd-engine = \{ path = "glyd-engine", version = "(?P<v>[^"]+)""#,
        &["glyd-engine = { path = \"glyd-engine\", version = \""],
        '"',
        "\"",
        1,
    ),
    code("glyd-gpu/Cargo.toml", false, CARGO_VERSION, &["version = \""], '"', "\"", 1),
    code("glyd-engine/Cargo.toml", false, CARGO_VERSION, &["version = \""], '"', "\"", 1), // the engine ships in the glyd-gpu wheels, as glyd-engine's own version
    code(
        "Cargo.lock",
        false,
        r#"^name = "glyd(?:-engine|-gpu|-store)?"\nversion = "(?P<v>[^"]+)""#,
        &["name = \"glyd-engine\"\nversion = \"", "name = \"glyd-gpu\"\nversion = \"", "name = \"glyd-store\"\nversion = \"", "name = \"glyd\"\nversion = \""],
        '"',
        "\"",
        4,
    ),
    code("bindings/python/pyproject.toml", true, CARGO_VERSION, &["version = \""], '"', "\"", 1),
    // the extras pin the glyd-gpu of the same version
    code("bindings/python/pyproject.toml", true, r#"^gpu = \["glyd-gpu\[gpu\]==(?P<v>[^"]+)"\]"#, &["gpu = [\"glyd-gpu[gpu]=="], '"', "\"]", 1),
    code("bindings/python/pyproject.toml", true, r#"^vllm = \["glyd-gpu\[vllm\]==(?P<v>[^"]+)"\]"#, &["vllm = [\"glyd-gpu[vllm]=="], '"', "\"]", 1),
    code("bindings/python/glyd/__init__.py", true, r#"^__version__ = "(?P<v>[^"]+)""#, &["__version__ = \""], '"', "\"", 1),
    code("glyd-gpu-py/pyproject.toml", true, CARGO_VERSION, &["version = \""], '"', "\"", 1), // glyd-gpu: the same version as glyd
    code("glyd-gpu-py/glyd_gpu/__init__.py", true, r#"^__version__ = "(?P<v>[^"]+)""#, &["__version__ = \""], '"', "\"", 1),
    // the release the installer pins
    code("scripts/install.sh", true, r#"^GLYD_VERSION="\$\{GLYD_VERSION:-(?P<v>[^}]+)\}""#, &["GLYD_VERSION=\"${GLYD_VERSION:-"], '}', "}\"", 1),
    // the Windows installer's pin
    code(
        "scripts/install.ps1",
        true,
        r"^\$GlydVersion = if \(\$env:GLYD_VERSION\) \{ \$env:GLYD_VERSION \} else \{ '(?P<v>[^']+)' \}",
        &["$GlydVersion = if ($env:GLYD_VERSION) { $env:GLYD_VERSION } else { '"],
        '\'',
        "' }",
        1,
    ),
];

/// Where a release is named in prose.
#[derive(Clone, Copy)]
enum Doc {
    /// ROADMAP.md's line `## Where it stands (vX.Y.Z, YYYY-MM-DD)`
    Roadmap,
    /// docs/index.html's `Rust, C ABI, CLI; vX.Y.Z, Month YYYY.`
    Index,
}

// (file, the pattern as the messages name it, which text)
const DOCS: &[(&str, &str, Doc)] = &[
    ("ROADMAP.md", r"^## Where it stands \(v(?P<v>[\d.]+), \d{4}-\d{2}-\d{2}\)$", Doc::Roadmap),
    ("docs/index.html", r"Rust, C ABI, CLI; v(?P<v>[\d.]+), [A-Z][a-z]+ \d{4}\.", Doc::Index),
];

const MONTHS: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

/// A day of the calendar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Date {
    y: u32,
    m: u32,
    d: u32,
}

impl Date {
    /// YYYY-MM-DD, a day that exists.
    pub fn parse(s: &str) -> Option<Date> {
        let b = s.as_bytes();
        if b.len() != 10 || b[4] != b'-' || b[7] != b'-' || !is_date(s) {
            return None;
        }
        let (y, m, d): (u32, u32, u32) = (s[..4].parse().ok()?, s[5..7].parse().ok()?, s[8..].parse().ok()?);
        let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
        let days = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
        (y >= 1 && (1..=12).contains(&m) && d >= 1 && d <= days[m as usize - 1]).then_some(Date { y, m, d })
    }

    /// Today here (the machine's own time zone, as the `date` program has it; UTC where there is none).
    pub fn today() -> Date {
        let local = Command::new("date").arg("+%Y-%m-%d").output().ok().and_then(|o| Date::parse(String::from_utf8_lossy(&o.stdout).trim()));
        local.unwrap_or_else(|| {
            // (days since 1970-01-01 to a civil date)
            let days = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() / 86400).unwrap_or(0) as i64;
            let z = days + 719468;
            let (era, doe) = (z.div_euclid(146097), z.rem_euclid(146097));
            let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
            let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
            let mp = (5 * doy + 2) / 153;
            let (d, m) = (doy - (153 * mp + 2) / 5 + 1, if mp < 10 { mp + 3 } else { mp - 9 });
            Date { y: (yoe + era * 400 + i64::from(m <= 2)) as u32, m: m as u32, d: d as u32 }
        })
    }

    fn iso(self) -> String {
        format!("{:04}-{:02}-{:02}", self.y, self.m, self.d)
    }

    fn month(self) -> String {
        format!("{} {}", MONTHS[self.m as usize - 1], self.y)
    }
}

fn digits(s: &str) -> usize {
    s.bytes().take_while(u8::is_ascii_digit).count()
}

/// `\d{4}-\d{2}-\d{2}`, all of s.
fn is_date(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 10 && b.iter().enumerate().all(|(i, c)| if i == 4 || i == 7 { *c == b'-' } else { c.is_ascii_digit() })
}

/// 'v0.26.0-rc.2', '0.26.0rc2' -> 0.26.0 rc 2; '0.26.0' -> 0.26.0.
pub fn parse(s: &str) -> Res<Version> {
    fn number(r: &mut &str) -> Option<String> {
        let k = digits(r);
        let n = r[..k].trim_start_matches('0');
        let n = if n.is_empty() { "0" } else { n }.to_string();
        *r = &r[k..];
        (k > 0).then_some(n)
    }
    fn version(s: &str) -> Option<Version> {
        let mut r = s.strip_prefix('v').unwrap_or(s);
        let x = number(&mut r)?;
        r = r.strip_prefix('.')?;
        let y = number(&mut r)?;
        r = r.strip_prefix('.')?;
        let z = number(&mut r)?;
        let n = [x, y, z];
        if r.is_empty() {
            return Some(Version { n, pre: None });
        }
        let sep = |r: &str| -> usize { usize::from(r.starts_with(['-', '.'])) };
        r = &r[sep(r)..];
        // (the longer name first: `a` alone leaves `lpha`)
        let (name, short) = [("alpha", "a"), ("a", "a"), ("beta", "b"), ("b", "b"), ("rc", "rc")].into_iter().find(|(name, _)| {
            let t = &r[if r.starts_with(name) { name.len() } else { return false }..];
            let t = &t[sep(t)..];
            !t.is_empty() && digits(t) == t.len()
        })?;
        r = &r[name.len()..];
        r = &r[sep(r)..];
        Some(Version { n, pre: Some((short, number(&mut r)?)) })
    }
    version(s).ok_or_else(|| fail(format!("{} is not a version here: X.Y.Z, or X.Y.Z with rcN, aN or bN", py::repr(s))))
}

/// The version as Python (PEP 440: 0.26.0rc3) or Cargo (semver: 0.26.0-rc.3) spells it.
pub fn spell(v: &Version, py: bool) -> String {
    let s = v.n.join(".");
    match &v.pre {
        None => s,
        Some((letter, n)) if py => format!("{s}{letter}{n}"),
        Some((letter, n)) => format!(
            "{s}-{}.{n}",
            match *letter {
                "a" => "alpha",
                "b" => "beta",
                _ => "rc",
            }
        ),
    }
}

impl Code {
    /// Where the version is in text, for every match: non-overlapping, in order.
    fn find(&self, text: &str) -> Vec<(usize, usize)> {
        let (mut found, mut from, mut line) = (Vec::new(), 0, 0);
        loop {
            if line >= from {
                for head in self.heads {
                    let Some(rest) = text[line..].strip_prefix(head) else { continue };
                    let start = line + head.len();
                    let Some(len) = rest.find(self.stop) else { continue };
                    if len > 0 && text[start + len..].starts_with(self.tail) {
                        found.push((start, start + len));
                        from = start + len + self.tail.len();
                        break;
                    }
                }
            }
            match text[line..].find('\n') {
                Some(i) => line += i + 1,
                None => return found,
            }
        }
    }

    /// text with the version of every match replaced by new; and the number of matches.
    fn swap(&self, text: &str, new: &str) -> (String, usize) {
        let spans = self.find(text);
        let (mut out, mut at) = (String::with_capacity(text.len()), 0);
        for (a, b) in &spans {
            out += &text[at..*a];
            out += new;
            at = *b;
        }
        out += &text[at..];
        (out, spans.len())
    }
}

impl Doc {
    /// The matches in text: (start, end, the version's start and end).
    fn find(self, text: &str) -> Vec<[usize; 4]> {
        let class = |s: &str| s.bytes().take_while(|c| c.is_ascii_digit() || *c == b'.').count();
        let mut found = Vec::new();
        match self {
            Doc::Roadmap => {
                const HEAD: &str = "## Where it stands (v";
                let mut line = 0;
                for l in text.split('\n') {
                    if let Some(r) = l.strip_prefix(HEAD) {
                        let k = class(r);
                        let date = r[k..].strip_prefix(", ").and_then(|t| t.strip_suffix(')'));
                        if k > 0 && date.is_some_and(is_date) {
                            found.push([line, line + l.len(), line + HEAD.len(), line + HEAD.len() + k]);
                        }
                    }
                    line += l.len() + 1;
                }
            }
            Doc::Index => {
                const HEAD: &str = "Rust, C ABI, CLI; v";
                let mut at = 0;
                while let Some(i) = text[at..].find(HEAD) {
                    let (start, v) = (at + i, at + i + HEAD.len());
                    let k = class(&text[v..]);
                    // `, [A-Z][a-z]+ \d{4}\.`
                    let end = text[v + k..].strip_prefix(", ").filter(|_| k > 0).and_then(|t| {
                        let b = t.as_bytes();
                        let low = b.iter().skip(1).take_while(|c| c.is_ascii_lowercase()).count();
                        let year = b.get(low + 2..low + 6)?;
                        (b[0].is_ascii_uppercase() && low > 0 && b[low + 1] == b' ' && year.iter().all(u8::is_ascii_digit) && b.get(low + 6) == Some(&b'.'))
                            .then_some(v + k + 2 + low + 7)
                    });
                    match end {
                        Some(end) => {
                            found.push([start, end, v, v + k]);
                            at = end;
                        }
                        None => at = start + 1,
                    }
                }
            }
        }
        found
    }

    /// What a release writes there.
    fn text(self, v: &str, date: Date) -> String {
        match self {
            Doc::Roadmap => format!("## Where it stands (v{v}, {})", date.iso()),
            Doc::Index => format!("Rust, C ABI, CLI; v{v}, {}.", date.month()),
        }
    }
}

fn read(root: &Path, path: &str) -> Res<String> {
    py::read(&root.join(path))
}

fn current(root: &Path) -> Res<Version> {
    let c = &CODE[0];
    let text = read(root, c.path)?;
    match c.find(&text).first() {
        Some((a, b)) => parse(&text[*a..*b]),
        None => Err(fail(format!("{}: no version", c.path))),
    }
}

fn out_of_date(path: &str, k: usize, pat: &str, n: usize) -> String {
    format!("{path}: {k} matches of {}, expected {n} (this script's table is out of date)", py::repr(pat))
}

/// What is wrong with the versions under root: every file must say `want`, in its own spelling.
pub fn problems(root: &Path, want: &Version) -> Res<Vec<String>> {
    let mut bad = Vec::new();
    for c in CODE {
        let text = read(root, c.path)?;
        let vs: Vec<&str> = c.find(&text).into_iter().map(|(a, b)| &text[a..b]).collect();
        if vs.len() != c.n {
            bad.push(out_of_date(c.path, vs.len(), c.pat, c.n));
        }
        let want = spell(want, c.py);
        bad.extend(vs.iter().filter(|v| **v != want).map(|v| format!("{}: {v}, expected {want}", c.path)));
    }
    if want.pre.is_none() {
        // a release: the docs name it
        let v = spell(want, true);
        for (path, _, doc) in DOCS {
            let text = read(root, path)?;
            match doc.find(&text).first() {
                Some([_, _, a, b]) if text[*a..*b] == v => {}
                Some([a, b, ..]) => bad.push(format!("{path}: does not name v{v} ({})", &text[*a..*b])),
                None => bad.push(format!("{path}: does not name v{v} (the text this script looks for is gone)")),
            }
        }
        let head = format!("## v{v} {DASH} ");
        if !read(root, CHANGELOG)?.split('\n').any(|l| l.strip_prefix(&head).is_some_and(is_date)) {
            bad.push(format!("{CHANGELOG}: no heading '## v{v} {DASH} DATE' (still Unreleased?)"));
        }
    }
    Ok(bad)
}

/// changelog.d's entries, newest first (by the commit that added each; by name outside git or for a tie).
fn fragments(root: &Path) -> Vec<String> {
    let mut names: Vec<String> = match std::fs::read_dir(root.join(FRAGMENTS)) {
        Ok(dir) => dir.filter_map(|e| e.ok()?.file_name().into_string().ok()).filter(|n| n.ends_with(".md") && n != "README.md").collect(),
        Err(_) => Vec::new(),
    };
    let added = |name: &str| -> i64 {
        let log =
            Command::new("git").arg("-C").arg(root).args(["log", "--diff-filter=A", "--format=%ct", "-1", "--"]).arg(format!("{FRAGMENTS}/{name}")).output();
        match log {
            Ok(o) if o.status.success() => match String::from_utf8_lossy(&o.stdout).trim() {
                "" => 0,
                t => t.parse().unwrap_or(0),
            },
            _ => 0,
        }
    };
    let mut keyed: Vec<(i64, String)> = names.drain(..).map(|n| (-added(&n), n)).collect();
    keyed.sort();
    keyed.into_iter().map(|(_, n)| n).collect()
}

/// CHANGELOG text with the fragments' entries put first under its `## Unreleased`-style heading (the first `## `).
fn gather(root: &Path, text: String, files: &[String]) -> Res<String> {
    let mut entries = String::new();
    for f in files {
        let entry = read(root, &format!("{FRAGMENTS}/{f}"))?;
        if !py::strip(&entry).is_empty() {
            entries += py::strip(&entry);
            entries.push('\n');
        }
    }
    if entries.is_empty() {
        return Ok(text);
    }
    // `^## .*\n\n?`: the first heading line that ends, and the empty line after it
    let (mut line, mut end) = (0, None);
    while end.is_none() {
        let Some(i) = text[line..].find('\n') else { break };
        if text[line..].starts_with("## ") {
            end = Some(line + i + 1 + usize::from(text[line + i + 1..].starts_with('\n')));
        }
        line += i + 1;
    }
    let end = end.ok_or_else(|| fail(format!("{CHANGELOG}: no '## ' heading to put changelog.d's entries under")))?;
    Ok(format!("{}{entries}\n{}", &text[..end], &text[end..]))
}

/// Set every file to version (a release: dated `date`); the files changed.
pub fn bump(root: &Path, want: &Version, date: Date) -> Res<Vec<String>> {
    let mut files: Vec<(&str, String)> = Vec::new();
    for c in CODE {
        let at = match files.iter().position(|(p, _)| *p == c.path) {
            Some(at) => at,
            None => {
                files.push((c.path, read(root, c.path)?));
                files.len() - 1
            }
        };
        let (text, k) = c.swap(&files[at].1, &spell(want, c.py));
        if k != c.n {
            return Err(fail(out_of_date(c.path, k, c.pat, c.n)));
        }
        files[at].1 = text;
    }
    let mut frags = Vec::new();
    if want.pre.is_none() {
        let v = spell(want, true);
        for (path, pat, doc) in DOCS {
            let text = read(root, path)?;
            let found = doc.find(&text);
            if found.len() != 1 {
                return Err(fail(format!("{path}: {} matches of {}, expected 1 (this script's table is out of date)", found.len(), py::repr(pat))));
            }
            let [a, b, ..] = found[0];
            files.push((path, format!("{}{}{}", &text[..a], doc.text(&v, date), &text[b..])));
        }
        // `## Unreleased`, `## vX.Y.Z (Unreleased)` or `## vX.Y.Z — DATE`: the release's heading, dated
        let (heading, dated, mut k) = (format!("## v{v} {DASH} {}", date.iso()), format!("## v{v} {DASH} "), 0);
        let old = read(root, CHANGELOG)?;
        let lines: Vec<&str> = old
            .split('\n')
            .map(|l| {
                let is = l == "## Unreleased"
                    || l.strip_prefix("## v").and_then(|t| t.strip_prefix(v.as_str())) == Some(" (Unreleased)")
                    || l.strip_prefix(&dated).is_some_and(is_date);
                k += usize::from(is);
                if is {
                    heading.as_str()
                } else {
                    l
                }
            })
            .collect();
        if k != 1 {
            return Err(fail(format!("{CHANGELOG}: {k} headings '## Unreleased' or '## v{v} (Unreleased)', expected 1")));
        }
        frags = fragments(root);
        files.push((CHANGELOG, gather(root, lines.join("\n"), &frags)?));
    }
    let mut changed = Vec::new();
    for (p, t) in &files {
        if *t != read(root, p)? {
            changed.push(p.to_string());
        }
    }
    for (p, t) in files.iter().filter(|(p, _)| changed.iter().any(|c| c == p)) {
        py::write(&root.join(p), t)?;
    }
    for f in &frags {
        // their entries are in CHANGELOG.md now
        let path = root.join(FRAGMENTS).join(f);
        std::fs::remove_file(&path).map_err(|e| fail(format!("{}: {e}", path.display())))?;
        changed.push(format!("{FRAGMENTS}/{f}"));
    }
    Ok(changed)
}

/// The files that carry the version, sorted.
fn carriers() -> Vec<&'static str> {
    let mut all: Vec<&str> = CODE.iter().map(|c| c.path).chain(DOCS.iter().map(|d| d.0)).chain([CHANGELOG]).collect();
    all.sort_unstable();
    all.dedup();
    all
}

const USAGE: &str = "usage: cargo xtask bump-version [VERSION] [--check] [--date YYYY-MM-DD] [--root DIR] | --files | --py VERSION";

pub fn main(repo: &Path, args: &[String], out: &mut String) -> Res<()> {
    let (mut version, mut check, mut date, mut root, mut files, mut py_of) = (None, false, None, repo.to_path_buf(), false, None);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let (name, inline) = match a.split_once('=') {
            Some((n, v)) if n.starts_with("--") => (n, Some(v.to_string())),
            _ => (a.as_str(), None),
        };
        let mut value = || inline.clone().or_else(|| it.next().cloned()).ok_or_else(|| usage(format!("{USAGE}\nbump-version: {name} takes a value")));
        match name {
            "--check" => check = true,
            "--files" => files = true,
            "--date" => {
                let d = value()?;
                date = Some(Date::parse(&d).ok_or_else(|| usage(format!("{USAGE}\nbump-version: --date: {} is not a day, YYYY-MM-DD", py::repr(&d))))?);
            }
            "--root" => root = value()?.into(),
            "--py" => py_of = Some(value()?),
            "-h" | "--help" => {
                *out += &format!("{USAGE}\n(xtask/src/bump_version.rs says what each does)\n");
                return Ok(());
            }
            _ if name.starts_with('-') => return Err(usage(format!("{USAGE}\nbump-version: no option {name}"))),
            _ if version.is_some() => return Err(usage(format!("{USAGE}\nbump-version: one version, not two"))),
            _ => version = Some(a.clone()),
        }
    }
    let version = version.filter(|v| !v.is_empty()); // (an empty version is none given, as the script took it)
    if files {
        *out += &(carriers().join("\n") + "\n");
        return Ok(());
    }
    if let Some(v) = py_of {
        *out += &format!("{}\n", spell(&parse(&v)?, true));
        return Ok(());
    }
    if !check && version.is_none() {
        return Err(usage(format!("{USAGE}\nbump-version: give a version to set, or --check")));
    }
    let v = match &version {
        Some(v) => parse(v)?,
        None => current(&root)?,
    };
    let (cargo, python) = (spell(&v, false), spell(&v, true));
    if check {
        let bad = problems(&root, &v)?;
        if !bad.is_empty() {
            return Err(fail(format!("version files disagree:\n  {}\n(cargo xtask bump-version VERSION sets them all)", bad.join("\n  "))));
        }
        *out += &format!("version files agree: Cargo {cargo}, Python {python}{}\n", if v.pre.is_none() { "; the docs name it" } else { "" });
        return Ok(());
    }
    let changed = bump(&root, &v, date.unwrap_or_else(Date::today))?;
    *out += &format!("{} files set to {cargo} (Cargo), {python} (Python): {}\n", changed.len(), changed.join(", "));
    if v.pre.is_none() {
        *out += &format!("the docs and the CHANGELOG heading name it; once the release is up: cargo xtask bump-formula {python}\n");
    } else {
        *out += "a pre-release: the docs stay at the last release\n";
    }
    Ok(())
}

#[cfg(test)]
#[path = "bump_version_tests.rs"]
mod tests;
