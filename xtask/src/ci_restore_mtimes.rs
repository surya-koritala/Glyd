//! Give the files of a fresh checkout the modification times they had when their content last changed.
//!
//! Cargo's freshness check compares source files' modification times with its build products', so a checkout (every file new) rebuilds
//! the whole workspace and reruns every build script, whatever changed. A CI runner that keeps its target directory calls this
//! after the checkout with a file of its own beside that directory: a file whose content (git's blob id) is the one the file
//! remembers gets its old time back (older than the build that used it), any other keeps its new time and is remembered. So only what
//! changed since the last build is built again. (This crate's own sources are new when cargo builds it for this call: it is built again each run, in seconds.)
//!
//!     cargo xtask ci-restore-mtimes /cache/mtimes-sm89.json
//!
//! This is scripts/ci_restore_mtimes.py in Rust (the script through v0.35.0): the same file of times, read and written as that script's (json.dump's one line).

use crate::json::Json;
use crate::{fail, py, usage, Res};
use std::fs::{File, FileTimes};
use std::path::Path;
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// The files of the repository at `dir` against the times remembered: what is remembered now, and how many got their old time back.
fn restore(dir: &Path, old: &Json) -> Res<(Json, usize)> {
    let listing =
        Command::new("git").current_dir(dir).args(["-c", "safe.directory=*", "ls-files", "-s", "-z"]).output().map_err(|e| fail(format!("git: {e}")))?;
    if !listing.status.success() {
        return Err(fail(format!("git ls-files: {}", String::from_utf8_lossy(&listing.stderr).trim())));
    }
    let (mut new, mut restored) = (Vec::new(), 0);
    for entry in listing.stdout.split(|b| *b == 0).filter(|e| !e.is_empty()) {
        // `MODE BLOB STAGE\tPATH`
        let entry = String::from_utf8_lossy(entry);
        let Some((meta, path)) = entry.split_once('\t') else { continue };
        let Some(blob) = meta.split_whitespace().nth(1) else { continue };
        let Ok(st) = std::fs::symlink_metadata(dir.join(path)) else { continue };
        if st.file_type().is_symlink() {
            continue;
        }
        let kept = old.get(path).filter(|k| k.arr().first().and_then(Json::str) == Some(blob));
        match kept.and_then(|k| Some((k, k.arr().get(1)?.num()?.f()))) {
            Some((k, secs)) => {
                let at = UNIX_EPOCH + Duration::try_from_secs_f64(secs).map_err(|_| fail(format!("{path}: the time remembered is {secs}")))?;
                let set = File::open(dir.join(path)).and_then(|f| f.set_times(FileTimes::new().set_accessed(at).set_modified(at)));
                set.map_err(|e| fail(format!("{path}: {e}")))?;
                new.push((path.to_string(), k.clone()));
                restored += 1;
            }
            None => {
                // (whole seconds, as the script kept them)
                let secs = st.modified().unwrap_or(SystemTime::now()).duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
                new.push((path.to_string(), Json::Arr(vec![Json::Str(blob.to_string()), Json::Num(secs.to_string())])));
            }
        }
    }
    Ok((Json::Obj(new), restored))
}

pub fn main(args: &[String], out: &mut String) -> Res<()> {
    let [manifest] = args else { return Err(usage("usage: cargo xtask ci-restore-mtimes MANIFEST.json")) };
    // (no file yet, or one that does not read: nothing is remembered)
    let old = std::fs::read_to_string(manifest).ok().and_then(|t| Json::parse(&t).ok()).unwrap_or(Json::Null);
    let here = std::env::current_dir().map_err(|e| fail(format!("the current folder: {e}")))?;
    let (new, restored) = restore(&here, &old)?;
    let n = if let Json::Obj(o) = &new { o.len() } else { 0 };
    let tmp = format!("{manifest}.tmp");
    py::write(Path::new(&tmp), &new.dump(None, true))?;
    std::fs::rename(&tmp, manifest).map_err(|e| fail(format!("{manifest}: {e}")))?;
    *out += &format!("mtimes: {restored} of {n} files back to their old times, {} new or changed\n", n - restored);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small repository and a file of times: the file written and the times set are the ones scripts/ci_restore_mtimes.py wrote and set for the same (run once
    /// beside this on two copies; the text below is its file).
    #[test]
    #[cfg(unix)]
    fn does_what_the_script_did() {
        let dir = std::env::temp_dir().join(format!("xtask-mtimes-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::create_dir_all(dir.join("dir with space")).unwrap();
        for (path, text) in [("a.txt", "a\n"), ("sub/b.rs", "b\n"), ("dir with space/c é.txt", "c\n"), ("d.txt", "d\n"), ("gone.txt", "gone\n")] {
            std::fs::write(dir.join(path), text).unwrap();
        }
        std::os::unix::fs::symlink("a.txt", dir.join("link")).unwrap();
        let git = |args: &[&str]| {
            let ids = ["-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null"];
            let out = Command::new("git").current_dir(&dir).args(ids).args(args).output().expect("git runs");
            assert!(out.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&out.stderr));
        };
        git(&["init", "-q"]);
        git(&["add", "-A"]);
        git(&["commit", "-q", "-m", "x"]);
        std::fs::remove_file(dir.join("gone.txt")).unwrap();
        let at = |secs: u64| UNIX_EPOCH + Duration::from_secs(secs);
        let touch = |path: &str, secs: u64| File::open(dir.join(path)).unwrap().set_times(FileTimes::new().set_modified(at(secs))).unwrap();
        touch("d.txt", 1767341045);
        touch("sub/b.rs", 1770109506);
        // a.txt and c as remembered, b.rs changed since, old.txt no longer a file of the tree
        let old = concat!(
            r#"{"a.txt": ["78981922613b2afb6025042ff6bd878ac1994e85", 1500000000], "#,
            r#""sub/b.rs": ["0000000000000000000000000000000000000000", 1400000000], "#,
            r#""dir with space/c \u00e9.txt": ["f2ad6c76f0115a6ba5b00456a849810e7ec0af20", 1300000000.5], "old.txt": ["x", 1]}"#,
        );
        let (new, restored) = restore(&dir, &Json::parse(old).unwrap()).unwrap();
        let written = concat!(
            r#"{"a.txt": ["78981922613b2afb6025042ff6bd878ac1994e85", 1500000000], "#,
            r#""d.txt": ["4bcfe98e640c8284511312660fb8709b0afa888e", 1767341045], "#,
            r#""dir with space/c \u00e9.txt": ["f2ad6c76f0115a6ba5b00456a849810e7ec0af20", 1300000000.5], "#,
            r#""sub/b.rs": ["61780798228d17af2d34fce4cfbdf35556832472", 1770109506]}"#,
        );
        assert_eq!((new.dump(None, true).as_str(), restored), (written, 2));
        let time = |path: &str| std::fs::metadata(dir.join(path)).unwrap().modified().unwrap();
        assert_eq!((time("a.txt"), time("sub/b.rs"), time("d.txt")), (at(1500000000), at(1770109506), at(1767341045)));
        assert_eq!(time("dir with space/c é.txt"), UNIX_EPOCH + Duration::from_millis(1300000000500));
        assert_eq!(std::fs::metadata(dir.join("a.txt")).unwrap().accessed().unwrap(), at(1500000000));
        // a second run: every file as remembered; a file of times that does not read: nothing remembered
        let (again, restored) = restore(&dir, &new).unwrap();
        assert_eq!((again.dump(None, true).as_str(), restored), (written, 4));
        assert_eq!(restore(&dir, &Json::Null).unwrap().1, 0);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
