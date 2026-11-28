//! Point Formula/glyd.rb at a published release, after its workflow has uploaded the tarballs:
//!
//!     cargo xtask bump-formula 0.21.0
//!
//! then copy the file to the tap, github.com/surya-koritala/homebrew-glyd (Formula/glyd.rb).
//! Every sha256 is computed from the file its url names (curl downloads it), so a missing asset fails here, not in brew.
//!
//! This is scripts/bump_formula.py in Rust (the script through v0.35.0); the file it writes is that script's byte for byte.

use crate::{fail, py, sha256, usage, Res};
use std::path::Path;
use std::process::Command;

const FORMULA: &str = "Formula/glyd.rb";

/// The formula's text at version `new`, every sha256 the digest of what its url gives; and a line for each: the digest and the url.
fn bumped(text: &str, new: &str, fetch: &dyn Fn(&str) -> Res<Vec<u8>>) -> Res<(String, String)> {
    // `^  version "([^"]+)"$`
    let old = text.split('\n').find_map(|l| l.strip_prefix("  version \"")?.strip_suffix('"').filter(|v| !v.is_empty() && !v.contains('"')));
    let old = old.ok_or_else(|| fail(format!("{FORMULA}: no line `  version \"X.Y.Z\"`")))?;
    // comments keep their versions (the license note names v0.28.0 on purpose)
    let text: String = text.split_inclusive('\n').map(|l| if py::strip_start(l).starts_with('#') { l.to_string() } else { l.replace(old, new) }).collect();
    // `(url "([^"]+)"\n\s+sha256 ")[0-9a-f]{64}`
    const URL: &str = "url \"";
    let (mut out, mut said, mut at) = (String::new(), String::new(), 0);
    while let Some(i) = text[at..].find(URL) {
        let start = at + i + URL.len();
        let found = text[start..].find('"').filter(|n| *n > 0).and_then(|n| {
            let rest = text[start + n + 1..].strip_prefix('\n')?;
            let key = py::strip_start(rest).strip_prefix("sha256 \"").filter(|_| py::strip_start(rest).len() < rest.len())?;
            let old = key.get(..64).filter(|d| d.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)))?;
            Some((&text[start..start + n], text.len() - key.len(), old.len()))
        });
        match found {
            Some((url, digest, len)) => {
                let new = sha256::hex(&fetch(url)?);
                said += &format!("{new} {url}\n");
                out += &text[at..digest];
                out += &new;
                at = digest + len;
            }
            None => {
                out += &text[at..at + i + 1];
                at += i + 1;
            }
        }
    }
    out += &text[at..];
    Ok((out, said))
}

fn curl(url: &str) -> Res<Vec<u8>> {
    let got = Command::new("curl").args(["-fsSL", "--retry", "3", "--", url]).output().map_err(|e| fail(format!("curl: {e}")))?;
    if !got.status.success() {
        return Err(fail(format!("{url}: {}", String::from_utf8_lossy(&got.stderr).trim())));
    }
    Ok(got.stdout)
}

pub fn main(repo: &Path, args: &[String], out: &mut String) -> Res<()> {
    let [version] = args else { return Err(usage("usage: cargo xtask bump-formula VERSION")) };
    let path = repo.join(FORMULA);
    let (text, said) = bumped(&py::read(&path)?, version.trim_start_matches('v'), &curl)?;
    *out += &said;
    py::write(&path, &text)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// tests/formula/: a formula, and what scripts/bump_formula.py wrote and printed for `v0.36.0` with every url's file the url's own text (recorded from it once).
    #[test]
    fn does_what_the_script_did() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/formula");
        let read = |name: &str| std::fs::read_to_string(dir.join(name)).unwrap();
        let (text, said) = bumped(&read("in.rb"), "0.36.0", &|url| Ok(url.as_bytes().to_vec())).unwrap();
        assert_eq!(text, read("out.rb"));
        assert_eq!(said, read("out.txt"));
        assert!(said.lines().count() == 4 && text.contains("v0.28.0") && !text.contains("0.35.0\""));
        let err = bumped(&read("in.rb"), "0.36.0", &|url| Err(fail(format!("{url}: 404")))).unwrap_err();
        assert!(err.code == 1 && err.msg.ends_with("glyd-v0.36.0-macos-arm64.tar.gz: 404"), "{}", err.msg);
        assert!(bumped("class Glyd < Formula\nend\n", "0.36.0", &|_| Ok(Vec::new())).is_err());
    }
}
