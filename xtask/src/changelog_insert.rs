//! Put a release's notes under CHANGELOG.md's `## Unreleased` heading (scripts/release.sh; bump-version turns that heading into the release's):
//!
//!     cargo xtask changelog-insert CHANGELOG.md NOTES.md
//!
//! This is the Python that release.sh held (through v0.35.0); the file it writes is that code's byte for byte.

use crate::{fail, py, usage, Res};
use std::path::Path;

const HEAD: &str = "## Unreleased\n";

fn inserted(text: &str, notes: &str, notes_name: &str) -> Res<String> {
    let at = text.find(HEAD).ok_or_else(|| fail("CHANGELOG.md has no '## Unreleased' heading"))? + HEAD.len();
    let body = notes.trim_matches('\n');
    if body.is_empty() {
        return Err(fail(format!("{notes_name} is empty")));
    }
    Ok(format!("{}\n{body}\n{}", &text[..at], &text[at..]))
}

pub fn main(args: &[String]) -> Res<()> {
    let [path, notes] = args else { return Err(usage("usage: cargo xtask changelog-insert CHANGELOG NOTES")) };
    let text = inserted(&py::read(Path::new(path))?, &py::read(Path::new(notes))?, notes)?;
    py::write(Path::new(path), &text)
}

#[cfg(test)]
mod tests {
    use super::inserted;

    /// What release.sh's Python wrote for the same texts (run once on them; the results are the strings here).
    #[test]
    fn does_what_the_script_did() {
        let log = "# Changelog\n\n## Unreleased\n\n- Waiting.\n\n## v0.35.0 — 2026-11-27\n\n## Unreleased\n";
        assert_eq!(
            inserted(log, "\n\n- A note.\n  Its second line.\n- Another. \n\n", "n.md").unwrap(),
            "# Changelog\n\n## Unreleased\n\n- A note.\n  Its second line.\n- Another. \n\n- Waiting.\n\n## v0.35.0 — 2026-11-27\n\n## Unreleased\n"
        );
        assert_eq!(inserted("## Unreleased\n", "- One.", "n.md").unwrap(), "## Unreleased\n\n- One.\n");
        assert_eq!(inserted("## Unreleased", "- One.", "n.md").unwrap_err().msg, "CHANGELOG.md has no '## Unreleased' heading");
        assert_eq!(inserted(log, "\n\n", "notes/n.md").unwrap_err().msg, "notes/n.md is empty");
        assert_eq!(inserted(log, " \n", "n.md").unwrap(), log.replacen("## Unreleased\n", "## Unreleased\n\n \n", 1));
        // (only newlines are trimmed)
    }
}
