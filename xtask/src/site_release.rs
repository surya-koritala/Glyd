//! Name a release in the site's data/release.json (scripts/release.sh's site step): its `release` and `installScript` become the tag, the rest stays.
//!
//!     cargo xtask site-release data/release.json v0.26.0
//!
//! This is the Python that release.sh held (through v0.35.0): the file is written as json.dump wrote it (two spaces a level, characters as they are, a newline last).

use crate::json::Json;
use crate::{fail, py, usage, Res};
use std::path::Path;

fn named(text: &str, tag: &str) -> Res<String> {
    let mut d = Json::parse(text)?;
    if !matches!(d, Json::Obj(_)) {
        return Err(fail("not an object"));
    }
    d.set("release", Json::Str(tag.to_string()));
    d.set("installScript", Json::Str(tag.to_string()));
    Ok(d.dump(Some(2), false) + "\n")
}

pub fn main(args: &[String]) -> Res<()> {
    let [path, tag] = args else { return Err(usage("usage: cargo xtask site-release RELEASE.json TAG")) };
    let text = named(&py::read(Path::new(path))?, tag).map_err(|e| fail(format!("{path}: {}", e.msg)))?;
    py::write(Path::new(path), &text)
}

#[cfg(test)]
mod tests {
    use crate::record;

    /// tests/site/*.txt: a release.json, and the file release.sh's Python made of it for v0.36.0 (recorded from it once).
    #[test]
    fn does_what_the_script_did() {
        let cases = record::cases("site", "");
        assert!(cases.len() >= 3, "{} cases", cases.len());
        for (name, c) in cases {
            assert_eq!(super::named(c.file("release.json"), "v0.36.0").unwrap(), c.one("OUT release.json"), "{name}");
        }
        assert!(super::named("[1]", "v0.36.0").is_err() && super::named("{", "v0.36.0").is_err());
    }
}
