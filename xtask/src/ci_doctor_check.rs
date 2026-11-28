//! The GPU check of `glyd doctor --bundle FILE`: on the runner's GPU the bundle says everything is fine, the CUDA backend is accepted, the sample plan loads, and
//! nothing secret or personal is in it.
//!
//!     cargo xtask ci-doctor-check BUNDLE.json
//!
//! This is scripts/ci_doctor_check.py in Rust (the script through v0.35.0): the same lines, the same verdict.

use crate::json::Json;
use crate::{fail, py, usage, Res};
use std::path::Path;

/// What the check prints, and whether the bundle is what it should be. `home`: the home folder; `secrets`: what must not be in the text.
fn check(text: &str, home: Option<&str>, secrets: &[String]) -> Res<(String, bool)> {
    fn key<'a>(v: &'a Json, k: &str) -> Res<&'a Json> {
        v.get(k).ok_or_else(|| fail(format!("the bundle has no {}", py::repr(k))))
    }
    let b = Json::parse(text)?;
    let checks = key(&b, "checks")?.arr();
    let mut bad = Vec::new();
    if key(&b, "schema")?.str() != Some("glyd-doctor/1") {
        bad.push(format!("schema {}", key(&b, "schema")?.text()));
    }
    if !key(&b, "ok")?.truthy() {
        let mut failed = Vec::new();
        for c in checks {
            if key(c, "status")?.str() == Some("fail") {
                failed.push(format!("{}: {}", key(c, "id")?.text(), key(c, "found")?.text()));
            }
        }
        bad.push(format!("not ok: {}", failed.join("; ")));
    }
    // (the last check of an id is the one a dict keeps)
    let mut cuda = None;
    for c in checks {
        if key(c, "id")?.str() == Some("backend-cuda") {
            cuda = Some(c);
        }
    }
    let found = key(cuda.ok_or_else(|| fail("the bundle has no check 'backend-cuda'"))?, "found")?;
    let found = found.str().ok_or_else(|| fail("backend-cuda: what was found is not text"))?;
    if !found.contains("accepted") {
        bad.push(format!("CUDA: {found}"));
    }
    let gpus = key(&b, "gpus")?;
    if !gpus.truthy() {
        bad.push("no GPU in the bundle".to_string());
    }
    let plan = key(&b, "sample_plan")?;
    let level = plan.get("level").unwrap_or(&Json::Null);
    if *level == Json::Null {
        bad.push(format!("the sample plan does not load here: {}", plan.repr()));
    }
    if home.is_some_and(|h| h.chars().count() > 1 && text.contains(h)) {
        bad.push("the home folder is in the bundle".to_string());
    }
    for s in secrets {
        if !s.is_empty() && text.contains(s.as_str()) {
            bad.push("a secret is in the bundle".to_string());
        }
    }
    for c in checks {
        if key(c, "status")?.str() == Some("fail") && *key(c, "fix")? == Json::Null {
            bad.push(format!("{} fails with no fix", key(c, "id")?.text()));
        }
    }
    let gpu = match gpus.arr().first() {
        Some(g) if gpus.truthy() => key(g, "name")?.text(),
        _ => "None".to_string(),
    };
    let mut out = format!("{} checks, GPU {gpu}, sample plan {}\n", checks.len(), level.text());
    if !bad.is_empty() {
        out += &format!("the bundle is not what it should be:\n  {}\n", bad.join("\n  "));
    }
    Ok((out, bad.is_empty()))
}

pub fn main(args: &[String], out: &mut String) -> Res<()> {
    let [bundle] = args else { return Err(usage("usage: cargo xtask ci-doctor-check BUNDLE.json")) };
    let secrets: Vec<String> = ["HF_TOKEN", "GLYD_API_KEY"].iter().filter_map(|k| std::env::var(k).ok()).collect();
    let (said, ok) = check(&py::read(Path::new(bundle))?, std::env::var("HOME").ok().as_deref(), &secrets)?;
    *out += &said;
    if ok {
        Ok(())
    } else {
        Err(fail(""))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record;

    /// tests/checks/doctor-*.txt: what scripts/ci_doctor_check.py printed for this bundle with this home folder and these secrets, and its exit code (recorded from it once).
    #[test]
    fn does_what_the_script_did() {
        let cases = record::cases("checks", "doctor-");
        assert!(cases.len() >= 8, "{} cases", cases.len());
        for (name, c) in cases {
            let env = |k: &str| c.one("ENV").lines().find_map(|l| l.strip_prefix(k)?.strip_prefix('=')).map(str::to_string);
            let secrets: Vec<String> = ["HF_TOKEN", "GLYD_API_KEY"].iter().filter_map(|k| env(k)).collect();
            match check(c.file("bundle.json"), env("HOME").as_deref(), &secrets) {
                Ok((said, ok)) => assert_eq!((said.as_str(), i32::from(!ok)), (c.one("STDOUT"), c.exit()), "{name}"),
                Err(e) => assert!(
                    c.exit() == 1 && c.one("STDOUT").is_empty() && c.one("STDERR").contains("Error"),
                    "{name}: {} where the script printed {}",
                    e.msg,
                    c.one("STDOUT")
                ),
            }
        }
    }
}
