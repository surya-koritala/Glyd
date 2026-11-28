//! The repository's own tasks: `cargo xtask NAME ...`.
//!
//!     cargo xtask bump-version 0.26.0       # set, or --check, the version in every file that carries it (bump_version.rs)
//!     cargo xtask bump-formula 0.26.0       # Formula/glyd.rb at a published release (bump_formula.rs)
//!     cargo xtask changelog-insert CHANGELOG.md NOTES.md   # a release's notes under `## Unreleased` (changelog_insert.rs)
//!     cargo xtask ci-plan-check FIT.json LOADED.json       # the GPU job: the plan is what loaded (ci_plan_check.rs)
//!     cargo xtask ci-doctor-check BUNDLE.json              # the GPU job: `glyd doctor --bundle`'s file (ci_doctor_check.rs)
//!     cargo xtask ci-restore-mtimes MANIFEST.json          # a kept target directory: a checkout's files get their old times (ci_restore_mtimes.rs)
//!     cargo xtask site-release RELEASE.json v0.26.0        # the site's data/release.json names the release (site_release.rs)
//!
//! A task prints what it did, or says what is wrong on stderr and exits 1 (2: the command line was not understood).

mod bump_formula;
mod bump_version;
mod changelog_insert;
mod ci_doctor_check;
mod ci_plan_check;
mod ci_restore_mtimes;
mod json;
mod py;
#[cfg(test)]
mod record;
mod sha256;
mod site_release;

use std::path::{Path, PathBuf};

/// A task's failure: the exit code, and the message for stderr.
#[derive(Debug)]
pub struct Exit {
    pub code: i32,
    pub msg: String,
}

pub type Res<T> = Result<T, Exit>;

pub fn fail(msg: impl Into<String>) -> Exit {
    Exit { code: 1, msg: msg.into() }
}

/// The command line was not understood (exit 2).
pub fn usage(msg: impl Into<String>) -> Exit {
    Exit { code: 2, msg: msg.into() }
}

/// The repository: this crate's parent folder.
fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("xtask/ is in the repository").to_path_buf()
}

const TASKS: &str = "tasks:
  bump-version [VERSION] [--check] [--date YYYY-MM-DD] [--root DIR] | --files | --py VERSION
  bump-formula VERSION
  changelog-insert CHANGELOG NOTES
  ci-plan-check FIT.json LOADED.json
  ci-doctor-check BUNDLE.json
  ci-restore-mtimes MANIFEST.json
  site-release RELEASE.json TAG";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut out = String::new();
    let done = match args.first().map(String::as_str) {
        Some("bump-version") => bump_version::main(&repo(), &args[1..], &mut out),
        Some("bump-formula") => bump_formula::main(&repo(), &args[1..], &mut out),
        Some("changelog-insert") => changelog_insert::main(&args[1..]),
        Some("ci-plan-check") => ci_plan_check::main(&args[1..], &mut out),
        Some("ci-doctor-check") => ci_doctor_check::main(&args[1..], &mut out),
        Some("ci-restore-mtimes") => ci_restore_mtimes::main(&args[1..], &mut out),
        Some("site-release") => site_release::main(&args[1..]),
        Some("-h" | "--help") | None => {
            println!("usage: cargo xtask TASK ...\n{TASKS}");
            return;
        }
        Some(t) => Err(usage(format!("xtask: no task {t}\n{TASKS}"))),
    };
    print!("{out}");
    if let Err(e) = done {
        if !e.msg.is_empty() {
            eprintln!("{}", e.msg);
        }
        std::process::exit(e.code);
    }
}
