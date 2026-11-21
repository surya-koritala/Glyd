//! `glyd` with the engine linked (the `engine` feature: cargo test -p glyd --features engine --test cli_linked_engine): the model commands run inside this program,
//! with no `glyd-engine` on PATH, and a link named glyd-engine to it is the engine's old command line.
#![cfg(all(unix, feature = "engine", feature = "deflate"))]

use std::fs;
use std::os::unix::fs::{symlink, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

/// This program copied into a folder of its own (an install's `~/.local/share/glyd/cli`), with nothing else beside it.
fn install(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("glyd-linked-test-{}-{}", std::process::id(), name));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    fs::copy(env!("CARGO_BIN_EXE_glyd"), d.join("glyd")).unwrap();
    d
}

fn run(exe: &Path, args: &[&str], home: &Path, env: &[(&str, &str)]) -> Output {
    let mut c = Command::new(exe);
    c.args(args).env_clear().env("PATH", "/nonexistent").env("HOME", home).env("HF_HOME", home.join("hf")).env("GLYD_CACHE", home.join("cache"));
    for (k, v) in env {
        c.env(k, v);
    }
    c.output().unwrap()
}

#[test]
fn the_engine_is_in_the_program_and_the_version_says_so() {
    let d = install("version");
    let o = run(&d.join("glyd"), &["--version"], &d, &[]);
    let out = text(&o.stdout);
    assert!(out.starts_with("glyd ") && out.contains("engine: linked"), "{out}");
}

#[test]
fn model_commands_run_with_no_engine_program_anywhere() {
    let d = install("commands");
    // `fit` with a machine that is not one: the engine's own refusal, said as glyd's (it was "Unknown option")
    let o = run(&d.join("glyd"), &["fit", "Qwen/Qwen3-8B", "--machine", "no-such-machine"], &d, &[]);
    let err = text(&o.stderr);
    assert!(err.contains("not a preset and not a file") && !err.contains("Unknown option") && !err.contains("the engine is not here"), "{err}");
    assert!(err.contains("rtx4080s"), "{err}"); // (the presets)
                                                // `doctor` answers from the engine, without a GPU
    let o = run(&d.join("glyd"), &["doctor"], &d, &[]);
    assert!(text(&o.stdout).starts_with("glyd engine "), "{}{}", text(&o.stdout), text(&o.stderr));
}

#[test]
fn a_link_named_glyd_engine_is_the_engines_old_command_line() {
    let d = install("alias");
    symlink("glyd", d.join("glyd-engine")).unwrap();
    let o = run(&d.join("glyd-engine"), &["--version"], &d, &[]);
    assert_eq!(text(&o.stdout).trim(), format!("glyd-engine {}", env!("CARGO_PKG_VERSION")), "{}", text(&o.stderr));
    // with no words at all it is the engine's usage, not the compression program's
    let o = run(&d.join("glyd-engine"), &[], &d, &[]);
    assert_eq!(o.status.code(), Some(1));
    assert!(text(&o.stderr).contains("usage: glyd-engine run MODEL"), "{}", text(&o.stderr));
}

#[test]
fn with_the_python_tool_the_link_beside_is_the_engine_it_is_told_about() {
    let d = install("tool");
    let bin = d.join("pybin");
    fs::create_dir_all(&bin).unwrap();
    let tool = bin.join("glyd");
    fs::write(&tool, "#!/bin/sh\n# from glyd.cli import main\necho \"python glyd: engine=$GLYD_ENGINE args=$*\"\n").unwrap();
    fs::set_permissions(&tool, fs::Permissions::from_mode(0o755)).unwrap();
    let path = bin.display().to_string();
    let go = |home: &Path| {
        let mut c = Command::new(d.join("glyd"));
        c.args(["run", "m"]).env_clear().env("PATH", &path).env("HOME", home);
        text(&c.output().unwrap().stdout)
    };
    // no glyd-engine beside: the tool is told of none (its own package has an engine, or vLLM runs it)
    assert_eq!(go(&d).trim(), "python glyd: engine= args=run m");
    // the link beside is the engine the tool starts (as its child: its name makes it the engine's command line)
    symlink("glyd", d.join("glyd-engine")).unwrap();
    assert_eq!(go(&d).trim(), format!("python glyd: engine={} args=run m", d.join("glyd-engine").display()));
}

#[test]
fn the_help_the_man_page_and_the_flag_refusals_are_made_from_the_command_table() {
    let d = install("table");
    let g = d.join("glyd");
    // glyd help COMMAND, and the command's --help, say the table's page (then the engine's options)
    let o = run(&g, &["help", "fit"], &d, &[]);
    assert!(text(&o.stdout).starts_with("Usage: glyd fit MODEL[:LEVEL]") && text(&o.stdout).contains("--machine NAME|FILE"), "{}", text(&o.stdout));
    let o = run(&g, &["ps", "--help"], &d, &[]);
    assert!(text(&o.stdout).starts_with("Usage: glyd ps") && text(&o.stdout).contains("Every option of the engine:"), "{}", text(&o.stdout));
    // the manual page is the table's, and the one in docs/ is it
    let o = run(&g, &["man"], &d, &[]);
    assert_eq!(text(&o.stdout), fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/glyd.1")).unwrap());
    // a flag a command does not take is refused, naming the commands that take it
    let o = run(&g, &["run", "gemma4", "--json"], &d, &[]);
    assert_eq!(o.status.code(), Some(1));
    assert!(
        text(&o.stderr).contains("--json is not a flag of this command") && text(&o.stderr).contains("ls, ps, service, fit, doctor"),
        "{}",
        text(&o.stderr)
    );
    let o = run(&g, &["ls", "--machine", "h100"], &d, &[]);
    assert_eq!(o.status.code(), Some(1));
    assert!(text(&o.stderr).contains("fit, doctor, bench"), "{}", text(&o.stderr));
    // and `ls --json` is JSON
    let o = run(&g, &["ls", "--json"], &d, &[]);
    assert!(text(&o.stdout).starts_with("{\"models\":["), "{}", text(&o.stdout));
}
