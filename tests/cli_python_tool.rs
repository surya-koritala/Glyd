//! `glyd run`, `glyd serve`, `glyd doctor`, `glyd login`, `glyd pack` and `glyd verify` reach Glyd's Python tool from the Rust `glyd`
//! (src/bin/glyd.rs), whichever of the two programs is first on PATH; where there is no tool, the answer says so, in place of the usage
//! text of a file named run.
#![cfg(all(unix, feature = "deflate"))]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("glyd-cli-test-{}-{}", std::process::id(), name));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// A stand-in for the Python tool's console script: a `#!` line and the import of glyd.cli, as uv writes it.
fn python_tool(dir: &Path, body: &str) -> PathBuf {
    fs::create_dir_all(dir).unwrap();
    let exe = dir.join("glyd");
    fs::write(&exe, format!("#!/bin/sh\n# from glyd.cli import main\n{body}\n")).unwrap();
    fs::set_permissions(&exe, fs::Permissions::from_mode(0o755)).unwrap();
    exe
}

fn glyd(args: &[&str], path: &str, home: &Path, env: &[(&str, &str)]) -> Output {
    let mut c = Command::new(env!("CARGO_BIN_EXE_glyd"));
    c.args(args).env_clear().env("PATH", path).env("HOME", home);
    for (k, v) in env {
        c.env(k, v);
    }
    c.output().unwrap()
}

fn text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

#[test]
fn the_python_tools_commands_are_passed_to_it_from_a_rust_first_path() {
    let t = tmp("pass");
    let bin = t.join("pybin");
    python_tool(&bin, r#"echo "python glyd: forwarded=$GLYD_FORWARDED args=$*""#);
    for cmd in ["run", "serve", "doctor", "login", "pack", "verify"] {
        let o = glyd(&[cmd, "Qwen/Qwen3-8B", "--prompt", "hi"], bin.to_str().unwrap(), &t, &[]);
        assert!(o.status.success(), "{cmd}: {}", text(&o.stderr));
        assert_eq!(text(&o.stdout).trim(), format!("python glyd: forwarded=1 args={cmd} Qwen/Qwen3-8B --prompt hi"), "{cmd}");
    }
    // its exit status is the tool's own
    python_tool(&bin, "exit 3");
    assert_eq!(glyd(&["run", "m"], bin.to_str().unwrap(), &t, &[]).status.code(), Some(3));
}

#[test]
fn the_tool_is_found_where_uv_puts_it_before_the_terminal_knows() {
    let t = tmp("uvbin");
    python_tool(&t.join(".local").join("bin"), r#"echo "python glyd: $*""#);
    let o = glyd(&["doctor"], "/nonexistent", &t, &[]); // (PATH has not got ~/.local/bin: a terminal opened before the install)
    assert_eq!(text(&o.stdout).trim(), "python glyd: doctor", "{}", text(&o.stderr));
    let other = t.join("elsewhere");
    python_tool(&other, r#"echo "python glyd, UV_TOOL_BIN_DIR: $*""#);
    let o = glyd(&["doctor"], "/nonexistent", &t, &[("UV_TOOL_BIN_DIR", other.to_str().unwrap())]);
    assert!(text(&o.stdout).contains("UV_TOOL_BIN_DIR"), "{}", text(&o.stdout));
}

#[test]
fn a_program_that_is_not_the_python_tool_is_not_taken_for_it() {
    let t = tmp("notool");
    // the Rust program's own directory (no #! line), and another file named glyd without glyd.cli in it
    let own = Path::new(env!("CARGO_BIN_EXE_glyd")).parent().unwrap().to_str().unwrap().to_string();
    let other = t.join("other");
    fs::create_dir_all(&other).unwrap();
    fs::write(other.join("glyd"), "#!/bin/sh\necho some other glyd\n").unwrap();
    fs::set_permissions(other.join("glyd"), fs::Permissions::from_mode(0o755)).unwrap();
    let path = format!("{}:{}:", other.display(), own); // (and an empty entry, the current directory)
    let o = glyd(&["run", "Qwen/Qwen3-8B"], &path, &t, &[]);
    assert_eq!(o.status.code(), Some(127));
    let err = text(&o.stderr);
    assert!(!text(&o.stdout).contains("some other glyd") && !err.contains("Usage:"), "{err}");
    if cfg!(target_os = "linux") {
        assert!(err.contains("not installed here") && err.contains("https://getglyd.com/install.sh"), "{err}");
    } else {
        assert!(err.contains("needs Linux with an NVIDIA GPU"), "{err}");
    }
    assert!(err.contains("./run"), "{err}"); // (a file named run is still reached)
}

#[test]
fn a_tool_that_sends_the_command_back_does_not_start_a_loop() {
    let t = tmp("loop");
    let bin = t.join("pybin");
    // an older tool: it passes what it does not know to the compression program, as it was told by its PATH
    let me = env!("CARGO_BIN_EXE_glyd");
    python_tool(&bin, &format!(r#"exec "{me}" "$@""#));
    let o = glyd(&["run", "m"], bin.to_str().unwrap(), &t, &[]);
    assert_eq!(o.status.code(), Some(127), "{}", text(&o.stderr));
    assert!(text(&o.stderr).contains("sent it back"), "{}", text(&o.stderr));
    // and with the variable set from outside, nothing is started
    python_tool(&bin, r#"echo started"#);
    let o = glyd(&["serve", "m"], bin.to_str().unwrap(), &t, &[("GLYD_FORWARDED", "1")]);
    assert_eq!(o.status.code(), Some(127));
    assert!(!text(&o.stdout).contains("started"));
}

#[test]
fn an_empty_guard_variable_is_not_a_guard() {
    // GLYD_FORWARDED="" is not set, for the Python tool (it tests the string) and so for this program: the command is passed on
    let t = tmp("emptyguard");
    let bin = t.join("pybin");
    python_tool(&bin, r#"echo "python glyd: $*""#);
    let o = glyd(&["run", "m"], bin.to_str().unwrap(), &t, &[("GLYD_FORWARDED", "")]);
    assert_eq!(text(&o.stdout).trim(), "python glyd: run m", "{}", text(&o.stderr));
}

#[test]
fn a_file_named_run_is_compressed_as_dot_slash_run() {
    let t = tmp("file");
    let data: Vec<u8> = b"glyd test line, repeated\n".iter().cycle().take(20000).copied().collect();
    fs::write(t.join("run"), &data).unwrap();
    let o = Command::new(env!("CARGO_BIN_EXE_glyd")).current_dir(&t).args(["./run", "-o", "run.glyd"]).output().unwrap();
    assert!(o.status.success(), "{}", text(&o.stderr));
    let o = Command::new(env!("CARGO_BIN_EXE_glyd")).current_dir(&t).args(["-d", "run.glyd", "-o", "run.out"]).output().unwrap();
    assert!(o.status.success(), "{}", text(&o.stderr));
    assert_eq!(fs::read(t.join("run.out")).unwrap(), data);
}
