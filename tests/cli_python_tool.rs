//! The routing of the model commands in the Rust `glyd` (src/bin/glyd.rs), as a program started: where the engine is a program of its own (this test's build has
//! the codec's `glyd` alone; the `engine` feature's build, where the engine is linked, is tests/cli_linked_engine.rs and the unit tests of the routing in the program).
//! `glyd run`, `glyd serve`, `glyd doctor` and `glyd login` reach Glyd's Python tool from the Rust `glyd` (src/bin/glyd.rs), whichever of
//! the two programs is first on PATH; where there is no tool, the answer says so, in place of the usage text of a file named run. Where
//! there is a tool, `glyd run` and `glyd serve` are its to decide (it asks the engine before anything is downloaded, runs it as its child,
//! and falls back to vLLM, with one line; `serve` is vLLM's unless GLYD_ENGINE asks for the engine): this program tells it which engine it found,
//! and says nothing. Where there is no tool, an engine program on PATH runs the command, with one line.
#![cfg(all(unix, feature = "deflate"))]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::{Mutex, MutexGuard};

/// Writing a program and starting a process take turns across the tests (which run on threads of one process): a process
/// started while another test's program is open for writing holds it open until it execs, and that program then cannot be
/// run ("Text file busy").
static TURN: Mutex<()> = Mutex::new(());

fn turn() -> MutexGuard<'static, ()> {
    TURN.lock().unwrap_or_else(|e| e.into_inner())
}

/// A program at `exe`: `text`, made executable.
fn program(exe: &Path, text: &str) {
    let _t = turn();
    fs::write(exe, text).unwrap();
    fs::set_permissions(exe, fs::Permissions::from_mode(0o755)).unwrap();
}

fn run(c: &mut Command) -> Output {
    let _t = turn();
    c.output().unwrap()
}

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
    program(&exe, &format!("#!/bin/sh\n# from glyd.cli import main\n{body}\n"));
    exe
}

/// This program copied into a folder of its own: an engine a build left beside it in target/ (glyd-engine's own tests build one) is not found before
/// the tests' stand-ins.
fn glyd_exe() -> &'static Path {
    static EXE: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    EXE.get_or_init(|| {
        let d = tmp("exe");
        let _t = turn();
        fs::copy(env!("CARGO_BIN_EXE_glyd"), d.join("glyd")).unwrap();
        d.join("glyd")
    })
}

fn glyd(args: &[&str], path: &str, home: &Path, env: &[(&str, &str)]) -> Output {
    let mut c = Command::new(glyd_exe());
    c.args(args).env_clear().env("PATH", path).env("HOME", home);
    for (k, v) in env {
        c.env(k, v);
    }
    run(&mut c)
}

fn text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

#[test]
fn the_python_tools_commands_are_passed_to_it_from_a_rust_first_path() {
    let t = tmp("pass");
    let bin = t.join("pybin");
    python_tool(&bin, r#"echo "python glyd: forwarded=$GLYD_FORWARDED args=$*""#);
    for cmd in ["run", "serve", "doctor", "login"] {
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
    let own = glyd_exe().parent().unwrap().to_str().unwrap().to_string();
    let other = t.join("other");
    fs::create_dir_all(&other).unwrap();
    program(&other.join("glyd"), "#!/bin/sh\necho some other glyd\n");
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
    let me = glyd_exe().display();
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
    let o = run(Command::new(env!("CARGO_BIN_EXE_glyd")).current_dir(&t).args(["./run", "-o", "run.glyd"]));
    assert!(o.status.success(), "{}", text(&o.stderr));
    let o = run(Command::new(env!("CARGO_BIN_EXE_glyd")).current_dir(&t).args(["-d", "run.glyd", "-o", "run.out"]));
    assert!(o.status.success(), "{}", text(&o.stderr));
    assert_eq!(fs::read(t.join("run.out")).unwrap(), data);
}

/// A stand-in for glyd-engine in `dir`: any command prints its arguments.
fn engine(dir: &Path) {
    fs::create_dir_all(dir).unwrap();
    let exe = dir.join("glyd-engine");
    program(&exe, "#!/bin/sh\necho \"engine: $*\"\n");
}

const STUB: &str = r#"echo "python glyd: forwarded=$GLYD_FORWARDED engine=$GLYD_ENGINE args=$*""#;

#[test]
fn the_tool_decides_where_there_is_one_and_is_told_which_engine_this_program_found() {
    let t = tmp("engine");
    let (py, en) = (t.join("pybin"), t.join("enbin"));
    python_tool(&py, STUB);
    engine(&en);
    let exe = en.join("glyd-engine");
    let path = format!("{}:{}", en.display(), py.display());
    // run: the tool's, with the engine this program found on PATH named in GLYD_ENGINE; this program says nothing, and starts no engine
    let o = glyd(&["run", "Qwen/Qwen3-8B", "--prompt", "hi", "-p", "x"], &path, &t, &[]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    assert_eq!(text(&o.stdout).trim(), format!("python glyd: forwarded=1 engine={} args=run Qwen/Qwen3-8B --prompt hi -p x", exe.display()));
    assert_eq!(text(&o.stderr), "");
    // serve is the tool's too, and vLLM's unless the engine is asked for (the engine serves up to 8 users at once): the engine this program found is named to the tool only where GLYD_ENGINE asks for it (1, on, yes, true)
    for (set, named) in [(None, false), (Some("auto"), false), (Some("1"), true), (Some("ON"), true), (Some("yes"), true), (Some("true"), true)] {
        let env: Vec<(&str, &str)> = set.map(|v| ("GLYD_ENGINE", v)).into_iter().collect();
        let o = glyd(&["serve", "Qwen/Qwen3-8B", "--prompt", "hi"], &path, &t, &env);
        let engine_is = if named { exe.display().to_string() } else { set.unwrap_or("").to_string() };
        assert_eq!(
            text(&o.stdout).trim(),
            format!("python glyd: forwarded=1 engine={engine_is} args=serve Qwen/Qwen3-8B --prompt hi"),
            "{set:?}: {}",
            text(&o.stderr)
        );
        assert_eq!(text(&o.stderr), "");
    }
    // an engine beside this program is found too (a copy of this program in a folder of its own, with a stand-in beside it)
    let beside = t.join("beside");
    fs::create_dir_all(&beside).unwrap();
    {
        let _t = turn();
        fs::copy(env!("CARGO_BIN_EXE_glyd"), beside.join("glyd")).unwrap();
    }
    engine(&beside);
    let o = run(Command::new(beside.join("glyd")).args(["run", "m"]).env_clear().env("PATH", &py).env("HOME", &t));
    assert_eq!(text(&o.stdout).trim(), format!("python glyd: forwarded=1 engine={} args=run m", beside.join("glyd-engine").display()), "{}", text(&o.stderr));
    // no engine here: the tool decides alone (its own package has one, or vLLM runs the model)
    let o = glyd(&["run", "m"], py.to_str().unwrap(), &t, &[]);
    assert_eq!((text(&o.stdout).trim(), text(&o.stderr)), ("python glyd: forwarded=1 engine= args=run m", String::new()));
    // what the user set is the tool's to read: GLYD_ENGINE=0, a program the user named (there or not) are passed as they are
    for v in ["0", "off", "/nonexistent/glyd-engine"] {
        let o = glyd(&["run", "m"], &path, &t, &[("GLYD_ENGINE", v)]);
        assert_eq!((text(&o.stdout).trim(), text(&o.stderr)), (format!("python glyd: forwarded=1 engine={v} args=run m").as_str(), String::new()), "{v}");
    }
    // doctor and login are the engine's where this program finds one, with no line
    let o = glyd(&["doctor"], &path, &t, &[]);
    assert_eq!((text(&o.stdout).trim(), text(&o.stderr)), ("engine: doctor", String::new()));
    // a missing model and the usage text are the tool's words
    let o = glyd(&["run", "--help"], &path, &t, &[]);
    assert_eq!(
        (text(&o.stdout).trim(), text(&o.stderr)),
        (format!("python glyd: forwarded=1 engine={} args=run --help", exe.display()).as_str(), String::new())
    );
    // vLLM's flags after a lone -- reach the tool whole
    let o = glyd(&["run", "m", "--", "--max-model-len", "2048"], &path, &t, &[]);
    assert!(text(&o.stdout).contains("args=run m -- --max-model-len 2048"), "{}", text(&o.stdout));
    // a command the tool sent here is not sent back to an engine, or to the tool
    let o = glyd(&["run", "m"], &path, &t, &[("GLYD_FORWARDED", "1")]);
    assert_eq!(o.status.code(), Some(127), "{}", text(&o.stderr));
}

#[test]
fn where_there_is_no_tool_an_engine_program_runs_the_command_and_says_so() {
    let t = tmp("notool-engine");
    let en = t.join("enbin");
    engine(&en);
    let o = glyd(&["run", "m", "-p", "hi"], en.to_str().unwrap(), &t, &[]);
    assert_eq!(text(&o.stdout).trim(), "engine: run m -p hi", "{}", text(&o.stderr));
    assert_eq!(text(&o.stderr).trim(), "glyd run: m on glyd-engine (GLYD_ENGINE=0: vLLM instead).");
    // the engine the user names runs it; one that is not there, or GLYD_ENGINE=0, leaves nothing to run it: the answer says there is no tool
    let own = t.join("own");
    engine(&own);
    let o = glyd(&["serve", "m"], "/nonexistent", &t, &[("GLYD_ENGINE", own.join("glyd-engine").to_str().unwrap())]);
    assert_eq!(text(&o.stdout).trim(), "engine: serve m", "{}", text(&o.stderr));
    for v in ["0", "/nonexistent/glyd-engine"] {
        let o = glyd(&["run", "m"], en.to_str().unwrap(), &t, &[("GLYD_ENGINE", v)]);
        assert_eq!(o.status.code(), Some(127), "{v}: {}", text(&o.stderr));
        assert!(text(&o.stdout).is_empty() && !text(&o.stderr).contains("on glyd-engine"), "{v}: {}", text(&o.stderr));
    }
    // a command that is no model's has no line
    let o = glyd(&["doctor"], en.to_str().unwrap(), &t, &[]);
    assert_eq!((text(&o.stdout).trim(), text(&o.stderr)), ("engine: doctor", String::new()));
}

/// A stand-in for glyd-engine that says what it was given: its arguments, and the key and the token of its environment.
fn engine_with_env(dir: &Path) {
    fs::create_dir_all(dir).unwrap();
    let exe = dir.join("glyd-engine");
    program(&exe, "#!/bin/sh\necho \"engine: $* key=$GLYD_API_KEY token=$HF_TOKEN\"\n");
}

#[test]
fn a_key_and_a_token_reach_the_engine_in_its_environment_and_not_on_its_command_line() {
    // where there is no tool and the engine runs the command: each of the two forms of a flag, among the flags it was given
    let t = tmp("secrets");
    let (py, en) = (t.join("py"), t.join("en"));
    engine_with_env(&en);
    let path = format!("{}:/usr/bin:/bin", en.display());
    let o = glyd(&["serve", "Qwen/Qwen3-8B", "--port", "8001", "--api-key", "K1", "--hf-token=T1"], &path, &t, &[]);
    assert_eq!(text(&o.stdout).trim(), "engine: serve Qwen/Qwen3-8B --port 8001 key=K1 token=T1", "{}", text(&o.stderr));
    assert!(
        text(&o.stderr).contains("the API key goes to the engine in GLYD_API_KEY")
            && text(&o.stderr).contains("the Hugging Face token goes to the engine in HF_TOKEN"),
        "{}",
        text(&o.stderr)
    );
    assert!(!text(&o.stderr).contains("K1") && !text(&o.stderr).contains("T1"), "a line said the secret: {}", text(&o.stderr));
    // no flag: the environment as it was; a flag with no value (`--hf-token` alone: the saved login) and a prompt that looks like a flag stay as they were
    let o = glyd(&["run", "m", "--prompt", "--api-key", "--hf-token"], &path, &t, &[("GLYD_API_KEY", "FROMENV")]);
    assert_eq!(text(&o.stdout).trim(), "engine: run m --prompt --api-key --hf-token key=FROMENV token=", "{}", text(&o.stderr));
    // where there is a tool the command is the tool's, whole: it moves the secrets itself (engine.py), before the engine is started
    python_tool(&py, r#"echo "python glyd: args=$*""#);
    let both = format!("{}:{}:/usr/bin:/bin", en.display(), py.display());
    let o = glyd(&["serve", "m", "--api-key", "K2", "--", "--hf-token", "T2"], &both, &t, &[]);
    assert_eq!(text(&o.stdout).trim(), "python glyd: args=serve m --api-key K2 -- --hf-token T2", "{}", text(&o.stderr));
}

#[test]
fn a_rust_only_install_passes_a_models_commands_to_the_engine() {
    // (install.sh's default on Linux with an NVIDIA GPU: this program, glyd-engine and the GPU library in one folder, no Python tool)
    let t = tmp("rust-only");
    let en = t.join("enbin");
    engine(&en);
    let path = en.display().to_string();
    for args in [
        &["fetch", "glyd/Qwen3.5-9B-linnet"][..],
        &["verify", "glyd/Qwen3.5-9B-linnet"][..],
        &["preflight", "Qwen/Qwen3.5-9B:linnet"][..],
        &["costs", "glyd/Qwen3.5-9B-linnet", "--predict"][..],
        &["supports", "m"][..],
        &["prepare", "m"][..],
    ] {
        let o = glyd(args, &path, &t, &[]);
        assert_eq!(text(&o.stdout).trim(), format!("engine: {}", args.join(" ")), "{args:?}: {}", text(&o.stderr));
    }
    // with the Python tool, a model's command stays the tool's (it has the engine's choice to make)
    let py = t.join("pybin");
    python_tool(&py, STUB);
    let o = glyd(&["fetch", "m"], &format!("{}:{}", en.display(), py.display()), &t, &[]);
    assert!(text(&o.stdout).contains("python glyd:") && text(&o.stdout).contains("args=fetch m"), "{}", text(&o.stdout));
    // no engine and no tool: a file named fetch is still a file to compress, not a command
    let o = glyd(&["fetch"], "/nonexistent", &t, &[]);
    assert!(!text(&o.stdout).starts_with("engine:"), "{}", text(&o.stdout));
}

#[test]
fn fit_and_pull_are_the_engines_with_or_without_the_tool_and_say_so_where_there_is_none() {
    let t = tmp("fit-pull");
    let en = t.join("enbin");
    engine(&en);
    let py = t.join("pybin");
    python_tool(&py, STUB);
    // the engine's, though the Python tool is installed: it has no `fit` (or `pull`) to run
    for path in [en.display().to_string(), format!("{}:{}", py.display(), en.display())] {
        for args in [&["fit", "Qwen/Qwen3-8B", "--context", "4096"][..], &["pull", "Qwen/Qwen3-0.6B"][..], &["rangeworker", "--layers", "0..4"][..]] {
            let o = glyd(args, &path, &t, &[]);
            assert_eq!(text(&o.stdout).trim(), format!("engine: {}", args.join(" ")), "{args:?}: {}", text(&o.stderr));
        }
    }
    // no engine: a name that is no file says what is missing (it was "Unknown option" or a read error), a file of that name is compressed
    let o = glyd(&["fit", "Qwen/Qwen3-8B"], "/nonexistent", &t, &[]);
    assert_eq!(o.status.code(), Some(127), "{}", text(&o.stderr));
    assert!(text(&o.stderr).contains("the engine is not here") && !text(&o.stderr).contains("Unknown option"), "{}", text(&o.stderr));
    assert!(text(&o.stdout).is_empty());
    // GLYD_ENGINE=0 keeps the engine out of it
    let o = glyd(&["pull", "m"], &en.display().to_string(), &t, &[("GLYD_ENGINE", "0")]);
    assert_eq!(o.status.code(), Some(127), "{}", text(&o.stderr));
}

#[test]
fn glyd_alone_explains_the_models_first_and_documents_its_exit_status() {
    let t = tmp("usage");
    let o = glyd(&[], "/nonexistent", &t, &[]);
    let err = text(&o.stderr);
    let (models, files) = (err.find("glyd run MODEL").expect("run"), err.find("compress or decompress a file").expect("files"));
    assert!(models < files, "{err}");
    for cmd in ["glyd serve", "glyd pull", "glyd ls", "glyd fit", "glyd doctor", "Exit status: 0 done, 1 an error, 3 not supported"] {
        assert!(err.contains(cmd), "{cmd}: {err}");
    }
}
