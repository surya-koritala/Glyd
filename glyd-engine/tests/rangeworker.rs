//! `glyd-engine rangeworker` as the build ships it, with no GPU and no weights (CI has no GPU): it starts on a model directory (Qwen3.8-27B's config.json and a
//! tiny tokenizer), answers describe, tokenize and detokenize, says that nothing is loaded before prepare, refuses another incarnation's handle and ends at exit.
//! The GPU's part, every logit against the whole model's, is `rangecheck --workers` (a development build's; docs/range-worker.md). RANGEWORKER_EXE: another
//! build of the program to run (a release build made apart).

use glyd_engine::dynamic::range::{read_frame, write_frame, Frame, MAX_BODY};
use glyd_gpu::json::{self, Value};
use std::process::{Command, Stdio};

#[test]
fn the_worker_answers_without_a_gpu() {
    let dir = std::env::temp_dir().join(format!("glyd-rangeworker-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(concat!(env!("CARGO_MANIFEST_DIR"), "/../research/lossy-2026/qwen38-config.json"), dir.join("config.json")).unwrap();
    // (tok.rs's tiny tiktoken model: a, b, c, space, ab, " a", abc)
    let ranks = ["YQ==", "Yg==", "Yw==", "IA==", "YWI=", "IGE=", "YWJj"];
    std::fs::write(dir.join("tiktoken.model"), ranks.iter().enumerate().map(|(i, t)| format!("{t} {i}\n")).collect::<String>()).unwrap();
    let exe = std::env::var("RANGEWORKER_EXE").unwrap_or_else(|_| env!("CARGO_BIN_EXE_glyd-engine").to_string());
    let mut child = Command::new(&exe).arg("rangeworker").arg("--model").arg(&dir).args(["--layers", "32..64"]).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().unwrap();
    let mut inp = child.stdin.take().unwrap();
    let mut out = std::io::BufReader::new(child.stdout.take().unwrap());
    let mut call = |head: &str| -> Value {
        write_frame(&mut inp, &Frame { head: json::parse(head).unwrap(), body: Vec::new() }).unwrap();
        read_frame(&mut out, MAX_BODY).unwrap().expect("a reply").head
    };
    let s = |v: &Value, path: &[&str]| path.iter().fold(Some(v), |v, k| v.and_then(|v| v.get(k))).map(json::to_compact).unwrap_or_default();

    let d = call(r#"{"op":"describe","id":"d1"}"#);
    assert_eq!(s(&d, &["protocol"]), r#""glyd-range/1""#, "{}", json::to_compact(&d));
    assert_eq!((s(&d, &["model", "layers"]), s(&d, &["model", "hidden"]), s(&d, &["stage", "last"]), s(&d, &["limits", "head_rows"])), ("64".into(), "5120".into(), "true".into(), "1".into()));
    assert!(s(&d, &["ops"]).contains(r#""tokenize","detokenize""#));
    // A config-only model is penguin, where verify chunks are not checked yet: none.
    assert_eq!((s(&d, &["limits", "verify_rows"]), s(&d, &["limits", "sessions"])), ("0".into(), "1".into()));
    let t = call(r#"{"op":"tokenize","id":"t1","text":"abc ab"}"#);
    assert_eq!(s(&t, &["ids"]), "[6,3,4]", "{}", json::to_compact(&t));
    let x = call(r#"{"op":"detokenize","id":"t2","ids":[6,3,4]}"#);
    assert_eq!(s(&x, &["text"]), r#""abc ab""#);
    let i = call(r#"{"op":"inspect","id":"i1"}"#);
    assert_eq!(s(&i, &["available"]), "false");
    let r = call(r#"{"op":"run_chunk","id":"r1","session":"s-0000000000000000-1","phase":"decode","position_start":0,"rows":1}"#);
    assert!(s(&r, &["ok"]) == "false" && s(&r, &["error"]).contains("incarnation"), "{}", json::to_compact(&r));
    assert_eq!(s(&call(r#"{"op":"exit","id":"e1"}"#), &["bye"]), "true");
    assert!(child.wait().unwrap().success());
    // --sessions 2: describe says it; past 2 the worker does not start
    let mut two = Command::new(&exe).arg("rangeworker").arg("--model").arg(&dir).args(["--layers", "0..32", "--sessions", "2"]).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().unwrap();
    let mut inp = two.stdin.take().unwrap();
    let mut out = std::io::BufReader::new(two.stdout.take().unwrap());
    write_frame(&mut inp, &Frame { head: json::parse(r#"{"op":"describe","id":"d2"}"#).unwrap(), body: Vec::new() }).unwrap();
    let d = read_frame(&mut out, MAX_BODY).unwrap().expect("a reply").head;
    assert_eq!(s(&d, &["limits", "sessions"]), "2", "{}", json::to_compact(&d));
    drop(inp);
    assert!(two.wait().unwrap().success());
    let three = Command::new(&exe).arg("rangeworker").arg("--model").arg(&dir).args(["--sessions", "3"]).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).status().unwrap();
    assert!(!three.success());
    let _ = std::fs::remove_dir_all(&dir);
}
