//! The file codec's command words: `glyd compress FILE -o OUT` and `glyd decompress FILE -o OUT` are `-c` and `-d`, and what they write is what the old forms write, byte for byte (no format changed);
//! a file named `compress` is still a file to compress (`./compress`).

use std::fs;
use std::process::Command;

fn glyd(dir: &std::path::Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_glyd")).current_dir(dir).env("GLYD_FORWARDED", "").args(args).output().unwrap()
}

#[test]
fn compress_and_decompress_are_the_codecs_words() {
    let d = std::env::temp_dir().join(format!("glyd-cli-codec-{}", std::process::id()));
    fs::create_dir_all(&d).unwrap();
    let text: Vec<u8> =
        (0..4000).flat_map(|i| format!("2026-11-20 10:{:02}:{:02} INFO request {i} served in {} ms\n", i / 60 % 60, i % 60, 3 + i % 17).into_bytes()).collect();
    fs::write(d.join("in.log"), &text).unwrap();
    // the words and the flags write the same bytes
    assert!(glyd(&d, &["compress", "in.log", "-o", "a.glyd"]).status.success());
    assert!(glyd(&d, &["-c", "in.log", "-o", "b.glyd"]).status.success());
    assert!(glyd(&d, &["in.log", "-o", "c.glyd"]).status.success());
    let a = fs::read(d.join("a.glyd")).unwrap();
    assert!(a.len() < text.len() / 2, "{} bytes of {}", a.len(), text.len());
    assert_eq!(a, fs::read(d.join("b.glyd")).unwrap());
    assert_eq!(a, fs::read(d.join("c.glyd")).unwrap());
    // and decompress gives the file back, from the words or the flag
    assert!(glyd(&d, &["decompress", "a.glyd", "-o", "out1.log"]).status.success());
    assert!(glyd(&d, &["-d", "a.glyd", "-o", "out2.log"]).status.success());
    assert_eq!(fs::read(d.join("out1.log")).unwrap(), text);
    assert_eq!(fs::read(d.join("out2.log")).unwrap(), text);
    // a levels' flag goes with the word
    assert!(glyd(&d, &["compress", "--max", "in.log", "-o", "m.glyd"]).status.success());
    assert!(glyd(&d, &["decompress", "m.glyd", "-o", "m.log"]).status.success());
    assert_eq!(fs::read(d.join("m.log")).unwrap(), text);
    // a file named compress is a file, as `./run` is
    fs::write(d.join("compress"), &text).unwrap();
    assert!(glyd(&d, &["compress", "-o", "cf.glyd"]).status.success());
    assert_eq!(fs::read(d.join("cf.glyd")).unwrap(), a);
    // the usage says what the program is for, the engine's commands first
    let help = String::from_utf8_lossy(&glyd(&d, &["--help"]).stderr).into_owned();
    let (engine, codec) = (help.find("glyd run MODEL").expect("run"), help.find("glyd compress|decompress").expect("compress"));
    assert!(engine < codec && help.contains("docs/codec/README.md"), "{help}");
    let _ = fs::remove_dir_all(&d);
}
