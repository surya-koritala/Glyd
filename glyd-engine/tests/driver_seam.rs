//! R1a (#134): the GPU driver is called from the device module alone. `ffi.rs` holds the driver's table and `dev.rs` is the CUDA (and HIP) device layer over it; every
//! other file asks `dev` for what it needs (a stream, an event, a thread bound to a GPU, peer access), so that a second backend is one module and not a search of the tree.

use std::path::Path;

/// The files that may call `driver()`.
const DEVICE_MODULE: [&str; 2] = ["src/dev.rs", "src/ffi.rs"];

fn sources(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            sources(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// The places in `text` where `driver()` is called: the word on its own (not the end of a longer name), outside a line comment.
fn calls(text: &str) -> Vec<usize> {
    text.lines()
        .enumerate()
        .filter(|(_, l)| {
            let code = l.split("//").next().unwrap_or("");
            code.match_indices("driver()").any(|(i, _)| code[..i].chars().next_back().is_none_or(|c| !(c.is_alphanumeric() || c == '_')))
        })
        .map(|(n, _)| n + 1)
        .collect()
}

#[test]
fn the_driver_is_called_from_the_device_module_alone() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    sources(&root.join("src"), &mut files);
    let mut found = Vec::new();
    for f in &files {
        let rel = f.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
        if DEVICE_MODULE.contains(&rel.as_str()) {
            continue;
        }
        for line in calls(&std::fs::read_to_string(f).unwrap()) {
            found.push(format!("{rel}:{line}"));
        }
    }
    assert!(found.is_empty(), "driver() called outside {DEVICE_MODULE:?} (ask dev.rs for it instead): {found:?}");
    // (the scan finds what it is for)
    assert_eq!(calls("let d = driver()?;\nfn update_the_driver() {}\n// driver() in a comment\nif let Ok(d) = crate::ffi::driver() {"), vec![1, 4]);
}
