//! R1a (#134): the GPU driver is called from the device layer alone. `ffi.rs` holds the driver's table, `device.rs` is the device layer's trait (`Device`) and its CUDA
//! (and HIP) implementation over that table, and `dev.rs` (buffers, streams, events, graphs, kernels) asks the trait; every other file asks `dev` for what it needs (a
//! stream, an event, a thread bound to a GPU, peer access), so that a second device layer is one implementation of the trait and not a search of the tree.

use std::path::Path;

/// The files that may call `driver()`.
const DEVICE_MODULE: [&str; 2] = ["src/device.rs", "src/ffi.rs"];

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
    assert!(found.is_empty(), "driver() called outside {DEVICE_MODULE:?} (ask dev.rs, or the device trait in device.rs, for it instead): {found:?}");
    // (the scan finds what it is for)
    assert_eq!(calls("let d = driver()?;\nfn update_the_driver() {}\n// driver() in a comment\nif let Ok(d) = crate::ffi::driver() {"), vec![1, 4]);
}

/// The driver functions of the table, by their names (ffi.rs's `table! { pub struct Driver { fn NAME(...` lines).
fn table_functions(ffi: &str) -> Vec<String> {
    let start = ffi.find("pub struct Driver {").expect("the driver's table in ffi.rs");
    let body = &ffi[start..start + ffi[start..].find("\n    }\n").expect("the table's end")];
    body.lines().filter_map(|l| l.trim().strip_prefix("fn ")).map(|l| l.split('(').next().unwrap_or("").to_string()).collect()
}

/// Every function of the driver's table is reached through the device trait: `device.rs` calls each one (but `cuInit` and `cuGetErrorString`, which `ffi.rs` calls
/// itself when it loads the driver and words a status), so the trait is the whole of what the engine asks of the driver.
#[test]
fn the_device_trait_covers_the_driver_table() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let ffi = std::fs::read_to_string(root.join("src/ffi.rs")).unwrap();
    let device = std::fs::read_to_string(root.join("src/device.rs")).unwrap();
    let names = table_functions(&ffi);
    assert!(names.len() > 40 && names.iter().any(|n| n == "cuLaunchKernel"), "the table's functions not found: {names:?}");
    let missing: Vec<&String> =
        names.iter().filter(|n| !["cuInit", "cuGetErrorString"].contains(&n.as_str()) && !device.contains(&format!(".{n})("))).collect();
    assert!(missing.is_empty(), "driver functions device.rs does not call (add them to the Device trait): {missing:?}");
}
