use serde_json::{Value, json};
use starlings_security::{PrivateDir, catalog::Catalog};
use std::process::Command;

#[test]
fn local_catalog_flow_preserves_explicit_approval_and_has_no_inference_claim() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let state = root.join("state");
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_starlings"))
            .arg("--state-dir")
            .arg(&state)
            .args(args)
            .output()
            .unwrap()
    };
    assert!(
        run(&["cluster", "init", "--listen", "127.0.0.1:0"])
            .status
            .success()
    );
    let source = root.join("source");
    let dir = PrivateDir::ensure(&source).unwrap();
    let mut draft: Value =
        serde_json::from_slice(include_bytes!("../../../examples/model-manifest.json")).unwrap();
    draft["fixture"] = json!(false);
    draft["source"]["revision"] = json!("a".repeat(40));
    for asset in draft["assets"].as_array_mut().unwrap() {
        let path = asset["path"].as_str().unwrap().to_owned();
        asset["size_bytes"] = json!(path.len().to_string());
        dir.write_atomic_durable(&path, path.as_bytes()).unwrap();
    }
    // Release the Windows creator handle before opening a contained read-only source.
    drop(dir);
    let file = root.join("manifest.json");
    std::fs::write(&file, serde_json::to_vec(&draft).unwrap()).unwrap();
    let sealed = run(&[
        "model",
        "seal",
        file.to_str().unwrap(),
        source.to_str().unwrap(),
    ]);
    assert!(
        sealed.status.success(),
        "{}",
        String::from_utf8_lossy(&sealed.stderr)
    );
    std::fs::write(&file, &sealed.stdout).unwrap();
    let manifest: Value = serde_json::from_slice(&sealed.stdout).unwrap();
    let id = manifest["manifest_digest"].as_str().unwrap();
    assert!(
        run(&["model", "register", file.to_str().unwrap()])
            .status
            .success()
    );
    assert!(!run(&["model", "check", id]).status.success());
    assert!(
        !run(&["model", "approve", id, "--redistribute"])
            .status
            .success()
    );
    assert!(run(&["model", "approve", id]).status.success());
    let ready = run(&["model", "check", id]);
    assert!(ready.status.success());
    let ready: Value = serde_json::from_slice(&ready.stdout).unwrap();
    assert_eq!(ready["inference_validated"], false);
    assert_eq!(ready["artifact_readiness"], "unchecked");
    let cost = run(&["model", "cost", id, "0", "1"]);
    assert!(cost.status.success());
    let cost: Value = serde_json::from_slice(&cost.stdout).unwrap();
    assert!(cost["backend_peak_host_bytes"].is_null());
    assert_eq!(cost["transfer_buffer_bytes"], "65536");
    assert_eq!(
        cost["max_asset_import_reserve_bytes"],
        (2 * manifest["assets"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a["size_bytes"].as_str().unwrap().parse::<u64>().unwrap())
            .max()
            .unwrap())
        .to_string()
    );
    assert!(!run(&["model", "ready", id]).status.success());
    assert!(
        run(&["model", "import", id, source.to_str().unwrap()])
            .status
            .success()
    );
    assert!(run(&["model", "ready", id]).status.success());
    let cache = state.join("cache/models");
    let owner = Catalog::open(&cache).unwrap();
    let concurrent = run(&["model", "import", id, source.to_str().unwrap()]);
    assert!(
        !concurrent.status.success(),
        "a second process must not publish while the catalog is owned"
    );
    drop(owner);
    let before = run(&["model", "storage"]);
    assert!(before.status.success());
    assert!(
        run(&["model", "import", id, source.to_str().unwrap()])
            .status
            .success()
    );
    let after = run(&["model", "storage"]);
    assert!(after.status.success());
    assert_eq!(
        serde_json::from_slice::<Value>(&before.stdout).unwrap(),
        serde_json::from_slice::<Value>(&after.stdout).unwrap()
    );
    let lease = "00000000-0000-4000-8000-000000000991";
    assert!(run(&["model", "pin", id, lease]).status.success());
    assert!(run(&["model", "gc"]).status.success());
    assert!(run(&["model", "ready", id]).status.success());
    assert!(run(&["model", "unpin", lease]).status.success());
    assert!(run(&["model", "gc"]).status.success());
    assert!(!run(&["model", "ready", id]).status.success());
    assert!(run(&["model", "revoke", id]).status.success());
    assert!(!run(&["model", "check", id]).status.success());
}
