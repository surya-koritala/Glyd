use std::process::Command;

#[test]
fn offline_calibration_is_file_only_and_never_implicitly_allows_fixtures() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("calibration.json");
    let source: serde_json::Value =
        serde_json::from_slice(include_bytes!("../../../examples/mode-preview.json")).unwrap();
    let input = serde_json::json!({"version":1,"fixture":true,"placement":source["variants"][0]["placement"],
        "candidate_index":0,"artifacts":[],"observations":[]});
    std::fs::write(&path, serde_json::to_vec(&input).unwrap()).unwrap();
    let run = |allow| {
        let mut c = Command::new(env!("CARGO_BIN_EXE_starlings"));
        c.args(["plan", "calibrate"]).arg(&path);
        if allow {
            c.arg("--allow-fixtures");
        }
        c.output().unwrap()
    };
    let refused = run(false);
    assert!(!refused.status.success());
    assert!(refused.stdout.is_empty());
    let output = run(true);
    if cfg!(feature = "test-backend") {
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(report["fixture"], true);
        assert_eq!(report["qualification"], "not_assessed");
        assert_eq!(report["admission"], false);
        assert_eq!(report["metrics"].as_array().unwrap().len(), 4);
        assert!(
            report["metrics"]
                .as_array()
                .unwrap()
                .iter()
                .all(|m| m["observations"].as_array().unwrap().is_empty())
        );
    } else {
        assert_eq!(output.status.code(), Some(2));
    }
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
}

#[test]
fn offline_modes_rejects_fixtures_by_default_and_keeps_file_inputs_bounded() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("modes.json");
    let bytes = include_bytes!("../../../examples/mode-preview.json");
    std::fs::write(&path, bytes).unwrap();
    let run = |allow| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_starlings"));
        command.args(["plan", "modes"]).arg(&path);
        if allow {
            command.arg("--allow-fixtures");
        }
        command.output().unwrap()
    };
    let refused = run(false);
    assert!(!refused.status.success());
    assert!(refused.stdout.is_empty());
    let allowed = run(true);
    if cfg!(feature = "test-backend") {
        assert!(
            allowed.status.success(),
            "{}",
            String::from_utf8_lossy(&allowed.stderr)
        );
        let report: serde_json::Value = serde_json::from_slice(&allowed.stdout).unwrap();
        assert_eq!(report["admission"], false);
        assert_eq!(report["qualification"], "not_assessed");
        assert_eq!(report["winner"]["mode"], "expert_offload");
        assert_eq!(report["runner_up"]["mode"], "whole_request");
    } else {
        assert_eq!(allowed.status.code(), Some(2));
    }
    let mut unknown: serde_json::Value = serde_json::from_slice(bytes).unwrap();
    unknown["unexpected"] = serde_json::json!(1);
    for invalid in [
        b"{\"version\":1,\"version\":1}".to_vec(),
        serde_json::to_vec(&unknown).unwrap(),
        vec![b' '; starlings_protocol::MAX_JSON_BYTES + 1],
    ] {
        std::fs::write(&path, invalid).unwrap();
        assert!(
            starlings_cli::planning::modes_file(&path, starlings_protocol::FixtureMode::Simulation)
                .is_err()
        );
    }
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
}

#[test]
fn offline_benchmark_inspection_verifies_evidence_and_preserves_failed_requests() {
    use starlings_protocol::{FixtureMode, byte_digest};
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let inventory = b"synthetic inventory";
    let raw = b"synthetic redacted failure log";
    std::fs::write(root.join("hardware.json"), inventory).unwrap();
    std::fs::write(root.join("raw.json"), raw).unwrap();
    let id = "00000000-0000-4000-8000-000000000001";
    let identity = byte_digest(b"synthetic backend/model/workload");
    let mut value = serde_json::json!({
        "kind":"BenchmarkRun", "schema_version":1, "fixture":true, "run_id":id,
        "status":"failed", "compute_commit":"1".repeat(40),
        "backend_build_digests":[identity], "manifest_digest":identity,
        "execution_profile":"synthetic", "plan_digest":identity, "baseline_run_id":null,
        "hardware_inventory_path":"hardware.json", "hardware_inventory_sha256":byte_digest(inventory),
        "workload_sha256":identity, "network_profile":"loopback-fixture", "concurrency":1,
        "started_at":"2026-10-15T00:00:00Z", "timing_source":"client-monotonic",
        "definitions_version":"benchmark-v1", "warmup_requests_excluded":0,
        "requests":[{"request_id":id,"outcome":"timeout","prompt_tokens":1,"output_tokens":0,
            "ttft_ms":null,"duration_ms":1,"inter_token_ms":[],"error_code":"TIMEOUT"},
            {"request_id":"00000000-0000-4000-8000-000000000002","outcome":"success",
            "prompt_tokens":1,"output_tokens":1,"ttft_ms":0.5,"duration_ms":2,
            "inter_token_ms":[],"error_code":null}],
        "peak_memory_pools":[{"node_id":id,"pool_id":"host","peak_bytes":"1"}],
        "raw_logs_path":"raw.json", "raw_logs_sha256":byte_digest(raw),
        "notes":"private fixture note never echoed"
    });
    let path = root.join("run.json");
    let write =
        |v: &serde_json::Value| std::fs::write(&path, serde_json::to_vec(v).unwrap()).unwrap();
    write(&value);
    let run = |allow: bool| {
        let mut c = Command::new(env!("CARGO_BIN_EXE_starlings"));
        c.args(["benchmark", "inspect"]).arg(&path);
        if allow {
            c.arg("--allow-fixtures");
        }
        c.output().unwrap()
    };
    let refused = run(false);
    assert!(!refused.status.success());
    assert!(refused.stdout.is_empty());
    let result = run(true);
    if cfg!(feature = "test-backend") {
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
        assert_eq!(report["qualification"], "not_assessed");
        assert_eq!(report["fixture"], true);
        assert_eq!(report["run_status"], "failed");
        assert_eq!(report["outcomes"]["timeout"], 1);
        assert_eq!(report["outcomes"]["success"], 1);
        assert_eq!(report["failures"][0]["error_code"], "TIMEOUT");
        assert_eq!(report["verified_artifacts"], 2);
        assert_eq!(
            report["record_sha256"],
            byte_digest(&std::fs::read(&path).unwrap()).as_str()
        );
        assert!(!String::from_utf8_lossy(&result.stdout).contains("private fixture note"));
    } else {
        assert_eq!(result.status.code(), Some(2));
    }
    let inspected = starlings_cli::benchmark::inspect_file(&path, FixtureMode::Simulation).unwrap();
    let report = serde_json::to_value(inspected).unwrap();
    assert_eq!(report["latency"]["success"]["duration_ms"]["samples"], 1);
    assert_eq!(report["latency"]["success"]["duration_ms"]["median"], 2.0);
    assert_eq!(report["latency"]["success"]["ttft_ms"]["median"], 0.5);
    assert_eq!(report["latency"]["timeout"]["duration_ms"]["median"], 1.0);
    assert!(report["latency"]["timeout"]["duration_ms"]["p99"].is_null());
    assert!(report["latency"]["success"]["duration_ms"]["p95"].is_null());
    assert!(report["latency"]["timeout"]["ttft_ms"].is_null());
    assert!(report["latency"]["oom"]["duration_ms"].is_null());
    std::fs::write(root.join("raw.json"), b"changed after record").unwrap();
    if cfg!(feature = "test-backend") {
        let output = run(true);
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert_eq!(
            String::from_utf8(output.stderr).unwrap().trim(),
            "BENCHMARK_EVIDENCE_REFUSED"
        );
    }
    assert_eq!(
        starlings_cli::benchmark::inspect_file(&path, FixtureMode::Simulation).err(),
        Some("BENCHMARK_EVIDENCE_REFUSED")
    );
    std::fs::write(root.join("raw.json"), raw).unwrap();
    value["raw_logs_path"] = serde_json::json!("../outside.json");
    write(&value);
    assert_eq!(
        starlings_cli::benchmark::inspect_file(&path, FixtureMode::Simulation).err(),
        Some("BENCHMARK_RECORD_REFUSED")
    );
}

#[test]
fn offline_scoreboard_preserves_missing_results_and_rejects_implicit_fixtures() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("matrix.json");
    let mut plan: serde_json::Value =
        serde_json::from_str(include_str!("../../../examples/scoreboard-plan.json")).unwrap();
    let run = |allow_fixtures: bool| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_starlings"));
        command
            .current_dir(directory.path())
            .args(["scoreboard", "report"])
            .arg(&path);
        if allow_fixtures {
            command.arg("--allow-fixtures");
        }
        command.output().unwrap()
    };
    std::fs::write(&path, serde_json::to_vec(&plan).unwrap()).unwrap();
    let result = run(false);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["qualification"], "not_assessed");
    assert_eq!(report["cells"].as_array().unwrap().len(), 22);
    assert!(
        report["cells"]
            .as_array()
            .unwrap()
            .iter()
            .all(|cell| { cell["status"] == "not_measured" && cell["rule_satisfied"].is_null() })
    );
    plan["fixture"] = true.into();
    std::fs::write(&path, serde_json::to_vec(&plan).unwrap()).unwrap();
    let result = run(false);
    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    assert!(String::from_utf8_lossy(&result.stderr).contains("SCOREBOARD_FIXTURE_REFUSED"));
    let result = run(true);
    if cfg!(feature = "test-backend") {
        assert!(result.status.success());
        let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
        assert_eq!(report["fixture"], true);
        assert_eq!(report["qualification"], "not_assessed");
    } else {
        assert_eq!(result.status.code(), Some(2));
        assert!(result.stdout.is_empty());
    }
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
}

#[test]
fn cli_reports_its_scope_and_rejects_unsupported_arguments() {
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_starlings"))
            .args(args)
            .output()
            .expect("CLI starts")
    };
    for args in [vec![], vec!["--help"], vec!["-h"]] {
        let result = run(&args);
        assert!(result.status.success());
        assert!(
            String::from_utf8_lossy(&result.stdout)
                .contains("explicitly approved local qualification")
        );
    }
    let result = run(&["--version"]);
    assert!(result.status.success());
    assert_eq!(
        String::from_utf8_lossy(&result.stdout).trim(),
        concat!("starlings ", env!("CARGO_PKG_VERSION"))
    );
    let result = run(&["doctor"]);
    assert!(result.status.success());
    assert!(String::from_utf8_lossy(&result.stdout).contains("inference=unavailable"));
    for args in [
        vec!["serve"],
        vec!["doctor", "ignored"],
        vec!["--version", "extra"],
    ] {
        let result = run(&args);
        assert_eq!(result.status.code(), Some(2));
        assert!(result.stdout.is_empty());
    }
    let result = run(&["fake-smoke"]);
    if cfg!(feature = "test-backend") {
        assert!(result.status.success());
        assert!(String::from_utf8_lossy(&result.stdout).contains("simulation only"));
    } else {
        assert_eq!(result.status.code(), Some(2));
        assert!(result.stdout.is_empty());
    }
}

#[test]
fn multiprocess_simulation_is_explicit_and_fault_checks_pass() {
    let result = Command::new(env!("CARGO_BIN_EXE_starlings"))
        .arg("simulate")
        .output()
        .expect("CLI starts");
    if cfg!(feature = "test-backend") {
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(
            String::from_utf8_lossy(&result.stdout)
                .contains("simulation only: 8 cluster scenarios passed")
        );
        assert!(result.stderr.is_empty());
    } else {
        assert_eq!(result.status.code(), Some(2));
        assert!(result.stdout.is_empty());
    }
}

#[test]
fn host_probe_never_advertises_inference_from_hardware_alone() {
    let result = Command::new(env!("CARGO_BIN_EXE_starlings"))
        .args(["probe", "."])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["inference"], "unavailable");
    assert_eq!(report["ttl_ms"], 5000);
    assert_eq!(report["thermal_throttled"]["quality"], "unavailable");
    assert_eq!(report["nvidia"]["quality"], "unavailable");
    assert!(report["memory_pools"].as_array().unwrap().len() <= 1);
    // Wrong approval rejects an actual local file before any native loader/device call.
    let file = tempfile::NamedTempFile::new().unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_starlings"))
        .args(["probe", ".", "--nvidia"])
        .arg(file.path())
        .arg("0".repeat(64))
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    let expected = if cfg!(any(target_os = "linux", target_os = "windows")) {
        "NVIDIA_LIBRARY_UNAPPROVED"
    } else {
        "NVIDIA_PROBE_UNSUPPORTED_OS"
    };
    assert!(String::from_utf8_lossy(&result.stderr).contains(expected));
}

#[test]
fn collector_cli_refuses_fixture_execution_by_default_and_never_echoes_inputs() {
    let temp = tempfile::tempdir().unwrap();
    let mut matrix: serde_json::Value =
        serde_json::from_str(include_str!("../../../examples/scoreboard-plan.json")).unwrap();
    matrix["fixture"] = true.into();
    std::fs::write(
        temp.path().join("matrix.json"),
        serde_json::to_vec(&matrix).unwrap(),
    )
    .unwrap();
    std::fs::write(
        temp.path().join("plan.json"),
        b"private invalid plan - never printed",
    )
    .unwrap();
    let run = |allow: bool| {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_starlings"));
        cmd.current_dir(temp.path()).args([
            "scoreboard",
            "collect",
            "matrix.json",
            "plan.json",
            "output.json",
        ]);
        if allow {
            cmd.arg("--allow-fixtures");
        }
        cmd.output().unwrap()
    };
    let rejected = run(false);
    assert!(!rejected.status.success());
    assert!(rejected.stdout.is_empty());
    assert_eq!(
        String::from_utf8(rejected.stderr).unwrap().trim(),
        "SCOREBOARD_FIXTURE_REFUSED"
    );
    let allowed = run(true);
    assert!(!allowed.status.success());
    assert!(allowed.stdout.is_empty());
    if cfg!(feature = "test-backend") {
        assert_eq!(
            String::from_utf8(allowed.stderr).unwrap().trim(),
            "COLLECT_INVALID_JSON"
        );
    } else {
        assert_eq!(allowed.status.code(), Some(2));
    }
    assert!(!temp.path().join("output.json").exists());
    assert!(!temp.path().join("output.raw.jsonl").exists());
}

#[test]
fn offline_preview_is_bounded_strict_and_does_not_admit() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("preview.json");
    let manifest: serde_json::Value =
        serde_json::from_slice(include_bytes!("../../../examples/model-manifest.json")).unwrap();
    let input = serde_json::json!({"version":1,"fixture":true,"manifest":manifest,"now_ms":"100","now_wall_ms":0,"nodes":[],"links":[],"objective":"latency","max_stages":8,"top_k":16,"max_states":1000,"max_cost_rows":10000});
    std::fs::write(&path, serde_json::to_vec(&input).unwrap()).unwrap();
    let run = |allow: bool| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_starlings"));
        command
            .current_dir(directory.path())
            .args(["plan", "preview"])
            .arg(&path);
        if allow {
            command.arg("--allow-fixtures");
        }
        command.output().unwrap()
    };
    let refused = run(false);
    assert!(!refused.status.success());
    assert!(refused.stdout.is_empty());
    if cfg!(feature = "test-backend") {
        let result = run(true);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
        assert_eq!(report["admission"], false);
        assert_eq!(report["qualification"], "not_assessed");
        assert_eq!(report["refusals"]["NO_ELIGIBLE_NODE"], 1);
        assert_eq!(report["node_refusals"], serde_json::json!({}));
        assert_eq!(report["candidates"], serde_json::json!([]));
    } else {
        assert_eq!(run(true).status.code(), Some(2));
    }
    for body in [
        "{\"version\":1,\"version\":1}".to_owned(),
        serde_json::to_string(&{
            let mut value = input;
            value["unexpected"] = serde_json::json!(true);
            value
        })
        .unwrap(),
        " ".repeat(starlings_protocol::MAX_JSON_BYTES + 1),
    ] {
        std::fs::write(&path, body).unwrap();
        assert!(
            starlings_cli::planning::preview_file(
                &path,
                starlings_protocol::FixtureMode::Simulation
            )
            .is_err()
        );
    }
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
}
