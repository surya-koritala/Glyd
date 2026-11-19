use super::*;
use serde_json::{Value, json};

const NODE: &str = "00000000-0000-4000-8000-000000000001";
const SECRET: &str = "PRIVATE-SENTINEL-NOT-FOR-EXPORT";
fn reports() -> [AdminReply; 3] {
    [
        AdminReply::ok("ENROLLED_IDLE", format!(
            "role=coordinator node={NODE} cluster={NODE} enrollment=192.0.2.1:9741 connections=3 handshakes=1 pending_admissions=2 pending_approvals=4 inference=127.0.0.1:9740 registry=active epoch=9 routing={SECRET}"
        )),
        AdminReply::ok("NODES", json!([
            {"node":NODE,"role":"worker","status":"online","schedulable":true,"fresh_links":1},
            {"node":"00000000-0000-4000-8000-000000000002","role":"controller","status":"offline","schedulable":false,"fresh_links":0}
        ]).to_string()),
        AdminReply::ok("SERVICE", json!({
            "state":"serving", "fixture":true, "inference_qualified":true,
            "process_group_absent":false,"cleanup_proven":false,"reserved_bytes":"123",
            "endpoint":"127.0.0.1:9740", "pin":NODE,
            "error":SECRET,"startup_error":SECRET,"model_architecture":SECRET,
            "backend_version_declared":SECRET,"execution_profile_declared":SECRET,
            "startup_properties":{"path":SECRET,"prompt":SECRET,"token":SECRET}
        }).to_string()),
    ]
}
fn bundle(reports: [Option<AdminReply>; 3]) -> Value {
    let mut replies = reports.into_iter();
    let mut calls = vec![];
    let result = collect_with(|command| {
        calls.push(command.to_owned());
        replies.next().unwrap()
    });
    assert_eq!(calls, ["status", "nodes", "service"]);
    assert!(replies.next().is_none());
    let mut bytes = Vec::new();
    write(&result, &mut bytes).unwrap();
    assert!(bytes.len() < 4096);
    let text = std::str::from_utf8(&bytes).unwrap();
    for private in [
        SECRET,
        NODE,
        "192.0.2.1",
        "127.0.0.1",
        "startup_properties",
        "\"endpoint\"",
        "backend_version_declared",
    ] {
        assert!(
            !text.contains(private),
            "private report field reached export"
        );
    }
    serde_json::from_slice(&bytes).unwrap()
}
#[test]
fn diagnostic_projection_is_bounded_redacted_and_uses_only_three_queries() {
    let value = bundle(reports().map(Some));
    assert_eq!(value["qualification"], "not_assessed");
    assert_eq!(value["routing"], "whole_request");
    assert_eq!(value["layer_sharding"], "not_qualified");
    assert_eq!(value["daemon"]["data"]["connections"], 3);
    assert_eq!(value["daemon"]["data"]["pending_admissions"], 2);
    assert_eq!(value["members"]["data"]["workers"], 1);
    assert_eq!(value["members"]["data"]["clients"], 1);
    assert_eq!(value["members"]["data"]["online"], 1);
    assert_eq!(value["members"]["data"]["offline"], 1);
    assert_eq!(value["members"]["data"]["schedulable_workers"], 1);
    assert_eq!(value["service"]["data"]["state"], "serving");
    assert_eq!(value["service"]["data"]["reserved_bytes"], "123");

    for state in [
        "waiting_for_registry",
        "verifying_catalog",
        "loading",
        "ready_unqualified",
        "ready",
        "serving",
        "stopping",
        "stopped",
        "recovered",
        "stopped_pins_retained",
        "failed_pins_retained",
        "stopped_cleanup_proven",
        "failed_cleanup_proven",
    ] {
        let mut reports = reports();
        let mut service: Value = serde_json::from_str(&reports[2].message).unwrap();
        service["state"] = state.into();
        service["reserved_bytes"] = Value::Null;
        reports[2].message = service.to_string();
        let value = bundle(reports.map(Some));
        assert_eq!(value["service"]["status"], "available");
        assert_eq!(value["service"]["data"]["state"], state);
        assert!(value["service"]["data"]["reserved_bytes"].is_null());
    }
    let mut reports = reports();
    reports[0].code = "DRAINING".into();
    reports[2].message = "null".into();
    let value = bundle(reports.map(Some));
    assert_eq!(value["daemon"]["data"]["draining"], true);
    assert_eq!(value["service"]["status"], "available");
    assert!(value["service"]["data"].is_null());
}

#[test]
fn diagnostic_failures_are_explicit_never_raw_reports_or_successful_zeroes() {
    let missing = bundle([None, Some(AdminReply::error(SECRET.into())), None]);
    for section in ["daemon", "members", "service"] {
        assert_eq!(missing[section]["status"], "unavailable");
        assert!(missing[section].get("data").is_none());
    }
    for section in 0..3 {
        let mut reports = reports();
        reports[section].code = SECRET.into();
        let value = bundle(reports.map(Some));
        assert_eq!(
            value[["daemon", "members", "service"][section]]["status"],
            "invalid_report"
        );
        for other in 0..3 {
            if other != section {
                assert_eq!(
                    value[["daemon", "members", "service"][other]]["status"],
                    "available"
                );
            }
        }
    }
    for text in [format!("role={SECRET}"), "role=worker role=coordinator".into(), "x".repeat(16385), "role=worker registry=online connections=4294967296 handshakes=0 pending_admissions=0 pending_approvals=0".into()] {
        let mut reports = reports();
        reports[0].message = text;
        assert_eq!(bundle(reports.map(Some))["daemon"]["status"], "invalid_report");
    }
    for text in [
        format!("{{\"state\":\"{SECRET}\"}}"),
        "{\"state\":\"ready\",\"state\":\"serving\"}".into(),
        "{\"state\":\"ready\"}".into(),
        "[]".into(),
    ] {
        let mut reports = reports();
        reports[2].message = text;
        assert_eq!(
            bundle(reports.map(Some))["service"]["status"],
            "invalid_report"
        );
    }
    let mut changed = reports();
    let mut service: Value = serde_json::from_str(&changed[2].message).unwrap();
    service["state"] = SECRET.into();
    changed[2].message = service.to_string();
    assert_eq!(
        bundle(changed.map(Some))["service"]["status"],
        "invalid_report"
    );
    let mut duplicate = reports();
    let node: Value = serde_json::from_str::<Value>(&duplicate[1].message).unwrap()[0].clone();
    duplicate[1].message = json!(vec![node.clone(); 2]).to_string();
    assert_eq!(
        bundle(duplicate.map(Some))["members"]["status"],
        "invalid_report"
    );
    let full: Vec<_> = (1..=65)
        .map(|index| {
            let mut node = node.clone();
            node["node"] = format!("00000000-0000-4000-8000-{index:012}").into();
            node
        })
        .collect();
    let mut complete = reports();
    complete[1].message = json!(full[..64]).to_string();
    assert_eq!(bundle(complete.map(Some))["members"]["data"]["workers"], 64);
    let mut reports = reports();
    reports[1].message = json!(full).to_string();
    assert_eq!(
        bundle(reports.map(Some))["members"]["status"],
        "invalid_report"
    );

    struct Broken;
    impl Write for Broken {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::ErrorKind::BrokenPipe.into())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    assert_eq!(
        write(&collect_with(|_| None), Broken),
        Err("DIAGNOSTIC_WRITE_FAILED")
    );
}
