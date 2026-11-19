use serde_json::{Value, json};
use starlings_protocol::*;
const MODE: FixtureMode = FixtureMode::Simulation;
const NODE: &[u8] = include_bytes!("../../../examples/node-capabilities.json");
const PLAN: &[u8] = include_bytes!("../../../examples/placement-plan.json");
const LEASE: &[u8] = include_bytes!("../../../examples/lease.json");
const MANIFEST: &[u8] = include_bytes!("../../../examples/model-manifest.json");
const TENSOR: &[u8] = include_bytes!("../../../examples/tensor-header.json");
const CHAT: &[u8] = include_bytes!("../../../examples/chat-request.json");

#[test]
fn pipeline_worker_owners_are_unique_even_when_device_names_differ() {
    let mut value: Value = serde_json::from_slice(PLAN).unwrap();
    let bytes = serde_json::to_vec(&value).unwrap();
    assert!(decode::<PlacementPlan>(&bytes, MODE).is_ok());
    // Device names are node-local: the same spelling on two nodes is valid.
    value["stages"][1]["device_id"] = value["stages"][0]["device_id"].clone();
    assert!(decode::<PlacementPlan>(&serde_json::to_vec(&value).unwrap(), MODE).is_ok());
    value["stages"][1]["node_id"] = value["stages"][0]["node_id"].clone();
    for device in ["mock:0", "mock:1"] {
        value["stages"][1]["device_id"] = json!(device);
        assert_eq!(
            decode::<PlacementPlan>(&serde_json::to_vec(&value).unwrap(), MODE)
                .unwrap_err()
                .to_string(),
            "worker owns multiple stages"
        );
    }
}

#[test]
fn benchmark_records_validate_timing_identity_failures_and_fixture_boundaries() {
    let digest = byte_digest(b"synthetic benchmark identity");
    let id = "00000000-0000-4000-8000-000000000001";
    let value = json!({
        "kind":"BenchmarkRun", "schema_version":1, "fixture":true, "run_id":id,
        "status":"measured", "compute_commit":"1".repeat(40),
        "backend_build_digests":[digest], "manifest_digest":digest,
        "execution_profile":"synthetic", "plan_digest":digest, "baseline_run_id":null,
        "hardware_inventory_path":"hardware.json", "hardware_inventory_sha256":digest,
        "workload_sha256":digest, "network_profile":"loopback-fixture", "concurrency":1,
        "started_at":"2026-10-15T00:00:00Z", "timing_source":"client-monotonic",
        "definitions_version":"benchmark-v1", "warmup_requests_excluded":0,
        "requests":[{"request_id":id, "outcome":"success", "prompt_tokens":1,
            "output_tokens":3, "ttft_ms":1.1, "duration_ms":3.3,
            "inter_token_ms":[1.1,1.1], "error_code":null}],
        "peak_memory_pools":[{"node_id":id,"pool_id":"host","peak_bytes":"1"}],
        "raw_logs_path":"raw.json", "raw_logs_sha256":digest, "notes":"synthetic unit data"
    });
    let bytes = serde_json::to_vec(&value).unwrap();
    roundtrip::<BenchmarkRun>(&bytes);
    assert!(decode::<BenchmarkRun>(&bytes, FixtureMode::Reject).is_err());
    for change in [
        "empty",
        "empty_pools",
        "commit",
        "failed",
        "success_code",
        "failure_code",
        "request_id",
        "pool",
        "build",
        "zero",
        "late",
        "negative",
        "missing_ttft",
        "interval_count",
        "interval_sum",
        "decimal",
        "path",
        "ads",
        "backslash",
        "baseline",
        "timestamp",
        "field",
        "fixture",
        "version",
        "zero_output",
    ] {
        invalid::<BenchmarkRun>(&bytes, |v| match change {
            "empty" => v["requests"] = json!([]),
            "empty_pools" => v["peak_memory_pools"] = json!([]),
            "commit" => v["compute_commit"] = json!("0".repeat(40)),
            "failed" => {
                v["requests"][0]["outcome"] = json!("timeout");
                v["requests"][0]["error_code"] = json!("TIMEOUT");
            }
            "success_code" => v["requests"][0]["error_code"] = json!("ERROR"),
            "failure_code" => {
                v["status"] = json!("failed");
                v["requests"][0]["outcome"] = json!("timeout");
            }
            "request_id" => {
                let r = v["requests"][0].clone();
                v["requests"].as_array_mut().unwrap().push(r);
            }
            "pool" => {
                let p = v["peak_memory_pools"][0].clone();
                v["peak_memory_pools"].as_array_mut().unwrap().push(p);
            }
            "build" => v["backend_build_digests"] = json!([digest, digest]),
            "zero" => v["backend_build_digests"] = json!(["0".repeat(64)]),
            "late" => v["requests"][0]["ttft_ms"] = json!(4),
            "negative" => v["requests"][0]["duration_ms"] = json!(-1),
            "missing_ttft" => v["requests"][0]["ttft_ms"] = Value::Null,
            "interval_count" => v["requests"][0]["inter_token_ms"] = json!([1.1]),
            "interval_sum" => v["requests"][0]["inter_token_ms"] = json!([2, 2]),
            "decimal" => v["peak_memory_pools"][0]["peak_bytes"] = json!("18446744073709551616"),
            "path" => v["raw_logs_path"] = json!("../private.json"),
            "ads" => v["raw_logs_path"] = json!("raw.json:private"),
            "backslash" => v["raw_logs_path"] = json!("..\\private.json"),
            "baseline" => v["baseline_run_id"] = json!(id),
            "timestamp" => v["started_at"] = json!("not-a-time"),
            "field" => v["unexpected"] = json!(true),
            "fixture" => {
                v.as_object_mut().unwrap().remove("fixture");
            }
            "version" => v["schema_version"] = json!(2),
            _ => v["requests"][0]["output_tokens"] = json!(0),
        });
    }
    let mut failed = value.clone();
    failed["status"] = json!("failed");
    // Successful requests do not turn a later cleanup/collection failure into success.
    roundtrip::<BenchmarkRun>(&serde_json::to_vec(&failed).unwrap());
    failed["requests"][0]["outcome"] = json!("oom");
    failed["requests"][0]["error_code"] = json!("OOM");
    roundtrip::<BenchmarkRun>(&serde_json::to_vec(&failed).unwrap());
    failed["requests"] = json!([]); // Failure before any request is still retained.
    failed["peak_memory_pools"] = json!([]); // Never fabricate missing peak observations.
    roundtrip::<BenchmarkRun>(&serde_json::to_vec(&failed).unwrap());
    let mut run: BenchmarkRun = decode(&bytes, MODE).unwrap();
    run.requests[0].duration_ms = f64::NAN;
    assert!(run.validate().is_err());
    run.requests[0].duration_ms = 3.3;
    run.requests[0].inter_token_ms[0] = f64::INFINITY;
    assert!(run.validate().is_err());
}

fn roundtrip<T: Contract + PartialEq + std::fmt::Debug>(input: &[u8]) {
    let first: T = decode(input, MODE).unwrap();
    assert_eq!(
        first,
        decode::<T>(&encode(&first, MODE).unwrap(), MODE).unwrap()
    );
}
fn invalid<T: Contract>(input: &[u8], change: impl FnOnce(&mut Value)) {
    let mut value = parse_json(input).unwrap();
    change(&mut value);
    assert!(decode::<T>(&serde_json::to_vec(&value).unwrap(), MODE).is_err());
}
#[test]
fn schemas_types_and_semantics_agree_on_supplied_fixtures() {
    roundtrip::<NodeCapabilities>(NODE);
    roundtrip::<PlacementPlan>(PLAN);
    roundtrip::<Lease>(LEASE);
    roundtrip::<ModelManifest>(MANIFEST);
    roundtrip::<TensorHeader>(TENSOR);
    roundtrip::<ChatRequest>(CHAT);
    let session = json!({"kind":"Session","protocol_version":1,"session_id":"00000000-0000-4000-8000-000000000104",
        "generation_id":"00000000-0000-4000-8000-000000000105","client_id":"00000000-0000-4000-8000-000000000107",
        "lease_id":"00000000-0000-4000-8000-000000000103","plan_digest":"0".repeat(64),"authority_epoch":"1",
        "prompt_token_count":2,"max_output_tokens":8,"context_limit":2048,
        "sampling":{"profile":"sample-v1","temperature":0,"top_p":1,"seed":1},"state":"queued"});
    roundtrip::<Session>(&serde_json::to_vec(&session).unwrap());
    let error = json!({"error":{"code":"INVALID_REQUEST","message":"Invalid request","request_id":"00000000-0000-4000-8000-000000000108","retryable":false}});
    roundtrip::<WireError>(&serde_json::to_vec(&error).unwrap());
}
#[test]
fn untrusted_json_is_bounded_and_duplicate_keys_cannot_be_hidden() {
    for bytes in [
        br#"{"a":1,"a":2}"#.as_slice(),
        br#"{"a":{"x":1,"\u0078":2}}"#,
        br#"[]{}"#,
        br#"{"x":1e999}"#,
    ] {
        assert!(parse_json(bytes).is_err());
    }
    assert!(parse_json(&vec![b' '; MAX_JSON_BYTES + 1]).is_err());
    assert!(parse_json(format!("[{}]", "0,".repeat(16_384) + "0").as_bytes()).is_err());
    assert!(parse_json(format!("{}0{}", "[".repeat(33), "]".repeat(33)).as_bytes()).is_err());
    for n in ["18446744073709551616", "01", "+1", "-1", ""] {
        assert!(Decimal::try_from(n.to_owned()).is_err());
    }
    assert_eq!(Decimal::try_from(u64::MAX.to_string()).unwrap().0, u64::MAX);
    assert!(BoundedU32::<1, 8>::new(0).is_err());
    assert!(BoundedU32::<1, 8>::new(9).is_err());
    assert!(
        "00000000-0000-1000-8000-000000000001"
            .parse::<NodeId>()
            .is_err()
    );
}
#[test]
fn schema_failures_and_cross_field_invariants_are_rejected() {
    invalid::<NodeCapabilities>(NODE, |v| v["protocol_version"] = json!(2));
    invalid::<NodeCapabilities>(NODE, |v| v["unknown"] = json!(true));
    invalid::<NodeCapabilities>(NODE, |v| v["observed_at"] = json!("not-a-date"));
    invalid::<NodeCapabilities>(NODE, |v| {
        v["memory_pools"][0]["available_bytes"] = json!(u64::MAX.to_string())
    });
    invalid::<NodeCapabilities>(NODE, |v| {
        v["backends"][0]["memory_pool_ids"] = json!(["absent"])
    });
    invalid::<NodeCapabilities>(NODE, |v| v["backends"][0]["boundary_profiles"] = json!([]));
    invalid::<PlacementPlan>(PLAN, |v| v["stages"][1]["layer_start"] = json!(15));
    invalid::<PlacementPlan>(PLAN, |v| {
        v["stages"][0]["resident_bytes"] = json!(u64::MAX.to_string())
    });
    invalid::<PlacementPlan>(PLAN, |v| v["stages"][1]["owns_embedding"] = json!(true));
    invalid::<PlacementPlan>(PLAN, |v| v["mode"] = json!("whole_request"));
    invalid::<PlacementPlan>(PLAN, |v| v["expires_at"] = v["created_at"].clone());
    invalid::<Lease>(LEASE, |v| {
        v["reserved_bytes"] = json!("18446744073709551616")
    });
    invalid::<Lease>(LEASE, |v| v["lifetime_ms"] = json!(999));
    invalid::<TensorHeader>(TENSOR, |v| {
        v["shape"] = json!([1048576, 1048576, 1048576, 1048576])
    });
    invalid::<TensorHeader>(TENSOR, |v| {
        v["chunk_offset_bytes"] = json!(u64::MAX.to_string())
    });
    invalid::<TensorHeader>(TENSOR, |v| v["end_of_tensor"] = json!(false));
    invalid::<TensorHeader>(TENSOR, |v| v["tensor_index"] = json!(1));
    invalid::<TensorHeader>(TENSOR, |v| v["dtype"] = json!("u32"));
    invalid::<TensorHeader>(TENSOR, |v| v["token_count"] = json!(1));
    invalid::<ChatRequest>(CHAT, |v| v["messages"][0]["image_url"] = json!("anything"));
    for path in [
        "../escape",
        "a/../b",
        "a//b",
        "CON.txt",
        "dir/NUL",
        "asset.",
    ] {
        invalid::<ModelManifest>(MANIFEST, |v| v["assets"][0]["path"] = json!(path));
    }
    invalid::<ModelManifest>(MANIFEST, |v| {
        v["layers"][0]["asset_paths"] = json!(["absent"])
    });
    invalid::<ModelManifest>(MANIFEST, |v| v["layers"][0]["index"] = json!(1));
    invalid::<ModelManifest>(MANIFEST, |v| v["execution_precision"] = json!("fp4"));
    invalid::<ModelManifest>(MANIFEST, |v| {
        v["assets"][1]["path"] = v["assets"][0]["path"].clone()
    });
}
#[test]
fn canonicalization_matches_independent_vectors_and_detects_tampering() {
    let golden: Value =
        serde_json::from_str(include_str!("../../../examples/digest-vectors.json")).unwrap();
    for (name, input, field) in [
        ("model-manifest", MANIFEST, "manifest_digest"),
        ("placement-plan", PLAN, "plan_digest"),
    ] {
        let mut value = parse_json(input).unwrap();
        let hash = canonical_digest(&value, field).unwrap();
        assert_eq!(hash.as_str(), golden[name]["sha256"].as_str().unwrap());
        let mut body = value.clone();
        body.as_object_mut().unwrap().remove(field);
        assert_eq!(
            serde_json_canonicalizer::to_string(&body).unwrap(),
            golden[name]["canonical"].as_str().unwrap()
        );
        value[field] = json!(hash.as_str());
        if name == "placement-plan" {
            assert!(decode::<PlacementPlan>(&serde_json::to_vec(&value).unwrap(), MODE).is_ok());
            value["context_limit"] = json!(1024);
            assert!(decode::<PlacementPlan>(&serde_json::to_vec(&value).unwrap(), MODE).is_err());
        }
    }
    // RFC 8785 sorts keys as UTF-16 code units, not UTF-8 or Unicode scalar order.
    assert_eq!(
        serde_json_canonicalizer::to_string(&json!({"\u{e000}":1,"\u{10000}":2})).unwrap(),
        "{\"\u{10000}\":2,\"\u{e000}\":1}"
    );
    assert_eq!(
        serde_json_canonicalizer::to_string(&json!({"z":-0.0,"a":1e30,"b":0.002})).unwrap(),
        r#"{"a":1e+30,"b":0.002,"z":0}"#
    );
    assert!(decode::<ModelManifest>(MANIFEST, FixtureMode::Reject).is_err());
    assert!(decode::<PlacementPlan>(PLAN, FixtureMode::Reject).is_err());
    assert!(decode::<NodeCapabilities>(NODE, FixtureMode::Reject).is_err());
}

#[test]
fn control_requests_bind_parameters_and_use_strict_versioned_schemas() {
    let lease: Lease = decode(LEASE, MODE).unwrap();
    let request = ControlRequest::new(
        "00000000-0000-4000-8000-000000000109".parse().unwrap(),
        1,
        1,
        Operation::Commit {
            lease_id: lease.lease_id,
        },
    )
    .unwrap();
    let bytes = serde_json::to_vec(&request).unwrap();
    assert_eq!(request, ControlRequest::decode(&bytes, MODE).unwrap());
    let mut changed = parse_json(&bytes).unwrap();
    changed["operation"]["op"] = json!("release");
    assert!(ControlRequest::decode(&serde_json::to_vec(&changed).unwrap(), MODE).is_err());
    changed = parse_json(&bytes).unwrap();
    changed["operation"]["unknown"] = json!(1);
    assert!(ControlRequest::decode(&serde_json::to_vec(&changed).unwrap(), MODE).is_err());
    changed = parse_json(&bytes).unwrap();
    changed["authority_epoch"] = json!("18446744073709551616");
    assert!(ControlRequest::decode(&serde_json::to_vec(&changed).unwrap(), MODE).is_err());
    changed = parse_json(&bytes).unwrap();
    changed["protocol_version"] = json!(2);
    assert!(ControlRequest::decode(&serde_json::to_vec(&changed).unwrap(), MODE).is_err());
    assert!(decode::<Lease>(LEASE, FixtureMode::Reject).is_err());

    let plan: PlacementPlan = decode(PLAN, MODE).unwrap();
    let session: Session = decode(include_bytes!("../../../examples/session.json"), MODE).unwrap();
    let header: TensorHeader = decode(TENSOR, MODE).unwrap();
    for operation in [
        Operation::Prepare {
            plan: Box::new(plan),
            lease: Box::new(lease.clone()),
        },
        Operation::Release {
            lease_id: lease.lease_id,
        },
        Operation::Begin {
            session: Box::new(session.clone()),
        },
        Operation::StartStep {
            header: Box::new(header.clone()),
            input_digest: byte_digest(b"fixture"),
        },
        Operation::CompleteStep {
            generation_id: session.generation_id,
            invocation_id: header.invocation_id,
        },
        Operation::Cancel {
            generation_id: session.generation_id,
        },
        Operation::Finish {
            generation_id: session.generation_id,
        },
    ] {
        let r = ControlRequest::new(request.request_id, 1, 1, operation).unwrap();
        assert_eq!(
            r,
            ControlRequest::decode(&serde_json::to_vec(&r).unwrap(), MODE).unwrap()
        );
    }
}

#[test]
fn request_digest_binds_large_sequence_numbers_exactly() {
    let lease: Lease = decode(LEASE, MODE).unwrap();
    let id = "00000000-0000-4000-8000-000000000109".parse().unwrap();
    let first = ControlRequest::new(
        id,
        1,
        (1_u64 << 53) + 1,
        Operation::Release {
            lease_id: lease.lease_id,
        },
    )
    .unwrap();
    let second = ControlRequest::new(id, 1, (1_u64 << 53) + 2, first.operation.clone()).unwrap();
    assert_ne!(first.request_hash, second.request_hash);
    let mut tampered = first.clone();
    tampered.request_sequence = second.request_sequence;
    assert!(tampered.validate(MODE).is_err());
    assert!(first.validate(MODE).is_ok());
}
