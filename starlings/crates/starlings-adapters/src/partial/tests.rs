use super::*;
use starlings_protocol::{TensorHeader, byte_digest};
const MODE: FixtureMode = FixtureMode::Simulation;
fn contract() -> BoundaryContract {
    BoundaryContract::decode(
        include_bytes!("../../../../examples/partial-boundary.json"),
        MODE,
    )
    .unwrap()
}
fn call(boundary: &BoundaryContract) -> RunChunk {
    let h: TensorHeader = starlings_protocol::decode(
        include_bytes!("../../../../examples/tensor-header.json"),
        MODE,
    )
    .unwrap();
    RunChunk {
        version: Version,
        step: StepIdentity {
            scope: (&h).into(),
            phase: h.phase,
            position_start: h.position_start,
            token_count: h.token_count,
        },
        inputs: boundary.layouts(2, 128).unwrap().0,
    }
}
#[test]
fn boundary_metadata_is_strict_fixture_isolated_and_bounded() {
    let b = contract();
    let bytes = serde_json::to_vec(&b).unwrap();
    assert_eq!(BoundaryContract::decode(&bytes, MODE).unwrap(), b);
    assert_eq!(
        BoundaryContract::decode(&bytes, FixtureMode::Reject),
        Err(Error::Invalid)
    );
    for change in [
        "version",
        "unknown",
        "pattern-unknown",
        "empty",
        "too-many",
        "axis",
        "role",
        "rank",
        "overflow",
        "u64-overflow",
        "workload",
    ] {
        let mut value = serde_json::to_value(&b).unwrap();
        match change {
            "version" => value["version"] = 2.into(),
            "unknown" => value["ignored"] = true.into(),
            "pattern-unknown" => value["inputs"][0]["ignored"] = true.into(),
            "empty" => value["inputs"] = serde_json::json!([]),
            "too-many" => value["inputs"] = serde_json::json!(vec![value["inputs"][0].clone(); 9]),
            "axis" => value["inputs"][0]["shape"][0] = serde_json::json!({"kind":"tokens"}),
            "role" => value["inputs"][0]["dtype"] = "u32".into(),
            "rank" => value["inputs"][0]["shape"] = serde_json::json!([{"kind":"tokens"}]),
            "overflow" => {
                value["inputs"][0]["shape"] = serde_json::json!([{"kind":"fixed","value":1048576},{"kind":"fixed","value":1048576},{"kind":"tokens"},{"kind":"fixed","value":1048576}])
            }
            "u64-overflow" => {
                value["inputs"][0]["shape"] = serde_json::json!([{"kind":"fixed","value":1048576},{"kind":"fixed","value":1048576},{"kind":"tokens"},{"kind":"fixed","value":1048576}]);
                value["key"]["workload"]["context_tokens"] = 1048576.into();
                value["key"]["workload"]["prompt_tokens"] = 524288.into();
                value["key"]["workload"]["prefill_chunk_tokens"] = 524288.into();
                value["key"]["workload"]["output_tokens"] = 1.into();
            }
            "workload" => value["key"]["workload"]["context_tokens"] = 1.into(),
            _ => unreachable!(),
        }
        assert!(
            BoundaryContract::decode(&serde_json::to_vec(&value).unwrap(), MODE).is_err(),
            "{change}"
        );
    }
    let duplicate = String::from_utf8(bytes).unwrap().replacen(
        "\"version\":1",
        "\"version\":1,\"version\":1",
        1,
    );
    assert!(BoundaryContract::decode(duplicate.as_bytes(), MODE).is_err());
    assert!(
        BoundaryContract::decode(&vec![b' '; starlings_protocol::MAX_JSON_BYTES + 1], MODE)
            .is_err()
    );
}
#[test]
fn ordered_layouts_cover_all_dtypes_sampled_tokens_and_reassembled_tensors() {
    let mut b = contract();
    for (dtype, width) in [
        (TensorHeaderDtype::Bf16, 2),
        (TensorHeaderDtype::F16, 2),
        (TensorHeaderDtype::F32, 4),
    ] {
        for p in b.inputs.iter_mut().chain(&mut b.outputs) {
            p.dtype = dtype;
        }
        let (input, output) = b.layouts(2, 1024).unwrap();
        assert_eq!(input[0].bytes.0, 4 * width);
        assert_eq!(input, output);
        assert_eq!(b.layouts(2, 16 * width - 1), Err(Error::Capacity));
    }
    b.outputs = vec![TensorPattern {
        role: TensorHeaderTensorRole::TokenIds,
        dtype: TensorHeaderDtype::U32,
        shape: vec![Dimension::Tokens],
    }];
    b.output_tokens = OutputTokens::SampledOne;
    b.validate(MODE).unwrap();
    let (_, output) = b.layouts(2, 1024).unwrap();
    assert_eq!(output[0].shape[0].get(), 1);
    assert_eq!(output[0].bytes.0, 4);
    b.inputs[0].shape[2] = Dimension::Fixed(BoundedU32::new(1048576).unwrap());
    b.inputs[0]
        .shape
        .insert(0, Dimension::Fixed(BoundedU32::new(2).unwrap()));
    let (input, _) = b.layouts(2, 32 * 1024 * 1024).unwrap();
    assert_eq!(input[0].bytes.0, 16 * 1024 * 1024);
    assert!(input[0].bytes.0 > starlings_protocol::DATA_PAYLOAD_LIMIT as u64);
}
#[test]
fn run_chunk_requires_the_trusted_step_and_complete_ordered_layouts() {
    let b = contract();
    let c = call(&b);
    RunChunk::decode(&serde_json::to_vec(&c).unwrap(), MODE)
        .unwrap()
        .validate(&b, &c.step, 128, MODE)
        .unwrap();
    assert!(c.validate(&b, &c.step, 128, FixtureMode::Reject).is_err());
    for change in [
        "scope", "sequence", "position", "phase", "missing", "order", "size", "shape",
    ] {
        let mut bad = c.clone();
        match change {
            "scope" => bad.step.scope.plan_digest = byte_digest(b"different plan"),
            "sequence" => bad.step.scope.sequence = Decimal(1),
            "position" => bad.step.position_start = BoundedU32::new(1).unwrap(),
            "phase" => bad.step.phase = TensorHeaderPhase::Decode,
            "missing" => {
                bad.inputs.pop();
            }
            "order" => bad.inputs.swap(0, 1),
            "size" => bad.inputs[0].bytes = Decimal(9),
            "shape" => bad.inputs[0].shape[2] = BoundedU32::new(3).unwrap(),
            _ => unreachable!(),
        }
        assert!(bad.validate(&b, &c.step, 128, MODE).is_err(), "{change}");
    }
    for field in ["missing", "shape", "tokens", "dtype", "bytes", "epoch"] {
        let mut bad = c.clone();
        match field {
            "missing" => bad.inputs.clear(),
            "shape" => bad.inputs[0].shape.clear(),
            "tokens" => bad.inputs[0].shape[1] = BoundedU32::new(1).unwrap(),
            "dtype" => bad.inputs[0].dtype = TensorHeaderDtype::U32,
            "bytes" => bad.inputs[0].bytes = Decimal(1),
            "epoch" => bad.step.scope.authority_epoch = Decimal(0),
            _ => unreachable!(),
        }
        assert!(
            RunChunk::decode(&serde_json::to_vec(&bad).unwrap(), MODE).is_err(),
            "{field}"
        );
    }
    let mut native = b.clone();
    native.fixture = false;
    assert!(
        c.validate(&native, &c.step, 128, FixtureMode::Reject)
            .is_err(),
        "fixture scope admitted by non-fixture descriptor"
    );
    let mut native_call = c.clone();
    native_call.step.scope.plan_digest = byte_digest(b"approved-plan-test");
    native_call
        .validate(&native, &native_call.step, 128, FixtureMode::Reject)
        .unwrap();
    let duplicate = serde_json::to_string(&c).unwrap().replacen(
        "\"version\":1",
        "\"version\":1,\"version\":1",
        1,
    );
    assert!(RunChunk::decode(duplicate.as_bytes(), MODE).is_err());
}
#[cfg(feature = "test-backend")]
#[test]
fn synthetic_lifecycle_refuses_native_use_duplicates_and_untrusted_drain() {
    let b = contract();
    let c = call(&b);
    let plan = c.step.scope.plan_digest.clone();
    assert!(SyntheticStage::prepare(b.clone(), plan.clone(), 128, FixtureMode::Reject).is_err());
    let mut not_fixture = b.clone();
    not_fixture.fixture = false;
    not_fixture.validate(FixtureMode::Reject).unwrap();
    assert!(SyntheticStage::prepare(not_fixture, plan.clone(), 128, MODE).is_err());
    let mut fake = SyntheticStage::prepare(b, plan, 128, MODE).unwrap();
    assert!(fake.describe().fixture);
    assert!(matches!(
        fake.inspect_model().costs,
        ModelCosts::Unavailable { .. }
    ));
    assert!(fake.begin(c.step.scope.clone()).is_err());
    fake.commit().unwrap();
    fake.begin(c.step.scope.clone()).unwrap();
    let bytes = vec![vec![0, 0, 0, 128, 128, 127, 193, 127]; 2];
    fake.run(&c, bytes.clone()).unwrap();
    assert_eq!(fake.executions(), 1);
    assert_eq!(fake.retained_bytes(), 16);
    assert!(fake.run(&c, bytes).is_err());
    assert!(!fake.cancel().unwrap());
    assert!(fake.end_session().is_err());
    assert!(fake.unload().is_err());
    assert_eq!(fake.retained_bytes(), 16);
    assert!(fake.finish_synthetic().unwrap().is_none());
    assert_eq!(fake.retained_bytes(), 0);
    fake.end_session().unwrap();
    assert!(fake.begin(c.step.scope).is_err());
    fake.unload().unwrap();
    assert!(fake.commit().is_err());
}
#[cfg(feature = "test-backend")]
#[test]
fn synthetic_completion_advances_once_and_preserves_raw_hidden_residual_bytes() {
    let b = contract();
    let mut c = call(&b);
    let mut fake = SyntheticStage::prepare(b, c.step.scope.plan_digest.clone(), 128, MODE).unwrap();
    fake.commit().unwrap();
    fake.begin(c.step.scope.clone()).unwrap();
    let payload = vec![vec![0, 0, 0, 128, 128, 127, 193, 127]; 2];
    fake.run(&c, payload.clone()).unwrap();
    assert_eq!(fake.finish_synthetic().unwrap().unwrap(), payload);
    assert!(fake.finish_synthetic().is_err());
    assert!(fake.run(&c, payload.clone()).is_err());
    c.step.scope.invocation_id = uuid::Uuid::new_v4().to_string().parse().unwrap();
    assert!(fake.run(&c, payload.clone()).is_err());
    c.step.scope.sequence = Decimal(1);
    c.step.position_start = BoundedU32::new(2).unwrap();
    fake.run(&c, payload.clone()).unwrap();
    assert_eq!(fake.finish_synthetic().unwrap().unwrap(), payload);
    assert_eq!(fake.executions(), 2);
}

#[cfg(feature = "test-backend")]
#[test]
fn synthetic_payload_capacity_and_context_limits_fail_before_execution() {
    let b = contract();
    let mut c = call(&b);
    let mut fake =
        SyntheticStage::prepare(b.clone(), c.step.scope.plan_digest.clone(), 128, MODE).unwrap();
    fake.commit().unwrap();
    fake.begin(c.step.scope.clone()).unwrap();
    let mut oversized = Vec::with_capacity(256);
    oversized.extend_from_slice(&[0; 8]);
    assert_eq!(
        fake.run(&c, vec![oversized, vec![0; 8]]),
        Err(Error::Capacity)
    );
    assert_eq!(fake.run(&c, vec![vec![0; 8]]), Err(Error::Capacity));
    assert_eq!(fake.executions(), 0);
    assert_eq!(fake.retained_bytes(), 0);
    for sequence in 0..6 {
        let tokens = if sequence < 2 { 2 } else { 1 };
        c.step.scope.invocation_id = uuid::Uuid::new_v4().to_string().parse().unwrap();
        c.step.scope.sequence = Decimal(sequence);
        c.step.position_start = BoundedU32::new(if sequence < 2 {
            sequence as u32 * 2
        } else {
            sequence as u32 + 2
        })
        .unwrap();
        c.step.phase = if sequence < 2 {
            TensorHeaderPhase::Prefill
        } else {
            TensorHeaderPhase::Decode
        };
        c.step.token_count = BoundedU32::new(tokens).unwrap();
        c.inputs = b.layouts(tokens, 128).unwrap().0;
        let bytes: Vec<Vec<u8>> = c
            .inputs
            .iter()
            .map(|l| vec![0; l.bytes.0 as usize])
            .collect();
        fake.run(&c, bytes.clone()).unwrap();
        assert_eq!(fake.finish_synthetic().unwrap(), Some(bytes));
    }
    c.step.scope.invocation_id = uuid::Uuid::new_v4().to_string().parse().unwrap();
    c.step.scope.sequence = Decimal(6);
    c.step.position_start = BoundedU32::new(8).unwrap();
    assert_eq!(fake.run(&c, vec![vec![0; 4]; 2]), Err(Error::Capacity));
    assert_eq!(fake.executions(), 6);
    assert_eq!(fake.retained_bytes(), 0);
}
