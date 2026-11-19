use serde_json::json;
use starlings_core::{CoreError, profiles::*};
use starlings_protocol::*;
const MODE: FixtureMode = FixtureMode::Simulation;
mod support;
use support::{age, fixture};
fn inspect(
    m: &ModelManifest,
    p: &ModelCostProfile,
    range: std::ops::Range<usize>,
    age: ProfileAge,
    approved: Option<&Digest>,
) -> Result<ProfileAssessment, CoreError> {
    assess(p, m, &p.key, range, age, approved, MODE)
}
#[test]
fn hybrid_stage_budgets_cover_each_pool_state_phase_and_owned_component() {
    let (m, p) = fixture();
    let both = inspect(&m, &p, 0..2, age(0), None).unwrap();
    assert_eq!(both.status, ProfileStatus::FixtureOnly);
    assert_eq!(
        inspect(&m, &p, 0..2, age(0), Some(&content_hash(&p).unwrap()))
            .unwrap()
            .status,
        ProfileStatus::FixtureOnly
    );
    assert_eq!(both.pools[0].pool_id, "host");
    assert_eq!(both.pools[1].pool_id, "device");
    // Five cost rows: stage overhead + two layers + embedding + head.
    assert_eq!(
        both.pools[0].resident_weight_bytes.0,
        30 + 10 + 10 + 20 + 20
    );
    assert_eq!(both.pools[0].total_bytes.0, 90 + 5 + 5 * (3 + 5 + 7 + 11));
    assert_eq!(
        both.pools[1].resident_weight_bytes.0,
        300 + 100 + 100 + 200 + 200
    );
    assert_eq!(both.pools[1].state_bytes.0, (2 * 10 + 3 + 5) * 2);
    assert_eq!(
        both.pools[1].total_bytes.0,
        900 + 5 + 56 + 5 * (3 + 5 + 7 + 11)
    );
    assert_eq!(both.timing.as_ref().unwrap().prefill_us.0, 5 * 13);
    assert_eq!(both.timing.unwrap().decode_us_per_token.0, 5 * 17);
    let first = inspect(&m, &p, 0..1, age(0), None).unwrap();
    let last = inspect(&m, &p, 1..2, age(0), None).unwrap();
    assert_eq!(first.pools[1].resident_weight_bytes.0, 300 + 100 + 200);
    assert_eq!(last.pools[1].resident_weight_bytes.0, 300 + 100 + 200);
    assert_eq!(first.pools[1].state_bytes.0, (3 + 5) * 2);
    assert_eq!(last.pools[1].state_bytes.0, 2 * 10 * 2);
    // Each stage has its own overhead; no host/device summation into one fictional pool.
    assert_eq!(first.pools.len(), 2);
}
#[test]
fn strict_contract_refuses_missing_state_coverage_unknown_fields_and_overflow() {
    let (m, p) = fixture();
    let base = serde_json::to_value(&p).unwrap();
    for variant in [
        "unknown",
        "version",
        "decimal",
        "duplicate_pool",
        "order",
        "active",
        "missing_layer",
        "missing_component",
        "profile",
        "disposition",
        "state",
        "workload",
        "confidence",
        "timestamp",
        "kv",
        "recurrent",
        "excluded",
    ] {
        let mut v = base.clone();
        match variant {
            "unknown" => v["unexpected"] = json!(true),
            "version" => v["version"] = json!(2),
            "decimal" => v["costs"]["layers"][0]["pools"][0]["resident_weight_bytes"] = json!("01"),
            "duplicate_pool" => v["costs"]["pool_ids"][1] = json!("host"),
            "order" => v["costs"]["layers"][0]["pools"]
                .as_array_mut()
                .unwrap()
                .reverse(),
            "active" => {
                v["costs"]["layers"][0]["pools"][0]["active_weight_bytes_per_token"] =
                    json!("18446744073709551616")
            }
            "missing_layer" => {
                v["costs"]["layers"].as_array_mut().unwrap().pop();
            }
            "missing_component" => {
                v["costs"]["components"].as_array_mut().unwrap().pop();
            }
            "profile" => v["key"]["execution_profile"] = json!("different-profile"),
            "disposition" => v["costs"]["components"][0]["owner_role"] = json!("none"),
            "state" => v["costs"]["layers"][0]["state_profile"] = json!("different-state"),
            "workload" => v["key"]["workload"]["prompt_tokens"] = json!(10),
            "confidence" => v["costs"]["measurement"]["basis"] = json!("measured"),
            "timestamp" => v["costs"]["measurement"]["observed_at"] = json!("not-a-date"),
            "kv" => v["costs"]["layers"][1]["pools"][1]["kv_bytes_per_token"] = json!("0"),
            "recurrent" => {
                v["costs"]["layers"][0]["pools"][1]["recurrent_bytes_per_session"] = json!("0");
                v["costs"]["layers"][0]["pools"][1]["convolution_bytes_per_session"] = json!("0");
            }
            "excluded" => v["costs"]["components"][2]["pools"][0]["reserve_bytes"] = json!("1"),
            _ => unreachable!(),
        }
        let result = decode::<ModelCostProfile>(&serde_json::to_vec(&v).unwrap(), MODE);
        assert!(
            result.is_err() || inspect(&m, &result.unwrap(), 0..2, age(0), None).is_err(),
            "{variant}"
        );
    }
    let mut v = base;
    v["costs"]["layers"][0]["pools"][1]["kv_bytes_per_token"] = json!(u64::MAX.to_string());
    let overflow: ModelCostProfile = decode(&serde_json::to_vec(&v).unwrap(), MODE).unwrap();
    assert_eq!(
        inspect(&m, &overflow, 0..2, age(0), None).err(),
        Some(CoreError::Limit)
    );
    assert!(decode::<ModelCostProfile>(b"{\"version\":1,\"version\":1}", MODE).is_err());
    assert!(decode::<ModelCostProfile>(&encode(&p, MODE).unwrap(), FixtureMode::Reject).is_err());
    assert_eq!(
        inspect(&m, &p, 0..0, age(0), None).err(),
        Some(CoreError::Invalid)
    );
    assert_eq!(
        inspect(&m, &p, 0..3, age(0), None).err(),
        Some(CoreError::Invalid)
    );
}
#[test]
fn freshness_confidence_approval_and_unavailable_are_visible_without_admission() {
    let (mut m, mut p) = fixture();
    assert_eq!(
        inspect(&m, &p, 0..2, age(990), None).unwrap().status,
        ProfileStatus::Stale
    );
    assert_eq!(
        inspect(
            &m,
            &p,
            0..2,
            ProfileAge {
                now_ms: 99,
                ..age(0)
            },
            None
        )
        .err(),
        Some(CoreError::Clock)
    );
    assert_eq!(
        inspect(
            &m,
            &p,
            0..2,
            ProfileAge {
                captured_at_ms: 0,
                now_ms: u64::MAX,
                ..age(0)
            },
            None
        )
        .err(),
        Some(CoreError::Clock)
    );
    let mut recaptured = age(990);
    recaptured.captured_at_ms = recaptured.now_ms;
    recaptured.captured_wall_ms = recaptured.now_wall_ms;
    assert_eq!(
        inspect(&m, &p, 0..2, recaptured, None).unwrap().status,
        ProfileStatus::Stale
    );
    let mut suspended = age(0);
    suspended.now_wall_ms += 2000;
    assert_eq!(
        inspect(&m, &p, 0..2, suspended, None).unwrap().status,
        ProfileStatus::Stale
    );
    let mut rollback = age(0);
    rollback.now_wall_ms -= 1;
    assert_eq!(
        inspect(&m, &p, 0..2, rollback, None).err(),
        Some(CoreError::Clock)
    );
    m.fixture = false;
    // Replace synthetic zero hashes before exercising the default production identity boundary.
    for a in &mut m.assets {
        a.sha256 = byte_digest(a.path.as_bytes());
    }
    m.config_sha256 = m.assets[0].sha256.clone();
    m.tokenizer_sha256 = m.assets[1].sha256.clone();
    m.template_sha256 = m.assets[2].sha256.clone();
    m.manifest_digest =
        canonical_digest(&serde_json::to_value(&m).unwrap(), "manifest_digest").unwrap();
    p.fixture = false;
    p.key.manifest_digest = m.manifest_digest.clone();
    assert_eq!(
        inspect(&m, &p, 0..2, age(0), None).unwrap().status,
        ProfileStatus::Estimated
    );
    assert_eq!(
        inspect(&m, &p, 0..2, age(0), Some(&content_hash(&p).unwrap()))
            .unwrap()
            .status,
        ProfileStatus::Estimated
    );
    if let ModelCosts::Available { measurement, .. } = &mut p.costs {
        measurement.basis = CostBasis::Measured;
        measurement.samples = BoundedU32::new(5).unwrap();
        measurement.relative_error_bps = Some(BoundedU32::new(500).unwrap());
    }
    assert_eq!(
        inspect(&m, &p, 0..2, age(0), None).unwrap().status,
        ProfileStatus::Unqualified
    );
    let approved = content_hash(&p).unwrap();
    let qualified = assess(
        &p,
        &m,
        &p.key,
        0..2,
        age(0),
        Some(&approved),
        FixtureMode::Reject,
    )
    .unwrap();
    assert_eq!(qualified.status, ProfileStatus::Qualified);
    assert_eq!(
        qualified
            .measurement
            .unwrap()
            .relative_error_bps
            .unwrap()
            .get(),
        500
    );
    assert_eq!(
        inspect(&m, &p, 0..2, age(990), Some(&approved))
            .unwrap()
            .status,
        ProfileStatus::Stale
    );
    let mut changed_config = p.key.clone();
    changed_config.runtime_config_digest = byte_digest(b"different offload/thread configuration");
    assert_eq!(
        assess(
            &p,
            &m,
            &changed_config,
            0..2,
            age(0),
            Some(&approved),
            FixtureMode::Reject
        )
        .err(),
        Some(CoreError::Invalid)
    );
    let mut other = p.key.clone();
    other.workload.sessions = BoundedU32::new(3).unwrap();
    assert_eq!(
        assess(
            &p,
            &m,
            &other,
            0..2,
            age(0),
            Some(&approved),
            FixtureMode::Reject
        )
        .err(),
        Some(CoreError::Invalid)
    );
    if let ModelCosts::Available {
        measurement,
        layers,
        ..
    } = &mut p.costs
    {
        measurement.samples = BoundedU32::new(6).unwrap();
        layers[0].timing = None;
    }
    let changed = inspect(&m, &p, 0..2, age(0), Some(&approved)).unwrap();
    assert_eq!(changed.status, ProfileStatus::Unqualified);
    assert!(changed.timing.is_none());
    p.costs = ModelCosts::Unavailable {
        code: "LAYER_BOUNDARY_UNSUPPORTED".into(),
    };
    let absent = inspect(&m, &p, 0..2, age(0), None).unwrap();
    assert_eq!(absent.status, ProfileStatus::Unavailable);
    assert!(absent.pools.is_empty());
    assert_eq!(
        absent.unavailable_code.unwrap(),
        "LAYER_BOUNDARY_UNSUPPORTED"
    );
}
