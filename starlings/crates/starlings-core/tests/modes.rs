use serde_json::json;
use starlings_core::{CoreError, modes::*, planner::*};
use starlings_protocol::*;
mod support;
use support::{age, fixture};
const MODE: FixtureMode = FixtureMode::Simulation;

fn observation() -> Observation {
    Observation {
        basis: CostBasis::Estimated,
        evidence_digest: byte_digest(b"synthetic cost evidence"),
        captured_at_ms: Decimal(100),
        age_at_capture_ms: Decimal(0),
        valid_for_ms: BoundedU32::new(1000).unwrap(),
    }
}
fn variant(count: usize, fits: bool) -> Variant {
    let (manifest, profile) = fixture();
    let nodes: Vec<_> = (1..=count).map(|i| {
        let mut cap: serde_json::Value = serde_json::from_slice(include_bytes!("../../../examples/node-capabilities.json")).unwrap();
        cap["node_id"] = json!(format!("00000000-0000-4000-8000-{i:012}"));
        cap["snapshot_id"] = json!(format!("00000000-0000-4000-8000-{:012}",i+100));
        cap["memory_pools"] = json!([
            {"pool_id":"host","kind":"system","total_bytes":"2000","available_bytes":"1000","reserved_bytes":"0","device_ids":["mock:0"]},
            {"pool_id":"device","kind":"device","total_bytes":"2000","available_bytes":if fits {"1500"}else{"900"},"reserved_bytes":"0","device_ids":["mock:0"]}
        ]);
        cap["backends"][0]["memory_pool_ids"] = json!(["host","device"]);
        cap["backends"][0]["max_sessions"] = json!(2);
        cap["backends"][0]["build_digest"] = json!(profile.key.backend_build_digest);
        NodeInput { capabilities: decode(&serde_json::to_vec(&cap).unwrap(),MODE).unwrap(),
            captured_at_ms: Decimal(100), family: Family::Nvidia, backend_id: "fake-v1".into(),
            expected_profile_key: profile.key.clone(), profile: profile.clone(), profile_clocks: age(0),
            approved_profile_digest: None, load_us: Decimal(0),
            copy: PhaseCost { prefill_us: Decimal(0), decode_us_per_token: Decimal(0) } }
    }).collect();
    let mut links = Vec::new();
    for source in &nodes {
        for destination in &nodes {
            if source.capabilities.node_id != destination.capabilities.node_id {
                links.push(LinkCost {
                    source: source.capabilities.node_id,
                    destination: destination.capabilities.node_id,
                    fixture: true,
                    boundary_profile: manifest.required_boundary_profile.clone(),
                    workload_digest: content_hash(&profile.key.workload).unwrap(),
                    evidence_digest: byte_digest(b"synthetic link"),
                    transfer: PhaseCost {
                        prefill_us: Decimal(1),
                        decode_us_per_token: Decimal(1),
                    },
                    age_at_capture_ms: Decimal(0),
                    captured_at_ms: Decimal(100),
                });
            }
        }
    }
    let placement = PreviewInput {
        version: Version,
        fixture: true,
        manifest,
        now_ms: Decimal(100),
        now_wall_ms: age(0).now_wall_ms,
        nodes,
        links,
        objective: Objective::Latency,
        max_stages: BoundedU32::new(3).unwrap(),
        top_k: BoundedU32::new(1).unwrap(),
        max_states: BoundedU32::new(1000).unwrap(),
        max_cost_rows: BoundedU32::new(10000).unwrap(),
    };
    let mut v = Variant {
        configuration: ConfigurationMode::Resident,
        placement,
        costs: ModeCosts {
            tiers: vec![],
            storage: vec![],
            links: vec![],
        },
        approved_costs_digest: None,
    };
    refresh(&mut v);
    v
}
fn refresh(v: &mut Variant) {
    v.costs.storage = v
        .placement
        .nodes
        .iter()
        .map(|n| StorageRead {
            node: n.capabilities.node_id,
            profile_digest: content_hash(&n.profile).unwrap(),
            traffic: StorageTraffic::NotUsed,
        })
        .collect();
    v.costs.tiers = v
        .placement
        .nodes
        .iter()
        .flat_map(|n| {
            n.capabilities.memory_pools.iter().map(|p| TierBandwidth {
                node: n.capabilities.node_id,
                profile_digest: content_hash(&n.profile).unwrap(),
                pool_id: p.pool_id.clone(),
                bytes_per_second: Decimal(1_000_000),
                observation: observation(),
            })
        })
        .collect();
    v.costs.links = v
        .placement
        .links
        .iter()
        .map(|l| LinkLatency {
            source: l.source,
            destination: l.destination,
            link_evidence_digest: l.evidence_digest.clone(),
            p50_us: Some(Decimal(10)),
            p99_us: Some(Decimal(30)),
            idle_wake_us: Some(Decimal(50)),
            observation: observation(),
        })
        .collect();
}
fn input(variants: Vec<Variant>) -> ModeInput {
    ModeInput {
        version: Version,
        fixture: true,
        family: Family::Nvidia,
        variants,
        link_limits: LinkLimits {
            p50_us: Decimal(20),
            p99_us: Decimal(40),
            idle_wake_us: Decimal(60),
        },
    }
}
fn offload(v: &Variant) -> Variant {
    let mut v = v.clone();
    v.configuration = ConfigurationMode::ExpertOffload;
    for node in &mut v.placement.nodes {
        node.profile.key.runtime_config_digest =
            byte_digest(b"synthetic expert offload configuration");
        node.expected_profile_key = node.profile.key.clone();
        if let ModelCosts::Available {
            layers,
            components,
            per_stage,
            ..
        } = &mut node.profile.costs
        {
            for pools in layers
                .iter_mut()
                .map(|l| &mut l.pools)
                .chain(components.iter_mut().map(|c| &mut c.pools))
                .chain(std::iter::once(per_stage))
            {
                let moved = pools[1].resident_weight_bytes.0 / 2;
                pools[1].resident_weight_bytes.0 -= moved;
                pools[0].resident_weight_bytes.0 += moved;
                for pool in pools {
                    pool.active_weight_bytes_per_token.0 /= 2;
                }
            }
        }
    }
    refresh(&mut v);
    v
}

#[test]
fn whole_and_offload_return_two_exact_configurations_without_admission() {
    let i: ModeInput =
        serde_json::from_slice(include_bytes!("../../../examples/mode-preview.json")).unwrap();
    let r = choose(&i, MODE).unwrap();
    assert!(r.search_complete);
    assert_eq!(r.reason, "SINGLE_DEVICE_FITS");
    assert!(!r.admission);
    assert_eq!(r.qualification, "not_assessed");
    let a = r.winner.unwrap();
    let b = r.runner_up.unwrap();
    assert_eq!(a.mode, PlacementMode::ExpertOffload);
    assert_eq!(b.mode, PlacementMode::WholeRequest);
    // Rows: stage + two layers + embedding/head. Active host/device bytes are45/450.
    assert_eq!(b.decode.memory_read_us.0, 495);
    assert_eq!(b.decode.total_us.0, 495);
    assert_eq!(b.decode.link_rtt_us.0, 0);
    assert!(a.decode.total_us < b.decode.total_us);
    assert_eq!(b.stages[0].pools[1].total_bytes.0, 1091);
    assert_ne!(
        a.profiles[0].runtime_config_digest,
        b.profiles[0].runtime_config_digest
    );
    assert!(choose(&i, FixtureMode::Reject).is_err());
    for family in [Family::Nvidia, Family::Apple] {
        let mut i = i.clone();
        i.family = family;
        for v in &mut i.variants {
            for n in &mut v.placement.nodes {
                n.family = family;
                if family == Family::Apple {
                    n.capabilities.os = NodeCapabilitiesOs::Macos;
                    n.capabilities.architecture = NodeCapabilitiesArchitecture::Aarch64;
                }
            }
        }
        assert_eq!(
            choose(&i, MODE).unwrap().winner.unwrap().mode,
            PlacementMode::ExpertOffload
        );
    }
}

#[test]
fn capacity_split_uses_tier_cost_inside_search_and_one_directed_round_trip() {
    let mut v = variant(3, false);
    // Native phase-time ranking prefers node3. Memory-tier ranking must exclude it.
    for (index, n) in v.placement.nodes.iter_mut().enumerate() {
        if let ModelCosts::Available {
            layers,
            per_stage_timing,
            ..
        } = &mut n.profile.costs
        {
            layers[0].timing.as_mut().unwrap().decode_us_per_token = Decimal([1000, 100, 1][index]);
            per_stage_timing.as_mut().unwrap().decode_us_per_token = Decimal([1000, 100, 1][index]);
        }
    }
    refresh(&mut v);
    let slow = v.placement.nodes[2].capabilities.node_id;
    for t in &mut v.costs.tiers {
        if t.node == slow {
            t.bytes_per_second = Decimal(1000);
        }
    }
    assert!(
        preview(&v.placement, MODE)
            .unwrap()
            .candidates
            .iter()
            .all(|c| c.stages.iter().any(|s| s.node == slow)),
        "old top-k is not a mode-cost shortlist"
    );
    let r = choose(&input(vec![v]), MODE).unwrap();
    assert_eq!(r.reason, "POOLING_FOR_CAPACITY");
    let best = r.winner.unwrap();
    assert_eq!(best.mode, PlacementMode::LayerSplit);
    assert!(best.stages.iter().all(|s| s.node != slow));
    assert_eq!(best.decode.memory_read_us.0, 660);
    assert_eq!(best.decode.link_rtt_us.0, 10);
    assert_eq!(best.decode.total_us.0, 670);
    assert!(r.runner_up.is_some());
    // A fitting offload configuration suppresses even an otherwise usable split.
    let v = variant(2, false);
    let r = choose(&input(vec![v.clone(), offload(&v)]), MODE).unwrap();
    assert_eq!(r.winner.unwrap().mode, PlacementMode::ExpertOffload);
    assert!(
        r.runner_up
            .is_none_or(|c| c.mode != PlacementMode::LayerSplit)
    );
    // A capacity-only resident search has enough budget to check all three
    // singles, but no budget for a split. A later fitting offload must still win.
    let mut v = variant(3, false);
    v.placement.max_states = BoundedU32::new(3).unwrap();
    let offload = offload(&v);
    for variants in [vec![v.clone(), offload.clone()], vec![offload, v]] {
        let r = choose(&input(variants), MODE).unwrap();
        assert!(r.search_complete);
        assert_eq!(r.winner.unwrap().mode, PlacementMode::ExpertOffload);
    }
}

#[test]
fn missing_stale_tail_overflow_and_search_caps_are_explicit_refusals() {
    for cause in [
        "missing",
        "stale",
        "future",
        "unknown_tail",
        "tail_limit",
        "active",
        "overflow",
        "storage_missing",
        "storage_unknown",
        "cap",
    ] {
        let mut v = variant(2, false);
        match cause {
            "missing" => v.costs.tiers.clear(),
            "stale" => {
                for t in &mut v.costs.tiers {
                    t.observation.age_at_capture_ms = Decimal(1000);
                }
            }
            "future" => {
                for t in &mut v.costs.tiers {
                    t.observation.captured_at_ms = Decimal(101);
                }
            }
            "unknown_tail" => {
                for l in &mut v.costs.links {
                    l.idle_wake_us = None;
                }
            }
            "tail_limit" => {
                for l in &mut v.costs.links {
                    l.p99_us = Some(Decimal(100));
                }
            }
            "cap" => v.placement.max_states = BoundedU32::new(1).unwrap(),
            "storage_missing" => v.costs.storage.clear(),
            "storage_unknown" => {
                for s in &mut v.costs.storage {
                    s.traffic = StorageTraffic::Unavailable;
                }
            }
            "active" | "overflow" => {
                for n in &mut v.placement.nodes {
                    if let ModelCosts::Available {
                        layers,
                        components,
                        per_stage,
                        ..
                    } = &mut n.profile.costs
                    {
                        for p in layers
                            .iter_mut()
                            .flat_map(|l| &mut l.pools)
                            .chain(components.iter_mut().flat_map(|c| &mut c.pools))
                            .chain(per_stage)
                        {
                            if cause == "overflow" && p.resident_weight_bytes.0 == 0 {
                                continue;
                            }
                            p.active_weight_bytes_per_token =
                                Decimal(if cause == "active" { 0 } else { u64::MAX / 8 });
                        }
                    }
                }
                refresh(&mut v);
                for t in &mut v.costs.tiers {
                    t.bytes_per_second = Decimal(1);
                }
            }
            _ => unreachable!(),
        }
        let r = choose(&input(vec![v]), MODE).unwrap();
        assert!(r.winner.is_none() && r.runner_up.is_none(), "{cause}");
        if cause == "cap" {
            assert_eq!(r.reason, "SEARCH_INCOMPLETE");
        } else {
            let expected = match cause {
                "missing" => "MODE_TIER_MISSING",
                "stale" | "future" => "MODE_TIER_UNQUALIFIED_OR_STALE",
                "unknown_tail" => "MODE_LINK_TAIL_UNKNOWN",
                "tail_limit" => "MODE_LINK_LIMIT",
                "active" => "MODE_ACTIVE_BYTES_MISSING",
                "overflow" => "MODE_COST_OVERFLOW",
                "storage_missing" => "MODE_STORAGE_MISSING",
                "storage_unknown" => "MODE_STORAGE_UNAVAILABLE",
                _ => unreachable!(),
            };
            assert!(
                r.refusals.contains_key(expected),
                "{cause}: {:?}",
                r.refusals
            );
        }
    }
}

#[test]
fn storage_reads_add_time_and_endpoint_bytes_without_becoming_memory() {
    let mut v = variant(1, true);
    v.costs.storage[0].traffic = StorageTraffic::Available {
        bytes_per_second: Decimal(1_000_000),
        layer_bytes_per_token: vec![Decimal(10), Decimal(20)],
        first_stage_bytes_per_token: Decimal(30),
        last_stage_bytes_per_token: Decimal(40),
        per_stage_bytes_per_token: Decimal(50),
        observation: observation(),
    };
    let r = choose(&input(vec![v.clone()]), MODE).unwrap();
    let c = r.winner.unwrap();
    assert_eq!(c.decode.storage_read_us.0, 150);
    assert_eq!(c.decode.total_us.0, 645);
    assert_eq!(c.stages[0].pools[1].total_bytes.0, 1091);
    let mut split = variant(2, false);
    for s in &mut split.costs.storage {
        s.traffic = v.costs.storage[0].traffic.clone();
    }
    let r = choose(&input(vec![split]), MODE).unwrap();
    assert_eq!(
        r.winner.unwrap().decode.storage_read_us.0,
        200,
        "stage I/O is repeated; first/last component I/O belongs only at its endpoint"
    );
    // Changing a declared storage observation does not relax any RAM/device fit.
    v.placement.nodes[0].capabilities.memory_pools[1].available_bytes = Decimal(900);
    let r = choose(&input(vec![v]), MODE).unwrap();
    assert!(r.winner.is_none());
}

#[test]
fn identities_snapshots_and_duplicate_rates_cannot_be_substituted() {
    let first = variant(1, true);
    let second = offload(&first);
    for cause in [
        "snapshot",
        "workload",
        "family",
        "tier_digest",
        "duplicate",
        "zero_bw",
    ] {
        let mut i = input(vec![first.clone(), second.clone()]);
        let v = &mut i.variants[1];
        match cause {
            "snapshot" => {
                v.placement.nodes[0].capabilities.memory_pools[0].available_bytes = Decimal(999)
            }
            "workload" => {
                v.placement.nodes[0].expected_profile_key.workload.sessions =
                    BoundedU32::new(1).unwrap()
            }
            "family" => v.placement.nodes[0].family = Family::Apple,
            "tier_digest" => v.costs.tiers[0].profile_digest = byte_digest(b"other profile"),
            "duplicate" => v.costs.tiers.push(v.costs.tiers[0].clone()),
            "zero_bw" => v.costs.tiers[0].bytes_per_second = Decimal(0),
            _ => unreachable!(),
        }
        assert_eq!(choose(&i, MODE).err(), Some(CoreError::Invalid), "{cause}");
    }
}

#[test]
fn declared_nonfixture_approvals_are_bound_but_never_grant_admission() {
    let mut v = variant(1, true);
    v.placement.fixture = false;
    let m = &mut v.placement.manifest;
    m.fixture = false;
    for asset in &mut m.assets {
        asset.sha256 = byte_digest(asset.path.as_bytes());
    }
    m.config_sha256 = m.assets[0].sha256.clone();
    m.tokenizer_sha256 = m.assets[1].sha256.clone();
    m.template_sha256 = m.assets[2].sha256.clone();
    m.manifest_digest =
        canonical_digest(&serde_json::to_value(&m).unwrap(), "manifest_digest").unwrap();
    for n in &mut v.placement.nodes {
        n.capabilities.fixture = false;
        n.profile.fixture = false;
        n.profile.key.manifest_digest = m.manifest_digest.clone();
        n.expected_profile_key = n.profile.key.clone();
        if let ModelCosts::Available { measurement, .. } = &mut n.profile.costs {
            measurement.basis = CostBasis::Measured;
            measurement.samples = BoundedU32::new(5).unwrap();
            measurement.relative_error_bps = Some(BoundedU32::new(500).unwrap());
        }
        n.approved_profile_digest = Some(content_hash(&n.profile).unwrap());
    }
    refresh(&mut v);
    let mut i = input(vec![v]);
    i.fixture = false;
    let r = choose(&i, FixtureMode::Reject).unwrap();
    assert!(r.winner.is_none());
    assert!(r.refusals.contains_key("UNAPPROVED_MODE_COSTS"));
    let v = &mut i.variants[0];
    v.approved_costs_digest = Some(content_hash(&(v.configuration, &v.costs)).unwrap());
    let r = choose(&i, FixtureMode::Reject).unwrap();
    assert!(r.winner.is_none());
    assert!(r.refusals.contains_key("MODE_TIER_UNQUALIFIED_OR_STALE"));
    let v = &mut i.variants[0];
    for t in &mut v.costs.tiers {
        t.observation.basis = CostBasis::Measured;
    }
    assert!(
        choose(&i, FixtureMode::Reject)
            .unwrap()
            .refusals
            .contains_key("UNAPPROVED_MODE_COSTS")
    );
    let v = &mut i.variants[0];
    v.approved_costs_digest = Some(content_hash(&(v.configuration, &v.costs)).unwrap());
    let r = choose(&i, FixtureMode::Reject).unwrap();
    assert!(r.winner.is_some());
    assert!(!r.admission);
    assert_eq!(r.qualification, "not_assessed");
}
