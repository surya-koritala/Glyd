use serde_json::json;
use starlings_core::{CoreError, planner::*};
use starlings_protocol::*;
mod support;
use support::{age, fixture};
const MODE: FixtureMode = FixtureMode::Simulation;

fn input(layer_count: usize) -> PreviewInput {
    let (mut manifest, mut profile) = fixture();
    let original = manifest.layers.clone();
    manifest.layers = (0..layer_count)
        .map(|i| {
            let mut layer = original[i % 2].clone();
            layer.index = BoundedU32::new(i as u32).unwrap();
            layer
        })
        .collect();
    manifest.manifest_digest =
        canonical_digest(&serde_json::to_value(&manifest).unwrap(), "manifest_digest").unwrap();
    profile.key.manifest_digest = manifest.manifest_digest.clone();
    if let ModelCosts::Available { layers, .. } = &mut profile.costs {
        let original = layers.clone();
        *layers = (0..layer_count)
            .map(|i| {
                let mut layer = original[i % 2].clone();
                layer.index = BoundedU32::new(i as u32).unwrap();
                layer
            })
            .collect();
    }
    let nodes: Vec<_> = (1..=3)
        .map(|i| {
            let mut cap: serde_json::Value = serde_json::from_slice(include_bytes!(
                "../../../examples/node-capabilities.json"
            ))
            .unwrap();
            cap["node_id"] = json!(format!("00000000-0000-4000-8000-{i:012}"));
            cap["snapshot_id"] = json!(format!("00000000-0000-4000-8000-{:012}", i + 100));
            cap["memory_pools"] = json!([
                {"pool_id":"host","kind":"system","total_bytes":"1000","available_bytes":"500","reserved_bytes":"0","device_ids":["mock:0"]},
                {"pool_id":"device","kind":"device","total_bytes":"2000","available_bytes":"900","reserved_bytes":"0","device_ids":["mock:0"]}
            ]);
            cap["backends"][0]["memory_pool_ids"] = json!(["host", "device"]);
            cap["backends"][0]["max_sessions"] = json!(2);
            cap["backends"][0]["build_digest"] = json!(profile.key.backend_build_digest);
            let capabilities: NodeCapabilities = decode(&serde_json::to_vec(&cap).unwrap(), MODE).unwrap();
            NodeInput {
                capabilities,
                captured_at_ms: Decimal(100),
                family: Family::Nvidia,
                backend_id: "fake-v1".into(),
                expected_profile_key: profile.key.clone(),
                profile: profile.clone(),
                profile_clocks: age(0),
                approved_profile_digest: None,
                load_us: Decimal(i * 5),
                copy: PhaseCost { prefill_us: Decimal(i * 2), decode_us_per_token: Decimal(i) },
            }
        })
        .collect();
    let mut links = Vec::new();
    for (a, source) in nodes.iter().enumerate() {
        for (b, destination) in nodes.iter().enumerate() {
            if a != b {
                links.push(LinkCost {
                    source: source.capabilities.node_id,
                    destination: destination.capabilities.node_id,
                    fixture: true,
                    boundary_profile: manifest.required_boundary_profile.clone(),
                    workload_digest: content_hash(&profile.key.workload).unwrap(),
                    evidence_digest: byte_digest(format!("edge {a} {b}").as_bytes()),
                    transfer: PhaseCost {
                        prefill_us: Decimal((7 + a * 3 + b) as u64),
                        decode_us_per_token: Decimal((2 + a * 3 + b) as u64),
                    },
                    age_at_capture_ms: Decimal(0),
                    captured_at_ms: Decimal(100),
                });
            }
        }
    }
    PreviewInput {
        version: Version,
        fixture: true,
        manifest,
        now_ms: Decimal(100),
        now_wall_ms: age(0).now_wall_ms,
        nodes,
        links,
        objective: Objective::Latency,
        max_stages: BoundedU32::new(3).unwrap(),
        top_k: BoundedU32::new(16).unwrap(),
        max_states: BoundedU32::new(100000).unwrap(),
        max_cost_rows: BoundedU32::new(1000000).unwrap(),
    }
}
type Ranked = (u64, Vec<(NodeId, usize, usize)>);
fn ranked(report: &Preview) -> Vec<Ranked> {
    report
        .candidates
        .iter()
        .map(|c| {
            (
                c.score_us.0,
                c.stages.iter().map(|s| (s.node, s.start, s.end)).collect(),
            )
        })
        .collect()
}

// Independent oracle: enumerate every per-layer node assignment (not paths/cuts),
// discard noncontiguous/reused nodes and calculate this fixture's bytes by hand.
fn exhaustive(input: &PreviewInput) -> Vec<Ranked> {
    let count = input.manifest.layers.len();
    let mut answers = Vec::new();
    for mut value in 0..input.nodes.len().pow(count as u32) {
        let mut assignment = Vec::new();
        for _ in 0..count {
            assignment.push(value % input.nodes.len());
            value /= input.nodes.len();
        }
        let mut groups = Vec::new();
        let mut start = 0;
        for end in 1..=count {
            if end == count || assignment[end] != assignment[start] {
                groups.push((assignment[start], start, end));
                start = end;
            }
        }
        let distinct: std::collections::BTreeSet<_> = groups.iter().map(|g| g.0).collect();
        if groups.len() > input.max_stages.get() as usize || groups.len() != distinct.len() {
            continue;
        }
        let mut load = 0;
        let mut prefill = 0;
        let mut decode = 0;
        let mut bottleneck = 0;
        let mut valid = true;
        for (i, &(node, start, end)) in groups.iter().enumerate() {
            let n = &input.nodes[node];
            let extras = usize::from(start == 0) + usize::from(end == count);
            let rows = (1 + end - start + extras) as u64;
            let state: u64 = (start..end).map(|l| if l % 2 == 0 { 16 } else { 40 }).sum();
            let bytes = [
                30 + 10 * (end - start) as u64 + 20 * extras as u64 + 27 * rows,
                300 + 100 * (end - start) as u64 + 200 * extras as u64 + 27 * rows + state,
            ];
            valid &= n
                .capabilities
                .memory_pools
                .iter()
                .zip(bytes)
                .all(|(p, b)| b <= p.available_bytes.0 - p.reserved_bytes.0);
            load += n.load_us.0;
            prefill += 13 * rows + n.copy.prefill_us.0;
            let mut cycle = 17 * rows + n.copy.decode_us_per_token.0;
            decode += cycle;
            if let Some(&(next, _, _)) = groups.get(i + 1) {
                let edge = input
                    .links
                    .iter()
                    .find(|l| {
                        l.source == n.capabilities.node_id
                            && l.destination == input.nodes[next].capabilities.node_id
                    })
                    .unwrap();
                prefill += edge.transfer.prefill_us.0;
                decode += edge.transfer.decode_us_per_token.0;
                cycle += edge.transfer.decode_us_per_token.0;
            }
            bottleneck = bottleneck.max(cycle);
        }
        if valid {
            let score = match input.objective {
                Objective::Latency => load + prefill + 4 * decode,
                Objective::Throughput => bottleneck,
            };
            answers.push((
                score,
                groups
                    .iter()
                    .map(|&(n, s, e)| (input.nodes[n].capabilities.node_id, s, e))
                    .collect(),
            ));
        }
    }
    answers.sort();
    answers.truncate(input.top_k.get() as usize);
    answers
}
#[test]
fn bounded_search_matches_exhaustive_assignments_and_never_reserves() {
    for layers in 2..=4 {
        for objective in [Objective::Latency, Objective::Throughput] {
            let mut input = input(layers);
            input.objective = objective;
            let before = serde_json::to_value(&input).unwrap();
            let report = preview(&input, MODE).unwrap();
            assert!(report.search_complete);
            assert!(!report.admission);
            assert_eq!(report.qualification, "not_assessed");
            assert_eq!(
                ranked(&report),
                exhaustive(&input),
                "layers={layers} objective={objective:?}"
            );
            assert!(!report.candidates.is_empty());
            assert_eq!(serde_json::to_value(&input).unwrap(), before);
            for candidate in &report.candidates {
                assert_eq!(candidate.stages[0].start, 0);
                assert_eq!(candidate.stages.last().unwrap().end, layers);
                assert!(candidate.stages.windows(2).all(|p| p[0].end == p[1].start));
                assert!(candidate.stages.iter().all(|s| s.start < s.end));
                for stage in &candidate.stages {
                    let node = input
                        .nodes
                        .iter()
                        .find(|n| n.capabilities.node_id == stage.node)
                        .unwrap();
                    assert_eq!(stage.snapshot, node.capabilities.snapshot_id);
                    assert_eq!(stage.profile_digest, content_hash(&node.profile).unwrap());
                    assert!(stage.pools.iter().all(|budget| {
                        let pool = node
                            .capabilities
                            .memory_pools
                            .iter()
                            .find(|p| p.pool_id == budget.pool_id)
                            .unwrap();
                        budget.total_bytes.0 <= pool.available_bytes.0 - pool.reserved_bytes.0
                    }));
                }
            }
            input.nodes.reverse();
            assert_eq!(ranked(&preview(&input, MODE).unwrap()), ranked(&report));
        }
    }
}
#[test]
fn single_machine_fit_is_preferred_and_slow_devices_are_optional() {
    let mut input = input(2);
    input.nodes[0].capabilities.memory_pools[1].available_bytes = Decimal(1500);
    input.nodes[0].capabilities.backends[0].layer_sharding = false;
    input.nodes[1].load_us = Decimal(1_000_000);
    let report = preview(&input, MODE).unwrap();
    assert_eq!(report.candidates.len(), 1);
    assert_eq!(report.candidates[0].stages.len(), 1);
    assert_eq!(
        report.candidates[0].stages[0].node,
        input.nodes[0].capabilities.node_id
    );
    // This is a placement policy, even if a hypothetical pooled score were smaller.
    assert_eq!(report.candidates[0].stages[0].pools[1].total_bytes.0, 1091);
    // Unknown single-worker timing must not manufacture a capacity need for pooling.
    let mut missing = input.clone();
    if let ModelCosts::Available { layers, .. } = &mut missing.nodes[0].profile.costs {
        layers[0].timing = None;
    }
    let no_pool = preview(&missing, MODE).unwrap();
    assert!(no_pool.candidates.is_empty());
    assert!(no_pool.refusals.contains_key("POOLING_NOT_NEEDED"));
    input.nodes[0].capabilities.memory_pools[1].available_bytes = Decimal(900);
    input.nodes[0].capabilities.backends[0].layer_sharding = true;
    let report = preview(&input, MODE).unwrap();
    assert!(
        report.candidates[0]
            .stages
            .iter()
            .all(|s| s.node != input.nodes[1].capabilities.node_id)
    );
}
#[test]
fn directed_edges_and_same_family_compatibility_are_required() {
    let mut input = input(2);
    input.links.truncate(1);
    let report = preview(&input, MODE).unwrap();
    assert_eq!(report.candidates.len(), 1);
    assert_eq!(report.candidates[0].stages[0].node, input.links[0].source);
    assert_eq!(
        report.candidates[0].stages[1].node,
        input.links[0].destination
    );
    assert_eq!(
        report.candidates[0].link_evidence,
        vec![input.links[0].evidence_digest.clone()]
    );
    input.nodes[1].family = Family::Apple;
    input.nodes[1].capabilities.os = NodeCapabilitiesOs::Macos;
    input.nodes[1].capabilities.architecture = NodeCapabilitiesArchitecture::Aarch64;
    let report = preview(&input, MODE).unwrap();
    assert!(report.candidates.is_empty());
    assert!(report.refusals.contains_key("FAMILY_MISMATCH"));
    input.nodes[0].family = Family::Apple;
    input.nodes[0].capabilities.os = NodeCapabilitiesOs::Macos;
    input.nodes[0].capabilities.architecture = NodeCapabilitiesArchitecture::Aarch64;
    assert_eq!(preview(&input, MODE).unwrap().candidates.len(), 1);
}

#[test]
fn node_refusals_identify_attempts_without_changing_ranking_or_admission() {
    let base = input(2);
    let first = base.nodes[0].capabilities.node_id;
    let second = base.nodes[1].capabilities.node_id;
    let baseline = preview(&base, MODE).unwrap();
    assert_eq!(ranked(&baseline), exhaustive(&base));
    // A whole-model rejection does not exclude this node from a smaller stage.
    assert!(baseline.node_refusals[&first].contains_key("INSUFFICIENT_POOL_MEMORY"));
    assert!(
        baseline
            .candidates
            .iter()
            .any(|c| c.stages.iter().any(|s| s.node == first))
    );
    for (variant, code) in [
        ("stale", "STALE_OR_UNAVAILABLE_CAPABILITY"),
        ("family", "FAMILY_MISMATCH"),
        ("backend", "INCOMPATIBLE_BACKEND"),
        ("pool", "POOL_BINDING_MISMATCH"),
        ("profile", "INVALID_PROFILE"),
    ] {
        let mut input = base.clone();
        match variant {
            "stale" => input.nodes[0].capabilities.sample_age_ms = BoundedU32::new(5000).unwrap(),
            "family" => input.nodes[0].family = Family::Apple,
            "backend" => input.nodes[0].backend_id = "unavailable-backend".into(),
            "pool" => {
                input.nodes[0].capabilities.backends[0]
                    .memory_pool_ids
                    .remove(0);
            }
            "profile" => {
                input.nodes[0].expected_profile_key.runtime_config_digest =
                    byte_digest(b"wrong config")
            }
            _ => unreachable!(),
        }
        let report = preview(&input, MODE).unwrap();
        assert_eq!(report.node_refusals[&first][code], 1, "{variant}");
        assert!(
            report
                .node_refusals
                .get(&second)
                .is_none_or(|r| !r.contains_key(code))
        );
        for (code, count) in &report.refusals {
            assert!(
                report
                    .node_refusals
                    .values()
                    .filter_map(|r| r.get(code))
                    .sum::<u32>()
                    <= *count
            );
        }
        assert!(!report.admission);
        assert_eq!(report.qualification, "not_assessed");
        let serialized = serde_json::to_string(&report).unwrap();
        assert!(!serialized.contains("tls://"));
        assert!(!serialized.contains("unavailable-backend"));
        assert!(!serialized.contains("wrong config"));
    }
    let mut capped = base;
    capped.max_states = BoundedU32::new(1).unwrap();
    let report = preview(&capped, MODE).unwrap();
    assert!(!report.search_complete);
    assert!(report.node_refusals.len() <= 3);
    assert_eq!(report.states, 1);
}
#[test]
fn limits_are_explicit_and_malformed_or_stale_inputs_do_not_become_plans() {
    let base = input(2);
    assert_eq!(
        preview(&base, FixtureMode::Reject).err(),
        Some(CoreError::Invalid)
    );
    for kind in ["states", "rows"] {
        let mut input = base.clone();
        if kind == "states" {
            input.max_states = BoundedU32::new(1).unwrap();
        } else {
            input.max_cost_rows = BoundedU32::new(1).unwrap();
        }
        let report = preview(&input, MODE).unwrap();
        assert!(!report.search_complete);
        assert!(report.candidates.is_empty());
    }
    let mut input = base.clone();
    input.links[0].age_at_capture_ms = Decimal(30_000);
    input.links.push(input.links[0].clone());
    assert_eq!(preview(&input, MODE).err(), Some(CoreError::Invalid));
    input = base.clone();
    input.nodes[1].capabilities.node_id = input.nodes[0].capabilities.node_id;
    assert_eq!(preview(&input, MODE).err(), Some(CoreError::Invalid));
    input = base.clone();
    input.nodes[0].profile_clocks.now_ms = 99;
    assert_eq!(preview(&input, MODE).err(), Some(CoreError::Invalid));
    input = base.clone();
    input.nodes[0].load_us = Decimal(u64::MAX);
    assert_eq!(preview(&input, MODE).err(), Some(CoreError::Limit));
    for variant in [
        "capability",
        "profile",
        "timing",
        "backend",
        "sharding",
        "pool",
        "host",
        "reserved",
        "context",
        "sessions",
        "task",
        "binding",
        "key",
        "family",
        "link",
    ] {
        let mut input = base.clone();
        for n in &mut input.nodes {
            match variant {
                "capability" => n.capabilities.sample_age_ms = BoundedU32::new(5000).unwrap(),
                "profile" => {
                    if let ModelCosts::Available { measurement, .. } = &mut n.profile.costs {
                        measurement.valid_for_ms = BoundedU32::new(1).unwrap();
                    }
                }
                "timing" => {
                    if let ModelCosts::Available { layers, .. } = &mut n.profile.costs {
                        layers[0].timing = None;
                        layers[1].timing = None;
                    }
                }
                "backend" => n.capabilities.backends[0].build_digest = byte_digest(b"other build"),
                "sharding" => n.capabilities.backends[0].layer_sharding = false,
                "pool" => n.capabilities.memory_pools[1].available_bytes = Decimal(1),
                "host" => n.capabilities.memory_pools[0].available_bytes = Decimal(100),
                "reserved" => n.capabilities.memory_pools[1].reserved_bytes = Decimal(200),
                "context" => {
                    n.capabilities.backends[0].max_context_tokens = BoundedU32::new(6).unwrap()
                }
                "sessions" => n.capabilities.backends[0].max_sessions = BoundedU32::new(1).unwrap(),
                "task" => n.capabilities.backends[0].task_types = vec![BackendTaskTypes::Embed],
                "binding" => {
                    n.capabilities.backends[0].memory_pool_ids.remove(0);
                }
                "key" => {
                    n.expected_profile_key.runtime_config_digest = byte_digest(b"other config")
                }
                "family" => n.family = Family::Apple,
                "link" => {}
                _ => unreachable!(),
            }
        }
        if variant == "link" {
            for l in &mut input.links {
                l.age_at_capture_ms = Decimal(30000);
            }
        }
        let report = preview(&input, MODE).unwrap();
        assert!(report.candidates.is_empty(), "{variant}");
        assert!(!report.refusals.is_empty(), "{variant}");
    }
}
