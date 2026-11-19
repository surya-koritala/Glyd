use starlings_core::{
    CoreError,
    admission::*,
    planner::{self, LinkCost, PreviewInput},
    profiles,
};
use starlings_protocol::*;
use std::collections::BTreeMap;

const MODE: FixtureMode = FixtureMode::Simulation;
fn input(split: bool) -> PreviewInput {
    let value: serde_json::Value =
        serde_json::from_slice(include_bytes!("../../../examples/mode-preview.json")).unwrap();
    let mut input: PreviewInput =
        serde_json::from_value(value["variants"][0]["placement"].clone()).unwrap();
    if split {
        let node = &input.nodes[0];
        let first = profiles::assess(
            &node.profile,
            &input.manifest,
            &node.expected_profile_key,
            0..1,
            node.profile_clocks,
            None,
            MODE,
        )
        .unwrap();
        let last = profiles::assess(
            &node.profile,
            &input.manifest,
            &node.expected_profile_key,
            1..2,
            node.profile_clocks,
            None,
            MODE,
        )
        .unwrap();
        for pool in &mut input.nodes[0].capabilities.memory_pools {
            let bytes = first
                .pools
                .iter()
                .chain(&last.pools)
                .filter(|p| p.pool_id == pool.pool_id)
                .map(|p| p.total_bytes.0)
                .max()
                .unwrap();
            pool.available_bytes = Decimal(bytes);
        }
        let mut second = input.nodes[0].clone();
        second.capabilities.node_id = "00000000-0000-4000-8000-000000000002".parse().unwrap();
        second.capabilities.snapshot_id = "00000000-0000-4000-8000-000000000102".parse().unwrap();
        let a = input.nodes[0].capabilities.node_id;
        let b = second.capabilities.node_id;
        input.nodes.push(second);
        input.max_stages = BoundedU32::new(2).unwrap();
        input.links = [(a, b), (b, a)]
            .into_iter()
            .map(|(source, destination)| LinkCost {
                source,
                destination,
                fixture: true,
                boundary_profile: input.manifest.required_boundary_profile.clone(),
                workload_digest: content_hash(&input.nodes[0].expected_profile_key.workload)
                    .unwrap(),
                evidence_digest: byte_digest(b"synthetic link"),
                transfer: PhaseCost {
                    prefill_us: Decimal(1),
                    decode_us_per_token: Decimal(1),
                },
                age_at_capture_ms: Decimal(0),
                captured_at_ms: input.now_ms,
            })
            .collect();
    }
    let preview = planner::preview(&input, MODE).unwrap();
    assert_eq!(
        preview.candidates[0].stages.len(),
        if split { 2 } else { 1 }
    );
    input
}
fn book(input: &PreviewInput, epoch: u64, incarnation: u64) -> AdmissionBook {
    AdmissionBook::new(
        input.nodes[0].capabilities.cluster_id,
        format!("00000000-0000-4000-8000-{incarnation:012x}")
            .parse()
            .unwrap(),
        epoch,
        MODE,
    )
    .unwrap()
}
fn actual(book: &AdmissionBook, id: ReservationId, stage: usize) -> BTreeMap<String, u64> {
    book.reservation(id).unwrap().candidate.stages[stage]
        .pools
        .iter()
        .map(|p| (p.pool_id.clone(), p.total_bytes.0))
        .collect()
}
fn cleanup(book: &mut AdmissionBook, id: ReservationId) {
    let stages = book.reservation(id).unwrap().candidate.stages.len();
    book.abort(id).unwrap();
    for stage in 0..stages {
        book.drained(id, stage).unwrap();
    }
}

#[test]
fn all_stage_prepare_and_commit_barriers_retain_rollback_ownership() {
    let input = input(true);
    let mut book = book(&input, 1, 10);
    let id = book.begin(&input, 0, 100).unwrap();
    assert_eq!(
        book.reservation(id).unwrap().input_digest,
        content_hash(&input).unwrap()
    );
    assert_eq!(book.start_commit(id, 100), Err(CoreError::State));
    assert_eq!(book.committed(id, 0, 100), Err(CoreError::State));
    book.prepared(id, 0, actual(&book, id, 0), 100).unwrap();
    assert_eq!(book.start_commit(id, 100), Err(CoreError::State));
    book.prepared(id, 1, actual(&book, id, 1), 100).unwrap();
    book.start_commit(id, 100).unwrap();
    book.committed(id, 0, 100).unwrap();
    assert_eq!(book.reservation(id).unwrap().state, State::Committing);
    book.committed(id, 1, 100).unwrap();
    assert_eq!(book.reservation(id).unwrap().state, State::Ready);
    assert_eq!(book.drained(id, 0), Err(CoreError::State));
    book.abort(id).unwrap();
    book.drained(id, 0).unwrap();
    assert_eq!(book.begin(&input, 0, 100), Err(CoreError::Capacity));
    book.drained(id, 1).unwrap();
    let next = book.begin(&input, 0, 100).unwrap();
    assert_ne!(next, id);
    assert_eq!(book.drained(id, 1), Err(CoreError::NotFound));
    assert_eq!(book.reservation(next).unwrap().state, State::Preparing);
}

#[test]
fn competing_plans_cannot_raise_live_capacity_with_new_telemetry_or_inventory() {
    let input = input(true);
    let mut book = book(&input, 1, 10);
    let id = book.begin(&input, 0, 100).unwrap();
    assert_eq!(book.begin(&input, 0, 100), Err(CoreError::Capacity));
    let mut inflated = input.clone();
    for node in &mut inflated.nodes {
        for pool in &mut node.capabilities.memory_pools {
            pool.available_bytes = pool.total_bytes;
        }
    }
    assert_eq!(book.begin(&inflated, 0, 100), Err(CoreError::Capacity));
    for node in &mut inflated.nodes {
        node.profile.key.device_digest = byte_digest(b"replaced device");
        node.expected_profile_key = node.profile.key.clone();
    }
    assert_eq!(book.begin(&inflated, 0, 100), Err(CoreError::Conflict));
    cleanup(&mut book, id);
    book.begin(&inflated, 0, 100).unwrap();
}

#[test]
fn expired_and_overbudget_prepare_fence_node_until_all_cleanup_is_trusted() {
    let mut input = input(true);
    let mut book = book(&input, 1, 10);
    let id = book.begin(&input, 0, 100).unwrap();
    let mut over = actual(&book, id, 0);
    *over.values_mut().next().unwrap() += 1;
    book.advance(200).unwrap();
    assert_eq!(book.reservation(id).unwrap().state, State::Releasing);
    assert_eq!(book.prepared(id, 0, over, 200), Err(CoreError::Capacity));
    input.now_ms = Decimal(200);
    for n in &mut input.nodes {
        n.profile_clocks.now_ms = 200;
        n.profile_clocks.now_wall_ms += 100;
    }
    input.now_wall_ms += 100;
    assert_eq!(book.begin(&input, 0, 100), Err(CoreError::State));
    book.drained(id, 0).unwrap();
    assert_eq!(book.begin(&input, 0, 100), Err(CoreError::State));
    book.drained(id, 1).unwrap();
    book.begin(&input, 0, 100).unwrap();
    assert_eq!(book.advance(199), Err(CoreError::Clock));
}

#[test]
fn exact_prepare_repeat_is_historical_but_conflicts_abort_without_releasing() {
    let input = input(false);
    let mut book = book(&input, 1, 10);
    let id = book.begin(&input, 0, 100).unwrap();
    let observed = actual(&book, id, 0);
    book.prepared(id, 0, observed.clone(), 100).unwrap();
    book.prepared(id, 0, observed.clone(), 100).unwrap();
    let mut changed = observed;
    *changed.values_mut().next().unwrap() -= 1;
    assert_eq!(book.prepared(id, 0, changed, 100), Err(CoreError::Conflict));
    assert_eq!(book.reservation(id).unwrap().state, State::Releasing);
    assert_eq!(book.start_commit(id, 100), Err(CoreError::State));
    assert_eq!(book.begin(&input, 0, 100), Err(CoreError::State));
    cleanup(&mut book, id);
    for _ in 0..8 {
        let fresh = book.begin(&input, 0, 100).unwrap();
        assert_ne!(id, fresh);
        cleanup(&mut book, fresh);
    }
}

#[test]
fn late_invalid_allocation_requires_fresh_cleanup_of_the_affected_stage() {
    let input = input(true);
    for conflict in [false, true] {
        let mut book = book(&input, 1, 10);
        let id = book.begin(&input, 0, 100).unwrap();
        let mut changed = actual(&book, id, 0);
        let error = if conflict {
            book.prepared(id, 0, changed.clone(), 100).unwrap();
            *changed.values_mut().next().unwrap() -= 1;
            CoreError::Conflict
        } else {
            *changed.values_mut().next().unwrap() += 1;
            CoreError::Capacity
        };
        book.abort(id).unwrap();
        book.drained(id, 0).unwrap();
        assert_eq!(book.prepared(id, 0, changed, 100), Err(error));
        book.drained(id, 1).unwrap();
        assert_eq!(book.reservation(id).unwrap().state, State::Releasing);
        assert_eq!(book.begin(&input, 0, 100), Err(CoreError::State));
        book.drained(id, 0).unwrap();
        assert!(matches!(book.reservation(id), Err(CoreError::NotFound)));
        book.begin(&input, 0, 100).unwrap();
    }
}

#[test]
fn owner_withdrawal_never_reopens_or_forgets_cleanup() {
    let input = input(false);
    let mut other = input.clone();
    other.nodes[0].capabilities.node_id = "00000000-0000-4000-8000-000000000002".parse().unwrap();
    other.nodes[0].capabilities.snapshot_id =
        "00000000-0000-4000-8000-000000000102".parse().unwrap();
    let mut empty = book(&input, 1, 10);
    empty.withdraw();
    assert_eq!(empty.begin(&input, 0, 100), Err(CoreError::State));
    for state in [State::Preparing, State::Committing, State::Ready] {
        let mut book = book(&input, 1, 10);
        let first = book.begin(&input, 0, 100).unwrap();
        let second = book.begin(&other, 0, 100).unwrap();
        let allocations = actual(&book, first, 0);
        if state != State::Preparing {
            book.prepared(first, 0, allocations.clone(), 100).unwrap();
            book.start_commit(first, 100).unwrap();
            if state == State::Ready {
                book.committed(first, 0, 100).unwrap();
            }
        }
        assert_eq!(book.reservation(first).unwrap().state, state);
        book.withdraw();
        for id in [first, second] {
            assert_eq!(book.reservation(id).unwrap().state, State::Releasing);
            assert_eq!(book.start_commit(id, 100), Err(CoreError::State));
            assert_eq!(book.committed(id, 0, 100), Err(CoreError::State));
        }
        assert_eq!(
            book.prepared(first, 0, allocations, 100),
            Err(CoreError::State)
        );
        book.drained(first, 0).unwrap();
        book.withdraw();
        assert_eq!(book.begin(&input, 0, 100), Err(CoreError::State));
        assert_eq!(book.reservation(second).unwrap().state, State::Releasing);
        book.drained(second, 0).unwrap();
        assert_eq!(book.begin(&input, 0, 100), Err(CoreError::State));
    }
    let input = self::input(true);
    let mut book = book(&input, 1, 10);
    let id = book.begin(&input, 0, 100).unwrap();
    book.abort(id).unwrap();
    book.drained(id, 0).unwrap();
    book.withdraw();
    book.drained(id, 1).unwrap();
    assert!(matches!(book.reservation(id), Err(CoreError::NotFound)));
    assert_eq!(book.begin(&input, 0, 100), Err(CoreError::State));
}

#[test]
fn fixture_stale_capture_and_cross_owner_handles_never_become_readiness() {
    let input = input(false);
    let mut owner = book(&input, 1, 10);
    let id = owner.begin(&input, 0, 100).unwrap();
    for mut other in [book(&input, 2, 10), book(&input, 1, 11)] {
        let own = other.begin(&input, 0, 100).unwrap();
        assert_eq!(other.abort(id), Err(CoreError::Conflict));
        assert_eq!(other.reservation(own).unwrap().state, State::Preparing);
    }
    let mut production = AdmissionBook::new(
        input.nodes[0].capabilities.cluster_id,
        "00000000-0000-4000-8000-000000000020".parse().unwrap(),
        1,
        FixtureMode::Reject,
    )
    .unwrap();
    assert_eq!(production.begin(&input, 0, 100), Err(CoreError::Invalid));
    let mut stale = input.clone();
    stale.nodes[0].capabilities.sample_age_ms = BoundedU32::new(5000).unwrap();
    assert_eq!(owner.begin(&stale, 0, 100), Err(CoreError::NotFound));
    assert_eq!(owner.begin(&input, 0, 0), Err(CoreError::Invalid));
    assert_eq!(
        owner.prepared(id, 3, BTreeMap::new(), 100),
        Err(CoreError::Invalid)
    );
    assert_eq!(owner.reservation(id).unwrap().state, State::Preparing);
}
