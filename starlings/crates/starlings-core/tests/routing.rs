use starlings_core::{CoreError, routing::*};
use starlings_protocol::*;
fn id<T: std::str::FromStr>(n: u32) -> T
where
    T::Err: std::fmt::Debug,
{
    format!("00000000-0000-4000-8000-{n:012x}").parse().unwrap()
}
fn node(n: u32, memory: u64) -> NodeCapabilities {
    let mut node: NodeCapabilities = decode(
        include_bytes!("../../../examples/node-capabilities.json"),
        FixtureMode::Simulation,
    )
    .unwrap();
    node.node_id = id(n);
    node.architecture = NodeCapabilitiesArchitecture::Aarch64;
    node.memory_pools[0].pool_id = "host".into();
    node.memory_pools[0].kind = MemoryPoolKind::System;
    node.memory_pools[0].available_bytes = Decimal(memory);
    node.backends[0].build_digest = byte_digest(b"synthetic CPU backend");
    node.backends[0].memory_pool_ids = vec!["host".into()];
    node.backends[0].max_sessions = BoundedU32::new(2).unwrap();
    node.backends[0].layer_sharding = false;
    node
}
fn replica(n: u32, cost_ms: u64) -> WarmReplica {
    WarmReplica {
        node: id(n),
        manifest: byte_digest(b"synthetic model"),
        profile: "fixture-bf16".into(),
        backend: "fake-v1".into(),
        build: byte_digest(b"synthetic CPU backend"),
        pool: "host".into(),
        evidence: byte_digest(b"synthetic test cost"),
        request_bytes: 100,
        measured_request_ms: cost_ms,
        context_limit: 2048,
        max_sessions: 2,
        accepting: true,
    }
}
fn task(n: u32, client: u32) -> Task {
    Task {
        id: id(n),
        client: id(client),
        manifest: byte_digest(b"synthetic model"),
        profile: "fixture-bf16".into(),
        context_tokens: 1024,
        deadline_ms: 10_000,
    }
}
#[test]
fn cpu_routing_is_exact_fair_bounded_and_retains_cancelled_reservations() {
    let mut router = TaskRouter::new(id(0x100), FixtureMode::Simulation);
    router.observe(node(1, 200), 0).unwrap();
    router.observe(node(2, 100), 0).unwrap();
    router.qualify(replica(1, 10)).unwrap();
    router.qualify(replica(2, 100)).unwrap();
    let mut incompatible = task(90, 90);
    incompatible.profile = "unsupported-quantization".into();
    router.enqueue(incompatible, 0).unwrap();
    for n in 1..=4 {
        router
            .enqueue(task(n, if n <= 3 { 10 } else { 11 }), 0)
            .unwrap();
    }
    let first = router.next(0).unwrap().unwrap();
    let second = router.next(0).unwrap().unwrap();
    assert_eq!((first.task.id, first.node), (id(1), id(1)));
    assert_eq!(second.node, id(1));
    let third = router.next(0).unwrap().unwrap();
    assert_eq!((third.task.id, third.node), (id(4), id(2))); // client 10 cannot consume every slot
    assert!(router.next(0).unwrap().is_none());
    router.output_started(first.task.id, 0).unwrap();
    router.cancel(first.task.id);
    assert_eq!(
        router.output_started(first.task.id, 0),
        Err(CoreError::OutputWithdrawn)
    );
    assert_eq!(
        router
            .assignments()
            .filter(|a| a.node == id(1))
            .map(|a| a.reserved_bytes)
            .sum::<u64>(),
        200
    );
    assert!(router.next(0).unwrap().is_none()); // cancel isn't drain
    router.drained(first.task.id).unwrap();
    assert_eq!(router.next(0).unwrap().unwrap().task.id, id(3));
    router.withdraw(id(2)); // hot/slow/revoked node does not cancel node 1
    assert!(
        router
            .assignments()
            .filter(|a| a.node == id(1))
            .all(|a| !a.cancelled)
    );
    assert!(
        router
            .assignments()
            .find(|a| a.node == id(2))
            .unwrap()
            .cancelled
    );
    assert!(router.qualify(replica(2, 1)).is_err()); // cannot rebind in-flight work
    assert!(router.next(5000).unwrap().is_none());
    assert!(router.assignments().all(|a| a.cancelled)); // stale reports withdraw output
    assert!(router.next(4999).is_err());
    router.next(10_000).unwrap();
    assert_eq!(router.take_expired().len(), 1); // incompatible queued task ends explicitly
    assert_eq!(router.queued(), 0);
}
#[test]
fn unqualified_insufficient_stale_and_changed_backends_cannot_admit() {
    for kind in [
        "unqualified",
        "memory",
        "profile",
        "build",
        "stale",
        "offline",
    ] {
        let mut router = TaskRouter::new(id(0x100), FixtureMode::Simulation);
        let mut report = node(1, if kind == "memory" { 99 } else { 100 });
        if kind == "offline" {
            report.state = NodeCapabilitiesState::Unavailable;
        }
        if kind == "profile" {
            report.backends[0].execution_profiles = vec!["different".into()];
        }
        if kind == "build" {
            report.backends[0].build_digest = byte_digest(b"different");
        }
        router.observe(report, 0).unwrap();
        if kind != "unqualified" {
            router.qualify(replica(1, 10)).unwrap();
        }
        router.enqueue(task(1, 10), 0).unwrap();
        assert!(
            router
                .next(if kind == "stale" { 5000 } else { 0 })
                .unwrap()
                .is_none(),
            "{kind}"
        );
    }
    let mut router = TaskRouter::new(id(0x100), FixtureMode::Simulation);
    router.observe(node(1, 200), 0).unwrap();
    router.qualify(replica(1, 1)).unwrap();
    for n in 1..=8 {
        router.enqueue(task(n, 10), 0).unwrap();
    }
    assert_eq!(router.enqueue(task(9, 10), 0), Err(CoreError::Limit));
    assert_eq!(router.enqueue(task(1, 10), 0), Err(CoreError::Conflict));
    let active = router.next(0).unwrap().unwrap();
    let mut changed = node(1, 200);
    changed.snapshot_id = id(500);
    changed.observed_at = "2026-10-12T00:01:00Z".into();
    changed.backends[0].build_digest = byte_digest(b"replaced service");
    router.observe(changed, 1).unwrap();
    assert_eq!(
        router.output_started(active.task.id, 1),
        Err(CoreError::OutputWithdrawn)
    );
}

#[test]
fn queue_expiry_is_bounded_and_delivered_before_capacity_is_reused() {
    let mut router = TaskRouter::new(id(0x100), FixtureMode::Simulation);
    for n in 0..128 {
        let mut task = task(n, 1000 + n);
        task.manifest = byte_digest(&(n / 32).to_be_bytes());
        router.enqueue(task, 0).unwrap();
    }
    assert_eq!(router.enqueue(task(999, 999), 0), Err(CoreError::Limit));
    assert!(router.next(10_000).unwrap().is_none());
    let mut later = task(999, 999);
    later.deadline_ms = 20_000;
    assert_eq!(router.enqueue(later.clone(), 10_000), Err(CoreError::Limit));
    assert_eq!(router.take_expired().len(), 128);
    router.enqueue(later, 10_000).unwrap();
    assert_eq!(router.queued(), 1);
    assert!(router.take_expired().is_empty());
}
