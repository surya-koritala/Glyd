use starlings_core::*;
use starlings_protocol::*;
const MODE: FixtureMode = FixtureMode::Simulation;
fn id<T: std::str::FromStr>(n: u64) -> T
where
    T::Err: std::fmt::Debug,
{
    format!("00000000-0000-4000-8000-{n:012x}").parse().unwrap()
}
fn plan() -> PlacementPlan {
    decode(
        include_bytes!("../../../examples/placement-plan.json"),
        MODE,
    )
    .unwrap()
}
fn lease() -> Lease {
    decode(include_bytes!("../../../examples/lease.json"), MODE).unwrap()
}
fn session() -> Session {
    decode(include_bytes!("../../../examples/session.json"), MODE).unwrap()
}
fn header() -> TensorHeader {
    let mut h: TensorHeader =
        decode(include_bytes!("../../../examples/tensor-header.json"), MODE).unwrap();
    h.destination_stage = BoundedU32::new(0).unwrap();
    h
}
fn worker() -> Worker {
    Worker::new(id(1), plan().cluster_id, 4_194_304, 1, MODE).unwrap()
}
fn request(n: u64, op: Operation) -> ControlRequest {
    ControlRequest::new(
        id(n),
        1,
        {
            static SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
            SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        },
        op,
    )
    .unwrap()
}
fn prepare(l: Lease) -> Operation {
    Operation::Prepare {
        plan: Box::new(plan()),
        lease: Box::new(l),
    }
}
fn ready(w: &mut Worker) {
    w.apply(&request(1000, prepare(lease())), 0).unwrap();
    w.apply(
        &request(
            1001,
            Operation::Commit {
                lease_id: lease().lease_id,
            },
        ),
        0,
    )
    .unwrap();
    w.apply(
        &request(
            1002,
            Operation::Begin {
                session: Box::new(session()),
            },
        ),
        0,
    )
    .unwrap();
}
fn start() -> Operation {
    Operation::StartStep {
        header: Box::new(header()),
        input_digest: byte_digest(b"fixture"),
    }
}

#[test]
fn qualified_whole_backend_shutdown_drains_without_reviving_history_or_output() {
    let mut w = worker();
    ready(&mut w);
    let step = request(9980, start());
    w.apply(&step, 0).unwrap();
    assert!(w.reserved_bytes() > 0);
    w.backend_shutdown_drained().unwrap();
    assert_eq!(w.reserved_bytes(), 0);
    assert!(w.backend_is_stopped());
    assert!(!w.can_publish(session().generation_id));
    assert_eq!(w.snapshot().epoch.0, 1);
    assert_eq!(w.apply(&step, 0), Err(CoreError::State));
    assert_eq!(w.reconcile_epoch(2), Err(CoreError::State));
    w.backend_shutdown_drained().unwrap();
    assert_eq!(w.reserved_bytes(), 0);
}

#[test]
fn local_backend_ownership_survives_step_completion_withdrawal_and_recovery() {
    use std::collections::BTreeSet;
    let mut foreign = worker();
    foreign
        .apply(&request(500_000, prepare(lease())), 0)
        .unwrap();
    let foreign_hold = foreign.hold_backend(lease().lease_id, 0).unwrap();
    for withdrawal in ["release", "expiry", "revocation", "epoch"] {
        for pending in [false, true] {
            let mut w = worker();
            w.apply(&request(500_001, prepare(lease())), 0).unwrap();
            let hold = w.hold_backend(lease().lease_id, 0).unwrap();
            assert_eq!(
                w.hold_backend(lease().lease_id, 0).err(),
                Some(CoreError::Conflict)
            );
            let recovery = w.recovery();
            assert_eq!(
                recovery.retained_backends,
                BTreeSet::from([lease().lease_id])
            );
            let recovery: Recovery =
                serde_json::from_slice(&serde_json::to_vec(&recovery).unwrap()).unwrap();
            assert_eq!(
                Worker::recover(id(1), plan().cluster_id, 4_194_304, 2, recovery, MODE).err(),
                Some(CoreError::State)
            );
            w.apply(
                &request(
                    500_002,
                    Operation::Commit {
                        lease_id: lease().lease_id,
                    },
                ),
                0,
            )
            .unwrap();
            w.apply(
                &request(
                    500_003,
                    Operation::Begin {
                        session: Box::new(session()),
                    },
                ),
                0,
            )
            .unwrap();
            if pending {
                w.apply(&request(500_004, start()), 0).unwrap();
            }
            assert_eq!(w.backend_unloaded(&hold), Err(CoreError::State));
            match withdrawal {
                "release" => {
                    w.apply(
                        &request(
                            500_005,
                            Operation::Release {
                                lease_id: lease().lease_id,
                            },
                        ),
                        0,
                    )
                    .unwrap();
                }
                "expiry" => w.advance(15_000).unwrap(),
                "revocation" => w.retain_plans(|_| false).unwrap(),
                "epoch" => assert_eq!(w.reconcile_epoch(2), Ok(false)),
                _ => unreachable!(),
            }
            assert_eq!(w.reserved_bytes(), lease().reserved_bytes.0, "{withdrawal}");
            assert_eq!(
                w.snapshot().instances[&lease().lease_id],
                InstanceState::Releasing
            );
            assert!(!w.can_publish(session().generation_id));
            if pending {
                assert_eq!(w.backend_unloaded(&hold), Err(CoreError::State));
                w.drained(session().generation_id, header().invocation_id)
                    .unwrap();
            }
            // A completed chunk does not unload the instance's weights, KV or state.
            assert_eq!(w.reserved_bytes(), lease().reserved_bytes.0);
            drop(hold.clone());
            assert_eq!(w.backend_unloaded(&foreign_hold), Err(CoreError::Conflict));
            assert_eq!(w.reserved_bytes(), lease().reserved_bytes.0);
            w.backend_unloaded(&hold).unwrap();
            w.backend_unloaded(&hold).unwrap();
            assert_eq!(w.reserved_bytes(), 0);
            assert_eq!(
                w.snapshot().instances[&lease().lease_id],
                InstanceState::Released
            );
            assert!(w.recovery().retained_backends.is_empty());
            assert_eq!(w.reconcile_epoch(2), Ok(true));
            assert_eq!(w.backend_unloaded(&hold), Err(CoreError::NotFound));
        }
    }

    let mut reused = worker();
    reused
        .apply(&request(500_100, prepare(lease())), 0)
        .unwrap();
    let old = reused.hold_backend(lease().lease_id, 0).unwrap();
    reused.withdraw_lease(lease().lease_id).unwrap();
    reused.backend_unloaded(&old).unwrap();
    // Reclaim the ordinary bounded lease table, then intentionally reuse a UUID.
    for n in 0..MAX_RECORDS {
        let mut l = lease();
        l.lease_id = id(510_000 + n as u64);
        reused
            .apply(&request(520_000 + n as u64, prepare(l.clone())), 0)
            .unwrap();
        reused.withdraw_lease(l.lease_id).unwrap();
    }
    reused
        .apply(&request(530_000, prepare(lease())), 0)
        .unwrap();
    let successor = reused.hold_backend(lease().lease_id, 0).unwrap();
    reused.withdraw_lease(lease().lease_id).unwrap();
    assert_eq!(reused.backend_unloaded(&old), Err(CoreError::Conflict));
    assert_eq!(reused.reserved_bytes(), lease().reserved_bytes.0);
    reused.backend_unloaded(&successor).unwrap();
    assert_eq!(reused.reserved_bytes(), 0);

    let mut dropped = worker();
    dropped
        .apply(&request(540_000, prepare(lease())), 0)
        .unwrap();
    drop(dropped.hold_backend(lease().lease_id, 0).unwrap());
    dropped.withdraw_lease(lease().lease_id).unwrap();
    assert_eq!(dropped.reserved_bytes(), lease().reserved_bytes.0);
    dropped.backend_shutdown_drained().unwrap();
    assert_eq!(dropped.reserved_bytes(), 0);
    assert!(dropped.backend_is_stopped());

    let mut shutdown = Worker::new(id(1), plan().cluster_id, 8_388_608, 1, MODE).unwrap();
    shutdown
        .apply(&request(550_000, prepare(lease())), 0)
        .unwrap();
    let first = shutdown.hold_backend(lease().lease_id, 0).unwrap();
    let mut second_lease = lease();
    second_lease.lease_id = id(550_001);
    shutdown
        .apply(&request(550_002, prepare(second_lease.clone())), 0)
        .unwrap();
    let second = shutdown.hold_backend(second_lease.lease_id, 0).unwrap();
    shutdown
        .apply(
            &request(
                550_003,
                Operation::Commit {
                    lease_id: lease().lease_id,
                },
            ),
            0,
        )
        .unwrap();
    shutdown
        .apply(
            &request(
                550_004,
                Operation::Begin {
                    session: Box::new(session()),
                },
            ),
            0,
        )
        .unwrap();
    shutdown.apply(&request(550_005, start()), 0).unwrap();
    shutdown.backend_shutdown_drained().unwrap();
    assert_eq!(shutdown.reserved_bytes(), 0);
    assert!(shutdown.backend_is_stopped());
    assert!(shutdown.recovery().retained_backends.is_empty());
    shutdown.backend_unloaded(&first).unwrap();
    shutdown.backend_unloaded(&second).unwrap();
    assert!(
        Worker::recover(
            id(1),
            plan().cluster_id,
            8_388_608,
            2,
            shutdown.recovery(),
            MODE
        )
        .is_ok()
    );

    let mut cpu_plan = plan();
    cpu_plan.mode = PlacementPlanMode::WholeRequest;
    cpu_plan.stages.truncate(1);
    cpu_plan.stages[0].layer_end_exclusive = cpu_plan.model_layer_count;
    cpu_plan.stages[0].owns_final_norm_and_head = true;
    let mut cpu = worker();
    cpu.apply(
        &request(
            560_000,
            Operation::Prepare {
                plan: Box::new(cpu_plan),
                lease: Box::new(lease()),
            },
        ),
        0,
    )
    .unwrap();
    assert_eq!(
        cpu.hold_backend(lease().lease_id, 0).err(),
        Some(CoreError::State)
    );
    cpu.advance(15_000).unwrap();
    assert_eq!(cpu.reserved_bytes(), 0);
    let bytes = serde_json::to_vec(&cpu.recovery()).unwrap();
    let mut legacy: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(legacy.get("retained_backends").is_none());
    let decoded: Recovery = serde_json::from_value(legacy.clone()).unwrap();
    assert_eq!(serde_json::to_vec(&decoded).unwrap(), bytes);
    legacy["retained_backends"] = serde_json::json!([]);
    let decoded: Recovery = serde_json::from_value(legacy).unwrap();
    assert_eq!(serde_json::to_vec(&decoded).unwrap(), bytes);
}
fn complete() -> Operation {
    Operation::CompleteStep {
        generation_id: session().generation_id,
        invocation_id: header().invocation_id,
    }
}

#[test]
fn local_complete_boundaries_start_once_with_retained_native_ownership() {
    fn prepared(
        stage_index: usize,
        scratch: u64,
        rows: u32,
    ) -> (Worker, BackendHold, Vec<TensorHeader>) {
        let mut p = plan();
        p.stages[stage_index].scratch_budget_bytes = Decimal(scratch);
        let stage = p.stages[stage_index].clone();
        let last = p.stages.last().unwrap().index;
        let mut l = lease();
        l.owner_node_id = stage.node_id;
        l.reserved_bytes = Decimal(stage.reserved_bytes().unwrap());
        let mut w =
            Worker::new(l.owner_node_id, p.cluster_id, l.reserved_bytes.0, 1, MODE).unwrap();
        w.apply(
            &request(
                570_000,
                Operation::Prepare {
                    plan: Box::new(p),
                    lease: Box::new(l.clone()),
                },
            ),
            0,
        )
        .unwrap();
        let hold = w.hold_backend(l.lease_id, 0).unwrap();
        w.apply(
            &request(
                570_001,
                Operation::Commit {
                    lease_id: l.lease_id,
                },
            ),
            0,
        )
        .unwrap();
        let mut s = session();
        s.lease_id = l.lease_id;
        s.prompt_token_count = BoundedU32::new(rows).unwrap();
        w.apply(
            &request(
                570_002,
                Operation::Begin {
                    session: Box::new(s),
                },
            ),
            0,
        )
        .unwrap();
        let mut first: TensorHeader =
            decode(include_bytes!("../../../examples/tensor-header.json"), MODE).unwrap();
        first.destination_stage = stage.index;
        first.token_count = BoundedU32::new(rows).unwrap();
        if stage.owns_embedding {
            first.source_stage = last;
            first.dtype = TensorHeaderDtype::U32;
            first.tensor_role = TensorHeaderTensorRole::TokenIds;
            first.shape = vec![BoundedU32::new(rows).unwrap()];
            first.total_tensor_bytes = Decimal(u64::from(rows) * 4);
            first.chunk_bytes = first.total_tensor_bytes;
            return (w, hold, vec![first]);
        }
        first.shape[1] = BoundedU32::new(rows).unwrap();
        first.total_tensor_bytes = Decimal(u64::from(rows) * 4096 * 2);
        first.chunk_bytes = first.total_tensor_bytes;
        first.tensor_count = BoundedU32::new(2).unwrap();
        first.tensor_role = TensorHeaderTensorRole::Residual;
        let mut second = first.clone();
        second.tensor_index = BoundedU32::new(1).unwrap();
        second.tensor_role = TensorHeaderTensorRole::Hidden;
        (w, hold, vec![first, second])
    }
    let (mut w, hold, headers) = prepared(1, 1_048_576, 2);
    assert_eq!(w.check_backend_hold(&hold), Ok(lease().lease_id));
    assert!(!w.can_publish_step(headers[0].generation_id, headers[0].invocation_id));
    let before = serde_json::to_value(w.snapshot()).unwrap();
    for change in [
        "scope", "epoch", "sequence", "position", "phase", "tokens", "profile", "index", "count",
        "order", "offset", "shape", "dtype", "role", "size", "budget",
    ] {
        let mut bad = headers.clone();
        match change {
            "scope" => bad[1].generation_id = id(570_050),
            "epoch" => bad[1].authority_epoch = Decimal(2),
            "sequence" => bad[1].sequence = Decimal(1),
            "position" => bad[1].position_start = BoundedU32::new(1).unwrap(),
            "phase" => bad[1].phase = TensorHeaderPhase::Decode,
            "tokens" => bad[1].token_count = BoundedU32::new(1).unwrap(),
            "profile" => bad[1].boundary_profile = "another-boundary".into(),
            "index" => bad[1].tensor_index = BoundedU32::new(0).unwrap(),
            "count" => bad[1].tensor_count = BoundedU32::new(1).unwrap(),
            "order" => bad.swap(0, 1),
            "offset" => {
                bad[1].chunk_offset_bytes = Decimal(2);
                bad[1].chunk_bytes.0 -= 2;
            }
            "shape" => bad[1].shape.clear(),
            "dtype" => bad[1].dtype = TensorHeaderDtype::U32,
            "role" => {
                bad[1].tensor_role = TensorHeaderTensorRole::TokenIds;
                bad[1].dtype = TensorHeaderDtype::U32;
                bad[1].shape = vec![BoundedU32::new(2).unwrap()];
                bad[1].total_tensor_bytes = Decimal(8);
                bad[1].chunk_bytes = Decimal(8);
            }
            "size" => bad[1].total_tensor_bytes = Decimal(u64::MAX),
            "budget" => {
                for h in &mut bad {
                    h.shape[2] = BoundedU32::new(196_608).unwrap();
                    h.total_tensor_bytes = Decimal(786_432);
                    h.chunk_bytes = h.total_tensor_bytes;
                }
            }
            _ => unreachable!(),
        }
        assert!(w.start_local_chunk(&hold, &bad, 0).is_err(), "{change}");
        assert_eq!(
            serde_json::to_value(w.snapshot()).unwrap(),
            before,
            "{change}"
        );
        assert!(!w.can_publish_step(headers[0].generation_id, headers[0].invocation_id));
    }
    let (_, foreign, _) = prepared(1, 1_048_576, 2);
    assert_eq!(w.check_backend_hold(&foreign), Err(CoreError::Conflict));
    assert_eq!(
        w.start_local_chunk(&foreign, &headers, 0),
        Err(CoreError::Conflict)
    );
    // Peer/wire StartStep is still the original single-tensor operation.
    assert_eq!(
        w.apply(
            &request(
                570_100,
                Operation::StartStep {
                    header: Box::new(headers[0].clone()),
                    input_digest: byte_digest(b"synthetic")
                }
            ),
            0
        ),
        Err(CoreError::Invalid)
    );
    let mut next = headers.clone();
    for h in &mut next {
        h.invocation_id = id(570_102);
        h.sequence = Decimal(1);
        h.phase = TensorHeaderPhase::Decode;
        h.position_start = BoundedU32::new(2).unwrap();
        h.token_count = BoundedU32::new(1).unwrap();
        h.shape[1] = BoundedU32::new(1).unwrap();
        h.total_tensor_bytes = Decimal(8192);
        h.chunk_bytes = h.total_tensor_bytes;
    }
    assert_eq!(w.validate_data_input_next(&next[0]), Err(CoreError::State));
    w.start_local_chunk(&hold, &headers, 0).unwrap();
    assert!(w.can_publish_step(headers[0].generation_id, headers[0].invocation_id));
    assert!(!w.can_publish_step(headers[0].generation_id, id(570_101)));
    // While step 0 runs, a receive may open for step 1 only, as step 0 will leave the session;
    // step 1's data is admitted only after step 0 drains.
    assert_eq!(w.validate_data_input(&next[0]), Err(CoreError::State));
    w.validate_data_input_next(&next[0]).unwrap();
    for change in ["sequence", "position", "phase", "replay"] {
        let mut bad = next[0].clone();
        match change {
            "sequence" => bad.sequence = Decimal(2),
            "position" => bad.position_start = BoundedU32::new(1).unwrap(),
            "phase" => bad.phase = TensorHeaderPhase::Prefill,
            "replay" => bad.invocation_id = headers[0].invocation_id,
            _ => unreachable!(),
        }
        assert!(w.validate_data_input_next(&bad).is_err(), "{change}");
    }
    assert!(w.start_local_chunk(&hold, &headers, 0).is_err());
    w.drained(headers[0].generation_id, headers[0].invocation_id)
        .unwrap();
    assert!(!w.can_publish_step(headers[0].generation_id, headers[0].invocation_id));
    w.validate_data_input(&next[0]).unwrap();
    assert_eq!(w.reserved_bytes(), lease().reserved_bytes.0);
    assert!(
        w.start_local_chunk(&hold, &headers, 0).is_err(),
        "completed invocation replay"
    );
    w.start_local_chunk(&hold, &next, 0).unwrap();
    w.withdraw_lease(lease().lease_id).unwrap();
    assert!(!w.can_publish_step(next[0].generation_id, next[0].invocation_id));
    assert!(w.start_local_chunk(&hold, &next, 0).is_err());
    w.drained(next[0].generation_id, next[0].invocation_id)
        .unwrap();
    assert_eq!(w.reserved_bytes(), lease().reserved_bytes.0);
    w.backend_unloaded(&hold).unwrap();
    assert_eq!(w.reserved_bytes(), 0);

    // GLY-360: a decode step may carry drafts (at most MAX_DECODE_ROWS rows). The next step may
    // start anywhere after the multi-row step's first row, up to its end; after a one-row step
    // only at its end. The output bound is on positions (prompt 2, at most 8 output tokens).
    let (mut d, hold, headers) = prepared(1, 1_048_576, 2);
    let generation = headers[0].generation_id;
    d.start_local_chunk(&hold, &headers, 0).unwrap();
    d.drained(generation, headers[0].invocation_id).unwrap();
    let step = |sequence: u64, start: u32, rows: u32| {
        let mut step = headers.clone();
        for h in &mut step {
            h.invocation_id = id(570_200 + sequence);
            h.sequence = Decimal(sequence);
            h.phase = TensorHeaderPhase::Decode;
            h.position_start = BoundedU32::new(start).unwrap();
            h.token_count = BoundedU32::new(rows).unwrap();
            h.shape[1] = BoundedU32::new(rows).unwrap();
            h.total_tensor_bytes = Decimal(u64::from(rows) * 4096 * 2);
            h.chunk_bytes = h.total_tensor_bytes;
        }
        step
    };
    assert!(
        d.validate_data_input(&step(1, 2, 9)[0]).is_err(),
        "past the output bound"
    );
    let verify = step(1, 2, 4);
    d.start_local_chunk(&hold, &verify, 0).unwrap();
    // While it runs, the next receive may open at any start it may leave.
    for start in [3, 6] {
        d.validate_data_input_next(&step(2, start, 1)[0]).unwrap();
    }
    for start in [2, 7] {
        assert!(
            d.validate_data_input_next(&step(2, start, 1)[0]).is_err(),
            "{start}"
        );
    }
    d.drained(generation, verify[0].invocation_id).unwrap();
    for start in [2, 7] {
        assert!(
            d.validate_data_input(&step(2, start, 1)[0]).is_err(),
            "{start}"
        );
    }
    assert!(
        d.validate_data_input(&step(2, 4, 7)[0]).is_err(),
        "past the output bound"
    );
    d.validate_data_input(&step(2, 4, 6)[0]).unwrap();
    // Two of its three drafts were kept: the next step starts at 4.
    let next = step(2, 4, 1);
    d.start_local_chunk(&hold, &next, 0).unwrap();
    d.drained(generation, next[0].invocation_id).unwrap();
    // After a one-row step nothing is taken back.
    assert!(d.validate_data_input(&step(3, 4, 1)[0]).is_err());
    d.validate_data_input(&step(3, 5, 1)[0]).unwrap();
    d.withdraw_lease(lease().lease_id).unwrap();
    d.backend_unloaded(&hold).unwrap();

    // Incoming authenticated token IDs retain the existing last-to-first direction.
    let (mut first, hold, headers) = prepared(0, 1_048_576, 1);
    first.validate_data_input(&headers[0]).unwrap();
    let before = serde_json::to_value(first.snapshot()).unwrap();
    let mut wrong = headers.clone();
    wrong[0].source_stage = BoundedU32::new(0).unwrap();
    assert_eq!(
        first.start_local_chunk(&hold, &wrong, 0),
        Err(CoreError::Invalid)
    );
    assert_eq!(serde_json::to_value(first.snapshot()).unwrap(), before);
    first.start_local_chunk(&hold, &headers, 0).unwrap();
    assert!(first.can_publish_step(headers[0].generation_id, headers[0].invocation_id));
    first.withdraw_lease(lease().lease_id).unwrap();
    first
        .drained(headers[0].generation_id, headers[0].invocation_id)
        .unwrap();
    assert!(first.reserved_bytes() > 0);
    first.backend_unloaded(&hold).unwrap();
    assert_eq!(first.reserved_bytes(), 0);

    // Authentic reassembled tensors can exceed a wire frame without falsifying its limits.
    let (mut large, hold, mut headers) = prepared(1, 32 * 1024 * 1024, 16);
    for h in &mut headers {
        h.shape[2] = BoundedU32::new(524_288).unwrap();
        h.total_tensor_bytes = Decimal(16 * 1024 * 1024);
        h.chunk_bytes = Decimal(8 * 1024 * 1024);
        h.end_of_tensor = false;
    }
    large.start_local_chunk(&hold, &headers, 0).unwrap();
    large.withdraw_lease(lease().lease_id).unwrap();
    large
        .drained(headers[0].generation_id, headers[0].invocation_id)
        .unwrap();
    assert!(large.reserved_bytes() > 0);
    large.backend_unloaded(&hold).unwrap();
    assert_eq!(large.reserved_bytes(), 0);

    for expiry in [false, true] {
        let (mut closed, hold, headers) = prepared(1, 1_048_576, 2);
        if !expiry {
            closed.withdraw_lease(lease().lease_id).unwrap();
        }
        assert!(
            closed
                .start_local_chunk(&hold, &headers, if expiry { 15_000 } else { 0 })
                .is_err()
        );
        assert_eq!(closed.reserved_bytes(), lease().reserved_bytes.0);
        assert!(!closed.can_publish_step(headers[0].generation_id, headers[0].invocation_id));
        closed.backend_unloaded(&hold).unwrap();
        assert_eq!(closed.reserved_bytes(), 0);
    }
}
#[test]
fn data_abort_is_scoped_idempotent_and_keeps_pending_execution_until_drain() {
    for pending in [false, true] {
        let mut w = Worker::new(id(1), plan().cluster_id, 8_388_608, 1, MODE).unwrap();
        ready(&mut w);
        let mut other_lease = lease();
        other_lease.lease_id = id(9905);
        w.apply(&request(9906, prepare(other_lease.clone())), 0)
            .unwrap();
        w.apply(
            &request(
                9907,
                Operation::Commit {
                    lease_id: other_lease.lease_id,
                },
            ),
            0,
        )
        .unwrap();
        let mut other = session();
        other.session_id = id(9900);
        other.generation_id = id(9901);
        other.lease_id = other_lease.lease_id;
        w.apply(
            &request(
                9902,
                Operation::Begin {
                    session: Box::new(other.clone()),
                },
            ),
            0,
        )
        .unwrap();
        let h = header();
        for mismatch in ["session", "plan", "epoch"] {
            let mut stale = h.clone();
            match mismatch {
                "session" => stale.session_id = other.session_id,
                "plan" => stale.plan_digest = byte_digest(b"another admitted plan"),
                "epoch" => stale.authority_epoch = Decimal(2),
                _ => unreachable!(),
            }
            assert_eq!(w.abort_data(&stale), Err(CoreError::Conflict));
            assert_eq!(
                w.snapshot().generations[&h.generation_id],
                SessionState::Admitted
            );
        }
        if pending {
            w.apply(&request(9903, start()), 0).unwrap();
        }
        let reserved = w.reserved_bytes();
        w.abort_data(&h).unwrap();
        w.abort_data(&h).unwrap();
        assert_eq!(
            w.snapshot().generations[&h.generation_id],
            if pending {
                SessionState::Cancelling
            } else {
                SessionState::Failed
            }
        );
        assert_eq!(
            w.snapshot().generations[&other.generation_id],
            SessionState::Admitted
        );
        assert_eq!(w.reserved_bytes(), reserved);
        assert!(!w.can_publish(h.generation_id));
        assert!(w.apply(&request(9904, start()), 0).is_err());
        w.withdraw_lease(lease().lease_id).unwrap();
        if pending {
            assert_eq!(w.reserved_bytes(), reserved);
            w.drained(h.generation_id, h.invocation_id).unwrap();
        }
        assert_eq!(w.reserved_bytes(), other_lease.reserved_bytes.0);
        assert_eq!(
            w.snapshot().generations[&h.generation_id],
            SessionState::Failed
        );
        assert_eq!(
            w.snapshot().generations[&other.generation_id],
            SessionState::Admitted
        );
        w.withdraw_lease(other_lease.lease_id).unwrap();
        assert_eq!(w.reserved_bytes(), 0);
    }
}
#[test]
fn reservations_are_atomic_replays_do_not_extend_leases_and_partial_prepare_rolls_back() {
    let mut a = worker();
    let r = request(1, prepare(lease()));
    assert_eq!(a.apply(&r, 0), Ok(false));
    assert_eq!(a.apply(&r, 14_999), Ok(true));
    let mut other = lease();
    other.lease_id = id(900);
    assert_eq!(
        a.apply(&request(2, prepare(other)), 14_999),
        Err(CoreError::Capacity)
    );
    let b = Worker::new(id(2), plan().cluster_id, 1, 1, MODE).unwrap();
    let mut b = b;
    let mut second = lease();
    second.owner_node_id = id(2);
    assert_eq!(
        b.apply(&request(3, prepare(second)), 0),
        Err(CoreError::Capacity)
    );
    a.apply(
        &request(
            4,
            Operation::Release {
                lease_id: lease().lease_id,
            },
        ),
        14_999,
    )
    .unwrap();
    assert_eq!(a.reserved_bytes() + b.reserved_bytes(), 0);
    let mut c = worker();
    c.apply(&r, 0).unwrap();
    c.advance(15_000).unwrap();
    assert_eq!(c.reserved_bytes(), 0);
    assert_eq!(c.apply(&r, 15_000), Ok(true));
    assert_eq!(c.reserved_bytes(), 0);
    assert_eq!(
        c.apply(
            &request(
                5,
                Operation::Commit {
                    lease_id: lease().lease_id
                }
            ),
            15_000
        ),
        Err(CoreError::Expired)
    );
    assert_eq!(c.advance(14_999), Err(CoreError::Clock));
    let mut overflow = worker();
    assert_eq!(overflow.apply(&r, u64::MAX), Err(CoreError::Clock));
}
#[test]
fn request_identity_epoch_and_content_are_bound() {
    let mut w = worker();
    let r = request(1, prepare(lease()));
    w.apply(&r, 0).unwrap();
    let mut altered = request(
        1,
        Operation::Release {
            lease_id: lease().lease_id,
        },
    );
    assert_eq!(w.apply(&altered, 0), Err(CoreError::Conflict));
    altered.request_hash = r.request_hash.clone();
    assert_eq!(w.apply(&altered, 0), Err(CoreError::Invalid));
    let mut stale = request(
        2,
        Operation::Commit {
            lease_id: lease().lease_id,
        },
    );
    stale = ControlRequest::new(
        stale.request_id,
        2,
        stale.request_sequence.0,
        stale.operation,
    )
    .unwrap();
    assert_eq!(w.apply(&stale, 0), Err(CoreError::StaleEpoch));
    assert_eq!(w.reserved_bytes(), 4_194_304);
}
#[test]
fn cancelling_inflight_work_retains_memory_until_acknowledged_and_cannot_revive() {
    for expiry in [false, true] {
        let mut w = worker();
        ready(&mut w);
        w.apply(&request(10, start()), 0).unwrap();
        let now = if expiry { 15_000 } else { 0 };
        if expiry {
            w.advance(now).unwrap()
        } else {
            w.apply(
                &request(
                    11,
                    Operation::Release {
                        lease_id: lease().lease_id,
                    },
                ),
                now,
            )
            .unwrap();
        }
        assert_eq!(w.reserved_bytes(), 4_194_304);
        assert_eq!(
            w.snapshot().generations[&session().generation_id],
            SessionState::Cancelling
        );
        assert_eq!(w.apply(&request(12, start()), now), Err(CoreError::State));
        w.apply(&request(13, complete()), now).unwrap();
        assert_eq!(w.reserved_bytes(), 0);
        let state = if expiry {
            SessionState::Failed
        } else {
            SessionState::Cancelled
        };
        assert_eq!(w.snapshot().generations[&session().generation_id], state);
        assert_eq!(
            w.apply(
                &request(
                    14,
                    Operation::Begin {
                        session: Box::new(session())
                    }
                ),
                now
            ),
            Err(CoreError::State)
        );
        w.apply(
            &request(
                15,
                Operation::Cancel {
                    generation_id: session().generation_id,
                },
            ),
            now,
        )
        .unwrap();
        assert_eq!(w.snapshot().generations[&session().generation_id], state);
    }
}
#[test]
fn completion_and_restart_preserve_terminal_generations_without_kv_state() {
    let mut w = worker();
    ready(&mut w);
    w.apply(&request(10, start()), 0).unwrap();
    w.apply(&request(11, complete()), 0).unwrap();
    w.apply(
        &request(
            12,
            Operation::Finish {
                generation_id: session().generation_id,
            },
        ),
        0,
    )
    .unwrap();
    let recovery = w.recovery();
    assert!(
        Worker::recover(
            id(1),
            plan().cluster_id,
            4_194_304,
            1,
            recovery.clone(),
            MODE
        )
        .is_err()
    );
    let mut restored =
        Worker::recover(id(1), plan().cluster_id, 4_194_304, 2, recovery, MODE).unwrap();
    assert_eq!(restored.reserved_bytes(), 0);
    assert_eq!(
        restored.snapshot().generations[&session().generation_id],
        SessionState::Completed
    );
    assert_eq!(
        restored.apply(&request(20, prepare(lease())), 0),
        Err(CoreError::StaleEpoch)
    );
    let mut s = session();
    s.authority_epoch = Decimal(2);
    let r = ControlRequest::new(
        id(21),
        2,
        1,
        Operation::Begin {
            session: Box::new(s),
        },
    )
    .unwrap();
    assert_eq!(restored.apply(&r, 0), Err(CoreError::State));
    let mut running = worker();
    ready(&mut running);
    let recovered = Worker::recover(
        id(1),
        plan().cluster_id,
        4_194_304,
        2,
        running.recovery(),
        MODE,
    )
    .unwrap();
    assert_eq!(
        recovered.snapshot().generations[&session().generation_id],
        SessionState::Failed
    );
}
#[test]
fn telemetry_cannot_refresh_itself_by_replay() {
    let n: NodeCapabilities = decode(
        include_bytes!("../../../examples/node-capabilities.json"),
        MODE,
    )
    .unwrap();
    let mut r = Registry::new(FixtureMode::Simulation);
    r.observe(n.clone(), 0).unwrap();
    assert!(r.eligible(n.node_id, 0, 1000).is_ok());
    r.observe(n.clone(), 900).unwrap();
    assert_eq!(
        r.eligible(n.node_id, 1000, 1000).unwrap_err(),
        CoreError::StaleCapability
    );
    let mut changed = n.clone();
    changed.state = NodeCapabilitiesState::Draining;
    assert_eq!(r.observe(changed, 1000), Err(CoreError::Conflict));
    let mut invalid = n;
    invalid.kind = "wrong".to_owned();
    invalid.snapshot_id = id(900);
    assert_eq!(r.observe(invalid, 1000), Err(CoreError::Invalid));
}

#[test]
fn admission_limits_boundary_memory_and_restart_identity_are_enforced() {
    let mut w = worker();
    ready(&mut w);
    let mut wrong = header();
    wrong.shape = vec![BoundedU32::new(1_048_576).unwrap()];
    wrong.total_tensor_bytes = Decimal(2_097_152);
    wrong.chunk_bytes = wrong.total_tensor_bytes;
    let op = Operation::StartStep {
        header: Box::new(wrong),
        input_digest: byte_digest(b"fixture"),
    };
    assert_eq!(w.apply(&request(10, op), 0), Err(CoreError::Invalid));
    let mut s = session();
    s.session_id = id(900);
    s.generation_id = id(901);
    assert_eq!(
        w.apply(
            &request(
                11,
                Operation::Begin {
                    session: Box::new(s)
                }
            ),
            0
        ),
        Err(CoreError::Capacity)
    );
    w.apply(
        &request(
            12,
            Operation::Cancel {
                generation_id: session().generation_id,
            },
        ),
        0,
    )
    .unwrap();
    assert_eq!(w.apply(&request(13, start()), 0), Err(CoreError::State));
    assert!(Worker::recover(id(2), plan().cluster_id, 4_194_304, 2, w.recovery(), MODE).is_err());
    let node: NodeCapabilities = decode(
        include_bytes!("../../../examples/node-capabilities.json"),
        MODE,
    )
    .unwrap();
    assert_eq!(
        Registry::default().observe(node.clone(), 0),
        Err(CoreError::Invalid)
    );
    let mut r = Registry::new(MODE);
    for (i, state) in [
        NodeCapabilitiesState::Ready,
        NodeCapabilitiesState::Draining,
        NodeCapabilitiesState::Unavailable,
        NodeCapabilitiesState::Ready,
    ]
    .into_iter()
    .enumerate()
    {
        let mut report = node.clone();
        report.snapshot_id = id(i as u64 + 900);
        report.state = state;
        report.observed_at = format!("2026-10-12T00:00:{i:02}Z");
        r.observe(report, i as u64).unwrap();
        assert_eq!(
            r.eligible(node.node_id, i as u64, 1000).is_ok(),
            state == NodeCapabilitiesState::Ready
        );
    }
}

#[test]
fn sustained_generations_reclaim_history_without_epoch_rollover() {
    let mut w = worker();
    let epoch = 1;
    let mut request_number = 10_000;
    for generation in 0..1000_u64 {
        let mut p = plan();
        p.authority_epoch = Decimal(epoch);
        let mut l = lease();
        l.authority_epoch = Decimal(epoch);
        l.lease_id = id(30_000 + generation);
        let mut session = session();
        session.authority_epoch = Decimal(epoch);
        session.lease_id = l.lease_id;
        session.generation_id = id(40_000 + generation);
        session.session_id = id(50_000 + generation);
        let output_tokens = if generation == 0 { 512 } else { 10 };
        let prompt_tokens = if generation == 0 { 1000 } else { 2 };
        session.prompt_token_count = BoundedU32::new(prompt_tokens).unwrap();
        session.max_output_tokens = BoundedU32::new(output_tokens).unwrap();
        let mut apply = |w: &mut Worker, op| {
            request_number += 1;
            let req = ControlRequest::new(id(request_number), epoch, request_number, op).unwrap();
            assert_eq!(w.apply(&req, 0), Ok(false));
            assert_eq!(w.apply(&req, 0), Ok(true));
        };
        apply(
            &mut w,
            Operation::Prepare {
                plan: Box::new(p),
                lease: Box::new(l.clone()),
            },
        );
        apply(
            &mut w,
            Operation::Commit {
                lease_id: l.lease_id,
            },
        );
        apply(
            &mut w,
            Operation::Begin {
                session: Box::new(session.clone()),
            },
        );
        for token in 0..=output_tokens {
            let mut h = header();
            h.authority_epoch = Decimal(epoch);
            h.lease_id = l.lease_id;
            h.generation_id = session.generation_id;
            h.session_id = session.session_id;
            h.invocation_id = id(60_000 + generation * 11 + u64::from(token));
            h.sequence = Decimal(u64::from(token));
            h.position_start = BoundedU32::new(if token == 0 {
                0
            } else {
                prompt_tokens + token - 1
            })
            .unwrap();
            h.token_count = BoundedU32::new(if token == 0 { prompt_tokens } else { 1 }).unwrap();
            h.shape[1] = h.token_count;
            h.shape[2] = BoundedU32::new(2).unwrap();
            h.total_tensor_bytes = Decimal(u64::from(h.token_count.get()) * 4);
            h.chunk_bytes = h.total_tensor_bytes;
            if token > 0 {
                h.phase = TensorHeaderPhase::Decode;
                h.token_count = BoundedU32::new(1).unwrap();
                h.shape[1] = BoundedU32::new(1).unwrap();
            }
            let invocation_id = h.invocation_id;
            apply(
                &mut w,
                Operation::StartStep {
                    header: Box::new(h),
                    input_digest: byte_digest(b"fixture"),
                },
            );
            apply(
                &mut w,
                Operation::CompleteStep {
                    generation_id: session.generation_id,
                    invocation_id,
                },
            );
        }
        apply(
            &mut w,
            Operation::Finish {
                generation_id: session.generation_id,
            },
        );
        apply(
            &mut w,
            Operation::Release {
                lease_id: l.lease_id,
            },
        );
        assert_eq!(w.reserved_bytes(), 0);
    }
    let recovery = w.recovery();
    assert!(
        Worker::recover(
            id(1),
            plan().cluster_id,
            4_194_304,
            epoch + 1,
            recovery,
            MODE
        )
        .is_ok()
    );
}

#[test]
fn cleanup_works_at_request_capacity_and_drain_cannot_admit_old_work() {
    let mut w = worker();
    ready(&mut w);
    for n in 0..starlings_core::MAX_REQUESTS - 3 {
        let r = request(
            20_000 + n as u64,
            Operation::Commit {
                lease_id: lease().lease_id,
            },
        );
        w.apply(&r, 0).unwrap();
    }
    let cleanup = request(
        30_000,
        Operation::Release {
            lease_id: lease().lease_id,
        },
    );
    assert_eq!(w.apply(&cleanup, 0), Ok(false));
    assert_eq!(w.apply(&cleanup, 0), Ok(true));
    let mut first = None;
    for n in 0..MAX_REQUESTS + 1 {
        let next = request(
            40_000 + n as u64,
            Operation::Release {
                lease_id: lease().lease_id,
            },
        );
        if first.is_none() {
            first = Some(next.clone());
        }
        assert_eq!(w.apply(&next, 0), Ok(false));
    }
    assert_eq!(w.apply(&first.unwrap(), 0), Err(CoreError::ReplayExpired));
    assert_eq!(w.apply(&cleanup, 0), Err(CoreError::ReplayExpired));
    assert_eq!(w.reserved_bytes(), 0);
    assert_eq!(w.reconcile_epoch(2), Ok(true));
    assert_eq!(
        w.apply(&request(1000, prepare(lease())), 0),
        Err(CoreError::StaleEpoch)
    );
    let mut pending = worker();
    ready(&mut pending);
    pending.apply(&request(10, start()), 0).unwrap();
    assert_eq!(pending.reconcile_epoch(2), Ok(false));
    let mut next = lease();
    next.lease_id = id(9999);
    assert_eq!(
        pending.apply(&request(11, prepare(next)), 0),
        Err(CoreError::StaleEpoch)
    );
    pending
        .drained(session().generation_id, header().invocation_id)
        .unwrap();
    assert_eq!(pending.reconcile_epoch(2), Ok(true));
}

#[test]
fn concurrent_long_sessions_do_not_compete_for_a_lifetime_request_budget() {
    let mut w = Worker::new(id(1), plan().cluster_id, 2 * 4_194_304, 1, MODE).unwrap();
    for n in 1..=2 {
        let mut l = lease();
        l.lease_id = id(80_000 + n);
        w.apply(&request(90_000 + n, prepare(l.clone())), 0)
            .unwrap();
        w.apply(
            &request(
                91_000 + n,
                Operation::Commit {
                    lease_id: l.lease_id,
                },
            ),
            0,
        )
        .unwrap();
        let mut s = session();
        s.lease_id = l.lease_id;
        s.generation_id = id(82_000 + n);
        s.session_id = id(83_000 + n);
        s.prompt_token_count = BoundedU32::new(1000).unwrap();
        s.max_output_tokens = BoundedU32::new(1000).unwrap();
        w.apply(
            &request(
                92_000 + n,
                Operation::Begin {
                    session: Box::new(s),
                },
            ),
            0,
        )
        .unwrap();
    }
    assert_eq!(w.snapshot().generations.len(), 2);
    assert_eq!(w.reserved_bytes(), 2 * 4_194_304);
}

#[test]
fn whole_request_has_one_invocation_and_retains_capacity_until_trusted_drain() {
    let mut p = plan();
    p.mode = PlacementPlanMode::WholeRequest;
    p.stages.truncate(1);
    p.stages[0].layer_end_exclusive = p.model_layer_count;
    p.stages[0].owns_final_norm_and_head = true;
    let mut s = session();
    s.prompt_token_count = BoundedU32::new(1000).unwrap();
    s.max_output_tokens = BoundedU32::new(500).unwrap();
    let generation = s.generation_id;
    let invocation = id(991);
    let chat = ChatRequest {
        model: p.manifest_digest.to_string(),
        messages: vec![ChatMessage {
            role: ChatMessageRole::User,
            content: "synthetic lifecycle input".into(),
        }],
        max_tokens: s.max_output_tokens,
        temperature: Some(s.sampling.temperature),
        top_p: Some(s.sampling.top_p),
        seed: Some(s.sampling.seed),
        stream: Some(true),
    };
    for outcome in ["success", "cancel", "failure", "expiry"] {
        let cancel = outcome == "cancel";
        let mut w = worker();
        w.apply(
            &request(
                8000,
                Operation::Prepare {
                    plan: Box::new(p.clone()),
                    lease: Box::new(lease()),
                },
            ),
            0,
        )
        .unwrap();
        w.apply(
            &request(
                8001,
                Operation::Commit {
                    lease_id: s.lease_id,
                },
            ),
            0,
        )
        .unwrap();
        let mut wrong = chat.clone();
        wrong.model = "unapproved-alias".into();
        assert_eq!(
            w.apply(
                &request(
                    8002,
                    Operation::StartRequest {
                        session: Box::new(s.clone()),
                        invocation_id: invocation,
                        request: Box::new(wrong)
                    }
                ),
                0
            ),
            Err(CoreError::Invalid)
        );
        let start = request(
            8003,
            Operation::StartRequest {
                session: Box::new(s.clone()),
                invocation_id: invocation,
                request: Box::new(chat.clone()),
            },
        );
        let wire = serde_json::to_vec(&start).unwrap();
        let start = ControlRequest::decode(&wire, MODE).unwrap();
        assert_eq!(w.apply(&start, 0), Ok(false));
        assert_eq!(w.apply(&start, 0), Ok(true));
        assert!(w.can_publish(generation));
        assert!(
            w.apply(
                &request(
                    8004,
                    Operation::Finish {
                        generation_id: generation
                    }
                ),
                0
            )
            .is_err()
        );
        assert!(w.drained(generation, id(992)).is_err());
        if cancel {
            w.apply(
                &request(
                    8005,
                    Operation::Cancel {
                        generation_id: generation,
                    },
                ),
                0,
            )
            .unwrap();
            w.apply(
                &request(
                    8006,
                    Operation::Release {
                        lease_id: s.lease_id,
                    },
                ),
                0,
            )
            .unwrap();
            assert!(!w.can_publish(generation));
            assert!(w.reserved_bytes() > 0);
            assert_eq!(
                w.snapshot().generations[&generation],
                SessionState::Cancelling
            );
        }
        if outcome == "expiry" {
            w.advance(15_000).unwrap();
            assert!(!w.can_publish(generation));
            assert!(w.reserved_bytes() > 0);
        }
        if outcome == "failure" {
            w.failed_and_drained(generation, invocation).unwrap();
        } else {
            w.drained(generation, invocation).unwrap();
        }
        assert_eq!(
            w.snapshot().generations[&generation],
            match outcome {
                "cancel" => SessionState::Cancelled,
                "failure" | "expiry" => SessionState::Failed,
                _ => SessionState::Completed,
            }
        );
        if cancel || outcome == "expiry" {
            assert_eq!(w.reserved_bytes(), 0);
        }
        assert_eq!(w.can_publish(generation), outcome == "success");
        assert!(w.drained(generation, invocation).is_err());
        let mut mismatch = chat.clone();
        mismatch.max_tokens = BoundedU32::new(501).unwrap();
        assert!(
            request(
                8007,
                Operation::StartRequest {
                    session: Box::new(s.clone()),
                    invocation_id: invocation,
                    request: Box::new(mismatch)
                }
            )
            .validate(MODE)
            .is_err()
        );
    }
}
