use super::*;
use crate::{
    ListenerBudget, PeerListener, RegistryClient, RegistryRequest,
    enrollment::tests::{enrolled, setup},
};
use std::{
    sync::atomic::{AtomicBool, Ordering},
    thread,
};

fn limits() -> ProbeLimits {
    ProbeLimits {
        version: Version,
        payload_bytes: 16 * 1024,
        ping_samples: 3,
        inter_ping_gap_ms: 0,
        minimum_interval_ms: 30_000,
        token_profile: None,
    }
}
fn admitted_wire(
    route: &ProbeRoute,
    worker: &Identity,
    invitation: &Invitation,
) -> transport::Wire {
    let deadline = Instant::now() + BATCH_TIME;
    let socket =
        crate::network::connect(route.destination, route.source, Duration::from_secs(2)).unwrap();
    let socket =
        crate::reconnect::client(socket, worker, invitation.coordinator.node, deadline).unwrap();
    let trust = Trust {
        ca: invitation.ca.clone(),
        cluster: worker.cluster(),
        members: vec![invitation.coordinator.clone()],
        access: None,
    };
    transport::client_until(
        socket,
        &trust,
        &invitation.coordinator,
        Some(worker),
        ALPN,
        deadline,
    )
    .unwrap()
}
#[test]
fn limits_cooldown_and_direction_statistics_are_bounded() {
    let mut cap = limits();
    cap.validate().unwrap();
    for size in [0, 4097, 256 * 1024 + 1] {
        cap.payload_bytes = size;
        assert!(cap.validate().is_err());
    }
    cap = limits();
    cap.ping_samples = 9;
    assert!(cap.validate().is_err());
    cap = limits();
    cap.minimum_interval_ms = 29_999;
    assert!(cap.validate().is_err());
    cap = limits();
    cap.inter_ping_gap_ms = 101;
    assert!(cap.validate().is_err());
    let route = ProbeRoute {
        version: Version,
        destination: "8.8.8.8:9741".parse().unwrap(),
        source: None,
        limits: limits(),
    };
    assert!(route.validate().is_err());
    let unknown = br#"{"version":1,"payload_bytes":16384,"ping_samples":3,"inter_ping_gap_ms":0,"minimum_interval_ms":30000,"unbounded":true}"#;
    assert!(crate::decode::<ProbeLimits>(unknown).is_err());
    let node: NodeId = "00000000-0000-4000-8000-000000000001".parse().unwrap();
    let budget = ProbeBudget::new(limits()).unwrap();
    budget.claim(node).unwrap();
    assert!(matches!(budget.claim(node), Err(Error::Capacity)));
    for _ in 0..63 {
        budget
            .claim(Uuid::new_v4().to_string().parse().unwrap())
            .unwrap();
    }
    assert!(matches!(
        budget.claim(Uuid::new_v4().to_string().parse().unwrap()),
        Err(Error::Capacity)
    ));
    assert_eq!(statistics(&[10, 20, 30]).unwrap(), (20, Decimal(66)));
    assert!(statistics(&[]).is_err());
    let peer: NodeId = "00000000-0000-4000-8000-000000000002".parse().unwrap();
    let fast = measurement(node, peer, 4096, Duration::from_millis(1), "fixture").unwrap();
    let congested = measurement(peer, node, 4096, Duration::from_millis(4), "fixture").unwrap();
    assert_eq!(
        fast.acknowledged_bytes_per_second.0,
        4 * congested.acknowledged_bytes_per_second.0
    );
    assert_ne!(
        fast.source_node, congested.source_node,
        "directions remain distinct"
    );
}

#[test]
fn authenticated_real_transfers_do_not_consume_the_persistent_heartbeat_slot() {
    let (_temp, path, mut authority, files) = setup();
    let worker = enrolled(&mut authority, &files, Role::Worker, "probe-worker");
    worker.save(&files, "probe-identity").unwrap();
    let heartbeat_budget = ListenerBudget::default();
    let heartbeat =
        PeerListener::bind("127.0.0.1:0".parse().unwrap(), heartbeat_budget.clone()).unwrap();
    authority
        .invite(
            &files,
            "registry-pin",
            heartbeat.local_addr().unwrap(),
            Role::Worker,
        )
        .unwrap();
    authority.start_registry().unwrap();
    let profile = path.parent().unwrap().join("probe-profile");
    crate::PrivateDir::ensure(&profile).unwrap();
    let mut registry = RegistryClient::open(
        &profile,
        Identity::load(&files, "probe-identity").unwrap(),
        Invitation::load(&files, "registry-pin").unwrap(),
    )
    .unwrap();
    registry.enable_persistent();
    let probe_budget = ListenerBudget::default();
    let listener =
        PeerListener::bind("127.0.0.1:0".parse().unwrap(), probe_budget.clone()).unwrap();
    let route = ProbeRoute {
        version: Version,
        destination: listener.local_addr().unwrap(),
        source: Some("127.0.0.1:0".parse().unwrap()),
        limits: limits(),
    };
    let server_budget = ProbeBudget::new(limits()).unwrap();
    let client_budget = ProbeBudget::new(limits()).unwrap();
    let invitation = Invitation::load(&files, "registry-pin").unwrap();
    let stopped = AtomicBool::new(false);
    thread::scope(|scope| {
        scope.spawn(|| {
            let until = Instant::now() + Duration::from_secs(3);
            let incoming = loop {
                if let Some(incoming) = heartbeat.try_accept_registry(&authority).unwrap() {
                    break incoming;
                }
                assert!(Instant::now() < until);
                thread::sleep(Duration::from_millis(1));
            };
            let RegistryRequest::Heartbeat(pending) = incoming
                .registry(authority.enrollment_reader().unwrap())
                .unwrap()
            else {
                panic!("heartbeat required")
            };
            let mut pending = *pending;
            while !stopped.load(Ordering::Acquire) {
                match pending.respond_next() {
                    Ok(Some(next)) => pending = next,
                    _ => break,
                }
            }
        });
        registry.heartbeat(None).unwrap();
        registry.heartbeat(None).unwrap();
        assert_eq!(heartbeat_budget.usage().unwrap().connections, 1);
        let server = scope.spawn(|| {
            let until = Instant::now() + Duration::from_secs(3);
            loop {
                if let Some(incoming) = listener.try_accept_registry(&authority).unwrap() {
                    break incoming
                        .probe(authority.enrollment_reader().unwrap(), &server_budget)
                        .unwrap();
                }
                assert!(Instant::now() < until);
                thread::sleep(Duration::from_millis(1));
            }
        });
        let mut observation = measure(
            &route,
            &worker,
            &invitation,
            registry.fence(),
            &client_budget,
        )
        .unwrap();
        let server_report = server.join().unwrap();
        let server_report = &server_report.report;
        let report = observation.current(&registry.fence()).unwrap();
        let inputs = observation
            .view(&registry.fence())
            .unwrap()
            .placement_inputs()
            .unwrap();
        assert_eq!(inputs[0].source, worker.node());
        assert_eq!(inputs[1].destination, worker.node());
        let mut invalid = report.clone();
        invalid.id = Uuid::nil();
        assert!(
            ProbeView {
                report: &invalid,
                age_ms: 0
            }
            .placement_inputs()
            .is_err()
        );
        assert_eq!(report.transfers[0].source_node, worker.node());
        assert_eq!(report.transfers[1].destination_node, worker.node());
        assert_eq!(report.rtt_samples_us.as_ref().unwrap().len(), 3);
        assert!(
            server_report.rtt_mean_us.is_none(),
            "server does not invent unmeasured RTTs"
        );
        assert_eq!(
            report.total_application_bytes,
            server_report.total_application_bytes
        );
        assert!(report.total_application_bytes.0 > report.application_payload_bytes.0);
        assert!(
            report
                .transfers
                .iter()
                .all(|m| m.acknowledged_bytes_per_second.0 > 0)
        );
        assert_eq!(probe_budget.usage().unwrap().connections, 0);
        registry.heartbeat(None).unwrap();
        // CI-only protocol coverage, sharing the existing enrolled fixture. No model,
        // native qualification or physical-link measurement is performed.
        for keepalive_ms in [None, Some(5)] {
            let mut cap = limits();
            cap.ping_samples = 5;
            cap.inter_ping_gap_ms = if keepalive_ms.is_some() { 100 } else { 20 };
            cap.token_profile = Some(TokenProfile {
                gap_basis: GapBasis::AfterReply,
                keepalive_ms,
            });
            let server_budget = ProbeBudget::new(cap.clone()).unwrap();
            let client_budget = ProbeBudget::new(cap.clone()).unwrap();
            thread::scope(|attempt| {
                let server = attempt.spawn(|| {
                    let until = Instant::now() + Duration::from_secs(3);
                    loop {
                        if let Some(incoming) = listener.try_accept_registry(&authority).unwrap() {
                            break incoming
                                .probe(authority.enrollment_reader().unwrap(), &server_budget)
                                .unwrap();
                        }
                        assert!(Instant::now() < until);
                        thread::sleep(Duration::from_millis(1));
                    }
                });
                let sample = measure(
                    &ProbeRoute {
                        limits: cap,
                        ..route.clone()
                    },
                    &worker,
                    &invitation,
                    registry.fence(),
                    &client_budget,
                )
                .unwrap();
                let received = server.join().unwrap();
                let client = sample.report.token_profile.as_ref().unwrap();
                let server = received.report.token_profile.as_ref().unwrap();
                assert_eq!(client.completed_samples, 5);
                assert_eq!(server.completed_samples, 5);
                assert!(client.error.is_none());
                assert!(server.error.is_none());
                assert!(client.local_tcp_nodelay && server.local_tcp_nodelay);
                assert!(!client.native_qualified && !server.native_qualified);
                assert_eq!(client.samples.len(), 5);
                assert!(
                    client
                        .samples
                        .iter()
                        .all(|s| s.started_us.is_some() && s.rtt_us.is_some() && s.error.is_none())
                );
                assert!(
                    client
                        .samples
                        .windows(2)
                        .all(|s| s[0].started_us.unwrap().0 <= s[1].started_us.unwrap().0)
                );
                assert!(server.samples.is_empty());
                assert!(sample.report.rtt_mean_us.is_some());
                assert!(received.report.rtt_mean_us.is_none());
                assert!(client.rtt_p99_us.is_none());
                assert_eq!(client.keepalive_frames, server.keepalive_frames);
                if keepalive_ms.is_none() {
                    assert_eq!(client.keepalive_frames, 0);
                }
                assert_eq!(
                    sample.report.total_application_bytes,
                    received.report.total_application_bytes
                );
                assert_eq!(
                    sample.report.application_payload_bytes,
                    Decimal(5 * (16384 + 64))
                );
                assert!(sample.report.total_application_bytes.0 <= 2 * 1024 * 1024);
                assert_eq!(sample.report.id, received.report.id);
                assert!(
                    sample
                        .view(&registry.fence())
                        .unwrap()
                        .placement_inputs()
                        .is_err()
                );
                assert!(!sample.token_profile_snapshot().unwrap().admission);
                registry.heartbeat(None).unwrap();
            });
        }
        // A malicious enrolled peer gets a failed, partial diagnostic record;
        // invalid opcode and an unapproved warm frame are never successful samples.
        for opcode in [0, 3] {
            let mut cap = limits();
            cap.ping_samples = 5;
            cap.inter_ping_gap_ms = 20;
            cap.token_profile = Some(TokenProfile {
                gap_basis: GapBasis::StartToStart,
                keepalive_ms: None,
            });
            let budget = ProbeBudget::new(cap.clone()).unwrap();
            thread::scope(|attempt| {
                let server = attempt.spawn(|| {
                    let until = Instant::now() + Duration::from_secs(3);
                    loop {
                        if let Some(incoming) = listener.try_accept_registry(&authority).unwrap() {
                            break incoming
                                .probe(authority.enrollment_reader().unwrap(), &budget)
                                .unwrap();
                        }
                        assert!(Instant::now() < until);
                        thread::sleep(Duration::from_millis(1));
                    }
                });
                let mut wire = admitted_wire(&route, &worker, &invitation);
                let hello = Hello {
                    version: Version,
                    id: Uuid::new_v4(),
                    epoch: registry.fence().status().unwrap().0,
                    limits: cap,
                };
                crate::send(&mut wire, &hello).unwrap();
                assert_eq!(crate::receive::<Hello>(&mut wire).unwrap(), hello);
                wire.write_all(&[1]).unwrap();
                wire.write_all(&[0x5a; 16383]).unwrap();
                wire.flush().unwrap();
                let mut reply = [0; 64];
                wire.read_exact(&mut reply).unwrap();
                assert_eq!(&reply[..16], hello.id.as_bytes());
                assert_eq!(&reply[16..20], &0u32.to_le_bytes());
                assert_eq!(&reply[20..], &[0x5a; 44]);
                wire.write_all(&[opcode]).unwrap();
                wire.flush().unwrap();
                drop(wire);
                let sample = server.join().unwrap();
                let report = sample.report.token_profile.as_ref().unwrap();
                assert_eq!(report.completed_samples, 1);
                assert_eq!(report.error, Some(Error::Invalid));
                assert_eq!(
                    report.failure_phase,
                    Some(token::TokenProfilePhase::Discriminator)
                );
                assert!(sample.current(&registry.fence()).is_some()); // Diagnostics retained, never admission.
                assert!(
                    sample
                        .view(&registry.fence())
                        .unwrap()
                        .placement_inputs()
                        .is_err()
                );
                registry.heartbeat(None).unwrap();
            });
        }
        // Each case is an independent synthetic attempt budget. The production
        // daemon retains one budget; these cases exercise failures after admission.
        for fault in [
            "limits",
            "epoch",
            "marker",
            "payload",
            "partial-eof",
            "deadline",
        ] {
            let mut cap = limits();
            cap.ping_samples = 1;
            let budget = ProbeBudget::new(cap.clone()).unwrap();
            let outcome = thread::scope(|attempt| {
                let server = attempt.spawn(|| {
                    let until = Instant::now() + Duration::from_secs(3);
                    loop {
                        if let Some(incoming) = listener.try_accept_registry(&authority).unwrap() {
                            break incoming.probe(authority.enrollment_reader().unwrap(), &budget);
                        }
                        assert!(Instant::now() < until);
                        thread::sleep(Duration::from_millis(1));
                    }
                });
                let mut wire = admitted_wire(&route, &worker, &invitation);
                let mut hello = Hello {
                    version: Version,
                    id: Uuid::new_v4(),
                    epoch: registry.fence().status().unwrap().0,
                    limits: cap,
                };
                if fault == "limits" {
                    hello.limits.ping_samples = 2;
                }
                if fault == "epoch" {
                    hello.epoch.number.0 += 1;
                }
                crate::send(&mut wire, &hello).unwrap();
                if fault != "limits" && fault != "epoch" {
                    let accepted: Hello = crate::receive(&mut wire).unwrap();
                    assert_eq!(accepted, hello);
                    if fault == "deadline" {
                        assert!(wire.read(&mut [0; 1]).is_err());
                    } else {
                        crate::send(
                            &mut wire,
                            &Marker {
                                id: if fault == "marker" {
                                    Uuid::new_v4()
                                } else {
                                    hello.id
                                },
                                sequence: 0,
                            },
                        )
                        .unwrap();
                        if fault != "marker" {
                            let ack: Marker = crate::receive(&mut wire).unwrap();
                            assert_eq!(ack.id, hello.id);
                            assert_eq!(ack.sequence, 0);
                            crate::send(
                                &mut wire,
                                &Marker {
                                    id: hello.id,
                                    sequence: 10,
                                },
                            )
                            .unwrap();
                            wire.write_all(&[if fault == "payload" { 0 } else { 0x5a }; BLOCK])
                                .unwrap();
                            wire.flush().unwrap();
                        }
                    }
                }
                drop(wire);
                server.join().unwrap()
            });
            let expected = match fault {
                "limits" | "marker" | "payload" => Error::Invalid,
                "epoch" => Error::Expired,
                _ => Error::Transport,
            };
            assert_eq!(
                outcome.err(),
                Some(expected),
                "{fault} must reject at its intended boundary"
            );
            assert!(
                budget.claim(worker.node()).is_err(),
                "failed authenticated attempts consume cooldown"
            );
            assert_eq!(probe_budget.usage().unwrap().connections, 0);
            registry.heartbeat(None).unwrap();
        }
        assert!(matches!(
            measure(
                &route,
                &worker,
                &invitation,
                registry.fence(),
                &client_budget
            ),
            Err(Error::Capacity)
        ));
        observation.wall -= PROBE_TTL;
        assert!(observation.current(&registry.fence()).is_none());
        observation.wall += PROBE_TTL;
        let original_wall = observation.wall;
        observation.wall = SystemTime::now() + Duration::from_secs(1);
        assert!(
            observation.current(&registry.fence()).is_none(),
            "backward wall clock cannot freshen observations"
        );
        observation.wall = original_wall;
        let revocation_budget = ProbeBudget::new(limits()).unwrap();
        thread::scope(|attempt| {
            let server = attempt.spawn(|| {
                let until = Instant::now() + Duration::from_secs(3);
                loop {
                    if let Some(incoming) = listener.try_accept_registry(&authority).unwrap() {
                        break incoming
                            .probe(authority.enrollment_reader().unwrap(), &revocation_budget);
                    }
                    assert!(Instant::now() < until);
                    thread::sleep(Duration::from_millis(1));
                }
            });
            let mut wire = admitted_wire(&route, &worker, &invitation);
            let hello = Hello {
                version: Version,
                id: Uuid::new_v4(),
                epoch: registry.fence().status().unwrap().0,
                limits: limits(),
            };
            crate::send(&mut wire, &hello).unwrap();
            let accepted: Hello = crate::receive(&mut wire).unwrap();
            assert_eq!(accepted, hello);
            authority.revoke(worker.node()).unwrap();
            let mut byte = [0; 1];
            assert!(
                !matches!(wire.read(&mut byte), Ok(n) if n > 0),
                "revocation must close a proved idle probe before payload"
            );
            assert!(server.join().unwrap().is_err());
            assert_eq!(probe_budget.usage().unwrap().connections, 0);
        });
        registry.heartbeat(None).unwrap();
        assert_eq!(registry.fence().status().unwrap().1, NodeStatus::Revoked);
        assert!(observation.current(&registry.fence()).is_none());
        assert!(matches!(
            measure(
                &route,
                &worker,
                &invitation,
                registry.fence(),
                &ProbeBudget::new(limits()).unwrap()
            ),
            Err(Error::Expired)
        ));
        stopped.store(true, Ordering::Release);
        drop(registry);
    });
}
