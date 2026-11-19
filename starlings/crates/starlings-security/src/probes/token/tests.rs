use super::*;
fn limits() -> ProbeLimits {
    ProbeLimits {
        version: Version,
        payload_bytes: PAYLOAD as u32,
        ping_samples: 5,
        inter_ping_gap_ms: 20,
        minimum_interval_ms: 30_000,
        token_profile: Some(TokenProfile {
            gap_basis: GapBasis::StartToStart,
            keepalive_ms: None,
        }),
    }
}
#[test]
fn legacy_hash_and_profile_limits_remain_distinct_and_bounded() {
    let mut legacy = limits();
    legacy.token_profile = None;
    legacy.ping_samples = 3;
    legacy.inter_ping_gap_ms = 0;
    legacy.validate().unwrap();
    assert_eq!(
        serde_json::to_value(starlings_protocol::content_hash(&legacy).unwrap()).unwrap(),
        "1916b44ace46e5c4008f1dfcd09dc11da4ed66dfc5406286c66c256a183afa4e"
    );
    legacy.ping_samples = 9;
    assert_eq!(legacy.validate(), Err(Error::Invalid));
    let mut profile = limits();
    profile.validate().unwrap();
    profile.ping_samples = 100;
    profile.validate().unwrap();
    for count in [0, 1, 4, 101, u32::MAX] {
        profile.ping_samples = count;
        assert_eq!(profile.validate(), Err(Error::Invalid));
    }
    for (payload, gap, wake) in [
        (4096, 20, None),
        (16384, 19, None),
        (16384, 101, None),
        (16384, 20, Some(0)),
        (16384, 20, Some(4)),
        (16384, 20, Some(20)),
        (16384, 100, Some(21)),
    ] {
        let mut invalid = limits();
        invalid.payload_bytes = payload;
        invalid.inter_ping_gap_ms = gap;
        invalid.token_profile.as_mut().unwrap().keepalive_ms = wake;
        assert_eq!(invalid.validate(), Err(Error::Invalid));
    }
    let mut warm = limits();
    warm.token_profile.as_mut().unwrap().keepalive_ms = Some(5);
    warm.validate().unwrap();
    assert_ne!(
        starlings_protocol::content_hash(&warm).unwrap(),
        starlings_protocol::content_hash(&limits()).unwrap()
    );
}
#[test]
fn gap_basis_reply_binding_and_tail_floor_use_no_network_or_sleep() {
    let start = Instant::now();
    let reply = start + Duration::from_millis(30);
    let mut config = limits().token_profile.unwrap();
    assert_eq!(
        gap_until(&config, start, reply, Duration::from_millis(20)),
        start + Duration::from_millis(20)
    );
    config.gap_basis = GapBasis::AfterReply;
    assert_eq!(
        gap_until(&config, start, reply, Duration::from_millis(100)),
        start + Duration::from_millis(130)
    );
    let id = Uuid::from_bytes([7; 16]);
    let bytes = response(id, 9);
    assert_eq!(bytes.len(), 64);
    assert_eq!(&bytes[..16], id.as_bytes());
    assert_eq!(&bytes[16..20], &9u32.to_le_bytes());
    assert_ne!(bytes, response(id, 10));
    assert_ne!(bytes, response(Uuid::from_bytes([8; 16]), 9));
    let (mean, var, p50, p99) = profile_statistics(&[50, 10, 30, 20, 40]).unwrap();
    assert_eq!((mean, var.0, p50, p99), (30, 200, 30, None));
    let samples: Vec<_> = (1..=100).rev().collect();
    let (mean, var, p50, p99) = profile_statistics(&samples).unwrap();
    assert_eq!((mean, var.0, p50, p99), (50, 833, 50, Some(99)));
    for invalid in [vec![], vec![0], vec![1; 101]] {
        assert!(profile_statistics(&invalid).is_err());
    }
}
#[test]
fn maximum_profile_reply_fits_admin_record_and_never_becomes_placement_input() {
    let mut cap = limits();
    cap.ping_samples = 100;
    let id = Uuid::from_bytes([7; 16]);
    let epoch = AuthorityEpoch {
        number: Decimal(1),
        incarnation: id,
    };
    let hello = Hello {
        version: Version,
        id,
        epoch,
        limits: cap.clone(),
    };
    let mut profile =
        TokenProfileReport::new(&hello, starlings_protocol::byte_digest(b"fixture producer"));
    profile.completed_samples = 100;
    profile.samples = (0..100)
        .map(|sequence| TokenSample {
            sequence,
            started_us: Some(Decimal(30_000_000)),
            rtt_us: Some(30_000_000),
            keepalive_frames: 20,
            error: None,
        })
        .collect();
    profile.keepalive_frames = WAKES;
    let node: NodeId = "00000000-0000-4000-8000-000000000001".parse().unwrap();
    let report = ProbeReport {
        version: Version,
        id,
        epoch,
        local_address: "[::1]:65535".parse().unwrap(),
        peer_address: "[::1]:65534".parse().unwrap(),
        selected_interface: None,
        rtt_samples_us: None,
        rtt_mean_us: Some(30_000_000),
        rtt_variance_us2: Some(Decimal(u64::MAX)),
        inter_ping_gap_ms: 100,
        transfers: [
            measurement(
                node,
                node,
                1_638_400,
                Duration::from_secs(30),
                "token-profile-client-window",
            )
            .unwrap(),
            measurement(
                node,
                node,
                6400,
                Duration::from_secs(30),
                "token-profile-client-window",
            )
            .unwrap(),
        ],
        application_payload_bytes: Decimal(1_644_800),
        total_application_bytes: Decimal(PROFILE_BYTES as u64),
        token_profile: Some(profile),
    };
    let observation = ProbeObservation::captured(report, Instant::now(), SystemTime::now());
    let snapshot = observation.token_profile_snapshot().unwrap();
    assert!(!snapshot.admission);
    let value = serde_json::json!({"state":"profile_complete","profile_limits":cap,"profile_attempt":snapshot,"measurement":null,"placement_inputs":null});
    let reply = serde_json::json!({"ok":true,"code":"LINKS","message":value.to_string()});
    assert!(serde_json::to_vec(&reply).unwrap().len() <= crate::MAX_RECORD);
    let view = ProbeView {
        report: &observation.report,
        age_ms: 0,
    };
    assert!(matches!(view.placement_inputs(), Err(Error::Denied)));
    let mut stale = observation;
    stale.at -= PROBE_TTL;
    assert!(stale.token_profile_snapshot().is_some());
    assert!(
        !stale
            .report
            .token_profile
            .as_ref()
            .unwrap()
            .native_qualified
    );
}
