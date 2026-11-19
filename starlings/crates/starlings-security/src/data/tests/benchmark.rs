use super::*;
use std::{fs::OpenOptions, hint::black_box};

fn nanos(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap()
}

fn sample(size: usize, count: usize) -> serde_json::Value {
    assert!([8192, DATA_PAYLOAD_LIMIT].contains(&size) && (1..=2).contains(&count));
    let total = (size * count) as u64;
    let f = Fixture::with_scratch(total);
    let mut h = f.header(&vectors()[2]);
    h.shape = vec![
        BoundedU32::new(1).unwrap(),
        BoundedU32::new(2).unwrap(),
        BoundedU32::new((size / 8) as u32).unwrap(),
    ];
    h.total_tensor_bytes = Decimal(size as u64);
    h.chunk_bytes = Decimal(size as u64);
    h.tensor_count = BoundedU32::new(count as u32).unwrap();
    let headers: Vec<_> = (0..count)
        .map(|index| {
            let mut h = h.clone();
            h.tensor_index = BoundedU32::new(index as u32).unwrap();
            if index != 0 {
                h.tensor_role = TensorHeaderTensorRole::Residual;
            }
            h
        })
        .collect();
    let setup = Instant::now();
    let (mut sender, mut receiver) = f.connect(&headers[0]);
    let setup_ns = nanos(setup);
    let source: Vec<_> = headers
        .iter()
        .map(|h| f.admission(&sender.peer, h.clone()))
        .collect();
    let destination: Vec<_> = headers
        .iter()
        .map(|h| f.admission(&receiver.peer, h.clone()))
        .collect();
    let before = serde_json::to_value(f.workers[1].snapshot().unwrap()).unwrap();
    let payloads: Vec<Vec<u8>> = (0..count)
        .map(|index| (0..size).map(|i| ((i + index) % 251) as u8).collect())
        .collect();
    let mut copy = vec![0; size];
    let copied = Instant::now();
    for bytes in &payloads {
        black_box(&mut copy).copy_from_slice(black_box(bytes));
        black_box(&copy);
    }
    let host_copy_ns = nanos(copied);
    let budget = ReceiveBudget::new(total, total, total).unwrap();
    let start = Instant::now();
    let deadline = start + Duration::from_secs(10);
    let (tensors, allocation_ns, read_ns, handoff_ns, credit_ns, send_ns, receipt_ns) =
        thread::scope(|threads| {
            let task = threads.spawn(|| {
                let begin = Instant::now();
                let mut rx = receiver
                    .begin_receive(&destination, &budget, deadline)
                    .unwrap();
                let allocation_ns = nanos(begin);
                assert_eq!(budget.held_bytes().unwrap(), total);
                for _ in 0..count {
                    rx.grant_credit(size as u64).unwrap();
                }
                let read = Instant::now();
                for _ in 0..count {
                    rx.receive_chunk().unwrap();
                }
                let read_ns = nanos(read);
                let handoff = Instant::now();
                let tensors = rx.finish().unwrap();
                (tensors, allocation_ns, read_ns, nanos(handoff))
            });
            let mut tx = sender.begin_send(&source, deadline).unwrap();
            let credit = Instant::now();
            for _ in 0..count {
                assert!(matches!(
                    tx.receive_control().unwrap(),
                    DataControl::Credit { .. }
                ));
            }
            let credit_ns = nanos(credit);
            let mut send_ns = 0;
            let mut receipt_ns = 0;
            for bytes in &payloads {
                let send = Instant::now();
                tx.send_chunk(bytes).unwrap();
                send_ns += nanos(send);
                let receipt = Instant::now();
                assert!(matches!(
                    tx.receive_control().unwrap(),
                    DataControl::ReceiptAck { .. }
                ));
                receipt_ns += nanos(receipt);
            }
            tx.finish().unwrap();
            let (tensors, allocation, read, handoff) = task.join().unwrap();
            (
                tensors, allocation, read, handoff, credit_ns, send_ns, receipt_ns,
            )
        });
    let elapsed_ns = nanos(start);
    for (index, (actual, expected)) in tensors.tensors().iter().zip(&payloads).enumerate() {
        assert!(
            actual.as_ref() == expected,
            "payload mismatch at tensor {index}"
        );
    }
    assert_eq!(tensors.tensors().len(), count);
    let returned_bytes: usize = tensors.tensors().iter().map(|b| b.len()).sum();
    assert_eq!(returned_bytes as u64, total);
    assert_eq!(budget.held_bytes().unwrap(), total);
    assert_eq!(
        serde_json::to_value(f.workers[1].snapshot().unwrap()).unwrap(),
        before
    );
    drop(tensors);
    assert_eq!(budget.held_bytes().unwrap(), 0);
    serde_json::json!({
        "kind": "SyntheticLoopbackTransportSample", "version": 1,
        "tensor_bytes": size, "tensor_count": count, "payload_bytes": total,
        "dtype": "f32", "tcp_tls_binding_setup_ns": setup_ns,
        "host_copy_baseline_ns": host_copy_ns,
        "receive_begin_policy_reserve_allocate_ns": allocation_ns,
        "receive_read_validate_ack_ns": read_ns, "input_handoff_ns": handoff_ns,
        "send_framed_tls_ns": send_ns, "credit_wait_validate_ns": credit_ns,
        "receipt_wait_validate_ns": receipt_ns, "application_elapsed_ns": elapsed_ns,
        "application_payload_bytes_per_second":
            (u128::from(total) * 1_000_000_000).checked_div(u128::from(elapsed_ns)).map(|v| u64::try_from(v).unwrap()),
        "retained_payload_bytes": returned_bytes, "after_drop_payload_bytes": 0,
        "exact_bytes": true, "core_state_unchanged": true,
        "raw_physical_link_bits_per_second": null, "tls_only_ns": null,
        "tls_internal_copy_bytes": null, "allocator_peak_bytes": null, "device_sync_ns": null,
        "native_qualified": false
    })
}

#[test]
fn transport_sample_accounts_for_two_inputs_and_keeps_unmeasured_costs_absent() {
    let row = sample(8192, 2);
    assert_eq!(row["payload_bytes"], 16384);
    assert_eq!(row["retained_payload_bytes"], 16384);
    assert_eq!(row["after_drop_payload_bytes"], 0);
    for key in [
        "raw_physical_link_bits_per_second",
        "tls_only_ns",
        "tls_internal_copy_bytes",
        "allocator_peak_bytes",
        "device_sync_ns",
    ] {
        assert!(row[key].is_null());
    }
    assert_eq!(row["native_qualified"], false);
}

#[test]
#[ignore = "explicit model-free loopback matrix; requires a new JSONL output path"]
fn record_bounded_transport_matrix() {
    let path = std::path::PathBuf::from(
        std::env::var_os("STARLINGS_TRANSPORT_BENCH_OUTPUT").expect("new output path required"),
    );
    assert!(path.is_absolute(), "absolute new output path required");
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .expect("output must be new and its parent directory must exist");
    let binary = std::fs::File::open(std::env::current_exe().unwrap()).unwrap();
    let identity = crate::catalog::fingerprint(&binary, binary.metadata().unwrap().len()).unwrap();
    let header = serde_json::json!({
        "kind": "SyntheticLoopbackTransportRun", "version": 1, "binary_sha256": identity,
        "package_version": env!("CARGO_PKG_VERSION"),
        "build_profile": "test", "debug_assertions": cfg!(debug_assertions),
        "os": std::env::consts::OS, "architecture": std::env::consts::ARCH,
        "link": "loopback", "fixture_model_digest": byte_digest(b"synthetic data fixture"),
        "payload_pattern": "(offset+tensor_index)%251",
        "backend": "none", "samples": 20, "native_qualified": false
    });
    serde_json::to_writer(&mut output, &header).unwrap();
    output.write_all(b"\n").unwrap();
    for size in [8192, DATA_PAYLOAD_LIMIT] {
        for count in [1, 2] {
            for trial in 0..5 {
                let mut row = sample(size, count);
                row["trial"] = serde_json::json!(trial);
                serde_json::to_writer(&mut output, &row).unwrap();
                output.write_all(b"\n").unwrap();
                output.flush().unwrap();
            }
        }
    }
    serde_json::to_writer(
        &mut output,
        &serde_json::json!({"complete": true, "samples": 20}),
    )
    .unwrap();
    output.write_all(b"\n").unwrap();
    output.sync_all().unwrap();
}
