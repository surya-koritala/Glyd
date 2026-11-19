//! Explicit CPU-only diagnostic; synthetic header, no backend or model execution.
use starlings_protocol::*;
use std::{hint::black_box, time::Instant};
fn main() {
    let header: TensorHeader = decode(
        include_bytes!("../../../examples/tensor-header.json"),
        FixtureMode::Simulation,
    )
    .unwrap();
    let request = ControlRequest::new(
        "00000000-0000-4000-8000-000000000991".parse().unwrap(),
        1,
        1,
        Operation::StartStep {
            header: Box::new(header),
            input_digest: byte_digest(b"synthetic"),
        },
    )
    .unwrap();
    let bytes = serde_json::to_vec(&request).unwrap();
    ControlRequest::decode(&bytes, FixtureMode::Simulation).unwrap();
    let count = 1000;
    let started = Instant::now();
    for _ in 0..count {
        let decoded = ControlRequest::decode(black_box(&bytes), FixtureMode::Simulation).unwrap();
        // Mirrors the additional validation at Worker::apply, without timing state transitions.
        black_box(decoded)
            .validate(FixtureMode::Simulation)
            .unwrap();
    }
    println!(
        "{{\"synthetic\":true,\"os\":\"{}\",\"arch\":\"{}\",\"iterations\":{},\"decode_plus_apply_validation_ns_mean\":{},\"includes_backend\":false}}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        count,
        started.elapsed().as_nanos() / count
    );
}
