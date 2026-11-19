#[path = "../../../fuzz/receive_check.rs"]
mod input;

fn action(trace: &mut Vec<u8>, kind: u8, offset: u64, bytes: u64, end: bool) {
    trace.push(kind);
    trace.extend_from_slice(&offset.to_le_bytes());
    trace.extend_from_slice(&bytes.to_le_bytes());
    trace.push(u8::from(end));
}

#[test]
fn ordered_split_tensors_conserve_credit_and_reject_corrupted_traces() {
    let mut trace = vec![1, 3, 1]; // Two tensors: four bytes, then two bytes.
    action(&mut trace, 0, 0, 2, false);
    action(&mut trace, 1, 0, 2, false);
    action(&mut trace, 0, 0, 4, false);
    action(&mut trace, 1, 2, 2, true);
    action(&mut trace, 1, 0, 2, true);
    action(&mut trace, 2, 0, 0, false);
    assert!(input::check(&trace));
    for length in 0..trace.len() {
        input::check(&trace[..length]);
    }
    for index in 0..trace.len() {
        let mut mutated = trace.clone();
        mutated[index] ^= 0xff;
        input::check(&mutated);
    }
    // Each trace fails at its first invalid operation; no unbounded retry loop.
    for (kind, offset, bytes, end) in [
        (0, 0, 0, false),
        (0, 0, 7, false),
        (0, 0, u64::MAX, false),
        (1, 0, 1, false),
        (1, 1, 1, false),
        (1, u64::MAX, u64::MAX, true),
        (2, 0, 0, false),
    ] {
        let mut invalid = vec![1, 3, 1];
        action(&mut invalid, kind, offset, bytes, end);
        assert!(!input::check(&invalid));
    }
    // A duplicate, gap, overshoot, exhausted credit, or false final flag after a read.
    for (offset, bytes, end) in [(0, 2, true), (3, 1, true), (2, 3, true), (2, 2, false)] {
        let mut invalid = trace[..3 + 3 * 18].to_vec();
        action(&mut invalid, 1, offset, bytes, end);
        assert!(!input::check(&invalid));
    }
    let mut exhausted = trace[..3 + 2 * 18].to_vec();
    action(&mut exhausted, 1, 2, 2, true);
    assert!(!input::check(&exhausted));
    let mut overflow = trace[..3 + 2 * 18].to_vec();
    action(&mut overflow, 0, 0, u64::MAX, false);
    assert!(!input::check(&overflow));
}

#[test]
fn whole_tensor_and_per_chunk_grants_share_the_same_bounds() {
    for count in 1..=8 {
        for length in [1u64, 2, 255, 256] {
            for split_grants in [false, true] {
                let mut trace = vec![count - 1];
                trace.extend(std::iter::repeat_n((length - 1) as u8, usize::from(count)));
                if !split_grants {
                    action(&mut trace, 0, 0, length * u64::from(count), false);
                }
                for _ in 0..count {
                    let first = length.div_ceil(2);
                    for (offset, bytes) in [(0, first), (first, length - first)] {
                        if bytes == 0 {
                            continue;
                        }
                        if split_grants {
                            action(&mut trace, 0, 0, bytes, false);
                        }
                        action(&mut trace, 1, offset, bytes, offset + bytes == length);
                    }
                }
                assert!(input::check(&trace));
            }
        }
    }
}
