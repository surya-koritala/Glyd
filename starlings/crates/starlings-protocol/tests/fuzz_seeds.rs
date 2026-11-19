use starlings_protocol::{DataHeader, DataKind, DataPrefix, FixtureMode};

#[path = "../../../fuzz/check.rs"]
mod input;

#[test]
fn every_frame_kind_has_a_valid_fuzz_seed_and_replays_the_actual_oracle() {
    input::check(&[]);
    for (kind, seed) in [
        (
            DataKind::Tensor,
            include_bytes!("../../../fuzz/seeds/tensor").as_slice(),
        ),
        (
            DataKind::Tensor,
            include_bytes!("../../../fuzz/seeds/tensor_limit").as_slice(),
        ),
        (
            DataKind::Credit,
            include_bytes!("../../../fuzz/seeds/credit").as_slice(),
        ),
        (
            DataKind::Receipt,
            include_bytes!("../../../fuzz/seeds/receipt").as_slice(),
        ),
        (
            DataKind::Execution,
            include_bytes!("../../../fuzz/seeds/execution").as_slice(),
        ),
        (
            DataKind::Cancel,
            include_bytes!("../../../fuzz/seeds/cancel").as_slice(),
        ),
        (
            DataKind::Error,
            include_bytes!("../../../fuzz/seeds/error").as_slice(),
        ),
    ] {
        let prefix = DataPrefix::parse(seed[..16].try_into().unwrap()).unwrap();
        assert_eq!(prefix.kind(), kind);
        DataHeader::decode(prefix, &seed[16..], FixtureMode::Simulation).unwrap();
        input::check(seed);
        for length in [0, 8, 15, 16, seed.len() - 1] {
            input::check(&seed[..length]);
        }
        let mut bad = seed.to_vec();
        bad[8..12].copy_from_slice(&u32::MAX.to_be_bytes());
        input::check(&bad);
        assert!(DataPrefix::parse(bad[..16].try_into().unwrap()).is_err());
    }
}
