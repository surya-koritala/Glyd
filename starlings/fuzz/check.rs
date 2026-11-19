use starlings_protocol::{
    DATA_HEADER_LIMIT, DATA_PAYLOAD_LIMIT, DataHeader, DataKind, DataPrefix, FixtureMode,
};

pub fn check(data: &[u8]) {
    let Some(raw) = data.get(..16) else { return };
    let Ok(prefix) = DataPrefix::parse(raw.try_into().unwrap()) else {
        return;
    };
    assert!((1..=DATA_HEADER_LIMIT).contains(&prefix.header_len()));
    assert!(prefix.payload_len() <= DATA_PAYLOAD_LIMIT);
    assert_eq!(prefix.kind() == DataKind::Tensor, prefix.payload_len() != 0);
    assert_eq!(DataPrefix::parse(prefix.bytes()).unwrap(), prefix);
    for mode in [FixtureMode::Reject, FixtureMode::Simulation] {
        if let Ok(header) = DataHeader::decode(prefix, &data[16..], mode) {
            let (encoded, bytes) = header.encode(mode).unwrap();
            assert_eq!(encoded.kind(), prefix.kind());
            assert_eq!(encoded.payload_len(), prefix.payload_len());
            assert_eq!(DataHeader::decode(encoded, &bytes, mode).unwrap(), header);
        }
    }
}
