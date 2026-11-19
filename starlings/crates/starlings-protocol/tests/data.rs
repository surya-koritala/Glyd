use serde_json::{Value, json};
use starlings_protocol::*;

const MODE: FixtureMode = FixtureMode::Simulation;
fn tensor() -> TensorHeader {
    decode(include_bytes!("../../../examples/tensor-header.json"), MODE).unwrap()
}
fn scope() -> Value {
    serde_json::to_value(DataScope::from(&tensor())).unwrap()
}

#[test]
fn fixed_prefix_golden_and_all_untrusted_extents_are_bounded() {
    let bytes = [71, 67, 80, 49, 0, 1, 0, 1, 0, 0, 64, 0, 0, 128, 0, 0];
    let p = DataPrefix::parse(bytes).unwrap();
    assert_eq!(
        p,
        DataPrefix::new(DataKind::Tensor, 16_384, 8_388_608).unwrap()
    );
    assert_eq!(p.bytes(), bytes);
    for index in 0..8 {
        let mut bad = bytes;
        bad[index] = 255;
        assert!(DataPrefix::parse(bad).is_err(), "prefix byte {index}");
    }
    for header in [
        0,
        1,
        DATA_HEADER_LIMIT,
        DATA_HEADER_LIMIT + 1,
        u32::MAX as usize,
    ] {
        for payload in [
            0,
            1,
            DATA_PAYLOAD_LIMIT,
            DATA_PAYLOAD_LIMIT + 1,
            u32::MAX as usize,
        ] {
            for kind in [
                DataKind::Tensor,
                DataKind::Credit,
                DataKind::Receipt,
                DataKind::Execution,
                DataKind::Cancel,
                DataKind::Error,
            ] {
                let result = DataPrefix::new(kind, header, payload);
                let expected = (1..=DATA_HEADER_LIMIT).contains(&header)
                    && payload <= DATA_PAYLOAD_LIMIT
                    && ((kind == DataKind::Tensor) == (payload != 0));
                assert_eq!(result.is_ok(), expected, "{kind:?} {header} {payload}");
                if let Ok(p) = result {
                    assert_eq!(DataPrefix::parse(p.bytes()).unwrap(), p);
                }
            }
        }
    }
    for kind in 0u16..=255 {
        let mut candidate = bytes;
        candidate[6..8].copy_from_slice(&kind.to_be_bytes());
        candidate[12..16].copy_from_slice(&if kind == 1 { 1u32 } else { 0 }.to_be_bytes());
        assert_eq!(
            DataPrefix::parse(candidate).is_ok(),
            (1..=6).contains(&kind)
        );
    }
}

#[test]
fn all_control_kinds_are_strict_bounded_and_distinct_from_tensor_payloads() {
    for (kind, extra) in [
        (
            DataKind::Credit,
            json!({"kind":"Credit","bytes":"67108864"}),
        ),
        (
            DataKind::Receipt,
            json!({"kind":"ReceiptAck","tensor_index":0,"received_through_bytes":"8"}),
        ),
        (DataKind::Execution, json!({"kind":"ExecutionResult"})),
        (
            DataKind::Cancel,
            json!({"kind":"Cancel","code":"cancelled"}),
        ),
        (DataKind::Error, json!({"kind":"Error","code":"protocol"})),
    ] {
        let mut value = json!({"protocol_version":1,"scope":scope()});
        value
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let bytes = serde_json::to_vec(&value).unwrap();
        let p = DataPrefix::new(kind, bytes.len(), 0).unwrap();
        let header = DataHeader::decode(p, &bytes, MODE).unwrap();
        let (encoded_prefix, encoded) = header.encode(MODE).unwrap();
        assert_eq!(encoded_prefix, p);
        assert_eq!(parse_json(&encoded).unwrap(), value);
        assert_eq!(header.scope(), DataScope::from(&tensor()));
        assert!(DataHeader::decode(p, &bytes, FixtureMode::Reject).is_err());
        let mismatch = if kind == DataKind::Credit {
            DataKind::Receipt
        } else {
            DataKind::Credit
        };
        assert!(
            DataHeader::decode(
                DataPrefix::new(mismatch, bytes.len(), 0).unwrap(),
                &bytes,
                MODE
            )
            .is_err()
        );
        for change in [
            "version",
            "epoch",
            "unknown",
            "unknown_scope",
            "overflow",
            "id",
            "missing",
            "decimal",
        ] {
            let mut bad = value.clone();
            match change {
                "version" => bad["protocol_version"] = json!(2),
                "epoch" => bad["scope"]["authority_epoch"] = json!("0"),
                "unknown" => bad["payload"] = json!("forbidden"),
                "unknown_scope" => bad["scope"]["token"] = json!("forbidden"),
                "overflow" => bad["scope"]["sequence"] = json!("18446744073709551616"),
                "id" => bad["scope"]["invocation_id"] = json!("not-an-id"),
                "missing" => {
                    bad.as_object_mut().unwrap().remove("scope");
                }
                "decimal" => bad["scope"]["authority_epoch"] = json!("01"),
                _ => unreachable!(),
            }
            assert!(
                decode::<DataControl>(&serde_json::to_vec(&bad).unwrap(), MODE).is_err(),
                "{kind:?}/{change}"
            );
        }
        let duplicate = String::from_utf8(bytes).unwrap().replacen(
            "\"protocol_version\":1",
            "\"protocol_version\":1,\"protocol_version\":1",
            1,
        );
        assert!(decode::<DataControl>(duplicate.as_bytes(), MODE).is_err());
    }
    for field in ["bytes", "received_through_bytes"] {
        for bad in ["0", "67108865", "18446744073709551615"] {
            let mut v = if field == "bytes" {
                json!({"kind":"Credit","protocol_version":1,"scope":scope(),"bytes":bad})
            } else {
                json!({"kind":"ReceiptAck","protocol_version":1,"scope":scope(),"tensor_index":0,"received_through_bytes":bad})
            };
            assert!(decode::<DataControl>(&serde_json::to_vec(&v).unwrap(), MODE).is_err());
            v[field] = json!("1");
            assert!(decode::<DataControl>(&serde_json::to_vec(&v).unwrap(), MODE).is_ok());
        }
    }
}

#[test]
fn tensor_header_and_prefix_agree_before_payload_allocation() {
    let original = DataHeader::Tensor(tensor());
    let (prefix, bytes) = original.encode(MODE).unwrap();
    assert_eq!(DataHeader::decode(prefix, &bytes, MODE).unwrap(), original);
    assert!(
        DataHeader::decode(
            DataPrefix::new(DataKind::Tensor, bytes.len(), 1).unwrap(),
            &bytes,
            MODE
        )
        .is_err()
    );
    assert!(DataHeader::decode(prefix, &bytes[..bytes.len() - 1], MODE).is_err());
    for change in ["overflow", "rank", "chunk", "end", "dtype", "tokens"] {
        let mut v = serde_json::to_value(tensor()).unwrap();
        match change {
            "overflow" => v["shape"] = json!([1048576, 1048576, 1048576, 1048576]),
            "rank" => v["shape"] = json!([1, 1, 2, 4096, 1]),
            "chunk" => v["chunk_offset_bytes"] = json!("18446744073709551615"),
            "end" => v["end_of_tensor"] = json!(false),
            "dtype" => v["dtype"] = json!("u32"),
            "tokens" => v["token_count"] = json!(3),
            _ => unreachable!(),
        }
        let b = serde_json::to_vec(&v).unwrap();
        let p = DataPrefix::new(DataKind::Tensor, b.len(), 16384).unwrap();
        assert!(DataHeader::decode(p, &b, MODE).is_err(), "{change}");
    }
}

#[test]
fn fixed_little_endian_vectors_keep_arbitrary_bits_without_float_conversion() {
    let vectors: Vec<Value> =
        serde_json::from_slice(include_bytes!("../../../examples/gcp1-golden.json")).unwrap();
    let expected = [
        vec![0, 0, 0, 128, 128, 127, 193, 127],
        vec![0, 0, 0, 128, 0, 124, 1, 126],
        vec![0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 128, 127, 1, 0, 192, 127],
        vec![0, 0, 0, 0, 1, 0, 0, 0, 120, 86, 52, 18, 255, 255, 255, 255],
    ];
    assert_eq!(vectors.len(), expected.len());
    for (vector, payload) in vectors.iter().zip(expected) {
        let prefix_bytes: [u8; 16] = serde_json::from_value(vector["prefix"].clone()).unwrap();
        let prefix = DataPrefix::parse(prefix_bytes).unwrap();
        let header_bytes = vector["header_utf8"].as_str().unwrap().as_bytes();
        let h = DataHeader::decode(prefix, header_bytes, MODE).unwrap();
        let supplied: Vec<u8> = serde_json::from_value(vector["payload"].clone()).unwrap();
        assert_eq!(supplied, payload);
        assert_eq!(prefix.payload_len(), payload.len());
        let DataHeader::Tensor(tensor) = h else {
            panic!("expected tensor")
        };
        assert_eq!(tensor.total_tensor_bytes.0, payload.len() as u64);
        assert_eq!(serde_json::to_value(tensor.dtype).unwrap(), vector["name"]);
        assert!(DataHeader::decode(prefix, header_bytes, FixtureMode::Reject).is_err());
    }
    let controls: Vec<Value> =
        serde_json::from_slice(include_bytes!("../../../examples/gcp1-control-golden.json"))
            .unwrap();
    assert_eq!(controls.len(), 5);
    for (i, v) in controls.iter().enumerate() {
        let bytes: [u8; 16] = serde_json::from_value(v["prefix"].clone()).unwrap();
        assert_eq!(&bytes[6..8], &((i + 2) as u16).to_be_bytes());
        let prefix = DataPrefix::parse(bytes).unwrap();
        assert_eq!(prefix.payload_len(), 0);
        assert_eq!(prefix.bytes(), bytes);
        let header = v["header_utf8"].as_str().unwrap().as_bytes();
        assert!(matches!(
            DataHeader::decode(prefix, header, MODE).unwrap(),
            DataHeader::Control(_)
        ));
        assert!(DataHeader::decode(prefix, header, FixtureMode::Reject).is_err());
    }
}

// No payload is allocated: accepted metadata must stay within the wire limits
// and survive a semantic round trip; structural validity confers no authority.
fn check_mutated_frame(prefix_bytes: [u8; 16], bytes: &[u8]) -> bool {
    let Ok(prefix) = DataPrefix::parse(prefix_bytes) else {
        return false;
    };
    assert_eq!(prefix.bytes(), prefix_bytes);
    assert!((1..=DATA_HEADER_LIMIT).contains(&prefix.header_len()));
    assert!(prefix.payload_len() <= DATA_PAYLOAD_LIMIT);
    assert_eq!(prefix.kind() == DataKind::Tensor, prefix.payload_len() != 0);
    let Ok(header) = DataHeader::decode(prefix, bytes, MODE) else {
        return false;
    };
    let (encoded_prefix, encoded) = header.encode(MODE).unwrap();
    assert_eq!(encoded_prefix.kind(), prefix.kind());
    assert_eq!(encoded_prefix.payload_len(), prefix.payload_len());
    assert_eq!(
        DataHeader::decode(encoded_prefix, &encoded, MODE).unwrap(),
        header
    );
    true
}

#[test]
fn bounded_mutation_corpus_checks_prefixes_and_headers_without_payload_allocation() {
    let mut vectors: Vec<Value> =
        serde_json::from_slice(include_bytes!("../../../examples/gcp1-golden.json")).unwrap();
    vectors.extend(
        serde_json::from_slice::<Vec<Value>>(include_bytes!(
            "../../../examples/gcp1-control-golden.json"
        ))
        .unwrap(),
    );
    assert_eq!(vectors.len(), 9);
    let mut accepted = 0;
    let mut rejected = 0;
    for vector in vectors {
        let prefix: [u8; 16] = serde_json::from_value(vector["prefix"].clone()).unwrap();
        let bytes = vector["header_utf8"].as_str().unwrap().as_bytes();
        assert!(check_mutated_frame(prefix, bytes));
        // Every prefix bit, including all length fields; bodies retain their actual size.
        for index in 0..16 {
            for bit in 0..8 {
                let mut changed = prefix;
                changed[index] ^= 1 << bit;
                if check_mutated_frame(changed, bytes) {
                    accepted += 1;
                } else {
                    rejected += 1;
                }
            }
        }
        // Fixed, evenly spaced mutations reach syntax, scalar values and scope fields.
        // Some changes remain legal metadata; they must not be mistaken for corruption.
        for index in 0..32 {
            let offset = index * (bytes.len() - 1) / 31;
            for mask in [1, 128, 255] {
                let mut changed = bytes.to_vec();
                changed[offset] ^= mask;
                if check_mutated_frame(prefix, &changed) {
                    accepted += 1;
                } else {
                    rejected += 1;
                }
            }
        }
        for index in 0..16 {
            let len = index * bytes.len() / 16;
            // Match the actual truncated length so the parser, not just framing, is checked.
            let mut changed = prefix;
            changed[8..12].copy_from_slice(&(len as u32).to_be_bytes());
            assert!(!check_mutated_frame(changed, &bytes[..len]));
            rejected += 1;
        }
    }
    assert!(accepted > 0 && rejected > 0);
    assert_eq!(accepted + rejected, 9 * (128 + 96 + 16));
    eprintln!(
        "GCP1_MUTATIONS: seeds=9 cases={} accepted={accepted} rejected={rejected}",
        accepted + rejected
    );
}
