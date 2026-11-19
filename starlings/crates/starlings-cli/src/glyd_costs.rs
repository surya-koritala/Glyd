//! File-only projection of provider data, never an admitted ModelCostProfile.
use serde::{Deserialize, Serialize};
use starlings_protocol::{Digest, MAX_JSON_BYTES, byte_digest, parse_json};
use std::{fs::File, path::Path};

type Result<T> = std::result::Result<T, &'static str>;

#[derive(Debug, Deserialize, Serialize)]
struct Build {
    version: String,
    commit: Option<String>,
    development: bool,
}
#[derive(Debug, Deserialize, Serialize)]
struct Model {
    id: String,
    repo: Option<String>,
    revision: Option<String>,
    layers: u32,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum SizeBasis {
    File,
    Estimated,
    Mixed,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum WorkingBasis {
    Measured,
    Unavailable,
}
#[derive(Debug, Deserialize, Serialize)]
struct Basis {
    sizes: SizeBasis,
    packs_from_files: u32,
    packs_estimated: u32,
    working_memory: WorkingBasis,
}
#[derive(Debug, Deserialize, Serialize)]
struct Workload {
    context_tokens: u32,
    prompt_tokens: u32,
    output_tokens: u32,
    prefill_chunk_tokens: u32,
    sessions: u32,
}
#[derive(Debug, Deserialize, Serialize)]
struct Resident {
    total: u64,
}
#[derive(Debug, Deserialize, Serialize)]
struct Layer {
    index: u32,
    kind: String,
    resident_bytes: Resident,
    active_bytes_per_token: u64,
    kv_bytes_per_token: u64,
    state_bytes_per_session: u64,
}
#[derive(Debug, Deserialize, Serialize)]
struct Totals {
    resident_bytes: u64,
    kv_bytes_per_token: u64,
    kv_bytes_at_context: u64,
    state_bytes_per_session: u64,
    working_bytes: Option<u64>,
    overhead_bytes: Option<u64>,
    engine_bytes_at_context: u64,
}
#[derive(Debug, Deserialize, Serialize)]
struct Rate {
    value: f64,
    basis: String,
    placement: String,
}
#[derive(Debug, Deserialize, Serialize)]
struct Timing {
    prefill_tok_s: Option<Rate>,
    decode_tok_s: Option<Rate>,
}
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum Blocker {
    RegisteredManifestBuildRuntimeAndDeviceIdentities,
    PhysicalPoolInventoryAndComponentOwnership,
    UnverifiedStateBasisAndSeparateStateAllocations,
    PerPhaseScratchStagingAndReserves,
    MeasuredPhaseTimingsAndReviewedConfidence,
    FreshnessRawEvidenceAndLocalApproval,
}
// Unknown provider fields are deliberately unprojected, not validated or adopted.
#[derive(Debug, Deserialize, Serialize)]
struct Document {
    object: String,
    schema: String,
    build: Build,
    model: Model,
    basis: Basis,
    workload: Workload,
    observed_at: u64,
    samples: u32,
    layers: Vec<Layer>,
    totals: Totals,
    timing: Timing,
}
#[derive(Debug, Serialize)]
pub struct Report {
    kind: &'static str,
    source_digest: Digest,
    admission: &'static str,
    reported: Document,
    unavailable_compute_fields: [Blocker; 6],
}

pub fn inspect_file(path: &Path) -> Result<Report> {
    let bytes = crate::scoreboard::read(
        File::open(path).map_err(|_| "GLYD_COSTS_READ_FAILED")?,
        MAX_JSON_BYTES,
    )
    .map_err(|_| "GLYD_COSTS_READ_FAILED")?;
    inspect(&bytes)
}

fn inspect(bytes: &[u8]) -> Result<Report> {
    let value = parse_json(bytes).map_err(|_| "GLYD_COSTS_INVALID_JSON")?;
    for pointer in [
        "/build/commit",
        "/model/repo",
        "/model/revision",
        "/totals/working_bytes",
        "/totals/overhead_bytes",
        "/timing/prefill_tok_s",
        "/timing/decode_tok_s",
    ] {
        if value.pointer(pointer).is_none() {
            return Err("GLYD_COSTS_INVALID_FIELDS");
        }
    }
    let doc: Document = serde_json::from_value(value).map_err(|_| "GLYD_COSTS_INVALID_FIELDS")?;
    if doc.object != "glyd.costs" || doc.schema != "glyd-costs/1" {
        return Err("GLYD_COSTS_UNSUPPORTED_SCHEMA");
    }
    for text in [
        Some(doc.build.version.as_str()),
        doc.build.commit.as_deref(),
        Some(doc.model.id.as_str()),
        doc.model.repo.as_deref(),
        doc.model.revision.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        if text.is_empty() || text.len() > 4096 {
            return Err("GLYD_COSTS_INVALID_FIELDS");
        }
    }
    let w = &doc.workload;
    if doc.layers.is_empty()
        || doc.layers.len() > 4096
        || doc.model.layers as usize != doc.layers.len()
        || [
            w.context_tokens,
            w.prompt_tokens,
            w.output_tokens,
            w.prefill_chunk_tokens,
        ]
        .iter()
        .any(|n| !(1..=1_048_576).contains(n))
        || w.prompt_tokens
            .checked_add(w.output_tokens)
            .is_none_or(|n| n > w.context_tokens)
        || w.sessions != 1
        || doc.observed_at == 0
        || doc.samples > 10000
        || matches!(doc.basis.working_memory, WorkingBasis::Measured)
            && doc.totals.working_bytes.is_none()
    {
        return Err("GLYD_COSTS_INVALID_FIELDS");
    }
    for (index, layer) in doc.layers.iter().enumerate() {
        if layer.index as usize != index
            || !["full_attention", "linear_attention", "mamba", "moe"]
                .contains(&layer.kind.as_str())
        {
            return Err("GLYD_COSTS_UNSUPPORTED_LAYERS");
        }
    }
    for rate in [&doc.timing.prefill_tok_s, &doc.timing.decode_tok_s]
        .into_iter()
        .flatten()
    {
        if !rate.value.is_finite()
            || rate.value <= 0.0
            || !["estimated", "measured"].contains(&rate.basis.as_str())
            || rate.placement.is_empty()
            || rate.placement.len() > 128
        {
            return Err("GLYD_COSTS_INVALID_TIMING");
        }
    }
    Ok(Report {
        kind: "GlydCostReceipt",
        source_digest: byte_digest(bytes),
        admission: "unsupported",
        reported: doc,
        unavailable_compute_fields: [
            Blocker::RegisteredManifestBuildRuntimeAndDeviceIdentities,
            Blocker::PhysicalPoolInventoryAndComponentOwnership,
            Blocker::UnverifiedStateBasisAndSeparateStateAllocations,
            Blocker::PerPhaseScratchStagingAndReserves,
            Blocker::MeasuredPhaseTimingsAndReviewedConfidence,
            Blocker::FreshnessRawEvidenceAndLocalApproval,
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn received_costs_preserve_unknowns_and_never_become_admission() {
        let mut input = json!({
            "object": "glyd.costs", "schema": "glyd-costs/1",
            "build": {"version": "synthetic", "commit": "a".repeat(40), "development": true},
            "model": {"id": "synthetic.invalid", "repo": null, "revision": null, "layers": 1},
            "basis": {"sizes": "file", "packs_from_files": 1, "packs_estimated": 0,
                "kv_and_state": "formula", "working_memory": "unavailable"},
            "workload": {"context_tokens": 4096, "prompt_tokens": 1024, "output_tokens": 256,
                "prefill_chunk_tokens": 2048, "sessions": 1},
            "observed_at": 1, "samples": 0,
            "layers": [{"index": 0, "kind": "mamba", "resident_bytes": {"total": 10},
                "active_bytes_per_token": 10, "kv_bytes_per_token": 0, "state_bytes_per_session": 8}],
            "totals": {"resident_bytes": 10, "kv_bytes_per_token": 0, "kv_bytes_at_context": 0,
                "state_bytes_per_session": 8, "working_bytes": null, "overhead_bytes": null,
                "engine_bytes_at_context": 18},
            "timing": {"prefill_tok_s": null,
                "decode_tok_s": {"value": 1.25, "basis": "estimated", "placement": "vram"}},
            "device": {"not_a_compute_device_identity": true}
        });
        let bytes = serde_json::to_vec(&input).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("synthetic-costs.json");
        assert!(inspect_file(&path).is_err());
        std::fs::write(&path, &bytes).unwrap();
        assert_eq!(
            inspect_file(&path).unwrap().source_digest,
            byte_digest(&bytes)
        );
        let report = inspect(&bytes).unwrap();
        assert_eq!(report.source_digest, byte_digest(&bytes));
        assert_eq!(report.admission, "unsupported");
        assert_eq!(report.reported.layers[0].kind, "mamba");
        assert_eq!(report.reported.layers[0].state_bytes_per_session, 8);
        assert_eq!(report.reported.workload.prefill_chunk_tokens, 2048);
        assert_eq!(report.reported.samples, 0);
        assert_eq!(
            report.reported.timing.decode_tok_s.unwrap().basis,
            "estimated"
        );
        assert!(report.reported.totals.working_bytes.is_none());
        assert!(report.reported.totals.overhead_bytes.is_none());
        let projected = serde_json::to_value(inspect(&bytes).unwrap()).unwrap();
        assert!(projected["reported"].get("device").is_none());
        assert!(projected.get("key").is_none());
        assert_eq!(
            projected["reported"]["totals"]["overhead_bytes"],
            json!(null)
        );
        assert_eq!(
            projected["unavailable_compute_fields"]
                .as_array()
                .unwrap()
                .len(),
            6
        );
        input["workload"]["prefill_chunk_tokens"] = json!(8192);
        assert_eq!(
            inspect(&serde_json::to_vec(&input).unwrap())
                .unwrap()
                .reported
                .workload
                .prefill_chunk_tokens,
            8192
        );
        input["workload"]["prefill_chunk_tokens"] = json!(2048);
        for pointer in [
            "/build/commit",
            "/model/repo",
            "/model/revision",
            "/totals/working_bytes",
            "/totals/overhead_bytes",
            "/timing/prefill_tok_s",
            "/timing/decode_tok_s",
        ] {
            let mut explicit_null = input.clone();
            *explicit_null.pointer_mut(pointer).unwrap() = json!(null);
            assert!(inspect(&serde_json::to_vec(&explicit_null).unwrap()).is_ok());
            let (parent, key) = pointer.rsplit_once('/').unwrap();
            explicit_null
                .pointer_mut(parent)
                .unwrap()
                .as_object_mut()
                .unwrap()
                .remove(key);
            assert_eq!(
                inspect(&serde_json::to_vec(&explicit_null).unwrap()).unwrap_err(),
                "GLYD_COSTS_INVALID_FIELDS"
            );
        }
        for (pointer, bad) in [
            ("/schema", json!("glyd-costs/2")),
            ("/object", json!("other")),
            ("/layers/0/index", json!(1)),
            ("/layers/0/kind", json!("unknown")),
            ("/layers/0/resident_bytes/total", json!(-1)),
            ("/layers/0/resident_bytes/total", json!(1.5)),
            ("/model/layers", json!(2)),
            ("/workload/output_tokens", json!(4096)),
            ("/workload/prefill_chunk_tokens", json!(0)),
            ("/workload/sessions", json!(2)),
            ("/timing/decode_tok_s/value", json!(0)),
            ("/timing/decode_tok_s/basis", json!("file")),
            ("/basis/working_memory", json!("measured")),
        ] {
            let old = input.pointer_mut(pointer).unwrap().take();
            *input.pointer_mut(pointer).unwrap() = bad;
            assert!(
                inspect(&serde_json::to_vec(&input).unwrap()).is_err(),
                "{pointer}"
            );
            *input.pointer_mut(pointer).unwrap() = old;
        }
        assert!(inspect(b"{\"schema\":1,\"schema\":1}").is_err());
        assert!(inspect(b"[]").is_err());
        assert!(inspect(b"{}").is_err());
        assert!(inspect(&vec![b' '; MAX_JSON_BYTES + 1]).is_err());
    }
}
