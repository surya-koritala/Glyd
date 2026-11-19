//! Offline comparison of planner estimates and supplied observations. No collection or admission.
use crate::scoreboard::{self, Artifact};
use serde::{Deserialize, Serialize};
use starlings_core::planner::{self, PreviewInput};
use starlings_protocol::{
    BoundedU32, Decimal, Digest, FixtureMode, InvocationId, Version, content_hash, parse_json,
};
use std::{collections::BTreeSet, fs::File, path::Path};

const DEFINITION: &str = "glyd-plan-calibration-v1";
type Result<T> = std::result::Result<T, &'static str>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    LoadSerial,
    PrefillPath,
    DecodePathPerToken,
    BottleneckPerToken,
}
impl Metric {
    const ALL: [Self; 4] = [
        Self::LoadSerial,
        Self::PrefillPath,
        Self::DecodePathPerToken,
        Self::BottleneckPerToken,
    ];
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum Outcome {
    Observed { elapsed_us: Decimal },
    Failed { code: String },
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub id: InvocationId,
    pub fixture: bool,
    pub metric: Metric,
    pub binding: Digest,
    pub raw: Digest,
    pub outcome: Outcome,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceSource {
    pub execution_plan: Digest,
    pub producer: Digest,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub version: Version,
    pub fixture: bool,
    pub placement: PreviewInput,
    pub candidate_index: BoundedU32<0, 15>,
    pub source: Option<EvidenceSource>,
    pub artifacts: Vec<Artifact>,
    pub observations: Vec<Observation>,
}
#[derive(Serialize)]
pub struct Deviation {
    pub observed_above_prediction: bool,
    pub magnitude_us: Decimal,
    pub relative_error_ratio: Option<f64>,
}
#[derive(Serialize)]
pub struct Compared {
    pub id: InvocationId,
    pub raw: Digest,
    pub outcome: Outcome,
    pub deviation: Option<Deviation>,
}
#[derive(Serialize)]
pub struct MetricReport {
    pub metric: Metric,
    pub binding: Digest,
    pub predicted_us: Decimal,
    pub observations: Vec<Compared>,
}
#[derive(Serialize)]
pub struct Report {
    pub version: Version,
    pub fixture: bool,
    pub qualification: &'static str,
    pub admission: bool,
    pub definition: &'static str,
    pub input_digest: Digest,
    pub placement_digest: Digest,
    pub candidate_digest: Digest,
    pub source: Option<EvidenceSource>,
    pub verified_artifacts: usize,
    pub metrics: Vec<MetricReport>,
}

pub fn report_file(path: &Path, mode: FixtureMode) -> Result<Report> {
    let path = path.canonicalize().map_err(|_| "CALIBRATION_READ_FAILED")?;
    let bytes = scoreboard::read(
        File::open(&path).map_err(|_| "CALIBRATION_READ_FAILED")?,
        starlings_protocol::MAX_JSON_BYTES,
    )
    .map_err(|_| "CALIBRATION_READ_FAILED")?;
    let input: Input =
        serde_json::from_value(parse_json(&bytes).map_err(|_| "CALIBRATION_INVALID_JSON")?)
            .map_err(|_| "CALIBRATION_INVALID_INPUT")?;
    compare(
        &input,
        path.parent().ok_or("CALIBRATION_READ_FAILED")?,
        mode,
    )
}
fn compare(input: &Input, root: &Path, mode: FixtureMode) -> Result<Report> {
    if input.fixture != input.placement.fixture || (input.fixture && mode == FixtureMode::Reject) {
        return Err("CALIBRATION_FIXTURE_REFUSED");
    }
    if input.observations.len() > 512 {
        return Err("CALIBRATION_OBSERVATION_LIMIT");
    }
    let preview =
        planner::preview(&input.placement, mode).map_err(|_| "CALIBRATION_PLACEMENT_REFUSED")?;
    let candidate = preview
        .candidates
        .get(input.candidate_index.get() as usize)
        .ok_or("CALIBRATION_CANDIDATE_MISSING")?;
    let digest = content_hash(candidate).map_err(|_| "CALIBRATION_BINDING_FAILED")?;
    let hashes = scoreboard::artifacts(&input.artifacts, root)?;
    if let Some(source) = &input.source {
        if !hashes.contains(&source.execution_plan) || !hashes.contains(&source.producer) {
            return Err("CALIBRATION_SOURCE_EVIDENCE_MISSING");
        }
    } else if !input.observations.is_empty() {
        return Err("CALIBRATION_SOURCE_REQUIRED");
    }
    let mut metrics = Vec::new();
    let mut ids = BTreeSet::new();
    for metric in Metric::ALL {
        let predicted_us = match metric {
            Metric::LoadSerial => candidate.load_us,
            Metric::PrefillPath => candidate.prefill_path_us,
            Metric::DecodePathPerToken => candidate.decode_path_us,
            Metric::BottleneckPerToken => candidate.bottleneck_us,
        };
        let binding = content_hash(&(
            DEFINITION,
            input.fixture,
            &preview.input_digest,
            &digest,
            &input.source,
            metric,
        ))
        .map_err(|_| "CALIBRATION_BINDING_FAILED")?;
        let mut observations = Vec::new();
        for observation in input.observations.iter().filter(|o| o.metric == metric) {
            if observation.fixture != input.fixture || observation.binding != binding {
                return Err("CALIBRATION_UNMATCHED_OBSERVATION");
            }
            if !ids.insert((observation.id, metric)) {
                return Err("CALIBRATION_DUPLICATE_OBSERVATION");
            }
            if !hashes.contains(&observation.raw) {
                return Err("CALIBRATION_EVIDENCE_MISSING");
            }
            let deviation = match &observation.outcome {
                Outcome::Observed { elapsed_us } => {
                    let magnitude = elapsed_us.0.abs_diff(predicted_us.0);
                    Some(Deviation {
                        observed_above_prediction: elapsed_us.0 > predicted_us.0,
                        magnitude_us: Decimal(magnitude),
                        relative_error_ratio: (predicted_us.0 != 0)
                            .then(|| magnitude as f64 / predicted_us.0 as f64),
                    })
                }
                Outcome::Failed { code } => {
                    if code.is_empty()
                        || code.len() > 64
                        || !code
                            .bytes()
                            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
                    {
                        return Err("CALIBRATION_INVALID_FAILURE_CODE");
                    }
                    None
                }
            };
            observations.push(Compared {
                id: observation.id,
                raw: observation.raw.clone(),
                outcome: observation.outcome.clone(),
                deviation,
            });
        }
        metrics.push(MetricReport {
            metric,
            binding,
            predicted_us,
            observations,
        });
    }
    Ok(Report {
        version: Version,
        fixture: input.fixture,
        qualification: "not_assessed",
        admission: false,
        definition: DEFINITION,
        input_digest: content_hash(input).map_err(|_| "CALIBRATION_BINDING_FAILED")?,
        placement_digest: preview.input_digest,
        candidate_digest: digest,
        source: input.source.clone(),
        verified_artifacts: hashes.len(),
        metrics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use starlings_protocol::byte_digest;

    fn input() -> Input {
        let value: serde_json::Value =
            serde_json::from_slice(include_bytes!("../../../examples/mode-preview.json")).unwrap();
        Input {
            version: Version,
            fixture: true,
            placement: serde_json::from_value(value["variants"][0]["placement"].clone()).unwrap(),
            candidate_index: BoundedU32::new(0).unwrap(),
            source: None,
            artifacts: vec![],
            observations: vec![],
        }
    }
    fn fixture() -> (tempfile::TempDir, Input) {
        let dir = tempfile::tempdir().unwrap();
        let mut input = input();
        input.placement.nodes[0].load_us = Decimal(100);
        for name in ["success.raw", "failure.raw", "plan.json", "producer.json"] {
            let bytes = format!("synthetic {name}");
            std::fs::write(dir.path().join(name), &bytes).unwrap();
            input.artifacts.push(Artifact {
                path: name.into(),
                sha256: byte_digest(bytes.as_bytes()),
            });
        }
        input.source = Some(EvidenceSource {
            execution_plan: input.artifacts[2].sha256.clone(),
            producer: input.artifacts[3].sha256.clone(),
        });
        let empty = compare(&input, dir.path(), FixtureMode::Simulation).unwrap();
        for (i, outcome) in [
            Outcome::Observed {
                elapsed_us: Decimal(empty.metrics[0].predicted_us.0 + 25),
            },
            Outcome::Failed {
                code: "TIMEOUT".into(),
            },
        ]
        .into_iter()
        .enumerate()
        {
            input.observations.push(Observation {
                id: format!("00000000-0000-4000-8000-{:012x}", i + 1)
                    .parse()
                    .unwrap(),
                fixture: true,
                metric: Metric::LoadSerial,
                binding: empty.metrics[0].binding.clone(),
                raw: input.artifacts[i].sha256.clone(),
                outcome,
            });
        }
        (dir, input)
    }
    #[test]
    fn retains_failed_attempts_and_distinguishes_estimates_from_observations() {
        let (dir, input) = fixture();
        let report = compare(&input, dir.path(), FixtureMode::Simulation).unwrap();
        assert_eq!(report.qualification, "not_assessed");
        assert!(!report.admission);
        assert!(report.fixture);
        assert_eq!(report.metrics.len(), 4);
        assert_eq!(report.metrics[0].observations.len(), 2);
        let deviation = report.metrics[0].observations[0]
            .deviation
            .as_ref()
            .unwrap();
        assert!(deviation.observed_above_prediction);
        assert_eq!(deviation.magnitude_us, Decimal(25));
        assert_eq!(
            deviation.relative_error_ratio,
            Some(25.0 / report.metrics[0].predicted_us.0 as f64)
        );
        assert!(report.metrics[0].observations[1].deviation.is_none());
        assert!(
            matches!(&report.metrics[0].observations[1].outcome,Outcome::Failed{code} if code=="TIMEOUT")
        );
        assert!(
            report.metrics[1..]
                .iter()
                .all(|m| m.observations.is_empty())
        );
    }
    #[test]
    fn refuses_changed_bindings_fixtures_duplicates_and_nonredacted_errors() {
        let (dir, input) = fixture();
        assert_eq!(
            compare(&input, dir.path(), FixtureMode::Reject).err(),
            Some("CALIBRATION_FIXTURE_REFUSED")
        );
        for change in 0..8 {
            let mut bad = input.clone();
            match change {
                0 => bad.placement.nodes[0].load_us.0 += 1,
                1 => bad.observations[0].binding = byte_digest(b"wrong candidate"),
                2 => bad.observations[0].fixture = false,
                3 => bad.observations.push(bad.observations[0].clone()),
                4 => {
                    bad.observations[1].outcome = Outcome::Failed {
                        code: "unredacted message".into(),
                    }
                }
                5 => bad.observations[0].raw = byte_digest(b"unreferenced evidence"),
                6 => bad.source = None,
                _ => {
                    bad.source.as_mut().unwrap().execution_plan =
                        byte_digest(b"absent plan receipt")
                }
            }
            let code = compare(&bad, dir.path(), FixtureMode::Simulation)
                .err()
                .unwrap();
            assert_eq!(
                code,
                match change {
                    0..=2 => "CALIBRATION_UNMATCHED_OBSERVATION",
                    3 => "CALIBRATION_DUPLICATE_OBSERVATION",
                    4 => "CALIBRATION_INVALID_FAILURE_CODE",
                    5 => "CALIBRATION_EVIDENCE_MISSING",
                    6 => "CALIBRATION_SOURCE_REQUIRED",
                    _ => "CALIBRATION_SOURCE_EVIDENCE_MISSING",
                }
            );
        }
        let mut many = input.clone();
        many.observations = vec![input.observations[0].clone(); 513];
        assert_eq!(
            compare(&many, dir.path(), FixtureMode::Simulation).err(),
            Some("CALIBRATION_OBSERVATION_LIMIT")
        );
    }
    #[test]
    fn zero_prediction_extreme_values_and_missing_phases_remain_explicit() {
        let (dir, mut input) = fixture();
        input.observations.clear();
        input.placement.nodes[0].load_us = Decimal(0);
        let empty = compare(&input, dir.path(), FixtureMode::Simulation).unwrap();
        assert!(empty.metrics.iter().all(|m| m.observations.is_empty()));
        std::fs::write(dir.path().join("extreme.raw"), b"synthetic extreme").unwrap();
        let raw = byte_digest(b"synthetic extreme");
        input.artifacts.push(Artifact {
            path: "extreme.raw".into(),
            sha256: raw.clone(),
        });
        input.observations.push(Observation {
            id: "00000000-0000-4000-8000-000000000001".parse().unwrap(),
            fixture: true,
            metric: Metric::LoadSerial,
            binding: empty.metrics[0].binding.clone(),
            raw,
            outcome: Outcome::Observed {
                elapsed_us: Decimal(u64::MAX),
            },
        });
        let report = compare(&input, dir.path(), FixtureMode::Simulation).unwrap();
        let deviation = report.metrics[0].observations[0]
            .deviation
            .as_ref()
            .unwrap();
        assert_eq!(deviation.magnitude_us, Decimal(u64::MAX));
        assert_eq!(deviation.relative_error_ratio, None);
        serde_json::to_vec(&report).unwrap();
    }
    #[test]
    fn file_reader_checks_strict_json_and_evidence_integrity() {
        let (dir, input) = fixture();
        let path = dir.path().join("input.json");
        std::fs::write(&path, serde_json::to_vec(&input).unwrap()).unwrap();
        report_file(&path, FixtureMode::Simulation).unwrap();
        std::fs::write(dir.path().join("success.raw"), b"tampered").unwrap();
        assert_eq!(
            report_file(&path, FixtureMode::Simulation).err(),
            Some("SCOREBOARD_ARTIFACT_MISMATCH")
        );
        std::fs::write(&path, b"{\"version\":1,\"version\":1}").unwrap();
        assert_eq!(
            report_file(&path, FixtureMode::Simulation).err(),
            Some("CALIBRATION_INVALID_JSON")
        );
        let mut value = serde_json::to_value(&input).unwrap();
        value["unknown"] = serde_json::json!(true);
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        assert_eq!(
            report_file(&path, FixtureMode::Simulation).err(),
            Some("CALIBRATION_INVALID_INPUT")
        );
    }
}
