//! Offline advisory input, never a registry snapshot, approval or executable plan.
use starlings_core::planner::{Preview, PreviewInput, preview};
use starlings_protocol::{FixtureMode, MAX_JSON_BYTES, parse_json};
use std::{fs::File, path::Path};

pub fn modes_file(
    path: &Path,
    mode: FixtureMode,
) -> Result<starlings_core::modes::ModeReport, &'static str> {
    let bytes = crate::scoreboard::read(
        File::open(path).map_err(|_| "MODES_READ_FAILED")?,
        MAX_JSON_BYTES,
    )
    .map_err(|_| "MODES_READ_FAILED")?;
    let input: starlings_core::modes::ModeInput =
        serde_json::from_value(parse_json(&bytes).map_err(|_| "MODES_INVALID_JSON")?)
            .map_err(|_| "MODES_INVALID_INPUT")?;
    starlings_core::modes::choose(&input, mode).map_err(|_| "MODES_REFUSED")
}

pub fn preview_file(path: &Path, mode: FixtureMode) -> Result<Preview, &'static str> {
    let bytes = crate::scoreboard::read(
        File::open(path).map_err(|_| "PREVIEW_READ_FAILED")?,
        MAX_JSON_BYTES,
    )
    .map_err(|_| "PREVIEW_READ_FAILED")?;
    let input: PreviewInput =
        serde_json::from_value(parse_json(&bytes).map_err(|_| "PREVIEW_INVALID_JSON")?)
            .map_err(|_| "PREVIEW_INVALID_INPUT")?;
    preview(&input, mode).map_err(|_| "PREVIEW_REFUSED")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    #[test]
    fn cost_profile_nullable_fields_must_be_present_in_all_file_consumers() {
        let source: Value =
            serde_json::from_slice(include_bytes!("../../../examples/mode-preview.json")).unwrap();
        let profile = &source["variants"][0]["placement"]["nodes"][0]["profile"];
        let mut fields = vec![
            ("/costs/measurement".to_string(), "relative_error_bps"),
            ("/costs".to_string(), "per_stage_timing"),
        ];
        for rows in ["layers", "components"] {
            for index in 0..profile["costs"][rows].as_array().unwrap().len() {
                fields.push((format!("/costs/{rows}/{index}"), "timing"));
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("input.json");
        for (parent, field) in fields {
            for missing in [false, true] {
                let mut profile = profile.clone();
                profile
                    .pointer_mut(&parent)
                    .unwrap()
                    .as_object_mut()
                    .unwrap()
                    .insert(field.into(), Value::Null);
                let typed: starlings_protocol::ModelCostProfile =
                    serde_json::from_value(profile.clone()).unwrap();
                let digest = starlings_protocol::content_hash(&typed).unwrap();
                assert_eq!(
                    serde_json::to_value(typed)
                        .unwrap()
                        .pointer(&format!("{parent}/{field}")),
                    Some(&Value::Null)
                );
                if missing {
                    profile
                        .pointer_mut(&parent)
                        .unwrap()
                        .as_object_mut()
                        .unwrap()
                        .remove(field)
                        .unwrap();
                }
                let mut modes = source.clone();
                modes["variants"][0]["placement"]["nodes"][0]["profile"] = profile;
                for rows in ["tiers", "storage"] {
                    for rate in modes["variants"][0]["costs"][rows].as_array_mut().unwrap() {
                        rate["profile_digest"] = json!(digest);
                    }
                }
                std::fs::write(&path, serde_json::to_vec(&modes).unwrap()).unwrap();
                let result = modes_file(&path, FixtureMode::Simulation);
                if missing {
                    assert_eq!(
                        result.err(),
                        Some("MODES_INVALID_INPUT"),
                        "{parent}/{field}"
                    );
                } else {
                    assert!(!result.unwrap().admission);
                }

                let placement = &modes["variants"][0]["placement"];
                std::fs::write(&path, serde_json::to_vec(placement).unwrap()).unwrap();
                let result = preview_file(&path, FixtureMode::Simulation);
                if missing {
                    assert_eq!(
                        result.err(),
                        Some("PREVIEW_INVALID_INPUT"),
                        "{parent}/{field}"
                    );
                } else {
                    assert!(!result.unwrap().admission);
                }

                let calibration = json!({
                    "version": 1, "fixture": true, "placement": placement,
                    "candidate_index": 0, "source": null, "artifacts": [], "observations": []
                });
                std::fs::write(&path, serde_json::to_vec(&calibration).unwrap()).unwrap();
                let result = crate::calibration::report_file(&path, FixtureMode::Simulation);
                if missing {
                    assert_eq!(
                        result.err(),
                        Some("CALIBRATION_INVALID_INPUT"),
                        "{parent}/{field}"
                    );
                } else {
                    // Explicitly unknown timing can remove the candidate, but is valid input.
                    assert!(matches!(
                        result.map(|r| r.admission),
                        Ok(false) | Err("CALIBRATION_CANDIDATE_MISSING")
                    ));
                }
            }
        }
    }
}
