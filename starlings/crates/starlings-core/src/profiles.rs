//! Pure cost accounting for the proposed InspectModel contract. Does not reserve or launch work.
use crate::CoreError;
use serde::{Deserialize, Serialize};
use starlings_protocol::*;
use std::ops::Range;

/// Receiver-owned wall/monotonic observations. A replay cannot reset the profile timestamp.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileAge {
    pub captured_wall_ms: i64,
    pub now_wall_ms: i64,
    pub captured_at_ms: u64,
    pub now_ms: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileStatus {
    Unavailable,
    Stale,
    FixtureOnly,
    Estimated,
    Unqualified,
    Qualified,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PoolBudget {
    pub pool_id: String,
    pub resident_weight_bytes: Decimal,
    pub resident_overhead_bytes: Decimal,
    pub state_bytes: Decimal,
    pub prefill_scratch_bytes: Decimal,
    pub decode_scratch_bytes: Decimal,
    pub staging_bytes: Decimal,
    pub reserve_bytes: Decimal,
    pub total_bytes: Decimal,
    pub active_weight_bytes_per_token: Decimal,
}
#[derive(Debug, Serialize)]
pub struct ProfileAssessment {
    pub profile_digest: Digest,
    pub status: ProfileStatus,
    pub unavailable_code: Option<String>,
    pub age_ms: Option<Decimal>,
    pub measurement: Option<CostMeasurement>,
    pub pools: Vec<PoolBudget>,
    pub timing: Option<PhaseCost>,
}
/// `approved` must come from trusted local qualification of the ENTIRE profile, not its own report.
/// `expected` includes the exact workload and actual device/pool inventory identities.
/// A qualified cost profile alone does not qualify layer execution or admit a plan.
pub fn assess(
    profile: &ModelCostProfile,
    manifest: &ModelManifest,
    expected: &ProfileKey,
    range: Range<usize>,
    age: ProfileAge,
    approved: Option<&Digest>,
    mode: FixtureMode,
) -> Result<ProfileAssessment, CoreError> {
    encode(profile, mode).map_err(|_| CoreError::Invalid)?;
    encode(manifest, mode).map_err(|_| CoreError::Invalid)?;
    profile
        .validate_manifest(manifest)
        .map_err(|_| CoreError::Invalid)?;
    if &profile.key != expected || range.start >= range.end || range.end > manifest.layers.len() {
        return Err(CoreError::Invalid);
    }
    let elapsed = age
        .now_ms
        .checked_sub(age.captured_at_ms)
        .ok_or(CoreError::Clock)?;
    if age.now_wall_ms < age.captured_wall_ms {
        return Err(CoreError::Clock);
    }
    let digest = content_hash(profile).map_err(|_| CoreError::Invalid)?;
    let mut assessment = ProfileAssessment {
        profile_digest: digest.clone(),
        status: ProfileStatus::Unavailable,
        unavailable_code: None,
        age_ms: None,
        measurement: None,
        pools: vec![],
        timing: None,
    };
    let ModelCosts::Available { measurement, .. } = &profile.costs else {
        if let ModelCosts::Unavailable { code } = &profile.costs {
            assessment.unavailable_code = Some(code.clone());
        }
        return Ok(assessment);
    };
    let measured_ms = time::OffsetDateTime::parse(
        &measurement.observed_at,
        &time::format_description::well_known::Rfc3339,
    )
    .map_err(|_| CoreError::Invalid)?
    .unix_timestamp_nanos()
        / 1_000_000;
    let at_capture = u64::try_from(i128::from(age.captured_wall_ms) - measured_ms)
        .map_err(|_| CoreError::Clock)?;
    let wall_age =
        u64::try_from(i128::from(age.now_wall_ms) - measured_ms).map_err(|_| CoreError::Clock)?;
    let age_ms = at_capture
        .checked_add(elapsed)
        .ok_or(CoreError::Clock)?
        .max(wall_age);
    assessment.age_ms = Some(Decimal(age_ms));
    assessment.measurement = Some(measurement.clone());
    assessment.status = if age_ms >= u64::from(measurement.valid_for_ms.get()) {
        ProfileStatus::Stale
    } else if profile.fixture {
        ProfileStatus::FixtureOnly
    } else if measurement.basis == CostBasis::Estimated {
        ProfileStatus::Estimated
    } else if approved != Some(&digest) {
        ProfileStatus::Unqualified
    } else {
        ProfileStatus::Qualified
    };
    (assessment.pools, assessment.timing) = stage_costs(profile, range)?;
    Ok(assessment)
}
/// Internal only: callers first validate/bind the complete profile once for a captured preview.
pub(crate) fn stage_costs(
    profile: &ModelCostProfile,
    range: Range<usize>,
) -> Result<(Vec<PoolBudget>, Option<PhaseCost>), CoreError> {
    let ModelCosts::Available {
        layers,
        components,
        per_stage,
        per_stage_timing,
        ..
    } = &profile.costs
    else {
        return Err(CoreError::Invalid);
    };
    if range.start >= range.end || range.end > layers.len() {
        return Err(CoreError::Invalid);
    }
    let expected = &profile.key;
    let mut budgets = Vec::new();
    // Sum scratch costs conservatively: overlapping allocator lifetimes are not inferred.
    let mut pools = per_stage.clone();
    let mut timing = per_stage_timing.clone();
    for (costs, phase) in layers
        .get(range.clone())
        .ok_or(CoreError::Invalid)?
        .iter()
        .map(|l| (&l.pools, &l.timing))
        .chain(
            components
                .iter()
                .filter(|c| {
                    c.disposition == ComponentDisposition::Executed
                        && match c.owner_role {
                            ComponentOwnerRole::First => range.start == 0,
                            ComponentOwnerRole::Last => range.end == layers.len(),
                            ComponentOwnerRole::None => false,
                        }
                })
                .map(|c| (&c.pools, &c.timing)),
        )
    {
        for (total, cost) in pools.iter_mut().zip(costs) {
            add(total, cost)?;
        }
        timing = match (timing, phase) {
            (Some(a), Some(b)) => Some(PhaseCost {
                prefill_us: Decimal(sum(a.prefill_us.0, b.prefill_us.0)?),
                decode_us_per_token: Decimal(sum(
                    a.decode_us_per_token.0,
                    b.decode_us_per_token.0,
                )?),
            }),
            _ => None,
        };
    }
    for p in pools {
        let per_session = sum(
            p.kv_bytes_per_token
                .0
                .checked_mul(u64::from(expected.workload.context_tokens.get()))
                .ok_or(CoreError::Limit)?,
            sum(
                p.recurrent_bytes_per_session.0,
                p.convolution_bytes_per_session.0,
            )?,
        )?;
        let state = per_session
            .checked_mul(u64::from(expected.workload.sessions.get()))
            .ok_or(CoreError::Limit)?;
        let total = sum(
            sum(
                sum(p.resident_weight_bytes.0, p.resident_overhead_bytes.0)?,
                state,
            )?,
            sum(
                sum(p.prefill_scratch_bytes.0, p.decode_scratch_bytes.0)?,
                sum(p.staging_bytes.0, p.reserve_bytes.0)?,
            )?,
        )?;
        budgets.push(PoolBudget {
            pool_id: p.pool_id,
            resident_weight_bytes: p.resident_weight_bytes,
            resident_overhead_bytes: p.resident_overhead_bytes,
            state_bytes: Decimal(state),
            prefill_scratch_bytes: p.prefill_scratch_bytes,
            decode_scratch_bytes: p.decode_scratch_bytes,
            staging_bytes: p.staging_bytes,
            reserve_bytes: p.reserve_bytes,
            total_bytes: Decimal(total),
            active_weight_bytes_per_token: p.active_weight_bytes_per_token,
        });
    }
    Ok((budgets, timing))
}

fn sum(a: u64, b: u64) -> Result<u64, CoreError> {
    a.checked_add(b).ok_or(CoreError::Limit)
}
fn add(a: &mut PoolCost, b: &PoolCost) -> Result<(), CoreError> {
    macro_rules! field {
        ($name:ident) => {
            a.$name.0 = sum(a.$name.0, b.$name.0)?;
        };
    }
    field!(resident_weight_bytes);
    field!(resident_overhead_bytes);
    field!(active_weight_bytes_per_token);
    field!(kv_bytes_per_token);
    field!(recurrent_bytes_per_session);
    field!(convolution_bytes_per_session);
    field!(prefill_scratch_bytes);
    field!(decode_scratch_bytes);
    field!(staging_bytes);
    field!(reserve_bytes);
    Ok(())
}
