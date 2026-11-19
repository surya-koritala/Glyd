//! Proposed external InspectModel result. Validation is not backend qualification.
use crate::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileWorkload {
    pub context_tokens: BoundedU32<1, 1048576>,
    pub prompt_tokens: BoundedU32<1, 1048576>,
    pub output_tokens: BoundedU32<1, 1048576>,
    pub prefill_chunk_tokens: BoundedU32<1, 1048576>,
    pub sessions: BoundedU32<1, 256>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileKey {
    pub manifest_digest: Digest,
    pub execution_profile: String,
    pub boundary_profile: String,
    pub backend_build_digest: Digest,
    pub runtime_config_digest: Digest,
    pub device_digest: Digest,
    pub pool_inventory_digest: Digest,
    pub workload: ProfileWorkload,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostBasis {
    Estimated,
    Measured,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CostMeasurement {
    pub basis: CostBasis,
    pub evidence_digest: Digest,
    pub observed_at: String,
    pub valid_for_ms: BoundedU32<1, 86400000>,
    pub samples: BoundedU32<0, 10000>,
    #[serde(deserialize_with = "Option::deserialize")]
    pub relative_error_bps: Option<BoundedU32<0, 10000>>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoolCost {
    pub pool_id: String,
    pub resident_weight_bytes: Decimal,
    pub resident_overhead_bytes: Decimal,
    pub active_weight_bytes_per_token: Decimal,
    pub kv_bytes_per_token: Decimal,
    pub recurrent_bytes_per_session: Decimal,
    pub convolution_bytes_per_session: Decimal,
    pub prefill_scratch_bytes: Decimal,
    pub decode_scratch_bytes: Decimal,
    pub staging_bytes: Decimal,
    pub reserve_bytes: Decimal,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseCost {
    pub prefill_us: Decimal,
    pub decode_us_per_token: Decimal,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayerCost {
    pub index: BoundedU32<0, 4095>,
    pub layer_type: LayerType,
    pub state_profile: String,
    pub pools: Vec<PoolCost>,
    #[serde(deserialize_with = "Option::deserialize")]
    pub timing: Option<PhaseCost>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComponentCost {
    pub name: String,
    pub disposition: ComponentDisposition,
    pub owner_role: ComponentOwnerRole,
    pub pools: Vec<PoolCost>,
    #[serde(deserialize_with = "Option::deserialize")]
    pub timing: Option<PhaseCost>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelCosts {
    Unavailable {
        code: String,
    },
    Available {
        measurement: CostMeasurement,
        pool_ids: Vec<String>,
        layers: Vec<LayerCost>,
        components: Vec<ComponentCost>,
        per_stage: Vec<PoolCost>,
        #[serde(deserialize_with = "Option::deserialize")]
        per_stage_timing: Option<PhaseCost>,
    },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelCostProfile {
    pub kind: String,
    pub version: Version,
    pub fixture: bool,
    pub key: ProfileKey,
    pub costs: ModelCosts,
}
impl ModelCostProfile {
    pub(crate) fn validate_costs(&self) -> Result<(), Error> {
        let w = &self.key.workload;
        if w.prompt_tokens
            .get()
            .checked_add(w.output_tokens.get())
            .is_none_or(|n| n > w.context_tokens.get())
            || w.prefill_chunk_tokens.get() > w.prompt_tokens.get()
        {
            return Err(Error("invalid profile workload"));
        }
        let ModelCosts::Available {
            measurement,
            pool_ids,
            layers,
            components,
            per_stage,
            per_stage_timing,
        } = &self.costs
        else {
            return Ok(());
        };
        if measurement.basis == CostBasis::Measured
            && (measurement.samples.get() == 0 || measurement.relative_error_bps.is_none())
        {
            return Err(Error("measured confidence required"));
        }
        let ids: BTreeSet<_> = pool_ids.iter().collect();
        if ids.len() != pool_ids.len() {
            return Err(Error("duplicate profile pool"));
        }
        let validate = |pools: &[PoolCost], timing: &Option<PhaseCost>| {
            if pools.len() != pool_ids.len()
                || pools.iter().zip(pool_ids).any(|(p, id)| &p.pool_id != id)
                || timing
                    .as_ref()
                    .is_some_and(|t| t.prefill_us.0 == 0 || t.decode_us_per_token.0 == 0)
            {
                return Err(Error("invalid profile costs"));
            }
            Ok(())
        };
        validate(per_stage, per_stage_timing)?;
        for (i, layer) in layers.iter().enumerate() {
            if layer.index.get() as usize != i {
                return Err(Error("invalid cost layer coverage"));
            }
            validate(&layer.pools, &layer.timing)?;
            if (layer.layer_type == LayerType::FullAttention
                && !layer.pools.iter().any(|p| p.kv_bytes_per_token.0 > 0))
                || (layer.layer_type == LayerType::LinearAttention
                    && !layer.pools.iter().any(|p| {
                        p.recurrent_bytes_per_session.0 > 0 || p.convolution_bytes_per_session.0 > 0
                    }))
            {
                return Err(Error("missing attention state costs"));
            }
        }
        let mut names = BTreeSet::new();
        for component in components {
            if !names.insert(&component.name) {
                return Err(Error("duplicate cost component"));
            }
            validate(&component.pools, &component.timing)?;
            match component.disposition {
                ComponentDisposition::Executed
                    if component.owner_role != ComponentOwnerRole::None => {}
                ComponentDisposition::NotInScope
                    if component.owner_role == ComponentOwnerRole::None
                        && component.timing.is_none()
                        && component.pools.iter().all(PoolCost::is_zero) => {}
                _ => return Err(Error("unsupported or unowned component costs")),
            }
        }
        Ok(())
    }
    /// Full immutable manifest coverage; no omitted vision/MTP/head and no inferred compression factor.
    pub fn validate_manifest(&self, manifest: &ModelManifest) -> Result<(), Error> {
        self.validate()?;
        manifest.validate()?;
        if self.fixture != manifest.fixture
            || self.key.manifest_digest != manifest.manifest_digest
            || self.key.execution_profile != manifest.execution_profile
            || self.key.boundary_profile != manifest.required_boundary_profile
        {
            return Err(Error("profile manifest mismatch"));
        }
        if let ModelCosts::Available {
            layers, components, ..
        } = &self.costs
            && (layers.len() != manifest.layers.len()
                || components.len() != manifest.non_layer_components.len()
                || layers.iter().zip(&manifest.layers).any(|(a, b)| {
                    a.index != b.index
                        || a.layer_type != b.layer_type
                        || a.state_profile != b.state_profile
                        || (b.resident_weight_bytes.0 > 0
                            && !a.pools.iter().any(|p| p.resident_weight_bytes.0 > 0))
                })
                || components
                    .iter()
                    .zip(&manifest.non_layer_components)
                    .any(|(a, b)| {
                        a.name != b.name
                            || a.disposition != b.disposition
                            || a.owner_role != b.owner_role
                            || (b.disposition == ComponentDisposition::Executed
                                && b.resident_weight_bytes.0 > 0
                                && !a.pools.iter().any(|p| p.resident_weight_bytes.0 > 0))
                    }))
        {
            return Err(Error("incomplete or mismatched profile coverage"));
        }
        Ok(())
    }
}
impl PoolCost {
    fn is_zero(&self) -> bool {
        [
            self.resident_weight_bytes,
            self.resident_overhead_bytes,
            self.active_weight_bytes_per_token,
            self.kv_bytes_per_token,
            self.recurrent_bytes_per_session,
            self.convolution_bytes_per_session,
            self.prefill_scratch_bytes,
            self.decode_scratch_bytes,
            self.staging_bytes,
            self.reserve_bytes,
        ]
        .iter()
        .all(|n| n.0 == 0)
    }
}
