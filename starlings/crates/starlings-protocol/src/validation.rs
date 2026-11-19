use crate::*;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use std::{collections::BTreeSet, sync::OnceLock};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FixtureMode {
    #[default]
    Reject,
    Simulation,
}

pub trait Contract: Serialize + DeserializeOwned {
    fn schema() -> &'static jsonschema::Validator;
    fn validate(&self) -> Result<(), Error>;
}
fn control_schema() -> &'static jsonschema::Validator {
    static SCHEMA: OnceLock<jsonschema::Validator> = OnceLock::new();
    SCHEMA.get_or_init(|| {
        jsonschema::draft202012::options()
            .should_validate_formats(true)
            .build(
                &serde_json::from_str::<Value>(include_str!("../../../spec/control.schema.json"))
                    .expect("embedded schema"),
            )
            .expect("offline schema")
    })
}
macro_rules! schema {
    ("control.schema.json") => {
        fn schema() -> &'static jsonschema::Validator {
            control_schema()
        }
    };
    ($file:literal) => {
        fn schema() -> &'static jsonschema::Validator {
            static SCHEMA: OnceLock<jsonschema::Validator> = OnceLock::new();
            SCHEMA.get_or_init(|| {
                let value: Value =
                    serde_json::from_str(include_str!(concat!("../../../spec/", $file)))
                        .expect("embedded schema parses");
                jsonschema::draft202012::options()
                    .should_validate_formats(true)
                    .build(&value)
                    .expect("embedded schema compiles offline")
            })
        }
    };
}

pub fn decode<T: Contract>(bytes: &[u8], fixtures: FixtureMode) -> Result<T, Error> {
    let value = parse_json(bytes)?;
    T::schema()
        .validate(&value)
        .map_err(|_| Error("schema violation"))?;
    validate_identity(&value, fixtures)?;
    let record: T = serde_json::from_value(value).map_err(|_| Error("invalid record field"))?;
    record.validate()?;
    Ok(record)
}

pub fn encode<T: Contract>(record: &T, fixtures: FixtureMode) -> Result<Vec<u8>, Error> {
    let bytes = serde_json::to_vec(record).map_err(|_| Error("serialization failed"))?;
    decode::<T>(&bytes, fixtures)?;
    Ok(bytes)
}

pub fn canonical_digest(value: &Value, field: &str) -> Result<Digest, Error> {
    if !matches!(field, "manifest_digest" | "plan_digest") {
        return Err(Error("invalid digest field"));
    }
    let mut body = value
        .as_object()
        .ok_or(Error("digest requires an object"))?
        .clone();
    if body.remove(field).is_none() {
        return Err(Error("missing self digest"));
    }
    let canonical =
        serde_json_canonicalizer::to_vec(&body).map_err(|_| Error("canonicalization failed"))?;
    Sha256::digest(canonical)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>()
        .try_into()
        .map_err(Error)
}

fn placeholder_digest(value: &Value) -> bool {
    match value {
        Value::Object(fields) => fields.iter().any(|(key, value)| {
            ((key.ends_with("digest") || key.ends_with("sha256"))
                && value
                    .as_str()
                    .is_some_and(|s| s.len() == 64 && s.bytes().all(|b| b == b'0')))
                || placeholder_digest(value)
        }),
        Value::Array(items) => items.iter().any(placeholder_digest),
        _ => false,
    }
}
fn validate_identity(value: &Value, fixtures: FixtureMode) -> Result<(), Error> {
    let fixture = value.get("fixture").and_then(Value::as_bool) == Some(true);
    if fixtures == FixtureMode::Reject && (fixture || placeholder_digest(value)) {
        return Err(Error("synthetic fixture forbidden"));
    }
    let field = match value.get("kind").and_then(Value::as_str) {
        Some("ModelManifest") => Some("manifest_digest"),
        Some("PlacementPlan") => Some("plan_digest"),
        _ => None,
    };
    if let Some(field) = field {
        let supplied = value[field].as_str().ok_or(Error("missing digest"))?;
        // Only explicit synthetic fixtures may retain a zero self-digest.
        if !(fixture && fixtures == FixtureMode::Simulation && supplied == "0".repeat(64))
            && canonical_digest(value, field)?.as_str() != supplied
        {
            return Err(Error("content digest mismatch"));
        }
    }
    Ok(())
}
fn unique<'a>(values: impl IntoIterator<Item = &'a str>) -> bool {
    let mut seen = BTreeSet::new();
    values.into_iter().all(|v| seen.insert(v))
}

impl Contract for BenchmarkRun {
    schema!("benchmark-run.schema.json");
    fn validate(&self) -> Result<(), Error> {
        if self.baseline_run_id == Some(self.run_id)
            || self.compute_commit.bytes().all(|b| b == b'0')
            || (self.status == BenchmarkStatus::Measured
                && (self.requests.is_empty()
                    || self.peak_memory_pools.is_empty()
                    || self
                        .requests
                        .iter()
                        .any(|r| r.outcome != BenchmarkOutcome::Success)))
            || !unique(self.backend_build_digests.iter().map(Digest::as_str))
            || self
                .backend_build_digests
                .iter()
                .chain([
                    &self.manifest_digest,
                    &self.plan_digest,
                    &self.hardware_inventory_sha256,
                    &self.workload_sha256,
                    &self.raw_logs_sha256,
                ])
                .any(|d| d.as_str().bytes().all(|b| b == b'0'))
        {
            return Err(Error("incoherent benchmark identity or status"));
        }
        let mut requests = BTreeSet::new();
        for request in &self.requests {
            let failed = request.outcome != BenchmarkOutcome::Success;
            if !requests.insert(request.request_id)
                || failed != request.error_code.is_some()
                || request.error_code.as_ref().is_some_and(|code| {
                    code.is_empty()
                        || code.len() > 64
                        || !code
                            .bytes()
                            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
                })
            {
                return Err(Error("incoherent benchmark request identity or failure"));
            }
            let duration = request.duration_ms;
            if !duration.is_finite()
                || duration < 0.0
                || request.ttft_ms.is_some_and(|t| {
                    !t.is_finite() || t < 0.0 || t > duration || request.output_tokens.get() == 0
                })
                || (!request.inter_token_ms.is_empty()
                    && (request.ttft_ms.is_none()
                        || request.inter_token_ms.len() as u64 + 1
                            != u64::from(request.output_tokens.get())))
                || request
                    .inter_token_ms
                    .iter()
                    .any(|t| !t.is_finite() || *t < 0.0)
            {
                return Err(Error("incoherent benchmark token timing"));
            }
            let elapsed =
                request.ttft_ms.unwrap_or(0.0) + request.inter_token_ms.iter().sum::<f64>();
            // Permit only accumulated floating-point rounding, not an extra timing budget.
            let rounding = duration * f64::EPSILON * (request.inter_token_ms.len() + 2) as f64;
            if !elapsed.is_finite() || elapsed - duration > rounding {
                return Err(Error("benchmark token timings exceed request duration"));
            }
        }
        let mut pools = BTreeSet::new();
        if self
            .peak_memory_pools
            .iter()
            .any(|p| !pools.insert((p.node_id, &p.pool_id)))
        {
            return Err(Error("duplicate benchmark memory pool"));
        }
        for path in [&self.hardware_inventory_path, &self.raw_logs_path] {
            if path.contains(['\\', ':'])
                || path.chars().any(char::is_control)
                || path.split('/').any(|part| matches!(part, "" | "." | ".."))
            {
                return Err(Error("unsafe benchmark evidence path"));
            }
        }
        Ok(())
    }
}

impl Contract for NodeCapabilities {
    schema!("control.schema.json");
    fn validate(&self) -> Result<(), Error> {
        if !unique(self.memory_pools.iter().map(|p| p.pool_id.as_str()))
            || !unique(self.backends.iter().map(|b| b.backend_id.as_str()))
        {
            return Err(Error("duplicate capability identity"));
        }
        for pool in &self.memory_pools {
            if pool
                .available_bytes
                .0
                .checked_add(pool.reserved_bytes.0)
                .is_none_or(|n| n > pool.total_bytes.0)
                || !unique(pool.device_ids.iter().map(String::as_str))
            {
                return Err(Error("incoherent memory pool"));
            }
        }
        for b in &self.backends {
            if !unique(b.memory_pool_ids.iter().map(String::as_str))
                || b.layer_sharding && b.boundary_profiles.is_empty()
            {
                return Err(Error("incoherent backend capability"));
            }
            for id in &b.memory_pool_ids {
                let pool = self
                    .memory_pools
                    .iter()
                    .find(|p| &p.pool_id == id)
                    .ok_or(Error("unknown memory pool"))?;
                if !pool.device_ids.contains(&b.device_id) {
                    return Err(Error("device/pool mismatch"));
                }
            }
        }
        Ok(())
    }
}
impl Stage {
    pub fn reserved_bytes(&self) -> Result<u64, Error> {
        [
            self.resident_bytes.0,
            self.state_budget_bytes.0,
            self.scratch_budget_bytes.0,
            self.safety_reserve_bytes.0,
        ]
        .into_iter()
        .try_fold(0u64, |a, b| {
            a.checked_add(b).ok_or(Error("reservation overflow"))
        })
    }
}
impl Contract for PlacementPlan {
    schema!("control.schema.json");
    fn validate(&self) -> Result<(), Error> {
        let mut end = 0;
        let mut owners = BTreeSet::new();
        if self.authority_epoch.0 == 0
            || self.mode == PlacementPlanMode::WholeRequest && self.stages.len() != 1
        {
            return Err(Error("invalid placement mode or epoch"));
        }
        for (i, s) in self.stages.iter().enumerate() {
            if !owners.insert(s.node_id) {
                return Err(Error("worker owns multiple stages"));
            }
            if s.index.get() as usize != i
                || s.layer_start.get() != end
                || s.layer_end_exclusive.get() <= end
                || s.owns_embedding != (i == 0)
                || s.owns_final_norm_and_head != (i + 1 == self.stages.len())
            {
                return Err(Error("invalid layer coverage or ownership"));
            }
            s.reserved_bytes()?;
            end = s.layer_end_exclusive.get();
        }
        if end != self.model_layer_count.get() {
            return Err(Error("incomplete layer coverage"));
        }
        let parse = |s: &str| {
            time::OffsetDateTime::parse(s, &time::format_description::well_known::Rfc3339)
                .map_err(|_| Error("invalid plan time"))
        };
        if parse(&self.expires_at)? <= parse(&self.created_at)? {
            return Err(Error("invalid plan lifetime"));
        }
        Ok(())
    }
}
impl Contract for Lease {
    schema!("control.schema.json");
    fn validate(&self) -> Result<(), Error> {
        if self.authority_epoch.0 == 0 || self.reserved_bytes.0 == 0 {
            return Err(Error("empty lease"));
        }
        Ok(())
    }
}
impl Contract for Session {
    schema!("control.schema.json");
    fn validate(&self) -> Result<(), Error> {
        if self.authority_epoch.0 == 0
            || self
                .prompt_token_count
                .get()
                .checked_add(self.max_output_tokens.get())
                .is_none_or(|n| n > self.context_limit.get())
        {
            return Err(Error("session exceeds context or epoch limit"));
        }
        Ok(())
    }
}
impl Contract for WireError {
    schema!("control.schema.json");
    fn validate(&self) -> Result<(), Error> {
        Ok(())
    }
}
impl Contract for ChatRequest {
    schema!("chat-request.schema.json");
    fn validate(&self) -> Result<(), Error> {
        if self
            .messages
            .iter()
            .try_fold(0usize, |a, m| a.checked_add(m.content.len()))
            .is_none_or(|n| n > MAX_JSON_BYTES)
        {
            return Err(Error("aggregate messages exceed limit"));
        }
        Ok(())
    }
}
impl Contract for DataControl {
    schema!("data-control.schema.json");
    fn validate(&self) -> Result<(), Error> {
        if self.scope().authority_epoch.0 == 0 {
            return Err(Error("invalid data authority epoch"));
        }
        match self {
            Self::Credit { bytes, .. } if bytes.0 == 0 || bytes.0 > DATA_TENSOR_LIMIT => {
                Err(Error("invalid data credit"))
            }
            Self::ReceiptAck {
                received_through_bytes,
                ..
            } if received_through_bytes.0 == 0 || received_through_bytes.0 > DATA_TENSOR_LIMIT => {
                Err(Error("invalid data receipt"))
            }
            _ => Ok(()),
        }
    }
}
impl Contract for TensorHeader {
    schema!("tensor-header.schema.json");
    fn validate(&self) -> Result<(), Error> {
        let width: u64 = match self.dtype {
            TensorHeaderDtype::Bf16 | TensorHeaderDtype::F16 => 2,
            _ => 4,
        };
        let total = self
            .shape
            .iter()
            .try_fold(width, |a, n| a.checked_mul(n.get().into()))
            .ok_or(Error("tensor size overflow"))?;
        let end = self
            .chunk_offset_bytes
            .0
            .checked_add(self.chunk_bytes.0)
            .ok_or(Error("chunk overflow"))?;
        let token_extent = if self.tensor_role == TensorHeaderTensorRole::TokenIds {
            self.shape.last()
        } else {
            self.shape.get(self.shape.len().saturating_sub(2))
        };
        if token_extent.is_none_or(|extent| extent.get() != self.token_count.get())
            || total != self.total_tensor_bytes.0
            || total > 64 * 1024 * 1024
            || self.chunk_bytes.0 == 0
            || self.chunk_bytes.0 > 8 * 1024 * 1024
            || end > total
            || self.end_of_tensor != (end == total)
            || !self.chunk_offset_bytes.0.is_multiple_of(width)
            || !self.chunk_bytes.0.is_multiple_of(width)
            || self.tensor_index.get() >= self.tensor_count.get()
            || self.authority_epoch.0 == 0
            || self
                .position_start
                .get()
                .checked_add(self.token_count.get())
                .is_none_or(|n| n > 1_048_576)
            || (self.tensor_role == TensorHeaderTensorRole::TokenIds)
                != (self.dtype == TensorHeaderDtype::U32)
        {
            return Err(Error("invalid tensor extent, role or context"));
        }
        Ok(())
    }
}
impl Contract for ModelManifest {
    schema!("model-manifest.schema.json");
    fn validate(&self) -> Result<(), Error> {
        let mut paths = BTreeSet::new();
        for asset in &self.assets {
            let path = asset.path.to_ascii_lowercase();
            for part in path.split('/') {
                let stem = part.split('.').next().unwrap_or("");
                if part.is_empty()
                    || matches!(part, "." | "..")
                    || part.ends_with('.')
                    || matches!(stem, "con" | "prn" | "aux" | "nul")
                    || (stem.len() == 4
                        && (stem.starts_with("com") || stem.starts_with("lpt"))
                        && matches!(stem.as_bytes()[3], b'1'..=b'9'))
                {
                    return Err(Error("unsafe asset path"));
                }
            }
            if !paths.insert(path) {
                return Err(Error("colliding asset path"));
            }
        }
        for path in &paths {
            let prefix = format!("{path}/");
            if paths
                .range(prefix.clone()..)
                .next()
                .is_some_and(|p| p.starts_with(&prefix))
            {
                return Err(Error("file/directory path collision"));
            }
        }
        let declared = |p: &String| self.assets.iter().any(|a| &a.path == p);
        for (i, layer) in self.layers.iter().enumerate() {
            if layer.index.get() as usize != i || !layer.asset_paths.iter().all(declared) {
                return Err(Error("invalid layer asset inventory"));
            }
        }
        if !unique(self.non_layer_components.iter().map(|c| c.name.as_str())) {
            return Err(Error("duplicate component"));
        }
        for c in &self.non_layer_components {
            if !c.asset_paths.iter().all(declared) {
                return Err(Error("unknown component asset"));
            }
        }
        for (role, digest) in [
            (AssetRole::Config, &self.config_sha256),
            (AssetRole::Tokenizer, &self.tokenizer_sha256),
            (AssetRole::Template, &self.template_sha256),
        ] {
            if self
                .assets
                .iter()
                .filter(|a| a.role == role && &a.sha256 == digest)
                .count()
                != 1
            {
                return Err(Error("missing or ambiguous model metadata"));
            }
        }
        if !self.is_lossy && self.source_precision != self.execution_precision {
            return Err(Error("implicit precision change"));
        }
        Ok(())
    }
}

impl Contract for ModelCostProfile {
    schema!("model-cost-profile.schema.json");
    fn validate(&self) -> Result<(), Error> {
        self.validate_costs()
    }
}
