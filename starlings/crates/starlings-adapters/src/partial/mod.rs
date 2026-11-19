//! Proposed partial-worker boundary metadata, not an installed backend API.
pub mod glyd;
use serde::{Deserialize, Serialize};
use starlings_protocol::{
    BoundedU32, DataScope, Decimal, FixtureMode, ModelCostProfile, ModelCosts, ProfileKey,
    TensorHeaderDtype, TensorHeaderPhase, TensorHeaderTensorRole, Version,
};

type Result<T> = std::result::Result<T, Error>;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Error {
    Invalid,
    Unsupported,
    Capacity,
    State,
    Conflict,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "value",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum Dimension {
    Fixed(BoundedU32<1, 1048576>),
    Tokens,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorPattern {
    pub role: TensorHeaderTensorRole,
    pub dtype: TensorHeaderDtype,
    pub shape: Vec<Dimension>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorLayout {
    pub role: TensorHeaderTensorRole,
    pub dtype: TensorHeaderDtype,
    pub shape: Vec<BoundedU32<1, 1048576>>,
    pub bytes: Decimal,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputTokens {
    SameChunk,
    SampledOne,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundaryContract {
    pub version: Version,
    pub fixture: bool,
    pub key: ProfileKey,
    pub inputs: Vec<TensorPattern>,
    pub outputs: Vec<TensorPattern>,
    pub output_tokens: OutputTokens,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StepIdentity {
    pub scope: DataScope,
    pub phase: TensorHeaderPhase,
    pub position_start: BoundedU32<0, 1048575>,
    pub token_count: BoundedU32<1, 1048576>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunChunk {
    pub version: Version,
    pub step: StepIdentity,
    pub inputs: Vec<TensorLayout>,
}
fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    serde_json::from_value(starlings_protocol::parse_json(bytes).map_err(|_| Error::Invalid)?)
        .map_err(|_| Error::Invalid)
}
impl TensorPattern {
    fn layout(&self, tokens: u32) -> Result<TensorLayout> {
        let token_ids = self.role == TensorHeaderTensorRole::TokenIds;
        let axis = self
            .shape
            .len()
            .checked_sub(if token_ids { 1 } else { 2 })
            .ok_or(Error::Invalid)?;
        if self.shape.len() > 4
            || token_ids != (self.dtype == TensorHeaderDtype::U32)
            || self
                .shape
                .iter()
                .enumerate()
                .any(|(i, d)| matches!(d, Dimension::Tokens) != (i == axis))
        {
            return Err(Error::Invalid);
        }
        let shape: Vec<_> = self
            .shape
            .iter()
            .map(|d| match d {
                Dimension::Fixed(n) => Ok(*n),
                Dimension::Tokens => BoundedU32::new(tokens).map_err(|_| Error::Invalid),
            })
            .collect::<Result<_>>()?;
        let width = match self.dtype {
            TensorHeaderDtype::Bf16 | TensorHeaderDtype::F16 => 2u64,
            _ => 4,
        };
        let bytes = shape
            .iter()
            .try_fold(width, |n, d| n.checked_mul(d.get().into()))
            .ok_or(Error::Capacity)?;
        if bytes > starlings_protocol::DATA_TENSOR_LIMIT {
            return Err(Error::Capacity);
        }
        Ok(TensorLayout {
            role: self.role,
            dtype: self.dtype,
            shape,
            bytes: Decimal(bytes),
        })
    }
}
impl BoundaryContract {
    pub fn decode(bytes: &[u8], mode: FixtureMode) -> Result<Self> {
        let value: Self = decode(bytes)?;
        value.validate(mode)?;
        Ok(value)
    }
    pub fn validate(&self, mode: FixtureMode) -> Result<()> {
        // Reuse identity/workload checks; an unavailable profile supplies no measurements.
        let profile = ModelCostProfile {
            kind: "ModelCostProfile".into(),
            version: Version,
            fixture: self.fixture,
            key: self.key.clone(),
            costs: ModelCosts::Unavailable {
                code: "UNSUPPORTED".into(),
            },
        };
        starlings_protocol::encode(&profile, mode).map_err(|_| Error::Invalid)?;
        self.layouts(self.key.workload.prefill_chunk_tokens.get(), u64::MAX)?;
        Ok(())
    }
    /// Checked host payload sizes only, never native allocation/aliasing qualification.
    pub fn layouts(
        &self,
        tokens: u32,
        combined_budget: u64,
    ) -> Result<(Vec<TensorLayout>, Vec<TensorLayout>)> {
        if self.inputs.is_empty()
            || self.inputs.len() > 8
            || self.outputs.is_empty()
            || self.outputs.len() > 8
            || (self.output_tokens == OutputTokens::SampledOne)
                != self
                    .outputs
                    .iter()
                    .all(|p| p.role == TensorHeaderTensorRole::TokenIds)
        {
            return Err(Error::Invalid);
        }
        if tokens == 0 || tokens > self.key.workload.prefill_chunk_tokens.get() {
            return Err(Error::Invalid);
        }
        let inputs = self
            .inputs
            .iter()
            .map(|p| p.layout(tokens))
            .collect::<Result<Vec<_>>>()?;
        let outputs = self
            .outputs
            .iter()
            .map(|p| {
                p.layout(match self.output_tokens {
                    OutputTokens::SameChunk => tokens,
                    OutputTokens::SampledOne => 1,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let sum = inputs
            .iter()
            .chain(&outputs)
            .try_fold(0u64, |n, l| n.checked_add(l.bytes.0))
            .ok_or(Error::Capacity)?;
        if sum > combined_budget {
            return Err(Error::Capacity);
        }
        Ok((inputs, outputs))
    }
}
impl RunChunk {
    pub fn decode(bytes: &[u8], mode: FixtureMode) -> Result<Self> {
        let value: Self = decode(bytes)?;
        value.validate_metadata(mode)?;
        Ok(value)
    }
    fn validate_metadata(&self, mode: FixtureMode) -> Result<()> {
        // Validate only the scope; constructing this record performs no cancellation.
        starlings_protocol::encode(
            &starlings_protocol::DataControl::Cancel {
                protocol_version: Version,
                scope: self.step.scope.clone(),
                code: starlings_protocol::DataFailure::Cancelled,
            },
            mode,
        )
        .map_err(|_| Error::Invalid)?;
        if self.inputs.is_empty()
            || self.inputs.len() > 8
            || self.step.scope.source_stage == self.step.scope.destination_stage
            // A decode step is its token and the drafts after it (GLY-360).
            || self.step.phase == TensorHeaderPhase::Decode
                && self.step.token_count.get() > starlings_protocol::MAX_DECODE_ROWS
            || self
                .step
                .position_start
                .get()
                .checked_add(self.step.token_count.get())
                .is_none_or(|end| end > 1_048_576)
        {
            return Err(Error::Invalid);
        }
        for input in &self.inputs {
            let axis = input
                .shape
                .len()
                .checked_sub(if input.role == TensorHeaderTensorRole::TokenIds {
                    1
                } else {
                    2
                })
                .ok_or(Error::Invalid)?;
            if input.shape.len() > 4 {
                return Err(Error::Invalid);
            }
            let pattern = TensorPattern {
                role: input.role,
                dtype: input.dtype,
                shape: input
                    .shape
                    .iter()
                    .enumerate()
                    .map(|(i, n)| {
                        if i == axis {
                            Dimension::Tokens
                        } else {
                            Dimension::Fixed(*n)
                        }
                    })
                    .collect(),
            };
            if pattern.layout(self.step.token_count.get())? != *input {
                return Err(Error::Invalid);
            }
        }
        Ok(())
    }
    /// `expected` is trusted current session state, not metadata adopted from this call.
    pub fn validate(
        &self,
        boundary: &BoundaryContract,
        expected: &StepIdentity,
        budget: u64,
        mode: FixtureMode,
    ) -> Result<()> {
        boundary.validate(mode)?;
        self.validate_metadata(mode)?;
        if &self.step != expected {
            return Err(Error::Conflict);
        }
        let scope = &self.step.scope;
        if scope.authority_epoch.0 == 0
            || scope.source_stage == scope.destination_stage
            || self
                .step
                .position_start
                .get()
                .checked_add(self.step.token_count.get())
                .is_none_or(|end| end > boundary.key.workload.context_tokens.get())
            || self.step.phase == TensorHeaderPhase::Decode
                && self.step.token_count.get() > starlings_protocol::MAX_DECODE_ROWS
        {
            return Err(Error::Invalid);
        }
        let (inputs, _) = boundary.layouts(self.step.token_count.get(), budget)?;
        if self.inputs != inputs {
            return Err(Error::Invalid);
        }
        Ok(())
    }
}
#[cfg(feature = "test-backend")]
pub mod fixture;
#[cfg(feature = "test-backend")]
mod synthetic;
#[cfg(feature = "test-backend")]
pub use synthetic::SyntheticStage;
#[cfg(test)]
mod tests;
