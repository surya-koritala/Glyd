use crate::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

/// Authenticated caller identity is supplied by M1, not inferred from this record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlRequest {
    pub protocol_version: Version,
    pub request_id: RequestId,
    pub authority_epoch: Decimal,
    pub request_sequence: Decimal,
    pub request_hash: Digest,
    pub operation: Operation,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Operation {
    Prepare {
        plan: Box<PlacementPlan>,
        lease: Box<Lease>,
    },
    Commit {
        lease_id: LeaseId,
    },
    Release {
        lease_id: LeaseId,
    },
    Begin {
        session: Box<Session>,
    },
    StartRequest {
        session: Box<Session>,
        invocation_id: InvocationId,
        request: Box<ChatRequest>,
    },
    StartStep {
        header: Box<TensorHeader>,
        input_digest: Digest,
    },
    CompleteStep {
        generation_id: GenerationId,
        invocation_id: InvocationId,
    },
    Cancel {
        generation_id: GenerationId,
    },
    Finish {
        generation_id: GenerationId,
    },
}

pub fn content_hash<T: Serialize>(value: &T) -> Result<Digest, Error> {
    let bytes =
        serde_json_canonicalizer::to_vec(value).map_err(|_| Error("canonicalization failed"))?;
    Ok(byte_digest(&bytes))
}
pub fn byte_digest(bytes: &[u8]) -> Digest {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>()
        .try_into()
        .expect("SHA-256 encoding")
}
impl ControlRequest {
    pub fn schema() -> &'static jsonschema::Validator {
        static SCHEMA: std::sync::OnceLock<jsonschema::Validator> = std::sync::OnceLock::new();
        SCHEMA.get_or_init(|| {
            let schema: serde_json::Value =
                serde_json::from_str(include_str!("../../../spec/control-request.schema.json"))
                    .expect("embedded control request schema");
            let control: serde_json::Value =
                serde_json::from_str(include_str!("../../../spec/control.schema.json"))
                    .expect("control schema");
            let tensor: serde_json::Value =
                serde_json::from_str(include_str!("../../../spec/tensor-header.schema.json"))
                    .expect("tensor schema");
            let chat: serde_json::Value =
                serde_json::from_str(include_str!("../../../spec/chat-request.schema.json"))
                    .expect("chat schema");
            let registry = jsonschema::Registry::new()
                .add("urn:starlings:chat-request:v1", chat)
                .expect("chat schema ID")
                .add("urn:starlings:control:v1", control)
                .expect("control schema ID")
                .add("urn:starlings:tensor-header:v1", tensor)
                .expect("tensor schema ID")
                .prepare()
                .expect("offline schema registry");
            jsonschema::draft202012::options()
                .with_registry(&registry)
                .should_validate_formats(true)
                .build(&schema)
                .expect("offline control request schema")
        })
    }
    pub fn new(
        request_id: RequestId,
        epoch: u64,
        sequence: u64,
        operation: Operation,
    ) -> Result<Self, Error> {
        let request_hash =
            content_hash(&(request_id, Decimal(epoch), Decimal(sequence), &operation))?;
        Ok(Self {
            protocol_version: Version,
            request_id,
            authority_epoch: Decimal(epoch),
            request_sequence: Decimal(sequence),
            request_hash,
            operation,
        })
    }
    pub fn decode(bytes: &[u8], mode: FixtureMode) -> Result<Self, Error> {
        let value = parse_json(bytes)?;
        Self::schema()
            .validate(&value)
            .map_err(|_| Error("invalid control request schema"))?;
        let request: Self =
            serde_json::from_value(value).map_err(|_| Error("invalid control request"))?;
        request.validate(mode)?;
        Ok(request)
    }
    pub fn validate(&self, mode: FixtureMode) -> Result<(), Error> {
        if self.authority_epoch.0 == 0
            || self.request_sequence.0 == 0
            || self.request_hash
                != content_hash(&(
                    self.request_id,
                    self.authority_epoch,
                    self.request_sequence,
                    &self.operation,
                ))?
        {
            return Err(Error("invalid authority or request digest"));
        }
        match &self.operation {
            Operation::Prepare { plan, lease } => {
                encode(plan.as_ref(), mode)?;
                encode(lease.as_ref(), mode)?;
            }
            Operation::Begin { session } => {
                encode(session.as_ref(), mode)?;
            }
            Operation::StartRequest {
                session, request, ..
            } => {
                encode(session.as_ref(), mode)?;
                encode(request.as_ref(), mode)?;
                if request.max_tokens != session.max_output_tokens
                    || request.temperature != Some(session.sampling.temperature)
                    || request.top_p != Some(session.sampling.top_p)
                    || request.seed != Some(session.sampling.seed)
                {
                    return Err(Error(
                        "whole-request sampling differs from admitted session",
                    ));
                }
            }
            Operation::StartStep { header, .. } => {
                encode(header.as_ref(), mode)?;
            }
            _ => {}
        }
        Ok(())
    }
}
