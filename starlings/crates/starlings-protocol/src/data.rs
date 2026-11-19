//! GCP1 wire contracts. Parsing proves structure, never permission or execution.
use crate::*;
use serde::{Deserialize, Serialize};

pub const DATA_HEADER_LIMIT: usize = 16 * 1024;
pub const DATA_PAYLOAD_LIMIT: usize = 8 * 1024 * 1024;
pub const DATA_TENSOR_LIMIT: u64 = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum DataKind {
    Tensor = 1,
    Credit = 2,
    Receipt = 3,
    Execution = 4,
    Cancel = 5,
    Error = 6,
}

/// Construct/parse this before allocating either variable-length body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DataPrefix {
    kind: DataKind,
    header_len: usize,
    payload_len: usize,
}
impl DataPrefix {
    pub fn new(kind: DataKind, header_len: usize, payload_len: usize) -> Result<Self, Error> {
        if header_len == 0
            || header_len > DATA_HEADER_LIMIT
            || payload_len > DATA_PAYLOAD_LIMIT
            || (kind == DataKind::Tensor) != (payload_len != 0)
        {
            return Err(Error("invalid data frame length"));
        }
        Ok(Self {
            kind,
            header_len,
            payload_len,
        })
    }
    pub fn parse(bytes: [u8; 16]) -> Result<Self, Error> {
        if &bytes[..4] != b"GCP1" || bytes[4..6] != [0, 1] {
            return Err(Error("unsupported data protocol"));
        }
        let kind = match u16::from_be_bytes([bytes[6], bytes[7]]) {
            1 => DataKind::Tensor,
            2 => DataKind::Credit,
            3 => DataKind::Receipt,
            4 => DataKind::Execution,
            5 => DataKind::Cancel,
            6 => DataKind::Error,
            _ => return Err(Error("unsupported data frame kind")),
        };
        Self::new(
            kind,
            u32::from_be_bytes(bytes[8..12].try_into().unwrap()) as usize,
            u32::from_be_bytes(bytes[12..16].try_into().unwrap()) as usize,
        )
    }
    pub fn bytes(self) -> [u8; 16] {
        let mut bytes = [0; 16];
        bytes[..4].copy_from_slice(b"GCP1");
        bytes[4..6].copy_from_slice(&1u16.to_be_bytes());
        bytes[6..8].copy_from_slice(&(self.kind as u16).to_be_bytes());
        bytes[8..12].copy_from_slice(&(self.header_len as u32).to_be_bytes());
        bytes[12..16].copy_from_slice(&(self.payload_len as u32).to_be_bytes());
        bytes
    }
    pub fn kind(self) -> DataKind {
        self.kind
    }
    pub fn header_len(self) -> usize {
        self.header_len
    }
    pub fn payload_len(self) -> usize {
        self.payload_len
    }
}

/// Compared with a trusted local admission; never accepted as its own grant.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataScope {
    pub plan_digest: Digest,
    pub authority_epoch: Decimal,
    pub lease_id: LeaseId,
    pub session_id: SessionId,
    pub generation_id: GenerationId,
    pub invocation_id: InvocationId,
    pub sequence: Decimal,
    pub source_stage: BoundedU32<0, 7>,
    pub destination_stage: BoundedU32<0, 7>,
}
impl From<&TensorHeader> for DataScope {
    fn from(h: &TensorHeader) -> Self {
        Self {
            plan_digest: h.plan_digest.clone(),
            authority_epoch: h.authority_epoch,
            lease_id: h.lease_id,
            session_id: h.session_id,
            generation_id: h.generation_id,
            invocation_id: h.invocation_id,
            sequence: h.sequence,
            source_stage: h.source_stage,
            destination_stage: h.destination_stage,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataFailure {
    Cancelled,
    Expired,
    Protocol,
    Capacity,
    Backend,
    Disconnected,
}

/// Receipt is byte ownership only. Execution is a peer report, never local drain proof.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub enum DataControl {
    Credit {
        protocol_version: Version,
        scope: DataScope,
        bytes: Decimal,
    },
    ReceiptAck {
        protocol_version: Version,
        scope: DataScope,
        tensor_index: BoundedU32<0, 7>,
        received_through_bytes: Decimal,
    },
    ExecutionResult {
        protocol_version: Version,
        scope: DataScope,
    },
    Cancel {
        protocol_version: Version,
        scope: DataScope,
        code: DataFailure,
    },
    Error {
        protocol_version: Version,
        scope: DataScope,
        code: DataFailure,
    },
}
impl DataControl {
    pub fn scope(&self) -> &DataScope {
        match self {
            Self::Credit { scope, .. }
            | Self::ReceiptAck { scope, .. }
            | Self::ExecutionResult { scope, .. }
            | Self::Cancel { scope, .. }
            | Self::Error { scope, .. } => scope,
        }
    }
    pub fn kind(&self) -> DataKind {
        match self {
            Self::Credit { .. } => DataKind::Credit,
            Self::ReceiptAck { .. } => DataKind::Receipt,
            Self::ExecutionResult { .. } => DataKind::Execution,
            Self::Cancel { .. } => DataKind::Cancel,
            Self::Error { .. } => DataKind::Error,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum DataHeader {
    Tensor(TensorHeader),
    Control(DataControl),
}
impl DataHeader {
    /// Called after the bounded header read, before any payload allocation.
    pub fn decode(prefix: DataPrefix, bytes: &[u8], fixtures: FixtureMode) -> Result<Self, Error> {
        if bytes.len() != prefix.header_len() {
            return Err(Error("data header length mismatch"));
        }
        if prefix.kind() == DataKind::Tensor {
            let header: TensorHeader = decode(bytes, fixtures)?;
            if header.chunk_bytes.0 != prefix.payload_len() as u64 {
                return Err(Error("data payload length mismatch"));
            }
            Ok(Self::Tensor(header))
        } else {
            let header: DataControl = decode(bytes, fixtures)?;
            if header.kind() != prefix.kind() {
                return Err(Error("data kind mismatch"));
            }
            Ok(Self::Control(header))
        }
    }
    pub fn encode(&self, fixtures: FixtureMode) -> Result<(DataPrefix, Vec<u8>), Error> {
        let (kind, payload_len, bytes) = match self {
            Self::Tensor(header) => (
                DataKind::Tensor,
                header.chunk_bytes.0,
                encode(header, fixtures)?,
            ),
            Self::Control(header) => (header.kind(), 0, encode(header, fixtures)?),
        };
        let payload_len =
            usize::try_from(payload_len).map_err(|_| Error("data payload overflow"))?;
        Ok((DataPrefix::new(kind, bytes.len(), payload_len)?, bytes))
    }
    pub fn scope(&self) -> DataScope {
        match self {
            Self::Tensor(h) => h.into(),
            Self::Control(h) => h.scope().clone(),
        }
    }
}
