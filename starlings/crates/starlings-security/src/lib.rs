//! Enrollment, verified TLS and live local authorization. No discovery or daemon.
mod admin;
mod authorization;
pub mod catalog;
mod client;
mod data;
mod generation;
pub use data::{DataFrame, DataStream, FlowReceiver, FlowSender, ReceiveBudget, ReceivedTensors};
pub use generation::{GenerationBinding, GenerationStream, GenerationWatch};
mod streaming;
pub use streaming::{ClientResponse, ClientStream, ClientWatch};
mod listener;
pub use admin::{AdminListener, AdminRequest, admin_call};
pub use client::{ClientListener, ClientRequest, IncomingClient};
pub use listener::{IncomingPeer, ListenerBudget, ListenerUsage, PeerListener};
mod enrollment;
pub mod network;
pub mod probes;
mod reconnect;
mod registry;
mod renewal;
mod runtime;
pub use registry::{
    AuthorityEpoch, LinkSample, NodeStatus, NodeSummary, PendingHeartbeat, REGISTRY_TTL,
    RegistryClient, RegistryFence, RegistryRequest, ResourceSample, ResourceView, RouteSummary,
    heartbeat_delay,
};
mod transport;
pub use runtime::StateLock;

pub use authorization::{
    Action, ClientAdmission, ClientToken, ClientTokenIssuer, DataAdmission, EnrolledWorker, Scope,
    SecuredWorker, StageBackend, StageChunk, StageGrant, StageIssuer, StageRuntime, StageWorker,
    step_invocation,
};
pub use enrollment::{
    ApprovedEnrollment, Authority, EnrollmentReader, Identity, Invitation, InvitationIssuer,
    PendingEnrollment, Role, Trust,
};
pub use private_fs::PrivateDir;
use serde::{Serialize, de::DeserializeOwned};
use sha2::{Digest as _, Sha256};
use std::{
    fmt,
    io::{Read, Write},
};
pub use transport::{Channel, PeerStream};

const MAX_RECORD: usize = 16 * 1024;
type Result<T> = std::result::Result<T, Error>;

/// Errors intentionally contain no input, credential, path or library diagnostic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, serde::Deserialize)]
pub enum Error {
    AdminOutcomeUnknown,
    ReenrollmentRequired,
    ReplayExpired,
    Storage,
    Busy,
    Crypto,
    Invalid,
    Denied,
    Expired,
    Replay,
    Capacity,
    Transport,
    ProfileInUse,
    AddressInUse,
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "security: {self:?}")
    }
}
impl std::error::Error for Error {}
macro_rules! redacted {
    ($kind:ident: $($ty:ty),+) => {$ (
        impl From<$ty> for Error { fn from(_: $ty) -> Self { Self::$kind } }
    )+};
}
redacted!(Storage: private_fs::Error);
impl From<rusqlite::Error> for Error {
    fn from(error: rusqlite::Error) -> Self {
        match error {
            rusqlite::Error::SqliteFailure(e, _)
                if matches!(
                    e.code,
                    rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked
                ) =>
            {
                Self::Busy
            }
            _ => Self::Storage,
        }
    }
}
redacted!(Crypto: rcgen::Error, rustls::Error);
redacted!(Invalid: serde_json::Error, starlings_protocol::Error);
impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        #[cfg(debug_assertions)]
        if std::env::var_os("STARLINGS_TEST_REGISTRY_TRACE").is_some() {
            eprintln!(
                "REGISTRY_IO_FAILURE kind={:?} os_code={:?}",
                error.kind(),
                error.raw_os_error()
            );
        }
        if error.kind() == std::io::ErrorKind::AddrInUse {
            Self::AddressInUse
        } else {
            Self::Transport
        }
    }
}

pub(crate) fn registry_failure(phase: &'static str, error: &Error) {
    #[cfg(debug_assertions)]
    if std::env::var_os("STARLINGS_TEST_REGISTRY_TRACE").is_some() {
        eprintln!("REGISTRY_FAILURE phase={phase} error={error:?}");
    }
    #[cfg(not(debug_assertions))]
    let _ = (phase, error);
}

fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
fn now() -> Result<i64> {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|_| Error::Invalid)?
        .as_secs();
    seconds.try_into().map_err(|_| Error::Invalid)
}
fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    if bytes.len() > MAX_RECORD {
        return Err(Error::Invalid);
    }
    Ok(serde_json::from_value(starlings_protocol::parse_json(
        bytes,
    )?)?)
}
fn receive<T: DeserializeOwned>(io: &mut impl Read) -> Result<T> {
    let mut prefix = [0; 4];
    io.read_exact(&mut prefix)?;
    let len = u32::from_be_bytes(prefix) as usize;
    if len > MAX_RECORD {
        return Err(Error::Invalid);
    }
    let mut bytes = vec![0; len];
    io.read_exact(&mut bytes)?;
    decode(&bytes)
}
fn send(io: &mut impl Write, record: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec(record)?;
    if bytes.len() > MAX_RECORD {
        return Err(Error::Invalid);
    }
    io.write_all(&record_frame(&bytes)?)?;
    io.flush()?;
    Ok(())
}
/// A record's length and bytes as one write: two small writes stall on Nagle and the peer's
/// delayed ACK (40 ms each on Linux).
fn record_frame(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut frame = Vec::new();
    frame
        .try_reserve_exact(4 + bytes.len())
        .map_err(|_| Error::Capacity)?;
    frame.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
    frame.extend_from_slice(bytes);
    Ok(frame)
}

#[cfg(test)]
mod review_tests;
