//! A bounded coordinator-to-worker channel, separate from registry heartbeats.
use crate::{
    Action, AuthorityEpoch, Error, Identity, Invitation, NodeStatus, RegistryFence, Result, Role,
    Trust,
    authorization::{Access, LiveConnection},
    listener::Permit,
    transport,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use starlings_protocol::{Digest, NodeId, Version};
use std::{
    io::{Read, Write},
    net::{Shutdown, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

const ALPN: &[u8] = b"starlings-generation/1";
const RECORD_TIME: Duration = Duration::from_secs(2);
const LIFETIME: Duration = Duration::from_secs(600);
const BYTE_LIMIT: usize = 32 * 1024 * 1024;
const RECORD_LIMIT: usize = 8192;

/// Locally approved route identity. Matching it does not qualify a native backend.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationBinding {
    pub version: Version,
    pub epoch: AuthorityEpoch,
    pub model: Digest,
    pub profile: String,
}
impl GenerationBinding {
    fn validate(&self) -> Result<()> {
        if self.epoch.number.0 == 0
            || self.epoch.incarnation.is_nil()
            || self.model.as_str().bytes().all(|b| b == b'0')
            || self.profile.is_empty()
            || self.profile.len() > 256
            || !self.profile.bytes().all(|b| b.is_ascii_graphic())
        {
            return Err(Error::Invalid);
        }
        Ok(())
    }
}
enum Authorization {
    Coordinator {
        access: Access,
        live: Arc<LiveConnection>,
    },
    Worker(RegistryFence),
}
struct Guard {
    authorization: Authorization,
    binding: GenerationBinding,
    socket: TcpStream,
    closed: AtomicBool,
    deadline: Instant,
    certificate_expires: i64,
    wall_high: Mutex<i64>,
}
impl Guard {
    fn close(&self) -> bool {
        let first = !self.closed.swap(true, Ordering::AcqRel);
        let _ = self.socket.shutdown(Shutdown::Both);
        first
    }
    fn check(&self) -> Result<()> {
        let mut _phase = "clock";
        let result = (|| {
            let wall = crate::now()?;
            _phase = "clock_lock";
            let mut high = self.wall_high.lock().map_err(|_| Error::Denied)?;
            _phase = "closed";
            if self.closed.load(Ordering::Acquire) {
                return Err(Error::Expired);
            }
            _phase = "lifetime";
            if Instant::now() >= self.deadline {
                return Err(Error::Expired);
            }
            _phase = "wall_rollback";
            if wall < *high {
                return Err(Error::Expired);
            }
            _phase = "certificate";
            if wall >= self.certificate_expires {
                return Err(Error::Expired);
            }
            *high = wall;
            drop(high);
            match &self.authorization {
                Authorization::Worker(fence) => {
                    _phase = "worker_registry";
                    if fence.status()? != (self.binding.epoch, NodeStatus::Online) {
                        return Err(Error::Expired);
                    }
                }
                Authorization::Coordinator { access, live } => {
                    _phase = "coordinator_policy";
                    let state = access.lock()?;
                    _phase = "coordinator_draining";
                    if state.draining {
                        return Err(Error::Expired);
                    }
                    _phase = "coordinator_registry";
                    if !state
                        .registry
                        .as_ref()
                        .is_some_and(|r| r.online(live.peer.node, self.binding.epoch))
                    {
                        return Err(Error::Expired);
                    }
                    _phase = "local_credential";
                    state.credential(&live.local)?;
                    _phase = "peer_credential";
                    state.credential(&live.peer)?;
                    _phase = "control_permission";
                    state.allows(live.local.node, &self.binding.model, Action::Control)?;
                    _phase = "infer_permission";
                    state.allows(live.peer.node, &self.binding.model, Action::Infer)?;
                }
            }
            Ok(())
        })();
        if let Err(_error) = result {
            let _first = self.close();
            #[cfg(debug_assertions)]
            if _first && std::env::var_os("STARLINGS_TEST_GENERATION_GATE_TRACE").is_some() {
                let side = match self.authorization {
                    Authorization::Worker(_) => "worker",
                    Authorization::Coordinator { .. } => "coordinator",
                };
                let _ = writeln!(
                    std::io::stderr(),
                    "GENERATION_GATE_REFUSED side={side} phase={_phase} reason={_error:?}"
                );
            }
        }
        result
    }
}

/// The supervisor checks this during backend waits, and cancels the backend on failure.
/// Closing interrupts blocked channel I/O. It never releases backend reservations.
#[derive(Clone)]
pub struct GenerationWatch(Arc<Guard>);
impl GenerationWatch {
    pub(crate) fn is_closed(&self) -> bool {
        self.0.closed.load(Ordering::Acquire)
    }
    pub(crate) fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    pub fn check(&self) -> Result<()> {
        self.0.check()
    }
    pub fn cancel(&self) {
        self.0.close();
    }
}

/// Authenticated records only; callers must still validate/apply the lifecycle protocol.
/// One owner performs I/O. Keep registry maintenance on its independent thread.
pub struct GenerationStream {
    wire: transport::Wire,
    guard: Arc<Guard>,
    remaining_bytes: usize,
    remaining_records: usize,
    permit: Option<Permit>,
}
impl GenerationStream {
    pub(crate) fn worker_fence(&self) -> Result<&RegistryFence> {
        match &self.guard.authorization {
            Authorization::Worker(fence) => Ok(fence),
            Authorization::Coordinator { .. } => Err(Error::Denied),
        }
    }
    pub fn connect(
        socket: TcpStream,
        local: &Identity,
        trust: &Trust,
        worker: NodeId,
        binding: GenerationBinding,
    ) -> Result<Self> {
        binding.validate()?;
        let access = trust.access.clone().ok_or(Error::Denied)?;
        let mut current = trust.clone();
        current.members = access.members()?;
        let expected = current.member(worker)?;
        if local.role() != Role::Coordinator
            || expected.role != Role::Worker
            || local.cluster() != current.cluster
        {
            return Err(Error::Denied);
        }
        // Requests and replies are small and back to back; none waits for an ACK.
        socket.set_nodelay(true)?;
        let wire = transport::client(socket, &current, expected, Some(local), ALPN)?;
        let peer = current
            .members
            .iter()
            .find(|m| {
                m.node == worker
                    && m.certificate == transport::peer_certificate(&wire).unwrap_or_default()
            })
            .ok_or(Error::Denied)?;
        let live = access.register(local.member.clone(), peer.clone(), wire.socket()?)?;
        let expires = certificate_expiry(&local.member.certificate)?
            .min(certificate_expiry(&peer.certificate)?);
        let mut stream = Self::new(
            wire,
            binding.clone(),
            Authorization::Coordinator { access, live },
            expires,
        )?;
        stream.send_record(&binding)?;
        let accepted: GenerationBinding = stream.receive_record()?;
        if accepted != binding {
            return Err(Error::Denied);
        }
        Ok(stream)
    }
    pub(crate) fn accept(
        socket: TcpStream,
        local: &Identity,
        invitation: &Invitation,
        fence: RegistryFence,
        binding: GenerationBinding,
        deadline: Instant,
    ) -> Result<Self> {
        binding.validate()?;
        invitation.verify_identity(local)?;
        fence.check_identity(local.node(), local.cluster())?;
        if local.role() != Role::Worker || fence.status()? != (binding.epoch, NodeStatus::Online) {
            return Err(Error::Denied);
        }
        let trust = Trust {
            ca: invitation.ca.clone(),
            cluster: invitation.cluster(),
            members: vec![invitation.coordinator.clone()],
            access: None,
        };
        socket.set_nodelay(true)?;
        let wire = transport::server_until(socket, local, Some(&trust), ALPN, deadline)?;
        let peer = transport::peer_certificate(&wire)?;
        // A renewed coordinator certificate may retain the enrolled key, but must still
        // chain to this CA and name exactly this cluster, role and coordinator node.
        verify_coordinator(peer, invitation)?;
        let expires = certificate_expiry(&local.member.certificate)?.min(certificate_expiry(peer)?);
        let mut stream = Self::new(wire, binding.clone(), Authorization::Worker(fence), expires)?;
        let requested: GenerationBinding = stream.receive_record()?;
        if requested != binding {
            return Err(Error::Denied);
        }
        stream.send_record(&binding)?;
        Ok(stream)
    }
    fn new(
        wire: transport::Wire,
        binding: GenerationBinding,
        authorization: Authorization,
        certificate_expires: i64,
    ) -> Result<Self> {
        let wall = crate::now()?;
        let lifetime = u64::try_from(certificate_expires - wall).map_err(|_| Error::Expired)?;
        let guard = Arc::new(Guard {
            socket: wire.socket()?,
            binding,
            authorization,
            closed: AtomicBool::new(false),
            certificate_expires,
            wall_high: Mutex::new(wall),
            deadline: Instant::now() + LIFETIME.min(Duration::from_secs(lifetime)),
        });
        guard.check()?;
        Ok(Self {
            wire,
            guard,
            remaining_bytes: BYTE_LIMIT,
            remaining_records: RECORD_LIMIT,
            permit: None,
        })
    }
    pub(crate) fn with_permit(mut self, permit: Permit) -> Self {
        self.permit = Some(permit);
        self
    }
    pub fn watch(&self) -> GenerationWatch {
        GenerationWatch(self.guard.clone())
    }
    pub fn binding(&self) -> &GenerationBinding {
        &self.guard.binding
    }
    pub(crate) fn coordinator_context(&self) -> Result<(Access, NodeId)> {
        self.guard.check()?;
        match &self.guard.authorization {
            Authorization::Coordinator { access, live } => Ok((access.clone(), live.peer.node)),
            _ => Err(Error::Denied),
        }
    }
    fn reserve(&mut self, length: usize) -> Result<()> {
        if length == 0 || length > starlings_protocol::MAX_JSON_BYTES || self.remaining_records == 0
        {
            return Err(Error::Capacity);
        }
        self.remaining_bytes = self
            .remaining_bytes
            .checked_sub(length + 4)
            .ok_or(Error::Capacity)?;
        self.remaining_records -= 1;
        Ok(())
    }
    fn start(&mut self) -> Result<()> {
        self.guard.check()?;
        self.wire
            .set_deadline(self.guard.deadline.min(Instant::now() + RECORD_TIME));
        Ok(())
    }
    pub fn send_record(&mut self, record: &impl Serialize) -> Result<()> {
        self.send_record_limit(record, starlings_protocol::MAX_JSON_BYTES)
    }
    pub(crate) fn send_record_limit(
        &mut self,
        record: &impl Serialize,
        limit: usize,
    ) -> Result<()> {
        let result = (|| {
            self.start()?;
            let bytes = serde_json::to_vec(record)?;
            if bytes.len() > limit {
                return Err(Error::Capacity);
            }
            self.reserve(bytes.len())?;
            starlings_protocol::parse_json(&bytes)?;
            self.wire.write_all(&crate::record_frame(&bytes)?)?;
            self.wire.flush()?;
            self.guard.check()
        })();
        if result.is_err() {
            self.guard.close();
        }
        result
    }
    pub fn receive_record<T: DeserializeOwned>(&mut self) -> Result<T> {
        self.receive_record_limit(starlings_protocol::MAX_JSON_BYTES)
    }
    pub(crate) fn receive_record_limit<T: DeserializeOwned>(&mut self, limit: usize) -> Result<T> {
        let result = (|| {
            self.start()?;
            let mut prefix = [0; 4];
            self.wire.read_exact(&mut prefix)?;
            let length = u32::from_be_bytes(prefix) as usize;
            if length > limit {
                return Err(Error::Capacity);
            }
            self.reserve(length)?;
            let mut bytes = vec![0; length];
            self.wire.read_exact(&mut bytes)?;
            let value = serde_json::from_value(starlings_protocol::parse_json(&bytes)?)?;
            self.guard.check()?;
            Ok(value)
        })();
        if result.is_err() {
            self.guard.close();
        }
        result
    }
}
impl Drop for GenerationStream {
    fn drop(&mut self) {
        self.guard.close();
    }
}
pub(crate) fn certificate_expiry(bytes: &[u8]) -> Result<i64> {
    let (rest, cert) = x509_parser::parse_x509_certificate(bytes).map_err(|_| Error::Invalid)?;
    if !rest.is_empty() {
        return Err(Error::Invalid);
    }
    Ok(cert.validity().not_after.timestamp())
}
pub(crate) fn verify_coordinator(peer: &[u8], invitation: &Invitation) -> Result<()> {
    use rustls::{
        RootCertStore,
        client::{WebPkiServerVerifier, danger::ServerCertVerifier},
        pki_types::{CertificateDer, ServerName, UnixTime},
    };
    if !transport::same_public_key(peer, &invitation.coordinator.certificate) {
        return Err(Error::Denied);
    }
    let mut roots = RootCertStore::empty();
    roots.add(CertificateDer::from(invitation.ca.as_slice()))?;
    let verifier = WebPkiServerVerifier::builder_with_provider(
        Arc::new(roots),
        Arc::new(rustls::crypto::ring::default_provider()),
    )
    .build()
    .map_err(|_| Error::Crypto)?;
    verifier.verify_server_cert(
        &CertificateDer::from(peer),
        &[],
        &ServerName::try_from(invitation.coordinator.name()).map_err(|_| Error::Invalid)?,
        &[],
        UnixTime::now(),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests;
