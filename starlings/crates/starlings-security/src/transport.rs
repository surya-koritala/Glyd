use crate::{Error, Identity, Result, Trust, enrollment::Member};
use rcgen::PublicKeyData;
use rustls::{
    ClientConfig, ClientConnection, Connection, RootCertStore, ServerConfig, ServerConnection,
    client::{Resumption, WebPkiServerVerifier, danger::ServerCertVerifier},
    pki_types::{CertificateDer, PrivatePkcs8KeyDer, ServerName, UnixTime},
    server::{NoServerSessionStorage, WebPkiClientVerifier},
};
use starlings_protocol::NodeId;
use std::{
    io::{self, Read, Write},
    net::TcpStream,
    sync::Arc,
    time::{Duration, Instant},
};

pub(crate) const TIMEOUT: Duration = Duration::from_secs(10);
/// Inside one frame, a passed membership check is reused this long; revocation by this
/// process also shuts the socket at once (`State::reconcile`).
const RECHECK: Duration = Duration::from_millis(10);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    Control,
    Data,
}
impl Channel {
    fn alpn(self) -> &'static [u8] {
        match self {
            Self::Control => b"starlings-control/1",
            Self::Data => b"starlings/1",
        }
    }
}
pub(crate) const BOOTSTRAP: &[u8] = b"starlings-enroll/1";

/// Application bytes are exposed only after CA, exact peer identity and ALPN verification.
/// Both credentials are checked against current membership before every read/write,
/// at most `RECHECK` apart within one frame.
/// Application model authorization uses with_access or SecuredWorker.
pub struct PeerStream {
    wire: Wire,
    pub(crate) permit: Option<crate::listener::Permit>,
    frame_deadline: Option<Instant>,
    checked: Option<Instant>,
    failed: bool,
    peer: NodeId,
    role: crate::Role,
    policy: PeerPolicy,
    local: NodeId,
    pub(crate) channel: Channel,
    pub(crate) ca: Vec<u8>,
}
enum PeerPolicy {
    Authority {
        access: crate::authorization::Access,
        live: Arc<crate::authorization::LiveConnection>,
    },
    Stage(crate::StageGrant),
}
impl PeerStream {
    pub fn connect(
        socket: TcpStream,
        local: &Identity,
        trust: &Trust,
        peer: NodeId,
        channel: Channel,
    ) -> Result<Self> {
        let access = trust.access.clone().ok_or(Error::Denied)?;
        let mut current = trust.clone();
        current.members = access.members()?;
        let trust = &current;
        check_local(local, trust)?;
        let expected = trust.member(peer)?;
        let wire = client(socket, trust, expected, Some(local), channel.alpn())?;
        Ok(Self {
            peer,
            role: expected.role,
            local: local.node(),
            policy: PeerPolicy::Authority {
                live: access.register(
                    local.member.clone(),
                    trusted_peer(&wire, trust)?.clone(),
                    wire.socket.socket.try_clone()?,
                )?,
                access,
            },
            channel,
            ca: trust.ca.clone(),
            wire,
            permit: None,
            frame_deadline: None,
            checked: None,
            failed: false,
        })
    }
    pub fn accept(
        socket: TcpStream,
        local: &Identity,
        trust: &Trust,
        channel: Channel,
    ) -> Result<Self> {
        Self::accept_until(socket, local, trust, channel, Instant::now() + TIMEOUT)
    }
    pub(crate) fn accept_until(
        socket: TcpStream,
        local: &Identity,
        trust: &Trust,
        channel: Channel,
        deadline: Instant,
    ) -> Result<Self> {
        let access = trust.access.clone().ok_or(Error::Denied)?;
        let mut current = trust.clone();
        current.members = access.members()?;
        let trust = &current;
        check_local(local, trust)?;
        let wire = server_until(socket, local, Some(trust), channel.alpn(), deadline)?;
        let certificate = peer_certificate(&wire)?;
        let member = trust
            .members
            .iter()
            .find(|m| m.cluster == trust.cluster && m.certificate == certificate)
            .ok_or(Error::Denied)?;
        Ok(Self {
            peer: member.node,
            role: member.role,
            local: local.node(),
            policy: PeerPolicy::Authority {
                live: access.register(
                    local.member.clone(),
                    member.clone(),
                    wire.socket.socket.try_clone()?,
                )?,
                access,
            },
            channel,
            ca: trust.ca.clone(),
            wire,
            permit: None,
            frame_deadline: None,
            checked: None,
            failed: false,
        })
    }
    pub(crate) fn authority(
        &self,
    ) -> Result<(
        &crate::authorization::Access,
        &crate::authorization::LiveConnection,
    )> {
        match &self.policy {
            PeerPolicy::Authority { access, live } => Ok((access, live)),
            PeerPolicy::Stage(_) => Err(Error::Denied),
        }
    }
    pub(crate) fn stage_grant(&self) -> Result<&crate::StageGrant> {
        match &self.policy {
            PeerPolicy::Stage(grant) => Ok(grant),
            _ => Err(Error::Denied),
        }
    }
    pub(crate) fn local(&self) -> NodeId {
        self.local
    }
    fn check(&self) -> Result<()> {
        match &self.policy {
            PeerPolicy::Authority { access, live } => access.check(live),
            PeerPolicy::Stage(grant) => grant.check(),
        }
    }
    fn check_io(&mut self) -> io::Result<()> {
        let fresh =
            self.frame_deadline.is_some() && self.checked.is_some_and(|at| at.elapsed() < RECHECK);
        if !fresh {
            self.checked = None;
            self.check().map_err(|_| {
                io::Error::new(io::ErrorKind::PermissionDenied, "secure transport denied")
            })?;
            self.checked = Some(Instant::now());
        }
        Ok(())
    }
    /// Enrolled worker-to-worker data only; the grant came from the pinned coordinator.
    pub fn connect_stage(
        socket: TcpStream,
        local: &Identity,
        pin: &crate::Invitation,
        grant: crate::StageGrant,
    ) -> Result<Self> {
        Self::stage(socket, local, pin, grant, true, Instant::now() + TIMEOUT)
    }
    pub(crate) fn accept_stage_until(
        socket: TcpStream,
        local: &Identity,
        pin: &crate::Invitation,
        grant: crate::StageGrant,
        deadline: Instant,
    ) -> Result<Self> {
        Self::stage(socket, local, pin, grant, false, deadline)
    }
    fn stage(
        socket: TcpStream,
        local: &Identity,
        pin: &crate::Invitation,
        grant: crate::StageGrant,
        connecting: bool,
        deadline: Instant,
    ) -> Result<Self> {
        let trust = grant.trust(local, pin)?;
        let expected = trust.member(grant.peer()?)?;
        let deadline = deadline.min(grant.deadline()?);
        let mut wire = if connecting {
            client_until(
                socket,
                &trust,
                expected,
                Some(local),
                Channel::Data.alpn(),
                deadline,
            )?
        } else {
            server_until(socket, local, Some(&trust), Channel::Data.alpn(), deadline)?
        };
        if peer_certificate(&wire)? != expected.certificate {
            return Err(Error::Denied);
        }
        grant.check()?;
        wire.socket.stage = Some(grant.clone());
        Ok(Self {
            wire,
            permit: None,
            frame_deadline: None,
            checked: None,
            failed: false,
            peer: expected.node,
            role: expected.role,
            local: local.node(),
            policy: PeerPolicy::Stage(grant),
            channel: Channel::Data,
            ca: trust.ca,
        })
    }
    /// A complete control record shares one absolute deadline, including every partial read.
    pub fn receive_record<T: serde::de::DeserializeOwned>(&mut self) -> Result<T> {
        self.frame_until(Instant::now() + TIMEOUT, crate::receive)
    }
    pub fn send_record(&mut self, record: &impl serde::Serialize) -> Result<()> {
        self.frame_until(Instant::now() + TIMEOUT, |io| crate::send(io, record))
    }
    pub(crate) fn frame_until<T>(
        &mut self,
        deadline: Instant,
        operation: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        if self.failed {
            return Err(Error::Expired);
        }
        let deadline = deadline.min(Instant::now() + TIMEOUT);
        self.frame_deadline = Some(deadline);
        self.checked = None;
        let result = frame_operation(deadline, Instant::now, || operation(self));
        self.frame_deadline = None;
        self.checked = None;
        if result.is_err() {
            self.failed = true;
            let _ = self.wire.socket.socket.shutdown(std::net::Shutdown::Both);
        }
        result
    }
    /// Install exact native ownership for every TLS/deadline check in this data operation.
    pub(crate) fn admitted_frame<T>(
        &mut self,
        admission: &crate::DataAdmission<'_>,
        deadline: Instant,
        operation: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        let Some(grant) = admission.transport_grant() else {
            return self.frame_until(deadline, operation);
        };
        if !grant.same(self.stage_grant()?) {
            return Err(Error::Denied);
        }
        let old = std::mem::replace(&mut self.policy, PeerPolicy::Stage(grant.clone()));
        let old_socket = self.wire.socket.stage.replace(grant);
        let result = self.frame_until(deadline, operation);
        self.policy = old;
        self.wire.socket.stage = old_socket;
        result
    }
    #[cfg(test)]
    pub(crate) fn io_deadline(&self) -> Instant {
        self.wire.socket.deadline
    }
    fn operation_deadline(&mut self) {
        self.wire.socket.deadline = self
            .frame_deadline
            .unwrap_or_else(|| Instant::now() + TIMEOUT);
    }
    pub fn peer(&self) -> NodeId {
        self.peer
    }
    pub fn role(&self) -> crate::Role {
        self.role
    }
}
fn frame_operation<T>(
    deadline: Instant,
    mut now: impl FnMut() -> Instant,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if now() >= deadline {
        return Err(Error::Expired);
    }
    operation().and_then(|value| {
        if now() >= deadline {
            Err(Error::Expired)
        } else {
            Ok(value)
        }
    })
}
impl Read for PeerStream {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        if self.failed {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "secure transport closed",
            ));
        }
        self.check_io()?;
        self.operation_deadline();
        let result = self.wire.read(bytes).map_err(redacted_io);
        if self.check_io().is_err() {
            bytes.fill(0);
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "secure transport denied",
            ));
        }
        result
    }
}
impl Write for PeerStream {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.failed {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "secure transport closed",
            ));
        }
        self.check_io()?;
        self.operation_deadline();
        self.wire.write(bytes).map_err(redacted_io)
    }
    fn flush(&mut self) -> io::Result<()> {
        if self.failed {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "secure transport closed",
            ));
        }
        self.check_io()?;
        self.operation_deadline();
        self.wire.flush().map_err(redacted_io)
    }
}
fn redacted_io(error: io::Error) -> io::Error {
    io::Error::new(error.kind(), "secure transport failed")
}
fn check_local(local: &Identity, trust: &Trust) -> Result<()> {
    if local.cluster() != trust.cluster
        || !trust
            .members
            .iter()
            .any(|m| m.node == local.node() && m.certificate == local.member.certificate)
    {
        return Err(Error::Denied);
    }
    Ok(())
}
fn roots(ca: &[u8]) -> Result<Arc<RootCertStore>> {
    let mut roots = RootCertStore::empty();
    roots.add(CertificateDer::from(ca))?;
    Ok(Arc::new(roots))
}
pub(crate) fn client(
    socket: TcpStream,
    trust: &Trust,
    expected: &Member,
    local: Option<&Identity>,
    alpn: &[u8],
) -> Result<Wire> {
    client_until(
        socket,
        trust,
        expected,
        local,
        alpn,
        Instant::now() + TIMEOUT,
    )
}
pub(crate) fn client_until(
    mut socket: TcpStream,
    trust: &Trust,
    expected: &Member,
    local: Option<&Identity>,
    alpn: &[u8],
    deadline: Instant,
) -> Result<Wire> {
    if alpn == crate::registry::ALPN
        && let Some(local) = local
    {
        socket = crate::reconnect::client(socket, local, expected.node, deadline)
            .inspect_err(|e| crate::registry_failure("client_proof", e))?;
    }
    let builder =
        ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
            .with_protocol_versions(&[&rustls::version::TLS13])?
            .with_root_certificates(roots(&trust.ca)?);
    let mut config = if let Some(identity) = local {
        builder.with_client_auth_cert(
            vec![identity.member.certificate.clone().into()],
            PrivatePkcs8KeyDer::from(identity.key.serialize_der()).into(),
        )?
    } else {
        builder.with_no_client_auth()
    };
    config.resumption = Resumption::disabled();
    config.enable_early_data = false;
    config.alpn_protocols = vec![alpn.to_vec()];
    let name = ServerName::try_from(expected.name()).map_err(|_| Error::Invalid)?;
    let wire = Wire::handshake_until(
        Connection::Client(ClientConnection::new(Arc::new(config), name)?),
        socket,
        alpn,
        deadline,
    )
    .inspect_err(|e| crate::registry_failure("client_tls", e))?;
    if !trust.members.iter().any(|m| {
        m.node == expected.node
            && m.role == expected.role
            && (m.certificate == peer_certificate(&wire).unwrap_or_default()
                || ((alpn == crate::registry::ALPN || alpn == crate::probes::ALPN)
                    && m.role == crate::Role::Coordinator
                    && same_public_key(
                        &m.certificate,
                        peer_certificate(&wire).unwrap_or_default(),
                    )))
    }) {
        return Err(Error::Denied);
    }
    Ok(wire)
}
pub(crate) fn server_until(
    socket: TcpStream,
    identity: &Identity,
    trust: Option<&Trust>,
    alpn: &[u8],
    deadline: Instant,
) -> Result<Wire> {
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let builder = ServerConfig::builder_with_provider(provider.clone())
        .with_protocol_versions(&[&rustls::version::TLS13])?;
    let builder = if let Some(trust) = trust {
        builder.with_client_cert_verifier(
            WebPkiClientVerifier::builder_with_provider(roots(&trust.ca)?, provider)
                .build()
                .map_err(|_| Error::Crypto)?,
        )
    } else {
        builder.with_no_client_auth()
    };
    let mut config = builder.with_single_cert(
        vec![identity.member.certificate.clone().into()],
        PrivatePkcs8KeyDer::from(identity.key.serialize_der()).into(),
    )?;
    config.send_tls13_tickets = 0;
    config.session_storage = Arc::new(NoServerSessionStorage {});
    config.max_early_data_size = 0;
    config.alpn_protocols = vec![alpn.to_vec()];
    Wire::handshake_until(
        Connection::Server(ServerConnection::new(Arc::new(config))?),
        socket,
        alpn,
        deadline,
    )
}
/// Bootstrap and registry share a port; only bootstrap may omit a certificate.
/// The registry dispatcher must check exact current membership before reading a record.
pub(crate) fn registry_server(
    socket: TcpStream,
    identity: &Identity,
    ca: &[u8],
    deadline: Instant,
) -> Result<Wire> {
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let verifier = WebPkiClientVerifier::builder_with_provider(roots(ca)?, provider.clone())
        .allow_unauthenticated()
        .build()
        .map_err(|_| Error::Crypto)?;
    let mut config = ServerConfig::builder_with_provider(provider)
        .with_protocol_versions(&[&rustls::version::TLS13])?
        .with_client_cert_verifier(verifier)
        .with_single_cert(
            vec![identity.member.certificate.clone().into()],
            PrivatePkcs8KeyDer::from(identity.key.serialize_der()).into(),
        )?;
    config.send_tls13_tickets = 0;
    config.session_storage = Arc::new(NoServerSessionStorage {});
    config.max_early_data_size = 0;
    config.alpn_protocols = vec![BOOTSTRAP.to_vec(), crate::registry::ALPN.to_vec()];
    Wire::handshake_protocols(
        Connection::Server(ServerConnection::new(Arc::new(config))?),
        socket,
        &[BOOTSTRAP, crate::registry::ALPN],
        deadline,
    )
}
fn trusted_peer<'a>(wire: &Wire, trust: &'a Trust) -> Result<&'a Member> {
    let certificate = peer_certificate(wire)?;
    trust
        .members
        .iter()
        .find(|m| m.cluster == trust.cluster && m.certificate == certificate)
        .ok_or(Error::Denied)
}
pub(crate) fn peer_certificate(wire: &Wire) -> Result<&[u8]> {
    let certs = wire.connection.peer_certificates().ok_or(Error::Denied)?;
    if certs.len() != 1 {
        return Err(Error::Denied);
    }
    Ok(certs[0].as_ref())
}
pub(crate) fn bootstrap_client(
    socket: TcpStream,
    trust: &Trust,
    coordinator: NodeId,
) -> Result<Wire> {
    client(socket, trust, trust.member(coordinator)?, None, BOOTSTRAP)
}
pub(crate) fn bootstrap_server_until(
    socket: TcpStream,
    identity: &Identity,
    deadline: Instant,
) -> Result<Wire> {
    server_until(socket, identity, None, BOOTSTRAP, deadline)
}
pub(crate) fn same_public_key(a: &[u8], b: &[u8]) -> bool {
    match (
        x509_parser::parse_x509_certificate(a),
        x509_parser::parse_x509_certificate(b),
    ) {
        (Ok(([], a)), Ok(([], b))) => a.public_key().raw == b.public_key().raw,
        _ => false,
    }
}
pub(crate) fn verify_issued(member: &Member, ca: &[u8], key: &rcgen::KeyPair) -> Result<()> {
    verify_issued_at(member, ca, key, UnixTime::now())
}
pub(crate) fn verify_issued_at_creation(
    member: &Member,
    ca: &[u8],
    key: &rcgen::KeyPair,
) -> Result<()> {
    let (_, cert) =
        x509_parser::parse_x509_certificate(&member.certificate).map_err(|_| Error::Invalid)?;
    let at: u64 = (cert.validity().not_before.timestamp() + 1)
        .try_into()
        .map_err(|_| Error::Invalid)?;
    if at > crate::now()? as u64 {
        return Err(Error::Invalid);
    }
    verify_issued_at(
        member,
        ca,
        key,
        UnixTime::since_unix_epoch(Duration::from_secs(at)),
    )
}
fn verify_issued_at(member: &Member, ca: &[u8], key: &rcgen::KeyPair, at: UnixTime) -> Result<()> {
    let verifier = WebPkiServerVerifier::builder_with_provider(
        roots(ca)?,
        Arc::new(rustls::crypto::ring::default_provider()),
    )
    .build()
    .map_err(|_| Error::Crypto)?;
    let name = ServerName::try_from(member.name()).map_err(|_| Error::Invalid)?;
    verifier.verify_server_cert(
        &CertificateDer::from(member.certificate.as_slice()),
        &[],
        &name,
        &[],
        at,
    )?;
    let (remaining, certificate) =
        x509_parser::parse_x509_certificate(&member.certificate).map_err(|_| Error::Invalid)?;
    if !remaining.is_empty() || certificate.public_key().raw != key.subject_public_key_info() {
        return Err(Error::Denied);
    }
    Ok(())
}

pub(crate) struct Wire {
    connection: Connection,
    socket: DeadlineSocket,
}
impl Wire {
    fn handshake_until(
        connection: Connection,
        socket: TcpStream,
        alpn: &[u8],
        deadline: Instant,
    ) -> Result<Self> {
        Self::handshake_protocols(connection, socket, &[alpn], deadline)
    }
    fn handshake_protocols(
        mut connection: Connection,
        socket: TcpStream,
        protocols: &[&[u8]],
        deadline: Instant,
    ) -> Result<Self> {
        let mut socket = DeadlineSocket {
            socket,
            deadline,
            handshake_bytes: Some(128 * 1024),
            stage: None,
        };
        connection.set_buffer_limit(Some(64 * 1024));
        connection.complete_io(&mut socket)?;
        if connection.is_handshaking()
            || connection.protocol_version() != Some(rustls::ProtocolVersion::TLSv1_3)
            || !connection
                .alpn_protocol()
                .is_some_and(|p| protocols.contains(&p))
        {
            return Err(Error::Denied);
        }
        socket.remaining()?;
        socket.handshake_bytes = None;
        if connection.alpn_protocol() == Some(Channel::Data.alpn()) {
            socket.socket.set_nodelay(true)?;
        }
        Ok(Self { connection, socket })
    }
    pub(crate) fn protocol(&self) -> Option<&[u8]> {
        self.connection.alpn_protocol()
    }
    pub(crate) fn local_addr(&self) -> Result<std::net::SocketAddr> {
        Ok(self.socket.socket.local_addr()?)
    }
    pub(crate) fn set_deadline(&mut self, deadline: Instant) {
        self.socket.deadline = deadline;
    }
    pub(crate) fn socket(&self) -> Result<TcpStream> {
        Ok(self.socket.socket.try_clone()?)
    }
    pub fn reset_deadline(&mut self) {
        self.socket.deadline = Instant::now() + TIMEOUT;
    }
}
impl Read for Wire {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.socket.remaining()?;
        match &mut self.connection {
            Connection::Client(conn) => rustls::Stream::new(conn, &mut self.socket).read(bytes),
            Connection::Server(conn) => rustls::Stream::new(conn, &mut self.socket).read(bytes),
        }
    }
}
impl Write for Wire {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.socket.remaining()?;
        match &mut self.connection {
            Connection::Client(conn) => rustls::Stream::new(conn, &mut self.socket).write(bytes),
            Connection::Server(conn) => rustls::Stream::new(conn, &mut self.socket).write(bytes),
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        self.socket.remaining()?;
        match &mut self.connection {
            Connection::Client(conn) => rustls::Stream::new(conn, &mut self.socket).flush(),
            Connection::Server(conn) => rustls::Stream::new(conn, &mut self.socket).flush(),
        }
    }
}
pub(crate) struct DeadlineSocket {
    pub(crate) socket: TcpStream,
    pub(crate) deadline: Instant,
    pub(crate) handshake_bytes: Option<usize>,
    pub(crate) stage: Option<crate::StageGrant>,
}
impl DeadlineSocket {
    fn wait_budget(&self) -> io::Result<Duration> {
        let remaining = self.remaining()?;
        let Some(grant) = &self.stage else {
            return Ok(remaining);
        };
        let until = grant.deadline().map_err(|_| {
            io::Error::new(io::ErrorKind::PermissionDenied, "secure transport denied")
        })?;
        let permission = until
            .checked_duration_since(Instant::now())
            .filter(|d| !d.is_zero())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::PermissionDenied, "secure transport denied")
            })?;
        Ok(self.remaining()?.min(permission))
    }
    fn renew_wait(&self, error: &io::Error) -> bool {
        self.stage.is_some()
            && matches!(
                error.kind(),
                io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
            )
    }
    fn remaining(&self) -> io::Result<Duration> {
        self.deadline
            .checked_duration_since(Instant::now())
            .filter(|d| !d.is_zero())
            .ok_or_else(|| io::Error::new(io::ErrorKind::TimedOut, "secure transport deadline"))
    }
}
impl Read for DeadlineSocket {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        let limit = self.handshake_bytes.unwrap_or(bytes.len()).min(bytes.len());
        if limit == 0 && !bytes.is_empty() {
            return Err(io::Error::other("handshake byte budget"));
        }
        let count = loop {
            self.socket.set_read_timeout(Some(self.wait_budget()?))?;
            match self.socket.read(&mut bytes[..limit]) {
                // Retry only this raw OS wait, retaining TLS state and the record deadline.
                Err(error) if self.renew_wait(&error) => continue,
                result => break result?,
            }
        };
        if let Some(left) = &mut self.handshake_bytes {
            *left -= count;
        }
        Ok(count)
    }
}
impl Write for DeadlineSocket {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let limit = self.handshake_bytes.unwrap_or(bytes.len()).min(bytes.len());
        if limit == 0 && !bytes.is_empty() {
            return Err(io::Error::other("handshake byte budget"));
        }
        let count = loop {
            self.socket.set_write_timeout(Some(self.wait_budget()?))?;
            match self.socket.write(&bytes[..limit]) {
                Err(error) if self.renew_wait(&error) => continue,
                result => break result?,
            }
        };
        if let Some(left) = &mut self.handshake_bytes {
            *left -= count;
        }
        Ok(count)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.socket.flush()
    }
}

#[cfg(test)]
mod limits_tests {
    use super::*;
    use crate::{
        ListenerBudget, PeerListener,
        enrollment::tests::{enrolled, setup},
    };
    use std::{net::TcpListener, thread};
    #[test]
    fn frame_clock_preserves_deadlines_and_first_errors_without_io() {
        let start = Instant::now();
        let deadline = start + Duration::from_millis(60);
        assert_eq!(
            frame_operation(
                deadline,
                || deadline,
                || -> Result<()> { panic!("expired operation started") }
            ),
            Err(Error::Expired)
        );
        for (finish, operation, expected) in [
            (start, Ok(17), Ok(17)),
            (deadline, Ok(17), Err(Error::Expired)),
            (deadline, Err(Error::Transport), Err(Error::Transport)),
            (deadline, Err(Error::Denied), Err(Error::Denied)),
        ] {
            let mut ticks = [start, finish].into_iter();
            assert_eq!(
                frame_operation(deadline, || ticks.next().unwrap(), || operation),
                expected
            );
        }
    }
    #[test]
    fn data_protocol_enables_nodelay_at_both_endpoints_and_preserves_other_channels() {
        let (_temp, _, mut authority, files) = setup();
        let worker = enrolled(&mut authority, &files, crate::Role::Worker, "nodelay");
        let trust = authority.trust().unwrap();
        for channel in [Channel::Control, Channel::Data] {
            for initial in [false, true] {
                let expected = channel == Channel::Data || initial;
                let listener = TcpListener::bind("127.0.0.1:0").unwrap();
                let address = listener.local_addr().unwrap();
                thread::scope(|threads| {
                    let server = threads.spawn(|| {
                        let socket = listener.accept().unwrap().0;
                        socket.set_nodelay(initial).unwrap();
                        let mut peer =
                            PeerStream::accept(socket, &authority.identity, &trust, channel)
                                .unwrap();
                        assert_eq!(peer.wire.socket.socket.nodelay().unwrap(), expected);
                        assert_eq!(peer.receive_record::<u32>().unwrap(), 17);
                        peer.send_record(&18u32).unwrap();
                    });
                    let socket = TcpStream::connect(address).unwrap();
                    socket.set_nodelay(initial).unwrap();
                    let mut peer = PeerStream::connect(
                        socket,
                        &worker,
                        &trust,
                        authority.identity.node(),
                        channel,
                    )
                    .unwrap();
                    assert_eq!(peer.wire.socket.socket.nodelay().unwrap(), expected);
                    peer.send_record(&17u32).unwrap();
                    assert_eq!(peer.receive_record::<u32>().unwrap(), 18);
                    server.join().unwrap();
                });
            }
        }
    }
    #[test]
    fn handshake_bytes_and_absolute_deadline_are_enforced() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let mut socket = DeadlineSocket {
            socket: listener.accept().unwrap().0,
            deadline: Instant::now() + Duration::from_millis(100),
            handshake_bytes: Some(8),
            stage: None,
        };
        client.write_all(&[0; 16]).unwrap();
        let mut bytes = [0; 16];
        assert_eq!(socket.read(&mut bytes).unwrap(), 8);
        assert!(socket.read(&mut bytes).is_err());
        assert!(socket.write(b"x").is_err());
        socket.handshake_bytes = None;
        socket.deadline = Instant::now() - Duration::from_millis(1);
        assert!(socket.read(&mut bytes).is_err());
        assert!(socket.write(b"x").is_err());
    }
    #[test]
    fn a_revocation_by_another_connection_stops_a_frame_within_the_recheck_window() {
        let (_temp, path, mut authority, files) = setup();
        let worker = enrolled(&mut authority, &files, crate::Role::Worker, "recheck");
        let trust = authority.trust().unwrap();
        let (mut server, mut client) =
            crate::enrollment::tests::pair(&authority.identity, &worker, &trust);
        client.write_all(&[7; 64]).unwrap();
        client.flush().unwrap();
        // (as an administration command in another process would write)
        let other = rusqlite::Connection::open(path.join("enrollment.sqlite")).unwrap();
        let result = server.frame_until(Instant::now() + TIMEOUT, |io| {
            let mut byte = [0; 1];
            io.read_exact(&mut byte)?;
            other
                .execute(
                    "INSERT INTO revoked (node) VALUES (?1)",
                    [worker.node().to_string()],
                )
                .unwrap();
            thread::sleep(RECHECK);
            io.read_exact(&mut byte)?;
            Ok(())
        });
        assert!(
            result.is_err(),
            "a read past the window rechecks membership"
        );
        assert!(server.receive_record::<u32>().is_err());
    }
    #[test]
    fn failed_frame_cannot_reuse_buffered_plaintext() {
        let (_temp, _, mut authority, files) = setup();
        let worker = enrolled(&mut authority, &files, crate::Role::Worker, "failed-frame");
        let trust = authority.trust().unwrap();
        let (mut server, mut client) =
            crate::enrollment::tests::pair(&authority.identity, &worker, &trust);
        let mut bytes = ((crate::MAX_RECORD + 1) as u32).to_be_bytes().to_vec();
        bytes.extend_from_slice(&2u32.to_be_bytes());
        bytes.extend_from_slice(b"{}");
        client.write_all(&bytes).unwrap();
        client.flush().unwrap();
        assert!(server.receive_record::<serde_json::Value>().is_err());
        assert!(server.receive_record::<serde_json::Value>().is_err());
        assert!(server.send_record(&serde_json::json!({})).is_err());
    }
    #[test]
    fn admitted_tls_and_control_records_release_permits() {
        let (_temp, _, mut authority, files) = setup();
        let worker = enrolled(&mut authority, &files, crate::Role::Worker, "bounded");
        let trust = authority.trust().unwrap();
        let budget = ListenerBudget::default();
        let listener = PeerListener::bind("127.0.0.1:0".parse().unwrap(), budget.clone()).unwrap();
        let address = listener.local_addr().unwrap();
        thread::scope(|s| {
            let client = s.spawn(|| {
                let mut stream = PeerStream::connect(
                    TcpStream::connect(address).unwrap(),
                    &worker,
                    &trust,
                    authority.identity.node(),
                    Channel::Control,
                )
                .unwrap();
                stream
                    .send_record(&serde_json::json!({"bounded":true}))
                    .unwrap();
                // A valid TLS peer drips a second frame, never completing it.
                let _ = stream.write_all(&100u32.to_be_bytes());
                for _ in 0..30 {
                    if stream.write_all(b" ").is_err() {
                        break;
                    }
                    thread::sleep(Duration::from_millis(20));
                }
            });
            let mut peer = listener
                .accept()
                .unwrap()
                .authenticate(&authority.identity, &trust, Channel::Control)
                .unwrap();
            assert_eq!(budget.usage().unwrap().connections, 1);
            assert_eq!(budget.usage().unwrap().handshakes, 0);
            assert_eq!(
                peer.receive_record::<serde_json::Value>().unwrap(),
                serde_json::json!({"bounded":true})
            );
            peer.frame_deadline = Some(Instant::now() + Duration::from_millis(150));
            let start = Instant::now();
            assert!(crate::receive::<serde_json::Value>(&mut peer).is_err());
            assert!(start.elapsed() < Duration::from_secs(1));
            drop(peer);
            client.join().unwrap();
        });
        assert_eq!(budget.usage().unwrap().connections, 0);
    }
}
