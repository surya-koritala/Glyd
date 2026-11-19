use crate::{Error, MAX_RECORD, Result, decode, hash, now, receive, send, transport};
use private_fs::PrivateDir;
use rcgen::{
    BasicConstraints, CertificateParams, CertificateSigningRequestParams, DistinguishedName,
    DnType, ExtendedKeyUsagePurpose, IsCa, Issuer, KeyPair, KeyUsagePurpose, PublicKeyData,
};
use ring::rand::{SecureRandom, SystemRandom};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};
use starlings_protocol::{ClusterId, NodeId, Version};
use std::{
    io::Read,
    net::{SocketAddr, TcpStream},
    path::Path,
};
use time::{Duration, OffsetDateTime};
use x509_parser::prelude::{FromDer, X509CertificationRequest};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Coordinator,
    Worker,
    Controller,
}
impl Role {
    fn label(self) -> &'static str {
        match self {
            Self::Coordinator => "coordinator",
            Self::Worker => "worker",
            Self::Controller => "controller",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Member {
    pub version: Version,
    pub cluster: ClusterId,
    pub node: NodeId,
    pub role: Role,
    pub certificate: Vec<u8>,
}
impl Member {
    pub fn name(&self) -> String {
        format!(
            "{}.{}.{}.glyd.invalid",
            self.node,
            self.role.label(),
            self.cluster
        )
    }
}

/// Contains private key material: deliberately has no Debug or serialization API.
pub struct Identity {
    pub(crate) member: Member,
    pub(crate) key: KeyPair,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedIdentity {
    version: Version,
    member: Member,
    key: Vec<u8>,
}
impl Identity {
    pub fn generate(cluster: ClusterId) -> Result<Self> {
        Ok(Self {
            member: Member {
                version: Version,
                cluster,
                node: uuid::Uuid::new_v4()
                    .to_string()
                    .parse()
                    .map_err(|_| Error::Invalid)?,
                role: Role::Worker,
                certificate: vec![],
            },
            key: KeyPair::generate()?,
        })
    }
    pub fn node(&self) -> NodeId {
        self.member.node
    }
    pub fn cluster(&self) -> ClusterId {
        self.member.cluster
    }
    pub fn role(&self) -> Role {
        self.member.role
    }
    /// Compare this fingerprint over an authenticated out-of-band channel before approving.
    pub fn request_fingerprint(&self) -> String {
        fingerprint(self.node(), &self.key.subject_public_key_info())
    }
    pub(crate) fn csr(&self) -> Result<Vec<u8>> {
        // No caller-controlled certificate extensions cross the approval boundary.
        Ok(CertificateParams::default()
            .serialize_request(&self.key)?
            .der()
            .to_vec())
    }
    pub fn save(&self, directory: &PrivateDir, name: &str) -> Result<()> {
        create_record(
            directory,
            name,
            &SavedIdentity {
                version: Version,
                member: self.member.clone(),
                key: self.key.serialize_der(),
            },
        )
    }
    pub(crate) fn replace_file(&self, directory: &PrivateDir, name: &str) -> Result<()> {
        directory.write_atomic_durable(
            name,
            &serde_json::to_vec(&SavedIdentity {
                version: Version,
                member: self.member.clone(),
                key: self.key.serialize_der(),
            })?,
        )?;
        Ok(())
    }
    pub fn load(directory: &PrivateDir, name: &str) -> Result<Self> {
        let saved: SavedIdentity = read_record(directory, name)?;
        Ok(Self {
            member: saved.member,
            key: KeyPair::try_from(saved.key)?,
        })
    }
}

/// Live local membership. Bootstrap invitations use a separate pinned snapshot.
#[derive(Clone)]
pub struct Trust {
    pub(crate) ca: Vec<u8>,
    pub(crate) cluster: ClusterId,
    pub(crate) members: Vec<Member>,
    pub(crate) access: Option<crate::authorization::Access>,
}
impl Trust {
    pub(crate) fn member(&self, node: NodeId) -> Result<&Member> {
        self.members
            .iter()
            .find(|m| m.node == node && m.cluster == self.cluster)
            .ok_or(Error::Denied)
    }
}

/// The entire file is a credential. Transfer privately; never put its contents in arguments.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Invitation {
    version: Version,
    pub(crate) endpoint: SocketAddr,
    pub(crate) coordinator: Member,
    pub(crate) ca: Vec<u8>,
    token: [u8; 32],
    expires: i64,
}
impl Invitation {
    pub fn load(directory: &PrivateDir, name: &str) -> Result<Self> {
        let invite: Self = read_record(directory, name)?;
        if invite.coordinator.role != Role::Coordinator
            || invite.ca.len() > 4096
            || invite.coordinator.certificate.len() > 4096
            || invite.endpoint.port() == 0
        {
            return Err(Error::Invalid);
        }
        Ok(invite)
    }
    pub fn endpoint(&self) -> SocketAddr {
        self.endpoint
    }
    /// Explicit local override retains the invitation certificate pin and approval.
    pub fn set_endpoint(&mut self, endpoint: SocketAddr) -> Result<()> {
        crate::network::validate_routes(&[endpoint], None)?;
        self.endpoint = endpoint;
        Ok(())
    }
    pub fn cluster(&self) -> ClusterId {
        self.coordinator.cluster
    }
    pub fn verify_identity(&self, identity: &Identity) -> Result<()> {
        if identity.cluster() != self.cluster()
            || !matches!(identity.role(), Role::Worker | Role::Controller)
        {
            return Err(Error::Denied);
        }
        transport::verify_issued(&identity.member, &self.ca, &identity.key)
    }
    pub fn enroll(&self, identity: &mut Identity) -> Result<()> {
        if now()? >= self.expires {
            return Err(Error::Expired);
        }
        if identity.cluster() != self.cluster() || !identity.member.certificate.is_empty() {
            return Err(Error::Denied);
        }
        let trust = Trust {
            ca: self.ca.clone(),
            cluster: self.cluster(),
            members: vec![self.coordinator.clone()],
            access: None,
        };
        let socket = crate::network::connect(self.endpoint, None, transport::TIMEOUT)?;
        let mut stream = transport::bootstrap_client(socket, &trust, self.coordinator.node)?;
        send(
            &mut stream,
            &Request {
                version: Version,
                node: identity.node(),
                token: self.token,
                csr: identity.csr()?,
            },
        )?;
        let issued: Member = receive(&mut stream)?;
        if issued.node != identity.node()
            || issued.cluster != identity.cluster()
            || issued.certificate.len() > 4096
        {
            return Err(Error::Denied);
        }
        transport::verify_issued(&issued, &self.ca, &identity.key)?;
        identity.member = issued;
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    version: Version,
    node: NodeId,
    token: [u8; 32],
    csr: Vec<u8>,
}

/// Only this object can be approved; it binds the CSR, credential and bootstrap connection.
pub struct PendingEnrollment {
    pub(crate) permit: Option<crate::listener::Permit>,
    request: Request,
    stream: transport::Wire,
    cluster: ClusterId,
    ca_hash: String,
    fingerprint: String,
    role: Role,
}
impl PendingEnrollment {
    pub fn role(&self) -> Role {
        self.role
    }
    pub fn node(&self) -> NodeId {
        self.request.node
    }
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }
}

/// An already committed enrollment response. Delivery never holds the control loop or signing key.
pub struct ApprovedEnrollment {
    pending: PendingEnrollment,
    member: Member,
}
impl ApprovedEnrollment {
    pub fn deliver(mut self) -> Result<()> {
        self.pending.stream.reset_deadline();
        send(&mut self.pending.stream, &self.member)
    }
}

pub struct EnrollmentReader {
    pub(crate) identity: Identity,
    pub(crate) access: crate::authorization::Access,
    pub(crate) ca: Vec<u8>,
    ca_hash: String,
}
impl EnrollmentReader {
    /// Connect using the current coordinator leaf and shared live policy, never a CA key.
    pub fn connect_generation(
        &self,
        socket: TcpStream,
        worker: NodeId,
        binding: crate::GenerationBinding,
    ) -> Result<crate::GenerationStream> {
        let trust = Trust {
            ca: self.ca.clone(),
            cluster: self.identity.cluster(),
            members: self.access.members()?,
            access: Some(self.access.clone()),
        };
        crate::GenerationStream::connect(socket, &self.identity, &trust, worker, binding)
    }
    pub(crate) fn receive(
        self,
        socket: TcpStream,
        deadline: std::time::Instant,
    ) -> Result<PendingEnrollment> {
        let stream = transport::bootstrap_server_until(socket, &self.identity, deadline)?;
        self.receive_stream(stream)
    }
    pub(crate) fn receive_stream(self, mut stream: transport::Wire) -> Result<PendingEnrollment> {
        let request: Request = receive(&mut stream)?;
        let role = invitation_role(&self.access.lock()?.database, &request.token, now()?)?;
        let csr = checked_csr(&request.csr)?;
        let fingerprint = fingerprint(request.node, &csr.public_key.subject_public_key_info());
        Ok(PendingEnrollment {
            permit: None,
            request,
            stream,
            cluster: self.identity.cluster(),
            ca_hash: self.ca_hash,
            fingerprint,
            role,
        })
    }
}

/// Local privileged enrollment authority. Never expose these methods as anonymous routes.
pub struct Authority {
    pub(crate) directory: PrivateDir,
    pub(crate) access: crate::authorization::Access,
    ca: Vec<u8>,
    pub(crate) issuer: Issuer<'static, KeyPair>,
    pub(crate) identity: Identity,
}
/// Invitation publication access only; contains public certificates, never signing keys.
pub struct InvitationIssuer {
    access: crate::authorization::Access,
    coordinator: Member,
    ca: Vec<u8>,
}
impl InvitationIssuer {
    /// Publish outside the policy lock, then commit only if admission is still live.
    /// The phase observer must not block or reenter policy operations.
    pub fn invite_observed(
        &self,
        directory: &PrivateDir,
        name: &str,
        endpoint: SocketAddr,
        role: Role,
        phase: impl FnMut(&'static str),
    ) -> Result<()> {
        self.invite_with(endpoint, role, phase, |invite| {
            create_record(directory, name, invite)
        })
    }
    fn invite_with(
        &self,
        endpoint: SocketAddr,
        role: Role,
        mut phase: impl FnMut(&'static str),
        publish: impl FnOnce(&Invitation) -> Result<()>,
    ) -> Result<()> {
        if role == Role::Coordinator || endpoint.port() == 0 {
            return Err(Error::Denied);
        }
        phase("clock");
        let current = now()?;
        let expires = current.checked_add(300).ok_or(Error::Invalid)?;
        phase("policy_lock");
        {
            let state = self.access.lock()?;
            if state.draining {
                return Err(Error::Denied);
            }
            phase("expired_rows");
            state
                .database
                .execute("DELETE FROM invitations WHERE expires <= ?1", [current])?;
            let count: i64 =
                state
                    .database
                    .query_row("SELECT count(*) FROM invitations", [], |r| r.get(0))?;
            if count >= 128 {
                return Err(Error::Capacity);
            }
        }
        let mut token = [0; 32];
        SystemRandom::new()
            .fill(&mut token)
            .map_err(|_| Error::Crypto)?;
        // No policy mutex or SQLite transaction spans private filesystem publication.
        phase("publish_private_file");
        publish(&Invitation {
            version: Version,
            endpoint,
            coordinator: self.coordinator.clone(),
            ca: self.ca.clone(),
            token,
            expires,
        })?;
        phase("policy_lock");
        let mut state = self.access.lock()?;
        if state.draining {
            return Err(Error::Denied);
        }
        if state.clock_floor() >= expires {
            return Err(Error::Expired);
        }
        let state_clock = state.clock_floor();
        phase("transaction");
        let committed = (|| {
            let transaction = state
                .database
                .transaction_with_behavior(TransactionBehavior::Immediate)?;
            let count: i64 =
                transaction.query_row("SELECT count(*) FROM invitations", [], |r| r.get(0))?;
            if count >= 128 {
                return Err(Error::Capacity);
            }
            transaction.execute(
                "INSERT INTO invitations VALUES (?1,?2,?3)",
                (hash(&token), expires, serde_json::to_string(&role)?),
            )?;
            // Commit the rollback floor with the token, without separate synchronous checkpoints.
            if transaction.execute(
                "UPDATE clock SET observed=MAX(observed,?1) WHERE id=1",
                [current.max(state_clock)],
            )? != 1
            {
                return Err(Error::Storage);
            }
            // A crash/failure before commit leaves an unusable file, never an untracked valid token.
            phase("commit");
            transaction.commit()?;
            Ok(())
        })();
        if committed == Err(Error::Storage) {
            state.fail_closed();
        }
        committed?;
        phase("finished");
        Ok(())
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedAuthority {
    version: Version,
    ca: Vec<u8>,
    key: Vec<u8>,
}

impl Authority {
    pub fn create(path: &Path) -> Result<Self> {
        Self::create_observed(path, |_| {})
    }
    /// Reports fixed operation names only. The observer must not block or reenter creation.
    pub fn create_observed(path: &Path, mut phase: impl FnMut(&'static str)) -> Result<Self> {
        phase("directory_open");
        let directory = match PrivateDir::open(path) {
            Ok(directory) => directory,
            Err(private_fs::Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                // Drop the creator's DELETE-capable handle before retaining read-only ownership.
                PrivateDir::ensure(path)?;
                PrivateDir::open(path)?
            }
            Err(error) => return Err(error.into()),
        };
        // Exclusive creation refuses to overwrite any prior authority, including partial init.
        phase("enrollment_create");
        let enrollment = directory.create("enrollment.sqlite")?;
        phase("enrollment_sync");
        enrollment.sync_all()?;
        drop(enrollment);
        phase("root_key");
        let key = KeyPair::generate()?;
        let mut ca_params = CertificateParams::default();
        ca_params.distinguished_name = DistinguishedName::new();
        ca_params
            .distinguished_name
            .push(DnType::CommonName, "Starlings cluster root");
        ca_params.is_ca = IsCa::Ca(BasicConstraints::Constrained(0));
        ca_params.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];
        ca_params.not_before = OffsetDateTime::now_utc() - Duration::minutes(1);
        ca_params.not_after = OffsetDateTime::now_utc() + Duration::days(3650);
        phase("root_certificate");
        let ca = ca_params.self_signed(&key)?.der().to_vec();
        phase("authority_publish");
        create_record(
            &directory,
            "authority.json",
            &SavedAuthority {
                version: Version,
                ca: ca.clone(),
                key: key.serialize_der(),
            },
        )?;
        let issuer = Issuer::new(ca_params, key);
        let cluster = uuid::Uuid::new_v4()
            .to_string()
            .parse()
            .map_err(|_| Error::Invalid)?;
        phase("identity_key");
        let mut identity = Identity::generate(cluster)?;
        identity.member.role = Role::Coordinator;
        phase("coordinator_certificate");
        identity.member.certificate = certificate_params(&identity.member)?
            .signed_by(&identity.key, &issuer)?
            .der()
            .to_vec();
        phase("coordinator_publish");
        identity.save(&directory, "coordinator.json")?;
        phase("database_open");
        let database = open_database(path, &directory)?;
        phase("database_schema");
        let transaction = database.unchecked_transaction()?;
        transaction.execute_batch("CREATE TABLE invitations (hash TEXT PRIMARY KEY, expires INTEGER NOT NULL, role TEXT NOT NULL);
            CREATE TABLE members (node TEXT PRIMARY KEY, record BLOB NOT NULL);
            CREATE TABLE clock (id INTEGER PRIMARY KEY CHECK(id=1), observed INTEGER NOT NULL);
            INSERT INTO clock VALUES (1,0);")?;
        phase("member_insert");
        transaction.execute(
            "INSERT INTO members VALUES (?1,?2)",
            (
                identity.node().to_string(),
                serde_json::to_vec(&identity.member)?,
            ),
        )?;
        transaction.commit()?;
        phase("ready_publish");
        create_record(&directory, "ready.json", &Version)?;
        phase("database_close");
        drop(database);
        phase("access_open");
        let access = crate::authorization::Access::open(path, identity.cluster())?;
        phase("finished");
        Ok(Self {
            directory,
            access,
            ca,
            issuer,
            identity,
        })
    }
    pub fn open(path: &Path) -> Result<Self> {
        let directory = PrivateDir::open(path)?;
        let _: Version = read_record(&directory, "ready.json")?;
        let saved: SavedAuthority = read_record(&directory, "authority.json")?;
        let key = KeyPair::try_from(saved.key)?;
        let (remaining, certificate) =
            x509_parser::parse_x509_certificate(&saved.ca).map_err(|_| Error::Invalid)?;
        if !remaining.is_empty()
            || !certificate.is_ca()
            || !certificate.validity().is_valid()
            || certificate.public_key().raw != key.subject_public_key_info()
        {
            return Err(Error::Denied);
        }
        let issuer = Issuer::from_ca_cert_der(&saved.ca.clone().into(), key)?;
        let mut identity = Identity::load(&directory, "coordinator.json")?;
        let access = crate::authorization::Access::open(path, identity.cluster())?;
        // Published pointer and certificate are one SQLite commit. A pre-commit crash only
        // leaves an unused protected key file; a post-commit restart loads the new identity.
        {
            let state = access.inspect()?;
            let db = &state.database;
            let saved: Option<String> = db
                .query_row(
                    "SELECT value FROM settings WHERE name='coordinator'",
                    [],
                    |r| r.get(0),
                )
                .optional()?;
            if let Some(name) = saved {
                identity = Identity::load(&directory, &name)?;
            }
        }
        if identity.role() != Role::Coordinator {
            return Err(Error::Denied);
        }
        // Verify the persisted CA, identity, key and membership before local expiry recovery.
        transport::verify_issued_at_creation(&identity.member, &saved.ca, &identity.key)?;
        if access.inspect()?.member(identity.node())?.certificate != identity.member.certificate {
            return Err(Error::Denied);
        }
        let mut authority = Self {
            directory,
            access,
            ca: saved.ca,
            issuer,
            identity,
        };
        authority.refresh_coordinator()?;
        transport::verify_issued(
            &authority.identity.member,
            &authority.ca,
            &authority.identity.key,
        )?;
        Ok(authority)
    }
    pub fn identity(&self) -> &Identity {
        &self.identity
    }
    pub fn trust(&self) -> Result<Trust> {
        let members = self.access.members()?;
        Ok(Trust {
            ca: self.ca.clone(),
            cluster: self.identity.cluster(),
            members,
            access: Some(self.access.clone()),
        })
    }
    /// Inspect the exact protected credential without issuing, consuming or extending it.
    pub fn invitation_active(&self, directory: &PrivateDir, name: &str) -> Result<bool> {
        let invite = Invitation::load(directory, name)?;
        if invite.ca != self.ca
            || invite.coordinator.name() != self.identity.member.name()
            || !transport::same_public_key(
                &invite.coordinator.certificate,
                &self.identity.member.certificate,
            )
        {
            return Err(Error::Denied);
        }
        let state = self.access.lock()?;
        if state.draining {
            return Ok(false);
        }
        let row: Option<(i64, String)> = state
            .database
            .query_row(
                "SELECT expires,role FROM invitations WHERE hash=?1",
                [hash(&invite.token)],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        let Some((expires, role)) = row else {
            return Ok(false);
        };
        if expires != invite.expires || state.clock_floor() >= expires {
            return Ok(false);
        }
        let role: Role = serde_json::from_str(&role)?;
        if !matches!(role, Role::Worker | Role::Controller) {
            return Err(Error::Denied);
        }
        Ok(true)
    }
    /// Default invitation lifetime is five minutes. Only the hash is stored by this authority.
    pub fn invite(
        &mut self,
        directory: &PrivateDir,
        name: &str,
        endpoint: SocketAddr,
        role: Role,
    ) -> Result<()> {
        self.invite_observed(directory, name, endpoint, role, |_| {})
    }
    /// Phase observer must not block or re-enter the authority; no credential data is supplied.
    pub fn invite_observed(
        &mut self,
        directory: &PrivateDir,
        name: &str,
        endpoint: SocketAddr,
        role: Role,
        phase: impl FnMut(&'static str),
    ) -> Result<()> {
        self.invitation_issuer()
            .invite_observed(directory, name, endpoint, role, phase)
    }
    /// Copy public invitation identity and shared policy access; no CA or node key leaves this owner.
    pub fn invitation_issuer(&self) -> InvitationIssuer {
        InvitationIssuer {
            access: self.access.clone(),
            coordinator: self.identity.member.clone(),
            ca: self.ca.clone(),
        }
    }
    pub fn receive_enrollment(&self, socket: TcpStream) -> Result<PendingEnrollment> {
        self.receive_enrollment_until(socket, std::time::Instant::now() + transport::TIMEOUT)
    }
    pub(crate) fn receive_enrollment_until(
        &self,
        socket: TcpStream,
        deadline: std::time::Instant,
    ) -> Result<PendingEnrollment> {
        self.enrollment_reader()?.receive(socket, deadline)
    }
    /// A bounded handshake worker owns only the node TLS credential and shared policy access,
    /// not the cluster signing key. Network I/O never holds the authorization lock.
    pub fn enrollment_reader(&self) -> Result<EnrollmentReader> {
        Ok(EnrollmentReader {
            identity: Identity {
                member: self.identity.member.clone(),
                key: KeyPair::try_from(self.identity.key.serialize_der())?,
            },
            access: self.access.clone(),
            ca: self.ca.clone(),
            ca_hash: hash(&self.ca),
        })
    }
    /// Call only after the local administrator compares the request fingerprint with the device.
    /// A lost response after commit requires a new invitation and node identity; never replay.
    pub fn approve(
        &mut self,
        pending: PendingEnrollment,
        approved_fingerprint: &str,
    ) -> Result<()> {
        self.approve_for_delivery(pending, approved_fingerprint)?
            .deliver()
    }
    pub fn approve_for_delivery(
        &mut self,
        pending: PendingEnrollment,
        approved_fingerprint: &str,
    ) -> Result<ApprovedEnrollment> {
        if approved_fingerprint != pending.fingerprint()
            || pending.cluster != self.identity.cluster()
            || pending.ca_hash != hash(&self.ca)
        {
            return Err(Error::Denied);
        }
        let member = self.issue(&pending.request)?;
        Ok(ApprovedEnrollment { pending, member })
    }
    fn issue(&mut self, request: &Request) -> Result<Member> {
        let current = self.check_clock()?;
        let mut csr = checked_csr(&request.csr)?;
        let mut state = self.access.lock()?;
        if state.draining {
            return Err(Error::Denied);
        }
        let transaction = state
            .database
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let role = invitation_role(&transaction, &request.token, now()?.max(current))?;
        let count: i64 = transaction.query_row("SELECT count(*) FROM members", [], |r| r.get(0))?;
        if count >= 64 {
            return Err(Error::Capacity);
        }
        if transaction.query_row(
            "SELECT EXISTS(SELECT 1 FROM revoked WHERE node=?1)",
            [request.node.to_string()],
            |r| r.get::<_, bool>(0),
        )? {
            return Err(Error::Denied);
        }
        let mut member = Member {
            version: Version,
            cluster: self.identity.cluster(),
            node: request.node,
            role,
            certificate: vec![],
        };
        csr.params = certificate_params(&member)?;
        member.certificate = csr.signed_by(&self.issuer)?.der().to_vec();
        transaction.execute(
            "INSERT INTO members VALUES (?1,?2)",
            (member.node.to_string(), serde_json::to_vec(&member)?),
        )?;
        transaction.execute(
            "DELETE FROM invitations WHERE hash=?1",
            [hash(&request.token)],
        )?;
        transaction.commit()?;
        Ok(member)
    }
    fn check_clock(&self) -> Result<i64> {
        self.access.checkpoint(true)?;
        now()
    }
}

fn invitation_role(database: &Connection, token: &[u8; 32], current: i64) -> Result<Role> {
    let invite: Option<(i64, String)> = database
        .query_row(
            "SELECT expires,role FROM invitations WHERE hash=?1",
            [hash(token)],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()?;
    let (expires, role) = invite.ok_or(Error::Replay)?;
    if current >= expires {
        return Err(Error::Expired);
    }
    let role: Role = serde_json::from_str(&role)?;
    if role == Role::Coordinator {
        return Err(Error::Denied);
    }
    Ok(role)
}

pub(crate) fn checked_csr(bytes: &[u8]) -> Result<CertificateSigningRequestParams> {
    if bytes.len() > 4096 {
        return Err(Error::Invalid);
    }
    let (remaining, _) = X509CertificationRequest::from_der(bytes).map_err(|_| Error::Invalid)?;
    if !remaining.is_empty() {
        return Err(Error::Invalid);
    }
    Ok(CertificateSigningRequestParams::from_der(&bytes.into())?)
}
pub(crate) fn certificate_params(member: &Member) -> Result<CertificateParams> {
    let mut params = CertificateParams::new(vec![member.name()])?;
    params.distinguished_name = DistinguishedName::new();
    params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
    params.extended_key_usages = vec![
        ExtendedKeyUsagePurpose::ServerAuth,
        ExtendedKeyUsagePurpose::ClientAuth,
    ];
    params.not_before = OffsetDateTime::now_utc() - Duration::minutes(1);
    params.not_after = OffsetDateTime::now_utc() + Duration::days(7);
    Ok(params)
}
pub(crate) fn open_database(path: &Path, directory: &PrivateDir) -> Result<Connection> {
    directory.open_file("enrollment.sqlite")?;
    // SQLite needs a path; the verified private directory excludes other principals.
    // Same-user/root namespace replacement is outside this storage boundary.
    let database = Connection::open_with_flags(
        path.join("enrollment.sqlite"),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE,
    )?;
    database.busy_timeout(std::time::Duration::ZERO)?;
    database.execute_batch("PRAGMA journal_mode=DELETE; PRAGMA synchronous=EXTRA; PRAGMA temp_store=MEMORY; PRAGMA trusted_schema=OFF;")?;
    Ok(database)
}
pub(crate) fn create_record(
    directory: &PrivateDir,
    name: &str,
    value: &impl Serialize,
) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    if bytes.len() > MAX_RECORD {
        return Err(Error::Invalid);
    }
    directory.create(name)?;
    directory.write_atomic_durable(name, &bytes)?;
    Ok(())
}
pub(crate) fn read_record<T: serde::de::DeserializeOwned>(
    directory: &PrivateDir,
    name: &str,
) -> Result<T> {
    let mut bytes = Vec::new();
    directory
        .open_file(name)?
        .take((MAX_RECORD + 1) as u64)
        .read_to_end(&mut bytes)?;
    decode(&bytes)
}

fn fingerprint(node: NodeId, public_key: &[u8]) -> String {
    hash(&[node.to_string().as_bytes(), public_key].concat())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{Channel, PeerStream};
    use std::{
        io::{Read, Write},
        net::TcpListener,
        sync::{Arc, Barrier},
        thread,
    };
    #[test]
    fn initialization_batches_durable_schema_and_member_commits() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("batched-initialization");
        let authority = Authority::create(&path).unwrap();
        let mut header = [0; 100];
        PrivateDir::open(&path)
            .unwrap()
            .open_file("enrollment.sqlite")
            .unwrap()
            .read_exact(&mut header)
            .unwrap();
        assert_eq!(&header[..16], b"SQLite format 3\0");
        assert_eq!(
            &header[18..20],
            &[1, 1],
            "this check requires rollback journaling"
        );
        // SQLite's documented header counter advances on each modified unlock.
        let commits = u32::from_be_bytes(header[24..28].try_into().unwrap());
        eprintln!("INIT_MODIFYING_TRANSACTIONS={commits}");
        assert!(
            commits <= 2,
            "initialization used {commits} modifying transactions"
        );
        let observer = Connection::open(path.join("enrollment.sqlite")).unwrap();
        assert_eq!(
            observer
                .query_row("SELECT count(*) FROM members", [], |r| r.get::<_, u32>(0))
                .unwrap(),
            1
        );
        for table in [
            "invitations",
            "clock",
            "revoked",
            "grants",
            "tokens",
            "previous",
            "settings",
        ] {
            observer.prepare(&format!("SELECT * FROM {table}")).unwrap();
        }
        assert_eq!(authority.identity().role(), Role::Coordinator);
        {
            let state = authority.access.inspect().unwrap();
            assert_eq!(
                state
                    .database
                    .query_row("PRAGMA synchronous", [], |r| r.get::<_, u32>(0))
                    .unwrap(),
                3
            );
            assert_eq!(
                state
                    .database
                    .query_row("PRAGMA busy_timeout", [], |r| r.get::<_, u32>(0))
                    .unwrap(),
                0
            );
        }
        drop(authority);
        Authority::open(&path).unwrap();
    }

    pub(crate) fn setup() -> (tempfile::TempDir, std::path::PathBuf, Authority, PrivateDir) {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let path = root.join("authority");
        let authority = Authority::create(&path).unwrap();
        let files = PrivateDir::ensure(root.join("device")).unwrap();
        (temp, path, authority, files)
    }
    fn invitation(authority: &mut Authority, files: &PrivateDir, name: &str) -> Invitation {
        authority
            .invite(files, name, "127.0.0.1:9741".parse().unwrap(), Role::Worker)
            .unwrap();
        Invitation::load(files, name).unwrap()
    }
    fn request(invitation: &Invitation, identity: &Identity) -> Request {
        Request {
            version: Version,
            node: identity.node(),
            token: invitation.token,
            csr: identity.csr().unwrap(),
        }
    }
    #[test]
    fn paused_invitation_publication_cannot_block_policy_maintenance() {
        let (_temp, _path, authority, files) = setup();
        let (entered, paused) = std::sync::mpsc::sync_channel(1);
        let (release, resume) = std::sync::mpsc::sync_channel(1);
        let (maintained, completed) = std::sync::mpsc::sync_channel(1);
        let responsive = thread::scope(|threads| {
            let authority = &authority;
            let files = &files;
            let publishing = threads.spawn(move || {
                authority.invitation_issuer().invite_with(
                    "127.0.0.1:9741".parse().unwrap(),
                    Role::Worker,
                    |_| {},
                    |invite| {
                        entered.send(()).unwrap();
                        resume
                            .recv_timeout(std::time::Duration::from_secs(3))
                            .unwrap();
                        create_record(files, "paused.json", invite)
                    },
                )
            });
            paused
                .recv_timeout(std::time::Duration::from_secs(3))
                .unwrap();
            let maintenance = threads.spawn(|| {
                let result = authority.maintain(0);
                maintained.send(result).unwrap();
            });
            let responsive = completed
                .recv_timeout(std::time::Duration::from_secs(1))
                .is_ok_and(|result| result.is_ok());
            release.send(()).unwrap();
            publishing.join().unwrap().unwrap();
            maintenance.join().unwrap();
            responsive
        });
        assert!(
            responsive,
            "private invitation publication blocked real policy maintenance"
        );
        assert!(authority.invitation_active(&files, "paused.json").unwrap());
    }

    #[test]
    fn publication_rechecks_drain_and_capacity_before_commit() {
        for drain in [true, false] {
            let (_temp, _path, authority, files) = setup();
            let result = authority.invitation_issuer().invite_with(
                "127.0.0.1:9741".parse().unwrap(),
                Role::Worker,
                |_| {},
                |invite| {
                    create_record(&files, "racing.json", invite)?;
                    if drain {
                        authority.drain()?;
                    } else {
                        let state = authority.access.lock()?;
                        for n in 0..128 {
                            state.database.execute(
                                "INSERT INTO invitations VALUES (?1,?2,?3)",
                                (
                                    format!("reserved-{n}"),
                                    invite.expires,
                                    serde_json::to_string(&Role::Worker)?,
                                ),
                            )?;
                        }
                    }
                    Ok(())
                },
            );
            assert_eq!(
                result,
                Err(if drain {
                    Error::Denied
                } else {
                    Error::Capacity
                })
            );
            assert!(!authority.invitation_active(&files, "racing.json").unwrap());
        }
    }

    #[test]
    fn failed_clock_floor_update_rolls_back_the_invitation_row() {
        let (_temp, _path, authority, files) = setup();
        authority.access.lock().unwrap().database.execute_batch(
            "CREATE TRIGGER refuse_clock BEFORE UPDATE ON clock BEGIN SELECT RAISE(ABORT, 'injected clock failure'); END;"
        ).unwrap();
        assert!(
            authority
                .invitation_issuer()
                .invite_observed(
                    &files,
                    "clock-failure.json",
                    "127.0.0.1:9741".parse().unwrap(),
                    Role::Worker,
                    |_| {}
                )
                .is_err()
        );
        assert!(
            files.open_file("clock-failure.json").is_ok(),
            "published file stays for inspection"
        );
        let state = authority.access.lock().unwrap();
        let count: i64 = state
            .database
            .query_row("SELECT count(*) FROM invitations", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0, "clock failure must not leave a usable token row");
        drop(state);
        assert!(
            !authority
                .invitation_active(&files, "clock-failure.json")
                .unwrap()
        );
    }

    #[test]
    fn invitation_status_requires_commit_and_never_consumes_or_extends_the_token() {
        let (_temp, path, mut authority, files) = setup();
        let mut invite = invitation(&mut authority, &files, "status.json");
        let original = serde_json::to_vec(&invite).unwrap();
        let before = authority.access.inspect().unwrap().database.total_changes();
        assert!(authority.invitation_active(&files, "status.json").unwrap());
        assert!(authority.invitation_active(&files, "status.json").unwrap());
        assert_eq!(
            authority.access.inspect().unwrap().database.total_changes(),
            before
        );
        assert_eq!(
            serde_json::to_vec(&Invitation::load(&files, "status.json").unwrap()).unwrap(),
            original
        );
        invite.expires -= 1;
        files
            .write_atomic_durable("status.json", &serde_json::to_vec(&invite).unwrap())
            .unwrap();
        assert!(!authority.invitation_active(&files, "status.json").unwrap());
        invite.expires += 1;
        invite.token[0] ^= 1;
        files
            .write_atomic_durable("status.json", &serde_json::to_vec(&invite).unwrap())
            .unwrap();
        assert!(!authority.invitation_active(&files, "status.json").unwrap());
        files
            .write_atomic_durable("status.json", &original)
            .unwrap();
        authority
            .access
            .lock()
            .unwrap()
            .database
            .execute(
                "UPDATE invitations SET role=?1 WHERE hash=?2",
                (
                    serde_json::to_string(&Role::Coordinator).unwrap(),
                    hash(&Invitation::load(&files, "status.json").unwrap().token),
                ),
            )
            .unwrap();
        assert_eq!(
            authority.invitation_active(&files, "status.json"),
            Err(Error::Denied)
        );
        authority
            .access
            .lock()
            .unwrap()
            .database
            .execute(
                "UPDATE invitations SET role=?1 WHERE hash=?2",
                (
                    serde_json::to_string(&Role::Worker).unwrap(),
                    hash(&Invitation::load(&files, "status.json").unwrap().token),
                ),
            )
            .unwrap();

        let published = Invitation {
            version: Version,
            endpoint: invite.endpoint,
            coordinator: invite.coordinator.clone(),
            ca: invite.ca.clone(),
            token: [17; 32],
            expires: invite.expires,
        };
        {
            let mut state = authority.access.lock().unwrap();
            let transaction = state.database.transaction().unwrap();
            transaction
                .execute(
                    "INSERT INTO invitations VALUES (?1,?2,?3)",
                    (
                        hash(&published.token),
                        published.expires,
                        serde_json::to_string(&Role::Worker).unwrap(),
                    ),
                )
                .unwrap();
            create_record(&files, "published-only.json", &published).unwrap();
            transaction.rollback().unwrap();
        }
        assert!(
            !authority
                .invitation_active(&files, "published-only.json")
                .unwrap()
        );
        let other = Authority::create(&path.parent().unwrap().join("other")).unwrap();
        assert_eq!(
            other.invitation_active(&files, "status.json"),
            Err(Error::Denied)
        );
        let mut wrong_key: Invitation = read_record(&files, "status.json").unwrap();
        wrong_key.coordinator.certificate = other.identity.member.certificate.clone();
        files
            .write_atomic_durable("status.json", &serde_json::to_vec(&wrong_key).unwrap())
            .unwrap();
        assert_eq!(
            authority.invitation_active(&files, "status.json"),
            Err(Error::Denied)
        );
        files
            .write_atomic_durable("status.json", &original)
            .unwrap();
        assert!(authority.invitation_active(&files, "missing.json").is_err());

        let consumed = invitation(&mut authority, &files, "consumed.json");
        let identity = Identity::generate(consumed.cluster()).unwrap();
        authority.issue(&request(&consumed, &identity)).unwrap();
        assert!(
            !authority
                .invitation_active(&files, "consumed.json")
                .unwrap()
        );
        drop(authority);
        let mut authority = Authority::open(&path).unwrap();
        assert!(authority.invitation_active(&files, "status.json").unwrap());
        let mut expired: Invitation = read_record(&files, "status.json").unwrap();
        expired.expires = 0;
        files
            .write_atomic_durable("status.json", &serde_json::to_vec(&expired).unwrap())
            .unwrap();
        authority
            .access
            .lock()
            .unwrap()
            .database
            .execute(
                "UPDATE invitations SET expires=0 WHERE hash=?1",
                [hash(&expired.token)],
            )
            .unwrap();
        assert!(!authority.invitation_active(&files, "status.json").unwrap());
        invitation(&mut authority, &files, "draining.json");
        assert!(
            authority
                .invitation_active(&files, "draining.json")
                .unwrap()
        );
        authority.drain().unwrap();
        assert!(
            !authority
                .invitation_active(&files, "draining.json")
                .unwrap()
        );
    }

    #[test]
    fn expired_invitation_leaf_and_mismatched_ca_key_are_refused() {
        let (_temp, path, mut authority, files) = setup();
        let mut invite = invitation(&mut authority, &files, "expired-client");
        let mut identity = Identity::generate(invite.cluster()).unwrap();
        invite.expires = now().unwrap() - 1;
        assert_eq!(invite.enroll(&mut identity), Err(Error::Expired));
        let mut device = enrolled(&mut authority, &files, Role::Worker, "expired-leaf");
        let mut params = certificate_params(&device.member).unwrap();
        params.not_before = time::OffsetDateTime::now_utc() - time::Duration::seconds(60);
        params.not_after = time::OffsetDateTime::now_utc() - time::Duration::seconds(1);
        device.member.certificate = params
            .signed_by(&device.key, &authority.issuer)
            .unwrap()
            .der()
            .to_vec();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let trust = authority.trust().unwrap();
        thread::scope(|scope| {
            let server = scope.spawn(|| {
                PeerStream::accept(
                    listener.accept().unwrap().0,
                    authority.identity(),
                    &trust,
                    Channel::Control,
                )
            });
            if let Ok(mut client) = transport::client(
                TcpStream::connect(listener.local_addr().unwrap()).unwrap(),
                &trust,
                &authority.identity.member,
                Some(&device),
                b"starlings-control/1",
            ) {
                let _ = send(&mut client, &Version);
                assert!(receive::<Version>(&mut client).is_err());
            }
            assert!(server.join().unwrap().is_err());
        });
        let mut saved: SavedAuthority =
            read_record(&authority.directory, "authority.json").unwrap();
        saved.key = KeyPair::generate().unwrap().serialize_der();
        authority
            .directory
            .write_atomic_durable("authority.json", &serde_json::to_vec(&saved).unwrap())
            .unwrap();
        drop(authority);
        assert!(matches!(Authority::open(&path), Err(Error::Denied)));
    }
    #[test]
    fn data_channel_access_is_live_and_scope_bound() {
        let (_temp, _, mut authority, files) = setup();
        let device = enrolled(&mut authority, &files, Role::Worker, "data-access");
        let permission = crate::Scope {
            model: starlings_protocol::byte_digest(b"data-model"),
            action: crate::Action::Infer,
        };
        authority
            .set_permissions(device.node(), std::slice::from_ref(&permission))
            .unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let trust = authority.trust().unwrap();
        let (server, _client) = thread::scope(|scope| {
            let server = scope.spawn(|| {
                PeerStream::accept(
                    listener.accept().unwrap().0,
                    authority.identity(),
                    &trust,
                    Channel::Data,
                )
                .unwrap()
            });
            let client = PeerStream::connect(
                TcpStream::connect(listener.local_addr().unwrap()).unwrap(),
                &device,
                &trust,
                authority.identity.node(),
                Channel::Data,
            )
            .unwrap();
            (server.join().unwrap(), client)
        });
        assert_eq!(server.with_access(&permission, || 7), Ok(7));
        authority.revoke(device.node()).unwrap();
        assert_eq!(server.with_access(&permission, || 7), Err(Error::Denied));
    }

    #[test]
    fn membership_capacity_and_drain_refuse_issuance_without_consuming_invites() {
        let (_temp, _, mut authority, files) = setup();
        for n in 0..63 {
            enrolled(&mut authority, &files, Role::Worker, &format!("member-{n}"));
        }
        let invite = invitation(&mut authority, &files, "over-capacity");
        let identity = Identity::generate(invite.cluster()).unwrap();
        let request = request(&invite, &identity);
        assert_eq!(authority.issue(&request).unwrap_err(), Error::Capacity);
        authority.drain().unwrap();
        assert_eq!(authority.issue(&request).unwrap_err(), Error::Denied);
        assert_eq!(
            authority
                .access
                .lock()
                .unwrap()
                .database
                .query_row("SELECT COUNT(*) FROM members", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            64
        );
    }

    #[test]
    fn enrollment_and_both_channels_survive_restart() {
        let (_temp, path, mut authority, files) = setup();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        authority
            .invite(
                &files,
                "invite.json",
                listener.local_addr().unwrap(),
                Role::Worker,
            )
            .unwrap();
        let invite = Invitation::load(&files, "invite.json").unwrap();
        let mut identity = Identity::generate(invite.cluster()).unwrap();
        identity.save(&files, "pending.json").unwrap();
        identity = Identity::load(&files, "pending.json").unwrap();
        let fingerprint = identity.request_fingerprint();
        let server = thread::spawn(move || {
            let pending = authority
                .receive_enrollment(listener.accept().unwrap().0)
                .unwrap();
            assert_eq!(pending.fingerprint(), fingerprint);
            authority.approve(pending, &fingerprint).unwrap();
        });
        invite.enroll(&mut identity).unwrap();
        server.join().unwrap();
        assert_eq!(identity.role(), Role::Worker);
        identity.save(&files, "enrolled.json").unwrap();
        identity = Identity::load(&files, "enrolled.json").unwrap();
        let mut authority = Authority::open(&path).unwrap();
        assert_eq!(
            authority.issue(&request(&invite, &identity)).unwrap_err(),
            Error::Replay
        );
        let trust = authority.trust().unwrap();
        assert_eq!(trust.members.len(), 2);
        for channel in [Channel::Control, Channel::Data] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            let trust = trust.clone();
            thread::scope(|scope| {
                let server = scope.spawn(|| {
                    let mut stream = PeerStream::accept(
                        listener.accept().unwrap().0,
                        &authority.identity,
                        &trust,
                        channel,
                    )
                    .unwrap();
                    assert_eq!(stream.peer(), identity.node());
                    assert_eq!(stream.role(), Role::Worker);
                    let mut bytes = [0; 4];
                    stream.read_exact(&mut bytes).unwrap();
                    assert_eq!(&bytes, b"ping");
                    stream.write_all(b"pong").unwrap();
                    stream.flush().unwrap();
                });
                let mut stream = PeerStream::connect(
                    TcpStream::connect(address).unwrap(),
                    &identity,
                    &trust,
                    authority.identity().node(),
                    channel,
                )
                .unwrap();
                stream.write_all(b"ping").unwrap();
                stream.flush().unwrap();
                let mut bytes = [0; 4];
                stream.read_exact(&mut bytes).unwrap();
                assert_eq!(&bytes, b"pong");
                server.join().unwrap();
            });
        }
    }

    #[test]
    fn invitations_are_atomic_expiring_and_csr_extensions_cannot_escalate() {
        let (_temp, path, mut authority, files) = setup();
        let invite = invitation(&mut authority, &files, "invite.json");
        let identity = Identity::generate(invite.cluster()).unwrap();
        let mut req = request(&invite, &identity);
        req.token[0] ^= 1;
        assert_eq!(authority.issue(&req).unwrap_err(), Error::Replay);
        req.token = invite.token;
        req.csr.push(0);
        assert_eq!(authority.issue(&req).unwrap_err(), Error::Invalid);
        req.csr = identity.csr().unwrap();
        let last = req.csr.len() - 1;
        req.csr[last] ^= 1;
        assert!(authority.issue(&req).is_err());
        req.csr = identity.csr().unwrap();
        authority
            .access
            .lock()
            .unwrap()
            .database
            .execute("UPDATE invitations SET expires=0", [])
            .unwrap();
        assert_eq!(authority.issue(&req).unwrap_err(), Error::Expired);
        authority
            .access
            .lock()
            .unwrap()
            .database
            .execute("UPDATE invitations SET expires=?1", [now().unwrap() + 300])
            .unwrap();
        let mut elevated = CertificateParams::new(vec!["admin.attacker.invalid".into()]).unwrap();
        elevated.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        elevated.key_usages = vec![KeyUsagePurpose::KeyCertSign];
        req.csr = elevated
            .serialize_request(&identity.key)
            .unwrap()
            .der()
            .to_vec();
        let barrier = Arc::new(Barrier::new(2));
        let mut second = Authority::open(&path).unwrap();
        let (first, second) = thread::scope(|scope| {
            let other = scope.spawn(|| {
                barrier.wait();
                second.issue(&req)
            });
            barrier.wait();
            (authority.issue(&req), other.join().unwrap())
        });
        assert_eq!(usize::from(first.is_ok()) + usize::from(second.is_ok()), 1);
        let issued = first.or(second).unwrap();
        let (_, certificate) = x509_parser::parse_x509_certificate(&issued.certificate).unwrap();
        assert!(!certificate.is_ca());
        assert_eq!(
            certificate
                .subject_alternative_name()
                .unwrap()
                .unwrap()
                .value
                .general_names
                .len(),
            1
        );
        transport::verify_issued(&issued, &authority.ca, &identity.key).unwrap();
        drop(authority);
        let mut reopened = Authority::open(&path).unwrap();
        assert_eq!(reopened.issue(&req).unwrap_err(), Error::Replay);
        reopened
            .access
            .lock()
            .unwrap()
            .database
            .execute("UPDATE clock SET observed=?1", [now().unwrap() + 3600])
            .unwrap();
        assert_eq!(reopened.check_clock().unwrap_err(), Error::Expired);
    }

    #[test]
    fn bootstrap_wrong_ca_wrong_coordinator_and_unapproved_fingerprint_fail() {
        let (_temp, _path, mut authority, files) = setup();
        let (_other_temp, _other_path, other, _) = setup();
        for failure in 0..5 {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            authority
                .invite(
                    &files,
                    &format!("invite{failure}"),
                    listener.local_addr().unwrap(),
                    Role::Worker,
                )
                .unwrap();
            let mut invite = Invitation::load(&files, &format!("invite{failure}")).unwrap();
            if failure == 0 {
                invite.ca = other.ca.clone();
            }
            if failure == 1 {
                invite.coordinator.node = other.identity.node();
            }
            let mut identity = Identity::generate(invite.cluster()).unwrap();
            thread::scope(|scope| {
                let server = scope.spawn(|| {
                    let pending = authority.receive_enrollment(listener.accept().unwrap().0);
                    if failure >= 2 {
                        let mut pending = pending.unwrap();
                        let fingerprint = if failure == 2 {
                            "unapproved".to_owned()
                        } else {
                            pending.fingerprint().to_owned()
                        };
                        if failure == 3 {
                            pending.ca_hash = "wrong-ca".into();
                        }
                        if failure == 4 {
                            pending.cluster = uuid::Uuid::new_v4().to_string().parse().unwrap();
                        }
                        assert_eq!(authority.approve(pending, &fingerprint), Err(Error::Denied));
                    } else {
                        assert!(pending.is_err());
                    }
                });
                assert!(invite.enroll(&mut identity).is_err());
                server.join().unwrap();
            });
            assert_eq!(authority.trust().unwrap().members.len(), 1);
        }
    }

    #[test]
    fn revoked_certificate_and_cross_channel_connections_are_denied() {
        let (_temp, _path, mut authority, files) = setup();
        let invite = invitation(&mut authority, &files, "invite.json");
        let mut identity = Identity::generate(invite.cluster()).unwrap();
        identity.member = authority.issue(&request(&invite, &identity)).unwrap();
        let known = authority.trust().unwrap();
        for unknown in [true, false] {
            let server_trust = known.clone();
            if unknown {
                authority.revoke(identity.node()).unwrap();
            } else {
                authority
                    .access
                    .lock()
                    .unwrap()
                    .database
                    .execute("DELETE FROM revoked", [])
                    .unwrap();
            }
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            thread::scope(|scope| {
                let server = scope.spawn(|| {
                    PeerStream::accept(
                        listener.accept().unwrap().0,
                        &authority.identity,
                        &server_trust,
                        Channel::Control,
                    )
                    .is_err()
                });
                let result = PeerStream::connect(
                    TcpStream::connect(address).unwrap(),
                    &identity,
                    &known,
                    authority.identity().node(),
                    if unknown {
                        Channel::Control
                    } else {
                        Channel::Data
                    },
                );
                assert!(result.is_err());
                assert!(server.join().unwrap());
            });
        }
    }

    #[test]
    fn tls12_and_oversized_frames_are_rejected() {
        use rustls::{ClientConfig, ClientConnection, RootCertStore, pki_types::ServerName};
        let (_temp, _path, authority, _) = setup();
        let trust = authority.trust().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let mut roots = RootCertStore::empty();
        roots.add(authority.ca.clone().into()).unwrap();
        // TLS 1.2 is enabled only as a dev dependency for this negative handshake.
        let config =
            ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
                .with_protocol_versions(&[&rustls::version::TLS12])
                .unwrap()
                .with_root_certificates(roots)
                .with_no_client_auth();
        let name = ServerName::try_from(authority.identity.member.name()).unwrap();
        thread::scope(|scope| {
            let server = scope.spawn(|| {
                PeerStream::accept(
                    listener.accept().unwrap().0,
                    &authority.identity,
                    &trust,
                    Channel::Control,
                )
                .is_err()
            });
            let mut socket = TcpStream::connect(address).unwrap();
            socket.set_read_timeout(Some(transport::TIMEOUT)).unwrap();
            socket.set_write_timeout(Some(transport::TIMEOUT)).unwrap();
            let error = ClientConnection::new(Arc::new(config), name)
                .unwrap()
                .complete_io(&mut socket)
                .unwrap_err();
            assert!(error.to_string().contains("ProtocolVersion"));
            assert!(server.join().unwrap());
        });
        let mut prefix = std::io::Cursor::new(((MAX_RECORD + 1) as u32).to_be_bytes());
        assert_eq!(
            receive::<Request>(&mut prefix).err().unwrap(),
            Error::Invalid
        );
        assert!(decode::<Request>(br#"{"version":2}"#).is_err());
    }

    #[test]
    fn private_files_reject_overwrite_traversal_corruption_and_disclosure() {
        let (temp, path, mut authority, files) = setup();
        let invite = invitation(&mut authority, &files, "invite.json");
        let identity = Identity::generate(invite.cluster()).unwrap();
        identity.save(&files, "key.json").unwrap();
        assert!(identity.save(&files, "key.json").is_err());
        assert!(identity.save(&files, "../escape").is_err());
        assert!(Authority::create(&path).is_err());
        let db = std::fs::read(path.join("enrollment.sqlite")).unwrap();
        assert!(!db.windows(32).any(|w| w == invite.token));
        assert!(
            !String::from_utf8_lossy(&db).contains(&serde_json::to_string(&invite.token).unwrap())
        );
        let bad = b"{\"token\":\"do-not-log-this\",\"token\":null}";
        let error = decode::<Invitation>(bad).err().unwrap();
        assert!(!format!("{error:?} {error}").contains("do-not-log-this"));
        let key_path = temp.path().join("device/key.json");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&key_path).unwrap().permissions().mode() & 0o777,
                0o600
            );
            std::fs::set_permissions(&key_path, std::fs::Permissions::from_mode(0o644)).unwrap();
        }
        #[cfg(windows)]
        {
            let status = std::process::Command::new("icacls")
                .arg(&key_path)
                .args(["/grant", "*S-1-1-0:(R)"])
                .output()
                .unwrap()
                .status;
            assert!(status.success());
        }
        assert!(Identity::load(&files, "key.json").is_err());
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&key_path, temp.path().join("device/link.json")).unwrap();
            assert!(Identity::load(&files, "link.json").is_err());
            identity.save(&files, "linked.json").unwrap();
            std::fs::hard_link(
                temp.path().join("device/linked.json"),
                temp.path().join("device/alias.json"),
            )
            .unwrap();
            assert!(Identity::load(&files, "linked.json").is_err());
        }
    }
    pub(crate) fn enrolled(
        authority: &mut Authority,
        files: &PrivateDir,
        role: Role,
        name: &str,
    ) -> Identity {
        authority
            .invite(files, name, "127.0.0.1:9741".parse().unwrap(), role)
            .unwrap();
        let invite = Invitation::load(files, name).unwrap();
        let mut identity = Identity::generate(invite.cluster()).unwrap();
        identity.member = authority.issue(&request(&invite, &identity)).unwrap();
        identity
    }
    pub(crate) fn pair(
        local: &Identity,
        remote: &Identity,
        trust: &Trust,
    ) -> (PeerStream, PeerStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        thread::scope(|scope| {
            let server = scope.spawn(|| {
                PeerStream::accept(listener.accept().unwrap().0, local, trust, Channel::Control)
                    .unwrap()
            });
            let client = PeerStream::connect(
                TcpStream::connect(address).unwrap(),
                remote,
                trust,
                local.node(),
                Channel::Control,
            )
            .unwrap();
            (server.join().unwrap(), client)
        })
    }
    fn scope(model: &starlings_protocol::Digest, action: crate::Action) -> crate::Scope {
        crate::Scope {
            model: model.clone(),
            action,
        }
    }
    #[test]
    fn scopes_tokens_and_runtime_ownership_fail_closed() {
        use crate::Action::*;
        let (_temp, path, mut authority, files) = setup();
        let worker = enrolled(&mut authority, &files, Role::Worker, "worker");
        let controller = enrolled(&mut authority, &files, Role::Controller, "controller");
        let model = starlings_protocol::byte_digest(b"model");
        let artifact = scope(&model, Artifact);
        assert!(authority.permissions(controller.node()).unwrap().is_empty());
        assert!(
            authority
                .permissions("00000000-0000-4000-8000-000000000999".parse().unwrap())
                .is_err()
        );
        assert!(
            authority
                .issue_client_token(controller.node(), std::slice::from_ref(&artifact), 60)
                .is_err()
        );
        assert!(
            authority
                .set_permissions(worker.node(), &[scope(&model, Control)])
                .is_err()
        );
        authority
            .set_permissions(controller.node(), &[artifact.clone(), scope(&model, Infer)])
            .unwrap();
        let changes = authority.access.inspect().unwrap().database.total_changes();
        assert_eq!(
            authority.permissions(controller.node()).unwrap(),
            vec![artifact.clone(), scope(&model, Infer)]
        );
        assert_eq!(
            authority.access.inspect().unwrap().database.total_changes(),
            changes
        );
        let token = authority
            .issue_client_token(controller.node(), std::slice::from_ref(&artifact), 60)
            .unwrap();
        token.save(&files, "client-token").unwrap();
        let token = crate::ClientToken::load(&files, "client-token").unwrap();
        assert_eq!(
            authority.with_client_access(&token, &artifact, || 42),
            Ok(42)
        );
        assert!(
            authority
                .with_client_access(&token, &scope(&model, Infer), || panic!("scope escalation"))
                .is_err()
        );
        assert!(
            authority
                .with_client_access(
                    &token,
                    &scope(&starlings_protocol::byte_digest(b"other"), Artifact),
                    || panic!("model escalation")
                )
                .is_err()
        );
        assert!(
            authority
                .issue_client_token(worker.node(), std::slice::from_ref(&artifact), 60)
                .is_err()
        );
        assert!(
            authority
                .issue_client_token(controller.node(), std::slice::from_ref(&artifact), 3601)
                .is_err()
        );
        let stored: String = authority
            .access
            .lock()
            .unwrap()
            .database
            .query_row("SELECT hash FROM tokens", [], |r| r.get(0))
            .unwrap();
        assert_eq!(stored.len(), 64);
        assert!(
            authority
                .access
                .lock()
                .unwrap()
                .database
                .prepare("SELECT secret FROM tokens")
                .is_err()
        );
        // An independent handle cannot dispatch or mutate policy while this runtime is live.
        let second = Authority::open(&path).unwrap();
        assert!(second.set_permissions(controller.node(), &[]).is_err());
        drop(second);
        authority.revoke_client_token(&token).unwrap();
        assert!(
            authority
                .with_client_access(&token, &artifact, || panic!("revoked token"))
                .is_err()
        );
        let token = authority
            .issue_client_token(controller.node(), std::slice::from_ref(&artifact), 60)
            .unwrap();
        authority
            .access
            .lock()
            .unwrap()
            .database
            .execute("UPDATE tokens SET expires=0", [])
            .unwrap();
        assert_eq!(
            authority.with_client_access(&token, &artifact, || ()),
            Err(Error::Expired)
        );
        let token = authority
            .issue_client_token(controller.node(), std::slice::from_ref(&artifact), 60)
            .unwrap();
        authority.set_permissions(controller.node(), &[]).unwrap();
        assert!(
            authority
                .with_client_access(&token, &artifact, || panic!("removed grant"))
                .is_err()
        );
        authority.revoke(controller.node()).unwrap();
        drop(authority);
        let authority = Authority::open(&path).unwrap();
        assert!(
            authority
                .set_permissions(controller.node(), std::slice::from_ref(&artifact))
                .is_err()
        );
        assert!(
            authority
                .with_client_access(&token, &artifact, || panic!("restart resurrected token"))
                .is_err()
        );
    }
    #[test]
    fn renewal_preserves_verification_and_expires_old_live_credentials() {
        let (_temp, path, mut authority, files) = setup();
        let old = enrolled(&mut authority, &files, Role::Worker, "worker");
        let trust = authority.trust().unwrap();
        let (mut server, mut client) = pair(authority.identity(), &old, &trust);
        let next = thread::scope(|scope| {
            let task = scope.spawn(|| old.renew(&mut client).unwrap());
            authority.renew_peer(&mut server).unwrap();
            task.join().unwrap()
        });
        assert_eq!(old.node(), next.node());
        assert_ne!(old.request_fingerprint(), next.request_fingerprint());
        crate::transport::verify_issued(&next.member, &authority.ca, &next.key).unwrap();
        // Existing trust handles resolve new certificates from durable state.
        let _new_pair = pair(authority.identity(), &next, &trust);
        client.write_all(b"x").unwrap();
        client.flush().unwrap();
        let mut byte = [0];
        server.read_exact(&mut byte).unwrap();
        // A second rotation cannot extend the old credential's overlap indefinitely.
        let (mut s2, mut c2) = pair(authority.identity(), &next, &trust);
        thread::scope(|scope| {
            let task = scope.spawn(|| next.renew(&mut c2));
            assert!(authority.renew_peer(&mut s2).is_err());
            drop(s2);
            assert!(task.join().unwrap().is_err());
        });
        authority
            .access
            .lock()
            .unwrap()
            .database
            .execute("UPDATE previous SET expires=0", [])
            .unwrap();
        authority.maintain(0).unwrap();
        assert!(client.write_all(b"denied").is_err());
        assert!(server.read(&mut byte).is_err());
        assert!(
            authority
                .access
                .check(client.authority().unwrap().1)
                .is_err()
        );
        let (mut server, mut client) = pair(authority.identity(), &next, &trust);
        client.write_all(b"y").unwrap();
        client.flush().unwrap();
        server.read_exact(&mut byte).unwrap();
        assert_eq!(byte, *b"y");
        // Coordinator rotation publishes its new private identity atomically with membership.
        let previous = authority.identity().member.certificate.clone();
        authority.renew_coordinator().unwrap();
        let current = authority.identity().member.certificate.clone();
        assert_ne!(previous, current);
        drop((server, client, _new_pair, trust));
        // The first pair's client still owns the old runtime until dropped.
        drop(authority);
        let authority = Authority::open(&path).unwrap();
        assert_eq!(authority.identity().member.certificate, current);
    }
    #[test]
    fn revocation_interrupts_blocked_tls_reads_and_denies_artifacts() {
        use crate::Action::Artifact;
        let (_temp, _path, mut authority, files) = setup();
        let device = enrolled(&mut authority, &files, Role::Worker, "worker");
        let model = starlings_protocol::byte_digest(b"model");
        let artifact = scope(&model, Artifact);
        authority
            .set_permissions(device.node(), std::slice::from_ref(&artifact))
            .unwrap();
        let trust = authority.trust().unwrap();
        let (mut server, client) = pair(authority.identity(), &device, &trust);
        assert_eq!(server.with_access(&artifact, || 7), Ok(7));
        let ready = Arc::new(Barrier::new(2));
        thread::scope(|scope| {
            let ready2 = ready.clone();
            let task = scope.spawn(move || {
                ready2.wait();
                server.read(&mut [0; 1])
            });
            ready.wait();
            let start = std::time::Instant::now();
            authority.revoke(device.node()).unwrap();
            assert!(task.join().unwrap().is_err());
            assert!(start.elapsed() < std::time::Duration::from_secs(2));
        });
        assert!(
            client
                .with_access(&artifact, || panic!("revoked local member"))
                .is_err()
        );
        assert!(
            authority
                .access
                .check(client.authority().unwrap().1)
                .is_err()
        );
    }
    #[test]
    fn revoked_plan_drains_safely_and_disconnected_leases_expire() {
        use crate::Action::*;
        use starlings_core::*;
        use starlings_protocol::*;
        for reason in [
            "node",
            "token",
            "grant",
            "clock",
            "expiry",
            "complete",
            "disconnect",
            "drain",
            "registry-epoch",
            "registry-offline",
            "registry-revoked",
            "registry-draining",
            "registry-output-race",
        ] {
            let (_temp, _path, mut authority, files) = setup();
            let device = enrolled(&mut authority, &files, Role::Worker, "worker");
            let other = enrolled(&mut authority, &files, Role::Worker, "other");
            let controller = enrolled(&mut authority, &files, Role::Controller, "controller");
            let mode = FixtureMode::Simulation;
            let mut plan: PlacementPlan = starlings_protocol::decode(
                include_bytes!("../../../examples/placement-plan.json"),
                mode,
            )
            .unwrap();
            plan.cluster_id = device.cluster();
            plan.stages[0].node_id = device.node();
            plan.stages[1].node_id = other.node();
            let mut lease: Lease =
                starlings_protocol::decode(include_bytes!("../../../examples/lease.json"), mode)
                    .unwrap();
            lease.cluster_id = device.cluster();
            lease.owner_node_id = device.node();
            let session: Session =
                starlings_protocol::decode(include_bytes!("../../../examples/session.json"), mode)
                    .unwrap();
            let mut header: TensorHeader = starlings_protocol::decode(
                include_bytes!("../../../examples/tensor-header.json"),
                mode,
            )
            .unwrap();
            header.destination_stage = BoundedU32::new(0).unwrap();
            let infer = scope(&plan.manifest_digest, Infer);
            authority
                .set_permissions(device.node(), std::slice::from_ref(&infer))
                .unwrap();
            authority
                .set_permissions(other.node(), std::slice::from_ref(&infer))
                .unwrap();
            authority
                .set_permissions(controller.node(), std::slice::from_ref(&infer))
                .unwrap();
            authority
                .set_permissions(
                    authority.identity().node(),
                    &[scope(&plan.manifest_digest, Control)],
                )
                .unwrap();
            let token = authority
                .issue_client_token(controller.node(), std::slice::from_ref(&infer), 60)
                .unwrap();
            let worker = authority
                .secure_worker(
                    Worker::new(device.node(), device.cluster(), 4_194_304, 1, mode).unwrap(),
                )
                .unwrap();
            let fence = crate::RegistryFence::test_online(device.node(), device.cluster());
            let worker = if reason.starts_with("registry-") {
                worker.bind_registry(fence.clone()).unwrap()
            } else {
                worker
            };
            let trust = authority.trust().unwrap();
            let (server, client) = pair(&device, authority.identity(), &trust);
            let req = |op| {
                ControlRequest::new(
                    uuid::Uuid::new_v4().to_string().parse().unwrap(),
                    1,
                    {
                        static SEQ: std::sync::atomic::AtomicU64 =
                            std::sync::atomic::AtomicU64::new(1);
                        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                    },
                    op,
                )
                .unwrap()
            };
            let prepare = req(Operation::Prepare {
                plan: Box::new(plan),
                lease: Box::new(lease.clone()),
            });
            worker.apply(&server, &token, &prepare, 0).unwrap();
            worker
                .apply(
                    &server,
                    &token,
                    &req(Operation::Commit {
                        lease_id: lease.lease_id,
                    }),
                    0,
                )
                .unwrap();
            worker
                .apply(
                    &server,
                    &token,
                    &req(Operation::Begin {
                        session: Box::new(session.clone()),
                    }),
                    0,
                )
                .unwrap();
            let step = req(Operation::StartStep {
                header: Box::new(header.clone()),
                input_digest: byte_digest(b"input"),
            });
            worker.apply(&server, &token, &step, 0).unwrap();
            if reason == "registry-output-race" {
                worker
                    .drained(session.generation_id, header.invocation_id)
                    .unwrap();
                let updating = fence.clone();
                let handle = worker
                    .with_output(&token, session.generation_id, 0, || {
                        let (started, ready) = std::sync::mpsc::sync_channel(1);
                        let (done, finished) = std::sync::mpsc::sync_channel(1);
                        let handle = thread::spawn(move || {
                            started.send(()).unwrap();
                            updating.test_transition("registry-epoch");
                            done.send(()).unwrap();
                        });
                        ready
                            .recv_timeout(std::time::Duration::from_secs(1))
                            .unwrap();
                        assert!(
                            finished
                                .recv_timeout(std::time::Duration::from_millis(50))
                                .is_err(),
                            "epoch changed inside output dispatch"
                        );
                        (handle, finished)
                    })
                    .unwrap();
                handle.0.join().unwrap();
                handle
                    .1
                    .recv_timeout(std::time::Duration::from_secs(1))
                    .unwrap();
                fence.maintain(0).unwrap();
                assert!(
                    worker
                        .with_output(&token, session.generation_id, 0, || panic!("old output"))
                        .is_err()
                );
                continue;
            }
            if reason == "complete" {
                worker
                    .drained(session.generation_id, header.invocation_id)
                    .unwrap();
                assert_eq!(
                    worker.with_output(&token, session.generation_id, 0, || 17),
                    Ok(17)
                );
                worker
                    .apply(
                        &server,
                        &token,
                        &req(Operation::Finish {
                            generation_id: session.generation_id,
                        }),
                        0,
                    )
                    .unwrap();
                assert_eq!(
                    worker.with_output(&token, session.generation_id, 0, || 19),
                    Ok(19)
                );
                assert!(
                    worker
                        .with_output(
                            &token,
                            session.generation_id,
                            lease.lifetime_ms.get().into(),
                            || panic!("expired output")
                        )
                        .is_err()
                );
                continue;
            }
            match reason {
                reason if reason.starts_with("registry-") => {
                    fence.test_transition(reason);
                    fence.maintain(0).unwrap();
                    assert_eq!(
                        worker.snapshot().unwrap().epoch.0,
                        1,
                        "must wait for old backend drain"
                    );
                }
                "drain" => {
                    assert_eq!(authority.drain(), Ok(4_194_304));
                    assert_eq!(authority.drain(), Ok(4_194_304));
                    assert!(
                        authority
                            .with_client_access(&token, &infer, || panic!("draining dispatch"))
                            .is_err()
                    );
                    assert!(
                        server
                            .with_access(&infer, || panic!("draining peer"))
                            .is_err()
                    );
                    assert!(
                        authority
                            .secure_worker(
                                Worker::new(device.node(), device.cluster(), 4_194_304, 1, mode)
                                    .unwrap()
                            )
                            .is_err()
                    );
                }
                "expiry" => {
                    authority
                        .access
                        .lock()
                        .unwrap()
                        .database
                        .execute("UPDATE tokens SET expires=0", [])
                        .unwrap();
                    authority.maintain(0).unwrap();
                }
                "node" => authority.revoke(other.node()).unwrap(),
                "token" => authority.revoke_client_token(&token).unwrap(),
                "grant" => authority.set_permissions(other.node(), &[]).unwrap(),
                "clock" => {
                    authority
                        .access
                        .lock()
                        .unwrap()
                        .database
                        .execute("UPDATE clock SET observed=?1", [now().unwrap() + 3600])
                        .unwrap();
                    assert_eq!(authority.maintain(0), Err(crate::Error::Expired));
                }
                _ => {
                    drop(client);
                    authority.maintain(lease.lifetime_ms.get().into()).unwrap();
                }
            }
            let snapshot = worker.snapshot().unwrap();
            assert_eq!(
                snapshot.generations[&session.generation_id],
                SessionState::Cancelling
            );
            assert_eq!(snapshot.reserved_bytes.0, 4_194_304);
            if reason != "disconnect" {
                assert!(worker.apply(&server, &token, &prepare, 0).is_err());
                assert!(worker.apply(&server, &token, &step, 0).is_err());
            }
            assert!(
                worker
                    .apply(
                        &server,
                        &token,
                        &req(Operation::CompleteStep {
                            generation_id: session.generation_id,
                            invocation_id: header.invocation_id
                        }),
                        0
                    )
                    .is_err()
            );
            assert!(
                worker
                    .with_output(
                        &token,
                        session.generation_id,
                        if reason == "disconnect" {
                            lease.lifetime_ms.get().into()
                        } else {
                            0
                        },
                        || panic!("cancelled output escaped")
                    )
                    .is_err()
            );
            worker
                .drained(session.generation_id, header.invocation_id)
                .unwrap();
            let snapshot = worker.snapshot().unwrap();
            assert_eq!(snapshot.reserved_bytes.0, 0);
            if reason == "registry-epoch" {
                fence.maintain(0).unwrap();
                assert_eq!(worker.snapshot().unwrap().epoch.0, 2);
                assert!(worker.apply(&server, &token, &prepare, 0).is_err());
                assert!(worker.apply(&server, &token, &step, 0).is_err());
            }
            if reason == "drain" {
                assert_eq!(authority.drain(), Ok(0));
            }
            assert!(matches!(
                snapshot.generations[&session.generation_id],
                SessionState::Cancelled | SessionState::Failed
            ));
        }
    }
    #[test]
    fn ca_valid_unregistered_leaf_never_becomes_a_peer() {
        let (_temp, _path, authority, _) = setup();
        let mut rogue = Identity::generate(authority.identity().cluster()).unwrap();
        rogue.member.certificate = certificate_params(&rogue.member)
            .unwrap()
            .signed_by(&rogue.key, &authority.issuer)
            .unwrap()
            .der()
            .to_vec();
        let trust = authority.trust().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let identity = authority.identity();
        thread::scope(|scope| {
            let server = scope.spawn(|| {
                PeerStream::accept(
                    listener.accept().unwrap().0,
                    identity,
                    &trust,
                    Channel::Control,
                )
                .is_err()
            });
            let _ = crate::transport::client(
                TcpStream::connect(address).unwrap(),
                &trust,
                &identity.member,
                Some(&rogue),
                b"starlings-control/1",
            );
            assert!(server.join().unwrap());
        });
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        thread::scope(|scope| {
            let reader = authority.enrollment_reader().unwrap();
            let server = scope.spawn(|| {
                reader.registry(
                    listener.accept().unwrap().0,
                    std::time::Instant::now() + std::time::Duration::from_secs(3),
                )
            });
            let _ = crate::transport::client(
                TcpStream::connect(listener.local_addr().unwrap()).unwrap(),
                &trust,
                &identity.member,
                Some(&rogue),
                crate::registry::ALPN,
            );
            assert!(server.join().unwrap().is_err());
        });
    }
}
