//! Capacity proof only. TLS still authenticates the connection and authorizes every request.
use crate::{Error, Identity, Result, Role, authorization::Access, enrollment::Member};
use ring::{
    rand::{SecureRandom, SystemRandom},
    signature,
};
use starlings_protocol::{ClusterId, NodeId};
use std::{
    io::{Read, Write},
    net::TcpStream,
    time::{Duration, Instant},
};

const MARKER: u8 = b'R';
const PROOF_LEN: usize = 36 + 64;
const PROBE_TIME: Duration = Duration::from_secs(1);

fn message(cluster: ClusterId, coordinator: NodeId, node: NodeId, nonce: &[u8; 32]) -> Vec<u8> {
    let mut bytes = format!("starlings-reconnect/1:{cluster}:{coordinator}:{node}:").into_bytes();
    bytes.extend_from_slice(nonce);
    bytes
}

fn proof(identity: &Identity, coordinator: NodeId, nonce: &[u8; 32]) -> Result<[u8; PROOF_LEN]> {
    let random = SystemRandom::new();
    let key = signature::EcdsaKeyPair::from_pkcs8(
        &signature::ECDSA_P256_SHA256_FIXED_SIGNING,
        &identity.key.serialize_der(),
        &random,
    )
    .map_err(|_| Error::Crypto)?;
    let signature = key
        .sign(
            &random,
            &message(identity.cluster(), coordinator, identity.node(), nonce),
        )
        .map_err(|_| Error::Crypto)?;
    let mut proof = [0; PROOF_LEN];
    proof[..36].copy_from_slice(identity.node().to_string().as_bytes());
    proof[36..].copy_from_slice(signature.as_ref());
    Ok(proof)
}

fn verify(
    access: &Access,
    identity: &Identity,
    nonce: &[u8; 32],
    proof: &[u8; PROOF_LEN],
) -> Result<NodeId> {
    let node: NodeId = std::str::from_utf8(&proof[..36])
        .map_err(|_| Error::Denied)?
        .parse()
        .map_err(|_| Error::Denied)?;
    // Indexed lookup only; no network I/O or signature verification under the policy lock.
    let records = {
        let state = access.lock()?;
        let mut query = state.database.prepare("SELECT record FROM members WHERE node=?1 UNION ALL SELECT record FROM previous WHERE node=?1 AND expires>?2")?;
        query
            .query_map(rusqlite::params![node.to_string(), crate::now()?], |r| {
                r.get::<_, Vec<u8>>(0)
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?
    };
    for record in records {
        let member: Member = crate::decode(&record)?;
        if member.cluster != identity.cluster()
            || !matches!(member.role, Role::Worker | Role::Controller)
        {
            continue;
        }
        let (_, cert) =
            x509_parser::parse_x509_certificate(&member.certificate).map_err(|_| Error::Storage)?;
        if cert.validity().is_valid()
            && signature::UnparsedPublicKey::new(
                &signature::ECDSA_P256_SHA256_FIXED,
                cert.public_key().subject_public_key.data.as_ref(),
            )
            .verify(
                &message(member.cluster, identity.node(), node, nonce),
                &proof[36..],
            )
            .is_ok()
        {
            return Ok(node);
        }
    }
    Err(Error::Denied)
}

pub(crate) fn client(
    socket: TcpStream,
    identity: &Identity,
    coordinator: NodeId,
    deadline: Instant,
) -> Result<TcpStream> {
    let mut socket = crate::transport::DeadlineSocket {
        socket,
        deadline,
        handshake_bytes: None,
        stage: None,
    };
    socket.write_all(&[MARKER])?;
    let mut nonce = [0; 32];
    socket.read_exact(&mut nonce)?;
    socket.write_all(&proof(identity, coordinator, &nonce)?)?;
    Ok(socket.socket)
}

pub(crate) struct Probe {
    pub socket: TcpStream,
    pub deadline: Instant,
    nonce: [u8; 32],
    marked: bool,
    sent: usize,
    proof: [u8; PROOF_LEN],
    received: usize,
    pub verified: Option<NodeId>,
}
impl Probe {
    pub fn new(socket: TcpStream) -> Result<Self> {
        socket.set_nonblocking(true)?;
        let mut nonce = [0; 32];
        SystemRandom::new()
            .fill(&mut nonce)
            .map_err(|_| Error::Crypto)?;
        Ok(Self {
            socket,
            deadline: Instant::now() + PROBE_TIME,
            nonce,
            marked: false,
            sent: 0,
            proof: [0; PROOF_LEN],
            received: 0,
            verified: None,
        })
    }
    /// None is incomplete; Some(None) is bootstrap TLS; Some(Some(node)) proved its key.
    pub fn progress(
        &mut self,
        access: &Access,
        identity: &Identity,
    ) -> Result<Option<Option<NodeId>>> {
        if Instant::now() >= self.deadline {
            return Err(Error::Expired);
        }
        if let Some(node) = self.verified {
            return Ok(Some(Some(node)));
        }
        let result = (|| -> std::io::Result<Option<Option<NodeId>>> {
            if !self.marked {
                let mut marker = [0];
                if self.socket.peek(&mut marker)? == 0 {
                    return Err(std::io::ErrorKind::UnexpectedEof.into());
                }
                if marker[0] == 22 {
                    return Ok(Some(None));
                }
                if marker[0] != MARKER {
                    return Err(std::io::ErrorKind::InvalidData.into());
                }
                self.socket.read_exact(&mut marker)?;
                self.marked = true;
            }
            if self.sent < self.nonce.len() {
                let count = self.socket.write(&self.nonce[self.sent..])?;
                if count == 0 {
                    return Err(std::io::ErrorKind::WriteZero.into());
                }
                self.sent += count;
                if self.sent < self.nonce.len() {
                    return Ok(None);
                }
            }
            if self.received < self.proof.len() {
                let count = self.socket.read(&mut self.proof[self.received..])?;
                if count == 0 {
                    return Err(std::io::ErrorKind::UnexpectedEof.into());
                }
                self.received += count;
            }
            Ok(None)
        })();
        match result {
            Ok(Some(route)) => return Ok(Some(route)),
            Ok(None) => (),
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
                ) =>
            {
                return Ok(None);
            }
            Err(e) => return Err(e.into()),
        }
        if self.received == self.proof.len() {
            match verify(access, identity, &self.nonce, &self.proof) {
                Ok(node) => {
                    self.verified = Some(node);
                    Ok(Some(Some(node)))
                }
                Err(Error::Busy) => Ok(None),
                Err(e) => Err(e),
            }
        } else {
            Ok(None)
        }
    }
}

/// Compatibility entry point for callers that do not use daemon-wide listener admission.
pub(crate) fn server(
    socket: TcpStream,
    access: &Access,
    identity: &Identity,
) -> Result<(TcpStream, Option<NodeId>)> {
    let mut probe = Probe::new(socket)?;
    loop {
        if let Some(node) = probe.progress(access, identity)? {
            probe.socket.set_nonblocking(false)?;
            return Ok((probe.socket, node));
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enrollment::tests::{enrolled, setup};

    #[test]
    fn proof_binds_challenge_node_cluster_and_coordinator() {
        let (_temp, _, mut authority, files) = setup();
        let worker = enrolled(&mut authority, &files, Role::Worker, "proof-worker");
        let second = enrolled(&mut authority, &files, Role::Worker, "proof-second");
        let nonce = [1; 32];
        let signed = proof(&worker, authority.identity.node(), &nonce).unwrap();
        assert_eq!(
            verify(&authority.access, &authority.identity, &nonce, &signed),
            Ok(worker.node())
        );
        assert_eq!(
            verify(&authority.access, &authority.identity, &[2; 32], &signed),
            Err(Error::Denied)
        );
        let mut forged = signed;
        forged[..36].copy_from_slice(second.node().to_string().as_bytes());
        assert_eq!(
            verify(&authority.access, &authority.identity, &nonce, &forged),
            Err(Error::Denied)
        );
        let wrong_target = proof(&worker, second.node(), &nonce).unwrap();
        assert_eq!(
            verify(
                &authority.access,
                &authority.identity,
                &nonce,
                &wrong_target
            ),
            Err(Error::Denied)
        );
        let mut unknown = Identity::generate(worker.cluster()).unwrap();
        unknown.member.node = worker.node();
        let forged = proof(&unknown, authority.identity.node(), &nonce).unwrap();
        assert_eq!(
            verify(&authority.access, &authority.identity, &nonce, &forged),
            Err(Error::Denied)
        );
        let mut wrong_cluster = Identity {
            member: worker.member.clone(),
            key: rcgen::KeyPair::try_from(worker.key.serialize_der()).unwrap(),
        };
        wrong_cluster.member.cluster = uuid::Uuid::new_v4().to_string().parse().unwrap();
        let signed = proof(&wrong_cluster, authority.identity.node(), &nonce).unwrap();
        assert_eq!(
            verify(&authority.access, &authority.identity, &nonce, &signed),
            Err(Error::Denied)
        );
        // Revoked identities may reserve only their own slot for the authenticated
        // tombstone; this proof never grants work or other-node capacity.
        authority.revoke(worker.node()).unwrap();
        let signed = proof(&worker, authority.identity.node(), &nonce).unwrap();
        assert_eq!(
            verify(&authority.access, &authority.identity, &nonce, &signed),
            Ok(worker.node())
        );
    }

    #[test]
    fn partial_proof_is_nonblocking_and_has_one_absolute_deadline() {
        let (_temp, _, mut authority, files) = setup();
        let worker = enrolled(&mut authority, &files, Role::Worker, "partial-worker");
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        let mut probe = Probe::new(listener.accept().unwrap().0).unwrap();
        assert_eq!(
            probe
                .progress(&authority.access, &authority.identity)
                .unwrap(),
            None
        );
        client.write_all(&[MARKER]).unwrap();
        while probe.sent != 32 {
            assert_eq!(
                probe
                    .progress(&authority.access, &authority.identity)
                    .unwrap(),
                None
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        let mut nonce = [0; 32];
        client.read_exact(&mut nonce).unwrap();
        let signed = proof(&worker, authority.identity.node(), &nonce).unwrap();
        client.write_all(&signed[..50]).unwrap();
        while probe.received != 50 {
            assert_eq!(
                probe
                    .progress(&authority.access, &authority.identity)
                    .unwrap(),
                None
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        probe.deadline = Instant::now();
        client.write_all(&signed[50..]).unwrap();
        assert_eq!(
            probe.progress(&authority.access, &authority.identity),
            Err(Error::Expired)
        );
    }

    #[test]
    fn capacity_proof_cannot_switch_tls_identity_or_enter_enrollment() {
        use rustls::{
            ClientConfig, ClientConnection, RootCertStore,
            pki_types::{PrivatePkcs8KeyDer, ServerName},
        };
        use std::{net::TcpListener, sync::Arc, thread};
        let (_temp, _, mut authority, files) = setup();
        let worker = enrolled(&mut authority, &files, Role::Worker, "tls-proof-worker");
        let other = enrolled(&mut authority, &files, Role::Worker, "tls-proof-other");
        for alpn in [crate::registry::ALPN, crate::transport::BOOTSTRAP] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            thread::scope(|scope| {
                let reader = authority.enrollment_reader().unwrap();
                let server = scope.spawn(|| {
                    reader.registry(
                        listener.accept().unwrap().0,
                        Instant::now() + Duration::from_secs(3),
                    )
                });
                let mut socket = client(
                    TcpStream::connect(listener.local_addr().unwrap()).unwrap(),
                    &worker,
                    authority.identity.node(),
                    Instant::now() + Duration::from_secs(3),
                )
                .unwrap();
                let mut roots = RootCertStore::empty();
                roots.add(authority.trust().unwrap().ca.into()).unwrap();
                let mut config = ClientConfig::builder_with_provider(Arc::new(
                    rustls::crypto::ring::default_provider(),
                ))
                .with_protocol_versions(&[&rustls::version::TLS13])
                .unwrap()
                .with_root_certificates(roots)
                .with_client_auth_cert(
                    vec![other.member.certificate.clone().into()],
                    PrivatePkcs8KeyDer::from(other.key.serialize_der()).into(),
                )
                .unwrap();
                config.alpn_protocols = vec![alpn.to_vec()];
                let mut tls = ClientConnection::new(
                    Arc::new(config),
                    ServerName::try_from(authority.identity.member.name()).unwrap(),
                )
                .unwrap();
                let _ = tls.complete_io(&mut socket);
                assert!(matches!(server.join().unwrap(), Err(Error::Denied)));
            });
        }
    }
}
