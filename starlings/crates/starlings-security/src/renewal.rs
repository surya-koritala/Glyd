use crate::authorization::OVERLAP_SECONDS;
use crate::enrollment::{Member, certificate_params, checked_csr};
use crate::{Authority, Channel, Error, Identity, PeerStream, Result, now};
use rcgen::{KeyPair, PublicKeyData};
use serde::{Deserialize, Serialize};
use starlings_protocol::Version;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Renewal {
    version: Version,
    csr: Vec<u8>,
}
impl Identity {
    /// Use an existing mutually authenticated control stream. Save the returned identity to
    /// a new protected file before switching consumers; never overwrite the old key in place.
    pub fn renew(&self, stream: &mut PeerStream) -> Result<Self> {
        self.renew_with(stream, &self.prepare_renewal()?)
    }
    /// Persist this candidate privately before requesting rotation. Reuse it after a lost
    /// response; generating another key cannot recover a committed rotation.
    pub fn prepare_renewal(&self) -> Result<Self> {
        Ok(Self {
            member: self.member.clone(),
            key: KeyPair::generate()?,
        })
    }
    pub fn renew_with(&self, stream: &mut PeerStream, candidate: &Self) -> Result<Self> {
        if candidate.node() != self.node()
            || candidate.cluster() != self.cluster()
            || candidate.role() != self.role()
        {
            return Err(Error::Denied);
        }
        let (_, live) = stream.authority()?;
        if stream.channel != Channel::Control
            || live.local.node != self.node()
            || live.local.certificate != self.member.certificate
            || live.peer.role != crate::Role::Coordinator
        {
            return Err(Error::Denied);
        }
        let mut next = Self {
            member: self.member.clone(),
            key: KeyPair::try_from(candidate.key.serialize_der())?,
        };
        stream.send_record(&Renewal {
            version: Version,
            csr: next.csr()?,
        })?;
        let member: Member = stream.receive_record()?;
        if member.node != self.node()
            || member.cluster != self.cluster()
            || member.role != self.role()
        {
            return Err(Error::Denied);
        }
        // Verify the replacement's signature, validity, exact name and new key, not just its fields.
        crate::transport::verify_issued(&member, &stream.ca, &next.key)?;
        next.member = member;
        Ok(next)
    }
}
impl Authority {
    /// Renewal requires the current credential and proof of possession of a fresh CSR key.
    pub fn renew_peer(&self, stream: &mut PeerStream) -> Result<()> {
        let (access, live) = stream.authority()?;
        if !self.access.same(access)
            || stream.channel != Channel::Control
            || live.local.node != self.identity.node()
            || live.peer.role == crate::Role::Coordinator
        {
            return Err(Error::Denied);
        }
        let peer = live.peer.clone();
        let request: Renewal = stream.receive_record()?;
        let member = self.rotate(&peer, &request.csr, None)?;
        stream.send_record(&member)
    }
    pub fn renew_coordinator(&mut self) -> Result<()> {
        let key = KeyPair::generate()?;
        let next = Identity {
            member: self.identity.member.clone(),
            key,
        };
        let member = self.rotate(&self.identity.member, &next.csr()?, Some(&next.key))?;
        self.identity = Identity {
            member,
            key: next.key,
        };
        Ok(())
    }
    fn rotate(
        &self,
        old: &Member,
        csr: &[u8],
        coordinator_key: Option<&KeyPair>,
    ) -> Result<Member> {
        let mut state = self.access.lock()?;
        state.credential(old)?;
        let current_member = state.member(old.node)?;
        let mut csr = checked_csr(csr)?;
        if current_member.certificate != old.certificate {
            let (_, certificate) = x509_parser::parse_x509_certificate(&current_member.certificate)
                .map_err(|_| Error::Invalid)?;
            return if certificate.public_key().raw == csr.public_key.subject_public_key_info() {
                Ok(current_member)
            } else {
                Err(Error::Denied)
            };
        }
        let (_, old_certificate) =
            x509_parser::parse_x509_certificate(&old.certificate).map_err(|_| Error::Invalid)?;
        if csr.public_key.subject_public_key_info() == old_certificate.public_key().raw {
            return Err(Error::Denied);
        }
        let current = now()?;
        let busy: bool = state.database.query_row(
            "SELECT EXISTS(SELECT 1 FROM previous WHERE node=?1 AND expires>?2)",
            (old.node.to_string(), current),
            |r| r.get(0),
        )?;
        if busy {
            return Err(Error::Denied);
        }
        let mut member = old.clone();
        csr.params = certificate_params(&member)?;
        member.certificate = csr.signed_by(&self.issuer)?.der().to_vec();
        let filename = format!("coordinator-{}.json", crate::hash(&member.certificate));
        if let Some(key) = coordinator_key {
            let saved = Identity {
                member: member.clone(),
                key: KeyPair::try_from(key.serialize_der())?,
            };
            saved.save(&self.directory, &filename)?;
        }
        let transaction = state
            .database
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        transaction.execute("INSERT INTO previous VALUES (?1,?2,?3) ON CONFLICT(node) DO UPDATE SET record=excluded.record,expires=excluded.expires", (old.node.to_string(), serde_json::to_vec(old)?, current + OVERLAP_SECONDS))?;
        transaction.execute(
            "UPDATE members SET record=?2 WHERE node=?1",
            (old.node.to_string(), serde_json::to_vec(&member)?),
        )?;
        if coordinator_key.is_some() {
            transaction.execute("INSERT INTO settings VALUES ('coordinator',?1) ON CONFLICT(name) DO UPDATE SET value=excluded.value", [&filename])?;
        }
        transaction.commit()?;
        // Publication is final. Live checks enforce current/previous credentials on I/O;
        // reconciliation failure must not turn a committed rotation into a failed outcome.
        let _ = state.reconcile();
        Ok(member)
    }
}

impl Identity {
    pub fn certificate_seconds_remaining(&self) -> Result<i64> {
        let (_, certificate) = x509_parser::parse_x509_certificate(&self.member.certificate)
            .map_err(|_| Error::Invalid)?;
        Ok(certificate.validity().not_after.timestamp() - now()?)
    }
    pub(crate) fn replace_saved(&self, directory: &crate::PrivateDir) -> Result<()> {
        self.replace_file(directory, "identity.json")
    }
}
impl Authority {
    /// Renew validity without changing the enrolled coordinator public-key pin.
    /// Explicit key rotation remains a separate operation with out-of-band pin recovery.
    pub fn refresh_coordinator(&mut self) -> Result<()> {
        if self.identity.certificate_seconds_remaining()? > 3 * 86400 {
            return Ok(());
        }
        let old = self.identity.member.clone();
        let next = self.refresh_member(&old, true)?;
        self.identity.member = next;
        Ok(())
    }
    pub(crate) fn refresh_member(&self, old: &Member, coordinator: bool) -> Result<Member> {
        let mut state = self.access.lock()?;
        let current = state.member(old.node)?;
        if current.cluster != old.cluster || current.role != old.role {
            return Err(Error::Denied);
        }
        if current.certificate != old.certificate {
            state.credential(old)?;
            return Ok(current);
        }
        if !coordinator {
            state.credential(old)?;
        }
        let (_, certificate) = x509_parser::parse_x509_certificate(&current.certificate)
            .map_err(|_| Error::Invalid)?;
        if certificate.validity().not_after.timestamp() - now()? > 3 * 86400 {
            return Ok(current);
        }
        // Public key comes from the verified stored leaf, never caller input.
        let key = rcgen::SubjectPublicKeyInfo::from_der(certificate.public_key().raw)?;
        let mut member = current.clone();
        member.certificate = certificate_params(&member)?
            .signed_by(&key, &self.issuer)?
            .der()
            .to_vec();
        let filename = format!("coordinator-{}.json", crate::hash(&member.certificate));
        if coordinator {
            Identity {
                member: member.clone(),
                key: KeyPair::try_from(self.identity.key.serialize_der())?,
            }
            .save(&self.directory, &filename)?;
        }
        let transaction = state
            .database
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        transaction.execute("INSERT INTO previous VALUES (?1,?2,?3) ON CONFLICT(node) DO UPDATE SET record=excluded.record,expires=excluded.expires", (old.node.to_string(), serde_json::to_vec(old)?, certificate.validity().not_after.timestamp()))?;
        transaction.execute(
            "UPDATE members SET record=?2 WHERE node=?1",
            (member.node.to_string(), serde_json::to_vec(&member)?),
        )?;
        if coordinator {
            transaction.execute("INSERT INTO settings VALUES ('coordinator',?1) ON CONFLICT(name) DO UPDATE SET value=excluded.value", [&filename])?;
        }
        transaction.commit()?;
        Ok(member)
    }
}

#[cfg(test)]
mod tests {
    use crate::enrollment::tests::{enrolled, setup};
    #[test]
    fn retrying_the_same_csr_recovers_a_committed_rotation() {
        let (_temp, _path, mut authority, files) = setup();
        let old = enrolled(&mut authority, &files, crate::Role::Worker, "rotate-retry");
        let candidate = old.prepare_renewal().unwrap();
        let csr = candidate.csr().unwrap();
        let first = authority.rotate(&old.member, &csr, None).unwrap();
        let retry = authority.rotate(&old.member, &csr, None).unwrap();
        assert_eq!(first.certificate, retry.certificate);
        assert!(
            authority
                .rotate(
                    &old.member,
                    &old.prepare_renewal().unwrap().csr().unwrap(),
                    None
                )
                .is_err()
        );
        crate::transport::verify_issued(&retry, &authority.trust().unwrap().ca, &candidate.key)
            .unwrap();
    }
}
