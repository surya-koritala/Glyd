//! Bounded directed acknowledged-application probes. No model or inference authorization.
use crate::{
    AuthorityEpoch, EnrollmentReader, Error, Identity, Invitation, NodeStatus, RegistryFence,
    Result, Role, Trust,
    authorization::{Access, LiveConnection},
    transport,
};
use serde::{Deserialize, Serialize};
use starlings_protocol::{Decimal, NodeId, Version};
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    net::{Shutdown, SocketAddr, TcpStream},
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime},
};
use uuid::Uuid;
#[cfg(test)]
mod tests;
mod token;
pub use token::{GapBasis, TokenProfile, TokenProfileReport, TokenProfileSnapshot};

pub(crate) const ALPN: &[u8] = b"starlings-probe/1";
const BATCH_TIME: Duration = Duration::from_secs(5);
pub const PROBE_TTL: Duration = Duration::from_secs(30);
const BLOCK: usize = 4096;
const BYTE_LIMIT: usize = 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProbeLimits {
    pub version: Version,
    pub payload_bytes: u32,
    pub ping_samples: u32,
    pub inter_ping_gap_ms: u32,
    pub minimum_interval_ms: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_profile: Option<TokenProfile>,
}
impl ProbeLimits {
    pub fn validate(&self) -> Result<()> {
        if !(BLOCK as u32..=256 * 1024).contains(&self.payload_bytes)
            || !self.payload_bytes.is_multiple_of(BLOCK as u32)
            || !(1..=if self.token_profile.is_some() { 100 } else { 8 })
                .contains(&self.ping_samples)
            || self.inter_ping_gap_ms > 100
            || !(30_000..=3_600_000).contains(&self.minimum_interval_ms)
        {
            return Err(Error::Invalid);
        }
        if let Some(profile) = &self.token_profile {
            profile.validate(self)?;
        }
        Ok(())
    }
}
/// Attempts consume the cooldown even if the peer later fails; no retry amplification.
pub struct ProbeBudget {
    limits: ProbeLimits,
    attempts: Mutex<BTreeMap<NodeId, Instant>>,
}
impl ProbeBudget {
    pub fn new(limits: ProbeLimits) -> Result<Self> {
        limits.validate()?;
        Ok(Self {
            limits,
            attempts: Mutex::new(BTreeMap::new()),
        })
    }
    pub fn limits(&self) -> &ProbeLimits {
        &self.limits
    }
    fn claim(&self, node: NodeId) -> Result<()> {
        let now = Instant::now();
        let mut attempts = self.attempts.lock().map_err(|_| Error::Denied)?;
        let interval = Duration::from_millis(u64::from(self.limits.minimum_interval_ms));
        // Enrollment itself is capped at 64; a bounded expired-entry scan retains active cooldowns.
        attempts.retain(|_, at| now.duration_since(*at) < interval);
        if attempts.contains_key(&node) {
            return Err(Error::Capacity);
        }
        if attempts.len() >= 64 {
            return Err(Error::Capacity);
        }
        attempts.insert(node, now);
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProbeRoute {
    pub version: Version,
    pub destination: SocketAddr,
    pub source: Option<SocketAddr>,
    pub limits: ProbeLimits,
}
impl ProbeRoute {
    pub fn validate(&self) -> Result<()> {
        self.limits.validate()?;
        crate::network::validate_routes(&[self.destination], self.source)
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Hello {
    version: Version,
    id: Uuid,
    epoch: AuthorityEpoch,
    limits: ProbeLimits,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Marker {
    id: Uuid,
    sequence: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DirectedMeasurement {
    pub source_node: NodeId,
    pub destination_node: NodeId,
    pub payload_bytes: u32,
    pub elapsed_us: Decimal,
    pub timing_basis: String,
    /// Includes control/acknowledgement overhead; not raw bandwidth or tensor throughput.
    pub acknowledged_bytes_per_second: Decimal,
}
#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProbeReport {
    pub version: Version,
    pub id: Uuid,
    pub epoch: AuthorityEpoch,
    pub local_address: SocketAddr,
    pub peer_address: SocketAddr,
    pub selected_interface: Option<crate::network::Interface>,
    pub rtt_samples_us: Option<Vec<u32>>,
    pub rtt_mean_us: Option<u32>,
    pub rtt_variance_us2: Option<Decimal>,
    pub inter_ping_gap_ms: u32,
    pub transfers: [DirectedMeasurement; 2],
    pub application_payload_bytes: Decimal,
    pub total_application_bytes: Decimal,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_profile: Option<TokenProfileReport>,
}
/// These timestamps are local, never deserialized. Viewing cannot refresh a measurement.
pub struct ProbeObservation {
    pub report: ProbeReport,
    at: Instant,
    wall: SystemTime,
}
#[derive(Serialize)]
pub struct ProbeView<'a> {
    pub report: &'a ProbeReport,
    pub age_ms: u32,
}
impl ProbeView<'_> {
    /// Validated directed facts only, never a calibrated stage/model placement decision.
    pub fn placement_inputs(&self) -> Result<[starlings_core::links::LinkInput; 2]> {
        if self.report.token_profile.is_some() {
            return Err(Error::Denied);
        }
        let sample_id = self
            .report
            .id
            .to_string()
            .parse()
            .map_err(|_| Error::Invalid)?;
        let authority_incarnation = self
            .report
            .epoch
            .incarnation
            .to_string()
            .parse()
            .map_err(|_| Error::Invalid)?;
        let inputs = self
            .report
            .transfers
            .clone()
            .map(|m| starlings_core::links::LinkInput {
                version: Version,
                fixture: false,
                sample_id,
                authority_epoch: self.report.epoch.number,
                authority_incarnation,
                source: m.source_node,
                destination: m.destination_node,
                payload_bytes: m.payload_bytes,
                elapsed_us: m.elapsed_us,
                acknowledged_bytes_per_second: m.acknowledged_bytes_per_second,
                rtt_mean_us: self.report.rtt_mean_us,
                rtt_variance_us2: self.report.rtt_variance_us2,
                inter_ping_gap_ms: self.report.inter_ping_gap_ms,
                timing_basis: m.timing_basis,
                age_ms: self.age_ms,
            });
        for input in &inputs {
            input
                .validate(starlings_protocol::FixtureMode::Reject)
                .map_err(|_| Error::Invalid)?;
        }
        Ok(inputs)
    }
}
impl ProbeObservation {
    /// Historical diagnostic only, including failed/stale profiles. Never live placement input.
    pub fn token_profile_snapshot(&self) -> Option<TokenProfileSnapshot<'_>> {
        self.report.token_profile.as_ref()?;
        Some(TokenProfileSnapshot {
            report: &self.report,
            age_ms: self
                .age()
                .and_then(|n| u64::try_from(n.as_millis()).ok())
                .map(Decimal),
            admission: false,
        })
    }
    pub(crate) fn captured(report: ProbeReport, at: Instant, wall: SystemTime) -> Self {
        Self { report, at, wall }
    }
    fn age(&self) -> Option<Duration> {
        Some(self.at.elapsed().max(self.wall.elapsed().ok()?))
    }
    pub fn current(&self, fence: &RegistryFence) -> Option<&ProbeReport> {
        let elapsed = self.age()?;
        (elapsed < PROBE_TTL && fence.status().ok()? == (self.report.epoch, NodeStatus::Online))
            .then_some(&self.report)
    }
    pub fn view(&self, fence: &RegistryFence) -> Option<ProbeView<'_>> {
        Some(ProbeView {
            report: self.current(fence)?,
            age_ms: self.age()?.as_millis().try_into().ok()?,
        })
    }
    pub fn view_authority(
        &self,
        authority: &crate::Authority,
        node: NodeId,
    ) -> Option<ProbeView<'_>> {
        if self.age()? >= PROBE_TTL || self.report.transfers[0].source_node != node {
            return None;
        }
        let state = authority.access.lock().ok()?;
        state.credential(&state.member(node).ok()?).ok()?;
        if state.draining || !state.registry.as_ref()?.online(node, self.report.epoch) {
            return None;
        }
        Some(ProbeView {
            report: &self.report,
            age_ms: self.age()?.as_millis().try_into().ok()?,
        })
    }
}
enum Authorization {
    Worker(RegistryFence),
    Coordinator {
        access: Access,
        live: Arc<LiveConnection>,
    },
}
struct Session {
    wire: transport::Wire,
    auth: Authorization,
    epoch: AuthorityEpoch,
    deadline: Instant,
    wall_high: i64,
    expires: i64,
    remaining: usize,
}
impl Session {
    fn check(&mut self) -> Result<()> {
        let now = crate::now()?;
        if Instant::now() >= self.deadline || now < self.wall_high || now >= self.expires {
            return Err(Error::Expired);
        }
        self.wall_high = now;
        match &self.auth {
            Authorization::Worker(fence) => {
                if fence.status()? != (self.epoch, NodeStatus::Online) {
                    return Err(Error::Expired);
                }
            }
            Authorization::Coordinator { access, live } => {
                let state = access.lock()?;
                state.credential(&live.local)?;
                state.credential(&live.peer)?;
                if state.draining
                    || !state
                        .registry
                        .as_ref()
                        .is_some_and(|r| r.online(live.peer.node, self.epoch))
                {
                    return Err(Error::Expired);
                }
            }
        }
        Ok(())
    }
    fn marker(&mut self, id: Uuid, sequence: u32) -> Result<()> {
        crate::send(self, &Marker { id, sequence })
    }
    fn expect(&mut self, id: Uuid, sequence: u32) -> Result<()> {
        let marker: Marker = crate::receive(self)?;
        if marker.id != id || marker.sequence != sequence {
            return Err(Error::Invalid);
        }
        Ok(())
    }
    fn send_payload(&mut self, bytes: u32) -> Result<()> {
        let block = [0x5a; BLOCK];
        for _ in 0..bytes as usize / BLOCK {
            self.write_all(&block)?;
        }
        self.flush()?;
        self.check()
    }
    fn receive_payload(&mut self, bytes: u32) -> Result<()> {
        let mut block = [0; BLOCK];
        for _ in 0..bytes as usize / BLOCK {
            self.read_exact(&mut block)?;
            if block != [0x5a; BLOCK] {
                return Err(Error::Invalid);
            }
        }
        self.check()
    }
}
impl Read for Session {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        self.check()
            .map_err(|_| std::io::Error::other("probe authorization expired"))?;
        if bytes.len() > self.remaining {
            return Err(std::io::Error::other("probe byte budget"));
        }
        let n = self.wire.read(bytes)?;
        self.remaining -= n;
        self.check()
            .map_err(|_| std::io::Error::other("probe authorization expired"))?;
        Ok(n)
    }
}
impl Write for Session {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.check()
            .map_err(|_| std::io::Error::other("probe authorization expired"))?;
        if bytes.len() > self.remaining {
            return Err(std::io::Error::other("probe byte budget"));
        }
        let n = self.wire.write(bytes)?;
        self.remaining -= n;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.check()
            .map_err(|_| std::io::Error::other("probe authorization expired"))?;
        self.wire.flush()
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        if let Ok(socket) = self.wire.socket() {
            let _ = socket.shutdown(Shutdown::Both);
        }
    }
}
fn expiry(member: &[u8]) -> Result<i64> {
    let (_, cert) = x509_parser::parse_x509_certificate(member).map_err(|_| Error::Invalid)?;
    Ok(cert.validity().not_after.timestamp())
}
fn measurement(
    source: NodeId,
    target: NodeId,
    bytes: u32,
    elapsed: Duration,
    timing_basis: &'static str,
) -> Result<DirectedMeasurement> {
    let micros: u64 = elapsed
        .as_micros()
        .max(1)
        .try_into()
        .map_err(|_| Error::Invalid)?;
    let rate = u64::from(bytes)
        .checked_mul(1_000_000)
        .ok_or(Error::Invalid)?
        / micros;
    Ok(DirectedMeasurement {
        source_node: source,
        destination_node: target,
        payload_bytes: bytes,
        elapsed_us: Decimal(micros),
        timing_basis: timing_basis.into(),
        acknowledged_bytes_per_second: Decimal(rate),
    })
}
fn statistics(samples: &[u32]) -> Result<(u32, Decimal)> {
    statistics_up_to(samples, 8)
}
fn statistics_up_to(samples: &[u32], maximum: usize) -> Result<(u32, Decimal)> {
    if samples.is_empty() || samples.len() > maximum || samples.contains(&0) {
        return Err(Error::Invalid);
    }
    let sum: u128 = samples.iter().map(|n| u128::from(*n)).sum();
    let mean = sum / samples.len() as u128;
    let variance = samples
        .iter()
        .map(|n| u128::from(*n).abs_diff(mean).pow(2))
        .sum::<u128>()
        / samples.len() as u128;
    Ok((
        mean.try_into().map_err(|_| Error::Invalid)?,
        Decimal(variance.try_into().map_err(|_| Error::Invalid)?),
    ))
}
/// Separate opt-in coordinator endpoint only. Signed reconnect proof must have been admitted.
pub(crate) fn serve(
    socket: TcpStream,
    reader: EnrollmentReader,
    node: NodeId,
    admitted_until: Instant,
    budget: &ProbeBudget,
) -> Result<ProbeReport> {
    let trust = Trust {
        ca: reader.ca.clone(),
        cluster: reader.identity.cluster(),
        members: reader.access.members()?,
        access: Some(reader.access.clone()),
    };
    let deadline = admitted_until.min(Instant::now() + BATCH_TIME);
    let mut wire = transport::server_until(socket, &reader.identity, Some(&trust), ALPN, deadline)?;
    let cert = transport::peer_certificate(&wire)?;
    let peer = trust
        .members
        .iter()
        .find(|m| m.node == node && m.certificate == cert)
        .ok_or(Error::Denied)?
        .clone();
    if peer.role != Role::Worker || reader.identity.role() != Role::Coordinator {
        return Err(Error::Denied);
    }
    {
        let state = reader.access.lock()?;
        state.credential(&peer)?;
        if state.draining {
            return Err(Error::Denied);
        }
    }
    budget.claim(peer.node)?;
    let hello: Hello = crate::receive(&mut wire)?;
    hello.limits.validate()?;
    if hello.id.get_version_num() != 4
        || hello.id.get_variant() != uuid::Variant::RFC4122
        || hello.limits != budget.limits
    {
        return Err(Error::Invalid);
    }
    let local = reader.identity.member.clone();
    let live = reader
        .access
        .register(local.clone(), peer.clone(), wire.socket()?)?;
    let mut session = Session {
        expires: expiry(&local.certificate)?.min(expiry(&peer.certificate)?),
        wall_high: crate::now()?,
        auth: Authorization::Coordinator {
            access: reader.access,
            live,
        },
        wire,
        epoch: hello.epoch,
        deadline,
        remaining: BYTE_LIMIT - serde_json::to_vec(&hello)?.len() - 4,
    };
    session.check()?;
    crate::send(&mut session, &hello)?;
    if hello.limits.token_profile.is_some() {
        let context = token::activate(&mut session, Instant::now() + token::TIME_LIMIT)?;
        return Ok(token::serve(
            &mut session,
            &hello,
            peer.node,
            local.node,
            context,
        ));
    }
    for sequence in 0..hello.limits.ping_samples {
        session.expect(hello.id, sequence)?;
        session.marker(hello.id, sequence)?;
    }
    session.expect(hello.id, 10)?;
    let upload = Instant::now();
    session.receive_payload(hello.limits.payload_bytes)?;
    session.marker(hello.id, 11)?;
    let incoming = measurement(
        peer.node,
        local.node,
        hello.limits.payload_bytes,
        upload.elapsed(),
        "receiver-start-to-ack-sent",
    )?;
    session.expect(hello.id, 12)?;
    let download = Instant::now();
    session.send_payload(hello.limits.payload_bytes)?;
    session.expect(hello.id, 13)?;
    let outgoing = measurement(
        local.node,
        peer.node,
        hello.limits.payload_bytes,
        download.elapsed(),
        "sender-start-to-ack-received",
    )?;
    // Client RTTs/direction timings are not invented by the server; return only its own report.
    session.marker(hello.id, 14)?;
    let local_address = session.wire.local_addr()?;
    let report = ProbeReport {
        version: Version,
        id: hello.id,
        epoch: hello.epoch,
        local_address,
        peer_address: session.wire.socket()?.peer_addr()?,
        selected_interface: crate::network::selected_interface(local_address),
        rtt_samples_us: None,
        rtt_mean_us: None,
        rtt_variance_us2: None,
        inter_ping_gap_ms: hello.limits.inter_ping_gap_ms,
        transfers: [incoming, outgoing],
        application_payload_bytes: Decimal(u64::from(hello.limits.payload_bytes) * 2),
        total_application_bytes: Decimal((BYTE_LIMIT - session.remaining) as u64),
        token_profile: None,
    };
    session.check()?;
    Ok(report)
}
pub fn measure(
    route: &ProbeRoute,
    identity: &Identity,
    invitation: &Invitation,
    fence: RegistryFence,
    budget: &ProbeBudget,
) -> Result<ProbeObservation> {
    route.validate()?;
    invitation.verify_identity(identity)?;
    if identity.role() != Role::Worker || route.limits != budget.limits {
        return Err(Error::Denied);
    }
    fence.check_identity(identity.node(), identity.cluster())?;
    let (epoch, status) = fence.status()?;
    if status != NodeStatus::Online {
        return Err(Error::Expired);
    }
    budget.claim(identity.node())?;
    let at = Instant::now();
    let wall = SystemTime::now();
    let deadline = at + BATCH_TIME;
    let socket = crate::network::connect(route.destination, route.source, Duration::from_secs(2))?;
    let socket = crate::reconnect::client(socket, identity, invitation.coordinator.node, deadline)?;
    let trust = Trust {
        ca: invitation.ca.clone(),
        cluster: identity.cluster(),
        members: vec![invitation.coordinator.clone()],
        access: None,
    };
    let wire = transport::client_until(
        socket,
        &trust,
        &invitation.coordinator,
        Some(identity),
        ALPN,
        deadline,
    )?;
    crate::generation::verify_coordinator(transport::peer_certificate(&wire)?, invitation)?;
    let expires =
        expiry(&identity.member.certificate)?.min(expiry(transport::peer_certificate(&wire)?)?);
    let mut session = Session {
        wire,
        auth: Authorization::Worker(fence),
        epoch,
        deadline,
        wall_high: crate::now()?,
        expires,
        remaining: BYTE_LIMIT,
    };
    let hello = Hello {
        version: Version,
        id: Uuid::new_v4(),
        epoch,
        limits: route.limits.clone(),
    };
    crate::send(&mut session, &hello)?;
    let accepted: Hello = crate::receive(&mut session)?;
    if accepted != hello {
        return Err(Error::Denied);
    }
    if hello.limits.token_profile.is_some() {
        let context = token::activate(&mut session, at + token::TIME_LIMIT)?;
        let report = token::measure(
            &mut session,
            &hello,
            identity.node(),
            invitation.coordinator.node,
            context,
        );
        return Ok(ProbeObservation { at, wall, report });
    }
    let mut samples = Vec::new();
    for sequence in 0..hello.limits.ping_samples {
        if sequence != 0 {
            std::thread::sleep(Duration::from_millis(u64::from(
                hello.limits.inter_ping_gap_ms,
            )));
        }
        let began = Instant::now();
        session.marker(hello.id, sequence)?;
        session.expect(hello.id, sequence)?;
        samples.push(
            began
                .elapsed()
                .as_micros()
                .max(1)
                .try_into()
                .map_err(|_| Error::Invalid)?,
        );
    }
    let upload = Instant::now();
    session.marker(hello.id, 10)?;
    session.send_payload(hello.limits.payload_bytes)?;
    session.expect(hello.id, 11)?;
    let outgoing = measurement(
        identity.node(),
        invitation.coordinator.node,
        hello.limits.payload_bytes,
        upload.elapsed(),
        "sender-start-to-ack-received",
    )?;
    let download = Instant::now();
    session.marker(hello.id, 12)?;
    session.receive_payload(hello.limits.payload_bytes)?;
    session.marker(hello.id, 13)?;
    let incoming = measurement(
        invitation.coordinator.node,
        identity.node(),
        hello.limits.payload_bytes,
        download.elapsed(),
        "start-request-to-receive-and-ack-sent",
    )?;
    session.expect(hello.id, 14)?;
    session.check()?;
    let (mean, variance) = statistics(&samples)?;
    let local = session.wire.local_addr()?;
    Ok(ProbeObservation {
        at,
        wall,
        report: ProbeReport {
            version: Version,
            id: hello.id,
            epoch,
            local_address: local,
            peer_address: route.destination,
            selected_interface: crate::network::selected_interface(local),
            rtt_samples_us: Some(samples),
            rtt_mean_us: Some(mean),
            rtt_variance_us2: Some(variance),
            inter_ping_gap_ms: hello.limits.inter_ping_gap_ms,
            transfers: [outgoing, incoming],
            application_payload_bytes: Decimal(u64::from(hello.limits.payload_bytes) * 2),
            total_application_bytes: Decimal((BYTE_LIMIT - session.remaining) as u64),
            token_profile: None,
        },
    })
}
