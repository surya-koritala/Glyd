//! Admission limits shared within a listener class; admin and client budgets are separate.
use crate::{Authority, Channel, Error, Identity, PeerStream, PendingEnrollment, Result, Trust};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    net::{IpAddr, SocketAddr, TcpListener, TcpStream},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

pub(crate) fn transient_accept(error: &std::io::Error) -> bool {
    use std::io::ErrorKind::*;
    matches!(
        error.kind(),
        WouldBlock
            | Interrupted
            | ConnectionAborted
            | ConnectionReset
            | OutOfMemory
            | NetworkDown
            | NetworkUnreachable
            | HostUnreachable
            | PermissionDenied
    ) || cfg!(target_os = "linux")
        && matches!(error.raw_os_error(), Some(64 | 71 | 92 | 95 | 105 | 112))
        || cfg!(target_os = "macos") && matches!(error.raw_os_error(), Some(42 | 55 | 64 | 100))
        || matches!(
            error.raw_os_error(),
            Some(23 | 24 | 10024 | 10055 | 10050 | 10051 | 10065)
        )
}
pub const MAX_CONNECTIONS: usize = 32;
pub const MAX_PER_IP: usize = 4;
pub const MAX_HANDSHAKES: usize = 8;
pub(crate) const REQUEST_TIME: Duration = Duration::from_secs(5);

/// Share within one listener class. Give admin/client traffic independent budgets.
#[derive(Clone)]
pub struct ListenerBudget(
    Arc<Mutex<State>>,
    Arc<Mutex<VecDeque<crate::reconnect::Probe>>>,
);
struct State {
    window: Instant,
    admitted: usize,
    active: usize,
    handshakes: usize,
    registry: BTreeSet<starlings_protocol::NodeId>,
    peers: BTreeMap<IpAddr, (usize, usize)>, // active, admissions in this window
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ListenerUsage {
    pub connections: usize,
    pub handshakes: usize,
    pub tracked_addresses: usize,
    pub pending_admissions: usize,
}
impl Default for ListenerBudget {
    fn default() -> Self {
        Self(
            Arc::new(Mutex::new(State {
                window: Instant::now(),
                admitted: 0,
                active: 0,
                handshakes: 0,
                peers: BTreeMap::new(),
                registry: BTreeSet::new(),
            })),
            Arc::new(Mutex::new(VecDeque::new())),
        )
    }
}
impl ListenerBudget {
    fn reconnect(&self, ip: IpAddr, node: starlings_protocol::NodeId) -> Result<Permit> {
        let mut state = self.0.lock().map_err(|_| Error::Denied)?;
        if state.registry.len() >= 63 || !state.registry.insert(node) {
            return Err(Error::Capacity);
        }
        Ok(Permit {
            budget: self.clone(),
            ip,
            handshake: false,
            node: Some(node),
        })
    }
    pub fn usage(&self) -> Result<ListenerUsage> {
        let pending_admissions = self.1.lock().map_err(|_| Error::Denied)?.len();
        let s = self.0.lock().map_err(|_| Error::Denied)?;
        Ok(ListenerUsage {
            connections: s.active + s.registry.len(),
            handshakes: s.handshakes,
            tracked_addresses: s.peers.len(),
            pending_admissions,
        })
    }
    pub(crate) fn acquire(&self, ip: IpAddr, handshake: bool) -> Result<Permit> {
        let mut s = self.0.lock().map_err(|_| Error::Denied)?;
        if s.window.elapsed() >= Duration::from_secs(1) {
            s.window = Instant::now();
            s.admitted = 0;
            s.peers.retain(|_, (active, rate)| {
                *rate = 0;
                *active != 0
            });
        }
        let (active, rate) = s.peers.get(&ip).copied().unwrap_or_default();
        if s.active >= MAX_CONNECTIONS
            || active >= MAX_PER_IP
            || s.admitted >= 32
            || rate >= 8
            || (handshake && s.handshakes >= MAX_HANDSHAKES)
            || (!s.peers.contains_key(&ip) && s.peers.len() >= 64)
        {
            return Err(Error::Capacity);
        }
        s.active += 1;
        s.admitted += 1;
        s.handshakes += usize::from(handshake);
        s.peers.insert(ip, (active + 1, rate + 1));
        Ok(Permit {
            budget: self.clone(),
            ip,
            handshake,
            node: None,
        })
    }
}
pub(crate) struct Permit {
    budget: ListenerBudget,
    ip: IpAddr,
    handshake: bool,
    node: Option<starlings_protocol::NodeId>,
}
impl Permit {
    fn registry(&mut self, node: starlings_protocol::NodeId) -> Result<()> {
        if let Some(reserved) = self.node {
            return if reserved == node {
                Ok(())
            } else {
                Err(Error::Denied)
            };
        }
        let mut state = self.budget.0.lock().map_err(|_| Error::Denied)?;
        if state.registry.len() >= 63 || !state.registry.insert(node) {
            return Err(Error::Capacity);
        }
        state.active -= 1;
        state.handshakes -= usize::from(self.handshake);
        if let Some((active, _)) = state.peers.get_mut(&self.ip) {
            *active -= 1;
        }
        self.handshake = false;
        self.node = Some(node);
        Ok(())
    }

    pub(crate) fn established(&mut self) {
        if self.handshake {
            if let Ok(mut s) = self.budget.0.lock() {
                s.handshakes -= 1;
            }
            self.handshake = false;
        }
    }
}
impl Drop for Permit {
    fn drop(&mut self) {
        if let Ok(mut s) = self.budget.0.lock() {
            if let Some(node) = self.node {
                s.registry.remove(&node);
                return;
            }
            s.active -= 1;
            s.handshakes -= usize::from(self.handshake);
            if let Some((active, _)) = s.peers.get_mut(&self.ip) {
                *active -= 1;
            }
        }
    }
}

/// LAN binding is explicit. Each accepted socket carries its admission through its lifetime.
pub struct PeerListener {
    listener: TcpListener,
    budget: ListenerBudget,
}
pub struct IncomingPeer {
    socket: TcpStream,
    permit: Permit,
    deadline: Instant,
}
impl PeerListener {
    /// Nonblocking admission for the coordinator's shared enrollment/registry port.
    /// Key proofs reserve one node-specific slot; bootstrap keeps the untrusted budget.
    pub fn try_accept_registry(&self, authority: &Authority) -> Result<Option<IncomingPeer>> {
        let mut probes = self.budget.1.lock().map_err(|_| Error::Denied)?;
        // Service existing progress before accepting replacements, so an idle flood
        // cannot hold every slot indefinitely. No thread or TLS object exists here.
        for _ in 0..probes.len() {
            let mut probe = probes.pop_front().ok_or(Error::Denied)?;
            match probe.progress(&authority.access, &authority.identity) {
                Ok(Some(node)) => {
                    let ip = match probe.socket.peer_addr() {
                        Ok(addr) => addr.ip(),
                        Err(_) => continue,
                    };
                    let permit = match node {
                        Some(node) => self.budget.reconnect(ip, node),
                        None => self.budget.acquire(ip, true),
                    };
                    let permit = match permit {
                        Ok(permit) => permit,
                        Err(Error::Capacity) => {
                            // A reply can reach the client before the old server slot drops.
                            // Keep one proved successor per node inside the original deadline.
                            if node.is_some() && !probes.iter().any(|p| p.verified == node) {
                                probes.push_back(probe);
                            }
                            continue;
                        }
                        Err(e) => return Err(e),
                    };
                    if probe.socket.set_nonblocking(false).is_err() {
                        continue;
                    }
                    return Ok(Some(IncomingPeer {
                        socket: probe.socket,
                        permit,
                        deadline: Instant::now() + REQUEST_TIME,
                    }));
                }
                Ok(None) => probes.push_back(probe),
                Err(_) => (), // Invalid/expired proof affects only this socket.
            }
        }
        for _ in 0..8 {
            match self.listener.accept() {
                Ok((socket, _)) => {
                    if probes.len() == 64 {
                        if let Some(index) = probes.iter().position(|p| p.verified.is_none()) {
                            probes.remove(index);
                        } else {
                            continue;
                        }
                    }
                    if let Ok(probe) = crate::reconnect::Probe::new(socket) {
                        probes.push_back(probe);
                    }
                }
                Err(e) if transient_accept(&e) => break,
                Err(e) => return Err(e.into()),
            }
        }
        Ok(None)
    }
    pub fn bind(address: SocketAddr, budget: ListenerBudget) -> Result<Self> {
        let listener = TcpListener::bind(address)?;
        listener.set_nonblocking(true)?;
        Ok(Self { listener, budget })
    }
    pub fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.listener.local_addr()?)
    }
    pub fn try_accept(&self) -> Result<Option<IncomingPeer>> {
        match self.listener.accept() {
            Ok((socket, peer)) => match self.accepted(socket, peer) {
                Ok(peer) => Ok(Some(peer)),
                Err(Error::Transport | Error::Denied) => Ok(None),
                Err(error) => Err(error),
            },
            Err(e) if transient_accept(&e) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
    pub fn accept(&self) -> Result<IncomingPeer> {
        loop {
            if let Some(peer) = self.try_accept()? {
                return Ok(peer);
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }
    /// Wait until `deadline` for the granted stage peer's data connection from `from`.
    /// A connection from another address, one that sends nothing within 250 ms, or one that
    /// fails the stage handshake for this grant, is dropped and counted; the wait goes on.
    /// Each handshake keeps the listener's request time and the grant's deadline, and gets
    /// at most one second. A listener-budget refusal is not counted; the wait goes on.
    // ponytail: one handshake at a time; a connection from the peer's address that sends a
    // first byte and then stalls holds the accept for up to a second.
    pub fn accept_stage(
        &self,
        local: &Identity,
        pin: &crate::Invitation,
        grant: &crate::StageGrant,
        from: IpAddr,
        deadline: Instant,
    ) -> Result<(PeerStream, u32)> {
        let mut refused = 0;
        loop {
            grant.check()?;
            if Instant::now() >= deadline {
                return Err(Error::Expired);
            }
            let (socket, address) = match self.listener.accept() {
                Ok(accepted) => accepted,
                Err(e) if transient_accept(&e) => {
                    std::thread::sleep(Duration::from_millis(2));
                    continue;
                }
                Err(e) => return Err(e.into()),
            };
            if address.ip() != from {
                refused += 1;
                continue;
            }
            let mut incoming = match self.accepted(socket, address) {
                Ok(incoming) => incoming,
                Err(Error::Capacity) => continue,
                Err(Error::Transport | Error::Denied) => {
                    refused += 1;
                    continue;
                }
                Err(error) => return Err(error),
            };
            // A TLS client speaks first; a silent socket does not get the handshake window.
            let first_byte = deadline
                .min(Instant::now() + Duration::from_millis(250))
                .checked_duration_since(Instant::now())
                .filter(|d| !d.is_zero());
            let spoke = first_byte.is_some_and(|wait| {
                incoming.socket.set_read_timeout(Some(wait)).is_ok()
                    && matches!(incoming.socket.peek(&mut [0]), Ok(1))
            });
            if !spoke {
                refused += 1;
                continue;
            }
            let until = incoming
                .deadline
                .min(deadline)
                .min(Instant::now() + Duration::from_secs(1));
            match PeerStream::accept_stage_until(incoming.socket, local, pin, grant.clone(), until)
            {
                Ok(mut stream) => {
                    incoming.permit.established();
                    stream.permit = Some(incoming.permit);
                    return Ok((stream, refused));
                }
                Err(_) => refused += 1,
            }
        }
    }
    fn accepted(&self, socket: TcpStream, peer: SocketAddr) -> Result<IncomingPeer> {
        socket.set_nonblocking(false)?;
        let permit = self.budget.acquire(peer.ip(), true)?;
        Ok(IncomingPeer {
            socket,
            permit,
            deadline: Instant::now() + REQUEST_TIME,
        })
    }
}
impl IncomingPeer {
    /// The local lifecycle owner selects the route grant before TLS admission.
    pub fn stage(
        mut self,
        identity: &Identity,
        pin: &crate::Invitation,
        grant: crate::StageGrant,
    ) -> Result<PeerStream> {
        let mut stream =
            PeerStream::accept_stage_until(self.socket, identity, pin, grant, self.deadline)?;
        self.permit.established();
        stream.permit = Some(self.permit);
        Ok(stream)
    }
    /// Use only a separately budgeted probe listener, never the heartbeat's per-node slot.
    pub fn probe(
        self,
        reader: crate::EnrollmentReader,
        budget: &crate::probes::ProbeBudget,
    ) -> Result<crate::probes::ProbeObservation> {
        let node = self.permit.node.ok_or(Error::Denied)?;
        let at = Instant::now();
        let wall = std::time::SystemTime::now();
        let report = crate::probes::serve(self.socket, reader, node, self.deadline, budget)?;
        Ok(crate::probes::ProbeObservation::captured(report, at, wall))
    }
    /// Worker-side generation TLS; the worker retains only its own enrolled identity.
    /// The connecting address, for a caller that admits only known peers before any handshake.
    pub fn peer_ip(&self) -> Option<std::net::IpAddr> {
        self.socket.peer_addr().ok().map(|a| a.ip())
    }
    /// A shorter handshake deadline than the listener's.
    pub fn within(mut self, limit: Duration) -> Self {
        self.deadline = self.deadline.min(Instant::now() + limit);
        self
    }
    pub fn generation(
        mut self,
        local: &Identity,
        invitation: &crate::Invitation,
        fence: crate::RegistryFence,
        binding: crate::GenerationBinding,
    ) -> Result<crate::GenerationStream> {
        let stream = crate::GenerationStream::accept(
            self.socket,
            local,
            invitation,
            fence,
            binding,
            self.deadline,
        )?;
        self.permit.established();
        Ok(stream.with_permit(self.permit))
    }
    pub fn authenticate(
        mut self,
        local: &Identity,
        trust: &Trust,
        channel: Channel,
    ) -> Result<PeerStream> {
        let mut stream =
            PeerStream::accept_until(self.socket, local, trust, channel, self.deadline)?;
        self.permit.established();
        stream.permit = Some(self.permit);
        Ok(stream)
    }
    /// Only invitation validation is reachable before enrollment. Approval remains local.
    pub fn enroll(self, authority: &Authority) -> Result<PendingEnrollment> {
        self.enroll_with(authority.enrollment_reader()?)
    }
    pub fn registry(mut self, reader: crate::EnrollmentReader) -> Result<crate::RegistryRequest> {
        let mut request = if let Some(node) = self.permit.node {
            reader.registry_admitted(self.socket, self.deadline, Some(node))?
        } else {
            reader.registry(self.socket, self.deadline)?
        };
        self.permit.established();
        match &mut request {
            crate::RegistryRequest::Enrollment(pending) => pending.permit = Some(self.permit),
            crate::RegistryRequest::Heartbeat(pending) => {
                self.permit.registry(pending.node())?;
                pending.permit = Some(self.permit);
            }
        }
        Ok(request)
    }
    pub fn enroll_with(mut self, reader: crate::EnrollmentReader) -> Result<PendingEnrollment> {
        let mut pending = reader.receive(self.socket, self.deadline)?;
        self.permit.established();
        pending.permit = Some(self.permit);
        Ok(pending)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn address_history_is_bounded_and_rate_refills() {
        let budget = ListenerBudget::default();
        let old: Vec<_> = (1..=32)
            .map(|n| budget.acquire(IpAddr::from([10, 0, 0, n]), false).unwrap())
            .collect();
        budget.0.lock().unwrap().window = Instant::now() - Duration::from_secs(2);
        assert!(matches!(
            budget.acquire("10.0.1.1".parse().unwrap(), false),
            Err(Error::Capacity)
        ));
        drop(old);
        let new: Vec<_> = (1..=32)
            .map(|n| budget.acquire(IpAddr::from([10, 0, 1, n]), false).unwrap())
            .collect();
        assert_eq!(budget.usage().unwrap().tracked_addresses, 64);
        let admin = ListenerBudget::default();
        assert!(admin.acquire("127.0.0.1".parse().unwrap(), false).is_ok());
        drop(new);
        budget.0.lock().unwrap().window = Instant::now() - Duration::from_secs(2);
        let permit = budget.acquire("10.0.2.1".parse().unwrap(), false).unwrap();
        assert_eq!(budget.usage().unwrap().tracked_addresses, 1);
        drop(permit);
    }

    #[test]
    fn registry_reserve_survives_untrusted_capacity_and_ipv6_hosts_are_independent() {
        let budget = ListenerBudget::default();
        let mut registry = Vec::new();
        for n in 1..=63u8 {
            budget.0.lock().unwrap().window = Instant::now() - Duration::from_secs(2);
            let mut permit = budget
                .acquire(IpAddr::V4(std::net::Ipv4Addr::new(10, 0, 0, n)), true)
                .unwrap();
            let node = format!("00000000-0000-4000-8000-{n:012x}").parse().unwrap();
            permit.registry(node).unwrap();
            registry.push(permit);
        }
        let mut handshakes = Vec::new();
        for n in 1..=8u8 {
            handshakes.push(
                budget
                    .acquire(IpAddr::V4(std::net::Ipv4Addr::new(10, 1, 0, n)), true)
                    .unwrap(),
            );
        }
        assert!(matches!(
            budget.acquire("10.2.0.1".parse().unwrap(), true),
            Err(Error::Capacity)
        ));
        assert_eq!(budget.usage().unwrap().connections, 71);
        drop(handshakes);
        assert_eq!(budget.usage().unwrap().connections, 63);
        drop(registry);
        assert_eq!(budget.usage().unwrap().connections, 0);
        let ipv6 = ListenerBudget::default();
        let mut permits = Vec::new();
        for _ in 1..=4 {
            permits.push(
                ipv6.acquire("fd00:1:2:3::1".parse().unwrap(), true)
                    .unwrap(),
            );
        }
        assert!(matches!(
            ipv6.acquire("fd00:1:2:3::1".parse().unwrap(), true),
            Err(Error::Capacity)
        ));
        assert!(
            ipv6.acquire("fd00:1:2:3::ffff".parse().unwrap(), true)
                .is_ok()
        );
        assert_eq!(ipv6.usage().unwrap().tracked_addresses, 2);
    }

    #[test]
    fn proved_successor_waits_for_the_previous_registry_slot_without_extending_deadline() {
        let (_temp, _, mut authority, files) = crate::enrollment::tests::setup();
        let worker = crate::enrollment::tests::enrolled(
            &mut authority,
            &files,
            crate::Role::Worker,
            "successor",
        );
        let budget = ListenerBudget::default();
        let listener = PeerListener::bind("127.0.0.1:0".parse().unwrap(), budget.clone()).unwrap();
        for expires in [false, true] {
            let held = budget
                .reconnect("127.0.0.1".parse().unwrap(), worker.node())
                .unwrap();
            std::thread::scope(|threads| {
                let client = threads.spawn(|| {
                    crate::reconnect::client(
                        TcpStream::connect(listener.local_addr().unwrap()).unwrap(),
                        &worker,
                        authority.identity.node(),
                        Instant::now() + Duration::from_secs(2),
                    )
                    .unwrap()
                });
                let deadline = Instant::now() + Duration::from_secs(2);
                let original = loop {
                    assert!(listener.try_accept_registry(&authority).unwrap().is_none());
                    if let Some(probe) = budget
                        .1
                        .lock()
                        .unwrap()
                        .iter()
                        .find(|p| p.verified == Some(worker.node()))
                    {
                        break probe.deadline;
                    }
                    assert!(Instant::now() < deadline);
                    std::thread::sleep(Duration::from_millis(1));
                };
                let socket = client.join().unwrap();
                assert_eq!(budget.usage().unwrap().connections, 1);
                assert!(listener.try_accept_registry(&authority).unwrap().is_none());
                assert_eq!(budget.1.lock().unwrap().front().unwrap().deadline, original);
                if expires {
                    budget.1.lock().unwrap().front_mut().unwrap().deadline = Instant::now();
                }
                drop(held);
                let admitted = listener.try_accept_registry(&authority).unwrap();
                assert_eq!(admitted.is_some(), !expires);
                drop(admitted);
                drop(socket);
                assert_eq!(budget.usage().unwrap().connections, 0);
                assert_eq!(budget.usage().unwrap().pending_admissions, 0);
            });
        }
    }

    #[test]
    fn admission_caps_and_release_are_shared() {
        let budget = ListenerBudget::default();
        let mut held = Vec::new();
        for n in 1..=32 {
            let ip = IpAddr::from([10, 0, 0, n]);
            held.push(budget.acquire(ip, n <= 8).unwrap());
        }
        assert_eq!(budget.usage().unwrap().connections, 32);
        assert!(matches!(
            budget.acquire(IpAddr::from([10, 0, 1, 1]), false),
            Err(Error::Capacity)
        ));
        drop(held);
        assert_eq!(budget.usage().unwrap().connections, 0);
        // Closing connections does not reset the rate budget.
        assert!(matches!(
            budget.acquire(IpAddr::from([10, 0, 1, 1]), false),
            Err(Error::Capacity)
        ));
        let budget = ListenerBudget::default();
        let held: Vec<_> = (1..=8)
            .map(|n| budget.acquire(IpAddr::from([10, 0, 0, n]), true).unwrap())
            .collect();
        assert!(matches!(
            budget.acquire(IpAddr::from([10, 0, 0, 9]), true),
            Err(Error::Capacity)
        ));
        drop(held);
    }
}

#[cfg(test)]
mod peer_deadline_tests {
    use super::*;
    use crate::enrollment::tests::setup;
    #[test]
    fn idle_bootstrap_times_out_and_releases_handshake_admission() {
        let (_temp, _, authority, _) = setup();
        let budget = ListenerBudget::default();
        let listener = PeerListener::bind("127.0.0.1:0".parse().unwrap(), budget.clone()).unwrap();
        let _client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let mut incoming = listener.accept().unwrap();
        incoming.deadline = Instant::now() + Duration::from_millis(100);
        assert_eq!(budget.usage().unwrap().handshakes, 1);
        let start = Instant::now();
        assert!(incoming.enroll(&authority).is_err());
        assert!(start.elapsed() < Duration::from_secs(1));
        assert_eq!(budget.usage().unwrap().connections, 0);
        assert_eq!(budget.usage().unwrap().handshakes, 0);
    }
}
