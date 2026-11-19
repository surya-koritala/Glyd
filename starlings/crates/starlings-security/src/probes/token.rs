//! Opt-in authenticated diagnostic exchanges, never model tokens or placement evidence.
use super::*;

pub(super) const TIME_LIMIT: Duration = Duration::from_secs(30);
const PROFILE_BYTES: usize = 2 * 1024 * 1024;
const PAYLOAD: usize = 16 * 1024;
const REPLY: usize = 64;
const WAKES: u32 = 2000;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GapBasis {
    StartToStart,
    AfterReply,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenProfile {
    pub gap_basis: GapBasis,
    pub keepalive_ms: Option<u32>,
}
impl TokenProfile {
    pub(super) fn validate(&self, limits: &ProbeLimits) -> Result<()> {
        if limits.payload_bytes != PAYLOAD as u32
            || !(5..=100).contains(&limits.ping_samples)
            || !(20..=100).contains(&limits.inter_ping_gap_ms)
            || self
                .keepalive_ms
                .is_some_and(|n| !(5..=20).contains(&n) || n >= limits.inter_ping_gap_ms)
        {
            return Err(Error::Invalid);
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenProfilePhase {
    Pause,
    Request,
    Reply,
    FinalReceipt,
    Statistics,
    Discriminator,
    Payload,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenSample {
    pub sequence: u32,
    pub started_us: Option<Decimal>,
    pub rtt_us: Option<u32>,
    pub keepalive_frames: u32,
    pub error: Option<Error>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenProfileReport {
    pub version: Version,
    pub producer_sha256: starlings_protocol::Digest,
    pub requested_samples: u32,
    pub completed_samples: u32,
    pub samples: Vec<TokenSample>,
    pub gap_basis: GapBasis,
    pub keepalive_ms: Option<u32>,
    pub keepalive_frames: u32,
    pub elapsed_us: Decimal,
    pub started_at_unix_ms: Option<Decimal>,
    pub rtt_p50_us: Option<u32>,
    pub rtt_p99_us: Option<u32>,
    pub error: Option<Error>,
    pub failure_phase: Option<TokenProfilePhase>,
    pub local_tcp_nodelay: bool,
    pub native_qualified: bool,
}
#[derive(Serialize)]
pub struct TokenProfileSnapshot<'a> {
    pub report: &'a ProbeReport,
    pub age_ms: Option<Decimal>,
    pub admission: bool,
}
fn profile_statistics(samples: &[u32]) -> Result<(u32, Decimal, u32, Option<u32>)> {
    let (mean, variance) = statistics_up_to(samples, 100)?;
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let p50 = sorted[(sorted.len() * 50).div_ceil(100) - 1];
    let p99 = (sorted.len() >= 100).then(|| sorted[(sorted.len() * 99).div_ceil(100) - 1]);
    Ok((mean, variance, p50, p99))
}
impl TokenProfileReport {
    fn new(hello: &Hello, producer_sha256: starlings_protocol::Digest) -> Self {
        let profile = hello
            .limits
            .token_profile
            .as_ref()
            .expect("profile dispatch");
        Self {
            version: Version,
            producer_sha256,
            requested_samples: hello.limits.ping_samples,
            completed_samples: 0,
            samples: Vec::new(),
            gap_basis: profile.gap_basis.clone(),
            keepalive_ms: profile.keepalive_ms,
            keepalive_frames: 0,
            elapsed_us: Decimal(0),
            started_at_unix_ms: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .ok()
                .and_then(|n| u64::try_from(n.as_millis()).ok())
                .map(Decimal),
            rtt_p50_us: None,
            rtt_p99_us: None,
            error: None,
            failure_phase: None,
            local_tcp_nodelay: true,
            native_qualified: false,
        }
    }
    fn fail(&mut self, error: Error, phase: TokenProfilePhase) {
        self.error = Some(error);
        self.failure_phase = Some(phase);
    }
}
pub(super) struct ProfileContext {
    addresses: (SocketAddr, SocketAddr),
    producer_sha256: starlings_protocol::Digest,
}
/// Called only after the approved limits/role/epoch and matching Hello are verified.
/// Pre-authentication, legacy probes and their deadlines are unchanged.
pub(super) fn activate(session: &mut Session, deadline: Instant) -> Result<ProfileContext> {
    session.check()?;
    if Instant::now() >= deadline {
        return Err(Error::Expired);
    }
    let socket = session.wire.socket()?;
    socket.set_nodelay(true)?;
    if !socket.nodelay()? {
        return Err(Error::Denied);
    }
    let addresses = (socket.local_addr()?, socket.peer_addr()?);
    let binary = std::fs::File::open(std::env::current_exe()?)?;
    let length = binary.metadata()?.len();
    if length > 128 * 1024 * 1024 {
        return Err(Error::Capacity);
    }
    let producer_sha256 = crate::catalog::fingerprint(binary, length)?;
    session.check()?;
    session.deadline = deadline;
    session.remaining += PROFILE_BYTES - BYTE_LIMIT;
    session.wire.set_deadline(deadline);
    Ok(ProfileContext {
        addresses,
        producer_sha256,
    })
}
fn response(id: Uuid, sequence: u32) -> [u8; REPLY] {
    let mut bytes = [0x5a; REPLY];
    bytes[..16].copy_from_slice(id.as_bytes());
    bytes[16..20].copy_from_slice(&sequence.to_le_bytes());
    bytes
}
fn payload(io: &mut Session, sending: bool) -> Result<()> {
    // The discriminator is the first byte of the exact16KiB request, not extra payload.
    let mut remaining = PAYLOAD - 1;
    let mut block = [0x5a; BLOCK];
    while remaining != 0 {
        let n = remaining.min(BLOCK);
        if sending {
            io.write_all(&block[..n])?;
        } else {
            io.read_exact(&mut block[..n])?;
            if block[..n] != [0x5a; BLOCK][..n] {
                return Err(Error::Invalid);
            }
        }
        remaining -= n;
    }
    Ok(())
}
fn gap_until(profile: &TokenProfile, start: Instant, reply: Instant, gap: Duration) -> Instant {
    match profile.gap_basis {
        GapBasis::StartToStart => start + gap,
        GapBasis::AfterReply => reply + gap,
    }
}
fn pause(
    io: &mut Session,
    until: Instant,
    keepalive_ms: Option<u32>,
    wakes: &mut u32,
) -> Result<()> {
    let interval = keepalive_ms.map(|n| Duration::from_millis(n.into()));
    let mut next = interval.map(|n| Instant::now() + n);
    loop {
        io.check()?;
        let now = Instant::now();
        if now >= until {
            return Ok(());
        }
        let target = next.unwrap_or(until).min(until).min(io.deadline);
        std::thread::sleep(target.saturating_duration_since(now));
        io.check()?;
        let now = Instant::now();
        if now >= until {
            return Ok(());
        }
        if next.is_some_and(|n| now >= n) {
            if *wakes >= WAKES {
                return Err(Error::Capacity);
            }
            io.write_all(&[0])?;
            io.flush()?;
            *wakes += 1;
            // A delayed host sends one wake, never a burst of missed timer ticks.
            next = interval.map(|n| Instant::now() + n);
        }
    }
}
fn finish_report(
    session: &Session,
    hello: &Hello,
    peers: [NodeId; 2],
    context: ProfileContext,
    began: Instant,
    mut profile: TokenProfileReport,
    client: bool,
) -> ProbeReport {
    let elapsed = began.elapsed();
    profile.elapsed_us = Decimal(u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX));
    let samples: Vec<_> = profile.samples.iter().filter_map(|s| s.rtt_us).collect();
    let stats = if client && !samples.is_empty() {
        match profile_statistics(&samples) {
            Ok((mean, variance, p50, p99)) => {
                profile.rtt_p50_us = Some(p50);
                profile.rtt_p99_us = p99;
                Some((mean, variance))
            }
            Err(error) => {
                profile.fail(error, TokenProfilePhase::Statistics);
                None
            }
        }
    } else {
        None
    };
    let forward = profile.completed_samples * PAYLOAD as u32;
    let backward = profile.completed_samples * REPLY as u32;
    let basis = if client {
        "token-profile-client-window"
    } else {
        "token-profile-server-window"
    };
    let [source, target] = peers;
    let (local_address, peer_address) = context.addresses;
    ProbeReport {
        version: Version,
        id: hello.id,
        epoch: hello.epoch,
        local_address,
        peer_address,
        selected_interface: crate::network::selected_interface(local_address),
        rtt_samples_us: None, // Per-exchange RTTs and start offsets are in token_profile.samples.
        rtt_mean_us: stats.as_ref().map(|s| s.0),
        rtt_variance_us2: stats.map(|s| s.1),
        inter_ping_gap_ms: hello.limits.inter_ping_gap_ms,
        transfers: [
            measurement(source, target, forward, elapsed, basis).expect("bounded sample window"),
            measurement(target, source, backward, elapsed, basis).expect("bounded sample window"),
        ],
        application_payload_bytes: Decimal(u64::from(forward) + u64::from(backward)),
        total_application_bytes: Decimal((PROFILE_BYTES - session.remaining) as u64),
        token_profile: Some(profile),
    }
}
pub(super) fn measure(
    io: &mut Session,
    hello: &Hello,
    source: NodeId,
    target: NodeId,
    context: ProfileContext,
) -> ProbeReport {
    let began = Instant::now();
    let config = hello
        .limits
        .token_profile
        .as_ref()
        .expect("profile dispatch");
    let gap = Duration::from_millis(hello.limits.inter_ping_gap_ms.into());
    let mut profile = TokenProfileReport::new(hello, context.producer_sha256.clone());
    let mut previous = None;
    let mut phase = TokenProfilePhase::Pause;
    for sequence in 0..hello.limits.ping_samples {
        let mut sample = TokenSample {
            sequence,
            started_us: None,
            rtt_us: None,
            keepalive_frames: 0,
            error: None,
        };
        let before = profile.keepalive_frames;
        let result = (|| -> Result<()> {
            phase = TokenProfilePhase::Pause;
            if let Some((start, reply)) = previous {
                pause(
                    io,
                    gap_until(config, start, reply, gap),
                    config.keepalive_ms,
                    &mut profile.keepalive_frames,
                )?;
            }
            let start = Instant::now();
            sample.started_us = Some(Decimal(
                u64::try_from(start.duration_since(began).as_micros())
                    .map_err(|_| Error::Invalid)?,
            ));
            phase = TokenProfilePhase::Request;
            io.write_all(&[1])?;
            payload(io, true)?;
            io.flush()?;
            phase = TokenProfilePhase::Reply;
            let mut bytes = [0; REPLY];
            io.read_exact(&mut bytes)?;
            if bytes != response(hello.id, sequence) {
                return Err(Error::Invalid);
            }
            sample.rtt_us = Some(
                u32::try_from(start.elapsed().as_micros().max(1)).map_err(|_| Error::Invalid)?,
            );
            previous = Some((start, Instant::now()));
            Ok(())
        })();
        sample.keepalive_frames = profile.keepalive_frames - before;
        if let Err(error) = result {
            sample.error = Some(error);
            profile.fail(error, phase);
            profile.samples.push(sample);
            break;
        }
        profile.completed_samples += 1;
        profile.samples.push(sample);
    }
    if profile.error.is_none() {
        let result = (|| -> Result<()> {
            io.write_all(&[2])?;
            io.flush()?;
            let mut ack = [0];
            io.read_exact(&mut ack)?;
            if ack != [2] {
                return Err(Error::Invalid);
            }
            io.check()
        })();
        if let Err(error) = result {
            profile.fail(error, TokenProfilePhase::FinalReceipt);
        }
    }
    finish_report(io, hello, [source, target], context, began, profile, true)
}
pub(super) fn serve(
    io: &mut Session,
    hello: &Hello,
    source: NodeId,
    target: NodeId,
    context: ProfileContext,
) -> ProbeReport {
    let began = Instant::now();
    let mut profile = TokenProfileReport::new(hello, context.producer_sha256.clone());
    let mut phase = TokenProfilePhase::Discriminator;
    let result = (|| -> Result<()> {
        loop {
            phase = TokenProfilePhase::Discriminator;
            let mut kind = [0];
            io.read_exact(&mut kind)?;
            match kind[0] {
                0 if profile.keepalive_ms.is_some() && profile.keepalive_frames < WAKES => {
                    profile.keepalive_frames += 1
                }
                1 if profile.completed_samples < hello.limits.ping_samples => {
                    phase = TokenProfilePhase::Payload;
                    payload(io, false)?;
                    phase = TokenProfilePhase::Reply;
                    io.write_all(&response(hello.id, profile.completed_samples))?;
                    io.flush()?;
                    profile.completed_samples += 1;
                }
                2 if profile.completed_samples == hello.limits.ping_samples => {
                    phase = TokenProfilePhase::FinalReceipt;
                    io.write_all(&[2])?;
                    io.flush()?;
                    return io.check();
                }
                _ => return Err(Error::Invalid),
            }
        }
    })();
    if let Err(error) = result {
        profile.fail(error, phase);
    }
    finish_report(io, hello, [source, target], context, began, profile, false)
}

#[cfg(test)]
mod tests;
