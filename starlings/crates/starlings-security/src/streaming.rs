//! Per-stream output gates: no database/file access on the output path.
use crate::{Error, Result, Scope, now};
use std::{
    io::Write,
    net::{Shutdown, TcpStream},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

const REFRESH_TTL: Duration = Duration::from_secs(2);
const STREAM_LIFETIME: Duration = Duration::from_secs(600);
const STREAM_BYTES: usize = 16 * 1024 * 1024;
const EVENT_BYTES: usize = 64 * 1024;

pub(crate) struct Grant {
    pub token_hash: String,
    pub scope: Scope,
    pub client: starlings_protocol::NodeId,
    gate: Mutex<Gate>,
}
struct Gate {
    allowed: bool,
    expires: i64,
    wall_high: i64,
    expires_monotonic: Instant,
    refreshed_until: Instant,
}
impl Grant {
    pub fn new(
        token_hash: String,
        client: starlings_protocol::NodeId,
        scope: Scope,
        expires: i64,
    ) -> Result<Self> {
        let wall = now()?;
        let lifetime = expires
            .checked_sub(wall)
            .filter(|n| *n > 0)
            .ok_or(Error::Expired)?;
        let started = Instant::now();
        Ok(Self {
            token_hash,
            client,
            scope,
            gate: Mutex::new(Gate {
                allowed: true,
                expires,
                wall_high: wall,
                expires_monotonic: started + Duration::from_secs(lifetime as u64),
                refreshed_until: started + REFRESH_TTL,
            }),
        })
    }
    pub fn close(&self) {
        self.gate.lock().unwrap_or_else(|e| e.into_inner()).allowed = false;
    }
    pub fn refresh(&self, check: impl FnOnce() -> Result<i64>) -> Result<()> {
        let mut gate = self.gate.lock().map_err(|_| Error::Denied)?;
        // A maintenance gap expires the grant even if no output attempted to use it.
        if gate.check().is_err() {
            return Ok(());
        }
        match check() {
            Ok(expires) => {
                if gate.check().is_err() {
                    return Ok(());
                }
                gate.expires = gate.expires.min(expires);
                gate.refreshed_until = Instant::now() + REFRESH_TTL;
                Ok(())
            }
            Err(error) => {
                gate.allowed = false;
                if error == Error::Busy {
                    Err(error)
                } else {
                    Ok(())
                }
            }
        }
    }
    pub fn enqueue<T>(&self, enqueue: impl FnOnce() -> T) -> Result<T> {
        let mut gate = self.gate.lock().map_err(|_| Error::Denied)?;
        gate.check()?;
        Ok(enqueue())
    }
}
impl Gate {
    fn check(&mut self) -> Result<()> {
        let wall = match now() {
            Ok(wall) => wall,
            Err(error) => {
                self.allowed = false;
                return Err(error);
            }
        };
        let time = Instant::now();
        if !self.allowed
            || wall < self.wall_high
            || wall >= self.expires
            || time >= self.expires_monotonic
            || time >= self.refreshed_until
        {
            #[cfg(debug_assertions)]
            if std::env::var_os("STARLINGS_TEST_CLIENT_GATE_TRACE").is_some() {
                let _ = writeln!(
                    std::io::stderr(),
                    "CLIENT_GATE_REFUSED closed={} wall_rollback={} credential_expired={} monotonic_expired={} maintenance_gap={}",
                    !self.allowed,
                    wall < self.wall_high,
                    wall >= self.expires,
                    time >= self.expires_monotonic,
                    time >= self.refreshed_until,
                );
            }
            self.allowed = false;
            return Err(Error::Expired);
        }
        self.wall_high = wall;
        Ok(())
    }
}

/// An authenticated response, with no headers sent yet. Slow work runs after admission.
pub struct ClientResponse {
    output: ClientStream,
}
impl ClientResponse {
    pub fn client(&self) -> starlings_protocol::NodeId {
        self.output.grant.client
    }
    pub fn scope(&self) -> &Scope {
        &self.output.grant.scope
    }
    pub(crate) fn token_hash(&self) -> &str {
        &self.output.grant.token_hash
    }
    pub(crate) fn new(socket: TcpStream, grant: Arc<Grant>) -> Result<Self> {
        socket.set_nonblocking(true)?;
        Ok(Self {
            output: ClientStream {
                socket,
                grant,
                deadline: Instant::now() + STREAM_LIFETIME,
                sent: 0,
                finished: false,
            },
        })
    }
    pub fn watch(&self) -> Result<ClientWatch> {
        self.output.watch()
    }
    pub fn stream(self) -> Result<ClientStream> {
        self.begin_stream(None)
    }
    pub fn stream_for(self, session: starlings_protocol::SessionId) -> Result<ClientStream> {
        self.begin_stream(Some(session))
    }
    fn begin_stream(
        mut self,
        session: Option<starlings_protocol::SessionId>,
    ) -> Result<ClientStream> {
        let session = session
            .map(|id| format!("X-Compute-Session-Id: {id}\r\n"))
            .unwrap_or_default();
        self.output.send(&[b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n", session.as_bytes(), b"\r\n"])?;
        Ok(self.output)
    }
    /// Send one bounded JSON result through the same live gate as SSE. Consumes the response.
    pub fn json(self, bytes: &[u8]) -> Result<()> {
        self.json_status("200 OK", bytes)
    }
    /// One terminal error before headers, through the same live authorization gate.
    /// Callers supply a redacted message, never backend diagnostics or request data.
    pub fn fail(self, error: &starlings_protocol::WireError) -> Result<()> {
        use starlings_protocol::ErrorDetailCode::*;
        let status = match error.error.code {
            InvalidRequest | UnsupportedModel | UnsupportedProfile | UnsupportedFeature => {
                "400 Bad Request"
            }
            Unauthenticated => "401 Unauthorized",
            Forbidden => "403 Forbidden",
            StaleEpoch | Cancelled | ArtifactMismatch | ProtocolMismatch => "409 Conflict",
            CapacityUnavailable | StaleCapability | WorkerLost => "503 Service Unavailable",
            DeadlineExceeded => "504 Gateway Timeout",
            Internal => "500 Internal Server Error",
        };
        self.json_status(
            status,
            &starlings_protocol::encode(error, starlings_protocol::FixtureMode::Reject)?,
        )
    }
    fn json_status(mut self, status: &'static str, bytes: &[u8]) -> Result<()> {
        if bytes.len() > starlings_protocol::MAX_JSON_BYTES {
            return Err(Error::Capacity);
        }
        starlings_protocol::parse_json(bytes)?;
        let header = format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n\r\n",
            bytes.len()
        );
        self.output.send(&[header.as_bytes(), bytes])?;
        self.output.finished = true;
        Ok(())
    }
}

/// Check from a bounded supervisor while the backend blocks, including before its first byte.
/// On failure, cancel backend work; keep reservations until trusted drain. Checking does not
/// start a thread or call the backend. Drop the watch when its response finishes.
pub struct ClientWatch {
    socket: TcpStream,
    grant: Arc<Grant>,
    deadline: Instant,
}
impl ClientWatch {
    pub fn check(&self) -> Result<()> {
        let result = (|| {
            crate::client::remaining(self.deadline)?;
            match self.grant.enqueue(|| self.socket.peek(&mut [0]))? {
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
                    ) =>
                {
                    Ok(())
                }
                Ok(0) => Err(Error::Transport),
                Ok(_) => Err(Error::Invalid), // One request only: pipelined input is forbidden.
                Err(_) => Err(Error::Transport),
            }
        })();
        if result.is_err() {
            self.grant.close();
            let _ = self.socket.shutdown(Shutdown::Both);
        }
        result
    }
}

/// Owns bounded SSE writes. Dropping a failed stream closes its socket; caller cancels backend work.
pub struct ClientStream {
    pub(crate) socket: TcpStream,
    pub(crate) grant: Arc<Grant>,
    pub(crate) deadline: Instant,
    pub(crate) sent: usize,
    pub(crate) finished: bool,
}
impl ClientStream {
    pub fn watch(&self) -> Result<ClientWatch> {
        Ok(ClientWatch {
            socket: self.socket.try_clone()?,
            grant: self.grant.clone(),
            deadline: self.deadline,
        })
    }
    pub fn event(&mut self, bytes: &[u8]) -> Result<()> {
        if self.finished {
            return Err(Error::Denied);
        }
        let result = self.event_inner(bytes);
        if result.is_err() {
            self.abort();
        }
        result
    }
    fn event_inner(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() > EVENT_BYTES {
            return Err(Error::Capacity);
        }
        let value = starlings_protocol::parse_json(bytes)?;
        // Compact serialization prevents literal newlines from injecting SSE fields.
        let bytes = serde_json::to_vec(&value)?;
        self.send(&[b"data: ", &bytes, b"\n\n"])
    }
    pub fn finish(&mut self) -> Result<()> {
        if self.finished {
            return Err(Error::Denied);
        }
        if let Err(error) = self.send(&[b"data: [DONE]\n\n"]) {
            self.abort();
            return Err(error);
        }
        self.finished = true;
        Ok(())
    }
    /// One redacted terminal error, only while the client remains authorized.
    /// Always closes the output gate; never follows it with a success marker.
    pub fn fail(&mut self, error: &starlings_protocol::WireError) -> Result<()> {
        if self.finished {
            return Err(Error::Denied);
        }
        let result = (|| {
            let bytes = starlings_protocol::encode(error, starlings_protocol::FixtureMode::Reject)?;
            if bytes.len() > EVENT_BYTES {
                return Err(Error::Capacity);
            }
            self.send(&[b"event: error\ndata: ", &bytes, b"\n\n"])
        })();
        self.abort();
        result
    }
    pub fn abort(&mut self) {
        self.finished = true;
        self.grant.close();
        let _ = self.socket.shutdown(Shutdown::Both);
    }
    fn send(&mut self, parts: &[&[u8]]) -> Result<()> {
        let total = parts.iter().try_fold(self.sent, |n, part| {
            n.checked_add(part.len()).ok_or(Error::Capacity)
        })?;
        if total > STREAM_BYTES {
            return Err(Error::Capacity);
        }
        for part in parts {
            if let Err(error) = write(&mut self.socket, &self.grant, part, self.deadline) {
                self.abort();
                return Err(error);
            }
        }
        self.sent = total;
        Ok(())
    }
}
impl Drop for ClientStream {
    fn drop(&mut self) {
        self.abort();
    }
}
fn write(socket: &mut TcpStream, grant: &Grant, mut bytes: &[u8], hard: Instant) -> Result<()> {
    let mut idle = Instant::now() + Duration::from_secs(2);
    while !bytes.is_empty() {
        if Instant::now() >= hard.min(idle) {
            return Err(Error::Expired);
        }
        match grant.enqueue(|| socket.write(&bytes[..bytes.len().min(4096)]))? {
            Ok(0) => return Err(Error::Transport),
            Ok(n) => {
                bytes = &bytes[n..];
                idle = Instant::now() + Duration::from_secs(2);
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(1))
            }
            Err(_) => return Err(Error::Transport),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Action, Role,
        enrollment::tests::{enrolled, setup},
    };
    #[test]
    fn deferred_json_rejects_invalid_oversized_and_expired_results_before_headers() {
        use std::{io::Read, net::TcpListener};
        for failure in 0..4 {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let mut reader = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
            reader
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let grant = Arc::new(
                Grant::new(
                    String::new(),
                    "00000000-0000-4000-8000-000000000001".parse().unwrap(),
                    Scope {
                        model: starlings_protocol::byte_digest(b"synthetic"),
                        action: Action::Infer,
                    },
                    now().unwrap() + 60,
                )
                .unwrap(),
            );
            let mut response = ClientResponse::new(listener.accept().unwrap().0, grant).unwrap();
            let result = match failure {
                0 => response.json(br#"{"duplicate":1,"duplicate":2}"#),
                1 => response.json(&vec![b' '; starlings_protocol::MAX_JSON_BYTES + 1]),
                2 => {
                    response.output.deadline = Instant::now();
                    assert_eq!(response.watch().unwrap().check(), Err(Error::Expired));
                    response.json(b"{}")
                }
                _ => {
                    response.output.grant.close();
                    response.fail(&starlings_protocol::WireError {
                        error: starlings_protocol::ErrorDetail {
                            code: starlings_protocol::ErrorDetailCode::WorkerLost,
                            message: "Generation stopped".into(),
                            request_id: "00000000-0000-4000-8000-000000000002".parse().unwrap(),
                            retryable: false,
                        },
                    })
                }
            };
            assert!(result.is_err());
            let mut bytes = Vec::new();
            reader.read_to_end(&mut bytes).unwrap();
            assert!(bytes.is_empty());
        }
    }
    #[test]
    fn invalid_events_capacity_and_deadlines_close_output_permanently() {
        use std::{io::Read, net::TcpListener};
        for failure in 0..3 {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let mut reader = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
            reader
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let socket = listener.accept().unwrap().0;
            let grant = Arc::new(
                Grant::new(
                    String::new(),
                    "00000000-0000-4000-8000-000000000001".parse().unwrap(),
                    Scope {
                        model: starlings_protocol::byte_digest(b"synthetic"),
                        action: Action::Infer,
                    },
                    now().unwrap() + 60,
                )
                .unwrap(),
            );
            let mut stream = ClientResponse::new(socket, grant)
                .unwrap()
                .stream()
                .unwrap();
            let result = match failure {
                0 => stream.event(br#"{"duplicate":1,"duplicate":2}"#),
                1 => {
                    stream.sent = STREAM_BYTES;
                    stream.finish()
                }
                _ => {
                    stream.deadline = Instant::now();
                    stream.event(b"{}")
                }
            };
            assert!(result.is_err());
            assert!(stream.finished);
            assert!(stream.event(b"{}").is_err());
            assert!(stream.finish().is_err());
            let mut bytes = String::new();
            reader.read_to_string(&mut bytes).unwrap();
            assert!(!bytes.contains("data:"));
        }
    }
    #[test]
    fn grant_checks_do_not_take_the_policy_lock_and_revocation_never_revives() {
        let (_temp, _, mut authority, files) = setup();
        let controller = enrolled(&mut authority, &files, Role::Controller, "stream-grant");
        let scope = Scope {
            model: starlings_protocol::byte_digest(b"synthetic"),
            action: Action::Infer,
        };
        authority
            .set_permissions(controller.node(), std::slice::from_ref(&scope))
            .unwrap();
        let token = authority
            .issue_client_token(controller.node(), std::slice::from_ref(&scope), 60)
            .unwrap();
        let grant = authority.client_admission().grant(&token, &scope).unwrap();
        std::thread::scope(|threads| {
            let guard = authority.access.lock().unwrap();
            let (tx, rx) = std::sync::mpsc::sync_channel(1);
            let grant = &grant;
            let check = threads.spawn(move || {
                let start = Instant::now();
                for _ in 0..10_000 {
                    grant.enqueue(|| ()).unwrap();
                }
                tx.send(start.elapsed()).unwrap();
            });
            let result = rx.recv_timeout(Duration::from_secs(2));
            drop(guard);
            check.join().unwrap();
            eprintln!(
                "stream_gate_ns_mean={}",
                result.unwrap().as_nanos() / 10_000
            );
        });
        authority.revoke_client_token(&token).unwrap();
        assert!(grant.enqueue(|| panic!("revoked output")).is_err());
        grant.refresh(|| Ok(now().unwrap() + 60)).unwrap();
        assert!(grant.enqueue(|| ()).is_err());
        let token = authority
            .issue_client_token(controller.node(), std::slice::from_ref(&scope), 60)
            .unwrap();
        let stale = authority.client_admission().grant(&token, &scope).unwrap();
        stale.gate.lock().unwrap().refreshed_until = Instant::now();
        // Maintenance must not revive a stale idle grant before any output detects expiry.
        authority.maintain(0).unwrap();
        assert!(stale.enqueue(|| ()).is_err());
        let grants: Vec<_> = (0..30)
            .map(|_| authority.client_admission().grant(&token, &scope).unwrap())
            .collect();
        assert!(matches!(
            authority.client_admission().grant(&token, &scope),
            Err(Error::Capacity)
        ));
        drop(grants);
        assert!(authority.client_admission().grant(&token, &scope).is_ok());
    }
}
