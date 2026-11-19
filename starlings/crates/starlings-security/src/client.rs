//! One bounded HTTP/1.1 request per loopback connection. No browser origins are trusted.
use crate::{
    Authority, ClientAdmission, ClientToken, Error, Result, Scope,
    listener::{ListenerBudget, Permit, REQUEST_TIME},
};
use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    time::{Duration, Instant},
};

const MAX_HEADERS: usize = 8192;
const MAX_BODY: usize = 1024 * 1024;
const MAX_RESPONSE: usize = 64 * 1024;
const CLIENT_SEND_BUFFER: usize = 64 * 1024;

pub struct ClientListener {
    listener: TcpListener,
    budget: ListenerBudget,
}
pub struct IncomingClient {
    socket: TcpStream,
    _permit: Permit,
    host: String,
    deadline: Instant,
}
/// Validated syntax is not application routing. The handler must bind its resource to the supplied scope.
pub struct ClientRequest {
    pub method: String,
    pub path: String,
    pub body: serde_json::Value,
}
impl ClientListener {
    pub fn bind_default(budget: ListenerBudget) -> Result<Self> {
        Self::bind(([127, 0, 0, 1], 9740).into(), budget)
    }
    pub fn bind(address: SocketAddr, budget: ListenerBudget) -> Result<Self> {
        if !address.ip().is_loopback() {
            return Err(Error::Denied);
        }
        let listener = TcpListener::bind(address)?;
        listener.set_nonblocking(true)?;
        Ok(Self { listener, budget })
    }
    pub fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.listener.local_addr()?)
    }
    pub fn accept(&self) -> Result<IncomingClient> {
        loop {
            if let Some(incoming) = self.try_accept()? {
                return Ok(incoming);
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }
    /// Admission only; move the result to a bounded worker before reading headers.
    pub fn try_accept(&self) -> Result<Option<IncomingClient>> {
        let (socket, peer) = match self.listener.accept() {
            Ok(pair) => pair,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        socket.set_nonblocking(false)?;
        // Bound queued output before admission; OS accounting may include extra overhead.
        socket2::SockRef::from(&socket).set_send_buffer_size(CLIENT_SEND_BUFFER)?;
        let permit = self.budget.acquire(peer.ip(), false)?;
        Ok(Some(IncomingClient {
            host: socket.local_addr()?.to_string(),
            socket,
            _permit: permit,
            deadline: Instant::now() + REQUEST_TIME,
        }))
    }
}
impl IncomingClient {
    /// Production HTTP admission reports bounded redacted failures. It never includes
    /// credentials/body fields, and authorizes before reading/allocating the body.
    pub fn prepare_with_errors(
        self,
        admission: &ClientAdmission,
        scope: &Scope,
        validate: impl FnOnce(&ClientRequest) -> Result<()>,
    ) -> Result<(ClientRequest, crate::ClientResponse)> {
        let mut socket = self.socket.try_clone()?;
        let mut authenticated = false;
        let result = self.prepare_with(admission, scope, |request| {
            authenticated = true;
            validate(request)
        });
        if let Err(error) = &result {
            use starlings_protocol::ErrorDetailCode as Code;
            let (status, code) = match error {
                Error::Denied if authenticated => ("403 Forbidden", Code::Forbidden),
                Error::Denied => ("401 Unauthorized", Code::Unauthenticated),
                Error::Invalid | Error::Capacity if !authenticated => {
                    ("400 Bad Request", Code::InvalidRequest)
                }
                Error::Invalid => ("400 Bad Request", Code::InvalidRequest),
                Error::Expired => ("504 Gateway Timeout", Code::DeadlineExceeded),
                Error::Capacity | Error::Busy => {
                    ("503 Service Unavailable", Code::CapacityUnavailable)
                }
                _ => ("500 Internal Server Error", Code::Internal),
            };
            let id = uuid::Uuid::new_v4()
                .to_string()
                .parse()
                .map_err(|_| Error::Invalid)?;
            let body = serde_json::to_vec(&starlings_protocol::WireError {
                error: starlings_protocol::ErrorDetail {
                    code,
                    message: "Request refused".into(),
                    request_id: id,
                    retryable: false,
                },
            })?;
            let mut bytes = format!("HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n\r\n", body.len()).into_bytes();
            bytes.extend(body);
            let deadline = Instant::now() + Duration::from_secs(2);
            let _ = (|| -> Result<()> {
                socket.set_nonblocking(false)?;
                let mut offset = 0;
                while offset < bytes.len() {
                    socket.set_write_timeout(Some(remaining(deadline)?))?;
                    match socket.write(&bytes[offset..])? {
                        0 => return Err(Error::Transport),
                        n => offset += n,
                    }
                }
                Ok(())
            })();
        }
        result
    }

    /// The server chooses scope from trusted route/catalog state, never from an unchecked body.
    /// Handler runs under the policy lock: short dispatch only, no blocking I/O or Authority reentry.
    /// Errors close the connection without reflecting credentials or request contents.
    pub fn serve(
        self,
        authority: &Authority,
        scope: &Scope,
        handler: impl FnOnce(ClientRequest) -> Result<Vec<u8>>,
    ) -> Result<()> {
        let (mut incoming, request, token) =
            self.read_request(&authority.client_admission(), scope)?;
        let response = authority.with_client_access(&token, scope, || handler(request))??;
        if response.len() > MAX_RESPONSE {
            return Err(Error::Capacity);
        }
        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n\r\n",
            response.len()
        );
        incoming.socket.set_nonblocking(true)?;
        write_authorized(
            &mut incoming.socket,
            authority,
            &token,
            scope,
            &[header.as_bytes(), response.as_slice()],
            Instant::now() + Duration::from_secs(2),
        )
    }
    /// Validate and dispatch outside the output gate. Validation must be short and bind
    /// the body model to `scope`. Authority maintenance must run at least once a second.
    pub fn stream(
        self,
        authority: &Authority,
        scope: &Scope,
        validate: impl FnOnce(&ClientRequest) -> Result<()>,
    ) -> Result<(ClientRequest, crate::ClientStream)> {
        let (request, response) = self.prepare(authority, scope, validate)?;
        Ok((request, response.stream()?))
    }
    /// Authorize a deferred response before dispatching slow backend work. The caller
    /// must bind the request to the trusted scope in the short validation callback,
    /// maintain Authority once a second, and monitor `ClientResponse::watch` while busy.
    pub fn prepare(
        self,
        authority: &Authority,
        scope: &Scope,
        validate: impl FnOnce(&ClientRequest) -> Result<()>,
    ) -> Result<(ClientRequest, crate::ClientResponse)> {
        self.prepare_with(&authority.client_admission(), scope, validate)
    }
    /// Admission on a background request thread without owning a signing key.
    pub fn prepare_with(
        self,
        admission: &ClientAdmission,
        scope: &Scope,
        validate: impl FnOnce(&ClientRequest) -> Result<()>,
    ) -> Result<(ClientRequest, crate::ClientResponse)> {
        let (incoming, request, token) = self.read_request(admission, scope)?;
        admission.with_access(&token, scope, || validate(&request))??;
        let grant = admission.grant(&token, scope)?;
        let response = crate::ClientResponse::new(incoming.socket, grant)?;
        // Authenticated responses use the authority's independent 32-grant budget.
        drop(incoming._permit);
        Ok((request, response))
    }
    fn read_request(
        mut self,
        admission: &ClientAdmission,
        scope: &Scope,
    ) -> Result<(Self, ClientRequest, ClientToken)> {
        let mut bytes = [0; MAX_HEADERS];
        let mut len = 0;
        let (method, path, token, body_len, start) = loop {
            if len == bytes.len() {
                return Err(Error::Capacity);
            }
            self.socket
                .set_read_timeout(Some(remaining(self.deadline)?))?;
            let read = self.socket.read(&mut bytes[len..])?;
            if read == 0 {
                return Err(Error::Invalid);
            }
            len += read;
            let mut headers = [httparse::EMPTY_HEADER; 32];
            let mut request = httparse::Request::new(&mut headers);
            let httparse::Status::Complete(start) =
                request.parse(&bytes[..len]).map_err(|_| Error::Invalid)?
            else {
                continue;
            };
            if request.version != Some(1) {
                return Err(Error::Invalid);
            }
            let method = request.method.ok_or(Error::Invalid)?;
            let path = request.path.ok_or(Error::Invalid)?;
            if !matches!(method, "GET" | "POST")
                || !path.starts_with('/')
                || path.starts_with("//")
                || path.contains(['?', '#', '%', '\\'])
                || path.split('/').any(|s| s == "." || s == "..")
            {
                return Err(Error::Invalid);
            }
            for (i, h) in request.headers.iter().enumerate() {
                if request.headers[..i]
                    .iter()
                    .any(|p| p.name.eq_ignore_ascii_case(h.name))
                {
                    return Err(Error::Invalid);
                }
                if matches!(
                    h.name.to_ascii_lowercase().as_str(),
                    "origin" | "transfer-encoding" | "content-encoding" | "expect" | "upgrade"
                ) || h.name.to_ascii_lowercase().starts_with("sec-fetch-")
                {
                    return Err(Error::Denied);
                }
            }
            let header = |name: &str| {
                request
                    .headers
                    .iter()
                    .find(|h| h.name.eq_ignore_ascii_case(name))
                    .map(|h| h.value)
            };
            if header("host") != Some(self.host.as_bytes()) {
                return Err(Error::Denied);
            }
            let auth = header("authorization").ok_or(Error::Denied)?;
            let token =
                ClientToken::from_bearer(auth.strip_prefix(b"Bearer ").ok_or(Error::Denied)?)?;
            // Authenticate before accepting body size, allocation or application parsing.
            admission.with_access(&token, scope, || ())?;
            let body_len = match header("content-length") {
                Some(value) if !value.is_empty() && value.iter().all(u8::is_ascii_digit) => {
                    std::str::from_utf8(value)
                        .map_err(|_| Error::Invalid)?
                        .parse::<usize>()
                        .map_err(|_| Error::Capacity)?
                }
                None if method == "GET" => 0,
                _ => return Err(Error::Invalid),
            };
            if body_len > MAX_BODY {
                return Err(Error::Capacity);
            }
            if (method == "GET" && body_len != 0)
                || (method == "POST"
                    && (body_len == 0 || header("content-type") != Some(b"application/json")))
            {
                return Err(Error::Invalid);
            }
            break (method.to_owned(), path.to_owned(), token, body_len, start);
        };
        if len - start > body_len {
            return Err(Error::Invalid);
        }
        let mut body = vec![0; body_len];
        body[..len - start].copy_from_slice(&bytes[start..len]);
        let mut offset = len - start;
        while offset < body.len() {
            self.socket
                .set_read_timeout(Some(remaining(self.deadline)?))?;
            let count = self.socket.read(&mut body[offset..])?;
            if count == 0 {
                return Err(Error::Invalid);
            }
            offset += count;
        }
        remaining(self.deadline)?;
        let body = if body.is_empty() {
            serde_json::Value::Null
        } else {
            starlings_protocol::parse_json(&body)?
        };
        remaining(self.deadline)?;
        Ok((self, ClientRequest { method, path, body }, token))
    }
}

fn write_authorized(
    writer: &mut impl Write,
    authority: &Authority,
    token: &ClientToken,
    scope: &Scope,
    parts: &[&[u8]],
    deadline: Instant,
) -> Result<()> {
    for part in parts {
        let mut offset = 0;
        while offset < part.len() {
            remaining(deadline)?;
            // Nonblocking write under the policy lock orders every enqueue against revocation.
            match authority.with_client_access(token, scope, || {
                writer.write(&part[offset..part.len().min(offset + 4096)])
            })? {
                Ok(0) => return Err(Error::Transport),
                Ok(n) => offset += n,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(1))
                }
                Err(_) => return Err(Error::Transport),
            }
        }
    }
    Ok(())
}

pub(crate) fn remaining(deadline: Instant) -> Result<Duration> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|d| !d.is_zero())
        .ok_or(Error::Expired)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Action, Role,
        enrollment::tests::{enrolled, setup},
    };
    use std::{
        sync::atomic::{AtomicUsize, Ordering},
        thread,
    };

    #[test]
    fn would_block_retries_partial_writes_and_stops_at_the_absolute_deadline() {
        struct Backpressure {
            blocks: usize,
            output: Vec<u8>,
        }
        impl Write for Backpressure {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                if self.blocks > 0 {
                    self.blocks -= 1;
                    return Err(std::io::ErrorKind::WouldBlock.into());
                }
                assert!(bytes.len() <= 4096);
                let count = bytes.len().min(3);
                self.output.extend_from_slice(&bytes[..count]);
                Ok(count)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let (_temp, _, mut authority, files) = setup();
        let controller = enrolled(&mut authority, &files, Role::Controller, "backpressure");
        let scope = Scope {
            model: starlings_protocol::byte_digest(b"backpressure"),
            action: Action::Infer,
        };
        authority
            .set_permissions(controller.node(), std::slice::from_ref(&scope))
            .unwrap();
        let token = authority
            .issue_client_token(controller.node(), std::slice::from_ref(&scope), 60)
            .unwrap();
        let mut writer = Backpressure {
            blocks: 2,
            output: Vec::new(),
        };
        let parts: &[&[u8]] = &[b"header", b"response"];
        write_authorized(
            &mut writer,
            &authority,
            &token,
            &scope,
            parts,
            Instant::now() + Duration::from_secs(1),
        )
        .unwrap();
        assert_eq!(writer.output, b"headerresponse");
        writer.blocks = 20;
        writer.output.clear();
        assert_eq!(
            write_authorized(
                &mut writer,
                &authority,
                &token,
                &scope,
                parts,
                Instant::now() + Duration::from_millis(10)
            ),
            Err(Error::Expired)
        );
        assert!(writer.output.is_empty());
        authority.revoke(controller.node()).unwrap();
        writer.blocks = 0;
        assert_eq!(
            write_authorized(
                &mut writer,
                &authority,
                &token,
                &scope,
                parts,
                Instant::now() + Duration::from_secs(1)
            ),
            Err(Error::Denied)
        );
        assert!(writer.output.is_empty());
    }

    fn exchange(
        authority: &Authority,
        scope: &Scope,
        token: &str,
        extra: &str,
        body: &str,
        calls: &AtomicUsize,
    ) -> (Result<()>, String) {
        let listener =
            ClientListener::bind("127.0.0.1:0".parse().unwrap(), ListenerBudget::default())
                .unwrap();
        let address = listener.local_addr().unwrap();
        let request = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: {address}\r\nAuthorization: Bearer {token}\r\nContent-Type: application/json\r\n{extra}Content-Length: {}\r\n\r\n{body}",
            body.len()
        );
        thread::scope(|s| {
            let client = s.spawn(|| {
                let mut socket = TcpStream::connect(address).unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let _ = socket.write_all(request.as_bytes());
                let mut out = String::new();
                let _ = socket.read_to_string(&mut out);
                out
            });
            let result = listener.accept().unwrap().serve(authority, scope, |req| {
                assert_eq!(req.method, "POST");
                assert_eq!(req.path, "/v1/chat/completions");
                assert_eq!(req.body["model"], "test-manifest");
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(b"{\"accepted\":true}".to_vec())
            });
            (result, client.join().unwrap())
        })
    }
    #[test]
    fn http_auth_framing_and_origin_checks_precede_dispatch() {
        let (_temp, _path, mut authority, files) = setup();
        let controller = enrolled(&mut authority, &files, Role::Controller, "http");
        let scope = Scope {
            model: starlings_protocol::byte_digest(b"test-manifest"),
            action: Action::Infer,
        };
        authority
            .set_permissions(controller.node(), std::slice::from_ref(&scope))
            .unwrap();
        let token = authority
            .issue_client_token(controller.node(), std::slice::from_ref(&scope), 60)
            .unwrap();
        let calls = AtomicUsize::new(0);
        let body = "{\"model\":\"test-manifest\"}";
        let (result, response) = exchange(&authority, &scope, &token.bearer(), "", body, &calls);
        assert!(result.is_ok());
        assert!(response.starts_with("HTTP/1.1 200 OK"));
        assert!(!response.contains(&token.bearer()));
        for extra in [
            "Origin: null\r\n",
            "Origin: https://attacker.invalid\r\n",
            "Sec-Fetch-Site: same-origin\r\n",
            "Host: attacker.invalid\r\n",
            "Content-Length: 1\r\n",
            "Transfer-Encoding: chunked\r\n",
            "Content-Encoding: gzip\r\n",
            "Expect: 100-continue\r\n",
            "Upgrade: websocket\r\n",
        ] {
            assert!(
                exchange(&authority, &scope, &token.bearer(), extra, body, &calls)
                    .0
                    .is_err()
            );
        }
        for body in ["{", "{\"model\":\"a\",\"model\":\"b\"}", "{\"model\":NaN}"] {
            assert!(
                exchange(&authority, &scope, &token.bearer(), "", body, &calls)
                    .0
                    .is_err()
            );
        }
        for extra in [
            format!("X-Padding: {}\r\n", "x".repeat(MAX_HEADERS)),
            (0..33).map(|n| format!("X-{n}: x\r\n")).collect(),
        ] {
            assert!(
                exchange(&authority, &scope, &token.bearer(), &extra, body, &calls)
                    .0
                    .is_err()
            );
        }
        assert_eq!(
            exchange(
                &authority,
                &scope,
                &token.bearer(),
                "",
                &" ".repeat(MAX_BODY + 1),
                &calls
            )
            .0,
            Err(Error::Capacity)
        );
        let other = Scope {
            model: starlings_protocol::byte_digest(b"other"),
            action: Action::Artifact,
        };
        assert_eq!(
            exchange(&authority, &other, &token.bearer(), "", body, &calls).0,
            Err(Error::Denied)
        );
        authority.revoke_client_token(&token).unwrap();
        assert_eq!(
            exchange(&authority, &scope, &token.bearer(), "", body, &calls).0,
            Err(Error::Denied)
        );
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }
    #[test]
    fn unauthorized_artifact_request_never_waits_for_body_or_dispatches() {
        let (_temp, _, authority, _) = setup();
        let budget = ListenerBudget::default();
        assert!(ClientListener::bind("0.0.0.0:0".parse().unwrap(), budget.clone()).is_err());
        let listener =
            ClientListener::bind("127.0.0.1:0".parse().unwrap(), budget.clone()).unwrap();
        let address = listener.local_addr().unwrap();
        let mut socket = TcpStream::connect(address).unwrap();
        socket.write_all(format!("POST /compute/v1/artifacts/model HTTP/1.1\r\nHost: {address}\r\nContent-Length: 999999999\r\n\r\n").as_bytes()).unwrap();
        let start = Instant::now();
        assert_eq!(
            listener.accept().unwrap().serve(
                &authority,
                &Scope {
                    model: starlings_protocol::byte_digest(b"model"),
                    action: Action::Artifact
                },
                |_| panic!("unauthenticated artifact access")
            ),
            Err(Error::Denied)
        );
        assert!(start.elapsed() < Duration::from_secs(1));
        assert_eq!(budget.usage().unwrap().connections, 0);
    }
    #[test]
    fn slow_senders_and_socket_floods_are_bounded() {
        let (_temp, _, mut authority, files) = setup();
        let controller = enrolled(&mut authority, &files, Role::Controller, "slow");
        let scope = Scope {
            model: starlings_protocol::byte_digest(b"model"),
            action: Action::Infer,
        };
        authority
            .set_permissions(controller.node(), std::slice::from_ref(&scope))
            .unwrap();
        let token = authority
            .issue_client_token(controller.node(), std::slice::from_ref(&scope), 60)
            .unwrap();
        let budget = ListenerBudget::default();
        let listener =
            ClientListener::bind("127.0.0.1:0".parse().unwrap(), budget.clone()).unwrap();
        let address = listener.local_addr().unwrap();
        let mut held = Vec::new();
        for _ in 0..4 {
            let client = TcpStream::connect(address).unwrap();
            held.push((client, listener.accept().unwrap()));
        }
        let _excess = TcpStream::connect(address).unwrap();
        assert!(matches!(listener.accept(), Err(Error::Capacity)));
        assert_eq!(budget.usage().unwrap().connections, 4);
        drop(held);
        for header_only in [true, false] {
            let mut client = TcpStream::connect(address).unwrap();
            let mut incoming = listener.accept().unwrap();
            incoming.deadline = Instant::now() + Duration::from_millis(150);
            if !header_only {
                client.write_all(format!("POST /v1/chat/completions HTTP/1.1\r\nHost: {address}\r\nAuthorization: Bearer {}\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n", token.bearer()).as_bytes()).unwrap();
            }
            thread::scope(|s| {
                s.spawn(move || {
                    for _ in 0..30 {
                        if client
                            .write_all(if header_only { b"G" } else { b" " })
                            .is_err()
                        {
                            break;
                        }
                        thread::sleep(Duration::from_millis(20));
                    }
                });
                let start = Instant::now();
                assert!(
                    incoming
                        .serve(&authority, &scope, |_| panic!("slow request dispatched"))
                        .is_err()
                );
                assert!(start.elapsed() < Duration::from_secs(1));
            });
        }
        assert_eq!(budget.usage().unwrap().connections, 0);
    }
    #[test]
    fn completed_socket_writes_do_not_prove_listener_admission() {
        let budget = ListenerBudget::default();
        let listener =
            ClientListener::bind("127.0.0.1:0".parse().unwrap(), budget.clone()).unwrap();
        let address = listener.local_addr().unwrap();
        let clients: Vec<_> = (0..5)
            .map(|_| {
                let mut socket = TcpStream::connect(address).unwrap();
                socket
                    .write_all(format!("POST /v1/chat/completions HTTP/1.1\r\nHost: {address}\r\nContent-Length: 2\r\n\r\n{{}}").as_bytes())
                    .unwrap();
                socket
            })
            .collect();
        let incoming: Vec<_> = (0..4).map(|_| listener.accept().unwrap()).collect();
        assert!(matches!(listener.accept(), Err(Error::Capacity)));
        assert_eq!(budget.usage().unwrap().connections, 4);
        drop(incoming);
        assert_eq!(budget.usage().unwrap().connections, 0);
        drop(clients);
    }

    #[test]
    fn ipv6_loopback_requires_the_exact_bound_host() {
        let (_temp, _, mut authority, files) = setup();
        let controller = enrolled(&mut authority, &files, Role::Controller, "ipv6");
        let scope = Scope {
            model: starlings_protocol::byte_digest(b"model"),
            action: Action::Infer,
        };
        authority
            .set_permissions(controller.node(), std::slice::from_ref(&scope))
            .unwrap();
        let token = authority
            .issue_client_token(controller.node(), std::slice::from_ref(&scope), 60)
            .unwrap();
        for valid in [true, false] {
            let listener =
                ClientListener::bind("[::1]:0".parse().unwrap(), ListenerBudget::default())
                    .unwrap();
            let address = listener.local_addr().unwrap();
            let host = if valid {
                address.to_string()
            } else {
                "::1".into()
            };
            let mut client = TcpStream::connect(address).unwrap();
            client.write_all(format!("GET /v1/models HTTP/1.1\r\nHost: {host}\r\nAuthorization: Bearer {}\r\nContent-Length: 0\r\n\r\n", token.bearer()).as_bytes()).unwrap();
            let result = listener
                .accept()
                .unwrap()
                .serve(&authority, &scope, |_| Ok(b"{}".to_vec()));
            assert_eq!(result, if valid { Ok(()) } else { Err(Error::Denied) });
        }
    }
}

#[cfg(test)]
mod stream_tests {
    use super::*;
    use crate::{
        Action, AdminListener, Invitation, NodeStatus, PeerListener, PrivateDir, RegistryClient,
        RegistryRequest, Role,
        enrollment::tests::{enrolled, setup},
    };
    #[test]
    fn deferred_work_has_no_policy_lock_and_idle_cancellation_closes_output() {
        use std::net::Shutdown;
        let (_temp, _, mut authority, files) = setup();
        let controller = enrolled(&mut authority, &files, Role::Controller, "deferred");
        let scope = Scope {
            model: starlings_protocol::byte_digest(b"deferred-model"),
            action: Action::Infer,
        };
        authority
            .set_permissions(controller.node(), std::slice::from_ref(&scope))
            .unwrap();
        for failure in ["none", "disconnect", "pipeline", "revoke", "drain"] {
            let token = authority
                .issue_client_token(controller.node(), std::slice::from_ref(&scope), 60)
                .unwrap();
            let budget = ListenerBudget::default();
            let listener =
                ClientListener::bind("127.0.0.1:0".parse().unwrap(), budget.clone()).unwrap();
            let address = listener.local_addr().unwrap();
            let mut client = TcpStream::connect(address).unwrap();
            client
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let body = format!("{{\"model\":\"{}\"}}", scope.model.as_str());
            client.write_all(format!("POST /v1/chat/completions HTTP/1.1\r\nHost: {address}\r\nAuthorization: Bearer {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", token.bearer(), body.len()).as_bytes()).unwrap();
            let (_, response) = listener
                .accept()
                .unwrap()
                .prepare(&authority, &scope, |req| {
                    if req.method == "POST"
                        && req.path == "/v1/chat/completions"
                        && req.body["model"] == scope.model.as_str()
                    {
                        Ok(())
                    } else {
                        Err(Error::Denied)
                    }
                })
                .unwrap();
            assert_eq!(budget.usage().unwrap().connections, 0);
            let watch = response.watch().unwrap();
            watch.check().unwrap();
            // No response headers before the application chooses JSON or SSE.
            client.set_nonblocking(true).unwrap();
            assert_eq!(
                client.peek(&mut [0]).unwrap_err().kind(),
                std::io::ErrorKind::WouldBlock
            );
            client.set_nonblocking(false).unwrap();
            match failure {
                "disconnect" => client.shutdown(Shutdown::Write).unwrap(),
                "pipeline" => client.write_all(b"GET /second HTTP/1.1\r\n").unwrap(),
                "revoke" => authority.revoke_client_token(&token).unwrap(),
                "drain" => {
                    authority.drain().unwrap();
                }
                _ => (),
            }
            if failure != "none" {
                let deadline = Instant::now() + Duration::from_secs(1);
                while watch.check().is_ok() {
                    assert!(Instant::now() < deadline);
                    std::thread::sleep(Duration::from_millis(1));
                }
                // Even a later successful backend result cannot be published.
                assert!(response.json(b"{\"late\":true}").is_err());
                let mut bytes = Vec::new();
                let _ = client.read_to_end(&mut bytes);
                assert!(bytes.is_empty());
            } else {
                // Hold the global policy lock: response output and liveness must not use it.
                let body = serde_json::to_vec(&serde_json::json!({"content":"x".repeat(100_000)}))
                    .unwrap();
                std::thread::scope(|threads| {
                    let guard = authority.access.lock().unwrap();
                    let reader = threads.spawn(move || {
                        let mut bytes = Vec::new();
                        client.read_to_end(&mut bytes).unwrap();
                        bytes
                    });
                    let (tx, rx) = std::sync::mpsc::sync_channel(1);
                    let body = &body;
                    let watch = &watch;
                    let work = threads.spawn(move || {
                        watch.check().unwrap();
                        tx.send(response.json(body)).unwrap();
                    });
                    let result = rx.recv_timeout(Duration::from_secs(2));
                    drop(guard);
                    work.join().unwrap();
                    result.unwrap().unwrap();
                    let bytes = reader.join().unwrap();
                    let start = bytes.windows(4).position(|v| v == b"\r\n\r\n").unwrap() + 4;
                    assert_eq!(&bytes[start..], body);
                });
            }
            assert!(watch.check().is_err());
        }
    }

    #[test]
    fn six_authenticated_streams_leave_admin_and_heartbeats_available() {
        let (_temp, path, mut authority, files) = setup();
        let controller = enrolled(
            &mut authority,
            &files,
            Role::Controller,
            "stream-controller",
        );
        let worker = enrolled(&mut authority, &files, Role::Worker, "stream-worker");
        let scope = Scope {
            model: starlings_protocol::byte_digest(b"stream-model"),
            action: Action::Infer,
        };
        authority
            .set_permissions(controller.node(), std::slice::from_ref(&scope))
            .unwrap();
        let token = authority
            .issue_client_token(controller.node(), std::slice::from_ref(&scope), 60)
            .unwrap();
        let client_budget = ListenerBudget::default();
        let listener =
            ClientListener::bind("127.0.0.1:0".parse().unwrap(), client_budget.clone()).unwrap();
        let address = listener.local_addr().unwrap();
        let peer =
            PeerListener::bind("127.0.0.1:0".parse().unwrap(), ListenerBudget::default()).unwrap();
        authority
            .invite(
                &files,
                "stream-pin",
                peer.local_addr().unwrap(),
                Role::Worker,
            )
            .unwrap();
        authority.start_registry().unwrap();
        let profile = path.parent().unwrap().join("stream-profile");
        PrivateDir::ensure(&profile).unwrap();
        let mut registry = RegistryClient::open(
            &profile,
            worker,
            Invitation::load(&files, "stream-pin").unwrap(),
        )
        .unwrap();
        let admin_path = path.parent().unwrap().join("stream-admin");
        PrivateDir::ensure(&admin_path).unwrap();
        let admin = AdminListener::bind(&admin_path, ListenerBudget::default()).unwrap();
        std::thread::scope(|threads| {
            let heartbeat = threads.spawn(|| {
                for _ in 0..3 {
                    let until = Instant::now() + Duration::from_secs(5);
                    let incoming = loop {
                        if let Some(incoming) = peer.try_accept_registry(&authority).unwrap() {
                            break incoming;
                        }
                        assert!(Instant::now() < until, "heartbeat was starved");
                        std::thread::sleep(Duration::from_millis(1));
                    };
                    match incoming
                        .registry(authority.enrollment_reader().unwrap())
                        .unwrap()
                    {
                        RegistryRequest::Heartbeat(pending) => pending.respond().unwrap(),
                        _ => panic!("unexpected enrollment"),
                    }
                }
            });
            registry.heartbeat(None).unwrap();
            registry.heartbeat(None).unwrap();
            let mut clients = Vec::new();
            let mut streams = Vec::new();
            for _ in 0..6 {
                let mut client = TcpStream::connect(address).unwrap();
                client
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                client.write_all(format!("POST /v1/chat/completions HTTP/1.1\r\nHost: {address}\r\nAuthorization: Bearer {}\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{{}}", token.bearer()).as_bytes()).unwrap();
                let (_, stream) = listener
                    .accept()
                    .unwrap()
                    .stream(&authority, &scope, |request| {
                        assert_eq!(request.path, "/v1/chat/completions");
                        Ok(())
                    })
                    .unwrap();
                clients.push(client);
                streams.push(stream);
            }
            assert_eq!(client_budget.usage().unwrap().connections, 0);
            authority.maintain(0).unwrap();
            for _ in 0..20 {
                for stream in &mut streams {
                    stream
                        .event(br#"{"choices":[{"delta":{"content":"synthetic"}}]}"#)
                        .unwrap();
                }
            }
            registry.heartbeat(None).unwrap();
            assert_eq!(registry.fence().status().unwrap().1, NodeStatus::Online);
            heartbeat.join().unwrap();
            for command in ["status", "stop"] {
                let response = threads.spawn(|| {
                    admin.accept().unwrap().respond(|value: serde_json::Value| {
                        if value["command"] == "stop" {
                            authority.drain().unwrap();
                        }
                        serde_json::json!({"ok":true})
                    })
                });
                let value: serde_json::Value =
                    crate::admin_call(&admin_path, &serde_json::json!({"command":command}))
                        .unwrap();
                assert_eq!(value["ok"], true);
                response.join().unwrap().unwrap();
            }
            for stream in &mut streams {
                assert!(stream.event(br#"{"late":true}"#).is_err());
            }
            drop(streams);
            for mut client in clients {
                let mut response = String::new();
                client.read_to_string(&mut response).unwrap();
                assert!(response.contains("Content-Type: text/event-stream"));
                assert_eq!(response.matches("data: ").count(), 20);
                assert!(!response.contains("late"));
            }
        });
    }
}
