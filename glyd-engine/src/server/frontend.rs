//! The server's front end: the connections. One tokio runtime accepts them, reads each request (the head within its bounds and its time, then the body), hands it to `dispatch`, and writes
//! the answer to the client's socket as the handler's thread writes it, so a connection that waits (for a request, for a reply, for a slow client to read) costs a task and not a thread.
//!
//! What it keeps from the server it replaces: every bound and every time of the request's reading (`read_head`, MAX_HEAD, HEAD_TIMEOUT, BODY_TIMEOUT, WRITE_TIMEOUT), the order of the checks
//! of `dispatch` (Host and Origin, preflight, key, drain), and the handlers: each answers through a `Conn`, here a `Piped`. What it changes: there is no cap on connections (what the host
//! allows is the limit, and a connection that sends nothing is closed by its time, not refused for the sake of a count); a connection whose client asks for `Connection: keep-alive` and that is
//! answered with a `Content-Length` is kept for the next request (until it is idle for IDLE); a client that goes while its answer is being made ends the answer's writes (`Piped::gone`), which is what cancels its sequence; and a
//! client that reads slowly holds its handler at the pipe's depth, not in memory.

use super::*;
use crate::conn::Piped;
use std::io::Cursor;
use std::sync::atomic::AtomicU64;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::time::timeout;

/// How long a connection that has had an answer may wait for its next request before it is closed.
pub const IDLE: Duration = Duration::from_secs(60);

/// Requests answered by the process so far: a reply's id names one of them (/v1/glyd/requests/ID finds it).
static REQUESTS: AtomicU64 = AtomicU64::new(0);

/// What the server holds of requests' bodies at once, in bytes: a body waits for its share before it is read (and holds it until its answer ends), so that connections without a cap do not
/// hold more memory than this between them (the server this replaces held at most 64 connections' worth).
const BODY_BUDGET_BYTES: usize = 1 << 30;
static BODY_BUDGET: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(BODY_BUDGET_BYTES);

/// The most connections open at once: far past what a load needs, and under what a process's files allow; the next is answered 503 and let go. (There is no cap in the sense of the 64 it had: a
/// connection that sends nothing is closed by its time, so these are connections that are doing something.)
pub const MAX_OPEN: usize = 50_000;
static OPEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// How much a client may send while its answer is made (the beginning of its next request): past it the connection is let go.
const PIPELINE_CAP: usize = MAX_HEAD;

/// The listen queue's length asked for: connections that arrive together wait in it until they are accepted (the system holds it to its own limit: Linux's 4096, macOS's 128).
const BACKLOG: u32 = 4096;

/// Starts the front end on `addr` (host:port): binds and listens (an address in use is an error here, said as it was), and runs a thread that owns the runtime for the rest of the process.
/// `make` builds what a request is answered under from the port that was bound (a port 0 is whichever the system gave).
pub(crate) fn serve(addr: &str, make: impl FnOnce(u16) -> Ctx) -> std::io::Result<std::net::SocketAddr> {
    use std::net::ToSocketAddrs;
    let want = addr.to_socket_addrs()?.next().ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "no address"))?;
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).thread_name("glyd-http").enable_io().enable_time().build()?;
    let listener = rt.block_on(async {
        let sock = if want.is_ipv4() { tokio::net::TcpSocket::new_v4()? } else { tokio::net::TcpSocket::new_v6()? };
        #[cfg(unix)]
        sock.set_reuseaddr(true)?;
        sock.bind(want)?;
        sock.listen(BACKLOG)
    })?;
    let bound = listener.local_addr()?;
    let ctx = make(bound.port());
    std::thread::Builder::new().name("glyd-http-accept".into()).spawn(move || {
        rt.block_on(async move {
            loop {
                match listener.accept().await {
                    Ok((mut sock, _)) => {
                        if OPEN.fetch_add(1, Ordering::SeqCst) >= MAX_OPEN {
                            OPEN.fetch_sub(1, Ordering::SeqCst);
                            crate::ops::refused(503);
                            let out = response_bytes(503, "application/json", "Retry-After: 1\r\n", error_body("too many connections", 503).as_bytes());
                            let _ = timeout(Duration::from_secs(1), sock.write_all(&out)).await;
                            continue;
                        }
                        let ctx = ctx.clone();
                        tokio::spawn(async move {
                            connection(sock, ctx).await;
                            OPEN.fetch_sub(1, Ordering::SeqCst);
                        });
                    }
                    // (out of file descriptors, or a connection reset before it was taken: the next one is tried after a moment)
                    Err(_) => tokio::time::sleep(Duration::from_millis(10)).await,
                }
            }
        })
    })?;
    Ok(bound)
}

/// The first `\r\n\r\n` in `b`: where the head of a request ends.
fn head_end(b: &[u8]) -> Option<usize> {
    b.windows(4).position(|w| w == b"\r\n\r\n").map(|p| p + 4)
}

/// More bytes of the request into `buf`, within `until`; None where the client closed before sending any (a clean end), an error where it was too slow or went mid-request.
async fn read_more(rd: &mut OwnedReadHalf, buf: &mut Vec<u8>, until: tokio::time::Instant) -> std::result::Result<Option<usize>, Rejected> {
    let mut chunk = [0u8; 16 << 10];
    match tokio::time::timeout_at(until, rd.read(&mut chunk)).await {
        Err(_) => Err((408, "the request was not sent in time".to_string())),
        Ok(Err(e)) => Err((400, e.to_string())),
        Ok(Ok(0)) => Ok(None),
        Ok(Ok(n)) => {
            buf.extend_from_slice(&chunk[..n]);
            Ok(Some(n))
        }
    }
}

/// An answer written whole (a request refused before it reached `dispatch`).
async fn say(wr: &mut OwnedWriteHalf, code: u16, why: &str) {
    let out = response_bytes(code, "application/json", "", error_body(why, code).as_bytes());
    let _ = timeout(WRITE_TIMEOUT, wr.write_all(&out)).await;
}

/// After an answer that was sent before the request was read to its end: the end of what this side sends, and what the client still sends read for a few seconds, so that the answer is not lost
/// to a reset while the client is writing.
async fn drain(rd: &mut OwnedReadHalf, wr: &mut OwnedWriteHalf) {
    let _ = wr.shutdown().await;
    let end = tokio::time::Instant::now() + Duration::from_secs(3);
    let mut sink = [0u8; 16 << 10];
    while let Ok(Ok(n)) = tokio::time::timeout_at(end, rd.read(&mut sink)).await {
        if n == 0 {
            break;
        }
    }
}

/// What the head of an answer says that decides whether the connection may carry another request.
struct Framing {
    /// The body's length where the answer says it (`Content-Length`), and it is not a chunked one.
    length: Option<usize>,
    /// The head, with its `Connection: close` made `keep-alive` where the connection is kept.
    head: Vec<u8>,
}

/// The head of an answer (up to and including its blank line) read for how its body is framed; `keep` says the request wanted the connection kept.
fn framing(head: &[u8], keep: bool, head_request: bool) -> Framing {
    let text = String::from_utf8_lossy(head);
    let mut length = None;
    let mut chunked = false;
    for l in text.lines().skip(1) {
        let low = l.to_ascii_lowercase();
        if let Some(v) = low.strip_prefix("content-length:") {
            length = v.trim().parse().ok();
        } else if low.starts_with("transfer-encoding:") && low.contains("chunked") {
            chunked = true;
        }
    }
    let kept = keep && !chunked && length.is_some();
    // (an answer to HEAD says a length and sends none)
    let length = if head_request { Some(0) } else { length.filter(|_| !chunked) };
    let head = if kept { text.replacen("Connection: close", "Connection: keep-alive", 1).into_bytes() } else { head.to_vec() };
    Framing { length: length.filter(|_| kept), head }
}

/// One connection, for as many requests as it keeps.
async fn connection(sock: tokio::net::TcpStream, ctx: Ctx) {
    let _ = sock.set_nodelay(true);
    let (mut rd, mut wr) = sock.into_split();
    let mut buf: Vec<u8> = Vec::new();
    let mut first = true;
    loop {
        // the head: a first request has HEAD_TIMEOUT from the connection's opening, and a later one waits for IDLE for its first byte and has HEAD_TIMEOUT from it (what the client sent while
        // the last answer was written is its beginning already)
        let mut started = !buf.is_empty() || first;
        let mut until = tokio::time::Instant::now() + if started { ctx.times.head } else { ctx.times.idle };
        let end = loop {
            if let Some(e) = head_end(&buf) {
                break e;
            }
            if buf.len() > MAX_HEAD {
                crate::ops::refused(431);
                say(&mut wr, 431, &format!("the request's header block is over {} KiB", MAX_HEAD >> 10)).await;
                return drain(&mut rd, &mut wr).await;
            }
            match read_more(&mut rd, &mut buf, until).await {
                Ok(Some(_)) if !started => {
                    started = true;
                    until = tokio::time::Instant::now() + ctx.times.head;
                }
                Ok(Some(_)) => {}
                Ok(None) => return,
                Err((code, why)) => {
                    // (an idle connection that is closed after its time is not answered: nothing was asked)
                    if started {
                        crate::ops::refused(code);
                        say(&mut wr, code, &why).await;
                    }
                    return;
                }
            }
        };
        let head_bytes: Vec<u8> = buf.drain(..end).collect();
        let h = match read_head(&mut Cursor::new(&head_bytes[..])) {
            Ok(Some(h)) => h,
            Ok(None) => return,
            Err((code, why)) => {
                crate::ops::refused(code);
                say(&mut wr, code, &why).await;
                return drain(&mut rd, &mut wr).await;
            }
        };
        if h.len > MAX_BODY {
            crate::ops::refused(413);
            say(&mut wr, 413, &format!("the request's body is over {} MiB", MAX_BODY >> 20)).await;
            return drain(&mut rd, &mut wr).await;
        }
        let lower = String::from_utf8_lossy(&head_bytes).to_ascii_lowercase();
        // (a connection is kept only for a client that asks for it, as the browsers, requests, httpx and the OpenAI SDKs do: a client that sends no word of it may read to the end of the connection
        // for its answer, as the server this replaces let every client do)
        let wants_close = lower.contains("\r\nconnection: close") || !lower.contains("\r\nconnection: keep-alive");
        if h.expect {
            let _ = timeout(WRITE_TIMEOUT, wr.write_all(b"HTTP/1.1 100 Continue\r\n\r\n")).await;
        }
        // the body: its share of what the server holds of bodies first (held until the answer ends), then within the time of the head's end
        let until = tokio::time::Instant::now() + ctx.times.body;
        let _share = if h.len > 0 {
            match tokio::time::timeout_at(until, BODY_BUDGET.acquire_many(h.len as u32)).await {
                Ok(Ok(p)) => Some(p),
                _ => {
                    crate::ops::refused(503);
                    let out = response_bytes(
                        503,
                        "application/json",
                        "Retry-After: 1\r\n",
                        error_body("the server holds as many request bodies as it takes: try again", 503).as_bytes(),
                    );
                    let _ = timeout(WRITE_TIMEOUT, wr.write_all(&out)).await;
                    return;
                }
            }
        } else {
            None
        };
        while buf.len() < h.len {
            match read_more(&mut rd, &mut buf, until).await {
                Ok(Some(_)) => {}
                Ok(None) => return,
                Err((code, why)) => {
                    crate::ops::refused(code);
                    say(&mut wr, code, &why).await;
                    return;
                }
            }
        }
        let body: Vec<u8> = buf.drain(..h.len).collect();
        let head_request = h.method == "HEAD";
        let req = req_of(h, body);
        let id = REQUESTS.fetch_add(1, Ordering::SeqCst) + 1;
        let (piped, mut rx, gone) = Piped::new();
        let ctx2 = ctx.clone();
        // (what answers runs on a thread: it writes with a blocking send, and the handlers of an answer that is generated wait for its pieces)
        tokio::task::spawn_blocking(move || dispatch(Box::new(piped), req, &ctx2, id));
        // the answer, as it is written, while the client is watched for going (what it sends meanwhile is the next request's beginning)
        let mut state = Answer::Waiting(Vec::new());
        let mut body_left: Option<usize> = None;
        let mut tmp = [0u8; 4096];
        let mut keep = false;
        let mut client_ok = true;
        loop {
            tokio::select! {
                chunk = rx.recv() => {
                    let Some(chunk) = chunk else { break };
                    let out: Vec<u8> = match &mut state {
                        Answer::Waiting(pending) => {
                            pending.extend_from_slice(&chunk);
                            let Some(e) = head_end(pending) else { continue };
                            let all = std::mem::take(pending);
                            let f = framing(&all[..e], !wants_close, head_request);
                            keep = f.length.is_some();
                            body_left = f.length.map(|l| l.saturating_sub(all.len() - e));
                            let mut o = f.head;
                            if !head_request {
                                o.extend_from_slice(&all[e..]);
                            }
                            state = Answer::Sent;
                            o
                        }
                        Answer::Sent => {
                            if let Some(l) = body_left.as_mut() {
                                *l = l.saturating_sub(chunk.len());
                            }
                            chunk
                        }
                    };
                    if timeout(WRITE_TIMEOUT, wr.write_all(&out)).await.map_or(true, |r| r.is_err()) {
                        client_ok = false;
                        break;
                    }
                    if body_left == Some(0) {
                        break;
                    }
                }
                n = rd.read(&mut tmp) => match n {
                    Ok(0) | Err(_) => {
                        // the client closed its end: nothing more is written to it, and what answers it is told (its writes end, and its sequence is cancelled)
                        client_ok = false;
                        break;
                    }
                    Ok(n) => {
                        buf.extend_from_slice(&tmp[..n]);
                        if buf.len() > PIPELINE_CAP {
                            // (more than the beginning of a next request, sent while an answer is made: the connection is let go, and what answers it is told)
                            client_ok = false;
                            break;
                        }
                    }
                }
            }
        }
        if !client_ok {
            gone.store(true, Ordering::Relaxed);
            return;
        }
        drop(rx);
        if !(keep && body_left == Some(0)) {
            let _ = wr.shutdown().await;
            return;
        }
        first = false;
    }
}

enum Answer {
    /// The answer's bytes so far, before its head is whole.
    Waiting(Vec<u8>),
    Sent,
}
