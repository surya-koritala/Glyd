//! What a request's answer is written to: a `Conn`. The handlers of every API (OpenAI's completions, Anthropic's Messages, OpenAI's Responses, Ollama's) write the HTTP answer to one, and
//! ask it whether the client has gone; they do not know whether it is a socket they own (`TcpStream`, the thread-per-connection server of before, kept for the programs that serve with one
//! thread each) or the pipe the async front end (src/server/frontend.rs) pumps to the client's socket (`Piped`).

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc::Sender;

/// Where an answer is written, and whether anyone is still there to read it.
pub trait Conn: Write + Send {
    /// Whether the client has closed its end (a request that waited in line for a reply nobody reads).
    fn gone(&self) -> bool;
}

impl Conn for std::net::TcpStream {
    fn gone(&self) -> bool {
        if self.set_nonblocking(true).is_err() {
            return true;
        }
        let mut b = [0u8; 1];
        let gone = match self.peek(&mut b) {
            Ok(0) => true,
            Ok(_) => false,
            Err(e) => e.kind() != io::ErrorKind::WouldBlock,
        };
        let _ = self.set_nonblocking(false);
        gone
    }
}

/// How many chunks an answer may have written that the client has not taken: past it a write waits (a client that reads slowly holds its request's thread, which is what keeps the engine's
/// output from piling up in memory), and the front end's own write timeout ends a client that reads nothing.
pub const PIPE_DEPTH: usize = 64;

/// The pipe from a handler's thread to the front end's task that writes to the client's socket: what is written goes through in order, a write waits where the pipe is full, and ends with an
/// error once the client is gone, which is how a request is cancelled when its client disconnects.
pub struct Piped {
    tx: Sender<Vec<u8>>,
    gone: Arc<AtomicBool>,
}

impl Piped {
    /// A pipe, and the other end: the chunks written, and the flag the pumping side sets when the client has gone.
    pub fn new() -> (Piped, tokio::sync::mpsc::Receiver<Vec<u8>>, Arc<AtomicBool>) {
        let (tx, rx) = tokio::sync::mpsc::channel(PIPE_DEPTH);
        let gone = Arc::new(AtomicBool::new(false));
        (Piped { tx, gone: gone.clone() }, rx, gone)
    }
}

impl Write for Piped {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.gone.load(Ordering::Relaxed) {
            return Err(io::ErrorKind::BrokenPipe.into());
        }
        if buf.is_empty() {
            return Ok(0);
        }
        // (called from a thread of its own, never from the async runtime: this waits for room)
        self.tx.blocking_send(buf.to_vec()).map_err(|_| io::Error::from(io::ErrorKind::BrokenPipe))?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Conn for Piped {
    fn gone(&self) -> bool {
        self.gone.load(Ordering::Relaxed) || self.tx.is_closed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn what_is_written_comes_out_in_order_and_a_closed_end_is_a_gone_client() {
        let (mut p, mut rx, gone) = Piped::new();
        p.write_all(b"one").unwrap();
        p.write_all(b"two").unwrap();
        assert!(!p.gone());
        assert_eq!((rx.blocking_recv().unwrap(), rx.blocking_recv().unwrap()), (b"one".to_vec(), b"two".to_vec()));
        gone.store(true, Ordering::Relaxed);
        assert!(p.gone());
        assert_eq!(p.write(b"x").unwrap_err().kind(), io::ErrorKind::BrokenPipe);
        let (mut q, rx, _) = Piped::new();
        drop(rx);
        assert!(q.gone());
        assert_eq!(q.write(b"x").unwrap_err().kind(), io::ErrorKind::BrokenPipe);
    }

    #[test]
    fn a_write_waits_where_the_client_is_slow_and_goes_on_when_it_reads() {
        let (mut p, mut rx, _) = Piped::new();
        let t = std::thread::spawn(move || {
            for i in 0..(PIPE_DEPTH * 3) {
                p.write_all(format!("{i}").as_bytes()).unwrap();
            }
        });
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(!t.is_finished(), "the writer is held where the pipe is full");
        let mut n = 0;
        while rx.blocking_recv().is_some() {
            n += 1;
            if n == PIPE_DEPTH * 3 {
                break;
            }
        }
        t.join().unwrap();
        assert_eq!(n, PIPE_DEPTH * 3);
    }
}
