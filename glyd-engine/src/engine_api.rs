//! The engine as a front end sees it: requests in, token streams out. Every API front end (OpenAI's completions, Anthropic's Messages, OpenAI's Responses, Ollama's, and later gRPC and a
//! token-level API for other routers) talks to the engine through this interface and nothing else: `submit` a request (a prompt as tokens, what the reply is asked to be, its images), read its
//! `Stream` of events (a token, its log-probabilities, the end), `cancel` it, and ask for the engine's `stats`. Nothing of HTTP is in it; nothing of a model is in it.
//!
//! The scheduler (src/sched.rs) is behind it: a request is a `Submit` on the engine's queue, and its events come back on the stream's channel. The interface is the one `Remote` has always been (the
//! handlers sent sequences through it); this names it, so that a front end can be written, and tested, against any `Engine`, a fake one included (the load test of the HTTP front end runs against a
//! scheduler over a CPU fake model).

use crate::sched::{Ev, Reply};
use crate::server::Remote;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::Arc;
use std::time::Duration;

/// What a front end asks of the engine for one sequence.
#[derive(Clone)]
pub struct Request {
    /// The prompt, as token ids.
    pub prompt: Vec<u32>,
    /// What the reply is asked to be: sampling, length, log-probabilities, the tokens that end it.
    pub reply: Reply,
    /// The prompt's images, in order (their places in it are their tokens), for a model that reads images.
    pub images: Vec<Arc<crate::vision::Image>>,
}

impl Request {
    pub fn new(prompt: Vec<u32>, reply: Reply) -> Request {
        Request { prompt, reply, images: Vec::new() }
    }
}

/// Why a request was not taken: said in words for the client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rejected(pub String);

impl std::fmt::Display for Rejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The end of a request asked for by whoever holds it (a client that went, a request that has all it wants): the sequence stops at its next step and its pages are the cache's again.
#[derive(Clone, Default)]
pub struct Cancel(pub(crate) Arc<AtomicBool>);

impl Cancel {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// A request's reply as it is made: events until `Ev::End`. Dropping it, or cancelling, ends the sequence.
pub struct Stream {
    events: Receiver<Ev>,
    cancel: Cancel,
}

impl Stream {
    pub(crate) fn new(events: Receiver<Ev>, cancel: Cancel) -> Stream {
        Stream { events, cancel }
    }

    /// The next event, waiting up to `wait`: Ok(None) where there is none yet, Err where the engine dropped the sequence (it ended without an `End`).
    pub fn next(&self, wait: Duration) -> Result<Option<Ev>, Rejected> {
        match self.events.recv_timeout(wait) {
            Ok(ev) => Ok(Some(ev)),
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => Err(Rejected("the engine dropped the request".into())),
        }
    }

    /// The handle that ends the request (it may be given to another thread).
    pub fn canceller(&self) -> Cancel {
        self.cancel.clone()
    }

    pub fn cancel(&self) {
        self.cancel.cancel();
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // (a reply nobody reads is not run on)
        self.cancel.cancel();
    }
}

/// What the engine says of itself, for health checks, metrics and a router's choice of where to send a request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Sequences the scheduler is running.
    pub running: u64,
    /// Sequences waiting for room.
    pub waiting: u64,
    /// The longest a request's prompt and reply may be together, in tokens.
    pub window: u64,
}

/// The engine behind a front end.
pub trait Engine: Send + Sync {
    /// Takes a request: its events come on the stream.
    fn submit(&self, request: Request) -> Result<Stream, Rejected>;
    fn stats(&self) -> Stats;
}

impl Engine for Remote {
    fn submit(&self, request: Request) -> Result<Stream, Rejected> {
        let h = self.start_with(&request.prompt, request.images, &request.reply).map_err(|e| Rejected(e.0))?;
        Ok(Stream::new(h.events, Cancel(h.abort)))
    }

    fn stats(&self) -> Stats {
        use std::sync::atomic::Ordering::Relaxed;
        Stats { running: crate::sched::SHARED[5].load(Relaxed), waiting: crate::sched::SHARED[6].load(Relaxed), window: self.window as u64 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sched::Finish;
    use crate::server::Job;
    use std::sync::mpsc::channel;

    /// A request goes to the engine's queue with its cancel flag, and its events come back on the stream.
    #[test]
    fn a_request_is_submitted_streamed_and_cancelled() {
        let (tx, rx) = channel();
        let remote = Remote { tx, window: 64 };
        let engine: &dyn Engine = &remote;
        let reply = Reply { max_new: 3, ..Reply::default() };
        let stream = engine.submit(Request::new(vec![1, 2, 3], reply)).unwrap();
        let Job::Seq(sub) = rx.recv().unwrap() else { panic!("a sequence") };
        assert_eq!((sub.prompt.clone(), sub.reply.max_new), (vec![1, 2, 3], 3));
        sub.out.send(Ev::Token(7)).unwrap();
        sub.out.send(Ev::End(Ok(Finish::Length))).unwrap();
        assert!(matches!(stream.next(Duration::from_secs(1)).unwrap(), Some(Ev::Token(7))));
        assert!(matches!(stream.next(Duration::from_secs(1)).unwrap(), Some(Ev::End(Ok(Finish::Length)))));
        assert!(matches!(stream.next(Duration::from_millis(10)), Ok(None)), "nothing more yet");
        // the cancel handle ends the sequence: the scheduler reads the flag it was given
        let canceller = stream.canceller();
        assert!(!sub.abort.load(Ordering::Relaxed) && !canceller.is_cancelled());
        canceller.cancel();
        assert!(sub.abort.load(Ordering::Relaxed));
        // a stream dropped is a reply nobody reads: cancelled too
        let (tx2, rx2) = channel();
        let remote2 = Remote { tx: tx2, window: 64 };
        let s2 = remote2.submit(Request::new(vec![1], Reply::default())).unwrap();
        let Job::Seq(sub2) = rx2.recv().unwrap() else { panic!("a sequence") };
        drop(s2);
        assert!(sub2.abort.load(Ordering::Relaxed));
    }

    #[test]
    fn a_request_the_engine_cannot_take_is_rejected_in_words() {
        let (tx, rx) = channel();
        let remote = Remote { tx, window: 4 };
        assert_eq!(remote.submit(Request::new(vec![], Reply::default())).err(), Some(Rejected("an empty prompt".into())));
        let long = remote.submit(Request::new(vec![1; 8], Reply::default())).err().unwrap();
        assert!(long.0.contains("8 tokens, where the context holds 4"), "{long}");
        drop(rx);
        let gone = remote.submit(Request::new(vec![1], Reply::default())).err().unwrap();
        assert!(gone.0.contains("the engine is not running"), "{gone}");
        assert_eq!(remote.stats().window, 4);
    }
}
