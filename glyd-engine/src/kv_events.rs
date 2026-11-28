//! KV cache events (R4c, #142): the blocks a served model's cache holds by name, as they are stored and taken away, with their tokens, for a router that sends a request
//! to the server that holds its start (D2, #203). The scheduler writes one batch a step it has any in; `GET /v1/glyd/kv/events?after=N` reads them.
//!
//! Off unless the server's operator turns them on (GLYD_ENGINE_KV_EVENTS=1, `Context::kv_events`): an event carries a prompt's tokens, and whoever reads them reads every
//! caller's prompts. The schema below is part of the public interface: a change to it is a breaking change.
//!
//! ```json
//! {"next": 43, "oldest": 1, "gap": false, "batches": [
//!   {"seq": 42, "ts": 1795366128.125, "events": [
//!     {"type": "stored", "blocks": ["<32 hex digits>", ...], "parent": "<32 hex digits>" or null, "tokens": [...], "block_size": 16, "medium": "gpu"},
//!     {"type": "removed", "blocks": ["<32 hex digits>", ...], "medium": "gpu"},
//!     {"type": "cleared"}]}]}
//! ```
//!
//! A block's name is `kv::block_hash` (the first 16 bytes of SHA-256 over its parent's name and its tokens, a request's chain beginning at its scope's root), as 32
//! hexadecimal digits; `tokens` are the stored blocks' tokens, `block_size` of them each, in order; `parent` is the name of the block before the first stored one (null:
//! the first stored is its sequence's first block). The first batch (seq 1) is `cleared`: the cache holds nothing when the server starts. The log keeps the last 1,024
//! batches; a reader whose `after` is older than the oldest kept gets `gap: true`, and should take what it knew as cleared.

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};
use std::time::Duration;

/// The batches the log keeps.
const KEEP: usize = 1024;

/// What happened to the cache's named blocks.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// Blocks found by name from now on: their names, the name of the block before the first (None: a sequence's first block) and their tokens.
    Stored { blocks: Vec<u128>, parent: Option<u128>, tokens: Vec<u32> },
    /// Blocks no longer found by these names.
    Removed { blocks: Vec<u128> },
    /// Nothing is found by name.
    Cleared,
}

/// One step's events.
#[derive(Clone, Debug, PartialEq)]
pub struct Batch {
    pub seq: u64,
    /// Seconds since 1970.
    pub ts: f64,
    pub events: Vec<Event>,
}

/// The last batches, numbered from 1.
pub struct Log {
    inner: Mutex<VecDeque<Batch>>,
    more: Condvar,
}

impl Default for Log {
    fn default() -> Self {
        Self::new()
    }
}

impl Log {
    /// A log whose first batch says the cache holds nothing.
    pub fn new() -> Log {
        let first = Batch { seq: 1, ts: now(), events: vec![Event::Cleared] };
        Log { inner: Mutex::new(VecDeque::from([first])), more: Condvar::new() }
    }

    /// A step's events, as the next batch (none: no batch).
    pub fn push(&self, events: Vec<Event>) {
        if events.is_empty() {
            return;
        }
        let mut q = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let seq = q.back().map_or(1, |b| b.seq + 1);
        q.push_back(Batch { seq, ts: now(), events });
        while q.len() > KEEP {
            q.pop_front();
        }
        drop(q);
        self.more.notify_all();
    }

    /// The batches after batch `after`, waiting up to `wait` for one where there is none yet; whether some after `after` are no longer kept; and the newest's number.
    pub fn since(&self, after: u64, wait: Duration) -> (Vec<Batch>, bool, u64) {
        let q = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let newest = |q: &VecDeque<Batch>| q.back().map_or(0, |b| b.seq);
        let q = if newest(&q) <= after && !wait.is_zero() {
            self.more.wait_timeout_while(q, wait, |q| newest(q) <= after).unwrap_or_else(|e| e.into_inner()).0
        } else {
            q
        };
        let oldest = q.front().map_or(1, |b| b.seq);
        (q.iter().filter(|b| b.seq > after).cloned().collect(), after + 1 < oldest, newest(&q))
    }

    /// `since`'s answer as the endpoint gives it (the schema above).
    pub fn json(&self, after: u64, wait: Duration) -> String {
        let (batches, gap, next) = self.since(after, wait);
        let oldest = self.inner.lock().unwrap_or_else(|e| e.into_inner()).front().map_or(1, |b| b.seq);
        let hex = |k: &u128| format!("{k:032x}");
        let event = |e: &Event| match e {
            Event::Stored { blocks, parent, tokens } => serde_json::json!({
                "type": "stored", "blocks": blocks.iter().map(hex).collect::<Vec<_>>(), "parent": parent.as_ref().map(hex),
                "tokens": tokens, "block_size": crate::dynamic::pages::BLOCK, "medium": "gpu",
            }),
            Event::Removed { blocks } => serde_json::json!({ "type": "removed", "blocks": blocks.iter().map(hex).collect::<Vec<_>>(), "medium": "gpu" }),
            Event::Cleared => serde_json::json!({ "type": "cleared" }),
        };
        let batches: Vec<serde_json::Value> =
            batches.iter().map(|b| serde_json::json!({ "seq": b.seq, "ts": b.ts, "events": b.events.iter().map(event).collect::<Vec<_>>() })).collect();
        serde_json::json!({ "next": next, "oldest": oldest, "gap": gap, "batches": batches }).to_string()
    }
}

fn now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first batch says the cache is empty; a step's events are the next batch, none no batch; a reader gets what is after its number, and the newest's number.
    #[test]
    fn batches_are_numbered_from_a_cleared_cache() {
        let log = Log::new();
        let (b, gap, next) = log.since(0, Duration::ZERO);
        assert_eq!((b.len(), b[0].seq, &b[0].events[..], gap, next), (1, 1, &[Event::Cleared][..], false, 1));
        log.push(Vec::new());
        log.push(vec![Event::Stored { blocks: vec![7], parent: None, tokens: (0..16).collect() }]);
        log.push(vec![Event::Removed { blocks: vec![7] }]);
        let (b, gap, next) = log.since(1, Duration::ZERO);
        assert_eq!((b.iter().map(|x| x.seq).collect::<Vec<_>>(), gap, next), (vec![2, 3], false, 3));
        assert_eq!(log.since(3, Duration::ZERO).0.len(), 0);
    }

    /// The log keeps the last batches; a reader behind them is told it missed some.
    #[test]
    fn a_reader_behind_the_kept_batches_is_told() {
        let log = Log::new();
        for i in 0..KEEP as u128 + 10 {
            log.push(vec![Event::Removed { blocks: vec![i] }]);
        }
        let (b, gap, next) = log.since(1, Duration::ZERO);
        assert_eq!((b.len(), gap, next, b[0].seq), (KEEP, true, KEEP as u64 + 11, 12));
        assert!(!log.since(11, Duration::ZERO).1, "a reader just at the oldest kept missed nothing");
    }

    /// A reader with nothing new waits for the next batch, up to its wait.
    #[test]
    fn a_reader_waits_for_the_next_batch() {
        let log = std::sync::Arc::new(Log::new());
        let w = log.clone();
        let t = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            w.push(vec![Event::Cleared]);
        });
        let (b, _, next) = log.since(1, Duration::from_secs(5));
        t.join().unwrap();
        assert_eq!((b.len(), next), (1, 2));
        let t0 = std::time::Instant::now();
        assert!(log.since(2, Duration::from_millis(30)).0.is_empty() && t0.elapsed() >= Duration::from_millis(25));
    }

    /// The endpoint's schema: names as 32 hex digits, the stored blocks' tokens, the block size.
    #[test]
    fn the_schema_is_the_documented_one() {
        let log = Log::new();
        log.push(vec![Event::Stored { blocks: vec![0xab, 0xcd], parent: Some(0x1), tokens: (0..32).collect() }, Event::Removed { blocks: vec![0xab] }]);
        let v: serde_json::Value = serde_json::from_str(&log.json(1, Duration::ZERO)).unwrap();
        assert_eq!((v["next"].as_u64(), v["oldest"].as_u64(), v["gap"].as_bool()), (Some(2), Some(1), Some(false)));
        let ev = &v["batches"][0]["events"];
        assert_eq!(ev[0]["blocks"][1].as_str(), Some("000000000000000000000000000000cd"));
        assert_eq!(
            (ev[0]["parent"].as_str().map(str::len), ev[0]["tokens"].as_array().map(Vec::len), ev[0]["block_size"].as_u64()),
            (Some(32), Some(32), Some(16))
        );
        assert_eq!((ev[1]["type"].as_str(), ev[1]["medium"].as_str()), (Some("removed"), Some("gpu")));
    }
}
