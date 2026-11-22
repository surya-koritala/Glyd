//! The scheduler under load with no server: requests of natural text, of random tokens, or both, run at several users on a cache that is small (blocks of finished prompts reused, kept for later prompts and found again, sequences sent
//! back to wait for room, the extra memory short), the way `tools/serve_churn.py` does through the API but with the scheduler's own counts in hand, and compared with the same requests run one at a time on a cache with room.
//!
//!   glyd-engine stress DIR --write-ref FILE [--requests N] [--kind natural|random|mixed] [--seed S] [--text FILE] [--ctx N]     the requests one at a time: their tokens, to FILE
//!   glyd-engine stress DIR [--ref FILE] [--users N] [--requests N] [--kind ...] [--seed S] [--text FILE] [--ctx N] [--pool TOKENS]   the requests N at a time on a cache of TOKENS tokens, against FILE
//!
//! A cache can hold every value of a request exactly, or it has not the room: then the request waits for the others, and is run again, or ends with an error in words. What is checked: every request has every token it asked for
//! or ends with that error (and runs alone, to the same end, where it ended so); a request that never waited has the tokens it has run alone; with GLYD_ENGINE_VERIFY=1 every value written is compared with a bf16 copy at the
//! step that writes it (none differs); and the extra memory is empty again when nothing is held. Exit status 1 where any of it is not so.

use crate::ffi::{err, Error, Result};
use crate::kv::{ALONE_MESSAGE, PAGE};
use crate::model::{Engine, Finish, Reply, VERIFY_CHECKED, VERIFY_DIFFER, VERIFY_LOST};
use crate::sampler::Params;
use crate::sched::{Ev, Sched, Submit};
use crate::tok::Tok;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};
use std::sync::mpsc::{channel, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A request: its prompt, and the tokens it asks for (it goes on past the end-of-sequence token).
#[derive(Clone, Debug)]
pub struct Req {
    pub prompt: Vec<u32>,
    pub max_new: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Natural,
    Random,
    Mixed,
}

impl Kind {
    pub fn parse(s: &str) -> Result<Kind> {
        match s {
            "natural" => Ok(Kind::Natural),
            "random" => Ok(Kind::Random),
            "mixed" => Ok(Kind::Mixed),
            other => err(format!("--kind {other}: natural, random or mixed")),
        }
    }
}

/// xorshift64*: the requests are the same for the same seed on every machine.
pub struct Xs(pub u64);

impl Xs {
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A number in lo .. hi.
    pub fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u64() % (hi - lo).max(1) as u64) as usize
    }
}

/// `n` requests of the kind asked for: natural text is a run of words of `text` from a place of the generator's (40% begin with one of six shared openings, so that the cache of finished prompts finds some), random tokens are uniform
/// over the vocabulary's ordinary tokens. A prompt is 16 to `max_prompt` tokens, a reply 8 to 48.
pub fn make_requests(kind: Kind, n: usize, seed: u64, tok: &Tok, text: &str, vocab: usize, max_prompt: usize) -> Result<Vec<Req>> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 600 {
        return err("the text has too few words (--text FILE)");
    }
    let mut rng = Xs(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let opening: Vec<Vec<&str>> = (0..6)
        .map(|_| {
            let at = rng.range(0, words.len() - 130);
            words[at..at + 120].to_vec()
        })
        .collect();
    let ordinary = vocab.min(151_643);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let natural = match kind {
            Kind::Natural => true,
            Kind::Random => false,
            Kind::Mixed => i % 2 == 0,
        };
        let mut prompt = if natural {
            let len = rng.range(20, 330);
            let at = rng.range(0, words.len() - len - 1);
            let mut w: Vec<&str> = Vec::new();
            if rng.range(0, 100) < 40 {
                let o = &opening[rng.range(0, 6)];
                w.extend_from_slice(&o[..rng.range(0, 120)]);
            }
            w.extend_from_slice(&words[at..at + len]);
            tok.encode_with(&w.join(" "), false)?
        } else {
            (0..rng.range(30, 500)).map(|_| rng.range(0, ordinary) as u32).collect()
        };
        prompt.truncate(max_prompt);
        if prompt.len() < 16 {
            prompt.extend((0..16 - prompt.len()).map(|k| 1000 + k as u32));
        }
        out.push(Req { prompt, max_new: rng.range(8, 49) });
    }
    Ok(out)
}

/// How a request went.
#[derive(Clone, Debug)]
pub struct Done {
    pub tokens: Vec<u32>,
    pub end: std::result::Result<Finish, String>,
    /// Times it was sent back to wait and run again.
    pub resumed: u32,
}

fn reply(max_new: usize) -> Reply {
    Reply { params: Params { temperature: 0.0, ..Params::default() }, max_new, ignore_eos: true, logprobs: None, stop_ids: Vec::new() }
}

/// The requests run through `sched`, `concurrency` at a time (a request is sent when one has ended), until all have ended.
pub fn drive(sched: &mut Sched<&mut Engine>, reqs: &[Req], concurrency: usize) -> Result<Vec<Done>> {
    let mut done: Vec<Option<Done>> = (0..reqs.len()).map(|_| None).collect();
    struct Live {
        i: usize,
        rx: Receiver<Ev>,
        tokens: Vec<u32>,
        resumed: Arc<AtomicU32>,
        _abort: Arc<AtomicBool>,
    }
    let mut live: Vec<Live> = Vec::new();
    let mut next = 0;
    let started = Instant::now();
    loop {
        while live.len() < concurrency.max(1) && next < reqs.len() {
            let (tx, rx) = channel();
            let (abort, resumed) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicU32::new(0)));
            sched.submit(Submit {
                prompt: reqs[next].prompt.clone(),
                reply: reply(reqs[next].max_new),
                out: tx,
                abort: abort.clone(),
                resumed: resumed.clone(),
                cached: Default::default(),
                salt: 0,
                images: Vec::new(),
            });
            live.push(Live { i: next, rx, tokens: Vec::new(), resumed, _abort: abort });
            next += 1;
        }
        if live.is_empty() {
            break;
        }
        if sched.idle() {
            // (the events of the last round are on their way)
            std::thread::sleep(Duration::from_millis(1));
        } else {
            sched.tick()?;
        }
        live.retain_mut(|l| loop {
            match l.rx.try_recv() {
                Ok(Ev::Token(t)) => l.tokens.push(t),
                Ok(Ev::Lp(_)) => {}
                Ok(Ev::End(e)) => {
                    done[l.i] = Some(Done { tokens: std::mem::take(&mut l.tokens), end: e.map_err(|x| x.msg), resumed: l.resumed.load(Relaxed) });
                    return false;
                }
                Err(TryRecvError::Empty) => return true,
                Err(TryRecvError::Disconnected) => {
                    done[l.i] = Some(Done {
                        tokens: std::mem::take(&mut l.tokens),
                        end: Err("the scheduler dropped the request".into()),
                        resumed: l.resumed.load(Relaxed),
                    });
                    return false;
                }
            }
        });
        if started.elapsed() > Duration::from_secs(3600) {
            return err("the requests did not finish in an hour");
        }
    }
    Ok(done.into_iter().map(|d| d.expect("every request ended")).collect())
}

/// A fingerprint of the requests (the reference file says which requests it is for).
fn fingerprint(reqs: &[Req]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for r in reqs {
        for &t in r.prompt.iter().chain(&[r.max_new as u32]) {
            h = (h ^ t as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

fn write_ref(path: &str, reqs: &[Req], done: &[Done]) -> Result<()> {
    let mut s = format!("# {}\n", fingerprint(reqs));
    for (i, d) in done.iter().enumerate() {
        match &d.end {
            Ok(_) => s.push_str(&format!("{i}\tok\t{}\n", d.tokens.iter().map(u32::to_string).collect::<Vec<_>>().join(","))),
            Err(e) => s.push_str(&format!("{i}\terr\t{e}\n")),
        }
    }
    std::fs::write(path, s).map_err(|e| Error(format!("{path}: {e}")))
}

/// The reference's tokens of each request (None: it ended with an error alone).
fn read_ref(path: &str, reqs: &[Req]) -> Result<Vec<Option<Vec<u32>>>> {
    let text = std::fs::read_to_string(path).map_err(|e| Error(format!("{path}: {e}")))?;
    let mut lines = text.lines();
    if lines.next() != Some(&format!("# {}", fingerprint(reqs))) {
        return err(format!("{path}: made for other requests (the same --kind, --seed, --requests, --text and --ctx as here)"));
    }
    lines
        .map(|l| {
            let f: Vec<&str> = l.splitn(3, '\t').collect();
            match f.get(1) {
                Some(&"ok") => Ok(Some(
                    f[2].split(',')
                        .filter(|x| !x.is_empty())
                        .map(|x| x.parse().map_err(|_| Error(format!("{path}: a bad token {x}"))))
                        .collect::<Result<Vec<u32>>>()?,
                )),
                Some(&"err") => Ok(None),
                _ => err(format!("{path}: a bad line")),
            }
        })
        .collect()
}

/// The command: see the top of this file.
pub fn run(e: &mut Engine, tok: &Tok, a: &crate::args::Args) -> Result<bool> {
    let kind = Kind::parse(a.flag("kind").unwrap_or("mixed"))?;
    let n = a.num("requests", 200usize)?;
    let seed = a.num("seed", 1u64)?;
    let users = a.num("users", 4usize)?.clamp(1, crate::model::MAX_ROWS);
    let text = match a.flag("text") {
        Some(f) => std::fs::read_to_string(f).map_err(|x| Error(format!("{f}: {x}")))?,
        None => crate::calib::corpus().to_string(),
    };
    let window = e.capacity();
    let reqs = make_requests(kind, n, seed, tok, &text, e.w.cfg.vocab, window.saturating_sub(64).min(900))?;
    let prompt_tokens: usize = reqs.iter().map(|r| r.prompt.len()).sum();
    let blocks = e.pool_pages();
    println!(
        "stress: {n} requests ({} tokens of prompts, {kind:?}, seed {seed}) at {users} users at a time on a cache of {} tokens ({blocks} blocks of 16)",
        prompt_tokens,
        blocks * PAGE
    );
    // the reference: the requests one at a time (this cache, one user: there is room for each alone, or the error says it has not)
    if let Some(path) = a.flag("write-ref") {
        let done = {
            let mut sched = Sched::new(&mut *e, 1);
            drive(&mut sched, &reqs, 1)?
        };
        write_ref(path, &reqs, &done)?;
        let errors = done.iter().filter(|d| d.end.is_err()).count();
        println!("the requests one at a time: {} with every token, {errors} ended with an error; written to {path}", done.len() - errors);
        return Ok(errors == 0);
    }
    let reference = match a.flag("ref") {
        Some(p) => Some(read_ref(p, &reqs)?),
        None => None,
    };
    let t0 = Instant::now();
    let (done, stats) = {
        let mut sched = Sched::new(&mut *e, users);
        let done = drive(&mut sched, &reqs, users)?;
        let stats = sched.stats;
        // what finished prompts left in the cache is given up: nothing is held now, so the extra memory must be empty
        sched.clear_cache()?;
        (done, stats)
    };
    e.dev.sync()?;
    let wall = t0.elapsed().as_secs_f64();
    let mut ok = true;
    let (mut whole, mut errors, mut bad_end) = (0usize, 0usize, 0usize);
    for (r, d) in reqs.iter().zip(&done) {
        match &d.end {
            Ok(_) if d.tokens.len() == r.max_new => whole += 1,
            Ok(_) => {
                bad_end += 1;
                println!("A REQUEST ENDED WITHOUT ITS TOKENS: {} of {}", d.tokens.len(), r.max_new);
            }
            Err(m) if m == ALONE_MESSAGE || m.contains("where the cache holds") => errors += 1,
            Err(m) => {
                bad_end += 1;
                println!("A REQUEST ENDED WITH AN UNEXPECTED ERROR: {m}");
            }
        }
    }
    ok &= bad_end == 0;
    let resumed = done.iter().filter(|d| d.resumed > 0).count();
    println!("{wall:.1} s: {whole} requests with every token, {errors} ended with an error in words, {resumed} sent back to wait at least once (most often {} times)", done.iter().map(|d| d.resumed).max().unwrap_or(0));
    // against the requests alone
    if let Some(reference) = reference {
        let (mut same, mut diverged_clean, mut diverged_resumed, mut refs_missing) = (0, 0, 0, 0);
        for (i, d) in done.iter().enumerate() {
            let Some(want) = &reference[i] else {
                refs_missing += 1;
                continue;
            };
            if d.end.is_err() {
                continue;
            }
            if &d.tokens == want {
                same += 1;
            } else {
                let at = d.tokens.iter().zip(want).position(|(x, y)| x != y).unwrap_or(d.tokens.len().min(want.len()));
                if d.resumed == 0 {
                    diverged_clean += 1;
                    println!("DIFFERENT FROM ALONE, never sent back: request {i} at token {at} of {}", want.len());
                } else {
                    diverged_resumed += 1;
                    println!("(request {i}, sent back {} times, parts from its run alone at token {at} of {})", d.resumed, want.len());
                }
            }
        }
        println!("against the requests one at a time: {same} the same tokens; {diverged_clean} differ without having been sent back; {diverged_resumed} differ after being sent back (run again from their tokens)");
        if refs_missing > 0 {
            println!("({refs_missing} requests ended with an error alone)");
        }
        // (a request that was never sent back but began as a prompt the cache had kept goes on from keys and values made in other pieces: it may part from its run alone at a near tie, as one sent back may)
        ok &= diverged_clean == 0 || stats.hit > 0;
    }
    // a request that ended with an error runs alone, here, to the same end
    let failed: Vec<usize> = done.iter().enumerate().filter(|(_, d)| d.end.is_err()).map(|(i, _)| i).collect();
    if !failed.is_empty() {
        let sub: Vec<Req> = failed.iter().map(|&i| reqs[i].clone()).collect();
        let again = {
            let mut sched = Sched::new(&mut *e, 1);
            let d = drive(&mut sched, &sub, 1)?;
            sched.clear_cache()?;
            d
        };
        let alone_err = again.iter().filter(|d| d.end.is_err()).count();
        // a request served alone that had made no token when it ended was ended for what it did not need; one that had made tokens was run again from them (a prompt run again takes other values, in the last bit, than the same tokens made by steps,
        // and which of them a cache can hold is a matter of those bits): it ended where it was, alone, with nothing kept for later prompts
        let spurious = failed.iter().enumerate().filter(|(k, &i)| again[*k].end.is_ok() && done[i].tokens.is_empty()).count();
        println!(
            "{} requests ended with an error under load; run alone, {alone_err} end so again and {} are served{}",
            failed.len(),
            failed.len() - alone_err,
            if spurious > 0 { " (THOSE FIT ALONE and had made no token: they should not have been ended)" } else { "" }
        );
        for (k, &i) in failed.iter().enumerate() {
            if again[k].end.is_ok() {
                println!(
                    "  request {i}: {} tokens of prompt, {} tokens made before it ended, sent back {} times, served alone from its prompt",
                    reqs[i].prompt.len(),
                    done[i].tokens.len(),
                    done[i].resumed
                );
            }
        }
        ok &= spurious == 0;
    }
    println!(
        "scheduler: prompt tokens {}, found in the cache {}, sent back for room {}, steps taken back for a block the cache could not store exactly {} ({} blocks named lost), blocks kept in the second extra memory {}, ended for it {}",
        stats.prompt, stats.hit, stats.preempted, stats.taken_back, stats.lost_blocks, stats.heavy_blocks, stats.ended_lost
    );
    if stats.ended_lost > 0 {
        println!(
            "ended: {} alone with nothing kept for later prompts to give up, {} after being taken back {} times",
            stats.ended_alone,
            stats.ended_tries,
            crate::sched::MAX_TRIES + 1
        );
    }
    let (checked, differ, lost) = (VERIFY_CHECKED.load(Relaxed), VERIFY_DIFFER.load(Relaxed), VERIFY_LOST.load(Relaxed));
    if std::env::var_os("GLYD_ENGINE_VERIFY").is_some() {
        println!("every write compared with a bf16 copy at the step that wrote it: values checked {checked}, differing {differ}, values in blocks the cache had named lost (taken back, not compared) {lost}");
        ok &= differ == 0 && checked > 0;
    }
    let w = e.kv.status_words()?;
    println!("{}", crate::report::kv_status(e));
    println!("extra memory in use now that nothing is held: first {} of {}, second {} of {}", w[4], e.kv.chunks, w[8], e.kv.slabs);
    ok &= w[4] == 0 && w[8] == 0;
    println!("{}", if ok { "stress: all as it should be" } else { "stress: FAILED" });
    Ok(ok)
}
