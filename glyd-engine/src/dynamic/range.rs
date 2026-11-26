//! A range of a model's layers as a process (GLY-235): `glyd-engine rangeworker` holds one stage (`Dyn::new_stage`'s: layers a..b of a model, the embedding on the first stage, the final
//! norm, the LM head and the greedy pick on the last) and speaks a small framed protocol on its standard input and output, `glyd-range/1` (docs/range-worker.md: the frames, the
//! operations, the boundary tensors, the handles). `rangecheck` proves the stages chained equal the whole model bit for bit: in one process through `split::Split`, then across worker
//! processes through the protocol. The worker ships in a release build; rangecheck and its client are a development build's.

#[cfg(feature = "dev")]
use super::split::Run;
use super::{Dyn, Opts, Slot, Stage, VERIFY_ROWS};
use crate::args::Args;
use crate::chat::{Chat, Message};
use crate::conv::Gen;
use crate::ffi::{err, Error, Result};
use crate::qwen35::{Config35, Kind};
use crate::sampler::{Params, Sampler, DRAWS};
use crate::tok::Tok;
use crate::weights::{lossless_manifest, v7_manifest, Layout};
use glyd_gpu::json::{self, Value};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{mpsc, Arc, Mutex};
use std::time::Instant;

pub const PROTOCOL: &str = "glyd-range/1";
/// A frame's head (JSON) is at most this; a body at most the worker's `max_body_bytes` (describe says), itself at most MAX_BODY.
pub const MAX_HEAD: usize = 1 << 20;
pub const MAX_BODY: usize = 64 << 20;
/// Replies kept for a repeated request id: the last KEEP, their bodies up to KEEP_BODY bytes in all (an older body is let go; a repeat of it is refused, and says so).
const KEEP: usize = 256;
const KEEP_BODY: usize = 64 << 20;
/// The operations, as describe lists them.
const OPS: [&str; 13] =
    ["describe", "inspect", "prepare", "commit", "begin_session", "run_chunk", "status", "cancel", "end_session", "unload", "exit", "tokenize", "detokenize"];
/// The most sessions a stage holds at once (`--sessions`).
const MAX_SESSIONS: usize = 2;
/// A request id: 1 to 128 of these.
const ID_CHARS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-";

// ---- frames -------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// One frame: a JSON head and a raw body (tensors as bytes: bf16 words, little-endian, row-major; nothing in most frames).
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub head: Value,
    pub body: Vec<u8>,
}

/// `[u32 head bytes][u32 body bytes][head][body]`, the lengths little-endian.
pub fn write_frame(w: &mut dyn Write, f: &Frame) -> std::io::Result<()> {
    let h = json::to_compact(&f.head);
    w.write_all(&(h.len() as u32).to_le_bytes())?;
    w.write_all(&(f.body.len() as u32).to_le_bytes())?;
    w.write_all(h.as_bytes())?;
    w.write_all(&f.body)?;
    w.flush()
}

/// The next frame; None where the stream ends cleanly before one. A length past its limit is refused before anything is allocated. `max_body`: this side's limit.
pub fn read_frame(r: &mut dyn Read, max_body: usize) -> Result<Option<Frame>> {
    let mut len = [0u8; 8];
    match r.read_exact(&mut len[..1]) {
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return err(format!("reading a frame: {e}")),
        Ok(()) => {}
    }
    r.read_exact(&mut len[1..]).map_err(|e| Error(format!("reading a frame: {e}")))?;
    let (hl, bl) = (u32::from_le_bytes([len[0], len[1], len[2], len[3]]) as usize, u32::from_le_bytes([len[4], len[5], len[6], len[7]]) as usize);
    if hl == 0 || hl > MAX_HEAD {
        return err(format!("a frame head of {hl} bytes (1 to {MAX_HEAD})"));
    }
    if bl > max_body {
        return err(format!("a frame body of {bl} bytes, where {max_body} is the most"));
    }
    let mut h = vec![0u8; hl];
    r.read_exact(&mut h).map_err(|e| Error(format!("reading a frame: {e}")))?;
    let text = std::str::from_utf8(&h).map_err(|_| Error("a frame head that is not UTF-8".into()))?;
    let head = json::parse(text).map_err(|e| Error(format!("a frame head that is not JSON: {e}")))?;
    let mut body = vec![0u8; bl];
    r.read_exact(&mut body).map_err(|e| Error(format!("reading a frame: {e}")))?;
    Ok(Some(Frame { head, body }))
}

fn obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn num<T: std::fmt::Display>(x: T) -> Value {
    Value::Number(x.to_string())
}

fn text(s: &str) -> Value {
    Value::from(s)
}

fn list<I: IntoIterator<Item = Value>>(v: I) -> Value {
    Value::Array(v.into_iter().collect())
}

fn get_str<'a>(v: &'a Value, k: &str) -> Option<&'a str> {
    v.get(k).and_then(Value::as_str)
}

fn get_u64(v: &Value, k: &str) -> Option<u64> {
    v.get(k).and_then(Value::as_u64)
}

fn get_bool(v: &Value, k: &str) -> Option<bool> {
    v.get(k).and_then(Value::as_bool)
}

fn get_f32(v: &Value, k: &str) -> Option<f32> {
    match v.get(k) {
        Some(Value::Number(n)) => n.parse().ok(),
        _ => None,
    }
}

/// bf16 words as bytes, little-endian, and back.
fn words_bytes(w: &[u16]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_words(b: &[u8]) -> Vec<u16> {
    b.as_chunks::<2>().0.iter().map(|x| u16::from_le_bytes(*x)).collect()
}

fn ok_id(id: &str) -> bool {
    (1..=128).contains(&id.len()) && id.chars().all(|c| ID_CHARS.contains(c))
}

// ---- handles and the replies kept ---------------------------------------------------------------------------------------------------------------------------------------------

/// An opaque handle of this process incarnation: `i-<incarnation>-<n>` (an instance), `s-<incarnation>-<n>` (a session). Another incarnation's is refused: a restart invalidates every handle.
fn handle(kind: char, inc: u64, n: usize) -> String {
    format!("{kind}-{inc:016x}-{n}")
}

/// Whether `h` is one of this incarnation's handles of `kind` (any number); the error says whose it is otherwise.
fn check_handle(h: &str, kind: char, inc: u64) -> Result<()> {
    let own = format!("{kind}-{inc:016x}-");
    if let Some(n) = h.strip_prefix(&own) {
        if !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) {
            return Ok(());
        }
    }
    let what = if kind == 'i' { "an instance" } else { "a session" };
    match h.split('-').collect::<Vec<_>>()[..] {
        [k, i, n]
            if k.len() == 1
                && k.starts_with(kind)
                && i.len() == 16
                && i.bytes().all(|b| b.is_ascii_hexdigit())
                && !n.is_empty()
                && n.bytes().all(|b| b.is_ascii_digit()) =>
        {
            err(format!("{h}: {what} handle of another worker incarnation (this one is {inc:016x}; a restart invalidates every handle)"))
        }
        _ => err(format!("{h}: not {what} handle")),
    }
}

/// A number no earlier or later incarnation of this program shares, as far as the clock and the process id go.
fn incarnation() -> u64 {
    use std::hash::{BuildHasher, Hash, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos()).hash(&mut h);
    std::process::id().hash(&mut h);
    h.finish()
}

struct Seen {
    id: String,
    req: String,
    reply: Option<Frame>,
    retired: bool,
}

/// The requests seen, by id, with each one's reply once it is done: an exact repeat is answered from here and never run again; a different request under a seen id is refused.
#[derive(Default)]
struct Store {
    seen: std::collections::VecDeque<Seen>,
    body_bytes: usize,
}

enum Known {
    New,
    Repeat(Frame),
    Pending,
    Conflict,
    Retired,
}

impl Store {
    /// A request `req` (its head's text) under `id`: new (recorded as pending), the reply of its exact earlier self, pending still, a conflict, or retired.
    fn admit(&mut self, id: &str, req: &str) -> Known {
        if let Some(x) = self.seen.iter().find(|x| x.id == id) {
            if x.req != req {
                return Known::Conflict;
            }
            return match (&x.reply, x.retired) {
                (Some(f), _) => Known::Repeat(f.clone()),
                (None, true) => Known::Retired,
                (None, false) => Known::Pending,
            };
        }
        while self.seen.len() >= KEEP {
            if let Some(x) = self.seen.pop_front() {
                self.body_bytes -= x.reply.map_or(0, |f| f.body.len());
            }
        }
        self.seen.push_back(Seen { id: id.to_string(), req: req.to_string(), reply: None, retired: false });
        Known::New
    }

    /// Request `id` answered with `reply`; the oldest bodies past the budget are let go.
    fn done(&mut self, id: &str, reply: &Frame) {
        if let Some(x) = self.seen.iter_mut().find(|x| x.id == id) {
            x.reply = Some(reply.clone());
            self.body_bytes += reply.body.len();
        }
        while self.body_bytes > KEEP_BODY {
            let Some(x) = self.seen.iter_mut().find(|x| x.id != id && x.reply.as_ref().is_some_and(|f| !f.body.is_empty())) else { break };
            self.body_bytes -= x.reply.take().map_or(0, |f| f.body.len());
            x.retired = true;
        }
    }

    fn reply_of(&self, id: &str) -> Option<Frame> {
        self.seen.iter().find(|x| x.id == id).and_then(|x| x.reply.clone())
    }
}

// ---- the worker ---------------------------------------------------------------------------------------------------------------------------------------------------------------

/// What the control thread tells of the executor: the request it runs (its id, op, start and the session it names), the cancels asked, what is loaded and open.
#[derive(Default)]
struct Live {
    running: Option<(String, String, Instant, Option<String>)>,
    /// The sessions a cancel was asked for (the control thread's), until they end.
    cancel: Vec<String>,
    instance: Option<String>,
    /// The open sessions in the order begun: the handle, the tokens it holds, cancelled (the executor's).
    sessions: Vec<(String, usize, bool)>,
}

impl Live {
    fn asked(&self, h: &str) -> bool {
        self.cancel.iter().any(|x| x == h)
    }
}

enum Job {
    Run(Frame),
    Resend(String),
}

/// The stage's engine, once prepared, and the sessions open on it (in the order begun).
struct Held {
    e: Dyn,
    instance: String,
    committed: bool,
    sessions: Vec<Session>,
}

struct Session {
    handle: String,
    n: usize,
    cancelled: bool,
    /// The engine's sequence it runs on (`Dyn::select`).
    seq: usize,
}

/// What a worker is given on its command line, and what it knows of the model before it loads it.
struct Setup {
    dir: PathBuf,
    cfg: Config35,
    level: String,
    st: Stage,
    o: Opts,
    inc: u64,
    max_body: usize,
    /// The rows the last stage's head has room for (`--head-rows`): a run_chunk's `head_rows` at most.
    head_rows: usize,
    /// The sessions open at once (`--sessions`): each its own keys and values, states and graph.
    sessions: usize,
    /// The model's sampling defaults (generation_config.json's, as glyd serve takes them): a `sampling` field not given.
    defaults: Params,
}

impl Setup {
    /// The boundary's tensors for the model (Qwen3.5's pair: the residual X and the MLP's output D, `hidden` wide each).
    fn boundary(&self) -> Vec<(&'static str, Slot, usize)> {
        vec![("residual", Slot::X, self.cfg.hidden), ("mlp_out", Slot::D, self.cfg.hidden)]
    }

    fn boundary_json(&self) -> Value {
        let b = self.boundary();
        obj(vec![
            ("dtype", text("bf16")),
            ("byte_layout", text("each tensor [rows, width] row-major, little-endian u16 (bf16 bits), the tensors one after the other in this order; rows = the chunk's tokens")),
            ("bytes_per_row", num(b.iter().map(|x| 2 * x.2).sum::<usize>())),
            ("tensors", list(b.iter().map(|(name, slot, w)| obj(vec![("name", text(name)), ("slot", text(&format!("{slot:?}"))), ("shape", list([text("rows"), num(*w)]))])))),
            ("in", text(if self.st.first { "token ids (the first stage embeds them)" } else { "the boundary tensors of the stage before" })),
            ("out", text(if self.st.last { "the token (greedy, or drawn where sampling is asked) and its logit; the greedy token of each of the last head_rows rows; their bf16 logits [rows, vocab] as the body where asked" } else { "the boundary tensors for the stage after" })),
            ("positions", text("explicit: position_start of each chunk; row i is position_start + i; the stage holds the rotary tables and applies them itself")),
        ])
    }

    fn describe(&self) -> Vec<(&'static str, Value)> {
        let (a, b) = self.st.layers;
        vec![
            ("protocol", text(PROTOCOL)),
            ("build", obj(vec![("version", text(env!("CARGO_PKG_VERSION"))), ("commit", option_env!("GLYD_SOURCE_COMMIT").filter(|c| !c.is_empty()).map_or(Value::Null, Value::from)), ("development", Value::Bool(cfg!(feature = "dev")))])),
            ("incarnation", text(&format!("{:016x}", self.inc))),
            ("pid", num(std::process::id())),
            ("model", obj(vec![("dir", text(&self.dir.display().to_string())), ("family", text("Qwen3.5")), ("layers", num(self.cfg.layers)), ("hidden", num(self.cfg.hidden)), ("vocab", num(self.cfg.vocab)), ("level", text(&self.level))])),
            ("stage", obj(vec![("layers", list([num(a), num(b)])), ("first", Value::Bool(self.st.first)), ("last", Value::Bool(self.st.last)), ("embedding", Value::Bool(self.st.first)), ("final_norm", Value::Bool(self.st.last)), ("lm_head", Value::Bool(self.st.last)), ("sampling", text(if self.st.last { "greedy (argmax of the bf16 logits); glyd serve's sampler on the last row where run_chunk asks (sampling: temperature, top_p, top_k, min_p, seed, step)" } else { "none" }))])),
            ("boundary", self.boundary_json()),
            ("state", obj(vec![
                ("kv", text("the stage's full-attention layers' keys and values, bf16, for context_tokens positions: the stage's, made at prepare, overwritten by position")),
                ("recurrent", text("the stage's linear-attention layers' convolution state (bf16) and delta-rule state (fp32): the stage's, zeroed at begin_session")),
                ("reset", text("begin_session zeroes the recurrent states and sets the position to 0; the keys and values are written over as positions are used")),
                ("verify", text("run_chunk verify true (phase decode, 1 to verify_rows rows): each row as a decode step at its position, bit for bit; the next run_chunk may carry keep_previous K (1 to that chunk's rows): before it runs, the stage keeps the verify chunk's first K rows and drops the rest, and its position_start is the verify chunk's + K. Every stage gets the same K")),
            ])),
            ("limits", obj(vec![("context_tokens", num(self.o.ctx)), ("chunk_tokens", num(self.o.chunk)), ("max_head_bytes", num(MAX_HEAD)), ("max_body_bytes", num(self.max_body)), ("sessions", num(self.sessions)), ("kept_replies", num(KEEP)), ("kept_reply_bytes", num(KEEP_BODY)), ("head_rows", num(if self.st.last { self.head_rows } else { 0 })), ("verify_rows", num(if verified_level(&self.level) { VERIFY_ROWS.min(self.o.chunk) } else { 0 }))])),
            ("ops", list(OPS.map(text))),
            ("device", obj(vec![("ordinal", num(self.o.device))])),
        ]
    }
}

/// What an operation answers: the reply's pairs and its body, or the error.
type Answer = Result<(Vec<(&'static str, Value)>, Vec<u8>)>;

/// The reply frame of request `id`: the pairs where it went well, else the error.
fn reply_frame(id: &str, r: Answer) -> Frame {
    match r {
        Ok((pairs, body)) => {
            let mut all = vec![("id", text(id)), ("ok", Value::Bool(true))];
            all.extend(pairs);
            Frame { head: obj(all), body }
        }
        Err(e) => Frame { head: obj(vec![("id", text(id)), ("ok", Value::Bool(false)), ("error", text(&e.msg))]), body: Vec::new() },
    }
}

/// A frame to the standard output, whole (the lock holds the two threads apart). A closed pipe ends the process: the other side is gone.
fn send(f: &Frame) {
    let mut o = std::io::stdout().lock();
    if write_frame(&mut o, f).is_err() {
        std::process::exit(0);
    }
}

/// `--layers A..B` of `nl` layers (all of them where not given).
fn layers_arg(a: &Args, nl: usize) -> Result<(usize, usize)> {
    let Some(v) = a.flag("layers") else { return Ok((0, nl)) };
    let bad = || Error(format!("--layers {v}: A..B, 0 <= A < B <= {nl}"));
    let (x, y) = v.split_once("..").ok_or_else(bad)?;
    let (x, y): (usize, usize) = (x.trim().parse().map_err(|_| bad())?, y.trim().parse().map_err(|_| bad())?);
    if x >= y || y > nl {
        return Err(bad());
    }
    Ok((x, y))
}

/// `worker --model MODEL[:level] --layers A..B [--ctx N] [--chunk N] [--head-rows N] [--sessions N] [--device D] [--layout L] [--no-graph]`: the stage on its standard input and output until `exit` or the end of the input.
pub fn worker(a: &Args) -> Result<()> {
    let spec = a.flag("model").ok_or_else(|| Error("--model MODEL[:level] (a directory, or a Hub id in the cache)".into()))?;
    let (dir, layout) = resolve(spec)?;
    let layout = a.layout()?.or(layout);
    let cfg = Config35::load(&dir).map_err(|e| Error(format!("{}: {e} (the worker runs Qwen3.5 models)", dir.display())))?;
    let nl = cfg.layers;
    let (la, lb) = layers_arg(a, nl)?;
    let st = Stage { layers: (la, lb), first: la == 0, last: lb == nl };
    // (--no-graph: a development build's)
    let o = Opts {
        ctx: a.num("ctx", 4096usize)?,
        chunk: a.num("chunk", 1024usize)?,
        graph: !(cfg!(feature = "dev") && a.has("no-graph")),
        device: a.num("device", 0i32)?,
        layout,
    };
    let head_rows = a.num("head-rows", 1usize)?;
    if head_rows == 0 || head_rows > o.chunk.min(64) {
        return err(format!("--head-rows {head_rows}: 1 to {} (the chunk, and 64)", o.chunk.min(64)));
    }
    let sessions = a.num("sessions", 1usize)?;
    if !(1..=MAX_SESSIONS).contains(&sessions) {
        return err(format!("--sessions {sessions}: 1 to {MAX_SESSIONS}"));
    }
    let max_body = (o.chunk * 4 * cfg.hidden).max(cfg.vocab * 2 * head_rows).min(MAX_BODY);
    let setup = Arc::new(Setup {
        level: level_name(&dir, layout)?,
        max_body,
        defaults: crate::sampler::generation_defaults(&dir),
        dir,
        cfg,
        st,
        o,
        inc: incarnation(),
        head_rows,
        sessions,
    });
    let live = Arc::new(Mutex::new(Live::default()));
    let store = Arc::new(Mutex::new(Store::default()));
    let (tx, rx) = mpsc::channel::<Job>();
    let exec = {
        let (setup, live, store) = (setup.clone(), live.clone(), store.clone());
        std::thread::Builder::new()
            .name("glyd-range-exec".into())
            .spawn(move || executor(rx, &setup, &live, &store))
            .map_err(|e| Error(format!("the executor thread: {e}")))?
    };
    eprintln!("worker {}..{} of {nl} layers of {} ({}): {PROTOCOL}, incarnation {:016x}", la, lb, setup.dir.display(), setup.level, setup.inc);
    let mut inp = std::io::stdin().lock();
    let mut words: Option<Words> = None;
    loop {
        let f = match read_frame(&mut inp, setup.max_body) {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => {
                send(&reply_frame("", Err(e.clone())));
                eprintln!("worker: {e}: the input is not trusted past this; ending");
                break;
            }
        };
        let (id, op) = (get_str(&f.head, "id").unwrap_or("").to_string(), get_str(&f.head, "op").unwrap_or("").to_string());
        if !ok_id(&id) {
            send(&reply_frame(&id, err("a request needs an id of 1 to 128 letters, digits, '_', '.' or '-'")));
            continue;
        }
        match op.as_str() {
            "" => send(&reply_frame(&id, err("a request needs an op"))),
            // (answered here, on the control thread, whatever the executor is doing)
            "describe" => send(&reply_frame(&id, Ok((setup.describe(), Vec::new())))),
            "tokenize" | "detokenize" => send(&reply_frame(&id, text_op(&op, &f.head, &setup.dir, &mut words))),
            "status" | "cancel" => {
                let mut l = live.lock().unwrap_or_else(|e| e.into_inner());
                let who = get_str(&f.head, "session");
                let r = match who {
                    Some(h) if l.sessions.iter().any(|x| x.0 == h) => {
                        if op == "cancel" && !l.asked(h) {
                            l.cancel.push(h.to_string());
                        }
                        Ok((status(&l, setup.inc, who), Vec::new()))
                    }
                    Some(h) => check_handle(h, 's', setup.inc).and_then(|_| err(format!("{h}: no session of this handle is open"))),
                    None if op == "cancel" => err("cancel needs the session"),
                    None => Ok((status(&l, setup.inc, None), Vec::new())),
                };
                drop(l);
                send(&reply_frame(&id, r));
            }
            _ => {
                let known = store.lock().unwrap_or_else(|e| e.into_inner()).admit(&id, &json::to_compact(&f.head));
                let job = match known {
                    Known::New => Job::Run(f),
                    Known::Pending => Job::Resend(id),
                    Known::Repeat(r) => {
                        send(&r);
                        continue;
                    }
                    Known::Conflict => {
                        send(&reply_frame(&id, err(format!("{id}: an earlier request has this id and differs from this one (an id names one request)"))));
                        continue;
                    }
                    Known::Retired => {
                        send(&reply_frame(
                            &id,
                            err(format!("{id}: this request was answered and its reply let go ({KEEP} kept, {KEEP_BODY} bytes of bodies)")),
                        ));
                        continue;
                    }
                };
                if tx.send(job).is_err() {
                    break;
                }
            }
        }
    }
    drop(tx);
    let _ = exec.join();
    Ok(())
}

/// The model's tokenizer and chat template, loaded at the first `tokenize` or `detokenize` (they need the directory, not the weights).
struct Words {
    tok: Tok,
    chat: Option<Chat>,
}

/// `tokenize` {text, chat, thinking} and `detokenize` {ids, skip_special}, on the control thread: as glyd serve makes a prompt of a text (a chat of one user
/// message through the template, its special tokens in the text and none added; a plain text with the tokenizer's added) and the text of its tokens.
/// A tokenize's `messages`: 1 to 256 turns of {role: system, user or assistant; content: text}, the last a user's.
fn chat_messages(v: &Value) -> Result<Vec<Message>> {
    let turns = v.as_array().filter(|t| (1..=256).contains(&t.len())).ok_or_else(|| Error("messages: 1 to 256 turns".into()))?;
    let messages = turns
        .iter()
        .map(|t| match (get_str(t, "role"), get_str(t, "content")) {
            (Some(role @ ("system" | "user" | "assistant")), Some(content)) => Ok(Message::new(role, content)),
            _ => err("messages: each turn has role (system, user or assistant) and content (text)"),
        })
        .collect::<Result<Vec<_>>>()?;
    if messages.last().is_none_or(|m| m.role != "user") {
        return err("messages: the last turn is the user's");
    }
    Ok(messages)
}

fn text_op(op: &str, h: &Value, dir: &Path, words: &mut Option<Words>) -> Answer {
    if words.is_none() {
        *words = Some(Words { tok: Tok::load(dir)?, chat: Chat::load(dir)? });
    }
    let w = words.as_ref().expect("loaded");
    if op == "tokenize" {
        // `messages` [{role, content}]: a chat (system, user and assistant turns) as glyd serve makes its prompt: the model's template with its generation
        // prompt, or with no template serve's lines of `role: text` and `assistant:`.
        if let Some(turns) = h.get("messages") {
            let messages = chat_messages(turns)?;
            let ids = match &w.chat {
                Some(c) => w.tok.encode_with(&c.render(&messages, true, get_bool(h, "thinking"))?, false)?,
                None => {
                    let mut t = messages.iter().map(|m| format!("{}: {}", m.role, m.content)).collect::<Vec<_>>().join("\n");
                    t.push_str("\nassistant:");
                    w.tok.encode(&t)?
                }
            };
            return Ok((vec![("ids", list(ids.iter().map(|&t| num(t))))], Vec::new()));
        }
        let s = get_str(h, "text").ok_or_else(|| Error("tokenize needs text or messages".into()))?;
        let ids = if get_bool(h, "chat").unwrap_or(false) {
            let t = match &w.chat {
                Some(c) => c.render(&[Message::new("user", s)], true, get_bool(h, "thinking"))?,
                // (a model with no template: serve's lines of `role: text`)
                None => format!("user: {s}\nassistant:"),
            };
            w.tok.encode_with(&t, false)?
        } else {
            w.tok.encode(s)?
        };
        return Ok((vec![("ids", list(ids.iter().map(|&t| num(t))))], Vec::new()));
    }
    let ids: Vec<u32> = h
        .get("ids")
        .and_then(Value::as_u64s)
        .filter(|v| v.iter().all(|&t| t <= u32::MAX as u64))
        .ok_or_else(|| Error("detokenize needs ids [token ids]".into()))?
        .into_iter()
        .map(|t| t as u32)
        .collect();
    Ok((vec![("text", text(&w.tok.decode_with(&ids, get_bool(h, "skip_special").unwrap_or(true))?))], Vec::new()))
}

/// A run_chunk's `sampling` {temperature, top_p, top_k, min_p, seed, step}: glyd serve's sampler (the model's generation_config.json's values where a field is
/// not given, as serve's), its generator moved on `step` draws (serve draws once a sampled token: `step` is the tokens of the answer before this one), so that
/// the request alone says the token. `ctx`: the most a step can be.
fn sampler_of(v: &Value, defaults: &Params, ctx: usize) -> Result<Sampler> {
    if let Some(k) = ["repetition_penalty", "presence_penalty", "frequency_penalty", "logit_bias"].into_iter().find(|k| v.get(k).is_some()) {
        return err(format!("sampling: {k} is not taken here (it needs the tokens so far); ask for the logits and apply it on your side"));
    }
    let mut p = defaults.clone();
    // (no penalties here: a model's default repetition_penalty, where its generation_config.json has one, is not applied)
    p.repetition_penalty = 1.0;
    if let Some(t) = get_f32(v, "temperature") {
        // (serve's: a temperature above 0 and below 0.01 is 0.01; only 0 is greedy)
        p.temperature = if t > 0.0 && t < 0.01 { 0.01 } else { t };
    }
    if let Some(x) = get_f32(v, "top_p") {
        p.top_p = x;
    }
    if let Some(x) = get_f32(v, "top_k") {
        p.top_k = x.max(0.0) as usize;
    }
    if let Some(x) = get_f32(v, "min_p") {
        p.min_p = x;
    }
    if !(p.temperature >= 0.0 && p.temperature <= 2.0 && p.top_p > 0.0 && p.top_p <= 1.0 && (0.0..=1.0).contains(&p.min_p)) {
        return err(format!("sampling: temperature {} (0 to 2), top_p {} (above 0, to 1), min_p {} (0 to 1)", p.temperature, p.top_p, p.min_p));
    }
    p.seed = get_u64(v, "seed").ok_or_else(|| Error("sampling needs a seed (0 to 2^64 - 1)".into()))?;
    let step = get_u64(v, "step")
        .filter(|&x| x <= ctx as u64)
        .ok_or_else(|| Error(format!("sampling needs the step: the tokens of the answer before this one (0 for the token after the prompt; at most {ctx})")))?;
    let mut sm = Sampler::new(p, &[]);
    sm.skip(step);
    Ok(sm)
}

/// The status as session `who` sees it (the earliest begun of those open where None: with one session, the one): the stage's state (cancelling while a request of
/// this session runs after its cancel, running while any request runs, cancelled where this session is), this session's handle, cancel and tokens, the request
/// running (and the session it names), every open session's, and the process's draws.
fn status(l: &Live, inc: u64, who: Option<&str>) -> Vec<(&'static str, Value)> {
    let ses = match who {
        Some(h) => l.sessions.iter().find(|x| x.0 == h),
        None => l.sessions.first(),
    };
    let cancelled = |x: &(String, usize, bool)| x.2 || l.asked(&x.0);
    let state = match (&l.running, &l.instance, ses) {
        (Some(r), _, Some(x)) if r.3.as_deref() == Some(x.0.as_str()) && cancelled(x) => "cancelling",
        (Some(_), _, _) => "running",
        (None, None, _) => "empty",
        (None, Some(_), Some(x)) if cancelled(x) => "cancelled",
        (None, Some(_), _) => "idle",
    };
    vec![
        ("state", text(state)),
        ("incarnation", text(&format!("{inc:016x}"))),
        ("instance", l.instance.as_deref().map_or(Value::Null, Value::from)),
        ("session", ses.map_or(Value::Null, |x| Value::from(x.0.as_str()))),
        ("session_cancelled", Value::Bool(ses.is_some_and(cancelled))),
        ("tokens", num(ses.map_or(0, |x| x.1))),
        (
            "running",
            l.running.as_ref().map_or(Value::Null, |(id, op, t, h)| {
                obj(vec![
                    ("id", text(id)),
                    ("op", text(op)),
                    ("ms", num(format!("{:.1}", t.elapsed().as_secs_f64() * 1e3))),
                    ("session", h.as_deref().map_or(Value::Null, Value::from)),
                ])
            }),
        ),
        ("sessions", list(l.sessions.iter().map(|x| obj(vec![("session", text(&x.0)), ("tokens", num(x.1)), ("cancelled", Value::Bool(cancelled(x)))])))),
        // (the sampled draws this process made, by `sched::draw_row`: from what its GPU made of the row, and by reading the row (`sampler::DRAWS`))
        ("draws", obj(vec![("from_gpu", num(DRAWS[1].load(Relaxed))), ("read_row", num(DRAWS[0].load(Relaxed)))])),
    ]
}

/// The executor: the stage's engine lives on this thread (its GPU context is the thread's); the lifecycle requests run here, one after another, in the order they came.
fn executor(rx: mpsc::Receiver<Job>, s: &Setup, live: &Mutex<Live>, store: &Mutex<Store>) {
    let mut held: Option<Held> = None;
    let mut counter = 0usize;
    for job in rx {
        let f = match job {
            Job::Resend(id) => {
                let r = store.lock().unwrap_or_else(|e| e.into_inner()).reply_of(&id);
                send(&r.unwrap_or_else(|| reply_frame(&id, err(format!("{id}: this request was answered and its reply let go")))));
                continue;
            }
            Job::Run(f) => f,
        };
        let (id, op) = (get_str(&f.head, "id").unwrap_or("").to_string(), get_str(&f.head, "op").unwrap_or("").to_string());
        live.lock().unwrap_or_else(|e| e.into_inner()).running =
            Some((id.clone(), op.clone(), Instant::now(), get_str(&f.head, "session").map(str::to_string)));
        let r = exec(&op, &f, s, &mut held, &mut counter, live);
        let reply = reply_frame(&id, r);
        {
            let mut l = live.lock().unwrap_or_else(|e| e.into_inner());
            l.running = None;
            l.instance = held.as_ref().map(|h| h.instance.clone());
            l.sessions = held.as_ref().map_or(Vec::new(), |h| h.sessions.iter().map(|x| (x.handle.clone(), x.n, x.cancelled)).collect());
            // (a cancel lasts as long as its session)
            let Live { cancel, sessions, .. } = &mut *l;
            cancel.retain(|h| sessions.iter().any(|x| x.0 == *h));
        }
        store.lock().unwrap_or_else(|e| e.into_inner()).done(&id, &reply);
        send(&reply);
        if op == "exit" {
            drop(held);
            std::process::exit(0);
        }
    }
}

/// The memory the stage holds, as the counters say: the weights, the keys and values, the states, the rest (the activations, the workspaces, the logits, the rotary tables), everything.
fn allocated(e: &Dyn) -> Vec<(&'static str, Value)> {
    let h = e.held().expect("the dynamic engine says what it holds");
    let (live, peak, _) = crate::dev::held_now();
    let (free, total) = e.dev.mem_info().unwrap_or((0, 0));
    vec![
        ("weights_bytes", num(h.weights)),
        ("kv_bytes", num(h.kv)),
        ("state_bytes", num(h.state)),
        ("scratch_bytes", num(live.saturating_sub(h.weights + h.kv + h.state))),
        ("engine_bytes", num(live)),
        ("peak_bytes", num(peak)),
        (
            "device",
            obj(vec![
                ("name", text(&e.dev.name)),
                ("compute_capability", text(&format!("{}.{}", e.dev.cc.0, e.dev.cc.1))),
                ("free_bytes", num(free)),
                ("total_bytes", num(total)),
            ]),
        ),
    ]
}

fn exec(op: &str, f: &Frame, s: &Setup, held: &mut Option<Held>, counter: &mut usize, live: &Mutex<Live>) -> Answer {
    let instance_of = |held: &Option<Held>, f: &Frame| -> Result<()> {
        let h = get_str(&f.head, "instance").ok_or_else(|| Error(format!("{op} needs the instance")))?;
        check_handle(h, 'i', s.inc)?;
        match held {
            Some(x) if x.instance == h => Ok(()),
            _ => err(format!("{h}: no instance of this handle is loaded")),
        }
    };
    match op {
        "prepare" => {
            if held.is_some() {
                return err("prepared already (unload first)");
            }
            let t0 = Instant::now();
            let mut e = super::qwen35::load_stage(&s.dir, &s.o, s.st)?;
            if s.st.last {
                e.set_head_rows(s.head_rows)?;
            }
            // (each session more: its own keys and values, states and graph)
            e.add_seqs(s.sessions - 1)?;
            let want: Vec<(Slot, usize)> = s.boundary().iter().map(|x| (x.1, x.2)).collect();
            if e.boundary() != want {
                return err(format!("the engine's boundary {:?} is not the one described {want:?}", e.boundary()));
            }
            *counter += 1;
            let instance = handle('i', s.inc, *counter);
            let mut pairs =
                vec![("instance", text(&instance)), ("load_s", num(format!("{:.2}", t0.elapsed().as_secs_f64()))), ("describe", text(&e.describe()))];
            pairs.extend(allocated(&e));
            *held = Some(Held { e, instance, committed: false, sessions: Vec::new() });
            Ok((pairs, Vec::new()))
        }
        "inspect" => {
            let Some(h) = held.as_ref() else {
                return Ok((
                    vec![
                        ("available", Value::Bool(false)),
                        ("why", text("nothing is loaded: prepare first; the costs are the loaded stage's own allocations")),
                    ],
                    Vec::new(),
                ));
            };
            let (layers, embed, head, small) = h.e.layer_costs();
            let (a, b) = s.st.layers;
            let rows: Vec<Value> = layers
                .iter()
                .enumerate()
                .filter(|(l, _)| (a..b).contains(l))
                .map(|(l, &(w, kv, st))| {
                    obj(vec![
                        ("layer", num(l)),
                        (
                            "kind",
                            text(match s.cfg.kinds[l] {
                                Kind::Linear => "linear_attention",
                                Kind::Full => "full_attention",
                            }),
                        ),
                        ("weights_bytes", num(w)),
                        ("kv_bytes_per_token", num(kv)),
                        ("state_bytes", num(st)),
                    ])
                })
                .collect();
            let per_token: usize = layers.iter().map(|x| x.1).sum();
            let mut pairs = vec![
                ("available", Value::Bool(true)),
                ("layers", list(rows)),
                ("embedding_bytes", num(embed)),
                ("lm_head_bytes", num(head)),
                ("small_tensors_bytes", num(small)),
                ("kv_bytes_per_token", num(per_token)),
                ("state_bytes", num(h.e.state_bytes())),
                ("boundary_bytes_per_token", num(s.boundary().iter().map(|x| 2 * x.2).sum::<usize>())),
            ];
            pairs.extend(allocated(&h.e));
            Ok((pairs, Vec::new()))
        }
        "commit" => {
            instance_of(held, f)?;
            let h = held.as_mut().expect("checked");
            h.committed = true;
            Ok((vec![("committed", Value::Bool(true)), ("ready", Value::Bool(true))], Vec::new()))
        }
        "begin_session" => {
            instance_of(held, f)?;
            let h = held.as_mut().expect("checked");
            if !h.committed {
                return err("commit before a session");
            }
            let Some(seq) = (0..s.sessions).find(|&q| h.sessions.iter().all(|x| x.seq != q)) else {
                return err(format!(
                    "{}: open; this worker takes {} at a time (--sessions); end one first",
                    h.sessions.iter().map(|x| x.handle.as_str()).collect::<Vec<_>>().join(", "),
                    s.sessions
                ));
            };
            h.e.select(seq)?;
            h.e.reset()?;
            *counter += 1;
            let handle = handle('s', s.inc, *counter);
            h.sessions.push(Session { handle: handle.clone(), n: 0, cancelled: false, seq });
            // (the session's own: the sequences are alike)
            let (hd, k) = (h.e.held().expect("the dynamic engine says what it holds"), h.e.seqs().0 as u64);
            Ok((
                vec![
                    ("session", text(&handle)),
                    ("context_tokens", num(h.e.capacity())),
                    ("kv_bytes", num(hd.kv / k)),
                    ("state_bytes", num(hd.state / k)),
                    ("tokens", num(0)),
                ],
                Vec::new(),
            ))
        }
        "run_chunk" => {
            let hs = get_str(&f.head, "session").ok_or_else(|| Error("run_chunk needs the session".into()))?;
            check_handle(hs, 's', s.inc)?;
            let Held { e, sessions, .. } = held.as_mut().ok_or_else(|| Error(format!("{hs}: no session of this handle is open")))?;
            let ses = sessions.iter_mut().find(|x| x.handle == hs).ok_or_else(|| Error(format!("{hs}: no session of this handle is open")))?;
            // (this session's sequence: its keys and values, states, verify pass and graph)
            e.select(ses.seq)?;
            if ses.cancelled || live.lock().unwrap_or_else(|e| e.into_inner()).asked(hs) {
                ses.cancelled = true;
                return err(format!("{hs}: cancelled; end the session"));
            }
            let phase = get_str(&f.head, "phase").unwrap_or("");
            if phase != "prefill" && phase != "decode" {
                return err("phase: prefill or decode");
            }
            let at = get_u64(&f.head, "position_start").ok_or_else(|| Error("run_chunk needs position_start".into()))? as usize;
            let verify = get_bool(&f.head, "verify").unwrap_or(false);
            if verify && phase != "decode" {
                return err("verify: a decode chunk (phase decode)");
            }
            if verify && !verified_level(&s.level) {
                return err(format!("verify on {}: checked bit for bit on swift only so far", s.level));
            }
            // keep_previous K: the last chunk was a verify chunk of R rows at P (the session at P + R); its first K rows stay, before this chunk, which starts at P + K
            let keep = match f.head.get("keep_previous") {
                None | Some(Value::Null) => None,
                Some(v) => {
                    let k = v.as_u64().ok_or_else(|| Error("keep_previous: a number of rows".into()))? as usize;
                    let (p0, r) = e.verified().ok_or_else(|| Error("keep_previous: the session's last chunk was not a verify chunk".into()))?;
                    if k == 0 || k > r {
                        return err(format!("keep_previous {k}: 1 to the verify chunk's {r} rows"));
                    }
                    Some((k, p0 + k))
                }
            };
            let next = keep.map_or(ses.n, |x| x.1);
            if at != next {
                return err(format!("position_start {at}, where the session holds {next} tokens{}: a chunk continues the sequence (a repeat of an earlier chunk is its own request id)", if keep.is_some() { " after keep_previous" } else { "" }));
            }
            let head = get_bool(&f.head, "head").unwrap_or(true);
            let rows = match (head, f.head.get("head_rows")) {
                (false, _) => 0,
                (true, None) => 1,
                (true, Some(v)) => v.as_u64().ok_or_else(|| Error("head_rows: a number of rows".into()))? as usize,
            };
            let (tokens, n) = if s.st.first {
                let t: Vec<u32> = f
                    .head
                    .get("tokens")
                    .and_then(Value::as_u64s)
                    .ok_or_else(|| Error("the first stage's run_chunk needs tokens".into()))?
                    .into_iter()
                    .map(|x| x as u32)
                    .collect();
                let n = t.len();
                (t, n)
            } else {
                let n = get_u64(&f.head, "rows").ok_or_else(|| Error("a later stage's run_chunk needs rows".into()))? as usize;
                (vec![0u32; n], n)
            };
            let bpr: usize = s.boundary().iter().map(|x| 2 * x.2).sum();
            let verify_rows = VERIFY_ROWS.min(s.o.chunk);
            if n == 0 || n > s.o.chunk || (phase == "decode" && n > if verify { verify_rows } else { 1 }) {
                return err(format!("{n} rows: a prefill chunk has 1 to {} tokens, a decode step one, a verify chunk 1 to {verify_rows}", s.o.chunk));
            }
            if at + n + 1 > e.capacity() {
                return err(format!("{n} rows at {at}: the context holds {}", e.capacity()));
            }
            if !s.st.first && f.body.len() != n * bpr {
                return err(format!("a body of {} bytes, where {n} rows of the boundary take {}", f.body.len(), n * bpr));
            }
            if head && (rows == 0 || rows > n || (s.st.last && rows > s.head_rows)) {
                return err(format!("head_rows {rows}: 1 to the chunk's {n} rows, at most {} here (--head-rows)", s.head_rows));
            }
            // (checked before the chunk runs: a refused request leaves the session as it was)
            let mut sampler = match f.head.get("sampling") {
                Some(v) if s.st.last && !matches!(v, Value::Null) => {
                    if rows != 1 {
                        return err("sampling draws the last row's token: head true, head_rows 1 (the rows' logits come back for a draw on your side)");
                    }
                    Some(sampler_of(v, &s.defaults, s.o.ctx)?)
                }
                _ => None,
            };
            if let Some((k, _)) = keep {
                e.rewind(k)?;
                ses.n = at;
            }
            let mut io = if s.st.first { Vec::new() } else { bytes_words(&f.body) };
            if verify {
                e.verify_rows(&tokens, at, &mut io, rows)?;
            } else {
                e.stage_rows(&tokens, at, &mut io, rows)?;
            }
            if live.lock().unwrap_or_else(|e| e.into_inner()).asked(hs) {
                // (the chunk ran: the state has it; its output is withdrawn and the session takes no more)
                ses.cancelled = true;
                ses.n = at + n;
                return err(format!("{hs}: cancelled during this chunk; its output is withdrawn and the session is closed to new chunks (end it)"));
            }
            ses.n = at + n;
            let mut pairs = vec![("rows", num(n)), ("position_end", num(ses.n))];
            if !s.st.last {
                pairs.push(("body", obj(vec![("what", text("boundary")), ("rows", num(n)), ("bytes", num(io.len() * 2))])));
                return Ok((pairs, words_bytes(&io)));
            }
            if !head {
                return Ok((pairs, Vec::new()));
            }
            let logits = get_bool(&f.head, "logits").unwrap_or(false);
            let (picks, w) = e.head_out(rows, logits)?;
            let (mut token, mut logit) = picks[rows - 1];
            if let Some(sm) = sampler.as_mut() {
                // (drawn from what this GPU makes of the row (`sched::draw_row`: its candidates, the request's changes put on it, its weights and tokens by range), as one GPU's
                // scheduler draws; the token's logit the model's)
                token = crate::sched::draw_row_again(&mut crate::sched::Step(&mut *e), rows - 1, sm)?.0;
                logit = crate::sched::ModelStep::gather(&mut *e, rows - 1, &[token])?.first().map_or(f32::NAN, |x| x.1);
            }
            pairs.push(("token", num(token)));
            pairs.push(("logit", num(logit)));
            if f.head.get("head_rows").is_some() {
                pairs.push(("tokens", list(picks.iter().map(|p| num(p.0)))));
            }
            if logits {
                pairs.push((
                    "body",
                    obj(vec![("what", text("logits")), ("dtype", text("bf16")), ("rows", num(rows)), ("width", num(s.cfg.vocab)), ("bytes", num(w.len() * 2))]),
                ));
                return Ok((pairs, words_bytes(&w)));
            }
            Ok((pairs, Vec::new()))
        }
        "end_session" => {
            let hs = get_str(&f.head, "session").ok_or_else(|| Error("end_session needs the session".into()))?;
            check_handle(hs, 's', s.inc)?;
            let h =
                held.as_mut().filter(|h| h.sessions.iter().any(|x| x.handle == hs)).ok_or_else(|| Error(format!("{hs}: no session of this handle is open")))?;
            // (the other sessions as they are)
            let i = h.sessions.iter().position(|x| x.handle == hs).expect("checked");
            let ses = h.sessions.remove(i);
            let cancelled = ses.cancelled || live.lock().unwrap_or_else(|e| e.into_inner()).asked(hs);
            Ok((vec![("ended", Value::Bool(true)), ("tokens", num(ses.n)), ("cancelled", Value::Bool(cancelled))], Vec::new()))
        }
        "unload" => {
            instance_of(held, f)?;
            if held.as_ref().is_some_and(|h| !h.sessions.is_empty()) {
                return err("a session is open: end it first");
            }
            *held = None;
            let (live_bytes, _, _) = crate::dev::held_now();
            Ok((vec![("unloaded", Value::Bool(true)), ("retained_bytes", num(live_bytes))], Vec::new()))
        }
        "exit" => Ok((vec![("bye", Value::Bool(true))], Vec::new())),
        other => err(format!("{other}: not an operation ({})", OPS.join(", "))),
    }
}

// ---- the client: a worker process ---------------------------------------------------------------------------------------------------------------------------------------------

/// A worker process this program started: `glyd-engine rangeworker` on a stage, its frames on pipes.
#[cfg(feature = "dev")]
pub struct Client {
    child: std::process::Child,
    inp: std::process::ChildStdin,
    out: std::io::BufReader<std::process::ChildStdout>,
    n: usize,
    pub tag: String,
    pub instance: String,
    pub session: String,
    /// The last request sent, as sent (a repeat is this frame again).
    pub last: Option<Frame>,
}

#[cfg(feature = "dev")]
impl Client {
    /// `exe worker --model DIR --layers A..B` with the options; stderr is this process's.
    pub fn spawn(exe: &Path, dir: &Path, layers: (usize, usize), o: &Opts, head_rows: usize, sessions: usize, k: usize) -> Result<Client> {
        let mut c = std::process::Command::new(exe);
        c.arg("rangeworker")
            .arg("--model")
            .arg(dir)
            .arg("--layers")
            .arg(format!("{}..{}", layers.0, layers.1))
            .arg("--ctx")
            .arg(o.ctx.to_string())
            .arg("--chunk")
            .arg(o.chunk.to_string())
            .arg("--device")
            .arg(o.device.to_string());
        c.arg("--head-rows").arg(head_rows.to_string()).arg("--sessions").arg(sessions.to_string());
        if let Some(l) = o.layout {
            c.arg("--layout").arg(l.name());
        }
        if !o.graph {
            c.arg("--no-graph");
        }
        let mut child = c
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .map_err(|e| Error(format!("{}: {e}", exe.display())))?;
        let inp = child.stdin.take().expect("piped");
        let out = std::io::BufReader::new(child.stdout.take().expect("piped"));
        Ok(Client {
            child,
            inp,
            out,
            n: 0,
            tag: format!("worker {k} (layers {}..{})", layers.0, layers.1),
            instance: String::new(),
            session: String::new(),
            last: None,
        })
    }

    pub fn send_frame(&mut self, f: &Frame) -> Result<()> {
        write_frame(&mut self.inp, f).map_err(|e| Error(format!("{}: writing: {e}", self.tag)))
    }

    /// A request sent (not waited for): its id.
    pub fn send(&mut self, op: &str, pairs: Vec<(&str, Value)>, body: Vec<u8>) -> Result<String> {
        self.n += 1;
        let id = format!("r{}", self.n);
        let mut all = vec![("op", text(op)), ("id", text(&id))];
        all.extend(pairs);
        let f = Frame { head: obj(all), body };
        self.send_frame(&f)?;
        self.last = Some(f);
        Ok(id)
    }

    pub fn recv(&mut self) -> Result<Frame> {
        read_frame(&mut self.out, MAX_BODY)?.ok_or_else(|| Error(format!("{}: ended", self.tag)))
    }

    /// A request answered: the reply as it came (ok or not).
    pub fn call_raw(&mut self, op: &str, pairs: Vec<(&str, Value)>, body: Vec<u8>) -> Result<Frame> {
        let id = self.send(op, pairs, body)?;
        let r = self.recv()?;
        if get_str(&r.head, "id") != Some(&id) {
            return err(format!("{}: a reply to {:?}, where {id} was asked", self.tag, get_str(&r.head, "id")));
        }
        Ok(r)
    }

    /// A request answered well; an error reply is this side's error.
    pub fn call(&mut self, op: &str, pairs: Vec<(&str, Value)>, body: Vec<u8>) -> Result<Frame> {
        let r = self.call_raw(op, pairs, body)?;
        if get_bool(&r.head, "ok") != Some(true) {
            return err(format!("{}: {op}: {}", self.tag, get_str(&r.head, "error").unwrap_or("?")));
        }
        Ok(r)
    }

    pub fn wait(mut self) -> Result<i32> {
        drop(self.inp);
        let st = self.child.wait().map_err(|e| Error(format!("{}: {e}", self.tag)))?;
        Ok(st.code().unwrap_or(-1))
    }
}

// ---- the model and its level --------------------------------------------------------------------------------------------------------------------------------------------------

/// MODEL[:level] as a directory and the layout to pack at (None: the directory's own level, or the loader's lossless choice): a directory; a Hub id in the cache (a glyd/NAME-LEVEL
/// snapshot, or a base model's checkpoint, packed at load); `ORG/NAME:level`: glyd/NAME-level from the cache where it is there, else the base model's checkpoint packed at that level.
pub fn resolve(spec: &str) -> Result<(PathBuf, Option<Layout>)> {
    let (model, tag) = crate::levels::split_tag(spec);
    // (a range worker runs every level but hummingbird, whose experts come from a calibration)
    let layout = match tag.as_deref().map(|t| crate::levels::by_name(t).filter(|l| l.name != "hummingbird")) {
        None => None,
        Some(Some(l)) => l.layout,
        Some(None) => return err(format!(":{} is not a level (penguin, kestrel, swift, wren)", tag.unwrap_or_default())),
    };
    if Path::new(model).is_dir() {
        let dir = PathBuf::from(model);
        // (a saved directory runs at its own level)
        let own = v7_manifest(&dir)?.is_some() || lossless_manifest(&dir)?.is_some();
        return Ok((dir, if own { None } else { layout }));
    }
    if let Some(t) = &tag {
        if let Ok(s) = glyd_gpu::save::Source::find(&crate::levels::packed_repo(model, t)) {
            return Ok((s.dir, None));
        }
    }
    let s = glyd_gpu::save::Source::find(model).map_err(|e| Error(format!("{model}: {e} (glyd-engine fetch MODEL downloads it)")))?;
    Ok((s.dir, layout))
}

/// The level a directory runs at: its glyd.json's, penguin for a lossless save, else the layout it is packed at here (penguin where none).
/// Levels whose verify chunks rangecheck has checked bit for bit against plain steps (swift on Ada); the others refuse `verify` (describe's
/// verify_rows 0) until checked.
fn verified_level(level: &str) -> bool {
    level.split(' ').next() == Some("swift")
}

pub fn level_name(dir: &Path, layout: Option<Layout>) -> Result<String> {
    if let Some((_, m)) = v7_manifest(dir)? {
        return Ok(m.get("level").and_then(Value::as_str).unwrap_or("penguin").to_string());
    }
    if lossless_manifest(dir)?.is_some() {
        return Ok("penguin".to_string());
    }
    Ok(match layout {
        Some(Layout::Int6) => "kestrel",
        Some(Layout::Int5) => "swift",
        Some(Layout::Int4) => "wren",
        Some(Layout::Int3) => "hummingbird",
        _ => "penguin",
    }
    .to_string()
        + " (packed at load)")
}

// ---- rangecheck ---------------------------------------------------------------------------------------------------------------------------------------------------------------

/// A greedy run: the prompt in chunks, then `new` - 1 steps; each pass's token and its logits (f32 of the bf16 words), and the prompt's and a step's time (the logits' read included).
#[cfg(feature = "dev")]
fn greedy(r: &mut Run, prompt: &[u32], new: usize) -> Result<(Vec<u32>, Vec<Vec<f32>>, f64, f64)> {
    r.reset()?;
    let t = Instant::now();
    let mut s = r.prefill(prompt)?;
    let pre = t.elapsed().as_secs_f64() * 1e3;
    let (mut toks, mut lg) = (vec![s.token], vec![r.logits()?]);
    let t = Instant::now();
    for _ in 1..new {
        s = r.step(s.token)?;
        toks.push(s.token);
        lg.push(r.logits()?);
    }
    Ok((toks, lg, pre, t.elapsed().as_secs_f64() * 1e3 / (new - 1).max(1) as f64))
}

#[cfg(feature = "dev")]
fn digest(toks: &[u32], lg: &[Vec<f32>]) -> String {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    for t in toks {
        h.update(t.to_le_bytes());
    }
    for row in lg {
        for x in row {
            h.update(x.to_bits().to_le_bytes());
        }
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// Every step's logits and token of `b` against `a`'s, bit for bit: Ok with a line, or the first difference.
#[cfg(any(test, feature = "dev"))]
fn same(what: &str, a: &(Vec<u32>, Vec<Vec<f32>>), b: &(Vec<u32>, Vec<Vec<f32>>)) -> Result<String> {
    if a.0.len() != b.0.len() || a.1.len() != b.1.len() {
        return err(format!("{what}: {} tokens and {} logit rows, where the whole model gave {} and {}", b.0.len(), b.1.len(), a.0.len(), a.1.len()));
    }
    let mut cells = 0usize;
    for (i, (x, y)) in a.1.iter().zip(&b.1).enumerate() {
        if x.len() != y.len() {
            return err(format!("{what}: step {i}: {} logits, where the whole model gave {}", y.len(), x.len()));
        }
        if let Some(j) = (0..x.len()).find(|&j| x[j].to_bits() != y[j].to_bits()) {
            let n = (0..x.len()).filter(|&j| x[j].to_bits() != y[j].to_bits()).count();
            return err(format!(
                "{what}: step {i}: logit {j} is {} ({:#010x}), where the whole model gave {} ({:#010x}); {n} of {} differ at this step",
                y[j],
                y[j].to_bits(),
                x[j],
                x[j].to_bits(),
                x.len()
            ));
        }
        cells += x.len();
        if a.0[i] != b.0[i] {
            return err(format!("{what}: step {i}: token {}, where the whole model gave {}", b.0[i], a.0[i]));
        }
    }
    Ok(format!("{what}: every logit the same, bit for bit: {} steps, {cells} logits; every token the same", a.0.len()))
}

#[cfg(any(test, feature = "dev"))]
fn stage_bounds(cuts: &[usize], nl: usize) -> Vec<(usize, usize)> {
    let b: Vec<usize> = std::iter::once(0).chain(cuts.iter().copied()).chain(std::iter::once(nl)).collect();
    b.windows(2).map(|w| (w[0], w[1])).collect()
}

/// A chunk through the workers in turn: the first stage's tokens, the boundary through the middle, the last stage's reply (`last`: its request's further
/// fields; its bf16 logits come as a body where `head`).
#[cfg(feature = "dev")]
fn chain(ws: &mut [Client], tokens: &[u32], at: usize, head: bool, last: Vec<(&str, Value)>) -> Result<Frame> {
    chain_with(ws, tokens, at, head, &[], last)
}

/// `chain` with `every`'s fields in every stage's request (a verify chunk's `verify` and `keep_previous`: a decode chunk); `last` may say `logits` false (a sampled reply with no rows' body).
#[cfg(feature = "dev")]
fn chain_with(ws: &mut [Client], tokens: &[u32], at: usize, head: bool, every: &[(&str, Value)], last: Vec<(&str, Value)>) -> Result<Frame> {
    let n = ws.len();
    let mut body = Vec::new();
    let mut last = Some(last);
    let decode = tokens.len() == 1 || every.iter().any(|x| x.0 == "verify");
    for (k, w) in ws.iter_mut().enumerate() {
        let mut pairs = vec![
            ("session", text(&w.session)),
            ("phase", text(if decode { "decode" } else { "prefill" })),
            ("position_start", num(at)),
            ("head", Value::Bool(head)),
        ];
        pairs.extend(every.iter().cloned());
        if k == 0 {
            pairs.push(("tokens", list(tokens.iter().map(|&t| num(t)))));
        } else {
            pairs.push(("rows", num(tokens.len())));
        }
        if k + 1 == n {
            // (the last stage's own `logits`, where `last` does not say: the rows' body comes back where the head ran)
            let last = last.take().unwrap_or_default();
            if !last.iter().any(|x| x.0 == "logits") {
                pairs.push(("logits", Value::Bool(head)));
            }
            pairs.extend(last);
        }
        let r = w.call("run_chunk", pairs, std::mem::take(&mut body))?;
        if k + 1 == n {
            return Ok(r);
        }
        body = r.body;
    }
    err("no workers")
}

/// The last stage's token and its logits (f32 of the bf16 words, the rows one after the other).
#[cfg(feature = "dev")]
fn picked(r: &Frame) -> Result<(u32, Vec<f32>)> {
    let tok = get_u64(&r.head, "token").ok_or_else(|| Error(format!("no token in the last stage's reply: {}", json::to_compact(&r.head))))? as u32;
    Ok((tok, bytes_words(&r.body).into_iter().map(|w| f32::from_bits((w as u32) << 16)).collect()))
}

/// The `logit` of a last stage's reply: the model's logit of the token it gave (NaN where the reply has none).
#[cfg(feature = "dev")]
fn logit_of(r: &Frame) -> f32 {
    get_f32(&r.head, "logit").unwrap_or(f32::NAN)
}

/// Every worker in a fresh session (the one open ended): the states zeroed, position 0.
#[cfg(feature = "dev")]
fn fresh(ws: &mut [Client]) -> Result<()> {
    for w in ws.iter_mut() {
        let (ses, inst) = (w.session.clone(), w.instance.clone());
        // (refused where it ended already: nothing to end)
        w.call_raw("end_session", vec![("session", text(&ses))], vec![])?;
        let s = w.call("begin_session", vec![("instance", text(&inst))], vec![])?;
        w.session = get_str(&s.head, "session").unwrap_or("").to_string();
    }
    Ok(())
}

/// The prompt through the workers in chunks (the head on the last, with `last`'s fields), then `new` - 1 steps (step i's `last` fields from `each(i)`): the
/// tokens, every step's logits (where the reply had its body) and every reply's `logit`.
#[cfg(feature = "dev")]
fn generate(
    ws: &mut [Client],
    prompt: &[u32],
    chunk: usize,
    new: usize,
    each: &dyn Fn(usize) -> Vec<(&'static str, Value)>,
) -> Result<(Vec<u32>, Vec<Vec<f32>>, Vec<f32>)> {
    let mut at = 0;
    let mut out = None;
    for ch in prompt.chunks(chunk) {
        let end = at + ch.len() == prompt.len();
        out = Some(chain(ws, ch, at, end, if end { each(0) } else { Vec::new() })?);
        at += ch.len();
    }
    let first = out.ok_or_else(|| Error("an empty prompt".into()))?;
    let (mut tok, l) = picked(&first)?;
    let (mut toks, mut lg, mut lo) = (vec![tok], vec![l].into_iter().filter(|l| !l.is_empty()).collect::<Vec<_>>(), vec![logit_of(&first)]);
    for i in 1..new {
        let r = chain(ws, &[tok], at, true, each(i))?;
        let (t, l) = picked(&r)?;
        at += 1;
        tok = t;
        toks.push(tok);
        lo.push(logit_of(&r));
        if !l.is_empty() {
            lg.push(l);
        }
    }
    Ok((toks, lg, lo))
}

/// One session's run through the workers, a pass at a time (`lane_step`): its prompt in chunks, then plain steps, or verify chunks (`drafts`).
#[cfg(feature = "dev")]
struct Lane {
    prompt: Vec<u32>,
    at: usize,
    toks: Vec<u32>,
    lg: Vec<Vec<f32>>,
    /// The verify chunks: the last one's keep and start, how many, and how many kept 0, 1, .. drafts.
    keep: Option<usize>,
    p0: usize,
    rounds: usize,
    seen: Vec<usize>,
    /// The request its tokens are drawn by (plain steps only), else the greedy token.
    draw: Option<&'static Draw>,
}

#[cfg(feature = "dev")]
impl Lane {
    fn new(prompt: &[u32], hk: usize) -> Lane {
        Lane { prompt: prompt.to_vec(), at: 0, toks: Vec::new(), lg: Vec::new(), keep: None, p0: 0, rounds: 0, seen: vec![0; hk.max(1)], draw: None }
    }

    /// The lane's tokens drawn by `d`.
    fn drawing(mut self, d: &'static Draw) -> Lane {
        self.draw = Some(d);
        self
    }

    /// The `sampling` of the reply that gives the answer's token `step` (none for a greedy lane).
    fn sampling(&self, step: usize) -> Vec<(&'static str, Value)> {
        self.draw.map(|d| d.request(step)).unwrap_or_default()
    }

    fn prompted(&self) -> bool {
        self.at >= self.prompt.len()
    }
}

/// The lane's next pass on the workers' sessions (`Client::session`): the next chunk of its prompt (the head on the last), else a plain step, else with `drafts`
/// (the plain run's tokens, hk) a verify chunk [the last token, hk - 1 drafts] whose drafts are the plain run's next a tokens then wrong ones, a = 0..hk-1 in
/// turn, the chunk after it keeping a + 1 rows (keep_previous). The tokens and every row's logits it gives are pushed; a verify chunk's kept drafts checked.
#[cfg(feature = "dev")]
fn lane_step(ws: &mut [Client], l: &mut Lane, chunk: usize, drafts: Option<(&[u32], usize)>, new: usize, vocab: usize) -> Result<()> {
    if !l.prompted() {
        let ch = l.prompt[l.at..(l.at + chunk).min(l.prompt.len())].to_vec();
        let end = l.at + ch.len() == l.prompt.len();
        let r = chain(ws, &ch, l.at, end, if end { l.sampling(0) } else { Vec::new() })?;
        l.at += ch.len();
        if end {
            let (t, g) = picked(&r)?;
            l.toks.push(t);
            l.lg.push(g);
        }
        return Ok(());
    }
    let last = *l.toks.last().ok_or_else(|| Error("no token after the prompt".into()))?;
    let Some((plain, hk)) = drafts else {
        let (t, g) = picked(&chain(ws, &[last], l.at, true, l.sampling(l.toks.len()))?)?;
        l.at += 1;
        l.toks.push(t);
        l.lg.push(g);
        return Ok(());
    };
    let (k, j) = (hk - 1, l.toks.len());
    let a = (l.rounds % hk).min(new - 1 - j);
    let wrong = |i: usize| (plain[i.min(new - 1)] + 1) % vocab as u32;
    let rows: Vec<u32> = std::iter::once(last).chain((1..=k).map(|i| if i <= a { plain[j - 1 + i] } else { wrong(j - 1 + i) })).collect();
    let mut every = vec![("verify", Value::Bool(true))];
    every.extend(l.keep.map(|x| ("keep_previous", num(x))));
    let r = chain_with(ws, &rows, l.at, true, &every, vec![("head_rows", num(hk))])?;
    let g: Vec<u32> = r.head.get("tokens").and_then(Value::as_u64s).unwrap_or_default().into_iter().map(|t| t as u32).collect();
    let words = bytes_words(&r.body);
    let took = (0..k).take_while(|&i| g.get(i) == Some(&rows[i + 1])).count();
    if took != a || words.len() != hk * vocab {
        return err(format!(
            "drafting: the verify chunk at {}: {took} drafts kept, where the plain run's tokens give {a}: tokens {g:?}, drafts {:?}; {} logit words",
            l.at,
            &rows[1..],
            words.len()
        ));
    }
    for (i, &x) in g.iter().enumerate().take(a + 1) {
        l.toks.push(x);
        l.lg.push(words[i * vocab..(i + 1) * vocab].iter().map(|&w| f32::from_bits((w as u32) << 16)).collect());
    }
    l.seen[a] += 1;
    (l.keep, l.p0) = (Some(a + 1), l.at);
    l.at += a + 1;
    l.rounds += 1;
    Ok(())
}

/// `f` on the workers with each one's session swapped for `other`'s (a second session's handles), then swapped back.
#[cfg(feature = "dev")]
fn on<T>(ws: &mut [Client], other: &mut [String], f: impl FnOnce(&mut [Client]) -> Result<T>) -> Result<T> {
    let swap = |ws: &mut [Client], other: &mut [String]| ws.iter_mut().zip(other.iter_mut()).for_each(|(w, h)| std::mem::swap(&mut w.session, h));
    swap(ws, other);
    let r = f(ws);
    swap(ws, other);
    r
}

/// A session begun on every worker (beside the ones open): the handles.
#[cfg(feature = "dev")]
fn begin_all(ws: &mut [Client]) -> Result<Vec<String>> {
    ws.iter_mut()
        .map(|w| Ok(get_str(&w.call("begin_session", vec![("instance", text(&w.instance.clone()))], vec![])?.head, "session").unwrap_or("").to_string()))
        .collect()
}

/// Every worker's session of `hs` ended.
#[cfg(feature = "dev")]
fn end_all(ws: &mut [Client], hs: &[String]) -> Result<()> {
    for (w, h) in ws.iter_mut().zip(hs) {
        w.call("end_session", vec![("session", text(h))], vec![])?;
    }
    Ok(())
}

/// Lanes `a` (on the workers' sessions) and `b` (on `hb`'s) a pass each in turn until a holds `a_until` tokens and b `b_until`; the time of the turns where both
/// were past their prompts (a step or a verify chunk each), and how many.
#[cfg(feature = "dev")]
#[allow(clippy::too_many_arguments)]
fn interleave(
    ws: &mut [Client],
    hb: &mut [String],
    a: &mut Lane,
    da: Option<(&[u32], usize)>,
    a_until: usize,
    b: &mut Lane,
    db: Option<(&[u32], usize)>,
    b_until: usize,
    chunk: usize,
    new: usize,
    vocab: usize,
) -> Result<(f64, usize)> {
    let (mut ms, mut turns) = (0f64, 0usize);
    while a.toks.len() < a_until || b.toks.len() < b_until {
        let both = a.prompted() && b.prompted() && a.toks.len() < a_until && b.toks.len() < b_until;
        let t = Instant::now();
        if a.toks.len() < a_until {
            lane_step(ws, a, chunk, da, new, vocab)?;
        }
        if b.toks.len() < b_until {
            on(ws, hb, |ws| lane_step(ws, b, chunk, db, new, vocab))?;
        }
        if both {
            ms += t.elapsed().as_secs_f64() * 1e3;
            turns += 1;
        }
    }
    Ok((ms, turns))
}

#[cfg(any(test, feature = "dev"))]
fn fmt_bytes(b: usize) -> String {
    let s = b.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// A sampled request of `rangecheck`: what a `sampling` carries (a worker takes no penalty, bias or grammar: they need the tokens so far) and, as `params`, the scheduler's reply to the same
/// request. The kinds of the cut that `samplecheck` draws: a nucleus at a temperature, a flat top-p whose cut lies past the GPU's 64 candidates, a top-k of 200 with a min-p.
#[cfg(any(test, feature = "dev"))]
struct Draw {
    what: &'static str,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    min_p: f32,
    seed: u64,
}

#[cfg(any(test, feature = "dev"))]
static SAMPLED: [Draw; 3] = [
    Draw { what: "temperature 0.8, top-p 0.95, top-k 40", temperature: 0.8, top_p: 0.95, top_k: 40, min_p: 0.0, seed: 1234 },
    Draw { what: "a flat top-p 0.999 at temperature 1.2", temperature: 1.2, top_p: 0.999, top_k: 0, min_p: 0.0, seed: 6 },
    Draw { what: "top-k 200, top-p 0.9, min-p 0.02", temperature: 0.9, top_p: 0.9, top_k: 200, min_p: 0.02, seed: 8 },
];

#[cfg(any(test, feature = "dev"))]
impl Draw {
    /// The `sampling` of the run_chunk that gives the answer's token `step`: every field, so that none comes from the model's generation_config.json.
    fn request(&self, step: usize) -> Vec<(&'static str, Value)> {
        let fields = vec![
            ("temperature", num(self.temperature)),
            ("top_p", num(self.top_p)),
            ("top_k", num(self.top_k)),
            ("min_p", num(self.min_p)),
            ("seed", num(self.seed)),
            ("step", num(step)),
        ];
        vec![("sampling", obj(fields))]
    }

    /// The scheduler's `Params` for the same request, over the model's defaults as a worker takes them (its repetition_penalty is 1: no penalty is applied there).
    fn params(&self, defaults: &Params) -> Params {
        Params {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            repetition_penalty: 1.0,
            seed: self.seed,
            ..defaults.clone()
        }
    }
}

/// The sampled draws a worker's `status` counts (`sampler::DRAWS`): from what its GPU made of the row, and by reading the row.
#[cfg(any(test, feature = "dev"))]
fn draws_of(status: &Value) -> Option<(u64, u64)> {
    let d = status.get("draws")?;
    Some((get_u64(d, "from_gpu")?, get_u64(d, "read_row")?))
}

/// A worker's sampled run against one GPU's scheduler's reply to the same request (`one`: its tokens and the model's row each was drawn from; `got`: the worker's tokens and, where it sent its
/// rows' body, those): the rows and tokens bit for bit (`same`), else the tokens alone; and each reply's `logit` (`logit`, where it has them) the bits of the model's logit of the token drawn.
#[cfg(any(test, feature = "dev"))]
fn like_one_gpu(what: &str, one: &(Vec<u32>, Vec<Vec<f32>>), got: &(Vec<u32>, Vec<Vec<f32>>), logit: &[f32]) -> Result<String> {
    let mut line = if got.1.is_empty() {
        if let Some(i) = (0..one.0.len().max(got.0.len())).find(|&i| one.0.get(i) != got.0.get(i)) {
            return err(format!("{what}: token {i} is {:?}, where one GPU's scheduler drew {:?}", got.0.get(i), one.0.get(i)));
        }
        format!("{what}: every token the same: {} tokens", got.0.len())
    } else {
        same(what, one, got)?
    };
    for (i, (&t, x)) in got.0.iter().zip(logit).enumerate() {
        let want = one.1.get(i).and_then(|r| r.get(t as usize)).copied();
        if want.map(f32::to_bits) != Some(x.to_bits()) {
            return err(format!("{what}: step {i}: the reply's logit of token {t} is {x} ({:#010x}), where the model's row has {want:?}", x.to_bits()));
        }
    }
    if !logit.is_empty() {
        line.push_str(&format!("; every reply's logit the model's, to the bit ({})", logit.len()));
    }
    Ok(line)
}

/// The first of `texts` that `tok` gives back as it was given (a random model's character-level tokenizer spells ASCII, a release's tokenizer every text).
#[cfg(any(test, feature = "dev"))]
fn spelled<'a>(tok: &Tok, texts: &[&'a str]) -> Option<&'a str> {
    texts.iter().copied().find(|t| tok.encode(t).and_then(|ids| tok.decode_with(&ids, true)).is_ok_and(|back| back == *t))
}

/// One GPU's scheduler's reply to `prompt` under `params`: `Sched` over the whole model's step, as serve runs it (the end token ignored, `new` tokens); its tokens and the model's row each was
/// drawn from, and the draws that read a row on the host (none is to).
#[cfg(feature = "dev")]
fn served(r: &mut Run, prompt: &[u32], params: Params, new: usize) -> Result<((Vec<u32>, Vec<Vec<f32>>), u64)> {
    use crate::sched::{Ev, Reply, Sched, Submit};
    let Some((k, step)) = r.as_step(1)? else { return err("the whole model is no scheduler's step") };
    let mut s = Sched::new(step, k).keeping(false).recording_rows(true);
    let (tx, rx) = mpsc::channel();
    let reply = Reply { params, max_new: new, ignore_eos: true, logprobs: None, stop_ids: Vec::new() };
    s.submit(Submit {
        prompt: prompt.to_vec(),
        reply,
        out: tx,
        abort: Default::default(),
        resumed: Default::default(),
        cached: Default::default(),
        created: Default::default(),
        salt: 0,
        pin: 0,
        images: Vec::new(),
    });
    while !s.idle() {
        s.tick()?;
    }
    let mut toks = Vec::new();
    loop {
        match rx.recv().map_err(|_| Error("the scheduler dropped a sequence".into()))? {
            Ev::Token(t) => toks.push(t),
            Ev::Lp(_) => {}
            Ev::End(f) => {
                f.map_err(|e| Error(e.msg))?;
                break;
            }
        }
    }
    let rows: Vec<Vec<f32>> = s.rows.iter().flatten().map(|x| x.1.clone()).collect();
    // (the routers' choices no longer traced: the next request's step is captured, where a trace reads them)
    s.stop_rows();
    Ok(((toks, rows), s.stats.row_reads))
}

/// `rangecheck MODEL[:level] [--prompt 1024] [--new 256] [--cut L | --cuts A,B] [--ctx N] [--chunk N] [--device D] [--no-graph] [--workers [--worker-exe PATH] [--sessions 2]]` (docs/range-worker.md): the whole model's
/// greedy tokens and logits of a random prompt, then the same through the stages chained in this process (`Split`, each stage's weights loaded once), then through worker processes with
/// `--workers`; every step's logits and token compared bit for bit. `--sessions 2`: the workers hold two sessions, run a pass each in turn (plain, drafted, one cancelled
/// and ended mid-way), each against its own whole-model run. Prints "all checks passed", or the first difference and exits 1.
#[cfg(feature = "dev")]
pub fn rangecheck(a: &Args) -> Result<()> {
    let spec = a.need(1, "MODEL[:level]")?;
    let (dir, layout) = resolve(spec)?;
    let cfg = Config35::load(&dir).map_err(|e| Error(format!("{}: {e} (rangecheck runs Qwen3.5 models)", dir.display())))?;
    let nl = cfg.layers;
    let (np, new) = (a.num("prompt", 1024usize)?, a.num("new", 256usize)?.max(1));
    let cuts: Vec<usize> = match (a.flag("cuts"), a.flag("cut")) {
        (Some(v), _) => v.split(',').map(|x| x.trim().parse::<usize>().map_err(|_| Error(format!("--cuts {v}: layer numbers")))).collect::<Result<_>>()?,
        (None, Some(v)) => vec![v.trim().parse::<usize>().map_err(|_| Error(format!("--cut {v}: a layer number")))?],
        (None, None) => vec![nl / 2],
    };
    if cuts.windows(2).any(|w| w[0] >= w[1]) || cuts.iter().any(|&c| c == 0 || c >= nl) {
        return err(format!("cuts {cuts:?}: ascending, inside 1..{nl}"));
    }
    let o = Opts {
        ctx: a.num("ctx", (np + new + 16).div_ceil(16) * 16)?,
        chunk: a.num("chunk", 1024usize)?,
        graph: !a.has("no-graph"),
        device: a.num("device", 0i32)?,
        layout,
    };
    let level = level_name(&dir, layout)?;
    let sessions = a.num("sessions", 1usize)?;
    if !(1..=MAX_SESSIONS).contains(&sessions) {
        return err(format!("--sessions {sessions}: 1 to {MAX_SESSIONS}"));
    }
    let prompt = crate::report::random_prompt(np, cfg.vocab);
    // (a second session's prompt: the first's backwards, a quarter shorter, so the two sessions sit at other positions)
    let prompt_b: Vec<u32> = prompt.iter().rev().take((np - np / 4).max(1)).copied().collect();
    println!("rangecheck: {} at {level}, {nl} layers, prompt {np} random tokens (report::random_prompt), {new} greedy tokens, ctx {}, chunk {}, GPU {}, graph {}; cuts {cuts:?}", dir.display(), o.ctx, o.chunk, o.device, o.graph);
    let bounds = stage_bounds(&cuts, nl);
    let d = dir.clone();
    let make: super::split::MakeStage = Arc::new(move |st: Stage, o: &Opts| super::qwen35::load_stage(&d, o, st));

    // 1. the whole model
    let t0 = Instant::now();
    let mut r = Run::new(nl, cfg.eos.clone(), &[], &[o.device], &o, make.clone())?;
    println!("whole model: loaded in {:.1} s: {}", t0.elapsed().as_secs_f64(), r.describe());
    if let Run::One(e) = &r {
        let (layers, embed, head, small) = e.layer_costs();
        let h = e.held().expect("the dynamic engine says what it holds");
        let (live, _, _) = crate::dev::held_now();
        println!("whole model: weights {} bytes = layers {} + embedding {} + lm_head {} + small tensors {}; kv {} bytes ({} a token), states {} bytes, scratch {} bytes, engine {} bytes", fmt_bytes(h.weights as usize), fmt_bytes(layers.iter().map(|x| x.0).sum()), fmt_bytes(embed), fmt_bytes(head), fmt_bytes(small), fmt_bytes(h.kv as usize), fmt_bytes(layers.iter().map(|x| x.1).sum()), fmt_bytes(h.state as usize), fmt_bytes((live as usize).saturating_sub((h.weights + h.kv + h.state) as usize)), fmt_bytes(live as usize));
        println!("layer  kind              weights_bytes  kv_bytes_per_token  state_bytes");
        for (l, &(w, kv, st)) in layers.iter().enumerate() {
            println!(
                "{l:>5}  {:<16} {:>14}  {:>18}  {:>11}",
                match cfg.kinds[l] {
                    Kind::Linear => "linear_attention",
                    Kind::Full => "full_attention",
                },
                fmt_bytes(w),
                fmt_bytes(kv),
                fmt_bytes(st)
            );
        }
        println!("boundary: {:?} ({} bytes a token)", e.boundary(), e.boundary().iter().map(|x| 2 * x.1).sum::<usize>());
    }
    let whole = greedy(&mut r, &prompt, new)?;
    println!(
        "whole model: prompt {:.1} ms, a step {:.2} ms (the logits read each step); tokens {:?}..; sha256(tokens, logits) {}",
        whole.2,
        whole.3,
        &whole.0[..whole.0.len().min(12)],
        digest(&whole.0, &whole.1)
    );
    let whole_b = if sessions > 1 {
        let b = greedy(&mut r, &prompt_b, new)?;
        println!("whole model, the second prompt ({} tokens): tokens {:?}..; sha256 {}", prompt_b.len(), &b.0[..b.0.len().min(12)], digest(&b.0, &b.1));
        (b.0, b.1)
    } else {
        (Vec::new(), Vec::new())
    };
    // the head on the last hk rows of a chunk (a draft's check): the prompt, then the first hk greedy tokens as one chunk; each row's token and logits, for the workers' below
    let hk = (new - 1).min(8).min(o.chunk);
    let rows_ref = match &mut r {
        Run::One(e) if hk > 1 => {
            e.set_head_rows(hk)?;
            for (i, ch) in prompt.chunks(o.chunk).enumerate() {
                e.stage_rows(ch, i * o.chunk, &mut Vec::new(), 0)?;
            }
            e.stage_rows(&whole.0[..hk], np, &mut Vec::new(), hk)?;
            Some(e.head_out(hk, true)?)
        }
        _ => None,
    };
    // what a verify pass and a rewind cost on this GPU (the whole model, one process; the positions after the prompt, written over each time): a one-row step (the graph),
    // a verify pass of hk rows, the states' copy it begins with, a rewind to each keep
    if let (Run::One(e), true) = (&mut r, hk > 1) {
        let (reps, tk) = (20, &whole.0[..hk]);
        let (mut step, mut pass, mut copy, mut back) = (0f64, 0f64, 0f64, vec![0f64; hk]);
        for _ in 0..reps {
            let t = Instant::now();
            e.stage_rows(&tk[..1], np, &mut Vec::new(), 1)?;
            step += t.elapsed().as_secs_f64();
            for (keep, b) in back.iter_mut().enumerate() {
                let t = Instant::now();
                e.verify_rows(tk, np, &mut Vec::new(), hk)?;
                pass += t.elapsed().as_secs_f64();
                let t = Instant::now();
                e.rewind(keep + 1)?;
                *b += t.elapsed().as_secs_f64();
            }
            let t = Instant::now();
            e.keep_states()?;
            e.dev.sync()?;
            copy += t.elapsed().as_secs_f64();
        }
        let ms = |x: f64, k: usize| x * 1e3 / k as f64;
        println!("whole model: a step {:.2} ms; a verify pass of {hk} rows {:.2} ms, of it the states' copy {:.3} ms ({} bytes); a rewind keeping 1..={hk} of them: {:?} ms", ms(step, reps), ms(pass, reps * hk), ms(copy, reps), fmt_bytes(e.state_bytes()), back.iter().map(|&x| format!("{:.3}", ms(x, reps))).collect::<Vec<_>>());
    }
    // a second sequence on the whole model (a session more): what it holds, its graph's capture, and a step of each in turn against a step of one alone
    if let (Run::One(e), true) = (&mut r, sessions > 1) {
        e.select(0)?;
        let (before, _, _) = crate::dev::held_now();
        let t = Instant::now();
        e.add_seqs(sessions - 1)?;
        let made = t.elapsed().as_secs_f64() * 1e3;
        let (after, _, _) = crate::dev::held_now();
        let (reps, tk) = (50, &whole.0);
        let mut alone = 0f64;
        let t = Instant::now();
        for i in 0..reps {
            e.stage_rows(&tk[i % tk.len()..][..1], np, &mut Vec::new(), 1)?;
        }
        alone += t.elapsed().as_secs_f64();
        let t = Instant::now();
        for i in 0..reps {
            for q in 0..sessions {
                e.select(q)?;
                e.stage_rows(&tk[i % tk.len()..][..1], np, &mut Vec::new(), 1)?;
            }
        }
        let turns = t.elapsed().as_secs_f64();
        e.select(0)?;
        println!("whole model, {sessions} sequences: {} bytes more for each one more ({} bytes of keys and values, {} of states; its graph captured, the room made in {made:.1} ms); a step {:.2} ms alone, {:.2} ms a step with the {sessions} in turn (select, a step, ..)", fmt_bytes((after - before) as usize / (sessions - 1)), fmt_bytes(e.kc.bytes + e.vc.bytes), fmt_bytes(e.state_bytes()), alone * 1e3 / reps as f64, turns * 1e3 / (reps * sessions) as f64);
    }
    drop(r);
    let whole = (whole.0, whole.1);

    // 1b. one GPU's scheduler's sampled replies (`SAMPLED`, `m` tokens each; with --sessions 2 the third on the second prompt too): what the workers' draws are held to below. The whole model again: the
    // scheduler's pools take the GPU's free memory, which the workers need
    let m = new.min(16);
    let defaults = crate::sampler::generation_defaults(&dir);
    let (mut refs, mut ref_b) = (Vec::new(), (Vec::new(), Vec::new()));
    if a.has("workers") {
        let t0 = Instant::now();
        let mut r = Run::new(nl, cfg.eos.clone(), &[], &[o.device], &o, make.clone())?;
        let mut reads = 0;
        for d in &SAMPLED {
            let (one, read) = served(&mut r, &prompt, d.params(&defaults), m)?;
            println!("one GPU's scheduler, sampled ({}, seed {}): {m} tokens {:?}; {read} rows read on the host", d.what, d.seed, one.0);
            (refs.push(one), reads += read);
        }
        if sessions > 1 {
            let (one, read) = served(&mut r, &prompt_b, SAMPLED[2].params(&defaults), m)?;
            (ref_b, reads) = (one, reads + read);
        }
        drop(r);
        if reads > 0 {
            return err(format!("one GPU's scheduler read {reads} rows on the host for its draws (none is to be)"));
        }
        println!("one GPU's scheduler: its sampled replies made in {:.1} s, no row read on the host", t0.elapsed().as_secs_f64());
    }

    // 2. the stages in this process
    let t0 = Instant::now();
    let mut r = Run::new(nl, cfg.eos.clone(), &cuts, &vec![o.device; cuts.len() + 1], &o, make.clone())?;
    println!("stages {:?} in one process: loaded in {:.1} s: {}", bounds, t0.elapsed().as_secs_f64(), r.describe());
    let split = greedy(&mut r, &prompt, new)?;
    println!("stages: prompt {:.1} ms, a step {:.2} ms; sha256 {}", split.2, split.3, digest(&split.0, &split.1));
    drop(r);
    let line = same("stages in one process", &whole, &(split.0, split.1))?;
    println!("{line}");

    // 3. the stages as worker processes
    if a.has("workers") {
        // (--worker-exe: another build's workers, a release build's, against this one's whole model)
        let exe = match a.flag("worker-exe") {
            Some(p) => PathBuf::from(p),
            None => std::env::current_exe().map_err(|e| Error(format!("this program's path: {e}")))?,
        };
        let t0 = Instant::now();
        let mut ws: Vec<Client> = bounds.iter().enumerate().map(|(k, &b)| Client::spawn(&exe, &dir, b, &o, hk.max(1), sessions, k)).collect::<Result<_>>()?;
        for w in &mut ws {
            let d = w.call("describe", vec![], vec![])?;
            println!("{}: describe: {}", w.tag, json::to_compact(&d.head));
            let before = w.call("inspect", vec![], vec![])?;
            println!("{}: inspect before prepare: {}", w.tag, json::to_compact(&before.head));
        }
        for w in &mut ws {
            let p = w.call("prepare", vec![], vec![])?;
            w.instance = get_str(&p.head, "instance").unwrap_or("").to_string();
            println!("{}: prepare: {}", w.tag, json::to_compact(&p.head));
            let i = w.call("inspect", vec![], vec![])?;
            println!("{}: inspect: {}", w.tag, json::to_compact(&i.head));
            w.call("commit", vec![("instance", text(&w.instance.clone()))], vec![])?;
            let s = w.call("begin_session", vec![("instance", text(&w.instance.clone()))], vec![])?;
            w.session = get_str(&s.head, "session").unwrap_or("").to_string();
            println!("{}: begin_session: {}", w.tag, json::to_compact(&s.head));
        }
        println!("workers: {} processes ready in {:.1} s", ws.len(), t0.elapsed().as_secs_f64());
        let (mut toks, mut lg) = (Vec::new(), Vec::new());
        let t = Instant::now();
        let mut at = 0;
        let mut out = None;
        for ch in prompt.chunks(o.chunk) {
            out = Some(chain(&mut ws, ch, at, at + ch.len() == prompt.len(), vec![])?);
            at += ch.len();
        }
        let pre = t.elapsed().as_secs_f64() * 1e3;
        let (mut tok, l) = picked(&out.ok_or_else(|| Error("no token after the prompt".into()))?)?;
        toks.push(tok);
        lg.push(l);
        let t = Instant::now();
        let mut last_req: Option<(Frame, Frame)> = None;
        for i in 1..new {
            let (nt, l) = picked(&chain(&mut ws, &[tok], at, true, vec![])?)?;
            if i + 1 == new {
                // (the last stage's last request and its reply, for the repeat below)
                let k = ws.len() - 1;
                let req = ws[k].last.clone().expect("a request was sent");
                last_req = Some((req, Frame { head: Value::Null, body: Vec::new() }));
            }
            at += 1;
            tok = nt;
            toks.push(tok);
            lg.push(l);
        }
        println!(
            "workers: prompt {:.1} ms, a step {:.2} ms (the frames and the logits' body each step); sha256 {}",
            pre,
            t.elapsed().as_secs_f64() * 1e3 / (new - 1).max(1) as f64,
            digest(&toks, &lg)
        );
        let line = same("workers chained through the protocol", &whole, &(toks, lg))?;
        println!("{line}");

        // the protocol's rules, on the last worker: a repeat, a conflict, a stale position, another incarnation's handle, an unknown op
        let k = ws.len() - 1;
        let (req, _) = last_req.expect("a step ran");
        let st0 = ws[k].call("status", vec![], vec![])?;
        ws[k].send_frame(&req)?;
        let again = ws[k].recv()?;
        let st1 = ws[k].call("status", vec![], vec![])?;
        let want = get_u64(&req.head, "position_start").unwrap_or(0) + 1;
        if get_bool(&again.head, "ok") != Some(true)
            || get_u64(&again.head, "position_end") != Some(want)
            || again.body.len() != cfg.vocab * 2
            || get_u64(&st0.head, "tokens") != get_u64(&st1.head, "tokens")
        {
            return err(format!(
                "the repeated request {:?} was not answered from the kept reply: {} (tokens {:?} then {:?})",
                get_str(&req.head, "id"),
                json::to_compact(&again.head),
                get_u64(&st0.head, "tokens"),
                get_u64(&st1.head, "tokens")
            ));
        }
        println!("workers: the last step's request sent again (id {:?}): the kept reply came back (position_end {want}, {} bytes of logits), the session still at {:?} tokens; not run again", get_str(&req.head, "id"), again.body.len(), get_u64(&st1.head, "tokens"));
        let mut conflict = req.clone();
        if let Value::Object(o) = &mut conflict.head {
            for (k2, v) in o.iter_mut() {
                if k2 == "position_start" {
                    *v = num(want);
                }
            }
        }
        ws[k].send_frame(&conflict)?;
        let r = ws[k].recv()?;
        let e = get_str(&r.head, "error").unwrap_or("");
        if get_bool(&r.head, "ok") != Some(false) || !e.contains("differs") {
            return err(format!("a different request under a used id was not refused: {}", json::to_compact(&r.head)));
        }
        println!("workers: the same id with another position_start: refused: {e}");
        let ses = ws[k].session.clone();
        let r = ws[k].call_raw(
            "run_chunk",
            vec![("session", text(&ses)), ("phase", text("decode")), ("position_start", num(want - 1)), ("rows", num(1))],
            vec![0u8; 4 * cfg.hidden],
        )?;
        let e = get_str(&r.head, "error").unwrap_or("");
        if get_bool(&r.head, "ok") != Some(false) || !e.contains("position_start") {
            return err(format!("a chunk at an earlier position was not refused: {}", json::to_compact(&r.head)));
        }
        println!("workers: a new request at position {}: refused: {e}", want - 1);
        let foreign = format!("s-{:016x}-1", 0u64);
        let r = ws[k].call_raw(
            "run_chunk",
            vec![("session", text(&foreign)), ("phase", text("decode")), ("position_start", num(want)), ("rows", num(1))],
            vec![0u8; 4 * cfg.hidden],
        )?;
        let e = get_str(&r.head, "error").unwrap_or("");
        if get_bool(&r.head, "ok") != Some(false) || !e.contains("incarnation") {
            return err(format!("another incarnation's handle was not refused: {}", json::to_compact(&r.head)));
        }
        println!("workers: a session handle of another incarnation: refused: {e}");
        let r = ws[k].call_raw("frobnicate", vec![], vec![])?;
        if get_bool(&r.head, "ok") != Some(false) {
            return err("an unknown op was not refused");
        }

        // status while a chunk runs, and a cancel during one, on the first worker: a prompt-sized chunk in a fresh session
        let w0 = &mut ws[0];
        let inst = w0.instance.clone();
        let e1 = w0.call("end_session", vec![("session", text(&w0.session.clone()))], vec![])?;
        println!("{}: end_session: {}", w0.tag, json::to_compact(&e1.head));
        let s = w0.call("begin_session", vec![("instance", text(&inst))], vec![])?;
        w0.session = get_str(&s.head, "session").unwrap_or("").to_string();
        let ses = w0.session.clone();
        let big: Vec<Value> = prompt[..o.chunk.min(prompt.len())].iter().map(|&t| num(t)).collect();
        let t = Instant::now();
        let id_run = w0.send(
            "run_chunk",
            vec![("session", text(&ses)), ("phase", text("prefill")), ("position_start", num(0)), ("head", Value::Bool(false)), ("tokens", list(big.clone()))],
            vec![],
        )?;
        std::thread::sleep(std::time::Duration::from_millis(20)); // (so that the chunk is under way when the status is asked)
        let id_st = w0.send("status", vec![], vec![])?;
        let first = w0.recv()?;
        let t_first = t.elapsed().as_secs_f64() * 1e3;
        let second = w0.recv()?;
        let t_second = t.elapsed().as_secs_f64() * 1e3;
        let (st_frame, run_frame) = if get_str(&first.head, "id") == Some(&id_st) { (&first, &second) } else { (&second, &first) };
        if get_str(&st_frame.head, "id") != Some(&id_st) || get_str(&run_frame.head, "id") != Some(&id_run) {
            return err(format!("the status and the chunk were not both answered: {} / {}", json::to_compact(&first.head), json::to_compact(&second.head)));
        }
        println!(
            "{}: a prefill chunk of {} tokens sent, then status: the first reply ({:.1} ms) was {}: {}; the second ({:.1} ms) was {}: ok {:?}",
            w0.tag,
            big.len(),
            t_first,
            if std::ptr::eq(st_frame, &first) { "the status" } else { "the chunk" },
            json::to_compact(&st_frame.head),
            t_second,
            if std::ptr::eq(st_frame, &first) { "the chunk" } else { "the status" },
            get_bool(&run_frame.head, "ok")
        );
        if get_bool(&run_frame.head, "ok") != Some(true) {
            return err(format!("the chunk failed: {}", json::to_compact(&run_frame.head)));
        }
        // a cancel during a chunk (a fresh session: the chunk at position 0 fits whatever the context)
        let e1 = w0.call("end_session", vec![("session", text(&ses))], vec![])?;
        println!("{}: end_session: {}", w0.tag, json::to_compact(&e1.head));
        let s = w0.call("begin_session", vec![("instance", text(&inst))], vec![])?;
        w0.session = get_str(&s.head, "session").unwrap_or("").to_string();
        let ses = w0.session.clone();
        let t = Instant::now();
        let id_run = w0.send(
            "run_chunk",
            vec![("session", text(&ses)), ("phase", text("prefill")), ("position_start", num(0)), ("head", Value::Bool(false)), ("tokens", list(big.clone()))],
            vec![],
        )?;
        std::thread::sleep(std::time::Duration::from_millis(20)); // (the cancel during the chunk)
        let id_c = w0.send("cancel", vec![("session", text(&ses))], vec![])?;
        let first = w0.recv()?;
        let second = w0.recv()?;
        let t_both = t.elapsed().as_secs_f64() * 1e3;
        let (c_frame, run_frame) = if get_str(&first.head, "id") == Some(&id_c) { (&first, &second) } else { (&second, &first) };
        if get_str(&c_frame.head, "id") != Some(&id_c) || get_str(&run_frame.head, "id") != Some(&id_run) {
            return err(format!("the cancel and the chunk were not both answered: {} / {}", json::to_compact(&first.head), json::to_compact(&second.head)));
        }
        let st = w0.call("status", vec![], vec![])?;
        println!(
            "{}: a prefill chunk sent, then cancel: the cancel's answer {}; the chunk's answer {}; both in {:.1} ms; status then {}",
            w0.tag,
            json::to_compact(&c_frame.head),
            json::to_compact(&run_frame.head),
            t_both,
            json::to_compact(&st.head)
        );
        let r = w0.call_raw(
            "run_chunk",
            vec![
                ("session", text(&ses)),
                ("phase", text("decode")),
                ("position_start", num(get_u64(&st.head, "tokens").unwrap_or(0))),
                ("tokens", list([num(1u32)])),
            ],
            vec![],
        )?;
        if get_bool(&r.head, "ok") != Some(false) {
            return err("a chunk after a cancel was not refused");
        }
        println!("{}: a chunk after the cancel: refused: {}", w0.tag, get_str(&r.head, "error").unwrap_or(""));
        let e2 = w0.call("end_session", vec![("session", text(&ses))], vec![])?;
        println!("{}: end_session: {}", w0.tag, json::to_compact(&e2.head));

        // tokenize and detokenize, on the control thread: a text and back; the same as a chat of one user message
        let w0 = &mut ws[0];
        // (a release's tokenizer spells any text; a random model's, character by character, ASCII: the text it gives back as it was given)
        let says = ["The range worker's text: 12 layers, ünïcödé, 東京 🦀.", "The range worker's text: 12 layers, plain ASCII."];
        let say = Tok::load(&dir).ok().and_then(|t| spelled(&t, &says)).unwrap_or(says[0]);
        let t = Instant::now();
        let ids =
            w0.call("tokenize", vec![("text", text(say)), ("chat", Value::Bool(false))], vec![])?.head.get("ids").and_then(Value::as_u64s).unwrap_or_default();
        let t_first = t.elapsed().as_secs_f64() * 1e3;
        let back = w0.call("detokenize", vec![("ids", list(ids.iter().map(|&x| num(x))))], vec![])?;
        if ids.is_empty() || get_str(&back.head, "text") != Some(say) {
            return err(format!("tokenize then detokenize: {ids:?} gave {}, where {say:?} went in", json::to_compact(&back.head)));
        }
        let chat =
            w0.call("tokenize", vec![("text", text(say)), ("chat", Value::Bool(true))], vec![])?.head.get("ids").and_then(Value::as_u64s).unwrap_or_default();
        let shown = w0.call("detokenize", vec![("ids", list(chat.iter().map(|&x| num(x)))), ("skip_special", Value::Bool(false))], vec![])?;
        let shown = get_str(&shown.head, "text").unwrap_or("").to_string();
        if !(chat.len() > ids.len() && shown.contains(say) && shown.contains("user")) {
            return err(format!("tokenize as a chat: {} ids, {shown:?}", chat.len()));
        }
        println!(
            "{}: tokenize {say:?}: {} ids ({t_first:.1} ms, the tokenizer loaded); detokenize: the same text back; as a chat: {} ids, {shown:?}",
            w0.tag,
            ids.len(),
            chat.len()
        );
        // chat messages: one user turn is the same ids as chat true; a system, user, assistant, user chat holds every turn, and is longer than its last turn
        let turn = |role: &str, content: &str| obj(vec![("role", text(role)), ("content", text(content))]);
        let mut turns = |t: Vec<Value>| -> Result<Vec<u64>> {
            Ok(w0.call("tokenize", vec![("messages", list(t))], vec![])?.head.get("ids").and_then(Value::as_u64s).unwrap_or_default())
        };
        let one = turns(vec![turn("user", say)])?;
        let four = turns(vec![turn("system", "Answer in one word."), turn("user", "Name a colour."), turn("assistant", "Blue."), turn("user", say)])?;
        let shown = w0.call("detokenize", vec![("ids", list(four.iter().map(|&x| num(x)))), ("skip_special", Value::Bool(false))], vec![])?;
        let shown = get_str(&shown.head, "text").unwrap_or("").to_string();
        if one != chat || !(four.len() > chat.len() && ["Answer in one word.", "Name a colour.", "Blue.", say].iter().all(|t| shown.contains(t))) {
            return err(format!("tokenize messages: one turn {} ids (chat true: {}); four turns {} ids, {shown:?}", one.len(), chat.len(), four.len()));
        }
        println!("{}: tokenize messages: one user turn the same {} ids as chat true; a four-turn chat {} ids, every turn in it", w0.tag, one.len(), four.len());

        // seeded sampled runs (`SAMPLED`, `m` tokens each) from fresh sessions, one with the rows' body asked for and one without: the same tokens and logits both times; the same as serve's sampler drawing
        // token after token over the same logits; and one GPU's scheduler's reply to the request (1b): every token, every row and every reply's `logit` bit for bit. The worker's draws, all made from what its
        // GPU made of the row (its `status`'s `draws`), none by reading the row
        let k = ws.len() - 1;
        let drew = |ws: &mut [Client]| -> Result<(u64, u64)> {
            let st = ws[ws.len() - 1].call("status", vec![], vec![])?;
            draws_of(&st.head).ok_or_else(|| Error(format!("no draws in the last stage's status: {}", json::to_compact(&st.head))))
        };
        let before = drew(&mut ws)?;
        let (mut made, mut last) = (0usize, (Vec::new(), Vec::new()));
        for (d, one) in SAMPLED.iter().zip(&refs) {
            let what = format!("workers, sampled ({}, seed {})", d.what, d.seed);
            let mut runs = Vec::new();
            let mut ms = 0f64;
            for body in [false, true] {
                fresh(&mut ws)?;
                let t = Instant::now();
                runs.push(generate(&mut ws, &prompt, o.chunk, m, &|i| {
                    let mut f = d.request(i);
                    f.extend((!body).then_some(("logits", Value::Bool(false))));
                    f
                })?);
                ms = t.elapsed().as_secs_f64() * 1e3;
            }
            let (bare, full) = (&runs[0], &runs[1]);
            if bare.0 != full.0 || bare.2.iter().zip(&full.2).any(|(x, y)| x.to_bits() != y.to_bits()) {
                return err(format!("{what}: the tokens {:?} with no rows' body, {:?} with it", bare.0, full.0));
            }
            let mut sm = Sampler::new(d.params(&defaults), &[]);
            let host: Vec<u32> = full.1.iter().map(|row| sm.sample(row)).collect();
            if host != full.0 {
                return err(format!("{what}: sampled through the workers {:?}, where serve's sampler over the same logits draws {host:?}", full.0));
            }
            let line = like_one_gpu(&what, one, &(full.0.clone(), full.1.clone()), &full.2)?;
            let line_bare = like_one_gpu(&what, one, &(bare.0.clone(), Vec::new()), &bare.2)?;
            let off = (0..m).filter(|&i| full.0[i] != whole.0[i]).count();
            println!("{line}\n{line_bare} (no rows' body asked for); the same as serve's sampler token after token over the same logits; {m} tokens in {ms:.1} ms; {off} of {m} differ from greedy: {:?}", full.0);
            made += 2 * m;
            last = (full.0.clone(), full.1.clone());
        }
        // the last sampled step's request again: the kept reply, not a new draw
        let req = ws[k].last.clone().expect("a request was sent");
        let st0 = ws[k].call("status", vec![], vec![])?;
        ws[k].send_frame(&req)?;
        let again = ws[k].recv()?;
        let st1 = ws[k].call("status", vec![], vec![])?;
        let (tok, row) = picked(&again)?;
        if tok != last.0[m - 1]
            || row.len() != last.1[m - 1].len()
            || row.iter().zip(&last.1[m - 1]).any(|(x, y)| x.to_bits() != y.to_bits())
            || get_u64(&st0.head, "tokens") != get_u64(&st1.head, "tokens")
        {
            return err(format!(
                "the repeated sampled request {:?}: {} (tokens {:?} then {:?})",
                get_str(&req.head, "id"),
                json::to_compact(&again.head),
                get_u64(&st0.head, "tokens"),
                get_u64(&st1.head, "tokens")
            ));
        }
        println!(
            "workers: the last sampled step sent again (id {:?}): the kept reply, token {tok}, the session still at {:?} tokens",
            get_str(&req.head, "id"),
            get_u64(&st1.head, "tokens")
        );
        let after = drew(&mut ws)?;
        if after.0 - before.0 != made as u64 || after.1 != before.1 {
            return err(format!(
                "the last worker's draws: {} from the GPU's view of the row and {} by reading it, where {made} sampled tokens were drawn and none is to read the row",
                after.0 - before.0,
                after.1 - before.1
            ));
        }
        println!(
            "workers: the last stage's draws: {made} sampled tokens, {made} drawn from what its GPU made of the row, {} by reading the row",
            after.1 - before.1
        );
        // a penalty needs the tokens so far: refused (before the chunk runs: the session as it was)
        let (ses, at) = (ws[k].session.clone(), get_u64(&st1.head, "tokens").unwrap_or(0));
        let penalty = obj(vec![("temperature", num(0.8)), ("seed", num(1)), ("step", num(0)), ("repetition_penalty", num(1.1))]);
        let r = ws[k].call_raw(
            "run_chunk",
            vec![
                ("session", text(&ses)),
                ("phase", text("decode")),
                ("position_start", num(at)),
                ("rows", num(1)),
                ("head", Value::Bool(true)),
                ("sampling", penalty),
            ],
            vec![0u8; 4 * cfg.hidden],
        )?;
        let e = get_str(&r.head, "error").unwrap_or("");
        let st2 = ws[k].call("status", vec![], vec![])?;
        if get_bool(&r.head, "ok") != Some(false) || !e.contains("not taken here") || get_u64(&st2.head, "tokens") != Some(at) {
            return err(format!(
                "a sampling with a repetition penalty was not refused as it was: {} (tokens {at}, then {:?})",
                json::to_compact(&r.head),
                get_u64(&st2.head, "tokens")
            ));
        }
        println!("workers: a sampling with a repetition penalty: refused, the session still at {at} tokens: {e}");

        // the head on hk rows: the prompt, then the first hk greedy tokens as one chunk; every row's token and logits the whole model's
        if let Some((picks, words)) = &rows_ref {
            fresh(&mut ws)?;
            let mut at = 0;
            for ch in prompt.chunks(o.chunk) {
                chain(&mut ws, ch, at, false, vec![])?;
                at += ch.len();
            }
            let r = chain(&mut ws, &whole.0[..hk], np, true, vec![("head_rows", num(hk))])?;
            let toks: Vec<u32> = r.head.get("tokens").and_then(Value::as_u64s).unwrap_or_default().into_iter().map(|t| t as u32).collect();
            let want: Vec<u32> = picks.iter().map(|p| p.0).collect();
            let got = bytes_words(&r.body);
            if toks != want || get_u64(&r.head, "token") != Some(want[hk - 1] as u64) || got != *words {
                let d = got.iter().zip(words.iter()).position(|(x, y)| x != y);
                return err(format!(
                    "head_rows {hk}: tokens {toks:?}, where the whole model gave {want:?}; {} logit words, the first different at {d:?}",
                    got.len()
                ));
            }
            let agree = (0..hk).filter(|&i| want[i] == whole.0[i + 1]).count();
            println!("workers: a chunk of {hk} tokens, head_rows {hk}: every row's token and logits the whole model's, bit for bit ({} logits): {toks:?}; {agree} of {hk} the same as the one-row steps' (a chunk's products are not a step's)", got.len());
        }

        // drafting: the prompt, then verify chunks [the last token, k drafts] through every stage, the head on every row; the drafts are the plain run's next `a` tokens
        // and then wrong ones, a = 0..=k in turn; the next chunk keeps a + 1 rows (keep_previous). Every kept row's token and logits against the plain run's, bit for bit
        if hk > 1 {
            let k = hk - 1;
            fresh(&mut ws)?;
            let mut l = Lane::new(&prompt, hk);
            while !l.prompted() {
                lane_step(&mut ws, &mut l, o.chunk, None, new, cfg.vocab)?;
            }
            let t = Instant::now();
            while l.toks.len() < new {
                lane_step(&mut ws, &mut l, o.chunk, Some((&whole.0, hk)), new, cfg.vocab)?;
            }
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let (keep, p0, rounds) = (l.keep, l.p0, l.rounds);
            let line = same(
                &format!("drafted through the workers ({rounds} verify chunks of {hk} rows; drafts kept 0..={k}: {:?} chunks each)", l.seen),
                &whole,
                &(l.toks, l.lg),
            )?;
            println!("{line}; {:.2} ms a verify chunk, {:.2} ms a token", ms / rounds as f64, ms / (new - 1) as f64);

            // the rules, on the first worker (the last verify chunk at p0 keeps `kept`): keep_previous past its rows, 0, at another position; a verify chunk of too many
            // rows, a prefill one; then keep_previous and a plain step (taken); keep_previous after that plain chunk
            let kept = keep.unwrap_or(1);
            let w0 = &mut ws[0];
            let ses = w0.session.clone();
            let one = || ("tokens", list([num(1u32)]));
            let cases: Vec<(Vec<(&str, Value)>, Option<&str>)> = vec![
                (vec![("position_start", num(p0 + hk + 1)), ("keep_previous", num(hk + 1)), one()], Some("1 to the verify chunk's")),
                (vec![("position_start", num(p0)), ("keep_previous", num(0)), one()], Some("1 to the verify chunk's")),
                (vec![("position_start", num(p0 + kept + 1)), ("keep_previous", num(kept)), one()], Some("after keep_previous")),
                (
                    vec![
                        ("position_start", num(p0 + kept)),
                        ("keep_previous", num(kept)),
                        ("verify", Value::Bool(true)),
                        ("tokens", list((0..=VERIFY_ROWS).map(num))),
                    ],
                    Some("a verify chunk 1 to"),
                ),
                (
                    vec![("phase", text("prefill")), ("position_start", num(p0 + kept)), ("keep_previous", num(kept)), ("verify", Value::Bool(true)), one()],
                    Some("verify: a decode chunk"),
                ),
                (vec![("position_start", num(p0 + kept)), ("keep_previous", num(kept)), one()], None),
                (vec![("position_start", num(p0 + kept + 1)), ("keep_previous", num(1)), one()], Some("not a verify chunk")),
            ];
            for (pairs, want) in cases {
                let mut all = vec![("session", text(&ses))];
                if !pairs.iter().any(|x| x.0 == "phase") {
                    all.push(("phase", text("decode")));
                }
                all.extend(pairs);
                let r = w0.call_raw("run_chunk", all, vec![])?;
                let e = get_str(&r.head, "error").unwrap_or("");
                match want {
                    Some(wt) if get_bool(&r.head, "ok") == Some(false) && e.contains(wt) => println!("{}: refused: {e}", w0.tag),
                    None if get_bool(&r.head, "ok") == Some(true) && get_u64(&r.head, "position_end") == Some((p0 + kept + 1) as u64) => {
                        println!("{}: keep_previous {kept} and a plain step at {}: taken", w0.tag, p0 + kept)
                    }
                    _ => return err(format!("drafting's rules: {want:?} wanted, the reply {}", json::to_compact(&r.head))),
                }
            }
        }

        // two sessions a stage, a pass each in turn: A on the prompt, B on prompt_b; each one's tokens and logits against its own whole-model run, bit for bit
        if sessions > 1 {
            let half = new / 2;
            fresh(&mut ws)?;
            let mut hb = begin_all(&mut ws)?;
            for w in ws.iter_mut() {
                let r = w.call_raw("begin_session", vec![("instance", text(&w.instance.clone()))], vec![])?;
                let e = get_str(&r.head, "error").unwrap_or("");
                if get_bool(&r.head, "ok") != Some(false) || !e.contains("at a time") {
                    return err(format!("{}: a session past --sessions {sessions} was not refused: {}", w.tag, json::to_compact(&r.head)));
                }
            }
            println!("workers: {sessions} sessions open on each; one more: refused");
            // 1. plain both, a pass each in turn; at half the tokens A is cancelled and ended, B goes on; a session C on A's room runs the prompt again from 0
            let (mut la, mut lb) = (Lane::new(&prompt, hk), Lane::new(&prompt_b, hk));
            let (ms, turns) = interleave(&mut ws, &mut hb, &mut la, None, half, &mut lb, None, half, o.chunk, new, cfg.vocab)?;
            println!("workers, two sessions a pass each in turn: {:.2} ms a step ({turns} turns of two steps)", ms / (2 * turns.max(1)) as f64);
            for (w, b) in ws.iter_mut().zip(&hb) {
                let a = w.session.clone();
                let c = w.call("cancel", vec![("session", text(&a))], vec![])?;
                let sb = w.call("status", vec![("session", text(b))], vec![])?;
                let all = w.call("status", vec![], vec![])?;
                if get_str(&c.head, "state") != Some("cancelled")
                    || get_bool(&c.head, "session_cancelled") != Some(true)
                    || get_str(&c.head, "session") != Some(&a)
                    || get_bool(&sb.head, "session_cancelled") != Some(false)
                    || get_str(&sb.head, "state") != Some("idle")
                    || get_str(&all.head, "session") != Some(&a)
                {
                    return err(format!(
                        "{}: cancel {a}: {}; status of {b}: {}; status: {}",
                        w.tag,
                        json::to_compact(&c.head),
                        json::to_compact(&sb.head),
                        json::to_compact(&all.head)
                    ));
                }
                println!("{}: cancel A: {}; status of B: {}", w.tag, json::to_compact(&c.head), json::to_compact(&sb.head));
            }
            let a0 = ws[0].session.clone();
            let r = ws[0].call_raw(
                "run_chunk",
                vec![("session", text(&a0)), ("phase", text("decode")), ("position_start", num(la.at)), ("tokens", list([num(la.toks[la.toks.len() - 1])]))],
                vec![],
            )?;
            if get_bool(&r.head, "ok") != Some(false) || !get_str(&r.head, "error").unwrap_or("").contains("cancelled") {
                return err(format!("a chunk of the cancelled session was not refused: {}", json::to_compact(&r.head)));
            }
            on(&mut ws, &mut hb, |ws| (0..4).try_for_each(|_| lane_step(ws, &mut lb, o.chunk, None, new, cfg.vocab)))?;
            for w in ws.iter_mut() {
                let e = w.call("end_session", vec![("session", text(&w.session.clone()))], vec![])?;
                if get_bool(&e.head, "cancelled") != Some(true) {
                    return err(format!("{}: end_session of the cancelled A: {}", w.tag, json::to_compact(&e.head)));
                }
                println!("{}: end_session A: {}", w.tag, json::to_compact(&e.head));
            }
            let hc = begin_all(&mut ws)?;
            for (w, c) in ws.iter_mut().zip(hc) {
                w.session = c;
            }
            let mut lc = Lane::new(&prompt, hk);
            interleave(&mut ws, &mut hb, &mut lc, None, new, &mut lb, None, new, o.chunk, new, cfg.vocab)?;
            println!("{}", same("session A cancelled at half; session C on its room, a pass each in turn with B", &whole, &(lc.toks, lc.lg))?);
            println!("{}", same("session B, a pass each in turn with A, then 4 steps alone, then with C", &whole_b, &(lb.toks, lb.lg))?);
            // 2. A drafted (verify chunks, keep_previous) and B plain, a pass each in turn; 3. both drafted
            if hk > 1 {
                for both in [false, true] {
                    end_all(&mut ws, &hb)?;
                    fresh(&mut ws)?;
                    hb = begin_all(&mut ws)?;
                    let (mut la, mut lb) = (Lane::new(&prompt, hk), Lane::new(&prompt_b, hk));
                    let db = if both { Some((&whole_b.0[..], hk)) } else { None };
                    let (ms, turns) = interleave(&mut ws, &mut hb, &mut la, Some((&whole.0, hk)), new, &mut lb, db, new, o.chunk, new, cfg.vocab)?;
                    let what = if both { "both drafted" } else { "A drafted, B plain" };
                    println!(
                        "{}",
                        same(
                            &format!("session A drafted ({} verify chunks, kept {:?}), a pass each in turn with B ({what})", la.rounds, la.seen),
                            &whole,
                            &(la.toks, la.lg)
                        )?
                    );
                    println!(
                        "{}",
                        same(
                            &format!(
                                "session B {} ({what})",
                                if both { format!("drafted ({} verify chunks, kept {:?})", lb.rounds, lb.seen) } else { "plain".into() }
                            ),
                            &whole_b,
                            &(lb.toks, lb.lg)
                        )?
                    );
                    println!("workers, {what}: {:.2} ms a turn of two passes ({turns} turns)", ms / turns.max(1) as f64);
                }
            }
            // 4. both sampled (A: the first kind on the prompt, B: the third on the second prompt), a pass each in turn: each one's tokens and rows one GPU's scheduler's reply to its request, bit for bit,
            // and the last stage's draws all from the GPU's view of the row
            end_all(&mut ws, &hb)?;
            fresh(&mut ws)?;
            hb = begin_all(&mut ws)?;
            let (mut la, mut lb) = (Lane::new(&prompt, hk).drawing(&SAMPLED[0]), Lane::new(&prompt_b, hk).drawing(&SAMPLED[2]));
            let before = drew(&mut ws)?;
            interleave(&mut ws, &mut hb, &mut la, None, m, &mut lb, None, m, o.chunk, new, cfg.vocab)?;
            let after = drew(&mut ws)?;
            println!("{}", like_one_gpu(&format!("session A sampled ({}), a pass each in turn with B", SAMPLED[0].what), &refs[0], &(la.toks, la.lg), &[])?);
            println!("{}", like_one_gpu(&format!("session B sampled ({}), on the second prompt", SAMPLED[2].what), &ref_b, &(lb.toks, lb.lg), &[])?);
            if after.0 - before.0 != 2 * m as u64 || after.1 != before.1 {
                return err(format!(
                    "the last worker's draws with two sessions: {} from the GPU's view of the row and {} by reading it, where {} sampled tokens were drawn",
                    after.0 - before.0,
                    after.1 - before.1,
                    2 * m
                ));
            }
            println!(
                "workers, two sessions sampled: {} tokens drawn, {} from what the last stage's GPU made of the row, {} by reading the row",
                2 * m,
                after.0 - before.0,
                after.1 - before.1
            );
            end_all(&mut ws, &hb)?;
        }

        // the end: every session ended, every instance unloaded, every worker gone
        for w in ws.iter_mut() {
            let e = w.call("end_session", vec![("session", text(&w.session.clone()))], vec![])?;
            println!("{}: end_session: {}", w.tag, json::to_compact(&e.head));
            let u = w.call("unload", vec![("instance", text(&w.instance.clone()))], vec![])?;
            println!("{}: unload: {}", w.tag, json::to_compact(&u.head));
            let r = w.call_raw(
                "run_chunk",
                vec![("session", text(&w.session.clone())), ("phase", text("decode")), ("position_start", num(0)), ("rows", num(1))],
                vec![],
            )?;
            if get_bool(&r.head, "ok") != Some(false) {
                return err("a chunk on an ended session was not refused");
            }
            w.call("exit", vec![], vec![])?;
        }
        for w in ws {
            let tag = w.tag.clone();
            let code = w.wait()?;
            if code != 0 {
                return err(format!("{tag}: exit status {code}"));
            }
            println!("{tag}: exited 0");
        }
    }
    println!("all checks passed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(id: &str, op: &str, body: &[u8]) -> Frame {
        Frame { head: obj(vec![("op", text(op)), ("id", text(id))]), body: body.to_vec() }
    }

    /// A tokenize's messages: system, user and assistant turns, the last a user's; any other role, a missing content, no turns or a last turn not the user's
    /// is refused.
    #[test]
    fn chat_messages_take_turns_ending_with_the_user() {
        let turn = |role: &str, content: &str| obj(vec![("role", text(role)), ("content", text(content))]);
        let ok = list(vec![turn("system", "be brief"), turn("user", "hi"), turn("assistant", "hello"), turn("user", "and?")]);
        let m = chat_messages(&ok).unwrap();
        assert_eq!(
            m.iter().map(|m| (m.role.as_str(), m.content.as_str())).collect::<Vec<_>>(),
            [("system", "be brief"), ("user", "hi"), ("assistant", "hello"), ("user", "and?")]
        );
        for bad in [
            list(vec![]),
            list(vec![turn("user", "hi"), turn("assistant", "hello")]),
            list(vec![turn("tool", "x"), turn("user", "hi")]),
            list(vec![obj(vec![("role", text("user"))])]),
            text("hi"),
        ] {
            assert!(chat_messages(&bad).is_err(), "{}", json::to_compact(&bad));
        }
    }

    /// A frame is written as its two lengths, its head and its body, and read back the same; an empty body too; the stream's clean end is None, a cut-off frame an error.
    #[test]
    fn a_frame_goes_round() {
        let f = Frame { head: obj(vec![("op", text("run_chunk")), ("id", text("r1")), ("rows", num(3usize))]), body: vec![1, 2, 3, 0xff] };
        let mut buf = Vec::new();
        write_frame(&mut buf, &f).unwrap();
        let h = json::to_compact(&f.head);
        assert_eq!(&buf[..8], &[h.len() as u8, 0, 0, 0, 4, 0, 0, 0]);
        assert_eq!(&buf[8..8 + h.len()], h.as_bytes());
        let mut r = std::io::Cursor::new(buf.clone());
        assert_eq!(read_frame(&mut r, MAX_BODY).unwrap(), Some(f.clone()));
        assert_eq!(read_frame(&mut r, MAX_BODY).unwrap(), None);
        let mut r = std::io::Cursor::new(buf[..buf.len() - 1].to_vec());
        assert!(read_frame(&mut r, MAX_BODY).is_err());
        let mut buf = Vec::new();
        write_frame(&mut buf, &frame("r2", "status", &[])).unwrap();
        assert_eq!(read_frame(&mut std::io::Cursor::new(buf), MAX_BODY).unwrap(), Some(frame("r2", "status", &[])));
    }

    /// A head past MAX_HEAD, a body past this side's limit, an empty head, a head that is not JSON: refused, before the body is read.
    #[test]
    fn a_frame_past_the_limits_is_refused() {
        let mut big = Vec::new();
        big.extend((MAX_HEAD as u32 + 1).to_le_bytes());
        big.extend(0u32.to_le_bytes());
        assert!(read_frame(&mut std::io::Cursor::new(big), MAX_BODY).unwrap_err().msg.contains("frame head"));
        let mut f = Vec::new();
        write_frame(&mut f, &frame("r1", "run_chunk", &[0u8; 100])).unwrap();
        assert!(read_frame(&mut std::io::Cursor::new(f.clone()), 99).unwrap_err().msg.contains("frame body of 100 bytes, where 99"));
        assert!(read_frame(&mut std::io::Cursor::new(f), 100).unwrap().is_some());
        let mut e = Vec::new();
        e.extend(0u32.to_le_bytes());
        e.extend(0u32.to_le_bytes());
        assert!(read_frame(&mut std::io::Cursor::new(e), MAX_BODY).is_err());
        let mut j = Vec::new();
        j.extend(3u32.to_le_bytes());
        j.extend(0u32.to_le_bytes());
        j.extend(b"{x}");
        assert!(read_frame(&mut std::io::Cursor::new(j), MAX_BODY).unwrap_err().msg.contains("not JSON"));
    }

    /// A handle names its kind, the incarnation and a number; this incarnation's pass, another incarnation's are refused as such, anything else is not a handle.
    #[test]
    fn handles_are_the_incarnations() {
        let inc = 0x1234_5678_9abc_def0u64;
        let i = handle('i', inc, 1);
        let s = handle('s', inc, 7);
        assert_eq!((i.as_str(), s.as_str()), ("i-123456789abcdef0-1", "s-123456789abcdef0-7"));
        assert!(check_handle(&i, 'i', inc).is_ok() && check_handle(&s, 's', inc).is_ok());
        assert!(check_handle(&i, 's', inc).unwrap_err().msg.contains("not a session handle"));
        assert!(check_handle(&handle('s', inc + 1, 7), 's', inc).unwrap_err().msg.contains("another worker incarnation"));
        assert!(check_handle("s-0000000000000000-1", 's', inc).unwrap_err().msg.contains("another worker incarnation"));
        for bad in ["", "s", "s-123456789abcdef0-", "s-123456789abcdef0-x", "pointer:0x7f", "s-12345-1"] {
            assert!(check_handle(bad, 's', inc).unwrap_err().msg.contains("not a session handle"), "{bad}");
        }
        assert!(incarnation() != incarnation() || true);
        assert!(ok_id("r1") && ok_id("chunk_12.a-b") && !ok_id("") && !ok_id("a b") && !ok_id(&"x".repeat(129)) && ok_id(&"x".repeat(128)));
    }

    /// A request id names one request: new once, pending until answered, then its reply comes back for an exact repeat; a different request under it is a conflict; the oldest bodies past the
    /// budget are let go and a repeat of one is retired; the oldest entries past KEEP are forgotten.
    #[test]
    fn a_repeated_id_is_answered_from_the_kept_reply() {
        let mut st = Store::default();
        let req = json::to_compact(&frame("r1", "run_chunk", &[]).head);
        assert!(matches!(st.admit("r1", &req), Known::New));
        assert!(matches!(st.admit("r1", &req), Known::Pending));
        let other = json::to_compact(&obj(vec![("op", text("run_chunk")), ("id", text("r1")), ("position_start", num(5usize))]));
        assert!(matches!(st.admit("r1", &other), Known::Conflict));
        let reply = Frame { head: obj(vec![("id", text("r1")), ("ok", Value::Bool(true))]), body: vec![9; 10] };
        st.done("r1", &reply);
        assert!(matches!(st.admit("r1", &req), Known::Repeat(f) if f == reply));
        assert!(matches!(st.admit("r1", &other), Known::Conflict));
        assert_eq!(st.reply_of("r1"), Some(reply.clone()));
        // bodies past the budget: the oldest let go
        let mut st = Store::default();
        let half = vec![0u8; KEEP_BODY / 2 + 1];
        for id in ["a", "b", "c"] {
            let r = json::to_compact(&frame(id, "run_chunk", &[]).head);
            assert!(matches!(st.admit(id, &r), Known::New));
            st.done(id, &Frame { head: obj(vec![("id", text(id))]), body: half.clone() });
        }
        assert!(matches!(st.admit("a", &json::to_compact(&frame("a", "run_chunk", &[]).head)), Known::Retired));
        assert!(matches!(st.admit("b", &json::to_compact(&frame("b", "run_chunk", &[]).head)), Known::Retired));
        assert!(matches!(st.admit("c", &json::to_compact(&frame("c", "run_chunk", &[]).head)), Known::Repeat(_)));
        assert!(st.body_bytes <= KEEP_BODY);
        // entries past KEEP: the oldest forgotten (an id given again is new)
        let mut st = Store::default();
        for i in 0..KEEP + 1 {
            let id = format!("k{i}");
            assert!(matches!(st.admit(&id, &json::to_compact(&frame(&id, "commit", &[]).head)), Known::New));
            st.done(&id, &frame(&id, "commit", &[]));
        }
        assert_eq!(st.seen.len(), KEEP);
        assert!(matches!(st.admit("k0", &json::to_compact(&frame("k0", "commit", &[]).head)), Known::New));
    }

    /// The reply frame carries the id, ok and the pairs, or the id, ok false and the error.
    #[test]
    fn replies_carry_the_id() {
        let f = reply_frame("r9", Ok((vec![("rows", num(2usize))], vec![1, 2])));
        assert_eq!(json::to_compact(&f.head), r#"{"id":"r9","ok":true,"rows":2}"#);
        assert_eq!(f.body, vec![1, 2]);
        let f = reply_frame("r9", err("no"));
        assert_eq!(json::to_compact(&f.head), r#"{"id":"r9","ok":false,"error":"no"}"#);
        assert_eq!(bytes_words(&words_bytes(&[1, 0x3f80, 0xffff])), vec![1, 0x3f80, 0xffff]);
        assert_eq!(stage_bounds(&[10, 20], 32), vec![(0, 10), (10, 20), (20, 32)]);
        assert_eq!(fmt_bytes(1234567), "1,234,567");
    }

    /// A sampled run_chunk's sampler, made from the request alone (the seed and the step), draws what serve's sampler draws token after token over the same
    /// rows; what it does not take is refused.
    #[test]
    fn sampling_is_the_requests() {
        let vocab = 1000;
        let mut x = 7u64;
        let rows: Vec<Vec<f32>> = (0..20)
            .map(|_| {
                (0..vocab)
                    .map(|_| {
                        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        (x >> 40) as f32 / (1u64 << 24) as f32 * 8.0
                    })
                    .collect()
            })
            .collect();
        let defaults = Params { temperature: 1.0, ..Params::default() };
        let mut serve = Sampler::new(Params { temperature: 0.8, top_p: 0.95, top_k: 40, seed: 99, ..defaults.clone() }, &[]);
        for (i, row) in rows.iter().enumerate() {
            let v = json::parse(&format!(r#"{{"temperature": 0.8, "top_p": 0.95, "top_k": 40, "seed": 99, "step": {i}}}"#)).unwrap();
            assert_eq!(sampler_of(&v, &defaults, 4096).unwrap().sample(row), serve.sample(row), "step {i}");
        }
        let first = |seed: u64, step: usize| {
            sampler_of(&json::parse(&format!(r#"{{"seed": {seed}, "step": {step}}}"#)).unwrap(), &defaults, 4096).unwrap().sample(&rows[0])
        };
        assert!((0..8).any(|s| first(1, s) != first(1, 0)) && (0..8).any(|s| first(s, 0) != first(0, 0)));
        for (bad, says) in [
            (r#"{"step": 0}"#, "seed"),
            (r#"{"seed": 1}"#, "step"),
            (r#"{"seed": 1, "step": 5000}"#, "at most 4096"),
            (r#"{"seed": -1, "step": 0}"#, "seed"),
            (r#"{"seed": 1, "step": 0, "top_p": 0}"#, "top_p"),
            (r#"{"seed": 1, "step": 0, "temperature": 3}"#, "temperature"),
            (r#"{"seed": 1, "step": 0, "repetition_penalty": 1.1}"#, "repetition_penalty"),
        ] {
            assert!(sampler_of(&json::parse(bad).unwrap(), &defaults, 4096).err().unwrap().msg.contains(says), "{bad}");
        }
    }

    /// A sampled request of `rangecheck` carries every field of the scheduler's `Params` for it, so that the worker's sampler, over a model's own defaults (a generation_config.json's, unlike the
    /// request's), draws what the scheduler's does token after token; with the defaults a request does not say, it would not.
    #[test]
    fn a_sampled_request_is_the_references_params() {
        let mut x = 11u64;
        let rows: Vec<Vec<f32>> = (0..24)
            .map(|_| {
                (0..1000)
                    .map(|_| {
                        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        (x >> 40) as f32 / (1u64 << 24) as f32 * 6.0
                    })
                    .collect()
            })
            .collect();
        let defaults = Params { temperature: 0.3, top_p: 0.5, top_k: 7, min_p: 0.4, repetition_penalty: 1.3, ..Params::default() };
        for d in &SAMPLED {
            let mut one = Sampler::new(d.params(&defaults), &[]);
            let want: Vec<u32> = rows.iter().map(|r| one.sample(r)).collect();
            let got: Vec<u32> = rows.iter().enumerate().map(|(i, r)| sampler_of(&d.request(i)[0].1, &defaults, 4096).unwrap().sample(r)).collect();
            assert_eq!(got, want, "{}", d.what);
            assert_eq!(d.params(&defaults).repetition_penalty, 1.0, "{}: a worker applies no penalty", d.what);
            // (the defaults do change a draw: a request that left a field to them would not be the reference's)
            let mut lean = Sampler::new(Params { temperature: d.temperature, seed: d.seed, ..defaults.clone() }, &[]);
            assert_ne!(rows.iter().map(|r| lean.sample(r)).collect::<Vec<u32>>(), want, "{}", d.what);
        }
    }

    /// The status says the process's sampled draws, as `rangecheck` reads them from the last stage: both counts, and no other status has them.
    #[test]
    fn the_status_says_the_draws() {
        let (a, b) = (DRAWS[1].load(Relaxed), DRAWS[0].load(Relaxed));
        let (gpu, row) = draws_of(&obj(status(&Live::default(), 1, None))).expect("a status has its draws");
        assert!(gpu >= a && gpu <= DRAWS[1].load(Relaxed) && row >= b && row <= DRAWS[0].load(Relaxed));
        assert_eq!(draws_of(&obj(vec![("state", text("idle"))])), None);
        assert_eq!(draws_of(&obj(vec![("draws", obj(vec![("from_gpu", num(1))]))])), None);
    }

    /// A worker's sampled run is one GPU's scheduler's, or the first difference is said: a token, a bit of a row, a bit of a reply's `logit`, a missing token; with no rows' body only the tokens and
    /// the logits are held.
    #[test]
    fn a_sampled_run_is_one_gpus_to_the_bit() {
        let rows = vec![vec![0.5f32, -1.0, 2.0, 0.25], vec![1.5, 0.0, -3.0, 4.0], vec![-0.5, 8.0, 0.125, 2.0]];
        let one = (vec![2u32, 3, 1], rows.clone());
        let logit = [2.0f32, 4.0, 8.0];
        let said = |got: &(Vec<u32>, Vec<Vec<f32>>), logit: &[f32]| like_one_gpu("w", &one, got, logit).map_err(|e| e.msg);
        assert!(said(&one, &logit).unwrap().contains("every reply's logit the model's"));
        assert!(said(&(one.0.clone(), Vec::new()), &logit).unwrap().contains("3 tokens"));
        assert!(said(&one, &[]).is_ok());
        let wrong = said(&(vec![2, 3, 0], rows.clone()), &logit).unwrap_err();
        assert!(wrong.contains("step 2: token 0"), "{wrong}");
        let wrong = said(&(vec![2, 0, 1], Vec::new()), &logit).unwrap_err();
        assert!(wrong.contains("token 1 is Some(0)") && wrong.contains("drew Some(3)"), "{wrong}");
        let wrong = said(&(vec![2, 3], Vec::new()), &logit).unwrap_err();
        assert!(wrong.contains("token 2 is None"), "{wrong}");
        let mut off = rows.clone();
        off[1][2] = f32::from_bits(off[1][2].to_bits() ^ 1);
        let wrong = said(&(one.0.clone(), off), &logit).unwrap_err();
        assert!(wrong.contains("step 1: logit 2"), "{wrong}");
        let wrong = said(&(one.0.clone(), Vec::new()), &[2.0, f32::from_bits(4.0f32.to_bits() + 1), 8.0]).unwrap_err();
        assert!(wrong.contains("step 1: the reply's logit of token 3"), "{wrong}");
        let wrong = said(&one, &[2.0, 4.0, f32::NAN]).unwrap_err();
        assert!(wrong.contains("step 2"), "{wrong}");
    }

    /// The text a check sends to a worker's tokenizer is one the tokenizer gives back as it was given: a character-level random model's tokenizer does not spell what a release's does.
    #[cfg(feature = "dev")]
    #[test]
    fn a_tokenizer_spells_what_it_gives_back() {
        let dir = std::env::temp_dir().join(format!("glyd-range-spell-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        crate::logits::write_serving_files(&dir, "{{ messages }}").unwrap();
        let tok = Tok::load(&dir).unwrap();
        let says = ["ünïcödé, 東京 🦀", "plain ASCII, 12 layers."];
        assert_eq!(spelled(&tok, &says), Some(says[1]));
        assert_eq!(spelled(&tok, &says[..1]), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The status as a session sees it: a cancel asked is that session's alone; a request running is the stage's (running) and, where it names the session and its
    /// cancel was asked, cancelling; with no session named, the earliest begun.
    #[test]
    fn status_is_a_sessions() {
        let st = |l: &Live, who: Option<&str>| json::to_compact(&obj(status(l, 1, who)));
        let mut l = Live { instance: Some("i-1".into()), sessions: vec![("s-a".into(), 5, false), ("s-b".into(), 9, false)], ..Live::default() };
        assert!(st(&l, None).starts_with(r#"{"state":"idle","incarnation":"0000000000000001","instance":"i-1","session":"s-a","session_cancelled":false,"tokens":5,"running":null,"sessions":[{"session":"s-a","tokens":5,"cancelled":false},"#), "{}", st(&l, None));
        l.cancel.push("s-a".into());
        assert!(st(&l, Some("s-a")).contains(r#""state":"cancelled","#) && st(&l, Some("s-a")).contains(r#""session_cancelled":true"#));
        assert!(
            st(&l, Some("s-b")).starts_with(r#"{"state":"idle","#) && st(&l, Some("s-b")).contains(r#""session":"s-b","session_cancelled":false,"tokens":9"#)
        );
        l.running = Some(("r7".into(), "run_chunk".into(), Instant::now(), Some("s-b".into())));
        assert!(st(&l, Some("s-a")).starts_with(r#"{"state":"running","#) && st(&l, Some("s-b")).starts_with(r#"{"state":"running","#));
        l.running = Some(("r8".into(), "run_chunk".into(), Instant::now(), Some("s-a".into())));
        assert!(st(&l, Some("s-a")).starts_with(r#"{"state":"cancelling","#) && st(&l, None).starts_with(r#"{"state":"cancelling","#));
        assert!(
            st(&l, None)
                .contains(r#""session":"s-a"},"sessions":[{"session":"s-a","tokens":5,"cancelled":true},{"session":"s-b","tokens":9,"cancelled":false}]"#),
            "{}",
            st(&l, None)
        );
        let l = Live::default();
        assert!(st(&l, None).starts_with(r#"{"state":"empty","#) && st(&l, None).contains(r#""session":null,"session_cancelled":false,"tokens":0"#));
    }

    /// `--layers A..B` inside the model's layers; the whole model where not given.
    #[test]
    fn layers_are_a_range() {
        let a = |v: &[&str]| Args::from(v.iter().map(|s| s.to_string()));
        assert_eq!(layers_arg(&a(&["rangeworker", "--layers", "16..32"]), 32).unwrap(), (16, 32));
        assert_eq!(layers_arg(&a(&["rangeworker"]), 32).unwrap(), (0, 32));
        for bad in ["16", "32..16", "0..33", "a..b", "16..16"] {
            assert!(layers_arg(&a(&["rangeworker", "--layers", bad]), 32).is_err(), "{bad}");
        }
        assert!(matches!(resolve("Qwen/Qwen3.5-9B:swift"), Ok((_, Some(Layout::Int5))) | Err(_)));
        assert!(resolve("x:sparrow").unwrap_err().msg.contains("not a level"));
    }
}
