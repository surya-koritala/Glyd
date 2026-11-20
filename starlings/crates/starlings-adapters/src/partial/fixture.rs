//! Development-only stand-in for `glyd-engine rangeworker` (glyd-range/1 over stdio): a
//! two-layer, hidden-4, vocab-64 "model" with zero boundaries and a fixed sampled token.
//! A prompt containing "HANG" makes the first stage hang inside that run_chunk; "SLOW" makes
//! it take 6.5 s, after which it reads a pending cancel like the real engine. "REPEAT" switches
//! the sampled token to `repeat` of the tokens the session holds (the first stage writes each
//! row's token into its boundary row), so drafts sometimes miss and a wrong rewind shows.
use serde_json::{Value, json};
use std::io::{Read, Write};

const HIDDEN: u64 = 4;
const VOCAB: u32 = 64;
const HANG: u32 = 63;
const SLOW: u32 = 62;
/// Never made from text: marks a "REPEAT" prompt.
const REPEAT: u32 = 0;

/// The next token after `held`: mostly the token three back, a new one every seventh position.
fn repeat(held: &[u32]) -> u32 {
    let n = held.len();
    if n >= 3 && !n.is_multiple_of(7) {
        held[n - 3]
    } else {
        (n as u32 * 5) % 50 + 2
    }
}

fn read(input: &mut impl Read) -> Option<(Value, Vec<u8>)> {
    let mut prefix = [0; 8];
    input.read_exact(&mut prefix).ok()?;
    let head = u32::from_le_bytes(prefix[..4].try_into().ok()?) as usize;
    let body = u32::from_le_bytes(prefix[4..].try_into().ok()?) as usize;
    let mut bytes = vec![0; head + body];
    input.read_exact(&mut bytes).ok()?;
    let body = bytes.split_off(head);
    Some((serde_json::from_slice(&bytes).ok()?, body))
}
fn write(output: &mut impl Write, mut head: Value, id: &Value, body: &[u8]) -> Option<()> {
    head["id"] = id.clone();
    head["ok"] = true.into();
    let head = serde_json::to_vec(&head).ok()?;
    output.write_all(&(head.len() as u32).to_le_bytes()).ok()?;
    output.write_all(&(body.len() as u32).to_le_bytes()).ok()?;
    output.write_all(&head).ok()?;
    output.write_all(body).ok()?;
    output.flush().ok()
}

/// One open session's state (`--sessions` of them at most).
struct Seq {
    handle: String,
    position: u64,
    held: Vec<u32>,
    rewind: Option<u64>,
    cancelled: bool,
}
/// Put the working state back into its session.
fn save(
    open: &mut [Seq],
    at: Option<usize>,
    (position, held, rewind, cancelled): (u64, &mut Vec<u32>, Option<u64>, bool),
) {
    if let Some(s) = at.and_then(|i| open.get_mut(i)) {
        s.position = position;
        s.held = std::mem::take(held);
        s.rewind = rewind;
        s.cancelled = cancelled;
    }
}

pub fn run() {
    let args: Vec<String> = std::env::args().collect();
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
            .unwrap_or_default()
    };
    let layers: Vec<u64> = value("--layers")
        .split("..")
        .filter_map(|n| n.parse().ok())
        .collect();
    let (ctx, chunk): (u64, u64) = (
        value("--ctx").parse().unwrap_or(0),
        value("--chunk").parse().unwrap_or(0),
    );
    let (first, last) = (layers.first() == Some(&0), layers.last() == Some(&2));
    let incarnation = format!("{:016x}", std::process::id());
    let (mut input, mut output) = (std::io::stdin().lock(), std::io::stdout().lock());
    let most: usize = value("--sessions").parse().unwrap_or(1);
    let (mut open, mut begun) = (Vec::<Seq>::new(), 0u64);
    // The named session's working state: its position, the tokens at its positions, the last
    // chunk's start when it can be rewound, and whether it was cancelled.
    let (mut position, mut cancelled) = (0u64, false);
    let (mut held, mut rewind) = (Vec::<u32>::new(), None::<u64>);
    while let Some((request, body)) = read(&mut input) {
        let id = &request["id"];
        // A call names its session; status without one reports the earliest open one.
        let at = match request["session"].as_str() {
            Some(handle) => match open.iter().position(|s| s.handle == handle) {
                Some(i) => Some(i),
                None => return,
            },
            None => (request["op"] == "status" && !open.is_empty()).then_some(0),
        };
        if let Some(s) = at.and_then(|i| open.get_mut(i)) {
            (position, cancelled, rewind) = (s.position, s.cancelled, s.rewind);
            held = std::mem::take(&mut s.held);
        }
        let reply = match request["op"].as_str().unwrap_or("") {
            "describe" => json!({"protocol": "glyd-range/1", "build": {"version": "fixture"},
                "incarnation": incarnation, "pid": std::process::id(),
                "model": {"dir": value("--model"), "family": "Fixture", "layers": 2,
                    "hidden": HIDDEN, "vocab": VOCAB, "level": "fixture"},
                "stage": {"layers": layers, "first": first, "last": last},
                "boundary": {"dtype": "bf16", "bytes_per_row": 4 * HIDDEN},
                "state": {"reset": "fixture"},
                "limits": {"sessions": most, "context_tokens": ctx, "chunk_tokens": chunk,
                    "max_head_bytes": 1 << 20, "max_body_bytes": 1 << 20},
                "ops": ["describe", "prepare", "commit", "begin_session", "run_chunk",
                    "status", "end_session", "tokenize", "detokenize"],
                "device": {"name": "fixture"}}),
            // Each session's room is a byte of keys and values and a byte of state.
            "prepare" => json!({"instance": "i-1", "weights_bytes": 1, "kv_bytes": most,
                "state_bytes": most, "scratch_bytes": 1, "engine_bytes": 2 + 2 * most,
                "peak_bytes": 2 + 2 * most,
                "device": {"name": "fixture", "free_bytes": 1u64 << 30, "total_bytes": 2u64 << 30}}),
            "commit" => json!({"committed": true, "ready": true}),
            "tokenize" => {
                // A chat's turns read as their contents, one per line.
                let joined = request["messages"].as_array().map(|turns| {
                    turns
                        .iter()
                        .filter_map(|t| t["content"].as_str())
                        .collect::<Vec<_>>()
                        .join("\n")
                });
                let text = joined
                    .as_deref()
                    .unwrap_or(request["text"].as_str().unwrap_or(""));
                let ids: Vec<u32> = if text.contains("HANG") {
                    vec![1, HANG]
                } else if text.contains("SLOW") {
                    vec![1, SLOW]
                } else if text.contains("REPEAT") {
                    [1, REPEAT]
                        .into_iter()
                        .chain(text.bytes().map(|b| u32::from(b) % 60 + 2))
                        .collect()
                } else {
                    std::iter::once(1)
                        .chain(text.bytes().map(|b| u32::from(b) % 60 + 2))
                        .collect()
                };
                json!({"ids": ids})
            }
            "detokenize" => {
                let ids = request["ids"].as_array().cloned().unwrap_or_default();
                let text: String = ids
                    .iter()
                    .filter_map(|id| id.as_u64())
                    .map(|id| char::from(b'a' + (id % 26) as u8))
                    .collect();
                json!({"text": text})
            }
            "begin_session" => {
                if open.len() >= most {
                    return;
                }
                begun += 1;
                let handle = format!("s-{begun}");
                open.push(Seq {
                    handle: handle.clone(),
                    position: 0,
                    held: Vec::new(),
                    rewind: None,
                    cancelled: false,
                });
                json!({"session": handle, "context_tokens": ctx, "tokens": 0,
                    "kv_bytes": 1, "state_bytes": 1})
            }
            "run_chunk" => {
                // Explicit fault-injection barrier, compiled only in the development fixture.
                // The test arms a phase, observes this engine's PID, then kills or releases it.
                if let Ok(directory) = std::env::var("STARLINGS_TEST_PHASE_GATE") {
                    let directory = std::path::Path::new(&directory);
                    if let Ok(phase) = std::fs::read_to_string(directory.join("arm"))
                        && request["phase"].as_str() == Some(phase.trim())
                    {
                        std::fs::write(directory.join("reached"), std::process::id().to_string())
                            .unwrap();
                        let deadline =
                            std::time::Instant::now() + std::time::Duration::from_secs(10);
                        while directory.join("arm").exists() && std::time::Instant::now() < deadline
                        {
                            std::thread::sleep(std::time::Duration::from_millis(10));
                        }
                    }
                }
                // A chunk continues the session, or keeps `keep_previous` rows of a multi-row
                // decode chunk and starts after them.
                let start = request["position_start"].as_u64().unwrap_or(0);
                let keep = request["keep_previous"].as_u64();
                match (rewind, keep) {
                    (_, None) if start == position => {}
                    (Some(from), Some(keep)) if start == from + keep && start < position => {
                        held.truncate(start as usize);
                    }
                    _ => return,
                }
                let rows = if first {
                    let tokens = request["tokens"].as_array().cloned().unwrap_or_default();
                    held.extend(tokens.iter().filter_map(|t| t.as_u64()).map(|t| t as u32));
                    if tokens.iter().any(|t| t.as_u64() == Some(u64::from(HANG))) {
                        loop {
                            std::thread::sleep(std::time::Duration::from_secs(60));
                        }
                    }
                    if tokens.iter().any(|t| t.as_u64() == Some(u64::from(SLOW))) {
                        std::thread::sleep(std::time::Duration::from_millis(6500));
                    }
                    tokens.len() as u64
                } else {
                    request["rows"].as_u64().unwrap_or(0)
                };
                if !first && body.len() as u64 != rows * 4 * HIDDEN {
                    return;
                }
                // Each row's token rides in the first bytes of its residual row.
                let row = |i: u64| (i * 2 * HIDDEN) as usize;
                if !first {
                    held.extend(
                        (0..rows).map(|i| {
                            u32::from_le_bytes(body[row(i)..row(i) + 4].try_into().unwrap())
                        }),
                    );
                }
                position = start + rows;
                // As the engine: a multi-row decode chunk is a `verify` chunk, the only kind a
                // later chunk may rewind.
                let verify = request["verify"] == true;
                if verify != (request["phase"] == "decode" && rows > 1) {
                    return;
                }
                rewind = verify.then_some(start);
                if !last {
                    let mut bytes = vec![0; (rows * 4 * HIDDEN) as usize];
                    for i in 0..rows {
                        let token = held[(start + i) as usize];
                        bytes[row(i)..row(i) + 4].copy_from_slice(&token.to_le_bytes());
                    }
                    save(&mut open, at, (position, &mut held, rewind, cancelled));
                    let reply = json!({"rows": rows, "position_end": position,
                        "body": {"what": "boundary", "rows": rows, "bytes": bytes.len()}});
                    if write(&mut output, reply, id, &bytes).is_none() {
                        return;
                    }
                    continue;
                }
                if request["head"] == true {
                    let next = |end: u64| match held.get(1) {
                        Some(&REPEAT) => repeat(&held[..end as usize]),
                        _ => 5,
                    };
                    let mut reply = json!({"rows": rows, "position_end": position, "token": next(position), "logit": 0.5});
                    if let Some(k) = request["head_rows"].as_u64() {
                        let tokens: Vec<u32> = (position - k + 1..=position).map(next).collect();
                        reply["tokens"] = json!(tokens);
                    }
                    reply
                } else {
                    json!({"rows": rows, "position_end": position})
                }
            }
            "status" => {
                let named = at.map(|i| open[i].handle.clone());
                json!({"state": "idle", "incarnation": incarnation, "instance": "i-1",
                    "session": named, "session_cancelled": named.is_some() && cancelled,
                    "tokens": if named.is_some() { position } else { 0 }, "running": null,
                    "sessions": open.iter().map(|s| json!({"session": s.handle,
                        "tokens": s.position, "cancelled": s.cancelled})).collect::<Vec<_>>()})
            }
            "cancel" => {
                cancelled = true;
                json!({"state": "cancelling", "incarnation": incarnation, "instance": "i-1",
                    "session": request["session"], "session_cancelled": true, "tokens": position,
                    "running": null})
            }
            "end_session" => {
                let reply = json!({"ended": true, "tokens": position, "cancelled": cancelled});
                open.retain(|s| Some(s.handle.as_str()) != request["session"].as_str());
                reply
            }
            _ => return,
        };
        save(
            &mut open,
            at.filter(|_| request["op"] != "end_session"),
            (position, &mut held, rewind, cancelled),
        );
        if write(&mut output, reply, id, &[]).is_none() {
            return;
        }
    }
}
