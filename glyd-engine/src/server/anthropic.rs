//! Anthropic's Messages API: `POST /v1/messages` and `/v1/messages/count_tokens`. A request is turned into the chat request it is the same as (`openai_of`: the system prompt a message of its own, an assistant's `tool_use`
//! blocks its tool calls, a `tool_result` block a tool message, the tools as functions, `stop_sequences` as `stop`, `thinking` the template's `enable_thinking`), made ready and run as any chat request is (`start_chat`:
//! its refusals, its grammar, its tools, its window), and the reply is written as a message, or as the events of a stream: `message_start`, `ping`, the content blocks (`text`, `tool_use`, `thinking`) with their
//! `content_block_start`, `content_block_delta` and `content_block_stop`, `message_delta` and `message_stop`. Errors are Anthropic's: `{"type": "error", "error": {"type", "message"}}`.
//!
//! Prompt caching (`cache_control`) is read and has no effect (P5e); the usage says no token was cached. A thinking block carries no signature (this server cannot make one): it is empty.

use super::*;
use tools::obj;

/// An Anthropic Messages request in the OpenAI form the rest of the server reads: the system prompt a message of its own, an assistant's tool_use blocks its tool calls, a tool_result block a tool message, the tools
/// as functions (the tools Anthropic runs, `web_search` and the like, are not the model's to call here: left out), `tool_choice` as OpenAI names it, `stop_sequences` as `stop`, and `thinking` the template's
/// enable_thinking (off unless it is enabled).
pub fn openai_of(v: &Value) -> std::result::Result<Value, String> {
    openai_of_with(v, crate::vision::PREP.get().is_some())
}

/// `openai_of`, its image blocks image_url parts where `images` (the model reads them), else left out.
pub fn openai_of_with(v: &Value, images: bool) -> std::result::Result<Value, String> {
    let text_of = |c: &Value| -> String {
        match c {
            Value::String(t) => t.clone(),
            Value::Array(b) => b
                .iter()
                .filter(|x| x.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|x| x.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        }
    };
    let mut msgs = Vec::new();
    if let Some(sys) = v.get("system").map(text_of).filter(|t| !t.is_empty()) {
        msgs.push(obj(vec![("role", Value::from("system")), ("content", Value::String(sys))]));
    }
    let list = v.get("messages").and_then(Value::as_array).ok_or("messages: Field required")?;
    if list.is_empty() {
        return Err("messages: at least one message is required".to_string());
    }
    for m in list {
        let role = m.get("role").and_then(Value::as_str).ok_or("messages: a message with no role")?;
        if role != "user" && role != "assistant" {
            return Err(format!("messages: a message with the role `{role}`: Input should be 'user' or 'assistant'"));
        }
        let blocks = match m.get("content") {
            Some(Value::Array(b)) => b.as_slice(),
            Some(c @ Value::String(_)) => {
                msgs.push(obj(vec![("role", Value::from(role)), ("content", c.clone())]));
                continue;
            }
            _ => return Err(format!("messages: a {role} message whose content is neither text nor a list of blocks")),
        };
        let (mut text, mut calls) = (String::new(), Vec::new());
        // (an image block: an image_url part, the message's content then the list of its text and image parts in order)
        let mut parts: Vec<Value> = Vec::new();
        for b in blocks {
            let get = |k: &str| b.get(k).cloned().unwrap_or(Value::Null);
            match b.get("type").and_then(Value::as_str) {
                Some("text") => {
                    let t = b.get("text").and_then(Value::as_str).unwrap_or("");
                    text.push_str(t);
                    parts.push(obj(vec![("type", Value::from("text")), ("text", Value::from(t))]));
                }
                // (where the model reads images; a text model's server leaves them out, as it always has)
                Some("image") if images => {
                    let src = b.get("source");
                    let field = |k: &str| src.and_then(|s| s.get(k)).and_then(Value::as_str).unwrap_or("");
                    let url = match field("type") {
                        "base64" => format!("data:{};base64,{}", field("media_type"), field("data")),
                        "url" => field("url").to_string(),
                        t => return Err(format!("an image block whose source is of type {t:?}: base64 or url")),
                    };
                    parts.push(obj(vec![("type", Value::from("image_url")), ("image_url", obj(vec![("url", Value::String(url))]))]));
                }
                Some("tool_use") => calls.push(obj(vec![
                    ("id", get("id")),
                    ("type", Value::from("function")),
                    ("function", obj(vec![("name", get("name")), ("arguments", b.get("input").cloned().unwrap_or(Value::Object(vec![])))])),
                ])),
                Some("tool_result") => {
                    let mut content = b.get("content").map(text_of).unwrap_or_default();
                    if b.get("is_error").and_then(Value::as_bool) == Some(true) && !content.is_empty() {
                        content = format!("Error: {content}");
                    }
                    msgs.push(obj(vec![("role", Value::from("tool")), ("tool_call_id", get("tool_use_id")), ("content", Value::String(content))]))
                }
                // (`thinking` and `redacted_thinking` blocks of an earlier turn: the model reads its answers, not the thinking behind them; documents and the rest are not read)
                _ => {}
            }
        }
        let has_image = parts.iter().any(|p| p.get("type").and_then(Value::as_str) == Some("image_url"));
        if !text.is_empty() || !calls.is_empty() || has_image {
            let mut o = vec![("role", Value::from(role)), ("content", if has_image { Value::Array(parts) } else { Value::String(text) })];
            if !calls.is_empty() {
                o.push(("tool_calls", Value::Array(calls)));
            }
            msgs.push(obj(o));
        }
    }
    let mut o = vec![("messages", Value::Array(msgs))];
    let tools: Vec<Value> = v
        .get("tools")
        .and_then(Value::as_array)
        .unwrap_or(&[])
        .iter()
        // (a custom tool has a name and a schema; the ones Anthropic runs have a versioned type, `web_search_20250305`: they are not here)
        .filter(|t| t.get("name").is_some() && t.get("type").and_then(Value::as_str).map_or(true, |ty| ty == "custom"))
        .map(|t| {
            let get = |k: &str, or: Value| t.get(k).cloned().unwrap_or(or);
            obj(vec![
                ("type", Value::from("function")),
                (
                    "function",
                    obj(vec![
                        ("name", get("name", Value::Null)),
                        ("description", get("description", Value::from(""))),
                        ("parameters", get("input_schema", Value::Object(vec![]))),
                    ]),
                ),
            ])
        })
        .collect();
    if !tools.is_empty() {
        o.push(("tools", Value::Array(tools)));
        let choice = v.get("tool_choice");
        let named = choice.and_then(|c| c.get("name")).cloned();
        match choice.and_then(|c| c.get("type")).and_then(Value::as_str) {
            Some("any") => o.push(("tool_choice", Value::from("required"))),
            Some("none") => o.push(("tool_choice", Value::from("none"))),
            Some("tool") if named.is_some() => {
                o.push(("tool_choice", obj(vec![("type", Value::from("function")), ("function", obj(vec![("name", named.unwrap_or(Value::Null))]))])))
            }
            _ => {}
        }
        if choice.and_then(|c| c.get("disable_parallel_tool_use")).and_then(Value::as_bool) == Some(true) {
            o.push(("parallel_tool_calls", Value::Bool(false)));
        }
    }
    for k in ["max_tokens", "temperature", "top_p", "top_k", "stream"] {
        if let Some(x) = v.get(k) {
            o.push((k, x.clone()));
        }
    }
    if let Some(x) = v.get("stop_sequences") {
        o.push(("stop", x.clone()));
    }
    o.push(("chat_template_kwargs", obj(vec![("enable_thinking", Value::Bool(thinking_on(v)))])));
    Ok(obj(o))
}

/// Whether the request asks for the model's thinking (`thinking: {"type": "enabled" | "adaptive"}`).
fn thinking_on(v: &Value) -> bool {
    matches!(v.get("thinking").and_then(|t| t.get("type")).and_then(Value::as_str), Some("enabled" | "adaptive"))
}

/// An error as Anthropic's API writes it, by the status: `{"type": "error", "error": {"type", "message"}}`.
pub(super) fn anthropic_body(code: u16, msg: &str) -> String {
    let kind = match code {
        400 | 405 | 422 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        503 | 529 => "overloaded_error",
        _ => "api_error",
    };
    format!("{{\"type\":\"error\",\"error\":{{\"type\":\"{kind}\",\"message\":{}}}}}", jstr(&said(msg)))
}

fn fail(s: &mut TcpStream, cors: &str, code: u16, msg: &str) {
    respond_with(s, code, "application/json", cors, anthropic_body(code, msg).as_bytes());
}

/// An event of a streamed Messages reply, as one chunk of the chunked body.
fn event(s: &mut TcpStream, kind: &str, fields: &str) -> bool {
    let ev = format!("event: {kind}\ndata: {{\"type\":\"{kind}\"{}{fields}}}\n\n", if fields.is_empty() { "" } else { "," });
    s.write_all(format!("{:x}\r\n{ev}\r\n", ev.len()).as_bytes()).and_then(|_| s.flush()).is_ok()
}

/// A tool call's id as Anthropic's are: `toolu_` and the id's digits.
fn tool_id(id: &str) -> String {
    format!("toolu_{}", id.strip_prefix("chatcmpl-tool-").unwrap_or(id))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Open {
    Text,
    Thinking,
    Tool,
}

/// The content blocks of a streamed Messages reply as its deltas come: the thinking (where the request asks for it) in a thinking block, text in a text block, each tool call a tool_use block of its own, its
/// arguments as `input_json_delta`s. A block of another kind ends the open one.
#[derive(Default)]
struct Blocks {
    next: usize,
    open: Option<Open>,
    /// A tool_use block was sent (the reply's stop_reason is then tool_use, a named tool's too, as Anthropic's API has it).
    called: bool,
}

impl Blocks {
    fn close(&mut self, s: &mut TcpStream) -> bool {
        let Some(open) = self.open.take() else { return true };
        // (a thinking block ends with its signature, which this server has no way to make: empty)
        let signed = open != Open::Thinking
            || event(s, "content_block_delta", &format!("\"index\":{},\"delta\":{{\"type\":\"signature_delta\",\"signature\":\"\"}}", self.next));
        self.next += 1;
        signed && event(s, "content_block_stop", &format!("\"index\":{}", self.next - 1))
    }

    fn begin(&mut self, s: &mut TcpStream, kind: Open, block: &str) -> bool {
        if self.open == Some(kind) {
            return true;
        }
        let ok = self.close(s) && event(s, "content_block_start", &format!("\"index\":{},\"content_block\":{block}", self.next));
        self.open = Some(kind);
        ok
    }

    fn delta(&mut self, s: &mut TcpStream, d: &Delta) -> bool {
        let mut ok = true;
        if !d.reasoning.is_empty() {
            ok &= self.begin(s, Open::Thinking, "{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}");
            ok &= event(
                s,
                "content_block_delta",
                &format!("\"index\":{},\"delta\":{{\"type\":\"thinking_delta\",\"thinking\":{}}}", self.next, jstr(&d.reasoning)),
            );
        }
        if !d.content.is_empty() {
            ok &= self.begin(s, Open::Text, "{\"type\":\"text\",\"text\":\"\"}");
            ok &= event(s, "content_block_delta", &format!("\"index\":{},\"delta\":{{\"type\":\"text_delta\",\"text\":{}}}", self.next, jstr(&d.content)));
        }
        for c in &d.calls {
            match c {
                ToolEv::Start { id, name, .. } => {
                    // (every call is a block of its own, even where one follows another)
                    ok &= self.close(s)
                        && event(
                            s,
                            "content_block_start",
                            &format!(
                                "\"index\":{},\"content_block\":{{\"type\":\"tool_use\",\"id\":{},\"name\":{},\"input\":{{}}}}",
                                self.next,
                                jstr(&tool_id(id)),
                                jstr(name)
                            ),
                        );
                    self.open = Some(Open::Tool);
                    self.called = true;
                }
                ToolEv::Args { args, .. } if self.open == Some(Open::Tool) => {
                    ok &= event(
                        s,
                        "content_block_delta",
                        &format!("\"index\":{},\"delta\":{{\"type\":\"input_json_delta\",\"partial_json\":{}}}", self.next, jstr(args)),
                    );
                }
                _ => {}
            }
        }
        ok
    }
}

/// How the reply ended as Anthropic names it: the stop string that did it, a call, the length, or the end of the turn.
fn stop_reason(done: &Done, called: bool) -> &'static str {
    if called {
        "tool_use"
    } else if done.stop.is_some() {
        "stop_sequence"
    } else if done.finish_reason == "length" {
        "max_tokens"
    } else {
        "end_turn"
    }
}

fn usage(input: usize, output: usize) -> String {
    format!("{{\"input_tokens\":{input},\"output_tokens\":{output},\"cache_creation_input_tokens\":0,\"cache_read_input_tokens\":0}}")
}

/// `max_tokens` as Anthropic's API wants it: there, and a whole number of at least 1.
fn max_tokens_ok(v: &Value) -> std::result::Result<(), String> {
    match v.get("max_tokens") {
        None | Some(Value::Null) => Err("max_tokens: Field required".to_string()),
        Some(Value::Number(n)) if n.parse::<i64>().is_ok_and(|n| n >= 1) => Ok(()),
        Some(_) => Err("max_tokens: Input should be a valid integer, at least 1".to_string()),
    }
}

/// One Messages request on `s`: the request read as the OpenAI one it is the same as, one choice made as a chat completion's, the reply in Anthropic's form.
pub(super) fn handle(sh: &Shared, tx: &Sender<Job>, s: &mut TcpStream, req: &Req, count: bool, id: u64) {
    let cors = req.cors.as_str();
    let Some(v) = std::str::from_utf8(&req.body).ok().and_then(|t| json::parse(t).ok()).filter(|v| matches!(v, Value::Object(_))) else {
        return fail(s, cors, 400, "Request body must be a valid JSON object");
    };
    if let Some(m) = (!count).then(|| max_tokens_ok(&v).err()).flatten() {
        return fail(s, cors, 400, &m);
    }
    let o = match openai_of(&v) {
        Ok(o) => o,
        Err(m) => return fail(s, cors, 400, &m),
    };
    let thinking = thinking_on(&v);
    if count {
        return match prepare(sh, &o) {
            Ok(p) => respond_with(s, 200, "application/json", cors, format!("{{\"input_tokens\":{}}}", p.pr.ids.len()).as_bytes()),
            Err((code, m)) => fail(s, cors, code, &m),
        };
    }
    let Started { h, cfg, prompt_tokens, api } = match start_chat(sh, tx, &o, !thinking) {
        Ok(st) => st,
        Err((code, m)) => return fail(s, cors, code, &m),
    };
    let abort = h.abort.clone();
    let mid = format!("msg_{:016x}{id:08x}", fresh_seed());
    let (name, n_in) = (jstr(&sh.name), prompt_tokens);
    let (out_tx, out_rx) = std::sync::mpsc::channel::<Out>();
    std::thread::scope(|sc| {
        let tok = &sh.tok;
        sc.spawn(move || drive(0, h, Choice::new(cfg), tok, !api.stream, out_tx, None));
        if api.stream {
            let head = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\n{cors}Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n");
            let mut ok = s.write_all(head.as_bytes()).is_ok()
                && event(s, "message_start", &format!("\"message\":{{\"id\":\"{mid}\",\"type\":\"message\",\"role\":\"assistant\",\"model\":{name},\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{}}}", usage(n_in, 1)))
                && event(s, "ping", "");
            let mut blocks = Blocks::default();
            while ok {
                match out_rx.recv_timeout(WAIT_TICK) {
                    // (waiting for its turn or its prompt: a ping, as Anthropic's API sends them, and a look at the client)
                    Err(RecvTimeoutError::Timeout) => ok = event(s, "ping", "") && !client_gone(s),
                    Ok(Out::Delta(_, d)) => ok = blocks.delta(s, &d),
                    Ok(Out::Done { last, done, .. }) => {
                        let _ = blocks.delta(s, &last)
                            && blocks.close(s)
                            && event(
                                s,
                                "message_delta",
                                &format!(
                                    "\"delta\":{{\"stop_reason\":\"{}\",\"stop_sequence\":{}}},\"usage\":{{\"input_tokens\":{n_in},\"output_tokens\":{}}}",
                                    stop_reason(&done, blocks.called),
                                    done.stop.as_deref().map_or("null".to_string(), jstr),
                                    done.tokens
                                ),
                            )
                            && event(s, "message_stop", "")
                            && s.write_all(b"0\r\n\r\n").is_ok();
                        return;
                    }
                    Ok(Out::Failed(e)) => {
                        eprintln!("glyd: a request failed: {e}");
                        let _ =
                            event(s, "error", &format!("\"error\":{{\"type\":\"api_error\",\"message\":{}}}", jstr(&e))) && s.write_all(b"0\r\n\r\n").is_ok();
                        return;
                    }
                    Err(RecvTimeoutError::Disconnected) => return,
                }
            }
            // (the client went away: the engine stops after the pass it is in, or skips the reply where its turn has not come)
            abort.store(true, Ordering::Relaxed);
            return;
        }
        loop {
            match out_rx.recv_timeout(WAIT_TICK) {
                Err(RecvTimeoutError::Timeout) if client_gone(s) => {
                    abort.store(true, Ordering::Relaxed);
                    return;
                }
                Err(RecvTimeoutError::Timeout) | Ok(Out::Delta(..)) => {}
                Ok(Out::Done { done, thinking: reasoning, .. }) => {
                    let mut content: Vec<String> = Vec::new();
                    if let Some(t) = reasoning.filter(|t| thinking && !t.is_empty()) {
                        content.push(format!("{{\"type\":\"thinking\",\"thinking\":{},\"signature\":\"\"}}", jstr(&t)));
                    }
                    content.extend(done.content.iter().filter(|t| !t.is_empty()).map(|t| format!("{{\"type\":\"text\",\"text\":{}}}", jstr(t))));
                    for c in &done.calls {
                        let input = json::parse(&c.arguments).ok().filter(|x| matches!(x, Value::Object(_))).unwrap_or(Value::Object(vec![]));
                        content.push(format!(
                            "{{\"type\":\"tool_use\",\"id\":{},\"name\":{},\"input\":{}}}",
                            jstr(&tool_id(&tools::new_id())),
                            jstr(&c.name),
                            json::to_compact(&input)
                        ));
                    }
                    let body = format!(
                        "{{\"id\":\"{mid}\",\"type\":\"message\",\"role\":\"assistant\",\"model\":{name},\"content\":[{}],\"stop_reason\":\"{}\",\"stop_sequence\":{},\"usage\":{}}}",
                        content.join(","),
                        stop_reason(&done, !done.calls.is_empty()),
                        done.stop.as_deref().map_or("null".to_string(), jstr),
                        usage(n_in, done.tokens)
                    );
                    return respond_with(s, 200, "application/json", cors, body.as_bytes());
                }
                Ok(Out::Failed(e)) => return fail(s, cors, 500, &e),
                Err(RecvTimeoutError::Disconnected) => return fail(s, cors, 500, "the engine stopped"),
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::tests::{at, call, serve_fake_with, serve_script, Fake};
    use std::io::Read;

    #[test]
    fn an_anthropic_request_as_the_openai_one() {
        let v = json::parse(r#"{"system":[{"type":"text","text":"Be brief."}],"max_tokens":50,"stop_sequences":["END"],"tool_choice":{"type":"any"},
            "tools":[{"name":"ls","description":"list","input_schema":{"type":"object"}}],
            "messages":[{"role":"user","content":"list"},{"role":"assistant","content":[{"type":"text","text":"Sure."},{"type":"tool_use","id":"t1","name":"ls","input":{"dir":"."}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":[{"type":"text","text":"a.txt"}]}]}]}"#).unwrap();
        let o = openai_of(&v).unwrap();
        let roles: Vec<&str> = o.get("messages").and_then(Value::as_array).unwrap().iter().filter_map(|m| m.get("role").and_then(Value::as_str)).collect();
        assert_eq!(roles, ["system", "user", "assistant", "tool"]);
        let m = o.get("messages").and_then(Value::as_array).unwrap();
        assert_eq!(
            m[2].get("tool_calls").and_then(Value::as_array).unwrap()[0].get("function").and_then(|f| f.get("name")).and_then(Value::as_str),
            Some("ls")
        );
        assert_eq!(m[3].get("content").and_then(Value::as_str), Some("a.txt"));
        assert_eq!(o.get("tool_choice").and_then(Value::as_str), Some("required"));
        assert_eq!(o.get("stop").and_then(Value::as_array).map(|a| a.len()), Some(1));
        assert_eq!(o.get("chat_template_kwargs").and_then(|k| k.get("enable_thinking")).and_then(Value::as_bool), Some(false));
        // (the messages the template reads)
        assert!(tools::template_messages(&o).is_ok());
        // an image block: left out where the model reads text alone; where it reads images, an image_url part beside the text, in order
        let v = json::parse(r#"{"max_tokens":9,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}},{"type":"text","text":"What is it?"},{"type":"image","source":{"type":"url","url":"https://x/y.jpg"}}]}]}"#).unwrap();
        let o = openai_of_with(&v, false).unwrap();
        assert_eq!(
            o.get("messages").and_then(Value::as_array).and_then(|m| m.first()).and_then(|m| m.get("content")).and_then(Value::as_str),
            Some("What is it?")
        );
        let o = openai_of_with(&v, true).unwrap();
        let mut urls = Vec::new();
        let m = tools::template_messages_with(&o, Some(&mut urls)).unwrap();
        assert_eq!(urls, ["data:image/png;base64,AAAA", "https://x/y.jpg"]);
        assert_eq!(m[0].get("content").and_then(Value::as_str), Some(format!("{0}What is it?{0}", tools::IMAGE_PLACE).as_str()));
    }

    fn parse(t: &str) -> Value {
        json::parse(t).unwrap()
    }

    /// The string at `path` where there is one (an event of another kind has no such field).
    fn sv<'a>(v: &'a Value, path: &[&str]) -> Option<&'a str> {
        path.iter().try_fold(v, |v, k| v.get(k)).and_then(Value::as_str)
    }

    /// The JSON of each event of a stream (its `data:` line), and each event's name (its `event:` line) that is the data's `type`.
    fn events(body: &str) -> Vec<Value> {
        let mut out = Vec::new();
        for e in body.split("\n\n").filter(|e| e.starts_with("event: ")) {
            let (head, data) = e.split_once("\ndata: ").unwrap_or_else(|| panic!("{e}"));
            let v = parse(data);
            assert_eq!(&head["event: ".len()..], at(&v, &["type"]).as_str().unwrap(), "{e}");
            out.push(v);
        }
        out
    }

    fn ask(addr: std::net::SocketAddr, body: &str) -> (u16, String) {
        call(addr, "/v1/messages", body)
    }

    /// What the SDKs send that the server reads as chat: the system prompt as blocks, a tool call and its result with an error, the tools Anthropic runs left out, `thinking`, `stop_sequences`.
    #[test]
    fn the_requests_of_the_sdks_and_claude_code() {
        let v = parse(
            r#"{"model":"claude-sonnet-4-5","max_tokens":64,"stream":true,"metadata":{"user_id":"u"},
            "system":[{"type":"text","text":"You are Claude Code.","cache_control":{"type":"ephemeral"}},{"type":"text","text":"Be brief."}],
            "thinking":{"type":"enabled","budget_tokens":1024},
            "tools":[{"name":"Bash","description":"run","input_schema":{"type":"object","properties":{"cmd":{"type":"string"}}},"cache_control":{"type":"ephemeral"}},
                     {"type":"web_search_20250305","name":"web_search","max_uses":3},{"type":"custom","name":"Read","input_schema":{"type":"object"}}],
            "tool_choice":{"type":"auto","disable_parallel_tool_use":true},
            "messages":[{"role":"user","content":[{"type":"text","text":"run ls","cache_control":{"type":"ephemeral"}}]},
              {"role":"assistant","content":[{"type":"thinking","thinking":"hm","signature":"x"},{"type":"tool_use","id":"toolu_1","name":"Bash","input":{"cmd":"ls"}}]},
              {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","is_error":true,"content":[{"type":"text","text":"denied"}]},{"type":"text","text":"try again"}]}]}"#,
        );
        let o = openai_of(&v).unwrap();
        assert_eq!(at(&o, &["messages", "0", "content"]).as_str(), Some("You are Claude Code.\nBe brief."));
        let roles: Vec<&str> = at(&o, &["messages"]).as_array().unwrap().iter().filter_map(|m| m.get("role").and_then(Value::as_str)).collect();
        assert_eq!(roles, ["system", "user", "assistant", "tool", "user"]);
        assert_eq!(at(&o, &["messages", "3", "content"]).as_str(), Some("Error: denied"));
        assert_eq!(at(&o, &["messages", "4", "content"]).as_str(), Some("try again"));
        let names: Vec<&str> = at(&o, &["tools"]).as_array().unwrap().iter().filter_map(|t| at(t, &["function", "name"]).as_str()).collect();
        assert_eq!(names, ["Bash", "Read"]);
        assert_eq!((at(&o, &["parallel_tool_calls"]).as_bool(), at(&o, &["chat_template_kwargs", "enable_thinking"]).as_bool()), (Some(false), Some(true)));
        assert!(tools::template_messages(&o).is_ok() && tools::tools_of(&o).unwrap().is_some());
        assert_eq!(
            at(
                &openai_of(&parse(r#"{"max_tokens":9,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":"x"}]}"#)).unwrap(),
                &["chat_template_kwargs", "enable_thinking"]
            )
            .as_bool(),
            Some(false)
        );
        for (bad, want) in [
            (r#"{"max_tokens":9}"#, "messages: Field required"),
            (r#"{"max_tokens":9,"messages":[]}"#, "at least one message"),
            (r#"{"max_tokens":9,"messages":[{"role":"system","content":"x"}]}"#, "Input should be 'user' or 'assistant'"),
            (r#"{"max_tokens":9,"messages":[{"role":"user","content":5}]}"#, "neither text nor a list of blocks"),
        ] {
            let e = openai_of(&parse(bad)).unwrap_err();
            assert!(e.contains(want), "{bad}: {e}");
        }
        assert_eq!(tool_id("chatcmpl-tool-00ff"), "toolu_00ff");
        assert!(
            anthropic_body(401, "no").contains("\"type\":\"authentication_error\"")
                && anthropic_body(404, "no").contains("not_found_error")
                && anthropic_body(503, "no").contains("overloaded_error")
        );
    }

    /// A message as the SDK parses it: the thinking block only where the request asked for it, the answer's text, the usage, a stop reason by how it ended, and `stop_sequence` the string that did it.
    #[test]
    fn a_message_whole() {
        let addr = serve_script("<think>\nsecret thought\n</think>\n\nthe answer");
        let q = |extra: &str| format!(r#"{{"model":"claude-x","max_tokens":200,"temperature":0,"messages":[{{"role":"user","content":"hi"}}]{extra}}}"#);
        let (st, body) = ask(addr, &q(""));
        assert_eq!(st, 200, "{body}");
        let d = parse(&body);
        assert_eq!(
            (at(&d, &["type"]).as_str(), at(&d, &["role"]).as_str(), at(&d, &["model"]).as_str(), at(&d, &["stop_reason"]).as_str()),
            (Some("message"), Some("assistant"), Some("fake"), Some("end_turn")),
            "{body}"
        );
        assert!(at(&d, &["id"]).as_str().is_some_and(|i| i.starts_with("msg_")));
        assert_eq!(at(&d, &["content"]).as_array().unwrap().len(), 1, "{body}");
        assert_eq!((at(&d, &["content", "0", "type"]).as_str(), at(&d, &["content", "0", "text"]).as_str().map(str::trim)), (Some("text"), Some("the answer")));
        assert!(!body.contains("secret"), "{body}");
        assert_eq!(at(&d, &["stop_sequence"]), &Value::Null);
        assert!(at(&d, &["usage", "input_tokens"]).as_u64().unwrap() > 0 && at(&d, &["usage", "output_tokens"]).as_u64().unwrap() > 0);
        assert_eq!((at(&d, &["usage", "cache_read_input_tokens"]).as_u64(), at(&d, &["usage", "cache_creation_input_tokens"]).as_u64()), (Some(0), Some(0)));
        // thinking asked for: its block first, with the signature field the SDK's type has
        let (_, body) = ask(addr, &q(r#","thinking":{"type":"enabled","budget_tokens":1024}"#));
        let d = parse(&body);
        assert_eq!(
            (
                at(&d, &["content", "0", "type"]).as_str(),
                at(&d, &["content", "0", "thinking"]).as_str().map(str::trim),
                at(&d, &["content", "0", "signature"]).as_str()
            ),
            (Some("thinking"), Some("secret thought"), Some("")),
            "{body}"
        );
        assert_eq!(at(&d, &["content", "1", "type"]).as_str(), Some("text"));
        // a stop sequence ends the reply and is named
        let (_, body) = ask(addr, &q(r#","stop_sequences":["swer"]"#));
        let d = parse(&body);
        assert_eq!(
            (at(&d, &["stop_reason"]).as_str(), at(&d, &["stop_sequence"]).as_str(), at(&d, &["content", "0", "text"]).as_str().map(str::trim)),
            (Some("stop_sequence"), Some("swer"), Some("the an")),
            "{body}"
        );
        // cut at max_tokens
        let (_, body) = ask(addr, &q("").replace("\"max_tokens\":200", "\"max_tokens\":3"));
        assert_eq!(at(&parse(&body), &["stop_reason"]).as_str(), Some("max_tokens"), "{body}");
    }

    /// The stream as the SDK reads it: `message_start`, a `ping`, each block started, its deltas, stopped, then `message_delta` with the stop reason and the usage, and `message_stop`; every event's name is its type.
    #[test]
    fn a_message_streamed() {
        let addr = serve_script("<think>\nsecret thought\n</think>\n\nthe answer");
        let q = |extra: &str| format!(r#"{{"max_tokens":200,"temperature":0,"stream":true,"messages":[{{"role":"user","content":"hi"}}]{extra}}}"#);
        let (st, body) = ask(addr, &q(""));
        assert_eq!(st, 200, "{body}");
        let evs = events(&body);
        let kinds: Vec<&str> = evs.iter().map(|e| at(e, &["type"]).as_str().unwrap()).collect();
        let mut seen = kinds.clone();
        seen.dedup();
        assert_eq!(
            seen,
            ["message_start", "ping", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"],
            "{body}"
        );
        assert_eq!(at(&evs[0], &["message", "usage", "input_tokens"]).as_u64().map(|n| n > 0), Some(true));
        assert_eq!((at(&evs[0], &["message", "role"]).as_str(), at(&evs[0], &["message", "stop_reason"])), (Some("assistant"), &Value::Null));
        let text: String = evs
            .iter()
            .filter(|e| at(e, &["type"]).as_str() == Some("content_block_delta"))
            .map(|e| {
                assert_eq!((at(e, &["index"]).as_u64(), sv(e, &["delta", "type"])), (Some(0), Some("text_delta")));
                at(e, &["delta", "text"]).as_str().unwrap().to_string()
            })
            .collect();
        assert_eq!(text.trim(), "the answer");
        let md = evs.iter().find(|e| at(e, &["type"]).as_str() == Some("message_delta")).unwrap();
        assert_eq!((at(md, &["delta", "stop_reason"]).as_str(), at(md, &["delta", "stop_sequence"])), (Some("end_turn"), &Value::Null));
        assert!(at(md, &["usage", "output_tokens"]).as_u64().unwrap() > 0);
        // with thinking: a thinking block (its deltas, then its signature), then the text block
        let (_, body) = ask(addr, &q(r#","thinking":{"type":"enabled","budget_tokens":1024}"#));
        let evs = events(&body);
        let starts: Vec<(u64, &str)> = evs
            .iter()
            .filter(|e| at(e, &["type"]).as_str() == Some("content_block_start"))
            .map(|e| (at(e, &["index"]).as_u64().unwrap(), sv(e, &["content_block", "type"]).unwrap()))
            .collect();
        assert_eq!(starts, [(0, "thinking"), (1, "text")], "{body}");
        let think: String = evs
            .iter()
            .filter(|e| sv(e, &["delta", "type"]) == Some("thinking_delta"))
            .map(|e| at(e, &["delta", "thinking"]).as_str().unwrap().to_string())
            .collect();
        assert_eq!(think.trim(), "secret thought");
        let sig = evs.iter().position(|e| sv(e, &["delta", "type"]) == Some("signature_delta")).expect("a signature");
        assert_eq!(at(&evs[sig + 1], &["type"]).as_str(), Some("content_block_stop"), "{body}");
        // a stop sequence
        let (_, body) = ask(addr, &q(r#","stop_sequences":["swer"]"#));
        let evs = events(&body);
        let md = evs.iter().find(|e| at(e, &["type"]).as_str() == Some("message_delta")).unwrap();
        assert_eq!((at(md, &["delta", "stop_reason"]).as_str(), at(md, &["delta", "stop_sequence"]).as_str()), (Some("stop_sequence"), Some("swer")), "{body}");
    }

    const CALL: &str = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
    const TOOL: &str = r#"{"name":"get_weather","description":"Get the weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}"#;

    /// A tool call: a `tool_use` block with its id (`toolu_`), the function's name and its input as an object; streamed as a block whose input comes as `input_json_delta`s that make the same object; the stop reason `tool_use`; the
    /// result sent back as a `tool_result` block is read.
    #[test]
    fn tool_use() {
        let addr = serve_script(CALL);
        let q = |extra: &str| {
            format!(r#"{{"max_tokens":200,"temperature":0,"tools":[{TOOL}],"messages":[{{"role":"user","content":"weather in Paris?"}}]{extra}}}"#)
        };
        let (st, body) = ask(addr, &q(""));
        assert_eq!(st, 200, "{body}");
        let d = parse(&body);
        let tu = d
            .get("content")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .find(|b| at(b, &["type"]).as_str() == Some("tool_use"))
            .unwrap_or_else(|| panic!("{body}"));
        assert_eq!(at(tu, &["name"]).as_str(), Some("get_weather"));
        assert!(at(tu, &["id"]).as_str().is_some_and(|i| i.starts_with("toolu_")));
        assert_eq!(at(tu, &["input"]), &parse(r#"{"city":"Paris"}"#));
        assert_eq!(at(&d, &["stop_reason"]).as_str(), Some("tool_use"));

        let (_, body) = ask(addr, &q(r#","stream":true"#));
        let evs = events(&body);
        let start = evs.iter().find(|e| sv(e, &["content_block", "type"]) == Some("tool_use")).unwrap_or_else(|| panic!("{body}"));
        assert!(at(start, &["content_block", "id"]).as_str().is_some_and(|i| i.starts_with("toolu_")));
        assert_eq!((at(start, &["content_block", "name"]).as_str(), at(start, &["content_block", "input"])), (Some("get_weather"), &parse("{}")));
        let json: String = evs
            .iter()
            .filter(|e| sv(e, &["delta", "type"]) == Some("input_json_delta"))
            .map(|e| at(e, &["delta", "partial_json"]).as_str().unwrap().to_string())
            .collect();
        assert_eq!(parse(&json), parse(r#"{"city":"Paris"}"#), "{body}");
        let md = evs.iter().find(|e| at(e, &["type"]).as_str() == Some("message_delta")).unwrap();
        assert_eq!(at(md, &["delta", "stop_reason"]).as_str(), Some("tool_use"));

        // the result goes back
        let back = r#","messages":[{"role":"user","content":"weather in Paris?"},{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Paris"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"18 C"}]}]"#;
        let body = q("").replace(r#","messages":[{"role":"user","content":"weather in Paris?"}]"#, back);
        let (st, b) = ask(addr, &body);
        assert_eq!(st, 200, "{b}");
        // no tool is called where the choice is none
        let (st, b) = ask(addr, &q(r#","tool_choice":{"type":"none"}"#));
        assert_eq!(st, 200, "{b}");
        assert!(!b.contains("\"tool_use\""), "{b}");
    }

    /// The counts, the refusals in Anthropic's error form, and the key: `x-api-key` as the SDKs send it, or a bearer token.
    #[test]
    fn counts_errors_and_the_key() {
        let addr = serve_fake_with(Fake::new(64, 32), None, 2, false, Some("k"), &[]);
        let send = |path: &str, headers: &str, body: &str| {
            let mut c = std::net::TcpStream::connect(addr).unwrap();
            c.write_all(format!("POST {path} HTTP/1.1\r\nHost: x\r\n{headers}Content-Length: {}\r\n\r\n{body}", body.len()).as_bytes()).unwrap();
            let mut out = String::new();
            c.read_to_string(&mut out).unwrap();
            let (head, body) = out.split_once("\r\n\r\n").unwrap();
            (head.split_whitespace().nth(1).unwrap().parse::<u16>().unwrap(), parse(body))
        };
        let ok = r#"{"max_tokens":4,"messages":[{"role":"user","content":"hello there"}]}"#;
        let shape = |b: &Value, kind: &str| {
            assert_eq!((at(b, &["type"]).as_str(), at(b, &["error", "type"]).as_str()), (Some("error"), Some(kind)), "{b:?}");
            assert!(at(b, &["error", "message"]).as_str().is_some_and(|m| !m.is_empty()));
        };
        let (st, b) = send("/v1/messages", "", ok);
        assert_eq!(st, 401);
        shape(&b, "authentication_error");
        let (st, b) = send("/v1/messages", "x-api-key: wrong\r\n", ok);
        assert_eq!(st, 401);
        shape(&b, "authentication_error");
        for h in ["x-api-key: k\r\n", "Authorization: Bearer k\r\n", "X-Api-Key: k\r\nanthropic-version: 2023-06-01\r\n"] {
            let (st, b) = send("/v1/messages?beta=true", h, ok);
            assert_eq!(st, 200, "{h}: {b:?}");
            assert_eq!(at(&b, &["type"]).as_str(), Some("message"));
        }
        let key = "x-api-key: k\r\n";
        let (st, b) = send("/v1/messages/count_tokens", key, r#"{"messages":[{"role":"user","content":"hello there"}]}"#);
        assert_eq!(st, 200, "{b:?}");
        assert!(at(&b, &["input_tokens"]).as_u64().is_some_and(|n| n > 0));
        for (body, want) in [
            (r#"{"messages":[{"role":"user","content":"hi"}]}"#, "max_tokens: Field required"),
            (r#"{"max_tokens":0,"messages":[{"role":"user","content":"hi"}]}"#, "max_tokens: Input should be a valid integer"),
            (r#"{"max_tokens":5}"#, "messages: Field required"),
            (r#"{"max_tokens":5,"messages":[{"role":"user","content":"hi"}],"temperature":"hot"}"#, "temperature"),
            ("nope", "valid JSON"),
        ] {
            let (st, b) = send("/v1/messages", key, body);
            assert_eq!(st, 400, "{body}: {b:?}");
            shape(&b, "invalid_request_error");
            assert!(at(&b, &["error", "message"]).as_str().unwrap().contains(want), "{body}: {b:?}");
        }
        let (st, b) = send("/v1/messages", key, &format!(r#"{{"max_tokens":5,"messages":[{{"role":"user","content":"{}"}}]}}"#, "word ".repeat(40_000)));
        assert_eq!(st, 400);
        shape(&b, "invalid_request_error");
        // the method a path does not take
        let mut c = std::net::TcpStream::connect(addr).unwrap();
        c.write_all(b"GET /v1/messages HTTP/1.1\r\nHost: x\r\nx-api-key: k\r\n\r\n").unwrap();
        let mut out = String::new();
        c.read_to_string(&mut out).unwrap();
        assert!(out.starts_with("HTTP/1.1 405") && out.contains("\"type\":\"error\""), "{out}");
    }
}
