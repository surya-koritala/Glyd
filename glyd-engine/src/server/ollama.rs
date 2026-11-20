//! Ollama's REST API, for the programs that talk to Ollama (Open WebUI, Continue, LangChain, `ollama` libraries): `POST /api/chat` and `/api/generate` (a stream of JSON lines, or one object where `stream` is false),
//! `GET /api/tags`, `/api/ps`, `/api/version`, `POST /api/show`, `/api/pull`, `DELETE /api/delete`. A request is turned into the chat (or plain) request it is the same as (`chat_of`, `generate_of`: the `options` as
//! the fields they are, `format` the response format, `think` the template's thinking), made ready and run as any chat request is (`start_as`: its refusals, its grammar, its tools, its window), and the reply is written in
//! Ollama's form. The server holds one model: a request that names another is answered by it, as the OpenAI endpoints do, and the model list is that one.

use super::*;
use tools::obj;

/// A body of one JSON object, as Ollama's server reads it.
fn parse_body(req: &Req) -> std::result::Result<Value, String> {
    let v = std::str::from_utf8(&req.body).ok().and_then(|t| json::parse(t).ok());
    match v {
        Some(v @ Value::Object(_)) => Ok(v),
        _ if req.body.iter().all(u8::is_ascii_whitespace) => Ok(Value::Object(vec![])),
        _ => Err("invalid JSON".to_string()),
    }
}

/// `{"error": "..."}`, as Ollama says what went wrong.
fn fail(s: &mut TcpStream, cors: &str, code: u16, msg: &str) {
    respond_with(s, code, "application/json; charset=utf-8", cors, format!("{{\"error\":{}}}", jstr(&said(msg))).as_bytes());
}

fn ok(s: &mut TcpStream, cors: &str, body: &str) {
    respond_with(s, 200, "application/json; charset=utf-8", cors, body.as_bytes());
}

/// A time as Ollama stamps it (`created_at`: RFC 3339 with the fraction).
fn stamp() -> String {
    let d = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    let t = rfc3339(d.as_secs());
    format!("{}.{:06}Z", &t[..t.len() - 1], d.subsec_micros())
}

/// The name the model has in the list: the repo (`Qwen/Qwen3-8B`) of a Hub cache's directory, the last name of a path, and a tag (`:latest`) where it has none, as Ollama's names all have one.
pub(super) fn listed_name(name: &str) -> String {
    let base = match name.split('/').find_map(|p| p.strip_prefix("models--")) {
        Some(repo) => repo.replace("--", "/"),
        None if name.starts_with('/') || name.starts_with("./") => name.trim_end_matches('/').rsplit('/').next().unwrap_or(name).to_string(),
        None => name.to_string(),
    };
    if base.contains(':') {
        base
    } else {
        format!("{base}:latest")
    }
}

/// The model's family by its name (what `details.family` and the keys of `model_info` say): the ones Glyd serves, else `glyd`.
fn family(name: &str) -> &'static str {
    let n = name.to_ascii_lowercase();
    [("qwen", "qwen3"), ("gemma", "gemma"), ("glm", "glm"), ("kimi", "kimi"), ("deepseek", "deepseek"), ("muse", "muse"), ("nemotron", "nemotron")]
        .into_iter()
        .find(|(k, _)| n.contains(k))
        .map_or("glyd", |(_, f)| f)
}

/// The size in the name (`8B`, `0.6B`, `27b`) as Ollama writes it (`8B`), or empty.
fn parameter_size(name: &str) -> String {
    let n = listed_name(name);
    let b = n.as_bytes();
    for i in 0..b.len() {
        if b[i].eq_ignore_ascii_case(&b'b') && i > 0 && b[i - 1].is_ascii_digit() && b.get(i + 1).map_or(true, |c| !c.is_ascii_alphanumeric()) {
            let start = (0..i).rev().take_while(|&j| b[j].is_ascii_digit() || b[j] == b'.').last().unwrap_or(i);
            if !(start > 0 && b[start - 1].is_ascii_alphabetic()) {
                return format!("{}B", &n[start..i]);
            }
        }
    }
    String::new()
}

/// Whether a name asks for the model that is served: the same repo, with a tag or without, in any case, or a name that ends the served one's (`qwen3:8b` for `Qwen/Qwen3-8B` is not: only the names that are its own).
fn is_served(asked: &str, name: &str) -> bool {
    let norm = |n: &str| listed_name(n).trim_end_matches(":latest").to_ascii_lowercase();
    let (a, b) = (norm(asked), norm(name));
    a == b || b.rsplit('/').next() == Some(a.as_str())
}

fn digest(name: &str) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(name.as_bytes()).iter().map(|b| format!("{b:02x}")).collect()
}

/// A model's details as the list and the show give them.
fn details(name: &str) -> String {
    let fam = family(name);
    format!(
        "{{\"parent_model\":\"\",\"format\":\"glyd\",\"family\":\"{fam}\",\"families\":[\"{fam}\"],\"parameter_size\":{},\"quantization_level\":\"\"}}",
        jstr(&parameter_size(name))
    )
}

/// The model as `/api/tags` and `/api/ps` list it.
fn model_entry(sh: &Shared, extra: &str) -> String {
    let name = listed_name(&sh.name);
    let size = INFO.get().and_then(|i| i.held.as_ref()).map_or(0, |h| h.weights);
    format!(
        "{{\"name\":{0},\"model\":{0},\"modified_at\":\"{1}\",\"size\":{size},\"digest\":\"{2}\",\"details\":{3}{extra}}}",
        jstr(&name),
        stamp(),
        digest(&sh.name),
        details(&sh.name)
    )
}

/// What the model can do, as `/api/show` says (and the clients read to offer tools, thinking or images): from its template and what the server loaded.
fn capabilities(sh: &Shared) -> Vec<&'static str> {
    let mut c = vec!["completion"];
    if let Some(chat) = sh.chat.as_ref() {
        if chat.mentions("tools") {
            c.push("tools");
        }
        if chat.mentions("enable_thinking") || chat.mentions("<think>") {
            c.push("thinking");
        }
    }
    if crate::vision::PREP.get().is_some() {
        c.push("vision");
    }
    if super::fim_tokens(&sh.tok).is_some() {
        c.push("insert");
    }
    c
}

fn show(sh: &Shared) -> String {
    let (name, fam) = (listed_name(&sh.name), family(&sh.name));
    let caps: Vec<String> = capabilities(sh).iter().map(|c| format!("\"{c}\"")).collect();
    format!(
        "{{\"license\":\"\",\"modelfile\":{},\"parameters\":\"\",\"template\":\"{{{{ .Prompt }}}}\",\"details\":{},\"model_info\":{{\"general.architecture\":\"{fam}\",\"general.basename\":{},\"{fam}.context_length\":{},\"glyd.vocabulary_size\":{}}},\"capabilities\":[{}],\"modified_at\":\"{}\"}}",
        jstr(&format!("# Modelfile generated by glyd\nFROM {name}\n")),
        details(&sh.name),
        jstr(&name),
        sh.window,
        sh.vocab,
        caps.join(","),
        stamp()
    )
}

/// An `options` number as the field of the chat request that has it.
fn set_options(o: &mut Vec<(&'static str, Value)>, opts: Option<&Value>) -> std::result::Result<(), String> {
    let Some(opts) = opts.filter(|v| **v != Value::Null) else { return Ok(()) };
    if !matches!(opts, Value::Object(_)) {
        return Err("options must be an object".to_string());
    }
    for (from, to) in [
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("min_p", "min_p"),
        ("seed", "seed"),
        ("repeat_penalty", "repetition_penalty"),
        ("presence_penalty", "presence_penalty"),
        ("frequency_penalty", "frequency_penalty"),
        ("stop", "stop"),
    ] {
        if let Some(x) = opts.get(from).filter(|x| **x != Value::Null) {
            o.push((to, x.clone()));
        }
    }
    // (`num_predict`: -1 is no limit and -2 fills the context; 0 is no limit here too; `num_ctx` is the server's own, set with --context)
    if let Some(n) = opts.get("num_predict").and_then(Value::as_i64).filter(|&n| n > 0) {
        o.push(("max_tokens", Value::from(n as u64)));
    }
    Ok(())
}

/// `format` as a response format: `"json"` any JSON object, a JSON schema that schema; an empty one nothing.
fn set_format(o: &mut Vec<(&'static str, Value)>, f: Option<&Value>) -> std::result::Result<(), String> {
    match f {
        None | Some(Value::Null) => {}
        Some(Value::String(t)) if t.is_empty() => {}
        Some(Value::String(t)) if t == "json" => o.push(("response_format", obj(vec![("type", Value::from("json_object"))]))),
        Some(sch @ Value::Object(_)) => o.push((
            "response_format",
            obj(vec![("type", Value::from("json_schema")), ("json_schema", obj(vec![("name", Value::from("format")), ("schema", sch.clone())]))]),
        )),
        Some(_) => return Err("invalid format: expected \"json\" or a JSON schema".to_string()),
    }
    Ok(())
}

/// `think`: true or false, or a level (`low`, `medium`, `high`) for the models that take one.
fn set_think(o: &mut Vec<(&'static str, Value)>, think: Option<&Value>) {
    match think {
        Some(Value::Bool(b)) => o.push(("chat_template_kwargs", obj(vec![("enable_thinking", Value::Bool(*b))]))),
        Some(Value::String(l)) if !l.is_empty() => o.push(("reasoning_effort", Value::from(l.as_str()))),
        _ => {}
    }
}

/// The images of a message (base64, as the Ollama libraries send them) as image parts after its text.
fn with_images(text: &str, images: Option<&Value>) -> Value {
    let list = images.and_then(Value::as_array).unwrap_or(&[]);
    if list.is_empty() {
        return Value::from(text);
    }
    let mut parts = vec![obj(vec![("type", Value::from("text")), ("text", Value::from(text))])];
    for i in list.iter().filter_map(Value::as_str) {
        let url = if i.starts_with("data:") || i.starts_with("http") { i.to_string() } else { format!("data:image/png;base64,{i}") };
        parts.push(obj(vec![("type", Value::from("image_url")), ("image_url", obj(vec![("url", Value::String(url))]))]));
    }
    Value::Array(parts)
}

/// `/api/chat`'s request as the chat request it is.
fn chat_of(v: &Value) -> std::result::Result<Value, String> {
    let list = v.get("messages").and_then(Value::as_array).ok_or("messages: Field required")?;
    let mut msgs = Vec::new();
    // (the ids of the calls the assistant made, by name, in order: the tool messages that answer them carry them)
    let (mut ids, mut next): (Vec<(String, String)>, usize) = (Vec::new(), 0);
    for m in list {
        let role = m.get("role").and_then(Value::as_str).ok_or("a message with no role")?;
        if !["system", "user", "assistant", "tool"].contains(&role) {
            return Err(format!("a message with the role `{role}`: system, user, assistant or tool"));
        }
        let text = m.get("content").and_then(Value::as_str).unwrap_or("");
        let mut o = vec![("role", Value::from(role)), ("content", with_images(text, m.get("images")))];
        if let (Some(Value::Array(calls)), "assistant") = (m.get("tool_calls"), role) {
            let mut out = Vec::new();
            for c in calls {
                let f = c.get("function").ok_or("a tool call with no function")?;
                let name = f.get("name").and_then(Value::as_str).ok_or("a tool call with no name")?;
                let id = c.get("id").and_then(Value::as_str).map_or_else(|| format!("call_{}", ids.len()), str::to_string);
                ids.push((name.to_string(), id.clone()));
                let args = match f.get("arguments") {
                    Some(a @ Value::Object(_)) => a.clone(),
                    Some(Value::String(t)) => json::parse(t).unwrap_or(Value::Object(vec![])),
                    _ => Value::Object(vec![]),
                };
                out.push(obj(vec![
                    ("id", Value::from(id.as_str())),
                    ("type", Value::from("function")),
                    ("function", obj(vec![("name", Value::from(name)), ("arguments", args)])),
                ]));
            }
            if !out.is_empty() {
                o.push(("tool_calls", Value::Array(out)));
            }
        }
        if role == "tool" {
            let named = m.get("tool_name").and_then(Value::as_str);
            let id = ids.iter().find(|(n, _)| Some(n.as_str()) == named).or_else(|| ids.get(next)).map(|(_, id)| id.clone());
            next += 1;
            if let Some(id) = id {
                o.push(("tool_call_id", Value::String(id)));
            }
        }
        msgs.push(obj(o));
    }
    let mut o = vec![("messages", Value::Array(msgs))];
    if let Some(t @ Value::Array(_)) = v.get("tools").filter(|t| t.as_array().is_some_and(|a| !a.is_empty())) {
        o.push(("tools", t.clone()));
    }
    set_options(&mut o, v.get("options"))?;
    set_format(&mut o, v.get("format"))?;
    set_think(&mut o, v.get("think"));
    Ok(obj(o))
}

/// `/api/generate`'s request as a chat request (the prompt a user message, `system` before it), or, where `raw`, a plain completion with the prompt as it is (and `suffix`).
fn generate_of(v: &Value) -> std::result::Result<(Value, bool), String> {
    let prompt = v.get("prompt").and_then(Value::as_str).unwrap_or("");
    let suffix = v.get("suffix").and_then(Value::as_str).filter(|t| !t.is_empty());
    // (a prompt that is not wrapped in the chat template: `raw`, or a `suffix`, which is the model's fill in the middle of the prompt and what follows)
    let raw = v.get("raw").and_then(Value::as_bool) == Some(true) || suffix.is_some();
    let mut o = Vec::new();
    if raw {
        o.push(("prompt", Value::from(prompt)));
        if let Some(sfx) = suffix {
            o.push(("suffix", Value::from(sfx)));
        }
    } else {
        let mut msgs = Vec::new();
        if let Some(sys) = v.get("system").and_then(Value::as_str).filter(|t| !t.is_empty()) {
            msgs.push(obj(vec![("role", Value::from("system")), ("content", Value::from(sys))]));
        }
        msgs.push(obj(vec![("role", Value::from("user")), ("content", with_images(prompt, v.get("images")))]));
        o.push(("messages", Value::Array(msgs)));
    }
    set_options(&mut o, v.get("options"))?;
    set_format(&mut o, v.get("format"))?;
    set_think(&mut o, v.get("think"));
    Ok((obj(o), !raw))
}

/// A call as Ollama has it: the function's name and its arguments as an object (the whole of it: Ollama sends a call at once, not in pieces).
fn call_json(index: usize, id: &str, name: &str, arguments: &str) -> String {
    let args = json::parse(arguments).ok().filter(|x| matches!(x, Value::Object(_))).unwrap_or(Value::Object(vec![]));
    format!("{{\"id\":{},\"function\":{{\"index\":{index},\"name\":{},\"arguments\":{}}}}}", jstr(id), jstr(name), json::to_compact(&args))
}

/// The durations and counts that end a reply (nanoseconds, as Ollama counts them).
fn stats(done: &Done, prompt_tokens: usize, begun: std::time::Instant, first: Option<std::time::Duration>) -> String {
    let total = begun.elapsed();
    let first = first.unwrap_or(total);
    format!(
        ",\"done\":true,\"done_reason\":\"{}\",\"total_duration\":{},\"load_duration\":0,\"prompt_eval_count\":{prompt_tokens},\"prompt_eval_duration\":{},\"eval_count\":{},\"eval_duration\":{}",
        done.finish_reason,
        total.as_nanos(),
        first.as_nanos(),
        done.tokens,
        total.saturating_sub(first).as_nanos()
    )
}

/// A line of a stream of JSON lines, as one chunk of the chunked body.
fn line(s: &mut TcpStream, text: &str) -> bool {
    let l = format!("{text}\n");
    s.write_all(format!("{:x}\r\n{l}\r\n", l.len()).as_bytes()).and_then(|_| s.flush()).is_ok()
}

fn start_lines(s: &mut TcpStream, cors: &str) -> bool {
    s.write_all(
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nCache-Control: no-store\r\n{cors}Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
        )
        .as_bytes(),
    )
    .is_ok()
}

/// A model's reply as a message (`/api/chat`) or a response (`/api/generate`): `text`, the thinking, and the calls.
struct Shape {
    chat: bool,
    model: String,
}

impl Shape {
    fn piece(&self, text: &str, think: &str, calls: &[String], done: Option<&str>) -> String {
        let t = if think.is_empty() { String::new() } else { format!(",\"thinking\":{}", jstr(think)) };
        let c = if calls.is_empty() { String::new() } else { format!(",\"tool_calls\":[{}]", calls.join(",")) };
        let head = format!("{{\"model\":{},\"created_at\":\"{}\"", jstr(&self.model), stamp());
        let body = if self.chat {
            format!(",\"message\":{{\"role\":\"assistant\",\"content\":{}{t}{c}}}", jstr(text))
        } else {
            format!(",\"response\":{}{t}", jstr(text))
        };
        format!("{head}{body}{}", done.unwrap_or(",\"done\":false"))
    }
}

/// `/api/chat` and `/api/generate`.
pub(super) fn generate(sh: &Shared, tx: &Sender<Job>, s: &mut TcpStream, req: &Req, chat: bool) {
    let cors = req.cors.as_str();
    let v = match parse_body(req) {
        Ok(v) => v,
        Err(m) => return fail(s, cors, 400, &m),
    };
    let model = listed_name(&sh.name);
    let shape = Shape { chat, model };
    let stream = v.get("stream").and_then(Value::as_bool) != Some(false);
    // (an empty prompt or no messages loads the model, or with `keep_alive: 0` unloads it: one is always loaded here)
    let empty = if chat {
        v.get("messages").and_then(Value::as_array).is_some_and(<[Value]>::is_empty)
    } else {
        v.get("prompt").and_then(Value::as_str).is_none_or(str::is_empty) && v.get("suffix").is_none()
    };
    if empty {
        let unload = v.get("keep_alive").is_some_and(|k| k.as_i64() == Some(0) || k.as_str().is_some_and(|k| k == "0" || k == "0s"));
        let reason = if unload { "unload" } else { "load" };
        let done = format!(",\"done\":true,\"done_reason\":\"{reason}\"");
        let body = shape.piece("", "", &[], Some(&done));
        return respond_with(s, 200, "application/x-ndjson", cors, format!("{body}}}\n").as_bytes());
    }
    let (o, as_chat) = if chat { (chat_of(&v), true) } else { generate_of(&v).map(|(o, c)| (Ok(o), c)).unwrap_or_else(|e| (Err(e), true)) };
    let o = match o {
        Ok(o) => o,
        Err(m) => return fail(s, cors, 400, &m),
    };
    // (the stream the engine is asked for is the reply's, text as it comes: where there are tools it is the whole reply, read for its calls, and sent at once)
    let tools = o.get("tools").is_some();
    let live = stream && !tools;
    let mut o = o;
    if let Value::Object(m) = &mut o {
        m.push(("stream".to_string(), Value::Bool(live)));
    }
    let begun = std::time::Instant::now();
    let Started { h, cfg, prompt_tokens, api } = match start_as(sh, tx, &o, as_chat, false) {
        Ok(st) => st,
        Err((code, m)) => return fail(s, cors, if code == 500 { 500 } else { 400 }, &m),
    };
    let abort = h.abort.clone();
    let (out_tx, out_rx) = std::sync::mpsc::channel::<Out>();
    std::thread::scope(|sc| {
        let (tok, whole) = (&sh.tok, !api.stream);
        sc.spawn(move || drive(0, h, Choice::new(cfg), tok, whole, out_tx, None));
        let (mut first, mut started) = (None, false);
        loop {
            match out_rx.recv_timeout(WAIT_TICK) {
                Err(RecvTimeoutError::Timeout) if client_gone(s) => {
                    abort.store(true, Ordering::Relaxed);
                    return;
                }
                Err(RecvTimeoutError::Timeout) => {}
                Ok(Out::Delta(_, d)) => {
                    first.get_or_insert_with(|| begun.elapsed());
                    if live {
                        started = started || start_lines(s, cors);
                        if !(started
                            && (d.content.is_empty() && d.reasoning.is_empty() || line(s, &format!("{}}}", shape.piece(&d.content, &d.reasoning, &[], None)))))
                        {
                            abort.store(true, Ordering::Relaxed);
                            return;
                        }
                    }
                }
                Ok(Out::Done { last, done, thinking, .. }) => {
                    let end = stats(&done, prompt_tokens, begun, first);
                    if live {
                        started = started || start_lines(s, cors);
                        let more = last.content.is_empty() && last.reasoning.is_empty()
                            || line(s, &format!("{}}}", shape.piece(&last.content, &last.reasoning, &[], None)));
                        let _ = started && more && line(s, &format!("{}}}", shape.piece("", "", &[], Some(&end)))) && s.write_all(b"0\r\n\r\n").is_ok();
                        return;
                    }
                    // the whole reply: its text, its thinking, its calls
                    let calls: Vec<String> =
                        done.calls.iter().enumerate().map(|(i, c)| call_json(i, &format!("call_{}", &tools::new_id()[14..]), &c.name, &c.arguments)).collect();
                    let text = done.content.as_deref().unwrap_or("");
                    let think = thinking.unwrap_or_default();
                    if stream {
                        // (a stream with tools: the reply in one line, then the end)
                        let _ = start_lines(s, cors)
                            && line(s, &format!("{}}}", shape.piece(text, &think, &calls, None)))
                            && line(s, &format!("{}}}", shape.piece("", "", &[], Some(&end))))
                            && s.write_all(b"0\r\n\r\n").is_ok();
                    } else {
                        respond_with(
                            s,
                            200,
                            "application/json; charset=utf-8",
                            cors,
                            format!("{}}}", shape.piece(text, &think, &calls, Some(&end))).as_bytes(),
                        );
                    }
                    return;
                }
                Ok(Out::Failed(e)) => {
                    eprintln!("glyd: a request failed: {e}");
                    if started {
                        let _ = line(s, &format!("{{\"error\":{}}}", jstr(&e))) && s.write_all(b"0\r\n\r\n").is_ok();
                    } else {
                        fail(s, cors, 500, &e);
                    }
                    return;
                }
                Err(RecvTimeoutError::Disconnected) => return fail(s, cors, 500, "the engine stopped"),
            }
        }
    });
}

/// `/api/pull`: the model served is there; another is not one this server can fetch (the server holds one model: `glyd pull NAME`, then `glyd serve NAME`).
fn pull(sh: &Shared, s: &mut TcpStream, req: &Req) {
    let cors = req.cors.as_str();
    let v = match parse_body(req) {
        Ok(v) => v,
        Err(m) => return fail(s, cors, 400, &m),
    };
    let Some(asked) = v.get("model").or_else(|| v.get("name")).and_then(Value::as_str).filter(|m| !m.is_empty()) else {
        return fail(s, cors, 400, "model is required");
    };
    if !is_served(asked, &sh.name) {
        return fail(
            s,
            cors,
            404,
            &format!("this server runs one model, {}: `glyd pull {asked}` fetches another, and `glyd serve {asked}` runs it", listed_name(&sh.name)),
        );
    }
    if v.get("stream").and_then(Value::as_bool) == Some(false) {
        return ok(s, cors, "{\"status\":\"success\"}");
    }
    let _ = start_lines(s, cors) && line(s, "{\"status\":\"pulling manifest\"}") && line(s, "{\"status\":\"success\"}") && s.write_all(b"0\r\n\r\n").is_ok();
}

/// `/api/delete`: a model that is served is not removed under its server.
fn delete(sh: &Shared, s: &mut TcpStream, req: &Req) {
    let cors = req.cors.as_str();
    let v = match parse_body(req) {
        Ok(v) => v,
        Err(m) => return fail(s, cors, 400, &m),
    };
    let Some(asked) = v.get("model").or_else(|| v.get("name")).and_then(Value::as_str).filter(|m| !m.is_empty()) else {
        return fail(s, cors, 400, "model is required");
    };
    if is_served(asked, &sh.name) {
        fail(s, cors, 400, &format!("{} is being served: stop the server, then `glyd rm {asked}`", listed_name(&sh.name)));
    } else {
        fail(s, cors, 404, &format!("model '{asked}' not found"));
    }
}

/// The Ollama version the API follows (what the clients check against): its own, and Glyd's in `glyd_version`.
pub const VERSION: &str = "0.12.6";

/// One request of Ollama's API on `s` (`POST /api/chat`, `/api/generate`, `/api/show`, `/api/pull`; `GET /api/tags`, `/api/ps`, `/api/version`; `DELETE /api/delete`).
pub(super) fn handle(sh: &Shared, tx: &Sender<Job>, s: &mut TcpStream, req: &Req) {
    let cors = req.cors.as_str();
    let path = req.path.split('?').next().unwrap_or("");
    match (req.method.as_str(), path) {
        ("POST", "/api/chat") => generate(sh, tx, s, req, true),
        ("POST", "/api/generate") => generate(sh, tx, s, req, false),
        ("GET" | "HEAD", "/api/tags") => ok(s, cors, &format!("{{\"models\":[{}]}}", model_entry(sh, ""))),
        ("GET" | "HEAD", "/api/ps") => {
            let size = INFO.get().and_then(|i| i.held.as_ref()).map_or(0, |h| h.weights);
            ok(
                s,
                cors,
                &format!(
                    "{{\"models\":[{}]}}",
                    model_entry(sh, &format!(",\"expires_at\":\"0001-01-01T00:00:00Z\",\"size_vram\":{size},\"context_length\":{}", sh.window))
                ),
            )
        }
        ("GET" | "HEAD", "/api/version") => ok(s, cors, &format!("{{\"version\":\"{VERSION}\",\"glyd_version\":\"{}\"}}", env!("CARGO_PKG_VERSION"))),
        ("POST", "/api/show") => match parse_body(req) {
            Ok(v) => match v.get("model").or_else(|| v.get("name")).and_then(Value::as_str) {
                Some(m) if !m.is_empty() && !is_served(m, &sh.name) => fail(s, cors, 404, &format!("model '{m}' not found")),
                _ => ok(s, cors, &show(sh)),
            },
            Err(m) => fail(s, cors, 400, &m),
        },
        ("POST", "/api/pull") => pull(sh, s, req),
        ("DELETE", "/api/delete") => delete(sh, s, req),
        ("POST", "/api/embed" | "/api/embeddings") => fail(s, cors, 501, "embeddings are not served by this model yet"),
        ("POST", "/api/create" | "/api/copy" | "/api/push") | ("HEAD", "/api/blobs") => {
            fail(s, cors, 501, "this server runs the model it was started with: `glyd pull`, `glyd ls` and `glyd rm` manage the models")
        }
        (
            _,
            "/api/chat" | "/api/generate" | "/api/tags" | "/api/ps" | "/api/version" | "/api/show" | "/api/pull" | "/api/delete" | "/api/embed"
            | "/api/embeddings",
        ) => fail(s, cors, 405, "method not allowed"),
        _ => fail(s, cors, 404, &format!("not found: {path}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::tests::{at, serve_fake_with, serve_script, Fake};
    use std::io::Read;

    const CALL: &str = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
    const THINK: &str = "<think>\nsecret thought\n</think>\n\nthe answer";

    fn parse(t: &str) -> Value {
        json::parse(t).unwrap()
    }

    /// A request and its whole answer: (status, headers, the body with a chunked one put together).
    fn http(addr: std::net::SocketAddr, method: &str, path: &str, headers: &str, body: &str) -> (u16, String, String) {
        let mut c = TcpStream::connect(addr).unwrap();
        c.write_all(format!("{method} {path} HTTP/1.1\r\nHost: x\r\n{headers}Content-Length: {}\r\n\r\n{body}", body.len()).as_bytes()).unwrap();
        let mut raw = Vec::new();
        c.read_to_end(&mut raw).unwrap();
        let raw = String::from_utf8(raw).unwrap();
        let (head, rest) = raw.split_once("\r\n\r\n").unwrap();
        let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
        if !head.to_ascii_lowercase().contains("transfer-encoding: chunked") {
            return (status, head.to_string(), rest.to_string());
        }
        let (mut out, mut at) = (String::new(), rest);
        loop {
            let (size, tail) = at.split_once("\r\n").unwrap();
            let n = usize::from_str_radix(size.trim(), 16).unwrap();
            if n == 0 {
                break;
            }
            out.push_str(&tail[..n]);
            at = &tail[n + 2..];
        }
        (status, head.to_string(), out)
    }

    fn post(addr: std::net::SocketAddr, path: &str, body: &str) -> (u16, String, String) {
        http(addr, "POST", path, "", body)
    }

    fn lines(body: &str) -> Vec<Value> {
        body.lines().filter(|l| !l.is_empty()).map(parse).collect()
    }

    /// What the Ollama libraries send: `options` as the fields they are, `format`, `think`, images, a call and its result linked by name.
    #[test]
    fn ollama_requests_as_chat_ones() {
        let o = chat_of(&parse(
            r#"{"model":"llama3.2","stream":false,"keep_alive":"5m","think":false,"format":"json",
            "options":{"temperature":0.2,"top_p":0.9,"top_k":40,"seed":7,"num_predict":50,"num_ctx":8192,"stop":["END"],"repeat_penalty":1.1,"mirostat":2},
            "tools":[{"type":"function","function":{"name":"get_weather","description":"w","parameters":{"type":"object"}}}],
            "messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"hi","images":["QUJD"]},
              {"role":"assistant","content":"","tool_calls":[{"function":{"name":"get_weather","arguments":{"city":"Paris"}}}]},
              {"role":"tool","tool_name":"get_weather","content":"18 C"}]}"#,
        ))
        .unwrap();
        let g = |p: &[&str]| at(&o, p).clone();
        assert_eq!(
            (g(&["temperature"]).as_u64(), g(&["top_k"]).as_u64(), g(&["seed"]).as_u64(), g(&["max_tokens"]).as_u64()),
            (None, Some(40), Some(7), Some(50))
        );
        assert_eq!((g(&["repetition_penalty"]).as_u64(), g(&["stop", "0"]).as_str()), (None, Some("END")));
        assert!(o.get("num_ctx").is_none() && o.get("mirostat").is_none() && o.get("keep_alive").is_none() && o.get("model").is_none());
        assert_eq!(g(&["response_format", "type"]).as_str(), Some("json_object"));
        assert_eq!(g(&["chat_template_kwargs", "enable_thinking"]).as_bool(), Some(false));
        assert_eq!(g(&["messages", "1", "content", "1", "image_url", "url"]).as_str(), Some("data:image/png;base64,QUJD"));
        // (the call the assistant made has an id, and the tool's answer carries it)
        assert_eq!(g(&["messages", "2", "tool_calls", "0", "id"]).as_str(), g(&["messages", "3", "tool_call_id"]).as_str());
        assert_eq!(g(&["messages", "2", "tool_calls", "0", "function", "arguments", "city"]).as_str(), Some("Paris"));
        assert!(tools::template_messages_with(&o, Some(&mut Vec::new())).is_ok() && tools::tools_of(&o).unwrap().is_some());
        // a JSON schema as `format`; a level as `think`; `num_predict` -1 is no limit
        let o = chat_of(&parse(r#"{"messages":[{"role":"user","content":"x"}],"format":{"type":"object","properties":{"a":{"type":"string"}}},"think":"high","options":{"num_predict":-1}}"#)).unwrap();
        assert_eq!(
            (at(&o, &["response_format", "json_schema", "schema", "type"]).as_str(), at(&o, &["reasoning_effort"]).as_str(), o.get("max_tokens")),
            (Some("object"), Some("high"), None)
        );
        assert!(constraint_spec(&o).unwrap().is_some());
        for (bad, want) in [
            (r#"{}"#, "messages: Field required"),
            (r#"{"messages":[{"content":"x"}]}"#, "no role"),
            (r#"{"messages":[{"role":"robot","content":"x"}]}"#, "the role `robot`"),
            (r#"{"messages":[],"format":"xml"}"#, "invalid format"),
            (r#"{"messages":[],"options":5}"#, "options must be an object"),
        ] {
            let e = chat_of(&parse(bad)).unwrap_err();
            assert!(e.contains(want), "{bad}: {e}");
        }
        // generate: the prompt a user message after the system's; raw or a suffix a plain prompt
        let (o, chat) = generate_of(&parse(r#"{"prompt":"hi","system":"Be brief.","images":["QUJD"]}"#)).unwrap();
        assert!(
            chat && at(&o, &["messages", "0", "role"]).as_str() == Some("system")
                && at(&o, &["messages", "1", "content", "1", "type"]).as_str() == Some("image_url")
        );
        let (o, chat) = generate_of(&parse(r#"{"prompt":"def f(","suffix":"return 1","options":{"temperature":0}}"#)).unwrap();
        assert!(!chat && at(&o, &["prompt"]).as_str() == Some("def f(") && at(&o, &["suffix"]).as_str() == Some("return 1"));
        let (o, chat) = generate_of(&parse(r#"{"prompt":"<|im_start|>x","raw":true}"#)).unwrap();
        assert!(!chat && o.get("suffix").is_none());
    }

    #[test]
    fn the_names_the_clients_see() {
        assert_eq!(listed_name("Qwen/Qwen3-8B"), "Qwen/Qwen3-8B:latest");
        assert_eq!(listed_name("Qwen/Qwen3.5-9B:swift"), "Qwen/Qwen3.5-9B:swift");
        assert_eq!(listed_name("/home/u/glyd-demo/hf/hub/models--Qwen--Qwen3-8B/snapshots/b968826d"), "Qwen/Qwen3-8B:latest");
        assert_eq!(listed_name("/models/Qwen3-0.6B/"), "Qwen3-0.6B:latest");
        assert_eq!(
            (parameter_size("Qwen/Qwen3-8B"), parameter_size("Qwen3-0.6B"), parameter_size("x/Gemma-27b-it"), parameter_size("Kimi-K3")),
            ("8B".into(), "0.6B".into(), "27B".into(), String::new())
        );
        assert_eq!(family("Qwen/Qwen3-8B"), "qwen3");
        assert_eq!(family("zai/GLM-5.3"), "glm");
        assert_eq!(family("acme/thing"), "glyd");
        assert!(is_served("Qwen/Qwen3-8B", "Qwen/Qwen3-8B") && is_served("qwen/qwen3-8b:latest", "Qwen/Qwen3-8B") && is_served("Qwen3-8B", "Qwen/Qwen3-8B"));
        assert!(!is_served("llama3.2", "Qwen/Qwen3-8B") && !is_served("Qwen/Qwen3-4B", "Qwen/Qwen3-8B"));
        assert_eq!(digest("a").len(), 64);
    }

    /// A chat as `ollama-python` and Open WebUI read it: one object where `stream` is false; else JSON lines, the pieces of the reply, then the end with its durations and counts.
    #[test]
    fn a_chat_whole_and_streamed() {
        let addr = serve_script(THINK);
        let q =
            |extra: &str| format!(r#"{{"model":"x","messages":[{{"role":"user","content":"hi"}}],"options":{{"temperature":0,"num_predict":200}}{extra}}}"#);
        let (st, head, body) = post(addr, "/api/chat", &q(r#","stream":false"#));
        assert_eq!(st, 200, "{body}");
        assert!(head.to_ascii_lowercase().contains("content-type: application/json"), "{head}");
        let d = parse(&body);
        assert_eq!(
            (at(&d, &["model"]).as_str(), at(&d, &["done"]).as_bool(), at(&d, &["done_reason"]).as_str()),
            (Some("fake:latest"), Some(true), Some("stop")),
            "{body}"
        );
        assert!(at(&d, &["created_at"]).as_str().is_some_and(|t| t.len() > 20 && t.ends_with('Z') && t.contains('.')));
        assert_eq!(
            (
                at(&d, &["message", "role"]).as_str(),
                at(&d, &["message", "content"]).as_str().map(str::trim),
                at(&d, &["message", "thinking"]).as_str().map(str::trim)
            ),
            (Some("assistant"), Some("the answer"), Some("secret thought")),
            "{body}"
        );
        assert!(at(&d, &["prompt_eval_count"]).as_u64().unwrap() > 0 && at(&d, &["eval_count"]).as_u64().unwrap() > 0);
        assert!(
            at(&d, &["total_duration"]).as_u64().unwrap() > 0
                && at(&d, &["load_duration"]).as_u64().is_some()
                && at(&d, &["eval_duration"]).as_u64().is_some()
                && at(&d, &["prompt_eval_duration"]).as_u64().is_some()
        );
        // streamed (the default): lines, the last the end
        let (st, head, body) = post(addr, "/api/chat", &q(""));
        assert_eq!(st, 200);
        assert!(head.to_ascii_lowercase().contains("content-type: application/x-ndjson"), "{head}");
        let ls = lines(&body);
        let (last, rest) = ls.split_last().unwrap();
        assert!(
            rest.len() > 3 && rest.iter().all(|l| at(l, &["done"]).as_bool() == Some(false) && at(l, &["message", "role"]).as_str() == Some("assistant")),
            "{body}"
        );
        let text: String = rest.iter().filter_map(|l| l.get("message").and_then(|m| m.get("content")).and_then(Value::as_str)).collect();
        let think: String = rest.iter().filter_map(|l| l.get("message").and_then(|m| m.get("thinking")).and_then(Value::as_str)).collect();
        assert_eq!((text.trim(), think.trim()), ("the answer", "secret thought"));
        assert_eq!(
            (at(last, &["done"]).as_bool(), at(last, &["done_reason"]).as_str(), at(last, &["message", "content"]).as_str()),
            (Some(true), Some("stop"), Some(""))
        );
        assert!(at(last, &["eval_count"]).as_u64().unwrap() > 0 && at(last, &["prompt_eval_count"]).as_u64().unwrap() > 0);
        // think: false asks for no thinking in the template; cut at the length: `length`
        let (_, _, body) = post(addr, "/api/chat", &q(r#","stream":false,"think":false"#).replace("\"num_predict\":200", "\"num_predict\":3"));
        assert_eq!(at(&parse(&body), &["done_reason"]).as_str(), Some("length"), "{body}");
    }

    /// A call of a tool as Ollama sends it: whole, in a message of its own, its arguments an object; the stream has it in one line and ends as any.
    #[test]
    fn tool_calls() {
        let addr = serve_script(CALL);
        let q = |extra: &str| {
            format!(
                r#"{{"messages":[{{"role":"user","content":"weather in Paris?"}}],"tools":[{{"type":"function","function":{{"name":"get_weather","description":"w","parameters":{{"type":"object","properties":{{"city":{{"type":"string"}}}}}}}}}}],"options":{{"temperature":0,"num_predict":200}}{extra}}}"#
            )
        };
        let (st, _, body) = post(addr, "/api/chat", &q(r#","stream":false"#));
        assert_eq!(st, 200, "{body}");
        let d = parse(&body);
        assert_eq!(
            (
                at(&d, &["message", "tool_calls", "0", "function", "name"]).as_str(),
                at(&d, &["message", "tool_calls", "0", "function", "arguments", "city"]).as_str()
            ),
            (Some("get_weather"), Some("Paris")),
            "{body}"
        );
        assert!(at(&d, &["message", "content"]).as_str().unwrap().trim().is_empty());
        let (_, _, body) = post(addr, "/api/chat", &q(""));
        let ls = lines(&body);
        assert_eq!(ls.len(), 2, "{body}");
        assert_eq!(
            (at(&ls[0], &["message", "tool_calls", "0", "function", "name"]).as_str(), at(&ls[0], &["done"]).as_bool(), at(&ls[1], &["done"]).as_bool()),
            (Some("get_weather"), Some(false), Some(true))
        );
    }

    /// /api/generate: the prompt in the chat template, or raw; its `response`; the empty prompt that loads the model.
    #[test]
    fn generate_and_load() {
        let addr = serve_script(THINK);
        let (st, _, body) = post(addr, "/api/generate", r#"{"prompt":"hi","stream":false,"options":{"temperature":0,"num_predict":200}}"#);
        assert_eq!(st, 200, "{body}");
        let d = parse(&body);
        assert_eq!(
            (at(&d, &["response"]).as_str().map(str::trim), at(&d, &["thinking"]).as_str().map(str::trim), at(&d, &["done"]).as_bool()),
            (Some("the answer"), Some("secret thought"), Some(true)),
            "{body}"
        );
        let (_, _, body) = post(addr, "/api/generate", r#"{"prompt":"hi","options":{"temperature":0,"num_predict":200}}"#);
        let ls = lines(&body);
        let text: String = ls.iter().filter_map(|l| l.get("response").and_then(Value::as_str)).collect();
        assert!(text.contains("the answer") && at(ls.last().unwrap(), &["done"]).as_bool() == Some(true), "{body}");
        // raw: the prompt as it is
        let (st, _, body) = post(addr, "/api/generate", r#"{"prompt":"hi","raw":true,"stream":false,"options":{"temperature":0,"num_predict":4}}"#);
        assert_eq!(st, 200, "{body}");
        assert!(at(&parse(&body), &["eval_count"]).as_u64().unwrap() > 0);
        // an empty prompt loads the model, and keep_alive 0 unloads it: nothing runs
        let (_, _, body) = post(addr, "/api/generate", r#"{"model":"m"}"#);
        let d = parse(body.trim());
        assert_eq!((at(&d, &["done"]).as_bool(), at(&d, &["done_reason"]).as_str(), at(&d, &["response"]).as_str()), (Some(true), Some("load"), Some("")));
        let (_, _, body) = post(addr, "/api/chat", r#"{"model":"m","messages":[],"keep_alive":0}"#);
        assert_eq!(at(&parse(body.trim()), &["done_reason"]).as_str(), Some("unload"));
        // refused: in Ollama's `{"error": "..."}`
        let (st, _, body) = post(addr, "/api/generate", r#"{"prompt":"hi","format":"xml"}"#);
        assert_eq!(st, 400);
        assert!(at(&parse(&body), &["error"]).as_str().unwrap().contains("invalid format"), "{body}");
        let (st, _, _) = post(addr, "/api/chat", "nope");
        assert_eq!(st, 400);
    }

    /// The rest of the API: the list, what is running, the version, what the model can do, the pulls and deletes of a server of one model.
    #[test]
    fn the_models_and_the_rest() {
        let addr = serve_script(THINK);
        let get = |p: &str| {
            let (st, _, b) = http(addr, "GET", p, "", "");
            (st, parse(&b))
        };
        let (st, tags) = get("/api/tags");
        assert_eq!(st, 200);
        let m = at(&tags, &["models", "0"]);
        assert_eq!((at(m, &["name"]).as_str(), at(m, &["model"]).as_str()), (Some("fake:latest"), Some("fake:latest")));
        assert!(
            at(m, &["digest"]).as_str().is_some_and(|d| d.len() == 64) && at(m, &["modified_at"]).as_str().is_some() && at(m, &["size"]).as_u64().is_some()
        );
        assert!(
            at(m, &["details", "family"]).as_str().is_some()
                && at(m, &["details", "families"]).as_array().is_some()
                && at(m, &["details", "parameter_size"]).as_str().is_some()
        );
        let (_, ps) = get("/api/ps");
        assert_eq!(at(&ps, &["models", "0", "name"]).as_str(), Some("fake:latest"));
        assert!(at(&ps, &["models", "0", "expires_at"]).as_str().is_some() && at(&ps, &["models", "0", "size_vram"]).as_u64().is_some());
        let (_, v) = get("/api/version");
        assert_eq!(at(&v, &["version"]).as_str(), Some(VERSION));
        let (st, show) = {
            let (st, _, b) = post(addr, "/api/show", r#"{"model":"fake"}"#);
            (st, parse(&b))
        };
        assert_eq!(st, 200);
        let caps: Vec<&str> = at(&show, &["capabilities"]).as_array().unwrap().iter().filter_map(Value::as_str).collect();
        assert!(caps.contains(&"completion") && caps.contains(&"tools") && caps.contains(&"thinking"), "{caps:?}");
        assert!(at(&show, &["model_info"]).as_object().unwrap().iter().any(|(k, v)| k.ends_with(".context_length") && v.as_u64() == Some(1024)), "{show:?}");
        assert!(at(&show, &["details", "format"]).as_str().is_some() && at(&show, &["modelfile"]).as_str().unwrap().contains("FROM fake:latest"));
        let (st, _, b) = post(addr, "/api/show", r#"{"model":"llama3.2"}"#);
        assert!(st == 404 && at(&parse(&b), &["error"]).as_str().unwrap().contains("not found"), "{b}");
        // pull: the model served is there (a stream of statuses, or one), another is not one this server fetches
        let (st, _, b) = post(addr, "/api/pull", r#"{"model":"fake"}"#);
        assert_eq!(st, 200);
        assert_eq!(lines(&b).last().map(|l| at(l, &["status"]).as_str().map(str::to_string)), Some(Some("success".to_string())), "{b}");
        let (_, _, b) = post(addr, "/api/pull", r#"{"model":"fake:latest","stream":false}"#);
        assert_eq!(at(&parse(&b), &["status"]).as_str(), Some("success"));
        let (st, _, b) = post(addr, "/api/pull", r#"{"model":"llama3.2"}"#);
        assert!(st == 404 && at(&parse(&b), &["error"]).as_str().unwrap().contains("glyd pull llama3.2"), "{b}");
        let (st, _, b) = http(addr, "DELETE", "/api/delete", "", r#"{"model":"fake"}"#);
        assert!(st == 400 && at(&parse(&b), &["error"]).as_str().unwrap().contains("is being served"), "{b}");
        let (st, _, _) = http(addr, "DELETE", "/api/delete", "", r#"{"model":"other"}"#);
        assert_eq!(st, 404);
        let (st, _, b) = post(addr, "/api/embed", r#"{"model":"fake","input":"x"}"#);
        assert!(st == 501 && at(&parse(&b), &["error"]).as_str().is_some(), "{b}");
        let (st, _, _) = http(addr, "GET", "/api/chat", "", "");
        assert_eq!(st, 405);
        let (st, _, _) = http(addr, "GET", "/api/nothing", "", "");
        assert_eq!(st, 404);
    }

    /// The key guards /api as it guards /v1.
    #[test]
    fn the_key_guards_the_api() {
        let addr = serve_fake_with(Fake::new(64, 32), None, 2, false, Some("k"), &[]);
        for (m, p) in [("GET", "/api/tags"), ("GET", "/api/version"), ("POST", "/api/show"), ("POST", "/api/chat")] {
            let (st, _, b) = http(addr, m, p, "", "{}");
            assert!(st == 401 && b.contains("\"error\""), "{p}: {st} {b}");
            let (st, _, _) = http(addr, m, p, "Authorization: Bearer wrong\r\n", "{}");
            assert_eq!(st, 401, "{p}");
        }
        let (st, _, b) = http(addr, "GET", "/api/tags", "Authorization: Bearer k\r\n", "");
        assert_eq!(st, 200, "{b}");
        let (st, _, b) = post(addr, "/api/chat", r#"{"messages":[{"role":"user","content":"hi"}],"stream":false,"options":{"num_predict":2}}"#);
        assert_eq!(st, 401, "{b}");
        let (st, _, b) = http(
            addr,
            "POST",
            "/api/chat",
            "Authorization: Bearer k\r\n",
            r#"{"messages":[{"role":"user","content":"hi"}],"stream":false,"options":{"num_predict":2}}"#,
        );
        assert_eq!(st, 200, "{b}");
    }
}
