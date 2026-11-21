//! OpenAI's Responses API, stateless: `POST /v1/responses`. A request is turned into the chat request it is the same as (`chat_of`: the instructions a system message, the input items messages, a function call and its output
//! an assistant's tool call and a tool message, `text.format` the `response_format`), made ready and run as any chat request is (`start_chat`: its refusals, its grammar, its tools, its window), and the reply is
//! written as a response object, or as the events of a stream: `response.created` ... `response.completed`, one output item at a time (the thinking as a `reasoning` item, the answer as a `message`, each tool call as a
//! `function_call`). The server keeps no conversations: `previous_response_id`, `conversation` and `background` are refused in words, a response is not stored (`store` is false in what it says).

use super::*;
use tools::obj;

fn fail(s: &mut TcpStream, cors: &str, code: u16, msg: &str) {
    respond_with(s, code, "application/json", cors, error_body(msg, code).as_bytes());
}

/// One message of the chat request as it is made: calls of the assistant join the message before them.
struct Msg {
    role: String,
    content: Value,
    calls: Vec<Value>,
    call_id: Option<Value>,
}

/// A message's content: a string as it is, a list of parts as chat parts (`input_text`, `output_text` and `text` are text, `input_image` an image).
fn parts_of(c: &Value) -> std::result::Result<Value, String> {
    let list = match c {
        Value::String(_) => return Ok(c.clone()),
        Value::Array(l) => l,
        _ => return Err("a message whose content is neither text nor a list of parts".to_string()),
    };
    let mut out = Vec::new();
    for p in list {
        match p.get("type").and_then(Value::as_str) {
            Some("input_text" | "output_text" | "text") => {
                out.push(obj(vec![("type", Value::from("text")), ("text", p.get("text").cloned().unwrap_or(Value::from("")))]))
            }
            Some("input_image") => {
                let url = p
                    .get("image_url")
                    .and_then(|u| u.as_str().or_else(|| u.get("url").and_then(Value::as_str)))
                    .ok_or("an input_image part with no image_url (a file_id is not read here)")?;
                out.push(obj(vec![("type", Value::from("image_url")), ("image_url", obj(vec![("url", Value::from(url))]))]));
            }
            Some("refusal") => {}
            Some(t) => return Err(format!("a content part of type `{t}` is not supported: input_text, output_text and input_image are")),
            None => return Err("a content part with no type".to_string()),
        }
    }
    Ok(Value::Array(out))
}

/// The text of a function call's output (a string, or a list of text parts).
fn output_text(o: Option<&Value>) -> String {
    match o {
        Some(Value::String(t)) => t.clone(),
        Some(Value::Array(l)) => l.iter().filter_map(|p| p.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join("\n"),
        Some(Value::Null) | None => String::new(),
        Some(other) => json::to_compact(other),
    }
}

/// The Responses request `v` as the chat request it asks for, or what is wrong with it.
pub(super) fn chat_of(v: &Value) -> std::result::Result<Value, String> {
    let set = |k: &str| v.get(k).filter(|x| **x != Value::Null);
    if set("previous_response_id").is_some() {
        return Err("previous_response_id is not supported: this server keeps no conversations; send the whole input each time".to_string());
    }
    if set("conversation").is_some() {
        return Err("conversation is not supported: this server keeps no conversations; send the whole input each time".to_string());
    }
    if set("background").and_then(Value::as_bool) == Some(true) {
        return Err("background is not supported: this server answers a request when it has run".to_string());
    }
    let mut msgs: Vec<Msg> = Vec::new();
    if let Some(Value::String(t)) = set("instructions").filter(|t| t.as_str().is_some_and(|t| !t.is_empty())) {
        msgs.push(Msg { role: "system".into(), content: Value::String(t.clone()), calls: Vec::new(), call_id: None });
    }
    match set("input") {
        Some(Value::String(t)) => msgs.push(Msg { role: "user".into(), content: Value::String(t.clone()), calls: Vec::new(), call_id: None }),
        Some(Value::Array(items)) => {
            for item in items {
                let ty = item.get("type").and_then(Value::as_str).unwrap_or(if item.get("role").is_some() { "message" } else { "" });
                match ty {
                    "message" => {
                        let role = match item.get("role").and_then(Value::as_str) {
                            Some("developer") => "system",
                            Some(r @ ("user" | "assistant" | "system")) => r,
                            Some(r) => return Err(format!("a message with the role `{r}`: user, assistant, system and developer are")),
                            None => return Err("a message with no role".to_string()),
                        };
                        let content = parts_of(item.get("content").ok_or("a message with no content")?)?;
                        msgs.push(Msg { role: role.into(), content, calls: Vec::new(), call_id: None });
                    }
                    "function_call" => {
                        let get = |k: &str| item.get(k).cloned().unwrap_or(Value::Null);
                        let args = match item.get("arguments") {
                            Some(a @ Value::String(_)) => a.clone(),
                            Some(Value::Null) | None => Value::from("{}"),
                            Some(other) => Value::String(json::to_compact(other)),
                        };
                        let call = obj(vec![
                            ("id", get("call_id")),
                            ("type", Value::from("function")),
                            ("function", obj(vec![("name", get("name")), ("arguments", args)])),
                        ]);
                        match msgs.last_mut().filter(|m| m.role == "assistant" && m.call_id.is_none()) {
                            Some(m) => m.calls.push(call),
                            None => msgs.push(Msg { role: "assistant".into(), content: Value::from(""), calls: vec![call], call_id: None }),
                        }
                    }
                    "function_call_output" => msgs.push(Msg {
                        role: "tool".into(),
                        content: Value::String(output_text(item.get("output"))),
                        calls: Vec::new(),
                        call_id: Some(item.get("call_id").cloned().unwrap_or(Value::Null)),
                    }),
                    // (what an earlier reply of the model's carried: the model reads its answers, not the thinking it made for them)
                    "reasoning" => {}
                    "item_reference" => {
                        return Err("item_reference is not supported: this server keeps no conversations; send the items themselves".to_string())
                    }
                    "" => return Err("an input item with no type".to_string()),
                    t => return Err(format!("an input item of type `{t}` is not supported: message, function_call and function_call_output are")),
                }
            }
        }
        _ => return Err("input: Field required (a string, or a list of items)".to_string()),
    }
    if msgs.iter().all(|m| m.role == "system") {
        return Err("input: no message to answer".to_string());
    }
    let messages: Vec<Value> = msgs
        .into_iter()
        .map(|m| {
            let mut o = vec![("role", Value::String(m.role)), ("content", m.content)];
            if !m.calls.is_empty() {
                o.push(("tool_calls", Value::Array(m.calls)));
            }
            if let Some(id) = m.call_id {
                o.push(("tool_call_id", id));
            }
            obj(o)
        })
        .collect();
    let mut o = vec![("messages", Value::Array(messages))];
    // the tools: functions (what the other kinds do, a search or a code run, happens at OpenAI's end, which this server is not)
    if let Some(Value::Array(list)) = set("tools") {
        let mut tools = Vec::new();
        for t in list {
            match t.get("type").and_then(Value::as_str) {
                Some("function") => {
                    let get = |k: &str| t.get(k).cloned().unwrap_or(Value::Null);
                    let mut f = vec![("name", get("name")), ("description", get("description")), ("parameters", get("parameters"))];
                    if let Some(b @ Value::Bool(_)) = t.get("strict") {
                        f.push(("strict", b.clone()));
                    }
                    tools.push(obj(vec![("type", Value::from("function")), ("function", obj(f))]));
                }
                other => return Err(format!("a tool of type `{}` is not supported: this server runs `function` tools", other.unwrap_or("(none)"))),
            }
        }
        if !tools.is_empty() {
            o.push(("tools", Value::Array(tools)));
        }
    }
    match set("tool_choice") {
        None => {}
        Some(c @ Value::String(_)) => o.push(("tool_choice", c.clone())),
        Some(c) => match (c.get("type").and_then(Value::as_str), c.get("name")) {
            (Some("function"), Some(n)) => o.push(("tool_choice", obj(vec![("type", Value::from("function")), ("function", obj(vec![("name", n.clone())]))]))),
            _ => return Err("tool_choice must be none, auto, required or {\"type\": \"function\", \"name\": ...}".to_string()),
        },
    }
    // the form of the text: a JSON schema holds the reply to it, as `response_format` does
    if let Some(f) = v.get("text").and_then(|t| t.get("format")).filter(|f| **f != Value::Null) {
        match f.get("type").and_then(Value::as_str) {
            Some("text") | None => {}
            Some("json_object") => o.push(("response_format", obj(vec![("type", Value::from("json_object"))]))),
            Some("json_schema") => {
                let get = |k: &str| f.get(k).cloned();
                let mut js =
                    vec![("name", get("name").unwrap_or(Value::from("response"))), ("schema", get("schema").ok_or("text.format json_schema: no schema")?)];
                if let Some(b) = get("strict") {
                    js.push(("strict", b));
                }
                o.push(("response_format", obj(vec![("type", Value::from("json_schema")), ("json_schema", obj(js))])));
            }
            Some(t) => return Err(format!("text.format of type `{t}` is not supported: text, json_object and json_schema are")),
        }
    }
    for (from, to) in [
        ("max_output_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("seed", "seed"),
        ("parallel_tool_calls", "parallel_tool_calls"),
        ("stream", "stream"),
    ] {
        if let Some(x) = set(from) {
            o.push((to, x.clone()));
        }
    }
    if let Some(e) = v.get("reasoning").and_then(|r| r.get("effort")).and_then(Value::as_str) {
        // (`minimal` is as little as the model's template has: none)
        o.push(("reasoning_effort", Value::from(if e == "minimal" { "none" } else { e })));
    }
    Ok(obj(o))
}

/// The tools as a response repeats them: each function with its description, its parameters and `strict` (false where the request gave none: this server does not hold arguments to the schema of a tool, only a forced call's).
fn tools_echo(v: &Value) -> Value {
    let list = v.get("tools").and_then(Value::as_array).unwrap_or(&[]);
    Value::Array(
        list.iter()
            .map(|t| {
                let get = |k: &str, or: Value| t.get(k).filter(|x| **x != Value::Null).cloned().unwrap_or(or);
                obj(vec![
                    ("type", get("type", Value::from("function"))),
                    ("name", get("name", Value::from(""))),
                    ("description", get("description", Value::Null)),
                    ("parameters", get("parameters", Value::Null)),
                    ("strict", get("strict", Value::Bool(false))),
                ])
            })
            .collect(),
    )
}

/// What a response says besides its output: the request's own settings as the API repeats them.
struct Ctx {
    id: String,
    created: u64,
    model: String,
    echo: Vec<(&'static str, Value)>,
}

impl Ctx {
    fn new(v: &Value, model: &str, id: u64) -> Ctx {
        let get = |k: &str, or: Value| v.get(k).filter(|x| **x != Value::Null).cloned().unwrap_or(or);
        let echo = vec![
            ("background", Value::Bool(false)),
            ("error", Value::Null),
            ("instructions", get("instructions", Value::Null)),
            ("max_output_tokens", get("max_output_tokens", Value::Null)),
            ("metadata", get("metadata", Value::Object(vec![]))),
            ("parallel_tool_calls", get("parallel_tool_calls", Value::Bool(true))),
            ("previous_response_id", Value::Null),
            // (`minimal` is run as none, and said so)
            (
                "reasoning",
                obj(vec![
                    (
                        "effort",
                        v.get("reasoning").and_then(|r| r.get("effort")).map_or(Value::Null, |e| {
                            if e.as_str() == Some("minimal") {
                                Value::from("none")
                            } else {
                                e.clone()
                            }
                        }),
                    ),
                    ("summary", Value::Null),
                ]),
            ),
            ("store", Value::Bool(false)),
            ("temperature", get("temperature", Value::from(1u64))),
            ("text", obj(vec![("format", v.get("text").and_then(|t| t.get("format")).cloned().unwrap_or_else(|| obj(vec![("type", Value::from("text"))])))])),
            ("tool_choice", get("tool_choice", Value::from("auto"))),
            ("tools", tools_echo(v)),
            ("top_p", get("top_p", Value::from(1u64))),
            // (what the specification's response has beside: the penalties asked for, no tool-call limit, the one tier, and the keys of a request, null where it sent none)
            ("presence_penalty", get("presence_penalty", Value::from(0u64))),
            ("frequency_penalty", get("frequency_penalty", Value::from(0u64))),
            ("top_logprobs", Value::from(0u64)),
            ("max_tool_calls", get("max_tool_calls", Value::Null)),
            ("service_tier", Value::from("default")),
            ("safety_identifier", get("safety_identifier", Value::Null)),
            ("prompt_cache_key", get("prompt_cache_key", Value::Null)),
            ("truncation", Value::from("disabled")),
            ("user", get("user", Value::Null)),
        ];
        Ctx { id: format!("resp_{:016x}{id:08x}", fresh_seed()), created: now(), model: model.to_string(), echo }
    }

    /// The response object: its `status` (`in_progress`, `completed`, `incomplete`), its output items (JSON each), and its usage once it is known.
    fn response(&self, status: &str, reason: Option<&str>, output: &[String], usage: Option<(usize, usize)>, error: Option<&str>) -> String {
        let mut f = vec![
            format!("\"id\":{}", jstr(&self.id)),
            "\"object\":\"response\"".to_string(),
            format!("\"created_at\":{}", self.created),
            // (when it ended: not before)
            format!("\"completed_at\":{}", if status == "in_progress" { "null".to_string() } else { now().to_string() }),
            format!("\"status\":\"{status}\""),
            format!("\"model\":{}", jstr(&self.model)),
            format!("\"output\":[{}]", output.join(",")),
            format!("\"incomplete_details\":{}", reason.map_or("null".to_string(), |r| format!("{{\"reason\":\"{r}\"}}"))),
        ];
        for (k, x) in &self.echo {
            if *k == "error" {
                f.push(format!("\"error\":{}", error.map_or("null".to_string(), |m| format!("{{\"code\":\"server_error\",\"message\":{}}}", jstr(m)))));
            } else {
                f.push(format!("\"{k}\":{}", json::to_compact(x)));
            }
        }
        f.push(format!(
            "\"usage\":{}",
            usage.map_or("null".to_string(), |(i, o)| format!(
                "{{\"input_tokens\":{i},\"input_tokens_details\":{{\"cached_tokens\":0}},\"output_tokens\":{o},\"output_tokens_details\":{{\"reasoning_tokens\":0}},\"total_tokens\":{}}}",
                i + o
            ))
        ));
        format!("{{{}}}", f.join(","))
    }
}

/// A new id for an item of the output: `msg_`, `rs_`, `fc_` and 16 hex digits.
fn item_id(prefix: &str) -> String {
    format!("{prefix}_{:016x}", fresh_seed())
}

fn message_item(id: &str, status: &str, text: Option<&str>) -> String {
    let content = text.map_or(String::new(), |t| format!("{{\"type\":\"output_text\",\"text\":{},\"annotations\":[],\"logprobs\":[]}}", jstr(t)));
    format!("{{\"id\":\"{id}\",\"type\":\"message\",\"status\":\"{status}\",\"role\":\"assistant\",\"content\":[{content}]}}")
}

fn reasoning_item(id: &str, text: &str) -> String {
    format!("{{\"id\":\"{id}\",\"type\":\"reasoning\",\"summary\":[],\"content\":[{{\"type\":\"reasoning_text\",\"text\":{}}}]}}", jstr(text))
}

fn call_item(id: &str, call_id: &str, name: &str, args: &str, status: &str) -> String {
    format!(
        "{{\"id\":\"{id}\",\"type\":\"function_call\",\"status\":\"{status}\",\"call_id\":{},\"name\":{},\"arguments\":{}}}",
        jstr(call_id),
        jstr(name),
        jstr(args)
    )
}

/// `incomplete` where the reply was cut at its length.
fn status_of(finish: &str) -> (&'static str, Option<&'static str>) {
    if finish == "length" {
        ("incomplete", Some("max_output_tokens"))
    } else {
        ("completed", None)
    }
}

/// The item of the output that is being written.
enum Cur {
    Think { id: String, text: String },
    Text { id: String, text: String },
    Call { id: String, call_id: String, name: String, args: String },
}

/// The events of a streamed response, each with its number in the order they go: one output item is open at a time, its events sent as the deltas come.
struct Events<'a> {
    s: &'a mut TcpStream,
    seq: u64,
    /// The items that are done, as JSON (the response's output where it ends).
    done: Vec<String>,
    cur: Option<Cur>,
}

impl Events<'_> {
    /// An event: its type, its number, and `fields`.
    fn send(&mut self, kind: &str, fields: Vec<(&str, Value)>) -> bool {
        let mut o = vec![("type", Value::from(kind)), ("sequence_number", Value::from(self.seq))];
        o.extend(fields);
        self.seq += 1;
        let ev = format!("event: {kind}\ndata: {}\n\n", json::to_compact(&obj(o)));
        self.s.write_all(format!("{:x}\r\n{ev}\r\n", ev.len()).as_bytes()).and_then(|_| self.s.flush()).is_ok()
    }

    fn raw(&mut self, kind: &str, key: &str, json: &str) -> bool {
        let v = json::parse(json).unwrap_or(Value::Null);
        self.send(kind, vec![(key, v)])
    }

    fn index(&self) -> Value {
        Value::from(self.done.len() as u64)
    }

    /// The open item ends: its closing events, and it is among the done.
    fn close(&mut self) -> bool {
        let Some(cur) = self.cur.take() else { return true };
        let (index, mut ok) = (self.index(), true);
        let item = match cur {
            Cur::Think { id, text } => {
                ok &= self.send(
                    "response.reasoning_text.done",
                    vec![
                        ("item_id", Value::from(id.as_str())),
                        ("output_index", index.clone()),
                        ("content_index", Value::from(0u64)),
                        ("text", Value::from(text.as_str())),
                    ],
                );
                reasoning_item(&id, &text)
            }
            Cur::Text { id, text } => {
                let part = format!("{{\"type\":\"output_text\",\"text\":{},\"annotations\":[],\"logprobs\":[]}}", jstr(&text));
                ok &= self.send(
                    "response.output_text.done",
                    vec![
                        ("item_id", Value::from(id.as_str())),
                        ("output_index", index.clone()),
                        ("content_index", Value::from(0u64)),
                        ("text", Value::from(text.as_str())),
                        ("logprobs", Value::Array(vec![])),
                    ],
                );
                ok &= self.send(
                    "response.content_part.done",
                    vec![
                        ("item_id", Value::from(id.as_str())),
                        ("output_index", index.clone()),
                        ("content_index", Value::from(0u64)),
                        ("part", json::parse(&part).unwrap_or(Value::Null)),
                    ],
                );
                message_item(&id, "completed", Some(&text))
            }
            Cur::Call { id, call_id, name, args } => {
                ok &= self.send(
                    "response.function_call_arguments.done",
                    vec![
                        ("item_id", Value::from(id.as_str())),
                        ("output_index", index.clone()),
                        ("name", Value::from(name.as_str())),
                        ("arguments", Value::from(args.as_str())),
                    ],
                );
                call_item(&id, &call_id, &name, &args, "completed")
            }
        };
        ok &= self.raw_item("response.output_item.done", &item);
        self.done.push(item);
        ok
    }

    fn raw_item(&mut self, kind: &str, item: &str) -> bool {
        let index = self.index();
        let v = json::parse(item).unwrap_or(Value::Null);
        self.send(kind, vec![("output_index", index), ("item", v)])
    }

    /// A message item opened with its (empty) text part: `output_item.added` and `content_part.added`, as the text's own events then refer to them (a client that keeps the response as the events build it needs the part to be there).
    fn open_text(&mut self) -> bool {
        let id = item_id("msg");
        let mut ok = self.raw_item("response.output_item.added", &message_item(&id, "in_progress", None));
        let index = self.index();
        ok &= self.send(
            "response.content_part.added",
            vec![
                ("item_id", Value::from(id.as_str())),
                ("output_index", index),
                ("content_index", Value::from(0u64)),
                ("part", json::parse("{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[],\"logprobs\":[]}").unwrap_or(Value::Null)),
            ],
        );
        self.cur = Some(Cur::Text { id, text: String::new() });
        ok
    }

    /// What a delta adds: thinking, answer text, and calls, each in an item of its own kind (a change of kind ends the open item and begins another).
    fn delta(&mut self, d: &Delta) -> bool {
        let mut ok = true;
        if !d.reasoning.is_empty() {
            if !matches!(self.cur, Some(Cur::Think { .. })) {
                ok &= self.close();
                let id = item_id("rs");
                ok &= self.raw_item("response.output_item.added", &format!("{{\"id\":\"{id}\",\"type\":\"reasoning\",\"summary\":[],\"content\":[]}}"));
                self.cur = Some(Cur::Think { id, text: String::new() });
            }
            if let Some(Cur::Think { id, text }) = &mut self.cur {
                text.push_str(&d.reasoning);
                let (id, index) = (id.clone(), Value::from(self.done.len() as u64));
                ok &= self.send(
                    "response.reasoning_text.delta",
                    vec![
                        ("item_id", Value::from(id.as_str())),
                        ("output_index", index),
                        ("content_index", Value::from(0u64)),
                        ("delta", Value::from(d.reasoning.as_str())),
                    ],
                );
            }
        }
        if !d.content.is_empty() {
            if !matches!(self.cur, Some(Cur::Text { .. })) {
                ok &= self.close();
                ok &= self.open_text();
            }
            if let Some(Cur::Text { id, text }) = &mut self.cur {
                text.push_str(&d.content);
                let (id, index) = (id.clone(), Value::from(self.done.len() as u64));
                ok &= self.send(
                    "response.output_text.delta",
                    vec![
                        ("item_id", Value::from(id.as_str())),
                        ("output_index", index),
                        ("content_index", Value::from(0u64)),
                        ("delta", Value::from(d.content.as_str())),
                        ("logprobs", Value::Array(vec![])),
                    ],
                );
            }
        }
        for c in &d.calls {
            match c {
                ToolEv::Start { id: call_id, name, .. } => {
                    ok &= self.close();
                    let id = item_id("fc");
                    ok &= self.raw_item("response.output_item.added", &call_item(&id, call_id, name, "", "in_progress"));
                    self.cur = Some(Cur::Call { id, call_id: call_id.clone(), name: name.clone(), args: String::new() });
                }
                ToolEv::Args { args, .. } => {
                    if let Some(Cur::Call { id, args: all, .. }) = &mut self.cur {
                        all.push_str(args);
                        let (id, index) = (id.clone(), Value::from(self.done.len() as u64));
                        ok &= self.send(
                            "response.function_call_arguments.delta",
                            vec![("item_id", Value::from(id.as_str())), ("output_index", index), ("delta", Value::from(args.as_str()))],
                        );
                    }
                }
                ToolEv::Content(_) => {}
            }
        }
        ok
    }
}

/// One Responses request on `s`.
pub(super) fn handle(sh: &Shared, tx: &Sender<Job>, s: &mut TcpStream, req: &Req, id: u64) {
    let cors = req.cors.as_str();
    let Some(v) = std::str::from_utf8(&req.body).ok().and_then(|t| json::parse(t).ok()).filter(|v| matches!(v, Value::Object(_))) else {
        return fail(s, cors, 400, "the body is not a JSON object");
    };
    let o = match chat_of(&v) {
        Ok(o) => o,
        Err(m) => return fail(s, cors, 400, &m),
    };
    let Started { h, cfg, prompt_tokens, api } = match start_chat(sh, tx, &o, false) {
        Ok(st) => st,
        Err((code, m)) => return fail(s, cors, code, &m),
    };
    let abort = h.abort.clone();
    let ctx = Ctx::new(&v, &sh.name, id);
    let (out_tx, out_rx) = std::sync::mpsc::channel::<Out>();
    std::thread::scope(|sc| {
        let (tok, whole) = (&sh.tok, !api.stream);
        sc.spawn(move || drive(0, h, Choice::new(cfg), tok, whole, out_tx, None));
        if api.stream {
            let head = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\n{cors}Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n");
            let mut ev = Events { s, seq: 0, done: Vec::new(), cur: None };
            let begun = ev.s.write_all(head.as_bytes()).is_ok()
                && ev.raw("response.created", "response", &ctx.response("in_progress", None, &[], None, None))
                && ev.raw("response.in_progress", "response", &ctx.response("in_progress", None, &[], None, None));
            let mut ok = begun;
            while ok {
                match out_rx.recv_timeout(WAIT_TICK) {
                    Err(RecvTimeoutError::Timeout) => ok = sse_comment(ev.s) && !client_gone(ev.s),
                    Ok(Out::Delta(_, d)) => ok = ev.delta(&d),
                    Ok(Out::Done { last, done, .. }) => {
                        let mut fine = ev.delta(&last) && ev.close();
                        if ev.done.is_empty() {
                            // (a reply with no words: the message is there, empty, as OpenAI's is)
                            fine = fine && ev.open_text() && ev.close();
                        }
                        let (status, reason) =
                            status_of(if ev.done.iter().any(|i| i.contains("\"type\":\"function_call\"")) { "tool_calls" } else { done.finish_reason });
                        let kind = if status == "completed" { "response.completed" } else { "response.incomplete" };
                        let body = ctx.response(status, reason, &ev.done, Some((prompt_tokens, done.tokens)), None);
                        let _ = fine && ev.raw(kind, "response", &body) && ev.s.write_all(b"0\r\n\r\n").is_ok();
                        return;
                    }
                    Ok(Out::Failed(e)) => {
                        eprintln!("glyd: a request failed: {e}");
                        let body = ctx.response("failed", None, &ev.done, None, Some(&e));
                        let _ = ev.raw("response.failed", "response", &body) && ev.s.write_all(b"0\r\n\r\n").is_ok();
                        return;
                    }
                    Err(RecvTimeoutError::Disconnected) => return,
                }
            }
            // (the client went away: the engine stops after the pass it is in)
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
                Ok(Out::Done { done, thinking, .. }) => {
                    let mut output = Vec::new();
                    if let Some(t) = thinking.filter(|t| !t.is_empty() && !api.hide_reasoning) {
                        output.push(reasoning_item(&item_id("rs"), &t));
                    }
                    let text = done.content.as_deref().filter(|t| !t.is_empty());
                    if text.is_some() || done.calls.is_empty() {
                        output.push(message_item(&item_id("msg"), "completed", Some(text.unwrap_or(""))));
                    }
                    for c in &done.calls {
                        output.push(call_item(&item_id("fc"), &tools::new_id(), &c.name, &c.arguments, "completed"));
                    }
                    let (status, reason) = status_of(if done.calls.is_empty() { done.finish_reason } else { "tool_calls" });
                    let body = ctx.response(status, reason, &output, Some((prompt_tokens, done.tokens)), None);
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
    use crate::server::tests::{at, call, serve_script};

    /// The JSON of each event of a stream (its `data:` line).
    fn events(body: &str) -> Vec<Value> {
        body.lines().filter_map(|l| l.strip_prefix("data: ")).map(|d| json::parse(d).unwrap()).collect()
    }

    const CALL: &str = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
    const WEATHER: &str = r#"{"type":"function","name":"get_weather","description":"Get the weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]},"strict":true}"#;

    fn parse(t: &str) -> Value {
        json::parse(t).unwrap()
    }

    /// The request shapes the official SDK sends (`client.responses.create`): a string input with instructions; a list of items with messages of every role, a call and its output; a JSON schema in `text.format`.
    #[test]
    fn a_responses_request_as_the_chat_one() {
        let o = chat_of(&parse(r#"{"model":"m","input":"hi","instructions":"Be brief.","max_output_tokens":50,"temperature":0.2,"top_p":0.9,"store":false,"stream":true,"metadata":{"a":"b"}}"#)).unwrap();
        let roles: Vec<&str> = at(&o, &["messages"]).as_array().unwrap().iter().filter_map(|m| m.get("role").and_then(Value::as_str)).collect();
        assert_eq!(roles, ["system", "user"]);
        assert_eq!(at(&o, &["messages", "0", "content"]).as_str(), Some("Be brief."));
        assert_eq!((at(&o, &["max_tokens"]).as_u64(), at(&o, &["stream"]).as_bool()), (Some(50), Some(true)));
        assert!(o.get("store").is_none() && o.get("metadata").is_none() && o.get("model").is_none());

        let items = format!(
            r#"{{"input":[{{"role":"developer","content":"Rules."}},{{"type":"message","role":"user","content":[{{"type":"input_text","text":"What is it?"}},{{"type":"input_image","image_url":"data:image/png;base64,AAAA"}}]}},
            {{"type":"message","role":"assistant","content":[{{"type":"output_text","text":"Let me look.","annotations":[]}}]}},
            {{"type":"reasoning","id":"rs_1","summary":[]}},
            {{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{{\"city\":\"Paris\"}}"}},
            {{"type":"function_call_output","call_id":"call_1","output":"sunny"}}],
            "tools":[{WEATHER}],"tool_choice":{{"type":"function","name":"get_weather"}},"parallel_tool_calls":false,
            "text":{{"format":{{"type":"json_schema","name":"w","schema":{{"type":"object"}},"strict":true}}}},"reasoning":{{"effort":"minimal"}}}}"#
        );
        let o = chat_of(&parse(&items)).unwrap();
        let m = at(&o, &["messages"]).as_array().unwrap();
        let roles: Vec<&str> = m.iter().filter_map(|m| m.get("role").and_then(Value::as_str)).collect();
        assert_eq!(roles, ["system", "user", "assistant", "tool"], "{o:?}");
        // (the call joins the assistant's message before it)
        assert_eq!(at(&m[2], &["tool_calls", "0", "function", "name"]).as_str(), Some("get_weather"));
        assert_eq!(at(&m[2], &["tool_calls", "0", "id"]).as_str(), Some("call_1"));
        assert_eq!((at(&m[3], &["tool_call_id"]).as_str(), at(&m[3], &["content"]).as_str()), (Some("call_1"), Some("sunny")));
        assert_eq!(at(&m[1], &["content", "1", "type"]).as_str(), Some("image_url"));
        assert_eq!(at(&o, &["tools", "0", "function", "name"]).as_str(), Some("get_weather"));
        assert_eq!(at(&o, &["tools", "0", "function", "strict"]).as_bool(), Some(true));
        assert_eq!(at(&o, &["tool_choice", "function", "name"]).as_str(), Some("get_weather"));
        assert_eq!(at(&o, &["response_format", "json_schema", "schema", "type"]).as_str(), Some("object"));
        assert_eq!((at(&o, &["parallel_tool_calls"]).as_bool(), at(&o, &["reasoning_effort"]).as_str()), (Some(false), Some("none")));
        // (what the chat path reads from it)
        assert!(tools::template_messages_with(&o, Some(&mut Vec::new())).is_ok());
        assert!(tools::tools_of(&o).unwrap().is_some());
        assert!(constraint_spec(&o).unwrap().is_some());

        for (bad, want) in [
            (r#"{"input":"x","previous_response_id":"resp_1"}"#, "previous_response_id is not supported"),
            (r#"{"input":"x","conversation":"conv_1"}"#, "conversation is not supported"),
            (r#"{"input":"x","background":true}"#, "background is not supported"),
            (r#"{"input":"x","tools":[{"type":"web_search_preview"}]}"#, "tool of type `web_search_preview` is not supported"),
            (r#"{"input":[{"type":"item_reference","id":"msg_1"}]}"#, "item_reference is not supported"),
            (r#"{"input":[{"type":"computer_call"}]}"#, "`computer_call` is not supported"),
            (r#"{"input":"x","text":{"format":{"type":"xml"}}}"#, "text.format of type `xml`"),
            (r#"{}"#, "input: Field required"),
            (r#"{"input":[],"instructions":"only"}"#, "no message to answer"),
        ] {
            let e = chat_of(&parse(bad)).unwrap_err();
            assert!(e.contains(want), "{bad}: {e}");
        }
    }

    fn ask(addr: std::net::SocketAddr, body: &str) -> (u16, String) {
        call(addr, "/v1/responses", body)
    }

    /// A response as the SDK reads it: the object with its status, its output (the thinking, then the answer), its usage, and the request's settings repeated.
    #[test]
    fn a_response_whole() {
        let addr = serve_script("<think>\nsecret thought\n</think>\n\nthe answer");
        let (st, body) =
            ask(addr, r#"{"model":"m","input":"hi","instructions":"Be brief.","max_output_tokens":200,"temperature":0,"metadata":{"k":"v"},"user":"u1"}"#);
        assert_eq!(st, 200, "{body}");
        let d = parse(&body);
        assert_eq!(
            (at(&d, &["object"]).as_str(), at(&d, &["status"]).as_str(), at(&d, &["model"]).as_str()),
            (Some("response"), Some("completed"), Some("fake"))
        );
        assert!(
            at(&d, &["id"]).as_str().is_some_and(|i| i.starts_with("resp_")) && at(&d, &["created_at"]).as_u64().is_some_and(|t| t > 1_700_000_000),
            "{body}"
        );
        assert_eq!(
            (at(&d, &["output", "0", "type"]).as_str(), at(&d, &["output", "0", "content", "0", "type"]).as_str()),
            (Some("reasoning"), Some("reasoning_text"))
        );
        assert_eq!(at(&d, &["output", "0", "content", "0", "text"]).as_str().map(str::trim), Some("secret thought"));
        assert_eq!(
            (at(&d, &["output", "1", "type"]).as_str(), at(&d, &["output", "1", "role"]).as_str(), at(&d, &["output", "1", "status"]).as_str()),
            (Some("message"), Some("assistant"), Some("completed"))
        );
        assert_eq!(at(&d, &["output", "1", "content", "0", "type"]).as_str(), Some("output_text"));
        assert_eq!(at(&d, &["output", "1", "content", "0", "text"]).as_str().map(str::trim), Some("the answer"));
        assert!(at(&d, &["output", "1", "content", "0", "annotations"]).as_array().is_some_and(<[Value]>::is_empty));
        let (i, o, t) = (
            at(&d, &["usage", "input_tokens"]).as_u64().unwrap(),
            at(&d, &["usage", "output_tokens"]).as_u64().unwrap(),
            at(&d, &["usage", "total_tokens"]).as_u64().unwrap(),
        );
        assert!(i > 0 && o > 0 && t == i + o, "{body}");
        assert_eq!(at(&d, &["usage", "output_tokens_details", "reasoning_tokens"]).as_u64(), Some(0));
        assert_eq!(
            (at(&d, &["instructions"]).as_str(), at(&d, &["max_output_tokens"]).as_u64(), at(&d, &["user"]).as_str(), at(&d, &["metadata", "k"]).as_str()),
            (Some("Be brief."), Some(200), Some("u1"), Some("v"))
        );
        assert_eq!(
            (at(&d, &["store"]).as_bool(), at(&d, &["parallel_tool_calls"]).as_bool(), at(&d, &["error"]), at(&d, &["incomplete_details"])),
            (Some(false), Some(true), &Value::Null, &Value::Null)
        );
        assert_eq!(at(&d, &["text", "format", "type"]).as_str(), Some("text"));
        // cut at the length asked for: incomplete, and why
        let (st, body) = ask(addr, r#"{"input":"hi","max_output_tokens":3}"#);
        let d = parse(&body);
        assert_eq!(
            (st, at(&d, &["status"]).as_str(), at(&d, &["incomplete_details", "reason"]).as_str()),
            (200, Some("incomplete"), Some("max_output_tokens")),
            "{body}"
        );
    }

    /// The stream the SDK reads: `response.created` first, each item opened, its deltas, closed, and `response.completed` last with the whole response; every event numbered in order, and the output the deltas make the output the response has.
    #[test]
    fn a_response_streamed() {
        let addr = serve_script("<think>\nsecret thought\n</think>\n\nthe answer");
        let (st, body) = ask(addr, r#"{"input":"hi","stream":true,"max_output_tokens":200,"temperature":0}"#);
        assert_eq!(st, 200, "{body}");
        // (each event has its `event:` line, the type of its data)
        for e in body.split("\n\n").filter(|e| e.starts_with("event: ")) {
            let (head, data) = e.split_once("\ndata: ").unwrap();
            assert_eq!(&head["event: ".len()..], parse(data).get("type").and_then(Value::as_str).unwrap(), "{e}");
        }
        let evs = events(&body);
        let kinds: Vec<&str> = evs.iter().map(|e| at(e, &["type"]).as_str().unwrap()).collect();
        let mut seen = kinds.clone();
        seen.dedup();
        assert_eq!(
            seen,
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.reasoning_text.delta",
                "response.reasoning_text.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed"
            ],
            "{body}"
        );
        for (n, e) in evs.iter().enumerate() {
            assert_eq!(at(e, &["sequence_number"]).as_u64(), Some(n as u64), "{body}");
        }
        assert_eq!(at(&evs[0], &["response", "status"]).as_str(), Some("in_progress"));
        let (mut think, mut text) = (String::new(), String::new());
        for e in &evs {
            match at(e, &["type"]).as_str() {
                Some("response.reasoning_text.delta") => think.push_str(at(e, &["delta"]).as_str().unwrap()),
                Some("response.output_text.delta") => {
                    assert_eq!((at(e, &["output_index"]).as_u64(), at(e, &["content_index"]).as_u64()), (Some(1), Some(0)));
                    text.push_str(at(e, &["delta"]).as_str().unwrap())
                }
                _ => {}
            }
        }
        let last = evs.last().unwrap();
        let d = at(last, &["response"]);
        assert_eq!((think.trim(), text.trim()), ("secret thought", "the answer"));
        assert_eq!(at(d, &["status"]).as_str(), Some("completed"));
        assert_eq!(at(d, &["output", "0", "content", "0", "text"]).as_str(), Some(think.as_str()));
        assert_eq!(at(d, &["output", "1", "content", "0", "text"]).as_str(), Some(text.as_str()));
        assert!(at(d, &["usage", "output_tokens"]).as_u64().is_some_and(|n| n > 0));
        // the item the first event named is the one the last holds
        let added = evs.iter().find(|e| at(e, &["type"]).as_str() == Some("response.output_item.added")).unwrap();
        assert_eq!(at(added, &["item", "id"]).as_str(), at(d, &["output", "0", "id"]).as_str());
        // a stream cut at its length ends `response.incomplete`
        let (_, body) = ask(addr, r#"{"input":"hi","stream":true,"max_output_tokens":3}"#);
        let evs = events(&body);
        let last = evs.last().unwrap();
        assert_eq!(
            (at(last, &["type"]).as_str(), at(last, &["response", "incomplete_details", "reason"]).as_str()),
            (Some("response.incomplete"), Some("max_output_tokens")),
            "{body}"
        );
    }

    /// A reply with no words still has its message, with the text part opened before the text's events refer to it: a client that builds the response from the events (the SDK's stream helper) indexes the part.
    #[test]
    fn an_empty_reply_opens_its_text_part() {
        let addr = serve_script("<|im_end|>");
        let (st, body) = ask(addr, r#"{"input":"hi","stream":true,"max_output_tokens":50,"reasoning":{"effort":"none"}}"#);
        assert_eq!(st, 200, "{body}");
        let kinds: Vec<String> = events(&body).iter().map(|e| at(e, &["type"]).as_str().unwrap().to_string()).collect();
        let pos = |k: &str| kinds.iter().position(|x| x == k).unwrap_or_else(|| panic!("no {k} in {kinds:?}"));
        assert!(
            pos("response.output_item.added") < pos("response.content_part.added") && pos("response.content_part.added") < pos("response.output_text.done"),
            "{kinds:?}"
        );
        assert!(
            pos("response.output_text.done") < pos("response.content_part.done") && pos("response.content_part.done") < pos("response.output_item.done"),
            "{kinds:?}"
        );
        let last = events(&body).pop().unwrap();
        assert_eq!(at(&last, &["response", "output", "0", "content", "0", "type"]).as_str(), Some("output_text"), "{body}");
    }

    /// A call of a function: a `function_call` item with the call's id, the function's name and its arguments as the string they are, in the whole response and in the stream (the arguments in deltas, then whole).
    #[test]
    fn a_function_call() {
        let addr = serve_script(CALL);
        let q = |extra: &str| format!(r#"{{"input":"weather in Paris?","tools":[{WEATHER}],"max_output_tokens":200,"temperature":0{extra}}}"#);
        let (st, body) = ask(addr, &q(""));
        assert_eq!(st, 200, "{body}");
        let d = parse(&body);
        let call = d
            .get("output")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .find(|i| at(i, &["type"]).as_str() == Some("function_call"))
            .unwrap_or_else(|| panic!("{body}"));
        assert_eq!((at(call, &["name"]).as_str(), at(call, &["status"]).as_str()), (Some("get_weather"), Some("completed")));
        assert!(at(call, &["call_id"]).as_str().is_some_and(|c| !c.is_empty()) && at(call, &["id"]).as_str().is_some_and(|c| c.starts_with("fc_")));
        assert_eq!(parse(at(call, &["arguments"]).as_str().unwrap()), parse(r#"{"city":"Paris"}"#));
        assert_eq!(at(&d, &["status"]).as_str(), Some("completed"));
        assert_eq!(at(&d, &["tools", "0", "name"]).as_str(), Some("get_weather"));

        let (_, body) = ask(addr, &q(",\"stream\":true"));
        let evs = events(&body);
        let kinds: Vec<&str> = evs.iter().map(|e| at(e, &["type"]).as_str().unwrap()).collect();
        assert!(kinds.contains(&"response.function_call_arguments.delta") && kinds.contains(&"response.function_call_arguments.done"), "{body}");
        let args: String = evs
            .iter()
            .filter(|e| at(e, &["type"]).as_str() == Some("response.function_call_arguments.delta"))
            .map(|e| at(e, &["delta"]).as_str().unwrap().to_string())
            .collect();
        assert_eq!(parse(&args), parse(r#"{"city":"Paris"}"#), "{body}");
        let done = evs.iter().find(|e| at(e, &["type"]).as_str() == Some("response.function_call_arguments.done")).unwrap();
        assert_eq!((at(done, &["name"]).as_str(), parse(at(done, &["arguments"]).as_str().unwrap())), (Some("get_weather"), parse(r#"{"city":"Paris"}"#)));
        let item = evs
            .iter()
            .find(|e| at(e, &["type"]).as_str() == Some("response.output_item.added") && at(e, &["item", "type"]).as_str() == Some("function_call"))
            .unwrap();
        assert_eq!((at(item, &["item", "status"]).as_str(), at(item, &["item", "name"]).as_str()), (Some("in_progress"), Some("get_weather")));
        let last = at(evs.last().unwrap(), &["response"]);
        let n = at(last, &["output"]).as_array().unwrap().len();
        assert_eq!(at(last, &["output", &(n - 1).to_string(), "type"]).as_str(), Some("function_call"), "{body}");
    }

    /// A JSON schema in `text.format` reaches the grammar the chat path holds a reply to (the toy model has no end token, which a grammar needs: the refusal is the grammar's own), and the refusals come in OpenAI's error form with its status.
    #[test]
    fn a_schema_and_the_errors() {
        let addr = serve_script("{\"city\":\"Paris\"}");
        let (st, body) = ask(
            addr,
            r#"{"input":"hi","max_output_tokens":100,"temperature":0,"text":{"format":{"type":"json_schema","name":"w","strict":true,"schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}}}}"#,
        );
        assert!(st == 400 && body.contains("constrained decoding is not available for this model"), "{st} {body}");
        for (body, status, kind) in [
            (r#"{"input":"x","previous_response_id":"resp_1"}"#, 400, "invalid_request_error"),
            (r#"{"input":"x","temperature":"hot"}"#, 400, "invalid_request_error"),
            (r#"not json"#, 400, "invalid_request_error"),
            (r#"[1]"#, 400, "invalid_request_error"),
        ] {
            let (st, b) = ask(addr, body);
            let d = parse(&b);
            assert_eq!((st, at(&d, &["error", "type"]).as_str(), at(&d, &["error", "param"])), (status, Some(kind), &Value::Null), "{body}: {b}");
            assert!(at(&d, &["error", "message"]).as_str().is_some_and(|m| !m.is_empty()) && d.get("error").and_then(|e| e.get("code")).is_some(), "{b}");
        }
        // a prompt past the window is the context length's error, by the code the SDKs' users look for
        let long = "word ".repeat(40_000);
        let (st, b) = ask(addr, &format!(r#"{{"input":"{long}"}}"#));
        assert_eq!((st, at(&parse(&b), &["error", "code"]).as_str()), (400, Some("context_length_exceeded")), "{b}");
    }
}
