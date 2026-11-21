//! Tool calls as vLLM serves them, in the format of the model's chat template (Qwen3's `<tool_call>{json}</tool_call>`, its `hermes` parser, or the XML of Qwen3.5 and Nemotron 3, its `qwen3_coder` parser): a request's `tools` and `tool_choice` read and checked, its messages made into what the template reads
//! (a tool call's arguments an object, text parts joined), and the model's `<tool_call>` blocks read back as OpenAI tool calls, whole (the parser's `extract_tool_calls`) or as they are written
//! (`extract_tool_calls_streaming`: the name when it is complete, the arguments in the pieces the model writes them). A block that is not a call stays text, and nothing here panics on what a model writes.

use crate::chat::{py_dumps, Chat};
use crate::sampler::fresh_seed;
use glyd_gpu::json::{self, Value};

pub const OPEN: &str = "<tool_call>";
pub const CLOSE: &str = "</tool_call>";
const GLM_KEY: &str = "<arg_key>";
pub(crate) const FN_OPEN: &str = "<function=";
const FN_CLOSE: &str = "</function>";
const PARAM_OPEN: &str = "<parameter=";
const PARAM_CLOSE: &str = "</parameter>";
const DSML_OPEN: &str = "<｜DSML｜tool_calls>";

/// The tags of a format whose calls are blocks of functions of parameters: XML's, and DeepSeek-V4's DSML (`<｜DSML｜tool_calls>`, in it `<｜DSML｜invoke name="NAME">`, in that
/// `<｜DSML｜parameter name="KEY" string="true|false">VALUE</｜DSML｜parameter>`).
pub(crate) struct Tags {
    pub(crate) open: &'static str,
    pub(crate) close: &'static str,
    pub(crate) func: &'static str,
    /// The end of a function's name.
    pub(crate) named: &'static str,
    pub(crate) func_end: &'static str,
    pub(crate) param: &'static str,
    pub(crate) param_end: &'static str,
}

const XML: Tags = Tags { open: OPEN, close: CLOSE, func: FN_OPEN, named: ">", func_end: FN_CLOSE, param: PARAM_OPEN, param_end: PARAM_CLOSE };
const DSML: Tags = Tags {
    open: DSML_OPEN,
    close: "</｜DSML｜tool_calls>",
    func: "<｜DSML｜invoke name=\"",
    named: "\">",
    func_end: "</｜DSML｜invoke>",
    param: "<｜DSML｜parameter name=\"",
    param_end: "</｜DSML｜parameter>",
};

/// Muse's ATEM calls: `<atem:function_calls>`, in it `<atem:invoke name="NAME">`, in that `<atem:parameter name="KEY">VALUE</atem:parameter>` (a string and a scalar as it is, a list and an object JSON; the value is not stripped).
const ATEM: Tags = Tags {
    open: "<atem:function_calls>",
    close: "</atem:function_calls>",
    func: "<atem:invoke name=\"",
    named: "\">",
    func_end: "</atem:invoke>",
    param: "<atem:parameter name=\"",
    param_end: "</atem:parameter>",
};

/// How a request chooses among its tools.
#[derive(Clone, Debug, PartialEq)]
pub enum ToolChoice {
    /// The tools are shown to the model and its text is not read for calls.
    None,
    Auto,
    /// The reply is a call: it begins with the call's opening.
    Required,
    /// The reply is a call of this function.
    Named(String),
}

/// How a model writes its tool calls, as its chat template shows them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// Qwen3's `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` (vLLM's `hermes` parser).
    Hermes,
    /// Qwen3.5's and Nemotron 3's `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>` (vLLM's `qwen3_coder` parser).
    Xml,
    /// DeepSeek-V4's DSML blocks (see `Tags`).
    Dsml,
    /// GLM's `<tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>...</tool_call>`, a value a string as it is or JSON (typed by the tool's schema).
    Glm,
    /// Muse's ATEM blocks (see `ATEM`).
    Atem,
    /// Llama 3.1 and later: `{"name": NAME, "parameters": {...}}` (`"arguments"` too) as the whole reply or after `<|python_tag|>`, several separated by `; ` (vLLM's `llama3_json` parser).
    Json,
}

/// What Llama 3.1+ writes before a call (and what its template mentions).
pub(crate) const PYTHON_TAG: &str = "<|python_tag|>";

impl Format {
    /// The format of a chat template's tool calls: DSML where it writes `｜DSML｜`, GLM's where it writes `<arg_key>`, XML where it writes `<function=`, else hermes.
    pub fn of(template: &Chat) -> Format {
        // (Kimi-K3's and Gemma 4's replies are read through reader::Reader, which writes their calls as hermes ones)
        if template.is_xtml() || template.is_gemma() {
            return Format::Hermes;
        }
        if template.mentions(PYTHON_TAG) && template.mentions("\"parameters\": dictionary of argument name") {
            return Format::Json;
        }
        if template.mentions("<atem:function_calls>") {
            Format::Atem
        } else if template.mentions("｜DSML｜") {
            Format::Dsml
        } else if template.mentions(GLM_KEY) {
            Format::Glm
        } else if template.mentions(FN_OPEN) {
            Format::Xml
        } else {
            Format::Hermes
        }
    }

    /// The text that opens a reply's calls.
    pub fn open(self) -> &'static str {
        match self {
            Format::Json => PYTHON_TAG,
            _ => self.tags().open,
        }
    }

    pub(crate) fn tags(self) -> &'static Tags {
        match self {
            Format::Dsml => &DSML,
            Format::Atem => &ATEM,
            _ => &XML,
        }
    }
}

/// A request's tools, as the chat template is given them, and how they are used.
#[derive(Clone, Debug)]
pub struct Tools {
    pub list: Vec<Value>,
    pub choice: ToolChoice,
    /// More than one call in a reply (`parallel_tool_calls`; false, or a named tool, keeps the first).
    pub parallel: bool,
}

/// A random id for a tool call, as vLLM writes them: `chatcmpl-tool-` and 16 hex digits.
pub fn new_id() -> String {
    format!("chatcmpl-tool-{:016x}", fresh_seed())
}

pub(crate) fn obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

/// The request's `tools` and `tool_choice` as vLLM reads them, or the message of its refusal. None where there are no tools. Each tool becomes what the template is given: `{"type": "function", "function":
/// {"name", "description", "parameters"}}` (a missing description or parameters as null, `strict` where given), as vLLM's own model dumps it.
pub fn tools_of(v: &Value) -> Result<Option<Tools>, String> {
    let choice = v.get("tool_choice").filter(|c| **c != Value::Null);
    let list = match v.get("tools") {
        None | Some(Value::Null) => {
            return match choice {
                Some(Value::String(s)) if s == "none" => Ok(None),
                Some(_) => Err("When using `tool_choice`, `tools` must be set.".to_string()),
                None => Ok(None),
            }
        }
        Some(Value::Array(a)) if a.is_empty() => {
            return Err("`tools` must not be an empty array. Either provide at least one tool or omit the field entirely.".to_string())
        }
        Some(Value::Array(a)) => a,
        Some(_) => return Err("`tools` must be a list.".to_string()),
    };
    let mut out = Vec::new();
    for t in list {
        let ty = t.get("type").and_then(Value::as_str).unwrap_or("function");
        let f = t.get("function").filter(|f| matches!(f, Value::Object(_)));
        let (true, Some(f)) = (ty == "function", f) else { return Err("each of `tools` is {\"type\": \"function\", \"function\": {...}}".to_string()) };
        let Some(name) = f.get("name").and_then(Value::as_str) else { return Err("a tool's function needs a `name`".to_string()) };
        let mut pairs = vec![
            ("name", Value::from(name)),
            ("description", f.get("description").cloned().filter(|d| matches!(d, Value::String(_))).unwrap_or(Value::Null)),
            ("parameters", f.get("parameters").cloned().filter(|p| matches!(p, Value::Object(_))).unwrap_or(Value::Null)),
        ];
        if let Some(b @ Value::Bool(_)) = f.get("strict") {
            pairs.push(("strict", b.clone()));
        }
        out.push(obj(vec![("type", Value::from("function")), ("function", obj(pairs))]));
    }
    let names: Vec<&str> = out.iter().filter_map(|t| t.get("function").and_then(|f| f.get("name")).and_then(Value::as_str)).collect();
    let choice = match choice {
        None => ToolChoice::Auto,
        Some(Value::String(s)) => match s.as_str() {
            "none" => ToolChoice::None,
            "auto" => ToolChoice::Auto,
            "required" => ToolChoice::Required,
            other => return Err(format!("Invalid value for `tool_choice`: {other}! Only named tools, \"none\", \"auto\" or \"required\" are supported.")),
        },
        Some(c @ Value::Object(_)) => {
            let correct = "Correct usage: `{\"type\": \"function\", \"function\": {\"name\": \"my_function\"}}`";
            let Some(f @ Value::Object(_)) = c.get("function") else {
                return Err(format!("Invalid value for `function`: `{}` in `tool_choice`! {correct}", c.get("function").map_or("None".to_string(), py_dumps)));
            };
            let Some(n) = f.get("name") else { return Err(format!("Expected field `name` in `function` in `tool_choice`! {correct}")) };
            match n.as_str() {
                Some(n) if !n.is_empty() => {
                    if !names.contains(&n) {
                        return Err("The tool specified in `tool_choice` does not match any of the specified `tools`".to_string());
                    }
                    ToolChoice::Named(n.to_string())
                }
                _ => return Err(format!("Invalid `name` in `function`: `{}` in `tool_choice`! {correct}", py_dumps(n))),
            }
        }
        Some(other) => {
            return Err(format!("Invalid value for `tool_choice`: {}! Only named tools, \"none\", \"auto\" or \"required\" are supported.", py_dumps(other)))
        }
    };
    // a named tool is one call to it, as vLLM's guided decoding makes it
    let parallel = !matches!(choice, ToolChoice::Named(_)) && v.get("parallel_tool_calls").and_then(Value::as_bool) != Some(false);
    Ok(Some(Tools { list: out, choice, parallel }))
}

/// The text of the content parts of a message, joined by a new line as vLLM joins them for a template that reads a string; a part that is not text is refused (the model reads text).
/// Where the model reads images (`images`), an image part (`image_url`: a URL, or an object with one) is the template's place for an image, its URL pushed to `images`, and a message
/// with one has its parts joined as the template joins a list of them (with nothing between).
fn parts_text(parts: &[Value], mut images: Option<&mut Vec<String>>) -> Result<String, String> {
    let mut texts = Vec::new();
    let mut saw_image = false;
    for p in parts {
        match p {
            Value::String(s) => texts.push(s.as_str()),
            Value::Object(_) => {
                let ty = p.get("type").and_then(Value::as_str).unwrap_or(if p.get("text").is_some() { "text" } else { "" });
                if let (Some(list), "image_url") = (images.as_deref_mut(), ty) {
                    let url =
                        p.get("image_url").and_then(|u| u.as_str().or_else(|| u.get("url").and_then(Value::as_str))).ok_or("an image_url part with no url")?;
                    list.push(url.to_string());
                    texts.push(IMAGE_PLACE);
                    saw_image = true;
                    continue;
                }
                if ty != "text" {
                    return Err(format!(
                        "a message part of type {ty:?}: this model reads text{}",
                        if images.is_some() { " and image_url parts" } else { " only" }
                    ));
                }
                texts.push(p.get("text").and_then(Value::as_str).ok_or("a text part with no text")?);
            }
            _ => return Err("a message part that is not an object".to_string()),
        }
    }
    Ok(texts.join(if saw_image { "" } else { "\n" }))
}

/// An image's place in the text as Qwen's chat templates write it: the image's tokens are the one `<|image_pad|>` token repeated (the server's, once the image is read).
pub const IMAGE_PLACE: &str = "<|vision_start|><|image_pad|><|vision_end|>";

/// An assistant's tool call as the template reads it: the arguments an object (a string is read as JSON; one that is not an object, or not JSON, is an empty object, as vLLM does so that a conversation with a bad
/// turn can go on).
fn call_for_template(c: &Value) -> Result<Value, String> {
    let Value::Object(pairs) = c else { return Err("assistant tool_calls entries must be objects.".to_string()) };
    let f = c.get("function").filter(|f| matches!(f, Value::Object(_)));
    if c.get("type").and_then(Value::as_str).unwrap_or("function") != "function" || f.is_none() {
        return Err("chat completions only support assistant tool_calls of type 'function'.".to_string());
    }
    let args = match f.and_then(|f| f.get("arguments")) {
        Some(Value::String(s)) if !s.is_empty() => json::parse(s).ok().filter(|v| matches!(v, Value::Object(_))).unwrap_or(Value::Object(vec![])),
        Some(o @ Value::Object(_)) => o.clone(),
        _ => Value::Object(vec![]),
    };
    let Some(Value::Object(fp)) = f else { unreachable!() };
    let mut fp = fp.clone();
    match fp.iter_mut().find(|(k, _)| k == "arguments") {
        Some(e) => e.1 = args,
        None => fp.push(("arguments".to_string(), args)),
    }
    Ok(Value::Object(pairs.iter().map(|(k, x)| if k == "function" { (k.clone(), Value::Object(fp.clone())) } else { (k.clone(), x.clone()) }).collect()))
}

/// The request's messages as the chat template reads them (as vLLM makes them): `content` a string (null is empty, text parts are joined by new lines), an assistant's `tool_calls` with their arguments as objects
/// (an empty list is none), its thinking as `reasoning_content` and `reasoning` both, a tool message's `tool_call_id`, and a `name`.
pub fn template_messages(v: &Value) -> Result<Vec<Value>, String> {
    template_messages_with(v, None)
}

/// `template_messages` where the model reads images too: their places in the messages' text, their URLs pushed to `images` in order (a system message may have none, as the template has it).
pub fn template_messages_with(v: &Value, mut images: Option<&mut Vec<String>>) -> Result<Vec<Value>, String> {
    let list = v.get("messages").and_then(Value::as_array).ok_or("no messages")?;
    if list.is_empty() {
        return Err("no messages".to_string());
    }
    let mut out = Vec::new();
    for m in list {
        if !matches!(m, Value::Object(_)) {
            return Err("a message that is not an object".to_string());
        }
        let role = m.get("role").and_then(Value::as_str).ok_or("a message with no role")?;
        let content = match m.get("content") {
            None | Some(Value::Null) => String::new(),
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(parts)) => {
                let before = images.as_deref().map_or(0, Vec::len);
                let t = parts_text(parts, images.as_deref_mut())?;
                if role == "system" && images.as_deref().map_or(0, Vec::len) > before {
                    return Err("System message cannot contain images.".to_string());
                }
                t
            }
            Some(_) => return Err("a message whose content is not text".to_string()),
        };
        let mut o = vec![("role", Value::from(role)), ("content", Value::String(content))];
        if role == "assistant" {
            if let Some(Value::Array(calls)) = m.get("tool_calls") {
                if !calls.is_empty() {
                    o.push(("tool_calls", Value::Array(calls.iter().map(call_for_template).collect::<Result<_, _>>()?)));
                }
            }
            if let Some(r) = m.get("reasoning").or_else(|| m.get("reasoning_content")).and_then(Value::as_str) {
                o.push(("reasoning", Value::from(r)));
                o.push(("reasoning_content", Value::from(r)));
            }
        }
        if role == "tool" {
            if let Some(id) = m.get("tool_call_id") {
                o.push(("tool_call_id", id.clone()));
            }
        }
        if let Some(n @ Value::String(_)) = m.get("name") {
            o.push(("name", n.clone()));
        }
        out.push(obj(o));
    }
    Ok(out)
}

/// What the prompt gets after the generation prompt where the reply must be a call: its opening and, for a named function, the name and where the arguments begin (hermes: their key, the reply is then the arguments; XML: the line
/// after the name, the reply is then the parameters); the model's words go on from there. Those words are part of the reply's text for the parsers.
pub fn forced_prefix(choice: &ToolChoice, format: Format) -> Option<String> {
    match (choice, format) {
        (ToolChoice::Required, Format::Hermes) => Some(format!("{OPEN}\n{{\"name\": \"")),
        (ToolChoice::Named(n), Format::Hermes) => Some(format!("{OPEN}\n{{\"name\": {}, \"arguments\": ", py_dumps(&Value::from(n.as_str())))),
        (ToolChoice::Required, Format::Json) => Some("{\"name\": \"".to_string()),
        (ToolChoice::Named(n), Format::Json) => Some(format!("{{\"name\": {}, \"parameters\": ", py_dumps(&Value::from(n.as_str())))),
        (ToolChoice::Required, Format::Xml) => Some(format!("{OPEN}\n{FN_OPEN}")),
        (ToolChoice::Named(n), Format::Xml) => Some(format!("{OPEN}\n{FN_OPEN}{n}>\n")),
        (ToolChoice::Required, Format::Dsml) => Some(format!("\n\n{DSML_OPEN}\n")),
        (ToolChoice::Named(n), Format::Dsml) => Some(format!("\n\n{DSML_OPEN}\n{}{n}\">\n", DSML.func)),
        (ToolChoice::Required, Format::Atem) => Some(format!("{}\n", ATEM.open)),
        (ToolChoice::Named(n), Format::Atem) => Some(format!("{}\n{}{n}\">\n", ATEM.open, ATEM.func)),
        (ToolChoice::Required, Format::Glm) => Some(OPEN.to_string()),
        (ToolChoice::Named(n), Format::Glm) => Some(format!("{OPEN}{n}")),
        _ => None,
    }
}

/// A tool call read from a reply: the function and its arguments as the JSON text a client parses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Call {
    pub name: String,
    pub arguments: String,
}

/// What a whole reply holds: whether it called tools, the calls, and the text before the first (None where it is empty or only white space).
#[derive(Clone, Debug, PartialEq)]
pub struct Extracted {
    pub called: bool,
    pub calls: Vec<Call>,
    pub content: Option<String>,
}

/// The reply's tool calls, read in the model's format (`tools`: the request's, for the types of an XML call's parameters).
pub fn extract(text: &str, format: Format, tools: &[Value]) -> Extracted {
    let mut e = match format {
        Format::Hermes => extract_hermes(text),
        Format::Xml | Format::Dsml | Format::Atem => extract_xml(text, tools, format),
        Format::Glm => extract_glm(text, tools),
        Format::Json => extract_json(text),
    };
    for c in &mut e.calls {
        c.name = known_name(&c.name, tools);
    }
    e
}

/// A call's name as the request's tools have it: the model's own where it is one of them, else without a `functions.` namespace where that one is (a model made to
/// begin a call at once, with no thinking first, can write OpenAI's: Nemotron 3.5 Lightning with `tool_choice` required), else as written.
fn known_name(name: &str, tools: &[Value]) -> String {
    let has = |n: &str| tools.iter().filter_map(|t| t.get("function")).any(|f| f.get("name").and_then(Value::as_str) == Some(n));
    match name.strip_prefix("functions.") {
        Some(s) if !has(name) && has(s) => s.to_string(),
        _ => name.to_string(),
    }
}

/// The reply's tool calls, as vLLM's hermes parser reads them from the whole text: every `<tool_call>` block (to its `</tool_call>`, or to the end) is JSON with a `name` and `arguments`, the arguments written
/// again as Python's `json.dumps` writes them. If one block is not that, there are no calls and all the text is content, and nothing fails.
fn extract_hermes(text: &str) -> Extracted {
    let plain = || Extracted { called: false, calls: vec![], content: Some(text.to_string()) };
    let Some(first) = text.find(OPEN) else { return plain() };
    let (mut calls, mut pos) = (Vec::new(), 0);
    while let Some(r) = text[pos..].find(OPEN) {
        let from = pos + r + OPEN.len();
        let (block, next) = match text[from..].find(CLOSE) {
            Some(e) => (&text[from..from + e], from + e + CLOSE.len()),
            None => (&text[from..], text.len()),
        };
        pos = next;
        let Ok(v) = json::parse(block) else { return plain() };
        let (Some(name), Some(args)) = (v.get("name").and_then(Value::as_str), v.get("arguments")) else { return plain() };
        calls.push(Call { name: name.to_string(), arguments: py_dumps(args) });
    }
    let content = Some(text[..first].to_string()).filter(|c| !c.trim().is_empty());
    Extracted { called: true, calls, content }
}

/// The JSON object at the start of `s` (a `{`): its value and where it ends, None where it is not whole (or not JSON). The braces are counted outside strings.
fn json_object(s: &str) -> Option<(Value, usize)> {
    if !s.starts_with('{') {
        return None;
    }
    let (mut depth, mut in_str, mut esc) = (0usize, false, false);
    for (i, c) in s.char_indices() {
        if in_str {
            match (esc, c) {
                (true, _) => esc = false,
                (false, '\\') => esc = true,
                (false, '"') => in_str = false,
                _ => {}
            }
        } else {
            match c {
                '"' => in_str = true,
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return json::parse(&s[..=i]).ok().map(|v| (v, i + 1));
                    }
                }
                _ => {}
            }
        }
    }
    None
}

/// The calls of a Llama 3.1+ reply as far as it is whole: the objects (each with a `name` and its `parameters` or `arguments`), separated by `;`, after an optional `<|python_tag|>`; the calls, and where the text
/// that is not one begins (None: all of it is calls, or `None` calls and a start of 0 for a reply that does not begin as a call).
fn json_calls(text: &str) -> (Vec<Call>, Option<usize>) {
    let body_from = if text.starts_with(PYTHON_TAG) { PYTHON_TAG.len() } else { 0 };
    if !text[body_from..].starts_with('{') {
        return (Vec::new(), Some(0));
    }
    let (mut calls, mut pos) = (Vec::new(), body_from);
    while pos < text.len() {
        let Some((v, end)) = json_object(&text[pos..]) else { return (calls, Some(pos)) };
        let (Some(name), Some(args)) = (v.get("name").and_then(Value::as_str), v.get("parameters").or_else(|| v.get("arguments"))) else {
            return (calls, Some(pos));
        };
        calls.push(Call { name: name.to_string(), arguments: py_dumps(args) });
        pos += end;
        // (calls are separated by `;`)
        let rest = &text[pos..];
        pos += rest.len() - rest.trim_start_matches([';', ' ', '\n']).len();
    }
    (calls, None)
}

/// A Llama 3.1+ reply's tool calls, as vLLM's `llama3_json` parser reads the whole text: a reply that begins with `<|python_tag|>` or `{` is JSON calls, one or several; if any of it is not a call there are no calls and all the
/// text is content, and nothing fails.
fn extract_json(text: &str) -> Extracted {
    let (calls, rest) = json_calls(text);
    if rest.is_some() || calls.is_empty() {
        return Extracted { called: false, calls: vec![], content: Some(text.to_string()) };
    }
    Extracted { called: true, calls, content: None }
}

/// GLM's blocks of `text`: each `<tool_call>`'s inside's inside (to its `</tool_call>`, or to the end: not closed), where it starts and ends.
fn glm_blocks(text: &str) -> Vec<(usize, usize, &str, bool)> {
    let (mut out, mut pos) = (Vec::new(), 0);
    while let Some(r) = text[pos..].find(OPEN) {
        let (from, at) = (pos + r, pos + r + OPEN.len());
        match text[at..].find(CLOSE) {
            Some(e) => {
                out.push((from, at + e + CLOSE.len(), &text[at..at + e], true));
                pos = at + e + CLOSE.len();
            }
            None => {
                out.push((from, text.len(), &text[at..], false));
                break;
            }
        }
    }
    out
}

/// A GLM block's call: its name (the text before the first `<arg_key>`) and its arguments, each value typed by the tool's schema as an XML call's are; None where it
/// has no name or a key without its value.
fn glm_call(block: &str, tools: &[Value]) -> Option<Call> {
    let (name, mut rest) = match block.find(GLM_KEY) {
        Some(i) => (block[..i].trim(), &block[i..]),
        None => (block.trim(), ""),
    };
    if name.is_empty() || name.contains('<') {
        return None;
    }
    let name = known_name(name, tools);
    let props = properties(tools, &name);
    let mut items: Vec<(String, String)> = Vec::new();
    while let Some(k0) = rest.find(GLM_KEY) {
        let after = &rest[k0 + GLM_KEY.len()..];
        let ke = after.find("</arg_key>")?;
        let key = after[..ke].trim().to_string();
        let after = &after[ke + "</arg_key>".len()..];
        let v0 = after.find("<arg_value>")? + "<arg_value>".len();
        let ve = after[v0..].find("</arg_value>")?;
        let item = typed(after[v0..v0 + ve].trim_matches('\n'), props.and_then(|p| p.get(&key)));
        match items.iter_mut().find(|(k, _)| *k == key) {
            Some(x) => x.1 = item,
            None => items.push((key, item)),
        }
        rest = &after[v0 + ve + "</arg_value>".len()..];
    }
    let args = items.iter().map(|(k, v)| format!("{}: {v}", py_dumps(&Value::from(k.as_str())))).collect::<Vec<_>>().join(", ");
    Some(Call { name, arguments: format!("{{{args}}}") })
}

/// The reply's GLM calls: every block that is one (else all the text is content, as for the other formats), and the text before the first.
fn extract_glm(text: &str, tools: &[Value]) -> Extracted {
    let plain = || Extracted { called: false, calls: vec![], content: Some(text.to_string()) };
    let blocks = glm_blocks(text);
    let Some(first) = blocks.first().map(|b| b.0) else { return plain() };
    let mut calls = Vec::new();
    for (_, _, inner, _) in &blocks {
        let Some(c) = glm_call(inner, tools) else { return plain() };
        calls.push(c);
    }
    let content = Some(text[..first].to_string()).filter(|c| !c.trim().is_empty());
    Extracted { called: true, calls, content }
}

/// The `properties` of the parameters of the tool `name` among a request's tools (None: there is no such tool, or its schema lists none).
fn properties<'a>(tools: &'a [Value], name: &str) -> Option<&'a Value> {
    let f = tools.iter().filter_map(|t| t.get("function")).find(|f| f.get("name").and_then(Value::as_str) == Some(name))?;
    f.get("parameters")?.get("properties")
}

/// A parameter's text as the JSON text of its value, typed by the parameter's schema (`schema`) as vLLM's `qwen3_coder` parser types it: `null` in any case is null; a string, and a parameter with no schema (it, or its function,
/// is not among the tools), stay text; an integer or a number is read where it is one (else it stays text; a number with no fraction is an integer); a boolean is true for `true` in any case and false for any other text;
/// an object, an array, an `anyOf` and any other type are read as JSON where the text is JSON (vLLM goes on to Python's `ast.literal_eval`, which is not done here), else they stay text.
fn typed(v: &str, schema: Option<&Value>) -> String {
    let text = || py_dumps(&Value::from(v));
    if v.to_lowercase() == "null" {
        return "null".to_string();
    }
    let Some(schema) = schema else { return text() };
    let ty = match (schema.get("type"), schema.get("anyOf")) {
        (Some(Value::String(t)), _) => t.trim().to_lowercase(),
        (Some(other), _) => py_dumps(other).to_lowercase(),
        (None, Some(_)) => "object".to_string(),
        (None, None) => "string".to_string(),
    };
    let starts = |words: &[&str]| words.iter().any(|w| ty.starts_with(w));
    if ["string", "str", "text", "varchar", "char", "enum"].contains(&ty.as_str()) {
        text()
    } else if starts(&["int", "uint", "long", "short", "unsigned"]) {
        v.trim().parse::<i128>().map_or_else(|_| text(), |n| n.to_string())
    } else if starts(&["num", "float"]) {
        match v.trim().parse::<f64>() {
            Ok(f) if f.is_finite() && f.fract() == 0.0 => format!("{:.0}", f + 0.0), // (+ 0.0: -0.0 is written 0, as Python's int writes it)
            Ok(f) if f.is_finite() => f.to_string(),
            _ => text(),
        }
    } else if ["boolean", "bool", "binary"].contains(&ty.as_str()) {
        (v.to_lowercase() == "true").to_string()
    } else {
        json::parse(v).map_or_else(|_| text(), |j| py_dumps(&j))
    }
}

/// A parameter as an item of its call's arguments: `"KEY": VALUE`, the value typed by the schema `props` (the function's `properties`) gives its key.
fn item(k: &str, v: &str, props: Option<&Value>) -> String {
    format!("{}: {}", py_dumps(&Value::from(k)), typed(v, props.and_then(|p| p.get(k))))
}

/// A parameter as the text from its `<parameter=` to its end, `KEY>VALUE`: the key and the value, which has one new line taken off each end (None where there is no `>`).
fn param_of(group: &str) -> Option<(&str, &str)> {
    let (k, v) = group.split_once('>')?;
    let v = v.strip_prefix('\n').unwrap_or(v);
    Some((k, v.strip_suffix('\n').unwrap_or(v)))
}

/// A parameter of a call as its key and its item of the arguments (`item`). XML: `KEY>VALUE` (`param_of`), typed by the schema `props`. DSML: `KEY" string="true|false">VALUE`, a string as it
/// is, else JSON (text that is not JSON stays a string). None where it is not that.
fn param_item<'a>(format: Format, g: &'a str, props: Option<&Value>) -> Option<(&'a str, String)> {
    if format == Format::Atem {
        // (`KEY">VALUE`: the value as written, typed by the schema)
        let (k, v) = g.split_once("\">")?;
        return Some((k, item(k, v, props)));
    }
    if format != Format::Dsml {
        let (k, v) = param_of(g)?;
        return Some((k, item(k, v, props)));
    }
    let (k, rest) = g.split_once("\" string=\"")?;
    let (flag, v) = rest.split_once("\">")?;
    let value = match (flag, json::parse(v)) {
        ("false", Ok(j)) => py_dumps(&j),
        _ => py_dumps(&Value::from(v)),
    };
    Some((k, format!("{}: {value}", py_dumps(&Value::from(k)))))
}

/// A `<function=NAME>` of an XML reply: its parameters whose end is written, each as `KEY>VALUE` (see `param_of`), and whether it has ended (its `</function>` is written, or the `</tool_call>` of its block).
struct XmlFn<'a> {
    name: &'a str,
    params: Vec<&'a str>,
    closed: bool,
}

/// A `<tool_call>` block of an XML reply: where it is in the text, and its functions (those whose name is written).
struct XmlBlock<'a> {
    from: usize,
    to: usize,
    funcs: Vec<XmlFn<'a>>,
}

/// The parameters of a function's text, each as `KEY>VALUE`: from a `<parameter=` to the first of its `</parameter>` and the next `<parameter=`, or, where `closed` (the function has ended), to the end of the text less its last
/// new line (the end of vLLM's pattern is before it); a parameter that has no end yet is not in the list.
fn xml_params<'a>(f: &'a str, closed: bool, tags: &Tags) -> Vec<&'a str> {
    let (mut out, mut pos) = (Vec::new(), 0);
    while let Some(r) = f[pos..].find(tags.param) {
        let start = pos + r + tags.param.len();
        let rest = &f[start..];
        let open = rest.find(tags.param);
        // (a `</parameter>` after the next `<parameter=` is not this one's end: the search stops there)
        match rest[..open.unwrap_or(rest.len())].find(tags.param_end).map(|e| (e, e + tags.param_end.len())).or(open.map(|e| (e, e))) {
            Some((end, next)) => {
                out.push(&rest[..end]);
                pos = start + next;
            }
            None if closed => {
                out.push(rest.strip_suffix('\n').unwrap_or(rest));
                break;
            }
            None => break,
        }
    }
    out
}

/// The functions of a block's text: each `<function=` to the first `</function>` after it (else to the end of the block, which ends it where `ended`), kept where its `>` is written.
fn xml_funcs<'a>(block: &'a str, ended: bool, tags: &Tags) -> Vec<XmlFn<'a>> {
    let (mut out, mut pos) = (Vec::new(), 0);
    while let Some(r) = block[pos..].find(tags.func) {
        let start = pos + r + tags.func.len();
        let (end, next, closed) = match block[start..].find(tags.func_end) {
            Some(e) => (start + e, start + e + tags.func_end.len(), true),
            None => (block.len(), block.len(), ended),
        };
        pos = next;
        let f = &block[start..end];
        if let Some(gt) = f.find(tags.named) {
            out.push(XmlFn { name: &f[..gt], params: xml_params(&f[gt + tags.named.len()..], closed, tags), closed });
        }
    }
    out
}

/// The blocks of an XML reply as vLLM's `qwen3_coder` parser finds them: each `<tool_call>` to its `</tool_call>`, or to the end of the text. `last`: the text is the whole reply, so a block, a function or a parameter that has
/// no end runs to the end of the text (less its last new line, for a block), and a reply with no `<tool_call>` is one block (the parser reads the functions of such a text too).
fn xml_scan<'a>(text: &'a str, last: bool, tags: &Tags) -> Vec<XmlBlock<'a>> {
    let (mut out, mut pos) = (Vec::new(), 0);
    while let Some(r) = text[pos..].find(tags.open) {
        let (from, body) = (pos + r, pos + r + tags.open.len());
        match text[body..].find(tags.close) {
            Some(e) => {
                out.push(XmlBlock { from, to: body + e + tags.close.len(), funcs: xml_funcs(&text[body..body + e], true, tags) });
                pos = body + e + tags.close.len();
            }
            None => {
                let rest = &text[body..];
                out.push(XmlBlock { from, to: text.len(), funcs: xml_funcs(if last { rest.strip_suffix('\n').unwrap_or(rest) } else { rest }, last, tags) });
                break;
            }
        }
    }
    if out.is_empty() && last {
        out.push(XmlBlock { from: 0, to: text.len(), funcs: xml_funcs(text, true, tags) });
    }
    out
}

/// The reply's tool calls, as vLLM's `qwen3_coder` parser (v0.21.0) reads them from the whole text: each `<function=NAME>` of a `<tool_call>` block is a call, and each `<parameter=KEY>` in it a key of its arguments, the value
/// with one new line taken off each end and typed by the tool's schema (`typed`); the arguments are written as Python's `json.dumps` writes them, a key that comes again keeping its place and taking the last value. The text
/// before the first `<tool_call>` is the content. A reply with no call, one with a parameter that has no `>`, and one whose functions all lack a `>` (vLLM drops its text) are all content, and nothing fails.
/// DSML (`format`) is read the same way, its parameters by `param_item`, and the two new lines before its block are not content.
fn extract_xml(text: &str, tools: &[Value], format: Format) -> Extracted {
    let plain = || Extracted { called: false, calls: vec![], content: Some(text.to_string()) };
    let tags = format.tags();
    if !text.contains(tags.func) {
        return plain();
    }
    let mut calls = Vec::new();
    for f in xml_scan(text, true, tags).iter().flat_map(|b| &b.funcs) {
        let props = properties(tools, &known_name(f.name, tools));
        let mut items: Vec<(&str, String)> = Vec::new();
        for g in &f.params {
            let Some((k, it)) = param_item(format, g, props) else { return plain() };
            match items.iter_mut().find(|(n, _)| *n == k) {
                Some(x) => x.1 = it,
                None => items.push((k, it)),
            }
        }
        calls.push(Call { name: f.name.to_string(), arguments: format!("{{{}}}", items.into_iter().map(|(_, it)| it).collect::<Vec<_>>().join(", ")) });
    }
    if calls.is_empty() {
        return plain();
    }
    let first = text.find(tags.open).or_else(|| text.find(tags.func)).unwrap_or(0);
    let before = if format == Format::Dsml { text[..first].strip_suffix("\n\n").unwrap_or(&text[..first]) } else { &text[..first] };
    let content = Some(before.to_string()).filter(|c| !c.trim().is_empty());
    Extracted { called: true, calls, content }
}

/// The longest start of `tag` that ends `text` (its length): text that may be the beginning of the tag (a start that ends between a character's bytes is none: DSML's `｜`).
fn partial_tag_overlap(text: &str, tag: &str) -> usize {
    (1..tag.len()).rev().find(|&k| tag.is_char_boundary(k) && text.ends_with(&tag[..k])).unwrap_or(0)
}

/// What a reply being written adds for the client.
#[derive(Clone, Debug, PartialEq)]
pub enum ToolEv {
    /// Text that is not a call.
    Content(String),
    /// A call's function (its name is complete): the call's index, its id and the name.
    Start { index: usize, id: String, name: String },
    /// More of a call's arguments (hermes: as the model wrote them; XML: the next items of the JSON object they make).
    Args { index: usize, args: String },
}

struct Started {
    name: Option<String>,
    streamed: String,
    /// An XML call: how many of its parameters were read (sent or skipped), and whether its arguments were closed.
    params: usize,
    closed: bool,
}

struct Block {
    json: String,
    complete: bool,
    from: usize,
    to: usize,
}

/// A reply read as it is written, as vLLM's parsers read it in a stream: the whole text so far is read again at each piece and what is new is sent. The text before the first `<tool_call>` is content (what may be the start
/// of the tag is held back). Hermes: a call's function is sent when its name is complete, then its arguments as the text after `"arguments":` grows, up to the closing of the call. XML: a function is sent when its
/// `<function=NAME>` is written, then the `{` that opens its arguments, each parameter as `"KEY": VALUE` (typed by the tool's schema) when its end is written, and the `}` where the function ends. The text of a block that
/// never named a function is content at the end.
pub struct ToolStream {
    text: String,
    sent: usize,
    calls: Vec<Started>,
    format: Format,
    tools: Vec<Value>,
    /// A call was sent (the reply is a tool call's: its finish reason is `tool_calls`).
    pub emitted: bool,
}

impl ToolStream {
    /// A reader of a reply in `format`; `tools` are the request's (an XML call's parameters are typed by their schemas).
    pub fn new(format: Format, tools: &[Value]) -> ToolStream {
        ToolStream { text: String::new(), sent: 0, calls: Vec::new(), format, tools: tools.to_vec(), emitted: false }
    }

    /// `delta` of the reply's text, and what it adds.
    pub fn feed(&mut self, delta: &str) -> Vec<ToolEv> {
        self.text.push_str(delta);
        self.step(false)
    }

    /// The reply's end: what was held back as the start of a tag, and the text of the blocks that were not calls.
    pub fn finish(&mut self) -> Vec<ToolEv> {
        let mut out = self.step(true);
        match self.format {
            Format::Hermes => {
                let named = self.calls.iter().filter(|c| c.name.is_some()).count();
                for b in self.blocks().iter().skip(named) {
                    out.push(ToolEv::Content(self.text[b.from..b.to].to_string()));
                }
            }
            Format::Xml | Format::Dsml | Format::Atem => {
                for b in xml_scan(&self.text, false, self.format.tags()).iter().filter(|b| b.funcs.is_empty()) {
                    out.push(ToolEv::Content(self.text[b.from..b.to].to_string()));
                }
            }
            Format::Json => {}
            Format::Glm => {
                // (a last block not closed: a call where it reads as one, else its text)
                let blocks: Vec<(String, bool, Option<Call>)> =
                    glm_blocks(&self.text).iter().map(|b| (self.text[b.0..b.1].to_string(), b.3, glm_call(b.2, &self.tools))).collect();
                for (i, (text, closed, call)) in blocks.into_iter().enumerate().skip(self.calls.len()) {
                    match (closed, call) {
                        (false, Some(c)) => self.glm_emit(i, c, &mut out),
                        _ => out.push(ToolEv::Content(text)),
                    }
                }
            }
        }
        out
    }

    /// A GLM call whole: its start and all its arguments (a call is read when its block closes).
    fn glm_emit(&mut self, index: usize, c: Call, out: &mut Vec<ToolEv>) {
        self.calls.push(Started { name: Some(c.name.clone()), streamed: c.arguments.clone(), params: 0, closed: true });
        self.emitted = true;
        out.push(ToolEv::Start { index, id: new_id(), name: c.name });
        out.push(ToolEv::Args { index, args: c.arguments });
    }

    /// The GLM calls whose blocks are closed: each sent whole, in order (a block that is not a call stops them: its text goes out at the end).
    fn glm_calls(&mut self, out: &mut Vec<ToolEv>) {
        let blocks: Vec<(bool, Option<Call>)> = glm_blocks(&self.text).iter().map(|b| (b.3, glm_call(b.2, &self.tools))).collect();
        for (i, (closed, call)) in blocks.into_iter().enumerate().skip(self.calls.len()) {
            match (closed, call) {
                (true, Some(c)) => self.glm_emit(i, c, out),
                _ => break,
            }
        }
    }

    fn blocks(&self) -> Vec<Block> {
        let (t, mut out, mut pos) = (&self.text, Vec::new(), 0);
        while let Some(r) = t[pos..].find(OPEN) {
            let (from, js) = (pos + r, pos + r + OPEN.len());
            match t[js..].find(CLOSE) {
                Some(e) => {
                    out.push(Block { json: t[js..js + e].trim().to_string(), complete: true, from, to: js + e + CLOSE.len() });
                    pos = js + e + CLOSE.len();
                }
                None => {
                    let raw = &t[js..];
                    let raw = &raw[..raw.len() - partial_tag_overlap(raw, CLOSE)];
                    let json = raw.trim().to_string();
                    let complete = !json.is_empty() && json::parse(&json).is_ok();
                    out.push(Block { json, complete, from, to: t.len() });
                    break;
                }
            }
        }
        out
    }

    /// A Llama 3.1+ reply as it grows: a reply that does not begin as a call (`<|python_tag|>` or `{`) is content from its first character (a start of the tag is held back); a call is sent whole (its name and its
    /// arguments) when its object is whole; text after the calls that is not one is content at the end.
    fn json_step(&mut self, last: bool) -> Vec<ToolEv> {
        let mut out = Vec::new();
        let (calls, rest) = json_calls(&self.text);
        if calls.is_empty() && rest == Some(0) {
            // (not a call, or not yet: the start of the tag waits)
            let held = !last && PYTHON_TAG.starts_with(self.text.as_str());
            if !held && self.text.len() > self.sent {
                out.push(ToolEv::Content(self.text[self.sent..].to_string()));
                self.sent = self.text.len();
            }
            return out;
        }
        for (i, c) in calls.into_iter().enumerate().skip(self.calls.len()) {
            let name = known_name(&c.name, &self.tools);
            self.calls.push(Started { name: Some(name.clone()), streamed: c.arguments.clone(), params: 0, closed: true });
            self.emitted = true;
            out.push(ToolEv::Start { index: i, id: new_id(), name });
            out.push(ToolEv::Args { index: i, args: c.arguments });
        }
        if let (true, Some(at)) = (last, rest) {
            // (what is left at the end: an object that never closed, or one that is no call)
            let from = at.max(self.sent);
            if from < self.text.len() {
                out.push(ToolEv::Content(self.text[from..].to_string()));
                self.sent = self.text.len();
            }
        }
        out
    }

    fn step(&mut self, last: bool) -> Vec<ToolEv> {
        if self.format == Format::Json {
            return self.json_step(last);
        }
        let mut out = Vec::new();
        let open = self.format.open();
        let found = self.text.find(open);
        let mut sendable = match found {
            Some(i) => i,
            None if last => self.text.len(),
            None => self.text.len() - partial_tag_overlap(&self.text, open),
        };
        // (DSML: the two new lines before a block are not text, so they wait)
        if self.format == Format::Dsml && (found.is_some() || !last) {
            let t = &self.text[..sendable];
            sendable = t.strip_suffix("\n\n").or(t.strip_suffix('\n')).unwrap_or(t).len();
        }
        if sendable > self.sent {
            out.push(ToolEv::Content(self.text[self.sent..sendable].to_string()));
            self.sent = sendable;
        }
        match self.format {
            Format::Hermes => self.hermes_calls(&mut out),
            Format::Xml | Format::Dsml | Format::Atem => self.xml_calls(&mut out),
            Format::Glm => self.glm_calls(&mut out),
            Format::Json => {}
        }
        out
    }

    /// The hermes calls so far: each block's function when its name is complete, then its arguments as they grow.
    fn hermes_calls(&mut self, out: &mut Vec<ToolEv>) {
        for (i, b) in self.blocks().iter().enumerate() {
            if i >= self.calls.len() {
                self.calls.push(Started { name: None, streamed: String::new(), params: 0, closed: false });
            }
            if self.calls[i].name.is_none() {
                // (a call after one that has no name yet waits for it)
                let Some(name) = name_in(&b.json).map(|n| known_name(&n, &self.tools)) else { break };
                self.calls[i].name = Some(name.clone());
                self.emitted = true;
                out.push(ToolEv::Start { index: i, id: new_id(), name });
            }
            if let Some(args) = args_in(&b.json, b.complete) {
                let done = self.calls[i].streamed.len();
                if args.len() > done {
                    out.push(ToolEv::Args { index: i, args: args[done..].to_string() });
                    self.calls[i].streamed = args;
                }
            }
        }
    }

    /// The XML calls so far: each function when its `<function=NAME>` is written, the `{` of its arguments a step later (as vLLM sends the name and the `{` in two chunks) or with whatever else is new, then its
    /// parameters, each when its end is written, and the `}` when the function ends.
    fn xml_calls(&mut self, out: &mut Vec<ToolEv>) {
        let blocks = xml_scan(&self.text, false, self.format.tags());
        for (i, f) in blocks.iter().flat_map(|b| &b.funcs).enumerate() {
            let fresh = i >= self.calls.len();
            if fresh {
                let name = known_name(f.name, &self.tools);
                self.calls.push(Started { name: Some(name.clone()), streamed: String::new(), params: 0, closed: false });
                self.emitted = true;
                out.push(ToolEv::Start { index: i, id: new_id(), name });
            }
            let (s, props) = (&mut self.calls[i], properties(&self.tools, &known_name(f.name, &self.tools)));
            let mut items = String::new();
            for g in f.params.get(s.params..).unwrap_or_default() {
                if let Some((_, it)) = param_item(self.format, g, props) {
                    if s.streamed.len() > 1 || !items.is_empty() {
                        items.push_str(", ");
                    }
                    items.push_str(&it);
                }
            }
            s.params = f.params.len();
            let close = f.closed && !s.closed;
            let mut add = String::new();
            if s.streamed.is_empty() && (!fresh || !items.is_empty() || close) {
                add.push('{');
            }
            add.push_str(&items);
            if close {
                add.push('}');
                s.closed = true;
            }
            if !add.is_empty() {
                s.streamed.push_str(&add);
                out.push(ToolEv::Args { index: i, args: add });
            }
        }
    }
}

/// The function's name in a block's JSON where it is complete: what follows `"name":` as a string with its closing quote.
fn name_in(j: &str) -> Option<String> {
    let mut at = 0;
    while let Some(r) = j[at..].find("\"name\"") {
        let rest = j[at + r + 6..].trim_start();
        if let Some(rest) = rest.strip_prefix(':').map(str::trim_start).and_then(|r| r.strip_prefix('"')) {
            if let Some(end) = rest.find('"').filter(|&e| e > 0) {
                return Some(rest[..end].to_string());
            }
        }
        at += r + 6;
    }
    None
}

/// The arguments in a block's JSON: the text after `"arguments":`, and where the block is complete without the `}` that closes the call.
fn args_in(j: &str, complete: bool) -> Option<String> {
    let mut at = 0;
    while let Some(r) = j[at..].find("\"arguments\"") {
        let rest = j[at + r + 11..].trim_start();
        if let Some(raw) = rest.strip_prefix(':').map(str::trim_start) {
            let mut raw = raw;
            if complete {
                raw = raw.trim_end();
                if let Some(r) = raw.strip_suffix('}') {
                    raw = r.trim_end();
                }
            }
            return Some(raw.to_string());
        }
        at += r + 11;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(t: &str) -> Value {
        json::parse(t).unwrap()
    }

    fn hermes(text: &str) -> Extracted {
        extract(text, Format::Hermes, &[])
    }

    const WEATHER: &str = r#"{"type":"function","function":{"name":"get_weather","description":"Get the weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}"#;

    /// `tools` and `tool_choice` as vLLM reads them: the template's tools are its model dump (a missing description or parameters is null), the choice defaults to auto where there are tools and to none where there are
    /// not, and what is wrong is said in its words.
    #[test]
    fn tools_and_the_choice_among_them() {
        let t = tools_of(&v(&format!(r#"{{"tools":[{WEATHER},{{"function":{{"name":"bare"}}}}]}}"#))).unwrap().unwrap();
        assert_eq!((t.choice.clone(), t.parallel), (ToolChoice::Auto, true));
        assert_eq!(
            py_dumps(&t.list[0]),
            r#"{"type": "function", "function": {"name": "get_weather", "description": "Get the weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}"#
        );
        assert_eq!(py_dumps(&t.list[1]), r#"{"type": "function", "function": {"name": "bare", "description": null, "parameters": null}}"#);
        let with = |extra: &str| tools_of(&v(&format!(r#"{{"tools":[{WEATHER}],{extra}}}"#)));
        assert_eq!(with(r#""tool_choice":"none""#).unwrap().unwrap().choice, ToolChoice::None);
        assert_eq!(with(r#""tool_choice":"required""#).unwrap().unwrap().choice, ToolChoice::Required);
        assert_eq!(with(r#""tool_choice":null"#).unwrap().unwrap().choice, ToolChoice::Auto);
        assert_eq!(
            with(r#""tool_choice":{"type":"function","function":{"name":"get_weather"}}"#).unwrap().unwrap().choice,
            ToolChoice::Named("get_weather".into())
        );
        assert!(!with(r#""parallel_tool_calls":false"#).unwrap().unwrap().parallel);
        assert!(!with(r#""tool_choice":{"type":"function","function":{"name":"get_weather"}}"#).unwrap().unwrap().parallel);
        // none: no tools, no choice, or a choice of none
        assert!(
            tools_of(&v("{}")).unwrap().is_none()
                && tools_of(&v(r#"{"tools":null}"#)).unwrap().is_none()
                && tools_of(&v(r#"{"tool_choice":"none"}"#)).unwrap().is_none()
        );
        // refused, in vLLM's words
        let bad = |t: &str| tools_of(&v(t)).unwrap_err();
        assert!(bad(r#"{"tools":[]}"#).contains("must not be an empty array"));
        assert!(bad(r#"{"tool_choice":"auto"}"#).contains("When using `tool_choice`, `tools` must be set."));
        assert!(bad(&format!(r#"{{"tools":[{WEATHER}],"tool_choice":"sometimes"}}"#)).contains("Invalid value for `tool_choice`: sometimes!"));
        assert!(bad(&format!(r#"{{"tools":[{WEATHER}],"tool_choice":{{"type":"function","function":{{"name":"nope"}}}}}}"#))
            .contains("does not match any of the specified `tools`"));
        assert!(bad(&format!(r#"{{"tools":[{WEATHER}],"tool_choice":{{"type":"function"}}}}"#)).contains("Invalid value for `function`"));
        assert!(bad(&format!(r#"{{"tools":[{WEATHER}],"tool_choice":{{"function":{{}}}}}}"#)).contains("Expected field `name`"));
        assert!(bad(r#"{"tools":[{"type":"retrieval"}]}"#).contains("type"));
        assert!(bad(r#"{"tools":[{"type":"function","function":{}}]}"#).contains("`name`"));
        assert!(bad(r#"{"tools":"x"}"#).contains("list"));
    }

    /// The messages as the template reads them: text parts joined by new lines, null content empty, a tool call's arguments an object (an unreadable string an empty one), the thinking under both names, a tool
    /// message's id, and what the model cannot read refused.
    #[test]
    fn messages_as_the_template_reads_them() {
        let m = template_messages(&v(r#"{"messages":[
            {"role":"system","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]},
            {"role":"user","content":"hi"},
            {"role":"assistant","content":null,"reasoning_content":"plan","tool_calls":[
                {"id":"c1","type":"function","function":{"name":"get_weather","arguments":"{\"city\": \"Paris\", \"days\": 2}"}},
                {"id":"c2","function":{"name":"bad","arguments":"not json"}},
                {"id":"c3","type":"function","function":{"name":"list","arguments":"[1]"}},
                {"id":"c4","type":"function","function":{"name":"obj","arguments":{"x":1}}},
                {"id":"c5","type":"function","function":{"name":"none"}}]},
            {"role":"assistant","content":"x","tool_calls":[]},
            {"role":"tool","tool_call_id":"c1","name":"get_weather","content":[{"type":"text","text":"sunny"},{"type":"text","text":"20C"}]}]}"#))
        .unwrap();
        assert_eq!(py_dumps(&m[0]), r#"{"role": "system", "content": "a\nb"}"#);
        assert_eq!(
            py_dumps(&m[2]),
            r#"{"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris", "days": 2}}}, {"id": "c2", "function": {"name": "bad", "arguments": {}}}, {"id": "c3", "type": "function", "function": {"name": "list", "arguments": {}}}, {"id": "c4", "type": "function", "function": {"name": "obj", "arguments": {"x": 1}}}, {"id": "c5", "type": "function", "function": {"name": "none", "arguments": {}}}], "reasoning": "plan", "reasoning_content": "plan"}"#
        );
        assert_eq!(py_dumps(&m[3]), r#"{"role": "assistant", "content": "x"}"#, "an empty list of calls is none");
        assert_eq!(py_dumps(&m[4]), r#"{"role": "tool", "content": "sunny\n20C", "tool_call_id": "c1", "name": "get_weather"}"#);
        let bad = |t: &str| template_messages(&v(t)).unwrap_err();
        assert!(bad(r#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]}"#).contains("image_url"));
        assert!(bad(r#"{"messages":[{"role":"assistant","tool_calls":[{"type":"code","function":{"name":"f"}}]}]}"#).contains("type 'function'"));
        assert!(bad(r#"{"messages":[{"role":"assistant","tool_calls":["x"]}]}"#).contains("must be objects"));
        assert!(
            bad(r#"{"messages":[{"content":"x"}]}"#).contains("no role")
                && bad(r#"{"messages":[]}"#).contains("no messages")
                && bad(r#"{"messages":[3]}"#).contains("not an object")
                && bad(r#"{"messages":[{"role":"user","content":5}]}"#).contains("not text")
        );
        // where the model reads images: an image part is its place in the text (the parts joined as the template joins them), its URL kept in order
        let mut urls = Vec::new();
        let m = template_messages_with(&v(r#"{"messages":[{"role":"user","content":[{"type":"text","text":"Look: "},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA"}},{"type":"image_url","image_url":"http://x/b.jpg"},{"type":"text","text":"which?"}]}]}"#), Some(&mut urls)).unwrap();
        assert_eq!(m[0].get("content").and_then(Value::as_str), Some(format!("Look: {IMAGE_PLACE}{IMAGE_PLACE}which?").as_str()));
        assert_eq!(urls, ["data:image/png;base64,AA", "http://x/b.jpg"]);
        assert!(template_messages_with(&v(r#"{"messages":[{"role":"system","content":[{"type":"image_url","image_url":{"url":"x"}}]}]}"#), Some(&mut urls))
            .unwrap_err()
            .contains("System message"));
    }

    /// The prompt's opening of a forced call, which the reply goes on from.
    #[test]
    fn a_forced_call_begins_in_the_prompt() {
        assert_eq!(forced_prefix(&ToolChoice::Auto, Format::Hermes), None);
        assert_eq!(forced_prefix(&ToolChoice::Required, Format::Hermes).unwrap(), "<tool_call>\n{\"name\": \"");
        assert_eq!(
            forced_prefix(&ToolChoice::Named("get_weather".into()), Format::Hermes).unwrap(),
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": "
        );
        // the prefix and what the model writes after it are one reply for the parser
        let e = hermes(&(forced_prefix(&ToolChoice::Named("f".into()), Format::Hermes).unwrap() + "{\"a\": 1}}\n</tool_call>"));
        assert_eq!(e.calls, vec![Call { name: "f".into(), arguments: "{\"a\": 1}".into() }]);
        let e = hermes(&(forced_prefix(&ToolChoice::Required, Format::Hermes).unwrap() + "g\", \"arguments\": {}}\n</tool_call>"));
        assert_eq!(e.calls, vec![Call { name: "g".into(), arguments: "{}".into() }]);
    }

    /// vLLM's hermes parser on a whole reply: the calls with their arguments written as Python writes them, the text before them as content (None where it is white space), and a reply with a block that is not a call is
    /// all text.
    #[test]
    fn calls_read_from_a_whole_reply() {
        let one = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\":\"Paris\",\"n\":2.50,\"é\":[1,null,true]}}\n</tool_call>";
        let e = hermes(one);
        assert_eq!(
            e,
            Extracted {
                called: true,
                calls: vec![Call { name: "get_weather".into(), arguments: r#"{"city": "Paris", "n": 2.5, "é": [1, null, true]}"#.into() }],
                content: None
            }
        );
        // text before the first call is the content; white space is none; two calls, in order; the text after the last is dropped, as vLLM drops it
        let e = hermes(&format!("Let me check.\n{one}\n<tool_call>\n{{\"name\": \"b\", \"arguments\": {{}}}}\n</tool_call>\nDone."));
        assert_eq!((e.content.as_deref(), e.calls.iter().map(|c| c.name.as_str()).collect::<Vec<_>>()), (Some("Let me check.\n"), vec!["get_weather", "b"]));
        assert_eq!(hermes(&format!("\n\n{one}")).content, None);
        // a block with no closing tag is read to the end of the text, where it is JSON
        let e = hermes("<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1}}");
        assert_eq!(e.calls, vec![Call { name: "f".into(), arguments: "{\"a\": 1}".into() }]);
        // arguments that are a string stay a string (written again as JSON text)
        assert_eq!(hermes("<tool_call>{\"name\": \"f\", \"arguments\": \"{\\\"a\\\": 1}\"}</tool_call>").calls[0].arguments, r#""{\"a\": 1}""#);
        // no tag: text
        assert_eq!(hermes("Hello <tool_call"), Extracted { called: false, calls: vec![], content: Some("Hello <tool_call".into()) });
        // malformed, in each way a model can: all text, no call, no failure
        for bad in [
            "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1</tool_call>",
            "<tool_call>\n{\"name\": \"f\"}\n</tool_call>",
            "<tool_call>\n{\"arguments\": {}}\n</tool_call>",
            "<tool_call>\n{\"name\": 3, \"arguments\": {}}\n</tool_call>",
            "<tool_call>\n</tool_call>",
            "<tool_call>\n[1, 2]\n</tool_call>",
            "<tool_call>\nnot json\n</tool_call>",
            "<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>\n<tool_call>\noops\n</tool_call>",
            "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1",
            "<tool_call>",
        ] {
            assert_eq!(hermes(bad), Extracted { called: false, calls: vec![], content: Some(bad.to_string()) }, "{bad:?}");
        }
    }

    /// A tool with a parameter of each type vLLM's `qwen3_coder` parser converts, for the tests of the XML format.
    const FORECAST: &str = r#"{"type":"function","function":{"name":"forecast","description":"Forecast","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"},"ratio":{"type":"number"},"metric":{"type":"boolean"},"tags":{"type":"array","items":{"type":"string"}},"opts":{"type":"object"},"either":{"anyOf":[{"type":"string"},{"type":"null"}]}},"required":["city"]}}}"#;

    fn xml(text: &str) -> Extracted {
        extract(text, Format::Xml, &[v(FORECAST)])
    }

    fn call(name: &str, arguments: &str) -> Call {
        Call { name: name.into(), arguments: arguments.into() }
    }

    fn text_only(text: &str) -> Extracted {
        Extracted { called: false, calls: vec![], content: Some(text.to_string()) }
    }

    /// A reply is read in the format of the chat template: XML where the template writes `<function=` (the tools parts of Qwen/Qwen3.5-0.8B's template at revision 2fc06364 and of Nemotron 3 Super's, in tests/fixtures), hermes
    /// for Qwen3's, which writes `<function-name>` and a JSON object.
    #[test]
    fn the_format_is_the_templates() {
        let of = |src: &str| Format::of(&Chat::from_template(src).unwrap());
        assert_eq!(of(include_str!("../tests/fixtures/chat_qwen35_tools.jinja")), Format::Xml);
        assert_eq!(of(include_str!("../tests/fixtures/chat_nemotron3_tools.jinja")), Format::Xml);
        let qwen3 = v(include_str!("../tests/fixtures/chat_qwen3.json"));
        assert_eq!(of(qwen3.get("template").unwrap().as_str().unwrap()), Format::Hermes);
    }

    /// The prompt's opening of a forced XML call: the function's name and the line after it for a named one, the reply then the parameters; for a required one the model names the function.
    #[test]
    fn a_forced_xml_call_begins_in_the_prompt() {
        assert_eq!(forced_prefix(&ToolChoice::Auto, Format::Xml), None);
        assert_eq!(forced_prefix(&ToolChoice::None, Format::Xml), None);
        let named = forced_prefix(&ToolChoice::Named("forecast".into()), Format::Xml).unwrap();
        assert_eq!(named, "<tool_call>\n<function=forecast>\n");
        let required = forced_prefix(&ToolChoice::Required, Format::Xml).unwrap();
        assert_eq!(required, "<tool_call>\n<function=");
        // the prefix and what the model writes after it are one reply for the parser
        let rest = "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(xml(&(named + rest)).calls, vec![call("forecast", r#"{"city": "Paris"}"#)]);
        assert_eq!(xml(&(required + "forecast>\n" + rest)).calls, vec![call("forecast", r#"{"city": "Paris"}"#)]);
    }

    /// vLLM's `qwen3_coder` parser on a whole reply: each function a call, its parameters the arguments written as Python writes them, the text before the first `<tool_call>` as content (None where it is white space), the
    /// text after the last block dropped, a value's one new line at each end taken off and no other white space.
    #[test]
    fn xml_calls_read_from_a_whole_reply() {
        let one = "<tool_call>\n<function=forecast>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(xml(one), Extracted { called: true, calls: vec![call("forecast", r#"{"city": "Paris", "days": 3}"#)], content: None });
        let rome = "<tool_call>\n<function=forecast>\n<parameter=city>\nRome\n</parameter>\n</function>\n</tool_call>";
        let e = xml(&format!("Let me check.\n\n{one}\n{rome}\nDone."));
        assert_eq!(
            (e.content.as_deref(), e.calls),
            (Some("Let me check.\n\n"), vec![call("forecast", r#"{"city": "Paris", "days": 3}"#), call("forecast", r#"{"city": "Rome"}"#)])
        );
        assert_eq!(xml(&format!("\n\n{one}")).content, None);
        // two functions in one block are two calls
        let both = "<tool_call>\n<function=forecast>\n<parameter=city>\nParis\n</parameter>\n</function>\n<function=forecast>\n<parameter=city>\nRome\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(xml(both).calls, vec![call("forecast", r#"{"city": "Paris"}"#), call("forecast", r#"{"city": "Rome"}"#)]);
        // a value that spans lines keeps them, its inner blank lines and spaces, and all but one new line at each end; non-ASCII text is written as it is
        let lines = "<tool_call>\n<function=forecast>\n<parameter=city>\nThis is the value\nthat can span\n\n  multiple lines é\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(xml(lines).calls, vec![call("forecast", r#"{"city": "This is the value\nthat can span\n\n  multiple lines é"}"#)]);
        let ends = "<tool_call>\n<function=forecast>\n<parameter=city>\nx\n\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(xml(ends).calls, vec![call("forecast", r#"{"city": "x\n"}"#)]);
        // a function with no parameters; a key that comes again keeps its place and takes the last value
        assert_eq!(xml("<tool_call>\n<function=forecast>\n</function>\n</tool_call>").calls, vec![call("forecast", "{}")]);
        let again = "<tool_call>\n<function=forecast>\n<parameter=city>\nA\n</parameter>\n<parameter=days>\n1\n</parameter>\n<parameter=city>\nB\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(xml(again).calls, vec![call("forecast", r#"{"city": "B", "days": 1}"#)]);
        // no call: an ordinary answer is content
        assert_eq!(xml("The weather is fine."), text_only("The weather is fine."));
    }

    /// Each parameter is read by the type its tool's schema gives it, as vLLM's parser reads it: a string stays text; an integer or a number is read where it is one (a number with no fraction is an integer, anything else stays
    /// text); a boolean is true for `true` in any case and false otherwise; an object, an array and `anyOf` are read as JSON where they are JSON; `null` is null whatever the type; a parameter the schema lacks, and a function it
    /// lacks, leave the text as it is.
    #[test]
    fn xml_values_are_typed_by_the_schema() {
        let typed = "<tool_call>\n<function=forecast>\n<parameter=city>\n12\n</parameter>\n<parameter=days>\nthree\n</parameter>\n<parameter=ratio>\n2.50\n</parameter>\n<parameter=metric>\nTrue\n</parameter>\n<parameter=tags>\n[\"a\", \"é\"]\n</parameter>\n<parameter=opts>\n{\"k\":[1,2.50,null,true]}\n</parameter>\n<parameter=either>\n7\n</parameter>\n<parameter=extra>\n5\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(
            xml(typed).calls,
            vec![call(
                "forecast",
                r#"{"city": "12", "days": "three", "ratio": 2.5, "metric": true, "tags": ["a", "é"], "opts": {"k": [1, 2.5, null, true]}, "either": 7, "extra": "5"}"#
            )]
        );
        let more = "<tool_call>\n<function=forecast>\n<parameter=days>\n 7 \n</parameter>\n<parameter=ratio>\n3.0\n</parameter>\n<parameter=metric>\nyes\n</parameter>\n<parameter=tags>\nnot json\n</parameter>\n<parameter=either>\nhi\n</parameter>\n<parameter=city>\nNULL\n</parameter>\n<parameter=opts>\nnull\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(
            xml(more).calls,
            vec![call("forecast", r#"{"days": 7, "ratio": 3, "metric": false, "tags": "not json", "either": "hi", "city": null, "opts": null}"#)]
        );
        let bad = "<tool_call>\n<function=forecast>\n<parameter=days>\n3.0\n</parameter>\n<parameter=ratio>\nabc\n</parameter>\n<parameter=metric>\nfalse\n</parameter>\n<parameter=opts>\n{\"a\": 1\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(xml(bad).calls, vec![call("forecast", r#"{"days": "3.0", "ratio": "abc", "metric": false, "opts": "{\"a\": 1"}"#)]);
        let unknown = "<tool_call>\n<function=mystery>\n<parameter=n>\n5\n</parameter>\n<parameter=z>\nnull\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(xml(unknown).calls, vec![call("mystery", r#"{"n": "5", "z": null}"#)]);
        // a request with no tools reads every value as text
        assert_eq!(
            extract("<tool_call>\n<function=forecast>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>", Format::Xml, &[]).calls,
            vec![call("forecast", r#"{"days": "3"}"#)]
        );
    }

    /// A reply that is not well formed, as vLLM's parser reads it: a function with no `</function>` and a parameter with no `</parameter>` end where the next thing begins or the block does, a reply cut off in a value is read
    /// to its end, and a function written without a `<tool_call>` is a call; a block that holds no function, a function with no `>` and a parameter with no `>` leave the whole reply as text (vLLM drops the text of the
    /// first), and a function after one that is not is a call.
    #[test]
    fn an_xml_reply_that_is_malformed() {
        let paris = vec![call("forecast", r#"{"city": "Paris"}"#)];
        assert_eq!(xml("<tool_call>\n<function=forecast>\n<parameter=city>\nParis\n</parameter>\n</tool_call>").calls, paris);
        assert_eq!(xml("<tool_call>\n<function=forecast>\n<parameter=city>\nParis\n</function>\n</tool_call>").calls, paris);
        assert_eq!(
            xml("<tool_call>\n<function=forecast>\n<parameter=city>\nParis\n<parameter=days>\n2\n</function>\n</tool_call>").calls,
            vec![call("forecast", r#"{"city": "Paris", "days": 2}"#)]
        );
        assert_eq!(xml("<tool_call>\n<function=forecast>\n<parameter=city>\nPar").calls, vec![call("forecast", r#"{"city": "Par"}"#)]);
        assert_eq!(xml("<tool_call>\n<function=forecast>\n<parameter=city>\nParis\n").calls, paris, "the new line that ends the text is not the value's");
        let bare = xml("Sure.\n<function=forecast>\n<parameter=city>\nParis\n</parameter>\n</function>");
        assert_eq!((bare.calls, bare.content.as_deref()), (paris, Some("Sure.\n")));
        let second = xml(
            "<tool_call>\n<function=forecast\n</tool_call>\n<tool_call>\n<function=forecast>\n<parameter=city>\nRome\n</parameter>\n</function>\n</tool_call>",
        );
        assert_eq!(second.calls, vec![call("forecast", r#"{"city": "Rome"}"#)]);
        for bad in [
            "<tool_call>\n</tool_call>",
            "<tool_call>\n</tool_call>\n<function=forecast>\n<parameter=city>\nParis\n</parameter>\n</function>",
            "<tool_call>\n{\"name\": \"forecast\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>",
            "<tool_call>\n<function=forecast\n</tool_call>",
            "<tool_call>\n<function=forecast>\n<parameter=city</parameter>\n</function>\n</tool_call>",
            "<tool_call>\n<function=",
            "<function=forecast",
            "<tool_call>",
        ] {
            assert_eq!(xml(bad), text_only(bad), "{bad:?}");
        }
    }

    /// What a stream says for a reply, in the order it says it: (the kind, what it carries) for each event.
    fn run(pieces: &[&str]) -> (Vec<String>, bool) {
        run_in(Format::Hermes, &[], pieces)
    }

    fn run_in(format: Format, tools: &[Value], pieces: &[&str]) -> (Vec<String>, bool) {
        let mut s = ToolStream::new(format, tools);
        let mut evs = Vec::new();
        for p in pieces {
            evs.extend(s.feed(p));
        }
        evs.extend(s.finish());
        let mut ids = Vec::new();
        let described = evs
            .iter()
            .map(|e| match e {
                ToolEv::Content(c) => format!("content {c:?}"),
                ToolEv::Start { index, id, name } => {
                    ids.push(id.clone());
                    assert!(id.starts_with("chatcmpl-tool-") && id.len() == "chatcmpl-tool-".len() + 16, "{id}");
                    format!("start {index} {name}")
                }
                ToolEv::Args { index, args } => format!("args {index} {args:?}"),
            })
            .collect();
        assert_eq!(ids.iter().collect::<std::collections::HashSet<_>>().len(), ids.len(), "the ids of the calls are different");
        (described, s.emitted)
    }

    fn chars(t: &str) -> Vec<String> {
        t.chars().map(|c| c.to_string()).collect()
    }

    /// A reply written a piece at a time: the text before the call as content (what may begin the tag held back), the function when its name is complete, its arguments as they are written, the same for a call of another
    /// index; and whatever the pieces are the calls are the same.
    #[test]
    fn calls_read_as_they_are_written() {
        let text = "Sure.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
        let whole = run(&[text]);
        assert_eq!(
            whole,
            (vec!["content \"Sure.\\n\"".to_string(), "start 0 get_weather".to_string(), "args 0 \"{\\\"city\\\": \\\"Paris\\\"}\"".to_string()], true)
        );
        // a character at a time: the pieces differ, what they add up to does not
        let pieces: Vec<String> = chars(text);
        let refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
        let (evs, emitted) = run(&refs);
        assert!(emitted);
        let (mut content, mut args, mut names) = (String::new(), String::new(), Vec::new());
        for e in &evs {
            if let Some(c) = e.strip_prefix("content ") {
                content.push_str(json::parse(c).unwrap().as_str().unwrap());
            } else if let Some(a) = e.strip_prefix("args 0 ") {
                args.push_str(json::parse(a).unwrap().as_str().unwrap());
            } else {
                names.push(e.clone());
            }
        }
        assert_eq!((content.as_str(), args.as_str(), names), ("Sure.\n", "{\"city\": \"Paris\"}", vec!["start 0 get_weather".to_string()]));
        // a call straight after the generation prompt, and two calls
        let two = "<tool_call>\n{\"name\": \"a\", \"arguments\": {\"x\": 1}}\n</tool_call>\n<tool_call>\n{\"name\": \"b\", \"arguments\": {}}\n</tool_call>";
        assert_eq!(run(&[two]).0, vec!["start 0 a", "args 0 \"{\\\"x\\\": 1}\"", "start 1 b", "args 1 \"{}\""]);
        // the tag split between pieces is held, not sent as content
        assert_eq!(run(&["Hi <tool", "_call>{\"name\": \"a\", \"arguments\": {}}</tool_call>"]).0, vec!["content \"Hi \"", "start 0 a", "args 0 \"{}\""]);
        // a piece that ends in what is not the tag after all is content
        assert_eq!(run(&["a <tool", " b"]).0, vec!["content \"a \"", "content \"<tool b\""]);
        assert_eq!(run(&["a <tool"]).0, vec!["content \"a \"", "content \"<tool\""], "the held text is sent where the reply ends");
        // plain text with tools offered: content, no call (an ordinary answer)
        assert_eq!(run(&["The weather ", "is fine."]), (vec!["content \"The weather \"".to_string(), "content \"is fine.\"".to_string()], false));
    }

    /// A block that never names a function is text at the end; one that did stays what was sent; nothing is lost and nothing fails.
    #[test]
    fn a_block_that_is_not_a_call_is_text() {
        let (evs, emitted) = run(&["Ok <tool_call>\nnot json at all\n</tool_call>"]);
        assert_eq!((evs, emitted), (vec!["content \"Ok \"".to_string(), "content \"<tool_call>\\nnot json at all\\n</tool_call>\"".to_string()], false));
        // cut off in the middle of the name
        let (evs, emitted) = run(&["<tool_call>\n{\"name\": \"get_wea"]);
        assert_eq!((evs, emitted), (vec!["content \"<tool_call>\\n{\\\"name\\\": \\\"get_wea\"".to_string()], false));
        // cut off in the arguments: the name and the arguments so far were sent
        let (evs, emitted) = run(&["<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1"]);
        assert_eq!((evs, emitted), (vec!["start 0 f".to_string(), "args 0 \"{\\\"a\\\": 1\"".to_string()], true));
        // a bad block after a good one: the good one was sent, the bad one is text
        let (evs, _) = run(&["<tool_call>{\"name\": \"f\", \"arguments\": {}}</tool_call><tool_call>garbage</tool_call>"]);
        assert_eq!(evs, vec!["start 0 f", "args 0 \"{}\"", "content \"<tool_call>garbage</tool_call>\""]);
    }

    /// The arguments a stream sends add up to the whole parse's, for well-formed calls, whatever the pieces.
    #[test]
    fn the_stream_and_the_whole_parse_agree() {
        let reply = "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": [1, 2, {\"b\": \"c\"}], \"d\": \"é\"}}\n</tool_call>";
        let want = hermes(reply).calls;
        for step in [1usize, 2, 3, 7, 50] {
            let cs: Vec<char> = reply.chars().collect();
            let mut s = ToolStream::new(Format::Hermes, &[]);
            let (mut name, mut args) = (String::new(), String::new());
            let mut feed = |evs: Vec<ToolEv>| {
                for e in evs {
                    match e {
                        ToolEv::Start { name: n, .. } => name = n,
                        ToolEv::Args { args: a, .. } => args.push_str(&a),
                        ToolEv::Content(c) => panic!("content {c:?}"),
                    }
                }
            };
            for chunk in cs.chunks(step) {
                feed(s.feed(&chunk.iter().collect::<String>()));
            }
            feed(s.finish());
            assert_eq!(vec![Call { name, arguments: args }], want, "pieces of {step}");
        }
    }

    /// What a stream of `pieces` adds up to: the content, and the calls (the name and the arguments the pieces add up to) by index.
    fn sum(format: Format, tools: &[Value], pieces: &[&str]) -> (String, Vec<Call>) {
        let mut s = ToolStream::new(format, tools);
        let mut evs = Vec::new();
        for p in pieces {
            evs.extend(s.feed(p));
        }
        evs.extend(s.finish());
        let (mut content, mut calls) = (String::new(), Vec::<Call>::new());
        for e in evs {
            match e {
                ToolEv::Content(c) => content.push_str(&c),
                ToolEv::Start { index, name, .. } => {
                    assert_eq!(index, calls.len(), "calls begin in order");
                    calls.push(call(&name, ""));
                }
                ToolEv::Args { index, args } => calls[index].arguments.push_str(&args),
            }
        }
        (content, calls)
    }

    /// An XML reply written a piece at a time, as vLLM's `qwen3_coder` parser streams it: the text before the first call as content (what may begin the tag held back), a function when its `<function=NAME>` is written, the
    /// `{` of its arguments with it, each parameter as `"KEY": VALUE` when its end is written, the `}` when the function ends; and whatever the pieces are the calls are the same.
    #[test]
    fn xml_calls_read_as_they_are_written() {
        let tools = [v(FORECAST)];
        let reply =
            "Sure.\n<tool_call>\n<function=forecast>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
        let whole = run_in(Format::Xml, &tools, &[reply]);
        assert_eq!(
            whole,
            (
                vec![
                    "content \"Sure.\\n\"".to_string(),
                    "start 0 forecast".to_string(),
                    "args 0 \"{\\\"city\\\": \\\"Paris\\\", \\\"days\\\": 3}\"".to_string()
                ],
                true
            )
        );
        // a character at a time: the pieces of arguments are the `{`, the parameters as their ends come, and the `}`
        let pieces = chars(reply);
        let refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
        let (evs, emitted) = run_in(Format::Xml, &tools, &refs);
        assert!(emitted);
        let calls: Vec<&str> = evs.iter().filter(|e| !e.starts_with("content")).map(String::as_str).collect();
        assert_eq!(calls, vec!["start 0 forecast", "args 0 \"{\"", "args 0 \"\\\"city\\\": \\\"Paris\\\"\"", "args 0 \", \\\"days\\\": 3\"", "args 0 \"}\""]);
        assert_eq!(sum(Format::Xml, &tools, &refs), ("Sure.\n".to_string(), vec![call("forecast", r#"{"city": "Paris", "days": 3}"#)]));
        // a call straight after the generation prompt, and two calls (two blocks, or two functions in one)
        let two = "<tool_call>\n<function=forecast>\n<parameter=city>\nA\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=bare>\n</function>\n</tool_call>";
        assert_eq!(run_in(Format::Xml, &tools, &[two]).0, vec!["start 0 forecast", "args 0 \"{\\\"city\\\": \\\"A\\\"}\"", "start 1 bare", "args 1 \"{}\""]);
        let one_block = "<tool_call>\n<function=forecast>\n</function>\n<function=bare>\n</function>\n</tool_call>";
        assert_eq!(run_in(Format::Xml, &tools, &[one_block]).0, vec!["start 0 forecast", "args 0 \"{}\"", "start 1 bare", "args 1 \"{}\""]);
        // the function is sent alone, the `{` with the next piece
        assert_eq!(
            run_in(Format::Xml, &tools, &["<tool_call>\n<function=bare>\n", "\n", "</function>"]).0,
            vec!["start 0 bare", "args 0 \"{\"", "args 0 \"}\""]
        );
        assert_eq!(
            run_in(Format::Xml, &tools, &["<tool_call>\n<function=bare>\n"]).0,
            vec!["start 0 bare", "args 0 \"{\""],
            "the reply's end is the next piece"
        );
        // the tag split between pieces is held, not sent as content; so is a function's tag, which is not content either
        assert_eq!(
            run_in(Format::Xml, &tools, &["Hi <tool", "_call>\n<func", "tion=bare>\n</function>"]).0,
            vec!["content \"Hi \"", "start 0 bare", "args 0 \"{}\""]
        );
        assert_eq!(run_in(Format::Xml, &tools, &["a <tool", " b"]).0, vec!["content \"a \"", "content \"<tool b\""]);
        // plain text with tools offered: content, no call (an ordinary answer); a function with no block is text, as the stream of vLLM's parser has it
        assert_eq!(
            run_in(Format::Xml, &tools, &["The weather ", "is fine."]),
            (vec!["content \"The weather \"".to_string(), "content \"is fine.\"".to_string()], false)
        );
        assert_eq!(run_in(Format::Xml, &tools, &["<function=bare>\n</function>"]), (vec!["content \"<function=bare>\\n</function>\"".to_string()], false));
        // the arguments wait for a parameter's end: its value is typed by then
        assert_eq!(
            run_in(Format::Xml, &tools, &["<tool_call>\n<function=forecast>\n<parameter=days>\n12", "\n</para", "meter>\n"]).0,
            vec!["start 0 forecast", "args 0 \"{\"", "args 0 \"\\\"days\\\": 12\""]
        );
    }

    /// A block that holds no function is text at the end; a call that was begun stays what was sent; a function with no `</function>` ends with its block; nothing is lost and nothing fails.
    #[test]
    fn an_xml_block_that_is_not_a_call_is_text() {
        let tools = [v(FORECAST)];
        let json_call = "<tool_call>\n{\"name\": \"forecast\", \"arguments\": {}}\n</tool_call>";
        assert_eq!(run_in(Format::Xml, &tools, &[&format!("Ok {json_call}")]), (vec!["content \"Ok \"".to_string(), format!("content {json_call:?}")], false));
        // cut off in the middle of the name
        assert_eq!(run_in(Format::Xml, &tools, &["<tool_call>\n<function=fore"]), (vec!["content \"<tool_call>\\n<function=fore\"".to_string()], false));
        // cut off in a parameter: the function and the parameters whose end was written were sent, not the rest and not the `}`
        let (evs, emitted) = run_in(Format::Xml, &tools, &["<tool_call>\n<function=forecast>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3"]);
        assert_eq!((evs, emitted), (vec!["start 0 forecast".to_string(), "args 0 \"{\\\"city\\\": \\\"Paris\\\"\"".to_string()], true));
        // a function with no `</function>` ends with its block, and a parameter with no `</parameter>` where the next begins or the function ends
        let (_, calls) = sum(Format::Xml, &tools, &["<tool_call>\n<function=forecast>\n<parameter=city>\nParis\n<parameter=days>\n2\n</tool_call>"]);
        assert_eq!(calls, vec![call("forecast", r#"{"city": "Paris", "days": 2}"#)]);
        // a bad block after a good one: the good one was sent, the bad one is text; before it: the same, and the call is the first
        let good = "<tool_call>\n<function=bare>\n</function>\n</tool_call>";
        assert_eq!(
            run_in(Format::Xml, &tools, &[&format!("{good}<tool_call>garbage</tool_call>")]).0,
            vec!["start 0 bare", "args 0 \"{}\"", "content \"<tool_call>garbage</tool_call>\""]
        );
        assert_eq!(
            run_in(Format::Xml, &tools, &[&format!("<tool_call>garbage</tool_call>{good}")]).0,
            vec!["start 0 bare", "args 0 \"{}\"", "content \"<tool_call>garbage</tool_call>\""]
        );
        // a parameter with no `>` is left out of the stream
        let (_, calls) = sum(
            Format::Xml,
            &tools,
            &["<tool_call>\n<function=forecast>\n<parameter=city</parameter>\n<parameter=days>\n4\n</parameter>\n</function>\n</tool_call>"],
        );
        assert_eq!(calls, vec![call("forecast", r#"{"days": 4}"#)]);
    }

    /// The calls a stream adds up to are those of the whole parse, whatever the pieces: a character, a few, uneven ones, the whole reply.
    #[test]
    fn the_xml_stream_and_the_whole_parse_agree() {
        let tools = [v(FORECAST)];
        let replies = [
            "Sure.\n<tool_call>\n<function=forecast>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>",
            "Let me look.\n\n<tool_call>\n<function=forecast>\n<parameter=city>\nThis is the value\nthat can span\n\n  multiple lines é\n</parameter>\n<parameter=ratio>\n2.50\n</parameter>\n<parameter=metric>\nTrue\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=forecast>\n<parameter=tags>\n[\"a\", \"é\"]\n</parameter>\n<parameter=opts>\n{\"k\":[1,2.50,null,true]}\n</parameter>\n<parameter=either>\n7\n</parameter>\n<parameter=extra>\nnull\n</parameter>\n</function>\n</tool_call>",
            "<tool_call>\n<function=forecast>\n<parameter=city>\nA\n</parameter>\n</function>\n<function=mystery>\n<parameter=n>\n5\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=bare>\n</function>\n</tool_call>",
            "<tool_call>\n<function=forecast>\n<parameter=city>\nx\n\n</parameter>\n<parameter=days>\n 7 \n</parameter>\n</function>\n</tool_call>",
        ];
        for reply in replies {
            let want = extract(reply, Format::Xml, &tools);
            let want_content = reply[..reply.find(OPEN).unwrap()].to_string();
            assert_eq!(want.content.clone().unwrap_or_default().trim(), want_content.trim());
            let cs: Vec<char> = reply.chars().collect();
            let uneven = [1usize, 4, 2, 9, 3, 16, 1, 5];
            let mut patterns: Vec<Vec<usize>> = [1usize, 2, 3, 7, 50, 1000].iter().map(|&n| vec![n]).collect();
            patterns.push(uneven.to_vec());
            patterns.push(uneven.iter().rev().cloned().collect());
            for pattern in patterns {
                let (mut pieces, mut at, mut k) = (Vec::new(), 0, 0);
                while at < cs.len() {
                    let n = pattern[k % pattern.len()].min(cs.len() - at);
                    pieces.push(cs[at..at + n].iter().collect::<String>());
                    at += n;
                    k += 1;
                }
                let refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
                let (content, calls) = sum(Format::Xml, &tools, &refs);
                assert_eq!((content, calls), (want_content.clone(), want.calls.clone()), "pieces {pattern:?} of {reply:?}");
            }
        }
    }

    /// A reply that is a flood of `<parameter=` (a model stuck in a loop writes such text) is read in time proportional to its length: each parameter's search for its end stops at the next `<parameter=`, so a closing tag
    /// far off is not looked for again and again (a flood of 1 MB took tens of seconds with the search to the end of the text).
    #[test]
    fn a_flood_of_parameter_tags_is_read_in_linear_time() {
        // (the bound has a wide margin for a slow CI runner: linear time takes well under a second here, the quadratic search took tens of seconds)
        let flood = "<parameter=".repeat(100_000);
        let t = std::time::Instant::now();
        assert_eq!(xml_params(&flood, false, &XML).len(), 99_999, "each but the last ends where the next begins");
        assert_eq!(xml_params(&flood, true, &XML).len(), 100_000);
        assert!(t.elapsed() < std::time::Duration::from_secs(10), "{:?}", t.elapsed());
        let t = std::time::Instant::now();
        let reply = format!("<tool_call>\n<function=forecast>\n{}", &flood[..11 * 20_000]);
        assert_eq!(xml(&reply).calls.len(), 0, "a parameter with no `>` leaves the reply as text");
        assert!(t.elapsed() < std::time::Duration::from_secs(10), "{:?}", t.elapsed());
    }

    const DSML_REPLY: &str = "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"lookup\">\n<｜DSML｜parameter name=\"word\" string=\"true\">glyph</｜DSML｜parameter>\n<｜DSML｜parameter name=\"limit\" string=\"false\">2</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"lookup\">\n<｜DSML｜parameter name=\"opts\" string=\"false\">{\"deep\":true,\"tags\":[\"a\"]}</｜DSML｜parameter>\n<｜DSML｜parameter name=\"n\" string=\"false\">not json</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"ping\">\n\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";

    /// DeepSeek-V4's DSML calls: its template's format; each invoke a call, a string parameter as it is, another as JSON (text that is not JSON stays a string); the text before the block, less
    /// its two new lines, the content; a reply with no block all text.
    /// A name the tools have without OpenAI's `functions.` namespace is read as theirs, whole and streamed; one they have with it, or one they do not have at all, as written.
    #[test]
    fn a_functions_namespace_is_the_tools_name() {
        let tools = vec![json::parse(
            r#"{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}}}}"#,
        )
        .unwrap()];
        let reply = "<tool_call>\n<function=functions.get_weather>\n<parameter=n>\n2\n</parameter>\n</function>\n</tool_call>";
        let e = extract(reply, Format::Xml, &tools);
        assert_eq!(e.calls, vec![Call { name: "get_weather".into(), arguments: "{\"n\": 2}".into() }]);
        let mut st = ToolStream::new(Format::Xml, &tools);
        let evs = [st.feed(reply), st.finish()].concat();
        assert!(evs.iter().any(|e| matches!(e, ToolEv::Start { name, .. } if name == "get_weather")));
        assert_eq!(known_name("functions.other", &tools), "functions.other");
        assert_eq!(known_name("get_weather", &tools), "get_weather");
        let hermes = extract(r#"<tool_call>{"name": "functions.get_weather", "arguments": {}}</tool_call>"#, Format::Hermes, &tools);
        assert_eq!(hermes.calls[0].name, "get_weather");
    }

    #[test]
    fn dsml_calls_read_from_a_whole_reply() {
        assert_eq!(Format::of(&Chat::from_template(crate::chat::DSV4_CHAT).unwrap()), Format::Dsml);
        let want = vec![
            call("lookup", r#"{"word": "glyph", "limit": 2}"#),
            call("lookup", r#"{"opts": {"deep": true, "tags": ["a"]}, "n": "not json"}"#),
            call("ping", "{}"),
        ];
        assert_eq!(extract(DSML_REPLY, Format::Dsml, &[]), Extracted { called: true, calls: want.clone(), content: None });
        assert_eq!(
            extract(&format!("Let me look.{DSML_REPLY}"), Format::Dsml, &[]),
            Extracted { called: true, calls: want, content: Some("Let me look.".into()) }
        );
        assert_eq!(extract("No tools needed.\n\n", Format::Dsml, &[]), text_only("No tools needed.\n\n"));
        // a forced call: the block's opening (and a named one's invoke) is in the prompt, and the reply read with it
        assert_eq!(forced_prefix(&ToolChoice::Required, Format::Dsml).unwrap(), "\n\n<｜DSML｜tool_calls>\n");
        let named = forced_prefix(&ToolChoice::Named("ping".into()), Format::Dsml).unwrap();
        assert_eq!(named, "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"ping\">\n");
        assert_eq!(extract(&format!("{named}\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"), Format::Dsml, &[]).calls, vec![call("ping", "{}")]);
    }

    /// A DSML reply streamed in any pieces adds up to the whole parse, and its content holds no new lines of the block's.
    #[test]
    fn the_dsml_stream_and_the_whole_parse_agree() {
        for reply in [DSML_REPLY.to_string(), format!("Let me look.{DSML_REPLY}"), "Just text\n\nwith lines\n\n".to_string()] {
            let want = extract(&reply, Format::Dsml, &[]);
            let cs: Vec<char> = reply.chars().collect();
            for n in [1usize, 2, 3, 7, 1000] {
                let pieces: Vec<String> = cs.chunks(n).map(|c| c.iter().collect()).collect();
                let refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
                let (content, calls) = sum(Format::Dsml, &[], &refs);
                assert_eq!((content, calls), (want.content.clone().unwrap_or_default(), want.calls.clone()), "pieces of {n} of {reply:?}");
            }
        }
    }

    /// GLM's calls, whole and streamed in pieces of any size: the name, each value typed by the tool's schema (a string as it is, an integer read, an object as JSON),
    /// the text before the first call as content; a block with no name is text; the forced prefixes; the format known by its template.
    #[test]
    fn glm_calls_whole_and_streamed() {
        let tools = vec![json::parse(r#"{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {"word": {"type": "string"}, "limit": {"type": "integer"}, "opts": {"type": "object"}}}}}"#).unwrap()];
        let calls = "<tool_call>lookup<arg_key>word</arg_key><arg_value>12</arg_value><arg_key>limit</arg_key><arg_value>2</arg_value><arg_key>opts</arg_key><arg_value>{\"deep\": true}</arg_value></tool_call><tool_call>lookup\n<arg_key>word</arg_key>\n<arg_value>glyph</arg_value>\n</tool_call>";
        let reply = format!("Let me look.{calls}");
        let want = vec![
            Call { name: "lookup".into(), arguments: r#"{"word": "12", "limit": 2, "opts": {"deep": true}}"#.into() },
            Call { name: "lookup".into(), arguments: r#"{"word": "glyph"}"#.into() },
        ];
        let e = extract(&reply, Format::Glm, &tools);
        assert_eq!((e.content.as_deref(), e.calls.clone()), (Some("Let me look."), want.clone()));
        let cs: Vec<char> = reply.chars().collect();
        for n in [1usize, 2, 3, 7, 1000] {
            let pieces: Vec<String> = cs.chunks(n).map(|c| c.iter().collect()).collect();
            let refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
            assert_eq!(sum(Format::Glm, &tools, &refs), ("Let me look.".to_string(), want.clone()), "pieces of {n}");
        }
        assert!(!extract("<tool_call><arg_key>x</arg_key><arg_value>1</arg_value></tool_call>", Format::Glm, &tools).called);
        assert_eq!(forced_prefix(&ToolChoice::Required, Format::Glm).as_deref(), Some("<tool_call>"));
        assert_eq!(forced_prefix(&ToolChoice::Named("lookup".into()), Format::Glm).as_deref(), Some("<tool_call>lookup"));
        let t = Chat::from_template("{% if tools %}<tool_call>{name}<arg_key>{key}</arg_key><arg_value>{value}</arg_value></tool_call>{% endif %}").unwrap();
        assert_eq!(Format::of(&t), Format::Glm);
    }
}
