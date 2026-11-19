//! Kimi-K3's replies (its tag format: `<|open|>TAG KEY="VALUE"...<|sep|>` ... `<|close|>TAG<|sep|>`; research/kimi-k3.md 8.3) read as the engine's other
//! replies are, so that the reasoning split and the tool calls are the ones the server already has: the think channel as `<think>` ... `</think>`, the
//! response's text as it is, each call as a hermes `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` (each argument by its `type`: a string's
//! text as a JSON string, any other's text the JSON it is), the rest of the markup (the message, the response's and the tools' tags) dropped.

use glyd_gpu::json::{self, Value};

const OPEN: &str = "<|open|>";
const CLOSE: &str = "<|close|>";
const SEP: &str = "<|sep|>";

/// A tag's name and attributes (`&amp;` and `&quot;` in a value read back).
fn tag(t: &str) -> (String, Vec<(String, String)>) {
    let t = t.trim();
    let (name, mut rest) = t.split_once(' ').unwrap_or((t, ""));
    let mut attrs = Vec::new();
    while let Some(eq) = rest.find("=\"") {
        let key = rest[..eq].trim().to_string();
        let after = &rest[eq + 2..];
        let Some(end) = after.find('"') else { break };
        attrs.push((key, after[..end].replace("&quot;", "\"").replace("&amp;", "&")));
        rest = &after[end + 1..];
    }
    (name.to_string(), attrs)
}

/// A forced call's opening, the prompt's end (`in_think`: the prompt ends inside the think channel, closed empty, as the format keeps it): the response (empty), the tools
/// and the call opened, to its tool's name (`name`: a named function's, its tag closed for the arguments).
pub fn forced(in_think: bool, name: Option<&str>) -> String {
    let close = |t: &str| format!("{CLOSE}{t}{SEP}");
    let mut s = if in_think { format!("{}{OPEN}response{SEP}", close("think")) } else { String::new() };
    s.push_str(&format!("{}{OPEN}tools{SEP}{OPEN}call tool=\"", close("response")));
    if let Some(n) = name {
        s.push_str(&format!("{}\" index=\"1\"{SEP}", n.replace('&', "&amp;").replace('"', "&quot;")));
    }
    s
}

fn attr<'a>(attrs: &'a [(String, String)], key: &str) -> Option<&'a str> {
    attrs.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
}

/// The reader: the text not yet read (what may be the start of a tag is held), where it is (the tags open), the call and the argument being read.
pub struct Xtml {
    buf: String,
    open: Vec<String>,
    call: Option<(String, Vec<(String, String)>)>,
    arg: Option<(String, String, String)>,
    raw: Option<String>,
    /// The `<think>` a reply that starts inside the think channel has not written, sent with its first text.
    lead: bool,
}

impl Xtml {
    /// `in_think`: the prompt ends inside the think channel (the template opens it).
    pub fn new(in_think: bool) -> Xtml {
        Xtml { buf: String::new(), open: if in_think { vec!["think".into()] } else { Vec::new() }, call: None, arg: None, raw: None, lead: in_think }
    }

    /// A piece of the reply's text, and what it reads as.
    pub fn feed(&mut self, text: &str) -> String {
        self.buf.push_str(text);
        let mut out = String::new();
        if std::mem::take(&mut self.lead) {
            out.push_str("<think>");
        }
        loop {
            let next = [self.buf.find(OPEN), self.buf.find(CLOSE)].into_iter().flatten().min();
            let Some(at) = next else {
                // (text up to what may begin a tag)
                let keep = [OPEN, CLOSE].iter().map(|t| (1..t.len()).rev().find(|&k| self.buf.ends_with(&t[..k])).unwrap_or(0)).max().unwrap_or(0);
                let n = self.buf.len() - keep;
                let t: String = self.buf.drain(..n).collect();
                self.text(&t, &mut out);
                break;
            };
            let t: String = self.buf.drain(..at).collect();
            self.text(&t, &mut out);
            let closing = self.buf.starts_with(CLOSE);
            let mark = if closing { CLOSE.len() } else { OPEN.len() };
            let Some(sep) = self.buf[mark..].find(SEP) else { break };
            let inner = self.buf[mark..mark + sep].to_string();
            self.buf.drain(..mark + sep + SEP.len());
            let (name, attrs) = tag(&inner);
            if closing {
                self.close(&name, &mut out);
            } else {
                self.start(name, attrs, &mut out);
            }
        }
        out
    }

    /// The reply's end: what was held, and a call left open written as far as it was read.
    pub fn finish(&mut self) -> String {
        let rest = std::mem::take(&mut self.buf);
        let mut out = String::new();
        self.text(&rest, &mut out);
        if self.arg.is_some() {
            self.close("argument", &mut out);
        }
        if self.call.is_some() {
            self.close("call", &mut out);
        }
        out
    }

    fn text(&mut self, t: &str, out: &mut String) {
        if t.is_empty() {
            return;
        }
        if let Some(a) = &mut self.arg {
            a.2.push_str(t);
        } else if let Some(r) = &mut self.raw {
            r.push_str(t);
        } else if self.call.is_none() && !self.open.iter().any(|o| o == "tools") {
            // (the think channel's text and the response's; text outside them, a model's stray words, too)
            out.push_str(t);
        }
    }

    fn start(&mut self, name: String, attrs: Vec<(String, String)>, out: &mut String) {
        match name.as_str() {
            "think" => out.push_str("<think>"),
            "call" => self.call = Some((attr(&attrs, "tool").unwrap_or("").to_string(), Vec::new())),
            "argument" => self.arg = Some((attr(&attrs, "key").unwrap_or("").to_string(), attr(&attrs, "type").unwrap_or("string").to_string(), String::new())),
            "json" => self.raw = Some(String::new()),
            _ => {}
        }
        self.open.push(name);
    }

    fn close(&mut self, name: &str, out: &mut String) {
        if let Some(i) = self.open.iter().rposition(|o| o == name) {
            self.open.truncate(i);
        }
        match name {
            "think" => out.push_str("</think>"),
            "argument" => {
                if let (Some((key, ty, text)), Some(call)) = (self.arg.take(), &mut self.call) {
                    // (a string's text as it is; another type's text the JSON it is, else kept as a string)
                    let value = match (ty.as_str(), json::parse(text.trim())) {
                        ("string", _) | (_, Err(_)) => Value::from(text.as_str()),
                        (_, Ok(v)) => v,
                    };
                    call.1.push((key, crate::chat::py_dumps(&value)));
                }
            }
            "call" => {
                if let Some((name, args)) = self.call.take() {
                    let args = match self.raw.take() {
                        Some(raw) => raw.trim().to_string(),
                        None => format!("{{{}}}", args.iter().map(|(k, v)| format!("{}: {v}", crate::chat::py_dumps(&Value::from(k.as_str())))).collect::<Vec<_>>().join(", ")),
                    };
                    out.push_str(&format!("<tool_call>{{\"name\": {}, \"arguments\": {args}}}</tool_call>", crate::chat::py_dumps(&Value::from(name.as_str()))));
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REPLY: &str = "Let me look up the weather.<|close|>think<|sep|><|open|>response<|sep|>Checking.<|close|>response<|sep|><|open|>tools<|sep|><|open|>call tool=\"get_weather\" index=\"1\"<|sep|><|open|>argument key=\"city\" type=\"string\"<|sep|>Paris<|close|>argument<|sep|><|open|>argument key=\"days\" type=\"number\"<|sep|>3<|close|>argument<|sep|><|open|>argument key=\"opts\" type=\"object\"<|sep|>{\"deep\": true}<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>";

    fn read(text: &str, pieces: usize, in_think: bool) -> String {
        let mut x = Xtml::new(in_think);
        let cs: Vec<char> = text.chars().collect();
        let mut out = String::new();
        for p in cs.chunks(pieces) {
            out.push_str(&x.feed(&p.iter().collect::<String>()));
        }
        out + &x.finish()
    }

    /// A reply that starts in the think channel reads as `<think>` reasoning `</think>`, the response's text, and its call as a hermes one, whatever the pieces it comes in.
    #[test]
    fn a_reply_reads_as_think_answer_and_hermes_calls() {
        let want = "<think>Let me look up the weather.</think>Checking.<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 3, \"opts\": {\"deep\": true}}}</tool_call>";
        for n in [1, 2, 5, 13, 1000] {
            assert_eq!(read(REPLY, n, true), want, "pieces of {n}");
        }
        // (thinking off: the prompt opened the response)
        assert_eq!(read("Paris.<|close|>response<|sep|><|close|>message<|sep|>", 3, false), "Paris.");
        // (a value with a quote, an argument of no JSON, a call's arguments as raw JSON)
        assert_eq!(tag(r#"call tool="a&quot;b" index="2""#), ("call".to_string(), vec![("tool".to_string(), "a\"b".to_string()), ("index".to_string(), "2".to_string())]));
        let raw = "<|open|>tools<|sep|><|open|>call tool=\"f\" index=\"1\"<|sep|><|open|>json type=\"object\"<|sep|>{\"a\": 1}<|close|>json<|sep|><|close|>call<|sep|><|close|>tools<|sep|>";
        assert_eq!(read(raw, 4, false), "<tool_call>{\"name\": \"f\", \"arguments\": {\"a\": 1}}</tool_call>");
    }

    /// A forced call: the reader primed with the opening the prompt ends with reads the model's rest as the call, by name or naming it.
    #[test]
    fn a_forced_call_reads_from_its_opening() {
        assert_eq!(forced(true, Some("get_weather")), "<|close|>think<|sep|><|open|>response<|sep|><|close|>response<|sep|><|open|>tools<|sep|><|open|>call tool=\"get_weather\" index=\"1\"<|sep|>");
        assert_eq!(forced(false, None), "<|close|>response<|sep|><|open|>tools<|sep|><|open|>call tool=\"");
        let rest = "<|open|>argument key=\"city\" type=\"string\"<|sep|>Paris<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>";
        let want = "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>";
        for (open, text) in [(forced(true, Some("get_weather")), rest.to_string()), (forced(true, None), format!("get_weather\" index=\"1\"<|sep|>{rest}"))] {
            let mut x = Xtml::new(false);
            x.feed(&open);
            let mut out = String::new();
            for p in text.chars().collect::<Vec<_>>().chunks(3) {
                out.push_str(&x.feed(&p.iter().collect::<String>()));
            }
            assert_eq!(out + &x.finish(), want);
        }
    }
}
