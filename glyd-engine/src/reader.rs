//! Families whose replies are written in a markup of their own, read as the engine's other replies are. A reader sits between a reply's text and the reasoning split and the tool parser behind them
//! (`conv::Stream`): it turns the family's markup into the shape those two read, the thinking as `<think>` ... `</think>` and each call as a hermes `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`
//! (arguments written as JSON as they come), and drops the rest of the markup. One reader per markup, all behind `Reader`: Kimi-K3's tags (`xtml::Xtml`) and Gemma 4's (`Gemma`).

use crate::chat::Chat;

/// A reply's text, read as it comes.
pub trait Reader: Send {
    /// A piece of the reply's text, and what it reads as (text the markup might still be beginning is held back).
    fn feed(&mut self, text: &str) -> String;
    /// The reply's end: what was held, and what is left open, closed.
    fn finish(&mut self) -> String;
    /// The prompt's end where it opened a forced call (its text, as the model would have written it): the reader goes on from there, and returns what the opening reads as (what the tool parser then starts from).
    fn prime(&mut self, opening: &str) -> String;
}

impl Reader for crate::xtml::Xtml {
    fn feed(&mut self, text: &str) -> String {
        crate::xtml::Xtml::feed(self, text)
    }

    fn finish(&mut self) -> String {
        crate::xtml::Xtml::finish(self)
    }

    fn prime(&mut self, opening: &str) -> String {
        // (Kimi-K3's reader writes a call when it closes: what the opening reads as is nothing yet)
        crate::xtml::Xtml::feed(self, opening);
        String::new()
    }
}

/// Which reader a reply goes through, and where the prompt left it (`in_think`: the prompt ended inside the think channel).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReaderKind {
    Kimi { in_think: bool },
    Gemma { in_think: bool },
    Muse,
}

impl ReaderKind {
    /// The reader the chat template asks for, from the prompt's text: None for the formats the server reads as they are.
    pub fn of(chat: &Chat, prompt: &str) -> Option<ReaderKind> {
        if chat.is_xtml() {
            Some(ReaderKind::Kimi { in_think: prompt.ends_with(crate::conv::XTML_THINK) })
        } else if chat.is_gemma() {
            Some(ReaderKind::Gemma { in_think: prompt.ends_with(GEMMA_THINK) })
        } else if chat.is_muse() {
            Some(ReaderKind::Muse)
        } else {
            None
        }
    }

    pub fn build(self) -> Box<dyn Reader> {
        match self {
            ReaderKind::Kimi { in_think } => Box::new(crate::xtml::Xtml::new(in_think)),
            ReaderKind::Gemma { in_think } => Box::new(Gemma::new(in_think)),
            ReaderKind::Muse => Box::new(Muse::new()),
        }
    }
}

/// Where Gemma 4's generation prompt ends inside the thought channel (after a tool's answer, with thinking on).
pub const GEMMA_THINK: &str = "<|channel>thought\n";
const CH_OPEN: &str = "<|channel>";
const CH_CLOSE: &str = "<channel|>";
const CALL_OPEN: &str = "<|tool_call>";
const CALL_CLOSE: &str = "<tool_call|>";
const QUOTE: &str = "<|\"|>";
/// Markers that carry nothing here: the turn's ends and the tool answer's.
const DROPPED: [&str; 5] = ["<|tool_response>", "<tool_response|>", "<turn|>", "<|turn>", CALL_CLOSE];

/// What a call's arguments are expecting.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Expect {
    Key,
    Colon,
    Value,
    Next,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    Text,
    Thought,
    /// After `<|tool_call>`: `call:NAME` up to the `{` of its arguments.
    Name,
    Args,
}

/// Gemma 4's markup: the thought channel `<|channel>thought\n` ... `<channel|>`, and a call `<|tool_call>call:NAME{key:<|"|>text<|"|>,count:3,more:{...},list:[...]}<tool_call|>` (a string between `<|"|>` marks, a
/// key bare or between them, a number, `true`, `false` and `null` bare, objects and lists in braces and brackets). The thought becomes `<think>` ... `</think>`, a call a hermes one whose arguments are JSON, written as they come.
pub struct Gemma {
    buf: String,
    mode: Mode,
    /// The `<think>` of a reply that starts inside the thought channel, sent with its first text.
    lead: bool,
    /// Just inside the channel: its first word, `thought`, and the new line after it, are the format's.
    skip_word: bool,
    stack: Vec<char>,
    expect: Expect,
    in_str: bool,
    str_is_key: bool,
    /// A bare key or value is being written (a key's quote is open until its colon).
    bare: Option<Expect>,
}

impl Gemma {
    /// `in_think`: the prompt ended inside the thought channel.
    pub fn new(in_think: bool) -> Gemma {
        Gemma {
            buf: String::new(),
            mode: if in_think { Mode::Thought } else { Mode::Text },
            lead: in_think,
            skip_word: false,
            stack: Vec::new(),
            expect: Expect::Key,
            in_str: false,
            str_is_key: false,
            bare: None,
        }
    }

    /// The longest end of `s` that is the start of one of `marks` (held back until it is known whether it is one).
    fn held(s: &str, marks: &[&str]) -> usize {
        marks.iter().map(|m| (1..m.len()).rev().find(|&k| m.is_char_boundary(k) && s.ends_with(&m[..k])).unwrap_or(0)).max().unwrap_or(0)
    }

    fn escape(c: char, out: &mut String) {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }

    /// Text and thought: up to a marker, which is acted on.
    fn text(&mut self, out: &mut String) -> bool {
        if std::mem::take(&mut self.lead) {
            out.push_str("<think>");
        }
        if self.skip_word {
            // (`thought` and its new line)
            let word = "thought";
            if word.starts_with(self.buf.as_str()) && self.buf.len() < word.len() {
                return false;
            }
            if let Some(rest) = self.buf.strip_prefix(word) {
                if rest.is_empty() {
                    return false;
                }
                let n = word.len() + rest.starts_with('\n') as usize;
                self.buf.drain(..n);
            }
            self.skip_word = false;
        }
        let marks: Vec<&str> = [CH_OPEN, CH_CLOSE, CALL_OPEN].into_iter().chain(DROPPED).collect();
        let found = marks.iter().filter_map(|m| self.buf.find(m).map(|i| (i, *m))).min_by_key(|&(i, m)| (i, std::cmp::Reverse(m.len())));
        let Some((at, mark)) = found else {
            let n = self.buf.len() - Self::held(&self.buf, &marks);
            let t: String = self.buf.drain(..n).collect();
            out.push_str(&t);
            return false;
        };
        let t: String = self.buf.drain(..at + mark.len()).collect();
        out.push_str(&t[..at]);
        match (mark, self.mode) {
            (CH_OPEN, Mode::Text) => {
                out.push_str("<think>");
                self.mode = Mode::Thought;
                self.skip_word = true;
            }
            (CH_CLOSE, Mode::Thought) => {
                out.push_str("</think>");
                self.mode = Mode::Text;
            }
            (CALL_OPEN, _) => {
                if self.mode == Mode::Thought {
                    out.push_str("</think>");
                }
                self.mode = Mode::Name;
            }
            _ => {}
        }
        true
    }

    /// `call:NAME{`: the call's opening, as a hermes one up to its arguments.
    fn name(&mut self, out: &mut String) -> bool {
        let Some(i) = self.buf.find('{') else { return false };
        let header: String = self.buf.drain(..=i).collect();
        let name = header[..i].trim().trim_start_matches("call:").trim().to_string();
        out.push_str(&format!("<tool_call>\n{{\"name\": {}, \"arguments\": {{", crate::server::jstr(&name)));
        self.stack = vec!['{'];
        (self.expect, self.in_str, self.str_is_key, self.bare) = (Expect::Key, false, false, None);
        self.mode = Mode::Args;
        true
    }

    /// The arguments, a character at a time: written as JSON until the `}` that closes them.
    fn args(&mut self, out: &mut String) -> bool {
        let mut pos = 0;
        while pos < self.buf.len() {
            let rest = &self.buf[pos..];
            if rest.starts_with(QUOTE) {
                pos += QUOTE.len();
                if self.in_str {
                    out.push('"');
                    self.in_str = false;
                    self.expect = if self.str_is_key { Expect::Colon } else { Expect::Next };
                } else {
                    self.close_bare(out);
                    self.str_is_key = self.expect == Expect::Key;
                    self.in_str = true;
                    out.push('"');
                }
                continue;
            }
            if QUOTE.starts_with(rest) {
                break; // (it may be the mark: wait)
            }
            let c = rest.chars().next().unwrap_or(' ');
            pos += c.len_utf8();
            if self.in_str {
                Self::escape(c, out);
                continue;
            }
            match c {
                ':' if self.bare == Some(Expect::Key) => {
                    out.push_str("\":");
                    self.bare = None;
                    self.expect = Expect::Value;
                }
                ':' => {
                    out.push(':');
                    self.expect = Expect::Value;
                }
                c if c.is_whitespace() => {}
                '{' | '[' => {
                    self.close_bare(out);
                    out.push(c);
                    self.stack.push(c);
                    self.expect = if c == '{' { Expect::Key } else { Expect::Value };
                }
                '}' | ']' => {
                    self.close_bare(out);
                    self.stack.pop();
                    if self.stack.is_empty() {
                        // the arguments are closed: the call's object, and its block
                        out.push_str("}}\n</tool_call>");
                        self.buf.drain(..pos);
                        self.mode = Mode::Text;
                        return true;
                    }
                    out.push(c);
                    self.expect = Expect::Next;
                }
                ',' => {
                    self.close_bare(out);
                    out.push(',');
                    self.expect = if self.stack.last() == Some(&'{') { Expect::Key } else { Expect::Value };
                }
                c => {
                    if self.bare.is_none() {
                        self.bare = Some(self.expect);
                        if self.expect == Expect::Key {
                            out.push('"');
                        }
                    }
                    if self.bare == Some(Expect::Key) {
                        Self::escape(c, out);
                    } else {
                        out.push(c);
                    }
                }
            }
        }
        self.buf.drain(..pos);
        false
    }

    /// A bare value (a number, `true`) ends where the next mark begins; a bare key's quote is closed by its colon.
    fn close_bare(&mut self, out: &mut String) {
        if self.bare == Some(Expect::Key) {
            out.push('"');
        }
        if self.bare.take().is_some() {
            self.expect = Expect::Next;
        }
    }
}

impl Reader for Gemma {
    fn feed(&mut self, text: &str) -> String {
        self.buf.push_str(text);
        let mut out = String::new();
        loop {
            let again = match self.mode {
                Mode::Text | Mode::Thought => self.text(&mut out),
                Mode::Name => self.name(&mut out),
                Mode::Args => self.args(&mut out),
            };
            if !again {
                break;
            }
        }
        out
    }

    fn finish(&mut self) -> String {
        let mut out = String::new();
        match self.mode {
            Mode::Text | Mode::Thought => {
                if std::mem::take(&mut self.lead) {
                    out.push_str("<think>");
                }
                out.push_str(&std::mem::take(&mut self.buf));
            }
            Mode::Name => {}
            Mode::Args => {
                // (a call left open: closed where it can be, so that what was read is not lost)
                if self.in_str {
                    out.push('"');
                } else if self.bare == Some(Expect::Key) {
                    out.push_str("\":null");
                } else if self.expect == Expect::Value && self.bare.is_none() {
                    out.push_str("null");
                }
                while let Some(open) = self.stack.pop() {
                    out.push(if open == '{' { '}' } else { ']' });
                }
                out.push_str("}\n</tool_call>");
                self.mode = Mode::Text;
            }
        }
        out
    }

    fn prime(&mut self, opening: &str) -> String {
        self.feed(opening)
    }
}

/// Muse's ATEM replies: a reply is messages, each `[<|start|>assistant] to=RECIPIENT<|message|>BODY<|eom|>` (more follows) or `...<|eot|>` (the turn ends): the recipient `self` is the thinking, `user` the answer, any other a tool,
/// whose body is its `<atem:function_calls>` block. The thinking becomes `<think>` ... `</think>`, the answer is text and a tool's block goes on as it is, for the tool parser (`tools::Format::Atem`); the markup is dropped.
pub struct Muse {
    buf: String,
    /// Reading a message's header (up to `<|message|>`), else its body for this recipient.
    body: Option<Recipient>,
    think_open: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Recipient {
    Myself,
    User,
    Tool,
}

const M_MSG: &str = "<|message|>";
const M_ENDS: [&str; 3] = ["<|eom|>", "<|eot|>", "<|start|>"];

impl Muse {
    pub fn new() -> Muse {
        Muse { buf: String::new(), body: None, think_open: false }
    }

    fn step(&mut self, out: &mut String) -> bool {
        match self.body {
            None => {
                let Some(i) = self.buf.find(M_MSG) else { return false };
                let header: String = self.buf.drain(..i + M_MSG.len()).collect();
                let header = &header[..i];
                let to = header.split("to=").nth(1).map(|t| t.split_whitespace().next().unwrap_or("")).unwrap_or("user");
                let r = match to {
                    "self" => Recipient::Myself,
                    "" | "user" => Recipient::User,
                    _ => Recipient::Tool,
                };
                if r != Recipient::Myself && std::mem::take(&mut self.think_open) {
                    out.push_str("</think>");
                }
                if r == Recipient::Myself && !self.think_open {
                    out.push_str("<think>");
                    self.think_open = true;
                }
                self.body = Some(r);
                true
            }
            Some(_) => {
                let found = M_ENDS.iter().filter_map(|m| self.buf.find(m).map(|i| (i, *m))).min_by_key(|&(i, _)| i);
                let Some((at, mark)) = found else {
                    let n = self.buf.len() - Gemma::held(&self.buf, &M_ENDS);
                    let t: String = self.buf.drain(..n).collect();
                    out.push_str(&t);
                    return false;
                };
                let t: String = self.buf.drain(..at + mark.len()).collect();
                out.push_str(&t[..at]);
                self.body = None;
                // (`<|start|>assistant`: the next message's own start, dropped; a header may follow without it)
                if mark == "<|start|>" {
                    self.buf = self.buf.trim_start_matches("assistant").to_string();
                }
                true
            }
        }
    }
}

impl Default for Muse {
    fn default() -> Muse {
        Muse::new()
    }
}

impl Reader for Muse {
    fn feed(&mut self, text: &str) -> String {
        self.buf.push_str(text);
        let mut out = String::new();
        // (`<|start|>assistant` at the head of a header is dropped as it comes)
        while self.step(&mut out) {
            if self.body.is_none() {
                for lead in ["<|start|>assistant", "<|start|>"] {
                    if let Some(rest) = self.buf.strip_prefix(lead) {
                        self.buf = rest.to_string();
                        break;
                    }
                }
            }
        }
        out
    }

    fn finish(&mut self) -> String {
        let mut out = String::new();
        if self.body.is_some() {
            out.push_str(&std::mem::take(&mut self.buf));
        }
        out
    }

    fn prime(&mut self, opening: &str) -> String {
        self.feed(opening)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reply read in pieces of `n` characters: what it reads as.
    fn read(reply: &str, n: usize, in_think: bool) -> String {
        let mut g = Gemma::new(in_think);
        let chars: Vec<char> = reply.chars().collect();
        let mut out = String::new();
        for c in chars.chunks(n.max(1)) {
            out.push_str(&g.feed(&c.iter().collect::<String>()));
        }
        out + &g.finish()
    }

    /// The same whatever the pieces: a marker split across tokens reads as one.
    fn same(reply: &str, want: &str, in_think: bool) {
        for n in [1, 2, 3, 5, 7, 1000] {
            assert_eq!(read(reply, n, in_think), want, "pieces of {n}: {reply:?}");
        }
    }

    #[test]
    fn the_thought_channel_is_the_think_block() {
        same("<|channel>thought\nLet me see.\n<channel|>It is 4.", "<think>Let me see.\n</think>It is 4.", false);
        // a reply that starts inside the channel (the prompt ended with `<|channel>thought\n`)
        same("Hmm.\n<channel|>Yes.<turn|>", "<think>Hmm.\n</think>Yes.", true);
        // no thinking: the text as it is, the turn's end dropped
        same("Hello there.<turn|>", "Hello there.", false);
        // an empty channel
        same("<|channel>thought\n<channel|>Fine.", "<think></think>Fine.", false);
        // a thought that never ends
        same("<|channel>thought\nStill thinking", "<think>Still thinking", false);
        // text that is only like a marker
        same("a < b and c <| d", "a < b and c <| d", false);
    }

    #[test]
    fn a_call_is_a_hermes_one() {
        let call = "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,days:3,metric:true,tags:[<|\"|>a<|\"|>,<|\"|>b<|\"|>],opts:{units:<|\"|>c<|\"|>,deep:{x:1.5}}}<tool_call|>";
        let want = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\":\"Paris\",\"days\":3,\"metric\":true,\"tags\":[\"a\",\"b\"],\"opts\":{\"units\":\"c\",\"deep\":{\"x\":1.5}}}}\n</tool_call>";
        same(call, want, false);
        // keys between the marks too, a string with what JSON escapes, no arguments
        same(
            "<|tool_call>call:f{<|\"|>k<|\"|>:<|\"|>say \"hi\"\nnow<|\"|>}<tool_call|>",
            "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"k\":\"say \\\"hi\\\"\\nnow\"}}\n</tool_call>",
            false,
        );
        same("<|tool_call>call:ping{}<tool_call|>", "<tool_call>\n{\"name\": \"ping\", \"arguments\": {}}\n</tool_call>", false);
        // text before, two calls, the tool answer's mark after
        same(
            "Looking.<|tool_call>call:a{x:1}<tool_call|><|tool_call>call:b{y:null}<tool_call|><|tool_response>",
            "Looking.<tool_call>\n{\"name\": \"a\", \"arguments\": {\"x\":1}}\n</tool_call><tool_call>\n{\"name\": \"b\", \"arguments\": {\"y\":null}}\n</tool_call>",
            false,
        );
        // thinking then a call
        same(
            "<|channel>thought\nNeed the weather.\n<channel|><|tool_call>call:w{c:<|\"|>P<|\"|>}<tool_call|>",
            "<think>Need the weather.\n</think><tool_call>\n{\"name\": \"w\", \"arguments\": {\"c\":\"P\"}}\n</tool_call>",
            false,
        );
    }

    /// A call is written as its arguments come: the opening when the name is complete, then each piece of the arguments.
    #[test]
    fn a_call_streams_its_arguments() {
        let mut g = Gemma::new(false);
        let pieces = ["<|tool_call>call:get_wea", "ther{ci", "ty:<|\"|>Par", "is<|\"|>,n:", "12}", "<tool_call|>"];
        let outs: Vec<String> = pieces.iter().map(|p| g.feed(p)).collect();
        assert_eq!(outs[0], "");
        assert_eq!(outs[1], "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"ci");
        assert_eq!(outs[2], "ty\":\"Par");
        assert_eq!(outs[3], "is\",\"n\":");
        assert_eq!(outs[4], "12}}\n</tool_call>");
        assert_eq!(outs[5], "");
    }

    /// A call that is cut is closed where it can be.
    #[test]
    fn a_call_cut_short_is_closed() {
        let cut = |s: &str| read(s, 4, false);
        assert_eq!(cut("<|tool_call>call:f{a:<|\"|>unfinished"), "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\":\"unfinished\"}}\n</tool_call>");
        assert_eq!(cut("<|tool_call>call:f{a:1,b:{c:2"), "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\":1,\"b\":{\"c\":2}}}\n</tool_call>");
        assert_eq!(cut("<|tool_call>call:f{a:"), "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\":null}}\n</tool_call>");
        assert_eq!(cut("<|tool_call>call:f"), "");
    }

    /// A forced call: the prompt ends with the opening, which reads as the hermes opening where the name is given.
    #[test]
    fn a_forced_call_is_primed() {
        let mut g = Gemma::new(false);
        assert_eq!(g.prime("<|tool_call>call:get_weather{"), "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {");
        assert_eq!(g.feed("city:<|\"|>P<|\"|>}<tool_call|>"), "\"city\":\"P\"}}\n</tool_call>");
        let mut g = Gemma::new(false);
        assert_eq!(g.prime("<|tool_call>call:"), "");
        assert_eq!(g.feed("get_weather{x:1}"), "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"x\":1}}\n</tool_call>");
    }
}
