//! One choice of a completions request as it is made: the sequence's tokens as text (never half a character), cut at a stop string, split into the thinking and the answer where the model writes
//! `<think>`, and the log-probabilities that came with the tokens, handed on with the text that follows them. The server runs one `Choice` for each of the request's `n`.

use crate::conv::{Part, Splitter, Stops, Stream};
use crate::ffi::Result;
use crate::model::Finish;
use crate::sampler::Lp;
use crate::tok::Tok;
use crate::tools::{self, Call, Format, ToolChoice, ToolEv, ToolStream};
use glyd_gpu::json::Value;

/// What a choice is made with.
#[derive(Clone, Debug, Default)]
pub struct Cfg {
    /// A chat: the thinking is split from the answer (a plain completion is all text).
    pub chat: bool,
    /// The prompt ended inside the thinking (a template that opens `<think>` itself).
    pub in_thinking: bool,
    pub stops: Vec<String>,
    /// The stop string that ends the reply stays in its text; the text leaves out the special tokens.
    pub include_stop: bool,
    pub skip_special: bool,
    /// The thinking is part of the reply (vLLM's `include_reasoning`; false: it is left out of what is sent, and counted).
    pub hide_reasoning: bool,
    /// Tool calls are read from the answer where there are tools and the choice is not none.
    pub tools: Option<ToolsCfg>,
    /// What the prompt already wrote of the answer (JSON mode's `{`): the answer's first text.
    pub lead: String,
    /// A family's own markup (Kimi-K3's tags, Gemma 4's), read as the other models' replies are (`reader::Reader`).
    pub reader: Option<crate::reader::ReaderKind>,
}

/// How the answer is read for tool calls.
#[derive(Clone, Debug)]
pub struct ToolsCfg {
    /// Auto, Required or Named (a choice of none reads no calls).
    pub choice: ToolChoice,
    /// More than one call in the reply (`parallel_tool_calls`).
    pub parallel: bool,
    /// How the model writes a call (its chat template's format).
    pub format: Format,
    /// The request's tools (an XML call's parameters are read by the types their schemas give them).
    pub list: Vec<Value>,
    /// What the prompt ends with where the reply must be a call (it is part of the reply's text for the parser).
    pub prefix: String,
    /// The reply is streamed: calls are read as they are written (else the whole answer is read when it ends).
    pub stream: bool,
}

/// What a token or the end of the reply adds to what the client has: new thinking, new answer, the log-probabilities of the tokens since the last delta (those whose text is held back ride with the next).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Delta {
    pub reasoning: String,
    pub content: String,
    /// Tool calls begun or continued (a streamed reply's: `ToolEv::Start` and `ToolEv::Args`).
    pub calls: Vec<ToolEv>,
    pub lps: Vec<Lp>,
}

impl Delta {
    pub fn is_empty(&self) -> bool {
        !self.has_text() && self.lps.is_empty()
    }

    fn has_text(&self) -> bool {
        !self.reasoning.is_empty() || !self.content.is_empty() || !self.calls.is_empty()
    }
}

/// How a choice ended: vLLM's finish reason, and the tokens it made (the end-of-sequence token is counted: vLLM counts it). Of a reply that is not streamed, the tool calls read from its answer and the answer's text
/// that is left (None: there is none, and the message's content is null).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Done {
    pub finish_reason: &'static str,
    pub tokens: usize,
    pub calls: Vec<Call>,
    pub content: Option<String>,
    /// The stop string that ended the reply, where one did.
    pub stop: Option<String>,
}

pub struct Choice {
    cfg: Cfg,
    stream: Stream,
    stops: Stops,
    split: Splitter,
    pending: Vec<Lp>,
    /// Everything shown so far, by part (what a reply that is not streamed is made of).
    pub reasoning: String,
    pub content: String,
    tokens: usize,
    /// The answer as calls are read from it as it is written, and what the forced opening of the reply already said.
    tool: Option<ToolStream>,
    opening: Vec<ToolEv>,
    /// The answer's first text, sent with the first delta.
    lead: String,
}

impl Choice {
    pub fn new(cfg: Cfg) -> Choice {
        let (mut tool, mut opening) = (None, Vec::new());
        let mut stream = if cfg.skip_special { Stream::skipping() } else { Stream::new() }.with_reader(cfg.reader);
        // (a forced opening is the family's own markup: its reader is primed with it. Kimi-K3's writes a call when the call closes, so the opening reads as nothing and the call it writes is the parser's whole; Gemma 4's
        // opening reads as the hermes call's beginning, which the tool parser starts from)
        if let Some(t) = cfg.tools.as_ref().filter(|t| cfg.reader.is_some() && !t.prefix.is_empty()) {
            stream.prime(&t.prefix);
        }
        let lead_in = |t: &ToolsCfg| if cfg.reader.is_some() { stream.primed().to_string() } else { t.prefix.clone() };
        if let Some(t) = cfg.tools.as_ref().filter(|t| t.stream) {
            let mut ts = ToolStream::new(t.format, &t.list);
            opening = ts.feed(&lead_in(t));
            tool = Some(ts);
        }
        let mut c = Choice {
            stream,
            stops: Stops::new(&cfg.stops).including(cfg.include_stop),
            split: Splitter::new(cfg.in_thinking),
            pending: Vec::new(),
            reasoning: String::new(),
            content: cfg.lead.clone(),
            tokens: 0,
            tool,
            opening: Vec::new(),
            lead: cfg.lead.clone(),
            cfg,
        };
        // (what the forced opening of the reply says is sent with the first delta)
        let mut first = Delta::default();
        for ev in opening.drain(..) {
            c.tool_event(ev, &mut first);
        }
        c.opening = first.calls;
        c
    }

    /// An event of the tool stream into a delta: text is content, a call is a call (the first only, where the request has no parallel calls).
    fn tool_event(&mut self, ev: ToolEv, d: &mut Delta) {
        match ev {
            ToolEv::Content(t) => {
                self.content.push_str(&t);
                d.content.push_str(&t);
            }
            ToolEv::Start { index, .. } | ToolEv::Args { index, .. } if index > 0 && !self.cfg.tools.as_ref().is_some_and(|t| t.parallel) => {}
            call => d.calls.push(call),
        }
    }

    /// The log-probabilities of the next token (they come before it).
    pub fn lp(&mut self, lp: Lp) {
        self.pending.push(lp);
    }

    /// Whether a stop string was found: the sequence is to be ended, what follows the string is not shown.
    pub fn stopped(&self) -> bool {
        self.stops.hit
    }

    fn route(&mut self, text: &str, d: &mut Delta) {
        if text.is_empty() {
            return;
        }
        if !self.cfg.chat {
            self.content.push_str(text);
            d.content.push_str(text);
            return;
        }
        for (part, t) in self.split.feed(text) {
            self.show(part, &t, d);
        }
    }

    fn show(&mut self, part: Part, t: &str, d: &mut Delta) {
        match part {
            Part::Reasoning => {
                self.reasoning.push_str(t);
                if !self.cfg.hide_reasoning {
                    d.reasoning.push_str(t);
                }
            }
            Part::Content => match self.tool.as_mut() {
                Some(ts) => {
                    for ev in ts.feed(t) {
                        self.tool_event(ev, d);
                    }
                }
                None => {
                    self.content.push_str(t);
                    d.content.push_str(t);
                }
            },
        }
    }

    /// A token of the reply: what its text adds (nothing where it ends in half a character or may begin a stop string).
    pub fn token(&mut self, tok: &Tok, id: u32) -> Result<Delta> {
        self.tokens += 1;
        let piece = self.stream.push(tok, id)?;
        let shown = self.stops.feed(&piece);
        let mut d = Delta { calls: std::mem::take(&mut self.opening), content: std::mem::take(&mut self.lead), ..Delta::default() };
        self.route(&shown, &mut d);
        if d.has_text() {
            d.lps = std::mem::take(&mut self.pending);
        }
        Ok(d)
    }

    /// The end of the reply: what was held back, the log-probabilities not sent yet, and how it ended.
    pub fn end(&mut self, tok: &Tok, finish: Finish) -> Result<(Delta, Done)> {
        let mut d = Delta { calls: std::mem::take(&mut self.opening), content: std::mem::take(&mut self.lead), ..Delta::default() };
        if !self.stops.hit {
            let tail = self.stream.finish(tok)?;
            let tail = self.stops.feed(&tail) + &self.stops.flush();
            self.route(&tail, &mut d);
        }
        if self.cfg.chat {
            for (part, t) in self.split.finish() {
                self.show(part, &t, &mut d);
            }
        }
        if let Some(mut ts) = self.tool.take() {
            for ev in ts.finish() {
                self.tool_event(ev, &mut d);
            }
            self.tool = Some(ts);
        }
        d.lps = std::mem::take(&mut self.pending);
        let mut finish_reason = match finish {
            _ if self.stops.hit => "stop",
            Finish::Length => "length",
            Finish::Stop | Finish::Stopped => "stop",
        };
        let (mut calls, mut content) = (Vec::new(), Some(self.content.clone()));
        if let Some(t) = self.cfg.tools.as_ref() {
            let named = matches!(t.choice, ToolChoice::Named(_));
            if t.stream {
                // a call was streamed: the reply is a tool call's (but a named function's, which ends as any reply does)
                if self.tool.as_ref().is_some_and(|ts| ts.emitted) && !named {
                    finish_reason = "tool_calls";
                }
            } else {
                let lead_in = if self.cfg.reader.is_some() { self.stream.primed() } else { t.prefix.as_str() };
                let mut ex = tools::extract(&(lead_in.to_string() + &self.content), t.format, &t.list);
                if ex.called {
                    if !t.parallel {
                        ex.calls.truncate(1);
                    }
                    if !named && (t.choice == ToolChoice::Auto || finish_reason == "stop") {
                        finish_reason = "tool_calls";
                    }
                    content = if t.choice == ToolChoice::Auto { ex.content } else { Some(ex.content.unwrap_or_default()) };
                    calls = ex.calls;
                }
            }
        }
        // the end-of-sequence token is not in the reply's text and is one of its tokens
        let tokens = self.tokens + (finish == Finish::Stop && !self.stops.hit) as usize;
        Ok((d, Done { finish_reason, tokens, calls, content, stop: self.stops.matched.clone() }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tok::tests::toy;

    fn lp(token: u32) -> Lp {
        Lp { token, logprob: -1.0, top: vec![] }
    }

    /// The text of a choice as it comes: held back where it may begin a stop string, cut where one is found, the log-probabilities with the text that follows their tokens, the end token counted.
    #[test]
    fn tokens_become_text_and_the_log_probabilities_go_with_them() {
        let t = toy(&["Hel", "lo", " wor", "ld", " EN", "D", " after"]);
        let mut c = Choice::new(Cfg { chat: false, stops: vec![" END".to_string()], ..Cfg::default() });
        let mut shown = String::new();
        let mut lps = Vec::new();
        for id in [1u32, 2, 3, 4, 5] {
            c.lp(lp(id));
            let d = c.token(&t, id).unwrap();
            shown.push_str(&d.content);
            lps.extend(d.lps.iter().map(|l| l.token));
            assert!(!c.stopped());
        }
        // " EN" may begin the stop string: its text is held back, and so is its log-probability, which rides with what comes after it
        assert_eq!(shown, "Hello world");
        assert_eq!(lps, vec![1, 2, 3, 4]);
        c.lp(lp(6));
        let d = c.token(&t, 6).unwrap();
        assert!(c.stopped() && d.is_empty(), "the stop string is not shown");
        // the choice is ended where the stop string is found: the tokens are the ones it took (the one with the stop string's end among them), and the log-probabilities held back come with the end
        let (d, done) = c.end(&t, Finish::Stopped).unwrap();
        assert_eq!((d.content.as_str(), done.finish_reason, done.tokens), ("", "stop", 6));
        assert_eq!(d.lps.iter().map(|l| l.token).collect::<Vec<_>>(), vec![5, 6]);
    }

    /// A reply that ends by its end-of-sequence token or by length, with the tokens counted as vLLM counts them, and the text a token held back sent at the end.
    #[test]
    fn the_end_of_a_reply() {
        let t = toy(&["a", "b", "c"]);
        let mut c = Choice::new(Cfg::default());
        for id in [1u32, 2] {
            c.lp(lp(id));
            assert_eq!(c.token(&t, id).unwrap().content, if id == 1 { "a" } else { "b" });
        }
        c.lp(lp(0)); // the end-of-sequence token's own, before the end
        let (d, done) = c.end(&t, Finish::Stop).unwrap();
        assert_eq!((d.content.as_str(), done.finish_reason, done.tokens), ("", "stop", 3));
        assert_eq!(d.lps.len(), 1);
        assert_eq!(c.content, "ab");
        let mut c = Choice::new(Cfg::default());
        c.token(&t, 3).unwrap();
        let (_, done) = c.end(&t, Finish::Length).unwrap();
        assert_eq!((done.finish_reason, done.tokens), ("length", 1));
        // a reply that is stopped by the client ends as a stop
        let mut c = Choice::new(Cfg::default());
        assert_eq!(c.end(&t, Finish::Stopped).unwrap().1.finish_reason, "stop");
    }

    /// A chat reply is split into thinking and answer; with the thinking hidden it is counted and kept apart, not sent.
    #[test]
    fn an_answer_the_prompt_began() {
        let t = toy(&["\"a\":", " 1}"]);
        let mut c = Choice::new(Cfg { chat: true, lead: "{".to_string(), ..Cfg::default() });
        let mut shown = c.token(&t, 1).unwrap().content;
        shown += &c.token(&t, 2).unwrap().content;
        let (last, done) = c.end(&t, Finish::Stop).unwrap();
        assert_eq!(shown + &last.content, "{\"a\": 1}");
        assert_eq!(done.content.as_deref(), Some("{\"a\": 1}"));
        let mut c = Choice::new(Cfg { chat: true, lead: "{".to_string(), ..Cfg::default() });
        let (_, done) = c.end(&t, Finish::Stop).unwrap();
        assert_eq!(done.content.as_deref(), Some("{"));
    }

    #[test]
    fn thinking_apart_from_the_answer() {
        let t = toy(&["<think>", "\n", "Hmm.", "</think>", "Yes."]);
        for hide in [false, true] {
            let mut c = Choice::new(Cfg { chat: true, hide_reasoning: hide, ..Cfg::default() });
            let (mut sent_r, mut sent_c) = (String::new(), String::new());
            for id in [1u32, 2, 3, 2, 4, 2, 5] {
                let d = c.token(&t, id).unwrap();
                sent_r.push_str(&d.reasoning);
                sent_c.push_str(&d.content);
            }
            let (d, done) = c.end(&t, Finish::Stop).unwrap();
            sent_r.push_str(&d.reasoning);
            sent_c.push_str(&d.content);
            assert_eq!(sent_c, "Yes.");
            assert_eq!(sent_r, if hide { "" } else { "Hmm." });
            assert_eq!((c.reasoning.as_str(), c.content.as_str(), done.tokens), ("Hmm.", "Yes.", 8));
        }
        // a prompt that opened the thinking itself
        let mut c = Choice::new(Cfg { chat: true, in_thinking: true, ..Cfg::default() });
        for id in [3u32, 2, 4, 5] {
            c.token(&t, id).unwrap();
        }
        c.end(&t, Finish::Stop).unwrap();
        assert_eq!((c.reasoning.as_str(), c.content.as_str()), ("Hmm.", "Yes."));
    }

    /// A Kimi-K3 reply, token by token (its tags' tokens as Kimi's tokenizer has them: <|open|>, <|sep|>, <|close|> apart from the tag's words): the think
    /// channel the reasoning, the response the answer, its call a tool call, whether the calls are streamed or read at the end.
    #[test]
    fn a_kimi_reply_reads_as_reasoning_answer_and_calls() {
        let words = [
            "Look",
            " it",
            " up.",
            "<|close|>",
            "think",
            "<|sep|>",
            "<|open|>",
            "response",
            "Checking.",
            "tools",
            "call tool=\"get_weather\" index=\"1\"",
            "argument key=\"city\" type=\"string\"",
            "Paris",
            "argument",
            "call",
            "message",
        ];
        let t = toy(&words);
        let id = |w: &str| words.iter().position(|x| *x == w).unwrap() as u32 + 1;
        let script = [
            "Look",
            " it",
            " up.",
            "<|close|>",
            "think",
            "<|sep|>",
            "<|open|>",
            "response",
            "<|sep|>",
            "Checking.",
            "<|close|>",
            "response",
            "<|sep|>",
            "<|open|>",
            "tools",
            "<|sep|>",
            "<|open|>",
            "call tool=\"get_weather\" index=\"1\"",
            "<|sep|>",
            "<|open|>",
            "argument key=\"city\" type=\"string\"",
            "<|sep|>",
            "Paris",
            "<|close|>",
            "argument",
            "<|sep|>",
            "<|close|>",
            "call",
            "<|sep|>",
            "<|close|>",
            "tools",
            "<|sep|>",
            "<|close|>",
            "message",
            "<|sep|>",
        ];
        for stream in [false, true] {
            let tools = ToolsCfg { choice: ToolChoice::Auto, parallel: true, format: Format::Hermes, list: Vec::new(), prefix: String::new(), stream };
            let mut c = Choice::new(Cfg { chat: true, tools: Some(tools), reader: Some(crate::reader::ReaderKind::Kimi { in_think: true }), ..Cfg::default() });
            let mut streamed = Vec::new();
            for w in script {
                streamed.extend(c.token(&t, id(w)).unwrap().calls);
            }
            let (d, done) = c.end(&t, Finish::Stop).unwrap();
            streamed.extend(d.calls);
            assert_eq!((c.reasoning.as_str(), done.finish_reason), ("Look it up.", "tool_calls"), "stream {stream}");
            assert_eq!(done.content.as_deref().unwrap_or("").trim(), "Checking.", "stream {stream}");
            let want = Call { name: "get_weather".into(), arguments: "{\"city\": \"Paris\"}".into() };
            if stream {
                assert!(streamed.iter().any(|e| matches!(e, ToolEv::Start { name, .. } if name == "get_weather")), "{streamed:?}");
            } else {
                assert_eq!(done.calls, vec![want]);
            }
        }
        // (a forced call: the prompt ended with its opening, the model writes the arguments on)
        let rest = &script[script.iter().position(|w| *w == "argument key=\"city\" type=\"string\"").unwrap() - 1..];
        for stream in [false, true] {
            let prefix = crate::xtml::forced(true, Some("get_weather"));
            let tools = ToolsCfg { choice: ToolChoice::Named("get_weather".into()), parallel: false, format: Format::Hermes, list: Vec::new(), prefix, stream };
            let mut c =
                Choice::new(Cfg { chat: true, tools: Some(tools), reader: Some(crate::reader::ReaderKind::Kimi { in_think: false }), ..Cfg::default() });
            let mut streamed = Vec::new();
            for w in rest {
                streamed.extend(c.token(&t, id(w)).unwrap().calls);
            }
            let (d, done) = c.end(&t, Finish::Stop).unwrap();
            streamed.extend(d.calls);
            if stream {
                assert!(streamed.iter().any(|e| matches!(e, ToolEv::Start { name, .. } if name == "get_weather")), "{streamed:?}");
            } else {
                assert_eq!(done.calls, vec![Call { name: "get_weather".into(), arguments: "{\"city\": \"Paris\"}".into() }], "{:?}", done.content);
            }
        }
    }
}
