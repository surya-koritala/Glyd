//! A conversation with the model: the messages through the chat template to token ids, a reply streamed as text (decoded a few tokens at a time, never half a character), cut at a stop string.

use crate::chat::{Chat, Message, Options};
use crate::ffi::{err, Result};
use crate::sampler::Params;
use crate::sched::{Finish, Reply};
use crate::tok::Tok;

/// What a request asks for besides its messages.
#[derive(Clone, Debug)]
pub struct Request {
    pub params: Params,
    pub max_new: usize,
    pub stop: Vec<String>,
    /// Some(false) closes the thinking of a template that has it (Qwen3's `enable_thinking`).
    pub thinking: Option<bool>,
    /// The other variables the request gives the chat template (`chat_template_kwargs`: `reasoning_effort`, `date_string` ...), as they were given.
    pub template_kwargs: Vec<(String, glyd_gpu::json::Value)>,
    /// Go on past the end-of-sequence token (vLLM's `ignore_eos`, for a run of a set length).
    pub ignore_eos: bool,
    /// Tokens besides the end tokens that end the reply (vLLM's `stop_token_ids`), kept in the text or not (`include_stop_str_in_output`), and whether the text leaves out the special tokens (`skip_special_tokens`: yes).
    pub stop_ids: Vec<u32>,
    pub include_stop: bool,
    pub skip_special: bool,
}

impl Request {
    /// What the chat template is given besides the messages: the generation prompt, `enable_thinking` where asked, the request's other variables.
    pub fn template_options(&self) -> Options {
        let mut o = Options::new(true).thinking(self.thinking);
        o.kwargs.extend(self.template_kwargs.iter().cloned());
        o
    }
}

impl Default for Request {
    fn default() -> Request {
        Request {
            params: Params::default(),
            max_new: 2048,
            stop: Vec::new(),
            thinking: None,
            template_kwargs: Vec::new(),
            ignore_eos: false,
            stop_ids: Vec::new(),
            include_stop: false,
            skip_special: true,
        }
    }
}

/// How a reply went: its text, why it ended, and the tokens it took and made.
#[derive(Clone, Debug)]
pub struct Outcome {
    pub text: String,
    pub finish: Finish,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

/// Where Kimi-K3's generation prompt ends inside the think channel (`xtml::Xtml` then starts the reply with `<think>`).
pub const XTML_THINK: &str = "<|open|>think<|sep|>";

/// Text from tokens as they come: the part of the decoding that is complete (it does not end in half a character) and new since the last.
pub struct Stream {
    ids: Vec<u32>,
    prefix: usize,
    read: usize,
    /// The text leaves out the special tokens (vLLM's `skip_special_tokens`).
    skip: bool,
    /// A family's markup read as the other models' replies are (`reader::Reader`: Kimi-K3's tags, Gemma 4's), and what a forced call's opening read as.
    reader: Option<Box<dyn crate::reader::Reader>>,
    primed: String,
}

impl Default for Stream {
    fn default() -> Stream {
        Stream::new()
    }
}

impl Stream {
    pub fn new() -> Stream {
        Stream { ids: Vec::new(), prefix: 0, read: 0, skip: false, reader: None, primed: String::new() }
    }

    /// The reply read through the reader `k` names, where one is.
    pub fn with_reader(mut self, k: Option<crate::reader::ReaderKind>) -> Stream {
        self.reader = k.map(crate::reader::ReaderKind::build);
        self
    }

    /// The reader told what the prompt ends with (a forced call's opening): the reply is read on from there; what the opening reads as is `primed`.
    pub fn prime(&mut self, opening: &str) {
        if let Some(r) = &mut self.reader {
            self.primed = r.prime(opening);
        }
    }

    /// What a forced call's opening read as (the tool parser starts from it).
    pub fn primed(&self) -> &str {
        &self.primed
    }

    /// A stream whose text leaves out the special tokens.
    pub fn skipping() -> Stream {
        Stream { skip: true, ..Stream::new() }
    }

    /// The text `token` completes, if any (empty where it ends in the middle of a character).
    pub fn push(&mut self, tok: &Tok, token: u32) -> Result<String> {
        self.ids.push(token);
        let before = tok.decode_with(&self.ids[self.prefix..self.read], self.skip)?;
        let now = tok.decode_with(&self.ids[self.prefix..], self.skip)?;
        if now.len() > before.len() && !now.ends_with('\u{FFFD}') {
            let out = now[before.len()..].to_string();
            self.prefix = self.read;
            self.read = self.ids.len();
            Ok(match &mut self.reader {
                Some(r) => r.feed(&out),
                None => out,
            })
        } else {
            Ok(String::new())
        }
    }

    /// What is left where the reply ended (a character that never finished comes out as it decodes).
    pub fn finish(&mut self, tok: &Tok) -> Result<String> {
        let before = tok.decode_with(&self.ids[self.prefix..self.read], self.skip)?;
        let now = tok.decode_with(&self.ids[self.prefix..], self.skip)?;
        let tail = now.get(before.len()..).unwrap_or("").to_string();
        Ok(match &mut self.reader {
            Some(r) => r.feed(&tail) + &r.finish(),
            None => tail,
        })
    }
}

/// Text held back where it might begin a stop string, and cut where one is found.
pub struct Stops {
    stops: Vec<String>,
    held: String,
    pub hit: bool,
    /// The stop string that was found (Anthropic's `stop_sequence`).
    pub matched: Option<String>,
    /// The stop string that was found is part of the text (vLLM's `include_stop_str_in_output`).
    include: bool,
}

impl Stops {
    pub fn new(stops: &[String]) -> Stops {
        Stops { stops: stops.iter().filter(|s| !s.is_empty()).cloned().collect(), held: String::new(), hit: false, matched: None, include: false }
    }

    /// Keeps the stop string that ends the text in it.
    pub fn including(mut self, yes: bool) -> Stops {
        self.include = yes;
        self
    }

    /// `text` in, the text that is safe to show out (and `hit` set where a stop string is complete: what came before it is shown, the string and the rest are not).
    pub fn feed(&mut self, text: &str) -> String {
        if self.hit {
            return String::new();
        }
        self.held.push_str(text);
        let cut = self.stops.iter().filter_map(|s| self.held.find(s.as_str()).map(|i| (i, s.len(), s))).min_by_key(|&(i, ..)| i);
        if let Some((i, len, which)) = cut {
            self.hit = true;
            self.matched = Some(which.clone());
            let out = self.held[..if self.include { i + len } else { i }].to_string();
            self.held.clear();
            return out;
        }
        // keep the end that could still become a stop string
        let mut keep = 0;
        for s in &self.stops {
            for k in (1..s.len()).rev() {
                if s.is_char_boundary(k) && self.held.ends_with(&s[..k]) {
                    keep = keep.max(k);
                    break;
                }
            }
        }
        let at = self.held.len() - keep;
        let at = (0..=at).rev().find(|&i| self.held.is_char_boundary(i)).unwrap_or(0);
        let out = self.held[..at].to_string();
        self.held.drain(..at);
        out
    }

    /// What is held when the reply ends without a stop string.
    pub fn flush(&mut self) -> String {
        if self.hit {
            return String::new();
        }
        std::mem::take(&mut self.held)
    }
}

/// Where a piece of a reply belongs: the model's thinking, or its answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    Reasoning,
    Content,
}

enum Phase {
    /// The reply has not shown yet whether it starts with `<think>`.
    Start,
    Thinking,
    /// Just after `</think>`: the template's newlines before the answer are not text.
    After,
    Answer,
}

/// A reply's text split into thinking and answer where the model writes `<think>` ... `</think>` first (Qwen3's way): what a reasoning parser does on a server. Text that may still be the
/// start of a tag is held back until it is not.
pub struct Splitter {
    phase: Phase,
    buf: String,
    fresh: bool,
}

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

impl Splitter {
    /// `in_thinking`: the prompt ended inside the thinking (a template that opens `<think>` itself).
    pub fn new(in_thinking: bool) -> Splitter {
        Splitter { phase: if in_thinking { Phase::Thinking } else { Phase::Start }, buf: String::new(), fresh: in_thinking }
    }

    fn held_tag(buf: &str, tag: &str) -> usize {
        (1..tag.len()).rev().find(|&k| buf.ends_with(&tag[..k])).unwrap_or(0)
    }

    pub fn feed(&mut self, text: &str) -> Vec<(Part, String)> {
        self.buf.push_str(text);
        let mut out = Vec::new();
        loop {
            match self.phase {
                Phase::Start => {
                    let t = self.buf.trim_start();
                    if t.is_empty() {
                        break;
                    }
                    if let Some(rest) = t.strip_prefix(OPEN) {
                        self.buf = rest.to_string();
                        self.phase = Phase::Thinking;
                        self.fresh = true;
                    } else if OPEN.starts_with(t) {
                        break;
                    } else {
                        self.phase = Phase::Answer;
                    }
                }
                Phase::Thinking => {
                    if self.fresh {
                        let t = self.buf.trim_start().to_string();
                        self.buf = t;
                        if self.buf.is_empty() {
                            break;
                        }
                        self.fresh = false;
                    }
                    if let Some(i) = self.buf.find(CLOSE) {
                        let r = self.buf[..i].trim_end_matches(['\n', ' ']).to_string();
                        if !r.is_empty() {
                            out.push((Part::Reasoning, r));
                        }
                        self.buf = self.buf[i + CLOSE.len()..].to_string();
                        self.phase = Phase::After;
                        continue;
                    }
                    // what may still be the closing tag, and the whitespace before it (which is not text if it is), is held
                    let body = self.buf.trim_end_matches(['\n', ' ', '\t']);
                    let tag_at = body.len() - Self::held_tag(body, CLOSE);
                    let cut = body[..tag_at].trim_end_matches(['\n', ' ', '\t']).len();
                    if cut > 0 {
                        out.push((Part::Reasoning, self.buf[..cut].to_string()));
                        self.buf.drain(..cut);
                    }
                    break;
                }
                Phase::After => {
                    let t = self.buf.trim_start().to_string();
                    if t.is_empty() {
                        self.buf.clear();
                        break;
                    }
                    self.buf = t;
                    self.phase = Phase::Answer;
                }
                Phase::Answer => {
                    if !self.buf.is_empty() {
                        out.push((Part::Content, std::mem::take(&mut self.buf)));
                    }
                    break;
                }
            }
        }
        out
    }

    /// What is held where the reply ends.
    pub fn finish(&mut self) -> Vec<(Part, String)> {
        let rest = std::mem::take(&mut self.buf);
        match self.phase {
            Phase::Start if !rest.is_empty() => vec![(Part::Content, rest)],
            Phase::Thinking if !rest.is_empty() => vec![(Part::Reasoning, rest.trim_end().to_string())],
            _ => Vec::new(),
        }
    }

    /// A whole reply split at once: (thinking, answer).
    pub fn split(text: &str, in_thinking: bool) -> (String, String) {
        let mut s = Splitter::new(in_thinking);
        let (mut think, mut answer) = (String::new(), String::new());
        for (part, t) in s.feed(text).into_iter().chain(s.finish()) {
            match part {
                Part::Reasoning => think.push_str(&t),
                Part::Content => answer.push_str(&t),
            }
        }
        (think, answer)
    }
}

/// What a conversation takes its replies from: the engine itself (one reply at a time) or the scheduler's sequences (`server::Remote`).
pub trait Gen {
    /// The tokens a prompt and its reply can take together.
    fn capacity(&self) -> usize;
    /// A reply to `prompt`: its tokens as they are chosen (`each` says whether to go on), with how it ended.
    fn reply(&mut self, prompt: &[u32], o: &Reply, each: &mut dyn FnMut(u32) -> bool) -> Result<(Vec<u32>, Finish)>;
    /// What the model holds on the GPU by what it is for (None: this engine does not say).
    fn held(&self) -> Option<crate::dev::Held> {
        None
    }
    /// The engine as a scheduler's step for up to `users` sequences at once, with the sequences it made room for (a model that runs one sequence at a time: one); None where
    /// the engine is not one (`serve` refuses it: every model it serves is a step).
    fn as_step(&mut self, _users: usize) -> Result<Option<(usize, &mut dyn crate::sched::ModelStep)>> {
        Ok(None)
    }
    /// The part (0 to 1) of the GPU's free memory `as_step` makes its keys and values and its sequences' states of (1: all of it; `serve --with` gives the first model half).
    fn share_memory(&mut self, _part: f64) {}
    /// The images of the next `reply`'s prompt, in order (each one's place in the prompt as many image tokens as its rows): an engine whose model has a vision
    /// tower reads them; any other refuses them.
    fn see(&mut self, images: Vec<std::sync::Arc<crate::vision::Image>>) -> Result<()> {
        if images.is_empty() {
            Ok(())
        } else {
            err("this model reads no images")
        }
    }
    /// A reply's speed is this engine's own, for the planner to keep (`plan::observe`): not where the replies are a scheduler's sequences (`server::Remote`), which
    /// are timed where they run.
    fn own_speed(&self) -> bool {
        true
    }
}

/// The ids a chat reply starts from, and whether the prompt ends inside the thinking (a template that opens `<think>` itself): the template's text read as ids, or, where the model has no template, `role: text` lines.
pub fn chat_prompt(chat: Option<&Chat>, tok: &Tok, messages: &[Message], r: &Request) -> Result<(Vec<u32>, bool)> {
    match chat {
        Some(c) => {
            let text = c.render_with(messages, &r.template_options())?;
            let in_thinking = text.trim_end().ends_with(OPEN);
            Ok((tok.encode_with(&text, false)?, in_thinking))
        }
        None => {
            let mut t = messages.iter().map(|m| format!("{}: {}", m.role, m.content)).collect::<Vec<_>>().join("\n");
            t.push_str("\nassistant:");
            Ok((tok.encode(&t)?, false))
        }
    }
}

pub struct Talk<'a> {
    pub e: &'a mut dyn Gen,
    pub tok: &'a Tok,
    pub chat: Option<&'a Chat>,
}

impl Talk<'_> {
    /// Whether the generation prompt of `messages` opens the thinking itself (a template that ends its prompt with `<think>`): the reply starts inside it.
    pub fn opens_thinking(&self, messages: &[Message], r: &Request) -> bool {
        self.chat.and_then(|c| c.render_with(messages, &r.template_options()).ok()).is_some_and(|t| t.trim_end().ends_with(OPEN))
    }

    /// The messages as the token ids a reply starts from (the template's, with its generation prompt).
    pub fn prompt_ids(&self, messages: &[Message], r: &Request) -> Result<Vec<u32>> {
        match self.chat {
            Some(c) => self.tok.encode_with(&c.render_with(messages, &r.template_options())?, false),
            None => {
                let mut t = messages.iter().map(|m| format!("{}: {}", m.role, m.content)).collect::<Vec<_>>().join("\n");
                t.push_str("\nassistant:");
                self.tok.encode(&t)
            }
        }
    }

    /// A reply to `messages`, its text given to `on_text` as it comes (which says whether to go on).
    pub fn reply(&mut self, messages: &[Message], r: &Request, on_text: &mut dyn FnMut(&str) -> bool) -> Result<Outcome> {
        let prompt = self.prompt_ids(messages, r)?;
        self.reply_to(&prompt, r, on_text)
    }

    /// A reply to a prompt already in tokens (a plain completion).
    pub fn reply_to(&mut self, prompt: &[u32], r: &Request, on_text: &mut dyn FnMut(&str) -> bool) -> Result<Outcome> {
        if prompt.len() + 1 > self.e.capacity() {
            return err(format!(
                "This prompt has {} tokens, longer than the model's context of {} tokens: maximum context length (a longer one: --context N)",
                prompt.len(),
                self.e.capacity()
            ));
        }
        let budget = self.e.capacity().saturating_sub(prompt.len());
        let opts = Reply { params: r.params.clone(), max_new: r.max_new.min(budget).max(1), ignore_eos: r.ignore_eos, logprobs: None, stop_ids: Vec::new() };
        let tok = self.tok;
        // (a family's markup, Kimi-K3's tags or Gemma 4's, through its reader, from inside the think channel where the prompt ends in it)
        let reader = self.chat.and_then(|c| crate::reader::ReaderKind::of(c, &tok.decode(&prompt[prompt.len().saturating_sub(6)..]).unwrap_or_default()));
        let (mut stream, mut stops) = (Stream::new().with_reader(reader), Stops::new(&r.stop));
        let mut text = String::new();
        let mut failure: Option<crate::ffi::Error> = None;
        let mut each = |t: u32| -> bool {
            let piece = match stream.push(tok, t) {
                Ok(p) => p,
                Err(e) => {
                    failure = Some(e);
                    return false;
                }
            };
            let shown = stops.feed(&piece);
            if !shown.is_empty() {
                text.push_str(&shown);
                if !on_text(&shown) {
                    return false;
                }
            }
            !stops.hit
        };
        let (t0, first) = (std::time::Instant::now(), std::cell::Cell::new(None));
        let (out, mut finish) = self.e.reply(prompt, &opts, &mut |t| {
            if first.get().is_none() {
                first.set(Some(std::time::Instant::now()));
            }
            each(t)
        })?;
        if let Some(f) = first.get().filter(|_| self.e.own_speed()) {
            crate::plan::observe(prompt.len(), (f - t0).as_secs_f64(), out.len(), f.elapsed().as_secs_f64());
        }
        if let Some(f) = failure {
            return Err(f);
        }
        if stops.hit {
            finish = Finish::Stop;
        } else {
            let tail = stream.finish(tok)?;
            let tail = stops.feed(&tail);
            let tail = tail + &stops.flush();
            if !tail.is_empty() {
                text.push_str(&tail);
                on_text(&tail);
            }
        }
        Ok(Outcome { text, finish, prompt_tokens: prompt.len(), completion_tokens: out.len() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thinking_is_split_from_the_answer() {
        assert_eq!(Splitter::split("<think>\nLet me see.\n</think>\n\nIt is 4.", false), ("Let me see.".to_string(), "It is 4.".to_string()));
        // no thinking (the template closed it in the prompt): all answer
        assert_eq!(Splitter::split("It is 4.", false), (String::new(), "It is 4.".to_string()));
        assert_eq!(Splitter::split("  Hello <think> not a tag here", false), (String::new(), "  Hello <think> not a tag here".to_string()));
        // a template that opened the thinking in the prompt
        assert_eq!(Splitter::split("Hmm.\n</think>\n\nYes.", true), ("Hmm.".to_string(), "Yes.".to_string()));
        // the thinking never ends: all of it is thinking
        assert_eq!(Splitter::split("<think>\nstill going", false), ("still going".to_string(), String::new()));
        // fed a few characters at a time, the same split, and no tag leaks into the text
        let text = "<think>\nA < B, so <b>.\n</think>\n\nDone <think>.";
        let mut s = Splitter::new(false);
        let (mut think, mut answer) = (String::new(), String::new());
        for c in text.chars() {
            for (p, t) in s.feed(&c.to_string()) {
                match p {
                    Part::Reasoning => think.push_str(&t),
                    Part::Content => answer.push_str(&t),
                }
            }
        }
        for (p, t) in s.finish() {
            match p {
                Part::Reasoning => think.push_str(&t),
                Part::Content => answer.push_str(&t),
            }
        }
        assert_eq!((think.as_str(), answer.as_str()), ("A < B, so <b>.", "Done <think>."));
        // a cut tag at the very end is text
        assert_eq!(Splitter::split("ab<th", false), (String::new(), "ab<th".to_string()));
        assert_eq!(Splitter::split("<th", false), (String::new(), "<th".to_string()));
    }

    #[test]
    fn stop_strings_cut_and_hold_back() {
        let mut s = Stops::new(&["END".to_string(), "\n\nUser:".to_string()]);
        assert_eq!(s.feed("The answer is 4. E"), "The answer is 4. ");
        assert_eq!(s.feed("N"), "");
        assert_eq!(s.feed("X"), "ENX");
        assert!(!s.hit);
        assert_eq!(s.feed(" fine\n\nUs"), " fine");
        assert_eq!(s.feed("er: hi"), "");
        assert!(s.hit);
        assert_eq!(s.flush(), "");
        let mut t = Stops::new(&["</s>".to_string()]);
        assert_eq!(t.feed("a</"), "a");
        assert_eq!(t.flush(), "</");
        // a multi-byte character is never split
        let mut u = Stops::new(&["xyz".to_string()]);
        assert_eq!(u.feed("é\u{4f60}x"), "é\u{4f60}");
        assert_eq!(u.feed("yz!"), "");
        assert!(u.hit);
        // the stop string kept in the text (vLLM's include_stop_str_in_output): the text up to the end of the string, the rest cut
        let mut k = Stops::new(&["END".to_string()]).including(true);
        assert_eq!(k.feed("a EN"), "a ");
        assert_eq!(k.feed("D and more"), "END");
        assert!(k.hit);
    }
}
