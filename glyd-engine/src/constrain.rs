//! Constrained decoding: a grammar (a JSON schema, a regular expression, a Lark grammar, a choice among strings) that the reply must follow, enforced at sampling time by llguidance's token masks.
//!
//! One hook serves every family: `Params::constraint` names the grammar, `Sampler::new` takes a matcher of it, and the sampler takes the tokens the matcher does not allow out of each row of logits
//! before anything else looks at it (sampler.rs). The tokens are the tokenizer's, as bytes (`Tok::grammars`); the rows may be wider than the vocabulary (the padding is never allowed).

use llguidance::api::TopLevelGrammar;
use llguidance::{Matcher, ParserFactory};
use std::sync::{Arc, Mutex};
use toktrie::{SimpleVob, TokEnv, TokRxInfo, TokTrie, TokenizerEnv};

/// What a reply must be.
#[derive(Clone, Debug, PartialEq)]
pub enum Spec {
    /// Any JSON object (`response_format: json_object`).
    JsonObject,
    /// JSON that matches this schema, written as compact JSON text (`response_format: json_schema`, `guided_json`).
    JsonSchema(String),
    /// Text that matches this regular expression, whole (`guided_regex`).
    Regex(String),
    /// A Lark grammar, or llguidance's own grammar list in JSON (`guided_grammar`).
    Grammar(String),
    /// One of these strings (`guided_choice`).
    Choice(Vec<String>),
}

/// How the model's tokenizer tokenizes a text (no special tokens, no token added at the start): where the grammar forces some bytes, llguidance allows the tokens that tokenize them as the model does.
pub type Tokenize = Box<dyn Fn(&str) -> Option<Vec<u32>> + Send + Sync>;

/// The tokenizer as llguidance wants it: the trie of the tokens' bytes, and the tokenizer's own tokenization.
struct Env(TokTrie, Tokenize);

impl TokenizerEnv for Env {
    fn tok_trie(&self) -> &TokTrie {
        &self.0
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<u32> {
        // (bytes that are not text, or that the tokenizer cannot take: the longest tokens that fit)
        std::str::from_utf8(s).ok().and_then(|t| (self.1)(t)).unwrap_or_else(|| self.0.greedy_tokenize(s))
    }
}

/// The names the end token goes by, where the model's directory does not say which it is.
const END_NAMES: &[&str] =
    &["<|im_end|>", "<|end|>", "<|eot_id|>", "<end_of_turn>", "<|endoftext|>", "<|end_of_text|>", "</s>", "<eos>", "<｜end▁of▁sentence｜>"];

/// What compiling a grammar needs of the model: its tokens and its end tokens. Made once, by `Tok::grammars`.
pub struct Grammars {
    factory: ParserFactory,
    ends: Vec<u32>,
    vocab: usize,
    /// Grammars compiled, by what asked for them (a client sends one schema again and again): the last `KEPT`.
    kept: Mutex<Vec<(Spec, Arc<Constraint>)>>,
}

/// How many compiled grammars are kept.
const KEPT: usize = 32;

impl Grammars {
    /// `bytes`: every token's bytes, special tokens marked (tok.rs); `ends`: the model's end tokens, or none (a token named like one is then taken); `vocab`: the width of a row of logits.
    pub fn new(bytes: Vec<Vec<u8>>, ends: &[u32], vocab: usize) -> Result<Grammars, String> {
        Grammars::with_tokenizer(bytes, ends, vocab, Box::new(|_| None))
    }

    /// `new`, with the model's own tokenization of a text (`Tokenize`).
    pub fn with_tokenizer(mut bytes: Vec<Vec<u8>>, ends: &[u32], vocab: usize, tokenize: Tokenize) -> Result<Grammars, String> {
        bytes.resize(vocab.max(bytes.len()), Vec::new());
        let mut ends: Vec<u32> = ends.iter().copied().filter(|&e| (e as usize) < bytes.len()).collect();
        if ends.is_empty() {
            let name = |b: &Vec<u8>| b.first() == Some(&TokTrie::SPECIAL_TOKEN_MARKER) && END_NAMES.iter().any(|n| b[1..] == *n.as_bytes());
            ends.extend(END_NAMES.iter().filter_map(|n| bytes.iter().position(|b| name(b) && b[1..] == *n.as_bytes()).map(|i| i as u32)).take(1));
        }
        let Some(&first) = ends.first() else { return Err("constrained decoding needs the model's end token, and its directory names none".to_string()) };
        let info = TokRxInfo::new(bytes.len() as u32, first);
        let trie = TokTrie::from(&info, &bytes).with_eos_tokens(&ends);
        let env: TokEnv = Arc::new(Env(trie, tokenize));
        let factory = ParserFactory::new_simple(&env).map_err(|e| e.to_string())?;
        Ok(Grammars { factory, ends, vocab: bytes.len(), kept: Mutex::new(Vec::new()) })
    }

    /// The grammar `spec` compiled; an error in words where it cannot be (a schema with a feature llguidance does not do, a regular expression that does not parse).
    pub fn compile(&self, spec: &Spec) -> Result<Arc<Constraint>, String> {
        if let Some((_, c)) = self.kept.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|(s, _)| s == spec) {
            return Ok(c.clone());
        }
        let c = self.compile_new(spec)?;
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if kept.len() >= KEPT {
            kept.remove(0);
        }
        kept.push((spec.clone(), c.clone()));
        Ok(c)
    }

    fn compile_new(&self, spec: &Spec) -> Result<Arc<Constraint>, String> {
        let top = match spec {
            Spec::JsonObject => TopLevelGrammar::from_json_schema(serde_json::json!({"type": "object"})),
            Spec::JsonSchema(s) => TopLevelGrammar::from_json_schema(serde_json::from_str(s).map_err(|e| format!("the JSON schema: {e}"))?),
            Spec::Regex(r) => TopLevelGrammar::from_regex(r),
            Spec::Grammar(g) => TopLevelGrammar::from_lark_or_grammar_list(g).map_err(|e| e.to_string())?,
            Spec::Choice(c) => {
                if c.is_empty() {
                    return Err("guided_choice: no choices".to_string());
                }
                TopLevelGrammar::from_lark(format!("start: {}", c.iter().map(|s| lark_str(s)).collect::<Vec<_>>().join(" | ")))
            }
        };
        let mut m = Matcher::new(self.factory.create_parser(top));
        if let Some(e) = m.get_error() {
            return Err(e);
        }
        // (the grammar's first mask is made here: a grammar that has no token to begin with is no grammar for this model)
        if let Err(e) = m.compute_mask() {
            return Err(e.to_string());
        }
        m.reset().map_err(|e| e.to_string())?;
        Ok(Arc::new(Constraint { matcher: Mutex::new(m), ends: self.ends.clone(), vocab: self.vocab }))
    }
}

/// A compiled grammar: what a request holds, and what each of its sequences takes a matcher of.
pub struct Constraint {
    matcher: Mutex<Matcher>,
    ends: Vec<u32>,
    vocab: usize,
}

impl std::fmt::Debug for Constraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Constraint(vocab {})", self.vocab)
    }
}

impl Constraint {
    /// A matcher at the start of the grammar, independent of the others; a worker makes it, and the first mask, at once (the prompt's prefill runs meanwhile).
    pub fn start(self: &Arc<Self>) -> Gram {
        let mut g = Gram { m: None, cell: None, current: None, ends: self.ends.clone(), vocab: self.vocab, masks: 0 };
        g.hand_over(Src::Start(self.clone()), None);
        g
    }
}

/// Where a worker's matcher comes from: a sequence's, or a new one of a grammar.
enum Src {
    Matcher(Matcher),
    Start(Arc<Constraint>),
}

/// A matcher with the token it is to take, where its next mask goes, and what the mask is made of.
struct Job {
    m: Src,
    tok: Option<u32>,
    out: Arc<MaskCell>,
    ends: Vec<u32>,
    vocab: usize,
}

/// What a worker leaves for a sequence: the matcher back and the mask of the tokens that may follow it. `mask` is None, and `m` too, where the worker died.
struct CellState {
    done: bool,
    m: Option<Matcher>,
    mask: Option<Arc<SimpleVob>>,
}

/// Where a worker leaves what it made, for the sequence (its matcher, `Gram::settle`) and for the engine, which may put the mask on the GPU before the step's top-k (`MaskHandle::words`) while the matcher stays.
pub struct MaskCell {
    st: Mutex<CellState>,
    cv: std::sync::Condvar,
}

impl MaskCell {
    fn new() -> Arc<MaskCell> {
        Arc::new(MaskCell { st: Mutex::new(CellState { done: false, m: None, mask: None }), cv: std::sync::Condvar::new() })
    }

    fn finish(&self, m: Option<Matcher>, mask: Option<Arc<SimpleVob>>) {
        let mut st = self.st.lock().unwrap_or_else(|e| e.into_inner());
        st.m = m;
        st.mask = mask;
        st.done = true;
        self.cv.notify_all();
    }

    fn wait(&self) -> std::sync::MutexGuard<'_, CellState> {
        let mut st = self.st.lock().unwrap_or_else(|e| e.into_inner());
        while !st.done {
            st = self.cv.wait(st).unwrap_or_else(|e| e.into_inner());
        }
        st
    }
}

/// A job's cell, finished empty if the worker dies with the job in hand (a waiting engine or sequence is then let go).
struct Leave(Arc<MaskCell>, bool);

impl Drop for Leave {
    fn drop(&mut self) {
        if !self.1 {
            self.0.finish(None, None);
        }
    }
}

/// The mask of the next step as an engine takes it: the tokens a sequence's grammar allows, for the GPU to put on the row before its top-k (`words`: a word of 32 tokens' bits, as the kernel reads them).
#[derive(Clone)]
pub enum MaskHandle {
    Pending(Arc<MaskCell>),
    Ready(Arc<SimpleVob>),
}

impl MaskHandle {
    /// The mask, waiting for the worker that makes it (it works beside the step: the wait is for what the GPU has not done yet); None where the worker died.
    pub fn words(&self) -> Option<Arc<SimpleVob>> {
        match self {
            MaskHandle::Ready(m) => Some(m.clone()),
            MaskHandle::Pending(c) => c.wait().mask.clone(),
        }
    }
}

/// The mask a grammar gives to a row, with the end tokens where the grammar is whole (and they alone where nothing more can follow, or where the matcher cannot say: the reply then ends and does not go on past the grammar).
fn finish_mask(made: Option<SimpleVob>, ends: &[u32], vocab: usize) -> SimpleVob {
    let mut mask = made.clone().unwrap_or_else(|| SimpleVob::alloc(vocab));
    if made.is_none() || ends.iter().any(|&e| mask.is_allowed(e)) {
        for &e in ends {
            if (e as usize) < mask.len() {
                mask.allow_token(e);
            }
        }
    }
    mask
}

/// How many threads make masks. A mask costs some 50 us and now and then a millisecond or more: they are made beside the GPU's step, not after it, so that a slow one does not hold the others up.
const WORKERS: usize = 4;

fn workers() -> &'static std::sync::mpsc::Sender<Job> {
    static POOL: std::sync::OnceLock<std::sync::mpsc::Sender<Job>> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<Job>();
        let rx = Arc::new(Mutex::new(rx));
        for i in 0..WORKERS {
            let rx = rx.clone();
            let _ = std::thread::Builder::new().name(format!("glyd-mask-{i}")).spawn(move || loop {
                let job = rx.lock().unwrap_or_else(|e| e.into_inner()).recv();
                let Ok(Job { m, tok, out, ends, vocab }) = job else { return };
                let mut leave = Leave(out, false);
                let mut m = match m {
                    Src::Matcher(m) => m,
                    Src::Start(c) => c.matcher.lock().unwrap_or_else(|e| e.into_inner()).deep_clone(),
                };
                if let Some(t) = tok {
                    let _ = m.consume_token(t);
                }
                let made = if m.is_stopped() { None } else { m.compute_mask().ok() };
                leave.1 = true;
                leave.0.finish(Some(m), Some(Arc::new(finish_mask(made, &ends, vocab))));
            });
        }
        tx
    })
}

/// A sequence's place in its grammar: its matcher, which is with a worker (taking the token chosen and making the next mask) between a token and the next step's draw.
pub struct Gram {
    m: Option<Matcher>,
    /// What the worker is making (or has made) for the step to come.
    cell: Option<Arc<MaskCell>>,
    /// The mask of the step in hand (a draw from the GPU's candidates may find it too narrow and ask again with the whole row's).
    current: Option<Arc<SimpleVob>>,
    ends: Vec<u32>,
    vocab: usize,
    /// Masks made so far.
    masks: u64,
}

impl Gram {
    /// Sends the matcher to a worker with the token it takes (if any); the mask it makes is then in the cell, when `settle` has the matcher back. Where no worker can be had the matcher stays and the mask is made at the draw.
    fn hand_over(&mut self, m: Src, tok: Option<u32>) {
        let out = MaskCell::new();
        if workers().send(Job { m, tok, out: out.clone(), ends: self.ends.clone(), vocab: self.vocab }).is_ok() {
            self.cell = Some(out);
        }
    }

    /// The matcher and the mask back from the worker, if they are out (a worker that died took the matcher: the grammar is then lost, and the reply ends).
    fn settle(&mut self) {
        if let Some(cell) = self.cell.take() {
            let mut st = cell.wait();
            self.m = st.m.take();
            if self.current.is_none() {
                self.current = st.mask.clone();
            }
        }
    }

    /// The mask of the next step for an engine to put on the GPU's row (None: the grammar has none to give, as where it has just taken a token and no worker is making the next).
    pub fn handle(&self) -> Option<MaskHandle> {
        match (&self.current, &self.cell) {
            (Some(m), _) => Some(MaskHandle::Ready(m.clone())),
            (None, Some(c)) => Some(MaskHandle::Pending(c.clone())),
            _ => None,
        }
    }

    /// The tokens that may be chosen next. Where the grammar is complete the end tokens may be chosen (and where nothing more can follow, they alone); where it has gone wrong (the matcher cannot say)
    /// they alone, so that the reply ends and does not go on past the grammar.
    pub fn mask(&mut self) -> SimpleVob {
        if let Some(m) = &self.current {
            return (**m).clone();
        }
        self.masks += 1;
        self.settle();
        let mask = match self.current.clone() {
            Some(m) => m,
            None => {
                let made = self.m.as_mut().and_then(|m| if m.is_stopped() { None } else { m.compute_mask().ok() });
                let m = Arc::new(finish_mask(made, &self.ends, self.vocab));
                self.current = Some(m.clone());
                m
            }
        };
        (*mask).clone()
    }

    /// The tokens that may be chosen next, a flag for each of the row's `vocab`.
    pub fn allowed(&mut self) -> Vec<bool> {
        let mask = self.mask();
        (0..self.vocab).map(|i| mask.is_allowed(i as u32)).collect()
    }

    /// Takes the logits of the tokens the grammar does not allow out of `logits` (negative infinity).
    pub fn apply(&mut self, logits: &mut [f32]) {
        let mask = self.mask();
        let words = mask.as_slice();
        for (i, chunk) in logits.chunks_mut(32).enumerate() {
            let w = words.get(i).copied().unwrap_or(0);
            if w == u32::MAX {
                continue;
            }
            if w == 0 || i * 32 >= mask.len() {
                chunk.fill(f32::NEG_INFINITY);
                continue;
            }
            for (b, x) in chunk.iter_mut().enumerate() {
                if (w >> b) & 1 == 0 || i * 32 + b >= mask.len() {
                    *x = f32::NEG_INFINITY;
                }
            }
        }
    }

    /// The token chosen, which the grammar then holds (an end token is not part of it); the next mask is made while the step that follows runs.
    pub fn consume(&mut self, tok: u32) {
        if self.ends.contains(&tok) {
            return;
        }
        self.settle();
        self.current = None;
        if let Some(m) = self.m.take() {
            self.hand_over(Src::Matcher(m), Some(tok));
        }
    }

    /// Whether the text so far is a whole of the grammar.
    pub fn accepting(&mut self) -> bool {
        self.settle();
        self.m.as_mut().is_some_and(|m| m.is_accepting().unwrap_or(false))
    }

    /// The end tokens.
    pub fn ends(&self) -> &[u32] {
        &self.ends
    }

    /// Masks made.
    pub fn masks(&self) -> u64 {
        self.masks
    }
}

/// A string as a Lark literal.
pub fn lark_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// The JSON schema of a tool's parameters (an object with any properties where the tool names none).
fn parameters_of(tool: &glyd_gpu::json::Value) -> String {
    tool.get("function")
        .and_then(|f| f.get("parameters"))
        .filter(|p| matches!(p, glyd_gpu::json::Value::Object(_)))
        .map_or_else(|| "{\"type\":\"object\"}".to_string(), glyd_gpu::json::to_compact)
}

fn name_of(tool: &glyd_gpu::json::Value) -> Option<&str> {
    tool.get("function").and_then(|f| f.get("name")).and_then(glyd_gpu::json::Value::as_str)
}

/// The Lark grammar of what a forced call's reply holds after the opening the prompt ends with (`tools::forced_prefix`), for every format: hermes (the arguments are JSON of the tool's schema), XML (each parameter written as
/// `<parameter=NAME>`, a string as it is, any other value as JSON), GLM (`<arg_key>NAME</arg_key><arg_value>VALUE</arg_value>`), and the blocks of invokes of DeepSeek-V4 (DSML) and Muse (ATEM: a parameter's `string` flag
/// says whether its value is text or JSON, DSML's). A string is the text up to its closing tag (`text`, a lazy rule: a lexeme of any characters would take the rest of the call too), any other value JSON of the property's
/// schema; the parameters are the tool's, in any order and any number. `named`: the function, where the call is of one (else the reply names it, among `tools`).
pub fn call_grammar(tools: &[glyd_gpu::json::Value], named: Option<&str>, format: crate::tools::Format) -> Option<String> {
    use crate::tools::Format;
    use glyd_gpu::json::Value;
    let pick: Vec<&Value> = match named {
        Some(n) => vec![tools.iter().find(|t| name_of(t) == Some(n))?],
        None => tools.iter().collect(),
    };
    if pick.is_empty() {
        return None;
    }
    // (a tool's parameters: each name, and whether its value is text)
    let props = |t: &Value| -> Vec<(String, bool, String)> {
        match t.get("function").and_then(|f| f.get("parameters")).and_then(|p| p.get("properties")) {
            Some(Value::Object(o)) => {
                o.iter().map(|(k, p)| (k.clone(), p.get("type").and_then(Value::as_str) == Some("string"), glyd_gpu::json::to_compact(p))).collect()
            }
            _ => Vec::new(),
        }
    };
    let mut rules = Vec::new();
    let mut alts = Vec::new();
    match format {
        Format::Hermes => {
            for t in &pick {
                let body = format!("%json{}", parameters_of(t));
                // (named: the prompt holds the name, and the reply begins at the arguments)
                alts.push(match named {
                    Some(_) => body,
                    None => format!("{} {}", lark_str(&format!("{}\", \"arguments\": ", name_of(t)?)), body),
                });
            }
            rules.push(format!("start: ({}) \"}}\" \"\\n\" \"</tool_call>\"", alts.join(" | ")));
        }
        Format::Xml => {
            for (i, t) in pick.iter().enumerate() {
                let mut params = Vec::new();
                for (k, string, schema) in props(t) {
                    let head = lark_str(&format!("<parameter={k}>\n"));
                    // (the text rule takes the end of the parameter with it)
                    params.push(if string { format!("{head} text") } else { format!("{head} %json{schema} {}", lark_str("\n</parameter>\n")) });
                }
                let rule = format!("params{i}");
                rules.push(if params.is_empty() { format!("{rule}: \"\"") } else { format!("{rule}: ({})*", params.join(" | ")) });
                let end = lark_str("</function>\n</tool_call>");
                alts.push(match named {
                    Some(_) => format!("{rule} {end}"),
                    None => format!("{} {rule} {end}", lark_str(&format!("{}>\n", name_of(t)?))),
                });
            }
            rules.insert(0, format!("start: {}", alts.join(" | ")));
            rules.push("text[stop=\"\\n</parameter>\\n\"]: /(?s:.*)/".to_string());
        }
        Format::Glm => {
            for (i, t) in pick.iter().enumerate() {
                let mut params = Vec::new();
                for (k, string, schema) in props(t) {
                    let head = format!("ws {} ws \"<arg_value>\"", lark_str(&format!("<arg_key>{k}</arg_key>")));
                    params.push(if string { format!("{head} text") } else { format!("{head} %json{schema} \"</arg_value>\"") });
                }
                let rule = format!("params{i}");
                rules.push(if params.is_empty() { format!("{rule}: \"\"") } else { format!("{rule}: ({})*", params.join(" | ")) });
                alts.push(match named {
                    Some(_) => rule,
                    None => format!("{} {rule}", lark_str(name_of(t)?)),
                });
            }
            rules.insert(0, format!("start: ({}) ws \"</tool_call>\"", alts.join(" | ")));
            // (white space as literals: a lexeme of it after a `text` that ends at its stop is not read as one)
            rules.push("ws: (\"\\n\" | \" \")*".to_string());
            // (the stop takes the new line the templates write after it: a token may end past the stop, but not a lexeme that is read as ending at it)
            rules.push("text[stop=\"</arg_value>\\n\"]: /(?s:.*)/".to_string());
        }
        Format::Dsml | Format::Atem => {
            let tg = format.tags();
            let dsml = format == Format::Dsml;
            for (i, t) in pick.iter().enumerate() {
                let mut params = Vec::new();
                for (k, string, schema) in props(t) {
                    let flag = |s: bool| if dsml { format!("{k}\" string=\"{s}\">") } else { format!("{k}\">") };
                    params.push(if string {
                        format!("{} text ws", lark_str(&format!("{}{}", tg.param, flag(true))))
                    } else {
                        format!("{} %json{schema} {} ws", lark_str(&format!("{}{}", tg.param, flag(false))), lark_str(tg.param_end))
                    });
                }
                let rule = format!("params{i}");
                rules.push(if params.is_empty() { format!("{rule}: \"\"") } else { format!("{rule}: ({})*", params.join(" | ")) });
                let body = format!("ws {rule} {} ws", lark_str(tg.func_end));
                alts.push(match named {
                    Some(_) => body,
                    None => format!("{} {body}", lark_str(&format!("{}{}{}", tg.func, name_of(t)?, tg.named))),
                });
            }
            // (a named call is the one invoke the prompt opened; else one or more)
            let invokes = if named.is_some() { alts.join(" | ") } else { format!("({})+", alts.join(" | ")) };
            rules.insert(0, format!("start: {invokes} {}", lark_str(tg.close)));
            // (white space as literals: a lexeme of it after a `text` that ends at its stop is not read as one)
            rules.push("ws: (\"\\n\" | \" \")*".to_string());
            rules.push(format!("text[stop={}]: /(?s:.*)/", lark_str(&format!("{}\n", tg.param_end))));
        }
    }
    Some(rules.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A vocabulary of the 256 bytes, some words and punctuation, a space-led word, and `<|end|>` (the end token): every grammar is checked against it on the CPU.
    fn vocab() -> Vec<Vec<u8>> {
        let mut v: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        for w in ["{\"", "\":", "\"}", "true", "false", "null", "name", "age", " the", "ab", "abc", "yes", "no", ", \"", "\n", "</tool_call>", "{}"] {
            v.push(w.as_bytes().to_vec());
        }
        v.push([&[TokTrie::SPECIAL_TOKEN_MARKER][..], b"<|end|>"].concat());
        v
    }

    fn end() -> u32 {
        vocab().len() as u32 - 1
    }

    fn grammars() -> Grammars {
        Grammars::new(vocab(), &[end()], vocab().len() + 5).unwrap()
    }

    fn id(s: &str) -> u32 {
        vocab().iter().position(|b| b == s.as_bytes()).unwrap() as u32
    }

    /// Greedy decoding from logits that prefer `want`'s tokens in order and, where the grammar does not allow one, the first token it allows: the text of what the mask lets through.
    fn walk(c: &Arc<Constraint>, want: &[u32], steps: usize) -> (String, bool) {
        let mut g = c.start();
        let mut out = Vec::new();
        for i in 0..steps {
            // (a model with no mind of its own: a fixed liking for each token, none for white space, and a wish to end where it may)
            let mut logits: Vec<f32> = (0..vocab().len() + 5)
                .map(|t| {
                    let b = vocab().get(t).cloned().unwrap_or_default();
                    if !b.is_empty() && b.iter().all(|c| c.is_ascii_whitespace()) {
                        -5.0
                    } else if b == b"\"" {
                        4.0
                    } else if b == b"}" {
                        3.5
                    } else if b == b"," {
                        3.0
                    } else {
                        ((t * 2654435761usize + i * 40503) % 1000) as f32 / 1000.0
                    }
                })
                .collect();
            logits[end() as usize] = 5.0;
            if let Some(&w) = want.get(i) {
                logits[w as usize] = 10.0;
            }
            // (a token past the vocabulary is the model's biggest wish at each step: the padding is never allowed)
            logits[vocab().len() + 2] = 100.0;
            g.apply(&mut logits);
            let (best, _) = logits.iter().enumerate().fold((0, f32::NEG_INFINITY), |a, (i, &x)| if x > a.1 { (i, x) } else { a });
            assert!(best < vocab().len(), "the padding was chosen");
            if best as u32 == end() {
                return (String::from_utf8_lossy(&out).into_owned(), true);
            }
            g.consume(best as u32);
            out.extend_from_slice(&vocab()[best]);
        }
        (String::from_utf8_lossy(&out).into_owned(), false)
    }

    #[test]
    fn a_schema_gives_json_that_matches_it() {
        let g = grammars();
        let c = g
            .compile(&Spec::JsonSchema(
                r#"{"type":"object","properties":{"age":{"type":"integer"},"ok":{"type":"boolean"}},"required":["age","ok"],"additionalProperties":false}"#
                    .into(),
            ))
            .unwrap();
        // the model wants `yes` and `no`; the mask makes it write the schema's JSON, and then end
        let (text, ended) = walk(&c, &[id("yes"), id("no"), id("yes")], 200);
        assert!(ended, "{text}");
        let v: serde_json::Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}: {text}"));
        assert!(v["age"].is_i64() && v["ok"].is_boolean(), "{text}");
        assert_eq!(v.as_object().unwrap().len(), 2, "{text}");
    }

    #[test]
    fn json_object_is_an_object() {
        let c = grammars().compile(&Spec::JsonObject).unwrap();
        let (text, ended) = walk(&c, &[id("abc"), id("no")], 200);
        assert!(ended, "{text}");
        assert!(serde_json::from_str::<serde_json::Value>(&text).unwrap().is_object(), "{text}");
    }

    #[test]
    fn a_regex_a_choice_and_a_grammar_are_followed() {
        let g = grammars();
        let (text, ended) = walk(&g.compile(&Spec::Regex("[a-c]{3}-[0-9]{2}".into())).unwrap(), &[id("yes")], 50);
        assert!(ended && regex_like(&text), "{text}");
        let (text, ended) = walk(&g.compile(&Spec::Choice(vec!["yes".into(), "maybe \"so\"".into()])).unwrap(), &[id("no"), id("abc")], 50);
        assert!(ended && (text == "yes" || text == "maybe \"so\""), "{text}");
        let (text, ended) = walk(&g.compile(&Spec::Grammar("start: \"a\" (\"b\" | \"c\")+".into())).unwrap(), &[id("yes")], 50);
        assert!(ended && text.starts_with('a') && text.len() >= 2, "{text}");
    }

    fn regex_like(t: &str) -> bool {
        let b = t.as_bytes();
        b.len() == 6 && b[..3].iter().all(|c| (b'a'..=b'c').contains(c)) && b[3] == b'-' && b[4..].iter().all(u8::is_ascii_digit)
    }

    /// The sampler as the engines use it: a request's params with a grammar, and the rows of a model that likes the padding best, at every temperature.
    #[test]
    fn the_sampler_holds_an_answer_to_its_grammar() {
        use crate::sampler::{Params, Sampler};
        let c = grammars().compile(&Spec::Regex("[a-c]{3}-[0-9]{2}".into())).unwrap();
        let width = vocab().len() + 5;
        for (temperature, top_k, top_p) in [(0.0, 0, 1.0), (1.0, 0, 1.0), (0.8, 40, 0.9), (1.5, 5, 1.0)] {
            for seed in 0..8 {
                let p = Params { temperature, top_k, top_p, seed, constraint: Some(c.clone()), ..Params::default() };
                // (a grammar is no greedy step: the GPU's largest logit is not the choice, nor are its largest logits the draw's whole)
                assert!(!p.greedy());
                let mut s = Sampler::new(p, &[]);
                // (the GPU's largest logits serve a draw that looks at no more of the row than they hold)
                assert_eq!(s.top_need().is_some(), temperature <= 0.0 || (1..=64).contains(&top_k));
                let mut out = Vec::new();
                for step in 0..20 {
                    let logits: Vec<f32> =
                        (0..width).map(|t| if t == width - 3 { 50.0 } else { ((t * 7919 + step * 31 + seed as usize) % 97) as f32 / 10.0 }).collect();
                    let t = s.sample(&logits);
                    assert!((t as usize) < vocab().len(), "the padding was chosen");
                    if t == end() {
                        break;
                    }
                    out.extend_from_slice(&vocab()[t as usize]);
                }
                let text = String::from_utf8(out).unwrap();
                assert!(regex_like(&text), "temperature {temperature} seed {seed}: {text:?}");
            }
        }
        // a request with no grammar is as it was
        assert!(Params::default().greedy() && Sampler::new(Params::default(), &[]).top_need().is_none());
    }

    /// The draw under a grammar is the draw from the row with the grammar's tokens taken out, as the sampler makes it with no grammar: a top-k, a top-p, whole or not.
    #[test]
    fn a_draw_under_a_grammar_is_a_draw_from_the_masked_row() {
        use crate::sampler::{Params, Sampler};
        let c = grammars().compile(&Spec::Regex("[a-c]{3}-[0-9]{2}".into())).unwrap();
        let width = vocab().len() + 5;
        let mut masked = vec![0.0f32; width];
        c.start().apply(&mut masked);
        assert!(masked.iter().any(|&x| x == f32::NEG_INFINITY) && masked.iter().any(|&x| x == 0.0));
        for (top_k, top_p, min_p) in [(5, 1.0, 0.0), (0, 0.9, 0.0), (3, 0.8, 0.0), (0, 1.0, 0.1), (100, 1.0, 0.0), (usize::MAX, 1.0, 0.0), (1 << 62, 0.9, 0.0)]
        {
            for seed in 0..20u64 {
                let row: Vec<f32> = (0..width).map(|t| ((t * 7919 + seed as usize * 13) % 101) as f32 / 10.0).collect();
                let p = Params { temperature: 0.9, top_k, top_p, min_p, seed, ..Params::default() };
                let mut by_grammar = Sampler::new(Params { constraint: Some(c.clone()), ..p.clone() }, &[]);
                let mut by_hand = Sampler::new(p, &[]);
                let mut m = row.clone();
                for (x, &ok) in m.iter_mut().zip(&masked) {
                    if ok == f32::NEG_INFINITY {
                        *x = f32::NEG_INFINITY;
                    }
                }
                assert_eq!(by_grammar.sample(&row), by_hand.sample(&m), "top_k {top_k} top_p {top_p} min_p {min_p} seed {seed}");
            }
        }
    }

    /// Forced calls in every format: the grammar takes the call a model writes after the prompt's opening (`tools::forced_prefix`), each byte allowed and the call whole at its end, and the reader of calls (`tools::extract`)
    /// reads from prefix and reply the one call asked for. The tools: one with a string, an integer and an object parameter, and one with none.
    #[test]
    fn forced_calls_of_every_format_are_the_calls_the_reader_reads() {
        use crate::tools::{extract, forced_prefix, Format, ToolChoice};
        let tools = glyd_gpu::json::parse(
            r#"[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"},"opts":{"type":"object","properties":{"hourly":{"type":"boolean"}}}},"required":["city"]}}},
                {"type":"function","function":{"name":"ping","parameters":{"type":"object","properties":{}}}}]"#,
        )
        .unwrap();
        let tools = tools.as_array().unwrap();
        let g =
            Grammars::new((0..=255u8).map(|b| vec![b]).chain([[&[TokTrie::SPECIAL_TOKEN_MARKER][..], b"<|end|>"].concat()]).collect(), &[256], 300).unwrap();
        let (dsml, atem) = (Format::Dsml.tags(), Format::Atem.tags());
        let weather = |city: &str| format!("{{\"city\": \"{city}\", \"days\": 3, \"opts\": {{\"hourly\": true}}}}");
        // (what the model writes after the prefix: for get_weather, named and then required, and for ping, named)
        let cases: Vec<(Format, String, String, String)> = vec![
            (
                Format::Hermes,
                format!("{}}}\n</tool_call>", weather("Zürich")),
                "get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>".into(),
                "{}}\n</tool_call>".into(),
            ),
            (
                Format::Xml,
                "<parameter=city>\nSão Paulo, \"Brazil\"\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=opts>\n{\"hourly\": true}\n</parameter>\n</function>\n</tool_call>".into(),
                "get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>".into(),
                "</function>\n</tool_call>".into(),
            ),
            (
                Format::Glm,
                "\n<arg_key>city</arg_key>\n<arg_value>São Paulo</arg_value>\n<arg_key>days</arg_key>\n<arg_value>3</arg_value>\n<arg_key>opts</arg_key>\n<arg_value>{\"hourly\": true}</arg_value>\n</tool_call>".into(),
                "get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n</tool_call>".into(),
                "</tool_call>".into(),
            ),
            (
                Format::Dsml,
                format!("<｜DSML｜parameter name=\"city\" string=\"true\">São Paulo</｜DSML｜parameter>\n<｜DSML｜parameter name=\"days\" string=\"false\">3</｜DSML｜parameter>\n</｜DSML｜invoke>\n{}", dsml.close),
                format!("{}get_weather\">\n<｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>\n</｜DSML｜invoke>\n{}", dsml.func, dsml.close),
                format!("\n</｜DSML｜invoke>\n{}", dsml.close),
            ),
            (
                Format::Atem,
                format!("<atem:parameter name=\"city\">São Paulo</atem:parameter>\n<atem:parameter name=\"days\">3</atem:parameter>\n</atem:invoke>\n{}", atem.close),
                format!("{}get_weather\">\n<atem:parameter name=\"city\">Paris</atem:parameter>\n</atem:invoke>\n{}", atem.func, atem.close),
                format!("</atem:invoke>\n{}", atem.close),
            ),
        ];
        for (format, named_text, required_text, ping_text) in cases {
            let walk_text = |named: Option<&str>, text: &str| {
                let lark = call_grammar(tools, named, format).unwrap_or_else(|| panic!("{format:?}: no grammar"));
                let c = g.compile(&Spec::Grammar(lark.clone())).unwrap_or_else(|e| panic!("{format:?} {named:?}: {e}\n{lark}"));
                let mut gram = c.start();
                for (i, &b) in text.as_bytes().iter().enumerate() {
                    assert!(gram.allowed()[b as usize], "{format:?} {named:?}: byte {i} of {text:?} ({:?}) is not allowed", &text[..i]);
                    gram.consume(b as u32);
                }
                assert!(gram.accepting(), "{format:?} {named:?}: {text:?} is not whole");
            };
            walk_text(Some("get_weather"), &named_text);
            walk_text(None, &required_text);
            walk_text(Some("ping"), &ping_text);
            // (the reader: the prompt's opening and the reply make the call)
            let read = |choice: &ToolChoice, text: &str| extract(&format!("{}{text}", forced_prefix(choice, format).unwrap()), format, tools);
            let named = ToolChoice::Named("get_weather".into());
            let e = read(&named, &named_text);
            assert!(e.called && e.calls.len() == 1 && e.calls[0].name == "get_weather", "{format:?}: {e:?}");
            assert!(e.calls[0].arguments.contains("São Paulo") || e.calls[0].arguments.contains("Zürich"), "{format:?}: {e:?}");
            let e = read(&ToolChoice::Required, &required_text);
            assert!(e.called && e.calls[0].name == "get_weather" && e.calls[0].arguments.contains("Paris"), "{format:?}: {e:?}");
        }
    }

    /// The mask an engine is handed for the next step is the one the sequence's draw then takes, before and after the draw has it, and the sampler offers it only where its draw looks at the candidates.
    #[test]
    fn the_mask_an_engine_puts_on_the_gpu_is_the_draws() {
        use crate::sampler::{Params, Sampler};
        let c = grammars().compile(&Spec::Choice(vec!["abc".into(), "yes".into()])).unwrap();
        let mut g = c.start();
        let bits = |m: &SimpleVob| (0..m.len()).map(|i| m.is_allowed(i as u32)).collect::<Vec<_>>();
        // (at the start: the worker's, waited for)
        let first = bits(&g.handle().expect("a handle at the start").words().unwrap());
        assert_eq!(first, g.allowed());
        // (after the draw has the mask it is the same one, ready)
        assert_eq!(bits(&g.handle().unwrap().words().unwrap()), first);
        g.consume(id("abc"));
        assert!(matches!(g.handle(), Some(MaskHandle::Pending(_))));
        let after = bits(&g.handle().unwrap().words().unwrap());
        assert_eq!(after, g.allowed());
        assert!(after[end() as usize] && after.iter().filter(|&&x| x).count() == 1);
        // the sampler: a handle where the draw looks at the GPU's candidates (greedy, or a top-k within them), none where it needs the row
        for (temperature, top_k, handle) in [(0.0, 0, true), (0.8, 40, true), (0.8, 0, false), (0.8, 65, false)] {
            let s = Sampler::new(Params { temperature, top_k, constraint: Some(c.clone()), ..Params::default() }, &[]);
            assert_eq!(s.mask_handle().is_some(), handle, "temperature {temperature} top_k {top_k}");
        }
        assert!(Sampler::new(Params { temperature: 0.8, top_k: 40, ..Params::default() }, &[]).mask_handle().is_none());
    }

    /// The GPU's largest logits with the grammar's mask put on them before the cut give the token the whole row gives, wherever they give one: at every place of a grammar, at temperature 0 and with a top-k.
    #[test]
    fn a_draw_from_the_candidates_is_the_draw_from_the_row() {
        use crate::sampler::{Params, Sampler};
        let c = grammars()
            .compile(&Spec::JsonSchema(
                r#"{"type":"object","properties":{"ab":{"type":"integer"},"yes":{"type":"boolean"}},"required":["ab"],"additionalProperties":false}"#.into(),
            ))
            .unwrap();
        let width = vocab().len() + 5;
        let (mut some, mut none) = (0, 0);
        for (temperature, top_k) in [(0.0, 0), (0.9, 1), (0.9, 5), (1.2, 40), (0.9, 64)] {
            for seed in 0..12u64 {
                let p = Params { temperature, top_k, seed, constraint: Some(c.clone()), ..Params::default() };
                let (mut by_top, mut by_row) = (Sampler::new(p.clone(), &[]), Sampler::new(p, &[]));
                // (the grammar is walked as far as it goes, the same tokens in both)
                for step in 0..40 {
                    let row: Vec<f32> = (0..width).map(|t| (((t * 7919 + step * 31 + seed as usize * 13) % 211) as f32 / 8.0).floor() / 4.0 - 3.0).collect();
                    let mut by: Vec<u32> = (0..width as u32).collect();
                    by.sort_by(|&a, &b| row[b as usize].partial_cmp(&row[a as usize]).unwrap().then(a.cmp(&b)));
                    let cands: Vec<(u32, f32)> = by[..64].iter().map(|&i| (i, row[i as usize])).collect();
                    let want = by_row.sample(&row);
                    let got = match by_top.top_need() {
                        Some(_) => by_top.sample_top(&cands, None),
                        None => None,
                    };
                    let tok = match got {
                        Some(t) => {
                            some += 1;
                            assert_eq!(t, want, "temperature {temperature} top_k {top_k} seed {seed} step {step}");
                            t
                        }
                        None => {
                            none += 1;
                            // (the whole row is read: the draw is the same one, the generator where it was)
                            assert_eq!(by_top.sample(&row), want, "temperature {temperature} top_k {top_k} seed {seed} step {step}");
                            want
                        }
                    };
                    if tok == end() {
                        break;
                    }
                }
            }
        }
        assert!(some > 50 && none > 0, "{some} draws from the candidates, {none} from the row");
    }

    /// A least length does not hold a reply that the grammar has ended past its end: the end token is chosen where nothing else may be.
    #[test]
    fn a_least_length_does_not_outlast_the_grammar() {
        use crate::sampler::{Params, Sampler};
        let c = grammars().compile(&Spec::Choice(vec!["ab".into()])).unwrap();
        let width = vocab().len() + 5;
        let mut s = Sampler::new(Params { temperature: 0.0, min_tokens: 10, constraint: Some(c), ..Params::default() }, &[]);
        s.set_suppress(vec![end()]);
        let logits = vec![0.0f32; width];
        let said: Vec<u32> = (0..4).map(|_| s.sample(&logits)).collect();
        let at = said.iter().position(|&t| t == end()).expect("the reply ended");
        let text: Vec<u8> = said[..at].iter().flat_map(|&t| vocab()[t as usize].clone()).collect();
        assert_eq!(text, b"ab");
        assert!(said[at..].iter().all(|&t| t == end()));
    }

    /// A draft token the grammar does not allow is never kept; one it does is.
    #[test]
    fn a_draft_is_kept_only_where_the_grammar_allows_it() {
        use crate::sampler::{Params, Sampler};
        let c = grammars().compile(&Spec::Choice(vec!["abc".into()])).unwrap();
        let width = vocab().len() + 5;
        let mut s = Sampler::new(Params { temperature: 1.0, constraint: Some(c), seed: 3, ..Params::default() }, &[]);
        let logits = vec![0.0f32; width];
        let (t, same) = s.check_draft(&logits, id("yes"));
        assert!(!same && t != id("yes"));
        assert_eq!(vocab()[t as usize][0], b'a');
    }

    #[test]
    fn a_bad_grammar_is_said() {
        let g = grammars();
        assert!(g.compile(&Spec::JsonSchema("{".into())).is_err());
        assert!(g.compile(&Spec::Regex("(".into())).is_err());
        assert!(g.compile(&Spec::Grammar("start: nothing".into())).is_err());
        assert!(g.compile(&Spec::Choice(Vec::new())).is_err());
        assert!(g.compile(&Spec::JsonSchema(r##"{"type":"object","properties":{"a":{"$ref":"#/nowhere"}}}"##.into())).is_err());
    }

    #[test]
    fn the_end_tokens_are_allowed_where_the_grammar_is_whole_and_not_before() {
        let c = grammars().compile(&Spec::Choice(vec!["ab".into()])).unwrap();
        let mut g = c.start();
        let before = g.allowed();
        assert!(!before[end() as usize]);
        assert!(!g.accepting());
        g.consume(id("ab"));
        let after = g.allowed();
        assert!(g.accepting());
        assert_eq!(after.iter().filter(|&&x| x).count(), 1);
        assert!(after[end() as usize]);
    }

    #[test]
    fn matchers_of_one_grammar_are_apart() {
        let c = grammars().compile(&Spec::Choice(vec!["ab".into(), "no".into()])).unwrap();
        let (mut a, mut b) = (c.start(), c.start());
        a.consume(id("ab"));
        assert!(a.accepting() && !b.accepting());
        b.consume(id("no"));
        assert!(b.accepting());
    }

    #[test]
    fn forced_calls_have_grammars_of_their_tools() {
        let tools = glyd_gpu::json::parse(
            r#"[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]}}},
                {"type":"function","function":{"name":"ping","parameters":{"type":"object","properties":{}}}}]"#,
        )
        .unwrap();
        let tools = tools.as_array().unwrap();
        let g = grammars();
        for (format, named) in [
            (crate::tools::Format::Hermes, Some("get_weather")),
            (crate::tools::Format::Hermes, None),
            (crate::tools::Format::Xml, Some("get_weather")),
            (crate::tools::Format::Xml, None),
        ] {
            let lark = call_grammar(tools, named, format).unwrap();
            g.compile(&Spec::Grammar(lark.clone())).unwrap_or_else(|e| panic!("{format:?} {named:?}: {e}\n{lark}"));
        }
        assert!(call_grammar(tools, Some("nope"), crate::tools::Format::Hermes).is_none());
        // a named hermes call, after its opening: the arguments, as the schema says, and the close
        let lark = call_grammar(tools, Some("get_weather"), crate::tools::Format::Hermes).unwrap();
        let c = g.compile(&Spec::Grammar(lark)).unwrap();
        let (text, ended) = walk(&c, &[id("yes")], 300);
        assert!(ended, "{text}");
        let body = text.strip_suffix("}\n</tool_call>").unwrap_or_else(|| panic!("{text}"));
        let v: serde_json::Value = serde_json::from_str(body).unwrap();
        assert!(v["city"].is_string(), "{text}");
    }

    /// The tokenizers of the model families, from the directories `GLYD_TEST_TOKENIZERS` names (separated by `:`; a model's directory with its tokenizer.json or tiktoken.model, and its config.json): each writes a JSON
    /// document, as it tokenizes it, and the grammar of the document's schema allows every token and ends accepting. Skipped where the variable is not set.
    #[test]
    fn every_tokenizer_writes_json_the_grammar_allows() {
        let Some(dirs) = std::env::var_os("GLYD_TEST_TOKENIZERS") else { return };
        let schema = r#"{"type":"object","properties":{"name":{"type":"string"},"tags":{"type":"array","items":{"type":"string"}},"n":{"type":"integer"},"x":{"type":"number"},"ok":{"type":"boolean"},"kind":{"enum":["a","b"]},"nested":{"type":"object","properties":{"k":{"type":"null"}}}},"required":["name"]}"#;
        let doc = r#"{"name": "Zoë \"quoted\" 日本語 😀\n", "tags": ["x", "yz", ""], "n": -12, "x": 3.5e2, "ok": true, "kind": "b", "nested": {"k": null}}"#;
        let compact = doc.replace(": ", ":").replace(", ", ",");
        for dir in std::env::split_paths(&dirs) {
            let tok = crate::tok::Tok::load(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
            let g = tok.grammars(tok.vocab_size() + 64).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
            let c = g.compile(&Spec::JsonSchema(schema.into())).unwrap();
            for text in [doc, compact.as_str()] {
                let ids = tok.encode_with(text, false).unwrap();
                let mut gram = c.start();
                let (mut total, mut worst) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
                for (i, &t) in ids.iter().enumerate() {
                    let began = std::time::Instant::now();
                    let ok = gram.allowed();
                    total += began.elapsed();
                    worst = worst.max(began.elapsed());
                    assert!(
                        ok[t as usize],
                        "{}: token {i} ({:?}, after {:?}) is not allowed",
                        dir.display(),
                        tok.decode(&[t]).unwrap(),
                        tok.decode(&ids[..i]).unwrap()
                    );
                    gram.consume(t);
                }
                // (what the draw waits for, the allowed tokens as a list included, in a build with no optimization: more than the masks)
                eprintln!("{}: {} tokens, a step waits {:?} on average, {:?} at most", dir.display(), ids.len(), total / ids.len() as u32, worst);
                assert!(gram.accepting(), "{}: the document is not whole", dir.display());
                assert!(gram.allowed().iter().filter(|&&x| x).count() <= 8, "{}: after the document only the end (and white space) may follow", dir.display());
            }
            // the calls a forced tool has, as the formats write them (what follows the call's opening in the prompt)
            let tools = glyd_gpu::json::parse(r#"[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"},"opts":{"type":"object","properties":{"hourly":{"type":"boolean"}}}},"required":["city"]}}}]"#).unwrap();
            let tools = tools.as_array().unwrap();
            for (format, named, text) in [
                (crate::tools::Format::Hermes, Some("get_weather"), "{\"city\": \"Zürich\", \"days\": 3}}\n</tool_call>"),
                (crate::tools::Format::Hermes, None, "get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"),
                (
                    crate::tools::Format::Xml,
                    Some("get_weather"),
                    "<parameter=city>\nSão Paulo, \"Brazil\"\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=opts>\n{\"hourly\": true}\n</parameter>\n</function>\n</tool_call>",
                ),
                (crate::tools::Format::Xml, None, "get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"),
                (
                    crate::tools::Format::Glm,
                    Some("get_weather"),
                    "\n<arg_key>city</arg_key>\n<arg_value>Zürich</arg_value>\n<arg_key>days</arg_key>\n<arg_value>3</arg_value>\n</tool_call>",
                ),
                (crate::tools::Format::Glm, None, "get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n</tool_call>"),
                (
                    crate::tools::Format::Dsml,
                    Some("get_weather"),
                    "<｜DSML｜parameter name=\"city\" string=\"true\">Zürich</｜DSML｜parameter>\n<｜DSML｜parameter name=\"days\" string=\"false\">3</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
                ),
                (
                    crate::tools::Format::Atem,
                    None,
                    "<atem:invoke name=\"get_weather\">\n<atem:parameter name=\"city\">Paris</atem:parameter>\n</atem:invoke>\n</atem:function_calls>",
                ),
            ] {
                let lark = call_grammar(tools, named, format).unwrap();
                let c = g.compile(&Spec::Grammar(lark)).unwrap_or_else(|e| panic!("{}: {format:?}: {e}", dir.display()));
                let ids = tok.encode_with(text, false).unwrap();
                let mut gram = c.start();
                for (i, &t) in ids.iter().enumerate() {
                    assert!(gram.allowed()[t as usize], "{}: {format:?} {named:?}: token {i} ({:?}, after {:?}) is not allowed", dir.display(), tok.decode(&[t]).unwrap(), tok.decode(&ids[..i]).unwrap());
                    gram.consume(t);
                }
                assert!(gram.accepting(), "{}: {format:?} {named:?}: the call is not whole", dir.display());
            }
        }
    }
}
