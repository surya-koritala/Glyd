//! Each family's tool calls and reasoning as its own chat template writes them: a conversation with a tool call and a reasoning is rendered through the family's real template, the call and the reasoning are cut out
//! of the rendered text (as the model writes them: a template shows its model what a call looks like), and read back through the family's reader and parser, whole and in pieces of every size. What comes
//! back is what went in. A change to a template or a parser that breaks a family breaks its test here.

use crate::chat::{Chat, Options};
use crate::reader::{Gemma, Muse, Reader};
use crate::tools::{self, Call, Format, ToolEv, ToolStream};
use glyd_gpu::json::{self, Value};

const QWEN38: &str = include_str!("../tests/fixtures/chat_qwen38_tools.jinja");
const GLM53: &str = include_str!("../tests/fixtures/chat_glm53_tools.jinja");
const GEMMA4: &str = include_str!("../tests/fixtures/chat_gemma4_tools.jinja");
const MUSE: &str = include_str!("../tests/fixtures/chat_muse_atem.jinja");

/// The tool the conversation calls: a string, an integer, a number, a boolean, a list and an object of its parameters.
const TOOL: &str = r#"{"type":"function","function":{"name":"plan_trip","description":"Plan a trip","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"},"budget":{"type":"number"},"refundable":{"type":"boolean"},"stops":{"type":"array","items":{"type":"string"}},"prefs":{"type":"object"}},"required":["city"]}}}"#;
const ARGS: &str = r#"{"city":"Zürich","days":3,"budget":1250.5,"refundable":true,"stops":["Bern","Basel"],"prefs":{"pace":"slow","museums":2}}"#;
const THOUGHT: &str = "The user wants a trip plan, so I call the tool.";

fn parse(t: &str) -> Value {
    json::parse(t).unwrap()
}

/// The conversation rendered through `template`: the user's question, the assistant's reasoning and call, the tool's answer; the text the template writes for the assistant's turn (from the reasoning to the call's end).
fn rendered(template: &str, kwargs: Vec<(&str, Value)>) -> String {
    let chat = Chat::from_template(template).unwrap();
    let req = parse(&format!(
        r#"{{"messages":[{{"role":"user","content":"Plan a trip."}},{{"role":"assistant","content":"","reasoning_content":{reason},"tool_calls":[{{"id":"call_1","type":"function","function":{{"name":"plan_trip","arguments":{args}}}}}]}},{{"role":"tool","tool_call_id":"call_1","name":"plan_trip","content":"ok"}}],"tools":[{TOOL}]}}"#,
        reason = crate::server::jstr(THOUGHT),
        args = crate::server::jstr(ARGS)
    ));
    let messages = tools::template_messages(&req).unwrap();
    let mut o = Options::new(false);
    o.tools = req.get("tools").and_then(|t| t.as_array()).map(<[Value]>::to_vec);
    o.kwargs = kwargs.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    chat.render_json(&Value::Array(messages), &o).unwrap()
}

fn tools() -> Vec<Value> {
    vec![parse(TOOL)]
}

/// A JSON value written with its keys in order (a template that sorts a call's keys writes them in another order than the tool's schema does).
fn canon(v: &Value) -> String {
    match v {
        Value::Object(m) => {
            let mut m: Vec<&(String, Value)> = m.iter().collect();
            m.sort_by(|a, b| a.0.cmp(&b.0));
            format!("{{{}}}", m.iter().map(|(k, v)| format!("{k:?}:{}", canon(v))).collect::<Vec<_>>().join(","))
        }
        Value::Array(a) => format!("[{}]", a.iter().map(canon).collect::<Vec<_>>().join(",")),
        other => json::to_compact(other),
    }
}

fn same_json(got: &str, want: &str) {
    assert_eq!(canon(&parse(got)), canon(&parse(want)), "{got}");
}

/// The calls a reply holds, whole.
fn whole(text: &str, format: Format) -> Vec<Call> {
    let e = tools::extract(text, format, &tools());
    assert!(e.called, "no call in {text:?}");
    e.calls
}

/// The same reply through the tool parser as a stream, in pieces of `n` characters: the calls made of what it sends (a call's name when it is complete, its arguments in the pieces they come).
fn streamed(text: &str, format: Format, n: usize) -> Vec<Call> {
    let mut s = ToolStream::new(format, &tools());
    let chars: Vec<char> = text.chars().collect();
    let mut calls: Vec<Call> = Vec::new();
    let mut take = |evs: Vec<ToolEv>| {
        for e in evs {
            match e {
                ToolEv::Start { index, name, .. } => {
                    assert_eq!(index, calls.len());
                    calls.push(Call { name, arguments: String::new() });
                }
                ToolEv::Args { index, args } => calls[index].arguments.push_str(&args),
                ToolEv::Content(_) => {}
            }
        }
    };
    for c in chars.chunks(n) {
        take(s.feed(&c.iter().collect::<String>()));
    }
    take(s.finish());
    calls
}

/// A family's call, read back in every way: whole, and streamed in pieces of several sizes.
fn round_trip(text: &str, format: Format) {
    for c in whole(text, format) {
        assert_eq!(c.name, "plan_trip");
        same_json(&c.arguments, ARGS);
    }
    for n in [1, 3, 8, 1000] {
        let calls = streamed(text, format, n);
        assert_eq!(calls.len(), 1, "pieces of {n}: {calls:?}");
        assert_eq!(calls[0].name, "plan_trip");
        same_json(&calls[0].arguments, ARGS);
    }
}

/// The text of the assistant's turn that follows the user's question, from `from` to the first `to` after it (both kept).
fn cut<'a>(text: &'a str, from: &str, to: &str) -> &'a str {
    let turn = &text[text.find("Plan a trip.").unwrap_or_else(|| panic!("no question in {text}"))..];
    let a = turn.find(from).unwrap_or_else(|| panic!("no {from} in {turn}"));
    let b = a + turn[a..].find(to).unwrap_or_else(|| panic!("no {to} in {turn}")) + to.len();
    &turn[a..b]
}

/// The reasoning, read through `reader` (where there is one) and the reasoning split, in pieces of several sizes.
fn reasoning(text: &str, reader: Option<fn() -> Box<dyn Reader>>, in_thinking: bool) {
    for n in [1, 4, 1000] {
        let mut r = reader.map(|f| f());
        let mut split = crate::conv::Splitter::new(in_thinking);
        let (mut think, mut answer) = (String::new(), String::new());
        let chars: Vec<char> = text.chars().collect();
        let mut feed = |piece: &str, split: &mut crate::conv::Splitter| {
            for (part, t) in split.feed(piece) {
                match part {
                    crate::conv::Part::Reasoning => think.push_str(&t),
                    crate::conv::Part::Content => answer.push_str(&t),
                }
            }
        };
        for c in chars.chunks(n) {
            let piece: String = c.iter().collect();
            let read = match &mut r {
                Some(r) => r.feed(&piece),
                None => piece,
            };
            feed(&read, &mut split);
        }
        let tail = r.as_mut().map(|r| r.finish()).unwrap_or_default();
        feed(&tail, &mut split);
        for (part, t) in split.finish() {
            match part {
                crate::conv::Part::Reasoning => think.push_str(&t),
                crate::conv::Part::Content => answer.push_str(&t),
            }
        }
        assert_eq!(think.trim(), THOUGHT, "pieces of {n}");
        assert!(answer.trim().is_empty(), "{answer:?}");
    }
}

/// Qwen3.8: `<tool_call><function=NAME><parameter=KEY>VALUE</parameter>...</function></tool_call>` (the XML of Qwen3.5 and Nemotron 3, vLLM's `qwen3_coder`), the thinking in `<think>`.
#[test]
fn qwen_3_8() {
    let t = rendered(QWEN38, vec![]);
    assert_eq!(Format::of(&Chat::from_template(QWEN38).unwrap()), Format::Xml);
    round_trip(cut(&t, "<tool_call>", "</tool_call>"), Format::Xml);
    reasoning(cut(&t, "<think>", "</think>"), None, false);
}

/// GLM-5.3-Flash: `<tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>...</tool_call>`, the thinking in `<think>`.
#[test]
fn glm_5_3_flash() {
    let t = rendered(GLM53, vec![]);
    assert_eq!(Format::of(&Chat::from_template(GLM53).unwrap()), Format::Glm);
    round_trip(cut(&t, "<tool_call>", "</tool_call>"), Format::Glm);
    reasoning(cut(&t, "<think>", "</think>"), None, false);
}

/// Gemma 4: `<|tool_call>call:NAME{key:<|"|>text<|"|>,...}<tool_call|>`, the thinking in `<|channel>thought\n...<channel|>`; both through its reader, the call then read as a hermes one.
#[test]
fn gemma_4() {
    let t = rendered(GEMMA4, vec![("preserve_thinking", Value::Bool(true))]);
    let chat = Chat::from_template(GEMMA4).unwrap();
    assert!(chat.is_gemma() && Format::of(&chat) == Format::Hermes);
    let call = cut(&t, "<|tool_call>", "<tool_call|>");
    for n in [1, 3, 8, 1000] {
        let mut g = Gemma::new(false);
        let chars: Vec<char> = call.chars().collect();
        let mut read: String = chars.chunks(n).map(|c| g.feed(&c.iter().collect::<String>())).collect();
        read.push_str(&g.finish());
        let calls = whole(&read, Format::Hermes);
        assert_eq!(calls.len(), 1, "pieces of {n}: {read}");
        assert_eq!(calls[0].name, "plan_trip");
        same_json(&calls[0].arguments, ARGS);
        // (read as a stream: the same call, its arguments in pieces)
        let calls = streamed(&read, Format::Hermes, 5);
        assert_eq!(calls.len(), 1);
        same_json(&calls[0].arguments, ARGS);
    }
    reasoning(cut(&t, "<|channel>thought\n", "<channel|>"), Some(|| Box::new(Gemma::new(false))), false);
}

/// Muse (ATEM): `<|start|>assistant to=self<|message|>REASONING<|eom|>`, then `to=NAME<|message|><atem:function_calls>...</atem:function_calls>`, the answer to `user`; through its reader.
#[test]
fn muse() {
    let t = rendered(MUSE, vec![]);
    let chat = Chat::from_template(MUSE).unwrap();
    assert!(chat.is_muse() && Format::of(&chat) == Format::Atem);
    // the call, as the model writes the message (header and block), read through the reader and parsed as ATEM
    let msg = cut(&t, "<|start|>assistant to=plan_trip<|message|>", "</atem:function_calls>");
    for n in [1, 3, 8, 1000] {
        let mut m = Muse::new();
        let chars: Vec<char> = msg.chars().collect();
        let mut read: String = chars.chunks(n).map(|c| m.feed(&c.iter().collect::<String>())).collect();
        read.push_str(&m.finish());
        assert!(read.starts_with("<atem:function_calls>"), "{read:?}");
        round_trip(&read, Format::Atem);
    }
    // the reasoning, then the call: the thinking, the markup gone
    let both = format!("{}{}", cut(&t, "<|start|>assistant to=self<|message|>", "<|eom|>"), msg);
    let mut m = Muse::new();
    let read = m.feed(both.trim_start_matches("<|start|>assistant")) + &m.finish();
    assert!(read.starts_with("<think>") && read.contains(&format!("{THOUGHT}</think><atem:function_calls>")), "{read}");
    reasoning(cut(&t, "<|start|>assistant to=self<|message|>", "<|eom|>"), Some(|| Box::<Muse>::default()), false);
}

/// An answer to the user after thinking: Muse's `to=user` message is the text, Gemma's text follows its channel.
#[test]
fn the_answer_after_the_reasoning() {
    let split = |reply: &str, mut r: Box<dyn Reader>| {
        let read = r.feed(reply) + &r.finish();
        crate::conv::Splitter::split(&read, false)
    };
    let want = (THOUGHT.to_string(), "Three days in Zürich.".to_string());
    assert_eq!(split(&format!("<|channel>thought\n{THOUGHT}<channel|>Three days in Zürich.<turn|>"), Box::new(Gemma::new(false))), want);
    assert_eq!(
        split(&format!(" to=self<|message|>{THOUGHT}<|eom|><|start|>assistant to=user<|message|>Three days in Zürich.<|eot|>"), Box::<Muse>::default()),
        want
    );
}

/// The reader that the template names, and none for the formats the server reads as they are.
#[test]
fn which_reader() {
    use crate::reader::ReaderKind;
    let kind = |src: &str, prompt: &str| ReaderKind::of(&Chat::from_template(src).unwrap(), prompt);
    assert_eq!(kind(GEMMA4, "<|turn>model\n<|channel>thought\n<channel|>"), Some(ReaderKind::Gemma { in_think: false }));
    assert_eq!(kind(GEMMA4, "<|turn>model\n<|channel>thought\n"), Some(ReaderKind::Gemma { in_think: true }));
    assert_eq!(kind(MUSE, "<|start|>assistant"), Some(ReaderKind::Muse));
    assert_eq!(kind(QWEN38, "<think>\n"), None);
    assert_eq!(kind(GLM53, "<|assistant|><think>"), None);
}

// ---- Kimi-K3 and DeepSeek-V4 (their formats as the engine's own templates write them: both releases keep their chat format in Python code, not a template) ----

/// Kimi-K3: the think channel, a response, and calls as `<|open|>call tool=...<|sep|><|open|>argument key=... type=...<|sep|>VALUE<|close|>argument<|sep|>`, read through its reader (a call is written when it closes).
#[test]
fn kimi_k3() {
    use crate::reader::Reader;
    let t = rendered(crate::chat::KIMI_CHAT, vec![]);
    let chat = Chat::from_template(crate::chat::KIMI_CHAT).unwrap();
    assert!(chat.is_xtml());
    // the reply as the model writes it, from inside the think channel the prompt opened, to the end of its calls
    let turn = &t[t.find("Plan a trip.").unwrap()..];
    let reply = {
        let a = turn.find("<|open|>think<|sep|>").unwrap() + "<|open|>think<|sep|>".len();
        let b = turn.find("<|close|>tools<|sep|>").unwrap() + "<|close|>tools<|sep|>".len();
        &turn[a..b]
    };
    for n in [1, 4, 1000] {
        let mut x = crate::xtml::Xtml::new(true);
        let chars: Vec<char> = reply.chars().collect();
        let mut read: String = chars.chunks(n).map(|c| Reader::feed(&mut x, &c.iter().collect::<String>())).collect();
        read.push_str(&Reader::finish(&mut x));
        let (think, answer) = crate::conv::Splitter::split(&read, false);
        assert_eq!(think.trim(), THOUGHT, "pieces of {n}");
        for c in whole(&answer, Format::Hermes) {
            assert_eq!(c.name, "plan_trip");
            same_json(&c.arguments, ARGS);
        }
    }
}

/// DeepSeek-V4: the thinking in `<think>`, and calls in DSML blocks (`<｜DSML｜tool_calls>`), a string by `string="true"`, any other value as JSON.
#[test]
fn deepseek_v4() {
    let t = rendered(crate::chat::DSV4_CHAT, vec![]);
    assert_eq!(Format::of(&Chat::from_template(crate::chat::DSV4_CHAT).unwrap()), Format::Dsml);
    let block = cut(&t, "<｜DSML｜tool_calls>", "</｜DSML｜tool_calls>");
    round_trip(block, Format::Dsml);
    // (its template drops the reasoning of an earlier turn: the reply that opens the think block, as the prompt ends with `<think>`)
    reasoning(&format!("{THOUGHT}</think>\n\n"), None, true);
}

// ---- the whole path: a recorded reply, token by token, as the server's choice reads it ----

/// What a client sees of a reply made of tokens: the reasoning's deltas, the content, each call's name and its arguments' pieces, and how the reply ended.
struct Seen {
    reasoning: Vec<String>,
    content: String,
    calls: Vec<(String, Vec<String>)>,
    finish: &'static str,
}

/// `reply` (the text a family's model writes, from where its generation prompt ends) as tokens of `n` characters, through the choice the server makes (its reader, its reasoning split, its tool parser, streaming).
fn seen(kind: Option<crate::reader::ReaderKind>, format: Format, in_thinking: bool, reply: &str, n: usize) -> Seen {
    use crate::choice::{Cfg, Choice, ToolsCfg};
    use crate::sched::Finish;
    use crate::tools::ToolChoice;
    let chars: Vec<char> = reply.chars().collect();
    let pieces: Vec<String> = chars.chunks(n).map(|c| c.iter().collect()).collect();
    let mut words: Vec<&str> = Vec::new();
    for p in &pieces {
        if !words.contains(&p.as_str()) {
            words.push(p);
        }
    }
    let tok = crate::tok::tests::toy(&words);
    let tools = ToolsCfg { choice: ToolChoice::Auto, parallel: true, format, list: tools(), prefix: String::new(), stream: true };
    let mut c = Choice::new(Cfg { chat: true, in_thinking, tools: Some(tools), reader: kind, ..Cfg::default() });
    let mut seen = Seen { reasoning: Vec::new(), content: String::new(), calls: Vec::new(), finish: "" };
    let take = |d: crate::choice::Delta, seen: &mut Seen| {
        if !d.reasoning.is_empty() {
            seen.reasoning.push(d.reasoning);
        }
        seen.content.push_str(&d.content);
        for e in d.calls {
            match e {
                ToolEv::Start { index, name, .. } => {
                    assert_eq!(index, seen.calls.len());
                    seen.calls.push((name, Vec::new()));
                }
                ToolEv::Args { index, args } => seen.calls[index].1.push(args),
                ToolEv::Content(_) => {}
            }
        }
    };
    for p in &pieces {
        let id = words.iter().position(|w| w == p).unwrap() as u32 + 1;
        let d = c.token(&tok, id).unwrap();
        take(d, &mut seen);
    }
    let (d, done) = c.end(&tok, Finish::Stop).unwrap();
    take(d, &mut seen);
    seen.finish = done.finish_reason;
    seen
}

/// The recorded reply of a family: its assistant turn as its template writes it, from where the generation prompt ends to the end of the call.
fn recorded(family: &str) -> (Option<crate::reader::ReaderKind>, Format, bool, String) {
    use crate::reader::ReaderKind;
    let turn_of = |t: &str| t[t.find("Plan a trip.").unwrap()..].to_string();
    let upto = |turn: &str, from: &str, to: &str| {
        let a = turn.find(from).unwrap_or_else(|| panic!("no {from} in {turn}")) + from.len();
        let b = a + turn[a..].find(to).unwrap_or_else(|| panic!("no {to} in {turn}")) + to.len();
        turn[a..b].to_string()
    };
    match family {
        "qwen" | "glm" => {
            let (src, format) = if family == "qwen" { (QWEN38, Format::Xml) } else { (GLM53, Format::Glm) };
            let turn = turn_of(&rendered(src, vec![]));
            // (the prompt ends with `<think>`: the reply is the thinking, its end and the call)
            (None, format, true, upto(&turn, "<think>", "</tool_call>").trim_start_matches('\n').to_string())
        }
        "gemma" => {
            let turn = turn_of(&rendered(GEMMA4, vec![("preserve_thinking", Value::Bool(true))]));
            let a = turn.find("<|channel>thought\n").unwrap();
            let b = turn.find("<tool_call|>").unwrap() + "<tool_call|>".len();
            (Some(ReaderKind::Gemma { in_think: false }), Format::Hermes, false, turn[a..b].to_string())
        }
        "muse" => {
            let turn = turn_of(&rendered(MUSE, vec![]));
            (Some(ReaderKind::Muse), Format::Atem, false, upto(&turn, "<|start|>assistant", "</atem:function_calls>"))
        }
        "kimi" => {
            let turn = turn_of(&rendered(crate::chat::KIMI_CHAT, vec![]));
            (Some(ReaderKind::Kimi { in_think: true }), Format::Hermes, false, upto(&turn, "<|open|>think<|sep|>", "<|close|>tools<|sep|>"))
        }
        "dsv4" => {
            let turn = turn_of(&rendered(crate::chat::DSV4_CHAT, vec![]));
            let block =
                turn[turn.find("<｜DSML｜tool_calls>").unwrap()..turn.find("</｜DSML｜tool_calls>").unwrap() + "</｜DSML｜tool_calls>".len()].to_string();
            (None, Format::Dsml, true, format!("{THOUGHT}</think>\n\n{block}"))
        }
        other => panic!("{other}"),
    }
}

/// Each family's reply, token by token: the reasoning arrives as deltas and is the reasoning, the answer is empty, the call is `plan_trip` with the arguments it was written with, and the reply ends as a call's does.
/// Where a family writes its arguments as it goes (XML, DSML, ATEM, Gemma's), they arrive in pieces that make the arguments; where it writes a call whole (GLM's, Kimi's), in one.
#[test]
fn every_family_token_by_token() {
    // (family, its arguments arrive in pieces)
    for (family, in_pieces) in [("qwen", true), ("glm", false), ("gemma", true), ("muse", true), ("kimi", false), ("dsv4", true)] {
        let (kind, format, in_thinking, reply) = recorded(family);
        for n in [1, 3, 7, 1000] {
            let s = seen(kind, format, in_thinking, &reply, n);
            assert_eq!(s.reasoning.concat().trim(), THOUGHT, "{family}, tokens of {n}");
            if n == 1 {
                assert!(s.reasoning.len() > 5, "{family}: the reasoning comes as it is written: {:?}", s.reasoning);
            }
            assert!(s.content.trim().is_empty(), "{family}, tokens of {n}: {:?}", s.content);
            assert_eq!(s.calls.len(), 1, "{family}, tokens of {n}: {:?}", s.calls);
            assert_eq!(s.calls[0].0, "plan_trip", "{family}");
            same_json(&s.calls[0].1.concat(), ARGS);
            if n == 1 {
                assert_eq!(s.calls[0].1.len() > 1, in_pieces, "{family}: {:?}", s.calls[0].1);
            }
            assert_eq!(s.finish, "tool_calls", "{family}, tokens of {n}");
        }
    }
}
