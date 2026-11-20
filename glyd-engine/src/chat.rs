//! A model's chat template (the Jinja text of tokenizer_config.json's `chat_template`, or chat_template.jinja) rendered as transformers does: the same variables
//! (`messages`, `tools`, `documents`, `add_generation_prompt`, the tokenizer's special tokens, the caller's extra ones such as `enable_thinking`), blocks trimmed as its
//! environment trims them (trim_blocks, lstrip_blocks), Python's string methods (split, strip, startswith ...) that the templates call, `tojson` as Python's json.dumps writes it, and `strftime_now`.

use crate::ffi::{err, Error, Result};
use glyd_gpu::json::{self, Value as Json};
use minijinja::value::Kwargs;
use minijinja::{Environment, ErrorKind, Value};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

pub struct Chat {
    env: Environment<'static>,
    /// The template's own text (what it mentions: `tools`, `enable_thinking`).
    src: String,
    /// The tokenizer's special tokens the template can name (`bos_token`, `eos_token`, `pad_token`, `unk_token`): only those the tokenizer sets, as transformers passes them.
    tokens: Vec<(&'static str, String)>,
}

/// One turn of a conversation: its role (system, user, assistant) and its text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn new(role: &str, content: &str) -> Message {
        Message { role: role.to_string(), content: content.to_string() }
    }
}

/// What a render takes besides the messages.
#[derive(Clone, Debug, Default)]
pub struct Options {
    pub add_generation_prompt: bool,
    /// The functions the model may call (the request's `tools`): None is `tools = none` in the template, as transformers passes it.
    pub tools: Option<Vec<Json>>,
    /// Variables for the template (the request's `chat_template_kwargs`: `enable_thinking`, `reasoning_effort`, `date_string` ...), after the tokens: a name given here wins.
    pub kwargs: Vec<(String, Json)>,
}

impl Options {
    pub fn new(add_generation_prompt: bool) -> Options {
        Options { add_generation_prompt, ..Options::default() }
    }

    /// `enable_thinking` set where `thinking` says (Some(false) closes the thinking of a template that has it).
    pub fn thinking(mut self, thinking: Option<bool>) -> Options {
        if let Some(b) = thinking {
            self.kwargs.push(("enable_thinking".to_string(), Json::Bool(b)));
        }
        self
    }
}

/// A date and time as `strftime_now` formats them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Civil {
    pub year: i64,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
    /// Days since Monday (Python's weekday()).
    pub weekday: u32,
    /// 1 to 366.
    pub yday: u32,
}

impl Civil {
    /// The calendar date and time of `unix` seconds (UTC) moved by `offset` seconds (a local time zone's).
    pub fn from_unix(unix: i64, offset: i64) -> Civil {
        let t = unix + offset;
        let (days, rem) = (t.div_euclid(86_400), t.rem_euclid(86_400));
        // days since 1970-01-01 to a civil date (Howard Hinnant's algorithm)
        let z = days + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z.rem_euclid(146_097);
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
        let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
        let year = yoe + era * 400 + (month <= 2) as i64;
        let leap = (year % 4 == 0 && year % 100 != 0) || year.rem_euclid(400) == 0;
        const CUM: [u32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
        let yday = CUM[month as usize - 1] + day + (leap && month > 2) as u32;
        Civil {
            year,
            month,
            day,
            hour: (rem / 3600) as u32,
            minute: (rem % 3600 / 60) as u32,
            second: (rem % 60) as u32,
            weekday: (days + 3).rem_euclid(7) as u32,
            yday,
        }
    }

    /// Now, in the computer's local time (as Python's `datetime.now()` gives it, which is what transformers' `strftime_now` reads).
    #[cfg(windows)]
    pub fn now() -> Civil {
        let unix = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64);
        Civil::from_unix(unix, crate::os::win::utc_offset())
    }

    #[cfg(unix)]
    pub fn now() -> Civil {
        #[repr(C)]
        struct Tm {
            sec: i32,
            min: i32,
            hour: i32,
            mday: i32,
            mon: i32,
            year: i32,
            wday: i32,
            yday: i32,
            isdst: i32,
            gmtoff: i64,
            zone: *const std::ffi::c_char,
        }
        extern "C" {
            fn time(t: *mut i64) -> i64;
            fn localtime_r(t: *const i64, out: *mut Tm) -> *mut Tm;
        }
        let unix = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64);
        // SAFETY: `time` and `localtime_r` of the C library, on a Tm of the layout glibc and Darwin share (nine ints, the offset, the zone's name); the result is read after the call fills it.
        let off = unsafe {
            let mut t = time(std::ptr::null_mut());
            if t == 0 {
                t = unix;
            }
            let mut tm = std::mem::MaybeUninit::<Tm>::zeroed();
            if localtime_r(&t, tm.as_mut_ptr()).is_null() {
                0
            } else {
                tm.assume_init().gmtoff
            }
        };
        Civil::from_unix(unix, off)
    }
}

const MONTHS: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const DAYS: [&str; 7] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

/// `fmt` as Python's strftime writes it for `t` (the C locale), for the directives templates use: %Y %y %m %d %e %H %M %S %j %a %A %b %B %p %I %%. Another is an error, not a guess.
pub fn strftime(fmt: &str, t: &Civil) -> Result<String> {
    let mut out = String::new();
    let mut it = fmt.chars();
    while let Some(c) = it.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        let Some(d) = it.next() else { return err("strftime: a format ending in %") };
        match d {
            'Y' => out.push_str(&t.year.to_string()),
            'y' => out.push_str(&format!("{:02}", t.year.rem_euclid(100))),
            'm' => out.push_str(&format!("{:02}", t.month)),
            'd' => out.push_str(&format!("{:02}", t.day)),
            'e' => out.push_str(&format!("{:2}", t.day)),
            'H' => out.push_str(&format!("{:02}", t.hour)),
            'I' => out.push_str(&format!("{:02}", if t.hour.is_multiple_of(12) { 12 } else { t.hour % 12 })),
            'M' => out.push_str(&format!("{:02}", t.minute)),
            'S' => out.push_str(&format!("{:02}", t.second)),
            'p' => out.push_str(if t.hour < 12 { "AM" } else { "PM" }),
            'j' => out.push_str(&format!("{:03}", t.yday)),
            'a' => out.push_str(&DAYS[t.weekday as usize][..3]),
            'A' => out.push_str(DAYS[t.weekday as usize]),
            'b' => out.push_str(&MONTHS[t.month as usize - 1][..3]),
            'B' => out.push_str(MONTHS[t.month as usize - 1]),
            '%' => out.push('%'),
            other => return err(format!("strftime: %{other} is not a directive this engine formats")),
        }
    }
    Ok(out)
}

/// `v` as Python's `json.dumps(v, ensure_ascii=False, indent=indent)` writes it: ", " and ": " between items without an indent; "," and a new line with one.
fn py_json(v: &Value, indent: Option<usize>, level: usize, out: &mut String) -> std::result::Result<(), minijinja::Error> {
    py_json_with(v, &JsonForm { indent, seps: None, sort: false }, level, out)
}

/// How json.dumps writes a value: its indent, its `separators` (item, key), its `sort_keys`.
struct JsonForm {
    indent: Option<usize>,
    seps: Option<(String, String)>,
    sort: bool,
}

fn py_json_with(v: &Value, f: &JsonForm, level: usize, out: &mut String) -> std::result::Result<(), minijinja::Error> {
    use minijinja::value::ValueKind;
    let indent = f.indent;
    let item_sep = f.seps.as_ref().map_or(if indent.is_some() { "," } else { ", " }, |s| s.0.as_str());
    let key_sep = f.seps.as_ref().map_or(": ", |s| s.1.as_str());
    let pad = |level: usize| " ".repeat(indent.unwrap_or(0) * level);
    match v.kind() {
        ValueKind::Undefined | ValueKind::None => out.push_str("null"),
        ValueKind::Bool => out.push_str(if v.is_true() { "true" } else { "false" }),
        ValueKind::Number => out.push_str(&v.to_string()),
        ValueKind::String => json_string(v.as_str().unwrap_or(""), out),
        ValueKind::Seq | ValueKind::Iterable => {
            let items: Vec<Value> = v.try_iter()?.collect();
            if items.is_empty() {
                out.push_str("[]");
                return Ok(());
            }
            out.push('[');
            for (i, x) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                if indent.is_some() {
                    out.push('\n');
                    out.push_str(&pad(level + 1));
                }
                py_json_with(x, f, level + 1, out)?;
            }
            if indent.is_some() {
                out.push('\n');
                out.push_str(&pad(level));
            }
            out.push(']');
        }
        ValueKind::Map => {
            let mut keys: Vec<Value> = v.try_iter()?.collect();
            if f.sort {
                keys.sort_by_key(|k| k.to_string());
            }
            if keys.is_empty() {
                out.push_str("{}");
                return Ok(());
            }
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                if indent.is_some() {
                    out.push('\n');
                    out.push_str(&pad(level + 1));
                }
                json_string(&k.to_string(), out);
                out.push_str(key_sep);
                py_json_with(&v.get_item(k)?, f, level + 1, out)?;
            }
            if indent.is_some() {
                out.push('\n');
                out.push_str(&pad(level));
            }
            out.push('}');
        }
        _ => return Err(minijinja::Error::new(ErrorKind::InvalidOperation, "tojson: a value JSON has no form for")),
    }
    Ok(())
}

fn json_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `v` as Python's `json.dumps(v, ensure_ascii=False)` writes it (the text `tojson` gives a template).
pub fn py_dumps(v: &Json) -> String {
    let mut out = String::new();
    let _ = py_json(&to_value(v), None, 0, &mut out);
    out
}

/// A JSON value as a template value (an object's keys in their order).
pub fn to_value(v: &Json) -> Value {
    match v {
        Json::Null => Value::from(()),
        Json::Bool(b) => Value::from(*b),
        Json::Number(n) => n.parse::<i64>().map(Value::from).or_else(|_| n.parse::<f64>().map(Value::from)).unwrap_or_else(|_| Value::from(n.as_str())),
        Json::String(s) => Value::from(s.as_str()),
        Json::Array(a) => a.iter().map(to_value).collect::<Vec<_>>().into_iter().collect(),
        Json::Object(o) => o.iter().map(|(k, x)| (Value::from(k.as_str()), to_value(x))).collect(),
    }
}

/// A template's `{% generation %}` and `{% endgeneration %}` (the marks of the reply a training mask reads, as Muse Spark's template has them; no
/// output of their own) as an `if true` block, with the same whitespace control.
fn generation_plain(src: &str) -> String {
    let (mut out, mut rest) = (String::with_capacity(src.len()), src);
    while let Some(i) = rest.find("{%") {
        out.push_str(&rest[..i]);
        let tag = &rest[i..];
        // (past the opening's own `%`: `{%}` has no end)
        let Some(j) = tag[2..].find("%}").map(|k| k + 2) else { break };
        let inner = &tag[2..j];
        let word = match inner.trim_matches('-').trim() {
            "generation" => Some("if true"),
            "endgeneration" => Some("endif"),
            _ => None,
        };
        match word {
            Some(w) => out.push_str(&format!("{{%{} {w} {}%}}", if inner.starts_with('-') { "-" } else { "" }, if inner.ends_with('-') { "-" } else { "" })),
            None => out.push_str(&tag[..j + 2]),
        }
        rest = &tag[j + 2..];
    }
    out.push_str(rest);
    out
}

/// DeepSeek-V4's chat format, from its release's encoding notes. The system's text follows the start token, then the tools where the request has them. A user
/// turn is `<｜User｜>TEXT`, and tool results join it as `<tool_result>TEXT</tool_result>`. Before an assistant turn, and at the generation prompt, come
/// `<｜Assistant｜>` and `<think>` where `enable_thinking` is set, else `</think>`. An assistant turn keeps its reasoning after the last user turn, or in every
/// turn where there are tools. Its calls are a DSML block (`<｜DSML｜tool_calls>`), which `tools::Format::Dsml` reads back.
pub const DSV4_CHAT: &str = r#"<｜begin▁of▁sentence｜>
{%- set thinking = enable_thinking is defined and enable_thinking -%}
{%- set keep = tools is defined and tools -%}
{%- set ns = namespace(last=-1) -%}
{%- for m in messages -%}{%- if m['role'] in ['user', 'tool'] -%}{%- set ns.last = loop.index0 -%}{%- endif -%}{%- endfor -%}
{%- set head -%}
## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜tool_calls>" block like the following:

<｜DSML｜tool_calls>
<｜DSML｜invoke name="$TOOL_NAME">
<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>
...
</｜DSML｜invoke>
<｜DSML｜invoke name="$TOOL_NAME2">
...
</｜DSML｜invoke>
</｜DSML｜tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas
{%- endset -%}
{%- set described -%}{%- if keep -%}{{ '\n\n' }}{{ head }}{{ '\n\n' }}
{%- for t in tools -%}{%- if not loop.first -%}{{ '\n' }}{%- endif -%}{{ t['function'] | tojson }}{%- endfor -%}
{{- '\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n' }}{%- endif -%}{%- endset -%}
{%- if messages[0]['role'] != 'system' -%}{{ described }}{%- endif -%}
{%- for m in messages -%}
{%- set before = loop.previtem['role'] if loop.previtem else none -%}
{%- set after = loop.nextitem['role'] if loop.nextitem else none -%}
{%- if m['role'] == 'system' -%}
{{- m['content'] -}}{%- if loop.first -%}{{ described }}{%- endif -%}
{%- elif m['role'] in ['user', 'tool'] -%}
{%- if before in ['user', 'tool'] -%}{{ '\n\n' }}{%- else -%}<｜User｜>{%- endif -%}
{%- if m['role'] == 'tool' -%}<tool_result>{{ m['content'] }}</tool_result>{%- else -%}{{ m['content'] }}{%- endif -%}
{%- if after == 'assistant' or (after is none and add_generation_prompt) -%}<｜Assistant｜>{%- if thinking and (keep or loop.index0 >= ns.last) -%}<think>{%- else -%}</think>{%- endif -%}{%- endif -%}
{%- elif m['role'] == 'assistant' -%}
{%- if thinking and (keep or loop.index0 > ns.last) -%}{{ m.get('reasoning_content') or '' }}</think>{%- endif -%}
{{- m['content'] -}}
{%- if m.get('tool_calls') -%}{{ '\n\n<｜DSML｜tool_calls>\n' }}
{%- for tc in m['tool_calls'] -%}{%- if not loop.first -%}{{ '\n' }}{%- endif -%}<｜DSML｜invoke name="{{ tc['function']['name'] }}">{{ '\n' }}
{%- for k, v in tc['function']['arguments'] | items -%}{%- if not loop.first -%}{{ '\n' }}{%- endif -%}<｜DSML｜parameter name="{{ k }}" string="{{ 'true' if v is string else 'false' }}">{{ v if v is string else v | tojson }}</｜DSML｜parameter>{%- endfor -%}
{{- '\n' }}</｜DSML｜invoke>
{%- endfor -%}
{{- '\n</｜DSML｜tool_calls>' }}
{%- endif -%}<｜end▁of▁sentence｜>
{%- endif -%}
{%- endfor -%}"#;

/// Kimi-K3's chat format as a template (the release has its encoder in Python alone, encoding_k3.py; research/kimi-k3.md 8.3): every message
/// `<|open|>message role="..."<|sep|>` its text `<|close|>message<|sep|><|end_of_msg|>`; the tools declared first in a system message of their JSON (keys
/// sorted, compact); an assistant's think channel (thinking is on unless `enable_thinking` is false: the reasoning of every turn kept), its response and
/// its calls (each argument by its type: a string as it is, any other value as JSON); a tool's result with its call's name and its place among the
/// results since the assistant. The generation prompt opens the think channel, or the response where thinking is off. An attribute's `&` and `"` escaped.
pub const KIMI_CHAT: &str = r#"
{%- set thinking = enable_thinking is not defined or enable_thinking -%}
{%- macro q(v) -%}{{ v | string | replace('&', '&amp;') | replace('"', '&quot;') }}{%- endmacro -%}
{%- if tools is defined and tools -%}
<|open|>message role="system" type="tool-declare"<|sep|># Tools
Here are the available tools, described in JSONSchema.

```json
{{ tools | tojson(sort_keys=true, separators=(',', ':')) }}
```<|close|>message<|sep|><|end_of_msg|>
{%- endif -%}
{%- set ns = namespace(calls=[], n=0) -%}
{%- for m in messages -%}
{%- if m['role'] == 'assistant' -%}
<|open|>message role="assistant"<|sep|>
{%- if thinking -%}<|open|>think<|sep|>{{ m.get('reasoning_content') or '' }}<|close|>think<|sep|>{%- endif -%}
<|open|>response<|sep|>{{ m['content'] or '' }}<|close|>response<|sep|>
{%- if m.get('tool_calls') -%}<|open|>tools<|sep|>
{%- for tc in m['tool_calls'] -%}<|open|>call tool="{{ q(tc['function']['name']) }}" index="{{ loop.index }}"<|sep|>
{%- for k, v in tc['function']['arguments'] | items -%}
{%- if v is none -%}{%- set t = 'null' -%}{%- elif v is boolean -%}{%- set t = 'boolean' -%}{%- elif v is number -%}{%- set t = 'number' -%}{%- elif v is string -%}{%- set t = 'string' -%}{%- elif v is mapping -%}{%- set t = 'object' -%}{%- else -%}{%- set t = 'array' -%}{%- endif -%}
<|open|>argument key="{{ q(k) }}" type="{{ t }}"<|sep|>{{ v if v is string else v | tojson }}<|close|>argument<|sep|>
{%- endfor -%}<|close|>call<|sep|>
{%- endfor -%}<|close|>tools<|sep|>
{%- endif -%}<|close|>message<|sep|><|end_of_msg|>
{%- set ns.calls = m.get('tool_calls') or [] -%}{%- set ns.n = 0 -%}
{%- elif m['role'] == 'tool' -%}
{%- set ns.n = ns.n + 1 -%}
{%- set nm = namespace(name='') -%}
{%- for tc in ns.calls -%}{%- if tc.get('id') == m.get('tool_call_id') -%}{%- set nm.name = tc['function']['name'] -%}{%- endif -%}{%- endfor -%}
<|open|>message role="tool" tool="{{ q(nm.name) }}" index="{{ ns.n }}"<|sep|>{{ m['content'] }}<|close|>message<|sep|><|end_of_msg|>
{%- else -%}
<|open|>message role="{{ q(m['role']) }}"<|sep|>{{ m['content'] }}<|close|>message<|sep|><|end_of_msg|>
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}<|open|>message role="assistant"<|sep|>{%- if thinking -%}<|open|>think<|sep|>{%- else -%}<|open|>response<|sep|>{%- endif -%}{%- endif -%}"#;

/// What a render may spend, in the template engine's instructions (a loop's turn is a few): a conversation of thousands of messages takes millions, and a template that loops over `range` inside `range`, as a
/// repository's file could, runs out of it and is refused where it would hold the engine's thread for hours.
const TEMPLATE_FUEL: u64 = 50_000_000;

impl Chat {
    pub fn from_template(src: &str) -> Result<Chat> {
        Chat::with_clock(src, Arc::new(Civil::now))
    }

    /// `from_template` with the clock `strftime_now` reads (a fixed one, for a test).
    pub fn with_clock(src: &str, clock: Arc<dyn Fn() -> Civil + Send + Sync>) -> Result<Chat> {
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_fuel(Some(TEMPLATE_FUEL));
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        // Python's `iterable` test: none is not one (minijinja iterates none as an empty sequence), and a template may ask `tools is iterable and tools | length > 0` of its `tools = none`
        env.add_test("iterable", |v: Value| !v.is_none() && v.try_iter().is_ok());
        env.add_function("raise_exception", |msg: String| -> std::result::Result<String, minijinja::Error> {
            Err(minijinja::Error::new(ErrorKind::InvalidOperation, msg))
        });
        env.add_function("strftime_now", move |fmt: String| -> std::result::Result<String, minijinja::Error> {
            strftime(&fmt, &clock()).map_err(|e| minijinja::Error::new(ErrorKind::InvalidOperation, e.to_string()))
        });
        // transformers' own `tojson` (not Jinja's, which writes no spaces and escapes < > & '): json.dumps with ensure_ascii off (GLM-5.3-Flash's template says so) unless asked on,
        // and the indent asked for
        env.add_filter("tojson", |v: Value, kwargs: Kwargs| -> std::result::Result<Value, minijinja::Error> {
            let indent: Option<usize> = kwargs.get("indent")?;
            let ascii: Option<bool> = kwargs.get("ensure_ascii")?;
            let sort: Option<bool> = kwargs.get("sort_keys")?;
            // (separators=(item, key): Kimi-K3's tools declaration is compact, (",", ":"))
            let seps: Option<Value> = kwargs.get("separators")?;
            kwargs.assert_all_used()?;
            let seps = match seps {
                Some(p) => {
                    let parts: Vec<String> = p.try_iter()?.map(|x| x.as_str().unwrap_or("").to_string()).collect();
                    match parts.as_slice() {
                        [a, b] => Some((a.clone(), b.clone())),
                        _ => return Err(minijinja::Error::new(ErrorKind::InvalidOperation, "tojson: separators is (item, key)")),
                    }
                }
                None => None,
            };
            let mut s = String::new();
            py_json_with(&v, &JsonForm { indent, seps, sort: sort.unwrap_or(false) }, 0, &mut s)?;
            if ascii == Some(true) {
                // (json.dumps's \uXXXX for every character past ASCII, a pair of them past the BMP)
                s = s
                    .chars()
                    .map(|c| if c.is_ascii() { c.to_string() } else { c.encode_utf16(&mut [0; 2]).iter().map(|u| format!("\\u{u:04x}")).collect() })
                    .collect();
            }
            Ok(Value::from(s))
        });
        env.add_template_owned("chat", generation_plain(src)).map_err(|e| Error(format!("the chat template: {e}")))?;
        Ok(Chat { env, src: src.to_string(), tokens: Vec::new() })
    }

    /// Whether the template's text says `word` (a variable it reads: `tools`, `enable_thinking`).
    pub fn mentions(&self, word: &str) -> bool {
        self.src.contains(word)
    }

    /// Whether the template writes Kimi-K3's tag format (its replies are read through `xtml::Xtml`).
    pub fn is_xtml(&self) -> bool {
        self.mentions("<|open|>message")
    }

    /// Whether the template writes Gemma 4's markup (its replies are read through `reader::Gemma`): the thought channel and the tool call's own marks.
    pub fn is_gemma(&self) -> bool {
        self.mentions("<|tool_call>call:") && self.mentions("<|channel>")
    }

    /// The special tokens (`bos_token` ...) the template may name.
    pub fn with_tokens(mut self, tokens: Vec<(&'static str, String)>) -> Chat {
        self.tokens = tokens;
        self
    }

    /// The template of the model's directory: tokenizer_config.json's `chat_template` (a string) or chat_template.jinja, with the tokenizer's special tokens; None where it has none.
    pub fn load(dir: &Path) -> Result<Option<Chat>> {
        Chat::load_or(dir, None)
    }

    /// `load`, with `fallback` where the directory has no template (a release that gives its chat format as code: DeepSeek-V4's, `DSV4_CHAT`).
    pub fn load_or(dir: &Path, fallback: Option<&str>) -> Result<Option<Chat>> {
        let p = dir.join("tokenizer_config.json");
        let cfg = match std::fs::read_to_string(&p) {
            Ok(text) => Some(json::parse(&text).map_err(|e| Error(format!("{}: {e}", p.display())))?),
            Err(_) => None,
        };
        let template = match std::fs::read_to_string(dir.join("chat_template.jinja")) {
            Ok(t) => Some(t),
            Err(_) => cfg.as_ref().and_then(|v| v.get("chat_template")).and_then(Json::as_str).map(str::to_string).or_else(|| fallback.map(str::to_string)),
        };
        let Some(template) = template else { return Ok(None) };
        let tokens = cfg.as_ref().map_or_else(Vec::new, tokens_of);
        Chat::from_template(&template).map(|c| Some(c.with_tokens(tokens)))
    }

    /// The conversation as text. `thinking`: Some(false) for a template that has `enable_thinking` (Qwen3's) to close its thinking at once.
    pub fn render(&self, messages: &[Message], add_generation_prompt: bool, thinking: Option<bool>) -> Result<String> {
        self.render_with(messages, &Options::new(add_generation_prompt).thinking(thinking))
    }

    pub fn render_with(&self, messages: &[Message], o: &Options) -> Result<String> {
        let msgs = Json::Array(
            messages
                .iter()
                .map(|m| Json::Object(vec![("role".to_string(), Json::from(m.role.as_str())), ("content".to_string(), Json::from(m.content.as_str()))]))
                .collect(),
        );
        self.render_json(&msgs, o)
    }

    /// The conversation given as the JSON messages of an API request (a message may carry `tool_calls`, a `tool_call_id` ...): what the template reads of them it reads as transformers' does.
    pub fn render_json(&self, messages: &Json, o: &Options) -> Result<String> {
        let mut ctx: BTreeMap<String, Value> = BTreeMap::new();
        for (k, v) in &self.tokens {
            ctx.insert((*k).to_string(), Value::from(v.as_str()));
        }
        ctx.insert("messages".to_string(), to_value(messages));
        ctx.insert("tools".to_string(), o.tools.as_ref().map_or_else(|| Value::from(()), |t| to_value(&Json::Array(t.clone()))));
        ctx.insert("documents".to_string(), Value::from(()));
        ctx.insert("add_generation_prompt".to_string(), Value::from(o.add_generation_prompt));
        for (k, v) in &o.kwargs {
            ctx.insert(k.clone(), to_value(v));
        }
        let t = self.env.get_template("chat").map_err(|e| Error(e.to_string()))?;
        t.render(Value::from(ctx)).map_err(|e| Error(format!("the chat template: {e}")))
    }
}

/// The special tokens a tokenizer_config.json sets (a string, or an added token's `content`).
fn tokens_of(cfg: &Json) -> Vec<(&'static str, String)> {
    ["bos_token", "eos_token", "pad_token", "unk_token"]
        .into_iter()
        .filter_map(|k| {
            let v = cfg.get(k)?;
            let s = v.as_str().or_else(|| v.get("content").and_then(Json::as_str))?;
            Some((k, s.to_string()))
        })
        .collect()
}

/// A conversation as the calibration corpus holds it: a JSON list of {"role": ..., "content": ...}.
pub fn parse_messages(json: &str) -> Result<Vec<Message>> {
    let v = glyd_gpu::json::parse(json).map_err(|e| Error(format!("messages: {e}")))?;
    let Some(list) = v.as_array() else { return err("messages: not a list") };
    list.iter()
        .map(|m| {
            let role = m.get("role").and_then(glyd_gpu::json::Value::as_str).ok_or_else(|| Error("a message with no role".into()))?;
            let content = m.get("content").and_then(glyd_gpu::json::Value::as_str).ok_or_else(|| Error("a message with no text".into()))?;
            Ok(Message::new(role, content))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A template of the Qwen3 kind: a system turn, the turns, thinking closed for the answers after the last question, the generation prompt.
    const SMALL: &str = "{%- for message in messages %}\n    {%- if message.role == 'system' and loop.first %}\n        {{- '<|im_start|>system\\n' + message.content + '<|im_end|>\\n' }}\n    {%- elif message.role == 'user' %}\n        {{- '<|im_start|>user\\n' + message.content + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>assistant\\n' + message.content.strip('\\n') + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}";

    /// A template that loops without end in all but name runs out of fuel and is an error (it held the process for minutes), and one of a big conversation still renders.
    #[test]
    fn a_template_that_loops_for_hours_is_refused() {
        let spin = Chat::from_template("{% for a in range(99999) %}{% for b in range(99999) %}{% endfor %}{% endfor %}x").unwrap();
        let start = std::time::Instant::now();
        let e = spin.render(&[Message::new("user", "hi")], false, None).unwrap_err().to_string();
        assert!(e.contains("fuel"), "{e}");
        assert!(start.elapsed() < std::time::Duration::from_secs(60), "{:?}", start.elapsed());
        let t = Chat::from_template("{% for m in messages %}{{ m.role }}: {{ m.content | trim }}\n{% endfor %}").unwrap();
        let many: Vec<Message> = (0..20_000).map(|i| Message::new(if i % 2 == 0 { "user" } else { "assistant" }, "some words of a turn")).collect();
        assert!(t.render(&many, true, None).unwrap().len() > 400_000);
    }

    /// A tag that opens and never closes, or closes on its own opening's `%` (`{%}`), is left to the template engine to refuse: it was a panic in the scan for `generation` marks.
    #[test]
    fn a_tag_with_no_end_is_an_error_not_a_panic() {
        for src in ["{%}", "{%", "a{%}b", "{% if x %}{%}", "{%-}", "{{ 1 }}{%", "{%%}"] {
            assert!(Chat::from_template(src).is_err() || generation_plain(src).contains("{%"), "{src:?}");
        }
        assert_eq!(generation_plain("{%}x"), "{%}x");
    }

    /// `{% generation %}` blocks render their text, whitespace control kept.
    #[test]
    fn generation_marks_are_plain_blocks() {
        let c = Chat::from_template("a{%- generation %}b{% endgeneration -%} c{% generation %}{{ messages | length }}{% endgeneration %}").unwrap();
        assert_eq!(c.render(&[Message::new("user", "x")], false, None).unwrap(), "abc1");
    }

    /// DeepSeek-V4's format: the case of its release's encoding notes (two turns, reasoning kept after the last question only), and the prompts a reply starts from.
    #[test]
    fn deepseek_v4_chat() {
        let c = Chat::from_template(DSV4_CHAT).unwrap();
        let m = json::parse(r#"[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello"},
            {"role": "assistant", "reasoning_content": "The user said hello, I should greet back.", "content": "Hi there! How can I help you?"},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "reasoning_content": "The user asks about the capital of France. It is Paris.", "content": "The capital of France is Paris."}]"#).unwrap();
        assert_eq!(c.render_json(&m, &Options::new(false).thinking(Some(true))).unwrap(), "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>Hello<｜Assistant｜></think>Hi there! How can I help you?<｜end▁of▁sentence｜><｜User｜>What is the capital of France?<｜Assistant｜><think>The user asks about the capital of France. It is Paris.</think>The capital of France is Paris.<｜end▁of▁sentence｜>");
        // without thinking, no turn shows its reasoning
        assert_eq!(c.render_json(&m, &Options::new(false)).unwrap(), "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>Hello<｜Assistant｜></think>Hi there! How can I help you?<｜end▁of▁sentence｜><｜User｜>What is the capital of France?<｜Assistant｜></think>The capital of France is Paris.<｜end▁of▁sentence｜>");
        let q = vec![Message::new("user", "What is 2+2?")];
        assert_eq!(c.render(&q, true, None).unwrap(), "<｜begin▁of▁sentence｜><｜User｜>What is 2+2?<｜Assistant｜></think>");
        assert_eq!(c.render(&q, true, Some(true)).unwrap(), "<｜begin▁of▁sentence｜><｜User｜>What is 2+2?<｜Assistant｜><think>");
    }

    /// DeepSeek-V4 with tools: the tools after the system's text, a call as a DSML block (a string as it is, other values as JSON), the tool's result in the
    /// next user turn, and every turn's reasoning kept.
    #[test]
    fn deepseek_v4_tools() {
        let c = Chat::from_template(DSV4_CHAT).unwrap();
        let tool = json::parse(r#"{"type": "function", "function": {"name": "lookup", "description": "Find a word", "parameters": {"type": "object", "properties": {"word": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["word"]}}}"#).unwrap();
        let m = json::parse(r#"[{"role": "system", "content": "Answer briefly."}, {"role": "user", "content": "Define glyph."},
            {"role": "assistant", "content": "", "reasoning_content": "Look it up.", "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": {"word": "glyph", "limit": 2}}}]},
            {"role": "tool", "content": "{\"meaning\": \"a carved symbol\"}"}, {"role": "assistant", "content": "A carved symbol.", "reasoning_content": "Got it."}]"#).unwrap();
        let o = Options { tools: Some(vec![tool]), ..Options::new(false) }.thinking(Some(true));
        let t = c.render_json(&m, &o).unwrap();
        assert!(t.starts_with("<｜begin▁of▁sentence｜>Answer briefly.\n\n## Tools\n\nYou have access to a set of tools"), "{t}");
        assert!(t.contains("### Available Tool Schemas\n\n{\"name\": \"lookup\", \"description\": \"Find a word\", \"parameters\": {\"type\": \"object\", \"properties\": {\"word\": {\"type\": \"string\"}, \"limit\": {\"type\": \"integer\"}}, \"required\": [\"word\"]}}\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n<｜User｜>"), "{t}");
        let calls = "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"lookup\">\n<｜DSML｜parameter name=\"word\" string=\"true\">glyph</｜DSML｜parameter>\n<｜DSML｜parameter name=\"limit\" string=\"false\">2</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        assert!(t.ends_with(&format!("<｜User｜>Define glyph.<｜Assistant｜><think>Look it up.</think>{calls}<｜end▁of▁sentence｜><｜User｜><tool_result>{{\"meaning\": \"a carved symbol\"}}</tool_result><｜Assistant｜><think>Got it.</think>A carved symbol.<｜end▁of▁sentence｜>")), "{t}");
        // without thinking; a generation prompt after the tool's result; no system message (the tools come first)
        let o = Options { tools: o.tools.clone(), ..Options::new(true) };
        let Json::Array(all) = &m else { unreachable!() };
        let t = c.render_json(&Json::Array(all[1..4].to_vec()), &o).unwrap();
        assert!(t.starts_with("<｜begin▁of▁sentence｜>\n\n## Tools\n\n"), "{t}");
        assert!(t.ends_with(&format!("tool calls.\n<｜User｜>Define glyph.<｜Assistant｜></think>{calls}<｜end▁of▁sentence｜><｜User｜><tool_result>{{\"meaning\": \"a carved symbol\"}}</tool_result><｜Assistant｜></think>")), "{t}");
    }

    /// Kimi-K3's format: the spec's plain turn with thinking on (research/kimi-k3.md 8.3), and with thinking off; the tools declared first (keys sorted,
    /// compact), a call's arguments by type, its result with the call's name and place.
    #[test]
    fn kimi_k3_chat() {
        let c = Chat::from_template(KIMI_CHAT).unwrap();
        assert!(c.is_xtml());
        let hi = json::parse(r#"[{"role": "user", "content": "Hi"}]"#).unwrap();
        let o = Options::new(true);
        assert_eq!(
            c.render_json(&hi, &o).unwrap(),
            "<|open|>message role=\"user\"<|sep|>Hi<|close|>message<|sep|><|end_of_msg|><|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>"
        );
        assert!(c.render_json(&hi, &Options::new(true).thinking(Some(false))).unwrap().ends_with("role=\"assistant\"<|sep|><|open|>response<|sep|>"));
        let tool = json::parse(
            r#"{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {"word": {"type": "string"}}}}}"#,
        )
        .unwrap();
        let m = json::parse(r#"[{"role": "user", "content": "Define glyph."},
            {"role": "assistant", "content": "", "reasoning_content": "Look it up.", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": {"word": "glyph", "limit": 2, "opts": {"deep": true}, "x": null}}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "a carved symbol"}]"#).unwrap();
        let o = Options { tools: Some(vec![tool]), ..Options::new(true) };
        let t = c.render_json(&m, &o).unwrap();
        assert!(t.starts_with("<|open|>message role=\"system\" type=\"tool-declare\"<|sep|># Tools\nHere are the available tools, described in JSONSchema.\n\n```json\n[{\"function\":{\"name\":\"lookup\",\"parameters\":{\"properties\":{\"word\":{\"type\":\"string\"}},\"type\":\"object\"}},\"type\":\"function\"}]\n```<|close|>message<|sep|><|end_of_msg|>"), "{t}");
        let call = "<|open|>tools<|sep|><|open|>call tool=\"lookup\" index=\"1\"<|sep|><|open|>argument key=\"word\" type=\"string\"<|sep|>glyph<|close|>argument<|sep|><|open|>argument key=\"limit\" type=\"number\"<|sep|>2<|close|>argument<|sep|><|open|>argument key=\"opts\" type=\"object\"<|sep|>{\"deep\": true}<|close|>argument<|sep|><|open|>argument key=\"x\" type=\"null\"<|sep|>null<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|>";
        assert!(t.contains(&format!("<|open|>think<|sep|>Look it up.<|close|>think<|sep|><|open|>response<|sep|><|close|>response<|sep|>{call}<|close|>message<|sep|><|end_of_msg|><|open|>message role=\"tool\" tool=\"lookup\" index=\"1\"<|sep|>a carved symbol<|close|>message<|sep|><|end_of_msg|>")), "{t}");
    }

    #[test]
    fn a_conversation_as_text() {
        let c = Chat::from_template(SMALL).unwrap();
        let m = vec![Message::new("system", "Be brief."), Message::new("user", "Hi"), Message::new("assistant", "\nHello\n")];
        assert_eq!(
            c.render(&m, false, None).unwrap(),
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello<|im_end|>\n"
        );
        assert_eq!(c.render(&m[..2], true, None).unwrap(), "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
        assert_eq!(
            c.render(&m[..2], true, Some(false)).unwrap(),
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    /// Nemotron 3's template asks `tools is iterable and tools | length > 0` of the `tools = none` that a render without tools gives it: none is not iterable, as in Python's Jinja (minijinja iterates it as an
    /// empty sequence, and the length of none is an error there); a list of tools, a string and the undefined are.
    #[test]
    fn none_is_not_iterable_as_in_python() {
        let c = Chat::from_template(
            "{%- if tools is iterable and tools | length > 0 %}tools: {{ tools | length }}{%- else %}none{%- endif %}|{{ messages | length }}",
        )
        .unwrap();
        let m = vec![Message::new("user", "Hi")];
        assert_eq!(c.render(&m, false, None).unwrap(), "none|1");
        let with = Options { tools: Some(vec![glyd_gpu::json::parse(r#"{"type": "function", "function": {"name": "f"}}"#).unwrap()]), ..Options::default() };
        assert_eq!(c.render_with(&m, &with).unwrap(), "tools: 1|1");
        let t =
            Chat::from_template("{{ 'a' is iterable }} {{ [1] is iterable }} {{ 3 is iterable }} {{ none is iterable }} {{ nothing is iterable }}").unwrap();
        assert_eq!(t.render(&m, false, None).unwrap(), "True True False False True");
    }

    #[test]
    fn messages_of_the_corpus() {
        let m = parse_messages(r#"[{"role": "user", "content": "a\nb"}, {"role": "assistant", "content": "é"}]"#).unwrap();
        assert_eq!(m, vec![Message::new("user", "a\nb"), Message::new("assistant", "é")]);
        assert!(parse_messages(r#"{"role": "user"}"#).is_err());
    }

    /// Qwen3's own template, as transformers renders it (tests/fixtures/chat_qwen3.json: tools/chat_fixtures.py): the corpus's chats and others, with and without the generation prompt and thinking.
    #[test]
    fn qwen3s_template_as_transformers() {
        let f = glyd_gpu::json::parse(include_str!("../tests/fixtures/chat_qwen3.json")).unwrap();
        let c = Chat::from_template(f.get("template").unwrap().as_str().unwrap()).unwrap();
        let cases = f.get("cases").unwrap().as_array().unwrap();
        assert!(cases.len() >= 30);
        for (i, case) in cases.iter().enumerate() {
            let msgs: Vec<Message> = case
                .get("messages")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|m| Message::new(m.get("role").unwrap().as_str().unwrap(), m.get("content").unwrap().as_str().unwrap()))
                .collect();
            let gen = case.get("gen").unwrap().as_bool().unwrap();
            let thinking = case.get("thinking").and_then(glyd_gpu::json::Value::as_bool);
            let want = case.get("text").unwrap().as_str().unwrap();
            assert_eq!(
                c.render(&msgs, gen, thinking).unwrap(),
                want,
                "case {i}: {:?} gen {gen} thinking {thinking:?}",
                msgs.iter().map(|m| &m.role).collect::<Vec<_>>()
            );
        }
    }

    fn fixed() -> Arc<dyn Fn() -> Civil + Send + Sync> {
        Arc::new(|| Civil::from_unix(1_791_210_600, 0)) // 2026-10-05 14:30:00 UTC, the date the fixtures were made on
    }

    /// Without the tokenizer's tokens a Llama prompt has no BOS (what the engine did before): the tokens are what put it there.
    #[test]
    fn the_bos_comes_from_the_tokenizers_tokens() {
        let t = "{{- bos_token }}{%- for m in messages %}[{{ m['content'] }}]{%- endfor %}{%- if eos_token is defined %}{{ eos_token }}{%- endif %}";
        let msgs = [Message::new("user", "hi")];
        let bare = Chat::from_template(t).unwrap();
        assert_eq!(bare.render(&msgs, false, None).unwrap(), "[hi]");
        let with = Chat::from_template(t).unwrap().with_tokens(vec![("bos_token", "<s>".to_string()), ("eos_token", "</s>".to_string())]);
        assert_eq!(with.render(&msgs, false, None).unwrap(), "<s>[hi]</s>");
    }

    /// tokenizer_config.json's tokens as a string or as an added token's object; null (Qwen's bos_token) is not passed, so a template that tests `bos_token is defined` sees it undefined.
    #[test]
    fn tokens_of_a_tokenizer_config() {
        let v = json::parse(r#"{"bos_token": null, "eos_token": "<|im_end|>", "pad_token": {"content": "<pad>", "lstrip": false}, "unk_token": {"__type": "AddedToken", "content": "<unk>"}}"#).unwrap();
        assert_eq!(tokens_of(&v), vec![("eos_token", "<|im_end|>".to_string()), ("pad_token", "<pad>".to_string()), ("unk_token", "<unk>".to_string())]);
        let dir = std::env::temp_dir().join(format!("glyd-engine-chat-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"bos_token": "<s>", "chat_template": "{{ bos_token }}{{ messages[0]['content'] }}{% if bos_token is defined %}!{% endif %}"}"#,
        )
        .unwrap();
        let c = Chat::load(&dir).unwrap().unwrap();
        assert_eq!(c.render(&[Message::new("user", "x")], false, None).unwrap(), "<s>x!");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// `tools` and `documents` are passed as none where none are given (transformers does), so `tools is defined` is true there; a list is passed as it is.
    #[test]
    fn tools_are_none_unless_given() {
        let c = Chat::from_template("{% if tools is none %}no tools{% else %}{{ tools | length }} tools{% endif %}, {{ documents is none }}").unwrap();
        let m = [Message::new("user", "x")];
        assert_eq!(c.render(&m, false, None).unwrap(), "no tools, True");
        let tool = json::parse(r#"{"type": "function", "function": {"name": "f"}}"#).unwrap();
        assert_eq!(c.render_with(&m, &Options { tools: Some(vec![tool]), ..Options::default() }).unwrap(), "1 tools, True");
    }

    /// The template's variables from the request's `chat_template_kwargs`, as they are (a string, a number, a boolean).
    #[test]
    fn extra_variables_reach_the_template() {
        let c = Chat::from_template("{{ reasoning_effort }}|{{ n + 1 }}|{{ enable_thinking is false }}").unwrap();
        let o = Options::new(false).thinking(Some(false));
        let o = Options { kwargs: [o.kwargs, vec![("reasoning_effort".to_string(), Json::from("low")), ("n".to_string(), Json::from(41u64))]].concat(), ..o };
        assert_eq!(c.render_with(&[], &o).unwrap(), "low|42|True");
    }

    /// `tojson` as Python's json.dumps(ensure_ascii=False): ", " and ": " between items, the order of the keys kept, non-ASCII as it is, < > & ' not escaped; `indent=4` as json.dumps(indent=4).
    #[test]
    fn tojson_as_python() {
        let c = Chat::from_template("{{ x | tojson }}\n{{ x | tojson(indent=4) }}").unwrap();
        let x = json::parse(r#"{"b": [1, 2.5, "é<&>'\n\"", null, true], "a": {}, "c": [], "d": {"e": false}}"#).unwrap();
        let msgs = Json::Array(vec![]);
        let o = Options { kwargs: vec![("x".to_string(), x)], ..Options::new(false) };
        let want = "{\"b\": [1, 2.5, \"é<&>'\\n\\\"\", null, true], \"a\": {}, \"c\": [], \"d\": {\"e\": false}}\n{\n    \"b\": [\n        1,\n        2.5,\n        \"é<&>'\\n\\\"\",\n        null,\n        true\n    ],\n    \"a\": {},\n    \"c\": [],\n    \"d\": {\n        \"e\": false\n    }\n}";
        assert_eq!(c.render_json(&msgs, &o).unwrap(), want);
        // (GLM-5.3-Flash's template asks for ensure_ascii=False, json.dumps's on escapes past ASCII: é, and 😀 as a pair)
        let c = Chat::from_template("{{ y | tojson(ensure_ascii=False) }}|{{ y | tojson(ensure_ascii=True) }}").unwrap();
        let o = Options { kwargs: vec![("y".to_string(), Json::String("é😀a".into()))], ..Options::new(false) };
        assert_eq!(c.render_json(&msgs, &o).unwrap(), "\"é😀a\"|\"\\u00e9\\ud83d\\ude00a\"");
        // (sort_keys and compact separators, as json.dumps(sort_keys=True, separators=(",", ":")) writes nested maps)
        let c = Chat::from_template("{{ z | tojson(sort_keys=true, separators=(',', ':')) }}").unwrap();
        let o = Options { kwargs: vec![("z".to_string(), json::parse(r#"{"b": [1, {"y": 2, "x": 1}], "a": "é"}"#).unwrap())], ..Options::new(false) };
        assert_eq!(c.render_json(&msgs, &o).unwrap(), r#"{"a":"é","b":[1,{"x":1,"y":2}]}"#);
    }

    /// strftime as Python writes it, for the dates templates ask for ("%d %b %Y", "%Y-%m-%d").
    #[test]
    fn strftime_as_python() {
        let t = Civil::from_unix(1_791_210_600, 0);
        assert_eq!(strftime("%d %b %Y", &t).unwrap(), "05 Oct 2026");
        assert_eq!(strftime("%Y-%m-%d %H:%M:%S %A %B %j %y|%e|%a %p %I %%", &t).unwrap(), "2026-10-05 14:30:00 Monday October 278 26| 5|Mon PM 02 %");
        let leap = Civil::from_unix(1_709_251_199, 0);
        assert_eq!(strftime("%d %b %Y|%Y-%m-%d|%A %B %j|%H:%M:%S", &leap).unwrap(), "29 Feb 2024|2024-02-29|Thursday February 060|23:59:59");
        assert_eq!(strftime("%d %b %Y|%A|%j", &Civil::from_unix(0, 0)).unwrap(), "01 Jan 1970|Thursday|001");
        assert_eq!(strftime("%Y-%m-%d|%A|%j|%y", &Civil::from_unix(978_264_000, 0)).unwrap(), "2000-12-31|Sunday|366|00");
        // a time zone's offset moves the date (New York, UTC-4, in October: 23:30 UTC of the 4th is 19:30 there)
        assert_eq!(strftime("%d %b %Y %H", &Civil::from_unix(1_791_210_600 - 15 * 3600, -4 * 3600)).unwrap(), "04 Oct 2026 19");
        assert_eq!(strftime("%d %b %Y %H", &Civil::from_unix(1_791_210_600 - 15 * 3600, 9 * 3600)).unwrap(), "05 Oct 2026 08");
        assert!(strftime("%Z", &t).unwrap_err().to_string().contains("%Z"));
        assert!(strftime("100%", &t).is_err());
        // the template's function
        let c = Chat::with_clock("{{ strftime_now('%d %b %Y') }}", fixed()).unwrap();
        assert_eq!(c.render(&[], false, None).unwrap(), "05 Oct 2026");
        assert!(Civil::now().year >= 2026);
    }
}
