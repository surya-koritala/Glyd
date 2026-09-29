//! JSON as far as a checkpoint needs it: a reader (safetensors headers,
//! glyd.json, config.json, an index), keeping each object's keys in their
//! order; and writers of Python's `json.dump(indent=N)` text and of the
//! compact text safetensors' headers are, byte for byte.

/// A JSON value; numbers as their text (read exactly on demand), objects as
/// their pairs in order.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

impl Value {
    /// An object's value for `key` (its first), else None.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(o) => o.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Number(n) => n.parse().ok(),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Number(n) => n.parse().ok(),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&[(String, Value)]> {
        match self {
            Value::Object(o) => Some(o),
            _ => None,
        }
    }

    /// An array of unsigned integers, else None.
    pub fn as_u64s(&self) -> Option<Vec<u64>> {
        self.as_array()?.iter().map(Value::as_u64).collect()
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Value {
        Value::String(s.to_string())
    }
}

impl From<u64> for Value {
    fn from(n: u64) -> Value {
        Value::Number(n.to_string())
    }
}

/// `text` read as one JSON value (whitespace around it), else where it is not.
pub fn parse(text: &str) -> Result<Value, String> {
    let mut p = Parser { b: text.as_bytes(), i: 0 };
    let v = p.value(0)?;
    p.ws();
    if p.i != p.b.len() {
        return Err(p.err("text after the value"));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

const DEPTH: usize = 128; // arrays and objects nested deeper are refused (no JSON a checkpoint holds is)

impl Parser<'_> {
    fn err(&self, what: &str) -> String {
        format!("JSON: {what} at byte {}", self.i)
    }

    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> bool {
        self.ws();
        let yes = self.b.get(self.i) == Some(&c);
        self.i += yes as usize;
        yes
    }

    fn word(&mut self, w: &str, v: Value) -> Result<Value, String> {
        if self.b[self.i..].starts_with(w.as_bytes()) {
            self.i += w.len();
            Ok(v)
        } else {
            Err(self.err("not a value"))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value, String> {
        if depth > DEPTH {
            return Err(self.err("nested too deep"));
        }
        self.ws();
        match self.b.get(self.i) {
            None => Err(self.err("no value")),
            Some(b'{') => {
                self.i += 1;
                let mut o = Vec::new();
                if !self.eat(b'}') {
                    loop {
                        self.ws();
                        let k = self.string()?;
                        if !self.eat(b':') {
                            return Err(self.err("no ':'"));
                        }
                        o.push((k, self.value(depth + 1)?));
                        if self.eat(b',') {
                            continue;
                        }
                        if self.eat(b'}') {
                            break;
                        }
                        return Err(self.err("no ',' or '}'"));
                    }
                }
                Ok(Value::Object(o))
            }
            Some(b'[') => {
                self.i += 1;
                let mut a = Vec::new();
                if !self.eat(b']') {
                    loop {
                        a.push(self.value(depth + 1)?);
                        if self.eat(b',') {
                            continue;
                        }
                        if self.eat(b']') {
                            break;
                        }
                        return Err(self.err("no ',' or ']'"));
                    }
                }
                Ok(Value::Array(a))
            }
            Some(b'"') => Ok(Value::String(self.string()?)),
            Some(b't') => self.word("true", Value::Bool(true)),
            Some(b'f') => self.word("false", Value::Bool(false)),
            Some(b'n') => self.word("null", Value::Null),
            Some(_) => {
                let from = self.i;
                while self.i < self.b.len() && matches!(self.b[self.i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') {
                    self.i += 1;
                }
                let n = std::str::from_utf8(&self.b[from..self.i]).unwrap();
                if n.is_empty() || n.parse::<f64>().is_err() {
                    return Err(self.err("not a value"));
                }
                Ok(Value::Number(n.to_string()))
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let h = self.b.get(self.i..self.i + 4).and_then(|h| std::str::from_utf8(h).ok()).and_then(|h| u32::from_str_radix(h, 16).ok());
        self.i += 4;
        h.ok_or_else(|| self.err("a bad \\u escape"))
    }

    fn string(&mut self) -> Result<String, String> {
        if self.b.get(self.i) != Some(&b'"') {
            return Err(self.err("not a string"));
        }
        self.i += 1;
        let mut out = Vec::new();
        loop {
            match *self.b.get(self.i).ok_or_else(|| self.err("an unterminated string"))? {
                b'"' => {
                    self.i += 1;
                    return String::from_utf8(out).map_err(|_| self.err("a string not UTF-8"));
                }
                b'\\' => {
                    self.i += 1;
                    let c = *self.b.get(self.i).ok_or_else(|| self.err("an unterminated string"))?;
                    self.i += 1;
                    let ch = match c {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let mut c = self.hex4()?;
                            if (0xD800..0xDC00).contains(&c) && self.b[self.i..].starts_with(b"\\u") {
                                self.i += 2;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err(self.err("a lone surrogate"));
                                }
                                c = 0x10000 + ((c - 0xD800) << 10) + (lo - 0xDC00);
                            }
                            char::from_u32(c).ok_or_else(|| self.err("a lone surrogate"))?
                        }
                        _ => return Err(self.err("a bad escape")),
                    };
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
                c if c < 0x20 => return Err(self.err("a control character in a string")),
                c => {
                    out.push(c);
                    self.i += 1;
                }
            }
        }
    }
}

/// `s` quoted as Python's json writes it (ensure_ascii: every character past
/// '~' as \uXXXX, those past U+FFFF as their surrogate pair).
fn python_string(s: &str, out: &mut String) {
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
            ' '..='~' => out.push(c),
            _ => {
                let mut units = [0u16; 2];
                for u in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{u:04x}"));
                }
            }
        }
    }
    out.push('"');
}

/// `v` as Python's `json.dump(v, f, indent=indent)` writes it (no newline at the end).
pub fn to_python(v: &Value, indent: usize) -> String {
    fn go(v: &Value, indent: usize, level: usize, out: &mut String) {
        let pad = |out: &mut String, level: usize| {
            out.push('\n');
            out.extend(std::iter::repeat_n(' ', indent * level));
        };
        match v {
            Value::Null => out.push_str("null"),
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::Number(n) => out.push_str(n),
            Value::String(s) => python_string(s, out),
            Value::Array(a) if a.is_empty() => out.push_str("[]"),
            Value::Object(o) if o.is_empty() => out.push_str("{}"),
            Value::Array(a) => {
                out.push('[');
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    pad(out, level + 1);
                    go(x, indent, level + 1, out);
                }
                pad(out, level);
                out.push(']');
            }
            Value::Object(o) => {
                out.push('{');
                for (i, (k, x)) in o.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    pad(out, level + 1);
                    python_string(k, out);
                    out.push_str(": ");
                    go(x, indent, level + 1, out);
                }
                pad(out, level);
                out.push('}');
            }
        }
    }
    let mut out = String::new();
    go(v, indent, 0, &mut out);
    out
}

/// `v` as serde_json's compact text (safetensors' headers): no spaces,
/// quotes, backslashes and control characters escaped, the rest as it is.
pub fn to_compact(v: &Value) -> String {
    fn string(s: &str, out: &mut String) {
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
    fn go(v: &Value, out: &mut String) {
        match v {
            Value::Null => out.push_str("null"),
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::Number(n) => out.push_str(n),
            Value::String(s) => string(s, out),
            Value::Array(a) => {
                out.push('[');
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    go(x, out);
                }
                out.push(']');
            }
            Value::Object(o) => {
                out.push('{');
                for (i, (k, x)) in o.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    string(k, out);
                    out.push(':');
                    go(x, out);
                }
                out.push('}');
            }
        }
    }
    let mut out = String::new();
    go(v, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_text() {
        // json.dumps({"a": [1, [], {}], "b": {"c": None, "d": True}, "é\n": "x\u2603\U0001f600"}, indent=1)
        let v = parse(r#"{"a": [1, [], {}], "b": {"c": null, "d": true}, "\u00e9\n": "x\u2603\ud83d\ude00"}"#).unwrap();
        let want = "{\n \"a\": [\n  1,\n  [],\n  {}\n ],\n \"b\": {\n  \"c\": null,\n  \"d\": true\n },\n \"\\u00e9\\n\": \"x\\u2603\\ud83d\\ude00\"\n}";
        assert_eq!(to_python(&v, 1), want);
        assert_eq!(parse(&to_python(&v, 2)).unwrap(), v);
        assert_eq!(to_compact(&v), "{\"a\":[1,[],{}],\"b\":{\"c\":null,\"d\":true},\"é\\n\":\"x\u{2603}\u{1f600}\"}");
    }

    #[test]
    fn refused() {
        for bad in ["", "{", "[1,]", "{\"a\" 1}", "\"\\x\"", "01x", "[1] 2", "\"\u{1}\"", &"[".repeat(200)] {
            assert!(parse(bad).is_err(), "{bad:?}");
        }
        assert_eq!(parse(" [ 1 , -2.5e3 ] ").unwrap(), Value::Array(vec![Value::Number("1".into()), Value::Number("-2.5e3".into())]));
    }
}
