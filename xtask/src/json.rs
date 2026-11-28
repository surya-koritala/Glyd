//! JSON as the tasks need it: read, a value's keys in the order they were written, and written back or printed the way Python's json and str() did
//! (the scripts these tasks replace printed values with them, and wrote files with json.dump).

use crate::{fail, py, Res};

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// A number, as written.
    Num(String),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

/// A number as Python has it: an int of any size the files here hold, or a float.
#[derive(Debug, Clone, Copy)]
pub enum Num {
    Int(i128),
    Float(f64),
}

impl Num {
    pub fn f(self) -> f64 {
        match self {
            Num::Int(i) => i as f64,
            Num::Float(f) => f,
        }
    }

    pub fn sub(self, o: Num) -> Num {
        match (self, o) {
            (Num::Int(a), Num::Int(b)) => Num::Int(a - b),
            _ => Num::Float(self.f() - o.f()),
        }
    }

    pub fn gt(self, o: Num) -> bool {
        match (self, o) {
            (Num::Int(a), Num::Int(b)) => a > b,
            _ => self.f() > o.f(),
        }
    }

    pub fn eq(self, o: Num) -> bool {
        match (self, o) {
            (Num::Int(a), Num::Int(b)) => a == b,
            _ => self.f() == o.f(),
        }
    }
}

/// repr() of a float: the shortest digits that read back, an exponent below 1e-4 and from 1e16.
fn float_repr(f: f64) -> String {
    if !f.is_finite() {
        return if f.is_nan() {
            "nan".into()
        } else if f > 0.0 {
            "inf".into()
        } else {
            "-inf".into()
        };
    }
    let sci = format!("{f:e}"); // d.ddde[-]x, the shortest that reads back
    let (mantissa, exp) = sci.split_once('e').unwrap();
    let (sign, mantissa) = mantissa.strip_prefix('-').map_or(("", mantissa), |m| ("-", m));
    let (digits, exp): (String, i32) = (mantissa.replace('.', ""), exp.parse().unwrap());
    if !(-4..16).contains(&exp) {
        let frac = if digits.len() > 1 { format!(".{}", &digits[1..]) } else { String::new() };
        return format!("{sign}{}{frac}e{}{:02}", &digits[..1], if exp < 0 { '-' } else { '+' }, exp.abs());
    }
    if exp < 0 {
        return format!("{sign}0.{}{digits}", "0".repeat((-exp - 1) as usize));
    }
    let point = exp as usize + 1;
    if digits.len() <= point {
        format!("{sign}{digits}{}.0", "0".repeat(point - digits.len()))
    } else {
        format!("{sign}{}.{}", &digits[..point], &digits[point..])
    }
}

impl std::fmt::Display for Num {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Num::Int(i) => write!(f, "{i}"),
            Num::Float(x) => f.write_str(&float_repr(*x)),
        }
    }
}

impl Json {
    pub fn parse(text: &str) -> Res<Json> {
        let mut p = Parser { s: text.as_bytes(), at: 0 };
        let v = p.value()?;
        p.space();
        if p.at != p.s.len() {
            return Err(p.bad("more after the value"));
        }
        Ok(v)
    }

    /// An object's value of this name.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(o) => o.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Set an object's value: in the place the name has, or after the last.
    pub fn set(&mut self, key: &str, value: Json) {
        if let Json::Obj(o) = self {
            match o.iter_mut().find(|(k, _)| k == key) {
                Some((_, v)) => *v = value,
                None => o.push((key.to_string(), value)),
            }
        }
    }

    pub fn num(&self) -> Option<Num> {
        match self {
            Json::Num(n) if n.contains(['.', 'e', 'E']) => n.parse().ok().map(Num::Float),
            Json::Num(n) => n.parse().ok().map(Num::Int),
            Json::Bool(b) => Some(Num::Int(i128::from(*b))),
            _ => None,
        }
    }

    pub fn str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn arr(&self) -> &[Json] {
        match self {
            Json::Arr(a) => a,
            _ => &[],
        }
    }

    /// Python's truth of the value: nothing, zero and the empty are false.
    pub fn truthy(&self) -> bool {
        match self {
            Json::Null => false,
            Json::Bool(b) => *b,
            Json::Num(_) => self.num().is_some_and(|n| n.f() != 0.0),
            Json::Str(s) => !s.is_empty(),
            Json::Arr(a) => !a.is_empty(),
            Json::Obj(o) => !o.is_empty(),
        }
    }

    /// Python's ==: numbers by value (1 == 1.0 == True), objects whatever their keys' order.
    pub fn same(&self, o: &Json) -> bool {
        match (self, o) {
            (Json::Null, Json::Null) => true,
            (Json::Str(a), Json::Str(b)) => a == b,
            (Json::Arr(a), Json::Arr(b)) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.same(y)),
            (Json::Obj(a), Json::Obj(b)) => a.len() == b.len() && a.iter().all(|(k, v)| o.get(k).is_some_and(|w| v.same(w))),
            _ => matches!((self.num(), o.num()), (Some(a), Some(b)) if a.eq(b)),
        }
    }

    /// repr() of the value json.load made: None, True, 'text', [1, 2], {'key': 1.5}.
    pub fn repr(&self) -> String {
        match self {
            Json::Null => "None".into(),
            Json::Bool(b) => if *b { "True" } else { "False" }.into(),
            Json::Num(_) => self.num().map_or_else(|| "nan".into(), |n| n.to_string()),
            Json::Str(s) => py::repr(s),
            Json::Arr(a) => format!("[{}]", a.iter().map(Json::repr).collect::<Vec<_>>().join(", ")),
            Json::Obj(o) => format!("{{{}}}", o.iter().map(|(k, v)| format!("{}: {}", py::repr(k), v.repr())).collect::<Vec<_>>().join(", ")),
        }
    }

    /// str() of it: a text as it is, anything else its repr.
    pub fn text(&self) -> String {
        match self {
            Json::Str(s) => s.clone(),
            v => v.repr(),
        }
    }

    /// json.dumps: `indent` None is one line (", " and ": "), Some(n) a line a value; `ascii` writes every other character as \uXXXX (ensure_ascii).
    pub fn dump(&self, indent: Option<usize>, ascii: bool) -> String {
        let mut out = String::new();
        self.write(&mut out, indent, ascii, 0);
        out
    }

    fn write(&self, out: &mut String, indent: Option<usize>, ascii: bool, depth: usize) {
        let line = |out: &mut String, depth: usize| {
            if let Some(n) = indent {
                out.push('\n');
                out.push_str(&" ".repeat(n * depth));
            }
        };
        let between = if indent.is_some() { "," } else { ", " };
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Num(_) => out.push_str(&self.num().map_or_else(|| "NaN".into(), |n| n.to_string())),
            Json::Str(s) => quote(out, s, ascii),
            Json::Arr(a) if a.is_empty() => out.push_str("[]"),
            Json::Obj(o) if o.is_empty() => out.push_str("{}"),
            Json::Arr(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    out.push_str(if i > 0 { between } else { "" });
                    line(out, depth + 1);
                    v.write(out, indent, ascii, depth + 1);
                }
                line(out, depth);
                out.push(']');
            }
            Json::Obj(o) => {
                out.push('{');
                for (i, (k, v)) in o.iter().enumerate() {
                    out.push_str(if i > 0 { between } else { "" });
                    line(out, depth + 1);
                    quote(out, k, ascii);
                    out.push_str(": ");
                    v.write(out, indent, ascii, depth + 1);
                }
                line(out, depth);
                out.push('}');
            }
        }
    }
}

fn quote(out: &mut String, s: &str, ascii: bool) {
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
            c if (c as u32) < 0x20 || (ascii && !c.is_ascii()) => {
                let mut units = [0u16; 2];
                for u in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{u:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

struct Parser<'a> {
    s: &'a [u8],
    at: usize,
}

impl Parser<'_> {
    fn bad(&self, what: &str) -> crate::Exit {
        fail(format!("not JSON: {what} at byte {}", self.at))
    }

    fn space(&mut self) {
        while matches!(self.s.get(self.at), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.at += 1;
        }
    }

    fn word(&mut self, word: &str, v: Json) -> Res<Json> {
        if self.s[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            Ok(v)
        } else {
            Err(self.bad("a value was expected"))
        }
    }

    fn value(&mut self) -> Res<Json> {
        self.space();
        match self.s.get(self.at) {
            Some(b'n') => self.word("null", Json::Null),
            Some(b't') => self.word("true", Json::Bool(true)),
            Some(b'f') => self.word("false", Json::Bool(false)),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b'[') => {
                self.at += 1;
                let mut a = Vec::new();
                self.space();
                if self.s.get(self.at) == Some(&b']') {
                    self.at += 1;
                    return Ok(Json::Arr(a));
                }
                loop {
                    a.push(self.value()?);
                    self.space();
                    match self.s.get(self.at) {
                        Some(b',') => self.at += 1,
                        Some(b']') => {
                            self.at += 1;
                            return Ok(Json::Arr(a));
                        }
                        _ => return Err(self.bad("`,` or `]` was expected")),
                    }
                }
            }
            Some(b'{') => {
                self.at += 1;
                let mut o = Json::Obj(Vec::new());
                self.space();
                if self.s.get(self.at) == Some(&b'}') {
                    self.at += 1;
                    return Ok(o);
                }
                loop {
                    self.space();
                    if self.s.get(self.at) != Some(&b'"') {
                        return Err(self.bad("a name was expected"));
                    }
                    let k = self.string()?;
                    self.space();
                    if self.s.get(self.at) != Some(&b':') {
                        return Err(self.bad("`:` was expected"));
                    }
                    self.at += 1;
                    let v = self.value()?;
                    o.set(&k, v); // (a name twice: the last value, in the first's place, as a dict)
                    self.space();
                    match self.s.get(self.at) {
                        Some(b',') => self.at += 1,
                        Some(b'}') => {
                            self.at += 1;
                            return Ok(o);
                        }
                        _ => return Err(self.bad("`,` or `}` was expected")),
                    }
                }
            }
            Some(b'-' | b'0'..=b'9') => {
                let start = self.at;
                self.at += usize::from(self.s[self.at] == b'-');
                let digits = |p: &mut Self| -> usize {
                    let from = p.at;
                    while matches!(p.s.get(p.at), Some(b'0'..=b'9')) {
                        p.at += 1;
                    }
                    p.at - from
                };
                let lead = self.s.get(self.at) == Some(&b'0');
                let n = digits(self);
                if n == 0 || (lead && n > 1) {
                    return Err(self.bad("a number was expected"));
                }
                if self.s.get(self.at) == Some(&b'.') {
                    self.at += 1;
                    if digits(self) == 0 {
                        return Err(self.bad("digits were expected"));
                    }
                }
                if matches!(self.s.get(self.at), Some(b'e' | b'E')) {
                    self.at += 1;
                    self.at += usize::from(matches!(self.s.get(self.at), Some(b'+' | b'-')));
                    if digits(self) == 0 {
                        return Err(self.bad("digits were expected"));
                    }
                }
                Ok(Json::Num(String::from_utf8_lossy(&self.s[start..self.at]).into_owned()))
            }
            _ => Err(self.bad("a value was expected")),
        }
    }

    fn hex4(&mut self) -> Res<u32> {
        let h = self.s.get(self.at..self.at + 4).and_then(|h| std::str::from_utf8(h).ok()).and_then(|h| u32::from_str_radix(h, 16).ok());
        self.at += 4;
        h.ok_or_else(|| self.bad("four hex digits were expected"))
    }

    fn string(&mut self) -> Res<String> {
        self.at += 1;
        let mut out = Vec::new();
        loop {
            let c = *self.s.get(self.at).ok_or_else(|| self.bad("the text does not end"))?;
            self.at += 1;
            match c {
                b'"' => return String::from_utf8(out).map_err(|_| self.bad("not UTF-8")),
                b'\\' => {
                    let e = *self.s.get(self.at).ok_or_else(|| self.bad("the text does not end"))?;
                    self.at += 1;
                    let c = match e {
                        b'"' | b'\\' | b'/' => e as char,
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'u' => {
                            let mut u = self.hex4()?;
                            if (0xd800..0xdc00).contains(&u) && self.s[self.at..].starts_with(b"\\u") {
                                self.at += 2;
                                let low = self.hex4()?;
                                u = 0x10000 + ((u - 0xd800) << 10) + (low.wrapping_sub(0xdc00) & 0x3ff);
                            }
                            char::from_u32(u).ok_or_else(|| self.bad("half of a character pair"))?
                        }
                        _ => return Err(self.bad("an escape was expected")),
                    };
                    out.extend(c.encode_utf8(&mut [0; 4]).as_bytes());
                }
                c if c < 0x20 => return Err(self.bad("a control character in a text")),
                c => out.push(c),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The texts are Python's own for these values (json.dumps, repr and str, run once on them).
    #[test]
    fn reads_and_writes_as_python() {
        let text = concat!(
            r#"{"a": [1, -2.50, 1e3, 1E-7, 12345678901234567890, 0.1, 1.5e16, 123456789.125], "#,
            r#""b": {"é": "x\ty\u00e9\ud834\udd1e\"\\/", "n": null}, "t": true, "f": false, "e": [], "o": {}, "a2": "it's"}"#,
        );
        let v = Json::parse(text).unwrap();
        assert_eq!(
            v.dump(None, true),
            concat!(
                r#"{"a": [1, -2.5, 1000.0, 1e-07, 12345678901234567890, 0.1, 1.5e+16, 123456789.125], "#,
                r#""b": {"\u00e9": "x\ty\u00e9\ud834\udd1e\"\\/", "n": null}, "t": true, "f": false, "e": [], "o": {}, "a2": "it's"}"#,
            )
        );
        assert_eq!(
            v.dump(Some(2), false),
            concat!(
                "{\n  \"a\": [\n    1,\n    -2.5,\n    1000.0,\n    1e-07,\n    12345678901234567890,\n    0.1,\n    1.5e+16,\n",
                "    123456789.125\n  ],\n  \"b\": {\n    \"é\": \"x\\ty\u{e9}\u{1d11e}\\\"\\\\/\",\n    \"n\": null\n  },\n",
                "  \"t\": true,\n  \"f\": false,\n  \"e\": [],\n  \"o\": {},\n  \"a2\": \"it's\"\n}",
            )
        );
        assert_eq!(
            v.repr(),
            concat!(
                "{'a': [1, -2.5, 1000.0, 1e-07, 12345678901234567890, 0.1, 1.5e+16, 123456789.125], 'b': {'é': 'x\\ty\u{e9}\u{1d11e}\"\\\\/', ",
                "'n': None}, 't': True, 'f': False, 'e': [], 'o': {}, 'a2': \"it's\"}",
            )
        );
        assert_eq!(
            (v.get("a2").unwrap().text(), v.get("t").unwrap().text(), v.get("b").unwrap().get("n").unwrap().text()),
            ("it's".to_string(), "True".to_string(), "None".to_string())
        );
        for bad in ["", "{", "[1,]", "{\"a\" 1}", "01", "1.", "nul", "\"a", "[1] 2", "{'a': 1}", "\"\\x\""] {
            assert!(Json::parse(bad).is_err(), "{bad}");
        }
        let n = |s: &str| Json::parse(s).unwrap();
        assert!(
            n("1").same(&n("1.0"))
                && n("true").same(&n("1"))
                && !n("1").same(&n("\"1\""))
                && n("{\"a\":1,\"b\":[2]}").same(&n("{\"b\":[2],\"a\":1.0}"))
                && !n("null").same(&n("0"))
        );
        assert!(!n("0").truthy() && !n("0.0").truthy() && !n("[]").truthy() && !n("\"\"").truthy() && n("[0]").truthy() && n("-1").truthy());
        assert_eq!(n("{\"a\":1,\"a\":2,\"b\":3}").dump(None, true), r#"{"a": 2, "b": 3}"#);
    }
}
