#![no_main]
use libfuzzer_sys::fuzz_target;

use glyd_engine::tools::{self, Format, ToolStream};
use glyd_gpu::json;

// A reply's tool calls read back out of the model's text, whole and as the stream's pieces come, in each format's tags, against the tools of a request (after a line holding a NUL).
fuzz_target!(|data: &[u8]| {
    let p = glyd_fuzz::parts(data, 2);
    let Ok(text) = std::str::from_utf8(p[0]) else { return };
    let tools: Vec<json::Value> = std::str::from_utf8(p[1])
        .ok()
        .and_then(|t| json::parse(t).ok())
        .and_then(|v| v.as_array().map(|a| a.to_vec()))
        .unwrap_or_else(|| vec![json::parse(r#"{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}}}}"#).unwrap()]);
    for format in [Format::Hermes, Format::Xml, Format::Dsml, Format::Glm] {
        let _ = tools::extract(text, format, &tools);
        let mut s = ToolStream::new(format, &tools);
        // the text in pieces of up to seven characters, as a stream gives it
        let mut piece = String::new();
        for (i, c) in text.chars().enumerate() {
            piece.push(c);
            if i % 7 == 6 {
                let _ = s.feed(&piece);
                piece.clear();
            }
        }
        let _ = s.feed(&piece);
        let _ = s.finish();
    }
});
