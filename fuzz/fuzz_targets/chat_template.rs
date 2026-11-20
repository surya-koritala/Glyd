#![no_main]
use libfuzzer_sys::fuzz_target;

use glyd_engine::chat::{Chat, Options};
use glyd_gpu::json;

// A chat template as a model's repository ships it, and the messages and tools of a request rendered through it (the template is split from the messages and the tools by a line holding a NUL).
fuzz_target!(|data: &[u8]| {
    let p = glyd_fuzz::parts(data, 3);
    let Ok(src) = std::str::from_utf8(p[0]) else { return };
    let Ok(chat) = Chat::from_template(src) else { return };
    let messages = std::str::from_utf8(p[1]).ok().and_then(|t| json::parse(t).ok()).unwrap_or_else(|| {
        json::parse(r#"[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f", "arguments": {"x": 1}}}]}, {"role": "tool", "tool_call_id": "1", "content": "ok"}]"#).unwrap()
    });
    let tools = std::str::from_utf8(p[2]).ok().and_then(|t| json::parse(t).ok()).and_then(|v| v.as_array().map(|a| a.to_vec()));
    let o = Options { add_generation_prompt: true, tools, ..Options::default() };
    let _ = chat.render_json(&messages, &o);
    let _ = chat.render_json(&messages, &Options::new(false).thinking(Some(false)));
});
