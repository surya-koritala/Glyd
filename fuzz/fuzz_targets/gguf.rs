#![no_main]
use libfuzzer_sys::fuzz_target;

// The GGUF reader: header, metadata and tensor table, and what is made of them (the model's description, a config.json, the tokenizer's files).
fuzz_target!(|data: &[u8]| {
    if let Ok(g) = glyd_engine::gguf::Gguf::parse(data) {
        let _ = g.types();
        if let Ok(d) = g.description() {
            let _ = glyd_engine::gguf::config_json(&g, &d);
        }
        let _ = glyd_engine::gguf::generation_json(&g);
        let _ = glyd_engine::gguf::tokenizer_files(&g);
    }
    let _ = glyd_engine::gguf::Gguf::parse_header(data, data.len() as u64 + 1000);
});
