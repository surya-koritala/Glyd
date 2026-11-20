#![no_main]
use libfuzzer_sys::fuzz_target;

// A machine described in JSON (`glyd fit --machine FILE`).
fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else { return };
    if let Ok(m) = glyd_engine::plan::Machine::from_json(text) {
        let shape = glyd_engine::plan::Shape::from_config(
            r#"{"num_hidden_layers": 4, "hidden_size": 256, "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 512, "vocab_size": 1000}"#,
        );
        if let Ok(s) = shape {
            let _ = glyd_engine::plan::plan(&s, &m, glyd_engine::plan::Ask::default());
        }
    }
});
