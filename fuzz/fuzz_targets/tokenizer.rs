#![no_main]
use libfuzzer_sys::fuzz_target;

use glyd_engine::constrain::Spec;
use glyd_engine::tok::Tok;

thread_local! {
    static DIR: glyd_fuzz::Scratch = glyd_fuzz::Scratch::new("tokenizer");
}

// A model directory's tokenizer: tokenizer.json, tokenizer_config.json, generation_config.json and tiktoken.model (parts of the input, each after a line holding a NUL), loaded, run on a text, and made into the
// token bytes that grammars compile against.
fuzz_target!(|data: &[u8]| {
    glyd_fuzz::tolerate_tokenizers_panics();
    let p = glyd_fuzz::parts(data, 4);
    DIR.with(|d| {
        d.put("tokenizer.json", p[0]);
        d.put("tokenizer_config.json", p[1]);
        d.put("generation_config.json", p[2]);
        d.put("tiktoken.model", p[3]);
        let Ok(t) = Tok::load(d.path()) else { return };
        let ids = t.encode("Hello, world! {\"a\": [1, 2.5, null]} \u{e9}\u{4e16}").unwrap_or_default();
        let _ = t.decode(&ids);
        let _ = t.decode(&[0, 1, 2, 3, 100, 70000]);
        let _ = t.vocab_size();
        let _ = t.added_ids();
        if let Ok(g) = t.grammars(t.vocab_size().min(1 << 16)) {
            let _ = g.compile(&Spec::Regex("[a-z]+ [0-9]{1,3}".to_string()));
        }
    });
});
