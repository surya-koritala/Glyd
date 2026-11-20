#![no_main]
use libfuzzer_sys::fuzz_target;

use glyd_engine::constrain::Spec;

// A grammar as a request gives it, compiled by llguidance against a small vocabulary, and a few tokens run through it: a JSON schema, a regular expression, a Lark grammar, a choice.
fuzz_target!(|data: &[u8]| {
    let Some((&kind, rest)) = data.split_first() else { return };
    let Ok(text) = std::str::from_utf8(rest) else { return };
    let spec = match kind % 5 {
        0 => Spec::JsonSchema(text.to_string()),
        1 => Spec::Regex(text.to_string()),
        2 => Spec::Grammar(text.to_string()),
        3 => Spec::Choice(text.split('\n').map(str::to_string).collect()),
        _ => Spec::JsonObject,
    };
    let Ok(c) = glyd_fuzz::grammars().compile(&spec) else { return };
    let mut g = c.start();
    for t in 0..24u32 {
        let allowed = g.allowed();
        // the first token the grammar allows, else stop
        match allowed.iter().position(|&a| a) {
            Some(i) => g.consume(((i as u32) + t * 7) % allowed.len() as u32),
            None => break,
        }
        if g.accepting() {
            break;
        }
    }
});
