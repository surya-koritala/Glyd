#![no_main]
use libfuzzer_sys::fuzz_target;

use glyd_engine::sampler::Params;
use glyd_engine::server;
use glyd_engine::tools;
use glyd_gpu::json;

// A chat completions request's body: what the server checks in it, the grammar it asks for (a schema, a regular expression, a Lark grammar) compiled, the sampling it sets, the Anthropic form
// turned into OpenAI's, and its tools and messages read.
fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else { return };
    let Ok(v) = json::parse(text) else { return };
    let _ = server::invalid(&v, true, 1000);
    let _ = server::invalid(&v, false, 1000);
    let _ = server::unsupported(&v);
    let _ = server::logit_bias_of(&v, 1000);
    let _ = server::request_of(&v, 4096, &Params::default());
    if let Ok(Some(spec)) = server::constraint_spec(&v) {
        let _ = glyd_fuzz::grammars().compile(&spec);
    }
    if let Ok(Some(t)) = tools::tools_of(&v) {
        let _ = t;
    }
    let _ = tools::template_messages(&v);
    if let Ok(o) = server::openai_of_with(&v, false) {
        let _ = server::invalid(&o, true, 1000);
        let _ = tools::tools_of(&o);
        let _ = tools::template_messages(&o);
    }
});
