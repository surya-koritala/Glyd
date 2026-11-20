#![no_main]
use libfuzzer_sys::fuzz_target;

use glyd_engine::{deepseek_v4, glm, kimi, model_gemma4, muse, nemotron, plan, qwen35, qwen4x, weights};

// A model's config.json (and its generation_config.json after a line holding a NUL) as the Hub serves it, read by every family's parser and by the planner that sizes a model from it.
fuzz_target!(|data: &[u8]| {
    let p = glyd_fuzz::parts(data, 2);
    let (Ok(text), Ok(gen)) = (std::str::from_utf8(p[0]), std::str::from_utf8(p[1])) else { return };
    let gen = (!gen.is_empty()).then_some(gen);
    let _ = weights::Config::parse(text, "config.json", gen);
    let _ = qwen35::Config35::parse(text, "config.json", gen);
    let _ = nemotron::ConfigNemotronH::parse(text, "config.json", gen);
    let _ = glm::ConfigGlm::parse(text, "config.json", gen);
    let _ = kimi::ConfigKimi::parse(text, "config.json", gen);
    let _ = muse::ConfigMuse::parse(text, "config.json", gen);
    let _ = qwen4x::ConfigQx::parse(text, "config.json", gen);
    let _ = deepseek_v4::ConfigV4::parse(text, "config.json", gen);
    let _ = model_gemma4::ConfigG4::parse(text, gen, false);
    let _ = model_gemma4::ConfigG4::parse(text, gen, true);
    if let Ok(shape) = plan::Shape::from_config(text) {
        let _ = shape.params();
        let _ = shape.is_moe();
        let _ = shape.routed_active();
        let _ = shape.clone().scaled_to(1e9);
        let m = plan::Machine::preset("dgx-spark").unwrap_or_else(plan::Machine::probe);
        let _ = plan::plan(&shape, &m, plan::Ask::default());
    }
});
