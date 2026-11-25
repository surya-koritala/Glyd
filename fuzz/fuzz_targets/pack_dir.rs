#![no_main]
use libfuzzer_sys::fuzz_target;

use glyd_engine::weights;

thread_local! {
    static DIR: glyd_fuzz::Scratch = glyd_fuzz::Scratch::new("pack_dir");
}

// A packed model directory as the engine loads it: config.json, glyd.json (the manifest), the checkpoint's shard and engine.safetensors (the embedding's sidecar), parts of the input, each after a line holding a
// NUL. Its plan is made from the manifest and the files' headers alone.
fuzz_target!(|data: &[u8]| {
    let p = glyd_fuzz::parts(data, 4);
    DIR.with(|d| {
        d.put("config.json", p[0]);
        d.put("glyd.json", p[1]);
        d.put("model.safetensors", p[2]);
        d.put("engine.safetensors", p[3]);
        let _ = weights::v7_manifest(d.path());
        let _ = weights::plan(d.path());
        let _ = weights::earlier_format(d.path());
    });
});
