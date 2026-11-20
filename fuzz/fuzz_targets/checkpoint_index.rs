#![no_main]
use libfuzzer_sys::fuzz_target;

thread_local! {
    static DIR: glyd_fuzz::Scratch = glyd_fuzz::Scratch::new("index");
}

// A sharded checkpoint: model.safetensors.index.json, config.json (its quantization_config) and two shards (parts of the input, each after a line holding a NUL).
fuzz_target!(|data: &[u8]| {
    let p = glyd_fuzz::parts(data, 4);
    DIR.with(|d| {
        d.put("model.safetensors.index.json", p[0]);
        d.put("config.json", p[1]);
        d.put("model-00001-of-00002.safetensors", p[2]);
        d.put("model-00002-of-00002.safetensors", p[3]);
        d.put("model.safetensors", b"");
        if let Ok(c) = glyd_gpu::safetensors::Checkpoint::open(d.path()) {
            let names: Vec<String> = c.names().map(str::to_string).collect();
            for n in names.iter().take(8) {
                let _ = c.matrix(n);
            }
        }
    });
});
