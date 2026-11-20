#![no_main]
use libfuzzer_sys::fuzz_target;

thread_local! {
    static DIR: glyd_fuzz::Scratch = glyd_fuzz::Scratch::new("safetensors");
}

// A safetensors file: the header's length, its JSON, and the tensors' offsets, checked by the engine's reader (from a file) and by the codec's (from bytes).
fuzz_target!(|data: &[u8]| {
    let _ = glyd::safetensors::is_safetensors(data);
    let _ = glyd::safetensors::tensors(data);
    let _ = glyd::safetensors::tensors_typed(data);
    DIR.with(|d| {
        d.put("model.safetensors", data);
        if let Ok(f) = glyd_gpu::safetensors::File::open(&d.path().join("model.safetensors")) {
            for (_, t) in &f.tensors {
                assert!(t.offset + t.bytes <= data.len() as u64);
            }
        }
        let _ = glyd_gpu::safetensors::Checkpoint::open(d.path());
    });
});
