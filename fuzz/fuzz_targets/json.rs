#![no_main]
use libfuzzer_sys::fuzz_target;

// The JSON reader every other parser starts from, and the writers that print what it read.
fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else { return };
    if let Ok(v) = glyd_gpu::json::parse(text) {
        let compact = glyd_gpu::json::to_compact(&v);
        // what it writes it reads back as the same value
        assert_eq!(glyd_gpu::json::parse(&compact).as_ref(), Ok(&v));
        let _ = glyd_gpu::json::to_python(&v, 2);
    }
});
