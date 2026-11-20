#![no_main]
use libfuzzer_sys::fuzz_target;

// The request line and headers of an HTTP request, as the server reads them off a socket.
fuzz_target!(|data: &[u8]| {
    let mut r = std::io::Cursor::new(data);
    let _ = glyd_engine::server::read_head(&mut r);
});
