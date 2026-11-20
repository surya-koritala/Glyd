#![no_main]
use libfuzzer_sys::fuzz_target;

use glyd_engine::chat::{strftime, Civil};

// The `strftime_now` a chat template calls: any format, at any time the clock could give.
fuzz_target!(|data: &[u8]| {
    if data.len() < 12 {
        return;
    }
    // (the clock and the time zone give the time: a year within a few billions of ours, an offset of a day at most)
    let unix = i64::from_le_bytes(data[..8].try_into().unwrap()).rem_euclid(1 << 42) - (1 << 41);
    let offset = (i32::from_le_bytes(data[8..12].try_into().unwrap()) % 86_400) as i64;
    let Ok(fmt) = std::str::from_utf8(&data[12..]) else { return };
    let _ = strftime(fmt, &Civil::from_unix(unix, offset));
});
