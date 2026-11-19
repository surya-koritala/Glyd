#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../check.rs"]
mod input;

fuzz_target!(|data: &[u8]| input::check(data));
