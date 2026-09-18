fn main() {
    let path = std::env::args().nth(1).expect("usage: decode_one <file>");
    let data = std::fs::read(&path).expect("read");
    eprintln!("decoding {} bytes", data.len());
    match simd_stream_codec::decompress(&data) {
        Ok(v) => eprintln!("OK: {} bytes out", v.len()),
        Err(e) => eprintln!("ERR: {:?}", e),
    }
    eprintln!("done");
}
