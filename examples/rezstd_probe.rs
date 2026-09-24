// Scratch: a zstd frame decoded and written again: where the port's bytes first differ. `rezstd_probe <file.zst>`
fn main() {
    let f = std::env::args().nth(1).unwrap();
    let frame = std::fs::read(&f).unwrap();
    let t = std::time::Instant::now();
    let plain = glyd::rezstd::decompress(&frame).expect("decodes");
    let dt = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    let again = glyd::rezstd::compress(&plain, glyd::rezstd::BUILDS[0]);
    let ct = t.elapsed().as_secs_f64();
    let first = again.iter().zip(frame.iter()).position(|(a, b)| a != b).unwrap_or(again.len().min(frame.len()));
    println!("{}: {} B -> {} B plain (decode {dt:.2} s); written again {} B (encode {ct:.2} s); first difference at {first} ({})", f.rsplit('/').next().unwrap(), frame.len(), plain.len(), again.len(), if again == frame { "IDENTICAL" } else { "differs" });
}
