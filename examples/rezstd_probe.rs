// Scratch: a zstd frame decoded and written again through every build: which one matched. `rezstd_probe <file.zst>...`
fn main() {
    for f in std::env::args().skip(1) {
        let frame = std::fs::read(&f).unwrap();
        let name = f.rsplit('/').next().unwrap();
        let t = std::time::Instant::now();
        match glyd::rezstd::reproduce(&frame) {
            Some((plain, build)) => println!("{name}: {} B -> {} B, reproduced by {build:?} in {:.2} s", frame.len(), plain.len(), t.elapsed().as_secs_f64()),
            None => {
                let Some(plain) = glyd::rezstd::decompress(&frame) else {
                    println!("{name}: does not decode");
                    continue;
                };
                println!("{name}: {} B -> {} B, NOT reproduced; per build, the first differing byte:", frame.len(), plain.len());
                for b in glyd::rezstd::BUILDS {
                    let b = glyd::rezstd::Build { checksum: frame[4] & 4 != 0, ..b };
                    let made = glyd::rezstd::compress(&plain, b);
                    let first = made.iter().zip(&frame).position(|(a, b)| a != b).unwrap_or(made.len().min(frame.len()));
                    println!("  {b:?}: {} B made, first difference at {first}", made.len());
                }
            }
        }
    }
}
