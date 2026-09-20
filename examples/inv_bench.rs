// Scratch: record inverse throughput per column type (2M single-field lines).
fn bench(name: &str, data: &[u8]) {
    let img = glyd::record::transform(data).expect("shape");
    let mut out = Vec::new();
    let t = std::time::Instant::now();
    let mut k = 0;
    while t.elapsed().as_secs_f64() < 1.0 { glyd::record::inverse_into(&img, &mut out).unwrap(); k += 1; }
    assert!(out == data);
    let s = t.elapsed().as_secs_f64() / k as f64;
    println!("{:8} {:>10} B  image {:>9} B  inverse {:>6.0} MB/s  {:>5.1} ns/value", name, data.len(), img.len(), data.len() as f64 / s / 1e6, s * 1e9 / (data.len() as f64 / 12.0));
}
fn main() {
    let n = 2_000_000u64;
    let mut x = 7u64;
    let mut rnd = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x };
    let ints: Vec<u8> = (0..n).map(|i| format!("{},{}\n", 1_000_000 + i * 3, rnd() % 1000)).collect::<String>().into_bytes();
    bench("int", &ints);
    let d8: Vec<u8> = (0..n).map(|_| format!("host{},{}\n", rnd() % 40, rnd() % 7)).collect::<String>().into_bytes();
    bench("dict8", &d8);
    let dict: Vec<u8> = (0..n).map(|_| format!("/path/item{},{}\n", rnd() % 30000, rnd() % 5000)).collect::<String>().into_bytes();
    bench("dict", &dict);
    let text: Vec<u8> = (0..n).map(|_| format!("u{:x},{}\n", rnd(), rnd() % 3)).collect::<String>().into_bytes();
    bench("text", &text);
    let time: Vec<u8> = (0..n).map(|i| format!("2024-01-15T{:02}:{:02}:{:02}Z,{}\n", (i / 3600) % 24, (i / 60) % 60, i % 60, rnd() % 9)).collect::<String>().into_bytes();
    bench("time", &time);
    let dec: Vec<u8> = (0..n).map(|_| format!("{}.{:02},{}\n", rnd() % 100000, rnd() % 100, rnd() % 4)).collect::<String>().into_bytes();
    bench("decimal", &dec);
}
