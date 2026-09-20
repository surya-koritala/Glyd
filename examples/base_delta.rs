// Scratch: compress a new version against its base with Glyd (max, ultra
// with a flag), byte-exact, timed; alongside the plain size.
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let old = std::fs::read(&a[1]).unwrap();
    let new = std::fs::read(&a[2]).unwrap();
    let ultra = a.get(3).map_or(false, |s| s == "ultra");
    let mut c = Vec::new();
    let t = std::time::Instant::now();
    glyd::compress_with_base(&old, &new, &mut c, ultra);
    let ct = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    let back = glyd::decompress_with_base(&old, &c).unwrap();
    let dt = t.elapsed().as_secs_f64();
    assert!(back == new, "round trip differs");
    let mut plain = Vec::new();
    glyd::compress_parallel_into_max(&new, &mut plain);
    println!("{} -> {}: base {}: {} B ({:.1}x; plain --max {} B) compress {:.2} s ({:.0} MB/s) decompress {:.2} s ({:.0} MB/s), exact", a[1].rsplit('/').next().unwrap(), a[2].rsplit('/').next().unwrap(), if ultra { "ultra" } else { "max" }, c.len(), new.len() as f64 / c.len() as f64, plain.len(), ct, new.len() as f64 / ct / 1e6, dt, new.len() as f64 / dt / 1e6);
}
