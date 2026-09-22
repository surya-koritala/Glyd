// Scratch: what opening a deflate container yields: the plain text and
// recipe sizes, and the plain text alone at each level, against the
// object as it is.
fn main() {
    let f = std::env::args().nth(1).expect("a file");
    let d = std::fs::read(&f).unwrap();
    let Some(o) = glyd::deflate::open(&d) else {
        println!("{f}: does not open");
        return;
    };
    let mut z = Vec::new();
    zstd::stream::copy_encode(&d[..], &mut z, 19).unwrap();
    let level = |name: &str, f: fn(&[u8], &mut Vec<u8>)| {
        let mut c = Vec::new();
        f(&o.plain, &mut c);
        println!("  plain at {name}: {} B (+ recipe = {} B, {:.1}% of the object)", c.len(), c.len() + o.recipe.len(), 100.0 * (c.len() + o.recipe.len()) as f64 / d.len() as f64);
    };
    println!("{}: {} B; zstd -19 as is {} B; opened: plain {} B ({:.2}x the object), recipe {} B ({:.1}% of the object)", f.rsplit('/').next().unwrap(), d.len(), z.len(), o.plain.len(), o.plain.len() as f64 / d.len() as f64, o.recipe.len(), 100.0 * o.recipe.len() as f64 / d.len() as f64);
    level("max", glyd::compress_parallel_into_max);
    level("max -r", glyd::compress_records_into_max);
    level("ultra", glyd::compress_parallel_into_ultra);
    level("cold", glyd::compress_parallel_into_cold);
}
