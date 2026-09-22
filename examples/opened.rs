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
        let mut raw = Vec::new();
        f(&d, &mut raw);
        println!("  plain at {name}: {} B (+ recipe = {} B, {:.1}% of the object); the object as it is at {name}: {} B ({:.1}%)", c.len(), c.len() + o.recipe.len(), 100.0 * (c.len() + o.recipe.len()) as f64 / d.len() as f64, raw.len(), 100.0 * raw.len() as f64 / d.len() as f64);
    };
    println!("{}: {} B; zstd -19 as is {} B; opened: plain {} B ({:.2}x the object), recipe {} B ({:.1}% of the object)", f.rsplit('/').next().unwrap(), d.len(), z.len(), o.plain.len(), o.plain.len() as f64 / d.len() as f64, o.recipe.len(), 100.0 * o.recipe.len() as f64 / d.len() as f64);
    // The levels on the plain text and on the raw bytes, the opener
    // switched off for the raw side by going through the plain
    // compressors' inner functions.
    level("max", |i, o| glyd::compress_parallel_into_max(i, o));
    level("ultra", |i, o| glyd::compress_parallel_into_ultra(i, o));
}
