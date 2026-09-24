// Scratch: what opening a container costs and yields. `open_probe <file>`.
fn main() {
    let f = std::env::args().nth(1).unwrap();
    let d = std::fs::read(&f).unwrap();
    let t = std::time::Instant::now();
    let is = glyd::deflate::is_container(&d);
    let o = glyd::deflate::open(&d);
    let secs = t.elapsed().as_secs_f64();
    match o {
        Some(o) => println!("{}: container {is}; opened in {secs:.2} s: plain {} B (input {} B), recipe {} B", f.rsplit('/').next().unwrap(), o.plain.len(), d.len(), o.recipe.len()),
        None => println!("{}: container {is}; open returned None in {secs:.2} s", f.rsplit('/').next().unwrap()),
    }
}
