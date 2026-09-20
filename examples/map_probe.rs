// Scratch: how a unit's sparse anchors land in the base's map.
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let base = std::fs::read(&a[1]).unwrap();
    let new = std::fs::read(&a[2]).unwrap();
    let mut map = Vec::new();
    glyd::ldm::sparse_anchors(&base, 0, &mut map);
    map.sort_unstable();
    println!("base {} MB: {} anchors", base.len() >> 20, map.len());
    for (a, b) in (0..new.len()).step_by(32 << 20).map(|a| (a, (a + (32 << 20)).min(new.len()))) {
        let mut an = Vec::new();
        glyd::ldm::sparse_anchors(&new[a..b], 0, &mut an);
        let mut ks = [0usize; 6]; // k = 0, 1, 2, 3-4, 5-16, >16
        for (h, _) in &an {
            let lo = map.partition_point(|e| e.0 < *h);
            let hi = map.partition_point(|e| e.0 <= *h);
            let k = hi - lo;
            ks[match k { 0 => 0, 1 => 1, 2 => 2, 3..=4 => 3, 5..=16 => 4, _ => 5 }] += 1;
        }
        println!("unit {:>4} MB: {:>6} anchors; k=0 {:>6} k=1 {:>6} k=2 {:>5} k=3-4 {:>5} k=5-16 {:>5} k>16 {:>5}", a >> 20, an.len(), ks[0], ks[1], ks[2], ks[3], ks[4], ks[5]);
    }
}
