// A bucket compressed across its objects: for each object, in arrival
// order, the stored object sharing the most fingerprints (one sparse
// anchor in 4 KB) is its base, and it is compressed against that base
// (--max --base) when they share enough; else alone (--max). Totals
// against zstd -3 and Glyd --max per object.
//   bucket [--zstd19] file...   (files in arrival order)
use std::collections::HashMap;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let with_19 = args.iter().any(|a| a == "--zstd19");
    args.retain(|a| a != "--zstd19");
    let mut index: HashMap<u64, Vec<u32>> = HashMap::new(); // fingerprint -> the last 8 objects holding it
    let mut names: Vec<String> = Vec::new();
    let mut depth: Vec<usize> = Vec::new();
    let mut parent: Vec<Option<u32>> = Vec::new();
    // A base is kept when the delta saves a fifth of the object's own
    // size (a weak base is a read dependency for nothing), and a chain
    // is at most MAX_DEPTH long: past that, the chain's root is the base.
    const MAX_DEPTH: usize = 4;
    let (mut raw, mut z3, mut z19, mut alone, mut bucket) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut t_alone, mut t_bucket) = (0.0f64, 0.0f64);
    println!("{:<36} {:>9} {:>9} {:>9} {:>9} {:>9}  base", "object", "MB", "zstd -3", "Glyd max", "in bucket", "shared");
    for path in &args {
        let data = std::fs::read(path).unwrap();
        let name = path.rsplit('/').next().unwrap().to_string();
        // Fingerprints: the sparse anchors whose hash has two more zero bits (one in 4 KB).
        let mut anchors = Vec::new();
        glyd::ldm::sparse_anchors(&data, 0, &mut anchors);
        let prints: Vec<u64> = anchors.iter().filter(|&&(h, _)| h >> 62 == 0).map(|&(h, _)| h).collect();
        // The stored object sharing the most.
        let mut hits: HashMap<u32, f64> = HashMap::new();
        for h in &prints {
            if let Some(objs) = index.get(h) {
                for &o in objs {
                    *hits.entry(o).or_insert(0.0) += 1.0 / objs.len() as f64;
                }
            }
        }
        // The best score; among objects within 5% of it, the most recent
        // (a copy of a copy shares its fingerprints with both).
        let top = hits.values().cloned().fold(0.0f64, f64::max);
        let best = hits.iter().filter(|(_, &s)| s >= top * 0.95).max_by_key(|(&o, _)| o).map(|(&o, &s)| (o, s / prints.len().max(1) as f64));
        let t = std::time::Instant::now();
        let mut c = Vec::new();
        glyd::compress_parallel_into_max(&data, &mut c);
        t_alone += t.elapsed().as_secs_f64();
        let z = zstd_size(path, 3);
        let z9 = if with_19 { zstd_size(path, 19) } else { 0 };
        let t = std::time::Instant::now();
        let (stored, base) = match best {
            Some((mut o, share)) if share >= 0.02 => {
                if depth[o as usize] >= MAX_DEPTH {
                    while let Some(p) = parent[o as usize] {
                        o = p;
                    }
                }
                let base = std::fs::read(&args[o as usize]).unwrap();
                let mut d = Vec::new();
                glyd::compress_with_base(&base, &data, &mut d, false);
                if d.len() * 5 < c.len() * 4 { (d.len(), Some((o, share))) } else { (c.len(), None) }
            }
            _ => (c.len(), None),
        };
        t_bucket += t.elapsed().as_secs_f64();
        let id = names.len() as u32;
        for h in prints {
            let e = index.entry(h).or_default();
            if e.len() == 8 {
                e.remove(0);
            }
            e.push(id);
        }
        depth.push(base.map_or(0, |(o, _)| depth[o as usize] + 1));
        parent.push(base.map(|(o, _)| o));
        println!("{:<36} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>8.0}%  {}", name, data.len() as f64 / 1e6, z as f64 / 1e6, c.len() as f64 / 1e6, stored as f64 / 1e6, base.map_or(0.0, |(_, s)| s * 100.0), base.map_or("-".to_string(), |(o, _)| format!("{} (depth {})", names[o as usize], depth[id as usize])));
        names.push(name);
        raw += data.len(); z3 += z; z19 += z9; alone += c.len(); bucket += stored;
    }
    println!("\n{} objects, {:.2} GB raw: zstd -3 per object {:.1} MB ({:.2}x){}; Glyd --max per object {:.1} MB ({:.2}x, {:.0} MB/s); Glyd across the bucket {:.1} MB ({:.2}x, {:.0} MB/s) = {:.2}x over zstd -3, {:.2}x over Glyd alone",
        names.len(), raw as f64 / 1e9, z3 as f64 / 1e6, raw as f64 / z3 as f64, if with_19 { format!("; zstd -19 {:.1} MB ({:.2}x)", z19 as f64 / 1e6, raw as f64 / z19 as f64) } else { String::new() }, alone as f64 / 1e6, raw as f64 / alone as f64, raw as f64 / t_alone / 1e6, bucket as f64 / 1e6, raw as f64 / bucket as f64, raw as f64 / t_bucket / 1e6, z3 as f64 / bucket as f64, alone as f64 / bucket as f64);
}

fn zstd_size(path: &str, level: i32) -> usize {
    let out = std::process::Command::new("zstd").args([&format!("-{level}"), "-T10", "-c", path]).output().unwrap();
    out.stdout.len()
}
