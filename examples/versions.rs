// How much of a new version of an object is new: content-defined chunk
// dedup against the old version (gear hash, mean CHUNK bytes) with the
// stored chunks zstd -19, and zstd -19 --patch-from (byte-level delta
// with the old version as the window, via the zstd CLI). Byte-exact.
use std::collections::HashMap;
fn chunks(data: &[u8], mean: usize) -> Vec<(usize, usize)> {
    let mut gear = [0u64; 256];
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    for g in gear.iter_mut() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; *g = x; }
    let bits = mean.trailing_zeros();
    let mask = ((1u64 << bits) - 1) << (64 - bits);
    let (lo, hi) = (mean / 4, mean * 4);
    let mut out = Vec::new();
    let (mut start, mut h) = (0usize, 0u64);
    for i in 0..data.len() {
        h = (h << 1).wrapping_add(gear[data[i] as usize]);
        let len = i + 1 - start;
        if (len >= lo && h & mask == 0) || len >= hi { out.push((start, i + 1)); start = i + 1; h = 0; }
    }
    if start < data.len() { out.push((start, data.len())); }
    out
}
fn sha(b: &[u8]) -> [u8; 16] {
    // A 128-bit content key: two independent 64-bit multiplicative hashes over 8-byte words (a stand-in for SHA-256 in this measurement).
    let (mut a, mut c) = (0x9E37_79B9_7F4A_7C15u64, 0xC2B2_AE3D_27D4_EB4Fu64);
    for w in b.chunks(8) {
        let mut buf = [0u8; 8]; buf[..w.len()].copy_from_slice(w);
        let v = u64::from_le_bytes(buf);
        a = (a ^ v).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29);
        c = (c.rotate_left(17) ^ v).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    }
    let mut k = [0u8; 16]; k[..8].copy_from_slice(&a.to_le_bytes()); k[8..].copy_from_slice(&c.to_le_bytes()); k
}
fn z19(b: &[u8]) -> usize { if b.is_empty() { 0 } else { zstd::bulk::compress(b, 19).unwrap().len() } }
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (old_p, new_p) = (&a[1], &a[2]);
    let mean: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(8192);
    let old = std::fs::read(old_p).unwrap(); let new = std::fs::read(new_p).unwrap();
    let oc = chunks(&old, mean); let nc = chunks(&new, mean);
    let mut have: HashMap<[u8; 16], (usize, usize)> = HashMap::with_capacity(oc.len());
    for &(s, e) in &oc { have.entry(sha(&old[s..e])).or_insert((s, e)); }
    let mut stored = Vec::new(); let mut refs: Vec<(bool, usize, usize)> = Vec::new(); let mut new_bytes = 0usize;
    for &(s, e) in &nc {
        let k = sha(&new[s..e]);
        match have.get(&k) {
            Some(&(a, b)) if old[a..b] == new[s..e] => refs.push((true, a, b - a)),
            _ => { refs.push((false, stored.len(), e - s)); stored.extend_from_slice(&new[s..e]); new_bytes += e - s; }
        }
    }
    let mut back = Vec::with_capacity(new.len());
    for &(o, s, l) in &refs { back.extend_from_slice(if o { &old[s..s + l] } else { &stored[s..s + l] }); }
    assert!(back == new, "chunk rebuild differs");
    let mut manifest = Vec::new();
    for &(o, s, l) in &refs { manifest.push(o as u8); manifest.extend_from_slice(&(s as u64).to_le_bytes()); manifest.extend_from_slice(&(l as u32).to_le_bytes()); }
    let (zn, zs, zm) = (z19(&new), z19(&stored), z19(&manifest));
    let wl = ((old.len().max(new.len()) - 1).max(1 << 20)).ilog2() as usize + 1;
    let wl = wl.clamp(20, 31);
    let patch = std::process::Command::new("zstd").args(["-q", "-19", "-T1", &format!("--long={wl}"), &format!("--patch-from={old_p}"), "-c", new_p]).output().unwrap();
    assert!(patch.status.success(), "zstd patch failed: {}", String::from_utf8_lossy(&patch.stderr));
    let patch = patch.stdout;
    let tmp = std::env::temp_dir().join("glyd_patch.zst");
    std::fs::write(&tmp, &patch).unwrap();
    let back = std::process::Command::new("zstd").args(["-q", "-d", &format!("--long={wl}"), &format!("--patch-from={old_p}"), "-c", tmp.to_str().unwrap()]).output().unwrap().stdout;
    assert!(back == new, "patch rebuild differs");
    let name = |p: &str| p.rsplit('/').next().unwrap().to_string();
    println!("{} -> {}: {} B; chunks {} (mean {} B); bytes not in the old version: {:.1}%", name(old_p), name(new_p), new.len(), nc.len(), new.len() / nc.len().max(1), 100.0 * new_bytes as f64 / new.len() as f64);
    println!("   new alone, zstd -19:            {:>14} B ({:.1}x)", zn, new.len() as f64 / zn as f64);
    println!("   chunk dedup vs old + zstd -19:  {:>14} B ({:.1}x)  -> {:.1}x smaller than alone (manifest {} B)", zs + zm, new.len() as f64 / (zs + zm) as f64, zn as f64 / (zs + zm) as f64, zm);
    println!("   zstd -19 --patch-from old:      {:>14} B ({:.1}x)  -> {:.1}x smaller than alone", patch.len(), new.len() as f64 / patch.len() as f64, zn as f64 / patch.len() as f64);
}
