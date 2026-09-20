// Scratch: far-match coverage of one base unit (region ++ unit).
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let old = std::fs::read(&a[1]).unwrap();
    let new = std::fs::read(&a[2]).unwrap();
    let unit: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let (ua, ub) = (unit * (32 << 20), ((unit + 1) * (32 << 20)).min(new.len()));
    let (r0, r1) = (ua.saturating_sub(32 << 20), (ub + (32 << 20)).min(old.len() - 64));
    let mut full = old[r0..r1].to_vec();
    full.extend_from_slice(&new[ua..ub]);
    let start = r1 - r0;
    let t = std::time::Instant::now();
    let m = glyd::ldm::Matches::find(&full, false);
    let s = t.elapsed().as_secs_f64();
    let (mut cov_base, mut cov_self, mut n_base, mut gaps) = (0usize, 0usize, 0usize, Vec::new());
    let mut last_end = start;
    for f in &m.list {
        let (st, len, off) = (f.start as usize, f.len as usize, f.off as usize);
        if st < start { continue; }
        if st > last_end { gaps.push(st - last_end); }
        last_end = st + len;
        if st - off < start { cov_base += len; n_base += 1; } else { cov_self += len; }
    }
    gaps.sort_unstable();
    let med = gaps.get(gaps.len() / 2).copied().unwrap_or(0);
    println!("unit {} ({} B): far matches into the base cover {:.1}% ({} matches, mean {} B), into itself {:.1}%; {} gaps, median {} B, p90 {} B; pass {:.2} s", unit, ub - ua, 100.0 * cov_base as f64 / (ub - ua) as f64, n_base, cov_base / n_base.max(1), 100.0 * cov_self as f64 / (ub - ua) as f64, gaps.len(), med, gaps.get(gaps.len() * 9 / 10).copied().unwrap_or(0), s);
}
