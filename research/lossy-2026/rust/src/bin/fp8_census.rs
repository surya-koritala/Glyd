//! Whether fixed Candidate A values are exactly values of a small float format (e4m3fn, or narrower fixed-width
//! formats with three fraction bits) times a power-of-two block scale. Counts only: no rounding beyond Candidate A,
//! no scale search beyond the smallest covering scale, and no model artifact. Run on CI, never a workstation.
use glyd_lossy_study::approximate;
use std::{env, error::Error, fs::{self, File}, io::{BufReader, Read, Seek, SeekFrom}};

/// Block shapes, output rows by input columns: the official checkpoint's 128 by 128, finer rows, then a whole row
/// (0: every column; one scale a row, as a per-channel product applies after its sums).
const GEOMETRIES: [(usize, usize); 4] = [(128, 128), (1, 128), (1, 32), (1, 0)];

/// A format with a sign, `ebits` exponent bits (IEEE-style subnormals, the given bias) and three fraction bits:
/// largest magnitude MAX_M * 2^MAX_P, subnormal step 2^SUB. e4m3fn's top code with fraction 7 is NaN.
#[derive(Clone, Copy, Debug)]
struct Format { name: &'static str, max_m: u32, max_p: i32, sub: i32 }
const fn format(name: &'static str, ebits: i32, bias: i32, nan_top: bool) -> Format {
    let top = (1 << ebits) - 1;
    if nan_top { Format { name, max_m: 7, max_p: top - bias - 2, sub: 1 - bias - 3 } }
    else { Format { name, max_m: 15, max_p: top - bias - 3, sub: 1 - bias - 3 } }
}
/// FP8 itself (8 bits), and 7- and 6-bit formats that become e4m3 by an exponent offset.
const FORMATS: [Format; 3] = [format("e4m3fn", 4, 7, true), format("e3m3", 3, 3, false), format("e2m3", 2, 1, false)];

/// A finite Candidate A word as significand M (0..=15) and power P: |value| = M * 2^P.
fn split(w: u16) -> Result<Option<(u32, i32)>, Box<dyn Error>> {
    let e = ((w >> 7) & 0xff) as i32;
    if e == 0xff { return Ok(None); }
    if w & 15 != 0 { return Err("finite word is not a Candidate A value".into()); }
    let m = ((w & 0x7f) >> 4) as u32;
    Ok(Some(if e == 0 { (m, -129) } else { (8 + m, e - 130) }))
}

/// Smallest k with M * 2^(P-k) at most the format's largest magnitude, for M >= 1: P - MAX_P - t, t the largest
/// integer with M * 2^t <= MAX_M.
fn smallest_scale(f: Format, m: u32, p: i32) -> i32 {
    let mut t: i32 = 0;
    if m <= f.max_m {
        while m << (t + 1) <= f.max_m { t += 1; }
    } else {
        t = -1;
        while m > f.max_m << -t { t -= 1; }
    }
    p - f.max_p - t
}

/// Exact after dividing by 2^k (k at least the smallest scale): at most four significant bits always hold,
/// so only the lowest set bit must reach the subnormal step.
fn exact(f: Format, m: u32, p: i32, k: i32) -> bool { m == 0 || p + m.trailing_zeros() as i32 >= k + f.sub }

/// Below the smallest subnormal after scaling: the value would round to zero or that step.
fn underflow(f: Format, m: u32, p: i32, k: i32) -> bool { m != 0 && p + (31 - m.leading_zeros() as i32) < k + f.sub }

#[derive(Clone, Debug, Default)]
struct Geometry { blocks: u64, blocks_inexact: u64, inexact: u64, underflow: u64, max_block_inexact: u64, min_scale: i32, max_scale: i32 }

#[derive(Debug, Default)]
struct Census { values: u64, zeros: u64, specials: u64, geometries: Vec<Geometry> }

fn block(f: Format, values: &[Option<(u32, i32)>], cols: usize, rows: std::ops::Range<usize>, span: std::ops::Range<usize>, g: &mut Geometry) {
    let cells = || rows.clone().flat_map(|r| span.clone().map(move |c| r * cols + c));
    g.blocks += 1;
    let Some(k) = cells().filter_map(|i| values[i]).filter(|v| v.0 != 0).map(|(m, p)| smallest_scale(f, m, p)).max() else { return };
    let (mut bad, mut low) = (0, 0);
    for (m, p) in cells().filter_map(|i| values[i]) {
        bad += !exact(f, m, p, k) as u64;
        low += underflow(f, m, p, k) as u64;
    }
    g.min_scale = g.min_scale.min(k);
    g.max_scale = g.max_scale.max(k);
    g.inexact += bad;
    g.underflow += low;
    g.blocks_inexact += (bad != 0) as u64;
    g.max_block_inexact = g.max_block_inexact.max(bad);
}

fn census<R: Read>(reader: &mut R, rows: usize, cols: usize) -> Result<Census, Box<dyn Error>> {
    let mut out = Census { geometries: vec![Geometry { min_scale: i32::MAX, max_scale: i32::MIN, ..Default::default() }; FORMATS.len() * GEOMETRIES.len()], ..Default::default() };
    let height = GEOMETRIES.iter().map(|g| g.0).max().unwrap();
    let mut raw = vec![0u8; height * cols * 2];
    for r0 in (0..rows).step_by(height) {
        let n = height.min(rows - r0);
        reader.read_exact(&mut raw[..n * cols * 2])?;
        let values = raw[..n * cols * 2].chunks_exact(2)
            .map(|b| split(approximate(u16::from_le_bytes([b[0], b[1]]), 3)))
            .collect::<Result<Vec<_>, _>>()?;
        out.values += values.len() as u64;
        out.specials += values.iter().filter(|v| v.is_none()).count() as u64;
        out.zeros += values.iter().filter(|v| matches!(v, Some((0, _)))).count() as u64;
        for (i, g) in out.geometries.iter_mut().enumerate() {
            let (f, (bh, bw)) = (FORMATS[i / GEOMETRIES.len()], GEOMETRIES[i % GEOMETRIES.len()]);
            for b0 in (0..n).step_by(bh) {
                let bw = if bw == 0 { cols } else { bw };
                for c0 in (0..cols).step_by(bw) {
                    block(f, &values, cols, b0..(b0 + bh).min(n), c0..(c0 + bw).min(cols), g);
                }
            }
        }
    }
    Ok(out)
}

impl Census {
    fn json(&self) -> String {
        let geometries: Vec<String> = self.geometries.iter().enumerate().map(|(i, g)| (FORMATS[i / GEOMETRIES.len()].name, GEOMETRIES[i % GEOMETRIES.len()], g)).map(|(name, (bh, bw), g)| format!(
            "{{\"format\":\"{name}\",\"block\":[{bh},{bw}],\"blocks\":{},\"blocks_inexact\":{},\"inexact\":{},\"underflow\":{},\"max_block_inexact\":{},\"min_scale_exponent\":{},\"max_scale_exponent\":{}}}",
            g.blocks, g.blocks_inexact, g.inexact, g.underflow, g.max_block_inexact,
            if g.min_scale == i32::MAX { 0 } else { g.min_scale }, if g.max_scale == i32::MIN { 0 } else { g.max_scale })).collect();
        format!("{{\"fraction_bits\":3,\"values\":{},\"zeros\":{},\"specials\":{},\"geometries\":[{}]}}\n",
            self.values, self.zeros, self.specials, geometries.join(","))
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<String> = env::args().collect();
    if a.len() != 7 { return Err("usage: glyd-fp8-census FILE OFFSET BYTES ROWS COLS OUTPUT.json".into()); }
    let (offset, bytes, rows, cols): (u64, u64, usize, usize) = (a[2].parse()?, a[3].parse()?, a[4].parse()?, a[5].parse()?);
    if rows == 0 || cols == 0 || (rows as u64).checked_mul(cols as u64 * 2) != Some(bytes) { return Err("matrix shape differs from its bytes".into()); }
    let file = File::open(&a[1])?;
    if offset.checked_add(bytes).ok_or("tensor range overflow")? > file.metadata()?.len() { return Err("tensor extends past input".into()); }
    let mut reader = BufReader::with_capacity(1 << 20, file);
    reader.seek(SeekFrom::Start(offset))?;
    fs::write(&a[6], census(&mut reader, rows, cols)?.json())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Every non-negative finite value of a format with `ebits` exponent bits, three fraction bits and the bias.
    fn values(ebits: i32, bias: i32, nan_top: bool) -> HashSet<u64> {
        let top = (1 << ebits) - 1;
        (0..=top).flat_map(|e| (0..8).map(move |m| (e, m))).filter(|&(e, m)| !(nan_top && e == top && m == 7)).map(|(e, m)| {
            (if e == 0 { m as f64 * 2f64.powi(1 - bias - 3) } else { (8 + m) as f64 * 2f64.powi(e - bias - 3) }).to_bits()
        }).collect()
    }

    #[test]
    fn representability_matches_enumeration_for_every_word() {
        for (f, (ebits, bias, nan_top)) in FORMATS.iter().zip([(4, 7, true), (3, 3, false), (2, 1, false)]) {
            let set = values(ebits, bias, nan_top);
            assert_eq!(set.len(), (8usize << ebits) - nan_top as usize, "{}", f.name);
            let (largest, step) = (f.max_m as f64 * 2f64.powi(f.max_p), 2f64.powi(f.sub));
            assert!(set.contains(&largest.to_bits()) && set.iter().all(|&b| f64::from_bits(b) <= largest), "{}", f.name);
            for w in 0..=u16::MAX {
                let Some((m, p)) = split(approximate(w, 3)).unwrap() else { continue };
                let v = m as f64 * 2f64.powi(p);
                if m == 0 { assert!(exact(*f, m, p, 0)); continue; }
                let k0 = smallest_scale(*f, m, p);
                assert!(v / 2f64.powi(k0) <= largest && v / 2f64.powi(k0 - 1) > largest, "{} scale not smallest for {w:#x}", f.name);
                for k in k0..k0 + 160 {
                    let q = v / 2f64.powi(k);
                    assert_eq!(exact(*f, m, p, k), set.contains(&q.to_bits()), "{} word {w:#x} scale {k}", f.name);
                    assert_eq!(underflow(*f, m, p, k), q < step, "{} word {w:#x} scale {k}", f.name);
                }
            }
        }
    }

    #[test]
    fn block_counts() {
        // Row 0: 1.0, then 2^-20 at column 1; every other value is 2^-3. Two rows of 64 columns.
        let mut words = vec![0x3e00u16; 128];
        words[0] = 0x3f80;
        words[1] = 107 << 7;
        words[64] = 0x7fc0; // NaN, retained unrounded
        words[65] = 0;
        let data: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let c = census(&mut &data[..], 2, 64).unwrap();
        assert_eq!((c.values, c.zeros, c.specials), (128, 1, 1));
        let g: Vec<_> = c.geometries.iter().map(|g| (g.blocks, g.blocks_inexact, g.inexact, g.underflow)).collect();
        assert_eq!(g, [(1, 1, 1, 1), (2, 1, 1, 1), (4, 1, 1, 1), (2, 1, 1, 1)].repeat(FORMATS.len()));
        assert!(census(&mut &data[..], 2, 65).is_err());
        let unrounded = 0x3f81u16.to_le_bytes();
        assert!(split(0x3f81).is_err() && census(&mut &unrounded[..], 1, 1).is_ok());
    }
}
