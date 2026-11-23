//! A known answer for the lossless KV cache of every head dimension the library holds (128, 256 and 512 channels; 256 and 512 are held as two and four of the page's heads of 128): bf16 keys and values made on the host
//! (in the windows of their channels, and out of them, and with every kind of special value), written through `glyd_gpu_kv_write`, read back through `glyd_gpu_kv_gather` and compared bit for bit with what was written, and
//! the decode attention (`glyd_gpu_kv_decode`) against attention computed on the host from the same values. No model: `glyd-engine kvshapes`, in the GPU job.

use crate::dev::{Buf, Dev};
use crate::ffi::{KvGeom, Lib, Result, KV_ONE_STREAM};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn f32_of(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// The bf16 nearest `x` (ties to even).
fn bf16_of(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

/// Special values: not-a-numbers with payloads, both infinities, subnormals, both zeros, the largest numbers.
const SPECIALS: [u16; 12] = [0x7FC0, 0x7F81, 0xFFC1, 0x7F80, 0xFF80, 0x0001, 0x8001, 0x007F, 0x0000, 0x8000, 0x7F7F, 0xFF7F];

/// `n` tokens of [h][d] values: each channel around an exponent of its own (so that the tables' windows hold most of them), `far` in a thousand far from it (finite: the decode attention reads them), and where `wild`,
/// as many in a thousand of the `SPECIALS`.
fn values(r: &mut Rng, n: usize, h: usize, d: usize, far: usize, wild: usize) -> Vec<u16> {
    let mut out = Vec::with_capacity(n * h * d);
    for _ in 0..n {
        for hh in 0..h {
            for c in 0..d {
                let mu = 124 + (hh * 5 + c * 3) % 7; // (a biased exponent: values of 0.5 to 16)
                let mut e = mu + r.below(3) - 1;
                if r.below(1000) < far {
                    e = [3, 20, 60, 200, 240][r.below(5)] + r.below(10);
                }
                let v = ((r.below(2) as u16) << 15) | ((e as u16) << 7) | r.below(128) as u16;
                out.push(if wild > 0 && r.below(1000) < wild { SPECIALS[r.below(SPECIALS.len())] } else { v });
            }
        }
    }
    out
}

/// The tables of `v` [n][h][d]: for each (head, channel) the base of the window of 8 values of exponent >> 1 that holds most of its finite values (0 to 120).
fn bases(v: &[u16], n: usize, h: usize, d: usize) -> Vec<u8> {
    let mut out = vec![0u8; h * d];
    for hh in 0..h {
        for c in 0..d {
            let mut hist = [0u32; 128];
            for t in 0..n {
                let b = v[(t * h + hh) * d + c];
                if (b >> 7) & 0xFF != 0xFF {
                    hist[((b >> 8) & 0x7F) as usize] += 1;
                }
            }
            let best = (0..=120).max_by_key(|&s| hist[s..s + 8].iter().sum::<u32>()).unwrap_or(0);
            out[hh * d + c] = best as u8;
        }
    }
    out
}

/// What `n` tokens of keys `k` and values `v` ([n][h][d]) give for queries `q` ([h g][d]): the attention of each query head over its kv head's tokens, in f64.
fn attention(q: &[u16], k: &[u16], v: &[u16], first: usize, n: usize, h: usize, g: usize, d: usize) -> Vec<f64> {
    let scale = 1.0 / (d as f64).sqrt();
    let mut out = vec![0.0; h * g * d];
    for hh in 0..h {
        for j in 0..g {
            let qh = &q[(hh * g + j) * d..(hh * g + j + 1) * d];
            if first >= n {
                continue; // (no token: a zero output)
            }
            let s: Vec<f64> = (first..n).map(|t| scale * (0..d).map(|c| f32_of(qh[c]) as f64 * f32_of(k[(t * h + hh) * d + c]) as f64).sum::<f64>()).collect();
            let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
            let z: f64 = w.iter().sum();
            for c in 0..d {
                out[(hh * g + j) * d + c] = (first..n).map(|t| w[t - first] * f32_of(v[(t * h + hh) * d + c]) as f64).sum::<f64>() / z;
            }
        }
    }
    out
}

/// One shape: `h` kv heads of `d` channels, `g` query heads to each. False where anything differs.
fn shape(lib: &Lib, dev: &Dev, h: usize, d: usize, g: usize) -> Result<bool> {
    const A: i64 = 15;
    const RING: i64 = 256;
    let hp = (h * d / 128) as i64; // (the page's heads of 128 channels)
    let n: usize = 16 * 5 + 7; // (five blocks and part of a sixth)
    let blocks = n.div_ceil(16);
    let page = lib.kv_page_bytes(h as i64, d as i64, A)?;
    let (nchunks, nslabs) = (4096i64, 64i64);
    let pool = Buf::zeroed(page as usize * blocks)?;
    let escape = Buf::zeroed(lib.kv_escape_bytes(nchunks)? as usize)?;
    let raw = Buf::zeroed(lib.kv_raw_bytes(nslabs, hp)? as usize)?;
    let status = Buf::zeroed(4 * (16 + 2 * RING) as usize)?;
    let geom = KvGeom { heads: h as i64, dim: d as i64, arena: A, page_stride: page, escape: escape.ptr, nchunks, raw: raw.ptr, nslabs };
    let mut r = Rng(0x9E37_79B9_7F4A_7C15 ^ (h * 1000 + d) as u64);
    let mut bad = 0;
    for (label, far, wild) in [("tame", 15, 0), ("wild", 15, 5)] {
        let (k, v) = (values(&mut r, n, h, d, far, wild), values(&mut r, n, h, d, far, wild));
        let (kb, vb) = (bases(&k, n, h, d), bases(&v, n, h, d));
        let (kbuf, vbuf) = (Buf::new(2 * k.len())?, Buf::new(2 * v.len())?);
        kbuf.write(0, &k)?;
        vbuf.write(0, &v)?;
        let (kbase, vbase) = (Buf::new(kb.len())?, Buf::new(vb.len())?);
        kbase.write(0, &kb)?;
        vbase.write(0, &vb)?;
        let slots: Vec<i64> = (0..n as i64).collect();
        let sl = Buf::new(8 * n)?;
        sl.write(0, &slots)?;
        let stride = (h * d) as i64;
        // SAFETY: live buffers of the sizes the kernels read and write; the stream is synchronized before the results are read.
        unsafe {
            lib.kv_write(&geom, pool.ptr, kbase.ptr, vbase.ptr, 0, kbuf.ptr, stride, vbuf.ptr, stride, sl.ptr, n as i64, status.ptr, 0, 0, RING, dev.stream)?
        };
        // read back: the pages into [blocks][16][h][2 d] (keys, then values), compared bit for bit with what was written
        let ids: Vec<i32> = (0..blocks as i32).collect();
        let (pages, dst) = (Buf::new(4 * blocks)?, Buf::new(4 * blocks)?);
        pages.write(0, &ids)?;
        dst.write(0, &ids)?;
        let out = Buf::zeroed(2 * blocks * 16 * h * 2 * d)?;
        unsafe { lib.kv_gather(&geom, pool.ptr, kbase.ptr, vbase.ptr, pages.ptr, dst.ptr, blocks as i64, out.ptr, dev.stream)? };
        dev.sync()?;
        let mut head = [0u32; 16];
        status.read(0, &mut head)?;
        let mut back = vec![0u16; blocks * 16 * h * 2 * d];
        out.read(0, &mut back)?;
        let (mut diff, mut first) = (0usize, None);
        for t in 0..n {
            for hh in 0..h {
                for c in 0..d {
                    let at = ((t * h + hh) * 2 * d) + c;
                    let want = (k[(t * h + hh) * d + c], v[(t * h + hh) * d + c]);
                    let got = (back[at], back[at + d]);
                    if got != want {
                        diff += 1;
                        first.get_or_insert((t, hh, c, want, got));
                    }
                }
            }
        }
        // and through unpack (what a paged attention that reads separate keys and values takes: [n][16][h][d] each, NHD)
        let (uk, uv) = (Buf::zeroed(2 * blocks * 16 * h * d)?, Buf::zeroed(2 * blocks * 16 * h * d)?);
        unsafe { lib.kv_unpack(&geom, pool.ptr, kbase.ptr, vbase.ptr, pages.ptr, blocks as i64, uk.ptr, uv.ptr, dev.stream)? };
        dev.sync()?;
        let (mut bk, mut bv) = (vec![0u16; blocks * 16 * h * d], vec![0u16; blocks * 16 * h * d]);
        uk.read(0, &mut bk)?;
        uv.read(0, &mut bv)?;
        let unpacked = bk[..n * h * d] == k[..] && bv[..n * h * d] == v[..];
        if !unpacked {
            bad += 1;
        }
        println!(
            "{} {h} heads x {d}, {label}: unpack into separate [pages][16][h][d] keys and values {}",
            if unpacked { "ok  " } else { "FAIL" },
            if unpacked { "equal bit for bit" } else { "differs" }
        );
        println!(
            "{} {h} heads x {d} ({hp} pages' heads), {label}: {} values read back, {diff} differ{}; extra chunks at most {}, raw pages {}, void pages {}",
            if diff == 0 && head[11] == 0 { "ok  " } else { "FAIL" },
            2 * n * h * d,
            first.map(|f| format!(" (first {f:?})")).unwrap_or_default(),
            head[5],
            head[10],
            head[11]
        );
        if diff != 0 || head[11] != 0 {
            bad += 1;
        }
        if wild == 0 {
            // the decode attention over the pages (queries small, so that the softmax is not one token's: a wrong score moves the weights; a sequence of all n tokens, and one of the first 2 blocks and 3), against the host's from the same values
            let q: Vec<u16> = (0..h * g * d).map(|_| bf16_of(((r.below(2000) as f32) - 1000.0) / 5000.0)).collect();
            let qbuf = Buf::new(2 * q.len())?;
            qbuf.write(0, &q)?;
            let width = (h * g * d) as i64;
            // (and windows, `kv_decode_from`: from a token inside a page, from a page's first token, the last token alone, and none: a zero output)
            for (len, first) in [(n, 0), (16 * 2 + 3, 0), (n, 21), (n, 48), (n, n - 1), (n, n)] {
                let table: Vec<i32> = (0..blocks as i32).collect();
                let (bt, sq) = (Buf::new(4 * blocks)?, Buf::new(4)?);
                bt.write(0, &table)?;
                sq.write(0, &[len as i32])?;
                let splits = lib.kv_decode_splits(1, h as i64, blocks as i64)?;
                let ws_bytes = lib.kv_decode_workspace(1, h as i64, g as i64, d as i64, splits)?;
                let ws = Buf::zeroed(ws_bytes)?;
                let o = Buf::zeroed(2 * h * g * d)?;
                let fb = Buf::new(4)?;
                fb.write(0, &[first as i32])?;
                unsafe {
                    if first == 0 {
                        lib.kv_decode(
                            &geom,
                            g as i64,
                            pool.ptr,
                            kbase.ptr,
                            vbase.ptr,
                            qbuf.ptr,
                            width,
                            bt.ptr,
                            blocks as i64,
                            sq.ptr,
                            1,
                            splits,
                            1.0 / (d as f64).sqrt(),
                            o.ptr,
                            width,
                            ws.ptr,
                            ws_bytes,
                            dev.stream,
                        )?
                    } else {
                        lib.kv_decode_from(
                            &geom,
                            g as i64,
                            pool.ptr,
                            kbase.ptr,
                            vbase.ptr,
                            qbuf.ptr,
                            width,
                            bt.ptr,
                            blocks as i64,
                            sq.ptr,
                            fb.ptr,
                            false,
                            1,
                            splits,
                            1.0 / (d as f64).sqrt(),
                            o.ptr,
                            width,
                            ws.ptr,
                            ws_bytes,
                            dev.stream,
                        )?
                    }
                };
                dev.sync()?;
                let mut got = vec![0u16; h * g * d];
                o.read(0, &mut got)?;
                let want = attention(&q, &k, &v, first, len, h, g, d);
                let (mut worst, mut off) = (0f64, 0usize);
                for (x, w) in got.iter().zip(&want) {
                    let e = (f32_of(*x) as f64 - w).abs() / (w.abs() * 0.015 + 2e-3);
                    worst = worst.max(e);
                    off += (e > 1.0 || !e.is_finite()) as usize;
                }
                println!("{} {h} heads x {d}, {g} queries to a head, {len} tokens from {first}: decode attention, {off} of {} values past 1.5% + 0.002 (worst {worst:.2} of the bound)", if off == 0 { "ok  " } else { "FAIL" }, got.len());
                if off != 0 {
                    bad += 1;
                }
            }
        }
    }
    Ok(bad == 0)
}

/// One stream: a tensor [n][h][d] held in the keys' layout alone (a latent padded to 5 heads of 128: Kimi-K3's 576 values; keys that are the values: Gemma's global layers), written with no value, read back
/// into one tensor and compared bit for bit; its page is smaller by the V tiles; and the calls that cannot take it refuse it.
fn one_stream(lib: &Lib, dev: &Dev, h: usize, d: usize, pad: usize) -> Result<bool> {
    const A: i64 = 15;
    const RING: i64 = 256;
    let hp = (h * d / 128) as i64;
    let n: usize = 16 * 5 + 7;
    let blocks = n.div_ceil(16);
    let two = lib.kv_page_bytes(h as i64, d as i64, A)?;
    let one = lib.kv_page_bytes(h as i64, d as i64 | KV_ONE_STREAM, A)?;
    let (nchunks, nslabs) = (4096i64, 64i64);
    let pool = Buf::zeroed(one as usize * blocks)?;
    let escape = Buf::zeroed(lib.kv_escape_bytes(nchunks)? as usize)?;
    let raw = Buf::zeroed(lib.kv_raw_bytes(nslabs, hp)? as usize)?;
    let status = Buf::zeroed(4 * (16 + 2 * RING) as usize)?;
    let geom = KvGeom { heads: h as i64, dim: d as i64 | KV_ONE_STREAM, arena: A, page_stride: one, escape: escape.ptr, nchunks, raw: raw.ptr, nslabs };
    let mut r = Rng(0xA5A5_5A5A_1234_4321 ^ (h * 1000 + d) as u64);
    let mut bad = 0;
    for (label, far, wild) in [("tame", 15, 0), ("wild", 15, 5)] {
        let mut k = values(&mut r, n, h, d, far, wild);
        // (the padding of a latent: the last `pad` channels of every token are zero)
        for t in 0..n {
            for c in d - pad..d {
                k[(t * h + h - 1) * d + c] = 0;
            }
        }
        let kb = bases(&k, n, h, d);
        let (kbuf, kbase) = (Buf::new(2 * k.len())?, Buf::new(kb.len())?);
        kbuf.write(0, &k)?;
        kbase.write(0, &kb)?;
        let slots: Vec<i64> = (0..n as i64).collect();
        let sl = Buf::new(8 * n)?;
        sl.write(0, &slots)?;
        let stride = (h * d) as i64;
        // SAFETY: live buffers of the sizes the kernels read and write; the stream is synchronized before the results are read.
        unsafe { lib.kv_write(&geom, pool.ptr, kbase.ptr, 0, 0, kbuf.ptr, stride, 0, 0, sl.ptr, n as i64, status.ptr, 0, 0, RING, dev.stream)? };
        let ids: Vec<i32> = (0..blocks as i32).collect();
        let pages = Buf::new(4 * blocks)?;
        pages.write(0, &ids)?;
        let out = Buf::zeroed(2 * blocks * 16 * h * d)?;
        unsafe { lib.kv_unpack(&geom, pool.ptr, kbase.ptr, 0, pages.ptr, blocks as i64, out.ptr, 0, dev.stream)? };
        dev.sync()?;
        let mut head = [0u32; 16];
        status.read(0, &mut head)?;
        let mut back = vec![0u16; blocks * 16 * h * d];
        out.read(0, &mut back)?;
        let diff = back[..n * h * d].iter().zip(&k).filter(|(a, b)| a != b).count();
        let ok = diff == 0 && head[11] == 0;
        println!(
            "{} one stream {h} heads x {d} ({hp} pages' heads, {pad} padding channels), {label}: {} values read back, {diff} differ; page {one} bytes against {two} for two streams; raw pages {}, void pages {}",
            if ok { "ok  " } else { "FAIL" },
            n * h * d,
            head[10],
            head[11]
        );
        if !ok || one >= two {
            bad += 1;
        }
    }
    // the calls that read or write two streams refuse one
    let refused = unsafe {
        let ids = Buf::new(8)?;
        lib.kv_gather(&geom, pool.ptr, 0, 0, ids.ptr, ids.ptr, 1, ids.ptr, dev.stream).is_err()
            && lib.kv_decode_workspace(1, h as i64, 4, d as i64 | KV_ONE_STREAM, 1).is_err()
    };
    println!("{} one stream: gather and decode refuse it", if refused { "ok  " } else { "FAIL" });
    if !refused {
        bad += 1;
    }
    Ok(bad == 0)
}

/// Every shape; false where any differs.
pub fn run() -> Result<bool> {
    let dev = Dev::new(0)?;
    let lib = Lib::find()?;
    let mut ok = true;
    for (h, d, g) in [(8, 128, 4), (4, 256, 4), (8, 256, 8), (2, 512, 4), (4, 512, 8), (8, 512, 4)] {
        ok &= shape(&lib, &dev, h, d, g)?;
    }
    // one stream: a 576-value latent in 5 heads of 128 (64 channels of padding), keys that are the values in 2 heads of 512 and one of 256
    for (h, d, pad) in [(5, 128, 64), (2, 512, 0), (1, 256, 0)] {
        ok &= one_stream(&lib, &dev, h, d, pad)?;
    }
    println!("{}", if ok { "the lossless KV cache of every head dimension reads back bit for bit and decodes" } else { "FAILED" });
    Ok(ok)
}
