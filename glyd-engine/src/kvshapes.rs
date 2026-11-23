//! A known answer for the lossless KV cache of every head dimension the library holds (128, 256 and 512 channels; 256 and 512 are held as two and four of the page's heads of 128): bf16 keys and values made on the host
//! (in the windows of their channels, and out of them, and with every kind of special value), written through `glyd_gpu_kv_write`, read back through `glyd_gpu_kv_gather` and compared bit for bit with what was written, and
//! the decode attention (`glyd_gpu_kv_decode`) and the prompt attention (`glyd_gpu_kv_prefill`) against attention computed on the host from the same values. No model: `glyd-engine kvshapes`, in the GPU job.
//! `glyd-engine kvprefill` times the prompt attention against the unpack of the pages to a bf16 ring and the bf16 kernel on it.

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
                // (the same values as a bf16 cache: the attention kind's decode over bf16 pages, glyd_gpu_kv_decode_bf16, held to the pages' bit for bit; the pages are lossless)
                let o2 = Buf::zeroed(2 * h * g * d)?;
                // SAFETY: as above; the workspace's counters are zero again after a call.
                unsafe {
                    lib.kv_decode_bf16(
                        qbuf.ptr,
                        width,
                        (kbuf.ptr, vbuf.ptr, bt.ptr, blocks as i64, 0),
                        (sq.ptr, if first == 0 { 0 } else { fb.ptr }, 1, splits),
                        (h as i64, g as i64, d as i64),
                        (1.0 / (d as f64).sqrt(), false),
                        o2.ptr,
                        width,
                        (ws.ptr, ws_bytes),
                        dev.stream,
                    )?
                };
                dev.sync()?;
                let mut bf = vec![0u16; h * g * d];
                o2.read(0, &mut bf)?;
                let differ = bf.iter().zip(&got).filter(|(x, y)| x != y).count();
                println!(
                    "{} {h} heads x {d}, {g} queries to a head, {len} tokens from {first}: the same over a bf16 cache (glyd_gpu_kv_decode_bf16), {differ} of {} values not the pages' bits",
                    if differ == 0 { "ok  " } else { "FAIL" },
                    bf.len()
                );
                bad += (differ != 0) as usize;
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

/// `n` tokens of [h][d] values as `values` makes them, but those out of their channel's window (`far` in a thousand, and 2 in a thousand of the finite specials: zeros and
/// subnormals) below it only: keys whose scores stay a few units (a softmax over many of them), and values with no term of an output far larger than the output. (Values of
/// 2^11 to 2^13 among them make terms of hundreds that cancel.)
fn near(r: &mut Rng, n: usize, h: usize, d: usize, far: usize) -> Vec<u16> {
    let mut out = Vec::with_capacity(n * h * d);
    for _ in 0..n {
        for hh in 0..h {
            for c in 0..d {
                let mu = 124 + (hh * 5 + c * 3) % 7;
                let e = if r.below(1000) < far { mu - 12 - r.below(4) } else { mu + r.below(3) - 1 };
                let v = ((r.below(2) as u16) << 15) | ((e as u16) << 7) | r.below(128) as u16;
                out.push(if r.below(1000) < 2 { [0x0000, 0x8000, 0x0001, 0x8001][r.below(4)] } else { v });
            }
        }
    }
    out
}

/// One prompt chunk's attention.
#[derive(Clone, Copy)]
struct Chunk {
    start: usize,
    n: usize,
    window: usize,
    rows: bool,
    round: bool,
    cap: f64,
    sinks: bool,
}

/// What a chunk's rows `q` ([n][h g][d]) give over the sequence's keys `k` and values `v` ([tokens][h][d]), in f64: each row over the keys from its window's first to itself; a
/// score the dot product (its bf16 where `round`) times `scale`, capped, a sink's logit joining the sum. With each output, the sum of its terms' magnitudes (each probability
/// times its value's magnitude): the kernel rounds each probability to bf16 for the values, which moves an output by up to 2^-8 of that.
fn prompt_attention(q: &[u16], k: &[u16], v: &[u16], c: Chunk, h: usize, g: usize, d: usize, scale: f64, sinks: &[f32]) -> (Vec<f64>, Vec<f64>) {
    let (mut out, mut mag) = (vec![0.0; c.n * h * g * d], vec![0.0; c.n * h * g * d]);
    for t in 0..c.n {
        let p = c.start + t;
        let lo = if c.window > 0 { (p + 1).saturating_sub(c.window) } else { 0 };
        for hh in 0..h {
            for j in 0..g {
                let qh = &q[((t * h + hh) * g + j) * d..][..d];
                let s: Vec<f64> = (lo..=p)
                    .map(|u| {
                        let dot = (0..d).map(|x| f32_of(qh[x]) as f64 * f32_of(k[(u * h + hh) * d + x]) as f64).sum::<f64>();
                        let dot = if c.round { f32_of(bf16_of(dot as f32)) as f64 } else { dot };
                        let x = dot * scale;
                        if c.cap > 0.0 {
                            c.cap * (x / c.cap).tanh()
                        } else {
                            x
                        }
                    })
                    .collect();
                let sink = if c.sinks { sinks[hh * g + j] as f64 } else { f64::NEG_INFINITY };
                let m = s.iter().cloned().fold(sink, f64::max);
                let w: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
                let z: f64 = w.iter().sum::<f64>() + (sink - m).exp();
                for x in 0..d {
                    let at = ((t * h + hh) * g + j) * d + x;
                    out[at] = (lo..=p).map(|u| w[u - lo] * f32_of(v[(u * h + hh) * d + x]) as f64).sum::<f64>() / z;
                    mag[at] = (lo..=p).map(|u| w[u - lo] * (f32_of(v[(u * h + hh) * d + x]) as f64).abs()).sum::<f64>() / z;
                }
            }
        }
    }
    (out, mag)
}

/// The prompt attention of one shape (`h` kv heads of `d` channels, `g` query heads to each) over a sequence of 230 tokens in pages taken out of order: a chunk at its start,
/// in its middle (its prefix in pages from a token inside a page), past a sliding window, the chunk's keys from the pages, a window with soft-capping and sinks, a single row;
/// against the host's attention from the same values. False where a value is past the decode check's bound and the probabilities' rounding to bf16 (2^-8 of the sum of its
/// terms' magnitudes: over a softmax of tens to hundreds of keys, that rounding alone took 0.3 to 0.7% of the outputs past the decode check's bound, at head 512 too, where the kernel's
/// outputs are Gemma 4's g4_fa512b's bit for bit).
fn prefill(lib: &Lib, dev: &Dev, h: usize, d: usize, g: usize) -> Result<bool> {
    const A: i64 = 15;
    const RING: i64 = 256;
    let hp = (h * d / 128) as i64;
    let total = 16 * 14 + 6;
    let blocks = 20usize;
    let table: Vec<i32> = (0..blocks as i32).map(|j| (7 * j + 3) % blocks as i32).collect(); // (the sequence's page j in block 7 j + 3 modulo 20)
    let page = lib.kv_page_bytes(h as i64, d as i64, A)?;
    let (nchunks, nslabs) = (4096i64, 64i64);
    let pool = Buf::zeroed(page as usize * blocks)?;
    let escape = Buf::zeroed(lib.kv_escape_bytes(nchunks)? as usize)?;
    let raw = Buf::zeroed(lib.kv_raw_bytes(nslabs, hp)? as usize)?;
    let status = Buf::zeroed(4 * (16 + 2 * RING) as usize)?;
    let geom = KvGeom { heads: h as i64, dim: d as i64, arena: A, page_stride: page, escape: escape.ptr, nchunks, raw: raw.ptr, nslabs };
    let mut r = Rng(0x5EED_0F_C4A7 ^ (h * 1000 + d * 10 + g) as u64);
    let (k, v) = (near(&mut r, total, h, d, 15), near(&mut r, total, h, d, 15));
    let (kb, vb) = (bases(&k, total, h, d), bases(&v, total, h, d));
    let (kbuf, vbuf, kbase, vbase) = (Buf::new(2 * k.len())?, Buf::new(2 * v.len())?, Buf::new(kb.len())?, Buf::new(vb.len())?);
    kbuf.write(0, &k)?;
    vbuf.write(0, &v)?;
    kbase.write(0, &kb)?;
    vbase.write(0, &vb)?;
    let slots: Vec<i64> = (0..total).map(|t| table[t / 16] as i64 * 16 + (t % 16) as i64).collect();
    let (sl, bt) = (Buf::new(8 * total)?, Buf::new(4 * blocks)?);
    sl.write(0, &slots)?;
    bt.write(0, &table)?;
    let w = (h * d) as i64;
    // SAFETY: live buffers of the sizes the kernels read and write; the stream is synchronized before the results are read.
    unsafe { lib.kv_write(&geom, pool.ptr, kbase.ptr, vbase.ptr, 0, kbuf.ptr, w, vbuf.ptr, w, sl.ptr, total as i64, status.ptr, 0, 0, RING, dev.stream)? };
    let qw = h * g * d;
    let q: Vec<u16> = (0..total * qw).map(|_| bf16_of(((r.below(2000) as f32) - 1000.0) / 5000.0)).collect();
    let sinks: Vec<f32> = (0..h * g).map(|i| (i % 5) as f32 * 0.5 - 1.0).collect();
    let (qbuf, sbuf) = (Buf::new(2 * q.len())?, Buf::new(4 * sinks.len())?);
    qbuf.write(0, &q)?;
    sbuf.write(0, &sinks)?;
    let scale = 1.0 / (d as f64).sqrt();
    let c = |start, n, window, rows, round, cap, sinks| Chunk { start, n, window, rows, round, cap, sinks };
    let cases = [
        c(0, 70, 0, true, true, 0.0, false),
        c(100, 70, 0, true, false, 0.0, false),
        c(150, 80, 40, true, true, 0.0, false),
        c(96, 37, 0, false, true, 0.0, false),
        c(130, 50, 64, false, false, 0.5, true),
        c(37, 1, 0, true, true, 0.0, false),
        c(200, 30, 0, true, true, 0.5, true),
    ];
    // (the same values as a bf16 cache in the pages' places: the attention kind's prompt attention over bf16 pages, glyd_gpu_kv_prefill_bf16)
    let (mut kp, mut vp) = (vec![0u16; blocks * 16 * w as usize], vec![0u16; blocks * 16 * w as usize]);
    for (t, &sl) in slots.iter().enumerate() {
        let (a, b, wu) = (sl as usize * w as usize, t * w as usize, w as usize);
        kp[a..a + wu].copy_from_slice(&k[b..b + wu]);
        vp[a..a + wu].copy_from_slice(&v[b..b + wu]);
    }
    let (kcache, vcache) = (Buf::new(2 * kp.len())?, Buf::new(2 * vp.len())?);
    kcache.write(0, &kp)?;
    vcache.write(0, &vp)?;
    let mut bad = 0;
    for ch in cases {
        let o = Buf::zeroed(2 * ch.n * qw)?;
        let rows = ch.rows.then_some((kbuf.ptr + (2 * ch.start) as u64 * w as u64, w, vbuf.ptr + (2 * ch.start) as u64 * w as u64, w));
        unsafe {
            lib.kv_prefill(
                &geom,
                g as i64,
                pool.ptr,
                kbase.ptr,
                vbase.ptr,
                bt.ptr,
                qbuf.ptr + (2 * ch.start * qw) as u64,
                qw as i64,
                rows,
                ch.n as i64,
                ch.start as i64,
                ch.window as i64,
                scale,
                ch.round,
                ch.cap,
                if ch.sinks { sbuf.ptr } else { 0 },
                o.ptr,
                qw as i64,
                dev.stream,
            )?
        };
        dev.sync()?;
        let mut got = vec![0u16; ch.n * qw];
        o.read(0, &mut got)?;
        let (want, mag) = prompt_attention(&q[ch.start * qw..(ch.start + ch.n) * qw], &k, &v, ch, h, g, d, scale, &sinks);
        let (mut worst, mut off, mut at) = (0f64, 0usize, None);
        for (i, ((x, w), m)) in got.iter().zip(&want).zip(&mag).enumerate() {
            let e = (f32_of(*x) as f64 - w).abs() / (w.abs() * 0.015 + 2e-3 + m / 256.0);
            worst = worst.max(e);
            if e > 1.0 || !e.is_finite() {
                off += 1;
                at.get_or_insert((i / qw, i % qw / d, i % d, *w, f32_of(*x)));
            }
        }
        if let Some((t, hq, c, w, x)) = at {
            println!("     the first: row {}, query head {hq}, channel {c}: {x} where the host's is {w:.5}", ch.start + t);
        }
        // (the chunk's rows given, as the executor gives them, and no sinks: the bf16 cache's prompt attention, held to the pages' bit for bit; the pages are lossless)
        if ch.rows && !ch.sinks {
            let o2 = Buf::zeroed(2 * ch.n * qw)?;
            // SAFETY: as above.
            unsafe {
                lib.kv_prefill_bf16(
                    qbuf.ptr + (2 * ch.start * qw) as u64,
                    qw as i64,
                    (kcache.ptr, vcache.ptr, bt.ptr, 0),
                    (ch.n as i64, ch.start as i64, ch.window as i64),
                    (h as i64, g as i64, d as i64),
                    (scale, ch.round, ch.cap),
                    o2.ptr,
                    qw as i64,
                    dev.stream,
                )?
            };
            dev.sync()?;
            let mut bf = vec![0u16; ch.n * qw];
            o2.read(0, &mut bf)?;
            let differ = bf.iter().zip(&got).filter(|(x, y)| x != y).count();
            println!(
                "{} {h} heads x {d}, {g} queries to a head, rows {}..{} (window {}): the same over a bf16 cache (glyd_gpu_kv_prefill_bf16), {differ} of {} values not the pages' bits",
                if differ == 0 { "ok  " } else { "FAIL" },
                ch.start,
                ch.start + ch.n,
                ch.window,
                bf.len()
            );
            bad += (differ != 0) as usize;
        }
        println!(
            "{} {h} heads x {d}, {g} queries to a head, rows {}..{} (window {}, keys {}{}{}{}): prompt attention, {off} of {} values past 1.5% + 0.002 + 2^-8 of the terms (worst {worst:.2} of the bound)",
            if off == 0 { "ok  " } else { "FAIL" },
            ch.start,
            ch.start + ch.n,
            ch.window,
            if ch.rows { "of the chunk from its rows" } else { "all from the pages" },
            if ch.round { ", scores bf16" } else { ", scores fp32" },
            if ch.cap > 0.0 { ", soft-capped" } else { "" },
            if ch.sinks { ", sinks" } else { "" },
            got.len()
        );
        bad += (off != 0) as usize;
    }
    let mut head = [0u32; 16];
    status.read(0, &mut head)?;
    if head[11] != 0 {
        println!("FAIL {h} heads x {d}: {} void pages", head[11]);
        bad += 1;
    }
    Ok(bad == 0)
}

/// `kvprefill [--chunk 256] [--prefixes 0,1024,4096,8192] [--reps 20]`: a chunk's prompt attention at gemma-4-12B's shapes (its sliding layers: 8 kv heads of 256, 16 query
/// heads, a window of 1,024; its global ones: 1 of 512, 16 query heads), three ways: the pages unpacked to a bf16 ring (`kv12_stage`) and the bf16 kernel
/// (glyd_gpu_kv_prefill_bf16) on it, `glyd_gpu_kv_prefill` on the pages and the chunk's rows (the executor's path), and the bf16 kernel alone on a ring of every token (what bf16
/// pages cost); the milliseconds of each, and how many outputs of the other two are not the pages' bits (the kernel is one, the values the same).
pub fn bench(a: &crate::args::Args) -> Result<()> {
    const A: i64 = 15;
    let dev = Dev::new(0)?;
    let lib = Lib::find()?;
    let s = dev.stream;
    let (n, reps) = (a.num("chunk", 256usize)?, a.num("reps", 20usize)?);
    let prefixes: Vec<usize> = a.flag("prefixes").unwrap_or("0,1024,4096,8192").split(',').filter_map(|x| x.trim().parse().ok()).collect();
    println!("{}: a chunk of {n} rows, {reps} runs each (ms a call)", dev.name);
    for (h, d, nq, window) in [(8usize, 256usize, 16usize, 1024usize), (1, 512, 16, 0)] {
        for &prefix in &prefixes {
            let total = prefix + n;
            let blocks = total.div_ceil(16);
            let mut r = Rng(0xB0A7 ^ (prefix * 7 + d) as u64);
            let (k, v) = (near(&mut r, total, h, d, 5), near(&mut r, total, h, d, 5));
            let (kb, vb) = (bases(&k, total, h, d), bases(&v, total, h, d));
            let page = lib.kv_page_bytes(h as i64, d as i64, A)?;
            let hp = (h * d / 128) as i64;
            let (nchunks, nslabs) = ((blocks * 8) as i64, 64i64);
            let pool = Buf::zeroed(page as usize * blocks)?;
            let escape = Buf::zeroed(lib.kv_escape_bytes(nchunks)? as usize)?;
            let raw = Buf::zeroed(lib.kv_raw_bytes(nslabs, hp)? as usize)?;
            let status = Buf::zeroed(4 * (16 + 2 * 256))?;
            let geom = KvGeom { heads: h as i64, dim: d as i64, arena: A, page_stride: page, escape: escape.ptr, nchunks, raw: raw.ptr, nslabs };
            let (kbuf, vbuf, kbase, vbase) = (Buf::new(2 * k.len())?, Buf::new(2 * v.len())?, Buf::new(kb.len())?, Buf::new(vb.len())?);
            kbuf.write(0, &k)?;
            vbuf.write(0, &v)?;
            kbase.write(0, &kb)?;
            vbase.write(0, &vb)?;
            let slots: Vec<i64> = (0..total as i64).collect();
            let table: Vec<i32> = (0..blocks as i32).collect();
            let (sl, bt) = (Buf::new(8 * total)?, Buf::new(4 * blocks)?);
            sl.write(0, &slots)?;
            bt.write(0, &table)?;
            let w = (h * d) as i64;
            unsafe { lib.kv_write(&geom, pool.ptr, kbase.ptr, vbase.ptr, 0, kbuf.ptr, w, vbuf.ptr, w, sl.ptr, total as i64, status.ptr, 0, 0, 256, s)? };
            let qw = nq * d;
            let q: Vec<u16> = (0..n * qw).map(|_| bf16_of(((r.below(2000) as f32) - 1000.0) / 5000.0)).collect();
            let qbuf = Buf::new(2 * q.len())?;
            qbuf.write(0, &q)?;
            // the executor's ring a prompt chunk's pages are unpacked into (kv12_stage), and a bf16 ring of every token (bf16 pages)
            let sb = if window > 0 { (window + n).div_ceil(16) + 2 } else { total.div_ceil(16) + 1 };
            let row = 16 * h * d * 2;
            let (stk, stv) = (Buf::zeroed(sb * row)?, Buf::zeroed(sb * row)?);
            let (o_stage, o_pages, o_bf) = (Buf::zeroed(2 * n * qw)?, Buf::zeroed(2 * n * qw)?, Buf::zeroed(2 * n * qw)?);
            let lo = if window > 0 { (prefix + 1).saturating_sub(window) } else { 0 };
            let (b0, b1) = (lo / 16, total.div_ceil(16));
            let unpack = || -> Result<()> {
                let mut b = b0;
                while b < b1 {
                    let at = b % sb;
                    let m = (b1 - b).min(sb - at);
                    unsafe {
                        lib.kv_unpack(
                            &geom,
                            pool.ptr,
                            kbase.ptr,
                            vbase.ptr,
                            bt.ptr + 4 * b as u64,
                            m as i64,
                            stk.ptr + (at * row) as u64,
                            stv.ptr + (at * row) as u64,
                            s,
                        )?
                    };
                    b += m;
                }
                Ok(())
            };
            // (the attention kind's prompt attention over a bf16 cache, glyd_gpu_kv_prefill_bf16: Gemma 4's arithmetic, scale 1, the scores bf16; a ring of `cap` places)
            let family = |kc: u64, vc: u64, cap: usize, out: u64| -> Result<()> {
                unsafe {
                    lib.kv_prefill_bf16(
                        qbuf.ptr,
                        qw as i64,
                        (kc, vc, 0, cap as i64),
                        (n as i64, prefix as i64, window as i64),
                        (h as i64, (nq / h) as i64, d as i64),
                        (1.0, true, 0.0),
                        out,
                        qw as i64,
                        s,
                    )
                }
            };
            let pages = || -> Result<()> {
                let rows = Some((kbuf.ptr + (2 * prefix) as u64 * w as u64, w, vbuf.ptr + (2 * prefix) as u64 * w as u64, w));
                // (Gemma 4's attention: scale 1, the scores bf16)
                unsafe {
                    lib.kv_prefill(
                        &geom,
                        (nq / h) as i64,
                        pool.ptr,
                        kbase.ptr,
                        vbase.ptr,
                        bt.ptr,
                        qbuf.ptr,
                        qw as i64,
                        rows,
                        n as i64,
                        prefix as i64,
                        window as i64,
                        1.0,
                        true,
                        0.0,
                        0,
                        o_pages.ptr,
                        qw as i64,
                        s,
                    )
                }
            };
            let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
                f()?;
                dev.sync()?;
                let t = crate::dev::Timer::new()?;
                t.start(s)?;
                for _ in 0..reps {
                    f()?;
                }
                Ok(t.stop(s)? as f64 / reps as f64)
            };
            let t_unpack = time(&unpack)?;
            let t_fa = time(&|| family(stk.ptr, stv.ptr, sb * 16, o_stage.ptr))?;
            let t_pages = time(&pages)?;
            // (the keys of every token in a bf16 ring of the sequence's length: the family's kernel as bf16 pages run it, less their table's reads)
            let t_bf = time(&|| family(kbuf.ptr, vbuf.ptr, total, o_bf.ptr))?;
            let (mut x, mut y, mut z) = (vec![0u16; n * qw], vec![0u16; n * qw], vec![0u16; n * qw]);
            o_stage.read(0, &mut x)?;
            o_pages.read(0, &mut y)?;
            o_bf.read(0, &mut z)?;
            let (stage_differ, bf_differ) = (x.iter().zip(&y).filter(|(a, b)| a != b).count(), z.iter().zip(&y).filter(|(a, b)| a != b).count());
            let what = format!("{h} x {d} ({nq} query heads{}), prefix {prefix:>5}", if window > 0 { format!(", window {window}") } else { String::new() });
            println!(
                "{what}: unpack {t_unpack:.3} + the bf16 kernel {t_fa:.3} = {:.3}; kv_prefill {t_pages:.3} ({:+.1}%); bf16 ring {t_bf:.3}; outputs not the pages' bits: {stage_differ} unpacked, {bf_differ} bf16 ring, of {}",
                t_unpack + t_fa,
                100.0 * (t_pages / (t_unpack + t_fa) - 1.0),
                x.len()
            );
        }
    }
    Ok(())
}

/// Every shape; false where any differs.
pub fn run() -> Result<bool> {
    let dev = Dev::new(0)?;
    let lib = Lib::find()?;
    let mut ok = true;
    for (h, d, g) in [(8, 128, 4), (4, 256, 4), (8, 256, 8), (2, 512, 4), (4, 512, 8), (8, 512, 4)] {
        ok &= shape(&lib, &dev, h, d, g)?;
    }
    // the prompt attention: gemma-4-12B's sliding (8 x 256, 2 queries a head) and global (1 x 512, 16) layers among them
    for (h, d, g) in [(8, 128, 4), (4, 256, 4), (8, 256, 2), (1, 512, 16), (2, 512, 4)] {
        ok &= prefill(&lib, &dev, h, d, g)?;
    }
    // one stream: a 576-value latent in 5 heads of 128 (64 channels of padding), keys that are the values in 2 heads of 512 and one of 256
    for (h, d, pad) in [(5, 128, 64), (2, 512, 0), (1, 256, 0)] {
        ok &= one_stream(&lib, &dev, h, d, pad)?;
    }
    println!("{}", if ok { "the lossless KV cache of every head dimension reads back bit for bit and decodes" } else { "FAILED" });
    Ok(ok)
}
