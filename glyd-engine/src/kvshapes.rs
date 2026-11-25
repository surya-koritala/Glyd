//! A known answer for the lossless KV cache of every head dimension the library holds (128, 256 and 512 channels; 256 and 512 are held as two and four of the page's heads of 128): bf16 keys and values made on the host
//! (in the windows of their channels, and out of them, and with every kind of special value), written through `glyd_gpu_kv_write`, read back through `glyd_gpu_kv_gather` and compared bit for bit with what was written, and
//! the decode attention (`glyd_gpu_kv_decode`) and the prompt attention (`glyd_gpu_kv_prefill`) against attention computed on the host from the same values; a batch of sequences in one decode
//! call against each alone and against a bf16 cache (`decode_batch`). No model: `glyd-engine kvshapes`, in the GPU job.
//! `glyd-engine kvprefill` times the prompt attention against the unpack of the pages to a bf16 ring and the bf16 kernel on it.

use crate::dev::{Buf, Dev};
use crate::ffi::{KvGeom, Lib, Result, KV_ONE_STREAM};
use crate::gpu_ops::GpuOps;

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

/// What `n` tokens of keys `k` and values `v` ([n][h][d]) give for queries `q` ([h g][d]): the attention of each query head over its kv head's tokens, in f64. With each
/// output, the sum of its terms' magnitudes (each probability times its value's magnitude): the kernel rounds each probability to bf16 for the values, which moves an
/// output by up to 2^-8 of that (as `prompt_attention` holds the prompt's kernel to).
fn attention(q: &[u16], k: &[u16], v: &[u16], first: usize, n: usize, h: usize, g: usize, d: usize) -> (Vec<f64>, Vec<f64>) {
    let scale = 1.0 / (d as f64).sqrt();
    let (mut out, mut mag) = (vec![0.0; h * g * d], vec![0.0; h * g * d]);
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
                mag[(hh * g + j) * d + c] = (first..n).map(|t| w[t - first] * (f32_of(v[(t * h + hh) * d + c]) as f64).abs()).sum::<f64>() / z;
            }
        }
    }
    (out, mag)
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
                let (want, mag) = attention(&q, &k, &v, first, len, h, g, d);
                let (mut worst, mut off, mut needed) = (0f64, 0usize, 0usize);
                for ((x, w), m) in got.iter().zip(&want).zip(&mag) {
                    let (e, tight) = over_bound(*x, *w, *m);
                    worst = worst.max(e);
                    off += (e > 1.0 || !e.is_finite()) as usize;
                    needed += (tight > 1.0) as usize;
                }
                let fine = off == 0 && needed <= rounding_allowed(got.len());
                println!("{} {h} heads x {d}, {g} queries to a head, {len} tokens from {first}: decode attention, {off} of {} values past 1.5% + 0.002 + 2^-8 of their terms (worst {worst:.2} of the bound), {needed} needed the 2^-8 term (at most {})", if fine { "ok  " } else { "FAIL" }, got.len(), rounding_allowed(got.len()));
                if !fine {
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

/// The sequences (length, window's first token) of `decode_batch`'s call: one token, a page, a page and one, a short reply's, a context over several CTAs, and windows that
/// begin inside a page.
const BATCH: [(usize, usize); 7] = [(1, 0), (16, 0), (17, 0), (170, 0), (645, 0), (645, 600), (300, 289)];

/// Several sequences in one decode call, as a batch of the engine's has them (`BATCH`), each in its own blocks of one pool (a table of rows, the blocks in a shuffled order), with
/// `h` kv heads of `d` channels and `g` query heads to each. Bit for bit: each sequence's outputs in the call are its outputs in a call of its own (its bits do not depend on the
/// batch); and over a bf16 cache (`glyd_gpu_kv_decode_bf16`) they are the pages', through the same table (the places no token was written to hold infinities and not-a-numbers:
/// the bf16 decode reads a page's every token and masks them after) and through a ring of `CAP` places, a token at its position modulo the ring, where the window fits in it.
/// The plan is the engine's (dynamic::attn::LibDecode): its splits for a 32,768-token context, which leave most of a short sequence's CTAs without a page, and its floor of 32.
fn decode_batch(lib: &Lib, dev: &Dev, h: usize, d: usize, g: usize, round: bool) -> Result<bool> {
    const A: i64 = 15;
    const CAP: usize = 200; // (not a multiple of 16: a page's tokens wrap inside it)
    let nb = BATCH.len();
    let pages_of: Vec<usize> = BATCH.iter().map(|s| s.0.div_ceil(16)).collect();
    let (stride, blocks) = (*pages_of.iter().max().unwrap_or(&1), pages_of.iter().sum::<usize>());
    let mut r = Rng(0xD3C0_DE00 ^ (h * 1000 + d * 10 + g) as u64);
    let mut order: Vec<usize> = (0..blocks).collect();
    for i in (1..blocks).rev() {
        order.swap(i, r.below(i + 1));
    }
    let mut table = vec![0i32; nb * stride];
    let mut next = order.into_iter();
    for (b, &np) in pages_of.iter().enumerate() {
        for p in 0..np {
            table[b * stride + p] = next.next().expect("a block a page") as i32;
        }
    }
    // (each sequence's keys and values; the pages written through the table; a bf16 cache of the pool's places, the places past a sequence's end infinities and not-a-numbers)
    let w = h * d;
    let kv: Vec<(Vec<u16>, Vec<u16>)> = BATCH.iter().map(|s| (values(&mut r, s.0, h, d, 15, 0), values(&mut r, s.0, h, d, 15, 0))).collect();
    let (mut kc, mut vc) = (vec![0u16; blocks * 16 * w], vec![0u16; blocks * 16 * w]);
    for (i, x) in kc.iter_mut().chain(vc.iter_mut()).enumerate() {
        *x = [0x7FC0, 0x7F80, 0xFF80, 0xFFC1][i % 4];
    }
    let (mut ks, mut vs, mut slots) = (Vec::new(), Vec::new(), Vec::new());
    for (b, (k, v)) in kv.iter().enumerate() {
        for t in 0..BATCH[b].0 {
            let place = table[b * stride + t / 16] as usize * 16 + t % 16;
            kc[place * w..(place + 1) * w].copy_from_slice(&k[t * w..(t + 1) * w]);
            vc[place * w..(place + 1) * w].copy_from_slice(&v[t * w..(t + 1) * w]);
            slots.push(place as i64);
        }
        ks.extend_from_slice(k);
        vs.extend_from_slice(v);
    }
    let n = slots.len();
    let (kb, vb) = (bases(&ks, n, h, d), bases(&vs, n, h, d));
    let up16 = |x: &[u16]| -> Result<Buf> {
        let b = Buf::new(2 * x.len())?;
        b.write(0, x)?;
        Ok(b)
    };
    let up32 = |x: &[i32]| -> Result<Buf> {
        let b = Buf::new(4 * x.len().max(1))?;
        b.write(0, x)?;
        Ok(b)
    };
    let (kbuf, vbuf, kcb, vcb) = (up16(&ks)?, up16(&vs)?, up16(&kc)?, up16(&vc)?);
    let (kbase, vbase) = (Buf::new(kb.len())?, Buf::new(vb.len())?);
    kbase.write(0, &kb)?;
    vbase.write(0, &vb)?;
    let sl = Buf::new(8 * n)?;
    sl.write(0, &slots)?;
    let page = lib.kv_page_bytes(h as i64, d as i64, A)?;
    let (nchunks, nslabs) = (4096i64, 64i64);
    let pool = Buf::zeroed(page as usize * blocks)?;
    let escape = Buf::zeroed(lib.kv_escape_bytes(nchunks)? as usize)?;
    let raw = Buf::zeroed(lib.kv_raw_bytes(nslabs, (w / 128) as i64)? as usize)?;
    let status = Buf::zeroed(4 * (16 + 2 * 256))?;
    let geom = KvGeom { heads: h as i64, dim: d as i64, arena: A, page_stride: page, escape: escape.ptr, nchunks, raw: raw.ptr, nslabs };
    // SAFETY: live buffers of the sizes the kernels read and write; the stream is synchronized before the results are read.
    unsafe {
        lib.kv_write(&geom, pool.ptr, kbase.ptr, vbase.ptr, 0, kbuf.ptr, w as i64, vbuf.ptr, w as i64, sl.ptr, n as i64, status.ptr, 0, 0, 256, dev.stream)?
    };
    let width = h * g * d;
    let q: Vec<u16> = (0..nb * width).map(|_| bf16_of(((r.below(2000) as f32) - 1000.0) / 5000.0)).collect();
    let (qbuf, tb) = (up16(&q)?, up32(&table)?);
    let (lens, firsts): (Vec<i32>, Vec<i32>) = BATCH.iter().map(|&(l, f)| (l as i32, f as i32)).unzip();
    let (lb, fb) = (up32(&lens)?, up32(&firsts)?);
    // (the engine's plan: its floor, and the splits of its default context)
    lib.kv_floor_set(32)?;
    let splits = lib.kv_decode_splits(1, h as i64, 32768 / 16)?;
    let ws_bytes = lib.kv_decode_workspace(nb as i64, h as i64, g as i64, d as i64, splits)?;
    let ws = Buf::zeroed(ws_bytes)?;
    let scale = 1.0 / (d as f64).sqrt();
    let (wb, wi) = ((2 * width) as u64, 4u64);
    // one call: the pages (bf16 false) or the bf16 cache (with the table, or a ring of `cap` places where `ring`), of `count` sequences from sequence `b0`
    let call = |bf16: bool, ring: Option<(u64, u64, usize)>, b0: usize, count: usize, out: u64| -> Result<()> {
        let (q, t, l, f) = (qbuf.ptr + b0 as u64 * wb, tb.ptr + (b0 * stride) as u64 * wi, lb.ptr + b0 as u64 * wi, fb.ptr + b0 as u64 * wi);
        // SAFETY: as above; the workspace's counters are zero again after a call.
        unsafe {
            match (bf16, ring) {
                (false, _) => lib.kv_decode_from(
                    &geom,
                    g as i64,
                    pool.ptr,
                    kbase.ptr,
                    vbase.ptr,
                    q,
                    width as i64,
                    t,
                    stride as i64,
                    l,
                    f,
                    round,
                    count as i64,
                    splits,
                    scale,
                    out,
                    width as i64,
                    ws.ptr,
                    ws_bytes,
                    dev.stream,
                ),
                (true, None) => lib.kv_decode_bf16(
                    q,
                    width as i64,
                    (kcb.ptr, vcb.ptr, t, stride as i64, 0),
                    (l, f, count as i64, splits),
                    (h as i64, g as i64, d as i64),
                    (scale, round),
                    out,
                    width as i64,
                    (ws.ptr, ws_bytes),
                    dev.stream,
                ),
                (true, Some((rk, rv, cap))) => lib.kv_decode_bf16(
                    q,
                    width as i64,
                    (rk, rv, 0, 0, cap as i64),
                    (l, f, count as i64, splits),
                    (h as i64, g as i64, d as i64),
                    (scale, round),
                    out,
                    width as i64,
                    (ws.ptr, ws_bytes),
                    dev.stream,
                ),
            }
        }
    };
    let read = |o: &Buf| -> Result<Vec<u16>> {
        dev.sync()?;
        let mut x = vec![0u16; o.bytes / 2];
        o.read(0, &mut x)?;
        Ok(x)
    };
    let (o_pages, o_bf) = (Buf::zeroed(2 * nb * width)?, Buf::zeroed(2 * nb * width)?);
    call(false, None, 0, nb, o_pages.ptr)?;
    call(true, None, 0, nb, o_bf.ptr)?;
    let (pages, bf) = (read(&o_pages)?, read(&o_bf)?);
    let mut bad = 0;
    let mut say = |ok: bool, what: String| {
        println!(
            "{} {h} heads x {d}, {g} queries to a head{}, {nb} sequences in one call: {what}",
            if ok { "ok  " } else { "FAIL" },
            if round { ", scores in bf16" } else { "" }
        );
        bad += (!ok) as usize;
    };
    let differ = |a: &[u16], b: &[u16]| a.iter().zip(b).filter(|(x, y)| x != y).count();
    let n_bf = differ(&bf, &pages);
    say(n_bf == 0, format!("over a bf16 cache through the table, {n_bf} of {} values not the pages' bits", pages.len()));
    let o1 = Buf::zeroed(2 * width)?;
    for (b, &(len, first)) in BATCH.iter().enumerate() {
        let mine = &pages[b * width..(b + 1) * width];
        call(false, None, b, 1, o1.ptr)?;
        let alone = read(&o1)?;
        call(true, None, b, 1, o1.ptr)?;
        let alone_bf = read(&o1)?;
        let (x, y) = (differ(&alone, mine), differ(&alone_bf, mine));
        say(x == 0 && y == 0, format!("{len} tokens from {first} alone, {x} values of the pages' and {y} of the bf16 cache's not its bits in the call"));
        if !round {
            let (k, v) = &kv[b];
            let (want, mag) = attention(&q[b * width..(b + 1) * width], k, v, first, len, h, g, d);
            let over = |((x, w), m): ((&u16, &f64), &f64)| over_bound(*x, *w, *m);
            let worst = mine.iter().zip(&want).zip(&mag).map(|t| over(t).0).fold(0f64, f64::max);
            let off = mine.iter().zip(&want).zip(&mag).filter(|&t| !(over(t).0 <= 1.0)).count();
            let needed = mine.iter().zip(&want).zip(&mag).filter(|&t| over(t).1 > 1.0).count();
            say(
                off == 0 && needed <= rounding_allowed(mine.len()),
                format!(
                    "{len} tokens from {first}, {off} values past 1.5% + 0.002 + 2^-8 of their terms of the host's attention (worst {worst:.2} of the bound), {needed} needed the 2^-8 term (at most {})",
                    rounding_allowed(mine.len())
                ),
            );
        }
        if len - first <= CAP {
            // (a ring: token t at t modulo CAP, the later over the earlier)
            let (mut rk, mut rv) = (vec![0u16; CAP * w], vec![0u16; CAP * w]);
            for t in 0..len {
                let p = t % CAP;
                rk[p * w..(p + 1) * w].copy_from_slice(&kv[b].0[t * w..(t + 1) * w]);
                rv[p * w..(p + 1) * w].copy_from_slice(&kv[b].1[t * w..(t + 1) * w]);
            }
            let (rkb, rvb) = (up16(&rk)?, up16(&rv)?);
            call(true, Some((rkb.ptr, rvb.ptr, CAP)), b, 1, o1.ptr)?;
            let ring = read(&o1)?;
            let z = differ(&ring, mine);
            say(z == 0, format!("{len} tokens from {first} over a bf16 ring of {CAP} places, {z} values not the pages' bits"));
        }
    }
    // (the batch as a step's graph pads it (src/dynamic/rows.rs `RowTabs`): each sequence's table copied twice, the call finding it at a stride of two tables, and PAD
    // sequences of no tokens after them; each sequence's bits the call's above, the padding's outputs zeros)
    const PAD: usize = 3;
    let rows = nb + PAD;
    let mut table2 = vec![0i32; rows * 2 * stride];
    for b in 0..nb {
        for j in 0..2 {
            table2[(2 * b + j) * stride..(2 * b + j + 1) * stride].copy_from_slice(&table[b * stride..(b + 1) * stride]);
        }
    }
    let q2: Vec<u16> = q.iter().copied().chain((0..PAD * width).map(|i| bf16_of((i % 7) as f32 - 3.0))).collect();
    let (lens2, firsts2): (Vec<i32>, Vec<i32>) = lens.iter().chain(&[0; PAD]).zip(firsts.iter().chain(&[0; PAD])).map(|(&l, &f)| (l, f)).unzip();
    let (q2b, t2b, l2b, f2b) = (up16(&q2)?, up32(&table2)?, up32(&lens2)?, up32(&firsts2)?);
    let ws2_bytes = lib.kv_decode_workspace(rows as i64, h as i64, g as i64, d as i64, splits)?;
    let ws2 = Buf::zeroed(ws2_bytes)?;
    for bf16 in [false, true] {
        let o2 = Buf::zeroed(2 * rows * width)?;
        // SAFETY: as above, the call's buffers made here for `rows` sequences.
        unsafe {
            if bf16 {
                lib.kv_decode_bf16(
                    q2b.ptr,
                    width as i64,
                    (kcb.ptr, vcb.ptr, t2b.ptr, 2 * stride as i64, 0),
                    (l2b.ptr, f2b.ptr, rows as i64, splits),
                    (h as i64, g as i64, d as i64),
                    (scale, round),
                    o2.ptr,
                    width as i64,
                    (ws2.ptr, ws2_bytes),
                    dev.stream,
                )?
            } else {
                lib.kv_decode_from(
                    &geom,
                    g as i64,
                    pool.ptr,
                    kbase.ptr,
                    vbase.ptr,
                    q2b.ptr,
                    width as i64,
                    t2b.ptr,
                    2 * stride as i64,
                    l2b.ptr,
                    f2b.ptr,
                    round,
                    rows as i64,
                    splits,
                    scale,
                    o2.ptr,
                    width as i64,
                    ws2.ptr,
                    ws2_bytes,
                    dev.stream,
                )?
            }
        }
        let got = read(&o2)?;
        let (x, zeros) = (differ(&got[..nb * width], &pages), got[nb * width..].iter().filter(|&&v| v != 0).count());
        say(
            x == 0 && zeros == 0,
            format!(
                "padded by {PAD} sequences of no tokens, tables at a stride of two{}: {x} values not the call's bits, {zeros} of the padding's not zero",
                if bf16 { ", over the bf16 cache" } else { "" }
            ),
        );
    }
    lib.kv_floor_set(0)?;
    Ok(bad == 0)
}

/// A GPU value's difference from the host's `w`, over the attention checks' bound and over that bound without its last term: 1.5% of the output and 0.002,
/// and 2^-8 of the sum of its terms' magnitudes `m`, which is what the kernel's rounding of each probability to bf16 can move an output by. The first past 1
/// fails the value; the second past 1 is a value that needed the rounding's term, which few may (`rounding_allowed`).
fn over_bound(x: u16, w: f64, m: f64) -> (f64, f64) {
    let d = (f32_of(x) as f64 - w).abs();
    (d / (w.abs() * 0.015 + 2e-3 + m / 256.0), d / (w.abs() * 0.015 + 2e-3))
}

/// How many of a comparison's `n` values may need the rounding's term: one in a hundred (one where it has fewer). On record: one value of a decode
/// comparison, and 0.3 to 0.7% of a prompt chunk's outputs. A comparison where more need it fails: the wider bound is for that rounding, not for a kernel
/// that is off by it everywhere.
fn rounding_allowed(n: usize) -> usize {
    (n / 100).max(1)
}

/// Tokens a GiB of a filled pool against the planner's (#332): `n` pages of a ring of `h` heads of `d` channels (two streams, or one: a latent of `width` values in heads of
/// 128, the rest zero), with the escape chunks and raw slabs the executor gives such a pool (`kvpages::room_of`), filled with every token of values whose share outside
/// their windows is 3 in a thousand (`values`' `far`); the library's sizes are first held to kv12.rs's replicas the planner uses. Ok: the sizes the same, no page void,
/// every value read back, and the pool's tokens a GiB (its pages, escape and raw pools, as allocated) never fewer than the planner's (`plan::sizing::kv12_token_bytes`,
/// every planner's: a pool never holds fewer tokens than a plan promised) and at most 1% more.
fn filled(lib: &Lib, dev: &Dev, h: usize, d: usize, one: bool, width: usize, label: &str) -> Result<bool> {
    use crate::dynamic::kv12::{escape_bytes, page_bytes_of, raw_bytes, KvShape};
    const A: i64 = 15;
    const RING: i64 = 256;
    let n: usize = 256;
    let s = KvShape { ring: 0, heads: h, channels: d, one, window: 0, scale: 1.0, round: false, width };
    let hp = (h * d / 128) as i64;
    let dim = d as i64 | if one { KV_ONE_STREAM } else { 0 };
    let page = lib.kv_page_bytes(h as i64, dim, A)?;
    let (nchunks, nslabs) = crate::dynamic::kv12_room(&s, n as u64);
    let (esc, raw_b) = (lib.kv_escape_bytes(nchunks as i64)?, lib.kv_raw_bytes(nslabs as i64, hp)?);
    let sizes = page as u64 == page_bytes_of(h, d, one) && esc as u64 == escape_bytes(nchunks) && raw_b as u64 == raw_bytes(nslabs, hp as u64);
    let pool = Buf::zeroed(page as usize * n)?;
    let escape = Buf::zeroed(esc as usize)?;
    let raw = Buf::zeroed(raw_b as usize)?;
    let status = Buf::zeroed(4 * (16 + 2 * RING) as usize)?;
    let geom = KvGeom { heads: h as i64, dim, arena: A, page_stride: page, escape: escape.ptr, nchunks: nchunks as i64, raw: raw.ptr, nslabs: nslabs as i64 };
    let mut r = Rng(0x7E57_1D00_0000_0001 ^ (h * 1000 + d + one as usize) as u64);
    let tokens = 16 * n;
    let mk = |r: &mut Rng| -> Vec<u16> {
        let mut x = values(r, tokens, h, d, 3, 0);
        if one {
            for t in 0..tokens {
                for c in width..h * d {
                    x[t * h * d + c] = 0;
                }
            }
        }
        x
    };
    let k = mk(&mut r);
    let v = if one { Vec::new() } else { mk(&mut r) };
    let (kb, vb) = (bases(&k, tokens, h, d), if one { Vec::new() } else { bases(&v, tokens, h, d) });
    let (kbuf, vbuf, kbase, vbase) = (Buf::new(2 * k.len())?, Buf::new(2 * v.len().max(1))?, Buf::new(kb.len())?, Buf::new(vb.len().max(1))?);
    kbuf.write(0, &k)?;
    kbase.write(0, &kb)?;
    if !one {
        vbuf.write(0, &v)?;
        vbase.write(0, &vb)?;
    }
    let slots: Vec<i64> = (0..tokens as i64).collect();
    let sl = Buf::new(8 * tokens)?;
    sl.write(0, &slots)?;
    let stride = (h * d) as i64;
    let ids: Vec<i32> = (0..n as i32).collect();
    let pages = Buf::new(4 * n)?;
    pages.write(0, &ids)?;
    let (out_k, out_v) = (Buf::new(2 * tokens * h * d)?, Buf::new(2 * if one { 1 } else { tokens * h * d })?);
    // SAFETY: live buffers of the sizes the kernels read and write; the stream is synchronized before the results are read.
    unsafe {
        let (vp, vbp) = if one { (0, 0) } else { (vbuf.ptr, vbase.ptr) };
        lib.kv_write(&geom, pool.ptr, kbase.ptr, vbp, 0, kbuf.ptr, stride, vp, stride, sl.ptr, tokens as i64, status.ptr, 0, 0, RING, dev.stream)?;
        lib.kv_unpack(&geom, pool.ptr, kbase.ptr, vbp, pages.ptr, n as i64, out_k.ptr, if one { 0 } else { out_v.ptr }, dev.stream)?;
    }
    dev.sync()?;
    let mut st = [0u32; 16];
    status.read(0, &mut st)?;
    let mut back = vec![0u16; tokens * h * d];
    out_k.read(0, &mut back)?;
    let mut differ = back.iter().zip(&k).filter(|(a, b)| a != b).count();
    if !one {
        out_v.read(0, &mut back)?;
        differ += back.iter().zip(&v).filter(|(a, b)| a != b).count();
    }
    let bytes = (page as usize * n) as f64 + esc as f64 + raw_b as f64 + (kb.len() + vb.len()) as f64;
    let gib = (1u64 << 30) as f64;
    let held = tokens as f64 / (bytes / gib);
    let planned_b = crate::plan::sizing::kv12_token_bytes(h, d, one) as f64;
    let off = held / (gib / planned_b) - 1.0;
    let ok = sizes && differ == 0 && st[11] == 0 && (0.0..0.01).contains(&off);
    println!(
        "{} tokens a GiB, {label}: {held:.0} held by a filled pool of {n} pages ({nchunks} escape chunks, {nslabs} raw slabs; {} pages raw, {} void, {differ} values differ), {:.0} planned ({:+.2}%); \
         the library's sizes {} kv12.rs's",
        if ok { "ok  " } else { "FAIL" },
        st[10],
        st[11],
        gib / planned_b,
        100.0 * off,
        if sizes { "are" } else { "are NOT" }
    );
    Ok(ok)
}

/// One stream: a tensor [n][h][d] held in the keys' layout alone (a latent padded to 5 heads of 128: Kimi-K3's 576 values; keys that are the values: Gemma's global layers), written with no value, read back
/// into one tensor and compared bit for bit; its page is smaller by the V tiles; and the calls that cannot take it refuse it. Written twice: the tokens in one call (a prompt's), and one token a call (a decode
/// step's, the write's path for a token alone in its page); and read back through a list whose places past the pages are -1 (as a latent attention lists a sequence's pages on the device), those places'
/// output left as it was.
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
    for (label, far, wild, alone) in
        [("tame", 15, 0, false), ("wild", 15, 5, false), ("tame, a token a call", 15, 0, true), ("wild, a token a call", 15, 5, true)]
    {
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
        unsafe {
            if alone {
                for t in 0..n {
                    lib.kv_write(
                        &geom,
                        pool.ptr,
                        kbase.ptr,
                        0,
                        0,
                        kbuf.ptr + (t * h * d * 2) as u64,
                        stride,
                        0,
                        0,
                        sl.ptr + 8 * t as u64,
                        1,
                        status.ptr,
                        0,
                        0,
                        RING,
                        dev.stream,
                    )?;
                }
            } else {
                lib.kv_write(&geom, pool.ptr, kbase.ptr, 0, 0, kbuf.ptr, stride, 0, 0, sl.ptr, n as i64, status.ptr, 0, 0, RING, dev.stream)?;
            }
        }
        // (the pages listed, then three places no page is in)
        let ids: Vec<i32> = (0..blocks as i32).chain([-1, -1, -1]).collect();
        let pages = Buf::new(4 * ids.len())?;
        pages.write(0, &ids)?;
        let out = Buf::new(2 * ids.len() * 16 * h * d)?;
        out.fill_async(0xA5, dev.stream)?;
        unsafe { lib.kv_unpack(&geom, pool.ptr, kbase.ptr, 0, pages.ptr, ids.len() as i64, out.ptr, 0, dev.stream)? };
        dev.sync()?;
        let mut head = [0u32; 16];
        status.read(0, &mut head)?;
        let mut back = vec![0u16; ids.len() * 16 * h * d];
        out.read(0, &mut back)?;
        let diff = back[..n * h * d].iter().zip(&k).filter(|(a, b)| a != b).count();
        let touched = back[blocks * 16 * h * d..].iter().filter(|&&x| x != 0xA5A5).count();
        let ok = diff == 0 && touched == 0 && head[11] == 0;
        println!(
            "{} one stream {h} heads x {d} ({hp} pages' heads, {pad} padding channels), {label}: {} values read back, {diff} differ; {touched} values written for the list's places past \
             the pages; page {one} bytes against {two} for two streams; raw pages {}, void pages {}",
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

/// A write whose tokens of one page do not follow one another, made so on purpose: tokens 0-7 to page 0, 8-15 to page 1, 16-23 to page 0 again, values that make
/// pages raw. Two runs of one call name page 0, so neither is written (the write's CTAs would clear, fill and make raw one page at once, and give its chunks and
/// slabs back twice): the status counts both and names the page, page 0 stays as it was, page 1 holds its tokens. Then the same tokens in a call that is right (every
/// page's tokens one after another): no run refused, every value read back. Each time, the pages released give every chunk and slab back and no count goes below
/// zero (a chunk or slab taken twice or given back twice would show).
fn clashing_runs(lib: &Lib, dev: &Dev) -> Result<bool> {
    const A: i64 = 15;
    const RING: i64 = 64;
    let (h, d, n) = (2usize, 128usize, 24usize);
    let hp = (h * d / 128) as i64;
    let page = lib.kv_page_bytes(h as i64, d as i64, A)?;
    let (nchunks, nslabs) = (64i64, 8i64);
    let pool = Buf::zeroed(page as usize * 2)?;
    let escape = Buf::zeroed(lib.kv_escape_bytes(nchunks)? as usize)?;
    let raw = Buf::zeroed(lib.kv_raw_bytes(nslabs, hp)? as usize)?;
    let status = Buf::zeroed(4 * (16 + 2 * RING) as usize)?;
    let geom = KvGeom { heads: h as i64, dim: d as i64, arena: A, page_stride: page, escape: escape.ptr, nchunks, raw: raw.ptr, nslabs };
    let mut r = Rng(0xC1A5_4000_0000_0001);
    let (k, v) = (values(&mut r, n, h, d, 15, 5), values(&mut r, n, h, d, 15, 5));
    let (kb, vb) = (bases(&k, n, h, d), bases(&v, n, h, d));
    let (kbuf, vbuf, kbase, vbase) = (Buf::new(2 * k.len())?, Buf::new(2 * v.len())?, Buf::new(kb.len())?, Buf::new(vb.len())?);
    kbuf.write(0, &k)?;
    vbuf.write(0, &v)?;
    kbase.write(0, &kb)?;
    vbase.write(0, &vb)?;
    let stride = (h * d) as i64;
    let (sl, ids, both) = (Buf::new(8 * n)?, Buf::new(8)?, Buf::new(4 * 2)?);
    ids.write(0, &[0i32, 1])?;
    both.write(0, &[0i32, 1])?;
    let (out_k, out_v) = (Buf::new(2 * 2 * 16 * h * d)?, Buf::new(2 * 2 * 16 * h * d)?);
    let words = |b: &Buf| -> Result<[u32; 16]> {
        let mut w = [0u32; 16];
        b.read(0, &mut w)?;
        Ok(w)
    };
    let mut bad = 0;
    // (the write of `slots`, the two pages read back, then released: the status before the release, the values back, the status after)
    let write = |slots: &[i64]| -> Result<([u32; 16], Vec<u16>, Vec<u16>, [u32; 16])> {
        sl.write(0, slots)?;
        // SAFETY: live buffers of the sizes the kernels read and write; the stream is synchronized before the results are read.
        unsafe {
            lib.kv_write(&geom, pool.ptr, kbase.ptr, vbase.ptr, 0, kbuf.ptr, stride, vbuf.ptr, stride, sl.ptr, n as i64, status.ptr, 0, 0, RING, dev.stream)?;
            lib.kv_unpack(&geom, pool.ptr, kbase.ptr, vbase.ptr, ids.ptr, 2, out_k.ptr, out_v.ptr, dev.stream)?;
        }
        dev.sync()?;
        let before = words(&status)?;
        let (mut bk, mut bv) = (vec![0u16; 2 * 16 * h * d], vec![0u16; 2 * 16 * h * d]);
        out_k.read(0, &mut bk)?;
        out_v.read(0, &mut bv)?;
        // SAFETY: the pool's two pages, which no call reads meanwhile; waited for.
        unsafe { lib.kv_release(&geom, pool.ptr, 2, both.ptr, 2, 0, status.ptr, dev.stream)? };
        dev.sync()?;
        Ok((before, bk, bv, words(&status)?))
    };
    let w = h * d;
    // (token t's row of the page it went to: page p's row o at (p 16 + o) w)
    let same = |back: &[u16], src: &[u16], tokens: std::ops::Range<usize>, row0: usize| {
        tokens.clone().all(|t| back[(row0 + t - tokens.start) * w..][..w] == src[t * w..][..w])
    };
    // (page 0's bytes as the write left them, before the release clears it)
    let page0 = || -> Result<bool> {
        let mut b = vec![0u8; page as usize];
        pool.read(0, &mut b)?;
        Ok(b.iter().all(|&x| x == 0))
    };
    let clash: Vec<i64> = (0..8).chain(16..24).chain(8..16).collect();
    sl.write(0, &clash)?;
    // SAFETY: as in `write`.
    unsafe {
        lib.kv_write(&geom, pool.ptr, kbase.ptr, vbase.ptr, 0, kbuf.ptr, stride, vbuf.ptr, stride, sl.ptr, n as i64, status.ptr, 0, 0, RING, dev.stream)?
    };
    dev.sync()?;
    let untouched = page0()?;
    // (the same call again through `write`: refused again, page 1 rewritten from its first token, both pages read back and released)
    let (st, bk, bv, after) = write(&clash)?;
    let page1 = same(&bk, &k, 8..16, 16) && same(&bv, &v, 8..16, 16);
    let ok = st[14] == 4 && st[15] == 1 && page1 && untouched && after[4] == 0 && after[8] == 0;
    println!(
        "{} two runs of one write name page 0 (tokens 0-7 and 16-23; 8-15 to page 1), twice: {} runs refused, the page named {}; page 0 {}; page 1 read back {}; the pools after \
         the release: {} chunks, {} slabs taken",
        if ok { "ok  " } else { "FAIL" },
        st[14],
        st[15] as i64 - 1,
        if untouched { "untouched" } else { "WRITTEN" },
        if page1 { "whole" } else { "NOT whole" },
        after[4],
        after[8]
    );
    bad += !ok as usize;
    // (the right call: every page's tokens one after another, tokens 0-15 to page 0 and 16-23 to page 1)
    let right: Vec<i64> = (0..n as i64).collect();
    let (st2, bk2, bv2, after2) = write(&right)?;
    let whole = same(&bk2, &k, 0..16, 0) && same(&bk2, &k, 16..24, 16) && same(&bv2, &v, 0..16, 0) && same(&bv2, &v, 16..24, 16);
    let ok2 = st2[14] == 4 && whole && after2[4] == 0 && after2[8] == 0;
    println!(
        "{} the same tokens, page 0's one after another: no run refused ({} as before), every value read back {}; pages raw {}; the pools after the release: {} chunks, {} slabs taken",
        if ok2 { "ok  " } else { "FAIL" },
        st2[14],
        if whole { "the same" } else { "NOT the same" },
        st2[10],
        after2[4],
        after2[8]
    );
    bad += !ok2 as usize;
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
/// against the host's attention from the same values. False where a value is past the bound (`over_bound`: the decode check's and the probabilities' rounding to bf16,
/// 2^-8 of the sum of its terms' magnitudes), or where more than one output in a hundred needed that rounding's term (`rounding_allowed`: over a softmax of tens to hundreds
/// of keys the rounding alone takes some outputs past 1.5% + 0.002, at head 512 too).
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
        let (mut worst, mut off, mut at, mut needed) = (0f64, 0usize, None, 0usize);
        for (i, ((x, w), m)) in got.iter().zip(&want).zip(&mag).enumerate() {
            let (e, tight) = over_bound(*x, *w, *m);
            worst = worst.max(e);
            needed += (tight > 1.0) as usize;
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
            "{} {h} heads x {d}, {g} queries to a head, rows {}..{} (window {}, keys {}{}{}{}): prompt attention, {off} of {} values past 1.5% + 0.002 + 2^-8 of the terms (worst {worst:.2} of the bound), {needed} needed the 2^-8 term (at most {})",
            if off == 0 && needed <= rounding_allowed(got.len()) { "ok  " } else { "FAIL" },
            ch.start,
            ch.start + ch.n,
            ch.window,
            if ch.rows { "of the chunk from its rows" } else { "all from the pages" },
            if ch.round { ", scores bf16" } else { ", scores fp32" },
            if ch.cap > 0.0 { ", soft-capped" } else { "" },
            if ch.sinks { ", sinks" } else { "" },
            got.len(),
            rounding_allowed(got.len())
        );
        bad += (off != 0 || needed > rounding_allowed(got.len())) as usize;
    }
    let mut head = [0u32; 16];
    status.read(0, &mut head)?;
    if head[11] != 0 {
        println!("FAIL {h} heads x {d}: {} void pages", head[11]);
        bad += 1;
    }
    Ok(bad == 0)
}

/// `kvdecode [--lens 40,1024,4096] [--ctx 32768] [--reps 100]`: a decode step's attention of one sequence and the write of its token, microseconds a call in a CUDA graph of --reps,
/// at heads of 128 (8 kv heads, 32 query heads), gemma-4-12B's sliding (8 of 256, 16, a window of 1,024) and global (1 of 512, 16 query heads in two calls' 8) layers
/// and the Qwen3.5 layout's (2 of 256, 8): over 12-bit pages (glyd_gpu_kv_decode_from), over a bf16 cache of the same values (glyd_gpu_kv_decode_bf16), and the token's write
/// into the pages (glyd_gpu_kv_write of one token, as a decode step's); with the engine's plan for a context of --ctx (its floor of 32), the outputs of the two decodes compared
/// bit for bit, and the pages' escape chunks and raw pages after the writes.
pub fn decode_bench(a: &crate::args::Args) -> Result<()> {
    const A: i64 = 15;
    let dev = Dev::new(0)?;
    let lib = Lib::find()?;
    let s = dev.stream;
    let (ctx, reps) = (a.num("ctx", 32768usize)?, a.num("reps", 100usize)?);
    let lens: Vec<usize> = a.flag("lens").unwrap_or("40,1024,4096").split(',').filter_map(|x| x.trim().parse().ok()).filter(|&x| x >= 1).collect();
    if std::env::var_os("GLYD_KV_FLOOR").is_none() {
        lib.kv_floor_set(32)?;
    }
    println!(
        "{} (sm_{}{}): a decode row's attention and its token's write, a context of {ctx} planned, {reps} calls a graph (us a call)",
        dev.name, dev.cc.0, dev.cc.1
    );
    // (what, kv heads, head, query heads, window, the scores in bf16, the model's most kv heads: the engine plans its splits for them)
    let shapes: [(&str, usize, usize, usize, usize, bool, usize); 4] = [
        ("the plain dense kind (8 x 128, 32 queries)", 8, 128, 32, 0, false, 8),
        ("gemma-4-12B sliding (8 x 256, 16 queries, window 1024)", 8, 256, 16, 1024, true, 8),
        ("gemma-4-12B global (1 x 512, 16 queries)", 1, 512, 16, 0, true, 8),
        ("the Qwen3.5 layout (2 x 256, 8 queries)", 2, 256, 8, 0, false, 2),
    ];
    let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
        let graph = dev.capture(|| (0..reps).try_for_each(|_| f()))?;
        // SAFETY: the graph's buffers are alive while it runs (the caller's).
        unsafe { graph.launch(s)? };
        dev.sync()?;
        let t = crate::dev::Timer::new()?;
        t.start(s)?;
        unsafe { graph.launch(s)? };
        Ok(t.stop(s)? as f64 * 1e3 / reps as f64)
    };
    for (what, h, d, nq, window, round, heads) in shapes {
        let (g, sp) = (nq / h, (nq / h).div_ceil(8));
        let splits = lib.kv_decode_splits(1, heads as i64, ctx.div_ceil(16) as i64)?;
        for &len in &lens {
            let blocks = len.div_ceil(16);
            let mut r = Rng(0xDEC0 ^ (len * 7 + d + h) as u64);
            let (k, v) = (near(&mut r, len, h, d, 5), near(&mut r, len, h, d, 5));
            let (kb, vb) = (bases(&k, len, h, d), bases(&v, len, h, d));
            let w = h * d;
            // (the bf16 cache: the same values, token t at place t, the last block's other places zero)
            let (mut kp, mut vp) = (k.clone(), v.clone());
            kp.resize(blocks * 16 * w, 0);
            vp.resize(blocks * 16 * w, 0);
            let up16 = |x: &[u16]| -> Result<Buf> {
                let b = Buf::new(2 * x.len())?;
                b.write(0, x)?;
                Ok(b)
            };
            let up32 = |x: &[i32]| -> Result<Buf> {
                let b = Buf::new(4 * x.len())?;
                b.write(0, x)?;
                Ok(b)
            };
            let (kc, vc) = (up16(&kp)?, up16(&vp)?);
            let (kbase, vbase) = (Buf::new(kb.len())?, Buf::new(vb.len())?);
            kbase.write(0, &kb)?;
            vbase.write(0, &vb)?;
            let page = lib.kv_page_bytes(h as i64, d as i64, A)?;
            let (nchunks, nslabs) = ((blocks * 8).max(1024) as i64, 64i64);
            let pool = Buf::zeroed(page as usize * blocks)?;
            let escape = Buf::zeroed(lib.kv_escape_bytes(nchunks)? as usize)?;
            let raw = Buf::zeroed(lib.kv_raw_bytes(nslabs, (w / 128) as i64)? as usize)?;
            let status = Buf::zeroed(4 * (16 + 2 * 256))?;
            let geom = KvGeom { heads: h as i64, dim: d as i64, arena: A, page_stride: page, escape: escape.ptr, nchunks, raw: raw.ptr, nslabs };
            let slots: Vec<i64> = (0..len as i64).collect();
            let sl = Buf::new(8 * len)?;
            sl.write(0, &slots)?;
            // SAFETY: live buffers of the sizes the kernels read and write; the stream is synchronized before anything is read back.
            unsafe {
                lib.kv_write(&geom, pool.ptr, kbase.ptr, vbase.ptr, 0, kc.ptr, w as i64, vc.ptr, w as i64, sl.ptr, len as i64, status.ptr, 0, 0, 256, s)?
            };
            let bt = up32(&(0..blocks as i32).collect::<Vec<i32>>())?;
            let first = if window > 0 { len.saturating_sub(window) } else { 0 };
            let (lb, fb) = (up32(&vec![len as i32; sp])?, up32(&vec![first as i32; sp])?);
            let q = up16(&(0..nq * d).map(|_| bf16_of(((r.below(2000) as f32) - 1000.0) / 5000.0)).collect::<Vec<u16>>())?;
            let (gs, qs) = ((g / sp) as i64, (nq * d / sp) as i64);
            let ws_bytes = lib.kv_decode_workspace(sp as i64, h as i64, gs, d as i64, splits)?;
            let ws = Buf::zeroed(ws_bytes)?;
            let (op, ob) = (Buf::zeroed(2 * nq * d)?, Buf::zeroed(2 * nq * d)?);
            let scale = 1.0 / (d as f64).sqrt();
            // (a call's sequences are the row's parts of query heads, all through the table's one row, as the executor makes them)
            let pages = || unsafe {
                lib.kv_decode_from(
                    &geom, gs, pool.ptr, kbase.ptr, vbase.ptr, q.ptr, qs, bt.ptr, 0, lb.ptr, fb.ptr, round, sp as i64, splits, scale, op.ptr, qs, ws.ptr,
                    ws_bytes, s,
                )
            };
            let bf16 = || unsafe {
                lib.kv_decode_bf16(
                    q.ptr,
                    qs,
                    (kc.ptr, vc.ptr, bt.ptr, 0, 0),
                    (lb.ptr, fb.ptr, sp as i64, splits),
                    (h as i64, gs, d as i64),
                    (scale, round),
                    ob.ptr,
                    qs,
                    (ws.ptr, ws_bytes),
                    s,
                )
            };
            // (the last token written again, as a decode step writes its own: its old entries killed first, the same bytes after)
            let at = ((len - 1) * w * 2) as u64;
            let write = || unsafe {
                lib.kv_write(
                    &geom,
                    pool.ptr,
                    kbase.ptr,
                    vbase.ptr,
                    0,
                    kc.ptr + at,
                    w as i64,
                    vc.ptr + at,
                    w as i64,
                    sl.ptr + 8 * (len - 1) as u64,
                    1,
                    status.ptr,
                    0,
                    0,
                    256,
                    s,
                )
            };
            let (t_pages, t_bf16, t_write) = (time(&pages)?, time(&bf16)?, time(&write)?);
            dev.sync()?;
            let (mut x, mut y, mut st) = (vec![0u16; nq * d], vec![0u16; nq * d], [0u32; 16]);
            op.read(0, &mut x)?;
            ob.read(0, &mut y)?;
            status.read(0, &mut st)?;
            let differ = x.iter().zip(&y).filter(|(p, q)| p != q).count();
            println!(
                "{what}, {len:>5} tokens ({splits} splits): 12-bit pages {t_pages:>7.2}, bf16 {t_bf16:>7.2} ({:+.0}%), the token's 12-bit write {t_write:>7.2}; {differ} of {} outputs not the pages' bits; escape chunks at most {}, raw pages {}, void pages {}",
                100.0 * (t_pages / t_bf16 - 1.0),
                x.len(),
                st[5],
                st[10],
                st[11]
            );
        }
    }
    Ok(())
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
    // a batch of sequences in one decode call, at gemma-4-12B's sliding (8 x 256, 2 queries a head) and global (1 x 512, 16 queries as two calls' 8) layers, the Qwen3.5
    // layout's (2 x 256, 4) and heads of 128
    for (h, d, g, round) in [(8, 256, 2, true), (1, 512, 8, true), (2, 256, 4, false), (4, 128, 8, false)] {
        ok &= decode_batch(&lib, &dev, h, d, g, round)?;
    }
    // the prompt attention: gemma-4-12B's sliding (8 x 256, 2 queries a head) and global (1 x 512, 16) layers among them
    for (h, d, g) in [(8, 128, 4), (4, 256, 4), (8, 256, 2), (1, 512, 16), (2, 512, 4)] {
        ok &= prefill(&lib, &dev, h, d, g)?;
    }
    // one stream: a 576-value latent in 5 heads of 128 (64 channels of padding), keys that are the values in 2 heads of 512 and one of 256
    for (h, d, pad) in [(5, 128, 64), (2, 512, 0), (1, 256, 0)] {
        ok &= one_stream(&lib, &dev, h, d, pad)?;
    }
    // a write whose tokens of one page do not follow one another: its runs of that page refused and counted, nothing of the pools lost
    ok &= clashing_runs(&lib, &dev)?;
    // tokens a GiB of a filled pool of each format against the planner's (#332): heads of 128, 256, 512 (K=V), and the latents of Kimi-K3 and GLM-5.3 (one stream)
    for (h, d, one, width, label) in [
        (8, 128, false, 1024, "8 heads of 128"),
        (8, 256, false, 2048, "8 heads of 256 (gemma-4-12B sliding)"),
        (1, 512, false, 512, "1 head of 512, K=V (gemma-4-12B global)"),
        (5, 128, true, 576, "a latent of 512 + 64 (Kimi-K3)"),
        (4, 128, true, 512, "a latent of 512 (GLM-5.3)"),
    ] {
        ok &= filled(&lib, &dev, h, d, one, width, label)?;
    }
    println!("{}", if ok { "the lossless KV cache of every head dimension reads back bit for bit and decodes" } else { "FAILED" });
    Ok(ok)
}

#[cfg(test)]
mod tests {
    /// The attention checks' bound has a term for the kernel's rounding of each probability to bf16, and few values may need it: a value past the whole bound
    /// fails, one within it only by that term is counted, and a comparison where more than one in a hundred are so fails.
    #[test]
    fn few_values_may_need_the_roundings_term() {
        use super::{over_bound, rounding_allowed};
        let bf16 = |x: f32| (x.to_bits() >> 16) as u16;
        // (an output of 1.0 where the host's is 1.0: within both; 1.03: past 1.5% + 0.002 alone, within with terms of magnitude 4; 1.1: past both)
        let (e, tight) = over_bound(bf16(1.0), 1.0, 4.0);
        assert!(e == 0.0 && tight == 0.0);
        let (e, tight) = over_bound(bf16(1.03125), 1.0, 4.0);
        assert!(e < 1.0 && tight > 1.0, "{e} {tight}");
        let (e, tight) = over_bound(bf16(1.125), 1.0, 4.0);
        assert!(e > 1.0 && tight > 1.0, "{e} {tight}");
        assert_eq!((rounding_allowed(50), rounding_allowed(100), rounding_allowed(4096)), (1, 1, 40));
    }

    /// An index proof on paper, which runs no kernel (the kernel's own values are what `kvshapes` holds on a GPU): the two closures below are this test's, so a
    /// change to the kernel cannot fail it. The bf16 decode's values (gpu/glyd_gpu_kernels.h `bf16_words`, then `transpose8x8` in attn_kernel): a lane reads words 8 i + q and 8 i + 4 + q of the page's tokens g and g + 8, as
    /// it reads the keys, and one 8 x 8 transpose (movmatrix .trans: lane l holds row l / 4, columns 2 (l % 4) and the next, the first in the low half, before and after) makes each
    /// of V^T's fragments, which the kernel read a value at a time before: f[i].x channel 16 i + g of tokens 2 q and 2 q + 1, .y the channel 8 on, .z and .w the tokens 8 on.
    #[test]
    fn the_values_words_transposed_are_the_gathered_fragments() {
        let v = |t: usize, c: usize| (t * 128 + c) as u32; // (a page's values [16 tokens][128 channels], each its own)
        let word = |t: usize, w: usize| v(t, 2 * w) | v(t, 2 * w + 1) << 16;
        let trans = |w: [u32; 32]| -> [u32; 32] {
            let m = |r: usize, c: usize| (w[4 * r + c / 2] >> (16 * (c % 2))) & 0xFFFF;
            std::array::from_fn(|l| m(2 * (l % 4), l / 4) | m(2 * (l % 4) + 1, l / 4) << 16)
        };
        for i in 0..8 {
            // (the words bf16_words gives a lane (g, q) = (l / 4, l % 4): .x row g word 8 i + q, .y word 8 i + 4 + q, .z and .w row g + 8)
            let read = |row8: usize, w4: usize| trans(std::array::from_fn(|l| word(l / 4 + row8, 8 * i + w4 + l % 4)));
            let (x, y, z, w) = (read(0, 0), read(0, 4), read(8, 0), read(8, 4));
            for l in 0..32 {
                let (g, q) = (l / 4, l % 4);
                let gathered = |t: usize, c: usize| v(t, c) | v(t + 1, c) << 16;
                assert_eq!(x[l], gathered(2 * q, 16 * i + g), "x, step {i}, lane {l}");
                assert_eq!(y[l], gathered(2 * q, 16 * i + 8 + g), "y, step {i}, lane {l}");
                assert_eq!(z[l], gathered(8 + 2 * q, 16 * i + g), "z, step {i}, lane {l}");
                assert_eq!(w[l], gathered(8 + 2 * q, 16 * i + 8 + g), "w, step {i}, lane {l}");
            }
        }
    }
}
