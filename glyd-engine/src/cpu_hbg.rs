//! The CPU reference of Hummingbird's experts for every mixture of experts (docs/design/hummingbird.md, #429; the GPU's are kernels/moe_hbg.cu, src/moe_hbg.rs has the layout): the same products with the same roundings, on the host. It is the
//! kernels' check (each pair's rows against the GPU's) and the host tier's compute for the pairs of experts the GPU has no place for. An expert is two matrices of codebook rows (glyd-gpu's codebook.rs: codes of 2 or 3 bits, a bf16 scale for each run of
//! 32 values, a bf16 codebook of the row's own; a weight is scale * entry, exact in fp32) in the basis of the layer's rotation R = (I x H_b) diag(signs) (src/hb_codec.rs): `gate_up` [U, W] and `down_t` [I, W], any W in runs of 32 and any block b
//! (a power of two from 32 to 1,024 that divides W). The rounding points are the GPU's:
//!
//! - x' = R x in fp32 once a token (`rot_in`: hb_codec's `rotate`, the butterflies of the kernels);
//! - the first product `up`: for each row n of gate_up, the dot product with x' in fp32, rounded to bf16: gu[n]. The activation (`activate`: moe_act's modes) works on the bf16 rows and gives the bf16 activation h [I];
//! - the second product `down`: part [W] (fp32, in the rotated basis, not rounded) = the sum over the neurons n of h[n] times down_t row n, dense (a gated activation has no zeros);
//! - `sum_rot`: the weighted sum of a token's parts in fp32 (fmaf, in the order of its choices), the host's `extra` sums added, then R^T (hb_codec's `unrotate`) and one rounding to bf16; or `unrot` of each pair's part where the family's sum
//!   is not linear (Muse Spark norms each expert's output before weighing it).
//!
//! The order of the float sums is the kernels': a row's first product keeps 32 lane sums, lane l taking the runs l, l + 32, ... in order (a run's sum is the 32 products added from 0 in order, each a fused multiply-add of the codebook entry
//! and x', and the lane adds the run's sum times its scale, fused), the 32 lanes added by the xor tree (offsets 1, 2, 4, 8, 16); an output of the second product is, for each block of 128 outputs, 64 chains of fused multiply-adds of the
//! codebook entry and `activation * scale` over the neurons n = 64 s + p (p the chain, s from 0), the 8 chains of each warp (p = 8 w + g) added by the xor tree over g (offsets 1, 2, 4 of g) and the 8 warps' sums added in order. The bits are
//! those of the kernels where the platform's fused multiply-add is exact (`f32::mul_add`) and its exp and tanh are CUDA's (the activation's: the rounding to bf16 hides the difference but where a value is on the edge of one).

use crate::cpu_hb::CbView;
use crate::ffi::{err, Result};
use crate::hb_codec::{bf16, f32_of, rotate, unrotate, RUN};
use crate::moe::Act;
use crate::moe_hbg::{HbgExpert, HbgHost};

/// Outputs a block of the second product takes (the kernels' `HBG_DOWN_OUT`: 4 runs of 32), and the chains it keeps: 8 warps of 8 neurons.
const OUT: usize = 128;
const CHAINS: usize = 64;

/// The codes of run `run` of row `n`, read from the row's little-endian bitstream (code q at stream bits bits q ..; stream bit i is bit i mod 8 of byte i div 8).
fn run_codes(m: &CbView, n: usize, run: usize) -> [u8; RUN] {
    let bits = m.bits as usize;
    let row = &m.codes[n * (m.cols / 8 * bits)..][..m.cols / 8 * bits];
    let bytes = &row[run * 4 * bits..][..4 * bits];
    let mask = (1u16 << bits) - 1;
    let mut out = [0u8; RUN];
    for (q, o) in out.iter_mut().enumerate() {
        let at = bits * q;
        let two = bytes[at / 8] as u16 | (bytes.get(at / 8 + 1).copied().unwrap_or(0) as u16) << 8;
        *o = ((two >> (at % 8)) & mask) as u8;
    }
    out
}

/// The row's codebook as f32 (the entries past 2^bits are zero).
fn book_of(m: &CbView, n: usize) -> [f32; 8] {
    let mut book = [0f32; 8];
    for (c, b) in book.iter_mut().enumerate().take(1 << m.bits) {
        *b = f32_of(m.entry(n, c));
    }
    book
}

fn check_matrix(m: &CbView, what: &str) -> Result<()> {
    if !m.is_valid() || m.cols % RUN != 0 {
        return err(format!("{what}: {} rows of {} values at {} bits that are not a codebook row layout's", m.rows, m.cols, m.bits));
    }
    Ok(())
}

/// R x for a token's row of x (bf16 values as f32): the layer's signs, then the Hadamard transform of each run of `b` values over sqrt b, in fp32.
pub fn rot_in(x: &[f32], signs: &[i8], b: usize) -> Vec<f32> {
    let mut xr = x.to_vec();
    rotate(&mut xr, signs, b);
    xr
}

/// The first product of one expert: gu [U] (bf16 bits) = each row of `m` (gate_up [U, W]) times xr [W] (fp32, the rotated input), in the kernels' order, rounded to bf16.
pub fn up(m: &CbView, xr: &[f32], gu: &mut [u16]) -> Result<()> {
    up_many(m, &[xr], &mut [gu])
}

/// `up` for several inputs of one expert (its pairs in a call): each row's codes and codebook decoded once for all of them, each input's sums in `up`'s order (its bits).
pub fn up_many(m: &CbView, xrs: &[&[f32]], gus: &mut [&mut [u16]]) -> Result<()> {
    check_matrix(m, "the first matrix")?;
    if xrs.len() != gus.len() || xrs.iter().zip(gus.iter()).any(|(x, g)| x.len() != m.cols || g.len() != m.rows) {
        return err(format!("a first product of {} rows of {} values given {} inputs and {} outputs of other sizes", m.rows, m.cols, xrs.len(), gus.len()));
    }
    let runs = m.cols / RUN;
    let mut lanes = vec![[0f32; 32]; xrs.len()];
    for n in 0..m.rows {
        let book = book_of(m, n);
        lanes.iter_mut().for_each(|l| *l = [0f32; 32]);
        for r in 0..runs {
            let (codes, scale) = (run_codes(m, n, r), f32_of(m.scale(n, r)));
            for (lane, xr) in lanes.iter_mut().zip(xrs) {
                let mut t = 0f32;
                for (c, x) in codes.iter().zip(&xr[r * RUN..(r + 1) * RUN]) {
                    t = book[*c as usize].mul_add(*x, t);
                }
                lane[r % 32] = scale.mul_add(t, lane[r % 32]);
            }
        }
        for (lane, gu) in lanes.iter_mut().zip(gus.iter_mut()) {
            let mut off = 1;
            while off < 32 {
                let prev = *lane;
                for (l, v) in lane.iter_mut().enumerate() {
                    *v = prev[l] + prev[l ^ off];
                }
                off <<= 1;
            }
            gu[n] = bf16(lane[0]);
        }
    }
    Ok(())
}

/// The second product of one expert: part [W] (fp32, the rotated basis) = the sum over the neurons n of h[n] (bf16 bits) times row n of `m` (down_t [I, W]), dense, in the kernels' order.
pub fn down(m: &CbView, h: &[u16], part: &mut [f32]) -> Result<()> {
    down_many(m, &[h], &mut [part])
}

/// `down` for several activations of one expert: each run's codes decoded once for all of them, each one's sums in `down`'s order (its bits).
pub fn down_many(m: &CbView, hs: &[&[u16]], parts: &mut [&mut [f32]]) -> Result<()> {
    check_matrix(m, "the second matrix")?;
    if hs.len() != parts.len() || hs.iter().zip(parts.iter()).any(|(h, p)| h.len() != m.rows || p.len() != m.cols) {
        return err(format!(
            "a second product of {} neurons and {} outputs given {} activations and {} outputs of other sizes",
            m.rows,
            m.cols,
            hs.len(),
            parts.len()
        ));
    }
    let runs = m.cols / RUN;
    let mut acc = vec![0f32; hs.len() * CHAINS * OUT];
    for chunk in 0..runs.div_ceil(OUT / RUN) {
        acc.iter_mut().for_each(|v| *v = 0.0);
        for n in 0..m.rows {
            let (book, chain) = (book_of(m, n), n % CHAINS);
            for r in 0..OUT / RUN {
                let run = chunk * (OUT / RUN) + r;
                if run >= runs {
                    break;
                }
                let (codes, scale) = (run_codes(m, n, run), f32_of(m.scale(n, run)));
                for (v, h) in hs.iter().enumerate() {
                    let c = f32_of(h[n]) * scale;
                    let dst = &mut acc[(v * CHAINS + chain) * OUT + r * RUN..][..RUN];
                    for (d, code) in dst.iter_mut().zip(codes) {
                        *d = book[code as usize].mul_add(c, *d);
                    }
                }
            }
        }
        for (v, part) in parts.iter_mut().enumerate() {
            for o in 0..OUT.min(m.cols - chunk * OUT) {
                let mut total = 0f32;
                for w in 0..CHAINS / 8 {
                    let x = |g: usize| acc[(v * CHAINS + w * 8 + g) * OUT + o];
                    let s = ((x(0) + x(1)) + (x(2) + x(3))) + ((x(4) + x(5)) + (x(6) + x(7)));
                    total = if w == 0 { s } else { total + s };
                }
                part[chunk * OUT + o] = total;
            }
        }
    }
    Ok(())
}

/// The activation between an expert's two products, as kernels/moe.cu's moe_act: h [I] (bf16 bits) from gu (bf16 bits: [2 I], the gate's I values then the up's, for a gated activation; [I] for relu squared), one rounding.
pub fn activate(a: &Act, gu: &[u16], out: &mut [u16]) {
    let inter = out.len();
    match *a {
        Act::Relu2 => {
            debug_assert_eq!(gu.len(), inter);
            for (o, &g) in out.iter_mut().zip(gu) {
                let x = f32_of(g).max(0.0);
                *o = bf16(x * x);
            }
        }
        _ => {
            debug_assert_eq!(gu.len(), 2 * inter);
            let lim = if let Act::SwiGlu { limit: Some(l) } = *a { l } else { f32::INFINITY };
            for (i, o) in out.iter_mut().enumerate() {
                let (g, u) = (f32_of(gu[i]), f32_of(gu[inter + i]));
                *o = bf16(match *a {
                    Act::Situ { beta, linear } => beta * (g / beta).tanh() / (1.0 + (-g).exp()) * linear * (u / linear).tanh(),
                    _ => {
                        let (g, u) = (g.min(lim), u.max(-lim).min(lim));
                        let act =
                            if matches!(a, Act::GeGlu) { 0.5 * g * (1.0 + (0.797_884_6 * (g + 0.044_715 * g * g * g)).tanh()) } else { g / (1.0 + (-g).exp()) };
                        act * u
                    }
                });
            }
        }
    }
}

/// The activation `a` as a function of a pair's first-product rows (the kernels' moe_act), for `pair` and `layer`.
pub fn act_of(a: Act) -> impl Fn(&[u16], &mut [u16]) {
    move |gu, out| activate(&a, gu, out)
}

/// One pair's two products over an expert: xr [W] (fp32, the rotated input) -> the first product (rounded to bf16) -> `act` -> the second product into part [W] (fp32, the rotated basis). `act` maps the first product's rows to the
/// activation's (a family's own: `act_of` is moe_act's).
pub fn pair(e: &HbgExpert, xr: &[f32], act: &dyn Fn(&[u16], &mut [u16]), part: &mut [f32]) -> Result<()> {
    pairs(e, &[xr], act, &mut [part])
}

/// `pair` for several inputs of one expert (its pairs in a call): its bytes read once for all of them (`up_many`, `down_many`), each part `pair`'s bits.
pub fn pairs(e: &HbgExpert, xrs: &[&[f32]], act: &dyn Fn(&[u16], &mut [u16]), parts: &mut [&mut [f32]]) -> Result<()> {
    let mut gu = vec![vec![0u16; e.up.rows]; xrs.len()];
    up_many(&e.up, xrs, &mut gu.iter_mut().map(|g| &mut g[..]).collect::<Vec<_>>())?;
    let mut h = vec![vec![0u16; e.down.rows]; xrs.len()];
    h.iter_mut().zip(&gu).for_each(|(h, g)| act(g, h));
    down_many(&e.down, &h.iter().map(|h| &h[..]).collect::<Vec<_>>(), parts)
}

/// A token's experts' sum: the weighted sum of its pairs' parts (fp32, fmaf in the order given), `extra` (fp32 [W] in the rotated basis, the host's sums of other pairs) added, then R^T and one rounding to bf16 (bits).
pub fn sum_rot(parts: &[&[f32]], wts: &[f32], extra: Option<&[f32]>, signs: &[i8], b: usize) -> Vec<u16> {
    let mut v = vec![0f32; signs.len()];
    for (p, &w) in parts.iter().zip(wts) {
        for (o, x) in v.iter_mut().zip(*p) {
            *o = w.mul_add(*x, *o);
        }
    }
    if let Some(x) = extra {
        v.iter_mut().zip(x).for_each(|(o, x)| *o += *x);
    }
    unrotate(&mut v, signs, b);
    v.iter().map(|&x| bf16(x)).collect()
}

/// A pair's part rotated back and rounded to bf16 (bits): the expert's output in the basis of the model.
pub fn unrot(part: &[f32], signs: &[i8], b: usize) -> Vec<u16> {
    let mut v = part.to_vec();
    unrotate(&mut v, signs, b);
    v.iter().map(|&x| bf16(x)).collect()
}

/// What the expert cache's host gives back for a pair it computes (src/offload.rs `HostHalf::Hbg`): its part times its weight in the rotated basis, which the GPU adds to its own pairs' weighted sum before R^T (the families
/// whose sum is linear in the experts' outputs); or (`Normed`) the pair's output rotated back, rounded to bf16, normed as Muse Spark's sum norms each expert's output (each value times the input scale `w` and rounded to bf16, over
/// the root of their mean square plus `eps`) and times its weight, which the GPU adds after its own sum.
#[derive(Clone, Debug, PartialEq)]
pub enum Finish {
    Rotated,
    Normed { signs: Vec<i8>, rot: usize, w: Vec<f32>, eps: f32 },
}

/// The pairs the expert cache's host computes: pair p is `experts[p]` on the row `tokens[p]` of `xr` ([tokens, W] fp32, the rotated inputs) with the weight `wts[p]`, its two products (`pair`, the activation `act`) finished as `finish`
/// says into row p of `out` [pairs, W] (fp32). On `threads` threads, an expert's pairs a thread at a time (its bytes read once for them all, `pairs`): the values do not depend on them.
#[allow(clippy::too_many_arguments)]
pub fn store_pairs(experts: &[HbgExpert], xr: &[f32], tokens: &[usize], wts: &[f32], act: Act, finish: &Finish, out: &mut [f32], threads: usize) -> Result<()> {
    let n = experts.len();
    let w = experts.first().map_or(0, |e| e.up.cols);
    if tokens.len() != n
        || wts.len() != n
        || out.len() != n * w
        || tokens.iter().any(|&t| (t + 1) * w > xr.len())
        || experts.iter().any(|e| e.up.cols != w || e.down.cols != w)
    {
        return err(format!(
            "{n} pairs of experts of width {w}, {} tokens, {} weights and {} outputs for rotated inputs of {} values",
            tokens.len(),
            wts.len(),
            out.len(),
            xr.len()
        ));
    }
    if let Finish::Normed { signs, w: scale, .. } = finish {
        if signs.len() != w || scale.len() != w {
            return err(format!("a norm of {} weights and {} signs for experts of width {w}", scale.len(), signs.len()));
        }
    }
    let a = act_of(act);
    // (the pairs by expert, an expert its first matrix's codes: a thread takes an expert and reads its bytes once for all its pairs, `pairs`)
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut at: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for (p, e) in experts.iter().enumerate() {
        let g = *at.entry(e.up.codes.as_ptr() as usize).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[g].push(p);
    }
    let finish_row = |p: usize, row: &mut [f32]| match finish {
        Finish::Rotated => row.iter_mut().for_each(|v| *v *= wts[p]),
        Finish::Normed { signs, rot, w: scale, eps } => {
            let y = unrot(row, signs, *rot);
            let v: Vec<f32> = y.iter().zip(scale).map(|(&b, &s)| f32_of(bf16(f32_of(b) * s))).collect();
            let r = 1.0 / (v.iter().map(|x| x * x).sum::<f32>() / w as f32 + eps).sqrt();
            row.iter_mut().zip(&v).for_each(|(o, &x)| *o = wts[p] * (x * r));
        }
    };
    let rows: Vec<std::sync::Mutex<&mut [f32]>> = out.chunks_mut(w.max(1)).map(std::sync::Mutex::new).collect();
    let group = |g: &[usize]| -> Result<()> {
        let xrs: Vec<&[f32]> = g.iter().map(|&p| &xr[tokens[p] * w..(tokens[p] + 1) * w]).collect();
        let mut held: Vec<std::sync::MutexGuard<&mut [f32]>> = g.iter().map(|&p| rows[p].lock().expect("a pair's row")).collect();
        let mut parts: Vec<&mut [f32]> = held.iter_mut().map(|r| &mut r[..]).collect();
        pairs(&experts[g[0]], &xrs, &a, &mut parts)?;
        g.iter().zip(parts).for_each(|(&p, row)| finish_row(p, row));
        Ok(())
    };
    let next = std::sync::atomic::AtomicUsize::new(0);
    let failed = std::sync::Mutex::new(None);
    std::thread::scope(|sc| {
        for _ in 0..threads.clamp(1, groups.len().max(1)) {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= groups.len() {
                    return;
                }
                if let Err(e) = group(&groups[i]) {
                    *failed.lock().expect("the error") = Some(e);
                    return;
                }
            });
        }
    });
    match failed.into_inner().expect("the error") {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// A layer's experts' sum for `x` [t, W] (bf16 values as f32) as a family whose sum is linear in them gives it: each token rotated once, each of its `k` choices (`ids` [t k], an expert below E counting; `wts` [t k]) through `pair` with `act`,
/// the weighted sum of the parts, `extra` ([t, W] fp32 in the rotated basis; none) added, R^T once, one rounding to bf16: y [t, W] (bits).
pub fn layer(host: &HbgHost, x: &[f32], ids: &[i64], wts: &[f32], k: usize, act: &dyn Fn(&[u16], &mut [u16]), extra: Option<&[f32]>) -> Result<Vec<u16>> {
    let (w, b) = (host.shape.width, host.shape.rot);
    if k == 0 || !x.len().is_multiple_of(w) || ids.len() != x.len() / w * k || wts.len() != ids.len() || extra.is_some_and(|e| e.len() != x.len()) {
        return err(format!(
            "{} values of width {w}, {} ids and {} weights ({k} a token), {} extra values",
            x.len(),
            ids.len(),
            wts.len(),
            extra.map_or(0, <[f32]>::len)
        ));
    }
    let mut y = Vec::with_capacity(x.len());
    for (t, row) in x.chunks_exact(w).enumerate() {
        let xr = rot_in(row, &host.signs, b);
        let (mut parts, mut ws) = (Vec::new(), Vec::new());
        for c in 0..k {
            let id = ids[t * k + c];
            if id < 0 || id as usize >= host.shape.experts {
                continue;
            }
            let mut part = vec![0f32; w];
            pair(&host.expert(id as usize)?, &xr, act, &mut part)?;
            parts.push(part);
            ws.push(wts[t * k + c]);
        }
        let refs: Vec<&[f32]> = parts.iter().map(Vec::as_slice).collect();
        y.extend(sum_rot(&refs, &ws, extra.map(|e| &e[t * w..(t + 1) * w]), &host.signs, b));
    }
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hb_codec::decode;
    use crate::moe_hbg::testing::*;

    /// A matrix as the CPU reads it.
    fn view(m: &glyd_gpu::codebook::CbRows) -> CbView<'_> {
        CbView { bits: m.bits, rows: m.rows, cols: m.cols, codes: &m.codes, scales: &m.scales, codebook: &m.codebook }
    }

    /// One bf16 ulp at `v` (2^-7 of its binade).
    fn ulp(v: f32) -> f32 {
        let e = ((v.abs().to_bits() >> 23) & 0xFF) as i32;
        2f32.powi(e.max(1) - 127 - 7)
    }

    /// A layer of experts of `bits` against the float64 product of the weights they decode to in the basis of the model (the rows unrotated): the first product's bf16 rounding is the float64 value's within one bf16 ulp (and mostly the same),
    /// the second product (fp32, not rounded) is the float64 sum to 1e-5 of its norm, and a token's weighted sum and R^T are the float64 ones to one bf16 ulp.
    fn layer_against_float64(width: usize, inter: usize, gated: bool, bits: &[u32], seed: u64) {
        let mut rng = Rng(seed);
        let (rows, _) = coded_layer(&mut rng, width, inter, gated, bits, 3);
        let host = HbgHost::of(&rows).unwrap();
        let (s, rot, u) = (&rows.signs, rows.rot, host.shape.up_rows());
        let act = if gated { Act::SwiGlu { limit: None } } else { Act::Relu2 };
        let x: Vec<f32> = (0..width).map(|_| rng.bf(1.0)).collect();
        let xr = rot_in(&x, s, rot);
        let (mut same, mut total) = (0usize, 0usize);
        let mut outs: Vec<Vec<f64>> = Vec::new();
        let mut parts: Vec<Vec<f32>> = Vec::new();
        for (e, ex) in rows.experts.iter().enumerate() {
            let v = host.expert(e).unwrap();
            // the weights as the model has them: the decoded rows, rotated back
            let (mut w_up, mut w_down) = (decode(&ex.gate_up).unwrap(), decode(&ex.down_t).unwrap());
            unrotate(&mut w_up, s, rot);
            unrotate(&mut w_down, s, rot);
            let mut gu = vec![0u16; u];
            up(&v.up, &xr, &mut gu).unwrap();
            for n in 0..u {
                let (mut dot, mut mag) = (0f64, 0f64);
                for j in 0..width {
                    let p = w_up[n * width + j] as f64 * x[j] as f64;
                    dot += p;
                    mag += p.abs();
                }
                let want = f32_of(bf16(dot as f32));
                let got = f32_of(gu[n]);
                // (within one bf16 ulp of the rounded float64 value; where the sum cancels the fp32 sum's own error, 2e-6 of the products' magnitude, is allowed too)
                assert!((got - want).abs() <= ulp(want) + 2e-6 * mag as f32, "W {width} expert {e} row {n}: {got} against {want}");
                same += (got == want) as usize;
                total += 1;
            }
            let mut h = vec![0u16; inter];
            activate(&act, &gu, &mut h);
            let mut part = vec![0f32; width];
            down(&v.down, &h, &mut part).unwrap();
            let mut back = part.clone();
            unrotate(&mut back, s, rot);
            let want: Vec<f64> = (0..width).map(|o| (0..inter).map(|n| f32_of(h[n]) as f64 * w_down[n * width + o] as f64).sum()).collect();
            let norm = want.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
            let diff = back.iter().zip(&want).map(|(a, b)| (*a as f64 - b).powi(2)).sum::<f64>().sqrt();
            assert!(diff <= 1e-5 * norm, "W {width} expert {e}: the second product is off by {diff:.3e} of {norm:.3e}");
            // (the pair's output in the basis of the model, rounded to bf16 once)
            let rows16 = unrot(&part, s, rot);
            assert!(rows16.iter().zip(&back).all(|(&r, &b)| r == bf16(b)));
            outs.push(want);
            parts.push(part);
        }
        assert!(same * 100 >= total * 98, "W {width}: {same} of {total} first products are the float64 value's rounding");
        // a token's three pairs (the experts in turn, the weights of a router) summed, R^T, one rounding to bf16
        let n = rows.experts.len();
        let (ids, wts) = ([0usize, 1 % n, 2 % n], [0.6f32, 1.3, 0.25]);
        let refs: Vec<&[f32]> = ids.iter().map(|&e| parts[e].as_slice()).collect();
        let y = sum_rot(&refs, &wts, None, s, rot);
        let v64: Vec<f64> = (0..width).map(|o| ids.iter().zip(&wts).map(|(&e, &w)| w as f64 * outs[e][o]).sum()).collect();
        let rms = (v64.iter().map(|v| v * v).sum::<f64>() / width as f64).sqrt() as f32;
        for (o, (&got, want)) in y.iter().zip(&v64).enumerate() {
            let want = f32_of(bf16(*want as f32));
            assert!((f32_of(got) - want).abs() <= ulp(want) + 1e-5 * rms, "W {width} output {o}: {} against {want}", f32_of(got));
        }
        // the whole layer through `layer` is the same pieces: with an id outside the experts (skipped) and the host's extra sums added before R^T
        let ids64: Vec<i64> = vec![ids[0] as i64, ids[1] as i64, -1, ids[2] as i64, 1000, ids[0] as i64];
        let w6 = [wts[0], wts[1], 9.0, wts[2], 9.0, 0.5];
        let x2: Vec<f32> = x.iter().copied().chain((0..width).map(|_| rng.bf(1.0))).collect();
        let extra: Vec<f32> = (0..2 * width).map(|i| if i < width { 0.01 * (i % 7) as f32 } else { 0.0 }).collect();
        let f = act_of(act);
        let got = layer(&host, &x2, &ids64, &w6, 3, &f, Some(&extra)).unwrap();
        assert_eq!(got.len(), 2 * width);
        let manual = |xs: &[f32], picks: &[(usize, f32)], ex: &[f32]| -> Vec<u16> {
            let xr = rot_in(xs, s, rot);
            let ps: Vec<Vec<f32>> = picks
                .iter()
                .map(|&(e, _)| {
                    let mut p = vec![0f32; width];
                    pair(&host.expert(e).unwrap(), &xr, &f, &mut p).unwrap();
                    p
                })
                .collect();
            let refs: Vec<&[f32]> = ps.iter().map(Vec::as_slice).collect();
            sum_rot(&refs, &picks.iter().map(|p| p.1).collect::<Vec<_>>(), Some(ex), s, rot)
        };
        assert_eq!(&got[..width], &manual(&x2[..width], &[(ids[0], wts[0]), (ids[1], wts[1])], &extra[..width])[..], "a skipped id adds nothing");
        assert_eq!(&got[width..], &manual(&x2[width..], &[(ids[2], wts[2]), (ids[0], 0.5)], &extra[width..])[..]);
    }

    /// The products against float64 at the shapes of the design (W = 256 with b = 256, W = 384 with b = 128, W = 3,584 with b = 512), both bit widths in each layer, gated (SwiGLU) and not (relu squared), and the tiny test models' W = 64 (b = 64)
    /// and 96 (b = 32) and a width of five runs (160: b = 32, the second product's last block of 128 outputs has one run).
    #[test]
    fn the_products_are_the_float64_products_of_the_decoded_weights() {
        for (width, inter, gated, bits, seed) in [
            (256usize, 64usize, true, vec![3u32, 2, 3, 2, 2], 1u64),
            (256, 32, false, vec![2, 3, 3, 2], 2),
            (384, 32, true, vec![2, 3, 2, 3], 3),
            (384, 64, false, vec![3, 2, 2, 3, 3], 4),
            (3584, 32, true, vec![3, 2, 2], 5),
            (3584, 16, false, vec![2, 3, 3], 6),
            (64, 16, true, vec![2, 3, 2, 3], 7),
            (64, 32, false, vec![3, 3, 2, 2], 8),
            (96, 24, true, vec![2, 3, 3], 9),
            (160, 16, true, vec![3, 2, 3], 10),
        ] {
            layer_against_float64(width, inter, gated, &bits, seed);
        }
    }

    /// The rows are read from the bytes as the layout says: the codes of a run from the bitstream at both widths, each weight scale times entry; a matrix that is not a layout's is refused, and so is a first product of other sizes.
    /// The expert cache's host pairs (`store_pairs`): a token's rows (each pair's part times its weight, in the rotated basis) summed and given as the `extra` of a sum with no pairs of its own are the layer's sum (within a bf16 ulp:
    /// the additions' order differs); the Muse norm of a pair is its unrotated output's, input-scaled, over the root of the mean square; the threads change no value; a shape that is not the experts' is refused.
    #[test]
    fn the_stores_host_pairs_are_the_layers() {
        let mut rng = Rng(5);
        let (w, k, t) = (256usize, 2usize, 3usize);
        let (rows, _) = coded_layer(&mut rng, w, 64, true, &[2, 3, 2, 3], 1);
        let host = HbgHost::of(&rows).unwrap();
        let x: Vec<f32> = (0..t * w).map(|_| rng.bf(1.0)).collect();
        let ids: Vec<i64> = vec![0, 3, 1, 2, 3, 0];
        let wts: Vec<f32> = (0..t * k).map(|i| 0.25 + i as f32 * 0.1).collect();
        let act = Act::SwiGlu { limit: None };
        let want = layer(&host, &x, &ids, &wts, k, &act_of(act), None).unwrap();
        let xr: Vec<f32> = x.chunks(w).flat_map(|r| rot_in(r, &host.signs, host.shape.rot)).collect();
        let experts: Vec<HbgExpert> = ids.iter().map(|&e| host.expert(e as usize).unwrap()).collect();
        let toks: Vec<usize> = (0..t * k).map(|j| j / k).collect();
        let mut out = vec![0f32; t * k * w];
        store_pairs(&experts, &xr, &toks, &wts, act, &Finish::Rotated, &mut out, 3).unwrap();
        let mut extra = vec![0f32; t * w];
        for (j, row) in out.chunks(w).enumerate() {
            extra[toks[j] * w..][..w].iter_mut().zip(row).for_each(|(a, b)| *a += b);
        }
        let got = layer(&host, &x, &[-1; 6], &wts, k, &act_of(act), Some(&extra)).unwrap();
        for (a, b) in got.iter().zip(&want) {
            let (a, b) = (f32_of(*a), f32_of(*b));
            assert!((a - b).abs() <= ulp(b), "{a} {b}");
        }
        let mut one = vec![0f32; out.len()];
        store_pairs(&experts, &xr, &toks, &wts, act, &Finish::Rotated, &mut one, 1).unwrap();
        assert_eq!(one, out);
        // (experts 0 and 3 have two pairs each, computed together: each pair's row is the one it has alone, to the bit)
        for (j, row) in out.chunks(w).enumerate() {
            let mut part = vec![0f32; w];
            pair(&experts[j], &xr[toks[j] * w..][..w], &act_of(act), &mut part).unwrap();
            assert_eq!(row, &part.iter().map(|v| v * wts[j]).collect::<Vec<_>>()[..], "pair {j}");
        }
        // Muse's norm of a pair
        let scale: Vec<f32> = (0..w).map(|i| 1.0 + (i % 7) as f32 * 0.05).collect();
        let fin = Finish::Normed { signs: host.signs.clone(), rot: host.shape.rot, w: scale.clone(), eps: 1e-6 };
        let mut normed = vec![0f32; t * k * w];
        store_pairs(&experts, &xr, &toks, &wts, act, &fin, &mut normed, 2).unwrap();
        let mut part = vec![0f32; w];
        pair(&experts[1], &xr[..w], &act_of(act), &mut part).unwrap();
        let v: Vec<f32> = unrot(&part, &host.signs, host.shape.rot).iter().zip(&scale).map(|(&b, &s)| f32_of(bf16(f32_of(b) * s))).collect();
        let r = 1.0 / (v.iter().map(|x| x * x).sum::<f32>() / w as f32 + 1e-6).sqrt();
        assert!(normed[w..2 * w].iter().zip(&v).all(|(o, x)| (o - wts[1] * x * r).abs() <= 1e-6 * (1.0 + o.abs())));
        // refused: a row past the inputs, outputs of the wrong size
        assert!(store_pairs(&experts[..1], &xr[..w], &[1], &wts[..1], act, &Finish::Rotated, &mut vec![0f32; w], 1).is_err());
        assert!(store_pairs(&experts[..1], &xr, &[0], &wts[..1], act, &Finish::Rotated, &mut vec![0f32; w - 1], 1).is_err());
    }

    #[test]
    fn rows_are_read_as_the_layout_says() {
        let mut rng = Rng(11);
        for bits in [2u32, 3] {
            let rows = random_layer(&mut rng, 128, 128, 8, false, &[bits], 1);
            let host = HbgHost::of(&rows).unwrap();
            let v = host.expert(0).unwrap();
            let m = &v.up;
            for n in [0usize, 3, 7] {
                for run in 0..4 {
                    let codes = run_codes(m, n, run);
                    for (q, &c) in codes.iter().enumerate() {
                        assert_eq!(c, m.code(n, run * 32 + q), "{bits} bits row {n} run {run} code {q}");
                    }
                }
            }
            let mut gu = vec![0u16; 8];
            assert!(up(m, &[0.0; 128], &mut gu).is_ok());
            assert!(up(m, &[0.0; 96], &mut gu).is_err() && up(m, &[0.0; 128], &mut gu[..7]).is_err());
            let mut part = vec![0f32; 128];
            assert!(
                down(&v.down, &[0u16; 8], &mut part).is_ok()
                    && down(&v.down, &[0u16; 7], &mut part).is_err()
                    && down(&v.down, &[0u16; 8], &mut part[..96]).is_err()
            );
            // zero rows of zero input give zeros; an activation of zeros gives a part of zeros
            assert!(gu.iter().all(|&g| f32_of(g) == 0.0) && part.iter().all(|&p| p == 0.0));
        }
    }

    /// The second product over a width that is not a multiple of 128 and neurons that are not a multiple of 64: the last block's runs past the row are left out, the neurons past I add nothing; against a plain float64 sum over the decoded rows.
    #[test]
    fn the_second_product_over_odd_shapes() {
        let mut rng = Rng(13);
        for (width, inter) in [(160usize, 8usize), (96, 72), (64, 200), (288, 136)] {
            for bits in [2u32, 3] {
                let m = random_matrix(&mut rng, bits, inter, width);
                let h: Vec<u16> = (0..inter).map(|_| bf16(rng.normal())).collect();
                let mut part = vec![0f32; width];
                down(&view(&m), &h, &mut part).unwrap();
                let w = decode(&m).unwrap();
                for o in 0..width {
                    let want: f64 = (0..inter).map(|n| f32_of(h[n]) as f64 * w[n * width + o] as f64).sum();
                    let mag: f64 = (0..inter).map(|n| (f32_of(h[n]) as f64 * w[n * width + o] as f64).abs()).sum();
                    assert!((part[o] as f64 - want).abs() <= 5e-6 * mag.max(1e-9), "{width} x {inter} at {bits} bits, output {o}: {} against {want}", part[o]);
                }
            }
        }
    }

    /// moe_act's modes on a few values: relu squared of x, SwiGLU (silu(gate) up, the gate at most the clamp and the up within it), GeGLU, situ; one rounding to bf16.
    #[test]
    fn the_activations_are_moe_acts() {
        let b = |v: f32| bf16(v);
        let f = |v: u16| f32_of(v);
        let mut out = [0u16; 3];
        activate(&Act::Relu2, &[b(-1.5), b(2.0), b(0.5)], &mut out);
        assert_eq!(out.map(f), [0.0, 4.0, 0.25]);
        let gu = [b(1.0), b(-2.0), b(20.0), b(3.0), b(0.5), b(-30.0)];
        activate(&Act::SwiGlu { limit: None }, &gu, &mut out);
        let silu = |g: f32| g / (1.0 + (-g).exp());
        for (i, (g, u)) in [(1.0f32, 3.0f32), (-2.0, 0.5), (20.0, -30.0)].into_iter().enumerate() {
            assert_eq!(f(out[i]), f(b(silu(g) * u)), "{i}");
        }
        // the clamp: the gate at most 10, the up within -10 and 10
        activate(&Act::SwiGlu { limit: Some(10.0) }, &gu, &mut out);
        assert_eq!(f(out[2]), f(b(silu(10.0) * -10.0)));
        assert_eq!(f(out[0]), f(b(silu(1.0) * 3.0)));
        activate(&Act::GeGlu, &gu, &mut out);
        let gelu = |g: f32| 0.5 * g * (1.0 + (0.797_884_6 * (g + 0.044_715 * g * g * g)).tanh());
        assert_eq!(f(out[1]), f(b(gelu(-2.0) * 0.5)));
        activate(&Act::Situ { beta: 4.0, linear: 25.0 }, &gu, &mut out);
        let situ = |g: f32, u: f32| 4.0 * (g / 4.0).tanh() / (1.0 + (-g).exp()) * 25.0 * (u / 25.0).tanh();
        assert_eq!(f(out[0]), f(b(situ(1.0, 3.0))));
        assert_eq!(f(out[2]), f(b(situ(20.0, -30.0))));
    }

    /// An expert gives the same bits wherever its bytes are: its group's stack with another expert before it, or the layer's order changed (what the kernels promise of any place in memory).
    #[test]
    fn an_expert_gives_the_same_bits_in_any_place() {
        let mut rng = Rng(17);
        let rows = random_layer(&mut rng, 256, 256, 32, true, &[3, 3, 2, 3], 2);
        let mut flipped = rows.clone();
        flipped.experts.reverse();
        let (a, b) = (HbgHost::of(&rows).unwrap(), HbgHost::of(&flipped).unwrap());
        let x: Vec<f32> = (0..256).map(|_| rng.bf(1.0)).collect();
        let xr = rot_in(&x, &rows.signs, rows.rot);
        let f = act_of(Act::SwiGlu { limit: None });
        for e in 0..4 {
            let (mut p, mut q) = (vec![0f32; 256], vec![0f32; 256]);
            pair(&a.expert(e).unwrap(), &xr, &f, &mut p).unwrap();
            pair(&b.expert(3 - e).unwrap(), &xr, &f, &mut q).unwrap();
            assert!(p.iter().zip(&q).all(|(x, y)| x.to_bits() == y.to_bits()), "expert {e}");
            assert!(p.iter().any(|&v| v != 0.0));
        }
    }

    // ---- the kernels' arithmetic, thread by thread -----------------------------------------------------------------------------------------------------------------------------------------------------------
    // What kernels/moe_hbg.cu does, written out lane by lane from the kernels' source (not from the functions above), so that the host tests hold the kernels' indexing, partial warps, guards and sums to the reference's bits: they are
    // not the GPU, but a kernel whose arithmetic these emulations get wrong would be wrong on it.

    /// `hbg_wht` for a block of `vals.len()` threads (four values each) over runs of `b` values: the in-thread stages, the lane stages (xor shuffles, the partner's value before the stage) while the stage's bit is below th = b / 4, then the
    /// stages through shared memory.
    fn emu_wht(vals: &mut [[f32; 4]], b: usize) {
        let th = b / 4;
        for v in vals.iter_mut() {
            let (w0, w1, w2, w3) = (v[0], v[1], v[2], v[3]);
            *v = [w0 + w1, w0 - w1, w2 + w3, w2 - w3];
            let (w0, w1, w2, w3) = (v[0], v[1], v[2], v[3]);
            *v = [w0 + w2, w1 + w3, w0 - w2, w1 - w3];
        }
        let mut m = 1;
        while m < 32 && m < th {
            let old = vals.to_vec();
            for (tid, v) in vals.iter_mut().enumerate() {
                let hi = (tid & 31) & m != 0;
                for q in 0..4 {
                    let o = old[tid ^ m][q];
                    v[q] = if hi { o - v[q] } else { v[q] + o };
                }
            }
            m <<= 1;
        }
        let mut m = 32;
        while m < th {
            let old = vals.to_vec();
            for (tid, v) in vals.iter_mut().enumerate() {
                for q in 0..4 {
                    let o = old[tid ^ m][q];
                    v[q] = if tid & m != 0 { o - v[q] } else { v[q] + o };
                }
            }
            m <<= 1;
        }
    }

    /// A token's row through `hbg_rot_in`'s grid (blocks of max(32, b / 4) threads, ceil(W / (4 threads)) of them).
    fn emu_rot(x: &[f32], signs: &[i8], b: usize, scale: f32) -> Vec<f32> {
        let (w, threads) = (x.len(), (b / 4).max(32));
        let mut out = vec![f32::NAN; w];
        for by in 0..w.div_ceil(4 * threads) {
            let mut vals: Vec<[f32; 4]> = (0..threads)
                .map(|tid| {
                    let c = by * 4 * threads + 4 * tid;
                    let mut v = [0f32; 4];
                    if c < w {
                        (0..4).for_each(|q| v[q] = signs[c + q] as f32 * x[c + q]);
                    }
                    v
                })
                .collect();
            emu_wht(&mut vals, b);
            for (tid, v) in vals.iter().enumerate() {
                let c = by * 4 * threads + 4 * tid;
                if c < w {
                    (0..4).for_each(|q| out[c + q] = scale * v[q]);
                }
            }
        }
        out
    }

    /// `hb_code`: code q of a run held as BITS 32-bit words (a code of 3 bits can straddle two).
    fn emu_code(w: &[u32], bits: usize, q: usize) -> usize {
        let (b, wi) = (bits * q, bits * q >> 5);
        let sh = b & 31;
        let mut c = w[wi] >> sh;
        if sh + bits > 32 {
            c |= w[wi + 1] << (32 - sh);
        }
        (c & ((1 << bits) - 1)) as usize
    }

    /// `hbg_up`'s block for one pair: the 32 lanes of each warp of 8 neurons' rows, a lane's runs l, l + 32, ..; the codes as 32-bit words, the table of the row; the xor tree.
    fn emu_up(m: &CbView, xr: &[f32]) -> Vec<u16> {
        let (r_runs, bits) = (m.cols / 32, m.bits as usize);
        let mut gu = vec![0u16; m.rows];
        for (n, out) in gu.iter_mut().enumerate() {
            let tab: Vec<f32> = (0..1usize << bits).map(|c| f32_of(m.entry(n, c))).collect();
            let row = &m.codes[n * r_runs * 4 * bits..][..r_runs * 4 * bits];
            let mut accs = [0f32; 32];
            for (lane, acc) in accs.iter_mut().enumerate() {
                for mm in 0..8 {
                    let r = lane + 32 * mm;
                    if r < r_runs {
                        let words: Vec<u32> = (0..bits).map(|w| u32::from_le_bytes(row[4 * bits * r + 4 * w..][..4].try_into().unwrap())).collect();
                        let mut t = 0f32;
                        for q in 0..32 {
                            t = tab[emu_code(&words, bits, q)].mul_add(xr[32 * r + q], t);
                        }
                        *acc = f32_of(m.scale(n, r)).mul_add(t, *acc);
                    }
                }
            }
            let mut off = 1;
            while off < 32 {
                let old = accs;
                (0..32).for_each(|l| accs[l] += old[l ^ off]);
                off <<= 1;
            }
            *out = bf16(accs[0]);
        }
        gu
    }

    /// `hbg_down`'s block for one pair and each block of 128 outputs: 8 warps of 32 lanes (g = lane / 4, r = lane % 4), the neurons n = 64 s + 8 warp + g, a lane's 32 accumulators over its neurons in the order of s, the xor tree over the lane
    /// bits 2 to 4, the warps' sums in the order of the warps.
    fn emu_down(m: &CbView, h: &[u16]) -> Vec<f32> {
        let (runs, bits, inter) = (m.cols / 32, m.bits as usize, m.rows);
        let row_bytes = runs * 4 * bits;
        let mut part = vec![f32::NAN; m.cols];
        for chunk in 0..m.cols.div_ceil(128) {
            let mut red = vec![[0f32; 128]; 8];
            for (warp, red_w) in red.iter_mut().enumerate() {
                let mut acc = vec![[0f32; 32]; 32]; // [lane][q]
                for lane in 0..32 {
                    let (g, r) = (lane >> 2, lane & 3);
                    let run = chunk * 4 + r;
                    for s in 0..inter.div_ceil(64) {
                        let n = s * 64 + warp * 8 + g;
                        if n < inter && run < runs {
                            let tab: Vec<f32> = (0..1usize << bits).map(|c| f32_of(m.entry(n, c))).collect();
                            let words: Vec<u32> =
                                (0..bits).map(|w| u32::from_le_bytes(m.codes[n * row_bytes + 4 * bits * run + 4 * w..][..4].try_into().unwrap())).collect();
                            let c = f32_of(h[n]) * f32_of(m.scale(n, run));
                            for q in 0..32 {
                                acc[lane][q] = tab[emu_code(&words, bits, q)].mul_add(c, acc[lane][q]);
                            }
                        }
                    }
                }
                for off in [4usize, 8, 16] {
                    let old = acc.clone();
                    for lane in 0..32 {
                        for q in 0..32 {
                            acc[lane][q] += old[lane ^ off][q];
                        }
                    }
                }
                for lane in (0..32).filter(|l| l >> 2 == 0) {
                    for q in 0..32 {
                        red_w[(lane & 3) * 32 + q] = acc[lane][q];
                    }
                }
            }
            for tid in 0..128 {
                let mut sum = red[0][tid];
                for w in 1..8 {
                    sum += red[w][tid];
                }
                if chunk * 128 + tid < m.cols {
                    part[chunk * 128 + tid] = sum;
                }
            }
        }
        part
    }

    /// The rotation kernels' grids (any block from 32 to 1,024, partial warps for blocks below 128, partial blocks past the width) give the codec's rotation and its inverse, bit for bit.
    #[test]
    fn the_rotation_kernels_give_the_codecs_bits() {
        let mut rng = Rng(23);
        for (w, b) in
            [(64usize, 64usize), (96, 32), (160, 32), (128, 128), (384, 128), (2560, 512), (2688, 128), (3584, 512), (4096, 1024), (256, 32), (512, 64)]
        {
            let signs = crate::hb_codec::signs(4, w);
            let x: Vec<f32> = (0..w).map(|_| rng.bf(1.0)).collect();
            let got = emu_rot(&x, &signs, b, crate::moe_hbg::scale_of(b));
            assert!(got.iter().zip(rot_in(&x, &signs, b)).all(|(a, e)| a.to_bits() == e.to_bits()), "rot_in at {w} / {b}");
            // (R^T: signs o (scale H v), what hbg_sum_rot and hbg_unrot do after the sums)
            let mut vals_in: Vec<f32> = (0..w).map(|_| rng.normal()).collect();
            vals_in.iter_mut().for_each(|v| *v = f32_of(bf16(*v)));
            let threads = (b / 4).max(32);
            let mut back = vec![0f32; w];
            for by in 0..w.div_ceil(4 * threads) {
                let mut vals: Vec<[f32; 4]> = (0..threads)
                    .map(|tid| {
                        let c = by * 4 * threads + 4 * tid;
                        if c < w {
                            [vals_in[c], vals_in[c + 1], vals_in[c + 2], vals_in[c + 3]]
                        } else {
                            [0.0; 4]
                        }
                    })
                    .collect();
                emu_wht(&mut vals, b);
                for (tid, v) in vals.iter().enumerate() {
                    let c = by * 4 * threads + 4 * tid;
                    if c < w {
                        (0..4).for_each(|q| back[c + q] = signs[c + q] as f32 * (crate::moe_hbg::scale_of(b) * v[q]));
                    }
                }
            }
            let want = unrot(&vals_in, &signs, b);
            assert!(back.iter().zip(&want).all(|(&a, &e)| bf16(a) == e), "R^T at {w} / {b}");
        }
    }

    /// The first and second products' kernels, lane by lane, give the reference's bits: both widths, widths of 2 to 112 runs (64, 96, 160, 384, 2,688: 84 runs, 3,584: 112 runs, and 8,192: 256 runs, a lane's 8 runs), neurons that are not a multiple of 64.
    #[test]
    fn the_products_kernels_give_the_references_bits() {
        let mut rng = Rng(29);
        for (width, inter) in [(64usize, 16usize), (96, 24), (160, 40), (256, 72), (384, 32), (2688, 16), (3584, 8), (8192, 8)] {
            let rot = crate::hb_codec::block_of(width).unwrap();
            let rows = random_layer(&mut rng, width, rot, inter, true, &[2, 3], 6);
            let host = HbgHost::of(&rows).unwrap();
            let x: Vec<f32> = (0..width).map(|_| rng.bf(1.0)).collect();
            let xr = rot_in(&x, &rows.signs, rot);
            for e in 0..2 {
                let v = host.expert(e).unwrap();
                let mut gu = vec![0u16; v.up.rows];
                up(&v.up, &xr, &mut gu).unwrap();
                assert_eq!(emu_up(&v.up, &xr), gu, "the first product at W {width}, expert {e}");
                let h: Vec<u16> = (0..inter).map(|_| bf16(rng.normal())).collect();
                let mut part = vec![0f32; width];
                down(&v.down, &h, &mut part).unwrap();
                let got = emu_down(&v.down, &h);
                assert!(got.iter().zip(&part).all(|(a, b)| a.to_bits() == b.to_bits()), "the second product at W {width}, expert {e}");
            }
        }
    }
}
