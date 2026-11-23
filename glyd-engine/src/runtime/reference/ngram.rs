//! Condor's hashed n-gram embedding (`kinds::Ngram`, Qwen3.8-Flash-Next's per-layer embedding): the hash's numbers (transformers'
//! `Qwen4ExpTextNGramEmbedding`: splitmix64 multipliers, prime table sizes), a position's table rows, and the block's body on one token
//! with its sequence's state. The gate's and the convolution's norms are the kind's `norm`, each stream normed on its own.

use super::{dot, norm_groups, sigmoid, silu, Params};
use crate::ffi::{err, Result};
use crate::runtime::kinds::Ngram;

const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

fn splitmix64(v: u64) -> u64 {
    let v = v.wrapping_add(GAMMA);
    let v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    v ^ (v >> 31)
}

/// The multipliers of n-gram layer `n.index`, one an order position: odd, and below (2^63 - 1) / vocab, so that a token times one is a
/// positive int64.
pub fn multipliers(n: &Ngram) -> Vec<u64> {
    let half = ((i64::MAX as u64) / n.vocab.max(1) as u64 / 2).max(1);
    let base = n.seed.wrapping_add(10007u64.wrapping_mul(n.index as u64));
    (0..n.order as u64).map(|i| 2 * (splitmix64(base.wrapping_add(GAMMA.wrapping_mul(i + 1))) % half) + 1).collect()
}

fn is_prime(v: u64) -> bool {
    if v < 2 {
        return false;
    }
    if v % 2 == 0 {
        return v == 2;
    }
    (3..).step_by(2).take_while(|d| d * d <= v).all(|d| v % d != 0)
}

/// The layer's tables: each one's size (the model's table g is the (g + 1)-th prime past `base` - 1) and its first row in the layer's
/// table (cumulative), bigrams' heads first.
pub fn tables(n: &Ngram) -> (Vec<u64>, Vec<u64>) {
    let h = (n.order - 1) * n.heads;
    let mut primes = Vec::with_capacity((n.index + 1) * h);
    let mut p = n.base.saturating_sub(1);
    while primes.len() < (n.index + 1) * h {
        p += 1;
        if is_prime(p) {
            primes.push(p);
        }
    }
    let sizes = primes.split_off(n.index * h);
    let offsets = sizes.iter().scan(0u64, |acc, &s| Some(std::mem::replace(acc, *acc + s))).collect();
    (sizes, offsets)
}

/// The layer table's rows: the tables' sizes summed, padded to a multiple of `pad`.
pub fn rows(n: &Ngram, pad: usize) -> usize {
    (tables(n).0.iter().sum::<u64>() as usize).next_multiple_of(pad.max(1))
}

/// The tokens a position's n-grams read: `ctx` the sequence's tokens before it (the last `order` - 1 at least, oldest first; fewer at
/// the sequence's start), `t` its own. Each order position p (0: the token itself) the token p places back, where no `eos` lies
/// between (the segment's start); else `eos`.
pub fn context(n: &Ngram, before: &[u32], t: u32) -> Vec<u32> {
    let mut out = vec![t];
    let mut cut = false;
    for p in 1..n.order {
        let v = if cut { n.eos } else { before.len().checked_sub(p).map_or(n.eos, |i| before[i]) };
        cut |= v == n.eos;
        out.push(v);
    }
    out
}

/// The table rows of a position whose order positions hold `ctx` (`context`), table by table: each order's mixed hash, int64 as
/// transformers keeps it, taken mod each of its tables' sizes (the remainder of the divisor's sign, as torch's) at the table's offset.
pub fn ids(n: &Ngram, mult: &[u64], sizes: &[u64], offsets: &[u64], ctx: &[u32]) -> Vec<u64> {
    let mut out = Vec::with_capacity(sizes.len());
    for o in 2..=n.order {
        let mixed = (0..o).fold(0u64, |m, p| m ^ (ctx[p] as u64).wrapping_mul(mult[p])) as i64;
        for h in (o - 2) * n.heads..(o - 1) * n.heads {
            out.push(mixed.rem_euclid(sizes[h] as i64) as u64 + offsets[h]);
        }
    }
    out
}

/// A sequence's state in an n-gram layer: its tokens so far (the last `order` - 1 kept) and the convolution's inputs of the last (conv - 1)
/// dilation positions (oldest first, zeros before the sequence).
pub struct NgramState {
    pub before: Vec<u32>,
    pub conv: Vec<Vec<f32>>,
}

impl NgramState {
    pub fn new(n: &Ngram, width: usize) -> NgramState {
        NgramState { before: Vec::new(), conv: vec![vec![0.0; width]; (n.conv - 1) * n.dilation] }
    }
}

/// The body on one token: `token`, and its streams normed by the block's norm `q` [streams hidden]; the output added to the streams.
pub fn step(n: &Ngram, p: &Params, st: &mut NgramState, q: &[f32], token: u32) -> Result<Vec<f32>> {
    let (s, w) = (n.streams, q.len());
    let d = w / s.max(1);
    let heads = (n.order - 1) * n.heads;
    let table = p.t.get("ngram").ok_or_else(|| crate::ffi::Error("no weight for role ngram".into()))?;
    let row = table.shape.get(1).copied().unwrap_or(0);
    if s * d != w || row * heads != n.dim || table.shape.len() != 2 {
        return err(format!("n-gram embedding: {w} values over {s} streams, a table of {:?} for {heads} tables of {}", table.shape, n.dim));
    }
    let ctx = context(n, &st.before, token);
    st.before.push(token);
    let keep = st.before.len().saturating_sub(n.order - 1);
    st.before.drain(..keep);
    let (sizes, offsets) = tables(n);
    let mut e = Vec::with_capacity(n.dim);
    for id in ids(n, &multipliers(n), &sizes, &offsets, &ctx) {
        let r = id as usize;
        if r >= table.shape[0] {
            return err(format!("n-gram embedding: row {r} of a table of {}", table.shape[0]));
        }
        e.extend_from_slice(&table.data[r * row..][..row]);
    }
    let k = norm_groups(&n.norm, &p.prod("key", &e)?, p.opt("norm_key"), d);
    let v = p.prod("value", &e)?;
    if k.len() != w || v.len() != d {
        return err(format!("n-gram embedding: key {} and value {} values for {s} streams of {d}", k.len(), v.len()));
    }
    let mut g = Vec::with_capacity(w);
    for i in 0..s {
        let x = dot(&k[i * d..][..d], &q[i * d..][..d]) / (d as f32).sqrt();
        let x = if x == 0.0 { 0.0 } else { x.abs().max(1e-6).sqrt() * x.signum() };
        g.extend(v.iter().map(|&vv| sigmoid(x) * vv));
    }
    let gn = norm_groups(&n.norm, &g, p.opt("norm_conv"), d);
    let cw = p.vec("conv")?;
    if cw.len() != w * n.conv {
        return err(format!("n-gram embedding: a convolution of {} values for {w} channels of {} taps", cw.len(), n.conv));
    }
    // (tap j reads the input (conv - 1 - j) dilation positions back; the last tap is this position's)
    st.conv.push(gn);
    let past = &st.conv;
    let out = (0..w)
        .map(|c| {
            let y: f32 = (0..n.conv).map(|j| cw[c * n.conv + j] * past[past.len() - 1 - (n.conv - 1 - j) * n.dilation][c]).sum();
            g[c] + silu(y)
        })
        .collect();
    st.conv.remove(0);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::kinds::Norm;
    use crate::runtime::reference::testing::fill;

    fn kind(base: u64, vocab: usize, index: usize) -> Ngram {
        Ngram { order: 3, heads: 8, dim: 256, base, vocab, seed: 1234, index, eos: 2000, streams: 4, conv: 4, dilation: 3, norm: Norm::rms_plus_one(1e-6) }
    }

    /// The hash's numbers are the ones transformers saved with the tiny Qwen3.8-Flash-Next of tests/fixtures/qwen4x_tiny_manifest.json (its
    /// int64 buffers, written by the release's modeling code from the config's seed, base and vocabulary), and the real config's tables
    /// start at the first primes past 20,000,000.
    #[test]
    fn the_hashs_numbers_are_the_checkpoints() {
        let m: serde_json::Value = serde_json::from_str(include_str!("../../../tests/fixtures/qwen4x_tiny_manifest.json")).unwrap();
        let ints = |suffix: &str| -> Vec<u64> {
            let t = m["tensors"].as_array().unwrap().iter().find(|t| t[0].as_str().unwrap().ends_with(suffix)).unwrap();
            t[3].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect()
        };
        let n = kind(1000, 2048, 0);
        assert_eq!(multipliers(&n), ints("layer_multipliers"));
        let (sizes, offsets) = tables(&n);
        assert_eq!((sizes, offsets), (ints("ngram_heads_vocab_sizes"), ints("ngram_heads_offsets")));
        // (16,826 rows in 128 shards of 132)
        assert_eq!(rows(&n, 128), 128 * 132);
        let real = kind(20_000_000, 248_320, 0);
        let (sizes, offsets) = tables(&real);
        assert_eq!((sizes[0], sizes.len(), offsets[1]), (20_000_003, 16, 20_000_003));
        assert!(sizes.windows(2).all(|w| w[0] < w[1]) && sizes.iter().all(|&s| is_prime(s)));
        // (a second n-gram layer's tables are the 16 primes after the first's)
        let (last, next) = (sizes[15], tables(&kind(20_000_000, 248_320, 1)).0[0]);
        assert!(next > last && is_prime(next) && (last + 1..next).all(|v| !is_prime(v)));
        assert!(multipliers(&real).iter().all(|&m| m % 2 == 1 && (m as u128) * 248_320 < (1u128 << 63)));
    }

    /// A position's context stops at the segment's start: the end token itself is read, what lies before it is not; the sequence's start
    /// reads as the end token.
    #[test]
    fn the_context_stops_at_the_segments_start() {
        let n = kind(1000, 2048, 0);
        assert_eq!(context(&n, &[], 7), [7, 2000, 2000]);
        assert_eq!(context(&n, &[5], 7), [7, 5, 2000]);
        assert_eq!(context(&n, &[4, 5], 7), [7, 5, 4]);
        assert_eq!(context(&n, &[4, 2000], 7), [7, 2000, 2000]);
        assert_eq!(context(&n, &[2000, 5], 7), [7, 5, 2000]);
        assert_eq!(context(&n, &[9, 4, 5], 2000), [2000, 5, 4]);
    }

    /// Condor's engine hashes a position as (token, the one before, the one before that unless the one before is the end token) with its
    /// own arithmetic (src/model_qwen4x.rs `Ngram::ids`, unsigned): the same rows, each order's mixed value XOR'd from the token up.
    #[test]
    fn the_rows_by_hand() {
        let n = kind(1000, 2048, 0);
        let (m, (sizes, offsets)) = (multipliers(&n), tables(&n));
        for ctx in [[7u32, 5, 4], [0, 2000, 2000], [2047, 2046, 2045]] {
            let bi = (ctx[0] as u64).wrapping_mul(m[0]) ^ (ctx[1] as u64).wrapping_mul(m[1]);
            let tri = bi ^ (ctx[2] as u64).wrapping_mul(m[2]);
            let want: Vec<u64> = (0..16).map(|h| (if h < 8 { bi } else { tri }) % sizes[h] + offsets[h]).collect();
            assert_eq!(ids(&n, &m, &sizes, &offsets, &ctx), want);
        }
    }

    /// The body by hand on two tokens: the rows, the per-stream gate's signed root, the convolution's taps three positions apart (the
    /// second token's reads the first's input on no tap: they are one position apart), and the state kept.
    #[test]
    fn the_body_by_hand() {
        let (d, s) = (8usize, 2usize);
        // (two tables of rows of 2: a bigram's and a trigram's)
        let n = Ngram {
            order: 3,
            heads: 1,
            dim: 4,
            base: 5,
            vocab: 16,
            seed: 1234,
            index: 0,
            eos: 15,
            streams: s,
            conv: 2,
            dilation: 3,
            norm: Norm::rms_plus_one(1e-6),
        };
        let (sizes, _) = tables(&n);
        let rows = rows(&n, 4);
        let mut p = Params::new();
        p.put("ngram", &[rows, 2], fill("ng.table", rows * 2, -1.0, 1.0));
        p.put("key", &[s * d, 4], fill("ng.key", s * d * 4, -0.5, 0.5));
        p.put("value", &[d, 4], fill("ng.value", d * 4, -0.5, 0.5));
        p.put("norm_key", &[s * d], fill("ng.nk", s * d, -0.2, 0.2));
        p.put("norm_conv", &[s * d], fill("ng.nc", s * d, -0.2, 0.2));
        p.put("conv", &[s * d, 2], fill("ng.conv", s * d * 2, -0.5, 0.5));
        let mut st = NgramState::new(&n, s * d);
        let qs = [fill("ng.q0", s * d, -1.0, 1.0), fill("ng.q1", s * d, -1.0, 1.0)];
        let outs: Vec<Vec<f32>> = [3u32, 9].iter().zip(&qs).map(|(&t, q)| step(&n, &p, &mut st, q, t).unwrap()).collect();
        assert_eq!((st.before.clone(), st.conv.len()), (vec![3, 9], 3));
        let (m, (_, offsets)) = (multipliers(&n), tables(&n));
        let mut inputs = Vec::new();
        for (i, ctx) in [[3u32, 15, 15], [9, 3, 15]].iter().enumerate() {
            let bi = ((ctx[0] as u64 * m[0]) ^ (ctx[1] as u64 * m[1])) % sizes[0] + offsets[0];
            let tri = ((ctx[0] as u64 * m[0]) ^ (ctx[1] as u64 * m[1]) ^ (ctx[2] as u64 * m[2])) % sizes[1] + offsets[1];
            let tab = p.vec("ngram").unwrap();
            let e: Vec<f32> = [bi, tri].iter().flat_map(|&r| tab[r as usize * 2..][..2].to_vec()).collect();
            let k = norm_groups(&n.norm, &p.prod("key", &e).unwrap(), p.opt("norm_key"), d);
            let v = p.prod("value", &e).unwrap();
            let g: Vec<f32> = (0..s)
                .flat_map(|j| {
                    let x = dot(&k[j * d..][..d], &qs[i][j * d..][..d]) / (d as f32).sqrt();
                    let x = x.signum() * x.abs().max(1e-6).sqrt();
                    v.iter().map(move |vv| sigmoid(x) * vv).collect::<Vec<_>>()
                })
                .collect();
            let gn = norm_groups(&n.norm, &g, p.opt("norm_conv"), d);
            // (two taps three positions apart: the earlier tap reads the zeros before the sequence)
            let cw = p.vec("conv").unwrap();
            let want: Vec<f32> = (0..s * d).map(|c| g[c] + silu(cw[c * 2 + 1] * gn[c])).collect();
            assert_eq!(outs[i], want, "token {i}");
            inputs.push(gn);
        }
        assert_eq!(st.conv[2], inputs[1]);
    }
}
