//! Ibis's hashed n-gram embedding (`kinds::Ngram`, Qwen3.8-Flash-Next's per-layer embedding): the hash's numbers (transformers'
//! `Qwen4ExpTextNGramEmbedding`: splitmix64 multipliers, prime table sizes), a position's table rows, and the block's body on one token
//! with its sequence's state. The gate's and the convolution's norms are the kind's `norm`, each stream normed on its own.

use super::{dot, norm_groups, sigmoid, silu, Params};
use crate::ffi::{err, Result};
use crate::runtime::kinds::{Ngram, NgramForm};

const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

fn splitmix64(v: u64) -> u64 {
    let v = v.wrapping_add(GAMMA);
    let v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    v ^ (v >> 31)
}

/// The multipliers of n-gram layer `n.index`, one an order position: odd, and below (2^63 - 1) / vocab, so that a token times one is a
/// positive int64. The segmented form's from splitmix64; Engram's numpy's draws (`NgramForm::Engram`).
pub fn multipliers(n: &Ngram) -> Result<Vec<u64>> {
    let half = ((i64::MAX as u64) / n.vocab.max(1) as u64 / 2).max(1);
    if n.form == NgramForm::Engram {
        let mut r = NumpyRng::new(n.seed);
        return (0..n.order).map(|_| Ok(2 * r.below(half)? + 1)).collect();
    }
    let base = n.seed.wrapping_add(10007u64.wrapping_mul(n.index as u64));
    Ok((0..n.order as u64).map(|i| 2 * (splitmix64(base.wrapping_add(GAMMA.wrapping_mul(i + 1))) % half) + 1).collect())
}

/// numpy's `default_rng(seed)`: its SeedSequence's state (a pool of four 32-bit words hashed from the seed) seeding PCG64, whose 64-bit draws are
/// the XSL-RR output of a 128-bit LCG; and `integers(0, high)` of int64 by Lemire's bounded multiplication. Engram's hash multipliers (DeepSeek-V4.1)
/// are its draws.
pub struct NumpyRng {
    state: u128,
    inc: u128,
}

impl NumpyRng {
    const MUL: u128 = (0x2360_ED05_1FC6_5DA4u128 << 64) | 0x4385_DF64_9FCC_F645;

    pub fn new(seed: u64) -> NumpyRng {
        // (SeedSequence: the seed's 32-bit words, mixed into a pool of 4, then 8 words drawn from it)
        let (init_a, mult_a, init_b, mult_b) = (0x43b0_d7e5u32, 0x931e_8875u32, 0x8b51_f9ddu32, 0x58f3_8dedu32);
        let entropy: Vec<u32> = if seed >> 32 == 0 { vec![seed as u32] } else { vec![seed as u32, (seed >> 32) as u32] };
        let mut hc = init_a;
        let mut hashmix = |v: u32| {
            let mut v = v ^ hc;
            hc = hc.wrapping_mul(mult_a);
            v = v.wrapping_mul(hc);
            v ^ (v >> 16)
        };
        let mix = |x: u32, y: u32| {
            let r = 0xca01_f9ddu32.wrapping_mul(x).wrapping_sub(0x4973_f715u32.wrapping_mul(y));
            r ^ (r >> 16)
        };
        let mut pool = [0u32; 4];
        for (i, p) in pool.iter_mut().enumerate() {
            *p = hashmix(entropy.get(i).copied().unwrap_or(0));
        }
        for src in 0..4 {
            for dst in 0..4 {
                if src != dst {
                    let h = hashmix(pool[src]);
                    pool[dst] = mix(pool[dst], h);
                }
            }
        }
        let mut hb = init_b;
        let words: Vec<u32> = (0..8)
            .map(|i| {
                let mut v = pool[i % 4] ^ hb;
                hb = hb.wrapping_mul(mult_b);
                v = v.wrapping_mul(hb);
                v ^ (v >> 16)
            })
            .collect();
        let w = |k: usize| words[2 * k] as u64 | (words[2 * k + 1] as u64) << 32;
        let (s, i) = ((w(0) as u128) << 64 | w(1) as u128, (w(2) as u128) << 64 | w(3) as u128);
        let mut r = NumpyRng { state: 0, inc: (i << 1) | 1 };
        r.step();
        r.state = r.state.wrapping_add(s);
        r.step();
        r
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(Self::MUL).wrapping_add(self.inc);
    }

    pub fn next_u64(&mut self) -> u64 {
        self.step();
        (((self.state >> 64) as u64) ^ self.state as u64).rotate_right((self.state >> 122) as u32)
    }

    /// `integers(0, high)`: a draw below `high`, which is past 2^32 (numpy takes 32-bit draws below that).
    pub fn below(&mut self, high: u64) -> Result<u64> {
        if high <= 1 << 32 {
            return err(format!("numpy's bounded draw below {high}: the 32-bit path is not this one"));
        }
        let mut m = self.next_u64() as u128 * high as u128;
        if (m as u64) < high {
            let threshold = high.wrapping_neg() % high;
            while (m as u64) < threshold {
                m = self.next_u64() as u128 * high as u128;
            }
        }
        Ok((m >> 64) as u64)
    }
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
/// between (the segment's start); else `eos`. Engram's (`pad`: its pad id as its compressed vocabulary has it) has no segments: only
/// the sequence's start reads as the pad.
pub fn context(n: &Ngram, before: &[u32], t: u32, pad: u32) -> Vec<u32> {
    let segments = n.form == NgramForm::Segmented;
    let mut out = vec![t];
    let mut cut = false;
    for p in 1..n.order {
        let v = if cut { pad } else { before.len().checked_sub(p).map_or(pad, |i| before[i]) };
        cut |= segments && v == n.eos;
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
        NgramState { before: Vec::new(), conv: vec![vec![0.0; width]; n.conv.saturating_sub(1) * n.dilation] }
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
    // (Engram's hash reads the compressed vocabulary's ids, its pad id's too: the release's `token_map[pad]`)
    let (token, pad) = match n.form {
        NgramForm::Segmented => (token, n.eos),
        NgramForm::Engram => {
            let m = p.vec("ngram_map")?;
            let id = |t: u32| match m.get(t as usize).map(|&v| v as u32) {
                Some(v) if (v as usize) < n.vocab => Ok(v),
                v => err(format!("Engram: token {t} maps to {v:?}, in a compressed vocabulary of {}", n.vocab)),
            };
            (id(token)?, id(n.eos)?)
        }
    };
    let ctx = context(n, &st.before, token, pad);
    st.before.push(token);
    let keep = st.before.len().saturating_sub(n.order - 1);
    st.before.drain(..keep);
    let (sizes, offsets) = tables(n);
    let mut e = Vec::with_capacity(n.dim);
    for id in ids(n, &multipliers(n)?, &sizes, &offsets, &ctx) {
        let r = id as usize;
        if r >= table.shape[0] {
            return err(format!("n-gram embedding: row {r} of a table of {}", table.shape[0]));
        }
        e.extend(table.w().row(r * row, row));
    }
    // (the key and the value as one product: the keys' rows first)
    let (k, v) = if p.has("kv") {
        let mut kv = p.prod("kv", &e)?;
        let v = kv.split_off(w.min(kv.len()));
        (kv, v)
    } else {
        (p.prod("key", &e)?, p.prod("value", &e)?)
    };
    let k = norm_groups(&n.norm, &k, p.opt("norm_key"), d);
    if k.len() != w || v.len() != d {
        return err(format!("n-gram embedding: key {} and value {} values for {s} streams of {d}", k.len(), v.len()));
    }
    let mut g = Vec::with_capacity(w);
    for i in 0..s {
        let x = dot(&k[i * d..][..d], &q[i * d..][..d]) / (d as f32).sqrt();
        // (the segmented form's root of 0 is 0; Engram's the clamp's, sqrt(1e-6))
        let x = if x == 0.0 && n.form == NgramForm::Segmented { 0.0 } else { x.abs().max(1e-6).sqrt() * x.signum() };
        g.extend(v.iter().map(|&vv| sigmoid(x) * vv));
    }
    if n.conv == 0 {
        return Ok(g);
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

    /// numpy's generator: `np.random.default_rng(42).random(3)` is 0.7739560485559633, 0.4388784397520523, 0.8585979199113825 (each draw's top
    /// 53 bits over 2^53).
    #[test]
    fn numpys_default_rng_draws() {
        let mut r = NumpyRng::new(42);
        let got: Vec<f64> = (0..3).map(|_| (r.next_u64() >> 11) as f64 / (1u64 << 53) as f64).collect();
        assert_eq!(got[0], 0.7739560485559633, "{got:?}");
        assert!((got[1] - 0.4388784397520523).abs() < 1e-15 && (got[2] - 0.8585979199113825).abs() < 1e-15, "{got:?}");
    }

    fn kind(base: u64, vocab: usize, index: usize) -> Ngram {
        Ngram {
            order: 3,
            heads: 8,
            dim: 256,
            base,
            vocab,
            seed: 1234,
            index,
            eos: 2000,
            streams: 4,
            conv: 4,
            dilation: 3,
            norm: Norm::rms_plus_one(1e-6),
            form: NgramForm::Segmented,
            table: Default::default(),
        }
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
        assert_eq!(multipliers(&n).unwrap(), ints("layer_multipliers"));
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
        assert!(multipliers(&real).unwrap().iter().all(|&m| m % 2 == 1 && (m as u128) * 248_320 < (1u128 << 63)));
    }

    /// A position's context stops at the segment's start: the end token itself is read, what lies before it is not; the sequence's start
    /// reads as the end token.
    #[test]
    fn the_context_stops_at_the_segments_start() {
        let n = kind(1000, 2048, 0);
        assert_eq!(context(&n, &[], 7, n.eos), [7, 2000, 2000]);
        assert_eq!(context(&n, &[5], 7, n.eos), [7, 5, 2000]);
        assert_eq!(context(&n, &[4, 5], 7, n.eos), [7, 5, 4]);
        assert_eq!(context(&n, &[4, 2000], 7, n.eos), [7, 2000, 2000]);
        assert_eq!(context(&n, &[2000, 5], 7, n.eos), [7, 5, 2000]);
        assert_eq!(context(&n, &[9, 4, 5], 2000, n.eos), [2000, 5, 4]);
    }

    /// The family's own engine (deleted in R3d) hashed a position as (token, the one before, the one before that unless the one before is the end token) with
    /// its own arithmetic (unsigned): the same rows, each order's mixed value XOR'd from the token up.
    #[test]
    fn the_rows_by_hand() {
        let n = kind(1000, 2048, 0);
        let (m, (sizes, offsets)) = (multipliers(&n).unwrap(), tables(&n));
        for ctx in [[7u32, 5, 4], [0, 2000, 2000], [2047, 2046, 2045]] {
            let bi = (ctx[0] as u64).wrapping_mul(m[0]) ^ (ctx[1] as u64).wrapping_mul(m[1]);
            let tri = bi ^ (ctx[2] as u64).wrapping_mul(m[2]);
            let want: Vec<u64> = (0..16).map(|h| (if h < 8 { bi } else { tri }) % sizes[h] + offsets[h]).collect();
            assert_eq!(ids(&n, &m, &sizes, &offsets, &ctx), want);
        }
    }

    /// Engram (DeepSeek-V4.1) by hand on three tokens: each token's compressed id (the map), the n-grams over those ids with no segment (the
    /// pad id only before the sequence: the end token cuts nothing), numpy's multipliers (odd, bounded, seeded by 10007 times the layer), the
    /// key and value of one product, and no convolution: the output is the gated value.
    #[test]
    fn engram_by_hand() {
        let (d, s) = (8usize, 2usize);
        let n = Ngram {
            order: 3,
            heads: 1,
            dim: 4,
            base: 5,
            vocab: 10,
            seed: 10007 * 14,
            index: 0,
            eos: 9,
            streams: s,
            conv: 0,
            dilation: 1,
            norm: Norm::rms(1e-6),
            form: NgramForm::Engram,
            table: Default::default(),
        };
        let m = multipliers(&n).unwrap();
        assert!(m.iter().all(|&x| x % 2 == 1 && (x as u128) * 10 < (1u128 << 63)), "{m:?}");
        assert_ne!(m, multipliers(&Ngram { form: NgramForm::Segmented, ..n }).unwrap());
        let (sizes, offsets) = tables(&n);
        let rows = rows(&n, 4);
        let mut p = Params::new();
        p.put("ngram", &[rows, 2], fill("eg.table", rows * 2, -1.0, 1.0));
        p.put("kv", &[(s + 1) * d, 4], fill("eg.kv", (s + 1) * d * 4, -0.5, 0.5));
        p.put("norm_key", &[s * d], fill("eg.nk", s * d, 0.5, 1.5));
        // (a vocabulary of 16 onto 10 ids; token 2, the model's end token, is a token like another)
        p.put("ngram_map", &[16], (0..16).map(|t| (t % 10) as f32).collect());
        let mut st = NgramState::new(&n, s * d);
        let toks = [13u32, 2, 7];
        let qs: Vec<Vec<f32>> = (0..3).map(|i| fill(&format!("eg.q{i}"), s * d, -1.0, 1.0)).collect();
        let outs: Vec<Vec<f32>> = toks.iter().zip(&qs).map(|(&t, q)| step(&n, &p, &mut st, q, t).unwrap()).collect();
        assert_eq!(st.before, [2, 7], "the compressed ids kept");
        for (i, ctx) in [[3u32, 9, 9], [2, 3, 9], [7, 2, 3]].iter().enumerate() {
            let bi = (ctx[0] as u64).wrapping_mul(m[0]) ^ (ctx[1] as u64).wrapping_mul(m[1]);
            let tri = bi ^ (ctx[2] as u64).wrapping_mul(m[2]);
            let tab = p.vec("ngram").unwrap();
            let e: Vec<f32> = [bi % sizes[0] + offsets[0], tri % sizes[1] + offsets[1]].iter().flat_map(|&r| tab[r as usize * 2..][..2].to_vec()).collect();
            let kv = p.prod("kv", &e).unwrap();
            let k = norm_groups(&n.norm, &kv[..s * d], p.opt("norm_key"), d);
            let want: Vec<f32> = (0..s)
                .flat_map(|j| {
                    let x = dot(&k[j * d..][..d], &qs[i][j * d..][..d]) / (d as f32).sqrt();
                    let x = x.signum() * x.abs().max(1e-6).sqrt();
                    kv[s * d..].iter().map(move |vv| sigmoid(x) * vv).collect::<Vec<_>>()
                })
                .collect();
            assert_eq!(outs[i], want, "token {i}");
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
            form: NgramForm::Segmented,
            table: Default::default(),
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
        let (m, (_, offsets)) = (multipliers(&n).unwrap(), tables(&n));
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
