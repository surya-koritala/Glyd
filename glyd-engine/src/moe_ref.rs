//! The CPU reference of the mixture-of-experts layer (src/moe.rs, kernels/moe.cu), one token at a time, from the weights as the GPU decodes them (bf16 values: `unpack_int5` and `unpack_int6` give them bit for bit): the
//! router in fp32, the sums of the products in f64, and bf16 where the layer stores bf16 (the first product's output where the activation is not fused into it, the activation's output, the routed sum, the shared
//! expert's outputs). The experts' weighted sum is added in f64 and rounded once, where the GPU adds fp32 rows. The GPU's kernels are checked against it (src/kcheck_moe.rs). A development build's only.

use crate::moe::{Act, MoeCfg, Router, Scoring};
use crate::qwen35_ref::{bf16, sigmoid, silu};

#[allow(clippy::excessive_precision)] // (sqrt(2 / pi) as the kernels write it)
pub fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.7978845608028654f32 * (x + 0.044715 * x * x * x)).tanh())
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}

/// The router's scores of one token's logits: softmax over the experts, or sigmoid (the logits first rounded to bf16 where the gate is a bf16 Linear).
pub fn scores(r: &Router, logits: &[f32]) -> Vec<f32> {
    let l: Vec<f32> = logits.iter().map(|&x| if r.bf16_logits { bf16(x) } else { x }).collect();
    match r.scoring {
        Scoring::Sigmoid => l.iter().map(|&x| sigmoid(x)).collect(),
        Scoring::SqrtSoftplus => l.iter().map(|&x| (if x > 20.0 { x } else { x.exp().ln_1p() }).sqrt()).collect(),
        Scoring::Softmax => {
            let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = l.iter().map(|&x| (x - m).exp()).collect();
            let z: f32 = e.iter().sum();
            e.iter().map(|&x| x / z).collect()
        }
    }
}

/// One token's k experts (largest score first, equal ones by the smaller expert) and their weights: the scores (+ `bias` to pick, with sigmoid scores), the chosen divided by their sum + 1e-20, times `escale` of
/// the chosen expert and `r.scale`.
pub fn route(r: &Router, logits: &[f32], bias: Option<&[f32]>, escale: Option<&[f32]>) -> (Vec<usize>, Vec<f32>) {
    let s = scores(r, logits);
    let rank: Vec<f32> = (0..s.len()).map(|e| if r.scoring != Scoring::Softmax { s[e] + bias.map_or(0.0, |b| b[e]) } else { s[e] }).collect();
    let mut ord: Vec<usize> = (0..s.len()).collect();
    ord.sort_by(|&a, &b| (rank[b] + 0.0).total_cmp(&(rank[a] + 0.0)).then(a.cmp(&b))); // (+ 0.0: a negative zero is a zero; a NaN sorts, and is no expert)
    ord.truncate(r.top_k);
    let tot: f32 = ord.iter().map(|&e| s[e]).sum();
    let w = ord
        .iter()
        .map(|&e| {
            let mut w = s[e];
            if r.renorm {
                w /= tot + 1e-20;
            }
            if let Some(es) = escale {
                w *= es[e];
            }
            w * r.scale
        })
        .collect();
    (ord, w)
}

/// The activation of the gate and up sums of an MLP (not relu squared): act(min(g, lim)) * clamp(u, -lim, lim).
fn gated(a: &Act, g: f32, u: f32) -> f32 {
    match *a {
        Act::SwiGlu { limit } => {
            let l = limit.unwrap_or(f32::INFINITY);
            silu(g.min(l)) * u.min(l).max(-l)
        }
        Act::GeGlu => gelu_tanh(g) * u,
        Act::Relu2 => unreachable!("relu squared has no gate"),
        Act::Situ { beta, linear } => beta * (g / beta).tanh() / (1.0 + (-g).exp()) * linear * (u / linear).tanh(),
    }
}

/// An MLP's intermediate row [inter] (bf16 values) from its first product's sums (`rows` of them: gate then up, or up alone): where `fused` the activation is applied to the sums and rounded once (the library's
/// product writes it so), else the sums are first rounded to bf16 (the product's output) and the activation, rounded to bf16, is `moe_act`'s.
pub fn intermediate(a: &Act, inter: usize, sums: &[f64], fused: bool) -> Vec<f32> {
    if let Act::Relu2 = a {
        return sums.iter().map(|&s| bf16(bf16(s as f32).max(0.0).powi(2))).collect();
    }
    (0..inter)
        .map(|i| {
            let (g, u) = (sums[i] as f32, sums[inter + i] as f32);
            bf16(if fused { gated(a, g, u) } else { gated(a, bf16(g), bf16(u)) })
        })
        .collect()
}

/// A layer's weights on the host: the router, the experts (`up` [experts * up_rows, hidden], `down` [experts * hidden, inter]) and the optional bias and per-expert scale, all bf16 values (fp32 the last two).
pub struct RefLayer<'a> {
    pub cfg: &'a MoeCfg,
    pub router: &'a [f32],
    pub bias: Option<&'a [f32]>,
    pub escale: Option<&'a [f32]>,
    pub up: &'a [f32],
    pub down: &'a [f32],
}

impl RefLayer<'_> {
    /// One token's router logits [experts] from x [router_hidden]: the sums in f64, rounded to fp32.
    pub fn logits(&self, x: &[f32]) -> Vec<f32> {
        let d = self.cfg.router_hidden;
        (0..self.cfg.experts).map(|e| dot(&self.router[e * d..(e + 1) * d], x) as f32).collect()
    }

    /// The first product's sums [up rows] of expert `e` for one token's x [hidden], in f64.
    pub fn up_sums(&self, e: usize, x: &[f32]) -> Vec<f64> {
        let (rows, d) = (self.cfg.up_rows(), self.cfg.hidden);
        (0..rows).map(|r| dot(&self.up[(e * rows + r) * d..][..d], x)).collect()
    }

    /// The intermediate row [inter] of expert `e` for one token's x [hidden].
    pub fn hidden_row(&self, e: usize, x: &[f32]) -> Vec<f32> {
        intermediate(&self.cfg.act, self.cfg.inter, &self.up_sums(e, x), self.cfg.fused() != 0)
    }

    /// The down product's sums [hidden] of expert `e` for an intermediate row.
    pub fn down_row(&self, e: usize, h: &[f32]) -> Vec<f64> {
        let (d, i) = (self.cfg.hidden, self.cfg.inter);
        (0..d).map(|o| dot(&self.down[(e * d + o) * i..][..i], h)).collect()
    }

    /// The routed experts' sum for one token's x [hidden] and its experts and weights, rounded to bf16 once.
    pub fn token(&self, x: &[f32], ids: &[usize], wts: &[f32]) -> Vec<f32> {
        let mut acc = vec![0f64; self.cfg.hidden];
        for (&e, &w) in ids.iter().zip(wts) {
            let y = self.down_row(e, &self.hidden_row(e, x));
            acc.iter_mut().zip(y).for_each(|(a, v)| *a += v * w as f64);
        }
        acc.into_iter().map(|a| bf16(a as f32)).collect()
    }

    /// The routed sums of `t` tokens (x [t, hidden]; ids [t, k] as the router writes them, -1: none) by as many threads as the machine has.
    pub fn layer(&self, x: &[f32], t: usize, ids: &[i64], wts: &[f32]) -> Vec<f32> {
        let (d, k) = (self.cfg.hidden, self.cfg.router.top_k);
        let mut out = vec![0f32; t * d];
        let per = t.div_ceil(std::thread::available_parallelism().map_or(1, |n| n.get())).max(1);
        std::thread::scope(|sc| {
            for (c, chunk) in out.chunks_mut(per * d).enumerate() {
                sc.spawn(move || {
                    for (i, row) in chunk.chunks_mut(d).enumerate() {
                        let tok = c * per + i;
                        let (id, w): (Vec<usize>, Vec<f32>) = (0..k).filter(|&j| ids[tok * k + j] >= 0).map(|j| (ids[tok * k + j] as usize, wts[tok * k + j])).unzip();
                        row.copy_from_slice(&self.token(&x[tok * d..(tok + 1) * d], &id, &w));
                    }
                });
            }
        });
        out
    }
}

/// The shared expert on the host: `up` [up rows, in] (gate then up, or up alone), `down` [out, inter], the gate [in] where it has one.
pub struct RefShared<'a> {
    pub inter: usize,
    pub up: &'a [f32],
    pub down: &'a [f32],
    pub gate: Option<&'a [f32]>,
}

impl RefShared<'_> {
    /// One token's shared output [out] (bf16, as the GPU stores it) and its gate g (1 where it has none): the first product's output rounded to bf16, the activation (the same as the routed experts'), the second product.
    pub fn row(&self, a: &Act, x: &[f32]) -> (Vec<f32>, f32) {
        let (rows, n) = (if a.gated() { 2 * self.inter } else { self.inter }, x.len());
        let sums: Vec<f64> = (0..rows).map(|r| dot(&self.up[r * n..][..n], x)).collect();
        let h = intermediate(a, self.inter, &sums, false);
        let out_w = self.down.len() / self.inter;
        let so = (0..out_w).map(|o| bf16(dot(&self.down[o * self.inter..][..self.inter], &h) as f32)).collect();
        (so, self.gate.map_or(1.0, |g| sigmoid(dot(g, x) as f32)))
    }

    /// y += g * the shared output, rounded to bf16 once (y a bf16 row).
    pub fn add(&self, a: &Act, x: &[f32], y: &mut [f32]) {
        let (so, g) = self.row(a, x);
        y.iter_mut().zip(so).for_each(|(y, s)| *y = bf16(*y + g * s));
    }
}

/// The plan `Lib::moe_way` makes of the pairs' experts: the experts hit (ascending), where each one's pairs start in the order (and the pairs' count after the last), and the pairs by expert, in order within each (a
/// pair whose expert is outside 0 to experts - 1 is in none).
#[derive(Debug, PartialEq, Eq)]
pub struct Plan {
    pub hits: Vec<i32>,
    pub start: Vec<i32>,
    pub order: Vec<i32>,
}

pub fn plan_of(ids: &[i64], experts: usize) -> Plan {
    let mut by: Vec<Vec<i32>> = vec![Vec::new(); experts];
    for (j, &e) in ids.iter().enumerate() {
        if e >= 0 && (e as usize) < experts {
            by[e as usize].push(j as i32);
        }
    }
    let (mut hits, mut start, mut order) = (Vec::new(), Vec::new(), Vec::new());
    for (e, pairs) in by.iter().enumerate() {
        if !pairs.is_empty() {
            hits.push(e as i32);
            start.push(order.len() as i32);
            order.extend(pairs);
        }
    }
    start.push(order.len() as i32);
    Plan { hits, start, order }
}

/// The plan the device made (int32 [2 + 2 experts + pairs]): its meaningful parts.
pub fn read_plan(raw: &[i32], experts: usize, pairs: usize) -> Plan {
    let h = (raw[0].max(0) as usize).min(experts); // (a plan the device did not make reads as one that differs, not as a panic)
    let n = (raw[1 + experts + h].max(0) as usize).min(pairs);
    Plan { hits: raw[1..1 + h].to_vec(), start: raw[1 + experts..2 + experts + h].to_vec(), order: raw[2 + 2 * experts..2 + 2 * experts + n].to_vec() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-6 * b.abs().max(1.0)
    }

    /// Softmax then the k largest, renormalized: the weights are the chosen probabilities over their sum.
    #[test]
    fn softmax_routing() {
        let r = Router::softmax(2);
        let (ids, w) = route(&r, &[0.0, 1.0, 2.0, 3.0], None, None);
        assert_eq!(ids, vec![3, 2]);
        let (a, b) = (3f32.exp(), 2f32.exp());
        assert!(close(w[0], a / (a + b)) && close(w[1], b / (a + b)), "{w:?}");
        // not renormalized: the probabilities themselves, times the scale
        let (ids, w) = route(&Router { renorm: false, scale: 2.0, ..r }, &[0.0, 1.0, 2.0, 3.0], None, None);
        let z: f32 = [0.0f32, 1.0, 2.0, 3.0].iter().map(|x| x.exp()).sum();
        assert_eq!(ids, vec![3, 2]);
        assert!(close(w[0], 2.0 * 3f32.exp() / z), "{w:?}");
        // a per-expert scale after the renormalization (Gemma 4)
        let (_, w) = route(&r, &[0.0, 1.0, 2.0, 3.0], None, Some(&[1.0, 1.0, 0.5, 3.0]));
        assert!(close(w[0], 3.0 * a / (a + b)) && close(w[1], 0.5 * b / (a + b)), "{w:?}");
    }

    /// DeepSeek's noaux_tc: the bias picks the experts and does not weigh them; the weights are the sigmoid scores renormalized, times the scale.
    #[test]
    fn sigmoid_routing_with_a_selection_bias() {
        let r = Router::sigmoid(2, 2.5);
        let logits = [2.0, 1.0, 0.0, -1.0];
        // without a bias: experts 0 and 1
        assert_eq!(route(&r, &logits, None, None).0, vec![0, 1]);
        // a bias that lifts expert 3 over 1: 3 is picked, and weighed by its own score (0.269), not its biased one
        let (ids, w) = route(&r, &logits, Some(&[0.0, 0.0, 0.0, 2.0]), None);
        assert_eq!(ids, vec![3, 0]);
        let (s3, s0) = (sigmoid(-1.0), sigmoid(2.0));
        assert!(close(w[0], 2.5 * s3 / (s3 + s0)) && close(w[1], 2.5 * s0 / (s3 + s0)), "{w:?}");
        // k = 1 and renormalized: the weight is the scale itself
        let (_, w) = route(&Router { top_k: 1, ..r }, &logits, None, None);
        assert!(close(w[0], 2.5));
    }

    /// Equal scores: the smaller expert first, whatever the scoring; and bf16 logits that round to the same value are equal.
    #[test]
    fn ties_go_to_the_smaller_expert() {
        assert_eq!(route(&Router::softmax(3), &[1.0; 8], None, None).0, vec![0, 1, 2]);
        assert_eq!(route(&Router::sigmoid(3, 1.0), &[0.5; 8], None, None).0, vec![0, 1, 2]);
        // 1.001 and 1.0 are one bf16 value
        assert_eq!(route(&Router::softmax(2), &[0.0, 1.001, 1.0, 0.5], None, None).0, vec![1, 2]);
        assert_eq!(route(&Router { bf16_logits: false, ..Router::softmax(2) }, &[0.0, 1.0, 1.001, 0.5], None, None).0, vec![2, 1]);
    }

    fn cfg(act: Act, inter: usize) -> MoeCfg {
        MoeCfg { experts: 4, router_hidden: 8, hidden: 8, inter, router: Router::softmax(2), act }
    }

    /// Small bf16-valued matrices from a seed.
    fn mat(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                bf16(((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale)
            })
            .collect()
    }

    /// One expert with weight 1 is its MLP: down(act(up x)), the activation on the bf16 sums where it is not fused; a clamp changes what is over it and nothing else.
    #[test]
    fn one_expert_is_its_mlp() {
        let c = cfg(Act::SwiGlu { limit: None }, 6);
        let (up, down, router) = (mat(4 * 12 * 8, 1, 4.0), mat(4 * 8 * 6, 2, 4.0), mat(4 * 8, 3, 1.0));
        let x = mat(8, 4, 8.0);
        let l = RefLayer { cfg: &c, router: &router, bias: None, escale: None, up: &up, down: &down };
        let (e, w) = (2usize, 1.0f32);
        let h: Vec<f32> = (0..6)
            .map(|i| {
                let (g, u) = (dot(&up[(e * 12 + i) * 8..][..8], &x) as f32, dot(&up[(e * 12 + 6 + i) * 8..][..8], &x) as f32);
                bf16(silu(bf16(g)) * bf16(u)) // (not fused: the product's sums are stored in bf16 first)
            })
            .collect();
        let want: Vec<f32> = (0..8).map(|o| bf16(dot(&down[(e * 8 + o) * 6..][..6], &h) as f32)).collect();
        assert_eq!(c.fused(), 0); // (an intermediate size of 6 is not a multiple of 64: the activation is not fused)
        assert_eq!(l.token(&x, &[e], &[w]), want);
        // weights add: two experts of weights 0.25 and 0.5
        let two = l.token(&x, &[1, 3], &[0.25, 0.5]);
        let (y1, y3) = (l.down_row(1, &l.hidden_row(1, &x)), l.down_row(3, &l.hidden_row(3, &x)));
        for o in 0..8 {
            assert_eq!(two[o], bf16((0.25 * y1[o] + 0.5 * y3[o]) as f32));
        }
        // the clamp: a gate over the limit is the limit, an up outside it is cut
        let clamped = MoeCfg { act: Act::SwiGlu { limit: Some(1.0) }, ..c };
        assert_eq!(gated(&clamped.act, 5.0, 3.0), silu(1.0) * 1.0);
        assert_eq!(gated(&clamped.act, -5.0, -3.0), -silu(-5.0));
        assert_eq!(gated(&clamped.act, 0.5, 0.25), silu(0.5) * 0.25);
        let lc = RefLayer { cfg: &clamped, ..l };
        assert_ne!(lc.token(&x, &[e], &[w]), want);
    }

    /// Relu squared without a gate, GELU, and the tokens of a layer each as if alone (the layer's threads cut the tokens in chunks).
    #[test]
    fn activations_and_the_tokens_of_a_layer() {
        let relu = cfg(Act::Relu2, 6);
        assert_eq!(relu.up_rows(), 6);
        assert_eq!(intermediate(&relu.act, 6, &[-2.0, 0.0, 3.0, 0.5, -0.1, 1.5], false), vec![0.0, 0.0, 9.0, 0.25, 0.0, 2.25]);
        assert!((gelu_tanh(1.0) - 0.841192).abs() < 1e-5 && gelu_tanh(0.0) == 0.0);
        for act in [Act::Relu2, Act::GeGlu, Act::SwiGlu { limit: Some(2.0) }] {
            let c = cfg(act, 6);
            let (up, down, router) = (mat(4 * c.up_rows() * 8, 5, 3.0), mat(4 * 8 * 6, 6, 3.0), mat(4 * 8, 7, 1.0));
            let l = RefLayer { cfg: &c, router: &router, bias: None, escale: None, up: &up, down: &down };
            let t = 7;
            let x = mat(t * 8, 8, 4.0);
            let mut ids = Vec::new();
            let mut wts = Vec::new();
            for i in 0..t {
                let (id, w) = route(&c.router, &l.logits(&x[i * 8..(i + 1) * 8]), None, None);
                ids.extend(id.iter().map(|&e| e as i64));
                wts.extend(w);
            }
            let got = l.layer(&x, t, &ids, &wts);
            for i in 0..t {
                let id: Vec<usize> = ids[i * 2..i * 2 + 2].iter().map(|&e| e as usize).collect();
                assert_eq!(got[i * 8..(i + 1) * 8], l.token(&x[i * 8..(i + 1) * 8], &id, &wts[i * 2..i * 2 + 2]), "{act:?} token {i}");
            }
        }
    }

    /// The shared expert: its output added once with the gate's sigmoid, or as it is.
    #[test]
    fn the_shared_expert() {
        let a = Act::SwiGlu { limit: None };
        let (up, down, gate) = (mat(12 * 8, 9, 2.0), mat(8 * 6, 10, 2.0), mat(8, 11, 2.0));
        let x = mat(8, 12, 4.0);
        let s = RefShared { inter: 6, up: &up, down: &down, gate: Some(&gate) };
        let (so, g) = s.row(&a, &x);
        assert!(close(g, sigmoid(dot(&gate, &x) as f32)));
        let mut y = mat(8, 13, 2.0);
        let y0 = y.clone();
        s.add(&a, &x, &mut y);
        for o in 0..8 {
            assert_eq!(y[o], bf16(y0[o] + g * so[o]));
        }
        let plain = RefShared { gate: None, ..s };
        assert_eq!(plain.row(&a, &x).1, 1.0);
    }

    /// The pairs sorted by expert, in order within each.
    #[test]
    fn the_plan_of_the_pairs() {
        let p = plan_of(&[2, 0, 2, 1, 0, 2], 4);
        assert_eq!(p, Plan { hits: vec![0, 1, 2], start: vec![0, 2, 3, 6], order: vec![1, 4, 3, 0, 2, 5] });
        // a pair routed nowhere is in no expert's; the device's layout [hits, hit experts .. E, starts .. E + 1, order] reads back the same
        let p = plan_of(&[3, -1, 3, 9], 4);
        assert_eq!(p, Plan { hits: vec![3], start: vec![0, 2], order: vec![0, 2] });
        let raw = [1, 3, 9, 9, 9, /* starts: */ 0, 2, 9, 9, 9, /* order: */ 0, 2, 7, 7];
        assert_eq!(read_plan(&raw, 4, 4), p);
    }
}
