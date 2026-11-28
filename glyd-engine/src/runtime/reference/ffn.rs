//! Feed-forward kinds: the dense MLP (`kinds::Mlp`) and the mixture of experts (`kinds::Moe`) with its router variants, shared experts
//! and latent projection. The routed sum is over the chosen experts in the router's order; equal scores pick the smaller expert.

use super::{gated, matvec_w, norm::norm, sigmoid, softmax, softplus, Params};
use crate::ffi::{err, Result};
use crate::runtime::kinds::{Clamp, Mlp, Moe, Router, RouterScore};

/// act(gate) * up, or act(up), from a gated MLP's two sums, with the SwiGLU limits (`Clamp`: the gate held before its activation, or the activated gate).
fn activate(m: &Mlp, g: Option<f32>, u: f32) -> f32 {
    match (g, m.clamp) {
        (Some(g), Some(Clamp::Gate(l))) => gated(m.act, g.min(l), u.clamp(-l, l)),
        (Some(g), Some(Clamp::Activated(l))) => super::act(m.act, g).min(l) * u.clamp(-l, l),
        (Some(g), None) => gated(m.act, g, u),
        (None, _) => super::act(m.act, u),
    }
}

/// An MLP from its first product's sums (gate rows then up rows, or up alone) through `down`.
fn finish(m: &Mlp, first: &[f32], down: impl FnOnce(&[f32]) -> Result<Vec<f32>>) -> Result<Vec<f32>> {
    let h: Vec<f32> = if m.gated {
        let i = first.len() / 2;
        (0..i).map(|j| activate(m, Some(first[j]), first[i + j])).collect()
    } else {
        first.iter().map(|&u| activate(m, None, u)).collect()
    };
    down(&h)
}

/// The dense MLP of roles `{prefix}gate`, `{prefix}up` (or `{prefix}gate_up`), `{prefix}down`.
pub fn mlp(m: &Mlp, p: &Params, prefix: &str, x: &[f32]) -> Result<Vec<f32>> {
    let first = if p.has_prod(&format!("{prefix}gate_up")) {
        p.prod(&format!("{prefix}gate_up"), x)?
    } else if m.gated {
        let mut v = p.prod(&format!("{prefix}gate"), x)?;
        v.extend(p.prod(&format!("{prefix}up"), x)?);
        v
    } else {
        p.prod(&format!("{prefix}up"), x)?
    };
    finish(m, &first, |h| p.prod(&format!("{prefix}down"), h))
}

/// A per-layer input's gated branch (`kinds::PerLayer`) on one token's residual `x` and its layer's input `input`: proj(act(gate x) input).
pub fn per_layer(pl: &crate::runtime::kinds::PerLayer, p: &Params, x: &[f32], input: &[f32]) -> Result<Vec<f32>> {
    let g = p.prod("gate", x)?;
    if g.len() != input.len() {
        return crate::ffi::err(format!("a per-layer branch of {} gates for an input of {}", g.len(), input.len()));
    }
    let m: Vec<f32> = g.iter().zip(input).map(|(&g, &i)| super::act(pl.act, g) * i).collect();
    p.prod("proj", &m)
}

/// One token's chosen experts (in order) and their weights, from the router's logits.
pub fn route(r: &Router, logits: &[f32], choice_bias: Option<&[f32]>, hash: Option<&[f32]>) -> Result<(Vec<usize>, Vec<f32>)> {
    route_margin(r, logits, choice_bias, hash).map(|(ids, w, _)| (ids, w))
}

/// The gap between the last chosen and the first not chosen of `rank` (largest first), over its largest magnitude: how near a choice of
/// `k` is to another (infinite where nothing is left out).
pub fn margin(rank: &[f32], k: usize) -> f32 {
    let mut v: Vec<f32> = rank.iter().copied().filter(|x| x.is_finite()).collect();
    v.sort_by(|a, b| b.total_cmp(a));
    if k == 0 || v.len() <= k {
        return f32::INFINITY;
    }
    (v[k - 1] - v[k]) / v.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-30)
}

/// One token's routing: the experts taken (in order) and their weights; the router's own choice (the experts taken, unless a check gave others:
/// `route_taking`), its `margin` (a hash table's: infinite) and every expert's rank (the score the choice is made by; a hash table's: none).
#[derive(Clone, Debug)]
pub struct Routed {
    pub ids: Vec<usize>,
    pub w: Vec<f32>,
    pub own: Vec<usize>,
    pub gap: f32,
    pub rank: Vec<f32>,
}

/// How far apart two choices of one token are by the router's `rank`: the largest rank among the experts only `own` holds less the smallest
/// among the experts only `other` holds, over the rank's largest magnitude (`margin`'s scale); the same experts, 0.
pub fn margin_between(rank: &[f32], own: &[usize], other: &[usize]) -> f32 {
    let top = own.iter().filter(|e| !other.contains(e)).map(|&e| rank[e]).fold(f32::NEG_INFINITY, f32::max);
    let low = other.iter().filter(|e| !own.contains(e)).map(|&e| rank[e]).fold(f32::INFINITY, f32::min);
    if top == f32::NEG_INFINITY || low == f32::INFINITY {
        return 0.0;
    }
    (top - low) / rank.iter().filter(|x| x.is_finite()).fold(0f32, |m, x| m.max(x.abs())).max(1e-30)
}

/// `route`, and the choice's `margin` (a hash table's: infinite).
pub fn route_margin(r: &Router, logits: &[f32], choice_bias: Option<&[f32]>, hash: Option<&[f32]>) -> Result<(Vec<usize>, Vec<f32>, f32)> {
    route_taking(r, logits, choice_bias, hash, None).map(|c| (c.ids, c.w, c.gap))
}

/// `route` with the experts `take` (a check's: another engine's choice of this token) in the place of the router's own where it is not empty,
/// in the router's order (by rank, so the router's own experts taken are the router's own sum), weighted by the router's rule on this token's
/// scores; the margin and the ranks are the router's own choice's.
pub fn route_taking(r: &Router, logits: &[f32], choice_bias: Option<&[f32]>, hash: Option<&[f32]>, take: Option<&[usize]>) -> Result<Routed> {
    route_scaled(r, logits, choice_bias, hash, take, None)
}

/// `route_taking` with the learned scale an expert `expert_scale` where the router has one (`Router::expert_scale`: each chosen expert's weight times its
/// own, after the sum's division). `logits`: every router's of the experts, a router after another (`Router::routers`): each router's scores are made
/// alone (a softmax over its own logits) and an expert's score is their sum, added in the routers' order.
pub fn route_scaled(
    r: &Router,
    logits: &[f32],
    choice_bias: Option<&[f32]>,
    hash: Option<&[f32]>,
    take: Option<&[usize]>,
    expert_scale: Option<&[f32]>,
) -> Result<Routed> {
    let routers = r.routers.max(1);
    let n = logits.len() / routers;
    if n * routers != logits.len() || r.expert_scale != expert_scale.is_some() || expert_scale.is_some_and(|e| e.len() != n) {
        return err(format!(
            "a router of {} logits for {routers} routers, a scale an expert of {:?} values where the kind says {}",
            logits.len(),
            expert_scale.map(<[f32]>::len),
            r.expert_scale
        ));
    }
    let mut gap = f32::INFINITY;
    let mut rank = Vec::new();
    let mut s = vec![0f32; n];
    for one in logits.chunks(n.max(1)) {
        let mut t: Vec<f32> = one.iter().map(|&l| l * r.logit_scale).collect();
        match r.score {
            RouterScore::Softmax => softmax(&mut t),
            RouterScore::Sigmoid => t.iter_mut().for_each(|v| *v = sigmoid(*v)),
            RouterScore::SqrtSoftplus => t.iter_mut().for_each(|v| *v = softplus(*v).sqrt()),
        }
        s.iter_mut().zip(&t).for_each(|(a, b)| *a += b);
    }
    let mut ids: Vec<usize> = if let Some(h) = hash {
        h.iter().map(|&e| e as usize).collect()
    } else {
        rank = (0..n).map(|e| s[e] + choice_bias.map_or(0.0, |b| b[e])).collect();
        if let Some((groups, keep)) = r.groups {
            if n % groups != 0 || keep > groups {
                return err(format!("a router of {n} experts in {groups} groups, keeping {keep}"));
            }
            let per = n / groups;
            let gs: Vec<f32> = (0..groups)
                .map(|g| {
                    let mut v = rank[g * per..][..per].to_vec();
                    v.sort_by(|a, b| b.total_cmp(a));
                    v.iter().take(2).sum()
                })
                .collect();
            let mut go: Vec<usize> = (0..groups).collect();
            go.sort_by(|&a, &b| gs[b].total_cmp(&gs[a]).then(a.cmp(&b)));
            for &g in &go[keep..] {
                rank[g * per..][..per].iter_mut().for_each(|v| *v = f32::NEG_INFINITY);
            }
        }
        let mut ord: Vec<usize> = (0..n).collect();
        ord.sort_by(|&a, &b| rank[b].total_cmp(&rank[a]).then(a.cmp(&b)));
        ord.truncate(r.top_k);
        gap = margin(&rank, r.top_k);
        ord
    };
    if ids.len() != r.top_k || ids.iter().any(|&e| e >= n) {
        return err(format!("{} experts chosen of {n}, {:?}", ids.len(), ids));
    }
    let own = ids.clone();
    if let Some(t) = take.filter(|t| !t.is_empty()) {
        if t.len() != r.top_k || t.iter().any(|&e| e >= n) || (1..t.len()).any(|i| t[..i].contains(&t[i])) {
            return err(format!("a choice to take of {:?}: {} experts of {n}, each once", t, r.top_k));
        }
        ids = t.to_vec();
        if !rank.is_empty() {
            ids.sort_by(|&a, &b| rank[b].total_cmp(&rank[a]).then(a.cmp(&b)));
        }
    }
    let sum: f32 = ids.iter().map(|&e| s[e]).sum();
    let w = ids.iter().map(|&e| if r.renorm { s[e] / (sum + 1e-20) } else { s[e] } * r.scale * expert_scale.map_or(1.0, |x| x[e])).collect();
    Ok(Routed { ids, w, own, gap, rank })
}

/// One token through the MoE layer (`token`: its id, for hash routing).
pub fn moe(m: &Moe, p: &Params, x: &[f32], token: u32) -> Result<Vec<f32>> {
    Ok(moe_routed(m, p, x, token)?.0)
}

/// `moe`, the experts the router chose (hummingbird's corpus pass counts them: src/hb_usage.rs) and the choice's `margin`.
pub fn moe_routed(m: &Moe, p: &Params, x: &[f32], token: u32) -> Result<(Vec<f32>, Vec<usize>, f32)> {
    moe_taking(m, p, x, token, None).map(|(y, c)| (y, c.ids, c.gap))
}

/// The block's router alone on the input `x` (the block's normed input; `token`: its id, for hash routing), with the experts `take` in the
/// place of its own choice where given (`route_taking`): the token's routing. A check holds an engine's router to it on the engine's own
/// input row (`logits::held_on_choices`).
pub fn route_of(m: &Moe, p: &Params, x: &[f32], token: u32, take: Option<&[usize]>) -> Result<Routed> {
    let logits = p.prod("router", x)?;
    if logits.len() != m.router.routers.max(1) * m.experts {
        return err(format!("a router of {} logits for {} experts by {} routers", logits.len(), m.experts, m.router.routers));
    }
    let hash = if m.router.hash {
        let t = p.vec("hash")?;
        let k = m.router.top_k;
        let row = token as usize * k;
        if row + k > t.len() {
            return err(format!("token {token} past the hash table"));
        }
        Some(&t[row..row + k])
    } else {
        None
    };
    let bias = if m.router.bias { Some(p.vec("choice_bias")?) } else { None };
    let scale = if m.router.expert_scale { Some(p.vec("expert_scale")?) } else { None };
    route_scaled(&m.router, &logits, bias, hash, take, scale)
}

/// `moe` with the token's experts `take` in the place of the router's own choice where given (`route_taking`), and the token's routing.
pub fn moe_taking(m: &Moe, p: &Params, x: &[f32], token: u32, take: Option<&[usize]>) -> Result<(Vec<f32>, Routed)> {
    let c = route_of(m, p, x, token, take)?;
    // the experts' input: x, or its latent projection
    let z = match &m.latent {
        Some(l) => {
            let z = p.prod("latent_down", x)?;
            match l.norm_in {
                Some(n) => norm(&n, &z, p.opt("latent_norm_in"), None),
                None => z,
            }
        }
        None => x.to_vec(),
    };
    let (gu, down) = (p.w(if m.expert.gated { "experts_gate_up" } else { "experts_up" })?, p.w("experts_down")?);
    let rows = gu.len() / (m.experts * z.len());
    let out = down.len() / (m.experts * if m.expert.gated { rows / 2 } else { rows });
    let inter = if m.expert.gated { rows / 2 } else { rows };
    if rows * m.experts * z.len() != gu.len() || out * inter * m.experts != down.len() {
        return err(format!("experts of {} and {} values for {} experts on {} inputs", gu.len(), down.len(), m.experts, z.len()));
    }
    // (where the reference simulates the release's quantizations and its experts are stored quantized: their input rounded as a quantized product's, and each pair's
    // weight in its activation, which the second product's input rounds, as the release weighs it)
    let sim = p.sim > 0 && m.latent.is_none() && p.quantized.contains(if m.expert.gated { "experts_gate_up" } else { "experts_up" });
    let z = if sim { p.fp8(&z) } else { z };
    let mut acc = vec![0f64; out];
    for (&e, &we) in c.ids.iter().zip(&c.w) {
        let first = matvec_w(gu.part(e * rows * z.len(), rows * z.len()), z.len(), &z);
        let dn = down.part(e * out * inter, out * inter);
        let mut y = if sim {
            finish(&m.expert, &first, |h| Ok(matvec_w(dn, inter, &p.fp8(&h.iter().map(|&v| we * v).collect::<Vec<_>>()))))?
        } else {
            finish(&m.expert, &first, |h| Ok(matvec_w(dn, inter, h)))?
        };
        if let Some(n) = m.latent.as_ref().and_then(|l| l.expert_norm) {
            y = norm(&n, &y, p.opt("expert_norm"), None);
        }
        let we = if sim { 1.0 } else { we };
        acc.iter_mut().zip(&y).for_each(|(a, &v)| *a += we as f64 * v as f64);
    }
    let mut y: Vec<f32> = acc.into_iter().map(|v| v as f32).collect();
    if let Some(l) = &m.latent {
        if let Some(n) = l.norm_out {
            y = norm(&n, &y, p.opt("latent_norm_out"), None);
        }
        y = p.prod("latent_up", &y)?;
    }
    if let Some(s) = &m.shared {
        let mut so = mlp(s, p, "shared_", x)?;
        if m.shared_gate {
            let g = sigmoid(p.prod("shared_router", x)?[0]);
            so.iter_mut().for_each(|v| *v *= g);
        }
        y.iter_mut().zip(&so).for_each(|(a, &b)| *a += b);
    }
    Ok((y, c))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::kinds::Act;
    use crate::runtime::reference::Libm;

    fn r(score: RouterScore, top_k: usize) -> Router {
        Router { score, logit_scale: 1.0, bias: false, groups: None, top_k, renorm: true, scale: 1.0, hash: false, routers: 1, expert_scale: false }
    }

    #[test]
    fn routing() {
        // softmax: the largest logits, weights renormed
        let (ids, w) = route(&r(RouterScore::Softmax, 2), &[0.0, 2.0, 1.0, 2.0], None, None).unwrap();
        assert_eq!(ids, [1, 3]);
        assert!((w[0] - 0.5).abs() < 1e-6 && (w[1] - 0.5).abs() < 1e-6);
        // the bias picks, the scores weigh
        let rb = Router { score: RouterScore::Sigmoid, bias: true, renorm: false, ..r(RouterScore::Sigmoid, 1) };
        let (ids, w) = route(&rb, &[1.0, 0.0], Some(&[0.0, 2.0]), None).unwrap();
        assert_eq!((ids, w), (vec![1], vec![0.5]));
        // groups: [0.9 0.8 | 0.95 0.1]: group 0 sums 1.7 against 1.05, so with one group kept expert 2 is not chosen
        let rg = Router { groups: Some((2, 1)), ..r(RouterScore::Sigmoid, 2) };
        let lg = |p: f32| (p / (1.0 - p)).lln();
        let (ids, _) = route(&rg, &[lg(0.9), lg(0.8), lg(0.95), lg(0.1)], None, None).unwrap();
        assert_eq!(ids, [0, 1]);
        // hash: the table's experts, weighed by their scores
        let rh = Router { hash: true, ..r(RouterScore::SqrtSoftplus, 2) };
        let (ids, w) = route(&rh, &[0.0, 1.0, 5.0], None, Some(&[0.0, 1.0])).unwrap();
        assert_eq!(ids, [0, 1]);
        assert!((w[0] / w[1] - (softplus(0.0) / softplus(1.0)).sqrt()).abs() < 1e-6);
    }

    /// A choice's margin: the gap at the cut over the largest magnitude; nothing left out, infinite; a tie, 0. The router's is its rank's
    /// (the scores and the choice bias), a hash table's infinite.
    /// Two routers whose sigmoid scores add, and a learned scale an expert (ERNIE's): the score of an expert is sigmoid(l) + sigmoid(l'), the choice by
    /// the score plus the bias, the weights the chosen scores over their sum, each times its expert's own scale. By hand; and one router of the summed
    /// form with scales of one is the router it was.
    #[test]
    fn routers_whose_scores_add_and_a_scale_an_expert() {
        let two = Router { bias: true, routers: 2, expert_scale: true, ..r(RouterScore::Sigmoid, 2) };
        let (a, b) = ([0.5f32, -1.0, 2.0, 0.0], [1.5f32, 3.0, -2.0, 0.25]);
        let logits: Vec<f32> = a.iter().chain(&b).copied().collect();
        let (bias, scale) = ([0.0f32, 0.0, 0.0, 1.0], [2.0f32, 0.5, 3.0, 0.25]);
        let s: Vec<f32> = (0..4).map(|e| sigmoid(a[e]) + sigmoid(b[e])).collect();
        // (ranks: s + bias; expert 3's bias lifts it to the first place, expert 0 follows: s_0 = 1.44, s_1 = 1.22, s_2 = 1.00, s_3 + 1 = 2.06)
        let got = route_scaled(&two, &logits, Some(&bias), None, None, Some(&scale)).unwrap();
        assert_eq!(got.ids, [3, 0]);
        let sum = s[3] + s[0];
        assert_eq!(got.w, [s[3] / (sum + 1e-20) * scale[3], s[0] / (sum + 1e-20) * scale[0]]);
        assert_eq!(got.rank, (0..4).map(|e| s[e] + bias[e]).collect::<Vec<_>>());
        // (the scores add: neither router's alone makes this choice's weights, and the order of the routers is no matter to the sum of two)
        let swapped: Vec<f32> = b.iter().chain(&a).copied().collect();
        assert_eq!(route_scaled(&two, &swapped, Some(&bias), None, None, Some(&scale)).unwrap().w, got.w);
        let one = Router { routers: 1, ..two.clone() };
        assert_ne!(route_scaled(&one, &a, Some(&bias), None, None, Some(&scale)).unwrap().w, got.w);
        // (not renormed: the summed scores themselves, times the scales)
        let raw = route_scaled(&Router { renorm: false, ..two.clone() }, &logits, Some(&bias), None, None, Some(&scale)).unwrap();
        assert_eq!(raw.w, [s[3] * scale[3], s[0] * scale[0]]);
        // (scores of nothing: no weight, not a division by nothing)
        let none = route_scaled(&two, &[-200.0; 8], Some(&bias), None, None, Some(&scale)).unwrap();
        assert_eq!(none.w, [0.0, 0.0]);
        // (one router, scales of one: the router without the fields)
        let plain = Router { bias: true, ..r(RouterScore::Sigmoid, 2) };
        let ones = Router { expert_scale: true, ..plain.clone() };
        let want = route_taking(&plain, &a, Some(&bias), None, None).unwrap();
        let same = route_scaled(&ones, &a, Some(&bias), None, None, Some(&[1.0; 4])).unwrap();
        assert_eq!((same.ids, same.w), (want.ids, want.w));
        // (the kind and what it is given agree: logits of every router, a scale where the kind has one and only there)
        assert!(route_scaled(&two, &logits[..7], Some(&bias), None, None, Some(&scale)).is_err());
        assert!(route_scaled(&two, &logits, Some(&bias), None, None, None).is_err());
        assert!(route_scaled(&plain, &a, Some(&bias), None, None, Some(&scale)).is_err());
        // (said in a description's text only where set)
        assert!(!format!("{plain:?}").contains("routers") && format!("{two:?}").ends_with("hash: false, routers: 2, expert_scale: true }"));
    }

    /// The experts' block reads the two routers as one stacked product and the scale as its role: the block's output is the hand-made sum of the chosen
    /// experts' outputs by the routed weights.
    #[test]
    fn a_block_of_experts_with_two_routers() {
        use crate::runtime::reference::{matvec, silu, testing::fill};
        let (d, inter, ne) = (6, 4, 4);
        let m = Moe {
            router: Router { bias: true, routers: 2, expert_scale: true, ..r(RouterScore::Sigmoid, 2) },
            experts: ne,
            expert: Mlp { act: Act::Silu, gated: true, bias: false, clamp: None },
            shared: None,
            shared_gate: false,
            latent: None,
        };
        let mut p = Params::new();
        p.put("router", &[2 * ne, d], fill("two.router", 2 * ne * d, -1.0, 1.0));
        p.put("choice_bias", &[ne], fill("two.bias", ne, -0.1, 0.1));
        p.put("expert_scale", &[ne], fill("two.scale", ne, 0.5, 2.0));
        p.put("experts_gate_up", &[ne, 2 * inter, d], fill("two.gu", ne * 2 * inter * d, -0.5, 0.5));
        p.put("experts_down", &[ne, d, inter], fill("two.down", ne * d * inter, -0.5, 0.5));
        let x = fill("two.x", d, -1.0, 1.0);
        let (y, c) = moe_taking(&m, &p, &x, 0, None).unwrap();
        let logits = p.prod("router", &x).unwrap();
        let want = route_scaled(&m.router, &logits, p.opt("choice_bias"), None, None, p.opt("expert_scale")).unwrap();
        assert_eq!((&c.ids, &c.w), (&want.ids, &want.w));
        let (gu, down) = (p.vec("experts_gate_up").unwrap(), p.vec("experts_down").unwrap());
        let mut acc = vec![0f64; d];
        for (&e, &w) in c.ids.iter().zip(&c.w) {
            let first = matvec(&gu[e * 2 * inter * d..][..2 * inter * d], d, &x);
            let h: Vec<f32> = (0..inter).map(|i| silu(first[i]) * first[inter + i]).collect();
            let o = matvec(&down[e * d * inter..][..d * inter], inter, &h);
            acc.iter_mut().zip(&o).for_each(|(a, &v)| *a += w as f64 * v as f64);
        }
        assert_eq!(y, acc.into_iter().map(|v| v as f32).collect::<Vec<_>>());
        // (a block without the scale's role, or with one router's rows, is refused)
        let mut q = p.clone();
        q.t.remove("expert_scale");
        assert!(moe_taking(&m, &q, &x, 0, None).is_err());
        q = p.clone();
        q.put("router", &[ne, d], fill("two.router", ne * d, -1.0, 1.0));
        assert!(moe_taking(&m, &q, &x, 0, None).is_err());
    }

    #[test]
    fn margins() {
        assert_eq!(margin(&[1.0, 4.0, 2.0], 2), 0.25);
        assert_eq!(margin(&[1.0, 4.0], 2), f32::INFINITY);
        assert_eq!(margin(&[3.0, 3.0, 1.0], 1), 0.0);
        let (_, _, g) = route_margin(&r(RouterScore::Softmax, 1), &[0.0, 1.0], None, None).unwrap();
        let p = sigmoid(1.0);
        assert!((g - (2.0 * p - 1.0) / p).abs() < 1e-6);
        let rh = Router { hash: true, ..r(RouterScore::Softmax, 1) };
        assert_eq!(route_margin(&rh, &[0.0, 1.0], None, Some(&[0.0])).unwrap().2, f32::INFINITY);
    }

    /// A choice taken in the router's place: those experts in that order, weighted by the router's rule on the token's scores; the margin
    /// and the ranks stay the router's own; a choice of another size, or naming an expert twice, is refused.
    #[test]
    fn a_choice_taken() {
        let rt = r(RouterScore::Softmax, 2);
        let l = [0.0f32, 1.0, 0.9];
        let own = route_taking(&rt, &l, None, None, None).unwrap();
        assert_eq!(own.ids, vec![1, 2]);
        assert_eq!((own.ids.clone(), own.w.clone(), own.gap), route_margin(&rt, &l, None, None).unwrap());
        assert_eq!(own.own, own.ids);
        let got = route_taking(&rt, &l, None, None, Some(&[0, 1])).unwrap();
        assert_eq!((got.ids.clone(), got.own.clone(), got.gap, &got.rank), (vec![1, 0], vec![1, 2], own.gap, &own.rank));
        // (the router's own experts in another order, and no experts: the router's own routing)
        for same in [&[2usize, 1][..], &[]] {
            let c = route_taking(&rt, &l, None, None, Some(same)).unwrap();
            assert_eq!((c.ids, c.w), (own.ids.clone(), own.w.clone()));
        }
        let s = &own.rank;
        assert!((got.w[0] - s[1] / (s[1] + s[0])).abs() < 1e-6 && (got.w[0] + got.w[1] - 1.0).abs() < 1e-6);
        // (the two choices differ by expert 2 for expert 0: their ranks' gap over the largest, which is the choice's margin here)
        assert_eq!(margin_between(s, &own.ids, &got.ids), (s[2] - s[0]) / s[1]);
        assert_eq!(margin_between(s, &own.ids, &got.ids), own.gap);
        assert_eq!(margin_between(s, &own.ids, &[2, 1]), 0.0);
        assert!(route_taking(&rt, &l, None, None, Some(&[1])).is_err());
        assert!(route_taking(&rt, &l, None, None, Some(&[1, 1])).is_err());
        assert!(route_taking(&rt, &l, None, None, Some(&[1, 3])).is_err());
    }

    #[test]
    fn activations() {
        let m = Mlp { act: Act::Silu, gated: true, bias: false, clamp: Some(Clamp::Gate(1.0)) };
        assert_eq!(activate(&m, Some(3.0), -5.0), -super::super::silu(1.0));
        // (the limit on the activated gate, Ling's: silu(3) is over 1 and held to it, where the gate held first gives silu(1); under the limit the two agree,
        // and the up is held both ways in both)
        let after = Mlp { clamp: Some(Clamp::Activated(1.0)), ..m };
        assert_eq!(activate(&after, Some(3.0), -5.0), -1.0);
        assert_eq!(activate(&after, Some(3.0), 0.25), 0.25);
        assert_eq!(activate(&after, Some(0.5), 0.25), activate(&m, Some(0.5), 0.25));
        assert_eq!(activate(&after, Some(-2.0), 7.0), super::super::silu(-2.0));
        assert_eq!((format!("{:?}", m.clamp), format!("{:?}", after.clamp)), ("Some(1.0)".to_string(), "Some(Activated(1.0))".to_string()));
        let k = Mlp { act: Act::Situ { beta: 4.0, linear: 25.0 }, ..m };
        // the kimi_ref formula
        let want = 4.0 * (0.5f32 / 4.0).ltanh() * sigmoid(0.5) * 25.0 * (2.0f32 / 25.0).ltanh();
        assert!((activate(&Mlp { clamp: None, ..k }, Some(0.5), 2.0) - want).abs() < 1e-6);
        let r2 = Mlp { act: Act::Relu2, gated: false, ..m };
        assert_eq!((activate(&r2, None, -1.0), activate(&r2, None, 3.0)), (0.0, 9.0));
        // (the gpt-oss SwiGLU, MiniMax M3's at alpha 1.702 and a limit of 7: min(g, 7) sigmoid(1.702 min(g, 7)) (clamp(u, -7, 7) + 1); the gate is held above
        // only, the shifted up reaches -6 and 8, and an up of -1 gives nothing)
        let oai = Mlp { act: Act::SwigluOai { alpha: 1.702 }, clamp: Some(Clamp::Gate(7.0)), ..m };
        let half = |g: f32| g / (1.0 + (-1.702 * g).lexp());
        for (g, u, want) in [
            (0.5f32, 2.0f32, half(0.5) * 3.0),
            (9.0, 2.0, half(7.0) * 3.0),
            (-9.0, 2.0, half(-9.0) * 3.0),
            (0.5, 20.0, half(0.5) * 8.0),
            (0.5, -20.0, half(0.5) * -6.0),
        ] {
            let got = activate(&oai, Some(g), u);
            assert!((got - want).abs() <= 1e-6 * want.abs().max(1.0), "gate {g}, up {u}: {got}, by hand {want}");
        }
        assert_eq!(activate(&oai, Some(3.0), -1.0), 0.0);
        // (at alpha 1 its gate is SiLU's, and the shift is what is left: silu(g) (u + 1))
        let one = Mlp { act: Act::SwigluOai { alpha: 1.0 }, clamp: None, ..m };
        assert_eq!(activate(&one, Some(0.75), 2.0), super::super::silu(0.75) * 3.0);
        assert_eq!((oai.not_run(), m.not_run()), (Some(Act::SWIGLU_OAI_NOT_RUN), None));
        assert_eq!(oai.act.name(), "swigluoai");
    }
}
