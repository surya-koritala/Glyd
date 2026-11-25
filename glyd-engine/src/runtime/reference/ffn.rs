//! Feed-forward kinds: the dense MLP (`kinds::Mlp`) and the mixture of experts (`kinds::Moe`) with its router variants, shared experts
//! and latent projection. The routed sum is over the chosen experts in the router's order; equal scores pick the smaller expert.

use super::{gated, matvec, norm::norm, sigmoid, softmax, softplus, Params};
use crate::ffi::{err, Result};
use crate::runtime::kinds::{Mlp, Moe, Router, RouterScore};

/// act(gate) * up, or act(up), from a gated MLP's two sums, with the SwiGLU limits.
fn activate(m: &Mlp, g: Option<f32>, u: f32) -> f32 {
    match (g, m.clamp) {
        (Some(g), Some(l)) => gated(m.act, g.min(l), u.clamp(-l, l)),
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
    let n = logits.len();
    let mut gap = f32::INFINITY;
    let mut rank = Vec::new();
    let mut s: Vec<f32> = logits.iter().map(|&l| l * r.logit_scale).collect();
    match r.score {
        RouterScore::Softmax => softmax(&mut s),
        RouterScore::Sigmoid => s.iter_mut().for_each(|v| *v = sigmoid(*v)),
        RouterScore::SqrtSoftplus => s.iter_mut().for_each(|v| *v = softplus(*v).sqrt()),
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
    let w = ids.iter().map(|&e| if r.renorm { s[e] / (sum + 1e-20) } else { s[e] } * r.scale).collect();
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

/// `moe` with the token's experts `take` in the place of the router's own choice where given (`route_taking`), and the token's routing.
pub fn moe_taking(m: &Moe, p: &Params, x: &[f32], token: u32, take: Option<&[usize]>) -> Result<(Vec<f32>, Routed)> {
    let logits = p.prod("router", x)?;
    if logits.len() != m.experts {
        return err(format!("a router of {} logits for {} experts", logits.len(), m.experts));
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
    let c = route_taking(&m.router, &logits, bias, hash, take)?;
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
    let (gu, down) = (p.vec(if m.expert.gated { "experts_gate_up" } else { "experts_up" })?, p.vec("experts_down")?);
    let rows = gu.len() / (m.experts * z.len());
    let out = down.len() / (m.experts * if m.expert.gated { rows / 2 } else { rows });
    let inter = if m.expert.gated { rows / 2 } else { rows };
    if rows * m.experts * z.len() != gu.len() || out * inter * m.experts != down.len() {
        return err(format!("experts of {} and {} values for {} experts on {} inputs", gu.len(), down.len(), m.experts, z.len()));
    }
    let mut acc = vec![0f64; out];
    for (&e, &we) in c.ids.iter().zip(&c.w) {
        let first = matvec(&gu[e * rows * z.len()..][..rows * z.len()], z.len(), &z);
        let mut y = finish(&m.expert, &first, |h| Ok(matvec(&down[e * out * inter..][..out * inter], inter, h)))?;
        if let Some(n) = m.latent.as_ref().and_then(|l| l.expert_norm) {
            y = norm(&n, &y, p.opt("expert_norm"), None);
        }
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

    fn r(score: RouterScore, top_k: usize) -> Router {
        Router { score, logit_scale: 1.0, bias: false, groups: None, top_k, renorm: true, scale: 1.0, hash: false }
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
        let lg = |p: f32| (p / (1.0 - p)).ln();
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
        let m = Mlp { act: Act::Silu, gated: true, bias: false, clamp: Some(1.0) };
        assert_eq!(activate(&m, Some(3.0), -5.0), -super::super::silu(1.0));
        let k = Mlp { act: Act::Situ { beta: 4.0, linear: 25.0 }, ..m };
        // the kimi_ref formula
        let want = 4.0 * (0.5f32 / 4.0).tanh() * sigmoid(0.5) * 25.0 * (2.0f32 / 25.0).tanh();
        assert!((activate(&Mlp { clamp: None, ..k }, Some(0.5), 2.0) - want).abs() < 1e-6);
        let r2 = Mlp { act: Act::Relu2, gated: false, ..m };
        assert_eq!((activate(&r2, None, -1.0), activate(&r2, None, 3.0)), (0.0, 9.0));
    }
}
