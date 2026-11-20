//! MHA, GQA and MQA attention (`kinds::Attn`): full or sliding window, optional q/k/v norms, rope (full, partial, proportional or none),
//! learned sinks, a score soft-cap and an output gate. Query head h reads KV head h / (heads / kv_heads). One token at a time against the
//! sequence's cache; a prompt is its tokens in order.

use super::{act, dot, norm::norm, rope, softmax, Params};
use crate::ffi::{err, Result};
use crate::runtime::kinds::{Attn, GateFrom};

/// A sequence's keys and values in one layer, each after its norm and rope, with their positions.
#[derive(Default)]
pub struct KvCache {
    pub pos: Vec<u32>,
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

/// One token's attention output (before nothing: `o` is applied) for the normed input `h` at `pos`; its key and value join the cache.
pub fn step(a: &Attn, p: &Params, c: &mut KvCache, h: &[f32], pos: u32) -> Result<Vec<f32>> {
    let (nh, nk, hd, vd) = (a.heads, a.kv_heads, a.head_dim, a.v_head_dim);
    if nh % nk != 0 || (a.k_eq_v && vd != hd) {
        return err(format!("attention: {nh} heads over {nk} KV heads, head dims {hd} and {vd}"));
    }
    let raw_q = p.prod("q", h)?;
    let in_q = a.gate.is_some_and(|g| g.from == GateFrom::InQuery);
    let (mut q, gate) = if in_q {
        let (mut q, mut g) = (Vec::with_capacity(nh * hd), Vec::with_capacity(nh * hd));
        for ch in raw_q.chunks(2 * hd) {
            q.extend_from_slice(&ch[..hd]);
            g.extend_from_slice(&ch[hd..]);
        }
        (q, Some(g))
    } else {
        let g = match a.gate {
            Some(_) => Some(p.prod("g", h)?),
            None => None,
        };
        (raw_q, g)
    };
    let raw_k = p.prod("k", h)?;
    let mut v = if a.k_eq_v { raw_k.clone() } else { p.prod("v", h)? };
    let mut k = raw_k;
    if q.len() != nh * hd || k.len() != nk * hd || v.len() != nk * vd {
        return err(format!("attention: products of {}, {} and {} values for {nh} x {hd}, {nk} x {hd}, {nk} x {vd}", q.len(), k.len(), v.len()));
    }
    let rot = a.rope.as_ref().map(|r| (r, rope::inv_freq_at(r, pos + 1)));
    let prep = |x: &mut [f32], n: &Option<crate::runtime::kinds::Norm>, role: &str, d: usize, rotate: bool| {
        for head in x.chunks_mut(d) {
            if let Some(n) = n {
                let y = norm(n, head, p.opt(role), None);
                head.copy_from_slice(&y);
            }
            if let (true, Some((r, (inv, af)))) = (rotate, &rot) {
                rope::apply(r, inv, *af, head, pos);
            }
        }
    };
    prep(&mut q, &a.q_norm, "q_norm", hd, true);
    prep(&mut k, &a.k_norm, "k_norm", hd, true);
    prep(&mut v, &a.v_norm, "v_norm", vd, false);
    c.pos.push(pos);
    c.k.push(k);
    c.v.push(v);
    let sinks = if a.sinks { Some(p.vec("sinks")?) } else { None };
    let mut out = Vec::with_capacity(nh * vd);
    for head in 0..nh {
        let kv = head / (nh / nk);
        let qh = &q[head * hd..][..hd];
        let seen: Vec<usize> = (0..c.pos.len()).filter(|&j| c.pos[j] <= pos && a.window.is_none_or(|w| pos - c.pos[j] < w as u32)).collect();
        let mut s: Vec<f32> = seen
            .iter()
            .map(|&j| {
                let x = dot(qh, &c.k[j][kv * hd..][..hd]) * a.scale;
                match a.softcap {
                    Some(cap) => cap * (x / cap).tanh(),
                    None => x,
                }
            })
            .collect();
        if let Some(sk) = sinks {
            s.push(sk[head]);
        }
        softmax(&mut s);
        let mut o = vec![0f64; vd];
        for (&j, &pj) in seen.iter().zip(&s) {
            for (ov, &vv) in o.iter_mut().zip(&c.v[j][kv * vd..][..vd]) {
                *ov += pj as f64 * vv as f64;
            }
        }
        let mut o: Vec<f32> = o.into_iter().map(|v| v as f32).collect();
        if let (Some(g), Some(gv)) = (a.gate, &gate) {
            if let Some(n) = g.norm {
                o = norm(&n, &o, None, None);
            }
            o.iter_mut().zip(&gv[head * vd..][..vd]).for_each(|(x, &z)| *x *= act(g.act, z));
        }
        out.extend(o);
    }
    p.prod("o", &out)
}
