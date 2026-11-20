//! Multi-head latent attention (`kinds::Mla`), in the expanded form: each cached latent through `kv_b` gives a head's key (nope part) and
//! value; the rope part of the key is one per token, shared by the heads. The absorbed form a backend runs is the same function.

use super::{act, dot, norm::norm, rope, softmax, Params};
use crate::ffi::{err, Result};
use crate::runtime::kinds::Mla;

/// A sequence's latents in one layer: per token the normed latent (kv_lora values) then the shared key part (rope_dim, rotated).
#[derive(Default)]
pub struct LatentCache {
    pub pos: Vec<u32>,
    pub lat: Vec<Vec<f32>>,
}

pub fn step(m: &Mla, p: &Params, c: &mut LatentCache, h: &[f32], pos: u32) -> Result<Vec<f32>> {
    let (nh, kr, nope, rd, vd) = (m.heads, m.kv_lora, m.nope, m.rope_dim, m.v_dim);
    let qw = nope + rd;
    let q = match m.q_lora {
        Some(_) => {
            let qa = p.prod("q_a", h)?;
            p.prod("q_b", &norm(&m.q_norm, &qa, p.opt("q_a_norm"), None))?
        }
        None => p.prod("q", h)?,
    };
    let ckr = p.prod("kv_a", h)?;
    if q.len() != nh * qw || ckr.len() != kr + rd {
        return err(format!("MLA: {} query values for {nh} x {qw}, {} latent values for {kr} + {rd}", q.len(), ckr.len()));
    }
    let rot = m.rope.as_ref().map(|r| (r, rope::inv_freq_at(r, pos + 1)));
    let mut lat = norm(&m.kv_norm, &ckr[..kr], p.opt("kv_a_norm"), None);
    let mut kr_part = ckr[kr..].to_vec();
    if let Some((r, (inv, af))) = &rot {
        rope::apply(r, inv, *af, &mut kr_part, pos);
    }
    lat.extend(kr_part);
    c.pos.push(pos);
    c.lat.push(lat);
    let kv_b = p.vec("kv_b")?;
    if kv_b.len() != nh * (nope + vd) * kr {
        return err(format!("MLA: kv_b of {} values for {nh} x ({nope} + {vd}) x {kr}", kv_b.len()));
    }
    let gate = match m.gate {
        Some(_) => Some(p.prod("g", h)?),
        None => None,
    };
    let seen: Vec<usize> = (0..c.pos.len()).filter(|&j| c.pos[j] <= pos).collect();
    // an indexer keeps every token while the context fits its choice; past that the reference has no indexer yet
    if let Some(ix) = &m.indexer {
        if seen.len() > ix.top_k * ix.block.max(1) {
            return err(format!("MLA: {} tokens past the indexer's {} (the reference does not select yet)", seen.len(), ix.top_k));
        }
    }
    let mut out = Vec::with_capacity(nh * vd);
    for head in 0..nh {
        let rows = |r: usize| &kv_b[(head * (nope + vd) + r) * kr..][..kr];
        let mut qh = q[head * qw..][..qw].to_vec();
        if let Some((r, (inv, af))) = &rot {
            rope::apply(r, inv, *af, &mut qh[nope..], pos);
        }
        let mut s: Vec<f32> = seen
            .iter()
            .map(|&j| {
                let l = &c.lat[j];
                let k_nope: Vec<f32> = (0..nope).map(|d| dot(rows(d), &l[..kr])).collect();
                (dot(&qh[..nope], &k_nope) + dot(&qh[nope..], &l[kr..])) * m.scale
            })
            .collect();
        softmax(&mut s);
        let mut o = vec![0f64; vd];
        for (&j, &pj) in seen.iter().zip(&s) {
            for (d, ov) in o.iter_mut().enumerate() {
                *ov += pj as f64 * dot(rows(nope + d), &c.lat[j][..kr]) as f64;
            }
        }
        let mut o: Vec<f32> = o.into_iter().map(|v| v as f32).collect();
        if let (Some(g), Some(gv)) = (m.gate, &gate) {
            if let Some(n) = g.norm {
                o = norm(&n, &o, None, None);
            }
            o.iter_mut().zip(&gv[head * vd..][..vd]).for_each(|(x, &z)| *x *= act(g.act, z));
        }
        out.extend(o);
    }
    p.prod("o", &out)
}
