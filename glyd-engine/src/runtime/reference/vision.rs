//! The vision tower kind (`kinds::VisionDesc`, the Qwen-VL vision transformer) on the CPU: transformers' `Qwen4ExpVisionModel` on an image's patches, written
//! for clarity. Its rounding points are the release's (transformers' bf16 modeling: each tensor it makes is bf16), `Rounding::Once`: the patch product
//! (cuDNN's convolution, then its bias added), the position rows (their four taps summed in f32) and their sum; each LayerNorm's output; each product with its
//! bias (cuBLASLt's: the f64 sum, its bias in f32, one rounding); the rotary's output (in f32 from the bf16 q and k, rounded once); the attention's
//! probabilities for the values (FlashAttention's, unnormalized, against the row's largest score; the sum that divides them f32) and its output; each
//! activation; each residual sum. `Rounding::F32`: none of them (the tower in f32 on the same weights: its noise floor).
//!
//! The image's patches are in its 2 x 2 blocks' order (`vision::Image`), each patch's pixels as the patch product reads them (channel, frame, row, column);
//! the output is a row of `out` for each block of `merge` x `merge` patches, in the same order.

use super::{act, bf16, dot, softmax};
use crate::ffi::{err, Error, Result};
use crate::runtime::describe::VisionRoles;
use crate::runtime::kinds::{Act, Rounding, VisionDesc};

/// A block's weights (f32 values of the checkpoint's bf16).
struct Block {
    n1: (Vec<f32>, Vec<f32>),
    n2: (Vec<f32>, Vec<f32>),
    qkv: (Vec<f32>, Vec<f32>),
    o: (Vec<f32>, Vec<f32>),
    up: (Vec<f32>, Vec<f32>),
    down: (Vec<f32>, Vec<f32>),
}

/// The tower's weights by its description's roles.
pub struct RefTower {
    pub v: VisionDesc,
    patch: (Vec<f32>, Vec<f32>),
    pos: Vec<f32>,
    blocks: Vec<Block>,
    merge_norm: (Vec<f32>, Vec<f32>),
    merge_up: (Vec<f32>, Vec<f32>),
    merge_down: (Vec<f32>, Vec<f32>),
}

/// What `RefTower::run` gives: the rows of the image's tokens (f32 values, bf16 where the rounding is), and the patches' rows after the embedding and after each
/// block (a check holds each block of an engine to the reference on the engine's own input: `RefTower::block`).
pub struct TowerRows {
    pub out: Vec<f32>,
    pub embedded: Vec<f32>,
    pub after: Vec<Vec<f32>>,
}

impl RefTower {
    /// The tower `v` whose tensors `roles` names, read by `src` (a name to its shape and f32 values); each at the shape the kind says.
    pub fn load(v: &VisionDesc, roles: &VisionRoles, src: &dyn Fn(&str) -> Option<(Vec<usize>, Vec<f32>)>) -> Result<RefTower> {
        let (h, m) = (v.hidden, v.hidden * v.merge * v.merge);
        let get = |block: Option<usize>, r: &str, shape: &[usize]| -> Result<Vec<f32>> {
            let n = roles.tensor(block, r)?;
            let (s, x) = src(n).ok_or_else(|| Error(format!("the vision tower's {n}: not in the checkpoint")))?;
            if s.iter().product::<usize>() != shape.iter().product::<usize>() {
                return err(format!("the vision tower's {n}: {s:?}, where {shape:?} is wanted"));
            }
            Ok(x)
        };
        let pair = |block: Option<usize>, r: &str, shape: &[usize]| -> Result<(Vec<f32>, Vec<f32>)> {
            Ok((get(block, r, shape)?, get(block, &format!("{r}_bias"), &shape[..1])?))
        };
        let blocks = (0..v.blocks)
            .map(|l| {
                let b = Some(l);
                Ok(Block {
                    n1: pair(b, "norm1", &[h])?,
                    n2: pair(b, "norm2", &[h])?,
                    qkv: pair(b, "qkv", &[3 * h, h])?,
                    o: pair(b, "o", &[h, h])?,
                    up: pair(b, "up", &[v.inter, h])?,
                    down: pair(b, "down", &[h, v.inter])?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(RefTower {
            v: v.clone(),
            patch: pair(None, "patch", &[h, v.patch_dim()])?,
            pos: get(None, "pos", &[v.table * v.table, h])?,
            blocks,
            merge_norm: pair(None, "merge_norm", &[h])?,
            merge_up: pair(None, "merge_up", &[m, m])?,
            merge_down: pair(None, "merge_down", &[v.out, m])?,
        })
    }

    /// The image of `gh` x `gw` patches (`patches`: bf16 values [gh gw, patch_dim] in its blocks' order) through the tower, rounded at `r`'s points.
    pub fn run(&self, patches: &[f32], gh: usize, gw: usize, r: Rounding) -> Result<TowerRows> {
        let v = &self.v;
        let (n, h) = (gh * gw, v.hidden);
        if gh % v.merge != 0 || gw % v.merge != 0 || patches.len() != n * v.patch_dim() || n == 0 {
            return err(format!(
                "an image of {gh} x {gw} patches and {} values, where the tower takes whole blocks of {} x {}",
                patches.len(),
                v.merge,
                v.merge
            ));
        }
        let rd = |x: f32| if r == Rounding::F32 { x } else { bf16(x) };
        // the patch product (its bias after the product's rounding) and the position table resampled to the grid
        let (rc, taps) = grid(gh, gw, v.merge, v.table);
        let mut x = rows(n, |t| {
            let e = product(&self.patch.0, None, &patches[t * v.patch_dim()..(t + 1) * v.patch_dim()], r);
            let mut p = vec![0f32; h];
            for (idx, w) in &taps[t] {
                for (c, p) in p.iter_mut().enumerate() {
                    *p += self.pos[idx * h + c] * w;
                }
            }
            e.iter().zip(&self.patch.1).zip(&p).map(|((&e, &b), &p)| rd(rd(e + b) + rd(p))).collect()
        });
        let embedded = x.concat();
        // the rotary's cos and sin of each patch: a quarter of a head's channels by its row, a quarter by its column, twice over
        let hd = v.head_dim();
        let q4 = hd / 4;
        let inv: Vec<f32> = (0..q4).map(|j| 1.0f32 / (v.theta as f32).powf((2 * j) as f32 / (hd / 2) as f32)).collect();
        let rot: Vec<(Vec<f32>, Vec<f32>)> = rc
            .iter()
            .map(|&(row, col)| {
                let a: Vec<f32> = (0..hd / 2).map(|j| if j < q4 { row as f32 * inv[j] } else { col as f32 * inv[j - q4] }).collect();
                (a.iter().map(|a| a.cos()).collect(), a.iter().map(|a| a.sin()).collect())
            })
            .collect();
        let mut after = Vec::with_capacity(v.blocks);
        for b in &self.blocks {
            x = self.block(b, &x, &rot, r);
            after.push(x.concat());
        }
        let out = self.merge_at(&x.concat(), r);
        Ok(TowerRows { out, embedded, after })
    }

    /// The merger on the patches' rows `x` ([n, hidden], n whole blocks of merge x merge in order): LayerNorm, a block of patches as one row, its two
    /// products with the GELU (erf) between; rounded at `r`'s points.
    pub fn merge_at(&self, x: &[f32], r: Rounding) -> Vec<f32> {
        let (v, h) = (&self.v, self.v.hidden);
        let rd = |x: f32| if r == Rounding::F32 { x } else { bf16(x) };
        let m = v.merge * v.merge;
        let normed: Vec<f32> = x.chunks(h).flat_map(|row| layer_norm(row, &self.merge_norm, v.eps, r)).collect();
        rows(x.len() / h / m, |t| {
            let row = &normed[t * m * h..(t + 1) * m * h];
            let up: Vec<f32> = product(&self.merge_up.0, Some(&self.merge_up.1), row, r).iter().map(|&u| rd(act(Act::Gelu, u))).collect();
            product(&self.merge_down.0, Some(&self.merge_down.1), &up, r)
        })
        .concat()
    }

    /// Block `l` on the patches' rows `x` ([n, hidden]) of an image of `gh` x `gw` patches, rounded at `r`'s points: its rows after it. A check's (each block of an
    /// engine held to the reference on the engine's own input).
    pub fn block_at(&self, l: usize, x: &[f32], gh: usize, gw: usize, r: Rounding) -> Result<Vec<f32>> {
        let (v, h) = (&self.v, self.v.hidden);
        let b = self.blocks.get(l).ok_or_else(|| Error(format!("the tower has no block {l}")))?;
        if x.len() != gh * gw * h {
            return err(format!("{} values for {gh} x {gw} patches of {h}", x.len()));
        }
        let (rc, _) = grid(gh, gw, v.merge, v.table);
        let hd = v.head_dim();
        let q4 = hd / 4;
        let inv: Vec<f32> = (0..q4).map(|j| 1.0f32 / (v.theta as f32).powf((2 * j) as f32 / (hd / 2) as f32)).collect();
        let rot: Vec<(Vec<f32>, Vec<f32>)> = rc
            .iter()
            .map(|&(row, col)| {
                let a: Vec<f32> = (0..hd / 2).map(|j| if j < q4 { row as f32 * inv[j] } else { col as f32 * inv[j - q4] }).collect();
                (a.iter().map(|a| a.cos()).collect(), a.iter().map(|a| a.sin()).collect())
            })
            .collect();
        let rows_in: Vec<Vec<f32>> = x.chunks(h).map(<[f32]>::to_vec).collect();
        Ok(self.block(b, &rows_in, &rot, r).concat())
    }

    /// One block: x + proj(attention(rotary(qkv(ln1(x))))), then + fc2(act(fc1(ln2(x)))).
    fn block(&self, b: &Block, x: &[Vec<f32>], rot: &[(Vec<f32>, Vec<f32>)], r: Rounding) -> Vec<Vec<f32>> {
        let v = &self.v;
        let (n, h, heads, hd) = (x.len(), v.hidden, v.heads, v.head_dim());
        let rd = |x: f32| if r == Rounding::F32 { x } else { bf16(x) };
        let qkv = rows(n, |t| product(&b.qkv.0, Some(&b.qkv.1), &layer_norm(&x[t], &b.n1, v.eps, r), r));
        // q and k turned (rotate_half over the head, the angles twice), each rounded once; v as it is
        let turn = |t: usize, at: usize| -> Vec<f32> {
            let (c, s) = &rot[t];
            (0..heads)
                .flat_map(|hh| {
                    let q = &qkv[t][at + hh * hd..at + (hh + 1) * hd];
                    (0..hd).map(move |d| {
                        let j = d % (hd / 2);
                        let other = if d < hd / 2 { -q[d + hd / 2] } else { q[d - hd / 2] };
                        rd(q[d] * c[j] + other * s[j])
                    })
                })
                .collect()
        };
        let q: Vec<Vec<f32>> = (0..n).map(|t| turn(t, 0)).collect();
        let k: Vec<Vec<f32>> = (0..n).map(|t| turn(t, h)).collect();
        let scale = (hd as f32).powf(-0.5);
        let att = rows(n, |t| {
            let mut o = vec![0f32; h];
            for hh in 0..heads {
                let qh = &q[t][hh * hd..(hh + 1) * hd];
                let mut s: Vec<f32> = (0..n).map(|u| dot(qh, &k[u][hh * hd..(hh + 1) * hd]) * scale).collect();
                if r == Rounding::F32 {
                    softmax(&mut s);
                    for (u, p) in s.iter().enumerate() {
                        for d in 0..hd {
                            o[hh * hd + d] += p * qkv[u][2 * h + hh * hd + d];
                        }
                    }
                } else {
                    // (FlashAttention's: exp of each score less the row's largest, the bf16 of each for the values, their f32 sum dividing at the end)
                    let mx = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let p: Vec<f32> = s.iter().map(|&x| (x - mx).exp()).collect();
                    let l: f32 = p.iter().sum();
                    let mut acc = vec![0f64; hd];
                    for (u, &p) in p.iter().enumerate() {
                        let p = bf16(p) as f64;
                        for (d, a) in acc.iter_mut().enumerate() {
                            *a += p * qkv[u][2 * h + hh * hd + d] as f64;
                        }
                    }
                    for (d, a) in acc.iter().enumerate() {
                        o[hh * hd + d] = bf16((*a as f32) / l);
                    }
                }
            }
            o
        });
        let x1: Vec<Vec<f32>> = (0..n).map(|t| product(&b.o.0, Some(&b.o.1), &att[t], r).iter().zip(&x[t]).map(|(&a, &x)| rd(x + a)).collect()).collect();
        rows(n, |t| {
            let up: Vec<f32> = product(&b.up.0, Some(&b.up.1), &layer_norm(&x1[t], &b.n2, v.eps, r), r).iter().map(|&u| rd(act(v.act, u))).collect();
            product(&b.down.0, Some(&b.down.1), &up, r).iter().zip(&x1[t]).map(|(&d, &x)| rd(x + d)).collect()
        })
    }
}

/// Each patch's (row, column) of an image of `gh` x `gw` patches in its `merge` x `merge` blocks' order, and its taps into the `table` x `table` position
/// table (bilinear, corners aligned: transformers' `get_vision_interpolation_indices_and_weights`), the weights in f32.
pub fn grid(gh: usize, gw: usize, merge: usize, table: usize) -> (Vec<(usize, usize)>, Vec<Vec<(usize, f32)>>) {
    let axis = |i: usize, size: usize| -> ([usize; 2], [f32; 2]) {
        let src = i as f32 * (table - 1) as f32 / (size - 1).max(1) as f32;
        let fl = src.floor();
        let t = |o: f32| ((fl + o) as i64).clamp(0, table as i64 - 1) as usize;
        let w = |o: f32| (1.0 - (src - fl - o).abs()).max(0.0);
        ([t(0.0), t(1.0)], [w(0.0), w(1.0)])
    };
    let (mut rc, mut taps) = (Vec::with_capacity(gh * gw), Vec::with_capacity(gh * gw));
    for br in 0..gh / merge {
        for bc in 0..gw / merge {
            for ir in 0..merge {
                for ic in 0..merge {
                    let (r, c) = (br * merge + ir, bc * merge + ic);
                    rc.push((r, c));
                    let ((hi, hw), (wi, ww)) = (axis(r, gh), axis(c, gw));
                    taps.push((0..2).flat_map(|a| (0..2).map(move |b| (hi[a] * table + wi[b], hw[a] * ww[b]))).collect());
                }
            }
        }
    }
    (rc, taps)
}

/// LayerNorm (mean and the biased variance in f32, `eps` inside the root) with its weight and bias, rounded once at `r`'s points.
fn layer_norm(x: &[f32], wb: &(Vec<f32>, Vec<f32>), eps: f32, r: Rounding) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let rs = 1.0 / (var + eps).sqrt();
    x.iter().zip(&wb.0).zip(&wb.1).map(|((&v, &w), &b)| (v - mean) * rs * w + b).map(|y| if r == Rounding::F32 { y } else { bf16(y) }).collect()
}

/// W x (each sum in f64, then f32) plus the bias where there is one (in f32), rounded once at `r`'s points; with no bias (the patch product, whose bias is
/// added after: `run`), rounded as it is.
fn product(w: &[f32], bias: Option<&[f32]>, x: &[f32], r: Rounding) -> Vec<f32> {
    let y = super::matvec(w, x.len(), x);
    let rd = |x: f32| if r == Rounding::F32 { x } else { bf16(x) };
    match bias {
        Some(b) => y.iter().zip(b).map(|(&y, &b)| rd(y + b)).collect(),
        None => y.into_iter().map(rd).collect(),
    }
}

/// `f` of each of `n` rows, on the computer's cores.
fn rows<F: Fn(usize) -> Vec<f32> + Sync>(n: usize, f: F) -> Vec<Vec<f32>> {
    let threads = std::thread::available_parallelism().map_or(1, |p| p.get()).min(n.max(1));
    let mut out: Vec<Vec<f32>> = vec![Vec::new(); n];
    std::thread::scope(|s| {
        for (k, part) in out.chunks_mut(n.div_ceil(threads).max(1)).enumerate() {
            let f = &f;
            let at = k * n.div_ceil(threads).max(1);
            s.spawn(move || {
                for (i, o) in part.iter_mut().enumerate() {
                    *o = f(at + i);
                }
            });
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The grid: each patch's place in its 2 x 2 block's order, and the four taps of a grid as large as the table are the table's own rows with weight one
    /// (corners aligned); a smaller grid's corners are the table's corners.
    #[test]
    fn patches_in_blocks_and_their_taps() {
        let (rc, taps) = grid(4, 6, 2, 4);
        assert_eq!(&rc[..8], &[(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3), (1, 2), (1, 3)]);
        let (rc, taps4) = grid(4, 4, 2, 4);
        for (i, &(r, c)) in rc.iter().enumerate() {
            let w: f32 = taps4[i].iter().map(|t| t.1).sum();
            assert!((w - 1.0).abs() < 1e-6);
            assert!(taps4[i].iter().any(|&(idx, w)| idx == r * 4 + c && w == 1.0), "{r} {c}: {:?}", taps4[i]);
        }
        // (the last patch of a 4 x 6 grid on a table of 4: its taps at the table's last row and column)
        let last = rc.len();
        let _ = last;
        let corner = taps.iter().zip(grid(4, 6, 2, 4).0).find(|(_, rc)| *rc == (3, 5)).unwrap().0;
        assert!(corner.iter().any(|&(idx, w)| idx == 15 && w == 1.0), "{corner:?}");
    }

    /// The tower on a tiny random checkpoint: rounded at the release's points it is within bf16's rounding of itself in f32 (its noise floor is small), a
    /// block alone on its input is the tower's own step, and its rows are its patches'.
    #[test]
    fn a_tiny_tower() {
        let v = VisionDesc {
            blocks: 2,
            hidden: 32,
            heads: 2,
            inter: 48,
            act: Act::GeluTanh,
            channels: 3,
            patch: 2,
            frames: 2,
            merge: 2,
            out: 24,
            table: 4,
            theta: 10_000.0,
            eps: 1e-6,
            image_token: 9,
        };
        let roles = crate::runtime::describe::VisionRoles {
            own: crate::runtime::names::roles(crate::runtime::names::part::VISION, "t", 0, |_| true),
            blocks: (0..2).map(|l| crate::runtime::names::roles(crate::runtime::names::part::VISION_BLOCK, &format!("t.blocks.{l}"), 0, |_| true)).collect(),
        };
        let mut seed = 7u64;
        let mut u = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f32 / (1u64 << 53) as f32 - 0.5
        };
        let mut w = std::collections::HashMap::new();
        for (_, s) in roles.own.iter().chain(roles.blocks.iter().flatten()) {
            let crate::runtime::describe::Source::Tensor(n) = s else { unreachable!() };
            let leaf = n.rsplit('.').next().unwrap();
            let len = |n: &str| -> usize {
                let h = 32;
                match () {
                    _ if n.contains("patch_embed.proj.weight") => h * 24,
                    _ if n.contains("pos_embed") => 16 * h,
                    _ if n.contains("merger.linear_fc1.weight") => 128 * 128,
                    _ if n.contains("merger.linear_fc1.bias") => 128,
                    _ if n.contains("merger.linear_fc2.weight") => 24 * 128,
                    _ if n.contains("merger.linear_fc2.bias") => 24,
                    _ if n.contains("qkv.weight") => 3 * h * h,
                    _ if n.contains("qkv.bias") => 3 * h,
                    _ if n.contains("proj.weight") => h * h,
                    _ if n.contains("fc1.weight") => 48 * h,
                    _ if n.contains("fc1.bias") => 48,
                    _ if n.contains("fc2.weight") => h * 48,
                    _ => h,
                }
            };
            let vals: Vec<f32> = (0..len(n))
                .map(|_| {
                    bf16(if n.contains("norm") && leaf == "weight" {
                        1.0 + u() * 0.2
                    } else if leaf == "bias" {
                        u() * 0.2
                    } else {
                        u() * 0.6
                    })
                })
                .collect();
            w.insert(n.clone(), vals);
        }
        let src = |n: &str| w.get(n).map(|x| (vec![x.len()], x.clone()));
        let t = RefTower::load(&v, &roles, &src).unwrap();
        let (gh, gw) = (4, 4);
        let px: Vec<f32> = (0..gh * gw * 24).map(|_| bf16(u() * 2.0)).collect();
        let a = t.run(&px, gh, gw, Rounding::Once).unwrap();
        let f = t.run(&px, gh, gw, Rounding::F32).unwrap();
        assert_eq!(a.out.len(), 4 * 24);
        let scale = f.out.iter().fold(0f32, |m, x| m.max(x.abs()));
        let far = a.out.iter().zip(&f.out).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max) / scale;
        assert!(far < 2e-2 && far > 0.0, "{far}");
        // a block alone, on the tower's own rows before it, is the tower's next step
        let again = t.block_at(1, &a.after[0], gh, gw, Rounding::Once).unwrap();
        assert_eq!(again, a.after[1]);
        // (an image of other blocks is other rows: a tower that read no patch would pass the checks above)
        let other: Vec<f32> = px.iter().map(|x| -x).collect();
        assert_ne!(t.run(&other, gh, gw, Rounding::F32).unwrap().out, f.out);
    }
}
