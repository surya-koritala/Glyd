//! The CPU half of GLM-5.3-Flash's expert offload (the `glyd-hbr-codes/2` pack): a routed expert held as codebook rows with no run scales, computed in place from system RAM for a pair (token, expert) whose expert is
//! not on the GPU. It is src/cpu_hb.rs for this pack, and its interface is the same: `GlmExpert` (borrowed views of the expert's arrays) for `HbExpert`, `GlmPair` for `HbPair`, `experts_pinned` for
//! `cpu_hb::experts_pinned`. What differs from the Nemotron experts: the width is 4,096 and the intermediate size 2,048, the first matrix has the gate's 2,048 rows and then the up's 2,048, the activation is
//! GLM's clamped SwiGLU, silu(min(gate, limit)) times clamp(up, -limit, limit), every neuron is active (the second product is dense), and a row's weights are the entries of its table (ternary: 3, two bits: 4, three: 8, four: 16),
//! with no scale for a run of 32 (the row's scale is in its entries). An expert's two matrices may be of different formats (`GlmExpert::mixed`).
//!
//! The bytes (`Fmt`, and glm_pack.py of the pack): `up` [4096 rows, 4096 columns] as codes [4096, row bytes] and a codebook [4096, entries] of bf16; `down` [2048 neuron rows, 4096 columns] the same way, row n holding the
//! weights that neuron n's activation multiplies (the second matrix transposed). z is the layer's input in the rotated basis (R z once for the layer: the pack's rows are in it), and a pair's row of `out` is the routing
//! weight times the sum over the neurons of the activation times row n, in the rotated basis: the caller adds a token's pairs and the engine rotates back once.
//!
//! The arithmetic (the pack's: README.md of the pack, testvec.safetensors): a first-product sum is fp32 (per row two vectors of 16 lane sums, a fused multiply-add per run of 32 codes, the lanes added as a tree, the two
//! vectors' sums added; a ternary row's lanes are its 32 bytes of a group of 160 codes, the five digits' products added to the lane's total), the activation made from the gate's and the up's in fp32 (`silu(x) = x / (1 + exp(-x))`)
//! and rounded to bf16 once (the pack's `act_bf16`); an output of the second product is a chain of fused multiply-adds over the neurons in order (the pack's `y_rot`). Every kernel (scalar, NEON, AVX2 with FMA,
//! AVX-512) makes the same bits, and the work is cut so that they do not depend on the number of threads: items of 64 neurons of the first product (both its matrices' rows), then items of 256 outputs (a ternary
//! matrix's 160) of the second, which wait for their pair's activations.

use crate::cpu_experts::{bf16_round, run, Kernel, Z};
use crate::cpu_hb::CbView;
use crate::cpu_simd as simd;
pub use crate::cpu_simd::{Fmt, GlmMat};
use crate::ffi::{err, Result};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::Mutex;

/// Neurons a first-product item takes (its gate rows and its up rows), and outputs an item of the second product.
const NEURON_BLOCK: usize = 64;
const OUT_SLICE: usize = 256;

/// A routed expert: `up` [2 inter, hidden] (the gate's rows, then the up's) and `down` [inter, hidden] (the second matrix transposed), codebook rows with no run scales, and the clamp of the SwiGLU (`swiglu_limit`, 10).
#[derive(Clone, Copy, Debug)]
pub struct GlmExpert<'a> {
    pub up: GlmMat<'a>,
    pub down: GlmMat<'a>,
    pub limit: f32,
}

impl<'a> GlmExpert<'a> {
    /// An expert whose matrices are both of two, three or four bits (`CbView::bits`; no scales), as `cpu_hb`'s views name them.
    pub fn new(up: CbView<'a>, down: CbView<'a>, limit: f32) -> Result<GlmExpert<'a>> {
        let mat = |v: CbView<'a>| -> Option<GlmMat<'a>> {
            let fmt = match v.bits {
                2 => Fmt::Cb2,
                3 => Fmt::Cb3,
                4 => Fmt::Cb4,
                _ => return None,
            };
            v.scales.is_empty().then_some(GlmMat { fmt, rows: v.rows, cols: v.cols, codes: v.codes, book: v.codebook })
        };
        match (mat(up), mat(down)) {
            (Some(u), Some(d)) => GlmExpert::mixed(u, d, limit),
            _ => err(format!("an expert's up is {} x {} at {} bits and its transposed down {} x {} at {} (two to four bits and no run scales are wanted)", up.rows, up.cols, up.bits, down.rows, down.cols, down.bits)),
        }
    }

    /// An expert whose matrices have formats of their own (the pack names a mixed expert 'X+Y': X for the gate and the up, Y for the down).
    pub fn mixed(up: GlmMat<'a>, down: GlmMat<'a>, limit: f32) -> Result<GlmExpert<'a>> {
        if !up.is_valid() || !down.is_valid() || up.rows != 2 * down.rows || up.cols != down.cols || down.rows % NEURON_BLOCK != 0 || !limit.is_finite() || limit <= 0.0 {
            return err(format!(
                "an expert's up is {} x {} ({}) and its transposed down {} x {} ({}) (not a shape this engine reads), with a clamp of {limit}",
                up.rows,
                up.cols,
                up.fmt.name(),
                down.rows,
                down.cols,
                down.fmt.name()
            ));
        }
        Ok(GlmExpert { up, down, limit })
    }

    /// Neurons (the intermediate size) and the width of the input and the output.
    pub fn inter(&self) -> usize {
        self.down.rows
    }

    pub fn hidden(&self) -> usize {
        self.down.cols
    }

    /// The bytes of the expert's arrays: what a pair reads.
    pub fn bytes(&self) -> usize {
        self.up.codes.len() + self.up.book.len() + self.down.codes.len() + self.down.book.len()
    }
}

/// A token's use of an expert: token `token` (a row of z) through `expert`, its result times `weight` (the routing weight).
#[derive(Clone, Copy, Debug)]
pub struct GlmPair<'a> {
    pub expert: &'a GlmExpert<'a>,
    pub token: usize,
    pub weight: f32,
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// The pairs' results with the best kernel (`Kernel::best`), `threads` threads (0: as many as the computer has): `out` [pairs, hidden] gets pair p's row, `z` [tokens, hidden] the rotated inputs. Returns the number of
/// activations that are not 0 of each pair. See [`experts_pinned`].
pub fn experts(z: Z, pairs: &[GlmPair], out: &mut [f32], threads: usize) -> Result<Vec<u32>> {
    experts_pinned(Kernel::best(), z, pairs, out, threads, false)
}

/// [`experts`] with the kernel asked for (an error where this computer cannot run it); `pin`: the threads of the crew each stay on a core of their own (see `cpu_experts::Workers`).
pub fn experts_pinned(kernel: Kernel, z: Z, pairs: &[GlmPair], out: &mut [f32], threads: usize, pin: bool) -> Result<Vec<u32>> {
    if !kernel.runs_here() {
        return err(format!("the {} kernel cannot run on this computer", kernel.name()));
    }
    let Some(first) = pairs.first() else { return Ok(Vec::new()) };
    let (hidden, inter) = (first.expert.hidden(), first.expert.inter());
    let zlen = match z {
        Z::F32(v) => v.len(),
        Z::Bf16(v) => v.len(),
    };
    if hidden == 0 || zlen % hidden != 0 || out.len() != pairs.len() * hidden {
        return err(format!("z holds {zlen} values and out {}, where rows of {hidden} and {} pairs are wanted", out.len(), pairs.len()));
    }
    let tokens = zlen / hidden;
    // z's rows that the pairs use, as fp32
    let (mut at, mut used) = (vec![usize::MAX; tokens], Vec::new());
    for (p, pair) in pairs.iter().enumerate() {
        let e = pair.expert;
        if !e.up.is_valid() || !e.down.is_valid() || e.hidden() != hidden || e.inter() != inter || e.up.rows != 2 * inter || pair.token >= tokens {
            return err(format!("pair {p}: an expert of {} neurons and width {} and token {}, where {inter}, {hidden} and 0 to {} are wanted", e.inter(), e.hidden(), pair.token, tokens.saturating_sub(1)));
        }
        if at[pair.token] == usize::MAX {
            at[pair.token] = used.len();
            used.push(pair.token);
        }
    }
    let mut zp = vec![0f32; used.len() * hidden];
    for (row, &t) in zp.chunks_mut(hidden).zip(&used) {
        match z {
            Z::F32(v) => row.copy_from_slice(&v[t * hidden..(t + 1) * hidden]),
            Z::Bf16(v) => row.iter_mut().zip(&v[t * hidden..(t + 1) * hidden]).for_each(|(d, &b)| *d = f32::from_bits((b as u32) << 16)),
        }
    }
    // a ternary first matrix reads z in the order of its groups of 160 (`tern_permute`), made for the tokens that use one
    let zlen = 160 * simd::tern_groups(hidden);
    let mut zq: Vec<f32> = Vec::new();
    if pairs.iter().any(|p| p.expert.up.fmt == Fmt::Tern) {
        zq = vec![0f32; used.len() * zlen];
        for (q, row) in zq.chunks_mut(zlen).zip(zp.chunks(hidden)) {
            simd::tern_permute(row, q);
        }
    }
    // the pairs of one expert (a verification's rows, the tokens of a chunk) are an item's together, each pair's own arithmetic as alone: the expert's rows are read from memory once and from the core's cache for the
    // others. Each group's second-product items: a slice of 256 outputs (a ternary matrix's 160) each
    let mut groups: Vec<Vec<usize>> = Vec::new();
    {
        let mut at: Vec<(*const u8, usize)> = Vec::new();
        for (p, pair) in pairs.iter().enumerate() {
            let key = pair.expert.up.codes.as_ptr();
            match at.iter().find(|(k, _)| *k == key) {
                Some(&(_, g)) => groups[g].push(p),
                None => {
                    at.push((key, groups.len()));
                    groups.push(vec![p]);
                }
            }
        }
    }
    let (nb1, slice_of) = (inter / NEURON_BLOCK, |g: usize| pairs[groups[g][0]].expert.down.fmt.slice());
    let mut start2 = vec![0usize; groups.len() + 1];
    for g in 0..groups.len() {
        start2[g + 1] = start2[g] + hidden.div_ceil(slice_of(g));
    }
    let n1 = groups.len() * nb1;
    // each pair's activations (written by its group's first-product items, read by its second-product items once `ready`), the items left of each group's first product, and the count of activations that are not 0
    let act: Vec<AtomicU32> = (0..pairs.len() * inter).map(|_| AtomicU32::new(0)).collect();
    let remaining: Vec<AtomicUsize> = (0..groups.len()).map(|_| AtomicUsize::new(nb1)).collect();
    let ready: Vec<AtomicBool> = (0..groups.len()).map(|_| AtomicBool::new(false)).collect();
    let nonzero: Vec<AtomicU32> = (0..pairs.len()).map(|_| AtomicU32::new(0)).collect();
    let failed = AtomicBool::new(false);
    // (each second-product item writes its own slice of its pair's row: no item's slice is another's, so the rows are written through a pointer, without a lock that ten items of a pair would queue on)
    let rows = OutRows(out.as_mut_ptr(), out.len());
    // (a development build's GLYD_HOST_TRACE=1: each item's start and end on the call's clock and the thread that ran it, printed after the call)
    #[cfg(feature = "dev")]
    let (tracing, t_call, spans) = (std::env::var("GLYD_HOST_TRACE").is_ok_and(|v| v == "1"), std::time::Instant::now(), Mutex::new(Vec::<(usize, f64, f64, u64)>::new()));
    let work = |item: usize| {
        #[cfg(feature = "dev")]
        let t_item = t_call.elapsed().as_secs_f64() * 1e6;
        #[cfg(feature = "dev")]
        let _span = Span(tracing, item, t_item, &t_call, &spans);
        if item < n1 {
            let (g, b) = (item / nb1, item % nb1);
            let n0 = b * NEURON_BLOCK;
            let body = || {
                // (trellis rows: a group's pairs up to four at a time through one decode of each run, each pair's sums the bits it has alone)
                let tq = pairs[groups[g][0]].expert.up.fmt.is_tq();
                for chunk in groups[g].chunks(if tq { 4 } else { 1 }) {
                    let e = pairs[chunk[0]].expert;
                    let (mut gs, mut u) = ([[0f32; NEURON_BLOCK]; 4], [[0f32; NEURON_BLOCK]; 4]);
                    if tq {
                        let n = chunk.len();
                        let zs: [&[f32]; 4] = core::array::from_fn(|j| {
                            let t = at[pairs[chunk[j.min(n - 1)]].token];
                            &zp[t * hidden..(t + 1) * hidden]
                        });
                        let [g0, g1, g2, g3] = &mut gs;
                        simd::dot_glm_multi(kernel, &e.up, &zs[..n], n0, n0 + NEURON_BLOCK, &mut [&mut g0[..], &mut g1[..], &mut g2[..], &mut g3[..]][..n]);
                        let [u0, u1, u2, u3] = &mut u;
                        simd::dot_glm_multi(kernel, &e.up, &zs[..n], inter + n0, inter + n0 + NEURON_BLOCK, &mut [&mut u0[..], &mut u1[..], &mut u2[..], &mut u3[..]][..n]);
                    } else {
                        let t = at[pairs[chunk[0]].token];
                        let zt = if e.up.fmt == Fmt::Tern { &zq[t * zlen..(t + 1) * zlen] } else { &zp[t * hidden..(t + 1) * hidden] };
                        simd::dot_glm(kernel, &e.up, zt, n0, n0 + NEURON_BLOCK, &mut gs[0]);
                        simd::dot_glm(kernel, &e.up, zt, inter + n0, inter + n0 + NEURON_BLOCK, &mut u[0]);
                    }
                    for (j, &p) in chunk.iter().enumerate() {
                        let mut nz = 0;
                        for i in 0..NEURON_BLOCK {
                            let a = bf16_round(silu(gs[j][i].min(e.limit)) * u[j][i].clamp(-e.limit, e.limit));
                            act[p * inter + n0 + i].store(a.to_bits(), Ordering::Relaxed);
                            nz += (a != 0.0) as u32;
                        }
                        nonzero[p].fetch_add(nz, Ordering::Relaxed);
                    }
                }
            };
            // (a group's second-product items wait for its activations: if a first-product item panics they are told so, and the panic goes on)
            let ok = catch_unwind(AssertUnwindSafe(body)).is_ok();
            if !ok {
                failed.store(true, Ordering::Relaxed);
            }
            if remaining[g].fetch_sub(1, Ordering::AcqRel) == 1 {
                ready[g].store(true, Ordering::Release);
            }
            assert!(ok, "a first-product item panicked");
        } else {
            let j = item - n1;
            let g = start2.partition_point(|&x| x <= j) - 1;
            let sidx = j - start2[g];
            while !ready[g].load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            if failed.load(Ordering::Relaxed) {
                return;
            }
            let (slice, o0) = (slice_of(g), sidx * slice_of(g));
            // SAFETY: a pair's activations were written before its group's `ready` was set (released and acquired above), and an AtomicU32 has the layout of the u32 whose bits are the f32.
            let act_of = |p: usize| unsafe { std::slice::from_raw_parts(act.as_ptr().add(p * inter) as *const f32, inter) };
            // (trellis rows: a group's pairs two at a time through one decode of each run, as two slices of 128 outputs, each pair's outputs the bits it has alone)
            let tq = pairs[groups[g][0]].expert.down.fmt.is_tq();
            for chunk in groups[g].chunks(if tq { 2 } else { 1 }) {
                let mut bufs = [[0f32; OUT_SLICE]; 2];
                if tq {
                    let n = chunk.len();
                    let acts: [&[f32]; 2] = core::array::from_fn(|j| act_of(chunk[j.min(n - 1)]));
                    for half in 0..OUT_SLICE / 128 {
                        let [b0, b1] = &mut bufs;
                        let (h0, h1) = (128 * half, 128 * half + 128);
                        simd::down_glm_multi(kernel, &pairs[chunk[0]].expert.down, &acts[..n], o0 + h0, &mut [&mut b0[h0..h1], &mut b1[h0..h1]][..n]);
                    }
                } else {
                    simd::down_glm(kernel, &pairs[chunk[0]].expert.down, act_of(chunk[0]), o0, &mut bufs[0]);
                }
                for (j, &p) in chunk.iter().enumerate() {
                    let end = (o0 + slice).min(hidden);
                    // SAFETY: outputs o0 .. end of pair p's row are this item's alone (the items of a group cut the row into disjoint slices, and a pair is in one group), inside `out` (p < pairs, end <= hidden).
                    let row = unsafe { rows.slice(p * hidden + o0, end - o0) };
                    for (o, v) in row.iter_mut().zip(bufs[j]) {
                        *o = v * pairs[p].weight;
                    }
                }
            }
        }
    };
    run(threads, pin, n1 + start2[groups.len()], &work);
    #[cfg(feature = "dev")]
    if tracing {
        let mut v = spans.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner);
        v.sort_by(|a, b| a.1.total_cmp(&b.1));
        let line: Vec<String> = v.iter().map(|(i, a, b, t)| format!("{i}{}@{t}:{a:.0}-{b:.0}", if *i < n1 { "u" } else { "d" })).collect();
        eprintln!("host call: {} pairs, {} groups, {:.0} us: {}", pairs.len(), groups.len(), t_call.elapsed().as_secs_f64() * 1e6, line.join(" "));
    }
    Ok(nonzero.iter().map(|n| n.load(Ordering::Relaxed)).collect())
}

/// [`experts_pinned`]'s results with the pairs of an expert taken together (the host's share of a prompt's streamed call: an expert's several tokens): a row of an expert's matrices is decoded once for up to four of its
/// tokens in the first product (`dot_glm_multi`) and two in the second (`down_glm_multi`), each pair's sums the same multiply-adds in the same order as alone (the same bits as `experts_pinned`). Items of 64 neurons of
/// an expert's first product (its tokens' gate and up rows), then items of 128 outputs of its second, which wait for its activations. No ternary matrices; `out` [pairs, hidden] gets pair p's row times its weight.
pub fn experts_grouped(kernel: Kernel, z: Z, pairs: &[GlmPair], out: &mut [f32], threads: usize) -> Result<()> {
    if !kernel.runs_here() {
        return err(format!("the {} kernel cannot run on this computer", kernel.name()));
    }
    let Some(first) = pairs.first() else { return Ok(()) };
    let (hidden, inter) = (first.expert.hidden(), first.expert.inter());
    let zlen = match z {
        Z::F32(v) => v.len(),
        Z::Bf16(v) => v.len(),
    };
    if hidden == 0 || hidden % 128 != 0 || zlen % hidden != 0 || out.len() != pairs.len() * hidden {
        return err(format!("z holds {zlen} values and out {}, where rows of {hidden} (a multiple of 128) and {} pairs are wanted", out.len(), pairs.len()));
    }
    let tokens = zlen / hidden;
    let (mut at, mut used) = (vec![usize::MAX; tokens], Vec::new());
    for (p, pair) in pairs.iter().enumerate() {
        let e = pair.expert;
        if e.up.fmt == Fmt::Tern || e.down.fmt == Fmt::Tern || e.hidden() != hidden || e.inter() != inter || e.up.rows != 2 * inter || pair.token >= tokens {
            return err(format!("pair {p}: an expert of {} neurons, width {} ({} and {}) and token {}, where {inter}, {hidden}, the pack's format and 0 to {} are wanted", e.inter(), e.hidden(), e.up.fmt.name(), e.down.fmt.name(), pair.token, tokens.saturating_sub(1)));
        }
        if at[pair.token] == usize::MAX {
            at[pair.token] = used.len();
            used.push(pair.token);
        }
    }
    let mut zp = vec![0f32; used.len() * hidden];
    for (row, &t) in zp.chunks_mut(hidden).zip(&used) {
        match z {
            Z::F32(v) => row.copy_from_slice(&v[t * hidden..(t + 1) * hidden]),
            Z::Bf16(v) => row.iter_mut().zip(&v[t * hidden..(t + 1) * hidden]).for_each(|(d, &b)| *d = f32::from_bits((b as u32) << 16)),
        }
    }
    // the pairs of each expert, by the address of its codes
    let mut groups: Vec<Vec<usize>> = Vec::new();
    {
        let mut key: Vec<(*const u8, usize)> = Vec::new();
        for (p, pair) in pairs.iter().enumerate() {
            let k = pair.expert.up.codes.as_ptr();
            match key.iter().find(|(c, _)| *c == k) {
                Some(&(_, g)) => groups[g].push(p),
                None => {
                    key.push((k, groups.len()));
                    groups.push(vec![p]);
                }
            }
        }
    }
    let (nb1, nb2) = (inter / NEURON_BLOCK, hidden / 128);
    let n1 = groups.len() * nb1;
    let act: Vec<AtomicU32> = (0..pairs.len() * inter).map(|_| AtomicU32::new(0)).collect();
    let remaining: Vec<AtomicUsize> = (0..groups.len()).map(|_| AtomicUsize::new(nb1)).collect();
    let ready: Vec<AtomicBool> = (0..groups.len()).map(|_| AtomicBool::new(false)).collect();
    let failed = AtomicBool::new(false);
    let rows: Vec<Mutex<&mut [f32]>> = out.chunks_mut(hidden).map(Mutex::new).collect();
    let work = |item: usize| {
        if item < n1 {
            let (g, b) = (item / nb1, item % nb1);
            let n0 = b * NEURON_BLOCK;
            let body = || {
                let e = pairs[groups[g][0]].expert;
                for batch in groups[g].chunks(4) {
                    let zs: Vec<&[f32]> = batch.iter().map(|&p| &zp[at[pairs[p].token] * hidden..(at[pairs[p].token] + 1) * hidden]).collect();
                    let (mut gs, mut us) = (vec![[0f32; NEURON_BLOCK]; batch.len()], vec![[0f32; NEURON_BLOCK]; batch.len()]);
                    {
                        let mut s: Vec<&mut [f32]> = gs.iter_mut().map(|x| &mut x[..]).collect();
                        simd::dot_glm_multi(kernel, &e.up, &zs, n0, n0 + NEURON_BLOCK, &mut s);
                        let mut s: Vec<&mut [f32]> = us.iter_mut().map(|x| &mut x[..]).collect();
                        simd::dot_glm_multi(kernel, &e.up, &zs, inter + n0, inter + n0 + NEURON_BLOCK, &mut s);
                    }
                    for (i, &p) in batch.iter().enumerate() {
                        for j in 0..NEURON_BLOCK {
                            let a = bf16_round(silu(gs[i][j].min(e.limit)) * us[i][j].clamp(-e.limit, e.limit));
                            act[p * inter + n0 + j].store(a.to_bits(), Ordering::Relaxed);
                        }
                    }
                }
            };
            let ok = catch_unwind(AssertUnwindSafe(body)).is_ok();
            if !ok {
                failed.store(true, Ordering::Relaxed);
            }
            if remaining[g].fetch_sub(1, Ordering::AcqRel) == 1 {
                ready[g].store(true, Ordering::Release);
            }
            assert!(ok, "a first-product item panicked");
        } else {
            let (g, sidx) = ((item - n1) / nb2, (item - n1) % nb2);
            while !ready[g].load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            if failed.load(Ordering::Relaxed) {
                return;
            }
            let o0 = sidx * 128;
            let e = pairs[groups[g][0]].expert;
            for batch in groups[g].chunks(2) {
                // SAFETY: the pairs' activations were written before the group's `ready` was set (released and acquired above), and an AtomicU32 has the layout of the u32 whose bits are the f32.
                let acts: Vec<&[f32]> = batch.iter().map(|&p| unsafe { std::slice::from_raw_parts(act.as_ptr().add(p * inter) as *const f32, inter) }).collect();
                let mut bufs = vec![[0f32; 128]; batch.len()];
                {
                    let mut o: Vec<&mut [f32]> = bufs.iter_mut().map(|x| &mut x[..]).collect();
                    simd::down_glm_multi(kernel, &e.down, &acts, o0, &mut o);
                }
                for (i, &p) in batch.iter().enumerate() {
                    let mut row = rows[p].lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                    for (o, v) in row[o0..o0 + 128].iter_mut().zip(bufs[i]) {
                        *o = v * pairs[p].weight;
                    }
                }
            }
        }
    };
    run(threads, false, n1 + groups.len() * nb2, &work);
    if failed.load(Ordering::Relaxed) {
        return err("an item of the host's experts failed");
    }
    Ok(())
}

/// A pack's expert against the pack's own test vector (a development build's `glyd-engine cb2check DIR --layer L --expert E`): layer L's file of DIR holds expert E, `testvec.safetensors` has the layer's input for it rotated and
/// the expected sums. The first product's 4,096 sums against `gate_up`, the activations against `act`, the pair's row against `y_rot`, and the row rotated back against `y` (the rotation read from the vector's `signs`).
#[cfg(feature = "dev")]
pub fn check_pack(dir: &std::path::Path, layer: usize, expert: usize) -> Result<()> {
    use crate::ffi::Error;
    use glyd_gpu::safetensors::File;
    let (lf, tv) = (File::open(&dir.join(format!("layer-{layer:03}.safetensors"))).map_err(|e| Error(format!("layer {layer}: {e}")))?, File::open(&dir.join("testvec.safetensors")).map_err(|e| Error(format!("testvec: {e}")))?);
    let read = |f: &File, name: &str| -> Result<Vec<u8>> {
        let t = f.get(name).ok_or_else(|| Error(format!("{name}: not in the file")))?;
        f.read(t).map_err(|e| Error(format!("{name}: {e}")))
    };
    let (mat, vec) = (|name: &str| read(&lf, &format!("e{expert:03}.{name}")), |name: &str| read(&tv, name));
    let (up_codes, up_book, down_codes, down_book) = (mat("u.0")?, mat("u.1")?, mat("d.0")?, mat("d.1")?);
    fn view<'a>(rows: usize, codes: &'a [u8], book: &'a [u8]) -> CbView<'a> {
        CbView { bits: 2, rows, cols: 4096, codes, scales: &[], codebook: book }
    }
    let e = GlmExpert::new(view(4096, &up_codes, &up_book), view(2048, &down_codes, &down_book), 10.0)?;
    let f32s = |b: Vec<u8>| -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() };
    let bf16s = |b: Vec<u8>| -> Vec<u16> { b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect() };
    let (x_rot, gate_up, act, act_bf16, y_rot, y, signs) = (bf16s(vec("x_rot_bf16")?), f32s(vec("gate_up")?), f32s(vec("act")?), bf16s(vec("act_bf16")?), f32s(vec("y_rot")?), f32s(vec("y")?), vec("signs")?);
    let rel = |got: &[f32], want: &[f32]| -> f64 {
        let (mut e2, mut n2) = (0f64, 0f64);
        got.iter().zip(want).for_each(|(&a, &b)| {
            e2 += (a as f64 - b as f64).powi(2);
            n2 += (b as f64).powi(2);
        });
        (e2 / n2.max(1e-30)).sqrt()
    };
    let kernel = Kernel::best();
    let z: Vec<f32> = x_rot.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
    let mut sums = vec![0f32; 4096];
    simd::dot_glm(kernel, &e.up, &z, 0, 4096, &mut sums);
    let a: Vec<f32> = (0..2048).map(|n| silu(sums[n].min(10.0)) * sums[2048 + n].clamp(-10.0, 10.0)).collect();
    let mut out = vec![0f32; 4096];
    experts_pinned(kernel, Z::Bf16(&x_rot), &[GlmPair { expert: &e, token: 0, weight: 1.0 }], &mut out, 16, false)?;
    // R^T y' = s * (4096^-0.5 H y'): the butterflies of the natural-order Hadamard transform, the signs after
    let mut back = out.clone();
    let mut h = 1;
    while h < back.len() {
        for i in (0..back.len()).step_by(2 * h) {
            for j in i..i + h {
                (back[j], back[j + h]) = (back[j] + back[j + h], back[j] - back[j + h]);
            }
        }
        h *= 2;
    }
    back.iter_mut().zip(&signs).for_each(|(v, &s)| *v *= (4096f32).powf(-0.5) * (if s as i8 > 0 { 1.0 } else { -1.0 }));
    let act_want: Vec<f32> = act_bf16.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
    let rows = [("the first product (gate_up)", rel(&sums, &gate_up), 2e-4), ("the activations (act)", rel(&a, &act), 5e-4), ("the pair's row (y_rot)", rel(&out, &y_rot), 6e-3), ("the row (y)", rel(&back, &y), 6e-3)];
    let mut ok = true;
    for (what, r, bound) in rows {
        println!("{} {what}: {r:.2e} of the vector's norm (bound {bound:.0e}), the {} kernel", if r < bound { "ok  " } else { "FAIL" }, kernel.name());
        ok &= r < bound;
    }
    println!("(the activations' bf16 against the vector's act_bf16: {:.2e})", rel(&a.iter().map(|&v| bf16_round(v)).collect::<Vec<_>>(), &act_want));
    if ok {
        Ok(())
    } else {
        err("the expert does not match the pack's test vector")
    }
}

/// The four formats against fixtures the pack's own writer made (a development build's `glyd-engine glmfmtcheck DIR`, DIR from tools/fmtfix.py: fmt-{tern,cb2,cb3,cb4}.safetensors holding `codes` as glm_pack.pack_rows wrote them, `codebook`
/// and `idx`, the codes before packing): for each format, the first product of every row and the second product's every slice, by every kernel this computer runs, against the f64 products of the weights `idx` names.
#[cfg(feature = "dev")]
pub fn check_formats(dir: &std::path::Path) -> Result<()> {
    use crate::cpu_experts::bf16_to_f32;
    use crate::ffi::Error;
    use glyd_gpu::safetensors::File;
    let mut bad = false;
    for fmt in [Fmt::Tern, Fmt::Cb2, Fmt::Cb3, Fmt::Cb4] {
        let path = dir.join(format!("fmt-{}.safetensors", fmt.name()));
        let f = File::open(&path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let read = |name: &str| -> Result<Vec<u8>> { f.read(f.get(name).ok_or_else(|| Error(format!("{}: no tensor {name}", path.display())))?).map_err(|e| Error(format!("{name}: {e}"))) };
        let (codes, book, idx) = (read("codes")?, read("codebook")?, read("idx")?);
        let rows = idx.len() / 4096;
        let m = GlmMat { fmt, rows, cols: 4096, codes: &codes, book: &book };
        if !m.is_valid() {
            return err(format!("{}: {} rows of {} bytes and {} bytes of tables are not the {} layout", path.display(), rows, codes.len() / rows, book.len(), fmt.name()));
        }
        let k = fmt.entries();
        let weight = |row: usize, c: usize| -> f64 {
            let e = idx[row * 4096 + c] as usize * 2 + row * k * 2;
            bf16_to_f32(u16::from_le_bytes([book[e], book[e + 1]])) as f64
        };
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        };
        let z: Vec<f32> = (0..4096).map(|_| next()).collect();
        let act: Vec<f32> = (0..rows).map(|_| next()).collect();
        let mut zq = vec![0f32; m.z_len()];
        simd::tern_permute(&z, &mut zq);
        let zin: &[f32] = if fmt == Fmt::Tern { &zq } else { &z };
        let (mut worst_dot, mut worst_down) = (0f64, 0f64);
        let mut first: Option<(Vec<f32>, Vec<f32>)> = None;
        for kernel in Kernel::ALL.into_iter().filter(|k| k.runs_here()) {
            let mut sums = vec![0f32; rows];
            simd::dot_glm(kernel, &m, zin, 0, rows, &mut sums);
            let mut down = vec![0f32; 4096];
            let slice = fmt.slice();
            for o0 in (0..4096).step_by(slice) {
                let mut buf = [0f32; 256];
                simd::down_glm(kernel, &m, &act, o0, &mut buf);
                let n = slice.min(4096 - o0);
                down[o0..o0 + n].copy_from_slice(&buf[..n]);
            }
            for r in 0..rows {
                let truth: f64 = (0..4096).map(|c| weight(r, c) * z[c] as f64).sum();
                worst_dot = worst_dot.max((sums[r] as f64 - truth).abs() / (1.0 + truth.abs()));
            }
            for o in 0..4096 {
                let truth: f64 = (0..rows).map(|n| act[n] as f64 * weight(n, o)).sum();
                worst_down = worst_down.max((down[o] as f64 - truth).abs() / (1.0 + truth.abs()));
            }
            match &first {
                None => first = Some((sums, down)),
                Some((s0, d0)) => {
                    let same = sums.iter().zip(s0).all(|(a, b)| a.to_bits() == b.to_bits()) && down.iter().zip(d0).all(|(a, b)| a.to_bits() == b.to_bits());
                    if !same {
                        println!("FAIL {}: the {} kernel does not make the first kernel's bits", fmt.name(), kernel.name());
                        bad = true;
                    }
                }
            }
        }
        let ok = worst_dot < 1e-4 && worst_down < 1e-4;
        bad |= !ok;
        println!("{} {}: {} rows of {} bytes: the first product {:.1e} and the second {:.1e} off the f64 products of the weights the writer's codes name (bound 1e-4), kernels: {}", if ok { "ok  " } else { "FAIL" }, fmt.name(), rows, codes.len() / rows, worst_dot, worst_down, Kernel::ALL.into_iter().filter(|k| k.runs_here()).map(|k| k.name()).collect::<Vec<_>>().join(" "));
    }
    if bad {
        err("a format does not match the pack writer's")
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_experts::{bf16_to_f32, Kernel};

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn f(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (self.next() >> 40) as f32 / (1u64 << 24) as f32 * (hi - lo)
        }
    }

    /// A matrix of random codes (the indices into each row's table) and tables (bf16 between -`amp` and `amp`), packed the way glm_pack.py's `pack_rows` packs a format.
    struct Mat {
        fmt: Fmt,
        rows: usize,
        cols: usize,
        idx: Vec<u8>,
        bytes: Vec<u8>,
        book: Vec<u8>,
        table: Vec<f64>,
    }

    fn pack_row(codes: &[u8], fmt: Fmt) -> Vec<u8> {
        match fmt {
            Fmt::Cb2 => codes.chunks(4).map(|c| c[0] | c[1] << 2 | c[2] << 4 | c[3] << 6).collect(),
            Fmt::Cb4 => codes.chunks(2).map(|c| c[0] | c[1] << 4).collect(),
            Fmt::Cb3 => codes
                .chunks(8)
                .flat_map(|c| {
                    let w: u32 = (0..8).map(|m| (c[m] as u32) << (3 * m)).sum();
                    [w as u8, (w >> 8) as u8, (w >> 16) as u8]
                })
                .collect(),
            // (five codes a byte, the first the units; the last byte's missing codes are 0)
            Fmt::Tern => codes.chunks(5).map(|c| c.iter().rev().fold(0u32, |b, &x| b * 3 + x as u32) as u8).collect(),
            Fmt::Tq | Fmt::Tq3 | Fmt::Tq4 => unreachable!("trellis rows are not codebook codes"),
        }
    }

    fn matrix(fmt: Fmt, rows: usize, cols: usize, amp: f32, r: &mut Rng) -> Mat {
        let k = fmt.entries();
        let idx: Vec<u8> = (0..rows * cols).map(|_| (r.next() % k as u64) as u8).collect();
        let bytes: Vec<u8> = idx.chunks(cols).flat_map(|row| pack_row(row, fmt)).collect();
        let vals: Vec<u16> = (0..rows * k).map(|_| (r.f(-amp, amp).to_bits() >> 16) as u16).collect();
        let book: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let table = vals.iter().map(|&v| bf16_to_f32(v) as f64).collect();
        Mat { fmt, rows, cols, idx, bytes, book, table }
    }

    impl Mat {
        fn view(&self) -> GlmMat<'_> {
            GlmMat { fmt: self.fmt, rows: self.rows, cols: self.cols, codes: &self.bytes, book: &self.book }
        }

        /// The weight (row, column): the entry the code names.
        fn weight(&self, row: usize, c: usize) -> f64 {
            self.table[row * self.fmt.entries() + self.idx[row * self.cols + c] as usize]
        }
    }

    /// An independent computation of a pair in f64 from the codes (the activations rounded to bf16 where the kernels round them).
    fn reference(up: &Mat, down: &Mat, limit: f32, z: &[f32], weight_of_pair: f32) -> Vec<f64> {
        let (hidden, inter) = (down.cols, down.rows);
        let round = |x: f64| bf16_round(x as f32) as f64;
        let act: Vec<f64> = (0..inter)
            .map(|n| {
                let dot = |row: usize| (0..hidden).map(|c| up.weight(row, c) * z[c] as f64).sum::<f64>();
                let (g, u) = (dot(n).min(limit as f64), dot(inter + n).clamp(-limit as f64, limit as f64));
                round(g / (1.0 + (-g).exp()) * u)
            })
            .collect();
        (0..hidden).map(|o| (0..inter).map(|n| act[n] * down.weight(n, o)).sum::<f64>() * weight_of_pair as f64).collect()
    }

    /// The grouped kernel makes `experts_pinned`'s bits: experts of the codebook formats with one, four and seven tokens each, at one and at several threads.
    #[test]
    fn grouped_experts_make_the_pairwise_bits() {
        let mut r = Rng(0x2545_F491_4F6C_DD1D);
        let (hidden, inter) = (512usize, 128usize);
        for (fu, fd) in [(Fmt::Cb2, Fmt::Cb2), (Fmt::Cb3, Fmt::Cb4), (Fmt::Cb4, Fmt::Cb3), (Fmt::Cb2, Fmt::Cb4)] {
            let mats: Vec<(Mat, Mat)> = (0..3).map(|_| (matrix(fu, 2 * inter, hidden, 0.7, &mut r), matrix(fd, inter, hidden, 0.5, &mut r))).collect();
            let es: Vec<GlmExpert> = mats.iter().map(|(u, d)| GlmExpert::mixed(u.view(), d.view(), 10.0).unwrap()).collect();
            let z: Vec<f32> = (0..9 * hidden).map(|_| r.f(-2.0, 2.0)).collect();
            let pairs: Vec<GlmPair> = es.iter().enumerate().flat_map(|(i, e)| (0..1 + 3 * i).map(move |t| GlmPair { expert: e, token: (t * 2 + i) % 9, weight: 0.25 + t as f32 })).collect();
            let mut want = vec![0f32; pairs.len() * hidden];
            experts_pinned(Kernel::best(), Z::F32(&z), &pairs, &mut want, 1, false).unwrap();
            for threads in [1usize, 4] {
                let mut got = vec![0f32; pairs.len() * hidden];
                experts_grouped(Kernel::best(), Z::F32(&z), &pairs, &mut got, threads).unwrap();
                assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "{fu:?}+{fd:?} at {threads} threads");
            }
        }
    }

    /// Every kernel this computer runs makes the scalar kernel's bits, at any number of threads, and the scalar kernel's rows are the f64 computation's to the roundings of bf16: for every format of each matrix.
    #[test]
    fn experts_of_every_format_of_the_pack() {
        let mut r = Rng(0x9E37_79B9_7F4A_7C15);
        let (hidden, inter) = (512usize, 128usize);
        let mixes = [
            (Fmt::Cb2, Fmt::Cb2),
            (Fmt::Cb3, Fmt::Cb3),
            (Fmt::Cb4, Fmt::Cb4),
            (Fmt::Tern, Fmt::Tern),
            (Fmt::Cb2, Fmt::Cb3),
            (Fmt::Tern, Fmt::Cb4),
            (Fmt::Cb3, Fmt::Tern),
            (Fmt::Cb4, Fmt::Cb2),
        ];
        for (fu, fd) in mixes {
            let (up, down) = (matrix(fu, 2 * inter, hidden, 0.7, &mut r), matrix(fd, inter, hidden, 0.5, &mut r));
            let z: Vec<f32> = (0..2 * hidden).map(|_| r.f(-2.0, 2.0)).collect();
            let e = GlmExpert::mixed(up.view(), down.view(), 10.0).unwrap();
            let pairs: Vec<GlmPair> = (0..5).map(|i| GlmPair { expert: &e, token: i % 2, weight: 0.5 + i as f32 }).collect();
            let mut want = vec![0f32; pairs.len() * hidden];
            let nz = experts_pinned(Kernel::Scalar, Z::F32(&z), &pairs, &mut want, 1, false).unwrap();
            assert!(nz.iter().all(|&n| n > inter as u32 / 2), "{fu:?}+{fd:?}: {nz:?}");
            for (p, pair) in pairs.iter().enumerate() {
                let truth = reference(&up, &down, 10.0, &z[pair.token * hidden..(pair.token + 1) * hidden], pair.weight);
                let (mut e2, mut n2) = (0f64, 0f64);
                for o in 0..hidden {
                    e2 += (want[p * hidden + o] as f64 - truth[o]).powi(2);
                    n2 += truth[o] * truth[o];
                }
                assert!((e2 / n2).sqrt() < 4e-3, "{fu:?}+{fd:?} pair {p}: {:.2e} off the f64 computation", (e2 / n2).sqrt());
            }
            for kernel in Kernel::ALL.into_iter().filter(|k| k.runs_here()) {
                for threads in [1, 3, 8] {
                    let mut got = vec![0f32; want.len()];
                    experts_pinned(kernel, Z::F32(&z), &pairs, &mut got, threads, false).unwrap();
                    assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "{fu:?}+{fd:?}: the {} kernel on {threads} threads is not the scalar kernel's bits", kernel.name());
                }
            }
        }
    }

    /// Pairs of experts of different formats in one call.
    #[test]
    fn a_call_of_experts_of_different_formats() {
        let mut r = Rng(77);
        let (hidden, inter) = (256usize, 64usize);
        let ms: Vec<(Mat, Mat)> = [(Fmt::Cb2, Fmt::Cb2), (Fmt::Tern, Fmt::Cb3), (Fmt::Cb4, Fmt::Tern)].into_iter().map(|(a, b)| (matrix(a, 2 * inter, hidden, 0.7, &mut r), matrix(b, inter, hidden, 0.5, &mut r))).collect();
        let es: Vec<GlmExpert> = ms.iter().map(|(u, d)| GlmExpert::mixed(u.view(), d.view(), 10.0).unwrap()).collect();
        let z: Vec<f32> = (0..hidden).map(|_| r.f(-2.0, 2.0)).collect();
        let pairs: Vec<GlmPair> = es.iter().enumerate().map(|(i, e)| GlmPair { expert: e, token: 0, weight: 1.0 + i as f32 }).collect();
        let mut want = vec![0f32; pairs.len() * hidden];
        experts_pinned(Kernel::Scalar, Z::F32(&z), &pairs, &mut want, 1, false).unwrap();
        for (p, (u, d)) in ms.iter().enumerate() {
            let truth = reference(u, d, 10.0, &z, 1.0 + p as f32);
            let (mut e2, mut n2) = (0f64, 0f64);
            (0..hidden).for_each(|o| {
                e2 += (want[p * hidden + o] as f64 - truth[o]).powi(2);
                n2 += truth[o] * truth[o];
            });
            assert!((e2 / n2).sqrt() < 4e-3, "pair {p}: {:.2e}", (e2 / n2).sqrt());
        }
        for kernel in Kernel::ALL.into_iter().filter(|k| k.runs_here()) {
            let mut got = vec![0f32; want.len()];
            experts_pinned(kernel, Z::F32(&z), &pairs, &mut got, 4, false).unwrap();
            assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "the {} kernel", kernel.name());
        }
    }

    /// The pack's real width: 4,096 values a row, a ternary row's last group of 160 holding 96 values (820 bytes a row).
    #[test]
    fn rows_of_the_packs_width() {
        let mut r = Rng(31);
        for fmt in [Fmt::Tern, Fmt::Cb2, Fmt::Cb3, Fmt::Cb4] {
            let m = matrix(fmt, 8, 4096, 0.3, &mut r);
            assert_eq!(m.bytes.len(), 8 * [820, 1024, 1536, 2048][[Fmt::Tern, Fmt::Cb2, Fmt::Cb3, Fmt::Cb4].iter().position(|&f| f == fmt).unwrap()]);
            let z: Vec<f32> = (0..4096).map(|_| r.f(-1.0, 1.0)).collect();
            let mut zq = vec![0f32; 160 * simd::tern_groups(4096)];
            simd::tern_permute(&z, &mut zq);
            let zin: &[f32] = if fmt == Fmt::Tern { &zq } else { &z };
            let mut want = vec![0f32; 8];
            simd::dot_glm(Kernel::Scalar, &m.view(), zin, 0, 8, &mut want);
            for row in 0..8 {
                let truth: f64 = (0..4096).map(|c| m.weight(row, c) * z[c] as f64).sum();
                assert!((want[row] as f64 - truth).abs() < 1e-3 * (1.0 + truth.abs()), "{fmt:?} row {row}: {} against {truth}", want[row]);
            }
            for kernel in Kernel::ALL.into_iter().filter(|k| k.runs_here()) {
                let mut got = vec![0f32; 8];
                simd::dot_glm(kernel, &m.view(), zin, 0, 8, &mut got);
                assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "{fmt:?}: the {} kernel", kernel.name());
            }
            // the second product over the 8 rows as neurons: every slice
            let act: Vec<f32> = (0..8).map(|_| r.f(-1.0, 1.0)).collect();
            let slice = fmt.slice();
            for o0 in (0..4096).step_by(slice) {
                let mut want = [0f32; 256];
                simd::down_glm(Kernel::Scalar, &m.view(), &act, o0, &mut want);
                for (j, &v) in want.iter().enumerate().take(slice.min(4096 - o0)) {
                    let truth: f64 = (0..8).map(|n| act[n] as f64 * m.weight(n, o0 + j)).sum();
                    assert!((v as f64 - truth).abs() < 1e-4 * (1.0 + truth.abs()), "{fmt:?} output {}: {v} against {truth}", o0 + j);
                }
                for kernel in Kernel::ALL.into_iter().filter(|k| k.runs_here()) {
                    let mut got = [0f32; 256];
                    simd::down_glm(kernel, &m.view(), &act, o0, &mut got);
                    assert!(got.iter().zip(&want).take(slice.min(4096 - o0)).all(|(a, b)| a.to_bits() == b.to_bits()), "{fmt:?} outputs {o0}: the {} kernel", kernel.name());
                }
            }
        }
    }

    #[test]
    fn ternary_digits_and_the_order_of_z() {
        for b in 0..243usize {
            let want = [b % 3, b / 3 % 3, b / 9 % 3, b / 27 % 3, b / 81 % 3];
            let got: Vec<usize> = (0..5).map(|d| simd::tern_digit(b as u8, d)).collect();
            assert_eq!(got, want, "byte {b}");
        }
        let z: Vec<f32> = (0..170).map(|i| i as f32).collect();
        let mut q = vec![-1f32; 320];
        simd::tern_permute(&z, &mut q);
        assert_eq!(q[32 * 2 + 7], z[5 * 7 + 2]);
        assert_eq!(q[160 + 32 * 2 + 1], z[160 + 5 * 1 + 2]);
        // (past the vector's end: zeros)
        assert_eq!(q[160 + 2], 0.0);
        assert_eq!(q[160 + 32 * 4 + 31], 0.0);
    }

    #[test]
    fn shapes_the_kernels_do_not_take_are_refused() {
        let mut r = Rng(5);
        let (up, down) = (matrix(Fmt::Cb2, 128, 256, 0.5, &mut r), matrix(Fmt::Cb2, 64, 256, 0.5, &mut r));
        assert!(GlmExpert::mixed(up.view(), down.view(), 10.0).is_ok());
        assert!(GlmExpert::mixed(up.view(), down.view(), 0.0).is_err());
        let short = matrix(Fmt::Cb2, 32, 256, 0.5, &mut r);
        assert!(GlmExpert::mixed(up.view(), short.view(), 10.0).is_err());
        let mut cut = up.view();
        cut.codes = &up.bytes[1..];
        assert!(GlmExpert::mixed(cut, down.view(), 10.0).is_err());
        // (three bits and the views of src/cpu_hb.rs)
        let (u3, d3) = (matrix(Fmt::Cb3, 128, 256, 0.5, &mut r), matrix(Fmt::Cb3, 64, 256, 0.5, &mut r));
        fn v(m: &Mat) -> CbView<'_> {
            CbView { bits: 3, rows: m.rows, cols: m.cols, codes: &m.bytes, scales: &[], codebook: &m.book }
        }
        assert!(GlmExpert::new(v(&u3), v(&d3), 10.0).is_ok());
    }
}

/// A development trace's record of one work item, made when the item ends (`GLYD_HOST_TRACE`).
#[cfg(feature = "dev")]
struct Span<'a>(bool, usize, f64, &'a std::time::Instant, &'a Mutex<Vec<(usize, f64, f64, u64)>>);

#[cfg(feature = "dev")]
impl Drop for Span<'_> {
    fn drop(&mut self) {
        if self.0 {
            let end = self.3.elapsed().as_secs_f64() * 1e6;
            // (the thread as the system numbers its core: where the item ran)
            #[cfg(unix)]
            let cpu = unsafe { sched_getcpu() } as u64;
            #[cfg(windows)]
            let cpu = unsafe { crate::os::win::GetCurrentProcessorNumber() } as u64;
            self.4.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push((self.1, self.2, end, cpu));
        }
    }
}

#[cfg(all(feature = "dev", unix))]
extern "C" {
    fn sched_getcpu() -> i32;
}

/// The pairs' output rows as the second-product items write them: each item its own slice (`OutRows::slice`).
struct OutRows(*mut f32, usize);

// SAFETY: the items that share it write disjoint slices of the buffer it points to (see `slice`), which outlives the crew's job.
unsafe impl Sync for OutRows {}

impl OutRows {
    /// `len` outputs from `at`.
    ///
    /// # Safety
    /// No other item writes or reads them while the slice lives; `at + len` is within the buffer.
    #[allow(clippy::mut_from_ref)]
    unsafe fn slice(&self, at: usize, len: usize) -> &mut [f32] {
        assert!(at + len <= self.1, "outputs {at} .. {} of {}", at + len, self.1);
        std::slice::from_raw_parts_mut(self.0.add(at), len)
    }
}

#[cfg(test)]
mod tq_tests {
    use super::*;

    /// Pairs of one trellis expert computed together (its runs decoded once for them) make each pair's bits as computed alone, with every kernel this computer runs.
    #[test]
    fn trellis_pairs_together_are_alone() {
        let (hidden, inter) = (256, 64);
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for fmt in [Fmt::Tq, Fmt::Tq3, Fmt::Tq4] {
            let mut mat = |rows: usize| -> (Vec<u8>, Vec<u8>) {
                let codes = (0..rows * fmt.row_bytes(hidden)).map(|_| next() as u8).collect();
                let book = (0..rows).flat_map(|r| (((0.02f32 + r as f32 * 1e-4).to_bits() >> 16) as u16).to_le_bytes()).collect();
                (codes, book)
            };
            let ((uc, ub), (dc, db)) = (mat(2 * inter), mat(inter));
            let e = GlmExpert::mixed(GlmMat { fmt, rows: 2 * inter, cols: hidden, codes: &uc, book: &ub }, GlmMat { fmt, rows: inter, cols: hidden, codes: &dc, book: &db }, f32::MAX).unwrap();
            let z: Vec<f32> = (0..3 * hidden).map(|_| (next() % 2001) as f32 / 500.0 - 2.0).collect();
            let weights = [0.5f32, 0.25, 0.125];
            for kernel in Kernel::ALL.into_iter().filter(|k| k.runs_here()) {
                let pairs: Vec<GlmPair> = (0..3).map(|t| GlmPair { expert: &e, token: t, weight: weights[t] }).collect();
                let mut together = vec![0f32; 3 * hidden];
                experts_pinned(kernel, Z::F32(&z), &pairs, &mut together, 2, false).unwrap();
                for (t, pair) in pairs.iter().enumerate() {
                    let mut alone = vec![0f32; hidden];
                    experts_pinned(kernel, Z::F32(&z), std::slice::from_ref(pair), &mut alone, 1, false).unwrap();
                    let same = alone.iter().zip(&together[t * hidden..(t + 1) * hidden]).all(|(a, b)| a.to_bits() == b.to_bits());
                    assert!(same, "{} ({}): pair {t} together differs from alone", kernel.name(), fmt.name());
                }
            }
        }
    }
}
