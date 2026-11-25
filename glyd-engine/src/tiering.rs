//! Expert tiering for every mixture of experts on the dynamic route (R12, #232; docs/design/runtime.md §6): a layer's experts in the expert cache (src/offload.rs `ExpertStore`), places for the busiest of them on the GPU and
//! every one in the computer's memory, for the formats whose experts are each the same bytes (`moe::ExpertStack::pooled`: the block layouts, the FP4 layouts as DeepSeek-V4 and Kimi-K3 release them). `MoeLayer::with_cache`
//! runs the layer over the places; a family's loader builds the stacks and the cache here (`cached_stacks`; FP8 experts as released, GLM-5.3-Flash's, each packed on its own: `cached_f8`; a lossless stack in the tiered layout,
//! Muse Spark's, its one pack cut per expert: `cached_tiered`). Hummingbird's two bit widths a layer are src/hbg_store.rs's.

use crate::f8::{F8Experts, F8Places};
use crate::ffi::{err, Result};
use crate::moe::{ExpertCache, ExpertStack, Joined};
use crate::offload::{ExpertStore, HostHalf, PackIn};
use std::rc::Rc;

/// A layer's experts (`up` and `down`, each every expert's pack end to end, as `Joined` makes them) in the expert cache `store` as its layer `layer`: `slots` places of each matrix on the GPU (`ExpertStack::pooled`), every
/// expert's bytes in pinned memory, the host computing from them where `cpu` says how (else the misses are copied). Returns the two stacks (the layer's `MoeWeights`) and the cache (`MoeLayer::with_cache`) for the GPU `ordinal`.
pub fn cached_stacks(
    up: &Joined,
    down: &Joined,
    slots: usize,
    store: &Rc<ExpertStore>,
    layer: usize,
    ordinal: i32,
    cpu: Option<HostHalf>,
) -> Result<(ExpertStack, ExpertStack, ExpertCache)> {
    if up.experts != down.experts {
        return err(format!("{} experts of the first matrix and {} of the second", up.experts, down.experts));
    }
    let slots = slots.clamp(1, up.experts);
    let (u, d) = (
        ExpertStack::pooled(up.tag, up.tiers, up.experts, slots, up.rows, up.cols)?,
        ExpertStack::pooled(down.tag, down.tiers, down.experts, slots, down.rows, down.cols)?,
    );
    for (j, s, what) in [(up, &u, "first"), (down, &d, "second")] {
        if j.data.len() as u64 != j.experts as u64 * s.data_stride || j.scales.len() as u64 != j.experts as u64 * s.scale_stride {
            return err(format!(
                "the {what} matrices' {} + {} bytes, where {} experts at the places' strides take {} + {}",
                j.data.len(),
                j.scales.len(),
                j.experts,
                j.experts as u64 * s.data_stride,
                j.experts as u64 * s.scale_stride
            ));
        }
    }
    store.add_layer(
        layer,
        up.experts,
        slots,
        &[
            PackIn { bytes: [&up.data, &up.scales], stride: [u.data_stride, u.scale_stride], dev: u.buffers() },
            PackIn { bytes: [&down.data, &down.scales], stride: [d.data_stride, d.scale_stride], dev: d.buffers() },
        ],
        cpu,
    )?;
    Ok((u, d, ExpertCache { store: store.clone(), layer, ordinal }))
}

/// A layer's FP8 experts as released (`up` and `down`, `F8Places::pack`'s: each expert packed on its own) in the expert cache `store` as its layer `layer`: `slots` places of each matrix on the GPU (`F8Places::places`), every
/// expert's bytes in pinned memory, copied only (no host computes FP8 experts). Returns the layer's experts (`MoeLayer::new_f8`, over the places) and the cache (`MoeLayer::with_cache`) for the GPU `ordinal`.
pub fn cached_f8(up: &F8Places, down: &F8Places, slots: usize, store: &Rc<ExpertStore>, layer: usize, ordinal: i32) -> Result<(F8Experts, ExpertCache)> {
    if up.experts != down.experts {
        return err(format!("{} FP8 experts of the first matrix and {} of the second", up.experts, down.experts));
    }
    let slots = slots.clamp(1, up.experts);
    let (u, d) = (up.places(slots)?, down.places(slots)?);
    let ([a, b], [c, e]) = (up.packs(&u), down.packs(&d));
    store.add_layer(layer, up.experts, slots, &[a, b, c, e], None)?;
    Ok((F8Experts { up: u, down: d, experts: up.experts }, ExpertCache { store: store.clone(), layer, ordinal }))
}

/// A stack of experts in the tiered layout (penguin's lossless bf16: Muse Spark's experts) for the expert cache: the stack's one pack (its tiers every expert's, as a product call takes one set) cut per expert. An expert's data is
/// its steps' (the same bytes a step); its blocks are the pack's bytes from 128 before the word its first block starts in to 256 after its last block (what the decode reads around them), padded with zeros to `stride`, so each
/// block keeps its word alignment, which the packer's flags depend on; its block entries are its steps' and the end, offsets from its own blocks' start, flags kept. glyd_gpu_mma_moe_placed multiplies places of such experts
/// (`ExpertStack::placed`).
pub struct TieredPlaces {
    pub tiers: [u32; 3],
    pub experts: usize,
    pub rows: usize,
    pub cols: usize,
    pub steps: usize,
    pub stride: usize,
    pub data: Vec<u8>,
    pub blocks: Vec<u8>,
    pub entries: Vec<u8>,
}

impl TieredPlaces {
    /// The pack `p` of `experts` experts' matrices stacked ([experts rows, cols]) cut per expert.
    pub fn of(p: &glyd_gpu::pack::Tiered, experts: usize) -> Result<TieredPlaces> {
        use glyd_gpu::pack::block_start;
        if experts == 0 || !p.rows.is_multiple_of(experts) || !(p.rows / experts).is_multiple_of(64) || !p.cols.is_multiple_of(16) || p.tiers[0] >> 24 != 2 {
            return err(format!("a tiered pack of {} x {} (layout {}) as {experts} experts' places", p.rows, p.cols, p.tiers[0] >> 24));
        }
        let (rows, cols) = (p.rows / experts, p.cols);
        let steps = rows / 64 * (cols / 16);
        if p.data.len() != experts * steps * 1280 || p.block_base.len() != experts * steps + 1 {
            return err(format!(
                "a tiered pack of {} bytes of steps and {} block entries, for {experts} experts of {steps} steps",
                p.data.len(),
                p.block_base.len()
            ));
        }
        let mut spans = Vec::with_capacity(experts);
        for e in 0..experts {
            let (b0, b1) = (block_start(p.block_base[e * steps]), block_start(p.block_base[(e + 1) * steps]));
            let lo = (b0 & !3).checked_sub(128);
            match lo {
                Some(lo) if b0 <= b1 && b1 + 256 <= p.blocks.len() => spans.push((lo, b1 + 256)),
                _ => {
                    return err(format!(
                        "expert {e}'s blocks {b0}..{b1} of a pack's {} bytes: not 128 bytes after the start and 256 before the end",
                        p.blocks.len()
                    ))
                }
            }
        }
        let stride = spans.iter().map(|&(lo, hi)| hi - lo).max().unwrap_or(0).next_multiple_of(128);
        let (mut blocks, mut entries) = (vec![0u8; experts * stride], Vec::with_capacity(experts * (steps + 1) * 4));
        for (e, &(lo, hi)) in spans.iter().enumerate() {
            blocks[e * stride..][..hi - lo].copy_from_slice(&p.blocks[lo..hi]);
            // (the last entry is the end alone: in the stack it is the next expert's first step's too, and its flag that step's)
            for (j, &v) in p.block_base[e * steps..=(e + 1) * steps].iter().enumerate() {
                let flag = if j < steps { v & i32::MIN } else { 0 };
                entries.extend(((block_start(v) - lo) as i32 | flag).to_le_bytes());
            }
        }
        Ok(TieredPlaces { tiers: p.tiers, experts, rows, cols, steps, stride, data: p.data.clone(), blocks, entries })
    }

    /// An expert's bytes of steps, of blocks, of block entries.
    pub fn strides(&self) -> [u64; 3] {
        [(self.steps * 1280) as u64, self.stride as u64, ((self.steps + 1) * 4) as u64]
    }

    /// Expert `e` as a pack of its own (its data, its blocks, its entries): what a place holds once the store has copied it in.
    pub fn expert(&self, e: usize) -> glyd_gpu::pack::Tiered {
        let [d, b, n] = self.strides().map(|x| x as usize);
        glyd_gpu::pack::Tiered {
            rows: self.rows,
            cols: self.cols,
            data: self.data[e * d..][..d].to_vec(),
            blocks: self.blocks[e * b..][..b].to_vec(),
            block_base: self.entries[e * n..][..n].chunks(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            tiers: self.tiers,
        }
    }
}

/// A layer's experts in the tiered layout (`up` and `down`, `TieredPlaces::of`'s) in the expert cache `store` as its layer `layer`: `slots` places of each matrix on the GPU (`ExpertStack::placed`), every expert's bytes in
/// pinned memory, copied only (no host computes them). Returns the two stacks (the layer's `MoeWeights`) and the cache (`MoeLayer::with_cache`) for the GPU `ordinal`.
pub fn cached_tiered(
    up: &TieredPlaces,
    down: &TieredPlaces,
    slots: usize,
    store: &Rc<ExpertStore>,
    layer: usize,
    ordinal: i32,
) -> Result<(ExpertStack, ExpertStack, ExpertCache)> {
    if up.experts != down.experts {
        return err(format!("{} tiered experts of the first matrix and {} of the second", up.experts, down.experts));
    }
    let slots = slots.clamp(1, up.experts);
    let (u, d) = (
        ExpertStack::placed(up.tiers, up.experts, slots, up.rows, up.cols, up.stride as u64)?,
        ExpertStack::placed(down.tiers, down.experts, slots, down.rows, down.cols, down.stride as u64)?,
    );
    let entries = |s: &ExpertStack| match s.pack.words {
        crate::ffi::Words::Tiered { block_base, .. } => block_base,
        crate::ffi::Words::Twelve { .. } => 0,
    };
    let (su, sd) = (up.strides(), down.strides());
    store.add_layer(
        layer,
        up.experts,
        slots,
        &[
            PackIn { bytes: [&up.data, &up.blocks], stride: [su[0], su[1]], dev: u.buffers() },
            PackIn { bytes: [&down.data, &down.blocks], stride: [sd[0], sd[1]], dev: d.buffers() },
            PackIn { bytes: [&up.entries, &down.entries], stride: [su[2], sd[2]], dev: [entries(&u), entries(&d)] },
        ],
        None,
    )?;
    Ok((u, d, ExpertCache { store: store.clone(), layer, ordinal }))
}

/// `glyd-engine tiercheck MODEL [--places N] [--tokens N] [--mode copy|cpu|auto] [--device-plan] [--against copy] [--layout i6|i5|i4] [--dense N] [--report]`: the model of the dynamic route (a release whose experts the cache takes, or a hummingbird
/// directory) on GPU 0 with every expert there, then with its experts in the expert cache (`--places` a layer, every expert in the computer's memory; `--mode` the misses', copy by default): the same prompt and decode
/// steps, and every position's logits compared; the positions that differ are named.
///
/// In copy mode the layers' products run over the places with the kernels of the resident layer, and the logits are the same bits where the cache's calls are the resident pass's: a call computed in waves (DeepSeek-V4's
/// release experts, GLM-5.3-Flash's at a block level, Kimi-K3's), or a prompt of a few tokens (`--tokens 6`) for the caches that run a call of as many tokens as their places hold (GLM-5.3-Flash's FP8 experts, Muse
/// Spark's). A longer prompt through those two differs in its last bits: the library splits a product's K over blocks by the size of the call, and the sums are added in another order (on an RTX 4080, 2026-11-22, the
/// random models' largest differences: 0.031 and 0.051 of a logit). A hummingbird directory's cache is the resident layer's bits where every pair of a call has a place (`--places 15 --tokens 6` on the random
/// DeepSeek-V4: none differs); the pairs whose experts find none are the host's in copy mode too (`ExpertStore::run_groups_at`), and differ as `--mode cpu`'s do (up to 2.7 of a logit there, with 2 places).
/// `--report` prints such differences and does not refuse them; without it a difference in copy mode fails the check. With the host computing some pairs the largest difference is printed, and each
/// position's cosine of the two rows of logits about their means. `--against copy`: the reference is the cache in copy mode (the resident layer's bits), for a model larger than the GPU.
#[cfg(feature = "dev")]
pub fn check(a: &crate::args::Args) -> Result<()> {
    use crate::ffi::Error;
    let dir = std::path::Path::new(a.need(1, "MODEL")?);
    let places = a.num("places", 2usize)?;
    // (`--dense N`: the last N layers' dense matrices in the computer's memory too, src/dense_ram.rs)
    let dense = a.num("dense", 0usize)?;
    let report = a.has("report");
    let mode = match a.flag("mode") {
        Some(m) => crate::offload::Mode::parse(m).ok_or_else(|| Error(format!("--mode {m}: copy, cpu, split or auto")))?,
        None => crate::offload::Mode::Copy,
    };
    let text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| Error(format!("{}: {e}", dir.display())))?;
    let vocab = crate::dynamic::split::config_num(&text, "vocab_size").unwrap_or(1000).max(16);
    let n = a.num("tokens", 24usize)?.max(6);
    let tokens: Vec<u32> = (0..n).map(|i| ((i * 7919 + 13) % (vocab - 4) + 2) as u32).collect();
    // (`--device-plan`: the decode calls planned on the GPU, src/devplan.rs, and the decode step captured in a graph, which it must be)
    let device = a.has("device-plan");
    // (`--layout`: the level a family's loader packs at, GLM's experts at a block level among them)
    let o = crate::dynamic::Opts { ctx: n + 64, chunk: 256, graph: device, device: 0, layout: a.layout()? };
    let (graphed, planned) = (std::cell::Cell::new(false), std::cell::Cell::new(0usize));
    // (`--against copy`: the reference the same cache in copy mode with the host's plan, the resident layer's bits (above), for a model larger than the GPU)
    let against_copy = match a.flag("against") {
        None => false,
        Some("copy") => true,
        Some(x) => return err(format!("--against {x}: copy")),
    };
    // (the router's choices of the run's passes, an eager run's: each selection read back, `moe::ROUTES`, from the prompt on)
    let tap =
        |on: Option<Vec<crate::moe::Seen>>| crate::moe::ROUTES.lock().map(|mut r| std::mem::replace(&mut *r, on)).map_err(|_| Error("the routes' lock".into()));
    type Routes = Option<Vec<crate::moe::Seen>>;
    // (`--pairs`: in the cached run each layer's pairs computed by the host and by the GPU from the same inputs and compared, `moe::PAIRS`; a mode with a host half)
    let pairs = a.has("pairs");
    if pairs && mode == crate::offload::Mode::Copy {
        return err("--pairs: a mode whose layers have a host half (cpu, split or auto)");
    }
    let run = |cached: bool| -> Result<(Vec<Vec<f32>>, Routes)> {
        let caching =
            |mode, device| crate::hbg_store::HbCaching { places, ram: None, offload: crate::offload::Offload { mode, device, ..Default::default() }, dense };
        let mut e = if cached {
            crate::dynamic::split::open_cached(dir, &o, &[], &[0], vec![caching(mode, device)])?
        } else if against_copy {
            crate::dynamic::split::open_cached(dir, &o, &[], &[0], vec![caching(crate::offload::Mode::Copy, false)])?
        } else {
            crate::dynamic::split::open(dir, &o, &[], &[0])?
        };
        if cached {
            graphed.set(matches!(&e, crate::dynamic::split::Run::One(d) if d.has_graph()));
            if let crate::dynamic::split::Run::One(d) = &e {
                planned.set(d.stores().iter().map(|s| s.device_layers()).sum());
            }
        }
        tap((!device).then(Vec::new))?;
        if cached && pairs {
            *crate::moe::PAIRS.lock().map_err(|_| Error("the pairs' lock".into()))? = Some(Default::default());
        }
        let passes = (|| -> Result<Vec<Vec<f32>>> {
            let mut out = Vec::new();
            e.prefill(&tokens[..n - 4])?;
            out.push(e.logits()?);
            for &t in &tokens[n - 4..] {
                e.step(t)?;
                out.push(e.logits()?);
            }
            Ok(out)
        })();
        let seen = tap(None)?;
        Ok((passes?, seen))
    };
    let ((resident, routes_r), (cached, routes_c)) = (run(false)?, run(true)?);
    // (each position's pass's tokens routed to other experts in some layer: the prompt's pass is the first selections' tokens, each decode step's one token; None: not compared)
    let passes = resident.len();
    // (each pass's tokens routed to other experts in some layer; at each such choice the reference's margin between the experts that changed places (the key of the one it chose
    // less the key of the one the cache's run chose, `moe::Seen::keys`) and how far that gap moved; and the most any expert's key moved where the choices stayed)
    let mut flips: Vec<(usize, usize, usize, f32, f32, f32)> = Vec::new();
    let mut moved_same = 0f32;
    let routing: Option<Vec<(usize, usize)>> = match (&routes_r, &routes_c) {
        (Some(a), Some(b)) if a.len() == b.len() && !a.is_empty() && a.len() % passes == 0 => {
            let per = a.len() / passes;
            Some(
                (0..passes)
                    .map(|i| {
                        let mut other = std::collections::BTreeSet::new();
                        let mut tokens = 0;
                        for j in i * per..(i + 1) * per {
                            let (x, y) = (&a[j], &b[j]);
                            let (k, e) = (x.k, x.keys.len() / (x.ids.len() / x.k).max(1));
                            tokens = x.ids.len() / k;
                            for tk in 0..tokens {
                                let (mut u, mut v) = (x.ids[tk * k..(tk + 1) * k].to_vec(), y.ids[tk * k..(tk + 1) * k].to_vec());
                                (u.sort_unstable(), v.sort_unstable());
                                let (kr, kc) = (&x.keys[tk * e..(tk + 1) * e], &y.keys[tk * e..(tk + 1) * e]);
                                let moved = kr.iter().zip(kc).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
                                if u == v {
                                    moved_same = moved_same.max(moved);
                                    continue;
                                }
                                other.insert(tk);
                                // (the closest of the swaps: an expert the reference chose and the cache's run did not, against one the other way)
                                let (mut margin, mut shift) = (f32::INFINITY, 0f32);
                                for &p in u.iter().filter(|p| !v.contains(p)) {
                                    for &q in v.iter().filter(|q| !u.contains(q)) {
                                        let (p, q) = (p as usize, q as usize);
                                        let m = kr[p] - kr[q];
                                        if m.abs() < margin.abs() {
                                            (margin, shift) = (m, m - (kc[p] - kc[q]));
                                        }
                                    }
                                }
                                let spread = {
                                    let mean = kr.iter().sum::<f32>() / e as f32;
                                    (kr.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / e as f32).sqrt()
                                };
                                flips.push((i, j - i * per, tk, margin, shift, spread));
                            }
                        }
                        (other.len(), tokens)
                    })
                    .collect(),
            )
        }
        _ => None,
    };
    let (mut differ, mut worst, mut least_cos) = (0usize, 0f32, 1f64);
    for (i, (r, c)) in resident.iter().zip(&cached).enumerate() {
        let (mut d, mut w) = (0usize, 0f32);
        for (x, y) in r.iter().zip(c) {
            if x.to_bits() != y.to_bits() {
                d += 1;
                w = w.max((x - y).abs());
            }
        }
        // (the two rows' cosine about their means and the reference's spread: a difference of rounding leaves the cosine near 1, a product of another expert's or of no expert does not)
        let n = r.len().max(1) as f64;
        let (mr, mc) = (r.iter().map(|&x| x as f64).sum::<f64>() / n, c.iter().map(|&x| x as f64).sum::<f64>() / n);
        let (mut rc, mut rr, mut cc) = (0f64, 0f64, 0f64);
        for (&x, &y) in r.iter().zip(c) {
            let (x, y) = (x as f64 - mr, y as f64 - mc);
            (rc, rr, cc) = (rc + x * y, rr + x * x, cc + y * y);
        }
        let cos = if rr > 0.0 && cc > 0.0 { rc / (rr * cc).sqrt() } else { 1.0 };
        least_cos = least_cos.min(cos);
        if d > 0 {
            // (which pass it was: the prompt's last row, or a decode step's one row; a step of one token has few pairs, where the library splits a product's K over blocks)
            let route = match routing.as_ref().map(|r| r[i]) {
                Some((0, _)) => ", the routing the same".to_string(),
                Some((o, t)) => format!(", the routing differs at {o} of the pass's {t} tokens"),
                None => String::new(),
            };
            println!(
                "  position {i} ({}): {d} of {} logits differ, the largest by {w:e} (the reference's spread {:.3}; cosine {cos:.6}{route})",
                if i == 0 { "the prompt's" } else { "a decode step's" },
                r.len(),
                (rr / n).sqrt()
            );
        }
        (differ, worst) = (differ + d, worst.max(w));
    }
    let total: usize = resident.iter().map(Vec::len).sum();
    for (i, sel, tk, margin, shift, spread) in &flips {
        println!(
            "  a choice changed: pass {i}, selection {sel}, token {tk}: the reference's margin {margin:.3e}, the gap moved {shift:.3e} (the keys' spread {spread:.3e}, {:.2e} of it)",
            margin.abs() / spread.max(1e-30)
        );
    }
    if routing.is_some() {
        println!("  where the choices stayed, an expert's key moved {moved_same:.3e} at most");
    }
    match &routing {
        Some(r) => println!(
            "  the routing differs at {} of {passes} positions' passes ({} tokens in all)",
            r.iter().filter(|x| x.0 > 0).count(),
            r.iter().map(|x| x.0).sum::<usize>()
        ),
        None if !device => println!("  the routing not compared: the two runs' selections differ in number"),
        None => println!("  the routing not compared: the decode step is one graph, whose selections are not read back"),
    }
    println!(
        "{}: {} logits of {} positions, {} against the expert cache ({places} places a layer, {mode:?}{}): {differ} differ, the largest by {worst:e}, the least cosine {least_cos:.6}",
        dir.display(),
        total,
        resident.len(),
        if against_copy { "the cache in copy mode" } else { "resident" },
        if device { ", the decode calls planned on the GPU" } else { "" }
    );
    if pairs {
        let all = crate::moe::PAIRS.lock().map_err(|_| Error("the pairs' lock".into()))?.take().unwrap_or_default();
        if all.is_empty() {
            println!("  pairs: none compared (no layer's calls through the store with a host half)");
        }
        for (layer, p) in &all {
            println!(
                "  pairs of layer {layer}, the host's against the GPU's from the same inputs: {} pairs of {} calls, {} values; the most {} last places apart; {} values more than 1, {} more than 4, {} more than 16; the largest difference {:.2e} of the pair's largest value; {} pairs more than 16 apart somewhere{}",
                p.pairs,
                p.calls,
                p.values,
                p.most,
                p.over[0],
                p.over[1],
                p.over[2],
                p.worst,
                p.far,
                p.first.map_or(String::new(), |(c, t, j, e, u)| format!(", the first: call {c}, token {t}, choice {j}, expert {e}, {u} last places"))
            );
        }
    }
    if device {
        // (the plan on the GPU takes layers whose host half is FP4 experts; another family's layers are the host's plan, said and not failed)
        if planned.get() == 0 {
            println!("  no layer's calls are planned on the GPU (its experts' host half is not FP4 experts): the host plans them");
        } else {
            println!(
                "  the decode calls planned on the GPU ({} layers): the decode step {} one graph",
                planned.get(),
                if graphed.get() { "is" } else { "is not" }
            );
        }
        if planned.get() > 0 && !graphed.get() {
            return err("--device-plan: the cached model's decode step was not captured in a graph");
        }
    }
    // (copy mode with the plan on the GPU: a decode call whose experts the places cannot all take sends the rest to the host, which the plan on the host would take in waves)
    if mode == crate::offload::Mode::Copy && differ > 0 && !report && !device {
        return err(format!("{differ} logits differ in copy mode, where the products over the places are the resident layer's"));
    }
    Ok(())
}

/// `glyd-engine storebench MODEL [--places N] [--rows 1,8,32] [--steps 8] [--warm 3] [--prompt 16] [--mode copy|cpu|split|auto] [--device-plan] [--plan] [--out FILE]` (development): batched decode through the expert cache on GPU 0
/// (the dynamic route's model, `--places` of a layer's experts on the GPU and every expert in the computer's memory, as `tiercheck` opens it): for each count of rows, that many sequences of `--prompt` tokens each,
/// `--warm` decode steps of all of them together, then `--steps` timed steps, each step's layer statistics (`ModelStep::expert_stats`). Prints a line for each count: the step's time, the experts asked for, found,
/// copied and computed by the host a step, the bytes read from the computer's memory for the misses against the distinct missed experts' bytes, and the rates the copies and the host ran at. `--out`: a JSON line
/// for each count (the rows, the mean step, an expert's bytes, each layer's statistics a step on average) for `ceiling --stats`.
#[cfg(feature = "dev")]
pub fn bench(a: &crate::args::Args) -> Result<()> {
    use crate::ffi::Error;
    use crate::offload::LayerStats;
    use crate::sched::{ModelStep, Row};
    let dir = std::path::Path::new(a.need(1, "MODEL")?);
    let places = a.num("places", 16usize)?;
    let rows_l: Vec<usize> = a
        .flag("rows")
        .unwrap_or("1,8,32")
        .split(',')
        .map(|x| x.trim().parse::<usize>().map_err(|_| Error(format!("--rows {x}: a number"))))
        .collect::<Result<_>>()?;
    let (steps, warm, plen) = (a.num("steps", 8usize)?.max(1), a.num("warm", 3usize)?, a.num("prompt", 16usize)?.max(1));
    let mode = a.flag("mode").map(|m| crate::offload::Mode::parse(m).ok_or_else(|| Error(format!("--mode {m}: copy, cpu, split or auto")))).transpose()?;
    let most = rows_l.iter().copied().max().unwrap_or(1);
    let text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| Error(format!("{}: {e}", dir.display())))?;
    let vocab = crate::dynamic::split::config_num(&text, "vocab_size").unwrap_or(1000).max(16);
    let o = crate::dynamic::Opts { ctx: plen + warm + steps + 64, chunk: plen.max(64), graph: false, device: 0, layout: a.layout()? };
    // (`--device-plan`: the decode calls planned on the GPU, src/devplan.rs)
    let offload = crate::offload::Offload { mode: mode.unwrap_or(crate::offload::Mode::Auto), device: a.has("device-plan"), ..Default::default() };
    let caching = crate::hbg_store::HbCaching { places, ram: None, offload, dense: 0 };
    let crate::dynamic::split::Run::One(mut e) = crate::dynamic::split::open_cached(dir, &o, &[], &[0], vec![caching])? else {
        return err("the model on one GPU was asked for");
    };
    let stores = e.stores();
    if stores.is_empty() {
        return err(format!("{}: no layer's experts are in the expert cache ({places} places a layer)", dir.display()));
    }
    if let Some(m) = mode {
        stores.iter().for_each(|s| s.set_offload(crate::offload::Offload { mode: m, ..s.offload() }));
    }
    let eb = stores[0].expert_bytes();
    let mut out = String::new();
    let (k, step) = crate::conv::Gen::as_step(&mut e, most)?.ok_or_else(|| Error("no scheduler's step for this model".into()))?;
    // (the counts of rows the memory holds are run; the others are said and skipped)
    let (rows_l, beyond): (Vec<usize>, Vec<usize>) = rows_l.into_iter().partition(|&n| n <= k);
    if rows_l.is_empty() {
        return err(format!("{beyond:?} rows asked for, where the memory holds {k} sequences"));
    }
    println!(
        "{}: {places} places a layer, an expert {:.2} MB, mode {}{}, {steps} steps after {warm}, prompts of {plen} tokens",
        dir.display(),
        eb as f64 / 1e6,
        stores[0].offload().mode.name(),
        if stores.iter().any(|s| s.device_layers() > 0) { ", decode calls planned on the GPU" } else { "" }
    );
    for &n in &beyond {
        println!("  {n:>3} rows: skipped: the memory holds {k} sequences");
    }
    for &n in &rows_l {
        let pages: Vec<[i32; 1]> = (0..n).map(|r| [r as i32]).collect();
        let mut at: Vec<(u32, usize)> = Vec::with_capacity(n);
        for (r, p) in pages.iter().enumerate() {
            let prompt: Vec<u32> = (0..plen).map(|i| ((i * 7919 + r * 104_729 + n * 31) % (vocab - 4) + 2) as u32).collect();
            let s = step.prefill(p, 0, &prompt, true)?.ok_or_else(|| Error("a prompt's last chunk gave no sample".into()))?;
            at.push((s.token, plen));
        }
        let (mut ms, mut layers): (Vec<f64>, Vec<LayerStats>) = (Vec::new(), Vec::new());
        for i in 0..warm + steps {
            let rows: Vec<Row> = at.iter().zip(&pages).map(|(&(token, pos), p)| Row::new(token, pos, p)).collect();
            let t0 = std::time::Instant::now();
            let got = step.decode(&rows)?;
            let t = t0.elapsed().as_secs_f64() * 1e3;
            at.iter_mut().zip(&got).for_each(|(a, s)| *a = (s.token, a.1 + 1));
            if i >= warm {
                ms.push(t);
                let st = step.expert_stats();
                layers.resize(layers.len().max(st.len()), LayerStats::default());
                layers.iter_mut().zip(&st).for_each(|(a, b)| a.add(b));
            }
        }
        for p in &pages {
            step.release(p)?;
        }
        let t = LayerStats::sum(&layers);
        let per = |x: u64| x as f64 / steps as f64;
        // (`--plan`: the planner's step for these rows on this machine with these places, its router taken as uniform, as a random model's is)
        if a.has("plan") {
            let mean = ms.iter().sum::<f64>() / steps as f64;
            match planned_step(dir, places, n, (plen + warm + steps / 2) as f64) {
                Ok(p) => println!("  {n:>3} rows: the planner's step {p:8.2} ms against {mean:.2} measured ({:+.1}%)", 100.0 * (p / mean - 1.0)),
                Err(e) => println!("  {n:>3} rows: no planner's step: {e}"),
            }
        }
        let missed = t.missed_bytes;
        let mean = ms.iter().sum::<f64>() / steps as f64;
        ms.sort_by(f64::total_cmp);
        println!(
            "  {n:>3} rows: {mean:8.2} ms a step (median {:.2}); a step {:.1} experts asked for, {:.1} found ({:.1}%), {:.1} copied, {:.1} by the host ({:.1} pairs), {:.1} copied behind; read for the misses {:.1} MB, the distinct missed experts' {:.1} MB ({:+.2}%); copies {:.1} GB/s ({:.2} ms a step), the host {:.2} ms a step ({:.1} GB/s)",
            ms[steps / 2],
            per(t.reads),
            per(t.hits),
            100.0 * t.hits as f64 / t.reads.max(1) as f64,
            per(t.copies),
            per(t.host),
            per(t.host_pairs),
            per(t.ahead),
            per(t.miss_bytes()) / 1e6,
            per(missed) / 1e6,
            if missed > 0 { 100.0 * (t.miss_bytes() as f64 / missed as f64 - 1.0) } else { 0.0 },
            if t.copy_ms > 0.0 { t.copy_bytes as f64 / t.copy_ms / 1e6 } else { 0.0 },
            t.copy_ms / steps as f64,
            t.host_ms / steps as f64,
            if t.host_ms > 0.0 { t.host_bytes as f64 / t.host_ms / 1e6 } else { 0.0 }
        );
        // (each layer's totals over the timed steps: a share of calls, of those copied whole among them, is kept as it is)
        let all: Vec<String> = layers.iter().enumerate().map(|(l, s)| s.json(l)).collect();
        out += &format!("{{\"rows\":{n},\"steps\":{steps},\"totals\":true,\"step_ms\":{mean:.4},\"expert_bytes\":{eb},\"layers\":[{}]}}\n", all.join(","));
    }
    if let Some(f) = a.flag("out") {
        std::fs::write(f, out).map_err(|e| Error(format!("{f}: {e}")))?;
    }
    Ok(())
}

/// The planner's decode step for `rows` rows of the model at `dir` on this machine, `places` of each layer's experts on the GPU and every expert in the computer's memory (src/plan.rs `speed_routed`, the
/// router's choices taken as uniform: a random model's), in ms; or why the planner gives none (`serve`'s plan for the rows, its places made `places`).
#[cfg(feature = "dev")]
fn planned_step(dir: &std::path::Path, places: usize, rows: usize, positions: f64) -> std::result::Result<f64, String> {
    let words = ["serve".to_string(), dir.display().to_string(), "--users".into(), rows.to_string(), "--no-draft".into()];
    let f = crate::plan::planned(&crate::args::Args::from(words)).map_err(|r| r.what)?;
    // (the planner refusing every option: the reasons it gives)
    let c = f.plan.chosen.as_ref().ok_or_else(|| {
        let why: Vec<String> =
            f.plan.options.iter().filter_map(|o| o.why.as_ref().map(|w| format!("{}: {w}", o.level))).chain(f.plan.notes.iter().cloned()).collect();
        format!("the planner has no plan ({})", if why.is_empty() { "it gives no reason".to_string() } else { why.join("; ") })
    })?;
    let mut lp = c.load.clone().ok_or("no plan to load")?;
    let sp = lp.spread.as_mut().ok_or("the plan is not the dynamic route's")?;
    if sp.host_bytes == 0 {
        return Err("the plan holds every expert on the GPU".into());
    }
    sp.places.iter_mut().for_each(|p| *p = places);
    let main = (0..f.machine.devices.len()).min_by_key(|&i| (std::cmp::Reverse(f.machine.devices[i].free), i)).ok_or("no GPU")?;
    let s = crate::plan::speed_routed(&f.shape, &f.machine, main, &lp, positions, rows, 1.0);
    if s > 0.0 {
        Ok(1e3 / s)
    } else {
        Err("the planner's speed is 0".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stack of three experts in the tiered layout (weights with exponents enough to need every tier and literals, so the blocks' lengths differ) cut into places: each expert's place, read as a pack of its own with the
    /// offsets the GPU adds (blocks from the place's start), decodes to that expert's weights, holds where the decode reads, and keeps each block's word alignment (the packer's flag depends on it).
    #[test]
    fn a_tiered_experts_place_decodes_to_its_own_weights() {
        let (experts, rows, cols) = (3usize, 128usize, 64usize);
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        // (expert e's exponents spread by e: the third's blocks the longest)
        let w: Vec<u16> = (0..experts * rows * cols)
            .map(|i| {
                let (e, r) = (i / (rows * cols), next());
                let exp = 120 + (r % (4 + 6 * e as u64)) as u16;
                ((r >> 32) as u16 & 0x8000) | (exp << 7) | ((r >> 40) as u16 & 0x7F)
            })
            .collect();
        let p = glyd_gpu::pack::pack_tiered_checked(&w, experts * rows, cols).expect("a lossless pack");
        assert!(p.block_base[..experts * rows * cols / 1024].iter().any(|&v| v < 0), "some step takes the general decode (its flag)");
        let pl = TieredPlaces::of(&p, experts).unwrap();
        assert_eq!(pl.strides(), [(pl.steps * 1280) as u64, pl.stride as u64, ((pl.steps + 1) * 4) as u64]);
        assert!(pl.stride.is_multiple_of(128) && pl.blocks.len() == experts * pl.stride);
        for e in 0..experts {
            let one = pl.expert(e);
            let mut out = vec![0u16; rows * cols];
            assert!(glyd_gpu::pack::unpack_tiered(&one, &mut out), "expert {e} decodes");
            assert_eq!(out, w[e * rows * cols..(e + 1) * rows * cols], "expert {e}'s weights");
            assert!(glyd_gpu::pack::tiered_blocks_hold(&one.data, &one.blocks, &one.block_base), "expert {e}'s place holds the decode's reads");
            for j in 0..=pl.steps {
                let (a, b) = (one.block_base[j], p.block_base[e * pl.steps + j]);
                assert_eq!(
                    (a < 0, glyd_gpu::pack::block_start(a) % 4),
                    (b < 0 && j < pl.steps, glyd_gpu::pack::block_start(b) % 4),
                    "expert {e} step {j}: flag and alignment"
                );
            }
        }
        // (a stack that is not whole experts' is refused)
        assert!(TieredPlaces::of(&p, 5).is_err());
    }
}
