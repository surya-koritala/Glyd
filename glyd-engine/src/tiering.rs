//! Expert tiering for every mixture of experts on the dynamic route (R12, #232; docs/design/runtime.md §6): a layer's experts in the expert cache (src/offload.rs `ExpertStore`), places for the busiest of them on the GPU and
//! every one in the computer's memory, for the formats whose experts are each the same bytes (`moe::ExpertStack::pooled`: the block layouts, the FP4 layouts as DeepSeek-V4 and Kimi-K3 release them). `MoeLayer::with_cache`
//! runs the layer over the places; a family's loader builds the stacks and the cache here (`cached_stacks`). Hummingbird's two bit widths a layer are src/hbg_store.rs's.

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

/// `glyd-engine tiercheck MODEL [--places N] [--tokens N] [--mode copy|cpu|auto] [--layout i6|i5|i4]`: the model of the dynamic route (a release whose experts the cache takes, or a hummingbird directory) on GPU 0 with every expert there, then with
/// its experts in the expert cache (`--places` a layer, every expert in the computer's memory; `--mode` the misses', copy by default): the same prompt and decode steps, and every position's logits compared. In copy mode the
/// layers' products run over the places with the kernels of the resident layer, so the logits are the same bits (the GPU job's parity record of R12 and of hummingbird's cache); with the host computing some pairs, the largest
/// difference is printed.
#[cfg(feature = "dev")]
pub fn check(a: &crate::args::Args) -> Result<()> {
    use crate::ffi::Error;
    let dir = std::path::Path::new(a.need(1, "MODEL")?);
    let places = a.num("places", 2usize)?;
    let mode = match a.flag("mode") {
        Some(m) => crate::offload::Mode::parse(m).ok_or_else(|| Error(format!("--mode {m}: copy, cpu, split or auto")))?,
        None => crate::offload::Mode::Copy,
    };
    let text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| Error(format!("{}: {e}", dir.display())))?;
    let vocab = crate::dynamic::split::config_num(&text, "vocab_size").unwrap_or(1000).max(16);
    let n = a.num("tokens", 24usize)?.max(6);
    let tokens: Vec<u32> = (0..n).map(|i| ((i * 7919 + 13) % (vocab - 4) + 2) as u32).collect();
    // (`--layout`: the level a family's loader packs at, GLM's experts at a block level among them)
    let o = crate::dynamic::Opts { ctx: n + 64, chunk: 256, graph: false, device: 0, layout: a.layout()? };
    let run = |cached: bool| -> Result<Vec<Vec<f32>>> {
        let mut e = if cached {
            let caching = crate::hbg_store::HbCaching { places, ram: None, offload: crate::offload::Offload { mode, ..Default::default() } };
            crate::dynamic::split::open_cached(dir, &o, &[], &[0], vec![caching])?
        } else {
            crate::dynamic::split::open(dir, &o, &[], &[0])?
        };
        let mut out = Vec::new();
        e.prefill(&tokens[..n - 4])?;
        out.push(e.logits()?);
        for &t in &tokens[n - 4..] {
            e.step(t)?;
            out.push(e.logits()?);
        }
        Ok(out)
    };
    let (resident, cached) = (run(false)?, run(true)?);
    let (mut differ, mut worst) = (0usize, 0f32);
    for (r, c) in resident.iter().zip(&cached) {
        for (x, y) in r.iter().zip(c) {
            if x.to_bits() != y.to_bits() {
                differ += 1;
                worst = worst.max((x - y).abs());
            }
        }
    }
    let total: usize = resident.iter().map(Vec::len).sum();
    println!(
        "{}: {} logits of {} positions, resident against the expert cache ({places} places a layer, {mode:?}): {differ} differ, the largest by {worst:e}",
        dir.display(),
        total,
        resident.len()
    );
    if mode == crate::offload::Mode::Copy && differ > 0 {
        return err(format!("{differ} logits differ in copy mode, where the products over the places are the resident layer's"));
    }
    Ok(())
}
