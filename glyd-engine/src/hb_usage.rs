//! `glyd-engine calibrate-experts MODEL OUT [--tokens N] [--devices 0,1 --split CUTS] [--context N]`: hummingbird's corpus pass
//! (docs/design/hummingbird.md, #429). The model runs on the engine over the built-in calibration corpus (`calib::corpus_docs`, the chat
//! sections through the model's template), prompt passes only, and each mixture-of-experts layer's router choices are read back and counted
//! per expert; OUT is the usage file (`hb_codec::Usage`) that `pack MODEL OUT --layout hummingbird --usage FILE` gives each expert its bits by.
//!
//! The counts are kept by the model's decoder layer while the pass runs (`record`, from the executor's MoE operations: src/dynamic) and
//! written by the description's block (`usage_of`: a decoder layer is two blocks, the mixer's then the feed-forward's, so layer l's experts
//! are block 2 l + 1). A layer whose experts are not all on the GPU routes in a block of its own and is not counted (`missed`): the pass
//! then says which, and the packer refuses the file for them.

use crate::ffi::{err, Error, Result};
use crate::hb_codec::{LayerUse, Usage};
use crate::hb_pack::moe_layers;
use crate::runtime::describe::Described;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// The tokens the pass runs where none is asked.
pub const TOKENS: u64 = 32768;

/// Each decoder layer's count of choices per expert, the tokens counted, and the layers that routed where the pass could not read.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Counts {
    pub layers: BTreeMap<usize, Vec<u64>>,
    pub missed: BTreeSet<usize>,
}

impl Counts {
    /// A call's choices (`ids`, the router's [tokens, k], int64) of layer `layer`, of `experts` experts.
    pub fn add(&mut self, layer: usize, experts: usize, ids: &[i64]) -> Result<()> {
        let c = self.layers.entry(layer).or_insert_with(|| vec![0; experts]);
        if c.len() != experts {
            return err(format!("layer {layer}: {experts} experts, where its earlier calls had {}", c.len()));
        }
        for &e in ids {
            match usize::try_from(e) {
                Ok(e) if e < experts => c[e] += 1,
                _ => return err(format!("layer {layer}: a choice of expert {e}, of {experts}")),
            }
        }
        Ok(())
    }
}

static ON: AtomicBool = AtomicBool::new(false);
static COUNTS: Mutex<Option<Counts>> = Mutex::new(None);

/// Whether a pass is counting (the executor reads the router's choices back only then).
pub fn recording() -> bool {
    ON.load(Ordering::Relaxed)
}

/// Starts counting, from nothing.
pub fn start() {
    *COUNTS.lock().expect("the counts") = Some(Counts::default());
    ON.store(true, Ordering::Relaxed);
}

/// Stops counting and returns the counts.
pub fn stop() -> Counts {
    ON.store(false, Ordering::Relaxed);
    COUNTS.lock().expect("the counts").take().unwrap_or_default()
}

/// The executor's call: layer `layer` chose `ids` among `experts`.
pub fn record(layer: usize, experts: usize, ids: &[i64]) -> Result<()> {
    match COUNTS.lock().expect("the counts").as_mut() {
        Some(c) => c.add(layer, experts, ids),
        None => Ok(()),
    }
}

/// The executor's call for a layer whose choices it cannot read (its experts in a cache that routes on its own).
pub fn missed(layer: usize) {
    if let Some(c) = COUNTS.lock().expect("the counts").as_mut() {
        c.missed.insert(layer);
    }
}

/// The usage file of `counts` for the model described by `d`: each MoE layer's counts by its block. An error where a MoE layer was not
/// counted, or a layer counted is not a MoE layer of the description.
pub fn usage_of(d: &Described, counts: &Counts, model: &str, tokens: u64) -> Result<Usage> {
    let moe = moe_layers(d)?;
    let mut layers = Vec::new();
    let mut missing = Vec::new();
    for m in &moe {
        if m.block % 2 != 1 {
            return err(format!("the MoE layer at block {} is not a decoder layer's feed-forward", m.block));
        }
        match counts.layers.get(&(m.block / 2)) {
            Some(c) if c.len() == m.experts => layers.push(LayerUse { block: m.block, counts: c.clone() }),
            Some(c) => return err(format!("decoder layer {}: {} experts counted, where the description has {}", m.block / 2, c.len(), m.experts)),
            None => missing.push(m.block / 2),
        }
    }
    if let Some(l) = counts.layers.keys().find(|&&l| !moe.iter().any(|m| m.block / 2 == l)) {
        return err(format!("decoder layer {l} was counted, and it is not a MoE layer of the description"));
    }
    if !missing.is_empty() {
        let why = if counts.missed.is_empty() {
            ""
        } else {
            " (their experts were in a cache that routes on its own: run the pass with every layer's experts on the GPUs)"
        };
        return err(format!("decoder layers {missing:?} were not counted{why}"));
    }
    Ok(Usage { model: model.to_string(), tokens, layers })
}

/// The corpus pass on the reference backend (tiny models, the host tests): each document's tokens as one sequence, the MoE layers' choices
/// counted by decoder layer.
pub fn count_on_ref(m: &mut crate::runtime::reference::model::RefModel, docs: &[Vec<u32>]) -> Result<Counts> {
    for l in &mut m.layers {
        l.routes = Some(Vec::new());
    }
    for d in docs {
        for l in &mut m.layers {
            l.seqs.clear();
        }
        m.forward(d, 0)?;
    }
    let mut c = Counts::default();
    for (b, l) in m.layers.iter_mut().enumerate() {
        let Some(r) = l.routes.take() else { continue };
        if let crate::runtime::kinds::Body::Ffn(crate::runtime::kinds::Ffn::Moe(moe)) = &l.block.body {
            let ids: Vec<i64> = r.into_iter().map(|e| e as i64).collect();
            c.add(b / 2, moe.experts, &ids)?;
        }
    }
    Ok(c)
}

fn io<T>(r: std::io::Result<T>) -> Result<T> {
    r.map_err(|e| Error(e.to_string()))
}

/// The pass on the GPU: `model` (a directory, or a Hub name in the cache) on `devices` (layers cut at `cuts` across them), `tokens` of the
/// corpus at most `ctx` a document; the usage file written to `out`.
pub fn run(model: &str, out: &std::path::Path, tokens: u64, devices: &[i32], cuts: &[usize], ctx: usize) -> Result<()> {
    run_with(model, out, tokens, devices, cuts, ctx, false)
}

/// `run`; `synthetic`: documents of token ids drawn over the vocabulary in place of the corpus (a random test model, which has no tokenizer: its counts are a test's, not a recipe's).
pub fn run_with(model: &str, out: &std::path::Path, tokens: u64, devices: &[i32], cuts: &[usize], ctx: usize, synthetic: bool) -> Result<()> {
    let t0 = std::time::Instant::now();
    let dir = io(glyd_gpu::save::Source::find(model))?.dir;
    let text = io(std::fs::read_to_string(dir.join("config.json")))?;
    let cfg: serde_json::Value = serde_json::from_str(&text).map_err(|e| Error(format!("config.json: {e}")))?;
    let d = crate::runtime::describe::describe(&cfg)?;
    if moe_layers(&d)?.is_empty() {
        return err(format!("{model}: no mixture of experts to count"));
    }
    let family = crate::dynamic::split::family(&text)
        .ok_or_else(|| Error(format!("{model}: not a model of the dynamic route (Kimi-K3, GLM-5.3-Flash, DeepSeek-V4, Muse Spark)")))?;
    let docs = if synthetic {
        let vocab = crate::dynamic::split::config_num(&text, "vocab_size").unwrap_or(1000).max(16) as u64;
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        (0..16)
            .map(|d| {
                let ids = (0..ctx.max(16))
                    .map(|_| {
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        (x % (vocab - 2) + 1) as u32
                    })
                    .collect();
                (String::new(), format!("synthetic {d}"), ids)
            })
            .collect::<Vec<(String, String, Vec<u32>)>>()
    } else {
        let tok = crate::tok::Tok::load(&dir)?;
        let own = match family {
            "DeepSeek-V4" => Some(crate::chat::DSV4_CHAT),
            "Kimi-K3" => Some(crate::chat::KIMI_CHAT),
            _ => None,
        };
        let chat = crate::chat::Chat::load_or(&dir, own)?;
        crate::calib::corpus_docs(&tok, chat.as_ref())?
    };
    // (one sequence a document, prompt passes only; no graph, for the choices are read back after each layer)
    let o = crate::dynamic::Opts { ctx: ctx + 1, chunk: 256, graph: false, device: devices.first().copied().unwrap_or(0), layout: None };
    let mut e = crate::dynamic::split::open(&dir, &o, cuts, devices)?;
    start();
    let mut ran = 0u64;
    let pass = (|| -> Result<()> {
        for (_, name, ids) in docs.iter().cycle().take(docs.len().max(1) * 64) {
            if ran >= tokens {
                break;
            }
            let take = ids.len().min(ctx).min((tokens - ran) as usize);
            if take < 2 {
                continue;
            }
            e.reset()?;
            e.prefill(&ids[..take]).map_err(|x| Error(format!("{name}: {}", x.0)))?;
            ran += take as u64;
        }
        Ok(())
    })();
    let counts = stop();
    pass?;
    let usage = usage_of(&d, &counts, model, ran)?;
    io(std::fs::write(out, usage.to_json()))?;
    eprintln!("{}: {} MoE layers counted over {ran} tokens of the corpus ({:.0} s)", out.display(), usage.layers.len(), t0.elapsed().as_secs_f64());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::reference::model::RefModel;
    use crate::runtime::reference::testing::Fixture;

    #[test]
    fn choices_are_counted_by_layer_and_refused_past_the_experts() {
        let mut c = Counts::default();
        c.add(3, 4, &[0, 1, 1, 3]).unwrap();
        c.add(3, 4, &[1, 2]).unwrap();
        assert_eq!(c.layers[&3], vec![1, 3, 1, 1]);
        assert!(c.add(3, 4, &[4]).is_err() && c.add(3, 4, &[-1]).is_err() && c.add(3, 5, &[0]).is_err());
    }

    /// A tiny GLM-5.3-Flash, DeepSeek-V4 and Qwen3.8 MoE (transformers' fixtures) run on the reference backend over a few documents: every
    /// MoE layer counted, each token's top-k choices once a layer, written by block; then packed from those counts. A layer missing from
    /// the counts, or a layer counted that has no experts, is refused.
    #[test]
    fn the_reference_pass_counts_every_moe_layer_and_feeds_the_packer() {
        for fixture in ["glm53flash_model", "dsv4_model", "qwen38_moe_model"] {
            let dir = std::env::temp_dir().join(format!("glyd-hbusage-{fixture}-{}", std::process::id()));
            let src = dir.join("src");
            let d = crate::hb_pack::tests::release(&src, fixture);
            let f = Fixture::load(fixture);
            let tensors = |n: &str| -> Option<(Vec<usize>, Vec<f32>)> { f.j["tensors"].get(n).map(|_| f.tensor(n)) };
            let mut m = RefModel::load(&d, &tensors).unwrap();
            let vocab = d.desc.vocab as u32;
            let docs: Vec<Vec<u32>> = (0..3u32).map(|k| (0..12u32).map(|i| (i * 7 + k * 13 + 1) % vocab).collect()).collect();
            let c = count_on_ref(&mut m, &docs).unwrap();
            let moe = moe_layers(&d).unwrap();
            assert_eq!(c.layers.len(), moe.len(), "{fixture}");
            for l in &moe {
                let Some(crate::runtime::kinds::Body::Ffn(crate::runtime::kinds::Ffn::Moe(k))) = d.desc.blocks.get(l.block).map(|b| &b.body) else { panic!() };
                assert_eq!(c.layers[&(l.block / 2)].iter().sum::<u64>(), 36 * k.router.top_k as u64, "{fixture} block {}", l.block);
            }
            let u = usage_of(&d, &c, "tiny", 36).unwrap();
            assert_eq!(u.layers.len(), moe.len());
            let up = dir.join("usage.json");
            std::fs::write(&up, u.to_json()).unwrap();
            crate::hb_pack::run(src.to_str().unwrap(), &dir.join("out"), 2, &up).unwrap();
            assert!(crate::hb_pack::HbDir::open(&dir.join("out")).unwrap().is_some());
            // refused: a layer not counted, a layer counted that has no experts
            let mut less = c.clone();
            less.layers.remove(&(moe[0].block / 2));
            less.missed.insert(moe[0].block / 2);
            let e = usage_of(&d, &less, "tiny", 36).unwrap_err().0;
            assert!(e.contains("were not counted") && e.contains("on the GPUs"), "{e}");
            let mut more = c.clone();
            more.layers.insert(9999, vec![0; 4]);
            assert!(usage_of(&d, &more, "tiny", 36).unwrap_err().0.contains("not a MoE layer"));
            std::fs::remove_dir_all(&dir).unwrap();
        }
    }
}
