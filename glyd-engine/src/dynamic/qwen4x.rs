//! Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`, Condor's model) on the executor (R3d, #140): its config's layer kinds lowered to the shared operations
//! (`lower::qwen4x`) and loaded here: the matrices packed at the level asked (penguin's lossless packs where none is), the ones the library does not pack and the table of n-grams
//! held as saved (bf16), the small tensors as saved but A_log in fp32 (the delta rule reads it so). The experts either all on the GPU at kestrel's six bits (`load`:
//! Condor's block, `QxMoe`, handed over as a `ResidentMoe`), or from a pack of codebook rows in the shared expert store (`load_cached`: Condor's codebook block,
//! `QxMoeCb`, the misses on the host). Condor's engine (src/model_qwen4x.rs) still holds what the executor does not take yet: the draft head, the table of n-grams
//! read from the disk, the vision tower.

use super::{Dyn, ModelDesc, Op, Opts, ResidentMoe, Stage};
use crate::cb_pack::CbPack;
use crate::dev::Dev;
use crate::ffi::{err, Result};
use crate::kern_nh::MoeBlock;
use crate::moe_qwen4x::{QxCbShared, QxMoeCb};
use crate::offload::{ExpertStore, Offload};
pub use crate::qwen4x::PackMode;
use crate::weights::Layout;
use glyd_gpu::safetensors::Checkpoint;
use std::path::Path;

/// The executor description's family name.
pub const FAMILY: &str = "Qwen3.8-Flash-Next";

/// The executor's description of the model in `dir` for sequences of up to `ctx` positions: its kinds lowered (`lower::qwen4x`), held to Condor's own reading of
/// the config (`ConfigQx::agrees`).
pub fn lowered(dir: &Path, ctx: usize) -> Result<ModelDesc> {
    let (kinds, label) = crate::runtime::describe::describe_dir(dir)?;
    let c = crate::qwen4x::ConfigQx::agrees(crate::qwen4x::ConfigQx::load(dir)?, &kinds).map_err(|e| e.context(&label))?;
    super::lower::qwen4x(&kinds, c.eos.clone(), ctx.min(c.max_pos)).map_err(|e| e.context(&label))
}

/// The model in `dir` on the executor's loader (`Dyn::new_with`: its matrices at the level asked or this GPU's, the table of n-grams and the shapes the library
/// does not pack as saved, A_log in fp32), its experts all on the GPU (`QxMoe`'s block, handed over as a `ResidentMoe`) at the level asked: a lossy level's codes
/// where `o.layout` is one, else lossless (penguin), as every family's.
pub fn load(dir: &Path, o: &Opts) -> Result<Dyn> {
    load_level(dir, o, experts_at(o.layout))
}

/// The experts' layout for the matrices' `layout`: a lossy level's block codes (six, five or four bits) where it is one, else the lossless layout.
pub fn experts_at(layout: Option<Layout>) -> Layout {
    match layout {
        Some(l @ (Layout::Int6 | Layout::Int5 | Layout::Int4)) => l,
        _ => Layout::Tiered,
    }
}

/// `load` with the experts at `level` (lossless, or a lossy level's codes: six, five or four bits).
pub fn load_level(dir: &Path, o: &Opts, level: Layout) -> Result<Dyn> {
    let c = crate::qwen4x::ConfigQx::load(dir)?;
    let d = lowered(dir, o.ctx + o.chunk + 16)?;
    // (what the GPU would hold: the experts at the level (lossless: two bytes a weight at most) and the table of n-grams as saved; past that, `load_cached`)
    let per = match level {
        Layout::Int6 => 6,
        Layout::Int5 => 5,
        Layout::Int4 => 4,
        _ => 16,
    };
    let experts = (c.layers * c.experts * 3 * c.inter * c.hidden) as u64 * per / 8;
    let table: u64 = held_as_saved(&d).iter().map(|&k| (d.mats[k].rows * d.mats[k].cols * 2) as u64).sum();
    let free = Dev::new(o.device)?.mem_info().map(|(f, _)| f as u64).unwrap_or(0);
    if experts + table > free {
        return err(format!(
            "{FAMILY}: its experts ({:.1} GB at {per} bits) and its table of n-grams ({:.1} GB) do not fit this GPU's {:.1} GB free: a pack of its experts serves it (`load_cached`)",
            experts as f64 / 1e9,
            table as f64 / 1e9,
            free as f64 / 1e9
        ));
    }
    let st = Stage::whole(d.layers.len());
    let blocks = |ck: &Checkpoint, _: Layout| -> Result<Vec<Box<dyn MoeBlock>>> {
        (0..c.layers).map(|l| Ok(Box::new(ResidentMoe(crate::model_qwen4x::QxMoe::load(l, &c, ck, level)?.into_layer())) as Box<dyn MoeBlock>)).collect()
    };
    Dyn::new_with(dir, d, None, st, o, &blocks)
}

/// Condor's routed experts from a pack of codebook rows (glyd-hbr-codes/2: each expert cb2, cb3, cb4 or ternary, the pack's mode's sidecar over them) in the
/// shared expert store (src/offload.rs): a layer's places on the GPU, the pack's file mapped as their host tier, the misses computed by the host or copied in as
/// `offload` says.
pub struct Caching {
    pub pack: std::path::PathBuf,
    pub mode: PackMode,
    /// places a layer (None: what the GPU has free after the always-on matrices, less `reserve`, over the layers)
    pub places: Option<usize>,
    pub offload: Offload,
    /// GPU bytes the places leave for the sequences' keys, values and states and the executor's scratch (`reserve_for`)
    pub reserve: u64,
}

impl Caching {
    /// The pack in `pack` served as `mode`, its places sized by the room the GPU has, its misses computed by the host, the mode's skipping, room kept for `users`
    /// sequences of the description `d`.
    pub fn new(pack: &Path, mode: PackMode, d: &ModelDesc, users: usize) -> Caching {
        let offload = Offload { mode: crate::offload::Mode::Cpu, skip_rank: mode.skip_rank(), async_copies: true, ..Offload::default() };
        Caching { pack: pack.to_path_buf(), mode, places: None, offload, reserve: reserve_for(d, users) }
    }
}

/// The GPU memory a model's places leave: each of `users` sequences' keys, values and states for the description's context, and the executor's slots and scratch
/// (1.5 GiB, not measured for this model: the final pass measures it).
pub fn reserve_for(d: &ModelDesc, users: usize) -> u64 {
    let ring: u64 = d.rings.iter().map(|r| (d.max_pos * r.nk * r.d * 2 * (1 + r.values as usize)) as u64).sum();
    let states: u64 = d.states.iter().map(|&b| b as u64).sum();
    users.max(1) as u64 * (ring + states) + (3 << 29)
}

/// Places a layer: the GPU's `free` bytes less `reserve`, over `layers` layers of places of `place` bytes; at least one, at most `kept` (all of a layer's).
pub fn places_for(free: u64, reserve: u64, layers: usize, place: usize, kept: usize) -> usize {
    ((free.saturating_sub(reserve) / (layers.max(1) * place.max(1)) as u64) as usize).clamp(1, kept.max(1))
}

/// The model in `dir` on the executor's loader with its routed experts in the expert cache (`Caching`): the always-on matrices as `load` loads them, each layer's
/// block Condor's codebook block (`QxMoeCb`) over one store shared by the layers.
pub fn load_cached(dir: &Path, o: &Opts, k: &Caching) -> Result<Dyn> {
    let c = crate::qwen4x::ConfigQx::load(dir)?;
    let d = lowered(dir, o.ctx + o.chunk + 16)?;
    let mut pack = CbPack::open_dims(&k.pack, c.hidden, c.inter)?;
    if let Some(side) = k.mode.side(&k.pack) {
        pack = pack.with_side(&side)?;
    }
    if let Some(l) = (0..c.layers).find(|&l| !pack.has(l)) {
        return err(format!("{}: no experts of layer {l} of the model's {}", k.pack.display(), c.layers));
    }
    let (store, common) = (ExpertStore::new(), QxCbShared::new());
    store.set_offload(k.offload);
    common.stream.set(true);
    let place = (0..c.layers).map(|l| pack.place_bytes(l)).max().unwrap_or(1);
    let kept = (0..c.layers).filter_map(|l| pack.kept(l).map(<[usize]>::len)).min().unwrap_or(1);
    let dev = o.device;
    let blocks = |ck: &Checkpoint, layout: Layout| -> Result<Vec<Box<dyn MoeBlock>>> {
        // (read after the always-on matrices are on the GPU: what is free now, less the reserve, is the places')
        let slots = match k.places {
            Some(p) => p,
            None => places_for(Dev::new(dev)?.mem_info().map(|(f, _)| f as u64).unwrap_or(0), k.reserve, c.layers, place, kept),
        };
        (0..c.layers)
            .map(|l| {
                crate::status::STATUS.beat();
                Ok(Box::new(QxMoeCb::load(l, &c, ck, layout, &pack, &store, slots, &common)?) as Box<dyn MoeBlock>)
            })
            .collect()
    };
    let e = Dyn::new_with(dir, d, None, Stage::whole(c.layers), o, &blocks)?;
    common.fill_page_tables();
    Ok(e)
}

/// `load_cached` of the pack in `pack` served as its mode (`PackMode::of`), room kept for `users` sequences.
pub fn load_pack(dir: &Path, pack: &Path, precise: bool, users: usize, o: &Opts) -> Result<Dyn> {
    let d = lowered(dir, o.ctx + o.chunk + 16)?;
    load_cached(dir, o, &Caching::new(pack, PackMode::of(pack, precise), &d, users))
}

/// The matrices held as saved: the tables of n-grams (`Op::NgramRows` reads their rows).
fn held_as_saved(d: &ModelDesc) -> Vec<usize> {
    d.layers.iter().flatten().filter_map(|op| if let Op::NgramRows(x) = op { Some(x.table) } else { None }).collect()
}

#[cfg(test)]
mod tests {
    use super::super::{HeadMix, IdxKeys, Op, Slot, StreamsMix};
    use crate::runtime::describe::describe;

    fn tiny() -> serde_json::Value {
        let m: serde_json::Value = serde_json::from_str(include_str!("../../tests/fixtures/qwen4x_tiny_manifest.json")).unwrap();
        serde_json::from_str(m["config"].as_str().unwrap()).unwrap()
    }

    /// The executor's loader takes every matrix of the tiny model (a whole model's stage wants each one), holds as saved the table of n-grams and the shapes the
    /// library does not pack (the inject weights' 4 rows, the b and a gates' 2 value heads), packs the rest, and widens A_log alone to fp32.
    #[test]
    fn the_loader_holds_each_matrix_and_small_tensor_as_its_operations_read_it() {
        let m = super::super::lower::qwen4x(&describe(&tiny()).unwrap(), vec![2000], 1100).unwrap();
        assert!(m.wanted(&super::super::Stage::whole(m.layers.len())).iter().all(|&w| w), "a matrix no operation multiplies by");
        let man: serde_json::Value = serde_json::from_str(include_str!("../../tests/fixtures/qwen4x_tiny_manifest.json")).unwrap();
        let rows: std::collections::HashMap<String, usize> =
            man["tensors"].as_array().unwrap().iter().map(|t| (t[0].as_str().unwrap().to_string(), t[2][0].as_u64().unwrap() as usize)).collect();
        let rows_of = |n: &str| rows.get(n).copied();
        let saved: Vec<usize> = (0..m.mats.len()).filter(|&k| super::super::held_as_saved(&m, k, &rows_of)).collect();
        let table = super::held_as_saved(&m);
        assert!(!table.is_empty() && table.iter().all(|k| saved.contains(k)), "{table:?} of {saved:?}");
        for &k in &saved {
            let it = &m.mats[k];
            assert!(
                table.contains(&k) || it.members.iter().any(|n| rows_of(n).unwrap() % 64 != 0) || it.cols % 16 != 0,
                "{}: {}x{}",
                it.members[0],
                it.rows,
                it.cols
            );
        }
        // (and packs the rest: their old rule, rows and columns each a multiple of 64, holds the same matrices as saved)
        for k in 0..m.mats.len() {
            let it = &m.mats[k];
            assert_eq!(saved.contains(&k), table.contains(&k) || it.rows % 64 != 0 || it.cols % 64 != 0, "{}", it.members[0]);
        }
        assert!(saved.len() > table.len(), "the unpacked shapes");
        let f32s = super::super::read_f32(&m);
        assert!(!f32s.is_empty() && f32s.iter().all(|&i| m.small[i].ends_with("A_log")), "{:?}", f32s.iter().map(|&i| &m.small[i]).collect::<Vec<_>>());
    }

    /// The tiny model of the `qwen4x` logits case, lowered: eight layers of gated residuals around each block (a gated mix before it, the gated update after),
    /// the n-gram embedding's three operations before the second layer's, Gated DeltaNet with the sigmoid gate, the sparse attention's index product, its
    /// queries and pools, its unweighted pick of 2 pools of 4 and the tail, the attention over the picks with its keys and values in states, the experts, and
    /// the final mixer on the last layer; every matrix and small tensor named is the checkpoint's, and every slot an operation reads is sized.
    #[test]
    fn condors_tiny_model_lowers_to_the_shared_operations() {
        let d = describe(&tiny()).unwrap();
        let m = super::super::lower::qwen4x(&d, vec![2000], 1100).unwrap();
        assert!(matches!(m.first, Op::Streams));
        // (each sparse attention layer's keys and values, its indexer's raw keys and its pools' keys: three rings, paged as every attention's, R4b)
        assert_eq!((m.layers.len(), m.hidden, m.rings.len(), m.ropes.len()), (8, 256, 6, 1));
        for three in m.rings.chunks(3) {
            let shapes: Vec<_> = three.iter().map(|r| (r.d, r.nk, r.window, r.values, r.ratio)).collect();
            assert_eq!(shapes, [(256, 2, 0, true, 1), (128, 1, 0, false, 1), (128, 1, 0, false, 4)]);
        }
        assert!(super::super::rows::pages_all(&m));
        let kinds: Vec<Vec<&'static str>> = m
            .layers
            .iter()
            .map(|ops| {
                ops.iter()
                    .map(|op| match op {
                        Op::StreamsPre { mix: StreamsMix::Gated { .. }, .. } => "pre",
                        Op::StreamsPost { gated: true, .. } => "post",
                        Op::StreamsHead { mix: HeadMix::Gated { .. }, .. } => "head",
                        Op::NgramRows(_) => "ngram rows",
                        Op::NgramAdd(_) => "ngram add",
                        Op::Linear { .. } => "linear",
                        Op::GdnConv { .. } => "conv",
                        Op::GdnDelta { .. } => "delta",
                        Op::GatedNorm { sigmoid: true, .. } => "sigmoid norm",
                        Op::IdxMean { .. } => "index",
                        Op::IdxPick(x) if !x.weighted && matches!(x.keys, IdxKeys::Pools(_)) && (x.ratio, x.top_k, x.heads) == (4, 2, 4) => "pick",
                        Op::Attn(a) if a.picked && (a.nq, a.nk, a.d) == (24, 2, 256) => "picked attention",
                        Op::Moe(_) => "experts",
                        o => panic!("{o:?}"),
                    })
                    .collect()
            })
            .collect();
        let gdn = ["pre", "linear", "linear", "conv", "delta", "sigmoid norm", "linear", "post", "pre", "experts", "post"];
        let attn = ["pre", "linear", "linear", "index", "pick", "picked attention", "linear", "post", "pre", "experts", "post"];
        assert_eq!(kinds[0], gdn);
        assert_eq!(kinds[1][..3], ["ngram rows", "linear", "ngram add"]);
        assert_eq!(kinds[1][3..], gdn);
        assert_eq!(kinds[3], attn);
        assert_eq!(kinds[7][..attn.len()], attn);
        assert_eq!(kinds[7][attn.len()..], ["head"]);
        // (each matrix's members are the checkpoint's tensors, its rows theirs stacked)
        let names = d.names();
        assert!(m.mats.iter().all(|it| it.members.iter().all(|n| names.contains(n))) && m.small.iter().all(|n| names.contains(n)));
        let rows: Vec<(usize, usize)> = m.mats.iter().map(|it| (it.rows, it.cols)).collect();
        // (layer 0: the gated residual's three, qkvz, ba, out, the experts' gated residual's three)
        assert_eq!(rows[..9], [(64, 1024), (1024, 64), (4, 1024), (2048, 256), (12, 256), (256, 768), (64, 1024), (1024, 64), (4, 1024)]);
        let table = m.layers[1].iter().find_map(|op| if let Op::NgramRows(x) = op { Some(*x) } else { None }).unwrap();
        assert_eq!((rows[table.table], table.per, table.row, table.eos), ((128 * 132, 16), 8, 16, 2000));
        // (the states: per GDN layer its convolution's and recurrence's; the n-gram's tokens and convolution; none of an attention layer's, which are rings)
        assert_eq!(m.states.iter().filter(|&&b| b == 1100 * 2 * 256 * 2 || b == 1100 * 128 * 2).count(), 0);
        assert_eq!(m.states.iter().filter(|&&b| b == 8).count(), 1);
        for (s, w) in
            [(Slot::S, 1024), (Slot::Hq, 1024), (Slot::Mid, 64), (Slot::Mix, 12288 + 1024), (Slot::Qr, 640), (Slot::Iq, 512), (Slot::Gu, 1280), (Slot::Kv, 4)]
        {
            assert_eq!(m.width(s), w, "{s:?}");
        }
        // (A_log the delta rule's, read in fp32)
        assert_eq!(super::super::read_f32(&m).len(), 6);
        assert!(super::super::read_f32(&m).iter().all(|&i| m.small[i].ends_with("linear_attn.A_log")));
    }

    /// The places take what the GPU has free past the reserve, a layer's share of it: at least one, at most all of a layer's experts; the reserve holds each
    /// sequence's keys, values and states beside the scratch, and grows with the users by them.
    #[test]
    fn the_places_take_the_room_past_the_reserve() {
        assert_eq!(super::places_for(10 << 30, 2 << 30, 48, 1 << 20, 512), 170);
        assert_eq!(super::places_for(1 << 30, 2 << 30, 48, 1 << 20, 512), 1);
        assert_eq!(super::places_for(1 << 40, 0, 48, 1 << 20, 512), 512);
        let m = super::super::lower::qwen4x(&describe(&tiny()).unwrap(), vec![2000], 1100).unwrap();
        let (one, four) = (super::reserve_for(&m, 1), super::reserve_for(&m, 4));
        let states: u64 = m.states.iter().map(|&b| b as u64).sum();
        let seq = one - (3 << 29);
        assert!(states > 0 && seq >= states && four - one == 3 * seq, "{one} {four} {states}");
    }

    /// The experts take a lossy level's codes where the matrices' layout is one, else the lossless layout (penguin's, as every family's).
    #[test]
    fn the_experts_are_lossless_unless_a_lossy_level_is_asked() {
        use crate::weights::Layout;
        assert_eq!(super::experts_at(None), Layout::Tiered);
        assert_eq!(super::experts_at(Some(Layout::Twelve)), Layout::Tiered);
        for l in [Layout::Int6, Layout::Int5, Layout::Int4] {
            assert_eq!(super::experts_at(Some(l)), l);
        }
    }

    /// The indexer's head size and pool are the config's: heads of 64 over pools of 2 lower to the raw keys' ring of 64, the pools' ring of entries of 2 tokens,
    /// the index product's rows of 64 a head and the pick's scores over heads of 64.
    #[test]
    fn the_indexers_heads_and_pools_are_the_configs() {
        let mut c = tiny();
        c["text_config"]["indexer_head_dim"] = 64.into();
        c["text_config"]["indexer_compress_ratio"] = 2.into();
        let m = super::super::lower::qwen4x(&describe(&c).unwrap(), vec![2000], 1100).unwrap();
        let ix = m.layers.iter().flatten().find_map(|op| if let Op::IdxMean { heads, dim, block, .. } = *op { Some((heads, dim, block)) } else { None });
        assert_eq!(ix, Some((4, 64, 2)));
        let pick = m.layers.iter().flatten().find_map(|op| if let Op::IdxPick(x) = op { Some(x.clone()) } else { None }).unwrap();
        assert_eq!((pick.dim(), pick.ratio, pick.top_k), (64, 2, 4));
        assert_eq!((m.width(Slot::Qr), m.width(Slot::Iq)), (5 * 64, 4 * 64));
        let mut c = tiny();
        c["text_config"]["indexer_head_dim"] = 100.into();
        assert!(super::super::lower::qwen4x(&describe(&c).unwrap(), vec![2000], 1100).unwrap_err().msg.contains("multiple of 32"));
    }

    /// The family's second random case (`logits::qwen4x_s2_random`: 2 streams, index heads of 64 over pools of 2, no n-gram embedding) lowers to the shared
    /// operations with those counts, and every matrix it multiplies by is a tensor of its manifest of the shape the lowering reads.
    #[test]
    fn the_second_random_case_lowers_with_its_counts() {
        let m: serde_json::Value = serde_json::from_str(&crate::logits::qwen4x_s2_manifest().unwrap()).unwrap();
        let c: serde_json::Value = serde_json::from_str(m["config"].as_str().unwrap()).unwrap();
        let d = super::super::lower::qwen4x(&describe(&c).unwrap(), vec![2000], 1100).unwrap();
        assert_eq!((d.streams(), d.width(Slot::S)), (2, 512));
        let ix = d.layers.iter().flatten().find_map(|op| if let Op::IdxMean { dim, block, .. } = *op { Some((dim, block)) } else { None });
        assert_eq!(ix, Some((64, 2)));
        assert!(!d.layers.iter().flatten().any(|op| matches!(op, Op::NgramRows(_))));
        let shapes: std::collections::HashMap<String, Vec<u64>> = m["tensors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| (t[0].as_str().unwrap().to_string(), t[2].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect()))
            .collect();
        for it in &d.mats {
            let rows: u64 = it.members.iter().map(|n| shapes.get(n).unwrap_or_else(|| panic!("{n}: not in the manifest"))[0]).sum();
            let cols = *shapes[&it.members[0]].last().unwrap();
            assert_eq!((rows as usize, cols as usize), (it.rows, it.cols), "{:?}", it.members);
        }
    }

    /// What the shared operations do not take is refused in words: a linear attention's silu gate, gated residuals over more streams than the kernels take.
    #[test]
    fn what_the_kernels_do_not_take_is_refused() {
        let mut c = tiny();
        c["text_config"]["output_gate_type"] = "silu".into();
        let e = super::super::lower::qwen4x(&describe(&c).unwrap(), vec![2000], 1100).unwrap_err().msg;
        assert!(e.contains("sigmoid gate"), "{e}");
        let mut c = tiny();
        c["text_config"]["hc_count"] = 5.into();
        let e = super::super::lower::qwen4x(&describe(&c).unwrap(), vec![2000], 1100).unwrap_err().msg;
        assert!(e.contains("streams"), "{e}");
    }

    /// The family's stream count is its config's: two streams lower to residual slots of two rows of the width and the inject's two weights a row (a model with
    /// no n-gram embedding: that block's kernels take four streams yet).
    #[test]
    fn the_streams_are_the_configs() {
        let mut c = tiny();
        c["text_config"]["hc_count"] = 2.into();
        let e = super::super::lower::qwen4x(&describe(&c).unwrap(), vec![2000], 1100).unwrap_err().msg;
        assert!(e.contains("four streams"), "{e}");
        (c["text_config"]["ple_layer_ids"], c["text_config"]["number_of_conv_states"]) = (serde_json::json!([]), 1.into());
        let m = super::super::lower::qwen4x(&describe(&c).unwrap(), vec![2000], 1100).unwrap();
        assert!(!m.layers.iter().flatten().any(|op| matches!(op, Op::NgramRows(_))));
        assert_eq!((m.streams(), m.width(Slot::S), m.width(Slot::Kv)), (2, 512, 2));
        let four = super::super::lower::qwen4x(&describe(&tiny()).unwrap(), vec![2000], 1100).unwrap();
        assert_eq!((four.streams(), four.width(Slot::S), four.width(Slot::Kv)), (4, 1024, 4));
    }
}
