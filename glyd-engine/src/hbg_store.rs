//! Hummingbird's experts of every mixture of experts in the expert cache (docs/design/hummingbird.md, #429): a layer's experts in the computer's memory, both bit widths, with places for the busiest on the GPU (src/offload.rs
//! `ExpertStore`), the misses copied in or computed by the CPU (src/cpu_hbg.rs); where the computer's memory does not hold them all, some of each group in memory and the rest read from the layer's expert file as the router
//! asks for them, in direct reads (src/disk.rs: no page cache). A family's loader builds such a layer with `cached_layer` from a hummingbird directory (src/hb_pack.rs) where its plan puts experts in RAM (`HbCaching`).
//!
//! The store holds a decoder layer l's groups as its layers 2 l and 2 l + 1 (the two-bit experts, the three-bit ones); `MoeLayer::hb_cached` (src/moe.rs) runs a call through it.

use crate::cpu_hbg::Finish;
use crate::disk::{Aligned, DirectFile};
use crate::ffi::{err, Error, Result};
use crate::hb_pack::{HbDir, LayerRec};
use crate::moe::{Act, HbCache};
use crate::moe_hbg::{group_index, HbgExperts, HbgHost, HbgPlace, HbgShape};
use crate::offload::{DiskInG, ExpertStore, HostHalf, HostMemory, Offload, PackIn, Reader};
use glyd_gpu::codebook::cb_strides;
use glyd_gpu::safetensors::File;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

/// The expert cache a stage's hummingbird layers are given: places a layer at the mean of the widths (split between the two by their share of the layer's experts), the computer's memory for the experts where it is less than all
/// of them (None: it holds them all), and the store's policy (copy or compute, the CPU's threads).
#[derive(Clone, Copy, Debug)]
pub struct HbCaching {
    pub places: usize,
    pub ram: Option<HostMemory>,
    pub offload: Offload,
}

/// A store for a stage's layers, as `c` says.
pub fn store_for(c: &HbCaching) -> Rc<ExpertStore> {
    let store = ExpertStore::new();
    store.set_offload(c.offload);
    store.set_host_memory(c.ram);
    store
}

/// Spans of an expert's tensors read in one direct read where they are this close in the file.
const GAP: u64 = 64 << 10;

/// Where one of an expert's six tensors is: in `meta` (scales, codebooks) or `data` (codes) of its place, at `to`, `len` bytes, from the file at `at`.
#[derive(Clone, Copy, Debug, Default)]
struct Seg {
    meta: bool,
    to: u32,
    len: u32,
    at: u64,
}

thread_local! {
    static BOUNCE: RefCell<Aligned> = const { RefCell::new(Aligned::new()) };
}

/// A layer's expert file (`experts-BBBB.safetensors`) read an expert at a time into a place of the host tier: each expert's six tensors (its `gate_up` and `down_t` codes, scales and codebooks), found once from the file's header.
pub struct HbgDisk {
    file: DirectFile,
    experts: Vec<(u8, [Seg; 6])>,
    places: [HbgPlace; 2],
}

impl HbgDisk {
    /// The expert file of the layer `rec` of `dir`: every expert's tensors held to its bits and the layer's shape.
    pub fn open(dir: &HbDir, rec: &LayerRec) -> Result<HbgDisk> {
        let path = dir.dir.join(&rec.file);
        let f = File::open(&path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let up = if rec.gated { 2 * rec.inter } else { rec.inter };
        let places = [HbgPlace::new(2, rec.width, up, rec.inter)?, HbgPlace::new(3, rec.width, up, rec.inter)?];
        let mut experts = Vec::with_capacity(rec.bits.len());
        for (e, &bits) in rec.bits.iter().enumerate() {
            let g = group_index(bits);
            let p = &places[g];
            let (cs, ss, bs) = cb_strides(bits, rec.width);
            let mut segs = [Seg::default(); 6];
            let mut i = 0;
            for (m, rows, codes, scales, book) in [("gate_up", up, 0, 0, p.up_book), ("down_t", rec.inter, p.down_codes, p.down_scales, p.down_book)] {
                for (k, bytes, meta, to) in [("codes", rows * cs, false, codes), ("scales", rows * ss, true, scales), ("codebook", rows * bs, true, book)] {
                    let name = format!("e{e:04}.{m}.{k}");
                    let t = f.get(&name).ok_or_else(|| Error(format!("{}: no {name}", path.display())))?;
                    if t.bytes != bytes as u64 {
                        return err(format!("{}: {name} is {} bytes, where expert {e}'s {k} at {bits} bits are {bytes}", path.display(), t.bytes));
                    }
                    segs[i] = Seg { meta, to: to as u32, len: bytes as u32, at: t.offset };
                    i += 1;
                }
            }
            experts.push((g as u8, segs));
        }
        let file = DirectFile::open(&path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        Ok(HbgDisk { file, experts, places })
    }

    /// Expert `e` (its number in the layer) into a place's bytes, `data` and `meta` of its group's sizes, the padding zeroed.
    pub fn read(&self, e: usize, data: &mut [u8], meta: &mut [u8]) -> Result<()> {
        let (g, segs) = self.experts.get(e).ok_or_else(|| Error(format!("expert {e} of {}", self.experts.len())))?;
        let place = &self.places[*g as usize];
        if data.len() != place.data || meta.len() != place.meta {
            return err(format!("a place of {} + {} bytes for an expert of {} + {}", data.len(), meta.len(), place.data, place.meta));
        }
        data.fill(0);
        meta.fill(0);
        let mut order = [0usize, 1, 2, 3, 4, 5];
        order.sort_by_key(|&i| segs[i].at);
        BOUNCE.with(|b| -> Result<()> {
            let mut bounce = b.borrow_mut();
            let mut i = 0;
            while i < order.len() {
                let first = segs[order[i]];
                let (mut end, mut j) = (first.at + first.len as u64, i + 1);
                while j < order.len() && segs[order[j]].at <= end + GAP {
                    end = end.max(segs[order[j]].at + segs[order[j]].len as u64);
                    j += 1;
                }
                let span = self.file.read_span(first.at, (end - first.at) as usize, &mut bounce).map_err(|x| Error(x.to_string()))?;
                for &k in &order[i..j] {
                    let sg = segs[k];
                    let from = (sg.at - first.at) as usize;
                    let to = if sg.meta { &mut *meta } else { &mut *data };
                    to[sg.to as usize..][..sg.len as usize].copy_from_slice(&span[from..from + sg.len as usize]);
                }
                i = j;
            }
            Ok(())
        })
    }

    /// The reader of a group's experts for the store (`offload::Reader`): the group's k-th expert is `members[k]` of the layer.
    pub fn reader(self: &Arc<HbgDisk>, members: Vec<u16>) -> Reader {
        let disk = self.clone();
        Arc::new(move |local, data, meta| {
            let e = *members.get(local).ok_or_else(|| Error(format!("expert {local} of a group of {}", members.len())))?;
            disk.read(e as usize, data, meta)
        })
    }
}

/// Each expert's group (its bits: 0 two, 1 three) and number in it, in the order of the experts (as `HbgHost::of` gives them), and each group's experts.
fn groups_of(bits: &[u32]) -> (Vec<u8>, Vec<u16>, [Vec<u16>; 2]) {
    let (mut group_of, mut local_of, mut members) = (Vec::with_capacity(bits.len()), Vec::with_capacity(bits.len()), [Vec::new(), Vec::new()]);
    for (e, &b) in bits.iter().enumerate() {
        let g = group_index(b);
        group_of.push(g as u8);
        local_of.push(members[g].len() as u16);
        members[g].push(e as u16);
    }
    (group_of, local_of, members)
}

/// Decoder layer `l`'s experts of the hummingbird directory `dir` in the expert cache `store`: `places` places on the GPU (`moe_hb::split_slots` between the widths), every expert in the computer's memory (pinned), or where the
/// store has less memory than all its experts (`HostMemory`) its share of slots of each group there and the rest read from the expert file (`HbgDisk`); the host computes a pair with the family's activation `act`, finished as
/// `finish` says. Returns the layer's experts in their places (for `MoeLayer::new_hb`) and the cache (`MoeLayer::with_hb_cache`); `ordinal` is the GPU's, `top_k` the layer's choices a token.
#[allow(clippy::too_many_arguments)]
pub fn cached_layer(
    dir: &HbDir,
    l: usize,
    places: usize,
    store: &Rc<ExpertStore>,
    act: Act,
    finish: Finish,
    ordinal: i32,
    top_k: usize,
) -> Result<(HbgExperts, HbCache)> {
    let rec = dir.layer(2 * l + 1).ok_or_else(|| Error(format!("{}: no experts for decoder layer {l} (block {})", dir.dir.display(), 2 * l + 1)))?.clone();
    let (group_of, local_of, members) = groups_of(&rec.bits);
    let counts = [members[0].len(), members[1].len()];
    let slots = crate::moe_hb::split_slots(places.max(1), counts);
    let shape = HbgShape { experts: rec.bits.len(), width: rec.width, rot: rec.rot, inter: rec.inter, gated: rec.gated };
    let up = shape.up_rows();
    let gplaces = [HbgPlace::new(2, rec.width, up, rec.inter)?, HbgPlace::new(3, rec.width, up, rec.inter)?];
    let disk = store.host_memory().filter(|m| m.bytes < m.total);
    let (experts, host) = match disk {
        // (the computer's memory holds some of the experts: the signs alone are read now, each expert when the router first asks for it)
        Some(_) => (HbgExperts::pooled_shape(shape, gplaces, counts, slots, &dir.read_signs(2 * l + 1)?)?, None),
        None => {
            let h = HbgHost::of(&dir.read_layer(2 * l + 1)?)?;
            (HbgExperts::pooled(&h, slots)?, Some(h))
        }
    };
    let mut layers = [None, None];
    let reader = match disk {
        Some(_) => Some(Arc::new(HbgDisk::open(dir, &rec)?)),
        None => None,
    };
    for g in 0..2 {
        if counts[g] == 0 {
            continue;
        }
        let at = 2 * l + g;
        let stack = experts.group(g).ok_or_else(|| Error(format!("no group {g} of the layer's places")))?;
        let (dev, _) = stack.places();
        let place = gplaces[g];
        match (&host, &reader) {
            (Some(h), _) => {
                let grp = h.groups[g].as_ref().ok_or_else(|| Error(format!("no group {g} in the layer")))?;
                store.add_layer(
                    at,
                    counts[g],
                    slots[g],
                    &[PackIn { bytes: [&grp.data, &grp.meta], stride: [place.data as u64, place.meta as u64], dev }],
                    Some(HostHalf::Hbg { place, act, finish: finish.clone() }),
                )?;
            }
            (None, Some(d)) => store.add_layer_on_disk_g(
                at,
                counts[g],
                slots[g],
                DiskInG { place, act, finish: finish.clone(), dev, ram: store.host_slots(counts[g], place.bytes())?, read: d.reader(members[g].clone()) },
            )?,
            (None, None) => unreachable!("a layer read whole or from its file"),
        }
        layers[g] = Some(at);
    }
    let cache = HbCache::new(store.clone(), layers, group_of, local_of, ordinal, top_k, rec.width)?;
    Ok((experts, cache))
}

/// A decoder layer's hummingbird experts as a family's loader builds its MoE layer from them: all on the GPU, or in the expert cache.
pub enum HbLayer {
    Gpu(HbgExperts),
    Cached(HbgExperts, HbCache),
}

impl HbLayer {
    /// The layer: `MoeLayer::new_hb` with the router, its bias and per-expert scale, and the shared expert; in the cache, `with_hb_cache` too.
    pub fn into_layer(
        self,
        cfg: crate::moe::MoeCfg,
        router: crate::dev::Buf,
        bias: Option<crate::dev::Buf>,
        escale: Option<crate::dev::Buf>,
        shared: Option<crate::moe::Shared>,
    ) -> Result<crate::moe::MoeLayer> {
        match self {
            HbLayer::Gpu(x) => crate::moe::MoeLayer::new_hb(cfg, router, bias, escale, x, shared),
            HbLayer::Cached(x, c) => crate::moe::MoeLayer::new_hb(cfg, router, bias, escale, x, shared)?.with_hb_cache(c),
        }
    }
}

/// Decoder layer `l`'s experts where `hb` is a hummingbird directory (None where it is not): in the expert cache where the stage is given one (`cached`: its caching and its store), else all on the GPU (`HbDir::experts_for`, which
/// refuses every layer across GPUs). `act` and `finish` are the family's (the host's pairs); `ordinal` the GPU's, `top_k` the layer's choices a token. A `Finish::Normed` given with no signs takes the layer's.
#[allow(clippy::too_many_arguments)]
pub fn hb_layer(
    hb: Option<&HbDir>,
    l: usize,
    across: bool,
    cached: Option<(&HbCaching, &Rc<ExpertStore>)>,
    act: Act,
    finish: Finish,
    ordinal: i32,
    top_k: usize,
) -> Result<Option<HbLayer>> {
    let Some(d) = hb else { return Ok(None) };
    let Some((c, store)) = cached else { return HbDir::experts_for(Some(d), l, across).map(|x| x.map(HbLayer::Gpu)) };
    if across {
        return err("a hummingbird directory's experts in the expert cache are whole on a GPU: every layer across GPUs takes the penguin level so far (cut the layers over the GPUs: --split)");
    }
    let finish = match finish {
        Finish::Normed { signs, w, eps, .. } if signs.is_empty() => {
            let rec = d.layer(2 * l + 1).ok_or_else(|| Error(format!("{}: no experts for decoder layer {l}", d.dir.display())))?;
            Finish::Normed { signs: d.read_signs(2 * l + 1)?, rot: rec.rot, w, eps }
        }
        f => f,
    };
    let (x, cache) = cached_layer(d, l, c.places, store, act, finish, ordinal, top_k)?;
    Ok(Some(HbLayer::Cached(x, cache)))
}

/// The bytes of a decoder layer's experts at their bits as the store holds them (each group's place size times its experts): what the planner counts the RAM and the disk by.
pub fn layer_store_bytes(rec: &LayerRec) -> Result<u64> {
    let up = if rec.gated { 2 * rec.inter } else { rec.inter };
    let places = [HbgPlace::new(2, rec.width, up, rec.inter)?, HbgPlace::new(3, rec.width, up, rec.inter)?];
    Ok(rec.bits.iter().map(|&b| places[group_index(b)].bytes()).sum())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny GLM-5.3-Flash packed at the hummingbird level: each expert read from its layer's file into a place of its group (`HbgDisk`, the store's reader) is, byte for byte, its place in the layer read whole (`HbgHost::of`),
    /// the padding zeroed; a place of the other group's size is refused; the groups and numbers are `HbgHost::of`'s.
    #[test]
    fn an_expert_read_from_its_file_is_its_place_in_the_layer() {
        let dir = std::env::temp_dir().join(format!("glyd-hbgdisk-{}", std::process::id()));
        let src = dir.join("src");
        let d = crate::hb_pack::tests::release(&src, "glm53flash_model");
        let mut x = 5u64;
        let u = crate::hb_codec::Usage {
            model: "tiny".into(),
            tokens: 1,
            layers: crate::hb_pack::moe_layers(&d)
                .unwrap()
                .iter()
                .map(|m| crate::hb_codec::LayerUse {
                    block: m.block,
                    counts: (0..m.experts)
                        .map(|_| {
                            x ^= x << 13;
                            x ^= x >> 7;
                            x ^= x << 17;
                            x % 100
                        })
                        .collect(),
                })
                .collect(),
        };
        std::fs::write(dir.join("usage.json"), u.to_json()).unwrap();
        crate::hb_pack::run(src.to_str().unwrap(), &dir.join("out"), 2, &dir.join("usage.json")).unwrap();
        let hb = HbDir::open(&dir.join("out")).unwrap().unwrap();
        for rec in &hb.layers {
            let whole = HbgHost::of(&hb.read_layer(rec.block).unwrap()).unwrap();
            let (group_of, local_of, members) = groups_of(&rec.bits);
            assert_eq!((group_of.clone(), local_of.clone()), (whole.group_of.clone(), whole.local_of.clone()));
            let disk = Arc::new(HbgDisk::open(&hb, rec).unwrap());
            for g in 0..2 {
                let Some(grp) = whole.groups[g].as_ref() else { continue };
                let read = disk.reader(members[g].clone());
                let p = grp.place;
                for k in 0..grp.experts {
                    let (mut data, mut meta) = (vec![0xAAu8; p.data], vec![0xAAu8; p.meta]);
                    read(k, &mut data, &mut meta).unwrap();
                    assert_eq!(&data[..], &grp.data[k * p.data..(k + 1) * p.data], "block {} group {g} expert {k}", rec.block);
                    assert_eq!(&meta[..], &grp.meta[k * p.meta..(k + 1) * p.meta]);
                }
                let (mut data, mut meta) = (vec![0u8; p.data + 16], vec![0u8; p.meta]);
                assert!(read(0, &mut data, &mut meta).is_err());
            }
            assert_eq!(layer_store_bytes(rec).unwrap(), whole.bytes());
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
