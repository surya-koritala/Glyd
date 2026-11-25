//! Muse Spark's text model (src/muse.rs) on the executor: its config's layer kinds lowered to the shared operations (`lower::muse`: each layer's attention, q, k, v
//! and the gate in one product, heads of 64 normed, the rotary on the window's layers, the output normed and gated; its experts in the latent, a projection in,
//! a centered norm, the experts with each output normed, the weighted sum, a projection out; the input-scaled norms and the residual gates; the normed embedding,
//! the final norm, the soft-cap of the logits times the output multiplier) and loaded by the executor's loader (`Dyn::new_placed`): the matrices bf16 as the
//! checkpoint has them, packed at the level asked (lossless by default), the small tensors made from their values as the operations read them
//! (`SmallForm`), each layer's experts the layer kind's block (`experts::held`), lossless whatever the level of the rest, the routers fp32 as they are. The
//! arithmetic is the kinds' (the attention's kernels/attn.cu, the norms' and residual gates' kernels/residual.cu, the experts' moe.rs) and kernels/muse.cu's;
//! src/muse_ref.rs is the CPU reference it is checked against (`musecheck`).

use super::{Dyn, ModelDesc, Opts, Placing, ResidentMoe, Stage};
use crate::dev::Dev;
use crate::ffi::Result;
use crate::kern_nh::MoeBlock;
use crate::muse::ConfigMuse;
use crate::weights::Layout;
use glyd_gpu::safetensors::Checkpoint;
use std::path::Path;

/// kernels/muse.cu's kernel the executor launches for the head (the experts' are moe.rs's, the attention's the attention kind's, the norms' and residual gates' the residual kinds').
pub struct KernMuse {
    pub(crate) softcap: crate::kt::muse_softcap,
}

impl KernMuse {
    pub fn load(dev: &Dev) -> Result<KernMuse> {
        Ok(KernMuse { softcap: dev.typed()? })
    }
}

pub fn load(dir: &Path, o: &Opts) -> Result<Dyn> {
    load_stage(dir, o, Stage::whole(ConfigMuse::load(dir)?.layers))
}

/// Stage `st` of the model (its layers; the embedding where it is the first, the final norm and the head where the last).
pub fn load_stage(dir: &Path, o: &Opts, st: Stage) -> Result<Dyn> {
    load_stage_hb(dir, o, st, None)
}

/// `load_stage` with its experts in the expert cache (`hc`: the stage's places and memory): a hummingbird directory's (src/hbg_store.rs), or the release's
/// lossless experts (src/tiering.rs `cached_tiered`); the stage's last layers' matrices in the computer's memory where the cache says so (src/dense_ram.rs).
/// Every layer is across the thread's group of GPUs where it has one (src/dynamic/tp.rs): this GPU's slice of a big matrix's rows, of each expert's.
pub fn load_stage_hb(dir: &Path, o: &Opts, st: Stage, hc: Option<&crate::hbg_store::HbCaching>) -> Result<Dyn> {
    let (kinds, label) = crate::runtime::describe::describe_dir(dir)?;
    let d = lowered(dir, &kinds, &st).map_err(|e| e.context(&label))?;
    // (a hummingbird directory, src/hb_pack.rs: the routed experts' codebook rows in place of the release's)
    let hb = crate::hb_pack::HbDir::open(dir)?;
    let store = hc.map(crate::hbg_store::store_for);
    let held = super::experts::Held { hb: hb.as_ref(), cache: hc.zip(store.as_ref()), ordinal: o.device };
    let hidden = kinds.desc.hidden;
    // (the experts lossless in the tiered layout, whatever the level of the rest)
    let experts = |ck: &Checkpoint, _: Layout| -> Result<Vec<Box<dyn MoeBlock>>> {
        super::experts::of_layers(&kinds, st.layers)
            .map(|(l, m, roles)| {
                crate::status::STATUS.beat();
                Ok(Box::new(ResidentMoe(super::experts::held(m, roles, hidden, ck, Layout::Tiered, Some((l, &held)))?)) as Box<dyn MoeBlock>)
            })
            .collect()
    };
    let placing = Placing { across: true, ram: crate::dense_ram::Placer::last(st.layers.0, st.layers.1, hc.map_or(0, |x| x.dense)) };
    let o = Opts { layout: Some(o.layout.unwrap_or(Layout::Tiered)), ..o.clone() };
    Dyn::new_placed(dir, d, None, st, &o, &experts, &placing)
}

/// Muse Spark's executor description for stage `st` of the model in `dir`, whose layer kinds are `kinds` (R3a): lowered (`lower::muse`), its end tokens and
/// its longest context the family's reader's.
pub fn lowered(dir: &Path, kinds: &crate::runtime::describe::Described, st: &Stage) -> Result<ModelDesc> {
    let c = ConfigMuse::load(dir)?;
    super::lower::muse(kinds, c.eos.clone(), c.max_pos, st)
}

#[cfg(all(test, feature = "dev"))]
mod tests {
    use super::super::{experts, unnamed, Op, Stage};
    use crate::moe::{Act, Router};
    use crate::muse::ConfigMuse;
    use crate::runtime::describe::describe_dir;

    fn stages(layers: usize) -> [Stage; 3] {
        let half = layers / 2;
        [Stage::whole(layers), Stage { layers: (0, half), first: true, last: false }, Stage { layers: (half, layers), first: false, last: true }]
    }

    /// The random model the logits checks run, lowered whole and as a split's stages: every matrix and small tensor its operations use is named, a tensor of
    /// the description's weight map (the weightless norm's ones alone made of no tensor), four matrices and nine small tensors a layer of the stage, and a
    /// block of experts a layer, in the layers' order, whose core layer is the kind's: the experts in the latent, the router over the model's width, the
    /// activation a kernel's of its own (their outputs are normed).
    #[test]
    fn the_random_model_lowers_to_named_tensors_and_its_kinds_blocks() {
        let dir = std::env::temp_dir().join(format!("glyd-muse-lowered-{}", std::process::id()));
        crate::muse_ref::random(&dir).unwrap();
        let c = ConfigMuse::load(&dir).unwrap();
        let (kinds, _) = describe_dir(&dir).unwrap();
        let names = kinds.names();
        for st in stages(c.layers) {
            let d = super::lowered(&dir, &kinds, &st).unwrap_or_else(|e| panic!("{}", e.msg));
            let n = st.layers.1 - st.layers.0;
            assert_eq!((unnamed(&d), d.mats.len(), d.small.len(), d.forms.len()), (None, 4 * n + 2, 1 + 9 * n + st.last as usize, 1 + 9 * n), "{st:?}");
            assert!(d.mats.iter().all(|it| it.members.iter().all(|m| names.contains(m))));
            assert!(d.small.iter().enumerate().all(|(i, s)| if i == 0 { s.is_empty() } else { names.contains(s) }));
            let blocks: Vec<usize> = d.layers.iter().flatten().filter_map(|op| if let Op::Moe(x) = op { Some(x.block) } else { None }).collect();
            assert_eq!(blocks, (0..n).collect::<Vec<_>>());
            let of: Vec<_> = experts::of_layers(&kinds, st.layers).collect();
            assert_eq!(of.iter().map(|b| b.0).collect::<Vec<_>>(), (st.layers.0..st.layers.1).collect::<Vec<_>>());
            for (_, m, _) in of {
                let k = experts::cfg(m, kinds.desc.hidden, c.inter).unwrap();
                assert_eq!((k.experts, k.router_hidden, k.hidden, k.inter, k.fused()), (c.experts, c.hidden, c.latent, c.inter, 0));
                assert_eq!((k.router, k.act), (Router::sigmoid(c.top_k, c.route_scale), Act::SwiGlu { limit: Some(f32::INFINITY) }));
                k.check().unwrap();
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
