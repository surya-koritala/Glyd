//! Every pre-norm dense decoder of the config types the description builder's `llama` builder reads (`runtime::describe::LLAMA`) on the
//! executor (R3c, #139), but one with an attention bias or a scaled rope, which `lower::dense` refuses by name though `is_dense` says yes to its
//! config: the config's layer kinds lowered to the shared operations (`lower::dense`: the attention kind, the gated SiLU MLP, the RMS norms),
//! loaded by the executor's own loader at any level its products take (penguin's lossless packs, kestrel's, swift's and wren's), served by the scheduler with
//! paged keys and values (12-bit pages where the tables are set).

use super::{Dyn, ModelDesc, Opts};
use crate::ffi::Result;
use crate::runtime::config::ModelConfig;
use std::path::Path;

/// The executor description's family name.
pub const FAMILY: &str = "plain dense";

/// Whether a config.json's text is one the `llama` builder reads (its text model's type).
pub fn is_dense(text: &str) -> bool {
    ModelConfig::parse(text).is_ok_and(|c| c.is_dense())
}

/// The end tokens (generation_config.json's, else the config's) and the longest context of the model in `dir`, whose config.json is `c`.
fn ends(dir: &Path, c: &ModelConfig) -> (Vec<i64>, usize) {
    let generation = ModelConfig::generation(dir);
    (crate::weights::eos_ids(&c.top, generation.as_deref()), c.shared.max_position_embeddings.unwrap_or(crate::support::DEFAULT_CTX))
}

/// The executor's description of the model in `dir`, from its config.json alone (its tensors' shapes the description's: a packed directory's too).
pub fn lowered(dir: &Path) -> Result<ModelDesc> {
    lowered_of(dir, &ModelConfig::load(dir)?)
}

/// `lowered` of a config.json parsed once.
pub fn lowered_of(dir: &Path, c: &ModelConfig) -> Result<ModelDesc> {
    let (kinds, label) = crate::runtime::describe::describe_dir_of(dir, c)?;
    let (eos, max_pos) = ends(dir, c);
    super::lower::dense(&kinds, eos, max_pos).map_err(|e| e.context(&label))
}

/// The model in `dir` on the executor: a Hub-format directory packed here at `o.layout` (the GPU's where none), a glyd-v7 directory read as saved.
pub fn load(dir: &Path, o: &Opts) -> Result<Dyn> {
    load_of(dir, &ModelConfig::load(dir)?, o)
}

/// `load` of a config.json parsed once (`m`: `dir`'s). A model whose feed-forwards are mixtures of experts (Mellum's) loads each layer's as the Moe kind's
/// block (`experts::block`: its router, its experts and a shared expert where it has one, at the level of the other matrices), every expert on the GPU.
pub fn load_of(dir: &Path, m: &ModelConfig, o: &Opts) -> Result<Dyn> {
    let (kinds, label) = crate::runtime::describe::describe_dir_of(dir, m)?;
    let (eos, max_pos) = ends(dir, m);
    let d = super::lower::dense(&kinds, eos, max_pos).map_err(|e| e.context(&label))?;
    if super::experts::of_layers(&kinds, (0, d.layers.len()), 2).next().is_none() {
        return Dyn::new(dir, m, d, None, o);
    }
    let st = super::Stage::whole(d.layers.len());
    let held = super::experts::Held { hb: None, cache: None, ordinal: o.device, f8: None, dense: None, codebooks: None, codes: None };
    let hidden = kinds.desc.hidden;
    let experts = |ck: &glyd_gpu::safetensors::Checkpoint, level: crate::weights::Layout| -> Result<Vec<Box<dyn super::experts::MoeBlock>>> {
        super::experts::of_layers(&kinds, st.layers, 2)
            .map(|(ix, moe, roles)| {
                crate::status::STATUS.beat(); // (a block of experts takes seconds to read)
                super::experts::block(moe, roles, hidden, ck, level, (ix, &held))
            })
            .collect()
    };
    Dyn::new_placed(dir, m, d, None, st, o, &experts, &super::Placing::default()).map_err(|e| e.context(&label))
}
