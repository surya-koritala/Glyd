//! Every pre-norm dense decoder of the config types the description builder's `llama` builder reads (`runtime::describe::LLAMA`) on the
//! executor (R3c, #139): the config's layer kinds lowered to the shared operations (`lower::dense`: the attention kind, the gated SiLU MLP, the RMS norms),
//! loaded by the executor's own loader at any level its products take (penguin's lossless packs, kestrel's, swift's and wren's), served by the scheduler with
//! paged keys and values (12-bit pages where the tables are set).

use super::{Dyn, ModelDesc, Opts};
use crate::ffi::{Error, Result};
use std::path::Path;

/// The executor description's family name.
pub const FAMILY: &str = "plain dense";

/// Whether a config.json's text is one the `llama` builder reads (its text model's type).
pub fn is_dense(text: &str) -> bool {
    crate::runtime::config::ModelConfig::parse(text).is_ok_and(|c| crate::runtime::describe::LLAMA.contains(&c.text_type.as_str()))
}

/// The end tokens (generation_config.json's, else the config's) and the longest context of the model in `dir`.
fn ends(dir: &Path) -> Result<(Vec<i64>, usize)> {
    let p = dir.join("config.json");
    let text = std::fs::read_to_string(&p).map_err(|e| Error(format!("{}: {e}", p.display())))?;
    let c = crate::runtime::config::ModelConfig::parse(&text).map_err(|e| e.context(p.display()))?;
    let generation = std::fs::read_to_string(dir.join("generation_config.json")).ok();
    Ok((crate::weights::eos_ids(&c.top, generation.as_deref()), c.shared.max_position_embeddings.unwrap_or(crate::support::DEFAULT_CTX)))
}

/// The executor's description of the model in `dir`, from its config.json alone (its tensors' shapes the description's: a packed directory's too).
pub fn lowered(dir: &Path) -> Result<ModelDesc> {
    let (kinds, label) = crate::runtime::describe::describe_dir(dir)?;
    let (eos, max_pos) = ends(dir)?;
    super::lower::dense(&kinds, eos, max_pos).map_err(|e| e.context(&label))
}

/// The model in `dir` on the executor: a Hub-format directory packed here at `o.layout` (the GPU's where none), a glyd-v7 directory read as saved.
pub fn load(dir: &Path, o: &Opts) -> Result<Dyn> {
    Dyn::new(dir, lowered(dir)?, None, o)
}
