//! Gemma 4's kernels (kernels/gemma4.cu) with typed launches (`KernG4`), the next product's prefetch's argument and setting (`kern::NextPf`, `kern::pf_kib`), which the executor's Gemma 4 runs.

use crate::dev::Dev;
use crate::ffi::Result;

/// kernels/gemma4.cu's kernels, each by its type (`crate::kt`): a launch with arguments of other types than the kernel's does not compile.
pub struct KernG4 {
    pub(crate) norm: crate::kt::g4_rmsnorm,
    pub(crate) embed_norm: crate::kt::g4_embed_norm,
    pub(crate) add_norm: crate::kt::g4_add_norm,
    pub(crate) gelu: crate::kt::g4_gelu_mul,
    pub(crate) softcap: crate::kt::g4_softcap,
    pub(crate) draft_in: crate::kt::g4_draft_in,
}

impl KernG4 {
    pub fn load(dev: &Dev) -> Result<KernG4> {
        Ok(KernG4 { norm: dev.typed()?, embed_norm: dev.typed()?, add_norm: dev.typed()?, gelu: dev.typed()?, softcap: dev.typed()?, draft_in: dev.typed()? })
    }
}
