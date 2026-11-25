//! Gemma 4's kernels (kernels/gemma4.cu) with typed launches (`KernG4`), its prefetch's argument (`G4Pf`) and setting (`pf_kib`), which the executor's Gemma 4 runs.

use crate::dev::Dev;
use crate::ffi::{err, Result};

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

/// kernels/gemma4.cu's G4Pf: a small kernel's spare blocks prefetch into L2 the start of the first-wave blocks of the skinny product that comes next (GLY-312).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct G4Pf {
    pub(crate) data: u64,
    pub(crate) scales: u64,
    pub(crate) dstride: i64,
    pub(crate) sstride: i64,
    pub(crate) dhead: i32,
    pub(crate) shead: i32,
    pub(crate) regions: i32,
    pub(crate) shift: i32,
}

/// GLYD_G4_PF: "a,n,g" KiB (unset: 128,32,8, the best measured on an RTX 4080 SUPER; 0,0,0 none).
pub(crate) fn pf_kib(v: Option<&str>) -> Result<[usize; 3]> {
    let Some(v) = v.filter(|v| !v.is_empty()) else { return Ok([128, 32, 8]) };
    let k: Vec<Option<usize>> = v.split(',').map(|x| x.trim().parse().ok()).collect();
    match k[..] {
        [Some(a), Some(n), Some(g)] => Ok([a, n, g]),
        _ => err(format!("the prefetch setting {v}: three numbers of KiB, a,n,g")),
    }
}
