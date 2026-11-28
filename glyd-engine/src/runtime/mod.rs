//! The M2 runtime (`docs/design/runtime.md`): the layer kinds a model description is made of, the seam a vendor backend implements, and
//! `Ref`, the CPU f32 reference backend every other backend's kinds are checked against (R2, #135).
//!
//! The kinds are data (`kinds`), a backend builds a `LayerImpl` per block and per piece around the blocks (`backend`), `reference` holds one
//! plain-f32 function per kind with its tests against the per-family references (`*_ref.rs`) and against Hugging Face transformers, and
//! `cuda` is the CUDA backend: every kind as a piece of the executor's (`dynamic`) own operations, which `kindcheck` holds to the reference.

pub mod backend;
pub mod config;
pub mod context;
pub mod cuda;
pub mod describe;
#[cfg(test)]
pub(crate) mod describe_tests;
pub mod kinds;
#[cfg(all(target_os = "macos", feature = "mlx"))]
pub mod mlx;
pub mod names;
pub mod reference;
pub mod seam;
