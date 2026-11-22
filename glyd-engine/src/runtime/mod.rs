//! The M2 runtime (`docs/design/runtime.md`): the layer kinds a model description is made of, the seam a vendor backend implements, and
//! `Ref`, the CPU f32 reference backend every other backend's kinds are checked against (R2, #135).
//!
//! Nothing here is on the serving path yet: the executor behind `ModelStep` (R4a) and the descriptions built from `config.json` (R3a)
//! come next. The kinds are data (`kinds`), a backend builds a `LayerImpl` per block (`backend`), and `reference` holds one plain-f32
//! function per kind with its tests against the per-family references (`*_ref.rs`) and against Hugging Face transformers.

pub mod backend;
pub mod context;
pub mod describe;
#[cfg(test)]
mod describe_tests;
pub mod kinds;
#[cfg(all(target_os = "macos", feature = "mlx"))]
pub mod mlx;
pub mod reference;
