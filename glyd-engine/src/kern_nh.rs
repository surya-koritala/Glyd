//! The interface a block of mixture-of-experts layers plugs into the executor by (`MoeBlock`, `MoeFactory`: src/moe_nh.rs makes Nemotron-H's blocks) and the
//! host hooks the other families' blocks share (`NhHost`). Nemotron-H's weights load by the executor's loader from its lowered description
//! (src/dynamic/nemotron.rs); its Mamba-2 layers run the layer kind's kernels (src/kern/mamba2.rs), its attention the attention kind's (src/dynamic/attn.rs).
//! The hand-written engine that used to be here is retired: its parity record is R7's fixtures and `paritycheck`'s (bits-identical at every level), and its
//! layer-by-layer tools (`nhlayers`, `nhtoy`, `nhoffload`) run on the executor. The layers' math: tools/nemotron_ref.py; the config and tensor names:
//! src/nemotron.rs.

use crate::dev::{Buf, Dev};
use crate::ffi::{err, Lib, Result};
use crate::kern35::Mat;
use crate::moe::MoeScratch;
use crate::nemotron::ConfigNemotronH;
use crate::weights::Layout;
use glyd_gpu::safetensors::Checkpoint;

/// A latent mixture-of-experts layer's mixer, which the engine's layers call and does not itself hold: router (full-width scores, the choice of `top_k`), the latent projections, the experts, the
/// shared expert. `h` [n, hidden] bf16 is the layer's normed input, `out` [n, hidden] bf16 its mixer's output (the engine adds it to the residual).
///
/// What a block does with the layer's tensors (transformers' NemotronHMoE): scores = sigmoid(h . gate^T) in fp32, the `top_k` of scores + e_score_correction_bias, their scores normed and times `scale`
/// (the router sees the full width); z = fc1_latent(h) [n, latent]; y = the chosen experts' down(relu2(up(z))) weighted, summed [n, latent]; out = fc2_latent(y) + shared_down(relu2(shared_up(h))).
/// With the mixture-of-experts core (GLY-101, `moe::MoeLayer`; src/moe_nh.rs): `route(h)`, `lin(fc1)`, `experts(z)`, `lin(fc2)`, `add_shared(h)`, all launches on the engine's stream and none reading a result on the host,
/// so `graphable` is true; its scratch is one for all the layers, in the engine's `moe_scratch`, made by the first block that runs (the engine's warm-up step runs before the graph is captured).
pub trait MoeBlock {
    /// # Safety
    /// Live device buffers of those sizes, used on the engine's stream.
    unsafe fn forward(&self, e: &dyn NhHost, h: u64, n: usize, out: u64) -> Result<()>;

    /// Whether the block takes an early guess of its experts (`early`): one whose experts are in a cache.
    fn early_wanted(&self) -> bool {
        false
    }

    /// A decode step's guess of the block's experts from `h` (one row: the router's input as it would be without the layer's attention), queued on the engine's
    /// stream before the attention; `forward` takes it up.
    ///
    /// # Safety
    /// `h` is a live device buffer of a row of the router's width.
    unsafe fn early(&self, _e: &dyn NhHost, _h: u64) -> Result<()> {
        Ok(())
    }
    /// Whether `forward` can be captured in a CUDA graph: no host read or copy that depends on the tokens.
    fn graphable(&self) -> bool {
        false
    }
    /// Device bytes it holds.
    fn bytes(&self) -> usize {
        0
    }
    /// The (rows, columns, whether 12-bit) of every pack it multiplies through the layers' `lin`, so that the engine sizes the library's workspace and its decoded-matrix scratch for them.
    fn pack_shapes(&self) -> Vec<(i64, i64, bool)> {
        Vec::new()
    }
    /// Where all its experts are on the GPU: the core's layer and the latent projections (the dynamic engine runs the layer itself, src/dynamic).
    fn resident(&self) -> Option<(&crate::moe::MoeLayer, Option<(&Mat, &Mat)>)> {
        None
    }
    /// The core's layer whose router leaves a call's choices in the experts' scratch (`MoeScratch::ids`), where every expert of it is on the GPU: what the corpus
    /// pass counts (src/hb_usage.rs). A resident block's; a block that makes its own launches over such a layer gives it too.
    fn core(&self) -> Option<&crate::moe::MoeLayer> {
        self.resident().map(|(l, _)| l)
    }
    /// The experts a token takes, where a call leaves its rows' choices in the experts' scratch (`MoeScratch::ids`) for a check to read
    /// (`Dyn::trace_routes`): a resident block's (`core`), or a block that routes into it itself.
    fn top_k(&self) -> Option<usize> {
        self.core().map(|l| l.cfg.router.top_k)
    }
    /// Whether a call reads the router's choices on the host (a layer of hummingbird experts in the expert cache, `MoeLayer::reads_host`): no graph captures a step with it.
    fn reads_host(&self) -> bool {
        self.resident().is_some_and(|(l, _)| l.reads_host())
    }
    /// The expert cache's store its experts are in (none: all on the GPU): the engine ends each step's statistics there (`ExpertStore::end_step`).
    fn store(&self) -> Option<std::rc::Rc<crate::offload::ExpertStore>> {
        self.resident().and_then(|(l, _)| l.store().cloned())
    }
}

/// What a block of experts asks of the engine it runs in (this one, or the dynamic engine): the GPU, the library, two buffers a mixture-of-experts layer leaves free (`mix` and `xc`, a chunk's rows at the
/// model's width at least), the chunk, the experts' scratch and the products.
pub trait NhHost {
    fn dev(&self) -> &Dev;
    fn lib(&self) -> &Lib;
    fn mix(&self) -> &Buf;
    fn xc(&self) -> &Buf;
    fn chunk(&self) -> usize;
    fn moe_scratch(&self) -> &std::cell::OnceCell<MoeScratch>;
    /// y [m, rows] = x [m, cols] W^T.
    ///
    /// # Safety
    /// Live device buffers of those sizes, on the engine's stream.
    unsafe fn lin(&self, w: &Mat, x: u64, m: usize, y: u64) -> Result<()>;
    /// y [m, n] = x [m, k] w^T in bf16, w a block's own rows on the GPU (rows it decodes for a call: Condor's shared expert).
    ///
    /// # Safety
    /// Live device buffers of those sizes, on the engine's stream.
    unsafe fn gemm(&self, _x: u64, _w: u64, _y: u64, _m: i32, _n: i32, _k: i32) -> Result<()> {
        err("this engine runs no product over a block's own rows")
    }
    /// A development build's stage marker (a profile's).
    fn mark(&self, _what: &'static str) {}
}

/// Makes layer `l`'s block of experts: the layer's number, the config, the checkpoint, the layout of the packs.
pub type MoeFactory<'a> = &'a dyn Fn(usize, &ConfigNemotronH, &Checkpoint, Layout) -> Result<Box<dyn MoeBlock>>;

#[cfg(test)]
mod tests {
    /// The Mamba-2 kind's kernels take Nemotron-H's released shape: its heads, state and chunk are the kernels' limits.
    #[test]
    fn the_kernels_limits_take_the_released_config() {
        use crate::kern::{MAMBA2_CHUNK, MAMBA2_HEAD, MAMBA2_STATE};
        let c = crate::nemotron::tests::parse();
        assert_eq!((c.m_head_dim, c.m_state, c.chunk), (MAMBA2_HEAD, MAMBA2_STATE, MAMBA2_CHUNK));
        assert!(c.inner() / c.m_groups <= 1024 && c.heads / c.kv_heads <= 16);
        let _ = crate::kern::norm_threads(c.hidden as i32).unwrap();
    }
}
