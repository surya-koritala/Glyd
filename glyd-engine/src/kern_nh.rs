//! Nemotron-H's kernel limits: its Mamba-2 layers run the layer kind's kernels (src/kern/mamba2.rs), its attention the attention kind's (src/dynamic/attn.rs), its weights load
//! by the executor's loader from its lowered description (src/dynamic/nemotron.rs). The hand-written engine that used to be here is retired: its parity record is R7's fixtures
//! and `paritycheck`'s (bits-identical at every level), and its layer-by-layer tools (`nhlayers`, `nhtoy`, `nhoffload`) run on the executor. The layers' math:
//! tools/nemotron_ref.py; the config and tensor names: src/nemotron.rs. The interface a block of experts plugs into the executor by (`MoeBlock`) and the host hooks it
//! calls (`NhHost`) are the Moe kind's: src/dynamic/experts.rs.

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
