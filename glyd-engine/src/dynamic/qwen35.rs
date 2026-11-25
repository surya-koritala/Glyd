//! Qwen3.5 (dense: 9B, 27B; Qwen3.6-27B, Qwen3.8-27B) as a description: Gated DeltaNet layers and gated full-attention layers, the (1 + w) norms,
//! the SiLU MLP, the final norm on the last row (models/qwen3_5; the arithmetic is the kernels', as src/kern35.rs names them). It loads from its layer kinds
//! lowered (`lower::qwen35`), the executor's loader reading the tensors the lowered description names (the small matrices dense, the fp32 parameters, the
//! gated norm's weight as vLLM holds it).

#[cfg(any(test, feature = "dev"))]
use super::attn::{AttnOp, GateAt, QkNorm, RopeAt};
#[cfg(any(test, feature = "dev"))]
use super::Slot;
use super::{Dyn, ModelDesc, Opts, Placing, Stage};
#[cfg(test)]
use super::{NormW, Op, Ring, Rope};
#[cfg(test)]
use crate::ffi::err;
use crate::ffi::{Error, Result};
use crate::kern_nh::MoeBlock;
use crate::qwen35::Kind;
use crate::runtime::describe::Described;
use crate::runtime::kinds::{Body, Ffn};
use crate::weights::Layout;
use glyd_gpu::safetensors::Checkpoint;
use std::path::Path;

pub fn load(dir: &Path, o: &Opts) -> Result<Dyn> {
    load_stage(dir, o, Stage::whole(ends(dir)?.0))
}

/// The model's layers and end-of-sequence ids (the dynamic route's `split::Loader::ends`): the dense layout's config's (`Config35`), the layout with experts'
/// by its description (a block of attention and one of its feed-forward a layer) and its config's ids, as `load_experts` takes them.
pub fn ends(dir: &Path) -> Result<(usize, Vec<i64>)> {
    match with_experts(dir) {
        Some((kinds, _)) => Ok((kinds.desc.blocks.len() / 2, experts_config(dir)?.0)),
        None => crate::qwen35::Config35::load(dir).map(|c| (c.layers, c.eos)),
    }
}

/// The model of the config `c` as the lossy route packs and sizes it (src/pack.rs `qwen35_plan`, src/plan.rs `lossy_plan_inner`), dense or with experts alike:
/// its description lowered whole (`lower::qwen35`: what the loader runs, its states and rings, its longest context), the matrices a level's save packs and the
/// tensors it keeps, in load order (`pack::described`: a layer's blocks of experts' too).
pub fn saved(c: &crate::runtime::config::ModelConfig) -> Result<(ModelDesc, Vec<crate::qwen35::Pack35>, Vec<crate::qwen35::Tensor35>)> {
    let d = crate::runtime::describe::describe_config(c)?;
    let max_pos = c.shared.max_position_embeddings.ok_or_else(|| Error("no max_position_embeddings".into()))?;
    let low = super::lower::qwen35(&d, Vec::new(), max_pos, &Stage::whole(d.desc.blocks.len() / 2))?;
    let (packs, kept) = crate::pack::described(&d, &low)?;
    Ok((low, packs, kept))
}

/// The description of the model in `dir` and its label, where a layer's feed-forward is a mixture of experts (None: the dense layout, or no description).
fn with_experts(dir: &Path) -> Option<(Described, String)> {
    crate::runtime::describe::describe_dir(dir).ok().filter(|(k, _)| k.desc.blocks.iter().any(|b| matches!(b.body, Body::Ffn(Ffn::Moe(_)))))
}

/// The layout with experts' end-of-sequence ids (generation_config.json's, else its text model's config's, else the top level's) and its longest context.
fn experts_config(dir: &Path) -> Result<(Vec<i64>, usize)> {
    let cfg = crate::runtime::config::ModelConfig::load(dir)?;
    let generation = std::fs::read_to_string(dir.join("generation_config.json")).ok();
    let mut eos = crate::runtime::config::eos_ids(&cfg.text, generation.as_deref());
    if eos.is_empty() {
        eos = crate::runtime::config::eos_ids(&cfg.top, None);
    }
    let max_pos = cfg.shared.max_position_embeddings.ok_or_else(|| Error("no max_position_embeddings".into()))?;
    Ok((eos, max_pos))
}

/// The dynamic route's stage (src/dynamic/split.rs `StageLoad`): `load_stage`, the layout with experts' experts in the expert cache where `caching` is given
/// (`load_experts`); the dense layout has none to keep there, and refuses a caching.
pub fn stage(dir: &Path, o: &Opts, st: Stage, caching: Option<&crate::hbg_store::HbCaching>) -> Result<Dyn> {
    match (with_experts(dir), caching) {
        (Some((kinds, label)), _) => load_experts(dir, o, st, &kinds, caching).map_err(|e| e.context(&label)),
        (None, Some(_)) => crate::ffi::err(format!("{}: the dense Qwen3.5 layout has no experts to keep in the expert cache", dir.display())),
        (None, None) => load_stage(dir, o, st),
    }
}

/// Stage `st` of it (a model split across GPUs: its layers, the embedding where it is the first, the head where the last; a layer of another
/// stage is an empty one here), on the executor's loader (`Dyn::new_stage`: the b and a gates' small parts held as saved, A_log and the gated norm's
/// weight in fp32, a saved directory's packs as saved). A model whose feed-forwards are mixtures of experts (the layout with experts, `Qwen3_5MoeForCausalLM`) loads by its
/// layer kinds alone (`load_experts`).
pub fn load_stage(dir: &Path, o: &Opts, st: Stage) -> Result<Dyn> {
    if let Some((kinds, label)) = with_experts(dir) {
        return load_experts(dir, o, st, &kinds, None).map_err(|e| e.context(&label));
    }
    let c = crate::qwen35::Config35::load(dir)?;
    Dyn::new_stage(dir, lowered(dir, &c, &st)?, None, st, o)
}

/// The layout with experts: its kinds lowered (`lower::qwen35`), each layer's experts the Moe kind's block (`experts::block`: the softmax router, the experts
/// and the gated shared expert at the level of the other matrices), the stage's layers' in order: every expert on the GPU, or in the expert cache as `caching`
/// says where it is given (src/tiering.rs: its places a layer on the GPU, every expert in the computer's memory, the cache's layer a block's number among the
/// model's blocks of experts), and the stage's last layers' matrices in the computer's memory where it says so (src/dense_ram.rs).
fn load_experts(dir: &Path, o: &Opts, st: Stage, kinds: &Described, caching: Option<&crate::hbg_store::HbCaching>) -> Result<Dyn> {
    let (eos, max_pos) = experts_config(dir)?;
    let d = super::lower::qwen35(kinds, eos, max_pos, &st)?;
    let store = caching.map(crate::hbg_store::store_for);
    let held = super::experts::Held { hb: None, cache: caching.zip(store.as_ref()), ordinal: o.device, f8: None, dense: None, codebooks: None };
    let hidden = kinds.desc.hidden;
    let experts = |ck: &Checkpoint, level: Layout| -> Result<Vec<Box<dyn MoeBlock>>> {
        super::experts::of_layers(kinds, st.layers, 2)
            .map(|(ix, m, roles)| {
                crate::status::STATUS.beat(); // (a block of experts takes seconds to read)
                super::experts::block(m, roles, hidden, ck, level, (ix, &held))
            })
            .collect()
    };
    let placing = Placing { across: false, ram: crate::dense_ram::Placer::last(st.layers.0, st.layers.1, caching.map_or(0, |x| x.dense)) };
    Dyn::new_placed(dir, d, None, st, o, &experts, &placing)
}

/// A layer's matrices and small tensors as the loader holds them: a linear-attention layer's qkvz, ba and out products and its MLP's two; its input and
/// post-attention norms, the convolution, A_log, dt_bias and the gated norm's weight. A full-attention layer's qkv and o products and its MLP's two; its norms and q
/// and k norms.
pub(crate) fn per_layer(k: Kind) -> (usize, usize) {
    match k {
        Kind::Linear => (5, 6),
        Kind::Full => (4, 4),
    }
}

/// Qwen3.5's executor description for stage `st` from the config.json in `dir` (R3a): its layer kinds lowered (`lower::qwen35`); `c` the same config as the
/// family's reader takes it, for its end tokens and its context. What the loader runs (the tests hold it to `describe` below).
pub fn lowered(dir: &Path, c: &crate::qwen35::Config35, st: &Stage) -> Result<ModelDesc> {
    let (kinds, label) = crate::runtime::describe::describe_dir(dir)?;
    super::lower::qwen35(&kinds, c.eos.clone(), c.max_pos, st).map_err(|e| e.context(&label))
}

/// The Qwen3.5 layout's gated attention (Qwen3.5, Qwen3.8 dense) as the tests' description and kcheck35 build it: heads of `d`, each query head with its gate in
/// Mix's rows; the (1 + w) q and k norms (small tensors `qk`) and the rotary on the first `rotary` dims in fp32 with one rounding; scale d^-0.5, the scores fp32;
/// the output times sigmoid(gate). Ring `ring`.
#[cfg(any(test, feature = "dev"))]
pub(crate) fn attention(d: usize, nq: usize, nk: usize, rotary: usize, qk: (usize, usize), ring: usize) -> AttnOp {
    AttnOp {
        src: Slot::Mix,
        stride: 2 * nq * d + 2 * nk * d,
        q_step: 2 * d,
        k_off: nq * 2 * d,
        v_off: Some(nq * 2 * d + nk * d),
        d,
        nq,
        nk,
        writes: true,
        ring,
        window: 0,
        norm: QkNorm::Plus1 { q: qk.0, k: qk.1 },
        v_norm: false,
        rope: Some(RopeAt { table: 0, span: rotary, pairs: rotary / 2 }),
        q_scale: 1.0,
        scale: (d as f32).powf(-0.5),
        softcap: 0.0,
        gate: Some(GateAt { off: d, step: 2 * d, norm: false }),
        round: false,
        sbf16: false,
        picked: false,
    }
}

/// The executor's description of Qwen3.5's layout for stage `st`, written by hand: what the tests hold the lowering of its layer kinds to (`lowered`).
#[cfg(test)]
pub fn describe(c: &crate::qwen35::Config35, st: &Stage) -> Result<ModelDesc> {
    if c.head_dim != 256 || c.lin_k_dim != 128 || c.lin_v_dim != 128 || c.conv_kernel != 4 {
        return err(format!(
            "head {}, linear heads {}/{}, convolution {}: this engine runs head 256, linear heads of 128 and a convolution of 4",
            c.head_dim, c.lin_k_dim, c.lin_v_dim, c.conv_kernel
        ));
    }
    let (nq, nk, nv) = (c.heads, c.kv_heads, c.lin_v_heads);
    let (cd, vd) = (c.conv_dim(), c.value_dim());
    let full_w = 2 * nq * 256 + 2 * nk * 256;
    let row = cd + vd;
    let (mut n_mats, mut n_small, mut rings, mut states, mut layers) = (0, 0, Vec::new(), Vec::new(), Vec::new());
    let mut first = None;
    for (l, &kind) in c.kinds.iter().enumerate() {
        if !(st.layers.0..st.layers.1).contains(&l) {
            // (another stage's layer: nothing here, never run)
            layers.push(Vec::new());
            continue;
        }
        let (b, s0) = (n_mats, n_small);
        let (m, n) = per_layer(kind);
        (n_mats, n_small) = (n_mats + m, n_small + n);
        let mut ops = Vec::new();
        if l == 0 {
            first = Some(Op::RmsNorm { x: Slot::X, w: s0, y: Slot::H, weight: NormW::PlusOne, eps: c.eps });
        } else {
            ops.push(Op::NormAdd { y: Slot::D, w: s0, plus1: true });
        }
        let mlp = match kind {
            Kind::Linear => {
                let (conv, ssm) = (states.len(), states.len() + 1);
                states.push(3 * cd * 2);
                states.push(nv * 128 * 128 * 4);
                ops.extend([
                    Op::Linear { w: b, x: Slot::H, y: Slot::Mix, name: "linear attention: qkvz product" },
                    Op::Linear { w: b + 1, x: Slot::H, y: Slot::Ba, name: "linear attention: ba product" },
                    Op::GdnConv { w: s0 + 2, state: conv, cd, row },
                    Op::GdnDelta { a_log: s0 + 3, dt_bias: s0 + 4, state: ssm, cd, nk: c.lin_k_heads, nv },
                    Op::GatedNorm { w: s0 + 5, row, nv, z_off: cd, sigmoid: false },
                    Op::Linear { w: b + 2, x: Slot::Core, y: Slot::O, name: "linear attention: out product" },
                ]);
                b + 3
            }
            Kind::Full => {
                let ring = rings.len();
                rings.push(Ring::kv(256, nk, 0));
                ops.extend([
                    Op::Linear { w: b, x: Slot::H, y: Slot::Mix, name: "attention: qkv product" },
                    Op::Attn(attention(256, nq, nk, c.rotary_dim, (s0 + 2, s0 + 3), ring)),
                    Op::Linear { w: b + 1, x: Slot::Att, y: Slot::O, name: "attention: o product" },
                ]);
                b + 2
            }
        };
        ops.extend([
            Op::NormAdd { y: Slot::O, w: s0 + 1, plus1: true },
            Op::GateUpSilu { w: mlp, inter: c.inter },
            Op::Linear { w: mlp + 1, x: Slot::M, y: Slot::D, name: "MLP: down product" },
        ]);
        layers.push(ops);
    }
    let final_norm = n_small;
    let e = n_mats;
    let lm = if c.tie { e } else { e + 1 };
    // (the tensors, named in the order the operations index them: a stage's layers' matrices and small tensors, then the final norm, the embedding and the head)
    let all = c.matrices();
    let item = |m: &crate::qwen35::Pack35| crate::weights::Item { members: m.parts.iter().map(|x| x.0.clone()).collect(), rows: m.rows(), cols: m.cols };
    let (mut mats, mut small) = (Vec::new(), Vec::new());
    for l in st.layers.0..st.layers.1.min(c.layers) {
        let p = format!("{}layers.{l}", c.prefix);
        mats.extend(all.iter().filter(|m| m.key.starts_with(&format!("{p}."))).map(item));
        small.extend([format!("{p}.input_layernorm.weight"), format!("{p}.post_attention_layernorm.weight")]);
        small.extend(match c.kinds[l] {
            Kind::Full => ["q_norm", "k_norm"].map(|x| format!("{p}.self_attn.{x}.weight")).to_vec(),
            Kind::Linear => ["conv1d.weight", "A_log", "dt_bias", "norm.weight"].map(|x| format!("{p}.linear_attn.{x}")).to_vec(),
        });
    }
    small.push(format!("{}norm.weight", c.prefix));
    mats.extend(all.iter().filter(|m| m.key == "embed" || m.key == "lm_head").map(item));
    let mut widths = [0usize; super::SLOTS];
    let q_w = nq * 256;
    for (s, w) in [
        (Slot::X, c.hidden),
        (Slot::H, c.hidden),
        (Slot::Mix, row.max(full_w)),
        (Slot::Ba, 2 * nv),
        (Slot::Xc, cd),
        (Slot::Core, vd),
        (Slot::Qo, q_w),
        (Slot::Att, q_w),
        (Slot::O, c.hidden),
        (Slot::Gu, 2 * c.inter),
        (Slot::M, c.inter),
        (Slot::D, c.hidden),
        (Slot::Hlast, c.hidden),
    ] {
        widths[s as usize] = w;
    }
    Ok(ModelDesc {
        family: "Qwen3.5",
        hidden: c.hidden,
        vocab: c.vocab,
        eps: c.eps,
        max_pos: c.max_pos,
        widths,
        mats,
        small,
        forms: Vec::new(),
        ropes: vec![Rope { theta: c.rope_theta, dim: c.rotary_dim, pairs: c.rotary_dim / 2, trig64: false }],
        rings,
        // (a later stage never runs the first operation)
        first: first.unwrap_or(Op::RmsNorm { x: Slot::X, w: 0, y: Slot::H, weight: NormW::PlusOne, eps: c.eps }),
        layers,
        states,
        final_norm: Some((final_norm, Slot::D, true)),
        decoded: true,
        // (the skinny kernels: 1.5% faster on the 4080 SUPER, their sums in another order; off until a quality check against the reference)
        skinny: false,
        embed: e,
        lm,
        softcap: 0.0,
        eos: c.eos.clone(),
        post: None,
        v4: None,
        hc: (0.0, 0),
        logit_mult: 0.0,
    })
}

#[cfg(test)]
mod tests {
    use super::super::{MoeOp, Op, Slot, Stage};
    use crate::runtime::describe::describe;

    fn tiny() -> serde_json::Value {
        serde_json::from_str(include_str!("../../tests/fixtures/qwen35moe_tiny_config.json")).unwrap()
    }

    /// The dynamic route's ends of the layout with experts (`split::Loader::ends`) are its description's layers and its config's ids: the dense layout's config
    /// reader refuses it (`Config35`), so `run` at penguin failed there before its first stage.
    #[test]
    fn the_layout_with_experts_ends_by_its_description() {
        let dir = std::env::temp_dir().join(format!("glyd-q35moe-ends-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), include_str!("../../tests/fixtures/qwen35moe_tiny_config.json")).unwrap();
        let loader = crate::dynamic::split::loader(&describe(&tiny()).unwrap()).unwrap();
        let (got, routed) = (super::ends(&dir), (loader.ends)(&dir));
        let dense = crate::qwen35::Config35::load(&dir).map(|c| c.layers);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!((got.unwrap(), routed.unwrap()), ((4, vec![2]), (4, vec![2])));
        assert!(dense.unwrap_err().msg.contains("mixture of experts"));
    }

    /// The layout with experts, lowered from its kinds: each layer's mixer as the dense layout's, then the post-attention norm and `Op::Moe` in the MLP's place
    /// (block k the k-th layer's, the experts' sum the layer's update); no MLP matrix is named (the experts' tensors are their block's), every matrix and
    /// small tensor named is the description's, and no slot of the MLP's is sized.
    #[test]
    fn a_layer_of_experts_runs_the_experts_operation_in_the_mlps_place() {
        let d = describe(&tiny()).unwrap();
        let m = super::super::lower::qwen35(&d, vec![2], 1100, &Stage::whole(4)).unwrap();
        assert_eq!((m.layers.len(), m.hidden, m.rings.len(), m.states.len()), (4, 1024, 1, 6));
        for (l, ops) in m.layers.iter().enumerate() {
            let n = ops.len();
            assert!(matches!(ops[n - 2], Op::NormAdd { y: Slot::O, plus1: true, .. }), "{:?}", ops[n - 2]);
            assert!(matches!(ops[n - 1], Op::Moe(x) if x == MoeOp { y: Slot::D, ..MoeOp::of(l) }), "{:?}", ops[n - 1]);
            assert!(!ops.iter().any(|op| matches!(op, Op::GateUpSilu { .. })));
        }
        // (three Gated DeltaNet layers' qkvz, ba and out products, the attention layer's qkv and o, the embedding and the head)
        assert_eq!(m.mats.len(), 3 * 3 + 2 + 2);
        assert_eq!((m.width(Slot::Gu), m.width(Slot::M), m.width(Slot::D)), (0, 0, 1024));
        assert!(m.wanted(&Stage::whole(4)).iter().all(|&w| w), "a matrix no operation multiplies by");
        let names = d.names();
        assert!(m.mats.iter().all(|it| it.members.iter().all(|n| names.contains(n))) && m.small.iter().all(|n| names.contains(n)));
        assert!(m.mats.iter().all(|it| it.members.iter().all(|n| !n.contains(".mlp."))));
    }

    /// A stage of a model split across GPUs counts its blocks of experts from its own first layer, as its loader reads them.
    #[test]
    fn a_stages_blocks_of_experts_count_from_its_first_layer() {
        let d = describe(&tiny()).unwrap();
        let m = super::super::lower::qwen35(&d, vec![2], 1100, &Stage { layers: (2, 4), first: false, last: true }).unwrap();
        let blocks: Vec<Vec<usize>> =
            m.layers.iter().map(|ops| ops.iter().filter_map(|op| if let Op::Moe(x) = op { Some(x.block) } else { None }).collect()).collect();
        assert_eq!(blocks, [vec![], vec![], vec![0], vec![1]]);
    }

    /// The dense layout's save by its description (`saved`) is what the dense layout's own lists name, in their order (`Config35::packs` and `kept`), with the
    /// same states a sequence, keys and values a token and longest context: the lossy route's saves and plans of a dense model are what they were. Every dense
    /// config of the fixtures and the family's two random cases.
    #[test]
    fn the_dense_layouts_save_by_its_description_is_its_own_lists() {
        let all: serde_json::Value = serde_json::from_str(include_str!("../../tests/fixtures/qwen35_configs.json")).unwrap();
        let mut texts: Vec<(String, String)> = all.as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.to_string())).collect();
        for m in [include_str!("../../tests/fixtures/qwen35_tiny_manifest.json"), include_str!("../../tests/fixtures/qwen35_wide_tiny_manifest.json")] {
            let man: serde_json::Value = serde_json::from_str(m).unwrap();
            texts.push(("a random case".into(), man["config"].as_str().unwrap().to_string()));
        }
        let mut dense = 0;
        for (name, text) in &texts {
            // (the layout with experts: the dense layout's reader refuses it)
            let Ok(c) = crate::qwen35::Config35::parse(text, name, None) else { continue };
            let (low, packs, kept) = super::saved(&crate::runtime::config::ModelConfig::parse(text).unwrap()).unwrap();
            assert_eq!((packs, kept), (c.packs(), c.kept()), "{name}");
            let (state, per) = (low.states.iter().sum::<usize>() as u64, low.rings.iter().map(|r| r.token_bytes()).sum::<usize>() as u64);
            assert_eq!((state, per, low.max_pos), (c.state_bytes(), c.kv_bytes_per_token(), c.max_pos), "{name}");
            dense += 1;
        }
        assert_eq!(dense, 7 + 2);
    }

    /// The dense layout lowers as it did: its MLP's two products, no block of experts.
    #[test]
    fn the_dense_layout_keeps_its_mlp() {
        let man: serde_json::Value = serde_json::from_str(include_str!("../../tests/fixtures/qwen35_tiny_manifest.json")).unwrap();
        let d = describe(&serde_json::from_str(man["config"].as_str().unwrap()).unwrap()).unwrap();
        let m = super::super::lower::qwen35(&d, vec![2], 1100, &Stage::whole(4)).unwrap();
        assert!(m.layers.iter().all(|ops| ops.iter().any(|op| matches!(op, Op::GateUpSilu { .. })) && !ops.iter().any(|op| matches!(op, Op::Moe(_)))));
        assert_eq!((m.mats.len(), m.width(Slot::Gu), m.width(Slot::M)), (3 * 5 + 4 + 1, 2 * 3584, 3584));
    }
}
