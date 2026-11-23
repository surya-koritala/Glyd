//! Nemotron-H's kernel table and weights, what the executor's description (src/dynamic/nemotron.rs) runs the model on: the Mamba-2 kernels (kernels/nemotron.cu; its attention is the attention kind's, src/dynamic/attn.rs) loaded once (`KernNh`), the weights of a glyd-v7 or
//! a Hub-format directory (`WeightsNh`: a layer at a time, the embedding, the head; the layers' matrices packed as they load), the interface a block of mixture-of-experts layers plugs in by (`MoeBlock`, `MoeFactory`, with a
//! stand-in that says so for a model loaded without one; src/moe_nh.rs is the block) and the host hooks the other families' blocks share (`NhHost`). The hand-written engine that used to be here is retired:
//! its parity record is R7's fixtures and `paritycheck`'s (bits-identical at every level), and its layer-by-layer tools (`nhlayers`, `nhtoy`, `nhoffload`) run on the executor. The layers' math: tools/nemotron_ref.py;
//! the config and tensor names: src/nemotron.rs.

use crate::dev::{Buf, Dev};
use crate::ffi::{err, Error, Lib, Result, Stream};
use crate::kern35::Mat;
use crate::launch;
use crate::moe::MoeScratch;
use crate::nemotron::{pack_layout, ConfigNemotronH, NhKind};
use crate::qwen35::Pack35;
use crate::weights::{choose_layout, ordered, read_pack, upload_bytes, upload_packed, v7_manifest, Layout, PackBuf};
use glyd_gpu::safetensors::Checkpoint;
use std::collections::HashMap;
use std::path::Path;

// The kernels' names: those the .cu sources give them, in every build (kname!).
/// A Mamba-2 head's channels, the state a channel has, the tokens of a chunk of the chunked scan (kernels/nemotron.cu).
pub const NH_P: usize = 64;
pub const NH_N: usize = 128;
pub const NH_L: usize = 128;

/// The Mamba-2 chunked scan's buffers for up to `rows` tokens a call: dt and its running sum, C B^T of every chunk and group, every chunk's own state.
pub struct SsdWs {
    pub rows: usize,
    dtv: Buf,
    acum: Buf,
    g: Buf,
    st: Buf,
}

impl SsdWs {
    pub fn new(c: &ConfigNemotronH, rows: usize) -> Result<SsdWs> {
        SsdWs::sized(c.m_heads, c.m_groups, rows)
    }

    /// For `heads` heads in `groups` groups.
    pub fn sized(heads: usize, groups: usize, rows: usize) -> Result<SsdWs> {
        let nch = rows.div_ceil(NH_L).max(1);
        Ok(SsdWs {
            rows,
            dtv: Buf::new(nch * NH_L * heads * 4)?,
            acum: Buf::new(nch * NH_L * heads * 4)?,
            g: Buf::new(nch * groups * NH_L * NH_L * 4)?,
            st: Buf::new(nch * heads * NH_P * NH_N * 4)?,
        })
    }

    pub fn bytes(&self) -> usize {
        self.dtv.bytes + self.acum.bytes + self.g.bytes + self.st.bytes
    }
}

/// Nemotron's kernels with typed launches.
pub struct KernNh {
    conv: crate::kt::nh_conv,
    ssm: crate::kt::nh_ssm,
    cum: crate::kt::nh_ssd_cum,
    gmat: crate::kt::nh_ssd_g,
    states: crate::kt::nh_ssd_states,
    pass: crate::kt::nh_ssd_pass,
    scan_y: crate::kt::nh_ssd_y,
    gnorm: crate::kt::nh_gated_norm,
    small: crate::kt::small_linear,
}

impl KernNh {
    pub fn load(dev: &Dev) -> Result<KernNh> {
        Ok(KernNh {
            conv: dev.typed()?,
            ssm: dev.typed()?,
            cum: dev.typed()?,
            gmat: dev.typed()?,
            states: dev.typed()?,
            pass: dev.typed()?,
            scan_y: dev.typed()?,
            gnorm: dev.typed()?,
            small: dev.typed()?,
        })
    }

    /// The causal convolution (kernel 4) with its bias and SiLU over the first `c` channels of `n` rows `xs` apart; the bf16 state [3, c] read and left holding the last three inputs; y [n, c].
    ///
    /// # Safety
    /// Live device buffers of the sizes the kernels read and write (their comments in kernels/nemotron.cu).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn conv(&self, s: Stream, x: u64, xs: i32, w: u64, bias: u64, st: u64, y: u64, n: i32, c: i32) -> Result<()> {
        launch!(self.conv, [(c as u32).div_ceil(256), 1, 1], [256, 1, 1], s; x, xs, w, bias, st, y, n, c)
    }

    /// The selective state update of n rows, one token after another (a decode step: n = 1): xc [n, cs] the convolution's output, dtr the dt columns of the in_proj's output (rows `dts` apart),
    /// state fp32 [nh, 64, 128] read and written back, y [n, nh 64].
    ///
    /// # Safety
    /// As `conv`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ssm(
        &self,
        s: Stream,
        xc: u64,
        cs: i32,
        dtr: u64,
        dts: i32,
        dt_bias: u64,
        a_log: u64,
        d: u64,
        state: u64,
        y: u64,
        n: i32,
        nh: i32,
        ng: i32,
        dt_min: f32,
    ) -> Result<()> {
        launch!(self.ssm, [nh as u32, (NH_P / 8) as u32, 1], [256, 1, 1], s; xc, cs, dtr, dts, dt_bias, a_log, d, state, y, n, nh, ng, dt_min)
    }

    /// The same for a prompt, in chunks of NH_L tokens (state-space duality); `ws` holds at least n rows.
    ///
    /// # Safety
    /// As `conv`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ssd(
        &self,
        s: Stream,
        ws: &SsdWs,
        xc: u64,
        cs: i32,
        dtr: u64,
        dts: i32,
        dt_bias: u64,
        a_log: u64,
        d: u64,
        state: u64,
        y: u64,
        n: i32,
        nh: i32,
        ng: i32,
        dt_min: f32,
    ) -> Result<()> {
        if n as usize > ws.rows {
            return err(format!("{n} rows in a scan that has buffers for {}", ws.rows));
        }
        let nch = (n as u32).div_ceil(NH_L as u32);
        launch!(self.cum, [nch, (nh as u32).div_ceil(128), 1], [128, 1, 1], s; dtr, dts, dt_bias, a_log, ws.dtv.ptr, ws.acum.ptr, n, nh, dt_min)?;
        launch!(self.gmat, [nch, ng as u32, 1], [256, 1, 1], s; xc, cs, ws.g.ptr, n, nh, ng)?;
        launch!(self.states, [nch, nh as u32, 1], [256, 1, 1], s; xc, cs, ws.dtv.ptr, ws.acum.ptr, ws.st.ptr, n, nh, ng)?;
        launch!(self.pass, [((nh as usize * NH_P * NH_N) as u32).div_ceil(256), 1, 1], [256, 1, 1], s; ws.acum.ptr, ws.st.ptr, state, n, nh, nch as i32)?;
        launch!(self.scan_y, [nch, nh as u32, 2], [256, 1, 1], s; xc, cs, ws.dtv.ptr, ws.acum.ptr, ws.g.ptr, ws.st.ptr, d, y, n, nh, ng)
    }

    /// The gated norm of n rows, in place on y [n, di]: z at z[t * zs ..], the weight bf16 [di], ng groups of di / ng channels (at most 1024).
    ///
    /// # Safety
    /// As `conv`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gated_norm(&self, s: Stream, y: u64, z: u64, zs: i32, w: u64, n: u32, di: i32, ng: i32, eps: f32) -> Result<()> {
        if ng == 0 || di % ng != 0 || di / ng > 1024 {
            return err(format!("a gated norm of {di} channels in {ng} groups: not supported"));
        }
        launch!(self.gnorm, [n, ng as u32, 1], [256, 1, 1], s; y, z, zs, w, di, ng, eps)
    }

    /// y [m, rows] = x [m, cols] W^T for a bf16 W the library does not pack.
    ///
    /// # Safety
    /// As `conv`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn small_linear(&self, s: Stream, x: u64, xs: i32, w: u64, y: u64, ys: i32, m: u32, rows: i32, cols: i32) -> Result<()> {
        launch!(self.small, [m, (rows.max(1) as u32).div_ceil(8), 1], [256, 1, 1], s; x, xs, w, y, ys, rows, cols)
    }
}

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
    /// A development build's stage marker (a profile's).
    fn mark(&self, _what: &'static str) {}
}

/// What a mixture-of-experts layer is where no block was given to the loader (src/moe_nh.rs makes one): it says so where it is run.
pub struct NoMoe;

impl MoeBlock for NoMoe {
    unsafe fn forward(&self, _: &dyn NhHost, _: u64, _: usize, _: u64) -> Result<()> {
        err("a mixture-of-experts layer with no block of experts to run it: the engine was loaded without one")
    }
}

/// Makes layer `l`'s block of experts: the layer's number, the config, the checkpoint, the layout of the packs.
pub type MoeFactory<'a> = &'a dyn Fn(usize, &ConfigNemotronH, &Checkpoint, Layout) -> Result<Box<dyn MoeBlock>>;

/// A layer's buffers, by kind: its matrices (Mamba-2: in_proj, out_proj; attention: qkv, o; a mixture of experts: none, its block holds them) and small tensors.
pub struct LayerNh {
    pub kind: NhKind,
    /// Its index among the layers of its kind (a Mamba-2 layer's state, an attention layer's cache).
    pub idx: usize,
    pub norm: Buf,
    pub mats: Vec<Mat>,
    /// Mamba-2: conv1d weight and bias, A_log, D, dt_bias, the gated norm's weight (bf16 as the checkpoint has them).
    pub small: Vec<Buf>,
    pub moe: Option<Box<dyn MoeBlock>>,
}

/// The model's weights in GPU memory. A layer not asked for (`only`) is None; so are the embedding, the LM head and the final norm then.
pub struct WeightsNh {
    pub cfg: ConfigNemotronH,
    pub layout: Layout,
    pub layers: Vec<Option<LayerNh>>,
    pub embed: Option<PackBuf>,
    pub head: Option<PackBuf>,
    pub norm_f: Option<Buf>,
    /// Device bytes of the weights.
    pub bytes: usize,
}

impl WeightsNh {
    /// A Hub-format directory (config.json, the safetensors) or a glyd-v7 directory (`glyd-engine pack MODEL OUT --layout i6|i5|i4`): the layers of `only` (all, with the embedding, the head and the final norm, where
    /// None) checked against the weight map, their matrices packed here by `threads` workers in `layout` (None: the GPU's; each matrix in `pack_layout` of it: at wren the experts in four-bit codes and the rest in
    /// six-bit ones) and copied to the GPU as they are done. A glyd-v7 directory is at one level (`layout` None, or the same): the packs it holds saved are read as they are, each in the layout its entry names, the rest
    /// (a matrix the packer does not take, kept bf16) as the Hub's. A mixture-of-experts layer holds its norm and
    /// its block, which is `moe`'s (else `NoMoe`): the block gets the checkpoint and the layout and loads what it needs of the layer's tensors.
    pub fn load(
        dir: &Path,
        threads: usize,
        layout: Option<Layout>,
        cc: (i32, i32),
        total_mem: u64,
        only: Option<&[usize]>,
        moe: Option<MoeFactory<'_>>,
    ) -> Result<WeightsNh> {
        let whole = only.is_none();
        WeightsNh::load_ends(dir, threads, layout, cc, total_mem, only, (whole, whole), moe)
    }

    /// As `load`, the embedding loaded where `ends.0` and the final norm and LM head where `ends.1` (a stage of a model split across GPUs: the
    /// first stage takes the embedding, the last the head).
    #[allow(clippy::too_many_arguments)]
    pub fn load_ends(
        dir: &Path,
        threads: usize,
        layout: Option<Layout>,
        cc: (i32, i32),
        total_mem: u64,
        only: Option<&[usize]>,
        ends: (bool, bool),
        moe: Option<MoeFactory<'_>>,
    ) -> Result<WeightsNh> {
        let cfg = ConfigNemotronH::load(dir)?;
        let ck = Checkpoint::open(dir).map_err(|e| Error(e.to_string()))?;
        let have: Vec<(String, String, Vec<u64>)> = ck
            .names()
            .map(|n| {
                let (_, t) = ck.get(n).expect("a tensor the checkpoint names");
                (n.to_string(), t.dtype.clone(), t.shape.clone())
            })
            .collect();
        let v7 = v7_manifest(dir)?;
        if let (Some((got, _)), Some(asked)) = (&v7, layout) {
            if *got != asked {
                return err(format!("{}: packed in the {} layout, where {} is asked", dir.display(), got.name(), asked.name()));
            }
        }
        let wanted: Vec<usize> = only.map_or_else(|| (0..cfg.layers).collect(), |l| l.to_vec());
        let held = if v7.is_some() { cfg.saved(&ck) } else { Default::default() };
        match only {
            None => {
                cfg.check(&have, &held)?;
            }
            Some(l) => cfg.check_layers(&have, l, &held)?,
        }
        // the matrices: packed where the packer takes them, else as the checkpoint has them
        let mut items: Vec<Pack35> = Vec::new();
        let mut dense: Vec<(String, Vec<String>, usize, usize)> = Vec::new();
        for &l in wanted.iter().filter(|&&l| cfg.kinds[l] != NhKind::Moe) {
            for p in cfg.layer_packs(l) {
                if p.packable() {
                    items.push(p);
                } else {
                    dense.push((p.key.clone(), p.parts.iter().map(|x| x.0.clone()).collect(), p.rows(), p.cols));
                }
            }
        }
        let (first, last) = ends;
        if first || (last && cfg.tie) {
            items.push(Pack35 { key: "embed".into(), parts: vec![(format!("{}embeddings.weight", cfg.prefix), cfg.vocab)], cols: cfg.hidden });
        }
        if last && !cfg.tie {
            items.push(Pack35 { key: "lm_head".into(), parts: vec![("lm_head.weight".into(), cfg.vocab)], cols: cfg.hidden });
        }
        let matrix_bytes: u64 = items.iter().map(|p| (p.rows() * p.cols * 2) as u64).sum();
        let layout = match &v7 {
            Some((l, _)) => *l,
            None => choose_layout(layout, cc, total_mem, matrix_bytes, 0).0,
        };
        let mut packed: HashMap<String, PackBuf> = HashMap::new();
        ordered(items.len(), threads, 2 * threads.max(1), &|k| read_pack(&ck, v7.as_ref(), &items[k], pack_layout(layout, &items[k].key)), |k, p| {
            crate::status::STATUS.beat(); // (a sign of life for the program that started the engine)
            packed.insert(items[k].key.clone(), upload_packed(p)?);
            Ok(())
        })?;
        let read = |name: &str| -> Result<Buf> { upload_bytes(&ck.read(name).map_err(|e| Error(e.to_string()))?) };
        let schedule = cfg.schedule();
        let mut layers: Vec<Option<LayerNh>> = (0..cfg.layers).map(|_| None).collect();
        let mut bytes = 0usize;
        for &l in &wanted {
            crate::status::STATUS.beat(); // (a mixture-of-experts layer's block takes seconds to read)
            let (kind, idx) = schedule[l];
            let mut mats = Vec::new();
            // (a mixture-of-experts layer's block loads its own tensors: router, latent projections, shared expert, experts)
            for p in cfg.layer_packs(l).into_iter().filter(|_| kind != NhKind::Moe) {
                mats.push(match packed.remove(&p.key) {
                    Some(b) => Mat::Packed(b),
                    None => {
                        let (_, members, rows, cols) = dense.iter().find(|d| d.0 == p.key).expect("a matrix kept as it is");
                        let mut b = Vec::with_capacity(rows * cols * 2);
                        for m in members {
                            b.extend(ck.read(m).map_err(|e| Error(e.to_string()))?);
                        }
                        Mat::Dense(upload_bytes(&b)?, *rows, *cols)
                    }
                });
            }
            let kept = cfg.layer_kept(l);
            let norm = read(&kept[0].name)?;
            let small = if kind == NhKind::Mamba { kept[1..].iter().map(|t| read(&t.name)).collect::<Result<Vec<_>>>()? } else { Vec::new() };
            let block: Option<Box<dyn MoeBlock>> = if kind == NhKind::Moe {
                Some(match &moe {
                    Some(make) => make(l, &cfg, &ck, layout)?,
                    None => Box::new(NoMoe),
                })
            } else {
                None
            };
            bytes += norm.bytes
                + mats.iter().map(Mat::bytes).sum::<usize>()
                + small.iter().map(|b| b.bytes).sum::<usize>()
                + block.as_ref().map_or(0, |b| b.bytes());
            layers[l] = Some(LayerNh { kind, idx, norm, mats, small, moe: block });
        }
        let embed = packed.remove("embed");
        let head = if last && !cfg.tie { Some(packed.remove("lm_head").ok_or_else(|| Error("the LM head is not packed".into()))?) } else { None };
        let norm_f = if last { Some(read(&format!("{}norm_f.weight", cfg.prefix))?) } else { None };
        bytes += embed.as_ref().map_or(0, |e| e.bytes) + head.as_ref().map_or(0, |h| h.bytes) + norm_f.as_ref().map_or(0, |b| b.bytes);
        if let Some(k) = packed.keys().next() {
            return err(format!("a pack the forward pass does not take: {k}"));
        }
        Ok(WeightsNh { cfg, layout, layers, embed, head, norm_f, bytes })
    }

    pub fn lm_head(&self) -> Option<&PackBuf> {
        self.head.as_ref().or(self.embed.as_ref())
    }

    /// Whether every layer, the embedding, the head and the final norm are here.
    pub fn whole(&self) -> bool {
        self.layers.iter().all(Option::is_some) && self.embed.is_some() && self.norm_f.is_some()
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    /// The stand-in for the mixture-of-experts layers says what it is; the hooks around it have the shapes a plug-in needs.
    #[test]
    fn the_stand_in_says_what_it_is() {
        let b: Box<dyn MoeBlock> = Box::new(NoMoe);
        assert!(!b.graphable() && b.bytes() == 0);
        let factory: MoeFactory = &|_, _, _, _| Ok(Box::new(NoMoe));
        let _ = factory;
    }

    #[test]
    fn the_kernels_constants_are_the_configs() {
        let c = crate::nemotron::tests::parse();
        assert_eq!((c.m_head_dim, c.m_state, c.chunk), (NH_P, NH_N, NH_L));
        assert!(c.inner() / c.m_groups <= 1024 && c.heads / c.kv_heads <= 16);
        let _ = crate::kern::norm_threads(c.hidden as i32).unwrap();
    }
}
