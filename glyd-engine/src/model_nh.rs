//! Nemotron 3 Super on the GPU, one sequence: the layers in the order of the config's pattern (Mamba-2, attention without a positional encoding, latent mixture of experts), each
//! `x += mixer(norm(x))`, the embedding, the final norm and the LM head packed as the other models' matrices are. The Mamba-2 and attention layers are this file and kernels/nemotron.cu; the
//! mixture-of-experts layers come from outside through `MoeBlock` (src/moe_nh.rs is the block, on the mixture-of-experts core; this file holds a stand-in that says so, for a model loaded without one). A prompt in chunks and a decode step as one CUDA graph, greedy tokens.
//! The weights are a glyd-v7 directory's (`glyd-engine pack MODEL OUT --layout i6|i5|i4`: one level, each pack in the layout it names: at wren the experts' four-bit codes and every other matrix's six-bit ones) or a Hub-format
//! directory's, packed as they load. `glyd-engine run MODEL:kestrel|swift|wren` serves the model, one user at a time
//! (src/cli.rs, `Gen`). The layers' math: tools/nemotron_ref.py; the config and tensor names: src/nemotron.rs.

use crate::dev::{copy_d2h_async, copy_h2d_async, memset_async, Buf, Dev, Graph, Kernel, Pinned};
use crate::ffi::{err, Error, Lib, Result, Stream};
use crate::kern::Kern;
use crate::kern35::Mat;
use crate::launch;
use crate::model::{embed_plan, Sample, TopGpu};
use crate::moe::MoeScratch;
use crate::nemotron::{pack_layout, ConfigNemotronH, NhKind};
use crate::qwen35::Pack35;
use crate::weights::{choose_layout, ordered, read_pack, upload_bytes, upload_packed, v7_manifest, Layout, PackBuf};
use glyd_gpu::safetensors::Checkpoint;
use std::collections::HashMap;
use std::path::Path;

// The kernels' names: those the .cu sources give them, in every build (kname!).
macro_rules! kname {
    ($n:literal) => {
        $n
    };
}

/// A Mamba-2 head's channels, the state a channel has, the tokens of a chunk of the chunked scan (kernels/nemotron.cu).
pub const NH_P: usize = 64;
pub const NH_N: usize = 128;
pub const NH_L: usize = 128;
/// The parts a decode step's attention splits each row's keys into, and the floats a part's result takes.
pub const NH_SPLITS: usize = 16;
pub(crate) const NH_PART: usize = 130;

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
    conv: Kernel,
    ssm: Kernel,
    cum: Kernel,
    gmat: Kernel,
    states: Kernel,
    pass: Kernel,
    scan_y: Kernel,
    gnorm: Kernel,
    kv_write: Kernel,
    attn: Kernel,
    attn_merge: Kernel,
    small: Kernel,
}

impl KernNh {
    pub fn load(dev: &Dev) -> Result<KernNh> {
        Ok(KernNh {
            conv: dev.kernel(kname!("nh_conv"))?,
            ssm: dev.kernel(kname!("nh_ssm"))?,
            cum: dev.kernel(kname!("nh_ssd_cum"))?,
            gmat: dev.kernel(kname!("nh_ssd_g"))?,
            states: dev.kernel(kname!("nh_ssd_states"))?,
            pass: dev.kernel(kname!("nh_ssd_pass"))?,
            scan_y: dev.kernel(kname!("nh_ssd_y"))?,
            gnorm: dev.kernel(kname!("nh_gated_norm"))?,
            kv_write: dev.kernel(kname!("nh_kv_write"))?,
            attn: dev.kernel(kname!("nh_attn_parts"))?,
            attn_merge: dev.kernel(kname!("nh_attn_merge"))?,
            small: dev.kernel(kname!("small_linear"))?,
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

    /// The keys and values of n qkv rows into the cache kv [positions, nk, 256] at each row's position (`pos`, int32 on the device).
    ///
    /// # Safety
    /// As `conv`; the cache holds every position in `pos`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn kv_write(&self, s: Stream, qkv: u64, stride: i32, pos: u64, kv: u64, n: u32, nq: i32, nk: i32) -> Result<()> {
        launch!(self.kv_write, [n, nk as u32, 1], [128, 1, 1], s; qkv, stride, pos, kv, nq, nk)
    }

    /// A decode step's attention (n rows, each at its own position, from `pos` on the device): `part` holds n nq NH_SPLITS NH_PART floats.
    ///
    /// # Safety
    /// As `conv`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attn_decode(&self, s: Stream, q: u64, qs: i32, kv: u64, pos: u64, n: i32, nq: i32, nk: i32, part: u64, out: u64, os: i32) -> Result<()> {
        if nk == 0 || nq % nk != 0 || nq / nk > 16 {
            return err(format!("{nq} query heads over {nk} key-value heads: not supported"));
        }
        let sl2 = (1.0f64 / 128.0f64.sqrt() * std::f64::consts::LOG2_E) as f32;
        let splits = NH_SPLITS as i32;
        launch!(self.attn, [n as u32, nk as u32, NH_SPLITS as u32], [32 * (nq / nk) as u32, 1, 1], s; q, qs, kv, pos, n, nq, nk, sl2, part, splits)?;
        launch!(self.attn_merge, [n as u32, nq as u32, 1], [128, 1, 1], s; part, splits, out, os, nq)
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
    /// The (rows, columns, whether 12-bit) of every pack it multiplies through `EngineNh::lin`, so that the engine sizes the library's workspace and its decoded-matrix scratch for them.
    fn pack_shapes(&self) -> Vec<(i64, i64, bool)> {
        Vec::new()
    }
    /// Where all its experts are on the GPU: the core's layer and the latent projections (the dynamic engine runs the layer itself, src/dynamic).
    fn resident(&self) -> Option<(&crate::moe::MoeLayer, Option<(&Mat, &Mat)>)> {
        None
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

impl NhHost for EngineNh {
    fn dev(&self) -> &Dev {
        &self.dev
    }
    fn lib(&self) -> &Lib {
        &self.lib
    }
    fn mix(&self) -> &Buf {
        &self.mix
    }
    fn xc(&self) -> &Buf {
        &self.xc
    }
    fn chunk(&self) -> usize {
        self.chunk
    }
    fn moe_scratch(&self) -> &std::cell::OnceCell<MoeScratch> {
        &self.moe_scratch
    }
    unsafe fn lin(&self, w: &Mat, x: u64, m: usize, y: u64) -> Result<()> {
        EngineNh::lin(self, w, x, m, y)
    }
}

/// What a mixture-of-experts layer is where no block was given to `EngineNh::new` (src/moe_nh.rs makes one): it says so where it is run.
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

/// How a run is set up.
pub struct OptsNh {
    /// Positions the attention cache holds (prompt and reply).
    pub ctx: usize,
    /// Tokens a prompt chunk takes at most.
    pub chunk: usize,
    pub graph: bool,
    pub device: i32,
    pub layout: Option<Layout>,
    /// A prompt's Mamba-2 scan in chunks (else one token after another, as a decode step).
    pub chunked: bool,
    /// Load just these layers (a development check's); none: the model.
    pub only: Option<Vec<usize>>,
}

impl Default for OptsNh {
    fn default() -> OptsNh {
        OptsNh { ctx: 4096, chunk: 2048, graph: true, device: 0, layout: None, chunked: true, only: None }
    }
}

pub struct EngineNh {
    pub dev: Dev,
    pub lib: Lib,
    pub(crate) kern: Kern,
    pub(crate) k: KernNh,
    pub w: WeightsNh,
    chunk: usize,
    ctx: usize,
    pub(crate) chunked: bool,
    // activations, `chunk` rows each
    pub(crate) x: Buf,
    pub(crate) h: Buf,
    pub(crate) o: Buf,
    pub(crate) mix: Buf,
    pub(crate) xc: Buf,
    pub(crate) core: Buf,
    pub(crate) att: Buf,
    hlast: Buf,
    logits: Buf,
    scratch: Buf,
    embed_groups: Buf,
    embed_pairs: Buf,
    pub(crate) pos: Buf,
    pub(crate) st: Buf,
    tok: Buf,
    val: Buf,
    /// The sampler's candidates for the last step, where its request draws (`Gen::reply`).
    pub top: TopGpu,
    /// A decode step's attention parts.
    attn_part: Buf,
    ssd: SsdWs,
    // the products
    ws: Buf,
    done: Buf,
    blas: Option<crate::ffi::Blas>,
    own: bool,
    scratch_w: Buf,
    gpu_code: i64,
    // what a sequence holds
    pub kv: Buf,
    pub ssm: Buf,
    pub convst: Buf,
    graph: Option<Graph>,
    host: Pinned,
    /// Tokens in the sequence.
    pub n: usize,
    /// Device bytes of the activations and workspaces.
    pub work_bytes: usize,
    /// The mixture-of-experts layers' scratch, one for all of them: the first block that runs makes it (src/moe_nh.rs).
    pub(crate) moe_scratch: std::cell::OnceCell<MoeScratch>,
}

/// 64-row groups of the embedding decoded at once for a prompt chunk.
const EMBED_GROUPS: usize = 32;

impl EngineNh {
    pub fn new(dir: &Path, o: &OptsNh, moe: Option<MoeFactory<'_>>) -> Result<EngineNh> {
        let dev = Dev::new(o.device)?;
        let lib = Lib::find()?;
        let kern = Kern::load(&dev)?;
        let k = KernNh::load(&dev)?;
        let (free0, _) = dev.mem_info()?;
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get().min(16));
        let w = WeightsNh::load(dir, threads, o.layout, dev.cc_pair(), dev.total_mem as u64, o.only.as_deref(), moe)?;
        let c = &w.cfg;
        if c.m_head_dim != NH_P
            || c.m_state != NH_N
            || c.conv_kernel != 4
            || c.head_dim != 128
            || c.inner() / c.m_groups > 1024
            || c.heads / c.kv_heads > 16
            || c.hidden > 8192
        {
            return err(format!(
                "Mamba-2 heads of {} channels with a state of {}, a convolution of {}, attention heads of {}: this engine runs heads of {NH_P} channels, a state of {NH_N}, a convolution of 4 and attention heads of 128",
                c.m_head_dim, c.m_state, c.conv_kernel, c.head_dim
            ));
        }
        let (hid, di, cd) = (c.hidden, c.inner(), c.conv_dim());
        let ch = o.chunk.max(1);
        let qkv_w = (c.heads + 2 * c.kv_heads) * c.head_dim;
        let act = |cols: usize, rows: usize| Buf::new(rows * cols * 2);
        let (x, h, o_) = (act(hid, ch)?, act(hid, ch)?, act(hid, ch)?);
        let mix = act(c.in_rows().max(qkv_w), ch)?;
        let (xc, core, att) = (act(cd, ch)?, act(di, ch)?, act(c.heads * c.head_dim, ch)?);
        let (hlast, logits) = (act(hid, 1)?, act(c.vocab, 1)?);
        let scratch = Buf::new(64 * hid * 2)?;
        let embed_groups = Buf::new(EMBED_GROUPS * 64 * hid * 2)?;
        let embed_pairs = Buf::new(ch * 8)?;
        let (pos, st) = (Buf::zeroed(4 * ch)?, Buf::zeroed(16)?);
        let (tok, val) = (Buf::zeroed(4)?, Buf::zeroed(4)?);
        let top = TopGpu::new(c.vocab)?;
        let attn_part = Buf::zeroed(c.heads * NH_SPLITS * NH_PART * 4)?;
        let ssd = SsdWs::new(c, ch)?;
        let want = crate::tune!("GLYD_ENGINE_GEMM").unwrap_or_default();
        let blas = if ch > 64 && (want == "blas" || (want.is_empty() && dev.cc != (8, 9))) { crate::ffi::Blas::find() } else { None };
        let own = want == "own" || (want.is_empty() && blas.is_none() && matches!(dev.cc.0, 8 | 9));
        let gpu_code = lib.gpu()? as i64;
        let mut shapes: Vec<(i64, i64, bool)> = w
            .layers
            .iter()
            .flatten()
            .flat_map(|l| l.mats.iter())
            .filter_map(|m| if let Mat::Packed(p) = m { Some((p.pack.rows, p.pack.cols, p.pack.is_twelve())) } else { None })
            .collect();
        shapes.extend(w.layers.iter().flatten().filter_map(|l| l.moe.as_ref()).flat_map(|b| b.pack_shapes()));
        let scratch_w = if blas.is_some() || own { Buf::new(shapes.iter().map(|s| (s.0 * s.1) as usize * 2).max().unwrap_or(16))? } else { Buf::new(16)? };
        shapes.sort_unstable();
        shapes.dedup();
        let mut need = 0usize;
        for m_ in 1..=ch as i64 {
            for &(oo, kk, twelve) in &shapes {
                need = need.max(lib.linear_workspace(twelve, oo, kk, m_)?);
            }
        }
        if let Some(head) = w.lm_head() {
            need = need.max(lib.linear_workspace(head.pack.is_twelve(), head.pack.rows, head.pack.cols, 1)?);
        }
        let ws = Buf::zeroed(need)?;
        let done = Buf::zeroed(4 << 18)?;
        let ctx = o.ctx.div_ceil(16) * 16;
        let kv = Buf::zeroed(c.layers_of(NhKind::Attention) * ctx * c.kv_heads * 256 * 2)?;
        let ssm = Buf::zeroed(c.layers_of(NhKind::Mamba) * c.m_heads * NH_P * NH_N * 4)?;
        let convst = Buf::zeroed(c.layers_of(NhKind::Mamba) * 3 * cd * 2)?;
        let host = Pinned::new(4096)?;
        let used = free0.saturating_sub(dev.mem_info()?.0);
        let work_bytes = used.saturating_sub(w.bytes + kv.bytes + ssm.bytes + convst.bytes);
        let mut e = EngineNh {
            dev,
            lib,
            kern,
            k,
            w,
            chunk: ch,
            ctx,
            chunked: o.chunked,
            x,
            h,
            o: o_,
            mix,
            xc,
            core,
            att,
            hlast,
            logits,
            scratch,
            embed_groups,
            embed_pairs,
            pos,
            st,
            tok,
            val,
            top,
            attn_part,
            ssd,
            ws,
            done,
            blas,
            own,
            scratch_w,
            gpu_code,
            kv,
            ssm,
            convst,
            graph: None,
            host,
            n: 0,
            work_bytes,
            moe_scratch: std::cell::OnceCell::new(),
        };
        if o.graph && e.w.whole() && e.w.layers.iter().flatten().all(|l| l.moe.as_ref().is_none_or(|b| b.graphable())) {
            e.capture_graph()?;
        }
        Ok(e)
    }

    pub fn describe(&self) -> String {
        let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
        let (free, total) = self.dev.mem_info().unwrap_or((0, 0));
        format!(
            "{} (sm_{}{}): weights {:.3} GiB (layout {}), states {:.3} GiB, KV cache {} positions {:.3} GiB, workspaces {:.3} GiB, {:.2} of {:.2} GiB in use",
            self.dev.name,
            self.dev.cc.0,
            self.dev.cc.1,
            gib(self.w.bytes),
            self.w.layout.name(),
            gib(self.ssm.bytes + self.convst.bytes),
            self.ctx,
            gib(self.kv.bytes),
            gib(self.work_bytes),
            gib(total - free),
            gib(total)
        )
    }

    /// Last forward pass's BF16 logits, widened exactly for development scoring.
    pub fn logits(&self) -> Result<Vec<f32>> {
        self.dev.sync()?;
        let mut words = vec![0u16; self.w.cfg.vocab];
        self.logits.read(0, &mut words)?;
        Ok(words.into_iter().map(|w| f32::from_bits((w as u32) << 16)).collect())
    }

    pub fn has_graph(&self) -> bool {
        self.graph.is_some()
    }

    pub fn chunk(&self) -> usize {
        self.chunk
    }

    /// The positions each attention layer's cache holds.
    pub fn ctx(&self) -> usize {
        self.ctx
    }

    /// A new sequence: the states zeroed (the cache's old positions are never read: a row reads positions up to its own).
    pub fn reset(&mut self) -> Result<()> {
        self.n = 0;
        self.top.ask = None;
        memset_async(self.ssm.ptr, 0, self.ssm.bytes, self.dev.stream)?;
        memset_async(self.convst.ptr, 0, self.convst.bytes, self.dev.stream)?;
        self.dev.sync()
    }

    /// Whether a product of `m` rows by `p` decodes the matrix and multiplies by a dense product (the library's route says DECODE or AHEAD), as Qwen3.5's engine's.
    unsafe fn decoded_way(&self, p: &PackBuf, m: usize) -> Result<bool> {
        if (self.blas.is_none() && !self.own) || m <= 64 || p.pack.cols % 64 != 0 || p.pack.rows % 128 != 0 {
            return Ok(false);
        }
        let route = self.lib.route(self.gpu_code, p.pack.is_twelve(), p.pack.rows, p.pack.cols, m as i64)?;
        Ok(route == 0 || route == 5)
    }

    unsafe fn lin_packed(&self, p: &PackBuf, x: u64, m: usize, y: u64) -> Result<()> {
        if self.decoded_way(p, m)? {
            self.lib.unpack(&p.pack, 0, p.pack.rows, self.scratch_w.ptr, self.dev.stream)?;
            if let Some(b) = &self.blas {
                return b.gemm_nt(self.dev.stream, self.scratch_w.ptr, x, y, p.pack.rows as i32, p.pack.cols as i32, m as i32);
            }
            let tile = self.kern.pick_tile(m, p.pack.rows as usize, p.pack.cols as usize);
            return self.kern.gemm(self.dev.stream, tile, x, self.scratch_w.ptr, y, m as i32, p.pack.rows as i32, p.pack.cols as i32);
        }
        self.lib.linear(&p.pack, x, m as i64, y, self.ws.ptr, self.ws.bytes, self.done.ptr, self.dev.stream)
    }

    /// Y [m, rows] = X [m, cols] W^T, X and Y dense (a mixture-of-experts block's products too).
    ///
    /// # Safety
    /// Live device buffers of those sizes, on the engine's stream.
    pub unsafe fn lin(&self, w: &Mat, x: u64, m: usize, y: u64) -> Result<()> {
        match w {
            Mat::Packed(p) => self.lin_packed(p, x, m, y),
            Mat::Dense(b, rows, cols) => self.k.small_linear(self.dev.stream, x, *cols as i32, b.ptr, y, *rows as i32, m as u32, *rows as i32, *cols as i32),
        }
    }

    /// Token rows into x (as the Qwen3.5 engine's `embed`).
    unsafe fn embed(&self, tokens: &[u32]) -> Result<()> {
        let hid = self.w.cfg.hidden;
        let s = self.dev.stream;
        let table = self.w.embed.as_ref().ok_or_else(|| Error("the embedding is not loaded".into()))?;
        if let Some(&t) = tokens.iter().find(|&&t| t as usize >= self.w.cfg.vocab) {
            return err(format!("token {t}: the vocabulary has {}", self.w.cfg.vocab));
        }
        if tokens.len() == 1 {
            let t = tokens[0] as i64;
            let row0 = (t / 64) * 64;
            self.lib.unpack(&table.pack, row0, 64, self.scratch.ptr, s)?;
            return self.kern.copy_row(s, self.scratch.ptr, (t - row0) as i32, hid as i32, self.x.ptr);
        }
        let (groups, by_batch) = embed_plan(tokens, EMBED_GROUPS);
        let counts: Vec<usize> = by_batch.iter().map(Vec::len).collect();
        let pairs: Vec<i32> = by_batch.iter().flatten().flat_map(|&(d, s_)| [d, s_]).collect();
        self.embed_pairs.write(0, &pairs)?;
        let mut done = 0usize;
        for (b, batch) in groups.chunks(EMBED_GROUPS).enumerate() {
            for (j, &g) in batch.iter().enumerate() {
                self.lib.unpack(&table.pack, g * 64, 64, self.embed_groups.ptr + (j * 64 * hid * 2) as u64, s)?;
            }
            self.kern.copy_rows(s, self.embed_groups.ptr, self.embed_pairs.ptr + (done * 8) as u64, counts[b] as i32, hid as i32, self.x.ptr)?;
            done += counts[b];
        }
        Ok(())
    }

    /// A Mamba-2 layer's mixer on the n rows in `h` (the layer's normed input) into `o`: the in_proj, the convolution, the scan (a prompt's in chunks, a step by the recurrence), the gated norm, the out_proj.
    unsafe fn mamba(&self, ly: &LayerNh, n: usize) -> Result<()> {
        let c = &self.w.cfg;
        let s = self.dev.stream;
        let (di, cd, w) = (c.inner(), c.conv_dim(), c.in_rows() as i32);
        let (nh, ng) = (c.m_heads as i32, c.m_groups as i32);
        let sm = &ly.small;
        let state = self.ssm.ptr + (ly.idx * c.m_heads * NH_P * NH_N * 4) as u64;
        let (z, xbc, dtr) = (self.mix.ptr, self.mix.ptr + (di * 2) as u64, self.mix.ptr + ((di + cd) * 2) as u64);
        self.lin(&ly.mats[0], self.h.ptr, n, self.mix.ptr)?;
        self.k.conv(s, xbc, w, sm[0].ptr, sm[1].ptr, self.convst.ptr + (ly.idx * 3 * cd * 2) as u64, self.xc.ptr, n as i32, cd as i32)?;
        if self.chunked && n > 1 {
            self.k.ssd(s, &self.ssd, self.xc.ptr, cd as i32, dtr, w, sm[4].ptr, sm[2].ptr, sm[3].ptr, state, self.core.ptr, n as i32, nh, ng, 0.0)?;
        } else {
            self.k.ssm(s, self.xc.ptr, cd as i32, dtr, w, sm[4].ptr, sm[2].ptr, sm[3].ptr, state, self.core.ptr, n as i32, nh, ng, 0.0)?;
        }
        self.k.gated_norm(s, self.core.ptr, z, w, sm[5].ptr, n as u32, di as i32, ng, c.eps)?;
        self.lin(&ly.mats[1], self.core.ptr, n, self.o.ptr)
    }

    /// An attention layer's mixer on the n rows in `h` into `o`: the qkv product, the keys and values into the cache at `pos` (the rows' positions, int32 on the device; `start` tokens came before
    /// these rows), the causal attention (a decode step's in parts, a prompt's by the engine's prompt kernel), the o product. No positional encoding.
    unsafe fn attention(&self, ly: &LayerNh, n: usize, pos: u64, start: usize) -> Result<()> {
        let c = &self.w.cfg;
        let s = self.dev.stream;
        let (nq, nk) = (c.heads as i32, c.kv_heads as i32);
        let qkv_w = ((c.heads + 2 * c.kv_heads) * c.head_dim) as i32;
        let kv = self.kv.ptr + (ly.idx * self.ctx * c.kv_heads * 256 * 2) as u64;
        self.lin(&ly.mats[0], self.h.ptr, n, self.mix.ptr)?;
        self.k.kv_write(s, self.mix.ptr, qkv_w, pos, kv, n as u32, nq, nk)?;
        if n == 1 {
            self.k.attn_decode(s, self.mix.ptr, qkv_w, kv, pos, 1, nq, nk, self.attn_part.ptr, self.att.ptr, nq * 128)?;
        } else {
            self.kern.prefill_attn(s, self.mix.ptr, qkv_w, kv, self.att.ptr, nq * 128, n as i32, start as i32, nq, nk)?;
        }
        self.lin(&ly.mats[1], self.att.ptr, n, self.o.ptr)
    }

    /// Layer `l`'s mixer on the n rows in `h` into `o` (a development check's way to one layer).
    ///
    /// # Safety
    /// Live buffers of the engine, sized for a chunk; the stream orders every step.
    pub(crate) unsafe fn mixer(&self, l: usize, n: usize, pos: u64, start: usize) -> Result<()> {
        let ly = self.w.layers[l].as_ref().ok_or_else(|| Error(format!("layer {l} is not loaded")))?;
        match ly.kind {
            NhKind::Mamba => self.mamba(ly, n),
            NhKind::Attention => self.attention(ly, n, pos, start),
            NhKind::Moe => ly.moe.as_ref().ok_or_else(|| Error(format!("layer {l} has no block of experts")))?.forward(self, self.h.ptr, n, self.o.ptr),
        }
    }

    /// The layers on the n rows in x (positions at `pos`, int32 on the device; `start` tokens before them); where `head`, the final norm, the LM head and the argmax on the last row.
    unsafe fn forward(&self, n: usize, pos: u64, start: usize, head: bool) -> Result<()> {
        let c = &self.w.cfg;
        let s = self.dev.stream;
        let (nu, hid_i, eps) = (n as u32, c.hidden as i32, c.eps);
        for (l, slot) in self.w.layers.iter().enumerate() {
            let ly = slot.as_ref().ok_or_else(|| Error(format!("layer {l} is not loaded")))?;
            if l == 0 {
                self.kern.rmsnorm(s, self.x.ptr, ly.norm.ptr, self.h.ptr, nu, hid_i, eps)?;
            } else {
                self.kern.rmsnorm_add(s, self.x.ptr, self.o.ptr, ly.norm.ptr, self.h.ptr, nu, hid_i, eps)?;
            }
            self.mixer(l, n, pos, start)?;
        }
        if head {
            let (norm_f, lm) = (self.w.norm_f.as_ref(), self.w.lm_head());
            let (Some(norm_f), Some(lm)) = (norm_f, lm) else { return err("the final norm or the LM head is not loaded") };
            let last = ((n - 1) * c.hidden * 2) as u64;
            self.kern.rmsnorm_add(s, self.x.ptr + last, self.o.ptr + last, norm_f.ptr, self.hlast.ptr, 1, hid_i, eps)?;
            self.lin_packed(lm, self.hlast.ptr, 1, self.logits.ptr)?;
            self.kern.argmax_rows(s, self.logits.ptr, c.vocab as i32, 1, self.tok.ptr, self.val.ptr)?;
        }
        Ok(())
    }

    /// The decode step as a graph (the position read from `st`).
    fn capture_graph(&mut self) -> Result<()> {
        self.st.write(0, &[0i32])?;
        // one eager step first (lazy initializations out of the capture); it writes position 0 and the states, which `reset` and a prompt rewrite
        // SAFETY: live buffers of the engine; the stream is synchronized after.
        unsafe {
            self.embed(&[0u32])?;
            self.forward(1, self.st.ptr, 0, true)?;
        }
        self.dev.sync()?;
        let g = self.dev.capture(|| {
            // SAFETY: as above; nothing runs during a capture.
            unsafe { self.forward(1, self.st.ptr, 0, true) }
        })?;
        self.graph = Some(g);
        self.reset()
    }

    /// The token chosen after the work queued (the largest logit).
    fn read_sample(&mut self) -> Result<Sample> {
        // SAFETY: the pinned buffer holds 4096 bytes; the copies complete at the synchronize.
        unsafe {
            copy_d2h_async(self.host.ptr, self.tok.ptr, 4, self.dev.stream)?;
            copy_d2h_async((self.host.ptr as *mut u8).add(8) as *mut _, self.val.ptr, 4, self.dev.stream)?;
            self.top.queue(&self.kern, &self.dev, self.logits.ptr, self.val.ptr)?;
        }
        self.dev.sync()?;
        // SAFETY: the pinned bytes were just written.
        let (token, logit) = unsafe { (*(self.host.ptr as *const i32), *((self.host.ptr as *const u8).add(8) as *const f32)) };
        Ok(Sample { token: token as u32, logit, logprob: None })
    }

    /// The prompt through the model after the tokens already in the sequence; the token that follows it.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Sample> {
        if tokens.is_empty() {
            return err("an empty prompt");
        }
        if self.n + tokens.len() + 1 > self.ctx {
            return err(format!("{} tokens and {} more, where the cache holds {}", self.n, tokens.len(), self.ctx));
        }
        let mut at = 0;
        while at < tokens.len() {
            let n = self.chunk.min(tokens.len() - at);
            let pos: Vec<i32> = (0..n).map(|i| (self.n + i) as i32).collect();
            self.dev.sync()?; // (the previous chunk's kernels read the positions until then)
            self.pos.write(0, &pos)?;
            let last = at + n == tokens.len();
            // SAFETY: live buffers of the engine, sized for a chunk; the stream orders every step.
            unsafe {
                self.embed(&tokens[at..at + n])?;
                self.forward(n, self.pos.ptr, self.n, last)?;
            }
            self.n += n;
            at += n;
        }
        self.read_sample()
    }

    /// One decode step: `token` at the next position; the token that follows it.
    pub fn step(&mut self, token: u32) -> Result<Sample> {
        if self.n + 1 > self.ctx {
            return err(format!("the cache holds {} positions", self.ctx));
        }
        let p = [self.n as i32];
        // SAFETY: pinned staging copied before the launches in stream order; live buffers.
        unsafe {
            let hp = (self.host.ptr as *mut u8).add(64);
            std::ptr::copy_nonoverlapping(p.as_ptr() as *const u8, hp, 4);
            copy_h2d_async(self.st.ptr, hp as *const _, 4, self.dev.stream)?;
            self.embed(&[token])?;
            match &self.graph {
                Some(g) => g.launch(self.dev.stream)?,
                None => self.forward(1, self.st.ptr, self.n, true)?,
            }
        }
        self.n += 1;
        self.read_sample()
    }

    /// Greedy tokens: the prompt, then up to `max_new` tokens (stopping at an end-of-sequence token unless `ignore_eos`).
    pub fn generate(&mut self, prompt: &[u32], max_new: usize, ignore_eos: bool) -> Result<Vec<Sample>> {
        self.reset()?;
        let mut out = Vec::with_capacity(max_new);
        let mut s = self.prefill(prompt)?;
        out.push(s);
        while out.len() < max_new {
            if !ignore_eos && self.w.cfg.eos.contains(&(s.token as i64)) {
                break;
            }
            s = self.step(s.token)?;
            out.push(s);
        }
        Ok(out)
    }
}

/// One user's replies (conv.rs's `Talk`), as Qwen3.5's engine gives them: the prompt from a new sequence, then each token greedy or sampled on the host, from the 64 largest logits the GPU makes behind the step where they are all
/// the draw looks at (`TopGpu`), else from the step's logits.
impl crate::conv::Gen for EngineNh {
    fn capacity(&self) -> usize {
        self.ctx
    }

    fn reply(&mut self, prompt: &[u32], o: &crate::model::Reply, each: &mut dyn FnMut(u32) -> bool) -> Result<(Vec<u32>, crate::model::Finish)> {
        use crate::model::Finish;
        if prompt.is_empty() {
            return err("an empty prompt");
        }
        if prompt.len() + 1 > self.ctx {
            return err(format!("{} tokens, where the context holds {}", prompt.len(), self.ctx));
        }
        self.reset()?;
        let greedy = o.params.greedy();
        let mut sampler = crate::sampler::Sampler::new(o.params.clone(), prompt);
        self.top.ask = sampler.top_need();
        self.top.mask = sampler.mask_handle();
        let mut s = self.prefill(prompt)?;
        let mut out = Vec::new();
        loop {
            let tok = if greedy { s.token } else { sampler.sample_with(self.top.read().as_ref(), || self.logits())? };
            if !o.ignore_eos && (self.w.cfg.eos.contains(&(tok as i64)) || o.stop_ids.contains(&tok)) {
                return Ok((out, Finish::Stop));
            }
            out.push(tok);
            if !each(tok) {
                return Ok((out, Finish::Stopped));
            }
            if out.len() >= o.max_new || self.n + 1 > self.ctx {
                return Ok((out, Finish::Length));
            }
            self.top.ask = sampler.top_need();
            self.top.mask = sampler.mask_handle();
            s = self.step(tok)?;
        }
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
