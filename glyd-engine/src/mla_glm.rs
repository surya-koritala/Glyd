//! GLM-5.3-Flash's latent attention layers (11 of the 45) on the GPU: `GlmMla`, the `MlaBlock` that model_glm.rs's layers call. No rotary; a token's cache is its latent (`kv_a_proj_with_mqa`'s product, normed, `kv_lora_rank` wide),
//! which is its key and, through `kv_b_proj`, its value, so the attention runs in the latent space: the query `q_h` becomes `Wk_h^T q_h`, the scores are its products with the latents, and the weighted sum of the latents is
//! multiplied by `Wv_h` at the end (the absorbed form: 1 KiB of cache a token and layer instead of 64 heads' keys and values). Each layer's indexer picks the tokens a query attends to: the keys of the complete pools of
//! `kpool` tokens (a softmax-weighted mean of the pool's normed keys) are scored by `sum_h w_h relu(q_h . k)`, the best `topk / kpool` pools are taken with all their tokens, and the incomplete last pool's tokens too; while
//! a query has no more pools than that it attends to everything before it. The kernels are kernels/glm.cu's (`glm_rms2` ... `glm_expand_v`), the arithmetic `tools/glm_ref.py`'s `mla_*` functions (the structure checked there
//! against transformers 5.19.0's Glm5NextTextAttention and Glm5NextTextIndexer).
//!
//! Every launch reads the rows' positions from the device (`pos`) and none reads a result on the host, so a decode step is captured into the engine's CUDA graph. The caches (the latents, the indexer's keys and gate scores, the
//! pool keys) are made on the first run (the engine's warm-up step) for the engine's context; the scratch, one for all the layers, likewise for its chunk. The matrices that are products of the library (q's two, k and v's down,
//! o) are packed as they load; `kv_b_proj` is held as the bf16 it is (the absorbed form multiplies by its rows), and the indexer's tensors as the checkpoint has them.

use crate::dev::Buf;
use crate::ffi::{err, Error, Result, Words};
use crate::glm::ConfigGlm;
use crate::kern35::Mat;
use crate::model_glm::{mark, GlmHost, MlaBlock};
use crate::qwen35::Pack35;
use crate::weights::{as_bytes, read_pack, upload_bytes, upload_packed, v7_manifest, Layout};
use glyd_gpu::json::Value;
use glyd_gpu::pack::{unpack_int4, unpack_int5, unpack_int6, Tiered, INT4_TAG, INT5_TAG, INT6_TAG};
use glyd_gpu::safetensors::Checkpoint;
use std::cell::{OnceCell, RefCell};

/// A layer's caches, a row a position: the latents (bf16 [ctx, kv_rank]), the indexer's normed keys and gate scores (bf16 [ctx, dim]) and the keys of the complete pools (bf16 [ctx / kpool + 1, dim]).
struct Cache {
    lat: Buf,
    ikc: Buf,
    gsc: Buf,
    pkc: Buf,
}

/// The scratch the layers share, for a chunk of rows: the projections' outputs (`t1` the one before a norm, `qr` the query's latent, `q`, `iq`, `wts`, `ov`) and, for the `sub` rows an attention block takes at a time,
/// the absorbed queries, the indexer's scores, the chosen tokens (`sel`, `cnt`) and the latent outputs.
pub struct MlaScratch {
    t1: Buf,
    qr: Buf,
    q: Buf,
    iq: Buf,
    wts: Buf,
    ov: Buf,
    qa: Buf,
    scores: Buf,
    sel: Buf,
    cnt: Buf,
    olat: Buf,
    /// The partial results of the attention of a few rows (their tokens split over blocks).
    part: Buf,
    /// Rows an attention block takes at a time, the pools a cache holds, the tokens a row attends to at most.
    sub: usize,
    pmax: usize,
    tmax: usize,
}

/// Bytes of the scores and of the latent outputs a block of attention rows may take.
const SUB_SCORES: usize = 256 << 20;
const SUB_OLAT: usize = 64 << 20;

/// A latent attention layer on the GPU.
pub struct GlmMla {
    nh: usize,
    nope: usize,
    vd: usize,
    kr: usize,
    qr: usize,
    ih: usize,
    id: usize,
    kp: usize,
    topk: usize,
    tail: bool,
    eps: f32,
    q_a: Mat,
    q_b: Mat,
    kv_a: Mat,
    o: Mat,
    q_a_ln: Buf,
    kv_a_ln: Buf,
    kv_b: Buf,
    wq_b: Buf,
    wk: Buf,
    k_norm_w: Buf,
    k_norm_b: Buf,
    wproj: Buf,
    ape: Buf,
    ix_gate: Buf,
    bytes: usize,
    /// Each sequence's caches (sched::MAX_USERS of them at most; made at its first call).
    cache: Box<[OnceCell<Cache>]>,
    /// A development check's collector of the stages that live in the scratch of one block of rows (`collect_stages`).
    dump: RefCell<Option<Dump>>,
}

/// The attention's choices and results of every block of rows of the last call, one block after another: `cnt` int32 [rows], `sel` int32 [rows, tokens at most], `olat` fp32 [rows, heads, latent].
#[derive(Default)]
struct Dump {
    cnt: Vec<u8>,
    sel: Vec<u8>,
    olat: Vec<u8>,
}

/// A matrix of the layer as the engine multiplies by it: packed in `layout` (from the FP8 or bf16 checkpoint or a glyd-v7 directory's saved pack) where the library packs it, else as the bf16 it is.
fn load_mat(ck: &Checkpoint, v7: Option<&(Layout, Value)>, pk: &Pack35, layout: Layout) -> Result<Mat> {
    if pk.packable() {
        // (a GPU of a group: its slice of the rows where the matrix is a big one, src/dynamic/tp.rs)
        let cut = crate::dynamic::tp::slicer().filter(|s| s.slices(pk.rows(), pk.cols)).map(|s| s.rows(pk.rows()));
        let mut b = upload_packed(crate::weights::read_pack_rows(ck, v7, pk, layout, cut)?)?;
        if cut.is_some() {
            b.tp_rows = pk.rows();
        }
        return Ok(Mat::Packed(b));
    }
    let (rows, cols) = (pk.rows(), pk.cols);
    let mut w = vec![0u16; rows * cols];
    let mut at = 0;
    for (name, r) in &pk.parts {
        ck.read_bf16(name, &mut w[at..at + r * cols]).map_err(|e| Error(format!("{name}: {e}")))?;
        at += r * cols;
    }
    Ok(Mat::Dense(upload_bytes(as_bytes(&w))?, rows, cols))
}

/// A tensor of the checkpoint, bf16 as it is, on the GPU.
fn tensor(ck: &Checkpoint, name: &str, elements: usize) -> Result<Buf> {
    let (_, t) = ck.get(name).ok_or_else(|| Error(format!("no tensor {name}")))?;
    if t.dtype != "BF16" || t.shape.iter().product::<u64>() as usize != elements {
        return err(format!("{name} is {} {:?}, where {elements} BF16 values are wanted", t.dtype, t.shape));
    }
    upload_bytes(&ck.read(name).map_err(|e| Error(format!("{name}: {e}")))?)
}

impl GlmMla {
    /// Layer `l` of the model `c` from the checkpoint `ck`, its library matrices packed in `layout`. `kv_b_proj` is the checkpoint's bf16 (from a glyd-v7 directory, which holds it packed, the pack decoded: the layer's
    /// numbers are then the pack's).
    pub fn load(l: usize, c: &ConfigGlm, ck: &Checkpoint, layout: Layout) -> Result<GlmMla> {
        if !c.has_layer(l) || c.kind_of(l) != crate::glm::GlmKind::Mla {
            return err(format!("layer {l} is not a latent attention layer"));
        }
        let (m, ix) = (&c.mla, &c.indexer);
        if m.nope > 256 || m.kv_rank > 512 || !m.kv_rank.is_multiple_of(64) || ix.dim > 256 || ix.kpool > 8 || ix.topk / ix.kpool > 1024 || m.q_rank > 4096 {
            return err(format!(
                "latent attention of heads of {}, latents of {}, a query latent of {}, an indexer of channels {} picking {} groups of {} keys: this engine runs heads up to 256, latents of 64s up to 512, a query latent up to \
                 4,096, indexer channels up to 256, groups up to 8 keys and up to 1,024 picked",
                m.nope, m.kv_rank, m.q_rank, ix.dim, ix.topk / ix.kpool.max(1), ix.kpool
            ));
        }
        // the dynamic shared memory of the attention (kv_rank + 2 tokens) and of the indexer's scores must fit the 48 KiB a launch may take without an opt-in
        let tmax = ix.topk + ix.kpool - 1;
        if 4 * (m.kv_rank + 2 * tmax) > 48 << 10 || 4 * (ix.heads * (ix.dim + 1) + ix.heads + 8 * ix.dim) > 48 << 10 {
            return err(format!("latent attention picking {tmax} keys with latents of {} and an indexer of {} heads of {} channels: its blocks need more than 48 KiB of shared memory, which this engine does not ask for", m.kv_rank, ix.heads, ix.dim));
        }
        let v7 = v7_manifest(&ck.dir)?;
        if let Some((got, _)) = &v7 {
            if *got != layout {
                return err(format!("{}: packed in the {got:?} layout, where {layout:?} is asked", ck.dir.display()));
            }
        }
        let packs = c.layer_packs(l);
        let find =
            |suffix: &str| -> Result<&Pack35> { packs.iter().find(|p| p.key.ends_with(suffix)).ok_or_else(|| Error(format!("layer {l}: no pack {suffix}"))) };
        let a = format!("{}layers.{l}.self_attn", c.prefix);
        let (q_a, q_b, kv_a, o) = (
            load_mat(ck, v7.as_ref(), find(".q_a_proj")?, layout)?,
            load_mat(ck, v7.as_ref(), find(".q_b_proj")?, layout)?,
            load_mat(ck, v7.as_ref(), find(".kv_a_proj_with_mqa")?, layout)?,
            load_mat(ck, v7.as_ref(), find(".o_proj")?, layout)?,
        );
        // kv_b_proj: the checkpoint's bf16; where the directory holds it as a pack (a glyd-v7 one), that pack decoded
        let kv_b_name = format!("{a}.kv_b_proj.weight");
        let kv_b = if ck.get(&kv_b_name).is_some() {
            let mut w = vec![0u16; m.heads * (m.nope + m.v) * m.kv_rank];
            ck.read_bf16(&kv_b_name, &mut w).map_err(|e| Error(format!("{kv_b_name}: {e}")))?;
            upload_bytes(as_bytes(&w))?
        } else {
            let pk = find(".kv_b_proj")?;
            upload_bytes(as_bytes(&unpacked(ck, v7.as_ref(), pk, layout)?))?
        };
        let (h, qr) = (c.hidden, m.q_rank);
        let ixn = format!("{a}.indexer");
        let me = GlmMla {
            nh: m.heads,
            nope: m.nope,
            vd: m.v,
            kr: m.kv_rank,
            qr,
            ih: ix.heads,
            id: ix.dim,
            kp: ix.kpool,
            topk: ix.topk,
            tail: ix.tail,
            eps: c.eps,
            q_a,
            q_b,
            kv_a,
            o,
            q_a_ln: tensor(ck, &format!("{a}.q_a_layernorm.weight"), qr)?,
            kv_a_ln: tensor(ck, &format!("{a}.kv_a_layernorm.weight"), m.kv_rank)?,
            kv_b,
            wq_b: tensor(ck, &format!("{ixn}.wq_b.weight"), ix.heads * ix.dim * qr)?,
            wk: tensor(ck, &format!("{ixn}.wk.weight"), ix.dim * h)?,
            k_norm_w: tensor(ck, &format!("{ixn}.k_norm.weight"), ix.dim)?,
            k_norm_b: tensor(ck, &format!("{ixn}.k_norm.bias"), ix.dim)?,
            wproj: tensor(ck, &format!("{ixn}.weights_proj.weight"), ix.heads * h)?,
            ape: tensor(ck, &format!("{ixn}.index_kpool_compress_ape"), ix.kpool * ix.dim)?,
            ix_gate: tensor(ck, &format!("{ixn}.index_kpool_compress_gate"), ix.dim * h)?,
            bytes: 0,
            cache: (0..crate::sched::MAX_USERS).map(|_| OnceCell::new()).collect(),
            dump: RefCell::new(None),
        };
        let bytes = [&me.q_a, &me.q_b, &me.kv_a, &me.o].iter().map(|m| m.bytes()).sum::<usize>()
            + [&me.q_a_ln, &me.kv_a_ln, &me.kv_b, &me.wq_b, &me.wk, &me.k_norm_w, &me.k_norm_b, &me.wproj, &me.ape, &me.ix_gate]
                .iter()
                .map(|b| b.bytes)
                .sum::<usize>();
        Ok(GlmMla { bytes, ..me })
    }

    /// The tokens a row attends to at most: the best pools' and the incomplete last pool's.
    fn tmax(&self) -> usize {
        self.topk + self.kp - 1
    }

    /// The bytes of one sequence's caches at `ctx` tokens.
    fn cache_bytes(&self, ctx: usize) -> usize {
        ctx * self.kr * 2 + 2 * ctx * self.id * 2 + (ctx / self.kp + 1) * self.id * 2
    }

    /// The first sequence's caches (the development checks').
    fn cache(&self, e: &dyn GlmHost) -> Result<&Cache> {
        self.cache_of(e, 0)
    }

    /// Sequence `q`'s caches, made at its first call.
    fn cache_of(&self, e: &dyn GlmHost, q: usize) -> Result<&Cache> {
        let cell = self.cache.get(q).ok_or_else(|| Error(format!("sequence {q}: at most {} hold latent caches", self.cache.len())))?;
        if cell.get().is_none() {
            let ctx = e.ctx();
            let c = Cache {
                lat: Buf::zeroed(ctx * self.kr * 2)?,
                ikc: Buf::zeroed(ctx * self.id * 2)?,
                gsc: Buf::zeroed(ctx * self.id * 2)?,
                pkc: Buf::zeroed((ctx / self.kp + 1) * self.id * 2)?,
            };
            let _ = cell.set(c);
        }
        cell.get().ok_or_else(|| Error("the latent attention cache is not made".into()))
    }

    fn scratch<'a>(&self, e: &'a dyn GlmHost) -> Result<&'a MlaScratch> {
        if e.mla_scratch().get().is_none() {
            let ch = e.chunk();
            let pmax = e.ctx() / self.kp + 1;
            let sub = ch.min((SUB_OLAT / (self.nh * self.kr * 4)).max(1)).min((SUB_SCORES / (pmax * 4)).max(1));
            // (a development build's check can ask for more rows at a time, to see a long call's stages whole)
            let sub = crate::tune!("GLYD_ENGINE_MLA_ROWS").and_then(|v| v.parse::<usize>().ok()).map_or(sub, |v| v.clamp(1, ch));
            let tmax = self.tmax();
            let wide = self.qr.max(self.kr).max(self.id).max(self.ih * self.id);
            let s = MlaScratch {
                t1: Buf::new(ch * wide * 2)?,
                qr: Buf::new(ch * self.qr * 2)?,
                q: Buf::new(ch * self.nh * self.nope * 2)?,
                iq: Buf::new(ch * self.ih * self.id * 2)?,
                wts: Buf::new(ch * self.ih * 2)?,
                ov: Buf::new(ch * self.nh * self.vd * 2)?,
                qa: Buf::new(sub * self.nh * self.kr * 4)?,
                scores: Buf::new(sub * pmax * 4)?,
                sel: Buf::new(sub * tmax * 4)?,
                cnt: Buf::new(sub * 4)?,
                olat: Buf::new(sub * self.nh * self.kr * 4)?,
                part: Buf::new(crate::model_glm::HC_FEW as usize * self.nh * 8 * (self.kr + 2) * 4)?,
                sub,
                pmax,
                tmax,
            };
            let _ = e.mla_scratch().set(s);
        }
        e.mla_scratch().get().ok_or_else(|| Error("the latent attention scratch is not made".into()))
    }
}

/// A pack of `ck` in the `level` layout (a glyd-v7 directory's saved one where `v7` is its manifest and it holds the pack, else packed from the Hub-format tensors) decoded to bf16 rows, as the library decodes it on the GPU.
pub(crate) fn unpacked(ck: &Checkpoint, v7: Option<&(Layout, Value)>, pk: &Pack35, level: Layout) -> Result<Vec<u16>> {
    let p = read_pack(ck, v7, pk, level)?;
    let Words::Tiered { tiers, .. } = p.words else { return err(format!("{}: not in a block layout", pk.key)) };
    let t = Tiered { rows: p.rows, cols: p.cols, data: p.data, blocks: p.b1, block_base: Vec::new(), tiers };
    let mut w = vec![0u16; p.rows * p.cols];
    let ok = match tiers[0] >> 24 {
        INT6_TAG => unpack_int6(&t, &mut w),
        INT5_TAG => unpack_int5(&t, &mut w),
        INT4_TAG => unpack_int4(&t, &mut w),
        tag => return err(format!("{}: a layout of tag {tag}", pk.key)),
    };
    if ok {
        Ok(w)
    } else {
        err(format!("{}: its buffers do not decode", pk.key))
    }
}

fn read_rows(b: &Buf, offset: usize, bytes: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; bytes];
    b.read(offset, &mut v)?;
    Ok(v)
}

impl MlaBlock for GlmMla {
    /// The layer's mixer on the n rows of `h` (the normed input) at the positions `pos` holds, into `out`: the query's and the token's latents (the latent into the cache at its position), the indexer's queries, keys, gate
    /// scores and head weights (the keys and scores into their caches, the pools the call completes keyed), then for the rows `sub` at a time the absorbed queries, the indexer's scores and choice, the attention over the
    /// chosen latents and its values, and the o product.
    unsafe fn forward(&self, e: &dyn GlmHost, h: u64, n: usize, pos: u64, _start: usize, out: u64) -> Result<()> {
        let sc = self.scratch(e)?;
        // (each sequence's caches its own: a call over several sequences writes and reads each one's rows in its own; the products run over every row)
        let seqs = e.seq_rows(n);
        let cas = seqs.iter().map(|&(_, _, q)| self.cache_of(e, q)).collect::<Result<Vec<&Cache>>>()?;
        let s = e.stream();
        let (nh, nope, vd, kr, qr, ih, id, kp) =
            (self.nh as i32, self.nope as i32, self.vd as i32, self.kr as i32, self.qr as i32, self.ih as i32, self.id as i32, self.kp as i32);
        let hid = e.hidden() as i32;
        let nr = n as u32;
        // (a development check's call of more rows than a block: the stages of each block are kept; never in a graph's capture, whose calls are single rows)
        let collecting = n > sc.sub && self.dump.borrow().is_some();
        if collecting {
            *self.dump.borrow_mut() = Some(Dump::default());
        }
        mark!(e, "mla start");
        // the query's latent, the query
        e.lin(&self.q_a, h, n, sc.t1.ptr)?;
        e.kg().rms2(s, sc.t1.ptr, qr, self.q_a_ln.ptr, sc.qr.ptr, qr, 0, qr, self.eps, nr)?;
        e.lin(&self.q_b, sc.qr.ptr, n, sc.q.ptr)?;
        mark!(e, "mla query (q_a, q_b)");
        // the token's latent into the cache
        e.lin(&self.kv_a, h, n, sc.t1.ptr)?;
        for (&(r0, m, _), ca) in seqs.iter().zip(&cas) {
            e.kg().rms2(s, sc.t1.ptr + (r0 * self.kr * 2) as u64, kr, self.kv_a_ln.ptr, ca.lat.ptr, kr, pos + (r0 * 4) as u64, kr, self.eps, m as u32)?;
        }
        mark!(e, "mla latent (kv_a)");
        // the indexer: queries and head weights for the rows, keys and gate scores into the caches
        e.kg().small_linear(s, sc.qr.ptr, qr, self.wq_b.ptr, sc.iq.ptr, ih * id, nr, ih * id, qr)?;
        e.kg().small_linear(s, h, hid, self.wk.ptr, sc.t1.ptr, id, nr, id, hid)?;
        for (&(r0, m, _), ca) in seqs.iter().zip(&cas) {
            let t = sc.t1.ptr + (r0 * self.id * 2) as u64;
            e.kg().layernorm(s, t, id, self.k_norm_w.ptr, self.k_norm_b.ptr, ca.ikc.ptr, id, pos + (r0 * 4) as u64, id, 1e-6, m as u32)?;
        }
        e.kg().small_linear(s, h, hid, self.ix_gate.ptr, sc.t1.ptr, id, nr, id, hid)?;
        for (&(r0, m, _), ca) in seqs.iter().zip(&cas) {
            e.kg().scatter(s, sc.t1.ptr + (r0 * self.id * 2) as u64, id, ca.gsc.ptr, id, pos + (r0 * 4) as u64, id, m as u32)?;
        }
        e.kg().small_linear(s, h, hid, self.wproj.ptr, sc.wts.ptr, ih, nr, ih, hid)?;
        for (&(r0, m, _), ca) in seqs.iter().zip(&cas) {
            e.kg().kp_keys(s, ca.ikc.ptr, ca.gsc.ptr, self.ape.ptr, ca.pkc.ptr, pos + (r0 * 4) as u64, m as i32, id, kp)?;
        }
        mark!(e, "mla indexer products and key groups");
        // the attention, a block of rows at a time
        let (scale, iscale, wscale) = ((self.nope as f32).powf(-0.5), (self.id as f32).powf(-0.5), (self.ih as f32).powf(-0.5));
        let ksel = (self.topk / self.kp) as i32;
        for (&(first, rows, _), ca) in seqs.iter().zip(&cas) {
            let mut r0 = first;
            while r0 < first + rows {
                let m = sc.sub.min(first + rows - r0);
                let mr = m as u32;
                let (p0, q0) = (pos + (r0 * 4) as u64, sc.q.ptr + (r0 * self.nh * self.nope * 2) as u64);
                e.kg().absorb(s, q0, nh * nope, self.kv_b.ptr, sc.qa.ptr, m as i32, nh, nope, vd, kr)?;
                mark!(e, "mla absorb");
                e.kg().idx_score(
                    s,
                    sc.iq.ptr + (r0 * self.ih * self.id * 2) as u64,
                    ih * id,
                    sc.wts.ptr + (r0 * self.ih * 2) as u64,
                    ih,
                    ca.pkc.ptr,
                    p0,
                    sc.scores.ptr,
                    sc.pmax as i32,
                    ih,
                    id,
                    kp,
                    ksel,
                    iscale,
                    wscale,
                    mr,
                )?;
                mark!(e, "mla indexer scores");
                e.kg().select(s, sc.scores.ptr, sc.pmax as i32, p0, sc.sel.ptr, sc.cnt.ptr, sc.tmax as i32, kp, ksel, self.tail, mr)?;
                mark!(e, "mla select");
                e.kg().mla_attn(s, sc.qa.ptr, ca.lat.ptr, sc.sel.ptr, sc.cnt.ptr, sc.olat.ptr, sc.part.ptr, sc.tmax as i32, nh, kr, scale, mr)?;
                mark!(e, "mla attention");
                e.kg().expand_v(s, sc.olat.ptr, self.kv_b.ptr, sc.ov.ptr + (r0 * self.nh * self.vd * 2) as u64, m as i32, nh, nope, vd, kr)?;
                mark!(e, "mla values");
                if let (true, Some(d)) = (collecting, self.dump.borrow_mut().as_mut()) {
                    e.sync()?;
                    d.cnt.extend(read_rows(&sc.cnt, 0, m * 4)?);
                    d.sel.extend(read_rows(&sc.sel, 0, m * sc.tmax * 4)?);
                    d.olat.extend(read_rows(&sc.olat, 0, m * self.nh * self.kr * 4)?);
                }
                r0 += m;
            }
        }
        e.lin(&self.o, sc.ov.ptr, n, out)
    }

    fn graphable(&self) -> bool {
        true
    }

    fn seq_bytes(&self, ctx: usize) -> usize {
        self.cache_bytes(ctx)
    }

    fn bytes(&self) -> usize {
        self.bytes
    }

    fn pack_shapes(&self) -> Vec<(i64, i64, bool)> {
        [&self.q_a, &self.q_b, &self.kv_a, &self.o]
            .iter()
            .filter_map(|m| if let Mat::Packed(p) = m { Some((p.pack.rows, p.pack.cols, p.pack.is_twelve())) } else { None })
            .collect()
    }

    /// The call's rows of the stages: `qr`, `q`, `iq`, `iw` (the indexer's head weights), `ov` (bf16), `c`, `ik`, `gs` (the caches' rows at the call's positions, bf16), `cnt` (int32), `sel` (int32 [rows, tokens at most]), `olat` (fp32). The
    /// scratch of the attention stages holds the last block of rows: a call of more rows than a block shows those only (the development checks' chunks are smaller than the block).
    fn stages(&self, e: &dyn GlmHost, n: usize, pos0: usize) -> Result<Vec<(&'static str, Vec<u8>)>> {
        e.sync()?;
        let (sc, ca) = (self.scratch(e)?, self.cache(e)?);
        let dump = self.dump.borrow();
        let (cnt, sel, olat) = if n > sc.sub {
            match dump.as_ref().filter(|d| d.cnt.len() == n * 4) {
                Some(d) => (d.cnt.clone(), d.sel.clone(), d.olat.clone()),
                None => return err(format!("a call of {n} rows, where the attention takes {} at a time (the check asks the block to keep the stages of every block of rows: `collect_stages`)", sc.sub)),
            }
        } else {
            (read_rows(&sc.cnt, 0, n * 4)?, read_rows(&sc.sel, 0, n * sc.tmax * 4)?, read_rows(&sc.olat, 0, n * self.nh * self.kr * 4)?)
        };
        Ok(vec![
            ("qr", read_rows(&sc.qr, 0, n * self.qr * 2)?),
            ("q", read_rows(&sc.q, 0, n * self.nh * self.nope * 2)?),
            ("c", read_rows(&ca.lat, pos0 * self.kr * 2, n * self.kr * 2)?),
            ("iq", read_rows(&sc.iq, 0, n * self.ih * self.id * 2)?),
            ("ik", read_rows(&ca.ikc, pos0 * self.id * 2, n * self.id * 2)?),
            ("gs", read_rows(&ca.gsc, pos0 * self.id * 2, n * self.id * 2)?),
            ("iw", read_rows(&sc.wts, 0, n * self.ih * 2)?),
            ("cnt", cnt),
            ("sel", sel),
            ("olat", olat),
            ("ov", read_rows(&sc.ov, 0, n * self.nh * self.vd * 2)?),
        ])
    }

    fn collect_stages(&self, on: bool) {
        *self.dump.borrow_mut() = on.then(Dump::default);
    }

    fn finish(&self, e: &dyn GlmHost, total: usize) -> Result<Vec<(&'static str, Vec<u8>)>> {
        e.sync()?;
        let ca = self.cache(e)?;
        Ok(vec![("pk", read_rows(&ca.pkc, 0, (total / self.kp) * self.id * 2)?)])
    }
}

/// The factory `EngineGlm::new` takes: each latent attention layer is a `GlmMla`, its library matrices in the layout the engine chose.
pub fn factory() -> impl Fn(usize, &ConfigGlm, &Checkpoint, Layout) -> Result<Box<dyn MlaBlock>> {
    |l, c, ck, layout| Ok(Box::new(GlmMla::load(l, c, ck, layout)?))
}
