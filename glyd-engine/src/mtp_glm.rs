//! GLM-5.3-Flash's multi-token-prediction head (the checkpoint's layer after the model's own, `num_nextn_predict_layers` 1): it drafts the token after the one the model has just picked from the model's last hidden state, to be
//! verified in one pass beside the model's own (`EngineGlm::verify`). transformers does not implement it; vLLM's `mtp.py` and `Glm5NextDecoderLayer(is_mtp_layer)` are the description, and `tools/glm_ref.py`'s `mtp_forward`
//! the reference of this file's arithmetic. For a row (the final normed hidden state `h` of position j, the token `t` that follows it): `enorm(embedding(t))` and `hnorm(h)` side by side are projected back to the model's
//! width by `eh_proj`; then one latent attention layer with experts (the same blocks as the model's, `GlmMla` and `GlmMoe`, with a cache of their own) on a plain residual stream: no hyper-connections, input norm, the
//! mixer, post-attention norm fused with the residual's addition, the experts and the shared expert, and `shared_head.norm` fused with the last addition; the result is the head's normed hidden state, which the model's LM head
//! turns into the draft token and which is the previous hidden state of the next draft. The head's cache is filled by the rows it is run on, one slot a position of the row's hidden state.

use crate::dev::Buf;
use crate::ffi::{err, Error, Result};
use crate::glm::ConfigGlm;
use crate::model35::Mat;
use crate::model_glm::{EngineGlm, MlaBlock, MlaFactory, MoeBlock, MoeFactory};
use crate::qwen35::Pack35;
use crate::weights::{as_bytes, read_pack, upload_bytes, upload_packed, v7_manifest, Layout};
use glyd_gpu::safetensors::Checkpoint;
use std::cell::OnceCell;
use std::path::Path;

/// The head's scratch for a chunk of rows, and for the logits of up to `ROWS` of them at a time.
struct Scratch {
    emb: Buf,
    cat: Buf,
    x: Buf,
    a: Buf,
    att: Buf,
    m: Buf,
    mo: Buf,
    /// The head's normed hidden states [chunk, hidden] bf16.
    hout: Buf,
    pos: Buf,
    lg: Buf,
    tk: Buf,
    vl: Buf,
}

/// Rows whose logits are made at once.
pub const ROWS: usize = 8;

/// The MTP head on the GPU.
pub struct MtpHead {
    mla: Box<dyn MlaBlock>,
    moe: Box<dyn MoeBlock>,
    eh: Mat,
    enorm: Buf,
    hnorm: Buf,
    in_ln: Buf,
    post_ln: Buf,
    head_norm: Buf,
    sc: OnceCell<Scratch>,
}

fn tensor(ck: &Checkpoint, name: &str, elements: usize) -> Result<Buf> {
    let (_, t) = ck.get(name).ok_or_else(|| Error(format!("no tensor {name}")))?;
    if t.dtype != "BF16" || t.shape.iter().product::<u64>() as usize != elements {
        return err(format!("{name} is {} {:?}, where {elements} BF16 values are wanted", t.dtype, t.shape));
    }
    upload_bytes(&ck.read(name).map_err(|e| Error(format!("{name}: {e}")))?)
}

impl MtpHead {
    /// The head of the model in `dir`: its layer's blocks from the factories the engine was given (the layer is `cfg.layers`), `eh_proj` packed in `layout` where the library packs it, and the four small tensors.
    pub fn load(c: &ConfigGlm, dir: &Path, layout: Layout, moe: Option<MoeFactory<'_>>, mla: Option<MlaFactory<'_>>) -> Result<MtpHead> {
        let ck = Checkpoint::open(dir).map_err(|e| Error(e.to_string()))?;
        let l = c.layers;
        let (Some(moe), Some(mla)) = (moe, mla) else { return err("the MTP head needs the engine's factories of experts and of latent attention") };
        let h = c.hidden;
        if h > 4096 {
            return err(format!("an MTP head over a width of {h}: its norms take up to 4,096"));
        }
        let p = format!("{}layers.{l}", c.prefix);
        let v7 = v7_manifest(dir)?;
        let name = format!("{p}.eh_proj.weight");
        let pk = Pack35 { key: format!("{p}.eh_proj"), parts: vec![(name.clone(), h)], cols: 2 * h };
        let eh = if pk.packable() {
            Mat::Packed(upload_packed(read_pack(&ck, v7.as_ref(), &pk, layout)?)?)
        } else {
            let mut w = vec![0u16; h * 2 * h];
            ck.read_bf16(&name, &mut w).map_err(|e| Error(format!("{name}: {e}")))?;
            Mat::Dense(upload_bytes(as_bytes(&w))?, h, 2 * h)
        };
        Ok(MtpHead {
            mla: mla(l, c, &ck, layout)?,
            moe: moe(l, c, &ck, layout)?,
            eh,
            enorm: tensor(&ck, &format!("{p}.enorm.weight"), h)?,
            hnorm: tensor(&ck, &format!("{p}.hnorm.weight"), h)?,
            in_ln: tensor(&ck, &format!("{p}.input_layernorm.weight"), h)?,
            post_ln: tensor(&ck, &format!("{p}.post_attention_layernorm.weight"), h)?,
            head_norm: tensor(&ck, &format!("{p}.shared_head.norm.weight"), h)?,
            sc: OnceCell::new(),
        })
    }

    /// Device bytes the head's weights take.
    pub fn bytes(&self) -> usize {
        self.mla.bytes()
            + self.moe.bytes()
            + self.eh.bytes()
            + [&self.enorm, &self.hnorm, &self.in_ln, &self.post_ln, &self.head_norm].iter().map(|b| b.bytes).sum::<usize>()
    }

    /// The (rows, columns, whether 12-bit) of the packs the head multiplies through `EngineGlm::lin`, for the engine's workspace.
    pub fn pack_shapes(&self) -> Vec<(i64, i64, bool)> {
        let mut v = self.mla.pack_shapes();
        v.extend(self.moe.pack_shapes());
        if let Mat::Packed(p) = &self.eh {
            v.push((p.pack.rows, p.pack.cols, p.pack.is_twelve()));
        }
        v
    }

    fn scratch(&self, e: &EngineGlm) -> Result<&Scratch> {
        if self.sc.get().is_none() {
            let (ch, h, v) = (e.chunk(), e.w.cfg.hidden, e.w.cfg.vocab);
            let rows = |w: usize| Buf::zeroed(ch * w * 2);
            let s = Scratch {
                emb: rows(h)?,
                cat: rows(2 * h)?,
                x: rows(h)?,
                a: rows(h)?,
                att: rows(h)?,
                m: rows(h)?,
                mo: rows(h)?,
                hout: rows(h)?,
                pos: Buf::zeroed(ch * 4)?,
                lg: Buf::zeroed(ROWS * v * 2)?,
                tk: Buf::zeroed(ROWS * 4)?,
                vl: Buf::zeroed(ROWS * 4)?,
            };
            let _ = self.sc.set(s);
        }
        self.sc.get().ok_or_else(|| Error("the MTP head's scratch is not made".into()))
    }

    /// The head on `tokens.len()` rows: `prev` (device, bf16 [rows, hidden]) the final normed hidden states of the positions `pos0 ..` and `tokens` the tokens after them; the rows go through the head in order, into
    /// its cache at those positions. Returns the device address of the head's normed hidden states [rows, hidden].
    ///
    /// # Safety
    /// `prev` a live device range of that size, used on the engine's stream.
    pub unsafe fn forward(&self, e: &EngineGlm, prev: u64, tokens: &[u32], pos0: usize) -> Result<u64> {
        let n = tokens.len();
        if n == 0 || n > e.chunk() {
            return err(format!("the MTP head on {n} rows, where 1 to {} are run at once", e.chunk()));
        }
        if pos0 + n + 1 > e.ctx() {
            return err(format!("the MTP head at positions {pos0} to {}, where a sequence may reach {}", pos0 + n, e.ctx()));
        }
        let sc = self.scratch(e)?;
        let (s, h, eps) = (e.dev.stream, e.w.cfg.hidden as i32, e.w.cfg.eps);
        let nr = n as u32;
        e.dev.sync()?; // (the head's earlier launches read the positions until then)
        sc.pos.write(0, &(pos0 as i32..(pos0 + n) as i32).collect::<Vec<_>>())?;
        e.embed_into(tokens, sc.emb.ptr)?;
        // the two norms side by side, projected back to the width
        // (the embedding's half first, as vLLM's concatenation is; a development build can ask for the other order, to see which one the checkpoint's projection was trained with)
        let swap = crate::tune!("GLYD_ENGINE_MTP_SWAP").is_some();
        let (at_e, at_h) = if swap { ((h as u64) * 2, 0) } else { (0, (h as u64) * 2) };
        e.k.rms2(s, sc.emb.ptr, h, self.enorm.ptr, sc.cat.ptr + at_e, 2 * h, 0, h, eps, nr)?;
        e.k.rms2(s, prev, h, self.hnorm.ptr, sc.cat.ptr + at_h, 2 * h, 0, h, eps, nr)?;
        e.lin(&self.eh, sc.cat.ptr, n, sc.x.ptr)?;
        // the layer on a plain residual stream: x is the residual
        e.kern.rmsnorm(s, sc.x.ptr, self.in_ln.ptr, sc.a.ptr, nr, h, eps)?;
        self.mla.forward(e, sc.a.ptr, n, sc.pos.ptr, pos0, sc.att.ptr)?;
        e.kern.rmsnorm_add(s, sc.x.ptr, sc.att.ptr, self.post_ln.ptr, sc.m.ptr, nr, h, eps)?;
        self.moe.forward(e, sc.m.ptr, n, sc.mo.ptr)?;
        e.kern.rmsnorm_add(s, sc.x.ptr, sc.mo.ptr, self.head_norm.ptr, sc.hout.ptr, nr, h, eps)?;
        Ok(sc.hout.ptr)
    }

    /// The token the model's LM head picks (the largest logit) for each of `rows` rows of the head's normed hidden states at `hout` (device, bf16 [rows, hidden]), and the logit.
    ///
    /// # Safety
    /// `hout` a live device range of that size.
    pub unsafe fn picks(&self, e: &EngineGlm, hout: u64, rows: usize) -> Result<Vec<(u32, f32)>> {
        if rows == 0 || rows > ROWS {
            return err(format!("the LM head on {rows} rows of the MTP head's, where 1 to {ROWS} are run at once"));
        }
        let sc = self.scratch(e)?;
        let lm = e.w.lm_head().ok_or_else(|| Error("the LM head is not loaded".into()))?;
        e.lin_packed(lm, hout, rows, sc.lg.ptr)?;
        e.kern.argmax_rows(e.dev.stream, sc.lg.ptr, e.w.cfg.vocab as i32, rows as u32, sc.tk.ptr, sc.vl.ptr)?;
        e.dev.sync()?;
        let (mut tk, mut vl) = (vec![0i32; rows], vec![0f32; rows]);
        sc.tk.read(0, &mut tk)?;
        sc.vl.read(0, &mut vl)?;
        Ok(tk.iter().zip(&vl).map(|(&t, &v)| (t as u32, v)).collect())
    }

    /// The head's normed hidden states of the last `forward` (bf16 bits, [rows, hidden]), for a development check.
    pub fn read_hidden(&self, e: &EngineGlm, rows: usize) -> Result<Vec<u16>> {
        e.dev.sync()?;
        let sc = self.scratch(e)?;
        let mut v = vec![0u16; rows * e.w.cfg.hidden];
        sc.hout.read(0, &mut v)?;
        Ok(v)
    }
}
