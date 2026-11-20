//! A multi-token-prediction draft head (`kinds::MtpHead`), DeepSeek-V3's form that GLM-5.3, DeepSeek-V4 and Qwen3.8 share: for the
//! model's final normed hidden state `h` at position j and the token `t` after it, x = eh_proj([norm_embed(embedding(t)),
//! norm_hidden(h)]) (the embedding first), then the head's blocks on a plain residual stream with a cache of their own, then
//! final_norm; the model's LM head turns the result into the draft, and the result is the next draft's `h`.
//!
//! Roles: `eh_proj` [hidden, 2 hidden], `norm_embed`, `norm_hidden`, `final_norm` [hidden]; each block's roles as a block of the model.

use super::{norm::norm, Params, RefLayer};
use crate::ffi::{err, Result};
use crate::runtime::kinds::MtpHead;

pub struct RefMtp {
    pub head: MtpHead,
    pub hidden: usize,
    pub p: Params,
    pub blocks: Vec<RefLayer>,
}

impl RefMtp {
    /// `blocks`: the head's blocks with their weights, in order.
    pub fn new(head: MtpHead, hidden: usize, p: Params, blocks: Vec<RefLayer>) -> Result<RefMtp> {
        if blocks.len() != head.blocks.len() || blocks.iter().zip(&head.blocks).any(|(l, b)| l.block != *b) {
            return err("an MTP head's layers are not its blocks");
        }
        Ok(RefMtp { head, hidden, p, blocks })
    }

    /// One row: the embedding of the next token, the model's hidden state before it, at `pos` of sequence `seq`; the head's normed hidden
    /// state out.
    pub fn step(&mut self, seq: u32, pos: u32, embedding: &[f32], hidden: &[f32]) -> Result<Vec<f32>> {
        let d = self.hidden;
        if embedding.len() != d || hidden.len() != d {
            return err(format!("an MTP row of {} and {} values for width {d}", embedding.len(), hidden.len()));
        }
        let mut cat = norm(&self.head.norm_embed, embedding, self.p.opt("norm_embed"), None);
        cat.extend(norm(&self.head.norm_hidden, hidden, self.p.opt("norm_hidden"), None));
        let mut x = self.p.prod("eh_proj", &cat)?;
        for b in &mut self.blocks {
            b.forward(&[pos], &[seq], &[], &mut x)?;
        }
        Ok(norm(&self.head.final_norm, &x, self.p.opt("final_norm"), None))
    }
}
