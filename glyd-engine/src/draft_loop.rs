//! The decode loop of `glmserve` with drafts (a development session's): after the prompt, greedy decoding to `max_new` tokens or a stop token, a plain step at a time or, where a draft pays, a pass that verifies the draft's rows
//! at once (`EngineGlm::verify` and `commit`: the model keeps the guesses it picks itself, so the tokens are the plain steps' whatever the guesses are). The drafts come from the model's multi-token-prediction head
//! (`mtp_request`); drafting from the text before was taken out (it followed another project's code, and Glyd's code is its own). `glm_serve.rs` calls `decode` and `mtp_request` and nothing else of them.

use crate::ffi::Result;
use crate::model::Sample;
use crate::model_glm::EngineGlm;
use crate::offload::ExpertStore;
use std::rc::Rc;
use std::time::Instant;

/// The session's drafting from the text before: none (taken out; `--lookup` and `{"set": {"lookup": true}}` are refused).
pub struct Drafting {
    pub on: bool,
}

impl Drafting {
    pub fn new() -> Drafting {
        Drafting { on: false }
    }
}

/// A traced decode step: the parts of each layer, the stages, the step's milliseconds (src/trace.rs).
pub type TracedStep = (Vec<crate::trace::Layer>, Vec<(&'static str, f64)>, f64);

/// What a decode produced: the tokens (the first is `first`'s), the milliseconds each of them took (a pass's split evenly), the traced steps, and the drafts' passes (passes, guesses made, guesses kept).
pub struct Decoded {
    pub made: Vec<u32>,
    pub times: Vec<f64>,
    pub traced: Vec<TracedStep>,
    pub passes: (usize, usize, usize),
}

/// The tokens after `prompt` (already through the model: `first` is the token it picked), up to `max_new` in all or a token of `stop`; `trace_level` 0, 1 or 2 traces the plain steps (src/trace.rs).
pub fn decode(e: &mut EngineGlm, _draft: &mut Drafting, trace_level: u8, _prompt: &[u32], first: Sample, max_new: usize, stop: &[u32]) -> Result<Decoded> {
    let mut s = first;
    let mut out = Decoded { made: vec![s.token], times: Vec::new(), traced: Vec::new(), passes: (0, 0, 0) };
    while out.made.len() < max_new && !stop.contains(&s.token) {
        let t = Instant::now();
        if trace_level > 0 {
            crate::trace::start(trace_level > 1);
        }
        s = e.step(s.token)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        out.times.push(ms);
        if trace_level > 0 {
            let marks = crate::trace::stop();
            out.traced.push((crate::trace::layers(&marks), crate::trace::stages(&marks), ms));
        }
        out.made.push(s.token);
    }
    Ok(out)
}

/// The reply to a request `{"ids": [...], "max_new": N, "mtp": K}` (`--mtp`): rounds of K guesses from the model's multi-token-prediction head, each verified in one pass; the tokens are the greedy ones (cut after the first of `stop`).
pub fn mtp_request(e: &mut EngineGlm, store: &Rc<ExpertStore>, ids: &[u32], max_new: usize, k: usize, stop: &[u32]) -> Result<String> {
    let t1 = Instant::now();
    let st0 = store.stats().unwrap_or_default();
    let (out, st) = e.generate_mtp(ids, max_new.max(1), k.max(1))?;
    let ms = t1.elapsed().as_secs_f64() * 1e3;
    let mut toks: Vec<u32> = out.iter().map(|x| x.token).collect();
    if let Some(i) = toks.iter().position(|t| stop.contains(t)) {
        toks.truncate(i + 1);
    }
    let st1 = store.stats().unwrap_or_default();
    eprintln!(
        "{} prompt tokens, {} tokens in {ms:.0} ms ({:.1} ms a token, prompt included); {} passes with {} guesses, {} kept ({:.2} tokens a pass); {:.1}% of the expert reads found in places, {:.1} pairs on the host a pass",
        ids.len(),
        toks.len(),
        ms / toks.len().max(1) as f64,
        st.passes,
        st.drafted,
        st.accepted,
        (st.accepted + st.passes) as f64 / st.passes.max(1) as f64,
        100.0 * (st1.hits - st0.hits) as f64 / ((st1.reads - st0.reads) as f64).max(1.0),
        (st1.cpu_pairs - st0.cpu_pairs) as f64 / st.passes.max(1) as f64
    );
    let list = toks.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
    Ok(format!("{{\"ids\":[{list}],\"passes\":{},\"drafted\":{},\"accepted\":{},\"ms\":{ms:.1}}}", st.passes, st.drafted, st.accepted))
}
