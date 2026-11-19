//! Quality-only execution of the study's rounded values in the native engine.
//! GPU weights use the existing checked lossless format, not the experimental representation.
use glyd_engine::args::Args;
use glyd_engine::ffi::{Error, Result};
use glyd_engine::model35::{Engine35, Opts35};
use glyd_gpu::json::{self, Value};
use std::path::Path;
use std::time::Instant;
use sha2::{Digest, Sha256};

const MODEL: &str = "Qwen/Qwen3.8-27B";
const REVISION: &str = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0";

#[derive(Debug)]
struct Row { id: String, ids: Vec<u32>, start: usize }

fn read(path: &Path) -> Result<Value> {
    json::parse(&std::fs::read_to_string(path).map_err(|e| Error(e.to_string()))?).map_err(|e| Error(e.to_string()))
}
fn round3(w: &mut [u16]) { for v in w { *v = glyd_lossy_study::approximate(*v, 3); } }
fn round2(w: &mut [u16]) { for v in w { *v = glyd_lossy_study::approximate(*v, 2); } }
fn transform(bits: usize) -> Result<Option<fn(&mut [u16])>> {
    match bits { 7 => Ok(None), 3 => Ok(Some(round3)), 2 => Ok(Some(round2)), _ => Err(Error("fraction bits must be 7 (control), 3 or 2".into())) }
}
fn text<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v.get(key).and_then(Value::as_str).ok_or_else(|| Error(format!("missing {key}")))
}
fn rows(v: &Value, vocab: usize, ctx: usize) -> Result<Vec<Row>> {
    if text(v, "model")? != MODEL || text(v, "revision")? != REVISION {
        return Err(Error("evaluation model/revision does not match the pinned checkpoint".into()));
    }
    let list = v.get("samples").and_then(Value::as_array).ok_or_else(|| Error("missing samples".into()))?;
    if list.is_empty() || list.len() > 64 { return Err(Error("expected 1 to 64 samples".into())); }
    let mut seen = std::collections::HashSet::new();
    list.iter().map(|r| {
        let id = text(r,"id")?.to_string();
        if !seen.insert(id.clone()) { return Err(Error("duplicate sample id".into())); }
        let ids = r.get("input_ids").and_then(Value::as_u64s).ok_or_else(|| Error("missing input_ids".into()))?;
        let start = r.get("score_start").and_then(Value::as_u64).ok_or_else(|| Error("missing score_start".into()))?;
        if start == 0 || start as usize >= ids.len() || ids.len() >= ctx || ids.iter().any(|&t| t >= vocab as u64) {
            return Err(Error(format!("{id}: invalid token ids, score interval or context")));
        }
        Ok(Row { id, ids: ids.into_iter().map(|t| t as u32).collect(), start: start as usize })
    }).collect()
}
fn score_logits(logits: &[f32], target: u32) -> Result<(u32, f64)> {
    if logits.is_empty() || logits.iter().any(|x| !x.is_finite()) || target as usize >= logits.len() {
        return Err(Error("invalid or non-finite score logits".into()));
    }
    let mut top = 0;
    for i in 1..logits.len() { if logits[i] > logits[top] { top = i; } }
    let mx = logits[top] as f64;
    let lse = mx + logits.iter().map(|&x| (x as f64 - mx).exp()).sum::<f64>().ln();
    let lp = logits[target as usize] as f64 - lse;
    if !lp.is_finite() { return Err(Error("non-finite target log probability".into())); }
    Ok((top as u32, lp))
}
fn run() -> Result<()> {
    let a = Args::parse();
    let dir = Path::new(a.need(0,"MODEL_DIR")?);
    let eval = Path::new(a.need(1,"EVAL.json")?);
    let out = Path::new(a.need(2,"OUT.json")?);
    let bits = a.num("fraction-bits", 7usize)?;
    let map = transform(bits)?;
    let ctx = a.num("ctx", 1024usize)?;
    let manifest = read(&dir.join("glyd-study-source.json"))?;
    if text(&manifest,"model")? != MODEL || text(&manifest,"revision")? != REVISION {
        return Err(Error("unpinned checkpoint directory".into()));
    }
    let cfg = glyd_engine::qwen35::Config35::load(dir)?;
    let evaluation = read(eval)?;
    let digest = Sha256::digest(std::fs::read(eval).map_err(|e|Error(e.to_string()))?);
    let evaluation_sha256 = digest.iter().map(|b|format!("{b:02x}")).collect::<String>();
    let samples = rows(&evaluation,cfg.vocab,ctx)?;
    let t0 = Instant::now();
    let opts = Opts35 { ctx, chunk: 256, graph: false, weight_transform: map, ..Opts35::default() };
    let mut e = Engine35::new(dir,&opts)?;
    let load_s = t0.elapsed().as_secs_f64();
    eprintln!("{}; study fraction bits {bits}; this GPU storage does not implement the experimental format",e.describe());
    let mut records = Vec::new();
    let mut total_nll = 0.0;
    let mut count = 0;
    for row in &samples {
        e.reset()?;
        e.prefill(&row.ids[..row.start],false)?;
        let mut lp = Vec::new(); let mut top = Vec::new();
        for pos in row.start..row.ids.len() {
            if pos > row.start { e.step(row.ids[pos-1],false)?; }
            let (token,logprob) = score_logits(&e.logits()?,row.ids[pos])?;
            top.push(token); lp.push(logprob); total_nll -= logprob; count += 1;
        }
        let mean = -lp.iter().sum::<f64>() / lp.len() as f64;
        eprintln!("{}: {} scored tokens, mean NLL {mean:.6}",row.id,lp.len());
        records.push(format!("{{\"id\":{},\"score_top1\":[{}],\"score_lp\":[{}],\"mean_nll\":{mean:.9}}}",
            json::to_compact(&Value::String(row.id.clone())),
            top.iter().map(u32::to_string).collect::<Vec<_>>().join(","),
            lp.iter().map(|x|format!("{x:.9}")).collect::<Vec<_>>().join(",")));
    }
    let nll = total_nll/count as f64;
    let ppl = nll.exp();
    if !ppl.is_finite() { return Err(Error("non-finite perplexity".into())); }
    let body=format!("{{\"model\":\"{MODEL}\",\"revision\":\"{REVISION}\",\"evaluation_sha256\":\"{evaluation_sha256}\",\"runtime\":\"glyd-engine\",\"scope\":\"language-only quality screening; existing lossless GPU storage\",\"fraction_bits\":{bits},\"scored_tokens\":{count},\"mean_nll\":{nll:.9},\"perplexity\":{ppl:.9},\"load_s\":{load_s:.3},\"weight_bytes\":{},\"checkpoint_loaded_bytes\":{},\"skipped_vision_mtp_tensors\":{},\"samples\":[{}]}}\n",e.w.bytes,e.w.ckpt_bytes,e.w.skipped,records.join(","));
    std::fs::write(out,body).map_err(|e|Error(e.to_string()))
}
fn main() { if let Err(e)=run() { eprintln!("{e}"); std::process::exit(1); } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn likelihood_uses_the_target_and_stable_normalization() {
        let (top,lp)=score_logits(&[1000.0,1001.0],0).unwrap();
        assert_eq!(top,1); assert!((lp + (1.0+std::f64::consts::E).ln()).abs()<1e-10);
        assert_eq!(score_logits(&[3.0,3.0],1).unwrap().0,0);
        assert!(score_logits(&[f32::NAN],0).is_err());
        assert!(score_logits(&[1.0],1).is_err());
    }
    #[test]
    fn selected_transforms_are_the_codec_reconstruction_values() {
        for m in [7,3,2] {
            let mut words: Vec<u16>=(0..=u16::MAX).collect();
            if let Some(f)=transform(m).unwrap() { f(&mut words); }
            let original: Vec<u16>=(0..=u16::MAX).collect();
            let packed=glyd_lossy_study::encode(&original,m as u8,1024).unwrap();
            assert_eq!(words,glyd_lossy_study::decode(&packed).unwrap());
        }
        assert!(transform(6).is_err());
    }
    #[test]
    fn scoring_input_cannot_silently_truncate_or_change_checkpoint() {
        let good=format!("{{\"model\":\"{MODEL}\",\"revision\":\"{REVISION}\",\"samples\":[{{\"id\":\"x\",\"input_ids\":[1,2,3],\"score_start\":1}}]}}");
        let v=json::parse(&good).unwrap();
        assert_eq!(rows(&v,4,8).unwrap()[0].start,1);
        assert!(rows(&v,3,8).is_err()); assert!(rows(&v,4,3).is_err());
        assert!(rows(&json::parse(&good.replace(REVISION,"other")).unwrap(),4,8).is_err());
        assert!(rows(&json::parse(&good.replace("[1,2,3]","[1,4294967296,3]")).unwrap(),4,8).is_err());
    }
}
