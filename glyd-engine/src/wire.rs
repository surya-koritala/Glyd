//! The OpenAI JSON of a reply that is more than a string: log-probabilities (a chat's and a plain completion's), a usage object. Written by hand (no serde), as the rest of the server's JSON is.

use crate::sampler::Lp;
use crate::server::jstr;
use crate::tok::Tok;
use std::collections::HashMap;

/// A log-probability as a JSON number: vLLM holds it to -9999 at the lowest (JSON has no infinity).
pub fn lp_num(x: f64) -> String {
    let x = if x.is_nan() { -9999.0 } else { x.max(-9999.0) };
    format!("{x}")
}

/// A token's text as vLLM writes it in log-probabilities: its own decoding, or `token_id:N` where the request asks for ids.
fn token_text(tok: &Tok, id: u32, as_ids: bool, cache: &mut HashMap<u32, String>) -> String {
    if as_ids {
        return format!("token_id:{id}");
    }
    cache.entry(id).or_insert_with(|| tok.decode(&[id]).unwrap_or_default()).clone()
}

fn bytes_json(s: &str) -> String {
    format!("[{}]", s.bytes().map(|b| b.to_string()).collect::<Vec<_>>().join(","))
}

/// A chat's log-probabilities of the tokens `lps`: `{"content": [{"token", "logprob", "bytes", "top_logprobs": [...]}, ...]}`.
pub fn chat_logprobs(tok: &Tok, lps: &[Lp], as_ids: bool) -> String {
    let mut cache = HashMap::new();
    let entries: Vec<String> = lps
        .iter()
        .map(|lp| {
            let t = token_text(tok, lp.token, as_ids, &mut cache);
            let top: Vec<String> = lp
                .top
                .iter()
                .map(|&(id, l)| {
                    let s = token_text(tok, id, as_ids, &mut cache);
                    format!("{{\"token\":{},\"logprob\":{},\"bytes\":{}}}", jstr(&s), lp_num(l), bytes_json(&s))
                })
                .collect();
            format!("{{\"token\":{},\"logprob\":{},\"bytes\":{},\"top_logprobs\":[{}]}}", jstr(&t), lp_num(lp.logprob), bytes_json(&t), top.join(","))
        })
        .collect();
    format!("{{\"content\":[{}]}}", entries.join(","))
}

/// A plain completion's log-probabilities of the tokens `lps`, as the legacy API has them: `text_offset` (where each token's text begins in the reply, from `*offset`, which moves on), `token_logprobs`, `tokens`
/// and `top_logprobs` (for each token an object of the likeliest tokens' text and log-probability).
pub fn text_logprobs(tok: &Tok, lps: &[Lp], as_ids: bool, offset: &mut usize) -> String {
    let mut cache = HashMap::new();
    let (mut offsets, mut lp_list, mut tokens, mut tops) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for lp in lps {
        let t = token_text(tok, lp.token, as_ids, &mut cache);
        offsets.push(offset.to_string());
        *offset += t.chars().count();
        lp_list.push(lp_num(lp.logprob));
        let mut seen: Vec<String> = Vec::new();
        let mut top = Vec::new();
        for &(id, l) in &lp.top {
            let s = token_text(tok, id, as_ids, &mut cache);
            if !seen.contains(&s) {
                top.push(format!("{}:{}", jstr(&s), lp_num(l)));
                seen.push(s);
            }
        }
        tokens.push(jstr(&t));
        tops.push(format!("{{{}}}", top.join(",")));
    }
    format!(
        "{{\"text_offset\":[{}],\"token_logprobs\":[{}],\"tokens\":[{}],\"top_logprobs\":[{}]}}",
        offsets.join(","),
        lp_list.join(","),
        tokens.join(","),
        tops.join(",")
    )
}

/// A usage object: the prompt counted once, the tokens of every choice added.
pub fn usage(prompt: usize, completion: usize) -> String {
    format!("{{\"prompt_tokens\":{prompt},\"completion_tokens\":{completion},\"total_tokens\":{}}}", prompt + completion)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tok::tests::toy;

    #[test]
    fn log_probabilities_as_the_openai_api_writes_them() {
        let t = toy(&["Hi", " there", "é"]);
        let lps = vec![Lp { token: 1, logprob: -0.25, top: vec![(1, -0.25), (2, -1.5)] }, Lp { token: 3, logprob: f64::NEG_INFINITY, top: vec![] }];
        let want = "{\"content\":[{\"token\":\"Hi\",\"logprob\":-0.25,\"bytes\":[72,105],\"top_logprobs\":[{\"token\":\"Hi\",\"logprob\":-0.25,\"bytes\":[72,105]},{\"token\":\" there\",\"logprob\":-1.5,\"bytes\":[32,116,104,101,114,101]}]},{\"token\":\"é\",\"logprob\":-9999,\"bytes\":[195,169],\"top_logprobs\":[]}]}";
        assert_eq!(chat_logprobs(&t, &lps, false), want);
        assert_eq!(chat_logprobs(&t, &lps[..1], true), "{\"content\":[{\"token\":\"token_id:1\",\"logprob\":-0.25,\"bytes\":[116,111,107,101,110,95,105,100,58,49],\"top_logprobs\":[{\"token\":\"token_id:1\",\"logprob\":-0.25,\"bytes\":[116,111,107,101,110,95,105,100,58,49]},{\"token\":\"token_id:2\",\"logprob\":-1.5,\"bytes\":[116,111,107,101,110,95,105,100,58,50]}]}]}");
        // the legacy form: offsets in characters from where the reply's text begins
        let mut at = 0;
        let got = text_logprobs(&t, &lps, false, &mut at);
        assert_eq!(
            got,
            "{\"text_offset\":[0,2],\"token_logprobs\":[-0.25,-9999],\"tokens\":[\"Hi\",\"é\"],\"top_logprobs\":[{\"Hi\":-0.25,\" there\":-1.5},{}]}"
        );
        assert_eq!(at, 3);
        assert_eq!(usage(10, 5), "{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}");
        assert!(glyd_gpu::json::parse(&got).is_ok() && glyd_gpu::json::parse(want).is_ok());
    }
}
