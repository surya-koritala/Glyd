//! A GGUF's tokenizer as the files transformers would have (tokenizer.json, tokenizer_config.json), from its vocabulary, merges or scores, token types, special-token ids and chat template, so that
//! `Tok::load` and `Chat::load` read it as they read any model's. Two kinds are read: byte-level BPE (`tokenizer.ggml.model` "gpt2", with the pre-tokenizer `tokenizer.ggml.pre` names:
//! src/gguf_pre.rs) and SentencePiece (`llama`: Llama 2, Mistral, Gemma), whose merges are derived from the pieces' scores as transformers' SentencePiece converter derives them. Any other kind
//! is refused, naming it.

use crate::gguf::{Gguf, Value};
use crate::gguf_pre::PRE;
use crate::ui::{Done, Refusal};
use glyd_gpu::json::{self, Value as J};
use std::collections::HashMap;

/// llama.cpp's token types: 2 unknown, 3 control, 4 user defined, 5 unused, 6 byte.
const UNKNOWN: u64 = 2;
const CONTROL: u64 = 3;
const USER: u64 = 4;
const UNUSED: u64 = 5;

fn s(x: &str) -> J {
    J::String(x.to_string())
}

fn n(x: impl ToString) -> J {
    J::Number(x.to_string())
}

fn obj(v: Vec<(&str, J)>) -> J {
    J::Object(v.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn refuse(why: String) -> Refusal {
    Refusal::unsupported(why, "Run the model's Hugging Face checkpoint (safetensors) instead")
}

/// The tokenizer.json and the tokenizer_config.json of the model `g` describes.
pub fn files(g: &Gguf) -> Done<(String, String)> {
    let model = g.str("tokenizer.ggml.model").unwrap_or("");
    let list = |k: &str| -> Done<&[Value]> { g.get(k).and_then(Value::as_array).ok_or_else(|| Refusal::new(format!("the GGUF has no {k}"), "")) };
    let tokens = list("tokenizer.ggml.tokens")?;
    let mut pieces = Vec::with_capacity(tokens.len());
    for (i, t) in tokens.iter().enumerate() {
        match t.as_str() {
            Some(p) => pieces.push(p),
            None => return Err(Refusal::new(format!("tokenizer.ggml.tokens[{i}] is not a string"), "")),
        }
    }
    let types = g.get("tokenizer.ggml.token_type").and_then(Value::as_array);
    let ty = |i: usize| types.and_then(|t| t.get(i)).and_then(Value::as_u64).unwrap_or(1);
    let tokenizer = match model {
        "gpt2" => bpe(g, &pieces, &ty, list("tokenizer.ggml.merges")?)?,
        "llama" => spm(g, &pieces, &ty)?,
        other => {
            return Err(refuse(format!(
                "the GGUF's tokenizer is {other:?}: the engine reads byte-level BPE (\"gpt2\") and SentencePiece (\"llama\") tokenizers from GGUF so far"
            )))
        }
    };
    Ok((json::to_compact(&tokenizer), config(g, &pieces)))
}

/// tokenizer_config.json: the chat template and the special tokens' text.
fn config(g: &Gguf, pieces: &[&str]) -> String {
    let token = |k: &str| g.u64(k).and_then(|i| pieces.get(i as usize)).map(|t| s(t));
    let mut cfg: Vec<(String, J)> = Vec::new();
    if let Some(t) = g.str("tokenizer.chat_template") {
        cfg.push(("chat_template".into(), s(t)));
    }
    for (key, id) in [("bos_token", "bos_token_id"), ("eos_token", "eos_token_id"), ("pad_token", "padding_token_id"), ("unk_token", "unknown_token_id")] {
        cfg.extend(token(&format!("tokenizer.ggml.{id}")).map(|t| (key.to_string(), t)));
    }
    cfg.push(("add_bos_token".into(), J::Bool(g.get("tokenizer.ggml.add_bos_token").and_then(Value::as_bool).unwrap_or(false))));
    json::to_python(&J::Object(cfg), 2)
}

/// The tokens the tokenizer is told of apart from the vocabulary's: control and user-defined ones (the control ones special).
fn added(pieces: &[&str], ty: &dyn Fn(usize) -> u64, in_vocab: bool) -> Vec<J> {
    let mut v = Vec::new();
    for (i, p) in pieces.iter().enumerate() {
        let t = ty(i);
        if matches!(t, CONTROL | USER) || (in_vocab && t == UNKNOWN) {
            v.push(obj(vec![
                ("id", n(i)),
                ("content", s(p)),
                ("single_word", J::Bool(false)),
                ("lstrip", J::Bool(false)),
                ("rstrip", J::Bool(false)),
                ("normalized", J::Bool(false)),
                ("special", J::Bool(t != USER)),
            ]));
        }
    }
    v
}

/// A post-processor that puts the beginning-of-sequence (and the end-of-sequence) token round a text, as the metadata says (`add_bos_token`, `add_eos_token`), or None.
fn template(g: &Gguf, pieces: &[&str], default_bos: bool) -> Option<J> {
    let flag = |k: &str, d: bool| g.get(k).and_then(Value::as_bool).unwrap_or(d);
    let tok = |k: &str, on: bool| on.then(|| g.u64(k)).flatten().and_then(|i| pieces.get(i as usize).map(|p| (i, p.to_string())));
    let bos = tok("tokenizer.ggml.bos_token_id", flag("tokenizer.ggml.add_bos_token", default_bos));
    let eos = tok("tokenizer.ggml.eos_token_id", flag("tokenizer.ggml.add_eos_token", false));
    if bos.is_none() && eos.is_none() {
        return None;
    }
    let special = |p: &str, t: u32| obj(vec![("SpecialToken", obj(vec![("id", s(p)), ("type_id", n(t))]))]);
    let seq = |id: &str, t: u32| obj(vec![("Sequence", obj(vec![("id", s(id)), ("type_id", n(t))]))]);
    let side = |a: &str, t: u32| -> Vec<J> { bos.iter().map(|b| special(&b.1, t)).chain([seq(a, t)]).chain(eos.iter().map(|e| special(&e.1, t))).collect() };
    let pair: Vec<J> = side("A", 0).into_iter().chain(side("B", 1)).collect();
    let mut specials: Vec<(String, J)> = Vec::new();
    for (id, p) in bos.iter().chain(eos.iter()) {
        if !specials.iter().any(|(k, _)| k == p) {
            specials.push((p.clone(), obj(vec![("id", s(p)), ("ids", J::Array(vec![n(id)])), ("tokens", J::Array(vec![s(p)]))])));
        }
    }
    Some(obj(vec![("type", s("TemplateProcessing")), ("single", J::Array(side("A", 0))), ("pair", J::Array(pair)), ("special_tokens", J::Object(specials))]))
}

fn byte_level(regex: bool) -> J {
    obj(vec![("type", s("ByteLevel")), ("add_prefix_space", J::Bool(false)), ("trim_offsets", J::Bool(false)), ("use_regex", J::Bool(regex))])
}

fn with_template(post: J, template: Option<J>) -> J {
    match template {
        Some(t) => obj(vec![("type", s("Sequence")), ("processors", J::Array(vec![post, t]))]),
        None => post,
    }
}

/// Byte-level BPE: the vocabulary and merges as they are, the pre-tokenizer the name says.
fn bpe(g: &Gguf, pieces: &[&str], ty: &dyn Fn(usize) -> u64, merges: &[Value]) -> Done<J> {
    let pre = g.str("tokenizer.ggml.pre").unwrap_or("default");
    let Some((_, regexes, ignore_merges, add_bos)) = PRE.iter().find(|(names, _, _, _)| names.contains(&pre)) else {
        return Err(refuse(format!(
            "the GGUF's byte-level BPE tokenizer has the pre-tokenizer {pre:?}, which the engine does not have (src/gguf_pre.rs has llama.cpp's others)"
        )));
    };
    let mut vocab = Vec::with_capacity(pieces.len());
    for (i, p) in pieces.iter().enumerate() {
        // (control and user-defined tokens are the added tokens', the unused ones, the padding up to the embedding's size, and the unknown ones are not in the vocabulary)
        if !matches!(ty(i), CONTROL | USER | UNUSED | UNKNOWN) {
            vocab.push((p.to_string(), n(i)));
        }
    }
    let splits: Vec<J> = regexes
        .iter()
        .map(|r| obj(vec![("type", s("Split")), ("pattern", obj(vec![("Regex", s(r))])), ("behavior", s("Isolated")), ("invert", J::Bool(false))]))
        .chain([byte_level(false)])
        .collect();
    // (Qwen's tokenizers normalize to NFC; llama.cpp's reading of the same vocabulary does not, and a text already in NFC is the same to both)
    let nfc = pre.starts_with("qwen") || pre == "deepseek-r1-qwen";
    let merges: Vec<J> = merges.iter().map(|m| s(m.as_str().unwrap_or(""))).collect();
    Ok(obj(vec![
        ("version", s("1.0")),
        ("truncation", J::Null),
        ("padding", J::Null),
        ("added_tokens", J::Array(added(pieces, ty, false))),
        ("normalizer", if nfc { obj(vec![("type", s("NFC"))]) } else { J::Null }),
        ("pre_tokenizer", obj(vec![("type", s("Sequence")), ("pretokenizers", J::Array(splits))])),
        ("post_processor", with_template(byte_level(false), template(g, pieces, *add_bos))),
        ("decoder", byte_level(false)),
        (
            "model",
            obj(vec![
                ("type", s("BPE")),
                ("dropout", J::Null),
                ("unk_token", J::Null),
                ("continuing_subword_prefix", s("")),
                ("end_of_word_suffix", s("")),
                ("fuse_unk", J::Bool(false)),
                ("byte_fallback", J::Bool(false)),
                ("ignore_merges", J::Bool(*ignore_merges)),
                ("vocab", J::Object(vocab)),
                ("merges", J::Array(merges)),
            ]),
        ),
    ]))
}

/// The merges of a SentencePiece vocabulary, in the order transformers' converter (`generate_merges`) gives them: for each piece, each way to split it into two pieces that are both in the
/// vocabulary (in the order of the pieces' ids), all sorted by the merged piece's score, the pieces' lengths in characters, from the highest, the ties as they were.
pub fn spm_merges(pieces: &[&str], scores: &[f64]) -> Vec<(String, String)> {
    // (a dict in Python: the id of a piece listed twice is the last's, its place the first's and its score the last's)
    let mut id: HashMap<&str, usize> = HashMap::with_capacity(pieces.len());
    let mut order: Vec<&str> = Vec::with_capacity(pieces.len());
    let mut score: HashMap<&str, f64> = HashMap::with_capacity(pieces.len());
    for (i, p) in pieces.iter().enumerate() {
        if id.insert(p, i).is_none() {
            order.push(p);
        }
        score.insert(p, scores[i]);
    }
    let mut all: Vec<(&str, &str, f64, usize, usize)> = Vec::new();
    for &piece in &order {
        let cuts: Vec<usize> = piece.char_indices().map(|(i, _)| i).skip(1).collect();
        let mut local: Vec<(&str, &str)> = cuts.iter().map(|&c| (&piece[..c], &piece[c..])).filter(|(l, r)| id.contains_key(l) && id.contains_key(r)).collect();
        local.sort_by_key(|(l, r)| (id[l], id[r]));
        let sc = score[piece];
        all.extend(local.into_iter().map(|(l, r)| (l, r, sc, l.chars().count(), r.chars().count())));
    }
    // (scores that tell the pieces apart rank the merges; a file whose scores were not kept, all zero as some converters write them, has the pieces' ids, which SentencePiece gives in the order its
    // merges were made)
    let distinct = scores.iter().map(|x| x.to_bits()).collect::<std::collections::HashSet<_>>().len();
    if distinct <= 1 || distinct * 100 < pieces.len() {
        all.sort_by_key(|m| id[&format!("{}{}", m.0, m.1)[..]]);
    } else {
        all.sort_by(|a, b| (b.2, b.3, b.4).partial_cmp(&(a.2, a.3, a.4)).unwrap_or(std::cmp::Ordering::Equal));
    }
    all.into_iter().map(|(l, r, _, _, _)| (l.to_string(), r.to_string())).collect()
}

/// SentencePiece (llama.cpp's `llama`): the pieces with merges derived from their scores, a space written as ▁ (and a ▁ put before the text where the file says `add_space_prefix`, which a
/// Llama or Mistral does and a Gemma does not), the bytes no piece has as the byte pieces `<0xNN>`.
fn spm(g: &Gguf, pieces: &[&str], ty: &dyn Fn(usize) -> u64) -> Done<J> {
    let scores: Vec<f64> = match g.get("tokenizer.ggml.scores").and_then(Value::as_array) {
        Some(a) if a.len() == pieces.len() => a.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect(),
        _ => return Err(Refusal::new("the GGUF's SentencePiece tokenizer has no scores for its pieces (tokenizer.ggml.scores)", "")),
    };
    let prefix = g.get("tokenizer.ggml.add_space_prefix").and_then(Value::as_bool).unwrap_or(true);
    let unk = g.u64("tokenizer.ggml.unknown_token_id").and_then(|i| pieces.get(i as usize)).map(|p| s(p)).unwrap_or(J::Null);
    let vocab: Vec<(String, J)> = pieces.iter().enumerate().map(|(i, p)| (p.to_string(), n(i))).collect();
    // (pairs: a piece may hold a space, which the text form "left right" cannot)
    let merges: Vec<J> = spm_merges(pieces, &scores).into_iter().map(|(l, r)| J::Array(vec![s(&l), s(&r)])).collect();
    let replace = |from: &str, to: &str| obj(vec![("type", s("Replace")), ("pattern", obj(vec![("String", s(from))])), ("content", s(to))]);
    // (a ▁ before the text where the file says so, and none after a special token in it: Metaspace's "first", as Mistral's own tokenizer.json has it; llama.cpp puts one after each. Without the
    // prefix, the spaces alone)
    let (normalizer, pre) = if prefix {
        (J::Null, obj(vec![("type", s("Metaspace")), ("replacement", s("▁")), ("prepend_scheme", s("first")), ("split", J::Bool(false))]))
    } else {
        (replace(" ", "▁"), J::Null)
    };
    let mut decoders = vec![replace("▁", " "), obj(vec![("type", s("ByteFallback"))]), obj(vec![("type", s("Fuse"))])];
    if prefix {
        decoders.push(obj(vec![("type", s("Strip")), ("content", s(" ")), ("start", n(1)), ("stop", n(0))]));
    }
    Ok(obj(vec![
        ("version", s("1.0")),
        ("truncation", J::Null),
        ("padding", J::Null),
        ("added_tokens", J::Array(added(pieces, ty, true))),
        ("normalizer", normalizer),
        ("pre_tokenizer", pre),
        ("post_processor", template(g, pieces, true).unwrap_or(J::Null)),
        ("decoder", obj(vec![("type", s("Sequence")), ("decoders", J::Array(decoders))])),
        (
            "model",
            obj(vec![
                ("type", s("BPE")),
                ("dropout", J::Null),
                ("unk_token", unk),
                ("continuing_subword_prefix", J::Null),
                ("end_of_word_suffix", J::Null),
                ("fuse_unk", J::Bool(true)),
                ("byte_fallback", J::Bool(true)),
                ("ignore_merges", J::Bool(false)),
                ("vocab", J::Object(vocab)),
                ("merges", J::Array(merges)),
            ]),
        ),
    ]))
}
