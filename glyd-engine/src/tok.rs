//! The tokenizer of the model's directory (tokenizer.json) through the `tokenizers` crate; a tiktoken model's (tiktoken.model: Kimi-K3's) made into one.

use crate::ffi::{Error, Result};
use std::path::Path;
use tokenizers::Tokenizer;

pub struct Tok {
    t: Tokenizer,
    /// The end-of-sequence ids of the model's generation_config.json and config.json (empty where the directory has none).
    ends: Vec<u32>,
    /// What constrained decoding needs of the tokenizer, made at the first request that asks (see `grammars`).
    grammars: std::sync::OnceLock<std::result::Result<std::sync::Arc<crate::constrain::Grammars>, String>>,
}

impl Tok {
    pub fn load(dir: &Path) -> Result<Tok> {
        let p = dir.join("tokenizer.json");
        if !p.exists() && dir.join("tiktoken.model").exists() {
            let json = tiktoken_json(dir)?;
            let t = parse(json.as_bytes()).map_err(|e| Error(format!("{}: {e}", dir.join("tiktoken.model").display())))?;
            return Ok(Tok::of(t, dir));
        }
        let bytes = std::fs::read(&p).map_err(|e| Error(format!("{}: {e}", p.display())))?;
        let t = parse(&bytes).map_err(|e| Error(format!("{}: {e}", p.display())))?;
        Ok(Tok::of(t, dir))
    }

    fn of(t: Tokenizer, dir: &Path) -> Tok {
        let mut ends = ends_of(dir);
        if ends.is_empty() {
            // (the tokenizer's own end token, as tokenizer_config.json names it)
            let name = std::fs::read_to_string(dir.join("tokenizer_config.json")).ok().and_then(|t| glyd_gpu::json::parse(&t).ok()).and_then(|c| {
                match c.get("eos_token") {
                    Some(glyd_gpu::json::Value::String(s)) => Some(s.clone()),
                    Some(o) => o.get("content").and_then(glyd_gpu::json::Value::as_str).map(str::to_string),
                    None => None,
                }
            });
            ends.extend(name.and_then(|n| t.token_to_id(&n)));
        }
        Tok { t, ends, grammars: std::sync::OnceLock::new() }
    }

    /// The token grammars are compiled against: the byte strings of the tokenizer's tokens, as wide as the model's rows of logits (`vocab`: the padding after the tokenizer's tokens is never chosen), and the end
    /// tokens. Made once.
    pub fn grammars(&self, vocab: usize) -> std::result::Result<std::sync::Arc<crate::constrain::Grammars>, String> {
        self.grammars
            .get_or_init(|| {
                let bytes = self.token_bytes(vocab)?;
                let t = plain(&self.t);
                let tokenize: crate::constrain::Tokenize = Box::new(move |text| t.encode(text, false).ok().map(|e| e.get_ids().to_vec()));
                crate::constrain::Grammars::with_tokenizer(bytes, &self.ends, vocab, tokenize).map(std::sync::Arc::new)
            })
            .clone()
    }

    /// Each token's bytes (as the text a model that chose it writes has them), the special tokens (the ones marked so, `<|im_end|>`) with the marker toktrie puts before their names: they are not text, and no grammar
    /// but one that names them allows them. Tokens the tokenizer has no name for (the rows are wider than its vocabulary) have the marker and a name of their own.
    fn token_bytes(&self, vocab: usize) -> std::result::Result<Vec<Vec<u8>>, String> {
        use tokenizers::DecoderWrapper;
        fn kinds(d: &DecoderWrapper, byte_level: &mut bool, space: &mut char) {
            match d {
                DecoderWrapper::ByteLevel(_) => *byte_level = true,
                DecoderWrapper::Metaspace(m) => {
                    *space = m.get_replacement();
                }
                DecoderWrapper::Replace(r) if r.content == " " => {
                    if let Some(c) = serde_json::to_value(r).ok().and_then(|v| v["pattern"]["String"].as_str().and_then(|p| p.chars().next())) {
                        *space = c;
                    }
                }
                DecoderWrapper::Sequence(l) => l.get_decoders().iter().for_each(|d| kinds(d, byte_level, space)),
                _ => {}
            }
        }
        let (mut byte_level, mut space) = (false, '\u{2581}');
        if let Some(d) = self.t.get_decoder() {
            kinds(d, &mut byte_level, &mut space);
        }
        let added = self.t.get_added_tokens_decoder();
        // (a tokenizer whose decoder says nothing of bytes: by the letters its tokens are written in, GPT-2's `\u{0120}` for a space or SentencePiece's `\u{2581}`)
        if !byte_level && !matches!(self.t.get_decoder(), Some(DecoderWrapper::Sequence(_) | DecoderWrapper::Metaspace(_) | DecoderWrapper::ByteFallback(_))) {
            byte_level = ["\u{0120}", "\u{010a}"].iter().any(|c| self.t.token_to_id(c).is_some());
        }
        let mut map = std::collections::HashMap::new();
        let mut k = 0x100u32;
        for b in 0..=255u8 {
            let c = b as char;
            if matches!(c, '!'..='~' | '\u{A1}'..='\u{AC}' | '\u{AE}'..='\u{FF}') {
                map.insert(c, b);
            } else {
                map.insert(char::from_u32(k).unwrap_or('\u{FFFD}'), b);
                k += 1;
            }
        }
        let n = self.t.get_vocab_size(true).max(added.keys().map(|&i| i as usize + 1).max().unwrap_or(0));
        let mut out: Vec<Vec<u8>> = Vec::with_capacity(vocab.max(n));
        for id in 0..vocab.max(n) as u32 {
            let special = |name: &str| {
                let mut b = name.as_bytes().to_vec();
                b.insert(0, toktrie::TokTrie::SPECIAL_TOKEN_MARKER);
                b
            };
            let a = added.get(&id);
            let name = self.t.id_to_token(id);
            let bytes = match (a, name) {
                (Some(a), _) if a.special => special(&a.content),
                (Some(a), _) => a.content.as_bytes().to_vec(),
                (None, None) => special(&format!(".<[{id}]>")),
                (None, Some(name)) if byte_level => {
                    let b: Option<Vec<u8>> = name.chars().map(|c| map.get(&c).copied()).collect();
                    match b {
                        Some(b) => b,
                        None => special(&name),
                    }
                }
                (None, Some(name)) => {
                    let hex = name.strip_prefix("<0x").and_then(|h| h.strip_suffix('>')).filter(|h| h.len() == 2).and_then(|h| u8::from_str_radix(h, 16).ok());
                    match hex {
                        Some(b) => vec![b],
                        None => name.replace(space, " ").into_bytes(),
                    }
                }
            };
            out.push(bytes);
        }
        out.truncate(vocab.max(n));
        Ok(out)
    }

    /// The tokens the tokenizer names, the added ones with them.
    pub fn vocab_size(&self) -> usize {
        self.t.get_vocab_size(true)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_with(text, true)
    }

    /// `text`'s ids, with the tokenizer's special tokens added (a chat template's text takes none: its own are in it).
    pub fn encode_with(&self, text: &str, special: bool) -> Result<Vec<u32>> {
        let e = self.t.encode(text, special).map_err(|e| Error(e.to_string()))?;
        Ok(e.get_ids().to_vec())
    }

    /// The ids of the tokens added to the vocabulary (the special ones among them).
    pub fn added_ids(&self) -> std::collections::HashSet<u32> {
        self.t.get_added_tokens_decoder().keys().copied().collect()
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.decode_with(ids, false)
    }

    /// `ids` as text, with or without the special tokens (`<|im_end|>` ...) in it.
    pub fn decode_with(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        self.t.decode(ids, skip_special).map_err(|e| Error(e.to_string()))
    }
}

/// A tokenizer.json read: its error in words, where the `tokenizers` crate gives one, and where it panics instead (its decoders' deserializer `expect`s the JSON to be well formed, so a syntax error inside a
/// `decoder` is a panic) the same: a repository's file must not stop the process that loads it.
fn parse(bytes: &[u8]) -> std::result::Result<Tokenizer, String> {
    match std::panic::catch_unwind(|| Tokenizer::from_bytes(bytes)) {
        Ok(r) => r.map_err(|e| e.to_string()),
        Err(p) => Err(format!(
            "not a tokenizer.json ({})",
            p.downcast_ref::<String>().map(String::as_str).or_else(|| p.downcast_ref::<&str>().copied()).unwrap_or("the reader panicked")
        )),
    }
}

/// The tokenizer as it tokenizes a piece of text that is not the start of one: no space put before it (SentencePiece's `Prepend` normalizer and `Metaspace` prefix), as toktrie_hf_tokenizers has it.
fn plain(t: &Tokenizer) -> Tokenizer {
    use tokenizers::{normalizers, pre_tokenizers, NormalizerWrapper, PreTokenizerWrapper};
    fn no_prepend(n: NormalizerWrapper) -> Option<NormalizerWrapper> {
        match n {
            NormalizerWrapper::Prepend(_) => None,
            NormalizerWrapper::Sequence(l) => {
                let kept: Vec<_> = l.as_ref().iter().filter_map(|n| no_prepend(n.clone())).collect();
                (!kept.is_empty()).then(|| NormalizerWrapper::Sequence(normalizers::Sequence::new(kept)))
            }
            n => Some(n),
        }
    }
    fn no_prefix(p: PreTokenizerWrapper) -> PreTokenizerWrapper {
        match p {
            PreTokenizerWrapper::Metaspace(mut m) => {
                m.prepend_scheme = pre_tokenizers::metaspace::PrependScheme::Never;
                PreTokenizerWrapper::Metaspace(m)
            }
            PreTokenizerWrapper::Sequence(l) => {
                PreTokenizerWrapper::Sequence(pre_tokenizers::sequence::Sequence::new(l.as_ref().iter().map(|p| no_prefix(p.clone())).collect()))
            }
            p => p,
        }
    }
    let mut t = t.clone();
    let n = t.get_normalizer().cloned();
    if let Some(n) = n {
        t.with_normalizer(no_prepend(n));
    }
    let p = t.get_pre_tokenizer().cloned();
    if let Some(p) = p {
        t.with_pre_tokenizer(Some(no_prefix(p)));
    }
    t
}

/// The end-of-sequence ids of the model's directory: generation_config.json's where it gives any, else config.json's (or its text_config's).
fn ends_of(dir: &Path) -> Vec<u32> {
    let Some(config) = crate::runtime::config::ModelConfig::load(dir).ok() else {
        return Vec::new();
    };
    let generation = std::fs::read_to_string(dir.join("generation_config.json")).ok();
    // (the config's own end tokens, a number or a list; or its text model's)
    let mut ends = crate::weights::eos_ids(&config.top, generation.as_deref());
    if ends.is_empty() {
        ends = crate::weights::eos_ids(&config.text, None);
    }
    ends.into_iter().filter_map(|e| u32::try_from(e).ok()).collect()
}

/// Kimi-K3's split of a text before its byte pairs are merged (tokenization_kimi.py's `pat_str`: runs of Han; words, their case and a contraction; up to three digits;
/// punctuation with its new lines; whitespace).
const KIMI_PATTERN: &str = concat!(
    r"[\p{Han}]+|",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|",
    r"\p{N}{1,3}|",
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|",
    r"\s*[\r\n]+|",
    r"\s+(?!\S)|",
    r"\s+",
);

/// The 256 bytes as the byte-level pre-tokenizer writes them (each printable byte itself, the others the code points from 256 on, in order).
fn byte_chars() -> [char; 256] {
    let mut out = ['\0'; 256];
    let mut extra = 0u32;
    for b in 0..256u32 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        out[b as usize] = if printable {
            char::from_u32(b).unwrap_or('?')
        } else {
            extra += 1;
            char::from_u32(255 + extra).unwrap_or('?')
        };
    }
    out
}

fn base64(s: &str) -> Option<Vec<u8>> {
    let val = |c: u8| -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        } as u32)
    };
    let (mut out, mut acc, mut bits) = (Vec::new(), 0u32, 0);
    for &c in s.trim_end_matches('=').as_bytes() {
        acc = (acc << 6) | val(c)?;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Some(out)
}

/// A tiktoken model (tiktoken.model: each line a token's bytes in base64 and its rank) as a tokenizers definition: a byte-level BPE whose vocabulary is the ranks and
/// whose merges are, for each token, its splits into two tokens that make it, in the order of the tokens' ranks (a word that is a token taken whole, as tiktoken does);
/// the split before it Kimi-K3's; the special tokens tokenizer_config.json's `added_tokens_decoder`, the 256 ids after the ranks the rest reserved.
fn tiktoken_json(dir: &Path) -> Result<String> {
    let p = dir.join("tiktoken.model");
    let text = std::fs::read_to_string(&p).map_err(|e| Error(format!("{}: {e}", p.display())))?;
    let mut ranks: Vec<Vec<u8>> = Vec::new();
    for (n, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let mut it = line.split_whitespace();
        let (Some(b), Some(r)) = (it.next().and_then(base64), it.next().and_then(|r| r.parse::<usize>().ok())) else {
            return Err(Error(format!("{} line {}: not a token and its rank", p.display(), n + 1)));
        };
        if r != ranks.len() {
            return Err(Error(format!("{} line {}: rank {r}, where {} comes next", p.display(), n + 1, ranks.len())));
        }
        ranks.push(b);
    }
    let bc = byte_chars();
    let word = |b: &[u8]| -> String { b.iter().map(|&x| bc[x as usize]).collect() };
    let rank: std::collections::HashMap<&[u8], usize> = ranks.iter().enumerate().map(|(i, b)| (b.as_slice(), i)).collect();
    let mut merges: Vec<(usize, usize, usize, usize)> = Vec::new();
    for (r, b) in ranks.iter().enumerate().filter(|(_, b)| b.len() > 1) {
        for i in 1..b.len() {
            if let (Some(&a), Some(&c)) = (rank.get(&b[..i]), rank.get(&b[i..])) {
                merges.push((r, a, c, i));
            }
        }
    }
    merges.sort_unstable();
    let vocab: Vec<String> = ranks.iter().enumerate().map(|(i, b)| format!("{}:{i}", crate::server::jstr(&word(b)))).collect();
    let merges: Vec<String> = merges.iter().map(|&(r, _, _, i)| crate::server::jstr(&format!("{} {}", word(&ranks[r][..i]), word(&ranks[r][i..])))).collect();
    // (each one's name and whether it is special as the config has them: Kimi-K3's <|open|>, <|close|> and <|sep|> are not, so they stay in a reply's text)
    let named: std::collections::HashMap<usize, (String, bool)> = std::fs::read_to_string(dir.join("tokenizer_config.json"))
        .ok()
        .and_then(|t| glyd_gpu::json::parse(&t).ok())
        .and_then(|v| {
            v.get("added_tokens_decoder").and_then(|d| {
                d.as_object().map(|o| {
                    o.iter()
                        .filter_map(|(k, x)| {
                            Some((k.parse().ok()?, (x.get("content")?.as_str()?.to_string(), x.get("special").and_then(|s| s.as_bool()).unwrap_or(true))))
                        })
                        .collect()
                })
            })
        })
        .unwrap_or_default();
    let added: Vec<String> = (ranks.len()..ranks.len() + 256)
        .map(|i| {
            let (name, special) = named.get(&i).cloned().unwrap_or_else(|| (format!("<|reserved_token_{i}|>"), true));
            format!(
                r#"{{"id":{i},"content":{},"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":{special}}}"#,
                crate::server::jstr(&name)
            )
        })
        .collect();
    Ok(format!(
        r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{}],"normalizer":null,"pre_tokenizer":{{"type":"Sequence","pretokenizers":[{{"type":"Split","pattern":{{"Regex":{}}},"behavior":"Isolated","invert":false}},{{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}}]}},"post_processor":null,"decoder":{{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}},"model":{{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{{{}}},"merges":[{}]}}}}"#,
        added.join(","),
        crate::server::jstr(KIMI_PATTERN),
        vocab.join(","),
        merges.join(",")
    ))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A tokenizer of a few words with Llama 3's way of making a BOS: `<begin>` an added special token that the post-processor puts first.
    const BOS_TOKENIZER: &str = r#"{"version":"1.0","truncation":null,"padding":null,
        "added_tokens":[{"id":0,"content":"<begin>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":3,"content":"<unk>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
        "normalizer":null,"pre_tokenizer":{"type":"Whitespace"},
        "post_processor":{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"<begin>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}}],"special_tokens":{"<begin>":{"id":"<begin>","ids":[0],"tokens":["<begin>"]}}},
        "decoder":null,"model":{"type":"WordLevel","vocab":{"<begin>":0,"hello":1,"world":2,"<unk>":3},"unk_token":"<unk>"}}"#;

    /// A syntax error inside a tokenizer.json's `decoder` was a panic of the `tokenizers` crate (it `expect`s), and a bad file is an error.
    #[test]
    fn a_tokenizer_json_with_a_broken_decoder_is_an_error() {
        let broken = BOS_TOKENIZER.replace("\"decoder\":null", "\"decoder\":{\"type\" \"Fuse\"}");
        assert_ne!(broken, BOS_TOKENIZER);
        let Err(e) = parse(broken.as_bytes()) else { panic!("an error") };
        assert!(e.contains("tokenizer.json") || e.contains("expected"), "{e}");
        assert!(parse(BOS_TOKENIZER.as_bytes()).is_ok());
        assert!(parse(b"").is_err());
    }

    fn tok() -> Tok {
        Tok::of(Tokenizer::from_bytes(BOS_TOKENIZER.as_bytes()).unwrap(), Path::new("."))
    }

    /// A tokenizer for the tests of what is made of a model's tokens: `words` are the ids 1, 2, ... (0 is `<unk>`; a word of the form `<|name|>` is a special token), a text is read by its whitespace, and ids decode to
    /// their words fused as they are (no space between).
    pub(crate) fn toy(words: &[&str]) -> Tok {
        let mut vocab = String::from("\"<unk>\":0");
        let mut added = Vec::new();
        for (i, w) in words.iter().enumerate() {
            vocab.push_str(&format!(",{}:{}", crate::server::jstr(w), i + 1));
            if w.starts_with("<|") && w.ends_with("|>") {
                added.push(format!(
                    "{{\"id\":{},\"content\":{},\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":true}}",
                    i + 1,
                    crate::server::jstr(w)
                ));
            }
        }
        let added = added.join(",");
        let json = format!(
            r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{added}],"normalizer":null,"pre_tokenizer":{{"type":"Whitespace"}},"post_processor":null,"decoder":{{"type":"Fuse"}},"model":{{"type":"WordLevel","vocab":{{{vocab}}},"unk_token":"<unk>"}}}}"#
        );
        Tok::of(Tokenizer::from_bytes(json.as_bytes()).unwrap(), Path::new("."))
    }

    #[test]
    fn the_toy_tokenizer_fuses_its_words() {
        let t = toy(&["Hello", " there", "<tool_call>", "é"]);
        assert_eq!(t.decode(&[1, 2, 4, 3]).unwrap(), "Hello thereé<tool_call>");
        assert_eq!(t.decode(&[2]).unwrap(), " there");
        assert_eq!(t.encode_with("Hello nothing", false).unwrap(), vec![1, 0]);
        // a special token is in the text unless it is skipped
        let t = toy(&["Hi", "<|im_end|>", "there"]);
        assert_eq!((t.decode(&[1, 2, 3]).unwrap().as_str(), t.decode_with(&[1, 2, 3], true).unwrap().as_str()), ("Hi<|im_end|>there", "Hithere"));
    }

    fn b64(b: &[u8]) -> String {
        const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for c in b.chunks(3) {
            let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
            for k in 0..4 {
                out.push(if k <= c.len() { A[(n >> (18 - 6 * k) & 63) as usize] as char } else { '=' });
            }
        }
        out
    }

    /// tiktoken's encoding of one piece, as its own algorithm makes it: the piece whole where it is a token, else its bytes merged, the pair of the lowest rank first.
    fn rank_merge(piece: &[u8], rank: &std::collections::HashMap<Vec<u8>, u32>) -> Vec<u32> {
        if let Some(&r) = rank.get(piece) {
            return vec![r];
        }
        let mut parts: Vec<Vec<u8>> = piece.iter().map(|&b| vec![b]).collect();
        loop {
            let best = (0..parts.len().saturating_sub(1)).filter_map(|i| rank.get(&[parts[i].clone(), parts[i + 1].clone()].concat()).map(|&r| (r, i))).min();
            let Some((_, i)) = best else { break };
            let joined = [parts[i].clone(), parts[i + 1].clone()].concat();
            parts.splice(i..i + 2, [joined]);
        }
        parts.iter().map(|p| rank[p]).collect()
    }

    /// A tiny tiktoken model: its ranks the vocabulary, a word that is a token taken whole, the others merged by rank; the special tokens after the ranks.
    #[test]
    fn a_tiktoken_model_is_its_ranks() {
        let dir = std::env::temp_dir().join(format!("glyd-tiktoken-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let toks: [&[u8]; 7] = [b"a", b"b", b"c", b" ", b"ab", b" a", b"abc"];
        let lines: Vec<String> = toks.iter().enumerate().map(|(i, t)| format!("{} {i}", b64(t))).collect();
        std::fs::write(dir.join("tiktoken.model"), lines.join("\n")).unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"added_tokens_decoder": {"7": {"content": "<|open|>", "special": false}, "8": {"content": "<|end|>", "special": true}}}"#,
        )
        .unwrap();
        let t = Tok::load(&dir).unwrap();
        // ("abc" whole; " ab": ab (rank 4) before " a" (5), then no " ab")
        assert_eq!(t.encode_with("abc ab", false).unwrap(), vec![6, 3, 4]);
        assert_eq!(t.encode_with("<|open|>a", false).unwrap(), vec![7, 0]);
        assert_eq!(t.decode(&[6, 3, 4]).unwrap(), "abc ab");
        assert_eq!(t.encode_with("<|reserved_token_9|>", false).unwrap(), vec![9]);
        // (a token that is not special stays in the text a reply is decoded to; a special one is skipped where asked)
        assert_eq!(t.decode_with(&[7, 0, 8], true).unwrap(), "<|open|>a");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The real Kimi-K3 tokenizer (KIMI_DIR: a directory with its tiktoken.model and tokenizer_config.json) against tiktoken's own algorithm on its pieces, for texts of
    /// Latin, Chinese, digits, punctuation, emoji and whitespace runs.
    #[test]
    #[ignore]
    fn kimi_tokens_are_tiktokens() {
        let dir = std::path::PathBuf::from(std::env::var("KIMI_DIR").expect("KIMI_DIR"));
        let t = Tok::load(&dir).unwrap();
        let text = std::fs::read_to_string(dir.join("tiktoken.model")).unwrap();
        let rank: std::collections::HashMap<Vec<u8>, u32> = text
            .lines()
            .filter_map(|l| {
                let mut i = l.split_whitespace();
                Some((super::base64(i.next()?)?, i.next()?.parse().ok()?))
            })
            .collect();
        let samples = [
            "Hello, world! It's 2026; we're testing 1234567 tokens.",
            "北京是中国的首都。东京也是。",
            "  spaces\n\n\ttabs\r\n  end ",
            "emoji 😀👍🏽 and café naïve",
            "fn main() { let x = vec![1, 2, 3]; }",
            "MixedCASE wordsWithCaps HTTPServer",
        ];
        for s in samples {
            let got = t.encode_with(s, false).unwrap();
            let pre = t.t.get_pre_tokenizer().unwrap();
            let mut ps = tokenizers::PreTokenizedString::from(s);
            tokenizers::PreTokenizer::pre_tokenize(pre, &mut ps).unwrap();
            let bc = super::byte_chars();
            let want: Vec<u32> = ps
                .get_splits(tokenizers::OffsetReferential::Original, tokenizers::OffsetType::Byte)
                .iter()
                .flat_map(|(piece, _, _)| {
                    let bytes: Vec<u8> = piece.chars().map(|c| bc.iter().position(|&x| x == c).unwrap() as u8).collect();
                    rank_merge(&bytes, &rank)
                })
                .collect();
            assert_eq!(got, want, "{s:?}");
            assert_eq!(t.decode(&got).unwrap(), s);
        }
    }

    /// The engine's two ways of making ids from text: a plain prompt (`encode`: the tokenizer's special tokens added, so a Llama prompt begins with its BOS) and a chat template's text
    /// (`encode_with(.., false)`: nothing added, the template's own `<begin>` read as the one special token it is, so the BOS is there once).
    #[test]
    fn the_bos_is_there_once() {
        let t = tok();
        assert_eq!(t.encode("hello world").unwrap(), vec![0, 1, 2], "a plain prompt gets the BOS from the tokenizer");
        assert_eq!(t.encode_with("hello world", false).unwrap(), vec![1, 2], "a chat template's text gets none from the tokenizer");
        assert_eq!(t.encode_with("<begin>hello world", false).unwrap(), vec![0, 1, 2], "the template's own BOS is one id");
        assert_eq!(
            t.encode_with("<begin>hello world", true).unwrap(),
            vec![0, 0, 1, 2],
            "(what the engine did with a template that has a bos_token: two BOS, had it passed the token)"
        );
    }

    /// The bytes of each token, as the families' tokenizers write them: SentencePiece's (`\u{2581}` for a space, `<0xNN>` for a byte, Gemma's), byte-level BPE's (GPT-2's letters for bytes, Qwen's and DeepSeek's), and a
    /// token that is added and not special (`<tool_call>`) as the text it is; a special one is marked, and one the tokenizer has no name for is.
    #[test]
    fn the_bytes_of_the_tokens_of_each_kind_of_tokenizer() {
        let marker = toktrie::TokTrie::SPECIAL_TOKEN_MARKER;
        let spm = r#"{"version":"1.0","truncation":null,"padding":null,
            "added_tokens":[{"id":0,"content":"<pad>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":5,"content":"<tool_call>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false}],
            "normalizer":null,"pre_tokenizer":null,"post_processor":null,
            "decoder":{"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"\u2581"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"}]},
            "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":true,
              "vocab":{"<pad>":0,"\u2581hello":1,"<0x0A>":2,"\u2581":3,"\u00e9":4,"<tool_call>":5},"merges":[]}}"#;
        let t = Tok::of(Tokenizer::from_bytes(spm.as_bytes()).unwrap(), Path::new("."));
        let b = t.token_bytes(8).unwrap();
        assert_eq!(b[1], b" hello");
        assert_eq!(b[2], vec![10]);
        assert_eq!(b[3], b" ");
        assert_eq!(b[4], "é".as_bytes());
        assert_eq!(b[5], b"<tool_call>");
        assert_eq!(b[0], [&[marker][..], b"<pad>"].concat());
        // (the rows are wider than the tokenizer: the extra tokens are named, special)
        assert_eq!(b[6][0], marker);
        assert_eq!(b.len(), 8);
        // byte-level BPE: the letter for a space is \u0120, for a new line \u010a, and two bytes of a letter are two letters
        let bl = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
            "post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
            "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,
              "vocab":{"a":0,"\u0120a":1,"\u010a":2,"\u00c3\u00a9":3},"merges":[]}}"#;
        let t = Tok::of(Tokenizer::from_bytes(bl.as_bytes()).unwrap(), Path::new("."));
        let b = t.token_bytes(4).unwrap();
        assert_eq!((b[0].as_slice(), b[1].as_slice(), b[2].as_slice(), b[3].as_slice()), (&b"a"[..], &b" a"[..], &b"\n"[..], "é".as_bytes()));
    }
}
