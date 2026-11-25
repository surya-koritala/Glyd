//! The calibration corpus: every kind of text the KV cache's tables are made over (the 12-bit pages' tables, src/dynamic/kv12.rs) and Hummingbird's use counts
//! are taken on (src/hb_usage.rs), as token ids by document. The tables of the earlier dense engine, which this module made on that engine, went with it (R3c, #139).

use crate::chat::{self, Chat};
use crate::ffi::Result;
use crate::kvcal;
use crate::tok::Tok;

/// The text the KV tables are calibrated on (data/kvcorpus.bin: the text zlib-compressed, the file the Python package ships too), inflated once.
pub fn corpus() -> &'static str {
    static TEXT: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    TEXT.get_or_init(|| {
        use std::io::Read;
        let mut s = String::new();
        flate2::read::ZlibDecoder::new(&include_bytes!("../data/kvcorpus.bin")[..]).read_to_string(&mut s).expect("the corpus is inflated");
        s
    })
}

/// The corpus as token ids by document: (kind, name, ids). A chat section goes through the chat template (a model without one: its turns as plain lines).
pub fn corpus_docs(tok: &Tok, chat: Option<&Chat>) -> Result<Vec<(String, String, Vec<u32>)>> {
    let mut out = Vec::new();
    for (kind, name, body) in kvcal::parse_corpus(corpus()) {
        let ids = if kind == "chat" {
            let msgs = chat::parse_messages(&body)?;
            match chat {
                Some(c) => tok.encode_with(&c.render(&msgs, false, None)?, false)?,
                None => tok.encode(&msgs.iter().map(|m| format!("{}: {}", m.role, m.content)).collect::<Vec<_>>().join("\n"))?,
            }
        } else {
            tok.encode(&body)?
        };
        out.push((kind, name, ids));
    }
    Ok(out)
}
