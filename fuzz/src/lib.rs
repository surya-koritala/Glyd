//! What the fuzz targets share: inputs that stand for several files, and a scratch directory to put them in for the readers that take a path.

use std::path::{Path, PathBuf};

/// The separator of the files in an input that stands for several: a line holding one NUL.
pub const SEP: &[u8] = b"\n\0\n";

/// `data` cut at the separator into at most `n` parts (the rest of the input stays in the last); fewer than `n` are padded with empty ones.
pub fn parts(data: &[u8], n: usize) -> Vec<&[u8]> {
    let mut out = Vec::with_capacity(n);
    let mut rest = data;
    while out.len() + 1 < n {
        match rest.windows(SEP.len()).position(|w| w == SEP) {
            Some(i) => {
                out.push(&rest[..i]);
                rest = &rest[i + SEP.len()..];
            }
            None => break,
        }
    }
    out.push(rest);
    out.resize(n, &[][..]);
    out
}

/// A directory of this process's own under the system's temporary one (TMPDIR: /dev/shm on a CI runner), its files rewritten for each input.
pub struct Scratch(PathBuf);

impl Scratch {
    pub fn new(target: &str) -> Scratch {
        let d = std::env::temp_dir().join(format!("glyd-fuzz-{target}-{}", std::process::id()));
        std::fs::create_dir_all(&d).expect("the scratch directory");
        Scratch(d)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }

    /// `name` holding `bytes`; an empty `bytes` leaves the file out.
    pub fn put(&self, name: &str, bytes: &[u8]) {
        let p = self.0.join(name);
        if bytes.is_empty() {
            let _ = std::fs::remove_file(&p);
        } else {
            std::fs::write(&p, bytes).expect("a scratch file");
        }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A grammar compiler over the 256 byte tokens, a few words and an end token: what the targets that reach llguidance compile against.
pub fn grammars() -> &'static glyd_engine::constrain::Grammars {
    static G: std::sync::OnceLock<glyd_engine::constrain::Grammars> = std::sync::OnceLock::new();
    G.get_or_init(|| {
        let mut bytes: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        for w in ["true", "false", "null", "\"a\"", "{\"", "\":", ", ", "ab", "the "] {
            bytes.push(w.as_bytes().to_vec());
        }
        let mut eos = vec![0xFFu8];
        eos.extend_from_slice(b"<|im_end|>");
        bytes.push(eos);
        let end = bytes.len() as u32 - 1;
        let n = bytes.len();
        glyd_engine::constrain::Grammars::new(bytes, &[end], n).expect("the fuzz vocabulary")
    })
}

/// Lets a panic inside the `tokenizers` crate pass: the engine catches those (a bad tokenizer.json is an error, tok.rs `parse`), and libFuzzer's own panic hook would stop at them first. A panic anywhere else still stops.
pub fn tolerate_tokenizers_panics() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if info.location().is_some_and(|l| l.file().contains("/tokenizers-")) {
                return;
            }
            previous(info);
        }));
    });
}
