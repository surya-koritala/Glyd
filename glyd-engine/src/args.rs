//! The command line: positional words and `--flag value` pairs (a flag with no value where it is one of the switches below).

use crate::ffi::{Error, Result};
use crate::weights::Layout;

/// Flags that take no value.
#[cfg(feature = "dev")]
const SWITCHES: &[&str] = &[
    "predict-only",
    "score",
    "no-graph",
    "no-draft",
    "compare",
    "quality",
    "bar",
    "cpu-ref",
    "sequence",
    "list-cases",
    "p2p",
    "ignore-eos",
    "logprobs",
    "profile",
    "profile-prefill",
    "attn-decode",
    "no-think",
    "help",
    "quiet",
    "chat",
    "version",
    "lossy",
    "speed",
    "no-check",
    "cpu-pin",
    "hb",
    "glm",
    "pin",
    "compare",
    "twice",
    "renorm",
    "huge",
    "cb2",
    "copy",
    "async-copies",
    "segmented",
    "skinny",
    "no-pin",
    "mtp",
    "spec",
    "no-read-ahead",
    "inventory",
    "stream",
    "no-side",
    "no-mtp",
    "populate",
    "no-segmented",
    "no-skinny",
    "no-async-copies",
    "no-stream",
    "precise",
    "public",
    "dynamic",
    "prof",
    "plain",
    "encode",
    "gpu",
    "old",
    "fp8",
    "real",
    "predict",
    "load",
    "no-gpu",
    "costs",
    "workers",
    "by-layers",
    "one",
    "decode",
    "json",
    "background",
    "all",
    "insecure-no-auth",
    "yes",
    "kv12",
    "card",
    "table",
    "device-plan",
];
#[cfg(not(feature = "dev"))]
const SWITCHES: &[&str] = &[
    "no-think",
    "predict-only",
    "score",
    "help",
    "quiet",
    "chat",
    "version",
    "lossy",
    "precise",
    "predict",
    "load",
    "no-gpu",
    "costs",
    "json",
    "background",
    "all",
    "compare",
    "bar",
    "quality",
    "no-draft",
    "insecure-no-auth",
    "yes",
    "card",
    "table",
];

#[derive(Clone)]
pub struct Args {
    pub pos: Vec<String>,
    pub flags: Vec<(String, Option<String>)>,
}

impl Args {
    pub fn parse() -> Args {
        Args::from(std::env::args().skip(1))
    }

    pub fn from(args: impl IntoIterator<Item = String>) -> Args {
        let (mut pos, mut flags) = (Vec::new(), Vec::new());
        let mut it = args.into_iter();
        while let Some(a) = it.next() {
            if a == "-p" {
                // (glyd run's short form of --prompt)
                flags.push(("prompt".to_string(), it.next()));
            } else if let Some(f) = a.strip_prefix("--") {
                // `--flag=value` as well
                if let Some((k, v)) = f.split_once('=') {
                    flags.push((k.to_string(), Some(v.to_string())));
                    continue;
                }
                let v = if SWITCHES.contains(&f) { None } else { it.next() };
                flags.push((f.to_string(), v));
            } else {
                pos.push(a);
            }
        }
        Args { pos, flags }
    }

    pub fn flag(&self, name: &str) -> Option<&str> {
        self.flags.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_deref().unwrap_or(""))
    }

    /// These arguments with `--NAMES[0] VALUE` in place of any of the names given (a flag's other names too).
    pub fn with(&self, names: &[&str], value: String) -> Args {
        let mut a = self.clone();
        a.flags.retain(|(k, _)| !names.contains(&k.as_str()));
        a.flags.push((names[0].to_string(), Some(value)));
        a
    }

    pub fn has(&self, name: &str) -> bool {
        self.flags.iter().any(|(k, _)| k == name)
    }

    pub fn num<T: std::str::FromStr>(&self, name: &str, default: T) -> Result<T> {
        match self.flag(name) {
            None => Ok(default),
            Some(v) => v.parse().map_err(|_| Error(format!("--{name} {v}: not a number"))),
        }
    }

    /// The first of two names for one flag (`--context`, `--ctx`).
    pub fn either<T: std::str::FromStr>(&self, a: &str, b: &str, default: T) -> Result<T> {
        if self.has(a) {
            self.num(a, default)
        } else {
            self.num(b, default)
        }
    }

    pub fn need(&self, i: usize, what: &str) -> Result<&str> {
        self.pos.get(i).map(String::as_str).ok_or_else(|| Error(format!("missing {what} (glyd-engine --help)")))
    }

    pub fn layout(&self) -> Result<Option<Layout>> {
        match self.flag("layout") {
            None | Some("auto") => Ok(None),
            Some("mma") => Ok(Some(Layout::Tiered)),
            Some("mma12") => Ok(Some(Layout::Twelve)),
            Some("a") => Ok(Some(Layout::Compact)),
            Some("i6") => Ok(Some(Layout::Int6)),
            Some("i5") => Ok(Some(Layout::Int5)),
            Some("i4") => Ok(Some(Layout::Int4)),
            Some("i4x") => Ok(Some(Layout::Int4H)),
            Some("i3") => Ok(Some(Layout::Int3)),
            Some(other) => Err(Error(format!("--layout {other}: auto, mma, mma12, a, i6, i5, i4, i4x or i3"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_with_values_and_switches() {
        let a = Args::from(["run", "Qwen/Qwen3-8B", "--prompt", "Say hi", "--no-think", "--context=4096", "--port", "8001"].map(String::from));
        assert_eq!(a.pos, ["run", "Qwen/Qwen3-8B"]);
        assert_eq!((a.flag("prompt"), a.has("no-think"), a.flag("no-think")), (Some("Say hi"), true, Some("")));
        assert_eq!(a.either("context", "ctx", 8192usize).unwrap(), 4096);
        assert_eq!(a.num("port", 8000u16).unwrap(), 8001);
        assert_eq!(Args::from(["x", "--ctx", "2048"].map(String::from)).either("context", "ctx", 8192usize).unwrap(), 2048);
        // (a switch takes no value: `--yes` before `--prompt` took the prompt flag as its value, and the run read none)
        let y = Args::from(["run", "m:penguin", "--yes", "--prompt", "Hello", "--max-tokens", "4"].map(String::from));
        assert_eq!((y.pos.len(), y.has("yes"), y.flag("prompt"), y.flag("max-tokens")), (2, true, Some("Hello"), Some("4")));
        assert!(Args::from(["x", "--layout", "tiny"].map(String::from)).layout().is_err());
        // (the layouts by name: the block ones are the lossy levels')
        for (name, layout) in [
            ("mma", Layout::Tiered),
            ("mma12", Layout::Twelve),
            ("a", Layout::Compact),
            ("i6", Layout::Int6),
            ("i5", Layout::Int5),
            ("i4", Layout::Int4),
            ("i3", Layout::Int3),
        ] {
            assert_eq!(Args::from(["x", "--layout", name].map(String::from)).layout().unwrap(), Some(layout), "{name}");
            assert_eq!(layout.name(), if layout == Layout::Compact { "mma" } else { name }, "{name}");
        }
        assert!(Args::from(["x", "--port", "abc"].map(String::from)).num("port", 1u16).is_err());
    }
}
