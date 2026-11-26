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
    "huge",
    "cb2",
    "copy",
    "no-side",
    "no-stream",
    "precise",
    "public",
    "prof",
    "plain",
    "encode",
    "gpu",
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
    "stale-ok",
    "no-logs",
    "best",
    "last",
    "no-mla",
    "no-speed",
    "peaked",
    "random-tokens",
    "record",
    "rgb",
    "servable",
    "chain",
    "join",
    "expect-apart",
    "tables",
    "decoded",
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
    "stale-ok",
    "no-logs",
    "best",
    "cpu-pin",
    "device-plan",
    "p2p",
    "no-side",
    "no-stream",
];

/// A flag that one command reads the other way than its name is read elsewhere: (the command, the flag, whether it takes a value there).
/// `packcheck --engine` is a switch where `bench --engine NAME` names an engine; `ceiling --fp8 PCT` takes a percent where `glmrandom --fp8` is a switch.
const BY_COMMAND: &[(&str, &str, bool)] = &[
    ("packcheck", "engine", false),
    ("tiercheck", "report", false),
    ("storebench", "plan", false),
    ("calibrate-experts", "synthetic", false),
    ("ceiling", "fp8", true),
    ("quality", "bar", true),
    ("seamcheck", "gpu", true),
    ("dsv4gen", "score", true),
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
                // (the command is the first word: its own reading of the flag where it has one, else the list's)
                let own = BY_COMMAND.iter().find(|(c, k, _)| *k == f && pos.first().map(String::as_str) == Some(*c));
                let v = if own.map_or(!SWITCHES.contains(&f), |x| x.2) { it.next() } else { None };
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

    /// A flag read as a switch takes no value: `bench --bar A B --stale-ok --noise 0.03` gave `--noise` to `--stale-ok` and read no noise.
    #[test]
    fn a_switch_leaves_the_next_word_alone() {
        let a = Args::from(["bench", "--bar", "a.json", "b.json", "--stale-ok", "--noise", "0.03"].map(String::from));
        assert_eq!((a.pos.len(), a.has("stale-ok"), a.flag("noise")), (3, true, Some("0.03")));
        // (the switches of the commands a release build has too: doctor, qwenget, run and serve's expert cache, qwenserve)
        let both = ["stale-ok", "no-logs", "best", "cpu-pin", "device-plan", "p2p", "no-side", "no-stream"];
        for s in both {
            let a = Args::from(["run", "m", &format!("--{s}"), "--context", "4096"].map(String::from));
            assert_eq!((a.pos.len(), a.flag(s), a.flag("context")), (2, Some(""), Some("4096")), "--{s}");
        }
        // (a switch of one command that names a value in another: `packcheck --engine --scratch DIR` beside `bench --engine NAME`)
        for (cmd, s) in [("packcheck", "engine"), ("tiercheck", "report"), ("storebench", "plan"), ("calibrate-experts", "synthetic")] {
            let a = Args::from([cmd, &format!("--{s}"), "--scratch", "d"].map(String::from));
            assert_eq!((a.flag(s), a.flag("scratch")), (Some(""), Some("d")), "{cmd} --{s}");
        }
        let b = Args::from(["bench", "--engine", "llama.cpp", "--report", "out"].map(String::from));
        assert_eq!((b.flag("engine"), b.flag("report")), (Some("llama.cpp"), Some("out")));
    }

    /// The names the sources give `call` (`a.has("`): a development build's, every command's.
    #[cfg(feature = "dev")]
    fn read_by(call: &str) -> std::collections::BTreeSet<String> {
        fn files(d: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            for e in std::fs::read_dir(d).unwrap().flatten() {
                let p = e.path();
                if p.is_dir() {
                    files(&p, out);
                } else if p.extension().is_some_and(|x| x == "rs") && !p.ends_with("args.rs") {
                    out.push(p);
                }
            }
        }
        let mut all = Vec::new();
        files(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"), &mut all);
        let mut names = std::collections::BTreeSet::new();
        for f in all {
            // (a call the formatter broke after its bracket is one call: the white space there is dropped)
            let (mut t, mut open) = (String::new(), false);
            for c in std::fs::read_to_string(&f).unwrap().chars() {
                if !(open && c.is_whitespace()) {
                    open = c == '(';
                    t.push(c);
                }
            }
            for (at, _) in t.match_indices(call) {
                // (`a.has(`, the arguments' own name everywhere, and not `data.has(`)
                let whole = !call.starts_with('a') || !t[..at].ends_with(|c: char| c.is_ascii_alphanumeric() || c == '_');
                let name = t[at + call.len()..].split('"').next().unwrap_or("");
                if whole && !name.is_empty() && name.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-') {
                    names.insert(name.to_string());
                }
            }
        }
        names
    }

    /// Every flag the sources read with `has` alone is a switch of the parser's (fourteen were not: `--stale-ok`, `doctor --no-logs`, `packcheck --record`, ...).
    #[cfg(feature = "dev")]
    #[test]
    fn every_flag_read_with_has_alone_is_a_switch() {
        let valued: std::collections::BTreeSet<String> = [".flag(\"", ".num(\"", ".either(\""].iter().flat_map(|c| read_by(c)).collect();
        let lone: Vec<String> = read_by("a.has(\"").into_iter().filter(|n| !valued.contains(n) && !SWITCHES.contains(&n.as_str())).collect();
        assert!(lone.is_empty(), "read with has() alone and not in SWITCHES: {lone:?}");
    }
    /// Every switch is read by a command, or is passed by a script that the parser must keep it a switch for (the word after it is not its value): a switch nothing reads is
    /// removed (twenty-four were, with nothing in the tree naming them).
    #[cfg(feature = "dev")]
    #[test]
    fn every_switch_is_read_or_passed_by_a_script() {
        // (`tools/runpod/jobs/multigpu_job.sh` passes --p2p and --no-stream; `condor/tools/serve.sh` and `tools/oneshot/run_all.sh` --no-side)
        const PASSED: &[&str] = &["p2p", "no-side", "no-stream"];
        let read: std::collections::BTreeSet<String> = [".has(\"", ".flag(\"", ".num(\"", ".either(\""].iter().flat_map(|c| read_by(c)).collect();
        let dead: Vec<&str> = SWITCHES.iter().copied().filter(|s| !read.contains(*s) && !PASSED.contains(s)).collect();
        assert!(dead.is_empty(), "in SWITCHES and read by no command: {dead:?}");
    }

    /// A switch that one command reads for a value takes it there: `ceiling MODEL --fp8 80` read no percent (`--fp8` is `glmrandom`'s switch), and
    /// `quality --card REF CAND --bar wren` no level (`--bar` is `bench`'s).
    #[test]
    fn a_command_takes_the_value_of_a_flag_that_is_a_switch_elsewhere() {
        let c = Args::from(["ceiling", "m", "--fp8", "80", "--gpus", "8"].map(String::from));
        assert_eq!((c.pos.len(), c.num("fp8", 0u64).unwrap(), c.flag("gpus")), (2, 80, Some("8")));
        // (where it stays a switch: `glmrandom`'s, a development build's command and list)
        if cfg!(feature = "dev") {
            let g = Args::from(["glmrandom", "out", "--fp8", "--real", "--layers", "4"].map(String::from));
            assert_eq!((g.pos.len(), g.flag("fp8"), g.has("real"), g.flag("layers")), (2, Some(""), true, Some("4")));
        }
        let q = Args::from(["quality", "--card", "ref.json", "q4.json", "--level", "Q4_K_M", "--bar", "wren", "--out", "card.json"].map(String::from));
        assert_eq!((q.pos.len(), q.flag("bar"), q.flag("out")), (3, Some("wren"), Some("card.json")));
        let b = Args::from(["bench", "--bar", "a.json", "b.json"].map(String::from));
        assert_eq!((b.pos.len(), b.flag("bar")), (3, Some("")));
        assert_eq!(Args::from(["seamcheck", "--gpu", "1"].map(String::from)).num("gpu", 0i32).unwrap(), 1);
        assert_eq!(Args::from(["dsv4gen", "dir", "--score", "text.txt"].map(String::from)).flag("score"), Some("text.txt"));
    }

    /// Every switch the sources also read for a value names the command that does in `BY_COMMAND`.
    #[cfg(feature = "dev")]
    #[test]
    fn every_switch_read_for_a_value_names_its_command() {
        let valued: std::collections::BTreeSet<String> = [".flag(\"", ".num(\""].iter().flat_map(|c| read_by(c)).collect();
        let lone: Vec<&str> = SWITCHES.iter().copied().filter(|s| valued.contains(*s) && !BY_COMMAND.iter().any(|x| x.1 == *s && x.2)).collect();
        assert!(lone.is_empty(), "in SWITCHES and read for a value by a command BY_COMMAND does not name: {lone:?}");
    }
}
