//! What this build of the engine runs (ROADMAP P16): one table of model families, the levels each runs at, where its
//! weights may go and the context it takes, read by `fit` (src/plan.rs) and by `run`/`serve` (src/cli.rs `tagged`) and
//! their refusals, so `fit` never plans a run that `run` refuses. The machine's side (vendor, compute capability, driver,
//! CPU architecture) is `machine`, on the same constants `machine::gpu_refusal` uses.

use crate::machine::{driver_min, driver_need, ARCHS, MAX_CONTEXT};
use crate::plan::{Machine, Vendor};
use crate::ui::Refusal;

/// A family of models the engine tells apart by its config.json.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// Qwen3 dense (Qwen3ForCausalLM): lossless, on the Qwen3 path.
    Qwen3,
    /// Qwen3.5, 3.6 and 3.8 dense (Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration): the lossy levels.
    Qwen35,
    NemotronH,
    /// Gemma 4 dense (12B, 31B); the 26B-A4B mixture of experts and the E models are refused.
    Gemma4,
    Kimi,
    Glm,
    DeepSeekV4,
    Muse,
    /// Qwen3.8-Flash-Next: Condor's model, run by `qwenserve` until `glyd run` takes it (ROADMAP P8).
    FlashNext,
}

/// One family's row.
#[derive(Debug)]
pub struct Support {
    pub family: Family,
    pub name: &'static str,
    /// The levels `run` takes this family at, in every build.
    pub levels: &'static [&'static str],
    /// The hummingbird level comes as a pre-packed download or directory only (its experts' codes are made apart).
    pub hummingbird_packed_only: bool,
    /// The context `run` takes where none is asked: Some(n), n tokens (less where the model's own limit is lower);
    /// None: the most that fits, to `LOSSY_CTX`.
    pub default_ctx: Option<usize>,
    /// The most context this path runs, besides the model's own limit.
    pub max_ctx: Option<usize>,
    /// The levels at which the weights may be split over the GPUs of one machine.
    pub split_levels: &'static [&'static str],
    /// The levels at which a mixture of experts may keep its experts in the computer's memory.
    pub experts_in_ram_levels: &'static [&'static str],
}

/// The most context the lossy paths take where none is asked (one constant: the planner's, src/plan.rs).
pub use crate::plan::LOSSY_CTX;
/// The context the Qwen3 path and the dynamic engine take where none is asked (src/fit.rs `CTX_WANT`, src/cli.rs `size_dynamic`).
pub const DEFAULT_CTX: usize = 8192;

const LOSSY: &[&str] = &["kestrel", "swift", "wren"];
const NONE: &[&str] = &[];
const PENGUIN: &[&str] = &["penguin"];
/// Lossless, or the routed experts at hummingbird's bits (a directory `glyd-engine pack --layout hummingbird --usage` makes: src/hb_pack.rs).
const PENGUIN_HB: &[&str] = &["penguin", "hummingbird"];

pub static TABLE: [Support; 9] = [
    Support {
        family: Family::Qwen3,
        name: "Qwen3",
        levels: PENGUIN,
        hummingbird_packed_only: false,
        default_ctx: Some(DEFAULT_CTX),
        max_ctx: Some(MAX_CONTEXT),
        split_levels: NONE,
        experts_in_ram_levels: NONE,
    },
    Support {
        family: Family::Qwen35,
        name: "Qwen3.5",
        levels: LOSSY,
        hummingbird_packed_only: false,
        default_ctx: None,
        max_ctx: None,
        split_levels: NONE,
        experts_in_ram_levels: NONE,
    },
    Support {
        family: Family::NemotronH,
        name: "Nemotron-H",
        levels: &["penguin", "kestrel", "swift", "wren", "hummingbird"],
        hummingbird_packed_only: true,
        default_ctx: None,
        max_ctx: None,
        split_levels: PENGUIN,
        experts_in_ram_levels: &["kestrel", "swift", "wren", "hummingbird"],
    },
    Support {
        family: Family::Gemma4,
        name: "Gemma 4",
        levels: &["penguin", "kestrel", "swift", "wren"],
        hummingbird_packed_only: false,
        default_ctx: None,
        max_ctx: None,
        split_levels: NONE,
        experts_in_ram_levels: NONE,
    },
    Support {
        family: Family::Kimi,
        name: "Kimi-K3",
        levels: PENGUIN_HB,
        hummingbird_packed_only: true,
        default_ctx: Some(DEFAULT_CTX),
        max_ctx: None,
        split_levels: PENGUIN_HB,
        experts_in_ram_levels: PENGUIN_HB,
    },
    Support {
        family: Family::Glm,
        name: "GLM-5.3-Flash",
        levels: PENGUIN_HB,
        hummingbird_packed_only: true,
        default_ctx: Some(DEFAULT_CTX),
        max_ctx: None,
        split_levels: PENGUIN_HB,
        experts_in_ram_levels: PENGUIN_HB,
    },
    Support {
        family: Family::DeepSeekV4,
        name: "DeepSeek-V4",
        levels: PENGUIN_HB,
        hummingbird_packed_only: true,
        default_ctx: Some(DEFAULT_CTX),
        max_ctx: None,
        split_levels: PENGUIN_HB,
        experts_in_ram_levels: PENGUIN_HB,
    },
    Support {
        family: Family::Muse,
        name: "Muse Spark",
        levels: PENGUIN_HB,
        hummingbird_packed_only: true,
        default_ctx: Some(DEFAULT_CTX),
        max_ctx: None,
        split_levels: PENGUIN_HB,
        experts_in_ram_levels: PENGUIN_HB,
    },
    Support {
        family: Family::FlashNext,
        name: "Qwen3.8-Flash-Next",
        levels: NONE,
        hummingbird_packed_only: false,
        default_ctx: None,
        max_ctx: None,
        split_levels: NONE,
        experts_in_ram_levels: NONE,
    },
];

/// The row of `family`.
pub fn row(family: Family) -> &'static Support {
    TABLE.iter().find(|s| s.family == family).expect("every family has a row")
}

/// The row by the name the table gives it (`Support::name`: "Kimi-K3", "Gemma 4").
pub fn row_named(name: &str) -> Option<&'static Support> {
    TABLE.iter().find(|s| s.name == name)
}

/// The context `row`'s path takes where none is asked (the model's own limit where lower): the table's, else the lossy paths' most.
pub fn default_context(row: &Support, model_max: usize) -> usize {
    row.default_ctx.unwrap_or(LOSSY_CTX).min(model_max)
}

/// The refusal for a context asked for beyond what `row`'s path runs or the model has (`model_max`): the one `fit`, `run`, `serve` and `preflight` all give.
pub fn context_check(name: &str, row: &Support, model_max: usize, asked: Option<usize>) -> Result<(), Refusal> {
    let most = model_max.min(row.max_ctx.unwrap_or(usize::MAX));
    match asked {
        Some(c) if c > most => Err(Refusal::unsupported(
            format!("{name} takes at most {most} tokens of context here, and {c} were asked for"),
            format!("Ask for less: --context {most}"),
        )),
        _ => Ok(()),
    }
}

impl Support {
    /// Whether `run` takes this family at `level` (`model`: the name asked; every build runs a family's levels since v0.34.0, wren
    /// included, which a release build ran on quantization-aware checkpoints alone before).
    pub fn runs_at(&self, level: &str, _model: &str) -> bool {
        self.allows(level)
    }

    /// Whether `level` is one of this family's.
    pub fn allows(&self, level: &str) -> bool {
        self.levels.contains(&level)
    }

    /// The levels this build runs this family at, for `model`.
    pub fn levels_now(&self, model: &str) -> Vec<&'static str> {
        self.levels.iter().copied().filter(|l| self.runs_at(l, model)).collect()
    }
}

/// The family of a config.json, or the refusal `run` gives for it. `name`: the model's short name, for the words.
pub fn family(cfg_text: &str, name: &str) -> Result<Family, Refusal> {
    // (config.json parsed once: the family by its names, `runtime::config`'s one rule)
    let c = crate::runtime::config::ModelConfig::parse(cfg_text).map_err(|e| Refusal::unsupported(format!("{name}: {}", e.msg), ""))?;
    if c.family() == Some(Family::Gemma4) {
        // (the 26B-A4B mixture of experts and the E models are refused whatever the level)
        if let Err(e) = crate::gemma4::ConfigG4::parse(cfg_text, None, false) {
            if e.msg.contains("not in this engine yet") {
                return Err(Refusal::unsupported(format!("{name}: {}", e.msg), "Run Gemma 4 12B or 31B: glyd run google/gemma-4-12B-it"));
            }
        }
    }
    let arch = c.arch.as_str();
    match c.family() {
        Some(f) => Ok(f),
        None if arch.is_empty() => {
            Err(Refusal::unsupported(format!("{name}: its config.json does not say what kind of model it is"), "The engine runs Qwen3 models so far"))
        }
        // (the frozen Python tool reads this sentence to hand the model to vLLM; keep its words)
        _ => Err(Refusal::unsupported(
            format!("{name} is a {arch} model, and this engine runs Qwen3 models so far"),
            crate::machine::instead(
                "Other models run with the vLLM-based glyd, which is installed here",
                &format!("Pick a Qwen3 model (glyd run Qwen/Qwen3-8B), or {}", crate::machine::FULL_TOOL),
            ),
        )),
    }
}

/// The row `run` uses for `cfg_text` at `level`, or its refusal. `model`: the name asked, without its level (`base` in
/// the fix); `name`: its short name.
pub fn check(cfg_text: &str, model: &str, name: &str, level: &str) -> Result<&'static Support, Refusal> {
    let s = row(family(cfg_text, name)?);
    if s.runs_at(level, model) {
        return Ok(s);
    }
    let now = s.levels_now(model);
    if now.is_empty() {
        return Err(Refusal::unsupported(
            format!("{name} ({}) does not run with glyd run yet", s.name),
            "Run it with Condor: condor/README.md (glyd run takes it with ROADMAP P8)",
        ));
    }
    let fix = match now[0] {
        "penguin" => format!("Run it lossless: glyd run {model}"),
        l => format!("Run it at the {l} level: glyd run {model}:{l}"),
    };
    let list = match now.as_slice() {
        [one] => format!("the {one} level"),
        [rest @ .., last] => format!("the {} and {last} levels", rest.join(", ")),
        [] => unreachable!(),
    };
    Err(Refusal::unsupported(format!("{name} ({}) runs at {list} in this engine, not {level}", s.name), fix))
}

/// The refusal `run` gives on `m`'s GPU with the most free memory (the one a run uses), or None where it runs there:
/// the same conditions as `machine::gpu_refusal` for a run that was asked for, and the vendors and the CPU-only machine
/// this build has no backend for.
pub fn machine(m: &Machine) -> Option<Refusal> {
    let Some(d) = m.devices.iter().max_by_key(|d| (d.free, std::cmp::Reverse(d.name.len()))) else {
        return Some(Refusal::unsupported(
            "no GPU was found: the engine's CPU backend is not built yet",
            "See ROADMAP V4; on this computer glyd compresses files: glyd --help",
        ));
    };
    match d.vendor {
        Vendor::Amd => {
            return Some(Refusal::unsupported(
                format!("{}: AMD GPUs run in development builds only (GLYD_HIP=1)", d.name),
                "See ROADMAP V1 for AMD in releases; CONTRIBUTING.md builds it",
            ))
        }
        Vendor::Apple => return Some(Refusal::unsupported(format!("{}: Apple GPUs are not supported yet", d.name), "See ROADMAP V3 (Metal through MLX)")),
        Vendor::Nvidia => {}
    }
    if d.cc.0 < 8 {
        return Some(Refusal::unsupported(
            format!("{} (compute capability {}.{}) is older than the engine runs on", d.name, d.cc.0, d.cc.1),
            "glyd run needs the Ampere generation or newer: RTX 30 and 40 series, A10, A100, L4, L40S, H100 (Turing: ROADMAP V5)",
        ));
    }
    if !ARCHS.contains(&m.arch.as_str()) {
        return Some(Refusal::unsupported(
            format!("the engine has not run on a {} computer yet", m.arch),
            format!("It runs on {} computers", ARCHS.join(" and ")),
        ));
    }
    if d.driver < driver_min(d.cc) {
        return Some(Refusal::unsupported(
            format!("the driver supports CUDA {}.{}: the engine's kernels need {}", d.driver / 1000, d.driver % 1000 / 10, driver_need(d.cc)),
            format!("Update the NVIDIA driver to {} or newer (https://www.nvidia.com/drivers)", crate::machine::driver_series(driver_min(d.cc))),
        ));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use glyd_gpu::json::{self, Value};

    fn fixture(name: &str) -> String {
        std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)).unwrap()
    }

    const QWEN3: &str = r#"{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128}"#;

    #[test]
    fn every_family_is_told_apart() {
        assert_eq!(family(QWEN3, "q").unwrap(), Family::Qwen3);
        let q35: Value = json::parse(&fixture("qwen35_configs.json")).unwrap();
        let first = q35.as_object().unwrap()[0].1.clone();
        assert_eq!(family(&json::to_compact(&first), "q").unwrap(), Family::Qwen35);
        assert_eq!(family(&fixture("nemotron_h_config.json"), "n").unwrap(), Family::NemotronH);
        assert_eq!(family(&fixture("gemma4_12b_config.json"), "g").unwrap(), Family::Gemma4);
        assert_eq!(family(&fixture("glm5_next_config.json"), "g").unwrap(), Family::Glm);
        assert_eq!(family(&fixture("qwen4x_config.json"), "x").unwrap(), Family::FlashNext);
        assert_eq!(family(r#"{"model_type": "deepseek_v4", "architectures": ["DeepseekV4ForCausalLM"]}"#, "d").unwrap(), Family::DeepSeekV4);
        let r = family(r#"{"architectures": ["Qwen2ForCausalLM"]}"#, "Qwen2.5-7B").unwrap_err();
        assert!(r.what == "Qwen2.5-7B is a Qwen2ForCausalLM model, and this engine runs Qwen3 models so far" && r.code == crate::ui::CANNOT, "{r:?}");
    }

    #[test]
    fn levels_follow_the_table() {
        assert!(check(QWEN3, "Qwen/Qwen3-8B", "Qwen3-8B", "penguin").is_ok());
        let r = check(QWEN3, "Qwen/Qwen3-8B", "Qwen3-8B", "swift").unwrap_err();
        assert_eq!(
            (r.what.as_str(), r.fix.as_str()),
            ("Qwen3-8B (Qwen3) runs at the penguin level in this engine, not swift", "Run it lossless: glyd run Qwen/Qwen3-8B")
        );
        let n = fixture("nemotron_h_config.json");
        for l in ["penguin", "kestrel", "swift", "hummingbird"] {
            assert!(check(&n, "nvidia/N", "N", l).is_ok(), "{l}");
        }
        assert!(check(&n, "nvidia/N", "N", "wren").is_ok(), "wren runs in every build");
        let x = check(&fixture("qwen4x_config.json"), "Qwen/X", "X", "penguin").unwrap_err();
        assert!(x.fix.contains("condor/README.md"), "{x:?}");
        let g = fixture("gemma4_12b_config.json");
        assert!(check(&g, "google/gemma-4-12B-it", "gemma-4-12B-it", "wren").is_ok(), "a QAT checkpoint's wren runs in every build");
        assert!(check(&g, "google/gemma-4-12B-it", "gemma-4-12B-it", "hummingbird").is_err());
    }

    #[test]
    fn the_machine_side_matches_gpu_refusal() {
        let ok = Machine::preset("rtx4080s").unwrap();
        assert!(machine(&ok).is_none());
        for (name, word) in [("rx7900xtx", "AMD"), ("m4max-64g", "Apple")] {
            let r = machine(&Machine::preset(name).unwrap()).unwrap_or_else(|| panic!("{name}"));
            assert!(r.what.contains(word) || r.fix.contains(word), "{name}: {r:?}");
            assert_eq!(r.code, crate::ui::CANNOT);
        }
        // (a DGX Spark is an aarch64 computer, which the engine runs on as on an x86_64 one; an architecture it is not built for is refused)
        assert!(machine(&Machine::preset("dgx-spark").unwrap()).is_none());
        let mut other = Machine::preset("rtx4080s").unwrap();
        other.arch = "riscv64".into();
        let r = machine(&other).unwrap();
        assert!(r.what.contains("riscv64") && r.fix.contains("x86_64 and aarch64"), "{r:?}");
        let mut old = Machine::preset("rtx4080s").unwrap();
        old.devices[0].driver = 12070;
        assert!(machine(&old).unwrap().what.contains("CUDA 12.7"));
    }
}
