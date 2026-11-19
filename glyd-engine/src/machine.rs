//! Whether `glyd run` hands a model to this engine on this computer, said before anything is downloaded (`glyd-engine preflight MODEL`): the checks of the GPU a run would use (above all whether the
//! engine is measured at least as fast as vLLM on it: `GPUS`), of its driver, of the model's config.json, of the disks, and of what the lossless KV cache takes. `glyd run` starts the engine only
//! where all of them pass, and runs the model the way the earlier release did (vLLM) everywhere else, with one line saying why. Pure functions of what the driver and the files report, so that every
//! branch is a host test.

use crate::dev::Gpu;
use crate::ui::{gb, Refusal};
use crate::weights::Config;

/// The CUDA version a driver of the series 580 reports (13.0): the engine's kernels are built with CUDA 13 and run on no older driver.
pub const DRIVER_MIN: i32 = 13000;

/// The GPUs `glyd run` hands a model to the engine on, one line each: the word the driver's name for the GPU holds (a whole word: "L4" is not an L40S, "H100" is not an H200) and the GPU's compute capability.
///
/// THE SPEED BAR: a GPU is here only where the engine has been measured at least as fast as vLLM at every load it would serve there, with less memory (the engine serves up to 8 users at once, so
/// `glyd serve` is vLLM's, whatever is here), and has run on the GPU. Measurements decide it, not what the engine can run: it is EMPTY until they are made, and while it is empty every model runs as
/// it did in v0.28 (vLLM). Every GPU not named here, and every compute capability, runs vLLM. Adding a GPU, once the engine is measured on it, is one line here, e.g. `("L4", (8, 9)),`.
pub const GPUS: &[(&str, (i32, i32))] = &[];

/// The architectures of the computers the engine has run on (`std::env::consts::ARCH`): a GH200 is a Grace computer, and the engine has not run on one.
pub const ARCHS: &[&str] = &["x86_64"];

/// The most positions the KV cache's tables cover.
pub const MAX_CONTEXT: usize = crate::kvcal::TIERS[crate::kvcal::TIERS.len() - 1];

/// The most query heads that share a KV head, which the engine's attention takes.
pub const MAX_GROUP: usize = 8;

/// The room kept on a disk beyond what a first run writes.
const SPARE: u64 = 1 << 30;

fn cannot(what: impl Into<String>, fix: impl Into<String>) -> Refusal {
    Refusal::unsupported(what, fix)
}

/// Whether `word` is in `text` between characters that are not letters, digits or underscores (a regular expression's `\bword\b`).
fn has_word(text: &str, word: &str) -> bool {
    let (b, w) = (text.as_bytes(), word.len());
    let in_word = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    text.match_indices(word).any(|(i, _)| (i == 0 || !in_word(b[i - 1])) && (i + w == b.len() || !in_word(b[i + w])))
}

/// Whether `g` is one of `GPUS`: its name has the word and its compute capability is the line's.
pub fn listed(g: &Gpu) -> bool {
    on_list(GPUS, g)
}

fn on_list(table: &[(&str, (i32, i32))], g: &Gpu) -> bool {
    table.iter().any(|(word, cc)| *cc == g.cc && has_word(&g.name, word))
}

/// "A100, L4, H100 or GH200".
fn joined(words: &[&str]) -> String {
    match words {
        [] => String::new(),
        [one] => one.to_string(),
        [rest @ .., last] => format!("{} or {last}", rest.join(", ")),
    }
}

/// A slice of a GPU (Multi-Instance GPU), which the driver names so ("NVIDIA A100-SXM4-40GB MIG 1g.5gb").
fn is_mig(name: &str) -> bool {
    has_word(name, "MIG")
}

/// The GPU a run uses: the one with the most free memory among those new enough (an Ampere or a later one) that are not a MIG slice, and where there is none of those the freest of the new
/// enough, then the freest (the earlier release's choice, which this one keeps whichever of the two programs runs the model). With the same room, the first. The driver lists the GPUs
/// CUDA_VISIBLE_DEVICES names, in its order, and no others.
pub fn pick(gpus: &[Gpu]) -> Option<&Gpu> {
    let capable: Vec<&Gpu> = gpus.iter().filter(|g| g.cc.0 >= 8).collect();
    let usable: Vec<&Gpu> = capable.iter().copied().filter(|g| !is_mig(&g.name)).collect();
    let set = if !usable.is_empty() {
        usable
    } else if !capable.is_empty() {
        capable
    } else {
        gpus.iter().collect()
    };
    set.into_iter().reduce(|best, g| if g.free > best.free { g } else { best })
}

/// Why `glyd run` does not hand a model to the engine on this GPU, or None where it does: where `GPUS` is empty, that (the speed bar); else a MIG slice or a card older than Ampere, the architecture
/// of the computer (`ARCHS`), the GPU's compute capability (those of `GPUS`), the driver (CUDA 13), and the GPU itself (`GPUS`: where it is measured). `arch` is `std::env::consts::ARCH`. `asked`:
/// the user asked for the engine by name (GLYD_ENGINE=1, which `glyd run` passes on as GLYD_ENGINE_ASKED=1): `GPUS` is not asked, and the rest is.
pub fn gpu_refusal(g: &Gpu, arch: &str, asked: bool) -> Option<Refusal> {
    refusal_on(GPUS, ARCHS, g, arch, asked)
}

fn refusal_on(table: &[(&str, (i32, i32))], archs: &[&str], g: &Gpu, arch: &str, asked: bool) -> Option<Refusal> {
    let (major, minor) = g.cc;
    let words: Vec<&str> = table.iter().map(|(w, _)| *w).collect();
    if table.is_empty() && !asked {
        // (no GPU is measured yet: the speed bar is the whole answer, whatever the GPU, its driver or its computer)
        return Some(cannot(not_measured(&g.name, &words), "vLLM runs the model on this GPU"));
    }
    if is_mig(&g.name) {
        return Some(cannot(format!("{} is a MIG slice of a GPU, which this engine does not size", g.name), "Use the whole GPU"));
    }
    if major < 8 {
        return Some(cannot(
            format!("{} (compute capability {major}.{minor}) is older than the engine runs on", g.name),
            "glyd run needs the Ampere generation or newer: RTX 30 and 40 series, A10, A100, L4, L40S, H100",
        ));
    }
    if !archs.contains(&arch) {
        return Some(cannot(format!("this engine has not run on {arch} computers yet ({})", g.name), format!("It runs on {} computers so far; vLLM runs the model here", archs.join(" and "))));
    }
    if !asked && !table.iter().any(|(_, cc)| *cc == g.cc) {
        return Some(cannot(
            format!("this engine has not run on {} (compute capability {major}.{minor}) yet: it runs on the {}", g.name, joined(&words)),
            "vLLM runs the model on this GPU",
        ));
    }
    if g.driver < DRIVER_MIN {
        return Some(cannot(
            format!("the NVIDIA driver here runs CUDA {}.{}, and this engine's kernels need CUDA 13 (driver 580 or newer)", g.driver / 1000, g.driver % 1000 / 10),
            "Update the NVIDIA driver to 580 or newer (https://www.nvidia.com/drivers), or run the model with vLLM",
        ));
    }
    if !asked && !on_list(table, g) {
        return Some(cannot(not_measured(&g.name, &words), "vLLM runs the model on this GPU"));
    }
    None
}

/// What is said of a GPU that is not one the engine is used on: it is not measured at least as fast as vLLM there (the speed bar), and where it is, on which GPUs.
fn not_measured(gpu: &str, words: &[&str]) -> String {
    let on = if words.is_empty() { String::new() } else { format!(" ({})", joined(words)) };
    format!("{gpu} is not a GPU the engine is measured at least as fast as vLLM on{on}")
}

/// Why the model's shape does not fit the engine's KV cache, or None: its attention heads take 128 values, and up to `MAX_GROUP` query heads share a KV head.
pub fn config_refusal(name: &str, c: &Config) -> Option<Refusal> {
    if c.head_dim != 128 {
        return Some(cannot(format!("{name}: its attention heads have {} values, and the engine's KV cache takes 128", c.head_dim), "Other models run with vLLM"));
    }
    if c.kv_heads == 0 || !c.heads.is_multiple_of(c.kv_heads) || c.heads / c.kv_heads > MAX_GROUP {
        return Some(cannot(format!("{name} has {} query heads to {} KV heads, and the engine takes up to {MAX_GROUP} to one", c.heads, c.kv_heads), "Other models run with vLLM"));
    }
    None
}

/// Why `ctx` tokens of context are more than the KV cache's tables cover, or None.
pub fn context_refusal(name: &str, ctx: usize) -> Option<Refusal> {
    (ctx > MAX_CONTEXT).then(|| {
        cannot(
            format!("{name} with {ctx} tokens of context: the lossless KV cache's tables cover up to {MAX_CONTEXT}"),
            format!("Ask for up to {MAX_CONTEXT}: --context {MAX_CONTEXT}, or run the model with vLLM"),
        )
    })
}

/// Why the disks cannot take what a first run writes, or None. `download`: the bytes of the model still to be fetched into the models' folder; `pack`: the bytes of the packed copy the engine's
/// cache is to take (0 where it is made already); `models` and `cache`: the free bytes of the disks holding the two (None where that cannot be told); `same`: they are one filesystem, where
/// the two sums add. A gibibyte is left over on each.
pub fn disk_refusal(name: &str, models: Option<u64>, cache: Option<u64>, same: bool, download: u64, pack: u64) -> Option<Refusal> {
    let fix = "Free some space, or keep the models on another drive: HF_HOME=/path/on/that/drive";
    let short = |free: Option<u64>, need: u64| free.filter(|&f| need > 0 && f < need + SPARE);
    if same {
        let free = models.or(cache);
        return short(free, download + pack).map(|f| cannot(format!("the disk has {} left, and the first run of {name} writes {} (the download and the packed copy)", gb(f), gb(download + pack)), fix));
    }
    if let Some(f) = short(models, download) {
        return Some(cannot(format!("the disk holding the downloaded models has {} left, and {name} takes {} to download", gb(f), gb(download)), fix));
    }
    short(cache, pack).map(|f| cannot(format!("the disk holding the engine's cache has {} left, and the packed copy of {name} takes {}", gb(f), gb(pack)), "Free some space, or put the cache on another drive: GLYD_CACHE=/path/on/that/drive"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lines of the kind `GPUS` gets once a GPU is measured, to test every rule with: the shipped list is a decision of the measurements (it is empty until they are made).
    const TABLE: &[(&str, (i32, i32))] = &[("A100", (8, 0)), ("L4", (8, 9)), ("H100", (9, 0)), ("GH200", (9, 0))];

    fn gpu(name: &str, cc: (i32, i32), free_gib: u64, driver: i32) -> Gpu {
        Gpu { name: name.to_string(), cc, sms: 80, total: free_gib << 30, free: free_gib << 30, driver, count: 1, ordinal: 0 }
    }

    fn qwen3_8b() -> Config {
        Config { hidden: 4096, layers: 36, heads: 32, kv_heads: 8, head_dim: 128, inter: 12288, vocab: 151936, eps: 1e-6, rope_theta: 1e6, max_pos: 40960, tie: false, eos: vec![] }
    }

    #[test]
    fn the_gpus_the_engine_runs_on_by_the_names_the_driver_gives() {
        let named = |name: &str, cc: (i32, i32)| gpu(name, cc, 24, 13000);
        for (yes, cc) in [
            ("NVIDIA A100-SXM4-80GB", (8, 0)),
            ("NVIDIA A100 80GB PCIe", (8, 0)),
            ("NVIDIA A100-PCIE-40GB", (8, 0)),
            ("NVIDIA L4", (8, 9)),
            ("NVIDIA H100 80GB HBM3", (9, 0)),
            ("NVIDIA H100 PCIe", (9, 0)),
            ("NVIDIA H100 NVL", (9, 0)),
            ("NVIDIA GH200 480GB", (9, 0)),
            ("L4", (8, 9)),
        ] {
            assert!(on_list(TABLE, &named(yes, cc)), "{yes}");
        }
        // the neighbours are not: an L40S is not an L4, an A100X is not an A100, nor an H200 an H100; and a name on the list with another compute capability is not that GPU
        for (no, cc) in [
            ("NVIDIA L40S", (8, 9)),
            ("NVIDIA L40", (8, 9)),
            ("NVIDIA H200", (9, 0)),
            ("NVIDIA A10G", (8, 6)),
            ("NVIDIA A10", (8, 6)),
            ("NVIDIA A100X", (8, 0)),
            ("NVIDIA GeForce RTX 4090", (8, 9)),
            ("NVIDIA RTX A6000", (8, 6)),
            ("NVIDIA B200", (10, 0)),
            ("NVIDIA GeForce RTX 5090", (12, 0)),
            ("Tesla T4", (7, 5)),
            ("", (8, 9)),
            ("NVIDIA L4X", (8, 9)),
            ("XL4", (8, 9)),
            ("NVIDIA A100-SXM4-80GB", (8, 9)),
        ] {
            assert!(!on_list(TABLE, &named(no, cc)), "{no}");
        }
    }

    /// The list is one constant: a GPU added to it (here an RTX 4090, as the measurement of one would add it) is handed models, and said in every refusal that names the list; nothing else changes.
    #[test]
    fn a_gpu_added_to_the_list_is_one_line() {
        let with = [TABLE, &[("RTX 4090", (8, 9))]].concat();
        let g4090 = gpu("NVIDIA GeForce RTX 4090", (8, 9), 24, 13000);
        assert!(refusal_on(TABLE, ARCHS, &g4090, "x86_64", false).is_some() && refusal_on(&with, ARCHS, &g4090, "x86_64", false).is_none());
        // (the neighbours of the new line stay out: a 4080 is no 4090, an L40S is not an RTX 4090)
        assert!(refusal_on(&with, ARCHS, &gpu("NVIDIA GeForce RTX 4080 SUPER", (8, 9), 16, 13000), "x86_64", false).is_some());
        assert!(refusal_on(&with, ARCHS, &gpu("NVIDIA L40S", (8, 9), 44, 13000), "x86_64", false).unwrap().what.contains("A100, L4, H100, GH200 or RTX 4090"));
        // and the refusals say the list the table holds, whatever it holds
        let said = refusal_on(TABLE, ARCHS, &gpu("NVIDIA B200", (10, 0), 180, 13000), "x86_64", false).unwrap().what;
        assert!(said.ends_with("it runs on the A100, L4, H100 or GH200"), "{said}");
        // an architecture added to ARCHS is one line too
        let grace = gpu("NVIDIA GH200 480GB", (9, 0), 90, 13000);
        assert!(refusal_on(TABLE, ARCHS, &grace, "aarch64", false).is_some() && refusal_on(TABLE, &["x86_64", "aarch64"], &grace, "aarch64", false).is_none());
        // the shipped list: an L4 is handed models by the line `("L4", (8, 9))`, which is all it takes
        let l4 = gpu("NVIDIA L4", (8, 9), 22, 13000);
        assert!(refusal_on(&[GPUS, &[("L4", (8, 9))]].concat(), ARCHS, &l4, "x86_64", false).is_none());
    }

    /// The speed bar: the engine is used only where it is measured at least as fast as vLLM, and `GPUS` is where that is written. Whatever it holds, a GPU it does not name is refused with the bar as the reason
    /// (and the fix is vLLM), a GPU it names has a compute capability the engine runs on, and the shipped list is no more than a decision of measurements: where it is empty, every GPU is refused first for the bar,
    /// whatever else is wrong with it, with no list to name.
    #[test]
    fn the_engine_is_used_only_where_it_is_measured_at_least_as_fast_as_vllm() {
        for (word, cc) in GPUS {
            assert!(!word.is_empty() && cc.0 >= 8, "{word} {cc:?}");
        }
        let unlisted = [("NVIDIA GeForce RTX 3060 Ti", (8, 6)), ("Some Future GPU 9000", (8, 9)), ("NVIDIA GeForce RTX 5090", (12, 0))];
        for (name, cc) in unlisted {
            let g = gpu(name, cc, 24, 13000);
            if on_list(GPUS, &g) {
                continue; // (a GPU the measurements added is the case that is meant)
            }
            let r = gpu_refusal(&g, "x86_64", false).unwrap_or_else(|| panic!("{name} was not refused"));
            assert!(r.code == 3 && r.what.contains(name) && r.what.contains("measured at least as fast as vLLM") && r.fix == "vLLM runs the model on this GPU", "{r:?}");
        }
        // an empty list: the bar first, for every kind of GPU and computer, and no list in the words
        for (g, arch) in [
            (gpu("NVIDIA L4", (8, 9), 22, 13000), "x86_64"),
            (gpu("NVIDIA A100-SXM4-80GB", (8, 0), 70, 13020), "x86_64"),
            (gpu("NVIDIA H100 80GB HBM3", (9, 0), 70, 13000), "x86_64"),
            (gpu("NVIDIA L4", (8, 9), 22, 12080), "x86_64"),
            (gpu("NVIDIA GH200 480GB", (9, 0), 90, 13000), "aarch64"),
            (gpu("NVIDIA A100-SXM4-40GB MIG 1g.5gb", (8, 0), 5, 13000), "x86_64"),
            (gpu("Tesla T4", (7, 5), 15, 13000), "x86_64"),
        ] {
            let r = refusal_on(&[], ARCHS, &g, arch, false).unwrap();
            assert_eq!((r.code, r.what.as_str()), (3, format!("{} is not a GPU the engine is measured at least as fast as vLLM on", g.name).as_str()), "{} {arch}", g.name);
            assert_eq!(r.fix, "vLLM runs the model on this GPU");
        }
        // and with a list, the bar is the last thing asked of a GPU that has run, and says where the engine is measured
        let r = refusal_on(TABLE, ARCHS, &gpu("NVIDIA GeForce RTX 4090", (8, 9), 24, 13000), "x86_64", false).unwrap();
        assert!(r.what.ends_with("vLLM on (A100, L4, H100 or GH200)"), "{r:?}");
        // asked for by name (GLYD_ENGINE=1): no list is asked, an empty one or not, and every other check is
        for table in [&[][..], TABLE] {
            for (name, cc) in [("NVIDIA GeForce RTX 4090", (8, 9)), ("NVIDIA GeForce RTX 5090", (12, 0)), ("NVIDIA B200", (10, 0)), ("NVIDIA A10", (8, 6))] {
                assert!(refusal_on(table, ARCHS, &gpu(name, cc, 24, 13000), "x86_64", true).is_none(), "{name}");
            }
            for (g, arch, said) in [
                (gpu("NVIDIA A100-SXM4-40GB MIG 1g.5gb", (8, 0), 5, 13000), "x86_64", "MIG slice"),
                (gpu("Tesla T4", (7, 5), 15, 13000), "x86_64", "older"),
                (gpu("NVIDIA GH200 480GB", (9, 0), 90, 13000), "aarch64", "aarch64"),
                (gpu("NVIDIA GeForce RTX 4090", (8, 9), 24, 12080), "x86_64", "CUDA 12.8"),
            ] {
                let r = refusal_on(table, ARCHS, &g, arch, true).unwrap();
                assert!(r.code == 3 && r.what.contains(said), "{r:?}");
            }
        }
    }

    #[test]
    fn the_freest_gpu_is_the_one_a_run_uses() {
        let a100 = Gpu { ordinal: 0, ..gpu("NVIDIA A100-SXM4-80GB", (8, 0), 70, 13000) };
        let l4 = Gpu { ordinal: 1, ..gpu("NVIDIA L4", (8, 9), 22, 13000) };
        assert_eq!(pick(&[a100.clone(), l4.clone()]).unwrap().ordinal, 0);
        // the A100 is busy: the L4 has more room
        let busy = Gpu { free: 5 << 30, ..a100.clone() };
        assert_eq!(pick(&[busy.clone(), l4.clone()]).unwrap().ordinal, 1);
        // the same room: the first
        let twin = Gpu { ordinal: 1, ..a100.clone() };
        assert_eq!(pick(&[a100.clone(), twin.clone()]).unwrap().ordinal, 0);
        assert_eq!(pick(&[twin.clone(), a100.clone()]).unwrap().ordinal, 1);
        // an older card with the most memory is not the one to use while a new enough one is there; a MIG slice is not either
        let t4 = Gpu { ordinal: 0, ..gpu("Tesla T4", (7, 5), 90, 13000) };
        assert_eq!(pick(&[t4.clone(), l4.clone()]).unwrap().ordinal, 1);
        let mig = Gpu { ordinal: 0, ..gpu("NVIDIA A100-SXM4-40GB MIG 1g.5gb", (8, 0), 100, 13000) };
        assert_eq!(pick(&[mig.clone(), l4.clone()]).unwrap().ordinal, 1);
        // where there is nothing else: the freest new enough (a MIG slice), then the freest of all
        assert_eq!(pick(std::slice::from_ref(&mig)).unwrap().ordinal, 0);
        let t4b = Gpu { ordinal: 1, free: 8 << 30, ..t4.clone() };
        assert_eq!(pick(&[t4b.clone(), t4.clone()]).unwrap().ordinal, 0);
        assert!(pick(&[]).is_none());
    }

    #[test]
    fn every_reason_a_gpu_is_refused_for_and_the_order_they_are_said_in() {
        let ok = gpu("NVIDIA L4", (8, 9), 22, 13000);
        let said = |g: Gpu, arch: &str| refusal_on(TABLE, ARCHS, &g, arch, false).map(|r| (r.code, r.what));
        assert!(said(ok.clone(), "x86_64").is_none());
        for (name, cc) in [("NVIDIA A100-SXM4-80GB", (8, 0)), ("NVIDIA H100 80GB HBM3", (9, 0))] {
            assert!(said(gpu(name, cc, 70, 13020), "x86_64").is_none(), "{name}");
        }
        // a slice, a card that is too old
        assert!(said(gpu("NVIDIA A100-SXM4-40GB MIG 1g.5gb", (8, 0), 5, 13000), "x86_64").unwrap().1.contains("MIG slice"));
        let old = said(gpu("Tesla T4", (7, 5), 15, 13000), "x86_64").unwrap();
        assert!(old.0 == 3 && old.1.contains("7.5") && old.1.contains("older"), "{old:?}");
        // a computer that is not x86-64 (a GH200 is a Grace and a Hopper: it has not run there)
        let arm = said(gpu("NVIDIA GH200 480GB", (9, 0), 90, 13000), "aarch64").unwrap();
        assert!(arm.0 == 3 && arm.1.contains("aarch64") && arm.1.contains("GH200"), "{arm:?}");
        // Blackwell, Ampere's A10, Turing are not among the capabilities it has run on
        for (name, cc) in [("NVIDIA B200", (10, 0)), ("NVIDIA GeForce RTX 5090", (12, 0)), ("NVIDIA A10", (8, 6))] {
            let r = said(gpu(name, cc, 24, 13000), "x86_64").unwrap();
            assert!(r.0 == 3 && r.1.contains("has not run on") && r.1.contains(name), "{r:?}");
        }
        // the driver: the CUDA 12 drivers (550 to 575) the earlier release accepted
        for (driver, text) in [(12040, "CUDA 12.4"), (12080, "CUDA 12.8"), (12090, "CUDA 12.9")] {
            let r = said(gpu("NVIDIA L4", (8, 9), 22, driver), "x86_64").unwrap();
            assert!(r.0 == 3 && r.1.contains(text) && r.1.contains("580"), "{r:?}");
        }
        assert!(said(gpu("NVIDIA L4", (8, 9), 22, 13000), "x86_64").is_none() && said(gpu("NVIDIA L4", (8, 9), 22, 13020), "x86_64").is_none());
        // the speed bar: measured on four GPUs; an RTX 4090 and an L40S are Ada, which has run, and are not among them
        for name in ["NVIDIA GeForce RTX 4090", "NVIDIA L40S"] {
            let r = said(gpu(name, (8, 9), 24, 13000), "x86_64").unwrap();
            assert!(r.0 == 3 && r.1.contains("measured at least as fast as vLLM") && r.1.contains(name), "{r:?}");
        }
        // an old driver on an RTX 4090 is told for its driver, which is what the user can change
        assert!(said(gpu("NVIDIA GeForce RTX 4090", (8, 9), 24, 12080), "x86_64").unwrap().1.contains("driver"));
    }

    #[test]
    fn the_shape_of_a_model_the_cache_takes() {
        assert!(config_refusal("Qwen3-8B", &qwen3_8b()).is_none());
        assert!(config_refusal("Qwen3-32B", &Config { heads: 64, ..qwen3_8b() }).is_none()); // (8 to a KV head: the most)
        let small_head = config_refusal("x", &Config { head_dim: 64, ..qwen3_8b() }).unwrap();
        assert!(small_head.code == 3 && small_head.what.contains("64 values"), "{small_head}");
        let wide = config_refusal("x", &Config { heads: 72, ..qwen3_8b() }).unwrap();
        assert!(wide.code == 3 && wide.what.contains("72 query heads"), "{wide}");
        assert!(config_refusal("x", &Config { heads: 30, kv_heads: 8, ..qwen3_8b() }).is_some());
        assert!(config_refusal("x", &Config { kv_heads: 0, ..qwen3_8b() }).is_some());
    }

    #[test]
    fn the_context_the_tables_cover() {
        assert_eq!(MAX_CONTEXT, 40960);
        assert!(context_refusal("m", 40960).is_none() && context_refusal("m", 8192).is_none());
        let r = context_refusal("Qwen3-4B-Instruct-2507", 100_000).unwrap();
        assert!(r.code == 3 && r.what.contains("100000") && r.what.contains("40960") && r.fix.contains("--context 40960"), "{r}");
    }

    #[test]
    fn the_disks_for_a_first_run() {
        let gb = |n: u64| n * 1_000_000_000;
        // Qwen3-8B: 16.4 GB to download, 12.4 GB packed
        let (dl, pk) = (16_381_470_720u64, 12_400_000_000u64);
        // one disk: both together, and a gibibyte over
        assert!(disk_refusal("Qwen3-8B", Some(gb(50)), Some(gb(50)), true, dl, pk).is_none());
        let r = disk_refusal("Qwen3-8B", Some(gb(20)), Some(gb(20)), true, dl, pk).unwrap();
        assert!(r.code == 3 && r.what.contains("20.0 GB left") && r.what.contains("28.8 GB") && r.what.contains("the download and the packed copy"), "{r}");
        // the download alone fits 20 GB; with the pack made already, nothing more is written
        assert!(disk_refusal("Qwen3-8B", Some(gb(20)), Some(gb(20)), true, dl, 0).is_none());
        // the model cached already (nothing to download) and a pack to make
        assert!(disk_refusal("Qwen3-8B", Some(gb(20)), Some(gb(20)), true, 0, pk).is_none());
        assert!(disk_refusal("Qwen3-8B", Some(gb(13)), Some(gb(13)), true, 0, pk).is_some());
        // two disks: each on its own
        assert!(disk_refusal("Qwen3-8B", Some(gb(20)), Some(gb(14)), false, dl, pk).is_none());
        let r = disk_refusal("Qwen3-8B", Some(gb(10)), Some(gb(500)), false, dl, pk).unwrap();
        assert!(r.what.contains("downloaded models") && r.what.contains("10.0 GB left"), "{r}");
        let r = disk_refusal("Qwen3-8B", Some(gb(500)), Some(gb(12)), false, dl, pk).unwrap();
        assert!(r.what.contains("engine's cache") && r.what.contains("12.4 GB") && r.fix.contains("GLYD_CACHE"), "{r}");
        // a disk that cannot be asked (not Linux) is not a reason
        assert!(disk_refusal("m", None, None, true, dl, pk).is_none() && disk_refusal("m", None, None, false, dl, pk).is_none());
        // a spare gibibyte: exactly the sum is not enough
        assert!(disk_refusal("m", Some(gb(10)), None, true, gb(10) - (1 << 30), 0).is_none());
        assert!(disk_refusal("m", Some(gb(10)), None, true, gb(10) - (1 << 30) + 1, 0).is_some());
    }
}
