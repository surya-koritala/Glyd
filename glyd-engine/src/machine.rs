//! Whether `glyd run` hands a model to this engine on this computer, said before anything is downloaded (`glyd-engine preflight MODEL`): the checks of the GPU a run would use (above all whether the
//! engine is measured at least as fast as vLLM on it: `GPUS`), of its driver, of the model's config.json, of the disks, and of what the lossless KV cache takes. `glyd run` starts the engine only
//! where all of them pass, and runs the model the way the earlier release did (vLLM) everywhere else, with one line saying why. Pure functions of what the driver and the files report, so that every
//! branch is a host test.

use crate::dev::Gpu;
use crate::ui::{gb, Refusal};
use crate::weights::Config;

/// The CUDA version a driver of the series 580 reports (13.0): what a build with CUDA 13's kernels alone runs on.
pub const DRIVER_MIN: i32 = 13000;

/// Whether this build carries CUDA 12's kernels too (build.rs: NVCC12 built them, or the nvcc was CUDA 12's: cfg `kernels12`): it then runs on a driver of CUDA 12.2 (series 535), and on
/// CUDA 12.8 (570) for Blackwell, whose code CUDA 12 has from 12.8.
pub const KERNELS12: bool = cfg!(kernels12);

/// The oldest driver, as the CUDA version it reports, that a build runs a GPU of compute capability `cc` on: where it has CUDA 12's kernels 12.2 (Blackwell, compute capability 10 and 12: 12.8), else 13.0.
pub const fn driver_min_for(kernels12: bool, cc: (i32, i32)) -> i32 {
    if !kernels12 {
        DRIVER_MIN
    } else if cc.0 >= 10 {
        12080
    } else {
        12020
    }
}

/// `driver_min_for` this build.
pub fn driver_min(cc: (i32, i32)) -> i32 {
    driver_min_for(KERNELS12, cc)
}

/// The driver series that reports CUDA version `cuda` (13000 for 13.0): 580 for CUDA 13.0, 575 for 12.9, 570 for 12.8, 550 for 12.4, 535 for 12.2.
pub fn driver_series(cuda: i32) -> i32 {
    match cuda {
        13000.. => 580,
        12090.. => 575,
        12080.. => 570,
        12040.. => 550,
        12020.. => 535,
        _ => 525,
    }
}

/// "CUDA 13.0 (driver 580 or newer)": what the engine needs on a GPU of compute capability `cc`, as the refusals say it.
pub fn driver_need(cc: (i32, i32)) -> String {
    driver_need_for(KERNELS12, cc)
}

/// `driver_need` for a build with or without CUDA 12's kernels.
pub fn driver_need_for(kernels12: bool, cc: (i32, i32)) -> String {
    let min = driver_min_for(kernels12, cc);
    format!("CUDA {}.{} (driver {} or newer)", min / 1000, min % 1000 / 10, driver_series(min))
}

/// Where `glyd run` looks for the Python tool (the one with vLLM): the directories of PATH, where `uv tool install` puts programs (UV_TOOL_BIN_DIR, XDG_BIN_HOME) and ~/.local/bin.
fn tool_dirs() -> Vec<std::path::PathBuf> {
    use std::env::{split_paths, var_os};
    let mut dirs: Vec<std::path::PathBuf> = var_os("PATH").map(|p| split_paths(&p).collect()).unwrap_or_default();
    for v in ["UV_TOOL_BIN_DIR", "XDG_BIN_HOME"] {
        dirs.extend(var_os(v).map(std::path::PathBuf::from));
    }
    dirs.extend(var_os("HOME").map(|h| std::path::Path::new(&h).join(".local").join("bin")));
    dirs.into_iter().filter(|d| !d.as_os_str().is_empty()).collect()
}

/// Whether vLLM is in one of `dirs`: a `vllm` program there, or beside the Python of a `glyd` script there (its `#!` line), as `uv tool install "glyd[vllm]"` leaves it
/// (the environment's bin/vllm, and its lib/python*/site-packages/vllm). No Python is started: the files say.
pub fn vllm_in(dirs: &[std::path::PathBuf]) -> bool {
    use std::io::Read;
    let beside = |py: &std::path::Path| -> bool {
        let Some(bin) = py.parent() else { return false };
        if bin.join("vllm").is_file() {
            return true;
        }
        let lib = bin.join("..").join("lib");
        std::fs::read_dir(lib).into_iter().flatten().flatten().any(|e| e.path().join("site-packages").join("vllm").is_dir())
    };
    dirs.iter().any(|d| {
        if d.join("vllm").is_file() {
            return true;
        }
        let mut head = Vec::new();
        if std::fs::File::open(d.join("glyd")).and_then(|f| f.take(256).read_to_end(&mut head)).is_err() || !head.starts_with(b"#!") {
            return false;
        }
        let line = String::from_utf8_lossy(&head);
        let first = line.lines().next().unwrap_or("").trim_start_matches("#!").split_whitespace().next().unwrap_or("");
        !first.is_empty() && !first.ends_with("/env") && beside(std::path::Path::new(first))
    })
}

#[cfg(test)]
thread_local! {
    /// A test's answer to `vllm_installed` (the tests of the words a refusal uses set it; the others see the machine's).
    pub static FORCE_VLLM: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Whether vLLM is installed here (`vllm_in` the places the Python tool is): a refusal names vLLM as what runs the model only where it is.
pub fn vllm_installed() -> bool {
    #[cfg(test)]
    if let Some(v) = FORCE_VLLM.with(|f| f.get()) {
        return v;
    }
    static HERE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *HERE.get_or_init(|| vllm_in(&tool_dirs()))
}

/// What a refusal says to do instead: `with_vllm` where vLLM is installed (the model runs with it), else `rust_only`: what a computer with the Rust install alone can do.
pub fn instead(with_vllm: &str, rust_only: &str) -> String {
    (if vllm_installed() { with_vllm } else { rust_only }).to_string()
}

/// The Rust-only install's way to the models and GPUs the engine does not run: the whole tool.
pub const FULL_TOOL: &str = "install the full glyd, which runs it another way: curl -LsSf https://getglyd.com/install.sh | sh";

/// A line of the list: the word the driver's name for the GPU holds, the GPU's compute capability, and the model families the engine is measured on there.
pub type Entry = (&'static str, (i32, i32), &'static [&'static str]);

/// The family of the Qwen3 dense models (Qwen3ForCausalLM), the one `glyd run` hands to the engine by default: the lossless levels of the others are vLLM's, and their lossy levels are the engine's by the user's asking
/// (`:kestrel`, `:swift`), where this list is not looked at (vLLM has no lossy levels).
pub const QWEN3: &str = "Qwen3";

/// For a path that is the engine's whatever the list (the user asked for it by name, or a lossy level): the list is not looked at, and no family is named.
pub const ANY_FAMILY: &str = "";

/// The (GPU, family) pairs `glyd run` hands a model to the engine on, one line each: the word the driver's name for the GPU holds (a whole word: "L4" is not an L40S, "H100" is not an H200), the GPU's compute
/// capability, and the families.
///
/// THE SPEED BAR: a family is here on a GPU only where the engine has been measured at least as fast as vLLM at every load it would serve there, with less memory (the engine serves up to 8 users at once, so
/// `glyd serve` is vLLM's, whatever is here), and has run on the GPU. Measurements decide it, not what the engine can run: it was EMPTY until they were made (the first line, the RTX 4080 SUPER, is of 2026-11-20), and while it is empty every model runs as
/// it did in v0.28 (vLLM). Every GPU not named here, and every compute capability, runs vLLM. Adding a GPU, once the engine is measured on it, is one line here, e.g. `("L4", (8, 9), &["Qwen3"]),` (docs/adding-a-gpu.md: `scripts/gpu_bar.py` reads the two measurement files).
pub const GPUS: &[Entry] = &[
    // measured 2026-11-20 on glyd-pc, Qwen3-8B, 1, 4 and 8 users at once with short, 2,000 and 4,000-token prompts, 5 runs of each engine: benchmarks/gpu/rtx4080s-bar-2026-11-20
    ("RTX 4080 SUPER", (8, 9), &["Qwen3"]),
];

/// The architectures the engine plans for (`std::env::consts::ARCH`): an aarch64 computer (GH200, GB200, DGX Spark, Jetson Thor, an Arm workstation) is planned as an x86_64 one is, and is
/// refused, as one is, only by its GPU: the compute capability, the driver, and the measured list (`GPUS`), which names no Arm-hosted GPU until one is measured (#459). Another architecture
/// is refused here.
pub const ARCHS: &[&str] = &["x86_64", "aarch64"];

/// The architectures CI builds the engine on and runs its host-side unit tests on, with no GPU (ci.yml: `ubuntu-24.04-arm` for aarch64, where the CPU expert kernels run as NEON): what a
/// refusal by `ARCHS` of another architecture says is proven where.
pub const ARCHS_BUILT: &[&str] = &["x86_64", "aarch64"];

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

/// Whether `g` is one of `GPUS` for `family`: its name has the word and its compute capability is the line's.
pub fn listed(g: &Gpu, family: &str) -> bool {
    on_list(GPUS, g, family)
}

/// Whether `g` is a line of `table` for `family`.
fn on_list(table: &[Entry], g: &Gpu, family: &str) -> bool {
    table.iter().any(|(word, cc, families)| *cc == g.cc && has_word(&g.name, word) && families.contains(&family))
}

/// Whether `g` is a line of `table` for some family.
fn on_gpu(table: &[Entry], g: &Gpu) -> bool {
    table.iter().any(|(word, cc, _)| *cc == g.cc && has_word(&g.name, word))
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
pub fn gpu_refusal(g: &Gpu, arch: &str, asked: bool, family: &str) -> Option<Refusal> {
    refusal_on(GPUS, ARCHS, g, arch, asked, family)
}

fn refusal_on(table: &[Entry], archs: &[&str], g: &Gpu, arch: &str, asked: bool, family: &str) -> Option<Refusal> {
    let (major, minor) = g.cc;
    let mut words: Vec<&str> = table.iter().map(|(w, _, _)| *w).collect();
    words.dedup();
    if table.is_empty() && !asked {
        // (no GPU is measured yet: the speed bar is the whole answer, whatever the GPU, its driver or its computer)
        return Some(cannot(not_measured(&g.name, &words), unmeasured_fix()));
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
        let what = if ARCHS_BUILT.contains(&arch) {
            format!("this engine builds and passes its unit tests on {arch} computers (CI, without a GPU), but has not run a model on one yet ({})", g.name)
        } else {
            format!("this engine has not run on {arch} computers yet ({})", g.name)
        };
        return Some(cannot(
            what,
            instead(
                &format!("It runs on {} computers so far; vLLM runs the model here", archs.join(" and ")),
                &format!("It runs on {} computers so far; {FULL_TOOL}", archs.join(" and ")),
            ),
        ));
    }
    if !asked && !table.iter().any(|(_, cc, _)| *cc == g.cc) {
        return Some(cannot(
            format!("this engine has not run on {} (compute capability {major}.{minor}) yet: it runs on the {}", g.name, joined(&words)),
            unmeasured_fix(),
        ));
    }
    if g.driver < driver_min(g.cc) {
        return Some(cannot(
            format!("the NVIDIA driver here runs CUDA {}.{}, and this engine's kernels need {}", g.driver / 1000, g.driver % 1000 / 10, driver_need(g.cc)),
            driver_fix(g.driver, g.cc, "run the model with vLLM"),
        ));
    }
    if !asked && !on_list(table, g, family) {
        if on_gpu(table, g) {
            // (the GPU is measured, for other families: the speed bar is per family, as a family that serves one request at a time loses to vLLM at several)
            let families: Vec<&str> = table.iter().filter(|(w, cc, _)| *cc == g.cc && has_word(&g.name, w)).flat_map(|(_, _, f)| f.iter().copied()).collect();
            return Some(cannot(
                format!("{} is a GPU the engine is measured at least as fast as vLLM on for {} models only, and this is not one", g.name, joined(&families)),
                unmeasured_fix(),
            ));
        }
        return Some(cannot(not_measured(&g.name, &words), unmeasured_fix()));
    }
    None
}

/// What to do on a GPU the engine is not measured on: vLLM runs the model where it is installed; else the engine can be asked for by name (GLYD_ENGINE=1: it runs where it can), or the whole tool installed.
fn unmeasured_fix() -> String {
    instead(
        "vLLM runs the model on this GPU",
        &format!("GLYD_ENGINE=1 glyd run MODEL uses the engine here all the same (it is not measured on this GPU yet), or {FULL_TOOL}"),
    )
}

/// What a GPU under Windows' WDDM needs saying: it can be given more memory than it has, as system memory the driver pages in without an error, and the run is then many times slower than
/// the plan says. `free` and `total`: the GPU's memory now and in all, in bytes. Returns what is found and the fix: the driver's setting that makes an allocation that does not fit fail.
pub fn wddm_note(name: &str, free: u64, total: u64) -> (String, String) {
    (
        format!(
            "{name} runs under Windows' WDDM: an allocation past its {} free (of {}) is given system memory without an error, and a run that does that is many times slower than the plan says",
            gb(free),
            gb(total)
        ),
        "Set NVIDIA Control Panel, Manage 3D settings, CUDA - Sysmem Fallback Policy, to 'Prefer No Sysmem Fallback' (driver 536 or newer): an allocation that does not fit then fails with the memory said; close programs that hold GPU memory (the browser, games) before a large model".to_string(),
    )
}

/// The exact update for a driver below `driver_min(cc)` (`driver`: the CUDA version it reports, 12080 for 12.8): the series to install and how, and where vLLM is installed the other way out.
pub fn driver_fix(driver: i32, cc: (i32, i32), with_vllm: &str) -> String {
    let have = format!("the driver here runs CUDA {}.{}", driver / 1000, driver % 1000 / 10);
    let series = driver_series(driver_min(cc));
    let update = format!(
        "{have}; the engine's kernels need {}: on Ubuntu `sudo apt install nvidia-driver-{series}` (or `sudo ubuntu-drivers install`) and reboot, elsewhere the installer at https://www.nvidia.com/drivers",
        driver_need(cc)
    );
    instead(&format!("{update}, or {with_vllm}"), &update)
}

/// What is said of a GPU that is not one the engine is used on: it is not measured at least as fast as vLLM there (the speed bar), and where it is, on which GPUs.
fn not_measured(gpu: &str, words: &[&str]) -> String {
    let on = if words.is_empty() { String::new() } else { format!(" ({})", joined(words)) };
    format!("{gpu} is not a GPU the engine is measured at least as fast as vLLM on{on}")
}

/// Why the model's shape does not fit the engine's KV cache, or None: its attention heads take 128 values, and up to `MAX_GROUP` query heads share a KV head.
pub fn config_refusal(name: &str, c: &Config) -> Option<Refusal> {
    if c.head_dim != 128 {
        return Some(cannot(
            format!("{name}: its attention heads have {} values, and the engine's KV cache takes 128", c.head_dim),
            instead("Other models run with vLLM", "Pick a Qwen3 model, which this engine runs: glyd run Qwen/Qwen3-8B"),
        ));
    }
    if c.kv_heads == 0 || !c.heads.is_multiple_of(c.kv_heads) || c.heads / c.kv_heads > MAX_GROUP {
        return Some(cannot(
            format!("{name} has {} query heads to {} KV heads, and the engine takes up to {MAX_GROUP} to one", c.heads, c.kv_heads),
            instead("Other models run with vLLM", "Pick a Qwen3 model, which this engine runs: glyd run Qwen/Qwen3-8B"),
        ));
    }
    None
}

/// Why `ctx` tokens of context are more than the KV cache's tables cover, or None.
pub fn context_refusal(name: &str, ctx: usize) -> Option<Refusal> {
    (ctx > MAX_CONTEXT).then(|| {
        cannot(
            format!("{name} with {ctx} tokens of context: the lossless KV cache's tables cover up to {MAX_CONTEXT}"),
            instead(
                &format!("Ask for up to {MAX_CONTEXT}: --context {MAX_CONTEXT}, or run the model with vLLM"),
                &format!("Ask for up to {MAX_CONTEXT}: --context {MAX_CONTEXT}"),
            ),
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
        return short(free, download + pack).map(|f| {
            cannot(format!("the disk has {} left, and the first run of {name} writes {} (the download and the packed copy)", gb(f), gb(download + pack)), fix)
        });
    }
    if let Some(f) = short(models, download) {
        return Some(cannot(format!("the disk holding the downloaded models has {} left, and {name} takes {} to download", gb(f), gb(download)), fix));
    }
    short(cache, pack).map(|f| {
        cannot(
            format!("the disk holding the engine's cache has {} left, and the packed copy of {name} takes {}", gb(f), gb(pack)),
            "Free some space, or put the cache on another drive: GLYD_CACHE=/path/on/that/drive",
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lines of the kind `GPUS` gets once a GPU is measured, to test every rule with: the shipped list is a decision of the measurements (it is empty until they are made).
    const TABLE: &[Entry] = &[("A100", (8, 0), &["Qwen3"]), ("L4", (8, 9), &["Qwen3"]), ("H100", (9, 0), &["Qwen3"]), ("GH200", (9, 0), &["Qwen3"])];

    fn gpu(name: &str, cc: (i32, i32), free_gib: u64, driver: i32) -> Gpu {
        Gpu { name: name.to_string(), cc, sms: 80, total: free_gib << 30, free: free_gib << 30, driver, count: 1, ordinal: 0 }
    }

    fn qwen3_8b() -> Config {
        Config {
            hidden: 4096,
            layers: 36,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            inter: 12288,
            vocab: 151936,
            eps: 1e-6,
            rope_theta: 1e6,
            max_pos: 40960,
            tie: false,
            eos: vec![],
        }
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
            assert!(on_list(TABLE, &named(yes, cc), "Qwen3"), "{yes}");
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
            assert!(!on_list(TABLE, &named(no, cc), "Qwen3"), "{no}");
        }
    }

    /// The list is one constant: a GPU added to it (here an RTX 4090, as the measurement of one would add it) is handed models, and said in every refusal that names the list; nothing else changes.
    #[test]
    fn a_gpu_added_to_the_list_is_one_line() {
        let with = [TABLE, &[("RTX 4090", (8, 9), &["Qwen3"])]].concat();
        let g4090 = gpu("NVIDIA GeForce RTX 4090", (8, 9), 24, 13000);
        assert!(refusal_on(TABLE, ARCHS, &g4090, "x86_64", false, "Qwen3").is_some() && refusal_on(&with, ARCHS, &g4090, "x86_64", false, "Qwen3").is_none());
        // (the neighbours of the new line stay out: a 4080 is no 4090, an L40S is not an RTX 4090)
        assert!(refusal_on(&with, ARCHS, &gpu("NVIDIA GeForce RTX 4080 SUPER", (8, 9), 16, 13000), "x86_64", false, "Qwen3").is_some());
        assert!(refusal_on(&with, ARCHS, &gpu("NVIDIA L40S", (8, 9), 44, 13000), "x86_64", false, "Qwen3")
            .unwrap()
            .what
            .contains("A100, L4, H100, GH200 or RTX 4090"));
        // and the refusals say the list the table holds, whatever it holds
        let said = refusal_on(TABLE, ARCHS, &gpu("NVIDIA B200", (10, 0), 180, 13000), "x86_64", false, "Qwen3").unwrap().what;
        assert!(said.ends_with("it runs on the A100, L4, H100 or GH200"), "{said}");
        // an aarch64 computer is planned as an x86_64 one: a GPU on the list is handed models on either, a GPU off it is refused for the bar on either; another architecture is refused for itself
        let grace = gpu("NVIDIA GH200 480GB", (9, 0), 90, 13000);
        assert!(refusal_on(TABLE, ARCHS, &grace, "aarch64", false, "Qwen3").is_none() && refusal_on(TABLE, ARCHS, &grace, "x86_64", false, "Qwen3").is_none());
        let off = gpu("NVIDIA B200", (10, 0), 180, 13000);
        assert_eq!(refusal_on(TABLE, ARCHS, &off, "aarch64", false, "Qwen3").map(|r| r.what), refusal_on(TABLE, ARCHS, &off, "x86_64", false, "Qwen3").map(|r| r.what));
        assert!(refusal_on(TABLE, ARCHS, &grace, "riscv64", false, "Qwen3").unwrap().what.contains("has not run on riscv64 computers"));
        // the shipped list: an L4 is handed models by the line `("L4", (8, 9), &["Qwen3"])`, which is all it takes
        let l4 = gpu("NVIDIA L4", (8, 9), 22, 13000);
        assert!(refusal_on(&[GPUS, &[("L4", (8, 9), &["Qwen3"])]].concat(), ARCHS, &l4, "x86_64", false, "Qwen3").is_none());
    }

    /// The speed bar: the engine is used only where it is measured at least as fast as vLLM, and `GPUS` is where that is written. Whatever it holds, a GPU it does not name is refused with the bar as the reason
    /// (and the fix is vLLM), a GPU it names has a compute capability the engine runs on, and the shipped list is no more than a decision of measurements: where it is empty, every GPU is refused first for the bar,
    /// whatever else is wrong with it, with no list to name.
    #[test]
    #[cfg(not(kernels12))] // (a driver of CUDA 12.x is below this build's minimum only where it carries CUDA 13's kernels alone)
    fn the_engine_is_used_only_where_it_is_measured_at_least_as_fast_as_vllm() {
        FORCE_VLLM.with(|f| f.set(Some(true)));
        for (word, cc, families) in GPUS {
            assert!(!families.is_empty());
            assert!(!word.is_empty() && cc.0 >= 8, "{word} {cc:?}");
        }
        let unlisted = [("NVIDIA GeForce RTX 3060 Ti", (8, 6)), ("Some Future GPU 9000", (8, 9)), ("NVIDIA GeForce RTX 5090", (12, 0))];
        for (name, cc) in unlisted {
            let g = gpu(name, cc, 24, 13000);
            if on_list(GPUS, &g, "Qwen3") {
                continue; // (a GPU the measurements added is the case that is meant)
            }
            let r = gpu_refusal(&g, "x86_64", false, "Qwen3").unwrap_or_else(|| panic!("{name} was not refused"));
            assert!(
                r.code == 3
                    && r.what.contains(name)
                    && (r.what.contains("measured at least as fast as vLLM") || r.what.contains("has not run on"))
                    && r.fix == "vLLM runs the model on this GPU",
                "{r:?}"
            );
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
            let r = refusal_on(&[], ARCHS, &g, arch, false, "Qwen3").unwrap();
            assert_eq!(
                (r.code, r.what.as_str()),
                (3, format!("{} is not a GPU the engine is measured at least as fast as vLLM on", g.name).as_str()),
                "{} {arch}",
                g.name
            );
            assert_eq!(r.fix, "vLLM runs the model on this GPU");
        }
        // and with a list, the bar is the last thing asked of a GPU that has run, and says where the engine is measured
        let r = refusal_on(TABLE, ARCHS, &gpu("NVIDIA GeForce RTX 4090", (8, 9), 24, 13000), "x86_64", false, "Qwen3").unwrap();
        assert!(r.what.ends_with("vLLM on (A100, L4, H100 or GH200)"), "{r:?}");
        // asked for by name (GLYD_ENGINE=1): no list is asked, an empty one or not, and every other check is
        for table in [&[][..], TABLE] {
            for (name, cc) in [("NVIDIA GeForce RTX 4090", (8, 9)), ("NVIDIA GeForce RTX 5090", (12, 0)), ("NVIDIA B200", (10, 0)), ("NVIDIA A10", (8, 6))] {
                assert!(refusal_on(table, ARCHS, &gpu(name, cc, 24, 13000), "x86_64", true, "Qwen3").is_none(), "{name}");
            }
            for (g, arch, said) in [
                (gpu("NVIDIA A100-SXM4-40GB MIG 1g.5gb", (8, 0), 5, 13000), "x86_64", "MIG slice"),
                (gpu("Tesla T4", (7, 5), 15, 13000), "x86_64", "older"),
                (gpu("NVIDIA GH200 480GB", (9, 0), 90, 13000), "riscv64", "riscv64"),
                (gpu("NVIDIA GeForce RTX 4090", (8, 9), 24, 12080), "x86_64", "CUDA 12.8"),
            ] {
                let r = refusal_on(table, ARCHS, &g, arch, true, "Qwen3").unwrap();
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
    #[cfg(not(kernels12))] // (a driver of CUDA 12.x is below this build's minimum only where it carries CUDA 13's kernels alone)
    fn every_reason_a_gpu_is_refused_for_and_the_order_they_are_said_in() {
        let ok = gpu("NVIDIA L4", (8, 9), 22, 13000);
        let said = |g: Gpu, arch: &str| refusal_on(TABLE, ARCHS, &g, arch, false, "Qwen3").map(|r| (r.code, r.what));
        assert!(said(ok.clone(), "x86_64").is_none());
        for (name, cc) in [("NVIDIA A100-SXM4-80GB", (8, 0)), ("NVIDIA H100 80GB HBM3", (9, 0))] {
            assert!(said(gpu(name, cc, 70, 13020), "x86_64").is_none(), "{name}");
        }
        // a slice, a card that is too old
        assert!(said(gpu("NVIDIA A100-SXM4-40GB MIG 1g.5gb", (8, 0), 5, 13000), "x86_64").unwrap().1.contains("MIG slice"));
        let old = said(gpu("Tesla T4", (7, 5), 15, 13000), "x86_64").unwrap();
        assert!(old.0 == 3 && old.1.contains("7.5") && old.1.contains("older"), "{old:?}");
        // an aarch64 computer (a GH200 is a Grace and a Hopper) is planned as an x86_64 one: what its GPU decides, nothing for the architecture; another architecture is refused for itself
        let gh200 = gpu("NVIDIA GH200 480GB", (9, 0), 90, 13000);
        assert_eq!(said(gh200.clone(), "aarch64"), said(gh200, "x86_64"));
        let other = said(gpu("NVIDIA GH200 480GB", (9, 0), 90, 13000), "riscv64").unwrap();
        assert!(other.1.contains("has not run on riscv64 computers yet") && !other.1.contains("unit tests"), "{other:?}");
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
    /// Where vLLM is not installed (the Rust install alone), no refusal sends the user to it: each says what the Rust install can do, which is a Qwen3 model, the engine by name, or the whole tool.
    #[test]
    #[cfg(not(kernels12))] // (a driver of CUDA 12.x is below this build's minimum only where it carries CUDA 13's kernels alone)
    fn a_refusal_names_vllm_only_where_it_is_installed() {
        let l4 = gpu("NVIDIA L4", (8, 9), 22, 13000);
        let old = gpu("NVIDIA L4", (8, 9), 22, 12080);
        let wide = Config { heads: 72, ..qwen3_8b() };
        let all = |installed: bool| {
            FORCE_VLLM.with(|f| f.set(Some(installed)));
            let rs = [
                refusal_on(&[], ARCHS, &l4, "x86_64", false, "Qwen3").unwrap(),
                refusal_on(TABLE, ARCHS, &gpu("NVIDIA L4", (8, 9), 22, 13000), "riscv64", true, "Qwen3").unwrap(),
                refusal_on(TABLE, ARCHS, &old, "x86_64", true, "Qwen3").unwrap(),
                config_refusal("x", &wide).unwrap(),
                context_refusal("m", 100_000).unwrap(),
            ];
            rs.map(|r| r.fix)
        };
        for fix in all(true) {
            assert!(fix.contains("vLLM"), "{fix}");
        }
        let alone = all(false);
        for fix in &alone {
            assert!(!fix.contains("vLLM") && !fix.contains("vllm"), "{fix}");
        }
        assert!(alone[0].contains("GLYD_ENGINE=1 glyd run MODEL") && alone[0].contains("install.sh"), "{}", alone[0]);
        assert!(alone[1].contains("install.sh") && alone[3].contains("Qwen/Qwen3-8B") && alone[4].contains("--context 40960"), "{alone:?}");
        // the driver's update is exact either way: what it has, what the engine needs, and the command
        for fix in [&alone[2], &all(true)[2]] {
            assert!(fix.contains("runs CUDA 12.8") && fix.contains("CUDA 13.0") && fix.contains("580") && fix.contains("nvidia-driver-580"), "{fix}");
        }
    }

    /// vLLM is found by files alone: a `vllm` program in a directory of PATH, or in the environment of the Python of a `glyd` script (its `#!` line), as `uv tool install "glyd[vllm]"` leaves it.
    #[test]
    fn vllm_is_found_beside_the_python_tool() {
        let root = std::env::temp_dir().join(format!("glyd-vllm-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let (empty, plain, tool, venv, with_prog) = (root.join("empty"), root.join("plain"), root.join("bin"), root.join("venv"), root.join("prog"));
        for d in [&empty, &plain, &tool, &venv.join("bin"), &venv.join("lib/python3.12/site-packages/vllm"), &with_prog] {
            std::fs::create_dir_all(d).unwrap();
        }
        // a glyd whose Python has vllm in its site-packages; a glyd whose Python has nothing; a vllm program on its own
        std::fs::write(tool.join("glyd"), format!("#!{}/bin/python\nimport glyd.cli\n", venv.display())).unwrap();
        let bare = root.join("bare");
        std::fs::create_dir_all(bare.join("bin")).unwrap();
        std::fs::write(plain.join("glyd"), format!("#!{}/bin/python\nimport glyd.cli\n", bare.display())).unwrap();
        std::fs::write(with_prog.join("vllm"), "#!/bin/sh\n").unwrap();
        assert!(!vllm_in(&[]) && !vllm_in(std::slice::from_ref(&empty)) && !vllm_in(std::slice::from_ref(&plain)));
        assert!(vllm_in(std::slice::from_ref(&tool)) && vllm_in(std::slice::from_ref(&with_prog)));
        assert!(vllm_in(&[empty, plain, tool]));
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// `doctor` fails on a driver below `DRIVER_MIN` and says the update; a new enough one passes.
    /// The RTX 4080 SUPER is measured (benchmarks/gpu/rtx4080s-bar-2026-11-20): it is on the list by the name the driver gives, and its neighbours, which are not measured, are not.
    /// The list is per (GPU, family): the 4080 SUPER is on it for Qwen3 and not for a family not measured there, whose refusal says which it is measured for.
    #[test]
    fn the_list_is_per_gpu_and_family() {
        FORCE_VLLM.with(|f| f.set(Some(true)));
        let g = gpu("NVIDIA GeForce RTX 4080 SUPER", (8, 9), 16, 13000);
        assert!(listed(&g, QWEN3) && !listed(&g, "Gemma4"));
        assert!(gpu_refusal(&g, "x86_64", false, QWEN3).is_none());
        let r = gpu_refusal(&g, "x86_64", false, "Gemma4").unwrap();
        assert!(r.code == 3 && r.what.contains("for Qwen3 models only") && r.fix == "vLLM runs the model on this GPU", "{r:?}");
        // asked for by name, or a lossy level: the list is not looked at
        assert!(gpu_refusal(&g, "x86_64", true, "Gemma4").is_none() && gpu_refusal(&g, "x86_64", true, ANY_FAMILY).is_none());
        let table: &[Entry] = &[("L4", (8, 9), &["Qwen3", "Gemma4"]), ("H100", (9, 0), &["Qwen3"])];
        assert!(on_list(table, &gpu("NVIDIA L4", (8, 9), 22, 13000), "Gemma4") && !on_list(table, &gpu("NVIDIA H100 80GB HBM3", (9, 0), 80, 13000), "Gemma4"));
    }

    #[test]
    fn the_rtx_4080_super_is_listed_and_its_neighbours_are_not() {
        let on = |name: &str, cc| on_list(GPUS, &gpu(name, cc, 16, 13000), "Qwen3");
        assert!(on("NVIDIA GeForce RTX 4080 SUPER", (8, 9)));
        for (no, cc) in [
            ("NVIDIA GeForce RTX 4080", (8, 9)),
            ("NVIDIA GeForce RTX 4090", (8, 9)),
            ("NVIDIA GeForce RTX 4070 SUPER", (8, 9)),
            ("NVIDIA GeForce RTX 4080 SUPER", (8, 6)),
        ] {
            assert!(!on(no, cc), "{no}");
        }
        FORCE_VLLM.with(|f| f.set(Some(true)));
        assert!(gpu_refusal(&gpu("NVIDIA GeForce RTX 4080 SUPER", (8, 9), 16, 13000), "x86_64", false, "Qwen3").is_none());
    }

    #[test]
    #[cfg(not(kernels12))] // (a driver of CUDA 12.x is below this build's minimum only where it carries CUDA 13's kernels alone)
    fn doctor_fails_a_driver_below_the_minimum() {
        FORCE_VLLM.with(|f| f.set(Some(false)));
        let said = crate::cli::driver_problem(&gpu("NVIDIA GeForce RTX 4080 SUPER", (8, 9), 16, 12080)).unwrap();
        assert!(said.contains("runs CUDA 12.8") && said.contains("driver 580 or newer") && said.contains("sudo apt install nvidia-driver-580"), "{said}");
        assert!(crate::cli::driver_problem(&gpu("x", (8, 9), 16, DRIVER_MIN)).is_none() && crate::cli::driver_problem(&gpu("x", (8, 9), 16, 13020)).is_none());
    }
    /// A build with CUDA 12's kernels runs on a driver of CUDA 12.2 (535), Blackwell on 12.8 (570); a build with CUDA 13's alone on 13.0 (580), whatever the GPU.
    #[test]
    fn the_driver_minimum_follows_the_kernels_the_build_carries() {
        for cc in [(8, 0), (8, 6), (8, 9), (9, 0)] {
            assert_eq!(driver_min_for(true, cc), 12020, "{cc:?}");
            assert_eq!(driver_min_for(false, cc), 13000, "{cc:?}");
        }
        for cc in [(10, 0), (12, 0), (12, 1)] {
            assert_eq!(driver_min_for(true, cc), 12080, "{cc:?}");
            assert_eq!(driver_min_for(false, cc), 13000, "{cc:?}");
        }
        assert_eq!(driver_need_for(true, (8, 9)), "CUDA 12.2 (driver 535 or newer)");
        assert_eq!(driver_need_for(true, (12, 0)), "CUDA 12.8 (driver 570 or newer)");
        assert_eq!(driver_need_for(false, (8, 9)), "CUDA 13.0 (driver 580 or newer)");
        for (cuda, series) in [(12000, 525), (12020, 535), (12040, 550), (12060, 550), (12080, 570), (12090, 575), (13000, 580), (13020, 580)] {
            assert_eq!(driver_series(cuda), series, "{cuda}");
        }
    }

    /// A GPU the build has no code for is told so, with the code the build carries and the way out; the driver is not the fix.
    #[test]
    fn a_gpu_with_no_code_is_not_told_to_update_the_driver() {
        let g = gpu("NVIDIA GeForce RTX 3050", (8, 6), 6, 13020);
        let said = crate::cli::no_code_for(&g);
        assert!(said.contains("this build has no code for NVIDIA GeForce RTX 3050") && said.contains("8.6") && !said.contains("update"), "{said}");
        let fix = crate::cli::no_code_fix(&g);
        assert!(fix.contains("GLYD_ENGINE_ARCHS=86") && !fix.to_lowercase().contains("driver"), "{fix}");
    }
}
