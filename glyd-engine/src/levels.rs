//! Glyd's levels and the `MODEL:level` tag, in one place: the five levels and the layout each packs weights in, the tag
//! parser every command uses, a level's name from a layout, the Hub repo a model is pre-packed in at a level, and the
//! checkpoints whose wren is exact (quantization-aware releases).
//!
//! Levels: penguin (lossless), kestrel (6.5 bits), swift (5.5), wren (4.5), hummingbird (MoE experts at 2-4 bits by use).

use crate::weights::Layout;

/// A level: its name, the layout its weights are packed in (None: penguin, lossless, the normal path), its bits per
/// weight as messages say them, whether it gives back the release's weights exactly, and the quality bar a pack at it is
/// published under (None: the lossless level, which needs none).
pub struct Level {
    pub name: &'static str,
    pub layout: Option<Layout>,
    pub bits: &'static str,
    pub lossless: bool,
    pub bar: Option<Bar>,
}

/// What a lossy pack's quality card (src/quality.rs, docs/quality.md) must show against the lossless level of the same model to be published:
/// the mean next-token KL over the fixed texts at most `kl_mean`, the best token the lossless level's at `top1` of the prefixes or more, and
/// each task's accuracy at most `task_drop` below the lossless level's. Set from the measured divergences of the levels' layouts on released
/// models (research/lossy-2026/kld_study.py: benchmarks/weights/kld-q56-2026-10-09, the cards' kl_bf16, and for hummingbird
/// benchmarks/weights/moe-quality-2026-10-10/nemotron-hummingbird), about 1.4 times the largest KL measured and three to four points under the
/// lowest top-1 agreement. Those were taken over WikiText-2 with 64 to 128 tokens a position; the suite's own texts and 20 tokens a position are
/// to be checked against them by the first cards it writes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bar {
    pub kl_mean: f64,
    pub top1: f64,
    pub task_drop: f64,
}

/// The levels a tag on MODEL names (`Qwen/Qwen3.8-27B:swift`). Wren is the four-bit layout with its plain rounding (exact on a
/// QAT checkpoint); hummingbird is a mixture of experts' routed experts at 3 and 2 bits by use (Kimi-K3, DeepSeek-V4, GLM-5.3-Flash and
/// Muse Spark from `glyd-engine pack --layout hummingbird --usage`, src/hb_pack.rs, the rest as released; Nemotron 3 Super's: src/moe_hb.rs, every
/// other matrix six-bit), from `glyd-engine pack MODEL OUT --layout hummingbird --codes DIR` or downloaded pre-packed. The
/// three-bit layout (`--layout i3`) is no level.
pub static LEVELS: [Level; 5] = [
    Level { name: "penguin", layout: None, bits: "about 10.8", lossless: true, bar: None },
    // (six-bit codes: KL 0.0048-0.0147 on the Qwen3.5 / 3.6 / 3.8 cards, top-1 96.9% on Qwen3.8-27B)
    Level { name: "kestrel", layout: Some(Layout::Int6), bits: "6.5", lossless: false, bar: Some(Bar { kl_mean: 0.02, top1: 0.94, task_drop: 0.05 }) },
    // (five-bit codes: KL 0.0139-0.0373, top-1 94.3%)
    Level { name: "swift", layout: Some(Layout::Int5), bits: "5.5", lossless: false, bar: Some(Bar { kl_mean: 0.05, top1: 0.91, task_drop: 0.075 }) },
    // (four-bit codes without calibration: KL 0.0534, top-1 88.9% on Qwen3.8-27B; a QAT checkpoint's wren is exact)
    Level { name: "wren", layout: Some(Layout::Int4), bits: "4.5", lossless: false, bar: Some(Bar { kl_mean: 0.08, top1: 0.86, task_drop: 0.10 }) },
    // (experts at 2-3 bits by use, hb40: KL 0.0831, top-1 93.96%, the record benchmarks/weights/moe-quality-2026-10-10)
    Level {
        name: "hummingbird",
        layout: Some(Layout::Hummingbird),
        bits: "2-4 on experts",
        lossless: false,
        bar: Some(Bar { kl_mean: 0.12, top1: 0.90, task_drop: 0.10 }),
    },
];

/// The names of the levels every build runs (wren too since v0.34.0: before it a release build ran wren on QAT checkpoints alone).
pub fn names_now() -> Vec<&'static str> {
    LEVELS.iter().map(|l| l.name).collect()
}

/// The level named `name`, in any case.
pub fn by_name(name: &str) -> Option<&'static Level> {
    LEVELS.iter().find(|l| l.name.eq_ignore_ascii_case(name))
}

/// MODEL and its tag: the text after the last ':' where it is all ASCII letters, lowercased (a Hub name has no ':'; in a
/// path the text after one has a '/' or more; `m:v2`, `m:4bit` carry no tag).
pub fn split_tag(model: &str) -> (&str, Option<String>) {
    match model.rsplit_once(':') {
        Some((rest, tag)) if !tag.is_empty() && tag.bytes().all(|b| b.is_ascii_alphabetic()) => (rest, Some(tag.to_ascii_lowercase())),
        _ => (model, None),
    }
}

/// Why a tag was refused, in the words a refusal shows: what is wrong, and what to do.
#[derive(Debug, PartialEq)]
pub struct TagError {
    pub what: String,
    pub fix: String,
}

/// MODEL without its tag, and the layout of its level (None: lossless). `--lossy` is `:kestrel`. A pre-packed repo's own name
/// (`glyd/NAME-swift`) is its level. Refused: a tag that is no level, and `--lossy` with another level.
pub fn parse(model: &str, lossy: bool) -> Result<(&str, Option<Layout>), TagError> {
    let (rest, tag) = split_tag(model);
    let Some(tag) = tag else {
        return Ok((model, lossy.then_some(Layout::Int6).or_else(|| repo_level(model))));
    };
    let Some(level) = by_name(&tag) else {
        return Err(TagError { what: format!(":{tag} is not a level"), fix: format!("The levels available now: {}", names_now().join(", ")) });
    };
    if lossy && level.layout != Some(Layout::Int6) {
        return Err(TagError { what: format!("--lossy is :kestrel, not :{}", level.name), fix: "Give one or the other".into() });
    }
    Ok((rest, level.layout))
}

/// A level's name from a layout (penguin for the lossless layouts; a layout that is no level, the three-bit one, by its own name).
pub fn name_of(layout: Layout) -> &'static str {
    match layout {
        Layout::Tiered | Layout::Twelve | Layout::Compact => "penguin",
        Layout::Int6 | Layout::Int6Row => "kestrel",
        Layout::Int5 | Layout::Int5Row => "swift",
        Layout::Int4 | Layout::Int4H | Layout::Int4Row => "wren",
        Layout::Hummingbird | Layout::Cb3Row | Layout::Cb2Row => "hummingbird",
        Layout::Int3 => "i3",
    }
}

/// A layout from a level's name or from glyd.json's layout word (`mma`, `i6`, `hb`, ...).
pub fn layout_of(word: &str) -> Option<Layout> {
    Some(match word {
        "penguin" | "mma" => Layout::Tiered,
        "mma12" => Layout::Twelve,
        "kestrel" | "i6" => Layout::Int6,
        "swift" | "i5" => Layout::Int5,
        "wren" | "i4" => Layout::Int4,
        "i4x" => Layout::Int4H,
        "i3" => Layout::Int3,
        "hummingbird" | "hb" => Layout::Hummingbird,
        _ => return None,
    })
}

/// A model's name on the Hub: the text after its org (`Qwen/Qwen3.8-27B` → `Qwen3.8-27B`).
pub fn short_name(model: &str) -> String {
    model.trim_end_matches('/').rsplit('/').next().unwrap_or(model).to_string()
}

/// A model's Hub name and the shorter one its pre-packed repos have (glyd-gpu-py's engine.REPO_NAMES is the same table).
/// Gemma 4: the instruction-tuned model is the plain name (its wren packed from Google's QAT checkpoint), the base model "-base".
const REPO_NAMES: [(&str, &str); 7] = [
    ("NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "nemotron-3"),
    ("gemma-4-31B-it-qat-q4_0-unquantized", "gemma-4-31B"),
    ("gemma-4-31B-it", "gemma-4-31B"),
    ("gemma-4-31B", "gemma-4-31B-base"),
    ("gemma-4-12B-it-qat-q4_0-unquantized", "gemma-4-12B"),
    ("gemma-4-12B-it", "gemma-4-12B"),
    ("gemma-4-12B", "gemma-4-12B-base"),
];

/// The Hub repo of MODEL (`Qwen/Qwen3.8-27B`) at `level` (`swift`) packed ahead: `glyd/Qwen3.8-27B-swift`, under a shorter name
/// where `REPO_NAMES` gives one (`glyd/nemotron-3-hummingbird`). A pre-packed repo's own name is its repo.
pub fn packed_repo(model: &str, level: &str) -> String {
    if model.strip_prefix("glyd/").is_some_and(|n| !n.contains('/') && n.ends_with(&format!("-{level}"))) {
        return model.to_string();
    }
    let name = short_name(model);
    let name = REPO_NAMES.iter().find(|(long, _)| *long == name).map_or(name.as_str(), |(_, short)| short);
    format!("glyd/{name}-{level}")
}

/// The layout of a pre-packed repo given by its own name (`glyd/Qwen3.8-27B-swift`), where MODEL is one.
pub fn repo_level(model: &str) -> Option<Layout> {
    let name = model.strip_prefix("glyd/").filter(|n| !n.contains('/'))?;
    LEVELS.iter().find(|l| name.ends_with(&format!("-{}", l.name))).and_then(|l| l.layout)
}

/// The models whose wren repo is packed from Google's QAT checkpoint of them.
pub const QAT_WREN: [&str; 2] = ["gemma-4-31B-it", "gemma-4-12B-it"];

/// Whether MODEL names a checkpoint trained to four bits (Google's "-qat-" releases), or a model whose wren repo is packed from
/// one: its wren is the exact four-bit layout (i4x).
pub fn is_qat(model: &str) -> bool {
    let name = short_name(model);
    name.to_ascii_lowercase().contains("-qat") || QAT_WREN.contains(&name.as_str())
}

/// The layout `layout` takes for MODEL: wren on a QAT checkpoint is the exact four-bit layout.
pub fn layout_for(model: &str, layout: Layout) -> Layout {
    if layout == Layout::Int4 && is_qat(model) {
        Layout::Int4H
    } else {
        layout
    }
}

/// The layout `ship` packs MODEL at for `level` (penguin, kestrel, swift or wren), else None (hummingbird comes from a calibration).
pub fn ship_layout(model: &str, level: &str) -> Option<Layout> {
    match by_name(level)?.layout {
        None => Some(Layout::Tiered),
        Some(Layout::Hummingbird) => None,
        Some(l) => Some(layout_for(model, l)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_names_each_level_once_and_its_layout_names_it_back() {
        for l in &LEVELS {
            assert!(std::ptr::eq(by_name(l.name).unwrap(), l));
            assert_eq!(by_name(&l.name.to_ascii_uppercase()).unwrap().name, l.name);
            assert_eq!(l.layout.map_or("penguin", name_of), l.name);
            assert_eq!(layout_of(l.name), Some(l.layout.unwrap_or(Layout::Tiered)));
        }
        assert_eq!(LEVELS.iter().filter(|l| l.lossless).count(), 1);
        // every lossy level has a quality bar, the lossless one none; a level of fewer bits is held to a looser one
        assert!(LEVELS.iter().all(|l| l.bar.is_some() != l.lossless));
        let bars: Vec<Bar> = ["kestrel", "swift", "wren"].iter().map(|n| by_name(n).unwrap().bar.unwrap()).collect();
        assert!(bars.windows(2).all(|w| w[0].kl_mean < w[1].kl_mean && w[0].top1 > w[1].top1 && w[0].task_drop <= w[1].task_drop));
        assert!(by_name("sparrow").is_none() && layout_of("x").is_none());
        assert_eq!((name_of(Layout::Twelve), name_of(Layout::Int4H), name_of(Layout::Cb2Row), name_of(Layout::Int3)), ("penguin", "wren", "hummingbird", "i3"));
    }

    #[test]
    fn a_tag_is_letters_after_the_last_colon() {
        assert_eq!(split_tag("Qwen/Qwen3.5-9B:Swift"), ("Qwen/Qwen3.5-9B", Some("swift".to_string())));
        assert_eq!(split_tag("a:b:kestrel"), ("a:b", Some("kestrel".to_string())));
        for plain in ["/a/b:c/d", "glyd/Qwen3.5-9B-swift", "m:v2", "m:4bit", "m:ké", "C:\\models\\Qwen", "m:", ":"] {
            assert_eq!(split_tag(plain), (plain, None), "{plain}");
        }
    }

    #[test]
    fn parse_refuses_what_is_no_level_and_lossy_with_another() {
        let m = "Qwen/Qwen3.8-27B";
        assert_eq!(parse(&format!("{m}:SWIFT"), false).unwrap(), (m, Some(Layout::Int5)));
        assert_eq!(parse(m, true).unwrap(), (m, Some(Layout::Int6)));
        assert_eq!(parse("glyd/Qwen3.5-9B-swift", false).unwrap(), ("glyd/Qwen3.5-9B-swift", Some(Layout::Int5)));
        assert_eq!(parse(&format!("{m}:falcon"), false).unwrap_err().what, ":falcon is not a level");
        assert_eq!(parse(&format!("{m}:swift"), true).unwrap_err().what, "--lossy is :kestrel, not :swift");
        // wren is exact on a QAT checkpoint in any build
        assert_eq!(parse("google/gemma-4-31B-it:wren", false).unwrap(), ("google/gemma-4-31B-it", Some(Layout::Int4)));
        assert_eq!(layout_for("google/gemma-4-31B-it", Layout::Int4), Layout::Int4H);
    }

    #[test]
    fn packed_repos_and_ship_layouts() {
        assert_eq!(packed_repo("Qwen/Qwen3.5-9B", "swift"), "glyd/Qwen3.5-9B-swift");
        assert_eq!(packed_repo("glyd/gemma-4-12B-wren", "wren"), "glyd/gemma-4-12B-wren");
        assert_eq!(packed_repo("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "hummingbird"), "glyd/nemotron-3-hummingbird");
        assert_eq!(repo_level("glyd/Qwen3.8-27B-kestrel"), Some(Layout::Int6));
        assert_eq!(repo_level("glyd/a/b-swift"), None);
        assert_eq!(
            (ship_layout("m", "penguin"), ship_layout("m", "hummingbird"), ship_layout("google/gemma-4-12B-it", "wren")),
            (Some(Layout::Tiered), None, Some(Layout::Int4H))
        );
    }
}
