//! P4b: a family named with no size picks the size for the machine, as `glyd fit` predicts: for each preset machine, the size `glyd-engine fit FAMILY --machine PRESET` takes is the largest whose own plan
//! (`fit FAMILY:SIZE --machine PRESET --json`) reaches the floors (10 tokens a second, 8,192 tokens of context). The family is made here: four sizes of a Gemma 4 config (the 12B's fixture, widened and narrowed), in a
//! Hub cache of config.json files alone, so nothing is downloaded and no GPU is needed. A size or a level given is taken as it is.

use std::path::Path;
use std::process::Command;

const PRESETS: [&str; 6] = ["rtx4060-laptop", "rtx4080s", "rtx4090", "a6000", "a100-80g", "h100"];
const SIZES: [(&str, &str, f64); 4] = [("4b", "t/gemma-s", 0.6), ("12b", "t/gemma-m", 1.0), ("31b", "t/gemma-l", 1.6), ("70b", "t/gemma-xl", 2.4)];

fn world(test: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("glyd-names-by-plan-{}-{test}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    let base: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gemma4_12b_config.json")).unwrap()).unwrap();
    let hash = "0123456789abcdef0123456789abcdef01234567";
    for (_, repo, k) in SIZES {
        let mut c = base.clone();
        for key in ["hidden_size", "intermediate_size"] {
            let v = c["text_config"][key].as_f64().unwrap();
            c["text_config"][key] = serde_json::json!(((v * k) / 128.0).round() as u64 * 128);
        }
        let r = d.join("hub").join(format!("models--{}", repo.replace('/', "--")));
        std::fs::create_dir_all(r.join("refs")).unwrap();
        std::fs::create_dir_all(r.join("snapshots").join(hash)).unwrap();
        std::fs::write(r.join("refs/main"), hash).unwrap();
        std::fs::write(r.join("snapshots").join(hash).join("config.json"), c.to_string()).unwrap();
    }
    let models: Vec<_> = SIZES.iter().map(|(s, r, _)| serde_json::json!({"size": s, "repo": r, "default": *s == "12b"})).collect();
    std::fs::write(
        d.join("models.json"),
        serde_json::json!({"version": 1, "updated": "2026-11-21", "families": [{"name": "gemmatest", "models": models}]}).to_string(),
    )
    .unwrap();
    d
}

fn glyd(d: &Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_glyd-engine"))
        .args(args)
        .env("GLYD_MODELS", d.join("models.json"))
        .env("HF_HUB_CACHE", d.join("hub"))
        .env("HF_HUB_OFFLINE", "1")
        .env("GLYD_CACHE", d.join("cache"))
        .env("GLYD_NO_REFRESH", "1")
        .env("GLYD_NO_PROBE", "1")
        .output()
        .unwrap()
}

/// (the size the line says, its stderr) for `fit gemmatest ARGS`.
fn picked(d: &Path, preset: &str, extra: &[&str]) -> (String, String) {
    let mut args = vec!["fit", "gemmatest", "--machine", preset];
    args.extend_from_slice(extra);
    let o = glyd(d, &args);
    let err = String::from_utf8_lossy(&o.stderr).into_owned();
    let size = err.split("gemmatest is gemmatest:").nth(1).and_then(|s| s.split_whitespace().next()).unwrap_or("").to_string();
    (size, err)
}

/// What `fit gemmatest:SIZE` predicts: (tokens a second, context), or None where it does not fit.
fn predicted(d: &Path, preset: &str, size: &str) -> Option<(f64, u64)> {
    let o = glyd(d, &["fit", &format!("gemmatest:{size}"), "--machine", preset, "--json"]);
    if !o.status.success() {
        return None; // (a size `run` would refuse)
    }
    let v: serde_json::Value = serde_json::from_slice(&o.stdout).ok()?;
    Some((v.get("tokens_per_second")?.as_f64()?, v.get("context")?.as_u64()?))
}

#[test]
fn a_family_picks_the_size_fit_predicts_on_each_machine() {
    let d = world("sizes");
    let mut seen = std::collections::BTreeSet::new();
    for preset in PRESETS {
        let (size, err) = picked(&d, preset, &[]);
        let want =
            SIZES.iter().rev().map(|(s, _, _)| *s).find(|s| predicted(&d, preset, s).is_some_and(|(tok, ctx)| tok >= 10.0 && ctx >= 8192)).or_else(|| {
                SIZES
                    .iter()
                    .map(|(s, _, _)| *s)
                    .filter_map(|s| predicted(&d, preset, s).map(|p| (s, p.0)))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|x| x.0)
            });
        assert_eq!(Some(size.as_str()).filter(|s| !s.is_empty()), want, "{preset}: {err}");
        if size.is_empty() {
            // (no size is one `run` takes here: a refusal that says why for each)
            assert!(err.contains("no size fits"), "{preset}: {err}");
            continue;
        }
        // one line says it, with the way to choose otherwise
        let line = err.lines().find(|l| l.contains("gemmatest is gemmatest:")).unwrap();
        assert!(
            line.contains(&format!("({})", SIZES.iter().find(|s| s.0 == size).unwrap().1))
                && line.contains("To choose another: gemmatest:SIZE[:LEVEL]")
                && line.contains(preset_gpu(preset)),
            "{line}"
        );
        seen.insert(size);
    }
    // the machines differ: a bigger one takes a bigger size
    assert!(seen.len() >= 2, "every machine took the same size: {seen:?}");
    let _ = std::fs::remove_dir_all(&d);
}

fn preset_gpu(preset: &str) -> &'static str {
    match preset {
        "rtx4060-laptop" => "RTX 4060",
        "rtx4080s" => "RTX 4080",
        "rtx4090" => "RTX 4090",
        "a6000" => "A6000",
        "a100-80g" => "A100",
        _ => "H100",
    }
}

#[test]
fn a_size_or_a_level_given_is_taken_as_it_is() {
    let d = world("given");
    // a size is not chosen: no line of the choice, the model is that size
    let o = glyd(&d, &["fit", "gemmatest:70b", "--machine", "h100"]);
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(err.contains("gemmatest:70b is t/gemma-xl") && !err.contains("To choose another"), "{err}");
    // a size and a level: both
    let o = glyd(&d, &["fit", "gemmatest:12b:kestrel", "--machine", "h100"]);
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(err.contains("gemmatest:12b:kestrel is t/gemma-m at level kestrel"), "{err}");
    // a level alone is asked of every size: the pick is made among the sizes at that level
    let (size, err) = picked(&d, "rtx4080s", &[]);
    let (size_k, err_k) = {
        let o = glyd(&d, &["fit", "gemmatest:kestrel", "--machine", "rtx4080s"]);
        let err = String::from_utf8_lossy(&o.stderr).into_owned();
        (err.split("gemmatest is gemmatest:").nth(1).and_then(|s| s.split_whitespace().next()).unwrap_or("").to_string(), err)
    };
    assert!(!size.is_empty() && !size_k.is_empty(), "{err} {err_k}");
    assert!(err_k.contains("at kestrel"), "{err_k}");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn a_machine_that_fits_none_says_why_for_each_size() {
    let d = world("none");
    let o = glyd(&d, &["fit", "gemmatest", "--machine", "rtx4060-laptop", "--context", "1048576"]);
    assert!(!o.status.success());
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(err.contains("gemmatest:") && (err.contains("no size fits") || err.contains("takes at most")), "{err}");
    let _ = std::fs::remove_dir_all(&d);
}

/// The Qwen3.5 family as the manifest has it (models.json), each size's released config.json (tests/fixtures/qwen35_configs.json) in a Hub cache of config files alone.
fn qwen35_world(test: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("glyd-names-by-plan-{}-{test}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    let configs: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_configs.json")).unwrap()).unwrap();
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("models.json")).unwrap()).unwrap();
    let family = manifest["families"].as_array().unwrap().iter().find(|f| f["name"] == "qwen3.5").unwrap().clone();
    let hash = "0123456789abcdef0123456789abcdef01234567";
    for m in family["models"].as_array().unwrap() {
        let repo = m["repo"].as_str().unwrap();
        let r = d.join("hub").join(format!("models--{}", repo.replace('/', "--")));
        std::fs::create_dir_all(r.join("refs")).unwrap();
        std::fs::create_dir_all(r.join("snapshots").join(hash)).unwrap();
        std::fs::write(r.join("refs/main"), hash).unwrap();
        std::fs::write(r.join("snapshots").join(hash).join("config.json"), configs[repo].to_string()).unwrap();
    }
    std::fs::write(d.join("models.json"), serde_json::json!({"version": 1, "updated": "2026-11-21", "families": [family]}).to_string()).unwrap();
    d
}

/// P4b's acceptance: `glyd run qwen3.5` on each preset machine takes the size `glyd fit` predicts: the largest whose own plan (`fit qwen3.5:SIZE --machine PRESET
/// --json`) reaches 10 tokens a second and 8,192 tokens of context, else the fastest that fits, else none (a refusal that says why for each size). Every size of the
/// family is sized by the planner's one sizing (src/plan.rs `sizing`, P2f), at the level the plan takes, the mixture of experts this engine does not run refused.
#[test]
fn run_qwen35_takes_the_size_fit_predicts_on_each_preset() {
    let d = qwen35_world("qwen35");
    let sizes = ["0.8b", "2b", "4b", "9b", "27b", "35b-a3b"];
    let presets = [
        "rtx4060-laptop",
        "rtx4080s",
        "rtx4090",
        "2x3090",
        "l4",
        "a6000",
        "a100-80g",
        "h100",
        "8xh100",
        "rx7900xtx",
        "m4max-64g",
        "dgx-spark",
        "strix-halo-128g",
    ];
    let mut seen = std::collections::BTreeSet::new();
    for preset in presets {
        let o = glyd(&d, &["run", "qwen3.5", "--machine", preset]);
        let err = String::from_utf8_lossy(&o.stderr).into_owned();
        let size = err.split("qwen3.5 is qwen3.5:").nth(1).and_then(|s| s.split_whitespace().next()).unwrap_or("").to_string();
        let predicted = |s: &str| -> Option<(f64, u64)> {
            let o = glyd(&d, &["fit", &format!("qwen3.5:{s}"), "--machine", preset, "--json"]);
            if !o.status.success() {
                return None;
            }
            let v: serde_json::Value = serde_json::from_slice(&o.stdout).ok()?;
            Some((v["tokens_per_second"].as_f64()?, v["context"].as_u64()?))
        };
        let plans: Vec<(&str, Option<(f64, u64)>)> = sizes.iter().map(|s| (*s, predicted(s))).collect();
        let want = plans
            .iter()
            .rev()
            .find(|(_, p)| p.is_some_and(|(tok, ctx)| tok >= 10.0 && ctx >= 8192))
            .map(|x| x.0)
            .or_else(|| plans.iter().filter_map(|(s, p)| p.map(|p| (*s, p.0))).max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).map(|x| x.0));
        assert_eq!(Some(size.as_str()).filter(|s| !s.is_empty()), want, "{preset}: {plans:?}\n{err}");
        match want {
            Some(s) => {
                seen.insert(s);
            }
            None => assert!(err.contains("no size fits"), "{preset}: {err}"),
        }
        // (the mixture of experts is not one this engine runs: never picked)
        assert!(plans.iter().any(|(s, p)| *s == "35b-a3b" && p.is_none()), "{preset}: {plans:?}");
    }
    // (machines of different memory take different sizes)
    assert!(seen.len() >= 2, "{seen:?}");
    let _ = std::fs::remove_dir_all(&d);
}

/// R14: `fit` shows each level's quality against the lossless level, from the level's card (cards/NAME-LEVEL.json), and says where none was measured.
#[test]
fn fit_shows_the_quality_of_each_level() {
    let d = qwen35_world("quality");
    let o = glyd(&d, &["fit", "qwen3.5:9b", "--machine", "h100"]);
    let out = String::from_utf8_lossy(&o.stdout);
    assert!(out.contains("Quality against the lossless level"), "{out}");
    assert!(out.contains("KL 0.0373 from bf16") && out.contains("cards/Qwen3.5-9B-swift.json"), "{out}");
    assert!(out.contains("wren         not measured"), "{out}");
    let o = glyd(&d, &["fit", "qwen3.5:9b", "--machine", "h100", "--json"]);
    let v: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(v["quality"]["swift"]["source"], "cards/Qwen3.5-9B-swift.json", "{v}");
    assert!(v["quality"]["wren"].is_null() && v["quality"].as_object().unwrap().contains_key("wren"), "{v}");
    let _ = std::fs::remove_dir_all(&d);
}
