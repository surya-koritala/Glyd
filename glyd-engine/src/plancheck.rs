//! `glyd-engine plancheck`: the planner (src/plan.rs) against what this machine measures, as a gate, and the records that set each GPU class's figures.
//!
//!   glyd-engine plancheck [--models M,M,...] [--moe MODEL [--expert-cache GIB]] [--loads 1k:1,4k:1,8k:1,1k:8] [--users 8] [--runs 3] [--out DIR] [--records FILE]
//!                         [--predict-only]
//!       1. predicts every case before anything is measured and saves it (DIR/predictions.json): for each model the plan `serve MODEL --users N --no-draft` loads
//!          (level, placement, context), and for each load decode tokens/s a user (at its users and positions), prompt tokens/s, the first token's time, and the
//!          GPU memory the plan takes; and a case at 85% of the plan's context (the context it allows, filled);
//!       2. measures each model's loads with `bench` (one server; its peak GPU memory from the driver, `nvidia-smi`, not the plan);
//!       3. prints and saves each case's errors (DIR/results.json) and the gate: decode and prompt speed within 15% of the prediction, the GPU memory the server took
//!          never above the plan's, and no failed request (an out-of-memory) at the plan's context; exit 0 where every case passes, 1 where one fails;
//!       4. appends a calibration record a case (`glyd-calib/1`: the GPU, its class, its driver and rates, the case, predicted and measured) to DIR/records.jsonl and
//!          FILE (the cache's calibration/records.jsonl): collected from machines, they set each class's figures (`plan::fit_consts`, leave-one-class-out).
//!   glyd-engine plancheck --score [FILE ...]
//!       each class's cases (the shipped records and FILEs') predicted from the figures the other classes' records fit: the errors class by class.

use crate::args::Args;
use crate::plan::{self, Machine};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

const MODELS: &str = "Qwen/Qwen3.5-0.8B,google/gemma-4-12B-it:wren,google/gemma-4-12B-it:kestrel";
const LOADS: &str = "1k:1,4k:1,8k:1,1k:8";
/// The gate: a speed within this share of the prediction.
pub const GATE: f64 = 0.15;
/// The tokens a bench reply makes (its `--max-tokens`): half of them is the decode's mean position past the prompt.
const MADE: usize = 128;

pub fn cmd(a: &Args) -> i32 {
    match if a.has("score") { score(a).map(|_| true) } else { run(a) } {
        Ok(true) => 0,
        Ok(false) => 1,
        Err(e) => {
            eprintln!("plancheck: {e}");
            2
        }
    }
}

/// A load's prompt tokens as bench makes them (`1k` 1,024; `27852t` 27,852).
fn nominal(kind: &str) -> usize {
    match kind.strip_suffix('k').and_then(|n| n.parse::<usize>().ok()) {
        Some(k) => k * 1024,
        None => kind.strip_suffix('t').and_then(|n| n.parse().ok()).unwrap_or(64),
    }
}

fn loads_of(text: &str) -> Result<Vec<(String, usize)>, String> {
    text.split(',')
        .map(|l| {
            let (k, n) = l.split_once(':').ok_or_else(|| format!("--loads {l}: KIND:USERS (1k:1)"))?;
            Ok((k.to_string(), n.parse().map_err(|_| format!("--loads {l}: USERS is a number"))?))
        })
        .collect()
}

fn slug(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '-' { c } else { '_' }).collect()
}

fn mib(b: u64) -> u64 {
    b >> 20
}

/// One model's plan and its cases' predictions, with what a record needs of the plan (its shape, the machine, the GPU, the plan).
struct Predicted {
    head: Value,
    cases: Vec<Value>,
    sh: plan::Shape,
    machine: Machine,
    main: usize,
    lp: plan::LoadPlan,
}

/// The predictions of one model's cases, from the plan `serve` would load.
fn predict(model: &str, a: &Args, users: usize, loads: &[(String, usize)]) -> Result<Predicted, String> {
    let mut words = vec!["serve".to_string(), model.to_string(), "--users".to_string(), users.to_string(), "--no-draft".to_string()];
    // (`--machine PRESET` with --predict-only: the predictions for a machine one does not have)
    for f in ["expert-cache", "host-memory", "machine"] {
        if let Some(v) = a.flag(f) {
            words.extend([format!("--{f}"), v.to_string()]);
        }
    }
    let f = plan::planned(&Args::from(words)).map_err(|r| format!("{model}: {}", r.what))?;
    if let Some(r) = &f.refusal {
        return Err(format!("{model}: {}", r.what));
    }
    let c = f.plan.chosen.as_ref().ok_or_else(|| format!("{model}: no plan"))?;
    let lp = c.load.clone().ok_or_else(|| format!("{model}: its path is not sized by its loader (no plan to load)"))?;
    let main = (0..f.machine.devices.len()).min_by_key(|&i| (std::cmp::Reverse(f.machine.devices[i].free), i)).ok_or("no GPU")?;
    let d = &f.machine.devices[main];
    let band = d.band();
    let in_vram = lp.experts.is_none() && lp.spread.as_ref().is_none_or(|s| s.host_bytes == 0 && s.dense == 0);
    let prefill = if in_vram { plan::prefill_of(&f.shape, d).map(|p| p * d.observed_prefill.unwrap_or(1.0)) } else { None };
    // (the context the plan allows, filled to 85%: no request may fail there)
    let fill = format!("{}t", lp.context * 85 / 100);
    let placement = match &c.placement {
        Some(plan::Placement::Gpu(i)) => format!("gpu {i}"),
        Some(plan::Placement::Split(g)) => format!("split {g:?}"),
        Some(plan::Placement::ExpertsInRam { cache, .. }) => format!("experts in RAM, {} MiB of them on the GPU", mib(*cache)),
        None => "none".into(),
    };
    let head = json!({
        "model": model, "level": lp.level, "placement": placement, "context": lp.context, "plan_users": lp.users, "chunk": lp.chunk,
        "planned_use_mib": mib(lp.gpu_total() - lp.reserve), "planned_total_mib": mib(lp.gpu_total()), "gpu": d.name, "rate_gbs": d.rate() / 1e9,
        "measured_rate": d.stream.is_some(), "tflops": d.tflops, "consts": {"eff": d.consts.eff, "layer_s": d.consts.layer_s, "prompt_eff": d.consts.prompt_eff,
        "prompt_layer_s": d.consts.prompt_layer_s}, "band": band,
    });
    // (a load whose prompt and reply the plan's context does not hold is not a case of this plan: said, not measured)
    let (cases, beyond): (Vec<(String, usize)>, Vec<(String, usize)>) =
        loads.iter().cloned().chain(std::iter::once((fill, 1))).partition(|(kind, _)| nominal(kind) + MADE <= lp.context);
    for (kind, n) in &beyond {
        eprintln!("plancheck: {model}: {kind}:{n} is beyond the plan's context of {} tokens: not a case", lp.context);
    }
    let cases = cases
        .into_iter()
        .map(|(kind, n)| {
            let prompt = nominal(&kind);
            let positions = (prompt + MADE / 2) as f64;
            let prior = plan::speed_users(&f.shape, &f.machine, main, &lp, positions, n);
            let decode = prior * band[1];
            json!({
                "model": model, "load": format!("{kind}:{n}"), "kind": kind, "users": n, "prompt_tokens": prompt, "positions": positions,
                "decode": decode, "decode_p10": prior * band[0], "decode_p90": prior * band[2], "prefill": prefill,
                // (the prompts of `n` users at once are read before the first of them has its first token; then one step)
                "ttft": prefill.map(|p| (n * prompt) as f64 / p + 1.0 / decode),
                "features": plan::features(&f.shape, &f.machine, main, &lp, positions, n),
            })
        })
        .collect();
    Ok(Predicted { head, cases, sh: f.shape, machine: f.machine, main, lp })
}

/// The bench file's cell of a load, where it ran.
fn cell<'a>(bench: &'a Value, kind: &str, n: usize) -> Option<&'a Value> {
    bench["cells"].as_array()?.iter().find(|c| c["prompt"] == kind && c["conc"].as_u64() == Some(n as u64))
}

fn err_of(pred: Option<f64>, meas: Option<f64>) -> Option<f64> {
    match (pred, meas) {
        (Some(p), Some(m)) if m > 0.0 => Some(p / m - 1.0),
        _ => None,
    }
}

fn run(a: &Args) -> Result<bool, String> {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let out = PathBuf::from(a.flag("out").map_or_else(|| format!("plancheck-{now}"), str::to_string));
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let users: usize = a.num("users", 8usize).map_err(|e| e.msg)?;
    let runs: usize = a.num("runs", 3usize).map_err(|e| e.msg)?;
    let loads = loads_of(a.flag("loads").unwrap_or(LOADS))?;
    let mut models: Vec<String> = a.flag("models").unwrap_or(MODELS).split(',').filter(|m| !m.is_empty()).map(str::to_string).collect();
    models.extend(a.flag("moe").map(str::to_string));
    // 1. every prediction, saved before anything is measured
    let mut plans: Vec<(String, Option<Predicted>)> = Vec::new();
    for m in &models {
        match predict(m, a, users, &loads) {
            Ok(p) => plans.push((m.clone(), Some(p))),
            Err(e) => {
                eprintln!("plancheck: {e}");
                plans.push((m.clone(), None));
            }
        }
    }
    // (the machine the plans were made for: this computer, its rates measured on a GPU's first use, or --machine's)
    let machine = plans.iter().find_map(|p| p.1.as_ref().map(|p| p.machine.clone())).unwrap_or_else(Machine::probe);
    let gpus = if a.has("machine") { Vec::new() } else { crate::dev::Dev::probe_all().unwrap_or_default() };
    let d = machine.devices.iter().max_by_key(|d| d.free);
    let gpu_json = json!({
        "gpu": d.map(|d| d.name.clone()), "cc": d.map(|d| [d.cc.0, d.cc.1]), "class": d.map(|d| plan::class_of(d.cc)), "driver": d.map(|d| d.driver),
        "sms": gpus.first().map(|g| g.sms), "stream_gbs": d.and_then(|d| d.stream), "ram_gbs": machine.ram_bandwidth, "tflops": d.map(|d| d.tflops),
    });
    let predictions = json!({"schema": "glyd-plancheck/1", "at": now, "machine": gpu_json,
        "models": plans.iter().map(|(m, p)| json!({"model": m, "plan": p.as_ref().map(|p| &p.head), "cases": p.as_ref().map(|p| &p.cases)})).collect::<Vec<_>>()});
    std::fs::write(out.join("predictions.json"), serde_json::to_string_pretty(&predictions).unwrap_or_default()).map_err(|e| e.to_string())?;
    println!(
        "predictions ({} cases) saved to {} before any measurement",
        plans.iter().filter_map(|p| p.1.as_ref()).map(|p| p.cases.len()).sum::<usize>(),
        out.join("predictions.json").display()
    );
    for (m, p) in &plans {
        let Some(Predicted { head: h, cases, .. }) = p else { continue };
        println!("{m}: {} level, {}, context {}, the GPU's {} MiB planned", h["level"], h["placement"], h["context"], h["planned_total_mib"]);
        for c in cases {
            println!(
                "  {:<12} decode {:>7.1} tok/s a user, prompts {:>8} tok/s, first token {:>7} s",
                c["load"].as_str().unwrap_or(""),
                c["decode"].as_f64().unwrap_or(0.0),
                c["prefill"].as_f64().map_or("-".into(), |p| format!("{p:.0}")),
                c["ttft"].as_f64().map_or("-".into(), |t| format!("{t:.3}"))
            );
        }
    }
    if a.has("predict-only") {
        return Ok(true);
    }
    // 2. each model's loads measured by bench (one server a model), 3. the errors and the gate
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let (mut results, mut records, mut pass) = (Vec::new(), Vec::new(), true);
    for (m, p) in &plans {
        let Some(p) = p else {
            pass = false;
            continue;
        };
        let (h, cases) = (&p.head, &p.cases);
        let file = out.join(format!("bench-{}.json", slug(m)));
        let all: Vec<String> = cases.iter().filter_map(|c| c["load"].as_str().map(str::to_string)).collect();
        let mut cmd = std::process::Command::new(&exe);
        cmd.args(["bench", m, "--loads", &all.join(","), "--runs", &runs.to_string(), "--warmup", "1", "--no-draft", "--users", &users.to_string()]);
        cmd.args(["--max-tokens", &MADE.to_string(), "--out"]).arg(&file);
        for f in ["expert-cache", "host-memory"] {
            if let Some(v) = a.flag(f) {
                cmd.args([format!("--{f}"), v.to_string()]);
            }
        }
        let status = cmd.status().map_err(|e| format!("bench: {e}"))?;
        let bench: Value = std::fs::read_to_string(&file).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null);
        // (the GPU memory the server took: the driver's peak during the loads less what was used before it started)
        let used = match (bench["peak_mib"].as_u64(), bench["idle_mib"].as_u64()) {
            (Some(p), Some(i)) if p > 0 => Some(p.saturating_sub(i)),
            _ => None,
        };
        let memory_ok = used.is_some_and(|u| u <= h["planned_total_mib"].as_u64().unwrap_or(0));
        for c in cases {
            let (kind, n) = (c["kind"].as_str().unwrap_or(""), c["users"].as_u64().unwrap_or(1) as usize);
            let cl = cell(&bench, kind, n);
            let med = |k: &str| cl.and_then(|c| c[k]["median"].as_f64());
            let (decode, prefill, ttft) = (med("decode_tps_per_user"), med("prefill_tps"), med("ttft_s"));
            let failed = cl.map_or(1, |c| c["errors"].as_u64().unwrap_or(0));
            let (ed, ep, et) = (err_of(c["decode"].as_f64(), decode), err_of(c["prefill"].as_f64(), prefill), err_of(c["ttft"].as_f64(), ttft));
            let speed_ok = ed.is_some_and(|e| e.abs() <= GATE) && ep.is_none_or(|e| e.abs() <= GATE);
            let ok = status.success() && failed == 0 && speed_ok && memory_ok;
            pass &= ok;
            let r = json!({
                "model": m, "load": c["load"], "level": h["level"], "context": h["context"],
                "predicted": {"decode": c["decode"], "prefill": c["prefill"], "ttft": c["ttft"], "planned_total_mib": h["planned_total_mib"], "planned_use_mib": h["planned_use_mib"]},
                "measured": {"decode": decode, "prefill": prefill, "ttft": ttft, "used_mib": used, "failed_requests": failed, "prompt_tokens": cl.map(|c| c["prompt_tokens"].clone())},
                "error": {"decode": ed, "prefill": ep, "ttft": et},
                "gate": {"speed": speed_ok, "memory": memory_ok, "no_failed_request": failed == 0 && status.success(), "pass": ok},
            });
            println!(
                "{:<34} {:<12} decode {:>7.1} / {:>7} ({:>6}) prompts ({:>6}) first token ({:>6}) memory {:>6} of {} MiB  {}",
                m,
                c["load"].as_str().unwrap_or(""),
                c["decode"].as_f64().unwrap_or(0.0),
                decode.map_or("-".into(), |x| format!("{x:.1}")),
                ed.map_or("-".into(), |e| format!("{:+.1}%", e * 100.0)),
                ep.map_or("-".into(), |e| format!("{:+.1}%", e * 100.0)),
                et.map_or("-".into(), |e| format!("{:+.1}%", e * 100.0)),
                used.map_or("-".into(), |u| u.to_string()),
                h["planned_total_mib"],
                if ok { "PASS" } else { "FAIL" }
            );
            // 4. the calibration record, its features at the positions the run had
            let positions = cl.and_then(|c| c["prompt_tokens"].as_f64()).map_or(c["positions"].as_f64().unwrap_or(0.0), |pt| pt + (MADE / 2) as f64);
            let features = plan::features(&p.sh, &p.machine, p.main, &p.lp, positions, n);
            records.push(json!({
                "schema": plan::CALIB_SCHEMA, "at": predictions["at"], "gpu": gpu_json["gpu"], "cc": gpu_json["cc"], "class": gpu_json["class"], "driver": gpu_json["driver"],
                "sms": gpu_json["sms"], "stream_gbs": h["rate_gbs"], "measured_rate": h["measured_rate"], "ram_gbs": gpu_json["ram_gbs"], "tflops": h["tflops"],
                "model": m, "level": h["level"], "load": c["load"], "users": n, "positions": positions, "features": features,
                "predicted": r["predicted"], "measured": r["measured"], "error": r["error"],
            }));
            results.push(r);
        }
    }
    std::fs::write(
        out.join("results.json"),
        serde_json::to_string_pretty(&json!({"schema": "glyd-plancheck/1", "pass": pass, "gate": GATE, "cases": results})).unwrap_or_default(),
    )
    .map_err(|e| e.to_string())?;
    let lines: String = records.iter().map(|r| r.to_string() + "\n").collect();
    let kept = a.flag("records").map_or_else(|| crate::store::cache_root().join("calibration").join("records.jsonl"), PathBuf::from);
    for f in [out.join("records.jsonl"), kept] {
        append(&f, &lines)?;
    }
    println!("{}: results in {}", if pass { "PASS" } else { "FAIL" }, out.display());
    Ok(pass)
}

fn append(f: &Path, text: &str) -> Result<(), String> {
    use std::io::Write;
    if let Some(d) = f.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    std::fs::OpenOptions::new().create(true).append(true).open(f).and_then(|mut o| o.write_all(text.as_bytes())).map_err(|e| format!("{}: {e}", f.display()))
}

/// `--score`: each class predicted from the others' figures.
fn score(a: &Args) -> Result<(), String> {
    let mut text = plan::CALIBRATION.to_string();
    for f in a.pos.iter().skip(1) {
        text += &std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?;
    }
    let records = plan::records_of(&text);
    println!("{} records; each class predicted from the figures the other classes' records fit (leave-one-class-out)", records.len());
    let line = |v: &[f64]| {
        if v.is_empty() {
            return "-".to_string();
        }
        let mean = v.iter().map(|e| e.abs()).sum::<f64>() / v.len() as f64;
        let worst = v.iter().copied().fold(0.0f64, |w, e| if e.abs() > w.abs() { e } else { w });
        format!("{} cases, mean |error| {:.1}%, worst {:+.1}%", v.len(), mean * 100.0, worst * 100.0)
    };
    for (class, dec, pre) in plan::score_classes(&records) {
        println!("{class}: decode {}; prompts {}", line(&dec), line(&pre));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A directory holding only a config.json (a fixture's), which the planner reads as a model on the disk.
    fn model_dir(root: &Path, name: &str, config: &str) -> String {
        let d = root.join(name);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("config.json"), config).unwrap();
        d.to_str().unwrap().to_string()
    }

    /// Every prediction is made and saved before anything is measured (`--predict-only` stops there): each model's plan for `serve --users 8 --no-draft` (its level,
    /// placement, context, the GPU memory it takes), each load's decode, prompt and first-token predictions, the case at 85% of the context, and no case past the
    /// plan's context.
    #[test]
    fn predictions_are_saved_before_anything_is_measured() {
        let fx = |n: &str| std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(n)).unwrap();
        let root = std::env::temp_dir().join(format!("glyd-plancheck-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let q35: Value = serde_json::from_str(&fx("qwen35_configs.json")).unwrap();
        let q08 = model_dir(&root, "Qwen3.5-0.8B", &q35["Qwen/Qwen3.5-0.8B"].to_string());
        let gemma = model_dir(&root, "gemma-4-12B", &fx("gemma4_12b_config.json"));
        let out = root.join("out");
        let models = format!("{q08},{gemma}:wren,{gemma}:kestrel");
        let a = Args::from(
            ["plancheck", "--predict-only", "--machine", "rtx4080s", "--models", &models, "--out", out.to_str().unwrap()].iter().map(|s| s.to_string()),
        );
        assert!(run(&a).unwrap());
        let v: Value = serde_json::from_str(&std::fs::read_to_string(out.join("predictions.json")).unwrap()).unwrap();
        let ms = v["models"].as_array().unwrap();
        assert_eq!(ms.len(), 3);
        for m in ms {
            let (plan, cases) = (&m["plan"], m["cases"].as_array().unwrap());
            let ctx = plan["context"].as_u64().unwrap() as usize;
            assert!(plan["planned_total_mib"].as_u64().unwrap() > 0 && plan["level"].is_string() && plan["placement"].is_string(), "{plan}");
            assert!(cases.iter().all(|c| c["decode"].as_f64().unwrap() > 0.0 && c["prompt_tokens"].as_u64().unwrap() as usize + MADE <= ctx), "{m}");
            assert!(cases.iter().any(|c| c["kind"].as_str().unwrap().ends_with('t')), "the context filled is a case: {m}");
            // (eight users at once: each slower than one alone, their first token later)
            let at = |l: &str| cases.iter().find(|c| c["load"] == l);
            if let (Some(one), Some(eight)) = (at("1k:1"), at("1k:8")) {
                assert!(eight["decode"].as_f64() < one["decode"].as_f64() && eight["ttft"].as_f64() > one["ttft"].as_f64(), "{m}");
            }
        }
        // (gemma-4-12B at kestrel with eight users holds 4,096 tokens each: its 4k and 8k loads are not cases)
        let kestrel = &ms[2];
        assert!(kestrel["cases"].as_array().unwrap().iter().all(|c| !["4k:1", "8k:1"].contains(&c["load"].as_str().unwrap())), "{kestrel}");
        assert!(!out.join("results.json").exists(), "nothing measured");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn loads_and_their_prompts() {
        assert_eq!(loads_of(LOADS).unwrap(), vec![("1k".into(), 1), ("4k".into(), 1), ("8k".into(), 1), ("1k".into(), 8)]);
        assert!(loads_of("1k").is_err() && loads_of("1k:x").is_err());
        assert_eq!((nominal("8k"), nominal("27852t")), (8192, 27852));
    }
}
