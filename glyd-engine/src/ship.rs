//! `ship MODEL LEVEL DIR`: one command from a Hugging Face model to its Glyd repo. The source is downloaded, packed at LEVEL into DIR/repo, its draft model put in DIR/repo/draft
//! (Gemma 4), verified bit for bit against the sha256 recorded from the source, measured through `serve` on this computer's GPU (speed, GPU memory, and with `--evals N` the GSM8K and MMLU-Pro of
//! tools/cards/eval_run.py), a lossy level's quality card made (src/quality.rs: the suite on the pack and on the model's lossless level, `--quality-ref FILE` where that was measured on
//! another GPU), its card written from ship_card.md.j2 with what was measured, and with `--repo glyd/NAME` uploaded by the `hf` command (private unless `--public`). A lossy pack is not
//! uploaded without a quality card of its level that says publish (R14). The hummingbird level goes the same way: its pack step codes the experts from a corpus pass's counts
//! (`--usage FILE`, what `glyd-engine calibrate-experts MODEL FILE` wrote; src/hb_pack.rs), and its quality card and the upload's refusal are every lossy level's.
//!
//! Each step that finds its output keeps it, so a run that stopped goes on where it was; `--steps measure,card` on another computer (DIR copied there) adds that GPU's numbers to the card.
//! What was measured is kept in DIR/measured/GPU.txt, one `key=value` a line, apart from the repo.

use crate::args::Args;
use crate::ffi::{Error, Result};
use crate::levels::draft_repo;
use glyd_gpu::json::{self, Value};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const STEPS: [&str; 8] = ["fetch", "pack", "draft", "verify", "measure", "quality", "card", "upload"];
const CARD: &str = include_str!("ship_card.md.j2");

pub fn run(a: &Args) -> Result<()> {
    let (model, level, dir) = (a.need(1, "MODEL")?, a.need(2, "LEVEL")?, PathBuf::from(a.need(3, "DIR")?));
    if crate::levels::by_name(level).is_none() {
        return Err(Error(format!("{level}: ship packs {}", crate::levels::names_now().join(", "))));
    }
    // (None: hummingbird, whose experts are coded from a corpus pass's counts and not by a layout of every matrix)
    let layout = crate::levels::ship_layout(model, level);
    let steps: Vec<&str> = a.flag("steps").map_or(STEPS.to_vec(), |s| s.split(',').map(str::trim).collect());
    if let Some(bad) = steps.iter().find(|s| !STEPS.contains(s)) {
        return Err(Error(format!("--steps {bad}: the steps are {}", STEPS.join(","))));
    }
    let repo = dir.join("repo");
    let draft = draft_repo(model);
    let t0 = Instant::now();
    for step in steps {
        eprintln!("== ship {step} ({:.0} s)", t0.elapsed().as_secs_f64());
        match step {
            "fetch" => {
                for m in std::iter::once(model).chain(draft.as_deref()) {
                    if crate::cli::fetch(&Args::from(["fetch", m].map(String::from))) != 0 {
                        return Err(Error(format!("fetch {m} failed")));
                    }
                }
            }
            "pack" => {
                let threads = a.num("threads", std::thread::available_parallelism().map_or(16, |n| n.get()))?;
                match (layout, a.flag("usage")) {
                    (Some(l), _) => crate::pack::run(model, &repo, threads, l)?,
                    (None, Some(usage)) => crate::hb_pack::run(model, &repo, threads, Path::new(usage))?,
                    (None, None) => {
                        return Err(Error(format!(
                            "ship pack: a {level} pack's experts are coded from a corpus pass's counts: glyd-engine calibrate-experts {model} usage.json, then --usage usage.json"
                        )))
                    }
                }
            }
            "draft" => {
                // (the source's LICENSE and NOTICE go with the weights; a Gemma 4 model's draft model goes in draft/ unchanged, as the engine reads it there)
                let src = source_dir(model)?;
                for f in ["LICENSE", "NOTICE"] {
                    if src.join(f).exists() {
                        link(&src.join(f), &repo.join(f))?;
                    }
                }
                if let Some(d) = &draft {
                    let from = source_dir(d)?;
                    std::fs::create_dir_all(repo.join("draft")).map_err(|e| Error(format!("{}: {e}", repo.display())))?;
                    for e in std::fs::read_dir(&from).map_err(|e| Error(format!("{}: {e}", from.display())))? {
                        let name = e.map_err(|e| Error(e.to_string()))?.file_name();
                        let name = name.to_string_lossy();
                        if name != "README.md" && !name.starts_with('.') {
                            link(&from.join(&*name), &repo.join("draft").join(&*name))?;
                        }
                    }
                }
            }
            "verify" => {
                if crate::cli::verify(&Args::from(["verify".to_string(), repo.display().to_string()])) != 0 {
                    return Err(Error(format!("{}: verify failed", repo.display())));
                }
            }
            "measure" => measure(a, &repo, &dir.join("measured"))?,
            "quality" => quality(a, model, level, &repo, &dir.join("measured"))?,
            "card" => card(a, model, level, draft.as_deref(), &repo, &dir.join("measured"))?,
            "upload" => {
                let Some(name) = a.flag("repo") else {
                    eprintln!("ship upload: no --repo glyd/NAME, nothing uploaded");
                    continue;
                };
                // (a lossy pack goes out with a quality card of its level that says publish, or not at all)
                crate::quality::publishable(level, &repo)?;
                // (`hf upload-large-folder` resumes where an upload stopped; the token is the hf command's own login, never this program's)
                let mut c = Command::new("hf");
                c.args(["upload-large-folder", name, &repo.display().to_string(), "--repo-type", "model"]);
                if !a.has("public") {
                    c.arg("--private");
                }
                let s = c.status().map_err(|e| Error(format!("hf: {e} (pip install huggingface_hub, then hf auth login)")))?;
                if !s.success() {
                    return Err(Error(format!("hf upload-large-folder {name}: {s}")));
                }
            }
            _ => unreachable!(),
        }
    }
    eprintln!("== ship done ({:.0} s): {}", t0.elapsed().as_secs_f64(), repo.display());
    Ok(())
}

fn source_dir(model: &str) -> Result<PathBuf> {
    glyd_gpu::save::Source::find(model).map(|s| s.dir).map_err(|e| Error(format!("{model}: {e}")))
}

/// `to` as a hard link of `from`'s file (the Hub cache's names are links to its blobs: the blob's), a copy where they are on different disks.
fn link(from: &Path, to: &Path) -> Result<()> {
    let from = std::fs::canonicalize(from).map_err(|e| Error(format!("{}: {e}", from.display())))?;
    if to.exists() {
        return Ok(());
    }
    std::fs::hard_link(&from, to).or_else(|_| std::fs::copy(&from, to).map(|_| ())).map_err(|e| Error(format!("{} -> {}: {e}", from.display(), to.display())))
}

fn sh(cmd: &str, args: &[&str]) -> String {
    Command::new(cmd).args(args).output().map_or("?".into(), |o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

/// The GPU's name as a file name: "NVIDIA GeForce RTX 4090" is rtx-4090.
fn slug(gpu: &str) -> String {
    let s: String =
        gpu.replace("NVIDIA", "").replace("GeForce", "").chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' }).collect();
    s.split('-').filter(|p| !p.is_empty()).collect::<Vec<_>>().join("-")
}

/// `serve` of the repo on this computer's GPU, its speed through the API (a short chat, then prompts of about 4K and 30K tokens, or `--lengths`), the GPU memory in use, and with `--evals N` the first
/// N of every second question of GSM8K and MMLU-Pro (tools/cards/eval_run.py, or `--evals-script`): written to measured/GPU.txt.
fn measure(a: &Args, repo: &Path, out: &Path) -> Result<()> {
    let gpu = sh("nvidia-smi", &["--query-gpu=name", "--format=csv,noheader"]).lines().next().unwrap_or("?").to_string();
    std::fs::create_dir_all(out).map_err(|e| Error(format!("{}: {e}", out.display())))?;
    let (file, log) = (out.join(format!("{}.txt", slug(&gpu))), out.join(format!("{}-serve.log", slug(&gpu))));
    let port = a.num("port", 8090u16)?;
    let url = format!("http://127.0.0.1:{port}");
    let exe = std::env::current_exe().map_err(|e| Error(e.to_string()))?;
    let logf = std::fs::File::create(&log).map_err(|e| Error(format!("{}: {e}", log.display())))?;
    let mut child = Command::new(exe)
        .args(["serve", &repo.display().to_string(), "--port", &port.to_string()])
        .env_remove("GLYD_API_KEY")
        .stdout(Stdio::from(logf.try_clone().map_err(|e| Error(e.to_string()))?))
        .stderr(Stdio::from(logf))
        .spawn()
        .map_err(|e| Error(e.to_string()))?;
    let result = (|| -> Result<String> {
        let t0 = Instant::now();
        while ureq::get(&format!("{url}/health")).call().map_or(true, |r| r.status() != 200) {
            if let Ok(Some(s)) = child.try_wait() {
                return Err(Error(format!("serve stopped ({s}): {}", log.display())));
            }
            if t0.elapsed() > Duration::from_secs(1800) {
                return Err(Error(format!("serve was not up in 30 minutes: {}", log.display())));
            }
            std::thread::sleep(Duration::from_secs(5));
        }
        let lengths: Vec<usize> = a.flag("lengths").unwrap_or("4000,30000").split(',').filter_map(|x| x.trim().parse().ok()).collect();
        let mut t = format!(
            "gpu={}\ncpu={}\nram={}\nwhen={}\nglyd={}\n",
            sh("nvidia-smi", &["--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
            crate::qwen_tools::cpu_name(),
            crate::qwen_tools::ram(),
            sh("date", &["+%Y-%m-%d"]),
            env!("CARGO_PKG_VERSION")
        );
        let served = std::fs::read_to_string(&log).unwrap_or_default();
        for l in served.lines() {
            if l.contains(" on the GPU at the ") {
                t.push_str(&format!("about={}\n", l.split_once(": ").map_or(l, |x| x.1)));
            } else if l.starts_with("Ready in ") {
                t.push_str(&format!("ready={l}\n"));
            }
        }
        for (name, prompt, first, rate, made) in crate::qwen_tools::speed(&url, &lengths)? {
            t.push_str(&format!("row={name}|{prompt}|{first:.2}|{rate:.1}|{made}\n"));
        }
        t.push_str(&format!("memory={}\n", sh("nvidia-smi", &["--query-gpu=memory.used,memory.total", "--format=csv,noheader"]).lines().next().unwrap_or("?")));
        let n = a.num("evals", 0usize)?;
        if n > 0 {
            let dir = out.join(format!("{}-evals", slug(&gpu)));
            let script = a.flag("evals-script").unwrap_or("tools/cards/eval_run.py");
            // (Inspect's openai-api provider reads GLYD_API_KEY for the service "glyd"; the server here has no key, and takes any)
            let s = Command::new("python3")
                .env("GLYD_API_KEY", "none")
                .args([script, &format!("{url}/v1"), "model", "1/2", &n.to_string(), &dir.display().to_string()])
                .status()
                .map_err(|e| Error(format!("python3 {script}: {e}")))?;
            if !s.success() {
                return Err(Error(format!("python3 {script}: {s}")));
            }
            let rows = std::fs::read_to_string(dir.join("evals.json")).map_err(|e| Error(e.to_string()))?;
            for r in json::parse(&rows).map_err(Error)?.as_array().unwrap_or(&[]) {
                let num = |k: &str| match r.get(k) {
                    Some(Value::Number(n)) => n.parse::<f64>().unwrap_or(0.0),
                    _ => 0.0,
                };
                let (task, n) = (r.get("task_id").and_then(Value::as_str).unwrap_or("?"), num("n"));
                t.push_str(&format!("eval={}|{:.0}|{n:.0}\n", if task == "gsm8k" { "GSM8K" } else { "MMLU-Pro" }, num("value") * n));
            }
        }
        Ok(t)
    })();
    let _ = child.kill();
    let _ = child.wait();
    let t = result?;
    std::fs::write(&file, &t).map_err(|e| Error(format!("{}: {e}", file.display())))?;
    eprint!("{t}");
    Ok(())
}

/// A lossy level's quality card (src/quality.rs): the suite on the model's lossless level (`--quality-ref FILE`: a result measured before, where this GPU does not hold the lossless
/// level) and on the pack, each kept in DIR/measured (a step run again keeps what it finds), and the card in the pack, repo/quality.json, uploaded with it. `--n`, `--tasks`, `--long`
/// and `--ctx` are the suite's. The lossless level needs none.
fn quality(a: &Args, model: &str, level: &str, repo: &Path, measured: &Path) -> Result<()> {
    if crate::levels::by_name(level).is_none_or(|l| l.lossless) {
        eprintln!("ship quality: {level} is lossless, no card");
        return Ok(());
    }
    std::fs::create_dir_all(measured).map_err(|e| Error(format!("{}: {e}", measured.display())))?;
    let suite = |target: &str, file: &Path| -> Result<serde_json::Value> {
        if let Ok(t) = std::fs::read_to_string(file) {
            return serde_json::from_str(&t).map_err(|e| Error(format!("{}: {e}", file.display())));
        }
        let mut words = vec!["quality".to_string(), target.to_string()];
        for f in ["n", "tasks", "long", "ctx"] {
            if let Some(v) = a.flag(f) {
                words.extend([format!("--{f}"), v.to_string()]);
            }
        }
        let r = crate::quality::measure(&Args::from(words))?;
        std::fs::write(file, serde_json::to_string_pretty(&r).unwrap_or_default() + "\n").map_err(|e| Error(format!("{}: {e}", file.display())))?;
        Ok(r)
    };
    let reference = match a.flag("quality-ref") {
        Some(f) => serde_json::from_str(&std::fs::read_to_string(f).map_err(|e| Error(format!("{f}: {e}")))?).map_err(|e| Error(format!("{f}: {e}")))?,
        None => suite(&format!("{model}:penguin"), &measured.join("quality-ref.json"))?,
    };
    let cand = suite(&repo.display().to_string(), &measured.join("quality.json"))?;
    let card = crate::quality::card(&reference, &cand, level, level)?;
    let out = repo.join("quality.json");
    std::fs::write(&out, serde_json::to_string_pretty(&card).unwrap_or_default() + "\n").map_err(|e| Error(format!("{}: {e}", out.display())))?;
    eprintln!("ship quality: {}: {}", out.display(), crate::quality::line(&card));
    Ok(())
}

/// The card's lines of quality from the pack's quality card, where it has one: the figures, the bar they were held to and the verdict.
fn quality_lines(repo: &Path) -> Option<minijinja::Value> {
    let c: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(repo.join("quality.json")).ok()?).ok()?;
    let (kl, bar) = (&c["kl"], &c["bar"]);
    let mut lines = vec![
        format!(
            "{:<34}{:>10}   (at most {})",
            format!("next-token KL ({} prefixes)", kl["prefixes"]),
            kl["mean"].as_f64().map_or("-".into(), |m| format!("{m:.4}")),
            bar["kl_mean"]
        ),
        format!(
            "{:<34}{:>9.1}%   (at least {:.0}%)",
            "best token the same",
            kl["top1"].as_f64().unwrap_or(0.0) * 100.0,
            bar["top1"].as_f64().unwrap_or(0.0) * 100.0
        ),
    ];
    for (name, t) in c["tasks"].as_object().into_iter().flatten() {
        let n = t["n"].as_f64().unwrap_or(0.0);
        lines.push(format!(
            "{:<34}{:>10}   (lossless {:.0})",
            name,
            format!("{:.0} / {n:.0}", t["candidate"].as_f64().unwrap_or(0.0) * n),
            t["reference"].as_f64().unwrap_or(0.0) * n
        ));
    }
    lines.push(format!("{:<34}{:>10}", "verdict", c["verdict"].as_str().unwrap_or("?")));
    Some(minijinja::Value::from_iter([
        ("gpu", minijinja::Value::from(c["measured"]["gpu"].as_str().unwrap_or("?").to_string())),
        ("lines", minijinja::Value::from(lines)),
    ]))
}

/// The bytes of the files in `dir` and below, decimal GB; `skip`: a folder left out.
fn gb(dir: &Path, skip: Option<&str>) -> f64 {
    let mut n = 0u64;
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let p = e.path();
        if p.is_dir() {
            if skip != Some(&*e.file_name().to_string_lossy()) {
                n += (gb(&p, None) * 1e9) as u64;
            }
        } else {
            n += e.metadata().map_or(0, |m| m.len());
        }
    }
    n as f64 / 1e9
}

/// The source's license as the Hub's model page has it (`cardData.license`, which a gated model shows too), else as its README's front matter says, else "other".
fn license(model: &str) -> String {
    let hub = ureq::get(&format!("https://huggingface.co/api/models/{model}"))
        .call()
        .ok()
        .and_then(|r| r.into_body().read_to_string().ok())
        .and_then(|t| json::parse(&t).ok());
    if let Some(l) = hub.as_ref().and_then(|v| v.get("cardData")).and_then(|c| c.get("license")).and_then(Value::as_str) {
        return l.to_string();
    }
    let card = source_dir(model).ok().and_then(|d| std::fs::read_to_string(d.join("README.md")).ok()).unwrap_or_default();
    card.lines().take_while(|l| !l.starts_with("# ")).find_map(|l| l.strip_prefix("license:").map(|v| v.trim().to_string())).unwrap_or_else(|| "other".into())
}

/// One machine's measurements, as the card shows them.
fn machine(text: &str) -> minijinja::Value {
    let get = |k: &str| text.lines().find_map(|l| l.strip_prefix(&format!("{k}="))).unwrap_or("?").to_string();
    let mut rows = vec![format!("{:<22}{:>8}{:>14}{:>16}", "", "prompt", "first token", "writing")];
    for r in text.lines().filter_map(|l| l.strip_prefix("row=")) {
        let f: Vec<&str> = r.split('|').collect();
        if let [name, prompt, first, rate, _made] = f[..] {
            rows.push(format!("{name:<22}{prompt:>8}{:>14}{:>16}", format!("{first} s"), format!("{rate} tok/s")));
        }
    }
    let evals: Vec<String> = text
        .lines()
        .filter_map(|l| l.strip_prefix("eval="))
        .map(|e| {
            let f: Vec<&str> = e.split('|').collect();
            format!(
                "{:<10}{} / {}     (every second question, the first {}; greedy, thinking off)",
                f[0],
                f.get(1).unwrap_or(&"?"),
                f.get(2).unwrap_or(&"?"),
                f.get(2).unwrap_or(&"?")
            )
        })
        .collect();
    let gpu = get("gpu");
    minijinja::Value::from_iter([
        ("gpu", minijinja::Value::from(gpu.split(',').next().unwrap_or("?").trim().to_string())),
        ("driver", minijinja::Value::from(gpu.rsplit(',').next().unwrap_or("?").trim().to_string())),
        ("cpu", minijinja::Value::from(get("cpu"))),
        ("ram", minijinja::Value::from(get("ram"))),
        ("when", minijinja::Value::from(get("when"))),
        ("glyd", minijinja::Value::from(get("glyd"))),
        ("about", minijinja::Value::from(get("about"))),
        ("ready", minijinja::Value::from(get("ready"))),
        ("memory", minijinja::Value::from(get("memory"))),
        ("rows", minijinja::Value::from(rows)),
        ("evals", minijinja::Value::from(evals)),
    ])
}

/// README.md of the repo: the card, from ship_card.md.j2 and every machine measured.
fn card(a: &Args, model: &str, level: &str, draft: Option<&str>, repo: &Path, measured: &Path) -> Result<()> {
    let mut machines = Vec::new();
    let mut files: Vec<PathBuf> =
        std::fs::read_dir(measured).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "txt")).collect();
    files.sort();
    for f in files {
        machines.push(machine(&std::fs::read_to_string(&f).map_err(|e| Error(format!("{}: {e}", f.display())))?));
    }
    if machines.is_empty() {
        return Err(Error(format!("{}: nothing measured yet (the measure step)", measured.display())));
    }
    let name = model.rsplit('/').next().unwrap_or(model);
    let minor = env!("CARGO_PKG_VERSION").rsplitn(2, '.').nth(1).unwrap_or(env!("CARGO_PKG_VERSION")).to_string();
    let ctx = minijinja::Value::from_iter([
        ("base", minijinja::Value::from(model.to_string())),
        ("title", minijinja::Value::from(a.flag("title").unwrap_or(name).to_string())),
        ("repo", minijinja::Value::from(a.flag("repo").map_or_else(|| crate::levels::packed_repo(model, level), str::to_string))),
        ("level", minijinja::Value::from(level.to_string())),
        ("license", minijinja::Value::from(license(model))),
        ("glyd", minijinja::Value::from(a.flag("glyd").map_or(minor, str::to_string))),
        ("disk", minijinja::Value::from(format!("{:.1} GB", gb(repo, Some("draft"))))),
        ("draft", minijinja::Value::from(draft.map(str::to_string))),
        ("draft_disk", minijinja::Value::from(format!("{:.2} GB", gb(&repo.join("draft"), None)))),
        ("machines", minijinja::Value::from(machines)),
        ("quality", quality_lines(repo).unwrap_or(minijinja::Value::from(()))),
    ]);
    let mut env = minijinja::Environment::new();
    env.set_trim_blocks(true);
    env.add_template("card", CARD).map_err(|e| Error(format!("ship_card.md.j2: {e}")))?;
    let text = env.get_template("card").and_then(|t| t.render(ctx)).map_err(|e| Error(format!("ship_card.md.j2: {e}")))?;
    let out = repo.join("README.md");
    std::fs::write(&out, text + "\n").map_err(|e| Error(format!("{}: {e}", out.display())))?;
    eprintln!("card: {}", out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_the_card() {
        assert_eq!(slug("NVIDIA GeForce RTX 4090"), "rtx-4090");
        assert_eq!(slug("NVIDIA H100 80GB HBM3"), "h100-80gb-hbm3");
        assert_eq!(draft_repo("google/gemma-4-12B-it").as_deref(), Some("google/gemma-4-12B-it-assistant"));
        assert_eq!(draft_repo("Qwen/Qwen3.5-9B"), None);
        // (a machine's file as measure writes it, rendered through the template)
        let m = machine("gpu=NVIDIA GeForce RTX 4090, 24564 MiB, 580.159.04\ncpu=AMD EPYC 7282\nram=504 GiB\nwhen=2026-10-16\nglyd=0.32.0\nabout=about 14.9 GiB on the GPU at the penguin level (about 10.8 bits per weight, lossless: every weight is the original's), in place of 22.2 GiB of bf16.\nready=Ready in 21s\nrow=Chat (short prompt)|34|0.02|89.8|512\nrow=4K prompt|3839|0.71|97.5|191\nmemory=17556 MiB, 24564 MiB\neval=GSM8K|72|75\n");
        let ctx = minijinja::Value::from_iter([
            ("base", minijinja::Value::from("google/gemma-4-12B-it")),
            ("title", minijinja::Value::from("Gemma 4 12B")),
            ("repo", minijinja::Value::from("glyd/gemma-4-12B-penguin")),
            ("level", minijinja::Value::from("penguin")),
            ("license", minijinja::Value::from("apache-2.0")),
            ("glyd", minijinja::Value::from("0.32")),
            ("disk", minijinja::Value::from("16.0 GB")),
            ("draft", minijinja::Value::from(Some("google/gemma-4-12B-it-assistant".to_string()))),
            ("draft_disk", minijinja::Value::from("0.85 GB")),
            ("machines", minijinja::Value::from(vec![m])),
        ]);
        let mut env = minijinja::Environment::new();
        env.set_trim_blocks(true);
        env.add_template("card", CARD).unwrap();
        let t = env.get_template("card").unwrap().render(&ctx).unwrap();
        assert!(t.starts_with("---\nlicense: apache-2.0\n"), "{t}");
        // (a lossy pack's card has the quality section of its quality.json)
        let dir = std::env::temp_dir().join(format!("glyd-ship-quality-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let card = serde_json::json!({"schema": crate::quality::CARD_SCHEMA, "level": "wren", "verdict": "publish", "measured": {"gpu": "NVIDIA H100 80GB HBM3"},
            "kl": {"mean": 0.0123, "top1": 0.964, "prefixes": 192}, "bar": {"kl_mean": 0.08, "top1": 0.86},
            "tasks": {"reasoning": {"n": 40, "candidate": 0.95, "reference": 0.975}}});
        std::fs::write(dir.join("quality.json"), card.to_string()).unwrap();
        let q = env.get_template("card").unwrap().render(minijinja::context! { quality => quality_lines(&dir).unwrap(), ..ctx });
        std::fs::remove_dir_all(&dir).ok();
        let q = q.unwrap_or_else(|e| panic!("{e:#}"));
        for want in ["## Quality", "NVIDIA H100 80GB HBM3", "0.0123", "96.4%", "38 / 40", "(lossless 39)", "publish"] {
            assert!(q.contains(want), "{want} not in:\n{q}");
        }
        assert!(!t.contains("## Quality"), "a card without quality.json has no quality section");
        for want in [
            "# Gemma 4 12B · Glyd Penguin",
            "glyd run glyd/gemma-4-12B-penguin",
            "89.8 tok/s",
            "GSM8K     72 / 75",
            "RTX 4090",
            "in draft/",
            "About 14.9 GiB on the GPU at the penguin level (about 10.8 bits per weight, lossless: every weight is the original's)",
        ] {
            assert!(t.contains(want), "{want} not in:\n{t}");
        }
    }

    /// The hummingbird level is shipped as every lossy level is: its pack step asks for the corpus pass's counts, and its upload is refused without a quality card of
    /// its level that says publish (before any `hf` command runs). What is no level is refused by name.
    #[test]
    fn a_hummingbird_pack_goes_through_the_quality_gate() {
        let dir = std::env::temp_dir().join(format!("glyd-ship-hb-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("repo")).unwrap();
        let ship = |level: &str, more: &[&str]| {
            let words = ["ship", "org/model", level, &dir.to_string_lossy()].iter().chain(more).map(|w| w.to_string()).collect::<Vec<_>>();
            run(&Args::from(words)).map_err(|e| e.msg)
        };
        let e = ship("hummingbird", &["--steps", "pack"]).unwrap_err();
        assert!(e.contains("calibrate-experts org/model usage.json") && e.contains("--usage"), "{e}");
        let e = ship("hummingbird", &["--steps", "upload", "--repo", "glyd/model-hummingbird"]).unwrap_err();
        assert!(e.contains("a hummingbird pack is published with its quality card"), "{e}");
        // (a card of another level, then one under the bar: still refused; one that says publish passes the gate, and the lossless level needs none)
        let card = |level: &str, verdict: &str| {
            let c =
                serde_json::json!({"schema": crate::quality::CARD_SCHEMA, "level": level, "verdict": verdict, "reasons": ["KL 0.2000 is past the bar's 0.12"]});
            std::fs::write(dir.join("repo/quality.json"), c.to_string()).unwrap();
        };
        card("wren", "publish");
        assert!(ship("hummingbird", &["--steps", "upload", "--repo", "glyd/x"]).unwrap_err().contains("the card is of"));
        card("hummingbird", "refuse");
        assert!(ship("hummingbird", &["--steps", "upload", "--repo", "glyd/x"]).unwrap_err().contains("under its quality bar: KL 0.2000"));
        card("hummingbird", "publish");
        assert!(crate::quality::publishable("hummingbird", &dir.join("repo")).is_ok());
        assert!(ship("hummingbird", &["--steps", "upload"]).is_ok(), "no --repo: nothing uploaded");
        assert!(ship("sparrow", &["--steps", "upload"]).unwrap_err().contains("sparrow: ship packs penguin, kestrel, swift, wren, hummingbird"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
