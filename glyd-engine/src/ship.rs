//! `ship MODEL LEVEL DIR`: one command from a Hugging Face model to its Glyd repo. The source is downloaded, packed at LEVEL into DIR/repo, its draft model put in DIR/repo/draft
//! (Gemma 4), verified bit for bit against the sha256 recorded from the source, measured through `serve` on this computer's GPU (speed, GPU memory, and with `--evals N` the GSM8K and MMLU-Pro of
//! tools/cards/eval_run.py), its card written from ship_card.md.j2 with what was measured, and with `--repo glyd/NAME` uploaded by the `hf` command (private unless `--public`).
//!
//! Each step that finds its output keeps it, so a run that stopped goes on where it was; `--steps measure,card` on another computer (DIR copied there) adds that GPU's numbers to the card.
//! What was measured is kept in DIR/measured/GPU.txt, one `key=value` a line, apart from the repo.

use crate::args::Args;
use crate::ffi::{Error, Result};
use glyd_gpu::json::{self, Value};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const STEPS: [&str; 7] = ["fetch", "pack", "draft", "verify", "measure", "card", "upload"];
const CARD: &str = include_str!("ship_card.md.j2");

pub fn run(a: &Args) -> Result<()> {
    let (model, level, dir) = (a.need(1, "MODEL")?, a.need(2, "LEVEL")?, PathBuf::from(a.need(3, "DIR")?));
    let layout = crate::cli::ship_layout(model, level).ok_or_else(|| Error(format!("{level}: ship packs penguin, kestrel, swift or wren")))?;
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
            "pack" => crate::pack::run(model, &repo, a.num("threads", std::thread::available_parallelism().map_or(16, |n| n.get()))?, layout)?,
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
            "card" => card(a, model, level, draft.as_deref(), &repo, &dir.join("measured"))?,
            "upload" => {
                let Some(name) = a.flag("repo") else {
                    eprintln!("ship upload: no --repo glyd/NAME, nothing uploaded");
                    continue;
                };
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

/// The draft model Google made for MODEL, a Gemma 4 model (google/gemma-4-12B-it: google/gemma-4-12B-it-assistant), else None.
fn draft_repo(model: &str) -> Option<String> {
    let name = model.rsplit('/').next().unwrap_or(model);
    name.to_ascii_lowercase().starts_with("gemma-4-").then(|| format!("google/{name}-assistant"))
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
    let s: String = gpu.replace("NVIDIA", "").replace("GeForce", "").chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' }).collect();
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
    let mut child = Command::new(exe).args(["serve", &repo.display().to_string(), "--port", &port.to_string()]).env_remove("GLYD_API_KEY").stdout(Stdio::from(logf.try_clone().map_err(|e| Error(e.to_string()))?)).stderr(Stdio::from(logf)).spawn().map_err(|e| Error(e.to_string()))?;
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
        let mut t = format!("gpu={}\ncpu={}\nram={}\nwhen={}\nglyd={}\n", sh("nvidia-smi", &["--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]), crate::qwen_tools::cpu_name(), crate::qwen_tools::ram(), sh("date", &["+%Y-%m-%d"]), env!("CARGO_PKG_VERSION"));
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
            let s = Command::new("python3").env("GLYD_API_KEY", "none").args([script, &format!("{url}/v1"), "model", "1/2", &n.to_string(), &dir.display().to_string()]).status().map_err(|e| Error(format!("python3 {script}: {e}")))?;
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
    let hub = ureq::get(&format!("https://huggingface.co/api/models/{model}")).call().ok().and_then(|r| r.into_body().read_to_string().ok()).and_then(|t| json::parse(&t).ok());
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
    let evals: Vec<String> = text.lines().filter_map(|l| l.strip_prefix("eval=")).map(|e| {
        let f: Vec<&str> = e.split('|').collect();
        format!("{:<10}{} / {}     (every second question, the first {}; greedy, thinking off)", f[0], f.get(1).unwrap_or(&"?"), f.get(2).unwrap_or(&"?"), f.get(2).unwrap_or(&"?"))
    }).collect();
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
    let mut files: Vec<PathBuf> = std::fs::read_dir(measured).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "txt")).collect();
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
        ("repo", minijinja::Value::from(a.flag("repo").map_or_else(|| crate::cli::packed_repo(model, level), str::to_string))),
        ("level", minijinja::Value::from(level.to_string())),
        ("license", minijinja::Value::from(license(model))),
        ("glyd", minijinja::Value::from(a.flag("glyd").map_or(minor, str::to_string))),
        ("disk", minijinja::Value::from(format!("{:.1} GB", gb(repo, Some("draft"))))),
        ("draft", minijinja::Value::from(draft.map(str::to_string))),
        ("draft_disk", minijinja::Value::from(format!("{:.2} GB", gb(&repo.join("draft"), None)))),
        ("machines", minijinja::Value::from(machines)),
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
        let t = env.get_template("card").unwrap().render(ctx).unwrap();
        assert!(t.starts_with("---\nlicense: apache-2.0\n"), "{t}");
        for want in ["# Gemma 4 12B · Glyd Penguin", "glyd run glyd/gemma-4-12B-penguin", "89.8 tok/s", "GSM8K     72 / 75", "RTX 4090", "in draft/", "About 14.9 GiB on the GPU at the penguin level (about 10.8 bits per weight, lossless: every weight is the original's)"] {
            assert!(t.contains(want), "{want} not in:\n{t}");
        }
    }
}
