//! Two tools beside `qwenserve` for the Qwen release: `qwenbench`, how fast it is on this computer (a short chat, then prompts of about 4K, 16K and 32K tokens against a running server, and a Markdown
//! report to share), and `qwenmcp`, an MCP server on the standard input and output so AI tools can check, start, stop and ask it.

use crate::args::Args;
use crate::ffi::{Error, Result};
use crate::server::jstr;
use glyd_gpu::json::{self, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn post(url: &str, body: &str) -> Result<impl Read> {
    let r = ureq::post(url).header("Content-Type", "application/json").send(body).map_err(|e| Error(format!("{url}: {e}")))?;
    Ok(r.into_body().into_reader())
}

/// One streamed chat request: (seconds to the first word, tokens a second after it, tokens written, prompt tokens).
fn chat(url: &str, content: &str, max_tokens: usize) -> Result<(f64, f64, u64, u64)> {
    let body = format!(
        "{{\"messages\":[{{\"role\":\"user\",\"content\":{}}}],\"max_tokens\":{max_tokens},\"stream\":true,\"stream_options\":{{\"include_usage\":true}},\"chat_template_kwargs\":{{\"enable_thinking\":false}}}}",
        jstr(content)
    );
    let t0 = Instant::now();
    let (mut first, mut made, mut prompt) = (None, 0, 0);
    for line in BufReader::new(post(&format!("{url}/v1/chat/completions"), &body)?).lines() {
        let line = line.map_err(|e| Error(e.to_string()))?;
        let Some(d) = line.strip_prefix("data:").map(str::trim) else { continue };
        if d == "[DONE]" {
            break;
        }
        let v = json::parse(d).map_err(Error)?;
        if let Some(e) = v.get("error") {
            return Err(Error(format!("{url}: {}", e.get("message").and_then(Value::as_str).unwrap_or(d))));
        }
        if let Some(u) = v.get("usage").filter(|u| !matches!(u, Value::Null)) {
            made = u.get("completion_tokens").and_then(Value::as_u64).unwrap_or(0);
            prompt = u.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0);
        }
        let wrote = v.get("choices").and_then(Value::as_array).unwrap_or(&[]).iter().any(|c| c.get("delta").and_then(|d| d.get("content")).and_then(Value::as_str).is_some_and(|t| !t.is_empty()));
        if wrote && first.is_none() {
            first = Some(t0.elapsed().as_secs_f64());
        }
    }
    let end = t0.elapsed().as_secs_f64();
    let first = first.unwrap_or(end);
    Ok((first, if made > 1 && end > first { (made - 1) as f64 / (end - first) } else { 0.0 }, made, prompt))
}

fn run(cmd: &str, args: &[&str]) -> String {
    std::process::Command::new(cmd).args(args).output().map_or("?".into(), |o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

const FILLER: &str = "The history of the printing press is a story of many hands. Workshops in several cities tried movable type, inks and presses, and each change spread along the rivers, carried by apprentices who set up shops of their own. ";

/// This computer's CPU, as /proc/cpuinfo names it.
pub fn cpu_name() -> String {
    std::fs::read_to_string("/proc/cpuinfo").ok().and_then(|t| t.lines().find_map(|l| l.strip_prefix("model name").and_then(|r| r.split_once(':')).map(|x| x.1.trim().to_string()))).unwrap_or("?".into())
}

/// This computer's memory, in GiB.
pub fn ram() -> String {
    crate::os::meminfo().and_then(|t| t.lines().find_map(|l| l.strip_prefix("MemTotal:").map(|r| r.trim().trim_end_matches("kB").trim().parse::<f64>().unwrap_or(0.0)))).map_or("?".into(), |kb| format!("{:.0} GiB", kb / 1048576.0))
}

/// A server's speed: a warm-up, a short chat of 512 tokens, then a prompt of about each of `lengths` tokens summarised in up to 256 (asked for about 300 words, so the rate is over all 256): (test, prompt tokens, seconds to the first word, tokens a second after it,
/// tokens written) for each.
pub fn speed(url: &str, lengths: &[usize]) -> Result<Vec<(String, u64, f64, f64, u64)>> {
    chat(url, "Say hello.", 16)?; // (a warm-up)
    let mut rows = Vec::new();
    let (first, rate, made, prompt) = chat(url, "Explain, step by step, how a hash table handles collisions, then compare open addressing with chaining.", 512)?;
    println!("chat: {rate:.1} tokens/s");
    rows.push(("Chat (short prompt)".to_string(), prompt, first, rate, made));
    for &want in lengths {
        let text: String = FILLER.repeat(want / 45 + 1).chars().take(want * 4).collect();
        let (first, rate, made, prompt) = chat(url, &format!("{text}\n\nWrite a detailed summary of the text above, in about 300 words."), 256)?;
        println!("{}K: read in {first:.1} s, then {rate:.1} tokens/s", want / 1000);
        rows.push((format!("{}K prompt", want / 1000), prompt, first, rate, made));
    }
    Ok(rows)
}

/// `qwenbench [--url http://127.0.0.1:8080] [--out bench-report.md] [--lengths 4000,16000,32000]`.
pub fn bench(a: &Args) -> Result<()> {
    let url = a.flag("url").unwrap_or("http://127.0.0.1:8080").trim_end_matches('/').to_string();
    let out = a.flag("out").unwrap_or("bench-report.md").to_string();
    let lengths: Vec<usize> = a.flag("lengths").unwrap_or("4000,16000,32000").split(',').filter_map(|x| x.trim().parse().ok()).collect();
    let gpu = run("nvidia-smi", &["--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]);
    let (cpu, ram) = (cpu_name(), ram());
    let rows = speed(&url, &lengths)?;
    let mut r = format!("# Qwen3.8-Flash-Next on Glyd: speed\n\n- GPU: {gpu}\n- CPU: {cpu}\n- RAM: {ram}\n- When: {}\n\n", run("date", &["+%Y-%m-%d %H:%M"]));
    r.push_str("| Test | Prompt tokens | Reading the prompt (s) | Writing (tokens/s) | Tokens written |\n|---|--:|--:|--:|--:|\n");
    for (name, prompt, first, rate, made) in rows {
        r.push_str(&format!("| {name} | {prompt} | {first:.1} | {rate:.1} | {made} |\n"));
    }
    std::fs::write(&out, r).map_err(|e| Error(format!("{out}: {e}")))?;
    println!("report: {out}");
    Ok(())
}

const TOOLS: &str = r#"[
{"name":"glyd_status","description":"Whether Qwen3.8-Flash-Next (on this computer's GPU, with Glyd) is running, and its address.","inputSchema":{"type":"object","properties":{}}},
{"name":"glyd_start","description":"Start the Glyd server (chat page and OpenAI/Anthropic API). Loading takes about a minute.","inputSchema":{"type":"object","properties":{"mode":{"type":"string","enum":["precise","fast","best"],"description":"precise (default), fast or best"}}}},
{"name":"glyd_stop","description":"Stop the Glyd server and free the GPU.","inputSchema":{"type":"object","properties":{}}},
{"name":"glyd_ask","description":"Ask the local model one question and get its answer.","inputSchema":{"type":"object","properties":{"prompt":{"type":"string"},"max_tokens":{"type":"integer"}},"required":["prompt"]}}
]"#;

struct Mcp {
    home: PathBuf,
    repo: PathBuf,
    url: String,
}

impl Mcp {
    fn healthy(&self) -> bool {
        let agent = ureq::Agent::new_with_config(ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(2))).build());
        agent.get(format!("{}/health", self.url)).call().is_ok_and(|r| r.status() == 200)
    }

    fn call(&self, name: &str, args: &Value) -> std::result::Result<String, String> {
        let (url, log) = (&self.url, self.home.join("server.log"));
        let pidfile = self.home.join("server.pid");
        match name {
            "glyd_status" => Ok(if self.healthy() { format!("running at {url} (chat page {url}/, API {url}/v1)") } else { "not running".into() }),
            "glyd_start" => {
                if self.healthy() {
                    return Ok(format!("already running at {url}"));
                }
                #[cfg(unix)]
                use std::os::unix::process::CommandExt;
                #[cfg(windows)]
                use std::os::windows::process::CommandExt;
                let f = std::fs::OpenOptions::new().create(true).append(true).open(&log).map_err(|e| format!("{}: {e}", log.display()))?;
                #[cfg(unix)]
                let mut cmd = std::process::Command::new("bash");
                #[cfg(unix)]
                {
                    cmd.arg(self.repo.join("tools/serve.sh"));
                    match args.get("mode").and_then(Value::as_str) {
                        Some("fast") => _ = cmd.arg("--fast"),
                        Some("best") => _ = cmd.arg("--best"),
                        _ => {}
                    }
                }
                // (Windows: the PowerShell twin)
                #[cfg(windows)]
                let mut cmd = std::process::Command::new("powershell");
                #[cfg(windows)]
                {
                    cmd.args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]).arg(self.repo.join("tools").join("serve.ps1"));
                    match args.get("mode").and_then(Value::as_str) {
                        Some("fast") => _ = cmd.arg("-Fast"),
                        Some("best") => _ = cmd.arg("-Best"),
                        _ => {}
                    }
                }
                // (a process group of its own, so that stopping it stops the server it starts; Windows: CREATE_NEW_PROCESS_GROUP)
                #[cfg(unix)]
                cmd.process_group(0);
                #[cfg(windows)]
                cmd.creation_flags(0x200);
                let mut child = cmd.stdout(f.try_clone().map_err(|e| e.to_string())?).stderr(f).stdin(std::process::Stdio::null()).spawn().map_err(|e| e.to_string())?;
                let _ = std::fs::write(&pidfile, child.id().to_string());
                for _ in 0..600 {
                    if self.healthy() {
                        // (the server answers while the model loads: wait for it to be ready)
                        let ready = ureq::get(format!("{url}/v1/glyd/status")).call().ok().and_then(|mut r| r.body_mut().read_to_string().ok()).is_some_and(|t| !t.contains("\"loading\""));
                        if ready {
                            return Ok(format!("started at {url}"));
                        }
                    }
                    if let Ok(Some(_)) = child.try_wait() {
                        return Ok(format!("the server exited; see {}", log.display()));
                    }
                    std::thread::sleep(Duration::from_secs(1));
                }
                Ok(format!("still loading; see {}", log.display()))
            }
            "glyd_stop" => {
                let Ok(pid) = std::fs::read_to_string(&pidfile) else { return Ok("not started by this tool".into()) };
                // (a process id, never 0 or 1: as a group those are this tool's own and every process's)
                let Some(pid) = pid.trim().parse::<u32>().ok().filter(|&p| p > 1) else { return Err(format!("{}: not a process id", pidfile.display())) };
                // (the server's process group; "--" first: procps-ng 3.3.17's kill, Ubuntu 22.04's, reads "-TERM -PID" as its own group)
                #[cfg(unix)]
                let _ = std::process::Command::new("kill").args(["-TERM", "--", &format!("-{pid}")]).status();
                // (Windows: the script and the server it started, its process tree)
                #[cfg(windows)]
                let _ = std::process::Command::new("taskkill").args(["/PID", &pid.to_string(), "/T", "/F"]).status();
                let _ = std::fs::remove_file(&pidfile);
                Ok("stopped".into())
            }
            "glyd_ask" => {
                let prompt = args.get("prompt").and_then(Value::as_str).ok_or("prompt: required")?;
                let max = args.get("max_tokens").and_then(Value::as_u64).unwrap_or(1024);
                let body = format!("{{\"messages\":[{{\"role\":\"user\",\"content\":{}}}],\"max_tokens\":{max}}}", jstr(prompt));
                let mut text = String::new();
                post(&format!("{url}/v1/chat/completions"), &body).map_err(|e| e.0)?.read_to_string(&mut text).map_err(|e| e.to_string())?;
                let v = json::parse(&text)?;
                v.get("choices").and_then(Value::as_array).and_then(|c| c.first()).and_then(|c| c.get("message")).and_then(|m| m.get("content")).and_then(Value::as_str).map(str::to_string).ok_or(text)
            }
            _ => Err(format!("no tool {name}")),
        }
    }
}

/// `qwenmcp --repo DIR [--url URL] [--home DIR]`: the MCP server (JSON-RPC a line on the standard input). --repo is this repository (its tools/serve.sh starts the server), --url the server's address
/// (http://127.0.0.1:8080), --home where the install keeps its log and the server's process id (~/.glyd-qwen-3.8-flash).
pub fn mcp(a: &Args) -> Result<()> {
    let home = a.flag("home").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(crate::os::home()).join(".glyd-qwen-3.8-flash"));
    let repo = PathBuf::from(a.flag("repo").ok_or_else(|| Error("qwenmcp: --repo DIR (this repository) is needed".into()))?);
    let m = Mcp { home, repo, url: a.flag("url").unwrap_or("http://127.0.0.1:8080").trim_end_matches('/').to_string() };
    let _ = std::fs::create_dir_all(&m.home);
    let mut out = std::io::stdout();
    for line in std::io::stdin().lock().lines() {
        let line = line.map_err(|e| Error(e.to_string()))?;
        let Ok(msg) = json::parse(&line) else { continue };
        // (a notification has no id: nothing to answer)
        let Some(id) = msg.get("id").filter(|i| !matches!(i, Value::Null)).map(json::to_compact) else { continue };
        let p = msg.get("params");
        let res = match msg.get("method").and_then(Value::as_str).unwrap_or("") {
            "initialize" => {
                let v = p.and_then(|p| p.get("protocolVersion")).and_then(Value::as_str).unwrap_or("2025-06-18");
                format!("{{\"protocolVersion\":{},\"capabilities\":{{\"tools\":{{}}}},\"serverInfo\":{{\"name\":\"condor\",\"version\":\"0.2.3\"}}}}", jstr(v))
            }
            // (one line: a message a line)
            "tools/list" => format!("{{\"tools\":{}}}", json::parse(TOOLS).map(|v| json::to_compact(&v)).unwrap_or_default()),
            "tools/call" => {
                let name = p.and_then(|p| p.get("name")).and_then(Value::as_str).unwrap_or("");
                let empty = Value::Object(vec![]);
                match m.call(name, p.and_then(|p| p.get("arguments")).unwrap_or(&empty)) {
                    Ok(t) => format!("{{\"content\":[{{\"type\":\"text\",\"text\":{}}}]}}", jstr(&t)),
                    Err(e) => format!("{{\"content\":[{{\"type\":\"text\",\"text\":{}}}],\"isError\":true}}", jstr(&e)),
                }
            }
            "ping" => "{}".into(),
            other => {
                let _ = writeln!(out, "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"error\":{{\"code\":-32601,\"message\":{}}}}}", jstr(&format!("no method {other}")));
                let _ = out.flush();
                continue;
            }
        };
        let _ = writeln!(out, "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"result\":{res}}}");
        let _ = out.flush();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_tools_are_json() {
        let v = glyd_gpu::json::parse(super::TOOLS).unwrap();
        assert_eq!(v.as_array().map(|t| t.len()), Some(4));
    }
}
