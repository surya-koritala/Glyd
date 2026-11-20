//! `glyd connect [TOOL]`: what to give a program so that it talks to the Glyd server on this computer, with the server's real port and model where one runs. A program that takes a base URL (the OpenAI and Anthropic SDKs,
//! Claude Code, LangChain, aider), one that talks to Ollama (Open WebUI, Continue, the `ollama` libraries), and a web UI in a container each get the lines that set them up, ready to paste; nothing is written or
//! changed. The key is never printed: where the server has one (`GLYD_API_KEY` is set here), the lines say `$GLYD_API_KEY`.

use crate::args::Args;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::time::Duration;

/// The port `glyd serve` listens on where it is not told another: 8000, or the next free one up to 8020 (`cli::listen`).
pub const DEFAULT_PORT: u16 = 8000;

/// Ollama's own port, where the background service listens (and `glyd serve --ollama-port` can): looked at after the ports `serve` moves through.
pub const OLLAMA_PORT: u16 = 11434;

/// The tools that have lines here, with what each is.
const TOOLS: &[(&str, &str)] = &[
    ("openai", "the OpenAI SDKs, LangChain's ChatOpenAI, aider, and any program with an OpenAI base URL"),
    ("anthropic", "the Anthropic SDKs and Claude Code"),
    ("ollama", "programs that talk to Ollama: the ollama libraries, LangChain's ChatOllama"),
    ("open-webui", "Open WebUI in a container"),
    ("continue", "Continue (VS Code and JetBrains): chat and autocomplete"),
    ("curl", "a request by hand"),
];

/// The server that was found: where it listens, and the model it serves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Found {
    pub host: String,
    pub port: u16,
    pub model: String,
}

/// A request's answer: (the status, the body).
fn get(addr: SocketAddr, host: &str, path: &str, key: Option<&str>) -> Option<(u16, String)> {
    let mut s = TcpStream::connect_timeout(&addr, Duration::from_millis(400)).ok()?;
    s.set_read_timeout(Some(Duration::from_secs(3))).ok()?;
    let auth = key.map_or(String::new(), |k| format!("Authorization: Bearer {k}\r\n"));
    s.write_all(format!("GET {path} HTTP/1.1\r\nHost: {host}\r\n{auth}Connection: close\r\n\r\n").as_bytes()).ok()?;
    let mut out = String::new();
    s.read_to_string(&mut out).ok()?;
    let (head, body) = out.split_once("\r\n\r\n")?;
    Some((head.split_whitespace().nth(1)?.parse().ok()?, body.to_string()))
}

/// The server on `host`: at `port`, or where none is given at the first of 8000 to 8020, then 11434, that answers as Glyd does (`/v1/models` with `owned_by` glyd). `None` where there is none.
pub fn find(host: &str, port: Option<u16>, key: Option<&str>) -> Option<Found> {
    let ports: Vec<u16> = port.map_or((DEFAULT_PORT..=DEFAULT_PORT + 20).chain([OLLAMA_PORT]).collect(), |p| vec![p]);
    for p in ports {
        let Some(addr) = (host, p).to_socket_addrs().ok().and_then(|mut a| a.next()) else { continue };
        let Some((200, body)) = get(addr, &format!("{host}:{p}"), "/v1/models", key) else { continue };
        let Ok(v) = glyd_gpu::json::parse(&body) else { continue };
        let first = v.get("data").and_then(|d| d.as_array()).and_then(|d| d.first());
        if let Some(m) = first.filter(|m| m.get("owned_by").and_then(|o| o.as_str()) == Some("glyd")) {
            return Some(Found { host: host.to_string(), port: p, model: m.get("id").and_then(|i| i.as_str()).unwrap_or("").to_string() });
        }
    }
    None
}

/// The lines for `tool` against `f` (the server found, or the default address where none runs). `key`: the server has a key (the lines name the variable, never the key); `linux`: Docker on Linux, whose containers
/// do not reach a server on this computer's own address except on the host's network.
pub fn settings(tool: &str, f: &Found, key: bool, linux: bool) -> Option<String> {
    let base = format!("http://{}:{}", f.host, f.port);
    let k = if key { "$GLYD_API_KEY" } else { "none" };
    let model = if f.model.is_empty() { "MODEL" } else { f.model.as_str() };
    let name = model.rsplit('/').next().unwrap_or(model);
    Some(match tool {
        "openai" => format!(
            "# OpenAI SDKs, LangChain (ChatOpenAI), aider, and the rest that read these:\nexport OPENAI_BASE_URL={base}/v1\nexport OPENAI_API_KEY={k}\n# the model name is any: the server answers with the one it runs ({model})\n# Python:  OpenAI(base_url=\"{base}/v1\", api_key=\"{k}\")\n"
        ),
        "anthropic" => format!(
            "# Claude Code and the Anthropic SDKs:\nexport ANTHROPIC_BASE_URL={base}\nexport ANTHROPIC_API_KEY={k}\nexport ANTHROPIC_MODEL={name}\nexport ANTHROPIC_SMALL_FAST_MODEL={name}\nclaude\n# Python:  Anthropic(base_url=\"{base}\", api_key=\"{k}\")\n"
        ),
        "ollama" => format!(
            "# the ollama libraries and programs that read OLLAMA_HOST:\nexport OLLAMA_HOST={base}\n# Python:  ollama.Client(host=\"{base}\"{})\n# LangChain: ChatOllama(base_url=\"{base}\", model=\"{model}\")\n# (a program that wants Ollama's own port: glyd serve ... --ollama-port 11434)\n",
            if key { ", headers={\"Authorization\": \"Bearer \" + os.environ[\"GLYD_API_KEY\"]}" } else { "" }
        ),
        "open-webui" => {
            let keys = if key { " -e OPENAI_API_KEY=$GLYD_API_KEY" } else { "" };
            if linux {
                // (a server on this computer's own address is reached from a container on the host's network, where it is `localhost`)
                format!(
                    "# Open WebUI at http://localhost:3000, on the host's network (the server listens on this computer's own address):\ndocker run -d --network=host -e PORT=3000 -e OLLAMA_BASE_URL={base} -e OPENAI_API_BASE_URL={base}/v1{keys} -v open-webui:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:main\n"
                )
            } else {
                format!(
                    "# Open WebUI at http://localhost:3000 (Docker Desktop reaches this computer's own address as host.docker.internal):\ndocker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -e OLLAMA_BASE_URL=http://host.docker.internal:{p} -e OPENAI_API_BASE_URL=http://host.docker.internal:{p}/v1{keys} -v open-webui:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:main\n",
                    p = f.port
                )
            }
        }
        "continue" => format!(
            "# ~/.continue/config.yaml (add under models):\nmodels:\n  - name: Glyd\n    provider: openai\n    model: {model}\n    apiBase: {base}/v1\n    apiKey: {}\n    roles: [chat, edit, apply]\n  - name: Glyd autocomplete\n    provider: ollama\n    model: {model}\n    apiBase: {base}\n    roles: [autocomplete]\n",
            if key { "YOUR_GLYD_API_KEY" } else { "none" }
        ),
        "curl" => format!(
            "curl -s {base}/v1/chat/completions -H 'Content-Type: application/json'{} -d '{{\"model\":\"{name}\",\"messages\":[{{\"role\":\"user\",\"content\":\"Say hi\"}}]}}'\n",
            if key { " -H \"Authorization: Bearer $GLYD_API_KEY\"" } else { "" }
        ),
        _ => return None,
    })
}

/// `glyd connect [TOOL] [--port N] [--host H]`: where a server runs on this computer and what each tool is given; the lines of TOOL alone where one is named.
pub fn connect(a: &Args) -> i32 {
    let host = a.flag("host").unwrap_or("127.0.0.1").to_string();
    let port = match a.flag("port").map(str::parse::<u16>) {
        None => None,
        Some(Ok(p)) => Some(p),
        Some(Err(_)) => {
            eprintln!("glyd: --port {}: not a port", a.flag("port").unwrap_or(""));
            return 1;
        }
    };
    let key = std::env::var("GLYD_API_KEY").ok().filter(|k| !k.is_empty()).or_else(|| std::env::var("VLLM_API_KEY").ok().filter(|k| !k.is_empty()));
    let tool = a.pos.get(1).map(String::as_str);
    if let Some(t) = tool.filter(|t| !TOOLS.iter().any(|(n, _)| n == t) && !["claude", "claude-code", "langchain", "aider", "openwebui"].contains(t)) {
        eprintln!("glyd: connect {t}: not a tool it knows. The ones it does:");
        for (n, what) in TOOLS {
            eprintln!("  {n:<11} {what}");
        }
        return 1;
    }
    let tool = tool.map(|t| match t {
        "claude" | "claude-code" => "anthropic",
        "langchain" | "aider" => "openai",
        "openwebui" => "open-webui",
        t => t,
    });
    let found = find(&host, port, key.as_deref());
    let f = match &found {
        Some(f) => f.clone(),
        None => {
            // (no server answered: the lines are for where `glyd serve` would put it)
            eprintln!(
                "glyd: no Glyd server answers at {host}:{} (start one: glyd serve MODEL). The lines below are for that address.",
                port.map_or(format!("{DEFAULT_PORT}-{} or {OLLAMA_PORT}", DEFAULT_PORT + 20), |p| p.to_string())
            );
            Found { host, port: port.unwrap_or(DEFAULT_PORT), model: String::new() }
        }
    };
    if found.is_some() {
        eprintln!("glyd: the server at http://{}:{} serves {}", f.host, f.port, f.model);
    }
    let linux = cfg!(target_os = "linux");
    let tools: Vec<&str> = tool.map_or(TOOLS.iter().map(|(n, _)| *n).collect(), |t| vec![t]);
    for t in tools {
        if let Some(text) = settings(t, &f, key.is_some(), linux) {
            println!("{text}");
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn found() -> Found {
        Found { host: "127.0.0.1".into(), port: 8001, model: "Qwen/Qwen3-8B".into() }
    }

    #[test]
    fn what_each_tool_is_given() {
        let f = found();
        let s = |t: &str, key: bool, linux: bool| settings(t, &f, key, linux).unwrap();
        assert!(s("openai", false, true).contains("OPENAI_BASE_URL=http://127.0.0.1:8001/v1") && s("openai", false, true).contains("OPENAI_API_KEY=none"));
        assert!(s("openai", true, true).contains("OPENAI_API_KEY=$GLYD_API_KEY"));
        let a = s("anthropic", false, true);
        assert!(a.contains("ANTHROPIC_BASE_URL=http://127.0.0.1:8001\n") && a.contains("ANTHROPIC_MODEL=Qwen3-8B"), "{a}");
        assert!(s("ollama", false, true).contains("OLLAMA_HOST=http://127.0.0.1:8001") && s("ollama", true, true).contains("headers="));
        // Open WebUI: on Linux the host's network (the server is on this computer's own address), elsewhere Docker Desktop's name for it
        let l = s("open-webui", true, true);
        assert!(
            l.contains("--network=host")
                && l.contains("OLLAMA_BASE_URL=http://127.0.0.1:8001")
                && l.contains("OPENAI_API_BASE_URL=http://127.0.0.1:8001/v1")
                && l.contains("OPENAI_API_KEY=$GLYD_API_KEY"),
            "{l}"
        );
        let d = s("open-webui", false, false);
        assert!(d.contains("-p 3000:8080") && d.contains("OLLAMA_BASE_URL=http://host.docker.internal:8001") && !d.contains("OPENAI_API_KEY"), "{d}");
        let c = s("continue", false, true);
        assert!(c.contains("apiBase: http://127.0.0.1:8001/v1") && c.contains("provider: ollama") && c.contains("roles: [autocomplete]"), "{c}");
        assert!(s("curl", true, true).contains("Authorization: Bearer $GLYD_API_KEY"));
        // the key itself is in none of them
        for (t, _) in TOOLS {
            assert!(!settings(t, &f, true, true).unwrap().contains("sk-"), "{t}");
        }
        assert!(settings("vim", &f, false, true).is_none());
        // no model known: a placeholder
        let none = Found { model: String::new(), ..found() };
        assert!(settings("continue", &none, false, true).unwrap().contains("model: MODEL"));
    }

    #[test]
    fn a_server_is_found_by_its_models_answer() {
        // a listener of the server's own answers /v1/models without an engine: `owned_by` glyd is what marks it
        let (tx, _rx) = std::sync::mpsc::channel();
        let addr = crate::server::listen("127.0.0.1:0", "Qwen/Qwen3-8B", 4096, Some("k".to_string()), true, tx).unwrap();
        assert_eq!(find("127.0.0.1", Some(addr.port()), Some("k")), Some(Found { host: "127.0.0.1".into(), port: addr.port(), model: "Qwen/Qwen3-8B".into() }));
        // wrong key, or nothing there: none
        assert_eq!(find("127.0.0.1", Some(addr.port()), Some("wrong")), None);
        let free = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        assert_eq!(find("127.0.0.1", Some(free), None), None);
    }
}
