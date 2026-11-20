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

/// What the server's refusal of a missing key says (`server::error_body`'s 401): the mark of a Glyd server among the programs on these ports.
const KEY_REFUSAL: &str = "the server's key is needed as";

/// Whether `host` is this computer's own (the only place a key is sent).
fn is_loopback(host: &str) -> bool {
    ["127.0.0.1", "localhost", "::1", "[::1]"].contains(&host)
}

/// A model name that is safe to write into a shell line: letters, digits and `. _ : / - @ +` (a repo's or a path's name); else it is not used.
fn plain_name(s: &str) -> bool {
    !s.is_empty() && s.len() <= 200 && s.bytes().all(|b| b.is_ascii_alphanumeric() || b"._:/-@+".contains(&b))
}

/// The server on `host`: at `port`, or where none is given at the first of 8000 to 8020, then 11434, that answers as Glyd does (`/v1/models` with `owned_by` glyd). `None` where there is none. Each port is asked without
/// the key first, and the key goes only to a port whose refusal is a Glyd server's, on this computer's own address: another program on these ports is never sent it.
pub fn find(host: &str, port: Option<u16>, key: Option<&str>) -> Option<Found> {
    let ports: Vec<u16> = port.map_or((DEFAULT_PORT..=DEFAULT_PORT + 20).chain([OLLAMA_PORT]).collect(), |p| vec![p]);
    for p in ports {
        let Some(addr) = (host, p).to_socket_addrs().ok().and_then(|mut a| a.next()) else { continue };
        let at = format!("{host}:{p}");
        let body = match get(addr, &at, "/v1/models", None) {
            Some((200, body)) => body,
            Some((401, refusal)) if refusal.contains(KEY_REFUSAL) && is_loopback(host) => match key.and_then(|k| get(addr, &at, "/v1/models", Some(k))) {
                Some((200, body)) => body,
                _ => continue,
            },
            _ => continue,
        };
        let Ok(v) = glyd_gpu::json::parse(&body) else { continue };
        let first = v.get("data").and_then(|d| d.as_array()).and_then(|d| d.first());
        if let Some(m) = first.filter(|m| m.get("owned_by").and_then(|o| o.as_str()) == Some("glyd")) {
            // (a name that is not plain is the server's to say, not a shell's to read: the lines then say MODEL)
            let model = m.get("id").and_then(|i| i.as_str()).filter(|n| plain_name(n)).unwrap_or("").to_string();
            return Some(Found { host: host.to_string(), port: p, model });
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
    if host.is_empty() || !host.bytes().all(|b| b.is_ascii_alphanumeric() || b".-:[]".contains(&b)) {
        eprintln!("glyd: --host {host}: not an address");
        return 1;
    }
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
        // wrong key, none given, or nothing there: none
        assert_eq!(find("127.0.0.1", Some(addr.port()), Some("wrong")), None);
        assert_eq!(find("127.0.0.1", Some(addr.port()), None), None);
        let free = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        assert_eq!(find("127.0.0.1", Some(free), None), None);
    }

    /// The key goes only to a server that says it is Glyd's: a program on a port `connect` looks at that answers with another refusal (or any answer) never sees it.
    #[test]
    fn the_key_is_not_sent_to_another_program() {
        let other = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = other.local_addr().unwrap().port();
        let seen = std::thread::spawn(move || {
            let mut all = String::new();
            for _ in 0..3 {
                let Ok((mut c, _)) = other.accept() else { break };
                let mut buf = [0u8; 2048];
                let n = c.read(&mut buf).unwrap_or(0);
                all.push_str(&String::from_utf8_lossy(&buf[..n]));
                let _ = c.write_all(b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 12\r\nConnection: close\r\n\r\nUnauthorized");
            }
            all
        });
        assert_eq!(find("127.0.0.1", Some(port), Some("sk-secret-key")), None);
        // (a connection that ends the accepting loop)
        let _ = std::net::TcpStream::connect(("127.0.0.1", port));
        let _ = std::net::TcpStream::connect(("127.0.0.1", port));
        let all = seen.join().unwrap();
        assert!(all.contains("GET /v1/models") && !all.contains("sk-secret-key") && !all.to_ascii_lowercase().contains("authorization"), "{all}");
    }

    #[test]
    fn only_plain_names_reach_a_shell_line() {
        for ok in ["Qwen/Qwen3-8B", "Qwen/Qwen3.5-9B:swift", "/home/u/models/Qwen3-8B", "a_b@c+d"] {
            assert!(plain_name(ok), "{ok}");
        }
        for bad in ["", "x'; rm -rf ~ #", "a b", "$(id)", "`id`", "a\nb", "x\"y", "a;b", "a|b", "a&b", "a>b"] {
            assert!(!plain_name(bad), "{bad:?}");
        }
        assert!(is_loopback("127.0.0.1") && is_loopback("localhost") && !is_loopback("192.168.1.5") && !is_loopback("example.com"));
    }
}
