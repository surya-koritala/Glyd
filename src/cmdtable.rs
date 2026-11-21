//! The commands of `glyd` that are about models, and the flags they share, as one table: `glyd --help`'s command list, `glyd help COMMAND`, the man page (`glyd man`,
//! docs/glyd.1) and the refusal of a flag a command does not take are all made from it. The engine (glyd-engine/src/entry.rs) includes this file by its path, so the program
//! that has the engine linked and the one that has not say the same. It has no dependencies.

/// A flag that more than one command takes, with what it means everywhere it is taken.
pub struct Flag {
    pub name: &'static str,
    /// The value it takes, empty for a switch.
    pub value: &'static str,
    pub text: &'static str,
}

/// One command.
pub struct Cmd {
    pub name: &'static str,
    /// The words after the name, as a synopsis (`MODEL[:LEVEL]`).
    pub args: &'static str,
    pub summary: &'static str,
    /// What it says that the summary does not.
    pub detail: &'static str,
    /// The shared flags it takes (names in `FLAGS`).
    pub flags: &'static [&'static str],
}

pub const FLAGS: &[Flag] = &[
    Flag { name: "context", value: "N", text: "the context window in tokens" },
    Flag { name: "users", value: "N", text: "how many users the server holds at once" },
    Flag { name: "devices", value: "LIST", text: "the GPUs to use, as the driver numbers them (0,1)" },
    Flag { name: "machine", value: "NAME|FILE", text: "plan for another computer: a preset (rtx4080s, h100, ...) or a JSON file, not the one this runs on" },
    Flag { name: "json", value: "", text: "the answer as JSON on the standard output, for a program to read" },
    Flag { name: "port", value: "N", text: "the port (a port in use is an error, not a move, where it is the service's)" },
    Flag { name: "host", value: "ADDRESS", text: "the address to listen on (127.0.0.1 unless said; another address opens it to the network)" },
    Flag { name: "all", value: "", text: "everything, not only what is downloaded or loaded" },
    Flag { name: "keep-alive", value: "TIME", text: "how long an idle model stays loaded: 90s, 5m, 2h, 0, never" },
    Flag { name: "max-loaded", value: "N", text: "how many models the service holds at once" },
];

pub const CMDS: &[Cmd] = &[
    Cmd {
        name: "run",
        args: "MODEL[:LEVEL]",
        summary: "chat with a model in the terminal and at one URL",
        detail: "MODEL is a short name (gemma4, qwen3.8:27b), a Hub id (ORG/NAME), a directory or a GGUF file. LEVEL is penguin (lossless, the default), kestrel, swift, wren or hummingbird. \
                 --prompt TEXT answers once and exits. Where the service is running the chat goes through it.",
        flags: &["context", "users", "devices", "port", "host"],
    },
    Cmd {
        name: "serve",
        args: "MODEL[:LEVEL]",
        summary: "the OpenAI-compatible API and the chat page",
        detail: "Also the Anthropic and Ollama request shapes (docs/serve-api.md). GLYD_API_KEY, in the environment, guards /v1.",
        flags: &["context", "users", "devices", "port", "host"],
    },
    Cmd {
        name: "pull",
        args: "MODEL[:LEVEL]",
        summary: "download a model by name or Hub id",
        detail: "The level's pre-packed repo where there is one, else the release's own weights, which the first run packs. Downloads resume and run in parallel; a gated model says where to accept \
                 its licence and to run glyd login.",
        flags: &[],
    },
    Cmd { name: "ls", args: "", summary: "the models downloaded, with their level and size on disk", detail: "The packed copies made for this GPU are totalled below.", flags: &["json", "all"] },
    Cmd {
        name: "rm",
        args: "MODEL[:LEVEL]",
        summary: "take a model out of the cache, and its packed copies",
        detail: "A name with no level takes every level of the model. No other model's files are touched.",
        flags: &[],
    },
    Cmd {
        name: "ps",
        args: "",
        summary: "what the background service holds, and when each model unloads",
        detail: "",
        flags: &["json", "port"],
    },
    Cmd { name: "stop", args: "MODEL | --all", summary: "unload a model from the service now", detail: "The service stays. glyd service stop ends the service itself.", flags: &["all", "port"] },
    Cmd {
        name: "service",
        args: "[install | uninstall | status | stop]",
        summary: "the background service: models load on request and unload when idle",
        detail: "With no word it runs in the foreground; install makes it a user unit started at login (systemd, launchd; Windows runs it by hand). docs/service.md.",
        flags: &["port", "host", "keep-alive", "max-loaded", "context", "users", "json"],
    },
    Cmd {
        name: "fit",
        args: "MODEL[:LEVEL]",
        summary: "what a run would do on this computer, before anything is downloaded",
        detail: "The level, where the weights go, the context and the predicted tokens a second. --machine plans for another computer.",
        flags: &["context", "users", "devices", "machine", "json"],
    },
    Cmd {
        name: "doctor",
        args: "[--bundle [FILE]] [--no-logs]",
        summary: "what this computer has for running models, each check with the line that fixes it",
        detail: "--bundle writes the same as one file to attach to an issue.",
        flags: &["machine", "json"],
    },
    Cmd {
        name: "names",
        args: "[update]",
        summary: "the model names (gemma4, qwen3.8:27b, ...) and their Hub repos; update fetches the latest release's",
        detail: "A name that is not known is looked up once a day in the latest release's manifest (GLYD_NO_REFRESH=1 stops that); the fetched file is checked against the sha256 the release lists.",
        flags: &["json"],
    },
    Cmd { name: "login", args: "", summary: "save a Hugging Face token (for a gated model)", detail: "The token is kept in the Hub client's own file.", flags: &[] },
    Cmd {
        name: "supports",
        args: "MODEL",
        summary: "whether the engine runs a model, from its config alone",
        detail: "Exit 0 it does, 3 it does not.",
        flags: &["json"],
    },
    Cmd {
        name: "preflight",
        args: "[MODEL[:LEVEL]]",
        summary: "whether the engine will certainly run a model on this computer, before a download",
        detail: "Exit 0 it will, 3 it will not (and why), 1 that could not be told.",
        flags: &["context", "users", "json"],
    },
    Cmd { name: "prepare", args: "MODEL[:LEVEL]", summary: "download a model and pack it for this GPU", detail: "The KV cache's tables are left for the first run.", flags: &["context"] },
    Cmd { name: "costs", args: "MODEL[:LEVEL] [--predict]", summary: "a model's cost profile and predicted decode speed", detail: "docs/cost-profile.md.", flags: &["context"] },
    Cmd { name: "rates", args: "[--dir DIR]", summary: "this computer's memory tiers, measured", detail: "The GPU's memory, the link to it, the main memory and the disk under DIR.", flags: &[] },
    Cmd {
        name: "bench",
        args: "MODEL[:LEVEL] [--suite gate|bar|full]",
        summary: "the benchmark harness: fixed prompts at fixed concurrencies, the speed as JSON",
        detail: "docs/bench.md. --compare BASE CAND is the regression gate; --bar is the speed bar of a GPU.",
        flags: &["context", "users", "machine", "json"],
    },
    Cmd {
        name: "connect",
        args: "[TOOL]",
        summary: "what to give a program so that it talks to the server on this computer",
        detail: "Lines to paste for the OpenAI SDKs, Claude Code, Ollama's clients, Open WebUI, Continue and curl; nothing is changed.",
        flags: &[],
    },
    Cmd {
        name: "verify",
        args: "MODEL | PATH",
        summary: "check a model's packed weights bit for bit",
        detail: "A model name is the engine's; a file or directory of packed weights is glyd-gpu's.",
        flags: &[],
    },
    Cmd { name: "pack", args: "MODEL OUT", summary: "a model's weights for Glyd's GPU layouts, packed on the CPU", detail: "glyd-gpu's command.", flags: &[] },
    Cmd {
        name: "cluster",
        args: "",
        summary: "one model across computers (not yet: ROADMAP.md, M5)",
        detail: "Today that is the starlings program.",
        flags: &[],
    },
];

/// Exit statuses, one table for the help and the man page.
pub const EXIT: &[(&str, &str)] = &[("0", "done"), ("1", "an error"), ("3", "not supported (glyd supports MODEL: the engine does not run it)")];

pub fn cmd(name: &str) -> Option<&'static Cmd> {
    CMDS.iter().find(|c| c.name == name)
}

pub fn flag(name: &str) -> Option<&'static Flag> {
    FLAGS.iter().find(|f| f.name == name)
}

fn flag_usage(f: &Flag) -> String {
    if f.value.is_empty() {
        format!("--{}", f.name)
    } else {
        format!("--{} {}", f.name, f.value)
    }
}

/// The command lines of `glyd --help`: one for each command, its summary beside it.
pub fn overview() -> String {
    let left: Vec<String> = CMDS.iter().map(|c| format!("glyd {} {}", c.name, c.args).trim_end().to_string()).collect();
    let w = left.iter().map(String::len).max().unwrap_or(0);
    let mut s = String::new();
    for (l, c) in left.iter().zip(CMDS) {
        s.push_str(&format!("       {l:<w$}  {}\n", c.summary));
    }
    s
}

/// `glyd help COMMAND`: the command's synopsis, what it does and the shared flags it takes, each said as everywhere else.
pub fn help(name: &str) -> Option<String> {
    let c = cmd(name)?;
    let mut s = format!("Usage: glyd {} {}\n\n{}.\n", c.name, c.args, capital(c.summary));
    if !c.detail.is_empty() {
        s.push_str(&format!("\n{}\n", c.detail));
    }
    if !c.flags.is_empty() {
        s.push_str("\nFlags:\n");
        for n in c.flags {
            if let Some(f) = flag(n) {
                s.push_str(&format!("    {:<24}{}\n", flag_usage(f), f.text));
            }
        }
    }
    s.push_str(&format!("\nExit status: {}\n", EXIT.iter().map(|(c, t)| format!("{c} {t}")).collect::<Vec<_>>().join(", ")));
    Some(s)
}

fn capital(s: &str) -> String {
    let mut c = s.chars();
    c.next().map_or(String::new(), |f| f.to_uppercase().collect::<String>() + c.as_str())
}

/// The shared flags that `given` has and `command` does not take: the refusal says so and names the commands that do. Only `--json` and `--machine` are asked about (the others
/// are read by more of the engine's own commands than the table lists); a command that is not in the table is not asked.
pub fn refuse_flag(command: &str, given: &[&str]) -> Option<String> {
    let c = cmd(command)?;
    for f in ["json", "machine"] {
        if given.contains(&f) && !c.flags.contains(&f) {
            let takes: Vec<&str> = CMDS.iter().filter(|o| o.flags.contains(&f)).map(|o| o.name).collect();
            return Some(format!(
                "glyd {command}: --{f} is not a flag of this command ({}). Those that take it: {}",
                flag(f).map_or("", |x| x.text),
                takes.join(", ")
            ));
        }
    }
    None
}

fn roff(s: &str) -> String {
    s.replace('\\', "\\\\").replace('-', "\\-")
}

/// The man page, glyd(1), in roff (`glyd man`; docs/glyd.1 is this text, and a test holds them equal).
pub fn man() -> String {
    let mut s = String::new();
    s.push_str(".TH GLYD 1 \"\" \"Glyd\" \"User Commands\"\n.SH NAME\nglyd \\- LLM weights and KV cache compressed without loss, the engine that runs them, and file compression\n");
    s.push_str(".SH SYNOPSIS\n.B glyd\n.I COMMAND\n[ARGS]\n.br\n.B glyd\n[OPTIONS] [INPUT] [\\-o OUTPUT]\n");
    s.push_str(".SH DESCRIPTION\nWith a command, glyd runs, serves and manages language models on Glyd's own engine. With a file and no command it compresses or decompresses the file (glyd \\-\\-help lists the options).\n");
    s.push_str(".SH COMMANDS\n");
    for c in CMDS {
        s.push_str(&format!(
            ".TP\n.B glyd {}\n{}\n.br\n{}\n",
            roff(&format!("{} {}", c.name, c.args).trim_end()),
            roff(&capital(c.summary)) + ".",
            roff(c.detail)
        ));
        if !c.flags.is_empty() {
            let list: Vec<String> = c.flags.iter().filter_map(|n| flag(n)).map(flag_usage).collect();
            s.push_str(&format!(".br\nFlags: {}\n", roff(&list.join(", "))));
        }
    }
    s.push_str(".SH SHARED FLAGS\nThe same flag means the same wherever a command takes it.\n");
    for f in FLAGS {
        s.push_str(&format!(".TP\n.B {}\n{}\n", roff(&flag_usage(f)), roff(f.text)));
    }
    s.push_str(".SH EXIT STATUS\n");
    for (c, t) in EXIT {
        s.push_str(&format!(".TP\n.B {c}\n{}\n", roff(t)));
    }
    s.push_str(".SH ENVIRONMENT\n.TP\n.B GLYD_API_KEY\nThe API key that guards /v1 of glyd serve and the service.\n.TP\n.B GLYD_CACHE\nWhere packed models are kept.\n.TP\n.B HF_TOKEN\nA Hugging Face token.\n.TP\n.B GLYD_ENGINE\n0 keeps vLLM, 1 uses the engine on any GPU it runs on.\n.TP\n.B GLYD_SERVICE_PORT\nThe service's port, for glyd ps, stop and run.\n");
    s.push_str(".SH SEE ALSO\nhttps://github.com/surya-koritala/Glyd\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_flag_a_command_names_is_in_the_table_and_no_command_is_twice() {
        for c in CMDS {
            for f in c.flags {
                assert!(flag(f).is_some(), "{}: --{f} is not in FLAGS", c.name);
            }
            assert_eq!(CMDS.iter().filter(|o| o.name == c.name).count(), 1, "{}", c.name);
            assert!(!c.summary.is_empty() && !c.summary.ends_with('.'), "{}", c.name);
        }
        for f in FLAGS {
            assert_eq!(FLAGS.iter().filter(|o| o.name == f.name).count(), 1, "{}", f.name);
        }
    }

    #[test]
    fn a_command_that_does_not_take_json_or_machine_refuses_it_and_names_those_that_do() {
        let e = refuse_flag("run", &["json"]).unwrap();
        assert!(e.contains("glyd run: --json is not a flag") && e.contains("ls, ps, service, fit, doctor, supports, preflight, bench"), "{e}");
        assert!(refuse_flag("ps", &["machine"]).unwrap().contains("fit, doctor, bench"));
        assert_eq!(refuse_flag("fit", &["json", "machine", "context"]), None);
        assert_eq!(refuse_flag("bench", &["json", "machine"]), None);
        assert_eq!(refuse_flag("microbench", &["json"]), None, "a command that is not in the table is not asked");
        assert_eq!(refuse_flag("run", &["context", "port"]), None);
    }

    #[test]
    fn help_and_the_overview_say_what_the_table_says() {
        let h = help("fit").unwrap();
        assert!(
            h.starts_with("Usage: glyd fit MODEL[:LEVEL]\n")
                && h.contains("--machine NAME|FILE")
                && h.contains("--json")
                && h.contains("Exit status: 0 done, 1 an error, 3 not supported"),
            "{h}"
        );
        assert!(help("nothing").is_none());
        let o = overview();
        assert_eq!(o.lines().count(), CMDS.len());
        assert!(o.contains("glyd run MODEL[:LEVEL]") && o.contains("chat with a model in the terminal"), "{o}");
    }

    #[test]
    fn the_man_page_in_docs_is_the_tables() {
        // docs/glyd.1 is `glyd man`: regenerate it with  cargo run -p glyd --bin glyd -- man > docs/glyd.1
        assert_eq!(include_str!("../docs/glyd.1"), man(), "docs/glyd.1 is not what the table says: regenerate it with glyd man > docs/glyd.1");
    }

    #[test]
    fn the_man_page_has_every_command_and_every_flag() {
        let m = man();
        assert!(m.starts_with(".TH GLYD 1"), "{m}");
        for c in CMDS {
            assert!(m.contains(&format!(".B glyd {}", roff(&format!("{} {}", c.name, c.args).trim_end()))), "{}", c.name);
        }
        for f in FLAGS {
            assert!(m.contains(&roff(&flag_usage(f))), "{}", f.name);
        }
        assert!(m.contains(".SH EXIT STATUS"));
    }
}
