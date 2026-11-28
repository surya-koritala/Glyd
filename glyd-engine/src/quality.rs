//! The quality bar of the lossy levels (R14, docs/quality.md). One suite, run on any OpenAI-compatible server: the next-token distributions after fixed
//! prefixes of four fixed texts (prose, a program, instructions, a dialogue), and three small task sets made here from a seed, whose answers are known
//! without a dataset (reasoning: word problems; code: what a short Python program prints; long context: one locker's code among several in 4K to 32K
//! tokens of filler). A lossy pack's quality card is its result against the lossless level's of the same model, held to the level's bar
//! (src/levels.rs `Bar`): publish or refuse. `ship` writes the card into the pack and does not upload a lossy pack without a passing one; `fit` shows
//! the cards' figures per level.
//!
//!   glyd-engine quality MODEL[:LEVEL] | --against URL  [--out FILE] [--label TEXT] [--tasks reasoning,code,long] [--n 40] [--long 4k,16k] [--engine NAME] [--bits B]
//!                                                      [--ctx N] [--no-draft] [--users N]
//!       measures one server (this program's `serve MODEL`, or any server at URL: llama.cpp, vLLM, SGLang, Ollama): `glyd-quality/1`.
//!   glyd-engine quality --card REF.json CAND.json [--level L | --bar L] [--out FILE]
//!       the candidate's card against the lossless reference (`glyd-quality-card/1`); exit 0 publish, 1 refuse. `--bar L` holds a candidate that is no
//!       Glyd level (a GGUF, an AWQ or FP8 checkpoint) to level L's bar, for comparisons at equal bits.
//!   glyd-engine quality --table CARD.json [CARD.json ...]
//!       the cards side by side by bits, as Markdown: the published comparison.

use crate::args::Args;
use crate::bench::{self, Rng};
use crate::ffi::{err, Error, Result};
use crate::levels;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};

pub const SCHEMA: &str = "glyd-quality/1";
pub const CARD_SCHEMA: &str = "glyd-quality-card/1";

/// The prefixes a text's next-token distributions are taken after (bench's `--quality` takes the same number of the prose text).
const PREFIXES: usize = 48;
/// The seed of the task sets: the same problems every time.
const SEED: u64 = 0x51_4C_31_34;

const CODE_TEXT: &str = r##"import json
from pathlib import Path


def load_orders(path):
    """Read one order a line and skip the lines that do not parse."""
    orders = []
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            order = json.loads(line)
        except json.JSONDecodeError:
            print(f"line {number}: not an order, skipped")
            continue
        orders.append(order)
    return orders


def totals_by_customer(orders):
    totals = {}
    for order in orders:
        customer = order.get("customer", "unknown")
        amount = sum(item["price"] * item.get("quantity", 1) for item in order["items"])
        totals[customer] = totals.get(customer, 0) + amount
    return totals


def best_customers(totals, count=3):
    ranked = sorted(totals.items(), key=lambda pair: pair[1], reverse=True)
    return [name for name, _ in ranked[:count]]


def main():
    orders = load_orders("orders.jsonl")
    totals = totals_by_customer(orders)
    for name in best_customers(totals):
        print(f"{name}: {totals[name]:.2f}")
    if not orders:
        print("no orders yet")


if __name__ == "__main__":
    main()
"##;

const HOWTO_TEXT: &str =
    "To repot a houseplant, start a day early by watering it well, so that the roots slide out of the old pot in one piece. Choose a new pot \
that is about two to five centimetres wider than the old one and has at least one drainage hole; a pot much larger than that holds water the roots cannot \
reach, and the soil stays wet for too long. Cover the hole with a piece of mesh or a shard of broken clay, then add a layer of fresh potting mix, deep \
enough that the top of the root ball will sit about two centimetres below the rim. Turn the old pot on its side, hold the plant at the base of its stems, \
and tap the pot until the plant comes free. If the roots circle the bottom in a tight mat, loosen them gently with your fingers or make three or four \
shallow cuts down the sides with a clean knife. Set the plant in the middle of the new pot, fill the space around it with mix, and press it down lightly \
to close the larger air pockets without packing it hard. Water slowly until a little runs out of the drainage hole, wait ten minutes, and empty the saucer. \
Keep the plant out of direct sun for about a week while the roots settle, and do not feed it for four to six weeks: fresh mix already holds enough \
nutrients, and fertilizer can burn roots that were cut or bent. Most plants need repotting every one to two years, in spring or early summer, when they \
grow fastest. Signs that it is time include roots growing out of the drainage hole, water running straight through the pot, and leaves that stay small \
even though the plant is fed and gets enough light.";

const DIALOGUE_TEXT: &str =
    "Visitor: Hello, I am trying to find a book about the history of maps, but the catalogue lists three and I do not know which one to pick. \
Librarian: Happy to help. Are you more interested in how maps were made, or in how people used them? Visitor: How they were made, mostly. My daughter \
asked me how anyone could draw a coastline before there were aeroplanes, and I realised I had no idea. Librarian: Then the green one on the second \
shelf is the best start. It follows surveyors from the first measured baselines to triangulation, with drawings of the instruments they carried. \
Visitor: Is it very technical? She is eleven. Librarian: The chapters on instruments are, but the first half reads like a set of travel stories, and the \
pictures carry a lot of it. You could read those together and leave the mathematics for later. Visitor: That sounds right. Can I take it home today? \
Librarian: Yes, with your card you can borrow it for three weeks, and you can renew it twice online unless someone else has asked for it. Visitor: \
And the other two books? Librarian: One is a large atlas of old maps, which has to stay in the reading room, and the other is about map projections, \
which is interesting but much drier. If she enjoys the first one, come back and I will show you the atlas; children usually love finding their own \
town on a map that is four hundred years old. Visitor: Thank you, that is very kind. Librarian: You are welcome. The self-service machine is by the \
door, or I can check it out for you here if you prefer.";

/// The fixed texts of the KL, by name.
pub fn texts() -> [(&'static str, &'static str); 4] {
    [("prose", bench::PASSAGE), ("code", CODE_TEXT), ("howto", HOWTO_TEXT), ("dialogue", DIALOGUE_TEXT)]
}

// ---- the task sets, made from the seed

/// One question with a known answer (a whole number, compared as GSM8K's are: the last number of the reply).
#[derive(Clone, Debug, PartialEq)]
pub struct Item {
    pub id: String,
    pub prompt: String,
    pub want: i64,
    pub max_tokens: usize,
}

fn pick(r: &mut Rng, lo: i64, hi: i64) -> i64 {
    lo + (r.next() % (hi - lo + 1) as u64) as i64
}

/// `n` word problems of two or three steps, five kinds in turn.
pub fn reasoning(n: usize) -> Vec<Item> {
    let mut r = Rng(SEED ^ 0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|i| {
            let (q, want) = match i % 5 {
                0 => {
                    let (a, b) = (pick(&mut r, 3, 9), pick(&mut r, 12, 24));
                    let c = pick(&mut r, 10, a * b / 2);
                    let d = pick(&mut r, 5, a * b - c - 1);
                    (format!("A bakery bakes {a} trays of {b} rolls each morning. It sells {c} rolls before noon and {d} after noon. How many rolls are left at the end of the day?"), a * b - c - d)
                }
                1 => {
                    let (p1, d1, p2, d2, left) = (pick(&mut r, 10, 30), pick(&mut r, 3, 7), pick(&mut r, 20, 45), pick(&mut r, 2, 5), pick(&mut r, 15, 120));
                    let book = p1 * d1 + p2 * d2 + left;
                    (format!("Maya's book has {book} pages. She reads {p1} pages a day for {d1} days, then {p2} pages a day for {d2} days. How many pages does she still have to read?"), left)
                }
                2 => {
                    let (s, e, w) = (pick(&mut r, 20, 90), pick(&mut r, 5, 15), pick(&mut r, 4, 10));
                    let x = pick(&mut r, 10, s);
                    (format!("Tom has {s} dollars. He earns {e} dollars a week for {w} weeks, and then spends {x} dollars on a gift. How many dollars does he have now?"), s + e * w - x)
                }
                3 => {
                    let (k, p) = (pick(&mut r, 6, 12), pick(&mut r, 15, 40));
                    let n = pick(&mut r, 50, 200);
                    (format!("A shop packs {n} bottles into crates of {k}. Only full crates are sold, at {p} dollars a crate. How many dollars does the shop get for the full crates?"), n / k * p)
                }
                _ => {
                    let (v1, t1, v2, t2) = (pick(&mut r, 40, 90), pick(&mut r, 2, 5), pick(&mut r, 30, 70), pick(&mut r, 1, 4));
                    (format!("A truck drives {t1} hours at {v1} kilometres an hour, then {t2} hours at {v2} kilometres an hour. How many kilometres does it drive in all?"), v1 * t1 + v2 * t2)
                }
            };
            Item { id: format!("reasoning-{i}"), prompt: format!("{q}\nThink step by step, then end with the final answer as a number alone on the last line."), want, max_tokens: 512 }
        })
        .collect()
}

fn collatz_steps(mut n: i64) -> i64 {
    let mut steps = 0;
    while n != 1 {
        n = if n % 2 == 0 { n / 2 } else { 3 * n + 1 };
        steps += 1;
    }
    steps
}

/// `n` short Python programs, six kinds in turn, and what each prints (worked out here).
pub fn code(n: usize) -> Vec<Item> {
    let mut r = Rng(SEED ^ 0x2545_F491_4F6C_DD1D);
    let words: Vec<&str> = bench::WORDS.split_whitespace().collect();
    (0..n)
        .map(|i| {
            let (src, want) = match i % 6 {
                0 => {
                    let (a, k) = (pick(&mut r, 1, 20), pick(&mut r, 3, 7));
                    let b = a + pick(&mut r, 15, 40);
                    (format!("total = 0\nfor i in range({a}, {b}):\n    if i % {k} == 0:\n        total += i\nprint(total)"), (a..b).filter(|i| i % k == 0).sum())
                }
                1 => {
                    let xs: Vec<i64> = (0..7).map(|_| pick(&mut r, 1, 30)).collect();
                    let (m, t) = (pick(&mut r, 2, 5), pick(&mut r, 8, 20));
                    let list = xs.iter().map(i64::to_string).collect::<Vec<_>>().join(", ");
                    (format!("xs = [{list}]\nys = [x * {m} for x in xs if x > {t}]\nprint(sum(ys))"), xs.iter().filter(|&&x| x > t).map(|x| x * m).sum())
                }
                2 => {
                    let n0 = pick(&mut r, 6, 40);
                    (format!("n = {n0}\nsteps = 0\nwhile n != 1:\n    n = n // 2 if n % 2 == 0 else 3 * n + 1\n    steps += 1\nprint(steps)"), collatz_steps(n0))
                }
                3 => {
                    let ws: Vec<&str> = (0..8).map(|_| words[(r.next() % words.len() as u64) as usize]).collect();
                    let c = &ws[(r.next() % ws.len() as u64) as usize][..1];
                    let want = ws.iter().filter(|w| w.starts_with(c)).map(|w| w.len() as i64).sum();
                    (format!("words = \"{}\".split()\nprint(sum(len(w) for w in words if w.startswith(\"{c}\")))", ws.join(" ")), want)
                }
                4 => {
                    let (a, b, c, d) = (pick(&mut r, 2, 6), pick(&mut r, 1, 9), pick(&mut r, 1, 9), pick(&mut r, 1, 9));
                    let f = |x: i64| x * a + b;
                    (format!("def f(x):\n    return x * {a} + {b}\n\n\ndef g(x):\n    return f(x) - {c}\n\n\nprint(g(f({d})))"), f(f(d)) - c)
                }
                _ => {
                    let n0 = pick(&mut r, 1000, 999_999);
                    let digits: i64 = n0.to_string().bytes().map(|b| (b - b'0') as i64).sum();
                    (format!("n = {n0}\ntotal = 0\nwhile n > 0:\n    total += n % 10\n    n //= 10\nprint(total)"), digits)
                }
            };
            Item {
                id: format!("code-{i}"),
                prompt: format!("What does this Python program print?\n\n```python\n{src}\n```\n\nWork it out step by step, then end with what it prints alone on the last line."),
                want,
                max_tokens: 512,
            }
        })
        .collect()
}

const COLORS: [&str; 8] = ["red", "blue", "green", "yellow", "orange", "purple", "silver", "white"];
/// Where in the filler the asked locker's code is, as a share of the words.
const DEPTHS: [f64; 4] = [0.1, 0.35, 0.65, 0.9];

/// For each length (tokens) and each depth of `DEPTHS`: filler words with four lockers' codes among them, and the question of one locker's, whose code is at that depth.
pub fn long(lengths: &[usize]) -> Vec<Item> {
    let mut r = Rng(SEED ^ 0x0123_4567_89AB_CDEF);
    let words: Vec<&str> = bench::WORDS.split_whitespace().collect();
    let mut out = Vec::new();
    for &len in lengths {
        // (the filler's words are about 1.09 tokens each, as bench's long prompts are)
        let n = (len.saturating_sub(120) as f64 / 1.09) as usize;
        for d in 0..DEPTHS.len() {
            let mut body: Vec<String> = (0..n).map(|_| words[(r.next() % words.len() as u64) as usize].to_string()).collect();
            let start = (r.next() % COLORS.len() as u64) as usize;
            let mut asked = (String::new(), 0i64);
            // (the four lockers at the four depths, the asked one at this item's: inserted from the back so that the earlier places do not move)
            for k in (0..DEPTHS.len()).rev() {
                let color = COLORS[(start + k) % COLORS.len()];
                let code = pick(&mut r, 100_000, 999_999);
                let at = ((DEPTHS[(d + k) % DEPTHS.len()] * n as f64) as usize).min(body.len());
                body.insert(at, format!("The code of the {color} locker is {code}."));
                if k == 0 {
                    asked = (color.to_string(), code);
                }
            }
            out.push(Item {
                id: format!("long-{}k-{d}", len / 1024),
                prompt: format!(
                    "Below is a long list of words with the codes of a few lockers among them.\n\n{}\n\nWhat is the code of the {} locker? Reply with the six-digit number alone.",
                    body.join(" "),
                    asked.0
                ),
                want: asked.1,
                max_tokens: 24,
            });
        }
    }
    out
}

/// The items' replies from the server, eight at once (as GSM8K's are), in the items' order: (the reply's last number, whether it is the answer).
fn answer(agent: &ureq::Agent, url: &str, model: &str, items: &[Item]) -> Vec<(Option<String>, bool)> {
    let next = AtomicUsize::new(0);
    let out: Vec<std::sync::Mutex<(Option<String>, bool)>> = items.iter().map(|_| std::sync::Mutex::new((None, false))).collect();
    std::thread::scope(|s| {
        for _ in 0..8 {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(it) = items.get(i) else { break };
                let got = bench::ask(agent, url, model, &it.prompt, it.max_tokens).and_then(|t| bench::last_number(&t));
                let ok = got.as_deref().is_some_and(|g| bench::same_number(g, &it.want.to_string()));
                if let Ok(mut o) = out[i].lock() {
                    *o = (got, ok);
                }
            });
        }
    });
    out.into_iter().map(|m| m.into_inner().unwrap_or((None, false))).collect()
}

fn task_result(items: &[Item], answers: &[(Option<String>, bool)]) -> Value {
    let correct = answers.iter().filter(|a| a.1).count();
    json!({"n": items.len(), "correct": correct, "accuracy": if items.is_empty() { Value::Null } else { json!(correct as f64 / items.len() as f64) },
        "items": items.iter().zip(answers).map(|(it, a)| json!({"id": it.id, "want": it.want, "got": a.0, "ok": a.1})).collect::<Vec<_>>()})
}

/// The tasks a run asks for (`--tasks`, default all three), the problems of each (`--n`, 40), and the long-context lengths (`--long`, 4k and 16k; `--long 0` none).
fn task_sets(a: &Args) -> Result<Vec<(&'static str, Vec<Item>)>> {
    let n: usize = a.num("n", 40)?;
    let want: Vec<&str> = a.flag("tasks").map_or(vec!["reasoning", "code", "long"], |t| t.split(',').map(str::trim).collect());
    let lengths: Vec<usize> = match a.flag("long").unwrap_or("4k,16k") {
        "0" | "none" => Vec::new(),
        l => l
            .split(',')
            .map(|k| bench::kind_tokens(k.trim()).and_then(|t| t.ok_or_else(|| Error(format!("--long {k}: 4k, 16k, 32k or NNNt")))))
            .collect::<Result<_>>()?,
    };
    // (`--long 0`: no long-context task at all, not one of no problems)
    want.iter()
        .filter(|t| **t != "long" || !lengths.is_empty())
        .map(|t| {
            Ok(match *t {
                "reasoning" => ("reasoning", reasoning(n)),
                "code" => ("code", code(n)),
                "long" => ("long", long(&lengths)),
                other => return err(format!("--tasks {other}: reasoning, code or long")),
            })
        })
        .collect()
}

/// `quality MODEL | --against URL`: the suite on one server, its result (`glyd-quality/1`).
pub fn measure(a: &Args) -> Result<Value> {
    let against = a.flag("against").map(|u| u.trim_end_matches('/').to_string());
    let model_arg = a.pos.get(1).map(String::as_str);
    if against.is_none() && model_arg.is_none() {
        return err("quality MODEL[:LEVEL] (this program's `serve` is measured), or quality --against URL (any OpenAI-compatible server)");
    }
    let sets = task_sets(a)?;
    // (the server this program starts holds the longest item and its reply, where --ctx does not say otherwise; eight requests at once)
    let longest =
        sets.iter().flat_map(|(_, items)| items.iter()).filter(|i| i.id.starts_with("long")).map(|i| i.prompt.split_whitespace().count() * 11 / 10).max();
    let sa = match longest.filter(|_| !a.has("ctx")) {
        Some(t) => a.with(&["ctx"], (t + 1024).next_multiple_of(1024).to_string()),
        None => a.clone(),
    };
    let server = match &against {
        Some(_) => None,
        None => Some(bench::start_server(model_arg.unwrap_or(""), &sa, 8)?),
    };
    let url = against.clone().unwrap_or_else(|| server.as_ref().map(|s| s.url.clone()).unwrap_or_default());
    let models = bench::get_json(&format!("{url}/v1/models")).ok_or_else(|| Error(format!("{url}/v1/models: no answer")))?;
    let model = a
        .flag("model")
        .map(str::to_string)
        .or_else(|| models["data"][0]["id"].as_str().map(str::to_string))
        .ok_or_else(|| Error("the server lists no model".into()))?;
    let info = bench::get_json(&format!("{url}/v1/glyd/info"));
    let level = info.as_ref().and_then(|i| i["level"]["effective"].as_str().map(str::to_string));
    let bits = a.flag("bits").and_then(|b| b.parse::<f64>().ok()).or_else(|| level.as_deref().and_then(levels::layout_of).map(crate::plan::sizing::bits));
    let agent = bench::agent();
    let quiet = a.has("quiet");
    let mut next_token = serde_json::Map::new();
    for (name, text) in texts() {
        let rows = bench::next_token_rows_of(&agent, &url, &model, text, PREFIXES);
        let empty = rows["rows"].as_array().map_or(0, |r| r.iter().filter(|x| x.as_array().is_none_or(|a| a.is_empty())).count());
        if !quiet {
            eprintln!(
                "quality: {name}: {} prefixes{}",
                rows["prefixes"],
                if empty > 0 { format!(", {empty} without log-probabilities") } else { String::new() }
            );
        }
        next_token.insert(name.to_string(), rows);
    }
    let mut tasks = serde_json::Map::new();
    for (name, items) in &sets {
        let r = task_result(items, &answer(&agent, &url, &model, items));
        if !quiet {
            eprintln!("quality: {name}: {} of {} right", r["correct"], r["n"]);
        }
        tasks.insert(name.to_string(), r);
    }
    let lengths: Vec<String> = a.flag("long").unwrap_or("4k,16k").split(',').map(str::to_string).collect();
    let result = json!({
        "schema": SCHEMA,
        "label": a.flag("label").unwrap_or(""),
        "recorded_unix": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()),
        "machine": bench::machine(),
        "model": {"id": model, "argument": model_arg, "level": level, "bits": bits, "engine": a.flag("engine").unwrap_or(if against.is_none() { "glyd" } else { "" })},
        "versions": {"glyd_engine": env!("CARGO_PKG_VERSION"), "git": bench::git_sha(), "server": info.as_ref().and_then(|i| i["build"]["version"].as_str())},
        "server": {"against": against, "started_by_quality": server.is_some()},
        "config": {"texts": texts().iter().map(|t| t.0).collect::<Vec<_>>(), "prefixes": PREFIXES, "top": 20, "seed": SEED,
            "tasks": sets.iter().map(|(n, items)| json!([n, items.len()])).collect::<Vec<_>>(), "long": lengths},
        "next_token": next_token,
        "tasks": tasks,
    });
    drop(server);
    Ok(result)
}

// ---- the card

fn load(path: &str, schema: &str) -> Result<Value> {
    let v: Value =
        serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| Error(format!("{path}: {e}")))?).map_err(|e| Error(format!("{path}: {e}")))?;
    if v["schema"] != schema {
        return err(format!("{path}: schema {}, where {schema} is read", v["schema"]));
    }
    Ok(v)
}

/// A candidate's quality card against the lossless reference of the same model, held to `bar_level`'s bar (`level` names what the candidate is: a Glyd level,
/// or a format given with --level for another engine's). Refused, with the reasons: a figure past the bar, a text or task the candidate lacks, a reference that is not lossless.
pub fn card(reference: &Value, cand: &Value, level: &str, bar_level: &str) -> Result<Value> {
    let bar =
        levels::by_name(bar_level).and_then(|l| l.bar).ok_or_else(|| Error(format!("{bar_level}: no quality bar (kestrel, linnet, wren or hummingbird)")))?;
    let mut reasons = Vec::new();
    if let Some(l) = reference["model"]["level"].as_str().filter(|l| *l != "penguin") {
        return err(format!("the reference is at {l}: a card is taken against the lossless level (penguin)"));
    }
    if reference["model"]["level"].is_null() {
        reasons.push("the reference server did not say its level: it must be the lossless one".to_string());
    }
    let (mut sum, mut n, mut agree, mut max) = (0.0f64, 0u64, 0u64, 0.0f64);
    let mut by_text = serde_json::Map::new();
    for (name, _) in texts() {
        let k = bench::kl_vs_ref(&reference["next_token"][name], &cand["next_token"][name]);
        let p = k["prefixes"].as_u64().unwrap_or(0);
        let want = reference["next_token"][name]["prefixes"].as_u64().unwrap_or(0);
        if p == 0 || p < want {
            reasons.push(format!("{name}: {p} of {want} prefixes have log-probabilities in both"));
        }
        if p > 0 {
            let (m, x, a) = (k["mean"].as_f64().unwrap_or(0.0), k["max"].as_f64().unwrap_or(0.0), k["top1_agree"].as_u64().unwrap_or(0));
            (sum, n, agree, max) = (sum + m * p as f64, n + p, agree + a, max.max(x));
            by_text.insert(name.into(), json!({"mean": m, "max": x, "top1": a as f64 / p as f64, "prefixes": p}));
        }
    }
    let r6 = |x: f64| (x * 1e6).round() / 1e6;
    let (mean, top1) = if n > 0 { (r6(sum / n as f64), r6(agree as f64 / n as f64)) } else { (f64::NAN, 0.0) };
    if n > 0 && mean > bar.kl_mean {
        reasons.push(format!("KL {mean:.4} is past the bar's {}", bar.kl_mean));
    }
    if n > 0 && top1 < bar.top1 {
        reasons.push(format!("the best token is the lossless level's at {:.1}% of the prefixes, under the bar's {:.0}%", top1 * 100.0, bar.top1 * 100.0));
    }
    let mut tasks = serde_json::Map::new();
    for (name, rt) in reference["tasks"].as_object().into_iter().flatten() {
        let ct = &cand["tasks"][name];
        let (ra, ca) = (rt["accuracy"].as_f64(), ct["accuracy"].as_f64());
        let Some(ca) = ca.filter(|_| ct["n"] == rt["n"]) else {
            reasons.push(format!("{name}: not measured on the candidate, or on other problems"));
            continue;
        };
        let ra = ra.unwrap_or(0.0);
        let same = rt["items"].as_array().into_iter().flatten().zip(ct["items"].as_array().into_iter().flatten()).filter(|(x, y)| x["got"] == y["got"]).count();
        let drop = r6(ra - ca);
        if drop > bar.task_drop + 1e-9 {
            reasons.push(format!(
                "{name}: {:.1}% right against the lossless level's {:.1}%, a drop past the bar's {:.1} points",
                ca * 100.0,
                ra * 100.0,
                bar.task_drop * 100.0
            ));
        }
        tasks.insert(name.clone(), json!({"n": rt["n"], "reference": ra, "candidate": ca, "drop": drop, "same_answer": same}));
    }
    Ok(json!({
        "schema": CARD_SCHEMA,
        "model": cand["model"]["id"], "level": level, "bits": cand["model"]["bits"], "engine": cand["model"]["engine"],
        "reference": {"model": reference["model"]["id"], "level": reference["model"]["level"], "label": reference["label"]},
        "measured": {"gpu": cand["machine"]["gpu"], "recorded_unix": cand["recorded_unix"], "glyd_engine": cand["versions"]["glyd_engine"], "git": cand["versions"]["git"]},
        "kl": {"mean": if n > 0 { json!(mean) } else { Value::Null }, "max": r6(max), "top1": top1, "prefixes": n, "by_text": by_text},
        "tasks": tasks,
        "bar": {"level": bar_level, "kl_mean": bar.kl_mean, "top1": bar.top1, "task_drop": bar.task_drop},
        "verdict": if reasons.is_empty() { "publish" } else { "refuse" },
        "reasons": reasons,
    }))
}

/// Whether a lossy pack may be published: its card (DIR/quality.json) is there, is the suite's, names the pack's level and says publish. The lossless level needs none.
pub fn publishable(level: &str, pack: &std::path::Path) -> Result<()> {
    if levels::by_name(level).is_none_or(|l| l.lossless) {
        return Ok(());
    }
    let f = pack.join("quality.json");
    let c = load(&f.to_string_lossy(), CARD_SCHEMA)
        .map_err(|e| Error(format!("{e}: a {level} pack is published with its quality card (glyd-engine quality --card, or ship's quality step)")))?;
    if c["level"] != level {
        return err(format!("{}: the card is of {}, the pack is {level}", f.display(), c["level"]));
    }
    if c["verdict"] != "publish" {
        let why: Vec<&str> = c["reasons"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
        return err(format!("{}: the {level} pack is under its quality bar: {}", f.display(), why.join("; ")));
    }
    Ok(())
}

/// One line of a card's figures: KL, top-1 agreement, each task's right answers against the lossless level's, the verdict.
pub fn line(c: &Value) -> String {
    let mut parts = vec![format!(
        "KL {} · top-1 {:.1}%",
        c["kl"]["mean"].as_f64().map_or("-".into(), |m| format!("{m:.4}")),
        c["kl"]["top1"].as_f64().unwrap_or(0.0) * 100.0
    )];
    for (name, t) in c["tasks"].as_object().into_iter().flatten() {
        let n = t["n"].as_f64().unwrap_or(0.0);
        parts.push(format!("{name} {:.0}/{n:.0} (lossless {:.0})", t["candidate"].as_f64().unwrap_or(0.0) * n, t["reference"].as_f64().unwrap_or(0.0) * n));
    }
    parts.push(c["verdict"].as_str().unwrap_or("?").to_string());
    parts.join(" · ")
}

/// `quality --table CARDS`: the cards by bits, as a Markdown table (the comparison at equal bits).
pub fn table(cards: &[Value]) -> String {
    let mut cs: Vec<&Value> = cards.iter().collect();
    cs.sort_by(|a, b| b["bits"].as_f64().unwrap_or(0.0).total_cmp(&a["bits"].as_f64().unwrap_or(0.0)));
    let tasks: Vec<String> = cs.iter().flat_map(|c| c["tasks"].as_object().into_iter().flatten().map(|(k, _)| k.clone())).fold(Vec::new(), |mut v, k| {
        if !v.contains(&k) {
            v.push(k);
        }
        v
    });
    let mut s = format!("| engine | level or format | bits a weight | KL from lossless | top-1 | {} | held to | verdict |\n", tasks.join(" | "));
    s += &format!("| :-- | :-- | --: | --: | --: | {}| :-- | :-- |\n", "--: | ".repeat(tasks.len()));
    for c in cs {
        let t: Vec<String> = tasks
            .iter()
            .map(|k| {
                let t = &c["tasks"][k];
                let n = t["n"].as_f64().unwrap_or(0.0);
                t["candidate"].as_f64().map_or("-".into(), |x| format!("{:.0}/{n:.0} ({:.0})", x * n, t["reference"].as_f64().unwrap_or(0.0) * n))
            })
            .collect();
        s += &format!(
            "| {} | {} | {} | {} | {:.1}% | {} | {} | {} |\n",
            c["engine"].as_str().filter(|e| !e.is_empty()).unwrap_or("?"),
            c["level"].as_str().unwrap_or("?"),
            c["bits"].as_f64().map_or("?".into(), |b| format!("{b:.2}")),
            c["kl"]["mean"].as_f64().map_or("-".into(), |m| format!("{m:.4}")),
            c["kl"]["top1"].as_f64().unwrap_or(0.0) * 100.0,
            t.join(" | "),
            c["bar"]["level"].as_str().unwrap_or("?"),
            c["verdict"].as_str().unwrap_or("?")
        );
    }
    s + "\n(tasks: right answers, the lossless level's in brackets)\n"
}

// ---- fit: the figures of a level

/// The cards of the pre-packed repos (cards/NAME-LEVEL.json), as the repository holds them.
const CARDS: &[(&str, &str)] = &[
    ("nemotron-3-hummingbird", include_str!("../../cards/nemotron-3-hummingbird.json")),
    ("Qwen3.5-0.8B-kestrel", include_str!("../../cards/Qwen3.5-0.8B-kestrel.json")),
    ("Qwen3.5-0.8B-linnet", include_str!("../../cards/Qwen3.5-0.8B-linnet.json")),
    ("Qwen3.5-27B-kestrel", include_str!("../../cards/Qwen3.5-27B-kestrel.json")),
    ("Qwen3.5-27B-linnet", include_str!("../../cards/Qwen3.5-27B-linnet.json")),
    ("Qwen3.5-2B-kestrel", include_str!("../../cards/Qwen3.5-2B-kestrel.json")),
    ("Qwen3.5-2B-linnet", include_str!("../../cards/Qwen3.5-2B-linnet.json")),
    ("Qwen3.5-4B-kestrel", include_str!("../../cards/Qwen3.5-4B-kestrel.json")),
    ("Qwen3.5-4B-linnet", include_str!("../../cards/Qwen3.5-4B-linnet.json")),
    ("Qwen3.5-9B-kestrel", include_str!("../../cards/Qwen3.5-9B-kestrel.json")),
    ("Qwen3.5-9B-penguin", include_str!("../../cards/Qwen3.5-9B-penguin.json")),
    ("Qwen3.5-9B-linnet", include_str!("../../cards/Qwen3.5-9B-linnet.json")),
    ("Qwen3.6-27B-kestrel", include_str!("../../cards/Qwen3.6-27B-kestrel.json")),
    ("Qwen3.6-27B-linnet", include_str!("../../cards/Qwen3.6-27B-linnet.json")),
    ("Qwen3.8-27B-kestrel", include_str!("../../cards/Qwen3.8-27B-kestrel.json")),
    ("Qwen3.8-27B-linnet", include_str!("../../cards/Qwen3.8-27B-linnet.json")),
];

/// A level's quality figures for MODEL where any were measured: the suite's card (glyd-quality-card/1) of a local pack (MODEL a directory of that level, or the
/// level's pre-packed repo in the Hub cache) or under a repository card's `quality`, else a repository card's earlier figures (KL from bf16 by kld_study.py,
/// GSM8K against bf16). (the source, a card of the suite or the earlier figures as a card-like object)
pub fn measured(model: &str, level: &str) -> Option<(String, Value)> {
    let repo = levels::packed_repo(model, level);
    let local = [Some(std::path::PathBuf::from(model)).filter(|p| p.is_dir()), glyd_gpu::save::Source::find(&repo).ok().map(|s| s.dir)];
    for dir in local.into_iter().flatten() {
        let f = dir.join("quality.json");
        if let Ok(c) = load(&f.to_string_lossy(), CARD_SCHEMA) {
            if c["level"] == level {
                return Some((f.display().to_string(), c));
            }
        }
    }
    let name = repo.strip_prefix("glyd/").unwrap_or(&repo);
    let (file, text) = CARDS.iter().find(|(n, _)| *n == name)?;
    let card: Value = serde_json::from_str(text).ok()?;
    let src = format!("cards/{file}.json");
    if card["quality"]["schema"] == CARD_SCHEMA {
        return Some((src, card["quality"].clone()));
    }
    let gsm = |k: &str| card[k].as_array().into_iter().flatten().find(|e| e["task_id"] == "gsm8k").and_then(|e| e["value"].as_f64());
    let kl = card["kl_bf16"].as_f64();
    (kl.is_some() || gsm("evals").is_some()).then(|| (src, json!({"kl_bf16": kl, "gsm8k": gsm("evals"), "gsm8k_bf16": gsm("evals_bf16")})))
}

/// `fit`'s lines of quality, one a level, and the same as JSON: the lossless level is the release's weights; a lossy one shows its card's figures, its earlier
/// figures, or that none were measured.
pub fn fit_block(model: &str, levels_shown: &[&str]) -> (String, Value) {
    let mut s = String::from("Quality against the lossless level (src/quality.rs; docs/quality.md):\n");
    let mut j = serde_json::Map::new();
    for &l in levels_shown {
        if levels::by_name(l).is_some_and(|x| x.lossless) {
            s += &format!("  {l:<12} lossless: every weight is the release's\n");
            j.insert(l.into(), json!({"lossless": true}));
            continue;
        }
        match measured(model, l) {
            Some((src, c)) if c["schema"] == CARD_SCHEMA => {
                s += &format!("  {l:<12} {} ({src})\n", line(&c));
                j.insert(l.into(), json!({"source": src, "card": c}));
            }
            Some((src, c)) => {
                let mut parts = Vec::new();
                if let Some(k) = c["kl_bf16"].as_f64() {
                    parts.push(format!("KL {k:.4} from bf16 (WikiText-2)"));
                }
                if let Some(g) = c["gsm8k"].as_f64() {
                    parts.push(format!("GSM8K {:.1}%{}", g * 100.0, c["gsm8k_bf16"].as_f64().map_or(String::new(), |b| format!(" (bf16 {:.1}%)", b * 100.0))));
                }
                s += &format!("  {l:<12} {} ({src}; measured before the suite, no verdict)\n", parts.join(" · "));
                j.insert(l.into(), json!({"source": src, "earlier": c}));
            }
            None => {
                s += &format!("  {l:<12} not measured\n");
                j.insert(l.into(), Value::Null);
            }
        }
    }
    (s, Value::Object(j))
}

// ---- the command

pub fn cmd(a: &Args) -> i32 {
    let r = (|| -> Result<i32> {
        let out = |v: &Value| -> Result<()> {
            let text = serde_json::to_string_pretty(v).unwrap_or_default() + "\n";
            match a.flag("out") {
                Some(f) => std::fs::write(f, text).map_err(|e| Error(format!("{f}: {e}"))),
                None => {
                    print!("{text}");
                    Ok(())
                }
            }
        };
        if a.has("card") {
            let (reference, cand) = (load(a.need(1, "REF.json")?, SCHEMA)?, load(a.need(2, "CAND.json")?, SCHEMA)?);
            let level = a.flag("level").map(str::to_string).or_else(|| cand["model"]["level"].as_str().map(str::to_string)).ok_or_else(|| {
                Error("the candidate did not say its level: --level NAME (a format: Q4_K_M, AWQ, FP8) and --bar LEVEL (the level it is held to)".into())
            })?;
            let c = card(&reference, &cand, &level, a.flag("bar").unwrap_or(&level))?;
            eprintln!("{} {level}: {}", c["model"].as_str().unwrap_or("?"), line(&c));
            for r in c["reasons"].as_array().into_iter().flatten() {
                eprintln!("  {}", r.as_str().unwrap_or(""));
            }
            out(&c)?;
            return Ok(if c["verdict"] == "publish" { 0 } else { 1 });
        }
        if a.has("table") {
            let cards: Vec<Value> = a.pos[1..].iter().map(|p| load(p, CARD_SCHEMA)).collect::<Result<_>>()?;
            if cards.is_empty() {
                return err("quality --table CARD.json [CARD.json ...]");
            }
            print!("{}", table(&cards));
            return Ok(0);
        }
        out(&measure(a)?)?;
        Ok(0)
    })();
    match r {
        Ok(code) => code,
        Err(e) => {
            eprintln!("glyd-engine quality: {e}");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_task_sets_are_the_same_every_time_and_their_answers_are_right() {
        assert_eq!(reasoning(40), reasoning(40));
        assert_eq!(code(12), code(12));
        let r = reasoning(10);
        assert_eq!(r.len(), 10);
        assert!(r.iter().all(|i| i.want >= 0 && i.prompt.contains("last line")), "{r:?}");
        // (a problem of each kind, worked out by hand from its text)
        let crates = &r[3];
        let nums: Vec<i64> = crates.prompt.split(|c: char| !c.is_ascii_digit()).filter_map(|w| w.parse().ok()).collect();
        assert_eq!(crates.want, nums[0] / nums[1] * nums[2], "{}", crates.prompt);
        let c = code(6);
        assert!(c[2].prompt.contains("while n != 1"));
        assert_eq!(collatz_steps(6), 8);
        assert_eq!(collatz_steps(27), 111);
        // the digits' sum
        let n: i64 = c[5].prompt.split("n = ").nth(1).unwrap().split('\n').next().unwrap().parse().unwrap();
        assert_eq!(c[5].want, n.to_string().bytes().map(|b| (b - b'0') as i64).sum::<i64>());
        // distinct problems
        let mut ps: Vec<&str> = r.iter().map(|i| i.prompt.as_str()).collect();
        ps.dedup();
        assert_eq!(ps.len(), r.len());
    }

    #[test]
    fn a_long_item_holds_four_lockers_and_asks_for_the_one_at_its_depth() {
        let items = long(&[4096]);
        assert_eq!(items.len(), DEPTHS.len());
        for (d, it) in items.iter().enumerate() {
            assert_eq!(it.prompt.matches("The code of the ").count(), 4, "{}", it.id);
            let color = it.prompt.rsplit("What is the code of the ").next().unwrap().split(' ').next().unwrap();
            let fact = format!("The code of the {color} locker is {}.", it.want);
            let at = it.prompt.find(&fact).expect("the asked locker's code is in the text") as f64;
            let share = at / it.prompt.len() as f64;
            assert!((share - DEPTHS[d]).abs() < 0.08, "{}: at {share:.2}, depth {}", it.id, DEPTHS[d]);
            // (about 4K tokens: 1.09 tokens a word)
            let words = it.prompt.split_whitespace().count() as f64;
            assert!((words * 1.09 - 4096.0).abs() < 300.0, "{words} words");
        }
        assert_eq!(long(&[4096, 16384]).len(), 2 * DEPTHS.len());
    }

    #[test]
    fn the_texts_prefixes_keep_their_lines() {
        for (name, text) in texts() {
            let (cuts, first, _) = bench::prefixes(text, PREFIXES);
            assert_eq!(cuts.len(), PREFIXES, "{name}");
            assert!(first == 24 && cuts.windows(2).all(|w| w[0].len() < w[1].len()), "{name}");
            assert!(text.starts_with(cuts[PREFIXES - 1]) && cuts.iter().all(|c| !c.ends_with(char::is_whitespace)), "{name}");
        }
        let (cuts, _, _) = bench::prefixes(CODE_TEXT, PREFIXES);
        assert!(cuts[10].contains("\n\n\ndef load_orders(path):\n"), "{}", cuts[10]);
        // the prose text's prefixes are bench's --quality's (the words joined by one space)
        let words: Vec<&str> = bench::PASSAGE.split_whitespace().collect();
        let (cuts, first, step) = bench::prefixes(bench::PASSAGE, PREFIXES);
        assert_eq!(cuts[5], words[..first + 5 * step].join(" "));
    }

    fn result(level: Option<&str>, shift: f64, right: usize) -> Value {
        // (each text's prefixes: the same distribution, the best token moved to the second where `shift` is large)
        let row = |s: f64| json!([["a", (0.6 - s).ln()], ["b", (0.3 + s).ln()], ["c", 0.05f64.ln()]]);
        let rows = |k: usize| (0..PREFIXES).map(|i| if i < k { row(shift) } else { row(0.0) }).collect::<Vec<_>>();
        let next: serde_json::Map<String, Value> =
            texts().iter().map(|(n, _)| (n.to_string(), json!({"prefixes": PREFIXES, "rows": rows(PREFIXES)}))).collect();
        let items = |n: usize, ok: usize| (0..n).map(|i| json!({"id": i, "want": 1, "got": if i < ok { "1" } else { "2" }, "ok": i < ok})).collect::<Vec<_>>();
        json!({"schema": SCHEMA, "model": {"id": "m", "level": level, "bits": 4.53, "engine": "glyd"}, "machine": {"gpu": "g"}, "versions": {}, "next_token": next,
            "tasks": {"reasoning": {"n": 40, "accuracy": right as f64 / 40.0, "items": items(40, right)}}})
    }

    #[test]
    fn a_card_publishes_within_the_bar_and_refuses_past_it() {
        let reference = result(Some("penguin"), 0.0, 36);
        // the same distributions, one answer fewer: within wren's bar
        let c = card(&reference, &result(Some("wren"), 0.0, 35), "wren", "wren").unwrap();
        assert_eq!(c["verdict"], "publish", "{c}");
        assert!(c["kl"]["mean"].as_f64().unwrap() < 1e-9 && c["kl"]["top1"] == 1.0 && c["kl"]["prefixes"] == 4 * PREFIXES as u64);
        // (35 right answers alike, and the 4 wrong ones the same wrong number)
        assert_eq!(c["tasks"]["reasoning"]["same_answer"], 39);
        // the best token moved at every prefix: refused on KL and top-1
        let moved = card(&reference, &result(Some("wren"), 0.35, 35), "wren", "wren").unwrap();
        assert_eq!(moved["verdict"], "refuse");
        let why = moved["reasons"].to_string();
        assert!(why.contains("past the bar") && why.contains("best token"), "{why}");
        // a task that drops past the level's points: refused (6 of 40 is 15 points; kestrel allows 5)
        let dropped = card(&reference, &result(Some("kestrel"), 0.0, 30), "kestrel", "kestrel").unwrap();
        assert_eq!(dropped["verdict"], "refuse");
        assert!(dropped["reasons"].to_string().contains("reasoning"), "{dropped}");
        // a task the candidate lacks is not a pass
        let mut lacking = result(Some("wren"), 0.0, 36);
        lacking["tasks"] = json!({});
        assert_eq!(card(&reference, &lacking, "wren", "wren").unwrap()["verdict"], "refuse");
        // a lossy reference is no reference; another engine's format is held to a level's bar by name
        assert!(card(&result(Some("kestrel"), 0.0, 36), &reference, "wren", "wren").is_err());
        let gguf = card(&reference, &result(None, 0.0, 36), "Q4_K_M", "wren").unwrap();
        assert_eq!((gguf["level"].as_str(), gguf["bar"]["level"].as_str(), gguf["verdict"].as_str()), (Some("Q4_K_M"), Some("wren"), Some("publish")));
        assert!(card(&reference, &result(None, 0.0, 36), "Q4_K_M", "penguin").is_err());
    }

    #[test]
    fn a_lossy_pack_is_published_only_with_a_passing_card_of_its_level() {
        let dir = std::env::temp_dir().join(format!("glyd-quality-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // the lossless level needs no card; a lossy one without one is refused
        assert!(publishable("penguin", &dir).is_ok());
        assert!(publishable("linnet", &dir).unwrap_err().msg.contains("quality card"));
        let reference = result(Some("penguin"), 0.0, 36);
        let write = |c: &Value| std::fs::write(dir.join("quality.json"), c.to_string()).unwrap();
        write(&card(&reference, &result(Some("linnet"), 0.0, 36), "linnet", "linnet").unwrap());
        assert!(publishable("linnet", &dir).is_ok());
        assert!(publishable("wren", &dir).unwrap_err().msg.contains("the card is of"));
        write(&card(&reference, &result(Some("linnet"), 0.35, 36), "linnet", "linnet").unwrap());
        assert!(publishable("linnet", &dir).unwrap_err().msg.contains("under its quality bar"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn fit_shows_each_levels_figures_or_that_none_were_measured() {
        let (text, j) = fit_block("Qwen/Qwen3.5-9B", &["penguin", "kestrel", "linnet", "wren"]);
        assert!(text.contains("penguin      lossless"), "{text}");
        assert!(text.contains("linnet       KL 0.0373 from bf16 (WikiText-2) · GSM8K 94.8% (bf16 95.5%) (cards/Qwen3.5-9B-linnet.json"), "{text}");
        assert!(text.contains("wren         not measured"), "{text}");
        assert_eq!(j["penguin"]["lossless"], true);
        assert!(j["wren"].is_null() && j["kestrel"]["source"] == "cards/Qwen3.5-9B-kestrel.json");
        // a suite card in a pack's directory is that level's figures
        let dir = std::env::temp_dir().join(format!("glyd-quality-fit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let c = card(&result(Some("penguin"), 0.0, 36), &result(Some("wren"), 0.0, 35), "wren", "wren").unwrap();
        std::fs::write(dir.join("quality.json"), c.to_string()).unwrap();
        let (text, j) = fit_block(&dir.to_string_lossy(), &["wren"]);
        assert!(text.contains("KL 0.0000 · top-1 100.0% · reasoning 35/40 (lossless 36) · publish"), "{text}");
        assert_eq!(j["wren"]["card"]["verdict"], "publish");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn every_card_in_the_repository_is_one_fit_reads() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../cards");
        let mut files: Vec<String> =
            std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()?.file_name().to_str()?.strip_suffix(".json").map(str::to_string)).collect();
        files.sort();
        let mut known: Vec<String> = CARDS.iter().map(|c| c.0.to_string()).collect();
        known.sort();
        assert_eq!(files, known, "cards/*.json and quality.rs CARDS differ: add the new card to CARDS");
        assert!(CARDS.iter().all(|(_, t)| serde_json::from_str::<Value>(t).is_ok()));
    }

    #[test]
    fn the_table_orders_the_cards_by_bits() {
        let reference = result(Some("penguin"), 0.0, 36);
        let mut a = card(&reference, &result(Some("wren"), 0.0, 35), "wren", "wren").unwrap();
        let mut b = card(&reference, &result(None, 0.0, 34), "Q6_K", "kestrel").unwrap();
        (a["bits"], b["bits"], b["engine"]) = (json!(4.53), json!(6.56), json!("llama.cpp"));
        let t = table(&[a, b]);
        let (q6, wren) = (t.find("| llama.cpp | Q6_K | 6.56 |").unwrap(), t.find("| glyd | wren | 4.53 |").unwrap());
        assert!(q6 < wren && t.contains("34/40 (36)") && t.contains("| kestrel | publish |"), "{t}");
    }

    /// The suite against a server as any engine's is measured (--against): the real handler and scheduler over the made-up model. Every text's prefixes have
    /// log-probabilities, every problem is asked and answered, and the server's result against its own is a card of no divergence and the same answers.
    #[test]
    fn the_suite_runs_against_a_server_and_a_server_against_itself_diverges_nowhere() {
        let addr = crate::server::tests::serve_fake(crate::sched::tests::Fake::new(4096, 64), None, 8);
        let words = ["quality", "--against", &format!("http://{addr}"), "--n", "3", "--long", "0", "--quiet", "--engine", "fake"];
        let mut r = measure(&Args::from(words.iter().map(|w| w.to_string()))).unwrap();
        assert_eq!(r["schema"], SCHEMA);
        for (name, _) in texts() {
            let rows = r["next_token"][name]["rows"].as_array().unwrap();
            assert!(rows.len() == PREFIXES && rows.iter().all(|x| !x.as_array().unwrap().is_empty()), "{name}: {}", r["next_token"][name]);
        }
        assert_eq!((r["tasks"]["reasoning"]["n"].as_u64(), r["tasks"]["code"]["n"].as_u64()), (Some(3), Some(3)));
        assert!(r["tasks"]["long"].is_null() && r["model"]["engine"] == "fake");
        assert!(r["tasks"]["code"]["items"].as_array().unwrap().iter().all(|i| i["want"].is_i64()));
        r["model"]["level"] = json!("penguin");
        let c = card(&r, &r, "wren", "wren").unwrap();
        assert_eq!(c["kl"]["mean"].as_f64(), Some(0.0), "{c}");
        assert_eq!((c["kl"]["top1"].as_f64(), c["tasks"]["code"]["same_answer"].as_u64(), c["verdict"].as_str()), (Some(1.0), Some(3), Some("publish")), "{c}");
    }

    #[test]
    fn the_suite_reads_both_shapes_of_log_probabilities() {
        let legacy = json!({"choices": [{"logprobs": {"top_logprobs": [{" the": -0.5, "a": -1.5}]}}]});
        let chat = json!({"choices": [{"logprobs": {"content": [{"token": " the", "logprob": -0.5, "top_logprobs": [{"id": 1, "token": " the", "logprob": -0.5}, {"id": 2, "token": "a", "logprob": -1.5}]}]}}]});
        let (x, y) = (bench::first_top(&legacy).unwrap(), bench::first_top(&chat).unwrap());
        let sorted = |mut v: Vec<Value>| {
            v.sort_by(|a, b| a.to_string().cmp(&b.to_string()));
            v
        };
        assert_eq!(sorted(x), sorted(y));
        assert!(bench::first_top(&json!({"choices": [{"text": "x"}]})).is_none());
    }
}
