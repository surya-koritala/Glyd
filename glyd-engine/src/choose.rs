//! A family name with no size (`glyd run gemma4`) picks the size for this computer: the largest size whose plan reaches the speed and the context floors (P4b). The plan is `fit`'s own (src/plan.rs `planned`:
//! the level and placement `run` would load, the speed predicted from the memory's bandwidth), so the pick is the size `glyd fit gemma4:SIZE` shows reaching them; nothing here sizes a model itself. A size
//! given (`gemma4:31b`), with or without a level (`gemma4:31b:wren`), is taken as it is; a level alone (`gemma4:wren`) is the level asked of every size.
//!
//! The floors: 10 tokens a second (a reader's pace, with room), and 8,192 tokens of context (the plan's preferred floor: a chat that holds a conversation). Where no size reaches both, the fastest that fits is taken and
//! the line says so; where none fits, the refusal says why for each size.

use crate::args::Args;
use crate::models::{Entry, Family};

/// The least speed a pick reaches, tokens a second as the plan predicts them for one user and a short prompt.
pub const MIN_TOK_S: f64 = 10.0;
/// The least context a pick has.
pub const MIN_CONTEXT: usize = crate::plan::PREFERRED_CTX;

/// What the plan says of one size.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub size: String,
    pub repo: String,
    /// Parameters, in billions.
    pub params_b: f64,
    /// The level the plan takes for it.
    pub level: &'static str,
    pub context: usize,
    pub tok_s: f64,
    /// Whether the entry is the manifest's default (it wins a tie).
    pub default: bool,
}

impl Row {
    fn reaches(&self) -> bool {
        self.tok_s >= MIN_TOK_S && self.context >= MIN_CONTEXT
    }
}

/// The pick, and what the other sizes came to.
#[derive(Debug, PartialEq)]
pub struct Choice {
    pub picked: Row,
    /// Both floors are reached (else the fastest size that fits is taken).
    pub floors_met: bool,
    /// The other sizes, by size: what the plan gave them (`4 tokens/s`) or why they do not fit.
    pub others: Vec<(String, String)>,
}

fn short(r: &Row) -> String {
    format!("{:.0} tokens/s, {} context", r.tok_s, r.context)
}

/// The pick among each size's plan (`rows`: a size and its plan, or why it has none).
pub fn pick(rows: Vec<(String, Result<Row, String>)>) -> Result<Choice, String> {
    let fit: Vec<&Row> = rows.iter().filter_map(|(_, r)| r.as_ref().ok()).collect();
    let best_of = |from: Vec<&Row>, key: &dyn Fn(&Row) -> f64| -> Option<Row> {
        from.into_iter().max_by(|a, b| key(a).partial_cmp(&key(b)).unwrap_or(std::cmp::Ordering::Equal).then(a.default.cmp(&b.default))).cloned()
    };
    let (picked, floors_met) = match best_of(fit.iter().copied().filter(|r| r.reaches()).collect(), &|r| r.params_b) {
        Some(r) => (r, true),
        None => match best_of(fit, &|r| r.tok_s) {
            Some(r) => (r, false),
            None => {
                let why: Vec<String> = rows.iter().map(|(s, r)| format!("{s}: {}", r.as_ref().err().map_or("", String::as_str))).collect();
                return Err(why.join("; "));
            }
        },
    };
    let others = rows
        .into_iter()
        .filter(|(s, _)| *s != picked.size)
        .map(|(s, r)| {
            let said = match r {
                Ok(r) => short(&r),
                Err(why) => why,
            };
            (s, said)
        })
        .collect();
    Ok(Choice { picked, floors_met, others })
}

/// The one line that says what was chosen and how to choose otherwise.
pub fn explain(family: &str, c: &Choice, machine: &str) -> String {
    let p = &c.picked;
    let floors = format!("{MIN_TOK_S:.0} tokens/s and {MIN_CONTEXT} tokens of context");
    let others: Vec<String> = c.others.iter().map(|(s, w)| format!("{s}: {w}")).collect();
    let rest = if others.is_empty() { String::new() } else { format!("; the others: {}", others.join("; ")) };
    let why = if c.floors_met {
        format!("the largest size that gets {floors} on {machine}")
    } else {
        format!("no size gets {floors} on {machine}: the fastest that fits")
    };
    format!("{family} is {family}:{} ({}) at {}: {why}, about {}{rest}. To choose another: {family}:SIZE[:LEVEL]", p.size, p.repo, p.level, short(p))
}

/// The plan of every size of `fam` (at `tag`, a level asked, where there is one) on the machine `a` names (`--machine`, else this computer) with its `--context` and `--users`, through `fit`'s own
/// planner; the machine's name for the line.
pub fn rows(fam: &Family, tag: Option<&str>, a: &Args) -> (Vec<(String, Result<Row, String>)>, String) {
    let mut machine = String::new();
    let rows = fam
        .models
        .iter()
        .map(|e| {
            let model = match tag {
                Some(t) => format!("{}:{t}", e.repo),
                None => e.repo.clone(),
            };
            let mut words = vec!["fit".to_string(), model];
            for f in ["machine", "context", "users"] {
                if let Some(v) = a.flag(f) {
                    words.extend([format!("--{f}"), v.to_string()]);
                }
            }
            (e.size.clone(), row_of(e, &words, &mut machine))
        })
        .collect();
    (rows, machine)
}

fn row_of(e: &Entry, words: &[String], machine: &mut String) -> Result<Row, String> {
    let planned = crate::plan::planned(&Args::from(words.iter().cloned())).map_err(|r| r.what)?;
    if machine.is_empty() {
        *machine = planned.machine.devices.first().map_or_else(|| "this computer".to_string(), |d| d.name.clone());
    }
    // (the refusal `run` would give, the machine's first: a size `run` would refuse is no size here)
    if let Some(r) = planned.refusal {
        return Err(r.what);
    }
    match planned.plan.chosen {
        Some(c) => Ok(Row {
            size: e.size.clone(),
            repo: e.repo.clone(),
            params_b: planned.shape.params() / 1e9,
            level: c.level,
            context: c.context,
            tok_s: c.tok_s,
            default: e.default,
        }),
        None => Err(planned.refusal.map_or_else(|| "does not fit".to_string(), |r| r.what)),
    }
}

/// The size of `fam` for this computer, or the refusal; with the line that says it.
pub fn choose(fam: &Family, tag: Option<&str>, a: &Args) -> Result<(Choice, String), String> {
    let (rows, machine) = rows(fam, tag, a);
    let c = pick(rows).map_err(|why| format!("{}: no size fits {}: {why}", fam.name, if machine.is_empty() { "this computer" } else { &machine }))?;
    let line = explain(&fam.name, &c, if machine.is_empty() { "this computer" } else { &machine });
    Ok((c, line))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(size: &str, params_b: f64, tok_s: f64, context: usize, default: bool) -> (String, Result<Row, String>) {
        (size.to_string(), Ok(Row { size: size.to_string(), repo: format!("o/{size}"), params_b, level: "kestrel", context, tok_s, default }))
    }

    fn no(size: &str, why: &str) -> (String, Result<Row, String>) {
        (size.to_string(), Err(why.to_string()))
    }

    /// The largest size that reaches both floors; the others are named with what they came to.
    #[test]
    fn the_largest_size_that_reaches_the_floors() {
        let c = pick(vec![row("4b", 4.0, 90.0, 32768, false), row("12b", 12.0, 38.0, 32768, true), row("31b", 31.0, 4.0, 8192, false)]).unwrap();
        assert_eq!((c.picked.size.as_str(), c.floors_met), ("12b", true));
        assert_eq!(c.others, [("4b".to_string(), "90 tokens/s, 32768 context".to_string()), ("31b".to_string(), "4 tokens/s, 8192 context".to_string())]);
        // a size too slow, one with too little context, one that does not fit at all: each is passed over
        let c =
            pick(vec![row("4b", 4.0, 90.0, 32768, false), row("12b", 12.0, 38.0, 4096, true), row("31b", 31.0, 9.9, 32768, false), no("70b", "does not fit")])
                .unwrap();
        assert_eq!(c.picked.size, "4b");
        // exactly on the floors is reached
        assert!(pick(vec![row("12b", 12.0, MIN_TOK_S, MIN_CONTEXT, true)]).unwrap().floors_met);
        // a tie in size: the manifest's default
        assert_eq!(pick(vec![row("a", 8.0, 50.0, 8192, false), row("b", 8.0, 50.0, 8192, true)]).unwrap().picked.size, "b");
    }

    /// Where no size reaches the floors the fastest that fits is taken, and the line says so; where none fits it is a refusal that says why for each.
    #[test]
    fn short_of_the_floors_the_fastest_that_fits_and_else_a_refusal() {
        let c = pick(vec![row("12b", 12.0, 6.0, 4096, true), row("31b", 31.0, 2.0, 4096, false)]).unwrap();
        assert_eq!((c.picked.size.as_str(), c.floors_met), ("12b", false));
        assert!(explain("gemma4", &c, "an RTX 4060 Laptop")
            .contains("no size gets 10 tokens/s and 8192 tokens of context on an RTX 4060 Laptop: the fastest that fits"));
        let e = pick(vec![no("12b", "12b needs 14 GiB"), no("31b", "31b needs 30 GiB")]).unwrap_err();
        assert_eq!(e, "12b: 12b needs 14 GiB; 31b: 31b needs 30 GiB");
    }

    /// One line that says the name, the size, the repo, the level, why, the speed and context, and how to choose otherwise.
    #[test]
    fn the_line_says_what_was_chosen_and_how_to_choose_otherwise() {
        let c = pick(vec![row("12b", 12.0, 38.4, 32768, true), row("31b", 31.0, 4.0, 8192, false)]).unwrap();
        let line = explain("gemma4", &c, "NVIDIA GeForce RTX 4080 SUPER");
        assert_eq!(
            line,
            "gemma4 is gemma4:12b (o/12b) at kestrel: the largest size that gets 10 tokens/s and 8192 tokens of context on NVIDIA GeForce RTX 4080 SUPER, about 38 tokens/s, 32768 context; the others: 31b: 4 tokens/s, 8192 context. To choose another: gemma4:SIZE[:LEVEL]"
        );
        assert!(!line.contains('\n'));
    }
}
