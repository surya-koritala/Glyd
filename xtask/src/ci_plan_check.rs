//! The GPU check that the plan is what loads: `glyd-engine fit MODEL --json` (the plan, before anything is loaded) against the report a run of the same model
//! wrote (GLYD_PLAN_REPORT: the plan the loader took and what the load used of the GPU).
//!
//!     cargo xtask ci-plan-check FIT.json LOADED.json
//!
//! This is scripts/ci_plan_check.py in Rust (the script through v0.35.0): the same lines, the same verdict.

use crate::json::{Json, Num};
use crate::{fail, py, usage, Res};
use std::path::Path;

/// What the check prints, and whether the plan is what loaded.
fn check(fit: &Json, loaded: &Json) -> Res<(String, bool)> {
    fn key<'a>(v: &'a Json, of: &str, k: &str) -> Res<&'a Json> {
        v.get(k).ok_or_else(|| fail(format!("{of} has no {}", py::repr(k))))
    }
    fn num(v: &Json, of: &str, k: &str) -> Res<Num> {
        key(v, of, k)?.num().ok_or_else(|| fail(format!("{of}: {k} is not a number")))
    }
    // (a / b - 1, as Python divides: by nothing is its error)
    fn off(a: Num, b: Num, what: &str) -> Res<f64> {
        if b.f() == 0.0 {
            return Err(fail(format!("{what}: division by zero")));
        }
        Ok((a.f() / b.f() - 1.0).abs())
    }
    let (plan, used) = (key(loaded, "the report", "plan")?, key(loaded, "the report", "loaded")?);
    let mut bad = Vec::new();

    // the plan fit printed and the plan run loaded: the same level, context, users and placement (the weights within 3%: fit sizes a model it has not
    // downloaded from its config, run from the files it has), the same reserve and work
    for k in ["schema", "family", "level", "context", "users", "chunk", "reserve_bytes", "work_bytes"] {
        let (f, p) = (key(fit, "fit", k)?, key(plan, "the plan", k)?);
        if !f.same(p) {
            bad.push(format!("{k}: fit {}, run {}", f.text(), p.text()));
        }
    }
    for k in ["weights_bytes", "kv_bytes"] {
        let (f, p) = (num(fit, "fit", k)?, num(plan, "the plan", k)?);
        if off(f, p, k)? > 0.03 {
            bad.push(format!("{k}: fit {f}, run {p}"));
        }
    }
    let (f, p) = (key(fit, "fit", "experts")?, key(plan, "the plan", "experts")?);
    if !f.same(p) {
        bad.push(format!("experts: fit {}, run {}", f.text(), p.text()));
    }
    let (f, p) = (fit.get("spread").unwrap_or(&Json::Null), plan.get("spread").unwrap_or(&Json::Null));
    if !f.same(p) {
        bad.push(format!("spread over the GPUs: fit {}, run {}", f.text(), p.text()));
    }

    // what the loader allocated against the plan: the weights the engine counts, and the memory the load took (the plan's total without its reserve)
    // (the multi-GPU route and Ibis do not count their weights apart: 0)
    let (weights, took) = (num(used, "the load", "weights_bytes")?, num(used, "the load", "gpu_used_bytes")?);
    let (on_gpu, total) = (num(plan, "the plan", "weights_on_gpu_bytes")?, num(plan, "the plan", "gpu_total_bytes")?);
    let (reserve, free) = (num(plan, "the plan", "reserve_bytes")?, num(plan, "the plan", "free_bytes")?);
    if weights.f() != 0.0 && off(weights, on_gpu, "weights")? > 0.03 {
        bad.push(format!("weights: planned {on_gpu}, loaded {weights}"));
    }
    if took.gt(total.sub(reserve)) {
        bad.push(format!("the load took {took} of the GPU, the plan has {} besides its reserve", total.sub(reserve)));
    }
    if total.gt(free) {
        bad.push(format!("the plan's {total} are more than the {free} free"));
    }

    let gib = |n: Num| n.f() / (1u64 << 30) as f64;
    let mut out = format!(
        "{} {}, {} tokens: planned {:.2} GiB of weights and {:.2} GiB of GPU memory; loaded {:.2} GiB of weights, the GPU lost {:.2} GiB ({:.2} GiB were free)\n",
        key(plan, "the plan", "family")?.text(),
        key(plan, "the plan", "level")?.text(),
        key(plan, "the plan", "context")?.text(),
        gib(on_gpu),
        gib(total.sub(reserve)),
        gib(weights),
        gib(took),
        gib(free)
    );
    if !bad.is_empty() {
        out += &format!("the plan is not what loaded:\n  {}\n", bad.join("\n  "));
    }
    Ok((out, bad.is_empty()))
}

pub fn main(args: &[String], out: &mut String) -> Res<()> {
    let [fit, loaded] = args else { return Err(usage("usage: cargo xtask ci-plan-check FIT.json LOADED.json")) };
    let read = |p: &String| Json::parse(&py::read(Path::new(p))?).map_err(|e| fail(format!("{p}: {}", e.msg)));
    let (said, ok) = check(&read(fit)?, &read(loaded)?)?;
    *out += &said;
    if ok {
        Ok(())
    } else {
        Err(fail(""))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record;

    /// tests/checks/plan-*.txt: what scripts/ci_plan_check.py printed for these two files, and its exit code (recorded from it once).
    #[test]
    fn does_what_the_script_did() {
        let cases = record::cases("checks", "plan-");
        assert!(cases.len() >= 8, "{} cases", cases.len());
        for (name, c) in cases {
            let json = |f: &str| Json::parse(c.file(f)).unwrap();
            match check(&json("fit.json"), &json("loaded.json")) {
                Ok((said, ok)) => assert_eq!((said.as_str(), i32::from(!ok)), (c.one("STDOUT"), c.exit()), "{name}"),
                Err(e) => assert!(
                    c.exit() == 1 && c.one("STDOUT").is_empty() && c.one("STDERR").contains("Error"),
                    "{name}: {} where the script printed {}",
                    e.msg,
                    c.one("STDOUT")
                ),
            }
        }
    }
}
