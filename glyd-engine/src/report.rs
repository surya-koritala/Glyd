//! The numbers the engine reports: memory, and a one-user decode benchmark.

use crate::ffi::Result;
use crate::model::Engine;
use std::time::Instant;

const GIB: f64 = (1u64 << 30) as f64;

pub fn memory(e: &Engine) -> String {
    let (free, total) = e.dev.mem_info().unwrap_or((0, 0));
    format!(
        "weights {:.3} GiB for {:.3} GiB of bf16 ({:.1}%); cache {} tokens in {:.3} GiB ({:.1} KiB a token); workspaces and activations {:.3} GiB; in use {:.3} of {:.3} GiB",
        e.w.bytes as f64 / GIB,
        e.w.bf16_bytes as f64 / GIB,
        100.0 * e.w.bytes as f64 / e.w.bf16_bytes as f64,
        e.kv.tokens(),
        e.kv.bytes as f64 / GIB,
        e.kv.bytes as f64 / e.kv.tokens() as f64 / 1024.0,
        e.work_bytes as f64 / GIB,
        (total - free) as f64 / GIB,
        total as f64 / GIB
    )
}

/// What the KV cache's write kernel counted: its extra memory taken at most, pages that needed it, raw and void pages.
pub fn kv_status(e: &Engine) -> String {
    match e.kv.status_words() {
        Ok(w) => format!(
            "KV cache: {} entries past a block's own room in {} blocks, {} blocks took units of the first extra memory (used at most {} of {} at once), {} blocks kept in the second (used at most {} of {} units), {} blocks lost",
            w[3], w[2], w[6], w[5], e.kv.chunks, w[10], w[9], e.kv.slabs, w[11]
        ),
        Err(x) => format!("KV status: {x}"),
    }
}

/// The bits a weight takes in each kind of matrix (the packs' bytes by the weights they hold).
pub fn bits(e: &Engine) -> String {
    e.w.kinds.iter().filter(|k| k.2 > 0).map(|(n, b, w)| format!("{n} {:.2}", *b as f64 * 8.0 / *w as f64)).collect::<Vec<_>>().join(", ")
}

/// A prompt of `n` token ids: uniform random over the vocabulary (as `vllm bench serve`'s random dataset), from a fixed seed.
pub fn random_prompt(n: usize, vocab: usize) -> Vec<u32> {
    let mut s = 0x9e3779b97f4a7c15u64;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % vocab as u64) as u32
        })
        .collect()
}

/// One user: a prompt of `n` random tokens, then `new` tokens, each step timed on the host (graph launch to the token read back). The prompt runs once untimed
/// and `reps` times timed (the median is reported); `profile_prefill`: one chunk's GPU time by kind of work.
pub fn bench(e: &mut Engine, n: usize, new: usize, load_s: f64, profile: bool, profile_prefill: bool, reps: usize) -> Result<()> {
    let prompt = random_prompt(n, e.w.cfg.vocab);
    // a first, short generation: lazy initializations out of the timing
    e.generate(&prompt[..prompt.len().min(64)], 4, true, false)?;
    e.reset();
    let mut s = e.prefill(&prompt, false)?; // (the whole prompt once, untimed: the large products' first use)
    let mut pre = Vec::new();
    for _ in 0..reps.max(1) {
        e.reset();
        let t0 = Instant::now();
        s = e.prefill(&prompt, false)?;
        pre.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    let all = pre.clone();
    pre.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let prefill_ms = pre[pre.len() / 2];
    let mut ms = Vec::with_capacity(new);
    for _ in 1..new {
        let t = Instant::now();
        s = e.step(s.token, false)?;
        ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    e.kv.check_status()?;
    eprintln!("{}", kv_status(e));
    eprintln!("prefill ms of the {} timed runs: {}", all.len(), all.iter().map(|x| format!("{x:.1}")).collect::<Vec<_>>().join(", "));
    if profile_prefill {
        e.reset();
        let rows = e.profile_prefill(&prompt[..prompt.len().min(e.chunk())])?;
        eprintln!("one prompt chunk of {} tokens, GPU time by kind of work (eager):", prompt.len().min(e.chunk()));
        let total = rows.last().map_or(1.0, |r| r.1);
        for (label, ms, n) in &rows {
            eprintln!("  {ms:8.3} ms {:5.1}%  {n:4} launches  {label}", 100.0 * ms / total);
        }
    }
    if profile {
        let rows = e.profile_steps(s.token, 20)?;
        eprintln!("one decode step's GPU time by kind of work (eager, {} steps' mean):", 20);
        let total = rows.last().map_or(1.0, |r| r.1);
        for (label, ms, n) in &rows {
            eprintln!("  {ms:7.3} ms {:5.1}%  {n:4} launches  {label}", 100.0 * ms / total);
        }
    }
    let mean = ms.iter().sum::<f64>() / ms.len().max(1) as f64;
    let mut sorted = ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |q: f64| sorted[((sorted.len() - 1) as f64 * q) as usize];
    println!(
        "{{\"prompt_tokens\":{n},\"new_tokens\":{new},\"prefill_ms\":{prefill_ms:.1},\"decode_ms_mean\":{mean:.3},\"decode_ms_p50\":{:.3},\"decode_ms_p99\":{:.3},\"decode_tokens_per_s\":{:.2},\"load_s\":{load_s:.2},\"weights_gib\":{:.3},\"cache_tokens\":{},\"graph\":{}}}",
        p(0.5),
        p(0.99),
        1000.0 / mean,
        e.w.bytes as f64 / GIB,
        e.kv.tokens(),
        e.has_graph()
    );
    Ok(())
}

/// The streaming read bandwidth of the GPU: `gib` GiB read `reps` times by a kernel that only reads (the ceiling a memory-bound decode step has).
pub fn bandwidth(dev: &crate::dev::Dev, gib: usize, reps: usize) -> Result<f64> {
    use crate::dev::{Buf, Timer};
    let k = crate::kern::Kern::load(dev)?;
    let bytes = gib << 30;
    let buf = Buf::zeroed(bytes)?;
    let out = Buf::zeroed(4)?;
    let blocks = (dev.sms as u32) * 16;
    // SAFETY: live buffers of the sizes given; the stream is synchronized by the timer.
    unsafe {
        k.read_stream(dev.stream, buf.ptr, (bytes / 16) as u64, out.ptr, blocks)?;
        dev.sync()?;
        let t = Timer::new()?;
        t.start(dev.stream)?;
        for _ in 0..reps {
            k.read_stream(dev.stream, buf.ptr, (bytes / 16) as u64, out.ptr, blocks)?;
        }
        let ms = t.stop(dev.stream)? as f64;
        Ok(bytes as f64 * reps as f64 / (ms / 1e3) / 1e9)
    }
}

/// `l2check [--mib 8,16,32] [--reps 20] [--gap-us 10]`: the time of a streaming read of a buffer of MiB, cold (L2 flushed by a read of 512 MiB first), after the buffer is
/// asked of L2 by prefetches (then a gap of a spinning block, or none), after loads of it, and warm (read just before): whether a small kernel's blocks can fetch the
/// next product's bytes while the memory is idle (GLY-312).
pub fn l2check(dev: &crate::dev::Dev, mibs: &[usize], reps: usize, gap_us: u64) -> Result<()> {
    use crate::dev::{Buf, Timer};
    use crate::launch;
    let k = crate::kern::Kern::load(dev)?;
    let ask = dev.typed::<crate::kt::l2_ask>()?;
    let gap = dev.typed::<crate::kt::gap_wait>()?;
    let flush = Buf::zeroed(512 << 20)?;
    let out = Buf::zeroed(4)?;
    let blocks = (dev.sms as u32) * 16;
    for &mib in mibs {
        let bytes = mib << 20;
        let buf = Buf::zeroed(bytes)?;
        // SAFETY: live buffers of the sizes given; each case's stream is synchronized by its timer.
        let case = |how: &str| -> Result<f64> {
            let mut total = 0f64;
            for _ in 0..reps {
                unsafe {
                    k.read_stream(dev.stream, flush.ptr, (flush.bytes / 16) as u64, out.ptr, blocks)?;
                    match how {
                        "prefetch" | "prefetch, no gap" => launch!(ask, [blocks, 1, 1], [256, 1, 1], dev.stream; buf.ptr, bytes as u64, 0i32, out.ptr)?,
                        "loads" => launch!(ask, [blocks, 1, 1], [256, 1, 1], dev.stream; buf.ptr, bytes as u64, 1i32, out.ptr)?,
                        "warm" => k.read_stream(dev.stream, buf.ptr, (bytes / 16) as u64, out.ptr, blocks)?,
                        _ => {}
                    }
                    if how != "prefetch, no gap" && gap_us > 0 {
                        launch!(gap, [1, 1, 1], [32, 1, 1], dev.stream; (gap_us * 1000) as i64)?;
                    }
                    let t = Timer::new()?;
                    t.start(dev.stream)?;
                    k.read_stream(dev.stream, buf.ptr, (bytes / 16) as u64, out.ptr, blocks)?;
                    total += t.stop(dev.stream)? as f64;
                }
            }
            Ok(total * 1000.0 / reps as f64)
        };
        let cold = case("cold")?;
        print!("{mib:>4} MiB: cold {cold:.1} us ({:.0} GB/s)", bytes as f64 / cold / 1e3);
        for how in ["prefetch", "prefetch, no gap", "loads", "warm"] {
            let us = case(how)?;
            print!(", {how} {us:.1} us");
        }
        println!();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The benches' prompt is the same on both sides of a comparison: these are the values tools/ref_speed.py draws for Qwen3's vocabulary.
    #[test]
    fn random_prompt_as_the_references() {
        let p = random_prompt(1024, 151936);
        assert_eq!(&p[..6], &[127021, 146550, 57142, 136564, 111212, 37497]);
        assert_eq!(&p[1021..], &[148310, 42223, 12491]);
    }
}
