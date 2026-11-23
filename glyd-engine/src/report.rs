//! A benchmark's prompt of random tokens, and the GPU's own read bandwidth and L2 (a development build's).

use crate::ffi::Result;

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
