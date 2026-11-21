//! Whether a model fits a GPU, and with how much context: the weights in their layout, the KV cache for the context (12 bits a value, a page's header and its share of the pool on top), the
//! activations of a prompt chunk and the library's workspaces, against the memory the GPU has free less its reserve
//! (src/plan.rs `reserve_of`, the one reserve of every path). Estimates: the engine reports what it measured after it loads.

use crate::plan::{reserve_of, Kind, LoadPlan, QWEN3_WORK};
use crate::ui::{gib, Refusal};
use crate::weights::{Config, Layout};

/// The context a chat should have where the GPU has room, and the least worth running.
pub const CTX_WANT: usize = crate::support::DEFAULT_CTX;
pub const CTX_MIN: usize = 4096;

pub fn bits(layout: Layout) -> f64 {
    match layout {
        Layout::Tiered => 10.80,
        Layout::Twelve => 12.04,
        Layout::Compact => 6.80,                // (Candidate A, lossy: run --lossy)
        Layout::Int6 | Layout::Int6Row => 6.53, // (six-bit codes, a bf16 scale a run of 32, a step's unread word: 6 + 0.5 + 0.03; a row layout's the same bytes)
        Layout::Int5 | Layout::Int5Row => 5.53, // (five-bit codes, a bf16 scale a run of 32, a step's unread word: 5 + 0.5 + 0.03)
        Layout::Int4 | Layout::Int4Row => 4.53, // (four-bit codes, a bf16 scale a run of 32, a step's unread word: 4 + 0.5 + 0.03)
        Layout::Int4H => 4.53,                  // (four-bit codes, a 16-bit scale a run of 32, a step's unread word, as Int4)
        Layout::Int3 => 3.53,                   // (three-bit codes, a bf16 scale a run of 32, a step's unread word: 3 + 0.5 + 0.03)
        Layout::Cb3Row => 3.625,                // (three-bit codes, a bf16 scale a run of 32 and eight bf16 entries a row of 1,024: 3 + 0.5 + 0.125)
        Layout::Cb2Row => 2.5625,               // (two-bit codes, a bf16 scale a run of 32 and four bf16 entries a row of 1,024: 2 + 0.5 + 0.0625)
        Layout::Hummingbird => 2.99, // (the experts of the recipe hb40: 205 of a layer's 512 at 3.625 and the rest at 2.5625; the directory's own bytes are what a plan uses where it has them)
    }
}

/// The weights on the GPU in `layout` (every matrix packed, the embedding too), with the norms and tables.
pub fn weights(cfg: &Config, layout: Layout) -> u64 {
    (cfg.matrix_bytes() as f64 * bits(layout) / 16.0) as u64 + (cfg.layers * cfg.hidden * 4 * 2) as u64 + (cfg.max_pos * cfg.head_dim * 4) as u64
}

/// A token's keys and values in the lossless cache, all layers: 12 bits a value and a page's header.
pub fn kv_per_token(cfg: &Config) -> u64 {
    ((cfg.layers * 2 * cfg.kv_heads * cfg.head_dim) as f64 * 1.5 * 1.06) as u64
}

/// The activations of a chunk of `chunk` tokens (the buffers of one forward pass), the decoded matrix a cuBLAS product reads, and the fixed workspaces.
pub fn work(cfg: &Config, chunk: usize, cublas: bool) -> u64 {
    let (hid, hd) = (cfg.hidden as u64, cfg.head_dim as u64);
    let qkv = (cfg.heads as u64 + 2 * cfg.kv_heads as u64) * hd;
    let per_token = 2 * (hid * 4 + qkv + 2 * cfg.heads as u64 * hd + cfg.kv_heads as u64 * hd + 3 * cfg.inter as u64) + 16;
    let decoded = if cublas { 2 * cfg.inter as u64 * hid * 2 } else { 0 };
    chunk as u64 * per_token + decoded + QWEN3_WORK
}

/// The chunk of prompt tokens the engine runs at a time where `free` bytes are left after the weights and the cache: 4096 where cuBLAS is (it decodes a matrix once a chunk), else 2048, smaller where
/// the room is short.
pub fn chunk_for(cfg: &Config, room: u64, cublas: bool) -> usize {
    let first = if cublas { 4096 } else { 2048 };
    [first, 2048, 1024, 512].into_iter().filter(|&c| c <= first).find(|&c| work(cfg, c, cublas) <= room).unwrap_or(512)
}

/// The context a model runs with on a GPU that has `free` bytes (`gpu` its name), for `users` sequences at once (the cache holds `users` of the context): `asked` where the user gave one (refused if it does not fit),
/// else the model's own up to `CTX_WANT`, less where the room is short (not below `CTX_MIN`). The plan the engine loads: `chunk` is the prompt chunk of the memory left.
#[allow(clippy::too_many_arguments)]
pub fn plan(cfg: &Config, name: &str, gpu: &str, free: u64, layout: Layout, asked: Option<usize>, users: usize, cublas: bool) -> Result<LoadPlan, Refusal> {
    let users = users.max(1);
    let w = weights(cfg, layout);
    let per = kv_per_token(cfg).saturating_mul(users as u64);
    let reserve = reserve_of(free);
    let room_after_weights = free.saturating_sub(w + reserve);
    let want = asked.unwrap_or(CTX_WANT.min(cfg.max_pos)).min(cfg.max_pos);
    let min_work = work(cfg, 512, cublas);
    let fits = |ctx: usize| -> bool { room_after_weights >= ctx as u64 * per + min_work };
    let max_ctx = (room_after_weights.saturating_sub(min_work) / per.max(1)) as usize / 256 * 256;
    let ctx = match asked {
        Some(a) if a > cfg.max_pos => {
            return Err(Refusal::new(
                format!("{name} has a context of {} tokens at most, and {a} were asked for", cfg.max_pos),
                format!("Ask for up to {}: --context {}", cfg.max_pos, cfg.max_pos),
            ));
        }
        Some(a) if !fits(a) => {
            return Err(if max_ctx >= 1024 {
                Refusal::unsupported(
                    format!(
                        "{a} tokens of context{} do not fit this GPU with {name} ({} free; the weights take {})",
                        if users > 1 { format!(" for each of {users} users") } else { String::new() },
                        gib(free),
                        gib(w)
                    ),
                    format!("Ask for less: --context {}", max_ctx.min(a).max(1024) / 256 * 256),
                )
            } else {
                too_big(name, free, w, per)
            });
        }
        Some(a) => a,
        None if fits(want) => want,
        None if max_ctx >= CTX_MIN => max_ctx.min(want),
        None => return Err(too_big(name, free, w, per)),
    };
    let room = room_after_weights.saturating_sub(ctx as u64 * per);
    let chunk = chunk_for(cfg, room, cublas);
    Ok(LoadPlan {
        kind: Kind::Qwen3,
        family: "Qwen3",
        level: crate::levels::name_of(layout),
        layout,
        context: ctx,
        users,
        chunk,
        weights: w,
        bf16: cfg.matrix_bytes(),
        kv: ctx as u64 * per,
        work: work(cfg, chunk, cublas),
        draft: 0,
        reserve,
        free,
        gpu: gpu.to_string(),
        experts: None,
        spread: None,
    })
}

fn too_big(name: &str, free: u64, w: u64, per: u64) -> Refusal {
    let need = w + reserve_of(free) + CTX_MIN as u64 * per + (500 << 20);
    Refusal::unsupported(
        format!(
            "{name} does not fit this GPU: it needs about {} (the weights {} with Glyd, {} of context and the runtime), and {} are free",
            gib(need),
            gib(w),
            CTX_MIN,
            gib(free)
        ),
        "Close programs that use the GPU (nvidia-smi lists them), or run a smaller model",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen3_8b() -> Config {
        Config {
            hidden: 4096,
            layers: 36,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            inter: 12288,
            vocab: 151936,
            eps: 1e-6,
            rope_theta: 1e6,
            max_pos: 40960,
            tie: false,
            eos: vec![],
        }
    }

    #[test]
    fn qwen3_8b_on_the_gpus() {
        let c = qwen3_8b();
        // weights near the measured 10.2 GiB of the engine's record
        let w = weights(&c, Layout::Tiered);
        assert!((10 << 30..11_500_000_000u64).contains(&w), "{w}");
        // an L4 (22.5 GiB free): the want of 8,192 tokens, the 4,096 chunk with cuBLAS
        let l4 = plan(&c, "Qwen3-8B", "NVIDIA L4", 22 << 30, Layout::Tiered, None, 1, true).unwrap();
        assert_eq!((l4.context, l4.chunk), (8192, 4096), "{l4:?}");
        assert!(l4.gpu_total() < 22 << 30);
        // the whole 40,960-token window fits an L4 too: the lossless cache is small
        assert_eq!(plan(&c, "Qwen3-8B", "NVIDIA L4", 22 << 30, Layout::Tiered, Some(40960), 1, true).unwrap().context, 40960);
        // asking for more than fits is refused, with what fits
        let r = plan(&c, "Qwen3-8B", "NVIDIA L4", 13 << 30, Layout::Tiered, Some(40960), 1, true).unwrap_err();
        assert!(r.fix.starts_with("Ask for less: --context ") && r.code == crate::ui::CANNOT, "{r}");
        // a smaller GPU: less context than the want, or a smaller chunk
        let small = plan(&c, "Qwen3-8B", "NVIDIA L4", 13 << 30, Layout::Tiered, None, 1, true).unwrap();
        assert!(small.context >= CTX_MIN, "{small:?}");
        assert!(small.gpu_total() < 13 << 30, "{small:?}");
        // too small altogether: a refusal that hands the model to vLLM
        assert_eq!(plan(&c, "Qwen3-8B", "NVIDIA L4", 8 << 30, Layout::Tiered, None, 1, true).unwrap_err().code, crate::ui::CANNOT);
        // more than the model's own context
        assert!(plan(&Config { max_pos: 2048, ..qwen3_8b() }, "tiny", "NVIDIA L4", 22 << 30, Layout::Tiered, Some(4096), 1, true).is_err());
    }
}
