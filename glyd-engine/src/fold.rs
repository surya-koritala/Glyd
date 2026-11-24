//! R17 (#238, glyd-engine/docs/design/one-kernel-build.md): the GPU library's host side, folded into the engine one family at a time, each
//! piece the same answer as the library's for every input (`foldcheck` compares them on a GPU, the library as the oracle).
//!
//! Folded so far, as `GpuOps` for `ffi::Lib` runs them:
//! - the packed products' way (glyd_gpu_mma_way, glyd_gpu_mma12_way: which kernel, or W decoded for a dense product, takes W [O, K] for M
//!   tokens on a GPU) and the GPU's code those ways are keyed by (glyd_gpu_gpu: compute capability plus a class by name). These are the
//!   measured per-GPU choices between Glyd's fused kernels and cuBLAS (V8a, #255: glyd-engine/docs/design/kernel-libraries.md);
//! - the paged KV cache's sizes (glyd_gpu_kv_leaf_bytes, _extra_bytes, _second_bytes) and its decode's workspace
//!   (glyd_gpu_kv_decode_workspace).
//!
//! Kernels (step 1): the mma layouts' unpacking (glyd_gpu_mma_unpack, glyd_gpu_mma12_unpack), from the library's own source
//! (gpu/glyd_gpu_kernels.h) in the engine's fatbin (kernels/fold.cu), launched here as the library launches it.

use crate::device::{Attr, CudaDevice as D, Device};
use crate::ffi::{Error, Kind, Pack, Result, Stream, Words};

/// The routes (glyd_gpu.h's GLYD_GPU_ROUTE_*): W decoded for the caller's dense product; the step product (to 64 tokens); the 12-bit layout's
/// staged product (cp.async, mma.sync); Hopper's TMA and wgmma product; the prompt kernel; DECODE with W decoded ahead beside the products
/// before it; the 12-bit prompt split over SMs set apart (opt-in); the step product with RTX 50's own tile (opt-in).
pub const ROUTE_DECODE: i32 = 0;
pub const ROUTE_GEMM: i32 = 1;
pub const ROUTE_MID: i32 = 2;
pub const ROUTE_WG: i32 = 3;
pub const ROUTE_BIG: i32 = 4;
pub const ROUTE_AHEAD: i32 = 5;
pub const ROUTE_SPLIT: i32 = 6;
pub const ROUTE_GEMM_CC12: i32 = 7;

/// A GPU's class (glyd_gpu.h's GLYD_GPU_*), added to its compute capability (major * 10 + minor) where that does not tell GPUs apart.
pub const GEFORCE: i64 = 1000;
pub const A10: i64 = 2000;
pub const L4: i64 = 3000;
pub const L40S: i64 = 4000;
pub const PCIE: i64 = 5000;
pub const GH200: i64 = 6000;
pub const H100: i64 = 7000;
/// A flag in a GPU's code: its routes with SPLIT, which the caller runs (opt-in; the engine never sets it).
pub const WITH_SPLIT: i64 = 1 << 20;

/// The library's status for an argument it refuses (cudaErrorInvalidValue), with the library's words for it.
fn invalid(call: &str) -> Error {
    Error::of(Kind::of_status(1), format!("{call}: invalid argument (1)"))
}

/// Whether `word` is in `name` between characters that are not letters, digits or '_' (Python's `\bword\b` in ASCII).
fn has_word(name: &str, word: &str) -> bool {
    let part = |c: Option<u8>| c.is_some_and(|c| c.is_ascii_alphanumeric() || c == b'_');
    let b = name.as_bytes();
    name.match_indices(word).any(|(i, _)| !part(i.checked_sub(1).map(|j| b[j])) && !part(b.get(i + word.len()).copied()))
}

/// A GPU's class by its name (glyd_gpu.h): "GeForce" in it; "A10" as a word (an A10, not an A10G, A100 or A40); "L4" as one (not an L40S or
/// L40); "L40S" as one; "PCIe" in any case (an A100 PCIe, an H100 PCIe); "GH200" as a word; "H100" as one, but not an H100 NVL; else 0.
pub fn gpu_class(name: &str) -> i64 {
    if name.contains("GeForce") {
        GEFORCE
    } else if has_word(name, "A10") {
        A10
    } else if has_word(name, "L4") {
        L4
    } else if has_word(name, "L40S") {
        L40S
    } else if name.to_ascii_lowercase().contains("pcie") {
        PCIE
    } else if has_word(name, "GH200") {
        GH200
    } else if has_word(name, "H100") && !has_word(name, "NVL") {
        H100
    } else {
        0
    }
}

/// A GPU's code as the routes take it: its compute capability, major * 10 + minor, plus its class by name.
pub fn gpu_code(cc: (i32, i32), name: &str) -> i64 {
    (cc.0 * 10 + cc.1) as i64 + gpu_class(name)
}

/// The current context's GPU's code (glyd_gpu_gpu).
pub fn current_gpu() -> Result<i32> {
    let dev = D::current_device()?;
    Ok(gpu_code((D::attribute(dev, Attr::CcMajor)?, D::attribute(dev, Attr::CcMinor)?), &D::name(dev)?) as i32)
}

/// The routes' thresholds (the library's RouteMins), read once: from the environment in a development build (GLYD_WG_MIN, GLYD_WG_MAX,
/// GLYD_MID_MIN, GLYD_DEC_MIN, GLYD_SPLIT_MIN, GLYD_SPLIT_MAX, GLYD_SPLIT_SMS), GLYD_SPLIT_MIN alone in a release one, as the library's
/// development and release builds read them.
struct RouteMins {
    wg_min: i64,
    wg_max: i64,
    mid_min: i64,
    /// 0: unset.
    dec_min: i64,
    /// 0: unset, the GPU's; split_min negative: never.
    split_min: i64,
    split_max: i64,
    split_sms: i64,
}

/// A whole number as strtoll reads it in base 10 with nothing but spaces after it, within i64; else None.
fn whole(v: &str) -> Option<i64> {
    v.trim().parse().ok()
}

fn route_mins() -> &'static RouteMins {
    static T: std::sync::OnceLock<RouteMins> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let get = |v: Option<String>, fallback: i64| v.as_deref().and_then(whole).unwrap_or(fallback);
        let split_min = get(std::env::var("GLYD_SPLIT_MIN").ok(), 0);
        RouteMins {
            wg_min: get(crate::tune!("GLYD_WG_MIN"), 17),
            wg_max: get(crate::tune!("GLYD_WG_MAX"), 1024),
            mid_min: get(crate::tune!("GLYD_MID_MIN"), 17),
            dec_min: get(crate::tune!("GLYD_DEC_MIN"), 0),
            split_min,
            split_max: get(crate::tune!("GLYD_SPLIT_MAX"), 0),
            split_sms: get(crate::tune!("GLYD_SPLIT_SMS"), 0),
        }
    })
}

/// C's atoll: spaces, a sign, digits, the rest ignored; 0 where there are no digits.
fn atoll(v: &str) -> i64 {
    let s = v.trim_start();
    let (neg, digits) = match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        _ => (false, s),
    };
    let n = digits.bytes().take_while(u8::is_ascii_digit).fold(0i64, |n, d| n.wrapping_mul(10).wrapping_add((d - b'0') as i64));
    if neg {
        n.wrapping_neg()
    } else {
        n
    }
}

/// The tiered layout's steps on Hopper by its TMA kernel (the route WG) to this many tokens: none until measured; GLYD_GPU_HT_MAX (a
/// development build, read at each call) up to 64.
fn tiered_wg_max() -> i64 {
    64.min(crate::tune!("GLYD_GPU_HT_MAX").map_or(0, |v| atoll(&v)))
}

/// A prompt decoded ahead from this many tokens: GeForce Ada's past 512 tiered and 1792 12-bit, an A10's from 512 and 640, an L40S's from
/// 1024 and 2048 (measured on each, gpu/glyd_gpu.cu); else never.
fn ahead_min(twelve: bool, gpu: i64) -> i64 {
    if gpu == L40S + 89 {
        return if twelve { 2048 } else { 1024 };
    }
    match gpu {
        g if g == GEFORCE + 89 => {
            if twelve {
                1793
            } else {
                513
            }
        }
        g if g == A10 + 86 => {
            if twelve {
                640
            } else {
                512
            }
        }
        _ => i64::MAX,
    }
}

/// A prompt decoded for cuBLAS, never fused, from this many tokens: the 12-bit layout from GLYD_DEC_MIN where set, else an A100's from
/// 769; an L4's from 896 tiered and 2560 12-bit (benchmarks/gpu/l4-routes-2026-09-29); else never.
fn dec_from(twelve: bool, gpu: i64, dec_min: i64) -> i64 {
    if twelve && dec_min != 0 {
        return dec_min;
    }
    if gpu == L4 + 89 {
        return if twelve { 2560 } else { 896 };
    }
    if twelve && gpu % 1000 == 80 {
        769
    } else {
        i64::MAX
    }
}

/// The route SPLIT's SMs for the decode (0: not the route): a 12-bit prompt on an A100 SXM from 769 to 4096 tokens (8192 for a matrix whose O
/// and K are both at least 5120), on a GH200 or an H100 SXM from 2048 to 8192 for such a matrix; only for a code with WITH_SPLIT
/// (benchmarks/gpu/option2-2026-09-29).
fn split_sms(twelve: bool, gpu: i64, o: i64, k: i64, m: i64) -> i64 {
    let t = route_mins();
    let code = gpu & !WITH_SPLIT;
    let cc = code % 1000;
    let (a100, hopper, large) = (code == 80, code == GH200 + 90 || code == H100 + 90, o >= 5120 && k >= 5120);
    if !twelve || gpu & WITH_SPLIT == 0 || k % 64 != 0 || t.split_min < 0 || cc < 80 {
        return 0;
    }
    let lo = if t.split_min != 0 {
        t.split_min
    } else if a100 {
        769
    } else if hopper && large {
        2048
    } else {
        i64::MAX
    };
    let hi = if t.split_max != 0 {
        t.split_max
    } else if a100 && !large {
        4096
    } else {
        8192
    };
    if m < lo || m > hi {
        return 0;
    }
    if t.split_sms != 0 {
        return t.split_sms;
    }
    if a100 {
        return if m < 1536 {
            12
        } else if m < 3072 {
            8
        } else {
            4
        };
    }
    if cc == 90 {
        return if m < 6144 { 12 } else { 4 };
    }
    12
}

fn route_for(twelve: bool, gpu: i64, o: i64, k: i64, m: i64) -> i32 {
    let t = route_mins();
    if split_sms(twelve, gpu, o, k, m) != 0 {
        return ROUTE_SPLIT;
    }
    let gpu = gpu & !WITH_SPLIT;
    let cc = gpu % 1000;
    let (a100, hopper, mid, k64) = (cc == 80, cc == 90, matches!(cc, 80 | 86 | 87 | 89), k % 64 == 0);
    if hopper && twelve && k64 && m >= t.wg_min && m <= t.wg_max {
        return ROUTE_WG;
    }
    if hopper && !twelve && k64 && m >= 1 && m <= tiered_wg_max() {
        return ROUTE_WG;
    }
    if mid && twelve && k64 && m >= t.mid_min && m <= if a100 { 128 } else { 64 } {
        return ROUTE_MID;
    }
    let (dec, ahead) = (dec_from(twelve, gpu, t.dec_min), ahead_min(twelve, gpu));
    if dec != i64::MAX && m >= dec {
        return ROUTE_DECODE;
    }
    if m <= 64 {
        return ROUTE_GEMM;
    }
    if ahead != i64::MAX && m >= ahead {
        return ROUTE_AHEAD;
    }
    if k64 && !hopper {
        ROUTE_BIG
    } else {
        ROUTE_DECODE
    }
}

/// The route the library takes W [o, k] for `m` tokens on the GPU of code `gpu`, in the 12-bit layout or the tiered (and block) ones
/// (glyd_gpu_mma_way, glyd_gpu_mma12_way).
pub fn route(gpu: i64, twelve: bool, o: i64, k: i64, m: i64) -> Result<i32> {
    if o < 1 || k < 1 || m < 0 {
        return Err(invalid("way"));
    }
    Ok(route_for(twelve, gpu, o, k, m))
}

/// The paged KV cache's geometry (the library's kvc): a page's tokens, a head-page's channels and its tiles' bytes, the chunks of the escape
/// pool a page's header names, the entries of a chunk, the decode workspace's counters.
const KV_D: i64 = 128;
const KV_TILE: i64 = 3072;
const KV_ND: i64 = 16;
const KV_CH: i64 = 64;
const KV_CNT_BYTES: i64 = 1 << 20;
/// `d` with this set: a tensor held in the keys' tile layout alone (glyd_gpu.h's GLYD_GPU_KV_ONE_STREAM).
const KV_ONE: i64 = 1 << 16;

fn kv_dim(d: i64) -> i64 {
    d & (KV_ONE - 1)
}

fn kv_shape_ok(h: i64, d: i64, a: i64) -> bool {
    let d = kv_dim(d);
    h >= 1 && (d == KV_D || d == 2 * KV_D || d == 4 * KV_D) && h * (d / KV_D) <= 32 && (0..=1023).contains(&a)
}

/// Bytes of a pool's bitmap and owners for `n` chunks or slabs (16-byte rows).
fn pool_head_bytes(n: i64) -> i64 {
    ((n + 31) / 32 + 3) / 4 * 16 + (n + 3) / 4 * 16
}

/// A page's bytes for `h` heads of `d` channels (and KV_ONE: one stream) with an arena of `a` entries (glyd_gpu_kv_leaf_bytes).
pub fn kv_page_bytes(h: i64, d: i64, a: i64) -> Result<i64> {
    if !kv_shape_ok(h, d, a) {
        return Err(invalid("kv_leaf_bytes"));
    }
    let heads = h * (kv_dim(d) / KV_D);
    let header = (16 * heads + 4 * (1 + KV_ND + a) + 127) & !127;
    Ok(header + heads * if d & KV_ONE != 0 { KV_TILE } else { 2 * KV_TILE })
}

/// The escape pool's bytes for `nchunks` chunks (glyd_gpu_kv_extra_bytes).
pub fn kv_escape_bytes(nchunks: i64) -> Result<i64> {
    if !(0..=1 << 30).contains(&nchunks) {
        return Err(invalid("kv_extra_bytes"));
    }
    Ok(pool_head_bytes(nchunks) + nchunks * KV_CH * 4)
}

/// The raw-slab pool's bytes for `nslabs` slabs of pages of `h` head-pages (glyd_gpu_kv_second_bytes).
pub fn kv_raw_bytes(nslabs: i64, h: i64) -> Result<i64> {
    if !(0..=1 << 30).contains(&nslabs) || !(1..=32).contains(&h) {
        return Err(invalid("kv_second_bytes"));
    }
    Ok(pool_head_bytes(nslabs) + nslabs * h * 2048)
}

/// The decode attention's workspace for `b` sequences of `h` kv heads of `g` query heads each, heads of `d` channels, split `splits` ways
/// (glyd_gpu_kv_decode_workspace): the counters, then each split's partial outputs and their two sums.
pub fn kv_decode_workspace(b: i64, h: i64, g: i64, d: i64, splits: i64) -> Result<usize> {
    if b < 1 || d & KV_ONE != 0 || !kv_shape_ok(h, d, 0) || !(1..=8).contains(&g) || !(1..=65535).contains(&splits) || b * h > KV_CNT_BYTES / 4 {
        return Err(invalid("kv_decode_workspace"));
    }
    Ok((KV_CNT_BYTES + b * h * splits * g * (2 + d) * 4) as usize)
}

/// The layouts' numbers in a pack's first tier word's top byte (glyd_gpu.cu).
const TIERED_TAG: u32 = 2;
const TIERED_V1_TAG: u32 = 0xFF;
const COMPACT_TAG: u32 = 3;
const INT6_TAG: u32 = 4;
const INT5_TAG: u32 = 5;
const INT4_TAG: u32 = 6;
const INT3_TAG: u32 = 7;
const INT4H_TAG: u32 = 13;
const FP4_TAG: u32 = 14;
const FP4R_TAG: u32 = 15;

/// A pack's layout as the library's unpack reads its words (glyd_gpu_mma_unpack: TieredV1's number first, then int6_words ... tiered_words in that
/// order; glyd_gpu_mma12_unpack: nib12), with what the engine's kernel of it takes besides the pointers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unpacker {
    /// The layout of v0.27 and before (no product reads it; the engine refuses such a pack at load): the library's kernel.
    V1,
    Int6,
    Int5,
    Int4,
    Int4H(u32),
    Fp4,
    Fp4R {
        ks: u32,
        mg: u32,
        sh: u32,
    },
    Int3,
    Compact,
    Tiered,
    /// The 12-bit layout: its high bytes' base in each byte, and the exceptions a step decodes in registers (GLYD_GPU_XMAX).
    Twelve {
        hb4: u32,
        xmax: i32,
    },
}

/// Fp4R's division by the steps of a row block as a multiply-high (glyd_gpu.cu fp4r_fmt): l = ceil(log2 ks) (1 at least), mg = 2^32 (2^l - ks) / ks + 1,
/// sh = l - 1.
fn fp4r(k: i64) -> Unpacker {
    let ks = (k / 16) as u64;
    let mut l = 1u32;
    while (1u64 << l) < ks {
        l += 1;
    }
    Unpacker::Fp4R { ks: ks as u32, mg: ((((1u64 << l) - ks) << 32) / ks + 1) as u32, sh: l - 1 }
}

/// The words of a pack as the library reads them for its unpack; refused where it refuses them (a block layout whose K is not a multiple of 32 or whose
/// buffers are not aligned, an Int4H base outside 1-239, a 12-bit base past 120, a layout number it does not know).
pub fn unpacker(w: &Pack) -> Result<Unpacker> {
    let k = w.cols;
    match w.words {
        Words::Tiered { blocks, tiers, .. } => {
            let (d16, s16) = (w.data % 16 == 0, blocks % 16 == 0);
            let block = |ok: bool, u: Unpacker| if k % 32 != 0 || !ok { Err(invalid("unpack")) } else { Ok(u) };
            match tiers[0] >> 24 {
                TIERED_V1_TAG => Ok(Unpacker::V1),
                INT6_TAG => block(d16 && s16, Unpacker::Int6),
                INT5_TAG => block(d16 && s16, Unpacker::Int5),
                INT4_TAG => block(d16 && s16, Unpacker::Int4),
                INT4H_TAG => block(d16 && s16 && (1..=239).contains(&tiers[1]), Unpacker::Int4H(tiers[1])),
                FP4_TAG => block(d16 && blocks % 8 == 0, Unpacker::Fp4),
                FP4R_TAG => block(d16 && s16 && k / 16 <= 0x7FFF_FFFF, fp4r(k)),
                INT3_TAG => block(d16 && s16, Unpacker::Int3),
                COMPACT_TAG => Ok(Unpacker::Compact),
                TIERED_TAG => Ok(Unpacker::Tiered),
                _ => Err(invalid("unpack")),
            }
        }
        Words::Twelve { sym, .. } => {
            let xmax = crate::tune!("GLYD_GPU_XMAX").map_or(4, |v| atoll(&v)).clamp(0, 64) as i32;
            if sym[0] <= 120 {
                return Ok(Unpacker::Twelve { hb4: sym[0] * 0x0101_0101, xmax });
            }
            let hb = sym[0] & 0xFF;
            if sym[0] != hb * 0x0101_0101 || hb > 120 || sym[1..] != [0, 0, 0] {
                return Err(invalid("unpack"));
            }
            Ok(Unpacker::Twelve { hb4: sym[0], xmax })
        }
    }
}

/// Whether the engine runs its own kernels for the operations folded so far: a build with kernels, unless a development build's GLYD_ENGINE_FOLD=lib
/// asks for the library's (the old path, for a comparison).
pub fn on() -> bool {
    !cfg!(no_kernels) && crate::tune!("GLYD_ENGINE_FOLD").as_deref() != Some("lib")
}

/// Rows [row0, row0 + rows) of W back to bf16 into `out` [rows, cols] (both multiples of 64) by the engine's kernel of W's layout (`unpacker`), on `cs`: a
/// warp a step, blocks of 256 threads, as glyd_gpu_mma_unpack launches its own (mma_unpack_run, no warps given).
///
/// # Safety
/// As `GpuOps::unpack`; a context with the engine's module is current (a `Dev` opened it).
pub unsafe fn unpack(u: Unpacker, w: &Pack, row0: i64, rows: i64, out: u64, cs: Stream) -> Result<()> {
    use crate::dev::here;
    use crate::kt;
    use crate::launch;
    if row0 % 64 != 0 || rows % 64 != 0 {
        return Err(invalid("unpack"));
    }
    let grid = (rows / 64 * (w.cols / 16) * 32 + 255) / 256;
    if grid < 1 || grid > u32::MAX as i64 {
        return Err(Error::of(Kind::of_status(9), "unpack: invalid configuration argument (9)"));
    }
    let (g, b, k) = ([grid as u32, 1, 1], [256u32, 1, 1], w.cols);
    match (u, w.words) {
        (Unpacker::Tiered, Words::Tiered { blocks, block_base, tiers: t }) => {
            launch!(here::<kt::fold_unpack_tiered>()?, g, b, cs; w.data, blocks, block_base, t[0], t[1], t[2], k, row0, rows, out)
        }
        (Unpacker::Compact, Words::Tiered { blocks, block_base, tiers: t }) => {
            launch!(here::<kt::fold_unpack_compact>()?, g, b, cs; w.data, blocks, block_base, t[0], t[1], t[2], k, row0, rows, out)
        }
        (Unpacker::Int6, Words::Tiered { blocks, .. }) => launch!(here::<kt::fold_unpack_int6>()?, g, b, cs; w.data, blocks, k, row0, rows, out),
        (Unpacker::Int5, Words::Tiered { blocks, .. }) => launch!(here::<kt::fold_unpack_int5>()?, g, b, cs; w.data, blocks, k, row0, rows, out),
        (Unpacker::Int4, Words::Tiered { blocks, .. }) => launch!(here::<kt::fold_unpack_int4>()?, g, b, cs; w.data, blocks, k, row0, rows, out),
        (Unpacker::Int4H(base), Words::Tiered { blocks, .. }) => {
            launch!(here::<kt::fold_unpack_int4h>()?, g, b, cs; w.data, blocks, base, k, row0, rows, out)
        }
        (Unpacker::Fp4, Words::Tiered { blocks, .. }) => launch!(here::<kt::fold_unpack_fp4>()?, g, b, cs; w.data, blocks, k, row0, rows, out),
        (Unpacker::Fp4R { ks, mg, sh }, Words::Tiered { blocks, .. }) => {
            launch!(here::<kt::fold_unpack_fp4r>()?, g, b, cs; w.data, blocks, ks, mg, sh, k, row0, rows, out)
        }
        (Unpacker::Int3, Words::Tiered { blocks, .. }) => launch!(here::<kt::fold_unpack_int3>()?, g, b, cs; w.data, blocks, k, row0, rows, out),
        (Unpacker::Twelve { hb4, xmax }, Words::Twelve { exc, exc_base, .. }) => {
            launch!(here::<kt::fold_unpack_twelve>()?, g, b, cs; w.data, exc, exc_base, hb4, xmax, k, row0, rows, out)
        }
        (u, _) => crate::ffi::err(format!("unpack: {u:?} has no kernel of the engine's for these words")),
    }
}

/// The plan of `pairs` pairs sorted by their experts' ids (glyd_gpu_moe_way, `GpuOps::moe_way`) by the engine's kernels, as the library launches its own: from
/// GLYD_GPU_ROUTE_MIN pairs (512; a development build reads it, 0 or less: never) one block of up to 32 warps, its counts a warp an expert in 48 KiB of
/// shared memory (at least 8 warps), else the one warp.
///
/// # Safety
/// As `GpuOps::moe_way`; a context with the engine's module is current.
pub unsafe fn moe_way(ids: u64, pairs: i64, experts: i64, plan: u64, cs: Stream) -> Result<()> {
    use crate::dev::here;
    use crate::kt;
    use crate::launch;
    let (p, e) = (pairs, experts);
    if p < 0 || p >= (1i64 << 31) - 2 * e || !(1..=12288).contains(&e) {
        return Err(invalid("moe_way"));
    }
    let min_pairs = crate::tune!("GLYD_GPU_ROUTE_MIN").map_or(512, |v| atoll(&v));
    let nw = 32.min((48 * 1024 / 4) / e - 1).min((p + 31) / 32);
    if min_pairs > 0 && p >= min_pairs && nw >= 8 {
        launch!(here::<kt::fold_moe_route_block>()?, [1, 1, 1], [(32 * nw) as u32, 1, 1], smem ((nw * e + e) * 4) as u32, cs; ids, p, e as i32, plan)
    } else {
        launch!(here::<kt::fold_moe_route>()?, [1, 1, 1], [32, 1, 1], smem (e * 4) as u32, cs; ids, p, e as i32, plan)
    }
}

/// `foldcheck` (R17): each folded piece against the library's on this GPU, the library the oracle: the GPU's code; the way of every GPU code the
/// library knows (with and without SPLIT), both layouts, matrices of every kind of K and size, at token counts on each side of every threshold;
/// the KV cache's sizes and decode workspace over valid and refused shapes; the experts' plan and the unpacking of every layout by the engine's own
/// kernels (`route_check`, `unpack_check`). Every answer the same, refusals included.
#[cfg(feature = "dev")]
pub fn check() -> Result<()> {
    use crate::gpu_ops::GpuOps;
    let dev = crate::dev::Dev::new(0)?;
    let lib = crate::ffi::Lib::find()?;
    let mut fails = 0;
    let mut say = |what: &str, n: usize, bad: Vec<String>| {
        println!(
            "{} {what}: {n} compared, {} differ{}",
            if bad.is_empty() { "PASS" } else { "FAIL" },
            bad.len(),
            bad.first().map_or(String::new(), |b| format!(" (first: {b})"))
        );
        fails += !bad.is_empty() as usize;
    };
    let same = |a: &Result<i64>, b: &Result<i64>| match (a, b) {
        (Ok(x), Ok(y)) => x == y,
        (Err(x), Err(y)) => x.kind == y.kind,
        _ => false,
    };
    let (ours, theirs) = (lib.gpu(), lib.oracle_gpu());
    say(
        "the GPU's code",
        1,
        if same(&ours.clone().map(i64::from), &theirs.clone().map(i64::from)) { vec![] } else { vec![format!("{ours:?} against {theirs:?}")] },
    );
    let ccs = [80, 86, 87, 89, 90, 100, 103, 110, 120, 121, 75];
    let classes = [0, GEFORCE, A10, L4, L40S, PCIE, GH200, H100];
    let shapes = [
        (4096, 4096),
        (5120, 5120),
        (5120, 4096),
        (4096, 5120),
        (12288, 4096),
        (151936, 4096),
        (2048, 1000),
        (1000, 2048),
        (64, 16),
        (128, 4160),
        (6144, 25600),
    ];
    let mut ms: Vec<i64> = (0..=140).collect();
    for t in [
        255, 256, 257, 511, 512, 513, 639, 640, 641, 767, 768, 769, 770, 895, 896, 897, 1023, 1024, 1025, 1535, 1536, 1792, 1793, 1794, 2047, 2048, 2049, 2559,
        2560, 2561,
    ] {
        ms.push(t);
    }
    ms.extend([3071, 3072, 4095, 4096, 4097, 6143, 6144, 8191, 8192, 8193, 16384, 1 << 20, 1 << 40]);
    let (mut n, mut bad) = (0, Vec::new());
    for &cc in &ccs {
        for &cl in &classes {
            for split in [0, WITH_SPLIT] {
                for twelve in [false, true] {
                    for &(o, k) in shapes.iter().chain(&[(0, 64), (64, 0)]) {
                        for &m in ms.iter().chain(&[-1]) {
                            let gpu = cc + cl + split;
                            let (a, b) = (route(gpu, twelve, o, k, m).map(i64::from), lib.oracle_route(gpu, twelve, o, k, m).map(i64::from));
                            n += 1;
                            if !same(&a, &b) && bad.len() < 8 {
                                bad.push(format!("gpu {gpu}, 12-bit {twelve}, {o} x {k}, {m} tokens: {a:?} against {b:?}"));
                            }
                        }
                    }
                }
            }
        }
    }
    say("the packed products' ways", n, bad);
    let (hs, ds, aa) = (
        [-1, 0, 1, 2, 4, 8, 16, 31, 32, 33],
        [0, 64, 128, 192, 256, 384, 512, 1024, 128 | KV_ONE, 256 | KV_ONE, 512 | KV_ONE, 576 | KV_ONE],
        [-1, 0, 1, 7, 33, 100, 1023, 1024],
    );
    let (mut n, mut bad) = (0, Vec::new());
    let mut cmp = |what: String, a: Result<i64>, b: Result<i64>| {
        n += 1;
        if !same(&a, &b) && bad.len() < 8 {
            bad.push(format!("{what}: {a:?} against {b:?}"));
        }
    };
    for &h in &hs {
        for &d in &ds {
            for &a in &aa {
                cmp(format!("page of {h} heads of {d}, arena {a}"), kv_page_bytes(h, d, a), lib.oracle_kv_page_bytes(h, d, a));
            }
        }
    }
    let counts = [-1, 0, 1, 31, 32, 33, 127, 128, 129, 1000, 65536, 1 << 30, (1 << 30) + 1];
    for &c in &counts {
        cmp(format!("escape pool of {c}"), kv_escape_bytes(c), lib.oracle_kv_escape_bytes(c));
        for &h in &hs {
            cmp(format!("raw pool of {c} slabs of {h}"), kv_raw_bytes(c, h), lib.oracle_kv_raw_bytes(c, h));
        }
    }
    for b in [0, 1, 2, 7, 64, 1000, 32768, 65536, 70000] {
        for &h in &hs {
            for g in [0, 1, 4, 8, 9] {
                for &d in &ds {
                    for s in [0, 1, 3, 64, 65535, 65536] {
                        let (x, y) = (kv_decode_workspace(b, h, g, d, s).map(|v| v as i64), lib.oracle_kv_decode_workspace(b, h, g, d, s).map(|v| v as i64));
                        cmp(format!("decode workspace of {b} x {h} x {g} heads of {d}, {s} splits"), x, y);
                    }
                }
            }
        }
    }
    say("the KV cache's sizes and the decode's workspace", n, bad);
    let (n, bad) = route_check(&dev, &lib)?;
    say("the experts' plan (moe_way), the engine's kernels against the library's (one warp and one block, ids out of range among them)", n, bad);
    let (n, bad) = unpack_check(&dev, &lib)?;
    say("the mma layouts' unpacking, the engine's kernels against the library's (every bf16 of each layout, whole and a row block)", n, bad);
    if fails > 0 {
        return crate::ffi::err(format!("foldcheck: {fails} failed"));
    }
    println!("foldcheck: PASS");
    Ok(())
}

/// `foldcheck`'s experts' plan: random ids (some outside 0 to E - 1: routed nowhere) for pair counts on both sides of the block kernel's 512 and experts from 8
/// to 12,288, the plan [2 + 2E + P] by the library and by the engine, every int32 compared; refused arguments refused by both. (compared, differences)
#[cfg(feature = "dev")]
fn route_check(dev: &crate::dev::Dev, lib: &crate::ffi::Lib) -> Result<(usize, Vec<String>)> {
    use crate::dev::Buf;
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let (mut n, mut bad) = (0, Vec::new());
    for e in [8i64, 64, 128, 512, 1364, 4096, 12288] {
        for p in [0i64, 1, 31, 33, 511, 512, 1856, 8192, 65536] {
            let ids: Vec<i64> = (0..p).map(|_| if next() % 50 == 0 { [-1, e, e + 7][(next() % 3) as usize] } else { (next() % e as u64) as i64 }).collect();
            let idb = Buf::new((p as usize).max(1) * 8)?;
            if p > 0 {
                idb.write(0, &ids)?;
            }
            let words = (2 + 2 * e + p) as usize;
            let (a, b) = (Buf::new(words * 4)?, Buf::new(words * 4)?);
            a.fill_async(0, dev.stream)?;
            b.fill_async(0, dev.stream)?;
            // SAFETY: ids [p] and two plans of 2 + 2e + p int32, live; one stream.
            let (ra, rb) = unsafe { (lib.library_moe_way(idb.ptr, p, e, a.ptr, dev.stream), moe_way(idb.ptr, p, e, b.ptr, dev.stream)) };
            dev.sync()?;
            n += 1;
            if let (Err(x), _) | (_, Err(x)) = (&ra, &rb) {
                bad.push(format!("{p} pairs of {e} experts: {x}"));
                continue;
            }
            let (mut va, mut vb) = (vec![0i32; words], vec![0i32; words]);
            a.read(0, &mut va)?;
            b.read(0, &mut vb)?;
            let differ = va.iter().zip(&vb).filter(|(x, y)| x != y).count();
            if differ > 0 && bad.len() < 8 {
                bad.push(format!("{p} pairs of {e} experts: {differ} of {words} words of the plan differ"));
            }
        }
    }
    for (p, e) in [(-1i64, 8i64), (8, 0), (8, 12289), ((1 << 31) - 16, 8)] {
        n += 1;
        // SAFETY: refused before anything is launched.
        let (ra, rb) = unsafe { (lib.library_moe_way(0, p, e, 0, dev.stream), moe_way(0, p, e, 0, dev.stream)) };
        if ra.is_ok() || rb.is_ok() {
            bad.push(format!(
                "{p} pairs of {e} experts: the library {}, the engine {}",
                if ra.is_ok() { "took it" } else { "refused" },
                if rb.is_ok() { "took it" } else { "refused" }
            ));
        }
    }
    Ok((n, bad))
}

/// `foldcheck`'s unpacking: random weights of each layout packed by glyd-gpu's packers, unpacked by the library's kernel and by the engine's, whole and
/// rows 64-127, every bf16 compared; a pack of a layout number neither knows refused by both. (compared, differences)
#[cfg(feature = "dev")]
fn unpack_check(dev: &crate::dev::Dev, lib: &crate::ffi::Lib) -> Result<(usize, Vec<String>)> {
    use crate::dev::Buf;
    use crate::weights::{upload_packed, Packed};
    use glyd_gpu::pack as gp;
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let (mut n, mut bad) = (0, Vec::new());
    for (rows, cols) in [(256usize, 1024usize), (128, 4096)] {
        // bf16 weights about N(0, 0.05): a sum of four uniforms
        let w: Vec<u16> = (0..rows * cols)
            .map(|_| {
                let u: f32 = (0..4).map(|_| (next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5).sum();
                ((u * 0.087).to_bits() >> 16) as u16
            })
            .collect();
        let rounded: Vec<u16> = w.iter().map(|&x| gp::round_a(x)).collect();
        let h4: Vec<u16> = (0..rows * cols / 32).map(|_| ((next() >> 40) as u16 & 0x07FF) | ((5 + (next() % 4) as u16) << 11)).collect();
        let w4h: Vec<u16> = (0..rows * cols).map(|i| gp::int4h_value((next() % 16) as u8, h4[i / 32], 110)).collect();
        let q: Vec<u8> = (0..rows * cols / 2).map(|_| next() as u8).collect();
        let (e, er): (Vec<u8>, Vec<u8>) = (0..rows * cols / 32).map(|_| (120 + (next() % 7) as u8, 121 + (next() % 4) as u8)).unzip();
        let tiered = |p: gp::Tiered| Packed {
            data: p.data,
            b1: p.blocks,
            b2: p.block_base.iter().flat_map(|x| x.to_le_bytes()).collect(),
            words: Words::Tiered { blocks: 0, block_base: 0, tiers: p.tiers },
            rows,
            cols,
        };
        let t12 = gp::pack_twelve(&w, rows, cols);
        let twelve = Packed {
            data: t12.data,
            b1: t12.exc.iter().flat_map(|x| x.to_le_bytes()).collect(),
            b2: t12.exc_base.iter().flat_map(|x| x.to_le_bytes()).collect(),
            words: Words::Twelve { exc: 0, exc_base: 0, sym: t12.sym },
            rows,
            cols,
        };
        let packs: Vec<(&str, Option<Packed>)> = vec![
            ("tiered", Some(tiered(gp::pack_tiered(&w, rows, cols)))),
            ("compact", gp::pack_compact_checked(&rounded, rows, cols).map(tiered)),
            ("int6", gp::pack_int6(&w, rows, cols).map(tiered)),
            ("int5", gp::pack_int5(&w, rows, cols).map(tiered)),
            ("int4", gp::pack_int4(&w, rows, cols).map(tiered)),
            ("int4h", gp::pack_int4h(&w4h, rows, cols).map(tiered)),
            ("fp4", gp::pack_fp4_checked(&q, &e, rows, cols).map(tiered)),
            ("fp4r", gp::pack_fp4r_checked(&q, &er, rows, cols).map(tiered)),
            ("int3", gp::pack_int3(&w, rows, cols).map(tiered)),
            ("12-bit", Some(twelve)),
        ];
        for (name, p) in packs {
            let Some(p) = p else {
                bad.push(format!("{name} {rows} x {cols}: glyd-gpu's packer did not pack the random weights"));
                continue;
            };
            let pb = upload_packed(p)?;
            let (a, b) = (Buf::new(rows * cols * 2)?, Buf::new(rows * cols * 2)?);
            for (row0, nr) in [(0i64, rows as i64), (64, 64)] {
                a.fill_async(0xA5, dev.stream)?;
                b.fill_async(0x5A, dev.stream)?;
                // SAFETY: the pack's buffers and two of rows x cols bf16, live; the same stream.
                let (ra, rb) = unsafe {
                    (
                        lib.library_unpack(&pb.pack, row0, nr, a.ptr, dev.stream),
                        unpacker(&pb.pack).and_then(|u| unpack(u, &pb.pack, row0, nr, b.ptr, dev.stream)),
                    )
                };
                dev.sync()?;
                n += 1;
                if let (Err(x), _) | (_, Err(x)) = (&ra, &rb) {
                    bad.push(format!("{name} {rows} x {cols} rows {row0}+{nr}: {x}"));
                    continue;
                }
                let (mut va, mut vb) = (vec![0u16; nr as usize * cols], vec![0u16; nr as usize * cols]);
                a.read(0, &mut va)?;
                b.read(0, &mut vb)?;
                let differ = va.iter().zip(&vb).filter(|(x, y)| x != y).count();
                if differ > 0 {
                    bad.push(format!("{name} {rows} x {cols} rows {row0}+{nr}: {differ} of {} bf16 differ", va.len()));
                }
            }
            // a layout number neither knows: both refuse
            let mut odd = pb.pack;
            if let Words::Tiered { ref mut tiers, .. } = odd.words {
                tiers[0] = (99 << 24) | (tiers[0] & 0xFF_FFFF);
                n += 1;
                // SAFETY: refused before anything is launched.
                let (ra, rb) = unsafe { (lib.library_unpack(&odd, 0, 64, a.ptr, dev.stream), unpacker(&odd).map(|_| ())) };
                if ra.is_ok() || rb.is_ok() {
                    bad.push(format!("{name}: an unknown layout number: the library {}, the engine {}", ra.is_ok(), rb.is_ok()));
                }
            }
        }
    }
    Ok((n, bad))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The class by name as glyd_gpu.h says, on the names the driver reports for these GPUs.
    #[test]
    fn a_gpus_class_is_read_from_its_name() {
        for (name, class) in [
            ("NVIDIA GeForce RTX 4080 SUPER", GEFORCE),
            ("NVIDIA A10", A10),
            ("NVIDIA A10G", 0),
            ("NVIDIA A100-SXM4-40GB", 0),
            ("NVIDIA A100-PCIE-40GB", PCIE),
            ("NVIDIA A40", 0),
            ("NVIDIA L4", L4),
            ("NVIDIA L40S", L40S),
            ("NVIDIA L40", 0),
            ("NVIDIA GH200 480GB", GH200),
            ("NVIDIA H100 80GB HBM3", H100),
            ("NVIDIA H100 NVL", 0),
            ("NVIDIA H100 PCIe", PCIE),
            ("NVIDIA H200", 0),
            ("NVIDIA RTX A6000", 0),
            ("L4_x", 0),
        ] {
            assert_eq!(gpu_class(name), class, "{name}");
        }
        assert_eq!(gpu_code((8, 9), "NVIDIA GeForce RTX 4090"), 1089);
        assert_eq!(gpu_code((9, 0), "NVIDIA H100 80GB HBM3"), 7090);
    }

    /// The ways the library's table gives, at its measured thresholds (gpu/glyd_gpu.cu route_for), with no tuning variable set.
    #[test]
    fn the_ways_are_the_librarys() {
        let r = |gpu: i64, twelve: bool, o: i64, k: i64, m: i64| route(gpu, twelve, o, k, m).unwrap();
        // an A100: the 12-bit layout's staged product from 17 to 128 tokens, its prompt decoded for cuBLAS from 769; the tiered one fused
        assert_eq!(
            [r(80, true, 4096, 4096, 16), r(80, true, 4096, 4096, 17), r(80, true, 4096, 4096, 128), r(80, true, 4096, 4096, 768)],
            [ROUTE_GEMM, ROUTE_MID, ROUTE_MID, ROUTE_BIG]
        );
        assert_eq!(r(80, true, 4096, 4096, 769), ROUTE_DECODE);
        assert_eq!([r(80, false, 4096, 4096, 64), r(80, false, 4096, 4096, 65), r(80, false, 4096, 4096, 1 << 20)], [ROUTE_GEMM, ROUTE_BIG, ROUTE_BIG]);
        // an L4: decoded from 896 tiered and 2560 12-bit; GeForce Ada and an A10 decode ahead; an L40S too
        assert_eq!(
            [r(L4 + 89, false, 4096, 4096, 895), r(L4 + 89, false, 4096, 4096, 896), r(L4 + 89, true, 4096, 4096, 2560)],
            [ROUTE_BIG, ROUTE_DECODE, ROUTE_DECODE]
        );
        assert_eq!(
            [r(GEFORCE + 89, false, 4096, 4096, 512), r(GEFORCE + 89, false, 4096, 4096, 513), r(GEFORCE + 89, true, 4096, 4096, 1793)],
            [ROUTE_BIG, ROUTE_AHEAD, ROUTE_AHEAD]
        );
        assert_eq!(
            [r(A10 + 86, false, 4096, 4096, 512), r(A10 + 86, true, 4096, 4096, 640), r(L40S + 89, false, 4096, 4096, 1024)],
            [ROUTE_AHEAD, ROUTE_AHEAD, ROUTE_AHEAD]
        );
        // Hopper: TMA and wgmma for the 12-bit layout from 17 to 1024, then decoded; the tiered layout's prompt decoded (no fused prompt kernel there)
        assert_eq!(
            [r(H100 + 90, true, 4096, 4096, 17), r(H100 + 90, true, 4096, 4096, 1024), r(H100 + 90, true, 4096, 4096, 1025)],
            [ROUTE_WG, ROUTE_WG, ROUTE_DECODE]
        );
        assert_eq!([r(90, false, 4096, 4096, 64), r(90, false, 4096, 4096, 65)], [ROUTE_GEMM, ROUTE_DECODE]);
        // K not a multiple of 64: no prompt kernel takes it
        assert_eq!([r(89, false, 4096, 1000, 65), r(89, true, 4096, 1000, 17)], [ROUTE_DECODE, ROUTE_GEMM]);
        // SPLIT only where asked: an A100's 12-bit prompts of 769 to 4096 tokens
        assert_eq!(
            [r(80 + WITH_SPLIT, true, 4096, 4096, 769), r(80 + WITH_SPLIT, true, 4096, 4096, 4097), r(80 + WITH_SPLIT, true, 5120, 5120, 8192)],
            [ROUTE_SPLIT, ROUTE_DECODE, ROUTE_SPLIT]
        );
        assert_eq!(split_sms(true, 80 + WITH_SPLIT, 4096, 4096, 1535), 12);
        assert_eq!(split_sms(true, 80 + WITH_SPLIT, 4096, 4096, 3072), 4);
        assert_eq!(r(GH200 + 90 + WITH_SPLIT, true, 5120, 5120, 2048), ROUTE_SPLIT);
        assert_eq!(r(GH200 + 90 + WITH_SPLIT, true, 4096, 4096, 2048), ROUTE_DECODE);
        // refused as the library refuses
        for (o, k, m) in [(0, 64, 1), (64, 0, 1), (64, 64, -1)] {
            assert_eq!(route(89, false, o, k, m).unwrap_err().kind, Kind::of_status(1));
        }
    }

    /// The KV cache's sizes by glyd_gpu.cu's formulas: a page is its header (16 bytes a head-page, the used word, 16 chunk words and the arena,
    /// to 128 bytes) and 6,144 bytes a head-page (3,072 for one stream); the pools' bitmaps and owners in 16-byte rows; the decode's counters
    /// (1 MiB) and its splits' partials.
    #[test]
    fn the_kv_caches_sizes_are_the_librarys() {
        assert_eq!(kv_page_bytes(8, 128, 0).unwrap(), 256 + 8 * 6144);
        assert_eq!(kv_page_bytes(8, 256, 0).unwrap(), 384 + 16 * 6144);
        assert_eq!(kv_page_bytes(1, 576 | KV_ONE, 0).unwrap_err().kind, Kind::of_status(1));
        assert_eq!(kv_page_bytes(1, 512 | KV_ONE, 33).unwrap(), 384 + 4 * 3072);
        assert_eq!(kv_page_bytes(4, 128, 1023).unwrap(), ((64 + 4 * 1040 + 127) & !127) + 4 * 6144);
        for (h, d, a) in [(0, 128, 0), (33, 128, 0), (9, 512, 0), (1, 64, 0), (1, 128, -1), (1, 128, 1024)] {
            assert!(kv_page_bytes(h, d, a).is_err(), "{h} {d} {a}");
        }
        assert_eq!(kv_escape_bytes(0).unwrap(), 0);
        assert_eq!(kv_escape_bytes(1).unwrap(), 16 + 16 + 256);
        assert_eq!(kv_escape_bytes(129).unwrap(), 32 + 33 * 16 + 129 * 256);
        assert!(kv_escape_bytes(-1).is_err() && kv_escape_bytes((1 << 30) + 1).is_err());
        assert_eq!(kv_raw_bytes(3, 8).unwrap(), 16 + 16 + 3 * 8 * 2048);
        assert!(kv_raw_bytes(1, 0).is_err() && kv_raw_bytes(1, 33).is_err());
        assert_eq!(kv_decode_workspace(2, 8, 4, 128, 3).unwrap(), (1 << 20) + 2 * 8 * 3 * 4 * 130 * 4);
        for (b, h, g, d, s) in
            [(0, 8, 4, 128, 1), (1, 8, 9, 128, 1), (1, 8, 4, 128 | KV_ONE, 1), (1, 8, 4, 128, 0), (1, 8, 4, 128, 65536), (32769, 8, 1, 128, 1)]
        {
            assert!(kv_decode_workspace(b, h, g, d, s).is_err(), "{b} {h} {g} {d} {s}");
        }
    }

    /// A pack's words read as the library's unpack reads them: the layout by its number, a block layout refused where K is not a multiple of 32 or a
    /// buffer is not aligned, Int4H's base kept, the layout of v0.27 and before left to the library, a 12-bit base as either form the library takes.
    #[test]
    fn a_packs_words_are_read_as_the_library_reads_them() {
        let pack = |data: u64, blocks: u64, cols: i64, t0: u32, t1: u32| Pack {
            data,
            words: Words::Tiered { blocks, block_base: 0, tiers: [t0, t1, 0] },
            rows: 64,
            cols,
        };
        let read = |p: Pack| unpacker(&p);
        assert_eq!(read(pack(0x1000, 0x2000, 1024, TIERED_TAG << 24, 0)).unwrap(), Unpacker::Tiered);
        assert_eq!(read(pack(0x1000, 0x2000, 1000, COMPACT_TAG << 24, 0)).unwrap(), Unpacker::Compact);
        assert_eq!(read(pack(0x1000, 0x2000, 1024, TIERED_V1_TAG << 24, 0)).unwrap(), Unpacker::V1);
        assert_eq!(read(pack(0x1000, 0x2000, 1024, INT4H_TAG << 24, 110)).unwrap(), Unpacker::Int4H(110));
        assert_eq!(read(pack(0x1000, 0x2008, 1024, FP4_TAG << 24, 0)).unwrap(), Unpacker::Fp4);
        for bad in [
            pack(0x1000, 0x2000, 1008, INT6_TAG << 24, 0),
            pack(0x1008, 0x2000, 1024, INT5_TAG << 24, 0),
            pack(0x1000, 0x2008, 1024, INT4_TAG << 24, 0),
            pack(0x1000, 0x2000, 1024, INT4H_TAG << 24, 0),
            pack(0x1000, 0x2000, 1024, INT4H_TAG << 24, 240),
            pack(0x1000, 0x2004, 1024, FP4_TAG << 24, 0),
            pack(0x1000, 0x2000, 1024, 99 << 24, 0),
        ] {
            assert_eq!(read(bad).unwrap_err().kind, Kind::of_status(1), "{bad:?}");
        }
        let twelve = |sym: [u32; 4]| unpacker(&Pack { data: 0, words: Words::Twelve { exc: 0, exc_base: 0, sym }, rows: 64, cols: 1024 });
        assert_eq!(twelve([37, 0, 0, 0]).unwrap(), Unpacker::Twelve { hb4: 0x2525_2525, xmax: 4 });
        assert_eq!(twelve([0x2525_2525, 0, 0, 0]).unwrap(), Unpacker::Twelve { hb4: 0x2525_2525, xmax: 4 });
        assert!(twelve([0x2525_2526, 0, 0, 0]).is_err() && twelve([0x2525_2525, 1, 0, 0]).is_err() && twelve([0x7979_7979, 0, 0, 0]).is_err());
    }

    /// Fp4R's step / ks by a multiply-high (fp4r_fmt): (t + (n - t) / 2) >> sh with t = umulhi(n, mg) is n / ks for any 32-bit n.
    #[test]
    fn fp4rs_division_is_exact() {
        for k in [32i64, 64, 96, 1024, 1056, 2880, 4096, 7168, 18432, 1 << 20] {
            let Unpacker::Fp4R { ks, mg, sh } = fp4r(k) else { unreachable!() };
            assert_eq!(ks as i64, k / 16);
            for n in (0..2000u32).chain([u32::MAX, u32::MAX - 1, 1 << 31, 123_456_789, ks * 1000 - 1, ks * 1000]) {
                let t = ((n as u64 * mg as u64) >> 32) as u32;
                assert_eq!((t + (n - t) / 2) >> sh, n / ks, "K {k}, n {n}");
            }
        }
    }

    #[test]
    fn atoll_reads_as_c_does() {
        assert_eq!([atoll("12"), atoll(" 12x"), atoll("-3"), atoll("+4"), atoll("x"), atoll("")], [12, 12, -3, 4, 0, 0]);
        assert_eq!([whole(" 17 "), whole("17x"), whole("-1"), whole("")], [Some(17), None, Some(-1), None]);
    }
}
