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

/// A development build's integer tuning variable as the library reads it (TUNING_INT: atoll of the variable, read at each call; a release build: the
/// fallback).
fn tuned(v: Option<String>, fallback: i64) -> i64 {
    v.map_or(fallback, |v| atoll(&v))
}

/// The step product's kernel for a launch (mma_gemm_run's choice): the layout's mma_gemm_kernel for MT tiles of 16 tokens, or the tiered layout's step loop
/// (`true`: W as the A operand, to 8 tokens).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StepK {
    Mt(u32),
    Tiered(bool),
}

/// mma_gemm_run's kernel for W of layout `u` (K its columns) and `m` tokens, as the library chooses with no tuning variable set; None where a variable
/// asks for a kernel not folded yet (GLYD_GPU_TMID's tiered_mid_run, the 12-bit layout's GLYD_GPU_PATCH_B and GLYD_GPU_PFD loops).
fn step_plan(u: Unpacker, k: i64, m: i64) -> Option<StepK> {
    if u == Unpacker::Tiered {
        let (mid, mt2) = (tuned(crate::tune!("GLYD_GPU_TMID"), 0), tuned(crate::tune!("GLYD_GPU_TMT2"), 0));
        if mid > 0 && m >= mid && k % 64 == 0 {
            return None;
        }
        if mt2 > 0 && m >= mt2 && m <= 32 {
            return Some(StepK::Mt(2));
        }
        if m <= 16 && tuned(crate::tune!("GLYD_GPU_STEP"), 1) != 0 {
            return Some(StepK::Tiered(m <= 8));
        }
    }
    if matches!(u, Unpacker::Twelve { .. }) && (tuned(crate::tune!("GLYD_GPU_PATCH_B"), 0) != 0 || pfd() > 0) {
        return None;
    }
    Some(StepK::Mt(if m <= 16 {
        1
    } else if m <= 32 {
        2
    } else {
        4
    }))
}

/// GLYD_GPU_PFD (0-16): the steps the 12-bit layout's small-M product asks of L2 ahead, which every step kernel is given (step_opts().pfd).
fn pfd() -> i32 {
    tuned(crate::tune!("GLYD_GPU_PFD"), 0).clamp(0, 16) as i32
}

/// The kernel's name in the engine's module (kernels/fold.cu), its threads and its dynamic shared memory (mma_gemm_mt's and step_launch's).
fn step_kernel(u: Unpacker, p: StepK) -> Option<(&'static str, u32, u32)> {
    const S2_BYTES: u32 = 1040;
    let name = match p {
        StepK::Tiered(true) => "fold_step_tiered_w",
        StepK::Tiered(false) => "fold_step_tiered_x",
        StepK::Mt(mt) => {
            let names: [&'static str; 3] = match u {
                Unpacker::Tiered => ["fold_gemm_tiered_1", "fold_gemm_tiered_2", "fold_gemm_tiered_4"],
                Unpacker::Compact => ["fold_gemm_compact_1", "fold_gemm_compact_2", "fold_gemm_compact_4"],
                Unpacker::Int6 => ["fold_gemm_int6_1", "fold_gemm_int6_2", "fold_gemm_int6_4"],
                Unpacker::Int5 => ["fold_gemm_int5_1", "fold_gemm_int5_2", "fold_gemm_int5_4"],
                Unpacker::Int4 => ["fold_gemm_int4_1", "fold_gemm_int4_2", "fold_gemm_int4_4"],
                Unpacker::Int4H(_) => ["fold_gemm_int4h_1", "fold_gemm_int4h_2", "fold_gemm_int4h_4"],
                Unpacker::Fp4 => ["fold_gemm_fp4_1", "fold_gemm_fp4_2", "fold_gemm_fp4_4"],
                Unpacker::Fp4R { .. } => ["fold_gemm_fp4r_1", "fold_gemm_fp4r_2", "fold_gemm_fp4r_4"],
                Unpacker::Int3 => ["fold_gemm_int3_1", "fold_gemm_int3_2", "fold_gemm_int3_4"],
                Unpacker::Twelve { .. } => ["fold_gemm_twelve_1", "fold_gemm_twelve_2", "fold_gemm_twelve_4"],
                Unpacker::V1 => return None,
            };
            names[(mt / 2) as usize]
        }
    };
    let table = !matches!(u, Unpacker::Twelve { .. });
    Some(match p {
        StepK::Mt(mt) => (name, 256, 4 * mt * 16 * 65 * 4 + if table { 8 * S2_BYTES } else { 0 }),
        StepK::Tiered(sw) => (name, 128, 4 * if sw { 8 } else { 16 } * 65 * 4 + 4 * S2_BYTES),
    })
}

/// The blocks an SM holds of an engine kernel at `threads` and `smem` in the current context, and the GPU's SMs (the library's per_sm and sm_count),
/// asked once a context.
fn per_sm(name: &'static str, threads: u32, smem: u32) -> Result<(i64, i64)> {
    type Seen = std::sync::Mutex<std::collections::HashMap<(usize, &'static str), (i64, i64)>>;
    static SEEN: std::sync::OnceLock<Seen> = std::sync::OnceLock::new();
    let ctx = D::current()? as usize;
    let seen = SEEN.get_or_init(Default::default);
    if let Some(&v) = seen.lock().unwrap_or_else(|e| e.into_inner()).get(&(ctx, name)) {
        return Ok(v);
    }
    let v = (crate::dev::here_named(name)?.blocks_per_sm(threads, smem)?, (D::attribute(D::current_device()?, Attr::Sms)? as i64).max(1));
    seen.lock().unwrap_or_else(|e| e.into_inner()).insert((ctx, name), v);
    Ok(v)
}

/// The current context's GPU's code (`current_gpu`), read once a device.
fn this_gpu() -> Result<i64> {
    static CODES: [std::sync::atomic::AtomicI64; 64] = [const { std::sync::atomic::AtomicI64::new(0) }; 64];
    let dev = D::current_device()?;
    let slot = CODES.get(dev as usize);
    if let Some(c) = slot.map(|c| c.load(std::sync::atomic::Ordering::Relaxed)).filter(|&c| c > 0) {
        return Ok(c - 1);
    }
    let code = current_gpu()? as i64;
    if let Some(c) = slot {
        c.store(code + 1, std::sync::atomic::Ordering::Relaxed);
    }
    Ok(code)
}

/// The step product's blocks and workspace for W [o, k] of layout `u`, `m` tokens, kernel `p` (mma_gemm_mt and step_launch: as many blocks as fit at once,
/// but at least 32 steps a block; their parts and the row blocks' sums, [blocks + o / 64][m][64] floats).
fn step_grid(u: Unpacker, p: StepK, o: i64, k: i64, m: i64) -> Result<Option<(&'static str, u32, u32, i64, usize)>> {
    let Some((name, threads, smem)) = step_kernel(u, p) else { return Ok(None) };
    let (per, sms) = per_sm(name, threads, smem)?;
    let (rb, total) = (o / 64, o / 64 * (k / 16));
    let nb = (per * sms).min(total / 32).max(1);
    Ok(Some((name, threads, smem, nb, ((nb + rb) * m * 64 * 4) as usize)))
}

/// Whether the engine's kernels take W [o, k] (the 12-bit layout or not) for `m` tokens on this GPU: the route GEMM, the library's default plans, a CUDA build.
fn gemm_route(twelve: bool, o: i64, k: i64, m: i64) -> Result<bool> {
    if cfg!(glyd_hip) || o % 64 != 0 || k % 16 != 0 || !(0..=64).contains(&m) {
        return Ok(false);
    }
    Ok(route(this_gpu()?, twelve, o, k, m).is_ok_and(|r| r == ROUTE_GEMM))
}

/// `GpuOps::linear_workspace` by the engine's plans where every layout's product of that size is the engine's (glyd_gpu_mma_linear_workspace: the most of
/// the tiered, compact, Int6, Int5, Int4 and Int3 layouts' workspaces; glyd_gpu_mma12_linear_workspace: the 12-bit layout's); None: the library's.
pub fn linear_workspace(twelve: bool, o: i64, k: i64, m: i64) -> Result<Option<usize>> {
    if !gemm_route(twelve, o, k, m)? {
        return Ok(None);
    }
    let layouts: &[Unpacker] = if twelve {
        &[Unpacker::Twelve { hb4: 0, xmax: 4 }]
    } else {
        &[Unpacker::Tiered, Unpacker::Compact, Unpacker::Int6, Unpacker::Int5, Unpacker::Int4, Unpacker::Int3]
    };
    let mut most = 0;
    for &u in layouts {
        let Some(p) = step_plan(u, k, m) else { return Ok(None) };
        let Some((.., bytes)) = step_grid(u, p, o, k, m)? else { return Ok(None) };
        most = most.max(bytes);
    }
    Ok(Some(most))
}

/// `GpuOps::linear` (Y [m, rows] = X [m, cols] W^T, no bias) by the engine's kernels where the route is GEMM and the plan the library's default; false
/// where the library takes it (another route, a variable's plan, words or a workspace it refuses: its own error then).
///
/// # Safety
/// As `GpuOps::linear`; a context with the engine's module is current.
#[allow(clippy::too_many_arguments)]
pub unsafe fn linear(w: &Pack, x: u64, m: i64, y: u64, ws: u64, ws_bytes: usize, done: u64, cs: Stream) -> Result<bool> {
    use crate::dev::here;
    use crate::kt;
    use crate::launch;
    let (o, k) = (w.rows, w.cols);
    let u = match unpacker(w) {
        Ok(Unpacker::V1) | Err(_) => return Ok(false),
        Ok(u) => u,
    };
    if !gemm_route(w.is_twelve(), o, k, m)? {
        return Ok(false);
    }
    let Some(p) = step_plan(u, k, m) else { return Ok(false) };
    let Some((_, threads, smem, nb, bytes)) = step_grid(u, p, o, k, m)? else { return Ok(false) };
    if done == 0 || (bytes > 0 && (ws == 0 || ws_bytes < bytes)) {
        return Ok(false);
    }
    let (g, b, pf) = ([nb as u32, 1, 1], [threads, 1, 1], pfd());
    macro_rules! mt {
        ($k1:ty, $k2:ty, $k4:ty; $($a:expr),+) => {
            match p {
                StepK::Mt(1) => launch!(here::<$k1>()?, g, b, smem smem, cs; $($a),+, o, k, m, x, 0u64, y, ws, done, pf),
                StepK::Mt(2) => launch!(here::<$k2>()?, g, b, smem smem, cs; $($a),+, o, k, m, x, 0u64, y, ws, done, pf),
                _ => launch!(here::<$k4>()?, g, b, smem smem, cs; $($a),+, o, k, m, x, 0u64, y, ws, done, pf),
            }
        };
    }
    match (u, w.words) {
        (Unpacker::Tiered, Words::Tiered { blocks, block_base, tiers: t }) => match p {
            StepK::Tiered(true) => {
                launch!(here::<kt::fold_step_tiered_w>()?, g, b, smem smem, cs; w.data, blocks, block_base, t[0], t[1], t[2], o, k, m, x, 0u64, y, ws, done)
            }
            StepK::Tiered(false) => {
                launch!(here::<kt::fold_step_tiered_x>()?, g, b, smem smem, cs; w.data, blocks, block_base, t[0], t[1], t[2], o, k, m, x, 0u64, y, ws, done)
            }
            _ => mt!(kt::fold_gemm_tiered_1, kt::fold_gemm_tiered_2, kt::fold_gemm_tiered_4; w.data, blocks, block_base, t[0], t[1], t[2]),
        },
        (Unpacker::Compact, Words::Tiered { blocks, block_base, tiers: t }) => {
            mt!(kt::fold_gemm_compact_1, kt::fold_gemm_compact_2, kt::fold_gemm_compact_4; w.data, blocks, block_base, t[0], t[1], t[2])
        }
        (Unpacker::Int6, Words::Tiered { blocks, .. }) => mt!(kt::fold_gemm_int6_1, kt::fold_gemm_int6_2, kt::fold_gemm_int6_4; w.data, blocks),
        (Unpacker::Int5, Words::Tiered { blocks, .. }) => mt!(kt::fold_gemm_int5_1, kt::fold_gemm_int5_2, kt::fold_gemm_int5_4; w.data, blocks),
        (Unpacker::Int4, Words::Tiered { blocks, .. }) => mt!(kt::fold_gemm_int4_1, kt::fold_gemm_int4_2, kt::fold_gemm_int4_4; w.data, blocks),
        (Unpacker::Int4H(base), Words::Tiered { blocks, .. }) => mt!(kt::fold_gemm_int4h_1, kt::fold_gemm_int4h_2, kt::fold_gemm_int4h_4; w.data, blocks, base),
        (Unpacker::Fp4, Words::Tiered { blocks, .. }) => mt!(kt::fold_gemm_fp4_1, kt::fold_gemm_fp4_2, kt::fold_gemm_fp4_4; w.data, blocks),
        (Unpacker::Fp4R { ks, mg, sh }, Words::Tiered { blocks, .. }) => {
            mt!(kt::fold_gemm_fp4r_1, kt::fold_gemm_fp4r_2, kt::fold_gemm_fp4r_4; w.data, blocks, ks, mg, sh)
        }
        (Unpacker::Int3, Words::Tiered { blocks, .. }) => mt!(kt::fold_gemm_int3_1, kt::fold_gemm_int3_2, kt::fold_gemm_int3_4; w.data, blocks),
        (Unpacker::Twelve { hb4, xmax }, Words::Twelve { exc, exc_base, .. }) => {
            mt!(kt::fold_gemm_twelve_1, kt::fold_gemm_twelve_2, kt::fold_gemm_twelve_4; w.data, exc, exc_base, hb4, xmax)
        }
        _ => return Ok(false),
    }?;
    Ok(true)
}

/// kvc::RawPool (gpu/glyd_gpu_kernels.h), field for field: the raw-slab pool as a kernel takes it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvRawPool {
    pub bits: u64,
    pub owner: u64,
    pub data: u64,
    pub ns: i32,
    pub plane: i64,
}

/// kvc::AttnParams (gpu/glyd_gpu_kernels.h), field for field: the decode attention's parameters, built as glyd_gpu_kv_decode_from and _decode_bf16 build them.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct KvAttnParams {
    pub q: u64,
    pub q_stride: i64,
    pub pool: u64,
    pub page_stride: i64,
    pub kbase: u64,
    pub vbase: u64,
    pub bt: u64,
    pub bt_stride: i64,
    pub slen: u64,
    pub ent: u64,
    pub nc: i32,
    pub raw: KvRawPool,
    pub ws_ml: u64,
    pub ws_o: u64,
    pub h: i32,
    pub g: i32,
    pub a: i32,
    pub hdrb: i32,
    pub splits: i32,
    pub hp: i32,
    pub sl2: f32,
    pub ppc_min: i32,
    pub pfloor: i32,
    pub cnt: u64,
    pub out: u64,
    pub out_stride: i64,
    pub first: u64,
    pub round: i32,
    pub kc: u64,
    pub vc: u64,
    pub cap: i32,
}

/// The decode attention's CTA: kvc::NW warps.
const KV_NW: i64 = 4;

/// C's atoi of a development build's variable (TUNING_ENV: none in a release build).
fn atoi(v: Option<String>) -> Option<i32> {
    v.map(|v| atoll(&v) as i32)
}

/// The plan's least pages a CTA (GLYD_KV_PPC_MIN: 4 to 64, else 16), read at each call.
fn kv_ppc_min() -> i32 {
    atoi(crate::tune!("GLYD_KV_PPC_MIN")).filter(|v| (KV_NW as i32..=64).contains(v)).unwrap_or(4 * KV_NW as i32)
}

/// `GpuOps::kv_floor_set`'s floor (0: the GPU's own), the library's g_kv_floor.
static KV_FLOOR: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

/// The floor the decode's plan takes (`kv_floor_set`, the library's glyd_gpu_kv_floor_set): 1 to 64; 0 the GPU's own.
pub fn kv_floor_set(floor: i32) -> Result<()> {
    if !(0..=64).contains(&floor) {
        return Err(invalid("kv_floor_set"));
    }
    KV_FLOOR.store(floor, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

/// The plan's CTAs a sequence has at least (kv_floor): GLYD_KV_FLOOR (1 to 64), else `kv_floor_set`'s, else on a GPU of 100 SMs or more 16 where the batch's
/// (sequence, kv head) pairs, 4 to an SM, are fewer than the SMs, else 1. The batch's size is B rounded up to a power of two.
fn kv_floor(b: i64, h: i64, sms: i64) -> i32 {
    let v = atoi(crate::tune!("GLYD_KV_FLOOR")).unwrap_or_else(|| KV_FLOOR.load(std::sync::atomic::Ordering::Relaxed));
    if (1..=64).contains(&v) {
        return v;
    }
    let bb = (b.max(1) as u64).next_power_of_two() as i64;
    if sms >= 100 && bb * h * 4 <= sms {
        16
    } else {
        1
    }
}

/// The GPU's SMs (the library's sm_count).
fn sm_count() -> Result<i64> {
    Ok((D::attribute(D::current_device()?, Attr::Sms)? as i64).max(1))
}

/// The most CTAs a (sequence, head) is split over (glyd_gpu_kv_decode_shares, kv_splits): enough for GLYD_KV_WAVES (6) waves of the machine of the kernel that
/// loads a page ahead with the merge in it, at the batch rounded up to a power of two, capped by the longest sequence's pages (ppc_min a CTA, or the floor at 4 a
/// warp).
pub fn kv_decode_splits(b: i64, h: i64, max_pages: i64) -> Result<Option<i64>> {
    if cfg!(glyd_hip) {
        return Ok(None);
    }
    if b < 1 || !(1..=32).contains(&h) || max_pages < 1 {
        return Err(invalid("kv_decode_shares"));
    }
    let (ctas, sms) = per_sm("fold_kv_attn_pipe_1", 32 * KV_NW as u32, 0)?;
    let waves = atoi(crate::tune!("GLYD_KV_WAVES")).filter(|w| (1..=64).contains(w)).unwrap_or(6) as i64;
    let bb = (b as u64).next_power_of_two() as i64;
    let ppc = kv_ppc_min() as i64;
    let want = ((sms * ctas * waves + bb * h - 1) / (bb * h)).max(1);
    let cap = ((max_pages + ppc - 1) / ppc).max((kv_floor(b, h, sms) as i64).min((max_pages + KV_NW - 1) / KV_NW)).max(1);
    Ok(Some(want.min(cap)))
}

/// Whether a development build asks for a decode kernel that is not folded (GLYD_KV_PIPE=1: the next page loaded ahead; GLYD_KV_FUSE=0: the merge a kernel of its own).
fn kv_variant() -> bool {
    crate::tune!("GLYD_KV_PIPE").is_some_and(|v| v.starts_with('1')) || crate::tune!("GLYD_KV_FUSE").is_some_and(|v| v.starts_with('0'))
}

/// The escape pool's entries (esc_of's `ent`) and its chunks, from the pool's base.
fn kv_escape(base: u64, nc: i64) -> (u64, i32) {
    if nc == 0 || base == 0 {
        return (0, 0);
    }
    (base + pool_head_bytes(nc) as u64, nc as i32)
}

/// The raw-slab pool as raw_of makes it, for pages of `hp` head-pages.
fn kv_raw(base: u64, ns: i64, hp: i64) -> KvRawPool {
    if ns == 0 || base == 0 {
        return KvRawPool::default();
    }
    let bits = ((ns + 31) / 32 + 3) / 4 * 16;
    KvRawPool { bits: base, owner: base + bits as u64, data: base + (bits + (ns + 3) / 4 * 16) as u64, ns: ns as i32, plane: hp * 2048 }
}

/// The decode attention over 12-bit pages by the engine's kernel (glyd_gpu_kv_decode_from: `first` 0 and `round` false is glyd_gpu_kv_decode), its
/// parameters as the library builds them. False where the library takes the call (arguments it refuses, with its own words for them; a variant's kernel).
///
/// # Safety
/// As `GpuOps::kv_decode_from`; a context with the engine's module is current.
#[allow(clippy::too_many_arguments)]
pub unsafe fn kv_decode(
    g: &crate::ffi::KvGeom,
    group: i64,
    (pool, kbase, vbase): (u64, u64, u64),
    (q, q_stride): (u64, i64),
    (block_table, bt_stride, seq_lens, first, round): (u64, i64, u64, u64, bool),
    b: i64,
    splits: i64,
    scale: f64,
    (out, out_stride): (u64, i64),
    (ws, ws_bytes): (u64, usize),
    cs: Stream,
) -> Result<bool> {
    let (h, d, a) = (g.heads, g.dim, g.arena);
    let esc_ok = (0..=1 << 30).contains(&g.nchunks) && (g.nchunks == 0 || (g.escape != 0 && g.escape % 16 == 0));
    let raw_ok = (0..=1 << 30).contains(&g.nslabs) && (g.nslabs == 0 || (g.raw != 0 && g.raw % 16 == 0));
    let Ok(page) = kv_page_bytes(h, d, a) else { return Ok(false) };
    if cfg!(glyd_hip)
        || kv_variant()
        || d & KV_ONE != 0
        || !(0..=65535).contains(&b)
        || g.page_stride < page
        || g.page_stride % 16 != 0
        || pool % 16 != 0
        || !esc_ok
        || !raw_ok
    {
        return Ok(false);
    }
    if b == 0 {
        return Ok(true);
    }
    let Ok(need) = kv_decode_workspace(b, h, group, d, splits) else { return Ok(false) };
    let w = h * group * d;
    if (need > 0 && (ws == 0 || ws_bytes < need))
        || q == 0
        || kbase == 0
        || vbase == 0
        || block_table == 0
        || seq_lens == 0
        || out == 0
        || q_stride < w
        || out_stride < w
        || q_stride % 2 != 0
        || q % 4 != 0
    {
        return Ok(false);
    }
    let hp = h * (d / KV_D);
    let (ent, nc) = kv_escape(g.escape, g.nchunks);
    let ws_ml = ws + KV_CNT_BYTES as u64;
    let p = KvAttnParams {
        q,
        q_stride,
        pool,
        page_stride: g.page_stride,
        kbase,
        vbase,
        bt: block_table,
        bt_stride,
        slen: seq_lens,
        ent,
        nc,
        raw: kv_raw(g.raw, g.nslabs, hp),
        ws_ml,
        ws_o: ws_ml + (b * h * splits * group * 2 * 4) as u64,
        h: h as i32,
        g: group as i32,
        a: a as i32,
        hdrb: ((16 * hp + 4 * (1 + KV_ND + a) + 127) & !127) as i32,
        splits: splits as i32,
        hp: hp as i32,
        sl2: (scale * std::f64::consts::LOG2_E) as f32,
        ppc_min: kv_ppc_min(),
        pfloor: kv_floor(b, h, sm_count()?),
        cnt: ws,
        out,
        out_stride,
        first,
        round: round as i32,
        ..KvAttnParams::default()
    };
    kv_attn(p, b, d, splits, cs)?;
    Ok(true)
}

/// `kv_decode` over a bf16 cache (glyd_gpu_kv_decode_bf16): the same kernel's instance of its own, its parameters as the library builds them.
///
/// # Safety
/// As `GpuOps::kv_decode_bf16`; a context with the engine's module is current.
#[allow(clippy::too_many_arguments)]
pub unsafe fn kv_decode_bf16(
    (q, q_stride): (u64, i64),
    (kc, vc, block_table, bt_stride, cap): (u64, u64, u64, i64, i64),
    (seq_lens, first, b, splits): (u64, u64, i64, i64),
    (h, group, d): (i64, i64, i64),
    (scale, round): (f64, bool),
    (out, out_stride): (u64, i64),
    (ws, ws_bytes): (u64, usize),
    cs: Stream,
) -> Result<bool> {
    if cfg!(glyd_hip) || kv_variant() || d % KV_D != 0 || d < KV_D || d > 4 * KV_D || d == 3 * KV_D || !(0..=65535).contains(&b) {
        return Ok(false);
    }
    if b == 0 {
        return Ok(true);
    }
    let Ok(need) = kv_decode_workspace(b, h, group, d, splits) else { return Ok(false) };
    let w = h * group * d;
    if (need > 0 && (ws == 0 || ws_bytes < need))
        || q == 0
        || kc == 0
        || vc == 0
        || seq_lens == 0
        || out == 0
        || (block_table == 0 && !(1..=i32::MAX as i64).contains(&cap))
        || q_stride < w
        || out_stride < w
        || q_stride % 2 != 0
        || q % 4 != 0
        || kc % 4 != 0
        || vc % 4 != 0
    {
        return Ok(false);
    }
    let ws_ml = ws + KV_CNT_BYTES as u64;
    let p = KvAttnParams {
        q,
        q_stride,
        bt: block_table,
        bt_stride,
        slen: seq_lens,
        ws_ml,
        ws_o: ws_ml + (b * h * splits * group * 2 * 4) as u64,
        h: h as i32,
        g: group as i32,
        splits: splits as i32,
        hp: (h * (d / KV_D)) as i32,
        sl2: (scale * std::f64::consts::LOG2_E) as f32,
        ppc_min: kv_ppc_min(),
        pfloor: kv_floor(b, h, sm_count()?),
        cnt: ws,
        out,
        out_stride,
        first,
        round: round as i32,
        kc,
        vc,
        cap: cap.min(i32::MAX as i64) as i32,
        ..KvAttnParams::default()
    };
    kv_attn(p, b, d, splits, cs)?;
    Ok(true)
}

/// kv_attn_launch's default kernel (the merge in it, no page ahead): grid (H, B, splits), 4 warps, by the head's width and the cache's kind.
unsafe fn kv_attn(p: KvAttnParams, b: i64, d: i64, splits: i64, cs: Stream) -> Result<()> {
    use crate::dev::here;
    use crate::kt;
    use crate::launch;
    let (g, blk) = ([p.h as u32, b as u32, splits as u32], [32 * KV_NW as u32, 1, 1]);
    match (p.kc != 0, d / KV_D) {
        (false, 1) => launch!(here::<kt::fold_kv_attn_1>()?, g, blk, cs; p),
        (false, 2) => launch!(here::<kt::fold_kv_attn_2>()?, g, blk, cs; p),
        (false, _) => launch!(here::<kt::fold_kv_attn_4>()?, g, blk, cs; p),
        (true, 1) => launch!(here::<kt::fold_kv_attn_bf16_1>()?, g, blk, cs; p),
        (true, 2) => launch!(here::<kt::fold_kv_attn_bf16_2>()?, g, blk, cs; p),
        (true, _) => launch!(here::<kt::fold_kv_attn_bf16_4>()?, g, blk, cs; p),
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
    let (n, bad) = kv_check(&dev, &lib)?;
    say("the decode attention over 12-bit pages and a bf16 cache, the engine's kernel against the library's, and its plan", n, bad);
    let (n, bad) = linear_check(&dev, &lib)?;
    say("the step product (route GEMM, 1 to 64 tokens), the engine's kernels against the library's, and its workspace", n, bad);
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

/// Random weights of `rows` x `cols` in each of the ten mma layouts, packed by glyd-gpu's packers (None where a packer did not take them), for `foldcheck`.
#[cfg(feature = "dev")]
fn test_packs(rows: usize, cols: usize, next: &mut impl FnMut() -> u64) -> Vec<(&'static str, Option<crate::weights::Packed>)> {
    use crate::weights::Packed;
    use glyd_gpu::pack as gp;
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
    packs
}

/// `foldcheck`'s step product: Y [m, o] = X W^T for every layout at 1 to 64 tokens, by the library's kernels and by the engine's where the GPU's route is
/// GEMM (the others are still the library's), every bf16 compared, and the workspace each asks, at 256 x 1024 and 128 x 4096. (compared, differences)
#[cfg(feature = "dev")]
fn linear_check(dev: &crate::dev::Dev, lib: &crate::ffi::Lib) -> Result<(usize, Vec<String>)> {
    use crate::dev::Buf;
    use crate::weights::upload_packed;
    let mut seed = 0xD1B5_4A32_D192_ED03u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let (mut n, mut bad, mut folded) = (0, Vec::new(), 0);
    let ms = [1i64, 2, 7, 8, 9, 15, 16, 17, 24, 32, 33, 48, 64];
    for (rows, cols) in [(256usize, 1024usize), (128, 4096)] {
        let x: Vec<u16> = (0..64 * cols).map(|_| (((next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5).to_bits() >> 16) as u16).collect();
        let xb = Buf::new(x.len() * 2)?;
        xb.write(0, &x)?;
        for twelve in [false, true] {
            for &m in &ms {
                n += 1;
                let (a, b) = (lib.library_linear_workspace(twelve, rows as i64, cols as i64, m), linear_workspace(twelve, rows as i64, cols as i64, m));
                match (&a, &b) {
                    (Ok(x), Ok(Some(y))) if x != y => {
                        bad.push(format!("workspace of {rows} x {cols}, 12-bit {twelve}, {m} tokens: the library's {x} bytes, the engine's {y}"))
                    }
                    (Err(e), _) | (_, Err(e)) => bad.push(format!("workspace of {rows} x {cols}, 12-bit {twelve}, {m} tokens: {e}")),
                    _ => {}
                }
            }
        }
        for (name, p) in test_packs(rows, cols, &mut next) {
            let Some(p) = p else { continue };
            let pb = upload_packed(p)?;
            for &m in &ms {
                let ws_bytes = lib.library_linear_workspace(pb.pack.is_twelve(), rows as i64, cols as i64, m)?.max(16);
                let (ws, da, db) = (Buf::new(ws_bytes)?, Buf::zeroed(4 * rows / 64 + 64)?, Buf::zeroed(4 * rows / 64 + 64)?);
                let (ya, yb) = (Buf::new(m as usize * rows * 2)?, Buf::new(m as usize * rows * 2)?);
                ya.fill_async(0xA5, dev.stream)?;
                yb.fill_async(0x5A, dev.stream)?;
                // SAFETY: the pack, X [64, cols], the workspace of the library's size (the engine's is the same or the product is the library's), zeroed
                // counters of each, Y [m, rows] each; one stream.
                let (ra, rb) = unsafe {
                    let ra = lib.library_linear(&pb.pack, xb.ptr, m, ya.ptr, ws.ptr, ws_bytes, da.ptr, dev.stream);
                    dev.sync()?;
                    (ra, linear(&pb.pack, xb.ptr, m, yb.ptr, ws.ptr, ws_bytes, db.ptr, dev.stream))
                };
                dev.sync()?;
                n += 1;
                match (&ra, &rb) {
                    (Err(e), _) | (_, Err(e)) => {
                        bad.push(format!("{name} {rows} x {cols}, {m} tokens: {e}"));
                        continue;
                    }
                    (_, Ok(false)) => continue,
                    _ => folded += 1,
                }
                let (mut va, mut vb) = (vec![0u16; m as usize * rows], vec![0u16; m as usize * rows]);
                ya.read(0, &mut va)?;
                yb.read(0, &mut vb)?;
                let differ = va.iter().zip(&vb).filter(|(x, y)| x != y).count();
                if differ > 0 && bad.len() < 12 {
                    bad.push(format!("{name} {rows} x {cols}, {m} tokens: {differ} of {} bf16 differ", va.len()));
                }
            }
        }
    }
    println!("  (the step product: {folded} products by the engine's kernels, the rest the library's route on this GPU)");
    Ok((n, bad))
}

/// `foldcheck`'s decode attention: 12-bit pages written by the library (random values around each channel's exponent, some far), then the decode of 1 and 3
/// sequences (and from a first token, with bf16 scores) by the library's kernel and by the engine's, and the same over a bf16 cache; every output bf16, the
/// plan's splits and the workspace compared. (compared, differences)
#[cfg(feature = "dev")]
fn kv_check(dev: &crate::dev::Dev, lib: &crate::ffi::Lib) -> Result<(usize, Vec<String>)> {
    use crate::dev::Buf;
    use crate::ffi::KvGeom;
    use crate::gpu_ops::GpuOps;
    let mut seed = 0x94D0_49BB_1331_11EBu64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let (mut n, mut bad) = (0, Vec::new());
    const A: i64 = 15;
    const RING: i64 = 256;
    for (h, d, g) in [(8usize, 128usize, 4usize), (4, 256, 2), (2, 512, 8), (1, 128, 1)] {
        let tokens: usize = 16 * 9 + 5;
        let blocks = tokens.div_ceil(16);
        let hp = (h * d / 128) as i64;
        let page = kv_page_bytes(h as i64, d as i64, A)?;
        let (nchunks, nslabs) = (4096i64, 64i64);
        let pool = Buf::zeroed(page as usize * blocks)?;
        let escape = Buf::zeroed(kv_escape_bytes(nchunks)? as usize)?;
        let raw = Buf::zeroed(kv_raw_bytes(nslabs, hp)? as usize)?;
        let status = Buf::zeroed(4 * (16 + 2 * RING) as usize)?;
        let geom = KvGeom { heads: h as i64, dim: d as i64, arena: A, page_stride: page, escape: escape.ptr, nchunks, raw: raw.ptr, nslabs };
        // values [tokens][h][d] around an exponent a channel, 1% far from it; each channel's window base the one that holds most of them
        let mut vals = || -> Vec<u16> {
            (0..tokens * h * d)
                .map(|i| {
                    let mu = 124 + (i % (h * d)) as u64 % 7;
                    let e = if next() % 100 == 0 { 3 + next() % 240 } else { mu + next() % 3 - 1 };
                    (((next() & 1) << 15) | (e << 7) | (next() % 128)) as u16
                })
                .collect()
        };
        let (k, v) = (vals(), vals());
        let bases = |x: &[u16]| -> Vec<u8> {
            (0..h * d)
                .map(|c| {
                    let mut hist = [0u32; 128];
                    for t in 0..tokens {
                        let b = x[t * h * d + c];
                        if (b >> 7) & 0xFF != 0xFF {
                            hist[((b >> 8) & 0x7F) as usize] += 1;
                        }
                    }
                    (0..=120usize).max_by_key(|&s| hist[s..s + 8].iter().sum::<u32>()).unwrap_or(0) as u8
                })
                .collect()
        };
        let (kb, vb) = (Buf::new(h * d)?, Buf::new(h * d)?);
        kb.write(0, &bases(&k))?;
        vb.write(0, &bases(&v))?;
        let (kbuf, vbuf, sl) = (Buf::new(2 * k.len())?, Buf::new(2 * v.len())?, Buf::new(8 * tokens)?);
        kbuf.write(0, &k)?;
        vbuf.write(0, &v)?;
        sl.write(0, &(0..tokens as i64).collect::<Vec<i64>>())?;
        let stride = (h * d) as i64;
        // SAFETY: the pool and its tables, the tokens' rows and slots, live; one stream.
        unsafe {
            lib.kv_write(&geom, pool.ptr, kb.ptr, vb.ptr, 0, kbuf.ptr, stride, vbuf.ptr, stride, sl.ptr, tokens as i64, status.ptr, 0, 0, RING, dev.stream)?
        };
        // the same values as a bf16 cache [places][h][d], the places the pages' tokens
        let (kc, vc) = (Buf::new(2 * k.len())?, Buf::new(2 * v.len())?);
        kc.write(0, &k)?;
        vc.write(0, &v)?;
        dev.sync()?;
        for (lens, first, round) in [(vec![tokens as i32], 0, false), (vec![tokens as i32, 37, 16 * 4], 0, false), (vec![tokens as i32, 70], 21, true)] {
            let b = lens.len();
            let table: Vec<i32> = (0..b).flat_map(|_| 0..blocks as i32).collect();
            let (bt, sq, fb) = (Buf::new(4 * table.len())?, Buf::new(4 * b)?, Buf::new(4 * b)?);
            bt.write(0, &table)?;
            sq.write(0, &lens)?;
            fb.write(0, &vec![first; b])?;
            let q: Vec<u16> = (0..b * h * g * d).map(|_| (((next() % 2000) as f32 - 1000.0) / 5000.0).to_bits().wrapping_shr(16) as u16).collect();
            let qb = Buf::new(2 * q.len())?;
            qb.write(0, &q)?;
            let splits = lib.library_kv_decode_splits(b as i64, h as i64, blocks as i64)?;
            n += 1;
            match kv_decode_splits(b as i64, h as i64, blocks as i64)? {
                Some(s) if s != splits => bad.push(format!("{h} x {d}, {b} sequences: the plan's splits {splits} (library) against {s}")),
                _ => {}
            }
            let ws_bytes = kv_decode_workspace(b as i64, h as i64, g as i64, d as i64, splits)?;
            let w = (h * g * d) as i64;
            for bf16 in [false, true] {
                let (wa, wb, oa, ob) = (Buf::zeroed(ws_bytes)?, Buf::zeroed(ws_bytes)?, Buf::new(2 * b * h * g * d)?, Buf::new(2 * b * h * g * d)?);
                oa.fill_async(0xA5, dev.stream)?;
                ob.fill_async(0x5A, dev.stream)?;
                let fp = if first > 0 { fb.ptr } else { 0 };
                // SAFETY: the pool (or the cache), its tables, queries [b][h g d], outputs and zeroed workspaces of the plan's size, live; one stream.
                let (ra, rb) = unsafe {
                    if bf16 {
                        let cache = (kc.ptr, vc.ptr, bt.ptr, blocks as i64, 0);
                        let seqs = (sq.ptr, fp, b as i64, splits);
                        let heads = (h as i64, g as i64, d as i64);
                        let ra = lib.library_kv_decode_bf16(qb.ptr, w, cache, seqs, heads, (0.0883883, round), oa.ptr, w, (wa.ptr, ws_bytes), dev.stream);
                        (ra, kv_decode_bf16((qb.ptr, w), cache, seqs, heads, (0.0883883, round), (ob.ptr, w), (wb.ptr, ws_bytes), dev.stream))
                    } else {
                        let ra = lib.library_kv_decode_from(
                            &geom,
                            g as i64,
                            pool.ptr,
                            kb.ptr,
                            vb.ptr,
                            qb.ptr,
                            w,
                            bt.ptr,
                            blocks as i64,
                            sq.ptr,
                            fp,
                            round,
                            b as i64,
                            splits,
                            0.0883883,
                            oa.ptr,
                            w,
                            wa.ptr,
                            ws_bytes,
                            dev.stream,
                        );
                        let tables = (bt.ptr, blocks as i64, sq.ptr, fp, round);
                        (
                            ra,
                            kv_decode(
                                &geom,
                                g as i64,
                                (pool.ptr, kb.ptr, vb.ptr),
                                (qb.ptr, w),
                                tables,
                                b as i64,
                                splits,
                                0.0883883,
                                (ob.ptr, w),
                                (wb.ptr, ws_bytes),
                                dev.stream,
                            ),
                        )
                    }
                };
                dev.sync()?;
                n += 1;
                let what =
                    format!("{h} x {d}, {g} queries a head, {b} sequences (first {first}, bf16 scores {round}){}", if bf16 { ", a bf16 cache" } else { "" });
                match (&ra, &rb) {
                    (Err(e), _) | (_, Err(e)) => {
                        bad.push(format!("{what}: {e}"));
                        continue;
                    }
                    (_, Ok(false)) => {
                        bad.push(format!("{what}: the engine's kernel did not take it"));
                        continue;
                    }
                    _ => {}
                }
                let (mut va, mut vb2) = (vec![0u16; b * h * g * d], vec![0u16; b * h * g * d]);
                oa.read(0, &mut va)?;
                ob.read(0, &mut vb2)?;
                let differ = va.iter().zip(&vb2).filter(|(x, y)| x != y).count();
                if differ > 0 {
                    bad.push(format!("{what}: {differ} of {} bf16 differ", va.len()));
                }
            }
        }
    }
    Ok((n, bad))
}

/// `foldcheck`'s unpacking: random weights of each layout packed by glyd-gpu's packers, unpacked by the library's kernel and by the engine's, whole and
/// rows 64-127, every bf16 compared; a pack of a layout number neither knows refused by both. (compared, differences)
#[cfg(feature = "dev")]
fn unpack_check(dev: &crate::dev::Dev, lib: &crate::ffi::Lib) -> Result<(usize, Vec<String>)> {
    use crate::dev::Buf;
    use crate::weights::upload_packed;
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let (mut n, mut bad) = (0, Vec::new());
    for (rows, cols) in [(256usize, 1024usize), (128, 4096)] {
        let packs = test_packs(rows, cols, &mut next);
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

    /// KvAttnParams is kvc::AttnParams byte for byte (the C struct's offsets on LP64: pointers and int64 at 8, int and float at 4).
    #[test]
    fn the_attention_params_are_the_kernels() {
        use std::mem::{offset_of, size_of};
        assert_eq!(size_of::<KvRawPool>(), 40);
        assert_eq!((offset_of!(KvRawPool, ns), offset_of!(KvRawPool, plane)), (24, 32));
        assert_eq!(size_of::<KvAttnParams>(), 248);
        let at = [
            offset_of!(KvAttnParams, nc),
            offset_of!(KvAttnParams, raw),
            offset_of!(KvAttnParams, ws_ml),
            offset_of!(KvAttnParams, h),
            offset_of!(KvAttnParams, pfloor),
            offset_of!(KvAttnParams, cnt),
            offset_of!(KvAttnParams, round),
            offset_of!(KvAttnParams, kc),
            offset_of!(KvAttnParams, cap),
        ];
        assert_eq!(at, [80, 88, 128, 144, 176, 184, 216, 224, 240]);
    }

    #[test]
    fn atoll_reads_as_c_does() {
        assert_eq!([atoll("12"), atoll(" 12x"), atoll("-3"), atoll("+4"), atoll("x"), atoll("")], [12, 12, -3, 4, 0, 0]);
        assert_eq!([whole(" 17 "), whole("17x"), whole("-1"), whole("")], [Some(17), None, Some(-1), None]);
    }
}
