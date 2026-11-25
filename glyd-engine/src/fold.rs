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
    Ok((KV_CNT_BYTES + (kv_ml_floats(b * h * splits * g) + b * h * splits * g * d) * 4) as usize)
}

/// The partials' maxima and sums (two floats each) after the decode workspace's counters, rounded up to 16 bytes: the partial outputs after them are read
/// as float4 (kv_ml_floats in glyd_gpu.cu).
fn kv_ml_floats(partials: i64) -> i64 {
    (partials * 2 + 3) / 4 * 4
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
    per_sm_most(name, threads, smem, smem)
}

/// `per_sm` with the kernel's shared memory allowed to `most` bytes first (the FP8 products: their 64 tokens' size, asked at each count's), asked once a
/// context and size.
fn per_sm_most(name: &'static str, threads: u32, smem: u32, most: u32) -> Result<(i64, i64)> {
    type Seen = std::sync::Mutex<std::collections::HashMap<(usize, &'static str, u32), (i64, i64)>>;
    static SEEN: std::sync::OnceLock<Seen> = std::sync::OnceLock::new();
    let ctx = D::current()? as usize;
    let seen = SEEN.get_or_init(Default::default);
    if let Some(&v) = seen.lock().unwrap_or_else(|e| e.into_inner()).get(&(ctx, name, smem)) {
        return Ok(v);
    }
    let k = crate::dev::here_named(name)?;
    k.allow_smem(most.max(smem))?;
    let v = (k.blocks_per_sm_allowed(threads, smem)?, (D::attribute(D::current_device()?, Attr::Sms)? as i64).max(1));
    seen.lock().unwrap_or_else(|e| e.into_inner()).insert((ctx, name, smem), v);
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

/// The route the library takes W [o, k] (the 12-bit layout or not) for `m` tokens on this GPU (route_for, as mma_linear_run asks it); None in an AMD build
/// (the library's own path there) or where it refuses the shape.
fn this_route(twelve: bool, o: i64, k: i64, m: i64) -> Result<Option<i32>> {
    if cfg!(glyd_hip) || m < 0 {
        return Ok(None);
    }
    Ok(route(this_gpu()?, twelve, o, k, m).ok())
}

/// Whether the step product (mma_gemm_run) takes W [o, k] for `m` tokens: rows a multiple of 64, columns of 16, 0 to 64 tokens.
fn step_ok(o: i64, k: i64, m: i64) -> bool {
    o % 64 == 0 && k % 16 == 0 && (0..=64).contains(&m)
}

/// The workspace of layout `u`'s product of W [o, k] for `m` tokens by route `r` (mma_linear_run's need); None: the library's (a route or plan not folded,
/// a call it refuses).
fn product_need(u: Unpacker, r: i32, o: i64, k: i64, m: i64) -> Result<Option<usize>> {
    let twelve = matches!(u, Unpacker::Twelve { .. });
    let step = || -> Result<Option<usize>> {
        if !step_ok(o, k, m) {
            return Ok(None);
        }
        let Some(p) = step_plan(u, k, m) else { return Ok(None) };
        Ok(step_grid(u, p, o, k, m)?.map(|g| g.4))
    };
    match r {
        ROUTE_GEMM => step(),
        ROUTE_MID if twelve => Ok(mid12_runs(o, k, m)?.map(|(_, need)| need)),
        ROUTE_MID | ROUTE_WG if !twelve && u != Unpacker::Tiered && m <= 64 => step(),
        ROUTE_DECODE | ROUTE_AHEAD | ROUTE_SPLIT | ROUTE_BIG if k % 64 == 0 && o % 64 == 0 => Ok(prompt_run(u, o, k, m)?.map(|r| r.bytes)),
        _ => Ok(None),
    }
}

/// `GpuOps::linear_workspace` by the engine's plans where every layout's product of that size is the engine's (glyd_gpu_mma_linear_workspace: the most of
/// the tiered, compact, Int6, Int5, Int4 and Int3 layouts' workspaces; glyd_gpu_mma12_linear_workspace: the 12-bit layout's); None: the library's.
pub fn linear_workspace(twelve: bool, o: i64, k: i64, m: i64) -> Result<Option<usize>> {
    let Some(r) = this_route(twelve, o, k, m)? else { return Ok(None) };
    let layouts: &[Unpacker] = if twelve {
        &[Unpacker::Twelve { hb4: 0, xmax: 4 }]
    } else {
        &[Unpacker::Tiered, Unpacker::Compact, Unpacker::Int6, Unpacker::Int5, Unpacker::Int4, Unpacker::Int3]
    };
    let mut most = 0;
    for &u in layouts {
        let Some(bytes) = product_need(u, r, o, k, m)? else { return Ok(None) };
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
    let (o, k) = (w.rows, w.cols);
    let u = match unpacker(w) {
        Ok(Unpacker::V1) | Err(_) => return Ok(false),
        Ok(u) => u,
    };
    let twelve = w.is_twelve();
    let Some(r) = this_route(twelve, o, k, m)? else { return Ok(false) };
    match r {
        ROUTE_GEMM => step_linear(u, w, x, m, y, (ws, ws_bytes, done), cs),
        ROUTE_MID if twelve => mid12_linear(u, w, x, m, y, (ws, ws_bytes, done), cs),
        ROUTE_MID | ROUTE_WG if !twelve && u != Unpacker::Tiered && m <= 64 => step_linear(u, w, x, m, y, (ws, ws_bytes, done), cs),
        ROUTE_DECODE | ROUTE_AHEAD | ROUTE_SPLIT | ROUTE_BIG if k % 64 == 0 => prompt_linear(u, w, x, m, y, (ws, ws_bytes, done), cs),
        // (Hopper's TMA and wgmma kernels, and what the library refuses: the library's)
        _ => Ok(false),
    }
}

/// The step product (mma_gemm_run, the route GEMM: 1 to 64 tokens) by the engine's kernels at the library's default plan.
///
/// # Safety
/// As `linear`.
unsafe fn step_linear(u: Unpacker, w: &Pack, x: u64, m: i64, y: u64, (ws, ws_bytes, done): (u64, usize, u64), cs: Stream) -> Result<bool> {
    use crate::dev::here;
    use crate::kt;
    use crate::launch;
    let (o, k) = (w.rows, w.cols);
    if !step_ok(o, k, m) {
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

/// CUtensorMap (cuda.h): a tensor map as the driver's cuTensorMapEncodeTiled writes it, 128 bytes on a 64-byte line, passed to Hopper's kernels by value.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorMap(pub [u64; 16]);

/// MoePairs (gpu/glyd_gpu_kernels.h), field for field: the prompt kernel's experts' plan (56 bytes); one matrix's product passes the library's
/// MoePairs{} (`Default`: `skip` 1, the rest zero).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoePairs {
    pub plan: u64,
    pub e: i32,
    pub k: i64,
    pub gather: i32,
    pub act: i32,
    pub w: u64,
    pub wf32: i32,
    pub xahead: i32,
    pub pf: i32,
    pub skip: i32,
}

impl Default for MoePairs {
    fn default() -> Self {
        MoePairs { plan: 0, e: 0, k: 0, gather: 0, act: 0, w: 0, wf32: 0, xahead: 0, pf: 0, skip: 1 }
    }
}

/// The prompt kernel's shape (Big<CW, PW, NB, RBB, CR>): its threads, tokens a block and shared memory (NB stages of X's tile and W's fragments).
fn big_shape(cw: i64, pw: i64, nb: i64, rbb: i64, cr: i64) -> (u32, i64, u32) {
    let tm = cr * cw / rbb;
    ((32 * (cw + pw)) as u32, tm, (nb * (tm * 64 * 2 + rbb * 4 * 4 * 32 * 16)) as u32)
}

/// A layout's name in the prompt kernels' (fold_big_*, fold_sk_*).
fn big_name(u: Unpacker) -> Option<&'static str> {
    Some(match u {
        Unpacker::Compact => "compact",
        Unpacker::Int3 => "int3",
        u => moe_name(u)?,
    })
}

/// The prompt kernel's plan for W [o, k] of layout `u` and `m` tokens on this GPU (mma_gemm_big_any, variant 0: blocks of 128 tokens by two row blocks, of 256 by
/// one past 128 tokens where the last block of 256 is more than half full or past the layout's most, the 12-bit layout's 256 by two on an A100; on GeForce Ada
/// stream-K): its kernel, launch and workspace (mma_gemm_grid_run, mma_gemm_big_run). None: no kernel of the layout.
struct PromptRun {
    name: &'static str,
    grid: [u32; 3],
    threads: u32,
    smem: u32,
    /// A grid's K split: the splits and the stages of each (the stream-K kernel: 0, 0)
    splits: i64,
    per: i64,
    /// stream-K: a unit's stages in several blocks (the counters needed)
    shared: bool,
    bytes: usize,
}

fn prompt_run(u: Unpacker, o: i64, k: i64, m: i64) -> Result<Option<PromptRun>> {
    let Some(tag) = big_name(u) else { return Ok(None) };
    let twelve = matches!(u, Unpacker::Twelve { .. });
    let dev = D::current_device()?;
    let cc = (D::attribute(dev, Attr::CcMajor)?, D::attribute(dev, Attr::CcMinor)?);
    let ada = cc == (8, 9) && D::name(dev)?.contains("GeForce");
    let a100 = twelve && cc == (8, 0);
    let most = if a100 {
        640
    } else if twelve {
        4224
    } else {
        1024
    };
    let variant = if m > 128 && (m % 256 == 0 || m % 256 > 128 || m > most || !(ada || a100)) {
        if a100 {
            3
        } else {
            2
        }
    } else {
        1
    };
    // (CW, PW, NB, RBB, CR, the kernel's tag; stream-K or a grid)
    let (cw, pw, nb, rbb, cr, v, sk) = match (variant, ada) {
        (3, _) if twelve => (8, 4, 2, 2, 64, "v3", false),
        (2, false) => (4, 4, 2, 1, 64, "v2", false),
        (_, false) => (4, 4, 3, 2, 64, "v1", false),
        (2, true) => (8, 4, 2, 1, 32, "v2", true),
        (_, true) if twelve => (8, 4, 2, 2, 32, "v1", true),
        _ => (4, 4, 3, 2, 64, "v1", true),
    };
    let name = leaked(format!("fold_{}_{tag}_{v}", if sk { "sk" } else { "big" }));
    let (threads, tm, smem) = big_shape(cw, pw, nb, rbb, cr);
    let (per_sm_blocks, sms) = per_sm(name, threads, smem)?;
    let stages = k / 16 / 4;
    let row_units = (o / 64 + rbb - 1) / rbb;
    let tiles = (m + tm - 1) / tm;
    if sk {
        let total = tiles * row_units * stages;
        let blocks = (per_sm_blocks * sms).min((8 * tiles * row_units).min(total / 4));
        let blocks = if total > 0 { blocks.max(1) } else { 0 };
        let shared = blocks > 0 && (total % blocks != 0 || total / blocks % stages != 0);
        let bytes = if shared { (2 * blocks * tm * 64 * rbb * 4) as usize } else { 0 };
        return Ok(Some(PromptRun { name, grid: [blocks as u32, 1, 1], threads, smem, splits: 0, per: 0, shared, bytes }));
    }
    // K split so the blocks fill their last wave: the fewest splits that leave under 15% of it idle (25% with eight consumer warps), else the fullest
    let cap = per_sm_blocks * sms;
    let (mut splits, mut best) = (1i64, 0f64);
    for sp in 1..=(stages / 4).max(1) {
        let nblocks = row_units * tiles * sp;
        let fill = nblocks as f64 / (((nblocks + cap - 1) / cap) * cap) as f64;
        if fill > best + 1e-9 {
            (best, splits) = (fill, sp);
        }
        if fill >= if cw >= 8 { 0.75 } else { 0.85 } {
            break;
        }
    }
    let per = (stages + splits - 1) / splits;
    let splits = (stages + per - 1) / per;
    let bytes = if splits > 1 { (splits * m * o * 4) as usize } else { 0 };
    Ok(Some(PromptRun { name, grid: [tiles as u32, row_units as u32, splits as u32], threads, smem, splits, per, shared: false, bytes }))
}

/// A launch of an engine kernel by name with its arguments in `a` (and the prompt kernel's plan last, where given).
///
/// # Safety
/// The arguments are the kernel's, and the buffers they name are live; a context with the engine's module is current.
unsafe fn launch_slots(name: &'static str, grid: [u32; 3], threads: u32, smem: u32, mut a: Slots, plan: Option<MoePairs>, cs: Stream) -> Result<()> {
    let mut plan = plan;
    let mut ptrs = crate::dev::KernelArgs::ptrs(&mut a);
    if let Some(p) = plan.as_mut() {
        ptrs.push(p as *mut MoePairs as *mut std::ffi::c_void);
    }
    crate::dev::here_named(name)?.launch_smem(grid, [threads, 1, 1], smem, cs, &mut ptrs)
}

/// The prompt kernel (mma_gemm_big_any: the routes BIG, DECODE, AHEAD and SPLIT) by the engine's kernels at the library's default plan; false where the library
/// takes it (a call it refuses: its own error then).
///
/// # Safety
/// As `linear`.
unsafe fn prompt_linear(u: Unpacker, w: &Pack, x: u64, m: i64, y: u64, (ws, ws_bytes, done): (u64, usize, u64), cs: Stream) -> Result<bool> {
    use crate::dev::here;
    use crate::kt;
    use crate::launch;
    let (o, k) = (w.rows, w.cols);
    if o % 64 != 0 || k < 1 || k % 64 != 0 || m < 0 || x % 16 != 0 {
        return Ok(false);
    }
    let Some(r) = prompt_run(u, o, k, m)? else { return Ok(false) };
    let Some(lead) = layout_args(u, w, None) else { return Ok(false) };
    if !fits(ws, ws_bytes, r.bytes as i64) || (r.shared && done == 0) {
        return Ok(false);
    }
    if r.per == 0 {
        if r.grid[0] > 0 {
            launch_slots(r.name, r.grid, r.threads, r.smem, lead.i64(o).i64(k).i64(m).p(x).p(0).p(y).p(ws).p(done), None, cs)?;
        }
        return Ok(true);
    }
    let y32 = if r.splits > 1 { ws } else { 0 };
    launch_slots(r.name, r.grid, r.threads, r.smem, lead.i64(o).i64(k).i64(m).i64(r.per).p(x).p(0).p(y).p(y32), Some(MoePairs::default()), cs)?;
    if r.splits > 1 {
        launch!(here::<kt::fold_finish>()?, [((m * o + 255) / 256) as u32, 1, 1], [256, 1, 1], cs; y32, r.splits, 0u64, m, o, y)?;
    }
    Ok(true)
}

/// The 12-bit layout's route MID (mma12_mid_any): its launches, 64 tokens each (an A100: 128, mma12_ws_kernel's four tiles), the smallest tile that holds
/// each one's tokens, and the workspace of the largest; None where the GPU has no such kernel (compute capability under 8). Each: (kernel, threads, shared
/// memory, blocks, first token, tokens).
#[allow(clippy::type_complexity)]
fn mid12_runs(o: i64, k: i64, m: i64) -> Result<Option<(Vec<(&'static str, u32, u32, i64, i64, i64)>, usize)>> {
    let dev = D::current_device()?;
    let cc = (D::attribute(dev, Attr::CcMajor)?, D::attribute(dev, Attr::CcMinor)?);
    if cc.0 < 8 || o % 64 != 0 || k < 1 || k % 64 != 0 || m < 0 {
        return Ok(None);
    }
    let (mut runs, mut need) = (Vec::new(), 0usize);
    if cc == (8, 0) {
        // (Ws12: tokens a unit, threads, shared memory; two row blocks a unit)
        for m0 in (0..m).step_by(128) {
            let mc = (m - m0).min(128);
            let (name, tm, threads, smem) = match mc {
                ..=32 => ("fold_ws12_32", 32, 384, 149_248),
                33..=64 => ("fold_ws12_64", 64, 384, 152_192),
                65..=96 => ("fold_ws12_96", 96, 384, 139_904),
                _ => ("fold_ws12_128", 128, 512, 152_192),
            };
            let (per, sms) = per_sm(name, threads, smem)?;
            let units = (o / 64 + 1) / 2 * ((mc + tm - 1) / tm);
            let nb = (per * sms).min(units * (k / 64) / 8).max(1);
            need = need.max(((nb + units) * tm * 64 * 2 * 4) as usize);
            runs.push((name, threads, smem, nb, m0, mc));
        }
        return Ok(Some((runs, need)));
    }
    // four warpgroups where 153 KB a block fit (Mid12<64, 4>), else two
    let wg = if D::attribute(dev, Attr::SmemOptin)? as i64 >= 156_352 { 4 } else { 2 };
    for m0 in (0..m).step_by(64) {
        let mc = (m - m0).min(64);
        let nt = match mc {
            ..=16 => 16,
            17..=32 => 32,
            _ => 64,
        };
        let smem = match (nt, wg) {
            (16, 2) => 70_336,
            (16, _) => 131_776,
            (32, 2) => 78_528,
            (32, _) => 139_968,
            (_, 2) => 94_912,
            _ => 156_352,
        };
        let (name, threads) = (leaked(format!("fold_mid12_{nt}_{wg}")), (128 * wg + 32) as u32);
        let (per, sms) = per_sm(name, threads, smem)?;
        let nb = (per * sms).min((o / 64 + wg - 1) / wg * (k / 64) / 8).max(1);
        need = need.max((2 * nb * nt * 64 * wg * 4) as usize);
        runs.push((name, threads, smem, nb, m0, mc));
    }
    Ok(Some((runs, need)))
}

/// The 12-bit layout's route MID by the engine's kernels (glyd_gpu_mma12_linear on it: mma12_mid_any by staged_run); false where the library takes it.
///
/// # Safety
/// As `linear`.
unsafe fn mid12_linear(u: Unpacker, w: &Pack, x: u64, m: i64, y: u64, (ws, ws_bytes, done): (u64, usize, u64), cs: Stream) -> Result<bool> {
    let (o, k) = (w.rows, w.cols);
    let Words::Twelve { exc, .. } = w.words else { return Ok(false) };
    if x % 16 != 0 || w.data % 16 != 0 || exc % 16 != 0 {
        return Ok(false);
    }
    let Some((runs, need)) = mid12_runs(o, k, m)? else { return Ok(false) };
    if !fits(ws, ws_bytes, need as i64) || done == 0 {
        return Ok(false);
    }
    for (name, threads, smem, nb, m0, mc) in runs {
        let Some(lead) = layout_args(u, w, None) else { return Ok(false) };
        let a = lead.i64(o).i64(k).i64(mc).p(x + (m0 * k * 2) as u64).p(0).p(y + (m0 * o * 2) as u64).p(ws).p(done);
        launch_slots(name, [nb as u32, 1, 1], threads, smem, a, None, cs)?;
    }
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
        ws_o: ws_ml + (kv_ml_floats(b * h * splits * group) * 4) as u64,
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
        ws_o: ws_ml + (kv_ml_floats(b * h * splits * group) * 4) as u64,
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

/// kvc::PrefillParams (gpu/glyd_gpu_kernels.h), field for field: the prompt attention's parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct KvPrefillParams {
    pub q: u64,
    pub q_stride: i64,
    pub key: u64,
    pub key_stride: i64,
    pub value: u64,
    pub value_stride: i64,
    pub pool: u64,
    pub page_stride: i64,
    pub kbase: u64,
    pub vbase: u64,
    pub bt: u64,
    pub ent: u64,
    pub nc: i32,
    pub raw: KvRawPool,
    pub sinks: u64,
    pub out: u64,
    pub out_stride: i64,
    pub kc: u64,
    pub vc: u64,
    pub cap: i32,
    pub n: i32,
    pub start: i32,
    pub window: i32,
    pub g: i32,
    pub a: i32,
    pub hdrb: i32,
    pub hp: i32,
    pub round: i32,
    pub sl2: f32,
    pub icap: f32,
    pub capl2: f32,
}

/// The rows and scores' arguments both prompt attentions share, as the library checks them; false where it refuses them.
fn kv_rows_ok(h: i64, g: i64, n: i64, start: i64, window: i64, scale: f64, softcap: f64) -> bool {
    n >= 0
        && start >= 0
        && window >= 0
        && start + n <= i32::MAX as i64
        && g >= 1
        && h * g <= 65535
        && softcap >= 0.0
        && scale.is_finite()
        && softcap.is_finite()
}

/// The scores' three numbers of PrefillParams: scale log2 e, scale / softcap (0: none), softcap log2 e.
fn kv_scores(scale: f64, softcap: f64) -> (f32, f32, f32) {
    let l2 = std::f64::consts::LOG2_E;
    ((scale * l2) as f32, if softcap > 0.0 { (scale / softcap) as f32 } else { 0.0 }, (softcap * l2) as f32)
}

/// glyd_gpu_kv_prefill's and _prefill_bf16's launch: rows of 16 by the CTA's groups (4 for heads of 128 and 256, 2 for 512), a CTA a query head, the shared
/// memory kvc::pf_shared takes (allowed first where past 48 KiB; refused as the library refuses where the GPU has not that much: CUDA_ERROR_NOT_SUPPORTED).
unsafe fn kv_prefill_launch(p: KvPrefillParams, hg: i64, hv: i64, cs: Stream) -> Result<()> {
    use crate::dev::here;
    use crate::kt;
    use crate::launch;
    const PKT: i64 = 32;
    const PSP: i64 = 33;
    let groups = if hv == 4 { 2 } else { 4 };
    let shared = (2 * PKT * (KV_D * hv + 8) * 2 + if hv > 1 { groups * hv * 16 * PSP * 4 } else { 0 }) as u32;
    let (g, b) = ([((p.n as i64 + 16 * groups - 1) / (16 * groups)) as u32, hg as u32, 1], [(32 * groups * hv) as u32, 1, 1]);
    let allow = |k: &crate::dev::Kernel| {
        if shared > 48 * 1024 && k.allow_smem(shared).is_err() {
            return Err(Error::of(Kind::of_status(801), "kv_prefill: operation not supported (801)"));
        }
        Ok(())
    };
    match hv {
        1 => {
            let k = here::<kt::fold_kv_prefill_1>()?;
            allow(&k)?;
            launch!(k, g, b, smem shared, cs; p)
        }
        2 => {
            let k = here::<kt::fold_kv_prefill_2>()?;
            allow(&k)?;
            launch!(k, g, b, smem shared, cs; p)
        }
        _ => {
            let k = here::<kt::fold_kv_prefill_4>()?;
            allow(&k)?;
            launch!(k, g, b, smem shared, cs; p)
        }
    }
}

/// A prompt chunk's attention over 12-bit pages by the engine's kernel (glyd_gpu_kv_prefill), its parameters as the library builds them; false where the
/// library takes the call (arguments it refuses, with its own words).
///
/// # Safety
/// As `GpuOps::kv_prefill`; a context with the engine's module is current.
#[allow(clippy::too_many_arguments)]
pub unsafe fn kv_prefill(
    g: &crate::ffi::KvGeom,
    group: i64,
    (pool, kbase, vbase, block_table): (u64, u64, u64, u64),
    (q, q_stride): (u64, i64),
    rows: Option<(u64, i64, u64, i64)>,
    (n, start, window): (i64, i64, i64),
    (scale, round, softcap, sinks): (f64, bool, f64, u64),
    (out, out_stride): (u64, i64),
    cs: Stream,
) -> Result<bool> {
    let (h, d, a) = (g.heads, g.dim, g.arena);
    let (key, key_stride, value, value_stride) = rows.unwrap_or((0, 0, 0, 0));
    let esc_ok = (0..=1 << 30).contains(&g.nchunks) && (g.nchunks == 0 || (g.escape != 0 && g.escape % 16 == 0));
    let raw_ok = (0..=1 << 30).contains(&g.nslabs) && (g.nslabs == 0 || (g.raw != 0 && g.raw % 16 == 0));
    let Ok(page) = kv_page_bytes(h, d, a) else { return Ok(false) };
    if cfg!(glyd_hip)
        || d & KV_ONE != 0
        || g.page_stride < page
        || g.page_stride % 16 != 0
        || pool % 16 != 0
        || !esc_ok
        || !raw_ok
        || !kv_rows_ok(h, group, n, start, window, scale, softcap)
    {
        return Ok(false);
    }
    if n == 0 {
        return Ok(true);
    }
    let w = h * group * d;
    let refused = q == 0
        || out == 0
        || kbase == 0
        || vbase == 0
        || (start > 0 && block_table == 0)
        || (key == 0 && block_table == 0)
        || q_stride < w
        || out_stride < w
        || q_stride % 2 != 0
        || out_stride % 2 != 0
        || q % 4 != 0
        || out % 4 != 0
        || (key != 0
            && (value == 0 || key_stride < h * d || value_stride < h * d || key_stride % 8 != 0 || value_stride % 8 != 0 || key % 16 != 0 || value % 16 != 0));
    if refused {
        return Ok(false);
    }
    let hp = h * (d / KV_D);
    let (ent, nc) = kv_escape(g.escape, g.nchunks);
    let (sl2, icap, capl2) = kv_scores(scale, softcap);
    let p = KvPrefillParams {
        q,
        q_stride,
        key,
        key_stride,
        value,
        value_stride,
        pool,
        page_stride: g.page_stride,
        kbase,
        vbase,
        bt: block_table,
        ent,
        nc,
        raw: kv_raw(g.raw, g.nslabs, hp),
        sinks,
        out,
        out_stride,
        n: n as i32,
        start: start as i32,
        window: window.min(i32::MAX as i64) as i32,
        g: group as i32,
        a: a as i32,
        hdrb: ((16 * hp + 4 * (1 + KV_ND + a) + 127) & !127) as i32,
        hp: hp as i32,
        round: round as i32,
        sl2,
        icap,
        capl2,
        ..KvPrefillParams::default()
    };
    kv_prefill_launch(p, h * group, d / KV_D, cs)?;
    Ok(true)
}

/// `kv_prefill` over a bf16 cache (glyd_gpu_kv_prefill_bf16; no sinks, as the engine calls it).
///
/// # Safety
/// As `GpuOps::kv_prefill_bf16`; a context with the engine's module is current.
#[allow(clippy::too_many_arguments)]
pub unsafe fn kv_prefill_bf16(
    (q, q_stride): (u64, i64),
    (kc, vc, block_table, cap): (u64, u64, u64, i64),
    (n, start, window): (i64, i64, i64),
    (h, group, d): (i64, i64, i64),
    (scale, round, softcap): (f64, bool, f64),
    (out, out_stride): (u64, i64),
    cs: Stream,
) -> Result<bool> {
    if cfg!(glyd_hip) || d % KV_D != 0 || d < KV_D || d > 4 * KV_D || d == 3 * KV_D || h < 1 || !kv_rows_ok(h, group, n, start, window, scale, softcap) {
        return Ok(false);
    }
    if n == 0 {
        return Ok(true);
    }
    let w = h * group * d;
    let refused = q == 0
        || out == 0
        || kc == 0
        || vc == 0
        || (block_table == 0 && !(1..=i32::MAX as i64).contains(&cap))
        || q_stride < w
        || out_stride < w
        || q_stride % 2 != 0
        || out_stride % 2 != 0
        || q % 4 != 0
        || out % 4 != 0
        || kc % 16 != 0
        || vc % 16 != 0;
    if refused {
        return Ok(false);
    }
    let (sl2, icap, capl2) = kv_scores(scale, softcap);
    let p = KvPrefillParams {
        q,
        q_stride,
        bt: block_table,
        out,
        out_stride,
        kc,
        vc,
        cap: cap.min(i32::MAX as i64) as i32,
        n: n as i32,
        start: start as i32,
        window: window.min(i32::MAX as i64) as i32,
        g: group as i32,
        hp: (h * (d / KV_D)) as i32,
        round: round as i32,
        sl2,
        icap,
        capl2,
        ..KvPrefillParams::default()
    };
    kv_prefill_launch(p, h * group, d / KV_D, cs)?;
    Ok(true)
}

/// kvc::EscapePool (gpu/glyd_gpu_kernels.h), field for field: the escape pool as the write kernel takes it (esc_of).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvEscapePool {
    pub bits: u64,
    pub owner: u64,
    pub ent: u64,
    pub nc: i32,
}

/// kvc::WriteParams (gpu/glyd_gpu_kernels.h), field for field: the write kernel's parameters, built as glyd_gpu_kv_write builds them.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvWriteParams {
    pub key: u64,
    pub value: u64,
    pub key_stride: i64,
    pub value_stride: i64,
    pub slots: u64,
    pub pool: u64,
    pub page_stride: i64,
    pub kbase: u64,
    pub vbase: u64,
    pub status: u64,
    pub esc: KvEscapePool,
    pub raw: KvRawPool,
    pub lost: u64,
    pub nlost: i64,
    pub ring: u32,
    pub layer_tag: u32,
    pub n: i32,
    pub h: i32,
    pub a: i32,
    pub lean: i32,
    pub one: i32,
}

/// The pool checks the library makes of a KV call (kv_escape_ok, kv_raw_ok, the page's bytes and alignment); false where it refuses them.
fn kv_pools_ok(g: &crate::ffi::KvGeom, pool: u64) -> bool {
    let esc_ok = (0..=1 << 30).contains(&g.nchunks) && (g.nchunks == 0 || (g.escape != 0 && g.escape % 16 == 0));
    let raw_ok = (0..=1 << 30).contains(&g.nslabs) && (g.nslabs == 0 || (g.raw != 0 && g.raw % 16 == 0));
    kv_page_bytes(g.heads, g.dim, g.arena).is_ok_and(|page| g.page_stride >= page && g.page_stride % 16 == 0 && pool % 16 == 0) && esc_ok && raw_ok
}

/// `n` tokens' keys and values into the layer's pool by the engine's write kernel (glyd_gpu_kv_write): a CTA a token, 256 threads; the layer's owner tag,
/// the status rings and the lost blocks' bits as the library passes them, GLYD_KV_LEAN (development builds) as it reads it. False where the library takes it.
///
/// # Safety
/// As `GpuOps::kv_write`; a context with the engine's module is current.
#[allow(clippy::too_many_arguments)]
pub unsafe fn kv_write(
    g: &crate::ffi::KvGeom,
    (pool, kbase, vbase, layer): (u64, u64, u64, i64),
    (key, key_stride, value, value_stride, slots, n): (u64, i64, u64, i64, u64, i64),
    (status, lost, nlost, ring): (u64, u64, i64, i64),
    cs: Stream,
) -> Result<bool> {
    use crate::dev::here;
    use crate::kt;
    use crate::launch;
    let (h, d) = (g.heads, g.dim);
    if cfg!(glyd_hip) || !(0..=i32::MAX as i64).contains(&n) || status == 0 || !kv_pools_ok(g, pool) {
        return Ok(false);
    }
    if !(0..=254).contains(&layer) || nlost < 0 || (nlost > 0 && (lost == 0 || lost % 4 != 0)) || !(1..=1 << 24).contains(&ring) {
        return Ok(false);
    }
    if n == 0 {
        return Ok(true);
    }
    let (dm, one) = (kv_dim(d), d & KV_ONE != 0);
    if key == 0 || slots == 0 || kbase == 0 || key_stride < h * dm || key_stride % 2 != 0 || key % 4 != 0 {
        return Ok(false);
    }
    if !one && (value == 0 || vbase == 0 || value_stride < h * dm || value_stride % 2 != 0 || value % 4 != 0) {
        return Ok(false);
    }
    let hp = h * (dm / KV_D);
    let esc = if g.nchunks == 0 || g.escape == 0 {
        KvEscapePool::default()
    } else {
        let bits = ((g.nchunks + 31) / 32 + 3) / 4 * 16;
        KvEscapePool { bits: g.escape, owner: g.escape + bits as u64, ent: g.escape + pool_head_bytes(g.nchunks) as u64, nc: g.nchunks as i32 }
    };
    let p = KvWriteParams {
        key,
        value,
        key_stride,
        value_stride,
        slots,
        pool,
        page_stride: g.page_stride,
        kbase,
        vbase,
        status,
        esc,
        raw: kv_raw(g.raw, g.nslabs, hp),
        lost: if nlost > 0 { lost } else { 0 },
        nlost,
        ring: ring as u32,
        layer_tag: ((layer + 1) as u32) << 24,
        n: n as i32,
        h: hp as i32,
        a: g.arena as i32,
        lean: !crate::tune!("GLYD_KV_LEAN").is_some_and(|v| v.starts_with('0')) as i32,
        one: one as i32,
    };
    launch!(here::<kt::fold_kv_write>()?, [n as u32, 1, 1], [256, 1, 1], cs; p)?;
    Ok(true)
}

/// Pages `pages` [n] back to bf16 keys and values [n][16][H][D] by the engine's kernel (glyd_gpu_kv_unpack: a warp a (page, head-page), 4 a CTA; one stream:
/// the keys alone, into `out_k`). False where the library takes it.
///
/// # Safety
/// As `GpuOps::kv_unpack`; a context with the engine's module is current.
#[allow(clippy::too_many_arguments)]
pub unsafe fn kv_unpack(
    g: &crate::ffi::KvGeom,
    (pool, kbase, vbase): (u64, u64, u64),
    pages: u64,
    n: i64,
    (out_k, out_v): (u64, u64),
    cs: Stream,
) -> Result<bool> {
    use crate::dev::here;
    use crate::kt;
    use crate::launch;
    let one = g.dim & KV_ONE != 0;
    let hp = g.heads * (kv_dim(g.dim) / KV_D);
    if cfg!(glyd_hip) || n < 0 || n * hp > i32::MAX as i64 || !kv_pools_ok(g, pool) {
        return Ok(false);
    }
    if n == 0 {
        return Ok(true);
    }
    if pages == 0 || kbase == 0 || out_k == 0 || (!one && (vbase == 0 || out_v == 0)) {
        return Ok(false);
    }
    let (ent, nc) = kv_escape(g.escape, g.nchunks);
    let raw = kv_raw(g.raw, g.nslabs, hp);
    let (vb, ov) = if one { (kbase, out_k) } else { (vbase, out_v) };
    let grid = [((n * hp + 3) / 4) as u32, 1, 1];
    launch!(here::<kt::fold_kv_unpack>()?, grid, [128, 1, 1], cs; pool, g.page_stride, pages, n as i32, kbase, vb, hp as i32, g.arena as i32, ent, nc, raw, out_k, ov, one as i32)?;
    Ok(true)
}

/// Pages `pages` [n] into the bf16 staging cache `out` [blocks][16][H][2 D], page j into block `dst`[j], by the engine's kernel (glyd_gpu_kv_gather). False
/// where the library takes it.
///
/// # Safety
/// As `GpuOps::kv_gather`; a context with the engine's module is current.
#[allow(clippy::too_many_arguments)]
pub unsafe fn kv_gather(g: &crate::ffi::KvGeom, (pool, kbase, vbase): (u64, u64, u64), (pages, dst, n): (u64, u64, i64), out: u64, cs: Stream) -> Result<bool> {
    use crate::dev::here;
    use crate::kt;
    use crate::launch;
    let hp = g.heads * (kv_dim(g.dim) / KV_D);
    if cfg!(glyd_hip) || g.dim & KV_ONE != 0 || n < 0 || n * hp > i32::MAX as i64 || !kv_pools_ok(g, pool) {
        return Ok(false);
    }
    if n == 0 {
        return Ok(true);
    }
    if pages == 0 || dst == 0 || kbase == 0 || vbase == 0 || out == 0 || out % 4 != 0 {
        return Ok(false);
    }
    let (ent, nc) = kv_escape(g.escape, g.nchunks);
    let raw = kv_raw(g.raw, g.nslabs, hp);
    let (grid, blk) = ([((n * hp + 3) / 4) as u32, 1, 1], [128u32, 1, 1]);
    let args = (pool, g.page_stride, pages, dst, n as i32, kbase, vbase, hp as i32, g.arena as i32, ent, nc, raw, out);
    match g.dim / KV_D {
        1 => {
            launch!(here::<kt::fold_kv_gather_1>()?, grid, blk, cs; args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7, args.8, args.9, args.10, args.11, args.12)
        }
        2 => {
            launch!(here::<kt::fold_kv_gather_2>()?, grid, blk, cs; args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7, args.8, args.9, args.10, args.11, args.12)
        }
        _ => {
            launch!(here::<kt::fold_kv_gather_4>()?, grid, blk, cs; args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7, args.8, args.9, args.10, args.11, args.12)
        }
    }?;
    Ok(true)
}

/// A kernel's arguments by value, each in an 8-byte slot (a 4-byte one in the slot's low bytes: little-endian), for kernels picked by name at run time (the
/// experts' many instances); `the_experts_arguments_are_the_kernels` holds the order and types to the kernels' own (crate::kt).
#[derive(Default)]
struct Slots(Vec<u64>, String);

impl Slots {
    fn p(mut self, v: u64) -> Self {
        self.0.push(v);
        self.1.push_str("u64, ");
        self
    }
    fn i64(mut self, v: i64) -> Self {
        self.0.push(v as u64);
        self.1.push_str("i64, ");
        self
    }
    fn i32(mut self, v: i32) -> Self {
        self.0.push(v as u32 as u64);
        self.1.push_str("i32, ");
        self
    }
    fn u32(mut self, v: u32) -> Self {
        self.0.push(v as u64);
        self.1.push_str("u32, ");
        self
    }
}

impl crate::dev::KernelArgs for Slots {
    fn ptrs(&mut self) -> Vec<*mut std::ffi::c_void> {
        self.0.iter_mut().map(|v| v as *mut u64 as *mut std::ffi::c_void).collect()
    }
}

/// The experts' layouts the grouped products take, by the name their kernels carry (glyd_gpu_mma_moe reads Int6, Int5, Int4, Int4H, Fp4, Fp4R and the tiered
/// layout's words, glyd_gpu_mma12_moe the 12-bit layout's, glyd_gpu_mma_moe_placed the tiered layout's places).
fn moe_name(u: Unpacker) -> Option<&'static str> {
    Some(match u {
        Unpacker::Tiered => "tiered",
        Unpacker::Int6 => "int6",
        Unpacker::Int5 => "int5",
        Unpacker::Int4 => "int4",
        Unpacker::Int4H(_) => "int4h",
        Unpacker::Fp4 => "fp4",
        Unpacker::Fp4R { .. } => "fp4r",
        Unpacker::Twelve { .. } => "twelve",
        _ => return None,
    })
}

/// A layout's words, the first arguments of its experts' kernels (`placed`: a tiered stack's places, their steps an expert and their blocks' stride).
fn layout_args(u: Unpacker, w: &Pack, placed: Option<(i64, i64)>) -> Option<Slots> {
    let s = Slots::default().p(w.data);
    Some(match (u, w.words) {
        (Unpacker::Tiered, Words::Tiered { blocks, block_base, tiers: t }) => {
            let s = s.p(blocks).p(block_base).u32(t[0]).u32(t[1]).u32(t[2]);
            match placed {
                Some((per, stride)) => s.i64(per).i64(stride),
                None => s,
            }
        }
        (Unpacker::Compact, Words::Tiered { blocks, block_base, tiers: t }) => s.p(blocks).p(block_base).u32(t[0]).u32(t[1]).u32(t[2]),
        (Unpacker::Int6 | Unpacker::Int5 | Unpacker::Int4 | Unpacker::Fp4 | Unpacker::Int3, Words::Tiered { blocks, .. }) => s.p(blocks),
        (Unpacker::Int4H(base), Words::Tiered { blocks, .. }) => s.p(blocks).u32(base),
        (Unpacker::Fp4R { ks, mg, sh }, Words::Tiered { blocks, .. }) => s.p(blocks).u32(ks).u32(mg).u32(sh),
        (Unpacker::Twelve { hb4, xmax }, Words::Twelve { exc, exc_base, .. }) => s.p(exc).p(exc_base).u32(hb4).i32(xmax),
        _ => return None,
    })
}

/// A kernel name made at run time, kept for the process (the experts' instances: a few hundred names at most).
fn leaked(name: String) -> &'static str {
    static NAMES: std::sync::Mutex<Vec<&'static str>> = std::sync::Mutex::new(Vec::new());
    let mut names = NAMES.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(n) = names.iter().find(|n| **n == name) {
        return n;
    }
    let n: &'static str = Box::leak(name.into_boxed_str());
    names.push(n);
    n
}

/// The GPU's kind as the experts' plans read it: an HBM part (hbm_gpu: compute capability 8.0, 9.0 or 10.0), an H100 (9.0 and "H100" in its name), the A100's
/// architecture (8.0).
fn moe_gpu() -> Result<(bool, bool, bool)> {
    let dev = D::current_device()?;
    let (major, minor) = (D::attribute(dev, Attr::CcMajor)?, D::attribute(dev, Attr::CcMinor)?);
    let hbm = minor == 0 && (8..=10).contains(&major);
    let h100 = (major, minor) == (9, 0) && has_word(&D::name(dev)?, "H100");
    Ok((hbm, h100, (major, minor) == (8, 0)))
}

/// The blocks splitting K for the experts' products (moe_splits): up to the blocks the GPU holds at once (an SM's one, an HBM part's two; GLYD_GPU_MOE_SLOTS),
/// at least 256 columns each.
fn moe_splits(units: i64, k: i64, hbm: bool) -> Result<i64> {
    let forced = tuned(crate::tune!("GLYD_GPU_MOE_SLOTS"), 0);
    let slots = if forced != 0 { forced } else { sm_count()? * if hbm { 2 } else { 1 } };
    Ok((slots / units).min(k / 256).max(1))
}

/// `GpuOps::moe_workspace` by the library's plan (mma_moe_run's need: the weighted sums' Y32 and the splits' parts, [P][O] floats each), where the 12-bit
/// layout's wgmma route cannot be taken (a GPU of compute capability 9.0 keeps the library's: the route's tail); None: the library's.
pub fn moe_workspace(twelve: bool, experts: i64, o: i64, cols: i64, t: i64, k: i64, act: i64, weighted: bool) -> Result<Option<usize>> {
    if cfg!(glyd_hip) || !(1..=12288).contains(&experts) || o < 64 || o % 64 != 0 || cols < 16 || cols % 16 != 0 || t < 0 || k < 1 || !(0..=2).contains(&act) {
        return Ok(None);
    }
    if act != 0 && (o % 128 != 0 || weighted) {
        return Ok(None);
    }
    let dev = D::current_device()?;
    if twelve && (D::attribute(dev, Attr::CcMajor)?, D::attribute(dev, Attr::CcMinor)?) == (9, 0) {
        return Ok(None);
    }
    let (hbm, ..) = moe_gpu()?;
    let p = t * k;
    let units = if act != 0 { o / 128 } else { o / 64 };
    let splits = if p > 0 { moe_splits(units * experts.min(p), cols, hbm)? } else { 1 };
    Ok(Some(((weighted as i64 + if splits > 1 { splits } else { 0 }) * p * o * 4) as usize))
}

/// `GpuOps::moe` and `moe_placed` (y = the experts' products of the plan's pairs, act 0, 1 SiLU or 2 GELU, weighted sums by `weights` and `ids`) by the
/// engine's kernels where the library's plan takes its step products (mma_moe_kernel, mma_moe_rows_kernel) in their default variants; false where the
/// library takes it (the prompt kernel past GLYD_GPU_MOE_BIG pairs an expert, the 12-bit layout's wgmma route on Hopper, a tuning variable's variant,
/// arguments it refuses).
///
/// # Safety
/// As `GpuOps::moe`; a context with the engine's module is current.
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe(
    w: &Pack,
    placed: Option<i64>,
    experts: i64,
    (x, t, k, gather): (u64, i64, i64, bool),
    (plan, act, weights, ids): (u64, i64, u64, u64),
    y: u64,
    (ws, ws_bytes, done): (u64, usize, u64),
    cs: Stream,
) -> Result<bool> {
    let Ok(u) = unpacker(w) else { return Ok(false) };
    let Some(name) = moe_name(u) else { return Ok(false) };
    if placed.is_some_and(|stride| u != Unpacker::Tiered || stride < 0 || stride % 4 != 0) {
        return Ok(false);
    }
    let (o, kk) = (w.rows / experts.max(1), w.cols);
    let twelve = w.is_twelve();
    let Some(need) = moe_workspace(twelve, experts, o, kk, t, k, act, weights != 0)? else { return Ok(false) };
    let p = t * k;
    let units = if act != 0 { o / 128 } else { o / 64 };
    let hits = experts.min(p);
    let (hbm, h100, a100) = moe_gpu()?;
    let splits = if p > 0 { moe_splits(units * hits, kk, hbm)? } else { 1 };
    if (need > 0 && (ws == 0 || ws_bytes < need)) || (weights != 0 && ids == 0) || (splits > 1 && done == 0) {
        return Ok(false);
    }
    if p == 0 {
        return Ok(true);
    }
    // the plan's choices as mma_moe_run makes them, the variables at their defaults
    let mt = if t <= 16 {
        1
    } else if t <= 32 {
        2
    } else {
        4
    };
    let big_set = tuned(crate::tune!("GLYD_GPU_MOE_BIG"), -1);
    let big_default = if big_set >= 0 {
        big_set
    } else if a100 {
        64
    } else if h100 {
        96
    } else {
        48
    };
    let big = if act != 0 { tuned(crate::tune!("GLYD_GPU_MOE_BIG_GU"), big_default) } else { tuned(crate::tune!("GLYD_GPU_MOE_BIG_DN"), big_default) };
    let lname = if placed.is_some() { "placed" } else { name };
    let per_stride = placed.map(|stride| (o * kk / 1024, stride));
    let Some(lead) = layout_args(u, w, per_stride) else { return Ok(false) };
    let y32 = if weights != 0 { ws } else { 0 };
    if p >= big * experts && kk % 64 == 0 && x % 16 == 0 {
        // the prompt kernel (mma_moe_big: 128 pairs by two row blocks a block, blockIdx.z the experts hit; GLYD_GPU_BIG_TM=256, a development build's, the library's)
        if tuned(crate::tune!("GLYD_GPU_BIG_TM"), 128) == 256 {
            return Ok(false);
        }
        let kname = leaked(format!("fold_big_{lname}_moe"));
        let (threads, tm, smem) = big_shape(4, 4, 3, 2, 64);
        per_sm(kname, threads, smem)?;
        let plan_args = MoePairs {
            plan,
            e: experts as i32,
            k,
            gather: gather as i32,
            act: act as i32,
            w: weights,
            wf32: 1,
            xahead: tuned(crate::tune!("GLYD_GPU_BIG_XAHEAD"), 0) as i32,
            pf: tuned(crate::tune!("GLYD_GPU_BIG_PF"), 0).clamp(0, 8) as i32,
            skip: (tuned(crate::tune!("GLYD_GPU_BIG_SKIP"), 1) != 0) as i32,
        };
        let grid = [((t + tm - 1) / tm) as u32, (if act != 0 { o / 128 } else { (o / 64 + 1) / 2 }) as u32, hits as u32];
        launch_slots(kname, grid, threads, smem, lead.i64(o).i64(kk).i64(0).i64(0).p(x).p(0).p(y).p(y32), Some(plan_args), cs)?;
        if weights != 0 {
            use crate::kt;
            use crate::launch;
            launch!(crate::dev::here::<kt::fold_moe_sum>()?, [((t * o / 4 + 255) / 256) as u32, 1, 1], [256, 1, 1], cs; ws, ids, experts as i32, t, k, o, y)?;
        }
        return Ok(true);
    }
    let pf = tuned(crate::tune!("GLYD_GPU_PF"), 0).clamp(0, 16);
    let patch_b = tuned(crate::tune!("GLYD_GPU_PATCH_B"), 0) != 0;
    let skip = tuned(crate::tune!("GLYD_GPU_NA"), (h100 || a100) as i64) != 0;
    let stages = tuned(crate::tune!("GLYD_GPU_STAGES"), 0);
    let table = !twelve;
    let depth = tuned(crate::tune!("GLYD_GPU_DEPTH"), if hbm && !matches!(u, Unpacker::Fp4R { .. }) { 2 } else { 1 });
    if pf > 0 || (!table && patch_b) || stages == 4 || stages == 8 {
        return Ok(false);
    }
    let v = (skip as i64 * 2) | if table && mt == 1 && depth == 2 { 8 } else { 0 };
    let rows_set = tuned(crate::tune!("GLYD_GPU_MOE_ROWS"), 1);
    let rows_from = match rows_set {
        0 => i64::MAX,
        r if r >= 2 => r,
        _ if h100 || a100 => 17,
        _ if hbm => i64::MAX,
        _ => 33,
    };
    let rows = act == 0 && mt >= 2 && t >= rows_from;
    let o_args = |s: Slots| s.i64(o).i64(kk).p(plan).i32(experts as i32).i64(k).i32(gather as i32).p(x).p(0).p(weights).i32(1).p(y).p(y32);
    if rows {
        let kname = leaked(format!("fold_moe_rows_{lname}_{mt}_{v}"));
        let mut a = o_args(lead).i32(pf as i32);
        crate::dev::here_named(kname)?.launch_smem(
            [((o / 64 + 7) / 8) as u32, hits as u32, 1],
            [256, 1, 1],
            8 * 1040,
            cs,
            &mut crate::dev::KernelArgs::ptrs(&mut a),
        )?;
    } else {
        let kname = leaked(format!("fold_moe_{lname}_{mt}_{act}_{v}"));
        let shared = (4 * mt * 16 * 65 * 4 + 8 * 1040) as u32;
        per_sm(kname, 256, shared)?;
        let parts = if splits > 1 { ws + if weights != 0 { (p * o * 4) as u64 } else { 0 } } else { 0 };
        let mut a = o_args(lead).i64(p).p(parts).p(done).i32(pf as i32);
        crate::dev::here_named(kname)?.launch_smem(
            [units as u32, hits as u32, splits as u32],
            [256, 1, 1],
            shared,
            cs,
            &mut crate::dev::KernelArgs::ptrs(&mut a),
        )?;
    }
    if weights != 0 {
        use crate::kt;
        use crate::launch;
        launch!(crate::dev::here::<kt::fold_moe_sum>()?, [((t * o / 4 + 255) / 256) as u32, 1, 1], [256, 1, 1], cs; ws, ids, experts as i32, t, k, o, y)?;
    }
    Ok(true)
}

/// f8c::F8Pk (gpu/glyd_gpu_kernels.h), field for field: the FP8 pack's matrix.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct F8Pk {
    pub w: u64,
    pub ld: i64,
    pub o: i32,
    pub k: i32,
    pub kt: i32,
    pub vec: i32,
    pub plain: i32,
}

/// f8c::F8Un (gpu/glyd_gpu_kernels.h), field for field: the FP8 unpack's pack and output.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct F8Un {
    pub core: u64,
    pub heads: u64,
    pub var: u64,
    pub out: u64,
    pub ids: u64,
    pub ld: i64,
    pub item: i64,
    pub o: i32,
    pub k: i32,
    pub kt: i32,
    pub per: i32,
    pub e: i32,
    pub vec: i32,
}

/// The FP8 format's patches of a matrix [o, k] (f8_shape: 64 rows by 128 columns, at most i32::MAX of them); None where the library refuses the shape.
fn f8_patches(o: i64, k: i64) -> Option<i64> {
    if o < 1 || k < 1 || o > i32::MAX as i64 - 64 || k > i32::MAX as i64 - 128 {
        return None;
    }
    let n = ((o + 63) / 64) * ((k + 127) / 128);
    (n <= i32::MAX as i64).then_some(n)
}

/// The FP8 format's zero words after the last patch's part of var (f8c::GUARD).
const F8_GUARD: i64 = 8;

/// `GpuOps::f8h_plain`'s choice (the library's g_f8h_plain: -1 the GPU's own).
static F8H_PLAIN: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);

/// Layout H's packs plain or coded (glyd_gpu_f8h_plain): Some(true) plain, Some(false) coded, None the GPU's own.
pub fn f8h_plain_set(plain: Option<bool>) {
    F8H_PLAIN.store(plain.map_or(-1, |p| p as i32), std::sync::atomic::Ordering::Relaxed);
}

/// Whether layout H's packs are plain (f8h_plain): GLYD_F8H_PLAIN (development builds), else `f8h_plain_set`'s, else an HBM GPU's own (plain).
fn f8h_plain() -> Result<bool> {
    let set = F8H_PLAIN.load(std::sync::atomic::Ordering::Relaxed);
    let fallback = if set >= 0 { set as i64 } else { moe_gpu()?.0 as i64 };
    Ok(tuned(crate::tune!("GLYD_F8H_PLAIN"), fallback) != 0)
}

/// `GpuOps::f8_size_workspace` (glyd_gpu_f8_size_workspace): 16 bytes for a shape the format takes.
pub fn f8_size_workspace(o: i64, k: i64) -> Result<usize> {
    match f8_patches(o, k) {
        Some(_) => Ok(16),
        None => Err(invalid("f8_size_workspace")),
    }
}

/// The lossless FP8 format's heads of W [o, k] and the var words its pack takes, by the engine's kernels (glyd_gpu_f8_size, layout H's f8h_size: a CTA a
/// patch, then the scan of the heads, its total read back: the stream is waited for). None where the library takes it.
///
/// # Safety
/// As `GpuOps::f8_size`; a context with the engine's module is current.
#[allow(clippy::too_many_arguments)]
pub unsafe fn f8_size(h: bool, w: u64, ld: i64, o: i64, k: i64, heads: u64, (ws, ws_bytes): (u64, usize), cs: Stream) -> Result<Option<i64>> {
    use crate::dev::here;
    use crate::kt;
    use crate::launch;
    let Some(patches) = f8_patches(o, k) else { return Ok(None) };
    if cfg!(glyd_hip) || w == 0 || heads == 0 || ld < k || heads % 16 != 0 || ws == 0 || ws % 8 != 0 || ws_bytes < 16 {
        return Ok(None);
    }
    let p = F8Pk {
        w,
        ld,
        o: o as i32,
        k: k as i32,
        kt: ((k + 127) / 128) as i32,
        vec: (w % 16 == 0 && ld % 16 == 0) as i32,
        plain: if h { f8h_plain()? as i32 } else { 0 },
    };
    let grid = [patches as u32, 1, 1];
    if h {
        launch!(here::<kt::fold_f8h_head>()?, grid, [256, 1, 1], cs; p, heads)?;
    } else {
        launch!(here::<kt::fold_f8_head>()?, grid, [256, 1, 1], cs; p, heads)?;
    }
    launch!(here::<kt::fold_f8_scan>()?, [1, 1, 1], [1024, 1, 1], cs; heads, patches, ws)?;
    let mut total = 0u64;
    D::d2h_async(&mut total as *mut u64 as *mut std::ffi::c_void, ws, 8, cs)?;
    D::stream_sync(cs)?;
    if total + F8_GUARD as u64 >= 1 << 32 {
        return Err(invalid("f8_size"));
    }
    Ok(Some(total as i64 + F8_GUARD))
}

/// W's patches into `core` and `var` from the heads `f8_size` made, by the engine's kernel (glyd_gpu_f8_pack, f8h_pack). False where the library takes it.
///
/// # Safety
/// As `GpuOps::f8_pack`; a context with the engine's module is current.
#[allow(clippy::too_many_arguments)]
pub unsafe fn f8_pack(h: bool, w: u64, ld: i64, o: i64, k: i64, heads: u64, (core, var, var_words): (u64, u64, i64), cs: Stream) -> Result<bool> {
    use crate::dev::here;
    use crate::kt;
    use crate::launch;
    let Some(patches) = f8_patches(o, k) else { return Ok(false) };
    if cfg!(glyd_hip)
        || w == 0
        || heads == 0
        || core == 0
        || var == 0
        || ld < k
        || !(F8_GUARD..1 << 32).contains(&var_words)
        || heads % 16 != 0
        || core % 16 != 0
    {
        return Ok(false);
    }
    let p = F8Pk {
        w,
        ld,
        o: o as i32,
        k: k as i32,
        kt: ((k + 127) / 128) as i32,
        vec: (w % 16 == 0 && ld % 16 == 0) as i32,
        plain: if h { f8h_plain()? as i32 } else { 0 },
    };
    let grid = [patches as u32, 1, 1];
    if h {
        launch!(here::<kt::fold_f8h_pack>()?, grid, [256, 1, 1], cs; p, heads, core, var, var_words)?;
    } else {
        launch!(here::<kt::fold_f8_pack>()?, grid, [256, 1, 1], cs; p, heads, core, var, var_words)?;
    }
    Ok(true)
}

/// W back, bit for bit, into `out` [o, k] (rows `ld` bytes apart) by the engine's kernel (glyd_gpu_f8_unpack: a CTA a patch; layout H's f8h_unpack, its CTAs
/// of 128). False where the library takes it.
///
/// # Safety
/// As `GpuOps::f8_unpack`; a context with the engine's module is current.
#[allow(clippy::too_many_arguments)]
pub unsafe fn f8_unpack(h: bool, (core, heads, var): (u64, u64, u64), o: i64, k: i64, (out, ld): (u64, i64), cs: Stream) -> Result<bool> {
    use crate::dev::here;
    use crate::kt;
    use crate::launch;
    let Some(patches) = f8_patches(o, k) else { return Ok(false) };
    if cfg!(glyd_hip) || core == 0 || heads == 0 || var == 0 || out == 0 || ld < k || core % 16 != 0 || heads % 16 != 0 {
        return Ok(false);
    }
    let p = F8Un {
        core,
        heads,
        var,
        out,
        ids: 0,
        ld,
        item: o * ld,
        o: o as i32,
        k: k as i32,
        kt: ((k + 127) / 128) as i32,
        per: patches as i32,
        e: 1,
        vec: (out % 16 == 0 && ld % 16 == 0) as i32,
    };
    if h {
        launch!(here::<kt::fold_f8h_unpack>()?, [patches as u32, 1, 1], [128, 1, 1], cs; p)?;
    } else {
        launch!(here::<kt::fold_f8_unpack>()?, [patches as u32, 1, 1], [256, 1, 1], cs; p)?;
    }
    Ok(true)
}

/// f8c::F8Lin (gpu/glyd_gpu_kernels.h), field for field: the FP8 products' parameters, built as glyd_gpu_f8_linear and _f8_moe build them.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct F8Lin {
    pub core: u64,
    pub heads: u64,
    pub var: u64,
    pub scale: u64,
    pub x: u64,
    pub y: u64,
    pub xq: u64,
    pub sx: u64,
    pub parts: u64,
    pub done: u64,
    pub lds: i64,
    pub ldx: i64,
    pub ldy: i64,
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub kt: i32,
    pub rg: i32,
    pub ctas: i32,
    pub xvec: i32,
    pub plan: u64,
    pub order: u64,
    pub e: i32,
    pub gk: i32,
    pub gather: i32,
}

/// The FP8 product's threads a CTA (f8c::LNT, 8 warps), its shared memory before the X slots (f8_lsmem<2>: three slots of heads, a warp's two of core and two
/// of the variable part; layout H's f8_lsmem<2, true>, its cores of 448 words), and layout H's one-token kernels' (8 warps of L1W words, f8b_lin1's BW).
const F8_LNT: u32 = 256;
const F8_LSMEM: u32 = 4 * (3 * 16 + 8 * (2 * 384 + 2 * 80));
const F8H_LSMEM: u32 = 4 * (3 * 16 + 8 * (2 * 448 + 2 * 80));
const F8H_ONE_CODED: u32 = 8 * (2 * 448 + 2 * 80 + 2 * 64) * 4;
const F8H_ONE_PLAIN: u32 = 8 * (2 * 512 + 2 * 64) * 4;

/// The product's shared memory for `m` tokens (f8_lin_shared at the default plan: the two X slots of m rows of 256 bytes after the rest).
fn f8_shared(m: i64, h: bool) -> u32 {
    (if h { F8H_LSMEM } else { F8_LSMEM }) + 2 * 256 * m as u32
}

/// The 8-token tiles a warp for `m` tokens (f8_lin_ntl).
fn f8_ntl(m: i64) -> u32 {
    match m {
        ..=8 => 1,
        9..=16 => 2,
        17..=32 => 4,
        _ => 8,
    }
}

/// The product's work for W [n, k]: its row groups of 128 rows times its patches across K (W in f8_lin_ctas), the most CTAs it is split over.
fn f8_work(n: i64, k: i64) -> i64 {
    ((n + 63) / 64 * 4 + 7) / 8 * ((k + 127) / 128)
}

/// The sums of a CTA's first and last row group where the grid splits the work (f8_lin_parts): bytes for `ctas` CTAs and `m` tokens.
fn f8_parts(ctas: i64, m: i64) -> i64 {
    if ctas > 1 {
        ctas * 2 * m * 128 * 4
    } else {
        0
    }
}

/// Layout H's one-token kernel (f8h_one: GLYD_F8H_ONE in a development build, else 3, f8b_lin1, for plain packs and 1, f8h_lin1, for coded ones; 0 none).
fn f8h_one() -> Result<i64> {
    let m = tuned(crate::tune!("GLYD_F8H_ONE"), if f8h_plain()? { 3 } else { 1 });
    Ok(if (0..=4).contains(&m) { m } else { 1 })
}

/// Whether the library's FP8 products are at their default plans, which the engine's kernels are: no development build's variable asks for X as e4m3
/// (GLYD_F8_LIN_A8), a decode ahead (_AHEAD), a spills' pass (_DENSE), three slots (GLYD_F8_SLOTS), 2 CTAs an SM of layout H's one-token kernel
/// (GLYD_F8H_CTAS) or its research kernels (GLYD_F8H_ONE 2 and 4); a CUDA build.
fn f8_default() -> Result<bool> {
    Ok(!cfg!(glyd_hip)
        && tuned(crate::tune!("GLYD_F8_LIN_A8"), 0) != 1
        && tuned(crate::tune!("GLYD_F8_LIN_AHEAD"), 0) != 1
        && tuned(crate::tune!("GLYD_F8_LIN_DENSE"), 0) != 1
        && tuned(crate::tune!("GLYD_F8_SLOTS"), 2) != 3
        && tuned(crate::tune!("GLYD_F8H_CTAS"), 3) != 2
        && !matches!(f8h_one()?, 2 | 4))
}

/// The FP8 products' kernels in the engine's module (kernels/fold.cu): `which` 0-4 one matrix's for its kind of scales (block, tensor, mx, ds32, channel),
/// 5 the experts', 6 and 7 layout H's one matrix and experts; `ntl` 8-token tiles a warp.
fn f8_name(which: usize, ntl: u32) -> &'static str {
    const TAGS: [&str; 8] = ["block", "tensor", "mx", "ds32", "channel", "moe", "h", "hmoe"];
    leaked(format!("fold_f8_{}_{ntl}", TAGS[which]))
}

/// Layout H's one-token kernel in the engine's module: f8b_lin1 (plain packs) or f8h_lin1, of one matrix or the experts' (`g`).
fn f8h_one_name(plain: bool, g: bool) -> &'static str {
    leaked(format!("fold_f8h_one_{}_{}", if plain { "plain" } else { "coded" }, g as u8))
}

/// The blocks an SM holds of kernel `which` for `m` tokens, its shared memory allowed to 64 tokens' first (f8_lin_allow, f8h_resident), and the SMs.
fn f8_resident(which: usize, m: i64) -> Result<(i64, i64)> {
    let h = which >= 6;
    per_sm_most(f8_name(which, f8_ntl(m)), F8_LNT, f8_shared(m, h), f8_shared(64, h))
}

/// The grid of one matrix's product for kind `kind` (f8_lin_ctas: as many CTAs as the GPU holds at once, at most the work over GLYD_F8_LIN_MIN; GLYD_F8_LIN_GRID's
/// in a development build; None: the most of the five kinds').
fn f8_lin_ctas(n: i64, k: i64, m: i64, kind: Option<i64>) -> Result<i64> {
    let w = f8_work(n, k);
    let c = tuned(crate::tune!("GLYD_F8_LIN_GRID"), 0);
    if c > 0 {
        return Ok(c.min(w));
    }
    let Some(kind) = kind else {
        let mut most = 1;
        for kd in 0..5 {
            most = most.max(f8_lin_ctas(n, k, m, Some(kd))?);
        }
        return Ok(most);
    };
    let (r, sms) = f8_resident(kind as usize, m)?;
    let least = tuned(crate::tune!("GLYD_F8_LIN_MIN"), 1).max(1);
    Ok((sms * r).min(w / least).max(1))
}

/// The experts' product's CTAs a slot (f8_moe_ctas: the GPU's at once shared among the slots, at most the work; GLYD_F8_MOE_GRID's in a development build).
fn f8_moe_ctas(n: i64, k: i64, m: i64, slots: i64) -> Result<i64> {
    let w = f8_work(n, k);
    let c = tuned(crate::tune!("GLYD_F8_MOE_GRID"), 0);
    if c > 0 {
        return Ok(c.min(w));
    }
    let (r, sms) = f8_resident(5, m)?;
    Ok((sms * r / slots.max(1)).min(w).max(1))
}

/// Layout H's grid at most (glyd_gpu_f8h_linear, _f8h_moe: the CTAs an SM of its own kernel, shared among the slots of the experts' product).
fn f8h_ctas(n: i64, k: i64, m: i64, slots: Option<i64>) -> Result<i64> {
    let (r, sms) = f8_resident(if slots.is_some() { 7 } else { 6 }, m)?;
    Ok((sms * r / slots.unwrap_or(1)).min(f8_work(n, k)).max(1))
}

/// Layout H's one-token kernel's CTAs a slot (f8h_one_ctas; `slots` 0: one matrix).
fn f8h_one_ctas(n: i64, k: i64, slots: i64) -> Result<i64> {
    let plain = f8h_one()? == 3;
    let smem = if plain { F8H_ONE_PLAIN } else { F8H_ONE_CODED };
    let (r, sms) = per_sm_most(f8h_one_name(plain, slots > 0), F8_LNT, smem, smem)?;
    Ok((sms * r / slots.max(1)).min(f8_work(n, k)).max(1))
}

/// `GpuOps::f8_linear_workspace` by the engine's plans (glyd_gpu_f8_linear_workspace: the sums of the grid of the most CTAs of the five kinds and of the
/// one-token kernel); None: the library's (a shape or count it refuses, or a plan a variable asks for).
pub fn f8_linear_workspace(n: i64, k: i64, m: i64) -> Result<Option<usize>> {
    if !(1..=64).contains(&m) || f8_patches(n, k).is_none() || !f8_default()? {
        return Ok(None);
    }
    let one = if m == 1 { f8h_one_ctas(n, k, 0)? } else { 1 };
    Ok(Some(((f8_parts(f8_lin_ctas(n, k, m, None)?.max(one), m) + 15) / 16 * 16) as usize))
}

/// `GpuOps::f8_moe_workspace` by the engine's plans (glyd_gpu_f8_moe_workspace: the sums of each slot's grid); None: the library's.
pub fn f8_moe_workspace(n: i64, k: i64, m: i64, slots: i64) -> Result<Option<usize>> {
    if !(1..=64).contains(&m) || !(1..=65535).contains(&slots) || f8_patches(n, k).is_none() || !f8_default()? {
        return Ok(None);
    }
    let one = if m == 1 { f8h_one_ctas(n, k, slots)? } else { 1 };
    Ok(Some((slots * f8_parts(f8_moe_ctas(n, k, m, slots)?.max(one), m)) as usize))
}

/// A launch of an FP8 product's kernel with its parameters, its shared memory allowed to `most` first (the library's cudaFuncSetAttribute at each launch).
///
/// # Safety
/// `p` names live buffers of the product's sizes; a context with the engine's module is current.
unsafe fn f8_launch(name: &'static str, grid: [u32; 3], (smem, most): (u32, u32), mut p: F8Lin, cs: Stream) -> Result<()> {
    let k = crate::dev::here_named(name)?;
    k.allow_smem(most)?;
    k.launch_smem(grid, [F8_LNT, 1, 1], smem, cs, &mut [&mut p as *mut F8Lin as *mut std::ffi::c_void])
}

/// Whether a workspace of `have` bytes at `ws` holds `need` (the library's fits; none needed: any).
fn fits(ws: u64, have: usize, need: i64) -> bool {
    need == 0 || (ws != 0 && have as i64 >= need)
}

/// `GpuOps::f8_linear` (Y [m, n] = X [m, k] W^T, X and Y rows k and n apart) by the engine's kernels at the library's default plan (glyd_gpu_f8_linear, layout H's
/// glyd_gpu_f8h_linear and its one-token kernel); false where the library takes it (arguments it refuses: its own error then; a plan a variable asks for).
///
/// # Safety
/// As `GpuOps::f8_linear`; a context with the engine's module is current.
pub unsafe fn f8_linear(
    h: bool,
    (core, heads, var): (u64, u64, u64),
    (n, k, kind, scale, lds): (i64, i64, i64, u64, i64),
    (x, m, y): (u64, i64, u64),
    (ws, ws_bytes, done): (u64, usize, u64),
    cs: Stream,
) -> Result<bool> {
    let (ldx, ldy) = (k, n);
    if f8_patches(n, k).is_none()
        || [core, heads, var, scale, x, y].contains(&0)
        || !(1..=64).contains(&m)
        || core % 16 != 0
        || heads % 16 != 0
        || var % 4 != 0
        || x % 2 != 0
        || y % 2 != 0
    {
        return Ok(false);
    }
    let (k128, k32) = ((k + 127) / 128, (k + 31) / 32);
    let scales_ok = if h {
        kind == 0 && lds >= k128 && scale % 4 == 0
    } else {
        match kind {
            0 => lds >= k128 && scale % 4 == 0,
            1 | 4 => scale % 4 == 0,
            2 | 3 => lds >= k32,
            _ => false,
        }
    };
    if !scales_ok || !f8_default()? {
        return Ok(false);
    }
    let mut ctas = f8_lin_ctas(n, k, m, Some(kind))?;
    if h {
        ctas = ctas.min(f8h_ctas(n, k, m, None)?);
    }
    let need = (f8_parts(ctas, m) + 15) / 16 * 16;
    if (ctas > 1 && done == 0) || !fits(ws, ws_bytes, need) || (need > 0 && ws % 16 != 0) {
        return Ok(false);
    }
    let mut p = F8Lin {
        core,
        heads,
        var,
        scale,
        x,
        y,
        parts: ws,
        done,
        lds,
        ldx,
        ldy,
        m: m as i32,
        n: n as i32,
        k: k as i32,
        kt: k128 as i32,
        rg: (((n + 63) / 64 * 4 + 7) / 8) as i32,
        ctas: ctas as i32,
        xvec: (x % 16 == 0 && ldx % 8 == 0) as i32,
        ..Default::default()
    };
    if h && m == 1 && f8h_one()? != 0 {
        let plain = f8h_one()? == 3;
        let one = f8h_one_ctas(n, k, 0)?;
        if (one > 1 && done == 0) || !fits(ws, ws_bytes, (f8_parts(one, 1) + 15) / 16 * 16) {
            return Ok(false);
        }
        p.ctas = one as i32;
        let smem = if plain { F8H_ONE_PLAIN } else { F8H_ONE_CODED };
        f8_launch(f8h_one_name(plain, false), [one as u32, 1, 1], (smem, smem), p, cs)?;
        return Ok(true);
    }
    let which = if h { 6 } else { kind as usize };
    f8_launch(f8_name(which, f8_ntl(m)), [ctas as u32, 1, 1], (f8_shared(m, h), f8_shared(64, h)), p, cs)?;
    Ok(true)
}

/// `GpuOps::f8_moe` (each pair of `plan` times its expert's matrix [n, k], block scales ceil(k / 128) a row group's, into its row of Y) by the engine's
/// kernels at the library's default plan (glyd_gpu_f8_moe, layout H's glyd_gpu_f8h_moe and its one-token kernel); false where the library takes it.
///
/// # Safety
/// As `GpuOps::f8_moe`; a context with the engine's module is current.
pub unsafe fn f8_moe(
    h: bool,
    (core, heads, var, experts, n, k, scale): (u64, u64, u64, i64, i64, i64, u64),
    (x, plan, k_, gather): (u64, u64, i64, bool),
    (m, slots, y): (i64, i64, u64),
    (ws, ws_bytes, done): (u64, usize, u64),
    cs: Stream,
) -> Result<bool> {
    let (ldx, ldy, lds) = (k, n, (k + 127) / 128);
    if [core, heads, var, scale, x, y, plan].contains(&0)
        || !(1..=i32::MAX as i64).contains(&experts)
        || n % 64 != 0
        || f8_patches(n, k).is_none()
        || !(1..=64).contains(&m)
        || !(1..=65535).contains(&slots)
        || !(1..=i32::MAX as i64).contains(&k_)
        || core % 16 != 0
        || heads % 16 != 0
        || var % 4 != 0
        || scale % 4 != 0
        || x % 2 != 0
        || y % 2 != 0
        || !f8_default()?
    {
        return Ok(false);
    }
    let mut ctas = f8_moe_ctas(n, k, m, slots)?;
    if h {
        ctas = ctas.min(f8h_ctas(n, k, m, Some(slots))?);
    }
    let need = slots * f8_parts(ctas, m);
    if (ctas > 1 && done == 0) || !fits(ws, ws_bytes, need) || (need > 0 && ws % 16 != 0) {
        return Ok(false);
    }
    let mut p = F8Lin {
        core,
        heads,
        var,
        scale,
        x,
        y,
        parts: ws,
        done,
        lds,
        ldx,
        ldy,
        m: m as i32,
        n: n as i32,
        k: k as i32,
        kt: lds as i32,
        rg: (((n + 63) / 64 * 4 + 7) / 8) as i32,
        ctas: ctas as i32,
        xvec: (x % 16 == 0 && ldx % 8 == 0) as i32,
        plan,
        e: experts as i32,
        gk: k_ as i32,
        gather: gather as i32,
        ..Default::default()
    };
    if h && m == 1 && f8h_one()? != 0 {
        let plain = f8h_one()? == 3;
        let one = f8h_one_ctas(n, k, slots)?;
        if (one > 1 && done == 0) || !fits(ws, ws_bytes, slots * f8_parts(one, 1)) {
            return Ok(false);
        }
        p.ctas = one as i32;
        let smem = if plain { F8H_ONE_PLAIN } else { F8H_ONE_CODED };
        f8_launch(f8h_one_name(plain, true), [one as u32, slots as u32, 1], (smem, smem), p, cs)?;
        return Ok(true);
    }
    f8_launch(f8_name(if h { 7 } else { 5 }, f8_ntl(m)), [ctas as u32, slots as u32, 1], (f8_shared(m, h), f8_shared(64, h)), p, cs)?;
    Ok(true)
}

/// foldcheck's wait for a check's launches: a fault comes back named by the launches it follows (so the failing kernel, the library's or the engine's,
/// is the one named).
#[cfg(feature = "dev")]
fn synced(dev: &crate::dev::Dev, what: impl FnOnce() -> String) -> Result<()> {
    dev.sync().map_err(|e| crate::ffi::Error { msg: format!("{}: {}", what(), e.msg), kind: e.kind })
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
    type Check = fn(&crate::dev::Dev, &crate::ffi::Lib) -> Result<(usize, Vec<String>)>;
    let checks: [(&str, Check); 7] = [
        ("the experts' plan (moe_way), the engine's kernels against the library's (one warp and one block, ids out of range among them)", route_check),
        (
            "the 12-bit pages' write, unpack and gather, and the decode and prompt attention over them and over a bf16 cache: the engine's kernels against the library's",
            kv_check,
        ),
        ("the lossless FP8 format's size, pack and unpack (both layouts), the engine's kernels against the library's", f8_check),
        ("the FP8 products (each kind of scales, both layouts, 1 to 64 tokens, the experts'), the engine's kernels against the library's", f8_product_check),
        ("the grouped experts' step products (no activation, SiLU, GELU, gathered, weighted), the engine's kernels against the library's", moe_check),
        ("the step product (route GEMM, 1 to 64 tokens), the engine's kernels against the library's, and its workspace", linear_check),
        ("the mma layouts' unpacking, the engine's kernels against the library's (every bf16 of each layout, whole and a row block)", unpack_check),
    ];
    for (what, run) in checks {
        println!("foldcheck: starting {what}");
        std::io::Write::flush(&mut std::io::stdout()).ok();
        let (n, bad) = run(&dev, &lib)?;
        say(what, n, bad);
    }
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
            let ra = unsafe { lib.library_moe_way(idb.ptr, p, e, a.ptr, dev.stream) };
            synced(dev, || format!("the library's moe_way, {e} experts, {p} pairs"))?;
            // SAFETY: as above.
            let rb = unsafe { moe_way(idb.ptr, p, e, b.ptr, dev.stream) };
            synced(dev, || format!("the engine's moe_way, {e} experts, {p} pairs"))?;
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
    // (1 to 64 tokens: the step product and the route MID; past 64 a prompt's: the prompt kernel by route BIG, DECODE or AHEAD, and the A100's MID to 128)
    let ms = [1i64, 2, 7, 8, 9, 15, 16, 17, 24, 32, 33, 48, 64, 65, 100, 128, 129, 200, 256, 300, 640, 1000];
    for (rows, cols) in [(256usize, 1024usize), (128, 4096)] {
        let x: Vec<u16> = (0..1000 * cols).map(|_| (((next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5).to_bits() >> 16) as u16).collect();
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
                // SAFETY: the pack, X [1000, cols], the workspace of the library's size (the engine's is the same or the product is the library's), zeroed
                // counters of each, Y [m, rows] each; one stream.
                let (ra, rb) = unsafe {
                    let ra = lib.library_linear(&pb.pack, xb.ptr, m, ya.ptr, ws.ptr, ws_bytes, da.ptr, dev.stream);
                    synced(dev, || format!("the library's step product, {name} {rows} x {cols}, {m} tokens"))?;
                    (ra, linear(&pb.pack, xb.ptr, m, yb.ptr, ws.ptr, ws_bytes, db.ptr, dev.stream))
                };
                synced(dev, || format!("the engine's step product, {name} {rows} x {cols}, {m} tokens"))?;
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
            lib.library_kv_write(
                &geom,
                pool.ptr,
                kb.ptr,
                vb.ptr,
                0,
                kbuf.ptr,
                stride,
                vbuf.ptr,
                stride,
                sl.ptr,
                tokens as i64,
                status.ptr,
                0,
                0,
                RING,
                dev.stream,
            )?
        };
        synced(dev, || format!("the library's KV write, {h} x {d}"))?;
        // the pages back by both kernels, unpacked (NHD keys and values) and gathered (the staging cache): every bf16 the same; then the same tokens written by the
        // engine's kernel into pools of their own, read back by the library's: every value the one written (the pools' bytes are not compared: the escape chunks a
        // page takes are found by atomics, in any order)
        let ids: Vec<i32> = (0..blocks as i32).collect();
        let (pages, dst) = (Buf::new(4 * blocks)?, Buf::new(4 * blocks)?);
        pages.write(0, &ids)?;
        dst.write(0, &ids)?;
        let words = blocks * 16 * h * d;
        let (uka, uva, ukb, uvb) = (Buf::zeroed(2 * words)?, Buf::zeroed(2 * words)?, Buf::zeroed(2 * words)?, Buf::zeroed(2 * words)?);
        let (gka, gkb) = (Buf::zeroed(4 * words)?, Buf::zeroed(4 * words)?);
        // SAFETY: the pool and its tables, the pages' and destinations' lists, outputs of `blocks` pages each; one stream.
        let r = unsafe {
            lib.library_kv_unpack(&geom, pool.ptr, kb.ptr, vb.ptr, pages.ptr, blocks as i64, uka.ptr, uva.ptr, dev.stream)?;
            synced(dev, || format!("the library's KV unpack, {h} x {d}"))?;
            lib.library_kv_gather(&geom, pool.ptr, kb.ptr, vb.ptr, pages.ptr, dst.ptr, blocks as i64, gka.ptr, dev.stream)?;
            synced(dev, || format!("the library's KV gather, {h} x {d}"))?;
            let u = kv_unpack(&geom, (pool.ptr, kb.ptr, vb.ptr), pages.ptr, blocks as i64, (ukb.ptr, uvb.ptr), dev.stream)?;
            synced(dev, || format!("the engine's KV unpack, {h} x {d}"))?;
            let gt = kv_gather(&geom, (pool.ptr, kb.ptr, vb.ptr), (pages.ptr, dst.ptr, blocks as i64), gkb.ptr, dev.stream)?;
            synced(dev, || format!("the engine's KV gather, {h} x {d}"))?;
            (u, gt)
        };
        let same = |a: &Buf, b: &Buf, len: usize| -> Result<usize> {
            let (mut x, mut y) = (vec![0u16; len], vec![0u16; len]);
            a.read(0, &mut x)?;
            b.read(0, &mut y)?;
            Ok(x.iter().zip(&y).filter(|(p, q)| p != q).count())
        };
        n += 2;
        if r != (true, true) {
            bad.push(format!("{h} x {d}: the engine's unpack or gather did not take the call ({r:?})"));
        } else {
            let (du, dg) = (same(&uka, &ukb, words)? + same(&uva, &uvb, words)?, same(&gka, &gkb, 2 * words)?);
            if du + dg > 0 {
                bad.push(format!("{h} x {d}: unpack {du}, gather {dg} bf16 differ"));
            }
        }
        let (pool2, escape2, raw2, status2) = (
            Buf::zeroed(page as usize * blocks)?,
            Buf::zeroed(kv_escape_bytes(nchunks)? as usize)?,
            Buf::zeroed(kv_raw_bytes(nslabs, hp)? as usize)?,
            Buf::zeroed(4 * (16 + 2 * RING) as usize)?,
        );
        let geom2 = KvGeom { escape: escape2.ptr, raw: raw2.ptr, ..geom };
        let (uk2, uv2) = (Buf::zeroed(2 * words)?, Buf::zeroed(2 * words)?);
        // SAFETY: the second pool and its tables, the tokens' rows and slots, outputs; one stream.
        let w2 = unsafe {
            let ok = kv_write(
                &geom2,
                (pool2.ptr, kb.ptr, vb.ptr, 0),
                (kbuf.ptr, stride, vbuf.ptr, stride, sl.ptr, tokens as i64),
                (status2.ptr, 0, 0, RING),
                dev.stream,
            )?;
            synced(dev, || format!("the engine's KV write, {h} x {d}"))?;
            lib.library_kv_unpack(&geom2, pool2.ptr, kb.ptr, vb.ptr, pages.ptr, blocks as i64, uk2.ptr, uv2.ptr, dev.stream)?;
            synced(dev, || format!("the library's KV unpack of the engine's write, {h} x {d}"))?;
            ok
        };
        n += 1;
        if !w2 {
            bad.push(format!("{h} x {d}: the engine's write did not take the call"));
        } else {
            let d2 = same(&uka, &uk2, words)? + same(&uva, &uv2, words)?;
            if d2 > 0 {
                bad.push(format!("{h} x {d}: the engine's write read back by the library: {d2} bf16 differ from the library's write"));
            }
        }
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
                let what =
                    format!("{h} x {d}, {g} queries a head, {b} sequences (first {first}, bf16 scores {round}){}", if bf16 { ", a bf16 cache" } else { "" });
                // SAFETY: the pool (or the cache), its tables, queries [b][h g d], outputs and zeroed workspaces of the plan's size, live; one stream.
                let (ra, rb) = unsafe {
                    if bf16 {
                        let cache = (kc.ptr, vc.ptr, bt.ptr, blocks as i64, 0);
                        let seqs = (sq.ptr, fp, b as i64, splits);
                        let heads = (h as i64, g as i64, d as i64);
                        let ra = lib.library_kv_decode_bf16(qb.ptr, w, cache, seqs, heads, (0.0883883, round), oa.ptr, w, (wa.ptr, ws_bytes), dev.stream);
                        synced(dev, || format!("the library's decode attention, {what}"))?;
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
                        synced(dev, || format!("the library's decode attention, {what}"))?;
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
                synced(dev, || format!("the engine's decode attention, {what}"))?;
                n += 1;
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
        // a prompt chunk's attention: rows from `start`, over the pages written (and over the bf16 cache), a window, bf16 scores, a soft cap
        let table: Vec<i32> = (0..blocks as i32).collect();
        let bt = Buf::new(4 * blocks)?;
        bt.write(0, &table)?;
        for (start, rows, window, round, softcap) in
            [(0i64, tokens as i64, 0i64, false, 0.0), (64, 40, 0, true, 0.0), (80, tokens as i64 - 80, 48, false, 30.0)]
        {
            let w = (h * g * d) as i64;
            let q: Vec<u16> = (0..rows as usize * h * g * d).map(|_| (((next() % 2000) as f32 - 1000.0) / 5000.0).to_bits().wrapping_shr(16) as u16).collect();
            let qb = Buf::new(2 * q.len())?;
            qb.write(0, &q)?;
            for bf16 in [false, true] {
                let (oa, ob) = (Buf::new(2 * q.len())?, Buf::new(2 * q.len())?);
                oa.fill_async(0xA5, dev.stream)?;
                ob.fill_async(0x5A, dev.stream)?;
                let (sc, scores) = ((0.0883883, round, softcap, 0u64), (0.0883883, round, softcap));
                let what = format!(
                    "{h} x {d}, {g} queries a head: a prompt's {rows} rows from {start} (window {window}, bf16 scores {round}, soft cap {softcap}){}",
                    if bf16 { ", a bf16 cache" } else { "" }
                );
                // SAFETY: the pages (or the cache), the table, queries and outputs [rows][h g d], live; one stream.
                let (ra, rb) = unsafe {
                    if bf16 {
                        let cache = (kc.ptr, vc.ptr, bt.ptr, 0);
                        let heads = (h as i64, g as i64, d as i64);
                        let ra = lib.library_kv_prefill_bf16(qb.ptr, w, cache, (rows, start, window), heads, scores, oa.ptr, w, dev.stream);
                        synced(dev, || format!("the library's prompt attention, {what}"))?;
                        (ra, kv_prefill_bf16((qb.ptr, w), cache, (rows, start, window), heads, scores, (ob.ptr, w), dev.stream))
                    } else {
                        let ra = lib.library_kv_prefill(
                            &geom, g as i64, pool.ptr, kb.ptr, vb.ptr, bt.ptr, qb.ptr, w, None, rows, start, window, 0.0883883, round, softcap, 0, oa.ptr, w,
                            dev.stream,
                        );
                        synced(dev, || format!("the library's prompt attention, {what}"))?;
                        (
                            ra,
                            kv_prefill(
                                &geom,
                                g as i64,
                                (pool.ptr, kb.ptr, vb.ptr, bt.ptr),
                                (qb.ptr, w),
                                None,
                                (rows, start, window),
                                sc,
                                (ob.ptr, w),
                                dev.stream,
                            ),
                        )
                    }
                };
                synced(dev, || format!("the engine's prompt attention, {what}"))?;
                n += 1;
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
                let (mut va, mut vb2) = (vec![0u16; q.len()], vec![0u16; q.len()]);
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

/// `foldcheck`'s grouped experts: 8 experts of 128 x 1024 in each layout the products take, a plan of random choices (2 a token) for 1 to 64 tokens, the
/// products with no activation, SiLU and GELU, gathered and not, and weighted sums: by the library's kernels and by the engine's where the plan takes the
/// step products (the prompt kernel and the wgmma route are the library's still), every bf16 compared; and the expert cache's places (moe_placed), the
/// tiered stack cut per expert as the store cuts it. (compared, differences)
#[cfg(feature = "dev")]
fn moe_check(dev: &crate::dev::Dev, lib: &crate::ffi::Lib) -> Result<(usize, Vec<String>)> {
    use crate::dev::Buf;
    use crate::weights::upload_packed;
    let mut seed = 0xA076_1D64_78BD_642Fu64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let (e, o, kk) = (8i64, 128i64, 1024usize);
    let (mut n, mut bad, mut folded) = (0, Vec::new(), 0);
    let x: Vec<u16> = (0..300 * 2 * kk).map(|_| (((next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5).to_bits() >> 16) as u16).collect();
    let xb = Buf::new(2 * x.len())?;
    xb.write(0, &x)?;
    for (name, p) in test_packs((e * o) as usize, kk, &mut next) {
        let Some(p) = p else { continue };
        if matches!(name, "compact" | "int3") {
            continue;
        }
        let pb = upload_packed(p)?;
        // (from 48 pairs an expert on average, 64 on an A100, 96 on an H100: the experts' prompt kernel)
        for t in [1i64, 4, 16, 17, 32, 33, 64, 200, 300] {
            let k = 2i64;
            let ids: Vec<i64> = (0..t * k).map(|_| (next() % e as u64) as i64).collect();
            let wts: Vec<f32> = (0..t * k).map(|_| (next() % 1000) as f32 / 1000.0).collect();
            let (idb, wb, plan) = (Buf::new(8 * ids.len())?, Buf::new(4 * wts.len())?, Buf::new(4 * (2 + 2 * e + t * k) as usize)?);
            idb.write(0, &ids)?;
            wb.write(0, &wts)?;
            // SAFETY: the ids and the plan's room; one stream.
            unsafe { lib.library_moe_way(idb.ptr, t * k, e, plan.ptr, dev.stream)? };
            for (act, gather, weighted) in [(0i64, true, false), (0, false, false), (0, true, true), (1, true, false), (2, false, false)] {
                let twelve = pb.pack.is_twelve();
                let ws_bytes = lib.library_moe_workspace(twelve, e, o, kk as i64, t, k, act, weighted)?.max(16);
                let what = format!("{name}, {t} tokens, act {act}, gather {gather}, weighted {weighted}");
                let rows_out = if weighted { t } else { t * k };
                let width = if act != 0 { o / 2 } else { o };
                let len = (rows_out * width) as usize;
                let units = (if act != 0 { o / 128 } else { o / 64 }) * e.min(t * k);
                let (ws, da, db, ya, yb) = (
                    Buf::zeroed(ws_bytes)?,
                    Buf::zeroed(4 * units as usize + 64)?,
                    Buf::zeroed(4 * units as usize + 64)?,
                    Buf::new(2 * len)?,
                    Buf::new(2 * len)?,
                );
                ya.fill_async(0xA5, dev.stream)?;
                yb.fill_async(0x5A, dev.stream)?;
                let (wp, ip) = if weighted { (wb.ptr, idb.ptr) } else { (0, 0) };
                // SAFETY: the stack, X [300 2, K], the plan, the weights and ids, a workspace of the library's size, zeroed counters each, Y each; one stream.
                let (ra, rb) = unsafe {
                    let ra = lib.library_moe(&pb.pack, e, xb.ptr, t, k, gather, plan.ptr, act, wp, ip, ya.ptr, ws.ptr, ws_bytes, da.ptr, dev.stream);
                    synced(dev, || format!("the library's experts, {what}"))?;
                    (ra, moe(&pb.pack, None, e, (xb.ptr, t, k, gather), (plan.ptr, act, wp, ip), yb.ptr, (ws.ptr, ws_bytes, db.ptr), dev.stream))
                };
                synced(dev, || format!("the engine's experts, {what}"))?;
                n += 1;
                match (&ra, &rb) {
                    (Err(x), _) | (_, Err(x)) => {
                        bad.push(format!("{what}: {x}"));
                        continue;
                    }
                    (_, Ok(false)) => continue,
                    _ => folded += 1,
                }
                let (mut va, mut vb) = (vec![0u16; len], vec![0u16; len]);
                ya.read(0, &mut va)?;
                yb.read(0, &mut vb)?;
                let differ = va.iter().zip(&vb).filter(|(p, q)| p != q).count();
                if differ > 0 && bad.len() < 12 {
                    bad.push(format!("{what}: {differ} of {len} bf16 differ"));
                }
            }
        }
    }
    // the expert cache's places: the tiered stack cut per expert (tiering::TieredPlaces: each place its blocks and entries), glyd_gpu_mma_moe_placed
    let w: Vec<u16> =
        (0..(e * o) as usize * kk).map(|_| (((next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.1).to_bits().wrapping_shr(16) as u16).collect();
    let places = crate::tiering::TieredPlaces::of(&glyd_gpu::pack::pack_tiered(&w, (e * o) as usize, kk), e as usize)?;
    let pb = upload_packed(crate::weights::Packed {
        data: places.data.clone(),
        b1: places.blocks.clone(),
        b2: places.entries.clone(),
        words: Words::Tiered { blocks: 0, block_base: 0, tiers: places.tiers },
        rows: (e * o) as usize,
        cols: kk,
    })?;
    for t in [1i64, 4, 17, 33, 64, 300] {
        let k = 2i64;
        let ids: Vec<i64> = (0..t * k).map(|_| (next() % e as u64) as i64).collect();
        let (idb, plan) = (Buf::new(8 * ids.len())?, Buf::new(4 * (2 + 2 * e + t * k) as usize)?);
        idb.write(0, &ids)?;
        // SAFETY: the ids and the plan's room; one stream.
        unsafe { lib.library_moe_way(idb.ptr, t * k, e, plan.ptr, dev.stream)? };
        for act in [0i64, 1] {
            let ws_bytes = lib.library_moe_workspace(false, e, o, kk as i64, t, k, act, false)?.max(16);
            let len = (t * k * if act != 0 { o / 2 } else { o }) as usize;
            let units = (if act != 0 { o / 128 } else { o / 64 }) * e.min(t * k);
            let (ws, da, db, ya, yb) =
                (Buf::zeroed(ws_bytes)?, Buf::zeroed(4 * units as usize + 64)?, Buf::zeroed(4 * units as usize + 64)?, Buf::new(2 * len)?, Buf::new(2 * len)?);
            ya.fill_async(0xA5, dev.stream)?;
            yb.fill_async(0x5A, dev.stream)?;
            let stride = places.stride as u64;
            // SAFETY: the places, X, the plan, a workspace of the library's size, zeroed counters each, Y each; one stream.
            let (ra, rb) = unsafe {
                let ra = lib.library_moe_placed(&pb.pack, stride, e, xb.ptr, t, k, true, plan.ptr, act, 0, 0, ya.ptr, ws.ptr, ws_bytes, da.ptr, dev.stream);
                synced(dev, || format!("the library's expert cache's places, {t} tokens, act {act}"))?;
                (ra, moe(&pb.pack, Some(stride as i64), e, (xb.ptr, t, k, true), (plan.ptr, act, 0, 0), yb.ptr, (ws.ptr, ws_bytes, db.ptr), dev.stream))
            };
            synced(dev, || format!("the engine's expert cache's places, {t} tokens, act {act}"))?;
            n += 1;
            let what = format!("the expert cache's places, {t} tokens, act {act}");
            match (&ra, &rb) {
                (Err(x), _) | (_, Err(x)) => {
                    bad.push(format!("{what}: {x}"));
                    continue;
                }
                (_, Ok(false)) => continue,
                _ => folded += 1,
            }
            let (mut va, mut vb) = (vec![0u16; len], vec![0u16; len]);
            ya.read(0, &mut va)?;
            yb.read(0, &mut vb)?;
            let differ = va.iter().zip(&vb).filter(|(p, q)| p != q).count();
            if differ > 0 {
                bad.push(format!("{what}: {differ} of {len} bf16 differ"));
            }
        }
    }
    println!("  (the experts: {folded} products by the engine's kernels, the rest the library's on this GPU)");
    Ok((n, bad))
}

/// `foldcheck`'s lossless FP8 format, both layouts: random e4m3 bytes (every byte a value) of three shapes packed by the library's kernels and by the engine's (the
/// size's words, the heads, the cores and var compared byte for byte), and unpacked by both (the bytes back, and the same). (compared, differences)
#[cfg(feature = "dev")]
fn f8_check(dev: &crate::dev::Dev, lib: &crate::ffi::Lib) -> Result<(usize, Vec<String>)> {
    use crate::dev::Buf;
    let mut seed = 0xBF58_476D_1CE4_E5B9u64;
    let (mut n, mut bad) = (0, Vec::new());
    for (o, k) in [(64i64, 128i64), (200, 300), (512, 1024)] {
        let bytes: Vec<u8> = (0..o * k)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                // (mostly small exponents, as weights have, and every byte now and then)
                if seed % 16 == 0 {
                    (seed >> 8) as u8
                } else {
                    (((seed >> 8) as u8) & 0x87) | 0x30
                }
            })
            .collect();
        let wb = Buf::new(bytes.len())?;
        wb.write(0, &bytes)?;
        let patches = f8_patches(o, k).unwrap_or(0);
        n += 1;
        if lib.library_f8_size_workspace(o, k)? != f8_size_workspace(o, k)? {
            bad.push(format!("{o} x {k}: the size's workspace differs"));
        }
        for h in [false, true] {
            let name = format!("{o} x {k}{}", if h { ", layout H" } else { "" });
            let (ha, hb, ws) = (Buf::zeroed((patches * 32) as usize)?, Buf::zeroed((patches * 32) as usize)?, Buf::zeroed(16)?);
            // SAFETY: W [o, k] bytes, heads of 8 words a patch, a workspace of 16 bytes; the stream is waited for by the size.
            let (va, vb) = unsafe {
                (lib.library_f8_size(h, wb.ptr, k, o, k, ha.ptr, ws.ptr, 16, dev.stream)?, f8_size(h, wb.ptr, k, o, k, hb.ptr, (ws.ptr, 16), dev.stream)?)
            };
            n += 1;
            let Some(vb) = vb else {
                bad.push(format!("{name}: the engine's size did not take it"));
                continue;
            };
            if va != vb {
                bad.push(format!("{name}: var words {va} (library) against {vb}"));
                continue;
            }
            let core_bytes = (patches * if h { 7168 } else { 6144 }) as usize;
            let (ca, cb, xa, xb) = (Buf::zeroed(core_bytes)?, Buf::zeroed(core_bytes)?, Buf::zeroed(va as usize * 4)?, Buf::zeroed(va as usize * 4)?);
            let (oa, ob) = (Buf::zeroed(bytes.len())?, Buf::zeroed(bytes.len())?);
            // SAFETY: the heads, cores of 6,144 (layout H 7,168) bytes a patch, var of the size's words, outputs [o, k]; one stream.
            let r = unsafe {
                lib.library_f8_pack(h, wb.ptr, k, o, k, ha.ptr, ca.ptr, xa.ptr, va, dev.stream)?;
                synced(dev, || format!("the library's FP8 pack, {name}"))?;
                let p = f8_pack(h, wb.ptr, k, o, k, hb.ptr, (cb.ptr, xb.ptr, vb), dev.stream)?;
                synced(dev, || format!("the engine's FP8 pack, {name}"))?;
                lib.library_f8_unpack(h, ca.ptr, ha.ptr, xa.ptr, o, k, oa.ptr, k, dev.stream)?;
                synced(dev, || format!("the library's FP8 unpack, {name}"))?;
                let u = f8_unpack(h, (cb.ptr, hb.ptr, xb.ptr), o, k, (ob.ptr, k), dev.stream)?;
                synced(dev, || format!("the engine's FP8 unpack, {name}"))?;
                (p, u)
            };
            n += 1;
            if r != (true, true) {
                bad.push(format!("{name}: the engine's pack or unpack did not take it ({r:?})"));
                continue;
            }
            let read = |b: &Buf, len: usize| -> Result<Vec<u8>> {
                let mut v = vec![0u8; len];
                b.read(0, &mut v)?;
                Ok(v)
            };
            let same = read(&ha, (patches * 32) as usize)? == read(&hb, (patches * 32) as usize)?
                && read(&ca, core_bytes)? == read(&cb, core_bytes)?
                && read(&xa, va as usize * 4)? == read(&xb, va as usize * 4)?;
            let (ua, ub) = (read(&oa, bytes.len())?, read(&ob, bytes.len())?);
            if !same || ua != ub || ua != bytes {
                bad.push(format!("{name}: the packs the same {same}, the unpacks the same {}, the library's unpack the bytes {}", ua == ub, ua == bytes));
            }
        }
    }
    Ok((n, bad))
}

/// `foldcheck`'s FP8 products: random e4m3 weights packed by the library (both layouts), each kind of scales (layout H: block), 1 to 64 tokens on each side
/// of the tiles' sizes (layout H's one token its own kernel), and the experts' product over a plan of 2 choices a token, gathered and not: by the library's
/// kernels and by the engine's, the workspaces' sizes and every bf16 compared. (compared, differences)
#[cfg(feature = "dev")]
fn f8_product_check(dev: &crate::dev::Dev, lib: &crate::ffi::Lib) -> Result<(usize, Vec<String>)> {
    use crate::dev::Buf;
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let (mut n, mut bad) = (0, Vec::new());
    let bf = |v: f32| (v.to_bits() >> 16) as u16;
    // W [o, k] packed by the library in layout `h`: (core, heads, var)
    let pack = |o: i64, k: i64, h: bool| -> Result<(Buf, Buf, Buf)> {
        let mut r = 0xD1B5_4A32_D192_ED03u64 ^ (o * 65_537 + k) as u64;
        let w: Vec<u8> = (0..o * k)
            .map(|_| {
                r ^= r << 13;
                r ^= r >> 7;
                r ^= r << 17;
                (((r >> 8) as u8) & 0x87) | 0x30
            })
            .collect();
        let wb = Buf::new(w.len())?;
        wb.write(0, &w)?;
        let patches = f8_patches(o, k).unwrap_or(0) as usize;
        let (heads, ws) = (Buf::zeroed(patches * 32)?, Buf::zeroed(16)?);
        // SAFETY: W [o, k], heads of 8 words a patch, the size's workspace, the core and var of its sizes; one stream (the size waits for it).
        unsafe {
            let words = lib.library_f8_size(h, wb.ptr, k, o, k, heads.ptr, ws.ptr, 16, dev.stream)?;
            let (core, var) = (Buf::zeroed(patches * if h { 7168 } else { 6144 })?, Buf::zeroed(words as usize * 4)?);
            lib.library_f8_pack(h, wb.ptr, k, o, k, heads.ptr, core.ptr, var.ptr, words, dev.stream)?;
            synced(dev, || format!("the library's FP8 pack of {o} x {k}"))?;
            Ok((core, heads, var))
        }
    };
    let xs: Vec<u16> = (0..64 * 4096).map(|i| bf(((i * 7919 % 2001) as f32 - 1000.0) / 4000.0)).collect();
    let xb = Buf::new(2 * xs.len())?;
    xb.write(0, &xs)?;
    for (o, k) in [(128i64, 256i64), (320, 1000), (512, 4096)] {
        for h in [false, true] {
            let (core, heads, var) = pack(o, k, h)?;
            let (k128, k32) = ((k + 127) / 128, (k + 31) / 32);
            for kind in if h { 0..1 } else { 0..5 } {
                // the scales: fp32 a 128 x 128 block, one fp32, e8m0 a 1 x 32 and a 32 x 32 block, fp32 a row
                let (rows, lds, f32s) = match kind {
                    0 => ((o + 127) / 128, k128, true),
                    1 => (1, 1, true),
                    2 => (o, k32, false),
                    3 => ((o + 31) / 32, k32, false),
                    _ => (o, 1, true),
                };
                let sb = Buf::new((rows * lds * 4) as usize)?;
                if f32s {
                    sb.write(0, &(0..rows * lds).map(|i| 0.5 + (i % 5) as f32 / 8.0).collect::<Vec<f32>>())?;
                } else {
                    sb.write(0, &(0..rows * lds).map(|i| 124 + (i % 6) as u8).collect::<Vec<u8>>())?;
                }
                for m in [1i64, 2, 8, 9, 16, 17, 33, 64] {
                    let what = format!("{o} x {k}{}, kind {kind}, {m} tokens", if h { ", layout H" } else { "" });
                    let ws_bytes = lib.library_f8_linear_workspace(o, k, m)?;
                    n += 1;
                    match f8_linear_workspace(o, k, m)? {
                        Some(b) if b != ws_bytes => bad.push(format!("{what}: workspace {ws_bytes} (library) against {b}")),
                        _ => {}
                    }
                    let ws = Buf::zeroed(ws_bytes.max(16))?;
                    let (da, db) = (Buf::zeroed((4 * (o / 128 + 1) + 64) as usize)?, Buf::zeroed((4 * (o / 128 + 1) + 64) as usize)?);
                    let (ya, yb) = (Buf::new((2 * m * o) as usize)?, Buf::new((2 * m * o) as usize)?);
                    ya.fill_async(0xA5, dev.stream)?;
                    yb.fill_async(0x5A, dev.stream)?;
                    let w = (core.ptr, heads.ptr, var.ptr);
                    // SAFETY: the pack, its scales, X [64, 4096] (rows k apart: within it), Y [m, o] each, the library's workspace, zeroed counters each; one stream.
                    let rb = unsafe {
                        lib.library_f8_linear(h, w.0, w.1, w.2, o, k, kind, sb.ptr, lds, xb.ptr, m, ya.ptr, ws.ptr, ws_bytes, da.ptr, dev.stream)?;
                        synced(dev, || format!("the library's FP8 product, {what}"))?;
                        let r = f8_linear(h, w, (o, k, kind, sb.ptr, lds), (xb.ptr, m, yb.ptr), (ws.ptr, ws_bytes, db.ptr), dev.stream)?;
                        synced(dev, || format!("the engine's FP8 product, {what}"))?;
                        r
                    };
                    n += 1;
                    if !rb {
                        bad.push(format!("{what}: the engine's kernel did not take it"));
                        continue;
                    }
                    let (mut va, mut vb) = (vec![0u16; (m * o) as usize], vec![0u16; (m * o) as usize]);
                    ya.read(0, &mut va)?;
                    yb.read(0, &mut vb)?;
                    let differ = va.iter().zip(&vb).filter(|(p, q)| p != q).count();
                    if differ > 0 {
                        bad.push(format!("{what}: {differ} of {} bf16 differ", va.len()));
                    }
                }
            }
        }
    }
    // the experts' product: 4 experts of [128, 256] as one [512, 256], block scales [4][1][2], 2 choices a token
    let (e, o, k) = (4i64, 128i64, 256i64);
    let sb = Buf::new((e * 2 * 4) as usize)?;
    sb.write(0, &(0..e * 2).map(|i| 0.25 + i as f32 / 16.0).collect::<Vec<f32>>())?;
    for h in [false, true] {
        let (core, heads, var) = pack(e * o, k, h)?;
        for t in [1i64, 3, 17, 40] {
            let ids: Vec<i64> = (0..t)
                .flat_map(|_| {
                    let a = (next() % e as u64) as i64;
                    [a, (a + 1 + (next() % (e as u64 - 1)) as i64) % e]
                })
                .collect();
            let most = (0..e).map(|x| ids.iter().filter(|&&i| i == x).count() as i64).max().unwrap_or(1).max(1);
            let (idb, plan) = (Buf::new(8 * ids.len())?, Buf::new(4 * (2 + 2 * e + 2 * t) as usize)?);
            idb.write(0, &ids)?;
            // SAFETY: the ids and the plan's room; one stream.
            unsafe { lib.library_moe_way(idb.ptr, 2 * t, e, plan.ptr, dev.stream)? };
            for gather in [true, false] {
                let what = format!("the experts' product{}, {t} tokens, gather {gather}", if h { ", layout H" } else { "" });
                let ws_bytes = lib.library_f8_moe_workspace(o, k, most, e)?;
                n += 1;
                match f8_moe_workspace(o, k, most, e)? {
                    Some(b) if b != ws_bytes => bad.push(format!("{what}: workspace {ws_bytes} (library) against {b}")),
                    _ => {}
                }
                let ws = Buf::zeroed(ws_bytes.max(16))?;
                let (da, db) = (Buf::zeroed((4 * e + 64) as usize)?, Buf::zeroed((4 * e + 64) as usize)?);
                let len = (2 * t * o) as usize;
                let (ya, yb) = (Buf::new(2 * len)?, Buf::new(2 * len)?);
                ya.fill_async(0xA5, dev.stream)?;
                yb.fill_async(0x5A, dev.stream)?;
                let f8 = (core.ptr, heads.ptr, var.ptr, e, o, k, sb.ptr);
                // SAFETY: the stack and its scales, X [64, 4096] (2t rows k apart: within it), the plan, Y [2t, o] each, the library's workspace, zeroed
                // counters each; one stream.
                let rb = unsafe {
                    lib.library_f8_moe(
                        h, f8.0, f8.1, f8.2, e, o, k, sb.ptr, xb.ptr, plan.ptr, 2, gather, most, e, ya.ptr, ws.ptr, ws_bytes, da.ptr, dev.stream,
                    )?;
                    synced(dev, || format!("the library's {what}"))?;
                    let r = f8_moe(h, f8, (xb.ptr, plan.ptr, 2, gather), (most, e, yb.ptr), (ws.ptr, ws_bytes, db.ptr), dev.stream)?;
                    synced(dev, || format!("the engine's {what}"))?;
                    r
                };
                n += 1;
                if !rb {
                    bad.push(format!("{what}: the engine's kernel did not take it"));
                    continue;
                }
                let (mut va, mut vb) = (vec![0u16; len], vec![0u16; len]);
                ya.read(0, &mut va)?;
                yb.read(0, &mut vb)?;
                let differ = va.iter().zip(&vb).filter(|(p, q)| p != q).count();
                if differ > 0 {
                    bad.push(format!("{what}: {differ} of {len} bf16 differ"));
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
                let ra = unsafe { lib.library_unpack(&pb.pack, row0, nr, a.ptr, dev.stream) };
                synced(dev, || format!("the library's unpack, {name} {rows} x {cols} rows {row0}+{nr}"))?;
                // SAFETY: as above.
                let rb = unsafe { unpacker(&pb.pack).and_then(|u| unpack(u, &pb.pack, row0, nr, b.ptr, dev.stream)) };
                synced(dev, || format!("the engine's unpack, {name} {rows} x {cols} rows {row0}+{nr}"))?;
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
        // an odd count of partials: their sums padded to 16 bytes, so the partial outputs after them are float4-aligned
        assert_eq!(kv_decode_workspace(1, 1, 1, 128, 1).unwrap(), (1 << 20) + 4 * 4 + 128 * 4);
        assert_eq!(kv_decode_workspace(1, 3, 1, 128, 3).unwrap(), (1 << 20) + 20 * 4 + 9 * 128 * 4);
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
        assert_eq!(size_of::<KvPrefillParams>(), 232);
        let pf = [
            offset_of!(KvPrefillParams, nc),
            offset_of!(KvPrefillParams, raw),
            offset_of!(KvPrefillParams, sinks),
            offset_of!(KvPrefillParams, cap),
            offset_of!(KvPrefillParams, capl2),
        ];
        assert_eq!(pf, [96, 104, 144, 184, 228]);
        assert_eq!((size_of::<KvEscapePool>(), size_of::<KvWriteParams>()), (32, 200));
        let wp = [
            offset_of!(KvWriteParams, esc),
            offset_of!(KvWriteParams, raw),
            offset_of!(KvWriteParams, lost),
            offset_of!(KvWriteParams, ring),
            offset_of!(KvWriteParams, one),
        ];
        assert_eq!(wp, [80, 112, 152, 168, 192]);
        assert_eq!((size_of::<F8Pk>(), size_of::<F8Un>()), (40, 80));
        assert_eq!((offset_of!(F8Pk, plain), offset_of!(F8Un, o), offset_of!(F8Un, vec)), (32, 56, 76));
        // f8c::F8Lin's offsets as a host compiler lays it out (gpu/fp8/emu_f8.cpp's build of the f8c block, 2026-11-23): 168 bytes
        assert_eq!(size_of::<F8Lin>(), 168);
        assert_eq!((size_of::<TensorMap>(), std::mem::align_of::<TensorMap>()), (128, 64));
        assert_eq!((size_of::<MoePairs>(), offset_of!(MoePairs, k), offset_of!(MoePairs, w), offset_of!(MoePairs, skip)), (56, 16, 32, 52));
        let at =
            [offset_of!(F8Lin, lds), offset_of!(F8Lin, m), offset_of!(F8Lin, xvec), offset_of!(F8Lin, plan), offset_of!(F8Lin, e), offset_of!(F8Lin, gather)];
        assert_eq!(at, [80, 104, 128, 136, 152, 160]);
        // the products' shared memory as f8_lin_shared and f8h_one_shared give it (f8_lsmem<2> 29,888, layout H's 33,984; L1W 1,184, BW 1,152 words a warp)
        assert_eq!((F8_LSMEM, F8H_LSMEM, F8H_ONE_CODED, F8H_ONE_PLAIN), (29_888, 33_984, 37_888, 36_864));
        assert_eq!((f8_shared(1, false), f8_shared(64, false), f8_shared(64, true)), (30_400, 62_656, 66_752));
        assert_eq!([1, 8, 9, 16, 17, 32, 33, 64].map(f8_ntl), [1, 1, 2, 2, 4, 4, 8, 8]);
        assert_eq!((f8_work(4096, 4096), f8_work(64, 128), f8_work(65, 129)), (32 * 32, 1, 2));
    }

    #[test]
    fn the_fp8_products_names_are_the_kernels() {
        use crate::dev::Typed;
        use crate::kt;
        assert_eq!(f8_name(0, 1), kt::fold_f8_block_1::NAME);
        assert_eq!(f8_name(4, 8), kt::fold_f8_channel_8::NAME);
        assert_eq!(f8_name(5, 4), kt::fold_f8_moe_4::NAME);
        assert_eq!(f8_name(7, 2), kt::fold_f8_hmoe_2::NAME);
        assert_eq!(f8h_one_name(true, true), kt::fold_f8h_one_plain_1::NAME);
        assert_eq!(f8h_one_name(false, false), kt::fold_f8h_one_coded_0::NAME);
        assert_eq!(std::any::type_name::<<kt::fold_f8_h_8 as Typed>::Args>(), std::any::type_name::<(F8Lin,)>());
    }

    /// The arguments the experts' launches build by name are, in order and type, the kernels' own (crate::kt, read from kernels/fold.cu).
    #[test]
    fn the_experts_arguments_are_the_kernels() {
        use crate::dev::Typed;
        use crate::kt;
        fn args_of<T: Typed>() -> String {
            std::any::type_name::<T::Args>().replace(' ', "")
        }
        let pack = |words| Pack { data: 0, words, rows: 128, cols: 64 };
        let tiers = Words::Tiered { blocks: 0, block_base: 0, tiers: [0; 3] };
        let common = |s: Slots| s.i64(0).i64(0).p(0).i32(0).i64(0).i32(0).p(0).p(0).p(0).i32(0).p(0).p(0);
        let full = |s: Slots| common(s).i64(0).p(0).p(0).i32(0);
        let built = |s: Slots| format!("({})", s.1.trim_end_matches(", ")).replace(' ', "");
        let lead = |u, w, placed| layout_args(u, &pack(w), placed).unwrap();
        assert_eq!(built(full(lead(Unpacker::Tiered, tiers, None))), args_of::<kt::fold_moe_tiered_1_1_0>());
        assert_eq!(built(full(lead(Unpacker::Tiered, tiers, Some((1, 2))))), args_of::<kt::fold_moe_placed_4_0_2>());
        assert_eq!(built(full(lead(Unpacker::Int6, tiers, None))), args_of::<kt::fold_moe_int6_2_2_0>());
        assert_eq!(built(full(lead(Unpacker::Int4H(110), tiers, None))), args_of::<kt::fold_moe_int4h_1_0_10>());
        assert_eq!(built(full(lead(Unpacker::Fp4R { ks: 4, mg: 1, sh: 1 }, tiers, None))), args_of::<kt::fold_moe_fp4r_1_1_8>());
        let twelve = Words::Twelve { exc: 0, exc_base: 0, sym: [0; 4] };
        assert_eq!(built(full(lead(Unpacker::Twelve { hb4: 0, xmax: 4 }, twelve, None))), args_of::<kt::fold_moe_twelve_4_2_2>());
        assert_eq!(built(common(lead(Unpacker::Fp4, tiers, None)).i32(0)), args_of::<kt::fold_moe_rows_fp4_2_0>());
        assert_eq!(built(common(lead(Unpacker::Twelve { hb4: 0, xmax: 4 }, twelve, None)).i32(0)), args_of::<kt::fold_moe_rows_twelve_4_2>());
    }

    #[test]
    fn atoll_reads_as_c_does() {
        assert_eq!([atoll("12"), atoll(" 12x"), atoll("-3"), atoll("+4"), atoll("x"), atoll("")], [12, 12, -3, 4, 0, 0]);
        assert_eq!([whole(" 17 "), whole("17x"), whole("-1"), whole("")], [Some(17), None, Some(-1), None]);
    }
}
