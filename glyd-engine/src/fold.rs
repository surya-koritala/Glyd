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
//! The kernels these plan for are still the library's; their launches come with the kernels (step 1).

use crate::device::{Attr, CudaDevice as D, Device};
use crate::ffi::{Error, Kind, Result};

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

/// `foldcheck` (R17): each folded piece against the library's on this GPU, the library the oracle: the GPU's code; the way of every GPU code the
/// library knows (with and without SPLIT), both layouts, matrices of every kind of K and size, at token counts on each side of every threshold;
/// the KV cache's sizes and decode workspace over valid and refused shapes. Every answer the same, refusals included.
#[cfg(feature = "dev")]
pub fn check() -> Result<()> {
    use crate::gpu_ops::GpuOps;
    let _dev = crate::dev::Dev::new(0)?;
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
    if fails > 0 {
        return crate::ffi::err(format!("foldcheck: {fails} failed"));
    }
    println!("foldcheck: PASS");
    Ok(())
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

    #[test]
    fn atoll_reads_as_c_does() {
        assert_eq!([atoll("12"), atoll(" 12x"), atoll("-3"), atoll("+4"), atoll("x"), atoll("")], [12, 12, -3, 4, 0, 0]);
        assert_eq!([whole(" 17 "), whole("17x"), whole("-1"), whole("")], [Some(17), None, Some(-1), None]);
    }
}
