//! Glyd GPU from Rust: a bf16 model's matrices held compressed in GPU memory
//! and decoded on the GPU bit for bit, whole or inside their products.
//!
//! [`Library`] is the CUDA library every release ships
//! (`libglyd_gpu_cudaN.so`, gpu/build_lib.sh) behind its C API
//! (gpu/glyd_gpu.h), loaded at run time and refused where its API version is
//! not this crate's: each function typed, its workspace query first, its
//! status a [`Result`]. [`cuda`] holds the few driver calls a caller without
//! a CUDA runtime of its own needs (a device's context, memory, streams).
//! Nothing is linked at build time: the crate builds without CUDA, and its
//! calls fail with [`Error::Load`] where there is no GPU.
//!
//! # Safety
//!
//! A product or decode takes device pointers and a stream from the caller,
//! as the C API does: every such function is `unsafe`, and its caller
//! promises what glyd_gpu.h asks of its arguments. Each pointer is device
//! memory of the current context's device holding at least the elements the
//! header gives for the sizes passed (a pack's arrays as its packer made
//! them), live and not written by other work until the stream has run the
//! call; bf16 is its bits, `u16`.
//!
//! License: BUSL-1.1 (LICENSE; the codec, the `glyd` crate, is BSD-3-Clause
//! OR GPL-2.0).

use std::ffi::{c_char, c_int, c_void, CStr};
use std::fmt;

pub mod cuda;
pub mod json;
pub mod pack;
pub mod safetensors;
pub mod save;

/// The C API this crate calls (glyd_gpu.h's `GLYD_GPU_API_VERSION`): a
/// library of another version is refused, as a C FFI does not see a call's
/// arguments.
pub const API_VERSION: i32 = 5;
/// A GPU's class by name in its code ([`Library::gpu`]): "GeForce" in its name.
pub const GEFORCE: i32 = 1000;
/// A GPU's class by name in its code: "A10" in its name as a word (an A10, not an A10G, A100 or A40).
pub const A10: i32 = 2000;
/// A GPU's class by name in its code: "L4" in its name as a word (an L4, not an L40S or L40).
pub const L4: i32 = 3000;
/// A GPU's class by name in its code: "L40S" in its name as a word (not an L40).
pub const L40S: i32 = 4000;
/// A status: `cudaErrorInvalidValue`, an argument out of range.
pub const INVALID_VALUE: i32 = 1;
/// A status: `cudaErrorNotSupported`, a kernel that is not for this GPU.
pub const NOT_SUPPORTED: i32 = 801;

/// What failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A call's status (the library's `cudaError_t`, the driver's `CUresult`) and its text.
    Cuda { call: &'static str, status: i32, text: String },
    /// The library or the driver not found, not loaded, or of another version.
    Load(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Cuda { call, status, text } => write!(f, "{call}: {text} ({status})"),
            Error::Load(why) => f.write_str(why),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// A CUDA stream (the runtime's `cudaStream_t`, the driver's `CUstream`: the
/// same handle); [`Stream::DEFAULT`] the device's default stream.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stream(pub *mut c_void);

impl Stream {
    pub const DEFAULT: Stream = Stream(std::ptr::null_mut());
}

/// dlopen, dlsym and dlclose, declared here (no crate): the library and the
/// driver are found at run time. Elsewhere than Unix none is found.
pub(crate) mod dl {
    use std::ffi::c_void;

    #[cfg(unix)]
    mod sys {
        use std::ffi::{c_char, c_int, c_void};

        #[cfg(target_os = "macos")]
        pub const FLAGS: c_int = 0x2 | 0x4; // RTLD_NOW | RTLD_LOCAL
        #[cfg(not(target_os = "macos"))]
        pub const FLAGS: c_int = 0x2; // RTLD_NOW (RTLD_LOCAL is 0)

        extern "C" {
            pub fn dlopen(file: *const c_char, flags: c_int) -> *mut c_void;
            pub fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
            pub fn dlerror() -> *const c_char;
            pub fn dlclose(handle: *mut c_void) -> c_int;
        }
    }

    /// The shared library at `path` (a bare name: the loader's search); else
    /// the loader's reason.
    #[cfg(unix)]
    pub fn open(path: &str) -> Result<*mut c_void, String> {
        use std::ffi::{CStr, CString};
        let c = CString::new(path).map_err(|_| format!("{path}: a NUL in the path"))?;
        // SAFETY: a NUL-terminated path; dlerror's text is copied before any other dl call.
        unsafe {
            let h = sys::dlopen(c.as_ptr(), sys::FLAGS);
            if h.is_null() {
                let e = sys::dlerror();
                return Err(if e.is_null() { format!("{path}: not loaded") } else { CStr::from_ptr(e).to_string_lossy().into_owned() });
            }
            Ok(h)
        }
    }

    #[cfg(not(unix))]
    pub fn open(path: &str) -> Result<*mut c_void, String> {
        Err(format!("{path}: Glyd's GPU library runs on Linux"))
    }

    /// Symbol `name` (NUL-terminated) of `handle`, or null.
    ///
    /// # Safety
    /// `handle` is one `open` gave.
    #[cfg(unix)]
    pub unsafe fn sym(handle: *mut c_void, name: &str) -> *mut c_void {
        sys::dlsym(handle, name.as_ptr() as *const std::ffi::c_char)
    }

    #[cfg(not(unix))]
    pub unsafe fn sym(_: *mut c_void, _: &str) -> *mut c_void {
        std::ptr::null_mut()
    }

    /// `handle` closed: a library refused, none of its functions called but its version's.
    ///
    /// # Safety
    /// `handle` is one `open` gave, and nothing of the library is used after.
    pub unsafe fn close(handle: *mut c_void) {
        #[cfg(unix)]
        sys::dlclose(handle);
        #[cfg(not(unix))]
        let _ = handle;
    }
}

/// A table of C functions found at run time: the struct `$table` with a field
/// for each, `$table::load(handle)`, and (tests) `$list`, each function's name
/// and its arguments' and result's Rust types as declared here, which a test
/// holds to the C header.
macro_rules! api {
    ($table:ident, $list:ident; $(fn $name:ident($($arg:ident: $ty:ty),* $(,)?) -> $ret:ty;)*) => {
        #[allow(non_snake_case)]
        pub(crate) struct $table {
            $(pub(crate) $name: unsafe extern "C" fn($($ty),*) -> $ret,)*
        }

        impl $table {
            /// Every function from `handle`, else the name of the first one missing.
            ///
            /// # Safety
            /// `handle` is a loaded library's whose functions of these names take these arguments.
            pub(crate) unsafe fn load(handle: *mut c_void) -> std::result::Result<$table, &'static str> {
                Ok($table {
                    $($name: {
                        let p = crate::dl::sym(handle, concat!(stringify!($name), "\0"));
                        if p.is_null() {
                            return Err(stringify!($name));
                        }
                        std::mem::transmute::<*mut c_void, unsafe extern "C" fn($($ty),*) -> $ret>(p)
                    },)*
                })
            }
        }

        #[cfg(test)]
        #[allow(dead_code)]
        pub(crate) const $list: &[(&str, &[&str], &str)] = &[$((stringify!($name), &[$(stringify!($ty)),*], stringify!($ret)),)*];
    };
}
pub(crate) use api;

api! { Api, DECLARED;
    fn glyd_gpu_api_version() -> c_int;
    fn glyd_gpu_cuda_version() -> c_int;
    fn glyd_gpu_error_string(status: c_int) -> *const c_char;

    fn glyd_gpu_mma_gemm_workspace(o: i64, k: i64, m: i64, bytes: *mut usize) -> c_int;
    fn glyd_gpu_mma_gemm(data: *const u8, blocks: *const u8, block_base: *const i32, tiers: *const u32, o: i64, k: i64, x: *const u16, m: i64, bias: *const u16, y: *mut u16, workspace: *mut c_void, workspace_bytes: usize, done: *mut c_int, cs: Stream) -> c_int;
    fn glyd_gpu_mma12_gemm_workspace(o: i64, k: i64, m: i64, bytes: *mut usize) -> c_int;
    fn glyd_gpu_mma12_gemm(data: *const u8, exc: *const u32, exc_base: *const i32, sym: *const u32, o: i64, k: i64, x: *const u16, m: i64, bias: *const u16, y: *mut u16, workspace: *mut c_void, workspace_bytes: usize, done: *mut c_int, cs: Stream) -> c_int;
    fn glyd_gpu_mma_gemm_big_workspace(o: i64, k: i64, m: i64, variant: i64, bytes: *mut usize) -> c_int;
    fn glyd_gpu_mma_gemm_big(data: *const u8, blocks: *const u8, block_base: *const i32, tiers: *const u32, o: i64, k: i64, x: *const u16, m: i64, bias: *const u16, y: *mut u16, variant: i64, workspace: *mut c_void, workspace_bytes: usize, done: *mut c_int, cs: Stream) -> c_int;
    fn glyd_gpu_mma12_gemm_big_workspace(o: i64, k: i64, m: i64, variant: i64, bytes: *mut usize) -> c_int;
    fn glyd_gpu_mma12_gemm_big(data: *const u8, exc: *const u32, exc_base: *const i32, sym: *const u32, o: i64, k: i64, x: *const u16, m: i64, bias: *const u16, y: *mut u16, variant: i64, workspace: *mut c_void, workspace_bytes: usize, done: *mut c_int, cs: Stream) -> c_int;
    fn glyd_gpu_mma12_gemm_mid_workspace(o: i64, k: i64, m: i64, bytes: *mut usize) -> c_int;
    fn glyd_gpu_mma12_gemm_mid(data: *const u8, exc: *const u32, exc_base: *const i32, sym: *const u32, o: i64, k: i64, x: *const u16, m: i64, bias: *const u16, y: *mut u16, workspace: *mut c_void, workspace_bytes: usize, done: *mut c_int, cs: Stream) -> c_int;
    fn glyd_gpu_mma12_gemm_wg_workspace(o: i64, k: i64, m: i64, bytes: *mut usize) -> c_int;
    fn glyd_gpu_mma12_gemm_wg(data: *const u8, exc: *const u32, exc_base: *const i32, sym: *const u32, o: i64, k: i64, x: *const u16, m: i64, bias: *const u16, y: *mut u16, workspace: *mut c_void, workspace_bytes: usize, done: *mut c_int, cs: Stream) -> c_int;
    fn glyd_gpu_mma_unpack(data: *const u8, blocks: *const u8, block_base: *const i32, tiers: *const u32, k: i64, row0: i64, rows: i64, out: *mut u16, warps: i64, cs: Stream) -> c_int;
    fn glyd_gpu_mma12_unpack(data: *const u8, exc: *const u32, exc_base: *const i32, sym: *const u32, k: i64, row0: i64, rows: i64, out: *mut u16, warps: i64, cs: Stream) -> c_int;
    fn glyd_gpu_hold(ns: i64, cs: Stream) -> c_int;

    fn glyd_gpu_gpu(gpu: *mut c_int) -> c_int;
    fn glyd_gpu_mma_route(gpu: i64, o: i64, k: i64, m: i64, route: *mut c_int, last: *mut i64) -> c_int;
    fn glyd_gpu_mma12_route(gpu: i64, o: i64, k: i64, m: i64, route: *mut c_int, last: *mut i64) -> c_int;
    fn glyd_gpu_mma_linear_workspace(o: i64, k: i64, m: i64, route: i64, bytes: *mut usize) -> c_int;
    fn glyd_gpu_mma_linear(data: *const u8, blocks: *const u8, block_base: *const i32, tiers: *const u32, o: i64, k: i64, x: *const u16, m: i64, bias: *const u16, y: *mut u16, route: i64, workspace: *mut c_void, workspace_bytes: usize, done: *mut c_int, cs: Stream) -> c_int;
    fn glyd_gpu_mma12_linear_workspace(o: i64, k: i64, m: i64, route: i64, bytes: *mut usize) -> c_int;
    fn glyd_gpu_mma12_linear(data: *const u8, exc: *const u32, exc_base: *const i32, sym: *const u32, o: i64, k: i64, x: *const u16, m: i64, bias: *const u16, y: *mut u16, route: i64, workspace: *mut c_void, workspace_bytes: usize, done: *mut c_int, cs: Stream) -> c_int;

    fn glyd_gpu_moe_route(ids: *const i64, p: i64, e: i64, plan: *mut i32, cs: Stream) -> c_int;
    fn glyd_gpu_mma_moe_workspace(e: i64, o: i64, k: i64, t: i64, topk: i64, act: i64, weighted: i64, bytes: *mut usize) -> c_int;
    fn glyd_gpu_mma_moe(data: *const u8, blocks: *const u8, block_base: *const i32, tiers: *const u32, e: i64, o: i64, k: i64, x: *const u16, t: i64, topk: i64, gather: i64, plan: *const i32, act: i64, bias: *const u16, w: *const c_void, wf32: i64, ids: *const i64, y: *mut u16, workspace: *mut c_void, workspace_bytes: usize, done: *mut c_int, cs: Stream) -> c_int;
    fn glyd_gpu_mma12_moe_workspace(e: i64, o: i64, k: i64, t: i64, topk: i64, act: i64, weighted: i64, bytes: *mut usize) -> c_int;
    fn glyd_gpu_mma12_moe(data: *const u8, exc: *const u32, exc_base: *const i32, sym: *const u32, e: i64, o: i64, k: i64, x: *const u16, t: i64, topk: i64, gather: i64, plan: *const i32, act: i64, bias: *const u16, w: *const c_void, wf32: i64, ids: *const i64, y: *mut u16, workspace: *mut c_void, workspace_bytes: usize, done: *mut c_int, cs: Stream) -> c_int;
    fn glyd_gpu_mma_moe_unpack(data: *const u8, blocks: *const u8, block_base: *const i32, tiers: *const u32, e: i64, o: i64, k: i64, p: i64, plan: *const i32, out: *mut u16, cs: Stream) -> c_int;
    fn glyd_gpu_mma12_moe_unpack(data: *const u8, exc: *const u32, exc_base: *const i32, sym: *const u32, e: i64, o: i64, k: i64, p: i64, plan: *const i32, out: *mut u16, cs: Stream) -> c_int;

    fn glyd_gpu_attn_decode_workspace(d: i64, tlen: i64, pairs: i64, p: i64, bytes: *mut usize) -> c_int;
    fn glyd_gpu_attn_decode(q: *const u16, d: i64, kd: *const u8, kb: *const u8, kbb: *const i32, kt: *const u32, vd: *const u8, vb: *const u8, vbb: *const i32, vt: *const u32, tk: *const u16, tv: *const u16, tlen: i64, pairs: i64, g: i64, p: i64, scale: f64, out: *mut u16, workspace: *mut c_void, workspace_bytes: usize, done: *mut c_int, cs: Stream) -> c_int;

    fn glyd_gpu_fast_gemv(sm: *const u8, planes: *const u32, exc: *const u8, exc_base: *const i32, top: u64, o: i64, k: i64, x: *const u16, bias: *const u16, y: *mut u16, cs: Stream) -> c_int;
    fn glyd_gpu_fast_decode(sm: *const u8, planes: *const u32, exc: *const u8, exc_base: *const i32, top: u64, row0: i64, rows: i64, row_ids: *const i64, n_ids: i64, k: i64, out: *mut u16, cs: Stream) -> c_int;
    fn glyd_gpu_fast_gemm_workspace(o: i64, k: i64, m: i64, bytes: *mut usize) -> c_int;
    fn glyd_gpu_fast_gemm(sm: *const u8, planes: *const u32, exc: *const u8, exc_base: *const i32, top: u64, o: i64, k: i64, x: *const u16, m: i64, bias: *const u16, y: *mut u16, workspace: *mut c_void, workspace_bytes: usize, cs: Stream) -> c_int;
    fn glyd_gpu_fast_bgemv_workspace(o: i64, k: i64, m: i64, bytes: *mut usize) -> c_int;
    fn glyd_gpu_fast_bgemv(sm: *const u8, planes: *const u32, exc: *const u8, exc_base: *const i32, top: u64, o: i64, k: i64, x: *const u16, m: i64, bias: *const u16, y: *mut u16, workspace: *mut c_void, workspace_bytes: usize, cs: Stream) -> c_int;

    fn glyd_gpu_lane_bits(w: *const u16, n: i64, len: *const u8, tw: i64, v: i64, bits: *mut u32, cs: Stream) -> c_int;
    fn glyd_gpu_write_codes(w: *const u16, n: i64, len: *const u8, code: *const u32, offs: *const u32, out: *mut u32, tw: i64, v: i64, cs: Stream) -> c_int;
    fn glyd_gpu_decode(sm: *const u8, stream: *const u32, stream_words: i64, offs: *const u32, tables: *const u32, n: i64, tw: i64, v: i64, tile_words: i64, tile_ids: *const i64, n_ids: i64, out: *mut u16, cs: Stream) -> c_int;
    fn glyd_gpu_gemv(sm: *const u8, stream: *const u32, stream_words: i64, offs: *const u32, tables: *const u32, o: i64, k: i64, tw: i64, v: i64, tile_words: i64, x: *const u16, bias: *const u16, y: *mut u16, sum: *mut f32, count: *mut c_int, cs: Stream) -> c_int;
}

/// How glyd.gpu multiplies by a packed matrix for m tokens on a GPU, as
/// measured there (glyd_gpu.h's `GLYD_GPU_ROUTE_*`): the kernel its product
/// takes, or the matrix decoded for a bf16 GEMM of the caller's own where
/// that is the faster.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// Decoded ([`Library::unpack`]), then the caller's GEMM.
    Decode = 0,
    /// [`Library::gemm`]: 0 to 64 tokens.
    Gemm = 1,
    /// [`Library::gemm_mid`].
    Mid = 2,
    /// [`Library::gemm_wg`].
    Wg = 3,
    /// [`Library::gemm_big`], variant 0.
    Big = 4,
    /// As Decode, the decode run ahead beside the products before (GeForce Ada's and an A10's prompts).
    Ahead = 5,
}

impl Route {
    fn from_c(r: c_int) -> Result<Route> {
        Ok(match r {
            0 => Route::Decode,
            1 => Route::Gemm,
            2 => Route::Mid,
            3 => Route::Wg,
            4 => Route::Big,
            5 => Route::Ahead,
            _ => return Err(Error::Cuda { call: "route", status: INVALID_VALUE, text: format!("route {r}, one this crate does not know") }),
        })
    }
}

/// A matrix in the tiered mma layout (about 10.8 bits a weight): `data`
/// [steps][1280], `blocks`, `block_base` [steps + 1] in device memory, its
/// tiers' words here (glyd_gpu.h has the layout; a step is 1024 weights).
#[derive(Clone, Copy, Debug)]
pub struct Tiered {
    pub data: *const u8,
    pub blocks: *const u8,
    pub block_base: *const i32,
    pub tiers: [u32; 3],
}

/// A matrix in the 12-bit mma layout (split byte): `data` [steps][1536],
/// `exc`, `exc_base` [steps + 1] in device memory, its words here: its base
/// hb (0-120) in each byte of the first, then 0, 0, 0
/// ([`pack::twelve_words`]; the library refuses any others).
#[derive(Clone, Copy, Debug)]
pub struct Twelve {
    pub data: *const u8,
    pub exc: *const u32,
    pub exc_base: *const i32,
    pub sym: [u32; 4],
}

/// A matrix's pack in either mma layout.
#[derive(Clone, Copy, Debug)]
pub enum Pack {
    Tiered(Tiered),
    Twelve(Twelve),
}

/// W [rows, cols] packed (rows a multiple of 64, cols of 16).
#[derive(Clone, Copy, Debug)]
pub struct Matrix {
    pub pack: Pack,
    pub rows: i64,
    pub cols: i64,
}

/// A product's operands, Y [m, rows] = X [m, cols] W^T (+ bias [rows]), bf16
/// in device memory; `bias` null for none. `workspace`: at least the bytes
/// the product's query gives for these sizes on this device (null where 0).
/// `done`: the int32 counters glyd_gpu.h gives the kernel, zero before its
/// first call (each call leaves them zero); a workspace and a set of
/// counters serve one stream at a time.
#[derive(Clone, Copy, Debug)]
pub struct Product {
    pub x: *const u16,
    pub m: i64,
    pub bias: *const u16,
    pub y: *mut u16,
    pub workspace: *mut c_void,
    pub workspace_bytes: usize,
    pub done: *mut c_int,
}

/// A mixture of experts' product over W, its `experts` matrices stacked
/// ([experts rows, cols]), by a plan (`moe_route`): X [tokens, cols] (gather:
/// a pair takes its token's row) or [tokens k, cols] (the pairs in the plan's
/// order). act 0: Y [pairs, O] (+ bias [experts, O]); 1 (SiLU) or 2 (GELU,
/// tanh): Y [pairs, O / 2] = act(gate) up. `weights` (act 0; bf16, or fp32
/// where `weights_f32`) with `ids` (int64 [tokens, k]): Y [tokens, O], a
/// token's k rows times their weights, added.
#[derive(Clone, Copy, Debug)]
pub struct Moe {
    pub experts: i64,
    pub x: *const u16,
    pub tokens: i64,
    pub k: i64,
    pub gather: bool,
    pub plan: *const i32,
    pub act: i64,
    pub bias: *const u16,
    pub weights: *const c_void,
    pub weights_f32: bool,
    pub ids: *const i64,
    pub y: *mut u16,
    pub workspace: *mut c_void,
    pub workspace_bytes: usize,
    pub done: *mut c_int,
}

/// Attention for one new token a sequence over a KV cache in the tiered
/// layout (gpu/kv.py): q and out [pairs G, head_dim]; the keys P pages of
/// [pairs 64 tokens, head_dim], the values [pairs head_dim, 64 tokens], then
/// a tail of `tail` (0-63) tokens as they are, `tail_keys` and `tail_values`
/// [pairs, tail, head_dim].
#[derive(Clone, Copy, Debug)]
pub struct Attention {
    pub q: *const u16,
    pub head_dim: i64,
    pub keys: Tiered,
    pub values: Tiered,
    pub tail_keys: *const u16,
    pub tail_values: *const u16,
    pub tail: i64,
    pub pairs: i64,
    pub queries: i64,
    pub pages: i64,
    pub scale: f64,
    pub out: *mut u16,
    pub workspace: *mut c_void,
    pub workspace_bytes: usize,
    pub done: *mut c_int,
}

/// A matrix [rows, cols] in the fast format (embeddings; cols a multiple of
/// 128): `sm` [rows cols], `planes`, `exc`, `exc_base` in device memory, its
/// seven exponents in `top`.
#[derive(Clone, Copy, Debug)]
pub struct Fast {
    pub sm: *const u8,
    pub planes: *const u32,
    pub exc: *const u8,
    pub exc_base: *const i32,
    pub top: u64,
    pub rows: i64,
    pub cols: i64,
}

/// n weights in the dense format, in tiles of `tw` (V 4 or 16 a lane a step).
#[derive(Clone, Copy, Debug)]
pub struct Dense {
    pub sm: *const u8,
    pub stream: *const u32,
    pub stream_words: i64,
    pub offs: *const u32,
    pub tables: *const u32,
    pub n: i64,
    pub tw: i64,
    pub v: i64,
    pub tile_words: i64,
}

/// A call on a matrix in either mma layout: its arrays and words first, then `$rest`.
macro_rules! mma {
    ($lib:expr, $w:expr, $tiered:ident, $twelve:ident, $($rest:expr),*) => {
        match $w.pack {
            Pack::Tiered(p) => ($lib.api.$tiered)(p.data, p.blocks, p.block_base, p.tiers.as_ptr(), $($rest),*),
            Pack::Twelve(p) => ($lib.api.$twelve)(p.data, p.exc, p.exc_base, p.sym.as_ptr(), $($rest),*),
        }
    };
}

fn null(p: *const u16) -> *const u16 {
    p
}

/// libglyd_gpu_cudaN.so, loaded (and never unloaded: its CUDA runtime lives
/// to the process's end), its C API this crate's.
pub struct Library {
    api: Api,
    path: String,
}

impl Library {
    /// The library at `path` (a bare file name: the loader's search path),
    /// refused where its C API is not [`API_VERSION`] (0.21.0's has no
    /// version: 1).
    pub fn load(path: &str) -> Result<Library> {
        let h = dl::open(path).map_err(Error::Load)?;
        // SAFETY: glyd_gpu_api_version takes no argument in every version that has it.
        let v = unsafe {
            let f = dl::sym(h, "glyd_gpu_api_version\0");
            if f.is_null() {
                1
            } else {
                std::mem::transmute::<*mut c_void, unsafe extern "C" fn() -> c_int>(f)()
            }
        };
        // SAFETY: the library's C API is the one declared above (its version checked); a library refused is closed
        // before any other of its functions is called (none of its runtime left in the process).
        let api = match v == API_VERSION {
            true => unsafe { Api::load(h) }.map_err(|name| format!("{path}: no {name}")),
            false => Err(format!("{path}: its C API is version {v}, this crate's {API_VERSION}: take the library of this crate's release (gpu/build_lib.sh)")),
        };
        match api {
            Ok(api) => Ok(Library { api, path: path.to_string() }),
            Err(why) => {
                unsafe { dl::close(h) };
                Err(Error::Load(why))
            }
        }
    }

    /// `$GLYD_GPU_LIB`, else libglyd_gpu_cuda13.so or libglyd_gpu_cuda12.so
    /// by the loader's search (CUDA 13's first where the driver runs it).
    pub fn find() -> Result<Library> {
        if let Ok(path) = std::env::var("GLYD_GPU_LIB") {
            return Library::load(&path);
        }
        let names = match cuda::driver_version() {
            Ok(v) if v < 13000 => &["libglyd_gpu_cuda12.so"][..],
            _ => &["libglyd_gpu_cuda13.so", "libglyd_gpu_cuda12.so"][..],
        };
        let mut why = Vec::new();
        for name in names {
            match Library::load(name) {
                Ok(lib) => return Ok(lib),
                Err(e) => why.push(e.to_string()),
            }
        }
        Err(Error::Load(format!("no Glyd GPU library ({}): name it in GLYD_GPU_LIB, or put it on the loader's path", why.join("; "))))
    }

    /// Where it was loaded from.
    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn api_version(&self) -> i32 {
        // SAFETY: no arguments.
        unsafe { (self.api.glyd_gpu_api_version)() }
    }

    /// The CUDA runtime built in, e.g. 13000.
    pub fn cuda_version(&self) -> i32 {
        // SAFETY: no arguments.
        unsafe { (self.api.glyd_gpu_cuda_version)() }
    }

    /// A status's text.
    pub fn error_string(&self, status: i32) -> String {
        // SAFETY: the library returns a static NUL-terminated string for any status.
        unsafe {
            let s = (self.api.glyd_gpu_error_string)(status);
            if s.is_null() {
                format!("status {status}")
            } else {
                CStr::from_ptr(s).to_string_lossy().into_owned()
            }
        }
    }

    fn check(&self, call: &'static str, status: c_int) -> Result<()> {
        if status == 0 {
            Ok(())
        } else {
            Err(Error::Cuda { call, status, text: self.error_string(status) })
        }
    }

    fn bytes(&self, call: &'static str, query: impl FnOnce(*mut usize) -> c_int) -> Result<usize> {
        let mut n = 0usize;
        self.check(call, query(&mut n))?;
        Ok(n)
    }

    /// Bytes of workspace [`Library::gemm`] needs for m tokens on the current device.
    pub fn gemm_workspace(&self, w: &Matrix, m: i64) -> Result<usize> {
        // SAFETY: sizes and a host out-pointer.
        self.bytes("gemm_workspace", |b| unsafe {
            match w.pack {
                Pack::Tiered(_) => (self.api.glyd_gpu_mma_gemm_workspace)(w.rows, w.cols, m, b),
                Pack::Twelve(_) => (self.api.glyd_gpu_mma12_gemm_workspace)(w.rows, w.cols, m, b),
            }
        })
    }

    /// A generation step's product, 0 to 64 tokens, the weights decoded in
    /// registers into the tensor cores' operands. done: rows / 64 counters.
    ///
    /// # Safety
    /// See the crate's: `w`'s arrays, `p`'s operands and counters as glyd_gpu.h asks.
    pub unsafe fn gemm(&self, w: &Matrix, p: &Product, cs: Stream) -> Result<()> {
        let r = mma!(self, w, glyd_gpu_mma_gemm, glyd_gpu_mma12_gemm, w.rows, w.cols, p.x, p.m, null(p.bias), p.y, p.workspace, p.workspace_bytes, p.done, cs);
        self.check("gemm", r)
    }

    /// Bytes of workspace [`Library::gemm_big`] needs.
    pub fn gemm_big_workspace(&self, w: &Matrix, m: i64, variant: i64) -> Result<usize> {
        // SAFETY: sizes and a host out-pointer.
        self.bytes("gemm_big_workspace", |b| unsafe {
            match w.pack {
                Pack::Tiered(_) => (self.api.glyd_gpu_mma_gemm_big_workspace)(w.rows, w.cols, m, variant, b),
                Pack::Twelve(_) => (self.api.glyd_gpu_mma12_gemm_big_workspace)(w.rows, w.cols, m, variant, b),
            }
        })
    }

    /// A prompt's product (cols a multiple of 64, X 16-byte aligned), each
    /// weight decoded once for a block of tokens. variant 0: by m and the GPU;
    /// 1: blocks of 128 tokens by 128 rows; 2: 256 by 64; 3 (12-bit): 256 by
    /// 128. done: (m + 127) / 128 x rows / 64 counters.
    ///
    /// # Safety
    /// See the crate's.
    pub unsafe fn gemm_big(&self, w: &Matrix, p: &Product, variant: i64, cs: Stream) -> Result<()> {
        let r = mma!(self, w, glyd_gpu_mma_gemm_big, glyd_gpu_mma12_gemm_big, w.rows, w.cols, p.x, p.m, null(p.bias), p.y, variant, p.workspace, p.workspace_bytes, p.done, cs);
        self.check("gemm_big", r)
    }

    fn twelve(w: &Matrix, call: &'static str) -> Result<Twelve> {
        match w.pack {
            Pack::Twelve(t) => Ok(t),
            Pack::Tiered(_) => Err(Error::Cuda { call, status: INVALID_VALUE, text: "the 12-bit layout's kernel".into() }),
        }
    }

    /// Bytes of workspace [`Library::gemm_mid`] needs (the 12-bit layout).
    pub fn gemm_mid_workspace(&self, w: &Matrix, m: i64) -> Result<usize> {
        Self::twelve(w, "gemm_mid_workspace")?;
        // SAFETY: sizes and a host out-pointer.
        self.bytes("gemm_mid_workspace", |b| unsafe { (self.api.glyd_gpu_mma12_gemm_mid_workspace)(w.rows, w.cols, m, b) })
    }

    /// Many tokens in the 12-bit layout (cols a multiple of 64; X, data and
    /// exc 16-byte aligned), the packs copied into shared memory a stage at a
    /// time: Ampere and later ([`NOT_SUPPORTED`] before). done: rows / 64.
    ///
    /// # Safety
    /// See the crate's.
    pub unsafe fn gemm_mid(&self, w: &Matrix, p: &Product, cs: Stream) -> Result<()> {
        let t = Self::twelve(w, "gemm_mid")?;
        let r = (self.api.glyd_gpu_mma12_gemm_mid)(t.data, t.exc, t.exc_base, t.sym.as_ptr(), w.rows, w.cols, p.x, p.m, null(p.bias), p.y, p.workspace, p.workspace_bytes, p.done, cs);
        self.check("gemm_mid", r)
    }

    /// Bytes of workspace [`Library::gemm_wg`] needs (the 12-bit layout).
    pub fn gemm_wg_workspace(&self, w: &Matrix, m: i64) -> Result<usize> {
        Self::twelve(w, "gemm_wg_workspace")?;
        // SAFETY: sizes and a host out-pointer.
        self.bytes("gemm_wg_workspace", |b| unsafe { (self.api.glyd_gpu_mma12_gemm_wg_workspace)(w.rows, w.cols, m, b) })
    }

    /// As [`Library::gemm_mid`] on Hopper (compute capability 9.0) alone, by
    /// TMA and wgmma. done: at least 1024, or rows / 64 where that is more.
    ///
    /// # Safety
    /// See the crate's.
    pub unsafe fn gemm_wg(&self, w: &Matrix, p: &Product, cs: Stream) -> Result<()> {
        let t = Self::twelve(w, "gemm_wg")?;
        let r = (self.api.glyd_gpu_mma12_gemm_wg)(t.data, t.exc, t.exc_base, t.sym.as_ptr(), w.rows, w.cols, p.x, p.m, null(p.bias), p.y, p.workspace, p.workspace_bytes, p.done, cs);
        self.check("gemm_wg", r)
    }

    /// Rows [row0, row0 + rows) of W (multiples of 64) back to bf16, into
    /// out [rows, cols]. warps 0: a warp a step; else that many in all, each
    /// taking every so many steps (a decode beside a product on another stream).
    ///
    /// # Safety
    /// See the crate's.
    pub unsafe fn unpack(&self, w: &Matrix, row0: i64, rows: i64, out: *mut u16, warps: i64, cs: Stream) -> Result<()> {
        let r = mma!(self, w, glyd_gpu_mma_unpack, glyd_gpu_mma12_unpack, w.cols, row0, rows, out, warps, cs);
        self.check("unpack", r)
    }

    /// The current device as the routes take it, its code: its compute
    /// capability, major * 10 + minor, plus its class by name where that
    /// does not tell GPUs apart ([`GEFORCE`], [`A10`], [`L4`], [`L40S`];
    /// glyd_gpu.h): 1089 an RTX 40, 3089 an L4, 4089 an L40S, 89 an L40,
    /// 2086 an A10, 86 an A10G, A40 or RTX A6000, 80 an A100.
    pub fn gpu(&self) -> Result<i32> {
        let mut g = 0;
        // SAFETY: a host out-pointer.
        self.check("gpu", unsafe { (self.api.glyd_gpu_gpu)(&mut g) })?;
        Ok(g)
    }

    /// The route of `w` for m tokens on `gpu` (as [`Library::gpu`] gives
    /// it), and the last token count from m on that takes it (`i64::MAX`:
    /// every one past m). The library reads GLYD_WG_MIN, GLYD_WG_MAX,
    /// GLYD_MID_MIN and GLYD_DEC_MIN once, at its first route; a value that
    /// is not a whole number is taken as unset (glyd_gpu.h).
    pub fn route(&self, gpu: i32, w: &Matrix, m: i64) -> Result<(Route, i64)> {
        let (mut r, mut last) = (0, 0i64);
        // SAFETY: sizes and host out-pointers.
        let s = unsafe {
            match w.pack {
                Pack::Tiered(_) => (self.api.glyd_gpu_mma_route)(gpu as i64, w.rows, w.cols, m, &mut r, &mut last),
                Pack::Twelve(_) => (self.api.glyd_gpu_mma12_route)(gpu as i64, w.rows, w.cols, m, &mut r, &mut last),
            }
        };
        self.check("route", s)?;
        Ok((Route::from_c(r)?, last))
    }

    /// Bytes of workspace [`Library::linear`] needs by `route` (None: the current GPU's);
    /// [`NOT_SUPPORTED`] where `linear` refuses the product (cols not a multiple of 64: see it).
    pub fn linear_workspace(&self, w: &Matrix, m: i64, route: Option<Route>) -> Result<usize> {
        let r = route.map_or(-1, |r| r as i64);
        // SAFETY: sizes and a host out-pointer.
        self.bytes("linear_workspace", |b| unsafe {
            match w.pack {
                Pack::Tiered(_) => (self.api.glyd_gpu_mma_linear_workspace)(w.rows, w.cols, m, r, b),
                Pack::Twelve(_) => (self.api.glyd_gpu_mma12_linear_workspace)(w.rows, w.cols, m, r, b),
            }
        })
    }

    /// Y = X W^T (+ bias) by a route (None: the current GPU's for p.m): its
    /// kernel; [`Route::Decode`] and [`Route::Ahead`] by the prompt kernel on
    /// every GPU. Where cols is not a multiple of 64 no kernel takes those two,
    /// and such a matrix's route is one of them past 64 tokens on every GPU
    /// (in the 12-bit layout also from GLYD_DEC_MIN tokens where that is set
    /// lower): there this and [`Library::linear_workspace`] return
    /// [`Error::Cuda`] with status [`NOT_SUPPORTED`], nothing launched; decode
    /// W ([`Library::unpack`]) for a GEMM of the caller's (glyd_gpu.h; the
    /// linear example). done: (m + 127) / 128 x rows / 64 counters, and at
    /// least 1024 (the WG route's, as [`Library::gemm_wg`]'s).
    ///
    /// # Safety
    /// See the crate's.
    pub unsafe fn linear(&self, w: &Matrix, p: &Product, route: Option<Route>, cs: Stream) -> Result<()> {
        let r = mma!(self, w, glyd_gpu_mma_linear, glyd_gpu_mma12_linear, w.rows, w.cols, p.x, p.m, null(p.bias), p.y, route.map_or(-1, |r| r as i64), p.workspace, p.workspace_bytes, p.done, cs);
        self.check("linear", r)
    }

    /// Stream cs held ns nanoseconds by one thread.
    ///
    /// # Safety
    /// `cs` is a stream of the current context (or the default).
    pub unsafe fn hold(&self, ns: i64, cs: Stream) -> Result<()> {
        self.check("hold", (self.api.glyd_gpu_hold)(ns, cs))
    }

    /// The plan (int32 [2 + 2E + P]) of P pairs whose experts are `ids` (int64 [P]), sorted by expert.
    ///
    /// # Safety
    /// See the crate's.
    pub unsafe fn moe_route(&self, ids: *const i64, pairs: i64, experts: i64, plan: *mut i32, cs: Stream) -> Result<()> {
        self.check("moe_route", (self.api.glyd_gpu_moe_route)(ids, pairs, experts, plan, cs))
    }

    /// Bytes of workspace [`Library::moe`] needs.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_workspace(&self, w: &Matrix, experts: i64, tokens: i64, k: i64, act: i64, weighted: bool) -> Result<usize> {
        let o = w.rows / experts.max(1);
        // SAFETY: sizes and a host out-pointer.
        self.bytes("moe_workspace", |b| unsafe {
            match w.pack {
                Pack::Tiered(_) => (self.api.glyd_gpu_mma_moe_workspace)(experts, o, w.cols, tokens, k, act, weighted as i64, b),
                Pack::Twelve(_) => (self.api.glyd_gpu_mma12_moe_workspace)(experts, o, w.cols, tokens, k, act, weighted as i64, b),
            }
        })
    }

    /// The experts' product by the plan. done: rows / experts / 64 (/ 128 with act) x min(experts, pairs) counters.
    ///
    /// # Safety
    /// See the crate's.
    pub unsafe fn moe(&self, w: &Matrix, m: &Moe, cs: Stream) -> Result<()> {
        let o = w.rows / m.experts.max(1);
        let r = mma!(self, w, glyd_gpu_mma_moe, glyd_gpu_mma12_moe, m.experts, o, w.cols, m.x, m.tokens, m.k, m.gather as i64, m.plan, m.act, null(m.bias), m.weights, m.weights_f32 as i64, m.ids, m.y, m.workspace, m.workspace_bytes, m.done, cs);
        self.check("moe", r)
    }

    /// Exact: the experts the plan's P pairs hit back to bf16, into their rows of out [rows, cols].
    ///
    /// # Safety
    /// See the crate's.
    pub unsafe fn moe_unpack(&self, w: &Matrix, experts: i64, pairs: i64, plan: *const i32, out: *mut u16, cs: Stream) -> Result<()> {
        let o = w.rows / experts.max(1);
        let r = mma!(self, w, glyd_gpu_mma_moe_unpack, glyd_gpu_mma12_moe_unpack, experts, o, w.cols, pairs, plan, out, cs);
        self.check("moe_unpack", r)
    }

    /// Bytes of workspace [`Library::attn_decode`] needs.
    pub fn attn_decode_workspace(&self, head_dim: i64, tail: i64, pairs: i64, pages: i64) -> Result<usize> {
        // SAFETY: sizes and a host out-pointer.
        self.bytes("attn_decode_workspace", |b| unsafe { (self.api.glyd_gpu_attn_decode_workspace)(head_dim, tail, pairs, pages, b) })
    }

    /// Attention for one new token a sequence. done: pairs counters.
    ///
    /// # Safety
    /// See the crate's.
    pub unsafe fn attn_decode(&self, a: &Attention, cs: Stream) -> Result<()> {
        let (k, v) = (a.keys, a.values);
        let r = (self.api.glyd_gpu_attn_decode)(a.q, a.head_dim, k.data, k.blocks, k.block_base, k.tiers.as_ptr(), v.data, v.blocks, v.block_base, v.tiers.as_ptr(), a.tail_keys, a.tail_values, a.tail, a.pairs, a.queries, a.pages, a.scale, a.out, a.workspace, a.workspace_bytes, a.done, cs);
        self.check("attn_decode", r)
    }

    /// y [rows] = W x (+ bias) for one token (the fast format).
    ///
    /// # Safety
    /// See the crate's.
    pub unsafe fn fast_gemv(&self, w: &Fast, x: *const u16, bias: *const u16, y: *mut u16, cs: Stream) -> Result<()> {
        let r = (self.api.glyd_gpu_fast_gemv)(w.sm, w.planes, w.exc, w.exc_base, w.top, w.rows, w.cols, x, bias, y, cs);
        self.check("fast_gemv", r)
    }

    /// Rows [row0, row0 + rows), or (`row_ids` non-empty: int64 in device
    /// memory, `n_ids` of them) those rows, into out [rows, cols].
    ///
    /// # Safety
    /// See the crate's.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn fast_decode(&self, w: &Fast, row0: i64, rows: i64, row_ids: *const i64, n_ids: i64, out: *mut u16, cs: Stream) -> Result<()> {
        let r = (self.api.glyd_gpu_fast_decode)(w.sm, w.planes, w.exc, w.exc_base, w.top, row0, rows, row_ids, n_ids, w.cols, out, cs);
        self.check("fast_decode", r)
    }

    /// Bytes of workspace [`Library::fast_gemm`] needs.
    pub fn fast_gemm_workspace(&self, w: &Fast, m: i64) -> Result<usize> {
        // SAFETY: sizes and a host out-pointer.
        self.bytes("fast_gemm_workspace", |b| unsafe { (self.api.glyd_gpu_fast_gemm_workspace)(w.rows, w.cols, m, b) })
    }

    /// Y = X W^T (+ bias) on the tensor cores (cols a multiple of 64, rows of 16); `p.done` unused.
    ///
    /// # Safety
    /// See the crate's.
    pub unsafe fn fast_gemm(&self, w: &Fast, p: &Product, cs: Stream) -> Result<()> {
        let r = (self.api.glyd_gpu_fast_gemm)(w.sm, w.planes, w.exc, w.exc_base, w.top, w.rows, w.cols, p.x, p.m, p.bias, p.y, p.workspace, p.workspace_bytes, cs);
        self.check("fast_gemm", r)
    }

    /// Bytes of workspace [`Library::fast_bgemv`] needs.
    pub fn fast_bgemv_workspace(&self, w: &Fast, m: i64) -> Result<usize> {
        // SAFETY: sizes and a host out-pointer.
        self.bytes("fast_bgemv_workspace", |b| unsafe { (self.api.glyd_gpu_fast_bgemv_workspace)(w.rows, w.cols, m, b) })
    }

    /// The same for 2, 4, 8 or 16 tokens on the CUDA cores (cols a multiple of 512); `p.done` unused.
    ///
    /// # Safety
    /// See the crate's.
    pub unsafe fn fast_bgemv(&self, w: &Fast, p: &Product, cs: Stream) -> Result<()> {
        let r = (self.api.glyd_gpu_fast_bgemv)(w.sm, w.planes, w.exc, w.exc_base, w.top, w.rows, w.cols, p.x, p.m, p.bias, p.y, p.workspace, p.workspace_bytes, cs);
        self.check("fast_bgemv", r)
    }

    /// The dense format's packer, first pass: each lane's stream's length in
    /// bits [tiles 32], for `len` [256] each exponent's code length.
    ///
    /// # Safety
    /// See the crate's.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn lane_bits(&self, w: *const u16, n: i64, len: *const u8, tw: i64, v: i64, bits: *mut u32, cs: Stream) -> Result<()> {
        self.check("lane_bits", (self.api.glyd_gpu_lane_bits)(w, n, len, tw, v, bits, cs))
    }

    /// Its second: the streams into out (zero before) from offs, code [256] each exponent's code.
    ///
    /// # Safety
    /// See the crate's.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn write_codes(&self, w: *const u16, n: i64, len: *const u8, code: *const u32, offs: *const u32, out: *mut u32, tw: i64, v: i64, cs: Stream) -> Result<()> {
        self.check("write_codes", (self.api.glyd_gpu_write_codes)(w, n, len, code, offs, out, tw, v, cs))
    }

    /// Every tile into out [n], or (n_ids > 0) tiles `tile_ids` into out [n_ids tw].
    ///
    /// # Safety
    /// See the crate's.
    pub unsafe fn decode(&self, d: &Dense, tile_ids: *const i64, n_ids: i64, out: *mut u16, cs: Stream) -> Result<()> {
        let r = (self.api.glyd_gpu_decode)(d.sm, d.stream, d.stream_words, d.offs, d.tables, d.n, d.tw, d.v, d.tile_words, tile_ids, n_ids, out, cs);
        self.check("decode", r)
    }

    /// y [o] = W x (+ bias) for W [o, k] in the dense format; where tiles
    /// split rows, sum (f32) and count (int32) [o], zero before and left zero.
    ///
    /// # Safety
    /// See the crate's.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemv(&self, d: &Dense, o: i64, k: i64, x: *const u16, bias: *const u16, y: *mut u16, sum: *mut f32, count: *mut c_int, cs: Stream) -> Result<()> {
        let r = (self.api.glyd_gpu_gemv)(d.sm, d.stream, d.stream_words, d.offs, d.tables, o, k, d.tw, d.v, d.tile_words, x, bias, y, sum, count, cs);
        self.check("gemv", r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A C parameter's or result's type as this crate declares it: `const
    /// uint8_t* data` *const u8, `const uint32_t tiers[3]` *const u32,
    /// `int* done` *mut c_int, `cudaStream_t cs` Stream.
    fn rust_type(c: &str) -> String {
        let c = c.split_whitespace().collect::<Vec<_>>().join(" ").replace(" *", "*");
        let (mut ty, name) = match c.rsplit_once(' ') {
            Some((t, n)) if !n.ends_with('*') => (t.to_string(), n.to_string()),
            _ => (c.clone(), String::new()),
        };
        if name.ends_with(']') {
            ty.push('*'); // an array of host words: a pointer to them
        }
        let konst = ty.starts_with("const ");
        let ty = ty.trim_start_matches("const ");
        let (base, ptr) = match ty.strip_suffix('*') {
            Some(b) => (b, true),
            None => (ty, false),
        };
        let base = match base {
            "int64_t" => "i64",
            "uint64_t" => "u64",
            "size_t" => "usize",
            "double" => "f64",
            "int" => "c_int",
            "void" => "c_void",
            "char" => "c_char",
            "uint8_t" => "u8",
            "uint16_t" => "u16",
            "int32_t" => "i32",
            "uint32_t" => "u32",
            "float" => "f32",
            "cudaStream_t" => "Stream",
            other => panic!("a C type this test does not know: {other}"),
        };
        match (ptr, konst) {
            (true, true) => format!("*const {base}"),
            (true, false) => format!("*mut {base}"),
            _ => base.to_string(),
        }
    }

    /// Every function of gpu/glyd_gpu.h declared here by the same arguments
    /// and result, and nothing else; its version API_VERSION.
    #[test]
    fn declarations_are_the_headers() {
        let Ok(h) = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../gpu/glyd_gpu.h")) else {
            eprintln!("skipped: ../gpu/glyd_gpu.h is not beside the crate (a published crate)");
            return;
        };
        let mut text = String::new();
        let mut rest = h.as_str();
        while let Some(i) = rest.find("/*") {
            text.push_str(&rest[..i]);
            rest = &rest[i + rest[i..].find("*/").unwrap() + 2..];
        }
        text.push_str(rest);
        let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
        let mut declared = Vec::new();
        for part in text.split(';') {
            let Some(at) = part.find("glyd_gpu_") else { continue };
            let Some(open) = part[at..].find('(').map(|i| at + i) else { continue };
            let ret = part[..at].trim();
            if !(ret == "int" || ret.ends_with(" int") || ret.ends_with("const char*")) {
                continue;
            }
            let ret = if ret.ends_with("const char*") { "*const c_char".to_string() } else { "c_int".to_string() };
            let name = &part[at..open];
            let args = part[open + 1..part.rfind(')').unwrap()].trim();
            let args: Vec<String> = if args == "void" { vec![] } else { args.split(',').map(rust_type).collect() };
            declared.push((name.to_string(), args, ret));
        }
        let ours: Vec<(String, Vec<String>, String)> = DECLARED.iter().map(|(n, a, r)| (n.to_string(), a.iter().map(|s| s.replace(' ', "").replace("*const", "*const ").replace("*mut", "*mut ")).collect(), r.replace(' ', "").replace("*const", "*const "))).collect();
        let mut theirs = declared.clone();
        theirs.sort();
        let mut mine = ours.clone();
        mine.sort();
        for d in &theirs {
            assert!(mine.contains(d), "glyd_gpu.h's {} is not declared so here: {:?}", d.0, d);
        }
        for d in &mine {
            assert!(theirs.contains(d), "{} is declared here, not so in glyd_gpu.h: {:?}, the header's {:?}", d.0, d, theirs.iter().find(|t| t.0 == d.0));
        }
        let define = |name: &str| h.split(&format!("#define {name} ")).nth(1).and_then(|v| v.split_whitespace().next()).and_then(|v| v.parse::<i32>().ok()).unwrap_or_else(|| panic!("glyd_gpu.h has no {name}"));
        assert_eq!(define("GLYD_GPU_API_VERSION"), API_VERSION, "GLYD_GPU_API_VERSION is not API_VERSION");
        for (name, r) in [("DECODE", Route::Decode), ("GEMM", Route::Gemm), ("MID", Route::Mid), ("WG", Route::Wg), ("BIG", Route::Big), ("AHEAD", Route::Ahead)] {
            assert_eq!(define(&format!("GLYD_GPU_ROUTE_{name}")), r as i32, "GLYD_GPU_ROUTE_{name}");
            assert_eq!(Route::from_c(r as c_int).unwrap(), r);
        }
        assert_eq!((define("GLYD_GPU_GEFORCE"), define("GLYD_GPU_A10"), define("GLYD_GPU_L4"), define("GLYD_GPU_L40S")), (GEFORCE, A10, L4, L40S));
    }

    #[test]
    fn no_library_is_an_error() {
        let e = Library::load("/no/such/libglyd_gpu_cuda13.so").err().unwrap();
        assert!(matches!(e, Error::Load(_)), "{e}");
    }

    /// With the library and a GPU (else nothing to check): `linear` takes every route this GPU's gives, in both
    /// layouts (its workspace query answers), but a prompt's Decode and Ahead where cols is not a multiple of 64,
    /// which it refuses with NOT_SUPPORTED (glyd_gpu.h).
    #[test]
    fn linear_takes_every_route_but_one() {
        let (Ok(lib), Ok(_ctx)) = (Library::find(), cuda::Context::new(0)) else { return };
        let gpu = lib.gpu().unwrap();
        let none = std::ptr::null();
        let packs = [
            Pack::Tiered(Tiered { data: none, blocks: none, block_base: none as *const i32, tiers: [0; 3] }),
            Pack::Twelve(Twelve { data: none, exc: none as *const u32, exc_base: none as *const i32, sym: pack::twelve_words(0) }),
        ];
        for pack in packs {
            for cols in [1024, 1040] {
                let w = Matrix { pack, rows: 128, cols };
                for m in [1, 17, 64, 65, 129, 600, 769, 1100, 2000] {
                    let (route, _) = lib.route(gpu, &w, m).unwrap();
                    let got = lib.linear_workspace(&w, m, None);
                    if cols % 64 != 0 && matches!(route, Route::Decode | Route::Ahead) {
                        assert!(matches!(got, Err(Error::Cuda { status: NOT_SUPPORTED, .. })), "{pack:?} {cols} {m} {route:?}: {got:?}");
                    } else {
                        assert!(got.is_ok(), "{pack:?} {cols} {m} {route:?}: {got:?}");
                    }
                }
            }
        }
    }
}
