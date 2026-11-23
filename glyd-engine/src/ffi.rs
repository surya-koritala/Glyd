//! The binding: the CUDA driver and Glyd's GPU library (the C API of gpu/glyd_gpu.h, version 16), both loaded at run
//! time; nothing is linked at build time. This is the only module that names a symbol: the library's functions are
//! resolved in `Lib::load` by their names (every function of glyd_gpu.h is exported by its name). The declarations are held to
//! gpu/glyd_gpu.h by a test (`declarations_are_the_header`).
//!
//! Device memory is a `u64` address (the driver's CUdeviceptr); the library takes it as a pointer. Every call returns
//! a `Result` with the status text of the driver or the library.

use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::fmt;
use std::sync::OnceLock;

/// What an error is, for the parts of the program that act on it and not on its words (ROADMAP R16): `run` and the planner (the GPU's memory, no driver), the server and the service (a GPU
/// whose context is not to be used again). A call of the driver or of the GPU library that fails gives its status here and its call in the words (docs/design/runtime.md section 4).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Kind {
    /// None of the others: its words say what.
    #[default]
    Other,
    /// The GPU has no memory for what was asked of it (the driver's and the library's status 2).
    OutOfMemory,
    /// This computer has no GPU driver to load.
    NoDriver,
    /// The request's own doing: a prompt longer than the context, or than the cache holds with nothing else running. The server answers it 400; every other kind is the server's (500).
    Request,
    /// A call of the driver or of the GPU library failed with `status`. `sticky`: the GPU's context is not to be used after it (every later call fails the same way), so the model that ran
    /// on it stops, and the service starts it again.
    Device { status: i32, sticky: bool },
}

impl Kind {
    /// The kind of a status of the driver (CUDA_ERROR_*) or of the GPU library, whose statuses are the CUDA runtime's (the same numbers) and its own from `GLYD_GPU_BLAS_ERROR` on.
    pub fn of_status(status: c_int) -> Kind {
        match status {
            2 => Kind::OutOfMemory,
            // deinitialized, no device, an ECC error the GPU could not correct, an illegal address, a launch timed out, a hardware stack error, the context destroyed, an assert in a kernel,
            // an illegal instruction, a misaligned address, an invalid address space, an invalid program counter, a launch failed, unknown: the context is lost
            4 | 100 | 214 | 700 | 702 | 708 | 709 | 710 | 715 | 716 | 717 | 718 | 719 | 999 => Kind::Device { status, sticky: true },
            _ => Kind::Device { status, sticky: false },
        }
    }
}

/// An error in words, and what kind it is (`Kind`). `Error(words)` makes one: where the words hold those of the last error of a kind made on this thread, which is how an error is wrapped
/// (`Error(format!("layer 3: {e}"))`), it keeps that kind; else it is `Kind::Other`. `Error::of` makes one of a kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub msg: String,
    pub kind: Kind,
}

thread_local! {
    /// The last error of a kind made on this thread: its words and its kind (see `Error`).
    static LAST: std::cell::RefCell<Option<(String, Kind)>> = const { std::cell::RefCell::new(None) };
}

/// An error of these words (see the type `Error`): `Kind::Other`, or the kind of the error whose words these hold.
#[allow(non_snake_case)]
pub fn Error(msg: String) -> Error {
    let kind = LAST.with(|l| match &*l.borrow() {
        Some((m, k)) if msg.contains(m.as_str()) => *k,
        _ => Kind::Other,
    });
    if kind == Kind::Other {
        Error { msg, kind }
    } else {
        Error::of(kind, msg)
    }
}

impl Error {
    /// An error of `kind` in these words.
    pub fn of(kind: Kind, msg: impl Into<String>) -> Error {
        let msg = msg.into();
        if kind != Kind::Other {
            LAST.with(|l| *l.borrow_mut() = Some((msg.clone(), kind)));
        }
        Error { msg, kind }
    }

    /// The same error with `what` before its words ("WHAT: words").
    pub fn context(self, what: impl fmt::Display) -> Error {
        Error::of(self.kind, format!("{what}: {}", self.msg))
    }

    /// Whether the GPU's context is not to be used after it (`Kind::Device`'s `sticky`).
    pub fn sticky(&self) -> bool {
        matches!(self.kind, Kind::Device { sticky: true, .. })
    }

    /// The status of the driver's or the library's call that failed: the kind's, or, for an error that came here as words (from another process, or through a channel as text), the "(N)" its
    /// words end in, which is how `cu` and `Lib::ck` write it.
    pub fn status(&self) -> Option<i32> {
        match self.kind {
            Kind::Device { status, .. } => Some(status),
            Kind::OutOfMemory => Some(2),
            _ => self.msg.rsplit_once('(').and_then(|(_, n)| n.strip_suffix(')')).and_then(|n| n.parse().ok()),
        }
    }

    /// Whether the GPU had no memory for what was asked of it.
    pub fn out_of_memory(&self) -> bool {
        self.kind == Kind::OutOfMemory || self.status() == Some(2)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.msg)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub fn err<T>(s: impl Into<String>) -> Result<T> {
    Err(Error(s.into()))
}

/// A CUDA stream (the runtime's cudaStream_t and the driver's CUstream are the same handle). Made by the device module alone (`dev::Dev`, `dev::Side`; the field is not
/// public, so a stream is not made from any pointer); `DEFAULT` is the context's legacy stream.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stream(pub(crate) *mut c_void);

unsafe impl Send for Stream {}

impl Stream {
    pub const DEFAULT: Stream = Stream(std::ptr::null_mut());
}

pub type CuFunction = *mut c_void;
pub type CuModule = *mut c_void;
pub type CuGraph = *mut c_void;
pub type CuGraphExec = *mut c_void;
pub type CuEvent = *mut c_void;

/// The CUDA driver's library, cuBLAS's and Glyd's GPU library by the system's names for them (`{}`: the CUDA major version).
#[cfg(all(unix, not(glyd_hip)))]
pub const CUDA_DRIVER: &str = "libcuda.so.1";
#[cfg(all(windows, not(glyd_hip)))]
pub const CUDA_DRIVER: &str = "nvcuda.dll";
#[cfg(unix)]
const CUBLAS: [&str; 2] = ["libcublas.so.13", "libcublas.so.12"];
#[cfg(windows)]
const CUBLAS: [&str; 2] = ["cublas64_13.dll", "cublas64_12.dll"];
#[cfg(all(unix, not(glyd_hip)))]
const GPU_LIB: [&str; 2] = ["libglyd_gpu_cuda13.so", "libglyd_gpu_cuda12.so"];
#[cfg(all(windows, not(glyd_hip)))]
const GPU_LIB: [&str; 2] = ["libglyd_gpu_cuda13.dll", "libglyd_gpu_cuda12.dll"];
// (an AMD build: the library hipcc made of the same source, gpu/build_lib_hip.sh)
#[cfg(all(unix, glyd_hip))]
const GPU_LIB: [&str; 1] = ["libglyd_gpu_hip.so"];
#[cfg(all(windows, glyd_hip))]
const GPU_LIB: [&str; 1] = ["libglyd_gpu_hip.dll"];

mod dl {
    use std::ffi::{c_char, c_void};
    #[cfg(unix)]
    use std::ffi::{c_int, CStr, CString};

    #[cfg(target_os = "macos")]
    const FLAGS: c_int = 0x2 | 0x4; // RTLD_NOW | RTLD_LOCAL
    #[cfg(all(unix, not(target_os = "macos")))]
    const FLAGS: c_int = 0x2; // RTLD_NOW

    #[cfg(unix)]
    extern "C" {
        fn dlopen(file: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
        fn dlerror() -> *const c_char;
    }

    #[cfg(windows)]
    pub fn open(path: &str) -> Result<*mut c_void, String> {
        // SAFETY: a path; the search never takes the current folder (a DLL of the name planted there is not loaded: os::win::load_library).
        let h = unsafe { crate::os::win::load_library(path) };
        if h.is_null() {
            return Err(format!("{path}: {}", std::io::Error::last_os_error()));
        }
        Ok(h)
    }

    /// # Safety
    /// `handle` is one `open` gave; `name` is NUL-terminated.
    #[cfg(windows)]
    pub unsafe fn sym(handle: *mut c_void, name: &str) -> *mut c_void {
        crate::os::win::GetProcAddress(handle, name.as_ptr() as *const c_char)
    }

    #[cfg(unix)]
    pub fn open(path: &str) -> Result<*mut c_void, String> {
        let c = CString::new(path).map_err(|_| format!("{path}: a NUL in the path"))?;
        // SAFETY: a NUL-terminated path; dlerror's text is copied before another dl call.
        unsafe {
            let h = dlopen(c.as_ptr(), FLAGS);
            if h.is_null() {
                let e = dlerror();
                return Err(if e.is_null() { format!("{path}: not loaded") } else { CStr::from_ptr(e).to_string_lossy().into_owned() });
            }
            Ok(h)
        }
    }

    /// # Safety
    /// `handle` is one `open` gave; `name` is NUL-terminated.
    #[cfg(unix)]
    pub unsafe fn sym(handle: *mut c_void, name: &str) -> *mut c_void {
        dlsym(handle, name.as_ptr() as *const c_char)
    }
}

/// A table of function pointers, each found by its field's name: `resolve` by the loader's symbol of that name, `resolve_with` by a lookup of the caller's.
macro_rules! table {
    ($vis:vis struct $name:ident { $( fn $f:ident ( $($arg:ty),* ) $(-> $ret:ty)? ; )* }) => {
        #[allow(non_snake_case, dead_code)]
        $vis struct $name { $( pub $f: unsafe extern "C" fn($($arg),*) $(-> $ret)? ),* }
        impl $name {
            /// # Safety
            /// `h` is a handle of a library whose functions of these names have these signatures.
            #[allow(dead_code)]
            unsafe fn resolve(h: *mut c_void) -> std::result::Result<$name, String> {
                Self::resolve_with(&|n| dl::sym(h, &format!("{n}\0")))
            }

            /// # Safety
            /// `find` gives, for a name, the address of a function of that name with the signature declared here (or null).
            unsafe fn resolve_with(find: &dyn Fn(&str) -> *mut c_void) -> std::result::Result<$name, String> {
                Ok($name { $( $f: {
                    let p = find(stringify!($f));
                    if p.is_null() { return Err(format!("no {}", stringify!($f))); }
                    std::mem::transmute::<*mut c_void, unsafe extern "C" fn($($arg),*) $(-> $ret)?>(p)
                } ),* })
            }

            /// (Tests) Each function's name and its arguments' and result's Rust types as declared here, which a test holds to the C header.
            #[cfg(test)]
            #[allow(dead_code)]
            pub(crate) const DECLARED: &'static [(&'static str, &'static [&'static str], &'static str)] =
                &[$((stringify!($f), &[$(stringify!($arg)),*], concat!("" $(, stringify!($ret))?)),)*];
        }
    };
}

table! {
    pub struct Driver {
        fn cuInit(c_uint) -> c_int;
        fn cuDriverGetVersion(*mut c_int) -> c_int;
        fn cuDeviceGet(*mut c_int, c_int) -> c_int;
        fn cuDeviceGetCount(*mut c_int) -> c_int;
        fn cuDeviceGetName(*mut c_char, c_int, c_int) -> c_int;
        fn cuDeviceGetAttribute(*mut c_int, c_int, c_int) -> c_int;
        fn cuDeviceTotalMem_v2(*mut usize, c_int) -> c_int;
        fn cuDevicePrimaryCtxRetain(*mut *mut c_void, c_int) -> c_int;
        fn cuDevicePrimaryCtxRelease_v2(c_int) -> c_int;
        fn cuCtxSetCurrent(*mut c_void) -> c_int;
        fn cuCtxGetCurrent(*mut *mut c_void) -> c_int;
        fn cuCtxCreate_v2(*mut *mut c_void, c_uint, c_int) -> c_int;
        fn cuCtxSynchronize() -> c_int;
        fn cuDeviceCanAccessPeer(*mut c_int, c_int, c_int) -> c_int;
        fn cuCtxEnablePeerAccess(*mut c_void, c_uint) -> c_int;
        fn cuMemGetInfo_v2(*mut usize, *mut usize) -> c_int;
        fn cuMemAlloc_v2(*mut u64, usize) -> c_int;
        fn cuMemFree_v2(u64) -> c_int;
        fn cuMemcpyHtoD_v2(u64, *const c_void, usize) -> c_int;
        fn cuMemcpyHtoDAsync_v2(u64, *const c_void, usize, Stream) -> c_int;
        fn cuMemcpyDtoH_v2(*mut c_void, u64, usize) -> c_int;
        fn cuMemcpyDtoHAsync_v2(*mut c_void, u64, usize, Stream) -> c_int;
        fn cuMemcpyDtoDAsync_v2(u64, u64, usize, Stream) -> c_int;
        fn cuMemsetD8_v2(u64, u8, usize) -> c_int;
        fn cuMemsetD8Async(u64, u8, usize, Stream) -> c_int;
        fn cuMemHostAlloc(*mut *mut c_void, usize, c_uint) -> c_int;
        fn cuMemHostRegister_v2(*mut c_void, usize, c_uint) -> c_int;
        fn cuMemHostUnregister(*mut c_void) -> c_int;
        fn cuMemFreeHost(*mut c_void) -> c_int;
        fn cuStreamCreate(*mut Stream, c_uint) -> c_int;
        fn cuStreamDestroy_v2(Stream) -> c_int;
        fn cuStreamSynchronize(Stream) -> c_int;
        fn cuStreamBeginCapture_v2(Stream, c_int) -> c_int;
        fn cuStreamEndCapture(Stream, *mut CuGraph) -> c_int;
        fn cuGraphInstantiateWithFlags(*mut CuGraphExec, CuGraph, u64) -> c_int;
        fn cuGraphLaunch(CuGraphExec, Stream) -> c_int;
        fn cuGraphExecDestroy(CuGraphExec) -> c_int;
        fn cuGraphDestroy(CuGraph) -> c_int;
        fn cuEventCreate(*mut CuEvent, c_uint) -> c_int;
        fn cuEventRecord(CuEvent, Stream) -> c_int;
        fn cuEventRecordWithFlags(CuEvent, Stream, c_uint) -> c_int;
        fn cuEventQuery(CuEvent) -> c_int;
        fn cuStreamWaitEvent(Stream, CuEvent, c_uint) -> c_int;
        fn cuEventSynchronize(CuEvent) -> c_int;
        fn cuEventElapsedTime(*mut f32, CuEvent, CuEvent) -> c_int;
        fn cuEventDestroy_v2(CuEvent) -> c_int;
        fn cuModuleLoadData(*mut CuModule, *const c_void) -> c_int;
        fn cuModuleGetFunction(*mut CuFunction, CuModule, *const c_char) -> c_int;
        fn cuFuncSetAttribute(CuFunction, c_int, c_int) -> c_int;
        fn cuLaunchKernel(CuFunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, Stream, *mut *mut c_void, *mut *mut c_void) -> c_int;
        fn cuGetErrorString(c_int, *mut *const c_char) -> c_int;
    }
}

/// AMD GPUs: a build with GLYD_HIP=1 (build.rs: the kernels by hipcc, cfg `glyd_hip`) takes the driver's table from HIP's runtime, libamdhip64, whose functions take
/// the driver API's calls with the same arguments and flags (the numbers of the flags the engine passes are the same in HIP: a non-blocking stream 1, a relaxed capture 2,
/// portable pinned memory 1, mapped 2, an event without timing 2, an external record 1, "not ready" 600). Two calls differ: a device's attributes go by HIP's numbers, and a kernel
/// needs no opt-in for its dynamic shared memory (a block takes up to the device's 64 KiB, RDNA 3 and 4; a launch asking more fails there).
/// The choice is the build's, not the machine's: a binary holds one vendor's kernels, so the other vendor's runtime could not run it anyway, and the CUDA build stays as it was.
#[cfg(glyd_hip)]
mod hip {
    use super::{dl, Driver};
    use std::ffi::{c_int, c_void};
    use std::sync::OnceLock;

    #[cfg(unix)]
    const LIBS: [&str; 2] = ["libamdhip64.so.7", "libamdhip64.so.6"];
    #[cfg(windows)]
    const LIBS: [&str; 2] = ["amdhip64_7.dll", "amdhip64_6.dll"];

    /// The driver table's functions by HIP's names (ROCm 6.4 on: hipEventRecordWithFlags), all but the two adapters' below.
    const NAMES: [(&str, &str); 49] = [
        ("cuInit", "hipInit"),
        ("cuDriverGetVersion", "hipDriverGetVersion"),
        ("cuDeviceGet", "hipDeviceGet"),
        ("cuDeviceGetCount", "hipGetDeviceCount"),
        ("cuDeviceGetName", "hipDeviceGetName"),
        ("cuDeviceTotalMem_v2", "hipDeviceTotalMem"),
        ("cuDevicePrimaryCtxRetain", "hipDevicePrimaryCtxRetain"),
        ("cuCtxCreate_v2", "hipCtxCreate"),
        ("cuDevicePrimaryCtxRelease_v2", "hipDevicePrimaryCtxRelease"),
        ("cuCtxSetCurrent", "hipCtxSetCurrent"),
        ("cuCtxGetCurrent", "hipCtxGetCurrent"),
        ("cuCtxSynchronize", "hipDeviceSynchronize"),
        ("cuDeviceCanAccessPeer", "hipDeviceCanAccessPeer"),
        ("cuCtxEnablePeerAccess", "hipCtxEnablePeerAccess"),
        ("cuMemGetInfo_v2", "hipMemGetInfo"),
        ("cuMemAlloc_v2", "hipMalloc"),
        ("cuMemFree_v2", "hipFree"),
        ("cuMemcpyHtoD_v2", "hipMemcpyHtoD"),
        ("cuMemcpyHtoDAsync_v2", "hipMemcpyHtoDAsync"),
        ("cuMemcpyDtoH_v2", "hipMemcpyDtoH"),
        ("cuMemcpyDtoHAsync_v2", "hipMemcpyDtoHAsync"),
        ("cuMemcpyDtoDAsync_v2", "hipMemcpyDtoDAsync"),
        ("cuMemsetD8_v2", "hipMemsetD8"),
        ("cuMemsetD8Async", "hipMemsetD8Async"),
        ("cuMemHostAlloc", "hipHostMalloc"),
        ("cuMemHostRegister_v2", "hipHostRegister"),
        ("cuMemHostUnregister", "hipHostUnregister"),
        ("cuMemFreeHost", "hipHostFree"),
        ("cuStreamCreate", "hipStreamCreateWithFlags"),
        ("cuStreamDestroy_v2", "hipStreamDestroy"),
        ("cuStreamSynchronize", "hipStreamSynchronize"),
        ("cuStreamBeginCapture_v2", "hipStreamBeginCapture"),
        ("cuStreamEndCapture", "hipStreamEndCapture"),
        ("cuGraphInstantiateWithFlags", "hipGraphInstantiateWithFlags"),
        ("cuGraphLaunch", "hipGraphLaunch"),
        ("cuGraphExecDestroy", "hipGraphExecDestroy"),
        ("cuGraphDestroy", "hipGraphDestroy"),
        ("cuEventCreate", "hipEventCreateWithFlags"),
        ("cuEventRecord", "hipEventRecord"),
        ("cuEventRecordWithFlags", "hipEventRecordWithFlags"),
        ("cuEventQuery", "hipEventQuery"),
        ("cuStreamWaitEvent", "hipStreamWaitEvent"),
        ("cuEventSynchronize", "hipEventSynchronize"),
        ("cuEventElapsedTime", "hipEventElapsedTime"),
        ("cuEventDestroy_v2", "hipEventDestroy"),
        ("cuModuleLoadData", "hipModuleLoadData"),
        ("cuModuleGetFunction", "hipModuleGetFunction"),
        ("cuLaunchKernel", "hipModuleLaunchKernel"),
        ("cuGetErrorString", "hipDrvGetErrorString"),
    ];

    static GET_ATTRIBUTE: OnceLock<usize> = OnceLock::new();

    /// cuDeviceGetAttribute by HIP's numbers (ROCm 7.1's hipDeviceAttribute_t) for the attributes the engine asks: the compute capability (75, 76: gfx11 11.x, gfx12 12.x),
    /// the multiprocessors (16: CUs) and the warp size (10: dev.rs refuses waves other than 32 lanes).
    unsafe extern "C" fn get_attribute(v: *mut c_int, attr: c_int, dev: c_int) -> c_int {
        let hip = match attr {
            75 => 23,
            76 => 61,
            16 => 63,
            10 => 87,
            _ => return 1, // (hipErrorInvalidValue: one the engine does not ask)
        };
        match GET_ATTRIBUTE.get() {
            // SAFETY: hipDeviceGetAttribute's address, of this signature.
            Some(&f) => std::mem::transmute::<usize, unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int>(f)(v, hip, dev),
            None => 1,
        }
    }

    unsafe extern "C" fn set_func_attribute(_f: *mut c_void, _attr: c_int, _v: c_int) -> c_int {
        0
    }

    /// The table by `sym` (HIP's name to its address; the adapters' where HIP's differ).
    ///
    /// # Safety
    /// `sym` gives the address of HIP's function of a name, with the signature HIP's headers give it, or null.
    unsafe fn table(sym: &dyn Fn(&str) -> *mut c_void) -> Result<Driver, String> {
        Driver::resolve_with(&|n| match n {
            "cuDeviceGetAttribute" if !sym("hipDeviceGetAttribute").is_null() => get_attribute as *mut c_void,
            "cuFuncSetAttribute" => set_func_attribute as *mut c_void,
            _ => NAMES.iter().find(|p| p.0 == n).map_or(std::ptr::null_mut(), |p| sym(p.1)),
        })
    }

    pub(super) fn driver() -> Result<Driver, String> {
        let mut why = Vec::new();
        for lib in LIBS {
            match dl::open(lib) {
                // SAFETY: the runtime's own functions by their names.
                Ok(h) => unsafe {
                    let _ = GET_ATTRIBUTE.set(dl::sym(h, "hipDeviceGetAttribute\0") as usize);
                    return table(&|n| dl::sym(h, &format!("{n}\0"))).map_err(|e| format!("{lib}: {e}"));
                },
                Err(e) => why.push(e),
            }
        }
        Err(format!("no HIP runtime ({})", why.join("; ")))
    }

    #[cfg(test)]
    mod tests {
        /// Every function of the driver table has its HIP name or an adapter (a name missing from NAMES would be a table that cannot be made).
        #[test]
        fn every_driver_function_has_a_hip_one() {
            let got = std::cell::RefCell::new(Vec::new());
            // SAFETY: nothing is called: every name is given the same non-null address.
            let t = unsafe {
                super::table(&|n| {
                    got.borrow_mut().push(n.to_string());
                    1usize as *mut std::ffi::c_void
                })
            };
            assert!(t.is_ok(), "{:?}", t.err());
            assert_eq!(got.borrow().len(), super::NAMES.len() + 1); // (and hipDeviceGetAttribute's, for its adapter)
        }
    }
}

/// The driver, loaded and initialized once.
pub fn driver() -> Result<&'static Driver> {
    static D: OnceLock<std::result::Result<Driver, (Kind, String)>> = OnceLock::new();
    D.get_or_init(|| {
        #[cfg(glyd_hip)]
        let d = hip::driver().map_err(|e| (Kind::NoDriver, e))?;
        #[cfg(not(glyd_hip))]
        let d = {
            let h = dl::open(CUDA_DRIVER).map_err(|e| (Kind::NoDriver, format!("no CUDA driver ({e})")))?;
            // SAFETY: libcuda's functions of these names (CUDA 11 on: the _v2 ones).
            unsafe { Driver::resolve(h) }.map_err(|e| (Kind::NoDriver, format!("{CUDA_DRIVER}: {e}")))?
        };
        // SAFETY: no pointers.
        match unsafe { (d.cuInit)(0) } {
            0 => Ok(d),
            s if cfg!(glyd_hip) => Err((Kind::of_status(s), format!("hipInit: HIP error {s} (100: no AMD GPU the runtime can use)"))),
            s => Err((Kind::of_status(s), format!("cuInit: CUDA error {s}"))),
        }
    })
    .as_ref()
    .map_err(|(k, e)| Error::of(*k, e.clone()))
}

/// A driver status as a `Result`, with the driver's text.
pub fn cu(call: &str, status: c_int) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    let mut text: *const c_char = std::ptr::null();
    // SAFETY: cuGetErrorString gives a static string.
    let text = unsafe {
        match driver() {
            Ok(d) if (d.cuGetErrorString)(status, &mut text) == 0 && !text.is_null() => CStr::from_ptr(text).to_string_lossy().into_owned(),
            _ => format!("CUDA error {status}"),
        }
    };
    Err(Error::of(Kind::of_status(status), format!("{call}: {text} ({status})")))
}

/// The C API version this binding is written for (glyd_gpu.h's GLYD_GPU_API_VERSION).
pub const API_VERSION: c_int = 20;

/// The name a library call goes by in an error: the function's own.
macro_rules! call {
    ($n:literal) => {
        $n
    };
}

table! {
    pub struct Api {
        fn glyd_gpu_api_version() -> c_int;
        fn glyd_gpu_cuda_version() -> c_int;
        fn glyd_gpu_error_string(c_int) -> *const c_char;
        fn glyd_gpu_gpu(*mut c_int) -> c_int;
        fn glyd_gpu_mma_way(i64, i64, i64, i64, *mut c_int, *mut i64) -> c_int;
        fn glyd_gpu_mma12_way(i64, i64, i64, i64, *mut c_int, *mut i64) -> c_int;
        fn glyd_gpu_mma12_linear_workspace(i64, i64, i64, i64, *mut usize) -> c_int;
        fn glyd_gpu_mma12_linear(*const u8, *const u32, *const i32, *const u32, i64, i64, *const u16, i64, *const u16, *mut u16, i64, *mut c_void, usize, *mut c_int, Stream) -> c_int;
        fn glyd_gpu_mma12_unpack(*const u8, *const u32, *const i32, *const u32, i64, i64, i64, *mut u16, i64, Stream) -> c_int;
        fn glyd_gpu_mma_linear_workspace(i64, i64, i64, i64, *mut usize) -> c_int;
        fn glyd_gpu_mma_linear(*const u8, *const u8, *const i32, *const u32, i64, i64, *const u16, i64, *const u16, *mut u16, i64, *mut c_void, usize, *mut c_int, Stream) -> c_int;
        fn glyd_gpu_mma_unpack(*const u8, *const u8, *const i32, *const u32, i64, i64, i64, *mut u16, i64, Stream) -> c_int;
        fn glyd_gpu_moe_way(*const i64, i64, i64, *mut i32, Stream) -> c_int;
        fn glyd_gpu_f8_size_workspace(i64, i64, *mut usize) -> c_int;
        fn glyd_gpu_f8_size(*const u8, i64, i64, i64, *mut u32, *mut c_void, usize, *mut i64, Stream) -> c_int;
        fn glyd_gpu_f8_pack(*const u8, i64, i64, i64, *const u32, *mut u8, *mut u32, i64, Stream) -> c_int;
        fn glyd_gpu_f8_unpack(*const u8, *const u32, *const u32, i64, i64, *mut u8, i64, Stream) -> c_int;
        fn glyd_gpu_f8_linear_workspace(i64, i64, i64, *mut usize) -> c_int;
        fn glyd_gpu_f8_linear(*const u8, *const u32, *const u32, i64, i64, i64, *const c_void, i64, *const u16, i64, i64, *mut u16, i64, *mut c_void, usize, *mut c_int, Stream) -> c_int;
        fn glyd_gpu_f8_moe_workspace(i64, i64, i64, i64, *mut usize) -> c_int;
        fn glyd_gpu_f8_moe(*const u8, *const u32, *const u32, i64, i64, i64, *const f32, i64, *const u16, i64, *const i32, i64, i64, i64, i64, *mut u16, i64, *mut c_void, usize, *mut c_int, Stream) -> c_int;
        fn glyd_gpu_f8h_size(*const u8, i64, i64, i64, *mut u32, *mut c_void, usize, *mut i64, Stream) -> c_int;
        fn glyd_gpu_f8h_pack(*const u8, i64, i64, i64, *const u32, *mut u8, *mut u32, i64, Stream) -> c_int;
        fn glyd_gpu_f8h_unpack(*const u8, *const u32, *const u32, i64, i64, *mut u8, i64, Stream) -> c_int;
        fn glyd_gpu_f8h_linear(*const u8, *const u32, *const u32, i64, i64, i64, *const c_void, i64, *const u16, i64, i64, *mut u16, i64, *mut c_void, usize, *mut c_int, Stream) -> c_int;
        fn glyd_gpu_f8h_moe(*const u8, *const u32, *const u32, i64, i64, i64, *const f32, i64, *const u16, i64, *const i32, i64, i64, i64, i64, *mut u16, i64, *mut c_void, usize, *mut c_int, Stream) -> c_int;
        fn glyd_gpu_f8h_plain(c_int) -> c_int;
        fn glyd_gpu_mma_moe_workspace(i64, i64, i64, i64, i64, i64, i64, *mut usize) -> c_int;
        fn glyd_gpu_mma_moe(*const u8, *const u8, *const i32, *const u32, i64, i64, i64, *const u16, i64, i64, i64, *const i32, i64, *const u16, *const c_void, i64, *const i64, *mut u16, *mut c_void, usize, *mut c_int, Stream) -> c_int;
        fn glyd_gpu_mma_moe_placed(*const u8, *const u8, *const i32, *const u32, i64, i64, i64, i64, *const u16, i64, i64, i64, *const i32, i64, *const u16, *const c_void, i64, *const i64, *mut u16, *mut c_void, usize, *mut c_int, Stream) -> c_int;
        fn glyd_gpu_mma12_moe_workspace(i64, i64, i64, i64, i64, i64, i64, *mut usize) -> c_int;
        fn glyd_gpu_mma12_moe(*const u8, *const u32, *const i32, *const u32, i64, i64, i64, *const u16, i64, i64, i64, *const i32, i64, *const u16, *const c_void, i64, *const i64, *mut u16, *mut c_void, usize, *mut c_int, Stream) -> c_int;
        fn glyd_gpu_kv_leaf_bytes(i64, i64, i64, *mut i64) -> c_int;
        fn glyd_gpu_kv_extra_bytes(i64, *mut i64) -> c_int;
        fn glyd_gpu_kv_second_bytes(i64, i64, *mut i64) -> c_int;
        fn glyd_gpu_kv_write(*const u16, i64, *const u16, i64, *const i64, i64, *mut u8, i64, *const u8, *const u8, i64, i64, i64, *mut u8, i64, *mut u8, i64, i64, *mut u32, *mut u32, i64, i64, Stream) -> c_int;
        fn glyd_gpu_kv_release(*mut u8, i64, i64, *const i32, i64, i64, i64, i64, *mut u8, i64, *mut u8, i64, i64, *mut u32, *mut u32, i64, Stream) -> c_int;
        fn glyd_gpu_kv_unpack(*const u8, i64, *const i32, i64, *const u8, *const u8, i64, i64, i64, *const u8, i64, *const u8, i64, *mut u16, *mut u16, Stream) -> c_int;
        fn glyd_gpu_kv_gather(*const u8, i64, *const i32, *const i32, i64, *const u8, *const u8, i64, i64, i64, *const u8, i64, *const u8, i64, *mut u16, Stream) -> c_int;
        fn glyd_gpu_kv_work_put(*const u16, i64, *const u16, i64, *const i64, i64, i64, i64, *mut u16, i64, Stream) -> c_int;
        fn glyd_gpu_kv_decode_shares(i64, i64, i64, *mut i64) -> c_int;
        fn glyd_gpu_kv_decode_workspace(i64, i64, i64, i64, i64, *mut usize) -> c_int;
        fn glyd_gpu_kv_floor_set(c_int) -> c_int;
        fn glyd_gpu_kv_decode(*const u16, i64, *const u8, i64, *const u8, *const u8, *const i32, i64, *const i32, i64, i64, i64, i64, i64, i64, *const u8, i64, *const u8, i64, f64, *mut u16, i64, *mut c_void, usize, Stream) -> c_int;
        fn glyd_gpu_kv_decode_from(*const u16, i64, *const u8, i64, *const u8, *const u8, *const i32, i64, *const i32, *const i32, i64, i64, i64, i64, i64, i64, i64, *const u8, i64, *const u8, i64, f64, *mut u16, i64, *mut c_void, usize, Stream) -> c_int;
        fn glyd_gpu_kv_prefill(*const u16, i64, *const u16, i64, *const u16, i64, i64, i64, i64, *const u8, i64, *const u8, *const u8, *const i32, i64, i64, i64, i64, *const u8, i64, *const u8, i64, f64, i64, f64, *const f32, *mut u16, i64, Stream) -> c_int;
        fn glyd_gpu_attn_prefill(*const u16, i64, *const u16, *const u16, *const i32, i64, i64, i64, i64, i64, i64, i64, f64, i64, f64, *const f32, *mut u16, i64, Stream) -> c_int;
    }
}

table! {
    pub struct BlasApi {
        fn cublasCreate_v2(*mut *mut c_void) -> c_int;
        fn cublasDestroy_v2(*mut c_void) -> c_int;
        fn cublasSetStream_v2(*mut c_void, Stream) -> c_int;
        fn cublasGemmEx(*mut c_void, c_int, c_int, c_int, c_int, c_int, *const c_void, *const c_void, c_int, c_int, *const c_void, c_int, c_int, *const c_void, *mut c_void, c_int, c_int, c_int, c_int) -> c_int;
    }
}

/// NVIDIA's cuBLAS, where the machine has it (opened at run time, never shipped): the product of a matrix decoded to bf16 for a long prompt, as the vLLM plugin's route DECODE does.
pub struct Blas {
    api: BlasApi,
    handle: *mut c_void,
    pub path: String,
}

// SAFETY: a handle used by one thread at a time (the engine's); cuBLAS handles may move between threads.
unsafe impl Send for Blas {}

impl Blas {
    /// Whether `find` would find it, without making a handle.
    pub fn present() -> bool {
        Blas::candidates().iter().any(|t| dl::open(t).is_ok_and(|h| unsafe { BlasApi::resolve(h) }.is_ok()))
    }

    fn candidates() -> Vec<String> {
        let mut tries: Vec<String> = Vec::new();
        // (an AMD build has no cuBLAS: a decoded matrix's product is the library's or the engine's own)
        if cfg!(glyd_hip) || crate::tune!("GLYD_ENGINE_BLAS").is_some_and(|v| v == "off") {
            return tries;
        }
        if let Some(p) = crate::tune!("GLYD_CUBLAS") {
            tries.push(p);
        }
        let beside = std::env::current_exe().ok().and_then(|e| e.parent().map(|p| p.to_path_buf()));
        for n in CUBLAS {
            if let Some(b) = &beside {
                tries.push(b.join(n).to_string_lossy().into_owned());
            }
            tries.push(n.to_string());
        }
        tries
    }

    /// `$GLYD_CUBLAS` (a path), else libcublas.so.13 or .12 on the loader's path or beside the executable; None where it is not there (or GLYD_ENGINE_BLAS=off).
    pub fn find() -> Option<Blas> {
        for t in Blas::candidates() {
            let Ok(h) = dl::open(&t) else { continue };
            // SAFETY: cuBLAS's functions of these names have these signatures.
            let Ok(api) = (unsafe { BlasApi::resolve(h) }) else { continue };
            let mut handle = std::ptr::null_mut();
            // SAFETY: a host out-pointer.
            if unsafe { (api.cublasCreate_v2)(&mut handle) } != 0 {
                continue;
            }
            return Some(Blas { api, handle, path: t });
        }
        None
    }

    /// Y [m, o] (row-major bf16) = X [m, k] (row-major bf16) W^T, W [o, k] row-major bf16: bf16 in and out, fp32 sums.
    ///
    /// # Safety
    /// Device addresses of live buffers of those sizes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_nt(&self, s: Stream, w: u64, x: u64, y: u64, o: i32, k: i32, m: i32) -> Result<()> {
        let (alpha, beta) = (1.0f32, 0.0f32);
        let r = (self.api.cublasSetStream_v2)(self.handle, s);
        if r != 0 {
            return err(format!("cublasSetStream: status {r}"));
        }
        // column-major: Y^T [o, m] = W (k x o, transposed) X^T (k x m)
        let r = (self.api.cublasGemmEx)(
            self.handle,
            1,
            0,
            o,
            m,
            k,
            &alpha as *const f32 as *const c_void,
            w as *const c_void,
            14,
            k,
            x as *const c_void,
            14,
            k,
            &beta as *const f32 as *const c_void,
            y as *mut c_void,
            14,
            o,
            68,
            -1,
        );
        if r != 0 {
            return err(format!("cublasGemmEx: status {r}"));
        }
        Ok(())
    }
}

impl Drop for Blas {
    fn drop(&mut self) {
        // SAFETY: the handle cublasCreate made, destroyed once.
        unsafe { (self.api.cublasDestroy_v2)(self.handle) };
    }
}

/// A packed matrix W [rows, cols] as its buffers sit in device memory: the tiered `mma` layout (10.8 bits a weight) or the 12-bit `mma12` one.
#[derive(Clone, Copy, Debug)]
pub struct Pack {
    pub data: u64,
    pub words: Words,
    pub rows: i64,
    pub cols: i64,
}

/// What a layout has besides its data.
#[derive(Clone, Copy, Debug)]
pub enum Words {
    Tiered { blocks: u64, block_base: u64, tiers: [u32; 3] },
    Twelve { exc: u64, exc_base: u64, sym: [u32; 4] },
}

impl Pack {
    pub fn is_twelve(&self) -> bool {
        matches!(self.words, Words::Twelve { .. })
    }
}

/// `KvGeom::dim` with this set: a tensor held in the keys' tile layout alone (no V tile; glyd_gpu.h).
pub const KV_ONE_STREAM: i64 = 1 << 16;

/// The paged KV cache's geometry and pools, as the kernels take them (a layer's pool is passed per call).
#[derive(Clone, Copy, Debug)]
pub struct KvGeom {
    pub heads: i64,
    pub dim: i64,
    pub arena: i64,
    pub page_stride: i64,
    pub escape: u64,
    pub nchunks: i64,
    pub raw: u64,
    pub nslabs: i64,
}

/// The library, loaded (never unloaded) and of this binding's C API version.
pub struct Lib {
    api: Api,
    pub path: String,
}

impl Lib {
    /// The library at `path`.
    pub fn load(path: &str) -> Result<Lib> {
        let h = dl::open(path).map_err(Error)?;
        // the version first, by its name (every library exports it; 0.21.0's has none: 1)
        // SAFETY: glyd_gpu_api_version takes no argument in every version that has it.
        let v = unsafe {
            let f = dl::sym(h, "glyd_gpu_api_version\0");
            if f.is_null() {
                1
            } else {
                std::mem::transmute::<*mut c_void, unsafe extern "C" fn() -> c_int>(f)()
            }
        };
        if v != API_VERSION {
            return err(format!("{path}: C API version {v}, where this engine is written for {API_VERSION}: take the library of this engine's release"));
        }
        // the functions by their names
        // SAFETY: `h` is the library's handle; its functions have the signatures declared in `Api` at this version.
        let api = unsafe { Api::resolve(h) }.map_err(|e| Error(format!("{path}: {e}")))?;
        Ok(Lib { api, path: path.to_string() })
    }

    /// `$GLYD_GPU_LIB`, else libglyd_gpu_cuda13.so or _cuda12.so beside the executable, else the loader's search
    /// (cuda13 where the driver supports CUDA 13).
    pub fn find() -> Result<Lib> {
        if let Ok(p) = std::env::var("GLYD_GPU_LIB") {
            return Lib::load(&p);
        }
        let mut names = GPU_LIB.to_vec();
        if let Ok(d) = driver() {
            let mut v = 0;
            // SAFETY: a host out-pointer.
            if unsafe { (d.cuDriverGetVersion)(&mut v) } == 0 && v < 13000 {
                names.reverse();
            }
        }
        let beside = std::env::current_exe().ok().and_then(|e| e.parent().map(|p| p.to_path_buf()));
        let mut why = Vec::new();
        for n in names {
            let mut tries = Vec::new();
            if let Some(b) = &beside {
                tries.push(b.join(n).to_string_lossy().into_owned());
            }
            tries.push(n.to_string());
            for t in tries {
                match Lib::load(&t) {
                    Ok(l) => return Ok(l),
                    Err(e) => why.push(e.msg),
                }
            }
        }
        err(format!("Glyd's GPU library not found (set GLYD_GPU_LIB): {}", why.join("; ")))
    }

    pub fn cuda_version(&self) -> i32 {
        // SAFETY: no arguments.
        unsafe { (self.api.glyd_gpu_cuda_version)() }
    }

    /// The current device's class code (compute capability and class by name, glyd_gpu.h's routes).
    pub fn gpu(&self) -> Result<i32> {
        let mut g = 0;
        // SAFETY: a host out-pointer.
        let r = unsafe { (self.api.glyd_gpu_gpu)(&mut g) };
        self.ck(call!("gpu"), r)?;
        Ok(g)
    }

    fn ck(&self, call: &str, status: c_int) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        // SAFETY: the library's static text.
        let t = unsafe {
            let p = (self.api.glyd_gpu_error_string)(status);
            if p.is_null() {
                String::new()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };
        Err(Error::of(Kind::of_status(status), format!("{call}: {t} ({status})")))
    }

    /// The route the library takes W [o, k] for `m` tokens on `gpu` (glyd_gpu.h's GLYD_GPU_ROUTE_*), in the 12-bit layout or the tiered one.
    pub fn route(&self, gpu: i64, twelve: bool, o: i64, k: i64, m: i64) -> Result<i32> {
        let (mut r, mut last) = (0, 0i64);
        // SAFETY: host out-pointers.
        let s = unsafe {
            if twelve {
                (self.api.glyd_gpu_mma12_way)(gpu, o, k, m, &mut r, &mut last)
            } else {
                (self.api.glyd_gpu_mma_way)(gpu, o, k, m, &mut r, &mut last)
            }
        };
        self.ck(call!("way"), s)?;
        Ok(r)
    }

    /// The workspace bytes of Y [m, o] = X W^T by the current GPU's route for m.
    pub fn linear_workspace(&self, twelve: bool, o: i64, k: i64, m: i64) -> Result<usize> {
        let mut b = 0usize;
        // SAFETY: a host out-pointer.
        let r = unsafe {
            if twelve {
                (self.api.glyd_gpu_mma12_linear_workspace)(o, k, m, -1, &mut b)
            } else {
                (self.api.glyd_gpu_mma_linear_workspace)(o, k, m, -1, &mut b)
            }
        };
        self.ck(call!("linear_workspace"), r)?;
        Ok(b)
    }

    /// Y [m, rows] = X [m, cols] W^T on `cs` (route: the current GPU's for m). `done`: int32 counters, zero before the first call.
    ///
    /// # Safety
    /// Device addresses of live buffers of the sizes the C API asks for (glyd_gpu.h); `done` and the workspace serve one stream at a time.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn linear(&self, w: &Pack, x: u64, m: i64, y: u64, ws: u64, ws_bytes: usize, done: u64, cs: Stream) -> Result<()> {
        let r = match w.words {
            Words::Tiered { blocks, block_base, tiers } => (self.api.glyd_gpu_mma_linear)(
                w.data as *const u8,
                blocks as *const u8,
                block_base as *const i32,
                tiers.as_ptr(),
                w.rows,
                w.cols,
                x as *const u16,
                m,
                std::ptr::null(),
                y as *mut u16,
                -1,
                ws as *mut c_void,
                ws_bytes,
                done as *mut c_int,
                cs,
            ),
            Words::Twelve { exc, exc_base, sym } => (self.api.glyd_gpu_mma12_linear)(
                w.data as *const u8,
                exc as *const u32,
                exc_base as *const i32,
                sym.as_ptr(),
                w.rows,
                w.cols,
                x as *const u16,
                m,
                std::ptr::null(),
                y as *mut u16,
                -1,
                ws as *mut c_void,
                ws_bytes,
                done as *mut c_int,
                cs,
            ),
        };
        self.ck(call!("linear"), r)
    }

    /// The lossless FP8 format's size for W [o, k] of E4M3 bytes on the device (rows `ld` bytes apart): its heads into `heads` [patches x 8] and the words its variable part takes (glyd_gpu.h's f8 functions;
    /// waits for the stream). `h`: layout H (glyd_gpu.h's f8h functions, for HBM GPUs) here and in the calls below.
    ///
    /// # Safety
    /// Live device buffers of those sizes; `ws` holds `f8_size_workspace` bytes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn f8_size(&self, h: bool, w: u64, ld: i64, o: i64, k: i64, heads: u64, ws: u64, ws_bytes: usize, cs: Stream) -> Result<i64> {
        let mut words = 0i64;
        let f = if h { self.api.glyd_gpu_f8h_size } else { self.api.glyd_gpu_f8_size };
        let r = f(w as *const u8, ld, o, k, heads as *mut u32, ws as *mut c_void, ws_bytes, &mut words, cs);
        self.ck(call!("f8_size"), r)?;
        Ok(words)
    }

    /// Layout H's packs for the process: plain (Some(true)), coded (Some(false)) or the GPU's own (None: plain on an HBM GPU); before any pack.
    pub fn f8h_plain(&self, plain: Option<bool>) -> Result<()> {
        // SAFETY: a value argument.
        let r = unsafe { (self.api.glyd_gpu_f8h_plain)(plain.map_or(-1, |p| p as c_int)) };
        self.ck(call!("f8h_plain"), r)
    }

    pub fn f8_size_workspace(&self, o: i64, k: i64) -> Result<usize> {
        let mut b = 0usize;
        // SAFETY: a host out-pointer.
        let r = unsafe { (self.api.glyd_gpu_f8_size_workspace)(o, k, &mut b) };
        self.ck(call!("f8_size_workspace"), r)?;
        Ok(b)
    }

    /// W's patches into `core` and `var` from the heads `f8_size` made.
    ///
    /// # Safety
    /// Live device buffers of those sizes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn f8_pack(&self, h: bool, w: u64, ld: i64, o: i64, k: i64, heads: u64, core: u64, var: u64, var_words: i64, cs: Stream) -> Result<()> {
        let f = if h { self.api.glyd_gpu_f8h_pack } else { self.api.glyd_gpu_f8_pack };
        let r = f(w as *const u8, ld, o, k, heads as *const u32, core as *mut u8, var as *mut u32, var_words, cs);
        self.ck(call!("f8_pack"), r)
    }

    /// W back, bit for bit, into `out` [o, k] (rows `ld` bytes apart).
    ///
    /// # Safety
    /// Live device buffers of those sizes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn f8_unpack(&self, h: bool, core: u64, heads: u64, var: u64, o: i64, k: i64, out: u64, ld: i64, cs: Stream) -> Result<()> {
        let f = if h { self.api.glyd_gpu_f8h_unpack } else { self.api.glyd_gpu_f8_unpack };
        let r = f(core as *const u8, heads as *const u32, var as *const u32, o, k, out as *mut u8, ld, cs);
        self.ck(call!("f8_unpack"), r)
    }

    pub fn f8_linear_workspace(&self, n: i64, k: i64, m: i64) -> Result<usize> {
        let mut b = 0usize;
        // SAFETY: a host out-pointer.
        let r = unsafe { (self.api.glyd_gpu_f8_linear_workspace)(n, k, m, &mut b) };
        self.ck(call!("f8_linear_workspace"), r)?;
        Ok(b)
    }

    /// Y [m, n] = X [m, k] W^T for m of 1 to 64 and W in the lossless FP8 format with its scales (`kind`, `scale`, `lds`: glyd_gpu.h's f8_linear), X and Y bf16 with rows k and n elements apart.
    ///
    /// # Safety
    /// Live device buffers of those sizes; `done` ceil(n / 128) zeroed counters.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn f8_linear(
        &self,
        h: bool,
        core: u64,
        heads: u64,
        var: u64,
        n: i64,
        k: i64,
        kind: i64,
        scale: u64,
        lds: i64,
        x: u64,
        m: i64,
        y: u64,
        ws: u64,
        ws_bytes: usize,
        done: u64,
        cs: Stream,
    ) -> Result<()> {
        let f = if h { self.api.glyd_gpu_f8h_linear } else { self.api.glyd_gpu_f8_linear };
        let r = f(
            core as *const u8,
            heads as *const u32,
            var as *const u32,
            n,
            k,
            kind,
            scale as *const c_void,
            lds,
            x as *const u16,
            k,
            m,
            y as *mut u16,
            n,
            ws as *mut c_void,
            ws_bytes,
            done as *mut c_int,
            cs,
        );
        self.ck(call!("f8_linear"), r)
    }

    pub fn f8_moe_workspace(&self, n: i64, k: i64, m: i64, slots: i64) -> Result<usize> {
        let mut b = 0usize;
        // SAFETY: a host out-pointer.
        let r = unsafe { (self.api.glyd_gpu_f8_moe_workspace)(n, k, m, slots, &mut b) };
        self.ck(call!("f8_moe_workspace"), r)?;
        Ok(b)
    }

    /// The experts' product (glyd_gpu.h's f8_moe): `experts` matrices [n, k] packed as one [experts n, k] with their block scales, each pair of `plan` (moe_way's) times its expert's matrix into its row of
    /// y [pairs, n] in the plan's order; x's row of a pair its token's (`gather`: x [tokens, k], `k_` choices a token) or its own in the plan's order (x [pairs, k]). `m`: the most pairs an expert has (1 to 64),
    /// `slots`: the most experts the plan hits.
    ///
    /// # Safety
    /// Live device buffers of those sizes; `done` slots ceil(n / 128) zeroed counters.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn f8_moe(
        &self,
        h: bool,
        core: u64,
        heads: u64,
        var: u64,
        experts: i64,
        n: i64,
        k: i64,
        scale: u64,
        x: u64,
        plan: u64,
        k_: i64,
        gather: bool,
        m: i64,
        slots: i64,
        y: u64,
        ws: u64,
        ws_bytes: usize,
        done: u64,
        cs: Stream,
    ) -> Result<()> {
        let f = if h { self.api.glyd_gpu_f8h_moe } else { self.api.glyd_gpu_f8_moe };
        let r = f(
            core as *const u8,
            heads as *const u32,
            var as *const u32,
            experts,
            n,
            k,
            scale as *const f32,
            (k + 127) / 128,
            x as *const u16,
            k,
            plan as *const i32,
            k_,
            gather as i64,
            m,
            slots,
            y as *mut u16,
            n,
            ws as *mut c_void,
            ws_bytes,
            done as *mut c_int,
            cs,
        );
        self.ck(call!("f8_moe"), r)
    }

    /// Rows [row0, row0 + rows) of W back to bf16 into `out` [rows, cols] (both multiples of 64).
    ///
    /// # Safety
    /// As `linear`.
    pub unsafe fn unpack(&self, w: &Pack, row0: i64, rows: i64, out: u64, cs: Stream) -> Result<()> {
        let r = match w.words {
            Words::Tiered { blocks, block_base, tiers } => (self.api.glyd_gpu_mma_unpack)(
                w.data as *const u8,
                blocks as *const u8,
                block_base as *const i32,
                tiers.as_ptr(),
                w.cols,
                row0,
                rows,
                out as *mut u16,
                0,
                cs,
            ),
            Words::Twelve { exc, exc_base, sym } => (self.api.glyd_gpu_mma12_unpack)(
                w.data as *const u8,
                exc as *const u32,
                exc_base as *const i32,
                sym.as_ptr(),
                w.cols,
                row0,
                rows,
                out as *mut u16,
                0,
                cs,
            ),
        };
        self.ck(call!("unpack"), r)
    }

    /// The plan (int32 [2 + 2 experts + pairs]) of `pairs` pairs sorted by expert, `ids` (int64 [pairs]) each pair's expert (outside 0 to experts - 1: routed nowhere): the experts hit and where each one's
    /// pairs start in the order, then the pairs by expert, in order within each. A token's k choices are its pairs (pair j: token j / k, choice j mod k). Experts up to 12,288.
    ///
    /// # Safety
    /// As `linear`; `plan` holds 2 + 2 experts + pairs int32.
    pub unsafe fn moe_way(&self, ids: u64, pairs: i64, experts: i64, plan: u64, cs: Stream) -> Result<()> {
        let r = (self.api.glyd_gpu_moe_way)(ids as *const i64, pairs, experts, plan as *mut i32, cs);
        self.ck(call!("moe_way"), r)
    }

    /// The workspace bytes of `moe`'s product of `experts` experts' matrices [o, cols] for `t` tokens of `k` choices (act 0, 1 or 2; `weighted`: with the choices' weights), on the current GPU.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_workspace(&self, twelve: bool, experts: i64, o: i64, cols: i64, t: i64, k: i64, act: i64, weighted: bool) -> Result<usize> {
        let mut b = 0usize;
        // SAFETY: a host out-pointer.
        let r = unsafe {
            if twelve {
                (self.api.glyd_gpu_mma12_moe_workspace)(experts, o, cols, t, k, act, weighted as i64, &mut b)
            } else {
                (self.api.glyd_gpu_mma_moe_workspace)(experts, o, cols, t, k, act, weighted as i64, &mut b)
            }
        };
        self.ck(call!("moe_workspace"), r)?;
        Ok(b)
    }

    /// The experts' product over `w`, the experts' matrices [o, cols] stacked as one pack [experts o, cols] (`experts` of them), by the plan of `moe_way`: x [t, cols] (`gather`: a pair takes its token's row) or [t k, cols]
    /// (the pairs in the plan's order). `act` 0: y [t k, o] in the plan's order; 1 (SiLU) or 2 (GELU, tanh): y [t k, o / 2] = act(gate) up, the gate an expert's first o / 2 rows (o a multiple of 128).
    /// `weights` (fp32 [t, k], act 0; 0: none) with `ids` (int64 [t, k]): y [t, o], each token's k rows times their weights, added in fp32. `done`: int32 counters, zero before, left zero.
    ///
    /// # Safety
    /// As `linear`; the workspace is `moe_workspace`'s bytes; the counters are o / 64 (o / 128 with an act) times min(experts, t k).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn moe(
        &self,
        w: &Pack,
        experts: i64,
        x: u64,
        t: i64,
        k: i64,
        gather: bool,
        plan: u64,
        act: i64,
        weights: u64,
        ids: u64,
        y: u64,
        ws: u64,
        ws_bytes: usize,
        done: u64,
        cs: Stream,
    ) -> Result<()> {
        let o = w.rows / experts;
        let (wf, nul) = (weights as *const c_void, std::ptr::null());
        let r = match w.words {
            Words::Tiered { blocks, block_base, tiers } => (self.api.glyd_gpu_mma_moe)(
                w.data as *const u8,
                blocks as *const u8,
                block_base as *const i32,
                tiers.as_ptr(),
                experts,
                o,
                w.cols,
                x as *const u16,
                t,
                k,
                gather as i64,
                plan as *const i32,
                act,
                nul,
                wf,
                1,
                ids as *const i64,
                y as *mut u16,
                ws as *mut c_void,
                ws_bytes,
                done as *mut c_int,
                cs,
            ),
            Words::Twelve { exc, exc_base, sym } => (self.api.glyd_gpu_mma12_moe)(
                w.data as *const u8,
                exc as *const u32,
                exc_base as *const i32,
                sym.as_ptr(),
                experts,
                o,
                w.cols,
                x as *const u16,
                t,
                k,
                gather as i64,
                plan as *const i32,
                act,
                nul,
                wf,
                1,
                ids as *const i64,
                y as *mut u16,
                ws as *mut c_void,
                ws_bytes,
                done as *mut c_int,
                cs,
            ),
        };
        self.ck(call!("moe"), r)
    }

    /// `moe` over an expert cache's places of a tiered-layout stack (glyd_gpu_mma_moe_placed: each place its own block entries, `stride` bytes of blocks a place); the weights, ids and the rest as `moe`'s.
    ///
    /// # Safety
    /// As `moe`; `w`'s block entries [places][steps an expert + 1] and blocks [places][stride].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn moe_placed(
        &self,
        w: &Pack,
        stride: u64,
        experts: i64,
        x: u64,
        t: i64,
        k: i64,
        gather: bool,
        plan: u64,
        act: i64,
        weights: u64,
        ids: u64,
        y: u64,
        ws: u64,
        ws_bytes: usize,
        done: u64,
        cs: Stream,
    ) -> Result<()> {
        let Words::Tiered { blocks, block_base, tiers } = w.words else {
            return err("an expert cache's places of a 12-bit pack: the tiered layout alone is placed");
        };
        let r = (self.api.glyd_gpu_mma_moe_placed)(
            w.data as *const u8,
            blocks as *const u8,
            block_base as *const i32,
            tiers.as_ptr(),
            stride as i64,
            experts,
            w.rows / experts,
            w.cols,
            x as *const u16,
            t,
            k,
            gather as i64,
            plan as *const i32,
            act,
            std::ptr::null(),
            weights as *const c_void,
            1,
            ids as *const i64,
            y as *mut u16,
            ws as *mut c_void,
            ws_bytes,
            done as *mut c_int,
            cs,
        );
        self.ck(call!("moe_placed"), r)
    }

    /// A page's bytes for `h` heads of `d` channels with an arena of `a` entries.
    pub fn kv_page_bytes(&self, h: i64, d: i64, a: i64) -> Result<i64> {
        let mut b = 0;
        // SAFETY: a host out-pointer.
        let r = unsafe { (self.api.glyd_gpu_kv_leaf_bytes)(h, d, a, &mut b) };
        self.ck(call!("kv_leaf_bytes"), r)?;
        Ok(b)
    }

    pub fn kv_escape_bytes(&self, nchunks: i64) -> Result<i64> {
        let mut b = 0;
        // SAFETY: a host out-pointer.
        let r = unsafe { (self.api.glyd_gpu_kv_extra_bytes)(nchunks, &mut b) };
        self.ck(call!("kv_extra_bytes"), r)?;
        Ok(b)
    }

    pub fn kv_raw_bytes(&self, nslabs: i64, h: i64) -> Result<i64> {
        let mut b = 0;
        // SAFETY: a host out-pointer.
        let r = unsafe { (self.api.glyd_gpu_kv_second_bytes)(nslabs, h, &mut b) };
        self.ck(call!("kv_second_bytes"), r)?;
        Ok(b)
    }

    /// `n` tokens' keys and values [n, H, D] (token t's row at key + t * key_stride elements) into the layer's pool at `slots` [n] (int64).
    ///
    /// # Safety
    /// As `linear`; `status` is uint32 [16 + 2 ring] zeroed once; `lost` is uint32 words of `nlost` bits (two to a block), zeroed once, or 0 and 0.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn kv_write(
        &self,
        g: &KvGeom,
        pool: u64,
        kbase: u64,
        vbase: u64,
        layer: i64,
        key: u64,
        key_stride: i64,
        value: u64,
        value_stride: i64,
        slots: u64,
        n: i64,
        status: u64,
        lost: u64,
        nlost: i64,
        ring: i64,
        cs: Stream,
    ) -> Result<()> {
        let r = (self.api.glyd_gpu_kv_write)(
            key as *const u16,
            key_stride,
            value as *const u16,
            value_stride,
            slots as *const i64,
            n,
            pool as *mut u8,
            g.page_stride,
            kbase as *const u8,
            vbase as *const u8,
            g.heads,
            g.dim,
            g.arena,
            g.escape as *mut u8,
            g.nchunks,
            g.raw as *mut u8,
            g.nslabs,
            layer,
            status as *mut u32,
            lost as *mut u32,
            nlost,
            ring,
            cs,
        );
        self.ck(call!("kv_write"), r)
    }

    /// The pages `pages` [n] (int32, block numbers) of the layer's pool decoded into two bf16 tensors, keys `out_k` and values `out_v`, each [n, 16, H, D] (page j's tokens at j * 16 on: NHD, as a paged attention reads a cache of separate keys and values).
    ///
    /// # Safety
    /// As `kv_write`; `pages` is an int32 array of n on the device, `out_k` and `out_v` hold n pages of 16 * H * D values each.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn kv_unpack(&self, g: &KvGeom, pool: u64, kbase: u64, vbase: u64, pages: u64, n: i64, out_k: u64, out_v: u64, cs: Stream) -> Result<()> {
        let r = (self.api.glyd_gpu_kv_unpack)(
            pool as *const u8,
            g.page_stride,
            pages as *const i32,
            n,
            kbase as *const u8,
            vbase as *const u8,
            g.heads,
            g.dim,
            g.arena,
            g.escape as *const u8,
            g.nchunks,
            g.raw as *const u8,
            g.nslabs,
            out_k as *mut u16,
            out_v as *mut u16,
            cs,
        );
        self.ck(call!("kv_unpack"), r)
    }

    /// The pages `pages` [n] (int32, block numbers) of the layer's pool decoded into the bf16 staging cache `out` [blocks, 16, H, 2 D], page j into block `dst`[j].
    ///
    /// # Safety
    /// As `kv_write`; `pages` and `dst` are int32 arrays of n on the device, `out` holds a block for every dst.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn kv_gather(&self, g: &KvGeom, pool: u64, kbase: u64, vbase: u64, pages: u64, dst: u64, n: i64, out: u64, cs: Stream) -> Result<()> {
        let r = (self.api.glyd_gpu_kv_gather)(
            pool as *const u8,
            g.page_stride,
            pages as *const i32,
            dst as *const i32,
            n,
            kbase as *const u8,
            vbase as *const u8,
            g.heads,
            g.dim,
            g.arena,
            g.escape as *const u8,
            g.nchunks,
            g.raw as *const u8,
            g.nslabs,
            out as *mut u16,
            cs,
        );
        self.ck(call!("kv_gather"), r)
    }

    /// The most CTAs a (sequence, head) is split over, for `b` sequences of at most `max_pages` pages.
    pub fn kv_decode_splits(&self, b: i64, h: i64, max_pages: i64) -> Result<i64> {
        let mut s = 0;
        // SAFETY: a host out-pointer.
        let r = unsafe { (self.api.glyd_gpu_kv_decode_shares)(b, h, max_pages, &mut s) };
        self.ck(call!("kv_decode_shares"), r)?;
        Ok(s)
    }

    /// The decode attention's least CTAs a sequence (1 to 64; 0: the GPU's own), for the whole process.
    pub fn kv_floor_set(&self, floor: i32) -> Result<()> {
        // SAFETY: a plain value.
        let r = unsafe { (self.api.glyd_gpu_kv_floor_set)(floor) };
        self.ck(call!("kv_floor_set"), r)
    }

    pub fn kv_decode_workspace(&self, b: i64, h: i64, g: i64, d: i64, splits: i64) -> Result<usize> {
        let mut n = 0usize;
        // SAFETY: a host out-pointer.
        let r = unsafe { (self.api.glyd_gpu_kv_decode_workspace)(b, h, g, d, splits, &mut n) };
        self.ck(call!("kv_decode_workspace"), r)?;
        Ok(n)
    }

    /// Attention for `b` sequences: query q [H G, D] at q + i * q_stride, `seq_lens` [b] (int32) tokens in the pages of
    /// `block_table` (row i at + i * bt_stride, int32), out [H G, D] at out + i * out_stride.
    ///
    /// # Safety
    /// As `kv_write`; the workspace is zero when made and left so.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn kv_decode(
        &self,
        g: &KvGeom,
        group: i64,
        pool: u64,
        kbase: u64,
        vbase: u64,
        q: u64,
        q_stride: i64,
        block_table: u64,
        bt_stride: i64,
        seq_lens: u64,
        b: i64,
        splits: i64,
        scale: f64,
        out: u64,
        out_stride: i64,
        ws: u64,
        ws_bytes: usize,
        cs: Stream,
    ) -> Result<()> {
        let r = (self.api.glyd_gpu_kv_decode)(
            q as *const u16,
            q_stride,
            pool as *const u8,
            g.page_stride,
            kbase as *const u8,
            vbase as *const u8,
            block_table as *const i32,
            bt_stride,
            seq_lens as *const i32,
            b,
            splits,
            g.heads,
            group,
            g.dim,
            g.arena,
            g.escape as *const u8,
            g.nchunks,
            g.raw as *const u8,
            g.nslabs,
            scale,
            out as *mut u16,
            out_stride,
            ws as *mut c_void,
            ws_bytes,
            cs,
        );
        self.ck(call!("kv_decode"), r)
    }

    /// `kv_decode` over each sequence's tokens from `first` (int32 [b], a sliding window's start; 0: none, the whole sequence) to its length; `round`: each
    /// score the bf16 of its dot product (a bf16 matmul's, Gemma 4's attention).
    ///
    /// # Safety
    /// As `kv_decode`; `first`, where not 0, holds b int32.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn kv_decode_from(
        &self,
        g: &KvGeom,
        group: i64,
        pool: u64,
        kbase: u64,
        vbase: u64,
        q: u64,
        q_stride: i64,
        block_table: u64,
        bt_stride: i64,
        seq_lens: u64,
        first: u64,
        round: bool,
        b: i64,
        splits: i64,
        scale: f64,
        out: u64,
        out_stride: i64,
        ws: u64,
        ws_bytes: usize,
        cs: Stream,
    ) -> Result<()> {
        let r = (self.api.glyd_gpu_kv_decode_from)(
            q as *const u16,
            q_stride,
            pool as *const u8,
            g.page_stride,
            kbase as *const u8,
            vbase as *const u8,
            block_table as *const i32,
            bt_stride,
            seq_lens as *const i32,
            first as *const i32,
            round as i64,
            b,
            splits,
            g.heads,
            group,
            g.dim,
            g.arena,
            g.escape as *const u8,
            g.nchunks,
            g.raw as *const u8,
            g.nslabs,
            scale,
            out as *mut u16,
            out_stride,
            ws as *mut c_void,
            ws_bytes,
            cs,
        );
        self.ck(call!("kv_decode_from"), r)
    }

    /// A prompt chunk's attention (glyd_gpu.h's `glyd_gpu_kv_prefill`): `n` rows of one sequence at positions `start` .. `start + n`, queries `q` [n, H G, D] (row t at q + t
    /// q_stride elements), causal over the keys from each row's window (`window` tokens, 0: none) to itself: those before `start` read from the pages `block_table` (int32: the
    /// sequence's row of the table) names, the chunk's own from its bf16 rows `key`, `value` [n, H, D] (0: from the pages too, written before); `round`: each score the bf16 of its dot
    /// product (Gemma 4's), `softcap` 0: none, `sinks` 0 or float [H G] on the device; out [n, H G, D].
    ///
    /// # Safety
    /// As `kv_decode`; `block_table` holds a page for every token before `start + n` that a row's window takes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn kv_prefill(
        &self,
        g: &KvGeom,
        group: i64,
        pool: u64,
        kbase: u64,
        vbase: u64,
        block_table: u64,
        q: u64,
        q_stride: i64,
        rows: Option<(u64, i64, u64, i64)>,
        n: i64,
        start: i64,
        window: i64,
        scale: f64,
        round: bool,
        softcap: f64,
        sinks: u64,
        out: u64,
        out_stride: i64,
        cs: Stream,
    ) -> Result<()> {
        let (key, key_stride, value, value_stride) = rows.unwrap_or((0, 0, 0, 0));
        let r = (self.api.glyd_gpu_kv_prefill)(
            q as *const u16,
            q_stride,
            key as *const u16,
            key_stride,
            value as *const u16,
            value_stride,
            n,
            start,
            window,
            pool as *const u8,
            g.page_stride,
            kbase as *const u8,
            vbase as *const u8,
            block_table as *const i32,
            g.heads,
            group,
            g.dim,
            g.arena,
            g.escape as *const u8,
            g.nchunks,
            g.raw as *const u8,
            g.nslabs,
            scale,
            round as i64,
            softcap,
            sinks as *const f32,
            out as *mut u16,
            out_stride,
            cs,
        );
        self.ck(call!("kv_prefill"), r)
    }

    /// `kv_prefill`'s attention with the keys and values in a bf16 cache `kc`, `vc` [places, H, D] (glyd_gpu.h's `glyd_gpu_attn_prefill`): position p's place
    /// `block_table[p / 16] * 16 + p % 16` (0: p modulo `cap`), the chunk's own written there before the call; `g` query heads a kv head. The same kernel and sums:
    /// where the cache holds what the pages hold, `kv_prefill`'s bits.
    ///
    /// # Safety
    /// Live device buffers: q and out [n, H g, D] (rows `q_stride`, `out_stride` apart), the cache's every place a row reads, the table's blocks.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attn_prefill(
        &self,
        q: u64,
        q_stride: i64,
        (kc, vc, block_table, cap): (u64, u64, u64, i64),
        (n, start, window): (i64, i64, i64),
        (h, g, d): (i64, i64, i64),
        (scale, round, softcap): (f64, bool, f64),
        out: u64,
        out_stride: i64,
        cs: Stream,
    ) -> Result<()> {
        let r = (self.api.glyd_gpu_attn_prefill)(
            q as *const u16,
            q_stride,
            kc as *const u16,
            vc as *const u16,
            block_table as *const i32,
            cap,
            n,
            start,
            window,
            h,
            g,
            d,
            scale,
            round as i64,
            softcap,
            std::ptr::null(),
            out as *mut u16,
            out_stride,
            cs,
        );
        self.ck(call!("attn_prefill"), r)
    }
}

/// A C string for a driver call.
pub fn cstr(s: &str) -> CString {
    CString::new(s).expect("no NUL")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A C parameter's or result's type as `Api` declares it, without spaces: `const uint8_t* data` *constu8,
    /// `const uint32_t tiers[3]` *constu32, `int* done` *mutc_int, `cudaStream_t cs` Stream.
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
        let (base, ptr2) = match base.strip_suffix('*') {
            Some(b) => (b, true), // a pointer to a pointer
            None => (base, false),
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
            "glyd_gpu_ring" | "glyd_gpu_blas" => "c_void", // (opaque here)
            other => panic!("a C type this test does not know: {other}"),
        };
        let base = if ptr2 { format!("*mut {base}") } else { base.to_string() };
        let ty = match (ptr, konst) {
            (true, true) => format!("*const {base}"),
            (true, false) => format!("*mut {base}"),
            _ => base,
        };
        ty.replace(' ', "")
    }

    /// Every function `Api` declares is in gpu/glyd_gpu.h with the same arguments and result, and the header's
    /// GLYD_GPU_API_VERSION is `API_VERSION` (the header is the library's C API: a signature changed there fails here).
    #[test]
    fn declarations_are_the_header() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../gpu/glyd_gpu.h");
        let h = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let mut text = String::new();
        let mut rest = h.as_str();
        while let Some(i) = rest.find("/*") {
            text.push_str(&rest[..i]);
            rest = &rest[i + rest[i..].find("*/").expect("a comment of the header is not closed") + 2..];
        }
        text.push_str(rest);
        let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
        let mut header = Vec::new();
        for part in text.split(';') {
            let Some(at) = part.find("glyd_gpu_") else { continue };
            let Some(open) = part[at..].find('(').map(|i| at + i) else { continue };
            let ret = part[..at].trim();
            if !(ret == "int" || ret.ends_with(" int") || ret.ends_with("const char*")) {
                continue;
            }
            let ret = if ret.ends_with("const char*") { "*constc_char".to_string() } else { "c_int".to_string() };
            let args = part[open + 1..part.rfind(')').unwrap()].trim();
            let args: Vec<String> = if args == "void" { vec![] } else { args.split(',').map(rust_type).collect() };
            header.push((part[at..open].to_string(), args, ret));
        }
        assert!(Api::DECLARED.len() > 40, "Api::DECLARED lists {} functions", Api::DECLARED.len());
        for (name, args, ret) in Api::DECLARED {
            let ours = (name.to_string(), args.iter().map(|a| a.replace(' ', "")).collect::<Vec<_>>(), ret.replace(' ', ""));
            match header.iter().find(|d| d.0 == *name) {
                None => panic!("{name} is declared in ffi.rs, not in gpu/glyd_gpu.h"),
                Some(d) => assert_eq!(*d, ours, "{name}: the header's declaration (left) is not ffi.rs's (right)"),
            }
        }
        let version = h
            .split("#define GLYD_GPU_API_VERSION ")
            .nth(1)
            .and_then(|v| v.split_whitespace().next())
            .and_then(|v| v.parse::<c_int>().ok())
            .expect("glyd_gpu.h has no GLYD_GPU_API_VERSION");
        assert_eq!(version, API_VERSION, "GLYD_GPU_API_VERSION is not API_VERSION");
    }

    /// A status of the driver or the library is a kind: 2 the GPU's memory, the statuses after which a context is lost sticky, the others a call that failed and no more. `cu` gives the call,
    /// the text and the status in the words and the kind beside them.
    #[test]
    fn a_status_is_a_kind() {
        assert_eq!(Kind::of_status(2), Kind::OutOfMemory);
        for s in [4, 100, 214, 700, 702, 708, 709, 710, 715, 716, 717, 718, 719, 999] {
            assert_eq!(Kind::of_status(s), Kind::Device { status: s, sticky: true }, "{s}");
        }
        for s in [1, 3, 101, 200, 209, 218, 222, 400, 600, 10000] {
            assert_eq!(Kind::of_status(s), Kind::Device { status: s, sticky: false }, "{s}");
        }
        assert!(cu("cuMemAlloc", 0).is_ok());
        let e = cu("cuMemAlloc", 2).unwrap_err();
        assert!(e.kind == Kind::OutOfMemory && e.out_of_memory() && !e.sticky() && e.status() == Some(2), "{e:?}");
        assert!(e.msg.starts_with("cuMemAlloc: ") && e.msg.ends_with(" (2)"), "{e}");
        let e = cu("cuLaunchKernel", 700).unwrap_err();
        assert!(e.sticky() && e.status() == Some(700) && !e.out_of_memory(), "{e:?}");
        let e = cu("cuModuleLoadData", 209).unwrap_err();
        assert!(!e.sticky() && e.status() == Some(209), "{e:?}");
    }

    /// An error made of another's words keeps its kind, through as many wraps as there are and through `context`; one of other words is of no kind; and an error that comes as words alone
    /// (another thread's, another process's) still tells its status by the "(N)" its words end in.
    #[test]
    fn an_error_made_of_anothers_words_keeps_its_kind() {
        let e = cu("cuLaunchKernel", 719).unwrap_err();
        let once = Error(format!("layer 3: {e}"));
        let twice = Error(format!("the step of 4 rows failed ({once})"));
        assert!(once.sticky() && twice.sticky() && twice.status() == Some(719), "{twice:?}");
        assert!(twice.msg.contains("layer 3: cuLaunchKernel"));
        let other = Error("the prompt is longer than the context".to_string());
        assert!(other.kind == Kind::Other && !other.sticky() && other.status().is_none());
        // (a later error of no kind does not take the earlier one's: its words are its own)
        assert_eq!(err::<()>("no file of that name").unwrap_err().kind, Kind::Other);
        let oom = cu("cuMemAlloc", 2).unwrap_err().context("the KV cache");
        assert!(oom.kind == Kind::OutOfMemory && oom.msg.starts_with("the KV cache: cuMemAlloc"), "{oom:?}");
        // (the wrap of the last one made, not of an older one that the words do not hold)
        assert_eq!(Error(format!("loading: {oom}")).kind, Kind::OutOfMemory);
        // words alone, on a thread that made no error of a kind
        let words = twice.msg.clone();
        let far = std::thread::spawn(move || Error(format!("the server said: {words}"))).join().unwrap();
        assert!(far.kind == Kind::Other && !far.sticky(), "a kind does not cross threads as words");
        let words = "cuMemAlloc: out of memory (2)".to_string();
        let far = std::thread::spawn(move || Error(words)).join().unwrap();
        assert!(far.kind == Kind::Other && far.status() == Some(2) && far.out_of_memory());
    }
}
