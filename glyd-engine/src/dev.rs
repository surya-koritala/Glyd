//! A GPU as the engine uses it: its primary context made current on this thread, one non-blocking stream, device
//! memory with an owner, the engine's own kernels (the fatbin build.rs made) and CUDA graphs. One `Dev` a thread.

use crate::ffi::{cu, driver, err, CuEvent, CuFunction, CuGraph, CuGraphExec, CuModule, Result, Stream};
use std::ffi::{c_char, c_int, c_void};
use std::sync::atomic::{AtomicI32, AtomicU64, AtomicUsize, Ordering::Relaxed};

#[cfg(not(no_kernels))]
static FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/engine.fatbin"));
#[cfg(no_kernels)]
static FATBIN: &[u8] = &[];
/// The kernels built a second time by CUDA 12's nvcc (build.rs, NVCC12: not compressed, so that a driver of CUDA 12.2 reads them); empty where there is none.
#[cfg(not(no_kernels))]
static FATBIN12: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/engine12.fatbin"));
#[cfg(no_kernels)]
static FATBIN12: &[u8] = &[];

/// `Dev::wddm` from what is read: `windows` the operating system, `tcc` the TCC attribute (1: the TCC mode, which a datacenter card on Windows can be put in; 0: a display driver, WDDM).
pub(crate) fn wddm_from(windows: bool, tcc: Option<i32>) -> bool {
    windows && tcc == Some(0)
}

/// The seconds GPU discovery may take (`GLYD_PROBE_TIMEOUT`: a number, 0 for no limit): 30 on Windows, where a first call into the driver can stall for the antivirus or a sleeping GPU,
/// and no limit elsewhere.
pub(crate) fn probe_timeout(env: Option<&str>, windows: bool) -> u64 {
    env.and_then(|v| v.trim().parse().ok()).unwrap_or(if windows { 30 } else { 0 })
}

/// `f` run on a thread of its own, waited for `secs` seconds: its result, or None where it is still running (the thread is left to finish; it holds nothing the caller needs).
pub(crate) fn within<T: Send + 'static>(secs: u64, f: impl FnOnce() -> T + Send + 'static) -> Option<T> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(std::time::Duration::from_secs(secs)).ok()
}

/// The fatbin a driver of CUDA version `driver` (13000 for 13.0) loads: CUDA 12's where the build has one and the driver reports CUDA 12 (or `force12`: GLYD_ENGINE_CUDA=12, to run it on a
/// newer driver, which reads it as well), else the main one.
fn pick_fatbin<'a>(driver: i32, force12: bool, main: &'a [u8], cuda12: &'a [u8]) -> &'a [u8] {
    if !cuda12.is_empty() && (driver < 13000 || force12) {
        cuda12
    } else {
        main
    }
}

/// A GPU as the driver reports it, before an engine is built on it.
#[derive(Clone, Debug)]
pub struct Gpu {
    pub name: String,
    pub cc: (i32, i32),
    pub sms: i32,
    pub total: u64,
    pub free: u64,
    /// The CUDA version the driver supports (13000 for 13.0).
    pub driver: i32,
    /// How many GPUs the driver shows (CUDA_VISIBLE_DEVICES applied), and this one's place among them.
    pub count: i32,
    pub ordinal: i32,
}

pub struct Dev {
    pub ordinal: i32,
    pub name: String,
    pub cc: (i32, i32),
    pub sms: i32,
    pub total_mem: usize,
    pub stream: Stream,
    module: CuModule,
    /// The context the Dev's memory, stream and kernels are in (the GPU's primary one, or its own: `new_own`).
    ctx: *mut c_void,
}

impl Dev {
    /// The most shared memory a block may ask for on this GPU once its kernel opts in (`Kernel::allow_smem`): CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
    /// (an AMD build: RDNA's 64 KiB).
    pub fn smem_optin(&self) -> Result<u32> {
        if cfg!(glyd_hip) {
            return Ok(64 << 10);
        }
        let d = driver()?;
        let (mut dev, mut v) = (0, 0);
        // SAFETY: host out-pointers; 97 is the attribute's number.
        unsafe {
            cu("cuDeviceGet", (d.cuDeviceGet)(&mut dev, self.ordinal))?;
            cu("cuDeviceGetAttribute", (d.cuDeviceGetAttribute)(&mut v, 97, dev))?;
        }
        Ok(v as u32)
    }

    /// Whether GPU `ordinal` runs under Windows' WDDM (the display driver model: the memory it has is the card's and, past that, system memory the driver pages in without an error): on Windows, a
    /// GPU that is not in the TCC mode (CU_DEVICE_ATTRIBUTE_TCC_DRIVER, 17). False elsewhere and where the driver does not say.
    pub fn wddm(ordinal: i32) -> bool {
        let Ok(d) = driver() else { return false };
        let (mut dev, mut tcc) = (0, -1);
        // SAFETY: host out-pointers; 17 is the TCC attribute's number.
        let ok = unsafe { (d.cuDeviceGet)(&mut dev, ordinal) == 0 && (d.cuDeviceGetAttribute)(&mut tcc, 17, dev) == 0 };
        wddm_from(cfg!(windows), ok.then_some(tcc))
    }

    /// The compute capability as a pair (major, minor).
    pub fn cc_pair(&self) -> (i32, i32) {
        self.cc
    }

    /// The GPU `ordinal`: its name, compute capability, memory (total and free now) and the driver's CUDA version.
    pub fn probe(ordinal: i32) -> Result<Gpu> {
        let d = driver()?;
        let (mut count, mut version) = (0, 0);
        // SAFETY: host out-pointers.
        unsafe {
            cu("cuDeviceGetCount", (d.cuDeviceGetCount)(&mut count))?;
            cu("cuDriverGetVersion", (d.cuDriverGetVersion)(&mut version))?;
        }
        if count == 0 || ordinal >= count {
            return err(format!("no GPU {ordinal}: the driver sees {count}"));
        }
        let (mut dev, mut ctx) = (0, std::ptr::null_mut());
        // SAFETY: host out-pointers; the primary context stays retained for the process, as `new` does.
        unsafe {
            cu("cuDeviceGet", (d.cuDeviceGet)(&mut dev, ordinal))?;
            cu("cuDevicePrimaryCtxRetain", (d.cuDevicePrimaryCtxRetain)(&mut ctx, dev))?;
            cu("cuCtxSetCurrent", (d.cuCtxSetCurrent)(ctx))?;
        }
        let mut b = [0 as c_char; 256];
        let (mut major, mut minor, mut sms) = (0, 0, 0);
        let (mut total, mut free) = (0usize, 0usize);
        // SAFETY: host buffers of the sizes passed; 75 and 76 are the compute capability's attributes, 16 the SM count.
        unsafe {
            cu("cuDeviceGetName", (d.cuDeviceGetName)(b.as_mut_ptr(), b.len() as c_int, dev))?;
            cu("cuDeviceGetAttribute", (d.cuDeviceGetAttribute)(&mut major, 75, dev))?;
            cu("cuDeviceGetAttribute", (d.cuDeviceGetAttribute)(&mut minor, 76, dev))?;
            cu("cuDeviceGetAttribute", (d.cuDeviceGetAttribute)(&mut sms, 16, dev))?;
            cu("cuMemGetInfo", (d.cuMemGetInfo_v2)(&mut free, &mut total))?;
        }
        // SAFETY: NUL-terminated within the buffer.
        let name = unsafe { std::ffi::CStr::from_ptr(b.as_ptr()) }.to_string_lossy().into_owned();
        Ok(Gpu { name, cc: (major, minor), sms, total: total as u64, free: free as u64, driver: version, count, ordinal })
    }

    /// Every GPU the driver shows, as `probe` reports it (one that cannot be read is left out; where none can, the first one's error). Each holds a context of its own after this, which `release` lets go
    /// of for the GPUs not used.
    pub fn probe_all() -> Result<Vec<Gpu>> {
        let secs = probe_timeout(std::env::var("GLYD_PROBE_TIMEOUT").ok().as_deref(), cfg!(windows));
        if secs == 0 {
            return Dev::probe_all_now();
        }
        // (a driver that stalls: Windows' first cuInit while the antivirus scans nvcuda.dll or the GPU wakes from a power state: the caller says so instead of hanging)
        within(secs, Dev::probe_all_now).unwrap_or_else(|| {
            err(format!(
                "the NVIDIA driver did not answer in {secs} s (the first call into it, cuInit, or reading the GPU): wait for the antivirus or the GPU to settle and run this again, or GLYD_PROBE_TIMEOUT=0 waits as long as it takes"
            ))
        })
    }

    fn probe_all_now() -> Result<Vec<Gpu>> {
        let d = driver()?;
        let mut count = 0;
        // SAFETY: a host out-pointer.
        unsafe { cu("cuDeviceGetCount", (d.cuDeviceGetCount)(&mut count))? };
        let all: Vec<Gpu> = (0..count).filter_map(|i| Dev::probe(i).ok()).collect();
        if all.is_empty() {
            return Dev::probe(0).map(|g| vec![g]);
        }
        // (a GPU under WDDM pages what does not fit into system memory without an error: the plans keep more of it free)
        crate::plan::set_wddm(all.iter().any(|g| Dev::wddm(g.ordinal)));
        Ok(all)
    }

    /// The context `probe` made on GPU `ordinal` let go (the memory it held is the GPU's again).
    pub fn release(ordinal: i32) {
        if let Ok(d) = driver() {
            let mut dev = 0;
            // SAFETY: a host out-pointer; a context `probe` retained, released once.
            unsafe {
                if (d.cuDeviceGet)(&mut dev, ordinal) == 0 {
                    (d.cuDevicePrimaryCtxRelease_v2)(dev);
                }
            }
        }
    }

    pub fn new(ordinal: i32) -> Result<Dev> {
        Dev::open(ordinal, false)
    }

    /// The GPU in a context of its own, not its primary one (what a second GPU is to the engine, where there is one GPU: the check of a model split over GPUs, `--gpus 0,0`). Current on this thread after it.
    pub fn new_own(ordinal: i32) -> Result<Dev> {
        Dev::open(ordinal, true)
    }

    /// This Dev's context made current on this thread: its work and the memory made for it go to its GPU (a model split over GPUs runs each GPU's part in turn on one thread).
    pub fn make_current(&self) -> Result<()> {
        // SAFETY: a context this Dev retained or made, alive for the process.
        unsafe { cu("cuCtxSetCurrent", (driver()?.cuCtxSetCurrent)(self.ctx)) }
    }

    /// This Dev's GPU may read and write GPU `peer`'s memory directly from now on (peer access); already on is no error. The error names the two where the GPUs cannot
    /// reach each other (`can_reach`). This Dev's context is current on the thread after it.
    pub fn reach_peer(&self, peer: i32) -> Result<()> {
        if !can_reach(self.ordinal, peer) {
            return err(format!("GPU {} cannot reach GPU {peer} directly (no peer access)", self.ordinal));
        }
        self.make_current()?;
        let d = driver()?;
        let (mut pd, mut ctx) = (0, std::ptr::null_mut());
        // SAFETY: host out-pointers; the peer's primary context is retained for the process, as `Dev::new` does; 704: access already on.
        unsafe {
            cu("cuDeviceGet", (d.cuDeviceGet)(&mut pd, peer))?;
            cu("cuDevicePrimaryCtxRetain", (d.cuDevicePrimaryCtxRetain)(&mut ctx, pd))?;
            let r = (d.cuCtxEnablePeerAccess)(ctx, 0);
            if r != 0 && r != 704 {
                cu("cuCtxEnablePeerAccess", r)?;
            }
        }
        self.make_current()
    }

    fn open(ordinal: i32, own: bool) -> Result<Dev> {
        let d = driver()?;
        let (mut dev, mut ctx) = (0, std::ptr::null_mut());
        // SAFETY: host out-pointers; the primary context (or the one made) stays for the process; cuCtxCreate makes its context current.
        unsafe {
            cu("cuDeviceGet", (d.cuDeviceGet)(&mut dev, ordinal))?;
            if own {
                cu("cuCtxCreate", (d.cuCtxCreate_v2)(&mut ctx, 0, dev))?;
            } else {
                cu("cuDevicePrimaryCtxRetain", (d.cuDevicePrimaryCtxRetain)(&mut ctx, dev))?;
            }
            cu("cuCtxSetCurrent", (d.cuCtxSetCurrent)(ctx))?;
        }
        // (the context's counts named by its GPU)
        if let Some(c) = count(ctx as usize) {
            c.gpu.store(ordinal, Relaxed);
        }
        let mut b = [0 as c_char; 256];
        let (mut major, mut minor, mut sms) = (0, 0, 0);
        let mut total = 0usize;
        // SAFETY: host buffers of the sizes passed; 75 and 76 are the compute capability's attributes, 16 the SM count.
        unsafe {
            cu("cuDeviceGetName", (d.cuDeviceGetName)(b.as_mut_ptr(), b.len() as c_int, dev))?;
            cu("cuDeviceGetAttribute", (d.cuDeviceGetAttribute)(&mut major, 75, dev))?;
            cu("cuDeviceGetAttribute", (d.cuDeviceGetAttribute)(&mut minor, 76, dev))?;
            cu("cuDeviceGetAttribute", (d.cuDeviceGetAttribute)(&mut sms, 16, dev))?;
            cu("cuDeviceTotalMem", (d.cuDeviceTotalMem_v2)(&mut total, dev))?;
        }
        // SAFETY: NUL-terminated within the buffer.
        let name = unsafe { std::ffi::CStr::from_ptr(b.as_ptr()) }.to_string_lossy().into_owned();
        #[cfg(glyd_hip)]
        {
            // (an AMD build: the kernels are written for 32-lane warps, RDNA 3 and 4 in wave32; a GPU of 64-lane waves, CDNA or GCN, is not one they run on)
            let mut lanes = 0;
            // SAFETY: a host out-pointer; 10 is the warp size's attribute.
            unsafe { cu("cuDeviceGetAttribute", (d.cuDeviceGetAttribute)(&mut lanes, 10, dev))? };
            if lanes != 32 {
                return err(format!("{name}: waves of {lanes} lanes, where the engine's AMD kernels are written for 32 (RDNA 3 and 4)"));
            }
        }
        let mut s = Stream::DEFAULT;
        // SAFETY: a host out-pointer; 1: CU_STREAM_NON_BLOCKING.
        unsafe { cu("cuStreamCreate", (d.cuStreamCreate)(&mut s, 1))? };
        let mut module = std::ptr::null_mut();
        let mut version = 0;
        // SAFETY: a host out-pointer.
        unsafe { cu("cuDriverGetVersion", (d.cuDriverGetVersion)(&mut version))? };
        let force12 = std::env::var("GLYD_ENGINE_CUDA").is_ok_and(|v| v.trim() == "12");
        let fatbin = pick_fatbin(version, force12, FATBIN, FATBIN12);
        if !fatbin.is_empty() {
            // SAFETY: the fatbin nvcc made, which lives for the process.
            unsafe { cu("cuModuleLoadData", (d.cuModuleLoadData)(&mut module, fatbin.as_ptr() as *const c_void))? };
        }
        Ok(Dev { ordinal, name, cc: (major, minor), sms, total_mem: total, stream: s, module, ctx })
    }

    /// A kernel of kernels/engine.cu by name.
    /// A kernel of the engine's by its type (`crate::kt`), which says its arguments' types.
    pub fn typed<T: Typed>(&self) -> Result<T> {
        self.kernel(T::NAME).map(T::of)
    }

    /// The kernel of type `T`, held by its arguments' types alone (`ByArgs`).
    pub fn by_args<T: Typed>(&self) -> Result<ByArgs<T::Args>> {
        self.kernel(T::NAME).map(|k| ByArgs(k, std::marker::PhantomData))
    }

    pub fn kernel(&self, name: &str) -> Result<Kernel> {
        if self.module.is_null() {
            return err("this build has no CUDA kernels (it was built without nvcc: set NVCC or CUDA_HOME and build again)");
        }
        let c = crate::ffi::cstr(name);
        let mut f = std::ptr::null_mut();
        // SAFETY: a host out-pointer and a NUL-terminated name.
        unsafe { cu(name, (driver()?.cuModuleGetFunction)(&mut f, self.module, c.as_ptr()))? };
        kprof_table()
            .lock()
            .map_err(|_| crate::ffi::Error("the kernel census is poisoned".into()))?
            .entry(f as usize)
            .or_insert_with(|| (name.to_string(), 0, 0.0));
        Ok(Kernel(f))
    }

    pub fn sync(&self) -> Result<()> {
        // SAFETY: a stream this Dev made.
        unsafe { cu("cuStreamSynchronize", (driver()?.cuStreamSynchronize)(self.stream)) }
    }

    /// (free, total) device memory in bytes.
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        let (mut f, mut t) = (0usize, 0usize);
        // SAFETY: host out-pointers.
        unsafe { cu("cuMemGetInfo", (driver()?.cuMemGetInfo_v2)(&mut f, &mut t))? };
        Ok((f, t))
    }

    /// The work queued on the stream captured into a graph: `body` runs with the stream capturing, nothing in it executes.
    pub fn capture(&self, body: impl FnOnce() -> Result<()>) -> Result<Graph> {
        let d = driver()?;
        // SAFETY: the stream is this Dev's; 2: CU_STREAM_CAPTURE_MODE_RELAXED.
        unsafe { cu("cuStreamBeginCapture", (d.cuStreamBeginCapture_v2)(self.stream, 2))? };
        let r = body();
        let mut g: CuGraph = std::ptr::null_mut();
        // SAFETY: ends the capture begun above, whether the body worked or not.
        let e = unsafe { (d.cuStreamEndCapture)(self.stream, &mut g) };
        r?;
        cu("cuStreamEndCapture", e)?;
        let mut x: CuGraphExec = std::ptr::null_mut();
        // SAFETY: a captured graph.
        unsafe { cu("cuGraphInstantiate", (d.cuGraphInstantiateWithFlags)(&mut x, g, 0))? };
        Ok(Graph { graph: g, exec: x })
    }
}

/// A kernel of the engine's module.
#[derive(Clone, Copy)]
pub struct Kernel(CuFunction);

impl Kernel {
    /// # Safety
    /// `args` point to values of the kernel's parameter types, in order, live for the call; device addresses are of live buffers.
    pub unsafe fn launch(&self, grid: [u32; 3], block: [u32; 3], stream: Stream, args: &mut [*mut c_void]) -> Result<()> {
        self.launch_smem(grid, block, 0, stream, args)
    }

    /// `launch` with `smem` bytes of dynamic shared memory (more than 48 KiB: `allow_smem` first).
    ///
    /// # Safety
    /// As `launch`.
    pub unsafe fn launch_smem(&self, grid: [u32; 3], block: [u32; 3], smem: u32, stream: Stream, args: &mut [*mut c_void]) -> Result<()> {
        let mut go = || {
            let r = cu(
                "cuLaunchKernel",
                (driver()?.cuLaunchKernel)(
                    self.0,
                    grid[0],
                    grid[1],
                    grid[2],
                    block[0],
                    block[1],
                    block[2],
                    smem,
                    stream,
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
            );
            // (a development build names the kernel and its launch where it fails)
            if cfg!(feature = "dev") {
                if let Err(e) = r {
                    let name = kprof_table().lock().ok().and_then(|t| t.get(&(self.0 as usize)).map(|v| v.0.clone())).unwrap_or_default();
                    return Err(crate::ffi::Error(format!("{} ({name}: grid {grid:?}, block {block:?}, smem {smem})", e.msg)));
                }
            }
            r
        };
        if !KPROF.load(std::sync::atomic::Ordering::Relaxed) {
            return go();
        }
        // (the census: the launch timed alone on its stream)
        KTIMER.with(|t| -> Result<()> {
            if t.get().is_none() {
                let _ = t.set(Timer::new()?);
            }
            let t = t.get().expect("a timer");
            t.start(stream)?;
            go()?;
            let ms = t.stop(stream)?;
            if let Some(x) = kprof_table().lock().ok().as_mut().and_then(|m| m.get_mut(&(self.0 as usize))) {
                x.1 += 1;
                x.2 += ms as f64 * 1e3;
            }
            Ok(())
        })
    }

    /// The most dynamic shared memory a launch of this kernel may ask for (CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES).
    pub fn allow_smem(&self, bytes: u32) -> Result<()> {
        // SAFETY: a function of the loaded module and a plain integer.
        cu("cuFuncSetAttribute", unsafe { (driver()?.cuFuncSetAttribute)(self.0, 8, bytes as c_int) })
    }
}

/// A kernel's arguments as the driver takes them: pointers to values of the kernel's parameter types, in order.
pub trait KernelArgs {
    fn ptrs(&mut self) -> Vec<*mut c_void>;
}

impl KernelArgs for () {
    fn ptrs(&mut self) -> Vec<*mut c_void> {
        Vec::new()
    }
}

/// A kernel of the engine's with its arguments' types (`crate::kt`, made by build.rs from kernels/*.cu): `launch!` with it takes exactly those types, in order, or does not
/// compile. `Dev::typed` loads one.
pub trait Typed {
    const NAME: &'static str;
    type Args: KernelArgs;
    fn of(k: Kernel) -> Self;
    fn kernel(&self) -> &Kernel;
}

/// What `launch!` launches with arguments `A`: a typed kernel for its own arguments' types only, an untyped `Kernel` for any (as before the types).
pub trait Launch<A> {
    fn kernel_of(&self) -> &Kernel;
}

impl<A: KernelArgs> Launch<A> for Kernel {
    fn kernel_of(&self) -> &Kernel {
        self
    }
}

impl<A: KernelArgs> Launch<A> for &Kernel {
    fn kernel_of(&self) -> &Kernel {
        self
    }
}

// (a typed kernel's own `Launch`, for it and for a reference to it, is in `crate::kt`: one impl a kernel, so that the arguments' types follow from it)

/// A typed kernel seen by its arguments' types alone (`crate::kt`'s `sig`): what a choice between kernels that take the same arguments is (`g4_prep256` or `g4_prep512`
/// by the head's width), still launched with those types only.
pub struct Sig<'a, A>(&'a Kernel, std::marker::PhantomData<fn(A)>);

impl<A> Clone for Sig<'_, A> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<A> Copy for Sig<'_, A> {}

impl<'a, A> Sig<'a, A> {
    pub(crate) fn of(k: &'a Kernel) -> Self {
        Sig(k, std::marker::PhantomData)
    }
}

impl<A: KernelArgs> Launch<A> for Sig<'_, A> {
    fn kernel_of(&self) -> &Kernel {
        self.0
    }
}

/// The arguments' types of a typed kernel (`crate::kt::g4_rmsnorm`'s `(u64, u64, u64, i32, f32)`).
pub type ArgsOf<T> = <T as Typed>::Args;

/// A typed kernel held by its arguments' types alone (`Dev::by_args`): what an array of kernels that take the same arguments holds (`moe_select8`, `16`, `32`); an
/// array of kernels that take others does not compile.
pub struct ByArgs<A>(Kernel, std::marker::PhantomData<fn(A)>);

impl<A> Clone for ByArgs<A> {
    fn clone(&self) -> Self {
        ByArgs(self.0, std::marker::PhantomData)
    }
}

impl<A: KernelArgs> Launch<A> for ByArgs<A> {
    fn kernel_of(&self) -> &Kernel {
        &self.0
    }
}

impl<A: KernelArgs> Launch<A> for &ByArgs<A> {
    fn kernel_of(&self) -> &Kernel {
        &self.0
    }
}

impl<A> std::ops::Deref for ByArgs<A> {
    type Target = Kernel;
    fn deref(&self) -> &Kernel {
        &self.0
    }
}

/// The kernel `l` is, where `A` are arguments it takes (`launch!`'s check of their types).
pub fn kernel_for<'a, A: KernelArgs, L: Launch<A>>(l: &'a L, _args: &A) -> &'a Kernel {
    l.kernel_of()
}

macro_rules! impl_args {
    ($($t:ident $i:tt),+) => {
        impl<$($t),+> KernelArgs for ($($t,)+) {
            fn ptrs(&mut self) -> Vec<*mut c_void> {
                vec![$(&mut self.$i as *mut $t as *mut c_void),+]
            }
        }
    };
}
impl_args!(A 0);
impl_args!(A 0, B 1);
impl_args!(A 0, B 1, C 2);
impl_args!(A 0, B 1, C 2, D 3);
impl_args!(A 0, B 1, C 2, D 3, E 4);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12, N 13);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12, N 13, O 14);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12, N 13, O 14, P 15);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12, N 13, O 14, P 15, Q 16);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12, N 13, O 14, P 15, Q 16, R 17);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12, N 13, O 14, P 15, Q 16, R 17, S 18);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12, N 13, O 14, P 15, Q 16, R 17, S 18, T 19);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12, N 13, O 14, P 15, Q 16, R 17, S 18, T 19, U 20);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12, N 13, O 14, P 15, Q 16, R 17, S 18, T 19, U 20, V 21);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12, N 13, O 14, P 15, Q 16, R 17, S 18, T 19, U 20, V 21, W 22);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12, N 13, O 14, P 15, Q 16, R 17, S 18, T 19, U 20, V 21, W 22, X 23);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12, N 13, O 14, P 15, Q 16, R 17, S 18, T 19, U 20, V 21, W 22, X 23, Y 24);
impl_args!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12, N 13, O 14, P 15, Q 16, R 17, S 18, T 19, U 20, V 21, W 22, X 23, Y 24, Z 25);

/// `launch!(kernel, [gx, gy, gz], [bx, by, bz], stream; a, b, c)`: the arguments are held in a tuple of their own types
/// for the call (each exactly the kernel's parameter type: `0i32`, not `0`).
#[macro_export]
macro_rules! launch {
    ($k:expr, $grid:expr, $block:expr, $s:expr; $($a:expr),+ $(,)?) => {{
        let mut vals = ($($a,)+);
        let held = &$k;
        let k = $crate::dev::kernel_for(held, &vals);
        let mut ptrs = $crate::dev::KernelArgs::ptrs(&mut vals);
        k.launch($grid, $block, $s, &mut ptrs)
    }};
    ($k:expr, $grid:expr, $block:expr, smem $smem:expr, $s:expr; $($a:expr),+ $(,)?) => {{
        let mut vals = ($($a,)+);
        let held = &$k;
        let k = $crate::dev::kernel_for(held, &vals);
        let mut ptrs = $crate::dev::KernelArgs::ptrs(&mut vals);
        k.launch_smem($grid, $block, $smem, $s, &mut ptrs)
    }};
}

/// Whether kernels take the shapes that fit 64 KiB of shared memory a block: an AMD build's (RDNA 3 and 4 have no more), or any GPU's in a development
/// build with GLYD_SMEM64=1 (those shapes checked where no such GPU is at hand). Elsewhere every kernel keeps the shape it has on NVIDIA's GPUs.
pub fn smem64() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| cfg!(glyd_hip) || crate::tune!("GLYD_SMEM64").is_some_and(|v| v == "1"))
}

/// A type whose values are any of its bytes, with no padding: what may be copied to and from device memory as bytes (`Buf::read`, `Buf::write`, `read_dev`,
/// `Pinned::as_slice_mut`). A `bool`, a `char`, an enum or a struct with padding is not.
///
/// # Safety
/// Every bit pattern of `size_of::<Self>()` bytes is a value of the type, and the type has no padding bytes.
pub unsafe trait Plain: Copy + 'static {}

macro_rules! plain {
    ($($t:ty),*) => { $(
        // SAFETY: a primitive number: any bits are a value, and it has no padding.
        unsafe impl Plain for $t {}
    )* };
}
plain!(u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, usize, isize, f32, f64);
// SAFETY: an array of plain values has no padding between them, and its bits are its elements'.
unsafe impl<T: Plain, const N: usize> Plain for [T; N] {}

/// `out` read from device memory at `src` (synchronously: the caller has waited for the work that writes it).
///
/// # Safety
/// `src .. src + size_of_val(out)` is device memory the caller may read.
pub unsafe fn read_dev<T: Plain>(src: u64, out: &mut [T]) -> Result<()> {
    let n = std::mem::size_of_val(out);
    // SAFETY: n bytes of host memory from n of device memory the caller names (its contract); T takes any bytes.
    unsafe { cu("cuMemcpyDtoH", (driver()?.cuMemcpyDtoH_v2)(out.as_mut_ptr() as *mut c_void, src, n)) }
}

/// Device memory, freed when dropped. It belongs to the context it was made in (its GPU's, for a model split over GPUs or a second model of the process): what it
/// reads, writes and frees, it does in that context, whichever is current on the thread then. `ptr` and `bytes` are for kernels' arguments; what the buffer does is by
/// its own copies of them, so a buffer is made only here and changing the two does not change what it frees.
pub struct Buf {
    pub ptr: u64,
    pub bytes: usize,
    at: u64,
    len: usize,
    ctx: usize,
}

/// The context current on this thread (0: none).
fn current_ctx() -> Result<usize> {
    let mut c = std::ptr::null_mut();
    // SAFETY: a host out-pointer.
    unsafe { cu("cuCtxGetCurrent", (driver()?.cuCtxGetCurrent)(&mut c))? };
    Ok(c as usize)
}

/// `f` run with context `ctx` current (as it was again after; `ctx` 0: as it is).
fn in_ctx<R>(ctx: usize, f: impl FnOnce() -> Result<R>) -> Result<R> {
    let now = current_ctx()?;
    if ctx == 0 || ctx == now {
        return f();
    }
    // SAFETY: a context a Dev retained or made for the process (a buffer's), then the thread's own again.
    unsafe { cu("cuCtxSetCurrent", (driver()?.cuCtxSetCurrent)(ctx as *mut c_void))? };
    let r = f();
    // SAFETY: as above.
    unsafe { cu("cuCtxSetCurrent", (driver()?.cuCtxSetCurrent)(now as *mut c_void))? };
    r
}

/// The bytes the engine's buffers hold on the GPUs now, the most they have held since the program started and when (milliseconds since 1970): counted where `Buf` allocates and frees, so
/// what the CUDA context, the kernels' code, a graph and a library's own allocations (cuBLAS's) take is not in them. The process's, and each context's (`held_by_context`).
static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);
static PEAK_AT: AtomicU64 = AtomicU64::new(0);

/// The same counts by context: a slot per context a `Dev` opened or that has held a buffer, taken in order and kept for the process (its GPU's ordinal where a `Dev` opened it, else -1);
/// a context past the slots is in the process's counts alone.
struct Count {
    ctx: AtomicUsize,
    gpu: AtomicI32,
    live: AtomicU64,
    peak: AtomicU64,
    peak_at: AtomicU64,
}

const COUNTED: usize = 32;
static COUNTS: [Count; COUNTED] =
    [const { Count { ctx: AtomicUsize::new(0), gpu: AtomicI32::new(-1), live: AtomicU64::new(0), peak: AtomicU64::new(0), peak_at: AtomicU64::new(0) } };
        COUNTED];

/// Context `ctx`'s slot (taken where it has none: the slots taken are the first ones, so a context's slot comes before any free one).
fn count(ctx: usize) -> Option<&'static Count> {
    if ctx == 0 {
        return None;
    }
    COUNTS.iter().find(|c| match c.ctx.compare_exchange(0, ctx, Relaxed, Relaxed) {
        Ok(_) => true,
        Err(x) => x == ctx,
    })
}

fn now_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

/// `bytes` more held in context `ctx` (fewer where `free`), the process's and the context's peaks moved on.
fn held_add(ctx: usize, bytes: u64, free: bool) {
    let add = |live: &AtomicU64, peak: &AtomicU64, at: &AtomicU64| {
        if free {
            live.fetch_sub(bytes, Relaxed);
            return;
        }
        let now = live.fetch_add(bytes, Relaxed) + bytes;
        if peak.fetch_max(now, Relaxed) < now {
            at.store(now_ms(), Relaxed);
        }
    };
    add(&LIVE, &PEAK, &PEAK_AT);
    if let Some(c) = count(ctx) {
        add(&c.live, &c.peak, &c.peak_at);
    }
}

/// (now, peak, the peak's time in ms since 1970) of the engine's buffers, on every GPU together.
pub fn held_now() -> (u64, u64, u64) {
    (LIVE.load(Relaxed), PEAK.load(Relaxed), PEAK_AT.load(Relaxed))
}

/// What one context's buffers hold: its GPU (-1 where no `Dev` opened it), the bytes now, the most since the program started and when.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeldOn {
    pub gpu: i32,
    pub live: u64,
    pub peak: u64,
    pub peak_at: u64,
}

/// Each context's counts, in the order they were first seen: a GPU's primary context, and any context of its own a check made on it.
pub fn held_by_context() -> Vec<HeldOn> {
    COUNTS
        .iter()
        .take_while(|c| c.ctx.load(Relaxed) != 0)
        .map(|c| HeldOn { gpu: c.gpu.load(Relaxed), live: c.live.load(Relaxed), peak: c.peak.load(Relaxed), peak_at: c.peak_at.load(Relaxed) })
        .collect()
}

/// One context's bytes now (none where it has no slot).
#[cfg(feature = "dev")]
fn held_in(ctx: usize) -> Option<u64> {
    COUNTS.iter().take_while(|c| c.ctx.load(Relaxed) != 0).find(|c| c.ctx.load(Relaxed) == ctx).map(|c| c.live.load(Relaxed))
}

/// What a model holds on the GPU by what it is for, allocated once when it loads: its weights, the keys and values for its whole context, the states a sequence carries (Gated DeltaNet's), and
/// what they are held in. The rest of `held_now` is the engine's working memory.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Held {
    pub weights: u64,
    pub kv: u64,
    pub state: u64,
    pub kv_type: &'static str,
    pub state_type: &'static str,
}

impl Buf {
    /// `bytes` bytes of device memory in the context current on this thread.
    pub fn new(bytes: usize) -> Result<Buf> {
        let ctx = current_ctx()?;
        let mut p = 0u64;
        // SAFETY: a host out-pointer; at least one byte is taken.
        unsafe { cu("cuMemAlloc", (driver()?.cuMemAlloc_v2)(&mut p, bytes.max(1)))? };
        held_add(ctx, bytes as u64, false);
        Ok(Buf { ptr: p, bytes, at: p, len: bytes, ctx })
    }

    /// A buffer of no bytes that holds nothing (a stand-in where a kernel's argument is not read).
    pub fn none() -> Buf {
        Buf { ptr: 0, bytes: 0, at: 0, len: 0, ctx: 0 }
    }

    pub fn zeroed(bytes: usize) -> Result<Buf> {
        let b = Buf::new(bytes)?;
        b.zero()?;
        Ok(b)
    }

    /// `data` copied to the buffer at byte `offset`, complete when this returns.
    pub fn write<T: Plain>(&self, offset: usize, data: &[T]) -> Result<()> {
        let n = std::mem::size_of_val(data);
        if offset.checked_add(n).is_none_or(|end| end > self.len) {
            return err(format!("a write of {n} bytes at {offset}, past the buffer's {}", self.len));
        }
        in_ctx(self.ctx, || {
            // SAFETY: n bytes of host memory to n of this buffer's (checked above), in its context.
            unsafe { cu("cuMemcpyHtoD", (driver()?.cuMemcpyHtoD_v2)(self.at + offset as u64, data.as_ptr() as *const c_void, n))? };
            settle()
        })
    }

    /// The buffer's bytes at `offset` into `out` (synchronously; the work queued on the stream is waited for by the caller).
    pub fn read<T: Plain>(&self, offset: usize, out: &mut [T]) -> Result<()> {
        let n = std::mem::size_of_val(out);
        if offset.checked_add(n).is_none_or(|end| end > self.len) {
            return err(format!("a read of {n} bytes at {offset}, past the buffer's {}", self.len));
        }
        in_ctx(self.ctx, || {
            // SAFETY: n of this buffer's bytes (checked above) to n of host memory, in its context; T takes any bytes.
            unsafe { cu("cuMemcpyDtoH", (driver()?.cuMemcpyDtoH_v2)(out.as_mut_ptr() as *mut c_void, self.at + offset as u64, n)) }
        })
    }

    /// Every byte of the buffer set to `v`, queued on `s` (ordered with the work on it).
    pub fn fill_async(&self, v: u8, s: Stream) -> Result<()> {
        if self.len == 0 {
            return Ok(());
        }
        // SAFETY: the buffer's own bytes.
        unsafe { memset_async(self.at, v, self.len, s) }
    }

    /// The buffer's bytes set to zero, complete when this returns.
    pub fn zero(&self) -> Result<()> {
        if self.at == 0 {
            return Ok(());
        }
        in_ctx(self.ctx, || {
            // SAFETY: the buffer's own bytes (at least the one byte allocated), in its context.
            unsafe { cu("cuMemsetD8", (driver()?.cuMemsetD8_v2)(self.at, 0, self.len.max(1)))? };
            settle()
        })
    }
}

/// The legacy stream waited for. `Buf::write`'s copy from pageable memory returns once the data is staged, before it reaches the GPU, and `cuMemsetD8` returns at once; both run
/// on the legacy stream, which a kernel on a `Dev`'s stream (a non-blocking one) is not ordered after.
fn settle() -> Result<()> {
    // SAFETY: the null stream is the context's legacy stream (these functions are not the per-thread-stream variants).
    unsafe { cu("cuStreamSynchronize", (driver()?.cuStreamSynchronize)(Stream::DEFAULT)) }
}

impl Drop for Buf {
    fn drop(&mut self) {
        if self.at == 0 {
            return;
        }
        let free = || {
            // SAFETY: allocated by Buf::new, freed once.
            unsafe { cu("cuMemFree", (driver()?.cuMemFree_v2)(self.at)) }
        };
        // (in its context; where the thread's context cannot be read, as it is)
        let _ = match current_ctx() {
            Ok(_) => in_ctx(self.ctx, free),
            Err(_) => free(),
        };
        held_add(self.ctx, self.len as u64, true);
    }
}

/// Pinned host memory (for async copies), freed when dropped.
pub struct Pinned {
    pub ptr: *mut c_void,
    pub bytes: usize,
}

impl Pinned {
    pub fn new(bytes: usize) -> Result<Pinned> {
        let mut p = std::ptr::null_mut();
        // SAFETY: a host out-pointer; 0: no flags.
        unsafe { cu("cuMemHostAlloc", (driver()?.cuMemHostAlloc)(&mut p, bytes.max(1), 0))? };
        Ok(Pinned { ptr: p, bytes })
    }

    /// Pinned host memory mapped into the device's address space (CU_MEMHOSTALLOC_DEVICEMAP; with unified addressing a kernel writes it through the same pointer): what a kernel hands the host without a copy.
    pub fn mapped(bytes: usize) -> Result<Pinned> {
        let mut p = std::ptr::null_mut();
        // SAFETY: a host out-pointer; 2: CU_MEMHOSTALLOC_DEVICEMAP.
        unsafe { cu("cuMemHostAlloc", (driver()?.cuMemHostAlloc)(&mut p, bytes.max(1), 2))? };
        Ok(Pinned { ptr: p, bytes })
    }

    /// Pinned host memory every context takes as pinned (CU_MEMHOSTALLOC_PORTABLE): what two GPUs copy through.
    pub fn portable(bytes: usize) -> Result<Pinned> {
        let mut p = std::ptr::null_mut();
        // SAFETY: a host out-pointer; 1: CU_MEMHOSTALLOC_PORTABLE.
        unsafe { cu("cuMemHostAlloc", (driver()?.cuMemHostAlloc)(&mut p, bytes.max(1), 1))? };
        Ok(Pinned { ptr: p, bytes })
    }

    pub fn as_slice_mut<T: Plain>(&mut self) -> &mut [T] {
        assert!((self.ptr as usize).is_multiple_of(std::mem::align_of::<T>()), "pinned memory not aligned for the type");
        // SAFETY: the allocation holds bytes / size_of::<T>() elements, aligned (checked; the driver's are page-aligned), and T takes any bytes.
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut T, self.bytes / std::mem::size_of::<T>()) }
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        if let Ok(d) = driver() {
            // SAFETY: allocated by Pinned::new, freed once.
            unsafe { (d.cuMemFreeHost)(self.ptr) };
        }
    }
}

/// An async copy of plain host data (pinned, or pageable which the driver stages) to the device on `s`.
///
/// # Safety
/// `src` stays valid until the stream has run the copy.
pub unsafe fn copy_h2d_async(dst: u64, src: *const c_void, bytes: usize, s: Stream) -> Result<()> {
    cu("cuMemcpyHtoDAsync", (driver()?.cuMemcpyHtoDAsync_v2)(dst, src, bytes, s))
}

/// # Safety
/// `dst` stays valid until the stream has run the copy.
pub unsafe fn copy_d2h_async(dst: *mut c_void, src: u64, bytes: usize, s: Stream) -> Result<()> {
    cu("cuMemcpyDtoHAsync", (driver()?.cuMemcpyDtoHAsync_v2)(dst, src, bytes, s))
}

/// A device-to-device copy on `s`.
///
/// # Safety
/// Two live device ranges of `bytes` that do not overlap, valid until the stream has run the copy.
pub unsafe fn copy_d2d_async(dst: u64, src: u64, bytes: usize, s: Stream) -> Result<()> {
    cu("cuMemcpyDtoDAsync", (driver()?.cuMemcpyDtoDAsync_v2)(dst, src, bytes, s))
}

/// `bytes` bytes of device memory at `dst` set to `v`, queued on `s` (a whole buffer: `Buf::fill_async`).
///
/// # Safety
/// `dst .. dst + bytes` is device memory the caller may write, and nothing queued reads it before this runs unless it is to see the bytes set.
pub unsafe fn memset_async(dst: u64, v: u8, bytes: usize, s: Stream) -> Result<()> {
    // SAFETY: the caller's device range (its contract).
    unsafe { cu("cuMemsetD8Async", (driver()?.cuMemsetD8Async)(dst, v, bytes, s)) }
}

// A development census of the kernels' launches (`kprof_start`, `kprof_stop`): each launch timed alone on its stream, by kernel. Off, it costs a launch an atomic load.
static KPROF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
thread_local!(static KTIMER: std::cell::OnceCell<Timer> = const { std::cell::OnceCell::new() });

type KTable = std::sync::Mutex<std::collections::HashMap<usize, (String, u64, f64)>>;

fn kprof_table() -> &'static KTable {
    static T: std::sync::OnceLock<KTable> = std::sync::OnceLock::new();
    T.get_or_init(Default::default)
}

/// The census begun (counts zeroed); a graph may not be captured while it runs (its launches wait for the stream).
pub fn kprof_start() {
    if let Ok(mut m) = kprof_table().lock() {
        m.values_mut().for_each(|x| (x.1, x.2) = (0, 0.0));
    }
    KPROF.store(true, std::sync::atomic::Ordering::Relaxed);
}

pub fn kprof_on() -> bool {
    KPROF.load(std::sync::atomic::Ordering::Relaxed)
}

/// The census ended: (kernel, launches, microseconds in all), the most time first.
pub fn kprof_stop() -> Vec<(String, u64, f64)> {
    KPROF.store(false, std::sync::atomic::Ordering::Relaxed);
    let mut v: Vec<(String, u64, f64)> = kprof_table().lock().map(|m| m.values().filter(|x| x.1 > 0).cloned().collect()).unwrap_or_default();
    v.sort_by(|a, b| b.2.total_cmp(&a.2));
    v
}

/// An instantiated CUDA graph.
pub struct Graph {
    graph: CuGraph,
    exec: CuGraphExec,
}

impl Graph {
    /// The captured work queued again on `s`, with the addresses and arguments it was captured with.
    ///
    /// # Safety
    /// Every buffer, table and kernel the capture used is still alive and still holds what the captured work expects there (the graph holds none of them).
    pub unsafe fn launch(&self, s: Stream) -> Result<()> {
        // SAFETY: an instantiated graph; what it reads and writes is alive (the caller's contract).
        unsafe { cu("cuGraphLaunch", (driver()?.cuGraphLaunch)(self.exec, s)) }
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        if let Ok(d) = driver() {
            // SAFETY: made by capture, destroyed once.
            unsafe {
                (d.cuGraphExecDestroy)(self.exec);
                (d.cuGraphDestroy)(self.graph);
            }
        }
    }
}

/// A timing event pair on the stream.
pub struct Timer {
    a: CuEvent,
    b: CuEvent,
}

impl Timer {
    pub fn new() -> Result<Timer> {
        let d = driver()?;
        let (mut a, mut b) = (std::ptr::null_mut(), std::ptr::null_mut());
        // SAFETY: host out-pointers; 0: default flags.
        unsafe {
            cu("cuEventCreate", (d.cuEventCreate)(&mut a, 0))?;
            cu("cuEventCreate", (d.cuEventCreate)(&mut b, 0))?;
        }
        Ok(Timer { a, b })
    }

    pub fn start(&self, s: Stream) -> Result<()> {
        // SAFETY: an event of this timer.
        unsafe { cu("cuEventRecord", (driver()?.cuEventRecord)(self.a, s)) }
    }

    /// Milliseconds from `start` to now on the stream.
    pub fn stop(&self, s: Stream) -> Result<f32> {
        let d = driver()?;
        let mut ms = 0f32;
        // SAFETY: events of this timer; a host out-pointer.
        unsafe {
            cu("cuEventRecord", (d.cuEventRecord)(self.b, s))?;
            cu("cuEventSynchronize", (d.cuEventSynchronize)(self.b))?;
            cu("cuEventElapsedTime", (d.cuEventElapsedTime)(&mut ms, self.a, self.b))?;
        }
        Ok(ms)
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        if let Ok(d) = driver() {
            // SAFETY: made in new, destroyed once.
            unsafe {
                (d.cuEventDestroy_v2)(self.a);
                (d.cuEventDestroy_v2)(self.b);
            }
        }
    }
}

/// Whether GPU `a` can read and write GPU `b`'s memory directly (peer access; `Dev::reach_peer` turns it on). False where the driver cannot say.
pub fn can_reach(a: i32, b: i32) -> bool {
    let Ok(d) = driver() else { return false };
    let mut can = 0;
    // SAFETY: a host out-pointer.
    unsafe { (d.cuDeviceCanAccessPeer)(&mut can, a, b) == 0 && can != 0 }
}

/// The calling thread works on GPU `ordinal` from now on, in its primary context (the one `Dev::new` retains for the process): a thread of the engine's that copies
/// for it (the expert stores' copy threads).
pub fn bind_thread(ordinal: i32) -> Result<()> {
    let d = driver()?;
    let (mut dev, mut ctx) = (0, std::ptr::null_mut());
    // SAFETY: host out-pointers; the device's primary context, which the engine retains for the process.
    unsafe {
        cu("cuDeviceGet", (d.cuDeviceGet)(&mut dev, ordinal))?;
        cu("cuDevicePrimaryCtxRetain", (d.cuDevicePrimaryCtxRetain)(&mut ctx, dev))?;
        cu("cuCtxSetCurrent", (d.cuCtxSetCurrent)(ctx))
    }
}

/// Host memory `[ptr, ptr + bytes)` (page-aligned) registered with the driver, so that copies from it go at the link's speed without a pinned copy (`flags`: the driver's
/// CU_MEMHOSTREGISTER flags; 8 for a file's pages, read only).
///
/// # Safety
/// The range is the caller's, stays mapped until `unregister_host`, and is registered once.
pub unsafe fn register_host(ptr: *mut c_void, bytes: usize, flags: u32) -> Result<()> {
    // SAFETY: as the caller promises.
    unsafe { cu("cuMemHostRegister", (driver()?.cuMemHostRegister_v2)(ptr, bytes, flags)) }
}

/// The registration of `register_host` at `ptr` undone.
///
/// # Safety
/// `ptr` was registered, and no copy from it is still queued.
pub unsafe fn unregister_host(ptr: *mut c_void) -> Result<()> {
    // SAFETY: as the caller promises.
    unsafe { cu("cuMemHostUnregister", (driver()?.cuMemHostUnregister)(ptr)) }
}

/// An event: a timing one (`new`), or one that only orders work (`untimed`, which the driver records more cheaply).
pub struct Ev(CuEvent);

impl Ev {
    pub fn new() -> Result<Ev> {
        Ev::with_flags(0)
    }

    /// An event with no timing (CU_EVENT_DISABLE_TIMING): for ordering work between streams and threads.
    pub fn untimed() -> Result<Ev> {
        Ev::with_flags(2)
    }

    fn with_flags(flags: u32) -> Result<Ev> {
        let mut e = std::ptr::null_mut();
        // SAFETY: a host out-pointer.
        unsafe { cu("cuEventCreate", (driver()?.cuEventCreate)(&mut e, flags))? };
        Ok(Ev(e))
    }

    pub fn record(&self, s: Stream) -> Result<()> {
        // SAFETY: an event of this wrapper.
        unsafe { cu("cuEventRecord", (driver()?.cuEventRecord)(self.0, s)) }
    }

    /// The event recorded as a node of the graph being captured on `s` (CU_EVENT_RECORD_EXTERNAL), which a replay records again: its time can be read after the replay (outside a capture the driver refuses it: CUDA_ERROR_ILLEGAL_STATE).
    pub fn record_in_graph(&self, s: Stream) -> Result<()> {
        // SAFETY: an event of this wrapper; 1 is CU_EVENT_RECORD_EXTERNAL.
        unsafe { cu("cuEventRecordWithFlags", (driver()?.cuEventRecordWithFlags)(self.0, s, 1)) }
    }

    /// Work queued on `s` from now on waits for the work this event was recorded after (a no-op for an event not recorded yet).
    pub fn wait(&self, s: Stream) -> Result<()> {
        // SAFETY: an event of this wrapper and a stream of the caller's.
        unsafe { cu("cuStreamWaitEvent", (driver()?.cuStreamWaitEvent)(s, self.0, 0)) }
    }

    /// The calling thread waits until the work this event was recorded after has run (the engine's stream may have more queued after it).
    pub fn wait_host(&self) -> Result<()> {
        // SAFETY: an event of this wrapper.
        unsafe { cu("cuEventSynchronize", (driver()?.cuEventSynchronize)(self.0)) }
    }

    /// Whether the work this event was recorded after has run (an event not recorded yet has).
    pub fn done(&self) -> Result<bool> {
        // SAFETY: an event of this wrapper; 600 is CUDA_ERROR_NOT_READY.
        match unsafe { (driver()?.cuEventQuery)(self.0) } {
            0 => Ok(true),
            600 => Ok(false),
            e => cu("cuEventQuery", e).map(|_| false),
        }
    }

    /// Milliseconds from `earlier` to this event (both recorded and complete).
    pub fn since(&self, earlier: &Ev) -> Result<f32> {
        let mut ms = 0f32;
        // SAFETY: events of this wrapper; a host out-pointer.
        unsafe {
            cu("cuEventSynchronize", (driver()?.cuEventSynchronize)(self.0))?;
            cu("cuEventElapsedTime", (driver()?.cuEventElapsedTime)(&mut ms, earlier.0, self.0))?;
        }
        Ok(ms)
    }
}

/// A handle to an event that another thread may record or make a stream wait for. Made only by `Ev::as_ref`, whose caller promises that the event outlives every use.
#[derive(Clone, Copy)]
pub struct EvRef(usize);

impl EvRef {
    pub fn record(&self, s: Stream) -> Result<()> {
        // SAFETY: the handle of an event that is alive (the caller keeps it so).
        unsafe { cu("cuEventRecord", (driver()?.cuEventRecord)(self.0 as CuEvent, s)) }
    }

    pub fn wait(&self, s: Stream) -> Result<()> {
        // SAFETY: as above.
        unsafe { cu("cuStreamWaitEvent", (driver()?.cuStreamWaitEvent)(s, self.0 as CuEvent, 0)) }
    }

    /// The calling thread waits until the work the event was last recorded after has run (a no-op for an event not recorded yet).
    pub fn wait_host(&self) -> Result<()> {
        // SAFETY: as above.
        unsafe { cu("cuEventSynchronize", (driver()?.cuEventSynchronize)(self.0 as CuEvent)) }
    }
}

impl Ev {
    /// A handle for another thread to record the event with, or to wait on (see `EvRef`).
    ///
    /// # Safety
    /// The event outlives every use of the handle and of its copies.
    pub unsafe fn as_ref(&self) -> EvRef {
        EvRef(self.0 as usize)
    }
}

impl Drop for Ev {
    fn drop(&mut self) {
        if let Ok(d) = driver() {
            // SAFETY: made in new, destroyed once.
            unsafe { (d.cuEventDestroy_v2)(self.0) };
        }
    }
}

/// Streams that sides gave back, by context (R15, #236). A `Stream` is a plain handle that can be copied, so a side's stream is never destroyed: one that a side's
/// copy outlived would be a handle the driver may give to anything. A side's stream goes back here when the side is dropped, idle, and the next side of that context
/// takes it; the streams made are as many as the most sides alive at once.
struct Spare(Vec<(usize, usize)>);

impl Spare {
    /// A stream context `ctx`'s sides gave back, the latest first.
    fn take(&mut self, ctx: usize) -> Option<usize> {
        let i = self.0.iter().rposition(|&(c, _)| c == ctx)?;
        Some(self.0.remove(i).1)
    }

    fn give(&mut self, ctx: usize, stream: usize) {
        self.0.push((ctx, stream));
    }
}

static SPARE: std::sync::Mutex<Spare> = std::sync::Mutex::new(Spare(Vec::new()));

/// Streams sides made (never destroyed: `Spare`), for `bufcheck`.
static SIDES_MADE: AtomicU64 = AtomicU64::new(0);

/// A second stream on the engine's device (non-blocking, as the engine's own), for work that runs beside the engine's: the expert cache's copies. Ordered against the engine's stream by events only.
pub struct Side {
    pub stream: Stream,
    /// The context it is in, whose sides take it when this one is dropped.
    ctx: usize,
}

impl Side {
    /// A side in the thread's current context: one a dropped side of it gave back, else a new stream.
    pub fn new() -> Result<Side> {
        let ctx = current_ctx()?;
        if let Some(s) = SPARE.lock().unwrap_or_else(|e| e.into_inner()).take(ctx) {
            return Ok(Side { stream: Stream(s as *mut c_void), ctx });
        }
        let mut s = Stream::DEFAULT;
        // SAFETY: a host out-pointer; 1: CU_STREAM_NON_BLOCKING.
        unsafe { cu("cuStreamCreate", (driver()?.cuStreamCreate)(&mut s, 1))? };
        SIDES_MADE.fetch_add(1, Relaxed);
        Ok(Side { stream: s, ctx })
    }

    pub fn sync(&self) -> Result<()> {
        // SAFETY: a stream this wrapper made.
        unsafe { cu("cuStreamSynchronize", (driver()?.cuStreamSynchronize)(self.stream)) }
    }
}

impl Drop for Side {
    fn drop(&mut self) {
        // (its work finished, in its own context, before another side takes it; a stream whose work failed is kept back: the next side would find the error)
        if in_ctx(self.ctx, || self.sync()).is_ok() {
            SPARE.lock().unwrap_or_else(|e| e.into_inner()).give(self.ctx, self.stream.0 as usize);
        }
    }
}

/// `bufcheck` (R1a: a buffer belongs to its context): two contexts on GPU 0, its primary one and one of its own (as the split's check on one GPU makes). A buffer of the
/// second is written, read, zeroed and freed while the first is current, and the other way round: each does it in its own context, the thread's context is the same
/// after, and the GPU's free memory comes back when the buffer goes (it was freed, not lost).
#[cfg(feature = "dev")]
pub fn buf_check() -> Result<()> {
    let a = Dev::new(0)?;
    let b = Dev::new_own(0)?;
    let ctx = |d: &Dev| d.ctx as usize;
    let mut fails = 0;
    let mut say = |ok: bool, what: String| {
        println!("{} {what}", if ok { "PASS" } else { "FAIL" });
        fails += !ok as usize;
    };
    const MB: usize = 256 << 20;
    for (made, other, name) in [(&b, &a, "a context of its own"), (&a, &b, "the primary context")] {
        made.make_current()?;
        let before = made.mem_info()?.0;
        let buf = Buf::new(MB)?;
        let pattern: Vec<u32> = (0..1024u32).map(|i| i.wrapping_mul(2_654_435_761)).collect();
        other.make_current()?;
        buf.write(4096, &pattern)?;
        let mut back = vec![0u32; 1024];
        buf.read(4096, &mut back)?;
        say(
            back == pattern && current_ctx()? == ctx(other),
            format!("a buffer of {name}, written and read while the other is current: the bytes back, the thread's context the same"),
        );
        buf.zero()?;
        buf.read(4096, &mut back)?;
        say(back.iter().all(|&x| x == 0), "the same buffer zeroed while the other is current".to_string());
        let held = other.mem_info()?.0;
        let (mine, theirs) = (held_in(ctx(made)).unwrap_or(0), held_in(ctx(other)).unwrap_or(0));
        drop(buf);
        let after = other.mem_info()?.0;
        say(
            current_ctx()? == ctx(other) && after >= held + MB / 2 && after + MB / 2 >= before,
            format!("the same buffer freed while the other is current: {} MB came back, the thread's context the same", (after.saturating_sub(held)) >> 20),
        );
        let (mine_after, theirs_after) = (held_in(ctx(made)).unwrap_or(0), held_in(ctx(other)).unwrap_or(0));
        say(
            mine >= MB as u64 && mine.saturating_sub(mine_after) == MB as u64 && theirs == theirs_after,
            format!(
                "its bytes counted in {name}'s figures, not the other's: {} MB there while it was held, {} MB fewer after",
                mine >> 20,
                mine.saturating_sub(mine_after) >> 20
            ),
        );
    }
    drop(Buf::none());
    say(true, "a buffer of no bytes dropped".to_string());
    a.make_current()?;
    // a side's stream, copied, outlives the side as a stream: an event records on it, and the next side of the context takes it; a hundred sides one after
    // another make no stream past the first
    let made = SIDES_MADE.load(Relaxed);
    let side = Side::new()?;
    let kept = side.stream;
    drop(side);
    let ev = Ev::new()?;
    let recorded = ev.record(kept).and_then(|_| ev.wait_host()).is_ok();
    let again = Side::new()?;
    say(recorded && again.stream == kept, "a side's stream, its copy kept past the side, is still a stream, and the next side takes it".to_string());
    drop(again);
    for _ in 0..100 {
        drop(Side::new()?);
    }
    let extra = SIDES_MADE.load(Relaxed) - made;
    say(extra <= 1, format!("a hundred sides one after another: {extra} streams made"));
    if fails > 0 {
        return err(format!("bufcheck: {fails} failed"));
    }
    println!("bufcheck: PASS");
    Ok(())
}

#[cfg(test)]
mod pick_tests {
    use super::pick_fatbin;

    /// A driver that reports CUDA 12 loads the CUDA 12 fatbin where the build has one; CUDA 13's driver loads the main one unless CUDA 12 is asked for; a build without one has one choice.
    #[test]
    fn wddm_is_a_windows_gpu_that_is_not_in_the_tcc_mode() {
        use super::wddm_from;
        assert!(wddm_from(true, Some(0)));
        assert!(!wddm_from(true, Some(1)), "TCC is not WDDM");
        assert!(!wddm_from(true, None), "the driver did not say");
        assert!(!wddm_from(false, Some(0)), "Linux reports 0 and has no WDDM");
    }

    #[test]
    fn discovery_gives_up_on_a_driver_that_stalls() {
        use super::{probe_timeout, within};
        assert_eq!(within(5, || 7), Some(7));
        assert_eq!(within(1, || std::thread::sleep(std::time::Duration::from_secs(3))), None);
        assert_eq!((probe_timeout(None, true), probe_timeout(None, false)), (30, 0));
        assert_eq!((probe_timeout(Some("5"), false), probe_timeout(Some(" 0 "), true), probe_timeout(Some("x"), true)), (5, 0, 30));
    }

    #[test]
    fn the_fatbin_follows_the_drivers_cuda_version() {
        let (main, c12) = (&b"13"[..], &b"12"[..]);
        assert_eq!(pick_fatbin(12020, false, main, c12), c12);
        assert_eq!(pick_fatbin(12090, false, main, c12), c12);
        assert_eq!(pick_fatbin(13000, false, main, c12), main);
        assert_eq!(pick_fatbin(13020, false, main, c12), main);
        assert_eq!(pick_fatbin(13020, true, main, c12), c12);
        assert_eq!(pick_fatbin(12020, false, main, &[]), main);
        assert_eq!(pick_fatbin(13020, true, main, &[]), main);
    }
}

/// Waits for the work queued on stream `s` (a Dev's, or a wrapper's) to finish.
pub fn sync_stream(s: Stream) -> Result<()> {
    // SAFETY: a stream a Dev or a wrapper made, live while its owner is.
    unsafe { cu("cuStreamSynchronize", (driver()?.cuStreamSynchronize)(s)) }
}

#[cfg(test)]
mod tests {
    /// A context's sides take back its own streams alone, the latest first.
    #[test]
    fn a_context_takes_its_own_spare_streams() {
        let mut sp = super::Spare(Vec::new());
        sp.give(1, 10);
        sp.give(2, 20);
        sp.give(1, 11);
        assert_eq!(sp.take(3), None);
        assert_eq!(sp.take(1), Some(11));
        assert_eq!(sp.take(2), Some(20));
        assert_eq!(sp.take(1), Some(10));
        assert_eq!(sp.take(1), None);
    }
}
