//! A GPU as the engine uses it: its primary context made current on this thread, one non-blocking stream, device
//! memory with an owner, the engine's own kernels (the fatbin build.rs made) and CUDA graphs. One `Dev` a thread.

use crate::ffi::{cu, driver, err, CuEvent, CuFunction, CuGraph, CuGraphExec, CuModule, Result, Stream};
use std::ffi::{c_char, c_int, c_void};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

#[cfg(not(no_kernels))]
static FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/engine.fatbin"));
#[cfg(no_kernels)]
static FATBIN: &[u8] = &[];
/// The kernels built a second time by CUDA 12's nvcc (build.rs, NVCC12: not compressed, so that a driver of CUDA 12.2 reads them); empty where there is none.
#[cfg(not(no_kernels))]
static FATBIN12: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/engine12.fatbin"));
#[cfg(no_kernels)]
static FATBIN12: &[u8] = &[];

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
        let d = driver()?;
        let mut count = 0;
        // SAFETY: a host out-pointer.
        unsafe { cu("cuDeviceGetCount", (d.cuDeviceGetCount)(&mut count))? };
        let all: Vec<Gpu> = (0..count).filter_map(|i| Dev::probe(i).ok()).collect();
        if all.is_empty() {
            return Dev::probe(0).map(|g| vec![g]);
        }
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
                    return Err(crate::ffi::Error(format!("{} ({name}: grid {grid:?}, block {block:?}, smem {smem})", e.0)));
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
        let mut ptrs = $crate::dev::KernelArgs::ptrs(&mut vals);
        $k.launch($grid, $block, $s, &mut ptrs)
    }};
    ($k:expr, $grid:expr, $block:expr, smem $smem:expr, $s:expr; $($a:expr),+ $(,)?) => {{
        let mut vals = ($($a,)+);
        let mut ptrs = $crate::dev::KernelArgs::ptrs(&mut vals);
        $k.launch_smem($grid, $block, $smem, $s, &mut ptrs)
    }};
}

/// Whether kernels take the shapes that fit 64 KiB of shared memory a block: an AMD build's (RDNA 3 and 4 have no more), or any GPU's in a development
/// build with GLYD_SMEM64=1 (those shapes checked where no such GPU is at hand). Elsewhere every kernel keeps the shape it has on NVIDIA's GPUs.
pub fn smem64() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| cfg!(glyd_hip) || crate::tune!("GLYD_SMEM64").is_some_and(|v| v == "1"))
}

/// `out` read from device memory at `src` (synchronously: the caller has waited for the work that writes it).
pub fn read_dev<T: Copy>(src: u64, out: &mut [T]) -> Result<()> {
    let n = std::mem::size_of_val(out);
    // SAFETY: n bytes of host memory from n of device memory the caller names; T is plain data.
    unsafe { cu("cuMemcpyDtoH", (driver()?.cuMemcpyDtoH_v2)(out.as_mut_ptr() as *mut c_void, src, n)) }
}

/// Device memory, freed when dropped.
pub struct Buf {
    pub ptr: u64,
    pub bytes: usize,
}

/// The bytes the engine's buffers hold on the GPU now, the most they have held since the program started and when (milliseconds since 1970): counted where `Buf` allocates and frees, so what
/// the CUDA context, the kernels' code, a graph and a library's own allocations (cuBLAS's) take is not in them. One GPU a process.
static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);
static PEAK_AT: AtomicU64 = AtomicU64::new(0);

/// (now, peak, the peak's time in ms since 1970) of the engine's buffers.
pub fn held_now() -> (u64, u64, u64) {
    (LIVE.load(Relaxed), PEAK.load(Relaxed), PEAK_AT.load(Relaxed))
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
    pub fn new(bytes: usize) -> Result<Buf> {
        let mut p = 0u64;
        // SAFETY: a host out-pointer; at least one byte is taken.
        unsafe { cu("cuMemAlloc", (driver()?.cuMemAlloc_v2)(&mut p, bytes.max(1)))? };
        let now = LIVE.fetch_add(bytes as u64, Relaxed) + bytes as u64;
        if PEAK.fetch_max(now, Relaxed) < now {
            PEAK_AT.store(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64), Relaxed);
        }
        Ok(Buf { ptr: p, bytes })
    }

    pub fn zeroed(bytes: usize) -> Result<Buf> {
        let b = Buf::new(bytes)?;
        b.zero()?;
        Ok(b)
    }

    /// `data` copied to the buffer at byte `offset`, complete when this returns.
    pub fn write<T: Copy>(&self, offset: usize, data: &[T]) -> Result<()> {
        let n = std::mem::size_of_val(data);
        if offset + n > self.bytes {
            return err(format!("a write of {n} bytes at {offset}, past the buffer's {}", self.bytes));
        }
        // SAFETY: n bytes of host memory to n of this buffer's.
        unsafe { cu("cuMemcpyHtoD", (driver()?.cuMemcpyHtoD_v2)(self.ptr + offset as u64, data.as_ptr() as *const c_void, n))? };
        settle()
    }

    /// The buffer's bytes at `offset` into `out` (synchronously; the work queued on the stream is waited for by the caller).
    pub fn read<T: Copy>(&self, offset: usize, out: &mut [T]) -> Result<()> {
        let n = std::mem::size_of_val(out);
        if offset + n > self.bytes {
            return err(format!("a read of {n} bytes at {offset}, past the buffer's {}", self.bytes));
        }
        // SAFETY: n of this buffer's bytes to n of host memory; T is plain data.
        unsafe { cu("cuMemcpyDtoH", (driver()?.cuMemcpyDtoH_v2)(out.as_mut_ptr() as *mut c_void, self.ptr + offset as u64, n)) }
    }

    /// The buffer's bytes set to zero, complete when this returns.
    pub fn zero(&self) -> Result<()> {
        // SAFETY: the buffer's own bytes.
        unsafe { cu("cuMemsetD8", (driver()?.cuMemsetD8_v2)(self.ptr, 0, self.bytes.max(1)))? };
        settle()
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
        if let Ok(d) = driver() {
            // SAFETY: allocated by Buf::new, freed once.
            unsafe { (d.cuMemFree_v2)(self.ptr) };
        }
        LIVE.fetch_sub(self.bytes as u64, Relaxed);
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

    pub fn as_slice_mut<T>(&mut self) -> &mut [T] {
        // SAFETY: the allocation holds bytes / size_of::<T>() elements; callers use plain data.
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

pub fn memset_async(dst: u64, v: u8, bytes: usize, s: Stream) -> Result<()> {
    // SAFETY: the caller's device range.
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
    pub fn launch(&self, s: Stream) -> Result<()> {
        // SAFETY: an instantiated graph.
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

/// A timing event.
pub struct Ev(CuEvent);

impl Ev {
    pub fn new() -> Result<Ev> {
        let mut e = std::ptr::null_mut();
        // SAFETY: a host out-pointer; 0: default flags.
        unsafe { cu("cuEventCreate", (driver()?.cuEventCreate)(&mut e, 0))? };
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

/// A handle to an event that another thread may record or make a stream wait for: the event (`Ev::as_ref`) must outlive every use.
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
}

impl Ev {
    /// A handle for another thread to record the event with (see `EvRef`).
    pub fn as_ref(&self) -> EvRef {
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

/// A second stream on the engine's device (non-blocking, as the engine's own), for work that runs beside the engine's: the expert cache's copies. Ordered against the engine's stream by events only.
pub struct Side {
    pub stream: Stream,
}

impl Side {
    pub fn new() -> Result<Side> {
        let mut s = Stream::DEFAULT;
        // SAFETY: a host out-pointer; 1: CU_STREAM_NON_BLOCKING.
        unsafe { cu("cuStreamCreate", (driver()?.cuStreamCreate)(&mut s, 1))? };
        Ok(Side { stream: s })
    }

    pub fn sync(&self) -> Result<()> {
        // SAFETY: a stream this wrapper made.
        unsafe { cu("cuStreamSynchronize", (driver()?.cuStreamSynchronize)(self.stream)) }
    }
}

impl Drop for Side {
    fn drop(&mut self) {
        if let Ok(d) = driver() {
            // SAFETY: made in new, destroyed once.
            unsafe { (d.cuStreamDestroy_v2)(self.stream) };
        }
    }
}

#[cfg(test)]
mod pick_tests {
    use super::pick_fatbin;

    /// A driver that reports CUDA 12 loads the CUDA 12 fatbin where the build has one; CUDA 13's driver loads the main one unless CUDA 12 is asked for; a build without one has one choice.
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
