//! R1a (#134): the CUDA and HIP device layer behind one interface (docs/design/runtime.md section 4, "Under CUDA and HIP: `Device`").
//!
//! `Device` is what a GPU vendor's device layer gives the engine: devices and their properties, contexts, memory and copies, pinned and
//! registered host memory, streams, events, graphs, modules and kernel launches, peer access. `CudaDevice` implements it over the driver
//! table `ffi.rs` loads at run time: NVIDIA's libcuda, or under `cfg(glyd_hip)` AMD's HIP runtime, whose functions the same table holds by
//! their HIP names. `dev.rs` (buffers, streams, events, graphs, kernels) reaches the driver through this trait alone, and is the only
//! caller of it; every other file asks `dev.rs`.
//!
//! Static dispatch: the methods are associated functions of a type that holds nothing, so a call is a direct call and never a virtual one
//! (a launch is one more inlined function, not a table lookup). Each call's error names the driver function and its status, as before.
//!
//! The GPU library's operations and the dense products are traits of their own (`gpu_ops::GpuOps`, `gpu_ops::Gemm`); the layer kinds a
//! vendor runs are `runtime::backend::Backend`, above all three.

use crate::ffi::{cu, driver, CuEvent, CuFunction, CuGraph, CuGraphExec, CuModule, Result, Stream};
use std::ffi::{c_char, c_int, c_void};

/// A device attribute the engine reads, by the driver's number for it (CU_DEVICE_ATTRIBUTE_*; an AMD build's table turns them into HIP's).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Attr {
    WarpSize = 10,
    Sms = 16,
    /// Windows' TCC mode (1) or a display driver, WDDM (0).
    Tcc = 35,
    CcMajor = 75,
    CcMinor = 76,
    /// The most shared memory a block may ask for once its kernel opts in.
    SmemOptin = 97,
}

/// What pinned host memory is besides pinned (the driver's CU_MEMHOSTALLOC_* flag).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostMem {
    Pinned = 0,
    /// Pinned for every context (what two GPUs copy through).
    Portable = 1,
    /// Mapped into the device's address space (a kernel writes it through the same pointer).
    Mapped = 2,
}

/// A GPU vendor's device layer. Handles are plain values the driver gives; what owns them (and frees them once) is `dev.rs`'s.
pub trait Device {
    /// A context (the GPU's primary one, or one of its own).
    type Ctx: Copy;
    type Stream: Copy;
    type Event: Copy;
    /// A captured graph and its instance.
    type Graph: Copy;
    type Exec: Copy;
    type Module: Copy;
    type Function: Copy;

    // Devices and their properties.

    /// The CUDA (or HIP) version the driver supports (13000 for 13.0).
    fn version() -> Result<i32>;
    /// How many GPUs the driver shows (CUDA_VISIBLE_DEVICES applied).
    fn count() -> Result<i32>;
    /// GPU `ordinal`'s device handle.
    fn device(ordinal: i32) -> Result<i32>;
    fn name(dev: i32) -> Result<String>;
    fn attribute(dev: i32, a: Attr) -> Result<i32>;
    fn total_mem(dev: i32) -> Result<usize>;
    /// Whether GPU `a` can read and write GPU `b`'s memory directly.
    fn can_access_peer(a: i32, b: i32) -> Result<bool>;

    // Contexts, one current a thread.

    /// The GPU's primary context, retained (for the process: `release_primary` lets it go).
    fn primary_ctx(dev: i32) -> Result<Self::Ctx>;
    fn release_primary(dev: i32) -> Result<()>;
    /// A context of its own on the GPU, current on this thread after it.
    fn new_ctx(dev: i32) -> Result<Self::Ctx>;
    /// # Safety
    /// `ctx` is a context this process retained or made and still holds.
    unsafe fn set_current(ctx: Self::Ctx) -> Result<()>;
    /// The context current on this thread (null where there is none).
    fn current() -> Result<Self::Ctx>;
    /// The device of the context current on this thread.
    fn current_device() -> Result<i32>;
    /// The current context's GPU may reach `peer`'s memory (the peer GPU's context) from now on; already on is no error.
    ///
    /// # Safety
    /// As `set_current`, for `peer`.
    unsafe fn enable_peer(peer: Self::Ctx) -> Result<()>;
    /// (free, total) bytes of the current context's GPU.
    fn mem_info() -> Result<(usize, usize)>;
    /// Waits for every stream's work in the current context.
    fn ctx_sync() -> Result<()>;

    // Device memory and copies (addresses are the driver's 64-bit device pointers).

    /// `bytes` (at least 1) of device memory in the current context.
    fn alloc(bytes: usize) -> Result<u64>;
    /// # Safety
    /// `ptr` is an allocation of `alloc` in the current context, freed once, and no queued work reads it.
    unsafe fn free(ptr: u64) -> Result<()>;
    /// Synchronous copies and fill (the legacy stream: complete when they return, but for a copy from pageable memory, which is staged).
    ///
    /// # Safety
    /// `dst .. dst + bytes` device memory the caller may write; `src .. src + bytes` host memory it may read.
    unsafe fn h2d(dst: u64, src: *const c_void, bytes: usize) -> Result<()>;
    /// # Safety
    /// `src .. src + bytes` device memory the caller may read; `dst .. dst + bytes` host memory it may write.
    unsafe fn d2h(dst: *mut c_void, src: u64, bytes: usize) -> Result<()>;
    /// # Safety
    /// `dst .. dst + bytes` device memory the caller may write.
    unsafe fn memset(dst: u64, v: u8, bytes: usize) -> Result<()>;
    /// Copies and fill queued on `s`, ordered with the work on it.
    ///
    /// # Safety
    /// As `h2d`, the host range valid until the stream has run the copy.
    unsafe fn h2d_async(dst: u64, src: *const c_void, bytes: usize, s: Self::Stream) -> Result<()>;
    /// # Safety
    /// As `d2h`, the host range valid until the stream has run the copy.
    unsafe fn d2h_async(dst: *mut c_void, src: u64, bytes: usize, s: Self::Stream) -> Result<()>;
    /// # Safety
    /// Two live device ranges of `bytes` that do not overlap, valid until the stream has run the copy.
    unsafe fn d2d_async(dst: u64, src: u64, bytes: usize, s: Self::Stream) -> Result<()>;
    /// # Safety
    /// As `memset`.
    unsafe fn memset_async(dst: u64, v: u8, bytes: usize, s: Self::Stream) -> Result<()>;

    // Host memory.

    /// `bytes` (at least 1) of pinned host memory of `kind`.
    fn host_alloc(bytes: usize, kind: HostMem) -> Result<*mut c_void>;
    /// # Safety
    /// `ptr` is an allocation of `host_alloc`, freed once, and no queued copy uses it.
    unsafe fn host_free(ptr: *mut c_void) -> Result<()>;
    /// Host memory `[ptr, ptr + bytes)` (page-aligned) registered with the driver (`flags`: CU_MEMHOSTREGISTER_*).
    ///
    /// # Safety
    /// The range is the caller's, stays mapped until `host_unregister`, and is registered once.
    unsafe fn host_register(ptr: *mut c_void, bytes: usize, flags: u32) -> Result<()>;
    /// # Safety
    /// `ptr` was registered, and no copy from it is still queued.
    unsafe fn host_unregister(ptr: *mut c_void) -> Result<()>;

    // Streams and graphs.

    /// A stream of the current context that does not wait for the legacy one (CU_STREAM_NON_BLOCKING).
    fn stream() -> Result<Self::Stream>;
    /// Waits for the work queued on `s`.
    fn stream_sync(s: Self::Stream) -> Result<()>;
    /// # Safety
    /// `s` was made by `stream`, has no work queued, and neither it nor a copy of it is used after (the engine keeps its streams: `dev::Side`).
    unsafe fn stream_destroy(s: Self::Stream) -> Result<()>;
    /// The work queued on `s` captured from now on, not run (relaxed mode: other threads may call the driver meanwhile).
    fn begin_capture(s: Self::Stream) -> Result<()>;
    /// The capture `begin_capture` began, ended.
    fn end_capture(s: Self::Stream) -> Result<Self::Graph>;
    /// # Safety
    /// `g` is a graph `end_capture` gave, not destroyed.
    unsafe fn instantiate(g: Self::Graph) -> Result<Self::Exec>;
    /// # Safety
    /// `x` is alive, and everything its capture used still is, holding what the captured work expects there.
    unsafe fn graph_launch(x: Self::Exec, s: Self::Stream) -> Result<()>;
    /// # Safety
    /// The two were made by `end_capture` and `instantiate`, are destroyed once, and no launch of them is queued.
    unsafe fn graph_destroy(g: Self::Graph, x: Self::Exec) -> Result<()>;
    /// # Safety
    /// `g` was made by `end_capture`, has no instance, and is destroyed once.
    unsafe fn graph_free(g: Self::Graph) -> Result<()>;

    // Events.

    /// An event; `timing`: one whose times can be read (else it only orders work, which the driver records more cheaply).
    fn event(timing: bool) -> Result<Self::Event>;
    /// # Safety
    /// `e` is an event `event` made and not destroyed, here and in the event calls below.
    unsafe fn record(e: Self::Event, s: Self::Stream) -> Result<()>;
    /// The event recorded as a node of the graph being captured on `s`, which a replay records again.
    ///
    /// # Safety
    /// As `record`.
    unsafe fn record_in_graph(e: Self::Event, s: Self::Stream) -> Result<()>;
    /// Work queued on `s` from now on waits for the work `e` was recorded after.
    ///
    /// # Safety
    /// As `record`.
    unsafe fn wait_event(s: Self::Stream, e: Self::Event) -> Result<()>;
    /// The calling thread waits for the work `e` was recorded after.
    ///
    /// # Safety
    /// As `record`.
    unsafe fn event_sync(e: Self::Event) -> Result<()>;
    /// Whether the work `e` was recorded after has run (an event not recorded yet has).
    ///
    /// # Safety
    /// As `record`.
    unsafe fn event_done(e: Self::Event) -> Result<bool>;
    /// Milliseconds from `a` to `b`, both recorded and complete, timing events.
    ///
    /// # Safety
    /// As `record`, for both.
    unsafe fn elapsed(a: Self::Event, b: Self::Event) -> Result<f32>;
    /// # Safety
    /// As `record`; destroyed once.
    unsafe fn event_destroy(e: Self::Event) -> Result<()>;

    // Modules and launches.

    /// A module from a fatbin (or code object bundle) in memory, in the current context.
    ///
    /// # Safety
    /// `image` is a whole fatbin the driver can read, alive for the module's life.
    unsafe fn module_load(image: *const c_void) -> Result<Self::Module>;
    /// A kernel of `m` by name (the error names the kernel).
    ///
    /// # Safety
    /// `m` is a module `module_load` made, loaded still.
    unsafe fn function(m: Self::Module, name: &str) -> Result<Self::Function>;
    /// The most dynamic shared memory a launch of `f` may ask for (past 48 KiB: an opt-in).
    ///
    /// # Safety
    /// `f` is a kernel `function` found, of a module loaded still.
    unsafe fn allow_smem(f: Self::Function, bytes: u32) -> Result<()>;
    /// The blocks of `threads` threads and `smem` bytes of dynamic shared memory an SM holds at once of `f` (its shared memory allowed first).
    ///
    /// # Safety
    /// As `allow_smem`.
    unsafe fn occupancy(f: Self::Function, threads: u32, smem: u32) -> Result<i32>;
    /// `f` launched on `s` with `smem` bytes of dynamic shared memory.
    ///
    /// # Safety
    /// `args` point to values of the kernel's parameter types, in order, live for the call; device addresses are of live buffers.
    unsafe fn launch(f: Self::Function, grid: [u32; 3], block: [u32; 3], smem: u32, s: Self::Stream, args: &mut [*mut c_void]) -> Result<()>;

    // Hopper (compute capability 9.0): its kernels' tensor maps and clusters. An AMD build: not supported.

    /// A tensor map of a bf16 matrix at `base` of `rows` rows of `cols` values, rows `row_bytes` apart, read in boxes of `box_cols` by `box_rows` with the
    /// 128-byte swizzle and L2's 256-byte promotion, rows past the matrix read as zeros (cuTensorMapEncodeTiled): the map a kernel takes as its parameter.
    fn tensor_map(base: u64, shape: (u64, u64, u64), boxes: (u32, u32)) -> Result<TensorMap>;
    /// The clusters of `cluster` blocks of `threads` and `smem` the GPU runs at once of `f`, for a grid of `grid` blocks (cuOccupancyMaxActiveClusters).
    ///
    /// # Safety
    /// As `occupancy`.
    unsafe fn max_clusters(f: Self::Function, grid: u32, threads: u32, smem: u32, cluster: u32) -> Result<i32>;
    /// `launch` of a grid in clusters of `cluster` blocks along x (cuLaunchKernelEx with the cluster's dimension).
    ///
    /// # Safety
    /// As `launch`.
    unsafe fn launch_clusters(f: Self::Function, grid: u32, threads: u32, smem: u32, cluster: u32, s: Self::Stream, args: &mut [*mut c_void]) -> Result<()>;
}

/// NVIDIA's device layer over libcuda's driver API, and AMD's over HIP's runtime in an AMD build (`cfg(glyd_hip)`: the same table, HIP's
/// functions by CUDA's names).
pub struct CudaDevice;

impl Device for CudaDevice {
    type Ctx = *mut c_void;
    type Stream = Stream;
    type Event = CuEvent;
    type Graph = CuGraph;
    type Exec = CuGraphExec;
    type Module = CuModule;
    type Function = CuFunction;

    fn version() -> Result<i32> {
        let mut v = 0;
        // SAFETY: a host out-pointer.
        unsafe { cu("cuDriverGetVersion", (driver()?.cuDriverGetVersion)(&mut v))? };
        Ok(v)
    }

    fn count() -> Result<i32> {
        let mut n = 0;
        // SAFETY: a host out-pointer.
        unsafe { cu("cuDeviceGetCount", (driver()?.cuDeviceGetCount)(&mut n))? };
        Ok(n)
    }

    fn device(ordinal: i32) -> Result<i32> {
        let mut dev = 0;
        // SAFETY: a host out-pointer.
        unsafe { cu("cuDeviceGet", (driver()?.cuDeviceGet)(&mut dev, ordinal))? };
        Ok(dev)
    }

    fn name(dev: i32) -> Result<String> {
        let mut b = [0 as c_char; 256];
        // SAFETY: a host buffer of the size passed.
        unsafe { cu("cuDeviceGetName", (driver()?.cuDeviceGetName)(b.as_mut_ptr(), b.len() as c_int, dev))? };
        // SAFETY: NUL-terminated within the buffer.
        Ok(unsafe { std::ffi::CStr::from_ptr(b.as_ptr()) }.to_string_lossy().into_owned())
    }

    fn attribute(dev: i32, a: Attr) -> Result<i32> {
        let mut v = 0;
        // SAFETY: a host out-pointer; the attribute's number.
        unsafe { cu("cuDeviceGetAttribute", (driver()?.cuDeviceGetAttribute)(&mut v, a as c_int, dev))? };
        Ok(v)
    }

    fn total_mem(dev: i32) -> Result<usize> {
        let mut t = 0usize;
        // SAFETY: a host out-pointer.
        unsafe { cu("cuDeviceTotalMem", (driver()?.cuDeviceTotalMem_v2)(&mut t, dev))? };
        Ok(t)
    }

    fn can_access_peer(a: i32, b: i32) -> Result<bool> {
        let mut can = 0;
        // SAFETY: a host out-pointer.
        unsafe { cu("cuDeviceCanAccessPeer", (driver()?.cuDeviceCanAccessPeer)(&mut can, a, b))? };
        Ok(can != 0)
    }

    fn primary_ctx(dev: i32) -> Result<*mut c_void> {
        let mut ctx = std::ptr::null_mut();
        // SAFETY: a host out-pointer; the context stays retained until `release_primary`.
        unsafe { cu("cuDevicePrimaryCtxRetain", (driver()?.cuDevicePrimaryCtxRetain)(&mut ctx, dev))? };
        Ok(ctx)
    }

    fn release_primary(dev: i32) -> Result<()> {
        // SAFETY: no pointers; a retain of `primary_ctx` let go.
        unsafe { cu("cuDevicePrimaryCtxRelease", (driver()?.cuDevicePrimaryCtxRelease_v2)(dev)) }
    }

    fn new_ctx(dev: i32) -> Result<*mut c_void> {
        let mut ctx = std::ptr::null_mut();
        // SAFETY: a host out-pointer; the context made stays for the process and is current after it.
        unsafe { cu("cuCtxCreate", (driver()?.cuCtxCreate_v2)(&mut ctx, 0, dev))? };
        Ok(ctx)
    }

    unsafe fn set_current(ctx: *mut c_void) -> Result<()> {
        // SAFETY: a context the process holds (the caller's contract).
        unsafe { cu("cuCtxSetCurrent", (driver()?.cuCtxSetCurrent)(ctx)) }
    }

    fn current() -> Result<*mut c_void> {
        let mut c = std::ptr::null_mut();
        // SAFETY: a host out-pointer.
        unsafe { cu("cuCtxGetCurrent", (driver()?.cuCtxGetCurrent)(&mut c))? };
        Ok(c)
    }

    fn current_device() -> Result<i32> {
        let mut dev = 0;
        // SAFETY: a host out-pointer.
        unsafe { cu("cuCtxGetDevice", (driver()?.cuCtxGetDevice)(&mut dev))? };
        Ok(dev)
    }

    unsafe fn enable_peer(peer: *mut c_void) -> Result<()> {
        // SAFETY: a context the process holds (the caller's contract); 704: access already on.
        let r = unsafe { (driver()?.cuCtxEnablePeerAccess)(peer, 0) };
        if r != 0 && r != 704 {
            cu("cuCtxEnablePeerAccess", r)?;
        }
        Ok(())
    }

    fn mem_info() -> Result<(usize, usize)> {
        let (mut f, mut t) = (0usize, 0usize);
        // SAFETY: host out-pointers.
        unsafe { cu("cuMemGetInfo", (driver()?.cuMemGetInfo_v2)(&mut f, &mut t))? };
        Ok((f, t))
    }

    fn ctx_sync() -> Result<()> {
        // SAFETY: no pointers.
        unsafe { cu("cuCtxSynchronize", (driver()?.cuCtxSynchronize)()) }
    }

    fn alloc(bytes: usize) -> Result<u64> {
        let mut p = 0u64;
        // SAFETY: a host out-pointer; at least one byte is taken.
        unsafe { cu("cuMemAlloc", (driver()?.cuMemAlloc_v2)(&mut p, bytes.max(1)))? };
        Ok(p)
    }

    unsafe fn free(ptr: u64) -> Result<()> {
        // SAFETY: the caller's allocation (its contract).
        unsafe { cu("cuMemFree", (driver()?.cuMemFree_v2)(ptr)) }
    }

    unsafe fn h2d(dst: u64, src: *const c_void, bytes: usize) -> Result<()> {
        // SAFETY: the caller's ranges (its contract).
        unsafe { cu("cuMemcpyHtoD", (driver()?.cuMemcpyHtoD_v2)(dst, src, bytes)) }
    }

    unsafe fn d2h(dst: *mut c_void, src: u64, bytes: usize) -> Result<()> {
        // SAFETY: the caller's ranges (its contract).
        unsafe { cu("cuMemcpyDtoH", (driver()?.cuMemcpyDtoH_v2)(dst, src, bytes)) }
    }

    unsafe fn memset(dst: u64, v: u8, bytes: usize) -> Result<()> {
        // SAFETY: the caller's range (its contract).
        unsafe { cu("cuMemsetD8", (driver()?.cuMemsetD8_v2)(dst, v, bytes)) }
    }

    unsafe fn h2d_async(dst: u64, src: *const c_void, bytes: usize, s: Stream) -> Result<()> {
        // SAFETY: the caller's ranges (its contract).
        unsafe { cu("cuMemcpyHtoDAsync", (driver()?.cuMemcpyHtoDAsync_v2)(dst, src, bytes, s)) }
    }

    unsafe fn d2h_async(dst: *mut c_void, src: u64, bytes: usize, s: Stream) -> Result<()> {
        // SAFETY: the caller's ranges (its contract).
        unsafe { cu("cuMemcpyDtoHAsync", (driver()?.cuMemcpyDtoHAsync_v2)(dst, src, bytes, s)) }
    }

    unsafe fn d2d_async(dst: u64, src: u64, bytes: usize, s: Stream) -> Result<()> {
        // SAFETY: the caller's ranges (its contract).
        unsafe { cu("cuMemcpyDtoDAsync", (driver()?.cuMemcpyDtoDAsync_v2)(dst, src, bytes, s)) }
    }

    unsafe fn memset_async(dst: u64, v: u8, bytes: usize, s: Stream) -> Result<()> {
        // SAFETY: the caller's range (its contract).
        unsafe { cu("cuMemsetD8Async", (driver()?.cuMemsetD8Async)(dst, v, bytes, s)) }
    }

    fn host_alloc(bytes: usize, kind: HostMem) -> Result<*mut c_void> {
        let mut p = std::ptr::null_mut();
        // SAFETY: a host out-pointer and the flag's number.
        unsafe { cu("cuMemHostAlloc", (driver()?.cuMemHostAlloc)(&mut p, bytes.max(1), kind as u32))? };
        Ok(p)
    }

    unsafe fn host_free(ptr: *mut c_void) -> Result<()> {
        // SAFETY: the caller's allocation (its contract).
        unsafe { cu("cuMemFreeHost", (driver()?.cuMemFreeHost)(ptr)) }
    }

    unsafe fn host_register(ptr: *mut c_void, bytes: usize, flags: u32) -> Result<()> {
        // SAFETY: the caller's range (its contract).
        unsafe { cu("cuMemHostRegister", (driver()?.cuMemHostRegister_v2)(ptr, bytes, flags)) }
    }

    unsafe fn host_unregister(ptr: *mut c_void) -> Result<()> {
        // SAFETY: the caller's registration (its contract).
        unsafe { cu("cuMemHostUnregister", (driver()?.cuMemHostUnregister)(ptr)) }
    }

    fn stream() -> Result<Stream> {
        let mut s = Stream::DEFAULT;
        // SAFETY: a host out-pointer; 1: CU_STREAM_NON_BLOCKING.
        unsafe { cu("cuStreamCreate", (driver()?.cuStreamCreate)(&mut s, 1))? };
        Ok(s)
    }

    fn stream_sync(s: Stream) -> Result<()> {
        // SAFETY: a stream made by `stream` (a `Stream` is made nowhere else) or the legacy one, never destroyed.
        unsafe { cu("cuStreamSynchronize", (driver()?.cuStreamSynchronize)(s)) }
    }

    unsafe fn stream_destroy(s: Stream) -> Result<()> {
        // SAFETY: the caller's stream, idle and not used again (its contract).
        unsafe { cu("cuStreamDestroy", (driver()?.cuStreamDestroy_v2)(s)) }
    }

    fn begin_capture(s: Stream) -> Result<()> {
        // SAFETY: as `stream_sync`; 2: CU_STREAM_CAPTURE_MODE_RELAXED.
        unsafe { cu("cuStreamBeginCapture", (driver()?.cuStreamBeginCapture_v2)(s, 2)) }
    }

    fn end_capture(s: Stream) -> Result<CuGraph> {
        let mut g: CuGraph = std::ptr::null_mut();
        // SAFETY: as `stream_sync`; a host out-pointer.
        unsafe { cu("cuStreamEndCapture", (driver()?.cuStreamEndCapture)(s, &mut g))? };
        Ok(g)
    }

    unsafe fn instantiate(g: CuGraph) -> Result<CuGraphExec> {
        let mut x: CuGraphExec = std::ptr::null_mut();
        // SAFETY: a captured graph (the caller's contract); a host out-pointer.
        unsafe { cu("cuGraphInstantiate", (driver()?.cuGraphInstantiateWithFlags)(&mut x, g, 0))? };
        Ok(x)
    }

    unsafe fn graph_launch(x: CuGraphExec, s: Stream) -> Result<()> {
        // SAFETY: an instantiated graph whose buffers are alive (the caller's contract).
        unsafe { cu("cuGraphLaunch", (driver()?.cuGraphLaunch)(x, s)) }
    }

    unsafe fn graph_destroy(g: CuGraph, x: CuGraphExec) -> Result<()> {
        let d = driver()?;
        // SAFETY: made by `end_capture` and `instantiate`, destroyed once (the caller's contract); the graph goes whether its instance went or not.
        let (rx, rg) = unsafe { ((d.cuGraphExecDestroy)(x), (d.cuGraphDestroy)(g)) };
        cu("cuGraphExecDestroy", rx).and(cu("cuGraphDestroy", rg))
    }

    unsafe fn graph_free(g: CuGraph) -> Result<()> {
        // SAFETY: made by `end_capture`, with no instance, destroyed once (the caller's contract).
        unsafe { cu("cuGraphDestroy", (driver()?.cuGraphDestroy)(g)) }
    }

    fn event(timing: bool) -> Result<CuEvent> {
        let mut e = std::ptr::null_mut();
        // SAFETY: a host out-pointer; 2: CU_EVENT_DISABLE_TIMING.
        unsafe { cu("cuEventCreate", (driver()?.cuEventCreate)(&mut e, if timing { 0 } else { 2 }))? };
        Ok(e)
    }

    unsafe fn record(e: CuEvent, s: Stream) -> Result<()> {
        // SAFETY: an event the caller holds (its contract).
        unsafe { cu("cuEventRecord", (driver()?.cuEventRecord)(e, s)) }
    }

    unsafe fn record_in_graph(e: CuEvent, s: Stream) -> Result<()> {
        // SAFETY: as `record`; 1 is CU_EVENT_RECORD_EXTERNAL.
        unsafe { cu("cuEventRecordWithFlags", (driver()?.cuEventRecordWithFlags)(e, s, 1)) }
    }

    unsafe fn wait_event(s: Stream, e: CuEvent) -> Result<()> {
        // SAFETY: as `record`.
        unsafe { cu("cuStreamWaitEvent", (driver()?.cuStreamWaitEvent)(s, e, 0)) }
    }

    unsafe fn event_sync(e: CuEvent) -> Result<()> {
        // SAFETY: as `record`.
        unsafe { cu("cuEventSynchronize", (driver()?.cuEventSynchronize)(e)) }
    }

    unsafe fn event_done(e: CuEvent) -> Result<bool> {
        // SAFETY: as `record`; 600 is CUDA_ERROR_NOT_READY.
        match unsafe { (driver()?.cuEventQuery)(e) } {
            0 => Ok(true),
            600 => Ok(false),
            r => cu("cuEventQuery", r).map(|_| false),
        }
    }

    unsafe fn elapsed(a: CuEvent, b: CuEvent) -> Result<f32> {
        let mut ms = 0f32;
        // SAFETY: as `record`; a host out-pointer.
        unsafe { cu("cuEventElapsedTime", (driver()?.cuEventElapsedTime)(&mut ms, a, b))? };
        Ok(ms)
    }

    unsafe fn event_destroy(e: CuEvent) -> Result<()> {
        // SAFETY: as `record`, destroyed once.
        unsafe { cu("cuEventDestroy", (driver()?.cuEventDestroy_v2)(e)) }
    }

    unsafe fn module_load(image: *const c_void) -> Result<CuModule> {
        let mut m = std::ptr::null_mut();
        // SAFETY: a fatbin alive for the module's life (the caller's contract); a host out-pointer.
        unsafe { cu("cuModuleLoadData", (driver()?.cuModuleLoadData)(&mut m, image))? };
        Ok(m)
    }

    unsafe fn function(m: CuModule, name: &str) -> Result<CuFunction> {
        let c = crate::ffi::cstr(name);
        let mut f = std::ptr::null_mut();
        // SAFETY: a host out-pointer and a NUL-terminated name; a loaded module (the caller's contract).
        unsafe { cu(name, (driver()?.cuModuleGetFunction)(&mut f, m, c.as_ptr()))? };
        Ok(f)
    }

    unsafe fn allow_smem(f: CuFunction, bytes: u32) -> Result<()> {
        // SAFETY: a function of a loaded module (the caller's contract) and a plain integer; 8: CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES.
        cu("cuFuncSetAttribute", unsafe { (driver()?.cuFuncSetAttribute)(f, 8, bytes as c_int) })
    }

    unsafe fn occupancy(f: CuFunction, threads: u32, smem: u32) -> Result<i32> {
        let mut n = 0;
        // SAFETY: a function of a loaded module (the caller's contract); a host out-pointer.
        unsafe {
            cu(
                "cuOccupancyMaxActiveBlocksPerMultiprocessor",
                (driver()?.cuOccupancyMaxActiveBlocksPerMultiprocessor)(&mut n, f, threads as c_int, smem as usize),
            )?
        };
        Ok(n)
    }

    unsafe fn launch(f: CuFunction, grid: [u32; 3], block: [u32; 3], smem: u32, s: Stream, args: &mut [*mut c_void]) -> Result<()> {
        // SAFETY: the caller's arguments (its contract).
        unsafe {
            cu(
                "cuLaunchKernel",
                (driver()?.cuLaunchKernel)(f, grid[0], grid[1], grid[2], block[0], block[1], block[2], smem, s, args.as_mut_ptr(), std::ptr::null_mut()),
            )
        }
    }

    #[cfg(not(glyd_hip))]
    fn tensor_map(base: u64, (cols, rows, row_bytes): (u64, u64, u64), (box_cols, box_rows): (u32, u32)) -> Result<TensorMap> {
        let h = crate::ffi::hopper()?;
        let mut map = TensorMap([0; 16]);
        let (dims, strides, boxes, unit) = ([cols, rows], [row_bytes], [box_cols, box_rows], [1u32, 1]);
        // SAFETY: the map the driver writes, 128 bytes at an address aligned to 64 at least (the driver's rule for it); the arrays of the rank's lengths.
        // 9: bf16, 0: no interleave, 3: the 128-byte swizzle, 3: L2's 256-byte promotion, 0: no NaN fill.
        unsafe {
            cu(
                "cuTensorMapEncodeTiled",
                (h.cuTensorMapEncodeTiled)(
                    &mut map as *mut TensorMap as *mut c_void,
                    9,
                    2,
                    base as *mut c_void,
                    dims.as_ptr(),
                    strides.as_ptr(),
                    boxes.as_ptr(),
                    unit.as_ptr(),
                    0,
                    3,
                    3,
                    0,
                ),
            )?
        };
        Ok(map)
    }

    #[cfg(glyd_hip)]
    fn tensor_map(_: u64, _: (u64, u64, u64), _: (u32, u32)) -> Result<TensorMap> {
        crate::ffi::err("tensor maps are Hopper's: not in an AMD build")
    }

    #[cfg(not(glyd_hip))]
    unsafe fn max_clusters(f: CuFunction, grid: u32, threads: u32, smem: u32, cluster: u32) -> Result<i32> {
        let h = crate::ffi::hopper()?;
        let mut attr = LaunchAttribute { id: 0, _pad: 0, value: [0; 16] };
        let cfg = cluster_config(grid, threads, smem, cluster, Stream(std::ptr::null_mut()), &mut attr);
        let mut n = 0;
        // SAFETY: a function of a loaded module (the caller's contract); the configuration and its attribute live for the call; a host out-pointer.
        unsafe { cu("cuOccupancyMaxActiveClusters", (h.cuOccupancyMaxActiveClusters)(&mut n, f, &cfg as *const LaunchConfig as *const c_void))? };
        Ok(n)
    }

    #[cfg(glyd_hip)]
    unsafe fn max_clusters(_: CuFunction, _: u32, _: u32, _: u32, _: u32) -> Result<i32> {
        crate::ffi::err("clusters are Hopper's: not in an AMD build")
    }

    #[cfg(not(glyd_hip))]
    unsafe fn launch_clusters(f: CuFunction, grid: u32, threads: u32, smem: u32, cluster: u32, s: Stream, args: &mut [*mut c_void]) -> Result<()> {
        let h = crate::ffi::hopper()?;
        let mut attr = LaunchAttribute { id: 0, _pad: 0, value: [0; 16] };
        let cfg = cluster_config(grid, threads, smem, cluster, s, &mut attr);
        // SAFETY: the caller's arguments (its contract); the configuration and its attribute live for the call.
        unsafe { cu("cuLaunchKernelEx", (h.cuLaunchKernelEx)(&cfg as *const LaunchConfig as *const c_void, f, args.as_mut_ptr(), std::ptr::null_mut())) }
    }

    #[cfg(glyd_hip)]
    unsafe fn launch_clusters(_: CuFunction, _: u32, _: u32, _: u32, _: u32, _: Stream, _: &mut [*mut c_void]) -> Result<()> {
        crate::ffi::err("clusters are Hopper's: not in an AMD build")
    }
}

/// cuda.h's CUtensorMap: the 128 bytes cuTensorMapEncodeTiled writes (at an address aligned to 64, the driver's rule) and Hopper's kernels take by value, aligned as the toolkit
/// of the main fatbin aligns it (build.rs: 128 by CUDA 13's cuda.h, cfg `tensor_map_128`; 64 by CUDA 12's, whose fatbin a binary of both also carries: 64 divides 128).
/// kernels/fold.cu holds each toolkit's to it.
#[cfg_attr(tensor_map_128, repr(C, align(128)))]
#[cfg_attr(not(tensor_map_128), repr(C, align(64)))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorMap(pub [u64; 16]);

/// cuda.h's CUlaunchConfig and CUlaunchAttribute (one: the cluster's dimension, CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION 4), as cuLaunchKernelEx and
/// cuOccupancyMaxActiveClusters take them.
#[repr(C)]
struct LaunchConfig {
    grid: [u32; 3],
    block: [u32; 3],
    smem: u32,
    stream: Stream,
    attrs: *const LaunchAttribute,
    num_attrs: u32,
}

#[repr(C)]
struct LaunchAttribute {
    id: u32,
    _pad: u32,
    value: [u32; 16],
}

fn cluster_config(grid: u32, threads: u32, smem: u32, cluster: u32, s: Stream, attr: &mut LaunchAttribute) -> LaunchConfig {
    *attr = LaunchAttribute { id: 4, _pad: 0, value: [0; 16] };
    attr.value[..3].copy_from_slice(&[cluster, 1, 1]);
    LaunchConfig { grid: [grid, 1, 1], block: [threads, 1, 1], smem, stream: s, attrs: attr, num_attrs: 1 }
}

#[cfg(test)]
mod tests {
    use super::{Attr, CudaDevice, Device, HostMem};

    /// The attribute and flag numbers are the driver's (CUDA's cuda.h; an AMD build's table maps the attributes to HIP's).
    #[test]
    fn the_numbers_are_the_drivers() {
        let attrs = [(Attr::WarpSize, 10), (Attr::Sms, 16), (Attr::Tcc, 35), (Attr::CcMajor, 75), (Attr::CcMinor, 76), (Attr::SmemOptin, 97)];
        assert!(attrs.iter().all(|&(a, n)| a as i32 == n));
        assert_eq!([HostMem::Pinned as u32, HostMem::Portable as u32, HostMem::Mapped as u32], [0, 1, 2]);
        // cuda.h's launch configuration and attribute: 56 and 72 bytes (the stream at 32, the attributes at 40)
        assert_eq!((std::mem::size_of::<super::LaunchConfig>(), std::mem::size_of::<super::LaunchAttribute>()), (56, 72));
        assert_eq!((std::mem::offset_of!(super::LaunchConfig, stream), std::mem::offset_of!(super::LaunchConfig, num_attrs)), (32, 48));
        // cuda.h's CUtensorMap: 128 bytes, aligned to 128 by CUDA 13's toolkit and to 64 by CUDA 12's (the main fatbin's, build.rs's cfg)
        let align = if cfg!(tensor_map_128) { 128 } else { 64 };
        assert_eq!((std::mem::size_of::<super::TensorMap>(), std::mem::align_of::<super::TensorMap>()), (128, align));
    }

    /// Where there is no driver (a machine with no GPU, a CPU runner), a call is an error of the kind that says so, not a crash.
    #[test]
    fn without_a_driver_a_call_says_so() {
        if crate::ffi::driver().is_ok() {
            return;
        }
        for r in [CudaDevice::count().map(|_| ()), CudaDevice::alloc(1).map(|_| ()), CudaDevice::stream().map(|_| ())] {
            assert_eq!(r.unwrap_err().kind, crate::ffi::Kind::NoDriver);
        }
    }
}
