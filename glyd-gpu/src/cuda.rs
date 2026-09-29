//! The CUDA driver's few calls a caller of [`crate::Library`] without a CUDA
//! runtime of its own needs: a device's primary context made current (the
//! one the library's runtime then uses too), device memory, copies and
//! streams. libcuda.so.1 is found at run time, as the library finds it.
//!
//! A [`Context`] is current on the thread that made it and stays there (it is
//! neither Send nor Sync), with its memory and streams; copies take plain
//! data alone ([`Plain`]: integers and floats, whose every bit pattern is a
//! value and which have no padding).

use crate::{api, Error, Result, Stream};
use std::ffi::{c_char, c_int, c_uint, c_void, CStr};
use std::marker::PhantomData;
use std::sync::OnceLock;

api! { Driver, DRIVER_DECLARED;
    fn cuInit(flags: c_uint) -> c_int;
    fn cuDriverGetVersion(version: *mut c_int) -> c_int;
    fn cuDeviceGet(device: *mut c_int, ordinal: c_int) -> c_int;
    fn cuDeviceGetCount(count: *mut c_int) -> c_int;
    fn cuDeviceGetName(name: *mut c_char, len: c_int, device: c_int) -> c_int;
    fn cuDeviceGetAttribute(value: *mut c_int, attribute: c_int, device: c_int) -> c_int;
    fn cuDevicePrimaryCtxRetain(ctx: *mut *mut c_void, device: c_int) -> c_int;
    fn cuDevicePrimaryCtxRelease_v2(device: c_int) -> c_int;
    fn cuCtxSetCurrent(ctx: *mut c_void) -> c_int;
    fn cuCtxSynchronize() -> c_int;
    fn cuMemAlloc_v2(ptr: *mut u64, bytes: usize) -> c_int;
    fn cuMemFree_v2(ptr: u64) -> c_int;
    fn cuMemcpyHtoD_v2(dst: u64, src: *const c_void, bytes: usize) -> c_int;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: u64, bytes: usize) -> c_int;
    fn cuMemsetD8_v2(dst: u64, value: u8, bytes: usize) -> c_int;
    fn cuStreamCreate(stream: *mut Stream, flags: c_uint) -> c_int;
    fn cuStreamDestroy_v2(stream: Stream) -> c_int;
    fn cuStreamSynchronize(stream: Stream) -> c_int;
    fn cuGetErrorString(status: c_int, text: *mut *const c_char) -> c_int;
}

/// The driver, loaded and initialized once for the process.
fn driver() -> Result<&'static Driver> {
    static DRIVER: OnceLock<std::result::Result<Driver, String>> = OnceLock::new();
    DRIVER
        .get_or_init(|| {
            let h = crate::dl::open("libcuda.so.1").map_err(|e| format!("no CUDA driver ({e})"))?;
            // SAFETY: libcuda's functions of these names take these arguments (CUDA 11 on: the _v2 ones).
            let d = unsafe { Driver::load(h) }.map_err(|name| format!("libcuda.so.1 has no {name}"))?;
            // SAFETY: no pointers.
            match unsafe { (d.cuInit)(0) } {
                0 => Ok(d),
                s => Err(format!("cuInit: CUDA error {s}")),
            }
        })
        .as_ref()
        .map_err(|e| Error::Load(e.clone()))
}

fn check(call: &'static str, status: c_int) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    let mut text: *const c_char = std::ptr::null();
    // SAFETY: the driver is loaded (a status came from it); cuGetErrorString gives a static string.
    let text = unsafe {
        match driver() {
            Ok(d) if (d.cuGetErrorString)(status, &mut text) == 0 && !text.is_null() => CStr::from_ptr(text).to_string_lossy().into_owned(),
            _ => format!("CUDA error {status}"),
        }
    };
    Err(Error::Cuda { call, status, text })
}

mod sealed {
    pub trait Sealed {}
}

/// Plain data for a copy to or from the GPU: every bit pattern a value, no padding (sealed: these types alone).
pub trait Plain: Copy + sealed::Sealed {}

macro_rules! plain {
    ($($t:ty),*) => {$(impl sealed::Sealed for $t {} impl Plain for $t {})*};
}
plain!(u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);

/// The driver's CUDA version, e.g. 13000.
pub fn driver_version() -> Result<i32> {
    let d = driver()?;
    let mut v = 0;
    // SAFETY: a host out-pointer.
    check("cuDriverGetVersion", unsafe { (d.cuDriverGetVersion)(&mut v) })?;
    Ok(v)
}

/// The GPUs the driver sees.
pub fn device_count() -> Result<i32> {
    let d = driver()?;
    let mut n = 0;
    // SAFETY: a host out-pointer.
    check("cuDeviceGetCount", unsafe { (d.cuDeviceGetCount)(&mut n) })?;
    Ok(n)
}

/// A GPU's primary context, current on the thread that made it (the context
/// the library's CUDA runtime takes for that device): the context of its
/// memory and streams, released when dropped; kept on that thread.
pub struct Context {
    device: c_int,
    _here: PhantomData<*const ()>, // neither Send nor Sync: current on its own thread alone
}

impl Context {
    /// GPU `ordinal`'s primary context, made current on this thread.
    pub fn new(ordinal: i32) -> Result<Context> {
        let d = driver()?;
        let (mut device, mut ctx) = (0, std::ptr::null_mut());
        // SAFETY: host out-pointers; the context retained is released in drop.
        unsafe {
            check("cuDeviceGet", (d.cuDeviceGet)(&mut device, ordinal))?;
            check("cuDevicePrimaryCtxRetain", (d.cuDevicePrimaryCtxRetain)(&mut ctx, device))?;
            let c = Context { device, _here: PhantomData };
            check("cuCtxSetCurrent", (d.cuCtxSetCurrent)(ctx))?;
            Ok(c)
        }
    }

    /// The GPU's name, e.g. "NVIDIA GeForce RTX 4080 SUPER".
    pub fn name(&self) -> Result<String> {
        let d = driver()?;
        let mut b = [0 as c_char; 256];
        // SAFETY: a host buffer of the length passed.
        check("cuDeviceGetName", unsafe { (d.cuDeviceGetName)(b.as_mut_ptr(), b.len() as c_int, self.device) })?;
        // SAFETY: NUL-terminated within the buffer.
        Ok(unsafe { CStr::from_ptr(b.as_ptr()) }.to_string_lossy().into_owned())
    }

    /// Its compute capability, e.g. (8, 9).
    pub fn capability(&self) -> Result<(i32, i32)> {
        let d = driver()?;
        let (mut major, mut minor) = (0, 0);
        // SAFETY: host out-pointers; 75 and 76: CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, _MINOR.
        unsafe {
            check("cuDeviceGetAttribute", (d.cuDeviceGetAttribute)(&mut major, 75, self.device))?;
            check("cuDeviceGetAttribute", (d.cuDeviceGetAttribute)(&mut minor, 76, self.device))?;
        }
        Ok((major, minor))
    }

    /// Everything queued on the context's streams done.
    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: no arguments.
        check("cuCtxSynchronize", unsafe { (driver()?.cuCtxSynchronize)() })
    }

    /// `bytes` of device memory (uninitialized; at least one byte is taken).
    pub fn alloc(&self, bytes: usize) -> Result<Buffer<'_>> {
        let mut ptr = 0u64;
        // SAFETY: a host out-pointer.
        check("cuMemAlloc", unsafe { (driver()?.cuMemAlloc_v2)(&mut ptr, bytes.max(1)) })?;
        Ok(Buffer { ptr, bytes, _context: PhantomData })
    }

    /// Device memory holding `data`'s bytes.
    pub fn upload<T: Plain>(&self, data: &[T]) -> Result<Buffer<'_>> {
        let mut b = self.alloc(std::mem::size_of_val(data))?;
        b.write(data)?;
        Ok(b)
    }

    /// A stream of its own (the driver's default flags), destroyed when dropped.
    pub fn stream(&self) -> Result<OwnedStream<'_>> {
        let mut s = Stream::DEFAULT;
        // SAFETY: a host out-pointer.
        check("cuStreamCreate", unsafe { (driver()?.cuStreamCreate)(&mut s, 0) })?;
        Ok(OwnedStream { stream: s, _context: PhantomData })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if let Ok(d) = driver() {
            // SAFETY: retained in new.
            unsafe { (d.cuDevicePrimaryCtxRelease_v2)(self.device) };
        }
    }
}

/// Device memory of a context, freed when dropped.
pub struct Buffer<'c> {
    ptr: u64,
    bytes: usize,
    _context: PhantomData<&'c Context>,
}

impl Buffer<'_> {
    /// Its address, as a pointer to T (for the library's calls).
    pub fn ptr<T>(&self) -> *mut T {
        self.ptr as usize as *mut T
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    fn fits(&self, n: usize, call: &'static str) -> Result<()> {
        if n <= self.bytes {
            Ok(())
        } else {
            Err(Error::Cuda { call, status: crate::INVALID_VALUE, text: format!("{n} bytes, past the buffer's {}", self.bytes) })
        }
    }

    /// `data` copied to its start (synchronously).
    pub fn write<T: Plain>(&mut self, data: &[T]) -> Result<()> {
        let n = std::mem::size_of_val(data);
        self.fits(n, "cuMemcpyHtoD")?;
        // SAFETY: n bytes of host memory to n of this buffer's.
        check("cuMemcpyHtoD", unsafe { (driver()?.cuMemcpyHtoD_v2)(self.ptr, data.as_ptr() as *const c_void, n) })
    }

    /// Its first bytes copied into `out` (synchronously, after the work queued before on the default stream).
    pub fn read<T: Plain>(&self, out: &mut [T]) -> Result<()> {
        let n = std::mem::size_of_val(out);
        self.fits(n, "cuMemcpyDtoH")?;
        // SAFETY: n of this buffer's bytes to n of host memory; T is Plain: any bytes a value of it.
        check("cuMemcpyDtoH", unsafe { (driver()?.cuMemcpyDtoH_v2)(out.as_mut_ptr() as *mut c_void, self.ptr, n) })
    }

    /// Every byte zero (a product's done counters before its first call).
    pub fn zero(&mut self) -> Result<()> {
        // SAFETY: the buffer's own bytes.
        check("cuMemsetD8", unsafe { (driver()?.cuMemsetD8_v2)(self.ptr, 0, self.bytes) })
    }
}

impl Drop for Buffer<'_> {
    fn drop(&mut self) {
        if let Ok(d) = driver() {
            // SAFETY: allocated by alloc, freed once.
            unsafe { (d.cuMemFree_v2)(self.ptr) };
        }
    }
}

/// A stream of a context, destroyed when dropped.
pub struct OwnedStream<'c> {
    stream: Stream,
    _context: PhantomData<&'c Context>,
}

impl OwnedStream<'_> {
    /// Its handle, for the library's calls.
    pub fn handle(&self) -> Stream {
        self.stream
    }

    /// The work queued on it done.
    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: a stream made by Context::stream.
        check("cuStreamSynchronize", unsafe { (driver()?.cuStreamSynchronize)(self.stream) })
    }
}

impl Drop for OwnedStream<'_> {
    fn drop(&mut self) {
        if let Ok(d) = driver() {
            // SAFETY: made by Context::stream, destroyed once.
            unsafe { (d.cuStreamDestroy_v2)(self.stream) };
        }
    }
}
