//! A file mapped into memory through libc's `mmap` (the library carries
//! no crate for it): read-only for an input the CLI compresses without
//! copying it first, read-write for the store's fingerprint table.
use std::fs::File;
use std::io::{Error, ErrorKind, Result};
use std::path::Path;

/// A file mapped into memory (libc's mmap; the library carries no crate
/// for it): read-write for the table, read-only for an input file the
/// CLI compresses without copying it first (`Mapping::read_only`).
pub struct Mapping {
    ptr: *mut u8,
    len: usize,
    /// The thread faulting the pages in (`populate`), joined on drop so
    /// it never outlives the mapping.
    populate: Option<std::thread::JoinHandle<()>>,
}

extern "C" {
    fn mmap(addr: *mut std::ffi::c_void, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut std::ffi::c_void;
    fn munmap(addr: *mut std::ffi::c_void, len: usize) -> i32;
    fn msync(addr: *mut std::ffi::c_void, len: usize, flags: i32) -> i32;
    fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
}
/// MADV_WILLNEED on Linux and macOS alike.
const MADV_WILLNEED: i32 = 3;
/// MADV_POPULATE_READ (Linux 5.14+): fault the pages in now. Elsewhere
/// it fails and the pages are touched instead.
const MADV_POPULATE_READ: i32 = 22;
const PROT_READ: i32 = 1;
const PROT_WRITE: i32 = 2;
const MAP_SHARED: i32 = 1;
/// Linux: fault the pages in at mmap time.
const MAP_POPULATE: i32 = 0x8000;
const MAP_PRIVATE: i32 = 2;
#[cfg(target_os = "linux")]
const MAP_ANONYMOUS: i32 = 0x20;
#[cfg(not(target_os = "linux"))]
const MAP_ANONYMOUS: i32 = 0x1000;
/// Linux: back the range with transparent huge pages when it can.
const MADV_HUGEPAGE: i32 = 14;
const HUGE: usize = 2 << 20;

/// An anonymous buffer of `len` zero bytes, aligned to 2 MB and, on
/// Linux, asked to be huge pages: an output buffer of hundreds of MB
/// costs a few hundred faults instead of tens of thousands, and no
/// memset (fresh pages are zero already). A `Vec` of the same size
/// zero-filled and 4 KB-faulted was most of a decode's CLI time.
pub struct Anon {
    map: *mut u8,
    map_len: usize,
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for Anon {}
unsafe impl Sync for Anon {}

impl Anon {
    pub fn new(len: usize) -> Result<Anon> {
        let map_len = len.max(1) + HUGE;
        let map = unsafe { mmap(std::ptr::null_mut(), map_len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0) };
        if map as isize == -1 {
            return Err(Error::new(ErrorKind::Other, "mmap failed"));
        }
        let ptr = ((map as usize + HUGE - 1) & !(HUGE - 1)) as *mut u8;
        if cfg!(target_os = "linux") && len >= HUGE {
            unsafe { madvise(ptr as *mut std::ffi::c_void, len & !(HUGE - 1), MADV_HUGEPAGE) };
        }
        Ok(Anon { map: map as *mut u8, map_len, ptr, len })
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

/// Ask for huge pages under a large buffer not yet touched (Linux;
/// elsewhere nothing): a zeroed allocation of hundreds of MB is then
/// faulted in 2 MB at a time as the decoder fills it.
pub fn huge_hint(buf: &mut [u8]) {
    if cfg!(target_os = "linux") && buf.len() >= 2 * HUGE {
        let start = (buf.as_ptr() as usize + HUGE - 1) & !(HUGE - 1);
        let end = (buf.as_ptr() as usize + buf.len()) & !(HUGE - 1);
        if end > start {
            unsafe { madvise(start as *mut std::ffi::c_void, end - start, MADV_HUGEPAGE) };
        }
    }
}

impl Drop for Anon {
    fn drop(&mut self) {
        unsafe {
            munmap(self.map as *mut _, self.map_len);
        }
    }
}
const MS_SYNC: i32 = 0x10;

impl Mapping {
    /// The whole of `file` (`len` bytes), read-write and shared: writes
    /// reach the file.
    pub fn read_write(file: &File, len: usize) -> Result<Mapping> {
        Self::map(file, len, PROT_READ | PROT_WRITE)
    }

    /// The whole of `path`, read-only; an empty file maps to no bytes.
    pub fn read_only(path: &Path) -> Result<Mapping> {
        Self::read_only_with(path, 0)
    }

    /// `read_only` with every page faulted in before it returns (Linux's
    /// MAP_POPULATE; elsewhere the helper thread), for a pass that would
    /// otherwise fault them in on every thread it runs on.
    pub fn read_only_populated(path: &Path) -> Result<Mapping> {
        let mut m = Self::read_only_with(path, if cfg!(target_os = "linux") { MAP_POPULATE } else { 0 })?;
        if !cfg!(target_os = "linux") {
            m.populate();
        }
        Ok(m)
    }

    fn read_only_with(path: &Path, extra_flags: i32) -> Result<Mapping> {
        let file = File::open(path)?;
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Ok(Mapping { ptr: std::ptr::NonNull::<u8>::dangling().as_ptr(), len: 0, populate: None });
        }
        Self::map_with(&file, len, PROT_READ, extra_flags)
    }

    /// Ask for the whole mapping to be read ahead: a pass that touches
    /// it on a few threads is otherwise paced by page faults.
    pub fn will_need(&self) {
        if self.len > 0 {
            unsafe { madvise(self.ptr as *mut std::ffi::c_void, self.len, MADV_WILLNEED) };
        }
    }

    /// Fault the pages in on a helper thread, front to back, so a pass
    /// that walks the mapping on one core is not paced by page faults:
    /// the thread stays ahead of a parse (a cached file populates at
    /// several GB/s). Joined when the mapping is dropped.
    pub fn populate(&mut self) {
        if self.len == 0 || self.populate.is_some() {
            return;
        }
        let (ptr, len) = (self.ptr as usize, self.len);
        self.populate = std::thread::Builder::new().name("populate".into()).spawn(move || {
            const STEP: usize = 8 << 20;
            let mut at = 0;
            while at < len {
                let n = STEP.min(len - at);
                // SAFETY: within the mapping, which outlives this thread
                // (joined in `Drop`).
                let r = unsafe { madvise((ptr + at) as *mut std::ffi::c_void, n, MADV_POPULATE_READ) };
                if r != 0 {
                    let mut p = at;
                    while p < at + n {
                        unsafe { std::ptr::read_volatile((ptr + p) as *const u8) };
                        p += 4096;
                    }
                }
                at += n;
            }
        }).ok();
    }

    fn map(file: &File, len: usize, prot: i32) -> Result<Mapping> {
        Self::map_with(file, len, prot, 0)
    }

    fn map_with(file: &File, len: usize, prot: i32, extra_flags: i32) -> Result<Mapping> {
        use std::os::unix::io::AsRawFd;
        let ptr = unsafe { mmap(std::ptr::null_mut(), len, prot, MAP_SHARED | extra_flags, file.as_raw_fd(), 0) };
        if ptr as isize == -1 {
            return Err(Error::new(ErrorKind::Other, "mmap failed"));
        }
        Ok(Mapping { ptr: ptr as *mut u8, len, populate: None })
    }

    pub fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
    /// Write the mapping's dirty pages to the file.
    pub fn sync(&self) {
        unsafe {
            msync(self.ptr as *mut _, self.len, MS_SYNC);
        }
    }
}

// A mapping is a buffer at an address: reading it from any thread is
// sound (a read-write mapping's `bytes_mut` still needs `&mut self`),
// and it is unmapped only when the last reference is gone.
unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}

impl Drop for Mapping {
    fn drop(&mut self) {
        if let Some(t) = self.populate.take() {
            let _ = t.join();
        }
        if self.len > 0 {
            unsafe {
                munmap(self.ptr as *mut _, self.len);
            }
        }
    }
}
