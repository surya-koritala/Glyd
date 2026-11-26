//! A file mapped read-only into memory: its pages are the system's page cache's, not pinned, and given back under memory pressure (the expert cache's host tier for a directory that does not fit in the memory the
//! computer can spare; src/offload.rs `HostMem`).

use crate::ffi::{err, Error, Result};
use std::ffi::c_void;
#[cfg(unix)]
use std::os::unix::io::AsRawFd;
use std::path::Path;

#[cfg(unix)]
extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
    fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
}

// (macOS has no posix_fadvise)
#[cfg(target_os = "linux")]
extern "C" {
    fn posix_fadvise(fd: i32, off: i64, len: i64, advice: i32) -> i32;
}

#[cfg(unix)]
const PROT_READ: i32 = 1;
#[cfg(unix)]
const MAP_PRIVATE: i32 = 2;
#[cfg(unix)]
const MADV_RANDOM: i32 = 1;
#[cfg(unix)]
const MADV_WILLNEED: i32 = 3;

/// A file's bytes at an address, unmapped when dropped; or bytes of pinned host memory (`Mapping::pinned`: a copy the GPU's copies read by DMA), freed when dropped.
pub struct Mapping {
    pub ptr: *mut c_void,
    pub len: usize,
    pinned: Option<crate::dev::Pinned>,
}

// SAFETY: the mapping is read-only and lives until dropped; the pointer is only read through.
unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}

impl Mapping {
    pub fn open(path: &Path) -> Result<Mapping> {
        let f = std::fs::File::open(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let len = f.metadata().map_err(|e| Error(format!("{}: {e}", path.display())))?.len() as usize;
        if len == 0 {
            return err(format!("{}: an empty file", path.display()));
        }
        // SAFETY: a read-only private mapping of a file we hold open; the mapping outlives the descriptor.
        #[cfg(unix)]
        let ptr = unsafe { mmap(std::ptr::null_mut(), len, PROT_READ, MAP_PRIVATE, f.as_raw_fd(), 0) };
        #[cfg(unix)]
        if ptr as usize == usize::MAX {
            return err(format!("{}: cannot be mapped: {}", path.display(), std::io::Error::last_os_error()));
        }
        #[cfg(windows)]
        let ptr = crate::os::win::map_read(&f).map_err(|e| Error(format!("{}: cannot be mapped: {e}", path.display())))?;
        Ok(Mapping { ptr, len, pinned: None })
    }

    /// `len` bytes of pinned host memory (cuMemHostAlloc, the context current), to be filled by the caller.
    pub fn pinned(len: usize) -> Result<Mapping> {
        let p = crate::dev::Pinned::new(len)?;
        Ok(Mapping { ptr: p.ptr, len, pinned: Some(p) })
    }

    /// Whether the bytes are pinned host memory.
    pub fn is_pinned(&self) -> bool {
        self.pinned.is_some()
    }

    /// The system told it need not keep `path`'s pages in its cache (posix_fadvise DONTNEED): a file whose bytes were copied elsewhere is not held twice.
    pub fn drop_cache(path: &Path) {
        // (Windows and macOS: no such advice; the system gives the pages back under pressure)
        #[cfg(not(target_os = "linux"))]
        let _ = path;
        #[cfg(target_os = "linux")]
        if let Ok(f) = std::fs::File::open(path) {
            // SAFETY: an advice about a file we hold open; it changes no byte.
            unsafe { posix_fadvise(f.as_raw_fd(), 0, 0, 4) };
        }
    }

    /// The page tables of `len` bytes from `offset` filled now (MADV_POPULATE_READ, Linux 5.14 on: the pages read in where they are not in memory), so that the first reads of them take no page faults; whether the system did.
    #[cfg(unix)]
    pub fn populate(&self, offset: usize, len: usize) -> bool {
        const MADV_POPULATE_READ: i32 = 22;
        let page = 4096;
        let start = offset / page * page;
        let end = (offset + len).min(self.len);
        // SAFETY: a range inside the mapping, page aligned at its start; the pages are only read in.
        end > start && unsafe { madvise((self.ptr as *mut u8).add(start) as *mut c_void, end - start, MADV_POPULATE_READ) } == 0
    }

    /// The same asked of Windows: a prefetch of the pages (PrefetchVirtualMemory), not a wait for them; whether the system took the request.
    #[cfg(windows)]
    pub fn populate(&self, offset: usize, len: usize) -> bool {
        let page = 4096;
        let start = offset / page * page;
        let end = (offset + len).min(self.len);
        // SAFETY: a range inside the mapping, page aligned at its start; the pages are only read in.
        end > start && unsafe { crate::os::win::prefetch((self.ptr as *mut u8).add(start) as *mut c_void, end - start) }
    }

    /// Asks the system to read `len` bytes from `offset` ahead of their use.
    pub fn will_need(&self, offset: usize, len: usize) {
        let page = 4096;
        let start = offset / page * page;
        let end = (offset + len).min(self.len);
        if end > start {
            // SAFETY: a range inside the mapping, page aligned at its start; an advice only.
            #[cfg(unix)]
            unsafe {
                madvise((self.ptr as *mut u8).add(start) as *mut c_void, end - start, MADV_WILLNEED)
            };
            #[cfg(windows)]
            unsafe {
                crate::os::win::prefetch((self.ptr as *mut u8).add(start) as *mut c_void, end - start)
            };
        }
    }
}

impl Mapping {
    /// Tells the system that `len` bytes from `offset` are read at random (rows of a table): a page not in memory is read alone, without the read-ahead that would fill the page cache with its neighbours.
    pub fn random(&self, offset: usize, len: usize) {
        let page = 4096;
        let start = offset / page * page;
        let end = (offset + len).min(self.len);
        if end > start {
            // SAFETY: a range inside the mapping, page aligned at its start; an advice only.
            #[cfg(unix)]
            unsafe {
                madvise((self.ptr as *mut u8).add(start) as *mut c_void, end - start, MADV_RANDOM)
            };
            // ponytail: Windows has no random-access advice for a view; if a table's row reads pull its neighbours into the page cache and evict the pack (Linux's 2026-10-11 case), read those rows with ReadFile instead
            #[cfg(windows)]
            let _ = (start, end);
        }
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        if self.pinned.is_none() {
            // SAFETY: mapped by `open`, unmapped once.
            #[cfg(unix)]
            unsafe {
                munmap(self.ptr, self.len)
            };
            #[cfg(windows)]
            unsafe {
                crate::os::win::unmap(self.ptr)
            };
        }
    }
}
