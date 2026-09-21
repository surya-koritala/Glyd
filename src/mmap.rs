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
}

extern "C" {
    fn mmap(addr: *mut std::ffi::c_void, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut std::ffi::c_void;
    fn munmap(addr: *mut std::ffi::c_void, len: usize) -> i32;
    fn msync(addr: *mut std::ffi::c_void, len: usize, flags: i32) -> i32;
}
const PROT_READ: i32 = 1;
const PROT_WRITE: i32 = 2;
const MAP_SHARED: i32 = 1;
const MS_SYNC: i32 = 0x10;

impl Mapping {
    /// The whole of `file` (`len` bytes), read-write and shared: writes
    /// reach the file.
    pub fn read_write(file: &File, len: usize) -> Result<Mapping> {
        Self::map(file, len, PROT_READ | PROT_WRITE)
    }

    /// The whole of `path`, read-only; an empty file maps to no bytes.
    pub fn read_only(path: &Path) -> Result<Mapping> {
        let file = File::open(path)?;
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Ok(Mapping { ptr: std::ptr::NonNull::<u8>::dangling().as_ptr(), len: 0 });
        }
        Self::map(&file, len, PROT_READ)
    }

    fn map(file: &File, len: usize, prot: i32) -> Result<Mapping> {
        use std::os::unix::io::AsRawFd;
        let ptr = unsafe { mmap(std::ptr::null_mut(), len, prot, MAP_SHARED, file.as_raw_fd(), 0) };
        if ptr as isize == -1 {
            return Err(Error::new(ErrorKind::Other, "mmap failed"));
        }
        Ok(Mapping { ptr: ptr as *mut u8, len })
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

impl Drop for Mapping {
    fn drop(&mut self) {
        if self.len > 0 {
            unsafe {
                munmap(self.ptr as *mut _, self.len);
            }
        }
    }
}
