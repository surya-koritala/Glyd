//! The same interface as `mmap.rs` for the systems with no `mmap` (Windows): a file's bytes held in memory instead of mapped. A read-only input is read
//! whole; a read-write table is read whole and written back to its file by `sync` and when it is dropped, as a shared mapping's dirty pages reach the file.
use std::fs::File;
use std::io::{Read, Result, Seek, SeekFrom, Write};
use std::path::Path;

/// A file's bytes in memory; read-write when it has the file to write them back to.
pub struct Mapping {
    bytes: Vec<u8>,
    file: Option<File>,
}

impl Mapping {
    /// The whole of `file` (`len` bytes), read-write: `sync` and drop write it back.
    pub fn read_write(file: &File, len: usize) -> Result<Mapping> {
        let mut file = file.try_clone()?;
        file.seek(SeekFrom::Start(0))?;
        let mut bytes = vec![0u8; len];
        file.read_exact(&mut bytes)?;
        Ok(Mapping { bytes, file: Some(file) })
    }

    /// The whole of `path`, read-only; an empty file is no bytes.
    pub fn read_only(path: &Path) -> Result<Mapping> {
        Ok(Mapping { bytes: std::fs::read(path)?, file: None })
    }

    /// `read_only`: the bytes are in memory already.
    pub fn read_only_populated(path: &Path) -> Result<Mapping> {
        Self::read_only(path)
    }

    /// Nothing to read ahead.
    pub fn will_need(&self) {}

    /// Nothing to fault in.
    pub fn populate(&mut self) {}

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }

    /// Write the bytes to the file.
    pub fn sync(&self) {
        if let Some(mut f) = self.file.as_ref() {
            // (`&File` writes; the clone shares the cursor with the caller's file, so seek first)
            let _ = f.seek(SeekFrom::Start(0)).and_then(|_| f.write_all(&self.bytes)).and_then(|_| f.flush());
        }
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        self.sync();
    }
}

/// A buffer of `len` zero bytes.
pub struct Anon {
    bytes: Vec<u8>,
}

impl Anon {
    pub fn new(len: usize) -> Result<Anon> {
        Ok(Anon { bytes: vec![0u8; len] })
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.bytes
    }
}

/// No huge pages to ask for.
pub fn huge_hint(_buf: &mut [u8]) {}
