//! What differs by system, in one place: positioned reads and writes of a file, /proc/meminfo's text, and Windows' own calls (kernel32) for the
//! engine's Windows paths (src/mapped.rs, src/ffi.rs). Linux and macOS take what they took before; only Windows takes the rest.

/// Positioned reads and writes: pread and pwrite on Unix (std's), ReadFile and WriteFile at an offset on Windows (`seek_read`, `seek_write`: these
/// also move the file's cursor, and no file of the engine's is read through its cursor while it is read at offsets).
#[cfg(unix)]
pub use std::os::unix::fs::FileExt;

#[cfg(windows)]
pub trait FileExt {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> std::io::Result<usize>;
    fn write_at(&self, buf: &[u8], offset: u64) -> std::io::Result<usize>;

    fn read_exact_at(&self, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
        while !buf.is_empty() {
            match self.read_at(buf, offset) {
                Ok(0) => return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "failed to fill whole buffer")),
                Ok(n) => {
                    buf = &mut buf[n..];
                    offset += n as u64;
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    fn write_all_at(&self, mut buf: &[u8], mut offset: u64) -> std::io::Result<()> {
        while !buf.is_empty() {
            match self.write_at(buf, offset) {
                Ok(0) => return Err(std::io::Error::new(std::io::ErrorKind::WriteZero, "failed to write whole buffer")),
                Ok(n) => {
                    buf = &buf[n..];
                    offset += n as u64;
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}

#[cfg(windows)]
impl FileExt for std::fs::File {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
        std::os::windows::fs::FileExt::seek_read(self, buf, offset)
    }
    fn write_at(&self, buf: &[u8], offset: u64) -> std::io::Result<usize> {
        std::os::windows::fs::FileExt::seek_write(self, buf, offset)
    }
}

/// The user's home folder: $HOME (on Windows, where it is seldom set, %USERPROFILE%); empty where neither is set.
pub fn home() -> std::ffi::OsString {
    #[cfg(windows)]
    if std::env::var_os("HOME").filter(|v| !v.is_empty()).is_none() {
        return std::env::var_os("USERPROFILE").unwrap_or_default();
    }
    std::env::var_os("HOME").unwrap_or_default()
}

/// A file whose ranges are written out of order made sparse first, where the system needs that (Windows: NTFS zero-fills up to each write
/// past the written end of a file that is not sparse, one write at a time: 8 downloaders made 46 MB/s of a link curl gets 290 from); nothing on Unix.
pub fn sparse(f: &std::fs::File) {
    #[cfg(windows)]
    win::set_sparse(f);
    #[cfg(not(windows))]
    let _ = f;
}

/// /proc/meminfo's text, for the parsers that read it; on Windows its MemTotal and MemAvailable lines, made from GlobalMemoryStatusEx. None where the system does not say.
pub fn meminfo() -> Option<String> {
    #[cfg(windows)]
    {
        let (total, avail) = win::memory()?;
        Some(format!("MemTotal: {} kB\nMemAvailable: {} kB\n", total / 1024, avail / 1024))
    }
    #[cfg(not(windows))]
    std::fs::read_to_string("/proc/meminfo").ok()
}

#[cfg(windows)]
#[allow(non_snake_case, clippy::upper_case_acronyms)]
pub mod win {
    use std::ffi::{c_char, c_void};

    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }

    #[repr(C)]
    struct MemoryRange {
        address: *mut c_void,
        bytes: usize,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
        fn AddDllDirectory(dir: *const u16) -> *mut c_void;
        pub fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
        fn CreateFileMappingW(file: *mut c_void, attributes: *mut c_void, protect: u32, size_high: u32, size_low: u32, name: *const u16) -> *mut c_void;
        fn MapViewOfFile(mapping: *mut c_void, access: u32, offset_high: u32, offset_low: u32, bytes: usize) -> *mut c_void;
        fn UnmapViewOfFile(address: *const c_void) -> i32;
        fn CloseHandle(handle: *mut c_void) -> i32;
        fn GetCurrentProcess() -> *mut c_void;
        fn PrefetchVirtualMemory(process: *mut c_void, entries: usize, ranges: *const MemoryRange, flags: u32) -> i32;
        fn GlobalMemoryStatusEx(status: *mut MemoryStatusEx) -> i32;
        pub fn GetCurrentProcessorNumber() -> u32;
        fn DeviceIoControl(
            device: *mut c_void,
            code: u32,
            input: *mut c_void,
            in_bytes: u32,
            output: *mut c_void,
            out_bytes: u32,
            returned: *mut u32,
            overlapped: *mut c_void,
        ) -> i32;
        fn K32GetProcessMemoryInfo(process: *mut c_void, counters: *mut [usize; 10], bytes: u32) -> i32;
    }

    /// The process's peak working set in bytes (PROCESS_MEMORY_COUNTERS on 64-bit Windows: cb and a count as the first word, then eight SIZE_Ts, the peak first).
    pub fn peak_working_set() -> Option<u64> {
        let mut c = [0usize; 10];
        // SAFETY: an 80-byte buffer for the 72-byte struct, its size given; the call fills it.
        (unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, 80) } != 0).then_some(c[1] as u64)
    }

    const LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR: u32 = 0x100;
    const LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: u32 = 0x1000; // (the program's folder, System32 and the folders AddDllDirectory added: never the current folder or PATH)

    /// A DLL loaded from the program's folder, System32 (the driver's nvcuda.dll) or %CUDA_PATH%\bin (cuBLAS), or from the absolute path given (its own folder
    /// then searched for what it needs): a bare name is never looked for in the current folder, where a planted DLL of that name would load instead.
    ///
    /// # Safety
    /// Loading a library runs its initialization.
    pub unsafe fn load_library(path: &str) -> *mut c_void {
        static DIRS: std::sync::Once = std::sync::Once::new();
        DIRS.call_once(|| {
            if let Some(bin) = std::env::var_os("CUDA_PATH").map(|p| std::path::PathBuf::from(p).join("bin")).filter(|p| p.is_absolute() && p.is_dir()) {
                let w: Vec<u16> = std::os::windows::ffi::OsStrExt::encode_wide(bin.as_os_str()).chain([0]).collect();
                AddDllDirectory(w.as_ptr());
            }
        });
        let w: Vec<u16> = path.encode_utf16().chain([0]).collect();
        let flags = LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | if std::path::Path::new(path).is_absolute() { LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR } else { 0 };
        LoadLibraryExW(w.as_ptr(), std::ptr::null_mut(), flags)
    }

    const PAGE_READONLY: u32 = 0x02;
    const FILE_MAP_READ: u32 = 0x04;

    /// The whole of an open file mapped read-only (the view keeps the mapping's section: its handle is closed here).
    pub fn map_read(f: &std::fs::File) -> std::io::Result<*mut c_void> {
        use std::os::windows::io::AsRawHandle;
        // SAFETY: a file handle we hold open; the section's handle is closed once the view holds it.
        unsafe {
            let section = CreateFileMappingW(f.as_raw_handle(), std::ptr::null_mut(), PAGE_READONLY, 0, 0, std::ptr::null());
            if section.is_null() {
                return Err(std::io::Error::last_os_error());
            }
            let view = MapViewOfFile(section, FILE_MAP_READ, 0, 0, 0);
            let why = std::io::Error::last_os_error();
            CloseHandle(section);
            if view.is_null() {
                return Err(why);
            }
            Ok(view)
        }
    }

    /// # Safety
    /// `address` is a view `map_read` gave, unmapped once.
    pub unsafe fn unmap(address: *mut c_void) {
        UnmapViewOfFile(address);
    }

    /// Asks the system to read `bytes` from `address` (inside a view) into memory ahead of their use; whether it took the request.
    ///
    /// # Safety
    /// The range lies inside a live view.
    pub unsafe fn prefetch(address: *mut c_void, bytes: usize) -> bool {
        let range = MemoryRange { address, bytes };
        PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0) != 0
    }

    #[repr(C)]
    struct TimeZoneInformation {
        bias: i32,
        standard_name: [u16; 32],
        standard_date: [u16; 8],
        standard_bias: i32,
        daylight_name: [u16; 32],
        daylight_date: [u16; 8],
        daylight_bias: i32,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetTimeZoneInformation(info: *mut TimeZoneInformation) -> u32;
    }

    /// The local time's offset from UTC now, in seconds (east positive), as `localtime_r`'s tm_gmtoff on Unix.
    pub fn utc_offset() -> i64 {
        let mut z = TimeZoneInformation {
            bias: 0,
            standard_name: [0; 32],
            standard_date: [0; 8],
            standard_bias: 0,
            daylight_name: [0; 32],
            daylight_date: [0; 8],
            daylight_bias: 0,
        };
        // SAFETY: a struct of TIME_ZONE_INFORMATION's layout, filled by the call.
        let state = unsafe { GetTimeZoneInformation(&mut z) };
        let bias = z.bias
            + match state {
                1 => z.standard_bias,
                2 => z.daylight_bias,
                _ => 0,
            };
        -(bias as i64) * 60
    }

    /// The file marked sparse (FSCTL_SET_SPARSE); a refusal leaves it as it was, only slower to fill out of order.
    pub fn set_sparse(f: &std::fs::File) {
        use std::os::windows::io::AsRawHandle;
        let mut got = 0u32;
        // SAFETY: a file handle we hold open; the control takes no buffers.
        unsafe { DeviceIoControl(f.as_raw_handle(), 0x0009_00C4, std::ptr::null_mut(), 0, std::ptr::null_mut(), 0, &mut got, std::ptr::null_mut()) };
    }

    /// Physical memory: the total and what is available, in bytes.
    pub fn memory() -> Option<(u64, u64)> {
        let mut s = MemoryStatusEx {
            length: std::mem::size_of::<MemoryStatusEx>() as u32,
            load: 0,
            total_phys: 0,
            avail_phys: 0,
            total_page_file: 0,
            avail_page_file: 0,
            total_virtual: 0,
            avail_virtual: 0,
            avail_extended_virtual: 0,
        };
        // SAFETY: a struct of the size its length says.
        (unsafe { GlobalMemoryStatusEx(&mut s) } != 0).then_some((s.total_phys, s.avail_phys))
    }
}
