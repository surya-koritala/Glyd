//! The disk side of the SSD tier (Linear GLY-130; src/offload.rs `Tier`, src/moe_hb.rs `HbDisk`): a computer whose memory holds only some of a mixture-of-experts model's routed experts keeps the rest in the packed directory and
//! reads one into a slot of the host tier when the router asks for it. Three things are here: a file read that leaves the page cache alone, a pool of threads that makes many such reads at once, and the process's peak
//! resident memory.
//!
//! Why the page cache is left alone: an expert read twice would then be in memory twice, once in the cache's slot (pinned, 2.5 MB) and once in the kernel's page cache, so that on a computer with 32 GB the cache of the
//! files would crowd out the experts the budget was made for; and on a pod with a great deal of memory every read after the first would come from memory, and a measurement of the tier would measure that. Linux reads
//! the file with O_DIRECT (offsets, lengths and buffers in multiples of 4,096 bytes: `read_span` reads the blocks that cover the bytes asked for into an aligned buffer and hands back the bytes among them), macOS with F_NOCACHE
//! (no alignment is asked there; the same reads are made), and a file system that takes no direct reads (tmpfs, some network and overlay file systems) is read as any file is, each range's pages dropped from the cache after
//! the read (`posix_fadvise` DONTNEED, Linux's). NVMe wants a queue of reads to give its speed, not one at a time: `Pool` is a set of threads that live as long as the process and take the reads of a call (16 of them).

use crate::os::FileExt;
use std::alloc::{alloc, dealloc, Layout};
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

/// The multiple a direct read's offset, length and buffer are held to (a logical block of up to 4,096 bytes).
const ALIGN: usize = 4096;

/// O_DIRECT's value where this Linux has one (it differs by architecture); None where the build does not know it (reads are then buffered and dropped from the cache).
#[cfg(target_os = "linux")]
const O_DIRECT: Option<i32> = if cfg!(any(target_arch = "x86_64", target_arch = "x86")) {
    Some(0x4000)
} else if cfg!(any(target_arch = "aarch64", target_arch = "arm")) {
    Some(0x10000)
} else {
    None
};

const EINVAL: i32 = 22;

#[cfg(target_os = "linux")]
extern "C" {
    fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
}

#[cfg(target_os = "macos")]
extern "C" {
    // (fcntl is variadic: Apple's arm64 passes a variadic argument on the stack, so it is declared as it is)
    fn fcntl(fd: i32, cmd: i32, ...) -> i32;
}

/// A buffer at an address that is a multiple of `ALIGN`, grown as reads need it (a thread keeps one: the biggest read it has made).
pub struct Aligned {
    ptr: *mut u8,
    cap: usize,
}

// SAFETY: the buffer is owned bytes behind a pointer, used by one thread at a time (a thread-local, or moved whole).
unsafe impl Send for Aligned {}

impl Default for Aligned {
    fn default() -> Aligned {
        Aligned::new()
    }
}

impl Aligned {
    pub const fn new() -> Aligned {
        Aligned { ptr: std::ptr::null_mut(), cap: 0 }
    }

    /// `n` bytes at an aligned address (what the buffer held before is not kept where it has to grow).
    fn reserve(&mut self, n: usize) -> io::Result<&mut [u8]> {
        if n > self.cap {
            let layout = Layout::from_size_align(n.next_multiple_of(ALIGN), ALIGN).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;
            // SAFETY: a layout of a size above zero (`n` is above the capacity, which is 0 or more).
            let p = unsafe { alloc(layout) };
            if p.is_null() {
                return Err(io::Error::new(io::ErrorKind::OutOfMemory, format!("{} MiB for a read buffer", n >> 20)));
            }
            self.release();
            (self.ptr, self.cap) = (p, layout.size());
        }
        // SAFETY: the allocation holds `cap >= n` bytes, and `&mut self` is the only way to them.
        Ok(unsafe { std::slice::from_raw_parts_mut(self.ptr, n) })
    }

    fn release(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: allocated in `reserve` with this size and alignment, freed once.
            unsafe { dealloc(self.ptr, Layout::from_size_align_unchecked(self.cap, ALIGN)) };
            (self.ptr, self.cap) = (std::ptr::null_mut(), 0);
        }
    }
}

impl Drop for Aligned {
    fn drop(&mut self) {
        self.release();
    }
}

/// A file opened to be read without keeping what is read in the page cache (see the top of the file). `read_span` is for threads at once: a read at an offset (`pread`) shares no position.
pub struct DirectFile {
    path: PathBuf,
    file: File,
    // whether `file` was opened with O_DIRECT (its reads must be of whole blocks)
    direct: bool,
    // whether the file system refused a direct read: the file is then read through `plain`, as any is, and its pages dropped after
    refused: AtomicBool,
    plain: OnceLock<io::Result<File>>,
}

impl DirectFile {
    pub fn open(path: &Path) -> io::Result<DirectFile> {
        let (file, direct) = open_uncached(path)?;
        Ok(DirectFile { path: path.to_path_buf(), file, direct, refused: AtomicBool::new(false), plain: OnceLock::new() })
    }

    /// The `len` bytes at `offset` of the file, read into `bounce` and handed back from it (a direct read takes the blocks that cover them). A range past the end of the file is an error.
    pub fn read_span<'a>(&self, offset: u64, len: usize, bounce: &'a mut Aligned) -> io::Result<&'a [u8]> {
        if len == 0 {
            return Ok(&[]);
        }
        let (skip, room) = loop {
            let aligned = self.aligned();
            let (start, end) =
                if aligned { (offset & !(ALIGN as u64 - 1), (offset + len as u64).next_multiple_of(ALIGN as u64)) } else { (offset, offset + len as u64) };
            let (skip, room) = ((offset - start) as usize, (end - start) as usize);
            if self.fill(bounce.reserve(room)?, start, skip + len, aligned)? {
                if !aligned {
                    self.forget(start, skip + len);
                }
                break (skip, room);
            }
        };
        Ok(&bounce.reserve(room)?[skip..skip + len])
    }

    /// How the file is read now, in a word (it is known once a read has been made: a file system that takes no direct reads says so at the first): "direct" (O_DIRECT), "no-cache" (macOS) or "buffered" (pages dropped from the
    /// page cache after each read).
    pub fn mode(&self) -> &'static str {
        if self.aligned() {
            "direct"
        } else if cfg!(target_os = "macos") && !self.refused.load(Ordering::Relaxed) {
            "no-cache"
        } else {
            "buffered"
        }
    }

    /// Whether reads are of whole blocks now.
    fn aligned(&self) -> bool {
        self.direct && !self.refused.load(Ordering::Relaxed)
    }

    /// The file reads are made on: the one opened for them, or (a direct read refused) the plain one.
    fn reader(&self) -> io::Result<&File> {
        if self.refused.load(Ordering::Relaxed) {
            self.plain.get_or_init(|| File::open(&self.path)).as_ref().map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", self.path.display())))
        } else {
            Ok(&self.file)
        }
    }

    /// `buf` (`room` bytes) filled from `start` on until `need` bytes are in it; false where the file system refused a direct read (nothing is kept: the read is made again, plain).
    fn fill(&self, buf: &mut [u8], start: u64, need: usize, aligned: bool) -> io::Result<bool> {
        let file = self.reader()?;
        let mut got = 0usize;
        while got < need {
            match file.read_at(&mut buf[got..], start + got as u64) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        format!("{}: the file ends at {}, {} bytes before the end of the range read", self.path.display(), start + got as u64, need - got),
                    ))
                }
                Ok(n) => {
                    got += n;
                    // (a direct read goes on from a whole block: a short read that ends inside one is the file's end, which is short of the range)
                    if aligned && got < need && !got.is_multiple_of(ALIGN) {
                        return Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            format!("{}: the file ends at {}, {} bytes before the end of the range read", self.path.display(), start + got as u64, need - got),
                        ));
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) if aligned && e.raw_os_error() == Some(EINVAL) => {
                    self.refused.store(true, Ordering::Relaxed);
                    return Ok(false);
                }
                Err(e) => return Err(io::Error::new(e.kind(), format!("{}: {e}", self.path.display()))),
            }
        }
        Ok(true)
    }

    /// The pages of `[offset, offset + len)` dropped from the page cache (a read of a file not opened with O_DIRECT left them there); macOS reads with F_NOCACHE and keeps none.
    #[allow(unused_variables)]
    fn forget(&self, offset: u64, len: usize) {
        #[cfg(target_os = "linux")]
        if let Ok(f) = self.reader() {
            use std::os::fd::AsRawFd;
            // SAFETY: an open descriptor; POSIX_FADV_DONTNEED is 4; advice is advice (a failure leaves the pages where they are).
            unsafe { posix_fadvise(f.as_raw_fd(), offset as i64, len as i64, 4) };
        }
    }
}

/// `path` opened for reads that keep nothing in the page cache, and whether they must be of whole blocks.
#[allow(unused_variables)]
fn open_uncached(path: &Path) -> io::Result<(File, bool)> {
    #[cfg(target_os = "linux")]
    if let Some(flag) = O_DIRECT {
        use std::os::unix::fs::OpenOptionsExt;
        match std::fs::OpenOptions::new().read(true).custom_flags(flag).open(path) {
            Ok(f) => return Ok((f, true)),
            Err(e) if e.raw_os_error() == Some(EINVAL) => {}
            Err(e) => return Err(io::Error::new(e.kind(), format!("{}: {e}", path.display()))),
        }
    }
    let f = File::open(path).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;
        // SAFETY: an open descriptor; F_NOCACHE is 48 and its argument an int (a failure leaves reads cached, which is all it can cost).
        unsafe { fcntl(f.as_raw_fd(), 48, 1) };
    }
    Ok((f, false))
}

// ---- many reads at once -----------------------------------------------------------------------------------------------------------------------------------------------------------

type Task = Box<dyn FnOnce() + Send>;

/// A job of the pool: a read, its error if it fails.
pub type Job = Box<dyn FnOnce() -> io::Result<()> + Send>;

/// Threads that wait for reads, made once and kept (starting 16 threads for every layer of every token would cost more than the reads of a layer).
pub struct Pool {
    tx: Mutex<Sender<Task>>,
    threads: usize,
}

/// The reads a call makes at once: 16, a queue deep enough for an NVMe drive.
pub const DEPTH: usize = 16;

/// The process's pool, its threads started by the first call.
pub fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| Pool::new(DEPTH))
}

/// What a set of jobs under way counts down: the jobs left and the first error.
type Latch = (Mutex<(usize, Option<io::Error>)>, Condvar);

/// Jobs under way on the pool (`Pool::start`): `wait` returns when all are done, with the first error of them. One that is dropped waits too, since the jobs may be writing memory that the caller is about to free.
pub struct Pending {
    latch: Option<Arc<Latch>>,
}

impl Pending {
    pub fn wait(mut self) -> io::Result<()> {
        self.join()
    }

    fn join(&mut self) -> io::Result<()> {
        let Some(latch) = self.latch.take() else { return Ok(()) };
        let mut g = latch.0.lock().unwrap_or_else(|e| e.into_inner());
        while g.0 > 0 {
            g = latch.1.wait(g).unwrap_or_else(|e| e.into_inner());
        }
        g.1.take().map_or(Ok(()), Err)
    }
}

impl Drop for Pending {
    fn drop(&mut self) {
        let _ = self.join();
    }
}

impl Pool {
    pub fn new(threads: usize) -> Pool {
        let (tx, rx) = channel::<Task>();
        let rx = Arc::new(Mutex::new(rx));
        let mut started = 0;
        for i in 0..threads.max(1) {
            let rx = rx.clone();
            // (a thread that cannot be started leaves the pool with the ones that were)
            let made = std::thread::Builder::new().name(format!("glyd-disk-{i}")).spawn(move || loop {
                let task = rx.lock().map(|r| r.recv());
                match task {
                    Ok(Ok(t)) => t(),
                    _ => return,
                }
            });
            started += made.is_ok() as usize;
        }
        Pool { tx: Mutex::new(tx), threads: started }
    }

    /// `jobs` started on the pool's threads (on this thread, where the pool has none): they run while the caller does something else, and `Pending::wait` returns when they are done.
    pub fn start(&self, jobs: Vec<Job>) -> io::Result<Pending> {
        if jobs.is_empty() {
            return Ok(Pending { latch: None });
        }
        let latch: Arc<Latch> = Arc::new((Mutex::new((jobs.len(), None)), Condvar::new()));
        let tx = self.tx.lock().map_err(|_| io::Error::other("the disk readers are gone"))?;
        for job in jobs {
            let latch = latch.clone();
            let task: Task = Box::new(move || {
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)).unwrap_or_else(|_| Err(io::Error::other("a disk read panicked")));
                let mut g = latch.0.lock().unwrap_or_else(|e| e.into_inner());
                if let Err(e) = r {
                    g.1.get_or_insert(e);
                }
                g.0 -= 1;
                latch.1.notify_all();
            });
            if self.threads == 0 {
                task();
            } else {
                tx.send(task).map_err(|_| io::Error::other("the disk readers are gone"))?;
            }
        }
        Ok(Pending { latch: Some(latch) })
    }

    /// `jobs` run on the pool's threads, all done when this returns; the first error of them (a job that panics is an error too).
    pub fn run(&self, jobs: Vec<Job>) -> io::Result<()> {
        self.start(jobs)?.wait()
    }
}

// ---- the process's memory ---------------------------------------------------------------------------------------------------------------------------------------------------------

/// The most resident memory the process has had, in bytes (getrusage; None where the system does not say).
#[cfg(unix)]
pub fn peak_rss() -> Option<u64> {
    // struct rusage: two timevals, then ru_maxrss and thirteen more counts, all longs on a 64-bit system
    #[repr(C)]
    struct Rusage {
        times: [i64; 4],
        maxrss: i64,
        rest: [i64; 13],
    }
    extern "C" {
        fn getrusage(who: i32, usage: *mut Rusage) -> i32;
    }
    let mut u = std::mem::MaybeUninit::<Rusage>::zeroed();
    // SAFETY: RUSAGE_SELF is 0; the struct is the size glibc's, musl's and Darwin's have on a 64-bit system (144 bytes), filled by the call.
    let m = unsafe {
        if getrusage(0, u.as_mut_ptr()) != 0 {
            return None;
        }
        u.assume_init().maxrss
    };
    // (Linux counts in KiB, macOS in bytes)
    u64::try_from(m).ok().filter(|&m| m > 0).map(|m| if cfg!(target_os = "macos") { m } else { m * 1024 })
}

/// The most memory the process has had in its working set, in bytes (Windows: K32GetProcessMemoryInfo's peak).
#[cfg(windows)]
pub fn peak_rss() -> Option<u64> {
    crate::os::win::peak_working_set()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("glyd-engine-disk-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    struct Xs(u64);

    impl Xs {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// A direct read hands back the bytes asked for whatever the alignment of the range: a few bytes inside a block, ranges across blocks, a range of megabytes, the last bytes of the file (the blocks that cover them
    /// run past its end), and a range past the end is an error; the bounce buffer is reused by a smaller read after a bigger one.
    #[test]
    fn a_read_returns_the_bytes_asked_for_at_any_alignment() {
        let d = temp("read");
        let mut x = Xs(0x9E37_79B9_7F4A_7C15);
        let data: Vec<u8> = (0..3_000_017usize).map(|_| (x.next() >> 24) as u8).collect();
        std::fs::write(d.join("f"), &data).unwrap();
        let f = DirectFile::open(&d.join("f")).unwrap();
        let mut bounce = Aligned::new();
        for (off, len) in [
            (0usize, 1usize),
            (1, 4095),
            (4095, 2),
            (4096, 4096),
            (5, 1_000_000),
            (123_457, 2_494_464 / 2),
            (data.len() - 7, 7),
            (data.len() - 4097, 4097),
            (0, data.len()),
        ] {
            assert_eq!(f.read_span(off as u64, len, &mut bounce).unwrap(), &data[off..off + len], "{len} bytes at {off}");
        }
        for _ in 0..300 {
            let off = (x.next() % data.len() as u64) as usize;
            let len = 1 + (x.next() % (data.len() - off).min(300_000) as u64) as usize;
            assert_eq!(f.read_span(off as u64, len, &mut bounce).unwrap(), &data[off..off + len], "{len} bytes at {off}");
        }
        assert!(["direct", "no-cache", "buffered"].contains(&f.mode()), "{}", f.mode());
        assert!(f.read_span(0, 0, &mut bounce).unwrap().is_empty());
        let e = f.read_span(data.len() as u64 - 3, 4, &mut bounce).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::UnexpectedEof, "{e}");
        assert!(DirectFile::open(&d.join("missing")).is_err());
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// A file system that takes no direct reads (procfs refuses O_DIRECT at open) is read as any file is, and says so.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_file_system_without_direct_reads_is_read_buffered() {
        let want = std::fs::read("/proc/version").unwrap();
        let f = DirectFile::open(Path::new("/proc/version")).unwrap();
        let mut bounce = Aligned::new();
        assert_eq!(f.read_span(3, 10, &mut bounce).unwrap(), &want[3..13]);
        assert_eq!(f.mode(), "buffered");
    }

    /// Reads from several threads at once, on one file, each with a buffer of its own, are all the file's bytes.
    #[test]
    fn reads_from_many_threads_are_the_files() {
        let d = temp("threads");
        let mut x = Xs(0xC0FFEE);
        let data: Arc<Vec<u8>> = Arc::new((0..1_500_000usize).map(|_| (x.next() >> 32) as u8).collect());
        std::fs::write(d.join("f"), &*data).unwrap();
        let f = Arc::new(DirectFile::open(&d.join("f")).unwrap());
        let bad = Arc::new(Mutex::new(Vec::<String>::new()));
        let jobs: Vec<Job> = (0..64u64)
            .map(|i| {
                let (f, data, bad) = (f.clone(), data.clone(), bad.clone());
                Box::new(move || {
                    let mut bounce = Aligned::new();
                    let (off, len) = ((i * 7919 * 13) as usize % 1_000_000, 1 + (i as usize * 104_729) % 400_000);
                    if f.read_span(off as u64, len, &mut bounce)? != &data[off..off + len] {
                        bad.lock().unwrap().push(format!("{len} bytes at {off}"));
                    }
                    Ok(())
                }) as Job
            })
            .collect();
        pool().run(jobs).unwrap();
        assert!(bad.lock().unwrap().is_empty(), "{:?}", bad.lock().unwrap());
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// The pool does every job before it returns, however many there are, runs them on more than one thread, hands back the first error, and is not stopped by a job that panics.
    #[test]
    fn the_pool_runs_every_job_and_reports_a_failure() {
        use std::sync::atomic::AtomicUsize;
        let pool = Pool::new(4);
        let done = Arc::new(AtomicUsize::new(0));
        let threads = Arc::new(Mutex::new(std::collections::HashSet::new()));
        let jobs = |n: usize| -> Vec<Job> {
            (0..n)
                .map(|_| {
                    let (done, threads) = (done.clone(), threads.clone());
                    Box::new(move || {
                        std::thread::sleep(std::time::Duration::from_millis(2));
                        threads.lock().unwrap().insert(std::thread::current().id());
                        done.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    }) as Job
                })
                .collect()
        };
        pool.run(jobs(37)).unwrap();
        assert_eq!(done.load(Ordering::SeqCst), 37);
        assert!(threads.lock().unwrap().len() > 1, "the jobs ran on one thread");
        pool.run(Vec::new()).unwrap();
        let mut more = jobs(5);
        more.push(Box::new(|| Err(io::Error::other("a read failed"))));
        more.push(Box::new(|| panic!("a job")));
        let e = pool.run(more).unwrap_err();
        assert!(e.to_string().contains("a read failed") || e.to_string().contains("panicked"), "{e}");
        assert_eq!(done.load(Ordering::SeqCst), 42, "the jobs beside the failing ones ran");
        pool.run(jobs(3)).unwrap();
        assert_eq!(done.load(Ordering::SeqCst), 45, "the pool works after a failure");
        // jobs started and waited for later; a Pending that is dropped waits for its jobs
        let started = pool.start(jobs(6)).unwrap();
        started.wait().unwrap();
        assert_eq!(done.load(Ordering::SeqCst), 51);
        drop(pool.start(jobs(6)).unwrap());
        assert_eq!(done.load(Ordering::SeqCst), 57, "a Pending dropped waits for its jobs");
        // a pool whose threads could not be started runs the jobs on the thread that calls
        let none = Pool { tx: Mutex::new(channel::<Task>().0), threads: 0 };
        none.run(jobs(4)).unwrap();
        assert_eq!(done.load(Ordering::SeqCst), 61);
    }

    #[test]
    fn the_process_has_a_peak_resident_size() {
        let before = peak_rss().expect("a peak resident size");
        assert!(before > (1 << 20), "{before}");
        let big = vec![1u8; 64 << 20];
        assert!(big.iter().map(|&b| b as u64).sum::<u64>() > 0);
        assert!(peak_rss().unwrap() >= before);
    }
}
