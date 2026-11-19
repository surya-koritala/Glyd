//! What the engine tells the program that started it, through the file `--status-file PATH` names: its modification time moves while the engine starts (a download, a packing, the KV cache's tables,
//! the load: every line or status it shows is a sign of life), `ready` is written when it is up (serving, or about to answer), and a refusal before that leaves its reason there, as one line, in
//! place of the terminal. The program that started the engine reads it: a start that ends in any way, or goes silent, before `ready` is run another way (vLLM), and what happens after `ready` is the
//! engine's own, said on the terminal and not started again. Nothing irreversible is done before `ready`: standard input is not read, no port is held.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime};

pub struct Status {
    path: OnceLock<PathBuf>,
    ready: AtomicBool,
    last: Mutex<Option<Instant>>,
}

/// This process's.
pub static STATUS: Status = Status::new();

impl Default for Status {
    fn default() -> Status {
        Status::new()
    }
}

impl Status {
    pub const fn new() -> Status {
        Status { path: OnceLock::new(), ready: AtomicBool::new(false), last: Mutex::new(None) }
    }

    /// The file the program that started the engine named.
    pub fn set(&self, path: PathBuf) {
        let _ = self.path.set(path);
    }

    /// The file, while the engine is not up: where a refusal leaves its reason.
    pub fn pending(&self) -> Option<&Path> {
        if self.ready.load(Ordering::Relaxed) {
            None
        } else {
            self.path.get().map(PathBuf::as_path)
        }
    }

    /// A sign of life: the file's modification time moved to now (at most once a second; nothing once the engine is up).
    pub fn beat(&self) {
        let Some(p) = self.pending() else { return };
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        if last.is_some_and(|t| t.elapsed() < Duration::from_secs(1)) {
            return;
        }
        *last = Some(Instant::now());
        let _ = std::fs::OpenOptions::new().write(true).open(p).and_then(|f| f.set_modified(SystemTime::now()));
    }

    /// The engine is up.
    pub fn ready(&self) {
        if let Some(p) = self.path.get() {
            let _ = std::fs::write(p, "ready\n");
        }
        self.ready.store(true, Ordering::Relaxed);
    }

    /// `what` left as the reason the engine stops before it is up, in one line: true where it was (a file is named and the engine is not up yet), and the terminal is left alone.
    pub fn refuse(&self, what: &str) -> bool {
        match self.pending() {
            Some(p) => std::fs::write(p, format!("{}\n", what.trim_end_matches('.').replace(['\n', '\r'], " "))).is_ok(),
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("glyd-engine-status-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d.join("status")
    }

    fn modified(p: &Path) -> SystemTime {
        std::fs::metadata(p).unwrap().modified().unwrap()
    }

    fn age(p: &Path) {
        let long_ago = SystemTime::now() - Duration::from_secs(3600);
        std::fs::OpenOptions::new().write(true).open(p).unwrap().set_modified(long_ago).unwrap();
    }

    /// Alive while it starts (the file's time moves, at most once a second), up (`ready`, and no more signs after it), and a reason left in one line before that and never after it.
    #[test]
    fn a_start_that_is_alive_then_up_or_stopped() {
        let p = temp("life");
        std::fs::write(&p, "").unwrap();
        let st = Status::new();
        // no file named: nothing is done, and nothing is said to a file
        st.beat();
        assert!(st.pending().is_none() && !st.refuse("a reason"));
        st.set(p.clone());
        assert_eq!(st.pending(), Some(p.as_path()));
        age(&p);
        st.beat();
        assert!(modified(&p) > SystemTime::now() - Duration::from_secs(60), "a beat moves the time");
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "", "and leaves what is in the file");
        age(&p);
        st.beat();
        assert!(modified(&p) < SystemTime::now() - Duration::from_secs(60), "not twice in a second");
        // a refusal before it is up: the reason, one line, no full stop
        assert!(st.refuse("Qwen3-8B does not fit this GPU: it needs about 14.2 GiB.\nA second line."));
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "Qwen3-8B does not fit this GPU: it needs about 14.2 GiB. A second line\n");
        // up: `ready`, and from then on no sign of life, and no reason in the file (a refusal is said where it always was)
        st.ready();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "ready\n");
        age(&p);
        st.beat();
        assert!(modified(&p) < SystemTime::now() - Duration::from_secs(60) && st.pending().is_none() && !st.refuse("too late"));
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "ready\n");
        std::fs::remove_dir_all(p.parent().unwrap()).unwrap();
    }

    #[test]
    fn a_beat_is_made_again_after_a_second() {
        let p = temp("again");
        std::fs::write(&p, "").unwrap();
        let st = Status::new();
        st.set(p.clone());
        st.beat();
        age(&p);
        std::thread::sleep(Duration::from_millis(1100));
        st.beat();
        assert!(modified(&p) > SystemTime::now() - Duration::from_secs(60));
        std::fs::remove_dir_all(p.parent().unwrap()).unwrap();
    }
}
