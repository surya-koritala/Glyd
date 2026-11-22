//! What the terminal shows while things start (stderr: stdout is the answer's): lines, a status line that is replaced by the next, one download or work bar, and the plain-words refusal.
//! Text goes out cleaned of control characters (a model's text or a server's error must not move the cursor or set the title).

use std::io::{IsTerminal, Write};
use std::time::Instant;

/// The exit status of a refusal that says this engine cannot run this model on this computer, before it is up (a model of another kind, a GPU or a driver it has not run on, a model that
/// does not fit, a KV cache that refuses the model): the program that started it runs the model another way. Every other refusal is 1.
pub const CANNOT: i32 = 3;

/// The exit status of a server whose GPU's context was lost while it ran (a sticky device error, `ffi::Error::sticky`: an illegal address in a kernel, a launch that failed, the GPU gone):
/// nothing more runs on that context, so the server ends. The service sees the program gone and starts the model again at the next request for it (src/service.rs `Pool::acquire`).
pub const DEVICE_LOST: i32 = 4;

/// Why a command stops, in words, and what to do about it.
#[derive(Debug)]
pub struct Refusal {
    pub what: String,
    pub fix: String,
    /// The exit status: 1, or `CANNOT`.
    pub code: i32,
}

impl Refusal {
    pub fn new(what: impl Into<String>, fix: impl Into<String>) -> Refusal {
        Refusal { what: what.into(), fix: fix.into(), code: 1 }
    }

    /// A refusal of status `CANNOT`.
    pub fn unsupported(what: impl Into<String>, fix: impl Into<String>) -> Refusal {
        Refusal { code: CANNOT, ..Refusal::new(what, fix) }
    }
}

impl From<crate::ffi::Error> for Refusal {
    fn from(e: crate::ffi::Error) -> Refusal {
        Refusal::new(e.msg, "")
    }
}

impl From<crate::hub::HubError> for Refusal {
    fn from(e: crate::hub::HubError) -> Refusal {
        Refusal::new(e.to_string(), "")
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.what)?;
        if !self.fix.is_empty() {
            write!(f, ": {}", self.fix)?;
        }
        Ok(())
    }
}

pub type Done<T> = std::result::Result<T, Refusal>;

/// `text` without its control characters (the newline and the tab stay; ESC and the C1 range go, so no escape sequence survives).
pub fn clean(text: &str) -> String {
    text.chars().filter(|&c| c == '\n' || c == '\t' || !(c.is_control())).collect()
}

/// Bytes as GB, one decimal ("16.4 GB"), 1 GB = 10^9 bytes (what a disk and the Hub say).
pub fn gb(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1} GB", n as f64 / 1e9)
    } else {
        format!("{:.0} MB", n as f64 / 1e6)
    }
}

/// "6.9 of 16.4 GB": two sizes in the unit of the second, as short as a progress line wants.
pub fn of(done: u64, total: u64) -> String {
    if total >= 1_000_000_000 {
        format!("{:.1} of {:.1} GB", done as f64 / 1e9, total as f64 / 1e9)
    } else {
        format!("{:.0} of {:.0} MB", done as f64 / 1e6, total as f64 / 1e6)
    }
}

/// "6.9/16.4 GB" (the same as `of`, shorter).
pub fn of_short(done: u64, total: u64) -> String {
    if total >= 1_000_000_000 {
        format!("{:.1}/{:.1} GB", done as f64 / 1e9, total as f64 / 1e9)
    } else {
        format!("{:.0}/{:.0} MB", done as f64 / 1e6, total as f64 / 1e6)
    }
}

/// Bytes as GiB, one decimal, for what a GPU says ("22.5 GiB").
pub fn gib(n: u64) -> String {
    format!("{:.1} GiB", n as f64 / (1u64 << 30) as f64)
}

pub fn fmt_time(s: f64) -> String {
    let s = s.max(0.0) as u64;
    if s >= 60 {
        format!("{}m{:02}s", s / 60, s % 60)
    } else {
        format!("{s}s")
    }
}

#[cfg(target_os = "linux")]
fn columns() -> usize {
    #[repr(C)]
    #[derive(Default)]
    struct WinSize {
        rows: u16,
        cols: u16,
        x: u16,
        y: u16,
    }
    extern "C" {
        fn ioctl(fd: i32, req: u64, ...) -> i32;
    }
    let mut w = WinSize::default();
    // SAFETY: TIOCGWINSZ (0x5413) on stderr's descriptor with a winsize struct to fill.
    if unsafe { ioctl(2, 0x5413, &mut w as *mut WinSize) } == 0 && w.cols > 0 {
        return w.cols as usize;
    }
    std::env::var("COLUMNS").ok().and_then(|c| c.parse().ok()).unwrap_or(80)
}

#[cfg(not(target_os = "linux"))]
fn columns() -> usize {
    std::env::var("COLUMNS").ok().and_then(|c| c.parse().ok()).unwrap_or(80)
}

pub struct Ui {
    tty: bool,
    live: bool,
    quiet: bool,
}

impl Default for Ui {
    fn default() -> Ui {
        Ui::new()
    }
}

impl Ui {
    pub fn new() -> Ui {
        Ui { tty: std::io::stderr().is_terminal() && std::env::var("TERM").map_or(true, |t| t != "dumb"), live: false, quiet: false }
    }

    /// A Ui that prints nothing (a one-shot answer's progress where the caller wants only the answer).
    pub fn silent() -> Ui {
        Ui { tty: false, live: false, quiet: true }
    }

    pub fn tty(&self) -> bool {
        self.tty
    }

    fn write(&self, text: &str) {
        if self.quiet {
            return;
        }
        let mut e = std::io::stderr();
        let _ = e.write_all(text.as_bytes());
        let _ = e.flush();
    }

    fn clear(&mut self) {
        if self.live {
            self.write("\r\x1b[K");
            self.live = false;
        }
    }

    pub fn line(&mut self, text: &str) {
        crate::status::STATUS.beat();
        self.clear();
        self.write(&format!("{}\n", clean(text)));
    }

    pub fn note(&mut self, text: &str) {
        self.line(&format!("Note: {text}"));
    }

    /// A line the next one replaces (a terminal only; elsewhere nothing is printed until `line`).
    pub fn status(&mut self, text: &str) {
        crate::status::STATUS.beat(); // (a sign of life to the program that started the engine, on a terminal or not)
        if self.tty {
            let t: String = clean(text).chars().take(columns().saturating_sub(1)).collect();
            self.write(&format!("\r\x1b[K{t}"));
            self.live = true;
        }
    }

    /// The status line (a progress bar) taken off the terminal, where a refusal is said some other way.
    pub fn clear_live(&mut self) {
        self.clear();
    }

    pub fn refuse(&mut self, r: &Refusal) {
        self.line(&format!("glyd: {}.", r.what.trim_end_matches('.')));
        for l in r.fix.lines() {
            self.line(&format!("  {l}"));
        }
    }
}

/// One bar: what is done of the total, at what rate, how long to go (a line at each 10% where there is no terminal).
pub struct Bar {
    title: String,
    total: u64,
    done: u64,
    tenth: u64,
    ticks: Vec<(Instant, u64)>,
}

impl Bar {
    pub fn new(title: &str, total: u64) -> Bar {
        Bar { title: title.to_string(), total: total.max(1), done: 0, tenth: 0, ticks: Vec::new() }
    }

    pub fn update(&mut self, ui: &mut Ui, done: u64, final_: bool) {
        let now = Instant::now();
        self.done = done.min(self.total).max(self.done);
        self.ticks.retain(|t| now.duration_since(t.0).as_secs_f64() < 8.0);
        self.ticks.push((now, self.done));
        let first = self.ticks[0];
        let dt = now.duration_since(first.0).as_secs_f64();
        let rate = if dt > 0.0 { (self.done - first.1) as f64 / dt } else { 0.0 };
        let frac = self.done as f64 / self.total as f64;
        if ui.tty() {
            let width = 16usize;
            let tail = if self.done >= self.total && !final_ {
                " finishing...".to_string()
            } else {
                let speed = if rate > 1e5 { format!(" {:.0} MB/s", rate / 1e6) } else { String::new() };
                let left =
                    if rate > 1e5 && self.done < self.total { format!(" {} left", fmt_time((self.total - self.done) as f64 / rate)) } else { String::new() };
                speed + &left
            };
            let hashes = ((frac * width as f64) as usize).min(width);
            let rest = format!(" [{}{}] {:3.0}% {}{tail}", "#".repeat(hashes), " ".repeat(width - hashes), frac * 100.0, of_short(self.done, self.total));
            // the title gives way where the line would be wider than the terminal
            let room = columns().saturating_sub(1).saturating_sub(rest.chars().count());
            let title: String =
                if self.title.chars().count() > room { self.title.chars().take(room.saturating_sub(1)).collect::<String>() + "~" } else { self.title.clone() };
            ui.status(&format!("{title}{rest}"));
        } else if (frac * 10.0) as u64 > self.tenth {
            self.tenth = (frac * 10.0) as u64;
            ui.line(&format!("{}: {}% ({} of {})", self.title, self.tenth * 10, gb(self.done), gb(self.total)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_is_cleaned_and_sizes_are_said() {
        assert_eq!(clean("a\x1b[31mred\x1b[0m\tb\nc\u{9b}d\x07"), "a[31mred[0m\tb\ncd");
        assert_eq!(gb(16_381_470_720), "16.4 GB");
        assert_eq!(gb(15_000_000), "15 MB");
        assert_eq!(gib(24 << 30), "24.0 GiB");
        assert_eq!(of(6_900_000_000, 16_381_470_720), "6.9 of 16.4 GB");
        assert_eq!(of_short(6_900_000_000, 16_381_470_720), "6.9/16.4 GB");
        assert_eq!(of(120_000_000, 805_000_000), "120 of 805 MB");
        assert_eq!(fmt_time(75.0), "1m15s");
        assert_eq!(fmt_time(9.4), "9s");
    }

    #[test]
    fn a_refusal_reads_in_words() {
        let r = Refusal::new("the disk is full", "free 12 GB");
        assert_eq!(r.to_string(), "the disk is full: free 12 GB");
        let mut ui = Ui::silent();
        ui.refuse(&r);
    }
}
