//! The runtime context (ROADMAP R16; docs/design/runtime.md section 4): what one served model owns of the state that was the process's, so that a process can serve more than one. A
//! scheduler writes its figures here, its sequences' tokens and ends are counted here, the first failure of its engine is kept here, and what `/v1/glyd/info` says of it is here; the listener
//! that answers for the model reads them from the same context (`server::listen_in`, `sched::Sched::in_context`).
//!
//! A program that serves one model names none: everything uses the process's own (`Context::process`), as it did when these were statics. What is one a process by its nature stays so: the
//! driver, the kernels' code, the CPU's kernels and cores, the settings read from the environment, the signals.

use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, Mutex, OnceLock};

pub struct Context {
    /// The scheduler's figures as its last round left them (`sched::Sched::gauges`' order; `sched::stats_json_of`).
    pub sched: [AtomicU64; crate::sched::FIGURES],
    /// The requests, tokens and ends counted for `/metrics` (src/ops.rs).
    pub figures: crate::ops::Figures,
    /// The engine's first failure (when, and what): nothing is sure of the GPU after one, so `/v1/glyd/info` says the server has failed from then on.
    pub failed: Mutex<Option<(f64, String)>>,
    /// What `/v1/glyd/info` says that does not change while the server runs (`serve` sets it once the model is loaded).
    pub info: OnceLock<crate::server::Info>,
    /// Whether a request with no cache scope of its own (`cache_salt`) is told, in its usage, how many of its prompt's tokens were found in the cache. Off unless the server's
    /// operator says so (GLYD_ENGINE_CACHED_USAGE=1: every caller is one user): the cache is one for all such requests, and a count would let one caller learn how another's
    /// prompt begins. Not decided by the listener's address: a server on a loopback address may answer callers from anywhere, behind `glyd service` or another proxy.
    pub tell_cached: AtomicBool,
}

impl Context {
    /// A context of its own: a second model of the process, or a test's server.
    pub fn new() -> Arc<Context> {
        Arc::new(Context {
            sched: [const { AtomicU64::new(0) }; crate::sched::FIGURES],
            figures: crate::ops::Figures::new(),
            failed: Mutex::new(None),
            info: OnceLock::new(),
            tell_cached: AtomicBool::new(std::env::var_os("GLYD_ENGINE_CACHED_USAGE").is_some_and(|v| v == "1")),
        })
    }

    /// The process's own: what a program that serves one model uses, and every caller that names no other.
    pub fn process() -> &'static Arc<Context> {
        static P: OnceLock<Arc<Context>> = OnceLock::new();
        P.get_or_init(Context::new)
    }

    /// The engine failed (`what`): kept if it is the first.
    pub fn fail(&self, what: &str) {
        let mut f = self.failed.lock().unwrap_or_else(|e| e.into_inner());
        if f.is_none() {
            *f = Some((crate::monitor::clock(), what.to_string()));
        }
    }
}
