//! GLY-188 worker side: one approved warm rangeworker stage serving its requests, at most its
//! route's `sessions` at once (GLY-190).
use crate::cluster::{Result, security};
use starlings_cli::{pipeline, routes::StageRoute};
use starlings_security::{
    Identity, Invitation, ListenerBudget, NodeStatus, PeerListener, PrivateDir, RegistryFence,
};
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

#[derive(Clone, serde::Serialize)]
pub struct Status {
    state: &'static str,
    error: Option<String>,
    requests: u64,
    failed_requests: u64,
    /// The engine's device report at prepare: name, free and total bytes.
    device: Option<(String, u64, u64)>,
    family: Option<String>,
}
pub struct Owner {
    stop: Arc<AtomicBool>,
    status: Arc<Mutex<Status>>,
    thread: Option<JoinHandle<()>>,
}
impl Owner {
    pub fn start(route: StageRoute, state: PathBuf, fence: RegistryFence) -> Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let status = Arc::new(Mutex::new(Status {
            state: "waiting_for_registry",
            error: None,
            requests: 0,
            failed_requests: 0,
            device: None,
            family: None,
        }));
        let (stopped, report) = (stop.clone(), status.clone());
        let thread = thread::Builder::new()
            .name("pipeline-stage".into())
            .spawn(move || {
                let result = supervise(&route, &state, &fence, &stopped, &report);
                if let (Err(error), Ok(mut report)) = (result, report.lock()) {
                    report.state = "failed";
                    report.error = Some(error);
                }
            })
            .map_err(|_| "THREAD_UNAVAILABLE")?;
        Ok(Self {
            stop,
            status,
            thread: Some(thread),
        })
    }
    pub fn status(&self) -> Result<Status> {
        self.status
            .lock()
            .map(|s| s.clone())
            .map_err(|_| "STAGE_STATE_FAILED".into())
    }
    pub fn announcement(&self) -> Arc<Mutex<Status>> {
        self.status.clone()
    }
    pub fn drain(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }
    pub fn join(&mut self) -> Result<()> {
        self.drain();
        if let Some(thread) = self.thread.take() {
            thread.join().map_err(|_| "STAGE_THREAD_FAILED")?;
        }
        Ok(())
    }
}
impl Drop for Owner {
    fn drop(&mut self) {
        let _ = self.join();
    }
}
fn set(status: &Mutex<Status>, state: &'static str) {
    if let Ok(mut report) = status.lock() {
        report.state = state;
    }
}

fn supervise(
    route: &StageRoute,
    state: &std::path::Path,
    fence: &RegistryFence,
    stop: &AtomicBool,
    status: &Mutex<Status>,
) -> Result<()> {
    let online = || fence.status().is_ok_and(|(_, s)| s == NodeStatus::Online);
    let until = Instant::now() + Duration::from_secs(15);
    while !stop.load(Ordering::Relaxed) && !online() && Instant::now() < until {
        thread::sleep(Duration::from_millis(20));
    }
    if stop.load(Ordering::Relaxed) {
        return Ok(());
    }
    if !online() {
        return Err("REGISTRY_UNAVAILABLE".into());
    }
    let directory = PrivateDir::open(state).map_err(|_| "PRIVATE_STATE_REQUIRED")?;
    let pin = Invitation::load(&directory, "bootstrap.json").map_err(security)?;
    set(status, "loading");
    let warm = pipeline::Warm::launch(route, &state.join("engine"), stop)
        .map_err(|e| format!("STAGE_LAUNCH: {e:?}"))?;
    let control = PeerListener::bind(route.listen, ListenerBudget::default()).map_err(security)?;
    // One data listener for each session slot (GLY-190).
    let data = std::iter::once(route.data_listen)
        .chain(route.more_data_listen.iter().copied())
        .map(|address| {
            let listener = PeerListener::bind(address, ListenerBudget::default());
            listener.map(|l| (address, l, AtomicBool::new(false)))
        })
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(security)?;
    {
        let mut report = status.lock().map_err(|_| "STAGE_STATE_FAILED")?;
        report.device = warm.device.clone();
        report.family = Some(warm.family.clone());
        report.state = "ready";
    }
    // Requests in flight: at most the route's sessions, each on its own thread and engine session.
    let active = std::sync::atomic::AtomicU32::new(0);
    // The stage's one runtime and its epoch, shared by the requests; a new epoch's is made once
    // none is in flight.
    let mut runtime: Option<(u64, Arc<starlings_security::StageRuntime>)> = None;
    thread::scope(|threads| -> Result<()> {
        while !stop.load(Ordering::Relaxed) {
            // Offline, the warm stage takes no connection; it stays loaded for the coordinator's
            // return.
            if !online() {
                set(status, "waiting_for_registry");
                thread::sleep(Duration::from_millis(20));
                continue;
            }
            if !warm.usable() {
                // No restart or reattachment after an engine state the adapter cannot account for.
                return Err("STAGE_ENGINE_UNUSABLE: restart the daemon".into());
            }
            let busy = active.load(Ordering::Acquire);
            set(status, if busy == 0 { "ready" } else { "serving" });
            // A data connection outside a request belongs to nobody. Only while none is in
            // flight: a request's peer may connect before this stage has read its Start.
            if busy == 0 {
                for (_, listener, _) in &data {
                    while let Ok(Some(stale)) = listener.try_accept() {
                        drop(stale);
                    }
                }
            }
            // With every slot serving, a new connection waits in the backlog until one frees.
            let accepted = if busy < route.sessions {
                control.try_accept()
            } else {
                Ok(None)
            };
            match accepted {
                Ok(Some(incoming)) => {
                    // Only the coordinator (its enrollment address, or this host) opens a request;
                    // anyone else on the network is dropped before the handshake, and a handshake
                    // gets 2 s, so silent connections cannot hold this one accept loop (GLY-366).
                    let coordinator = pin.endpoint().ip();
                    if incoming
                        .peer_ip()
                        .is_none_or(|ip| ip != coordinator && !ip.is_loopback())
                    {
                        continue;
                    }
                    let incoming = incoming.within(Duration::from_secs(2));
                    // Reload the atomically refreshed leaf for each connection.
                    let identity = Identity::load(&directory, "identity.json").map_err(security)?;
                    let epoch = fence.status().map_err(security)?.0;
                    if runtime.as_ref().is_none_or(|(e, _)| *e != epoch.number.0) {
                        if busy > 0 {
                            // The old epoch's requests finish first.
                            continue;
                        }
                        runtime = None;
                        let worker = starlings_core::Worker::new(
                            identity.node(),
                            identity.cluster(),
                            route.engine_bytes_limit.0,
                            epoch.number.0,
                            starlings_protocol::FixtureMode::Reject,
                        )
                        .map_err(|_| "STAGE_RUNTIME_FAILED")?;
                        // The last epoch's runtime may outlive its requests for a moment: this
                        // connection is then dropped and the coordinator's next one finds it gone.
                        let Ok(made) = starlings_security::StageRuntime::new(
                            worker,
                            fence.clone(),
                            route.sessions as usize,
                        ) else {
                            continue;
                        };
                        runtime = Some((epoch.number.0, Arc::new(made)));
                    }
                    let shared = runtime
                        .as_ref()
                        .map(|(_, r)| r.clone())
                        .ok_or("STAGE_RUNTIME")?;
                    let binding = warm.binding(route, epoch);
                    let Ok(channel) = incoming.generation(&identity, &pin, fence.clone(), binding)
                    else {
                        continue;
                    };
                    active.fetch_add(1, Ordering::AcqRel);
                    let (warm, data, pin, active) = (&warm, &data, &pin, &active);
                    threads.spawn(move || {
                        let shared = shared;
                        let served = warm.serve(
                            &pipeline::Stage {
                                route,
                                identity: &identity,
                                pin,
                                fence,
                                runtime: &shared,
                                data,
                                stop,
                            },
                            channel,
                        );
                        if let Ok(mut report) = status.lock() {
                            report.requests += 1;
                            if let Err(error) = served {
                                report.failed_requests += 1;
                                report.error = Some(format!("LAST_REQUEST: {error:?}"));
                            }
                            // A killed or ambiguous engine is never advertised again.
                            if !warm.usable() {
                                report.state = "failed";
                            }
                        }
                        // The request's hold on the runtime goes before its slot does.
                        drop(shared);
                        active.fetch_sub(1, Ordering::AcqRel);
                    });
                }
                Ok(None) | Err(starlings_security::Error::Capacity) => (),
                Err(error) => return Err(security(error)),
            }
            thread::sleep(Duration::from_millis(5));
        }
        Ok(())
    })?;
    set(status, "stopped");
    Ok(())
}

/// Advertise the warm stage only while it can take a request: its backend, its layer
/// sharding and the device memory the engine reported at prepare.
// ponytail: device bytes are the engine's prepare-time report, not a live NVML sample.
pub fn qualify_sample(
    report: &Mutex<Status>,
    route: &StageRoute,
    sample: &mut starlings_protocol::NodeCapabilities,
) {
    use starlings_protocol::*;
    let Ok(report) = report.lock() else { return };
    let (Some((_, free, total)), Some(family)) = (&report.device, &report.family) else {
        return;
    };
    if !matches!(report.state, "ready" | "serving") {
        return;
    }
    let (Ok(context), Ok(sessions)) = (
        BoundedU32::new(route.context_tokens),
        BoundedU32::new(route.sessions),
    ) else {
        return;
    };
    sample.state = NodeCapabilitiesState::Ready;
    sample.control_endpoint = format!("tls://{}", route.listen);
    sample.tensor_endpoint = format!("tls://{}", route.data_listen);
    sample.memory_pools.push(MemoryPool {
        pool_id: route.device_id.clone(),
        kind: MemoryPoolKind::Device,
        total_bytes: Decimal(*total),
        available_bytes: Decimal(*free),
        reserved_bytes: Decimal(0),
        device_ids: vec![route.device_id.clone()],
    });
    sample.backends = vec![Backend {
        backend_id: route.backend_id.clone(),
        build_digest: route.executable_sha256.clone(),
        architectures: vec![family.clone()],
        execution_profiles: vec![route.execution_profile.clone()],
        boundary_profiles: vec![route.boundary_profile.clone()],
        task_types: vec![BackendTaskTypes::Generate],
        layer_sharding: true,
        max_context_tokens: context,
        max_sessions: sessions,
        device_id: route.device_id.clone(),
        memory_pool_ids: vec![route.device_id.clone()],
    }];
}
