//! Explicit worker warm-up. API readiness cannot grant execution qualification.
use crate::cluster::{Result, security};
use starlings_adapters::service::{Cancellation, Service, ServiceConfig};
use starlings_protocol::{AssetRole, Digest, LeaseId};
use starlings_security::{NodeStatus, RegistryFence, catalog::Catalog};
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
    pin: Option<LeaseId>,
    process_group_absent: bool,
    inference_qualified: bool,
    error: Option<String>,
    registry_failure: Option<NodeStatus>,
    startup_error: Option<String>,
    startup_phase: Option<&'static str>,
    endpoint: Option<std::net::SocketAddr>,
    fixture: bool,
    model_architecture: Option<String>,
    backend_version_declared: String,
    execution_profile_declared: String,
    startup_properties: Option<starlings_adapters::service::Properties>,
    reserved_bytes: Option<starlings_protocol::Decimal>,
    cleanup_proven: bool,
    #[serde(skip)]
    worker: Option<Arc<starlings_security::EnrolledWorker>>,
    #[serde(skip)]
    resources: Option<Arc<starlings_cli::contribution::ContributionGate>>,
}
pub struct Owner {
    stop: Arc<AtomicBool>,
    status: Arc<Mutex<Status>>,
    thread: Option<JoinHandle<()>>,
    monitor: Option<JoinHandle<()>>,
    cancellation: Arc<Mutex<Option<Cancellation>>>,
    cache: PathBuf,
}
impl Owner {
    pub fn start(
        cache: PathBuf,
        config: ServiceConfig,
        approved: Digest,
        fence: RegistryFence,
        execution: Option<(starlings_cli::routes::WorkerRoute, PathBuf)>,
    ) -> Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let resources = execution
            .as_ref()
            .and_then(|(route, _)| {
                route.contribution.clone().map(|policy| {
                    starlings_cli::contribution::ContributionGate::new(policy, &route.qualification)
                })
            })
            .transpose()
            .map_err(security)?
            .map(Arc::new);
        let status = Arc::new(Mutex::new(Status {
            state: "waiting_for_registry",
            pin: None,
            process_group_absent: false,
            inference_qualified: false,
            error: None,
            registry_failure: None,
            startup_error: None,
            startup_phase: None,
            endpoint: None,
            fixture: config.backend_version == "synthetic-v1",
            reserved_bytes: None,
            model_architecture: None,
            backend_version_declared: config.backend_version.clone(),
            execution_profile_declared: config.execution_profile.clone(),
            startup_properties: None,
            worker: None,
            cleanup_proven: false,
            resources: resources.clone(),
        }));
        let cancellation: Arc<Mutex<Option<Cancellation>>> = Arc::new(Mutex::new(None));
        let monitor = if let Some(resources) = resources {
            let stopped = stop.clone();
            let report = status.clone();
            let control = Arc::downgrade(&cancellation);
            #[cfg(feature = "test-backend")]
            let pressure = execution
                .as_ref()
                .filter(|(r, _)| r.qualification.fixture)
                .and_then(|(r, _)| r.service.environment.get("COMPUTE_FIXTURE_PRESSURE"))
                .map(PathBuf::from);
            Some(
                thread::Builder::new()
                    .name("contribution-sample".into())
                    .spawn(move || {
                        while !stopped.load(Ordering::Relaxed) {
                            let sample = if resources.needs_temperature() {
                                starlings_adapters::resources::HostSnapshot::collect_contribution()
                            } else {
                                starlings_adapters::resources::HostSnapshot::collect_memory()
                            };
                            #[cfg(feature = "test-backend")]
                            if pressure.as_ref().is_some_and(|p| p.exists()) {
                                resources.withdraw();
                            }
                            if resources.update(sample).is_err() {
                                let worker = if let Ok(mut report) = report.lock() {
                                    report.error = Some("RESOURCE_POLICY_PRESSURE".into());
                                    report.worker.clone()
                                } else {
                                    None
                                };
                                stopped.store(true, Ordering::Relaxed);
                                if let Some(worker) = worker {
                                    let _ = worker.withdraw_local();
                                }
                                if let Some(control) = control.upgrade()
                                    && let Ok(control) = control.lock()
                                    && let Some(control) = &*control
                                {
                                    let _ = control.cancel();
                                }
                                return;
                            }
                            let until = Instant::now() + Duration::from_secs(1);
                            while Instant::now() < until && !stopped.load(Ordering::Relaxed) {
                                thread::sleep(Duration::from_millis(20));
                            }
                        }
                    })
                    .map_err(|_| "THREAD_UNAVAILABLE")?,
            )
        } else {
            None
        };
        let control = cancellation.clone();
        let stopped = stop.clone();
        let report = status.clone();
        let recovery_cache = cache.clone();
        let thread = thread::Builder::new()
            .name("catalog-service".into())
            .spawn(move || {
                if let Err(error) = supervise(
                    cache,
                    (config, approved),
                    fence,
                    execution,
                    &stopped,
                    &report,
                    &control,
                ) && let Ok(mut report) = report.lock()
                {
                    report.state = "failed_pins_retained";
                    report.error = Some(error);
                }
                stopped.store(true, Ordering::Relaxed);
            })
            .map_err(|_| "THREAD_UNAVAILABLE")?;
        Ok(Self {
            stop,
            status,
            thread: Some(thread),
            monitor,
            cancellation,
            cache: recovery_cache,
        })
    }
    pub fn status(&self) -> Result<Status> {
        self.status
            .lock()
            .map(|s| {
                let mut report = s.clone();
                report.reserved_bytes = report
                    .worker
                    .as_ref()
                    .and_then(|w| w.snapshot().ok())
                    .map(|s| s.reserved_bytes);
                report
            })
            .map_err(|_| "SERVICE_STATE_FAILED".into())
    }
    pub fn drain(&self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Ok(control) = self.cancellation.lock()
            && let Some(control) = &*control
        {
            let _ = control.cancel();
        }
    }
    pub fn announcement(&self) -> Arc<Mutex<Status>> {
        self.status.clone()
    }
    pub fn endpoint(&self) -> Option<std::net::SocketAddr> {
        self.status.lock().ok().and_then(|s| {
            if matches!(s.state, "ready" | "serving")
                && s.resources.as_ref().is_none_or(|g| {
                    if s.state == "serving" {
                        g.maintain().is_ok()
                    } else {
                        g.check().is_ok()
                    }
                })
            {
                s.endpoint
            } else {
                None
            }
        })
    }
    pub fn recover(&mut self, lease: LeaseId, configuration: &Digest) -> Result<bool> {
        if self.thread.as_ref().is_some_and(|t| !t.is_finished()) {
            return Err("SERVICE_RECOVERY_NOT_READY".into());
        }
        if let Some(worker) = self.status()?.worker
            && worker.snapshot().map_err(security)?.reserved_bytes.0 != 0
        {
            return Err("SERVICE_DRAIN_INCOMPLETE".into());
        }
        let repeated = Catalog::open(&self.cache)
            .map_err(security)?
            .recover_service(lease, configuration)
            .map_err(security)?;
        let mut report = self.status.lock().map_err(|_| "SERVICE_STATE_FAILED")?;
        report.state = "recovered";
        report.pin = None;
        Ok(repeated)
    }
    pub fn join(&mut self) -> Result<()> {
        self.drain();
        if let Some(thread) = self.thread.take() {
            thread.join().map_err(|_| "SERVICE_THREAD_FAILED")?;
        }
        // A hung read-only OS sensor cannot hold stop/admin or refresh an old sample.
        // It owns no model/cache/credential, and only a weak cancellation reference.
        // Process exit ends an OS read that remains blocked.
        if let Some(monitor) = self.monitor.take()
            && monitor.is_finished()
        {
            monitor.join().map_err(|_| "RESOURCE_MONITOR_FAILED")?;
        }
        Ok(())
    }
}
impl Drop for Owner {
    fn drop(&mut self) {
        let _ = self.join();
    }
}
fn set_state(status: &Mutex<Status>, state: &'static str) {
    if let Ok(mut report) = status.lock() {
        report.state = state;
    }
}

fn spawn(
    config: ServiceConfig,
    approved: &Digest,
    mut phase: impl FnMut(&'static str),
) -> std::result::Result<Service, starlings_adapters::service::Error> {
    phase("key_generation");
    let key = format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    );
    #[cfg(feature = "test-backend")]
    if config.backend_version == "synthetic-v1" {
        // The synthetic child owns port zero and reports its bound address. This
        // development-only path cannot qualify a production backend or native run.
        config.validate_settings_observed(approved, &mut phase)?;
        // The fixture changes its rendezvous environment. Its final launcher
        // still validates all paths and executable bytes immediately before spawn.
        return starlings_adapters::service::fixture::spawn_observed(config, key, phase);
    }
    Service::spawn_observed(config, approved, key, phase)
}

/// No shell expansion or argument rewriting. Every weights asset must appear as
/// a complete argument; approval covers all remaining backend-specific arguments.
fn validate(
    catalog: &Catalog,
    config: &ServiceConfig,
    approved: &Digest,
    mut phase: impl FnMut(&'static str),
) -> Result<starlings_protocol::ModelManifest> {
    phase("configuration_approval");
    if &config.digest().map_err(|e| e.to_string())? != approved {
        return Err(starlings_adapters::service::Error::Unapproved.to_string());
    }
    phase("manifest_approval");
    let manifest = catalog
        .authorized(&config.manifest_digest, false)
        .map_err(security)?;
    phase("profile_binding");
    if manifest.execution_profile != config.execution_profile {
        return Err("SERVICE_PROFILE_MISMATCH".into());
    }
    let mut weights = 0;
    for asset in &manifest.assets {
        phase("asset_binding");
        let path = catalog
            .asset_path(&manifest.manifest_digest, &asset.path)
            .map_err(security)?;
        if asset.role == AssetRole::Weights {
            weights += 1;
            if !config
                .arguments
                .iter()
                .any(|arg| std::path::Path::new(arg) == path)
            {
                return Err("SERVICE_WEIGHTS_NOT_BOUND".into());
            }
        }
    }
    if weights == 0 {
        return Err("SERVICE_WEIGHTS_REQUIRED".into());
    }
    config
        .validate_observed(approved, phase)
        .map_err(|e| e.to_string())?;
    Ok(manifest)
}

fn launch_catalog(
    cache: &std::path::Path,
    wait_for_registry: impl FnOnce() -> Result<bool>,
    mut phase: impl FnMut(&'static str),
) -> Result<Option<Catalog>> {
    // The registry gates a fresh launch, not observation of durable local ownership.
    phase("catalog_open");
    let catalog = Catalog::open(cache).map_err(security)?;
    phase("catalog_ownership");
    catalog
        .ensure_no_service()
        .map_err(|_| "SERVICE_CLEANUP_REQUIRED: prior launch remains unqualified")?;
    let online = wait_for_registry()?;
    Ok(online.then_some(catalog))
}

fn supervise(
    cache: PathBuf,
    launch: (ServiceConfig, Digest),
    fence: RegistryFence,
    execution: Option<(starlings_cli::routes::WorkerRoute, PathBuf)>,
    stop: &AtomicBool,
    status: &Mutex<Status>,
    cancellation: &Mutex<Option<Cancellation>>,
) -> Result<()> {
    let (config, approved) = launch;
    let online = || fence.status().is_ok_and(|(_, s)| s == NodeStatus::Online);
    set_state(status, "verifying_catalog");
    let phase = |phase| {
        if let Ok(mut report) = status.lock() {
            report.startup_phase = Some(phase);
        }
    };
    let Some(mut catalog) = launch_catalog(
        &cache,
        || {
            set_state(status, "waiting_for_registry");
            let until = Instant::now() + Duration::from_secs(15);
            while !stop.load(Ordering::Relaxed) && !online() && Instant::now() < until {
                thread::sleep(Duration::from_millis(20));
            }
            if stop.load(Ordering::Relaxed) {
                return Ok(false);
            }
            if !online() {
                return Err("REGISTRY_UNAVAILABLE".into());
            }
            Ok(true)
        },
        phase,
    )?
    else {
        set_state(status, "stopped");
        return Ok(());
    };
    set_state(status, "verifying_catalog");
    let manifest = validate(&catalog, &config, &approved, phase)?;
    if let Some((route, _)) = &execution
        && (manifest.layers.len() != route.qualification.model_layer_count as usize
            || manifest.required_boundary_profile != route.qualification.boundary_profile)
    {
        return Err("SERVICE_MANIFEST_PLAN_MISMATCH".into());
    }
    status
        .lock()
        .map_err(|_| "SERVICE_STATE_FAILED")?
        .model_architecture = Some(manifest.architecture);
    if let Some((route, _)) = &execution {
        let gate = status
            .lock()
            .map_err(|_| "SERVICE_STATE_FAILED")?
            .resources
            .clone();
        if let Some(gate) = gate {
            let until = Instant::now() + starlings_adapters::resources::TELEMETRY_TTL;
            while !gate.has_sample() && !stop.load(Ordering::Relaxed) && Instant::now() < until {
                thread::sleep(Duration::from_millis(20));
            }
            gate.loading_allowed()
                .map_err(|_| "RESOURCE_POLICY_LOADING")?;
        }
        let memory = starlings_adapters::resources::HostSnapshot::collect_memory().report();
        if !memory.memory_pools.iter().any(|p| {
            p.pool_id == "host" && p.available_bytes.0 >= route.qualification.loading_bytes.0
        }) {
            return Err("SERVICE_LOADING_CAPACITY".into());
        }
    }
    if stop.load(Ordering::Relaxed) || !online() {
        set_state(status, "stopped");
        return Ok(());
    }
    let pin: LeaseId = uuid::Uuid::new_v4()
        .to_string()
        .parse()
        .map_err(|_| "INVALID_SERVICE_PIN")?;
    let cleanup_hash = execution
        .as_ref()
        .and_then(|(r, _)| r.cleanup.as_ref())
        .map(starlings_protocol::content_hash)
        .transpose()
        .map_err(|_| "INVALID_CLEANUP_QUALIFICATION")?;
    catalog
        .begin_service_qualified(
            &config.manifest_digest,
            pin,
            &approved,
            cleanup_hash.clone(),
        )
        .map_err(security)?;
    status.lock().map_err(|_| "SERVICE_STATE_FAILED")?.pin = Some(pin);
    if stop.load(Ordering::Relaxed) || !online() {
        set_state(status, "stopped_pins_retained");
        return Ok(());
    }
    // No reattachment or automatic restart after an uncertain prior launch.
    // Pins survive crashes, failed spawn and unqualified process/device cleanup.
    set_state(status, "loading");
    status
        .lock()
        .map_err(|_| "SERVICE_STATE_FAILED")?
        .startup_phase = Some("spawn");
    let mut service = spawn(config, &approved, |phase| {
        if let Ok(mut report) = status.lock() {
            report.startup_phase = Some(phase);
        }
    })
    .map_err(|e| e.to_string())?;
    *cancellation.lock().map_err(|_| "SERVICE_STATE_FAILED")? = Some(service.cancellation());
    let until = Instant::now() + Duration::from_secs(60);
    let mut ready = false;
    let mut startup_error = "BACKEND_READINESS_DEADLINE".to_string();
    while !stop.load(Ordering::Relaxed) && online() && Instant::now() < until {
        match service.probe_observed(|phase| {
            if let Ok(mut report) = status.lock() {
                report.startup_phase = Some(phase);
            }
        }) {
            Ok(()) => {
                let mut report = status.lock().map_err(|_| "SERVICE_STATE_FAILED")?;
                report.startup_properties = service.properties().cloned();
                report.startup_error = None;
                ready = true;
                break;
            }
            Err(error) => {
                startup_error = format!("BACKEND_PROBE: {error}");
                status
                    .lock()
                    .map_err(|_| "SERVICE_STATE_FAILED")?
                    .startup_error = Some(startup_error.clone());
                if !matches!(
                    error,
                    starlings_adapters::service::Error::Transport
                        | starlings_adapters::service::Error::Unavailable
                ) {
                    break;
                }
            }
        }
        if let Err(error) = service.check_alive() {
            startup_error = format!("BACKEND_PROCESS: {error}");
            break;
        }
        thread::sleep(Duration::from_millis(100));
    }
    if !ready {
        if stop.load(Ordering::Relaxed) {
            startup_error = "STARTUP_STOPPED".into();
        } else if !online() {
            startup_error = "STARTUP_REGISTRY_LOST".into();
        }
        status
            .lock()
            .map_err(|_| "SERVICE_STATE_FAILED")?
            .startup_error = Some(startup_error);
    }
    if ready {
        let result = if let Some((route, state)) = &execution {
            execute(&mut service, route, state, fence.clone(), stop, status)
        } else {
            set_state(status, "ready_unqualified");
            (|| -> Result<()> {
                while !stop.load(Ordering::Relaxed) {
                    // A warm service has no generation owner yet; never advertise a replica.
                    for _ in 0..50 {
                        if stop.load(Ordering::Relaxed) {
                            break;
                        }
                        require_online(&fence, status)?;
                        thread::sleep(Duration::from_millis(20));
                    }
                    if !stop.load(Ordering::Relaxed) {
                        require_online(&fence, status)?;
                        service.probe().map_err(|e| e.to_string())?;
                    }
                }
                Ok(())
            })()
        };
        if let Err(error) = result {
            status.lock().map_err(|_| "SERVICE_STATE_FAILED")?.error = Some(error);
            ready = false;
        }
    }
    set_state(status, "stopping");
    let absent = service.confirm_process_group_exit().is_ok();
    // The owner has joined generation/handshake work before this point. Only
    // a pre-approved CPU/no-escape policy can promote group absence to cleanup.
    let mut cleanup_proven = false;
    if absent && let Some(qualification) = &cleanup_hash {
        let worker = status
            .lock()
            .map_err(|_| "SERVICE_STATE_FAILED")?
            .worker
            .clone();
        if let Some(worker) = worker {
            worker.backend_shutdown_drained().map_err(security)?;
        }
        catalog
            .record_service_drained(pin, &approved, qualification)
            .map_err(security)?;
        cleanup_proven = true;
    }
    let mut report = status.lock().map_err(|_| "SERVICE_STATE_FAILED")?;
    report.process_group_absent = absent;
    report.cleanup_proven = cleanup_proven;
    report.error = report.error.clone().or_else(|| {
        if !absent {
            Some("PROCESS_GROUP_UNCONFIRMED".into())
        } else if !ready {
            Some("BACKEND_NOT_READY".into())
        } else {
            None
        }
    });
    report.inference_qualified = false;
    report.state = if cleanup_proven {
        if ready {
            "stopped_cleanup_proven"
        } else {
            "failed_cleanup_proven"
        }
    } else if ready && absent {
        "stopped_pins_retained"
    } else {
        "failed_pins_retained"
    };
    // Without the pre-reviewed policy, group absence remains only an observation.
    drop(catalog);
    Ok(())
}

fn require_online(fence: &RegistryFence, status: &Mutex<Status>) -> Result<()> {
    let (_, registry) = fence.status().map_err(security)?;
    if registry != NodeStatus::Online {
        status
            .lock()
            .map_err(|_| "SERVICE_STATE_FAILED")?
            .registry_failure = Some(registry);
        return Err("REGISTRY_LOST".into());
    }
    Ok(())
}

fn execute(
    service: &mut Service,
    route: &starlings_cli::routes::WorkerRoute,
    state: &std::path::Path,
    fence: RegistryFence,
    stop: &AtomicBool,
    status: &Mutex<Status>,
) -> Result<()> {
    use starlings_security::{
        EnrolledWorker, GenerationBinding, Identity, Invitation, ListenerBudget, PeerListener,
        PrivateDir,
    };
    let q = &route.qualification;
    let resources = status
        .lock()
        .map_err(|_| "SERVICE_STATE_FAILED")?
        .resources
        .clone();
    let directory = PrivateDir::open(state).map_err(|_| "PRIVATE_STATE_REQUIRED")?;
    let invitation = Invitation::load(&directory, "bootstrap.json").map_err(security)?;
    let epoch = fence.status().map_err(security)?.0;
    let worker = starlings_core::Worker::new(
        q.replica.node,
        invitation.cluster(),
        q.replica.request_bytes,
        epoch.number.0,
        starlings_protocol::FixtureMode::Reject,
    )
    .map_err(|_| "INVALID_WORKER")?;
    let worker =
        Arc::new(EnrolledWorker::new(worker, fence.clone(), q.replica.clone()).map_err(security)?);
    let listener = PeerListener::bind(route.listen, ListenerBudget::default()).map_err(security)?;
    {
        let mut report = status.lock().map_err(|_| "SERVICE_STATE_FAILED")?;
        report.endpoint = Some(listener.local_addr().map_err(security)?);
        report.inference_qualified = !q.fixture;
        report.state = "ready";
        report.worker = Some(worker.clone());
    }
    thread::scope(|threads| {
        let (send, receive) =
            std::sync::mpsc::sync_channel::<starlings_security::GenerationStream>(1);
        let mut handshakes: Vec<thread::ScopedJoinHandle<'_, ()>> = Vec::new();
        let mut health = Instant::now();
        while !stop.load(Ordering::Relaxed) {
            require_online(&fence, status)?;
            if resources.as_ref().is_some_and(|g| g.maintain().is_err()) {
                worker.withdraw_local().map_err(security)?;
                return Err("RESOURCE_POLICY_PRESSURE".into());
            }
            let mut index = 0;
            while index < handshakes.len() {
                if handshakes[index].is_finished() {
                    handshakes
                        .swap_remove(index)
                        .join()
                        .map_err(|_| "GENERATION_HANDSHAKE_FAILED")?;
                } else {
                    index += 1;
                }
            }
            if let Ok(channel) = receive.try_recv() {
                let memory = starlings_adapters::resources::HostSnapshot::collect_memory();
                if memory.qualification_target().as_ref() != Some(&q.target) {
                    return Err("QUALIFICATION_TARGET_CHANGED".into());
                }
                if !memory.report().memory_pools.iter().any(|p| {
                    p.pool_id == q.replica.pool && p.available_bytes.0 >= q.replica.request_bytes
                }) {
                    channel.watch().cancel();
                    continue;
                }
                set_state(status, "serving");
                let done = match starlings_cli::inference::serve_worker_qualified_guarded(
                    service,
                    &worker,
                    channel,
                    q,
                    resources.as_deref(),
                ) {
                    Ok(Some(done)) => done,
                    Ok(None) | Err(starlings_security::Error::Capacity)
                        if service.check_alive().is_ok()
                            && worker.snapshot().map_err(security)?.reserved_bytes.0 == 0 =>
                    {
                        require_online(&fence, status)?;
                        set_state(status, "ready");
                        continue;
                    }
                    Ok(None) => return Err("SERVICE_IDLE_ADMISSION_STATE_CONFLICT".into()),
                    Err(error) => return Err(security(error)),
                };
                // Only the reviewed local qualification authorizes request-level drain.
                worker
                    .drained(done.generation, done.invocation)
                    .map_err(security)?;
                set_state(status, "ready");
            }
            if health.elapsed() >= Duration::from_secs(1) {
                service.probe().map_err(|e| e.to_string())?;
                health = Instant::now();
            }
            match listener.try_accept() {
                Ok(Some(incoming)) => {
                    // Reload the atomically refreshed leaf for each connection.
                    let identity = Identity::load(&directory, "identity.json").map_err(security)?;
                    let invitation =
                        Invitation::load(&directory, "bootstrap.json").map_err(security)?;
                    let binding = GenerationBinding {
                        version: starlings_protocol::Version,
                        epoch: fence.status().map_err(security)?.0,
                        model: q.replica.manifest.clone(),
                        profile: q.replica.profile.clone(),
                    };
                    let send = send.clone();
                    let fence = fence.clone();
                    // The existing independent listener budget bounds TLS threads.
                    handshakes.push(
                        thread::Builder::new()
                            .name("generation-handshake".into())
                            .spawn_scoped(threads, move || {
                                if let Ok(channel) =
                                    incoming.generation(&identity, &invitation, fence, binding)
                                {
                                    let _ = send.try_send(channel);
                                }
                            })
                            .map_err(|_| "THREAD_UNAVAILABLE")?,
                    );
                }
                Ok(None) | Err(starlings_security::Error::Capacity) => (),
                Err(error) => return Err(security(error)),
            }
            thread::sleep(Duration::from_millis(20));
        }
        drop(receive);
        drop(send);
        Ok(())
    })
}

pub fn qualify_sample(
    report: &Mutex<Status>,
    route: &starlings_cli::routes::WorkerRoute,
    sample: &mut starlings_protocol::NodeCapabilities,
) {
    use starlings_protocol::*;
    let Ok(report) = report.lock() else { return };
    if !matches!(report.state, "ready" | "serving")
        || report.endpoint.is_none()
        || report.resources.as_ref().is_some_and(|g| {
            if report.state == "serving" {
                g.maintain().is_err()
            } else {
                g.check().is_err()
            }
        })
    {
        return;
    }
    let q = &route.qualification.replica;
    sample.state = NodeCapabilitiesState::Ready;
    sample.control_endpoint = format!("tls://{}", report.endpoint.unwrap());
    sample.backends = vec![Backend {
        backend_id: q.backend.clone(),
        build_digest: q.build.clone(),
        architectures: report.model_architecture.clone().into_iter().collect(),
        execution_profiles: vec![q.profile.clone()],
        boundary_profiles: vec![],
        task_types: vec![BackendTaskTypes::Generate],
        layer_sharding: false,
        max_context_tokens: BoundedU32::new(q.context_limit).unwrap(),
        max_sessions: BoundedU32::new(1).unwrap(),
        device_id: "cpu".into(),
        memory_pool_ids: vec![q.pool.clone()],
    }];
}

#[cfg(test)]
mod tests {
    use super::*;
    use starlings_security::{PrivateDir, catalog::ServiceRecord};
    use std::cell::Cell;
    #[cfg(feature = "test-backend")]
    #[test]
    fn fixture_validation_preserves_approval_and_final_bytes_without_duplicate_hashing() {
        use starlings_adapters::service::Error;
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().canonicalize().unwrap();
        let executable = directory.join("unlaunchable.bin");
        std::fs::write(&executable, b"fixture bytes").unwrap();
        let mut config = ServiceConfig {
            executable,
            executable_sha256: starlings_protocol::byte_digest(b"fixture bytes"),
            arguments: vec![],
            directory,
            environment: Default::default(),
            key_environment: "COMPUTE_FIXTURE_KEY".into(),
            endpoint: "127.0.0.1:1".parse().unwrap(),
            backend_version: "synthetic-v1".into(),
            properties: None,
            api: Default::default(),
            glyd_identity: None,
            manifest_digest: starlings_protocol::byte_digest(b"fixture manifest"),
            execution_profile: "fixture-bf16".into(),
            backend_model: "synthetic-backend".into(),
            reasoning_content: false,
            request_timeout_ms: 1000,
        };
        let mut phases = Vec::new();
        let settings_phases = [
            "key_generation",
            "configuration_digest",
            "configuration_paths",
        ];
        assert!(
            spawn(config.clone(), &config.digest().unwrap(), |p| phases
                .push(p))
            .is_err_and(|e| e == Error::Invalid)
        );
        assert_eq!(phases, settings_phases);
        phases.clear();
        config
            .environment
            .insert("COMPUTE_FIXTURE_MODE".into(), "normal".into());
        let approved = config.digest().unwrap();
        config.backend_model = "changed after approval".into();
        assert!(
            spawn(config.clone(), &approved, |p| phases.push(p))
                .is_err_and(|e| e == Error::Unapproved)
        );
        assert_eq!(phases, ["key_generation", "configuration_digest"]);
        phases.clear();
        let mut oversized = config.clone();
        oversized
            .environment
            .insert("COMPUTE_FIXTURE_REPORT".into(), "x".repeat(64 * 1024));
        assert!(
            spawn(oversized.clone(), &oversized.digest().unwrap(), |p| phases
                .push(p))
            .is_err_and(|e| e == Error::Invalid)
        );
        assert_eq!(phases, settings_phases);
        phases.clear();
        config.executable_sha256 = starlings_protocol::byte_digest(b"wrong bytes");
        assert!(
            spawn(config.clone(), &config.digest().unwrap(), |p| phases
                .push(p))
            .is_err_and(|e| e == Error::Unapproved)
        );
        assert_eq!(
            phases,
            [
                "key_generation",
                "configuration_digest",
                "configuration_paths",
                "fixture_directory",
                "configuration_digest",
                "configuration_paths",
                "executable_digest"
            ]
        );
    }

    #[test]
    fn durable_ownership_refuses_before_registry_without_mutating_records() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let digest: Digest = "1".repeat(64).try_into().unwrap();
        let record = ServiceRecord {
            version: 2,
            lease: "00000000-0000-4000-8000-000000000993".parse().unwrap(),
            manifest: digest.clone(),
            configuration: digest.clone(),
            cleanup_qualification: Some(digest),
            phase: starlings_security::catalog::ServicePhase::Quarantined,
        };
        for (i, bytes) in [
            serde_json::to_vec(&record).unwrap(),
            serde_json::to_vec(&ServiceRecord {
                phase: starlings_security::catalog::ServicePhase::Drained,
                ..record.clone()
            })
            .unwrap(),
            serde_json::to_vec(&ServiceRecord {
                version: 1,
                cleanup_qualification: None,
                ..record.clone()
            })
            .unwrap(),
            b"invalid".to_vec(),
        ]
        .into_iter()
        .enumerate()
        {
            let cache = root.join(i.to_string());
            PrivateDir::ensure(&cache)
                .unwrap()
                .write_atomic_durable("service-start.json", &bytes)
                .unwrap();
            let called = Cell::new(false);
            let mut phases = Vec::new();
            let error = launch_catalog(
                &cache,
                || {
                    called.set(true);
                    Err("REGISTRY_UNAVAILABLE".into())
                },
                |phase| phases.push(phase),
            )
            .err()
            .unwrap();
            assert_eq!(
                error,
                "SERVICE_CLEANUP_REQUIRED: prior launch remains unqualified"
            );
            assert!(!called.get());
            assert_eq!(phases, ["catalog_open", "catalog_ownership"]);
            assert_eq!(
                std::fs::read(cache.join("service-start.json")).unwrap(),
                bytes
            );
            assert!(
                Catalog::open(&cache).is_ok(),
                "failed preflight releases lock"
            );
        }
        let cache = root.join("fresh");
        assert!(
            launch_catalog(&cache, || Err("REGISTRY_UNAVAILABLE".into()), |_| {})
                .is_err_and(|e| e == "REGISTRY_UNAVAILABLE")
        );
        assert!(
            launch_catalog(&cache, || Ok(false), |_| {})
                .unwrap()
                .is_none()
        );
        let catalog = launch_catalog(&cache, || Ok(true), |_| {})
            .unwrap()
            .unwrap();
        assert!(
            Catalog::open(&cache).is_err(),
            "fresh launch retains exclusive owner"
        );
        assert!(
            !cache.join("service-start.json").exists(),
            "preflight cannot launch"
        );
        drop(catalog);
        assert!(Catalog::open(&cache).is_ok());
        let recovered = serde_json::to_vec(&ServiceRecord {
            phase: starlings_security::catalog::ServicePhase::Recovered,
            ..record
        })
        .unwrap();
        PrivateDir::open(&cache)
            .unwrap()
            .write_atomic_durable("service-start.json", &recovered)
            .unwrap();
        assert!(
            launch_catalog(&cache, || Err("REGISTRY_UNAVAILABLE".into()), |_| {})
                .is_err_and(|e| e == "REGISTRY_UNAVAILABLE")
        );
        assert!(
            launch_catalog(&cache, || Ok(true), |_| {})
                .unwrap()
                .is_some()
        );
        assert_eq!(
            std::fs::read(cache.join("service-start.json")).unwrap(),
            recovered
        );
    }
}
