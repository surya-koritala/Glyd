//! Bounded control loop; optional explicitly approved worker warm-up runs separately.
use crate::cluster::{self, Command, Config, Paths, Reply, Result, role_label, security};
use starlings_protocol::NodeId;
use starlings_security::{
    AdminListener, Authority, Identity, Invitation, InvitationIssuer, ListenerBudget, PeerListener,
    PendingEnrollment, PrivateDir, RegistryClient, RegistryFence, RegistryRequest, Role, StateLock,
    heartbeat_delay,
};
use std::{
    collections::BTreeMap,
    io::Write,
    net::{IpAddr, SocketAddr},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU8, Ordering},
        mpsc::{self, SyncSender},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const POLL: Duration = Duration::from_millis(20);
pub enum Launch {
    Warmup(
        Box<starlings_adapters::service::ServiceConfig>,
        starlings_protocol::Digest,
    ),
    Worker(Box<starlings_cli::routes::WorkerRoute>),
    Coordinator(Box<starlings_cli::routes::CoordinatorRoute>),
    Stage(Box<starlings_cli::routes::StageRoute>),
    Pipeline(Box<starlings_cli::routes::PipelineRoute>),
}
/// GLY-188: the coordinator's two-stage pipeline route.
struct Pipeline {
    config: starlings_cli::routes::PipelineRoute,
    listener: starlings_security::ClientListener,
    /// One flag per session slot the stages offer (GLY-190): set while a request holds it.
    slots: Arc<Vec<AtomicBool>>,
    /// Each stage's control request sequence: increasing across requests and restarts (it
    /// starts at the clock's microseconds), one request applied at a time.
    sequences: Arc<Vec<std::sync::Mutex<u64>>>,
}
struct Route {
    config: starlings_cli::routes::CoordinatorRoute,
    listener: starlings_security::ClientListener,
    pool: Arc<crate::task_routing::Pool>,
    last_sample: Instant,
}
enum Event {
    Admin(Command, SyncSender<Reply>, Arc<AtomicU8>),
    Enrollment(Box<PendingEnrollment>),
    Heartbeat(Box<starlings_security::PendingHeartbeat>),
}
struct Runtime {
    state: std::path::PathBuf,
    authority: Option<Authority>,
    registry: Option<RegistryClient>,
    fence: Option<RegistryFence>,
    config: Config,
    addresses: Vec<SocketAddr>,
    mdns: Option<IpAddr>,
    draining: bool,
    stopping: bool,
    approvals: BTreeMap<NodeId, (String, Instant)>,
    inviting: Arc<AtomicBool>,
    publishing_token: Arc<AtomicBool>,
    observing_topology: Arc<AtomicBool>,
    service: Option<crate::service::Owner>,
    route: Option<Route>,
    stage: Option<crate::stage::Owner>,
    pipeline: Option<Pipeline>,
    probe_server: Option<crate::links::Server>,
    probe_worker: Option<crate::links::Worker>,
}
impl Runtime {
    fn invitation_issuer(
        &self,
        role: Role,
        via: Option<SocketAddr>,
    ) -> Result<(InvitationIssuer, SocketAddr)> {
        if self.draining || !matches!(role, Role::Worker | Role::Controller) {
            return Err("ADMISSION_CLOSED: invitation refused".into());
        }
        let endpoint = via
            .or_else(|| self.addresses.first().copied())
            .ok_or("ENROLLMENT_DISABLED")?;
        if !self.addresses.contains(&endpoint) {
            return Err("UNBOUND_ADDRESS: choose an active listener".into());
        }
        Ok((
            self.authority
                .as_ref()
                .ok_or("COORDINATOR_REQUIRED")?
                .invitation_issuer(),
            endpoint,
        ))
    }
    fn admin(
        &mut self,
        command: Command,
        directory: &PrivateDir,
        budget: &ListenerBudget,
        reply: SyncSender<Reply>,
        stage: Arc<AtomicU8>,
        workers: &mut Vec<JoinHandle<()>>,
    ) -> Result<()> {
        // Queued commands that already timed out must never start later.
        if stage
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Ok(());
        }
        if let Command::Invite { role, output, via } = command {
            match self.invitation_issuer(role, via) {
                Ok((issuer, endpoint)) => {
                    if let Some(job) = admin_job(
                        self.inviting.clone(),
                        reply,
                        stage.clone(),
                        "invitation-publication",
                        "INVITATION_BUSY",
                        move || publish_invitation(issuer, endpoint, role, output, Some(&stage)),
                    )? {
                        workers.push(job);
                    }
                }
                Err(error) => finish_admin(reply, &stage, Err(error), Instant::now()),
            }
        } else if let Command::Token {
            node,
            scopes,
            lifetime,
            output,
        } = command
        {
            let issuer: Result<_> = if self.draining {
                Err("ADMISSION_CLOSED".into())
            } else {
                self.authority
                    .as_ref()
                    .ok_or_else(|| "COORDINATOR_REQUIRED".into())
                    .map(Authority::client_token_issuer)
            };
            match issuer {
                Ok(issuer) => {
                    if let Some(job) = admin_job(
                        self.publishing_token.clone(),
                        reply,
                        stage,
                        "token-publication",
                        "TOKEN_BUSY",
                        move || publish_token(issuer, node, scopes, lifetime, output),
                    )? {
                        workers.push(job);
                    }
                }
                Err(error) => finish_admin(reply, &stage, Err(error), Instant::now()),
            }
        } else if matches!(command, Command::Topology) {
            let fence = self.fence.clone();
            let addresses = self.addresses.clone();
            let mdns = self.mdns;
            if let Some(job) = admin_job(
                self.observing_topology.clone(),
                reply,
                stage,
                "topology-observation",
                "TOPOLOGY_BUSY",
                move || topology(fence, addresses, mdns),
            )? {
                workers.push(job);
            }
        } else {
            let began = Instant::now();
            let result = self.command(command, directory, budget, Some(&stage));
            finish_admin(reply, &stage, result, began);
        }
        Ok(())
    }
    fn activate_route(&mut self, config: starlings_cli::routes::CoordinatorRoute) -> Result<()> {
        if self.authority.is_none()
            || self.draining
            || self.route.is_some()
            || config.plan.cluster_id != self.config.cluster
        {
            return Err("ROUTE_ACTIVATION_REFUSED".into());
        }
        self.route = Some(Route {
            listener: starlings_security::ClientListener::bind(
                config.listen,
                ListenerBudget::default(),
            )
            .map_err(security)?,
            pool: Arc::new(crate::task_routing::Pool::new(config.clone())),
            last_sample: Instant::now() - Duration::from_millis(250),
            config,
        });
        Ok(())
    }
    fn command(
        &mut self,
        cmd: Command,
        directory: &PrivateDir,
        budget: &ListenerBudget,
        phase: Option<&AtomicU8>,
    ) -> Result<Reply> {
        let result = match cmd {
            Command::ProbeListen { config, approved } => {
                if self.authority.is_none() || self.draining || self.probe_server.is_some() {
                    return Err("PROBE_ACTIVATION_REFUSED".into());
                }
                let server = crate::links::Server::bind(*config, &approved)?;
                let address = server.listener.local_addr().map_err(security)?;
                self.probe_server = Some(server);
                Reply::ok("PROBE_LISTENING", address.to_string())
            }
            Command::Probe { config, approved } => {
                if self.config.role != Role::Worker || self.draining {
                    return Err("PROBE_WORKER_REQUIRED".into());
                }
                if self.probe_worker.is_none() {
                    self.probe_worker =
                        Some(crate::links::Worker::new(*config.clone(), &approved)?);
                }
                self.probe_worker
                    .as_mut()
                    .ok_or("PROBE_STATE_UNAVAILABLE")?
                    .start(
                        &config,
                        &approved,
                        self.state.clone(),
                        self.fence.clone().ok_or("WORKER_REQUIRED")?,
                    )?;
                Reply::ok(
                    "PROBE_STARTED",
                    "Bounded background measurement; inspect cluster links NODE".into(),
                )
            }
            Command::Links { node } => {
                let view = if let (Some(server), Some(authority)) =
                    (&self.probe_server, &self.authority)
                {
                    server.view(authority, node)?
                } else if node == self.config.node {
                    self.probe_worker
                        .as_ref()
                        .ok_or("PROBE_NOT_ACTIVATED")?
                        .view(self.fence.as_ref().ok_or("WORKER_REQUIRED")?)?
                } else {
                    return Err("PROBE_NODE_REFUSED".into());
                };
                Reply::ok("LINKS", view.to_string())
            }
            Command::RecoverService {
                lease,
                configuration,
            } => {
                let repeated = self
                    .service
                    .as_mut()
                    .ok_or("WORKER_SERVICE_REQUIRED")?
                    .recover(lease, &configuration)?;
                Reply::ok(
                    "SERVICE_RECOVERED",
                    format!("qualified cleanup recovered; repeated={repeated}"),
                )
            }
            Command::Route { config, approved } => {
                config.validate(&approved).map_err(security)?;
                self.activate_route(*config)?;
                Reply::ok(
                    "ROUTE_ACTIVE",
                    "Explicitly approved fixed client route enabled".into(),
                )
            }
            Command::Service if self.stage.is_some() => Reply::ok(
                "SERVICE",
                serde_json::to_string(
                    &self
                        .stage
                        .as_ref()
                        .map(crate::stage::Owner::status)
                        .transpose()?,
                )
                .map_err(|_| "ENCODING_FAILED")?,
            ),
            Command::Service => Reply::ok(
                "SERVICE",
                serde_json::to_string(
                    &self
                        .service
                        .as_ref()
                        .map(crate::service::Owner::status)
                        .transpose()?,
                )
                .map_err(|_| "ENCODING_FAILED")?,
            ),
            Command::Resources { node } => {
                let view = self
                    .authority
                    .as_ref()
                    .ok_or("COORDINATOR_REQUIRED")?
                    .registry_resources(node)
                    .map_err(security)?;
                Reply::ok(
                    "RESOURCES",
                    serde_json::to_string(&view).map_err(|_| "ENCODING_FAILED")?,
                )
            }
            Command::Topology => topology(self.fence.clone(), self.addresses.clone(), self.mdns)?,
            Command::Nodes => {
                let nodes = self
                    .authority
                    .as_ref()
                    .ok_or("COORDINATOR_REQUIRED")?
                    .registry_nodes()
                    .map_err(security)?;
                Reply::ok(
                    "NODES",
                    serde_json::to_string(&nodes).map_err(|_| "ENCODING_FAILED")?,
                )
            }
            Command::Status => {
                let usage = budget.usage().map_err(security)?;
                let (registry, admission) = if let Some(authority) = &self.authority {
                    let (epoch, closed) = authority.registry_status().map_err(security)?;
                    (
                        format!("active epoch={}", epoch.number.0),
                        if closed { "closed" } else { "open" },
                    )
                } else if let Some(fence) = &self.fence {
                    let (epoch, status) = fence.status().map_err(security)?;
                    (
                        format!("{:?} epoch={}", status, epoch.number.0).to_lowercase(),
                        match status {
                            starlings_security::NodeStatus::Online => "open",
                            starlings_security::NodeStatus::Draining
                            | starlings_security::NodeStatus::Revoked => "closed",
                            starlings_security::NodeStatus::Offline => "unknown",
                        },
                    )
                } else {
                    ("unavailable".into(), "unknown")
                };
                Reply::ok(
                    if self.draining {
                        "DRAINING"
                    } else {
                        "ENROLLED_IDLE"
                    },
                    format!(
                        "role={} node={} cluster={} enrollment={} connections={} handshakes={} pending_admissions={} pending_approvals={} inference={} registry={registry} admission={admission} routing={}",
                        role_label(self.config.role),
                        self.config.node,
                        self.config.cluster,
                        self.addresses
                            .first()
                            .map(|a| a.to_string())
                            .unwrap_or_else(|| "disabled".into()),
                        usage.connections,
                        usage.handshakes,
                        usage.pending_admissions,
                        self.approvals.len(),
                        self.route
                            .as_ref()
                            .and_then(|r| r.listener.local_addr().ok())
                            .or_else(|| self
                                .pipeline
                                .as_ref()
                                .and_then(|p| p.listener.local_addr().ok()))
                            .or_else(|| self
                                .service
                                .as_ref()
                                .and_then(crate::service::Owner::endpoint))
                            .map(|a| a.to_string())
                            .unwrap_or_else(|| "unavailable".into()),
                        self.route
                            .as_ref()
                            .map(|r| r.pool.status())
                            .unwrap_or(serde_json::Value::Null)
                    ),
                )
            }
            Command::Drain | Command::Stop => {
                if let Some(service) = &self.service {
                    service.drain();
                }
                if let Some(stage) = &self.stage {
                    stage.drain();
                }
                self.stopping |= matches!(cmd, Command::Stop);
                self.draining = true;
                if let Some(fence) = &self.fence {
                    fence.drain().map_err(security)?;
                }
                self.approvals.clear();
                let reserved = self
                    .authority
                    .as_ref()
                    .map(|a| a.drain())
                    .transpose()
                    .map_err(security)?
                    .unwrap_or(0);
                Reply::ok(
                    if self.stopping {
                        "STOPPING"
                    } else {
                        "DRAINING"
                    },
                    format!(
                        "admission=closed reserved_bytes={reserved}; identity and model data retained"
                    ),
                )
            }
            Command::Leave => {
                if let Some(service) = &self.service {
                    service.drain();
                }
                if let Some(stage) = &self.stage {
                    stage.drain();
                }
                cluster::mark_left(directory, &self.config)?;
                self.stopping = true;
                self.draining = true;
                if let Some(fence) = &self.fence {
                    fence.drain().map_err(security)?;
                }
                self.approvals.clear();
                Reply::ok("LEFT_LOCAL", "Local membership disabled; identity and cache retained. Revoke this node on the coordinator.".into())
            }
            Command::Invite { role, output, via } => {
                let (issuer, endpoint) = self.invitation_issuer(role, via)?;
                publish_invitation(issuer, endpoint, role, output, phase)?
            }
            Command::InviteStatus { file } => {
                let authority = self.authority.as_ref().ok_or("COORDINATOR_REQUIRED")?;
                let (directory, name) = cluster::private_parent(&file)?;
                let active = authority
                    .invitation_active(&directory, &name)
                    .map_err(security)?;
                Reply::ok("INVITATION_STATUS", format!("active={active}"))
            }
            Command::PermissionsStatus { node } => {
                let scopes = self
                    .authority
                    .as_ref()
                    .ok_or("COORDINATOR_REQUIRED")?
                    .permissions(node)
                    .map_err(security)?;
                Reply::ok(
                    "PERMISSIONS_STATUS",
                    serde_json::to_string(&scopes).map_err(|_| "PERMISSIONS_ENCODING_FAILED")?,
                )
            }
            Command::Permissions { node, scopes } => {
                if self.draining {
                    return Err("ADMISSION_CLOSED".into());
                }
                self.authority
                    .as_ref()
                    .ok_or("COORDINATOR_REQUIRED")?
                    .set_permissions(node, &scopes)
                    .map_err(security)?;
                Reply::ok(
                    "PERMISSIONS_REPLACED",
                    "Complete model allowlist updated".into(),
                )
            }
            Command::Token { .. } => return Err("TOKEN_BACKGROUND_REQUIRED".into()),
            Command::TokenStatus { input } => {
                let authority = self.authority.as_ref().ok_or("COORDINATOR_REQUIRED")?;
                let (files, name) = cluster::private_parent(&input)?;
                let token =
                    starlings_security::ClientToken::load(&files, &name).map_err(security)?;
                let active = authority.client_token_active(&token).map_err(security)?;
                Reply::ok("TOKEN_STATUS", format!("active={active}"))
            }
            Command::RevokeToken { input } => {
                let (files, name) = cluster::private_parent(&input)?;
                let token =
                    starlings_security::ClientToken::load(&files, &name).map_err(security)?;
                self.authority
                    .as_ref()
                    .ok_or("COORDINATOR_REQUIRED")?
                    .revoke_client_token(&token)
                    .map_err(security)?;
                Reply::ok(
                    "TOKEN_REVOKED",
                    "Token and its live output grants revoked".into(),
                )
            }
            Command::Approve { node, fingerprint } => {
                if self.draining || self.authority.is_none() {
                    return Err("ADMISSION_CLOSED: approval refused".into());
                }
                if fingerprint.len() != 64
                    || !fingerprint
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                {
                    return Err("INVALID_FINGERPRINT".into());
                }
                if self.approvals.len() >= 8 && !self.approvals.contains_key(&node) {
                    return Err("APPROVAL_CAPACITY: wait for an approval to expire".into());
                }
                self.approvals.insert(
                    node,
                    (fingerprint, Instant::now() + Duration::from_secs(300)),
                );
                Reply::ok("FINGERPRINT_APPROVED", "One matching enrollment may connect within five minutes; run cluster join on that device.".into())
            }
            Command::Revoke { node } => {
                if node == self.config.node {
                    return Err("COORDINATOR_REVOCATION_REFUSED: stop this daemon instead".into());
                }
                self.authority
                    .as_ref()
                    .ok_or("COORDINATOR_REQUIRED")?
                    .revoke(node)
                    .map_err(security)?;
                self.approvals.remove(&node);
                Reply::ok("REVOKED", "Credential permanently revoked; the next authenticated heartbeat reports revocation, and disconnected registry authority expires within six seconds.".into())
            }
        };
        Ok(result)
    }
    fn enrollment(
        &mut self,
        pending: PendingEnrollment,
    ) -> Option<starlings_security::ApprovedEnrollment> {
        if self.draining {
            return None;
        }
        let valid = self
            .approvals
            .get(&pending.node())
            .is_some_and(|(fp, expires)| fp == pending.fingerprint() && *expires > Instant::now());
        if !valid {
            return None;
        }
        let (fingerprint, _) = self.approvals.remove(&pending.node())?;
        self.authority
            .as_mut()?
            .approve_for_delivery(pending, &fingerprint)
            .ok()
    }
}

pub fn run(
    paths: Paths,
    mdns: Option<IpAddr>,
    launch: Option<Launch>,
    ui: Option<starlings_cli::ui::Config>,
) -> Result<()> {
    let lock = cluster::lock_existing(&paths.state)?;
    if paths.state.join("left.json").exists() {
        return Err("LEFT_LOCAL: use a new state directory to enroll again".into());
    }
    let config = Config::load(lock.directory())?;
    if ui.is_some() && config.role != Role::Coordinator {
        return Err("COORDINATOR_REQUIRED: --ui is the coordinator's".into());
    }
    match &launch {
        Some(Launch::Warmup(..) | Launch::Worker(..) | Launch::Stage(..))
            if config.role != Role::Worker =>
        {
            return Err("WORKER_REQUIRED".into());
        }
        Some(Launch::Coordinator(..) | Launch::Pipeline(..))
            if config.role != Role::Coordinator =>
        {
            return Err("COORDINATOR_REQUIRED".into());
        }
        Some(Launch::Pipeline(r)) if r.plan.cluster_id != config.cluster => {
            return Err("CLUSTER_MISMATCH".into());
        }
        Some(Launch::Worker(r)) if r.qualification.replica.node != config.node => {
            return Err("IDENTITY_MISMATCH".into());
        }
        Some(Launch::Coordinator(r)) if r.plan.cluster_id != config.cluster => {
            return Err("CLUSTER_MISMATCH".into());
        }
        _ => (),
    }
    std::fs::create_dir_all(config.cache_dir.parent().ok_or("INVALID_CACHE_PATH")?)
        .map_err(|_| "CACHE_UNAVAILABLE")?;
    PrivateDir::ensure(&config.cache_dir).map_err(|_| "PRIVATE_CACHE_REQUIRED")?;
    let mut registry = None;
    let authority = if config.role == Role::Coordinator {
        let authority = Authority::open(&paths.state).map_err(security)?;
        if authority.identity().node() != config.node
            || authority.identity().cluster() != config.cluster
        {
            return Err("IDENTITY_MISMATCH".into());
        }
        Some(authority)
    } else {
        let identity = Identity::load(lock.directory(), "identity.json").map_err(security)?;
        let bootstrap = Invitation::load(lock.directory(), "bootstrap.json").map_err(security)?;
        bootstrap.verify_identity(&identity).map_err(security)?;
        if identity.node() != config.node
            || identity.cluster() != config.cluster
            || identity.role() != config.role
        {
            return Err("IDENTITY_MISMATCH".into());
        }
        let fallback = bootstrap.endpoint();
        let mut client =
            RegistryClient::open(&paths.state, identity, bootstrap).map_err(security)?;
        client.enable_persistent();
        client
            .configure_routes(
                if config.routes.is_empty() {
                    vec![fallback]
                } else {
                    config.routes.clone()
                },
                config.source,
            )
            .map_err(security)?;
        registry = Some(client);
        None
    };
    let stopping = Arc::new(AtomicBool::new(false));
    let signal = stopping.clone();
    // This standalone process owns termination; nohup may inherit ignored signals.
    ctrlc::set_handler(move || signal.store(true, Ordering::Relaxed))
        .map_err(|_| "SIGNAL_HANDLER_UNAVAILABLE")?;
    lock.cleanup_admin().map_err(security)?;
    let result = serve(
        &paths,
        &lock,
        Runtime {
            state: paths.state.clone(),
            authority,
            fence: registry.as_ref().map(RegistryClient::fence),
            registry,
            config,
            addresses: Vec::new(),
            mdns,
            draining: false,
            stopping: false,
            approvals: BTreeMap::new(),
            inviting: Arc::new(AtomicBool::new(false)),
            publishing_token: Arc::new(AtomicBool::new(false)),
            observing_topology: Arc::new(AtomicBool::new(false)),
            service: None,
            route: None,
            stage: None,
            pipeline: None,
            probe_server: None,
            probe_worker: None,
        },
        stopping,
        launch,
        ui,
    );
    let cleanup = lock.cleanup_admin().map_err(security);
    result.and(cleanup)
}
fn serve(
    paths: &Paths,
    lock: &StateLock,
    mut runtime: Runtime,
    signal: Arc<AtomicBool>,
    launch: Option<Launch>,
    ui: Option<starlings_cli::ui::Config>,
) -> Result<()> {
    let budget = ListenerBudget::default();
    let admin = AdminListener::bind(&paths.state, ListenerBudget::default()).map_err(security)?;
    let peers = runtime
        .config
        .listen
        .into_iter()
        .chain(runtime.config.additional_listen.iter().copied())
        .map(|address| PeerListener::bind(address, budget.clone()))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(security)?;
    runtime.addresses = peers
        .iter()
        .map(PeerListener::local_addr)
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(security)?;
    let _advertisement = runtime
        .mdns
        .map(|ip| {
            if runtime.config.role != Role::Coordinator {
                return Err("COORDINATOR_REQUIRED".into());
            }
            let address = runtime
                .addresses
                .iter()
                .find(|a| a.ip() == ip)
                .ok_or("MDNS_INTERFACE_NOT_LISTENING")?;
            crate::discovery::advertise(*address, runtime.config.node)
        })
        .transpose()?;
    // Publish assigned ports while the sockets remain bound; restart reuses them.
    if runtime.config.listen.is_some_and(|a| a.port() == 0)
        || runtime
            .config
            .additional_listen
            .iter()
            .any(|a| a.port() == 0)
    {
        runtime.config.listen = runtime.addresses.first().copied();
        runtime.config.additional_listen = runtime.addresses.iter().skip(1).copied().collect();
        runtime.config.save(lock.directory())?;
    }
    if let Some(authority) = &runtime.authority {
        authority.start_registry().map_err(security)?;
        authority.maintain(0).map_err(security)?;
    }
    let _ = writeln!(
        std::io::stdout(),
        "DAEMON_READY: role={} node={} enrollment={} inference=unavailable",
        role_label(runtime.config.role),
        runtime.config.node,
        runtime
            .addresses
            .first()
            .map(|a| a.to_string())
            .unwrap_or_else(|| "disabled".into())
    );
    let _ = std::io::stdout().flush();
    // Bounded admission: 63 established registry streams plus network and local admin slots.
    let (tx, rx) = mpsc::sync_channel::<Event>(128);
    let mut workers: Vec<JoinHandle<()>> = Vec::new();
    let start = Instant::now();
    let worker_route = match &launch {
        Some(Launch::Worker(r)) => Some(*r.clone()),
        _ => None,
    };
    let stage_route = match &launch {
        Some(Launch::Stage(r)) => Some(*r.clone()),
        _ => None,
    };
    match launch {
        Some(Launch::Warmup(config, approved)) => {
            runtime.service = Some(crate::service::Owner::start(
                runtime.config.cache_dir.join("models"),
                *config,
                approved,
                runtime.fence.clone().ok_or("WORKER_REQUIRED")?,
                None,
            )?)
        }
        Some(Launch::Worker(route)) => {
            runtime.service = Some(crate::service::Owner::start(
                runtime.config.cache_dir.join("models"),
                route.service.clone(),
                route.qualification.configuration.clone(),
                runtime.fence.clone().ok_or("WORKER_REQUIRED")?,
                Some((*route, paths.state.clone())),
            )?)
        }
        Some(Launch::Coordinator(route)) => {
            runtime.activate_route(*route)?;
        }
        Some(Launch::Stage(route)) => {
            runtime.stage = Some(crate::stage::Owner::start(
                *route,
                paths.state.clone(),
                runtime.fence.clone().ok_or("WORKER_REQUIRED")?,
            )?)
        }
        Some(Launch::Pipeline(route)) => {
            if runtime.authority.is_none() || runtime.route.is_some() {
                return Err("ROUTE_ACTIVATION_REFUSED".into());
            }
            runtime.pipeline = Some(Pipeline {
                listener: starlings_security::ClientListener::bind(
                    route.listen,
                    ListenerBudget::default(),
                )
                .map_err(security)?,
                slots: Arc::new(
                    (0..=route.stages[0].more_data.len())
                        .map(|_| AtomicBool::new(false))
                        .collect(),
                ),
                sequences: Arc::new(
                    route
                        .stages
                        .iter()
                        .map(|_| {
                            std::sync::Mutex::new(
                                std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .map_or(0, |d| d.as_micros() as u64),
                            )
                        })
                        .collect(),
                ),
                config: *route,
            });
        }
        None => (),
    }
    let service_report = runtime
        .service
        .as_ref()
        .map(crate::service::Owner::announcement);
    let stage_report = runtime
        .stage
        .as_ref()
        .map(crate::stage::Owner::announcement);
    let ui = ui
        .map(|config| ui_listener(paths, &runtime, config, signal.clone()))
        .transpose()?;
    let registry_failed = Arc::new(AtomicBool::new(false));
    if let Some(mut registry) = runtime.registry.take() {
        let failed = registry_failed.clone();
        let stopped = signal.clone();
        let resource_identity = (runtime.config.role == Role::Worker)
            .then_some((runtime.config.node, runtime.config.cluster));
        let memory = Arc::new(Mutex::new(None));
        if resource_identity.is_some() {
            let memory = memory.clone();
            let stopped = stopped.clone();
            workers.push(
                thread::Builder::new()
                    .name("registry-memory".into())
                    .spawn(move || {
                        while !stopped.load(Ordering::Relaxed) {
                            // Collect outside the mailbox lock; a blocked host API cannot block TLS.
                            let snapshot = registry_memory(&stopped);
                            let Ok(mut latest) = memory.lock() else {
                                stopped.store(true, Ordering::Relaxed);
                                return;
                            };
                            *latest = Some(snapshot);
                            drop(latest);
                            let next = Instant::now() + Duration::from_secs(1);
                            while Instant::now() < next && !stopped.load(Ordering::Relaxed) {
                                thread::sleep(POLL);
                            }
                        }
                    })
                    .map_err(|_| "THREAD_UNAVAILABLE")?,
            );
        }
        workers.push(
            thread::Builder::new()
                .name("registry-heartbeat".into())
                .spawn(move || {
                    let mut sample_sequence = 0u64;
                    while !stopped.load(Ordering::Relaxed) {
                        let sample = if let Some((node, cluster)) = resource_identity {
                            let floor = registry.sampling_floor();
                            let Some(next) = sample_sequence
                                .checked_add(1)
                                .map(|n| n.max(floor.map_or(1, |f| f.0)))
                            else {
                                failed.store(true, Ordering::Relaxed);
                                stopped.store(true, Ordering::Relaxed);
                                return;
                            };
                            sample_sequence = next;
                            let capabilities = memory.lock().ok().and_then(|snapshot| {
                                snapshot.as_ref()?.capabilities(node, cluster)
                            });
                            capabilities.and_then(|mut capabilities| {
                                if floor.is_some_and(|(_, rejected)| rejected == capabilities.snapshot_id) {
                                    return None;
                                }
                                if let (Some(route), Some(report)) = (&worker_route, &service_report) {
                                    crate::service::qualify_sample(report, route, &mut capabilities);
                                }
                                if let (Some(route), Some(report)) = (&stage_route, &stage_report) {
                                    crate::stage::qualify_sample(report, route, &mut capabilities);
                                }
                                Some(starlings_security::ResourceSample {
                                    sequence: starlings_protocol::Decimal(next),
                                    capabilities,
                                    links: Vec::new(),
                                })
                            })
                        } else {
                            None
                        };
                        let began = Instant::now();
                        let heartbeat = registry.heartbeat(sample).inspect_err(|_error| {
                            #[cfg(feature = "test-backend")]
                            let _ = writeln!(std::io::stderr(), "REGISTRY_HEARTBEAT_FAILED reason={_error:?}");
                        });
                        if heartbeat == Err(starlings_security::Error::ReenrollmentRequired) {
                            eprintln!("REENROLLMENT_REQUIRED: registry counter is stale; use a new state directory and enroll again");
                        }
                        if matches!(
                            heartbeat,
                            Err(starlings_security::Error::Storage
                                | starlings_security::Error::ReenrollmentRequired
                                | starlings_security::Error::Capacity)
                        ) {
                            let _ = registry.fence().drain();
                            failed.store(true, Ordering::Relaxed);
                            stopped.store(true, Ordering::Relaxed);
                            return;
                        }
                        if registry.fence().status().is_ok_and(|(_, status)| {
                            status == starlings_security::NodeStatus::Revoked
                        }) {
                            return;
                        }
                        let delay = heartbeat_delay();
                        let next = next_heartbeat(began, Instant::now(), delay, heartbeat.is_err());
                        while Instant::now() < next && !stopped.load(Ordering::Relaxed) {
                            thread::sleep(POLL);
                        }
                    }
                    let _ = registry.fence().drain();
                    let _ = registry.heartbeat(None);
                })
                .map_err(|_| "THREAD_UNAVAILABLE")?,
        );
    }
    let mut maintained = Instant::now();
    let mut ui_published = Instant::now();
    if let (Some(ui), Some(authority)) = (&ui, &runtime.authority) {
        ui.publish(starlings_cli::ui::snapshot(authority, ui_route(&runtime)));
    }
    let result = (|| -> Result<()> {
        while !runtime.stopping && !signal.load(Ordering::Relaxed) {
            runtime
                .approvals
                .retain(|_, (_, deadline)| *deadline > Instant::now());
            if maintained.elapsed() >= Duration::from_millis(250) {
                if let Some(authority) = &mut runtime.authority {
                    match authority.refresh_coordinator() {
                        Ok(()) | Err(starlings_security::Error::Busy) => (),
                        Err(error) => return Err(security(error)),
                    }
                    let maintenance = authority.maintain(
                        start
                            .elapsed()
                            .as_millis()
                            .try_into()
                            .map_err(|_| "CLOCK_OVERFLOW")?,
                    );
                    match maintenance {
                        Ok(()) | Err(starlings_security::Error::Busy) => (),
                        Err(error) => return Err(security(error)),
                    }
                }
                if let Some(fence) = &runtime.fence {
                    fence
                        .maintain(
                            start
                                .elapsed()
                                .as_millis()
                                .try_into()
                                .map_err(|_| "CLOCK_OVERFLOW")?,
                        )
                        .map_err(security)?;
                }
                maintained = Instant::now();
            }
            for event in rx.try_iter().take(32) {
                match event {
                    Event::Admin(command, reply, stage) => {
                        runtime.admin(
                            command,
                            lock.directory(),
                            &budget,
                            reply,
                            stage,
                            &mut workers,
                        )?;
                    }
                    Event::Heartbeat(pending) => {
                        if let Some(authority) = &runtime.authority {
                            let pending = pending.refresh(authority).map_err(security)?;
                            let tx = tx.clone();
                            workers.push(thread::spawn(move || {
                                if let Ok(Some(next)) = pending.respond_next() {
                                    let _ = tx.try_send(Event::Heartbeat(Box::new(next)));
                                }
                            }));
                        }
                    }
                    Event::Enrollment(pending) => {
                        if let Some(approved) = runtime.enrollment(*pending) {
                            workers.push(
                                thread::Builder::new()
                                    .name("enrollment-reply".into())
                                    .spawn(move || {
                                        let _ = approved.deliver();
                                    })
                                    .map_err(|_| "THREAD_UNAVAILABLE")?,
                            );
                        }
                    }
                }
            }
            reap(&mut workers)?;
            if runtime.stopping {
                break;
            }
            if let Some(route) = &mut runtime.route {
                if route.last_sample.elapsed() >= Duration::from_millis(250) {
                    let authority = runtime.authority.as_ref().ok_or("COORDINATOR_REQUIRED")?;
                    let resources: Vec<_> = route
                        .pool
                        .workers()
                        .iter()
                        .map(|worker| {
                            // Age each view from before its own read, including that read's cost.
                            let captured = Instant::now();
                            let view = authority
                                .registry_resources(worker.qualification.replica.node)
                                .ok()
                                .flatten();
                            (view, captured)
                        })
                        .collect();
                    if authority
                        .registry_epoch()
                        .and_then(|epoch| route.pool.refresh(epoch, &resources))
                        .is_err()
                    {
                        route.pool.invalidate();
                    }
                    route.last_sample = Instant::now();
                }
                match route.listener.try_accept() {
                    Ok(Some(incoming)) => {
                        let authority = runtime.authority.as_ref().ok_or("COORDINATOR_REQUIRED")?;
                        if let (Ok(reader), Ok(epoch)) =
                            (authority.enrollment_reader(), authority.registry_epoch())
                        {
                            let admission = authority.client_admission();
                            let config = route.config.clone();
                            let pool = route.pool.clone();
                            workers.push(
                                thread::Builder::new()
                                    .name("client-generation".into())
                                    .spawn(move || {
                                        let _ = dispatch_client(
                                            incoming, admission, reader, config, pool, epoch,
                                        )
                                        .inspect_err(|_error| {
                                            #[cfg(feature = "test-backend")]
                                            let _ = writeln!(
                                                std::io::stderr(),
                                                "CLIENT_DISPATCH_EXIT reason={_error:?}"
                                            );
                                        });
                                    })
                                    .map_err(|_| "THREAD_UNAVAILABLE")?,
                            );
                        }
                    }
                    Ok(None) | Err(starlings_security::Error::Capacity) => (),
                    Err(_) => (),
                }
            }
            if let Some(pipeline) = &runtime.pipeline
                && let Ok(Some(incoming)) = pipeline.listener.try_accept()
            {
                let authority = runtime.authority.as_ref().ok_or("COORDINATOR_REQUIRED")?;
                if let (Ok(reader), Ok(epoch)) =
                    (authority.enrollment_reader(), authority.registry_epoch())
                {
                    let admission = authority.client_admission();
                    let config = pipeline.config.clone();
                    let (slots, sequences) = (pipeline.slots.clone(), pipeline.sequences.clone());
                    workers.push(
                        thread::Builder::new()
                            .name("pipeline-generation".into())
                            .spawn(move || {
                                let _ = starlings_cli::pipeline::dispatch(
                                    incoming, &admission, &reader, &config, epoch, &slots,
                                    &sequences,
                                )
                                .inspect_err(|error| {
                                    let _ = writeln!(
                                        std::io::stderr(),
                                        "PIPELINE_REQUEST_FAILED reason={error:?}"
                                    );
                                });
                            })
                            .map_err(|_| "THREAD_UNAVAILABLE")?,
                    );
                }
            }
            if let Some(ui) = &ui {
                if ui_published.elapsed() >= Duration::from_secs(1) {
                    let authority = runtime.authority.as_ref().ok_or("COORDINATOR_REQUIRED")?;
                    ui.publish(starlings_cli::ui::snapshot(authority, ui_route(&runtime)));
                    ui_published = Instant::now();
                }
                if let Some(job) = ui.try_accept() {
                    workers.push(job);
                }
            }
            match admin.try_accept() {
                Ok(Some(request)) => {
                    let tx = tx.clone();
                    workers.push(
                        thread::Builder::new()
                            .name("local-admin".into())
                            .spawn(move || {
                                let _ =
                                    request.respond(|command: Command| control_reply(&tx, command));
                            })
                            .map_err(|_| "THREAD_UNAVAILABLE")?,
                    );
                }
                Ok(None) | Err(starlings_security::Error::Capacity) => (),
                Err(error) => return Err(security(error)),
            }
            for peer in &peers {
                for _ in 0..8 {
                    match peer.try_accept_registry(
                        runtime.authority.as_ref().ok_or("COORDINATOR_REQUIRED")?,
                    ) {
                        Ok(Some(incoming)) => {
                            let reader = runtime
                                .authority
                                .as_ref()
                                .ok_or("COORDINATOR_REQUIRED")?
                                .enrollment_reader();
                            let reader = match reader {
                                Ok(reader) => reader,
                                Err(starlings_security::Error::Busy) => continue,
                                Err(error) => return Err(security(error)),
                            };
                            let tx = tx.clone();
                            workers.push(
                                thread::Builder::new()
                                    .name("enrollment".into())
                                    .spawn(move || match incoming.registry(reader) {
                                        Ok(RegistryRequest::Enrollment(pending)) => {
                                            let _ = tx.try_send(Event::Enrollment(pending));
                                        }
                                        Ok(RegistryRequest::Heartbeat(pending)) => {
                                            let _ = tx.try_send(Event::Heartbeat(pending));
                                        }
                                        Err(_) => (),
                                    })
                                    .map_err(|_| "THREAD_UNAVAILABLE")?,
                            );
                        }
                        Ok(None) | Err(starlings_security::Error::Capacity) => break,
                        Err(error) => return Err(security(error)),
                    }
                }
            }
            if let Some(server) = &runtime.probe_server {
                let authority = runtime.authority.as_ref().ok_or("COORDINATOR_REQUIRED")?;
                match server.listener.try_accept_registry(authority) {
                    Ok(Some(incoming)) => {
                        let reader = authority.enrollment_reader().map_err(security)?;
                        let budget = server.budget.clone();
                        let samples = server.samples.clone();
                        workers.push(
                            thread::Builder::new()
                                .name("probe-transfer".into())
                                .spawn(move || {
                                    if let Ok(observation) = incoming.probe(reader, &budget) {
                                        let node = observation.report.transfers[0].source_node;
                                        if let Ok(mut samples) = samples.lock()
                                            && (samples.len() < 64 || samples.contains_key(&node))
                                        {
                                            samples.insert(node, observation);
                                        }
                                    }
                                })
                                .map_err(|_| "THREAD_UNAVAILABLE")?,
                        );
                    }
                    Ok(None) | Err(starlings_security::Error::Capacity) => (),
                    Err(error) => return Err(security(error)),
                }
            }
            thread::sleep(POLL);
        }
        Ok(())
    })();
    signal.store(true, Ordering::Relaxed);
    if let Some(service) = &runtime.service {
        service.drain();
    }
    if let Some(stage) = &runtime.stage {
        stage.drain();
    }
    drop(peers);
    runtime.probe_server = None;
    drop(admin);
    drop(rx);
    drop(tx);
    runtime.approvals.clear();
    // Network I/O is bounded; private filesystem publication may outlive its caller.
    // Retain its owned worker until it returns; drain denies late credential activation.
    // Keep policy maintenance alive while joining.
    while workers.iter().any(|h| !h.is_finished()) {
        if let Some(authority) = &runtime.authority {
            let _ = authority.drain();
            let _ = authority.maintain(start.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
        }
        if let Some(fence) = &runtime.fence {
            let _ = fence.drain();
            let _ = fence.maintain(start.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
        }
        thread::sleep(POLL);
    }
    let joined = reap(&mut workers);
    let service_joined = runtime
        .service
        .as_mut()
        .map(crate::service::Owner::join)
        .transpose();
    let stage_joined = runtime
        .stage
        .as_mut()
        .map(crate::stage::Owner::join)
        .transpose();
    result?;
    joined?;
    service_joined?;
    stage_joined?;
    if let Some(worker) = &mut runtime.probe_worker {
        worker.join()?;
    }
    if let Some(fence) = &runtime.fence {
        fence.drain().map_err(security)?;
        if fence
            .maintain(
                start
                    .elapsed()
                    .as_millis()
                    .try_into()
                    .map_err(|_| "CLOCK_OVERFLOW")?,
            )
            .map_err(security)?
            != 0
        {
            return Err("DRAIN_INCOMPLETE: backend completion required; state retained".into());
        }
    }
    if registry_failed.load(Ordering::Relaxed) {
        return Err("REGISTRY_STATE_FAILED: daemon stopped; state retained".into());
    }
    if let Some(authority) = &runtime.authority
        && authority.drain().map_err(security)? != 0
    {
        return Err("DRAIN_INCOMPLETE: backend completion required; state retained".into());
    }
    let _ = writeln!(
        std::io::stdout(),
        "STOPPED: listeners closed, transient endpoints removed on exit; identity and model data retained"
    );
    Ok(())
}
/// GLY-364: the opt-in local web UI. Its chat acts as `--ui-client`, which must already hold
/// infer permission for the pipeline's model; without a pipeline or a client, chat is off.
fn ui_listener(
    paths: &Paths,
    runtime: &Runtime,
    config: starlings_cli::ui::Config,
    stop: Arc<AtomicBool>,
) -> Result<starlings_cli::ui::Ui> {
    let authority = runtime.authority.as_ref().ok_or("COORDINATOR_REQUIRED")?;
    let chat = match (&runtime.pipeline, config.client) {
        (Some(pipeline), Some(client)) => {
            let model = pipeline.config.plan.manifest_digest.clone();
            if !authority
                .permissions(client)
                .map_err(security)?
                .iter()
                .any(|s| s.model == model && s.action == starlings_security::Action::Infer)
            {
                return Err(
                    "UI_CLIENT_REFUSED: --ui-client needs infer permission for the pipeline's model"
                        .into(),
                );
            }
            Some(starlings_cli::ui::Chat {
                address: pipeline.listener.local_addr().map_err(security)?,
                model,
                lifetime_ms: pipeline.config.lifetime_ms,
                client,
                issuer: authority.client_token_issuer(),
            })
        }
        (None, Some(_)) => return Err("UI_CLIENT_REFUSED: chat needs --pipeline".into()),
        _ => None,
    };
    let join = serde_json::json!({"state_dir": paths.state.display().to_string(),
        "enrollment": runtime.addresses});
    let ui = starlings_cli::ui::Ui::bind(config.listen, chat, join, stop)?;
    let _ = writeln!(std::io::stdout(), "UI_READY: {}", ui.url());
    let _ = std::io::stdout().flush();
    Ok(ui)
}
fn ui_route(runtime: &Runtime) -> Option<(&starlings_cli::routes::PipelineRoute, SocketAddr)> {
    let pipeline = runtime.pipeline.as_ref()?;
    Some((&pipeline.config, pipeline.listener.local_addr().ok()?))
}
fn topology(
    fence: Option<RegistryFence>,
    addresses: Vec<SocketAddr>,
    mdns: Option<IpAddr>,
) -> Result<Reply> {
    #[cfg(test)]
    assert!(
        tests::TOPOLOGY_ALLOWED.get(),
        "topology OS query on control thread"
    );
    let routes = fence
        .as_ref()
        .map(RegistryFence::topology)
        .transpose()
        .map_err(security)?
        .unwrap_or_default();
    let record = serde_json::json!({
        "policy": starlings_security::network::POLICY,
        "interfaces": starlings_security::network::interfaces().map_err(security)?,
        "listeners": addresses,
        "mdns_interface": mdns,
        "direction": "this_node_to_coordinator",
        "routes": routes,
        "reverse_reachability": "unknown",
        "peer_to_peer_reachability": "unknown",
        "discovery_confers_trust": false
    });
    Ok(Reply::ok("TOPOLOGY", record.to_string()))
}
fn finish_admin(reply: SyncSender<Reply>, stage: &AtomicU8, result: Result<Reply>, began: Instant) {
    stage.store(3, Ordering::Release);
    if began.elapsed() >= Duration::from_secs(3) {
        eprintln!(
            "CONTROL_EXECUTION_SLOW elapsed_ms={}",
            began.elapsed().as_millis()
        );
    }
    let _ = reply.try_send(result.unwrap_or_else(Reply::error));
}
fn admin_job(
    pending: Arc<AtomicBool>,
    reply: SyncSender<Reply>,
    stage: Arc<AtomicU8>,
    name: &'static str,
    busy: &'static str,
    work: impl FnOnce() -> Result<Reply> + Send + 'static,
) -> Result<Option<JoinHandle<()>>> {
    if pending
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        finish_admin(reply, &stage, Err(busy.into()), Instant::now());
        // A timed-out caller cannot create another job while its work is still pending.
        return Ok(None);
    }
    let owned = pending.clone();
    let job = thread::Builder::new().name(name.into()).spawn(move || {
        struct Release(Arc<AtomicBool>);
        impl Drop for Release {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Release);
            }
        }
        let _release = Release(owned);
        let began = Instant::now();
        finish_admin(reply, &stage, work(), began);
    });
    job.map(Some).map_err(|_| {
        pending.store(false, Ordering::Release);
        "THREAD_UNAVAILABLE".into()
    })
}
fn publish_token(
    issuer: starlings_security::ClientTokenIssuer,
    node: NodeId,
    scopes: Vec<starlings_security::Scope>,
    lifetime: u32,
    output: std::path::PathBuf,
) -> Result<Reply> {
    let (files, name) = cluster::private_parent(&output)?;
    issuer
        .issue_saved(node, &scopes, lifetime, &files, &name)
        .map_err(security)?;
    Ok(Reply::ok(
        "TOKEN_CREATED",
        "Scoped token saved to the private output file; never print or log it".into(),
    ))
}
fn publish_invitation(
    issuer: InvitationIssuer,
    endpoint: SocketAddr,
    role: Role,
    output: std::path::PathBuf,
    phase: Option<&AtomicU8>,
) -> Result<Reply> {
    if let Some(phase) = phase {
        phase.store(11, Ordering::Release);
    }
    let (directory, name) = cluster::private_parent(&output)?;
    issuer
        .invite_observed(&directory, &name, endpoint, role, |step| {
            if let Some(phase) = phase {
                phase.store(
                    match step {
                        "clock" => 12,
                        "policy_lock" => 13,
                        "expired_rows" => 14,
                        "transaction" => 15,
                        "publish_private_file" => 16,
                        "commit" => 17,
                        _ => 18,
                    },
                    Ordering::Release,
                );
            }
        })
        .map_err(security)?;
    Ok(Reply::ok(
        "INVITATION_CREATED",
        "Private invitation file created; expires in five minutes. Transfer it privately.".into(),
    ))
}
fn reap(workers: &mut Vec<JoinHandle<()>>) -> Result<()> {
    let mut i = 0;
    while i < workers.len() {
        if workers[i].is_finished() {
            workers
                .swap_remove(i)
                .join()
                .map_err(|_| "WORKER_FAILED: stopping safely")?;
        } else {
            i += 1;
        }
    }
    Ok(())
}
fn control_reply(tx: &SyncSender<Event>, command: Command) -> Reply {
    let (reply, received) = mpsc::sync_channel(1);
    let stage = Arc::new(AtomicU8::new(0));
    if tx
        .try_send(Event::Admin(command, reply, stage.clone()))
        .is_err()
    {
        return Reply::error("CONTROL_BUSY".into());
    }
    received
        .recv_timeout(Duration::from_secs(3))
        .unwrap_or_else(|_| {
            if stage
                .compare_exchange(0, 2, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                eprintln!("CONTROL_WAIT_EXPIRED phase=queued");
                Reply::error("CONTROL_EXPIRED: not executed".into())
            } else {
                let observed = stage.load(Ordering::Acquire);
                let phase = match observed {
                    1 | 11..=18 => "executing",
                    3 => "finished",
                    _ => "unavailable",
                };
                let step = match observed {
                    11 => "private_parent",
                    12 => "clock",
                    13 => "policy_lock",
                    14 => "expired_rows",
                    15 => "transaction",
                    16 => "publish_private_file",
                    17 => "commit",
                    18 => "finished",
                    _ => "none",
                };
                let _ = writeln!(
                    std::io::stderr(),
                    "CONTROL_WAIT_EXPIRED phase={phase} invitation_step={step}"
                );
                Reply::error("CONTROL_UNAVAILABLE: outcome unknown; check status".into())
            }
        })
}

fn next_heartbeat(began: Instant, finished: Instant, delay: Duration, failed: bool) -> Instant {
    if failed {
        finished + delay / 8
    } else {
        began + delay
    }
}

fn registry_memory(_stopped: &AtomicBool) -> starlings_adapters::resources::HostSnapshot {
    #[cfg(feature = "test-backend")]
    if let Some(release) = std::env::var_os("STARLINGS_TEST_MEMORY_RELEASE") {
        let release = std::path::PathBuf::from(release);
        let _ = std::fs::write(release.with_extension("entered"), b"");
        let until = Instant::now() + Duration::from_secs(30);
        while !release.exists() && !_stopped.load(Ordering::Relaxed) {
            if Instant::now() >= until {
                let _ = std::fs::write(release.with_extension("expired"), b"");
                break;
            }
            thread::sleep(POLL);
        }
    }
    starlings_adapters::resources::HostSnapshot::collect_memory()
}

fn dispatch_client(
    incoming: starlings_security::IncomingClient,
    admission: starlings_security::ClientAdmission,
    reader: starlings_security::EnrollmentReader,
    config: starlings_cli::routes::CoordinatorRoute,
    pool: Arc<crate::task_routing::Pool>,
    epoch: starlings_security::AuthorityEpoch,
) -> std::result::Result<(), starlings_security::Error> {
    use starlings_security::{Error, GenerationBinding};
    let q = &config.qualification;
    let binding = GenerationBinding {
        version: starlings_protocol::Version,
        epoch,
        model: q.replica.manifest.clone(),
        profile: q.replica.profile.clone(),
    };
    let (chat, response) = starlings_cli::inference::admit(incoming, &admission, &binding)?;
    let fail = |response: starlings_security::ClientResponse, error| {
        let id = uuid::Uuid::new_v4()
            .to_string()
            .parse()
            .map_err(|_| Error::Invalid)?;
        let _ = response.fail(&starlings_cli::inference::failure(error, id));
        Err(error)
    };
    if let Err(error) = q.admit(chat.max_tokens.get()) {
        return fail(response, error);
    }
    let mut excluded = std::collections::BTreeSet::new();
    let mut last = Error::Capacity;
    let deadline = pool.deadline()?;
    let bytes = serde_json::to_vec(&chat)?.len();
    let client_watch = response.watch()?;
    for _ in 0..2 {
        client_watch.check()?;
        let mut pending = match pool.submit(epoch, response.client(), &excluded, deadline, bytes) {
            Ok(pending) => pending,
            Err(error) => return fail(response, error),
        };
        let selected = loop {
            client_watch.check().inspect_err(|_error| {
                #[cfg(feature = "test-backend")]
                let _ = writeln!(
                    std::io::stderr(),
                    "CLIENT_QUEUE_EXIT phase=watch reason={_error:?}"
                );
            })?;
            match pending.poll(epoch) {
                Ok(Some(selected)) => break selected,
                Ok(None) => thread::sleep(POLL),
                Err(error) => {
                    #[cfg(feature = "test-backend")]
                    let _ = writeln!(
                        std::io::stderr(),
                        "CLIENT_QUEUE_EXIT phase=pool reason={error:?}"
                    );
                    return fail(response, error);
                }
            }
        };
        let q = &selected.config.qualification;
        let binding = GenerationBinding {
            version: starlings_protocol::Version,
            epoch,
            model: q.replica.manifest.clone(),
            profile: q.replica.profile.clone(),
        };
        let mut owner = match selected.owner.try_lock() {
            Ok(owner) => owner,
            Err(_) => {
                pool.drained(selected.id)?;
                excluded.insert(q.replica.node);
                continue;
            }
        };
        if owner
            .as_ref()
            .is_none_or(|(previous, _)| *previous != epoch)
        {
            let mut plan = selected.config.plan.clone();
            plan.authority_epoch = epoch.number;
            plan.plan_digest =
                starlings_protocol::canonical_digest(&serde_json::to_value(&plan)?, "plan_digest")?;
            *owner = Some((
                epoch,
                starlings_cli::inference::Dispatch::new(binding.clone(), plan, &q.replica, 60_000)?,
            ));
        }
        let channel = (|| {
            let socket = std::net::TcpStream::connect_timeout(
                &selected.config.worker_endpoint,
                Duration::from_secs(2),
            )?;
            reader.connect_generation(socket, q.replica.node, binding)
        })();
        let channel = match channel {
            Ok(channel) => channel,
            Err(error) => {
                // No Count/Apply/Generate was sent: this failure cannot mutate backend state.
                pool.drained(selected.id)?;
                excluded.insert(q.replica.node);
                last = error;
                continue;
            }
        };
        let finished = std::sync::atomic::AtomicBool::new(false);
        struct Finish<'a>(&'a std::sync::atomic::AtomicBool);
        impl Drop for Finish<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        let watch = channel.watch();
        let result = thread::scope(|threads| {
            threads.spawn(|| {
                while !finished.load(Ordering::Acquire) {
                    if pool.cancelled(selected.id) {
                        watch.cancel();
                        break;
                    }
                    thread::sleep(POLL);
                }
            });
            let _finished = Finish(&finished);
            owner
                .as_mut()
                .ok_or(Error::Invalid)?
                .1
                .run(response, &chat, channel)
        });
        // Count is read-only. No mutating control was attempted: release only
        // router bookkeeping, never worker buffers or service quarantine.
        let rejected_before_control = !owner.as_ref().ok_or(Error::Invalid)?.1.mutation_possible();
        drop(owner);
        if result.is_ok() || rejected_before_control {
            pool.drained(selected.id)?;
        } else {
            pool.failed(selected.id, q.replica.node);
        }
        return result;
    }
    fail(response, last)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn successful_heartbeat_latency_does_not_extend_the_next_interval() {
        let began = Instant::now();
        let exchange = Duration::from_millis(2500);
        let finished = began + exchange;
        for delay in [Duration::from_secs(2), Duration::from_secs(3)] {
            let next = next_heartbeat(began, finished, delay, false);
            assert!(
                next.max(finished) + exchange < began + starlings_security::REGISTRY_TTL,
                "two valid exchanges must not expire the fence because of added idle delay"
            );
            assert_eq!(next, began + delay);
            assert_eq!(
                next_heartbeat(began, finished, delay, true),
                finished + delay / 8,
                "failed exchanges retain their existing paced backoff"
            );
        }
    }
    std::thread_local! {
        pub(super) static TOPOLOGY_ALLOWED: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    }
    #[test]
    fn expired_queued_admin_command_cannot_start_later() {
        let (tx, rx) = mpsc::sync_channel(1);
        let client = thread::spawn(move || control_reply(&tx, Command::Status));
        let Event::Admin(_, reply_sender, stage) = rx.recv_timeout(Duration::from_secs(1)).unwrap()
        else {
            panic!("admin event required")
        };
        let reply = client.join().unwrap();
        drop(reply_sender);
        assert!(reply.message.contains("not executed"));
        assert_eq!(stage.load(Ordering::Acquire), 2);
        assert!(
            stage
                .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        );
    }
    #[test]
    fn started_admin_timeout_keeps_an_unknown_outcome() {
        let (tx, rx) = mpsc::sync_channel(1);
        let client = thread::spawn(move || control_reply(&tx, Command::Status));
        let Event::Admin(_, reply_sender, stage) = rx.recv_timeout(Duration::from_secs(1)).unwrap()
        else {
            panic!("admin event required")
        };
        stage
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .unwrap();
        stage.store(16, Ordering::Release);
        let reply = client.join().unwrap();
        drop(reply_sender);
        assert!(reply.message.contains("outcome unknown"));
        assert_eq!(stage.load(Ordering::Acquire), 16);
    }
    #[test]
    fn drain_status_does_not_confuse_request_intent_with_authority_withdrawal() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("coordinator");
        let authority = Authority::create(&path).unwrap();
        let epoch = authority.start_registry().unwrap();
        let directory = PrivateDir::open(&path).unwrap();
        let mut runtime = Runtime {
            state: path.clone(),
            config: Config {
                version: starlings_protocol::Version,
                role: Role::Coordinator,
                cluster: authority.identity().cluster(),
                node: authority.identity().node(),
                listen: None,
                additional_listen: vec![],
                routes: vec![],
                source: None,
                cache_dir: temp.path().join("cache"),
            },
            authority: Some(authority),
            registry: None,
            fence: None,
            addresses: vec![],
            mdns: None,
            draining: false,
            stopping: false,
            approvals: BTreeMap::new(),
            inviting: Arc::new(AtomicBool::new(false)),
            publishing_token: Arc::new(AtomicBool::new(false)),
            observing_topology: Arc::new(AtomicBool::new(false)),
            service: None,
            route: None,
            stage: None,
            pipeline: None,
            probe_server: None,
            probe_worker: None,
        };
        let budget = ListenerBudget::default();
        let status = runtime
            .command(Command::Status, &directory, &budget, None)
            .unwrap();
        assert!(status.message.contains("admission=open"));
        let database = rusqlite::Connection::open(path.join("enrollment.sqlite")).unwrap();
        database.execute_batch("BEGIN EXCLUSIVE").unwrap();
        let failed = runtime.command(Command::Drain, &directory, &budget, None);
        database.execute_batch("ROLLBACK").unwrap();
        assert_eq!(failed.err().unwrap(), "SECURITY_Busy: operation refused");
        let status = runtime
            .command(Command::Status, &directory, &budget, None)
            .unwrap();
        assert_eq!(status.code, "DRAINING");
        assert!(
            status.message.contains("admission=open"),
            "{}",
            status.message
        );
        assert_eq!(
            runtime
                .authority
                .as_ref()
                .unwrap()
                .registry_status()
                .unwrap(),
            (epoch, false)
        );

        // A new explicit command is the caller's decision, never an automatic retry.
        let drained = runtime
            .command(Command::Drain, &directory, &budget, None)
            .unwrap();
        assert_eq!(drained.code, "DRAINING");
        let status = runtime
            .command(Command::Status, &directory, &budget, None)
            .unwrap();
        assert!(status.message.contains("admission=closed"));
        assert_eq!(
            runtime
                .authority
                .as_ref()
                .unwrap()
                .registry_status()
                .unwrap(),
            (epoch, true)
        );
        assert!(drained.message.contains("reserved_bytes=0"));
        runtime.authority = None;
        let status = runtime
            .command(Command::Status, &directory, &budget, None)
            .unwrap();
        assert!(status.message.contains("admission=unknown"));
    }
    #[test]
    fn background_admin_keeps_status_available_and_stop_remains_latched() {
        let temp = tempfile::tempdir().unwrap();
        let directory = PrivateDir::ensure(temp.path().join("profile")).unwrap();
        let id = "00000000-0000-4000-8000-000000000001";
        let mut runtime = Runtime {
            state: temp.path().join("profile"),
            authority: None,
            registry: None,
            fence: None,
            config: Config {
                version: starlings_protocol::Version,
                role: Role::Worker,
                cluster: id.parse().unwrap(),
                node: id.parse().unwrap(),
                listen: None,
                additional_listen: vec![],
                routes: vec![],
                source: None,
                cache_dir: temp.path().join("cache"),
            },
            addresses: vec![],
            mdns: None,
            draining: false,
            stopping: false,
            approvals: BTreeMap::new(),
            inviting: Arc::new(AtomicBool::new(false)),
            publishing_token: Arc::new(AtomicBool::new(false)),
            observing_topology: Arc::new(AtomicBool::new(false)),
            service: None,
            route: None,
            stage: None,
            pipeline: None,
            probe_server: None,
            probe_worker: None,
        };
        // Exercise the actual admin dispatch with host queries forbidden on this thread.
        let (topology_reply, topology_result) = mpsc::sync_channel(1);
        let mut topology_workers = Vec::new();
        TOPOLOGY_ALLOWED.set(false);
        runtime
            .admin(
                Command::Topology,
                &directory,
                &ListenerBudget::default(),
                topology_reply,
                Arc::new(AtomicU8::new(0)),
                &mut topology_workers,
            )
            .unwrap();
        TOPOLOGY_ALLOWED.set(true);
        let topology = topology_result
            .recv_timeout(Duration::from_secs(3))
            .unwrap();
        assert!(topology.ok, "{}", topology.message);
        assert_eq!(topology.code, "TOPOLOGY");
        let record: serde_json::Value = serde_json::from_str(&topology.message).unwrap();
        assert_eq!(record["discovery_confers_trust"], false);
        assert_eq!(record["routes"], serde_json::json!([]));
        assert_eq!(record["listeners"], serde_json::json!([]));
        for job in topology_workers {
            job.join().unwrap();
        }
        let authority = Authority::create(&temp.path().join("authority")).unwrap();
        authority.start_registry().unwrap();
        runtime.authority = Some(authority);
        let token_command = || Command::Token {
            node: id.parse().unwrap(),
            scopes: vec![],
            lifetime: 1,
            output: temp.path().join("profile/token.json"),
        };
        let (token_reply, token_result) = mpsc::sync_channel(1);
        let mut token_workers = Vec::new();
        runtime
            .admin(
                token_command(),
                &directory,
                &ListenerBudget::default(),
                token_reply,
                Arc::new(AtomicU8::new(0)),
                &mut token_workers,
            )
            .unwrap();
        assert_eq!(
            token_workers.len(),
            1,
            "token publication ran on control thread"
        );
        token_workers.pop().unwrap().join().unwrap();
        assert!(
            token_result
                .recv()
                .unwrap()
                .message
                .starts_with("SECURITY_Invalid")
        );
        assert!(!runtime.publishing_token.load(Ordering::Acquire));
        assert!(!temp.path().join("profile/token.json").exists());
        for (pending, name, busy_code) in [
            (
                runtime.inviting.clone(),
                "invitation-publication",
                "INVITATION_BUSY",
            ),
            (
                runtime.publishing_token.clone(),
                "token-publication",
                "TOKEN_BUSY",
            ),
            (
                runtime.observing_topology.clone(),
                "topology-observation",
                "TOPOLOGY_BUSY",
            ),
        ] {
            let (entered, paused) = mpsc::sync_channel(1);
            let (release, resume) = mpsc::sync_channel(1);
            let (reply, result) = mpsc::sync_channel(1);
            let stage = Arc::new(AtomicU8::new(1));
            let job = admin_job(
                pending.clone(),
                reply,
                stage.clone(),
                name,
                busy_code,
                move || {
                    entered.send(()).unwrap();
                    resume.recv_timeout(Duration::from_secs(3)).unwrap();
                    Ok(Reply::ok("COMPLETED", String::new()))
                },
            )
            .unwrap()
            .unwrap();
            paused.recv_timeout(Duration::from_secs(3)).unwrap();
            let (second_reply, second_result) = mpsc::sync_channel(1);
            let mut workers = Vec::new();
            if busy_code == "TOPOLOGY_BUSY" || busy_code == "TOKEN_BUSY" {
                runtime
                    .admin(
                        if busy_code == "TOKEN_BUSY" {
                            token_command()
                        } else {
                            Command::Topology
                        },
                        &directory,
                        &ListenerBudget::default(),
                        second_reply,
                        Arc::new(AtomicU8::new(0)),
                        &mut workers,
                    )
                    .unwrap();
                assert!(workers.is_empty(), "busy command spawned another job");
            } else {
                assert!(
                    admin_job(
                        pending.clone(),
                        second_reply,
                        Arc::new(AtomicU8::new(1)),
                        name,
                        busy_code,
                        || { panic!("second job must not start") }
                    )
                    .unwrap()
                    .is_none()
                );
            }
            let busy = second_result.recv().unwrap();
            let (status_reply, status_result) = mpsc::sync_channel(1);
            runtime
                .admin(
                    Command::Status,
                    &directory,
                    &ListenerBudget::default(),
                    status_reply,
                    Arc::new(AtomicU8::new(0)),
                    &mut workers,
                )
                .unwrap();
            let responsive = status_result
                .recv_timeout(Duration::from_millis(250))
                .is_ok_and(|reply| reply.ok && reply.code == "ENROLLED_IDLE");
            if busy_code == "TOPOLOGY_BUSY" {
                let (stop_reply, stop_result) = mpsc::sync_channel(1);
                runtime
                    .admin(
                        Command::Stop,
                        &directory,
                        &ListenerBudget::default(),
                        stop_reply,
                        Arc::new(AtomicU8::new(0)),
                        &mut workers,
                    )
                    .unwrap();
                assert_eq!(stop_result.recv().unwrap().code, "STOPPING");
            }
            // Release before asserting: a failed regression must not strand the job.
            release.send(()).unwrap();
            job.join().unwrap();
            assert_eq!(busy.code, "REFUSED");
            assert_eq!(busy.message, busy_code);
            assert!(responsive, "paused background job blocked sibling status");
            assert_eq!(result.recv().unwrap().code, "COMPLETED");
            assert_eq!(stage.load(Ordering::Acquire), 3);
            assert!(!pending.load(Ordering::Acquire));
        }
        // A failed observation with a departed caller must still release its permit.
        let (reply, result) = mpsc::sync_channel(1);
        drop(result);
        let stage = Arc::new(AtomicU8::new(1));
        admin_job(
            runtime.observing_topology.clone(),
            reply,
            stage.clone(),
            "topology-observation",
            "TOPOLOGY_BUSY",
            || Err("OBSERVATION_FAILED".into()),
        )
        .unwrap()
        .unwrap()
        .join()
        .unwrap();
        assert!(!runtime.observing_topology.load(Ordering::Acquire));
        assert_eq!(stage.load(Ordering::Acquire), 3);
        let budget = ListenerBudget::default();
        assert_eq!(
            runtime
                .command(Command::Stop, &directory, &budget, None)
                .unwrap()
                .code,
            "STOPPING"
        );
        assert_eq!(
            runtime
                .command(Command::Drain, &directory, &budget, None)
                .unwrap()
                .code,
            "STOPPING"
        );
        assert!(runtime.stopping && runtime.draining);
    }
}
