//! Local CLI configuration and authenticated administrative commands.
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use starlings_protocol::{ClusterId, NodeId, Version};
use starlings_security::{
    Authority, Identity, Invitation, PrivateDir, Role, StateLock, admin_call,
};
use std::{
    ffi::OsString,
    io::{Read, Write},
    net::SocketAddr,
    path::{Component, Path, PathBuf},
};

pub type Result<T> = std::result::Result<T, String>;
pub fn security(error: starlings_security::Error) -> String {
    match error {
        starlings_security::Error::ProfileInUse => {
            "PROFILE_IN_USE: another process owns this profile".into()
        }
        starlings_security::Error::AddressInUse => {
            "PORT_IN_USE: selected endpoint is already occupied".into()
        }
        _ => format!("SECURITY_{error:?}: operation refused"),
    }
}
fn storage<T>(_: T) -> String {
    "STATE_UNAVAILABLE: check the private state directory".into()
}
pub fn lock_existing(path: &Path) -> Result<StateLock> {
    if !path.is_dir() {
        return Err("STATE_DIR_NOT_FOUND: initialize or select an existing profile".into());
    }
    StateLock::acquire(path).map_err(security)
}
pub const USAGE: &str = "INVALID_ARGUMENTS: use --help";

fn acknowledgement(mut output: impl Write, message: std::fmt::Arguments<'_>) -> Result<()> {
    writeln!(output, "{message}")
        .and_then(|()| output.flush())
        .map_err(|_| {
            "OUTPUT_UNAVAILABLE: command completed but output failed; inspect state before retrying"
                .into()
        })
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub version: Version,
    pub role: Role,
    pub cluster: ClusterId,
    pub node: NodeId,
    pub listen: Option<SocketAddr>,
    pub cache_dir: PathBuf,
    #[serde(default)]
    pub additional_listen: Vec<SocketAddr>,
    #[serde(default)]
    pub routes: Vec<SocketAddr>,
    #[serde(default)]
    pub source: Option<SocketAddr>,
}
impl Config {
    pub fn load(directory: &PrivateDir) -> Result<Self> {
        let config: Self = read(directory, "config.json")?;
        absolute(&config.cache_dir)?;
        let listeners: Vec<_> = config
            .listen
            .into_iter()
            .chain(config.additional_listen.iter().copied())
            .collect();
        if listeners.len() > starlings_security::network::MAX_ROUTES
            || listeners.iter().enumerate().any(|(i, a)| {
                !starlings_security::network::local_address(*a, true)
                    || (a.port() != 0 && listeners[..i].contains(a))
            })
            || (config.role != Role::Coordinator && !listeners.is_empty())
            || (config.role == Role::Coordinator
                && (!config.routes.is_empty() || config.source.is_some()))
        {
            return Err("INVALID_CONFIG: network policy refused".into());
        }
        if !config.routes.is_empty() {
            starlings_security::network::validate_routes(&config.routes, config.source)
                .map_err(security)?;
        } else if config.source.is_some() {
            return Err("INVALID_CONFIG: source requires explicit routes".into());
        }
        match (config.role, config.listen) {
            (Role::Coordinator, Some(addr))
                if !addr.ip().is_unspecified() && !addr.ip().is_multicast() => {}
            (Role::Worker | Role::Controller, None) => (),
            _ => return Err("INVALID_CONFIG: role or enrollment address is invalid".into()),
        }
        Ok(config)
    }
    pub(crate) fn save(&self, directory: &PrivateDir) -> Result<()> {
        directory
            .write_atomic_durable("config.json", &serde_json::to_vec(self).map_err(storage)?)
            .map_err(security_fs)
    }
}
fn security_fs(_: impl std::fmt::Display) -> String {
    "PRIVATE_STATE_REQUIRED: unsafe or existing state refused".into()
}
pub fn read<T: DeserializeOwned>(directory: &PrivateDir, name: &str) -> Result<T> {
    let file = directory.open_file(name).map_err(security_fs)?;
    let mut bytes = Vec::new();
    file.take(16 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(storage)?;
    if bytes.len() > 16 * 1024 {
        return Err("INVALID_CONFIG: record too large".into());
    }
    serde_json::from_value(starlings_protocol::parse_json(&bytes).map_err(storage)?)
        .map_err(storage)
}
pub fn absolute(path: &Path) -> Result<()> {
    if !path.is_absolute()
        || path
            .components()
            .any(|c| matches!(c, Component::ParentDir | Component::CurDir))
    {
        return Err("INVALID_PATH: use an absolute path without dot segments".into());
    }
    Ok(())
}
pub fn private_parent(path: &Path) -> Result<(PrivateDir, String)> {
    absolute(path)?;
    let parent = path.parent().ok_or(USAGE)?;
    let name = path
        .file_name()
        .and_then(|v| v.to_str())
        .ok_or(USAGE)?
        .to_owned();
    Ok((PrivateDir::open(parent).map_err(security_fs)?, name))
}
pub struct Paths {
    pub state: PathBuf,
    pub cache: PathBuf,
}
impl Paths {
    pub fn parse(args: &[OsString]) -> Result<(Self, &[OsString])> {
        let (state, rest) = if args.first().is_some_and(|a| a == "--state-dir") {
            let path = PathBuf::from(args.get(1).ok_or(USAGE)?);
            (path, &args[2..])
        } else {
            (
                dirs::state_dir()
                    .or_else(dirs::data_local_dir)
                    .ok_or("HOME_UNAVAILABLE: use --state-dir")?
                    .join("starlings"),
                args,
            )
        };
        absolute(&state)?;
        let cache = if args.first().is_some_and(|a| a == "--state-dir") {
            state.join("cache")
        } else {
            dirs::cache_dir()
                .ok_or("HOME_UNAVAILABLE: use --state-dir")?
                .join("starlings")
        };
        // Windows local data and cache resolve to the same OS directory.
        let cache = if cache == state {
            cache.join("cache")
        } else {
            cache
        };
        absolute(&cache)?;
        Ok((Self { state, cache }, rest))
    }
    fn lock_new(&self) -> Result<StateLock> {
        std::fs::create_dir_all(self.state.parent().ok_or(USAGE)?).map_err(storage)?;
        StateLock::acquire(&self.state).map_err(security)
    }
    pub fn cache(&self) -> Result<()> {
        std::fs::create_dir_all(self.cache.parent().ok_or(USAGE)?).map_err(storage)?;
        PrivateDir::ensure(&self.cache).map_err(security_fs)?;
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case", deny_unknown_fields)]
pub enum Command {
    Status,
    Service,
    RecoverService {
        lease: starlings_protocol::LeaseId,
        configuration: starlings_protocol::Digest,
    },
    Nodes,
    Resources {
        node: NodeId,
    },
    Topology,
    ProbeListen {
        config: Box<crate::links::ServerConfig>,
        approved: starlings_protocol::Digest,
    },
    Probe {
        config: Box<starlings_security::probes::ProbeRoute>,
        approved: starlings_protocol::Digest,
    },
    Links {
        node: NodeId,
    },
    Drain,
    Stop,
    Leave,
    Invite {
        role: Role,
        output: PathBuf,
        via: Option<SocketAddr>,
    },
    InviteStatus {
        file: PathBuf,
    },
    Approve {
        node: NodeId,
        fingerprint: String,
    },
    Revoke {
        node: NodeId,
    },
    PermissionsStatus {
        node: NodeId,
    },
    Permissions {
        node: NodeId,
        scopes: Vec<starlings_security::Scope>,
    },
    Token {
        node: NodeId,
        scopes: Vec<starlings_security::Scope>,
        lifetime: u32,
        output: PathBuf,
    },
    RevokeToken {
        input: PathBuf,
    },
    TokenStatus {
        input: PathBuf,
    },
    Route {
        config: Box<starlings_cli::routes::CoordinatorRoute>,
        approved: starlings_protocol::Digest,
    },
}
pub use starlings_cli::AdminReply as Reply;
fn node(arg: &OsString) -> Result<NodeId> {
    arg.to_str().ok_or(USAGE)?.parse().map_err(|_| USAGE.into())
}
fn fingerprint(arg: &OsString) -> Result<String> {
    let value = arg.to_str().ok_or(USAGE)?;
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(USAGE.into());
    }
    Ok(value.into())
}
pub fn role_label(role: Role) -> &'static str {
    match role {
        Role::Coordinator => "coordinator",
        Role::Worker => "worker",
        Role::Controller => "client",
    }
}
pub fn run(args: &[OsString]) -> Result<()> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        crate::help();
        return Ok(());
    }
    let mut normalized = Vec::new();
    let mut state = None;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        let value = if arg == "--state-dir" {
            Some(iter.next().ok_or(USAGE)?.clone())
        } else {
            arg.to_str()
                .and_then(|s| s.strip_prefix("--state-dir="))
                .map(OsString::from)
        };
        if let Some(value) = value {
            if state.replace(value).is_some() {
                return Err(USAGE.into());
            }
        } else {
            normalized.push(arg.clone());
        }
    }
    if let Some(state) = state {
        normalized.splice(0..0, [OsString::from("--state-dir"), state]);
    }
    let (paths, args) = Paths::parse(&normalized)?;
    if let [cmd, action, output] = args
        && cmd == "service"
        && action == "template"
    {
        return crate::service_template::run(&paths, Path::new(output));
    }
    if let Some((cmd, rest)) = args.split_first()
        && cmd == "model"
    {
        return crate::models::run(paths, rest);
    }
    // GLY-364: `daemon ... --ui 127.0.0.1:PORT [--ui-client NODE]`, after any other daemon form.
    let (args, ui) = match args.iter().position(|a| a == "--ui") {
        Some(i) if args.first().is_some_and(|a| a == "daemon") => {
            let client = match &args[i + 1..] {
                [_] => None,
                [_, flag, node] if flag == "--ui-client" => {
                    Some(node.to_str().ok_or(USAGE)?.parse().map_err(|_| USAGE)?)
                }
                _ => return Err(USAGE.into()),
            };
            let listen = args[i + 1].to_str().ok_or(USAGE)?;
            let listen = listen.parse().map_err(|_| USAGE)?;
            (
                &args[..i],
                Some(starlings_cli::ui::Config { listen, client }),
            )
        }
        _ => (args, None),
    };
    if args == [OsString::from("daemon")] {
        return crate::daemon::run(paths, None, None, ui);
    }
    if let [cmd, flag, file, approve, digest] = args
        && cmd == "daemon"
        && flag == "--service"
        && approve == "--approve"
    {
        let (directory, name) = private_parent(Path::new(file))?;
        let config = read::<starlings_adapters::service::ServiceConfig>(&directory, &name)?;
        let digest = digest
            .to_str()
            .ok_or(USAGE)?
            .to_owned()
            .try_into()
            .map_err(|_| USAGE)?;
        return crate::daemon::run(
            paths,
            None,
            Some(crate::daemon::Launch::Warmup(Box::new(config), digest)),
            ui,
        );
    }
    if let [cmd, flag, file, approve, digest] = args
        && cmd == "daemon"
        && (flag == "--stage" || flag == "--pipeline")
        && approve == "--approve"
    {
        let (directory, name) = private_parent(Path::new(file))?;
        let approved: starlings_protocol::Digest = digest
            .to_str()
            .ok_or(USAGE)?
            .to_owned()
            .try_into()
            .map_err(|_| USAGE)?;
        let launch = if flag == "--stage" {
            let route: starlings_cli::routes::StageRoute = read(&directory, &name)?;
            route.validate(&approved).map_err(security)?;
            crate::daemon::Launch::Stage(Box::new(route))
        } else {
            let route: starlings_cli::routes::PipelineRoute = read(&directory, &name)?;
            route.validate(&approved).map_err(security)?;
            crate::daemon::Launch::Pipeline(Box::new(route))
        };
        return crate::daemon::run(paths, None, Some(launch), ui);
    }
    if let [cmd, flag, file, approve, digest] = args
        && cmd == "daemon"
        && (flag == "--execute" || flag == "--route")
        && approve == "--approve"
    {
        let (directory, name) = private_parent(Path::new(file))?;
        let approved: starlings_protocol::Digest = digest
            .to_str()
            .ok_or(USAGE)?
            .to_owned()
            .try_into()
            .map_err(|_| USAGE)?;
        let launch = if flag == "--execute" {
            let route: starlings_cli::routes::WorkerRoute = read(&directory, &name)?;
            route.validate(&approved).map_err(security)?;
            crate::daemon::Launch::Worker(Box::new(route))
        } else {
            let route: starlings_cli::routes::CoordinatorRoute = read(&directory, &name)?;
            route.validate(&approved).map_err(security)?;
            crate::daemon::Launch::Coordinator(Box::new(route))
        };
        return crate::daemon::run(paths, None, Some(launch), ui);
    }
    if let [cmd, flag, ip] = args
        && cmd == "daemon"
        && flag == "--mdns"
    {
        return crate::daemon::run(
            paths,
            Some(ip.to_str().ok_or(USAGE)?.parse().map_err(|_| USAGE)?),
            None,
            ui,
        );
    }
    if let [cmd, rest @ ..] = args
        && cmd == "up"
    {
        return crate::up::run(&paths.state, rest);
    }
    if let [cmd, rest @ ..] = args
        && cmd == "join"
    {
        return crate::up::join(&paths.state, rest);
    }
    let [prefix, rest @ ..] = args else {
        return Err(USAGE.into());
    };
    if prefix != "cluster" {
        return Err(USAGE.into());
    }
    let command = match rest {
        [cmd] if cmd == "diagnostics" => {
            let bundle = starlings_cli::diagnostics::collect(&paths.state);
            return starlings_cli::diagnostics::write(&bundle, std::io::stdout().lock())
                .map_err(str::to_owned);
        }
        [cmd, flags @ ..] if cmd == "init" => {
            let mut addresses = Vec::new();
            for pair in flags.chunks(2) {
                let [flag, addr] = pair else {
                    return Err(USAGE.into());
                };
                if flag != "--listen" {
                    return Err(USAGE.into());
                }
                addresses.push(addr.to_str().ok_or(USAGE)?.parse().map_err(|_| USAGE)?);
            }
            if addresses.is_empty() {
                addresses.push("127.0.0.1:9741".parse().map_err(storage)?);
            }
            return init(paths, addresses);
        }
        [cmd, flag, ip] if cmd == "discover" && flag == "--interface" => {
            return crate::discovery::browse(ip.to_str().ok_or(USAGE)?.parse().map_err(|_| USAGE)?);
        }
        [cmd, args @ ..] if cmd == "routes" => return routes(paths, args),
        [cmd, file] if cmd == "probe-digest" => {
            let (files, name) = private_parent(Path::new(file))?;
            let value: serde_json::Value = read(&files, &name)?;
            if let Ok(config) = serde_json::from_value::<crate::links::ServerConfig>(value.clone())
            {
                config.limits.validate().map_err(security)?;
                if !starlings_security::network::local_address(config.listen, true) {
                    return Err("PROBE_ADDRESS_REFUSED".into());
                }
                println!(
                    "{}",
                    starlings_protocol::content_hash(&config).map_err(|_| "PROBE_INVALID")?
                );
            } else {
                let route: starlings_security::probes::ProbeRoute =
                    serde_json::from_value(value).map_err(|_| "PROBE_INVALID")?;
                route.validate().map_err(security)?;
                println!(
                    "{}",
                    starlings_protocol::content_hash(&route).map_err(|_| "PROBE_INVALID")?
                );
            }
            return Ok(());
        }
        [cmd, invite, flag, addr] if cmd == "join" && flag == "--via" => {
            return join(
                paths,
                Path::new(invite),
                Some(addr.to_str().ok_or(USAGE)?.parse().map_err(|_| USAGE)?),
            );
        }
        [cmd, invite] if cmd == "prepare" => return prepare(paths, Path::new(invite)),
        [cmd, invite] if cmd == "join" => return join(paths, Path::new(invite), None),
        [cmd] if cmd == "status" => Command::Status,
        [cmd] if cmd == "service" => Command::Service,
        [cmd] if cmd == "nodes" => Command::Nodes,
        [cmd, id] if cmd == "resources" => Command::Resources { node: node(id)? },
        [cmd] if cmd == "topology" => Command::Topology,
        [cmd, id] if cmd == "links" => Command::Links { node: node(id)? },
        [cmd, id, output] if cmd == "links-record" => {
            return crate::links::record(&paths, node(id)?, Path::new(output));
        }
        [cmd, file, flag, digest]
            if (cmd == "probe-listen" || cmd == "probe") && flag == "--approve" =>
        {
            let (directory, name) = private_parent(Path::new(file))?;
            let approved: starlings_protocol::Digest = digest
                .to_str()
                .ok_or(USAGE)?
                .to_owned()
                .try_into()
                .map_err(|_| USAGE)?;
            if cmd == "probe-listen" {
                Command::ProbeListen {
                    config: Box::new(read(&directory, &name)?),
                    approved,
                }
            } else {
                Command::Probe {
                    config: Box::new(read(&directory, &name)?),
                    approved,
                }
            }
        }
        [cmd] if cmd == "drain" => Command::Drain,
        [cmd] if cmd == "stop" => Command::Stop,
        [cmd] if cmd == "leave" => Command::Leave,
        [cmd, file] if cmd == "invite-status" => Command::InviteStatus {
            file: PathBuf::from(file),
        },
        [cmd, role, output, rest @ ..] if cmd == "invite" => Command::Invite {
            role: match role.to_str() {
                Some("worker") => Role::Worker,
                Some("client") => Role::Controller,
                _ => return Err(USAGE.into()),
            },
            output: PathBuf::from(output),
            via: match rest {
                [] => None,
                [flag, addr] if flag == "--via" => {
                    Some(addr.to_str().ok_or(USAGE)?.parse().map_err(|_| USAGE)?)
                }
                _ => return Err(USAGE.into()),
            },
        },
        [cmd, id, fp] if cmd == "approve" => Command::Approve {
            node: node(id)?,
            fingerprint: fingerprint(fp)?,
        },
        [cmd, id] if cmd == "revoke" => Command::Revoke { node: node(id)? },
        [cmd, id] if cmd == "permissions-status" => Command::PermissionsStatus { node: node(id)? },
        [cmd, id, file] if cmd == "permissions" => {
            let (files, name) = private_parent(Path::new(file))?;
            Command::Permissions {
                node: node(id)?,
                scopes: read(&files, &name)?,
            }
        }
        [cmd, id, file, lifetime, output] if cmd == "token" => {
            let (files, name) = private_parent(Path::new(file))?;
            Command::Token {
                node: node(id)?,
                scopes: read(&files, &name)?,
                lifetime: lifetime.to_str().ok_or(USAGE)?.parse().map_err(|_| USAGE)?,
                output: PathBuf::from(output),
            }
        }
        [cmd, lease, configuration] if cmd == "recover-service" => Command::RecoverService {
            lease: lease.to_str().ok_or(USAGE)?.parse().map_err(|_| USAGE)?,
            configuration: configuration
                .to_str()
                .ok_or(USAGE)?
                .to_owned()
                .try_into()
                .map_err(|_| USAGE)?,
        },
        [cmd, input] if cmd == "revoke-token" => Command::RevokeToken {
            input: PathBuf::from(input),
        },
        [cmd, input] if cmd == "token-status" => Command::TokenStatus {
            input: PathBuf::from(input),
        },
        [cmd, file, flag, digest] if cmd == "route" && flag == "--approve" => {
            let (files, name) = private_parent(Path::new(file))?;
            Command::Route {
                config: Box::new(read(&files, &name)?),
                approved: digest
                    .to_str()
                    .ok_or(USAGE)?
                    .to_owned()
                    .try_into()
                    .map_err(|_| USAGE)?,
            }
        }
        _ => return Err(USAGE.into()),
    };
    let reply: Reply = match admin_call(&paths.state, &command) {
        Ok(reply) => reply,
        Err(starlings_security::Error::AdminOutcomeUnknown) => {
            return Err("REFUSED: CONTROL_UNAVAILABLE: outcome unknown; check status".into());
        }
        Err(error) if matches!(command, Command::Leave) => {
            let lock = lock_existing(&paths.state).map_err(|lock_error| {
                format!("OFFLINE_LEAVE_REFUSED: {lock_error}; admin request failed ({error})")
            })?;
            let config = Config::load(lock.directory())?;
            mark_left(lock.directory(), &config)?;
            Reply::ok("LEFT_LOCAL", "Local membership disabled; identity and cache retained. Revoke this node on the coordinator.".into())
        }
        Err(error) => {
            return Err(format!(
                "DAEMON_UNAVAILABLE: start the daemon or check private endpoint permissions ({error})"
            ));
        }
    };
    if !reply.ok {
        return Err(format!("{}: {}", reply.code, reply.message));
    }
    acknowledgement(
        std::io::stdout().lock(),
        format_args!("{}: {}", reply.code, reply.message),
    )
}
fn init(paths: Paths, addresses: Vec<SocketAddr>) -> Result<()> {
    if addresses.is_empty()
        || addresses.len() > starlings_security::network::MAX_ROUTES
        || addresses.iter().enumerate().any(|(i, a)| {
            !starlings_security::network::local_address(*a, true)
                || (a.port() != 0 && addresses[..i].contains(a))
        })
    {
        return Err("INVALID_CONFIG: choose the machine's concrete LAN or loopback IP".into());
    }
    let listen = addresses[0];
    #[cfg(any(debug_assertions, feature = "test-backend"))]
    let trace_exit = std::env::var_os("STARLINGS_TEST_INIT_EXIT_PHASE").is_some();
    #[cfg(any(debug_assertions, feature = "test-backend"))]
    if trace_exit {
        eprintln!("INIT_START_PHASE=state_lock");
    }
    let lock = paths.lock_new()?;
    if paths.state.join("config.json").exists() || paths.state.join("pending.json").exists() {
        return Err("ALREADY_INITIALIZED: use a separate state directory".into());
    }
    #[cfg(any(debug_assertions, feature = "test-backend"))]
    if trace_exit {
        eprintln!("INIT_START_PHASE=cache");
    }
    paths.cache()?;
    #[cfg(any(debug_assertions, feature = "test-backend"))]
    if trace_exit {
        eprintln!("INIT_START_PHASE=authority_create");
    }
    #[cfg(any(debug_assertions, feature = "test-backend"))]
    let authority = Authority::create_observed(&paths.state, |phase| {
        if trace_exit {
            eprintln!("INIT_AUTHORITY_PHASE={phase}");
        }
    })
    .map_err(security)?;
    #[cfg(not(any(debug_assertions, feature = "test-backend")))]
    let authority = Authority::create(&paths.state).map_err(security)?;
    let identity = authority.identity();
    #[cfg(any(debug_assertions, feature = "test-backend"))]
    if trace_exit {
        eprintln!("INIT_START_PHASE=config_publish");
    }
    Config {
        version: Version,
        role: Role::Coordinator,
        cluster: identity.cluster(),
        node: identity.node(),
        listen: Some(listen),
        additional_listen: addresses[1..].to_vec(),
        routes: Vec::new(),
        source: None,
        cache_dir: paths.cache,
    }
    .save(lock.directory())?;
    let output = acknowledgement(
        std::io::stdout().lock(),
        format_args!(
            "INITIALIZED: role=coordinator node={} cluster={} configured_listener={listen}; run daemon to bind and report enrollment address",
            identity.node(),
            identity.cluster()
        ),
    );
    #[cfg(any(debug_assertions, feature = "test-backend"))]
    if trace_exit {
        eprintln!("INIT_EXIT_PHASE=authority_drop");
    }
    drop(authority);
    #[cfg(any(debug_assertions, feature = "test-backend"))]
    if trace_exit {
        eprintln!("INIT_EXIT_PHASE=state_lock_drop");
    }
    drop(lock);
    #[cfg(any(debug_assertions, feature = "test-backend"))]
    if trace_exit {
        eprintln!("INIT_EXIT_PHASE=resources_closed");
    }
    output
}
fn prepare(paths: Paths, invite: &Path) -> Result<()> {
    let (input, name) = private_parent(invite)?;
    let invitation = Invitation::load(&input, &name).map_err(security)?;
    let lock = paths.lock_new()?;
    if paths.state.join("config.json").exists() || paths.state.join("left.json").exists() {
        return Err("ALREADY_INITIALIZED: use a separate state directory".into());
    }
    let identity = if paths.state.join("pending.json").exists() {
        Identity::load(lock.directory(), "pending.json").map_err(security)?
    } else {
        let id = Identity::generate(invitation.cluster()).map_err(security)?;
        id.save(lock.directory(), "pending.json")
            .map_err(security)?;
        id
    };
    if identity.cluster() != invitation.cluster() {
        return Err("CLUSTER_MISMATCH: pending identity belongs to another cluster".into());
    }
    acknowledgement(
        std::io::stdout().lock(),
        format_args!(
            "APPROVAL_REQUIRED: node={} fingerprint={}; compare on the coordinator, run cluster approve NODE FINGERPRINT, then cluster join INVITATION",
            identity.node(),
            identity.request_fingerprint()
        ),
    )
}
fn join(paths: Paths, invite: &Path, via: Option<SocketAddr>) -> Result<()> {
    let (input, name) = private_parent(invite)?;
    let mut invitation = Invitation::load(&input, &name).map_err(security)?;
    if let Some(via) = via {
        invitation.set_endpoint(via).map_err(security)?;
    }
    let lock = paths.lock_new()?;
    if paths.state.join("config.json").exists() || paths.state.join("left.json").exists() {
        return Err("ALREADY_INITIALIZED: use a separate state directory".into());
    }
    let pending = Identity::load(lock.directory(), "pending.json")
        .map_err(|_| "APPROVAL_REQUIRED: run cluster prepare INVITATION first".to_string())?;
    let identity = if paths.state.join("identity.json").exists() {
        let id = Identity::load(lock.directory(), "identity.json").map_err(security)?;
        if id.node() != pending.node() || id.request_fingerprint() != pending.request_fingerprint()
        {
            return Err("IDENTITY_MISMATCH: recovery refused".into());
        }
        id
    } else {
        let mut id = pending;
        invitation.enroll(&mut id).map_err(security)?;
        id.save(lock.directory(), "identity.json")
            .map_err(security)?;
        id
    };
    invitation.verify_identity(&identity).map_err(security)?;
    paths.cache()?;
    lock.directory()
        .write_atomic_durable(
            "bootstrap.json",
            &serde_json::to_vec(&invitation).map_err(storage)?,
        )
        .map_err(security_fs)?;
    Config {
        version: Version,
        role: identity.role(),
        cluster: identity.cluster(),
        node: identity.node(),
        listen: None,
        additional_listen: Vec::new(),
        routes: Vec::new(),
        source: None,
        cache_dir: paths.cache,
    }
    .save(lock.directory())?;
    acknowledgement(
        std::io::stdout().lock(),
        format_args!(
            "JOINED: role={} node={} cluster={}; run daemon",
            role_label(identity.role()),
            identity.node(),
            identity.cluster()
        ),
    )
}
fn routes(paths: Paths, args: &[OsString]) -> Result<()> {
    let mut routes = Vec::new();
    let mut source = None;
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        if arg == "--source" {
            if source.is_some() {
                return Err(USAGE.into());
            }
            source = Some(
                args.next()
                    .ok_or(USAGE)?
                    .to_str()
                    .ok_or(USAGE)?
                    .parse()
                    .map_err(|_| USAGE)?,
            );
        } else {
            routes.push(arg.to_str().ok_or(USAGE)?.parse().map_err(|_| USAGE)?);
        }
    }
    starlings_security::network::validate_routes(&routes, source).map_err(security)?;
    let lock = lock_existing(&paths.state)?;
    let mut config = Config::load(lock.directory())?;
    if config.role == Role::Coordinator {
        return Err("REMOTE_PROFILE_REQUIRED".into());
    }
    config.routes = routes;
    config.source = source;
    config.save(lock.directory())?;
    acknowledgement(
        std::io::stdout().lock(),
        format_args!(
            "ROUTES_SAVED: explicit addresses retain the enrolled certificate pin; start daemon"
        ),
    )
}
pub fn mark_left(directory: &PrivateDir, config: &Config) -> Result<()> {
    if config.role == Role::Coordinator {
        return Err("COORDINATOR_LEAVE_REFUSED: stop the coordinator instead".into());
    }
    directory
        .write_atomic_durable("left.json", b"{\"version\":1}")
        .map_err(security_fs)
}

#[cfg(test)]
mod tests {
    #[test]
    fn completed_acknowledgement_preserves_text_and_reports_write_or_flush_failure() {
        use std::io::{self, Write};

        struct FailedOutput {
            fail_flush: bool,
        }
        impl Write for FailedOutput {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                if self.fail_flush {
                    Ok(bytes.len())
                } else {
                    Err(io::ErrorKind::BrokenPipe.into())
                }
            }
            fn flush(&mut self) -> io::Result<()> {
                Err(io::ErrorKind::BrokenPipe.into())
            }
        }

        let mut output = Vec::new();
        super::acknowledgement(&mut output, format_args!("{}: {}", "DONE", "state saved")).unwrap();
        assert_eq!(output, b"DONE: state saved\n");
        for fail_flush in [false, true] {
            assert_eq!(
                super::acknowledgement(FailedOutput { fail_flush }, format_args!("DONE: state saved")),
                Err("OUTPUT_UNAVAILABLE: command completed but output failed; inspect state before retrying".into())
            );
        }
    }

    #[test]
    fn native_default_paths_keep_disposable_cache_separate_from_identity() {
        let (paths, _) = super::Paths::parse(&[]).unwrap();
        assert_ne!(paths.state, paths.cache);
        assert!(paths.state.is_absolute() && paths.cache.is_absolute());
    }
}
