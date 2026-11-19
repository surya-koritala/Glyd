//! Explicit, separately admitted link diagnostics; no backend or inference qualification.
use crate::cluster::{Result, security};
use serde::{Deserialize, Serialize};
use starlings_protocol::{Digest, NodeId, Version};
use starlings_security::{
    Authority, Identity, Invitation, ListenerBudget, PeerListener, PrivateDir, RegistryFence,
    probes::{ProbeBudget, ProbeLimits, ProbeObservation, ProbeRoute},
};
use std::{
    collections::BTreeMap,
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex},
    thread::{self, JoinHandle},
};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    pub version: Version,
    pub listen: SocketAddr,
    pub limits: ProbeLimits,
}
pub struct Server {
    pub listener: PeerListener,
    pub budget: Arc<ProbeBudget>,
    pub samples: Arc<Mutex<BTreeMap<NodeId, ProbeObservation>>>,
}
impl Server {
    pub fn bind(config: ServerConfig, approved: &Digest) -> Result<Self> {
        approved_record(&config, approved)?;
        if !starlings_security::network::local_address(config.listen, true) {
            return Err("PROBE_ADDRESS_REFUSED".into());
        }
        Ok(Self {
            listener: PeerListener::bind(config.listen, ListenerBudget::default())
                .map_err(security)?,
            budget: Arc::new(ProbeBudget::new(config.limits).map_err(security)?),
            samples: Arc::new(Mutex::new(BTreeMap::new())),
        })
    }
    pub fn view(&self, authority: &Authority, node: NodeId) -> Result<serde_json::Value> {
        let samples = self.samples.lock().map_err(|_| "PROBE_STATE_UNAVAILABLE")?;
        let view = samples
            .get(&node)
            .and_then(|s| s.view_authority(authority, node));
        let current = view.is_some();
        let view = view.filter(|v| v.report.token_profile.is_none());
        let inputs = view
            .as_ref()
            .map(|v| v.placement_inputs())
            .transpose()
            .map_err(security)?;
        Ok(profile_view(
            serde_json::json!({"state":if view.is_some() {"measured"} else {"unavailable"},
            "listen":self.listener.local_addr().map_err(security)?, "measurement":view, "placement_inputs":inputs}),
            self.budget.limits(),
            samples.get(&node),
            current,
        ))
    }
}
pub struct Worker {
    route: ProbeRoute,
    budget: Arc<ProbeBudget>,
    result: Arc<Mutex<(Option<ProbeObservation>, Option<starlings_security::Error>)>>,
    job: Option<JoinHandle<()>>,
}
impl Worker {
    pub fn new(route: ProbeRoute, approved: &Digest) -> Result<Self> {
        approved_record(&route, approved)?;
        route.validate().map_err(security)?;
        let budget = Arc::new(ProbeBudget::new(route.limits.clone()).map_err(security)?);
        Ok(Self {
            route,
            budget,
            result: Arc::new(Mutex::new((None, None))),
            job: None,
        })
    }
    pub fn start(
        &mut self,
        route: &ProbeRoute,
        approved: &Digest,
        state: PathBuf,
        fence: RegistryFence,
    ) -> Result<()> {
        approved_record(route, approved)?;
        if starlings_protocol::content_hash(route).map_err(|_| "PROBE_INVALID")?
            != starlings_protocol::content_hash(&self.route).map_err(|_| "PROBE_INVALID")?
        {
            return Err("PROBE_ROUTE_IMMUTABLE".into());
        }
        if self.job.as_ref().is_some_and(|j| !j.is_finished()) {
            return Err("PROBE_BUSY".into());
        }
        self.join()?;
        *self.result.lock().map_err(|_| "PROBE_STATE_UNAVAILABLE")? = (None, None);
        let route = self.route.clone();
        let budget = self.budget.clone();
        let output = self.result.clone();
        self.job = Some(
            thread::Builder::new()
                .name("directed-probe".into())
                .spawn(move || {
                    let result = (|| {
                        let files = PrivateDir::open(state)?;
                        let identity = Identity::load(&files, "identity.json")?;
                        let invitation = Invitation::load(&files, "bootstrap.json")?;
                        starlings_security::probes::measure(
                            &route,
                            &identity,
                            &invitation,
                            fence,
                            &budget,
                        )
                    })();
                    if let Ok(mut output) = output.lock() {
                        match result {
                            Ok(sample) => output.0 = Some(sample),
                            Err(error) => output.1 = Some(error),
                        }
                    }
                })
                .map_err(|_| "THREAD_UNAVAILABLE")?,
        );
        Ok(())
    }
    pub fn view(&self, fence: &RegistryFence) -> Result<serde_json::Value> {
        let result = self.result.lock().map_err(|_| "PROBE_STATE_UNAVAILABLE")?;
        let current = result.0.as_ref().and_then(|s| s.view(fence));
        let is_current = current.is_some();
        let current = current.filter(|v| v.report.token_profile.is_none());
        let inputs = current
            .as_ref()
            .map(|v| v.placement_inputs())
            .transpose()
            .map_err(security)?;
        let mut value = profile_view(
            serde_json::json!({"state":if current.is_some() {"measured"} else if self.job.as_ref().is_some_and(|j| !j.is_finished()) {"pending"} else if result.1.is_some() {"failed"} else {"unavailable"},
            "measurement":current, "placement_inputs":inputs, "error":result.1.map(security)}),
            &self.route.limits,
            result.0.as_ref(),
            is_current,
        );
        if self.route.limits.token_profile.is_some() {
            value["setup_error"] = serde_json::json!(result.1);
        }
        Ok(value)
    }
    pub fn join(&mut self) -> Result<()> {
        if let Some(job) = self.job.take() {
            job.join().map_err(|_| "PROBE_JOB_FAILED")?;
        }
        Ok(())
    }
}
fn profile_view(
    mut value: serde_json::Value,
    limits: &ProbeLimits,
    observation: Option<&ProbeObservation>,
    current: bool,
) -> serde_json::Value {
    if limits.token_profile.is_some() {
        value["profile_limits"] = serde_json::json!(limits);
        value["profile_attempt"] =
            serde_json::json!(observation.and_then(ProbeObservation::token_profile_snapshot));
        if let Some(report) = observation.and_then(|o| o.report.token_profile.as_ref()) {
            if let Some(error) = &report.error {
                value["state"] = "failed".into();
                value["error"] = format!("{error:?}").into();
            } else {
                value["state"] = if current {
                    "profile_complete"
                } else {
                    "unavailable"
                }
                .into();
            }
        }
    }
    value
}
impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.join();
    }
}
fn approved_record(value: &impl Serialize, approved: &Digest) -> Result<()> {
    if &starlings_protocol::content_hash(value).map_err(|_| "PROBE_INVALID")? != approved {
        return Err("PROBE_CONFIG_UNAPPROVED".into());
    }
    Ok(())
}

/// One existing read-only admin call and one new private JSONL file; never starts a profile.
pub fn record(paths: &crate::cluster::Paths, node: NodeId, output: &std::path::Path) -> Result<()> {
    use std::io::Write;
    let (directory, name) = crate::cluster::private_parent(output)?;
    let reply: starlings_cli::AdminReply =
        starlings_security::admin_call(&paths.state, &crate::cluster::Command::Links { node })
            .map_err(security)?;
    let snapshot = starlings_cli::link_record::project(&reply).map_err(str::to_owned)?;
    let collector = starlings_adapters::service::executable_digest(
        &std::env::current_exe().map_err(|_| "PROFILE_COLLECTOR_UNAVAILABLE")?,
    )
    .map_err(|_| "PROFILE_COLLECTOR_UNAVAILABLE")?;
    let captured_at = time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .map_err(|_| "PROFILE_CLOCK_UNAVAILABLE")?;
    let record = serde_json::json!({"kind":"DiagnosticTokenLinkProfileSnapshot","version":1,"qualification":"not_assessed","requested_node":node,"collector_sha256":collector,"captured_at":captured_at,"snapshot":snapshot});
    let mut file = directory
        .create(name)
        .map_err(|_| "PROFILE_NEW_OUTPUT_REQUIRED")?;
    serde_json::to_writer(&mut file, &record).map_err(|_| "PROFILE_RECORD_WRITE_FAILED")?;
    file.write_all(b"\n")
        .map_err(|_| "PROFILE_RECORD_WRITE_FAILED")?;
    file.sync_all().map_err(|_| "PROFILE_RECORD_WRITE_FAILED")?;
    writeln!(
        std::io::stdout().lock(),
        "PROFILE_SNAPSHOT_SAVED: qualification=not_assessed"
    )
    .map_err(|_| "OUTPUT_UNAVAILABLE: snapshot was saved".into())
}
