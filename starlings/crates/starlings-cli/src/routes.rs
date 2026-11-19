//! Explicit local qualification and fixed routes. Evidence is reviewed local policy,
//! not an attestation supplied by a worker or inferred from service readiness.
use serde::{Deserialize, Serialize};
use starlings_adapters::service::{ServiceApi, ServiceConfig};
use starlings_core::routing::WarmReplica;
use starlings_protocol::{Digest, FixtureMode, PlacementPlan, Version};
use starlings_security::Error;
use std::net::SocketAddr;
type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qualification {
    pub version: Version,
    pub fixture: bool,
    pub os: String,
    pub architecture: String,
    pub device_kind: String,
    pub target: Digest,
    pub loading_bytes: starlings_protocol::Decimal,
    pub model_layer_count: u32,
    pub boundary_profile: String,
    pub request_drain_profile: String,
    pub configuration: Digest,
    pub replica: WarmReplica,
    pub max_output_tokens: u32,
    pub memory_qualified: bool,
    pub exact_count_qualified: bool,
    pub request_drain_qualified: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_threads: Option<u32>,
}
impl Qualification {
    pub fn validate(&self) -> Result<()> {
        self.replica.validate().map_err(|_| Error::Invalid)?;
        if (self.fixture && !cfg!(feature = "test-backend"))
            || !matches!(self.os.as_str(), "macos" | "linux" | "windows")
            || !matches!(self.architecture.as_str(), "aarch64" | "x86_64")
            || self.replica.pool != "host"
            || self.device_kind != "cpu"
            || self.replica.backend.len() > 128
            || !self.replica.profile.bytes().all(|b| b.is_ascii_graphic())
            || self.replica.max_sessions != 1
            || !self.replica.accepting
            || !self.memory_qualified
            || !self.exact_count_qualified
            || !self.request_drain_qualified
            || self.max_output_tokens == 0
            || self.max_output_tokens >= self.replica.context_limit
            || self.replica.measured_request_ms > 60_000
            || self.loading_bytes.0 < self.replica.request_bytes
            || !(1..=4096).contains(&self.model_layer_count)
            || self.boundary_profile.is_empty()
            || self.boundary_profile.len() > 128
            || !self.boundary_profile.bytes().all(|b| b.is_ascii_graphic())
            || self.request_drain_profile != "slots-idle-v1"
            || self.configuration.as_str().bytes().all(|b| b == b'0')
            || self.target.as_str().bytes().all(|b| b == b'0')
            || self.cpu_threads.is_some_and(|n| !(1..=4096).contains(&n))
        {
            return Err(Error::Denied);
        }
        Ok(())
    }
    pub fn admit(&self, output: u32) -> Result<()> {
        if output > self.max_output_tokens {
            return Err(Error::Invalid);
        }
        Ok(())
    }
    pub fn check_plan(&self, plan: &PlacementPlan) -> Result<()> {
        let q = &self.replica;
        let [stage] = plan.stages.as_slice() else {
            return Err(Error::Denied);
        };
        if plan.mode != starlings_protocol::PlacementPlanMode::WholeRequest
            || plan.manifest_digest != q.manifest
            || plan.execution_profile != q.profile
            || plan.model_layer_count.get() != self.model_layer_count
            || plan.boundary_profile != self.boundary_profile
            || plan.max_sessions.get() != 1
            || plan.context_limit.get() != q.context_limit
            || stage.node_id != q.node
            || stage.device_id != "cpu"
            || stage.backend_id != q.backend
            || stage.backend_build_digest != q.build
            || stage.memory_pool_id != q.pool
            || stage.reserved_bytes()? != q.request_bytes
        {
            return Err(Error::Denied);
        }
        Ok(())
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerRoute {
    pub version: Version,
    pub listen: SocketAddr,
    pub service: ServiceConfig,
    pub qualification: Qualification,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cleanup: Option<CleanupQualification>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contribution: Option<crate::contribution::CpuContribution>,
}
/// Reviewed local policy for a CPU-only backend that cannot leave allocations
/// in another device or an escaped process. Group absence alone supplies neither assertion.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CleanupQualification {
    pub version: Version,
    pub configuration: Digest,
    pub target: Digest,
    pub evidence: Digest,
    pub scope: String,
    pub no_device_work_reviewed: bool,
    pub no_escaped_children_reviewed: bool,
}
impl CleanupQualification {
    pub fn validate(&self, q: &Qualification) -> Result<()> {
        if self.configuration != q.configuration
            || self.target != q.target
            || self.scope != "posix-cpu-group-v1"
            || q.device_kind != "cpu"
            || !matches!(q.os.as_str(), "linux" | "macos")
            || !self.no_device_work_reviewed
            || !self.no_escaped_children_reviewed
            || self.evidence.as_str().bytes().all(|b| b == b'0')
        {
            return Err(Error::Denied);
        }
        Ok(())
    }
}
impl WorkerRoute {
    pub fn validate(&self, approved: &Digest) -> Result<()> {
        approved_record(self, approved)?;
        self.qualification.validate()?;
        if let Some(policy) = &self.contribution {
            policy.validate(&self.qualification)?;
        }
        if let Some(cleanup) = &self.cleanup {
            cleanup.validate(&self.qualification)?;
        }
        let q = &self.qualification;
        let s = &self.service;
        if s.api != ServiceApi::InputTokensV1
            || !starlings_security::network::local_address(self.listen, true)
            || q.os != std::env::consts::OS
            || q.architecture != std::env::consts::ARCH
            || starlings_adapters::resources::HostSnapshot::collect_memory()
                .qualification_target()
                .as_ref()
                != Some(&q.target)
            || s.digest().map_err(|_| Error::Invalid)? != q.configuration
            || s.manifest_digest != q.replica.manifest
            || s.execution_profile != q.replica.profile
            || s.executable_sha256 != q.replica.build
            || s.request_timeout_ms > 60_000
            || (s.backend_version == "synthetic-v1") != q.fixture
            || s.properties
                .as_ref()
                .is_some_and(|p| p.context_tokens < q.replica.context_limit)
        {
            return Err(Error::Denied);
        }
        s.validate(&q.configuration).map_err(|_| Error::Denied)
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoordinatorRoute {
    pub version: Version,
    pub listen: SocketAddr,
    pub worker_endpoint: SocketAddr,
    pub plan: PlacementPlan,
    pub qualification: Qualification,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub replicas: Vec<CoordinatorReplica>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoordinatorReplica {
    pub worker_endpoint: SocketAddr,
    pub plan: PlacementPlan,
    pub qualification: Qualification,
}
impl CoordinatorReplica {
    pub fn validate(&self) -> Result<()> {
        self.qualification.validate()?;
        starlings_protocol::encode(&self.plan, FixtureMode::Reject)?;
        if !starlings_security::network::local_address(self.worker_endpoint, false) {
            return Err(Error::Denied);
        }
        self.qualification.check_plan(&self.plan)
    }
}
impl CoordinatorRoute {
    pub fn workers(&self) -> Vec<CoordinatorReplica> {
        let mut workers = vec![CoordinatorReplica {
            worker_endpoint: self.worker_endpoint,
            plan: self.plan.clone(),
            qualification: self.qualification.clone(),
        }];
        workers.extend(self.replicas.iter().cloned());
        workers
    }
    pub fn validate(&self, approved: &Digest) -> Result<()> {
        approved_record(self, approved)?;
        if !self.listen.ip().is_loopback()
            || self.replicas.len() >= 64
            || serde_json::to_vec(self)?.len() > 16 * 1024
        {
            return Err(Error::Denied);
        }
        let mut nodes = std::collections::BTreeSet::new();
        let mut endpoints = std::collections::BTreeSet::new();
        for worker in self.workers() {
            worker.validate()?;
            let q = &worker.qualification;
            if !nodes.insert(q.replica.node)
                || !endpoints.insert(worker.worker_endpoint)
                || worker.plan.cluster_id != self.plan.cluster_id
                || q.fixture != self.qualification.fixture
                || q.replica.manifest != self.qualification.replica.manifest
                || q.replica.profile != self.qualification.replica.profile
                || q.replica.context_limit != self.qualification.replica.context_limit
                || q.max_output_tokens != self.qualification.max_output_tokens
            {
                return Err(Error::Denied);
            }
        }
        Ok(())
    }
}
/// One warm `glyd-engine rangeworker` stage owned by a worker daemon (GLY-188). The engine
/// bytes, model path, layer range, launch arguments and its `describe` are all approved here.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageRoute {
    pub version: Version,
    /// The coordinator's per-request generation connection.
    pub listen: SocketAddr,
    /// The adjacent stage's boundaries (or the last stage's tokens) arrive here.
    pub data_listen: SocketAddr,
    pub executable: std::path::PathBuf,
    pub executable_sha256: Digest,
    pub model: std::path::PathBuf,
    pub layers: [u32; 2],
    pub context_tokens: u32,
    pub chunk_tokens: u32,
    pub max_prompt_tokens: u32,
    pub max_output_tokens: u32,
    /// `descriptor_digest` of the engine's `describe` for exactly these arguments.
    pub descriptor: Digest,
    pub execution_profile: String,
    pub boundary_profile: String,
    pub backend_id: String,
    /// Device and pool name the plan uses for this stage, e.g. `cuda:0`.
    pub device_id: String,
    pub engine_bytes_limit: starlings_protocol::Decimal,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub environment: std::collections::BTreeMap<String, String>,
    /// The approved bytes of the library `GLYD_GPU_LIB` names; required with it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_library_sha256: Option<Digest>,
    /// GLY-360: the engine's `--head-rows` on the last stage, room for a drafted step's
    /// per-row tokens (a pipeline's `draft_tokens` + 1 at most). 0: not passed, as before.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub head_rows: u32,
    /// GLY-190: requests the stage serves at once, each in its own engine session (the
    /// engine's `--sessions`, 1 or 2). Each slot after the first has its own data listener.
    #[serde(default = "one", skip_serializing_if = "is_one")]
    pub sessions: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub more_data_listen: Vec<SocketAddr>,
}
fn one() -> u32 {
    1
}
fn is_one(n: &u32) -> bool {
    *n == 1
}
impl StageRoute {
    pub fn validate(&self, approved: &Digest) -> Result<()> {
        approved_record(self, approved)?;
        let [start, end] = self.layers;
        let label =
            |s: &str| !s.is_empty() && s.len() <= 128 && s.bytes().all(|b| b.is_ascii_graphic());
        if !starlings_security::network::local_address(self.listen, false)
            || !starlings_security::network::local_address(self.data_listen, false)
            || self.listen == self.data_listen
            || !self.executable.is_absolute()
            || !self.model.is_absolute()
            || start >= end
            || self.chunk_tokens == 0
            || self.chunk_tokens > self.max_prompt_tokens
            || self.max_output_tokens == 0
            || self
                .max_prompt_tokens
                .checked_add(self.max_output_tokens)
                .is_none_or(|n| n > self.context_tokens)
            || self.engine_bytes_limit.0 == 0
            || self.head_rows > starlings_protocol::MAX_DECODE_ROWS.min(self.chunk_tokens)
            || !(1..=2).contains(&self.sessions)
            || self.more_data_listen.len() + 1 != self.sessions as usize
            || self.more_data_listen.iter().any(|&a| {
                !starlings_security::network::local_address(a, false)
                    || a == self.listen
                    || a == self.data_listen
            })
            || ![
                &self.execution_profile,
                &self.boundary_profile,
                &self.backend_id,
                &self.device_id,
            ]
            .into_iter()
            .all(|s| label(s))
            || self.environment.len() > 16
            || self.environment.keys().any(|k| {
                k.is_empty()
                    || k.starts_with("LD_")
                    || k.starts_with("DYLD_")
                    || !k
                        .bytes()
                        .all(|b| b.is_ascii_uppercase() || b == b'_' || b.is_ascii_digit())
            })
            || self.environment.contains_key("GLYD_GPU_LIB") != self.gpu_library_sha256.is_some()
        {
            return Err(Error::Denied);
        }
        Ok(())
    }
    /// The approved launch; nothing loads before `prepare`.
    pub fn arguments(&self) -> Vec<String> {
        let mut arguments = vec![
            "rangeworker".into(),
            "--model".into(),
            self.model.to_string_lossy().into_owned(),
            "--layers".into(),
            format!("{}..{}", self.layers[0], self.layers[1]),
            "--ctx".into(),
            self.context_tokens.to_string(),
            "--chunk".into(),
            self.chunk_tokens.to_string(),
        ];
        // Room for a drafted step's per-row tokens on the last stage (GLY-360).
        if self.head_rows > 0 {
            arguments.extend(["--head-rows".into(), self.head_rows.to_string()]);
        }
        // One engine session for each request served at once (GLY-190).
        if self.sessions > 1 {
            arguments.extend(["--sessions".into(), self.sessions.to_string()]);
        }
        arguments
    }
}

/// Coordinator route for one approved two-stage layer pipeline (GLY-188). The plan is a
/// template: its epoch, ID, times and digest are replaced for each request.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineRoute {
    pub version: Version,
    pub listen: SocketAddr,
    pub plan: PlacementPlan,
    /// Each stage's generation and data listeners, in plan order.
    pub stages: Vec<PipelineStage>,
    pub chunk_tokens: u32,
    pub max_output_tokens: u32,
    /// The lease each stage holds for one request (at most 60 s, not renewable).
    pub lifetime_ms: u32,
    /// Sampled IDs that end a generation, e.g. the model's EOS tokens.
    pub stop_tokens: Vec<u32>,
    /// GLY-360: guessed tokens after each greedy decode step's own (0: none). Absent in
    /// records approved before it, so their digests still hold.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub draft_tokens: u32,
}
fn is_zero(n: &u32) -> bool {
    *n == 0
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineStage {
    pub endpoint: SocketAddr,
    pub data: SocketAddr,
    /// GLY-190: the stage's data listeners for its further session slots, in slot order;
    /// every stage has as many.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub more_data: Vec<SocketAddr>,
}
impl PipelineRoute {
    pub fn validate(&self, approved: &Digest) -> Result<()> {
        approved_record(self, approved)?;
        starlings_protocol::encode(&self.plan, FixtureMode::Reject)?;
        // ponytail: two stages (first and last); a middle stage needs boundary routes both ways.
        if !self.listen.ip().is_loopback()
            || self.plan.mode != starlings_protocol::PlacementPlanMode::LayerPipeline
            || self.plan.stages.len() != 2
            || self.stages.len() != 2
            || self.plan.max_sessions.get() != 1
            || self.stages.iter().any(|s| {
                !starlings_security::network::local_address(s.endpoint, false)
                    || !starlings_security::network::local_address(s.data, false)
                    || s.more_data.len() > 1
                    || s.more_data.len() != self.stages[0].more_data.len()
                    || s.more_data
                        .iter()
                        .any(|&a| !starlings_security::network::local_address(a, false))
            })
            || self.chunk_tokens == 0
            || self.max_output_tokens == 0
            || self.max_output_tokens >= self.plan.context_limit.get()
            || !(1000..=60_000).contains(&self.lifetime_ms)
            || self.stop_tokens.len() > 64
            || self.draft_tokens >= starlings_protocol::MAX_DECODE_ROWS
            || serde_json::to_vec(self)?.len() > 16 * 1024
        {
            return Err(Error::Denied);
        }
        Ok(())
    }
}

pub fn approved_record(record: &impl Serialize, approved: &Digest) -> Result<()> {
    if &starlings_protocol::content_hash(record)? == approved {
        Ok(())
    } else {
        Err(Error::Denied)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn qualification_is_explicit_strict_and_bounded() {
        let digest = starlings_protocol::byte_digest(b"synthetic unit fixture, not evidence");
        let mut q = Qualification {
            version: Version,
            fixture: true,
            os: "linux".into(),
            architecture: "aarch64".into(),
            device_kind: "cpu".into(),
            target: digest.clone(),
            loading_bytes: starlings_protocol::Decimal(2048),
            model_layer_count: 24,
            boundary_profile: "whole-request-fixture".into(),
            request_drain_profile: "slots-idle-v1".into(),
            configuration: digest.clone(),
            replica: WarmReplica {
                node: "00000000-0000-4000-8000-000000000001".parse().unwrap(),
                manifest: digest.clone(),
                profile: "fixture-profile".into(),
                backend: "fixture-backend".into(),
                build: digest.clone(),
                pool: "host".into(),
                evidence: digest,
                request_bytes: 1024,
                measured_request_ms: 1,
                context_limit: 128,
                max_sessions: 1,
                accepting: true,
            },
            max_output_tokens: 8,
            memory_qualified: true,
            exact_count_qualified: true,
            request_drain_qualified: true,
            cpu_threads: None,
        };
        assert_eq!(q.validate().is_ok(), cfg!(feature = "test-backend"));
        q.fixture = false;
        q.memory_qualified = false;
        assert!(q.validate().is_err());
        q.memory_qualified = true;
        q.device_kind = "cuda".into();
        assert!(q.validate().is_err());
        q.device_kind = "cpu".into();
        let hash = starlings_protocol::content_hash(&q).unwrap();
        let mut value = serde_json::to_value(&q).unwrap();
        assert!(
            value.get("cpu_threads").is_none(),
            "old absent declaration must not change its digest"
        );
        assert_eq!(value["replica"]["request_bytes"], "1024");
        value["extra"] = serde_json::json!(true);
        assert!(serde_json::from_value::<Qualification>(value).is_err());
        q.max_output_tokens = 128;
        assert!(q.validate().is_err());
        assert!(
            approved_record(&q, &hash).is_err(),
            "approval must cover the whole record"
        );
        q.max_output_tokens = 8;
        let mut cleanup = CleanupQualification {
            version: Version,
            configuration: q.configuration.clone(),
            target: q.target.clone(),
            evidence: starlings_protocol::byte_digest(b"synthetic cleanup fixture"),
            scope: "posix-cpu-group-v1".into(),
            no_device_work_reviewed: true,
            no_escaped_children_reviewed: false,
        };
        assert!(cleanup.validate(&q).is_err());
        let mut policy = crate::contribution::CpuContribution {
            version: Version,
            max_host_bytes: starlings_protocol::Decimal(4096),
            min_free_host_bytes: starlings_protocol::Decimal(0),
            max_cpu_threads: 2,
            max_temperature_c: None,
        };
        assert!(
            policy.validate(&q).is_err(),
            "missing qualified CPU thread count"
        );
        q.cpu_threads = Some(1);
        policy.validate(&q).unwrap();
        policy.max_host_bytes.0 = 3071;
        assert!(policy.validate(&q).is_err());
        policy.max_host_bytes.0 = 4096;
        policy.min_free_host_bytes.0 = u64::MAX;
        assert!(policy.validate(&q).is_err());
        policy.min_free_host_bytes.0 = 0;
        q.cpu_threads = Some(3);
        assert!(policy.validate(&q).is_err());
        q.cpu_threads = Some(1);
        let gate = crate::contribution::ContributionGate::new(policy.clone(), &q).unwrap();
        assert!(gate.check().is_err(), "no initial sample");
        gate.update(starlings_adapters::resources::HostSnapshot::collect_memory())
            .unwrap();
        gate.loading_allowed().unwrap();
        let another = crate::contribution::ContributionGate::new(policy, &q).unwrap();
        another
            .update(starlings_adapters::resources::HostSnapshot::collect_memory())
            .unwrap();
        gate.withdraw();
        assert!(gate.check().is_err());
        assert!(
            gate.update(starlings_adapters::resources::HostSnapshot::collect_memory())
                .is_err()
        );
        assert!(
            another.check().is_ok(),
            "a different owner's gate must remain eligible"
        );
        std::thread::sleep(starlings_adapters::resources::TELEMETRY_TTL);
        assert!(
            another.check().is_err(),
            "reads cannot refresh a five-second-old sample"
        );
        cleanup.no_escaped_children_reviewed = true;
        cleanup.validate(&q).unwrap();
        q.os = "windows".into();
        assert!(cleanup.validate(&q).is_err());
        q.os = "linux".into();
        cleanup.configuration = starlings_protocol::byte_digest(b"other config");
        assert!(cleanup.validate(&q).is_err());
    }
    #[test]
    fn pipeline_records_are_approved_whole_and_bounded() {
        let digest = starlings_protocol::byte_digest(b"synthetic stage record, not evidence");
        // Absolute on every platform; nothing is read from these paths.
        let root = std::env::temp_dir();
        let mut stage = StageRoute {
            version: Version,
            listen: "127.0.0.1:9800".parse().unwrap(),
            data_listen: "127.0.0.1:9801".parse().unwrap(),
            executable: root.join("glyd-engine"),
            executable_sha256: digest.clone(),
            model: root.join("qwen"),
            layers: [0, 16],
            context_tokens: 1024,
            chunk_tokens: 32,
            max_prompt_tokens: 768,
            max_output_tokens: 256,
            descriptor: digest.clone(),
            execution_profile: "glyd-range".into(),
            boundary_profile: "qwen35-residual-mlp-bf16".into(),
            backend_id: "glyd-rangeworker".into(),
            device_id: "cuda:0".into(),
            engine_bytes_limit: starlings_protocol::Decimal(1 << 30),
            environment: [("GLYD_GPU_LIB".into(), "/opt/lib.so".into())].into(),
            gpu_library_sha256: Some(digest.clone()),
            head_rows: 0,
            sessions: 1,
            more_data_listen: vec![],
        };
        stage
            .validate(&starlings_protocol::content_hash(&stage).unwrap())
            .unwrap();
        assert_eq!(stage.arguments()[4], "0..16");
        assert!(
            stage.validate(&digest).is_err(),
            "approval covers the record"
        );
        for change in 0..6 {
            let mut bad = stage.clone();
            match change {
                0 => bad.max_output_tokens = 257,
                1 => bad.listen = "8.8.8.8:9800".parse().unwrap(),
                2 => bad.model = "relative".into(),
                3 => bad.gpu_library_sha256 = None,
                4 => {
                    bad.environment
                        .insert("LD_PRELOAD".into(), "/tmp/x.so".into());
                }
                _ => {
                    bad.environment.insert("lower".into(), String::new());
                }
            }
            assert!(
                bad.validate(&starlings_protocol::content_hash(&bad).unwrap())
                    .is_err(),
                "{change}"
            );
        }
        stage.chunk_tokens = 0;
        assert!(
            stage
                .validate(&starlings_protocol::content_hash(&stage).unwrap())
                .is_err()
        );
    }
}
