//! Read-only host telemetry. Discovery never grants an inference capability.
mod nvidia;
pub use nvidia::{NvidiaDevice, NvidiaMemory, NvidiaReport};
use serde::Serialize;
use starlings_protocol::{
    BoundedU32, ClusterId, Decimal, MemoryPool, MemoryPoolKind, NodeCapabilities,
    NodeCapabilitiesArchitecture, NodeCapabilitiesOs, NodeCapabilitiesState, NodeId, Version,
};
use std::{
    collections::BTreeMap,
    path::Path,
    time::{Duration, Instant, SystemTime},
};
use sysinfo::{Components, DiskRefreshKind, Disks, MemoryRefreshKind, System};

pub const TELEMETRY_TTL: Duration = Duration::from_secs(5);

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "quality", content = "value", rename_all = "snake_case")]
pub enum Observation<T> {
    Measured(T),
    Estimated(T),
    Unavailable,
}
#[derive(Clone, Debug, Serialize)]
pub struct HostReport {
    pub os: &'static str,
    pub architecture: &'static str,
    pub age_ms: u64,
    pub ttl_ms: u64,
    pub stale: bool,
    pub memory_pools: Vec<MemoryPool>,
    pub available_memory_quality: &'static str,
    pub available_cpu_threads: Observation<usize>,
    pub cache_available_bytes: Observation<Decimal>,
    pub max_sensor_temperature_c: Observation<f32>,
    pub thermal_throttled: Observation<bool>,
    pub nvidia: Observation<NvidiaReport>,
    pub inference: &'static str,
}
/// One capture identity and retained timestamps; publication cannot create a fresh capture.
pub struct HostSnapshot {
    snapshot_id: uuid::Uuid,
    at: Instant,
    wall: SystemTime,
    report: HostReport,
}
impl HostSnapshot {
    /// Explicit trusted local library only; never called by registry/heartbeat handling.
    pub fn collect_nvidia(
        cache: &Path,
        library: &Path,
        approved: &starlings_protocol::Digest,
    ) -> Result<Self, &'static str> {
        let mut snapshot = Self::collect(cache);
        let report = nvidia::collect(library, approved)?;
        snapshot.add_nvidia(report)?;
        Ok(snapshot)
    }
    fn add_nvidia(&mut self, report: NvidiaReport) -> Result<(), &'static str> {
        for device in &report.devices {
            if let Some(pool) = device.pool() {
                if self
                    .report
                    .memory_pools
                    .iter()
                    .any(|p| p.pool_id == pool.pool_id)
                {
                    return Err("DUPLICATE_DEVICE_POOL");
                }
                self.report.memory_pools.push(pool);
            }
        }
        self.report.nvidia = Observation::Measured(report);
        Ok(())
    }
    pub fn collect(cache: &Path) -> Self {
        let mut snapshot = Self::collect_contribution();
        let disks =
            Disks::new_with_refreshed_list_specifics(DiskRefreshKind::nothing().with_storage());
        snapshot.report.cache_available_bytes = cache
            .canonicalize()
            .ok()
            .and_then(|cache| {
                disks
                    .iter()
                    .filter(|disk| cache.starts_with(disk.mount_point()))
                    .max_by_key(|disk| disk.mount_point().components().count())
                    .map(|disk| Observation::Measured(Decimal(disk.available_space())))
            })
            .unwrap_or(Observation::Unavailable);
        snapshot
    }
    /// Local policy sampler only: no disk, process enumeration, NVML or backend execution.
    pub fn collect_contribution() -> Self {
        let mut snapshot = Self::collect_memory();
        snapshot.report.max_sensor_temperature_c = Components::new_with_refreshed_list()
            .iter()
            .filter_map(|sensor| sensor.temperature())
            .filter(|value| value.is_finite() && (-100.0..=200.0).contains(value))
            .reduce(f32::max)
            .map(Observation::Measured)
            .unwrap_or(Observation::Unavailable);
        snapshot
    }
    /// Heartbeat path: host RAM/cgroup APIs only, never disks, sensors or backend execution.
    pub fn collect_memory() -> Self {
        let at = Instant::now();
        let wall = SystemTime::now();
        let mut system = System::new();
        system.refresh_memory_specifics(MemoryRefreshKind::nothing().with_ram());
        let mut total = system.total_memory();
        let mut available = system.available_memory().min(total);
        if let Some(cgroup) = system.cgroup_limits() {
            total = total.min(cgroup.total_memory);
            available = available.min(cgroup.free_memory).min(total);
        }
        let memory_pools = host_pool(
            std::env::consts::OS,
            std::env::consts::ARCH,
            total,
            available,
        );
        Self {
            snapshot_id: uuid::Uuid::new_v4(),
            at,
            wall,
            report: HostReport {
                os: std::env::consts::OS,
                architecture: std::env::consts::ARCH,
                age_ms: 0,
                ttl_ms: TELEMETRY_TTL.as_millis() as u64,
                stale: false,
                available_memory_quality: if memory_pools.is_empty() {
                    "unavailable"
                } else {
                    "estimated"
                },
                memory_pools,
                available_cpu_threads: std::thread::available_parallelism()
                    .ok()
                    .map(|n| Observation::Measured(n.get()))
                    .unwrap_or(Observation::Unavailable),
                cache_available_bytes: Observation::Unavailable,
                max_sensor_temperature_c: Observation::Unavailable,
                // A temperature reading does not establish the device's throttle state.
                thermal_throttled: Observation::Unavailable,
                nvidia: Observation::Unavailable,
                inference: "unavailable",
            },
        }
    }
    pub fn report(&self) -> HostReport {
        self.at_time(Instant::now(), SystemTime::now())
    }
    /// Stable CPU/host qualification inputs, excluding diagnostic GPUs, free memory and time.
    /// Node enrollment binds the device; this detects OS/host-budget/CPU-affinity changes.
    pub fn qualification_target(&self) -> Option<starlings_protocol::Digest> {
        let report = self.report();
        if report.stale || !report.memory_pools.iter().any(|p| p.pool_id == "host") {
            return None;
        }
        starlings_protocol::content_hash(&serde_json::json!({
            "os":report.os, "architecture":report.architecture,
            "os_version":System::os_version(), "kernel_version":System::kernel_version(),
            "memory_pools":report.memory_pools.iter().filter(|p| p.pool_id == "host").map(|p| (&p.pool_id, p.total_bytes)).collect::<Vec<_>>(),
            "available_cpu_threads":report.available_cpu_threads,
        })).ok()
    }
    /// Diagnostic announcement only: no installed/qualified backend or serving endpoint is inferred.
    pub fn capabilities(&self, node: NodeId, cluster: ClusterId) -> Option<NodeCapabilities> {
        let report = self.report();
        if report.stale || report.memory_pools.is_empty() {
            return None;
        }
        Some(NodeCapabilities {
            kind: "NodeCapabilities".into(),
            protocol_version: Version,
            fixture: false,
            node_id: node,
            cluster_id: cluster,
            snapshot_id: self.snapshot_id.to_string().parse().ok()?,
            observed_at: time::OffsetDateTime::from(self.wall)
                .format(&time::format_description::well_known::Rfc3339)
                .ok()?,
            sample_age_ms: BoundedU32::new(report.age_ms.try_into().ok()?).ok()?,
            state: NodeCapabilitiesState::Unavailable,
            os: match report.os {
                "linux" => NodeCapabilitiesOs::Linux,
                "macos" => NodeCapabilitiesOs::Macos,
                "windows" => NodeCapabilitiesOs::Windows,
                "ios" => NodeCapabilitiesOs::Ios,
                "android" => NodeCapabilitiesOs::Android,
                _ => return None,
            },
            architecture: match report.architecture {
                "aarch64" => NodeCapabilitiesArchitecture::Aarch64,
                "x86_64" => NodeCapabilitiesArchitecture::X8664,
                _ => return None,
            },
            memory_pools: report.memory_pools,
            backends: Vec::new(),
            control_endpoint: "unavailable".into(),
            tensor_endpoint: "unavailable".into(),
        })
    }
    fn at_time(&self, now: Instant, wall: SystemTime) -> HostReport {
        let elapsed = now
            .checked_duration_since(self.at)
            .zip(wall.duration_since(self.wall).ok())
            .map(|(a, b)| a.max(b));
        let mut report = self.report.clone();
        report.age_ms = elapsed
            .map(|d| d.as_millis().min(u64::MAX as u128) as u64)
            .unwrap_or(u64::MAX);
        report.stale = elapsed.is_none_or(|d| d >= TELEMETRY_TTL);
        if report.stale {
            // Stale diagnostics may be displayed, but they provide no allocatable memory.
            for pool in &mut report.memory_pools {
                pool.available_bytes = Decimal(0);
            }
            report.available_memory_quality = "stale";
        }
        report
    }
    /// Reservations are keyed by physical pool, never separately by CPU and unified GPU.
    pub fn reservable(
        &self,
        reservations: &BTreeMap<String, u64>,
    ) -> Result<Vec<MemoryPool>, &'static str> {
        let mut report = self.report();
        if report.stale || report.memory_pools.is_empty() {
            return Err("STALE_OR_UNAVAILABLE_RESOURCE");
        }
        for (id, amount) in reservations {
            let pool = report
                .memory_pools
                .iter_mut()
                .find(|p| &p.pool_id == id)
                .ok_or("UNKNOWN_MEMORY_POOL")?;
            pool.available_bytes.0 = pool
                .available_bytes
                .0
                .checked_sub(*amount)
                .ok_or("CAPACITY_UNAVAILABLE")?;
            pool.reserved_bytes = Decimal(*amount);
        }
        Ok(report.memory_pools)
    }
}
fn host_pool(os: &str, arch: &str, total: u64, available: u64) -> Vec<MemoryPool> {
    if total == 0 || available > total {
        return Vec::new();
    }
    let unified = os == "macos" && arch == "aarch64";
    vec![MemoryPool {
        pool_id: "host".into(),
        kind: if unified {
            MemoryPoolKind::Unified
        } else {
            MemoryPoolKind::System
        },
        total_bytes: Decimal(total),
        available_bytes: Decimal(available),
        reserved_bytes: Decimal(0),
        device_ids: if unified {
            vec!["cpu".into(), "apple-unified".into()]
        } else {
            vec!["cpu".into()]
        },
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn discrete_pools_are_reserved_separately_and_expire_together() {
        let id = "GPU-00000000-0000-4000-8000-000000000003";
        let device = NvidiaDevice {
            device_id: id.into(),
            name: Observation::Unavailable,
            memory: Observation::Estimated(NvidiaMemory {
                total_bytes: Decimal(1000),
                available_bytes: Decimal(700),
            }),
            unshared_device_memory: Observation::Measured(true),
            temperature_c: Observation::Unavailable,
            thermal_throttled: Observation::Unavailable,
        };
        let gpu = NvidiaReport {
            library: starlings_protocol::byte_digest(b"synthetic library, not native evidence"),
            driver: Observation::Unavailable,
            devices: vec![device],
        };
        let mut snapshot = HostSnapshot::collect_memory();
        snapshot.report.memory_pools = host_pool("linux", "x86_64", 2000, 1200);
        let target = snapshot.qualification_target();
        snapshot.add_nvidia(gpu.clone()).unwrap();
        let pool_id = format!("nvidia:{id}");
        let pools = snapshot
            .reservable(&BTreeMap::from([("host".into(), 800), (pool_id, 300)]))
            .unwrap();
        assert_eq!(pools[0].available_bytes.0, 400);
        assert_eq!(pools[1].available_bytes.0, 400);
        assert!(snapshot.add_nvidia(gpu).is_err());
        assert_eq!(target, snapshot.qualification_target());
        let node = "00000000-0000-4000-8000-000000000001".parse().unwrap();
        let cluster = "00000000-0000-4000-8000-000000000002".parse().unwrap();
        let cap = snapshot.capabilities(node, cluster).unwrap();
        starlings_protocol::encode(&cap, starlings_protocol::FixtureMode::Reject).unwrap();
        assert!(cap.backends.is_empty());
        let old = snapshot.at_time(snapshot.at + TELEMETRY_TTL, snapshot.wall + TELEMETRY_TTL);
        assert!(old.memory_pools.iter().all(|p| p.available_bytes.0 == 0));
        snapshot.report.memory_pools.retain(|p| p.pool_id != "host");
        assert!(snapshot.qualification_target().is_none());
    }
    #[test]
    fn published_memory_is_diagnostic_and_stale_samples_cannot_be_announced() {
        let node = "00000000-0000-4000-8000-000000000001".parse().unwrap();
        let cluster = "00000000-0000-4000-8000-000000000002".parse().unwrap();
        let mut snapshot = HostSnapshot::collect_memory();
        // Synthetic RAM exercises production encoding, not a native allocation claim.
        snapshot.report.memory_pools = host_pool("linux", "aarch64", 1000, 700);
        let capability = snapshot.capabilities(node, cluster).unwrap();
        starlings_protocol::encode(&capability, starlings_protocol::FixtureMode::Reject).unwrap();
        assert!(capability.backends.is_empty());
        assert_eq!(capability.state, NodeCapabilitiesState::Unavailable);
        assert_eq!(capability.control_endpoint, "unavailable");
        assert_eq!(capability.tensor_endpoint, "unavailable");
        assert_eq!(capability.memory_pools[0].available_bytes.0, 700);
        assert_eq!(
            capability.snapshot_id,
            snapshot.capabilities(node, cluster).unwrap().snapshot_id
        );
        snapshot.at -= Duration::from_millis(100);
        let repeated = snapshot.capabilities(node, cluster).unwrap();
        assert_eq!(capability.snapshot_id, repeated.snapshot_id);
        assert_eq!(capability.observed_at, repeated.observed_at);
        assert!(repeated.sample_age_ms.get() >= 100);
        assert_ne!(
            HostSnapshot::collect_memory().snapshot_id,
            snapshot.snapshot_id
        );
        snapshot.wall -= TELEMETRY_TTL;
        assert!(snapshot.capabilities(node, cluster).is_none());
    }
    #[test]
    fn unified_reservations_and_stale_samples_cannot_inflate_capacity() {
        let mut snapshot = HostSnapshot::collect(Path::new("."));
        snapshot.report.memory_pools = host_pool("macos", "aarch64", 1000, 800);
        assert_eq!(snapshot.report.memory_pools.len(), 1);
        assert_eq!(snapshot.report.memory_pools[0].device_ids.len(), 2);
        let mut reservations = BTreeMap::from([("host".into(), 600)]);
        assert_eq!(
            snapshot.reservable(&reservations).unwrap()[0]
                .available_bytes
                .0,
            200
        );
        reservations.insert("host".into(), 801);
        assert!(snapshot.reservable(&reservations).is_err());
        let stale = snapshot.at_time(snapshot.at + TELEMETRY_TTL, snapshot.wall + TELEMETRY_TTL);
        assert!(stale.stale);
        assert_eq!(stale.memory_pools[0].available_bytes.0, 0);
        assert!(
            snapshot
                .at_time(snapshot.at, snapshot.wall - Duration::from_secs(1))
                .stale
        );
        // Wall time catches suspend when the platform monotonic clock pauses.
        assert!(
            snapshot
                .at_time(snapshot.at, snapshot.wall + TELEMETRY_TTL)
                .stale
        );
        assert_eq!(snapshot.report.inference, "unavailable");
        let linux = host_pool("linux", "aarch64", 1000, 800);
        assert_eq!(linux[0].kind, MemoryPoolKind::System);
        assert_eq!(linux[0].device_ids, ["cpu"]);
        assert!(host_pool("linux", "aarch64", 0, 0).is_empty());
    }
}
