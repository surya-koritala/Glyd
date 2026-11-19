//! Trusted local CPU contribution budgets; these are admission limits, not OS quotas.
use crate::routes::Qualification;
use serde::{Deserialize, Serialize};
use starlings_adapters::resources::{HostReport, HostSnapshot, Observation};
use starlings_protocol::{Decimal, Version};
use starlings_security::Error;
use std::sync::{
    Mutex,
    atomic::{AtomicBool, Ordering},
};

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CpuContribution {
    pub version: Version,
    pub max_host_bytes: Decimal,
    pub min_free_host_bytes: Decimal,
    pub max_cpu_threads: u32,
    pub max_temperature_c: Option<u32>,
}
impl CpuContribution {
    pub fn validate(&self, q: &Qualification) -> Result<(), Error> {
        if self.max_host_bytes.0 == 0
            || !(1..=4096).contains(&self.max_cpu_threads)
            || self
                .max_temperature_c
                .is_some_and(|t| !(1..=100).contains(&t))
            || q.cpu_threads
                .is_none_or(|n| n == 0 || n > self.max_cpu_threads)
            || q.loading_bytes
                .0
                .checked_add(q.replica.request_bytes)
                .is_none_or(|n| n > self.max_host_bytes.0)
            || self
                .min_free_host_bytes
                .0
                .checked_add(self.max_host_bytes.0)
                .is_none()
        {
            return Err(Error::Denied);
        }
        Ok(())
    }
    fn check(&self, report: &HostReport, bytes: u64, threads: u32) -> Result<(), Error> {
        let enough_memory = self
            .min_free_host_bytes
            .0
            .checked_add(bytes)
            .is_some_and(|need| {
                report
                    .memory_pools
                    .iter()
                    .any(|p| p.pool_id == "host" && p.available_bytes.0 >= need)
            });
        let enough_threads = matches!(report.available_cpu_threads, Observation::Measured(n) if n >= threads as usize);
        let temperature_ok = self.max_temperature_c.is_none_or(|limit|
            matches!(report.max_sensor_temperature_c, Observation::Measured(t) if t.is_finite() && t <= limit as f32));
        if report.stale
            || !enough_memory
            || !enough_threads
            || !temperature_ok
            || matches!(report.thermal_throttled, Observation::Measured(true))
        {
            Err(Error::Capacity)
        } else {
            Ok(())
        }
    }
}
/// Locally collected timestamped sample; no remote deserialization or timer refresh on reads.
pub struct ContributionGate {
    policy: CpuContribution,
    loading: u64,
    request: u64,
    threads: u32,
    sample: Mutex<Option<HostSnapshot>>,
    closed: AtomicBool,
}
impl ContributionGate {
    pub fn new(policy: CpuContribution, q: &Qualification) -> Result<Self, Error> {
        policy.validate(q)?;
        Ok(Self {
            policy,
            loading: q.loading_bytes.0,
            request: q.replica.request_bytes,
            threads: q.cpu_threads.ok_or(Error::Denied)?,
            sample: Mutex::new(None),
            closed: AtomicBool::new(false),
        })
    }
    pub fn update(&self, sample: HostSnapshot) -> Result<(), Error> {
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::Capacity);
        }
        *self.sample.lock().map_err(|_| Error::Denied)? = Some(sample);
        self.maintain()
    }
    pub fn needs_temperature(&self) -> bool {
        self.policy.max_temperature_c.is_some()
    }
    pub fn check(&self) -> Result<(), Error> {
        self.with_eligible(|| Ok(()))
    }
    /// Loaded/in-flight work already owns its budget; check pressure without reserving it twice.
    pub fn maintain(&self) -> Result<(), Error> {
        self.with_budget(0, || Ok(()))
    }
    /// Only short local state/enqueue work in the callback; never backend/network I/O.
    pub fn with_eligible<T>(&self, action: impl FnOnce() -> Result<T, Error>) -> Result<T, Error> {
        self.with_budget(self.request, action)
    }
    fn with_budget<T>(
        &self,
        bytes: u64,
        action: impl FnOnce() -> Result<T, Error>,
    ) -> Result<T, Error> {
        let sample = self.sample.lock().map_err(|_| Error::Denied)?;
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::Capacity);
        }
        if self
            .policy
            .check(
                &sample.as_ref().ok_or(Error::Capacity)?.report(),
                bytes,
                self.threads,
            )
            .is_err()
        {
            self.closed.store(true, Ordering::Release);
            return Err(Error::Capacity);
        }
        action()
    }
    pub fn withdraw(&self) {
        let _sample = self.sample.lock();
        self.closed.store(true, Ordering::Release);
    }
    pub fn has_sample(&self) -> bool {
        self.sample.lock().is_ok_and(|s| s.is_some())
    }
    pub fn loading_allowed(&self) -> Result<(), Error> {
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::Capacity);
        }
        let sample = self.sample.lock().map_err(|_| Error::Denied)?;
        self.policy.check(
            &sample.as_ref().ok_or(Error::Capacity)?.report(),
            self.loading,
            self.threads,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pressure_and_missing_required_temperature_never_grant_capacity() {
        let mut report = HostSnapshot::collect_memory().report();
        let policy = CpuContribution {
            version: Version,
            max_host_bytes: Decimal(4096),
            min_free_host_bytes: Decimal(100),
            max_cpu_threads: 2,
            max_temperature_c: Some(70),
        };
        report.memory_pools[0].available_bytes = Decimal(500);
        report.available_cpu_threads = Observation::Measured(2);
        report.max_sensor_temperature_c = Observation::Measured(60.0);
        assert!(policy.check(&report, 400, 2).is_ok());
        report.stale = true;
        assert!(policy.check(&report, 400, 2).is_err());
        report.stale = false;
        report.max_sensor_temperature_c = Observation::Unavailable;
        assert!(policy.check(&report, 400, 2).is_err());
        report.max_sensor_temperature_c = Observation::Measured(71.0);
        assert!(policy.check(&report, 400, 2).is_err());
        report.max_sensor_temperature_c = Observation::Measured(f32::NAN);
        assert!(policy.check(&report, 400, 2).is_err());
        report.max_sensor_temperature_c = Observation::Measured(60.0);
        report.memory_pools[0].available_bytes = Decimal(499);
        assert!(policy.check(&report, 400, 2).is_err());
        assert!(
            policy.check(&report, 0, 2).is_ok(),
            "an active allocation is not requested a second time"
        );
        report.memory_pools[0].available_bytes = Decimal(500);
        report.available_cpu_threads = Observation::Unavailable;
        assert!(policy.check(&report, 400, 2).is_err());
    }
}
