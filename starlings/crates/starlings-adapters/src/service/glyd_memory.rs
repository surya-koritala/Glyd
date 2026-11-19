//! Process-owned allocator receipts, never a physical pool inventory or native drain proof.
use super::*;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlydMemory {
    pub gpu: u32,
    pub measured_at: f64,
    pub engine_bytes: u64,
    pub weights_bytes: Option<u64>,
    pub kv_bytes: Option<u64>,
    pub state_bytes: Option<u64>,
    pub scratch_bytes: Option<u64>,
    pub peak_bytes: u64,
    pub peak_at: f64,
    pub peak_since: f64,
    pub device_total_bytes: u64,
    pub device_free_bytes: u64,
    pub not_counted: Vec<String>,
}
impl GlydMemory {
    fn from_info(info: &serde_json::Value) -> Result<Self> {
        let value = &info["memory"];
        // A missing allocation is not an explicitly unavailable component.
        for key in ["weights_bytes", "kv_bytes", "state_bytes", "scratch_bytes"] {
            value.get(key).ok_or(Error::Invalid)?;
        }
        let memory: Self = serde_json::from_value(value.clone()).map_err(|_| Error::Invalid)?;
        let parts = [
            memory.weights_bytes,
            memory.kv_bytes,
            memory.state_bytes,
            memory.scratch_bytes,
        ];
        let known = parts
            .iter()
            .flatten()
            .try_fold(0u64, |sum, n| sum.checked_add(*n))
            .ok_or(Error::Invalid)?;
        if info["device"]["ordinal"].as_u64() != Some(u64::from(memory.gpu))
            || ![memory.measured_at, memory.peak_at, memory.peak_since]
                .iter()
                .all(|t| t.is_finite() && *t >= 0.0)
            || memory.peak_since > memory.peak_at
            || memory.peak_at > memory.measured_at
            || memory.device_total_bytes == 0
            || memory.device_free_bytes > memory.device_total_bytes
            || memory.engine_bytes > memory.peak_bytes
            || memory.peak_bytes > memory.device_total_bytes
            || known > memory.engine_bytes
            || (parts.iter().all(Option::is_some) && known != memory.engine_bytes)
            || memory.not_counted.len() > 128
            || memory
                .not_counted
                .iter()
                .any(|s| s.is_empty() || s.len() > 256 || s.chars().any(char::is_control))
        {
            return Err(Error::Invalid);
        }
        Ok(memory)
    }
}

pub(super) struct GlydMemorySample {
    memory: GlydMemory,
    observed: Instant,
}
impl GlydMemorySample {
    pub(super) fn observe(
        previous: Option<&Self>,
        info: &serde_json::Value,
        now: Instant,
    ) -> Result<Self> {
        let memory = GlydMemory::from_info(info)?;
        let mut observed = now;
        if let Some(old) = previous {
            if memory.gpu != old.memory.gpu
                || memory.device_total_bytes != old.memory.device_total_bytes
                || memory.peak_since != old.memory.peak_since
                || memory.measured_at < old.memory.measured_at
                || memory.peak_bytes < old.memory.peak_bytes
                || memory.peak_at < old.memory.peak_at
            {
                return Err(Error::Unapproved);
            }
            if memory.measured_at == old.memory.measured_at {
                // Millisecond timestamps are not unique capture IDs. Accept coherent
                // changes (including other processes' free memory) without renewing age.
                observed = old.observed;
            }
        }
        let sample = Self { memory, observed };
        sample.fresh(now).ok_or(Error::Unavailable)?;
        Ok(sample)
    }
    pub(super) fn fresh(&self, now: Instant) -> Option<&GlydMemory> {
        now.checked_duration_since(self.observed)
            .filter(|age| *age < crate::resources::TELEMETRY_TTL)
            .map(|_| &self.memory)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn allocator_receipts_preserve_unknowns_retention_and_capture_age() {
        let value = super::super::fixture::glyd_info();
        let now = Instant::now();
        let sample = GlydMemorySample::observe(None, &value, now).unwrap();
        assert_eq!(sample.memory.kv_bytes, Some(20));
        assert_eq!(sample.memory.engine_bytes, 100);
        assert!(!sample.memory.not_counted.is_empty());
        let same = GlydMemorySample::observe(Some(&sample), &value, now).unwrap();
        assert_eq!(same.observed, sample.observed);
        let mut same_millisecond = value.clone();
        same_millisecond["memory"]["device_free_bytes"] = 699.into();
        let same = GlydMemorySample::observe(Some(&same), &same_millisecond, now).unwrap();
        assert_eq!(same.observed, sample.observed);
        assert_eq!(same.memory.device_free_bytes, 699);
        assert!(same.fresh(now + crate::resources::TELEMETRY_TTL).is_none());
        assert!(
            GlydMemorySample::observe(Some(&same), &value, now + crate::resources::TELEMETRY_TTL)
                .is_err()
        );
        let mut later = value.clone();
        later["memory"]["measured_at"] = 3.0.into();
        let ended = GlydMemorySample::observe(Some(&sample), &later, now).unwrap();
        // Ended requests keep their preallocated KV/state; no subtraction or release.
        assert_eq!(ended.memory.engine_bytes, sample.memory.engine_bytes);
        let mut unavailable = value.clone();
        for key in ["weights_bytes", "kv_bytes", "state_bytes", "scratch_bytes"] {
            unavailable["memory"][key] = serde_json::Value::Null;
        }
        assert_eq!(GlydMemory::from_info(&unavailable).unwrap().kv_bytes, None);
        unavailable["memory"]
            .as_object_mut()
            .unwrap()
            .remove("kv_bytes");
        assert!(GlydMemory::from_info(&unavailable).is_err());
        for (key, replacement) in [
            ("gpu", serde_json::json!(1)),
            ("measured_at", serde_json::json!(-1)),
            ("peak_at", serde_json::json!(4)),
            ("peak_since", serde_json::json!(3)),
            ("engine_bytes", serde_json::json!(99)),
            ("peak_bytes", serde_json::json!(99)),
            ("device_free_bytes", serde_json::json!(1001)),
            ("weights_bytes", serde_json::json!(u64::MAX)),
            ("kv_bytes", serde_json::json!("20")),
            ("not_counted", serde_json::json!([null])),
        ] {
            let mut invalid = value.clone();
            invalid["memory"][key] = replacement;
            assert!(GlydMemory::from_info(&invalid).is_err(), "{key}");
        }
        for (key, replacement) in [
            ("measured_at", serde_json::json!(1.0)),
            ("peak_bytes", serde_json::json!(100)),
            ("peak_since", serde_json::json!(0.0)),
            ("device_total_bytes", serde_json::json!(1001)),
        ] {
            let mut reset = value.clone();
            reset["memory"][key] = replacement;
            assert!(
                GlydMemorySample::observe(Some(&ended), &reset, now).is_err(),
                "{key}"
            );
        }
    }
}
