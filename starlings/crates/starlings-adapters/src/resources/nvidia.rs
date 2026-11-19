//! Installed NVML is trusted native code, explicitly selected and content-approved locally.
use super::Observation;
use nvml_wrapper::{
    Nvml, bitmasks::device::ThrottleReasons, enum_wrappers::device::TemperatureSensor,
    enums::device::BusType, error::NvmlError,
};
use serde::Serialize;
use starlings_protocol::{Decimal, Digest, MemoryPool, MemoryPoolKind};
use std::{collections::BTreeSet, path::Path};

#[derive(Clone, Debug, Serialize)]
pub struct NvidiaReport {
    pub library: Digest,
    pub driver: Observation<String>,
    pub devices: Vec<NvidiaDevice>,
}
#[derive(Clone, Debug, Serialize)]
pub struct NvidiaDevice {
    pub device_id: String,
    pub name: Observation<String>,
    pub memory: Observation<NvidiaMemory>,
    pub unshared_device_memory: Observation<bool>,
    pub temperature_c: Observation<u32>,
    pub thermal_throttled: Observation<bool>,
}
#[derive(Clone, Debug, Serialize)]
pub struct NvidiaMemory {
    pub total_bytes: Decimal,
    pub available_bytes: Decimal,
}
impl NvidiaDevice {
    pub(super) fn pool(&self) -> Option<MemoryPool> {
        let Observation::Estimated(memory) = &self.memory else {
            return None;
        };
        if !matches!(self.unshared_device_memory, Observation::Measured(true))
            || !valid_id(&self.device_id)
            || memory.total_bytes.0 == 0
            || memory.available_bytes.0 > memory.total_bytes.0
        {
            return None;
        }
        Some(MemoryPool {
            pool_id: format!("nvidia:{}", self.device_id),
            kind: MemoryPoolKind::Device,
            total_bytes: memory.total_bytes,
            available_bytes: memory.available_bytes,
            reserved_bytes: Decimal(0),
            device_ids: vec![self.device_id.clone()],
        })
    }
}
fn valid_id(id: &str) -> bool {
    id.strip_prefix("GPU-")
        .and_then(|s| uuid::Uuid::parse_str(s).ok().map(|u| u.to_string() == s))
        .unwrap_or(false)
}
pub(super) fn collect(library: &Path, approved: &Digest) -> Result<NvidiaReport, &'static str> {
    if !matches!(std::env::consts::OS, "linux" | "windows") {
        return Err("NVIDIA_PROBE_UNSUPPORTED_OS");
    }
    // No loader search via PATH/current directory; local approval does not attest dependencies.
    if !library.is_absolute()
        || library
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err("NVIDIA_LIBRARY_ABSOLUTE_PATH_REQUIRED");
    }
    #[cfg(windows)]
    if library.components().any(|c| {
        matches!(c, std::path::Component::Prefix(p)
        if !matches!(p.kind(), std::path::Prefix::Disk(_) | std::path::Prefix::VerbatimDisk(_)))
    }) {
        return Err("NVIDIA_LIBRARY_LOCAL_PATH_REQUIRED");
    }
    let library = library
        .canonicalize()
        .map_err(|_| "NVIDIA_LIBRARY_UNAVAILABLE")?;
    if !library.is_file()
        || super::super::service::executable_digest(&library).as_ref() != Ok(approved)
    {
        return Err("NVIDIA_LIBRARY_UNAPPROVED");
    }
    let nvml = Nvml::builder()
        .lib_path(library.as_os_str())
        .init()
        .map_err(|_| "NVIDIA_PROBE_UNAVAILABLE")?;
    let count = nvml
        .device_count()
        .map_err(|_| "NVIDIA_DEVICE_INVENTORY_UNAVAILABLE")?;
    // NodeCapabilities permits sixteen pools including host RAM. Never silently truncate.
    if count > 15 {
        return Err("NVIDIA_DEVICE_LIMIT");
    }
    let mut devices = Vec::new();
    let mut ids = BTreeSet::new();
    for index in 0..count {
        let device = nvml
            .device_by_index(index)
            .map_err(|_| "NVIDIA_DEVICE_UNAVAILABLE")?;
        let id = device.uuid().map_err(|_| "NVIDIA_DEVICE_ID_UNAVAILABLE")?;
        if !valid_id(&id) || !ids.insert(id.clone()) {
            return Err("NVIDIA_DEVICE_ID_INVALID");
        }
        let mig_disabled = match device.mig_mode() {
            Ok(mode) => Some(mode.current == 0 && mode.pending == 0),
            Err(NvmlError::NotSupported) => Some(true),
            Err(_) => None,
        };
        let separate = match device.numa_node_id() {
            Ok(node) => Some(node == u32::MAX),
            Err(NvmlError::NotSupported) => Some(true),
            Err(_) => None,
        };
        #[cfg(target_os = "linux")]
        let physical = device
            .virtualization_mode()
            .ok()
            .map(|mode| mode == nvml_wrapper::enum_wrappers::device::GpuVirtualizationMode::Bare);
        #[cfg(not(target_os = "linux"))]
        let physical = None; // Wrapper lacks the Windows virtualization query; unknown is not bare.
        let memory = device
            .memory_info()
            .ok()
            .and_then(|m| memory(m.total, m.free, m.used, m.reserved))
            .map(Observation::Estimated)
            .unwrap_or(Observation::Unavailable);
        devices.push(NvidiaDevice {
            device_id: id,
            name: observed(device.name()),
            memory,
            unshared_device_memory: unshared_memory([
                physical,
                separate,
                mig_disabled,
                device.bus_type().ok().map(|b| b == BusType::Pcie),
            ]),
            temperature_c: observed(device.temperature(TemperatureSensor::Gpu).and_then(|t| {
                if t <= 200 {
                    Ok(t)
                } else {
                    Err(NvmlError::Unknown)
                }
            })),
            thermal_throttled: observed(device.current_throttle_reasons_strict().map(thermal)),
        });
    }
    Ok(NvidiaReport {
        library: approved.clone(),
        driver: observed(nvml.sys_driver_version()),
        devices,
    })
}
fn memory(total: u64, free: u64, used: u64, reserved: u64) -> Option<NvidiaMemory> {
    // NVML v2 includes driver-reserved memory in used; adding it again rejects valid samples.
    (total > 0 && reserved <= used && used.checked_add(free).is_some_and(|n| n <= total)).then_some(
        NvidiaMemory {
            total_bytes: Decimal(total),
            available_bytes: Decimal(free),
        },
    )
}
fn unshared_memory(states: [Option<bool>; 4]) -> Observation<bool> {
    if states.contains(&Some(false)) {
        Observation::Measured(false)
    } else if states.iter().all(|s| *s == Some(true)) {
        Observation::Measured(true)
    } else {
        Observation::Unavailable
    }
}
fn observed<T>(value: Result<T, NvmlError>) -> Observation<T> {
    value
        .map(Observation::Measured)
        .unwrap_or(Observation::Unavailable)
}
fn thermal(reasons: ThrottleReasons) -> bool {
    reasons.intersects(ThrottleReasons::SW_THERMAL_SLOWDOWN | ThrottleReasons::HW_THERMAL_SLOWDOWN)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn memory_pools_require_unique_physical_ids_and_do_not_infer_throttle_from_idle() {
        let mut device = NvidiaDevice {
            device_id: "GPU-00000000-0000-4000-8000-000000000003".into(),
            name: Observation::Unavailable,
            memory: Observation::Estimated(NvidiaMemory {
                total_bytes: Decimal(1000),
                available_bytes: Decimal(700),
            }),
            unshared_device_memory: Observation::Measured(true),
            temperature_c: Observation::Unavailable,
            thermal_throttled: Observation::Unavailable,
        };
        assert_eq!(device.pool().unwrap().available_bytes.0, 700);
        device.unshared_device_memory = Observation::Measured(false);
        assert!(
            device.pool().is_none(),
            "MIG/virtual/shared/unknown memory cannot become a discrete pool"
        );
        device.unshared_device_memory = Observation::Measured(true);
        device.device_id = "GPU-invalid".into();
        assert!(device.pool().is_none());
        #[cfg(windows)]
        assert!(matches!(
            collect(
                Path::new(r"\\server\share\nvml.dll"),
                &starlings_protocol::byte_digest(b"fixture")
            ),
            Err("NVIDIA_LIBRARY_LOCAL_PATH_REQUIRED")
        ));
        assert!(!thermal(
            ThrottleReasons::GPU_IDLE | ThrottleReasons::SW_POWER_CAP
        ));
        assert!(thermal(ThrottleReasons::HW_THERMAL_SLOWDOWN));
        assert!(thermal(ThrottleReasons::SW_THERMAL_SLOWDOWN));
        assert!(memory(1000, 100, 900, 100).is_some());
        assert!(memory(1000, 200, 900, 100).is_none());
        assert!(memory(u64::MAX, u64::MAX, 1, 0).is_none());
        assert!(memory(1000, 900, 100, 101).is_none());
        assert!(matches!(
            unshared_memory([Some(true); 4]),
            Observation::Measured(true)
        ));
        assert!(matches!(
            unshared_memory([Some(true), None, Some(true), Some(true)]),
            Observation::Unavailable
        ));
        assert!(matches!(
            unshared_memory([Some(true), None, Some(false), Some(true)]),
            Observation::Measured(false)
        ));
        assert!(
            collect(
                Path::new("untrusted-relative-library"),
                &starlings_protocol::byte_digest(b"fixture")
            )
            .is_err()
        );
    }
}
