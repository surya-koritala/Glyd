//! Verified immutable objects, bounded resumable staging and durable lease pins.
use super::{Catalog, fingerprint, open_source, read_bounded};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use starlings_protocol::{Asset, Decimal, Digest, LeaseId, ModelManifest, parse_json};
use std::{
    collections::BTreeSet,
    io::{Read, Seek, SeekFrom, Write},
};

const DEFAULT_QUOTA: u64 = 16 * 1024 * 1024 * 1024;
const MAX_ENTRIES: usize = 16_384;
const STAGING_COPIES: u64 = 2;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Pin {
    version: u32,
    lease: LeaseId,
    manifest: Digest,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServicePhase {
    #[default]
    Quarantined,
    Drained,
    Recovered,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceRecord {
    pub version: u32,
    pub lease: LeaseId,
    pub manifest: Digest,
    pub configuration: Digest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cleanup_qualification: Option<Digest>,
    #[serde(default)]
    pub phase: ServicePhase,
}

#[derive(Serialize)]
pub struct StorageCost {
    pub whole_manifest_storage_bytes: Decimal,
    pub selected_storage_bytes: Decimal,
    pub declared_resident_weight_bytes: Decimal,
    pub transfer_buffer_bytes: Decimal,
    /// Conservative extra quota for staging/checkpointing the largest selected asset.
    /// Does not include current cache files or an external backend's allocations.
    pub max_asset_import_reserve_bytes: Decimal,
    pub backend_peak_host_bytes: Option<Decimal>,
}

impl Catalog {
    /// Whole storage shards are required even when executing only some of their layers.
    pub fn storage_cost(&self, digest: &Digest, start: u32, end: u32) -> Result<StorageCost> {
        use starlings_protocol::{AssetRole, ComponentDisposition, ComponentOwnerRole};
        let manifest = self.authorized(digest, false)?;
        if start >= end || end as usize > manifest.layers.len() {
            return Err(Error::Invalid);
        }
        let mut paths = BTreeSet::new();
        let mut resident = 0u64;
        for layer in &manifest.layers[start as usize..end as usize] {
            paths.extend(layer.asset_paths.iter());
            resident = resident
                .checked_add(layer.resident_weight_bytes.0)
                .ok_or(Error::Capacity)?;
        }
        for component in &manifest.non_layer_components {
            if component.disposition == ComponentDisposition::Executed
                && ((start == 0 && component.owner_role == ComponentOwnerRole::First)
                    || (end as usize == manifest.layers.len()
                        && component.owner_role == ComponentOwnerRole::Last))
            {
                paths.extend(component.asset_paths.iter());
                resident = resident
                    .checked_add(component.resident_weight_bytes.0)
                    .ok_or(Error::Capacity)?;
            }
        }
        let mut all = std::collections::BTreeMap::new();
        let mut selected = BTreeSet::new();
        for asset in &manifest.assets {
            if all
                .insert(&asset.sha256, asset.size_bytes.0)
                .is_some_and(|old| old != asset.size_bytes.0)
            {
                return Err(Error::Invalid);
            }
            if asset.role != AssetRole::Weights || paths.contains(&asset.path) {
                selected.insert(&asset.sha256);
            }
        }
        let whole = all
            .values()
            .try_fold(0u64, |n, b| n.checked_add(*b).ok_or(Error::Capacity))?;
        let max_import_reserve = selected
            .iter()
            .map(|id| all[id])
            .max()
            .unwrap_or(0)
            .checked_mul(STAGING_COPIES)
            .ok_or(Error::Capacity)?;
        let selected = selected
            .into_iter()
            .try_fold(0u64, |n, id| n.checked_add(all[id]).ok_or(Error::Capacity))?;
        Ok(StorageCost {
            whole_manifest_storage_bytes: Decimal(whole),
            selected_storage_bytes: Decimal(selected),
            declared_resident_weight_bytes: Decimal(resident),
            transfer_buffer_bytes: Decimal(64 * 1024),
            max_asset_import_reserve_bytes: Decimal(max_import_reserve),
            backend_peak_host_bytes: None,
        })
    }
    pub fn quota(&self) -> Result<u64> {
        match self.lock.directory().open_file("quota.json") {
            Ok(file) => {
                let value: Decimal = serde_json::from_value(parse_json(&read_bounded(file)?)?)?;
                Ok(value.0)
            }
            Err(private_fs::Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
                Ok(DEFAULT_QUOTA)
            }
            Err(e) => Err(e.into()),
        }
    }
    pub fn set_quota(&mut self, bytes: u64) -> Result<()> {
        if bytes == 0 || bytes < self.storage_bytes()? {
            return Err(Error::Capacity);
        }
        self.write_record("quota.json", &serde_json::to_vec(&Decimal(bytes))?)?;
        Ok(())
    }
    pub(super) fn write_record(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        let names = self.entries()?;
        if !names.iter().any(|n| n == name) && names.len() > MAX_ENTRIES - 2 {
            return Err(Error::Capacity);
        }
        self.lock.directory().write_atomic_durable(name, bytes)?;
        Ok(())
    }
    fn entries(&self) -> Result<Vec<String>> {
        let mut names = Vec::new();
        for entry in self.files.entries().map_err(|_| Error::Storage)? {
            let name = entry
                .map_err(|_| Error::Storage)?
                .file_name()
                .into_string()
                .map_err(|_| Error::Storage)?;
            if names.len() == MAX_ENTRIES {
                return Err(Error::Capacity);
            }
            names.push(name);
        }
        Ok(names)
    }
    /// Include staging and orphaned atomic files, not just indexed/ready objects.
    pub fn storage_bytes(&self) -> Result<u64> {
        self.entries()?.into_iter().try_fold(0u64, |total, name| {
            let mut file = self.lock.directory().open_file(name)?;
            total
                .checked_add(file.seek(SeekFrom::End(0)).map_err(|_| Error::Storage)?)
                .ok_or(Error::Capacity)
        })
    }
    fn asset(&self, manifest: &Digest, path: &str) -> Result<Asset> {
        self.authorized(manifest, false)?
            .assets
            .into_iter()
            .find(|a| a.path == path)
            .ok_or(Error::Denied)
    }
    fn verify_object(&self, asset: &Asset) -> Result<private_fs::PrivateFile> {
        let mut file = self
            .lock
            .directory()
            .open_file(format!("object-{}", asset.sha256.as_str()))?;
        if fingerprint(&mut file, asset.size_bytes.0)? != asset.sha256 {
            return Err(Error::Invalid);
        }
        file.rewind().map_err(|_| Error::Storage)?;
        Ok(file)
    }
    pub fn open_asset(&self, manifest: &Digest, path: &str) -> Result<private_fs::PrivateFile> {
        self.verify_object(&self.asset(manifest, path)?)
    }
    /// Verified native path for an approved external process. Keep this catalog
    /// locked and pin the manifest before launching. The local account is trusted
    /// not to replace files between verification and the process opening them.
    pub fn asset_path(&self, manifest: &Digest, path: &str) -> Result<std::path::PathBuf> {
        let current = crate::PrivateDir::open(&self.path)?;
        #[cfg(unix)]
        {
            let held = rustix::fs::fstat(self.lock.directory()).map_err(|_| Error::Storage)?;
            let named = rustix::fs::fstat(&current).map_err(|_| Error::Storage)?;
            if held.st_dev != named.st_dev || held.st_ino != named.st_ino {
                return Err(Error::Denied);
            }
        }
        let asset = self.asset(manifest, path)?;
        let name = format!("object-{}", asset.sha256.as_str());
        let file = current.open_file(&name)?;
        if fingerprint(file, asset.size_bytes.0)? != asset.sha256 {
            return Err(Error::Invalid);
        }
        let path = self.path.join(name);
        if path.canonicalize().map_err(|_| Error::Storage)? != path {
            return Err(Error::Denied);
        }
        Ok(path)
    }
    /// Full revalidation is preparation work, not a per-token/per-chunk operation.
    pub fn ready(&self, manifest: &Digest) -> Result<ModelManifest> {
        let manifest = self.authorized(manifest, false)?;
        for asset in &manifest.assets {
            self.verify_object(asset)?;
        }
        Ok(manifest)
    }
    pub fn resume_offset(&self, manifest: &Digest, path: &str) -> Result<u64> {
        let asset = self.asset(manifest, path)?;
        match self
            .lock
            .directory()
            .open_file(format!("partial-{}", asset.sha256.as_str()))
        {
            Ok(mut file) => {
                let length = file.seek(SeekFrom::End(0)).map_err(|_| Error::Storage)?;
                if length > asset.size_bytes.0 {
                    return Err(Error::Invalid);
                }
                Ok(length)
            }
            Err(private_fs::Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => Ok(0),
            Err(e) => Err(e.into()),
        }
    }
    /// Reader starts at `offset`; final SHA-256 covers prefix and suffix together.
    /// A failed source read retains only bounded partial bytes, never a ready object.
    pub fn import_asset(
        &mut self,
        manifest: &Digest,
        path: &str,
        offset: u64,
        mut reader: impl Read,
    ) -> Result<()> {
        let asset = self.asset(manifest, path)?;
        if self.verify_object(&asset).is_ok() {
            return Ok(());
        }
        if self.pinned_objects()?.contains(&asset.sha256) {
            return Err(Error::Denied);
        }
        if offset > asset.size_bytes.0
            || (offset != 0 && offset != self.resume_offset(manifest, path)?)
        {
            return Err(Error::Invalid);
        }
        if self.entries()?.len() > MAX_ENTRIES - 4 {
            return Err(Error::Capacity);
        }
        let reserve = asset
            .size_bytes
            .0
            .checked_mul(STAGING_COPIES)
            .ok_or(Error::Capacity)?;
        let quota = self.quota()?;
        if self
            .storage_bytes()?
            .checked_add(reserve)
            .is_none_or(|n| n > quota)
        {
            return Err(Error::Capacity);
        }
        let partial = format!("partial-{}", asset.sha256.as_str());
        let mut stage = self
            .lock
            .directory()
            .atomic_durable(format!("object-{}", asset.sha256.as_str()))?;
        if offset != 0 {
            let copied = std::io::copy(
                &mut self.lock.directory().open_file(&partial)?.take(offset),
                &mut stage,
            )
            .map_err(|_| Error::Storage)?;
            if copied != offset {
                return Err(Error::Invalid);
            }
        }
        let mut written = offset;
        let mut buffer = [0; 64 * 1024];
        let result = (|| -> Result<()> {
            while written < asset.size_bytes.0 {
                let wanted = (asset.size_bytes.0 - written).min(buffer.len() as u64) as usize;
                let count = reader.read(&mut buffer[..wanted])?;
                if count == 0 {
                    return Err(Error::Transport);
                }
                stage
                    .write_all(&buffer[..count])
                    .map_err(|_| Error::Storage)?;
                written += count as u64;
            }
            if reader.read(&mut buffer[..1])? != 0 {
                return Err(Error::Invalid);
            }
            stage.rewind().map_err(|_| Error::Storage)?;
            if fingerprint(&mut stage, asset.size_bytes.0)? != asset.sha256 {
                return Err(Error::Invalid);
            }
            Ok(())
        })();
        if let Err(error) = result {
            // A digest mismatch must restart, rather than resume a known-invalid prefix.
            if error == Error::Invalid {
                self.lock.directory().write_atomic_durable(&partial, &[])?;
            } else if written != 0 && error != Error::Storage {
                stage.rewind().map_err(|_| Error::Storage)?;
                let mut checkpoint = self.lock.directory().atomic_durable(&partial)?;
                let copied = std::io::copy(&mut stage.take(written), &mut checkpoint)
                    .map_err(|_| Error::Storage)?;
                if copied != written {
                    return Err(Error::Storage);
                }
                checkpoint.commit()?;
            }
            return Err(error);
        }
        stage.commit()?;
        self.remove_if_present(&partial)?;
        Ok(())
    }
    /// Import source contents read-only; stored names are digests, not source paths.
    pub fn import_directory(&mut self, manifest: &Digest, source: &std::path::Path) -> Result<()> {
        let source = cap_std::fs::Dir::open_ambient_dir(source, cap_std::ambient_authority())
            .map_err(|_| Error::Storage)?;
        for asset in self.authorized(manifest, false)?.assets {
            self.import_asset(manifest, &asset.path, 0, open_source(&source, &asset.path)?)?;
        }
        Ok(())
    }
    /// Refuse another launch while previous process/device cleanup is unqualified.
    pub fn ensure_no_service(&self) -> Result<()> {
        if self
            .service_record()?
            .is_some_and(|s| s.phase != ServicePhase::Recovered)
        {
            Err(Error::Denied)
        } else {
            Ok(())
        }
    }
    pub fn service_record(&self) -> Result<Option<ServiceRecord>> {
        match self.lock.directory().open_file("service-start.json") {
            Ok(file) => {
                let start: ServiceRecord =
                    serde_json::from_value(parse_json(&read_bounded(file)?)?)?;
                if !matches!(start.version, 1 | 2)
                    || start.version == 1
                        && (start.cleanup_qualification.is_some()
                            || start.phase != ServicePhase::Quarantined)
                    || start.phase != ServicePhase::Quarantined
                        && start.cleanup_qualification.is_none()
                    || start
                        .cleanup_qualification
                        .as_ref()
                        .is_some_and(|d| d.as_str().bytes().all(|b| b == b'0'))
                {
                    return Err(Error::Storage);
                }
                Ok(Some(start))
            }
            Err(private_fs::Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
    /// Persist before spawn; no parent PID or process-group-only recovery.
    pub fn begin_service(
        &mut self,
        manifest: &Digest,
        lease: LeaseId,
        configuration: &Digest,
    ) -> Result<()> {
        self.begin_service_qualified(manifest, lease, configuration, None)
    }
    /// Cleanup qualification is approved before launch, never retrofitted to an old quarantine.
    pub fn begin_service_qualified(
        &mut self,
        manifest: &Digest,
        lease: LeaseId,
        configuration: &Digest,
        cleanup_qualification: Option<Digest>,
    ) -> Result<()> {
        self.ensure_no_service()?;
        if self.service_record()?.is_some_and(|s| s.lease == lease)
            || cleanup_qualification
                .as_ref()
                .is_some_and(|d| d.as_str().bytes().all(|b| b == b'0'))
        {
            return Err(Error::Replay);
        }
        self.pin(manifest, lease)?;
        self.write_record(
            "service-start.json",
            &serde_json::to_vec(&ServiceRecord {
                version: 2,
                lease,
                manifest: manifest.clone(),
                configuration: configuration.clone(),
                cleanup_qualification,
                phase: ServicePhase::Quarantined,
            })?,
        )
    }
    /// Trusted local owner only: caller has completed the exact qualified backend/device cleanup.
    /// Network completion, an operator-supplied receipt or current PID absence cannot call this path.
    pub fn record_service_drained(
        &mut self,
        lease: LeaseId,
        configuration: &Digest,
        qualification: &Digest,
    ) -> Result<()> {
        let mut start = self.service_record()?.ok_or(Error::Denied)?;
        if start.lease != lease
            || &start.configuration != configuration
            || start.cleanup_qualification.as_ref() != Some(qualification)
            || start.phase == ServicePhase::Recovered
        {
            return Err(Error::Denied);
        }
        start.phase = ServicePhase::Drained;
        self.write_record("service-start.json", &serde_json::to_vec(&start)?)
    }
    /// Explicit local recovery of an already durable, qualified cleanup receipt.
    /// Pin removal precedes the atomic terminal record; interruption keeps quarantine protection.
    pub fn recover_service(&mut self, lease: LeaseId, configuration: &Digest) -> Result<bool> {
        let start = self.service_record()?.ok_or(Error::Denied)?;
        if start.lease != lease
            || &start.configuration != configuration
            || start.phase == ServicePhase::Quarantined
        {
            return Err(Error::Denied);
        }
        if start.phase == ServicePhase::Recovered {
            return Ok(true);
        }
        #[cfg(not(unix))]
        {
            Err(Error::Denied)
        }
        #[cfg(unix)]
        {
            let mut start = start;
            let name = format!("pin-{lease}.json");
            match self.lock.directory().open_file(&name) {
                Ok(file) => {
                    let pin: Pin = serde_json::from_value(parse_json(&read_bounded(file)?)?)?;
                    if pin.version != 1 || pin.lease != lease || pin.manifest != start.manifest {
                        return Err(Error::Storage);
                    }
                }
                Err(private_fs::Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => (),
                Err(e) => return Err(e.into()),
            }
            self.remove_if_present(&name)?;
            // Make absence durable before releasing the marker's independent protection.
            rustix::fs::fsync(self.lock.directory()).map_err(|_| Error::Storage)?;
            start.phase = ServicePhase::Recovered;
            self.write_record("service-start.json", &serde_json::to_vec(&start)?)?;
            Ok(false)
        }
    }
    /// Pins survive restart. Only explicit release after backend drain removes them.
    pub fn pin(&mut self, manifest: &Digest, lease: LeaseId) -> Result<()> {
        self.ready(manifest)?;
        let name = format!("pin-{lease}.json");
        match self.lock.directory().open_file(&name) {
            Ok(file) => {
                let old: Pin = serde_json::from_value(parse_json(&read_bounded(file)?)?)?;
                if old.version != 1 || old.lease != lease || &old.manifest != manifest {
                    return Err(Error::Replay);
                }
                return Ok(());
            }
            Err(private_fs::Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => (),
            Err(e) => return Err(e.into()),
        }
        if self.entries()?.len() >= MAX_ENTRIES - 1 {
            return Err(Error::Capacity);
        }
        self.write_record(
            &name,
            &serde_json::to_vec(&Pin {
                version: 1,
                lease,
                manifest: manifest.clone(),
            })?,
        )?;
        Ok(())
    }
    pub fn unpin(&mut self, lease: LeaseId) -> Result<()> {
        if self
            .service_record()?
            .is_some_and(|s| s.lease == lease && s.phase != ServicePhase::Recovered)
        {
            return Err(Error::Denied);
        }
        self.remove_if_present(&format!("pin-{lease}.json"))
    }
    fn pinned_objects(&self) -> Result<BTreeSet<Digest>> {
        let mut pinned = BTreeSet::new();
        if let Some(start) = self
            .service_record()?
            .filter(|s| s.phase != ServicePhase::Recovered)
        {
            pinned.extend(
                self.manifest(&start.manifest)?
                    .assets
                    .into_iter()
                    .map(|a| a.sha256),
            );
        }
        for name in self
            .entries()?
            .into_iter()
            .filter(|n| n.starts_with("pin-"))
        {
            let pin: Pin = serde_json::from_value(parse_json(&read_bounded(
                self.lock.directory().open_file(&name)?,
            )?)?)?;
            if pin.version != 1 || name != format!("pin-{}.json", pin.lease) {
                return Err(Error::Storage);
            }
            // Revocation does not release potentially active backend file references.
            pinned.extend(
                self.manifest(&pin.manifest)?
                    .assets
                    .into_iter()
                    .map(|a| a.sha256),
            );
        }
        Ok(pinned)
    }
    /// Local files only, under exclusive StateLock. No manifest path is used for deletion.
    fn remove_if_present(&self, name: &str) -> Result<()> {
        match self.lock.directory().open_file(name) {
            Ok(file) => {
                drop(file);
                self.files.remove_file(name).map_err(|_| Error::Storage)?;
            }
            Err(private_fs::Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => (),
            Err(e) => return Err(e.into()),
        }
        Ok(())
    }
    /// Explicit GC keeps all pinned content even when its manifest approval is revoked.
    pub fn gc(&mut self) -> Result<u64> {
        let pinned = self.pinned_objects()?;
        let mut removed = 0;
        for name in self.entries()? {
            let object = name
                .strip_prefix("object-")
                .map(|s| s.to_owned().try_into())
                .transpose()
                .map_err(|_| Error::Storage)?;
            let disposable = object.as_ref().is_some_and(|hash| !pinned.contains(hash))
                || name
                    .strip_prefix("partial-")
                    .is_some_and(|s| Digest::try_from(s.to_owned()).is_ok())
                || name.starts_with(".private-fs-") && name.ends_with(".tmp");
            if disposable {
                self.remove_if_present(&name)?;
                removed += 1;
            }
        }
        Ok(removed)
    }
}
