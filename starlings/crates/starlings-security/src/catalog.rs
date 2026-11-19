//! Exact manifest identity and explicit local approval, separate from artifact readiness.
use crate::{Error, PrivateDir, Result, StateLock};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use starlings_protocol::{
    AssetRole, ComponentDisposition, ComponentOwnerRole, Contract, Digest, FixtureMode,
    MAX_JSON_BYTES, ModelManifest, canonical_digest, decode, encode, parse_json,
};
use std::{io::Read, path::Path};

mod cache;
pub use cache::{ServicePhase, ServiceRecord, StorageCost};
pub mod hub;

/// An approval is local operator policy. A source manifest cannot approve itself.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Approval {
    version: u32,
    manifest: Digest,
    enabled: bool,
    redistribution: bool,
}

/// Mutating operations require exclusive access in-process; StateLock excludes other processes.
pub struct Catalog {
    files: cap_std::fs::Dir,
    path: std::path::PathBuf,
    lock: StateLock,
}
impl Catalog {
    /// Exclusive local owner; no network access, hidden download or backend startup.
    pub fn open(path: &Path) -> Result<Self> {
        Ok(Self {
            lock: StateLock::acquire(path)?,
            files: cap_std::fs::Dir::open_ambient_dir(path, cap_std::ambient_authority())
                .map_err(|_| Error::Storage)?,
            path: path.canonicalize().map_err(|_| Error::Storage)?,
        })
    }

    /// Stores a strictly validated immutable manifest but grants no execution permission.
    pub fn register(&mut self, bytes: &[u8]) -> Result<ModelManifest> {
        let manifest: ModelManifest = decode(bytes, FixtureMode::Reject)?;
        inventory(&manifest)?;
        self.write_record(
            &format!("manifest-{}.json", manifest.manifest_digest.as_str()),
            &encode(&manifest, FixtureMode::Reject)?,
        )?;
        Ok(manifest)
    }

    pub fn manifest(&self, digest: &Digest) -> Result<ModelManifest> {
        let bytes = read_bounded(
            self.lock
                .directory()
                .open_file(format!("manifest-{}.json", digest.as_str()))?,
        )?;
        let manifest: ModelManifest = decode(&bytes, FixtureMode::Reject)?;
        inventory(&manifest)?;
        if &manifest.manifest_digest != digest {
            return Err(Error::Denied);
        }
        Ok(manifest)
    }

    /// Caller must explicitly approve this exact digest after inspecting source/license.
    /// Permission to use locally does not imply permission to redistribute weights.
    pub fn approve(&mut self, digest: &Digest, redistribution: bool) -> Result<()> {
        let manifest = self.manifest(digest)?;
        if redistribution && !manifest.license.redistribution_approved {
            return Err(Error::Denied);
        }
        self.save_approval(&Approval {
            version: 1,
            manifest: digest.clone(),
            enabled: true,
            redistribution,
        })
    }

    pub fn revoke(&mut self, digest: &Digest) -> Result<()> {
        self.manifest(digest)?;
        self.save_approval(&Approval {
            version: 1,
            manifest: digest.clone(),
            enabled: false,
            redistribution: false,
        })
    }

    /// Re-read policy for each new operation; an old returned manifest is not a grant.
    pub fn authorized(&self, digest: &Digest, redistribution: bool) -> Result<ModelManifest> {
        let bytes = read_bounded(
            self.lock
                .directory()
                .open_file(format!("approval-{}.json", digest.as_str()))?,
        )?;
        let approval: Approval = serde_json::from_value(parse_json(&bytes)?)?;
        let manifest = self.manifest(digest)?;
        if approval.version != 1
            || &approval.manifest != digest
            || !approval.enabled
            || redistribution
                && (!approval.redistribution || !manifest.license.redistribution_approved)
        {
            return Err(Error::Denied);
        }
        Ok(manifest)
    }

    fn save_approval(&mut self, approval: &Approval) -> Result<()> {
        self.write_record(
            &format!("approval-{}.json", approval.manifest.as_str()),
            &serde_json::to_vec(approval)?,
        )?;
        Ok(())
    }
}

pub fn read_bounded(reader: impl Read) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader
        .take(MAX_JSON_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_JSON_BYTES {
        return Err(Error::Capacity);
    }
    Ok(bytes)
}

/// Streaming digest with exact expected length; memory is independent of object size.
pub fn fingerprint(mut reader: impl Read, expected_bytes: u64) -> Result<Digest> {
    let mut hash = Sha256::new();
    let mut remaining = expected_bytes;
    let mut buffer = [0; 64 * 1024];
    while remaining != 0 {
        let wanted = remaining.min(buffer.len() as u64) as usize;
        let count = reader
            .read(&mut buffer[..wanted])
            .map_err(|_| Error::Storage)?;
        if count == 0 {
            return Err(Error::Invalid);
        }
        hash.update(&buffer[..count]);
        remaining -= count as u64;
    }
    if reader.read(&mut buffer[..1]).map_err(|_| Error::Storage)? != 0 {
        return Err(Error::Invalid);
    }
    hex_digest(hash.finalize().as_slice())
}

fn hex_digest(bytes: &[u8]) -> Result<Digest> {
    bytes
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>()
        .try_into()
        .map_err(|_| Error::Invalid)
}

/// Rehash all declared assets through protected directory handles. Draft paths are
/// validated before any open. A manifest inventories storage shards, not tensor decoding.
pub fn seal(bytes: &[u8], source: &PrivateDir) -> Result<ModelManifest> {
    seal_with(bytes, |asset| {
        fingerprint(source.open_file(&asset.path)?, asset.size_bytes.0)
    })
}

/// Read-only preloaded source; no permission changes or writes to the source tree.
pub fn seal_preloaded(bytes: &[u8], source: &Path) -> Result<ModelManifest> {
    let directory = cap_std::fs::Dir::open_ambient_dir(source, cap_std::ambient_authority())
        .map_err(|_| Error::Storage)?;
    seal_with(bytes, |asset| {
        fingerprint(open_source(&directory, &asset.path)?, asset.size_bytes.0)
    })
}

fn open_source(directory: &cap_std::fs::Dir, path: &str) -> Result<cap_std::fs::File> {
    let mut options = cap_std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use cap_std::fs::OpenOptionsExt;
        options.custom_flags(rustix::fs::OFlags::NONBLOCK.bits() as i32);
    }
    let file = directory
        .open_with(path, &options)
        .map_err(|_| Error::Storage)?;
    if !file.metadata().map_err(|_| Error::Storage)?.is_file() {
        return Err(Error::Invalid);
    }
    Ok(file)
}

fn seal_with(
    bytes: &[u8],
    mut hash: impl FnMut(&starlings_protocol::Asset) -> Result<Digest>,
) -> Result<ModelManifest> {
    let value = parse_json(bytes)?;
    ModelManifest::schema()
        .validate(&value)
        .map_err(|_| Error::Invalid)?;
    let mut manifest: ModelManifest = serde_json::from_value(value)?;
    manifest.validate()?;
    if manifest.fixture {
        return Err(Error::Denied);
    }
    inventory(&manifest)?;
    for asset in &mut manifest.assets {
        asset.sha256 = hash(asset)?;
    }
    for (role, digest) in [
        (AssetRole::Config, &mut manifest.config_sha256),
        (AssetRole::Tokenizer, &mut manifest.tokenizer_sha256),
        (AssetRole::Template, &mut manifest.template_sha256),
    ] {
        *digest = manifest
            .assets
            .iter()
            .find(|a| a.role == role)
            .ok_or(Error::Invalid)?
            .sha256
            .clone();
    }
    manifest.manifest_digest =
        canonical_digest(&serde_json::to_value(&manifest)?, "manifest_digest")?;
    encode(&manifest, FixtureMode::Reject)?;
    Ok(manifest)
}

fn inventory(manifest: &ModelManifest) -> Result<()> {
    validate_repository(&manifest.source.repository)?;
    // Revisions are pinned by the existing schema; no branch/latest aliases are resolved.
    for role in [AssetRole::Config, AssetRole::Tokenizer, AssetRole::Template] {
        if manifest.assets.iter().filter(|a| a.role == role).count() != 1 {
            return Err(Error::Invalid);
        }
    }
    if !manifest.assets.iter().any(|a| a.role == AssetRole::Weights)
        || manifest
            .execution_scope
            .iter()
            .enumerate()
            .any(|(i, s)| manifest.execution_scope[..i].contains(s))
    {
        return Err(Error::Invalid);
    }
    for role in [ComponentOwnerRole::First, ComponentOwnerRole::Last] {
        if !manifest.non_layer_components.iter().any(|c| {
            c.owner_role == role
                && c.disposition == ComponentDisposition::Executed
                && !c.asset_paths.is_empty()
        }) {
            return Err(Error::Invalid);
        }
    }
    for component in &manifest.non_layer_components {
        if component.disposition == ComponentDisposition::Executed
            && component.asset_paths.is_empty()
        {
            return Err(Error::Invalid);
        }
    }
    for asset in manifest
        .assets
        .iter()
        .filter(|a| a.role == AssetRole::Weights)
    {
        if !manifest
            .layers
            .iter()
            .any(|l| l.asset_paths.contains(&asset.path))
            && !manifest
                .non_layer_components
                .iter()
                .any(|c| c.asset_paths.contains(&asset.path))
        {
            return Err(Error::Invalid);
        }
    }
    manifest.assets.iter().try_fold(0u64, |total, a| {
        total.checked_add(a.size_bytes.0).ok_or(Error::Capacity)
    })?;
    Ok(())
}

fn validate_repository(repository: &str) -> Result<()> {
    let repo: Vec<_> = repository.split('/').collect();
    if repo.len() != 2
        || repo.iter().any(|s| {
            s.is_empty()
                || matches!(*s, "." | "..")
                || !s
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        })
    {
        return Err(Error::Invalid);
    }
    Ok(())
}
