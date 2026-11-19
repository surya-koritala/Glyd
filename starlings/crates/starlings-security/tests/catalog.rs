use starlings_protocol::{FixtureMode, ModelManifest, decode, encode};
use starlings_security::{
    Error, PrivateDir,
    catalog::{Catalog, fingerprint, seal},
};
use std::io::Cursor;

fn source(path: &std::path::Path) -> (PrivateDir, Vec<u8>) {
    let dir = PrivateDir::ensure(path).unwrap();
    let mut draft: ModelManifest = decode(
        include_bytes!("../../../examples/model-manifest.json"),
        FixtureMode::Simulation,
    )
    .unwrap();
    // Synthetic storage bytes exercise production validation, never model execution.
    draft.fixture = false;
    draft.source.revision = "a".repeat(40);
    for asset in &mut draft.assets {
        let bytes = asset.path.as_bytes();
        asset.size_bytes.0 = bytes.len() as u64;
        dir.write_atomic_durable(&asset.path, bytes).unwrap();
    }
    (dir, serde_json::to_vec(&draft).unwrap())
}

#[test]
fn exact_identity_approval_and_restart_are_distinct() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let (dir, draft) = source(&root.join("source"));
    let manifest = seal(&draft, &dir).unwrap();
    let bytes = encode(&manifest, FixtureMode::Reject).unwrap();
    let mut catalog = Catalog::open(&root.join("catalog")).unwrap();
    assert!(Catalog::open(&root.join("catalog")).is_err());
    catalog.register(&bytes).unwrap();
    assert!(
        catalog
            .authorized(&manifest.manifest_digest, false)
            .is_err()
    );
    catalog.approve(&manifest.manifest_digest, false).unwrap();
    assert_eq!(
        catalog
            .authorized(&manifest.manifest_digest, false)
            .unwrap(),
        manifest
    );
    assert_eq!(
        catalog.approve(&manifest.manifest_digest, true),
        Err(Error::Denied)
    );
    assert!(catalog.authorized(&manifest.manifest_digest, true).is_err());
    // Re-registering immutable metadata must not change its independent approval.
    catalog.register(&bytes).unwrap();
    drop(catalog);
    let mut catalog = Catalog::open(&root.join("catalog")).unwrap();
    catalog
        .authorized(&manifest.manifest_digest, false)
        .unwrap();
    catalog.revoke(&manifest.manifest_digest).unwrap();
    assert!(
        catalog
            .authorized(&manifest.manifest_digest, false)
            .is_err()
    );
    assert_eq!(
        catalog.manifest(&manifest.manifest_digest).unwrap(),
        manifest
    );
    for name in [
        "weights.safetensors",
        "config.json",
        "tokenizer.json",
        "template.txt",
    ] {
        let old = name.as_bytes();
        let mut new = old.to_vec();
        new[0] ^= 1;
        dir.write_atomic_durable(name, &new).unwrap();
        assert_ne!(
            seal(&draft, &dir).unwrap().manifest_digest,
            manifest.manifest_digest
        );
        dir.write_atomic_durable(name, old).unwrap();
    }
    let mut value = starlings_protocol::parse_json(&draft).unwrap();
    value["execution_profile"] = serde_json::json!("another-profile");
    assert_ne!(
        seal(&serde_json::to_vec(&value).unwrap(), &dir)
            .unwrap()
            .manifest_digest,
        manifest.manifest_digest
    );
}

#[test]
fn unsafe_incomplete_truncated_and_tampered_inputs_never_approve() {
    let temp = tempfile::tempdir().unwrap();
    let (dir, draft) = source(&temp.path().join("source"));
    let mut catalog = Catalog::open(&temp.path().join("catalog")).unwrap();
    assert!(
        catalog
            .register(include_bytes!("../../../examples/model-manifest.json"))
            .is_err()
    );
    for path in ["../secret", "a/../secret", "CON", "/absolute"] {
        let mut value = starlings_protocol::parse_json(&draft).unwrap();
        value["assets"][0]["path"] = serde_json::json!(path);
        assert!(seal(&serde_json::to_vec(&value).unwrap(), &dir).is_err());
    }
    for change in ["revision", "inventory", "size"] {
        let mut value = starlings_protocol::parse_json(&draft).unwrap();
        match change {
            "revision" => value["source"]["revision"] = serde_json::json!("main"),
            "inventory" => value["non_layer_components"] = serde_json::json!([]),
            _ => value["assets"][0]["size_bytes"] = serde_json::json!("1"),
        }
        assert!(seal(&serde_json::to_vec(&value).unwrap(), &dir).is_err());
    }
    for length in [0, 2, 4, u64::MAX] {
        assert!(fingerprint(Cursor::new(b"abc"), length).is_err());
    }
    assert_eq!(
        fingerprint(Cursor::new(b"abc"), 3).unwrap().as_str(),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    struct ExhaustedStorage;
    impl std::io::Read for ExhaustedStorage {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::ErrorKind::StorageFull.into())
        }
    }
    for size in [0, 1] {
        assert_eq!(fingerprint(ExhaustedStorage, size), Err(Error::Storage));
    }
    let manifest = seal(&draft, &dir).unwrap();
    let mut value = serde_json::to_value(&manifest).unwrap();
    value["is_lossy"] = serde_json::json!(true);
    assert!(
        catalog
            .register(&serde_json::to_vec(&value).unwrap())
            .is_err()
    );
}

#[cfg(unix)]
#[test]
fn source_symlinks_are_rejected_before_hashing() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("source");
    let (dir, draft) = source(&path);
    std::fs::remove_file(path.join("config.json")).unwrap();
    std::os::unix::fs::symlink("tokenizer.json", path.join("config.json")).unwrap();
    assert!(seal(&draft, &dir).is_err());
}

#[test]
fn interrupted_and_corrupt_imports_resume_without_publishing_partial_content() {
    use std::io::Read;
    let temp = tempfile::tempdir().unwrap();
    let (dir, draft) = source(&temp.path().join("source"));
    let manifest = seal(&draft, &dir).unwrap();
    let mut catalog = Catalog::open(&temp.path().join("cache")).unwrap();
    catalog
        .register(&encode(&manifest, FixtureMode::Reject).unwrap())
        .unwrap();
    let id = &manifest.manifest_digest;
    catalog.approve(id, false).unwrap();
    let asset = &manifest.assets[3];
    let bytes = asset.path.as_bytes();
    assert!(
        catalog
            .import_asset(id, &asset.path, 0, Cursor::new(&bytes[..4]))
            .is_err()
    );
    assert_eq!(catalog.resume_offset(id, &asset.path).unwrap(), 4);
    assert!(catalog.open_asset(id, &asset.path).is_err());
    assert!(catalog.ready(id).is_err());
    assert!(
        catalog
            .import_asset(id, &asset.path, 3, Cursor::new(&bytes[3..]))
            .is_err()
    );
    catalog
        .import_asset(id, &asset.path, 4, Cursor::new(&bytes[4..]))
        .unwrap();
    let mut actual = Vec::new();
    catalog
        .open_asset(id, &asset.path)
        .unwrap()
        .read_to_end(&mut actual)
        .unwrap();
    assert_eq!(actual, bytes);
    let used = catalog.storage_bytes().unwrap();
    // A verified object deduplicates without reading a duplicate source stream.
    struct Unread;
    impl Read for Unread {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            panic!("duplicate read")
        }
    }
    catalog.import_asset(id, &asset.path, 0, Unread).unwrap();
    assert_eq!(catalog.storage_bytes().unwrap(), used);
    let corrupt = &manifest.assets[0];
    assert!(
        catalog
            .import_asset(
                id,
                &corrupt.path,
                0,
                Cursor::new(vec![0; corrupt.size_bytes.0 as usize])
            )
            .is_err()
    );
    assert_eq!(catalog.resume_offset(id, &corrupt.path).unwrap(), 0);
    assert!(catalog.open_asset(id, &corrupt.path).is_err());
    assert!(
        catalog
            .import_asset(id, "../escape", 0, Cursor::new(b"anything"))
            .is_err()
    );
}

#[test]
fn pins_survive_restart_revocation_quota_pressure_and_orphan_cleanup() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let (source, draft) = source(&root.join("source"));
    let manifest = seal(&draft, &source).unwrap();
    drop(source);
    let id = &manifest.manifest_digest;
    let cache = root.join("cache");
    let mut catalog = Catalog::open(&cache).unwrap();
    catalog
        .register(&encode(&manifest, FixtureMode::Reject).unwrap())
        .unwrap();
    catalog.approve(id, false).unwrap();
    let cost = catalog.storage_cost(id, 0, 1).unwrap();
    // Both layers share one storage shard, so a one-layer range still needs its full file.
    assert_eq!(
        cost.selected_storage_bytes,
        cost.whole_manifest_storage_bytes
    );
    assert_eq!(
        cost.whole_manifest_storage_bytes.0,
        manifest.assets.iter().map(|a| a.size_bytes.0).sum::<u64>()
    );
    assert!(cost.backend_peak_host_bytes.is_none());
    assert_eq!(
        cost.max_asset_import_reserve_bytes.0,
        2 * manifest
            .assets
            .iter()
            .map(|a| a.size_bytes.0)
            .max()
            .unwrap()
    );
    assert!(catalog.storage_cost(id, 1, 1).is_err());
    let baseline = catalog.storage_bytes().unwrap();
    catalog.set_quota(baseline + 1).unwrap();
    assert_eq!(
        catalog.import_directory(id, &root.join("source")),
        Err(Error::Capacity)
    );
    assert!(catalog.ready(id).is_err());
    catalog.set_quota(1024 * 1024).unwrap();
    catalog.import_directory(id, &root.join("source")).unwrap();
    let lease = "00000000-0000-4000-8000-000000000992".parse().unwrap();
    catalog.pin(id, lease).unwrap();
    catalog.pin(id, lease).unwrap();
    drop(catalog);
    PrivateDir::open(&cache)
        .unwrap()
        .write_atomic_durable(".private-fs-orphan.tmp", b"orphan")
        .unwrap();
    let mut catalog = Catalog::open(&cache).unwrap();
    assert_eq!(catalog.gc().unwrap(), 1);
    catalog.ready(id).unwrap();
    catalog.revoke(id).unwrap();
    assert_eq!(catalog.gc().unwrap(), 0);
    assert!(catalog.ready(id).is_err());
    catalog.approve(id, false).unwrap();
    catalog.ready(id).unwrap();
    catalog.unpin(lease).unwrap();
    assert_eq!(catalog.gc().unwrap(), manifest.assets.len() as u64);
    assert!(catalog.ready(id).is_err());
}

#[cfg(unix)]
#[test]
fn failed_object_publication_preserves_pinned_content_and_can_be_retried() {
    use std::{io::Read, os::unix::fs::PermissionsExt};
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let source_path = root.join("source");
    let (source, draft) = source(&source_path);
    let original = seal(&draft, &source).unwrap();
    let cache = root.join("cache");
    let mut catalog = Catalog::open(&cache).unwrap();
    let id = &original.manifest_digest;
    catalog
        .register(&encode(&original, FixtureMode::Reject).unwrap())
        .unwrap();
    catalog.approve(id, false).unwrap();
    catalog.import_directory(id, &source_path).unwrap();
    let lease = "00000000-0000-4000-8000-000000000994".parse().unwrap();
    catalog.pin(id, lease).unwrap();

    let weights = original
        .assets
        .iter()
        .find(|a| a.role == starlings_protocol::AssetRole::Weights)
        .unwrap();
    let changed = vec![b'z'; weights.size_bytes.0 as usize];
    source
        .write_atomic_durable(&weights.path, &changed)
        .unwrap();
    let next = seal(&draft, &source).unwrap();
    catalog
        .register(&encode(&next, FixtureMode::Reject).unwrap())
        .unwrap();
    catalog.approve(&next.manifest_digest, false).unwrap();

    struct RefusePublication<'a> {
        data: Cursor<&'a [u8]>,
        cache: &'a std::path::Path,
    }
    impl Read for RefusePublication<'_> {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            // Trigger a real namespace/write failure after the staging file opens.
            std::fs::set_permissions(self.cache, std::fs::Permissions::from_mode(0o500))?;
            self.data.read(buffer)
        }
    }
    let failed = catalog.import_asset(
        &next.manifest_digest,
        &weights.path,
        0,
        RefusePublication {
            data: Cursor::new(&changed),
            cache: &cache,
        },
    );
    std::fs::set_permissions(&cache, std::fs::Permissions::from_mode(0o700)).unwrap();
    assert_eq!(failed, Err(Error::Storage));
    assert!(catalog.ready(&next.manifest_digest).is_err());
    catalog.ready(id).unwrap();
    drop(catalog);

    let mut catalog = Catalog::open(&cache).unwrap();
    catalog.gc().unwrap();
    catalog.ready(id).unwrap();
    assert!(catalog.ready(&next.manifest_digest).is_err());
    catalog
        .import_directory(&next.manifest_digest, &source_path)
        .unwrap();
    catalog.ready(&next.manifest_digest).unwrap();
    // Only the new, unpinned object is collectible; the original remains usable.
    assert_eq!(catalog.gc().unwrap(), 1);
    catalog.ready(id).unwrap();
}

#[cfg(unix)]
#[test]
fn preloaded_import_is_read_only_and_cannot_follow_an_escaping_symlink() {
    use std::os::unix::fs::PermissionsExt;
    let temp = tempfile::tempdir().unwrap();
    let source_path = temp.path().join("source");
    let (dir, draft) = source(&source_path);
    let manifest = seal(&draft, &dir).unwrap();
    let mut catalog = Catalog::open(&temp.path().join("cache")).unwrap();
    catalog
        .register(&encode(&manifest, FixtureMode::Reject).unwrap())
        .unwrap();
    catalog.approve(&manifest.manifest_digest, false).unwrap();
    for a in &manifest.assets {
        std::fs::set_permissions(
            source_path.join(&a.path),
            std::fs::Permissions::from_mode(0o400),
        )
        .unwrap();
    }
    std::fs::set_permissions(&source_path, std::fs::Permissions::from_mode(0o500)).unwrap();
    let result = catalog.import_directory(&manifest.manifest_digest, &source_path);
    assert_eq!(
        std::fs::metadata(&source_path)
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o500
    );
    std::fs::set_permissions(&source_path, std::fs::Permissions::from_mode(0o700)).unwrap();
    result.unwrap();
    let outside = temp.path().join("outside");
    std::fs::write(&outside, b"config.json").unwrap();
    std::fs::remove_file(source_path.join("config.json")).unwrap();
    std::os::unix::fs::symlink("../outside", source_path.join("config.json")).unwrap();
    assert!(starlings_security::catalog::seal_preloaded(&draft, &source_path).is_err());
}

#[test]
fn service_paths_and_quarantine_preserve_approved_objects_across_restart() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let (source, draft) = source(&root.join("source"));
    let manifest = seal(&draft, &source).unwrap();
    drop(source);
    let id = &manifest.manifest_digest;
    let cache = root.join("cache");
    let mut catalog = Catalog::open(&cache).unwrap();
    catalog
        .register(&encode(&manifest, FixtureMode::Reject).unwrap())
        .unwrap();
    assert!(catalog.asset_path(id, "config.json").is_err());
    catalog.approve(id, false).unwrap();
    catalog.import_directory(id, &root.join("source")).unwrap();
    let path = catalog.asset_path(id, "config.json").unwrap();
    assert!(path.is_absolute());
    assert_eq!(std::fs::read(&path).unwrap(), b"config.json");
    assert!(catalog.asset_path(id, "../config.json").is_err());
    PrivateDir::open(&cache)
        .unwrap()
        .write_atomic_durable(path.file_name().unwrap(), b"corrupt")
        .unwrap();
    assert!(catalog.asset_path(id, "config.json").is_err());
    catalog.import_directory(id, &root.join("source")).unwrap();
    let lease = "00000000-0000-4000-8000-000000000993".parse().unwrap();
    catalog.begin_service(id, lease, id).unwrap();
    assert_eq!(catalog.unpin(lease), Err(Error::Denied));
    assert_eq!(catalog.begin_service(id, lease, id), Err(Error::Denied));
    drop(catalog);
    // A missing ordinary pin record must not erase the durable launch ownership.
    std::fs::remove_file(cache.join(format!("pin-{lease}.json"))).unwrap();
    let mut catalog = Catalog::open(&cache).unwrap();
    assert_eq!(catalog.ensure_no_service(), Err(Error::Denied));
    catalog.revoke(id).unwrap();
    assert_eq!(catalog.gc().unwrap(), 0);
    assert!(path.exists());
}

#[test]
fn cleanup_receipt_binds_launch_and_recovery_preserves_pins_until_commit() {
    use starlings_security::catalog::ServicePhase;
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let (source, draft) = source(&root.join("source"));
    let manifest = seal(&draft, &source).unwrap();
    drop(source);
    let cache = root.join("cache");
    let id = &manifest.manifest_digest;
    let lease = "00000000-0000-4000-8000-000000000995".parse().unwrap();
    let other = "00000000-0000-4000-8000-000000000996".parse().unwrap();
    let config = starlings_protocol::byte_digest(b"synthetic service config; no workload");
    let qualification =
        starlings_protocol::byte_digest(b"synthetic cleanup qualification; not native proof");
    let wrong = starlings_protocol::byte_digest(b"unrelated qualification/configuration");
    let mut catalog = Catalog::open(&cache).unwrap();
    catalog
        .register(&encode(&manifest, FixtureMode::Reject).unwrap())
        .unwrap();
    catalog.approve(id, false).unwrap();
    catalog.import_directory(id, &root.join("source")).unwrap();
    catalog
        .begin_service_qualified(id, lease, &config, Some(qualification.clone()))
        .unwrap();
    assert!(catalog.recover_service(lease, &config).is_err());
    assert!(
        catalog
            .record_service_drained(other, &config, &qualification)
            .is_err()
    );
    assert!(
        catalog
            .record_service_drained(lease, &wrong, &qualification)
            .is_err()
    );
    assert!(
        catalog
            .record_service_drained(lease, &config, &wrong)
            .is_err()
    );
    assert_eq!(catalog.gc().unwrap(), 0);
    catalog
        .record_service_drained(lease, &config, &qualification)
        .unwrap();
    assert_eq!(
        catalog.service_record().unwrap().unwrap().phase,
        ServicePhase::Drained
    );
    assert!(catalog.ensure_no_service().is_err());
    drop(catalog);
    // Interruption after unpin must remain protected by the durable drained marker.
    std::fs::remove_file(cache.join(format!("pin-{lease}.json"))).unwrap();
    let mut catalog = Catalog::open(&cache).unwrap();
    assert_eq!(catalog.gc().unwrap(), 0);
    assert!(catalog.recover_service(other, &config).is_err());
    #[cfg(unix)]
    {
        assert!(!catalog.recover_service(lease, &config).unwrap());
        assert!(catalog.recover_service(lease, &config).unwrap());
        catalog.ensure_no_service().unwrap();
        assert_eq!(
            catalog.service_record().unwrap().unwrap().phase,
            ServicePhase::Recovered
        );
        assert!(
            catalog.begin_service(id, lease, &config).is_err(),
            "launch IDs must be fresh"
        );
        catalog.begin_service(id, other, &config).unwrap();
        assert!(
            catalog.recover_service(lease, &config).is_err(),
            "old cleanup cannot clear a new launch"
        );
    }
    #[cfg(not(unix))]
    assert!(catalog.recover_service(lease, &config).is_err());
}

#[cfg(unix)]
#[test]
fn failed_recovery_write_and_legacy_quarantine_never_release_objects() {
    use starlings_security::catalog::ServicePhase;
    use std::os::unix::fs::PermissionsExt;
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let (source, draft) = source(&root.join("source"));
    let manifest = seal(&draft, &source).unwrap();
    let cache = root.join("cache");
    let id = &manifest.manifest_digest;
    let lease = "00000000-0000-4000-8000-000000000997".parse().unwrap();
    let qualification = starlings_protocol::byte_digest(b"synthetic cleanup policy");
    let mut catalog = Catalog::open(&cache).unwrap();
    catalog
        .register(&encode(&manifest, FixtureMode::Reject).unwrap())
        .unwrap();
    catalog.approve(id, false).unwrap();
    catalog.import_directory(id, &root.join("source")).unwrap();
    catalog
        .begin_service_qualified(id, lease, id, Some(qualification.clone()))
        .unwrap();
    catalog
        .record_service_drained(lease, id, &qualification)
        .unwrap();
    std::fs::set_permissions(&cache, std::fs::Permissions::from_mode(0o500)).unwrap();
    let failed = catalog.recover_service(lease, id);
    std::fs::set_permissions(&cache, std::fs::Permissions::from_mode(0o700)).unwrap();
    assert!(failed.is_err());
    assert_eq!(
        catalog.service_record().unwrap().unwrap().phase,
        ServicePhase::Drained
    );
    assert_eq!(catalog.gc().unwrap(), 0);
    catalog.recover_service(lease, id).unwrap();
    let legacy = serde_json::json!({"version":1,"lease":lease,"manifest":id,"configuration":id});
    PrivateDir::open(&cache)
        .unwrap()
        .write_atomic_durable("service-start.json", &serde_json::to_vec(&legacy).unwrap())
        .unwrap();
    assert!(
        catalog
            .record_service_drained(lease, id, &qualification)
            .is_err()
    );
    assert!(catalog.recover_service(lease, id).is_err());
    assert_eq!(catalog.gc().unwrap(), 0);
}

#[cfg(unix)]
#[test]
fn service_path_rejects_a_renamed_catalog_and_replacement_directory() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let (source, draft) = source(&root.join("source"));
    let manifest = seal(&draft, &source).unwrap();
    let cache = root.join("cache");
    let mut catalog = Catalog::open(&cache).unwrap();
    catalog
        .register(&encode(&manifest, FixtureMode::Reject).unwrap())
        .unwrap();
    catalog.approve(&manifest.manifest_digest, false).unwrap();
    catalog
        .import_directory(&manifest.manifest_digest, &root.join("source"))
        .unwrap();
    std::fs::rename(&cache, root.join("moved")).unwrap();
    PrivateDir::ensure(&cache).unwrap();
    assert_eq!(
        catalog.asset_path(&manifest.manifest_digest, "config.json"),
        Err(Error::Denied)
    );
}
