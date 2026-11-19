//! Regression checks for the M0/M1 review findings.
use crate::enrollment::{
    certificate_params,
    tests::{enrolled, setup},
};
use crate::*;
#[cfg(unix)]
use std::io::Write;
use std::time::Instant;

#[test]
fn authorization_is_read_only_under_database_contention() {
    let (_temp, path, authority, _) = setup();
    let connection = rusqlite::Connection::open(path.join("enrollment.sqlite")).unwrap();
    connection
        .execute_batch("BEGIN; SELECT * FROM clock;")
        .unwrap();
    let before = authority.access.lock().unwrap().database.total_changes();
    for _ in 0..1000 {
        authority
            .access
            .lock()
            .unwrap()
            .credential(&authority.identity.member)
            .unwrap();
    }
    assert_eq!(
        authority.access.lock().unwrap().database.total_changes(),
        before
    );
    // Force an actual busy checkpoint; existing credentials remain valid afterwards.
    let start = Instant::now();
    authority.access.checkpoint(true).unwrap();
    assert!(start.elapsed() < std::time::Duration::from_secs(1));
    authority
        .access
        .lock()
        .unwrap()
        .credential(&authority.identity.member)
        .unwrap();
    connection.execute_batch("ROLLBACK").unwrap();
}

#[test]
fn expired_coordinator_recovers_locally_but_expired_peer_tls_is_refused() {
    let (_temp, path, mut authority, files) = setup();
    let ca = std::fs::read(path.join("authority.json")).unwrap();
    let mut device = enrolled(&mut authority, &files, Role::Worker, "device");
    let age = |identity: &mut Identity, authority: &Authority| {
        let mut params = certificate_params(&identity.member).unwrap();
        params.not_before = time::OffsetDateTime::now_utc() - time::Duration::seconds(30);
        params.not_after = time::OffsetDateTime::now_utc() - time::Duration::seconds(1);
        identity.member.certificate = params
            .signed_by(&identity.key, &authority.issuer)
            .unwrap()
            .der()
            .to_vec();
    };
    age(&mut device, &authority);
    assert!(
        crate::transport::verify_issued(
            &device.member,
            &authority.trust().unwrap().ca,
            &device.key
        )
        .is_err()
    );
    let mut coordinator = Identity::load(&authority.directory, "coordinator.json").unwrap();
    age(&mut coordinator, &authority);
    coordinator
        .replace_file(&authority.directory, "coordinator.json")
        .unwrap();
    authority
        .access
        .lock()
        .unwrap()
        .database
        .execute(
            "UPDATE members SET record=?2 WHERE node=?1",
            (
                coordinator.node().to_string(),
                serde_json::to_vec(&coordinator.member).unwrap(),
            ),
        )
        .unwrap();
    let pin = coordinator.request_fingerprint();
    drop(authority);
    let recovered = Authority::open(&path).unwrap();
    assert_eq!(recovered.identity().request_fingerprint(), pin);
    assert!(
        recovered
            .identity()
            .certificate_seconds_remaining()
            .unwrap()
            > 6 * 86400
    );
    assert_eq!(std::fs::read(path.join("authority.json")).unwrap(), ca);
}

#[test]
fn drain_refuses_library_enrollment_and_invites() {
    let (_temp, _, mut authority, files) = setup();
    authority.drain().unwrap();
    assert_eq!(
        authority.invite(
            &files,
            "refused",
            "127.0.0.1:9741".parse().unwrap(),
            Role::Worker
        ),
        Err(Error::Denied)
    );
}

#[cfg(unix)]
#[test]
fn cleanup_uses_original_directory_after_profile_rename() {
    let temp = tempfile::tempdir().unwrap();
    let original = temp.path().join("profile");
    let moved = temp.path().join("moved");
    let lock = StateLock::acquire(&original).unwrap();
    lock.directory()
        .create("admin.cap")
        .unwrap()
        .write_all(b"first")
        .unwrap();
    std::fs::rename(&original, &moved).unwrap();
    let replacement = PrivateDir::ensure(&original).unwrap();
    replacement
        .create("admin.cap")
        .unwrap()
        .write_all(b"second")
        .unwrap();
    lock.cleanup_admin().unwrap();
    assert!(!moved.join("admin.cap").exists());
    assert_eq!(
        std::fs::read(original.join("admin.cap")).unwrap(),
        b"second"
    );
}

#[test]
fn private_state_tampering_and_invitation_caps_are_refused() {
    let (_temp, path, mut authority, files) = setup();
    authority
        .invite(
            &files,
            "first",
            "127.0.0.1:9741".parse().unwrap(),
            Role::Worker,
        )
        .unwrap();
    {
        // Seed occupied slots in one transaction; repeated durable file creation
        // adds no coverage to this capacity boundary (and is costly on Windows).
        let state = authority.access.lock().unwrap();
        let template: String = state
            .database
            .query_row("SELECT hash FROM invitations", [], |r| r.get(0))
            .unwrap();
        let transaction = state.database.unchecked_transaction().unwrap();
        for n in 0..126 {
            transaction
                .execute(
                    "INSERT INTO invitations SELECT ?1,expires,role FROM invitations WHERE hash=?2",
                    (format!("{n:064x}"), &template),
                )
                .unwrap();
        }
        transaction.commit().unwrap();
    }
    authority
        .invite(
            &files,
            "last",
            "127.0.0.1:9741".parse().unwrap(),
            Role::Worker,
        )
        .unwrap();
    assert_eq!(
        authority.invite(
            &files,
            "overflow",
            "127.0.0.1:9741".parse().unwrap(),
            Role::Worker
        ),
        Err(Error::Capacity)
    );
    authority
        .access
        .lock()
        .unwrap()
        .database
        .execute(
            "INSERT INTO settings VALUES ('coordinator','../outside')",
            [],
        )
        .unwrap();
    drop(authority);
    assert!(Authority::open(&path).is_err());
}
