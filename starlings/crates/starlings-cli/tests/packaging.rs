#[cfg(unix)]
#[test]
fn candidate_archive_fixture_preserves_hashes_and_refuses_unsafe_outputs() {
    let scripts = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts");
    let result = std::process::Command::new("bash")
        .arg(scripts.join("build-candidate.sh"))
        .arg("--self-test")
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    assert!(
        String::from_utf8(result.stdout)
            .unwrap()
            .ends_with("candidate archive checks passed\n")
    );
    let result = std::process::Command::new("python3")
        .arg("-I")
        .arg(scripts.join("verify-candidate.py"))
        .arg("--self-test")
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    assert_eq!(result.stdout, b"candidate verifier checks passed\n");
}
