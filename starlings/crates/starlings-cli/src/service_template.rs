//! Export inactive per-user descriptors; never invoke a service manager.
use crate::cluster::{Paths, Result};
use std::path::Path;

#[cfg(any(target_os = "linux", target_os = "macos"))]
use {
    crate::cluster::{Config, absolute, private_parent},
    starlings_security::PrivateDir,
    std::io::Write,
};

const REFUSED: &str = "SERVICE_TEMPLATE_REFUSED: check private paths and supported platform";

#[cfg(any(target_os = "linux", target_os = "macos"))]
pub fn run(paths: &Paths, output: &Path) -> Result<()> {
    export(paths, output)?;
    writeln!(
        std::io::stdout().lock(),
        "SERVICE_TEMPLATE: installed=false started=false"
    )
    .map_err(|_| "OUTPUT_UNAVAILABLE: descriptor was exported".into())
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub fn run(_: &Paths, _: &Path) -> Result<()> {
    Err(REFUSED.into())
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn export(paths: &Paths, output: &Path) -> Result<()> {
    absolute(&paths.state)?;
    let profile = PrivateDir::open(&paths.state).map_err(|_| REFUSED)?;
    let config = Config::load(&profile)?;
    let state = paths.state.canonicalize().map_err(|_| REFUSED)?;
    let executable = std::env::current_exe()
        .and_then(|p| p.canonicalize())
        .map_err(|_| REFUSED)?;
    let args = [
        path_text(&executable)?,
        "--state-dir",
        path_text(&state)?,
        "daemon",
    ];
    let label = format!("com.getglyd.compute.{}", config.node);
    #[cfg(target_os = "linux")]
    let (name, bytes) = (format!("{label}.service"), systemd(&args).into_bytes());
    #[cfg(target_os = "macos")]
    let (name, bytes) = (format!("{label}.plist"), launch_agent(&label, &args)?);
    let (parent, leaf) = private_parent(output)?;
    let directory = parent.create_dir(leaf).map_err(|_| REFUSED)?;
    let mut file = directory.create(name).map_err(|_| REFUSED)?;
    file.write_all(&bytes).map_err(|_| REFUSED)?;
    file.sync_all().map_err(|_| REFUSED.into())
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn path_text(path: &Path) -> Result<&str> {
    absolute(path)?;
    let text = path.to_str().ok_or(REFUSED)?;
    if text.len() > 4096 || text.chars().any(char::is_control) {
        return Err(REFUSED.into());
    }
    Ok(text)
}

#[cfg(any(target_os = "linux", all(test, target_os = "macos")))]
fn systemd(args: &[&str; 4]) -> String {
    let words: Vec<_> = args
        .iter()
        .map(|arg| {
            format!(
                "\"{}\"",
                arg.replace('\\', "\\\\")
                    .replace('"', "\\\"")
                    .replace('%', "%%")
            )
        })
        .collect();
    format!(
        "[Unit]\nDescription=Starlings control plane\n\n[Service]\nType=exec\nExecStart=:{}\nUMask=0077\nNoNewPrivileges=yes\nRestart=no\n",
        words.join(" ")
    )
}

#[cfg(target_os = "macos")]
fn launch_agent(label: &str, args: &[&str; 4]) -> Result<Vec<u8>> {
    use std::process::{Command, Stdio};
    let json = serde_json::to_vec(&serde_json::json!({
        "Label": label, "ProgramArguments": args,
        "KeepAlive": false, "RunAtLoad": false, "Umask": 63,
    }))
    .map_err(|_| REFUSED)?;
    let mut child = Command::new("/usr/bin/plutil")
        .args(["-convert", "xml1", "-o", "-", "--", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|_| REFUSED)?;
    let written = child.stdin.take().ok_or(REFUSED)?.write_all(&json);
    let output = child.wait_with_output().map_err(|_| REFUSED)?;
    written.map_err(|_| REFUSED)?;
    if !output.status.success() || output.stdout.len() > 64 * 1024 {
        return Err(REFUSED.into());
    }
    Ok(output.stdout)
}

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
mod tests {
    use super::*;
    use starlings_protocol::Version;
    use starlings_security::Role;
    use std::{
        fs,
        os::unix::fs::{PermissionsExt, symlink},
    };

    #[test]
    fn descriptors_preserve_literal_paths_and_refuse_existing_or_unsafe_outputs() {
        let awkward = "/space \\\"'$HOME%h<&/é";
        let unit = systemd(&["/bin/starlings", "--state-dir", awkward, "daemon"]);
        assert!(unit.contains("ExecStart=:\"/bin/starlings\" \"--state-dir\" \"/space \\\\\\\"'$HOME%%h<&/é\" \"daemon\"\n"));
        assert!(unit.ends_with("UMask=0077\nNoNewPrivileges=yes\nRestart=no\n"));
        assert!(!unit.contains("[Install]"));
        for invalid in ["relative", "/line\nbreak", "/null\0byte", "/dot/../segment"] {
            assert!(path_text(Path::new(invalid)).is_err());
        }
        assert!(path_text(Path::new(&format!("/{}", "x".repeat(4096)))).is_err());
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap().join("private");
        PrivateDir::ensure(&root).unwrap();
        let state = root.join("profile $HOME%h\"\\é");
        let profile = PrivateDir::ensure(&state).unwrap();
        let config = Config {
            version: Version,
            role: Role::Worker,
            cluster: "a53837fb-96ea-4e52-bfb9-b86bde501c53".parse().unwrap(),
            node: "026f0b1b-15bd-4c31-9972-2f3881b3c599".parse().unwrap(),
            listen: None,
            cache_dir: root.join("cache"),
            additional_listen: vec![],
            routes: vec![],
            source: None,
        };
        config.save(&profile).unwrap();
        let original = fs::read(state.join("config.json")).unwrap();
        let paths = Paths {
            state,
            cache: root.join("unused"),
        };
        let output = root.join("inactive");
        export(&paths, &output).unwrap();
        let entries: Vec<_> = fs::read_dir(&output)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        assert_eq!(entries.len(), 1);
        assert_eq!(
            fs::metadata(&output).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            fs::metadata(&entries[0]).unwrap().permissions().mode() & 0o777,
            0o600
        );
        let bytes = fs::read(&entries[0]).unwrap();
        #[cfg(target_os = "linux")]
        {
            let runtime = root.join("native-runtime");
            let generators = root.join("empty-generators");
            PrivateDir::ensure(&runtime).unwrap();
            PrivateDir::ensure(&generators).unwrap();
            let verify = |path: &Path| {
                std::process::Command::new("/usr/bin/systemd-analyze")
                    .args([
                        "--user",
                        "--man=no",
                        "--generators=no",
                        "--recursive-errors=yes",
                        "verify",
                    ])
                    .env("XDG_RUNTIME_DIR", &runtime)
                    .env("SYSTEMD_ENVIRONMENT_GENERATOR_PATH", &generators)
                    .arg(path)
                    .output()
                    .expect("Linux CLI checks require systemd-analyze >=250")
            };
            let valid = verify(&entries[0]);
            assert!(
                valid.status.success() && valid.stdout.is_empty() && valid.stderr.is_empty(),
                "native unit verification: {valid:?}"
            );
            let invalid = root.join("invalid.service");
            let mut malformed = bytes.clone();
            malformed.extend_from_slice(b"GlydInvalidSetting=yes\n");
            fs::write(&invalid, malformed).unwrap();
            let refused = verify(&invalid);
            assert!(
                !refused.status.success(),
                "native parser accepted an unknown directive"
            );
            assert!(
                String::from_utf8_lossy(&refused.stderr).contains("GlydInvalidSetting"),
                "unexpected native refusal: {refused:?}"
            );
        }
        #[cfg(target_os = "macos")]
        {
            let parsed = std::process::Command::new("/usr/bin/plutil")
                .args(["-convert", "json", "-o", "-", "--"])
                .arg(&entries[0])
                .output()
                .unwrap();
            assert!(parsed.status.success());
            let value: serde_json::Value = serde_json::from_slice(&parsed.stdout).unwrap();
            assert_eq!(
                value["ProgramArguments"][0],
                std::env::current_exe()
                    .unwrap()
                    .canonicalize()
                    .unwrap()
                    .to_str()
                    .unwrap()
            );
            assert_eq!(value["ProgramArguments"][1], "--state-dir");
            assert_eq!(value["ProgramArguments"][2], paths.state.to_str().unwrap());
            assert_eq!(value["ProgramArguments"][3], "daemon");
            assert_eq!(value["ProgramArguments"].as_array().unwrap().len(), 4);
            assert_eq!(
                value["Label"],
                format!("com.getglyd.compute.{}", config.node)
            );
            assert_eq!(value["Umask"], 63);
            assert_eq!(value["KeepAlive"], false);
            assert_eq!(value["RunAtLoad"], false);
        }
        assert!(export(&paths, &output).is_err());
        assert_eq!(fs::read(&entries[0]).unwrap(), bytes);
        let linked = root.join("linked");
        symlink(&output, &linked).unwrap();
        assert!(export(&paths, &linked).is_err());
        assert!(export(&paths, &linked.join("child")).is_err());
        let relative = Path::new("relative-output");
        assert!(export(&paths, relative).is_err());
        assert_eq!(fs::read(paths.state.join("config.json")).unwrap(), original);
        assert!(!paths.cache.exists());
        assert_eq!(fs::read_dir(&paths.state).unwrap().count(), 1);
    }
}
