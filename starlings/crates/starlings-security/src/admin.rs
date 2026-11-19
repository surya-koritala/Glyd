//! Local administration uses native IPC plus a private, per-listener capability.
use crate::{
    Error, PrivateDir, Result,
    client::remaining,
    listener::{ListenerBudget, Permit, REQUEST_TIME},
};
use interprocess::local_socket::{Listener, ListenerOptions, Stream, prelude::*};
use std::{
    io::{Read, Write},
    net::IpAddr,
    path::Path,
    time::{Duration, Instant},
};

pub struct AdminListener {
    listener: Listener,
    directory: PrivateDir,
    capability: [u8; 32],
    budget: ListenerBudget,
    #[cfg(unix)]
    owner: u32,
}
pub struct AdminRequest {
    io: AdminIo,
    capability: [u8; 32],
    capability_file: private_fs::PrivateFile,
}
struct AdminIo {
    stream: Stream,
    _permit: Option<Permit>,
    deadline: Instant,
}
impl AdminListener {
    /// Requires an existing protected directory and an unused endpoint/capability name.
    /// Existing names are never replaced; restart cleanup belongs to the daemon lifecycle.
    pub fn bind(path: &Path, budget: ListenerBudget) -> Result<Self> {
        let directory = PrivateDir::open(path)?;
        #[cfg(unix)]
        let path = path.canonicalize()?;
        let mut capability = [0; 32];
        ring::rand::SecureRandom::fill(&ring::rand::SystemRandom::new(), &mut capability)
            .map_err(|_| Error::Crypto)?;

        #[cfg(unix)]
        let listener = {
            use interprocess::local_socket::GenericFilePath;
            use std::os::unix::fs::PermissionsExt;
            let endpoint = path.join("admin.sock");
            let listener = ListenerOptions::new()
                .name(endpoint.as_path().to_fs_name::<GenericFilePath>()?)
                .reclaim_name(false)
                .create_sync()?;
            // The 0700 parent protects the socket before chmod, including on macOS.
            std::fs::set_permissions(&endpoint, std::fs::Permissions::from_mode(0o600))?;
            listener
        };
        #[cfg(windows)]
        let listener = windows_listener(&capability, "D:P(A;;GA;;;OW)")?;
        let mut file = directory.create("admin.cap")?;
        file.write_all(&capability)?;
        file.sync_all()?;
        listener.set_nonblocking(interprocess::local_socket::ListenerNonblockingMode::Accept)?;
        Ok(Self {
            listener,
            directory,
            capability,
            budget,
            #[cfg(unix)]
            owner: {
                use std::os::unix::fs::MetadataExt;
                std::fs::metadata(&path)?.uid()
            },
        })
    }
    pub fn try_accept(&self) -> Result<Option<AdminRequest>> {
        match self.listener.accept() {
            Ok(stream) => match self.accepted(stream) {
                Ok(request) => Ok(Some(request)),
                Err(Error::Denied | Error::Transport) => Ok(None),
                Err(error) => Err(error),
            },
            Err(e) if crate::listener::transient_accept(&e) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
    pub fn accept(&self) -> Result<AdminRequest> {
        self.directory.open_file("admin.cap")?.verify()?;
        loop {
            match self.listener.accept() {
                Ok(stream) => return self.accepted(stream),
                Err(e) if crate::listener::transient_accept(&e) => {
                    std::thread::sleep(Duration::from_millis(20))
                }
                Err(error) => return Err(error.into()),
            }
        }
    }
    fn accepted(&self, stream: Stream) -> Result<AdminRequest> {
        let capability_file = self.directory.open_file("admin.cap")?;
        let permit = self.budget.acquire(IpAddr::from([127, 0, 0, 1]), false)?;
        #[cfg(unix)]
        if stream.peer_creds()?.euid() != Some(self.owner) {
            return Err(Error::Denied);
        }
        stream.set_nonblocking(true)?;
        Ok(AdminRequest {
            io: AdminIo {
                stream,
                _permit: Some(permit),
                deadline: Instant::now() + REQUEST_TIME,
            },
            capability: self.capability,
            capability_file,
        })
    }
}
#[cfg(windows)]
fn windows_listener(capability: &[u8; 32], descriptor: &str) -> Result<Listener> {
    use interprocess::{
        local_socket::GenericNamespaced,
        os::windows::{local_socket::ListenerOptionsExt, security_descriptor::SecurityDescriptor},
    };
    let descriptor = widestring::U16CString::from_str(descriptor).map_err(|_| Error::Invalid)?;
    let descriptor = SecurityDescriptor::deserialize(&descriptor)?;
    let name = format!("starlings-{}", crate::hash(capability));
    Ok(ListenerOptions::new()
        .name(name.to_ns_name::<GenericNamespaced>()?)
        .security_descriptor(descriptor)
        .create_sync()?)
}

impl AdminRequest {
    /// Fresh client challenge, server HMAC proof, then 32 capability bytes and one
    /// length-prefixed JSON record capped at 16 KiB. Dispatch belongs to the caller.
    pub fn receive<T: serde::de::DeserializeOwned>(mut self) -> Result<T> {
        self.read_command()
    }
    /// One authenticated command and bounded reply. A client receipt prevents unbounded
    /// Windows pipe flush-on-drop workers; a lost reply is an unknown command outcome.
    pub fn respond<T: serde::de::DeserializeOwned, R: serde::Serialize>(
        mut self,
        handler: impl FnOnce(T) -> R,
    ) -> Result<()> {
        let command = self.read_command()?;
        remaining(self.io.deadline)?;
        let reply = handler(command);
        crate::send(&mut self.io, &reply)?;
        let mut ack = [0];
        self.io.read_exact(&mut ack)?;
        if ack != [1] {
            return Err(Error::Invalid);
        }
        Ok(())
    }
    fn read_command<T: serde::de::DeserializeOwned>(&mut self) -> Result<T> {
        let mut challenge = [0; 32];
        self.io.read_exact(&mut challenge)?;
        let proof = ring::hmac::sign(
            &ring::hmac::Key::new(ring::hmac::HMAC_SHA256, &self.capability),
            &challenge,
        );
        self.io.write_all(proof.as_ref())?;
        let mut capability = [0; 32];
        self.io.read_exact(&mut capability)?;
        // HMAC verification is constant-time and does not expose the capability in errors.
        let key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, &self.capability);
        let supplied = ring::hmac::sign(
            &ring::hmac::Key::new(ring::hmac::HMAC_SHA256, &capability),
            b"starlings-admin/1",
        );
        ring::hmac::verify(&key, b"starlings-admin/1", supplied.as_ref())
            .map_err(|_| Error::Denied)?;
        let record = crate::receive(&mut self.io)?;
        self.capability_file.verify()?;
        Ok(record)
    }
}
impl Read for AdminIo {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        loop {
            remaining(self.deadline)
                .map_err(|_| std::io::Error::new(std::io::ErrorKind::TimedOut, "admin deadline"))?;
            match self.stream.read(bytes) {
                // interprocess maps Windows PIPE_NOWAIT / ERROR_NO_DATA to Ok(0),
                // indistinguishable from EOF. Retain the fixed deadline on both.
                #[cfg(windows)]
                Ok(0) if !bytes.is_empty() => std::thread::sleep(Duration::from_millis(1)),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(1))
                }
                result => return result,
            }
        }
    }
}

impl Write for AdminIo {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        // PIPE_NOWAIT has a small native buffer. Bound each write, not the JSON frame.
        #[cfg(windows)]
        let bytes = &bytes[..bytes.len().min(256)];
        loop {
            remaining(self.deadline)
                .map_err(|_| std::io::Error::new(std::io::ErrorKind::TimedOut, "admin deadline"))?;
            match self.stream.write(bytes) {
                // A nonblocking byte pipe can make zero progress when its buffer is full.
                #[cfg(windows)]
                Ok(0) if !bytes.is_empty() => std::thread::sleep(Duration::from_millis(1)),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(1))
                }
                result => return result,
            }
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
impl Drop for AdminIo {
    fn drop(&mut self) {
        #[cfg(windows)]
        {
            let Stream::NamedPipe(pipe) = &self.stream;
            pipe.inner().assume_flushed();
        }
    }
}
/// Connect only to the protected local endpoint. No network or caller-supplied pipe name.
pub fn admin_call<T: serde::Serialize, R: serde::de::DeserializeOwned>(
    path: &Path,
    request: &T,
) -> Result<R> {
    let directory = PrivateDir::open(path)?;
    #[cfg(unix)]
    let path = path.canonicalize()?;
    let mut capability = [0; 32];
    let mut file = directory.open_file("admin.cap")?;
    file.read_exact(&mut capability)?;
    #[cfg(unix)]
    let name = path
        .join("admin.sock")
        .to_fs_name::<interprocess::local_socket::GenericFilePath>()?;
    #[cfg(unix)]
    let stream = interprocess::local_socket::ConnectOptions::new()
        .name(name)
        .wait_mode(interprocess::ConnectWaitMode::Timeout(Duration::from_secs(
            1,
        )))
        .nonblocking_stream(true)
        .connect_sync()?;
    #[cfg(windows)]
    let stream = {
        use interprocess::os::windows::named_pipe::{self, DuplexPipeStream, pipe_mode::Bytes};
        // The 2.4.5 local-socket adapter ignores ConnectOptions.wait_mode on Windows.
        // Use its native bounded connector and safe conversion instead.
        let name = format!(r"\\.\pipe\starlings-{}", crate::hash(&capability));
        let pipe = DuplexPipeStream::<Bytes>::connect_by_path_with_wait_mode(
            name.as_str(),
            interprocess::ConnectWaitMode::Timeout(Duration::from_secs(1)),
        )?;
        pipe.set_nonblocking(true)?;
        Stream::from(named_pipe::local_socket::Stream::from(pipe))
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if stream.peer_creds()?.euid() != Some(std::fs::metadata(&path)?.uid()) {
            return Err(Error::Denied);
        }
    }
    let mut io = AdminIo {
        stream,
        _permit: None,
        deadline: Instant::now() + REQUEST_TIME,
    };
    let mut challenge = [0; 32];
    ring::rand::SecureRandom::fill(&ring::rand::SystemRandom::new(), &mut challenge)
        .map_err(|_| Error::Crypto)?;
    io.write_all(&challenge)?;
    let mut proof = [0; 32];
    io.read_exact(&mut proof)?;
    ring::hmac::verify(
        &ring::hmac::Key::new(ring::hmac::HMAC_SHA256, &capability),
        &challenge,
        &proof,
    )
    .map_err(|_| Error::Denied)?;
    io.write_all(&capability)?;
    crate::send(&mut io, request).map_err(|error| {
        // Encoding/size validation happens before any request bytes are written.
        if error == Error::Invalid {
            error
        } else {
            Error::AdminOutcomeUnknown
        }
    })?;
    let reply = crate::receive(&mut io).map_err(|_| Error::AdminOutcomeUnknown)?;
    complete_reply(&mut io, &file, reply)
}

fn complete_reply<R>(io: &mut AdminIo, file: &private_fs::PrivateFile, reply: R) -> Result<R> {
    file.verify().map_err(|_| Error::AdminOutcomeUnknown)?;
    // Receipt assists bounded server cleanup; it cannot invalidate a verified reply.
    let _ = io.write_all(&[1]);
    Ok(reply)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn connect(path: &Path) -> Stream {
        #[cfg(unix)]
        let name = path
            .join("admin.sock")
            .to_fs_name::<interprocess::local_socket::GenericFilePath>()
            .unwrap();
        #[cfg(windows)]
        let capability = std::fs::read(path.join("admin.cap")).unwrap();
        #[cfg(windows)]
        let name = format!("starlings-{}", crate::hash(&capability))
            .to_ns_name::<interprocess::local_socket::GenericNamespaced>()
            .unwrap();
        Stream::connect(name).unwrap()
    }

    #[test]
    fn decoded_reply_remains_known_when_receipt_deadline_expires() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().canonicalize().unwrap().join("receipt");
        PrivateDir::ensure(&path).unwrap();
        let listener = AdminListener::bind(&path, ListenerBudget::default()).unwrap();
        let mut io = AdminIo {
            stream: connect(&path),
            _permit: None,
            deadline: Instant::now() + REQUEST_TIME,
        };
        io.stream.set_nonblocking(true).unwrap();
        let mut incoming = listener.accept().unwrap();
        let expected = serde_json::json!({"revoked": true});
        crate::send(&mut incoming.io, &expected).unwrap();
        let reply: serde_json::Value = crate::receive(&mut io).unwrap();
        let directory = PrivateDir::open(&path).unwrap();
        let file = directory.open_file("admin.cap").unwrap();
        // Model elapsed time after bounded reply decoding; never renew the deadline.
        io.deadline = Instant::now();
        assert_eq!(complete_reply(&mut io, &file, reply), Ok(expected));
    }

    #[cfg(unix)]
    #[test]
    fn decoded_reply_still_requires_the_protected_capability_file() {
        use std::os::unix::fs::PermissionsExt;
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().canonicalize().unwrap().join("unverified-reply");
        PrivateDir::ensure(&path).unwrap();
        let listener = AdminListener::bind(&path, ListenerBudget::default()).unwrap();
        let mut io = AdminIo {
            stream: connect(&path),
            _permit: None,
            deadline: Instant::now() + REQUEST_TIME,
        };
        let _incoming = listener.accept().unwrap();
        let file = PrivateDir::open(&path)
            .unwrap()
            .open_file("admin.cap")
            .unwrap();
        std::fs::set_permissions(
            path.join("admin.cap"),
            std::fs::Permissions::from_mode(0o644),
        )
        .unwrap();
        assert_eq!(
            complete_reply(&mut io, &file, serde_json::json!({"revoked":true})),
            Err(Error::AdminOutcomeUnknown)
        );
    }

    #[test]
    fn dispatched_command_with_lost_reply_has_an_unknown_outcome() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().canonicalize().unwrap().join("lost-reply");
        PrivateDir::ensure(&path).unwrap();
        let listener = AdminListener::bind(&path, ListenerBudget::default()).unwrap();
        let executed = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            let client = scope.spawn(|| {
                admin_call::<_, serde_json::Value>(&path, &serde_json::json!({"revoke":true}))
            });
            let mut incoming = listener.accept().unwrap();
            assert_eq!(
                incoming.read_command::<serde_json::Value>().unwrap(),
                serde_json::json!({"revoke":true})
            );
            executed.fetch_add(1, Ordering::Relaxed);
            drop(incoming); // Authenticated dispatch happened; no reply was delivered.
            let error = client.join().unwrap().unwrap_err();
            assert_eq!(executed.load(Ordering::Relaxed), 1);
            assert_eq!(error, Error::AdminOutcomeUnknown);
        });
    }
    #[cfg(windows)]
    #[test]
    fn administrators_group_owned_pipe_still_requires_the_private_capability() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().canonicalize().unwrap().join("group-owner");
        PrivateDir::ensure(&path).unwrap();
        let AdminListener {
            listener,
            directory,
            capability,
            budget,
        } = AdminListener::bind(&path, ListenerBudget::default()).unwrap();
        drop(listener);
        // Explicit native owner SID exercises the elevated-owner case regardless
        // of whether this runner's default owner is the user or Administrators.
        let listener = AdminListener {
            listener: windows_listener(&capability, "O:BAD:P(A;;GA;;;OW)").unwrap(),
            directory,
            capability,
            budget,
        };
        listener
            .listener
            .set_nonblocking(interprocess::local_socket::ListenerNonblockingMode::Accept)
            .unwrap();
        let accept = || {
            let deadline = Instant::now() + Duration::from_secs(3);
            loop {
                if let Some(request) = listener.try_accept().unwrap() {
                    break request;
                }
                assert!(
                    Instant::now() < deadline,
                    "group-owned pipe connection timed out"
                );
                std::thread::sleep(Duration::from_millis(1));
            }
        };
        std::thread::scope(|scope| {
            let client = scope.spawn(|| {
                admin_call::<_, serde_json::Value>(&path, &serde_json::json!({"status":true}))
            });
            accept()
                .respond(|request: serde_json::Value| request)
                .unwrap();
            assert_eq!(
                client.join().unwrap().unwrap(),
                serde_json::json!({"status":true})
            );
        });
        std::thread::scope(|scope| {
            let client = scope.spawn(|| {
                let mut stream = connect(&path);
                stream.write_all(&[0; 32]).unwrap();
                let mut proof = [0; 32];
                stream.read_exact(&mut proof).unwrap();
                stream.write_all(&[0; 32]).unwrap();
            });
            let result = accept().receive::<serde_json::Value>();
            assert!(matches!(result, Err(Error::Denied)));
            client.join().unwrap();
        });
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn foreign_uid_connection_cannot_stop_local_administration() {
        // CI's elevated helper bypasses filesystem access solely to exercise SO_PEERCRED.
        // Needs passwordless sudo: GitHub's runners have it (the ARM64 job on main), the PC runner does not.
        let sudo = std::process::Command::new("sudo")
            .args(["-n", "true"])
            .status();
        if std::env::var_os("CI").is_none() || !sudo.is_ok_and(|s| s.success()) {
            return;
        }
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("foreign-uid");
        PrivateDir::ensure(&path).unwrap();
        let listener = AdminListener::bind(&path, ListenerBudget::default()).unwrap();
        let status = std::process::Command::new("sudo").args(["-n", "python3", "-c", "import socket,sys; s=socket.socket(socket.AF_UNIX); s.connect(sys.argv[1]); s.close()"])
            .arg(path.join("admin.sock")).status().unwrap();
        assert!(status.success());
        assert!(listener.try_accept().unwrap().is_none());
        std::thread::scope(|scope| {
            let client = scope.spawn(|| {
                admin_call::<_, serde_json::Value>(&path, &serde_json::json!({"status":true}))
            });
            listener
                .accept()
                .unwrap()
                .respond(|request: serde_json::Value| request)
                .unwrap();
            assert_eq!(
                client.join().unwrap().unwrap(),
                serde_json::json!({"status":true})
            );
        });
    }

    #[test]
    fn local_admin_waits_for_delayed_proof_and_response_within_one_deadline() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().canonicalize().unwrap().join("delayed");
        PrivateDir::ensure(&path).unwrap();
        let listener = AdminListener::bind(&path, ListenerBudget::default()).unwrap();
        std::thread::scope(|scope| {
            let client = scope.spawn(|| {
                admin_call::<_, serde_json::Value>(&path, &serde_json::json!({"command":"status"}))
            });
            let incoming = listener.accept().unwrap();
            std::thread::sleep(Duration::from_millis(25));
            incoming
                .respond(|value: serde_json::Value| {
                    std::thread::sleep(Duration::from_millis(25));
                    value
                })
                .unwrap();
            assert_eq!(
                client.join().unwrap().unwrap(),
                serde_json::json!({"command":"status"})
            );
        });
    }
    #[test]
    fn local_admin_large_request_and_response_preserve_every_byte() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().canonicalize().unwrap().join("large");
        PrivateDir::ensure(&path).unwrap();
        let listener = AdminListener::bind(&path, ListenerBudget::default()).unwrap();
        let value = serde_json::json!({"record": "0123456789abcdef".repeat(768)});
        std::thread::scope(|scope| {
            let client = scope.spawn(|| admin_call::<_, serde_json::Value>(&path, &value));
            listener
                .accept()
                .unwrap()
                .respond(|received: serde_json::Value| received)
                .unwrap();
            assert_eq!(client.join().unwrap().unwrap(), value);
        });
    }
    #[test]
    fn local_admin_rejects_wrong_server_proof_before_sending_capability() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().canonicalize().unwrap().join("spoof");
        PrivateDir::ensure(&path).unwrap();
        let listener = AdminListener::bind(&path, ListenerBudget::default()).unwrap();
        std::thread::scope(|scope| {
            let client = scope.spawn(|| {
                admin_call::<_, serde_json::Value>(&path, &serde_json::json!({"command":"status"}))
            });
            let mut incoming = listener.accept().unwrap();
            incoming.io.deadline = Instant::now() + Duration::from_millis(250);
            let mut challenge = [0; 32];
            incoming.io.read_exact(&mut challenge).unwrap();
            incoming.io.write_all(&[0; 32]).unwrap();
            assert!(matches!(client.join().unwrap(), Err(Error::Denied)));
            let mut byte = [0];
            assert!(!matches!(incoming.io.read(&mut byte), Ok(1)));
        });
    }
    #[test]
    fn local_admin_requires_private_capability_and_bounded_frame() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().canonicalize().unwrap().join("ipc");
        let directory = PrivateDir::ensure(&path).unwrap();
        let budget = ListenerBudget::default();
        let listener = AdminListener::bind(&path, budget.clone()).unwrap();
        assert!(AdminListener::bind(&path, budget.clone()).is_err());
        let mut capability = [0; 32];
        directory
            .open_file("admin.cap")
            .unwrap()
            .read_exact(&mut capability)
            .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(path.join("admin.sock"))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
        for case in 0..4 {
            let mut client = connect(&path);
            let mut incoming = listener.accept().unwrap();
            incoming.io.deadline = Instant::now() + Duration::from_millis(100);
            client.write_all(&[0; 32]).unwrap();
            client
                .write_all(if case == 1 { &[0; 32] } else { &capability })
                .unwrap();
            match case {
                0 => {
                    client.write_all(&2u32.to_be_bytes()).unwrap();
                    client.write_all(b"{}").unwrap();
                }
                2 => client
                    .write_all(&((crate::MAX_RECORD + 1) as u32).to_be_bytes())
                    .unwrap(),
                _ => (),
            }
            let start = Instant::now();
            let result = incoming.receive::<serde_json::Value>();
            if case == 0 {
                assert_eq!(result.unwrap(), serde_json::json!({}));
            } else {
                assert!(result.is_err());
            }
            assert!(start.elapsed() < Duration::from_secs(1));
        }
        assert_eq!(budget.usage().unwrap().connections, 0);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(
                path.join("admin.cap"),
                std::fs::Permissions::from_mode(0o644),
            )
            .unwrap();
            assert!(listener.accept().is_err());
        }
    }
}
