//! Exclusive local process ownership and narrowly scoped transient IPC cleanup.
use crate::{
    Error, PrivateDir, Result,
    authorization::{FileLock, lock_file},
};
use std::path::Path;
#[cfg(windows)]
use std::{fs::File, path::PathBuf};

/// Retain this guard from before initialization/cleanup until all daemon threads have stopped.
/// The lock file is persistent: unlinking it would permit competing lock owners.
pub struct StateLock {
    directory: PrivateDir,
    #[cfg(windows)]
    path: PathBuf,
    #[cfg(windows)]
    _rename_guard: File,
    _file: FileLock,
}
impl StateLock {
    pub fn acquire(path: &Path) -> Result<Self> {
        let directory = match PrivateDir::open(path) {
            Ok(directory) => directory,
            Err(private_fs::Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                // Drop the creator's DELETE-capable handle before retaining read-only ownership.
                PrivateDir::ensure(path)?;
                PrivateDir::open(path)?
            }
            Err(error) => return Err(error.into()),
        };
        let file = match directory.open_file("daemon.lock") {
            Ok(file) => file,
            Err(_) => directory.create("daemon.lock")?,
        };
        let file = lock_file(&file)?;
        file.0.try_lock().map_err(|_| Error::ProfileInUse)?;
        #[cfg(windows)]
        let rename_guard = {
            use std::os::windows::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .read(true)
                .share_mode(3)
                .custom_flags(0x02000000)
                .open(path)?
        };
        Ok(Self {
            _file: file,
            #[cfg(windows)]
            _rename_guard: rename_guard,
            directory,
            #[cfg(windows)]
            path: path.canonicalize()?,
        })
    }
    pub fn directory(&self) -> &PrivateDir {
        &self.directory
    }
    pub fn cleanup_admin(&self) -> Result<()> {
        private_fs::verify(&self._file.0)?;
        #[cfg(unix)]
        {
            use rustix::fs::{AtFlags, FileType, Mode, fstat, statat, unlinkat};
            let owner = fstat(&self.directory).map_err(std::io::Error::from)?.st_uid;
            for name in ["admin.cap", "admin.sock"] {
                let metadata = match statat(&self.directory, name, AtFlags::SYMLINK_NOFOLLOW) {
                    Ok(meta) => meta,
                    Err(rustix::io::Errno::NOENT) => continue,
                    Err(error) => return Err(std::io::Error::from(error).into()),
                };
                if name == "admin.cap" {
                    self.directory.open_file(name)?.verify()?;
                } else if FileType::from_raw_mode(metadata.st_mode) != FileType::Socket
                    || Mode::from_raw_mode(metadata.st_mode).as_raw_mode() & 0o777 != 0o600
                    || metadata.st_uid != owner
                {
                    return Err(Error::Denied);
                }
                unlinkat(&self.directory, name, AtFlags::empty()).map_err(std::io::Error::from)?;
            }
        }
        #[cfg(windows)]
        {
            // The held no-delete-share directory handle prevents live-profile rename.
            match self.directory.open_file("admin.cap") {
                Ok(file) => {
                    file.verify()?;
                    drop(file);
                    std::fs::remove_file(self.path.join("admin.cap"))?;
                }
                Err(private_fs::Error::Io(error))
                    if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn owner_drop_releases_its_lock_while_a_duplicate_description_survives() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("state");
        let owner = StateLock::acquire(&path).unwrap();
        let alias = owner._file.0.try_clone().unwrap();
        for _ in 0..2 {
            assert!(matches!(
                StateLock::acquire(&path),
                Err(Error::ProfileInUse)
            ));
        }
        drop(owner);
        let next = StateLock::acquire(&path).expect("the owner released its profile");
        assert!(matches!(
            StateLock::acquire(&path),
            Err(Error::ProfileInUse)
        ));
        drop(alias);
        assert!(matches!(
            StateLock::acquire(&path),
            Err(Error::ProfileInUse)
        ));
        drop(next);
        assert!(StateLock::acquire(&path).is_ok());
    }
    #[cfg(unix)]
    #[test]
    fn cleanup_refuses_replaced_socket_without_deleting_it() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().canonicalize().unwrap().join("state");
        let lock = StateLock::acquire(&path).unwrap();
        lock.directory
            .write_atomic_durable("admin.sock", b"retained")
            .unwrap();
        assert_eq!(lock.cleanup_admin(), Err(Error::Denied));
        assert_eq!(std::fs::read(path.join("admin.sock")).unwrap(), b"retained");
        std::fs::remove_file(path.join("admin.sock")).unwrap();
        std::os::unix::fs::symlink("daemon.lock", path.join("admin.sock")).unwrap();
        assert_eq!(lock.cleanup_admin(), Err(Error::Denied));
        assert!(
            std::fs::symlink_metadata(path.join("admin.sock"))
                .unwrap()
                .is_symlink()
        );
    }
}
