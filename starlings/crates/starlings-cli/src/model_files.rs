//! Content identity for the beta's local checkpoint, independent of its absolute path.
use starlings_protocol::Digest;
use std::{collections::BTreeMap, fs, path::Path};

/// Hash every root file and chat template, including all weight shards. Hub
/// snapshots may use file symlinks into their blob store; directory symlinks are refused.
/// This verifies files at load time, not hostile same-user mutation while a model is running.
pub fn digest(model: &Path) -> Result<Digest, String> {
    let mut assets = BTreeMap::new();
    let mut directories = vec![(model.to_path_buf(), String::new())];
    while let Some((directory, prefix)) = directories.pop() {
        for entry in fs::read_dir(&directory).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| "non-UTF-8 model filename")?;
            let kind = entry.file_type().map_err(|e| e.to_string())?;
            if prefix.is_empty() && name == "chat_templates" {
                if !kind.is_dir() {
                    return Err("chat_templates must be a real directory".into());
                }
                directories.push((entry.path(), "chat_templates/".into()));
                continue;
            }
            if kind.is_dir() && prefix.is_empty() {
                // The supported engine reads root assets and chat_templates only.
                continue;
            }
            let metadata = fs::metadata(entry.path()).map_err(|e| e.to_string())?;
            if !metadata.is_file() || assets.len() >= 65_536 {
                return Err("invalid model asset or too many files".into());
            }
            let mut file = fs::File::open(entry.path()).map_err(|e| e.to_string())?;
            let before = file.metadata().map_err(|e| e.to_string())?;
            if !before.is_file() {
                return Err("model asset is not a regular file".into());
            }
            let hash = starlings_security::catalog::fingerprint(&mut file, before.len())
                .map_err(|e| format!("model asset {name}: {e:?}"))?;
            let after = file.metadata().map_err(|e| e.to_string())?;
            if before.len() != after.len() || before.modified().ok() != after.modified().ok() {
                return Err("model changed while hashing".into());
            }
            assets.insert(format!("{prefix}{name}"), (before.len(), hash));
        }
    }
    if !assets.contains_key("config.json")
        || !assets.contains_key("tokenizer.json")
        || !assets.keys().any(|name| name.ends_with(".safetensors"))
    {
        return Err("model needs config.json, tokenizer.json and safetensors weights".into());
    }
    starlings_protocol::content_hash(&(1u32, assets)).map_err(|e| format!("model identity: {e:?}"))
}

pub fn verify(model: &Path, expected: &Digest) -> Result<(), String> {
    if &digest(model)? != expected {
        return Err(
            "model files changed; approve the checkpoint again in a new state directory".into(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_covers_bytes_names_and_templates_but_not_absolute_paths() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        for root in [a.path(), b.path()] {
            for (name, bytes) in [
                ("config.json", b"config".as_slice()),
                ("tokenizer.json", b"tokens"),
                ("model.safetensors", b"weights"),
            ] {
                fs::write(root.join(name), bytes).unwrap();
            }
            fs::create_dir(root.join("chat_templates")).unwrap();
            fs::write(root.join("chat_templates/default.jinja"), b"template").unwrap();
        }
        let approved = digest(a.path()).unwrap();
        assert_eq!(approved, digest(b.path()).unwrap());
        for name in [
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "chat_templates/default.jinja",
        ] {
            let path = b.path().join(name);
            let original = fs::read(&path).unwrap();
            fs::write(&path, vec![b'x'; original.len()]).unwrap();
            assert!(verify(b.path(), &approved).is_err(), "{name}");
            fs::write(&path, original).unwrap();
        }
        fs::write(b.path().join("extra.safetensors"), b"extra").unwrap();
        assert!(verify(b.path(), &approved).is_err());
        fs::remove_file(b.path().join("extra.safetensors")).unwrap();
        fs::remove_file(b.path().join("tokenizer.json")).unwrap();
        assert!(digest(b.path()).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn hub_blob_symlinks_are_hashed_and_directory_symlinks_are_refused() {
        let root = tempfile::tempdir().unwrap();
        let blob = tempfile::NamedTempFile::new().unwrap();
        fs::write(blob.path(), b"weights").unwrap();
        fs::write(root.path().join("config.json"), b"{}").unwrap();
        fs::write(root.path().join("tokenizer.json"), b"{}").unwrap();
        std::os::unix::fs::symlink(blob.path(), root.path().join("model.safetensors")).unwrap();
        let approved = digest(root.path()).unwrap();
        fs::write(blob.path(), b"changed").unwrap();
        assert!(verify(root.path(), &approved).is_err());
        std::os::unix::fs::symlink(root.path(), root.path().join("chat_templates")).unwrap();
        assert!(digest(root.path()).is_err());
    }
}
