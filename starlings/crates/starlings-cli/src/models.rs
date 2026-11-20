//! Explicit catalog/storage commands. Only resolve/fetch may contact the provider.
use crate::cluster::{Config, Paths, Result, USAGE, security};
use starlings_protocol::{Digest, FixtureMode, ModelManifest, decode};
use starlings_security::{
    PrivateDir,
    catalog::{Catalog, hub::Hub, read_bounded, seal_preloaded},
};
use std::{ffi::OsString, fs::File, io::Write, path::Path};

fn input(path: &OsString) -> Result<Vec<u8>> {
    read_bounded(File::open(path).map_err(|_| "MANIFEST_UNAVAILABLE")?).map_err(security)
}
fn digest(value: &OsString) -> Result<Digest> {
    value
        .to_str()
        .ok_or(USAGE)?
        .to_owned()
        .try_into()
        .map_err(|_| USAGE.into())
}
fn print(value: &impl serde::Serialize) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|_| "INVALID_MANIFEST")?;
    let mut out = std::io::stdout().lock();
    out.write_all(&bytes)
        .and_then(|()| out.write_all(b"\n"))
        .map_err(|_| "OUTPUT_UNAVAILABLE".into())
}
pub fn run(mut paths: Paths, args: &[OsString]) -> Result<()> {
    match args {
        [command, model] if command == "files-digest" => {
            let manifest = starlings_cli::model_files::digest(Path::new(model))?;
            return print(&serde_json::json!({"model_files_sha256": manifest,
                "approval_granted": false}));
        }
        [command] if command == "qualification-target" => {
            return print(
                &serde_json::json!({"target":starlings_adapters::resources::HostSnapshot::collect_memory().qualification_target(), "approval_granted":false}),
            );
        }
        [command, file] if command == "execution-digest" || command == "route-digest" => {
            let (directory, name) = crate::cluster::private_parent(Path::new(file))?;
            let digest = if command == "execution-digest" {
                starlings_protocol::content_hash(&crate::cluster::read::<
                    starlings_cli::routes::WorkerRoute,
                >(&directory, &name)?)
            } else {
                starlings_protocol::content_hash(&crate::cluster::read::<
                    starlings_cli::routes::CoordinatorRoute,
                >(&directory, &name)?)
            }
            .map_err(|_| "INVALID_ROUTE")?;
            return print(&serde_json::json!({"configuration":digest, "approval_granted":false}));
        }
        [command, file] if command == "stage-digest" || command == "pipeline-digest" => {
            let (directory, name) = crate::cluster::private_parent(Path::new(file))?;
            let digest = if command == "stage-digest" {
                starlings_protocol::content_hash(&crate::cluster::read::<
                    starlings_cli::routes::StageRoute,
                >(&directory, &name)?)
            } else {
                starlings_protocol::content_hash(&crate::cluster::read::<
                    starlings_cli::routes::PipelineRoute,
                >(&directory, &name)?)
            }
            .map_err(|_| "INVALID_ROUTE")?;
            return print(&serde_json::json!({"configuration":digest, "approval_granted":false}));
        }
        // GLY-362: where to cut MODEL between two GPUs with these free GiB (it approves nothing).
        [command, model, free0, free1, rest @ ..] if command == "split" => {
            let gib = |v: &std::ffi::OsString| -> std::result::Result<u64, String> {
                let v = v.to_string_lossy();
                let g: f64 = v.parse().map_err(|_| format!("{v}: free GiB"))?;
                Ok((g * f64::from(1u32 << 30)) as u64)
            };
            let flag = |name: &str, default: u32| {
                rest.windows(2)
                    .find(|w| w[0] == name)
                    .map_or(Ok(default), |w| {
                        (w[1].to_string_lossy().parse()).map_err(|_| format!("{name}: a number"))
                    })
            };
            let (context, sessions) = (flag("--ctx", 4096)?, flag("--sessions", 1)?);
            let shape = starlings_cli::split::read(Path::new(model), context, sessions)?;
            let reserve = starlings_cli::split::RESERVE;
            let free = [gib(free0)?, gib(free1)?];
            let cut = starlings_cli::split::cut(&shape, free, reserve);
            let need = |range: std::ops::Range<usize>, extra: u64| {
                shape.layers[range].iter().sum::<u64>() + extra + reserve
            };
            let layers = shape.layers.len();
            return print(&serde_json::json!({
                "layers": layers, "context": context, "sessions": sessions,
                "cut": cut,
                "stages": cut.map(|c| {
                    let c = c as usize;
                    [serde_json::json!({"layers": [0, c], "bytes": need(0..c, shape.first), "free": free[0]}),
                     serde_json::json!({"layers": [c, layers], "bytes": need(c..layers, shape.last), "free": free[1]})]
                }),
            }));
        }
        [command, file] if command == "service-digest" => {
            let (directory, name) = crate::cluster::private_parent(Path::new(file))?;
            let config: starlings_adapters::service::ServiceConfig =
                crate::cluster::read(&directory, &name)?;
            return print(
                &serde_json::json!({ "configuration": config.digest().map_err(|e| e.to_string())?, "inference_qualified": false }),
            );
        }
        [command, repository, revision] if command == "resolve" => {
            return print(
                &Hub::new(None)
                    .map_err(security)?
                    .resolve(
                        repository.to_str().ok_or(USAGE)?,
                        revision.to_str().ok_or(USAGE)?,
                    )
                    .map_err(security)?,
            );
        }
        [command, file] if command == "inspect" => {
            let manifest: ModelManifest =
                decode(&input(file)?, FixtureMode::Reject).map_err(|_| "INVALID_MANIFEST")?;
            return print(&manifest);
        }
        [command, file, directory] if command == "seal" => {
            return print(&seal_preloaded(&input(file)?, Path::new(directory)).map_err(security)?);
        }
        [command, _]
            if matches!(
                command.to_str(),
                Some(
                    "register"
                        | "approve"
                        | "revoke"
                        | "show"
                        | "check"
                        | "ready"
                        | "paths"
                        | "fetch"
                        | "quota"
                        | "unpin"
                )
            ) => {}
        [command] if command == "gc" || command == "storage" || command == "service-state" => (),
        [command, _, _]
            if command == "import" || command == "pin" || command == "recover-service" => {}
        [command, _, _, _] if command == "cost" => (),
        [command, _, flag] if command == "approve" && flag == "--redistribute" => (),
        _ => return Err(USAGE.into()),
    }
    paths.cache = Config::load(
        &PrivateDir::open(&paths.state)
            .map_err(|_| "STATE_DIR_NOT_FOUND: initialize or select an existing profile")?,
    )?
    .cache_dir;
    paths.cache()?;
    // Drop the creator handle before StateLock retains read-only Windows ownership.
    PrivateDir::open(&paths.cache)
        .map_err(|_| "PRIVATE_CACHE_REQUIRED")?
        .ensure_dir("models")
        .map_err(|_| "PRIVATE_CACHE_REQUIRED")?;
    let mut catalog = Catalog::open(&paths.cache.join("models")).map_err(security)?;
    match args {
        [command] if command == "service-state" => {
            print(&catalog.service_record().map_err(security)?)
        }
        [command, lease, configuration] if command == "recover-service" => {
            let lease = lease.to_str().ok_or(USAGE)?.parse().map_err(|_| USAGE)?;
            let repeated = catalog
                .recover_service(lease, &digest(configuration)?)
                .map_err(security)?;
            print(
                &serde_json::json!({"service_recovered":true,"repeated":repeated,"backend_started":false}),
            )
        }
        [command, id] if command == "paths" => {
            let id = digest(id)?;
            let manifest = catalog.authorized(&id, false).map_err(security)?;
            let paths = manifest
                .assets
                .iter()
                .map(|asset| {
                    catalog
                        .asset_path(&id, &asset.path)
                        .map(|path| (asset.path.clone(), path))
                })
                .collect::<std::result::Result<std::collections::BTreeMap<_, _>, _>>()
                .map_err(security)?;
            print(&paths)
        }
        [command, id, start, end] if command == "cost" => {
            let number = |n: &OsString| n.to_str().ok_or(USAGE)?.parse::<u32>().map_err(|_| USAGE);
            print(
                &catalog
                    .storage_cost(&digest(id)?, number(start)?, number(end)?)
                    .map_err(security)?,
            )
        }
        [command, id, source] if command == "import" => {
            catalog
                .import_directory(&digest(id)?, Path::new(source))
                .map_err(security)?;
            print(&serde_json::json!({"ready":true,"inference_validated":false}))
        }
        [command, id] if command == "fetch" => {
            Hub::new(None)
                .map_err(security)?
                .fetch(&mut catalog, &digest(id)?)
                .map_err(security)?;
            print(&serde_json::json!({"ready":true,"inference_validated":false}))
        }
        [command, id] if command == "ready" => {
            catalog.ready(&digest(id)?).map_err(security)?;
            print(&serde_json::json!({"ready":true,"inference_validated":false}))
        }
        [command, bytes] if command == "quota" => {
            let bytes = bytes
                .to_str()
                .ok_or(USAGE)?
                .parse::<u64>()
                .map_err(|_| USAGE)?;
            catalog.set_quota(bytes).map_err(security)?;
            print(&serde_json::json!({"quota_bytes":bytes.to_string()}))
        }
        [command] if command == "storage" => print(
            &serde_json::json!({"used_bytes":catalog.storage_bytes().map_err(security)?.to_string(),"quota_bytes":catalog.quota().map_err(security)?.to_string()}),
        ),
        [command] if command == "gc" => {
            print(&serde_json::json!({"removed_files":catalog.gc().map_err(security)?}))
        }
        [command, id, lease] if command == "pin" => {
            catalog
                .pin(
                    &digest(id)?,
                    lease.to_str().ok_or(USAGE)?.parse().map_err(|_| USAGE)?,
                )
                .map_err(security)?;
            print(&serde_json::json!({"pinned":true}))
        }
        [command, lease] if command == "unpin" => {
            catalog
                .unpin(lease.to_str().ok_or(USAGE)?.parse().map_err(|_| USAGE)?)
                .map_err(security)?;
            print(&serde_json::json!({"pinned":false}))
        }
        [command, file] if command == "register" => {
            let manifest = catalog.register(&input(file)?).map_err(security)?;
            print(
                &serde_json::json!({"manifest_digest":manifest.manifest_digest,"registered":true,"inference_validated":false}),
            )
        }
        [command, id] if command == "show" => {
            print(&catalog.manifest(&digest(id)?).map_err(security)?)
        }
        [command, id] if command == "check" => {
            catalog.authorized(&digest(id)?, false).map_err(security)?;
            print(
                &serde_json::json!({"approved":true,"artifact_readiness":"unchecked","inference_validated":false}),
            )
        }
        [command, id] if command == "revoke" => {
            catalog.revoke(&digest(id)?).map_err(security)?;
            print(&serde_json::json!({"approved":false}))
        }
        [command, id, ..] if command == "approve" => {
            let redistribution = args.len() == 3;
            catalog
                .approve(&digest(id)?, redistribution)
                .map_err(security)?;
            print(
                &serde_json::json!({"approved":true,"redistribution_approved":redistribution,"inference_validated":false}),
            )
        }
        _ => Err(USAGE.into()),
    }
}
