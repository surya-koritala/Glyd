mod cluster;
mod daemon;
mod discovery;
mod links;
mod models;
mod service;
mod service_template;
#[cfg(feature = "test-backend")]
mod simulation;
mod stage;
mod task_routing;
mod up;
use std::{env, process::ExitCode};

fn main() -> ExitCode {
    let args: Vec<_> = env::args_os().skip(1).collect();
    #[cfg(any(debug_assertions, feature = "test-backend"))]
    if env::var_os("STARLINGS_TEST_INIT_EXIT_PHASE").is_some()
        && args
            .windows(2)
            .any(|pair| pair[0] == "cluster" && pair[1] == "init")
    {
        eprintln!("INIT_START_PHASE=entry");
    }
    match args.as_slice() {
        [] => help(),
        [arg] if arg == "--help" || arg == "-h" => help(),
        [arg] if arg == "--version" || arg == "-V" => {
            println!("starlings {}", env!("CARGO_PKG_VERSION"));
        }
        [arg] if arg == "doctor" => {
            println!(
                "protocol={} os={} arch={} inference=unavailable",
                starlings_protocol::PROTOCOL_VERSION,
                env::consts::OS,
                env::consts::ARCH,
            );
        }
        [command, action, path] if command == "plan" && action == "inspect-glyd-costs" => {
            return print_report(starlings_cli::glyd_costs::inspect_file(
                std::path::Path::new(path),
            ));
        }
        [command, action, path] if command == "plan" && action == "calibrate" => {
            return print_report(starlings_cli::calibration::report_file(
                std::path::Path::new(path),
                starlings_protocol::FixtureMode::Reject,
            ));
        }
        #[cfg(feature = "test-backend")]
        [command, action, path, flag]
            if command == "plan" && action == "calibrate" && flag == "--allow-fixtures" =>
        {
            return print_report(starlings_cli::calibration::report_file(
                std::path::Path::new(path),
                starlings_protocol::FixtureMode::Simulation,
            ));
        }
        [command, action, path] if command == "plan" && action == "modes" => {
            return print_report(starlings_cli::planning::modes_file(
                std::path::Path::new(path),
                starlings_protocol::FixtureMode::Reject,
            ));
        }
        #[cfg(feature = "test-backend")]
        [command, action, path, flag]
            if command == "plan" && action == "modes" && flag == "--allow-fixtures" =>
        {
            return print_report(starlings_cli::planning::modes_file(
                std::path::Path::new(path),
                starlings_protocol::FixtureMode::Simulation,
            ));
        }
        [command, action, path] if command == "plan" && action == "preview" => {
            return print_report(starlings_cli::planning::preview_file(
                std::path::Path::new(path),
                starlings_protocol::FixtureMode::Reject,
            ));
        }
        #[cfg(feature = "test-backend")]
        [command, action, path, flag]
            if command == "plan" && action == "preview" && flag == "--allow-fixtures" =>
        {
            return print_report(starlings_cli::planning::preview_file(
                std::path::Path::new(path),
                starlings_protocol::FixtureMode::Simulation,
            ));
        }
        [command, action, matrix, plan, output]
            if command == "scoreboard" && action == "collect" =>
        {
            return print_report(starlings_cli::collector::collect_file(
                std::path::Path::new(matrix),
                std::path::Path::new(plan),
                std::path::Path::new(output),
                starlings_protocol::FixtureMode::Reject,
            ));
        }
        #[cfg(feature = "test-backend")]
        [command, action, matrix, plan, output, flag]
            if command == "scoreboard" && action == "collect" && flag == "--allow-fixtures" =>
        {
            return print_report(starlings_cli::collector::collect_file(
                std::path::Path::new(matrix),
                std::path::Path::new(plan),
                std::path::Path::new(output),
                starlings_protocol::FixtureMode::Simulation,
            ));
        }
        [command, action, path] if command == "scoreboard" && action == "report" => {
            return print_report(starlings_cli::scoreboard::report_file(
                std::path::Path::new(path),
                starlings_protocol::FixtureMode::Reject,
            ));
        }
        #[cfg(feature = "test-backend")]
        [command, action, path, flag]
            if command == "scoreboard" && action == "report" && flag == "--allow-fixtures" =>
        {
            return print_report(starlings_cli::scoreboard::report_file(
                std::path::Path::new(path),
                starlings_protocol::FixtureMode::Simulation,
            ));
        }
        [command, action, path] if command == "benchmark" && action == "inspect" => {
            return print_report(starlings_cli::benchmark::inspect_file(
                std::path::Path::new(path),
                starlings_protocol::FixtureMode::Reject,
            ));
        }
        #[cfg(feature = "test-backend")]
        [command, action, path, flag]
            if command == "benchmark" && action == "inspect" && flag == "--allow-fixtures" =>
        {
            return print_report(starlings_cli::benchmark::inspect_file(
                std::path::Path::new(path),
                starlings_protocol::FixtureMode::Simulation,
            ));
        }
        [arg, directory] if arg == "probe" => {
            let snapshot = starlings_adapters::resources::HostSnapshot::collect(
                std::path::Path::new(directory),
            );
            return print_probe(snapshot);
        }
        [arg, directory, flag, library, approved] if arg == "probe" && flag == "--nvidia" => {
            let approved = approved.to_str().and_then(|s| s.to_owned().try_into().ok());
            let Some(approved) = approved else {
                eprintln!("NVIDIA_LIBRARY_DIGEST_REQUIRED");
                return ExitCode::from(2);
            };
            match starlings_adapters::resources::HostSnapshot::collect_nvidia(
                std::path::Path::new(directory),
                std::path::Path::new(library),
                &approved,
            ) {
                Ok(snapshot) => return print_probe(snapshot),
                Err(error) => {
                    eprintln!("{error}");
                    return ExitCode::FAILURE;
                }
            }
        }
        #[cfg(feature = "test-backend")]
        [arg, ..] if arg == "rangeworker" => starlings_adapters::partial::fixture::run(),
        #[cfg(feature = "test-backend")]
        [arg, ..] if arg == "fixture-service" => {
            starlings_adapters::service::fixture::run();
        }
        #[cfg(feature = "test-backend")]
        [arg] if arg == "fake-smoke" => {
            if let Err(error) = simulation::run() {
                eprintln!("simulation failed: {error}");
                return ExitCode::FAILURE;
            }
            println!("simulation only: fake backend lifecycle simulation passed");
        }
        #[cfg(feature = "test-backend")]
        [arg] if arg == "simulate" => {
            if let Err(error) = simulation::run() {
                eprintln!("simulation failed: {error}");
                return ExitCode::FAILURE;
            }
        }
        #[cfg(feature = "test-backend")]
        [arg, node, capacity] if arg == "fake-worker" => {
            let parsed = node
                .to_str()
                .and_then(|v| v.parse::<u64>().ok())
                .zip(capacity.to_str().and_then(|v| v.parse::<u64>().ok()));
            let Some((node, capacity)) = parsed else {
                return ExitCode::from(2);
            };
            if starlings_adapters::fake::worker_loop(node, capacity).is_err() {
                eprintln!("simulation worker rejected input");
                return ExitCode::FAILURE;
            }
        }
        _ => {
            if let Err(error) = cluster::run(&args) {
                eprintln!("{error}");
                return if error == cluster::USAGE {
                    ExitCode::from(2)
                } else {
                    ExitCode::FAILURE
                };
            }
        }
    }
    ExitCode::SUCCESS
}
fn print_probe(snapshot: starlings_adapters::resources::HostSnapshot) -> ExitCode {
    use std::io::Write;
    let result = serde_json::to_writer_pretty(std::io::stdout().lock(), &snapshot.report());
    if result.is_err() || writeln!(std::io::stdout()).is_err() {
        eprintln!("OUTPUT_UNAVAILABLE");
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn print_report<T: serde::Serialize>(result: Result<T, &'static str>) -> ExitCode {
    use std::io::Write;
    let result = result.and_then(|report| {
        let mut out = std::io::stdout().lock();
        serde_json::to_writer_pretty(&mut out, &report).map_err(|_| "OUTPUT_UNAVAILABLE")?;
        writeln!(out).map_err(|_| "OUTPUT_UNAVAILABLE")
    });
    if let Err(error) = result {
        let _ = writeln!(std::io::stderr(), "{error}");
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn help() {
    println!("Run a model across two GPUs (docs/beta.md):");
    println!("  starlings up --model glyd/Qwen3.8-27B-swift              one machine");
    println!("  starlings up --model glyd/Qwen3.8-27B-swift --machines 2  first of two machines");
    println!("  starlings join --model glyd/Qwen3.8-27B-swift            second machine");
    println!(
        "  (options: --ctx N --sessions 1|2 --drafts N --host IP --engine FILE --gpu-lib FILE --yes)"
    );
    println!();
    println!("Usage: starlings [--help | --version | doctor | probe CACHE_DIRECTORY]");
    println!("NVIDIA diagnostics: probe CACHE_DIRECTORY --nvidia ABSOLUTE_NVML_LIBRARY SHA256");
    println!(
        "Offline evidence: scoreboard report MATRIX_JSON (no backend execution or qualification)"
    );
    println!(
        "Bounded client measurements: scoreboard collect MATRIX_JSON PLAN_JSON NEW_MATRIX_JSON"
    );
    println!("Offline record check: benchmark inspect RUN_JSON (no workload or qualification)");
    println!(
        "Private link snapshot: cluster links-record NODE NEW_PRIVATE_JSONL (no probe activation)"
    );
    println!("Advisory placement: plan preview INPUT_JSON (no reservations or execution)");
    println!(
        "Offline prediction errors: plan calibrate INPUT_JSON (no collection or qualification)"
    );
    println!("Advisory tier costs: plan modes INPUT_JSON (no reservations or execution)");
    println!(
        "Provider cost receipt: plan inspect-glyd-costs PROFILE_JSON (no admission or qualification)"
    );
    println!(
        "Directed links: cluster probe-digest PRIVATE_FILE | probe-listen PRIVATE_FILE --approve SHA256 | probe PRIVATE_FILE --approve SHA256 | links NODE"
    );
    println!(
        "Trusted local daemon and peer registry. Generation requires explicitly approved local qualification."
    );
    println!("[--state-dir ABSOLUTE_PATH] daemon [--mdns LOCAL_IPV4] | cluster COMMAND");
    println!(
        "Web UI (coordinator): daemon ... --ui 127.0.0.1:PORT [--ui-client CLIENT_NODE]; open the printed UI_READY address"
    );
    println!(
        "Worker warm-up: daemon --service PRIVATE_CONFIG --approve CONFIG_DIGEST (no inference qualification)"
    );
    println!("Service state: cluster service (readiness, retained pin and process-group cleanup)");
    println!("Token activation snapshot: cluster token-status PRIVATE_TOKEN (no secret output)");
    println!(
        "Inactive descriptor export: service template NEW_PRIVATE_DIRECTORY (no install or start)"
    );
    println!(
        "Redacted local bundle: cluster diagnostics (three existing read-only queries; JSON on stdout)"
    );
    println!(
        "Qualified recovery: cluster recover-service LEASE CONFIG_DIGEST | model service-state | model recover-service LEASE CONFIG_DIGEST (offline); never forces unknown cleanup or starts a backend"
    );
    println!(
        "Qualified CPU routing: daemon --execute PRIVATE_CONFIG --approve DIGEST (worker) | daemon --route PRIVATE_CONFIG --approve DIGEST (coordinator)"
    );
    println!(
        "Two-stage pipeline: daemon --stage PRIVATE_CONFIG --approve DIGEST (worker) | daemon --pipeline PRIVATE_CONFIG --approve DIGEST (coordinator); model stage-digest | pipeline-digest PRIVATE_CONFIG; model files-digest MODEL_DIR"
    );
    println!(
        "Live route activation: cluster route PRIVATE_CONFIG --approve DIGEST; one route per coordinator process"
    );
    println!(
        "Local policy: cluster permissions NODE PRIVATE_SCOPES | permissions-status NODE | token CLIENT_NODE PRIVATE_SCOPES SECONDS PRIVATE_OUTPUT | revoke-token PRIVATE_TOKEN"
    );
    println!(
        "Route preparation: model execution-digest PRIVATE_CONFIG | route-digest PRIVATE_CONFIG; never grants approval or starts a backend"
    );
    println!(
        "Service preparation: model paths DIGEST | service-digest PRIVATE_CONFIG (never starts a backend)"
    );
    println!(
        "Model: model seal DRAFT SOURCE_DIR | inspect FILE | register FILE | approve DIGEST [--redistribute] | revoke DIGEST | show DIGEST | check DIGEST"
    );
    println!(
        "Artifacts: model resolve REPOSITORY REVISION | fetch DIGEST | import DIGEST SOURCE_DIR | ready DIGEST | quota BYTES | storage | cost DIGEST FIRST_LAYER END_EXCLUSIVE | pin DIGEST LEASE | unpin LEASE | gc"
    );
    println!("Resource diagnostics: cluster resources NODE (coordinator only)");
    println!(
        "Cluster: init [--listen IP:PORT ...] | invite worker|client FILE [--via IP:PORT] | invite-status FILE | prepare FILE | approve NODE FINGERPRINT | join FILE [--via IP:PORT] | routes IP:PORT ... [--source IP:0] | discover --interface LOCAL_IPV4 | status | nodes | topology | drain | stop | leave | revoke NODE"
    );
    #[cfg(feature = "test-backend")]
    println!("Development only: fake-smoke | simulate (simulation, no model execution)");
}
