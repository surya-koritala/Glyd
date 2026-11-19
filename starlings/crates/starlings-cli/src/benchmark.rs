//! Offline integrity/consistency inspection. Never starts or qualifies a workload.
use serde::Serialize;
use starlings_protocol::{
    BenchmarkOutcome, BenchmarkRun, BenchmarkStatus, Digest, FixtureMode, InvocationId, RequestId,
    Version, byte_digest, decode,
};
use std::{collections::BTreeMap, fs::File, path::Path};

#[derive(Serialize)]
pub struct FailedRequest {
    pub request_id: RequestId,
    pub outcome: BenchmarkOutcome,
    pub error_code: String,
}
#[derive(Serialize)]
pub struct Latency {
    pub duration_ms: Option<crate::scoreboard::Summary>,
    pub ttft_ms: Option<crate::scoreboard::Summary>,
}
#[derive(Serialize)]
pub struct Inspection {
    pub version: Version,
    pub fixture: bool,
    pub qualification: &'static str,
    pub run_id: InvocationId,
    pub run_status: BenchmarkStatus,
    pub record_sha256: Digest,
    pub verified_artifacts: usize,
    pub outcomes: BTreeMap<BenchmarkOutcome, usize>,
    pub failures: Vec<FailedRequest>,
    pub latency: BTreeMap<BenchmarkOutcome, Latency>,
}

pub fn inspect_file(path: &Path, mode: FixtureMode) -> Result<Inspection, &'static str> {
    let path = path.canonicalize().map_err(|_| "BENCHMARK_READ_FAILED")?;
    if !path.is_file() {
        return Err("BENCHMARK_REGULAR_FILE_REQUIRED");
    }
    let bytes = crate::scoreboard::read(
        File::open(&path).map_err(|_| "BENCHMARK_READ_FAILED")?,
        starlings_protocol::MAX_JSON_BYTES,
    )
    .map_err(|_| "BENCHMARK_READ_FAILED")?;
    let run: BenchmarkRun = decode(&bytes, mode).map_err(|_| "BENCHMARK_RECORD_REFUSED")?;
    let artifacts = [
        crate::scoreboard::Artifact {
            path: run.hardware_inventory_path,
            sha256: run.hardware_inventory_sha256,
        },
        crate::scoreboard::Artifact {
            path: run.raw_logs_path,
            sha256: run.raw_logs_sha256,
        },
    ];
    crate::scoreboard::artifacts(&artifacts, path.parent().ok_or("BENCHMARK_READ_FAILED")?)
        .map_err(|_| "BENCHMARK_EVIDENCE_REFUSED")?;
    let mut outcomes: BTreeMap<_, _> = [
        BenchmarkOutcome::Success,
        BenchmarkOutcome::Oom,
        BenchmarkOutcome::Cancelled,
        BenchmarkOutcome::Timeout,
        BenchmarkOutcome::Error,
    ]
    .into_iter()
    .map(|outcome| (outcome, 0))
    .collect();
    let mut failures = Vec::new();
    let latency = outcomes
        .keys()
        .map(|&outcome| {
            let matching = run.requests.iter().filter(|r| r.outcome == outcome);
            let durations: Vec<_> = matching.clone().map(|r| r.duration_ms).collect();
            let ttfts: Vec<_> = matching.filter_map(|r| r.ttft_ms).collect();
            (
                outcome,
                Latency {
                    duration_ms: (!durations.is_empty())
                        .then(|| crate::scoreboard::summary(durations)),
                    ttft_ms: (!ttfts.is_empty()).then(|| crate::scoreboard::summary(ttfts)),
                },
            )
        })
        .collect();
    for request in run.requests {
        *outcomes.entry(request.outcome).or_default() += 1;
        if let Some(error_code) = request.error_code {
            failures.push(FailedRequest {
                request_id: request.request_id,
                outcome: request.outcome,
                error_code,
            });
        }
    }
    Ok(Inspection {
        version: Version,
        fixture: run.fixture,
        qualification: "not_assessed",
        run_id: run.run_id,
        run_status: run.status,
        record_sha256: byte_digest(&bytes),
        verified_artifacts: artifacts.len(),
        outcomes,
        failures,
        latency,
    })
}
