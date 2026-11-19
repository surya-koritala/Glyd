//! Offline comparison of supplied evidence. Never launches or qualifies a backend.
use serde::{Deserialize, Serialize};
use starlings_protocol::{
    Digest, FixtureMode, InvocationId, Version, byte_digest, content_hash, parse_json,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::{Component, Path},
};

type Result<T> = std::result::Result<T, &'static str>;
const MAX_ARTIFACT: usize = 16 * 1024 * 1024;
const MAX_TOTAL: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    DecodeTokensPerSecond,
    PrefillTokensPerSecond,
    TtftMilliseconds,
    ScalingRatio,
    ConcurrentTokensPerSecond,
    PeakResidentBytes,
    LargestModelParameters,
    MaximumContextTokens,
    QualityKl,
    QualityTop1Fraction,
    QualityPerplexity,
    SetupMilliseconds,
    SetupManualSteps,
    ReliabilitySuccesses20,
    RecoveryMilliseconds,
    CancelRetainedBytes,
    ApiPassFraction,
    ModelFamiliesCount,
}
impl Metric {
    pub const ALL: [Self; 18] = [
        Self::DecodeTokensPerSecond,
        Self::PrefillTokensPerSecond,
        Self::TtftMilliseconds,
        Self::ScalingRatio,
        Self::ConcurrentTokensPerSecond,
        Self::PeakResidentBytes,
        Self::LargestModelParameters,
        Self::MaximumContextTokens,
        Self::QualityKl,
        Self::QualityTop1Fraction,
        Self::QualityPerplexity,
        Self::SetupMilliseconds,
        Self::SetupManualSteps,
        Self::ReliabilitySuccesses20,
        Self::RecoveryMilliseconds,
        Self::CancelRetainedBytes,
        Self::ApiPassFraction,
        Self::ModelFamiliesCount,
    ];
    pub fn unit(self) -> &'static str {
        match self {
            Self::DecodeTokensPerSecond
            | Self::PrefillTokensPerSecond
            | Self::ConcurrentTokensPerSecond => "tokens_per_second",
            Self::TtftMilliseconds | Self::SetupMilliseconds | Self::RecoveryMilliseconds => {
                "milliseconds"
            }
            Self::PeakResidentBytes | Self::CancelRetainedBytes => "bytes",
            Self::QualityTop1Fraction | Self::ApiPassFraction => "fraction",
            Self::ScalingRatio => "ratio",
            Self::QualityKl => "kl_divergence",
            Self::QualityPerplexity => "perplexity",
            _ => "count",
        }
    }
    fn lower(self) -> bool {
        matches!(
            self,
            Self::TtftMilliseconds
                | Self::PeakResidentBytes
                | Self::QualityKl
                | Self::QualityPerplexity
                | Self::SetupMilliseconds
                | Self::SetupManualSteps
                | Self::RecoveryMilliseconds
                | Self::CancelRetainedBytes
        )
    }
    fn speed(self) -> bool {
        matches!(
            self,
            Self::DecodeTokensPerSecond
                | Self::PrefillTokensPerSecond
                | Self::TtftMilliseconds
                | Self::ScalingRatio
                | Self::ConcurrentTokensPerSecond
                | Self::SetupMilliseconds
                | Self::RecoveryMilliseconds
        )
    }
    fn value(self, value: f64) -> bool {
        value.is_finite()
            && (0.0..=1e15).contains(&value)
            && (!matches!(self.unit(), "count" | "bytes") || value.fract() == 0.0)
            && (self.unit() != "fraction" || value <= 1.0)
            && (self != Self::ReliabilitySuccesses20 || value <= 20.0)
            && (!self.speed() || value > 0.0)
            && (self != Self::QualityPerplexity || value >= 1.0)
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub path: String,
    pub sha256: Digest,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Program {
    pub repository: String,
    pub source_commit: String,
    pub build: Digest,
    pub recipe: Digest,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Programs {
    pub exo: Option<Program>,
    pub glyd: Option<Program>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Matched {
    pub model: Digest,
    pub profile: Digest,
    pub hardware: Digest,
    pub workload: Digest,
    pub definition: Digest,
    pub parameters: Digest,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum System {
    Exo,
    Glyd,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum Outcome {
    Measured { value: f64 },
    Failed { code: String },
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub fixture: bool,
    pub id: InvocationId,
    pub system: System,
    pub repeat: u8,
    pub program_digest: Digest,
    pub match_digest: Digest,
    pub unit: String,
    pub raw: Digest,
    pub outcome: Outcome,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cell {
    pub id: String,
    pub metric: Metric,
    pub matched: Option<Matched>,
    pub unavailable: Option<String>,
    pub observations: Vec<Observation>,
}
impl Cell {
    pub fn binding(&self) -> Result<Digest> {
        content_hash(&(&self.id, self.metric, &self.matched))
            .map_err(|_| "SCOREBOARD_INVALID_BINDING")
    }
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Matrix {
    pub version: Version,
    pub fixture: bool,
    pub programs: Programs,
    pub artifacts: Vec<Artifact>,
    pub cells: Vec<Cell>,
}
#[derive(Serialize)]
pub struct Summary {
    pub samples: usize,
    pub minimum: f64,
    pub median: f64,
    pub maximum: f64,
    pub p95: Option<f64>,
    pub p99: Option<f64>,
}
#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Won,
    Lost,
    NotMeasured,
    Fixture,
}
#[derive(Serialize)]
pub struct Failure {
    pub id: InvocationId,
    pub system: System,
    pub repeat: u8,
    pub code: String,
}
#[derive(Serialize)]
pub struct CellReport {
    pub id: String,
    pub metric: Metric,
    pub unit: &'static str,
    pub status: Status,
    pub reason: String,
    pub binding: Option<Digest>,
    pub rule_satisfied: Option<bool>,
    pub exo: Option<Summary>,
    pub glyd: Option<Summary>,
    pub failures: Vec<Failure>,
}
#[derive(Serialize)]
pub struct Report {
    pub version: Version,
    pub fixture: bool,
    pub policy: &'static str,
    pub qualification: &'static str,
    pub program_digests: BTreeMap<System, Digest>,
    pub matrix_digest: Digest,
    pub cells: Vec<CellReport>,
}

pub(crate) fn read(file: File, limit: usize) -> Result<Vec<u8>> {
    if !file
        .metadata()
        .map_err(|_| "SCOREBOARD_READ_FAILED")?
        .is_file()
    {
        return Err("SCOREBOARD_REGULAR_FILE_REQUIRED");
    }
    let mut bytes = Vec::new();
    file.take(limit as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| "SCOREBOARD_READ_FAILED")?;
    if bytes.len() > limit {
        return Err("SCOREBOARD_BYTE_LIMIT");
    }
    Ok(bytes)
}
pub(crate) fn artifacts(items: &[Artifact], root: &Path) -> Result<BTreeSet<Digest>> {
    if items.len() > 512 {
        return Err("SCOREBOARD_ARTIFACT_LIMIT");
    }
    let root = root.canonicalize().map_err(|_| "SCOREBOARD_READ_FAILED")?;
    let mut seen = BTreeSet::new();
    let mut paths = BTreeSet::new();
    let mut remaining = MAX_TOTAL;
    for item in items {
        let path = Path::new(&item.path);
        if item.path.is_empty()
            || item.path.len() > 256
            || item.path.chars().any(char::is_control)
            || path
                .components()
                .any(|p| !matches!(p, Component::Normal(_)))
        {
            return Err("SCOREBOARD_RELATIVE_ARTIFACT_REQUIRED");
        }
        let path = root
            .join(path)
            .canonicalize()
            .map_err(|_| "SCOREBOARD_READ_FAILED")?;
        if !path.starts_with(&root)
            || !paths.insert(path.clone())
            || !seen.insert(item.sha256.clone())
        {
            return Err("SCOREBOARD_ARTIFACT_CONFLICT");
        }
        if !path.is_file() {
            return Err("SCOREBOARD_REGULAR_FILE_REQUIRED");
        }
        let bytes = read(
            File::open(path).map_err(|_| "SCOREBOARD_READ_FAILED")?,
            MAX_ARTIFACT.min(remaining),
        )?;
        remaining -= bytes.len();
        if byte_digest(&bytes) != item.sha256 {
            return Err("SCOREBOARD_ARTIFACT_MISMATCH");
        }
    }
    Ok(seen)
}
fn label(value: &str, maximum: usize) -> bool {
    !value.is_empty()
        && value.len() <= maximum
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b" _-./:".contains(&b))
}
pub(crate) fn summary(mut values: Vec<f64>) -> Summary {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Summary {
        samples: values.len(),
        minimum: values[0],
        maximum: values[values.len() - 1],
        median: if values.len().is_multiple_of(2) {
            values[middle - 1].midpoint(values[middle])
        } else {
            values[middle]
        },
        p95: (values.len() >= 20).then(|| values[(values.len() * 95).div_ceil(100) - 1]),
        p99: (values.len() >= 100).then(|| values[(values.len() * 99).div_ceil(100) - 1]),
    }
}

pub fn report_file(path: &Path, mode: FixtureMode) -> Result<Report> {
    let path = path.canonicalize().map_err(|_| "SCOREBOARD_READ_FAILED")?;
    if !path.is_file() {
        return Err("SCOREBOARD_REGULAR_FILE_REQUIRED");
    }
    let bytes = read(
        File::open(&path).map_err(|_| "SCOREBOARD_READ_FAILED")?,
        starlings_protocol::MAX_JSON_BYTES,
    )?;
    let matrix: Matrix =
        serde_json::from_value(parse_json(&bytes).map_err(|_| "SCOREBOARD_INVALID_JSON")?)
            .map_err(|_| "SCOREBOARD_INVALID_RECORD")?;
    compare(
        &matrix,
        path.parent().ok_or("SCOREBOARD_READ_FAILED")?,
        mode,
    )
}

pub(crate) fn compare(matrix: &Matrix, root: &Path, mode: FixtureMode) -> Result<Report> {
    if matrix.fixture && mode == FixtureMode::Reject {
        return Err("SCOREBOARD_FIXTURE_REFUSED");
    }
    if matrix.cells.len() > 64
        || matrix
            .cells
            .iter()
            .map(|c| c.observations.len())
            .sum::<usize>()
            > 512
    {
        return Err("SCOREBOARD_CELL_LIMIT");
    }
    let present: BTreeSet<_> = matrix.cells.iter().map(|c| c.metric).collect();
    if Metric::ALL.iter().any(|m| !present.contains(m)) {
        return Err("SCOREBOARD_PLAN_INCOMPLETE");
    }
    let hashes = artifacts(&matrix.artifacts, root)?;
    let require = |digest: &Digest| {
        if hashes.contains(digest) {
            Ok(())
        } else {
            Err("SCOREBOARD_EVIDENCE_MISSING")
        }
    };
    let mut programs = BTreeMap::new();
    for (system, program) in [
        (System::Exo, &matrix.programs.exo),
        (System::Glyd, &matrix.programs.glyd),
    ] {
        if let Some(program) = program {
            let repository = program
                .repository
                .strip_prefix("https://github.com/")
                .ok_or("SCOREBOARD_SOURCE_PIN_REQUIRED")?;
            let mut parts = repository.split('/');
            let pair = [
                parts.next().unwrap_or_default(),
                parts.next().unwrap_or_default(),
            ];
            if parts.next().is_some()
                || pair.iter().any(|p| {
                    p.is_empty()
                        || p.len() > 128
                        || *p == "."
                        || *p == ".."
                        || !p
                            .bytes()
                            .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
                })
                || (system == System::Exo
                    && program.repository != "https://github.com/exo-explore/exo")
            {
                return Err("SCOREBOARD_SOURCE_PIN_REQUIRED");
            }
            if program.source_commit.len() != 40
                || !program
                    .source_commit
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            {
                return Err("SCOREBOARD_SOURCE_PIN_REQUIRED");
            }
            require(&program.build)?;
            require(&program.recipe)?;
            programs.insert(
                system,
                content_hash(&(system, program)).map_err(|_| "SCOREBOARD_INVALID_BINDING")?,
            );
        }
    }
    let mut ids = BTreeSet::new();
    let mut cells = Vec::new();
    for cell in &matrix.cells {
        if !label(&cell.id, 64) || !ids.insert(&cell.id) {
            return Err("SCOREBOARD_CELL_CONFLICT");
        }
        let mut row = CellReport {
            id: cell.id.clone(),
            metric: cell.metric,
            unit: cell.metric.unit(),
            status: Status::NotMeasured,
            reason: String::new(),
            binding: None,
            rule_satisfied: None,
            exo: None,
            glyd: None,
            failures: Vec::new(),
        };
        if let Some(matched) = &cell.matched {
            for digest in [
                &matched.model,
                &matched.profile,
                &matched.hardware,
                &matched.workload,
                &matched.definition,
                &matched.parameters,
            ] {
                require(digest)?;
            }
            row.binding = Some(cell.binding()?);
        }
        if let Some(reason) = &cell.unavailable {
            if !label(reason, 256) || !cell.observations.is_empty() {
                return Err("SCOREBOARD_UNAVAILABLE_CONFLICT");
            }
            row.reason = reason.clone();
            cells.push(row);
            continue;
        }
        let binding = row.binding.as_ref().ok_or("SCOREBOARD_MATCH_REQUIRED")?;
        let mut observations = BTreeSet::new();
        let mut repeats = BTreeSet::new();
        let mut exo = BTreeMap::new();
        let mut glyd = BTreeMap::new();
        for observation in &cell.observations {
            if observation.fixture != matrix.fixture {
                return Err("SCOREBOARD_FIXTURE_CONFLICT");
            }
            if !(1..=32).contains(&observation.repeat)
                || !observations.insert(observation.id)
                || !repeats.insert((observation.system, observation.repeat))
            {
                return Err("SCOREBOARD_REPEAT_CONFLICT");
            }
            if &observation.match_digest != binding
                || observation.unit != cell.metric.unit()
                || programs.get(&observation.system) != Some(&observation.program_digest)
            {
                return Err("SCOREBOARD_UNMATCHED_OBSERVATION");
            }
            require(&observation.raw)?;
            match &observation.outcome {
                Outcome::Measured { value } => {
                    if !cell.metric.value(*value) {
                        return Err("SCOREBOARD_INVALID_VALUE");
                    }
                    match observation.system {
                        System::Exo => &mut exo,
                        System::Glyd => &mut glyd,
                    }
                    .insert(observation.repeat, *value);
                }
                Outcome::Failed { code } => {
                    if !label(code, 64) {
                        return Err("SCOREBOARD_INVALID_ERROR");
                    }
                    row.failures.push(Failure {
                        id: observation.id,
                        system: observation.system,
                        repeat: observation.repeat,
                        code: code.clone(),
                    });
                }
            }
        }
        if !row.failures.is_empty() {
            row.reason = "failed observations retained".into();
        } else if exo.len() < 5 || !exo.keys().eq(glyd.keys()) {
            row.reason = "at least five paired repeats required".into();
        } else {
            let baseline = summary(exo.into_values().collect());
            let candidate = summary(glyd.into_values().collect());
            let nonoverlap = if cell.metric.lower() {
                candidate.maximum < baseline.minimum
            } else {
                candidate.minimum > baseline.maximum
            };
            let margin = if !cell.metric.speed() {
                true
            } else if cell.metric.lower() {
                candidate.median * 10.0 <= baseline.median * 9.0
            } else {
                candidate.median * 10.0 >= baseline.median * 11.0
            };
            let won = nonoverlap && margin;
            row.status = if matrix.fixture {
                Status::Fixture
            } else if won {
                Status::Won
            } else {
                Status::Lost
            };
            row.reason = if matrix.fixture {
                "synthetic evidence only"
            } else if won {
                "comparison rule satisfied"
            } else {
                "comparison rule not satisfied"
            }
            .into();
            row.rule_satisfied = Some(won);
            row.exo = Some(baseline);
            row.glyd = Some(candidate);
        }
        cells.push(row);
    }
    Ok(Report {
        version: Version,
        fixture: matrix.fixture,
        policy: "gly234-conservative-v1",
        qualification: "not_assessed",
        program_digests: programs,
        matrix_digest: content_hash(matrix).map_err(|_| "SCOREBOARD_INVALID_BINDING")?,
        cells,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn observed_tails_require_samples_and_medians_remain_finite() {
        for (count, p95, p99) in [
            (1, None, None),
            (19, None, None),
            (20, Some(19.0), None),
            (99, Some(95.0), None),
            (100, Some(95.0), Some(99.0)),
        ] {
            let result = summary((1..=count).rev().map(f64::from).collect());
            assert_eq!(result.samples, count as usize);
            assert_eq!(result.minimum, 1.0);
            assert_eq!(result.maximum, f64::from(count));
            assert_eq!(result.median, f64::from(count + 1) / 2.0);
            assert_eq!(result.p95, p95);
            assert_eq!(result.p99, p99);
        }
        let extreme = summary(vec![f64::MAX, f64::MAX]);
        assert_eq!(extreme.median, f64::MAX);
        assert!(extreme.p95.is_none() && extreme.p99.is_none());
    }
    fn plan() -> Matrix {
        Matrix {
            version: Version,
            fixture: false,
            programs: Programs {
                exo: None,
                glyd: None,
            },
            artifacts: vec![],
            cells: Metric::ALL
                .into_iter()
                .enumerate()
                .map(|(i, metric)| Cell {
                    id: format!("planned_{i}"),
                    metric,
                    matched: None,
                    unavailable: Some(
                        "not run - qualified hardware and workload approval pending".into(),
                    ),
                    observations: vec![],
                })
                .collect(),
        }
    }
    fn fixture(metric: Metric, baseline: f64, candidate: f64) -> (tempfile::TempDir, Matrix) {
        let temp = tempfile::tempdir().unwrap();
        let raw = b"explicit synthetic evidence; not a native benchmark";
        std::fs::write(temp.path().join("raw.txt"), raw).unwrap();
        let digest = byte_digest(raw);
        let mut matrix = plan();
        matrix.fixture = true;
        matrix.artifacts.push(Artifact {
            path: "raw.txt".into(),
            sha256: digest.clone(),
        });
        matrix.programs = Programs {
            exo: Some(Program {
                repository: "https://github.com/exo-explore/exo".into(),
                source_commit: "a".repeat(40),
                build: digest.clone(),
                recipe: digest.clone(),
            }),
            glyd: Some(Program {
                repository: "https://github.com/Sigbound/starlings".into(),
                source_commit: "b".repeat(40),
                build: digest.clone(),
                recipe: digest.clone(),
            }),
        };
        let cell = matrix
            .cells
            .iter_mut()
            .find(|c| c.metric == metric)
            .unwrap();
        cell.unavailable = None;
        cell.matched = Some(Matched {
            model: digest.clone(),
            profile: digest.clone(),
            hardware: digest.clone(),
            workload: digest.clone(),
            definition: digest.clone(),
            parameters: digest.clone(),
        });
        for (system, program, value) in [
            (System::Exo, matrix.programs.exo.as_ref().unwrap(), baseline),
            (
                System::Glyd,
                matrix.programs.glyd.as_ref().unwrap(),
                candidate,
            ),
        ] {
            for repeat in 1..=5 {
                cell.observations.push(Observation {
                    fixture: true,
                    id: uuid::Uuid::new_v4().to_string().parse().unwrap(),
                    system,
                    repeat,
                    program_digest: content_hash(&(system, program)).unwrap(),
                    match_digest: cell.binding().unwrap(),
                    unit: metric.unit().into(),
                    raw: digest.clone(),
                    outcome: Outcome::Measured { value },
                });
            }
        }
        (temp, matrix)
    }
    fn cell(matrix: &mut Matrix) -> &mut Cell {
        matrix
            .cells
            .iter_mut()
            .find(|c| c.unavailable.is_none())
            .unwrap()
    }
    fn row(report: &Report) -> &CellReport {
        report
            .cells
            .iter()
            .find(|r| r.rule_satisfied.is_some())
            .unwrap()
    }
    fn error(matrix: &Matrix, root: &Path) -> &'static str {
        compare(matrix, root, FixtureMode::Simulation)
            .err()
            .expect("invalid record accepted")
    }

    #[test]
    fn unmeasured_plan_is_complete_and_fixture_defaults_are_closed() {
        let temp = tempfile::tempdir().unwrap();
        let matrix = plan();
        let report = compare(&matrix, temp.path(), FixtureMode::Reject).unwrap();
        assert_eq!(report.cells.len(), Metric::ALL.len());
        assert!(
            report
                .cells
                .iter()
                .all(|r| r.status == Status::NotMeasured && r.rule_satisfied.is_none())
        );
        assert_eq!(report.qualification, "not_assessed");
        let (temp, mut matrix) = fixture(Metric::DecodeTokensPerSecond, 100.0, 110.0);
        assert_eq!(
            compare(&matrix, temp.path(), FixtureMode::Reject).err(),
            Some("SCOREBOARD_FIXTURE_REFUSED")
        );
        let report = compare(&matrix, temp.path(), FixtureMode::Simulation).unwrap();
        assert_eq!(row(&report).status, Status::Fixture);
        assert_eq!(row(&report).rule_satisfied, Some(true));
        assert!(!report.cells.iter().any(|r| r.status == Status::Won));
        matrix.fixture = false;
        assert_eq!(error(&matrix, temp.path()), "SCOREBOARD_FIXTURE_CONFLICT");
    }
    #[test]
    fn rules_require_margin_nonoverlap_and_both_directions() {
        for (metric, baseline, candidate) in [
            (Metric::DecodeTokensPerSecond, 100.0, 110.0),
            (Metric::TtftMilliseconds, 100.0, 90.0),
            (Metric::PeakResidentBytes, 1024.0, 512.0),
            (Metric::ModelFamiliesCount, 10.0, 11.0),
        ] {
            let (temp, matrix) = fixture(metric, baseline, candidate);
            assert_eq!(
                row(&compare(&matrix, temp.path(), FixtureMode::Simulation).unwrap())
                    .rule_satisfied,
                Some(true)
            );
            let (temp, matrix) = fixture(metric, candidate, baseline);
            assert_eq!(
                row(&compare(&matrix, temp.path(), FixtureMode::Simulation).unwrap())
                    .rule_satisfied,
                Some(false)
            );
        }
        let (temp, mut matrix) = fixture(Metric::DecodeTokensPerSecond, 100.0, 110.0);
        cell(&mut matrix).observations[0].outcome = Outcome::Measured { value: 120.0 };
        assert_eq!(
            row(&compare(&matrix, temp.path(), FixtureMode::Simulation).unwrap()).rule_satisfied,
            Some(false)
        );
        let (temp, matrix) = fixture(Metric::DecodeTokensPerSecond, 100.0, 109.0);
        assert_eq!(
            row(&compare(&matrix, temp.path(), FixtureMode::Simulation).unwrap()).rule_satisfied,
            Some(false)
        );
    }
    #[test]
    fn failed_partial_duplicate_and_unmatched_runs_cannot_win() {
        let (temp, matrix) = fixture(Metric::DecodeTokensPerSecond, 100.0, 120.0);
        let mut partial = matrix.clone();
        cell(&mut partial).observations.pop();
        let report = compare(&partial, temp.path(), FixtureMode::Simulation).unwrap();
        assert!(report.cells.iter().all(|r| r.status == Status::NotMeasured));
        let mut failed = matrix.clone();
        cell(&mut failed).observations[0].outcome = Outcome::Failed { code: "OOM".into() };
        let report = compare(&failed, temp.path(), FixtureMode::Simulation).unwrap();
        let failed = report
            .cells
            .iter()
            .find(|r| !r.failures.is_empty())
            .unwrap();
        assert_eq!(failed.status, Status::NotMeasured);
        assert_eq!(failed.failures[0].code, "OOM");
        assert!(failed.exo.is_none() && failed.glyd.is_none());
        let mut duplicate = matrix.clone();
        cell(&mut duplicate).observations[1].repeat = 1;
        assert_eq!(error(&duplicate, temp.path()), "SCOREBOARD_REPEAT_CONFLICT");
        let mut unmatched = matrix.clone();
        cell(&mut unmatched).observations[0].unit = "seconds".into();
        assert_eq!(
            error(&unmatched, temp.path()),
            "SCOREBOARD_UNMATCHED_OBSERVATION"
        );
        let mut build = matrix.clone();
        cell(&mut build).observations[0].program_digest =
            byte_digest(b"different compiled build or recipe");
        assert_eq!(
            error(&build, temp.path()),
            "SCOREBOARD_UNMATCHED_OBSERVATION"
        );
        cell(&mut unmatched).observations[0].unit = "tokens_per_second".into();
        cell(&mut unmatched).observations[0].match_digest =
            byte_digest(b"different model/profile/hardware/definition");
        assert_eq!(
            error(&unmatched, temp.path()),
            "SCOREBOARD_UNMATCHED_OBSERVATION"
        );
        let mut invalid = matrix.clone();
        cell(&mut invalid).observations[0].outcome = Outcome::Measured { value: f64::NAN };
        assert_eq!(error(&invalid, temp.path()), "SCOREBOARD_INVALID_VALUE");
        let mut missing = matrix.clone();
        missing.cells.pop();
        assert_eq!(error(&missing, temp.path()), "SCOREBOARD_PLAN_INCOMPLETE");
    }
    #[test]
    fn artifacts_and_strict_files_are_verified_before_reporting() {
        let (temp, mut matrix) = fixture(Metric::DecodeTokensPerSecond, 100.0, 120.0);
        let file = temp.path().join("matrix.json");
        std::fs::write(&file, serde_json::to_vec(&matrix).unwrap()).unwrap();
        assert!(report_file(&file, FixtureMode::Simulation).is_ok());
        std::fs::write(temp.path().join("raw.txt"), b"tampered").unwrap();
        assert_eq!(error(&matrix, temp.path()), "SCOREBOARD_ARTIFACT_MISMATCH");
        matrix.artifacts[0].path = "../outside.txt".into();
        assert_eq!(
            error(&matrix, temp.path()),
            "SCOREBOARD_RELATIVE_ARTIFACT_REQUIRED"
        );
        let mut json = serde_json::to_value(plan()).unwrap();
        json["unexpected"] = serde_json::json!(true);
        std::fs::write(&file, serde_json::to_vec(&json).unwrap()).unwrap();
        assert_eq!(
            report_file(&file, FixtureMode::Reject).err(),
            Some("SCOREBOARD_INVALID_RECORD")
        );
        std::fs::write(&file, b"{\"version\":1,\"version\":1}").unwrap();
        assert_eq!(
            report_file(&file, FixtureMode::Reject).err(),
            Some("SCOREBOARD_INVALID_JSON")
        );
        assert_eq!(
            report_file(temp.path(), FixtureMode::Reject).err(),
            Some("SCOREBOARD_REGULAR_FILE_REQUIRED")
        );
        matrix.artifacts[0].path = "raw.txt".into();
        File::create(temp.path().join("raw.txt"))
            .unwrap()
            .set_len(MAX_ARTIFACT as u64 + 1)
            .unwrap();
        assert_eq!(error(&matrix, temp.path()), "SCOREBOARD_BYTE_LIMIT");
    }
}
