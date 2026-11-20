//! Operator helpers for a pipeline set up by hand across machines (the daemon test does these
//! steps in Rust). Development tool: it starts the engine directly and approves nothing.
//!
//!   pipeline_tool describe STAGE.json PRIVATE_DIR
//!       the stage's `describe` digest, model digest and layer count (loads nothing)
//!   pipeline_tool plan PLAN.json
//!       the plan with its canonical `plan_digest`
//!   pipeline_tool reference STAGE.json PRIVATE_DIR TEXT NEW TEMPERATURE TOP_P SEED STOP_IDS
//!       the whole model (layers 0..all) on this machine: prompt and token digests
use starlings_adapters::partial::{
    RunChunk, StepIdentity,
    glyd::{self, Event, Output},
};
use starlings_cli::routes::StageRoute;
use starlings_protocol::*;
use std::{path::Path, time::Instant};

fn digest(ids: &[u32]) -> Digest {
    byte_digest(&ids.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<_>>())
}
fn spawn(
    stage: &StageRoute,
    private: &Path,
) -> (
    glyd::RangeProcess,
    std::process::ChildStdout,
    std::process::ChildStdin,
    serde_json::Value,
) {
    starlings_cli::model_files::verify(&stage.model, &stage.model_files_sha256)
        .expect("approved model files");
    let (process, mut stdout, mut stdin) = glyd::RangeProcess::spawn(
        &stage.executable,
        &stage.executable_sha256,
        &stage.arguments(),
        &stage.environment,
        stage.gpu_library_sha256.as_ref(),
        private,
    )
    .expect("engine launch");
    let d = glyd::describe(&mut stdout, &mut stdin).expect("describe");
    (process, stdout, stdin, d)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let read = |path: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(path).expect("read")).expect("json")
    };
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["describe", stage, private] => {
            let stage: StageRoute = serde_json::from_value(read(stage)).expect("stage route");
            let (process, _, _, d) = spawn(&stage, Path::new(private));
            process.stop();
            println!(
                "{}",
                serde_json::json!({"descriptor": glyd::descriptor_digest(&d).unwrap(),
                    "model": stage.model_files_sha256, "layers": d["model"]["layers"]})
            );
        }
        ["plan", plan] => {
            let mut plan: PlacementPlan = serde_json::from_value(read(plan)).expect("plan");
            plan.plan_digest =
                canonical_digest(&serde_json::to_value(&plan).unwrap(), "plan_digest").unwrap();
            encode(&plan, FixtureMode::Reject).expect("valid plan");
            println!("{}", serde_json::to_string(&plan).unwrap());
        }
        [
            "reference",
            stage,
            private,
            text,
            new,
            temperature,
            top_p,
            seed,
            stop,
        ] => {
            let mut stage: StageRoute = serde_json::from_value(read(stage)).expect("stage route");
            let (new, temperature, top_p, seed): (u32, f64, f64, u32) = (
                new.parse().unwrap(),
                temperature.parse().unwrap(),
                top_p.parse().unwrap(),
                seed.parse().unwrap(),
            );
            let stop: Vec<u32> = stop.split(',').filter_map(|s| s.parse().ok()).collect();
            // One session, whatever the stage route serves.
            (stage.sessions, stage.more_data_listen) = (1, Vec::new());
            let (probe, _, _, d) = spawn(&stage, Path::new(private));
            probe.stop();
            stage.layers = [0, d["model"]["layers"].as_u64().unwrap() as u32];
            let (process, stdout, stdin, d) = spawn(&stage, Path::new(private));
            let workload = ProfileWorkload {
                context_tokens: BoundedU32::new(stage.context_tokens).unwrap(),
                prompt_tokens: BoundedU32::new(stage.max_prompt_tokens).unwrap(),
                output_tokens: BoundedU32::new(stage.max_output_tokens).unwrap(),
                prefill_chunk_tokens: BoundedU32::new(stage.chunk_tokens).unwrap(),
                sessions: BoundedU32::new(1).unwrap(),
            };
            let (mut contract, first, last, vocabulary) = glyd::qwen_boundary(
                &d,
                &stage.executable_sha256,
                &stage.execution_profile,
                &stage.boundary_profile,
                workload,
            )
            .unwrap();
            contract.key.manifest_digest = stage.model_files_sha256.clone();
            let mut worker = glyd::Worker::new(
                stdout,
                stdin,
                glyd::Approval {
                    descriptor: glyd::descriptor_digest(&d).unwrap(),
                    boundary: contract.clone(),
                    engine_bytes_limit: stage.engine_bytes_limit.0,
                    payload_bytes_limit: 128 << 20,
                    first,
                    last,
                    vocabulary,
                },
                FixtureMode::Reject,
            )
            .unwrap();
            worker.describe().unwrap();
            assert!(matches!(worker.receive().unwrap(), Event::Described));
            worker.prepare().unwrap();
            assert!(matches!(worker.receive().unwrap(), Event::Prepared(_)));
            worker.commit().unwrap();
            assert!(matches!(worker.receive().unwrap(), Event::Committed));
            worker.tokenize(text).unwrap();
            let Event::Tokens(prompt) = worker.receive().unwrap() else {
                panic!("tokenize")
            };
            let mut template: TensorHeader = decode(
                include_bytes!("../../../examples/tensor-header.json"),
                FixtureMode::Simulation,
            )
            .unwrap();
            template.plan_digest = byte_digest(b"whole-model reference run");
            worker
                .begin_session(
                    (&template).into(),
                    prompt.len() as u32,
                    new,
                    Some(Sampling {
                        profile: "sample-v1".into(),
                        temperature,
                        top_p,
                        seed: BoundedU32::new(seed).unwrap(),
                    }),
                )
                .unwrap();
            let Event::Begun(session) = worker.receive().unwrap() else {
                panic!("begin_session")
            };
            let mut step = 0;
            let mut run = |start: u32, ids: &[u32], head: bool| {
                let mut scope = DataScope::from(&template);
                scope.invocation_id = uuid::Uuid::new_v4().to_string().parse().unwrap();
                scope.sequence = Decimal(step);
                step += 1;
                let rows = BoundedU32::new(ids.len() as u32).unwrap();
                let call = RunChunk {
                    version: Version,
                    inputs: contract.layouts(rows.get(), 128 << 20).unwrap().0,
                    step: StepIdentity {
                        scope,
                        phase: if start < prompt.len() as u32 {
                            TensorHeaderPhase::Prefill
                        } else {
                            TensorHeaderPhase::Decode
                        },
                        position_start: BoundedU32::new(start).unwrap(),
                        token_count: rows,
                    },
                };
                let bytes: Vec<u8> = ids.iter().flat_map(|t| t.to_le_bytes()).collect();
                worker
                    .run_chunk(&session, &call, &call.step, &[bytes], head, false)
                    .unwrap();
                match worker.receive().unwrap() {
                    Event::Chunk { output, .. } => output.map(|o| match o {
                        Output::Token { token, .. } => token,
                        other => panic!("{other:?}"),
                    }),
                    other => panic!("{other:?}"),
                }
            };
            let started = Instant::now();
            let mut tokens = Vec::new();
            let chunk = stage.chunk_tokens;
            for (i, ids) in prompt.chunks(chunk as usize).enumerate() {
                let start = i as u32 * chunk;
                let head = start as usize + ids.len() == prompt.len();
                if let Some(token) = run(start, ids, head) {
                    tokens.push(token);
                }
            }
            while (tokens.len() as u32) < new && !stop.contains(tokens.last().unwrap()) {
                let start = prompt.len() as u32 + tokens.len() as u32 - 1;
                let token = run(start, &[*tokens.last().unwrap()], true).unwrap();
                tokens.push(token);
            }
            let ms = started.elapsed().as_secs_f64() * 1e3;
            // The answer's text, to compare with a release daemon's reply (its report has no digest).
            worker.detokenize(&tokens).unwrap();
            let Event::Text(text) = worker.receive().unwrap() else {
                panic!("detokenize")
            };
            process.stop();
            println!(
                "{}",
                serde_json::json!({"prompt_tokens": prompt.len(), "prompt_sha256": digest(&prompt),
                    "tokens": tokens.len(), "tokens_sha256": digest(&tokens),
                    "stopped": stop.contains(tokens.last().unwrap()), "ms": ms, "text": text})
            );
        }
        _ => {
            eprintln!("usage: see the header of examples/pipeline_tool.rs");
            std::process::exit(2);
        }
    }
}
