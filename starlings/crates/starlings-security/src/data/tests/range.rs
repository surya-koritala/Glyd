//! Two enrolled nodes run one model through real `glyd-engine rangeworker` processes (glyd-range/1).
//! Each step's boundary crosses enrolled TLS as one GCP1 tensor, and every token and logit must
//! equal the whole model's, bit for bit. Runs only with STARLINGS_RANGE_ENGINE and
//! STARLINGS_RANGE_MODEL set; the caller holds the GPU lock. The sampled token returns to stage 0
//! through this coordinator, not over the data channel.
use super::*;
use serde_json::{Value, json};
use starlings_adapters::partial::{
    BoundaryContract, RunChunk, StepIdentity,
    glyd::{Approval, Event, Output, Worker, descriptor_digest},
};
use std::{
    io::{Read, Write},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
};

const CHUNK: u32 = 32;
const PAYLOAD: u64 = 128 << 20;

struct Stage {
    worker: Worker<ChildStdout, ChildStdin>,
    contract: BoundaryContract,
    child: Child,
    workload: (u32, u32),
    /// The engine's handle for the open session.
    session: String,
}
/// Starts a worker and reads its descriptor; it loads nothing before `prepare`.
fn launch(
    engine: &str,
    model: &str,
    layers: (u32, u32),
    ctx: u32,
) -> (Child, ChildStdin, ChildStdout, Value) {
    let mut child = Command::new(engine)
        .args(["rangeworker", "--model", model, "--layers"])
        .arg(format!("{}..{}", layers.0, layers.1))
        .args(["--ctx", &ctx.to_string(), "--chunk", &CHUNK.to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let (mut stdin, mut stdout) = (child.stdin.take().unwrap(), child.stdout.take().unwrap());
    let head = br#"{"op":"describe","id":"approve"}"#;
    stdin.write_all(&(head.len() as u32).to_le_bytes()).unwrap();
    stdin.write_all(&[0; 4]).unwrap();
    stdin.write_all(head).unwrap();
    stdin.flush().unwrap();
    let mut prefix = [0; 8];
    stdout.read_exact(&mut prefix).unwrap();
    assert_eq!(prefix[4..], [0; 4]);
    let mut reply = vec![0; u32::from_le_bytes(prefix[..4].try_into().unwrap()) as usize];
    stdout.read_exact(&mut reply).unwrap();
    (
        child,
        stdin,
        stdout,
        serde_json::from_slice(&reply).unwrap(),
    )
}
impl Stage {
    /// The owner approves the descriptor the worker gives before the adapter takes its pipes.
    fn new(
        engine: &str,
        model: &str,
        layers: (u32, u32),
        workload: (u32, u32),
        build: &Digest,
    ) -> Self {
        let (prompt, new) = workload;
        let (child, stdin, stdout, d) = launch(engine, model, layers, prompt + new);
        let (first, last) = (d["stage"]["first"] == true, d["stage"]["last"] == true);
        let hidden = &d["model"]["hidden"];
        let pair = json!([
            {"role": "residual", "dtype": "bf16", "shape": [{"kind": "tokens"}, {"kind": "fixed", "value": hidden}]},
            {"role": "hidden", "dtype": "bf16", "shape": [{"kind": "tokens"}, {"kind": "fixed", "value": hidden}]},
        ]);
        let tokens = json!([{"role": "token_ids", "dtype": "u32", "shape": [{"kind": "tokens"}]}]);
        let mut c: Value = serde_json::from_slice(include_bytes!(
            "../../../../../examples/partial-boundary.json"
        ))
        .unwrap();
        c["fixture"] = false.into();
        c["key"]["manifest_digest"] = json!(content_hash(&d["model"]).unwrap());
        c["key"]["backend_build_digest"] = json!(build);
        c["key"]["execution_profile"] = "glyd-range-greedy".into();
        c["key"]["boundary_profile"] = "qwen35-residual-mlp-bf16".into();
        c["key"]["workload"] = json!({"context_tokens": prompt + new, "prompt_tokens": prompt,
            "output_tokens": new, "prefill_chunk_tokens": CHUNK, "sessions": 1});
        c["inputs"] = if first { tokens.clone() } else { pair.clone() };
        c["outputs"] = if last { tokens } else { pair };
        c["output_tokens"] = if last { "sampled_one" } else { "same_chunk" }.into();
        let contract: BoundaryContract = serde_json::from_value(c).unwrap();
        let approval = Approval {
            descriptor: descriptor_digest(&d).unwrap(),
            boundary: contract.clone(),
            engine_bytes_limit: 15 << 30,
            payload_bytes_limit: PAYLOAD,
            first,
            last,
            vocabulary: d["model"]["vocab"].as_u64().unwrap() as u32,
        };
        let mut worker = Worker::new(stdout, stdin, approval, FixtureMode::Reject).unwrap();
        worker.describe().unwrap();
        assert!(matches!(worker.receive().unwrap(), Event::Described));
        worker.prepare().unwrap();
        assert!(matches!(worker.receive().unwrap(), Event::Prepared(_)));
        worker.commit().unwrap();
        assert!(matches!(worker.receive().unwrap(), Event::Committed));
        Self {
            worker,
            contract,
            child,
            workload,
            session: String::new(),
        }
    }
    fn begin(&mut self, scope: DataScope) {
        let (prompt, new) = self.workload;
        self.worker.begin_session(scope, prompt, new, None).unwrap();
        let Event::Begun(session) = self.worker.receive().unwrap() else {
            panic!("begin_session")
        };
        self.session = session;
    }
    fn run(
        &mut self,
        scope: DataScope,
        phase: TensorHeaderPhase,
        start: u32,
        rows: u32,
        inputs: &[&[u8]],
        head: bool,
    ) -> Option<Output> {
        let call = RunChunk {
            version: Version,
            inputs: self.contract.layouts(rows, PAYLOAD).unwrap().0,
            step: StepIdentity {
                scope,
                phase,
                position_start: BoundedU32::new(start).unwrap(),
                token_count: BoundedU32::new(rows).unwrap(),
            },
        };
        self.worker
            .run_chunk(&self.session, &call, &call.step, inputs, head, head)
            .unwrap();
        match self.worker.receive().unwrap() {
            Event::Chunk { output, .. } => output,
            other => panic!("{other:?}"),
        }
    }
    /// The end of its input ends the worker.
    fn close(self) {
        let Self {
            worker, mut child, ..
        } = self;
        drop(worker);
        assert!(child.wait().unwrap().success());
    }
}
fn token(output: Option<Output>) -> Option<(u32, Vec<u8>)> {
    output.map(|o| match o {
        Output::Token { token, logits, .. } => (token, logits.unwrap()),
        other => panic!("{other:?}"),
    })
}
fn le(tokens: &[u32]) -> Vec<u8> {
    tokens.iter().flat_map(|t| t.to_le_bytes()).collect()
}
struct Generated {
    tokens: Vec<u32>,
    logits: Vec<u8>,
    decode: Duration,
}
/// Greedy generation through `forward(step, phase, position, tokens, head)`.
fn generate(
    prompt: &[u32],
    new: u32,
    mut forward: impl FnMut(u64, TensorHeaderPhase, u32, &[u32], bool) -> Option<(u32, Vec<u8>)>,
) -> Generated {
    let (mut tokens, mut logits, mut step) = (Vec::new(), Vec::new(), 0);
    for (i, chunk) in prompt.chunks(CHUNK as usize).enumerate() {
        let start = i as u32 * CHUNK;
        let head = start as usize + chunk.len() == prompt.len();
        let out = forward(step, TensorHeaderPhase::Prefill, start, chunk, head);
        assert_eq!(out.is_some(), head);
        if let Some((t, l)) = out {
            tokens.push(t);
            logits.extend(l);
        }
        step += 1;
    }
    let decode = Instant::now();
    for i in 1..new {
        let position = prompt.len() as u32 + i - 1;
        let last = [*tokens.last().unwrap()];
        let (t, l) = forward(step, TensorHeaderPhase::Decode, position, &last, true).unwrap();
        tokens.push(t);
        logits.extend(l);
        step += 1;
    }
    Generated {
        tokens,
        logits,
        decode: decode.elapsed(),
    }
}
fn digest(g: &Generated) -> Digest {
    let mut bytes = le(&g.tokens);
    bytes.extend(&g.logits);
    byte_digest(&bytes)
}
fn ms_per_token(g: &Generated) -> f64 {
    g.decode.as_secs_f64() * 1e3 / (g.tokens.len() - 1) as f64
}

#[test]
fn real_range_workers_across_enrolled_tls_equal_the_whole_model() {
    let (Ok(engine), Ok(model)) = (
        std::env::var("STARLINGS_RANGE_ENGINE"),
        std::env::var("STARLINGS_RANGE_MODEL"),
    ) else {
        return;
    };
    let var = |name: &str| std::env::var(name).ok().map(|v| v.parse::<u32>().unwrap());
    let (prompt_len, new) = (
        var("STARLINGS_RANGE_PROMPT").unwrap_or(64),
        var("STARLINGS_RANGE_NEW").unwrap_or(32),
    );
    let build = byte_digest(&std::fs::read(&engine).unwrap());
    let (mut probe, stdin, _, d) = launch(&engine, &model, (0, 1), prompt_len + new);
    drop(stdin);
    assert!(probe.wait().unwrap().success());
    let model_u64 = |key: &str| d["model"][key].as_u64().unwrap();
    let (layers, vocab, hidden) = (
        model_u64("layers") as u32,
        model_u64("vocab"),
        model_u64("hidden"),
    );
    let cut = var("STARLINGS_RANGE_CUT").unwrap_or(layers / 2);
    // (deterministic token ids across the vocabulary; bit equality needs no meaningful text)
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    let prompt: Vec<u32> = (0..prompt_len)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 33) % vocab) as u32
        })
        .collect();

    let mut template: TensorHeader = decode(
        include_bytes!("../../../../../examples/tensor-header.json"),
        FixtureMode::Simulation,
    )
    .unwrap();
    // (the reference run has no cluster plan; real mode refuses the example's all-zero digest)
    template.plan_digest = byte_digest(b"whole-model reference run");
    let mut whole = Stage::new(&engine, &model, (0, layers), (prompt_len, new), &build);
    whole.begin((&template).into());
    let reference = generate(&prompt, new, |step, phase, start, ids, head| {
        let mut scope = DataScope::from(&template);
        scope.invocation_id = uuid::Uuid::new_v4().to_string().parse().unwrap();
        scope.sequence = Decimal(step);
        token(whole.run(scope, phase, start, ids.len() as u32, &[&le(ids)], head))
    });
    whole.close();

    let mut stages = [
        Stage::new(&engine, &model, (0, cut), (prompt_len, new), &build),
        Stage::new(&engine, &model, (cut, layers), (prompt_len, new), &build),
    ];
    // The lease's clock starts here, after both stages have loaded.
    let boundary_bytes = |rows: u32| 4 * u64::from(rows) * hidden;
    let f = Fixture::with_workload(boundary_bytes(CHUNK), prompt_len, new);
    let header = |step: u64, phase, start: u32, rows: u32| {
        let mut h = f.header(&vectors()[0]);
        h.shape = [2, rows, hidden as u32]
            .map(|n| BoundedU32::new(n).unwrap())
            .to_vec();
        h.total_tensor_bytes = Decimal(boundary_bytes(rows));
        h.chunk_bytes = h.total_tensor_bytes;
        h.invocation_id = uuid::Uuid::new_v4().to_string().parse().unwrap();
        h.sequence = Decimal(step);
        h.phase = phase;
        h.position_start = BoundedU32::new(start).unwrap();
        h.token_count = BoundedU32::new(rows).unwrap();
        h
    };
    let first = header(0, TensorHeaderPhase::Prefill, 0, CHUNK);
    for stage in &mut stages {
        stage.begin((&first).into());
    }
    let (mut sender, mut receiver) = f.connect(&first);
    let budget = ReceiveBudget::new(
        boundary_bytes(CHUNK),
        boundary_bytes(CHUNK),
        boundary_bytes(CHUNK),
    )
    .unwrap();
    let mut control_sequence = 3;
    let mut control = |operation| {
        control_sequence += 1;
        let request = ControlRequest::new(
            uuid::Uuid::new_v4().to_string().parse().unwrap(),
            1,
            control_sequence,
            operation,
        )
        .unwrap();
        f.workers[1]
            .apply(
                &f.controls[1].0,
                &f.token,
                &request,
                f.clock.elapsed().as_millis() as u64,
            )
            .unwrap();
    };
    let mut moved = 0;
    const PARTS: [&str; 5] = ["admission", "stage_0", "transfer", "start_step", "stage_1"];
    let mut parts = [Duration::ZERO; 5];
    let split = generate(&prompt, new, |step, phase, start, ids, head| {
        let rows = ids.len() as u32;
        let t0 = Instant::now();
        let h = if step == 0 {
            first.clone()
        } else {
            header(step, phase, start, rows)
        };
        let source = [f.admission(&sender.peer, h.clone())];
        let destination = [f.admission(&receiver.peer, h.clone())];
        let t1 = Instant::now();
        let deadline = Instant::now() + Duration::from_secs(10);
        // The receiver grants credit while stage 0 computes: the round trip is off the token's path.
        let (received, t2) = thread::scope(|threads| {
            let task = threads.spawn(|| {
                let mut flow = receiver
                    .begin_receive(&destination, &budget, deadline)
                    .unwrap();
                flow.grant_credit(h.total_tensor_bytes.0).unwrap();
                flow.receive_chunk().unwrap();
                flow.finish().unwrap()
            });
            let Some(Output::Boundary { bytes, .. }) =
                stages[0].run((&h).into(), phase, start, rows, &[&le(ids)], false)
            else {
                panic!("stage 0 gave no boundary");
            };
            let t2 = Instant::now();
            let mut flow = sender.begin_send(&source, deadline).unwrap();
            assert!(matches!(
                flow.receive_control().unwrap(),
                DataControl::Credit { .. }
            ));
            flow.send_chunk(&bytes).unwrap();
            assert!(matches!(
                flow.receive_control().unwrap(),
                DataControl::ReceiptAck { .. }
            ));
            flow.finish().unwrap();
            moved += bytes.len() as u64;
            (task.join().unwrap(), t2)
        });
        let t3 = Instant::now();
        let input = &received.tensors()[0];
        control(Operation::StartStep {
            header: Box::new(h.clone()),
            input_digest: byte_digest(input),
        });
        let t4 = Instant::now();
        let (residual, mlp_out) = input.split_at(input.len() / 2);
        let out = stages[1].run((&h).into(), phase, start, rows, &[residual, mlp_out], head);
        // Completion is the local engine owner's acknowledgement, never a network command.
        f.workers[1]
            .drained(h.generation_id, h.invocation_id)
            .unwrap();
        if phase == TensorHeaderPhase::Decode {
            let t5 = Instant::now();
            for (part, (a, b)) in parts
                .iter_mut()
                .zip([t0, t1, t2, t3, t4].into_iter().zip([t1, t2, t3, t4, t5]))
            {
                *part += b - a;
            }
        }
        token(out)
    });
    for stage in stages {
        stage.close();
    }

    let equal = split.tokens == reference.tokens && split.logits == reference.logits;
    let record = json!({
        "test": "real_range_workers_across_enrolled_tls_equal_the_whole_model",
        "engine_sha256": build,
        "model": model,
        "layers": layers,
        "cut": cut,
        "prompt_tokens": prompt_len,
        "new_tokens": new,
        "chunk_tokens": CHUNK,
        "whole_sha256": digest(&reference),
        "split_sha256": digest(&split),
        "tokens_and_logits_equal": equal,
        "tokens": split.tokens,
        "whole_decode_ms_per_token": ms_per_token(&reference),
        "split_decode_ms_per_token": ms_per_token(&split),
        "split_decode_ms_per_token_by_part": PARTS.iter().zip(parts).map(|(name, d)| {
            (name.to_string(), json!(d.as_secs_f64() * 1e3 / f64::from(new - 1)))
        }).collect::<serde_json::Map<_, _>>(),
        "boundary_bytes_over_tls": moved,
    });
    println!("{record}");
    if let Ok(path) = std::env::var("STARLINGS_RANGE_RECORD") {
        std::fs::write(path, serde_json::to_vec_pretty(&record).unwrap()).unwrap();
    }
    assert!(equal, "split run differs from the whole model");
}
