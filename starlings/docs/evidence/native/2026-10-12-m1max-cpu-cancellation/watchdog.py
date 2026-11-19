import hashlib, json, os, pathlib, signal, socket, subprocess, time
root = pathlib.Path(__file__).resolve().parent
started = time.monotonic()
run = root / 'cancel-check'
run.mkdir(exist_ok=True)
with (run / 'started-once').open('x') as marker:
    marker.write('One bounded native cancellation attempt; do not remove to retry.\n')
executable = (root/'build/bin/llama-server').resolve()
model = (root/'Qwen3.5-0.8B-BF16.gguf').resolve()
def verify(path, expected, size=None):
    if size is not None and path.stat().st_size != size:
        raise RuntimeError('ARTIFACT_SIZE_MISMATCH')
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(8 * 1024 * 1024):
            if time.monotonic() - started >= 280:
                raise RuntimeError('NATIVE_RUN_DEADLINE')
            digest.update(block)
    if digest.hexdigest() != expected:
        raise RuntimeError('ARTIFACT_DIGEST_MISMATCH')
verify(executable, '66b94f533a684c97b8954b80a1bda9eb952f8e924f15c5eb16e71a36209e1398')
verify(model, 'cedf89af31c9041b601fa58303285bc46d99c51baee1b13f5e919626ca526ee5', 1516744736)
with socket.socket() as reservation:
    reservation.bind(('127.0.0.1', 0))
    port = reservation.getsockname()[1]
arguments = ['--model', str(model), '--alias', 'glyd-native-smoke', '--host', '127.0.0.1', '--port', str(port),
    '--threads', '2', '--threads-batch', '2', '--threads-http', '2', '--parallel', '1', '--ctx-size', '512',
    '--n-predict', '8', '--batch-size', '64', '--ubatch-size', '64', '--device', 'none', '--n-gpu-layers', '0',
    '--no-kv-offload', '--no-op-offload', '--fit', 'off', '--no-repack', '--cache-type-k', 'f32', '--cache-type-v', 'f32',
    '--flash-attn', 'off', '--no-warmup', '--no-cache-prompt', '--cache-ram', '0', '--no-cache-idle-slots',
    '--offline', '--no-mmproj', '--no-webui', '--no-agent', '--no-ui-mcp-proxy', '--no-slots',
    '--reasoning', 'on', '--reasoning-format', 'deepseek', '--log-disable']
config = {'executable': str(executable), 'executable_sha256': hashlib.sha256(executable.read_bytes()).hexdigest(),
    'arguments': arguments, 'directory': str(root), 'environment': {}, 'key_environment':'LLAMA_API_KEY',
    'endpoint': f'127.0.0.1:{port}', 'backend_version':'llama.cpp-0.6.0-d81235049384534c167caea52b85a694f6103d14',
    'manifest_digest':'cedf89af31c9041b601fa58303285bc46d99c51baee1b13f5e919626ca526ee5',
    'execution_profile':'unqualified-cpu-bf16-weights-f32-kv', 'backend_model':'glyd-native-smoke',
    'reasoning_content':True, 'request_timeout_ms':60000}
data = json.dumps(config, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
digest = hashlib.sha256(data).hexdigest()
path = run / 'service-config.json'
path.write_bytes(data)
path.chmod(0o600)
binary = root.parent / 'glyd-compute/target/debug/examples/service_check'
owned = set()
peak_rss_kib = 0
def processes():
    result = subprocess.run(['/bin/ps','-axo','pid=,ppid=,pgid=,rss='],capture_output=True,text=True,check=True,timeout=2)
    return {int(v[0]): tuple(map(int,v[1:])) for line in result.stdout.splitlines() if len(v:=line.split()) == 4}
def terminate_service():
    current = processes()
    for pid in owned:
        if pid not in current or current[pid][1] != pid:
            continue
        name = subprocess.run(['/bin/ps','-ww','-p',str(pid),'-o','comm='],capture_output=True,text=True,timeout=2).stdout.strip()
        if name == str(executable):
            try: os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError: pass
with (run/'cancellation-result.json').open('w') as output, (run/'cancellation-error.txt').open('w') as error:
    child = subprocess.Popen([str(binary), str(path), digest, '--cancel-after-text'], stdout=output, stderr=error, start_new_session=True)
    try:
        while child.poll() is None:
            current = processes()
            for pid,(parent,group,rss) in current.items():
                if parent == child.pid and group == pid:
                    owned.add(pid)
                if pid in owned:
                    peak_rss_kib = max(peak_rss_kib, rss)
            if time.monotonic() - started >= 280:
                raise RuntimeError('NATIVE_RUN_DEADLINE')
            time.sleep(0.1)
    finally:
        try:
            terminate_service()
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=3)
    record = {'exit_code':child.returncode, 'elapsed_seconds':round(time.monotonic()-started,3),
        'configuration_digest':digest, 'owned_server_observed':bool(owned),
        'owned_server_remaining':any(pid in processes() for pid in owned),
        'owned_group_remaining':any(group in owned for parent,group,rss in processes().values()),
        'scope':'one eight-token CPU request; cancel after unfinished text',
        'sampled_server_peak_rss_kib':peak_rss_kib, 'rss_sample_period_ms':100,
        'rss_is_reservation_profile':False, 'external_spend_usd':0,
        'manifest_identity_is_catalog_qualified':False}
    (run/'watchdog-record.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    if child.returncode or not owned or record['owned_group_remaining']:
        raise SystemExit(1)
