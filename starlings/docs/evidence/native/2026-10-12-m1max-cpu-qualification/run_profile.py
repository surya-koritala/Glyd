"""Single bounded CPU qualification experiment. Requires fresh workload approval."""
import hashlib, json, math, os, pathlib, platform, resource, secrets, signal, socket, subprocess, sys, threading, time, urllib.error, urllib.request
ROOT=pathlib.Path(__file__).resolve().parent
if sys.argv[1:] != ['--run-approved']: raise SystemExit('Fresh bounded workload approval is required before --run-approved')
START=time.monotonic(); DEADLINE=START+280; LIMIT=4*1024**3
with (ROOT/'started-once').open('x') as out:out.write('One CPU profiling attempt; do not remove to retry.\n')
if platform.system()!='Darwin' or platform.machine()!='arm64': raise RuntimeError('APPROVED_TARGET_MISMATCH')
retained=ROOT.parent; repo=retained.parent/'glyd-compute'; binary=repo/'target/debug/glyd-compute'
manifest=json.loads((ROOT/'manifest.json').read_text())
paths=json.loads((ROOT/'cache-paths.json').read_text())
config=json.loads((retained/'service-config.json').read_text())
exe=pathlib.Path(config['executable']); model=pathlib.Path(paths['weights.gguf'])
def verify(path,expected):
 h=hashlib.sha256()
 with path.open('rb') as src:
  while block:=src.read(8*1024**2):
   if time.monotonic()>=DEADLINE:raise RuntimeError('JOB_DEADLINE')
   h.update(block)
 if h.hexdigest()!=expected:raise RuntimeError('ARTIFACT_HASH_MISMATCH')
verify(exe,'66b94f533a684c97b8954b80a1bda9eb952f8e924f15c5eb16e71a36209e1398')
verify(model,'cedf89af31c9041b601fa58303285bc46d99c51baee1b13f5e919626ca526ee5')
with socket.socket() as reservation:
 reservation.bind(('127.0.0.1',0)); port=reservation.getsockname()[1]
args=config['arguments'];args[args.index('--model')+1]=str(model);args[args.index('--port')+1]=str(port)
args[args.index('--no-slots')]='--slots'
config.update(arguments=args,endpoint=f'127.0.0.1:{port}',manifest_digest=manifest['manifest_digest'],execution_profile=manifest['execution_profile'])
path=ROOT/'profile-service.json';path.write_text(json.dumps(config));path.chmod(0o600)
config_digest=json.loads(subprocess.run([str(binary),'model','service-digest',str(path)],capture_output=True,check=True,timeout=10).stdout)['configuration']
target=json.loads(subprocess.run([str(binary),'model','qualification-target'],capture_output=True,check=True,timeout=10).stdout)['target']
key=secrets.token_hex(32); endpoint=f'http://127.0.0.1:{port}'
env={'PATH':'/usr/bin:/bin',config['key_environment']:key}
child=subprocess.Popen([str(exe),*args],cwd=config['directory'],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
stop=threading.Event();fault=[];samples=[];remaining_groups=[];termination_lock=threading.Lock()
def stats():
 rows=subprocess.run(['/bin/ps','-axo','pid=,ppid=,pgid=,rss='],capture_output=True,text=True,check=True,timeout=2).stdout.splitlines()
 return [tuple(map(int,row.split())) for row in rows if len(row.split())==4]
def poll():
 with termination_lock:return child.poll()
def terminate():
 with termination_lock:
  if child.poll() is None:
   try:os.killpg(child.pid,signal.SIGKILL)
   except ProcessLookupError:pass
  child.wait(timeout=3)
def monitor():
 while not stop.is_set():
  try:
   rows=stats();rss=sum(row[3]*1024 for row in rows if row[2]==child.pid)
   samples.append((time.monotonic(),rss))
   if rss>LIMIT or time.monotonic()>=DEADLINE:
    fault.append('RSS_LIMIT' if rss>LIMIT else 'JOB_DEADLINE');terminate();return
  except Exception:fault.append('WATCHDOG_FAILED');terminate();return
  stop.wait(.1)
watch=threading.Thread(target=monitor,daemon=True);watch.start()
def request(route,body=None,authorized=True,timeout=2):
 if fault or time.monotonic()>=DEADLINE:raise RuntimeError('WATCHDOG_STOP')
 data=None if body is None else json.dumps(body).encode()
 req=urllib.request.Request(endpoint+route,data=data,headers={'Content-Type':'application/json',**({'Authorization':'Bearer '+key} if authorized else {})})
 return urllib.request.urlopen(req,timeout=min(timeout,max(.01,DEADLINE-time.monotonic())))
def read_json(route,body=None,timeout=2):
 with request(route,body,timeout=timeout) as r:
  data=r.read(1024**2+1)
  if len(data)>1024**2:raise RuntimeError('RESPONSE_LIMIT')
  return json.loads(data)
def idle():
 until=min(DEADLINE,time.monotonic()+2)
 while time.monotonic()<until:
  slots=read_json('/slots')
  if not isinstance(slots,list) or len(slots)!=1 or type(slots[0].get('is_processing')) is not bool:raise RuntimeError('INVALID_SLOT_STATE')
  if not slots[0]['is_processing']:return
  time.sleep(.02)
 raise RuntimeError('REQUEST_DRAIN_UNCONFIRMED')
counts=0;generations=0;runs=[]
def chat(text,stream=False):return {'model':config['backend_model'],'messages':[{'role':'user','content':text}],'max_tokens':8,'temperature':0,'top_p':1,'seed':42,'stream':stream,**({'stream_options':{'include_usage':True}} if stream else {})}
def count(text):
 global counts
 counts+=1
 if counts>12:raise RuntimeError('COUNT_BUDGET')
 result=read_json('/v1/chat/completions/input_tokens',chat(text))
 n=result.get('input_tokens')
 if not isinstance(n,int) or not 1<=n<=1048576:raise RuntimeError('INVALID_COUNT')
 return n
def generate(text,n,stream):
 global generations
 generations+=1
 if generations>4 or n+8>128:raise RuntimeError('GENERATION_BUDGET')
 before=time.monotonic(); body=chat(text,stream)
 if not stream:
  value=read_json('/v1/chat/completions',body,60);message=value['choices'][0]['message']
  result={'content':message.get('content') or '', 'reasoning_content':message.get('reasoning_content') or '','finish':value['choices'][0]['finish_reason'],'usage':value['usage']}
 else:
  result={'content':'','reasoning_content':'','finish':None,'usage':None};done=False
  with request('/v1/chat/completions',body,timeout=60) as r:
   while True:
    line=r.readline(65537)
    if len(line)>65536:raise RuntimeError('STREAM_EVENT_LIMIT')
    if not line:break
    if not line.startswith(b'data: '):continue
    data=line[6:].strip()
    if done:raise RuntimeError('OUTPUT_AFTER_DONE')
    if data==b'[DONE]':done=True;continue
    value=json.loads(data)
    if value['choices']:
     choice=value['choices'][0];delta=choice.get('delta',{})
     result['content']+=delta.get('content') or '';result['reasoning_content']+=delta.get('reasoning_content') or ''
     if choice.get('finish_reason') is not None:result['finish']=choice['finish_reason']
    if value.get('usage') is not None:result['usage']=value['usage']
  if not done:raise RuntimeError('STREAM_INCOMPLETE')
 idle();usage=result['usage']
 if not usage or usage['prompt_tokens']!=n or not 0<=usage['completion_tokens']<=8 or usage['total_tokens']!=n+usage['completion_tokens']:raise RuntimeError('EXACT_USAGE_MISMATCH')
 runs.append({'stream':stream,'prompt_tokens':n,'completion_tokens':usage['completion_tokens'],'elapsed_ms':round((time.monotonic()-before)*1000),'slot_idle':True})
 return result
success=False;idle_rss=0
try:
 until=min(DEADLINE,time.monotonic()+60)
 while True:
  try:
   read_json('/health');models=read_json('/v1/models')
   if [m['id'] for m in models['data']] != [config['backend_model']]:raise RuntimeError('MODEL_ALIAS_MISMATCH')
   break
  except (urllib.error.URLError,TimeoutError):
   if poll() is not None or time.monotonic()>=until:raise RuntimeError('READINESS_FAILED')
   time.sleep(.1)
 try:
  with request('/v1/models',authorized=False):pass
  raise RuntimeError('UNAUTHENTICATED_BACKEND')
 except urllib.error.HTTPError as error:
  if error.code not in (401,403):raise RuntimeError('AUTHENTICATION_FAILED')
 idle();short='Reply with the single word ready.';n=count(short)
 reference=generate(short,n,False);idle_rss=sum(row[3]*1024 for row in stats() if row[2]==child.pid)
 if generate(short,n,True)!=reference:raise RuntimeError('SHORT_SEMANTIC_PARITY')
 low=1;high=128;long='word ';long_n=count(long)
 for _ in range(8):
  if low>high:break
  mid=(low+high)//2;text='word '*mid;tokens=count(text)
  if tokens<=120:long=text;long_n=tokens;low=mid+1
  else:high=mid-1
 reference=generate(long,long_n,False)
 if generate(long,long_n,True)!=reference:raise RuntimeError('LONG_SEMANTIC_PARITY')
 success=True
finally:
 terminate();stop.set();watch.join(timeout=3)
 remaining_groups=[row[0] for row in stats() if row[2]==child.pid]
 peak=int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
 rounded=lambda n:math.ceil(n/(64*1024**2))*(64*1024**2)
 record={'kind':'CPUQualificationExperiment','success':success and not fault and not remaining_groups,'configuration':config_digest,'target':target,'manifest':manifest['manifest_digest'],
  'executable':config['executable_sha256'],'artifact':'cedf89af31c9041b601fa58303285bc46d99c51baee1b13f5e919626ca526ee5','context_envelope':128,'max_output_tokens':8,
  'runs':runs,'counts':counts,'generations':generations,'semantic_parity':success,'slot_idle_verified':success,'observed_loaded_idle_rss_bytes':idle_rss,
  'os_child_peak_rss_bytes':peak,'sampled_group_peak_rss_bytes':max((n for _,n in samples),default=0),'sample_interval_ms':100,
  'proposed_loading_bytes':str(rounded(peak+1024**3)),'proposed_request_bytes':str(rounded(max(peak-idle_rss,0)+1024**3)),
  'faults':fault,'owned_group_remaining':bool(remaining_groups),'elapsed_seconds':round(time.monotonic()-START,3),'external_spend_usd':0,
  'qualification_issued':False,'base_conversion_fidelity_qualified':False,'service_cleanup_qualified':False,'daemon_activated':False}
 (ROOT/'profile-result.json').write_text(json.dumps(record,indent=2));(ROOT/'profile-result.json').chmod(0o600)
 print(json.dumps(record),flush=True)
 if not record['success']:raise SystemExit(1)
