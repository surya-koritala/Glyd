"""One separately approved CPU recovery attempt; no retries or old quarantine reset."""
import hashlib,json,os,re,pathlib,platform,signal,socket,subprocess,sys,threading,time,urllib.error,urllib.request
ROOT=pathlib.Path(__file__).resolve().parent
if sys.argv[1:] not in [['--preflight'],['--run-approved']]:raise SystemExit('Use --preflight for non-inference checks; fresh approval is required for --run-approved')
MODE=sys.argv[1]
control_events=[];control_next={};resource_history=[];phase='preflight';failure_stage=None
START=time.monotonic();DEADLINE=START+280;CAP=4*1024**3
SOURCE=ROOT.parent/'qualification-check';STATE=pathlib.Path(json.loads((ROOT/'state-root.json').read_text())['path']);BIN=ROOT/'glyd-compute'
prep=json.loads((ROOT/'preparation.json').read_text());worker_route=json.loads((ROOT/'worker-route.proposed.json').read_text());template=json.loads((ROOT/'coordinator-route.template.json').read_text());ids=json.loads((ROOT/'identities.json').read_text())
coordinator=STATE/'coordinator';worker=STATE/'worker';model=worker_route['qualification']['replica']['manifest']
state_lock=threading.Lock();children={};locks={};groups=set();samples=[];faults=[];watch_stop=threading.Event();runs=[];statuses=[];requests=0;success=False;service_history=[];token_path=coordinator/'daemon-acceptance-token.json'
assert not token_path.exists()
def save(path,record):path.write_text(json.dumps(record,indent=2)+'\n');path.chmod(0o600)
def call(state,*args,timeout=3,allow_failure=False):
 if time.monotonic()>START+298:raise RuntimeError('JOB_DEADLINE')
 if args[0]=='cluster':
  delay=control_next.get(str(state),0)-time.monotonic()
  if delay>0:time.sleep(delay)
  control_next[str(state)]=time.monotonic()+.2
 r=subprocess.run([str(BIN),'--state-dir',str(state),*map(str,args)],capture_output=True,timeout=timeout)
 if r.returncode:
  code=r.stderr.decode(errors='replace').partition(':')[0].strip()
  control_events.append({'operation':'/'.join(map(str,args[:2])),'role':state.name,'returncode':r.returncode,'code':code if re.fullmatch(r'[A-Z][A-Z0-9_]{0,64}',code) else 'UNKNOWN_CONTROL_ERROR'})
 if r.returncode and not allow_failure:raise RuntimeError('CONTROL_COMMAND_FAILED')
 return r.stdout.decode().strip()
def field(text,key):return next(w[len(key):].rstrip(';') for w in text.split() if w.startswith(key))
def report(state,*args):return json.loads(call(state,'cluster',*args).split(': ',1)[1])
def rows():return [tuple(map(int,r.split())) for r in subprocess.run(['/bin/ps','-axo','pid=,ppid=,pgid=,rss='],capture_output=True,text=True,timeout=2,check=True).stdout.splitlines() if len(r.split())==4]
def remaining_groups():
 try:return any(group in groups for _,_,group,_ in rows())
 except Exception:
  faults.append('PROCESS_OBSERVATION_UNAVAILABLE')
  return None

def preflight():
 if platform.system()!='Darwin' or platform.machine()!='arm64':raise RuntimeError('APPROVED_TARGET_MISMATCH')
 if MODE=='--run-approved' and (ROOT/'started-once').exists():raise RuntimeError('ATTEMPT_ALREADY_CONSUMED')
 assert hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()==prep['runner_sha256']
 assert hashlib.sha256(BIN.read_bytes()).hexdigest()==prep['binary_sha256']
 assert hashlib.sha256((ROOT/'worker-route.proposed.json').read_bytes()).hexdigest()==prep['worker_file_sha256']
 assert hashlib.sha256((ROOT/'coordinator-route.template.json').read_bytes()).hexdigest()==prep['coordinator_template_sha256']
 assert hashlib.sha256((ROOT/'cleanup-policy-source-review.json').read_bytes()).hexdigest()==worker_route['cleanup']['evidence']
 for name,expected in prep['old_ownership_files_sha256'].items():assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()==expected
 assert not (STATE/'worker/cache/models/service-start.json').exists()
 data=rows()
 if not any(pid==os.getpid() for pid,_,_,_ in data):raise RuntimeError('PROCESS_OBSERVATION_UNAVAILABLE')
 if token_path.exists():raise RuntimeError('TEMPORARY_TOKEN_ALREADY_EXISTS')
 host,port=worker_route['service']['endpoint'].split(':')
 with socket.socket() as test:test.bind((host,int(port)))
 target=json.loads(call(worker,'model','qualification-target'))['target']
 if target!=worker_route['qualification']['target']:raise RuntimeError('APPROVED_TARGET_MISMATCH')
 resources=json.loads(subprocess.run([str(BIN),'probe',str(worker/'cache/models')],capture_output=True,check=True,timeout=3).stdout)
 if resources['stale'] or not any(p['pool_id']=='host' and int(p['available_bytes'])>=int(worker_route['qualification']['loading_bytes']) for p in resources['memory_pools']):raise RuntimeError('LOADING_CAPACITY_UNAVAILABLE')
 return {'kind':'NativeDaemonPreflight','success':True,'model_started':False,'approval_consumed':False,'process_observation_available':True,'loopback_bind_available':True,'configuration':prep['configuration'],'binary_sha256':prep['binary_sha256']}

try:
 checked=preflight()
except Exception as error:
 code=error.args[0] if error.args and isinstance(error.args[0],str) and error.args[0] in {'APPROVED_TARGET_MISMATCH','ATTEMPT_ALREADY_CONSUMED','PROCESS_OBSERVATION_UNAVAILABLE','TEMPORARY_TOKEN_ALREADY_EXISTS','LOADING_CAPACITY_UNAVAILABLE','CONTROL_COMMAND_FAILED'} else type(error).__name__
 checked={'kind':'NativeDaemonPreflight','success':False,'model_started':False,'approval_consumed':False,'fault':'PREFLIGHT_'+code}
 save(ROOT/'preflight-result.json',checked);print(json.dumps(checked),flush=True);raise SystemExit(2)
save(ROOT/'preflight-result.json',checked)
if MODE=='--preflight':print(json.dumps(checked),flush=True);raise SystemExit(0)
with (ROOT/'started-once').open('x') as out:out.write('One approved native daemon attempt; never remove to retry.\n')
(ROOT/'started-once').chmod(0o600)
def interrupt(name):
 p=children.get(name)
 if p is not None:
  with locks[name]:
   if p.poll() is None:p.send_signal(signal.SIGINT)
def observe():
 data=rows()
 with state_lock:owned={p.pid for p in children.values()}
 while True:
  expanded=owned|{pid for pid,parent,_,_ in data if parent in owned}
  if expanded==owned:break
  owned=expanded
 for pid,_,group,_ in data:
  if pid in owned:groups.add(group)
 return sum(rss*1024 for pid,_,group,rss in data if pid in owned or group in groups)
def monitor():
 while not watch_stop.is_set():
  try:
   rss=observe();samples.append(rss)
   if rss>CAP or time.monotonic()>=DEADLINE:
    faults.append('RSS_LIMIT' if rss>CAP else 'JOB_DEADLINE');interrupt('worker');interrupt('coordinator');return
  except Exception:faults.append('WATCHDOG_FAILED');interrupt('worker');interrupt('coordinator');return
  watch_stop.wait(.1)
def launch(name,*args):
 log=open(ROOT/(name+'.log'),'wb')
 p=subprocess.Popen([str(BIN),'--state-dir',str(STATE/name),*args],stdout=log,stderr=log,start_new_session=True);log.close()
 with state_lock:
  locks[name]=threading.Lock();children[name]=p
 # Only the watchdog mutates the observed group set.
 until=time.monotonic()+10
 while time.monotonic()<until:
  try:call(STATE/name,'cluster','status');return
  except RuntimeError:
   with locks[name]:
    if p.poll() is not None:raise RuntimeError('DAEMON_START_FAILED')
   time.sleep(.1)
 raise RuntimeError('DAEMON_READY_TIMEOUT')
def wait_for(check,seconds):
 until=min(DEADLINE,time.monotonic()+seconds)
 while time.monotonic()<until:
  if faults:raise RuntimeError('WATCHDOG_STOP')
  value=check()
  if value:return value
  time.sleep(.1)
 raise RuntimeError('STATE_TIMEOUT')
def service_report():
 value=report(worker,'service')
 if not service_history or value!=service_history[-1]:service_history.append(value)
 return value
def ready_resource():
 value=report(coordinator,'resources',ids['worker'])
 if not resource_history or value!=resource_history[-1]:resource_history.append(value)
 return value if value is not None and value['sample']['capabilities']['state']=='ready' else None
def ready_service():
 value=service_report()
 if value['state'].startswith('failed'):raise RuntimeError('SERVICE_STARTUP_FAILED')
 return value if value['state']=='ready' else None
def settled():
 value=service_report()
 return value if value['state']=='ready' and value['reserved_bytes']=='0' else None
watch=threading.Thread(target=monitor,daemon=True);watch.start()
def chat(stream=False,text='Reply with the single word ready.',limit=8):return {'model':model,'messages':[{'role':'user','content':text}],'max_tokens':limit,'temperature':0,'top_p':1,'seed':42,'stream':stream}
def send(body,authorized=True):
 global requests
 if faults or time.monotonic()>=DEADLINE:raise RuntimeError('WATCHDOG_STOP')
 data=json.dumps(body).encode();req=urllib.request.Request('http://'+address+'/v1/chat/completions',data=data,headers={'Content-Type':'application/json',**({'Authorization':'Bearer '+token} if authorized else {})})
 return urllib.request.urlopen(req,timeout=min(60,max(.01,DEADLINE-time.monotonic())))
def refused(body,code,authorized=True):
 try:
  with send(body,authorized):pass
  raise RuntimeError('EXPECTED_REFUSAL')
 except urllib.error.HTTPError as error:
  assert error.code==code;statuses.append(code)
def decode_stream(response,revoke=False):
 value={'content':'','reasoning_content':'','finish':None,'usage':None};done=False;revoked=False
 while True:
  line=response.readline(65537)
  if len(line)>65536:raise RuntimeError('STREAM_EVENT_LIMIT')
  if not line:break
  if not line.startswith(b'data: '):continue
  if done:raise RuntimeError('OUTPUT_AFTER_DONE')
  data=line[6:].strip()
  if data==b'[DONE]':done=True;continue
  event=json.loads(data)
  if event.get('error'):
   if revoke:continue
   raise RuntimeError('STREAM_FAILURE')
  if event.get('choices'):
   c=event['choices'][0];delta=c.get('delta',{})
   value['content']+=delta.get('content') or '';value['reasoning_content']+=delta.get('reasoning_content') or ''
   if c.get('finish_reason') is not None:value['finish']=c['finish_reason']
   if revoke and not revoked and (delta.get('content') or delta.get('reasoning_content')):
    assert not done;call(coordinator,'cluster','revoke-token',token_path);revoked=True
  if event.get('usage') is not None:value['usage']=event['usage']
 if revoke:
  assert revoked and not done
  return None
 assert done and response.headers.get('X-Compute-Session-Id')
 return value
def generate(stream=False,revoke=False):
 global requests
 requests+=1
 if requests>2:raise RuntimeError('GENERATION_BUDGET')
 before=time.monotonic()
 with send(chat(stream)) as response:
  assert response.status==200
  if stream:value=decode_stream(response,revoke)
  else:
   raw=response.read(1024**2+1);assert len(raw)<=1024**2
   body=json.loads(raw);c=body['choices'][0];m=c['message'];value={'content':m.get('content') or '','reasoning_content':m.get('reasoning_content') or '','finish':c['finish_reason'],'usage':body['usage']}
 if value is not None:
  u=value['usage'];assert u['prompt_tokens']==17 and 0<=u['completion_tokens']<=8 and u['total_tokens']==17+u['completion_tokens']
  wait_for(settled,5)
 runs.append({'stream':stream,'revoked_midstream':revoke,'elapsed_ms':round((time.monotonic()-before)*1000),'trusted_request_drained':value is not None})
 return value
try:
 # Hash/canonical asset checks occur before the only model start.
 for path,expected in [(pathlib.Path(worker_route['service']['executable']),worker_route['service']['executable_sha256']),(pathlib.Path(json.loads((ROOT/'cache-paths.json').read_text())['weights.gguf']),'cedf89af31c9041b601fa58303285bc46d99c51baee1b13f5e919626ca526ee5')]:
  h=hashlib.sha256()
  with path.open('rb') as source:
   while block:=source.read(8*1024**2):
    if time.monotonic()>=DEADLINE:raise RuntimeError('JOB_DEADLINE')
    h.update(block)
  assert h.hexdigest()==expected
 phase='coordinator_start'
 launch('coordinator','daemon')
 phase='permissions_and_token'
 scopes=coordinator/'daemon-acceptance-scopes.json'
 for node,action in [(ids['coordinator'],'control'),(ids['worker'],'infer'),(ids['client'],'infer')]:
  save(scopes,[{'model':model,'action':action}]);call(coordinator,'cluster','permissions',node,scopes)
 call(coordinator,'cluster','token',ids['client'],scopes,'300',token_path)
 secret=json.loads(token_path.read_text())['secret'];assert len(secret)==32;token=bytes(secret).hex();del secret
 phase='worker_start'
 launch('worker','daemon','--execute',ROOT/'worker-route.proposed.json','--approve',prep['worker_record_digest'])
 phase='service_readiness'
 ready=wait_for(ready_service,70)
 assert ready['inference_qualified'] and not ready['fixture'] and ready['model_architecture']=='qwen35'
 phase='route_activation'
 template['worker_endpoint']=ready['endpoint'];route=coordinator/'daemon-acceptance-route.json';save(route,template)
 route_digest=json.loads(call(coordinator,'model','route-digest',route))['configuration'];call(coordinator,'cluster','route',route,'--approve',route_digest)
 address=field(call(coordinator,'cluster','status'),'inference=')
 phase='resource_readiness'
 wait_for(ready_resource,10)
 phase='first_generation'
 first=generate()
 # The only second generation is revoked after real text; no generation retry.
 phase='revoked_generation'
 generate(True,True)
 failed=wait_for(lambda:service_report() if service_report()['state']=='failed_cleanup_proven' else None,10)
 assert failed['reserved_bytes']=='0' and failed['pin'] and failed['process_group_absent'] and failed['cleanup_proven']
 assert 'role=worker' in call(worker,'cluster','status') and 'role=coordinator' in call(coordinator,'cluster','status')
 phase='recovery_checks'
 pin=failed['pin'];configuration=prep['configuration']
 receipt=json.loads(call(worker,'model','service-state'))
 assert receipt['phase']=='drained' and receipt['lease']==pin and receipt['configuration']==configuration
 assert receipt['cleanup_qualification']==hashlib.sha256(json.dumps(worker_route['cleanup'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
 # Wrong identity cannot recover this launch. No force/receipt import path exists.
 bad=subprocess.run([str(BIN),'--state-dir',str(worker),'cluster','recover-service','00000000-0000-4000-8000-000000000999',configuration],capture_output=True,timeout=3)
 assert bad.returncode!=0
 assert 'repeated=false' in call(worker,'cluster','recover-service',pin,configuration)
 assert 'repeated=true' in call(worker,'cluster','recover-service',pin,configuration)
 recovered=service_report();assert recovered['state']=='recovered' and recovered['pin'] is None and recovered['reserved_bytes']=='0'
 receipt=json.loads(call(worker,'model','service-state'));assert receipt['phase']=='recovered' and receipt['lease']==pin
 assert not (worker/'cache/models'/('pin-'+pin+'.json')).exists()
 assert json.loads(call(worker,'model','recover-service',pin,configuration))['repeated']
 # Verified objects remain present; this experiment does not run GC or restart the model.
 paths=json.loads(call(worker,'model','paths',model,timeout=30));assert paths['weights.gguf']
 success=True
except Exception as error:
 failure_stage=phase
 faults.append('ACCEPTANCE_'+type(error).__name__)
finally:
 if token_path.exists() and 'coordinator' in children:
  try:call(coordinator,'cluster','revoke-token',token_path)
  except Exception:faults.append('TOKEN_WITHDRAWAL_FAILED')
 for name in ['worker','coordinator']:
  if name not in children:continue
  try:call(STATE/name,'cluster','stop')
  except Exception:faults.append('DAEMON_STOP_COMMAND_FAILED');interrupt(name)
  try:
   with locks[name]:children[name].wait(timeout=8 if name=='worker' else 4)
  except Exception:faults.append('DAEMON_STOP_UNCONFIRMED');interrupt(name)
 watch_stop.set();watch.join(timeout=3)
 if watch.is_alive():faults.append('WATCHDOG_STOP_UNCONFIRMED')
 try:
  old_preserved=all(hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()==expected for name,expected in prep['old_ownership_files_sha256'].items())
 except Exception:old_preserved=False
 if not old_preserved:faults.append('PRIOR_OWNERSHIP_CHANGED')
 remaining=remaining_groups()
 server_groups=groups-{p.pid for p in children.values()}
 if success and not server_groups:faults.append('SERVER_GROUP_NOT_OBSERVED')
 if success and any(p.returncode!=0 for p in children.values()):faults.append('DAEMON_EXIT_FAILED')
 passed=success and not faults and remaining is False
 result={'kind':'NativeServiceRecoveryExperiment','success':passed,'failure_stage':failure_stage,'control_failures':control_events,'control_spacing_seconds':.2,'resource_history':resource_history,'service_status_history':service_history,'implementation':prep['implementation_commit'],'binary_sha256':prep['binary_sha256'],'manifest':model,'service_configuration':prep['configuration'],'worker_record':prep['worker_record_digest'],'profile_evidence':prep['profile_evidence'],'generations_requested':requests,'count_call_upper_bound':requests*2,'count_calls_instrumented':False,'output_tokens_total_cap':requests*8, 'cleanup_policy_source_review':worker_route['cleanup']['evidence'], 'cleanup_policy':worker_route['cleanup'],'runs':runs,'refusal_statuses':statuses,'json_sse_semantic_parity':False,'revocation_withdraws_done':success,'qualified_failed_reservation_released':success,'recovery_record_retained':(worker/'cache/models/service-start.json').exists(), 'exact_launch_recovered':success, 'old_quarantines_preserved':old_preserved,'process_observation_available':remaining is not None,'owned_server_groups_observed':len(server_groups),'owned_group_remaining':remaining,'sampled_owned_peak_rss_bytes':max(samples,default=0),'sample_interval_ms':100,'elapsed_seconds':round(time.monotonic()-START,3),'external_spend_usd':0,'faults':faults,'general_production_qualification_issued':False,'bounded_cpu_service_cleanup_qualified':passed,'base_conversion_fidelity_qualified':False}
 save(ROOT/'daemon-result.json',result);print(json.dumps(result),flush=True)
 if not result['success']:raise SystemExit(1)
