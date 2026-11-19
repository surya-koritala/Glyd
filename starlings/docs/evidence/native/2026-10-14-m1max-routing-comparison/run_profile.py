"""One approved current-route CPU comparison; two ordered service starts, no retries or quarantine reset."""
import hashlib,json,os,pathlib,platform,signal,socket,subprocess,sys,threading,time,urllib.error,urllib.request,sqlite3
ROOT=pathlib.Path(__file__).resolve().parent
if sys.argv[1:] not in [['--preflight'],['--run-approved']]:raise SystemExit('Use --preflight for non-inference checks; fresh approval is required for --run-approved')
MODE=sys.argv[1]
START=time.monotonic();DEADLINE=START+240;CAP=4*1024**3
SOURCE=ROOT.parent/'qualification-check';STATE=pathlib.Path(json.loads((ROOT/'state-root.json').read_text())['path']);BIN=ROOT/'glyd-compute'
prep=json.loads((ROOT/'preparation.json').read_text());worker_route=json.loads((ROOT/'worker-route.proposed.json').read_text());template=json.loads((ROOT/'coordinator-route.template.json').read_text());ids=json.loads((ROOT/'identities.json').read_text())
coordinator=STATE/'coordinator';worker=STATE/'worker';model=worker_route['qualification']['replica']['manifest']
state_lock=threading.Lock();children={};locks={};groups=set();samples=[];faults=[];watch_stop=threading.Event();runs=[];statuses=[];requests=0;success=False;stage='direct';failure_phase=None;phase='preflight';direct_generations=0;reference=None;service_history=[];token_path=coordinator/'routing-profile-token.json'
assert not token_path.exists()
def save(path,record):path.write_text(json.dumps(record,indent=2)+'\n');path.chmod(0o600)
def call(state,*args,timeout=3,allow_failure=False):
 if time.monotonic()>START+298:raise RuntimeError('JOB_DEADLINE')
 r=subprocess.run([str(BIN),'--state-dir',str(state),*map(str,args)],capture_output=True,timeout=timeout)
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

def verify_reference_pin():
 pin_bytes=(ROOT/'direct-pin.json').read_bytes();assert hashlib.sha256(pin_bytes).hexdigest()==prep['direct_pin_record_sha256']
 pin=json.loads(pin_bytes);assert pin['manifest']==worker_route['qualification']['replica']['manifest'] and pin['retained'] is True
 pin_bytes=(worker/'cache/models'/('pin-'+pin['pin']+'.json')).read_bytes();assert hashlib.sha256(pin_bytes).hexdigest()==prep['direct_pin_file_sha256']
 owned=json.loads(pin_bytes);assert owned=={'version':1,'manifest':pin['manifest'],'lease':pin['pin']}

def preflight():
 if platform.system()!='Darwin' or platform.machine()!='arm64':raise RuntimeError('APPROVED_TARGET_MISMATCH')
 if MODE=='--run-approved' and (ROOT/'started-once').exists():raise RuntimeError('ATTEMPT_ALREADY_CONSUMED')
 assert hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()==prep['runner_sha256']
 for name,expected in prep['old_ownership_files_sha256'].items():assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()==expected
 assert hashlib.sha256(BIN.read_bytes()).hexdigest()==prep['binary_sha256']
 assert hashlib.sha256((ROOT/'worker-route.proposed.json').read_bytes()).hexdigest()==prep['worker_file_sha256']
 assert hashlib.sha256((ROOT/'coordinator-route.template.json').read_bytes()).hexdigest()==prep['coordinator_template_sha256']
 assert not (STATE/'worker/cache/models/service-start.json').exists()
 verify_reference_pin()
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
    faults.append('RSS_LIMIT' if rss>CAP else 'JOB_DEADLINE');interrupt('direct');interrupt('worker');interrupt('coordinator');return
  except Exception:faults.append('WATCHDOG_FAILED');interrupt('direct');interrupt('worker');interrupt('coordinator');return
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
def ready_service():
 value=service_report()
 if value['state']=='failed_pins_retained':raise RuntimeError('SERVICE_STARTUP_FAILED')
 return value if value['state']=='ready' else None
def settled():
 value=service_report()
 return value if value['state']=='ready' and value['reserved_bytes']=='0' else None
watch=threading.Thread(target=monitor,daemon=True);watch.start()
def chat(stream=False,text='Reply with the single word ready.',limit=8):
 body={'model':model,'messages':[{'role':'user','content':text}],'max_tokens':limit,'temperature':0,'top_p':1,'seed':42,'stream':stream}
 if stream and stage=='direct':body['stream_options']={'include_usage':True}
 return body
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
 assert done
 if stage=='proxy':assert response.headers.get('X-Compute-Session-Id')
 return value
def generate(stream=False,revoke=False):
 global requests
 requests+=1
 if requests>20:raise RuntimeError('GENERATION_BUDGET')
 before=time.monotonic()
 with send(chat(stream)) as response:
  assert response.status==200
  if stream:value=decode_stream(response,revoke)
  else:
   raw=response.read(1024**2+1);assert len(raw)<=1024**2
   body=json.loads(raw);c=body['choices'][0];m=c['message'];value={'content':m.get('content') or '','reasoning_content':m.get('reasoning_content') or '','finish':c['finish_reason'],'usage':body['usage']}
 if value is not None:
  u=value['usage']
  if not isinstance(u,dict) or any(type(u.get(k)) is not int for k in ('prompt_tokens','completion_tokens','total_tokens')) or u['prompt_tokens']!=17 or not 0<=u['completion_tokens']<=8 or u['total_tokens']!=17+u['completion_tokens']:raise RuntimeError('USAGE_MISSING_OR_INVALID')
  if stage=='proxy':wait_for(settled,5)
  else:wait_for(direct_idle,2)
 runs.append({'stream':stream,'stage':stage,'elapsed_ms':round((time.monotonic()-before)*1000),'trusted_request_drained':value is not None})
 return value
try:
 phase='asset_hash_verification'
 # Hash/canonical asset checks occur before the only model start.
 for path,expected in [(pathlib.Path(worker_route['service']['executable']),worker_route['service']['executable_sha256']),(pathlib.Path(json.loads((ROOT/'cache-paths.json').read_text())['weights.gguf']),'cedf89af31c9041b601fa58303285bc46d99c51baee1b13f5e919626ca526ee5')]:
  h=hashlib.sha256()
  with path.open('rb') as source:
   while block:=source.read(8*1024**2):
    if time.monotonic()>=DEADLINE:raise RuntimeError('JOB_DEADLINE')
    h.update(block)
  assert h.hexdigest()==expected
 # Direct reference uses the same cached artifact/runtime/precision/threads and fixed input.
 direct_config=worker_route['service']; model=direct_config['backend_model']
 direct_key=os.urandom(32).hex();token=direct_key
 with socket.socket() as reserved:reserved.bind(('127.0.0.1',0));port=reserved.getsockname()[1]
 address='127.0.0.1:'+str(port)
 args=list(direct_config['arguments']);args[args.index('--port')+1]=str(port)
 phase='direct_launch'
 log=open(ROOT/'direct.log','wb')
 direct=subprocess.Popen([direct_config['executable'],*args],cwd=direct_config['directory'],env={direct_config['key_environment']:direct_key},stdout=log,stderr=log,start_new_session=True);log.close()
 with state_lock:children['direct']=direct;locks['direct']=threading.Lock()
 def direct_get(path):
  req=urllib.request.Request('http://'+address+path,headers={'Authorization':'Bearer '+direct_key})
  with urllib.request.urlopen(req,timeout=2) as response:
   data=response.read(65537);assert len(data)<=65536
   return json.loads(data)
 def direct_ready():
  try:
   models=direct_get('/v1/models')['data']
   assert len(models)==1 and models[0]['id']==model
   return True
  except urllib.error.HTTPError as error:
   if error.code!=503:raise
   return False
  except urllib.error.URLError:return False
 def direct_idle():
  slots=direct_get('/slots');return len(slots)==1 and slots[0].get('is_processing') is False
 phase='direct_readiness'
 wait_for(direct_ready,60)
 phase='direct_comparison'
 for _ in range(5):
  for stream in (False,True):
   value=generate(stream)
   if reference is None:reference=value
   assert value==reference
 interrupt('direct')
 with locks['direct']:direct.wait(timeout=8)
 assert not any(group==direct.pid for _,_,group,_ in rows()),'DIRECT_GROUP_REMAINING'
 del direct_key;stage='proxy';model=worker_route['qualification']['replica']['manifest']
 phase='proxy_control_preparation'
 launch('coordinator','daemon')
 scopes=coordinator/'daemon-acceptance-scopes.json'
 for node,action in [(ids['coordinator'],'control'),(ids['worker'],'infer'),(ids['client'],'infer')]:
  save(scopes,[{'model':model,'action':action}]);call(coordinator,'cluster','permissions',node,scopes)
 call(coordinator,'cluster','token',ids['client'],scopes,'300',token_path)
 secret=json.loads(token_path.read_text())['secret'];assert len(secret)==32;token=bytes(secret).hex();del secret
 phase='worker_launch'
 launch('worker','daemon','--execute',ROOT/'worker-route.proposed.json','--approve',prep['worker_record_digest'])
 phase='worker_readiness'
 ready=wait_for(ready_service,70)
 assert ready['inference_qualified'] and not ready['fixture'] and ready['model_architecture']=='qwen35'
 template['worker_endpoint']=ready['endpoint'];route=coordinator/'daemon-acceptance-route.json';save(route,template)
 route_digest=json.loads(call(coordinator,'model','route-digest',route))['configuration'];call(coordinator,'cluster','route',route,'--approve',route_digest)
 address=field(call(coordinator,'cluster','status'),'inference=')
 wait_for(lambda:report(coordinator,'resources',ids['worker'])['sample']['capabilities']['state']=='ready',10)
 phase='proxy_comparison'
 for _ in range(5):
  for stream in (False,True):assert generate(stream)==reference
 assert requests==20 and len(runs)==20
 routing=field(call(coordinator,'cluster','status'),'routing=')
 routing=json.loads(routing);assert routing['retained_requests']==0 and routing['retained_bytes']=='0'
 success=True
except Exception as error:
 failure_phase=phase
 faults.append('ACCEPTANCE_'+('USAGE_MISSING_OR_INVALID' if str(error)=='USAGE_MISSING_OR_INVALID' else type(error).__name__))
finally:
 if token_path.exists() and 'coordinator' in children:
  try:call(coordinator,'cluster','revoke-token',token_path)
  except Exception:faults.append('TOKEN_REVOCATION_UNCONFIRMED')
 for name in ['direct','worker','coordinator']:
  if name not in children:continue
  try:
   if name=='direct':interrupt(name)
   else:call(STATE/name,'cluster','stop',allow_failure=True)
  except Exception:interrupt(name)
  try:
   with locks[name]:children[name].wait(timeout=8 if name in ('worker','direct') else 4)
  except Exception:faults.append('DAEMON_STOP_UNCONFIRMED');interrupt(name)
 watch_stop.set();watch.join(timeout=3)
 if watch.is_alive():faults.append('WATCHDOG_STOP_UNCONFIRMED')
 remaining=remaining_groups()
 try:
  with sqlite3.connect('file:'+str(coordinator/'enrollment.sqlite')+'?mode=ro',uri=True) as db:token_rows=db.execute('SELECT count(*) FROM tokens').fetchone()[0]
  if token_rows!=0:faults.append('TOKEN_RETAINED')
  old_preserved=all(hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()==expected for name,expected in prep['old_ownership_files_sha256'].items())
  if not old_preserved:faults.append('OLD_OWNERSHIP_CHANGED')
 except Exception:token_rows=None;old_preserved=False;faults.append('FINAL_OBSERVATION_UNAVAILABLE')
 server_groups=groups-{p.pid for name,p in children.items() if name!='direct'}
 if success and not server_groups:faults.append('SERVER_GROUP_NOT_OBSERVED')
 result={'kind':'NativeCurrentRouteCPUComparison','success':success and not faults and remaining is False,'service_status_history':service_history,'implementation':prep['implementation_commit'],'binary_sha256':prep['binary_sha256'],'manifest':worker_route['qualification']['replica']['manifest'],'service_configuration':prep['configuration'],'worker_record':prep['worker_record_digest'],'profile_evidence':prep['profile_evidence'],'generations_requested':requests,'count_call_upper_bound':requests*2+(1 if requests else 0),'count_calls_instrumented':False,'output_tokens_total_cap':160,'runs':runs,'json_sse_semantic_parity':success,'matched_direct_proxy':success,'physical_hosts':1,'two_machine_qualification':False,'temporary_token_rows':token_rows,'old_ownership_preserved':old_preserved,'direct_pin_retained':(ROOT/'direct-pin.json').exists(),'quarantine_retained':(worker/'cache/models/service-start.json').exists(),'process_observation_available':remaining is not None,'owned_server_groups_observed':len(server_groups),'owned_group_remaining':remaining,'sampled_owned_peak_rss_bytes':max(samples,default=0),'sample_interval_ms':100,'elapsed_seconds':round(time.monotonic()-START,3),'external_spend_usd':0,'faults':faults,'failure_phase':failure_phase,'general_production_qualification_issued':False,'service_cleanup_qualified':False,'base_conversion_fidelity_qualified':False}
 save(ROOT/'daemon-result.json',result);print(json.dumps(result),flush=True)
 if not result['success']:raise SystemExit(1)
