"""Prepare fresh isolated control/catalog state; no model process starts."""
import hashlib,json,pathlib,re,subprocess,time,uuid
ROOT=pathlib.Path(__file__).resolve().parent;SOURCE=ROOT.parent/'qualification-check';BINARY=ROOT/'glyd-compute';STATE=pathlib.Path('/private/tmp/glyd-m2-recovery-2-'+uuid.uuid4().hex[:8]);STATE.mkdir(mode=0o700)
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n');path.chmod(0o600)
def call(state,*args):
 r=subprocess.run([str(BINARY),'--state-dir',str(state),*map(str,args)],capture_output=True,text=True,timeout=30)
 if r.returncode:raise RuntimeError('PREPARATION_COMMAND_FAILED:'+r.stderr.strip())
 return r.stdout.strip()
def field(text,key):return next(x[len(key):].rstrip(';') for x in text.split() if x.startswith(key))
coordinator=STATE/'coordinator';call(coordinator,'cluster','init','--listen','127.0.0.1:0');log=open(ROOT/'control-preparation.log','wb');process=subprocess.Popen([str(BINARY),'--state-dir',str(coordinator),'daemon'],stdout=log,stderr=log)
try:
 until=time.monotonic()+10
 while True:
  try:call(coordinator,'cluster','status');break
  except RuntimeError:
   if process.poll() is not None or time.monotonic()>until:raise
   time.sleep(.1)
 identities={}
 for role,name in [('worker','worker'),('client','client')]:
  invite=coordinator/(name+'-invite.json');call(coordinator,'cluster','invite',role,invite);time.sleep(.2)
  state=STATE/name;prepared=call(state,'cluster','prepare',invite);node=field(prepared,'node=');call(coordinator,'cluster','approve',node,field(prepared,'fingerprint='));time.sleep(.2);call(state,'cluster','join',invite);identities[name]=node;time.sleep(.2)
 cfg=json.loads((coordinator/'config.json').read_text());identities.update(coordinator=cfg['node'],cluster=cfg['cluster'])
 manifest=json.loads((SOURCE/'manifest.json').read_text());worker=STATE/'worker';digest=manifest['manifest_digest'];call(worker,'model','register',SOURCE/'manifest.json');call(worker,'model','approve',digest);call(worker,'model','import',digest,SOURCE/'source');paths=json.loads(call(worker,'model','paths',digest))
 save(ROOT/'state-root.json',{'path':str(STATE)});save(ROOT/'identities.json',identities);save(ROOT/'cache-paths.json',paths)
 route=json.loads((ROOT.parent/'daemon-check-2/worker-route.proposed.json').read_text());service=route['service'];service['arguments'][service['arguments'].index('--model')+1]=paths['weights.gguf'];q=route['qualification'];q['replica']['node']=identities['worker'];service['environment']={'GGML_NO_BACKTRACE':'1'};save(ROOT/'service.json',service);configuration=json.loads(call(worker,'model','service-digest',ROOT/'service.json'))['configuration'];q['configuration']=configuration;route['cleanup']={'version':1,'configuration':configuration,'target':q['target'],'evidence':'697cab7b280d1bf94871df36601f7816446d1c753616386cc274d5e4f70861a0','scope':'posix-cpu-group-v1','no_device_work_reviewed':True,'no_escaped_children_reviewed':True};save(ROOT/'worker-route.proposed.json',route)
 template=json.loads((ROOT.parent/'daemon-check-2/coordinator-route.template.json').read_text());template['qualification']=q;plan=template['plan'];plan['cluster_id']=identities['cluster'];plan['plan_id']=str(uuid.uuid4());plan['stages'][0]['node_id']=identities['worker'];value={k:v for k,v in plan.items() if k!='plan_digest'};encoded=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False);assert encoded.isascii();plan['plan_digest']=hashlib.sha256(encoded.encode()).hexdigest();save(ROOT/'coordinator-route.template.json',template)
 save(ROOT/'preparation.json',{'kind':'NativeServiceRecoveryPreparation','implementation_commit':'419a6fdbb6e26d008d7e407ce3c4121738dbd86f','binary_sha256':hashlib.sha256(BINARY.read_bytes()).hexdigest(),'configuration':configuration,'worker_record_digest':json.loads(call(worker,'model','execution-digest',ROOT/'worker-route.proposed.json'))['configuration'],'profile_evidence':q['replica']['evidence'],'worker_file_sha256':hashlib.sha256((ROOT/'worker-route.proposed.json').read_bytes()).hexdigest(),'coordinator_template_sha256':hashlib.sha256((ROOT/'coordinator-route.template.json').read_bytes()).hexdigest(),'policy_scope':'proposed fixed-input monitored CPU experiment; verified cache path/node/cluster plus disabled diagnostic backtrace and exact pre-reviewed cleanup policy; bounded experimental route, not a general qualification','native_started':False,'approval_granted':False,'prior_quarantine_preserved':True})
 call(coordinator,'cluster','stop');process.wait(timeout=10);assert process.returncode==0
finally:
 if process.poll() is None:process.kill();process.wait(timeout=3)
 log.close()
print(json.dumps({'kind':'NativeDaemonOfflinePreparation','state':str(STATE),'configuration':configuration,'native_started':False,'approval_granted':False,'prior_quarantine_preserved':True}))
