import ast,json,pathlib,re,subprocess,time,uuid
root=pathlib.Path('work/native-check/recovery-check-2').resolve();tree=ast.parse((root/'run_daemon.py').read_text());namespace={'subprocess':subprocess,'time':time,'re':re,'BIN':root/'glyd-compute','START':time.monotonic(),'control_next':{},'control_events':[]}
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'call','ready_resource'}]
exec(compile(ast.Module(body=functions,type_ignores=[]),'bounded-runner-functions','exec'),namespace)
state=pathlib.Path('/private/tmp/glyd-control-paced-'+uuid.uuid4().hex[:8]);state.mkdir(mode=0o700);call=namespace['call'];call(state,'cluster','init','--listen','127.0.0.1:0');log=open('work/m2-recovery-paced-control.log','wb');p=subprocess.Popen([str(namespace['BIN']),'--state-dir',str(state),'daemon'],stdout=log,stderr=log)
try:
 time.sleep(.5)
 for _ in range(12):assert 'role=coordinator' in call(state,'cluster','status')
 assert not namespace['control_events']
 call(state,'cluster','stop');p.wait(timeout=5);assert p.returncode==0
 try:call(state,'cluster','status');raise AssertionError('closed endpoint accepted')
 except RuntimeError:pass
 assert namespace['control_events'][-1]['operation']=='cluster/status'
 assert namespace['control_events'][-1]['code']=='DAEMON_UNAVAILABLE'
finally:
 if p.poll() is None:p.terminate();p.wait(timeout=5)
 log.close()
# Missing resource observation is not ready; it never grants inference.
namespace.update(coordinator=state,ids={'worker':'fixture'},resource_history=[])
namespace['report']=lambda *args:None
assert namespace['ready_resource']() is None
namespace['report']=lambda *args:{'sample':{'capabilities':{'state':'ready'}}}
assert namespace['ready_resource']()
record={'kind':'ModelFreeCorrectedRecoveryRunnerCheck','model_started':False,'paced_status_calls_passed':12,'control_failures_recorded':True,'missing_resource_is_not_ready':True,'ready_resource_required':True,'spend_usd':0};pathlib.Path('outputs/M2-recovery-control-check.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
