"""Continuation bookkeeping/guard tests, no query execution or heldout tuning."""
import copy,hashlib,json,sys,tempfile
from pathlib import Path
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'code'))
if len(sys.argv)>1:sys.path.append(sys.argv[1])
import run_heldout as m
import continuation as c
checks=0
negative={'tag':'reject-prunets-full-state','label':'prunets256','valid':True};correct=[{'tag':'correctness-'+s,'label':s,'valid':True} for s in c.ORDER];timings=[{'tag':'timing-p%d-%s'%x,'label':x[1],'valid':True,'timing_eligible':True} for x in c.schedule()[:8]];entries=[negative]+correct+timings
assert len(c.slots(entries))==8 and c.schedule()[8:]==[(1,'bmp'),(1,'prunets256'),(2,'range'),(2,'compact'),(2,'bmp'),(2,'prunets256'),(2,'prunets1024')];checks+=1
for bad in [entries+[timings[0]],entries[:-1]+[{'tag':'timing-p1-bmp','label':'bmp'}],entries[:6]+list(reversed(timings))]:
 try:c.slots(bad)
 except AssertionError:checks+=1
 else:raise AssertionError('Invalid prefix accepted')
with tempfile.TemporaryDirectory() as d:
 d=Path(d);p=d/'runner.stdout'
 for text in ['', 'notjson', '{}', '[]']:
  p.write_text(text)
  try:m.runner_result(p)
  except RuntimeError:checks+=1
  else:raise AssertionError('Bad runner output accepted')
 p.write_text(json.dumps({'id':'RUN_FINISHED-fixture','payload':{'status':'EXECUTED'}}));assert m.runner_result(p)['id']=='RUN_FINISHED-fixture';checks+=1
 bound=d/'bound';bound.write_text('original');v={'version':1,'completed':entries,'remaining':[list(x) for x in c.schedule()[8:]],'origin_root':str(d),'bound_files':{str(bound):c.sha(bound)}};p=d/'continuation.json';p.write_text(json.dumps(v));assert c.load_manifest(p,c.sha(p))==v;checks+=1
 bound.write_text('changed')
 try:c.load_manifest(p,c.sha(p))
 except AssertionError:checks+=1
 else:raise AssertionError('Mutated inherited evidence accepted')
 try:c.load_manifest(p,'0'*64)
 except AssertionError:checks+=1
 else:raise AssertionError('Wrong manifest SHA accepted')
 v['bound_files']={};v['completed']=copy.deepcopy(entries);v['completed'][1]['label']=v['completed'][2]['label'];p.write_text(json.dumps(v))
 try:c.load_manifest(p,c.sha(p))
 except AssertionError:checks+=1
 else:raise AssertionError('Missing correctness control accepted')
print(json.dumps({'status':'PASS','checks':checks,'scope':'prefix/duplicates/holes/order,missing malformed runner JSON,manifest/evidence drift,correctness coverage;noqueryexecution'}))
