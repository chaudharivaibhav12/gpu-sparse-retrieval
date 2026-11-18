import csv,gzip,lzma,importlib.util,json,signal,sys,tempfile,types,zipfile,io
from pathlib import Path,PurePosixPath
from unittest.mock import patch
P=Path(__file__).resolve().parent;ROOT=Path(__file__).resolve().parents[3]
R=ROOT
B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B/'code'));spec=importlib.util.spec_from_file_location('matched_controller',B/'code/run_calibration.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
qids=[x.split()[0] for x in (R/'results/calibration-500.tsv').read_text().splitlines()]
gold={q:[] for q in qids}
with zipfile.ZipFile(ROOT/'results/2026-11-15-full-collection-baselines/20261115-cuda5-reference01/raw-evidence.zip') as z:
 name=next(n for n in z.namelist() if n.endswith('full-top10.tsv'))
 for row in csv.DictReader(io.StringIO(z.read(name).decode()),delimiter='\t'):gold[row['query_id']].append((int(row['original_id']),int(row['score'])))
checks=0
with tempfile.TemporaryDirectory() as d:
 d=Path(d)
 run=R/'results/2026-11-18-prune-seed-select/20261118-cuda5-ts-01/raw'
 paths={'csv':d/'timing.csv','top':d/'top.tsv'}
 paths['csv'].write_bytes(gzip.decompress((run/'timing-r0-prunets-w256-s64.csv.gz').read_bytes()))
 source=next(run.glob('timing-r0-prunets-w256-s64.tsv.*'));paths['top'].write_bytes((gzip if source.suffix=='.gz' else lzma).decompress(source.read_bytes()))
 lat,validation=m.validate_outputs('prunets256',paths,gold,qids,20,3);assert validation['groups']==11500 and len(lat)==500;checks+=1
 raw=ROOT/'results/2026-11-17-gpusparse-u8-storage/20261118-cuda5-u8-retry03';manifest=json.loads((raw/'execution-manifest.json').read_text())
 for label in ['range','compact']:
  entry=next(x for x in manifest['completed'] if x.get('label')==label and x['tag'].startswith('timing-'))
  paths={}
  with zipfile.ZipFile(raw/'raw-evidence.zip') as z:
   for key in ['csv','top']:
    relative=PurePosixPath(entry[key]).relative_to('/scratch/vc2836/gpu-project/sessions/u8-storage-f263fee-retry03-cuda5/results/2026-11-17-gpusparse-u8-storage/20261118-cuda5-u8-retry03');paths[key]=d/(label+'-'+key);paths[key].write_bytes(z.read(str(relative)))
  lat,validation=m.validate_outputs(label,paths,gold,qids,20,3);assert validation['groups']==11500 and len(lat)==500;checks+=1
 run=R/'results/2026-11-18-bmp-full-collection/20261118-cuda5-bmp-02/raw/timing-r0-bmp-b32.bmp'
 top=d/'bmp-results.tsv';top.write_bytes(gzip.decompress((run/'results.tsv.gz').read_bytes()))
 paths=dict(csv=run/'samples.tsv',top=top,cpu_monitor=run/'monitoring.tsv')
 lat,validation=m.validate_outputs('bmp',paths,gold,qids,20,3);assert len(lat)==500 and validation['iowait_share']<=.01;checks+=1
 # A duplicated rank or fractional score must be rejected, even with plausible timing.
 original=top.read_text();lines=original.splitlines(keepends=True)
 for kind in ['rank','score']:
  fields=lines[1].rstrip('\n').split('\t');fields[5 if kind=='rank' else 6]='2' if kind=='rank' else '1.5'
  top.write_text(lines[0]+'\t'.join(fields)+'\n'+''.join(lines[2:]))
  try:m.validate_outputs('bmp',paths,gold,qids,20,3)
  except AssertionError:checks+=1
  else:raise AssertionError(kind+' drift accepted')
 class Stop:
  def __init__(self):self.done=False
  def is_set(self):return self.done
  def wait(self,_):self.done=True
 class Process:
  def __init__(self,value):self.value=value
  def poll(self):return self.value
 for owned,swapped,alive,exited in [(False,False,True,False),(True,False,True,False),(False,False,True,True),(False,True,True,False),(False,True,False,False)]:
  result={};kills=[]
  def group(_):
   if exited:raise ProcessLookupError()
   return 123 if owned else 999
  with patch.object(m,'gpu_state',return_value={'free_mib':10000,'contexts':[(777,7)]}),patch.object(m,'own_ticks',return_value=(0,2*1024**2 if swapped else 0)),patch.object(m,'busy_ticks',return_value=0),patch.object(m,'cpu0_ticks',return_value=0),patch.object(m.os,'sysconf',create=True,return_value=100),patch.object(m.os,'getpgid',create=True,side_effect=group),patch.object(m.os,'killpg',create=True,side_effect=lambda g,s:kills.append((g,s))):
   m.monitor(Stop(),d/('monitor-'+str(checks)+'.jsonl'),123,result,Process(None if alive else 0))
  frame=json.loads((d/('monitor-'+str(checks)+'.jsonl')).read_text());assert frame['run_process_group']==123 and frame['context_ownership'][0]['owned']==owned and frame['context_ownership'][0]['pgid']==(None if exited else 123 if owned else 999)
  assert result['foreign_pids']==([] if owned else [777]) and result['own_gpu_context_seen']==owned
  assert kills==([(123,signal.SIGTERM)] if swapped and alive else []) and all(g!=777 for g,_ in kills);checks+=1
 def command(argv,**_):
  return types.SimpleNamespace(stdout=m.TARGET+', 11000, 2505, P0, 30, 0, 45\n' if any(x.startswith('--query-gpu=') for x in argv) else '',returncode=0)
 with patch.object(m.subprocess,'run',side_effect=command):assert m.gpu_state()==dict(free_mib=11000,contexts=[],telemetry=dict(sm_clock_mhz='2505',pstate='P0',power_w='30',utilization_percent='0',temperature_c='45'));checks+=1
 calls=[]
 def transient(label,tag,*_):
  calls.append(tag)
  if len(calls)<3:raise m.ResourceIneligible('sampled contention')
  return 'accepted'
 assert m.resource_retries(transient,'range','timing-p0-range')=='accepted' and calls==['timing-p0-range','timing-p0-range-resource-retry1','timing-p0-range-resource-retry2'];checks+=1
 calls=[]
 def broken(*_):calls.append(1);raise ValueError('correctness failure')
 try:m.resource_retries(broken,'range','bad')
 except ValueError:assert len(calls)==1;checks+=1
 else:raise AssertionError('Technical failure was silently retried')
 calls=[]
 def occupied(*_):calls.append(1);raise m.ResourceIneligible('busy')
 try:m.resource_retries(occupied,'range','busy')
 except m.ResourceIneligible:assert len(calls)==3;checks+=1
 else:raise AssertionError('Resource retry cap ignored')
print(json.dumps({'status':'PASS','checks':checks,'accepted_full_raw_table_replays':4,'invalid_output_rejections':2,'monitor_safety_cases':5,'idle_state_case':1}))
