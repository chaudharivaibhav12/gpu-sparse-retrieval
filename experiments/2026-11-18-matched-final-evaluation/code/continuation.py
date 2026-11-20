"""Verify and bind preserved heldout stages; no performance-based selection."""
import csv,decimal,hashlib,importlib.util,itertools,json,math,re
from pathlib import Path,PurePosixPath
ORDER=['prunets256','prunets1024','range','compact','bmp']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def event(root,identity):
 e=read(Path(root)/'research/events'/(identity+'.json'));h=e.pop('sha256');assert hashlib.sha256(json.dumps(e,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()==h;return e

def schedule():return [(b,s) for b in range(3) for s in ORDER[b:]+ORDER[:b]]
def slots(entries):
 result={}
 for e in entries:
  if not e['tag'].startswith('timing-'):continue
  m=re.fullmatch(r'timing-p([012])-(prunets256|prunets1024|range|compact|bmp)(?:-resource-retry[12])?',e['tag']);assert m and m[2]==e['label'];key=(int(m[1]),e['label']);assert key not in result,'Duplicate accepted slot';result[key]=e
 assert list(result)==schedule()[:len(result)],'Accepted work must be a nonselected prefix of fixed schedule'
 return result

def stream_outputs(directory,label,gold,qids,repeats,warmups,verify):
 directory=Path(directory);timing=directory/('samples.tsv' if label=='bmp' else 'timing.csv');top=directory/('results.tsv' if label=='bmp' else 'top.tsv');expected=lambda:((phase,str(r),q) for phase,n in [('warmup',warmups),('measured',repeats)] for r in range(n) for q in qids);sentinel=object();groups=0
 with timing.open() as f:
  for wanted,row in itertools.zip_longest(expected(),csv.DictReader(f,delimiter='\t' if label=='bmp' else ','),fillvalue=sentinel):
   assert wanted is not sentinel and row is not sentinel and wanted==(row['phase'],row['round'],row['query_id']);v=int(row['elapsed_ns'])/1e6 if label=='bmp' else float(row['query_wall_ms']);assert math.isfinite(v) and v>0;groups+=1
   for key in ['ordered_id_mismatches','result_score_mismatches','schedule_mismatches']:
    if key in row:assert row[key]==('0' if verify else 'not_checked')
 with top.open() as f:
  wanted=((phase,r,q,rank,answer) for phase,r,q in expected() for rank,answer in enumerate(gold[q],1))
  for item,row in itertools.zip_longest(wanted,csv.DictReader(f,delimiter='\t'),fillvalue=sentinel):
   assert item is not sentinel and row is not sentinel;phase,r,q,rank,answer=item;score=decimal.Decimal(row['score']);assert score.is_finite() and score==score.to_integral_value() and 0<=score<2**24;assert (row['phase'],row['round'],row['query_id'],int(row['rank']))==(phase,r,q,rank);assert (int(row['document_id'] if label=='bmp' else row['original_id']),int(score))==answer
 return groups*10

def verify_origin(root,output,archive_sha,monitor_module):
 root=Path(root);output=Path(output);launch=read(root/'evaluation-launch.json');live_root=launch['root'];remote=lambda relative:str(PurePosixPath(live_root)/PurePosixPath(*relative.parts));live_output=remote(output.relative_to(root));inputs=read(output/'heldout-inputs.json');assert inputs['method_freeze']=='FREEZE-20261119-MATCHED-HELDOUT-METHOD03';assert read(output/'status.json')['state']=='failed';bindings={}
 def bind(p):
  p=Path(p);assert p.is_relative_to(root);bindings[remote(p.relative_to(root))]=sha(p)
 receipt=read(root/'deployment-receipt.json')
 for n,h in {**receipt['files'],**receipt['support_files']}.items():assert sha(root/n)==h;bind(root/n)
 query=Path(inputs['queries']['path']);local_query=root/query.relative_to(live_root);local_gold=root/Path(inputs['oracle']['path']).relative_to(live_root);assert sha(local_query)==inputs['queries']['sha256'] and sha(local_gold)==inputs['oracle']['sha256'];bind(local_query);bind(local_gold);qids=[l.split()[0] for l in local_query.read_text().splitlines()];assert len(qids)==len(set(qids))==6480;gold={q:[] for q in qids}
 for row in csv.DictReader(local_gold.open(),delimiter='\t'):gold[row['query_id']].append((int(row['original_id']),int(row['score'])))
 assert all(len(x)==10 for x in gold.values());accepted=[];attempts=[];scores=0
 sp=importlib.util.spec_from_file_location('monitor_replay',monitor_module);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 for directory in sorted((output/'raw').iterdir()):
  if not (directory/'runner.stdout').exists():continue
  tag=directory.name;path=output/'analysis'/(tag+'.json')
  if not path.exists():
   attempts.append(dict(tag=tag,label='bmp',valid=False,timing_eligible=False,interrupted_unknown=True,raw_directory=str(PurePosixPath(live_output)/'raw'/tag),origin_root=live_root));continue
  entry=read(path);bind(path);assert entry['tag']==tag;entry=dict(entry,raw_directory=str(PurePosixPath(live_output)/'raw'/tag),origin_root=live_root,inherited=True);attempts.append(entry)
  if not entry['valid'] or tag.startswith('timing-') and not entry['timing_eligible']:continue
  e=event(root,entry['run']);p=e['payload'];bind(root/'research/events'/(entry['run']+'.json'));plan=event(root,p['experiment'])['payload'];bind(root/'research/events'/(p['experiment']+'.json'));assert p['runtime']==dict(config=plan['expected'],behavior=plan['behavior']) or tag.startswith('reject-')
  for x in p['artifacts']:
   blob=root/x['blob'];assert sha(blob)==x['sha256']
  for name in ['runner.stdout','runner.stderr','monitor-summary.json','monitor.jsonl','process-identity.json']:
   if (directory/name).exists():bind(directory/name)
  if tag.startswith('reject-'):
   assert p['status']=='SPEC_VIOLATION' and set(p['runtime_delta'])=={'config.seed_ranges'} and not (directory/'top.tsv').exists();accepted.append(entry);continue
  assert p['status']=='EXECUTED' and p['exit_code']==0 and not p['runtime_delta'];cfg=p['runtime']['config'];assert cfg['docs']==8841823 and cfg['query_count']==6480;timing=tag.startswith('timing-');scores+=stream_outputs(directory,entry['label'],gold,qids,20 if timing else 1,3 if timing else 0,0 if timing else 1)
  for name in ['timing.csv','top.tsv','samples.tsv','results.tsv','monitoring.tsv','physical-arithmetic.json','jit-manifest.json','native-validation.json']:
   f=directory/name
   if f.exists():
    bind(f)
    if name not in ['physical-arithmetic.json']:
     item=next(x for x in p['artifacts'] if x['name']==name);assert sha(f)==item['sha256']
  if timing:
   mon=m.replay_monitor(directory,entry['label']);entry['process_block']=int(re.match(r'timing-p([012])-',tag)[1])
   if entry['label']=='bmp':assert entry['validation']['iowait_share']<=.01
  accepted.append(entry)
 # Canonical order independent of filesystem order, and no selecting by observed time.
 negatives=[e for e in accepted if e['tag'].startswith('reject-')];correct={e['label']:e for e in accepted if e['tag'].startswith('correctness-')};timings={(e['process_block'],e['label']):e for e in accepted if e['tag'].startswith('timing-')};assert len(negatives)==1 and set(correct)==set(ORDER) and set(timings)==set(schedule()[:8]);accepted=negatives+[correct[s] for s in ORDER]+[timings[k] for k in schedule()[:8]];slots(accepted)
 return dict(version=1,origin_root=live_root,origin_output=live_output,parent_archive_sha256=archive_sha,source_commit=inputs['source_commit'],backend_source_commit=inputs['backend_source_commit'],method_freeze=inputs['method_freeze'],queries_sha256=inputs['queries']['sha256'],oracle_sha256=inputs['oracle']['sha256'],completed=accepted,attempts=attempts,bound_files=bindings,remaining=[list(x) for x in schedule()[8:]],validation={'ranked_scores_checked':scores,'accepted_correctness':5,'accepted_timings':8,'deliberate_rejections':1},limitations=['Interrupted study spans sessions; preserve assigned paired blocks,disclose temporal separation;conditional intervals only','No speed estimates used to choose carried-forward stages','Preserve excluded and interrupted attempts; no invented run_finished'])

def load_manifest(path,digest):
 assert sha(path)==digest;v=read(path);assert v['version']==1 and len(v['completed'])==14 and len(slots(v['completed']))==8 and v['remaining']==[list(x) for x in schedule()[8:]]
 negatives=[e for e in v['completed'] if e['tag'].startswith('reject-')];correct=[e for e in v['completed'] if e['tag'].startswith('correctness-')];assert len(negatives)==1 and len(correct)==5 and {e['label'] for e in correct}==set(ORDER) and all(e['valid'] for e in v['completed'])
 assert all(e['timing_eligible'] for e in v['completed'] if e['tag'].startswith('timing-'))
 root=Path(v['origin_root'])
 for name,h in v['bound_files'].items():assert Path(name).is_relative_to(root) and sha(name)==h,name
 return v
