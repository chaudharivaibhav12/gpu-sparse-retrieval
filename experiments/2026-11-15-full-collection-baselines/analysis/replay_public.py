"""Replay public lossless evidence without the private recording harness.
Usage: python experiments/2026-11-15-full-collection-baselines/analysis/replay_public.py
"""
import csv,hashlib,io,json,math,struct,tempfile,zipfile
from pathlib import Path,PurePosixPath
from independent_statistics import compute
ROOT=Path(__file__).resolve().parents[3];E='2026-11-15-full-collection-baselines';RID='20261116-cuda5-baseline-repair02'
def sha(data):return hashlib.sha256(data).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(data,delimiter=','):return csv.DictReader(io.StringIO(data.decode('utf-8')),delimiter=delimiter)
def close(a,b):
 if isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:close(a[k],b[k])
 elif isinstance(a,list):
  assert len(a)==len(b)
  for x,y in zip(a,b):close(x,y)
 elif isinstance(a,float):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-15),(a,b)
 else:assert a==b

def main():
 run=ROOT/'results'/E/RID;manifest=load(run/'manifest.json');assert manifest['status']=='complete'
 for name,item in manifest['artifacts'].items():assert sha((run/name).read_bytes())==item['sha256']
 attempts=load(run/'attempts.json');assert len(attempts)==42
 reference_run=ROOT/'results'/E/'20261115-cuda5-reference01';reference_manifest=load(reference_run/'manifest.json')
 assert sha((reference_run/'raw-evidence.zip').read_bytes())==reference_manifest['artifacts']['raw-evidence.zip']['sha256']
 with zipfile.ZipFile(reference_run/'raw-evidence.zip') as z:reference=z.read('full-top10.tsv')
 assert sha(reference)=='7a63bb5214891b0c820d7d8205a16701b473292f2f3bf6b05fc0c2494ff2b6ed'
 qdata=(ROOT/'results/calibration-500.tsv').read_bytes();assert sha(qdata)=='150917d784f009fddb453a1635bc62d3ec238c4a101347d46914adaad371d091'
 queries={}
 for line in qdata.decode().splitlines():
  f=line.split();queries[f[0]]={int(t):int(w) for t,w in (x.split(':') for x in f[1:])}
 assert len(queries)==500 and 255*max(sum(w.values()) for w in queries.values())==180540
 oracle={q:[] for q in queries}
 for row in rows(reference,'\t'):oracle[row['query_id']].append((int(row['original_id']),int(row['score'])))
 assert all(len(v)==10 for v in oracle.values());groups=scores=0;paths={k:[] for k in ['range','mono','csr']}
 with zipfile.ZipFile(run/'raw-evidence.zip') as z,tempfile.TemporaryDirectory() as td:
  assert z.testzip() is None
  for name,item in manifest['raw_members'].items():assert sha(z.read(name))==item['sha256']
  candidates=[json.loads(x) for x in z.read('analysis/candidate-rows.jsonl').decode().splitlines()];docs={r['original_id']:r['terms'] for r in candidates};positions={r['position']:r['original_id'] for r in candidates}
  for q,values in oracle.items():
   for d,score in values:assert sum(queries[q].get(t,0)*w for t,w in docs[d])==score
  # Keep registered process order from the completed execution manifest.
  completed=load(run/'execution-manifest.json')['completed'];ordered={x['tag']:i for i,x in enumerate(completed)}
  for attempt in sorted(attempts,key=lambda x:ordered[x['tag']]):
   tag=attempt['tag'];runtime=attempt['resolved_runtime'];cfg=runtime['config'];actual=runtime['behavior'];label={'cuda-range-v2':'range','gpusparse-u64-canonical':'mono','cuda-fullbound-pruning':'csr'}[cfg['mode']]
   if tag.startswith('reject-'):
    assert attempt['status']=='SPEC_VIOLATION' and len(attempt['runtime_delta'])==1;continue
   assert attempt['status']=='EXECUTED' and cfg==attempt['expected'] and actual==attempt['expected_behavior']
   full=tag.startswith(('correctness-','timing-'))
   if full:expected=oracle
   else:
    role=next(x for x in attempt['inputs'] if x['role']=='reference');remote_run=PurePosixPath(attempt['cwd'])/'results'/E/RID;member=str(PurePosixPath(role['path']).relative_to(remote_run));expected={q:[tuple(v) for v in values[:min(10,cfg['docs'])]] for q,values in json.loads(z.read(member)).items()}
   wanted=[(phase,n,q) for phase,count in [('warmup',cfg['warmup_rounds']),('measured',cfg['repeats'])] for n in range(count) for q in expected]
   actual_groups={};observed=[]
   for row in rows(z.read('raw/'+tag+'/top.tsv'),'\t'):
    key=(row['phase'],int(row['round']),row['query_id'])
    if key not in actual_groups:actual_groups[key]=[];observed.append(key)
    assert int(row['rank'])==len(actual_groups[key])+1;actual_groups[key].append((int(row['original_id']),int(row['score'])));scores+=1
    if full and label=='mono':assert positions[int(row['internal_doc_position'])]==int(row['original_id'])
   assert observed==(wanted if cfg['docs'] else []);assert all(actual_groups.get(key,[])==expected[key[2]] for key in wanted);groups+=len(wanted)
   data=z.read('raw/'+tag+'/timing.csv');assert [(r['phase'],int(r['round']),r['query_id']) for r in rows(data)]==wanted
   if tag.startswith('timing-'):
    path=Path(td)/(tag+'.csv');path.write_bytes(data);paths[label].append(path)
  result=compute(paths,list(queries));reported=load(run/'execution-manifest.json')['analysis']
  close(result['comparisons'],reported['comparisons'])
  for label,values in result['summary'].items():
   for key,value in values.items():close(value,reported['summary'][label][key])
 print(json.dumps(dict(success=True,groups=groups,scores=scores,oracle_scores_checked=5000,independent_bootstrap_draws=10000,qualification='500calibration queries; canonicalU64adapted GPUSparse, not unmodified native. Partial source rows are remotely hash-bound, not a local full-index membership proof.')))
if __name__=='__main__':main()
