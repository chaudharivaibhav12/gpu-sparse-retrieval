"""Independent canonical-output, sampling and full-buffer-accounting validation."""
import csv,json,math
from pathlib import Path
from expected import allocation,QUERY_ARGUMENT_BYTES

def reference(path,k):
 path=Path(path)
 if path.suffix=='.json':return {q:[tuple(map(int,x)) for x in values[:k]] for q,values in json.loads(path.read_text(encoding='utf-8')).items()}
 out={}
 with path.open(encoding='utf-8',newline='') as f:
  for row in csv.DictReader(f,delimiter='\t'):out.setdefault(row['query_id'],[]).append((int(row['original_id']),int(row['score'])))
 return {q:values[:k] for q,values in out.items()}

def validate(csv_path,top_path,oracle,cfg,meta,query_rows):
 target=reference(oracle,min(cfg['k'],cfg['docs']));qids=[q for q,t in query_rows];lengths={q:len(t) for q,t in query_rows};assert set(qids)<=set(target)
 assert len(query_rows)==cfg['query_count'] and len(set(qids))==len(qids)
 assert all(len(target[q])==min(cfg['k'],cfg['docs']) for q in qids)
 wanted=[(phase,str(r),q) for phase,n in [('warmup',cfg['warmup_rounds']),('measured',cfg['repeats'])] for r in range(n) for q in qids]
 with Path(csv_path).open(newline='') as f:rows=list(csv.DictReader(f))
 assert [(r['phase'],r['round'],r['query_id']) for r in rows]==wanted
 checks=dict(docs=cfg['docs'],k=cfg['k'],range_size=cfg['range_size'],ranges=cfg['ranges'],score_threads=cfg['score_threads'],merge_threads=cfg['merge_threads'],device_allocation_bytes=allocation(meta,cfg['k'],cfg['score_threads']),index_posting_bytes=cfg['index_posting_bytes'],result_transfer_bytes=min(cfg['k'],cfg['docs'])*24,query_transfer_bytes=QUERY_ARGUMENT_BYTES)
 for row in rows:
  assert all(int(row[key])==value for key,value in checks.items()),row
  assert int(row['query_terms'])==lengths[row['query_id']]
  assert row['ordered_id_mismatches']==row['result_score_mismatches']==('0' if cfg['verify_scores'] else 'not_checked')
  assert math.isfinite(float(row['query_wall_ms'])) and float(row['query_wall_ms'])>0
  assert math.isfinite(float(row['gpu_span_ms'])) and float(row['gpu_span_ms'])>=0
 actual={};returned=0
 with Path(top_path).open(newline='') as f:
  for row in csv.DictReader(f,delimiter='\t'):
   key=(row['phase'],row['round'],row['query_id']);actual.setdefault(key,[]).append((int(row['rank']),int(row['original_id']),int(row['score'])));returned+=1
 for key in wanted:assert actual.get(key,[])==[(i,d,s) for i,(d,s) in enumerate(target[key[2]],1)],key
 assert not set(actual)-set(wanted)
 return dict(success=True,groups=len(wanted),returned_scores=returned,canonical_mismatches=0,device_allocation_bytes=checks['device_allocation_bytes'],query_transfer_bytes=QUERY_ARGUMENT_BYTES,qualification='Canonical ordered IDs/scores match independent fixture/CPU oracle; explicit application buffer formulas checked. CPU verification state kept separate from post-run oracle replay; not physical board peak.')
