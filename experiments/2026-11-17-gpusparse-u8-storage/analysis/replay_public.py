"""Standalone public replay: hashes, raw answers, source-row arithmetic and statistics.
Run from any checkout; requires only Python's standard library.
"""
import csv,hashlib,json,math,random,statistics,struct,tempfile,zipfile,shutil
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[3]
E='2026-11-17-gpusparse-u8-storage';RID='20261118-cuda5-u8-retry03'
PROTOCOL=json.loads((ROOT/'experiments'/E/'configs/protocol.json').read_text())
def replay(R):
 run=R/'results/2026-11-17-gpusparse-u8-storage/20261118-cuda5-u8-retry03';m=json.loads((run/'execution-manifest.json').read_text());saved=json.loads((run/'analysis/paired-results.json').read_text());receipt=json.loads((run/'source-receipt.json').read_text())
 def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 assert m['source_commit']==receipt['source_commit']=='f263feed61a47f9b6317fff45df9788430b0ff2c'
 protocol=PROTOCOL;qfile=R/'results/calibration-500.tsv';qids=[l.split()[0] for l in qfile.read_text().splitlines()];assert len(qids)==500
 oracle=R/'results/2026-11-15-full-collection-baselines/20261115-cuda5-reference01/full-top10.tsv';assert sha(oracle)==protocol['inputs']['oracle']['sha256'];gold={q:[] for q in qids}
 for row in csv.DictReader(oracle.open(),delimiter='\t'):gold[row['query_id']].append((int(row['original_id']),int(row['score'])))
 def local(remote):return R/Path(remote).relative_to('/scratch/vc2836/gpu-project/sessions/u8-storage-f263fee-retry03-cuda5')
 attempts={x['tag']:x for x in json.loads((run/'attempts.json').read_text())};matrices={x:{} for x in ['range','control','compact']};groups=scores=0;candidate_rows=[json.loads(line) for line in (run/'analysis/candidate-rows.jsonl').read_text().splitlines()];positions={x['position']:x['original_id'] for x in candidate_rows};docs={x['original_id']:x['terms'] for x in candidate_rows};row_proof=json.loads((run/'analysis/candidate-rows-manifest.json').read_text());assert row_proof['index_sha256']==PROTOCOL['inputs']['gsp']['sha256'] and sha(run/'analysis/candidate-rows.jsonl')==row_proof['output_sha256'];queries={f[0]:{int(t):int(w) for t,w in (x.split(':') for x in f[1:])} for f in (line.split() for line in qfile.read_text().splitlines())}
 for q,values in gold.items():
  for doc,score in values:assert sum(queries[q].get(t,0)*w for t,w in docs[doc])==score
 for e in m['completed']:
  event=dict(attempts[e['tag']]);event['runtime']=event['resolved_runtime'];assert event['status']!='EXECUTED' or (event['runtime']['config']==event['expected'] and event['runtime']['behavior']==event['expected_behavior'])
  if e.get('kind')=='engineering':assert event['status']=='SPEC_VIOLATION';continue
  assert event['status']=='EXECUTED'
  tag=e['tag'];isfull=tag.startswith(('correctness-','timing-'))
  if isfull:expected=gold
  else:
   label=tag[len('fixture-'):].rsplit('-',1)[0];f=run/'fixtures'/label;data=(f/('fixture.gsp' if label=='numeric-uint64' else 'index.gsp')).read_bytes();n,v,z=struct.unpack_from('<IIQ',data,8);ids=struct.unpack_from('<%dQ'%n,data,24);at=24+8*n;offs=struct.unpack_from('<%dQ'%(n+1),data,at);at+=8*(n+1);terms=struct.unpack_from('<%dI'%z,data,at);weights=data[at+4*z:at+5*z];expected={}
   for l in (f/('fixture.tsv' if label=='numeric-uint64' else 'queries.tsv')).read_text().splitlines():
    a=l.split();q={int(t):int(w) for t,w in (s.split(':') for s in a[1:])};expected[a[0]]=sorted([(ids[d],sum(q.get(terms[j],0)*weights[j] for j in range(offs[d],offs[d+1]))) for d in range(n)],key=lambda p:(-p[1],p[0]))[:10]
  cfg=event['runtime']['config'];seen=[];actual={}
  for row in csv.DictReader(local(e['top']).open(),delimiter='\t'):
   key=(row['phase'],int(row['round']),row['query_id']);a=actual.setdefault(key,[]);assert int(row['rank'])==len(a)+1;a.append((int(row['original_id']),int(row['score'])));scores+=1
   if isfull and e['label']!='range':assert positions[int(row['internal_doc_position'])]==int(row['original_id'])
  order=[(p,r,q) for p,count in [('warmup',cfg['warmup_rounds']),('measured',cfg['repeats'])] for r in range(count) for q in expected]
  for k in order:assert actual.get(k,[])==expected[k[2]],tag
  assert set(actual)<=set(order);groups+=len(order)
  vals={q:[] for q in expected}
  for row in csv.DictReader(local(e['csv']).open()):
   key=(row['phase'],int(row['round']),row['query_id']);seen.append(key);v=float(row['query_wall_ms']);assert math.isfinite(v) and v>0
   if row['phase']=='measured':vals[row['query_id']].append(v)
  assert seen==order
  if tag.startswith('timing-'):
   assert e['eligible'] and not e['monitor']['external_pids'] and not e['monitor']['sampling_errors'] and e['monitor']['owned_context_seen'];rep=int(tag.split('-')[1][1:]);frames=[json.loads(line) for line in (run/'logs'/(tag+'.monitor.jsonl')).read_text().splitlines()];assert frames and not any(f.get('error') or any(not p['owned'] for p in f.get('contexts',[])) for f in frames);assert max(sum(p['used_mib'] for p in f.get('contexts',[]) if p['owned']) for f in frames)==e['monitor']['sampled_own_peak_mib'];matrices[e['label']][rep]=[statistics.median(vals[q]) for q in qids]
 def quantile(x,p):
  x=sorted(x);t=(len(x)-1)*p;i=math.floor(t);j=math.ceil(t);return x[i]+(x[j]-x[i])*(t-i)
 contrasts={}
 for offset,(name,left,right) in enumerate([('compact_over_control','compact','control'),('range_over_compact','range','compact'),('range_over_control','range','control')]):
  logs=[[math.log(x/y) for x,y in zip(matrices[left][p],matrices[right][p])] for p in range(3)];rng=random.Random(2026111701+offset);draws=[]
  for _ in range(10000):
   ps=rng.choices(range(3),k=3);qs=rng.choices(range(500),k=500);draws.append(math.exp(math.fsum(logs[p][q] for p in ps for q in qs)/1500))
  ratio=math.exp(math.fsum(v for row in logs for v in row)/1500);ci=[quantile(draws,.025),quantile(draws,.975)];want=saved['comparisons'][name];assert math.isclose(ratio,want['paired_latency_ratio'],rel_tol=1e-12);assert all(math.isclose(x,y,rel_tol=1e-12) for x,y in zip(ci,want['conditional95']));contrasts[name]={'ratio':ratio,'conditional95':ci,'inverse':1/ratio}
 for label,p in matrices.items():assert set(p)=={0,1,2};assert math.isclose(statistics.median(v for row in p.values() for v in row),saved['summary'][label]['median_query_process_median_ms'],rel_tol=1e-12)
 report={'success':True,'groups_checked':groups,'ranked_scores_checked':scores,'timing_processes':9,'comparisons':contrasts};print(json.dumps(report));return report

def main():
 with tempfile.TemporaryDirectory() as td:
  base=Path(td)
  for folder in sorted((ROOT/'results'/E).iterdir()):
   if not folder.is_dir() or not (folder/'manifest.json').exists():continue
   manifest=json.loads((folder/'manifest.json').read_text())
   for name,item in manifest['artifacts'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==item['sha256'],name
   target=base/'results'/E/folder.name;target.mkdir(parents=True)
   for name in ['attempts.json','execution-manifest.json','source-receipt.json']:
    if (folder/name).exists():shutil.copyfile(folder/name,target/name)
   with zipfile.ZipFile(folder/'raw-evidence.zip') as z:
    assert z.testzip() is None
    for name,item in manifest['raw_members'].items():
     assert not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts
     raw=z.read(name);assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
     p=target/name;assert p.resolve().is_relative_to(target.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
  reference=ROOT/'results/2026-11-15-full-collection-baselines/20261115-cuda5-reference01'
  ref_target=base/'results/2026-11-15-full-collection-baselines/20261115-cuda5-reference01';ref_target.mkdir(parents=True)
  with zipfile.ZipFile(reference/'raw-evidence.zip') as z:(ref_target/'full-top10.tsv').write_bytes(z.read('full-top10.tsv'))
  shutil.copyfile(ROOT/'results/calibration-500.tsv',base/'results/calibration-500.tsv')
  return replay(base)
if __name__=='__main__':main()
